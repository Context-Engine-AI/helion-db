use super::*;
use crate::helix_engine::vector_core::named_vectors::DenseVectorSegmentRole;

#[test]
fn optimizer_retires_fully_superseded_lsm_merge_sources() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _quiesced = EnvVarGuard::set("HELIX_INDEX_SUBTHRESHOLD_QUIESCED_TAILS", "0");
    let _delete_tombstones = EnvVarGuard::set("HELIX_LSM_DELETE_TOMBSTONES", "1");
    for deleted in [false, true] {
        // Given indexed rows that are all superseded, plus an unrelated live tail.
        let dir = TempDir::new().unwrap();
        let mut config = Config::new(8, 32, 64, 1).with_lsm_in_memory();
        config.vector_config.flat_scan_threshold = Some(0);
        let collections =
            Arc::new(CollectionManager::new(dir.path().join("data"), config.clone()).unwrap());
        let name = "empty_merge";
        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::CreateCollection {
                name: name.into(),
                vectors: HashMap::from([(
                    "dense".into(),
                    NamedVectorConfig {
                        size: 4,
                        distance: DistanceMetric::Cosine,
                        spindle: Default::default(),
                    },
                )]),
                sparse_vectors: HashMap::new(),
                hnsw_overrides: None,
            },
        )
        .unwrap();
        let storage = collections.get_collection(name).unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let point = |id, vector| ReplicatedPoint {
            id,
            vectors: HashMap::from([("dense".into(), vector)]),
            sparse_vectors: HashMap::new(),
            payload: HashMap::new(),
        };
        let upsert = |points| {
            apply_mutation(
                &collections,
                &config,
                &ReplicatedMutation::UpsertPoints {
                    collection: name.into(),
                    points,
                },
            )
            .unwrap();
        };
        let source_count = NamedVectorManager::max_indexed_segments_target() + 1;
        let last_id = u128::try_from(source_count).unwrap();
        let survivor_id = last_id.checked_add(1).unwrap();
        for id in 1..=last_id {
            upsert(vec![point(id, vec![1.0, 0.0, 0.0, 0.0])]);
            seal_lsm_dense_tail(&storage, "dense", &hnsw).unwrap();
            let prepared = {
                let read = storage.backend.begin_read().unwrap();
                storage
                    .named_vectors
                    .prepare_building_indices_limited_be(&read, "dense", 1)
                    .unwrap()
            };
            assert_eq!(prepared.len(), 1);
            run_chunked_indices_publish(&storage, "dense", prepared).unwrap();
        }
        let sources: Vec<_> = storage.named_vectors.list_dense_vector_spaces()["dense"]
            .segments
            .iter()
            .filter(|segment| segment.role == DenseVectorSegmentRole::Indexed)
            .map(|segment| segment.physical_name.clone())
            .collect();
        assert_eq!(sources.len(), source_count);
        if deleted {
            apply_mutation(
                &collections,
                &config,
                &ReplicatedMutation::DeletePoints {
                    collection: name.into(),
                    ids: (1..=last_id).collect(),
                },
            )
            .unwrap();
        } else {
            upsert(
                (1..=last_id)
                    .map(|id| point(id, vec![0.0, 1.0, 0.0, 0.0]))
                    .collect(),
            );
        }
        upsert(vec![point(survivor_id, vec![0.0, 0.0, 1.0, 0.0])]);

        // When the production optimizer merges the now-empty indexed sources.
        run_index_job(IndexJob {
            collection: name.into(),
            storage: Arc::downgrade(&storage),
            spaces_to_check: vec!["dense".into()],
            hnsw_config: hnsw,
            flat_scan_threshold: 1_000,
        });

        // Then a merge retires obsolete sources and current tail rows stay intact.
        let spaces = storage.named_vectors.list_dense_vector_spaces();
        assert!(
            spaces["dense"]
                .segments
                .iter()
                .filter(|segment| sources.contains(&segment.physical_name))
                .count()
                < source_count,
            "empty merge sources were retained: deleted={deleted}, {:?}",
            spaces["dense"].segments
        );
        let read = storage.backend.begin_read().unwrap();
        let cores = storage.named_vectors.cores_read().unwrap();
        let live: HashMap<_, _> = spaces["dense"]
            .segments
            .iter()
            .filter(|segment| segment.role == DenseVectorSegmentRole::Mutable)
            .flat_map(|segment| {
                cores[&segment.physical_name]
                    .export_level_zero(&read)
                    .unwrap()
            })
            .map(|(id, vector, _)| (id, vector))
            .collect();
        let mut expected = HashMap::from([(survivor_id, vec![0.0, 0.0, 1.0, 0.0])]);
        if !deleted {
            expected.extend((1..=last_id).map(|id| (id, vec![0.0, 1.0, 0.0, 0.0])));
        }
        assert_eq!(live, expected);
    }
}
