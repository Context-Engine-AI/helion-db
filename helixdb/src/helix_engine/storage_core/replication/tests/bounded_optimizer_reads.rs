//! The optimizer's per-turn probes on LSM (is a segment empty, how many rows
//! are unindexed) answer from bounded reads. These pin that the answers, and
//! the sealing decisions made from them, are the ones the full row counts gave.

use super::*;
use crate::helix_engine::vector_core::named_vectors::DenseVectorSegmentRole;

fn hnsw() -> HNSWConfig {
    HNSWConfig::new(Some(8), Some(32), Some(64))
}

/// In-memory LSM collection `name` with one 4-d dense space. Threshold 0 keeps
/// upserts from queueing index jobs, so a test owns every seal, build and job.
fn lsm_collection(
    dir: &TempDir,
    name: &str,
) -> (Arc<CollectionManager>, Config, Arc<HelixGraphStorage>) {
    let mut config = Config::new(8, 32, 64, 1).with_lsm_in_memory();
    config.vector_config.flat_scan_threshold = Some(0);
    let collections =
        Arc::new(CollectionManager::new(dir.path().join("data"), config.clone()).unwrap());
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
    assert_eq!(storage.backend.kind(), BackendKind::Lsm);
    (collections, config, storage)
}

fn upsert(
    collections: &Arc<CollectionManager>,
    config: &Config,
    name: &str,
    ids: std::ops::RangeInclusive<u128>,
) {
    apply_mutation(
        collections,
        config,
        &ReplicatedMutation::UpsertPoints {
            collection: name.into(),
            points: ids
                .map(|id| ReplicatedPoint {
                    id,
                    vectors: HashMap::from([("dense".into(), vec![id as f32, 1.0, 0.0, 0.0])]),
                    sparse_vectors: HashMap::new(),
                    payload: HashMap::new(),
                })
                .collect(),
        },
    )
    .unwrap();
}

/// Seals the mutable tail and publishes its index, as an optimizer turn does.
fn index_tail(storage: &Arc<HelixGraphStorage>) {
    seal_lsm_dense_tail(storage, "dense", &hnsw()).unwrap();
    let prepared = {
        let read = storage.backend.begin_read().unwrap();
        storage
            .named_vectors
            .prepare_building_indices_limited_be(&read, "dense", 1)
            .unwrap()
    };
    assert_eq!(prepared.len(), 1);
    run_chunked_indices_publish(storage, "dense", prepared).unwrap();
}

fn run_job(storage: &Arc<HelixGraphStorage>, name: &str, flat_scan_threshold: usize) {
    run_index_job(IndexJob {
        collection: name.into(),
        storage: Arc::downgrade(storage),
        spaces_to_check: vec!["dense".into()],
        hnsw_config: hnsw(),
        flat_scan_threshold,
    });
    // A turn that ran out of budget queues a follow-up.
    wait_for_index_executor(name);
}

/// `(all rows, rows in segments without an index)` from the full row count of
/// every segment: what the optimizer compared before the bounded count.
fn full_counts(storage: &HelixGraphStorage) -> (u64, u64) {
    let read = storage.backend.begin_read().unwrap();
    let stats = storage
        .named_vectors
        .get_dense_space_stats_be(&read, "dense")
        .unwrap();
    (
        stats.vectors_count,
        stats.vectors_count - stats.indexed_vectors_count,
    )
}

#[test]
fn bounded_unindexed_count_answers_the_optimizer_tests_without_reading_indexed_rows() {
    let _guard = ENV_LOCK.lock().unwrap();
    let dir = TempDir::new().unwrap();
    let name = "bounded_count";
    let (collections, config, storage) = lsm_collection(&dir, name);
    let AnyBackend::Lsm(backend) = &*storage.backend else {
        panic!("fixture must run on the LSM writer backend");
    };
    // Given three indexed segments of 16 rows, first without and then with a
    // 5-row unindexed tail.
    for first in [1u128, 17, 33] {
        upsert(&collections, &config, name, first..=first + 15);
        index_tail(&storage);
    }
    for tail in [0u64, 5] {
        if tail > 0 {
            upsert(&collections, &config, name, 49..=48 + u128::from(tail));
        }
        let read = storage.backend.begin_read().unwrap();
        backend.reset_read_count();
        let stats = storage
            .named_vectors
            .get_dense_space_stats_be(&read, "dense")
            .unwrap();
        let full_reads = backend.read_count();
        let unindexed = stats.vectors_count - stats.indexed_vectors_count;
        assert_eq!((stats.indexed_vectors_count, unindexed), (48, tail));

        // When the tail is counted up to each limit the optimizer passes: 1 for
        // "any tail row", max(threshold, 1) for "at least threshold".
        for limit in [1u64, 2, 4, 5, 6, 48, 1_000] {
            backend.reset_read_count();
            let bounded = storage
                .named_vectors
                .mutable_tail_count_upto_be(&read, "dense", limit)
                .unwrap();
            let bounded_reads = backend.read_count();

            // Then both of its comparisons come out as on the full count.
            assert_eq!(bounded, unindexed.min(limit), "limit {limit}");
            assert_eq!(bounded >= limit, unindexed >= limit, "limit {limit}");
            assert_eq!(bounded > 0, unindexed > 0, "limit {limit}");
            // And no indexed segment's rows were read to get them.
            assert!(
                bounded_reads + 48 <= full_reads,
                "limit {limit}: {bounded_reads} reads bounded, {full_reads} for the full count"
            );
        }
    }
}

#[test]
fn empty_segment_locate_names_the_same_segments_from_one_row_each() {
    let _guard = ENV_LOCK.lock().unwrap();
    // Hard deletes: the rows leave the segment instead of being tombstoned.
    let _hard_deletes = EnvVarGuard::unset("HELIX_LSM_DELETE_TOMBSTONES");
    let dir = TempDir::new().unwrap();
    let name = "empty_locate";
    let (collections, config, storage) = lsm_collection(&dir, name);
    let AnyBackend::Lsm(backend) = &*storage.backend else {
        panic!("fixture must run on the LSM writer backend");
    };
    // Given three indexed segments of 16 rows and a tail, with every row of
    // the middle segment deleted.
    for first in [1u128, 17, 33] {
        upsert(&collections, &config, name, first..=first + 15);
        index_tail(&storage);
    }
    upsert(&collections, &config, name, 49..=53);
    apply_mutation(
        &collections,
        &config,
        &ReplicatedMutation::DeletePoints {
            collection: name.into(),
            ids: (17..=32).collect(),
        },
    )
    .unwrap();

    // When the sealed segments are counted in full, as the locate used to.
    let read = storage.backend.begin_read().unwrap();
    let spaces = storage.named_vectors.list_dense_vector_spaces();
    backend.reset_read_count();
    let cores = storage.named_vectors.cores_read().unwrap();
    let counted: Vec<(String, u64)> = spaces["dense"]
        .segments
        .iter()
        .filter(|segment| segment.role != DenseVectorSegmentRole::Mutable)
        .map(|segment| {
            let rows = cores[&segment.physical_name]
                .level_zero_count(&read)
                .unwrap();
            (segment.physical_name.clone(), rows)
        })
        .collect();
    drop(cores);
    let full_reads = backend.read_count();
    let empty: Vec<String> = counted
        .iter()
        .filter(|(_, rows)| *rows == 0)
        .map(|(segment, _)| segment.clone())
        .collect();
    assert_eq!(empty.len(), 1, "{counted:?}");
    assert_eq!(
        counted.iter().filter(|(_, rows)| *rows == 16).count(),
        2,
        "{counted:?}"
    );

    // And located by the optimizer.
    backend.reset_read_count();
    let located = storage
        .named_vectors
        .cleanup_empty_dense_segments_locate_be(&read, "dense")
        .unwrap();
    let reads = backend.read_count();

    // Then it names the same segments, having stopped at the first row of
    // each non-empty one instead of reading all 16.
    assert_eq!(located, empty);
    assert!(
        reads + 30 <= full_reads,
        "{reads} reads to locate, {full_reads} to count every row"
    );
}

#[test]
fn lsm_index_job_seals_the_tail_in_the_cases_the_full_count_did() {
    let _guard = ENV_LOCK.lock().unwrap();
    let dir = TempDir::new().unwrap();
    let name = "bounded_seal";
    let (collections, config, storage) = lsm_collection(&dir, name);
    upsert(&collections, &config, name, 1..=8);
    index_tail(&storage);
    assert_eq!(full_counts(&storage), (8, 0));

    {
        // Without quiesced-tail sealing only the threshold seals a tail.
        let _threshold_only = EnvVarGuard::set("HELIX_INDEX_SUBTHRESHOLD_QUIESCED_TAILS", "0");
        upsert(&collections, &config, name, 9..=11);
        run_job(&storage, name, 4);
        assert_eq!(
            full_counts(&storage),
            (11, 3),
            "a tail below the threshold stays flat"
        );
        run_job(&storage, name, 3);
        assert_eq!(
            full_counts(&storage),
            (11, 0),
            "a tail at the threshold is sealed and built"
        );
        upsert(&collections, &config, name, 12..=13);
        run_job(&storage, name, 1_000);
        assert_eq!(full_counts(&storage), (13, 2));
    }

    // With it (the default), any unindexed row seals once writes went quiet.
    let _quiesced_tails = EnvVarGuard::set("HELIX_INDEX_SUBTHRESHOLD_QUIESCED_TAILS", "1");
    run_job(&storage, name, 1_000);
    assert_eq!(
        full_counts(&storage),
        (13, 0),
        "a quiesced tail below the threshold is sealed and built"
    );
}

/// Backend reads (rows scanned plus point reads) of one optimizer turn over
/// `indexed` indexed rows and a 3-row tail it has no reason to seal.
fn idle_turn_reads(indexed: u128) -> usize {
    let dir = TempDir::new().unwrap();
    let name = format!("idle_turn_{indexed}");
    let (collections, config, storage) = lsm_collection(&dir, &name);
    let AnyBackend::Lsm(backend) = &*storage.backend else {
        panic!("fixture must run on the LSM writer backend");
    };
    upsert(&collections, &config, &name, 1..=indexed);
    index_tail(&storage);
    upsert(&collections, &config, &name, indexed + 1..=indexed + 3);
    // The first turn absorbs one-time work; the second is the steady state.
    run_job(&storage, &name, 1_000);
    backend.reset_read_count();
    run_job(&storage, &name, 1_000);
    let reads = backend.read_count();
    assert_eq!(full_counts(&storage), (indexed as u64 + 3, 3));
    reads
}

#[test]
fn idle_lsm_optimizer_turn_does_not_read_every_indexed_row() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _threshold_only = EnvVarGuard::set("HELIX_INDEX_SUBTHRESHOLD_QUIESCED_TAILS", "0");
    let small = idle_turn_reads(32);
    let large = idle_turn_reads(160);
    // Each of the turn's probes used to scan every row, so 128 more indexed
    // rows cost several times 128 more reads.
    assert!(
        large < small + 128,
        "an idle optimizer turn read {small} rows over 32 indexed rows and {large} over 160"
    );
}
