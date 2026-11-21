use super::*;
use crate::helix_engine::vector_core::named_vectors::{
    DenseVectorSegmentMetadata, DenseVectorSegmentRole, DenseVectorSpaceMetadata,
};

#[test]
fn coreless_building_segment_does_not_queue_an_unbuildable_job() {
    let _guard = ENV_LOCK.lock().unwrap();
    let dir = TempDir::new().unwrap();
    let config = Config::new(8, 32, 64, 1).with_lsm_in_memory();
    let collections = CollectionManager::new(dir.path().to_path_buf(), config.clone()).unwrap();
    let storage = collections.create_collection("coreless_building").unwrap();
    let metadata = HashMap::from([(
        "dense".to_string(),
        DenseVectorSpaceMetadata {
            next_segment_id: 1,
            segments: vec![DenseVectorSegmentMetadata {
                physical_name: "missing_core".into(),
                role: DenseVectorSegmentRole::Building,
            }],
        },
    )]);
    storage
        .named_vectors
        .refresh_dense_view_from_metadata_lsm(
            dir.path(),
            &HashMap::new(),
            &metadata,
            HNSWConfig::new(Some(8), Some(32), Some(64)),
            None,
        )
        .unwrap();

    let debt = storage.named_vectors.dense_segment_debt();
    let submit = optimizer_debt_has_submit_work(
        debt,
        NamedVectorManager::max_indexed_segments_target(),
        storage.named_vectors.has_dense_space_over_merge_target(),
        storage.named_vectors.has_building_segment_with_core(),
    );

    assert_eq!((debt.active_segments, debt.indexed_segments), (1, 0));
    assert!(!submit);
    assert!(!optimizer_has_pending_work(
        &storage,
        &["dense".into()],
        1_000,
        true
    ));
    assert_eq!(
        submit_post_open_vector_maintenance(&storage, &config, "coreless_building").unwrap(),
        0
    );
}

#[test]
fn already_indexed_collection_reopen_does_not_queue_a_scan_job() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "merged_reopen_test");
    let store = Arc::new(slatedb::object_store::memory::InMemory::new());
    let _store = crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store(store);
    let dir = TempDir::new().unwrap();
    let (_collections, mut config, storage, _) =
        sealed_lsm_collection(dir.path(), "merged_reopen", &["dense"], 16, 64);
    config.vector_config.flat_scan_threshold = Some(1_000);
    storage.named_vectors.set_indexing_threshold(usize::MAX);
    let prepared = {
        let read = storage.backend.begin_read().unwrap();
        storage
            .named_vectors
            .prepare_building_indices_limited_be(&read, "dense", 1)
            .unwrap()
    };
    assert_eq!(prepared.len(), 1);
    run_chunked_indices_publish(&storage, "dense", prepared).unwrap();
    run_index_job(IndexJob {
        collection: "merged_reopen".into(),
        storage: Arc::downgrade(&storage),
        spaces_to_check: vec!["dense".into()],
        hnsw_config: HNSWConfig::new(Some(16), Some(64), Some(128)),
        flat_scan_threshold: 1_000,
    });

    let submitted =
        submit_post_open_vector_maintenance(&storage, &config, "merged_reopen").unwrap();

    assert_eq!(
        submitted,
        0,
        "threshold {}, building {}, repair {:?}, debt {:?}",
        storage
            .named_vectors
            .effective_indexing_threshold(config.vector_flat_scan_threshold()),
        storage.named_vectors.has_building_segment_with_core(),
        storage
            .named_vectors
            .has_externalized_marker_repair_debt("dense"),
        storage.named_vectors.dense_segment_debt()
    );
}
