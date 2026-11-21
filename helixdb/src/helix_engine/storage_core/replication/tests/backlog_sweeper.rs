use super::*;
use crate::helix_engine::vector_core::named_vectors::DenseVectorSegmentRole;

fn lsm_collections(dir: &TempDir) -> (Arc<CollectionManager>, Config) {
    let mut config = Config::new(8, 32, 64, 1).with_lsm_in_memory();
    // 0 keeps upserts from queueing their own index job behind the test's back.
    config.vector_config.flat_scan_threshold = Some(0);
    let collections =
        Arc::new(CollectionManager::new(dir.path().join("data"), config.clone()).unwrap());
    (collections, config)
}

fn create_collection(
    collections: &Arc<CollectionManager>,
    config: &Config,
    name: &str,
    spaces: &[&str],
) {
    apply_mutation(
        collections,
        config,
        &ReplicatedMutation::CreateCollection {
            name: name.into(),
            vectors: spaces
                .iter()
                .map(|space| {
                    (
                        space.to_string(),
                        NamedVectorConfig {
                            size: 4,
                            distance: DistanceMetric::Cosine,
                            spindle: Default::default(),
                        },
                    )
                })
                .collect(),
            sparse_vectors: HashMap::new(),
            hnsw_overrides: None,
        },
    )
    .unwrap();
}

fn hnsw() -> HNSWConfig {
    HNSWConfig::new(Some(8), Some(32), Some(64))
}

/// Upserts point `id` into `space` and seals it: one Building segment.
fn add_building_segment(
    collections: &Arc<CollectionManager>,
    config: &Config,
    name: &str,
    space: &str,
    id: u128,
) {
    apply_mutation(
        collections,
        config,
        &ReplicatedMutation::UpsertPoints {
            collection: name.into(),
            points: vec![ReplicatedPoint {
                id,
                vectors: HashMap::from([(space.to_string(), vec![1.0, 0.0, 0.0, 0.0])]),
                sparse_vectors: HashMap::new(),
                payload: HashMap::new(),
            }],
        },
    )
    .unwrap();
    let storage = collections.get_collection(name).unwrap();
    seal_lsm_dense_tail(&storage, space, &hnsw()).unwrap();
}

/// Adds `count` Indexed segments of one point each to `space`.
fn add_indexed_segments(
    collections: &Arc<CollectionManager>,
    config: &Config,
    name: &str,
    space: &str,
    count: usize,
    next_id: &mut u128,
) {
    let storage = collections.get_collection(name).unwrap();
    for _ in 0..count {
        *next_id += 1;
        add_building_segment(collections, config, name, space, *next_id);
        let prepared = {
            let read = storage.backend.begin_read().unwrap();
            storage
                .named_vectors
                .prepare_building_indices_limited_be(&read, space, 1)
                .unwrap()
        };
        assert_eq!(prepared.len(), 1);
        run_chunked_indices_publish(&storage, space, prepared).unwrap();
    }
}

fn executor_state() -> std::sync::MutexGuard<'static, IndexExecutorState> {
    INDEX_EXECUTOR.state.0.lock().unwrap()
}

/// Waits until the executor holds no job for `name`, so none outlives the test.
fn wait_for_index_jobs(name: &str) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while index_job_queued_or_active(name) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(10));
    }
    assert!(!index_job_queued_or_active(name));
}

#[test]
fn backlog_sweeper_submits_only_when_a_space_has_a_merge() {
    let _guard = ENV_LOCK.lock().unwrap();
    let dir = TempDir::new().unwrap();
    let (collections, config) = lsm_collections(&dir);
    let name = "merged_multi_space";
    create_collection(&collections, &config, name, &["a", "b"]);
    let storage = collections.get_collection(name).unwrap();
    let mut next_id = 0u128;
    let mut add_indexed_segments = |space: &str, count: usize| {
        add_indexed_segments(&collections, &config, name, space, count, &mut next_id)
    };
    let target = NamedVectorManager::max_indexed_segments_target();
    // What the index job merges on, per space.
    let job_selects_merge = |space: &str| {
        let read = storage.backend.begin_read().unwrap();
        storage
            .named_vectors
            .select_merge_candidates_with_limits_be(
                &read,
                space,
                DenseMergeCandidateLimits::new(target, NamedVectorManager::merge_max_fan_in()),
            )
            .unwrap()
            .is_some()
    };
    let sweeper_submits = || {
        optimizer_debt_has_submit_work(
            storage.named_vectors.dense_segment_debt(),
            target,
            storage.named_vectors.has_dense_space_over_merge_target(),
            storage.named_vectors.has_building_segment_with_core(),
        )
    };

    // Given two spaces that are each within the merge target but together over
    // the sweeper's collection-wide threshold (the six-space production shape).
    add_indexed_segments("a", target);
    add_indexed_segments("b", 1);
    let debt = storage.named_vectors.dense_segment_debt();
    assert!(debt.indexed_segments > target);
    assert_eq!(debt.merge_debt, 0);
    assert_eq!(debt.gate_debt, 0);

    // Then the job has nothing to merge and the sweeper does not submit it.
    assert!(!job_selects_merge("a") && !job_selects_merge("b"));
    assert!(!sweeper_submits());

    // When one space goes over the target, both agree there is a merge.
    add_indexed_segments("a", 1);
    assert!(job_selects_merge("a") && !job_selects_merge("b"));
    assert!(sweeper_submits());

    // And once the job has merged it, the sweeper stops submitting.
    run_index_job(IndexJob {
        collection: name.into(),
        storage: Arc::downgrade(&storage),
        spaces_to_check: vec!["a".into(), "b".into()],
        hnsw_config: hnsw(),
        flat_scan_threshold: 1_000,
    });
    let indexed_in_a = storage.named_vectors.list_dense_vector_spaces()["a"]
        .segments
        .iter()
        .filter(|segment| segment.role == DenseVectorSegmentRole::Indexed)
        .count();
    assert!(
        indexed_in_a <= target,
        "space a kept {indexed_in_a} segments"
    );
    assert!(!sweeper_submits());
}

#[test]
fn backlog_sweeper_submits_a_building_segment_until_it_is_built() {
    let _guard = ENV_LOCK.lock().unwrap();
    let dir = TempDir::new().unwrap();
    let (collections, config) = lsm_collections(&dir);
    let name = "building_only";
    create_collection(&collections, &config, name, &["dense"]);
    let storage = collections.get_collection(name).unwrap();
    let sweeper_submits = || {
        optimizer_debt_has_submit_work(
            storage.named_vectors.dense_segment_debt(),
            NamedVectorManager::max_indexed_segments_target(),
            storage.named_vectors.has_dense_space_over_merge_target(),
            storage.named_vectors.has_building_segment_with_core(),
        )
    };
    assert!(!sweeper_submits());

    // Given a sealed segment whose build did not happen (its job ended on an
    // error), in a collection with no merge debt and no gate pressure.
    add_building_segment(&collections, &config, name, "dense", 1);
    let debt = storage.named_vectors.dense_segment_debt();
    assert_eq!((debt.indexed_segments, debt.active_segments), (0, 1));
    assert_eq!(debt.merge_debt, 0);

    // Then the sweeper submits it, and the submit takes it on that segment.
    assert!(sweeper_submits());
    assert!(optimizer_has_pending_work(
        &storage,
        &["dense".to_string()],
        1_000,
        true
    ));

    // And one job turn builds it, which leaves the sweeper nothing to resubmit.
    run_index_job(IndexJob {
        collection: name.into(),
        storage: Arc::downgrade(&storage),
        spaces_to_check: vec!["dense".into()],
        hnsw_config: hnsw(),
        flat_scan_threshold: 1_000,
    });
    let debt = storage.named_vectors.dense_segment_debt();
    assert_eq!((debt.indexed_segments, debt.active_segments), (1, 1));
    assert!(!sweeper_submits());
    wait_for_index_jobs(name);
}

#[test]
fn backlog_sweeper_loop_submits_only_collections_with_work() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _sweep = EnvVarGuard::set("HELIX_OPTIMIZER_BACKLOG_SWEEP_MS", "25");
    // Ratio 1 puts the collection-wide threshold at the per-space target, so
    // two spaces overshoot it the way six do against production's 2 x target.
    let _ratio = EnvVarGuard::set("HELIX_OPTIMIZER_BACKLOG_MIN_RATIO", "1");
    let dir = TempDir::new().unwrap();
    let (collections, config) = lsm_collections(&dir);
    let target = NamedVectorManager::max_indexed_segments_target();
    let mut next_id = 0u128;
    // Given a merged collection that is over the threshold only when its spaces
    // are summed, and one with a space over the merge target.
    create_collection(&collections, &config, "loop_merged", &["a", "b"]);
    add_indexed_segments(
        &collections,
        &config,
        "loop_merged",
        "a",
        target,
        &mut next_id,
    );
    add_indexed_segments(&collections, &config, "loop_merged", "b", 1, &mut next_id);
    create_collection(&collections, &config, "loop_debt", &["a"]);
    add_indexed_segments(
        &collections,
        &config,
        "loop_debt",
        "a",
        target + 1,
        &mut next_id,
    );
    let submitted = |name: &str| {
        BACKLOG_SWEEPER_SUBMITS
            .lock()
            .unwrap()
            .iter()
            .any(|submitted| submitted == name)
    };

    // When the sweeper runs over both.
    let _replication = ReplicationManager::new(Arc::clone(&collections), config.clone()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while !submitted("loop_debt") && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(10));
    }

    // Then it submits the one with a merge to do, and ten more sweeps still
    // leave the merged one alone.
    assert!(submitted("loop_debt"));
    thread::sleep(Duration::from_millis(250));
    assert!(!submitted("loop_merged"));
    wait_for_index_jobs("loop_debt");
}

#[test]
fn backlog_sweeper_submit_never_opens_or_touches() {
    let _guard = ENV_LOCK.lock().unwrap();
    let dir = TempDir::new().unwrap();
    let (collections, config) = lsm_collections(&dir);
    create_collection(&collections, &config, "older", &["dense"]);
    create_collection(&collections, &config, "newer", &["dense"]);
    drop(collections.get_collection("older").unwrap());
    drop(collections.get_collection("newer").unwrap());
    let by_recency = || collections.loaded_collection_names_by_recency().unwrap();
    assert_eq!(by_recency(), ["newer", "older"]);

    // A loaded collection goes through the submit without an LRU/idle touch.
    assert!(
        submit_loaded_collection_optimizer(&collections, &config, "older")
            .unwrap()
            .is_some()
    );
    assert_eq!(by_recency(), ["newer", "older"]);

    // One evicted between the sweeper's scan and its submit is not reopened.
    collections.evict_collection("older");
    assert_eq!(
        submit_loaded_collection_optimizer(&collections, &config, "older").unwrap(),
        None
    );
    assert_eq!(by_recency(), ["newer"]);
}

#[test]
fn backlog_sweeper_skips_a_fenced_writer() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "backlog_sweeper_fenced_writer");
    let store: Arc<dyn slatedb::object_store::ObjectStore> =
        Arc::new(slatedb::object_store::memory::InMemory::new());
    let _store = crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store(store);
    let config = Config::new(8, 32, 64, 1).with_storage_backend(StorageBackendConfig::Lsm {
        storage: LsmStorage::ObjectStore,
        role: LsmRole::Writer,
    });
    let dir = TempDir::new().unwrap();
    let name = "sweeper_fenced_writer";
    let write_node = |storage: &Arc<HelixGraphStorage>, id: u128| {
        storage.with_write_backend(|write| {
            storage.create_node_be(write, "Node", Vec::<(String, Value)>::new(), None, Some(id))
        })
    };
    // Given a loaded writer that another process has fenced.
    let stale = Arc::new(CollectionManager::new(dir.path().to_path_buf(), config.clone()).unwrap());
    let stale_storage = stale.create_collection(name).unwrap();
    write_node(&stale_storage, 1).unwrap();
    let owner = CollectionManager::new(dir.path().to_path_buf(), config.clone()).unwrap();
    let owner_storage = owner.get_collection(name).unwrap();
    write_node(&owner_storage, 2).unwrap();
    let fenced = write_node(&stale_storage, 3).unwrap_err();
    assert!(fenced.to_string().contains("fenced"), "got {fenced}");
    drop(stale_storage);
    assert_eq!(stale.loaded_count(), 1);

    // When the sweeper submits it.
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let submitted = metrics::with_local_recorder(&recorder, || {
        submit_loaded_collection_optimizer(&stale, &config, name)
    })
    .unwrap();

    // Then it is skipped under its own outcome, not as a collection that left.
    assert_eq!(submitted, None);
    let outcomes: Vec<String> = handle
        .render()
        .lines()
        .filter(|line| line.starts_with("helix_optimizer_backlog_sweeper_total"))
        .map(str::to_string)
        .collect();
    assert_eq!(
        outcomes,
        [r#"helix_optimizer_backlog_sweeper_total{outcome="fenced_writer"} 1"#]
    );
}

#[test]
fn idle_eviction_fires_while_background_loops_run() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _idle_ttl = EnvVarGuard::set("HELIX_COLLECTION_IDLE_EVICTION_MS", "1500");
    let _idle_sweep = EnvVarGuard::set("HELIX_COLLECTION_IDLE_SWEEP_MS", "25");
    let _min_cache = EnvVarGuard::set("HELIX_COLLECTION_IDLE_MIN_CACHE_SIZE", "1");
    let _backlog_sweep = EnvVarGuard::set("HELIX_OPTIMIZER_BACKLOG_SWEEP_MS", "25");
    // No cooldown: the backlog sweeper submits a candidate on every sweep.
    let _cooldown = EnvVarGuard::set("HELIX_OPTIMIZER_BACKLOG_SUBMIT_COOLDOWN_MS", "0");
    let _ratio = EnvVarGuard::set("HELIX_OPTIMIZER_BACKLOG_MIN_RATIO", "1");
    let dir = TempDir::new().unwrap();
    let (collections, config) = lsm_collections(&dir);
    // Given a collection that stays a backlog candidate: a space over the merge
    // target that no job will merge, because the collection is degraded.
    create_collection(&collections, &config, "debt", &["dense"]);
    let debt = collections.get_collection("debt").unwrap();
    let over_target = NamedVectorManager::max_indexed_segments_target() + 1;
    add_indexed_segments(&collections, &config, "debt", "dense", over_target, &mut 0);
    debt.mark_collection_degraded(
        &GraphError::FatalCollectionStorage {
            code: "test",
            message: "no optimizer job runs here".into(),
        },
        "test",
    );
    drop(debt);
    // And the idle sweeper and the optimizer backlog sweeper both running.
    let _replication = ReplicationManager::new(Arc::clone(&collections), config.clone()).unwrap();
    for name in ["one", "two"] {
        create_collection(&collections, &config, name, &["dense"]);
    }
    // Creating them can outlast the TTL, so only the newest is sure to be loaded.
    assert!(collections.loaded_count() > 0);

    // When the periodic snapshot loop also visits every loaded collection far
    // more often than the idle TTL and nothing else uses them.
    let deadline = Instant::now() + Duration::from_secs(20);
    while collections.loaded_count() > 0 && Instant::now() < deadline {
        collections.snapshot_loaded_collections().unwrap();
        thread::sleep(Duration::from_millis(20));
    }

    // Then all of them age out.
    assert_eq!(collections.loaded_count(), 0);
}

#[test]
fn idle_eviction_waits_for_a_queued_index_job() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _idle_ttl = EnvVarGuard::set("HELIX_COLLECTION_IDLE_EVICTION_MS", "1000");
    let _idle_sweep = EnvVarGuard::set("HELIX_COLLECTION_IDLE_SWEEP_MS", "25");
    let _min_cache = EnvVarGuard::set("HELIX_COLLECTION_IDLE_MIN_CACHE_SIZE", "1");
    let dir = TempDir::new().unwrap();
    let (collections, config) = lsm_collections(&dir);
    let name = "queued_follow_up";
    create_collection(&collections, &config, name, &["dense"]);
    let storage = collections.get_collection(name).unwrap();
    wait_for_index_jobs(name);

    // Given an idle collection with a follow-up job queued behind a running
    // one: `submit` parks it in `queued_jobs` until `finish_job` re-queues it.
    executor_state().queued_jobs.insert(
        name.to_string(),
        IndexJob {
            collection: name.into(),
            storage: Arc::downgrade(&storage),
            spaces_to_check: vec!["dense".into()],
            hnsw_config: hnsw(),
            flat_scan_threshold: 1_000,
        },
    );
    drop(storage);

    // When three idle TTLs pass, it is still loaded.
    thread::sleep(Duration::from_secs(3));
    let loaded_with_queued_job = collections.loaded_count();

    // And once the queue is empty, it ages out.
    executor_state().queued_jobs.remove(name);
    assert_eq!(loaded_with_queued_job, 1);
    let deadline = Instant::now() + Duration::from_secs(20);
    while collections.loaded_count() > 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(collections.loaded_count(), 0);
}
