use super::*;
use crate::helix_engine::vector_core::named_vectors::DenseVectorSegmentRole;
use crate::helix_engine::vector_core::spindle::{SpindleConfig, SpindleMode};
use crate::helix_engine::vector_core::vector_core::{
    with_small_segment_exact_for_test, SMALL_SEGMENT_EXACT_MAX_ROWS,
};

const THRESHOLD: usize = 1_000;
/// Far longer than any test: the collection is quiet only when a test says so.
const LONG_QUIET_MS: &str = "600000";

fn dense_collection(
    dir: &TempDir,
    name: &str,
) -> (Arc<CollectionManager>, Config, Arc<HelixGraphStorage>) {
    dense_collection_with(dir, name, exact_spindle())
}

/// Raw f32 storage, as production creates collections: the tail has a sidecar.
fn exact_spindle() -> SpindleConfig {
    SpindleConfig {
        mode: SpindleMode::None,
        keep_original: false,
        rescore: false,
        ..SpindleConfig::default()
    }
}

fn dense_collection_with(
    dir: &TempDir,
    name: &str,
    spindle: SpindleConfig,
) -> (Arc<CollectionManager>, Config, Arc<HelixGraphStorage>) {
    let mut config = Config::new(8, 32, 64, 1).with_lsm_in_memory();
    config.vector_config.flat_scan_threshold = Some(THRESHOLD);
    let collections =
        Arc::new(CollectionManager::new(dir.path().join("data"), config.clone()).unwrap());
    let storage = create_dense(&collections, &config, name, spindle);
    (collections, config, storage)
}

fn create_dense(
    collections: &Arc<CollectionManager>,
    config: &Config,
    name: &str,
    spindle: SpindleConfig,
) -> Arc<HelixGraphStorage> {
    apply_mutation(
        collections,
        config,
        &ReplicatedMutation::CreateCollection {
            name: name.into(),
            vectors: HashMap::from([(
                "dense".to_string(),
                NamedVectorConfig {
                    size: 4,
                    distance: DistanceMetric::Cosine,
                    spindle,
                },
            )]),
            sparse_vectors: HashMap::new(),
            hnsw_overrides: None,
        },
    )
    .unwrap();
    collections.get_collection(name).unwrap()
}

/// Upserts `ids` in one batch and lets the index job that batch submits finish.
fn upsert(
    collections: &Arc<CollectionManager>,
    config: &Config,
    name: &str,
    ids: std::ops::Range<u128>,
) {
    apply_mutation(
        collections,
        config,
        &ReplicatedMutation::UpsertPoints {
            collection: name.into(),
            points: ids
                .map(|id| ReplicatedPoint {
                    id,
                    vectors: HashMap::from([("dense".to_string(), row_vector(id))]),
                    sparse_vectors: HashMap::new(),
                    payload: HashMap::new(),
                })
                .collect(),
        },
    )
    .unwrap();
    wait_for_index_executor(name);
}

fn row_vector(id: u128) -> Vec<f32> {
    vec![1.0, 0.01 * id as f32, 0.0, 0.0]
}

/// Rows in segments without a usable index: the tail's, a building segment's
/// and those of an indexed segment that lost its entry point.
fn unindexed(storage: &HelixGraphStorage) -> u64 {
    let read = storage.backend.begin_read().unwrap();
    let stats = storage
        .named_vectors
        .get_dense_space_stats_be(&read, "dense")
        .unwrap();
    stats.vectors_count - stats.indexed_vectors_count
}

fn tail_rows(storage: &HelixGraphStorage) -> u64 {
    let read = storage.backend.begin_read().unwrap();
    storage
        .named_vectors
        .mutable_tail_count_upto_be(&read, "dense", u64::MAX)
        .unwrap()
}

/// Appends rows to the mutable tail in `write`, as an upsert chunk does.
fn stage_rows(
    storage: &HelixGraphStorage,
    write: &mut crate::helix_engine::storage_core::backend_any::AnyWrite<'_>,
    ids: std::ops::Range<u128>,
) {
    for id in ids {
        storage
            .named_vectors
            .dense_append_to_mutable_be(write, "dense", &row_vector(id), id, HashMap::new())
            .unwrap();
    }
}

/// Segments of `role`, oldest first.
fn segments(storage: &HelixGraphStorage, role: DenseVectorSegmentRole) -> Vec<String> {
    dense_layout(storage)
        .into_iter()
        .filter(|(_, segment_role)| *segment_role == role)
        .map(|(name, _)| name)
        .collect()
}

fn has_pending_work(storage: &Arc<HelixGraphStorage>) -> bool {
    optimizer_has_pending_work(storage, &["dense".to_string()], THRESHOLD, true)
}

fn run_job(storage: &Arc<HelixGraphStorage>, name: &str) {
    run_index_job(IndexJob {
        collection: name.into(),
        storage: Arc::downgrade(storage),
        spaces_to_check: vec!["dense".into()],
        hnsw_config: HNSWConfig::new(Some(8), Some(32), Some(64)),
        flat_scan_threshold: THRESHOLD,
    });
}

/// As if the last mutation had ended long ago.
fn go_quiet(storage: &HelixGraphStorage) {
    storage.last_upsert_at.store(1, Ordering::Release);
}

#[test]
fn seal_rule_for_a_tail_below_the_threshold() {
    // Quiet seals any tail that has rows.
    assert!(subthreshold_tail_is_sealable(1, true, 2048));
    assert!(!subthreshold_tail_is_sealable(0, true, 2048));
    // Otherwise the tail waits for the seal minimum.
    assert!(!subthreshold_tail_is_sealable(2047, false, 2048));
    assert!(subthreshold_tail_is_sealable(2048, false, 2048));
    // 0 leaves quiet as the only trigger, as before the debounce.
    assert!(!subthreshold_tail_is_sealable(1_000_000, false, 0));

    let _guard = ENV_LOCK.lock().unwrap();
    let _min = EnvVarGuard::set("HELIX_INDEX_TAIL_SEAL_MIN_VECTORS", "5000");
    assert_eq!(index_tail_seal_min_vectors(THRESHOLD), THRESHOLD);
    assert_eq!(index_tail_seal_min_vectors(16_384), 5000);
}

#[test]
fn tail_stays_flat_until_it_reaches_the_seal_minimum() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _quiet = EnvVarGuard::set("HELIX_INDEX_TAIL_QUIET_MS", LONG_QUIET_MS);
    let _min = EnvVarGuard::set("HELIX_INDEX_TAIL_SEAL_MIN_VECTORS", "8");
    let dir = TempDir::new().unwrap();
    let name = "debounce_seal_minimum";
    let (collections, config, storage) = dense_collection(&dir, name);

    // Given a tail below the minimum in a collection that is not quiet.
    upsert(&collections, &config, name, 0..7);

    // Then neither the upsert's own job nor another one seals it or waits for
    // quiet, and nothing reports it as work.
    run_job(&storage, name);
    assert_eq!(unindexed(&storage), 7);
    assert!(!has_pending_work(&storage));

    // When an upsert takes the tail to the minimum, its own job seals and builds it.
    upsert(&collections, &config, name, 7..8);
    assert_eq!(unindexed(&storage), 0);

    // And the next rows start a new flat tail.
    upsert(&collections, &config, name, 8..10);
    assert_eq!(unindexed(&storage), 2);
}

#[test]
fn quiet_collection_seals_a_tail_below_the_minimum() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _quiet = EnvVarGuard::set("HELIX_INDEX_TAIL_QUIET_MS", LONG_QUIET_MS);
    let _min = EnvVarGuard::set("HELIX_INDEX_TAIL_SEAL_MIN_VECTORS", "8");
    let dir = TempDir::new().unwrap();
    let name = "debounce_quiet";
    let (collections, config, storage) = dense_collection(&dir, name);
    upsert(&collections, &config, name, 0..3);
    assert_eq!(unindexed(&storage), 3);
    assert!(!has_pending_work(&storage));

    // A mutation being applied is not quiet, however old the last stamp is,
    // and it stamps its end.
    go_quiet(&storage);
    {
        let _in_flight = PointMutationInFlight::begin(&storage);
        assert!(!point_mutations_quiet_for(&storage, 600_000));
        assert!(!has_pending_work(&storage));
    }
    assert!(!point_mutations_quiet_for(&storage, 600_000));

    // A delete is a mutation too.
    go_quiet(&storage);
    apply_mutation(
        &collections,
        &config,
        &ReplicatedMutation::DeletePoints {
            collection: name.into(),
            ids: vec![2],
        },
    )
    .unwrap();
    assert_eq!(storage.point_mutations_in_flight.load(Ordering::Acquire), 0);
    assert!(!point_mutations_quiet_for(&storage, 600_000));
    assert!(!has_pending_work(&storage));

    // Once the collection is quiet the tail is work, and one job seals it.
    go_quiet(&storage);
    assert!(has_pending_work(&storage));
    run_job(&storage, name);
    assert_eq!(unindexed(&storage), 0);
    assert!(!has_pending_work(&storage));
}

#[test]
fn backlog_sweeper_submits_a_collection_once_it_is_quiet() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _quiet = EnvVarGuard::set("HELIX_INDEX_TAIL_QUIET_MS", LONG_QUIET_MS);
    let _min = EnvVarGuard::set("HELIX_INDEX_TAIL_SEAL_MIN_VECTORS", "8");
    let _sweep = EnvVarGuard::set("HELIX_OPTIMIZER_BACKLOG_SWEEP_MS", "25");
    let _cooldown = EnvVarGuard::set("HELIX_OPTIMIZER_BACKLOG_URGENT_COOLDOWN_MS", "0");
    let dir = TempDir::new().unwrap();
    let name = "debounce_sweeper";
    let (collections, config, storage) = dense_collection(&dir, name);
    upsert(&collections, &config, name, 0..3);
    let submits = || {
        BACKLOG_SWEEPER_SUBMITS
            .lock()
            .unwrap()
            .iter()
            .filter(|submitted| *submitted == name)
            .count()
    };

    // Given the sweeper running over a collection with a small tail that is
    // not quiet: ten sweeps leave it alone.
    let _replication = ReplicationManager::new(Arc::clone(&collections), config.clone()).unwrap();
    thread::sleep(Duration::from_millis(250));
    assert_eq!(submits(), 0);
    assert_eq!(unindexed(&storage), 3);

    // When it goes quiet, the sweeper submits it and the job seals the tail.
    go_quiet(&storage);
    let sealed = |storage: &HelixGraphStorage| {
        let deadline = Instant::now() + Duration::from_secs(20);
        while unindexed(storage) > 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        unindexed(storage) == 0
    };
    assert!(sealed(&storage));

    // Then that one submit settles the quiet period: twelve more sweeps do
    // not submit the collection again, whatever its job did.
    wait_for_index_executor(name);
    thread::sleep(Duration::from_millis(300));
    assert_eq!(submits(), 1);

    // And the next mutation's quiet period gets its own submit.
    upsert(&collections, &config, name, 3..5);
    assert_eq!(tail_rows(&storage), 2);
    storage.last_upsert_at.store(2, Ordering::Release);
    assert!(sealed(&storage));
    wait_for_index_executor(name);
    thread::sleep(Duration::from_millis(300));
    assert_eq!(submits(), 2);
}

#[test]
fn backlog_sweeper_leaves_a_quiet_bulk_load_flat() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _quiet = EnvVarGuard::set("HELIX_INDEX_TAIL_QUIET_MS", LONG_QUIET_MS);
    let _sweep = EnvVarGuard::set("HELIX_OPTIMIZER_BACKLOG_SWEEP_MS", "25");
    let _cooldown = EnvVarGuard::set("HELIX_OPTIMIZER_BACKLOG_URGENT_COOLDOWN_MS", "0");
    let dir = TempDir::new().unwrap();
    let name = "debounce_sweeper_bulk";
    let (collections, config, storage) = dense_collection(&dir, name);

    // Given a collection in bulk mode (indexing threshold 0) that has gone quiet.
    storage.named_vectors.set_indexing_threshold(0);
    upsert(&collections, &config, name, 0..3);
    go_quiet(&storage);

    // When the sweeper passes over it twelve times.
    let _replication = ReplicationManager::new(Arc::clone(&collections), config.clone()).unwrap();
    thread::sleep(Duration::from_millis(300));

    // Then it submitted no job, and the tail is as the load left it.
    let submitted = BACKLOG_SWEEPER_SUBMITS.lock().unwrap().clone();
    assert!(!submitted.iter().any(|submitted| submitted == name));
    assert_eq!(tail_rows(&storage), 3);
}

/// A collection in its cooldown takes none of a sweep's submits. Thirty-two
/// with debt no job clears, a sweep's whole count in production, sort before
/// a collection that goes quiet, and the quiet one still gets its submit.
#[test]
fn backlog_sweeper_submits_a_quiet_collection_behind_a_sweep_of_cooling_ones() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _quiet = EnvVarGuard::set("HELIX_INDEX_TAIL_QUIET_MS", LONG_QUIET_MS);
    let _min = EnvVarGuard::set("HELIX_INDEX_TAIL_SEAL_MIN_VECTORS", "8");
    let _sweep = EnvVarGuard::set("HELIX_OPTIMIZER_BACKLOG_SWEEP_MS", "25");
    let _per_sweep = EnvVarGuard::set("HELIX_OPTIMIZER_BACKLOG_PER_SWEEP", "32");
    let _cooldown = EnvVarGuard::set("HELIX_OPTIMIZER_BACKLOG_SUBMIT_COOLDOWN_MS", LONG_QUIET_MS);
    // A space with rows and no indexed segment counts as gate debt, which
    // takes the urgent cooldown: every collection here does.
    let _urgent = EnvVarGuard::set("HELIX_OPTIMIZER_BACKLOG_URGENT_COOLDOWN_MS", LONG_QUIET_MS);
    let dir = TempDir::new().unwrap();
    // The sweeper takes equal scores by name: every `cooling_` before it.
    let name = "quiet_behind_cooling";
    let (collections, config, storage) = dense_collection(&dir, name);
    upsert(&collections, &config, name, 0..3);

    // Given thirty-two collections with a sealed segment left to build. A
    // degraded collection's submit ends before the executor, so the sweeper
    // counts each as submitted and no job ever builds the segment.
    let cooling: Vec<(String, Arc<HelixGraphStorage>)> = (0..32)
        .map(|n| {
            let name = format!("cooling_{n:02}");
            let storage = create_dense(&collections, &config, &name, exact_spindle());
            upsert(&collections, &config, &name, 0..1);
            seal_lsm_dense_tail(
                &storage,
                "dense",
                &HNSWConfig::new(Some(8), Some(32), Some(64)),
            )
            .unwrap();
            storage.mark_collection_degraded(
                &GraphError::FatalCollectionStorage {
                    code: "test",
                    message: "debt no job clears".into(),
                },
                "test",
            );
            (name, storage)
        })
        .collect();
    let submits = |name: &str| {
        BACKLOG_SWEEPER_SUBMITS
            .lock()
            .unwrap()
            .iter()
            .filter(|submitted| *submitted == name)
            .count()
    };
    let wait_for = |done: &dyn Fn() -> bool| {
        let deadline = Instant::now() + Duration::from_secs(20);
        while !done() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        done()
    };

    // And the sweeper has submitted each of them once: all thirty-two are in
    // their cooldown, their debt as it was.
    let _replication = ReplicationManager::new(Arc::clone(&collections), config.clone()).unwrap();
    assert!(wait_for(&|| cooling
        .iter()
        .all(|(name, _)| submits(name) == 1)));
    for (name, storage) in &cooling {
        assert!(
            storage.named_vectors.has_building_segment_with_core(),
            "{name}"
        );
    }
    assert_eq!(submits(name), 0);

    // When the collection behind them goes quiet.
    go_quiet(&storage);

    // Then it is submitted, its job seals the tail, and none of the others
    // was submitted again.
    assert!(
        wait_for(&|| unindexed(&storage) == 0),
        "the quiet collection was submitted {} times",
        submits(name)
    );
    assert_eq!(submits(name), 1);
    for (name, _) in &cooling {
        assert_eq!(submits(name), 1, "{name}");
    }
    wait_for_index_executor(name);
}

/// A tail sealed for its size is a segment the exact scan still takes. Its
/// job seals it with the rows that arrived since it reached the minimum: at
/// a minimum of 2,048 such tails held 2,050 to 2,300 rows on object storage.
/// The default leaves that many under the scan's row limit.
#[test]
fn default_seal_minimum_seals_a_tail_under_the_exact_scan_limit() {
    let past_the_minimum = 2_300 - 2_048;
    assert!(
        INDEX_TAIL_SEAL_MIN_VECTORS as u64 + past_the_minimum <= SMALL_SEGMENT_EXACT_MAX_ROWS,
        "a tail sealed at {INDEX_TAIL_SEAL_MIN_VECTORS} rows and {past_the_minimum} more is walked"
    );
}

#[test]
#[serial_test::serial]
fn rows_of_a_segment_without_its_entry_point_do_not_seal_the_tail() {
    let _guard = ENV_LOCK.lock().unwrap();
    // Hard deletes: the entry point's rows leave the segment, its key stays.
    // A delete moves the key to a surviving row now, so this one runs with
    // that off: segments stored before it are in this state, and a delete
    // that finds no survivor in reach still leaves one. `serial` with the
    // tests that flip the same switch.
    let _hard_deletes = EnvVarGuard::unset("HELIX_LSM_DELETE_TOMBSTONES");
    let _repoint = EnvVarGuard::set("HELIX_DENSE_REPOINT_ENTRY", "0");
    let _quiet = EnvVarGuard::set("HELIX_INDEX_TAIL_QUIET_MS", LONG_QUIET_MS);
    let _min = EnvVarGuard::set("HELIX_INDEX_TAIL_SEAL_MIN_VECTORS", "8");
    let dir = TempDir::new().unwrap();
    let name = "debounce_dead_entry_point";
    let (collections, config, storage) = dense_collection(&dir, name);

    // Given an indexed segment of 16 rows whose entry point is then deleted:
    // its 15 rows count as unindexed from here on, twice the seal minimum.
    upsert(&collections, &config, name, 0..16);
    let indexed = segments(&storage, DenseVectorSegmentRole::Indexed);
    assert_eq!((indexed.len(), unindexed(&storage)), (1, 0));
    let entry_point = {
        let read = storage.backend.begin_read().unwrap();
        let cores = storage.named_vectors.cores_read().unwrap();
        cores[&indexed[0]].get_entry_point(&read).unwrap().get_id()
    };
    apply_mutation(
        &collections,
        &config,
        &ReplicatedMutation::DeletePoints {
            collection: name.into(),
            ids: vec![entry_point],
        },
    )
    .unwrap();
    wait_for_index_executor(name);
    assert_eq!((unindexed(&storage), tail_rows(&storage)), (15, 0));

    // When batches below the minimum arrive.
    upsert(&collections, &config, name, 100..103);
    upsert(&collections, &config, name, 103..106);

    // Then they share one flat tail: no batch's job sealed it, and no job
    // reports the collection as work.
    assert_eq!(tail_rows(&storage), 6);
    assert_eq!(segments(&storage, DenseVectorSegmentRole::Indexed), indexed);
    assert!(segments(&storage, DenseVectorSegmentRole::Building).is_empty());
    assert!(!has_pending_work(&storage));

    // And once quiet, the tail is the work, not the other segment's rows:
    // one job seals it and nothing is pending afterwards.
    go_quiet(&storage);
    assert!(has_pending_work(&storage));
    run_job(&storage, name);
    wait_for_index_executor(name);
    assert_eq!((unindexed(&storage), tail_rows(&storage)), (15, 0));
    assert!(!has_pending_work(&storage));
}

#[test]
fn graph_of_a_sealed_tail_includes_the_batch_in_flight_at_the_seal() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _quiet = EnvVarGuard::set("HELIX_INDEX_TAIL_QUIET_MS", LONG_QUIET_MS);
    let _min = EnvVarGuard::set("HELIX_INDEX_TAIL_SEAL_MIN_VECTORS", "8");
    // Far longer than the batch below lasts: the job is still waiting for it
    // when it ends. A job out of budget hands the build over whatever it
    // waited for, which would show nothing here.
    let _budget = EnvVarGuard::set("HELIX_INDEX_JOB_MAX_RUNTIME_MS", "60000");
    let dir = TempDir::new().unwrap();
    let name = "debounce_seal_in_flight";
    let (collections, config, storage) = dense_collection(&dir, name);
    upsert(&collections, &config, name, 0..7);
    assert_eq!(tail_rows(&storage), 7);

    // Given a batch being applied: its first chunk committed took the tail
    // past the minimum, its second is staged into the same tail.
    storage.point_mutations.fetch_add(1, Ordering::AcqRel);
    let in_flight = PointMutationInFlight::begin(&storage);
    let mut first_chunk = storage.backend.begin_write().unwrap();
    stage_rows(&storage, &mut first_chunk, 7..9);
    storage.backend.commit(first_chunk).unwrap();
    let mut second_chunk = storage.backend.begin_write().unwrap();
    stage_rows(&storage, &mut second_chunk, 9..11);
    // While it lasts, the wait for it ends only by giving up.
    assert!(!point_mutation_in_flight_has_ended(&storage, || true));

    // When an index job runs now.
    let job = {
        let storage = Arc::clone(&storage);
        thread::spawn(move || run_job(&storage, name))
    };

    // Then it seals the tail, and builds no graph while the batch is in
    // flight: the staged rows are in no read yet.
    let deadline = Instant::now() + Duration::from_secs(30);
    while segments(&storage, DenseVectorSegmentRole::Building).is_empty() {
        assert!(Instant::now() < deadline, "the job did not seal the tail");
        thread::sleep(Duration::from_millis(5));
    }
    let sealed = segments(&storage, DenseVectorSegmentRole::Building);
    assert_eq!(sealed.len(), 1);
    assert_eq!(tail_rows(&storage), 0);
    thread::sleep(Duration::from_millis(500));
    assert_eq!(segments(&storage, DenseVectorSegmentRole::Building), sealed);
    assert!(segments(&storage, DenseVectorSegmentRole::Indexed).is_empty());

    // When the batch commits and ends, the job builds the graph.
    storage.backend.commit(second_chunk).unwrap();
    drop(in_flight);
    assert!(point_mutation_in_flight_has_ended(&storage, || true));
    job.join().unwrap();
    wait_for_index_executor(name);
    assert_eq!(segments(&storage, DenseVectorSegmentRole::Indexed), sealed);

    // Then a walk of that graph reaches every row, the staged ones included.
    let read = storage.backend.begin_read().unwrap();
    let cores = storage.named_vectors.cores_read().unwrap();
    let mut reached: Vec<u128> = with_small_segment_exact_for_test(false, || {
        cores[&sealed[0]].search_with_id_filter_ef_observed(
            &read,
            &row_vector(10),
            16,
            None::<&[fn(u128) -> bool]>,
            false,
            None,
            Some(128),
            None,
        )
    })
    .unwrap()
    .into_iter()
    .map(|hit| hit.get_id())
    .collect();
    reached.sort_unstable();
    assert_eq!(reached, (0..11).collect::<Vec<u128>>());
}

#[test]
fn quantized_collection_keeps_the_rule_before_the_debounce() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _quiet = EnvVarGuard::set("HELIX_INDEX_TAIL_QUIET_MS", LONG_QUIET_MS);
    let _min = EnvVarGuard::set("HELIX_INDEX_TAIL_SEAL_MIN_VECTORS", "8");
    let dir = TempDir::new().unwrap();
    let name = "debounce_quantized";
    // Given a collection whose dense space is quantized: its tail has no
    // sidecar, so a reader replica reads every tail row it scores.
    let (collections, config, storage) =
        dense_collection_with(&dir, name, SpindleConfig::default());
    assert!(storage.named_vectors.has_spindle_dense_space());

    // Then the upsert's own job waits one second of quiet out and seals a
    // tail below the minimum, with no sweeper running.
    let started = Instant::now();
    upsert(&collections, &config, name, 0..3);
    assert_eq!(unindexed(&storage), 0);
    assert!(started.elapsed() >= Duration::from_millis(1000));
}

#[test]
fn rule_before_the_debounce_is_one_second_of_quiet_and_no_minimum() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _quiet = EnvVarGuard::set("HELIX_INDEX_TAIL_QUIET_MS", "1000");
    let _min = EnvVarGuard::set("HELIX_INDEX_TAIL_SEAL_MIN_VECTORS", "0");
    let dir = TempDir::new().unwrap();
    let name = "debounce_off";
    let (collections, config, storage) = dense_collection(&dir, name);

    // The upsert's own job waits the second out and seals the tail, with no
    // sweeper running.
    let started = Instant::now();
    upsert(&collections, &config, name, 0..3);
    assert_eq!(unindexed(&storage), 0);
    assert!(started.elapsed() >= Duration::from_millis(1000));
}

/// A create that names the spaces a collection already has makes no space,
/// and stores no dense layout. It runs on a request thread beside the index
/// job and takes nothing the job takes: stored there, a layout it read
/// before a seal's flip could land after the seal's own store and leave the
/// stored layout without the flip. The client sends such a create at the start of
/// an index run.
#[test]
fn create_of_a_collection_that_has_its_spaces_stores_no_dense_layout() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _quiet = EnvVarGuard::set("HELIX_INDEX_TAIL_QUIET_MS", LONG_QUIET_MS);
    let dir = TempDir::new().unwrap();
    let name = "create_again_stores_no_layout";
    let (collections, config, storage) = dense_collection(&dir, name);
    upsert(&collections, &config, name, 0..3);
    let stored = || {
        let read = storage.backend.begin_read().unwrap();
        storage.get_metadata_be(&read).unwrap().dense_vector_spaces["dense"]
            .segments
            .iter()
            .map(|segment| (segment.physical_name.clone(), segment.role))
            .collect::<Vec<_>>()
    };
    // The create that made the space stored its layout.
    let before = stored();
    assert_eq!(before, dense_layout(&storage));

    // Given a layout in memory that has moved on from the stored one, as it
    // has between a seal's flip and the seal's own store.
    {
        let read = storage.backend.begin_read().unwrap();
        storage
            .named_vectors
            .seal_mutable_tail_lsm(
                &read,
                storage.collection_path(),
                "dense",
                HNSWConfig::new(Some(8), Some(32), Some(64)),
            )
            .unwrap()
            .expect("a tail with rows is sealed");
    }
    assert_ne!(dense_layout(&storage), before);

    // When the collection is created again with the space it has.
    apply_mutation(
        &collections,
        &config,
        &ReplicatedMutation::CreateCollection {
            name: name.into(),
            vectors: HashMap::from([(
                "dense".to_string(),
                NamedVectorConfig {
                    size: 4,
                    distance: DistanceMetric::Cosine,
                    spindle: exact_spindle(),
                },
            )]),
            sparse_vectors: HashMap::new(),
            hnsw_overrides: None,
        },
    )
    .unwrap();

    // Then it stored no layout: what is stored is what the last store of a
    // layout left, and the next one is the seal's.
    assert_eq!(stored(), before);
}

/// A create makes its space in memory and then stores it. When that store
/// fails, the space stays in memory and the collection is not degraded. The
/// next create finds the space there and makes none, and it has to be the
/// one that stores it: nothing else stores the config of a named vector, so
/// the collection would open without its space the next time.
#[test]
fn create_after_a_create_whose_store_failed_stores_the_space() {
    let _guard = ENV_LOCK.lock().unwrap();
    let dir = TempDir::new().unwrap();
    let name = "create_again_after_failed_store";
    let config = Config::new(8, 32, 64, 1).with_lsm_in_memory();
    let collections =
        Arc::new(CollectionManager::new(dir.path().join("data"), config.clone()).unwrap());
    let storage = collections.create_collection(name).unwrap();
    let create = ReplicatedMutation::CreateCollection {
        name: name.into(),
        vectors: HashMap::from([(
            "dense".to_string(),
            NamedVectorConfig {
                size: 4,
                distance: DistanceMetric::Cosine,
                spindle: exact_spindle(),
            },
        )]),
        sparse_vectors: HashMap::new(),
        hnsw_overrides: None,
    };
    let stored = || {
        let read = storage.backend.begin_read().unwrap();
        let metadata = storage.get_metadata_be(&read).unwrap();
        (
            metadata.named_vectors.contains_key("dense"),
            metadata.dense_vector_spaces.contains_key("dense"),
        )
    };
    let in_memory = || {
        storage
            .named_vectors
            .list_dense_vector_spaces()
            .contains_key("dense")
    };

    // Given a create whose store failed. The flag is how this test fails
    // one store: the create makes its space without asking for it and the
    // store asks first. It is taken back, as a store that failed with an
    // error that is not fatal leaves the collection usable.
    storage.collection_degraded.store(true, Ordering::Release);
    let failed = apply_mutation(&collections, &config, &create);
    storage.collection_degraded.store(false, Ordering::Release);
    assert!(failed.is_err(), "the create's store went through");
    assert!(in_memory(), "the create failed before it made its space");
    assert_eq!(stored(), (false, false));

    // When the collection is created again.
    apply_mutation(&collections, &config, &create).unwrap();

    // Then the space is stored.
    assert_eq!(stored(), (true, true));
}

/// The job's cleanup takes a `Building` segment with no rows for a merge
/// target that never published, and drops it from the layout. A row count
/// it could not read is not a count of none: a sealed tail that waits for
/// its build stays in the layout, with its rows.
#[test]
fn orphan_cleanup_leaves_a_sealed_tail_whose_rows_it_could_not_count() {
    use crate::helix_engine::storage_core::backend_lsm::{
        allow_lsm_blocking_cancellable, LsmReadCancellation,
    };

    let _guard = ENV_LOCK.lock().unwrap();
    let _quiet = EnvVarGuard::set("HELIX_INDEX_TAIL_QUIET_MS", LONG_QUIET_MS);
    let dir = TempDir::new().unwrap();
    let name = "orphan_cleanup_count_error";
    let (collections, config, storage) = dense_collection(&dir, name);
    upsert(&collections, &config, name, 0..3);
    let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
    assert!(seal_lsm_dense_tail(&storage, "dense", &hnsw).unwrap());
    let sealed = segments(&storage, DenseVectorSegmentRole::Building);
    assert_eq!(sealed.len(), 1);
    let read = storage.backend.begin_read().unwrap();
    let cleanup = || {
        storage
            .named_vectors
            .gc_orphan_building_segments_deferred_be(&read, "dense")
    };

    // When the cleanup's reads fail: here its caller has left.
    let gone = LsmReadCancellation::new();
    gone.cancel();
    let orphans = allow_lsm_blocking_cancellable(gone, || {
        let count = storage
            .named_vectors
            .with_core(&sealed[0], |core| core.level_zero_count(&read));
        assert!(count.is_err(), "the count was read: {count:?}");
        cleanup()
    })
    .unwrap();

    // Then it takes nothing for an orphan, and the tail is where it was.
    assert_eq!(orphans, Vec::<String>::new());
    assert_eq!(segments(&storage, DenseVectorSegmentRole::Building), sealed);
    // As it is after a cleanup whose reads work: the tail has its rows.
    assert_eq!(cleanup().unwrap(), Vec::<String>::new());
    assert_eq!(segments(&storage, DenseVectorSegmentRole::Building), sealed);
}
