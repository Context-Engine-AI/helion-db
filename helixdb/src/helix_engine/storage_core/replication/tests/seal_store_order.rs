//! The layout a collection has stored against the one its memory has: what a
//! writer that died beside a seal left for the next writer and for a reader,
//! what a seal that failed or died midway left in memory, and what two
//! changes of the layout that run beside each other leave stored.

use super::publish_crash::WalSnapshots;
use super::*;
use crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store;
use crate::helix_engine::vector_core::named_vectors::{
    DenseVectorSegmentMetadata, DenseVectorSegmentRole, DenseVectorSpaceMetadata,
};
use crate::helix_engine::vector_core::spindle::{SpindleConfig, SpindleMode};
use object_store::{memory::InMemory, ObjectStore};
use std::sync::atomic::AtomicUsize;

const NAME: &str = "seal_store_order";
/// A point the collection has before the seal, written again beside it.
const WRITTEN_AGAIN: u128 = 3;
/// A point first written beside the seal.
const NEW: u128 = 100;
/// The tail a seal of the first tail (`dense`) opens.
const SECOND_TAIL: &str = "dense__seg_000001";

/// A space of raw f32 rows, as production creates its collections. A
/// quantized space is sealed after a second of quiet.
pub(super) fn raw_space() -> NamedVectorConfig {
    NamedVectorConfig {
        size: 4,
        distance: DistanceMetric::Cosine,
        spindle: SpindleConfig {
            mode: SpindleMode::None,
            keep_original: false,
            rescore: false,
            ..SpindleConfig::default()
        },
    }
}

fn hnsw() -> HNSWConfig {
    HNSWConfig::new(Some(16), Some(64), Some(128))
}

fn row(id: u128, turn: f32) -> Vec<f32> {
    vec![1.0, 0.01 * id as f32, turn, 0.0]
}

fn upsert(writer: &Arc<CollectionManager>, config: &Config, ids: &[u128], turn: f32) {
    apply_mutation(
        writer,
        config,
        &ReplicatedMutation::UpsertPoints {
            collection: NAME.into(),
            points: ids
                .iter()
                .map(|id| ReplicatedPoint {
                    id: *id,
                    vectors: HashMap::from([("dense".to_string(), row(*id, turn))]),
                    sparse_vectors: HashMap::new(),
                    payload: HashMap::new(),
                })
                .collect(),
        },
    )
    .unwrap();
}

/// The settings every test here runs under: one WAL object a commit, and no
/// index job that seals the tail by itself.
fn settings(prefix: &'static str) -> [EnvVarGuard; 4] {
    [
        EnvVarGuard::set("HELIX_LSM_PREFIX", prefix),
        EnvVarGuard::unset("HELIX_LSM_GROUP_COMMIT"),
        EnvVarGuard::set("HELIX_INDEX_TAIL_QUIET_MS", "600000"),
        EnvVarGuard::set("HELIX_INDEX_TAIL_SEAL_MIN_VECTORS", "5000"),
    ]
}

fn writer_config() -> Config {
    let mut config = Config::new(16, 64, 128, 1).with_storage_backend(StorageBackendConfig::Lsm {
        storage: LsmStorage::ObjectStore,
        role: LsmRole::Writer,
    });
    config.vector_config.flat_scan_threshold = Some(1_000);
    config
}

fn reader_config() -> Config {
    Config::new(16, 64, 128, 1).with_storage_backend(StorageBackendConfig::Lsm {
        storage: LsmStorage::ObjectStore,
        role: LsmRole::Reader,
    })
}

/// A writer with one dense space and eight points in its tail, on the store
/// the test has set. No index job seals that tail: the tests do the seal.
fn writer_with_a_tail(
    dir: &std::path::Path,
) -> (Arc<CollectionManager>, Config, Arc<HelixGraphStorage>) {
    let config = writer_config();
    let writer = Arc::new(CollectionManager::new(dir.to_path_buf(), config.clone()).unwrap());
    apply_mutation(
        &writer,
        &config,
        &ReplicatedMutation::CreateCollection {
            name: NAME.into(),
            vectors: HashMap::from([("dense".to_string(), raw_space())]),
            sparse_vectors: HashMap::new(),
            hnsw_overrides: None,
        },
    )
    .unwrap();
    upsert(&writer, &config, &[1, 2, 3, 4, 5, 6, 7, 8], 0.0);
    wait_for_index_executor(NAME);
    let storage = writer.get_collection(NAME).unwrap();
    (writer, config, storage)
}

/// Seals the tail of `dense`: a tail with rows is sealed.
fn seal(storage: &Arc<HelixGraphStorage>) {
    assert!(
        seal_lsm_dense_tail(storage, "dense", &hnsw()).unwrap(),
        "no tail was sealed: {}",
        tail_state(storage)
    );
}

/// The segments the collection has open that hold a vector of point `id`.
fn segments_holding(storage: &HelixGraphStorage, id: u128) -> Vec<String> {
    let read = storage.backend.begin_read().unwrap();
    let cores = storage.named_vectors.cores_read().unwrap();
    let mut holding: Vec<String> = cores
        .iter()
        .filter(|(_, core)| core.contains_id(&read, id).unwrap())
        .map(|(name, _)| name.clone())
        .collect();
    holding.sort();
    holding
}

/// What the collection serves of `NEW` and of `WRITTEN_AGAIN`: the vector a
/// read of the point finds, none for a point no listed segment holds, and
/// whether a search for the vector they were last written with returns the
/// point. The search reads the segments a search reads: a sealed tail only
/// while it is marked.
fn served(storage: &HelixGraphStorage) -> [(Option<Vec<f32>>, bool); 2] {
    let read = storage.backend.begin_read().unwrap();
    [NEW, WRITTEN_AGAIN].map(|id| {
        let stored = storage
            .named_vectors
            .stored_dense_vector_be(&read, "dense", id);
        (stored, found_by_a_search(storage, id, 1.0))
    })
}

/// Whether a search of `dense` for the vector point `id` was written with
/// on `turn` returns the point. The ten nearest are every point there is.
fn found_by_a_search(storage: &HelixGraphStorage, id: u128, turn: f32) -> bool {
    let read = storage.backend.begin_read().unwrap();
    storage
        .named_vectors
        .dense_search_with_id_filter_ef_be::<fn(u128) -> bool>(
            &read,
            "dense",
            &row(id, turn),
            10,
            None,
            false,
            None,
            Some(128),
        )
        .unwrap()
        .iter()
        .any(|hit| hit.get_id() == id)
}

/// The vector the collection serves of `WRITTEN_AGAIN` now: the one a
/// test's second write of the point has to replace.
fn first_write(storage: &HelixGraphStorage) -> Vec<f32> {
    served(storage)[1]
        .0
        .clone()
        .expect("a point of the first eight")
}

/// `served` after both points were written beside a seal, by the writer
/// that took the writes: each point has a vector and is found by a search,
/// and `WRITTEN_AGAIN` no longer has the vector of its `first_write`.
fn served_whole(
    storage: &HelixGraphStorage,
    first_write: &[f32],
    who: &str,
) -> [(Option<Vec<f32>>, bool); 2] {
    let served = served(storage);
    assert!(
        served
            .iter()
            .all(|(vector, found)| vector.is_some() && *found),
        "by {who}: {served:?}"
    );
    assert_ne!(served[1].0.as_deref(), Some(first_write), "by {who}");
    served
}

/// The layout in memory and the space's row counts, for a failed step.
fn tail_state(storage: &HelixGraphStorage) -> String {
    let read = storage.backend.begin_read().unwrap();
    let stats = storage
        .named_vectors
        .get_dense_space_stats_be(&read, "dense")
        .unwrap();
    format!(
        "layout in memory {:?}, vectors {}, of them indexed {}",
        dense_layout(storage),
        stats.vectors_count,
        stats.indexed_vectors_count
    )
}

/// The space `dense` as the collection's memory has it: its segments, their
/// roles and its next id.
fn in_memory(storage: &HelixGraphStorage) -> DenseVectorSpaceMetadata {
    storage.named_vectors.list_dense_vector_spaces()["dense"].clone()
}

/// The same as the store has it.
fn stored(storage: &HelixGraphStorage) -> DenseVectorSpaceMetadata {
    let read = storage.backend.begin_read().unwrap();
    storage.get_metadata_be(&read).unwrap().dense_vector_spaces["dense"].clone()
}

/// The segments the collection has a core open for.
fn cores_open(storage: &HelixGraphStorage) -> Vec<String> {
    let mut open: Vec<String> = storage
        .named_vectors
        .cores_read()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    open.sort();
    open
}

/// The space `dense` with the tail `dense` and nothing else: before a seal.
fn first_tail_alone() -> DenseVectorSpaceMetadata {
    DenseVectorSpaceMetadata {
        next_segment_id: 1,
        segments: vec![DenseVectorSegmentMetadata {
            physical_name: "dense".into(),
            role: DenseVectorSegmentRole::Mutable,
        }],
    }
}

/// The same once the first tail is sealed and not yet built.
fn first_tail_sealed() -> DenseVectorSpaceMetadata {
    DenseVectorSpaceMetadata {
        next_segment_id: 2,
        segments: vec![
            DenseVectorSegmentMetadata {
                physical_name: "dense".into(),
                role: DenseVectorSegmentRole::Building,
            },
            DenseVectorSegmentMetadata {
                physical_name: SECOND_TAIL.into(),
                role: DenseVectorSegmentRole::Mutable,
            },
        ],
    }
}

/// What a writer or a reader (as `config` says) started on `left` serves of
/// the two points, and for a failure's message where it found them and the
/// layout it found stored.
fn served_by_one_started_on(
    config: &Config,
    left: InMemory,
) -> ([(Option<Vec<f32>>, bool); 2], String) {
    let _left = set_lsm_test_object_store(Arc::new(left));
    let dir = TempDir::new().unwrap();
    let started =
        Arc::new(CollectionManager::new(dir.path().to_path_buf(), config.clone()).unwrap());
    let storage = started.get_collection(NAME).unwrap();
    wait_for_index_executor(NAME);
    (
        served(&storage),
        format!(
            "point {NEW} is held by {:?}, point {WRITTEN_AGAIN} by {:?}, stored layout {:?}",
            segments_holding(&storage, NEW),
            segments_holding(&storage, WRITTEN_AGAIN),
            stored(&storage)
        ),
    )
}

/// `served_by_one_started_on` for the next writer and for a reader, each on
/// a copy of `left` of its own: both serve what the writer that took the
/// writes served, `served_live`.
fn assert_served_by_the_next_writer_and_a_reader(
    left: &InMemory,
    when: &str,
    served_live: &[(Option<Vec<f32>>, bool); 2],
) {
    for (config, who) in [(writer_config(), "writer"), (reader_config(), "reader")] {
        let (served_after, found) = served_by_one_started_on(&config, left.fork());
        assert_eq!(
            &served_after, served_live,
            "a {who} started on the store as it was {when}: {found}"
        );
    }
}

/// Runs `beside` in another thread from the middle of `stage` on this thread
/// and gives it two seconds: it runs to its end there, or it is held off for
/// as long as this thread holds what it holds at `stage` and ends after.
/// Returns what `beside` returned once `run` is over.
fn with_another_thread_at<T: Send + 'static>(
    stage: &'static str,
    before: impl FnOnce() + 'static,
    beside: impl FnOnce() -> T + Send + 'static,
    run: impl FnOnce(),
) -> T {
    let started: Arc<Mutex<Option<thread::JoinHandle<T>>>> = Arc::new(Mutex::new(None));
    let keep = Arc::clone(&started);
    test_support::with_hook(
        stage,
        move || {
            before();
            let handle = thread::spawn(beside);
            let deadline = Instant::now() + Duration::from_secs(2);
            while !handle.is_finished() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(5));
            }
            *keep.lock().unwrap() = Some(handle);
        },
        run,
    );
    let handle = started
        .lock()
        .unwrap()
        .take()
        .unwrap_or_else(|| panic!("nothing reached {stage}"));
    handle.join().unwrap()
}

/// An upsert a client was told succeeded is served by the writer that starts
/// on what the dead one left, and by a reader, whenever the dead one died
/// after it. `stage` is where in the seal the upsert is applied.
///
/// A seal opens the next tail, stores the layout that names it and gives it
/// to memory, where an upsert applied beside the seal takes its tail from.
/// With the next tail in memory before the layout is stored, the upsert's
/// rows are under a name no stored layout has, and the older row of a point
/// written again is deleted from the sealed tail by the same batch. With the
/// layout stored first, the upsert's rows go to the old tail wherever in
/// the seal it is applied.
fn an_upsert_acknowledged_inside_a_seal_is_served_from_what_was_left(
    prefix: &'static str,
    stage: &'static str,
) {
    let _settings = settings(prefix);
    let store = Arc::new(WalSnapshots::default());
    let _store = set_lsm_test_object_store(Arc::clone(&store) as Arc<dyn ObjectStore>);
    let dir = TempDir::new().unwrap();
    let (writer, config, storage) = writer_with_a_tail(dir.path());
    let first_write = first_write(&storage);

    // The store is kept as it is after each WAL object from the upsert's on:
    // the first of them is what a writer that died right after it had
    // acknowledged the upsert left.
    let arming = Arc::clone(&store);
    let (upserting, upsert_config) = (Arc::clone(&writer), config.clone());
    let sealing = Arc::clone(&storage);
    with_another_thread_at(
        stage,
        move || arming.armed.store(true, Ordering::Release),
        move || upsert(&upserting, &upsert_config, &[WRITTEN_AGAIN, NEW], 1.0),
        move || seal(&sealing),
    );
    storage.backend.flush_durable().unwrap();
    store.armed.store(false, Ordering::Release);
    let new_is_in = segments_holding(&storage, NEW);
    let served_live = served_whole(&storage, &first_write, "the writer that sealed");
    wait_for_index_executor(NAME);
    assert_eq!(stored(&storage), in_memory(&storage));
    let mut left: Vec<InMemory> = std::mem::take(&mut *store.snapshots.lock().unwrap());
    assert!(
        !left.is_empty(),
        "an acknowledged upsert is in a WAL object"
    );
    // And as a writer that was shut down after it all left it.
    left.push(store.inner.fork());
    drop(storage);
    drop(writer);

    let writers = left.len();
    for (at, left) in left.iter().enumerate() {
        assert_served_by_the_next_writer_and_a_reader(
            left,
            &format!("{at} WAL objects after the upsert's, of {writers}"),
            &served_live,
        );
    }
    // Wherever in the seal it was applied, the upsert took the old tail.
    assert_eq!(new_is_in, ["dense"]);
}

/// The upsert is applied once the seal has stored its layout and before it
/// has given memory the next tail: its rows go to the tail the stored layout
/// already shows sealed. A reader on that layout reads the tail for as long
/// as it is marked, the rows written after the store with it, and the next
/// writer builds it with them.
#[test]
fn an_upsert_acknowledged_in_the_middle_of_a_seal_is_served_by_the_next_writer() {
    let _guard = ENV_LOCK.lock().unwrap();
    an_upsert_acknowledged_inside_a_seal_is_served_from_what_was_left(
        "seal_store_order_midway",
        "seal_midway",
    );
}

/// The upsert is applied once the seal has opened its next tail and before
/// it has stored anything: the first of the stores left behind has the
/// upsert and the layout from before the seal.
#[test]
fn an_upsert_acknowledged_before_a_seals_store_is_served_by_the_next_writer() {
    let _guard = ENV_LOCK.lock().unwrap();
    an_upsert_acknowledged_inside_a_seal_is_served_from_what_was_left(
        "seal_store_order_before_store",
        "seal_prepared",
    );
}

/// The control: the seal is over before the upsert is applied, and a writer
/// started on what the dead one left serves both points.
#[test]
fn an_upsert_acknowledged_after_a_seal_is_served_by_the_next_writer() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _settings = settings("seal_store_order_control");
    let store = Arc::new(WalSnapshots::default());
    let _store = set_lsm_test_object_store(Arc::clone(&store) as Arc<dyn ObjectStore>);
    let dir = TempDir::new().unwrap();
    let (writer, config, storage) = writer_with_a_tail(dir.path());
    let first_write = first_write(&storage);
    seal(&storage);
    assert_eq!(in_memory(&storage), first_tail_sealed());
    assert_eq!(stored(&storage), first_tail_sealed());

    store.armed.store(true, Ordering::Release);
    upsert(&writer, &config, &[WRITTEN_AGAIN, NEW], 1.0);
    store.armed.store(false, Ordering::Release);
    let left = store
        .snapshots
        .lock()
        .unwrap()
        .drain(..)
        .next()
        .expect("an acknowledged upsert is in a WAL object");
    let served_live = served_whole(&storage, &first_write, "the writer that sealed");
    assert_eq!(segments_holding(&storage, NEW), [SECOND_TAIL]);
    wait_for_index_executor(NAME);
    drop(storage);
    drop(writer);

    assert_served_by_the_next_writer_and_a_reader(&left, "right after the upsert", &served_live);
}

/// A seal whose store fails leaves memory as it was: the space has its tail
/// and its next id, no core is open for a next tail, and the upserts that go
/// on are served by the next writer. The seal after it takes the id the
/// failed one opened and did not spend.
///
/// The store fails here before its transaction begins, on the flag that
/// refuses a write to a degraded collection, taken back afterwards as a
/// store that failed with an error that is not fatal leaves the collection
/// usable. The tail has its marker already, so the layout is the first thing
/// the seal writes.
#[test]
fn a_seal_whose_store_fails_leaves_memory_as_it_was() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _settings = settings("seal_store_order_store_fails");
    let store = Arc::new(WalSnapshots::default());
    let _store = set_lsm_test_object_store(Arc::clone(&store) as Arc<dyn ObjectStore>);
    let dir = TempDir::new().unwrap();
    let (writer, config, storage) = writer_with_a_tail(dir.path());
    let first_write = first_write(&storage);
    storage
        .with_write_backend(|w| {
            storage
                .named_vectors
                .mark_sealed_tail_be(w, "dense")
                .map_err(GraphError::from)
        })
        .unwrap();
    let cores_before = cores_open(&storage);

    storage.collection_degraded.store(true, Ordering::Release);
    let failed = seal_lsm_dense_tail(&storage, "dense", &hnsw());
    storage.collection_degraded.store(false, Ordering::Release);
    assert!(failed.is_err(), "the seal's store went through");
    assert_eq!(
        (in_memory(&storage), cores_open(&storage)),
        (first_tail_alone(), cores_before),
        "a seal whose store failed changed memory"
    );
    assert_eq!(stored(&storage), first_tail_alone());

    store.armed.store(true, Ordering::Release);
    upsert(&writer, &config, &[WRITTEN_AGAIN, NEW], 1.0);
    store.armed.store(false, Ordering::Release);
    let left = store
        .snapshots
        .lock()
        .unwrap()
        .drain(..)
        .next()
        .expect("an acknowledged upsert is in a WAL object");
    let served_live = served_whole(&storage, &first_write, "the writer whose seal failed");

    // The upsert's own index job is over: nothing else changes the layout.
    wait_for_index_executor(NAME);
    seal(&storage);
    assert_eq!(in_memory(&storage), first_tail_sealed());
    assert_eq!(stored(&storage), first_tail_sealed());
    wait_for_index_executor(NAME);
    drop(storage);
    drop(writer);
    assert_served_by_the_next_writer_and_a_reader(&left, "right after the upsert", &served_live);
}

/// The other writers of the metadata row a layout is stored in (a payload
/// index, a sparse space, a threshold) make the store's transaction run
/// again. One that commits while the seal's store runs: the store's second
/// run stores the seal on top of what that writer stored, and the seal has
/// spent one id.
#[test]
fn a_seal_whose_store_runs_twice_stores_the_seal_once() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _settings = settings("seal_store_order_store_twice");
    let _store = set_lsm_test_object_store(Arc::new(InMemory::new()));
    let dir = TempDir::new().unwrap();
    let (_writer, _config, storage) = writer_with_a_tail(dir.path());

    let (sealing, other) = (Arc::clone(&storage), Arc::clone(&storage));
    with_another_thread_at(
        "layout_read",
        || {},
        move || set_indexing_threshold(&other, 7),
        move || seal(&sealing),
    );

    assert_eq!(in_memory(&storage), first_tail_sealed());
    assert_eq!(stored(&storage), first_tail_sealed());
    let read = storage.backend.begin_read().unwrap();
    assert_eq!(
        storage
            .get_metadata_be(&read)
            .unwrap()
            .indexing_threshold_override,
        Some(7),
        "the seal's store ran once, on the row as it was before the other writer"
    );
}

/// A writer of the metadata row that is no store of the layout.
pub(super) fn set_indexing_threshold(storage: &HelixGraphStorage, threshold: usize) {
    storage
        .with_write_backend(|w| {
            storage
                .update_metadata_be(w, |metadata| {
                    metadata.set_indexing_threshold_override(Some(threshold));
                    Ok(())
                })
                .map(|_| ())
        })
        .unwrap();
}

/// A store that meets another writer of the metadata row on every one of
/// its runs gives up, and the seal with it. Its closure ran as often as the
/// transaction was tried, and none of those runs spent an id, opened a core
/// into memory or moved the tail: the seal after it is the first.
#[test]
fn a_seal_whose_store_gives_up_after_its_runs_leaves_memory_as_it_was() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _settings = settings("seal_store_order_store_gives_up");
    let _store = set_lsm_test_object_store(Arc::new(InMemory::new()));
    let dir = TempDir::new().unwrap();
    let (_writer, _config, storage) = writer_with_a_tail(dir.path());
    let cores_before = cores_open(&storage);

    let runs = Arc::new(AtomicUsize::new(0));
    let (counted, other) = (Arc::clone(&runs), Arc::clone(&storage));
    let mut failed = None;
    test_support::with_hook_each_time(
        "layout_read",
        move || {
            let run = counted.fetch_add(1, Ordering::AcqRel) + 1;
            let other = Arc::clone(&other);
            thread::spawn(move || set_indexing_threshold(&other, run))
                .join()
                .unwrap();
        },
        || failed = Some(seal_lsm_dense_tail(&storage, "dense", &hnsw())),
    );

    assert!(failed.as_ref().is_some_and(Result::is_err), "{failed:?}");
    // `LSM_METADATA_TXN_MAX_ATTEMPTS`.
    assert_eq!(runs.load(Ordering::Acquire), 8);
    assert_eq!(
        (in_memory(&storage), cores_open(&storage)),
        (first_tail_alone(), cores_before),
        "a seal whose store gave up changed memory"
    );
    assert_eq!(stored(&storage), first_tail_alone());

    seal(&storage);
    assert_eq!(in_memory(&storage), first_tail_sealed());
    assert_eq!(stored(&storage), first_tail_sealed());
}

/// A seal that dies between its store and its change of memory (a panic on
/// the index thread, whose worker catches it and goes on to its next job)
/// leaves a stored layout that is ahead of memory: the tail it shows sealed
/// still takes the appends. They
/// are served by the next writer and by a reader, which find the tail
/// marked and build it; and by this writer, whose next seal stores the same
/// layout, takes the same id and gives it to memory. The change it held is
/// taken over from the dead holder.
#[test]
fn a_seal_that_dies_between_its_store_and_memory_loses_nothing() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _settings = settings("seal_store_order_dies_midway");
    let store = Arc::new(WalSnapshots::default());
    let _store = set_lsm_test_object_store(Arc::clone(&store) as Arc<dyn ObjectStore>);
    let dir = TempDir::new().unwrap();
    let (writer, config, storage) = writer_with_a_tail(dir.path());
    let first_write = first_write(&storage);
    let cores_before = cores_open(&storage);

    let died = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        test_support::with_hook(
            "seal_midway",
            || panic!("the seal died between its store and memory"),
            || {
                let _ = seal_lsm_dense_tail(&storage, "dense", &hnsw());
            },
        )
    }));
    assert!(died.is_err(), "the seal did not reach its midway");
    assert_eq!(stored(&storage), first_tail_sealed());
    assert_eq!(
        (in_memory(&storage), cores_open(&storage)),
        (first_tail_alone(), cores_before)
    );

    store.armed.store(true, Ordering::Release);
    upsert(&writer, &config, &[WRITTEN_AGAIN, NEW], 1.0);
    store.armed.store(false, Ordering::Release);
    let left = store
        .snapshots
        .lock()
        .unwrap()
        .drain(..)
        .next()
        .expect("an acknowledged upsert is in a WAL object");
    assert_eq!(segments_holding(&storage, NEW), ["dense"]);
    let served_live = served_whole(&storage, &first_write, "the writer whose seal died");

    // The upsert's own index job is over: nothing else changes the layout.
    wait_for_index_executor(NAME);
    seal(&storage);
    assert_eq!(in_memory(&storage), first_tail_sealed());
    assert_eq!(stored(&storage), first_tail_sealed());
    assert_eq!(
        served_whole(&storage, &first_write, "the writer once it sealed"),
        served_live
    );
    wait_for_index_executor(NAME);
    drop(storage);
    drop(writer);
    assert_served_by_the_next_writer_and_a_reader(&left, "right after the upsert", &served_live);
}

/// Runs `run` with `beside` run in another thread from its `stage`: two
/// changes of the layout beside each other. Returns the space `dense` as a
/// replica then finds it stored and as the writer's memory had it. What a
/// collection has stored names every segment its memory has once its
/// writers are done.
fn stored_and_in_memory_after(
    prefix: &'static str,
    stage: &'static str,
    beside: impl FnOnce(&Arc<CollectionManager>, &Config, &Arc<HelixGraphStorage>) + Send + 'static,
    run: impl FnOnce(&Arc<CollectionManager>, &Config, &Arc<HelixGraphStorage>),
) -> (DenseVectorSpaceMetadata, DenseVectorSpaceMetadata) {
    let _settings = settings(prefix);
    let store = Arc::new(WalSnapshots::default());
    let _store = set_lsm_test_object_store(Arc::clone(&store) as Arc<dyn ObjectStore>);
    let dir = TempDir::new().unwrap();
    let (writer, config, storage) = writer_with_a_tail(dir.path());

    let (beside_writer, beside_config, beside_storage) =
        (Arc::clone(&writer), config.clone(), Arc::clone(&storage));
    with_another_thread_at(
        stage,
        || {},
        move || beside(&beside_writer, &beside_config, &beside_storage),
        || run(&writer, &config, &storage),
    );
    let in_memory = in_memory(&storage);
    storage.backend.flush_durable().unwrap();
    let left = store.inner.fork();
    drop(storage);
    drop(writer);

    // A replica stores nothing of its own.
    let _left = set_lsm_test_object_store(Arc::new(left));
    let dir = TempDir::new().unwrap();
    let reader = CollectionManager::new(dir.path().to_path_buf(), reader_config()).unwrap();
    let replica = reader.get_collection(NAME).unwrap();
    (stored(&replica), in_memory)
}

/// PATCH of the collection with a dense space it does not have.
fn add_a_space_by_update(
    writer: &Arc<CollectionManager>,
    config: &Config,
    _storage: &Arc<HelixGraphStorage>,
) {
    apply_mutation(
        writer,
        config,
        &ReplicatedMutation::UpdateCollection {
            name: NAME.into(),
            vectors: HashMap::from([("added".to_string(), raw_space())]),
            sparse_vectors: HashMap::new(),
        },
    )
    .unwrap();
}

/// A create that names a space the collection does not have, which is how a
/// release that adds a named vector reaches a collection.
fn add_a_space_by_create(
    writer: &Arc<CollectionManager>,
    config: &Config,
    _storage: &Arc<HelixGraphStorage>,
) {
    apply_mutation(
        writer,
        config,
        &ReplicatedMutation::CreateCollection {
            name: NAME.into(),
            vectors: HashMap::from([
                ("dense".to_string(), raw_space()),
                ("added".to_string(), raw_space()),
            ]),
            sparse_vectors: HashMap::new(),
            hnsw_overrides: None,
        },
    )
    .unwrap();
}

fn seal_the_tail(
    _writer: &Arc<CollectionManager>,
    _config: &Config,
    storage: &Arc<HelixGraphStorage>,
) {
    seal(storage);
}

/// An update that adds a dense space reads the layout to store it, and a
/// seal of another space's tail wants to run in between: the seal's layout
/// is the one left stored.
#[test]
fn an_update_that_adds_a_space_beside_a_seal_leaves_the_seals_layout_stored() {
    let _guard = ENV_LOCK.lock().unwrap();
    let (stored, in_memory) = stored_and_in_memory_after(
        "seal_store_order_update",
        "layout_read",
        seal_the_tail,
        add_a_space_by_update,
    );
    assert_eq!(in_memory, first_tail_sealed());
    assert_eq!(stored, in_memory);
}

/// The same for a create that names a space the collection does not have.
#[test]
fn a_create_that_adds_a_space_beside_a_seal_leaves_the_seals_layout_stored() {
    let _guard = ENV_LOCK.lock().unwrap();
    let (stored, in_memory) = stored_and_in_memory_after(
        "seal_store_order_create",
        "layout_read",
        seal_the_tail,
        add_a_space_by_create,
    );
    assert_eq!(in_memory, first_tail_sealed());
    assert_eq!(stored, in_memory);
}

/// The other way round: the seal has stored its layout and memory does not
/// have it yet, and an update that adds a space wants to store the layout
/// it finds in memory. It waits for the seal's change to end: stored in
/// between, its layout is the one without the next tail, and memory gets
/// that tail right after.
#[test]
fn an_update_that_adds_a_space_in_the_middle_of_a_seal_leaves_the_seals_layout_stored() {
    let _guard = ENV_LOCK.lock().unwrap();
    let (stored, in_memory) = stored_and_in_memory_after(
        "seal_store_order_update_midway",
        "seal_midway",
        add_a_space_by_update,
        seal_the_tail,
    );
    assert_eq!(in_memory, first_tail_sealed());
    assert_eq!(stored, in_memory);
}

/// The same for a create that names a space the collection does not have.
#[test]
fn a_create_that_adds_a_space_in_the_middle_of_a_seal_leaves_the_seals_layout_stored() {
    let _guard = ENV_LOCK.lock().unwrap();
    let (stored, in_memory) = stored_and_in_memory_after(
        "seal_store_order_create_midway",
        "seal_midway",
        add_a_space_by_create,
        seal_the_tail,
    );
    assert_eq!(in_memory, first_tail_sealed());
    assert_eq!(stored, in_memory);
}

/// A reader that has the layout a seal stored has the marker that keeps the
/// sealed tail searched: the marker is written, and durable, before the
/// layout that shows the tail `Building`. A reader started on the store as
/// it was after any WAL object of the seal finds a point of the tail.
#[test]
fn a_reader_started_at_any_point_of_a_seal_finds_the_sealed_tails_points() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _settings = settings("seal_store_order_marker_first");
    let store = Arc::new(WalSnapshots::default());
    let _store = set_lsm_test_object_store(Arc::clone(&store) as Arc<dyn ObjectStore>);
    let dir = TempDir::new().unwrap();
    let (writer, _config, storage) = writer_with_a_tail(dir.path());

    store.armed.store(true, Ordering::Release);
    seal(&storage);
    store.armed.store(false, Ordering::Release);
    let left: Vec<InMemory> = std::mem::take(&mut *store.snapshots.lock().unwrap());
    assert!(
        left.len() >= 2,
        "the marker and the layout are a WAL object each: {}",
        left.len()
    );
    drop(storage);
    drop(writer);

    for (at, left) in left.into_iter().enumerate() {
        let _left = set_lsm_test_object_store(Arc::new(left));
        let dir = TempDir::new().unwrap();
        let reader = CollectionManager::new(dir.path().to_path_buf(), reader_config()).unwrap();
        let replica = reader.get_collection(NAME).unwrap();
        assert!(
            found_by_a_search(&replica, WRITTEN_AGAIN, 0.0),
            "a reader started {at} WAL objects into the seal, stored layout {:?}",
            stored(&replica)
        );
    }
}

/// The store of the layout fails when the layout cannot be read from
/// memory, and leaves what is stored: read through the listing, which
/// answers a layout of no space then, it stored a collection with no
/// segment.
#[test]
fn a_store_of_a_layout_that_cannot_be_read_from_memory_fails() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _settings = settings("seal_store_order_unreadable");
    let _store = set_lsm_test_object_store(Arc::new(InMemory::new()));
    let dir = TempDir::new().unwrap();
    let (_writer, _config, storage) = writer_with_a_tail(dir.path());

    storage.named_vectors.poison_dense_spaces();
    assert!(storage.named_vectors.list_dense_vector_spaces().is_empty());
    let held = storage.named_vectors.begin_dense_layout_change();
    let failed = storage.with_write_backend(|w| store_dense_layout(&storage, &held, w, None));

    assert!(failed.is_err(), "a layout of no space was stored");
    assert_eq!(stored(&storage), first_tail_alone());
}

/// A layout that is stored with a space's segments gone, or with a space
/// that has segments and no tail to append to, is said so: neither is a
/// state a change of the layout should come to.
#[test]
fn a_store_of_a_space_without_its_segments_or_without_a_tail_is_reported() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _settings = settings("seal_store_order_faults");
    let _store = set_lsm_test_object_store(Arc::new(InMemory::new()));
    let dir = TempDir::new().unwrap();
    let (_writer, _config, storage) = writer_with_a_tail(dir.path());
    let faults = |stored: DenseVectorSpaceMetadata, next: Option<DenseVectorSpaceMetadata>| {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        metrics::with_local_recorder(&recorder, || {
            report_dense_layout_faults(
                &storage,
                &HashMap::from([("dense".to_string(), stored)]),
                &next
                    .map(|next| HashMap::from([("dense".to_string(), next)]))
                    .unwrap_or_default(),
            )
        });
        let rendered = recorder.handle().render();
        ["space_without_segments", "space_without_tail"].map(|fault| {
            rendered.contains("helix_dense_layout_store_fault_total")
                && rendered.contains(&format!("fault=\"{fault}\""))
        })
    };
    let no_segments = DenseVectorSpaceMetadata {
        next_segment_id: 2,
        segments: Vec::new(),
    };
    let mut no_tail = first_tail_sealed();
    no_tail.segments.pop();

    assert_eq!(
        faults(first_tail_alone(), Some(first_tail_sealed())),
        [false, false]
    );
    assert_eq!(
        faults(first_tail_sealed(), Some(no_segments.clone())),
        [true, false]
    );
    assert_eq!(faults(first_tail_sealed(), None), [true, false]);
    // A space that never had a segment stored and has none is no fault.
    assert_eq!(
        faults(no_segments.clone(), Some(no_segments)),
        [false, false]
    );
    assert_eq!(faults(first_tail_alone(), Some(no_tail)), [false, true]);
}
