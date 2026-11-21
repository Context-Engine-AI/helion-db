//! The dense segments that left a stored layout: the key each is recorded
//! under by the store that drops it, what a writer started afterwards knows
//! of them, what takes a key away again, and what an image that keeps no
//! such record does to them.

use super::publish_crash::WalSnapshots;
use super::seal_store_order::{raw_space, set_indexing_threshold};
use super::*;
use crate::helix_engine::storage_core::backend::KeyRange;
use crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store;
use crate::helix_engine::vector_core::named_vectors::{
    dead_segment_key, DenseVectorSegmentMetadata, DenseVectorSegmentRole, DenseVectorSpaceMetadata,
    DEAD_SEGMENT_KEY_PREFIX,
};
use object_store::{memory::InMemory, ObjectStore};
use std::collections::BTreeMap;

const NAME: &str = "dead_segment_names";
/// The collection's one dense space, and the name of its first segment. No
/// other test has a segment of this name or of the names after it, so the
/// turns the reaper takes for one of them can be made to fail by name.
const SPACE: &str = "deadname";
/// The tail a seal of the first tail opens, and the two after it.
const SECOND: &str = "deadname__seg_000001";
const THIRD: &str = "deadname__seg_000002";
const FOURTH: &str = "deadname__seg_000003";

/// The settings every test here runs under: one WAL object a commit, no
/// index job that seals a tail by itself, and the record of dead names kept.
fn settings(prefix: &'static str) -> [EnvVarGuard; 5] {
    [
        EnvVarGuard::set("HELIX_LSM_PREFIX", prefix),
        EnvVarGuard::unset("HELIX_LSM_GROUP_COMMIT"),
        EnvVarGuard::set("HELIX_INDEX_TAIL_QUIET_MS", "600000"),
        EnvVarGuard::set("HELIX_INDEX_TAIL_SEAL_MIN_VECTORS", "5000"),
        EnvVarGuard::unset("HELIX_DENSE_DEAD_NAMES"),
    ]
}

fn hnsw() -> HNSWConfig {
    HNSWConfig::new(Some(16), Some(64), Some(128))
}

fn writer_config() -> Config {
    let mut config = Config::new(16, 64, 128, 1).with_storage_backend(StorageBackendConfig::Lsm {
        storage: LsmStorage::ObjectStore,
        role: LsmRole::Writer,
    });
    config.vector_config.flat_scan_threshold = Some(0);
    config
}

fn reader_config() -> Config {
    Config::new(16, 64, 128, 1).with_storage_backend(StorageBackendConfig::Lsm {
        storage: LsmStorage::ObjectStore,
        role: LsmRole::Reader,
    })
}

/// The vector of point `id`.
fn row(id: u128) -> Vec<f32> {
    vec![1.0, 0.01 * id as f32, 0.5, 0.0]
}

/// A writer whose one space, of raw f32 rows as production's are, has
/// points 1 to 8 in an indexed segment (`deadname`) and an empty tail after
/// it (`deadname__seg_000001`), on the store the test has set. Returns the
/// eight points' vectors with it.
fn writer_with_a_built_segment(
    dir: &std::path::Path,
) -> (
    Arc<CollectionManager>,
    Config,
    Arc<HelixGraphStorage>,
    Vec<Vec<f32>>,
) {
    let config = writer_config();
    let writer = Arc::new(CollectionManager::new(dir.to_path_buf(), config.clone()).unwrap());
    apply_mutation(
        &writer,
        &config,
        &ReplicatedMutation::CreateCollection {
            name: NAME.into(),
            vectors: HashMap::from([(SPACE.to_string(), raw_space())]),
            sparse_vectors: HashMap::new(),
            hnsw_overrides: None,
        },
    )
    .unwrap();
    let ids: Vec<u128> = (1..=8).collect();
    upsert(&writer, &config, &ids);
    let storage = writer.get_collection(NAME).unwrap();
    seal(&storage);
    let rows = ids.iter().map(|id| row(*id)).collect();
    let prepared = {
        let read = storage.backend.begin_read().unwrap();
        storage
            .named_vectors
            .prepare_building_indices_limited_be(&read, SPACE, 1)
            .unwrap()
    };
    assert_eq!(prepared.len(), 1);
    assert!(run_chunked_indices_publish(&storage, SPACE, prepared).unwrap());
    wait_for_index_executor(NAME);
    assert_eq!(names(&stored(&storage)), [SPACE, SECOND]);
    assert_eq!(stored(&storage), in_memory(&storage));
    (writer, config, storage, rows)
}

/// Writes points `ids` into the space's tail.
fn upsert(writer: &Arc<CollectionManager>, config: &Config, ids: &[u128]) {
    apply_mutation(
        writer,
        config,
        &ReplicatedMutation::UpsertPoints {
            collection: NAME.into(),
            points: ids
                .iter()
                .map(|id| ReplicatedPoint {
                    id: *id,
                    vectors: HashMap::from([(SPACE.to_string(), row(*id))]),
                    sparse_vectors: HashMap::new(),
                    payload: HashMap::new(),
                })
                .collect(),
        },
    )
    .unwrap();
    wait_for_index_executor(NAME);
}

/// Seals the space's tail, which has rows: a store of the layout by a site
/// that takes no segment off it.
fn seal(storage: &Arc<HelixGraphStorage>) {
    assert!(seal_lsm_dense_tail(storage, SPACE, &hnsw()).unwrap());
}

/// The space as the collection's memory has it.
fn in_memory(storage: &HelixGraphStorage) -> DenseVectorSpaceMetadata {
    storage.named_vectors.list_dense_vector_spaces()[SPACE].clone()
}

/// The same as the store has it.
fn stored(storage: &HelixGraphStorage) -> DenseVectorSpaceMetadata {
    let read = storage.backend.begin_read().unwrap();
    storage.get_metadata_be(&read).unwrap().dense_vector_spaces[SPACE].clone()
}

/// The segments `space` lists, in its order.
fn names(space: &DenseVectorSpaceMetadata) -> Vec<String> {
    space
        .segments
        .iter()
        .map(|segment| segment.physical_name.clone())
        .collect()
}

/// The dead names the store has, each with its time: read from the metadata
/// table by the prefix, as an open reads them.
fn dead_stored(storage: &HelixGraphStorage) -> BTreeMap<String, u64> {
    let read = storage.backend.begin_read().unwrap();
    let mut dead = BTreeMap::new();
    storage
        .backend
        .scan(
            &read,
            Namespace::Metadata,
            KeyRange::prefix(DEAD_SEGMENT_KEY_PREFIX),
            |key, value| {
                let name = String::from_utf8(key[DEAD_SEGMENT_KEY_PREFIX.len()..].to_vec());
                dead.insert(name.unwrap(), u64::from_be_bytes(value.try_into().unwrap()));
                true
            },
        )
        .unwrap();
    dead
}

/// The same as the collection's memory has them: a writer keeps the record.
fn dead_in_memory(storage: &HelixGraphStorage) -> BTreeMap<String, u64> {
    storage
        .named_vectors
        .dead_segment_times()
        .expect("a writer keeps the record of dead names")
        .into_iter()
        .collect()
}

/// Puts a dead key for `segment` into the store, as a writer before this
/// one left it.
fn plant_dead_key(storage: &HelixGraphStorage, segment: &str, at: u64) {
    storage
        .with_write_backend(|w| {
            storage
                .backend
                .put(
                    w,
                    Namespace::Metadata,
                    &dead_segment_key(segment),
                    &at.to_be_bytes(),
                )
                .map_err(|e| GraphError::New(e.to_string()))
        })
        .unwrap();
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn wait_until(what: &str, done: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !done() {
        assert!(Instant::now() < deadline, "never: {what}");
        thread::sleep(Duration::from_millis(10));
    }
}

/// Makes every turn the reaper takes for `segment` fail, so it gives the
/// job up after its tries with no row of the segment cleared: what a writer
/// that stopped with the job still queued left, without the stop.
fn the_reaper_gives_up_on(segment: &str) {
    let turns = segment_reaper_retry_attempts() + 1;
    WORKER_PANICS_FOR_TEST
        .lock()
        .unwrap()
        .extend((0..turns).map(|_| segment.to_string()));
}

/// Waits until the reaper has taken its last turn for `segment`.
fn wait_for_the_reaper_to_give_up(segment: &str) {
    wait_until("the reaper took every turn it had for the segment", || {
        let left = WORKER_PANICS_FOR_TEST.lock().unwrap();
        !left.iter().any(|listed| listed == segment)
    });
    wait_for_the_reaper();
}

/// Waits until the reaper has nothing queued or running.
fn wait_for_the_reaper() {
    wait_until("the reaper is idle", || SEGMENT_REAPER.idle());
}

/// Runs `look` on the collection as a writer or a reader (as `config` says)
/// started on a copy of `left` has it. Returns what `look` did, and the copy
/// as the one that was started left it, for whoever starts after it.
fn started_on<T>(
    left: &InMemory,
    config: &Config,
    look: impl FnOnce(&Arc<CollectionManager>, &Arc<HelixGraphStorage>) -> T,
) -> (T, InMemory) {
    let store = Arc::new(left.fork());
    let _store = set_lsm_test_object_store(Arc::clone(&store) as Arc<dyn ObjectStore>);
    let dir = TempDir::new().unwrap();
    let started =
        Arc::new(CollectionManager::new(dir.path().to_path_buf(), config.clone()).unwrap());
    let storage = started.get_collection(NAME).unwrap();
    wait_for_index_executor(NAME);
    let looked = look(&started, &storage);
    wait_for_index_executor(NAME);
    (looked, store.fork())
}

/// A segment that leaves the layout is recorded dead by the store that drops
/// it: in whatever a writer left that died from then on, the layout lists
/// the segment or the segment has its key, and never neither. The writer
/// started on it knows the name as dead from its open; a reader opens the
/// same store and keeps no record. The name stays recorded when the reaper
/// has given its job up, with the segment's rows still there.
#[test]
fn a_segment_that_leaves_the_layout_is_recorded_dead_by_the_store_that_drops_it() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _settings = settings("dead_names_recorded");
    let store = Arc::new(WalSnapshots::default());
    let _store = set_lsm_test_object_store(Arc::clone(&store) as Arc<dyn ObjectStore>);
    let dir = TempDir::new().unwrap();
    let (writer, _config, storage, _rows) = writer_with_a_built_segment(dir.path());
    assert!(dead_stored(&storage).is_empty());
    assert!(dead_in_memory(&storage).is_empty());
    let rows_before = segment_dump(&storage, SPACE);
    assert!(!rows_before.is_empty());

    // As a writer that died before the retire left it, then after each WAL
    // object from the retire's on.
    let mut left = vec![store.inner.fork()];
    the_reaper_gives_up_on(SPACE);
    let before = now_ms();
    store.armed.store(true, Ordering::Release);
    retire_dense_segment(&storage, SPACE, SPACE).unwrap();
    wait_for_the_reaper_to_give_up(SPACE);
    store.armed.store(false, Ordering::Release);
    let after = now_ms();
    left.append(&mut store.snapshots.lock().unwrap());
    assert!(left.len() >= 2, "the retire's store is a WAL object");

    let dead = dead_stored(&storage);
    assert_eq!(dead.keys().collect::<Vec<_>>(), [SPACE]);
    assert!(
        (before..=after).contains(&dead[SPACE]),
        "{dead:?} is not from {before} to {after}"
    );
    assert_eq!(dead_in_memory(&storage), dead);
    assert_eq!(names(&stored(&storage)), [SECOND]);
    assert_eq!(
        segment_dump(&storage, SPACE),
        rows_before,
        "the reaper gave the job up and a row of the segment is gone"
    );
    drop(storage);
    drop(writer);

    let mut dropped = 0;
    for (at, left) in left.iter().enumerate() {
        let ((listed, key, known), _) = started_on(left, &writer_config(), |_, next| {
            (
                names(&stored(next)).iter().any(|name| name == SPACE),
                dead_stored(next).get(SPACE).copied(),
                dead_in_memory(next).get(SPACE).copied(),
            )
        });
        assert_ne!(
            listed,
            key.is_some(),
            "{at} WAL objects into the retire: listed {listed}, dead key {key:?}"
        );
        assert_eq!(known, key, "{at} WAL objects into the retire");
        dropped += usize::from(!listed);

        started_on(left, &reader_config(), |_, replica| {
            assert!(replica.named_vectors.dead_segment_times().is_none());
            assert_eq!(in_memory(replica), stored(replica));
        });
    }
    assert!(dropped > 0, "no store was kept from after the retire");
}

/// A site that takes a segment out of memory and whose own store then fails
/// leaves the segment in the stored layout, unknown to the reaper. The next
/// store that lands, by whichever site, takes it off the stored layout: that
/// store records it. A writer started where the first one failed finds the
/// segment listed and not dead, and records it when it retires it itself.
#[test]
fn a_segment_dropped_where_the_store_failed_is_recorded_by_the_next_store_that_lands() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _settings = settings("dead_names_failed_store");
    let store = Arc::new(InMemory::new());
    let _store = set_lsm_test_object_store(Arc::clone(&store) as Arc<dyn ObjectStore>);
    let dir = TempDir::new().unwrap();
    let (writer, config, storage, _rows) = writer_with_a_built_segment(dir.path());

    // The store fails before its transaction begins, on the flag that
    // refuses a write to a degraded collection, taken back afterwards.
    storage.collection_degraded.store(true, Ordering::Release);
    let failed = retire_dense_segment(&storage, SPACE, SPACE);
    storage.collection_degraded.store(false, Ordering::Release);
    assert!(failed.is_err(), "the retire's store went through");
    assert_eq!(names(&in_memory(&storage)), [SECOND]);
    assert_eq!(names(&stored(&storage)), [SPACE, SECOND]);
    assert!(dead_stored(&storage).is_empty());
    assert!(dead_in_memory(&storage).is_empty());
    let left_there = store.fork();

    // Another site stores the layout: a seal of the tail.
    upsert(&writer, &config, &[21, 22]);
    seal(&storage);
    assert_eq!(names(&stored(&storage)), [SECOND, THIRD]);
    let dead = dead_stored(&storage);
    assert_eq!(dead.keys().collect::<Vec<_>>(), [SPACE]);
    assert_eq!(dead_in_memory(&storage), dead);
    wait_for_index_executor(NAME);
    drop(storage);
    drop(writer);

    started_on(&left_there, &writer_config(), |_, next| {
        assert_eq!(names(&stored(next)), [SPACE, SECOND]);
        assert_eq!(in_memory(next), stored(next));
        assert!(dead_stored(next).is_empty());
        the_reaper_gives_up_on(SPACE);
        retire_dense_segment(next, SPACE, SPACE).unwrap();
        wait_for_the_reaper_to_give_up(SPACE);
        let dead = dead_stored(next);
        assert_eq!(dead.keys().collect::<Vec<_>>(), [SPACE]);
        assert_eq!(dead_in_memory(next), dead);
    });
}

/// A dead name is not handed out again. An allocation passes over an id
/// whose name is dead, where the stored layout was ahead of memory; and an
/// open lifts a space's next id above every dead name the store has for it,
/// so the first id a writer takes after it is a new one.
#[test]
fn a_dead_name_is_never_handed_out_again() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _settings = settings("dead_names_refused");
    let store = Arc::new(InMemory::new());
    let _store = set_lsm_test_object_store(Arc::clone(&store) as Arc<dyn ObjectStore>);
    let dir = TempDir::new().unwrap();
    let (writer, config, storage, _rows) = writer_with_a_built_segment(dir.path());
    assert_eq!(in_memory(&storage).next_segment_id, 2);

    // The id the next seal would take is a dead name's.
    storage
        .named_vectors
        .dead_segments_stored(&[(THIRD.to_string(), 1)], &[]);
    upsert(&writer, &config, &[21, 22]);
    seal(&storage);
    assert_eq!(names(&in_memory(&storage)), [SPACE, SECOND, FOURTH]);
    assert_eq!(in_memory(&storage).next_segment_id, 4);
    assert_eq!(stored(&storage), in_memory(&storage));

    plant_dead_key(&storage, "deadname__seg_000009", 7);
    wait_for_index_executor(NAME);
    drop(storage);
    drop(writer);

    started_on(&store, &writer_config(), |next_writer, next| {
        assert_eq!(
            dead_in_memory(next),
            BTreeMap::from([("deadname__seg_000009".to_string(), 7)])
        );
        assert_eq!(in_memory(next).next_segment_id, 10);
        upsert(next_writer, &config, &[31, 32]);
        seal(next);
        assert_eq!(
            names(&in_memory(next)),
            [SPACE, SECOND, FOURTH, "deadname__seg_000010"]
        );
        assert_eq!(stored(next), in_memory(next));
        assert_eq!(stored(next).next_segment_id, 11);
    });
}

/// A segment the layout lists is served, whatever a key says: a writer
/// started on a store that has a dead key for a listed segment serves its
/// points, and its next store of the layout deletes the key.
#[test]
fn a_listed_segment_with_a_dead_key_is_served_and_loses_the_key_at_the_next_store() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _settings = settings("dead_names_listed_again");
    let store = Arc::new(InMemory::new());
    let _store = set_lsm_test_object_store(Arc::clone(&store) as Arc<dyn ObjectStore>);
    let dir = TempDir::new().unwrap();
    let (writer, config, storage, rows) = writer_with_a_built_segment(dir.path());
    plant_dead_key(&storage, SPACE, 7);
    wait_for_index_executor(NAME);
    drop(storage);
    drop(writer);

    started_on(&store, &writer_config(), |next_writer, next| {
        assert!(next.named_vectors.is_dead_segment(SPACE));
        assert_eq!(names(&in_memory(next)), [SPACE, SECOND]);
        let found = |storage: &HelixGraphStorage| {
            let read = storage.backend.begin_read().unwrap();
            storage
                .named_vectors
                .dense_search_with_id_filter_ef_be::<fn(u128) -> bool>(
                    &read,
                    SPACE,
                    &rows[0],
                    10,
                    None,
                    false,
                    None,
                    Some(128),
                )
                .unwrap()
                .iter()
                .any(|hit| hit.get_id() == 1)
        };
        assert!(
            found(next.as_ref()),
            "a point of the listed segment is not served"
        );

        upsert(next_writer, &config, &[21, 22]);
        seal(next);
        assert_eq!(names(&stored(next)), [SPACE, SECOND, THIRD]);
        assert!(dead_stored(next).is_empty(), "{:?}", dead_stored(next));
        assert!(dead_in_memory(next).is_empty());
        assert!(found(next.as_ref()));
    });
}

/// A store that takes every stored segment of a space off at once records
/// none of them: no change of a healthy space does that, and keys for them
/// would call every row the space ever had dead.
#[test]
fn a_store_that_takes_every_segment_of_a_space_off_records_none_of_them() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _settings = settings("dead_names_withheld");
    let _store = set_lsm_test_object_store(Arc::new(InMemory::new()));
    let dir = TempDir::new().unwrap();
    let (_writer, _config, storage, _rows) = writer_with_a_built_segment(dir.path());
    /// A layout whose one space lists `names`.
    fn layout(names: &[&str]) -> HashMap<String, DenseVectorSpaceMetadata> {
        let space = DenseVectorSpaceMetadata {
            next_segment_id: 9,
            segments: names
                .iter()
                .map(|name| DenseVectorSegmentMetadata {
                    physical_name: name.to_string(),
                    role: DenseVectorSegmentRole::Indexed,
                })
                .collect(),
        };
        HashMap::from([(SPACE.to_string(), space)])
    }
    /// The names a store of `next` over `stored` records, and how many it
    /// does not because their space lost them all.
    fn dead(
        storage: &HelixGraphStorage,
        stored: HashMap<String, DenseVectorSpaceMetadata>,
        next: HashMap<String, DenseVectorSpaceMetadata>,
    ) -> (Vec<String>, usize) {
        let dead = DeadNamesOfAStore::of(storage, &stored, &next)
            .expect("a writer keeps the record of dead names");
        let recorded = dead.recorded.into_iter().map(|(name, _)| name).collect();
        (recorded, dead.withheld)
    }
    let none: Vec<String> = Vec::new();

    assert_eq!(
        dead(&storage, layout(&["a", "b", "t"]), layout(&["m", "t"])),
        (vec!["a".to_string(), "b".to_string()], 0)
    );
    assert_eq!(
        dead(&storage, layout(&["a", "t"]), layout(&["a", "t", "u"])),
        (none.clone(), 0)
    );
    assert_eq!(
        dead(&storage, layout(&["a", "t"]), layout(&[])),
        (none.clone(), 2)
    );
    assert_eq!(
        dead(&storage, layout(&["a", "t"]), HashMap::new()),
        (none.clone(), 2)
    );
    assert_eq!(
        dead(&storage, layout(&["a", "t"]), layout(&["m"])),
        (none.clone(), 2)
    );
    assert_eq!(dead(&storage, layout(&[]), layout(&["t"])), (none, 0));

    // And through the store itself.
    let held = storage.named_vectors.begin_dense_layout_change();
    storage
        .named_vectors
        .apply_dense_segment_removal(SPACE, &[SPACE.to_string(), SECOND.to_string()])
        .unwrap();
    storage
        .with_write_backend(|w| store_dense_layout(&storage, &held, w, None))
        .unwrap();
    drop(held);
    assert!(stored(&storage).segments.is_empty());
    assert!(dead_stored(&storage).is_empty());
    assert!(dead_in_memory(&storage).is_empty());
}

/// A dead name leaves the record once the reaper has read each of its tables
/// empty, in the store and in memory. A segment with no core open counts as
/// drained without a read of any table: its key stays.
#[test]
fn a_dead_name_leaves_the_record_only_once_its_tables_were_read_empty() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _settings = settings("dead_names_read_empty");
    let _store = set_lsm_test_object_store(Arc::new(InMemory::new()));
    let dir = TempDir::new().unwrap();
    let (_writer, _config, storage, _rows) = writer_with_a_built_segment(dir.path());

    retire_dense_segment(&storage, SPACE, SPACE).unwrap();
    wait_for_the_reaper();
    assert!(segment_dump(&storage, SPACE).is_empty());
    assert!(
        dead_stored(&storage).is_empty(),
        "{:?}",
        dead_stored(&storage)
    );
    assert!(dead_in_memory(&storage).is_empty());

    // A dead name a writer before this one left: no core is open for it.
    let unopened = "deadname__seg_000009";
    plant_dead_key(&storage, unopened, 7);
    storage
        .named_vectors
        .dead_segments_stored(&[(unopened.to_string(), 7)], &[]);
    SEGMENT_REAPER.submit(Arc::clone(&storage), unopened.to_string());
    wait_for_the_reaper();
    let dead = dead_stored(&storage);
    assert_eq!(dead, BTreeMap::from([(unopened.to_string(), 7)]));
    assert_eq!(dead_in_memory(&storage), dead);
}

/// A name stays recorded while the reaper cannot write. A collection marked
/// degraded refuses each turn of the job, the reaper drops the job after its
/// tries, and the key and every row of the segment are still there. The job
/// submitted again once the collection takes writes clears the rows, and
/// then the key.
#[test]
fn a_dead_name_stays_recorded_while_the_reaper_cannot_write() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _settings = settings("dead_names_reaper_refused");
    let _store = set_lsm_test_object_store(Arc::new(InMemory::new()));
    let dir = TempDir::new().unwrap();
    let (_writer, _config, storage, _rows) = writer_with_a_built_segment(dir.path());
    let rows_before = segment_dump(&storage, SPACE);
    assert!(!rows_before.is_empty());

    // The retire's own job gives up before it clears a row, so the job whose
    // turns are refused is the one submitted below.
    the_reaper_gives_up_on(SPACE);
    retire_dense_segment(&storage, SPACE, SPACE).unwrap();
    wait_for_the_reaper_to_give_up(SPACE);
    let dead = dead_stored(&storage);
    assert_eq!(dead.keys().collect::<Vec<_>>(), [SPACE]);

    storage.collection_degraded.store(true, Ordering::Release);
    SEGMENT_REAPER.submit(Arc::clone(&storage), SPACE.to_string());
    wait_for_the_reaper();
    storage.collection_degraded.store(false, Ordering::Release);
    assert_eq!(dead_stored(&storage), dead);
    assert_eq!(dead_in_memory(&storage), dead);
    assert_eq!(
        segment_dump(&storage, SPACE),
        rows_before,
        "a turn that was refused cleared a row"
    );

    SEGMENT_REAPER.submit(Arc::clone(&storage), SPACE.to_string());
    wait_for_the_reaper();
    assert!(segment_dump(&storage, SPACE).is_empty());
    assert!(
        dead_stored(&storage).is_empty(),
        "{:?}",
        dead_stored(&storage)
    );
    assert!(dead_in_memory(&storage).is_empty());
}

/// A writer that keeps no record (`HELIX_DENSE_DEAD_NAMES=0`, and so an
/// image from before the record) opens a store that has dead keys, writes to
/// it, takes a segment off its layout, and leaves every key as it found it.
/// The writer that keeps the record again knows the names the store has, and
/// not the one that left the layout meanwhile.
#[test]
fn a_writer_that_keeps_no_record_leaves_the_dead_keys_as_it_found_them() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _settings = settings("dead_names_switched_off");
    let store = Arc::new(InMemory::new());
    let _store = set_lsm_test_object_store(Arc::clone(&store) as Arc<dyn ObjectStore>);
    let dir = TempDir::new().unwrap();
    let (writer, config, storage, _rows) = writer_with_a_built_segment(dir.path());
    the_reaper_gives_up_on(SPACE);
    retire_dense_segment(&storage, SPACE, SPACE).unwrap();
    wait_for_the_reaper_to_give_up(SPACE);
    let dead = dead_stored(&storage);
    assert_eq!(dead.keys().collect::<Vec<_>>(), [SPACE]);
    wait_for_index_executor(NAME);
    drop(storage);
    drop(writer);

    let left_by_the_old = {
        let _off = EnvVarGuard::set("HELIX_DENSE_DEAD_NAMES", "0");
        started_on(&store, &writer_config(), |old_writer, old| {
            assert!(old.named_vectors.dead_segment_times().is_none());
            upsert(old_writer, &config, &[21, 22]);
            seal(old);
            assert_eq!(names(&stored(old)), [SECOND, THIRD]);
            the_reaper_gives_up_on(SECOND);
            retire_dense_segment(old, SPACE, SECOND).unwrap();
            wait_for_the_reaper_to_give_up(SECOND);
            assert_eq!(names(&stored(old)), [THIRD]);
            assert_eq!(dead_stored(old), dead);
            assert!(old.named_vectors.dead_segment_times().is_none());
        })
        .1
    };

    started_on(&left_by_the_old, &writer_config(), |_, next| {
        assert_eq!(dead_in_memory(next), dead);
        assert_eq!(dead_stored(next), dead);
        assert_eq!(names(&stored(next)), [THIRD]);
    });
}

/// The other writers of the metadata row make the store's transaction run
/// again. One that commits while a retire's store runs: the second run
/// records the segment, on top of what that writer stored.
#[test]
fn a_store_that_runs_twice_records_the_segment_with_the_run_that_lands() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _settings = settings("dead_names_store_twice");
    let _store = set_lsm_test_object_store(Arc::new(InMemory::new()));
    let dir = TempDir::new().unwrap();
    let (_writer, _config, storage, _rows) = writer_with_a_built_segment(dir.path());

    the_reaper_gives_up_on(SPACE);
    let other = Arc::clone(&storage);
    test_support::with_hook(
        "layout_read",
        move || {
            thread::spawn(move || set_indexing_threshold(&other, 7))
                .join()
                .unwrap()
        },
        || retire_dense_segment(&storage, SPACE, SPACE).unwrap(),
    );
    wait_for_the_reaper_to_give_up(SPACE);

    assert_eq!(names(&stored(&storage)), [SECOND]);
    let dead = dead_stored(&storage);
    assert_eq!(dead.keys().collect::<Vec<_>>(), [SPACE]);
    assert_eq!(dead_in_memory(&storage), dead);
    let read = storage.backend.begin_read().unwrap();
    assert_eq!(
        storage
            .get_metadata_be(&read)
            .unwrap()
            .indexing_threshold_override,
        Some(7),
        "the retire's store ran once, on the row as it was before the other writer"
    );
}

/// A store that meets another writer of the metadata row on every one of its
/// runs gives up. Each run decided the segment dead and none committed: the
/// record in memory, which is all a later reader of it would go by, has
/// nothing of it, and the store has no key. The retire had taken the segment
/// out of memory first, so the next store that lands records it.
#[test]
fn a_store_that_gives_up_leaves_the_record_of_dead_names_as_it_was() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _settings = settings("dead_names_store_gives_up");
    let _store = set_lsm_test_object_store(Arc::new(InMemory::new()));
    let dir = TempDir::new().unwrap();
    let (writer, config, storage, _rows) = writer_with_a_built_segment(dir.path());

    let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
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
        || failed = Some(retire_dense_segment(&storage, SPACE, SPACE)),
    );

    assert!(failed.as_ref().is_some_and(Result::is_err), "{failed:?}");
    // `LSM_METADATA_TXN_MAX_ATTEMPTS`.
    assert_eq!(runs.load(Ordering::Acquire), 8);
    assert_eq!(names(&in_memory(&storage)), [SECOND]);
    assert_eq!(names(&stored(&storage)), [SPACE, SECOND]);
    assert!(dead_in_memory(&storage).is_empty());
    assert!(dead_stored(&storage).is_empty());

    upsert(&writer, &config, &[21, 22]);
    seal(&storage);
    let dead = dead_stored(&storage);
    assert_eq!(dead.keys().collect::<Vec<_>>(), [SPACE]);
    assert_eq!(dead_in_memory(&storage), dead);
}

/// A merge target that nothing published is in the stored layout from before
/// its first row. The writer started after the merge's writer stopped takes
/// it off the layout, and that store records it: the writer after that one
/// knows it as dead, though the one between stopped before it cleared a row.
#[test]
fn a_merge_target_nothing_published_is_recorded_when_the_next_writer_takes_it_off() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _settings = settings("dead_names_merge_target");
    let store = Arc::new(InMemory::new());
    let _store = set_lsm_test_object_store(Arc::clone(&store) as Arc<dyn ObjectStore>);
    let dir = TempDir::new().unwrap();
    let (writer, _config, storage, _rows) = writer_with_a_built_segment(dir.path());
    let target =
        write_merge_target(&storage, SPACE, &hnsw(), prepared_merge_of(&storage, SPACE)).unwrap();
    assert_eq!(names(&stored(&storage)), [SPACE, SECOND, target.as_str()]);
    assert!(dead_stored(&storage).is_empty());
    storage.backend.flush_durable().unwrap();
    wait_for_index_executor(NAME);
    drop(storage);
    drop(writer);

    the_reaper_gives_up_on(&target);
    let (dead, left_by_the_second) = started_on(&store, &writer_config(), |_, next| {
        test_support::optimize(next, NAME, SPACE);
        wait_for_the_reaper_to_give_up(&target);
        assert_eq!(names(&stored(next)), [SPACE, SECOND]);
        let dead = dead_stored(next);
        assert_eq!(dead.keys().collect::<Vec<_>>(), [&target]);
        assert_eq!(dead_in_memory(next), dead);
        assert!(!segment_dump(next, &target).is_empty());
        dead
    });

    started_on(&left_by_the_second, &writer_config(), |_, last| {
        assert_eq!(dead_in_memory(last), dead);
        assert!(last.named_vectors.is_dead_segment(&target));
        assert_eq!(names(&stored(last)), [SPACE, SECOND]);
    });
}
