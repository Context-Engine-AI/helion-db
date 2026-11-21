//! A regraph that takes a segment out because a rebuild would keep no row of
//! it: what a reader replica and a writer search see across it, the write
//! that stands it down, the rows it must not count as replaced, and a writer
//! reopened at any point of it.

use super::publish_crash::WalSnapshots;
use super::*;
use crate::helix_engine::storage_core::backend_any::{set_lsm_test_object_store, AnyRead};
use crate::helix_engine::vector_core::named_vectors::DenseVectorSegmentRole;
use object_store::{memory::InMemory, ObjectStore};

/// Rows enough in a segment for a regraph to take it.
const POINTS: usize = 520;

/// Points `1..` with `rows` as their vectors and the payload
/// `sealed_lsm_collection` writes, under file names that begin with `file`.
fn points(rows: &[Vec<f32>], file: &str) -> Vec<ReplicatedPoint> {
    rows.iter()
        .enumerate()
        .map(|(i, vector)| ReplicatedPoint {
            id: i as u128 + 1,
            vectors: HashMap::from([("dense".to_string(), vector.clone())]),
            sparse_vectors: HashMap::new(),
            payload: HashMap::from([(
                "path".to_string(),
                Value::String(format!("src/{file}{i}.rs")),
            )]),
        })
        .collect()
}

fn upsert(
    writer: &Arc<CollectionManager>,
    config: &Config,
    name: &str,
    points: Vec<ReplicatedPoint>,
) {
    let mutation = ReplicatedMutation::UpsertPoints {
        collection: name.into(),
        points,
    };
    apply_mutation(writer, config, &mutation).unwrap();
}

/// The Indexed segments of `dense`, in the order the metadata has them.
fn indexed(storage: &HelixGraphStorage) -> Vec<String> {
    dense_layout(storage)
        .into_iter()
        .filter(|(_, role)| *role == DenseVectorSegmentRole::Indexed)
        .map(|(name, _)| name)
        .collect()
}

/// An object-store writer whose `dense` holds 520 points written twice with
/// different payloads, each write sealed and built into a segment of its own.
/// The caller keeps `HELIX_LSM_REUPSERT_DELETE` at `0`, so the first rows
/// stay: every row of the first segment has its point's own copy in the
/// second, and a regraph keeps none of them. Returns the two segments, the
/// first write's first.
#[allow(clippy::type_complexity)]
fn written_twice(
    dir: &std::path::Path,
    name: &str,
) -> (
    Arc<CollectionManager>,
    Config,
    Arc<HelixGraphStorage>,
    Vec<Vec<f32>>,
    [String; 2],
) {
    let (writer, config, storage, rows) = indexed_lsm_collection(dir, name, 16, POINTS);
    let first = sole_indexed_segment(&storage);
    upsert(&writer, &config, name, points(&rows, "g"));
    test_support::seal_and_index(&storage, "dense");
    let second = indexed(&storage)
        .into_iter()
        .find(|segment| *segment != first)
        .expect("the second write's segment");
    storage.backend.flush_durable().unwrap();
    (writer, config, storage, rows, [first, second])
}

/// How many ids have a row in an Indexed segment, and how many of those have
/// more than one.
fn stored(storage: &HelixGraphStorage) -> (usize, usize) {
    let read = storage.backend.begin_read().unwrap();
    let cores = storage.named_vectors.cores_read().unwrap();
    let mut copies: HashMap<u128, usize> = HashMap::new();
    for segment in indexed(storage) {
        for (id, _, _) in cores[&segment].export_level_zero(&read).unwrap() {
            *copies.entry(id).or_default() += 1;
        }
    }
    let doubled = copies.values().filter(|count| **count > 1).count();
    (copies.len(), doubled)
}

/// The ids of the ten points nearest `query`, ascending.
fn nearest(storage: &HelixGraphStorage, read: &AnyRead<'_>, query: &[f32]) -> Vec<u128> {
    let mut ids: Vec<u128> = storage
        .named_vectors
        .dense_search_with_id_filter_ef_be::<fn(u128) -> bool>(
            read,
            "dense",
            query,
            10,
            None,
            false,
            None,
            Some(128),
        )
        .unwrap()
        .into_iter()
        .map(|hit| hit.get_id())
        .collect();
    ids.sort_unstable();
    ids
}

/// The points a search for their own vector does not find.
fn lost(storage: &HelixGraphStorage, rows: &[Vec<f32>]) -> Vec<u128> {
    lost_of_every(1, storage, rows)
}

/// `lost`, asked of one point in `nth`.
fn lost_of_every(nth: usize, storage: &HelixGraphStorage, rows: &[Vec<f32>]) -> Vec<u128> {
    let read = storage.backend.begin_read().unwrap();
    (1..=rows.len() as u128)
        .step_by(nth)
        .filter(|id| !nearest(storage, &read, &rows[*id as usize - 1]).contains(id))
        .collect()
}

/// Waits until the reaper has nothing queued or running.
fn wait_for_the_reaper() {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !SEGMENT_REAPER.idle() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(20));
    }
    assert!(SEGMENT_REAPER.idle(), "the reaper is still at work");
}

fn reader_config() -> Config {
    Config::new(16, 64, 128, 1).with_storage_backend(StorageBackendConfig::Lsm {
        storage: LsmStorage::ObjectStore,
        role: LsmRole::Reader,
    })
}

/// A reader replica across a regraph that retires one segment and rebuilds
/// the other. A read pinned before it finds its points for as long as the
/// replica keeps the cores of the two, and counts neither as a segment the
/// metadata names and nothing can serve. A read begun once the replica has
/// the new list finds the same points, each once.
#[test]
fn a_reader_pinned_before_a_retire_finds_its_points_through_the_grace() {
    use crate::helix_engine::vector_core::named_vectors::take_missing_segment_note;

    let _guard = ENV_LOCK.lock().unwrap();
    let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "reader_across_retire");
    let _poll = EnvVarGuard::set("HELIX_LSM_READER_MANIFEST_POLL_MS", "60000");
    let _pinned = EnvVarGuard::set("HELIX_LSM_READER_SNAPSHOT_READS", "1");
    let _switch = EnvVarGuard::set("HELIX_LSM_REUPSERT_DELETE", "0");
    let _store = set_lsm_test_object_store(Arc::new(InMemory::new()));
    let name = "reader_across_retire";
    let dir = TempDir::new().unwrap();
    let (_writer, _config, storage, rows, sources) = written_twice(dir.path(), name);

    let reader_dir = TempDir::new().unwrap();
    let reader = CollectionManager::new(reader_dir.path().to_path_buf(), reader_config()).unwrap();
    let replica = reader.get_collection(name).unwrap();
    replica
        .named_vectors
        .set_retired_core_grace(Duration::from_secs(3600));
    let before = replica.backend.begin_read().unwrap();
    let expected = nearest(&replica, &before, &rows[0]);
    assert!(
        expected.len() == 10 && expected.windows(2).all(|pair| pair[0] < pair[1]),
        "ten points, each once: {expected:?}"
    );

    let status = test_support::regraph(&storage, name, "dense");
    assert_eq!(
        (status.segments_retired, status.segments_rebuilt),
        (1, 1),
        "{status:?}"
    );
    storage.backend.flush_durable().unwrap();

    // The view moves (a reopen stands in for the poll), then another read
    // arrives and reconciles.
    replica
        .backend
        .refresh_lsm_reader_with_poll_interval(Duration::from_secs(60))
        .unwrap();
    drop(reader.get_collection(name).unwrap());
    let layout = dense_layout(&replica);
    assert!(
        !layout.iter().any(|(segment, _)| sources.contains(segment)),
        "the shared segment list did not follow the view: {layout:?}"
    );
    let after = replica.backend.begin_read().unwrap();
    assert_eq!(
        nearest(&replica, &after, &rows[0]),
        expected,
        "a read begun after the retire"
    );
    // The two are off the list and the one before it after two more
    // reconciles; their cores stay for the grace.
    for reconciles in 0..3 {
        assert_eq!(
            nearest(&replica, &before, &rows[0]),
            expected,
            "a read pinned before the retire, {reconciles} reconciles later"
        );
        replica.refresh_reader_view_best_effort();
    }
    for segment in &sources {
        assert!(
            !take_missing_segment_note(name, "dense", segment),
            "{segment} was counted as a segment nothing can serve"
        );
    }
}

/// The second write stays in the tail, which no regraph rebuilds, so the
/// retire is the one change to the writer's list. A read begun while the
/// first rows were the points' only ones has none of its points in the list
/// that is left: its search is answered on a newer read, as after any drop
/// from the list, right after the retire and once the reaper has taken the
/// segment. A read begun between the second write and the retire is at the
/// sequence the segment was dropped at and stays on its own: it has every
/// point's own copy in the tail, as the regraph found them.
#[test]
fn a_writer_search_begun_before_a_retire_is_answered_on_a_newer_read() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "writer_read_before_retire");
    let _switch = EnvVarGuard::set("HELIX_LSM_REUPSERT_DELETE", "0");
    let _store = set_lsm_test_object_store(Arc::new(InMemory::new()));
    let name = "writer_read_before_retire";
    let dir = TempDir::new().unwrap();
    let (writer, config, storage, rows) = indexed_lsm_collection(dir.path(), name, 16, POINTS);
    let source = sole_indexed_segment(&storage);
    let before = storage.backend.begin_read().unwrap();
    let expected = nearest(&storage, &before, &rows[0]);
    assert_eq!(expected.len(), 10, "before the second write");
    upsert(&writer, &config, name, points(&rows, "g"));
    let at_the_retire = storage.backend.begin_read().unwrap();

    let status = test_support::regraph(&storage, name, "dense");
    assert_eq!(
        (status.segments_retired, status.segments_rebuilt),
        (1, 0),
        "{status:?}"
    );
    for when in ["right after the retire", "once the reaper took the segment"] {
        if when.starts_with("once") {
            wait_for_the_reaper();
            assert!(
                !storage
                    .named_vectors
                    .cores_read()
                    .unwrap()
                    .contains_key(&source),
                "the reaper left the segment's core"
            );
        }
        let (found, newer_read) = with_newer_read_lines(|| nearest(&storage, &before, &rows[0]));
        assert_eq!(
            found, expected,
            "a read begun before the second write, {when}"
        );
        assert!(
            !newer_read.is_empty(),
            "the older read was not moved to a newer one, {when}"
        );
        let (found, newer_read) =
            with_newer_read_lines(|| nearest(&storage, &at_the_retire, &rows[0]));
        assert_eq!(found, expected, "a read begun at the retire, {when}");
        assert!(newer_read.is_empty(), "{when}: {newer_read:?}");
    }
}

/// A write that lands after a regraph has read the segment it would retire,
/// and before it takes the gate, stands the retire down: the segment stays,
/// and the next regraph takes it out.
#[test]
fn a_write_after_the_export_stands_a_retire_down() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "write_before_retire");
    let _switch = EnvVarGuard::set("HELIX_LSM_REUPSERT_DELETE", "0");
    let _store = set_lsm_test_object_store(Arc::new(InMemory::new()));
    let name = "write_before_retire";
    let dir = TempDir::new().unwrap();
    let (writer, config, storage, rows, [first, _]) = written_twice(dir.path(), name);

    // One more point, into the tail.
    let mut late = points(&rows[..1], "late");
    late[0].id = POINTS as u128 + 1;
    let write = {
        let (writer, config) = (Arc::clone(&writer), config.clone());
        move || upsert(&writer, &config, name, late)
    };
    let mut status = RegraphStatus::default();
    test_support::with_hook("regraph_exported", write, || {
        status = test_support::regraph(&storage, name, "dense");
    });
    assert_eq!(
        (status.state, status.segments_busy, status.segments_retired),
        ("busy", 1, 0),
        "{status:?}"
    );
    assert!(
        indexed(&storage).contains(&first),
        "the segment went all the same: {:?}",
        dense_layout(&storage)
    );
    assert_eq!(lost(&storage, &rows), Vec::<u128>::new());

    let status = test_support::regraph(&storage, name, "dense");
    assert_eq!(
        (status.state, status.segments_retired),
        ("done", 1),
        "{status:?}"
    );
    assert!(!indexed(&storage).contains(&first));
    assert_eq!(stored(&storage), (POINTS, 0));
    assert_eq!(lost(&storage, &rows), Vec::<u128>::new());
}

/// Under `HELIX_LSM_DELETE_TOMBSTONES=1` a delete leaves a sealed segment's
/// row where it is, marked. 520 points sit in a merged segment numbered above
/// the tail; they are deleted, written again with the payload they had, and
/// the tail is sealed. Each point's row is in the older segment, and a dead
/// row of it, with the same payload, in the more recent one. A dead row
/// replaces nothing: the regraph rebuilds the older segment with every row,
/// and takes out the more recent one, which it would keep no row of.
#[test]
fn regraph_keeps_a_row_whose_more_recent_copy_is_a_deleted_one() {
    use crate::helix_engine::vector_core::named_vectors::PreparedMerge;

    let _guard = ENV_LOCK.lock().unwrap();
    let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "regraph_dead_copy_above");
    let _store = set_lsm_test_object_store(Arc::new(InMemory::new()));
    let name = "regraph_dead_copy_above";
    let hnsw = HNSWConfig::new(Some(16), Some(64), Some(128));
    let dir = TempDir::new().unwrap();
    let (writer, config, storage, rows) = indexed_lsm_collection(dir.path(), name, 16, POINTS);
    let source = sole_indexed_segment(&storage);
    let tail = storage
        .named_vectors
        .mutable_dense_segment_name("dense")
        .expect("a mutable tail");
    // A merge of the one segment: its rows go to a segment made after the
    // tail was.
    let merge = PreparedMerge {
        merge_targets: vec![source.clone()],
        prepared_index: prepared_merge_of(&storage, &source),
    };
    run_chunked_merge_publish(&storage, "dense", &hnsw, merge, None).unwrap();
    let merged = sole_indexed_segment(&storage);
    assert!(merged > tail, "{merged} is not numbered above {tail}");

    let _tombstones = EnvVarGuard::set("HELIX_LSM_DELETE_TOMBSTONES", "1");
    let delete = ReplicatedMutation::DeletePoints {
        collection: name.into(),
        ids: (1..=POINTS as u128).collect(),
    };
    apply_mutation(&writer, &config, &delete).unwrap();
    upsert(&writer, &config, name, points(&rows, "f"));
    test_support::seal_and_index(&storage, "dense");
    let mut two = indexed(&storage);
    two.sort();
    assert_eq!(
        (two, stored(&storage)),
        (vec![tail.clone(), merged.clone()], (POINTS, POINTS)),
        "not the arrangement: a row of each point in each of the two"
    );
    assert_eq!(lost(&storage, &rows), Vec::<u128>::new(), "before");

    let status = test_support::regraph(&storage, name, "dense");
    assert_eq!(
        (
            status.segments_retired,
            status.segments_rebuilt,
            status.points_rebuilt
        ),
        (1, 1, POINTS),
        "{status:?}"
    );
    let left = indexed(&storage);
    assert!(
        left.len() == 1 && left[0] != tail && left[0] != merged,
        "{left:?}"
    );
    assert_eq!(stored(&storage), (POINTS, 0));
    assert_eq!(lost(&storage, &rows), Vec::<u128>::new(), "after");
}

/// A writer that dies at any durable point of a regraph that retires one
/// segment and rebuilds the other, the first of them between the write that
/// takes the segment off the list and the reaper's first pass over its rows,
/// and one that is reopened once it is all done. Each opens, finds every
/// point, and after a regraph of its own keeps one row of each.
#[test]
fn a_writer_reopened_at_any_point_of_a_retire_keeps_every_point() {
    let _guard = ENV_LOCK.lock().unwrap();
    let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "retire_crash");
    let _group = EnvVarGuard::unset("HELIX_LSM_GROUP_COMMIT");
    let _switch = EnvVarGuard::set("HELIX_LSM_REUPSERT_DELETE", "0");
    let store = Arc::new(WalSnapshots::default());
    let _store = set_lsm_test_object_store(Arc::clone(&store) as Arc<dyn ObjectStore>);
    let name = "retire_crash";
    let dir = TempDir::new().unwrap();
    let (writer, config, storage, rows, [first, _]) = written_twice(dir.path(), name);

    store.armed.store(true, Ordering::Release);
    let status = test_support::regraph(&storage, name, "dense");
    wait_for_the_reaper();
    storage.backend.flush_durable().unwrap();
    store.armed.store(false, Ordering::Release);
    assert_eq!(
        (status.segments_retired, status.segments_rebuilt),
        (1, 1),
        "{status:?}"
    );
    let mut snapshots = std::mem::take(&mut *store.snapshots.lock().unwrap());
    assert!(
        snapshots.len() >= 2,
        "the retire and the rebuild are {} durable writes",
        snapshots.len()
    );
    // And as a writer that was shut down after it all has it.
    snapshots.push(store.inner.fork());
    drop(storage);
    drop(writer);

    let writers = snapshots.len();
    let mut retired_at_open = 0;
    for (at, snapshot) in snapshots.into_iter().enumerate() {
        let _crash = set_lsm_test_object_store(Arc::new(snapshot));
        let dir = TempDir::new().unwrap();
        let writer =
            Arc::new(CollectionManager::new(dir.path().to_path_buf(), config.clone()).unwrap());
        let storage = writer.get_collection(name).unwrap();
        wait_for_index_executor(name);
        retired_at_open += usize::from(!indexed(&storage).contains(&first));
        // Every point has a row in a segment search serves; one in 13 is
        // searched for (a search of each takes the test past two minutes).
        assert_eq!(stored(&storage).0, POINTS, "reopened at write {at}");
        assert_eq!(
            lost_of_every(13, &storage, &rows),
            Vec::<u128>::new(),
            "reopened at write {at}"
        );
        let status = test_support::regraph(&storage, name, "dense");
        assert_eq!(status.state, "done", "write {at}: {status:?}");
        assert_eq!(stored(&storage), (POINTS, 0), "write {at}: {status:?}");
        assert_eq!(
            lost_of_every(13, &storage, &rows),
            Vec::<u128>::new(),
            "write {at}, after a regraph"
        );
        wait_for_the_reaper();
    }
    println!("{writers} writers reopened, {retired_at_open} with the segment off their list");
    assert!(
        retired_at_open > 0,
        "no writer opened with the segment off its list"
    );
}
