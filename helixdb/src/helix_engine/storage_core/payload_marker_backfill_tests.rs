use super::*;
use crate::helix_engine::storage_core::collection_manager::CollectionManager;
use crate::helix_engine::storage_core::replication::{
    apply_mutation, ReplicatedMutation, ReplicatedPoint,
};
use crate::helix_engine::vector_core::named_vectors::{DistanceMetric, NamedVectorConfig};
use serde_json::json;
use serial_test::serial;
use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;
use tempfile::TempDir;

const FIELD: &str = "metadata.path";

/// Sets env vars for one test and restores them on drop.
struct Env(Vec<(&'static str, Option<std::ffi::OsString>)>);

impl Env {
    fn set(pairs: &[(&'static str, &str)]) -> Self {
        let saved = pairs
            .iter()
            .map(|(key, value)| {
                let previous = std::env::var_os(key);
                unsafe { std::env::set_var(key, value) };
                (*key, previous)
            })
            .collect();
        Self(saved)
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        for (key, previous) in self.0.drain(..) {
            match previous {
                Some(value) => unsafe { std::env::set_var(key, value) },
                None => unsafe { std::env::remove_var(key) },
            }
        }
    }
}

struct Fixture {
    _dir: TempDir,
    manager: Arc<CollectionManager>,
    config: Config,
    storage: Arc<HelixGraphStorage>,
}

impl Fixture {
    /// A collection indexed on [`FIELD`] holding `points`, with every index
    /// row rewritten the way binaries before markers wrote it.
    fn legacy(points: Vec<(u128, serde_json::Value)>) -> Self {
        let dir = TempDir::new().unwrap();
        let config = Config::default().with_lsm_in_memory();
        let manager =
            Arc::new(CollectionManager::new(dir.path().to_path_buf(), config.clone()).unwrap());
        let create = ReplicatedMutation::CreateCollection {
            name: "c".into(),
            vectors: HashMap::from([(
                "dense".into(),
                NamedVectorConfig {
                    size: 2,
                    distance: DistanceMetric::Cosine,
                    spindle: Default::default(),
                },
            )]),
            sparse_vectors: HashMap::new(),
            hnsw_overrides: None,
        };
        apply_mutation(&manager, &config, &create).unwrap();
        let storage = manager.get_collection("c").unwrap();
        upsert(&manager, &config, points);
        storage
            .create_payload_index(FIELD, PayloadIndexSchema::Keyword)
            .unwrap();
        storage.strip_payload_markers_for_test(FIELD);
        Self {
            _dir: dir,
            manager,
            config,
            storage,
        }
    }
}

fn upsert(
    manager: &Arc<CollectionManager>,
    config: &Config,
    points: Vec<(u128, serde_json::Value)>,
) {
    let points = points
        .into_iter()
        .map(|(id, payload)| ReplicatedPoint {
            id,
            vectors: HashMap::from([("dense".into(), vec![1.0, id as f32])]),
            sparse_vectors: HashMap::new(),
            payload: match Value::from(payload) {
                Value::Object(payload) => payload,
                other => panic!("payload {other:?}"),
            },
        })
        .collect();
    let mutation = ReplicatedMutation::UpsertPoints {
        collection: "c".into(),
        points,
    };
    apply_mutation(manager, config, &mutation).unwrap();
}

fn path(value: serde_json::Value) -> serde_json::Value {
    json!({"metadata": {"path": value}})
}

/// Every payload shape: ids `% 6 == 5` hold plain strings.
fn shapes() -> Vec<(u128, serde_json::Value)> {
    (1..=30u128)
        .map(|id| {
            let s = format!("src/p{id}.rs");
            let payload = match id % 6 {
                0 => path(json!([s])),
                1 => path(json!([[s]])),
                2 => path(json!(id as u64)),
                3 => path(json!({"inner": s})),
                4 => json!({"metadata": {"other": s}}),
                _ => path(json!(s)),
            };
            (id, payload)
        })
        .collect()
}

fn db_name() -> String {
    HelixGraphStorage::payload_index_db_name(FIELD, &PayloadIndexSchema::Keyword)
}

/// `(row key, node id) -> stored value` of every [`FIELD`] index row.
fn rows(storage: &HelixGraphStorage) -> BTreeMap<(Vec<u8>, u128), Vec<u8>> {
    let db_name = db_name();
    let read = storage.backend.begin_read().unwrap();
    let mut rows = BTreeMap::new();
    storage
        .backend
        .scan_dup_stored(
            &read,
            Namespace::PayloadIndex(&db_name),
            KeyRange::all(),
            |key, id, stored| {
                let id = u128::from_be_bytes(id.try_into().unwrap());
                rows.insert((key.to_vec(), id), stored.to_vec());
                true
            },
        )
        .unwrap();
    rows
}

/// The row key of `id`'s current value when it is a top-level string.
fn plain_key(storage: &HelixGraphStorage, read: &AnyRead<'_>, id: u128) -> Option<Vec<u8>> {
    let node = storage.get_node_be(read, &id).ok()?;
    match HelixGraphStorage::payload_value_for_key(&node.properties, FIELD) {
        Some(value @ Value::String(_)) => {
            Some(HelixGraphStorage::stable_index_key_for_value(value).unwrap())
        }
        _ => None,
    }
}

/// A string row is marked only when its node holds its key as a top-level
/// string (and, with `complete`, every such row is marked). Returns marked
/// string rows. Walks ignore markers on other rows (the write path marks
/// every non-array value).
fn assert_markers(storage: &HelixGraphStorage, complete: bool) -> usize {
    let read = storage.backend.begin_read().unwrap();
    let mut marked = 0;
    for ((key, id), stored) in rows(storage) {
        if !matches!(bincode::deserialize::<Value>(&key), Ok(Value::String(_))) {
            continue;
        }
        let plain = plain_key(storage, &read, id).as_ref() == Some(&key);
        let is_marked = stored == PAYLOAD_INDEX_PLAIN_ROW;
        assert!(
            is_marked <= plain,
            "node {id}: marked a row it does not hold as a string"
        );
        if complete {
            assert_eq!(is_marked, plain, "node {id}: plain row left unmarked");
        }
        marked += usize::from(is_marked);
    }
    marked
}

fn run(storage: &Arc<HelixGraphStorage>) -> MarkerBackfillRun {
    HelixGraphStorage::run_payload_marker_backfill(&Arc::downgrade(storage)).unwrap()
}

fn progress(storage: &HelixGraphStorage) -> Option<Vec<u8>> {
    let read = storage.backend.begin_read().unwrap();
    storage
        .backend
        .get_with(&read, Namespace::Metadata, &progress_key(FIELD), |v| {
            v.map(<[u8]>::to_vec)
        })
        .unwrap()
}

fn node_ids(storage: &HelixGraphStorage) -> Vec<u128> {
    let read = storage.backend.begin_read().unwrap();
    let mut ids = Vec::new();
    storage
        .backend
        .scan(&read, Namespace::Nodes, KeyRange::all(), |key, _| {
            ids.push(u128::from_be_bytes(key.try_into().unwrap()));
            true
        })
        .unwrap();
    ids
}

#[test]
#[serial]
fn backfill_marks_plain_string_rows_only() {
    let _env = Env::set(&[
        ("HELIX_PAYLOAD_MARKER_BACKFILL_BATCH", "4"),
        ("HELIX_PAYLOAD_MARKER_BACKFILL_PAUSE_MS", "0"),
    ]);
    let fx = Fixture::legacy(shapes());
    // A ghost: a row for a value node 5 no longer holds.
    let ghost = (
        HelixGraphStorage::stable_index_key_for_value(&Value::String("src/old.rs".into())).unwrap(),
        5u128,
    );
    let mut write = fx.storage.backend.begin_write().unwrap();
    fx.storage
        .backend
        .put_dup(
            &mut write,
            Namespace::PayloadIndex(&db_name()),
            &ghost.0,
            &ghost.1.to_be_bytes(),
        )
        .unwrap();
    fx.storage.backend.commit(write).unwrap();
    let before = rows(&fx.storage);
    assert!(before.values().all(Vec::is_empty));
    assert_eq!(assert_markers(&fx.storage, false), 0);

    let done = run(&fx.storage);
    assert!(done.completed && !done.superseded, "{done:?}");
    assert_eq!((done.scanned, done.conflicts), (30, 0), "{done:?}");
    let after = rows(&fx.storage);
    assert!(after.keys().eq(before.keys()), "rows were created or lost");
    assert_eq!(assert_markers(&fx.storage, true), 5);
    assert_eq!(done.marked, 5);
    assert!(after[&ghost].is_empty(), "ghost row marked");
    assert_eq!(progress(&fx.storage).as_deref(), Some(PROGRESS_DONE));

    // Finished: a later open reads the progress key and nothing else.
    let again = run(&fx.storage);
    assert_eq!(
        (again.scanned, again.marked, again.completed),
        (0, 0, false)
    );
}

#[test]
#[serial]
fn backfill_resumes_from_its_cursor() {
    let _env = Env::set(&[
        ("HELIX_PAYLOAD_MARKER_BACKFILL_BATCH", "4"),
        ("HELIX_PAYLOAD_MARKER_BACKFILL_PAUSE_MS", "0"),
    ]);
    let fx = Fixture::legacy(shapes());
    let mut batches = 0;
    let interrupted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_marker_backfill_hook_for_test(
            move || {
                batches += 1;
                assert!(batches < 3, "interrupted before the third commit");
            },
            || run(&fx.storage),
        )
    }));
    assert!(interrupted.is_err());
    // Batches without markers commit nothing: the two commits were nodes
    // 5..=8 and 9..=12, holding plain strings 5 and 11.
    let ids = node_ids(&fx.storage);
    assert_eq!(progress(&fx.storage), Some(ids[11].to_be_bytes().to_vec()));
    assert_eq!(assert_markers(&fx.storage, false), 2);

    let resumed = run(&fx.storage);
    assert!(resumed.completed, "{resumed:?}");
    assert_eq!((resumed.scanned, resumed.marked), (18, 3), "{resumed:?}");
    assert_eq!(assert_markers(&fx.storage, true), 5);
}

/// A node written after a batch began refuses that batch's commit; the retry
/// rereads it. The node-only rewrite (a write path that leaves index rows
/// alone) touches no index row, so only the node read check catches it.
#[test]
#[serial]
fn node_writes_during_a_batch_refuse_its_markers() {
    let _env = Env::set(&[
        ("HELIX_PAYLOAD_MARKER_BACKFILL_BATCH", "16"),
        ("HELIX_PAYLOAD_MARKER_BACKFILL_PAUSE_MS", "0"),
    ]);
    for node_only in [true, false] {
        let fx = Fixture::legacy(shapes());
        let (manager, config, storage) = (
            Arc::clone(&fx.manager),
            fx.config.clone(),
            Arc::clone(&fx.storage),
        );
        let mut fired = false;
        let done = with_marker_backfill_hook_for_test(
            move || {
                if std::mem::replace(&mut fired, true) {
                    return;
                }
                if node_only {
                    let read = storage.backend.begin_read().unwrap();
                    let mut node = storage.get_node_be(&read, &11).unwrap();
                    drop(read);
                    node.properties.insert(
                        "metadata".into(),
                        Value::from(json!({"path": ["src/p11.rs"]})),
                    );
                    let mut write = storage.backend.begin_write().unwrap();
                    storage
                        .backend
                        .put(
                            &mut write,
                            Namespace::Nodes,
                            &11u128.to_be_bytes(),
                            &SerializedNode::encode_node(&node).unwrap(),
                        )
                        .unwrap();
                    storage.backend.commit(write).unwrap();
                } else {
                    upsert(&manager, &config, vec![(11, path(json!(["src/p11.rs"])))]);
                }
            },
            || run(&fx.storage),
        );
        assert!(
            done.completed && done.conflicts >= 1,
            "{node_only}: {done:?}"
        );
        let key =
            HelixGraphStorage::stable_index_key_for_value(&Value::String("src/p11.rs".into()))
                .unwrap();
        assert_eq!(
            rows(&fx.storage)[&(key, 11)],
            b"",
            "{node_only}: array row marked"
        );
        assert_eq!(assert_markers(&fx.storage, true), 4, "{node_only}");
    }
}

/// Upserts flipping values between strings and arrays race the backfill:
/// no marked row ever disagrees with its node.
#[test]
#[serial]
fn concurrent_flips_never_leave_a_marked_array_row() {
    let _env = Env::set(&[
        ("HELIX_PAYLOAD_MARKER_BACKFILL_BATCH", "8"),
        ("HELIX_PAYLOAD_MARKER_BACKFILL_PAUSE_MS", "0"),
    ]);
    let plain = |id: u128| (id, path(json!(format!("src/p{id}.rs"))));
    let fx = Fixture::legacy((1..=200).map(plain).collect());
    let stop = AtomicBool::new(false);
    let flips = std::thread::scope(|scope| {
        let flipper = scope.spawn(|| {
            let mut flips = 0u128;
            while !stop.load(Ordering::Relaxed) {
                let id = 1 + (flips * 37) % 200;
                let s = format!("src/p{id}.rs");
                let value = if flips % 2 == 0 { json!([s]) } else { json!(s) };
                upsert(&fx.manager, &fx.config, vec![(id, path(value))]);
                flips += 1;
            }
            flips
        });
        let done = run(&fx.storage);
        stop.store(true, Ordering::Relaxed);
        assert!(done.completed, "{done:?}");
        flipper.join().unwrap()
    });
    assert!(flips > 10, "{flips} flips");
    assert_markers(&fx.storage, false);
}

/// A dropped index gets no markers; a rebuilt one is a new generation,
/// built with markers and registered done.
#[test]
#[serial]
fn backfill_never_marks_a_dropped_or_rebuilt_index() {
    let _env = Env::set(&[
        ("HELIX_PAYLOAD_MARKER_BACKFILL_BATCH", "4"),
        ("HELIX_PAYLOAD_MARKER_BACKFILL_PAUSE_MS", "0"),
    ]);
    for rebuild in [false, true] {
        let fx = Fixture::legacy(shapes());
        let storage = Arc::clone(&fx.storage);
        let mut fired = false;
        let stopped = with_marker_backfill_hook_for_test(
            move || {
                if std::mem::replace(&mut fired, true) {
                    return;
                }
                storage.delete_payload_index(FIELD).unwrap();
                if rebuild {
                    storage
                        .create_payload_index(FIELD, PayloadIndexSchema::Keyword)
                        .unwrap();
                }
            },
            || run(&fx.storage),
        );
        assert!(
            stopped.superseded && stopped.marked == 0,
            "{rebuild}: {stopped:?}"
        );
        if !rebuild {
            assert_eq!(progress(&fx.storage), None);
            assert!(
                rows(&fx.storage).values().all(Vec::is_empty),
                "dropped index marked"
            );
            assert!(run(&fx.storage).scanned == 0, "ran without an index");
            fx.storage
                .create_payload_index(FIELD, PayloadIndexSchema::Keyword)
                .unwrap();
        }
        assert_eq!(progress(&fx.storage).as_deref(), Some(PROGRESS_DONE));
        let later = run(&fx.storage);
        assert!(
            later.scanned == 0 && later.marked == 0,
            "{rebuild}: {later:?}"
        );
        assert_eq!(assert_markers(&fx.storage, true), 5);
    }
}

/// A node written during every attempt shrinks its batch to itself and is
/// then skipped unmarked; the rest of the collection is still marked.
#[test]
#[serial]
fn a_node_that_keeps_conflicting_is_skipped_unmarked() {
    let _env = Env::set(&[
        ("HELIX_PAYLOAD_MARKER_BACKFILL_BATCH", "16"),
        ("HELIX_PAYLOAD_MARKER_BACKFILL_PAUSE_MS", "0"),
    ]);
    let plain = |id: u128| (id, path(json!(format!("src/p{id}.rs"))));
    let fx = Fixture::legacy((1..=8).map(plain).collect());
    let storage = Arc::clone(&fx.storage);
    // Rewrites node 1 as it is, before every commit.
    let done = with_marker_backfill_hook_for_test(
        move || {
            let read = storage.backend.begin_read().unwrap();
            let node = storage.get_node_be(&read, &1).unwrap();
            drop(read);
            let mut write = storage.backend.begin_write().unwrap();
            storage
                .backend
                .put(
                    &mut write,
                    Namespace::Nodes,
                    &1u128.to_be_bytes(),
                    &SerializedNode::encode_node(&node).unwrap(),
                )
                .unwrap();
            storage.backend.commit(write).unwrap();
        },
        || run(&fx.storage),
    );
    // Batches of 16, 8, 4, 2 and 1 nodes conflicted.
    assert!(done.completed, "{done:?}");
    assert_eq!(
        (done.conflicts, done.skipped, done.scanned, done.marked),
        (5, 1, 8, 7),
        "{done:?}"
    );
    let key =
        HelixGraphStorage::stable_index_key_for_value(&Value::String("src/p1.rs".into())).unwrap();
    assert_eq!(rows(&fx.storage)[&(key, 1)], b"");
    assert_eq!(assert_markers(&fx.storage, false), 7);
}

#[test]
#[serial]
fn kill_switch_and_background_submit() {
    let fx = Fixture::legacy(shapes());
    {
        let _off = Env::set(&[("HELIX_PAYLOAD_MARKER_BACKFILL", "0")]);
        assert!(!fx.storage.submit_payload_marker_backfill("c"));
    }
    let _on = Env::set(&[
        ("HELIX_PAYLOAD_MARKER_BACKFILL", "1"),
        ("HELIX_PAYLOAD_MARKER_BACKFILL_PAUSE_MS", "0"),
    ]);
    {
        let _fields = Env::set(&[("HELIX_PAYLOAD_MARKER_BACKFILL_FIELDS", "metadata.repo")]);
        assert!(!fx.storage.submit_payload_marker_backfill("c"), "no index");
    }
    assert_eq!(assert_markers(&fx.storage, false), 0);
    assert!(fx.storage.submit_payload_marker_backfill("c"));
    let deadline = Instant::now() + Duration::from_secs(30);
    while progress(&fx.storage).as_deref() != Some(PROGRESS_DONE) {
        assert!(
            Instant::now() < deadline,
            "background backfill did not finish"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(assert_markers(&fx.storage, true), 5);
}
