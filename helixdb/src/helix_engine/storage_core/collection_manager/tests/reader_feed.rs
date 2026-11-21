//! The reader's change-feed poller (`apply_reader_feed_change`): which
//! changes move a collection's `DbReader` to the short poll, by a poll of it
//! in place or by a reopen, and which only record the write and leave it to
//! the reader's own poll.

use super::super::{apply_reader_feed_change, demote_reader_after_feed_quiet};
use super::*;
use crate::helix_engine::storage_core::backend::{Namespace, StorageBackend};
use futures::{stream, StreamExt};
use slatedb::object_store::memory::InMemory;
use slatedb::object_store::ObjectStore;
use std::future::Future;

/// A keyspace nothing in the engine reads, for rows the tests write directly.
const ROWS: Namespace<'static> = Namespace::SecondaryIndex("reader_feed_test");
/// The interval a promotion moves to. No test waits for it.
const ACTIVE: Duration = Duration::from_secs(60);
/// How long a write stamp is looked back for: any stamp of the test.
const STAMPED: Duration = Duration::from_secs(600);

fn put_row(storage: &HelixGraphStorage, key: &[u8]) {
    let mut txn = storage.backend.begin_write().unwrap();
    storage.backend.put(&mut txn, ROWS, key, b"v").unwrap();
    storage.backend.commit(txn).unwrap();
}

fn has_row(storage: &HelixGraphStorage, key: &[u8]) -> bool {
    let txn = storage.backend.begin_read().unwrap();
    storage
        .backend
        .get_with(&txn, ROWS, key, |value| value.is_some())
        .unwrap()
}

/// `DbReader`s built for the replica so far: its cold open and every reopen.
fn reader_builds(replica: &HelixGraphStorage) -> u64 {
    let (swaps, _) = replica.backend.lsm_reader_view().expect("a reader replica");
    swaps + 1
}

/// A writer and a reader replica over one object store. The reader has no
/// feed to poll, so a collection it opens starts on the short poll, as one
/// does in production, and nothing but the test applies a change.
struct Pair {
    writer: CollectionManager,
    reader: CollectionManager,
    _dirs: (TempDir, TempDir),
    _store: backend_any::LsmTestObjectStoreGuard,
    _env: Vec<EnvRestore>,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl Pair {
    fn new(root: &'static str, short_poll_ms: &str) -> Self {
        Self::over(root, short_poll_ms, Arc::new(InMemory::new()))
    }

    fn over(root: &'static str, short_poll_ms: &str, store: Arc<dyn ObjectStore>) -> Self {
        let lock = ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let env = vec![
            EnvRestore::set("HELIX_LSM_PREFIX", root),
            EnvRestore::set("HELIX_LSM_READER_MANIFEST_POLL_MS", short_poll_ms),
            EnvRestore::unset("HELIX_LSM_WRITER_FEED_URL"),
            EnvRestore::unset("HELIX_LSM_READER_IDLE_AFTER_MS"),
        ];
        let store = backend_any::set_lsm_test_object_store(store);
        let manager = |role| {
            let dir = TempDir::new().unwrap();
            let config = test_config().with_storage_backend(StorageBackendConfig::Lsm {
                storage: LsmStorage::ObjectStore,
                role,
            });
            let manager = CollectionManager::new(dir.path().to_path_buf(), config).unwrap();
            (manager, dir)
        };
        let (writer, writer_dir) = manager(LsmRole::Writer);
        let (reader, reader_dir) = manager(LsmRole::Reader);
        Self {
            writer,
            reader,
            _dirs: (writer_dir, reader_dir),
            _store: store,
            _env: env,
            _lock: lock,
        }
    }

    /// A collection holding one row: the writer's handle, and the replica a
    /// cold open on the reader gives.
    fn collection(&self, name: &str) -> (Arc<HelixGraphStorage>, Arc<HelixGraphStorage>) {
        let source = self.writer.create_collection(name).unwrap();
        put_row(&source, b"before");
        let replica = self.reader.get_collection(name).unwrap();
        assert!(has_row(&replica, b"before"));
        (source, replica)
    }
}

/// A cold open leaves the collection on the short poll. The first change the
/// feed reports for it used to reopen its reader all the same, because the
/// poller had not promoted it itself: a second replay of the whole WAL tail
/// from the object store, seconds after the first.
#[test]
fn first_change_after_a_cold_open_builds_no_second_reader() {
    let pair = Pair::new("reader_feed_cold_open", "60000");
    let (source, replica) = pair.collection("opened");
    assert!(replica.reader_poll_tier_is_fast());
    assert_eq!(reader_builds(&replica), 1);
    assert!(!replica.write_promoted_recently(STAMPED));

    put_row(&source, b"after");
    assert!(!apply_reader_feed_change(&replica, ACTIVE).unwrap());
    assert_eq!(
        reader_builds(&replica),
        1,
        "the first change after a cold open reopened the reader"
    );
    assert!(
        replica.write_promoted_recently(STAMPED),
        "the change left no stamp to keep the demotion sweep away"
    );

    // Nor does any later change of the burst.
    for _ in 0..10 {
        assert!(!apply_reader_feed_change(&replica, ACTIVE).unwrap());
    }
    assert_eq!(reader_builds(&replica), 1);
}

/// A collection dropped and created again under a reader that stays on the
/// short poll. No tier change reopens that reader, and the first change of
/// the new collection only records the write, so the reader's own poll is
/// what meets the new database: it used to go on in it from its own WAL
/// cursor, kept the dropped collection's rows and never showed a row of the
/// new one, until a demotion reopened it. Its own checkpoint is not in the
/// new manifest, and the poll now ends the reader as a drop does. The sweep
/// evicts it and the next read opens the collection that is there.
#[test]
fn collection_created_again_is_not_read_from_the_dropped_one() {
    let pair = Pair::new("reader_feed_created_again", "500");
    let (source, replica) = pair.collection("again");
    drop(source);
    pair.writer.drop_collection("again").unwrap();
    let source = pair.writer.create_collection("again").unwrap();
    put_row(&source, b"after");
    assert!(!apply_reader_feed_change(&replica, ACTIVE).unwrap());

    let shows = |key: &[u8]| -> String {
        match replica.backend.begin_read() {
            Err(error) => format!("{error:?}"),
            Ok(txn) => format!(
                "{:?}",
                replica
                    .backend
                    .get_with(&txn, ROWS, key, |value| value.is_some())
            ),
        }
    };
    // The slack is for a stalled runner: the reader polls every 500 ms.
    let written = Instant::now();
    while replica.backend.unhealthy_lsm_writer_close_reason() != Some(CloseReason::DatabaseMissing)
    {
        assert!(
            written.elapsed() < Duration::from_secs(10),
            "the reader went on in the dropped collection: before={} after={}",
            shows(b"before"),
            shows(b"after"),
        );
        thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(reader_builds(&replica), 1);
    // Until the sweep comes a read of it is refused; it is not answered from
    // the dropped collection.
    assert!(shows(b"before").contains("does not exist"));
    drop(replica);

    // What the reader's sweep does on its next round.
    {
        let mut collections = pair.reader.collections.write().unwrap();
        CollectionManager::evict_unhealthy_writers(&mut collections, &pair.reader.closing);
    }
    assert!(pair
        .reader
        .get_loaded_collection("again")
        .unwrap()
        .is_none());
    let reopened = pair.reader.get_collection("again").unwrap();
    assert!(has_row(&reopened, b"after"));
    assert!(!has_row(&reopened, b"before"));
}

/// The same drop and create under a reader on the idle poll, where a change
/// reopens the reader. The new `DbReader` reads the new collection rightly:
/// it opened that manifest and wrote its own checkpoint into it, so no poll
/// of its own finds anything wrong. But it used to be swapped in under the
/// storage object of the dropped collection, which keeps what it derived
/// from the dropped rows. The reopen now finds the old reader's checkpoint
/// absent from the manifest the new one opened and does not swap: the old
/// reader is closed, and the storage object goes the way a dropped
/// collection's does.
#[test]
fn collection_created_again_is_not_reopened_under_the_old_storage() {
    use crate::helix_engine::storage_core::backend::BackendError;

    let pair = Pair::new("reader_feed_created_again_idle", "60000");
    let (source, replica) = pair.collection("again");
    // As the demotion sweep leaves a collection nobody read or wrote.
    assert!(replica
        .backend
        .refresh_lsm_reader_with_poll_interval(Duration::from_secs(300))
        .unwrap());
    replica.set_reader_poll_tier_fast(false);
    assert_eq!(reader_builds(&replica), 2);

    drop(source);
    pair.writer.drop_collection("again").unwrap();
    let source = pair.writer.create_collection("again").unwrap();
    // The reader built above polls once as it opens. A checkpoint that poll
    // wrote from the dropped collection's manifest must not be what the
    // create opened.
    assert!(
        !has_row(&source, b"before"),
        "the create opened the dropped collection's manifest"
    );
    put_row(&source, b"after");

    let shows = |key: &[u8]| -> String {
        match replica.backend.begin_read() {
            Err(error) => format!("{error:?}"),
            Ok(txn) => format!(
                "{:?}",
                replica
                    .backend
                    .get_with(&txn, ROWS, key, |value| value.is_some())
            ),
        }
    };
    let changed = apply_reader_feed_change(&replica, ACTIVE);
    assert!(
        matches!(
            &changed,
            Err(BackendError::Io(message)) if message.starts_with(LSM_READER_DATABASE_MISSING)
        ),
        "the storage of the dropped collection got a reader of the new one: \
         change {changed:?}, builds {}, before={} after={}",
        reader_builds(&replica),
        shows(b"before"),
        shows(b"after"),
    );
    assert_eq!(reader_builds(&replica), 2);
    assert!(!replica.reader_poll_tier_is_fast());
    assert_eq!(
        replica.backend.unhealthy_lsm_writer_close_reason(),
        Some(CloseReason::DatabaseMissing)
    );
    // Until the sweep comes a read of it is refused, as after a poll that
    // met the new collection.
    assert!(shows(b"before").contains("does not exist"));
    assert!(shows(b"after").contains("does not exist"));
    let old_storage = Arc::downgrade(&replica);
    drop(replica);

    // What the reader's sweep does on its next round.
    {
        let mut collections = pair.reader.collections.write().unwrap();
        CollectionManager::evict_unhealthy_writers(&mut collections, &pair.reader.closing);
    }
    assert!(pair
        .reader
        .get_loaded_collection("again")
        .unwrap()
        .is_none());
    let reopened = pair.reader.get_collection("again").unwrap();
    assert!(old_storage
        .upgrade()
        .is_none_or(|old| !Arc::ptr_eq(&old, &reopened)));
    assert!(has_row(&reopened, b"after"));
    assert!(!has_row(&reopened, b"before"));
}

/// What the storage object of a dropped collection answers with when a
/// reopen gives it a reader of the new one: a list of ids walked from the
/// dropped collection's payload index. The list is kept under the field, the
/// text, the index handle and the sequence number of the read, and a
/// collection created again and filled the same way stands at that very
/// sequence number.
#[test]
fn list_walked_in_a_dropped_collection_does_not_answer_for_the_new_one() {
    use crate::helix_engine::storage_core::metadata::PayloadIndexSchema;
    use crate::helix_engine::storage_core::replication::{
        apply_mutation, ReplicatedMutation, ReplicatedPoint,
    };
    use crate::helix_engine::vector_core::named_vectors::{DistanceMetric, NamedVectorConfig};

    let _lock = ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _prefix = EnvRestore::set("HELIX_LSM_PREFIX", "reader_feed_walked_list");
    let _poll = EnvRestore::set("HELIX_LSM_READER_MANIFEST_POLL_MS", "60000");
    let _pinned = EnvRestore::set("HELIX_LSM_READER_SNAPSHOT_READS", "1");
    let (_guard, _wdir, writer, _rdir, reader) = shared_store_reader_and_writer();
    let writer = Arc::new(writer);
    // 0 keeps upserts from queueing background index jobs.
    let mut config = writer.config.clone();
    config.vector_config.flat_scan_threshold = Some(0);
    let fill = |ids: std::ops::RangeInclusive<u128>| {
        let mutations = [
            ReplicatedMutation::CreateCollection {
                name: "repo".into(),
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
            ReplicatedMutation::UpsertPoints {
                collection: "repo".into(),
                points: ids
                    .map(|id| ReplicatedPoint {
                        id,
                        vectors: HashMap::from([("dense".into(), vec![id as f32, 1.0, 0.0, 0.0])]),
                        sparse_vectors: HashMap::new(),
                        payload: HashMap::from([(
                            "scope".to_string(),
                            crate::protocol::value::Value::String("all".into()),
                        )]),
                    })
                    .collect(),
            },
        ];
        for mutation in &mutations {
            apply_mutation(&writer, &config, mutation).unwrap();
        }
        writer
            .get_collection("repo")
            .unwrap()
            .create_payload_index("scope", PayloadIndexSchema::Keyword)
            .unwrap();
    };
    // The sequence number of a read, and the ids its walk gives.
    let walk = |storage: &HelixGraphStorage| {
        let read = storage.backend.begin_read().unwrap();
        let seq = read.snapshot_seq().expect("a pinned read");
        let (mut ids, _) = storage
            .get_nodes_by_payload_text_exact_be(&read, "scope", "AL", 0)
            .unwrap()
            .unwrap();
        ids.sort_unstable();
        (seq, ids)
    };

    fill(1..=6);
    let replica = reader.get_collection("repo").unwrap();
    let (dropped_seq, ids) = walk(&replica);
    assert_eq!(ids, (1..=6).collect::<Vec<u128>>());
    assert_eq!(replica.text_walk_cache_stats(), (1, 0));

    writer.drop_collection("repo").unwrap();
    fill(11..=16);
    // A tier change. Refused, and the storage evicted by the sweep; it used
    // to go through and leave the list with the storage that serves on.
    let _ = replica
        .backend
        .refresh_lsm_reader_with_poll_interval(Duration::from_secs(300));
    drop(replica);
    {
        let mut collections = reader.collections.write().unwrap();
        CollectionManager::evict_unhealthy_writers(&mut collections, &reader.closing);
    }
    let serving = reader.get_collection("repo").unwrap();
    let (_, hits) = serving.text_walk_cache_stats();
    let (seq, ids) = walk(&serving);
    assert_eq!(
        seq, dropped_seq,
        "test needs the new collection at the dropped one's sequence number"
    );
    assert_eq!(
        (serving.text_walk_cache_stats().1, ids),
        (hits, (11..=16).collect::<Vec<u128>>()),
        "a list walked in the dropped collection answered for the new one"
    );
}

/// Drops `name` and creates it again with one row, `after`, then waits for
/// the replica's own poll to end its reader.
fn create_again_and_wait_for_the_close(pair: &Pair, name: &str, replica: &HelixGraphStorage) {
    pair.writer.drop_collection(name).unwrap();
    let source = pair.writer.create_collection(name).unwrap();
    put_row(&source, b"after");
    // The slack is for a stalled runner: the reader polls every 500 ms.
    let written = Instant::now();
    while replica.backend.unhealthy_lsm_writer_close_reason() != Some(CloseReason::DatabaseMissing)
    {
        assert!(
            written.elapsed() < Duration::from_secs(10),
            "the reader went on in the dropped collection"
        );
        thread::sleep(Duration::from_millis(20));
    }
}

/// A reader the check has ended stays ended. Every tier change is a reopen,
/// and a reopen swapped a new `DbReader` in without a look at the one it
/// replaced: the close reason went back to none, no sweep evicted the
/// storage any more, and the storage object of the dropped collection served
/// the new one. All five callers of a reopen go through the two backend
/// calls made here; the feed's and a read's are also made as they make them.
#[test]
fn reader_ended_by_the_check_is_not_reopened() {
    let pair = Pair::new("reader_feed_ended_stays_ended", "500");
    let (source, replica) = pair.collection("again");
    drop(source);
    create_again_and_wait_for_the_close(&pair, "again", &replica);
    let ended = |after: &str| {
        assert_eq!(
            (
                replica.backend.unhealthy_lsm_writer_close_reason(),
                reader_builds(&replica)
            ),
            (Some(CloseReason::DatabaseMissing), 1),
            "{after} brought the ended reader back"
        );
    };

    // The demotion sweep's and the feed poller's demotion, and the warm-up's.
    let reopened = replica
        .backend
        .refresh_lsm_reader_with_poll_interval(Duration::from_secs(300));
    assert!(reopened.is_err(), "{reopened:?}");
    ended("a reopen");
    // A read's promotion.
    let reopened = replica
        .backend
        .refresh_lsm_reader_with_poll_interval_bounded(ACTIVE, Duration::from_secs(5));
    assert!(reopened.is_err(), "{reopened:?}");
    ended("a bounded reopen");

    // The feed's change, for a collection on the idle poll.
    replica.set_reader_poll_tier_fast(false);
    let changed = apply_reader_feed_change(&replica, ACTIVE);
    assert!(changed.is_err(), "{changed:?}");
    ended("a feed change");
    // A read of it: the promotion runs on a thread of its own.
    let _idle_after = EnvRestore::set("HELIX_LSM_READER_IDLE_AFTER_MS", "120000");
    replica.note_reader_read_and_maybe_promote();
    let asked = Instant::now();
    while replica.poll_tier_transition_in_flight() {
        assert!(asked.elapsed() < Duration::from_secs(10));
        thread::sleep(Duration::from_millis(10));
    }
    ended("a read's promotion");
    assert!(!replica.reader_poll_tier_is_fast());
}

/// No sweep is waited for. The request that finds an ended reader evicts it
/// and opens the collection that is there, through the lookup that opens and
/// through the one that only hands out what is loaded. Until the sweep came,
/// up to 30 s later, searches were refused and a plain count and the
/// collection's info were answered from the dropped collection's memory.
#[test]
fn request_that_finds_an_ended_reader_gets_the_collection_that_is_there() {
    let pair = Pair::new("reader_feed_no_refusals", "500");
    for (name, loaded_only) in [("opened", false), ("loaded", true)] {
        let (source, replica) = pair.collection(name);
        drop(source);
        create_again_and_wait_for_the_close(&pair, name, &replica);
        let old_storage = Arc::downgrade(&replica);
        drop(replica);

        if loaded_only {
            let loaded = pair.reader.get_loaded_collection(name).unwrap();
            assert!(
                loaded.is_none(),
                "the storage of the dropped collection was handed out as loaded"
            );
        }
        let serving = pair.reader.get_collection(name).unwrap();
        assert!(
            old_storage
                .upgrade()
                .is_none_or(|old| !Arc::ptr_eq(&old, &serving)),
            "the storage of the dropped collection was handed out"
        );
        assert!(has_row(&serving, b"after"));
        assert!(!has_row(&serving, b"before"));
    }
}

/// What a read gets from a block that is gone, and the gateway hands to the
/// manager after it.
fn missing_block_error() -> GraphError {
    let error = GraphError::StorageError(
        "io error: object at location again/compacted/01JD0000000000000000000000.sst not found"
            .to_string(),
    );
    assert!(error.should_quarantine_collection());
    error
}

/// A read reaches a reader before its poll has met a drop and a create, and
/// fetches a block of the collection it opened. The block is gone, the error
/// is a fatal one, and it used to quarantine the name: on a reader nothing
/// clears a quarantine, so every request for the new collection got "open
/// not tried" until the pod restarted. The manager now asks first whether a
/// drop explains the error, and evicts where it does.
#[test]
fn fatal_read_error_that_a_drop_explains_does_not_quarantine_the_name() {
    use super::super::{open_failure_backoff_error, open_failure_clear};

    // A reader polls once as it opens, and here not again for a minute: it
    // meets a drop by itself only if that first poll runs late.
    let pair = Pair::new("reader_feed_quarantine", "60000");
    let quarantined = |name: &str| open_failure_backoff_error(name).is_some();
    let names = ["created_again", "being_dropped", "broken", "not_loaded"];
    names.iter().for_each(|name| open_failure_clear(name));

    // Dropped and created again.
    let (source, replica) = pair.collection("created_again");
    drop(source);
    pair.writer.drop_collection("created_again").unwrap();
    let source = pair.writer.create_collection("created_again").unwrap();
    put_row(&source, b"after");
    drop(replica);
    pair.reader
        .quarantine_collection_after_storage_error("created_again", &missing_block_error());
    assert!(
        !quarantined("created_again"),
        "a collection created again was quarantined for a block of the dropped one"
    );
    let serving = pair.reader.get_collection("created_again").unwrap();
    assert!(has_row(&serving, b"after"));
    assert!(!has_row(&serving, b"before"));

    // In the middle of a drop: the tombstone is written, the manifest still
    // holds the reader's checkpoint.
    let (source, replica) = pair.collection("being_dropped");
    drop((source, replica));
    backend_any::persist_lsm_drop_tombstone_from_env(
        &pair.writer.collection_path("being_dropped"),
        pair.writer.config.storage_backend,
    )
    .unwrap();
    pair.reader
        .quarantine_collection_after_storage_error("being_dropped", &missing_block_error());
    assert!(!quarantined("being_dropped"));
    assert!(pair
        .reader
        .get_loaded_collection("being_dropped")
        .unwrap()
        .is_none());

    // No drop: the error is the collection's own, and quarantines as before.
    let (source, replica) = pair.collection("broken");
    drop((source, replica));
    pair.reader
        .quarantine_collection_after_storage_error("broken", &missing_block_error());
    assert!(quarantined("broken"));
    assert!(pair
        .reader
        .get_loaded_collection("broken")
        .unwrap()
        .is_none());

    // Not loaded: the error is of a storage object this reader no longer
    // serves, as after the eviction above.
    drop(pair.writer.create_collection("not_loaded").unwrap());
    pair.reader
        .quarantine_collection_after_storage_error("not_loaded", &missing_block_error());
    assert!(!quarantined("not_loaded"));

    names.iter().for_each(|name| open_failure_clear(name));
}

/// A quarantine standing on a name when its reader ends is of the database
/// that reader had open, and goes with the reader.
#[test]
fn eviction_of_an_ended_reader_clears_a_quarantine_on_its_name() {
    use super::super::{
        open_failure_backoff_error, open_failure_clear, open_failure_record_quarantine,
    };

    let pair = Pair::new("reader_feed_quarantine_cleared", "500");
    let name = "quarantined_then_created_again";
    open_failure_clear(name);
    let (source, replica) = pair.collection(name);
    drop(source);
    create_again_and_wait_for_the_close(&pair, name, &replica);
    drop(replica);
    // Taken on the reader while its ended reader is still loaded. (After
    // the writer's drop and create: in a test they clear the same map.)
    open_failure_record_quarantine(name, &missing_block_error());

    let serving = pair.reader.get_collection(name);
    assert!(
        serving.is_ok() && open_failure_backoff_error(name).is_none(),
        "the quarantine outlived the collection it was taken for: {:?}",
        serving.as_ref().err()
    );
    assert!(has_row(&serving.unwrap(), b"after"));
    open_failure_clear(name);
}

/// A quarantine on a name with nothing loaded for it has no reader whose end
/// could clear it. On a reader it stood until the pod restarted, and the
/// collection its owner had dropped and created again was refused with the
/// dropped one's error. The reader now ends it when it learns of the drop
/// or of the create: the tombstone its sweep lists, and the feed's list of
/// the names created again over a drop, which is all that tells it once the
/// create has cleared the tombstone.
#[test]
fn quarantine_ends_when_the_reader_learns_of_the_drop_or_the_create() {
    use super::super::{
        apply_reader_feed_reply, open_failure_backoff_error, open_failure_clear,
        open_failure_record_quarantine, sweep_reader_drop_tombstones,
    };
    use crate::helix_engine::storage_core::change_feed::change_feed;

    let pair = Pair::new("reader_feed_quarantine_lifted", "60000");
    // The map of quarantines is the process's: names no other test has.
    const REBUILT: &str = "quarantine_lifted_rebuilt";
    const DROPPED: &str = "quarantine_lifted_dropped";
    const WRITTEN: &str = "quarantine_lifted_written";
    let names = [REBUILT, DROPPED, WRITTEN];
    names.iter().for_each(|name| open_failure_clear(name));
    let quarantined = |name: &str| open_failure_backoff_error(name).is_some();
    // The sweep lists the tombstones through a reader that is loaded.
    let _loaded = pair.collection("stays_loaded");
    // A fault of the collection's own: evicted, and refused from then on.
    let quarantine = |name: &str| {
        drop(pair.collection(name));
        pair.reader
            .quarantine_collection_after_storage_error(name, &missing_block_error());
        assert!(pair.reader.get_loaded_collection(name).unwrap().is_none());
        assert!(quarantined(name) && pair.reader.get_collection(name).is_err());
    };
    // The writer's drop and create clear the name in the map they share with
    // the reader in a test. A reader pod's quarantine stands through both.
    let stands = |name: &str| {
        open_failure_record_quarantine(name, &missing_block_error());
        assert!(pair.reader.get_collection(name).is_err());
    };
    let sweep_lists = || CollectionManager::reader_drop_tombstones(&pair.reader.collections);

    // Dropped and created again between two sweeps: no tombstone is left
    // for the sweep, and the feed names the collection to the next poll.
    quarantine(REBUILT);
    quarantine(WRITTEN);
    let polled = change_feed().changes_since(0).cursor;
    pair.writer.drop_collection(REBUILT).unwrap();
    let source = pair.writer.create_collection(REBUILT).unwrap();
    put_row(&source, b"after");
    stands(REBUILT);
    assert!(!sweep_lists().contains(ObjPath::from(REBUILT).as_ref()));
    // A write is no create: the quarantine of a collection that is only
    // written lasts as long as it did.
    put_row(&pair.writer.get_collection(WRITTEN).unwrap(), b"more");
    let snapshot = change_feed().changes_since(polled);
    assert!(snapshot.changes.iter().any(|name| name == WRITTEN));
    // What the poller's loop does with the reply.
    apply_reader_feed_reply(
        &pair.reader.collections,
        &snapshot,
        &mut HashMap::new(),
        Instant::now(),
        ACTIVE,
    );
    assert!(quarantined(WRITTEN));
    let serving = pair.reader.get_collection(REBUILT);
    assert!(
        serving.is_ok() && !quarantined(REBUILT),
        "the quarantine outlived the collection it was taken for: {:?}",
        serving.as_ref().err()
    );
    let serving = serving.unwrap();
    assert!(has_row(&serving, b"after"));
    assert!(!has_row(&serving, b"before"));

    // Dropped: the tombstone is there for the sweep, and the name is free
    // for whatever is created at it.
    quarantine(DROPPED);
    pair.writer.drop_collection(DROPPED).unwrap();
    stands(DROPPED);
    // What the sweeper's loop does at the head of a round.
    let tombstoned = sweep_reader_drop_tombstones(&pair.reader.collections);
    assert!(tombstoned.contains(ObjPath::from(DROPPED).as_ref()));
    assert!(!quarantined(DROPPED), "a dropped name stayed quarantined");
    assert!(quarantined(WRITTEN));
    let source = pair.writer.create_collection(DROPPED).unwrap();
    put_row(&source, b"after");
    let serving = pair.reader.get_collection(DROPPED).unwrap();
    assert!(has_row(&serving, b"after"));
    assert!(!has_row(&serving, b"before"));

    names.iter().for_each(|name| open_failure_clear(name));
}

/// On the short poll a change only stamps, and the reader's own poll is what
/// meets a collection created again: up to a whole interval later, in which
/// reads are answered from the dropped one. The first change after an
/// interval without one now has the reader's check asked at once. Here the
/// next poll is a minute away.
#[test]
fn first_change_of_a_collection_created_again_ends_its_reader_at_once() {
    let pair = Pair::new("reader_feed_asks", "60000");
    let (source, replica) = pair.collection("again");
    drop(source);
    pair.writer.drop_collection("again").unwrap();
    let source = pair.writer.create_collection("again").unwrap();
    put_row(&source, b"after");

    assert!(!apply_reader_feed_change(&replica, ACTIVE).unwrap());
    // The check is not waited for on the feed's thread; the slack is for a
    // stalled runner.
    let changed = Instant::now();
    while replica.backend.unhealthy_lsm_writer_close_reason() != Some(CloseReason::DatabaseMissing)
    {
        assert!(
            changed.elapsed() < Duration::from_secs(5),
            "the reader of the dropped collection was left to its next poll"
        );
        thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(reader_builds(&replica), 1);
    assert!(replica.write_promoted_recently(STAMPED));
    drop(replica);

    let serving = pair.reader.get_collection("again").unwrap();
    assert!(has_row(&serving, b"after"));
    assert!(!has_row(&serving, b"before"));
}

/// What replaces the reopen: the reader's own poll, which replays only the
/// WAL objects written since its last one.
#[test]
fn change_on_the_short_poll_is_read_within_a_poll_interval() {
    let poll = Duration::from_millis(500);
    let pair = Pair::new("reader_feed_poll", "500");
    let (source, replica) = pair.collection("polled");

    put_row(&source, b"after");
    let written = Instant::now();
    assert!(!apply_reader_feed_change(&replica, ACTIVE).unwrap());
    // The slack is for a stalled runner, not for the reader.
    let late = poll + Duration::from_secs(3);
    while !has_row(&replica, b"after") {
        assert!(
            written.elapsed() < late,
            "a write was not read {late:?} after it, on a {poll:?} poll"
        );
        thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(reader_builds(&replica), 1);
}

/// A change that finds the collection on the idle poll moves it to the short
/// one, and to the write at once: the next idle poll is minutes away. In
/// place that is one poll of the open reader; where a tier change is a reopen
/// it builds a reader, as the demotion before it did.
#[test]
fn change_on_the_idle_poll_promotes_and_reads_the_write() {
    for (root, in_place, built_by_a_tier_change) in [
        ("reader_feed_idle", "1", 0),
        ("reader_feed_idle_reopen", "0", 1),
    ] {
        let pair = Pair::new(root, "60000");
        let _mode = EnvRestore::set("HELIX_LSM_READER_TIER_IN_PLACE", in_place);
        let (source, replica) = pair.collection("idle");
        // As the demotion sweep leaves a collection nobody read or wrote.
        let idle =
            crate::helix_engine::storage_core::backend_lsm::reader_idle_poll_interval_from_env();
        assert!(replica.backend.retier_lsm_reader(idle).unwrap());
        // In place the stored flag is left saying fast, as two tier changes
        // at once can leave it: the reader is asked, not the flag.
        if in_place == "0" {
            replica.set_reader_poll_tier_fast(false);
        }
        assert_eq!(
            reader_builds(&replica),
            1 + built_by_a_tier_change,
            "in place {in_place}"
        );
        assert!(!replica.reader_poll_tier_is_fast());

        put_row(&source, b"after");
        assert!(apply_reader_feed_change(&replica, ACTIVE).unwrap());
        assert_eq!(
            reader_builds(&replica),
            1 + 2 * built_by_a_tier_change,
            "in place {in_place}"
        );
        assert!(replica.reader_poll_tier_is_fast());
        assert!(replica.write_promoted_recently(STAMPED));
        assert!(
            has_row(&replica, b"after"),
            "in place {in_place}: the promotion did not bring the write with it"
        );
    }
}

/// While a demotion is in flight the tier read here can still be the fast
/// one, and the demoted reader may have listed the WAL before the change was
/// written. The change catches the reader up itself: by a poll of it in
/// place, by a reopen where a tier change is one.
#[test]
fn change_during_a_demotion_catches_the_reader_up() {
    for (root, in_place, builds) in [
        ("reader_feed_demoting", "1", 1),
        ("reader_feed_demoting_reopen", "0", 2),
    ] {
        let pair = Pair::new(root, "60000");
        let _mode = EnvRestore::set("HELIX_LSM_READER_TIER_IN_PLACE", in_place);
        let (source, replica) = pair.collection("demoting");
        assert!(replica.try_begin_poll_tier_transition());

        put_row(&source, b"after");
        assert!(apply_reader_feed_change(&replica, ACTIVE).unwrap());
        replica.end_poll_tier_transition();
        assert_eq!(reader_builds(&replica), builds, "in place {in_place}");
        assert!(has_row(&replica, b"after"), "in place {in_place}");
    }
}

/// The poller's own demotion holds the tier transition and stands down while
/// another holds it, here as a read's promotion does. Run beside one, its idle
/// reader could land after the promotion's and the promotion's flag after its
/// own: an idle reader under a flag that says fast, which no read promotes
/// and every later change only stamps.
#[test]
fn demotion_after_quiet_stands_down_for_a_transition_in_flight() {
    for (root, in_place, built_by_a_tier_change) in [
        ("reader_feed_quiet", "1", 0),
        ("reader_feed_quiet_reopen", "0", 1),
    ] {
        let pair = Pair::new(root, "60000");
        let _mode = EnvRestore::set("HELIX_LSM_READER_TIER_IN_PLACE", in_place);
        let (_source, replica) = pair.collection("quiet");
        let idle =
            crate::helix_engine::storage_core::backend_lsm::reader_idle_poll_interval_from_env();

        assert!(replica.try_begin_poll_tier_transition());
        assert!(
            !demote_reader_after_feed_quiet(&replica, idle).unwrap(),
            "in place {in_place}: demoted beside a transition in flight"
        );
        assert_eq!(reader_builds(&replica), 1, "in place {in_place}");
        assert!(replica.reader_poll_tier_is_fast());
        replica.end_poll_tier_transition();

        // On its own it demotes, and lets go of the transition.
        assert!(demote_reader_after_feed_quiet(&replica, idle).unwrap());
        assert_eq!(
            reader_builds(&replica),
            1 + built_by_a_tier_change,
            "in place {in_place}"
        );
        assert!(!replica.reader_poll_tier_is_fast());
        assert!(!replica.poll_tier_transition_in_flight());
    }
}

/// Object store whose reads under `prefix`, once `armed`, wait until
/// `release` is closed; `reached` says one is waiting. Clones share the
/// flags, so a test keeps one to steer the store it gave the managers.
#[derive(Debug, Clone)]
struct ParkedReads {
    inner: Arc<dyn ObjectStore>,
    prefix: &'static str,
    armed: Arc<AtomicBool>,
    reached: Arc<AtomicBool>,
    release: Arc<tokio::sync::Semaphore>,
}

impl ParkedReads {
    fn under(prefix: &'static str) -> Self {
        Self {
            inner: Arc::new(InMemory::new()),
            prefix,
            armed: Arc::default(),
            reached: Arc::default(),
            release: Arc::new(tokio::sync::Semaphore::new(0)),
        }
    }

    fn parks(&self, location: &ObjPath) -> bool {
        self.armed.load(Ordering::SeqCst) && location.to_string().starts_with(self.prefix)
    }
}

impl std::fmt::Display for ParkedReads {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ParkedReads({})", self.inner)
    }
}

type Boxed<'a, T> = std::pin::Pin<Box<dyn Future<Output = object_store::Result<T>> + Send + 'a>>;

impl ObjectStore for ParkedReads {
    fn put_opts<'a, 'b, 'c>(
        &'a self,
        location: &'b ObjPath,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> Boxed<'c, object_store::PutResult>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        self.inner.put_opts(location, payload, opts)
    }

    fn put_multipart_opts<'a, 'b, 'c>(
        &'a self,
        location: &'b ObjPath,
        opts: object_store::PutMultipartOptions,
    ) -> Boxed<'c, Box<dyn object_store::MultipartUpload>>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        self.inner.put_multipart_opts(location, opts)
    }

    fn get_opts<'a, 'b, 'c>(
        &'a self,
        location: &'b ObjPath,
        options: object_store::GetOptions,
    ) -> Boxed<'c, object_store::GetResult>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        if !self.parks(location) {
            return self.inner.get_opts(location, options);
        }
        Box::pin(async move {
            self.reached.store(true, Ordering::SeqCst);
            // Errors once the test closes the semaphore; that is the release.
            let _ = self.release.acquire().await;
            self.inner.get_opts(location, options).await
        })
    }

    fn delete_stream(
        &self,
        locations: stream::BoxStream<'static, object_store::Result<ObjPath>>,
    ) -> stream::BoxStream<'static, object_store::Result<ObjPath>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&ObjPath>,
    ) -> stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        if !prefix.is_some_and(|prefix| self.parks(prefix)) {
            return self.inner.list(prefix);
        }
        let (inner, prefix) = (Arc::clone(&self.inner), prefix.cloned());
        let reached = Arc::clone(&self.reached);
        let release = Arc::clone(&self.release);
        stream::once(async move {
            reached.store(true, Ordering::SeqCst);
            let _ = release.acquire().await;
            // Listed once the wait is over, as a slow store answers: with
            // what is there by then.
            inner.list(prefix.as_ref())
        })
        .flatten()
        .boxed()
    }

    fn list_with_delimiter<'a, 'b, 'c>(
        &'a self,
        prefix: Option<&'b ObjPath>,
    ) -> Boxed<'c, object_store::ListResult>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        self.inner.list_with_delimiter(prefix)
    }

    fn copy_opts<'a, 'b, 'c, 'd>(
        &'a self,
        from: &'b ObjPath,
        to: &'c ObjPath,
        options: object_store::CopyOptions,
    ) -> Boxed<'d, ()>
    where
        'a: 'd,
        'b: 'd,
        'c: 'd,
        Self: 'd,
    {
        self.inner.copy_opts(from, to, options)
    }
}

/// Lets the parked reads through even when the test fails first, so a
/// failure is reported instead of hanging on them.
struct ReleaseOnDrop(Arc<tokio::sync::Semaphore>);

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.0.close();
    }
}

/// A reader is asked for its checkpoint, and a tier change replaces it
/// before the store has answered. The swap closes the replaced reader 30 s
/// on, and a reader deletes its checkpoint as it closes: an answer that
/// comes back after that finds the checkpoint gone, of a reader that is no
/// longer the collection's. It ended the collection's reader all the same:
/// an eviction and a cold open for a collection with nothing wrong. Both
/// asks, the one that waits for the answer and the one that does not.
#[test]
fn late_answer_about_a_replaced_reader_does_not_end_the_one_in_its_place() {
    use crate::helix_engine::storage_core::backend_any::AnyBackend;
    use crate::helix_engine::storage_core::backend_lsm::shared_lsm_handle;

    for (waited, root, manifests) in [
        (
            true,
            "reader_feed_late_waited",
            "reader_feed_late_waited/asked/manifest",
        ),
        (
            false,
            "reader_feed_late_left",
            "reader_feed_late_left/asked/manifest",
        ),
    ] {
        let store = ParkedReads::under(manifests);
        let pair = Pair::over(root, "60000", Arc::new(store.clone()));
        let (_source, replica) = pair.collection("asked");
        let AnyBackend::LsmReader(handle) = replica.backend.as_ref() else {
            panic!("not a reader replica");
        };
        let first = handle.current_reader_for_test();
        // The poll a reader makes as it opens is through, and reads no
        // manifest while the store holds reads back.
        thread::sleep(Duration::from_millis(200));

        store.armed.store(true, Ordering::SeqCst);
        thread::scope(|scope| {
            let _release = ReleaseOnDrop(Arc::clone(&store.release));
            let answer = scope.spawn(|| {
                if waited {
                    replica
                        .backend
                        .lsm_reader_ended_or_checkpoint_absent()
                        .map(Some)
                } else {
                    replica
                        .backend
                        .ask_whether_lsm_reader_checkpoint_is_absent();
                    Ok(None)
                }
            });
            let deadline = Instant::now() + Duration::from_secs(10);
            while !store.reached.load(Ordering::SeqCst) {
                assert!(Instant::now() < deadline, "the ask read no manifest");
                thread::sleep(Duration::from_millis(5));
            }
            store.armed.store(false, Ordering::SeqCst);

            // The tier change, then what its delayed task does 30 s on.
            handle.refresh_with_poll_interval(ACTIVE).unwrap();
            shared_lsm_handle()
                .block_on(async { first.close().await })
                .unwrap();
            // And the store answers the ask.
            store.release.close();
            let answer = answer.join().unwrap().unwrap();
            assert!(
                answer != Some(true),
                "waited={waited}: a replaced reader's lost checkpoint was taken for the collection's"
            );
            // The ask that was not waited for holds the reader it asked
            // until it is through; the swap's delayed close holds it too.
            while Arc::strong_count(&first) > 2 {
                assert!(Instant::now() < deadline, "the ask did not come back");
                thread::sleep(Duration::from_millis(5));
            }
        });
        assert_eq!(
            (
                replica.backend.unhealthy_lsm_writer_close_reason(),
                reader_builds(&replica)
            ),
            (None, 2),
            "waited={waited}: the reader that replaced the one asked was ended"
        );
    }
}

/// The poller applies every collection's change on one thread. A reopen for
/// the first held that thread for the whole replay of its WAL tail, and the
/// next collection's change waited behind it. Here every read of the first
/// collection's objects is parked, so a reopen of it would never return.
#[test]
fn change_on_the_short_poll_does_not_hold_the_feed_thread() {
    let store = ParkedReads::under("reader_feed_thread/slow/");
    let pair = Pair::over("reader_feed_thread", "60000", Arc::new(store.clone()));
    let (_slow_source, slow) = pair.collection("slow");
    let (_next_source, next) = pair.collection("next");
    let applied = AtomicBool::new(false);
    let reopened = AtomicBool::new(false);

    store.armed.store(true, Ordering::SeqCst);
    thread::scope(|scope| {
        let _release = ReleaseOnDrop(Arc::clone(&store.release));
        // As the poller's loop: one thread, one change after the other.
        scope.spawn(|| {
            for replica in [&slow, &next] {
                assert!(!apply_reader_feed_change(replica, ACTIVE).unwrap());
            }
            applied.store(true, Ordering::SeqCst);
        });
        let deadline = Instant::now() + Duration::from_secs(10);
        while !applied.load(Ordering::SeqCst) {
            assert!(
                Instant::now() < deadline,
                "the feed thread is held by a store read for the first collection"
            );
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(reader_builds(&slow), 1);

        // The control: a reopen of the first collection does wait on the
        // parked store, so the loop above cannot have made one.
        scope.spawn(|| {
            let reopen = slow.backend.refresh_lsm_reader_with_poll_interval(ACTIVE);
            reopened.store(true, Ordering::SeqCst);
            assert!(reopen.unwrap());
        });
        while !store.reached.load(Ordering::SeqCst) {
            assert!(
                Instant::now() < deadline,
                "a reopen read nothing under the parked prefix, so this test holds nothing"
            );
            thread::sleep(Duration::from_millis(5));
        }
        assert!(!reopened.load(Ordering::SeqCst));
    });
    assert!(slow.write_promoted_recently(STAMPED));
    assert!(next.write_promoted_recently(STAMPED));
    assert_eq!(reader_builds(&slow), 2, "the control reopen, once released");
}

/// A collection dropped and created again under a reader on the idle poll,
/// where a tier change is made in place. The first change of the new
/// collection promotes the reader by a poll of the `DbReader` it has, and
/// that one is of the dropped collection. A reopen built a reader of the new
/// one; a poll that went on in the new database from the old WAL cursor kept
/// the dropped collection's rows and showed none of the new, and nothing
/// reopened that reader again.
///
/// The fork's poll ends a reader whose own running lease is not in the
/// manifest it loaded ("End a reader whose own running lease is not in the
/// manifest it polls"). This test needs that commit below it: the promotion
/// fails on the closed reader, the sweep evicts it, and the next read opens
/// the collection that is there.
#[test]
fn collection_created_again_is_not_promoted_in_place_on_the_dropped_one() {
    let pair = Pair::new("reader_feed_created_again_in_place", "60000");
    let _mode = EnvRestore::set("HELIX_LSM_READER_TIER_IN_PLACE", "1");
    let (source, replica) = pair.collection("again");
    let idle = crate::helix_engine::storage_core::backend_lsm::reader_idle_poll_interval_from_env();
    assert!(replica.backend.retier_lsm_reader(idle).unwrap());
    assert!(!replica.reader_poll_tier_is_fast());

    drop(source);
    pair.writer.drop_collection("again").unwrap();
    let source = pair.writer.create_collection("again").unwrap();
    put_row(&source, b"after");

    let shows = |key: &[u8]| -> String {
        match replica.backend.begin_read() {
            Err(error) => format!("{error:?}"),
            Ok(txn) => format!(
                "{:?}",
                replica
                    .backend
                    .get_with(&txn, ROWS, key, |value| value.is_some())
            ),
        }
    };
    let promoted = apply_reader_feed_change(&replica, ACTIVE);
    assert!(
        promoted.is_err(),
        "the reader was promoted in the dropped collection: {promoted:?}, before={} after={}",
        shows(b"before"),
        shows(b"after"),
    );
    assert_eq!(
        replica.backend.unhealthy_lsm_writer_close_reason(),
        Some(CloseReason::DatabaseMissing)
    );
    assert_eq!(reader_builds(&replica), 1);
    // Until the sweep comes a read of it is refused; it is not answered from
    // the dropped collection.
    assert!(shows(b"before").contains("does not exist"));
    drop(replica);

    // What the reader's sweep does on its next round.
    {
        let mut collections = pair.reader.collections.write().unwrap();
        CollectionManager::evict_unhealthy_writers(&mut collections, &pair.reader.closing);
    }
    assert!(pair
        .reader
        .get_loaded_collection("again")
        .unwrap()
        .is_none());
    let reopened = pair.reader.get_collection("again").unwrap();
    assert!(has_row(&reopened, b"after"));
    assert!(!has_row(&reopened, b"before"));
}
