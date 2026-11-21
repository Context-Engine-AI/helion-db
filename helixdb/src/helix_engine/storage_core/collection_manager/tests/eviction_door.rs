//! An eviction's take of a collection it found idle (`take_idle_entry`).
//!
//! A pass for capacity, memory or age looks at the holders of every loaded
//! collection, then closes the ones nothing else held, one after another and
//! each close inside the pass. A job keeps its collection by a `Weak` and
//! upgrades it when a worker takes the job. Upgraded after the look, the
//! collection was taken out of the map all the same: open in the job's hands
//! with nothing in the map to say so, and the next request for its name
//! opened a second writer on its path.

use super::idle_gc::{get_row, put_row, Writer};
use super::*;
use crate::helix_engine::storage_core::backend_lsm::flush_gate::test_support::{
    joined, wait_until, HeldSsts,
};
use crate::helix_engine::storage_core::metadata::PayloadIndexSchema;
use crate::helix_engine::storage_core::replication::index_job_queued_or_active;
use crate::helix_engine::storage_core::storage_core::PAYLOAD_INDEX_JOBS_HELD_FOR_TEST;
use slatedb::object_store::ObjectStore;

const FIRST: &str = "eviction_door_first";
const SECOND: &str = "eviction_door_second";

/// Two idle collections and a pass that looks at both and closes the older
/// first. Its close is held at its upload: there the pass has looked at the
/// younger and not yet taken it, and a holder by a `Weak` upgrades it.
fn a_collection_held_since_the_look_stays_loaded(
    root: &'static str,
    pass: fn(&mut HashMap<String, CachedCollection>) -> u64,
) {
    let store = HeldSsts::holding(&format!("/{FIRST}/compacted/"));
    let writer = Writer::over(root, Arc::clone(&store) as Arc<dyn ObjectStore>);
    let [first, second] = [FIRST, SECOND].map(|name| {
        let storage = writer.manager.create_collection(name).unwrap();
        put_row(&storage, b"k1", b"v1");
        storage
    });
    // What a job keeps of its collection until a worker takes it.
    let job = Arc::downgrade(&second);
    // The open's own job is over, and the cache holds the last reference of
    // each: both are idle.
    wait_until("no index job on either collection", || {
        ![FIRST, SECOND].into_iter().any(index_job_queued_or_active)
    });
    drop((first, second));

    let passing = {
        let manager = Arc::clone(&writer.manager);
        thread::spawn(move || pass(&mut manager.collections.write().unwrap()))
    };
    wait_until("the first collection's close at its upload", || {
        store.held() == 1
    });
    // The close runs inside the pass, which holds the map: a request for a
    // name the pass has taken waits for it there, and opens after the close.
    assert!(
        writer.manager.collections.try_read().is_err(),
        "the map is free while a taken collection is closing"
    );
    let held = job
        .upgrade()
        .expect("the pass has not come to the second collection");
    store.release_all();
    let evicted = joined("the eviction pass", passing);

    // The second collection is still in the map: the next request is served
    // the storage that is open, and opens nothing.
    let served = writer.manager.get_collection(SECOND).unwrap();
    assert!(
        Arc::ptr_eq(&served, &held),
        "the next request opened a second writer on the path of one that is open"
    );
    assert_eq!(evicted, 1, "a collection with a holder counted as evicted");
    put_row(&held, b"k2", b"v2");
    assert_eq!(get_row(&served, b"k1").as_deref(), Some(&b"v1"[..]));
    assert_eq!(get_row(&served, b"k2").as_deref(), Some(&b"v2"[..]));

    // The first was taken and closed: it opens again with its row.
    assert!(!writer
        .manager
        .collections
        .read()
        .unwrap()
        .contains_key(FIRST));
    let first = writer.manager.get_collection(FIRST).unwrap();
    assert_eq!(get_row(&first, b"k1").as_deref(), Some(&b"v1"[..]));
}

#[test]
fn an_eviction_for_capacity_leaves_a_collection_held_since_its_look() {
    a_collection_held_since_the_look_stays_loaded("eviction_door_lru", |collections| {
        CollectionManager::evict_lru_idle(collections, 2, "lru")
    });
}

#[test]
fn an_eviction_by_age_leaves_a_collection_held_since_its_look() {
    a_collection_held_since_the_look_stays_loaded("eviction_door_age", |collections| {
        CollectionManager::evict_idle_by_age(collections, u64::MAX, 1, 2, 0, "idle")
    });
}

// ── A payload index build in the queue ─────────────────────────────────────

/// A payload index build of the collection `name` that stays in the queue,
/// for all that anything but the worker that keeps it can tell, until this
/// is dropped.
struct QueuedBuild(&'static str);

impl QueuedBuild {
    fn of(storage: &Arc<HelixGraphStorage>, name: &'static str) -> Self {
        PAYLOAD_INDEX_JOBS_HELD_FOR_TEST
            .lock()
            .unwrap()
            .push(name.to_string());
        let queued = Self(name);
        let answer = storage
            .enqueue_payload_index_build(name, "path", PayloadIndexSchema::Keyword, false)
            .unwrap();
        assert_eq!(answer.status, "building");
        queued
    }
}

impl Drop for QueuedBuild {
    fn drop(&mut self) {
        PAYLOAD_INDEX_JOBS_HELD_FOR_TEST
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|held| held != self.0);
    }
}

/// One idle collection whose caller has just been told its index is
/// `building`. The job waits in the queue with a `Weak`: closed under it,
/// the collection lost the build, the job was skipped as
/// `collection_dropped`, and the next open knew no such index.
fn a_collection_with_an_index_build_queued_stays_loaded(
    name: &'static str,
    pass: fn(&mut HashMap<String, CachedCollection>) -> u64,
) {
    let writer = Writer::new(name);
    let storage = writer.manager.create_collection(name).unwrap();
    put_row(&storage, b"k1", b"v1");
    wait_until("the open's own job over", || {
        !index_job_queued_or_active(name)
    });
    let queued = QueuedBuild::of(&storage, name);
    drop(storage);

    let evicted = pass(&mut writer.manager.collections.write().unwrap());
    assert_eq!(
        evicted, 0,
        "an eviction took a collection whose index build was queued"
    );
    // And an operator reads it as pinned, though nothing holds its storage.
    let pinned = || CollectionManager::pinned(&writer.manager.collections.read().unwrap());
    assert_eq!(pinned(), 1, "a collection kept for its queued build");

    // A worker takes the job, and the build ends on the collection it was
    // asked of.
    drop(queued);
    wait_until("the build over and the collection idle", || {
        let collections = writer.manager.collections.read().unwrap();
        let storage = &collections[name].storage;
        storage
            .payload_index_status("path")
            .unwrap()
            .is_some_and(|index| index.status == "ready")
            && Arc::strong_count(storage) == 1
            && !index_job_queued_or_active(name)
    });
    assert_eq!(
        pinned(),
        0,
        "a collection with no build queued and no holder"
    );
    // With no build queued the pass takes it.
    let evicted = pass(&mut writer.manager.collections.write().unwrap());
    assert_eq!(evicted, 1, "a collection with no build queued was left");
}

#[test]
fn an_eviction_for_capacity_leaves_a_collection_whose_index_build_is_queued() {
    a_collection_with_an_index_build_queued_stays_loaded(
        "eviction_door_queued_lru",
        |collections| CollectionManager::evict_lru_idle(collections, 1, "lru"),
    );
}

#[test]
fn an_eviction_by_age_leaves_a_collection_whose_index_build_is_queued() {
    a_collection_with_an_index_build_queued_stays_loaded(
        "eviction_door_queued_age",
        |collections| CollectionManager::evict_idle_by_age(collections, u64::MAX, 1, 1, 0, "idle"),
    );
}
