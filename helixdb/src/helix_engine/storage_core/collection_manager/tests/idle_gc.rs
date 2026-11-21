//! The idle GC sweep (`collection_manager_idle_gc.rs`): which collections it
//! collects, which it leaves alone, and that a pass never overlaps an open nor
//! keeps one waiting.

use super::super::idle_gc::{run_idle_gc_sweep, sweep_interval, sweeper_start, SweepTotals};
use super::*;
use crate::helix_engine::storage_core::backend::{Namespace, StorageBackend};
use crate::helix_engine::storage_core::backend_lsm::{
    closed_gc_plan, shared_lsm_handle, ClosedGcPass, LsmBackend,
};
use futures::{stream, StreamExt, TryStreamExt};
use slatedb::object_store::memory::InMemory;
use slatedb::object_store::{ObjectStore, ObjectStoreExt};
use std::collections::BTreeMap;
use std::future::Future;

/// A keyspace nothing in the engine reads, for rows the tests write directly.
pub(super) const ROWS: Namespace<'static> = Namespace::SecondaryIndex("idle_gc_test");

/// Where a collection's metadata sidecar is kept under its prefix.
const SIDECAR_OBJECT: &str = "_helix/metadata.json";

/// The knob that makes a sweep put the sidecars the object store lacks. The
/// fixture leaves it unset.
const PUT_SIDECARS: &str = "HELIX_LSM_IDLE_GC_PUT_METADATA_SIDECARS";

fn object_store_config(role: LsmRole) -> Config {
    test_config().with_storage_backend(StorageBackendConfig::Lsm {
        storage: LsmStorage::ObjectStore,
        role,
    })
}

pub(super) fn put_row(storage: &HelixGraphStorage, key: &[u8], value: &[u8]) {
    let mut txn = storage.backend.begin_write().unwrap();
    storage.backend.put(&mut txn, ROWS, key, value).unwrap();
    storage.backend.commit(txn).unwrap();
}

pub(super) fn get_row(storage: &HelixGraphStorage, key: &[u8]) -> Option<Vec<u8>> {
    let txn = storage.backend.begin_read().unwrap();
    storage
        .backend
        .get_with(&txn, ROWS, key, |value| value.map(<[u8]>::to_vec))
        .unwrap()
}

fn assert_rows(storage: &HelixGraphStorage) {
    assert_eq!(get_row(storage, b"k1").as_deref(), Some(&b"v1"[..]));
    assert_eq!(get_row(storage, b"k2").as_deref(), Some(&b"v2"[..]));
}

/// An object-store writer over its own root prefix, with what no pass runs
/// without: a cache directory and the role spelt out. Its collections collect
/// with a one-second minimum age (and a floor to match, in place of fifteen
/// minutes) and deletion switched on, and the writer's own collector stays out
/// of the way after its tick at open.
pub(super) struct Writer {
    pub(super) manager: Arc<CollectionManager>,
    store: Arc<dyn ObjectStore>,
    pub(super) root: &'static str,
    _dir: TempDir,
    _cache: TempDir,
    _store_guard: backend_any::LsmTestObjectStoreGuard,
    _env: Vec<EnvRestore>,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl Writer {
    pub(super) fn new(root: &'static str) -> Self {
        Self::over(root, Arc::new(InMemory::new()))
    }

    pub(super) fn over(root: &'static str, store: Arc<dyn ObjectStore>) -> Self {
        let lock = ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let cache = TempDir::new().unwrap();
        let env = vec![
            EnvRestore::set("HELIX_LSM_PREFIX", root),
            EnvRestore::set("HELIX_LSM_ROLE", "writer"),
            EnvRestore::set("HELIX_LSM_CACHE_DIR", cache.path().to_str().unwrap()),
            EnvRestore::set("HELIX_LSM_GC_INTERVAL_SECS", "3600"),
            EnvRestore::set("HELIX_LSM_GC_MIN_AGE_SECS", "1"),
            EnvRestore::set("HELIX_LSM_TEST_IDLE_GC_MIN_AGE_FLOOR_SECS", "1"),
            EnvRestore::unset("HELIX_LSM_IDLE_GC_SWEEP_SECS"),
            EnvRestore::set("HELIX_LSM_IDLE_GC_DRY_RUN", "0"),
            EnvRestore::unset("HELIX_LSM_IDLE_GC_DELETE_PREFIXES"),
            EnvRestore::unset(PUT_SIDECARS),
        ];
        let store_guard = backend_any::set_lsm_test_object_store(Arc::clone(&store));
        let dir = TempDir::new().unwrap();
        let manager = Arc::new(
            CollectionManager::new(
                dir.path().to_path_buf(),
                object_store_config(LsmRole::Writer),
            )
            .unwrap(),
        );
        Self {
            manager,
            store,
            root,
            _dir: dir,
            _cache: cache,
            _store_guard: store_guard,
            _env: env,
            _lock: lock,
        }
    }

    /// A closed collection holding rows `k1` and `k2`. It was opened twice, so
    /// it has manifest versions nothing references, and the sleep puts them
    /// past the minimum age.
    fn closed_collection(&self, name: &str) {
        {
            // A writer's own collector ticks at open; under the one-second
            // minimum age the second open could take the first one's manifests.
            let _keep = EnvRestore::set("HELIX_LSM_GC_MIN_AGE_SECS", "3600");
            let storage = self.manager.create_collection(name).unwrap();
            put_row(&storage, b"k1", b"v1");
            drop(storage);
            self.close(name);
            let storage = self.manager.get_collection(name).unwrap();
            put_row(&storage, b"k2", b"v2");
            drop(storage);
            self.close(name);
        }
        thread::sleep(Duration::from_millis(1_200));
    }

    /// Evict `name` and wait until its `Db` is closed.
    pub(super) fn close(&self, name: &str) {
        self.manager.evict_collection(name);
        let deadline = Instant::now() + Duration::from_secs(30);
        while self
            .manager
            .closing
            .read()
            .unwrap()
            .get(name)
            .is_some_and(|weak| weak.strong_count() > 0)
        {
            assert!(Instant::now() < deadline, "{name} never closed");
            thread::sleep(Duration::from_millis(20));
        }
    }

    /// Every object under the root: location to e_tag. The e_tag changes when
    /// an object is rewritten, so equal maps mean nothing was written either.
    pub(super) fn objects(&self) -> BTreeMap<String, Option<String>> {
        let store = Arc::clone(&self.store);
        let root = ObjPath::from(self.root);
        shared_lsm_handle().block_on(async move {
            store
                .list(Some(&root))
                .map_ok(|meta| (meta.location.to_string(), meta.e_tag))
                .try_collect()
                .await
                .unwrap()
        })
    }

    fn manifest_ids(&self, name: &str) -> Vec<u64> {
        let prefix = format!("{}/{name}/manifest/", self.root);
        self.objects()
            .keys()
            .filter_map(|location| {
                location
                    .strip_prefix(&prefix)?
                    .strip_suffix(".manifest")?
                    .parse()
                    .ok()
            })
            .collect()
    }

    fn sidecar_object(&self, name: &str) -> ObjPath {
        ObjPath::from(format!("{}/{name}/{SIDECAR_OBJECT}", self.root))
    }

    /// The collection's metadata sidecar in the object store, if one is there.
    fn sidecar(&self, name: &str) -> Option<Vec<u8>> {
        let (store, object) = (Arc::clone(&self.store), self.sidecar_object(name));
        shared_lsm_handle().block_on(async move {
            match store.get(&object).await {
                Ok(found) => Some(found.bytes().await.unwrap().to_vec()),
                Err(object_store::Error::NotFound { .. }) => None,
                Err(error) => panic!("{error}"),
            }
        })
    }

    /// Takes the collection's metadata sidecar out of the object store, which
    /// leaves the collection as one the writer has not opened since sidecars
    /// are kept there, and hands back the writer's local copy.
    fn take_sidecar_object(&self, name: &str) -> Vec<u8> {
        let (store, object) = (Arc::clone(&self.store), self.sidecar_object(name));
        shared_lsm_handle()
            .block_on(async move { store.delete(&object).await })
            .unwrap();
        fs::read(HelixGraphStorage::metadata_sidecar_path(
            &self.manager.collection_path(name),
        ))
        .unwrap()
    }
}

/// One sweep over the writer's catalog, starting anywhere.
fn sweep(writer: &Writer) -> SweepTotals {
    run_idle_gc_sweep(&Arc::downgrade(&writer.manager), &mut None).unwrap()
}

/// With the dry-run knob on `prefixes`, one sweep deletes in the collection
/// the delete prefixes cover and makes a dry run of the other: its candidates
/// are counted as before and every object of it stays as it was. Without the
/// prefixes it deletes nowhere. An off value on the knob deletes everywhere,
/// and nowhere while the prefixes are still set.
#[test]
fn sweep_deletes_only_in_the_collections_the_delete_prefixes_cover() {
    const KNOB: &str = "HELIX_LSM_IDLE_GC_DRY_RUN";
    const PREFIXES: &str = "HELIX_LSM_IDLE_GC_DELETE_PREFIXES";
    let writer = Writer::new("idle_gc_prefixes");
    writer.closed_collection("admin_quiet");
    writer.closed_collection("other_quiet");
    // What a dry run finds in each, which is what a real pass deletes.
    let found = |name: &str| {
        let _dry_run = EnvRestore::unset(KNOB);
        let (label, pass) = writer.manager.gc_collection_if_idle(name);
        assert_eq!(label, "collected");
        assert!(
            pass.dry_run && pass.candidates > 0 && pass.objects == 0,
            "{pass:?}"
        );
        pass
    };
    let (admin, other) = (found("admin_quiet"), found("other_quiet"));
    let objects_of = |name: &str| -> BTreeMap<String, Option<String>> {
        let prefix = format!("{}/{name}/", writer.root);
        writer
            .objects()
            .into_iter()
            .filter(|(location, _)| location.starts_with(&prefix))
            .collect()
    };
    let before = writer.objects();
    let other_before = objects_of("other_quiet");

    {
        let _canary = EnvRestore::set(KNOB, "prefixes");
        // No prefixes: no collection is covered.
        let totals = sweep(&writer);
        assert_eq!(writer.objects(), before);
        assert_eq!(
            (totals.deleted, totals.dry_run_candidates),
            (0, admin.candidates + other.candidates)
        );

        let _admin_only = EnvRestore::set(PREFIXES, "admin_");
        let totals = sweep(&writer);
        assert_eq!(writer.manifest_ids("admin_quiet").len(), 1);
        assert_eq!(objects_of("other_quiet"), other_before);
        assert_eq!(
            totals,
            SweepTotals {
                candidates: admin.candidates + other.candidates,
                candidate_bytes: admin.candidate_bytes + other.candidate_bytes,
                deleted: admin.candidates,
                deleted_bytes: admin.candidate_bytes,
                deletes_not_started: 0,
                dry_run_candidates: other.candidates,
                dry_run_candidate_bytes: other.candidate_bytes,
            }
        );
    }

    // The writer's own value is an off one. Next to prefixes left in the env
    // it deletes nowhere; alone, the other collection is collected too.
    {
        let _left_over = EnvRestore::set(PREFIXES, "admin_");
        let totals = sweep(&writer);
        assert_eq!(objects_of("other_quiet"), other_before);
        assert_eq!(
            (totals.deleted, totals.dry_run_candidates),
            (0, totals.candidates)
        );
    }
    let totals = sweep(&writer);
    assert_eq!(writer.manifest_ids("other_quiet").len(), 1);
    assert!(
        totals.deleted >= other.candidates && totals.dry_run_candidates == 0,
        "{totals:?}"
    );
    for name in ["admin_quiet", "other_quiet"] {
        assert_rows(&writer.manager.get_collection(name).unwrap());
    }
}

#[test]
fn sweep_collects_a_closed_collection_without_opening_it() {
    let writer = Writer::new("idle_gc_collects");
    writer.closed_collection("idle_gc_quiet");
    let before = writer.manifest_ids("idle_gc_quiet");
    assert!(before.len() > 1, "no stale manifest to collect: {before:?}");

    // Until deletion is switched on, a sweep only looks: it leaves every
    // object as it was and adds none.
    {
        let _dry_run = EnvRestore::unset("HELIX_LSM_IDLE_GC_DRY_RUN");
        let objects = writer.objects();
        sweep(&writer);
        assert_eq!(writer.objects(), objects);
    }

    sweep(&writer);

    // Only the latest manifest is left, and it is the one the collection was
    // closed on: an open would have written a newer one to claim the writer
    // epoch.
    assert_eq!(
        writer.manifest_ids("idle_gc_quiet"),
        vec![*before.last().unwrap()]
    );
    assert_eq!(writer.manager.loaded_count(), 0);
    assert_rows(&writer.manager.get_collection("idle_gc_quiet").unwrap());
}

#[test]
fn loaded_or_still_referenced_collection_is_skipped() {
    let writer = Writer::new("idle_gc_loaded");
    writer.closed_collection("idle_gc_busy");
    let storage = writer.manager.get_collection("idle_gc_busy").unwrap();
    assert_eq!(
        writer.manager.gc_collection_if_idle("idle_gc_busy"),
        ("loaded", ClosedGcPass::default())
    );

    // Evicted, but a request still holds the handle: its `Db` is open.
    writer.manager.evict_collection("idle_gc_busy");
    assert_eq!(
        writer.manager.gc_collection_if_idle("idle_gc_busy"),
        ("loaded", ClosedGcPass::default())
    );

    // Closed, it is still a collection the writer had moments ago: under an
    // hour's minimum age, written too recently.
    drop(storage);
    writer.close("idle_gc_busy");
    {
        let _hour = EnvRestore::set("HELIX_LSM_GC_MIN_AGE_SECS", "3600");
        assert_eq!(
            writer.manager.gc_collection_if_idle("idle_gc_busy"),
            ("recently_written", ClosedGcPass::default())
        );
    }
    // The open wrote to its WAL. Under the fixture's one second that is past.
    thread::sleep(Duration::from_millis(1_200));
    assert_eq!(
        writer.manager.gc_collection_if_idle("idle_gc_busy").0,
        "collected"
    );
}

#[test]
fn dropped_collection_is_skipped_and_nothing_lands_under_its_prefix() {
    let writer = Writer::new("idle_gc_dropped");
    writer.closed_collection("idle_gc_gone");
    let path = writer.manager.collection_path("idle_gc_gone");
    let backend = writer.manager.config.storage_backend;
    let left_alone = |reason: &'static str| {
        let before = writer.objects();
        assert_eq!(
            writer.manager.gc_collection_if_idle("idle_gc_gone"),
            (reason, ClosedGcPass::default())
        );
        sweep(&writer);
        assert_eq!(writer.objects(), before, "{reason}");
    };

    // What a drop leaves in the bucket, and all a restarted writer knows of it.
    backend_any::persist_lsm_drop_tombstone_from_env(&path, backend).unwrap();
    left_alone("tombstoned");
    backend_any::clear_lsm_drop_tombstone_from_env(&path, backend).unwrap();
    {
        let _unreadable = EnvRestore::set("HELIX_LSM_TEST_FAIL_TOMBSTONE_READ", "1");
        left_alone("error");
    }

    // What a drop holds in this process until its purge is confirmed.
    writer.manager.mark_dropping("idle_gc_gone");
    left_alone("dropping");
    writer.manager.clear_dropping("idle_gc_gone");
    lsm_purge_pending_register(&path);
    left_alone("dropping");
    lsm_purge_pending_clear(&path);

    // Nothing but those markers held the pass back.
    let (label, pass) = writer.manager.gc_collection_if_idle("idle_gc_gone");
    assert_eq!(label, "collected");
    assert!(pass.objects > 0);
}

#[test]
fn collection_that_failed_to_open_or_is_degraded_keeps_every_object() {
    let writer = Writer::new("idle_gc_quarantine");
    let name = "idle_gc_suspect";
    writer.closed_collection(name);
    let left_alone = || {
        let before = writer.objects();
        assert_eq!(
            writer.manager.gc_collection_if_idle(name),
            ("quarantined", ClosedGcPass::default())
        );
        assert_eq!(writer.objects(), before);
    };

    open_failure_record_quarantine(name, &GraphError::New("injected open failure".into()));
    left_alone();
    open_failure_clear(name);

    let marker = writer
        .manager
        .collection_path(name)
        .join(".helix_collection_degraded");
    fs::create_dir_all(marker.parent().unwrap()).unwrap();
    fs::write(&marker, "storage_corruption\ninjected\n").unwrap();
    left_alone();
    fs::remove_file(&marker).unwrap();

    assert_eq!(writer.manager.gc_collection_if_idle(name).0, "collected");
}

#[test]
fn names_that_are_not_canonical_path_segments_are_never_swept() {
    let writer = Writer::new("idle_gc_names");
    for name in ["", ".", "..", "a/b", "hash#tag", "with%20escape"] {
        assert_eq!(
            writer.manager.gc_collection_if_idle(name),
            ("unsupported_name", ClosedGcPass::default()),
            "{name:?}"
        );
    }
    // The root graph engine's own database is named after the data directory.
    let data_dir = writer._dir.path().file_name().unwrap().to_str().unwrap();
    assert_eq!(
        writer.manager.gc_collection_if_idle(data_dir),
        ("root_engine", ClosedGcPass::default())
    );
    assert!(writer.manager.open_gates.read().unwrap().is_empty());
}

#[test]
fn sweep_interval_is_a_positive_whole_number_of_seconds_or_none() {
    assert_eq!(sweep_interval(None), Err("unset"));
    for value in ["0", "00", " 0 "] {
        assert_eq!(sweep_interval(Some(value)), Err("zero"), "{value:?}");
    }
    for value in [
        "",
        " ",
        "off",
        "false",
        "6h",
        "30s",
        "-30",
        "1.5",
        "0x10",
        "99999999999999999999",
    ] {
        assert_eq!(sweep_interval(Some(value)), Err("unparsable"), "{value:?}");
    }
    assert_eq!(sweep_interval(Some("30")), Ok(Duration::from_secs(30)));
    assert_eq!(
        sweep_interval(Some(" 21600 ")),
        Ok(Duration::from_secs(21_600))
    );
}

#[test]
fn sweeper_needs_the_interval_knob_and_a_pass_that_can_run() {
    let _writer = Writer::new("idle_gc_start");
    let plan = closed_gc_plan().unwrap();
    for role in ["writer", " Writer ", "WRITER"] {
        assert_eq!(
            sweeper_start(Some("30"), Some(role), || Ok(plan)),
            Ok((Duration::from_secs(30), plan)),
            "{role:?}"
        );
    }
    // Asked for, but a pass could not run safely.
    for reason in [
        "cache_dir_unset",
        "min_age_unset",
        "min_age_under_900s",
        "quiet_under_900s",
    ] {
        assert_eq!(
            sweeper_start(Some("30"), Some("writer"), || Err(reason)),
            Err(reason)
        );
    }
    // Asked for on a pod that is a writer only by default, or by a typo:
    // whether a pass could run is not even looked at.
    for role in [
        None,
        Some(""),
        Some("reader"),
        Some("primary"),
        Some("writer-0"),
    ] {
        assert_eq!(
            sweeper_start(Some("30"), role, || panic!("the plan was consulted")),
            Err("role_not_writer"),
            "{role:?}"
        );
    }
    // Not asked for: nothing else is looked at.
    for (value, reason) in [
        (None, "unset"),
        (Some("0"), "zero"),
        (Some("soon"), "unparsable"),
    ] {
        assert_eq!(
            sweeper_start(value, None, || panic!("the plan was consulted")),
            Err(reason),
            "{value:?}"
        );
    }
}

#[test]
fn sweeper_starts_only_on_an_object_store_writer_with_a_positive_interval() {
    let writer = Writer::new("idle_gc_spawn");
    // Unset, zero, empty or not a number: no thread.
    assert!(spawn_idle_gc_sweeper(&writer.manager).is_none());
    for value in ["0", "", "soon", "-1", "1.5"] {
        let _interval = EnvRestore::set("HELIX_LSM_IDLE_GC_SWEEP_SECS", value);
        assert!(
            spawn_idle_gc_sweeper(&writer.manager).is_none(),
            "{value:?}"
        );
    }

    let _interval = EnvRestore::set("HELIX_LSM_IDLE_GC_SWEEP_SECS", "1");
    // Without a cache directory a pass cannot see a handle that is still open.
    // Without a minimum age from the env it would apply SlateDB's own, and the
    // fixture's one second is under the floor once the test's own is gone.
    // Without a role, this is a writer only by default.
    for knob in [
        "HELIX_LSM_CACHE_DIR",
        "HELIX_LSM_GC_MIN_AGE_SECS",
        "HELIX_LSM_TEST_IDLE_GC_MIN_AGE_FLOOR_SECS",
        "HELIX_LSM_ROLE",
    ] {
        let _unset = EnvRestore::unset(knob);
        assert!(spawn_idle_gc_sweeper(&writer.manager).is_none(), "{knob}");
    }
    for role in ["", "reader", "primary", "writer-0"] {
        let _role = EnvRestore::set("HELIX_LSM_ROLE", role);
        assert!(spawn_idle_gc_sweeper(&writer.manager).is_none(), "{role:?}");
    }
    for config in [
        object_store_config(LsmRole::Reader),
        test_config().with_lsm_in_memory(),
        test_config(),
    ] {
        let dir = TempDir::new().unwrap();
        let manager = Arc::new(CollectionManager::new(dir.path().to_path_buf(), config).unwrap());
        assert!(spawn_idle_gc_sweeper(&manager).is_none());
    }

    // Spelt any way a role is read.
    let _role = EnvRestore::set("HELIX_LSM_ROLE", " Writer ");
    writer.closed_collection("idle_gc_swept");
    let latest = *writer.manifest_ids("idle_gc_swept").last().unwrap();
    let sweeper = spawn_idle_gc_sweeper(&writer.manager).expect("an object-store writer sweeps");
    let deadline = Instant::now() + Duration::from_secs(30);
    while writer.manifest_ids("idle_gc_swept") != vec![latest] {
        assert!(Instant::now() < deadline, "the sweeper never collected");
        thread::sleep(Duration::from_millis(100));
    }

    // The thread holds the manager weakly and ends once it is gone.
    drop(writer.manager);
    let deadline = Instant::now() + Duration::from_secs(30);
    while !sweeper.is_finished() {
        assert!(
            Instant::now() < deadline,
            "the sweeper outlived its manager"
        );
        thread::sleep(Duration::from_millis(50));
    }
    sweeper.join().unwrap();
}

#[test]
fn pass_stays_out_while_the_open_gate_is_held() {
    let writer = Writer::new("idle_gc_gate");
    writer.closed_collection("idle_gc_held");
    let gate = writer.manager.open_gate("idle_gc_held");
    let before = writer.objects();
    {
        // What a create, a cold open, a drop or a purge holds while it runs.
        let _lifecycle = gate.lock().unwrap();
        assert_eq!(
            writer.manager.gc_collection_if_idle("idle_gc_held"),
            ("busy", ClosedGcPass::default())
        );
    }
    // What one of them holds from the moment it asks for the gate: the pass
    // gives way before it has read anything.
    assert_eq!(
        writer.manager.gc_collection_if_idle("idle_gc_held"),
        (
            "yielded",
            ClosedGcPass {
                yielded: true,
                ..ClosedGcPass::default()
            }
        )
    );
    assert_eq!(writer.objects(), before);
    drop(gate);

    let (label, pass) = writer.manager.gc_collection_if_idle("idle_gc_held");
    assert_eq!(label, "collected");
    assert!(pass.objects > 0);
    // A sweep does not leave a gate behind for every collection it visits.
    assert!(!writer
        .manager
        .open_gates
        .read()
        .unwrap()
        .contains_key("idle_gc_held"));
}

/// The calls a [`Paused`] store holds back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Hold {
    Deletes,
    Listings,
    /// The HEAD of a metadata sidecar.
    SidecarLooks,
    /// The put of a metadata sidecar.
    SidecarPuts,
}

/// Object store whose deletes, listings, sidecar HEADs or sidecar puts, once
/// `armed`, wait until `release` is closed. Clones share the flags, so a test
/// keeps one to steer the store it gave the writer.
///
/// It also counts the HEADs and the puts of a metadata sidecar it is asked
/// for, and while `sidecar_hidden` is set a HEAD of one finds nothing: the
/// look of a put that somebody else's put then beats. And it fails the next
/// `failing_sidecar_puts` puts of one.
#[derive(Debug, Clone)]
struct Paused {
    inner: Arc<dyn ObjectStore>,
    holds: Hold,
    armed: Arc<AtomicBool>,
    reached: Arc<AtomicBool>,
    release: Arc<tokio::sync::Semaphore>,
    sidecar_heads: Arc<std::sync::atomic::AtomicUsize>,
    sidecar_puts: Arc<std::sync::atomic::AtomicUsize>,
    sidecar_hidden: Arc<AtomicBool>,
    failing_sidecar_puts: Arc<std::sync::atomic::AtomicUsize>,
}

impl Paused {
    fn new(holds: Hold) -> Self {
        Self {
            inner: Arc::new(InMemory::new()),
            holds,
            armed: Arc::default(),
            reached: Arc::default(),
            release: Arc::new(tokio::sync::Semaphore::new(0)),
            sidecar_heads: Arc::default(),
            sidecar_puts: Arc::default(),
            sidecar_hidden: Arc::default(),
            failing_sidecar_puts: Arc::default(),
        }
    }

    fn parks(&self, call: Hold) -> bool {
        self.holds == call && self.armed.load(Ordering::SeqCst)
    }

    /// Wait until a call is parked.
    fn wait_until_reached(&self) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while !self.reached.load(Ordering::SeqCst) {
            assert!(
                Instant::now() < deadline,
                "the pass never reached one of the {:?}",
                self.holds
            );
            thread::sleep(Duration::from_millis(10));
        }
    }
}

impl std::fmt::Display for Paused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Paused({})", self.inner)
    }
}

type Boxed<'a, T> = std::pin::Pin<Box<dyn Future<Output = object_store::Result<T>> + Send + 'a>>;

impl ObjectStore for Paused {
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
        if location.as_ref().ends_with(SIDECAR_OBJECT) {
            self.sidecar_puts.fetch_add(1, Ordering::SeqCst);
            let fails = self.failing_sidecar_puts.fetch_update(
                Ordering::SeqCst,
                Ordering::SeqCst,
                |left| left.checked_sub(1),
            );
            if fails.is_ok() {
                // Two texts that quarantine a collection where an error of
                // its own store carries one.
                return Box::pin(async {
                    Err(object_store::Error::Generic {
                        store: "Paused",
                        source: "a sidecar put this test fails: NoSuchKey, checksum".into(),
                    })
                });
            }
            if self.parks(Hold::SidecarPuts) {
                return Box::pin(async move {
                    self.reached.store(true, Ordering::SeqCst);
                    let _ = self.release.acquire().await;
                    self.inner.put_opts(location, payload, opts).await
                });
            }
        }
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
        if options.head && location.as_ref().ends_with(SIDECAR_OBJECT) {
            self.sidecar_heads.fetch_add(1, Ordering::SeqCst);
            if self.sidecar_hidden.load(Ordering::SeqCst) {
                let hidden = object_store::Error::NotFound {
                    path: location.to_string(),
                    source: "hidden from this HEAD".into(),
                };
                return Box::pin(async move { Err(hidden) });
            }
            if self.parks(Hold::SidecarLooks) {
                return Box::pin(async move {
                    self.reached.store(true, Ordering::SeqCst);
                    let _ = self.release.acquire().await;
                    self.inner.get_opts(location, options).await
                });
            }
        }
        self.inner.get_opts(location, options)
    }

    fn delete_stream(
        &self,
        locations: stream::BoxStream<'static, object_store::Result<ObjPath>>,
    ) -> stream::BoxStream<'static, object_store::Result<ObjPath>> {
        if !self.parks(Hold::Deletes) {
            return self.inner.delete_stream(locations);
        }
        let inner = Arc::clone(&self.inner);
        let reached = Arc::clone(&self.reached);
        let release = Arc::clone(&self.release);
        stream::once(async move {
            reached.store(true, Ordering::SeqCst);
            // Errors once the test closes the semaphore; that is the release.
            let _ = release.acquire().await;
            inner.delete_stream(locations)
        })
        .flatten()
        .boxed()
    }

    fn list(
        &self,
        prefix: Option<&ObjPath>,
    ) -> stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        let listing = self.inner.list(prefix);
        if !self.parks(Hold::Listings) {
            return listing;
        }
        let reached = Arc::clone(&self.reached);
        let release = Arc::clone(&self.release);
        stream::once(async move {
            reached.store(true, Ordering::SeqCst);
            let _ = release.acquire().await;
            listing
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

/// Lets the paused calls through even when the test fails first, so a
/// failure is reported instead of hanging on the parked pass.
struct ReleaseOnDrop(Arc<tokio::sync::Semaphore>);

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.0.close();
    }
}

#[test]
fn open_waits_for_the_delete_a_pass_has_in_flight_and_for_no_more() {
    let store = Paused::new(Hold::Deletes);
    let writer = Writer::over("idle_gc_overlap", Arc::new(store.clone()));
    writer.closed_collection("idle_gc_raced");
    let manager = &writer.manager;
    let opened = AtomicBool::new(false);

    store.armed.store(true, Ordering::SeqCst);
    thread::scope(|scope| {
        let release = ReleaseOnDrop(Arc::clone(&store.release));
        let pass = scope.spawn(|| manager.gc_collection_if_idle("idle_gc_raced"));
        store.wait_until_reached();

        let open = scope.spawn(|| {
            let storage = manager
                .get_collection("idle_gc_raced")
                .map_err(|error| error.to_string());
            opened.store(true, Ordering::SeqCst);
            storage
        });
        // The pass is parked mid-delete with the gate held. A delete on the
        // wire is not abandoned: landing later, it could hit whatever the
        // prefix holds by then. However long it takes, the open must not get
        // past it.
        thread::sleep(Duration::from_millis(300));
        assert!(
            !opened.load(Ordering::SeqCst),
            "an open overlapped a pass in flight"
        );
        assert!(!pass.is_finished());

        // Once those deletes are through, the pass starts nothing more: the
        // open is waiting.
        drop(release);
        let (label, pass) = pass.join().unwrap();
        assert_eq!(label, "yielded");
        assert!(pass.yielded && pass.objects > 0);
        assert_rows(&open.join().unwrap().unwrap());
    });
}

#[test]
fn pass_waiting_on_the_object_store_gives_the_gate_to_an_open() {
    let store = Paused::new(Hold::Listings);
    let writer = Writer::over("idle_gc_yield", Arc::new(store.clone()));
    writer.closed_collection("idle_gc_wanted");
    let manager = &writer.manager;

    store.armed.store(true, Ordering::SeqCst);
    thread::scope(|scope| {
        let _release = ReleaseOnDrop(Arc::clone(&store.release));
        let pass = scope.spawn(|| manager.gc_collection_if_idle("idle_gc_wanted"));
        store.wait_until_reached();
        // Only the listing the pass is in stays parked; the open's own go
        // through.
        store.armed.store(false, Ordering::SeqCst);

        let open = scope.spawn(|| {
            manager
                .get_collection("idle_gc_wanted")
                .map_err(|error| error.to_string())
        });
        // That listing never answers. The pass gives it up when the open asks
        // for the gate, having written and deleted nothing.
        let deadline = Instant::now() + Duration::from_secs(30);
        while !pass.is_finished() {
            assert!(
                Instant::now() < deadline,
                "the pass kept the gate from an open"
            );
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(
            pass.join().unwrap(),
            (
                "yielded",
                ClosedGcPass {
                    yielded: true,
                    ..ClosedGcPass::default()
                }
            )
        );
        assert_rows(&open.join().unwrap().unwrap());
    });
}

/// Not a GC test; it is here for the store that parks deletes. A drop keeps
/// the collection's gate until its purge is through, and a request for that
/// collection must not wait there: it holds its caller's blocking slot.
#[test]
fn request_for_a_collection_being_dropped_does_not_wait_for_the_drop() {
    let store = Paused::new(Hold::Deletes);
    let writer = Writer::over("drop_in_flight", Arc::new(store.clone()));
    let manager = &writer.manager;
    // Both closed: each request below has to open, which asks for the gate.
    for name in ["drop_held", "drop_bystander"] {
        let storage = manager.create_collection(name).unwrap();
        put_row(&storage, b"k1", b"v1");
        put_row(&storage, b"k2", b"v2");
        drop(storage);
        writer.close(name);
    }

    store.armed.store(true, Ordering::SeqCst);
    thread::scope(|scope| {
        let release = ReleaseOnDrop(Arc::clone(&store.release));
        let dropping = scope.spawn(|| manager.drop_collection("drop_held"));
        // The drop is parked in its purge: tombstone written, gate held.
        store.wait_until_reached();
        // Only that delete stays parked; an open's own calls go through.
        store.armed.store(false, Ordering::SeqCst);

        let request = scope.spawn(|| {
            manager
                .get_collection("drop_held")
                .map(drop)
                .map_err(|error| error.to_string())
        });
        let deadline = Instant::now() + Duration::from_secs(10);
        while !request.is_finished() {
            assert!(
                Instant::now() < deadline,
                "a request for a collection being dropped waited for the drop"
            );
            thread::sleep(Duration::from_millis(10));
        }
        let error = request.join().unwrap().unwrap_err();
        assert!(error.contains("not found (drop in progress)"), "{error}");
        // Another collection opens and reads meanwhile.
        assert_rows(&manager.get_collection("drop_bystander").unwrap());
        assert!(!dropping.is_finished());

        drop(release);
        dropping.join().unwrap().unwrap();
    });
    assert!(!manager.is_dropping("drop_held"));
    let error = manager.get_collection("drop_held").map(drop).unwrap_err();
    assert!(error.to_string().contains("not found"), "{error}");
}

#[test]
fn sweep_that_ends_early_starts_the_next_one_after_the_collection_it_reached() {
    let writer = Writer::new("idle_gc_cursor");
    let names = ["idle_gc_a", "idle_gc_b", "idle_gc_c"];
    for name in names {
        drop(writer.manager.create_collection(name).unwrap());
        writer.close(name);
    }
    let manager = Arc::downgrade(&writer.manager);
    // Sorts before every name: the first sweep starts with the first one.
    let mut reached = Some(String::new());

    // Every pass fails outright, so each sweep ends after two of them. The
    // next one goes on from there and around the end of the catalog: no
    // collection waits behind others that keep failing.
    {
        let _unreadable = EnvRestore::set("HELIX_LSM_TEST_FAIL_TOMBSTONE_READ", "1");
        for last in ["idle_gc_b", "idle_gc_a", "idle_gc_c"] {
            assert!(run_idle_gc_sweep(&manager, &mut reached).is_some());
            assert_eq!(reached.as_deref(), Some(last));
        }
    }

    // A pass that gives way has not failed. With every collection wanted, the
    // sweep still visits all three.
    let wanted: Vec<_> = names
        .iter()
        .map(|name| writer.manager.open_gate(name))
        .collect();
    reached = Some("idle_gc_a".to_string());
    assert!(run_idle_gc_sweep(&manager, &mut reached).is_some());
    assert_eq!(reached.as_deref(), Some("idle_gc_a"));
    drop(wanted);
}

/// A collection the writer has not opened since sidecars are kept in the
/// object store has its metadata in the writer's local copy alone, and a
/// reader answers the info route of a collection it has not loaded from the
/// object. Where its knob asks for it, the sweep puts the local copy there,
/// once: the next one finds it and puts nothing. Without the knob, and in a
/// dry run, a sweep does not ask the object store about a sidecar at all.
#[test]
fn sweep_puts_the_metadata_sidecar_a_closed_collection_lacks_and_puts_it_once() {
    let store = Paused::new(Hold::Deletes);
    let writer = Writer::over("idle_gc_sidecar", Arc::new(store.clone()));
    let name = "idle_gc_untouched";
    writer.closed_collection(name);
    let local = writer.take_sidecar_object(name);
    let heads = || store.sidecar_heads.load(Ordering::SeqCst);
    let puts = || store.sidecar_puts.load(Ordering::SeqCst);
    let (no_head, before) = (heads(), puts());
    let nothing_asked = |why: &str| {
        let (label, pass) = writer.manager.gc_collection_if_idle(name);
        assert_eq!((label, pass.metadata_sidecar), ("collected", None), "{why}");
        sweep(&writer);
        assert_eq!(
            (writer.sidecar(name), heads(), puts()),
            (None, no_head, before),
            "{why}"
        );
    };

    nothing_asked("the knob is unset");
    for not_a_yes in ["0", "", "sidecars"] {
        let _knob = EnvRestore::set(PUT_SIDECARS, not_a_yes);
        nothing_asked(not_a_yes);
    }
    let _knob = EnvRestore::set(PUT_SIDECARS, "1");
    {
        let _dry_run = EnvRestore::unset("HELIX_LSM_IDLE_GC_DRY_RUN");
        nothing_asked("a dry run");
    }

    // Two sweeps, counted by a recorder of their own.
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let counted = recorder.handle();
    let counts = || {
        let mut counts: Vec<String> = counted
            .render()
            .lines()
            .filter(|line| line.starts_with("helix_lsm_idle_gc_metadata_sidecars_total{"))
            .map(str::to_string)
            .collect();
        counts.sort();
        counts
    };

    metrics::with_local_recorder(&recorder, || sweep(&writer));
    // The bytes the writer's own info route answers from, and a sidecar.
    assert_eq!(writer.sidecar(name).as_deref(), Some(&local[..]));
    sonic_rs::from_slice::<StorageMetadataSidecar>(&local).unwrap();
    assert_eq!((heads(), puts()), (no_head + 1, before + 1));
    assert_eq!(
        counts(),
        [r#"helix_lsm_idle_gc_metadata_sidecars_total{outcome="uploaded"} 1"#]
    );

    metrics::with_local_recorder(&recorder, || sweep(&writer));
    assert_eq!(puts(), before + 1, "a sidecar that was there was put again");
    assert_eq!(
        counts(),
        [
            r#"helix_lsm_idle_gc_metadata_sidecars_total{outcome="present"} 1"#,
            r#"helix_lsm_idle_gc_metadata_sidecars_total{outcome="uploaded"} 1"#,
        ]
    );
    assert_eq!(writer.manager.loaded_count(), 0);
}

/// With deletes limited to name prefixes, a pass over any other collection is
/// a dry run. The sweep puts a sidecar only where a pass may delete, and does
/// not ask the object store about the other collection's.
#[test]
fn sweep_limited_to_name_prefixes_puts_a_sidecar_only_where_a_pass_may_delete() {
    let store = Paused::new(Hold::Deletes);
    let writer = Writer::over("idle_gc_sidecar_prefixes", Arc::new(store.clone()));
    for name in ["admin_quiet", "other_quiet"] {
        writer.closed_collection(name);
        writer.take_sidecar_object(name);
    }
    let _knob = EnvRestore::set(PUT_SIDECARS, "1");
    let _prefixes = EnvRestore::set("HELIX_LSM_IDLE_GC_DRY_RUN", "prefixes");
    let _admin_only = EnvRestore::set("HELIX_LSM_IDLE_GC_DELETE_PREFIXES", "admin_");
    let heads = store.sidecar_heads.load(Ordering::SeqCst);

    sweep(&writer);

    assert!(
        writer.sidecar("admin_quiet").is_some(),
        "the collection the prefixes cover got no sidecar"
    );
    assert_eq!(
        writer.sidecar("other_quiet"),
        None,
        "a collection whose pass is a dry run got a sidecar"
    );
    assert_eq!(store.sidecar_heads.load(Ordering::SeqCst), heads + 1);
}

/// The smallest collections have a manifest that names no SST. A pass has
/// nothing to collect there and says `empty`; the sidecar is put all the same.
#[test]
fn collection_with_nothing_to_collect_gets_its_missing_metadata_sidecar_too() {
    use crate::helix_engine::storage_core::metadata::StorageMetadata;

    let writer = Writer::new("idle_gc_sidecar_empty");
    let _knob = EnvRestore::set(PUT_SIDECARS, "1");
    let name = "idle_gc_bare";
    // A database nothing was written to, and the writer's local copy of its
    // metadata.
    LsmBackend::open_with_store(
        &format!("{}/{name}", writer.root),
        Arc::clone(&writer.store),
    )
    .unwrap()
    .close()
    .unwrap();
    let path = writer.manager.collection_path(name);
    HelixGraphStorage::persist_metadata_sidecar_at_path(
        &path,
        &StorageMetadata::new(Vec::new()),
        None,
        None,
    )
    .unwrap();

    let (label, pass) = writer.manager.gc_collection_if_idle(name);
    assert_eq!((label, pass.metadata_sidecar), ("empty", Some("uploaded")));
    assert_eq!(
        writer.sidecar(name),
        Some(fs::read(HelixGraphStorage::metadata_sidecar_path(&path)).unwrap())
    );
}

/// Without the writer's local copy there is nothing to put, and a copy that
/// does not read as a sidecar is not handed to the readers.
#[test]
fn sweep_puts_no_metadata_sidecar_without_a_local_copy_that_reads() {
    let writer = Writer::new("idle_gc_sidecar_local");
    let _knob = EnvRestore::set(PUT_SIDECARS, "1");
    let name = "idle_gc_no_copy";
    writer.closed_collection(name);
    writer.take_sidecar_object(name);
    let local = HelixGraphStorage::metadata_sidecar_path(&writer.manager.collection_path(name));
    let outcome = || {
        writer
            .manager
            .gc_collection_if_idle(name)
            .1
            .metadata_sidecar
    };

    fs::write(&local, b"{").unwrap();
    assert_eq!((outcome(), writer.sidecar(name)), (Some("failed"), None));
    fs::remove_file(&local).unwrap();
    assert_eq!(
        (outcome(), writer.sidecar(name)),
        (Some("no_local_copy"), None)
    );
}

/// A name that is dropped, or whose prefix a drop has purged, gets no sidecar
/// even where the writer's local copy outlived the drop: an object under the
/// prefix would list the name in the catalog again.
#[test]
fn dropped_or_purged_name_gets_no_metadata_sidecar() {
    let writer = Writer::new("idle_gc_sidecar_dropped");
    let _knob = EnvRestore::set(PUT_SIDECARS, "1");
    let name = "idle_gc_gone";
    writer.closed_collection(name);
    writer.take_sidecar_object(name);
    let path = writer.manager.collection_path(name);
    let backend = writer.manager.config.storage_backend;
    let none_put = |reason: &'static str| {
        let before = writer.objects();
        assert_eq!(
            writer.manager.gc_collection_if_idle(name),
            (reason, ClosedGcPass::default())
        );
        sweep(&writer);
        assert_eq!(writer.objects(), before, "{reason}");
    };

    backend_any::persist_lsm_drop_tombstone_from_env(&path, backend).unwrap();
    none_put("tombstoned");
    backend_any::clear_lsm_drop_tombstone_from_env(&path, backend).unwrap();
    writer.manager.mark_dropping(name);
    none_put("dropping");
    writer.manager.clear_dropping(name);
    lsm_purge_pending_register(&path);
    none_put("dropping");
    lsm_purge_pending_clear(&path);

    // The purge is through and the tombstone cleared, and the local copy is
    // still on the writer's disk, as after a removal that failed.
    backend_any::purge_lsm_prefix_from_env(&path, backend).unwrap();
    assert!(HelixGraphStorage::metadata_sidecar_path(&path).exists());
    none_put("no_manifest");
    assert!(writer.objects().is_empty());
}

/// A put that finds a sidecar there after its look found none has lost to
/// whoever put that one: theirs stays, and it is no failure.
#[test]
fn metadata_sidecar_put_that_loses_to_another_leaves_theirs() {
    let store = Paused::new(Hold::Deletes);
    let writer = Writer::over("idle_gc_sidecar_race", Arc::new(store.clone()));
    let _knob = EnvRestore::set(PUT_SIDECARS, "1");
    let name = "idle_gc_raced";
    writer.closed_collection(name);
    let theirs = b"theirs".to_vec();
    {
        let (store, object) = (Arc::clone(&writer.store), writer.sidecar_object(name));
        let theirs = theirs.clone();
        shared_lsm_handle()
            .block_on(async move { store.put(&object, theirs.into()).await })
            .unwrap();
    }
    let puts = || store.sidecar_puts.load(Ordering::SeqCst);
    let before = puts();

    store.sidecar_hidden.store(true, Ordering::SeqCst);
    let (label, pass) = writer.manager.gc_collection_if_idle(name);
    assert_eq!(
        (label, pass.metadata_sidecar),
        ("collected", Some("present"))
    );
    assert_eq!(puts(), before + 1, "the put that loses was not tried");
    assert_eq!(writer.sidecar(name), Some(theirs));
}

/// The look for a sidecar is under the collection's gate like the pass before
/// it, and gives the gate to an open like the pass: a HEAD that has no answer
/// yet is given up, and the sweep has put nothing.
#[test]
fn metadata_sidecar_look_gives_the_gate_to_an_open() {
    let store = Paused::new(Hold::SidecarLooks);
    let writer = Writer::over("idle_gc_sidecar_yield", Arc::new(store.clone()));
    let _knob = EnvRestore::set(PUT_SIDECARS, "1");
    let name = "idle_gc_sidecar_wanted";
    writer.closed_collection(name);
    writer.take_sidecar_object(name);
    let manager = &writer.manager;

    store.armed.store(true, Ordering::SeqCst);
    thread::scope(|scope| {
        let _release = ReleaseOnDrop(Arc::clone(&store.release));
        let pass = scope.spawn(|| manager.gc_collection_if_idle(name));
        store.wait_until_reached();
        store.armed.store(false, Ordering::SeqCst);
        assert_eq!(writer.sidecar(name), None);
        // That HEAD never answers. The sweep gives it up when the open asks
        // for the gate.
        let open = scope.spawn(|| {
            manager
                .get_collection(name)
                .map_err(|error| error.to_string())
        });
        let deadline = Instant::now() + Duration::from_secs(30);
        while !pass.is_finished() {
            assert!(
                Instant::now() < deadline,
                "the look for a sidecar kept the gate from an open"
            );
            thread::sleep(Duration::from_millis(10));
        }
        let (label, pass) = pass.join().unwrap();
        assert_eq!((label, pass.metadata_sidecar), ("collected", None));
        assert_rows(&open.join().unwrap().unwrap());
    });
}

/// The sidecar of an OPEN collection, which no sweep touches. A put of it
/// that fails used to be only logged: the object stayed a change behind until
/// the collection's next change. Now the optimizer's backlog sweeper sends it
/// again (its call for every loaded collection is
/// `upload_held_metadata_sidecar`), not inside the put interval, and the
/// object then holds the newest metadata.
#[test]
fn sidecar_put_of_an_open_collection_that_failed_is_sent_again() {
    use crate::helix_engine::storage_core::metadata::PayloadIndexSchema;

    let store = Paused::new(Hold::Deletes);
    let writer = Writer::over("sidecar_put_again", Arc::new(store.clone()));
    let _at_once = EnvRestore::set("HELIX_LSM_METADATA_SIDECAR_PUT_INTERVAL_MS", "0");
    let name = "open_and_changed";
    let storage = writer.manager.create_collection(name).unwrap();
    let opened = writer.sidecar(name).expect("the open put the sidecar");
    let puts = || store.sidecar_puts.load(Ordering::SeqCst);
    let indexed = |sidecar: &[u8]| {
        sonic_rs::from_slice::<StorageMetadataSidecar>(sidecar)
            .unwrap()
            .metadata
            .payload_indices
            .contains_key("repo")
    };

    // A change of the metadata whose put fails.
    store.failing_sidecar_puts.store(1, Ordering::SeqCst);
    storage
        .with_write_backend(|w| {
            storage
                .update_metadata_be(w, |metadata| {
                    let mut indices = metadata.payload_indices.clone();
                    indices.insert("repo".to_string(), PayloadIndexSchema::Keyword);
                    metadata.set_payload_indices(indices);
                    Ok(())
                })
                .map(|_| ())
        })
        .unwrap();
    assert_eq!(
        store.failing_sidecar_puts.load(Ordering::SeqCst),
        0,
        "the change put no sidecar"
    );
    let tried = puts();
    assert_eq!(
        writer.sidecar(name),
        Some(opened),
        "the failed put left the object as it was"
    );

    // Inside the put interval the sweeper's call sends nothing.
    {
        let _hour = EnvRestore::set("HELIX_LSM_METADATA_SIDECAR_PUT_INTERVAL_MS", "3600000");
        storage.upload_held_metadata_sidecar();
        assert_eq!(puts(), tried, "sent again inside the put interval");
    }
    // Past it, the call sends the newest metadata, once.
    storage.upload_held_metadata_sidecar();
    let sent = writer.sidecar(name).expect("the object is there");
    assert!(indexed(&sent), "the object is still a change behind");
    assert_eq!(
        sent,
        fs::read(HelixGraphStorage::metadata_sidecar_path(
            &writer.manager.collection_path(name)
        ))
        .unwrap()
    );
    assert_eq!(puts(), tried + 1);
    storage.upload_held_metadata_sidecar();
    assert_eq!(puts(), tried + 1, "a sidecar that arrived was sent again");
}

// The adjacency backfill's mark in the metadata sidecar: what a reader's
// collection info answers for a graph it has not opened. These tests count
// sidecar puts under a put interval they set, so they are serial with the
// tests elsewhere that set it too, as well as under this file's env lock.

/// A relationship row, which the adjacency backfill derives an edge from.
fn relationship_row(
    id: u128,
    callee: &str,
) -> crate::helix_engine::storage_core::upsert::NodeUpsert {
    let text = |value: &str| Value::String(value.to_string());
    crate::helix_engine::storage_core::upsert::NodeUpsert {
        id,
        label: "point".into(),
        properties: HashMap::from([
            ("caller_symbol".to_string(), text("a_fn")),
            ("callee_symbol".to_string(), text(callee)),
            ("caller_path".to_string(), text("src/a.rs")),
            ("callee_path".to_string(), text("src/b.rs")),
            ("edge_type".to_string(), text("calls")),
        ]),
    }
}

/// A graph of two relationship rows on a writer whose put interval is an
/// hour: after the put at its creation, only a put that the mark asks for
/// reaches the object store.
fn graph_of_two_rows(writer: &Writer, name: &str) -> Arc<HelixGraphStorage> {
    let storage = writer.manager.create_collection(name).unwrap();
    storage
        .with_write_backend(|w| {
            storage.upsert_node_be(w, &relationship_row(1, "b_fn"))?;
            storage.upsert_node_be(w, &relationship_row(2, "c_fn"))?;
            Ok(())
        })
        .unwrap();
    storage
}

/// The mark in the object store's sidecar of `name`.
fn sidecar_mark(writer: &Writer, name: &str) -> Option<bool> {
    let sidecar = writer
        .sidecar(name)
        .expect("the object store has a sidecar");
    sonic_rs::from_slice::<StorageMetadataSidecar>(&sidecar)
        .unwrap()
        .adjacency_backfill_complete
}

/// The mark in the store.
fn stored_mark(storage: &HelixGraphStorage) -> bool {
    let read = storage.backend.begin_read().unwrap();
    storage
        .backend
        .get_with(
            &read,
            Namespace::Metadata,
            b"adj_backfill_from_points:complete",
            |value| matches!(value, Some(b"1")),
        )
        .unwrap()
}

/// Calls of a row each, forced or not at the first, until the pass is over:
/// how many it took. Before every call but the last the object store's
/// sidecar says "not complete".
fn run_the_pass(writer: &Writer, storage: &HelixGraphStorage, name: &str, force: bool) -> usize {
    let mut calls = 0;
    loop {
        let complete = storage
            .backfill_adjacency_from_points_batch(name, 1, force && calls == 0, false)
            .unwrap()
            .2;
        calls += 1;
        if complete {
            return calls;
        }
        assert_eq!(
            (sidecar_mark(writer, name), stored_mark(storage)),
            (Some(false), false),
            "call {calls} is not the pass's last"
        );
    }
}

/// The object store's sidecar says "complete" only once the pass's last
/// write is stored, and then at once: the mark alone is put, inside a put
/// interval that holds every other change back. A call of a finished pass
/// puts nothing more.
#[test]
#[serial_test::serial]
fn adjacency_backfill_mark_is_put_when_the_pass_is_stored_and_not_before() {
    let store = Paused::new(Hold::Deletes);
    let writer = Writer::over("backfill_mark_put", Arc::new(store.clone()));
    let _hour = EnvRestore::set("HELIX_LSM_METADATA_SIDECAR_PUT_INTERVAL_MS", "3600000");
    let name = "calls_graph";
    let storage = graph_of_two_rows(&writer, name);
    assert_eq!(
        sidecar_mark(&writer, name),
        Some(false),
        "a new collection's sidecar"
    );
    let puts = || store.sidecar_puts.load(Ordering::SeqCst);

    let before = puts();
    assert!(run_the_pass(&writer, &storage, name, false) > 1);
    assert!(stored_mark(&storage));
    assert_eq!(
        (sidecar_mark(&writer, name), puts()),
        (Some(true), before + 1),
        "one put, for the mark"
    );
    assert_eq!(storage.adjacency_backfill_complete(), Some(true));

    let finished = storage
        .backfill_adjacency_from_points_batch(name, 1, false, false)
        .unwrap();
    assert_eq!((finished.0, finished.1, finished.2), (0, 0, true));
    assert_eq!(puts(), before + 1, "the sidecar already said so");
}

/// Takes the mark out of the object store's sidecar of `name`: the sidecar
/// as a writer that did not keep the mark there left it.
fn put_the_sidecar_without_the_mark(writer: &Writer, name: &str) {
    let mut older: serde_json::Value =
        serde_json::from_slice(&writer.sidecar(name).unwrap()).unwrap();
    older
        .as_object_mut()
        .unwrap()
        .remove("adjacency_backfill_complete")
        .expect("the sidecar carried the mark");
    let (objects, object) = (Arc::clone(&writer.store), writer.sidecar_object(name));
    shared_lsm_handle()
        .block_on(async move {
            objects
                .put(&object, serde_json::to_vec(&older).unwrap().into())
                .await
        })
        .unwrap();
    assert_eq!(
        sidecar_mark(writer, name),
        None,
        "an older sidecar says nothing"
    );
}

/// A graph that is whole, whose sidecar in the object store was written
/// before the mark was kept there. The open of such a graph puts its sidecar
/// with the mark it read, and a call of the finished pass then puts nothing.
/// Where the open's put did not arrive, the unforced call that finds the
/// pass complete puts the sidecar with the mark, once, and commits nothing;
/// a second call writes no object at all.
#[test]
#[serial_test::serial]
fn finished_pass_fills_a_sidecar_that_does_not_say_so_and_fills_it_once() {
    let store = Paused::new(Hold::Deletes);
    let writer = Writer::over("backfill_mark_fill", Arc::new(store.clone()));
    let _hour = EnvRestore::set("HELIX_LSM_METADATA_SIDECAR_PUT_INTERVAL_MS", "3600000");
    let name = "calls_graph";
    let storage = graph_of_two_rows(&writer, name);
    run_the_pass(&writer, &storage, name, false);
    drop(storage);
    let puts = || store.sidecar_puts.load(Ordering::SeqCst);
    let finished = |storage: &HelixGraphStorage| {
        storage
            .backfill_adjacency_from_points_batch(name, 1_000, false, false)
            .unwrap()
            .2
    };

    // The open's own put carries the mark.
    writer.close(name);
    put_the_sidecar_without_the_mark(&writer, name);
    let storage = writer.manager.get_collection(name).unwrap();
    assert_eq!(
        (
            sidecar_mark(&writer, name),
            storage.adjacency_backfill_complete()
        ),
        (Some(true), Some(true)),
        "the open read the mark and put its sidecar"
    );
    let before = puts();
    assert!(finished(&storage));
    assert_eq!(puts(), before, "the sidecar already said so");
    drop(storage);

    // The open's put fails: the object store keeps the older sidecar.
    writer.close(name);
    put_the_sidecar_without_the_mark(&writer, name);
    store.failing_sidecar_puts.store(1, Ordering::SeqCst);
    let storage = writer.manager.get_collection(name).unwrap();
    assert_eq!(
        store.failing_sidecar_puts.load(Ordering::SeqCst),
        0,
        "the open tried its put"
    );
    assert_eq!(
        (
            sidecar_mark(&writer, name),
            storage.adjacency_backfill_complete()
        ),
        (None, Some(true))
    );
    let before = puts();
    assert!(finished(&storage));
    assert_eq!(
        (sidecar_mark(&writer, name), puts()),
        (Some(true), before + 1),
        "the call filled the sidecar"
    );
    let written = writer.objects();
    assert!(finished(&storage));
    assert_eq!(puts(), before + 1, "a second call put a sidecar");
    assert_eq!(writer.objects(), written, "a second call wrote an object");
}

/// A forced call starts the pass over. The object store's sidecar says "not
/// complete" before that call writes, whatever the put interval: a call
/// whose put fails is refused and has written nothing, and the collection
/// serves on. The same call again puts "not complete" and then clears the
/// mark; a further forced call puts nothing more.
#[test]
#[serial_test::serial]
fn forced_pass_says_not_complete_in_the_object_store_before_it_writes() {
    let store = Paused::new(Hold::Deletes);
    let writer = Writer::over("backfill_mark_forced", Arc::new(store.clone()));
    let _hour = EnvRestore::set("HELIX_LSM_METADATA_SIDECAR_PUT_INTERVAL_MS", "3600000");
    let name = "calls_graph";
    let storage = graph_of_two_rows(&writer, name);
    run_the_pass(&writer, &storage, name, false);
    assert_eq!(sidecar_mark(&writer, name), Some(true));
    let puts = || store.sidecar_puts.load(Ordering::SeqCst);
    let forced = || storage.backfill_adjacency_from_points_batch(name, 1, true, false);

    let written = writer.objects();
    store.failing_sidecar_puts.store(1, Ordering::SeqCst);
    let refused = forced().expect_err("a forced call went on without its put");
    // The transfer's error carries two texts that would quarantine the graph
    // as errors of its own store: a sidecar that does not arrive quarantines
    // nobody.
    let text = refused.to_string();
    assert!(
        text.contains("NoSuchKey") && text.contains("checksum"),
        "{text}"
    );
    assert!(!refused.should_quarantine_collection(), "{text}");
    assert_eq!(
        store.failing_sidecar_puts.load(Ordering::SeqCst),
        0,
        "the call tried its put"
    );
    assert!(stored_mark(&storage), "the refused call cleared the mark");
    assert_eq!(
        writer.objects(),
        written,
        "the refused call wrote an object"
    );

    let before = puts();
    assert!(!forced().unwrap().2, "a row a call: the pass is not over");
    assert_eq!(
        (sidecar_mark(&writer, name), stored_mark(&storage), puts()),
        (Some(false), false, before + 1)
    );
    assert_eq!(storage.adjacency_backfill_complete(), Some(false));
    assert!(!forced().unwrap().2);
    assert_eq!(puts(), before + 1, "the sidecar already said so");

    run_the_pass(&writer, &storage, name, false);
    assert_eq!(
        (sidecar_mark(&writer, name), stored_mark(&storage), puts()),
        (Some(true), true, before + 2)
    );
}

/// A regular put of the sidecar that is on its way when a forced call starts
/// the pass over is the earlier of the two. The forced call waits for it,
/// the one wait on the collection's put lock, and then puts "not complete",
/// which is what the object store ends with. Without the lock the forced
/// call's put would arrive first and the one that says "complete" after it.
#[test]
#[serial_test::serial]
fn forced_call_waits_for_the_sidecar_put_on_its_way_and_lands_after_it() {
    let store = Paused::new(Hold::SidecarPuts);
    let writer = Writer::over("backfill_mark_put_order", Arc::new(store.clone()));
    let name = "calls_graph";
    let storage = {
        let _hour = EnvRestore::set("HELIX_LSM_METADATA_SIDECAR_PUT_INTERVAL_MS", "3600000");
        let storage = graph_of_two_rows(&writer, name);
        run_the_pass(&writer, &storage, name, false);
        // A change inside the put interval: its sidecar, which says
        // "complete", is held for the sweeper.
        storage
            .with_write_backend(|w| {
                storage
                    .upsert_node_be(w, &relationship_row(3, "d_fn"))
                    .map(|_| ())
            })
            .unwrap();
        storage
    };
    assert_eq!(sidecar_mark(&writer, name), Some(true));
    let puts = || store.sidecar_puts.load(Ordering::SeqCst);
    let before = puts();

    let _at_once = EnvRestore::set("HELIX_LSM_METADATA_SIDECAR_PUT_INTERVAL_MS", "0");
    store.armed.store(true, Ordering::SeqCst);
    thread::scope(|scope| {
        let release = ReleaseOnDrop(Arc::clone(&store.release));
        let regular = scope.spawn(|| storage.upload_held_metadata_sidecar());
        store.wait_until_reached();
        // Only the put that is on its way stays parked.
        store.armed.store(false, Ordering::SeqCst);

        let forced =
            scope.spawn(|| storage.backfill_adjacency_from_points_batch(name, 1, true, false));
        thread::sleep(Duration::from_millis(300));
        assert!(
            !forced.is_finished(),
            "the forced call did not wait for the put on its way"
        );
        assert_eq!(puts(), before + 1, "a second put beside the one on its way");

        drop(release);
        regular.join().unwrap();
        assert!(!forced.join().unwrap().unwrap().2);
    });
    assert_eq!(
        (
            sidecar_mark(&writer, name),
            storage.adjacency_backfill_complete()
        ),
        (Some(false), Some(false)),
        "the put that says complete landed after the forced call's"
    );
}

/// A put that answers an error may have arrived all the same. After a pass
/// whose put of "complete" failed, a forced call puts "not complete", though
/// the last put this process saw arrive already said so.
#[test]
#[serial_test::serial]
fn forced_call_puts_again_after_a_put_of_complete_that_answered_an_error() {
    let store = Paused::new(Hold::Deletes);
    let writer = Writer::over("backfill_mark_unknown", Arc::new(store.clone()));
    let _hour = EnvRestore::set("HELIX_LSM_METADATA_SIDECAR_PUT_INTERVAL_MS", "3600000");
    let name = "calls_graph";
    let storage = graph_of_two_rows(&writer, name);
    let puts = || store.sidecar_puts.load(Ordering::SeqCst);

    // The whole pass in one call, whose put of "complete" fails.
    store.failing_sidecar_puts.store(1, Ordering::SeqCst);
    let pass = storage
        .backfill_adjacency_from_points_batch(name, 1_000, false, false)
        .unwrap();
    assert!(pass.2, "the pass is over though its put failed");
    assert_eq!(
        store.failing_sidecar_puts.load(Ordering::SeqCst),
        0,
        "the pass tried its put"
    );
    assert_eq!(
        (sidecar_mark(&writer, name), stored_mark(&storage)),
        (Some(false), true)
    );

    let before = puts();
    storage
        .backfill_adjacency_from_points_batch(name, 1, true, false)
        .unwrap();
    assert_eq!(
        (sidecar_mark(&writer, name), puts()),
        (Some(false), before + 1),
        "the forced call took the failed put for one that did not arrive"
    );
}

/// A pass whose last write fails says nothing of its end: the mark is not
/// stored, and the object store's sidecar stays at "not complete".
#[test]
#[serial_test::serial]
fn pass_whose_last_write_fails_does_not_say_complete() {
    let store = Paused::new(Hold::Deletes);
    let writer = Writer::over("backfill_mark_failed_write", Arc::new(store.clone()));
    let name = "calls_graph";
    let storage = graph_of_two_rows(&writer, name);
    // The edge the first row derives is stored as bytes that do not decode,
    // so the call that takes the row fails where it reads the stored edge.
    let edge = crate::protocol::deterministic_id::edge_id(
        name, "CALLS", "a_fn", "b_fn", "src/a.rs", "src/b.rs",
    );
    let mut txn = storage.backend.begin_write().unwrap();
    storage
        .backend
        .put(&mut txn, Namespace::Edges, &edge.to_be_bytes(), &[0xFF])
        .unwrap();
    storage.backend.commit(txn).unwrap();

    assert!(storage
        .backfill_adjacency_from_points_batch(name, 1_000, false, false)
        .is_err());
    assert!(!stored_mark(&storage));
    assert_eq!(sidecar_mark(&writer, name), Some(false));
    assert_eq!(storage.adjacency_backfill_complete(), Some(false));
}
