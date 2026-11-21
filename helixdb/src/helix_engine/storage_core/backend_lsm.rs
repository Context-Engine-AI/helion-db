//! `LsmBackend` — the [`StorageBackend`] implementation over SlateDB
//! (an LSM-tree engine on object storage / S3).
//!
//! SlateDB is async; Helion's storage call sites are synchronous. We bridge with
//! a dedicated multi-thread tokio runtime owned by the backend: every operation
//! runs via `rt.block_on(...)`. The runtime is multi-thread so SlateDB's
//! background flush/compaction tasks keep progressing between calls. (Production
//! call sites must invoke these from blocking threads, never from inside the
//! gateway's async runtime — enforced during the call-site migration.)
//!
//! SlateDB is a single flat ordered keyspace, so [`Namespace`] becomes a key
//! prefix (`<ns_name>\0` ++ user-key). Scans bound to a namespace's prefix.

#![allow(dead_code)]

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap, HashSet};
#[path = "backend_lsm_cache_evictor.rs"]
pub mod cache_evictor;
#[path = "backend_lsm_closed_gc.rs"]
mod closed_gc;
#[path = "backend_lsm_dup_visit.rs"]
mod dup_visit;
#[path = "backend_lsm_flush_gate.rs"]
pub(crate) mod flush_gate;
pub(crate) use closed_gc::{
    closed_gc_deletes, closed_gc_plan, closed_gc_puts_metadata_sidecars, gc_closed_s3_collection,
    put_missing_s3_metadata_sidecar, ClosedGcOutcome, ClosedGcPass, ClosedGcPlan, GateWanted,
};
#[cfg(test)]
pub(crate) use closed_gc::{
    gc_closed_collection_with_store, put_missing_metadata_sidecar_with_store,
};
use std::future::Future;
use std::ops::Bound;
use std::sync::Arc;

use futures::{stream, StreamExt, TryStreamExt};
use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjPath;
use object_store::Error as ObjectStoreError;
use parking_lot::RwLock;
use slatedb::bytes::Bytes;
use slatedb::config::{
    CompressionCodec, DbReaderOptions, GarbageCollectorDirectoryOptions, GarbageCollectorOptions,
    ObjectStoreCacheOptions, ScanOptions, Settings, WriteOptions,
};
use slatedb::db_cache::foyer::{FoyerCache, FoyerCacheOptions};
use slatedb::db_cache::{CacheUsageSnapshot, DbCache, SplitCache};
use slatedb::object_store::memory::InMemory;
use slatedb::object_store::{ObjectStore, ObjectStoreExt};
use slatedb::{
    BloomFilterPolicy, CloseReason, Db, DbIterator, DbSnapshot, FilterPolicy, IsolationLevel,
    KeyValue, MergeOperator, MergeOperatorError, PrefixExtractor, PrefixTarget, WriteBatch,
};
use std::sync::OnceLock;
use tokio::runtime::{Builder, Handle, Runtime};

use crate::helix_engine::types::GraphError;

use super::backend::{
    is_dup, next_prefix, BackendError, KeyRange, Namespace, ReadTxn, StorageBackend,
    TransactionRow, WriteTxn,
};
use super::backend_lmdb::ns_name;
use super::metadata::{
    apply_lsm_counter_delta, decode_lsm_counter_delta, decode_lsm_counter_value,
    encode_lsm_counter_delta, encode_lsm_counter_value,
};

thread_local! {
    static LSM_BLOCKING_ALLOWED: Cell<bool> = const { Cell::new(false) };
    /// Cooperative cancellation for foreground requests served by either an
    /// LSM reader replica or the writer's local `LsmBackend`. The async gateway
    /// installs this only for read-like routes; durable writer operations
    /// deliberately never receive a token.
    static LSM_READ_CANCELLATION: RefCell<Option<LsmReadCancellation>> = const { RefCell::new(None) };
    /// `Some` while this thread is inside [`opening_a_new_database`] and has
    /// not opened its writer yet; `Some(true)` once an open there found a
    /// database at its path.
    static LSM_OPENS_A_NEW_DATABASE: Cell<Option<bool>> = const { Cell::new(None) };
    /// Test-only: sequential SlateDB round trips (`block_on_lsm_mapped` calls)
    /// issued by this thread.
    #[cfg(test)]
    static LSM_ROUND_TRIPS_FOR_TEST: Cell<usize> = const { Cell::new(0) };
    /// Test-only: for each batched point read by key this thread made
    /// (`collect_values_many` of a writer, `collect_values_many_at` of a
    /// reader), whether a batched row read of its collection was marked in
    /// flight then ([`batch_row_read_in_flight`]).
    #[cfg(test)]
    static KEY_BATCHES_MARKED_FOR_TEST: std::cell::RefCell<Vec<bool>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// Test-only: rows pulled by dup-namespace scans on this thread, matching
    /// or not — the quantity a prefix scan must keep proportional to its hits.
    #[cfg(test)]
    static DUP_SCAN_ROWS_FOR_TEST: Cell<usize> = const { Cell::new(0) };
    /// Test-only: what the writers this thread opens get in place of the
    /// process's flush gate ([`with_flush_gate_for_test`]).
    #[cfg(test)]
    static FLUSH_GATE_FOR_TEST: RefCell<Option<Option<Arc<dyn slatedb::L0FlushGate>>>> =
        const { RefCell::new(None) };
    /// Test-only: what the writers this thread opens pass their SlateDB
    /// builder through ([`with_db_builder_for_test`]).
    #[cfg(test)]
    static DB_BUILDER_FOR_TEST: RefCell<Option<DbBuilderForTest>> = const { RefCell::new(None) };
    /// Test-only: this thread's next [`LsmBackend::present_for_update`] of
    /// any row fails ([`fail_next_present_for_update_for_test`]).
    #[cfg(test)]
    static FAIL_NEXT_PRESENT_FOR_UPDATE_FOR_TEST: Cell<bool> = const { Cell::new(false) };
    /// Test-only: what [`lsm_multi_get_enabled`] answers on this thread
    /// ([`with_lsm_multi_get_for_test`]).
    #[cfg(test)]
    static MULTI_GET_FOR_TEST: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Test-only: the current thread's next [`LsmBackend::present_for_update`]
/// that has a row to read answers an error in place of reading.
#[cfg(test)]
pub(crate) fn fail_next_present_for_update_for_test() {
    FAIL_NEXT_PRESENT_FOR_UPDATE_FOR_TEST.set(true);
}

/// Test-only: see [`with_db_builder_for_test`].
#[cfg(test)]
type DbBuilderForTest =
    std::rc::Rc<dyn for<'a> Fn(slatedb::DbBuilder<&'a str>) -> slatedb::DbBuilder<&'a str>>;

/// Test-only: runs `open`, in which every WRITER this thread opens passes its
/// SlateDB builder through `add` last, after what the product sets. A reader
/// replica's builder is not passed. Open one writer inside it: a second one
/// opened there by the same thread is given `add` as well.
#[cfg(test)]
pub(crate) fn with_db_builder_for_test<T>(
    add: impl for<'a> Fn(slatedb::DbBuilder<&'a str>) -> slatedb::DbBuilder<&'a str> + 'static,
    open: impl FnOnce() -> T,
) -> T {
    struct Leave(Option<DbBuilderForTest>);
    impl Drop for Leave {
        fn drop(&mut self) {
            DB_BUILDER_FOR_TEST.set(self.0.take());
        }
    }
    let _leave = Leave(DB_BUILDER_FOR_TEST.replace(Some(std::rc::Rc::new(add))));
    open()
}

/// Test-only: runs `open`, in which every writer this thread opens asks
/// `gate` before a memtable flush, or no gate for `None`.
#[cfg(test)]
pub(crate) fn with_flush_gate_for_test<T>(
    gate: Option<Arc<dyn slatedb::L0FlushGate>>,
    open: impl FnOnce() -> T,
) -> T {
    struct Leave(Option<Option<Arc<dyn slatedb::L0FlushGate>>>);
    impl Drop for Leave {
        fn drop(&mut self) {
            FLUSH_GATE_FOR_TEST.set(self.0.take());
        }
    }
    let _leave = Leave(FLUSH_GATE_FOR_TEST.replace(Some(gate)));
    open()
}

/// Test-only: runs `f` with [`lsm_multi_get_enabled`] forced on this thread:
/// `false` is how production's readers run (`HELIX_LSM_MULTI_GET=0`).
#[cfg(test)]
pub(crate) fn with_lsm_multi_get_for_test<T>(on: bool, f: impl FnOnce() -> T) -> T {
    let previous = MULTI_GET_FOR_TEST.replace(Some(on));
    let out = f();
    MULTI_GET_FOR_TEST.set(previous);
    out
}

/// Test-only: SlateDB round trips issued by the current thread so far.
#[cfg(test)]
pub(crate) fn lsm_round_trips_for_test() -> usize {
    LSM_ROUND_TRIPS_FOR_TEST.with(Cell::get)
}

/// Test-only: notes one batched point read by key of the collection at
/// `root`, with whether a batched row read of that collection is marked in
/// flight now.
#[cfg(test)]
pub(crate) fn note_key_batch_for_test(root: &str) {
    let object = format!("{root}/compacted/x.sst");
    let marked = slatedb::cached_object_store_stats::ranged_miss_stats()
        .in_batch_row_read(&object.as_str().into());
    KEY_BATCHES_MARKED_FOR_TEST.with(|seen| seen.borrow_mut().push(marked));
}

/// Test-only: what [`note_key_batch_for_test`] noted on this thread since
/// the last call, in order.
#[cfg(test)]
pub(crate) fn key_batches_marked_for_test() -> Vec<bool> {
    KEY_BATCHES_MARKED_FOR_TEST.with(|seen| std::mem::take(&mut *seen.borrow_mut()))
}

/// Test-only: dup-namespace rows scanned by the current thread so far.
#[cfg(test)]
pub(crate) fn dup_scan_rows_for_test() -> usize {
    DUP_SCAN_ROWS_FOR_TEST.with(Cell::get)
}

/// Runs `open`, in which the first writer database this thread opens must
/// be one it creates: where a manifest is already at its path the open fails
/// and leaves that manifest as it found it. For a prefix the caller has just
/// purged, where a manifest can only be one put there since. Returns what
/// `open` returned and whether a manifest was found. A writer opened in
/// `open` after the first one is opened as ever.
pub(crate) fn opening_a_new_database<T>(open: impl FnOnce() -> T) -> (T, bool) {
    struct Leave(Option<bool>);
    impl Drop for Leave {
        fn drop(&mut self) {
            LSM_OPENS_A_NEW_DATABASE.set(self.0);
        }
    }
    let leave = Leave(LSM_OPENS_A_NEW_DATABASE.replace(Some(false)));
    let opened = open();
    let found = LSM_OPENS_A_NEW_DATABASE.get() == Some(true);
    drop(leave);
    (opened, found)
}

struct LsmReadCancellationInner {
    cancelled: std::sync::atomic::AtomicBool,
    notify: tokio::sync::Notify,
}

/// Cancellation signal shared between an async HTTP request and the blocking
/// worker serving its read-only SlateDB operation. Dropping an Axum request
/// future (for example when the gateway's reader attempt times out and closes
/// the connection) signals this token; [`block_on_lsm_read`] then drops the
/// pending side-effect-free SlateDB future instead of leaving the blocking
/// worker occupied.
#[derive(Clone)]
pub(crate) struct LsmReadCancellation {
    inner: Arc<LsmReadCancellationInner>,
}

impl LsmReadCancellation {
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(LsmReadCancellationInner {
                cancelled: std::sync::atomic::AtomicBool::new(false),
                notify: tokio::sync::Notify::new(),
            }),
        }
    }

    pub(crate) fn cancel(&self) {
        if !self
            .inner
            .cancelled
            .swap(true, std::sync::atomic::Ordering::AcqRel)
        {
            // `notify_one` stores a permit if the select branch has not been
            // polled yet, closing the load/register race around cancellation.
            self.inner.notify.notify_one();
        }
    }

    #[cfg(test)]
    pub(crate) fn is_cancelled(&self) -> bool {
        self.is_cancelled_inner()
    }

    fn is_cancelled_inner(&self) -> bool {
        self.inner
            .cancelled
            .load(std::sync::atomic::Ordering::Acquire)
    }

    async fn cancelled(&self) {
        while !self
            .inner
            .cancelled
            .load(std::sync::atomic::Ordering::Acquire)
        {
            self.inner.notify.notified().await;
        }
    }
}

struct LsmBlockingContext {
    previous: bool,
}

impl LsmBlockingContext {
    fn enter() -> Self {
        let previous = LSM_BLOCKING_ALLOWED.with(|allowed| {
            let previous = allowed.get();
            allowed.set(true);
            previous
        });
        Self { previous }
    }
}

impl Drop for LsmBlockingContext {
    fn drop(&mut self) {
        LSM_BLOCKING_ALLOWED.with(|allowed| allowed.set(self.previous));
    }
}

struct LsmReadCancellationContext {
    previous: Option<LsmReadCancellation>,
}

impl LsmReadCancellationContext {
    fn enter(cancellation: LsmReadCancellation) -> Self {
        let previous = LSM_READ_CANCELLATION.with(|slot| slot.replace(Some(cancellation)));
        Self { previous }
    }
}

impl Drop for LsmReadCancellationContext {
    fn drop(&mut self) {
        LSM_READ_CANCELLATION.with(|slot| {
            slot.replace(self.previous.take());
        });
    }
}

/// Temporarily hides a request's read-cancellation token while an LSM cold open
/// is in its non-abortable phase. The token is restored on drop so the handler's
/// first ordinary read still observes a disconnect.
pub(crate) struct LsmReadCancellationMask {
    previous: Option<LsmReadCancellation>,
}

impl Drop for LsmReadCancellationMask {
    fn drop(&mut self) {
        LSM_READ_CANCELLATION.with(|slot| {
            slot.replace(self.previous.take());
        });
    }
}

/// Atomically reject an already-cancelled request or mask its live token. A
/// cancellation racing just after the check is deferred until the returned
/// guard drops, which lets a side-effecting LSM build reach the collection
/// cache instead of discarding the completed handle.
pub(crate) fn mask_lsm_read_cancellation_for_cold_open(
) -> Result<LsmReadCancellationMask, BackendError> {
    let previous = LSM_READ_CANCELLATION.with(|slot| slot.replace(None));
    if previous
        .as_ref()
        .is_some_and(LsmReadCancellation::is_cancelled_inner)
    {
        LSM_READ_CANCELLATION.with(|slot| {
            slot.replace(previous);
        });
        return Err(lsm_read_cancelled_error());
    }
    Ok(LsmReadCancellationMask { previous })
}

/// What [`allow_lsm_blocking`] / [`allow_lsm_blocking_cancellable`] set up on
/// the calling thread, for a helper thread that serves part of the same
/// read-only request. Both are thread-local: without them a helper's nested
/// read (a filter reading a node from inside a scan) is refused and its reads
/// ignore the request's cancellation.
#[derive(Clone)]
pub(crate) struct LsmThreadContext {
    blocking_allowed: bool,
    cancellation: Option<LsmReadCancellation>,
}

impl LsmThreadContext {
    pub(crate) fn current() -> Self {
        Self {
            blocking_allowed: LSM_BLOCKING_ALLOWED.with(Cell::get),
            cancellation: LSM_READ_CANCELLATION.with(|slot| slot.borrow().clone()),
        }
    }

    /// Runs `f` on this thread as it would run on the captured one.
    pub(crate) fn enter<T>(&self, f: impl FnOnce() -> T) -> T {
        let _blocking_context = self.blocking_allowed.then(LsmBlockingContext::enter);
        let _cancellation_context = self
            .cancellation
            .clone()
            .map(LsmReadCancellationContext::enter);
        f()
    }
}

pub(crate) fn allow_lsm_blocking<T>(f: impl FnOnce() -> T) -> T {
    let _context = LsmBlockingContext::enter();
    f()
}

/// Like [`allow_lsm_blocking`], but also makes a cooperative cancellation
/// signal visible to side-effect-free LSM async-to-sync bridges. This must
/// remain restricted to read-like gateway routes: cancelling a durable write
/// future can leave commit outcome ambiguous.
pub(crate) fn allow_lsm_blocking_cancellable<T>(
    cancellation: LsmReadCancellation,
    f: impl FnOnce() -> T,
) -> T {
    let _blocking_context = LsmBlockingContext::enter();
    let _cancellation_context = LsmReadCancellationContext::enter(cancellation);
    f()
}

fn io<E: std::fmt::Display>(e: E) -> BackendError {
    BackendError::Io(e.to_string())
}

/// Classify a SlateDB WRITE/commit/flush failure. SlateDB epoch-fencing (a newer
/// writer took the manifest — "detected newer DB client") and CAS / conditional-
/// write conflicts (transaction or manifest version conflicts) must be kept
/// distinct from transient I/O, so both map to [`BackendError::Conflict`];
/// callers then decide whether a conflict is retryable.
///
/// slatedb 0.13.1 exposes typed variants publicly (`ErrorKind::Transaction` and
/// `ErrorKind::Closed(CloseReason::Fenced)`), but the [`block_on_lsm`] bridge
/// erases the concrete error to `Display` before we see it, so we match on the
/// stable substrings those variants render. The crate is exact-version-pinned, so
/// the messages are fixed: fencing renders "detected newer DB client"; CAS /
/// manifest conflicts render "...conflict" / "...version already exists".
/// Conservative on purpose — only clear fencing/CAS signals become `Conflict`.
fn map_slatedb_err<E: std::fmt::Display>(e: E) -> BackendError {
    let msg = e.to_string();
    if lsm_error_message_is_conflict(&msg) {
        BackendError::Conflict(msg)
    } else {
        BackendError::Io(msg)
    }
}

fn lsm_error_message_is_conflict(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    // Bare "cas" is intentionally excluded: it false-positives inside ordinary
    // words ("case", "cascade", "cast", "broadcast") that can appear in transient
    // I/O messages. The spelled-out "compare-and-swap" is the safe CAS signal.
    const CONFLICT_SIGNALS: &[&str] = &[
        "detected newer db client", // SlateDBError::Fenced (epoch fencing)
        "fenced",
        "fencing",
        "epoch",
        "conflict", // TransactionConflict -> "transaction conflict"
        "conditional",
        "precondition", // object_store S3 412 conditional-write failure
        "compare-and-swap",
        "version mismatch",
        "version already exists", // TransactionalObjectVersionExists (manifest CAS)
    ];
    CONFLICT_SIGNALS.iter().any(|sig| lower.contains(sig))
}

fn lsm_error_message_is_fenced(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    const FENCED_SIGNALS: &[&str] = &[
        "detected newer db client", // SlateDB CloseReason::Fenced
        "detected newer client",    // slatedb-txn-obj fence wording
        "fenced",
        "fencing",
    ];
    FENCED_SIGNALS.iter().any(|sig| lower.contains(sig))
}

fn lsm_fenced_error(context: &'static str, message: String) -> BackendError {
    BackendError::Conflict(format!(
        "{context}: SlateDB writer is fenced by a newer writer; failing closed until process replacement: {message}"
    ))
}

fn lsm_retryable_cas_error(context: &'static str, message: String) -> BackendError {
    BackendError::Conflict(format!(
        "{context}: SlateDB CAS/transaction conflict retry exhausted without claiming a new writer epoch: {message}"
    ))
}

fn lsm_conflict_message(error: &BackendError) -> Option<&str> {
    match error {
        BackendError::Conflict(message) => Some(message.as_str()),
        _ => None,
    }
}

const LSM_METADATA_TXN_MAX_ATTEMPTS: usize = 8;
const LSM_BATCH_POINT_READ_CONCURRENCY: usize = 96;

/// Env knob: local SSD mount root for SlateDB's object-store cache. When set,
/// reads are served from this local filesystem and S3 is hit only on cold miss.
/// Unset/empty → no object-store cache (bare open, current behavior).
const ENV_CACHE_DIR: &str = "HELIX_LSM_CACHE_DIR";
/// Env knob: optional cap on the cache size in bytes. When `ENV_CACHE_DIR` is
/// set but this is unset/unparsable, SlateDB's default cap applies (16 GiB).
const ENV_CACHE_MAX_BYTES: &str = "HELIX_LSM_CACHE_MAX_BYTES";
/// Env knob: max share of the cgroup memory limit that the local object-store
/// cache may target. Linux charges hot cache files to the pod as page cache, so
/// a disk-sized cache budget can OOM a memory-limited pod even when Rust heap is
/// small. Set `0` to disable the memory-aware clamp.
const ENV_CACHE_MEMORY_MAX_PCT: &str = "HELIX_LSM_CACHE_MEMORY_MAX_PCT";
const DEFAULT_CACHE_MEMORY_MAX_PCT: usize = 25;
const CGROUP_V2_MEMORY_MAX: &str = "/sys/fs/cgroup/memory.max";
const CGROUP_V1_MEMORY_LIMIT: &str = "/sys/fs/cgroup/memory/memory.limit_in_bytes";
/// Env knob: object-store cache evictor sweep interval in seconds. Lower than the
/// SlateDB default (3600s) tightens the cache size bound under bursty cold reads.
/// `0` disables recurring sweeps after the initial cache scan.
const ENV_CACHE_SCAN_SECS: &str = "HELIX_LSM_CACHE_SCAN_SECS";
/// Env knob: object-store cache part-file size in bytes. Each cached S3 object
/// is split into part files of this size, and every part read/write pushes one
/// event onto the per-collection evictor's bounded queue (100 slots, hardcoded
/// in SlateDB). When that queue backpressures, SlateDB SKIPS the cache write
/// (best-effort), so scan-heavy load with small parts floods the queue and
/// keeps hot data out of the local cache. Larger parts cut the event rate
/// proportionally. Must be a non-zero multiple of 1024 (SlateDB requirement);
/// unset/invalid → SlateDB default (4 MiB).
const ENV_CACHE_PART_BYTES: &str = "HELIX_LSM_CACHE_PART_BYTES";
/// Env knob: WAL flush interval in milliseconds (SlateDB `flush_interval`). The
/// WAL buffer flushes at least this often. An `await_durable` write does NOT
/// trigger a flush: it waits for the next one (this tick, a full WAL buffer,
/// or an explicit `db.flush()`), so every durable `commit` waits up to this
/// interval, and nearly all of it when commits follow each other in a loop.
/// Only the group-commit barrier (`flush_durable`, `db.flush()`) flushes at
/// once. Unset → SlateDB default (100 ms). Durability is unaffected: a durable
/// write returns only once its WAL is on the object store.
const ENV_FLUSH_INTERVAL_MS: &str = "HELIX_LSM_FLUSH_INTERVAL_MS";
/// Env knob: SlateDB manifest poll interval in milliseconds. Writer and reader
/// handles poll object storage for newer manifests at this cadence; raising it
/// trims background S3 churn when Helion owns explicit refresh/reconcile loops.
/// Unset -> SlateDB default (1s).
const ENV_MANIFEST_POLL_MS: &str = "HELIX_LSM_MANIFEST_POLL_MS";
/// Env knob: reader-replica manifest poll interval in milliseconds. The writer
/// poll cadence is an S3-cost dial (it only detects fencing), but the reader
/// cadence IS the replica's freshness window — one knob must not drive both.
/// Unset -> falls back to [`ENV_MANIFEST_POLL_MS`], then the SlateDB default.
const ENV_READER_MANIFEST_POLL_MS: &str = "HELIX_LSM_READER_MANIFEST_POLL_MS";
/// Env knob: base URL of the writer's HTTP gateway (e.g. `http://helix:6969`).
/// When set on a reader replica, the replica polls the writer's durable-commit
/// change feed (`GET /internal/changes`) and refreshes only the collections
/// that actually changed, so freshly-opened `DbReader`s take the IDLE poll
/// interval below as an S3 safety net instead of the serving cadence. Unset →
/// feed disabled; every reader `DbReader` polls S3 at
/// [`ENV_READER_MANIFEST_POLL_MS`] (previous behavior).
const ENV_WRITER_FEED_URL: &str = "HELIX_LSM_WRITER_FEED_URL";
/// Env knob: reader change-feed poll cadence in ms (default 1000). In-cluster
/// HTTP against the writer — no S3 requests.
const ENV_READER_FEED_POLL_MS: &str = "HELIX_LSM_READER_FEED_POLL_MS";
/// Env knob: idle-tier `DbReader` manifest poll in ms while the change feed is
/// enabled (default 300000). Pure safety net: the feed promotes changed
/// collections to the active tier long before this fires.
const ENV_READER_IDLE_POLL_MS: &str = "HELIX_LSM_READER_IDLE_POLL_MS";
/// Env knob: seconds a collection stays in the active (short-poll) tier after
/// its last feed signal before demotion back to the idle tier (default 600).
const ENV_READER_ACTIVE_TTL_SECS: &str = "HELIX_LSM_READER_ACTIVE_TTL_SECS";
/// Env knob: time budget in ms for the background reader poll-tier promotion
/// reopen (`storage_core::note_reader_read_and_maybe_promote`). The reopen is
/// normally a single S3 round-trip, but cold cache and open churn across many
/// resident collections can push healthy opens well past 2s. It runs off the
/// foreground read path, so a generous budget favors eventual successful
/// promotion over a chronically failing 8s timeout. Floor 1000; default 8000.
const ENV_READER_PROMOTE_BUDGET_MS: &str = "HELIX_LSM_READER_PROMOTE_BUDGET_MS";
/// Env knob: milliseconds a reader-replica collection may go without serving
/// a read before the read-gated poll-tier sweep demotes it back to the idle
/// tier (`ENV_READER_IDLE_POLL_MS`). A collection the write-feed reported a
/// change for within the same window is left on the active tier: it is still
/// being written, and its first reader must not start from a view one idle
/// poll old. Promotion back to the active tier
/// (`ENV_READER_MANIFEST_POLL_MS`) happens synchronously on the next read.
/// `0` disables read-gated tiering entirely: collections
/// are left wherever the write-feed / static config puts them (today's
/// behavior) — the kill switch. Default 600000 (10 min).
const ENV_READER_READ_IDLE_AFTER_MS: &str = "HELIX_LSM_READER_IDLE_AFTER_MS";
/// Env knob: SlateDB garbage-collector interval in seconds, applied to the six
/// object-store background sweeps (manifest, WAL, compacted SSTs, compactions,
/// WAL-fence scan, clone detach).
/// Raising this trims background LIST churn across many resident collections;
/// Per-directory knobs below override this.
const ENV_GC_INTERVAL_SECS: &str = "HELIX_LSM_GC_INTERVAL_SECS";
const ENV_GC_MANIFEST_INTERVAL_SECS: &str = "HELIX_LSM_GC_MANIFEST_INTERVAL_SECS";
const ENV_GC_WAL_INTERVAL_SECS: &str = "HELIX_LSM_GC_WAL_INTERVAL_SECS";
const ENV_GC_COMPACTED_INTERVAL_SECS: &str = "HELIX_LSM_GC_COMPACTED_INTERVAL_SECS";
const ENV_GC_COMPACTIONS_INTERVAL_SECS: &str = "HELIX_LSM_GC_COMPACTIONS_INTERVAL_SECS";
/// Env knob: SlateDB garbage-collector minimum object age in seconds, applied
/// to the four object-store directory GC tasks. SlateDB's default is 5 minutes,
/// which is too aggressive for long-lived reader snapshots and in-flight query
/// snapshots: a reader can still reference compacted SSTs after the writer's GC
/// considers them old enough. Storage is cheap; missing SSTs break reads.
const ENV_GC_MIN_AGE_SECS: &str = "HELIX_LSM_GC_MIN_AGE_SECS";
const ENV_GC_MANIFEST_MIN_AGE_SECS: &str = "HELIX_LSM_GC_MANIFEST_MIN_AGE_SECS";
const ENV_GC_WAL_MIN_AGE_SECS: &str = "HELIX_LSM_GC_WAL_MIN_AGE_SECS";
const ENV_GC_COMPACTED_MIN_AGE_SECS: &str = "HELIX_LSM_GC_COMPACTED_MIN_AGE_SECS";
const ENV_GC_COMPACTIONS_MIN_AGE_SECS: &str = "HELIX_LSM_GC_COMPACTIONS_MIN_AGE_SECS";
/// Env knob: interval in seconds for SlateDB's WAL-fence scanner, which LISTs
/// each collection's WAL directory for zero-byte fence objects. It runs
/// dry-run (log-only, never deletes — fence deletion is the data-loss hazard
/// SlateDB warns about), so the default 60s loop is a pure LIST cost per
/// resident collection. Falls back to [`ENV_GC_INTERVAL_SECS`].
const ENV_GC_WAL_FENCE_INTERVAL_SECS: &str = "HELIX_LSM_GC_WAL_FENCE_INTERVAL_SECS";
/// Env knob: interval in seconds for SlateDB's clone-detach pass. Helion never
/// clones databases, so the default 60s sweep per resident collection is pure
/// background churn. Falls back to [`ENV_GC_INTERVAL_SECS`].
const ENV_GC_DETACH_INTERVAL_SECS: &str = "HELIX_LSM_GC_DETACH_INTERVAL_SECS";
/// Env knob: target L0 SST size in bytes (SlateDB `l0_sst_size_bytes`). Larger →
/// fewer, bigger L0 files under sustained ingest (less compaction churn) at the
/// cost of more memory per memtable. Unset → SlateDB default.
const ENV_L0_SST_SIZE_BYTES: &str = "HELIX_LSM_L0_SST_SIZE_BYTES";
/// Env knob: max unflushed bytes before SlateDB applies write backpressure
/// (`max_unflushed_bytes`). Higher absorbs larger ingest bursts before stalling
/// writers, at the cost of RAM and longer recovery. Unset → SlateDB default.
const ENV_MAX_UNFLUSHED_BYTES: &str = "HELIX_LSM_MAX_UNFLUSHED_BYTES";
/// Env knob: per-collection open file-handle cap on the object-store cache
/// (`ObjectStoreCacheOptions::max_open_file_handles`). Each open collection's
/// SlateDB `CachedObjectStore` keeps up to this many cache files open; with many
/// concurrently-open collections the product (handles × collections) can exhaust
/// the process FD limit (EMFILE). Lowering this trades cache-hit fd footprint for
/// scale (more collections per writer). Unset → a bounded Helion default derived
/// from the maintenance FD gate and expected resident collection count.
const ENV_CACHE_MAX_OPEN_FILES: &str = "HELIX_LSM_CACHE_MAX_OPEN_FILES";
/// Env knob: ranged object-store cache misses (vendored fork
/// `ObjectStoreCacheOptions::ranged_miss`). A cold read of at most 1/16 of a
/// part GETs only its own bytes while the whole part fills in the background,
/// instead of waiting for the full part (4 MiB on readers, 64 MiB on the writer).
/// Unset/blank → on. Kill switch: `0`/`false`/`off`; any other value that is
/// not `1`/`true`/`yes`/`on` also turns it off.
const ENV_CACHE_RANGED_MISS: &str = "HELIX_LSM_CACHE_RANGED_MISS";
/// Env knob: how many object-store cache parts of one read are fetched at once
/// (vendored fork `ObjectStoreCacheOptions::part_fetch_concurrency`). A read
/// spanning several parts, such as a segment's `.hvec` sidecar blob, used to
/// fetch its missing parts one after the other. The bytes returned are the
/// same. The fork lowers the number until the parts in flight hold at most
/// 32 MiB per read, so readers (4 MiB parts) fetch up to 8 at once and the
/// writer (64 MiB parts) stays at one. Unset/blank → 8. Kill switch: `1` (one
/// part at a time); `0` or anything unparsable also means one at a time.
const ENV_CACHE_PART_FETCH_CONCURRENCY: &str = "HELIX_LSM_CACHE_PART_FETCH_CONCURRENCY";
/// Env knob: max concurrent background compactions
/// (`CompactorOptions::max_concurrent_compactions`). Lower bounds CPU/IO and
/// per-collection FD pressure from compaction when many collections compact at
/// once; higher speeds compaction at the cost of resources. Unset → SlateDB
/// default (4).
const ENV_MAX_CONCURRENT_COMPACTIONS: &str = "HELIX_LSM_MAX_CONCURRENT_COMPACTIONS";
/// Env knob: compactor coordinator poll cadence in ms
/// (`CompactorOptions::poll_interval`). Every tick re-reads manifest state from
/// S3 per open collection, so with many resident collections the default (5s)
/// dominates Tier-1 LIST request cost. Raising it delays when new L0 SSTs are
/// *noticed* for compaction scheduling. Once a collection reaches its L0 limit,
/// that delay becomes write latency because SlateDB cannot flush more memtables.
/// Unset/0 → SlateDB default (5s).
const ENV_COMPACTOR_POLL_MS: &str = "HELIX_LSM_COMPACTOR_POLL_MS";
/// Env knob: compactor `CommitCompacted` ticker cadence in ms
/// (`CompactorOptions::commit_compacted_interval`). Workers, including the
/// embedded worker, persist an intermediate `Compacted` result; this coordinator
/// ticker publishes that result to the manifest and removes its L0 sources. A
/// long interval therefore keeps L0 write backpressure active after the worker
/// has already finished. Unset/0 → SlateDB default (1s).
const ENV_COMPACTOR_COMMIT_COMPACTED_MS: &str = "HELIX_LSM_COMMIT_COMPACTED_MS";
/// Env knob: embedded compaction worker `.compactions` poll cadence in ms
/// (`CompactionWorkerOptions::compactions_poll_interval`). Same
/// S3-poll-per-tick-per-collection cost profile as the coordinator poll. It
/// controls how quickly an already-scheduled compaction job is picked up, which
/// also bounds how long an L0-saturated collection can remain write-stalled.
/// Unset/0 → SlateDB default (5s).
const ENV_COMPACTOR_WORKER_POLL_MS: &str = "HELIX_LSM_COMPACTOR_WORKER_POLL_MS";
/// Env knobs: the three options of the fork's size-tiered compaction scheduler
/// (`CompactorOptions::scheduler_options`). It merges sorted runs only when at
/// least MIN_SOURCES of them stand in a row, newest first, each no larger than
/// SIZE_THRESHOLD times the smallest so far in that row, and takes at most
/// MAX_SOURCES in one merge. Unset → the fork's defaults (4, 8, 4.0), under
/// which runs that roughly double in size are never merged, whatever share of
/// their rows a later run has deleted.
///
/// FOR A SMOKE POD ON A PRIVATE COPY ONLY. The three are read from the process
/// environment at every collection open, so one value reaches every collection
/// a writer opens, and none of them is a setting to run production on:
/// - a minimum of 2 with a large threshold makes any two runs a row, so after
///   every L0 compaction (each makes a new run) the scheduler merges every run
///   of the collection into the last one again: the whole collection is
///   rewritten once for every two L0 SSTs flushed;
/// - a large threshold with the maximum left at 8 makes every collection of 8
///   or more runs fail the scheduler's own backpressure check on its L0
///   compaction until merges have brought it under 8 runs: L0 fills to its
///   limit meanwhile and the collection's writes wait.
/// What is taken and what is left to its default:
/// [`compaction_scheduler_options_from_env`].
const ENV_COMPACTION_MIN_SOURCES: &str = "HELIX_LSM_COMPACTION_MIN_SOURCES";
const ENV_COMPACTION_MAX_SOURCES: &str = "HELIX_LSM_COMPACTION_MAX_SOURCES";
const ENV_COMPACTION_SIZE_THRESHOLD: &str = "HELIX_LSM_COMPACTION_SIZE_THRESHOLD";
/// Env knob: L0 SST count threshold that triggers compaction (`l0_max_ssts`).
/// Larger tolerates more L0 files before compacting (fewer, larger compactions
/// under sustained ingest) at the cost of more SSTs to merge on read; smaller
/// compacts sooner. Unset → SlateDB default (8).
const ENV_L0_MAX_SSTS: &str = "HELIX_LSM_L0_MAX_SSTS";
/// Env knob: per-key L0 overlap cap (`l0_max_ssts_per_key`). This bounds point
/// read amplification independently of the global L0 SST cap. Unset → SlateDB
/// default (8).
const ENV_L0_MAX_SSTS_PER_KEY: &str = "HELIX_LSM_L0_MAX_SSTS_PER_KEY";
/// Env knob: force an L0 flush after this many durable WAL flushes even when the
/// memtable has not reached `l0_sst_size_bytes`. Lower values bound the WAL
/// tail every reader open/refresh must replay. SlateDB's `DbBuilder::build`
/// rejects values below 4096 (`Invalid`), which would fail every writer open,
/// so values below 4096 are ignored (SlateDB default).
const ENV_MAX_WAL_FLUSHES_BEFORE_L0_FLUSH: &str = "HELIX_LSM_MAX_WAL_FLUSHES_BEFORE_L0_FLUSH";
/// Env knob: reader-replica WAL replay prefetch width
/// (`DbReaderOptions::wal_replay_sst_batch_size`). Each in-flight WAL SST is
/// one HEAD+GET, so this is the open-time S3 concurrency per collection.
/// Unset/0 → [`DEFAULT_READER_WAL_REPLAY_BATCH`]; capped at
/// [`MAX_READER_WAL_REPLAY_BATCH`].
const ENV_READER_WAL_REPLAY_BATCH: &str = "HELIX_LSM_READER_WAL_REPLAY_BATCH";
const DEFAULT_READER_WAL_REPLAY_BATCH: usize = 32;
const MAX_READER_WAL_REPLAY_BATCH: usize = 256;
/// Env switch: a reader's `DbReader` reads a WAL object of up to 256 KiB with
/// one GET after its HEAD instead of four ranged ones
/// ([`super::wal_whole_get`]). Unset → on; `0` restores the four reads. Read
/// at every reader open.
const ENV_READER_WAL_WHOLE_GET: &str = "HELIX_LSM_READER_WAL_WHOLE_GET";
/// Env switch: a reader replica moves a collection between its poll tiers by
/// telling the open `DbReader` its new interval, and asks it for a poll,
/// where it used to open a new `DbReader` (a checkpoint write and a replay of
/// the whole WAL tail for every tier change). Unset → the reopen; `1` → in
/// place. Read when a reader is opened.
const ENV_READER_TIER_IN_PLACE: &str = "HELIX_LSM_READER_TIER_IN_PLACE";
/// Env knob: number of parallel L0 flush workers (`l0_flush_parallelism`).
/// Unset → SlateDB default (4).
const ENV_L0_FLUSH_PARALLELISM: &str = "HELIX_LSM_L0_FLUSH_PARALLELISM";
/// Env knob: minimum SST key count before SlateDB writes a Bloom filter.
/// Lowering this helps point-heavy workloads with many small SSTs; raising it
/// shrinks SST metadata. Unset → SlateDB default (1000).
const ENV_MIN_FILTER_KEYS: &str = "HELIX_LSM_MIN_FILTER_KEYS";
/// Env knob: add prefix-aware SlateDB Bloom filters in addition to the default
/// full-key filter. These let prefix scans skip SSTs that cannot contain the
/// requested namespace / sparse posting-list prefix.
const ENV_PREFIX_FILTERS: &str = "HELIX_LSM_PREFIX_FILTERS";
const PREFIX_FILTER_BITS_PER_KEY: u32 = 10;
/// Env knob: SlateDB SST compression codec (`snappy`, `lz4`, or `zstd`).
/// Unset/empty/`none` → SlateDB default (no compression). The codec is recorded
/// in each SST's metadata, so old uncompressed SSTs and newly compressed SSTs can
/// coexist while rollout happens gradually.
const ENV_COMPRESSION_CODEC: &str = "HELIX_LSM_COMPRESSION_CODEC";
const ENV_SCAN_READ_AHEAD_BYTES: &str = "HELIX_LSM_SCAN_READ_AHEAD_BYTES";
const ENV_SCAN_CACHE_BLOCKS: &str = "HELIX_LSM_SCAN_CACHE_BLOCKS";
const ENV_SCAN_MAX_FETCH_TASKS: &str = "HELIX_LSM_SCAN_MAX_FETCH_TASKS";
const DEFAULT_SCAN_READ_AHEAD_BYTES: usize = 1024 * 1024;
const DEFAULT_SCAN_MAX_FETCH_TASKS: usize = 4;
/// Env knob: total in-memory SST *block* cache cap in MiB, SHARED across every
/// open SlateDB handle in this process. SlateDB's default `SplitCache` (foyer
/// block + meta) is built PER `Db` instance (512 MiB block + 128 MiB meta each),
/// so with the collection-manager LRU keeping many handles resident the in-memory
/// cache scales as `resident_handles × ~640 MiB` — tens of GiB of RSS on the
/// writer. A single process-shared cache bounds it to one pool regardless of
/// resident count. Unset → 512 MiB (matches SlateDB's per-instance block default,
/// now global).
const ENV_BLOCK_CACHE_MB: &str = "HELIX_LSM_BLOCK_CACHE_MB";
/// Env knob: total in-memory SST *meta* (index/filter/stats) cache cap in MiB,
/// shared across all handles. Unset → 128 MiB (SlateDB's per-instance meta
/// default, now global). See [`ENV_BLOCK_CACHE_MB`].
const ENV_META_CACHE_MB: &str = "HELIX_LSM_META_CACHE_MB";
/// Env knob: shard count of the shared meta cache. Unset/0 → one shard per
/// [`META_CACHE_SHARD_BYTES`] of capacity, capped at foyer's default (one per
/// host CPU). Kill switch: `16` restores the previous split on a 16-vCPU node.
const ENV_META_CACHE_SHARDS: &str = "HELIX_LSM_META_CACHE_SHARDS";
/// Floor on a meta-cache shard. Foyer splits capacity evenly across shards and
/// admits an entry only into its hash's shard. Decoded SST indexes run 5-8 MiB
/// and filters up to 13 MiB on prod collections (2026-10-21), so the per-CPU
/// split of 128 MiB (8 MiB shards on a 16-vCPU node) held about one index per
/// shard and re-read and re-decoded megabytes of metadata on most lookups.
/// 256 MiB rather than 64: with 64 MiB shards hash skew still overflowed one
/// shard with one large collection's hot metadata alone in ~8% of placements (review model,
/// 2026-10-22), and 4 vs 16 shards measured the same read throughput.
const META_CACHE_SHARD_BYTES: usize = 256 * 1024 * 1024;
/// Env knob: share compacted-SST block/index/filter entries in the shared cache
/// across every `Db`/`DbReader` handle (vendored fork switch). Without it each
/// handle keys entries by its own scope, so every reader reopen (poll-tier
/// promote/demote, refresh) starts its collection cold and strands the old
/// handle's entries until LRU eviction. Default on; `0` is the kill switch.
const ENV_SHARE_SST_CACHE: &str = "HELIX_LSM_SHARE_SST_CACHE";
/// Env knob: per-request S3 client timeout in ms. Unset/0 → object_store's
/// defaults (30s total-request, 5s connect). This bounds each HTTP attempt,
/// NOT a durable commit: SlateDB retries failed flushes above the client, so
/// an S3 outage mid-commit still stalls the writer until S3 recovers (chaos
/// finding #6 — deliberate, because a commit abandoned on timeout may still
/// land when S3 recovers: silent commit-on-recovery split-brain). This knob
/// only tunes how fast each attempt fails/retries — raise it for very large
/// SST GETs/PUTs on slow links, lower it for faster failover to a retry.
const ENV_S3_TIMEOUT_MS: &str = "HELIX_LSM_S3_TIMEOUT_MS";
/// Env knob: serve batched point reads through SlateDB's native `multi_get`
/// (vendored fork API: one ordered lookup pass sharing block fetches across
/// keys) instead of N concurrent independent `get`s. Kill switch: set `0` to
/// restore the buffered-individual-gets path. Default ON.
///
/// One caller asks for a multi-get whatever this says: the flat scan's
/// lookups of members a segment does not list, which almost never find a row
/// (`HELIX_FLAT_SCAN_ROW_BATCH=0` takes those back to one get each).
const ENV_MULTI_GET: &str = "HELIX_LSM_MULTI_GET";
/// Env knob: point deletes read their plan in batches (one multi-get for the
/// node rows, one for every segment ownership probe, concurrent prefix scans
/// for the rows to remove) instead of one SlateDB round trip per id and
/// segment. Kill switch: set `0` to restore the one-read-at-a-time path.
/// Default ON.
const ENV_DELETE_BATCH_READS: &str = "HELIX_LSM_DELETE_BATCH_READS";
/// Env knob: a writer commit (batch or transaction) still pending after
/// [`STALLED_WRITE_REFRESH_EVERY`], with L0 at its limit in the writer's view
/// of the manifest, re-reads the manifest, and looks again every such interval
/// until it returns. SlateDB parks writes while L0 is full and frozen
/// memtables cannot flush, and resumes them only once its own manifest poll
/// (`manifest_poll_interval`, 30 s in production) sees the L0 slots the
/// compactor freed. This issues that same poll early, which removes the wait
/// for the writer's poll (about 14 s on average, 28 s at most), not the time
/// the compactor takes to free the slots. At most one refresh per collection
/// per interval, each one LIST and two GETs. Kill switch: set `0` to wait for
/// the poll as before. Default ON.
const ENV_STALLED_WRITE_REFRESH: &str = "HELIX_LSM_STALLED_WRITE_REFRESH";
/// See [`ENV_STALLED_WRITE_REFRESH`].
const STALLED_WRITE_REFRESH_EVERY: std::time::Duration = std::time::Duration::from_secs(2);
/// Env switch: a memtable flush starts only into memory the pod has free as it
/// starts ([`flush_gate`]), and a collection whose flush waits for that takes
/// no writes until the flush is in L0. Nothing bounded how many collections
/// flush at the same moment, and a flush holds its SST encoded before it
/// frees its memtable: twelve at once took a 6 Gi writer over its limit in
/// 1.45 s. Kill switch: `0` opens every writer with no gate, as before. Read
/// once, when the process opens its first writer. Default ON.
const ENV_FLUSH_GATE: &str = "HELIX_FLUSH_GATE";
const LSM_METADATA_SIDECAR_OBJECT: &str = "_helix/metadata.json";

/// Marks a batched row read of the exact scorer on the collection at `root`
/// (the SlateDB path its store was opened with) in flight until the value is
/// dropped. The part cache makes the fill of a part of that collection missed
/// meanwhile, by whatever read, wait its turn: one fetch at a time, going on
/// after the mark is gone. A miss of any other collection's part, like a miss
/// with no mark in flight, starts its fill at once. A fill it leaves out (too
/// much waiting, or the budget spent when the turn comes) is
/// `helix_lsm_cache_ranged_fill_total{outcome="skipped_in_batch"}`;
/// `outcome="skipped"` counts every other skip, as it did.
pub(crate) fn batch_row_read_in_flight(
    root: &str,
) -> slatedb::cached_object_store_stats::BatchRowRead {
    slatedb::cached_object_store_stats::ranged_miss_stats().batch_row_read(root.into())
}

/// Whether batched point reads use SlateDB `multi_get` ([`ENV_MULTI_GET`],
/// default true). `pub(crate)` so the reader replica honors the same knob.
pub(crate) fn lsm_multi_get_enabled() -> bool {
    #[cfg(test)]
    if let Some(forced) = MULTI_GET_FOR_TEST.get() {
        return forced;
    }
    env_bool_opt(ENV_MULTI_GET).unwrap_or(true)
}

/// Whether point deletes batch their reads ([`ENV_DELETE_BATCH_READS`],
/// default true).
pub(crate) fn lsm_delete_batch_reads_enabled() -> bool {
    env_bool_opt(ENV_DELETE_BATCH_READS).unwrap_or(true)
}

/// Whether a stalled writer commit refreshes the manifest
/// ([`ENV_STALLED_WRITE_REFRESH`], default true).
fn lsm_stalled_write_refresh_enabled() -> bool {
    env_bool_opt(ENV_STALLED_WRITE_REFRESH).unwrap_or(true)
}

/// Apply [`ENV_S3_TIMEOUT_MS`] to an S3 builder. Unset/blank/0/unparsable →
/// builder unchanged (object_store's default 30s request timeout applies).
/// `pub(crate)` so the reader replica's S3 open applies the same knob.
pub(crate) fn apply_s3_timeout(builder: AmazonS3Builder) -> AmazonS3Builder {
    match std::env::var(ENV_S3_TIMEOUT_MS)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|ms| *ms > 0)
    {
        Some(ms) => builder.with_client_options(
            object_store::ClientOptions::new().with_timeout(std::time::Duration::from_millis(ms)),
        ),
        None => builder,
    }
}

/// One process-shared SlateDB block/meta cache for every open collection handle.
///
/// SlateDB builds a fresh `SplitCache` per `Db` by default, so N resident handles
/// hold N × (block + meta) bytes of in-memory cache. On the LSM writer the
/// collection-manager LRU keeps up to `HELIX_MAX_OPEN_COLLECTIONS` handles
/// resident, so the per-handle default silos sum to tens of GiB of RSS. Sharing
/// ONE cache across all handles bounds the in-memory cache to a single pool
/// regardless of resident count: SST cache keys are per-ULID (globally unique),
/// so cross-collection sharing is collision-free, and a miss simply falls through
/// to the object store. `DbBuilder::with_db_cache` replaces the per-instance
/// default with this shared `Arc`.
pub(crate) fn shared_db_cache() -> Arc<dyn DbCache> {
    static CACHE: OnceLock<Arc<dyn DbCache>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            const MIB: usize = 1024 * 1024;
            let block_bytes = env_usize_opt(ENV_BLOCK_CACHE_MB).unwrap_or(512).max(1) * MIB;
            let meta_bytes = env_usize_opt(ENV_META_CACHE_MB).unwrap_or(128).max(1) * MIB;
            let meta_shards = meta_cache_shards(meta_bytes, FoyerCacheOptions::default().shards);
            // Set before the first handle is built: wrappers read it at build time.
            slatedb::db_cache::set_share_compacted_entries(
                env_bool_opt(ENV_SHARE_SST_CACHE).unwrap_or(true),
            );
            let (block, cache) = build_db_cache_parts(block_bytes, meta_bytes, meta_shards);
            // Kept for its byte accounting: the split cache answers the
            // block and meta parts summed.
            let _ = SHARED_BLOCK_CACHE.set(block);
            cache
        })
        .clone()
}

/// The block part of [`shared_db_cache`], once that is built.
static SHARED_BLOCK_CACHE: OnceLock<Arc<dyn DbCache>> = OnceLock::new();

/// Sets `helix_lsm_block_cache_used_bytes` and
/// `helix_lsm_block_cache_capacity_bytes` from the shared block cache's own
/// accounting, and nothing before the process's first LSM open has built
/// that cache. One cache serves every open collection, so the two series
/// carry no collection. The cache counts no evictions: used at its capacity
/// is the nearest sign of them.
pub(crate) fn export_shared_block_cache_usage() {
    let Some(block) = SHARED_BLOCK_CACHE.get() else {
        return;
    };
    if let CacheUsageSnapshot::Ready {
        used_bytes,
        capacity_bytes: Some(capacity_bytes),
    } = block.usage_snapshot().memory
    {
        metrics::gauge!("helix_lsm_block_cache_used_bytes").set(used_bytes as f64);
        metrics::gauge!("helix_lsm_block_cache_capacity_bytes").set(capacity_bytes as f64);
    }
}

/// [`ENV_FLUSH_GATE`]: on unless set to a false value.
fn flush_gate_enabled() -> bool {
    env_bool_opt(ENV_FLUSH_GATE).unwrap_or(true)
}

/// The one flush gate of the process, shared by every writer it opens so
/// that their flushes are bounded together, or none with [`ENV_FLUSH_GATE`]
/// off. It reads the pod's free memory as the shutdown close does
/// (`collection_manager::free_memory_bytes`).
fn shared_flush_gate() -> Option<Arc<dyn slatedb::L0FlushGate>> {
    static GATE: OnceLock<Option<Arc<dyn slatedb::L0FlushGate>>> = OnceLock::new();
    GATE.get_or_init(|| {
        let on = flush_gate_enabled();
        tracing::info!(
            on,
            switch = ENV_FLUSH_GATE,
            "LSM flush gate: a memtable flush starts only into free memory"
        );
        on.then(|| {
            Arc::new(flush_gate::FlushGate::new(Box::new(
                super::collection_manager::free_memory_bytes,
            ))) as Arc<dyn slatedb::L0FlushGate>
        })
    })
    .clone()
}

/// The gate a writer opened now by this thread gets: [`shared_flush_gate`],
/// or in a test what [`with_flush_gate_for_test`] set.
fn flush_gate_for_open() -> Option<Arc<dyn slatedb::L0FlushGate>> {
    #[cfg(test)]
    {
        if let Some(gate) = FLUSH_GATE_FOR_TEST.with(|gate| gate.borrow().clone()) {
            return gate;
        }
    }
    shared_flush_gate()
}

/// Shard count for the shared meta cache; see [`ENV_META_CACHE_SHARDS`].
fn meta_cache_shards(meta_bytes: usize, default_shards: usize) -> usize {
    env_usize_opt(ENV_META_CACHE_SHARDS)
        .filter(|shards| *shards > 0)
        .unwrap_or_else(|| (meta_bytes / META_CACHE_SHARD_BYTES).clamp(1, default_shards.max(1)))
}

/// The cache alone, as the tests build one of their own.
#[cfg(test)]
pub(crate) fn build_db_cache(
    block_bytes: usize,
    meta_bytes: usize,
    meta_shards: usize,
) -> Arc<dyn DbCache> {
    build_db_cache_parts(block_bytes, meta_bytes, meta_shards).1
}

/// The block part, and the split cache it goes into. Block cache keeps
/// foyer's default sharding (4 KiB entries); the meta cache takes
/// `meta_shards`.
fn build_db_cache_parts(
    block_bytes: usize,
    meta_bytes: usize,
    meta_shards: usize,
) -> (Arc<dyn DbCache>, Arc<dyn DbCache>) {
    let block: Arc<dyn DbCache> = Arc::new(FoyerCache::new_with_opts(FoyerCacheOptions {
        max_capacity: block_bytes as u64,
        ..Default::default()
    }));
    let meta = FoyerCache::new_with_opts(FoyerCacheOptions {
        max_capacity: meta_bytes as u64,
        shards: meta_shards.max(1),
    });
    let cache = Arc::new(
        SplitCache::new()
            .with_block_cache(Some(Arc::clone(&block)))
            .with_meta_cache(Some(Arc::new(meta)))
            .build(),
    ) as Arc<dyn DbCache>;
    (block, cache)
}

fn target_bytes(target: &PrefixTarget) -> &[u8] {
    match target {
        PrefixTarget::Point(bytes) | PrefixTarget::Prefix(bytes) => bytes.as_ref(),
    }
}

fn namespace_prefix_len(bytes: &[u8]) -> Option<usize> {
    bytes.iter().position(|byte| *byte == 0).map(|idx| idx + 1)
}

fn known_dup_namespace(name: &[u8]) -> bool {
    matches!(name, b"out_edges" | b"in_edges")
        || name.starts_with(b"midx_")
        || name.starts_with(b"sparse_inv_")
        || name.starts_with(b"pidxb_")
}

fn dup_key_prefix_len(bytes: &[u8]) -> Option<usize> {
    let ns_end = namespace_prefix_len(bytes)?;
    let namespace = &bytes[..ns_end.saturating_sub(1)];
    if !known_dup_namespace(namespace) {
        return None;
    }
    let len_end = ns_end.checked_add(4)?;
    let key_len_bytes: [u8; 4] = bytes.get(ns_end..len_end)?.try_into().ok()?;
    let key_len = u32::from_be_bytes(key_len_bytes) as usize;
    let key_end = len_end.checked_add(key_len)?;
    (bytes.len() >= key_end).then_some(key_end)
}

struct HelixNamespacePrefixExtractor;

impl PrefixExtractor for HelixNamespacePrefixExtractor {
    fn name(&self) -> &str {
        "helion-ns-v1"
    }

    fn prefix_len(&self, target: &PrefixTarget) -> Option<usize> {
        namespace_prefix_len(target_bytes(target))
    }
}

struct HelixDupKeyPrefixExtractor;

impl PrefixExtractor for HelixDupKeyPrefixExtractor {
    fn name(&self) -> &str {
        "helion-dup-v1"
    }

    fn prefix_len(&self, target: &PrefixTarget) -> Option<usize> {
        dup_key_prefix_len(target_bytes(target))
    }
}

pub(crate) fn prefix_filter_policies_from_env() -> Option<Vec<Arc<dyn FilterPolicy>>> {
    if !env_bool_opt(ENV_PREFIX_FILTERS).unwrap_or(false) {
        return None;
    }
    Some(vec![
        Arc::new(BloomFilterPolicy::new(PREFIX_FILTER_BITS_PER_KEY)),
        Arc::new(
            BloomFilterPolicy::new(PREFIX_FILTER_BITS_PER_KEY)
                .with_prefix_extractor(Arc::new(HelixNamespacePrefixExtractor))
                .with_whole_key_filtering(false),
        ),
        Arc::new(
            BloomFilterPolicy::new(PREFIX_FILTER_BITS_PER_KEY)
                .with_prefix_extractor(Arc::new(HelixDupKeyPrefixExtractor))
                .with_whole_key_filtering(false),
        ),
    ])
}

struct HelixCounterMergeOperator;

impl HelixCounterMergeOperator {
    fn callback_error(error: GraphError) -> MergeOperatorError {
        MergeOperatorError::Callback {
            message: error.to_string(),
        }
    }
}

impl MergeOperator for HelixCounterMergeOperator {
    fn merge(
        &self,
        key: &Bytes,
        existing_value: Option<Bytes>,
        operand: Bytes,
    ) -> Result<Bytes, MergeOperatorError> {
        self.merge_batch(key, existing_value, std::slice::from_ref(&operand))
    }

    /// Operands are signed deltas. SlateDB also calls this with no base value
    /// to fold a write batch's or a memtable's operands, and stores that fold
    /// as another operand, so it must keep its sign: flooring it at zero lost
    /// every decrement. Only a real base value is floored at zero.
    fn merge_batch(
        &self,
        _key: &Bytes,
        existing_value: Option<Bytes>,
        operands: &[Bytes],
    ) -> Result<Bytes, MergeOperatorError> {
        let mut delta = 0i64;
        for operand in operands {
            let next = decode_lsm_counter_delta(operand).map_err(Self::callback_error)?;
            delta = delta.checked_add(next).ok_or_else(|| {
                Self::callback_error(GraphError::StorageError(
                    "metadata counter delta overflow".to_string(),
                ))
            })?;
        }
        let Some(base) = existing_value else {
            return Ok(Bytes::copy_from_slice(&encode_lsm_counter_delta(delta)));
        };
        let value = decode_lsm_counter_value(&base).map_err(Self::callback_error)?;
        let next = apply_lsm_counter_delta(value, delta).map_err(Self::callback_error)?;
        Ok(Bytes::copy_from_slice(&encode_lsm_counter_value(next)))
    }
}

pub(crate) fn shared_merge_operator() -> Arc<dyn MergeOperator + Send + Sync> {
    static MERGE_OPERATOR: OnceLock<Arc<dyn MergeOperator + Send + Sync>> = OnceLock::new();
    MERGE_OPERATOR
        .get_or_init(|| Arc::new(HelixCounterMergeOperator))
        .clone()
}

/// Parse a `usize` env knob, returning `None` when unset/blank/unparsable so the
/// caller keeps the SlateDB default (override-only semantics).
fn env_usize_opt(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
}

/// The most one writer holds unflushed before it stalls its writes
/// (`max_unflushed_bytes`): [`ENV_MAX_UNFLUSHED_BYTES`], or SlateDB's default.
pub(crate) fn lsm_max_unflushed_bytes() -> usize {
    env_usize_opt(ENV_MAX_UNFLUSHED_BYTES)
        .unwrap_or_else(|| Settings::default().max_unflushed_bytes)
}

/// Parse a millisecond env knob into a `Duration`, returning `None` when
/// unset/blank/unparsable/zero so the caller keeps the SlateDB default
/// (override-only semantics).
fn env_duration_ms_opt(key: &str) -> Option<std::time::Duration> {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
        .map(std::time::Duration::from_millis)
}

/// One of the compaction scheduler's settings as the environment gives it:
/// `None` when unset or blank, `Err` with the text when it does not parse.
fn scheduler_setting<T: std::str::FromStr>(key: &str) -> Option<Result<T, String>> {
    let raw = std::env::var(key).ok()?;
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    Some(raw.parse::<T>().map_err(|_| raw.to_string()))
}

/// Says, once a process for each setting, that a compaction scheduler setting
/// was left to the fork's default.
fn warn_scheduler_setting_ignored(key: &'static str, why: &str) {
    static WARNED: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
    let mut warned = WARNED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if warned.contains(&key) {
        return;
    }
    warned.push(key);
    tracing::warn!(
        setting = key,
        reason = why,
        "compaction scheduler setting ignored, the fork's default stands"
    );
}

/// The compaction scheduler's options that the environment sets and the
/// scheduler can merge by, under the fork's own keys
/// ([`ENV_COMPACTION_MIN_SOURCES`] and its two neighbours). Empty when none is
/// set, so the fork's defaults stand. `l0_floor` is the L0 SST count at which
/// the writer stops flushing ([`l0_stall_floor`]). Left to its default, with a
/// WARN:
/// - a minimum under 2 (one run is no merge: the fork would rewrite runs
///   alone, poll after poll) or above `l0_floor` (L0 is compacted only once it
///   holds the minimum, which it could then never reach: no L0 SST would be
///   compacted again and every write would wait at the floor);
/// - a maximum under the minimum in force (the fork cuts a row of runs down to
///   the maximum, so it would merge fewer runs than the minimum asks for, and
///   cut down to 0 the row is empty, on which its scheduler panics);
/// - a minimum above the maximum in force, for the same reason;
/// - a threshold that is not a finite number above 1 (at 1 or under, no run
///   larger than the one before it is ever taken).
fn compaction_scheduler_options_from_env(l0_floor: usize) -> Vec<(&'static str, String)> {
    let defaults = slatedb::config::SizeTieredCompactionSchedulerOptions::default();
    let mut ignored: Vec<(&'static str, String)> = Vec::new();

    let min = match scheduler_setting::<usize>(ENV_COMPACTION_MIN_SOURCES) {
        Some(Ok(min)) if (2..=l0_floor).contains(&min) => Some(min),
        Some(Ok(min)) => {
            ignored.push((
                ENV_COMPACTION_MIN_SOURCES,
                format!("{min} is not between 2 and the L0 limit of {l0_floor}"),
            ));
            None
        }
        Some(Err(text)) => {
            ignored.push((
                ENV_COMPACTION_MIN_SOURCES,
                format!("{text:?} is not a whole number"),
            ));
            None
        }
        None => None,
    };
    let min_in_force = min.unwrap_or(defaults.min_compaction_sources);
    let max = match scheduler_setting::<usize>(ENV_COMPACTION_MAX_SOURCES) {
        Some(Ok(max)) if max >= min_in_force => Some(max),
        Some(Ok(max)) => {
            ignored.push((
                ENV_COMPACTION_MAX_SOURCES,
                format!("{max} is under the minimum in force, {min_in_force}"),
            ));
            None
        }
        Some(Err(text)) => {
            ignored.push((
                ENV_COMPACTION_MAX_SOURCES,
                format!("{text:?} is not a whole number"),
            ));
            None
        }
        None => None,
    };
    let max_in_force = max.unwrap_or(defaults.max_compaction_sources);
    let min = match min {
        Some(min) if min > max_in_force => {
            ignored.push((
                ENV_COMPACTION_MIN_SOURCES,
                format!("{min} is above the maximum in force, {max_in_force}"),
            ));
            None
        }
        other => other,
    };
    let threshold = match scheduler_setting::<f32>(ENV_COMPACTION_SIZE_THRESHOLD) {
        Some(Ok(threshold)) if threshold.is_finite() && threshold > 1.0 => Some(threshold),
        Some(Ok(threshold)) => {
            ignored.push((
                ENV_COMPACTION_SIZE_THRESHOLD,
                format!("{threshold} is not a finite number above 1"),
            ));
            None
        }
        Some(Err(text)) => {
            ignored.push((
                ENV_COMPACTION_SIZE_THRESHOLD,
                format!("{text:?} is not a number"),
            ));
            None
        }
        None => None,
    };
    for (key, why) in &ignored {
        warn_scheduler_setting_ignored(key, why);
    }

    let mut options = Vec::new();
    if let Some(min) = min {
        options.push(("min_compaction_sources", min.to_string()));
    }
    if let Some(max) = max {
        options.push(("max_compaction_sources", max.to_string()));
    }
    if let Some(threshold) = threshold {
        options.push(("include_size_threshold", threshold.to_string()));
    }
    options
}

fn env_bool_opt(key: &str) -> Option<bool> {
    let value = std::env::var(key).ok()?;
    let normalized = value.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn compression_codec_from_env() -> Option<CompressionCodec> {
    let value = std::env::var(ENV_COMPRESSION_CODEC).ok()?;
    let normalized = value.trim().to_ascii_lowercase();
    if normalized.is_empty() || matches!(normalized.as_str(), "none" | "off" | "false" | "0") {
        return None;
    }
    normalized.parse::<CompressionCodec>().map_or_else(
        |_| {
            eprintln!(
                "WARN {ENV_COMPRESSION_CODEC}={value:?} is not supported; expected snappy, lz4, zstd, or none"
            );
            None
        },
        Some,
    )
}

fn duration_secs_from_env(key: &str) -> Option<std::time::Duration> {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
        .map(std::time::Duration::from_secs)
}

fn set_gc_directory_options(
    target: &mut Option<GarbageCollectorDirectoryOptions>,
    interval: Option<std::time::Duration>,
    min_age: Option<std::time::Duration>,
) {
    let mut options = (*target).unwrap_or_default();
    if let Some(interval) = interval {
        options.interval = Some(interval);
    }
    if let Some(min_age) = min_age {
        options.min_age = min_age;
    }
    *target = Some(options);
}

fn gc_options_from_env() -> Option<GarbageCollectorOptions> {
    let default_interval = duration_secs_from_env(ENV_GC_INTERVAL_SECS);
    let manifest_interval =
        duration_secs_from_env(ENV_GC_MANIFEST_INTERVAL_SECS).or(default_interval);
    let wal_interval = duration_secs_from_env(ENV_GC_WAL_INTERVAL_SECS).or(default_interval);
    let compacted_interval =
        duration_secs_from_env(ENV_GC_COMPACTED_INTERVAL_SECS).or(default_interval);
    let compactions_interval =
        duration_secs_from_env(ENV_GC_COMPACTIONS_INTERVAL_SECS).or(default_interval);
    let wal_fence_interval =
        duration_secs_from_env(ENV_GC_WAL_FENCE_INTERVAL_SECS).or(default_interval);
    let detach_interval = duration_secs_from_env(ENV_GC_DETACH_INTERVAL_SECS).or(default_interval);
    let default_min_age = duration_secs_from_env(ENV_GC_MIN_AGE_SECS);
    let manifest_min_age = duration_secs_from_env(ENV_GC_MANIFEST_MIN_AGE_SECS).or(default_min_age);
    let wal_min_age = duration_secs_from_env(ENV_GC_WAL_MIN_AGE_SECS).or(default_min_age);
    let compacted_min_age =
        duration_secs_from_env(ENV_GC_COMPACTED_MIN_AGE_SECS).or(default_min_age);
    let compactions_min_age =
        duration_secs_from_env(ENV_GC_COMPACTIONS_MIN_AGE_SECS).or(default_min_age);
    if manifest_interval.is_none()
        && wal_interval.is_none()
        && compacted_interval.is_none()
        && compactions_interval.is_none()
        && wal_fence_interval.is_none()
        && detach_interval.is_none()
        && manifest_min_age.is_none()
        && wal_min_age.is_none()
        && compacted_min_age.is_none()
        && compactions_min_age.is_none()
    {
        return None;
    }

    let mut options = GarbageCollectorOptions::default();
    if manifest_interval.is_some() || manifest_min_age.is_some() {
        set_gc_directory_options(
            &mut options.manifest_options,
            manifest_interval,
            manifest_min_age,
        );
    }
    if wal_interval.is_some() || wal_min_age.is_some() {
        set_gc_directory_options(&mut options.wal_options, wal_interval, wal_min_age);
    }
    if compacted_interval.is_some() || compacted_min_age.is_some() {
        set_gc_directory_options(
            &mut options.compacted_options,
            compacted_interval,
            compacted_min_age,
        );
    }
    if compactions_interval.is_some() || compactions_min_age.is_some() {
        set_gc_directory_options(
            &mut options.compactions_options,
            compactions_interval,
            compactions_min_age,
        );
    }
    // Interval only: min_age keeps the SlateDB default and dry_run stays true
    // (log-only) — this knob only slows the fence scanner's LIST loop, it never
    // turns fence deletion on.
    if wal_fence_interval.is_some() {
        set_gc_directory_options(&mut options.wal_fence_options, wal_fence_interval, None);
    }
    if let Some(interval) = detach_interval {
        let mut detach = options.detach_options.unwrap_or_default();
        detach.interval = Some(interval);
        options.detach_options = Some(detach);
    }
    Some(options)
}

fn lsm_scan_options_with_cache_blocks(cache_blocks: bool) -> ScanOptions {
    let read_ahead_bytes = env_usize_opt(ENV_SCAN_READ_AHEAD_BYTES)
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_SCAN_READ_AHEAD_BYTES);
    let max_fetch_tasks = env_usize_opt(ENV_SCAN_MAX_FETCH_TASKS)
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_SCAN_MAX_FETCH_TASKS);
    ScanOptions::default()
        .with_read_ahead_bytes(read_ahead_bytes)
        .with_cache_blocks(cache_blocks)
        .with_max_fetch_tasks(max_fetch_tasks)
}

pub(crate) fn lsm_scan_options() -> ScanOptions {
    let cache_blocks = env_bool_opt(ENV_SCAN_CACHE_BLOCKS).unwrap_or(true);
    lsm_scan_options_with_cache_blocks(cache_blocks)
}

pub(crate) fn lsm_streaming_scan_options() -> ScanOptions {
    lsm_scan_options_with_cache_blocks(false)
}

pub(crate) fn scan_prefix_for_namespace_range(
    namespace_prefix: &[u8],
    range: &KeyRange,
) -> Option<Vec<u8>> {
    match (&range.start, &range.end) {
        (Bound::Unbounded, Bound::Unbounded) => Some(namespace_prefix.to_vec()),
        (Bound::Included(start), Bound::Excluded(end))
            if next_prefix(start).as_deref() == Some(end.as_slice()) =>
        {
            let mut prefix = Vec::with_capacity(namespace_prefix.len() + start.len());
            prefix.extend_from_slice(namespace_prefix);
            prefix.extend_from_slice(start);
            Some(prefix)
        }
        (Bound::Included(start), Bound::Unbounded) if next_prefix(start).is_none() => {
            let mut prefix = Vec::with_capacity(namespace_prefix.len() + start.len());
            prefix.extend_from_slice(namespace_prefix);
            prefix.extend_from_slice(start);
            Some(prefix)
        }
        _ => None,
    }
}

pub(crate) fn range_covers_all_logical_keys(range: &KeyRange) -> bool {
    matches!(
        (&range.start, &range.end),
        (Bound::Unbounded, Bound::Unbounded)
    ) || matches!((&range.start, &range.end), (Bound::Included(start), Bound::Unbounded) if start.is_empty())
}

/// Env knob: expected number of collections open concurrently — the divisor
/// that bounds the PER-COLLECTION cache cap so N per-collection cache subdirs
/// sum to the configured TOTAL cap. Reuses the collection-manager open-collection
/// LRU cap so the two stay coupled (default 256, see
/// `collection_manager::max_open_collections`).
const ENV_MAX_OPEN_COLLECTIONS: &str = "HELIX_MAX_OPEN_COLLECTIONS";
/// Mirror of `collection_manager::max_open_collections`'s default. Kept in sync
/// by reusing the SAME env var name so an operator setting it once tunes both
/// the LRU residency and the cache-cap divisor.
const DEFAULT_MAX_OPEN_COLLECTIONS: usize = 256;

/// Expected number of concurrently-open collections, the cache-cap divisor.
/// Mirrors `collection_manager::max_open_collections` (same env var, same
/// default) so the per-collection cache budget divides the total cap across
/// exactly the collections the LRU keeps resident. Floored at 1.
fn max_open_collections_for_cache() -> usize {
    std::env::var(ENV_MAX_OPEN_COLLECTIONS)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_OPEN_COLLECTIONS)
        .max(1)
}

/// Env knob: divisor for the PER-COLLECTION cache cap, decoupled from the
/// residency cap. Unset/0 keeps the old coupling (`HELIX_MAX_OPEN_COLLECTIONS`).
/// Setting it lets an operator raise the open-handle cap without shrinking each
/// collection's cache share: the cap is `total / divisor` whatever max-open is.
/// The resident sum can then exceed the total (up to `max_open / divisor` x),
/// which only grows reclaimable page cache and /cache disk use, not heap.
const ENV_CACHE_COLLECTION_DIVISOR: &str = "HELIX_LSM_CACHE_COLLECTION_DIVISOR";

fn cache_collection_divisor() -> usize {
    env_usize_opt(ENV_CACHE_COLLECTION_DIVISOR)
        .filter(|divisor| *divisor > 0)
        .unwrap_or_else(max_open_collections_for_cache)
}

fn cache_max_open_file_handles_default() -> usize {
    let max_open = max_open_collections_for_cache();
    let fd_high = std::env::var("HELIX_MAINTENANCE_FD_HIGH")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(4096);
    let budget = fd_high.saturating_sub(fd_high / 2).max(max_open);
    (budget / max_open).clamp(8, 128)
}

fn cgroup_memory_limit_bytes() -> Option<usize> {
    [CGROUP_V2_MEMORY_MAX, CGROUP_V1_MEMORY_LIMIT]
        .into_iter()
        .find_map(|path| {
            let value = std::fs::read_to_string(path).ok()?;
            let value = value.trim();
            if value == "max" {
                return None;
            }
            value.parse::<usize>().ok()
        })
}

fn cache_memory_max_pct() -> usize {
    let pct = env_usize_opt(ENV_CACHE_MEMORY_MAX_PCT).unwrap_or(DEFAULT_CACHE_MEMORY_MAX_PCT);
    if pct == 0 {
        0
    } else {
        pct.clamp(1, 100)
    }
}

fn memory_bounded_cache_cap(
    configured_cap: usize,
    memory_limit: Option<usize>,
    memory_max_pct: usize,
) -> usize {
    if memory_max_pct == 0 {
        return configured_cap;
    }
    let Some(limit) = memory_limit else {
        return configured_cap;
    };
    let memory_cap = limit.saturating_mul(memory_max_pct).saturating_div(100);
    configured_cap.min(memory_cap.max(1))
}

/// Says, once a process, that the configured cache share of a collection was
/// cut to what the cache volume holds (see the VOLUME BOUND in
/// [`cache_options_from_env`]).
fn warn_cache_share_cut_to_volume(cache_dir: &str, configured: usize, held: usize) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            cache_dir,
            configured_share_bytes = configured,
            share_bytes = held,
            max_open = max_open_collections_for_cache(),
            "cache share per collection cut to what the cache volume's budget holds with every handle open"
        );
    });
}

/// Sanitize a collection identifier (its SlateDB path, e.g. `helion/my_coll`)
/// into a SINGLE filesystem-safe directory component: every char outside
/// `[A-Za-z0-9._-]` (notably the `/` path separator) becomes `_`. This yields a
/// per-collection cache root that the SlateDB evictor's whole-root `WalkDir`
/// scan cannot escape, so one collection's evictor never observes or deletes
/// another collection's cached files.
fn sanitize_cache_component(collection_id: &str) -> String {
    collection_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

pub(crate) fn cache_root_from_env() -> Option<std::path::PathBuf> {
    let cache_dir = std::env::var(ENV_CACHE_DIR).ok()?;
    let cache_dir = cache_dir.trim();
    if cache_dir.is_empty() {
        return None;
    }
    Some(std::path::PathBuf::from(cache_dir))
}

pub(crate) fn cache_subtree_for_lsm_path(lsm_path: &str) -> Option<std::path::PathBuf> {
    Some(cache_root_from_env()?.join(sanitize_cache_component(lsm_path)))
}

fn metadata_sidecar_object_path(collection_path: &str) -> ObjPath {
    ObjPath::from(format!(
        "{}/{}",
        collection_path.trim_matches('/'),
        LSM_METADATA_SIDECAR_OBJECT
    ))
}

pub(crate) struct S3StoreConfig<'a> {
    pub(crate) bucket: &'a str,
    pub(crate) region: Option<&'a str>,
    pub(crate) endpoint: Option<&'a str>,
    pub(crate) allow_http: bool,
}

/// One S3 client per (bucket, region, endpoint, allow_http) for the whole
/// process: every writer/reader open and every env-based helper (tombstone
/// HEAD, prefix probe, purge, catalog LIST) shares it, so a cold open no
/// longer pays a fresh client build + IRSA credential exchange per call.
static S3_STORES: std::sync::LazyLock<
    std::sync::Mutex<HashMap<(String, Option<String>, Option<String>, bool), Arc<dyn ObjectStore>>>,
> = std::sync::LazyLock::new(Default::default);

pub(crate) fn build_s3_store(
    config: &S3StoreConfig<'_>,
) -> Result<Arc<dyn ObjectStore>, BackendError> {
    let key = (
        config.bucket.to_string(),
        config.region.map(str::to_string),
        config.endpoint.map(str::to_string),
        config.allow_http,
    );
    if let Some(store) = S3_STORES
        .lock()
        .ok()
        .and_then(|stores| stores.get(&key).cloned())
    {
        return Ok(store);
    }
    let store = build_s3_store_uncached(config)?;
    if let Ok(mut stores) = S3_STORES.lock() {
        stores.insert(key, Arc::clone(&store));
    }
    Ok(store)
}

fn build_s3_store_uncached(
    config: &S3StoreConfig<'_>,
) -> Result<Arc<dyn ObjectStore>, BackendError> {
    let mut builder = AmazonS3Builder::from_env().with_bucket_name(config.bucket);
    if let Some(r) = config.region {
        builder = builder.with_region(r);
    }
    if let Some(endpoint) = config.endpoint {
        builder = builder.with_endpoint(endpoint);
    }
    if config.allow_http {
        builder = builder.with_allow_http(true);
    }
    builder = apply_s3_timeout(builder);
    Ok(Arc::new(builder.build().map_err(io)?))
}

/// Local-FS object-store cache options for ONE collection, or `None` when
/// [`ENV_CACHE_DIR`] is unset/empty (cache-less open). Shared by the writer
/// [`Settings`] and the reader [`DbReaderOptions`] so both caches configure
/// identically.
///
/// PER-COLLECTION cache root. SlateDB builds one `CachedObjectStore` (and one
/// background evictor) per `Db`/`DbReader`, and that evictor enforces its size
/// cap by `WalkDir`-scanning and `remove_file`-ing across its ENTIRE
/// `root_folder` (slatedb 0.13.1 `FsCacheEvictor::scan_entries` /
/// `evict_to_target_size` in `cached_object_store/storage_fs.rs`). A SHARED root
/// therefore makes every collection's evictor observe and evict every OTHER
/// collection's cached files (cross-instance thrash, cold-miss spikes). Scoping
/// the root to `<HELIX_LSM_CACHE_DIR>/<sanitized collection>` confines each
/// evictor to its own subtree.
///
/// CAP MATH (total on-disk cache must NOT exceed the configured total cap): the
/// total cap (`HELIX_LSM_CACHE_MAX_BYTES`, else 8 GiB) is DIVIDED by the expected
/// number of concurrently-open collections (`HELIX_MAX_OPEN_COLLECTIONS`, the
/// collection-manager LRU cap, default 256) to get each collection's cap. The
/// LRU keeps at most `max_open` collections resident, so the sum across resident
/// per-collection subdirs is `max_open × (total / max_open) ≈ total` — bounded
/// by the configured total. (Integer division rounds the per-collection cap
/// down, so the sum is ≤ total, never above.) `HELIX_LSM_CACHE_COLLECTION_DIVISOR`
/// replaces `max_open` as the divisor when set (see [`ENV_CACHE_COLLECTION_DIVISOR`]).
/// Whatever the divisor, a collection's cap is no larger than the cache
/// volume's budget split over `max_open`
/// ([`cache_evictor::share_the_volume_holds`]).
fn cache_options_from_env(collection_id: &str) -> Option<ObjectStoreCacheOptions> {
    let dir = std::env::var(ENV_CACHE_DIR).ok()?;
    if dir.trim().is_empty() {
        return None;
    }
    let mut cache = ObjectStoreCacheOptions::default();
    // Per-collection cache root: scopes the SlateDB evictor's whole-root scan to
    // this collection so it never evicts another collection's cached files.
    cache.root_folder =
        Some(std::path::PathBuf::from(&dir).join(sanitize_cache_component(collection_id)));
    // SAFETY: object_store's default cap is usize::MAX (UNBOUNDED). An unbounded
    // FS cache fills the volume and crashes the pod (ENOSPC). NEVER leave it
    // unbounded: honor HELIX_LSM_CACHE_MAX_BYTES, else fall back to a conservative
    // cap (8 GiB) and warn. The effective cap is memory-clamped below because the
    // kernel charges hot cache files to the pod as page cache.
    const DEFAULT_CACHE_CAP_BYTES: usize = 8 * 1024 * 1024 * 1024;
    let configured_total_cap = std::env::var(ENV_CACHE_MAX_BYTES)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or_else(|| {
            eprintln!(
                "WARN {ENV_CACHE_MAX_BYTES} unset for cache dir {dir}; capping object-store \
                cache at {DEFAULT_CACHE_CAP_BYTES} bytes to avoid filling the volume — set it \
                below both memory and cache-volume budgets"
            );
            DEFAULT_CACHE_CAP_BYTES
        });
    let total_cap = memory_bounded_cache_cap(
        configured_total_cap,
        cgroup_memory_limit_bytes(),
        cache_memory_max_pct(),
    );
    // Divide the TOTAL cap across the expected concurrently-open collections so
    // the per-collection subdirs sum to <= total_cap (see CAP MATH above), or
    // across `HELIX_LSM_CACHE_COLLECTION_DIVISOR` when set.
    // Floored at 1 byte so a huge divisor never yields a zero (no-op) cap.
    let per_collection_cap = (total_cap / cache_collection_divisor()).max(1);
    // VOLUME BOUND. The divisor bounds one collection, the handle limit decides
    // how many are open at once, and the /cache budget evictor deletes closed
    // collections' directories only. So the share is no larger than what the
    // volume's budget holds with every handle open: no divisor, and no claim
    // made anew at a smaller size, can fill the volume with open directories.
    let per_collection_cap = match cache_evictor::share_the_volume_holds(
        std::path::Path::new(&dir),
        max_open_collections_for_cache(),
    ) {
        Some(held) if held < per_collection_cap => {
            warn_cache_share_cut_to_volume(&dir, per_collection_cap, held);
            held
        }
        _ => per_collection_cap,
    };
    cache.max_cache_size_bytes = Some(per_collection_cap);
    // Evictor sweep cadence. The cache also evicts on write + backpressures.
    // Unset/invalid → SlateDB default; explicit 0 disables recurring sweeps
    // after the initial cache scan.
    if let Some(secs) = std::env::var(ENV_CACHE_SCAN_SECS)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
    {
        cache.scan_interval = (secs > 0).then(|| std::time::Duration::from_secs(secs));
    }
    cache.max_open_file_handles = env_usize_opt(ENV_CACHE_MAX_OPEN_FILES)
        .filter(|h| *h > 0)
        .unwrap_or_else(cache_max_open_file_handles_default);
    // Part size must be a non-zero multiple of 1024 or SlateDB rejects the
    // options at open; invalid values fall back to the SlateDB default (4 MiB).
    if let Some(bytes) = env_usize_opt(ENV_CACHE_PART_BYTES).filter(|b| *b > 0 && b % 1024 == 0) {
        cache.part_size_bytes = bytes;
    }
    // Cache compacted-SST WRITES too (vendored-fork admission policy; the
    // fork default is false). Pre-fork slatedb cached all written parts, so
    // the writer served just-flushed SSTs off the SSD instead of refetching
    // them from S3 for compaction/serving — keep that behavior. WAL and
    // manifest traffic is never disk-cached under the fork's policy
    // regardless of this setting (a strict improvement: replay-once WAL parts
    // previously churned the evictor for no read benefit).
    cache.cache_puts = true;
    // Unset or blank keeps the default (on); anything unparsable turns it off.
    cache.ranged_miss = match std::env::var(ENV_CACHE_RANGED_MISS) {
        Ok(v) if !v.trim().is_empty() => env_bool_opt(ENV_CACHE_RANGED_MISS) == Some(true),
        _ => true,
    };
    // Unset or blank keeps the default; anything unparsable means one at a time.
    if let Ok(v) = std::env::var(ENV_CACHE_PART_FETCH_CONCURRENCY) {
        if !v.trim().is_empty() {
            cache.part_fetch_concurrency = v.trim().parse::<usize>().unwrap_or(1);
        }
    }
    Some(cache)
}

/// Build SlateDB [`Settings`] from the environment. Starts from
/// `Settings::default()` and overrides only the knobs that are explicitly set:
/// `object_store_cache_options` ([`ENV_CACHE_DIR`]), `flush_interval`
/// ([`ENV_FLUSH_INTERVAL_MS`]), `manifest_poll_interval`
/// ([`ENV_MANIFEST_POLL_MS`]), `l0_sst_size_bytes` ([`ENV_L0_SST_SIZE_BYTES`]),
/// `max_unflushed_bytes` ([`ENV_MAX_UNFLUSHED_BYTES`]), `min_filter_keys`
/// ([`ENV_MIN_FILTER_KEYS`]), `max_wal_flushes_before_l0_flush`
/// ([`ENV_MAX_WAL_FLUSHES_BEFORE_L0_FLUSH`]), `l0_max_ssts_per_key`
/// ([`ENV_L0_MAX_SSTS_PER_KEY`]), `l0_flush_parallelism`
/// ([`ENV_L0_FLUSH_PARALLELISM`]),
/// `compression_codec` ([`ENV_COMPRESSION_CODEC`]),
/// `compactor_options.max_concurrent_compactions`
/// ([`ENV_MAX_CONCURRENT_COMPACTIONS`]), `compactor_options.poll_interval`
/// ([`ENV_COMPACTOR_POLL_MS`]), `compactor_options.commit_compacted_interval`
/// ([`ENV_COMPACTOR_COMMIT_COMPACTED_MS`]),
/// `compactor_options.worker.compactions_poll_interval`
/// ([`ENV_COMPACTOR_WORKER_POLL_MS`]), `compactor_options.scheduler_options`
/// ([`ENV_COMPACTION_MIN_SOURCES`], [`ENV_COMPACTION_MAX_SOURCES`],
/// [`ENV_COMPACTION_SIZE_THRESHOLD`]), and `l0_max_ssts` ([`ENV_L0_MAX_SSTS`]).
/// Returns `None` when NO knob is configured so callers take the cache-less
/// bare-open path (keeps in-memory/test opens free of temp dirs and at SlateDB
/// defaults).
fn lsm_settings_from_env(collection_id: &str) -> Option<Settings> {
    let cache = cache_options_from_env(collection_id);
    let flush_ms = std::env::var(ENV_FLUSH_INTERVAL_MS)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok());
    let manifest_poll_interval = lsm_manifest_poll_interval_from_env();
    let l0_sst = env_usize_opt(ENV_L0_SST_SIZE_BYTES);
    let max_unflushed = env_usize_opt(ENV_MAX_UNFLUSHED_BYTES);
    let max_concurrent_compactions =
        env_usize_opt(ENV_MAX_CONCURRENT_COMPACTIONS).filter(|v| *v > 0);
    let compactor_poll_ms = env_duration_ms_opt(ENV_COMPACTOR_POLL_MS);
    let commit_compacted_ms = env_duration_ms_opt(ENV_COMPACTOR_COMMIT_COMPACTED_MS);
    let compactor_worker_poll_ms = env_duration_ms_opt(ENV_COMPACTOR_WORKER_POLL_MS);
    let l0_max_ssts = env_usize_opt(ENV_L0_MAX_SSTS).filter(|v| *v > 0);
    let l0_max_ssts_per_key = env_usize_opt(ENV_L0_MAX_SSTS_PER_KEY).filter(|v| *v > 0);
    // The scheduler's minimum is checked against the L0 limits this build ends with.
    let scheduler_options = {
        let defaults = Settings::default();
        compaction_scheduler_options_from_env(
            l0_max_ssts
                .unwrap_or(defaults.l0_max_ssts)
                .min(l0_max_ssts_per_key.unwrap_or(defaults.l0_max_ssts_per_key)),
        )
    };
    let max_wal_flushes_before_l0_flush =
        env_usize_opt(ENV_MAX_WAL_FLUSHES_BEFORE_L0_FLUSH).filter(|v| *v >= 4096);
    let l0_flush_parallelism = env_usize_opt(ENV_L0_FLUSH_PARALLELISM).filter(|v| *v > 0);
    let min_filter_keys = env_usize_opt(ENV_MIN_FILTER_KEYS).and_then(|v| u32::try_from(v).ok());
    let compression_codec = compression_codec_from_env();
    let gc_options = gc_options_from_env();
    if cache.is_none()
        && flush_ms.is_none()
        && manifest_poll_interval.is_none()
        && l0_sst.is_none()
        && max_unflushed.is_none()
        && max_concurrent_compactions.is_none()
        && compactor_poll_ms.is_none()
        && commit_compacted_ms.is_none()
        && compactor_worker_poll_ms.is_none()
        && scheduler_options.is_empty()
        && l0_max_ssts.is_none()
        && l0_max_ssts_per_key.is_none()
        && max_wal_flushes_before_l0_flush.is_none()
        && l0_flush_parallelism.is_none()
        && min_filter_keys.is_none()
        && compression_codec.is_none()
        && gc_options.is_none()
    {
        return None;
    }
    let mut settings = Settings::default();
    if let Some(cache) = cache {
        settings.object_store_cache_options = cache;
    }
    if let Some(ms) = flush_ms {
        settings.flush_interval = Some(std::time::Duration::from_millis(ms));
    }
    if let Some(interval) = manifest_poll_interval {
        settings.manifest_poll_interval = interval;
    }
    if let Some(bytes) = l0_sst {
        settings.l0_sst_size_bytes = bytes;
    }
    if let Some(bytes) = max_unflushed {
        settings.max_unflushed_bytes = bytes;
    }
    // `compactor_options` is `Option<CompactorOptions>`; the SlateDB default is
    // `Some(CompactorOptions::default())` (which itself sets
    // max_concurrent_compactions = 4). Mutate the existing options in place so all
    // other compactor defaults are preserved; fall back to a default-constructed
    // CompactorOptions if it were somehow None.
    if max_concurrent_compactions.is_some()
        || compactor_poll_ms.is_some()
        || commit_compacted_ms.is_some()
        || compactor_worker_poll_ms.is_some()
        || !scheduler_options.is_empty()
    {
        let mut compactor = settings.compactor_options.unwrap_or_default();
        if let Some(value) = max_concurrent_compactions {
            compactor.max_concurrent_compactions = value;
        }
        if let Some(interval) = compactor_poll_ms {
            compactor.poll_interval = interval;
        }
        if let Some(interval) = commit_compacted_ms {
            compactor.commit_compacted_interval = interval;
        }
        if let Some(interval) = compactor_worker_poll_ms {
            if let Some(worker) = compactor.worker.as_mut() {
                worker.compactions_poll_interval = interval;
            }
        }
        for (key, value) in scheduler_options {
            compactor.scheduler_options.insert(key.to_string(), value);
        }
        settings.compactor_options = Some(compactor);
    }
    // `l0_max_ssts` is a plain `usize` (not Option); assign directly.
    if let Some(value) = l0_max_ssts {
        settings.l0_max_ssts = value;
    }
    if let Some(value) = l0_max_ssts_per_key {
        settings.l0_max_ssts_per_key = value;
    }
    if let Some(value) = max_wal_flushes_before_l0_flush {
        settings.max_wal_flushes_before_l0_flush = value as u64;
    }
    if let Some(value) = l0_flush_parallelism {
        settings.l0_flush_parallelism = value;
    }
    if let Some(value) = min_filter_keys {
        settings.min_filter_keys = value;
    }
    if let Some(codec) = compression_codec {
        settings.compression_codec = Some(codec);
    }
    if let Some(options) = gc_options {
        settings.garbage_collector_options = Some(options);
    }
    Some(settings)
}

pub(crate) fn lsm_manifest_poll_interval_from_env() -> Option<std::time::Duration> {
    std::env::var(ENV_MANIFEST_POLL_MS)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
        .map(std::time::Duration::from_millis)
}

/// Reader-replica manifest poll cadence: the dedicated reader knob wins, then
/// the shared writer knob, then the SlateDB default. Split so raising the
/// writer poll for S3 cost cannot silently widen reader staleness.
pub(crate) fn lsm_reader_manifest_poll_interval_from_env() -> Option<std::time::Duration> {
    std::env::var(ENV_READER_MANIFEST_POLL_MS)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
        .map(std::time::Duration::from_millis)
        .or_else(lsm_manifest_poll_interval_from_env)
}

pub(crate) fn reader_options_from_env(collection_id: &str) -> DbReaderOptions {
    let mut options = DbReaderOptions::default();
    if let Some(cache) = cache_options_from_env(collection_id) {
        options.object_store_cache_options = cache;
    }
    if let Some(interval) = lsm_reader_manifest_poll_interval_from_env() {
        options.manifest_poll_interval = interval;
    }
    // Feed mode with the read-gated sweeper off: fresh opens take the IDLE
    // tier and only the feed poller promotes changed collections.
    if !reader_cold_open_starts_fast() {
        options.manifest_poll_interval = reader_idle_poll_interval_from_env();
    }
    options.wal_replay_sst_batch_size = reader_wal_replay_batch_from_env();
    clamp_reader_checkpoint_lifetime(&mut options);
    options
}

fn reader_wal_replay_batch_from_env() -> usize {
    env_usize_opt(ENV_READER_WAL_REPLAY_BATCH)
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_READER_WAL_REPLAY_BATCH)
        .min(MAX_READER_WAL_REPLAY_BATCH)
}

/// [`ENV_READER_WAL_WHOLE_GET`]: on unless set to a false value.
pub(crate) fn reader_wal_whole_get_enabled() -> bool {
    env_bool_opt(ENV_READER_WAL_WHOLE_GET).unwrap_or(true)
}

/// [`ENV_READER_TIER_IN_PLACE`]: off unless set to a true value. A reader
/// that is not reopened keeps one replay table for every poll that found WAL
/// objects, until the writer's next L0 flush, and every read looks at all of
/// them; it stays off until a reader on S3 has shown what that costs a
/// collection that is written and read at once.
pub(crate) fn reader_tier_in_place_enabled() -> bool {
    env_bool_opt(ENV_READER_TIER_IN_PLACE).unwrap_or(false)
}

/// True if a freshly cold-opened reader-replica `DbReader` starts on the fast
/// (serving) poll tier. `HelixGraphStorage` seeds its `reader_poll_tier_fast`
/// belief from this at construction, so it starts in sync with the
/// `DbReader`'s actual poll interval ([`reader_options_from_env`] shares this
/// exact check). Cold opens start fast whenever the read-gated sweeper can
/// demote them later: a write that lands while the open is still in flight is
/// skipped by the feed poller (the collection is not resident yet), so an
/// idle-tier open would serve that stale view for the whole idle interval.
/// Only the sweeper kill switch (`HELIX_LSM_READER_IDLE_AFTER_MS=0`) with the
/// write-feed configured keeps the idle-tier cold open, since nothing would
/// ever demote a fast one there.
pub(crate) fn reader_cold_open_starts_fast() -> bool {
    writer_feed_url_from_env().is_none() || !reader_idle_after_from_env().is_zero()
}

/// Whether the standalone read-gated poll-tier demotion sweep
/// (`collection_manager::spawn_reader_poll_tier_sweeper`) should run at all:
/// only `idle_after` being non-zero (its own kill switch) matters now. The
/// sweeper used to stand down entirely whenever the write-feed was
/// configured, on the theory that the feed poller's own demotion already
/// covered that mode — but the feed poller only demotes entries in ITS OWN
/// active set (collections it promoted for a WRITE reason), so a collection
/// this sweep's own read-gated promotion brought to the fast tier — then
/// abandoned once reads stopped — never demoted in feed mode, defeating the
/// S3-cost goal. The sweeper now runs in both modes; it coordinates with the
/// feed poller per-collection via `HelixGraphStorage::write_promoted_recently`
/// instead (never demoting a collection the feed poller just promoted for
/// write reasons), rather than by standing down globally.
pub(crate) fn reader_poll_tier_sweeper_enabled(idle_after: std::time::Duration) -> bool {
    !idle_after.is_zero()
}

/// Keeps a reader's checkpoint lease at three manifest poll intervals or more.
///
/// A `DbReader` renews its lease at the first poll past the lease's half-life,
/// and a manifest stores the expiry rounded down to the second. At twice the
/// interval, the least SlateDB accepts, the poll one interval in comes a
/// manifest load short of half-life and the renewal waits for the next, which
/// lands on the expiry itself: up to a second after the stored one, where a GC
/// pass strips the lease from under the open reader. At three intervals the
/// renewal two intervals in has a whole interval in hand. The default lifetime
/// is 600 s, so this raises it from a 200 s poll up: the idle tier's 300 s poll
/// gets 900 s, and a reader that dies there holds its files up to 15 minutes.
///
/// A reader that changes tiers in place ([`ENV_READER_TIER_IN_PLACE`]) keeps
/// the lease it was opened with, and its polls are not in step with it: it
/// can be demoted right before the renewal at half-life, and the poll that
/// renews then comes an idle interval later. Four idle intervals leave a
/// whole one in hand there as well, so every such reader is opened with
/// that: 1,200 s at the 300 s idle poll, for a `DbReader` opened on the 5 s
/// poll too, and a reader that dies holds its files up to 20 minutes.
pub(crate) fn clamp_reader_checkpoint_lifetime(options: &mut DbReaderOptions) {
    let mut min_lifetime = options.manifest_poll_interval.saturating_mul(3);
    if reader_tier_in_place_enabled() {
        min_lifetime = min_lifetime.max(reader_idle_poll_interval_from_env().saturating_mul(4));
    }
    if options.checkpoint_lifetime < min_lifetime {
        options.checkpoint_lifetime = min_lifetime;
    }
}

/// Writer change-feed base URL, when configured on this reader replica.
pub(crate) fn writer_feed_url_from_env() -> Option<String> {
    std::env::var(ENV_WRITER_FEED_URL)
        .ok()
        .map(|url| url.trim().trim_end_matches('/').to_string())
        .filter(|url| !url.is_empty())
}

pub(crate) fn reader_feed_poll_interval_from_env() -> std::time::Duration {
    std::time::Duration::from_millis(
        std::env::var(ENV_READER_FEED_POLL_MS)
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(1_000),
    )
}

pub(crate) fn reader_idle_poll_interval_from_env() -> std::time::Duration {
    std::time::Duration::from_millis(
        std::env::var(ENV_READER_IDLE_POLL_MS)
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(300_000),
    )
}

/// Active-tier `DbReader` poll: the serving freshness cadence
/// ([`ENV_READER_MANIFEST_POLL_MS`], 5s fallback).
pub(crate) fn reader_active_poll_interval_from_env() -> std::time::Duration {
    lsm_reader_manifest_poll_interval_from_env()
        .unwrap_or_else(|| std::time::Duration::from_secs(5))
}

/// Read-gated poll-tier idle threshold ([`ENV_READER_READ_IDLE_AFTER_MS`]).
/// Unlike the other reader knobs, `0` is a valid, meaningful value (the kill
/// switch) rather than "unset" — only a parse failure or missing var falls
/// back to the default.
pub(crate) fn reader_idle_after_from_env() -> std::time::Duration {
    std::time::Duration::from_millis(
        std::env::var(ENV_READER_READ_IDLE_AFTER_MS)
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(600_000),
    )
}

/// Poll-tier promotion reopen budget ([`ENV_READER_PROMOTE_BUDGET_MS`],
/// 8s fallback, 1s floor).
pub(crate) fn reader_promote_budget_from_env() -> std::time::Duration {
    std::time::Duration::from_millis(
        std::env::var(ENV_READER_PROMOTE_BUDGET_MS)
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(8_000)
            .max(1_000),
    )
}

pub(crate) fn reader_active_ttl_from_env() -> std::time::Duration {
    std::time::Duration::from_secs(
        std::env::var(ENV_READER_ACTIVE_TTL_SECS)
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(600),
    )
}

/// Env knob: worker-thread count for the process-shared LSM tokio runtime.
/// Default = `available_parallelism` (min 1). Tune down on small nodes, up on
/// ingest-heavy ones.
const ENV_RUNTIME_THREADS: &str = "HELIX_LSM_RUNTIME_THREADS";

/// Process-global multi-thread tokio runtime shared by EVERY `LsmBackend` and
/// `LsmReader`.
///
/// SlateDB is async; Helion's call sites are sync, so each backend bridges via
/// `block_on`. Giving every collection its own `Runtime::new()` would spawn one
/// worker pool PER collection — at tens of thousands of collections that is tens
/// of thousands of runtimes × N worker threads (thread/FD explosion, scheduler
/// thrash). One shared runtime keeps the thread count bounded by node size; each
/// backend stores only a cheap `Handle` clone. Concurrent `block_on` calls from
/// different blocking threads are fine on a multi-thread runtime — each drives
/// its own future while SlateDB's background flush/compaction tasks share the
/// pool.
pub(crate) fn shared_lsm_runtime() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        let default_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        let threads = std::env::var(ENV_RUNTIME_THREADS)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|n| *n >= 1)
            .unwrap_or(default_threads);
        let rt = Builder::new_multi_thread()
            .worker_threads(threads)
            .thread_name("helix-lsm")
            .enable_all()
            .build()
            .unwrap_or_else(|error| {
                eprintln!("failed to build shared LSM tokio runtime: {error}");
                std::process::abort();
            });
        // Start the cache evictor on the shared runtime. Runs once at startup
        // (background) and then periodically to bound /cache disk usage.
        cache_evictor::start_cache_evictor(&rt.handle(), cache_root_from_env());
        rt
    })
}

/// A `Handle` to the process-shared LSM runtime (see [`shared_lsm_runtime`]).
/// `Handle` is cheap to clone and is what each backend stores in place of an
/// owned `Runtime`.
pub(crate) fn shared_lsm_handle() -> Handle {
    shared_lsm_runtime().handle().clone()
}

pub(crate) fn block_on_lsm<T, E>(
    rt: &Handle,
    future: impl Future<Output = Result<T, E>>,
) -> Result<T, BackendError>
where
    E: std::fmt::Display,
{
    block_on_lsm_mapped(rt, future, io)
}

/// Read-only variant of [`block_on_lsm`] that cooperatively abandons a pending
/// SlateDB operation when the serving HTTP request is cancelled. Dropping a
/// get/list/scan/snapshot future is safe and releases its in-flight object-store
/// work. Database opens and writer commit/flush/persist/purge paths must keep
/// using [`block_on_lsm`] or [`block_on_lsm_mapped`].
pub(crate) fn block_on_lsm_read<T, E>(
    rt: &Handle,
    future: impl Future<Output = Result<T, E>>,
) -> Result<T, BackendError>
where
    E: std::fmt::Display,
{
    block_on_lsm_read_mapped(rt, future, io)
}

fn lsm_read_cancelled_error() -> BackendError {
    // Retain the deployed metric name for dashboard continuity; it now covers
    // writer-local reads too.
    metrics::counter!("helix_lsm_reader_request_cancelled_total").increment(1);
    BackendError::Cancelled
}

/// Cheap cancellation checkpoint for operations that must finish atomically
/// once started (notably SlateDB writer/reader opens). Callers check before the
/// uncancellable future begins and again after a successful build, preventing
/// queued abandoned requests from opening a database without dropping the
/// side-effecting build future midway through.
pub(crate) fn ensure_lsm_read_not_cancelled() -> Result<(), BackendError> {
    if lsm_read_cancelled() {
        Err(lsm_read_cancelled_error())
    } else {
        Ok(())
    }
}

/// Whether the request this thread serves has been cancelled. Counts nothing,
/// unlike [`ensure_lsm_read_not_cancelled`]: for a loop that turned a failed
/// read into "no match" and must tell that from a caller that has gone.
pub(crate) fn lsm_read_cancelled() -> bool {
    LSM_READ_CANCELLATION.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(LsmReadCancellation::is_cancelled_inner)
    })
}

/// Like [`block_on_lsm_read`], but preserves a caller-provided backend error
/// mapping. This mirrors [`block_on_lsm_mapped`] for read operations whose async
/// body already returns [`BackendError`].
pub(crate) fn block_on_lsm_read_mapped<T, E>(
    rt: &Handle,
    future: impl Future<Output = Result<T, E>>,
    map_err: impl FnOnce(E) -> BackendError,
) -> Result<T, BackendError> {
    let cancellation = LSM_READ_CANCELLATION.with(|slot| slot.borrow().clone());
    block_on_lsm_mapped(
        rt,
        async move {
            match cancellation {
                Some(cancellation) => {
                    tokio::select! {
                        biased;
                        _ = cancellation.cancelled() => {
                            Err(lsm_read_cancelled_error())
                        }
                        result = future => result.map_err(map_err),
                    }
                }
                None => future.await.map_err(map_err),
            }
        },
        |error| error,
    )
}

/// Like [`block_on_lsm`] but maps a backend failure through `map_err` instead of
/// the default [`io`]. Write/commit/flush paths pass [`map_slatedb_err`] so a
/// fencing/CAS failure surfaces as [`BackendError::Conflict`] rather than a
/// generic [`BackendError::Io`]; read paths keep using [`block_on_lsm`] (reads
/// don't fence-conflict). The runtime-guard check is shared so both behave
/// identically when misused from inside an async runtime.
pub(crate) fn block_on_lsm_mapped<T, E>(
    rt: &Handle,
    future: impl Future<Output = Result<T, E>>,
    map_err: impl FnOnce(E) -> BackendError,
) -> Result<T, BackendError> {
    #[cfg(test)]
    LSM_ROUND_TRIPS_FOR_TEST.with(|count| count.set(count.get() + 1));
    let allowed = LSM_BLOCKING_ALLOWED.with(Cell::get);
    if tokio::runtime::Handle::try_current().is_ok() {
        if !allowed {
            return Err(BackendError::Unsupported(
                "LSM backend operations must run from a blocking thread, not inside a Tokio runtime"
                    .to_string(),
            ));
        }
        return tokio::task::block_in_place(|| rt.block_on(future)).map_err(map_err);
    }
    rt.block_on(future).map_err(map_err)
}

/// Never resolves: every `every`, runs one `refresh` to completion.
async fn refresh_every<F: Future<Output = ()>>(
    every: std::time::Duration,
    mut refresh: impl FnMut() -> F,
) -> std::convert::Infallible {
    loop {
        tokio::time::sleep(every).await;
        refresh().await;
    }
}

/// Resolves with `op`. While `op` stays pending, `refresh` runs every `every`
/// beside it, never ahead of it: an `op` that completes first is never
/// refreshed, and a refresh that hangs does not hold up `op`.
async fn refresh_while_pending<T, F: Future<Output = ()>>(
    op: impl Future<Output = T>,
    every: std::time::Duration,
    refresh: impl FnMut() -> F,
) -> T {
    let mut op = std::pin::pin!(op);
    let mut refresher = std::pin::pin!(refresh_every(every, refresh));
    std::future::poll_fn(|cx| {
        let out = op.as_mut().poll(cx);
        if out.is_pending() {
            // Never ready; polling it keeps its timer and refresh running.
            let _ = refresher.as_mut().poll(cx);
        }
        out
    })
    .await
}

/// The L0 SST count at which SlateDB can refuse to flush a frozen memtable:
/// the lower of its two L0 limits.
fn l0_stall_floor(settings: &Settings) -> usize {
    settings.l0_max_ssts.min(settings.l0_max_ssts_per_key)
}

/// Lets one stalled-write manifest refresh run per collection at a time, and
/// start at most one per interval, however many commits are parked.
#[derive(Default)]
struct RefreshGate(tokio::sync::Mutex<Option<std::time::Instant>>);

impl RefreshGate {
    /// The permit for one refresh, held until dropped. `None` while another
    /// holds it, or within `every` of the last one's start.
    fn try_enter(
        &self,
        every: std::time::Duration,
    ) -> Option<tokio::sync::MutexGuard<'_, Option<std::time::Instant>>> {
        let mut last_started = self.0.try_lock().ok()?;
        if last_started.is_some_and(|at| at.elapsed() < every) {
            return None;
        }
        *last_started = Some(std::time::Instant::now());
        Some(last_started)
    }
}

/// Namespace key prefix: `<ns_name>\0`.
pub(crate) fn ns_prefix(ns: Namespace<'_>) -> Vec<u8> {
    let mut p = ns_name(ns).into_bytes();
    p.push(0u8);
    p
}

/// Full SlateDB key: namespace prefix ++ user key.
pub(crate) fn prefixed(ns: Namespace<'_>, key: &[u8]) -> Vec<u8> {
    let mut k = ns_prefix(ns);
    k.extend_from_slice(key);
    k
}

/// Key prefix for all values stored under `key` in a multi-value namespace:
/// `<ns_prefix><key_len: u32-BE><key>`. The length prefix prevents keys of
/// different lengths from colliding once the value is appended.
pub(crate) fn dup_key_prefix(ns: Namespace<'_>, key: &[u8]) -> Vec<u8> {
    let mut k = ns_prefix(ns);
    k.extend_from_slice(&(key.len() as u32).to_be_bytes());
    k.extend_from_slice(key);
    k
}

/// Composite SlateDB key encoding one `(key, value)` of a multi-value namespace.
pub(crate) fn dup_composite(ns: Namespace<'_>, key: &[u8], value: &[u8]) -> Vec<u8> {
    let mut k = dup_key_prefix(ns, key);
    k.extend_from_slice(value);
    k
}

/// Decode a dup-namespace composite SlateDB key (`<ns_prefix><key_len: u32-BE>
/// <key><value>`) into its `(logical_key, logical_value)` byte slices, given the
/// namespace prefix it was scanned under. `Ok(None)` when the row is shorter than
/// the length header (skip it); `Err(Corruption)` when the declared key length
/// overruns the row. Shared by the writer's `LsmBackend` dup scan and the reader
/// replica so both decode the writer's dup rows identically.
pub(crate) fn decode_dup_full_key<'k>(
    full_key: &'k [u8],
    prefix: &[u8],
) -> Result<Option<(&'k [u8], &'k [u8])>, BackendError> {
    if full_key.len() < prefix.len() + 4 {
        return Ok(None);
    }
    let len_start = prefix.len();
    let key_len = u32::from_be_bytes(
        full_key[len_start..len_start + 4]
            .try_into()
            .map_err(|_| BackendError::Corruption("invalid dup key length".into()))?,
    ) as usize;
    let key_start = len_start + 4;
    let value_start = key_start + key_len;
    if full_key.len() < value_start {
        return Err(BackendError::Corruption(
            "truncated duplicate namespace key".into(),
        ));
    }
    Ok(Some((
        &full_key[key_start..value_start],
        &full_key[value_start..],
    )))
}

pub(crate) fn logical_key_in_range(key: &[u8], range: &KeyRange) -> bool {
    let after_start = match &range.start {
        Bound::Included(start) => key >= start.as_slice(),
        Bound::Excluded(start) => key > start.as_slice(),
        Bound::Unbounded => true,
    };
    let before_end = match &range.end {
        Bound::Included(end) => key <= end.as_slice(),
        Bound::Excluded(end) => key < end.as_slice(),
        Bound::Unbounded => true,
    };
    after_start && before_end
}

/// Composite-key bounds for one dup-namespace scan over a logical `range`.
pub(crate) struct DupScanPlan {
    pub lo: Bound<Vec<u8>>,
    pub hi: Bound<Vec<u8>>,
    /// Whole-namespace scans go through SlateDB's prefix scan.
    pub scan_prefix: Option<Vec<u8>>,
    /// Set when `range` is `KeyRange::prefix(p)`: lets [`DupScanCursor`] seek
    /// past rows that cannot match instead of visiting the namespace.
    pub seek_prefix: Option<Vec<u8>>,
}

/// Dup rows are keyed `<ns><key_len: u32-BE><key><value>`, so logical keys of
/// one length form a contiguous region and an exact key is a single slice of
/// its region. A logical prefix `p` matches the slice `<ns><len><p>..` of every
/// region with `len >= p.len()`; the scan starts at the first such slice and
/// the cursor seeks between them. Any other range still scans the namespace
/// and filters per row.
pub(crate) fn dup_scan_plan(ns: Namespace<'_>, range: &KeyRange) -> DupScanPlan {
    let prefix = ns_prefix(ns);
    let excluded_end = |start: &[u8]| match next_prefix(start) {
        Some(end) => Bound::Excluded(end),
        None => Bound::Unbounded,
    };
    let scan_prefix = range_covers_all_logical_keys(range).then(|| prefix.clone());
    match (&range.start, &range.end) {
        // Keep exact lookups on range scans to avoid extra Bloom-filter reads.
        (Bound::Included(start), Bound::Included(end)) if start == end => {
            let lo = dup_key_prefix(ns, start);
            DupScanPlan {
                hi: excluded_end(&lo),
                lo: Bound::Included(lo),
                scan_prefix,
                seek_prefix: None,
            }
        }
        (Bound::Included(start), Bound::Excluded(end))
            if next_prefix(start).as_deref() == Some(end.as_slice()) =>
        {
            DupScanPlan {
                lo: Bound::Included(dup_key_prefix(ns, start)),
                hi: excluded_end(&prefix),
                scan_prefix,
                seek_prefix: Some(start.clone()),
            }
        }
        _ => DupScanPlan {
            hi: excluded_end(&prefix),
            lo: Bound::Included(prefix),
            scan_prefix,
            seek_prefix: None,
        },
    }
}

/// Drives a dup-namespace iterator, yielding only rows whose logical key lies
/// in `range`. With a `seek_prefix` the cursor costs O(matching rows + distinct
/// key lengths) instead of O(namespace): `drop_node_be` scans a node's edges by
/// prefix once per deleted point, which used to read every edge row in the
/// collection per point.
pub(crate) struct DupScanCursor<'a> {
    iter: DbIterator,
    ns_prefix: &'a [u8],
    range: &'a KeyRange,
    seek_prefix: Option<&'a [u8]>,
    map_err: fn(slatedb::Error) -> BackendError,
    visited: usize,
}

impl<'a> DupScanCursor<'a> {
    pub(crate) fn new(
        iter: DbIterator,
        ns_prefix: &'a [u8],
        range: &'a KeyRange,
        seek_prefix: Option<&'a [u8]>,
        map_err: fn(slatedb::Error) -> BackendError,
    ) -> Self {
        Self {
            iter,
            ns_prefix,
            range,
            seek_prefix,
            map_err,
            visited: 0,
        }
    }

    /// Rows pulled from the backend so far, matching or not.
    pub(crate) fn visited(&self) -> usize {
        self.visited
    }

    pub(crate) async fn next(&mut self) -> Result<Option<KeyValue>, BackendError> {
        while let Some(kv) = self.iter.next().await.map_err(self.map_err)? {
            self.visited += 1;
            let Some((logical_key, _)) = decode_dup_full_key(kv.key.as_ref(), self.ns_prefix)?
            else {
                continue;
            };
            if logical_key_in_range(logical_key, self.range) {
                return Ok(Some(kv));
            }
            let Some(prefix) = self.seek_prefix else {
                continue;
            };
            // Rows are sorted, so nothing else in this key-length region can
            // match: jump to the prefix within it if we landed before the
            // prefix, otherwise to the prefix in the next region. The scan
            // began at `<ns><p.len()><p>`, so every row seen here is at least
            // `p.len()` long and the target is strictly past the current row.
            let len = logical_key.len() as u32;
            let region = if logical_key < prefix {
                Some(len)
            } else {
                len.checked_add(1)
            };
            let Some(region) = region else {
                return Ok(None);
            };
            let mut target = self.ns_prefix.to_vec();
            target.extend_from_slice(&region.to_be_bytes());
            target.extend_from_slice(prefix);
            self.iter.seek(target).await.map_err(self.map_err)?;
        }
        Ok(None)
    }
}

/// The logical key lengths of at least `min_len` that have rows in a dup
/// namespace, ascending, with the number of rows read to find them. Each
/// length is one contiguous region of the namespace, so its first row is one
/// seek away; the probe reads a block at a time because it wants one row.
async fn dup_key_lengths(
    snapshot: &DbSnapshot,
    ns_prefix: &[u8],
    min_len: usize,
) -> Result<(Vec<u32>, usize), BackendError> {
    let Ok(min_len) = u32::try_from(min_len) else {
        return Ok((Vec::new(), 0));
    };
    let lo = [ns_prefix, &min_len.to_be_bytes()[..]].concat();
    let hi = match next_prefix(ns_prefix) {
        Some(end) => Bound::Excluded(end),
        None => Bound::Unbounded,
    };
    let probe = lsm_scan_options()
        .with_read_ahead_bytes(1)
        .with_max_fetch_tasks(1);
    let mut iter = snapshot
        .scan_with_options((Bound::Included(lo), hi), &probe)
        .await
        .map_err(io)?;
    let (mut lengths, mut visited) = (Vec::new(), 0);
    while let Some(kv) = iter.next().await.map_err(io)? {
        visited += 1;
        let Some((key, _)) = decode_dup_full_key(kv.key.as_ref(), ns_prefix)? else {
            continue;
        };
        let len = key.len() as u32;
        lengths.push(len);
        let Some(next) = len.checked_add(1) else {
            break;
        };
        iter.seek([ns_prefix, &next.to_be_bytes()[..]].concat())
            .await
            .map_err(io)?;
    }
    Ok((lengths, visited))
}

/// The collections root prefix for a SlateDB collection path: everything up to
/// (not including) the final `/<name>` segment (`"helion/my_collection"` ->
/// `"helion"`, `"my_collection"` -> `""`). Each collection's SlateDB lives at
/// `<root>/<name>`, so the root is the prefix whose immediate children are the
/// collection names.
pub(crate) fn collections_root_prefix(collection_path: &str) -> &str {
    match collection_path.rfind('/') {
        Some(idx) => &collection_path[..idx],
        None => "",
    }
}

const LSM_DROP_TOMBSTONE_DIR: &str = "_helix_drop_tombstones";
const LSM_DROP_TOMBSTONE_BODY: &[u8] = b"pending\n";

pub(crate) fn drop_tombstone_object_path(collection_path: &str) -> ObjPath {
    let root = collections_root_prefix(collection_path).trim_matches('/');
    let collection = collection_path
        .trim_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or(collection_path);
    if root.is_empty() {
        ObjPath::from(format!("{}/{}", LSM_DROP_TOMBSTONE_DIR, collection))
    } else {
        ObjPath::from(format!(
            "{}/{}/{}",
            root, LSM_DROP_TOMBSTONE_DIR, collection
        ))
    }
}

fn tombstone_collection_name(path: &ObjPath) -> Option<String> {
    let raw = path.as_ref();
    let (parent, name) = raw.rsplit_once('/')?;
    (parent.rsplit('/').next() == Some(LSM_DROP_TOMBSTONE_DIR)).then(|| name.to_string())
}

pub(crate) fn list_drop_tombstones(
    rt: &Handle,
    store: &Arc<dyn ObjectStore>,
    root_prefix: &str,
) -> Result<HashSet<String>, BackendError> {
    let tombstone_prefix = if root_prefix.trim_matches('/').is_empty() {
        ObjPath::from(LSM_DROP_TOMBSTONE_DIR)
    } else {
        ObjPath::from(format!(
            "{}/{}",
            root_prefix.trim_matches('/'),
            LSM_DROP_TOMBSTONE_DIR
        ))
    };
    block_on_lsm_read(rt, async move {
        store
            .list(Some(&tombstone_prefix))
            .map_ok(|meta| meta.location)
            .try_filter_map(|path| async move { Ok(tombstone_collection_name(&path)) })
            .try_collect::<HashSet<String>>()
            .await
            .map_err(io)
    })
}

/// List the immediate child collection names under `root_prefix` in `store` via
/// one `list_with_delimiter` (the object-store "directory listing"): each
/// returned common-prefix is `<root>/<name>`, and its last path segment is the
/// collection `name`. Shared by the writer/reader instance methods and the
/// env-based catalog-rediscovery helper so all callers parse names identically.
pub(crate) fn list_child_collection_names(
    rt: &Handle,
    store: &Arc<dyn ObjectStore>,
    root_prefix: &str,
) -> Result<Vec<String>, BackendError> {
    list_child_collection_catalog(rt, store, root_prefix).map(|catalog| catalog.names)
}

#[derive(Default)]
pub(crate) struct CollectionCatalog {
    pub names: Vec<String>,
    pub tombstoned: HashSet<String>,
}

pub(crate) fn list_child_collection_catalog(
    rt: &Handle,
    store: &Arc<dyn ObjectStore>,
    root_prefix: &str,
) -> Result<CollectionCatalog, BackendError> {
    let root = ObjPath::from(root_prefix);
    let result = block_on_lsm_read(rt, store.list_with_delimiter(Some(&root)))?;
    let tombstoned = list_drop_tombstones(rt, store, root_prefix)?;
    let mut names: Vec<String> = result
        .common_prefixes
        .iter()
        .filter_map(|p| p.filename().map(|name| name.to_string()))
        .filter(|name| name != LSM_DROP_TOMBSTONE_DIR && !tombstoned.contains(name))
        .collect();
    names.sort();
    names.dedup();
    Ok(CollectionCatalog { names, tombstoned })
}

/// Open an S3 (or S3-compatible) object store from explicit connection parts and
/// list the collection names under `root_prefix`. The read-only twin of
/// [`LsmBackend::open_s3_with_options`] for catalog rediscovery: it builds the
/// store the same way but opens NO `Db`, so a writer that started with an empty
/// local registry (fresh node / PVC loss) can discover collections that exist in
/// the object store. `endpoint`/`allow_http` target MinIO and other
/// S3-compatible stores; `region` overrides the environment when provided.
pub fn list_s3_collection_prefixes(
    bucket: &str,
    root_prefix: &str,
    region: Option<&str>,
    endpoint: Option<&str>,
    allow_http: bool,
) -> Result<Vec<String>, BackendError> {
    let config = S3StoreConfig {
        bucket,
        region,
        endpoint,
        allow_http,
    };
    list_s3_collection_catalog(&config, root_prefix).map(|catalog| catalog.names)
}

pub(crate) fn list_s3_collection_catalog(
    config: &S3StoreConfig<'_>,
    root_prefix: &str,
) -> Result<CollectionCatalog, BackendError> {
    let store = build_s3_store(config)?;
    list_child_collection_catalog(&shared_lsm_handle(), &store, root_prefix)
}

/// Collection names with a drop tombstone under `root_prefix`; the S3 twin of
/// [`list_drop_tombstones`] for a writer that has no `Db` open yet (startup).
pub(crate) fn list_s3_drop_tombstones(
    config: &S3StoreConfig<'_>,
    root_prefix: &str,
) -> Result<HashSet<String>, BackendError> {
    let store = build_s3_store(config)?;
    list_drop_tombstones(&shared_lsm_handle(), &store, root_prefix)
}

pub(crate) fn read_s3_metadata_sidecar(
    config: &S3StoreConfig<'_>,
    collection_prefix: &str,
) -> Result<Option<Vec<u8>>, BackendError> {
    let store = build_s3_store(config)?;
    let path = metadata_sidecar_object_path(collection_prefix);
    block_on_lsm_read(&shared_lsm_handle(), async move {
        match store.get(&path).await {
            Ok(result) => result
                .bytes()
                .await
                .map(|bytes| Some(bytes.to_vec()))
                .map_err(io),
            Err(ObjectStoreError::NotFound { .. }) => Ok(None),
            Err(err) => Err(io(err)),
        }
    })
}

pub(crate) fn persist_s3_drop_tombstone(
    config: &S3StoreConfig<'_>,
    collection_prefix: &str,
) -> Result<(), BackendError> {
    let store = build_s3_store(config)?;
    persist_drop_tombstone_with_store(store, collection_prefix)
}

pub(crate) fn s3_collection_prefix_exists(
    config: &S3StoreConfig<'_>,
    collection_prefix: &str,
) -> Result<bool, BackendError> {
    let store = build_s3_store(config)?;
    collection_prefix_exists_with_store(store, collection_prefix)
}

pub(crate) fn collection_prefix_exists_with_store(
    store: Arc<dyn ObjectStore>,
    collection_prefix: &str,
) -> Result<bool, BackendError> {
    #[cfg(test)]
    if std::env::var("HELIX_LSM_TEST_FAIL_PREFIX_PROBE").is_ok() {
        return Err(BackendError::Io(
            "injected test object-store prefix probe failure".to_string(),
        ));
    }
    let prefix = ObjPath::from(collection_prefix);
    block_on_lsm_read(&shared_lsm_handle(), async move {
        store
            .list(Some(&prefix))
            .try_next()
            .await
            .map(|object| object.is_some())
            .map_err(io)
    })
}

pub(crate) fn s3_collection_prefix_bytes(
    config: &S3StoreConfig<'_>,
    collection_prefix: &str,
) -> Result<u64, BackendError> {
    let store = build_s3_store(config)?;
    collection_prefix_bytes_with_store(store, collection_prefix)
}

/// Total object bytes under a collection prefix, from one LIST pass, without
/// opening the collection.
pub(crate) fn collection_prefix_bytes_with_store(
    store: Arc<dyn ObjectStore>,
    collection_prefix: &str,
) -> Result<u64, BackendError> {
    let prefix = ObjPath::from(collection_prefix);
    block_on_lsm_read(&shared_lsm_handle(), async move {
        let mut entries = store.list(Some(&prefix));
        let mut total = 0_u64;
        while let Some(entry) = entries.try_next().await.map_err(io)? {
            total = total.saturating_add(entry.size as u64);
        }
        Ok::<u64, BackendError>(total)
    })
}

pub(crate) fn read_s3_drop_tombstone(
    config: &S3StoreConfig<'_>,
    collection_prefix: &str,
) -> Result<bool, BackendError> {
    let store = build_s3_store(config)?;
    read_drop_tombstone_with_store(store, collection_prefix)
}

pub(crate) fn clear_s3_drop_tombstone(
    config: &S3StoreConfig<'_>,
    collection_prefix: &str,
) -> Result<(), BackendError> {
    let store = build_s3_store(config)?;
    clear_drop_tombstone_with_store(store, collection_prefix)
}

pub(crate) fn persist_drop_tombstone_with_store(
    store: Arc<dyn ObjectStore>,
    collection_prefix: &str,
) -> Result<(), BackendError> {
    #[cfg(test)]
    if std::env::var("HELIX_LSM_TEST_FAIL_TOMBSTONE_WRITE").is_ok() {
        return Err(BackendError::Io(
            "injected test object-store drop tombstone write failure".to_string(),
        ));
    }
    let path = drop_tombstone_object_path(collection_prefix);
    block_on_lsm(&shared_lsm_handle(), async move {
        store
            .put(&path, LSM_DROP_TOMBSTONE_BODY.to_vec().into())
            .await
            .map_err(io)?;
        Ok::<(), BackendError>(())
    })
}

pub(crate) fn read_drop_tombstone_with_store(
    store: Arc<dyn ObjectStore>,
    collection_prefix: &str,
) -> Result<bool, BackendError> {
    #[cfg(test)]
    if std::env::var("HELIX_LSM_TEST_FAIL_TOMBSTONE_READ").is_ok() {
        return Err(BackendError::Io(
            "injected test object-store drop tombstone read failure".to_string(),
        ));
    }
    let path = drop_tombstone_object_path(collection_prefix);
    block_on_lsm_read(&shared_lsm_handle(), async move {
        match store.head(&path).await {
            Ok(_) => Ok(true),
            Err(ObjectStoreError::NotFound { .. }) => Ok(false),
            Err(err) => Err(io(err)),
        }
    })
}

pub(crate) fn clear_drop_tombstone_with_store(
    store: Arc<dyn ObjectStore>,
    collection_prefix: &str,
) -> Result<(), BackendError> {
    let path = drop_tombstone_object_path(collection_prefix);
    block_on_lsm(&shared_lsm_handle(), async move {
        match store.delete(&path).await {
            Ok(()) | Err(ObjectStoreError::NotFound { .. }) => Ok(()),
            Err(err) => Err(io(err)),
        }
    })
}

/// SlateDB-backed storage engine.
/// Env knob: write-quiet seconds after which a resident WRITER flushes its
/// memtable to L0. Every reader cold open and checkpoint refresh replays each
/// WAL SST written since the last L0 flush, and SlateDB flushes only at
/// `l0_sst_size_bytes` or 4096 WAL flushes (or on close), so trickle-written
/// collections that stay resident carry tails of hundreds to thousands of WAL
/// SSTs. Unset/0 disables. Cost: at most one L0 SST + manifest update per
/// write-quiet period per resident writer collection; a clean memtable
/// flushes nothing.
const ENV_IDLE_FLUSH_SECS: &str = "HELIX_LSM_IDLE_FLUSH_SECS";

fn idle_flush_after_from_env() -> Option<std::time::Duration> {
    env_usize_opt(ENV_IDLE_FLUSH_SECS)
        .filter(|secs| *secs > 0)
        .map(|secs| std::time::Duration::from_secs(secs as u64))
}

/// Env knob: seconds a committed write may stay unflushed on a resident WRITER
/// whose idle flush is on. A steady trickle (a write every few seconds for
/// hours) never goes write-quiet, so the quiet period alone lets its WAL tail
/// grow to SlateDB's own bound of 4096 WAL SSTs, all replayed by every reader
/// cold open. Unset → [`DEFAULT_UNFLUSHED_MAX_AGE_SECS`]; 0 → quiet period
/// only. Cost: at most one L0 SST + manifest update per period per trickled
/// collection.
const ENV_UNFLUSHED_MAX_AGE_SECS: &str = "HELIX_LSM_UNFLUSHED_MAX_AGE_SECS";
const DEFAULT_UNFLUSHED_MAX_AGE_SECS: usize = 300;

fn unflushed_max_age_from_env() -> Option<std::time::Duration> {
    let secs = env_usize_opt(ENV_UNFLUSHED_MAX_AGE_SECS).unwrap_or(DEFAULT_UNFLUSHED_MAX_AGE_SECS);
    (secs > 0).then(|| std::time::Duration::from_secs(secs as u64))
}

/// Write bookkeeping for the idle memtable flush ([`ENV_IDLE_FLUSH_SECS`]).
struct IdleFlushState {
    created: std::time::Instant,
    /// Committed-write generation. Starts one ahead of `flushed` so a WAL tail
    /// replayed into the memtable at open is flushed after one quiet period.
    writes: std::sync::atomic::AtomicU64,
    /// `writes` value covered by the last successful idle flush.
    flushed: std::sync::atomic::AtomicU64,
    /// Millis after `created` of the last committed write.
    last_write_ms: std::sync::atomic::AtomicU64,
    /// Millis after `created` of the oldest write no flush has covered, or
    /// just before it. Starts at the open: a replayed tail is at least that old.
    unflushed_since_ms: std::sync::atomic::AtomicU64,
}

impl IdleFlushState {
    fn new() -> Self {
        Self {
            created: std::time::Instant::now(),
            writes: std::sync::atomic::AtomicU64::new(1),
            flushed: std::sync::atomic::AtomicU64::new(0),
            last_write_ms: std::sync::atomic::AtomicU64::new(0),
            unflushed_since_ms: std::sync::atomic::AtomicU64::new(0),
        }
    }

    fn elapsed_ms(&self) -> u64 {
        u64::try_from(self.created.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    /// Called after every committed write: `commit`, `commit_buffered`,
    /// `update_key_transactional`, and `compare_patch_nodes` are this
    /// backend's only `Db` write paths (`db()` is private to this module).
    fn note_write(&self) {
        use std::sync::atomic::Ordering;
        let now = self.elapsed_ms();
        self.last_write_ms.store(now, Ordering::Release);
        // The first write after a flush that covered everything starts the
        // unflushed tail. A flush in flight has not stored `flushed` yet, so a
        // write landing under it keeps that flush's start instead.
        if self.writes.load(Ordering::Acquire) == self.flushed.load(Ordering::Acquire) {
            self.unflushed_since_ms.store(now, Ordering::Release);
        }
        self.writes.fetch_add(1, Ordering::AcqRel);
    }

    /// A successful flush that began at `started_ms` covered generation
    /// `writes`. Anything committed since is unflushed from about its start.
    fn note_flushed(&self, writes: u64, started_ms: u64) {
        use std::sync::atomic::Ordering;
        self.unflushed_since_ms.store(started_ms, Ordering::Release);
        self.flushed.store(writes, Ordering::Release);
    }

    /// The write generation to flush, and why: writes are unflushed and either
    /// none landed within `quiet` or the oldest of them is `max_age` old.
    fn due(
        &self,
        quiet: std::time::Duration,
        max_age: Option<std::time::Duration>,
    ) -> Option<(u64, &'static str)> {
        use std::sync::atomic::Ordering;
        let writes = self.writes.load(Ordering::Acquire);
        if writes == self.flushed.load(Ordering::Acquire) {
            return None;
        }
        let now = self.elapsed_ms();
        let ms_since =
            |mark: &std::sync::atomic::AtomicU64| now.saturating_sub(mark.load(Ordering::Acquire));
        let ms = |limit: std::time::Duration| u64::try_from(limit.as_millis()).unwrap_or(u64::MAX);
        if ms_since(&self.last_write_ms) >= ms(quiet) {
            Some((writes, "quiet"))
        } else if max_age.is_some_and(|age| ms_since(&self.unflushed_since_ms) >= ms(age)) {
            Some((writes, "age"))
        } else {
            None
        }
    }
}

/// Background idle flusher for one writer `Db`. Exits once the backend is
/// dropped (its state `Arc` is gone) or the `Db` is closed or fenced.
fn spawn_idle_flusher(
    rt: &Handle,
    db: Db,
    state: &Arc<IdleFlushState>,
    quiet: std::time::Duration,
    max_age: Option<std::time::Duration>,
) {
    let state = Arc::downgrade(state);
    let shortest = max_age.map_or(quiet, |age| age.min(quiet));
    let tick = (shortest / 4).max(std::time::Duration::from_secs(1));
    rt.spawn(async move {
        loop {
            tokio::time::sleep(tick).await;
            let Some(state) = state.upgrade() else {
                return;
            };
            if db.status().close_reason.is_some() {
                return;
            }
            let started_ms = state.elapsed_ms();
            let Some((writes, trigger)) = state.due(quiet, max_age) else {
                continue;
            };
            let flushed = db
                .flush_with_options(slatedb::config::FlushOptions {
                    flush_type: slatedb::config::FlushType::MemTable,
                })
                .await;
            match flushed {
                Ok(()) => {
                    state.note_flushed(writes, started_ms);
                    metrics::counter!("helix_lsm_idle_flush_total", "outcome" => "flushed", "trigger" => trigger)
                        .increment(1);
                }
                Err(error) => {
                    metrics::counter!("helix_lsm_idle_flush_total", "outcome" => "error", "trigger" => trigger)
                        .increment(1);
                    tracing::debug!(%error, "idle memtable flush failed; retrying next tick");
                }
            }
        }
    });
}

pub struct LsmBackend {
    db: RwLock<Db>,
    /// `Handle` to the process-shared LSM runtime (see [`shared_lsm_runtime`]),
    /// NOT an owned runtime — so opening N collections does not spawn N pools.
    rt: Handle,
    /// Retained object store for prefix purge on collection drop.
    store: Arc<dyn ObjectStore>,
    /// SlateDB root path (= the object-store prefix for this collection,
    /// e.g. `"shadow/my_collection"`). Used to compute the purge prefix and
    /// the SSD-cache subtree path on drop.
    path: String,
    settings: Option<Settings>,
    /// True when `commit_buffered` writes landed since the last `flush_durable`
    /// barrier, so the barrier knows to publish this collection on the
    /// durable-commit change feed (see [`super::change_feed`]).
    buffered_since_flush: std::sync::atomic::AtomicBool,
    /// Write bookkeeping for the idle memtable flush ([`ENV_IDLE_FLUSH_SECS`]).
    idle_flush: Arc<IdleFlushState>,
    /// Rate limit of this collection's [`ENV_STALLED_WRITE_REFRESH`] refreshes.
    stalled_refresh: RefreshGate,
    /// RAII guard that unregisters this collection's cache dir from the
    /// open registry on drop, so the evictor can reclaim it.
    _cache_dir_guard: Option<cache_evictor::CacheDirGuard>,
    /// Test-only counter of backend reads served (one per `get_with` point read,
    /// one per KV yielded by `scan`/`scan_dup_namespace`). Lets tests assert that
    /// an indexed search is HNSW-bounded rather than a full namespace scan. Never
    /// touched on production (non-test) builds.
    #[cfg(test)]
    read_count: std::sync::atomic::AtomicUsize,
}

/// Read handle. Holds a SlateDB snapshot so all reads/scans in one logical
/// transaction observe the same committed sequence.
#[derive(Clone)]
pub struct LsmRead {
    snapshot: Arc<DbSnapshot>,
    /// Read-your-writes overlay (full SlateDB key -> Some(value)|None) cloned
    /// from a LsmWrite when this read view is derived from a write batch via
    /// WriteView::read_view; empty for ordinary begin_read() snapshots.
    pending: HashMap<Vec<u8>, Option<Bytes>>,
    /// Taken just before the snapshot was opened: every commit that returned
    /// before this instant is visible in `snapshot`.
    opened_at: std::time::Instant,
}
impl ReadTxn for LsmRead {}

impl LsmRead {
    /// Build a read handle whose point reads first consult `pending`
    /// (read-your-writes overlay derived from a live write batch).
    pub(crate) fn with_pending(
        snapshot: Arc<DbSnapshot>,
        pending: HashMap<Vec<u8>, Option<Bytes>>,
        opened_at: std::time::Instant,
    ) -> Self {
        Self {
            snapshot,
            pending,
            opened_at,
        }
    }

    pub(crate) fn opened_at(&self) -> std::time::Instant {
        self.opened_at
    }

    /// The snapshot's sequence when this view reads only committed state;
    /// `None` for a write view carrying uncommitted puts or deletes.
    pub(crate) fn snapshot_seq(&self) -> Option<u64> {
        self.pending.is_empty().then(|| self.snapshot.seq())
    }
}

/// Write batch accumulating puts/deletes, applied atomically on commit.
///
/// `pending` mirrors the batch's effects keyed by FULL SlateDB key
/// (`Some(value)` for a put, `None` for a delete) so `get_for_update` can offer
/// read-your-writes — SlateDB's `WriteBatch` is write-only.
pub struct LsmWrite {
    batch: WriteBatch,
    pending: HashMap<Vec<u8>, Option<Bytes>>,
    has_writes: bool,
    /// Merges are not in `pending`; see [`LsmCheckedTxn::commit_write`].
    merged: bool,
    /// Run when the batch ends: `true` once it committed, `false` when it is
    /// dropped uncommitted or the commit fails. Lets a writer that published
    /// in-memory state ahead of the commit take it back on abort.
    post_txn: Vec<PostTxnHook>,
}
impl WriteTxn for LsmWrite {}

/// See [`LsmWrite::on_txn_end`].
pub(crate) type PostTxnHook = Box<dyn FnOnce(bool) + Send>;

/// Newest first, so undo hooks stacked on one key unwind in order.
fn run_post_txn(hooks: Vec<PostTxnHook>, committed: bool) {
    for hook in hooks.into_iter().rev() {
        hook(committed);
    }
}

impl LsmWrite {
    /// The batch's buffered effects (full SlateDB key -> Some(value)|None),
    /// cloned into a read view to give read-your-writes (see `LsmRead`).
    pub(crate) fn pending(&self) -> &HashMap<Vec<u8>, Option<Bytes>> {
        &self.pending
    }

    /// Register `hook` to run with `true` after this batch commits, or with
    /// `false` if it is dropped uncommitted or its commit fails.
    pub(crate) fn on_txn_end(&mut self, hook: PostTxnHook) {
        self.post_txn.push(hook);
    }

    /// Detach the batch and its hooks for a commit; what is left drops as an
    /// empty batch with nothing to run.
    fn take_for_commit(mut self) -> (WriteBatch, Vec<PostTxnHook>, bool) {
        (
            std::mem::replace(&mut self.batch, WriteBatch::new()),
            std::mem::take(&mut self.post_txn),
            self.has_writes,
        )
    }
}

impl Drop for LsmWrite {
    fn drop(&mut self) {
        // Reached only for a batch that never committed: hooks taken by a
        // commit leave this empty.
        run_post_txn(std::mem::take(&mut self.post_txn), false);
    }
}

impl LsmBackend {
    /// Open a SlateDB database at `path` backed by `store` (S3, GCS, local FS, …).
    ///
    /// When [`ENV_CACHE_DIR`] is set, the db is opened via the builder with a
    /// local-filesystem object-store cache (SSD hot cache) so cold reads hit the
    /// object store only on cache miss. Unset → bare open (current behavior).
    pub fn open_with_store(path: &str, store: Arc<dyn ObjectStore>) -> Result<Self, BackendError> {
        Self::open_with_store_settings(path, store, lsm_settings_from_env(path))
    }

    /// Open `store` at `path`, applying `settings` via the builder when present;
    /// a bare `Db::open` otherwise. The `settings` channel is how the optional
    /// object-store cache (and any future tuning) is attached.
    fn open_with_store_settings(
        path: &str,
        store: Arc<dyn ObjectStore>,
        settings: Option<Settings>,
    ) -> Result<Self, BackendError> {
        let rt = shared_lsm_handle();
        // Register cache dir BEFORE opening so the evictor cannot delete it during build.
        // If open fails, the guard's Drop will unregister it.
        let cache_dir_guard =
            cache_subtree_for_lsm_path(path).map(cache_evictor::register_cache_dir);
        let db = Self::open_db_for_request(&rt, path, store.clone(), settings.clone())?;
        let idle_flush = Arc::new(IdleFlushState::new());
        if let Some(quiet) = idle_flush_after_from_env() {
            spawn_idle_flusher(
                &rt,
                db.clone(),
                &idle_flush,
                quiet,
                unflushed_max_age_from_env(),
            );
        }
        Ok(Self {
            db: RwLock::new(db),
            rt,
            store,
            path: path.to_string(),
            settings,
            buffered_since_flush: std::sync::atomic::AtomicBool::new(false),
            idle_flush,
            stalled_refresh: RefreshGate::default(),
            _cache_dir_guard: cache_dir_guard,
            #[cfg(test)]
            read_count: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// [`batch_row_read_in_flight`] on this collection.
    pub(crate) fn batch_row_read_in_flight(
        &self,
    ) -> slatedb::cached_object_store_stats::BatchRowRead {
        batch_row_read_in_flight(&self.path)
    }

    /// Publish this collection on the durable-commit change feed. The feed key
    /// is the collection name as readers/the manager know it — the last path
    /// component of the SlateDB prefix (e.g. `"prod/my_coll"` → `"my_coll"`).
    fn publish_durable_commit(&self) {
        let name = self.path.rsplit('/').next().unwrap_or(&self.path);
        super::change_feed::change_feed().publish(name);
    }

    /// Opening a writer is deliberately uncancellable. SlateDB's build path
    /// can create/fence manifests and start background tasks before the future
    /// resolves, so dropping it would make the open outcome ambiguous just as
    /// dropping a commit would.
    fn open_db_uncancellable(
        rt: &Handle,
        path: &str,
        store: Arc<dyn ObjectStore>,
        settings: Option<Settings>,
    ) -> Result<Db, BackendError> {
        // Share ONE in-memory block/meta cache across every collection handle
        // instead of SlateDB's per-`Db` default `SplitCache` (which would silo
        // ~640 MiB per resident handle). See [`shared_db_cache`].
        let cache = shared_db_cache();
        // See `opening_a_new_database`.
        let create_only = LSM_OPENS_A_NEW_DATABASE.get().is_some();
        let mut builder = Db::builder(path, store)
            .with_db_cache(cache)
            .with_merge_operator(shared_merge_operator())
            .with_manifest_id_floor(purged_manifest_ceiling(path))
            .with_create_only(create_only)
            .with_l0_flush_gate(flush_gate_for_open());
        if let Some(settings) = settings {
            builder = builder.with_settings(settings);
        }
        if let Some(policies) = prefix_filter_policies_from_env() {
            builder = builder.with_filter_policies(policies);
        }
        #[cfg(test)]
        if let Some(add) = DB_BUILDER_FOR_TEST.with(|add| add.borrow().clone()) {
            builder = add(builder);
        }
        let opened = block_on_lsm_mapped(rt, builder.build(), |error: slatedb::Error| {
            if error.code() == Some(slatedb::ErrorCode::DatabaseExists) {
                LSM_OPENS_A_NEW_DATABASE.set(Some(true));
            }
            io(error)
        });
        if create_only && opened.is_ok() {
            LSM_OPENS_A_NEW_DATABASE.set(None);
        }
        opened
    }

    /// Open a writer for a foreground operation without dropping the
    /// side-effecting build future. An already-abandoned request never starts
    /// the build. Once the build starts it must finish and the new handle must
    /// be installed even if cancellation arrives: the build may have fenced a
    /// prior writer, so discarding its result could strand the collection.
    /// The first subsequent side-effect-free read observes cancellation.
    fn open_db_for_request(
        rt: &Handle,
        path: &str,
        store: Arc<dyn ObjectStore>,
        settings: Option<Settings>,
    ) -> Result<Db, BackendError> {
        ensure_lsm_read_not_cancelled()?;
        Self::open_db_uncancellable(rt, path, store, settings)
    }

    fn db(&self) -> Db {
        self.db.read().clone()
    }

    /// Resolves with `write`, a commit on `db`. Unless
    /// [`ENV_STALLED_WRITE_REFRESH`] is off, a write still pending after
    /// [`STALLED_WRITE_REFRESH_EVERY`] runs [`Self::refresh_for_stalled_write`]
    /// every such interval until it returns. The write, what it waits for and
    /// its result are the caller's, unchanged.
    async fn refresh_if_stalled<T>(&self, db: &Db, write: impl Future<Output = T>) -> T {
        if !lsm_stalled_write_refresh_enabled() {
            return write.await;
        }
        let pending_since = std::time::Instant::now();
        let announced = Cell::new(false);
        refresh_while_pending(write, STALLED_WRITE_REFRESH_EVERY, || {
            self.refresh_for_stalled_write(db, pending_since, &announced)
        })
        .await
    }

    /// One manifest refresh for a commit pending since `pending_since`,
    /// counted in `helix_lsm_stalled_write_refresh_total` by outcome: `freed`
    /// (it found L0 slots the compactor freed, which is what resumes the
    /// parked writes), `ok` (L0 unchanged), `error`, or `skipped`. Skipped
    /// unless L0 is at its limit in this writer's view of the manifest (the
    /// one wait a refresh can end), and while [`RefreshGate`] holds it back.
    /// A failed refresh is only logged: it fails when the `Db` is closed or
    /// closing, which the commit reports itself. One still in flight when its
    /// commit returns is dropped uncounted.
    async fn refresh_for_stalled_write(
        &self,
        db: &Db,
        pending_since: std::time::Instant,
        announced: &Cell<bool>,
    ) {
        // shortcut: the root tree's L0 only, upgrade if a segment extractor is ever configured.
        let l0_ssts = db.manifest().l0().len();
        let l0_limit = self
            .settings
            .as_ref()
            .map_or_else(|| l0_stall_floor(&Settings::default()), l0_stall_floor);
        let permit = (l0_ssts >= l0_limit)
            .then(|| self.stalled_refresh.try_enter(STALLED_WRITE_REFRESH_EVERY))
            .flatten();
        let outcome = match permit {
            None => "skipped",
            Some(_permit) => {
                if !announced.replace(true) {
                    tracing::info!(
                        path = %self.path,
                        pending_ms = pending_since.elapsed().as_millis() as u64,
                        l0_ssts,
                        "LSM commit stalled on a full L0: refreshing the manifest"
                    );
                }
                match db.refresh_manifest().await {
                    Ok(()) if db.manifest().l0().len() < l0_ssts => "freed",
                    Ok(()) => "ok",
                    Err(error) => {
                        tracing::debug!(%error, "manifest refresh for a stalled LSM write failed");
                        "error"
                    }
                }
            }
        };
        metrics::counter!("helix_lsm_stalled_write_refresh_total", "outcome" => outcome)
            .increment(1);
    }

    fn is_conflict(error: &BackendError) -> bool {
        matches!(error, BackendError::Conflict(_))
    }

    fn is_fenced_conflict(error: &BackendError) -> bool {
        lsm_conflict_message(error).is_some_and(lsm_error_message_is_fenced)
    }

    /// Test-only: number of backend reads served since the last `reset_read_count`
    /// (one per `get_with` point read, one per KV yielded by a scan).
    #[cfg(test)]
    pub fn read_count(&self) -> usize {
        self.read_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Test-only: flush the memtable to an L0 SST.
    #[cfg(test)]
    pub(crate) fn flush_memtable_for_test(&self) {
        block_on_lsm(
            &self.rt,
            self.db().flush_with_options(slatedb::config::FlushOptions {
                flush_type: slatedb::config::FlushType::MemTable,
            }),
        )
        .unwrap();
    }

    /// Test-only: flush the memtable to an L0 SST, then close the writer.
    #[cfg(test)]
    pub(crate) fn flush_memtable_and_close_for_test(&self) {
        self.flush_memtable_for_test();
        self.close().unwrap();
    }

    /// Test-only: zero the read counter before a measured operation.
    #[cfg(test)]
    pub fn reset_read_count(&self) {
        self.read_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Increment the test-only read counter; compiled away in non-test builds.
    #[inline(always)]
    fn bump_reads(&self, _n: usize) {
        #[cfg(test)]
        self.read_count
            .fetch_add(_n, std::sync::atomic::Ordering::Relaxed);
    }

    /// Account the rows one dup-namespace scan pulled from the backend.
    #[inline(always)]
    fn note_dup_scan_rows(&self, visited: usize) {
        self.bump_reads(visited);
        #[cfg(test)]
        DUP_SCAN_ROWS_FOR_TEST.with(|count| count.set(count.get() + visited));
    }

    /// Open an in-memory SlateDB (for tests). Always cache-less: caching an
    /// in-memory store is pointless and would create temp dirs in tests, so this
    /// bypasses the [`ENV_CACHE_DIR`] env path regardless of the environment.
    pub fn open_in_memory(path: &str) -> Result<Self, BackendError> {
        Self::open_with_store_settings(path, Arc::new(InMemory::new()), None)
    }

    /// Begin a fresh committed snapshot with a read-your-writes overlay
    /// (`pending`) attached. Used by `WriteView::read_view` to read a live write
    /// batch's own buffered writes on the LSM backend.
    pub(crate) fn begin_read_with_pending(
        &self,
        pending: HashMap<Vec<u8>, Option<Bytes>>,
    ) -> Result<LsmRead, BackendError> {
        let db = self.db();
        let opened_at = std::time::Instant::now();
        let snapshot = match block_on_lsm_read_mapped(&self.rt, db.snapshot(), map_slatedb_err) {
            Ok(snapshot) => snapshot,
            Err(error) => return Err(error),
        };
        Ok(LsmRead::with_pending(snapshot, pending, opened_at))
    }

    pub(crate) fn collect_values_many_with(
        &self,
        txn: &LsmRead,
        ns: Namespace<'_>,
        keys: &[Vec<u8>],
    ) -> Result<Vec<Option<Vec<u8>>>, BackendError> {
        self.collect_values_many(txn, ns, keys, lsm_multi_get_enabled())
    }

    /// [`Self::collect_values_many_with`] with the read path chosen by the
    /// caller: `multi_get` for a batch of keys that mostly do not exist.
    pub(crate) fn collect_values_many(
        &self,
        txn: &LsmRead,
        ns: Namespace<'_>,
        keys: &[Vec<u8>],
        multi_get: bool,
    ) -> Result<Vec<Option<Vec<u8>>>, BackendError> {
        #[cfg(test)]
        note_key_batch_for_test(&self.path);
        self.collect_full_keys(
            txn,
            keys.iter().map(|key| prefixed(ns, key)).collect(),
            multi_get,
        )
    }

    /// Whether each `(namespace, key)` is present, in input order. Every
    /// namespace of a collection shares one keyspace, so probes spanning many
    /// namespaces still cost ONE batched read.
    pub(crate) fn contains_many_with(
        &self,
        txn: &LsmRead,
        keys: &[(Namespace<'_>, Vec<u8>)],
    ) -> Result<Vec<bool>, BackendError> {
        let full_keys = keys.iter().map(|(ns, key)| prefixed(*ns, key)).collect();
        let values = self.collect_full_keys_with(txn, full_keys)?;
        Ok(values.into_iter().map(|value| value.is_some()).collect())
    }

    /// Values of fully prefixed keys, aligned with `full_keys`: the overlay in
    /// `txn` wins, the rest come from one batched snapshot read.
    fn collect_full_keys_with(
        &self,
        txn: &LsmRead,
        full_keys: Vec<Vec<u8>>,
    ) -> Result<Vec<Option<Vec<u8>>>, BackendError> {
        self.collect_full_keys(txn, full_keys, lsm_multi_get_enabled())
    }

    fn collect_full_keys(
        &self,
        txn: &LsmRead,
        full_keys: Vec<Vec<u8>>,
        multi_get: bool,
    ) -> Result<Vec<Option<Vec<u8>>>, BackendError> {
        self.bump_reads(full_keys.len());
        let mut out = vec![None; full_keys.len()];
        let mut snapshot_reads = Vec::new();

        for (index, full) in full_keys.into_iter().enumerate() {
            if let Some(buffered) = txn.pending.get(&full) {
                out[index] = buffered.as_ref().map(|bytes| bytes.to_vec());
            } else {
                snapshot_reads.push((index, full));
            }
        }

        if snapshot_reads.is_empty() {
            return Ok(out);
        }

        let snapshot = Arc::clone(&txn.snapshot);
        if multi_get {
            // One ordered multi_get pass over the snapshot: the fork's
            // implementation shares block fetches across keys and preserves
            // input order, so values line up with `snapshot_reads`.
            let keys: Vec<Vec<u8>> = snapshot_reads
                .iter()
                .map(|(_, full)| full.clone())
                .collect();
            let fetched = block_on_lsm_read_mapped(
                &self.rt,
                async move { snapshot.multi_get(&keys).await.map_err(map_slatedb_err) },
                |e| e,
            )?;
            for ((index, _), value) in snapshot_reads.into_iter().zip(fetched) {
                out[index] = value.map(|bytes| bytes.to_vec());
            }
            return Ok(out);
        }
        let fetched = block_on_lsm_read_mapped(
            &self.rt,
            async move {
                let reads = snapshot_reads.into_iter().map(|(index, full)| {
                    let snapshot = Arc::clone(&snapshot);
                    async move {
                        snapshot
                            .get(full.as_slice())
                            .await
                            .map(|value| (index, value.map(|bytes| bytes.to_vec())))
                            .map_err(map_slatedb_err)
                    }
                });
                stream::iter(reads)
                    .buffered(LSM_BATCH_POINT_READ_CONCURRENCY)
                    .try_collect::<Vec<_>>()
                    .await
            },
            |e| e,
        )?;
        for (index, value) in fetched {
            out[index] = value;
        }
        Ok(out)
    }

    /// Batched [`StorageBackend::get_for_update`]: values buffered in `txn`
    /// win (read-your-writes); the rest come from ONE multi-get against a fresh
    /// committed snapshot instead of one SlateDB round trip per key.
    pub(crate) fn get_for_update_many(
        &self,
        txn: &LsmWrite,
        ns: Namespace<'_>,
        keys: &[Vec<u8>],
    ) -> Result<Vec<Option<Vec<u8>>>, BackendError> {
        let pending = keys
            .iter()
            .filter_map(|key| {
                let full = prefixed(ns, key);
                txn.pending.get(&full).map(|value| (full, value.clone()))
            })
            .collect();
        let read = self.begin_read_with_pending(pending)?;
        self.collect_values_many_with(&read, ns, keys)
    }

    /// Whether each row is stored, or staged on `txn`, in input order:
    /// `(ns, key, None)` a plain key, `(ns, key, Some(value))` one row of a
    /// multi-value namespace. A row `txn` puts is there and one it deletes
    /// is not, whatever the store holds; the rest come from ONE multi-get,
    /// as in [`Self::get_for_update_many`]. A row is there by its key: the
    /// stored value of a multi-value row is empty.
    pub(crate) fn present_for_update(
        &self,
        txn: &LsmWrite,
        rows: &[(Namespace<'_>, &[u8], Option<&[u8]>)],
    ) -> Result<Vec<bool>, BackendError> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        #[cfg(test)]
        if FAIL_NEXT_PRESENT_FOR_UPDATE_FOR_TEST.replace(false) {
            return Err(BackendError::Io("the keys' read fails for the test".into()));
        }
        let full_keys: Vec<Vec<u8>> = rows
            .iter()
            .map(|(ns, key, value)| match value {
                Some(value) => dup_composite(*ns, key, value),
                None => prefixed(*ns, key),
            })
            .collect();
        let pending = full_keys
            .iter()
            .filter_map(|full| {
                txn.pending
                    .get(full)
                    .map(|value| (full.clone(), value.clone()))
            })
            .collect();
        let read = self.begin_read_with_pending(pending)?;
        // A key the read does not answer stays `None` there: not seen.
        let values = self.collect_full_keys_with(&read, full_keys)?;
        Ok(values.iter().map(Option::is_some).collect())
    }

    /// The keys under each `(namespace, key prefix)`, namespace prefix
    /// stripped, in input order: the prefix scans run concurrently in ONE
    /// round trip instead of one each. Serves plain namespaces on a snapshot
    /// without a write overlay; `None` otherwise, and the caller scans one
    /// prefix at a time.
    pub(crate) fn scan_prefix_keys_many_with(
        &self,
        txn: &LsmRead,
        prefixes: &[(Namespace<'_>, Vec<u8>)],
    ) -> Result<Option<Vec<Vec<Vec<u8>>>>, BackendError> {
        if !txn.pending.is_empty() || prefixes.iter().any(|(ns, _)| is_dup(*ns)) {
            return Ok(None);
        }
        let scans: Vec<(usize, Vec<u8>)> = prefixes
            .iter()
            .map(|(ns, key_prefix)| {
                let mut full = ns_prefix(*ns);
                let strip = full.len();
                full.extend_from_slice(key_prefix);
                (strip, full)
            })
            .collect();
        let snapshot = Arc::clone(&txn.snapshot);
        let rows = block_on_lsm_read_mapped(
            &self.rt,
            async move {
                let scans = scans.into_iter().map(|(strip, full)| {
                    let snapshot = Arc::clone(&snapshot);
                    async move {
                        let mut keys = Vec::new();
                        let mut iter = snapshot
                            .scan_prefix_with_options(full, .., &lsm_scan_options())
                            .await
                            .map_err(io)?;
                        while let Some(kv) = iter.next().await.map_err(io)? {
                            let key = kv.key.as_ref();
                            keys.push(key.get(strip..).unwrap_or(key).to_vec());
                        }
                        Ok::<Vec<Vec<u8>>, BackendError>(keys)
                    }
                });
                stream::iter(scans)
                    .buffered(LSM_BATCH_POINT_READ_CONCURRENCY)
                    .try_collect::<Vec<_>>()
                    .await
            },
            |e| e,
        )?;
        self.bump_reads(rows.iter().map(Vec::len).sum());
        Ok(Some(rows))
    }

    /// The dup values whose logical key starts with each prefix, in input
    /// order and in the order [`Self::scan`] over `KeyRange::prefix` yields
    /// them, read concurrently in ONE round trip instead of one each. Every
    /// prefix reads exactly its own rows: the key lengths present in the
    /// namespace are found once, and a prefix is one bounded range per length.
    /// With a `row_cap` a prefix stops at its first `row_cap` values and reads
    /// a block at a time: an existence check wants one row of a prefix that
    /// may own tens of thousands. A cap of 0 counts as 1, so an empty answer
    /// always means the prefix has no row. `None` on a snapshot with a write
    /// overlay, and the caller scans one prefix at a time.
    pub(crate) fn scan_dup_prefix_values_many_with(
        &self,
        txn: &LsmRead,
        ns: Namespace<'_>,
        key_prefixes: &[Vec<u8>],
        row_cap: Option<usize>,
    ) -> Result<Option<Vec<Vec<Vec<u8>>>>, BackendError> {
        if !txn.pending.is_empty() || !is_dup(ns) {
            return Ok(None);
        }
        let Some(shortest) = key_prefixes.iter().map(Vec::len).min() else {
            return Ok(Some(Vec::new()));
        };
        let prefix = ns_prefix(ns);
        let options = match row_cap {
            Some(_) => lsm_scan_options()
                .with_read_ahead_bytes(1)
                .with_max_fetch_tasks(1),
            None => lsm_scan_options(),
        };
        let row_cap = row_cap.map_or(usize::MAX, |cap| cap.max(1));
        let (rows, visited) = block_on_lsm_read_mapped(
            &self.rt,
            async {
                let snapshot = &txn.snapshot;
                let (lengths, probed) = dup_key_lengths(snapshot, &prefix, shortest).await?;
                let scans = key_prefixes.iter().map(|key_prefix| {
                    let (prefix, lengths) = (prefix.as_slice(), lengths.as_slice());
                    let options = &options;
                    async move {
                        let mut values = Vec::new();
                        for len in lengths {
                            if values.len() >= row_cap {
                                break;
                            }
                            if (*len as usize) < key_prefix.len() {
                                continue;
                            }
                            let lo =
                                [prefix, &len.to_be_bytes()[..], key_prefix.as_slice()].concat();
                            let hi = match next_prefix(&lo) {
                                Some(end) => Bound::Excluded(end),
                                None => Bound::Unbounded,
                            };
                            let mut iter = snapshot
                                .scan_with_options((Bound::Included(lo), hi), options)
                                .await
                                .map_err(io)?;
                            while values.len() < row_cap {
                                let Some(kv) = iter.next().await.map_err(io)? else {
                                    break;
                                };
                                if let Some((_, value)) =
                                    decode_dup_full_key(kv.key.as_ref(), prefix)?
                                {
                                    values.push(value.to_vec());
                                }
                            }
                        }
                        Ok::<_, BackendError>(values)
                    }
                });
                let rows = stream::iter(scans)
                    .buffered(LSM_BATCH_POINT_READ_CONCURRENCY)
                    .try_collect::<Vec<_>>()
                    .await?;
                Ok::<_, BackendError>((rows, probed))
            },
            |e| e,
        )?;
        self.note_dup_scan_rows(visited + rows.iter().map(Vec::len).sum::<usize>());
        Ok(Some(rows))
    }

    pub(crate) fn collect_dup_values_many_with(
        &self,
        txn: &LsmRead,
        ns: Namespace<'_>,
        keys: &[Vec<u8>],
    ) -> Result<Vec<Vec<Vec<u8>>>, BackendError> {
        let prefixes: Vec<Vec<u8>> = keys.iter().map(|key| dup_key_prefix(ns, key)).collect();
        let snapshot = Arc::clone(&txn.snapshot);
        let pending = Arc::new(txn.pending.clone());
        block_on_lsm_read_mapped(
            &self.rt,
            async move {
                let scans = prefixes.into_iter().map(|kp| {
                    let snapshot = Arc::clone(&snapshot);
                    let pending = Arc::clone(&pending);
                    async move {
                        let mut rows = BTreeMap::new();
                        let mut iter = snapshot
                            .scan_prefix_with_options(kp.clone(), .., &lsm_scan_options())
                            .await
                            .map_err(io)?;
                        while let Some(kv) = iter.next().await.map_err(io)? {
                            let full_key = kv.key.as_ref();
                            if pending.contains_key(full_key) {
                                continue;
                            }
                            let value = if full_key.len() >= kp.len() {
                                full_key[kp.len()..].to_vec()
                            } else {
                                Vec::new()
                            };
                            rows.insert(full_key.to_vec(), value);
                        }
                        for (full_key, pending_value) in pending.iter() {
                            if !full_key.starts_with(&kp) {
                                continue;
                            }
                            if pending_value.is_none() {
                                rows.remove(full_key);
                                continue;
                            }
                            let value = if full_key.len() >= kp.len() {
                                full_key[kp.len()..].to_vec()
                            } else {
                                Vec::new()
                            };
                            rows.insert(full_key.clone(), value);
                        }
                        Ok::<Vec<Vec<u8>>, BackendError>(rows.into_values().collect())
                    }
                });
                stream::iter(scans)
                    .buffered(LSM_BATCH_POINT_READ_CONCURRENCY)
                    .try_collect::<Vec<_>>()
                    .await
            },
            |e| e,
        )
    }

    pub(crate) fn update_key_transactional<T>(
        &self,
        ns: Namespace<'_>,
        key: &[u8],
        mut update: impl FnMut(Option<&[u8]>) -> Result<(Vec<u8>, T), BackendError>,
    ) -> Result<T, BackendError> {
        self.update_key_in_transaction(ns, key, &[], true, |value| {
            update(value).map(|(next, result)| (next, Vec::new(), result))
        })
    }

    /// [`Self::update_key_transactional`] with the rows `also` (key and value,
    /// in `ns` too) written by the same transaction, committed BUFFERED:
    /// visible at once and not yet durable. The caller owes `flush_durable`
    /// before it acknowledges the write.
    pub(crate) fn update_key_transactional_buffered<T>(
        &self,
        ns: Namespace<'_>,
        key: &[u8],
        also: &[(&[u8], &[u8])],
        mut update: impl FnMut(Option<&[u8]>) -> Result<(Vec<u8>, T), BackendError>,
    ) -> Result<T, BackendError> {
        self.update_key_in_transaction(ns, key, also, false, |value| {
            update(value).map(|(next, result)| (next, Vec::new(), result))
        })
    }

    /// [`Self::update_key_transactional`] whose `update` also names rows of
    /// `ns` for the same transaction: a key with the value to put, or with
    /// none to delete it. They are decided from the value the transaction
    /// read, and decided again by each run a conflict makes, so what commits
    /// beside the new value is what follows from the value it replaced.
    pub(crate) fn update_key_transactional_with_rows<T>(
        &self,
        ns: Namespace<'_>,
        key: &[u8],
        update: impl FnMut(Option<&[u8]>) -> Result<(Vec<u8>, Vec<TransactionRow>, T), BackendError>,
    ) -> Result<T, BackendError> {
        self.update_key_in_transaction(ns, key, &[], true, update)
    }

    fn update_key_in_transaction<T>(
        &self,
        ns: Namespace<'_>,
        key: &[u8],
        also: &[(&[u8], &[u8])],
        await_durable: bool,
        mut update: impl FnMut(Option<&[u8]>) -> Result<(Vec<u8>, Vec<TransactionRow>, T), BackendError>,
    ) -> Result<T, BackendError> {
        let db = self.db();
        let full = prefixed(ns, key);
        let also: Vec<(Vec<u8>, &[u8])> = also
            .iter()
            .map(|(key, value)| (prefixed(ns, key), *value))
            .collect();
        let options = WriteOptions {
            await_durable,
            ..Default::default()
        };
        for attempt in 0..LSM_METADATA_TXN_MAX_ATTEMPTS {
            let result = block_on_lsm_mapped(
                &self.rt,
                async {
                    let txn = db
                        .begin(IsolationLevel::SerializableSnapshot)
                        .await
                        .map_err(map_slatedb_err)?;
                    let current = txn.get(full.as_slice()).await.map_err(map_slatedb_err)?;
                    let (next, rows, result) = update(current.as_deref())?;
                    txn.put(full.as_slice(), next.as_slice())
                        .map_err(map_slatedb_err)?;
                    for (key, value) in &also {
                        txn.put(key.as_slice(), *value).map_err(map_slatedb_err)?;
                    }
                    for (key, value) in &rows {
                        let key = prefixed(ns, key);
                        match value {
                            Some(value) => txn.put(key.as_slice(), value.as_slice()),
                            None => txn.delete(key.as_slice()),
                        }
                        .map_err(map_slatedb_err)?;
                    }
                    self.refresh_if_stalled(&db, txn.commit_with_options(&options))
                        .await
                        .map_err(map_slatedb_err)?;
                    self.idle_flush.note_write();
                    if !await_durable {
                        // The flush that makes it durable tells the readers.
                        self.buffered_since_flush
                            .store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                    Ok(result)
                },
                |e| e,
            );
            match result {
                Err(error) if Self::is_fenced_conflict(&error) => {
                    let message = lsm_conflict_message(&error).unwrap_or("").to_string();
                    return Err(lsm_fenced_error("metadata_transaction", message));
                }
                Err(error)
                    if Self::is_conflict(&error) && attempt + 1 < LSM_METADATA_TXN_MAX_ATTEMPTS =>
                {
                    std::thread::yield_now();
                    continue;
                }
                Err(error) if Self::is_conflict(&error) => {
                    let message = lsm_conflict_message(&error).unwrap_or("").to_string();
                    return Err(lsm_retryable_cas_error("metadata_transaction", message));
                }
                other => return other,
            }
        }
        Err(BackendError::Conflict(
            "metadata transaction retry loop exhausted unexpectedly".to_string(),
        ))
    }

    /// All-or-none node-row patch with the comparison re-run on transaction conflicts.
    /// Callers must preserve indexed fields; this is not a general node upsert.
    pub(crate) fn compare_patch_nodes(
        &self,
        ids: &[u128],
        mut patch: impl FnMut(u128, &[u8]) -> Result<Option<Vec<u8>>, BackendError>,
    ) -> Result<bool, BackendError> {
        let db = self.db();
        for attempt in 0..LSM_METADATA_TXN_MAX_ATTEMPTS {
            let result = block_on_lsm_mapped(
                &self.rt,
                async {
                    let txn = db
                        .begin(IsolationLevel::SerializableSnapshot)
                        .await
                        .map_err(map_slatedb_err)?;
                    for id in ids {
                        let key = prefixed(Namespace::Nodes, &id.to_be_bytes());
                        let Some(current) =
                            txn.get(key.as_slice()).await.map_err(map_slatedb_err)?
                        else {
                            return Ok(false);
                        };
                        let Some(next) = patch(*id, &current)? else {
                            return Ok(false);
                        };
                        txn.put(key.as_slice(), next.as_slice())
                            .map_err(map_slatedb_err)?;
                    }
                    self.refresh_if_stalled(&db, txn.commit())
                        .await
                        .map_err(map_slatedb_err)?;
                    self.idle_flush.note_write();
                    self.publish_durable_commit();
                    Ok(true)
                },
                |e| e,
            );
            match result {
                Err(error) if Self::is_fenced_conflict(&error) => {
                    return Err(lsm_fenced_error("node_compare_patch", error.to_string()));
                }
                Err(error)
                    if Self::is_conflict(&error) && attempt + 1 < LSM_METADATA_TXN_MAX_ATTEMPTS =>
                {
                    std::thread::yield_now();
                }
                Err(error) if Self::is_conflict(&error) => {
                    return Err(lsm_retryable_cas_error(
                        "node_compare_patch",
                        error.to_string(),
                    ));
                }
                other => return other,
            }
        }
        Err(BackendError::Conflict(
            "node compare-and-patch retry loop exhausted".to_string(),
        ))
    }

    /// Begin a [`LsmCheckedTxn`]: reads see one sequence, and the commit
    /// fails if another commit touched a written or marked-read key since.
    pub(crate) fn begin_checked(&self) -> Result<LsmCheckedTxn<'_>, BackendError> {
        let txn = block_on_lsm_mapped(
            &self.rt,
            self.db().begin(IsolationLevel::Snapshot),
            map_slatedb_err,
        )?;
        Ok(LsmCheckedTxn { backend: self, txn })
    }
}

/// Snapshot-isolation transaction for background writers that derive writes
/// from what they read and must lose to any foreground write of those keys.
/// SlateDB tracks every commit's keys (plain write batches included) while a
/// transaction is open, so the commit is refused when another commit after
/// [`LsmBackend::begin_checked`] wrote a key this transaction writes or
/// passed to [`Self::mark_read`].
pub(crate) struct LsmCheckedTxn<'a> {
    backend: &'a LsmBackend,
    txn: slatedb::DbTransaction,
}

impl LsmCheckedTxn<'_> {
    /// Visit rows of the non-dup namespace `ns` in `range` (unprefixed keys)
    /// as of the transaction's start, until `visit` returns false. Streams
    /// without filling the block cache.
    pub(crate) fn scan(
        &self,
        ns: Namespace<'_>,
        range: KeyRange,
        mut visit: impl FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        debug_assert!(!is_dup(ns));
        let prefix = ns_prefix(ns);
        let lo = match &range.start {
            Bound::Unbounded => Bound::Included(prefix.clone()),
            other => LsmBackend::prefixed_bound(&prefix, other),
        };
        let hi = match &range.end {
            Bound::Unbounded => next_prefix(&prefix).map_or(Bound::Unbounded, Bound::Excluded),
            other => LsmBackend::prefixed_bound(&prefix, other),
        };
        block_on_lsm_mapped(
            &self.backend.rt,
            async {
                let mut iter = self
                    .txn
                    .scan_with_options((lo, hi), &lsm_streaming_scan_options())
                    .await
                    .map_err(io)?;
                while let Some(kv) = iter.next().await.map_err(io)? {
                    self.backend.bump_reads(1);
                    if !visit(&kv.key[prefix.len()..], kv.value.as_ref()) {
                        break;
                    }
                }
                Ok::<(), BackendError>(())
            },
            |e| e,
        )
    }

    /// Stored values of the dup rows `(key, value)` of `ns`; `None` = no row.
    pub(crate) fn get_dup_stored(
        &self,
        ns: Namespace<'_>,
        rows: &[(&[u8], &[u8])],
    ) -> Result<Vec<Option<Bytes>>, BackendError> {
        let keys: Vec<Vec<u8>> = rows
            .iter()
            .map(|(key, value)| dup_composite(ns, key, value))
            .collect();
        block_on_lsm_mapped(&self.backend.rt, self.txn.multi_get(&keys), io)
    }

    /// Fail the commit if another commit writes `key` of `ns` after begin.
    pub(crate) fn mark_read(&self, ns: Namespace<'_>, key: &[u8]) -> Result<(), BackendError> {
        self.txn.mark_read([prefixed(ns, key)]).map_err(io)
    }

    pub(crate) fn put(
        &self,
        ns: Namespace<'_>,
        key: &[u8],
        value: &[u8],
    ) -> Result<(), BackendError> {
        self.txn.put(prefixed(ns, key), value).map_err(io)
    }

    /// [`LsmBackend::put_dup_stored`] in this transaction.
    pub(crate) fn put_dup_stored(
        &self,
        ns: Namespace<'_>,
        key: &[u8],
        value: &[u8],
        stored: &[u8],
    ) -> Result<(), BackendError> {
        self.txn
            .put(dup_composite(ns, key, value), stored)
            .map_err(io)
    }

    /// Commit `write`'s puts and deletes in this transaction (see
    /// [`Self::commit`]). Its reads were not tracked: mark what they depend on.
    pub(crate) fn commit_write(self, mut write: LsmWrite) -> Result<bool, BackendError> {
        if write.merged {
            return Err(BackendError::Unsupported(
                "checked transactions cannot carry merges".to_string(),
            ));
        }
        for (key, value) in &write.pending {
            match value {
                Some(value) => self.txn.put(key, value),
                None => self.txn.delete(key),
            }
            .map_err(io)?;
        }
        let hooks = std::mem::take(&mut write.post_txn);
        let committed = self.commit();
        run_post_txn(hooks, matches!(committed, Ok(true)));
        committed
    }

    /// Commit durably. `Ok(false)`: refused for a conflicting commit, nothing
    /// written. Other failures (CAS, fencing) are errors.
    pub(crate) fn commit(self) -> Result<bool, BackendError> {
        let Self { backend, txn } = self;
        let db = backend.db();
        let committed = block_on_lsm_mapped(
            &backend.rt,
            async {
                match backend.refresh_if_stalled(&db, txn.commit()).await {
                    Ok(_) => Ok(true),
                    Err(error) if error.kind() == slatedb::ErrorKind::Transaction => Ok(false),
                    Err(error) => Err(error),
                }
            },
            map_slatedb_err,
        );
        match committed {
            Ok(true) => {
                backend.idle_flush.note_write();
                backend.publish_durable_commit();
                Ok(true)
            }
            Err(error) if LsmBackend::is_fenced_conflict(&error) => {
                Err(lsm_fenced_error("checked_txn", error.to_string()))
            }
            other => other,
        }
    }
}

impl LsmBackend {
    /// Open SlateDB on AWS S3. `prefix` is the key prefix (a path within the
    /// bucket); `bucket` is the S3 bucket. Credentials and region resolve from
    /// the standard AWS environment (env vars, shared config, or the
    /// instance/IRSA role); `region` overrides when provided. SlateDB's manifest
    /// CAS uses S3 conditional writes (on by default in object_store's AWS
    /// client), so no external lock table is required.
    pub fn open_s3(bucket: &str, prefix: &str, region: Option<&str>) -> Result<Self, BackendError> {
        Self::open_s3_with_options(bucket, prefix, region, None, false)
    }

    /// Open SlateDB on an S3-compatible endpoint. Used for MinIO/local proof
    /// runs and real S3 alike; `endpoint=None` leaves object_store on AWS S3.
    pub fn open_s3_with_options(
        bucket: &str,
        prefix: &str,
        region: Option<&str>,
        endpoint: Option<&str>,
        allow_http: bool,
    ) -> Result<Self, BackendError> {
        let store = build_s3_store(&S3StoreConfig {
            bucket,
            region,
            endpoint,
            allow_http,
        })?;
        Self::open_with_store(prefix, store)
    }

    /// Purge a collection prefix without opening SlateDB.
    pub(crate) fn purge_s3_prefix_with_options(
        bucket: &str,
        prefix: &str,
        region: Option<&str>,
        endpoint: Option<&str>,
        allow_http: bool,
    ) -> Result<(), BackendError> {
        let config = S3StoreConfig {
            bucket,
            region,
            endpoint,
            allow_http,
        };
        let store = build_s3_store(&config)?;
        purge_prefix_from_store(shared_lsm_handle(), store, prefix)
    }

    pub(crate) fn prefixed_bound(prefix: &[u8], b: &Bound<Vec<u8>>) -> Bound<Vec<u8>> {
        match b {
            Bound::Included(v) => Bound::Included([prefix, v.as_slice()].concat()),
            Bound::Excluded(v) => Bound::Excluded([prefix, v.as_slice()].concat()),
            Bound::Unbounded => Bound::Unbounded, // caller fixes up per start/end
        }
    }

    /// Dup-namespace scan visiting `(logical_key, logical_value, stored)`,
    /// where `stored` is the row's own SlateDB value (empty unless written by
    /// [`Self::put_dup_stored`]).
    fn scan_dup_namespace_with_options(
        &self,
        r: &LsmRead,
        ns: Namespace<'_>,
        range: KeyRange,
        scan_options: ScanOptions,
        mut visit: impl FnMut(&[u8], &[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        let prefix = ns_prefix(ns);
        let plan = dup_scan_plan(ns, &range);
        let open_iter = || async {
            match &plan.scan_prefix {
                Some(scan_prefix) => {
                    r.snapshot
                        .scan_prefix_with_options(scan_prefix.clone(), .., &scan_options)
                        .await
                }
                None => {
                    r.snapshot
                        .scan_with_options((plan.lo.clone(), plan.hi.clone()), &scan_options)
                        .await
                }
            }
            .map_err(io)
        };

        if r.pending.is_empty() {
            block_on_lsm_read_mapped(
                &self.rt,
                async {
                    let mut rows = DupScanCursor::new(
                        open_iter().await?,
                        &prefix,
                        &range,
                        plan.seek_prefix.as_deref(),
                        io,
                    );
                    while let Some(kv) = rows.next().await? {
                        let (logical_key, logical_value) =
                            match decode_dup_full_key(kv.key.as_ref(), &prefix)? {
                                Some(parts) => parts,
                                None => continue,
                            };
                        if !visit(logical_key, logical_value, kv.value.as_ref()) {
                            break;
                        }
                    }
                    self.note_dup_scan_rows(rows.visited());
                    Ok::<(), BackendError>(())
                },
                |e| e,
            )?;
            return Ok(());
        }

        let mut pending_rows: BTreeMap<Vec<u8>, (Vec<u8>, Vec<u8>, Bytes)> = BTreeMap::new();
        for (full_key, pending) in &r.pending {
            let Some(stored) = pending else {
                continue;
            };
            if !full_key.starts_with(&prefix) {
                continue;
            }
            let Some((logical_key, logical_value)) = decode_dup_full_key(full_key, &prefix)? else {
                continue;
            };
            if !logical_key_in_range(logical_key, &range) {
                continue;
            }
            pending_rows.insert(
                full_key.clone(),
                (logical_key.to_vec(), logical_value.to_vec(), stored.clone()),
            );
        }

        block_on_lsm_read_mapped(
            &self.rt,
            async {
                let mut pending_iter = pending_rows.into_iter().peekable();
                let mut rows = DupScanCursor::new(
                    open_iter().await?,
                    &prefix,
                    &range,
                    plan.seek_prefix.as_deref(),
                    io,
                );
                while let Some(kv) = rows.next().await? {
                    let full_key = kv.key.as_ref();
                    while let Some((pending_key, _pending_value)) = pending_iter.peek() {
                        if pending_key.as_slice() >= full_key {
                            break;
                        }
                        let Some((_pending_key, (logical_key, logical_value, stored))) =
                            pending_iter.next()
                        else {
                            break;
                        };
                        if !visit(&logical_key, &logical_value, &stored) {
                            self.note_dup_scan_rows(rows.visited());
                            return Ok(());
                        }
                    }
                    if r.pending.contains_key(full_key) {
                        continue;
                    }
                    let Some((logical_key, logical_value)) =
                        decode_dup_full_key(full_key, &prefix)?
                    else {
                        continue;
                    };
                    if !visit(logical_key, logical_value, kv.value.as_ref()) {
                        self.note_dup_scan_rows(rows.visited());
                        return Ok(());
                    }
                }
                self.note_dup_scan_rows(rows.visited());
                for (_full_key, (logical_key, logical_value, stored)) in pending_iter {
                    if !visit(&logical_key, &logical_value, &stored) {
                        break;
                    }
                }
                Ok::<(), BackendError>(())
            },
            |e| e,
        )?;
        Ok(())
    }

    fn scan_dup_namespace(
        &self,
        r: &LsmRead,
        ns: Namespace<'_>,
        range: KeyRange,
        mut visit: impl FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        self.scan_dup_namespace_with_options(r, ns, range, lsm_scan_options(), |k, v, _| {
            visit(k, v)
        })
    }

    /// `put_dup` whose row carries `stored` as its SlateDB value. Readers that
    /// ignore it (`scan`, `for_each_dup`, older binaries) see a plain dup row;
    /// rewriting the row with `put_dup` resets it to empty.
    pub(crate) fn put_dup_stored(
        &self,
        txn: &mut LsmWrite,
        ns: Namespace<'_>,
        key: &[u8],
        value: &[u8],
        stored: &[u8],
    ) -> Result<(), BackendError> {
        let ck = dup_composite(ns, key, value);
        let stored = Bytes::copy_from_slice(stored);
        txn.batch.put(ck.as_slice(), stored.clone());
        txn.pending.insert(ck, Some(stored));
        txn.has_writes = true;
        Ok(())
    }

    /// [`Self::scan_dup_namespace`] that also yields each row's stored value.
    pub(crate) fn scan_dup_stored(
        &self,
        r: &LsmRead,
        ns: Namespace<'_>,
        range: KeyRange,
        visit: impl FnMut(&[u8], &[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        self.scan_dup_namespace_with_options(r, ns, range, lsm_scan_options(), visit)
    }

    /// [`StorageBackend::scan`] of a range that holds a few small rows: it
    /// reads a block at a time, where a scan reads ahead of itself in every
    /// table for the rows it expects to follow.
    pub(crate) fn scan_few(
        &self,
        txn: &LsmRead,
        ns: Namespace<'_>,
        range: KeyRange,
        visit: impl FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        let options = lsm_scan_options()
            .with_read_ahead_bytes(1)
            .with_max_fetch_tasks(1);
        self.scan_with_lsm_options(txn, ns, range, options, visit)
    }

    pub(crate) fn scan_streaming(
        &self,
        txn: &LsmRead,
        ns: Namespace<'_>,
        range: KeyRange,
        visit: impl FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        self.scan_with_lsm_options(txn, ns, range, lsm_streaming_scan_options(), visit)
    }

    fn scan_with_lsm_options(
        &self,
        txn: &LsmRead,
        ns: Namespace<'_>,
        range: KeyRange,
        scan_options: ScanOptions,
        mut visit: impl FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        if is_dup(ns) {
            return self.scan_dup_namespace_with_options(
                txn,
                ns,
                range,
                scan_options,
                |k, v, _| visit(k, v),
            );
        }

        let prefix = ns_prefix(ns);
        let lo: Bound<Vec<u8>> = match &range.start {
            Bound::Unbounded => Bound::Included(prefix.clone()),
            other => Self::prefixed_bound(&prefix, other),
        };
        let hi: Bound<Vec<u8>> = match &range.end {
            Bound::Unbounded => match next_prefix(&prefix) {
                Some(end) => Bound::Excluded(end),
                None => Bound::Unbounded,
            },
            other => Self::prefixed_bound(&prefix, other),
        };
        let scan_prefix = scan_prefix_for_namespace_range(&prefix, &range);

        if txn.pending.is_empty() {
            block_on_lsm_read_mapped(
                &self.rt,
                async {
                    let mut iter = if let Some(scan_prefix) = scan_prefix {
                        txn.snapshot
                            .scan_prefix_with_options(scan_prefix, .., &scan_options)
                            .await
                    } else {
                        txn.snapshot
                            .scan_with_options((lo, hi), &scan_options)
                            .await
                    }
                    .map_err(io)?;
                    while let Some(kv) = iter.next().await.map_err(io)? {
                        self.bump_reads(1);
                        let k = kv.key.as_ref();
                        let stripped = if k.len() >= prefix.len() {
                            &k[prefix.len()..]
                        } else {
                            k
                        };
                        if !visit(stripped, kv.value.as_ref()) {
                            break;
                        }
                    }
                    Ok::<(), BackendError>(())
                },
                |e| e,
            )?;
            return Ok(());
        }

        let mut pending_rows: BTreeMap<Vec<u8>, Bytes> = BTreeMap::new();
        for (full_key, pending) in &txn.pending {
            let Some(value) = pending.as_ref() else {
                continue;
            };
            if !full_key.starts_with(&prefix) {
                continue;
            }
            let logical_key = &full_key[prefix.len()..];
            if !logical_key_in_range(logical_key, &range) {
                continue;
            }
            pending_rows.insert(full_key.clone(), value.clone());
        }

        block_on_lsm_read_mapped(
            &self.rt,
            async {
                let mut pending_iter = pending_rows.into_iter().peekable();
                let mut iter = if let Some(scan_prefix) = scan_prefix {
                    txn.snapshot
                        .scan_prefix_with_options(scan_prefix, .., &scan_options)
                        .await
                } else {
                    txn.snapshot
                        .scan_with_options((lo, hi), &scan_options)
                        .await
                }
                .map_err(io)?;
                while let Some(kv) = iter.next().await.map_err(io)? {
                    self.bump_reads(1);
                    let full_key = kv.key.as_ref();
                    while let Some((pending_key, pending_value)) = pending_iter.peek() {
                        if pending_key.as_slice() >= full_key {
                            break;
                        }
                        if !visit(&pending_key[prefix.len()..], pending_value.as_ref()) {
                            return Ok(());
                        }
                        pending_iter.next();
                    }
                    if txn.pending.contains_key(full_key) {
                        continue;
                    }
                    let stripped = if full_key.len() >= prefix.len() {
                        &full_key[prefix.len()..]
                    } else {
                        full_key
                    };
                    if !visit(stripped, kv.value.as_ref()) {
                        return Ok(());
                    }
                }
                for (full_key, value) in pending_iter {
                    if !visit(&full_key[prefix.len()..], value.as_ref()) {
                        break;
                    }
                }
                Ok::<(), BackendError>(())
            },
            |e| e,
        )?;
        Ok(())
    }

    // ── Collection-destroy helpers ────────────────────────────────────────────

    /// Quiesce SlateDB: flush the WAL and stop the background compactor/GC.
    ///
    /// Must be called BEFORE [`purge_prefix`] so the compactor cannot
    /// re-write objects we are about to delete. `Db::close` is idempotent
    /// after the first call; any holder's next op will receive a `Closed`
    /// error (expected — the collection is being dropped).
    pub(crate) fn close(&self) -> Result<(), BackendError> {
        let db = self.db();
        block_on_lsm(&self.rt, db.close())
    }

    pub(crate) fn close_reason(&self) -> Option<CloseReason> {
        self.db().status().close_reason
    }

    /// Recursively delete every object under this collection's SlateDB
    /// prefix from the object store (e.g. `shadow/<name>/wal/`, `compacted/`,
    /// manifest, …). Uses the retained `store` handle so it is independent of
    /// the `Db` lifecycle.
    ///
    /// Also removes this collection's SSD object-store cache subtree when
    /// `HELIX_LSM_CACHE_DIR` is set. The cache root is namespaced per collection
    /// (see [`cache_options_from_env`]), so the whole subtree is at
    /// `<HELIX_LSM_CACHE_DIR>/<sanitized self.path>/`.
    pub(crate) fn purge_prefix(&self) -> Result<(), BackendError> {
        purge_prefix_from_store(self.rt.clone(), self.store.clone(), &self.path)
    }

    /// Drop-time destroy: quiesce SlateDB, then purge the S3 prefix and
    /// the local SSD cache subtree. Called by [`AnyBackend::destroy_lsm`]
    /// during collection drop on the LSM path.
    pub(crate) fn destroy(&self) -> Result<(), BackendError> {
        // Best-effort close: stop the compactor/GC so it cannot race with the
        // subsequent object-store delete. We proceed to purge regardless of the
        // close result — a partially-quiesced engine is still better than
        // leaving the prefix orphaned.
        let _ = self.close();
        self.purge_prefix()
    }

    /// List the sibling collection names in the object store: the immediate
    /// children of this collection's root prefix (the parent of `self.path`).
    /// Lets a loaded writer enumerate the object-store catalog (collections that
    /// exist in S3 but may not be in the local registry) using its retained
    /// `store` handle.
    pub fn list_collection_prefixes(&self) -> Result<Vec<String>, BackendError> {
        let root = collections_root_prefix(&self.path);
        list_child_collection_names(&self.rt, &self.store, root)
    }

    /// Collection names with a live drop tombstone under this writer's root.
    pub(crate) fn list_drop_tombstones(&self) -> Result<HashSet<String>, BackendError> {
        let root = collections_root_prefix(&self.path);
        list_drop_tombstones(&self.rt, &self.store, root)
    }

    pub fn storage_bytes(&self) -> Result<u64, BackendError> {
        let store = Arc::clone(&self.store);
        let prefix = ObjPath::from(self.path.trim_matches('/'));
        block_on_lsm_read(&self.rt, async move {
            let mut entries = store.list(Some(&prefix));
            let mut total = 0_u64;
            while let Some(entry) = entries.try_next().await.map_err(io)? {
                total = total.saturating_add(entry.size as u64);
            }
            Ok::<u64, BackendError>(total)
        })
    }

    pub(crate) fn persist_metadata_sidecar(&self, bytes: Vec<u8>) -> Result<(), BackendError> {
        let path = metadata_sidecar_object_path(&self.path);
        let store = Arc::clone(&self.store);
        block_on_lsm(&self.rt, async move {
            store.put(&path, bytes.into()).await.map_err(io)?;
            Ok::<(), BackendError>(())
        })
    }
}

/// Highest SlateDB manifest id each purged prefix held, by prefix. A recreate
/// of the same name in this process starts its manifest ids above it
/// ([`DbBuilder::with_manifest_id_floor`]), so a reader's late lease append
/// from the dropped incarnation — which carries that incarnation's `N + 1` —
/// lands at or below the new database's first id and is never its latest.
/// Process-local: after a writer restart the window is long gone.
static LSM_PURGED_MANIFEST_CEILING: std::sync::LazyLock<std::sync::Mutex<HashMap<String, u64>>> =
    std::sync::LazyLock::new(Default::default);

/// Manifest id of `manifest/<id>.manifest` objects; `None` for anything else.
fn manifest_id_of(location: &ObjPath) -> Option<u64> {
    let mut parts = location.parts().collect::<Vec<_>>();
    let file = parts.pop()?;
    (parts.last()?.as_ref() == "manifest")
        .then(|| file.as_ref().strip_suffix(".manifest")?.parse().ok())
        .flatten()
}

fn record_purged_manifest_ceiling(path: &str, max_manifest_id: u64) {
    if max_manifest_id == 0 {
        return;
    }
    if let Ok(mut ceilings) = LSM_PURGED_MANIFEST_CEILING.lock() {
        let ceiling = ceilings
            .entry(path.trim_matches('/').to_string())
            .or_default();
        *ceiling = (*ceiling).max(max_manifest_id);
    }
}

pub(crate) fn purged_manifest_ceiling(path: &str) -> u64 {
    LSM_PURGED_MANIFEST_CEILING
        .lock()
        .ok()
        .and_then(|ceilings| ceilings.get(path.trim_matches('/')).copied())
        .unwrap_or(0)
}

/// Keys per delete request of a prefix purge: the most S3 takes in one
/// multi-object delete. The requests go one at a time, in listing order.
const PURGE_DELETE_BATCH: usize = 1_000;

pub(crate) fn purge_prefix_from_store(
    rt: Handle,
    store: Arc<dyn ObjectStore>,
    path: &str,
) -> Result<(), BackendError> {
    #[cfg(test)]
    if std::env::var("HELIX_LSM_TEST_FAIL_PURGE").is_ok() {
        return Err(BackendError::Io(
            "injected test object-store purge failure".to_string(),
        ));
    }
    let prefix = ObjPath::from(path);
    let started = std::time::Instant::now();
    // `manifest/` goes first so a racing opener gets 404 instead of a manifest
    // whose SSTs this pass is deleting. A reader that loaded the manifest before
    // that can still append one more version into the emptied prefix, so the
    // whole prefix is listed and swept twice; the drop tombstone's grace pass
    // (`collection_manager`) catches anything appended after the second sweep.
    let manifest_prefix = prefix.clone().join("manifest");
    let purged = block_on_lsm(&rt, async move {
        let mut purged = 0usize;
        for prefix in [&manifest_prefix, &prefix, &prefix] {
            let locations: Vec<ObjPath> = store
                .list(Some(prefix))
                .map_ok(|meta| meta.location)
                .try_collect()
                .await
                .map_err(|e| e.to_string())?;
            // Recorded before the deletes: a pass that fails midway must not
            // forget the ids it already removed, or the retry's recreate
            // (which finds no manifests) would start at 1.
            record_purged_manifest_ceiling(
                path,
                locations
                    .iter()
                    .filter_map(manifest_id_of)
                    .max()
                    .unwrap_or(0),
            );
            // One multi-object delete per batch, batches in listing order and one
            // at a time: a delete per key took about 40 ms on S3, so a drop of a
            // few thousand objects outlived its caller's timeout.
            for batch in locations.chunks(PURGE_DELETE_BATCH) {
                let keys = stream::iter(batch.to_vec())
                    .map(Ok::<ObjPath, ObjectStoreError>)
                    .boxed();
                store
                    .delete_stream(keys)
                    .try_for_each(|_| std::future::ready(Ok(())))
                    .await
                    .map_err(|e| e.to_string())?;
                purged += batch.len();
            }
        }
        Ok::<usize, String>(purged)
    })?;
    tracing::info!(
        path,
        objects = purged,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "LSM prefix purged from the object store"
    );

    if let Some(subtree) = cache_subtree_for_lsm_path(path) {
        if subtree.exists() {
            if let Err(e) = std::fs::remove_dir_all(&subtree) {
                tracing::warn!(
                    path,
                    cache_subtree = %subtree.display(),
                    error = %e,
                    "LSM cache subtree removal failed after S3 purge (non-fatal)"
                );
            }
        }
    }

    Ok(())
}

impl StorageBackend for LsmBackend {
    type Read<'s>
        = LsmRead
    where
        Self: 's;
    type Write<'s>
        = LsmWrite
    where
        Self: 's;

    fn begin_read(&self) -> Result<Self::Read<'_>, BackendError> {
        let db = self.db();
        let opened_at = std::time::Instant::now();
        let snapshot = match block_on_lsm_read_mapped(&self.rt, db.snapshot(), map_slatedb_err) {
            Ok(snapshot) => snapshot,
            Err(error) => return Err(error),
        };
        Ok(LsmRead::with_pending(snapshot, HashMap::new(), opened_at))
    }

    fn begin_write(&self) -> Result<Self::Write<'_>, BackendError> {
        Ok(LsmWrite {
            batch: WriteBatch::new(),
            pending: HashMap::new(),
            has_writes: false,
            merged: false,
            post_txn: Vec::new(),
        })
    }

    fn commit(&self, txn: Self::Write<'_>) -> Result<(), BackendError> {
        let (batch, hooks, has_writes) = txn.take_for_commit();
        if !has_writes {
            run_post_txn(hooks, true);
            return Ok(());
        }
        // `Db::write` returns a `WriteHandle` (durability tracker); we await the
        // batch durably and discard it. Classify fencing/CAS as Conflict.
        let db = self.db();
        let result = match block_on_lsm_mapped(
            &self.rt,
            self.refresh_if_stalled(&db, db.write(batch.clone())),
            map_slatedb_err,
        ) {
            Ok(_handle) => Ok(()),
            Err(error) if Self::is_fenced_conflict(&error) => {
                let message = lsm_conflict_message(&error).unwrap_or("").to_string();
                Err(lsm_fenced_error("commit", message))
            }
            Err(error) if Self::is_conflict(&error) => block_on_lsm_mapped(
                &self.rt,
                self.refresh_if_stalled(&db, db.write(batch)),
                map_slatedb_err,
            )
            .map(|_handle| ()),
            Err(error) => Err(error),
        };
        run_post_txn(hooks, result.is_ok());
        if result.is_ok() {
            self.idle_flush.note_write();
            self.publish_durable_commit();
        }
        result
    }

    /// Group-commit half: apply the batch to the memtable (visible to later
    /// reads) but DON'T wait for the WAL to reach the object store. `Db::write`
    /// with `await_durable: false` returns once the single-writer event loop has
    /// applied the batch (SlateDB awaits the apply, only skipping the durable
    /// wait — see `write_with_options`), so read-your-writes across buffered
    /// commits holds. Durability is provided later by [`flush_durable`].
    fn commit_buffered(&self, txn: Self::Write<'_>) -> Result<(), BackendError> {
        let (batch, hooks, has_writes) = txn.take_for_commit();
        if !has_writes {
            run_post_txn(hooks, true);
            return Ok(());
        }
        let db = self.db();
        let options = WriteOptions {
            await_durable: false,
            ..Default::default()
        };
        let result = match block_on_lsm_mapped(
            &self.rt,
            self.refresh_if_stalled(&db, db.write_with_options(batch.clone(), &options)),
            map_slatedb_err,
        ) {
            Ok(_handle) => Ok(()),
            Err(error) if Self::is_fenced_conflict(&error) => {
                let message = lsm_conflict_message(&error).unwrap_or("").to_string();
                Err(lsm_fenced_error("commit_buffered", message))
            }
            Err(error) if Self::is_conflict(&error) => {
                let options = WriteOptions {
                    await_durable: false,
                    ..Default::default()
                };
                block_on_lsm_mapped(
                    &self.rt,
                    self.refresh_if_stalled(&db, db.write_with_options(batch, &options)),
                    map_slatedb_err,
                )
                .map(|_handle| ())
            }
            Err(error) => Err(error),
        };
        run_post_txn(hooks, result.is_ok());
        if result.is_ok() {
            self.idle_flush.note_write();
            self.buffered_since_flush
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        result
    }

    /// Group-commit barrier: flush the WAL durably to the object store. One
    /// object-store flush makes ALL prior [`commit_buffered`] writes durable.
    fn flush_durable(&self) -> Result<(), BackendError> {
        let db = self.db();
        // Taken before the flush, not after it: a write that lands while this
        // flush is in flight is not part of it and sets the flag again, so the
        // flush that makes that write durable publishes it. Cleared afterwards,
        // that later flush found no flag and the readers were not told.
        let buffered = self
            .buffered_since_flush
            .swap(false, std::sync::atomic::Ordering::Relaxed);
        let flushed = block_on_lsm_mapped(&self.rt, db.flush(), map_slatedb_err);
        let result = if matches!(self.close_reason(), Some(CloseReason::Fenced)) {
            Err(lsm_fenced_error(
                "flush_durable",
                "SlateDB writer close reason is Fenced".to_string(),
            ))
        } else {
            match flushed {
                Err(error) if Self::is_fenced_conflict(&error) => {
                    let message = lsm_conflict_message(&error).unwrap_or("").to_string();
                    Err(lsm_fenced_error("flush_durable", message))
                }
                flushed => flushed,
            }
        };
        if buffered {
            match &result {
                // Publish only when buffered writes actually became durable —
                // a no-op barrier must not signal readers.
                Ok(()) => self.publish_durable_commit(),
                // Nothing was published: the flag is again as this call found
                // it, for the flush that does make those writes durable.
                Err(_) => self
                    .buffered_since_flush
                    .store(true, std::sync::atomic::Ordering::Relaxed),
            }
        }
        result
    }

    fn get_with<R>(
        &self,
        txn: &Self::Read<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        f: impl FnOnce(Option<&[u8]>) -> R,
    ) -> Result<R, BackendError> {
        let full = prefixed(ns, key);
        self.bump_reads(1);
        // Read-your-writes overlay first (empty for ordinary begin_read()):
        // Some(v) -> Some(&v); None -> buffered delete (key absent).
        if let Some(slot) = txn.pending.get(&full) {
            return Ok(f(slot.as_deref()));
        }
        let val = block_on_lsm_read(&self.rt, txn.snapshot.get(full.as_slice()))?;
        Ok(f(val.as_deref()))
    }

    fn scan(
        &self,
        txn: &Self::Read<'_>,
        ns: Namespace<'_>,
        range: KeyRange,
        visit: impl FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        self.scan_with_lsm_options(txn, ns, range, lsm_scan_options(), visit)
    }

    fn put(
        &self,
        txn: &mut Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        val: &[u8],
    ) -> Result<(), BackendError> {
        let full = prefixed(ns, key);
        let val = Bytes::copy_from_slice(val);
        txn.batch.put(full.as_slice(), val.clone());
        txn.pending.insert(full, Some(val));
        txn.has_writes = true;
        Ok(())
    }

    fn delete(
        &self,
        txn: &mut Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
    ) -> Result<(), BackendError> {
        let full = prefixed(ns, key);
        txn.batch.delete(full.as_slice());
        txn.pending.insert(full, None);
        txn.has_writes = true;
        Ok(())
    }

    fn merge(
        &self,
        txn: &mut Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        val: &[u8],
    ) -> Result<(), BackendError> {
        let full = prefixed(ns, key);
        txn.batch.merge(full.as_slice(), val);
        txn.has_writes = true;
        txn.merged = true;
        Ok(())
    }

    fn put_dup(
        &self,
        txn: &mut Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        value: &[u8],
    ) -> Result<(), BackendError> {
        // Multi-value modeled as composite-key -> empty.
        self.put_dup_stored(txn, ns, key, value, &[])
    }

    fn delete_dup(
        &self,
        txn: &mut Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        value: &[u8],
    ) -> Result<(), BackendError> {
        let ck = dup_composite(ns, key, value);
        txn.batch.delete(ck.as_slice());
        txn.pending.insert(ck, None);
        txn.has_writes = true;
        Ok(())
    }

    fn for_each_dup(
        &self,
        txn: &Self::Read<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        visit: impl FnMut(&[u8]) -> bool,
    ) -> Result<(), BackendError> {
        self.for_each_dup_rows(txn, ns, key, visit)
    }

    fn get_for_update<R>(
        &self,
        txn: &Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        f: impl FnOnce(Option<&[u8]>) -> R,
    ) -> Result<R, BackendError> {
        let full = prefixed(ns, key);
        // Read-your-writes: consult the batch's buffered effects first.
        if let Some(buffered) = txn.pending.get(&full) {
            return Ok(f(buffered.as_deref()));
        }
        let db = self.db();
        let val = match block_on_lsm_mapped(&self.rt, db.get(full.as_slice()), map_slatedb_err) {
            Ok(val) => val,
            Err(error) => return Err(error),
        };
        Ok(f(val.as_deref()))
    }
}

#[cfg(test)]
mod tests {
    mod closed_gc;
    mod fenced_write;
    mod flush_gate;

    use super::*;
    use crate::helix_engine::storage_core::backend::{KeyRange, Namespace};
    use serial_test::serial;

    /// SlateDB folds a write batch's (or a memtable's) merge operands with no
    /// base value and stores the fold as another operand, so the fold must keep
    /// its sign; only a real base value floors at zero.
    #[test]
    fn counter_merge_operator_keeps_signed_operand_folds_and_floors_only_on_a_base() {
        let op = HelixCounterMergeOperator;
        let key = Bytes::from_static(b"ctr");
        let delta = |d: i64| Bytes::copy_from_slice(&encode_lsm_counter_delta(d));
        let value = |v: u64| Bytes::copy_from_slice(&encode_lsm_counter_value(v));

        let mixed = op.merge_batch(&key, None, &[delta(-1), delta(2)]).unwrap();
        assert_eq!(decode_lsm_counter_delta(&mixed).unwrap(), 1);
        let lone = op.merge_batch(&key, None, &[delta(-1)]).unwrap();
        assert_eq!(decode_lsm_counter_delta(&lone).unwrap(), -1);
        assert_eq!(
            op.merge_batch(&key, Some(value(106)), &[mixed, lone.clone(), lone])
                .unwrap(),
            value(105)
        );
        assert_eq!(
            op.merge_batch(&key, Some(value(3)), &[delta(-5)]).unwrap(),
            value(0)
        );
        assert_eq!(
            op.merge(&key, Some(value(10)), delta(-4)).unwrap(),
            value(6)
        );
        // A negative fold that ends up stored as a value reads as zero.
        let negative = op.merge_batch(&key, None, &[delta(-3)]).unwrap();
        assert_eq!(decode_lsm_counter_value(&negative).unwrap(), 0);

        // On-disk compatibility: the pre-fix operator decoded every operand as
        // an i64 delta and every base as a u64, so it applies a new signed fold
        // correctly, and the non-negative folds it stored are valid deltas.
        let pre_fix_merge = |base: u64, operand: &Bytes| {
            apply_lsm_counter_delta(base, decode_lsm_counter_delta(operand).unwrap()).unwrap()
        };
        let fold = op.merge_batch(&key, None, &[delta(-4), delta(1)]).unwrap();
        assert_eq!(pre_fix_merge(106, &fold), 103);
        assert_eq!(encode_lsm_counter_value(7), encode_lsm_counter_delta(7));
    }

    #[test]
    fn lsm_counter_merges_apply_negative_and_mixed_sign_deltas_exactly() {
        let be = LsmBackend::open_in_memory("counter-merge-signs").unwrap();
        let key = b"ctr";
        let read = || {
            let r = be.begin_read().unwrap();
            be.get_with(&r, Namespace::Metadata, key, |v| {
                v.map(|b| decode_lsm_counter_value(b).unwrap())
            })
            .unwrap()
        };
        let merge = |deltas: &[i64]| {
            let mut w = be.begin_write().unwrap();
            for d in deltas {
                be.merge(
                    &mut w,
                    Namespace::Metadata,
                    key,
                    &encode_lsm_counter_delta(*d),
                )
                .unwrap();
            }
            be.commit(w).unwrap();
        };
        // Flushing moves folds into SSTs, below later operands and above the base.
        let flush_memtable = || {
            block_on_lsm(
                &be.rt,
                be.db().flush_with_options(slatedb::config::FlushOptions {
                    flush_type: slatedb::config::FlushType::MemTable,
                }),
            )
            .unwrap()
        };

        let mut w = be.begin_write().unwrap();
        be.put(
            &mut w,
            Namespace::Metadata,
            key,
            &encode_lsm_counter_value(105),
        )
        .unwrap();
        be.commit(w).unwrap();
        flush_memtable();

        merge(&[-1]);
        assert_eq!(read(), Some(104), "a lone negative delta");
        merge(&[-1, 2]);
        assert_eq!(read(), Some(105), "a mixed-sign batch");
        flush_memtable();
        assert_eq!(read(), Some(105), "after the operands were flushed");
        merge(&[-3, 1]);
        flush_memtable();
        merge(&[-1]);
        assert_eq!(
            read(),
            Some(102),
            "operands spread across SSTs and memtable"
        );
        merge(&[-500]);
        assert_eq!(read(), Some(0), "a real base still floors at zero");
    }

    #[test]
    fn compare_patch_nodes_rechecks_stable_ids_and_rolls_back_the_batch() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            mpsc,
        };
        use std::time::Duration;
        let backend = Arc::new(LsmBackend::open_in_memory("node-cas-race").unwrap());
        let mut write = backend.begin_write().unwrap();
        for id in [10_u128, 11] {
            backend
                .put(&mut write, Namespace::Nodes, &id.to_be_bytes(), b"H1")
                .unwrap();
        }
        backend.commit(write).unwrap();
        let (read_tx, read_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let attempts = Arc::new(AtomicUsize::new(0));
        let worker = {
            let backend = backend.clone();
            let attempts = attempts.clone();
            std::thread::spawn(move || {
                backend.compare_patch_nodes(&[10, 11], |id, bytes| {
                    if id == 11 && attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                        read_tx.send(()).unwrap();
                        resume_rx.recv_timeout(Duration::from_secs(10)).unwrap();
                    }
                    Ok((bytes == b"H1").then(|| b"H1-correct-size".to_vec()))
                })
            })
        };
        read_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        // An ordinary WriteBatch update keeps the chunk ID while changing its file snapshot.
        let mut write = backend.begin_write().unwrap();
        backend
            .put(
                &mut write,
                Namespace::Nodes,
                &11_u128.to_be_bytes(),
                b"H2-new-size-and-metadata",
            )
            .unwrap();
        backend.commit(write).unwrap();
        resume_tx.send(()).unwrap();
        assert!(!worker.join().unwrap().unwrap());
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            2,
            "OCC retries must re-run the guard against H2"
        );
        let read = backend.begin_read().unwrap();
        assert_eq!(
            get_owned(&backend, &read, Namespace::Nodes, &10_u128.to_be_bytes()),
            Some(b"H1".to_vec()),
            "first patch was rolled back"
        );
        assert_eq!(
            get_owned(&backend, &read, Namespace::Nodes, &11_u128.to_be_bytes()),
            Some(b"H2-new-size-and-metadata".to_vec())
        );
        assert!(!backend
            .compare_patch_nodes(&[10, 99], |_, _| Ok(Some(b"changed".to_vec())))
            .unwrap());
        let error = backend
            .compare_patch_nodes(&[10, 11], |id, _| {
                if id == 11 {
                    Err(BackendError::Io("injected patch failure".into()))
                } else {
                    Ok(Some(b"changed".to_vec()))
                }
            })
            .unwrap_err();
        assert!(error.to_string().contains("injected patch failure"));
        let read = backend.begin_read().unwrap();
        assert_eq!(
            get_owned(&backend, &read, Namespace::Nodes, &10_u128.to_be_bytes()),
            Some(b"H1".to_vec()),
            "missing targets and patch errors both roll back"
        );
    }

    fn get_owned(be: &LsmBackend, r: &LsmRead, ns: Namespace<'_>, k: &[u8]) -> Option<Vec<u8>> {
        be.get_with(r, ns, k, |v| v.map(|b| b.to_vec())).unwrap()
    }

    #[test]
    fn shared_db_cache_returns_process_singleton() {
        let first = shared_db_cache();
        let second = shared_db_cache();

        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    #[serial]
    fn meta_cache_shards_keep_each_shard_at_least_256_mib() {
        const MIB: usize = 1024 * 1024;
        let _unset = EnvGuard::remove(ENV_META_CACHE_SHARDS);
        // The old per-CPU split of the 128 MiB default made 8 MiB shards.
        assert_eq!(meta_cache_shards(128 * MIB, 16), 1);
        assert_eq!(meta_cache_shards(1024 * MIB, 16), 4);
        assert_eq!(
            meta_cache_shards(4096 * MIB, 16),
            16,
            "capped at the CPU split"
        );
        assert_eq!(meta_cache_shards(16 * MIB, 16), 1, "never zero shards");
        assert_eq!(meta_cache_shards(1024 * MIB, 0), 1);
        {
            let _restore = EnvGuard::set(ENV_META_CACHE_SHARDS, "16");
            assert_eq!(meta_cache_shards(128 * MIB, 16), 16, "kill switch");
        }
        let _zero = EnvGuard::set(ENV_META_CACHE_SHARDS, "0");
        assert_eq!(meta_cache_shards(1024 * MIB, 16), 4, "0 means derived");
    }

    /// The shared block cache's fill reaches `/metrics` as two gauges, of
    /// the block part alone: the shared cache's own snapshot adds the meta
    /// part's bytes to it.
    #[test]
    fn shared_block_cache_usage_is_exported_as_two_gauges_of_the_block_part() {
        let both_parts = match shared_db_cache().usage_snapshot().memory {
            CacheUsageSnapshot::Ready {
                capacity_bytes: Some(both_parts),
                ..
            } => both_parts as f64,
            other => panic!("unexpected usage snapshot {other:?}"),
        };
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, export_shared_block_cache_usage);
        let rendered = handle.render();
        let gauge = |name: &str| -> f64 {
            rendered
                .lines()
                .find_map(|line| line.strip_prefix(name)?.trim().parse().ok())
                .unwrap_or_else(|| panic!("no {name}in:\n{rendered}"))
        };
        let capacity = gauge("helix_lsm_block_cache_capacity_bytes ");
        assert!(
            capacity >= 1024.0 * 1024.0 && capacity < both_parts,
            "{capacity} of {both_parts}"
        );
        assert!(gauge("helix_lsm_block_cache_used_bytes ") >= 0.0);
    }

    #[test]
    fn block_on_lsm_rejects_unmarked_tokio_runtime_threads() {
        let lsm_rt = Runtime::new().unwrap();
        let outer_rt = Runtime::new().unwrap();

        let result = outer_rt
            .block_on(async { block_on_lsm(lsm_rt.handle(), async { Ok::<_, &'static str>(()) }) });

        assert!(
            matches!(result, Err(BackendError::Unsupported(ref message)) if message.contains("blocking thread")),
            "unmarked async runtime use should fail cleanly: {result:?}"
        );
    }

    #[test]
    fn block_on_lsm_allows_marked_spawn_blocking_threads() {
        let outer_rt = Runtime::new().unwrap();

        let result = outer_rt.block_on(async {
            tokio::task::spawn_blocking(|| {
                let lsm_rt = Runtime::new().unwrap();
                allow_lsm_blocking(|| {
                    block_on_lsm(lsm_rt.handle(), async { Ok::<_, &'static str>(()) })
                })
            })
            .await
            .expect("spawn_blocking task should not panic")
        });

        assert!(
            result.is_ok(),
            "marked spawn_blocking use should be accepted: {result:?}"
        );
    }

    #[test]
    fn block_on_lsm_allows_marked_tokio_runtime_threads() {
        let lsm_rt = Runtime::new().unwrap();
        let outer_rt = Runtime::new().unwrap();

        let result = outer_rt.block_on(async {
            allow_lsm_blocking(|| {
                block_on_lsm(lsm_rt.handle(), async { Ok::<_, &'static str>(()) })
            })
        });

        assert!(
            result.is_ok(),
            "marked runtime use should enter a blocking section: {result:?}"
        );
    }

    /// A helper thread that enters the request thread's context may read from
    /// inside a runtime (as a nested read does) and sees the request cancelled.
    #[test]
    fn lsm_thread_context_carries_blocking_and_cancellation_to_a_helper_thread() {
        let lsm_rt = Runtime::new().unwrap();
        let outer_rt = Runtime::new().unwrap();
        // (a read from inside a runtime is allowed, the request is still live)
        let on_helper = |context: &LsmThreadContext| {
            std::thread::scope(|scope| {
                scope
                    .spawn(|| {
                        context.enter(|| {
                            let nested_read = outer_rt.block_on(async {
                                block_on_lsm(lsm_rt.handle(), async { Ok::<_, &'static str>(()) })
                            });
                            (nested_read.is_ok(), ensure_lsm_read_not_cancelled().is_ok())
                        })
                    })
                    .join()
                    .unwrap()
            })
        };

        assert_eq!(on_helper(&LsmThreadContext::current()), (false, true));
        let unmarked_after = on_helper(&allow_lsm_blocking(LsmThreadContext::current));
        assert_eq!(unmarked_after, (true, true));

        let cancellation = LsmReadCancellation::new();
        let context =
            allow_lsm_blocking_cancellable(cancellation.clone(), LsmThreadContext::current);
        assert_eq!(on_helper(&context), (true, true));
        cancellation.cancel();
        assert_eq!(on_helper(&context), (true, false));
    }

    #[test]
    fn cancelled_read_bridge_drops_pending_async_read() {
        let cancellation = LsmReadCancellation::new();
        let worker_cancellation = cancellation.clone();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (result_tx, result_rx) = std::sync::mpsc::channel();

        let worker = std::thread::spawn(move || {
            let rt = shared_lsm_handle();
            let result = allow_lsm_blocking_cancellable(worker_cancellation, || {
                block_on_lsm_read(&rt, async move {
                    started_tx.send(()).unwrap();
                    futures::future::pending::<Result<(), &'static str>>().await
                })
            });
            result_tx.send(result).unwrap();
        });

        started_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("read future should start");
        cancellation.cancel();

        let result = result_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("cancellation should release the blocking read worker");
        assert!(
            matches!(result, Err(BackendError::Cancelled)),
            "cancelled read should return a clear backend error: {result:?}"
        );
        worker.join().unwrap();
    }

    #[test]
    fn writer_backend_read_list_get_and_scan_honor_cancellation() {
        let backend = LsmBackend::open_in_memory("writer-read-cancellation").unwrap();
        let read = backend.begin_read().unwrap();

        let cancellation = LsmReadCancellation::new();
        cancellation.cancel();
        let begin_read = allow_lsm_blocking_cancellable(cancellation, || backend.begin_read());
        assert!(matches!(begin_read, Err(BackendError::Cancelled)));

        let cancellation = LsmReadCancellation::new();
        cancellation.cancel();
        let get = allow_lsm_blocking_cancellable(cancellation, || {
            backend.get_with(&read, Namespace::Nodes, b"missing", |_| ())
        });
        assert!(matches!(get, Err(BackendError::Cancelled)));

        let cancellation = LsmReadCancellation::new();
        cancellation.cancel();
        let scan = allow_lsm_blocking_cancellable(cancellation, || {
            backend.scan(&read, Namespace::Nodes, KeyRange::all(), |_key, _value| {
                true
            })
        });
        assert!(matches!(scan, Err(BackendError::Cancelled)));

        let cancellation = LsmReadCancellation::new();
        cancellation.cancel();
        let list =
            allow_lsm_blocking_cancellable(cancellation, || backend.list_collection_prefixes());
        assert!(matches!(list, Err(BackendError::Cancelled)));
    }

    #[test]
    fn writer_open_rejects_cancellation_before_side_effecting_build() {
        let cancellation = LsmReadCancellation::new();
        cancellation.cancel();
        let mask = allow_lsm_blocking_cancellable(cancellation.clone(), || {
            mask_lsm_read_cancellation_for_cold_open()
        });
        assert!(
            matches!(mask, Err(BackendError::Cancelled)),
            "pre-cancelled writer-open mask must reject the request"
        );
        let open = allow_lsm_blocking_cancellable(cancellation, || {
            LsmBackend::open_with_store("writer-open-not-cancellable", Arc::new(InMemory::new()))
        });
        assert!(
            matches!(open, Err(BackendError::Cancelled)),
            "pre-cancelled writer open must stop before builder side effects"
        );
    }

    #[test]
    fn cold_open_mask_defers_inflight_cancellation_until_first_read() {
        let cancellation = LsmReadCancellation::new();
        let worker_cancellation = cancellation.clone();
        allow_lsm_blocking_cancellable(worker_cancellation, || {
            let mask = mask_lsm_read_cancellation_for_cold_open()
                .expect("live request should enter the writer-open mask");
            cancellation.cancel();
            ensure_lsm_read_not_cancelled()
                .expect("in-flight writer construction must defer cancellation");

            let backend = LsmBackend::open_with_store(
                "writer-open-mask-cancellation",
                Arc::new(InMemory::new()),
            )
            .expect("side-effecting writer build must complete under the mask");
            drop(mask);

            let read = backend.begin_read();
            assert!(
                matches!(read, Err(BackendError::Cancelled)),
                "the first read after writer publication must observe cancellation"
            );
            backend
                .close()
                .expect("writer close remains uncancellable after the read aborts");
        });
    }

    #[test]
    fn writer_commit_ignores_read_cancellation_context() {
        let backend = LsmBackend::open_in_memory("writer-commit-not-cancellable").unwrap();
        let mut write = backend.begin_write().unwrap();
        backend
            .put(&mut write, Namespace::Nodes, b"key", b"value")
            .unwrap();

        let cancellation = LsmReadCancellation::new();
        cancellation.cancel();
        allow_lsm_blocking_cancellable(cancellation, || backend.commit(write))
            .expect("durable commit must ignore read cancellation");

        let read = backend.begin_read().unwrap();
        assert_eq!(
            get_owned(&backend, &read, Namespace::Nodes, b"key"),
            Some(b"value".to_vec())
        );
    }

    /// Finding 4: the write-path classifier must distinguish a fenced/stale LSM
    /// writer (epoch-fencing / CAS conflict) from transient I/O. SlateDB renders
    /// fencing as "detected newer DB client" and CAS as "...conflict"; those map
    /// to `BackendError::Conflict` so the write layer can fail loudly / fail over,
    /// while a generic I/O message stays `BackendError::Io`.
    #[test]
    fn map_slatedb_err_classifies_fencing_and_cas_as_conflict() {
        // Epoch fencing (a newer writer took the manifest) -> Conflict.
        assert!(
            matches!(
                map_slatedb_err("Closed error: detected newer DB client"),
                BackendError::Conflict(_)
            ),
            "fencing must map to Conflict"
        );
        // CAS / write conflict -> Conflict.
        assert!(
            matches!(
                map_slatedb_err("Transaction error: transaction conflict"),
                BackendError::Conflict(_)
            ),
            "CAS write conflict must map to Conflict"
        );
        // Transient I/O is NOT a fencing/split-brain signal -> stays Io.
        assert!(
            matches!(map_slatedb_err("broken pipe"), BackendError::Io(_)),
            "transient I/O must stay Io"
        );
    }

    #[test]
    fn metadata_fence_fails_closed_while_normal_cas_retries() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::mpsc;

        let be = Arc::new(LsmBackend::open_in_memory("metadata-txn-retry").unwrap());
        let attempt_count = Arc::new(AtomicUsize::new(0));
        let (slow_read_tx, slow_read_rx) = mpsc::channel();
        let (release_slow_tx, release_slow_rx) = mpsc::channel();

        let slow_be = Arc::clone(&be);
        let slow_attempt_count = Arc::clone(&attempt_count);
        let slow_writer = std::thread::spawn(move || {
            slow_be.update_key_transactional(Namespace::Metadata, b"metadata-current", |value| {
                let attempt = slow_attempt_count.fetch_add(1, Ordering::SeqCst);
                if attempt == 0 {
                    assert!(value.is_none());
                    slow_read_tx.send(()).unwrap();
                    release_slow_rx.recv().unwrap();
                } else {
                    assert_eq!(value, Some(&b"contender"[..]));
                }
                Ok((b"slow-after-retry".to_vec(), attempt))
            })
        });

        slow_read_rx.recv().unwrap();
        be.update_key_transactional(Namespace::Metadata, b"metadata-current", |value| {
            assert!(value.is_none());
            Ok((b"contender".to_vec(), ()))
        })
        .unwrap();
        release_slow_tx.send(()).unwrap();

        let retry_attempt = slow_writer.join().unwrap().unwrap();
        assert_eq!(retry_attempt, 1);
        assert_eq!(attempt_count.load(Ordering::SeqCst), 2);

        let read = be.begin_read().unwrap();
        assert_eq!(
            get_owned(&be, &read, Namespace::Metadata, b"metadata-current"),
            Some(b"slow-after-retry".to_vec())
        );

        use slatedb::object_store::local::LocalFileSystem;

        let data_dir = tempfile::TempDir::new().unwrap();
        let store: Arc<dyn ObjectStore> =
            Arc::new(LocalFileSystem::new_with_prefix(data_dir.path()).unwrap());
        let path = "helion/metadata-fence-fails-closed";

        let be1 = LsmBackend::open_with_store(path, store.clone()).unwrap();
        be1.update_key_transactional(Namespace::Metadata, b"metadata-current", |value| {
            assert!(value.is_none());
            Ok((b"initial".to_vec(), ()))
        })
        .unwrap();

        let be2 = LsmBackend::open_with_store(path, store.clone()).unwrap();
        be2.update_key_transactional(Namespace::Metadata, b"metadata-current", |value| {
            assert_eq!(value, Some(&b"initial"[..]));
            Ok((b"fencer".to_vec(), ()))
        })
        .unwrap();

        be1.update_key_transactional(Namespace::Metadata, b"metadata-current", |_value| {
            Ok((b"after-fence".to_vec(), ()))
        })
        .expect_err("fenced metadata transaction must fail closed");
        be1.compare_patch_nodes(&[10], |_, _| Ok(Some(b"after-fence".to_vec())))
            .expect_err("fenced node compare-and-patch must also fail closed");

        let read = be2.begin_read().unwrap();
        assert_eq!(
            get_owned(&be2, &read, Namespace::Metadata, b"metadata-current"),
            Some(b"fencer".to_vec())
        );
    }

    #[test]
    fn empty_write_batch_commit_is_noop() {
        let be = LsmBackend::open_in_memory("empty-write-batch-commit-is-noop").unwrap();
        let w = be.begin_write().unwrap();

        be.commit(w).unwrap();
    }

    /// #3 shared runtime: opening many backends must NOT spawn many tokio
    /// runtimes. The shared runtime is a process singleton (one `OnceLock`), and
    /// every `open_*` pulls a `Handle` from it — so the runtime/thread count is
    /// bounded by node size, not by collection count.
    #[test]
    fn shared_lsm_runtime_is_singleton_across_backends() {
        let first = shared_lsm_runtime();
        assert!(
            std::ptr::eq(first, shared_lsm_runtime()),
            "shared LSM runtime must be a process singleton"
        );

        // Opening real backends must reuse the SAME runtime, never replace it.
        let _be1 = LsmBackend::open_in_memory("shared-rt-singleton-1").unwrap();
        let _be2 = LsmBackend::open_in_memory("shared-rt-singleton-2").unwrap();
        let _be3 = LsmBackend::open_in_memory("shared-rt-singleton-3").unwrap();
        assert!(
            std::ptr::eq(first, shared_lsm_runtime()),
            "opening backends must not create a second runtime"
        );
    }

    /// #1 group-commit: many `commit_buffered` calls are each immediately VISIBLE
    /// to a fresh read snapshot (memtable apply, not deferred), and one
    /// `flush_durable` barrier makes them all durable. This is the intra-apply
    /// "N buffered commits → 1 durable flush" contract the upsert path relies on.
    #[test]
    fn buffered_commits_are_visible_then_one_flush_durable_barrier() {
        let be = LsmBackend::open_in_memory("group-commit-buffered").unwrap();

        // Two independent buffered commits (durability deferred).
        let mut w1 = be.begin_write().unwrap();
        be.put(&mut w1, Namespace::Nodes, b"k1", b"v1").unwrap();
        be.commit_buffered(w1).unwrap();

        let mut w2 = be.begin_write().unwrap();
        be.put(&mut w2, Namespace::Nodes, b"k2", b"v2").unwrap();
        be.delete(&mut w2, Namespace::Nodes, b"k1").unwrap();
        be.commit_buffered(w2).unwrap();

        // Visible to a fresh snapshot BEFORE the durability barrier (read-your-
        // writes across buffered commits, just like across durable chunks).
        let r = be.begin_read().unwrap();
        assert_eq!(get_owned(&be, &r, Namespace::Nodes, b"k1"), None);
        assert_eq!(
            get_owned(&be, &r, Namespace::Nodes, b"k2").as_deref(),
            Some(&b"v2"[..])
        );

        // One barrier makes every prior buffered commit durable.
        be.flush_durable().unwrap();

        // Still consistent after the barrier.
        let r2 = be.begin_read().unwrap();
        assert_eq!(get_owned(&be, &r2, Namespace::Nodes, b"k1"), None);
        assert_eq!(
            get_owned(&be, &r2, Namespace::Nodes, b"k2").as_deref(),
            Some(&b"v2"[..])
        );

        // flush_durable on a backend with nothing newly buffered is a safe no-op.
        be.flush_durable().unwrap();
    }

    /// The wait behind [`LsmBackend::refresh_if_stalled`]: only a write that
    /// is still pending gets refreshes, one per interval until it completes,
    /// and a refresh that never returns does not hold the write.
    #[test]
    fn stalled_write_wait_refreshes_only_a_pending_write_and_never_holds_it() {
        let rt = shared_lsm_handle();
        let every = std::time::Duration::from_millis(5);

        // A write that completes at once is never refreshed.
        let refreshes = Cell::new(0usize);
        let out = rt.block_on(refresh_while_pending(async { 7u8 }, every, || {
            refreshes.set(refreshes.get() + 1);
            async {}
        }));
        assert_eq!((out, refreshes.get()), (7, 0));

        // Nor is one that completes before the interval has passed.
        let out = rt.block_on(refresh_while_pending(
            async {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                8u8
            },
            std::time::Duration::from_secs(3600),
            || {
                refreshes.set(refreshes.get() + 1);
                async {}
            },
        ));
        assert_eq!((out, refreshes.get()), (8, 0));

        // A parked write is refreshed every interval until it completes (the
        // third refresh releases this one).
        let refreshes = Cell::new(0usize);
        let (release, parked) = tokio::sync::oneshot::channel::<u8>();
        let mut release = Some(release);
        let out = rt.block_on(refresh_while_pending(parked, every, || {
            refreshes.set(refreshes.get() + 1);
            if refreshes.get() == 3 {
                if let Some(release) = release.take() {
                    let _ = release.send(9);
                }
            }
            async {}
        }));
        assert_eq!((out, refreshes.get()), (Ok(9), 3));

        // A refresh that never returns does not hold the write either.
        let (release, parked) = tokio::sync::oneshot::channel::<u8>();
        let mut release = Some(release);
        let out = rt.block_on(refresh_while_pending(parked, every, || {
            if let Some(release) = release.take() {
                let _ = release.send(4);
            }
            std::future::pending::<()>()
        }));
        assert_eq!(out, Ok(4));
    }

    /// The rule that holds however many commits are parked on a collection:
    /// one refresh at a time, and the next no sooner than an interval after
    /// the last one started.
    #[test]
    fn stalled_write_refresh_gate_admits_one_refresh_per_interval() {
        use std::time::Duration;
        let gate = RefreshGate::default();
        let hour = Duration::from_secs(3600);

        // While one refresh holds the permit, no other enters.
        let permit = gate.try_enter(hour).expect("the first refresh enters");
        assert!(gate.try_enter(Duration::ZERO).is_none());
        drop(permit);

        // Once it is done, the next enters only an interval after it started.
        assert!(gate.try_enter(hour).is_none());
        assert!(gate.try_enter(Duration::ZERO).is_some());
    }

    /// One durable commit of the single node row `key`.
    fn commit_row(be: &LsmBackend, key: u8) -> Result<(), BackendError> {
        let mut w = be.begin_write()?;
        be.put(
            &mut w,
            Namespace::Nodes,
            &u128::from(key).to_be_bytes(),
            b"v",
        )?;
        be.commit(w)
    }

    /// A writer whose next commit SlateDB parks. Every commit freezes its
    /// memtable into one L0 SST and four fill L0, so the fifth commit leaves
    /// a frozen memtable that cannot flush. Four L0 SSTs is also when the
    /// compactor empties L0 (within a fraction of a second here), which this
    /// writer only learns from a manifest refresh, and its own poll is an
    /// hour away.
    fn writer_with_full_l0(name: &str) -> Arc<LsmBackend> {
        let soon = std::time::Duration::from_millis(50);
        let mut settings = Settings::default();
        settings.manifest_poll_interval = std::time::Duration::from_secs(3600);
        settings.l0_sst_size_bytes = 1;
        settings.l0_max_ssts = 4;
        settings.max_unflushed_bytes = 1;
        if let Some(compactor) = settings.compactor_options.as_mut() {
            compactor.poll_interval = soon;
            compactor.commit_compacted_interval = soon;
            if let Some(worker) = compactor.worker.as_mut() {
                worker.compactions_poll_interval = soon;
            }
        }
        let be = Arc::new(
            LsmBackend::open_with_store_settings(name, Arc::new(InMemory::new()), Some(settings))
                .unwrap(),
        );
        for key in 1..=5 {
            commit_row(&be, key).unwrap();
        }
        assert_eq!(be.db().manifest().l0().len(), 4, "L0 is not full");
        be
    }

    /// A commit that parks on a [`writer_with_full_l0`]; `Ok(true)`: committed.
    type Commit = fn(&LsmBackend) -> Result<bool, BackendError>;

    /// What a parked commit sends once it returns: its result and the metrics
    /// it recorded, as Prometheus text.
    type Returned = std::sync::mpsc::Receiver<(Result<bool, BackendError>, String)>;

    /// Runs `commit` on its own thread, under its own metrics recorder.
    fn park(be: &Arc<LsmBackend>, commit: Commit) -> Returned {
        let (done, returned) = std::sync::mpsc::channel();
        let be = Arc::clone(be);
        std::thread::spawn(move || {
            let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
            let result = metrics::with_local_recorder(&recorder, || commit(&be));
            let _ = done.send((result, recorder.handle().render()));
        });
        returned
    }

    /// The `outcome` count of the stalled-write refresh counter in `rendered`.
    fn refresh_count(rendered: &str, outcome: &str) -> u64 {
        let series = format!("helix_lsm_stalled_write_refresh_total{{outcome=\"{outcome}\"}} ");
        rendered
            .lines()
            .find_map(|line| line.strip_prefix(series.as_str()))
            .map_or(0, |count| count.parse().expect("a counter value"))
    }

    /// One parked commit of each kind, each on its own [`writer_with_full_l0`]:
    /// a batch commit, a buffered commit, a metadata transaction, a node
    /// compare-and-patch and a checked transaction.
    fn park_each_kind_of_commit(tag: &str) -> [(&'static str, Arc<LsmBackend>, Returned); 5] {
        let commit: Commit = |be| commit_row(be, 6).map(|()| true);
        let commit_buffered: Commit = |be| {
            let mut w = be.begin_write()?;
            be.put(&mut w, Namespace::Nodes, &[6], b"v")?;
            be.commit_buffered(w).map(|()| true)
        };
        let transaction: Commit = |be| {
            be.update_key_transactional(Namespace::Metadata, b"k", |_| Ok((b"v".to_vec(), true)))
        };
        let compare_patch: Commit =
            |be| be.compare_patch_nodes(&[1], |_, _| Ok(Some(b"w".to_vec())));
        let checked_transaction: Commit = |be| {
            let txn = be.begin_checked()?;
            txn.put(Namespace::Nodes, &[6], b"v")?;
            txn.commit()
        };
        [
            ("commit", commit),
            ("commit_buffered", commit_buffered),
            ("transaction", transaction),
            ("compare_patch", compare_patch),
            ("checked_transaction", checked_transaction),
        ]
        .map(|(kind, commit)| {
            let be = writer_with_full_l0(&format!("stalled-{tag}-{kind}"));
            let returned = park(&be, commit);
            (kind, be, returned)
        })
    }

    /// L0 is full and the compactor's relief is in a manifest this writer has
    /// not read. Each kind of parked commit must return on its own refresh
    /// instead of waiting for the poll, and with the kill switch it must wait
    /// as before, without refreshing.
    #[test]
    #[serial]
    fn stalled_write_refreshes_the_manifest_instead_of_waiting_for_the_poll() {
        use std::sync::mpsc::TryRecvError;
        use std::time::{Duration, Instant};

        {
            // Given the refresh switched off and one parked commit of each kind.
            let _off = EnvGuard::set(ENV_STALLED_WRITE_REFRESH, "0");
            assert!(!lsm_stalled_write_refresh_enabled());
            let parked = park_each_kind_of_commit("poll-only");

            // Then each is still parked once the refresh interval has passed,
            std::thread::sleep(STALLED_WRITE_REFRESH_EVERY + Duration::from_secs(1));
            for (kind, be, returned) in parked {
                assert!(
                    matches!(returned.try_recv(), Err(TryRecvError::Empty)),
                    "with the refresh off a parked {kind} must wait for the manifest poll"
                );
                // and the refresh it would have issued is what releases it.
                let deadline = Instant::now() + Duration::from_secs(60);
                let (result, rendered) = loop {
                    block_on_lsm(&be.rt, be.db().refresh_manifest()).unwrap();
                    match returned.recv_timeout(Duration::from_millis(250)) {
                        Ok(sent) => break sent,
                        Err(_) => assert!(
                            Instant::now() < deadline,
                            "a manifest refresh did not release the parked {kind}"
                        ),
                    }
                };
                assert!(matches!(result, Ok(true)), "{kind}: {result:?}");
                assert!(
                    !rendered.contains("helix_lsm_stalled_write_refresh_total"),
                    "with the refresh off a parked {kind} must not count one: {rendered}"
                );
                be.close().unwrap();
            }
        }

        // Given the default: each parked commit refreshes the manifest itself,
        // finds the freed L0 slots and returns, with the poll still an hour away.
        let _on = EnvGuard::remove(ENV_STALLED_WRITE_REFRESH);
        assert!(lsm_stalled_write_refresh_enabled());
        for (kind, be, returned) in park_each_kind_of_commit("refresh") {
            let (result, rendered) = returned
                .recv_timeout(Duration::from_secs(60))
                .unwrap_or_else(|_| panic!("a parked {kind} must refresh the manifest and return"));
            assert!(matches!(result, Ok(true)), "{kind}: {result:?}");
            assert!(
                refresh_count(&rendered, "freed") >= 1,
                "a parked {kind} must count the refresh that released it: {rendered}"
            );
            be.close().unwrap();
        }
    }

    /// Commits parked on one writer share its refreshes: at most one per
    /// interval between them, and that one releases them all.
    #[test]
    #[serial]
    fn stalled_writes_on_one_writer_share_one_refresh_per_interval() {
        use std::time::{Duration, Instant};

        let _on = EnvGuard::remove(ENV_STALLED_WRITE_REFRESH);
        let be = writer_with_full_l0("stalled-shared");
        let commit: Commit = |be| commit_row(be, 6).map(|()| true);
        let parked_at = Instant::now();
        let parked = [(); 3].map(|()| park(&be, commit));

        let (mut sent, mut freed) = (0, 0);
        for returned in parked {
            let (result, rendered) = returned
                .recv_timeout(Duration::from_secs(60))
                .expect("a parked commit must return");
            assert!(matches!(result, Ok(true)), "{result:?}");
            let found_freed = refresh_count(&rendered, "freed");
            freed += found_freed;
            sent +=
                found_freed + refresh_count(&rendered, "ok") + refresh_count(&rendered, "error");
        }
        // A refresh starts no sooner than an interval after the commits parked
        // or after the one before it.
        let intervals = parked_at.elapsed().as_secs() / STALLED_WRITE_REFRESH_EVERY.as_secs();
        assert!(
            freed >= 1 && sent <= intervals,
            "{sent} refreshes ({freed} that found freed L0 slots) in {intervals} intervals"
        );
        be.close().unwrap();
    }

    /// Sets an env var for the duration of a test and restores the prior value
    /// (or removes it) on drop, so `#[serial]` cache tests don't leak env state.
    struct EnvGuard {
        key: &'static str,
        prev: Option<String>,
    }
    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self { key, prev }
        }

        fn remove(key: &'static str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::remove_var(key);
            Self { key, prev }
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }

    /// `reader_idle_after_from_env` (`HELIX_LSM_READER_IDLE_AFTER_MS`) treats
    /// `0` as a meaningful, valid value (the kill switch) rather than "unset"
    /// — unlike the other reader knobs, which filter out `0` and fall back to
    /// their default. Also covers the default (unset) and a custom value.
    #[test]
    #[serial]
    fn reader_idle_after_from_env_treats_zero_as_kill_switch() {
        std::env::remove_var(ENV_READER_READ_IDLE_AFTER_MS);
        assert_eq!(
            reader_idle_after_from_env(),
            std::time::Duration::from_millis(600_000),
            "unset must fall back to the 10-minute default"
        );

        {
            let _guard = EnvGuard::set(ENV_READER_READ_IDLE_AFTER_MS, "0");
            assert_eq!(
                reader_idle_after_from_env(),
                std::time::Duration::ZERO,
                "0 must disable read-gated tiering, not fall back to the default"
            );
        }

        {
            let _guard = EnvGuard::set(ENV_READER_READ_IDLE_AFTER_MS, "45000");
            assert_eq!(
                reader_idle_after_from_env(),
                std::time::Duration::from_millis(45_000)
            );
        }

        // Unparseable falls back to the default, same as the other knobs.
        {
            let _guard = EnvGuard::set(ENV_READER_READ_IDLE_AFTER_MS, "not-a-number");
            assert_eq!(
                reader_idle_after_from_env(),
                std::time::Duration::from_millis(600_000)
            );
        }
    }

    /// `reader_cold_open_starts_fast` must agree with `reader_options_from_env`
    /// (HIGH #2 from the poll-tiering review): `HelixGraphStorage` seeds its
    /// `reader_poll_tier_fast` belief from this predicate, so it must match the
    /// interval the cold open actually used. Cold opens start FAST whenever the
    /// read-gated sweeper can demote them later; only the sweeper kill switch
    /// (`HELIX_LSM_READER_IDLE_AFTER_MS=0`) with the feed on keeps idle opens.
    #[test]
    #[serial]
    fn reader_cold_open_starts_fast_matches_feed_url_configuration() {
        std::env::remove_var(ENV_WRITER_FEED_URL);
        std::env::remove_var(ENV_READER_READ_IDLE_AFTER_MS);
        assert!(
            reader_cold_open_starts_fast(),
            "no write-feed configured: cold opens serve at the fast/manifest-poll cadence"
        );
        {
            let _idle_after = EnvGuard::set(ENV_READER_READ_IDLE_AFTER_MS, "0");
            assert!(
                reader_cold_open_starts_fast(),
                "no write-feed and no sweeper: nothing would ever promote an idle open"
            );
        }

        let _guard = EnvGuard::set(ENV_WRITER_FEED_URL, "http://helix:6969");
        assert!(
            reader_cold_open_starts_fast(),
            "write-feed configured with the sweeper on: a cold open starts fast so a \
             write that lands while the open is in flight (skipped by the feed poller \
             as not-yet-resident) is picked up by the next fast poll, not 300s later"
        );
        let _idle_after = EnvGuard::set(ENV_READER_READ_IDLE_AFTER_MS, "0");
        assert!(
            !reader_cold_open_starts_fast(),
            "sweeper kill switch: nothing would ever demote, so cold opens stay idle"
        );
    }

    /// A reader renews its lease at the first poll past the lease's half-life,
    /// and a poll comes a little short of its tick's age, so the poll that
    /// falls on half-life renews nothing. Whatever the interval, the renewal
    /// must come a whole poll before the lease ends: the manifest stores the
    /// expiry rounded down to the second, and at twice the interval the
    /// renewal fell on the expiry itself.
    #[test]
    #[serial]
    fn reader_lease_is_renewed_a_whole_poll_before_it_ends() {
        let _idle = EnvGuard::remove(ENV_READER_IDLE_POLL_MS);
        let lease_secs = |poll_secs: u64| {
            let mut options = DbReaderOptions::default();
            options.manifest_poll_interval = std::time::Duration::from_secs(poll_secs);
            clamp_reader_checkpoint_lifetime(&mut options);
            options.checkpoint_lifetime.as_secs()
        };
        for in_place in ["0", "1"] {
            let _mode = EnvGuard::set(ENV_READER_TIER_IN_PLACE, in_place);
            for poll in [5, 199, 200, 201, 250, 299, 300, 301, 3_600] {
                let lease = lease_secs(poll);
                let polls_before_renewal = (lease / 2) / poll + 1;
                let renewed_at = polls_before_renewal * poll;
                assert!(
                    renewed_at + poll <= lease,
                    "in place {in_place}, poll {poll} s, lease {lease} s: renewed {renewed_at} s in"
                );
            }
        }
        {
            let _reopen = EnvGuard::set(ENV_READER_TIER_IN_PLACE, "0");
            assert_eq!(lease_secs(5), 600, "the serving tier keeps the default");
            assert_eq!(lease_secs(300), 900, "the idle tier");
        }
        // In place, the reader opened on the serving tier is the one that is
        // polled at the idle interval later, on the lease it was opened with.
        // Its demotion comes at any moment of that lease: at worst with the
        // last poll before half-life, and the renewal is one idle poll on.
        let _in_place = EnvGuard::set(ENV_READER_TIER_IN_PLACE, "1");
        let idle = reader_idle_poll_interval_from_env().as_secs();
        for opened_at in [5, 60, idle] {
            let lease = lease_secs(opened_at);
            let renewed_at = lease / 2 + idle;
            assert!(
                renewed_at + idle <= lease,
                "opened at {opened_at} s, lease {lease} s: demoted at half-life, renewed {renewed_at} s in"
            );
        }
        assert_eq!(lease_secs(5), 1_200, "four idle polls, on any tier");
        assert_eq!(lease_secs(300), 1_200);
        assert_eq!(
            lease_secs(3_600),
            10_800,
            "a longer poll still gets its own"
        );
    }

    /// The standalone read-gated demotion sweep's only gate is its own
    /// idle-after kill switch — it must run regardless of write-feed
    /// configuration (the MEDIUM finding from the poll-tiering review's final
    /// round: standing down entirely in feed mode left read-promoted,
    /// then-abandoned collections permanently on the fast tier, since the
    /// feed poller only demotes entries in its own write-driven active set).
    /// Per-collection coordination with the feed poller now happens via
    /// `HelixGraphStorage::write_promoted_recently` inside the sweep itself,
    /// not by disabling the whole sweeper.
    #[test]
    #[serial]
    fn reader_poll_tier_sweeper_enabled_only_gated_by_idle_after_kill_switch() {
        std::env::remove_var(ENV_WRITER_FEED_URL);

        assert!(
            reader_poll_tier_sweeper_enabled(std::time::Duration::from_millis(600_000)),
            "no feed configured and idle-after non-zero: sweeper runs"
        );
        assert!(
            !reader_poll_tier_sweeper_enabled(std::time::Duration::ZERO),
            "idle-after 0 is the sweeper's own kill switch"
        );

        {
            let _guard = EnvGuard::set(ENV_WRITER_FEED_URL, "http://helix:6969");
            assert!(
                reader_poll_tier_sweeper_enabled(std::time::Duration::from_millis(600_000)),
                "feed configured: sweeper must still run — it now coordinates \
                 per-collection via write-recency instead of standing down"
            );
            assert!(
                !reader_poll_tier_sweeper_enabled(std::time::Duration::ZERO),
                "idle-after 0 remains the kill switch even with the feed configured"
            );
        }
    }

    fn count_files(dir: &std::path::Path) -> usize {
        let mut n = 0;
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    n += count_files(&path);
                } else {
                    n += 1;
                }
            }
        }
        n
    }

    /// `HELIX_LSM_CACHE_DIR` wires SlateDB's local-FS object-store cache: opening
    /// via `open_with_store` builds `Settings` with
    /// `object_store_cache_options.root_folder` = that dir. Under the vendored
    /// fork's tag-based admission policy only COMPACTED-SST traffic is disk-
    /// cached (WAL and manifest parts are deliberately bypassed — they are
    /// replayed once and previously just churned the evictor), and
    /// `cache_options_from_env` sets `cache_puts = true`, so the memtable→L0
    /// flush performed by `close()` writes the new SST through the cache.
    /// After write → close (L0 flush) → reopen → read-back: (a) values are
    /// correct AND (b) the cache root contains files.
    #[test]
    #[serial]
    fn lsm_object_store_cache_populates_ssd_dir() {
        let cache_dir = tempfile::TempDir::new().unwrap();
        let _guard = EnvGuard::set(ENV_CACHE_DIR, cache_dir.path().to_str().unwrap());

        // FS cache wraps the in-memory store via `open_with_store` (NOT
        // `open_in_memory`, which is deliberately cache-less). The store is
        // shared so the reopened backend sees the closed backend's L0 SSTs.
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());

        {
            let be = LsmBackend::open_with_store("/helion-cache-test", store.clone())
                .expect("open SlateDB with FS object-store cache over InMemory");
            let mut w = be.begin_write().unwrap();
            be.put(&mut w, Namespace::Nodes, b"ck1", b"cv1").unwrap();
            be.put(&mut w, Namespace::Nodes, b"ck2", b"cv2").unwrap();
            be.put(&mut w, Namespace::Metadata, b"ck1", b"meta")
                .unwrap();
            be.commit(w).unwrap();
            // close() flushes memtables to L0; with cache_puts=true the L0
            // SST write lands in the FS cache on the way to the store.
            be.close().expect("close flushes memtables to L0");
        }

        let cached_files_after_flush = count_files(cache_dir.path());
        assert!(
            cached_files_after_flush > 0,
            "SlateDB object-store cache dir {} should contain compacted-SST cache files \
             after the close-time L0 flush (cache_puts=true), found none",
            cache_dir.path().display()
        );

        // (a) values read back correctly through a fresh open + snapshot.
        let be = LsmBackend::open_with_store("/helion-cache-test", store)
            .expect("reopen SlateDB with FS object-store cache");
        let r = be.begin_read().unwrap();
        assert_eq!(
            get_owned(&be, &r, Namespace::Nodes, b"ck1"),
            Some(b"cv1".to_vec())
        );
        assert_eq!(
            get_owned(&be, &r, Namespace::Nodes, b"ck2"),
            Some(b"cv2".to_vec())
        );
        assert_eq!(
            get_owned(&be, &r, Namespace::Metadata, b"ck1"),
            Some(b"meta".to_vec())
        );

        let mut keys = Vec::new();
        be.scan(&r, Namespace::Nodes, KeyRange::all(), |k, _v| {
            keys.push(k.to_vec());
            true
        })
        .unwrap();
        assert_eq!(keys, vec![b"ck1".to_vec(), b"ck2".to_vec()]);
    }

    /// `lsm_settings_from_env`: no cache dir -> `None` (bare open); a set dir ->
    /// `Settings` whose cache root is namespaced PER COLLECTION and whose byte
    /// cap is the TOTAL cap divided by the expected open-collection count.
    #[test]
    #[serial]
    fn lsm_settings_from_env_reflects_cache_knobs() {
        {
            let _dir = EnvGuard::set(ENV_CACHE_DIR, "");
            assert!(
                lsm_settings_from_env("helion/c").is_none(),
                "empty cache dir -> no cache"
            );
        }
        {
            let _dir = EnvGuard::set(ENV_CACHE_DIR, "/cache/slatedb");
            let _cap = EnvGuard::set(ENV_CACHE_MAX_BYTES, "1048576");
            let _memory_cap = EnvGuard::set(ENV_CACHE_MEMORY_MAX_PCT, "0");
            let _open = EnvGuard::set(ENV_MAX_OPEN_COLLECTIONS, "4");
            let s =
                lsm_settings_from_env("helion/my_coll").expect("cache dir set -> Some(Settings)");
            // Per-collection namespaced root (collection id sanitized to a single
            // filesystem component).
            assert_eq!(
                s.object_store_cache_options.root_folder,
                Some(std::path::PathBuf::from("/cache/slatedb/helion_my_coll"))
            );
            // Total cap divided across the expected open-collection count: 1048576/4.
            assert_eq!(
                s.object_store_cache_options.max_cache_size_bytes,
                Some(262_144)
            );
        }
    }

    #[test]
    #[serial]
    fn lsm_settings_from_env_reflects_manifest_poll_knob() {
        let _cache = EnvGuard::set(ENV_CACHE_DIR, "");
        let _poll = EnvGuard::set(ENV_MANIFEST_POLL_MS, "10000");

        let settings =
            lsm_settings_from_env("helion/c").expect("manifest poll knob -> Some(Settings)");
        let reader_options = reader_options_from_env("helion/c");

        assert_eq!(
            settings.manifest_poll_interval,
            std::time::Duration::from_secs(10)
        );
        // Without a dedicated reader knob the shared value still applies.
        assert_eq!(
            reader_options.manifest_poll_interval,
            std::time::Duration::from_secs(10)
        );
    }

    /// The three compactor cadence knobs land on `CompactorOptions` (and the
    /// embedded worker) while untouched compactor defaults are preserved.
    #[test]
    #[serial]
    fn lsm_settings_from_env_reflects_compactor_interval_knobs() {
        let _cache = EnvGuard::set(ENV_CACHE_DIR, "");
        let _poll = EnvGuard::set(ENV_COMPACTOR_POLL_MS, "30000");
        let _commit = EnvGuard::set(ENV_COMPACTOR_COMMIT_COMPACTED_MS, "30000");
        let _worker = EnvGuard::set(ENV_COMPACTOR_WORKER_POLL_MS, "30000");

        let settings =
            lsm_settings_from_env("helion/c").expect("compactor knobs -> Some(Settings)");
        let compactor = settings
            .compactor_options
            .expect("compactor options present");
        assert_eq!(compactor.poll_interval, std::time::Duration::from_secs(30));
        assert_eq!(
            compactor.commit_compacted_interval,
            std::time::Duration::from_secs(30)
        );
        assert_eq!(
            compactor
                .worker
                .expect("embedded worker enabled by default")
                .compactions_poll_interval,
            std::time::Duration::from_secs(30)
        );
        // Untouched compactor defaults survive the in-place mutation.
        assert_eq!(compactor.max_concurrent_compactions, 4);
    }

    /// What the fork's scheduler reads out of the options a settings build hands it.
    fn scheduler_options_of(
        settings: Settings,
    ) -> (
        HashMap<String, String>,
        slatedb::config::SizeTieredCompactionSchedulerOptions,
    ) {
        let handed = settings
            .compactor_options
            .expect("compactor options present")
            .scheduler_options;
        let read = slatedb::config::SizeTieredCompactionSchedulerOptions::from(&handed);
        (handed, read)
    }

    /// With none of its three settings in the environment the scheduler is
    /// handed nothing and merges by the fork's defaults, as before the settings
    /// existed: four runs in a row, eight at most, each within four times the
    /// smallest.
    #[test]
    #[serial]
    fn compaction_scheduler_keeps_the_forks_defaults_unless_set() {
        let _cache = EnvGuard::set(ENV_CACHE_DIR, "");
        let _poll = EnvGuard::set(ENV_COMPACTOR_POLL_MS, "30000");
        let _min = EnvGuard::remove(ENV_COMPACTION_MIN_SOURCES);
        let _max = EnvGuard::remove(ENV_COMPACTION_MAX_SOURCES);
        let _threshold = EnvGuard::remove(ENV_COMPACTION_SIZE_THRESHOLD);

        let (handed, read) = scheduler_options_of(
            lsm_settings_from_env("helion/c").expect("compactor knob -> Some(Settings)"),
        );

        assert!(handed.is_empty(), "nothing set, nothing handed: {handed:?}");
        assert_eq!(read.min_compaction_sources, 4);
        assert_eq!(read.max_compaction_sources, 8);
        assert_eq!(read.include_size_threshold, 4.0);
    }

    /// The three settings reach the scheduler under the fork's own keys, and
    /// are enough by themselves for a settings build (no other knob is set).
    #[test]
    #[serial]
    fn compaction_scheduler_follows_its_three_settings() {
        let _cache = EnvGuard::set(ENV_CACHE_DIR, "");
        let _min = EnvGuard::set(ENV_COMPACTION_MIN_SOURCES, "2");
        let _max = EnvGuard::set(ENV_COMPACTION_MAX_SOURCES, "999");
        let _threshold = EnvGuard::set(ENV_COMPACTION_SIZE_THRESHOLD, " 1000000 ");

        let settings =
            lsm_settings_from_env("helion/c").expect("a scheduler setting alone -> Some(Settings)");
        let cadence = settings
            .compactor_options
            .as_ref()
            .expect("compactor options present")
            .poll_interval;
        let (handed, read) = scheduler_options_of(settings);

        assert_eq!(handed.len(), 3, "{handed:?}");
        assert_eq!(read.min_compaction_sources, 2);
        assert_eq!(read.max_compaction_sources, 999);
        assert_eq!(read.include_size_threshold, 1_000_000.0);
        // The compactor's other options stay the fork's.
        assert_eq!(
            cadence,
            Settings::default().compactor_options.unwrap().poll_interval
        );
    }

    /// A value the scheduler could not merge by is left out, so its default
    /// stands, and the values beside it are judged against what is then in
    /// force. Each row: minimum, maximum, threshold, the two L0 limits, and the
    /// (minimum, maximum, threshold) the scheduler ends with.
    #[test]
    #[serial]
    fn compaction_scheduler_ignores_a_setting_it_cannot_merge_by() {
        let _cache = EnvGuard::set(ENV_CACHE_DIR, "");
        let _poll = EnvGuard::set(ENV_COMPACTOR_POLL_MS, "30000");
        let rows: [(&str, &str, &str, &str, &str, (usize, usize, f32)); 17] = [
            // One run is no merge, a maximum of 0 panics the fork's scheduler,
            // and a threshold has to be a finite number above 1.
            ("1", "1", "0.5", "", "", (4, 8, 4.0)),
            ("0", "0", "NaN", "", "", (4, 8, 4.0)),
            ("two", "-3", "inf", "", "", (4, 8, 4.0)),
            ("", " ", "-4", "", "", (4, 8, 4.0)),
            ("", "", "1", "", "", (4, 8, 4.0)),
            ("", "", "1.0", "", "", (4, 8, 4.0)),
            ("", "", "1.5", "", "", (4, 8, 1.5)),
            // L0 is compacted only once it holds the minimum, and the writer
            // stops flushing at the lower of the two L0 limits (8 and 8 unset).
            ("9", "", "", "", "", (4, 8, 4.0)),
            ("8", "", "", "", "", (8, 8, 4.0)),
            ("5", "", "", "", "4", (4, 8, 4.0)),
            ("9", "12", "", "16", "16", (9, 12, 4.0)),
            // The fork cuts a row of runs down to the maximum: a maximum under
            // the minimum in force is left out, and so is a minimum above the
            // maximum in force.
            ("12", "", "", "16", "16", (4, 8, 4.0)),
            ("12", "10", "", "16", "16", (4, 8, 4.0)),
            ("5", "2", "", "", "", (5, 8, 4.0)),
            ("", "3", "", "", "", (4, 8, 4.0)),
            ("6", "5", "", "", "", (6, 8, 4.0)),
            ("2", "2", "", "", "", (2, 2, 4.0)),
        ];
        for (min, max, threshold, l0_max, l0_max_per_key, want) in rows {
            let _min = EnvGuard::set(ENV_COMPACTION_MIN_SOURCES, min);
            let _max = EnvGuard::set(ENV_COMPACTION_MAX_SOURCES, max);
            let _threshold = EnvGuard::set(ENV_COMPACTION_SIZE_THRESHOLD, threshold);
            let _l0_max = EnvGuard::set(ENV_L0_MAX_SSTS, l0_max);
            let _l0_max_per_key = EnvGuard::set(ENV_L0_MAX_SSTS_PER_KEY, l0_max_per_key);
            let row = format!("{min:?} {max:?} {threshold:?} L0 {l0_max:?} {l0_max_per_key:?}");

            let (handed, read) = scheduler_options_of(
                lsm_settings_from_env("helion/c").expect("compactor knob -> Some(Settings)"),
            );

            let got = (
                read.min_compaction_sources,
                read.max_compaction_sources,
                read.include_size_threshold,
            );
            assert_eq!(got, want, "{row}: handed {handed:?}");
            // What is left out is not handed at all, so the default is the fork's.
            let taken =
                usize::from(want.0 != 4) + usize::from(want.1 != 8) + usize::from(want.2 != 4.0);
            assert_eq!(handed.len(), taken, "{row}: handed {handed:?}");
        }
    }

    /// Deployment contract: the scheduler's three settings are for a smoke pod
    /// on a private copy. They are read from the process environment at every
    /// collection open, so a line in a production manifest would reach every
    /// collection a pod opens.
    #[test]
    fn public_kubernetes_config_sets_none_of_the_compaction_scheduler_settings() {
        let manifests = [(
            "10-configmap.yaml",
            include_str!("../../../../deploy/lsm-cloud/kubernetes/10-configmap.yaml"),
        )];
        for (name, text) in manifests {
            for key in [
                ENV_COMPACTION_MIN_SOURCES,
                ENV_COMPACTION_MAX_SOURCES,
                ENV_COMPACTION_SIZE_THRESHOLD,
            ] {
                let set = text
                    .lines()
                    .filter(|line| !line.trim_start().starts_with('#'))
                    .any(|line| line.contains(key));
                assert!(
                    !set,
                    "{name} must not set {key}: it is for a smoke pod only"
                );
            }
        }
    }

    /// Deployment contract: all three phases that release L0 pressure must
    /// retain the bounded production cadence. A minutes-long value on any one
    /// phase can reintroduce the same collection-wide write stall.
    #[test]
    fn public_kubernetes_config_bounds_all_compactor_poll_phases() {
        let config = include_str!("../../../../deploy/lsm-cloud/kubernetes/10-configmap.yaml");
        for key in [
            ENV_COMPACTOR_POLL_MS,
            ENV_COMPACTOR_COMMIT_COMPACTED_MS,
            ENV_COMPACTOR_WORKER_POLL_MS,
        ] {
            let expected = format!("{key}: \"30000\"");
            assert!(
                config.lines().any(|line| line.trim() == expected),
                "public Kubernetes config must keep {key} at the bounded 30s cadence"
            );
        }
    }

    #[test]
    #[serial]
    fn lsm_reader_manifest_poll_knob_overrides_shared_knob() {
        let _cache = EnvGuard::set(ENV_CACHE_DIR, "");
        let _poll = EnvGuard::set(ENV_MANIFEST_POLL_MS, "30000");
        let _reader_poll = EnvGuard::set(ENV_READER_MANIFEST_POLL_MS, "5000");

        let settings =
            lsm_settings_from_env("helion/c").expect("manifest poll knob -> Some(Settings)");
        let reader_options = reader_options_from_env("helion/c");

        // Writer keeps the S3-cost cadence; the reader keeps its own freshness
        // window instead of inheriting 30s staleness.
        assert_eq!(
            settings.manifest_poll_interval,
            std::time::Duration::from_secs(30)
        );
        assert_eq!(
            reader_options.manifest_poll_interval,
            std::time::Duration::from_secs(5)
        );
    }

    #[test]
    #[serial]
    fn reader_wal_replay_batch_knob_defaults_clamps_and_rejects_zero() {
        let _unset = EnvGuard::remove(ENV_READER_WAL_REPLAY_BATCH);
        assert_eq!(
            reader_options_from_env("helion/c").wal_replay_sst_batch_size,
            DEFAULT_READER_WAL_REPLAY_BATCH
        );
        let _eight = EnvGuard::set(ENV_READER_WAL_REPLAY_BATCH, "8");
        assert_eq!(
            reader_options_from_env("helion/c").wal_replay_sst_batch_size,
            8
        );
        let _zero = EnvGuard::set(ENV_READER_WAL_REPLAY_BATCH, "0");
        assert_eq!(
            reader_options_from_env("helion/c").wal_replay_sst_batch_size,
            DEFAULT_READER_WAL_REPLAY_BATCH
        );
        let _huge = EnvGuard::set(ENV_READER_WAL_REPLAY_BATCH, "100000");
        assert_eq!(
            reader_options_from_env("helion/c").wal_replay_sst_batch_size,
            MAX_READER_WAL_REPLAY_BATCH
        );
    }

    #[test]
    #[serial]
    fn reader_options_start_fast_when_feed_enabled_unless_sweeper_off() {
        let _cache = EnvGuard::set(ENV_CACHE_DIR, "");
        let _reader_poll = EnvGuard::set(ENV_READER_MANIFEST_POLL_MS, "5000");
        let _feed = EnvGuard::set(ENV_WRITER_FEED_URL, "http://helix:6969");
        let _idle = EnvGuard::set(ENV_READER_IDLE_POLL_MS, "300000");
        std::env::remove_var(ENV_READER_READ_IDLE_AFTER_MS);

        // Fresh opens serve at the ACTIVE cadence; the read-gated sweeper
        // demotes them to the idle safety net once they go unread.
        let reader_options = reader_options_from_env("helion/c");
        assert_eq!(
            reader_options.manifest_poll_interval,
            std::time::Duration::from_secs(5)
        );
        assert_eq!(
            reader_active_poll_interval_from_env(),
            std::time::Duration::from_secs(5)
        );

        // Sweeper kill switch: a fast open would never be demoted, so the
        // feed-era idle-tier cold open is kept for that configuration.
        let _idle_after = EnvGuard::set(ENV_READER_READ_IDLE_AFTER_MS, "0");
        assert_eq!(
            reader_options_from_env("helion/c").manifest_poll_interval,
            std::time::Duration::from_secs(300)
        );
    }

    #[test]
    #[serial]
    fn lsm_settings_from_env_reflects_gc_interval_knobs() {
        let _cache = EnvGuard::set(ENV_CACHE_DIR, "");
        let _gc = EnvGuard::set(ENV_GC_INTERVAL_SECS, "300");
        let _wal_gc = EnvGuard::set(ENV_GC_WAL_INTERVAL_SECS, "900");
        let _fence = EnvGuard::set(ENV_GC_WAL_FENCE_INTERVAL_SECS, "3600");
        let _min_age = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "86400");
        let _compacted_min_age = EnvGuard::set(ENV_GC_COMPACTED_MIN_AGE_SECS, "172800");

        let settings = lsm_settings_from_env("helion/c").expect("GC knob -> Some(Settings)");
        let gc = settings
            .garbage_collector_options
            .expect("GC options stay enabled");

        assert_eq!(
            gc.manifest_options.and_then(|options| options.interval),
            Some(std::time::Duration::from_secs(300))
        );
        assert_eq!(
            gc.wal_options.and_then(|options| options.interval),
            Some(std::time::Duration::from_secs(900))
        );
        assert_eq!(
            gc.compacted_options.and_then(|options| options.interval),
            Some(std::time::Duration::from_secs(300))
        );
        assert_eq!(
            gc.compactions_options.and_then(|options| options.interval),
            Some(std::time::Duration::from_secs(300))
        );
        assert_eq!(
            gc.manifest_options.map(|options| options.min_age),
            Some(std::time::Duration::from_secs(86400))
        );
        assert_eq!(
            gc.wal_options.map(|options| options.min_age),
            Some(std::time::Duration::from_secs(86400))
        );
        assert_eq!(
            gc.compacted_options.map(|options| options.min_age),
            Some(std::time::Duration::from_secs(172800))
        );
        assert_eq!(
            gc.compactions_options.map(|options| options.min_age),
            Some(std::time::Duration::from_secs(86400))
        );
        // Fence keeps its dedicated override and stays dry-run (log-only);
        // detach falls back to the generic interval.
        let fence = gc.wal_fence_options.expect("fence sweep stays enabled");
        assert_eq!(fence.interval, Some(std::time::Duration::from_secs(3600)));
        assert!(fence.dry_run);
        assert_eq!(
            gc.detach_options.and_then(|options| options.interval),
            Some(std::time::Duration::from_secs(300))
        );
    }

    fn object_count(store: &Arc<dyn ObjectStore>, prefix: &str) -> usize {
        block_on_lsm(&shared_lsm_handle(), async {
            store
                .list(Some(&ObjPath::from(prefix)))
                .try_collect::<Vec<_>>()
                .await
                .map_err(|error| error.to_string())
        })
        .unwrap()
        .len()
    }

    fn put_one(be: &LsmBackend, key: &[u8]) {
        let mut w = be.begin_write().unwrap();
        be.put(&mut w, Namespace::Nodes, key, b"v").unwrap();
        be.commit(w).unwrap();
    }

    /// A resident writer that goes write-quiet flushes its memtable to L0 while
    /// still open, so a reader cold open no longer replays the WAL tail. It
    /// does not flush while writes keep arriving inside the quiet period.
    #[test]
    #[serial]
    fn idle_writer_flushes_memtable_to_l0_after_write_quiet_period() {
        use crate::helix_engine::storage_core::backend_lsm_reader::LsmReader;
        let _cache = EnvGuard::set(ENV_CACHE_DIR, "");
        let _idle = EnvGuard::set(ENV_IDLE_FLUSH_SECS, "1");
        let path = "idle-flush/busy";
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let be = LsmBackend::open_with_store(path, store.clone()).unwrap();
        let compacted = format!("{path}/compacted");

        for i in 0..8u8 {
            put_one(&be, &[b'k', i]);
            std::thread::sleep(std::time::Duration::from_millis(300));
        }
        assert_eq!(
            object_count(&store, &compacted),
            0,
            "writes every 300ms keep the 1s quiet period from elapsing"
        );

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while object_count(&store, &compacted) == 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        assert!(
            object_count(&store, &compacted) > 0,
            "a write-quiet writer must flush its memtable to an L0 SST without closing"
        );

        let reader = LsmReader::open_with_store(path, store.clone()).unwrap();
        for i in 0..8u8 {
            let got = reader
                .get_with(Namespace::Nodes, &[b'k', i], |v| v.map(|b| b.to_vec()))
                .unwrap();
            assert_eq!(got, Some(b"v".to_vec()));
        }
        be.close().unwrap();
    }

    /// A writer that took over without a clean close (crash, fenced rollout)
    /// replays the old WAL tail into its memtable; with no new writes it must
    /// still flush that tail after one quiet period.
    #[test]
    #[serial]
    fn idle_flush_drains_a_wal_tail_replayed_at_open() {
        let _cache = EnvGuard::set(ENV_CACHE_DIR, "");
        let path = "idle-flush/replayed";
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let compacted = format!("{path}/compacted");
        let crashed = {
            let _idle = EnvGuard::remove(ENV_IDLE_FLUSH_SECS);
            let be = LsmBackend::open_with_store(path, store.clone()).unwrap();
            put_one(&be, b"tail-1");
            put_one(&be, b"tail-2");
            be
        };
        assert_eq!(object_count(&store, &compacted), 0);

        let _idle = EnvGuard::set(ENV_IDLE_FLUSH_SECS, "1");
        let be = LsmBackend::open_with_store(path, store.clone()).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while object_count(&store, &compacted) == 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        assert!(
            object_count(&store, &compacted) > 0,
            "the replayed WAL tail must be flushed to L0 without new writes"
        );
        let r = be.begin_read().unwrap();
        assert_eq!(
            get_owned(&be, &r, Namespace::Nodes, b"tail-2"),
            Some(b"v".to_vec())
        );
        be.close().unwrap();
        // Stop the fenced first writer's tasks; its close result is irrelevant.
        let _ = crashed.close();
    }

    /// A steady trickle never goes write-quiet. Its memtable is still flushed
    /// each time its oldest unflushed write is [`ENV_UNFLUSHED_MAX_AGE_SECS`]
    /// old, so the WAL tail a reader cold open replays is bounded by time.
    #[test]
    #[serial]
    fn steady_trickle_is_flushed_when_its_oldest_write_is_old_enough() {
        let _cache = EnvGuard::set(ENV_CACHE_DIR, "");
        let _idle = EnvGuard::set(ENV_IDLE_FLUSH_SECS, "30");
        let _age = EnvGuard::set(ENV_UNFLUSHED_MAX_AGE_SECS, "1");
        let path = "idle-flush/trickle";
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let be = LsmBackend::open_with_store(path, store.clone()).unwrap();
        let compacted = format!("{path}/compacted");

        let mut written = 0u8;
        let mut trickle_until = |ssts: usize| {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while object_count(&store, &compacted) < ssts && std::time::Instant::now() < deadline {
                put_one(&be, &[b't', written]);
                written += 1;
                std::thread::sleep(std::time::Duration::from_millis(300));
            }
            object_count(&store, &compacted)
        };
        assert!(
            trickle_until(1) >= 1,
            "a write every 300ms is never 30s quiet; the 1s age bound must flush it"
        );
        assert!(
            trickle_until(2) >= 2,
            "the writes after that flush are flushed in turn once they are 1s old"
        );
        be.close().unwrap();
    }

    /// The age of the unflushed tail counts from its first write, not from the
    /// flush before it: a burst after a long pause is not flushed at once.
    #[test]
    fn unflushed_age_counts_from_the_first_write_after_a_flush() {
        use std::sync::atomic::Ordering;
        // Two seconds of room: a runner that stalls between a write and the
        // check after it must not make the write look old.
        let (hour, bound) = (
            std::time::Duration::from_secs(3600),
            std::time::Duration::from_secs(2),
        );
        let longer_than_bound = std::time::Duration::from_millis(2500);
        let state = IdleFlushState::new();
        state.note_flushed(state.writes.load(Ordering::Acquire), 0);
        assert_eq!(
            state.due(hour, Some(std::time::Duration::ZERO)),
            None,
            "nothing unflushed is never due"
        );

        std::thread::sleep(longer_than_bound);
        state.note_write();
        assert_eq!(
            state.due(hour, Some(bound)),
            None,
            "2.5s after the flush, the one unflushed write is new"
        );
        assert_eq!(state.due(hour, None), None, "no age bound, not quiet");

        std::thread::sleep(longer_than_bound);
        let writes = state.writes.load(Ordering::Acquire);
        assert_eq!(state.due(hour, Some(bound)), Some((writes, "age")));
        assert_eq!(state.due(bound, None), Some((writes, "quiet")));
    }

    /// The age bound is on wherever the idle flush is on; 0 leaves the quiet
    /// period as the only trigger.
    #[test]
    #[serial]
    fn unflushed_max_age_is_on_by_default_and_zero_turns_it_off() {
        let _unset = EnvGuard::remove(ENV_UNFLUSHED_MAX_AGE_SECS);
        assert_eq!(
            unflushed_max_age_from_env(),
            Some(std::time::Duration::from_secs(300))
        );
        let _off = EnvGuard::set(ENV_UNFLUSHED_MAX_AGE_SECS, "0");
        assert_eq!(unflushed_max_age_from_env(), None);
        let _set = EnvGuard::set(ENV_UNFLUSHED_MAX_AGE_SECS, "120");
        assert_eq!(
            unflushed_max_age_from_env(),
            Some(std::time::Duration::from_secs(120))
        );
    }

    /// The idle flush is off unless configured, and on a clean memtable it
    /// writes nothing: reopening an already-flushed collection costs no PUTs.
    #[test]
    #[serial]
    fn idle_flush_is_off_by_default_and_free_on_a_clean_memtable() {
        let _cache = EnvGuard::set(ENV_CACHE_DIR, "");
        let path = "idle-flush/clean";
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        {
            let _idle = EnvGuard::remove(ENV_IDLE_FLUSH_SECS);
            let be = LsmBackend::open_with_store(path, store.clone()).unwrap();
            put_one(&be, b"a");
            std::thread::sleep(std::time::Duration::from_millis(2500));
            assert_eq!(
                object_count(&store, &format!("{path}/compacted")),
                0,
                "unset knob: no idle flush"
            );
            be.close().unwrap();
        }
        let _idle = EnvGuard::set(ENV_IDLE_FLUSH_SECS, "1");
        let be = LsmBackend::open_with_store(path, store.clone()).unwrap();
        let before = object_count(&store, path);
        std::thread::sleep(std::time::Duration::from_millis(3500));
        assert_eq!(
            object_count(&store, path),
            before,
            "a clean memtable must not write WAL, SST, or manifest objects"
        );
        be.close().unwrap();
    }

    /// SlateDB's builder rejects `max_wal_flushes_before_l0_flush < 4096`, so a
    /// sub-4096 knob must be ignored rather than fail every writer open.
    #[test]
    #[serial]
    fn writer_opens_when_wal_flush_bound_is_below_slatedb_minimum() {
        let _cache = EnvGuard::set(ENV_CACHE_DIR, "");
        let _wal_flushes = EnvGuard::set(ENV_MAX_WAL_FLUSHES_BEFORE_L0_FLUSH, "256");
        let backend =
            LsmBackend::open_with_store("wal-flush-bound-floor", Arc::new(InMemory::new()))
                .expect("a sub-4096 WAL flush bound must not fail the writer open");
        backend.close().unwrap();
    }

    #[test]
    #[serial]
    fn lsm_settings_from_env_reflects_read_amplification_knobs() {
        let _cache = EnvGuard::set(ENV_CACHE_DIR, "");
        let _min_filter = EnvGuard::set(ENV_MIN_FILTER_KEYS, "1");
        let _wal_flushes = EnvGuard::set(ENV_MAX_WAL_FLUSHES_BEFORE_L0_FLUSH, "8192");
        let _l0_per_key = EnvGuard::set(ENV_L0_MAX_SSTS_PER_KEY, "4");
        let _l0_parallelism = EnvGuard::set(ENV_L0_FLUSH_PARALLELISM, "2");

        let settings =
            lsm_settings_from_env("helion/c").expect("LSM read-amplification knobs -> Settings");

        assert_eq!(settings.min_filter_keys, 1);
        assert_eq!(settings.max_wal_flushes_before_l0_flush, 8192);
        assert_eq!(settings.l0_max_ssts_per_key, 4);
        assert_eq!(settings.l0_flush_parallelism, 2);
    }

    #[test]
    #[serial]
    fn lsm_settings_from_env_reflects_compression_codec() {
        let _cache = EnvGuard::set(ENV_CACHE_DIR, "");
        let _compression = EnvGuard::set(ENV_COMPRESSION_CODEC, "lz4");

        let settings =
            lsm_settings_from_env("helion/c").expect("compression codec knob -> Settings");

        assert_eq!(settings.compression_codec, Some(CompressionCodec::Lz4));
    }

    #[test]
    #[serial]
    fn lsm_settings_from_env_treats_none_compression_as_default() {
        let _cache = EnvGuard::set(ENV_CACHE_DIR, "");
        let _compression = EnvGuard::set(ENV_COMPRESSION_CODEC, "none");

        assert!(lsm_settings_from_env("helion/c").is_none());
    }

    #[test]
    #[serial]
    fn lsm_scan_options_cache_hot_scans_and_keep_streaming_uncached() {
        {
            let _read_ahead = EnvGuard::set(ENV_SCAN_READ_AHEAD_BYTES, "");
            let _cache = EnvGuard::set(ENV_SCAN_CACHE_BLOCKS, "");
            let _fetch = EnvGuard::set(ENV_SCAN_MAX_FETCH_TASKS, "");
            let options = lsm_scan_options();

            assert_eq!(options.read_ahead_bytes, DEFAULT_SCAN_READ_AHEAD_BYTES);
            assert_eq!(options.max_fetch_tasks, DEFAULT_SCAN_MAX_FETCH_TASKS);
            assert!(options.cache_blocks);
            assert!(!lsm_streaming_scan_options().cache_blocks);
        }
        {
            let _read_ahead = EnvGuard::set(ENV_SCAN_READ_AHEAD_BYTES, "262144");
            let _cache = EnvGuard::set(ENV_SCAN_CACHE_BLOCKS, "false");
            let _fetch = EnvGuard::set(ENV_SCAN_MAX_FETCH_TASKS, "7");
            let options = lsm_scan_options();

            assert_eq!(options.read_ahead_bytes, 262_144);
            assert_eq!(options.max_fetch_tasks, 7);
            assert!(!options.cache_blocks);
        }
    }

    #[test]
    fn lsm_scan_prefix_detects_only_exact_prefix_ranges() {
        let namespace = b"nodes\0";

        assert_eq!(
            scan_prefix_for_namespace_range(namespace, &KeyRange::all()),
            Some(b"nodes\0".to_vec())
        );
        assert_eq!(
            scan_prefix_for_namespace_range(namespace, &KeyRange::prefix(b"app:")),
            Some(b"nodes\0app:".to_vec())
        );
        assert_eq!(
            scan_prefix_for_namespace_range(namespace, &KeyRange::prefix(&[0xff])),
            Some(b"nodes\0\xff".to_vec())
        );

        let arbitrary = KeyRange {
            start: Bound::Included(b"app:1".to_vec()),
            end: Bound::Included(b"app:9".to_vec()),
        };
        assert_eq!(scan_prefix_for_namespace_range(namespace, &arbitrary), None);
    }

    #[test]
    fn helix_prefix_extractors_use_only_stable_key_prefixes() {
        let ns = HelixNamespacePrefixExtractor;
        assert_eq!(
            ns.prefix_len(&PrefixTarget::Point(Bytes::from_static(b"nodes\0abc"))),
            Some(b"nodes\0".len())
        );
        assert_eq!(
            ns.prefix_len(&PrefixTarget::Prefix(Bytes::from_static(
                b"sparse_inv_code\0"
            ))),
            Some(b"sparse_inv_code\0".len())
        );
        assert_eq!(
            ns.prefix_len(&PrefixTarget::Prefix(Bytes::from_static(b"nodes"))),
            None
        );

        let dup = HelixDupKeyPrefixExtractor;
        let sparse_ns = Namespace::SparseSegment {
            physical_name: "code",
            db: crate::helix_engine::storage_core::backend::SparseDb::Inv,
        };
        let prefix = dup_key_prefix(sparse_ns, b"term");
        let full_key = dup_composite(sparse_ns, b"term", b"doc1");
        assert_eq!(
            dup.prefix_len(&PrefixTarget::Point(Bytes::from(full_key))),
            Some(prefix.len())
        );
        assert_eq!(
            dup.prefix_len(&PrefixTarget::Prefix(Bytes::from(prefix.clone()))),
            Some(prefix.len())
        );
        assert_eq!(
            dup.prefix_len(&PrefixTarget::Prefix(Bytes::from(ns_prefix(sparse_ns)))),
            None
        );
        assert_eq!(
            dup.prefix_len(&PrefixTarget::Point(Bytes::from_static(
                b"nodes\0\0\0\0\x01a"
            ))),
            None
        );
    }

    #[test]
    #[serial]
    fn prefix_filter_policies_are_env_gated_and_keep_default_bloom() {
        let _disabled = EnvGuard::set(ENV_PREFIX_FILTERS, "0");
        assert!(prefix_filter_policies_from_env().is_none());
        drop(_disabled);

        let _enabled = EnvGuard::set(ENV_PREFIX_FILTERS, "1");
        let policies = prefix_filter_policies_from_env().expect("prefix filters enabled");
        let names: Vec<&str> = policies.iter().map(|policy| policy.name()).collect();
        assert_eq!(
            names,
            vec!["_bf", "_bf:p=helion-ns-v1:wh=0", "_bf:p=helion-dup-v1:wh=0"]
        );
    }

    /// Finding 3: each collection's SlateDB evictor enforces its cap by scanning
    /// & deleting across its ENTIRE `root_folder`, so a shared root makes
    /// evictors evict each other's files. `cache_options_from_env` therefore (a)
    /// namespaces the root per collection (distinct, sanitized, single-component
    /// dirs) so each evictor is confined to its own subtree, and (b) divides the
    /// TOTAL cap by the expected open-collection count so the resident subdirs
    /// sum to <= the configured total cap.
    #[test]
    #[serial]
    fn cache_options_namespaces_root_and_divides_cap_per_collection() {
        const ONE_GIB: usize = 1024 * 1024 * 1024;
        let _dir = EnvGuard::set(ENV_CACHE_DIR, "/mnt/ssd-cache");
        let _cap = EnvGuard::set(ENV_CACHE_MAX_BYTES, &ONE_GIB.to_string());
        let _memory_cap = EnvGuard::set(ENV_CACHE_MEMORY_MAX_PCT, "0");
        let _open = EnvGuard::set(ENV_MAX_OPEN_COLLECTIONS, "8");

        let a = cache_options_from_env("helion/alpha").expect("cache dir set -> Some");
        let b = cache_options_from_env("helion/beta").expect("cache dir set -> Some");

        // (a) Distinct, sanitized, single-component roots: one evictor's
        // whole-root scan can never reach the other collection's files.
        assert_eq!(
            a.root_folder,
            Some(std::path::PathBuf::from("/mnt/ssd-cache/helion_alpha"))
        );
        assert_eq!(
            b.root_folder,
            Some(std::path::PathBuf::from("/mnt/ssd-cache/helion_beta"))
        );
        assert_ne!(a.root_folder, b.root_folder);

        // (b) Per-collection cap = total / max_open; the sum across the <=8
        // resident collections stays within the configured total cap.
        let per = ONE_GIB / 8;
        assert_eq!(a.max_cache_size_bytes, Some(per));
        assert_eq!(b.max_cache_size_bytes, Some(per));
        assert!(
            per * 8 <= ONE_GIB,
            "sum of per-collection caps must not exceed the total cap"
        );
    }

    /// Raising the residency cap must not shrink each collection's cache share
    /// once `HELIX_LSM_CACHE_COLLECTION_DIVISOR` pins the divisor.
    #[test]
    #[serial]
    fn cache_collection_divisor_decouples_cap_from_max_open() {
        const ONE_GIB: usize = 1024 * 1024 * 1024;
        let _dir = EnvGuard::set(ENV_CACHE_DIR, "/mnt/ssd-cache");
        let _cap = EnvGuard::set(ENV_CACHE_MAX_BYTES, &ONE_GIB.to_string());
        let _memory_cap = EnvGuard::set(ENV_CACHE_MEMORY_MAX_PCT, "0");
        let _open = EnvGuard::set(ENV_MAX_OPEN_COLLECTIONS, "64");

        {
            let _divisor = EnvGuard::remove(ENV_CACHE_COLLECTION_DIVISOR);
            let cache = cache_options_from_env("helion/prod").expect("cache dir set -> Some");
            assert_eq!(cache.max_cache_size_bytes, Some(ONE_GIB / 64));
        }
        {
            let _divisor = EnvGuard::set(ENV_CACHE_COLLECTION_DIVISOR, "0");
            let cache = cache_options_from_env("helion/prod").expect("cache dir set -> Some");
            assert_eq!(
                cache.max_cache_size_bytes,
                Some(ONE_GIB / 64),
                "0 keeps the max-open divisor"
            );
        }
        {
            let _divisor = EnvGuard::set(ENV_CACHE_COLLECTION_DIVISOR, "32");
            let cache = cache_options_from_env("helion/prod").expect("cache dir set -> Some");
            assert_eq!(cache.max_cache_size_bytes, Some(ONE_GIB / 32));
            // The FD budget still divides by the residency cap.
            let _handle_override = EnvGuard::set(ENV_CACHE_MAX_OPEN_FILES, "0");
            let _fd_high = EnvGuard::set("HELIX_MAINTENANCE_FD_HIGH", "4096");
            let cache = cache_options_from_env("helion/prod").expect("cache dir set -> Some");
            assert_eq!(cache.max_open_file_handles, 2048 / 64);
        }
    }

    /// The share is held to the cache volume: with every handle open, the
    /// directories at their full share stay within the budget evictor's line,
    /// whatever the divisor says. A share the volume holds is left alone.
    #[test]
    #[serial]
    fn cache_share_is_cut_to_what_the_volume_holds_with_every_handle_open() {
        let volume = tempfile::tempdir().expect("tempdir");
        let total = cache_evictor::volume_usage(volume.path())
            .expect("statvfs of the temp dir")
            .total;
        let _dir = EnvGuard::set(
            ENV_CACHE_DIR,
            volume.path().to_str().expect("utf-8 temp path"),
        );
        let _memory_cap = EnvGuard::set(ENV_CACHE_MEMORY_MAX_PCT, "0");
        // Not the default of 70: the bound reads the budget the evictor reads.
        let _budget = EnvGuard::set("HELIX_LSM_CACHE_BUDGET_PCT", "40");
        let _open = EnvGuard::set(ENV_MAX_OPEN_COLLECTIONS, "4");
        let _divisor = EnvGuard::set(ENV_CACHE_COLLECTION_DIVISOR, "1");

        // A share as large as the whole volume: four of them open cannot fit.
        {
            let _cap = EnvGuard::set(ENV_CACHE_MAX_BYTES, &total.to_string());
            let cache = cache_options_from_env("helion/prod").expect("cache dir set -> Some");
            let held = usize::try_from(total as u128 * 40 / 100 / 4).expect("fits usize");
            assert_eq!(cache.max_cache_size_bytes, Some(held));
        }
        // A share the volume holds four times over is what the divisor says.
        {
            let small = usize::try_from(total / 100).expect("fits usize");
            let _cap = EnvGuard::set(ENV_CACHE_MAX_BYTES, &small.to_string());
            let cache = cache_options_from_env("helion/prod").expect("cache dir set -> Some");
            assert_eq!(cache.max_cache_size_bytes, Some(small));
        }
    }

    #[test]
    #[serial]
    fn cache_options_default_file_handles_stays_under_fd_gate_with_128_open_collections() {
        const ONE_GIB: usize = 1024 * 1024 * 1024;
        let _dir = EnvGuard::set(ENV_CACHE_DIR, "/mnt/ssd-cache");
        let _cap = EnvGuard::set(ENV_CACHE_MAX_BYTES, &ONE_GIB.to_string());
        let _memory_cap = EnvGuard::set(ENV_CACHE_MEMORY_MAX_PCT, "0");
        let _open = EnvGuard::set(ENV_MAX_OPEN_COLLECTIONS, "128");
        let _fd_high = EnvGuard::set("HELIX_MAINTENANCE_FD_HIGH", "4096");
        let _handle_override = EnvGuard::set(ENV_CACHE_MAX_OPEN_FILES, "0");

        let cache = cache_options_from_env("helion/prod").expect("cache dir set -> Some");

        assert_eq!(cache.max_open_file_handles, 16);
    }

    #[test]
    #[serial]
    fn cache_options_scan_interval_distinguishes_zero_unset_and_positive() {
        const ONE_GIB: usize = 1024 * 1024 * 1024;
        let _dir = EnvGuard::set(ENV_CACHE_DIR, "/mnt/ssd-cache");
        let _cap = EnvGuard::set(ENV_CACHE_MAX_BYTES, &ONE_GIB.to_string());
        let _memory_cap = EnvGuard::set(ENV_CACHE_MEMORY_MAX_PCT, "0");

        {
            let _scan = EnvGuard::remove(ENV_CACHE_SCAN_SECS);
            let cache = cache_options_from_env("helion/prod").expect("cache dir set -> Some");
            assert_eq!(
                cache.scan_interval,
                Some(std::time::Duration::from_secs(3600)),
                "unset must preserve SlateDB's recurring-scan default"
            );
        }

        {
            let _scan = EnvGuard::set(ENV_CACHE_SCAN_SECS, "0");
            let cache = cache_options_from_env("helion/prod").expect("cache dir set -> Some");
            assert_eq!(
                cache.scan_interval, None,
                "explicit zero must disable recurring scans"
            );
        }

        {
            let _scan = EnvGuard::set(ENV_CACHE_SCAN_SECS, "300");
            let cache = cache_options_from_env("helion/prod").expect("cache dir set -> Some");
            assert_eq!(
                cache.scan_interval,
                Some(std::time::Duration::from_secs(300)),
                "a positive value must set the recurring-scan cadence"
            );
        }
    }

    #[test]
    #[serial]
    fn cache_scan_disabled_reader_close_releases_background_tasks() {
        use slatedb::DbReader;

        const ONE_GIB: usize = 1024 * 1024 * 1024;
        let cache_dir = tempfile::TempDir::new().unwrap();
        let _dir = EnvGuard::set(ENV_CACHE_DIR, cache_dir.path().to_str().unwrap());
        let _cap = EnvGuard::set(ENV_CACHE_MAX_BYTES, &ONE_GIB.to_string());
        let _memory_cap = EnvGuard::set(ENV_CACHE_MEMORY_MAX_PCT, "0");
        let _scan = EnvGuard::set(ENV_CACHE_SCAN_SECS, "0");
        let _feed = EnvGuard::set(ENV_WRITER_FEED_URL, "");

        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let metrics = rt.metrics();
        rt.block_on(async {
            let path = "cache-scan-disabled-reader-close";
            let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
            let db = Db::builder(path, store.clone())
                .with_db_cache_disabled()
                .build()
                .await
                .expect("open writer");
            db.put(b"key", b"value").await.expect("write test value");
            db.close().await.expect("close writer");
            drop(db);

            for _ in 0..100 {
                if metrics.num_alive_tasks() == 0 {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            assert_eq!(
                metrics.num_alive_tasks(),
                0,
                "writer setup tasks must stop before measuring reader lifecycle"
            );

            for iteration in 0..3 {
                let options = reader_options_from_env(path);
                assert_eq!(options.object_store_cache_options.scan_interval, None);
                let reader = DbReader::builder(path, store.clone())
                    .with_options(options)
                    .with_db_cache_disabled()
                    .build()
                    .await
                    .expect("open cached reader");
                assert_eq!(
                    reader.get(b"key").await.expect("read test value"),
                    Some(Bytes::from_static(b"value"))
                );
                assert!(
                    metrics.num_alive_tasks() > 0,
                    "reader iteration {iteration} must start background tasks"
                );

                reader.close().await.expect("close reader");
                drop(reader);
                for _ in 0..100 {
                    if metrics.num_alive_tasks() == 0 {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
                assert_eq!(
                    metrics.num_alive_tasks(),
                    0,
                    "reader iteration {iteration} leaked a background task"
                );
            }
        });
    }

    #[test]
    #[serial]
    fn cache_options_part_size_knob_applies_and_rejects_unaligned() {
        const ONE_GIB: usize = 1024 * 1024 * 1024;
        let _dir = EnvGuard::set(ENV_CACHE_DIR, "/mnt/ssd-cache");
        let _cap = EnvGuard::set(ENV_CACHE_MAX_BYTES, &ONE_GIB.to_string());
        let _memory_cap = EnvGuard::set(ENV_CACHE_MEMORY_MAX_PCT, "0");

        let default_part = {
            let _part = EnvGuard::set(ENV_CACHE_PART_BYTES, "");
            cache_options_from_env("helion/prod")
                .expect("cache dir set -> Some")
                .part_size_bytes
        };

        {
            let _part = EnvGuard::set(ENV_CACHE_PART_BYTES, "16777216");
            let cache = cache_options_from_env("helion/prod").expect("cache dir set -> Some");
            assert_eq!(cache.part_size_bytes, 16 * 1024 * 1024);
        }

        // Not a multiple of 1024 (SlateDB would reject it at open) -> default.
        {
            let _part = EnvGuard::set(ENV_CACHE_PART_BYTES, "1000000");
            let cache = cache_options_from_env("helion/prod").expect("cache dir set -> Some");
            assert_eq!(cache.part_size_bytes, default_part);
        }
    }

    #[test]
    #[serial]
    fn cache_options_ranged_miss_defaults_on_and_unparsable_turns_it_off() {
        let _dir = EnvGuard::set(ENV_CACHE_DIR, "/mnt/ssd-cache");
        let _cap = EnvGuard::set(ENV_CACHE_MAX_BYTES, "1073741824");
        let ranged_miss = |value: Option<&str>| {
            let _knob = match value {
                Some(v) => EnvGuard::set(ENV_CACHE_RANGED_MISS, v),
                None => EnvGuard::remove(ENV_CACHE_RANGED_MISS),
            };
            cache_options_from_env("helion/prod")
                .expect("cache dir set -> Some")
                .ranged_miss
        };

        for on in [None, Some(""), Some("1"), Some("true"), Some(" ON ")] {
            assert!(ranged_miss(on), "{on:?} should leave ranged misses on");
        }
        for off in [
            Some("0"),
            Some("false"),
            Some("off"),
            Some("2"),
            Some("nope"),
        ] {
            assert!(!ranged_miss(off), "{off:?} should turn ranged misses off");
        }
    }

    #[test]
    #[serial]
    fn cache_options_part_fetch_concurrency_defaults_to_8_and_unparsable_means_one_at_a_time() {
        let _dir = EnvGuard::set(ENV_CACHE_DIR, "/mnt/ssd-cache");
        let _cap = EnvGuard::set(ENV_CACHE_MAX_BYTES, "1073741824");
        let concurrency = |value: Option<&str>| {
            let _knob = match value {
                Some(v) => EnvGuard::set(ENV_CACHE_PART_FETCH_CONCURRENCY, v),
                None => EnvGuard::remove(ENV_CACHE_PART_FETCH_CONCURRENCY),
            };
            cache_options_from_env("helion/prod")
                .expect("cache dir set -> Some")
                .part_fetch_concurrency
        };

        for (value, expected) in [
            (None, 8),
            (Some(""), 8),
            (Some("  "), 8),
            (Some("1"), 1),
            (Some("0"), 0),
            (Some(" 4 "), 4),
            (Some("16"), 16),
            (Some("off"), 1),
            (Some("-3"), 1),
        ] {
            assert_eq!(concurrency(value), expected, "{value:?}");
        }
    }

    #[test]
    #[serial]
    fn cache_options_explicit_file_handle_override_wins() {
        const ONE_GIB: usize = 1024 * 1024 * 1024;
        let _dir = EnvGuard::set(ENV_CACHE_DIR, "/mnt/ssd-cache");
        let _cap = EnvGuard::set(ENV_CACHE_MAX_BYTES, &ONE_GIB.to_string());
        let _memory_cap = EnvGuard::set(ENV_CACHE_MEMORY_MAX_PCT, "0");
        let _open = EnvGuard::set(ENV_MAX_OPEN_COLLECTIONS, "64");
        let _handle_override = EnvGuard::set(ENV_CACHE_MAX_OPEN_FILES, "48");

        let cache = cache_options_from_env("helion/prod").expect("cache dir set -> Some");

        assert_eq!(cache.max_open_file_handles, 48);
    }

    #[test]
    fn memory_bounded_cache_cap_clamps_disk_cache_to_cgroup_budget() {
        const GIB: usize = 1024 * 1024 * 1024;

        assert_eq!(
            memory_bounded_cache_cap(80 * GIB, Some(64 * GIB), 25),
            16 * GIB
        );
        assert_eq!(
            memory_bounded_cache_cap(8 * GIB, Some(64 * GIB), 25),
            8 * GIB
        );
        assert_eq!(
            memory_bounded_cache_cap(80 * GIB, Some(64 * GIB), 0),
            80 * GIB
        );
        assert_eq!(memory_bounded_cache_cap(80 * GIB, None, 25), 80 * GIB);
    }

    /// The collection id (its SlateDB path) is sanitized into a SINGLE
    /// filesystem-safe directory component: no `/` survives, so per-collection
    /// roots never nest into one another and the evictor scan stays scoped.
    #[test]
    fn sanitize_cache_component_makes_single_safe_dir() {
        assert_eq!(sanitize_cache_component("helion/my_coll"), "helion_my_coll");
        assert_eq!(sanitize_cache_component("a/b/c"), "a_b_c");
        // Already-safe chars are preserved.
        assert_eq!(sanitize_cache_component("ok.name-1_2"), "ok.name-1_2");
        // No path separator can survive sanitization.
        assert!(!sanitize_cache_component("/abs/path").contains('/'));
    }

    /// Real-S3 smoke test against a MinIO S3 endpoint on :9000. Ignored by
    /// default (needs `deploy/minio/docker-compose.yml` up + the
    /// `helion-test` bucket); run with
    /// `cargo test -p helixdb lsm_backend_on_minio_s3 -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn lsm_backend_on_minio_s3() {
        let store = AmazonS3Builder::new()
            .with_endpoint("http://localhost:9000")
            .with_region("us-east-1")
            .with_bucket_name("helion-test")
            .with_access_key_id("test")
            .with_secret_access_key("testtest123")
            .with_allow_http(true)
            .build()
            .expect("build MinIO S3 store");
        let be = LsmBackend::open_with_store("helion", Arc::new(store))
            .expect("open SlateDB on MinIO S3");

        let mut w = be.begin_write().unwrap();
        be.put(&mut w, Namespace::Nodes, b"k1", b"v1").unwrap();
        be.put(&mut w, Namespace::Nodes, b"k2", b"v2").unwrap();
        be.put_dup(&mut w, Namespace::OutEdges, b"n1", b"e1")
            .unwrap();
        be.commit(w).unwrap();

        let r = be.begin_read().unwrap();
        assert_eq!(
            be.get_with(&r, Namespace::Nodes, b"k1", |v| v.map(|b| b.to_vec()))
                .unwrap(),
            Some(b"v1".to_vec())
        );

        let mut keys = Vec::new();
        be.scan(&r, Namespace::Nodes, KeyRange::all(), |k, _v| {
            keys.push(k.to_vec());
            true
        })
        .unwrap();
        assert_eq!(keys, vec![b"k1".to_vec(), b"k2".to_vec()]);

        let mut dups = Vec::new();
        be.for_each_dup(&r, Namespace::OutEdges, b"n1", |v| {
            dups.push(v.to_vec());
            true
        })
        .unwrap();
        assert_eq!(dups, vec![b"e1".to_vec()]);
    }

    /// Single-writer fencing on real S3 (MinIO): when a SECOND writer opens the
    /// same SlateDB db, the first is fenced (SlateDB bumps the writer epoch), so
    /// the first writer's next commit must fail while the new writer succeeds.
    /// Run with MinIO up: `cargo test -p helixdb lsm_single_writer_fencing_on_minio_s3 -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn lsm_single_writer_fencing_on_minio_s3() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let mk_store = || {
            AmazonS3Builder::new()
                .with_endpoint("http://localhost:9000")
                .with_region("us-east-1")
                .with_bucket_name("helion-test")
                .with_access_key_id("test")
                .with_secret_access_key("testtest123")
                .with_allow_http(true)
                .build()
                .expect("build MinIO S3 store")
        };
        // Unique db path per run so retained bucket state doesn't interfere.
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = format!("fence-{nanos}");

        let be1 = LsmBackend::open_with_store(&path, Arc::new(mk_store())).unwrap();
        let mut w = be1.begin_write().unwrap();
        be1.put(&mut w, Namespace::Nodes, b"a", b"1").unwrap();
        be1.commit(w).unwrap();

        // Second writer opens the SAME db -> fences be1.
        let be2 = LsmBackend::open_with_store(&path, Arc::new(mk_store())).unwrap();

        let mut w1 = be1.begin_write().unwrap();
        be1.put(&mut w1, Namespace::Nodes, b"b", b"2").unwrap();
        be1.commit(w1)
            .expect_err("fenced writer must fail closed instead of reopening");

        let mut w2 = be2.begin_write().unwrap();
        be2.put(&mut w2, Namespace::Nodes, b"c", b"3").unwrap();
        be2.commit(w2).unwrap();
    }

    #[test]
    #[serial]
    fn stale_a_failed_commit_does_not_fence_b_and_no_failed_batch_write() {
        use slatedb::object_store::local::LocalFileSystem;

        let data_dir = tempfile::TempDir::new().unwrap();
        let store: Arc<dyn ObjectStore> =
            Arc::new(LocalFileSystem::new_with_prefix(data_dir.path()).unwrap());
        let path = "helion/stale-a-failed-commit-does-not-fence-b";

        let be1 = LsmBackend::open_with_store(path, store.clone()).unwrap();
        let mut initial = be1.begin_write().unwrap();
        be1.put(&mut initial, Namespace::Nodes, b"a", b"1").unwrap();
        be1.commit(initial).unwrap();

        let be2 = LsmBackend::open_with_store(path, store.clone()).unwrap();
        let mut fencer = be2.begin_write().unwrap();
        be2.put(&mut fencer, Namespace::Nodes, b"b", b"2").unwrap();
        be2.commit(fencer).unwrap();

        let mut stale = be1.begin_write().unwrap();
        be1.put(&mut stale, Namespace::Nodes, b"c", b"3").unwrap();
        let err = be1
            .commit(stale)
            .expect_err("stale fenced writer must fail closed");
        assert!(matches!(err, BackendError::Conflict(_)), "got {err:?}");

        let mut follow_up = be2.begin_write().unwrap();
        be2.put(&mut follow_up, Namespace::Nodes, b"d", b"4")
            .unwrap();
        be2.commit(follow_up)
            .expect("active writer must not be fenced by stale writer failure");

        let r = be2.begin_read().unwrap();
        assert_eq!(
            get_owned(&be2, &r, Namespace::Nodes, b"a"),
            Some(b"1".to_vec())
        );
        assert_eq!(
            get_owned(&be2, &r, Namespace::Nodes, b"b"),
            Some(b"2".to_vec())
        );
        assert_eq!(get_owned(&be2, &r, Namespace::Nodes, b"c"), None);
        assert_eq!(
            get_owned(&be2, &r, Namespace::Nodes, b"d"),
            Some(b"4".to_vec())
        );
    }

    #[test]
    #[serial]
    fn lsm_fenced_buffered_commit_fails_without_reopen() {
        use slatedb::object_store::local::LocalFileSystem;

        let data_dir = tempfile::TempDir::new().unwrap();
        let store: Arc<dyn ObjectStore> =
            Arc::new(LocalFileSystem::new_with_prefix(data_dir.path()).unwrap());
        let path = "helion/reopen-buffered-lost";

        let be1 = LsmBackend::open_with_store(path, store.clone()).unwrap();
        let mut buffered = be1.begin_write().unwrap();
        be1.put(&mut buffered, Namespace::Nodes, b"a", b"1")
            .unwrap();
        be1.commit_buffered(buffered).unwrap();

        let be2 = LsmBackend::open_with_store(path, store.clone()).unwrap();
        let mut fencer = be2.begin_write().unwrap();
        be2.put(&mut fencer, Namespace::Nodes, b"b", b"2").unwrap();
        be2.commit(fencer).unwrap();

        let mut stale = be1.begin_write().unwrap();
        be1.put(&mut stale, Namespace::Nodes, b"c", b"3").unwrap();
        let err = be1
            .commit(stale)
            .expect_err("fenced durable commit must fail closed");
        assert!(matches!(err, BackendError::Conflict(_)), "got {err:?}");

        let mut active = be2.begin_write().unwrap();
        be2.put(&mut active, Namespace::Nodes, b"d", b"4").unwrap();
        be2.commit(active).unwrap();

        let r = be2.begin_read().unwrap();
        assert_eq!(get_owned(&be2, &r, Namespace::Nodes, b"c"), None);
        assert_eq!(
            get_owned(&be2, &r, Namespace::Nodes, b"d"),
            Some(b"4".to_vec())
        );
    }

    #[test]
    #[serial]
    fn lsm_fenced_flush_durable_fails_closed() {
        use slatedb::object_store::local::LocalFileSystem;

        let data_dir = tempfile::TempDir::new().unwrap();
        let store: Arc<dyn ObjectStore> =
            Arc::new(LocalFileSystem::new_with_prefix(data_dir.path()).unwrap());
        let path = "helion/reopen-buffered-chunk";

        let be1 = LsmBackend::open_with_store(path, store.clone()).unwrap();
        let mut first = be1.begin_write().unwrap();
        be1.put(&mut first, Namespace::Nodes, b"a", b"1").unwrap();
        be1.commit_buffered(first).unwrap();

        let be2 = LsmBackend::open_with_store(path, store.clone()).unwrap();
        let mut fencer = be2.begin_write().unwrap();
        be2.put(&mut fencer, Namespace::Nodes, b"b", b"2").unwrap();
        be2.commit(fencer).unwrap();

        let mut stale = be1.begin_write().unwrap();
        be1.put(&mut stale, Namespace::Nodes, b"c", b"3").unwrap();
        be1.commit(stale)
            .expect_err("stale writer must observe genuine fencing");

        let err = be1
            .flush_durable()
            .expect_err("fenced barrier must fail closed without reopening");
        assert!(matches!(err, BackendError::Conflict(_)), "got {err:?}");

        let mut active = be2.begin_write().unwrap();
        be2.put(&mut active, Namespace::Nodes, b"d", b"4").unwrap();
        be2.commit(active).unwrap();

        let r = be2.begin_read().unwrap();
        assert_eq!(get_owned(&be2, &r, Namespace::Nodes, b"c"), None);
    }

    #[test]
    fn lsm_round_trip_scan_and_namespace_isolation() {
        let be = LsmBackend::open_in_memory("/helion-test").unwrap();

        let mut w = be.begin_write().unwrap();
        be.put(&mut w, Namespace::Nodes, b"k1", b"v1").unwrap();
        be.put(&mut w, Namespace::Nodes, b"k2", b"v2").unwrap();
        // Different namespace, same logical key bytes — must stay isolated.
        be.put(&mut w, Namespace::Metadata, b"k1", b"other")
            .unwrap();
        be.commit(w).unwrap();

        let r = be.begin_read().unwrap();
        assert_eq!(
            get_owned(&be, &r, Namespace::Nodes, b"k1"),
            Some(b"v1".to_vec())
        );
        assert_eq!(
            get_owned(&be, &r, Namespace::Metadata, b"k1"),
            Some(b"other".to_vec())
        );
        assert_eq!(get_owned(&be, &r, Namespace::Nodes, b"missing"), None);

        // Scan of Nodes must NOT leak the Metadata entry.
        let mut keys = Vec::new();
        be.scan(&r, Namespace::Nodes, KeyRange::all(), |k, _v| {
            keys.push(k.to_vec());
            true
        })
        .unwrap();
        assert_eq!(keys, vec![b"k1".to_vec(), b"k2".to_vec()]);

        let mut w = be.begin_write().unwrap();
        be.delete(&mut w, Namespace::Nodes, b"k1").unwrap();
        be.commit(w).unwrap();
        let r = be.begin_read().unwrap();
        assert_eq!(get_owned(&be, &r, Namespace::Nodes, b"k1"), None);
        assert_eq!(
            get_owned(&be, &r, Namespace::Nodes, b"k2"),
            Some(b"v2".to_vec())
        );
    }

    #[test]
    fn lsm_collect_values_many_preserves_order_and_pending_overlay() {
        // Given: committed SlateDB rows plus an uncommitted write batch overlay.
        let be = LsmBackend::open_in_memory("/collect-values-many-test").unwrap();

        let mut w = be.begin_write().unwrap();
        be.put(&mut w, Namespace::Nodes, b"k1", b"old1").unwrap();
        be.put(&mut w, Namespace::Nodes, b"k2", b"old2").unwrap();
        be.commit(w).unwrap();

        let mut w = be.begin_write().unwrap();
        be.delete(&mut w, Namespace::Nodes, b"k1").unwrap();
        be.put(&mut w, Namespace::Nodes, b"k2", b"new2").unwrap();
        be.put(&mut w, Namespace::Nodes, b"k3", b"new3").unwrap();
        let r = be.begin_read_with_pending(w.pending().clone()).unwrap();

        // When: a batched point read asks for deleted, updated, missing, and new keys.
        let keys = vec![
            b"k1".to_vec(),
            b"k2".to_vec(),
            b"missing".to_vec(),
            b"k3".to_vec(),
        ];
        let rows = be
            .collect_values_many_with(&r, Namespace::Nodes, &keys)
            .unwrap();

        // Then: the response preserves input order and the pending overlay wins.
        assert_eq!(
            rows,
            vec![None, Some(b"new2".to_vec()), None, Some(b"new3".to_vec())]
        );
    }

    fn segment_ns(
        physical_name: &'static str,
        db: crate::helix_engine::storage_core::backend::SegmentDb,
    ) -> Namespace<'static> {
        Namespace::Segment { physical_name, db }
    }

    #[test]
    fn lsm_contains_many_answers_several_namespaces_in_one_round_trip() {
        use crate::helix_engine::storage_core::backend::SegmentDb;
        let be = LsmBackend::open_in_memory("/contains-many-test").unwrap();
        let first = segment_ns("dense__seg_000001", SegmentDb::Vectors);
        let second = segment_ns("dense__seg_000002", SegmentDb::Vectors);

        let mut w = be.begin_write().unwrap();
        be.put(&mut w, Namespace::Nodes, b"k1", b"node").unwrap();
        be.put(&mut w, first, b"k1", b"vector").unwrap();
        be.put(&mut w, second, b"k2", b"vector").unwrap();
        be.commit(w).unwrap();

        let r = be.begin_read().unwrap();
        let keys = vec![
            (first, b"k1".to_vec()),
            (first, b"k2".to_vec()),
            (second, b"k1".to_vec()),
            (second, b"k2".to_vec()),
            (Namespace::Nodes, b"k1".to_vec()),
            (Namespace::Nodes, b"k2".to_vec()),
        ];
        let before = lsm_round_trips_for_test();
        let present = be.contains_many_with(&r, &keys).unwrap();
        assert_eq!(lsm_round_trips_for_test() - before, 1);
        assert_eq!(present, vec![true, false, false, true, true, false]);
        // Each answer is the one a single-key read of that namespace gives.
        for ((ns, key), expected) in keys.iter().zip(&present) {
            let single = be.get_with(&r, *ns, key, |v| v.is_some()).unwrap();
            assert_eq!(single, *expected);
        }
    }

    #[test]
    fn lsm_scan_prefix_keys_many_matches_one_scan_per_prefix() {
        use crate::helix_engine::storage_core::backend::SegmentDb;
        let be = LsmBackend::open_in_memory("/scan-prefix-keys-many-test").unwrap();
        let vectors = segment_ns("dense__seg_000001", SegmentDb::Vectors);
        let out = segment_ns("dense__seg_000001", SegmentDb::HnswOut);
        let other = segment_ns("dense__seg_000002", SegmentDb::Vectors);

        let mut w = be.begin_write().unwrap();
        be.put(&mut w, vectors, b"v:a0", b"row").unwrap();
        be.put(&mut w, vectors, b"v:a1", b"row").unwrap();
        be.put(&mut w, vectors, b"v:b0", b"row").unwrap();
        be.put(&mut w, out, b"a-x", b"row").unwrap();
        be.put(&mut w, other, b"v:a0", b"row").unwrap();
        be.commit(w).unwrap();

        let r = be.begin_read().unwrap();
        let prefixes = vec![
            (vectors, b"v:a".to_vec()),
            (vectors, b"v:c".to_vec()),
            (out, b"a".to_vec()),
            (other, b"v:".to_vec()),
        ];
        let before = lsm_round_trips_for_test();
        let rows = be
            .scan_prefix_keys_many_with(&r, &prefixes)
            .unwrap()
            .expect("plain namespaces without an overlay are batched");
        assert_eq!(lsm_round_trips_for_test() - before, 1);
        assert_eq!(
            rows,
            vec![
                vec![b"v:a0".to_vec(), b"v:a1".to_vec()],
                Vec::new(),
                vec![b"a-x".to_vec()],
                vec![b"v:a0".to_vec()],
            ]
        );
        // Each row set is the one a scan of that prefix alone gives.
        for ((ns, prefix), expected) in prefixes.iter().zip(&rows) {
            let mut keys = Vec::new();
            be.scan(&r, *ns, KeyRange::prefix(prefix), |k, _| {
                keys.push(k.to_vec());
                true
            })
            .unwrap();
            assert_eq!(&keys, expected);
        }

        // A write overlay or a multi-value namespace is left to the caller.
        let mut w = be.begin_write().unwrap();
        be.put(&mut w, vectors, b"v:a2", b"row").unwrap();
        let overlay = be.begin_read_with_pending(w.pending().clone()).unwrap();
        assert!(be
            .scan_prefix_keys_many_with(&overlay, &prefixes)
            .unwrap()
            .is_none());
        assert!(be
            .scan_prefix_keys_many_with(&r, &[(Namespace::OutEdges, b"a".to_vec())])
            .unwrap()
            .is_none());
    }

    #[test]
    fn lsm_pending_scan_overlays_put_update_and_delete() {
        let be = LsmBackend::open_in_memory("/pending-scan-test").unwrap();

        let mut w = be.begin_write().unwrap();
        be.put(&mut w, Namespace::Nodes, b"k1", b"old1").unwrap();
        be.put(&mut w, Namespace::Nodes, b"k2", b"old2").unwrap();
        be.commit(w).unwrap();

        let mut w = be.begin_write().unwrap();
        be.delete(&mut w, Namespace::Nodes, b"k1").unwrap();
        be.put(&mut w, Namespace::Nodes, b"k2", b"new2").unwrap();
        be.put(&mut w, Namespace::Nodes, b"k3", b"new3").unwrap();
        be.put(&mut w, Namespace::Metadata, b"k4", b"other")
            .unwrap();
        let r = be.begin_read_with_pending(w.pending().clone()).unwrap();

        let mut rows = Vec::new();
        be.scan(&r, Namespace::Nodes, KeyRange::all(), |k, v| {
            rows.push((k.to_vec(), v.to_vec()));
            true
        })
        .unwrap();
        assert_eq!(
            rows,
            vec![
                (b"k2".to_vec(), b"new2".to_vec()),
                (b"k3".to_vec(), b"new3".to_vec()),
            ],
            "LSM scan read view must overlay pending put/update/delete effects"
        );
    }

    #[test]
    fn lsm_pending_dup_scan_overlays_put_and_delete() {
        let be = LsmBackend::open_in_memory("/pending-dup-scan-test").unwrap();

        let mut w = be.begin_write().unwrap();
        be.put_dup(&mut w, Namespace::OutEdges, b"n1", b"a")
            .unwrap();
        be.put_dup(&mut w, Namespace::OutEdges, b"n1", b"b")
            .unwrap();
        be.commit(w).unwrap();

        let mut w = be.begin_write().unwrap();
        be.delete_dup(&mut w, Namespace::OutEdges, b"n1", b"a")
            .unwrap();
        be.put_dup(&mut w, Namespace::OutEdges, b"n1", b"c")
            .unwrap();
        let r = be.begin_read_with_pending(w.pending().clone()).unwrap();

        let mut vals = Vec::new();
        be.for_each_dup(&r, Namespace::OutEdges, b"n1", |v| {
            vals.push(v.to_vec());
            true
        })
        .unwrap();
        assert_eq!(vals, vec![b"b".to_vec(), b"c".to_vec()]);

        let mut pairs = Vec::new();
        be.scan(&r, Namespace::OutEdges, KeyRange::all(), |k, v| {
            pairs.push((k.to_vec(), v.to_vec()));
            true
        })
        .unwrap();
        assert_eq!(
            pairs,
            vec![
                (b"n1".to_vec(), b"b".to_vec()),
                (b"n1".to_vec(), b"c".to_vec()),
            ],
            "LSM dup scan read view must overlay pending dup put/delete effects"
        );
    }

    #[test]
    fn lsm_dup_namespace_multi_value() {
        let be = LsmBackend::open_in_memory("/dup-test").unwrap();
        let mut w = be.begin_write().unwrap();
        be.put_dup(&mut w, Namespace::OutEdges, b"n1", b"eA")
            .unwrap();
        be.put_dup(&mut w, Namespace::OutEdges, b"n1", b"eB")
            .unwrap();
        be.put_dup(&mut w, Namespace::OutEdges, b"n2", b"eC")
            .unwrap();
        be.commit(w).unwrap();

        let r = be.begin_read().unwrap();
        let mut vals = Vec::new();
        be.for_each_dup(&r, Namespace::OutEdges, b"n1", |v| {
            vals.push(v.to_vec());
            true
        })
        .unwrap();
        assert_eq!(vals, vec![b"eA".to_vec(), b"eB".to_vec()]);

        let mut w = be.begin_write().unwrap();
        be.delete_dup(&mut w, Namespace::OutEdges, b"n1", b"eA")
            .unwrap();
        be.commit(w).unwrap();
        let r = be.begin_read().unwrap();
        let mut vals2 = Vec::new();
        be.for_each_dup(&r, Namespace::OutEdges, b"n1", |v| {
            vals2.push(v.to_vec());
            true
        })
        .unwrap();
        assert_eq!(vals2, vec![b"eB".to_vec()]);
    }

    #[test]
    fn lsm_dup_scan_returns_logical_keys_and_values() {
        let be = LsmBackend::open_in_memory("/dup-scan-test").unwrap();
        let mut w = be.begin_write().unwrap();
        be.put_dup(&mut w, Namespace::OutEdges, b"node-a:rel", b"edge-1")
            .unwrap();
        be.put_dup(&mut w, Namespace::OutEdges, b"node-a:rel", b"edge-2")
            .unwrap();
        be.put_dup(&mut w, Namespace::OutEdges, b"node-b:rel", b"edge-3")
            .unwrap();
        be.commit(w).unwrap();

        let r = be.begin_read().unwrap();
        let mut pairs = Vec::new();
        be.scan(
            &r,
            Namespace::OutEdges,
            KeyRange::prefix(b"node-a"),
            |k, v| {
                pairs.push((k.to_vec(), v.to_vec()));
                true
            },
        )
        .unwrap();

        assert_eq!(
            pairs,
            vec![
                (b"node-a:rel".to_vec(), b"edge-1".to_vec()),
                (b"node-a:rel".to_vec(), b"edge-2".to_vec()),
            ],
            "dup scans must expose logical LMDB-style key/value pairs, not SlateDB composite keys"
        );
    }

    /// A logical prefix scan over a dup namespace must read only the rows that
    /// can match (one probe per key-length region), not the whole namespace:
    /// `drop_node_be` runs one per deleted point over every edge in the
    /// collection.
    #[test]
    fn lsm_dup_prefix_scan_reads_only_matching_regions() {
        let be = LsmBackend::open_in_memory("/dup-prefix-scan-test").unwrap();
        let mut w = be.begin_write().unwrap();
        // Keys of several lengths, before/inside/after the prefix in each.
        for (key, value) in [
            (b"n".as_slice(), b"short".as_slice()),
            (b"m-long-key", b"before-long"),
            (b"n1", b"a"),
            (b"n1", b"b"),
            (b"n1-suffix", b"c"),
            (b"n1-suffix-more", b"d"),
            (b"n2", b"other"),
            (b"n2-long-key", b"after-long"),
            (b"zz", b"tail"),
        ] {
            be.put_dup(&mut w, Namespace::OutEdges, key, value).unwrap();
        }
        for i in 0..200u32 {
            be.put_dup(&mut w, Namespace::OutEdges, b"n9", &i.to_be_bytes())
                .unwrap();
        }
        be.commit(w).unwrap();

        let r = be.begin_read().unwrap();
        be.reset_read_count();
        let mut pairs = Vec::new();
        be.scan(&r, Namespace::OutEdges, KeyRange::prefix(b"n1"), |k, v| {
            pairs.push((k.to_vec(), v.to_vec()));
            true
        })
        .unwrap();
        assert_eq!(
            pairs,
            vec![
                (b"n1".to_vec(), b"a".to_vec()),
                (b"n1".to_vec(), b"b".to_vec()),
                (b"n1-suffix".to_vec(), b"c".to_vec()),
                (b"n1-suffix-more".to_vec(), b"d".to_vec()),
            ]
        );
        // 4 matches plus at most one out-of-range probe per key length visited.
        assert!(
            be.read_count() <= 4 + 6,
            "prefix scan read {} rows for 4 matches",
            be.read_count()
        );

        be.reset_read_count();
        let mut misses = 0;
        be.scan(&r, Namespace::OutEdges, KeyRange::prefix(b"n5"), |_, _| {
            misses += 1;
            true
        })
        .unwrap();
        assert_eq!(misses, 0);
        assert!(
            be.read_count() <= 6,
            "empty prefix scan read {} rows",
            be.read_count()
        );
    }

    /// The batched prefix scan must return what the one-prefix scan returns,
    /// in the same order, while reading only each prefix's own rows plus one
    /// probe per key length present in the namespace. With a row cap it
    /// returns the first values of that order and pulls no further.
    #[test]
    fn lsm_dup_prefix_batch_scan_matches_the_one_prefix_scan() {
        let be = LsmBackend::open_in_memory("/dup-prefix-batch-scan-test").unwrap();
        let mut w = be.begin_write().unwrap();
        for (key, value) in [
            (b"n".as_slice(), b"short".as_slice()),
            (b"m-long-key", b"before-long"),
            (b"n1", b"a"),
            (b"n1", b"b"),
            (b"n1-suffix", b"c"),
            (b"n1-suffix-more", b"d"),
            (b"n2", b"other"),
            (b"n2-long-key", b"after-long"),
            (b"zz", b"tail"),
        ] {
            be.put_dup(&mut w, Namespace::OutEdges, key, value).unwrap();
        }
        for i in 0..200u32 {
            be.put_dup(&mut w, Namespace::OutEdges, b"n9", &i.to_be_bytes())
                .unwrap();
        }
        // A row too short to hold a key length (no dup write produces one):
        // both scans step over it.
        be.put(&mut w, Namespace::OutEdges, &[0, 0, 1], b"")
            .unwrap();
        be.commit(w).unwrap();

        let r = be.begin_read().unwrap();
        let prefixes: Vec<Vec<u8>> = [
            b"n1".as_slice(),
            b"n5",
            b"n9",
            b"n",
            b"zz",
            b"n1-suffix",
            b"",
            b"a-prefix-longer-than-any-key",
        ]
        .iter()
        .map(|prefix| prefix.to_vec())
        .collect();
        let one_at_a_time: Vec<Vec<Vec<u8>>> = prefixes
            .iter()
            .map(|prefix| {
                let mut values = Vec::new();
                be.scan(&r, Namespace::OutEdges, KeyRange::prefix(prefix), |_, v| {
                    values.push(v.to_vec());
                    true
                })
                .unwrap();
                values
            })
            .collect();
        assert_eq!(one_at_a_time[0].len(), 4, "fixture: n1 matches 4 rows");
        assert_eq!(one_at_a_time[6].len(), 209, "fixture: the empty prefix");

        be.reset_read_count();
        let batched = be
            .scan_dup_prefix_values_many_with(&r, Namespace::OutEdges, &prefixes, None)
            .unwrap()
            .expect("a snapshot without a write overlay is served in one batch");
        assert_eq!(batched, one_at_a_time);
        let matched: usize = batched.iter().map(Vec::len).sum();
        // The fixture holds keys of six lengths: one probe row for each, and
        // the probe steps over the short row.
        assert!(
            be.read_count() <= matched + 7,
            "batched prefix scans read {} rows for {matched} matches",
            be.read_count()
        );

        // Capped, a prefix keeps the first values of its full read, across
        // key lengths (`n1` has rows under three), and the rest of a long
        // prefix (`n9`, the empty prefix) is never pulled. A cap of 0 reads
        // as 1: it must not answer "no row" for a prefix that has one.
        for (cap, kept) in [(0usize, 1usize), (1, 1), (3, 3)] {
            be.reset_read_count();
            let capped = be
                .scan_dup_prefix_values_many_with(&r, Namespace::OutEdges, &prefixes, Some(cap))
                .unwrap()
                .expect("a capped read is served in the same batch");
            let first: Vec<Vec<Vec<u8>>> = one_at_a_time
                .iter()
                .map(|values| values.iter().take(kept).cloned().collect())
                .collect();
            assert_eq!(capped, first, "cap {cap}");
            assert!(
                be.read_count() <= prefixes.len() * kept + 7,
                "capped prefix scans read {} rows under a cap of {cap}",
                be.read_count()
            );
        }

        let none: Vec<Vec<u8>> = Vec::new();
        assert_eq!(
            be.scan_dup_prefix_values_many_with(&r, Namespace::OutEdges, &none, None)
                .unwrap(),
            Some(Vec::new())
        );
        // Plain namespaces and write overlays stay with the one-prefix scan.
        assert!(be
            .scan_dup_prefix_values_many_with(&r, Namespace::Nodes, &prefixes, None)
            .unwrap()
            .is_none());
        let mut w = be.begin_write().unwrap();
        be.put_dup(&mut w, Namespace::OutEdges, b"n5", b"unwritten")
            .unwrap();
        let overlay = be.begin_read_with_pending(w.pending().clone()).unwrap();
        assert!(be
            .scan_dup_prefix_values_many_with(&overlay, Namespace::OutEdges, &prefixes, Some(1))
            .unwrap()
            .is_none());
    }

    #[test]
    fn lsm_exact_dup_scan_limits_reads_and_preserves_pending_changes() {
        let be = LsmBackend::open_in_memory("/exact-dup-scan-test").unwrap();
        let mut w = be.begin_write().unwrap();
        for (key, value) in [
            (b"".as_slice(), b"empty".as_slice()),
            (b"n1", b"a"),
            (b"n1", b"b"),
            (b"n10", b"longer"),
            (b"n2", b"other"),
        ] {
            be.put_dup(&mut w, Namespace::OutEdges, key, value).unwrap();
        }
        be.commit(w).unwrap();
        let exact = |key: &[u8]| KeyRange {
            start: Bound::Included(key.to_vec()),
            end: Bound::Included(key.to_vec()),
        };
        let read_values = |r: &LsmRead, key: &[u8]| {
            let mut values = Vec::new();
            be.scan(r, Namespace::OutEdges, exact(key), |actual_key, value| {
                assert_eq!(actual_key, key);
                values.push(value.to_vec());
                true
            })
            .unwrap();
            values
        };
        let r = be.begin_read().unwrap();
        be.reset_read_count();
        assert_eq!(read_values(&r, b"n1"), vec![b"a".to_vec(), b"b".to_vec()]);
        assert_eq!(be.read_count(), 2);
        be.reset_read_count();
        assert!(read_values(&r, b"missing").is_empty());
        assert_eq!(be.read_count(), 0);
        assert_eq!(read_values(&r, b""), vec![b"empty".to_vec()]);

        let mut w = be.begin_write().unwrap();
        be.delete_dup(&mut w, Namespace::OutEdges, b"n1", b"a")
            .unwrap();
        be.put_dup(&mut w, Namespace::OutEdges, b"n1", b"c")
            .unwrap();
        be.put_dup(&mut w, Namespace::OutEdges, b"n11", b"unrelated")
            .unwrap();
        let pending = be.begin_read_with_pending(w.pending().clone()).unwrap();
        be.reset_read_count();
        assert_eq!(
            read_values(&pending, b"n1"),
            vec![b"b".to_vec(), b"c".to_vec()]
        );
        assert_eq!(
            be.read_count(),
            2,
            "pending overlay must not widen the backend scan"
        );
        assert_eq!(
            read_values(&r, b"n1"),
            vec![b"a".to_vec(), b"b".to_vec()],
            "pinned reader must keep its original view"
        );
        let mut calls = 0;
        be.scan(&r, Namespace::OutEdges, exact(b"n1"), |_, _| {
            calls += 1;
            false
        })
        .unwrap();
        assert_eq!(calls, 1, "visitor early exit is preserved");
    }

    /// `collections_root_prefix` parses the collections root from a SlateDB
    /// collection path (the part before the final `/<name>` segment).
    #[test]
    fn collections_root_prefix_strips_collection_segment() {
        assert_eq!(collections_root_prefix("helion/my_collection"), "helion");
        assert_eq!(collections_root_prefix("shadow/team/colA"), "shadow/team");
        assert_eq!(collections_root_prefix("bare"), "");
    }

    /// Catalog rediscovery primitive: several collections written to ONE shared
    /// object store (each at `<root>/<name>`) must be enumerable from that store
    /// via the immediate children of the root — both through the standalone
    /// `list_child_collection_names` core AND through a loaded writer's
    /// `LsmBackend::list_collection_prefixes`. A shared `Arc<InMemory>` stands in
    /// for the single S3 bucket every collection's writer commits into; this is
    /// the listing the writer uses to rediscover collections after a fresh start
    /// / PVC loss. (Each `open_in_memory` makes its OWN isolated store, so the
    /// shared-store path is the one that mirrors production S3.)
    #[test]
    fn list_collection_prefixes_enumerates_shared_object_store() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());

        // Three collections committed into the same store under the "helion" root.
        for name in ["alpha", "beta", "gamma"] {
            let be = LsmBackend::open_with_store(&format!("helion/{name}"), store.clone())
                .expect("open writer on shared store");
            let mut w = be.begin_write().unwrap();
            be.put(&mut w, Namespace::Metadata, b"config", b"{}")
                .unwrap();
            be.commit(w).unwrap();
        }

        // Core listing primitive over the shared store at the root prefix.
        let names = list_child_collection_names(&shared_lsm_handle(), &store, "helion")
            .expect("list child collection names");
        assert_eq!(
            names,
            vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()],
            "every collection committed under helion/ must be discoverable from the shared store"
        );

        // The instance method derives the root from its own path and returns the
        // same catalog (a loaded writer enumerates its siblings).
        let alpha = LsmBackend::open_with_store("helion/alpha", store.clone()).unwrap();
        let mut via_instance = alpha.list_collection_prefixes().expect("instance listing");
        via_instance.sort();
        assert_eq!(
            via_instance,
            vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()],
        );
    }

    #[test]
    #[serial]
    fn purge_prefix_from_store_deletes_closed_collection_objects() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let rt = shared_lsm_handle();
        block_on_lsm(&rt, async {
            store
                .put(
                    &ObjPath::from("helion/dead/manifest/0001"),
                    Bytes::from_static(b"manifest").into(),
                )
                .await
                .unwrap();
            store
                .put(
                    &ObjPath::from("helion/dead/compacted/0001.sst"),
                    Bytes::from_static(b"sst").into(),
                )
                .await
                .unwrap();
            store
                .put(
                    &ObjPath::from("helion/live/manifest/0001"),
                    Bytes::from_static(b"keep").into(),
                )
                .await
                .unwrap();
            Ok::<(), String>(())
        })
        .unwrap();

        purge_prefix_from_store(rt.clone(), store.clone(), "helion/dead").unwrap();

        let dead: Vec<ObjPath> = block_on_lsm(&rt, async {
            store
                .list(Some(&ObjPath::from("helion/dead")))
                .map_ok(|meta| meta.location)
                .try_collect()
                .await
                .map_err(|error| error.to_string())
        })
        .unwrap();
        let live: Vec<ObjPath> = block_on_lsm(&rt, async {
            store
                .list(Some(&ObjPath::from("helion/live")))
                .map_ok(|meta| meta.location)
                .try_collect()
                .await
                .map_err(|error| error.to_string())
        })
        .unwrap();

        assert!(dead.is_empty());
        assert_eq!(live, vec![ObjPath::from("helion/live/manifest/0001")]);
    }

    /// Object store that plays the racing reader: on the first delete after the
    /// purge's `append_after_lists`-th LIST it appends `zombie` (the reader's
    /// checkpoint-lease manifest, written from an in-memory copy after that
    /// LIST), and it records every deleted location in order.
    #[derive(Debug)]
    struct RacingReaderStore {
        inner: Arc<dyn ObjectStore>,
        zombie: ObjPath,
        append_after_lists: usize,
        /// Fail every delete once the purge has issued this many LISTs
        /// (`usize::MAX`: never): an object-store error mid-purge.
        fail_deletes_after_lists: usize,
        lists: std::sync::atomic::AtomicUsize,
        appended: std::sync::atomic::AtomicBool,
        deleted: Arc<std::sync::Mutex<Vec<ObjPath>>>,
        /// Delete requests the purge issued (one per `delete_stream` call).
        delete_calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl std::fmt::Display for RacingReaderStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "RacingReaderStore({})", self.inner)
        }
    }

    type Boxed<'a, T> =
        std::pin::Pin<Box<dyn Future<Output = object_store::Result<T>> + Send + 'a>>;

    impl ObjectStore for RacingReaderStore {
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
            self.inner.get_opts(location, options)
        }

        fn delete_stream(
            &self,
            locations: stream::BoxStream<'static, object_store::Result<ObjPath>>,
        ) -> stream::BoxStream<'static, object_store::Result<ObjPath>> {
            self.delete_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if self.lists.load(std::sync::atomic::Ordering::SeqCst) >= self.fail_deletes_after_lists
            {
                return stream::once(async {
                    Err(object_store::Error::Generic {
                        store: "RacingReaderStore",
                        source: "injected delete failure".into(),
                    })
                })
                .boxed();
            }
            let deleted = Arc::clone(&self.deleted);
            let locations = locations
                .inspect(move |location| {
                    if let Ok(location) = location {
                        deleted.lock().unwrap().push(location.clone());
                    }
                })
                .boxed();
            let inner = Arc::clone(&self.inner);
            if self.lists.load(std::sync::atomic::Ordering::SeqCst) < self.append_after_lists
                || self
                    .appended
                    .swap(true, std::sync::atomic::Ordering::SeqCst)
            {
                return inner.delete_stream(locations);
            }
            let zombie = self.zombie.clone();
            stream::once(async move {
                inner
                    .put(&zombie, Bytes::from_static(b"lease").into())
                    .await
                    .unwrap();
                inner.delete_stream(locations)
            })
            .flatten()
            .boxed()
        }

        fn list(
            &self,
            prefix: Option<&ObjPath>,
        ) -> stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
            self.lists.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.list(prefix)
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

    /// Object store that keeps a WAL put waiting while `hold` is set, and with
    /// it the flush that sent it.
    #[derive(Debug, Default)]
    struct HeldWalStore {
        inner: InMemory,
        hold: std::sync::atomic::AtomicBool,
        /// A WAL put has arrived and waits.
        held: std::sync::atomic::AtomicBool,
        release: tokio::sync::Notify,
    }

    impl std::fmt::Display for HeldWalStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "HeldWalStore({})", self.inner)
        }
    }

    impl ObjectStore for HeldWalStore {
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
            Box::pin(async move {
                if location.as_ref().contains("/wal/")
                    && self.hold.load(std::sync::atomic::Ordering::SeqCst)
                {
                    self.held.store(true, std::sync::atomic::Ordering::SeqCst);
                    self.release.notified().await;
                }
                self.inner.put_opts(location, payload, opts).await
            })
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
            self.inner.get_opts(location, options)
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
            self.inner.list(prefix)
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

    /// A write that lands while another caller's flush is in flight is not part
    /// of that flush: its own caller's flush makes it durable, and that flush has
    /// to tell the readers. The first flush published before the write was on
    /// the store.
    #[test]
    #[serial]
    fn a_write_that_lands_during_a_flush_is_published_by_its_own_flush() {
        use std::sync::atomic::Ordering::SeqCst;
        let store = Arc::new(HeldWalStore::default());
        // No flush on a timer: only `flush_durable` writes the WAL here.
        let mut settings = Settings::default();
        settings.flush_interval = None;
        let be = Arc::new(
            LsmBackend::open_with_store_settings(
                "helion/flush-overlap",
                Arc::clone(&store) as Arc<dyn ObjectStore>,
                Some(settings),
            )
            .unwrap(),
        );
        let feed = crate::helix_engine::storage_core::change_feed::change_feed();
        let cursor = || feed.changes_since(u64::MAX).cursor;
        let published_since = |cursor: u64| {
            feed.changes_since(cursor)
                .changes
                .iter()
                .any(|name| name == "flush-overlap")
        };
        let write_buffered = |key: &[u8]| {
            let mut w = be.begin_write().unwrap();
            be.put(&mut w, Namespace::Nodes, key, b"1").unwrap();
            be.commit_buffered(w).unwrap();
        };

        write_buffered(b"first");
        let before = cursor();
        store.hold.store(true, SeqCst);
        let first_flush = {
            let be = Arc::clone(&be);
            std::thread::spawn(move || be.flush_durable())
        };
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while !store.held.load(SeqCst) {
            assert!(
                std::time::Instant::now() < deadline,
                "the flush never reached its WAL put"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        // The first flush has taken what was buffered and waits for the store.
        write_buffered(b"second");
        store.hold.store(false, SeqCst);
        store.release.notify_one();
        first_flush.join().unwrap().unwrap();
        assert!(published_since(before), "the first flush published nothing");

        let after_first = cursor();
        be.flush_durable().unwrap();
        assert!(
            published_since(after_first),
            "the second write became durable and the readers were not told"
        );
    }

    /// A flush that is refused before it is sent made nothing durable: the write
    /// it would have published is published by the flush that does.
    #[test]
    #[serial]
    fn a_refused_flush_leaves_its_publish_to_the_next_flush() {
        let be = LsmBackend::open_in_memory("helion/flush-refused").unwrap();
        let feed = crate::helix_engine::storage_core::change_feed::change_feed();
        let mut w = be.begin_write().unwrap();
        be.put(&mut w, Namespace::Nodes, b"k", b"1").unwrap();
        be.commit_buffered(w).unwrap();

        // Inside a runtime and without the blocking allowance, the flush is
        // refused before it reaches the store.
        let refused = Runtime::new()
            .unwrap()
            .block_on(async { be.flush_durable() });
        assert!(
            matches!(refused, Err(BackendError::Unsupported(_))),
            "{refused:?}"
        );

        let before = feed.changes_since(u64::MAX).cursor;
        be.flush_durable().unwrap();
        assert!(
            feed.changes_since(before)
                .changes
                .iter()
                .any(|name| name == "flush-refused"),
            "the write became durable and the readers were not told"
        );
    }

    fn racing_reader_purge(
        prefix: &str,
        append_after_lists: usize,
    ) -> (Arc<std::sync::Mutex<Vec<ObjPath>>>, Vec<ObjPath>) {
        let (deleted, left, purged) =
            racing_reader_purge_opts(prefix, append_after_lists, usize::MAX);
        purged.unwrap();
        (deleted, left)
    }

    /// Like [`racing_reader_purge`], but deletes fail once the purge has issued
    /// `fail_deletes_after_lists` LISTs, and the purge's result is returned.
    fn racing_reader_purge_opts(
        prefix: &str,
        append_after_lists: usize,
        fail_deletes_after_lists: usize,
    ) -> (
        Arc<std::sync::Mutex<Vec<ObjPath>>>,
        Vec<ObjPath>,
        Result<(), BackendError>,
    ) {
        let rt = shared_lsm_handle();
        let inner: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        block_on_lsm(&rt, async {
            for object in [
                "compacted/0001.sst",
                "manifest.boundary",
                "manifest/00000000000000000001.manifest",
                "manifest/00000000000000000002.manifest",
                "wal/0001.sst",
            ] {
                inner
                    .put(
                        &ObjPath::from(format!("{prefix}/{object}")),
                        Bytes::from_static(b"x").into(),
                    )
                    .await
                    .unwrap();
            }
            Ok::<(), String>(())
        })
        .unwrap();
        let deleted = Arc::new(std::sync::Mutex::new(Vec::new()));
        let store: Arc<dyn ObjectStore> = Arc::new(RacingReaderStore {
            inner: Arc::clone(&inner),
            zombie: ObjPath::from(format!("{prefix}/manifest/00000000000000000003.manifest")),
            append_after_lists,
            fail_deletes_after_lists,
            lists: std::sync::atomic::AtomicUsize::new(0),
            appended: std::sync::atomic::AtomicBool::new(false),
            deleted: Arc::clone(&deleted),
            delete_calls: Arc::default(),
        });

        let purged = purge_prefix_from_store(rt.clone(), store, prefix);

        let left: Vec<ObjPath> = block_on_lsm(&rt, async {
            inner
                .list(Some(&ObjPath::from(prefix)))
                .map_ok(|meta| meta.location)
                .try_collect()
                .await
                .map_err(|error| error.to_string())
        })
        .unwrap();
        (deleted, left, purged)
    }

    /// A reader's lease append that lands after the purge's LIST must not
    /// survive the purge as a manifest pointing at deleted SSTs.
    #[test]
    #[serial]
    fn purge_prefix_from_store_sweeps_manifest_appended_during_purge() {
        let (_, left) = racing_reader_purge("helion/racing", 1);
        assert!(
            left.is_empty(),
            "zombie manifest survived the purge: {left:?}"
        );
    }

    /// Same race one pass later: an append that lands after the second LIST
    /// (during the second sweep) needs the third pass to catch it.
    #[test]
    #[serial]
    fn purge_prefix_from_store_sweeps_manifest_appended_during_second_pass() {
        let (_, left) = racing_reader_purge("helion/racing-late", 2);
        assert!(
            left.is_empty(),
            "manifest appended during the second sweep survived: {left:?}"
        );
    }

    /// A purge that empties `manifest/` and then fails must still remember the
    /// ids it deleted: the retry worker finds no manifests, so without this the
    /// recreate would start at 1 and a stale lease append could become latest.
    #[test]
    #[serial]
    fn purge_prefix_from_store_records_manifest_ceiling_before_a_failed_pass() {
        let prefix = "helion/ceiling-on-failure";
        let (_, left, purged) = racing_reader_purge_opts(prefix, usize::MAX, 2);
        assert!(purged.is_err(), "the second pass was meant to fail");
        assert!(
            !left.is_empty(),
            "the failed pass must leave objects behind"
        );
        assert_eq!(purged_manifest_ceiling(prefix), 2);
    }

    fn manifest_ids(rt: &Handle, store: &Arc<dyn ObjectStore>, path: &str) -> Vec<u64> {
        block_on_lsm(rt, async {
            store
                .list(Some(&ObjPath::from(format!("{path}/manifest"))))
                .map_ok(|meta| meta.location)
                .try_collect::<Vec<_>>()
                .await
                .map_err(|error| error.to_string())
        })
        .unwrap()
        .iter()
        .filter_map(manifest_id_of)
        .collect()
    }

    /// Drop + recreate of the same name in one writer process: the new database
    /// must start its manifest ids above the purged incarnation's, so a late
    /// lease append from a reader still holding the OLD manifest (its `N + 1`)
    /// can never become the new database's latest manifest.
    #[test]
    #[serial]
    fn recreated_collection_manifest_ids_start_above_purged_incarnation() {
        use crate::helix_engine::storage_core::backend_any::AnyBackend;
        use crate::helix_engine::storage_core::backend_lsm_reader::LsmReader;
        use object_store::{PutMode, PutOptions};

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let rt = shared_lsm_handle();
        let path = "helion/recreate-floor";

        let old = LsmBackend::open_with_store(path, store.clone()).unwrap();
        let mut w = old.begin_write().unwrap();
        old.put(&mut w, Namespace::Nodes, b"k", b"old").unwrap();
        old.commit(w).unwrap();
        old.close().unwrap();
        drop(old);
        let old_ids = manifest_ids(&rt, &store, path);
        let old_max = *old_ids.iter().max().unwrap();
        // The reader loaded the manifest just below the latest one...
        let stale_id = old_max - 1;
        assert!(old_ids.contains(&stale_id), "old manifest ids: {old_ids:?}");
        let stale_bytes = block_on_lsm(&rt, async {
            store
                .get(&ObjPath::from(format!(
                    "{path}/manifest/{stale_id:020}.manifest"
                )))
                .await
                .map_err(|error| error.to_string())?
                .bytes()
                .await
                .map_err(|error| error.to_string())
        })
        .unwrap();

        purge_prefix_from_store(rt.clone(), store.clone(), path).unwrap();
        assert!(manifest_ids(&rt, &store, path).is_empty());

        let new = LsmBackend::open_with_store(path, store.clone()).unwrap();
        let new_ids = manifest_ids(&rt, &store, path);
        let new_min = *new_ids.iter().min().unwrap();
        assert!(
            new_min > old_max,
            "re-created db must start above the purged ids: old max {old_max}, new {new_ids:?}"
        );
        let mut w = new.begin_write().unwrap();
        new.put(&mut w, Namespace::Nodes, b"k", b"new").unwrap();
        new.commit(w).unwrap();
        let latest = *manifest_ids(&rt, &store, path).iter().max().unwrap();

        // ...and its slow lease append (`stale_id + 1`) lands after the recreate.
        block_on_lsm(&rt, async {
            store
                .put_opts(
                    &ObjPath::from(format!("{path}/manifest/{old_max:020}.manifest")),
                    stale_bytes.into(),
                    PutOptions::from(PutMode::Create),
                )
                .await
                .map_err(|error| error.to_string())?;
            Ok::<(), String>(())
        })
        .unwrap();
        assert_eq!(
            *manifest_ids(&rt, &store, path).iter().max().unwrap(),
            latest,
            "the stale append became the re-created database's latest manifest"
        );
        let reader =
            AnyBackend::LsmReader(LsmReader::open_with_store(path, store.clone()).unwrap());
        let r = reader.begin_read().unwrap();
        let got = reader
            .get_with(&r, Namespace::Nodes, b"k", |v| v.map(|b| b.to_vec()))
            .unwrap();
        assert_eq!(got, Some(b"new".to_vec()));
    }

    /// `manifest/` must be deleted before anything else so a concurrent opener
    /// finds no manifest (404) rather than one whose SSTs are being deleted.
    #[test]
    #[serial]
    fn purge_prefix_from_store_deletes_manifests_first() {
        let (deleted, _) = racing_reader_purge("helion/ordered", 1);
        let deleted = deleted.lock().unwrap();
        assert_eq!(
            deleted[..2],
            [
                ObjPath::from("helion/ordered/manifest/00000000000000000001.manifest"),
                ObjPath::from("helion/ordered/manifest/00000000000000000002.manifest"),
            ],
            "purge order: {deleted:?}"
        );
    }

    /// A purge deletes in multi-object requests, not one request per key: two
    /// manifests and 2,500 other objects are 1 + 3 delete requests, and the
    /// third pass finds nothing left to delete.
    #[test]
    #[serial]
    fn purge_prefix_from_store_deletes_in_bulk_requests() {
        let rt = shared_lsm_handle();
        let prefix = "helion/bulk";
        let inner: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        block_on_lsm(&rt, async {
            let objects = (1..=2)
                .map(|id| format!("manifest/{id:020}.manifest"))
                .chain((0..2_500).map(|id| format!("wal/{id:020}.sst")));
            for object in objects {
                inner
                    .put(
                        &ObjPath::from(format!("{prefix}/{object}")),
                        Bytes::from_static(b"x").into(),
                    )
                    .await
                    .unwrap();
            }
            Ok::<(), String>(())
        })
        .unwrap();
        let deleted = Arc::new(std::sync::Mutex::new(Vec::new()));
        let delete_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let store: Arc<dyn ObjectStore> = Arc::new(RacingReaderStore {
            inner: Arc::clone(&inner),
            zombie: ObjPath::from(format!("{prefix}/never-appended")),
            append_after_lists: usize::MAX,
            fail_deletes_after_lists: usize::MAX,
            lists: std::sync::atomic::AtomicUsize::new(0),
            appended: std::sync::atomic::AtomicBool::new(false),
            deleted: Arc::clone(&deleted),
            delete_calls: Arc::clone(&delete_calls),
        });

        purge_prefix_from_store(rt.clone(), store, prefix).unwrap();

        assert_eq!(delete_calls.load(std::sync::atomic::Ordering::SeqCst), 4);
        let deleted = deleted.lock().unwrap();
        assert_eq!(deleted.len(), 2_502);
        assert!(
            deleted[..2]
                .iter()
                .all(|location| manifest_id_of(location).is_some()),
            "manifests were not deleted first: {:?}",
            &deleted[..4]
        );
        let left: Vec<ObjPath> = block_on_lsm(&rt, async {
            inner
                .list(Some(&ObjPath::from(prefix)))
                .map_ok(|meta| meta.location)
                .try_collect()
                .await
                .map_err(|error| error.to_string())
        })
        .unwrap();
        assert!(left.is_empty(), "objects survived the purge: {left:?}");
    }

    #[test]
    fn lsm_scan_prefix_within_namespace() {
        let be = LsmBackend::open_in_memory("/helion-test-2").unwrap();
        let mut w = be.begin_write().unwrap();
        be.put(&mut w, Namespace::Metadata, b"app:1", b"x").unwrap();
        be.put(&mut w, Namespace::Metadata, b"app:2", b"x").unwrap();
        be.put(&mut w, Namespace::Metadata, b"zzz:1", b"x").unwrap();
        be.commit(w).unwrap();

        let r = be.begin_read().unwrap();
        let mut keys = Vec::new();
        be.scan(
            &r,
            Namespace::Metadata,
            KeyRange::prefix(b"app:"),
            |k, _v| {
                keys.push(k.to_vec());
                true
            },
        )
        .unwrap();
        assert_eq!(keys, vec![b"app:1".to_vec(), b"app:2".to_vec()]);
    }

    /// Post-txn hooks run with `true` once the batch commits and with `false`
    /// when the batch is dropped uncommitted; a commit never leaves a hook
    /// behind for the drop to run again.
    #[test]
    fn post_txn_hooks_distinguish_commit_from_abort() {
        use std::sync::atomic::{AtomicU8, Ordering};
        let be = LsmBackend::open_in_memory("post-txn-hooks").unwrap();
        let outcome = Arc::new(AtomicU8::new(0));
        let record = |slot: &Arc<AtomicU8>| -> PostTxnHook {
            let slot = Arc::clone(slot);
            Box::new(move |committed| {
                slot.fetch_add(if committed { 1 } else { 10 }, Ordering::SeqCst);
            })
        };

        let mut w = be.begin_write().unwrap();
        be.put(&mut w, Namespace::Nodes, b"k", b"v").unwrap();
        w.on_txn_end(record(&outcome));
        be.commit(w).unwrap();
        assert_eq!(
            outcome.load(Ordering::SeqCst),
            1,
            "commit runs hooks once with true"
        );

        let mut w = be.begin_write().unwrap();
        be.put(&mut w, Namespace::Nodes, b"k2", b"v").unwrap();
        w.on_txn_end(record(&outcome));
        drop(w);
        assert_eq!(
            outcome.load(Ordering::SeqCst),
            11,
            "drop runs hooks with false"
        );

        let mut w = be.begin_write().unwrap();
        be.put(&mut w, Namespace::Nodes, b"k3", b"v").unwrap();
        w.on_txn_end(record(&outcome));
        be.commit_buffered(w).unwrap();
        assert_eq!(
            outcome.load(Ordering::SeqCst),
            12,
            "buffered commit runs hooks with true"
        );
    }

    /// A commit that fails (fenced by a newer writer) runs its hooks with
    /// `false`, so a publication made ahead of it is taken back.
    #[test]
    #[serial]
    fn post_txn_hooks_see_false_when_the_commit_fails() {
        use slatedb::object_store::local::LocalFileSystem;
        use std::sync::atomic::{AtomicU8, Ordering};

        let data_dir = tempfile::TempDir::new().unwrap();
        let store: Arc<dyn ObjectStore> =
            Arc::new(LocalFileSystem::new_with_prefix(data_dir.path()).unwrap());
        let path = "helion/post-txn-fenced";
        let be1 = LsmBackend::open_with_store(path, store.clone()).unwrap();
        let mut first = be1.begin_write().unwrap();
        be1.put(&mut first, Namespace::Nodes, b"a", b"1").unwrap();
        be1.commit_buffered(first).unwrap();

        let be2 = LsmBackend::open_with_store(path, store.clone()).unwrap();
        let mut fencer = be2.begin_write().unwrap();
        be2.put(&mut fencer, Namespace::Nodes, b"b", b"2").unwrap();
        be2.commit(fencer).unwrap();

        let outcome = Arc::new(AtomicU8::new(0));
        let mut stale = be1.begin_write().unwrap();
        be1.put(&mut stale, Namespace::Nodes, b"c", b"3").unwrap();
        let slot = Arc::clone(&outcome);
        stale.on_txn_end(Box::new(move |committed| {
            slot.store(if committed { 1 } else { 2 }, Ordering::SeqCst);
        }));
        be1.commit(stale).expect_err("fenced commit must fail");
        assert_eq!(
            outcome.load(Ordering::SeqCst),
            2,
            "a failed commit must run its hooks with false"
        );
    }

    /// Equivalence oracle for the bounded dup scan (`dup_scan_plan` +
    /// `DupScanCursor`): for random prefix, exact and bounded ranges over keys
    /// of mixed lengths built from 0x00/0x01/0x61/0xFE/0xFF bytes, `scan` must
    /// return exactly what the unbounded `KeyRange::all()` scan (the untouched
    /// code path) returns after the per-row `logical_key_in_range` filter — same
    /// rows, same order, same early-stop prefix. Rows span two flushed L0 SSTs
    /// plus the memtable, with committed tombstones over L0 rows and a pending
    /// write overlay, so the SlateDB `seek` is exercised through every layer the
    /// writer reads. The reader replica is checked against its own full scan.
    #[test]
    fn lsm_dup_scan_matches_full_scan_filter_oracle() {
        use crate::helix_engine::storage_core::backend_lsm_reader::LsmReader;

        const ALPHABET: [u8; 5] = [0x00, 0x01, 0x61, 0xFE, 0xFF];
        struct Rng(u64);
        impl Rng {
            fn below(&mut self, n: usize) -> usize {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                (self.0 % n as u64) as usize
            }
            fn bytes(&mut self, len: usize) -> Vec<u8> {
                (0..len)
                    .map(|_| ALPHABET[self.below(ALPHABET.len())])
                    .collect()
            }
            fn key(&mut self) -> Vec<u8> {
                let mut len = self.below(6);
                if self.below(10) == 0 {
                    len += 6;
                }
                self.bytes(len)
            }
            fn value(&mut self) -> Vec<u8> {
                let len = 1 + self.below(2);
                self.bytes(len)
            }
            fn bound(&mut self) -> Bound<Vec<u8>> {
                match self.below(3) {
                    0 => Bound::Unbounded,
                    1 => Bound::Included(self.key()),
                    _ => Bound::Excluded(self.key()),
                }
            }
        }
        let clone_range = |r: &KeyRange| KeyRange {
            start: r.start.clone(),
            end: r.end.clone(),
        };
        let describe = |r: &KeyRange| format!("{:?}..{:?}", r.start, r.end);

        let namespaces = [Namespace::OutEdges, Namespace::MultiIndex("t")];
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = "/dup-scan-oracle";
        let be = LsmBackend::open_with_store(path, store.clone()).unwrap();
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut rows: Vec<(usize, Vec<u8>, Vec<u8>)> = Vec::new();

        // Three commits: two flushed to L0 SSTs, the last left in the memtable;
        // the last also tombstones a slice of the first commit's rows.
        for round in 0..3 {
            let mut w = be.begin_write().unwrap();
            for _ in 0..220 {
                let (ns_idx, key, value) = (rng.below(namespaces.len()), rng.key(), rng.value());
                be.put_dup(&mut w, namespaces[ns_idx], &key, &value)
                    .unwrap();
                rows.push((ns_idx, key, value));
            }
            if round == 2 {
                for i in (0..rows.len() / 3).step_by(3) {
                    let (ns_idx, key, value) = &rows[i];
                    be.delete_dup(&mut w, namespaces[*ns_idx], key, value)
                        .unwrap();
                }
            }
            be.commit(w).unwrap();
            if round < 2 {
                be.flush_durable().unwrap();
            }
        }

        // Uncommitted overlay: new rows plus deletes of committed rows.
        let mut w = be.begin_write().unwrap();
        for _ in 0..60 {
            let (ns_idx, key, value) = (rng.below(namespaces.len()), rng.key(), rng.value());
            be.put_dup(&mut w, namespaces[ns_idx], &key, &value)
                .unwrap();
        }
        for i in (1..rows.len()).step_by(7) {
            let (ns_idx, key, value) = &rows[i];
            be.delete_dup(&mut w, namespaces[*ns_idx], key, value)
                .unwrap();
        }
        let committed = be.begin_read().unwrap();
        let overlaid = be.begin_read_with_pending(w.pending().clone()).unwrap();
        let reader = LsmReader::open_with_store(path, store).unwrap();

        let mut queries: Vec<KeyRange> = vec![
            KeyRange::prefix(b""),
            KeyRange::prefix(&[0xFF]),
            KeyRange::prefix(&[0xFF, 0xFF]),
            KeyRange::prefix(&[0x61, 0xFF]),
            KeyRange::prefix(&[0x00]),
            KeyRange::prefix(&[
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ]),
            KeyRange::all(),
        ];
        for _ in 0..600 {
            queries.push(match rng.below(4) {
                0 | 1 => {
                    let len = rng.below(8);
                    KeyRange::prefix(&rng.bytes(len))
                }
                2 => {
                    let k = rng.key();
                    KeyRange {
                        start: Bound::Included(k.clone()),
                        end: Bound::Included(k),
                    }
                }
                _ => KeyRange {
                    start: rng.bound(),
                    end: rng.bound(),
                },
            });
        }

        let mut reader_rows = 0usize;
        for ns in namespaces {
            let scan_all = |scan: &dyn Fn(KeyRange, &mut dyn FnMut(&[u8], &[u8]) -> bool)| {
                let mut all = Vec::new();
                scan(KeyRange::all(), &mut |k, v| {
                    all.push((k.to_vec(), v.to_vec()));
                    true
                });
                all
            };
            let writer_committed =
                |range: KeyRange, visit: &mut dyn FnMut(&[u8], &[u8]) -> bool| {
                    be.scan(&committed, ns, range, |k, v| visit(k, v)).unwrap()
                };
            let writer_overlaid = |range: KeyRange, visit: &mut dyn FnMut(&[u8], &[u8]) -> bool| {
                be.scan(&overlaid, ns, range, |k, v| visit(k, v)).unwrap()
            };
            let replica = |range: KeyRange, visit: &mut dyn FnMut(&[u8], &[u8]) -> bool| {
                reader.scan_dup_with(ns, range, |k, v| visit(k, v)).unwrap()
            };
            let lanes: [(
                &str,
                &dyn Fn(KeyRange, &mut dyn FnMut(&[u8], &[u8]) -> bool),
            ); 3] = [
                ("writer", &writer_committed),
                ("writer+pending", &writer_overlaid),
                ("reader", &replica),
            ];
            for (lane, scan) in lanes {
                let all = scan_all(scan);
                assert!(all.len() > 100, "{lane} sees only {} rows", all.len());
                if lane == "reader" {
                    reader_rows += all.len();
                }
                for (i, range) in queries.iter().enumerate() {
                    let expected: Vec<_> = all
                        .iter()
                        .filter(|(k, _)| logical_key_in_range(k, range))
                        .cloned()
                        .collect();
                    let mut actual = Vec::new();
                    be.reset_read_count();
                    scan(clone_range(range), &mut |k, v| {
                        actual.push((k.to_vec(), v.to_vec()));
                        true
                    });
                    assert_eq!(
                        actual,
                        expected,
                        "{lane} ns={ns:?} query #{i} {} differs from full-scan oracle",
                        describe(range)
                    );
                    // A non-empty prefix must seek through L0 SSTs and the
                    // memtable alike: at most two probes per key length (12
                    // lengths here) on top of its hits, never the namespace.
                    if lane == "writer" {
                        if let (Bound::Included(p), Bound::Excluded(end)) =
                            (&range.start, &range.end)
                        {
                            if !p.is_empty() && next_prefix(p).as_deref() == Some(end.as_slice()) {
                                assert!(
                                    be.read_count() <= expected.len() + 2 * 12,
                                    "{lane} ns={ns:?} query #{i} {} read {} rows for {} hits",
                                    describe(range),
                                    be.read_count(),
                                    expected.len()
                                );
                            }
                        }
                    }
                    // Early stop must hand back the same leading rows.
                    let limit = i % 4;
                    let mut limited = Vec::new();
                    scan(clone_range(range), &mut |k, v| {
                        limited.push((k.to_vec(), v.to_vec()));
                        limited.len() < limit
                    });
                    let want = &expected[..expected.len().min(limit.max(1))];
                    assert_eq!(
                        limited,
                        if expected.is_empty() {
                            vec![]
                        } else {
                            want.to_vec()
                        },
                        "{lane} ns={ns:?} query #{i} {} early-stop differs",
                        describe(range)
                    );
                }
            }
        }
        assert!(reader_rows > 0);
        drop(w);
    }
}
