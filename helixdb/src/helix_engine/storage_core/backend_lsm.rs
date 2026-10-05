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
use slatedb::db_cache::{DbCache, SplitCache};
use slatedb::object_store::memory::InMemory;
use slatedb::object_store::{ObjectStore, ObjectStoreExt};
use slatedb::{
    BloomFilterPolicy, CloseReason, Db, DbSnapshot, FilterPolicy, IsolationLevel, MergeOperator,
    MergeOperatorError, PrefixExtractor, PrefixTarget, WriteBatch,
};
use std::sync::OnceLock;
use tokio::runtime::{Builder, Handle, Runtime};

use crate::helix_engine::types::GraphError;

use super::backend::{
    is_dup, next_prefix, BackendError, KeyRange, Namespace, ReadTxn, StorageBackend, WriteTxn,
};
use super::backend_lmdb::ns_name;
use super::metadata::{
    apply_lsm_counter_delta, decode_lsm_counter_delta, decode_lsm_counter_value,
    encode_lsm_counter_value,
};

thread_local! {
    static LSM_BLOCKING_ALLOWED: Cell<bool> = const { Cell::new(false) };
    /// Cooperative cancellation for foreground requests served by either an
    /// LSM reader replica or the writer's local `LsmBackend`. The async gateway
    /// installs this only for read-like routes; durable writer operations
    /// deliberately never receive a token.
    static LSM_READ_CANCELLATION: RefCell<Option<LsmReadCancellation>> = const { RefCell::new(None) };
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
/// WAL buffer flushes at least this often; `await_durable` writes still trigger
/// an immediate flush, so this mainly bounds how long a BUFFERED (group-commit)
/// write may sit before the next periodic flush. Unset → SlateDB default
/// (100 ms). Durability is unaffected (the group-commit barrier `db.flush()`
/// forces durability before ack regardless of this interval).
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
/// tier (`ENV_READER_IDLE_POLL_MS`) regardless of write-feed activity — most
/// resident collections are idle from a READ standpoint even when they are
/// being written (e.g. mid-reindex with nobody querying yet). Promotion back
/// to the active tier (`ENV_READER_MANIFEST_POLL_MS`) happens synchronously
/// on the next read. `0` disables read-gated tiering entirely: collections
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
/// memtable has not reached `l0_sst_size_bytes`. Lower values reduce reader WAL
/// replay/cold-open work for low-write collections. SlateDB requires >= 4096.
const ENV_MAX_WAL_FLUSHES_BEFORE_L0_FLUSH: &str = "HELIX_LSM_MAX_WAL_FLUSHES_BEFORE_L0_FLUSH";
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
const ENV_MULTI_GET: &str = "HELIX_LSM_MULTI_GET";
const LSM_METADATA_SIDECAR_OBJECT: &str = "_helix/metadata.json";

/// Whether batched point reads use SlateDB `multi_get` ([`ENV_MULTI_GET`],
/// default true). `pub(crate)` so the reader replica honors the same knob.
pub(crate) fn lsm_multi_get_enabled() -> bool {
    env_bool_opt(ENV_MULTI_GET).unwrap_or(true)
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
            let block = FoyerCache::new_with_opts(FoyerCacheOptions {
                max_capacity: block_bytes as u64,
                ..Default::default()
            });
            let meta = FoyerCache::new_with_opts(FoyerCacheOptions {
                max_capacity: meta_bytes as u64,
                ..Default::default()
            });
            Arc::new(
                SplitCache::new()
                    .with_block_cache(Some(Arc::new(block)))
                    .with_meta_cache(Some(Arc::new(meta)))
                    .build(),
            ) as Arc<dyn DbCache>
        })
        .clone()
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
        _key: &Bytes,
        existing_value: Option<Bytes>,
        operand: Bytes,
    ) -> Result<Bytes, MergeOperatorError> {
        let value = match existing_value {
            Some(bytes) => decode_lsm_counter_value(&bytes).map_err(Self::callback_error)?,
            None => 0,
        };
        let delta = decode_lsm_counter_delta(&operand).map_err(Self::callback_error)?;
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

fn build_s3_store(config: &S3StoreConfig<'_>) -> Result<Arc<dyn ObjectStore>, BackendError> {
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
/// down, so the sum is ≤ total, never above.)
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
    // the per-collection subdirs sum to <= total_cap (see CAP MATH above).
    // Floored at 1 byte so a huge max_open never yields a zero (no-op) cap.
    let per_collection_cap = (total_cap / max_open_collections_for_cache()).max(1);
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
/// ([`ENV_COMPACTOR_WORKER_POLL_MS`]), and `l0_max_ssts` ([`ENV_L0_MAX_SSTS`]).
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
    // With the writer change feed enabled, fresh opens take the IDLE tier; the
    // feed poller promotes changed collections to the serving cadence.
    if !reader_cold_open_starts_fast() {
        options.manifest_poll_interval = reader_idle_poll_interval_from_env();
    }
    clamp_reader_checkpoint_lifetime(&mut options);
    options
}

/// True if a freshly cold-opened reader-replica `DbReader` starts on the fast
/// (serving) poll tier — i.e. the write-feed is NOT configured, so
/// [`reader_options_from_env`] (which shares this exact check) has no reason
/// to cold-open on the idle safety net instead. `HelixGraphStorage` seeds its
/// `reader_poll_tier_fast` belief from this at construction, so it starts in
/// sync with the `DbReader`'s actual poll interval rather than always
/// assuming "fast" (which is wrong — and makes read-gated promotion inert —
/// whenever the write-feed cold-opens new collections idle).
pub(crate) fn reader_cold_open_starts_fast() -> bool {
    writer_feed_url_from_env().is_none()
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

/// SlateDB rejects a `DbReader` open unless `checkpoint_lifetime` is at least
/// double `manifest_poll_interval` (the self-refreshing checkpoint must
/// outlive the poll that renews it). The default lifetime is 600s, so any
/// poll interval above 300s — e.g. the idle feed tier — must raise it too.
pub(crate) fn clamp_reader_checkpoint_lifetime(options: &mut DbReaderOptions) {
    let min_lifetime = options.manifest_poll_interval.saturating_mul(2);
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
        Builder::new_multi_thread()
            .worker_threads(threads)
            .thread_name("helix-lsm")
            .enable_all()
            .build()
            .unwrap_or_else(|error| {
                eprintln!("failed to build shared LSM tokio runtime: {error}");
                std::process::abort();
            })
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
    let cancelled = LSM_READ_CANCELLATION.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(LsmReadCancellation::is_cancelled_inner)
    });
    if cancelled {
        Err(lsm_read_cancelled_error())
    } else {
        Ok(())
    }
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

fn list_drop_tombstones(
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
    Ok(names)
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
    let store = build_s3_store(&config)?;
    list_child_collection_names(&shared_lsm_handle(), &store, root_prefix)
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
}
impl ReadTxn for LsmRead {}

impl LsmRead {
    /// Build a read handle whose point reads first consult `pending`
    /// (read-your-writes overlay derived from a live write batch).
    pub(crate) fn with_pending(
        snapshot: Arc<DbSnapshot>,
        pending: HashMap<Vec<u8>, Option<Bytes>>,
    ) -> Self {
        Self { snapshot, pending }
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
}
impl WriteTxn for LsmWrite {}

impl LsmWrite {
    /// The batch's buffered effects (full SlateDB key -> Some(value)|None),
    /// cloned into a read view to give read-your-writes (see `LsmRead`).
    pub(crate) fn pending(&self) -> &HashMap<Vec<u8>, Option<Bytes>> {
        &self.pending
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
        let db = Self::open_db_for_request(&rt, path, store.clone(), settings.clone())?;
        Ok(Self {
            db: RwLock::new(db),
            rt,
            store,
            path: path.to_string(),
            settings,
            buffered_since_flush: std::sync::atomic::AtomicBool::new(false),
            #[cfg(test)]
            read_count: std::sync::atomic::AtomicUsize::new(0),
        })
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
        let mut builder = Db::builder(path, store)
            .with_db_cache(cache)
            .with_merge_operator(shared_merge_operator());
        if let Some(settings) = settings {
            builder = builder.with_settings(settings);
        }
        if let Some(policies) = prefix_filter_policies_from_env() {
            builder = builder.with_filter_policies(policies);
        }
        block_on_lsm(rt, builder.build())
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
        let snapshot = match block_on_lsm_read_mapped(&self.rt, db.snapshot(), map_slatedb_err) {
            Ok(snapshot) => snapshot,
            Err(error) => return Err(error),
        };
        Ok(LsmRead::with_pending(snapshot, pending))
    }

    pub(crate) fn collect_values_many_with(
        &self,
        txn: &LsmRead,
        ns: Namespace<'_>,
        keys: &[Vec<u8>],
    ) -> Result<Vec<Option<Vec<u8>>>, BackendError> {
        self.bump_reads(keys.len());
        let mut out = vec![None; keys.len()];
        let mut snapshot_reads = Vec::new();

        for (index, key) in keys.iter().enumerate() {
            let full = prefixed(ns, key);
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
        if lsm_multi_get_enabled() {
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
        let db = self.db();
        let full = prefixed(ns, key);
        for attempt in 0..LSM_METADATA_TXN_MAX_ATTEMPTS {
            let result = block_on_lsm_mapped(
                &self.rt,
                async {
                    let txn = db
                        .begin(IsolationLevel::SerializableSnapshot)
                        .await
                        .map_err(map_slatedb_err)?;
                    let current = txn.get(full.as_slice()).await.map_err(map_slatedb_err)?;
                    let (next, result) = update(current.as_deref())?;
                    txn.put(full.as_slice(), next.as_slice())
                        .map_err(map_slatedb_err)?;
                    txn.commit().await.map_err(map_slatedb_err)?;
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
        let mut builder = AmazonS3Builder::from_env().with_bucket_name(bucket);
        if let Some(r) = region {
            builder = builder.with_region(r);
        }
        if let Some(endpoint) = endpoint {
            builder = builder.with_endpoint(endpoint);
        }
        if allow_http {
            builder = builder.with_allow_http(true);
        }
        builder = apply_s3_timeout(builder);
        let store = builder.build().map_err(io)?;
        Self::open_with_store(prefix, Arc::new(store))
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

    fn scan_dup_namespace_with_options(
        &self,
        r: &LsmRead,
        ns: Namespace<'_>,
        range: KeyRange,
        scan_options: ScanOptions,
        mut visit: impl FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        let prefix = ns_prefix(ns);
        let lo: Bound<Vec<u8>> = Bound::Included(prefix.clone());
        let hi: Bound<Vec<u8>> = match next_prefix(&prefix) {
            Some(end) => Bound::Excluded(end),
            None => Bound::Unbounded,
        };
        let scan_prefix = if range_covers_all_logical_keys(&range) {
            Some(prefix.clone())
        } else {
            None
        };

        if r.pending.is_empty() {
            block_on_lsm_read_mapped(
                &self.rt,
                async {
                    let mut iter = if let Some(scan_prefix) = scan_prefix {
                        r.snapshot
                            .scan_prefix_with_options(scan_prefix, .., &scan_options)
                            .await
                    } else {
                        r.snapshot.scan_with_options((lo, hi), &scan_options).await
                    }
                    .map_err(io)?;
                    while let Some(kv) = iter.next().await.map_err(io)? {
                        self.bump_reads(1);
                        let (logical_key, logical_value) =
                            match decode_dup_full_key(kv.key.as_ref(), &prefix)? {
                                Some(parts) => parts,
                                None => continue,
                            };
                        if !logical_key_in_range(logical_key, &range) {
                            continue;
                        }
                        if !visit(logical_key, logical_value) {
                            break;
                        }
                    }
                    Ok::<(), BackendError>(())
                },
                |e| e,
            )?;
            return Ok(());
        }

        let mut pending_rows: BTreeMap<Vec<u8>, (Vec<u8>, Vec<u8>)> = BTreeMap::new();
        for (full_key, pending) in &r.pending {
            if pending.is_none() {
                continue;
            }
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
                (logical_key.to_vec(), logical_value.to_vec()),
            );
        }

        block_on_lsm_read_mapped(
            &self.rt,
            async {
                let mut pending_iter = pending_rows.into_iter().peekable();
                let mut iter = if let Some(scan_prefix) = scan_prefix {
                    r.snapshot
                        .scan_prefix_with_options(scan_prefix, .., &scan_options)
                        .await
                } else {
                    r.snapshot.scan_with_options((lo, hi), &scan_options).await
                }
                .map_err(io)?;
                while let Some(kv) = iter.next().await.map_err(io)? {
                    self.bump_reads(1);
                    let full_key = kv.key.as_ref();
                    while let Some((pending_key, _pending_value)) = pending_iter.peek() {
                        if pending_key.as_slice() >= full_key {
                            break;
                        }
                        let Some((_pending_key, (logical_key, logical_value))) =
                            pending_iter.next()
                        else {
                            break;
                        };
                        if !visit(&logical_key, &logical_value) {
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
                    if !logical_key_in_range(logical_key, &range) {
                        continue;
                    }
                    if !visit(logical_key, logical_value) {
                        return Ok(());
                    }
                }
                for (_full_key, (logical_key, logical_value)) in pending_iter {
                    if !visit(&logical_key, &logical_value) {
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
        visit: impl FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        self.scan_dup_namespace_with_options(r, ns, range, lsm_scan_options(), visit)
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
            return self.scan_dup_namespace_with_options(txn, ns, range, scan_options, visit);
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
    block_on_lsm(&rt, async move {
        let locations: Vec<ObjPath> = store
            .list(Some(&prefix))
            .map_ok(|meta| meta.location)
            .try_collect()
            .await
            .map_err(|e| e.to_string())?;
        for loc in &locations {
            store.delete(loc).await.map_err(|e| e.to_string())?;
        }
        Ok::<(), String>(())
    })?;

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
        let snapshot = match block_on_lsm_read_mapped(&self.rt, db.snapshot(), map_slatedb_err) {
            Ok(snapshot) => snapshot,
            Err(error) => return Err(error),
        };
        Ok(LsmRead {
            snapshot,
            pending: HashMap::new(),
        })
    }

    fn begin_write(&self) -> Result<Self::Write<'_>, BackendError> {
        Ok(LsmWrite {
            batch: WriteBatch::new(),
            pending: HashMap::new(),
            has_writes: false,
        })
    }

    fn commit(&self, txn: Self::Write<'_>) -> Result<(), BackendError> {
        if !txn.has_writes {
            return Ok(());
        }
        // `Db::write` returns a `WriteHandle` (durability tracker); we await the
        // batch durably and discard it. Classify fencing/CAS as Conflict.
        let batch = txn.batch;
        let db = self.db();
        match block_on_lsm_mapped(&self.rt, db.write(batch.clone()), map_slatedb_err) {
            Ok(_handle) => {
                self.publish_durable_commit();
                Ok(())
            }
            Err(error) if Self::is_fenced_conflict(&error) => {
                let message = lsm_conflict_message(&error).unwrap_or("").to_string();
                Err(lsm_fenced_error("commit", message))
            }
            Err(error) if Self::is_conflict(&error) => {
                block_on_lsm_mapped(&self.rt, db.write(batch), map_slatedb_err)
                    .map(|_handle| self.publish_durable_commit())
            }
            Err(error) => Err(error),
        }
    }

    /// Group-commit half: apply the batch to the memtable (visible to later
    /// reads) but DON'T wait for the WAL to reach the object store. `Db::write`
    /// with `await_durable: false` returns once the single-writer event loop has
    /// applied the batch (SlateDB awaits the apply, only skipping the durable
    /// wait — see `write_with_options`), so read-your-writes across buffered
    /// commits holds. Durability is provided later by [`flush_durable`].
    fn commit_buffered(&self, txn: Self::Write<'_>) -> Result<(), BackendError> {
        if !txn.has_writes {
            return Ok(());
        }
        let batch = txn.batch;
        let db = self.db();
        let options = WriteOptions {
            await_durable: false,
            ..Default::default()
        };
        match block_on_lsm_mapped(
            &self.rt,
            db.write_with_options(batch.clone(), &options),
            map_slatedb_err,
        ) {
            Ok(_handle) => {
                self.buffered_since_flush
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
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
                    db.write_with_options(batch, &options),
                    map_slatedb_err,
                )?;
                self.buffered_since_flush
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    /// Group-commit barrier: flush the WAL durably to the object store. One
    /// object-store flush makes ALL prior [`commit_buffered`] writes durable.
    fn flush_durable(&self) -> Result<(), BackendError> {
        let db = self.db();
        let flushed = block_on_lsm_mapped(&self.rt, db.flush(), map_slatedb_err);
        if matches!(self.close_reason(), Some(CloseReason::Fenced)) {
            return Err(lsm_fenced_error(
                "flush_durable",
                "SlateDB writer close reason is Fenced".to_string(),
            ));
        }
        match flushed {
            Ok(()) => {
                // Publish only when buffered writes actually became durable —
                // a no-op barrier must not signal readers.
                if self
                    .buffered_since_flush
                    .swap(false, std::sync::atomic::Ordering::Relaxed)
                {
                    self.publish_durable_commit();
                }
                Ok(())
            }
            Err(error) if Self::is_fenced_conflict(&error) => {
                let message = lsm_conflict_message(&error).unwrap_or("").to_string();
                Err(lsm_fenced_error("flush_durable", message))
            }
            Err(error) => Err(error),
        }
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
        let ck = dup_composite(ns, key, value);
        let empty = Bytes::new();
        txn.batch.put(ck.as_slice(), empty.clone());
        txn.pending.insert(ck, Some(empty));
        txn.has_writes = true;
        Ok(())
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
        mut visit: impl FnMut(&[u8]) -> bool,
    ) -> Result<(), BackendError> {
        let kp = dup_key_prefix(ns, key);
        if txn.pending.is_empty() {
            block_on_lsm_read_mapped(
                &self.rt,
                async {
                    let mut iter = txn
                        .snapshot
                        .scan_prefix_with_options(kp.clone(), .., &lsm_scan_options())
                        .await
                        .map_err(io)?;
                    while let Some(kv) = iter.next().await.map_err(io)? {
                        let k = kv.key.as_ref();
                        let value: &[u8] = if k.len() >= kp.len() {
                            &k[kp.len()..]
                        } else {
                            &[]
                        };
                        if !visit(value) {
                            break;
                        }
                    }
                    Ok::<(), BackendError>(())
                },
                |e| e,
            )?;
            return Ok(());
        }

        let mut pending_rows: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
        for (full_key, pending) in &txn.pending {
            if pending.is_none() {
                continue;
            }
            if !full_key.starts_with(&kp) {
                continue;
            }
            let value = &full_key[kp.len()..];
            pending_rows.insert(full_key.clone(), value.to_vec());
        }

        block_on_lsm_read_mapped(
            &self.rt,
            async {
                let mut pending_iter = pending_rows.into_iter().peekable();
                let mut iter = txn
                    .snapshot
                    .scan_prefix_with_options(kp.clone(), .., &lsm_scan_options())
                    .await
                    .map_err(io)?;
                while let Some(kv) = iter.next().await.map_err(io)? {
                    let k = kv.key.as_ref();
                    while let Some((pending_key, pending_value)) = pending_iter.peek() {
                        if pending_key.as_slice() >= k {
                            break;
                        }
                        if !visit(pending_value) {
                            return Ok(());
                        }
                        pending_iter.next();
                    }
                    if txn.pending.contains_key(k) {
                        continue;
                    }
                    let value: &[u8] = if k.len() >= kp.len() {
                        &k[kp.len()..]
                    } else {
                        &[]
                    };
                    if !visit(value) {
                        return Ok(());
                    }
                }
                for (_full_key, value) in pending_iter {
                    if !visit(&value) {
                        break;
                    }
                }
                Ok::<(), BackendError>(())
            },
            |e| e,
        )?;
        Ok(())
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
    use super::*;
    use crate::helix_engine::storage_core::backend::{KeyRange, Namespace};
    use serial_test::serial;

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
    /// (HIGH #2 from the poll-tiering review): with the write-feed URL
    /// configured, fresh `DbReader` opens take the idle poll interval, so a
    /// `HelixGraphStorage` seeded from this predicate must start believing
    /// it's idle too — otherwise read-gated promotion never fires because it
    /// thinks a freshly-idle-opened collection is already fast.
    #[test]
    #[serial]
    fn reader_cold_open_starts_fast_matches_feed_url_configuration() {
        std::env::remove_var(ENV_WRITER_FEED_URL);
        assert!(
            reader_cold_open_starts_fast(),
            "no write-feed configured: cold opens serve at the fast/manifest-poll cadence"
        );

        let _guard = EnvGuard::set(ENV_WRITER_FEED_URL, "http://helix:6969");
        assert!(
            !reader_cold_open_starts_fast(),
            "write-feed configured: cold opens must start believing idle, matching \
             reader_options_from_env's actual idle-tier cold open"
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
    fn reader_options_take_idle_tier_when_feed_enabled() {
        let _cache = EnvGuard::set(ENV_CACHE_DIR, "");
        let _reader_poll = EnvGuard::set(ENV_READER_MANIFEST_POLL_MS, "5000");
        let _feed = EnvGuard::set(ENV_WRITER_FEED_URL, "http://helix:6969");
        let _idle = EnvGuard::set(ENV_READER_IDLE_POLL_MS, "300000");

        // Fresh opens take the idle safety net; the serving cadence becomes
        // the ACTIVE tier the feed poller promotes changed collections to.
        let reader_options = reader_options_from_env("helion/c");
        assert_eq!(
            reader_options.manifest_poll_interval,
            std::time::Duration::from_secs(300)
        );
        assert_eq!(
            reader_active_poll_interval_from_env(),
            std::time::Duration::from_secs(5)
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

    #[test]
    #[serial]
    fn lsm_settings_from_env_reflects_read_amplification_knobs() {
        let _cache = EnvGuard::set(ENV_CACHE_DIR, "");
        let _min_filter = EnvGuard::set(ENV_MIN_FILTER_KEYS, "1");
        let _wal_flushes = EnvGuard::set(ENV_MAX_WAL_FLUSHES_BEFORE_L0_FLUSH, "4096");
        let _l0_per_key = EnvGuard::set(ENV_L0_MAX_SSTS_PER_KEY, "4");
        let _l0_parallelism = EnvGuard::set(ENV_L0_FLUSH_PARALLELISM, "2");

        let settings =
            lsm_settings_from_env("helion/c").expect("LSM read-amplification knobs -> Settings");

        assert_eq!(settings.min_filter_keys, 1);
        assert_eq!(settings.max_wal_flushes_before_l0_flush, 4096);
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
}
