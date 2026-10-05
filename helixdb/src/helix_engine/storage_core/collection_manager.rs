use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering},
    Arc, Mutex, RwLock, Weak,
};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use heed3::CompactionOption;
use slatedb::CloseReason;

use crate::helix_engine::graph_core::config::Config;
use crate::helix_engine::storage_core::backend::{BackendKind, StorageBackendConfig};
use crate::helix_engine::storage_core::backend_any;
use crate::helix_engine::storage_core::backend_lsm::mask_lsm_read_cancellation_for_cold_open;
use crate::helix_engine::storage_core::backend_lsm_reader::LSM_READER_DATABASE_MISSING;
use crate::helix_engine::storage_core::metadata::StorageMetadataSidecar;
use crate::helix_engine::storage_core::storage_core::{
    HelixGraphStorage, PayloadIndexGcFieldResult, RecountCounters,
};
use crate::helix_engine::types::{graph_error_from_backend_error, GraphError};
use tracing::{debug, info, warn};

/// Total budget (ms) for retrying an LMDB env open on `EnvAlreadyOpen` and
/// for waiting on a `pending_drops` entry to drain. LMDB maintains a
/// process-wide open-env registry keyed by canonical path; when a collection
/// was just dropped but some other thread still holds an
/// `Arc<HelixGraphStorage>`, a concurrent `get/create` on the same name will
/// trip `EnvAlreadyOpened` until that Arc drops. Default sized against the
/// pod's 300 s request timeout so long in-flight scrolls / scans don't
/// surface drop→recreate races to the caller.
fn open_retry_budget_ms() -> u64 {
    std::env::var("HELIX_OPEN_RETRY_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30_000)
}

/// Ceiling (ms) on the deferred-cleanup thread's wait for the Arc to become
/// unique. If exceeded, the thread logs and exits — the Arc still closes
/// naturally when its last owner drops it, but `remove_dir_all` is skipped.
fn drop_deferred_cleanup_ms() -> u64 {
    std::env::var("HELIX_DROP_DEFERRED_CLEANUP_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60_000)
}

/// Ceiling (ms) for the cache-drop advisory worker to wait until the evicted
/// storage Arc is fully gone. The worker only issues page-cache hints after
/// the LMDB env and mmap sidecars have been unmapped by normal Arc drop.
fn cache_drop_hint_wait_ms() -> u64 {
    std::env::var("HELIX_CACHE_DROP_HINT_WAIT_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30_000)
}

fn cache_drop_hint_enabled() -> bool {
    std::env::var("HELIX_CACHE_DROP_HINT")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(true)
}

fn lsm_reader_cold_open_enabled(storage_backend: StorageBackendConfig) -> bool {
    storage_backend.is_reader()
}

fn lsm_cold_open_needs_cancellation_mask(storage_backend: StorageBackendConfig) -> bool {
    storage_backend.is_lsm()
}

fn lsm_collection_cache_unbounded(storage_backend: StorageBackendConfig) -> bool {
    storage_backend.is_lsm() && lsm_collection_cache_unbounded_enabled()
}

fn lsm_collection_cache_unbounded_enabled() -> bool {
    std::env::var("HELIX_LSM_COLLECTION_CACHE_UNBOUNDED")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

fn skip_periodic_snapshot_for_backend(kind: BackendKind) -> bool {
    kind == BackendKind::Lsm
}

/// Interval (ms) between reader-replica dense-view reconciles per collection.
/// Defaults to 5 000 ms. Set `HELIX_LSM_READER_REFRESH_MS` to override.
pub(crate) fn reader_refresh_ttl_ms() -> u64 {
    std::env::var("HELIX_LSM_READER_REFRESH_MS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(5_000)
}

// ─── Ratio-triggered auto-compaction knobs ─────────────────────────────────
//
// LMDB never shrinks: freed pages stay on the env freelist and `data.mdb`
// remains a high-water mark until a compact-copy + restore-in-place runs.
// These knobs gate the *automatic* trigger for that existing machinery.

/// Master kill-switch default. The whole feature is OFF for the first release
/// so it can be enabled per-environment after staging validation; flipping
/// this single const to `true` (or setting `HELIX_AUTO_COMPACT=1`) turns it on.
const AUTO_COMPACT_DEFAULT_ON: bool = false;

/// Whether ratio-triggered auto-compaction is enabled. `HELIX_AUTO_COMPACT`
/// overrides the [`AUTO_COMPACT_DEFAULT_ON`] compile-time default.
fn auto_compact_enabled() -> bool {
    match std::env::var("HELIX_AUTO_COMPACT") {
        Ok(value) => {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        }
        Err(_) => AUTO_COMPACT_DEFAULT_ON,
    }
}

/// Compact when `data.mdb` file_size / live_bytes exceeds this ratio.
/// Default 2.5 (i.e. > 60% dead space). Floored at 1.1 so it can never
/// thrash-compact a healthy file.
fn auto_compact_ratio() -> f64 {
    std::env::var("HELIX_COMPACT_RATIO")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 1.1)
        .unwrap_or(2.5)
}

/// Never auto-compact files smaller than this. Default 2 GiB: small files'
/// dead space is not worth the copy + page-cache churn.
fn auto_compact_min_bytes() -> u64 {
    std::env::var("HELIX_COMPACT_MIN_BYTES")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(2 * 1024 * 1024 * 1024)
}

/// Upper bound on the `data.mdb` file size eligible for auto-compaction.
/// `0` (default) = unlimited. Above this, compaction is skipped: the copy holds
/// the per-collection `open_gate` for its whole duration and stalls new writers
/// to that collection (see H2), so operators can exclude very large envs where
/// the stall is unacceptable until an off-peak/manual window.
fn auto_compact_max_file_bytes() -> u64 {
    std::env::var("HELIX_COMPACT_MAX_FILE_BYTES")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0)
}

/// Minimum interval between auto-compactions of the same collection.
/// Default 6 h.
fn auto_compact_cooldown_secs() -> u64 {
    std::env::var("HELIX_COMPACT_COOLDOWN_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(6 * 60 * 60)
}

/// A collection must have had no upserts for this long to be considered
/// write-idle and eligible for auto-compaction. Default 60 s.
fn auto_compact_write_idle_ms() -> u64 {
    std::env::var("HELIX_COMPACT_WRITE_IDLE_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(60_000)
}

/// Live (non-free) bytes for an LMDB env: the sum over the unnamed DB and
/// every named sub-DB of (leaf + branch + overflow pages) × page_size.
///
/// This is the cheap in-process freshness/dead-space signal used by the
/// ratio trigger. It is deliberately a clean-room reimplementation rather
/// than heed3's `Env::non_free_pages_size`: that built-in skips any catalog
/// key containing a NUL byte, and LMDB stores sub-DB names NUL-padded in the
/// unnamed DB, so on a real Helion env (named DBs `nodes`, `edges`, vector
/// segment DBs, …) it silently under-counts to ~one page and would make every
/// collection look like pure dead space. Verified: a 33 MB named DB reports
/// 16 KiB through the built-in vs 33 MB through this path. We trim trailing
/// NULs from each catalog key and open the sub-DB via the safe typed API.
///
/// NOTE: this is live *logical* bytes (B-tree pages actually holding data),
/// NOT `last_page_number × page_size` (the high-water mark / file size).
fn env_live_bytes(
    env: &heed3::Env<heed3::WithTls>,
    rtxn: &heed3::RoTxn,
) -> Result<u64, GraphError> {
    use heed3::types::Bytes;

    let page_pages = |s: &heed3::DatabaseStat| -> u64 {
        (s.leaf_pages + s.branch_pages + s.overflow_pages) as u64 * s.page_size as u64
    };

    let mut total = 0u64;
    let Some(main) = env.open_database::<Bytes, Bytes>(rtxn, None)? else {
        return Ok(0);
    };
    total = total.saturating_add(page_pages(&main.stat(rtxn)?));

    // The keys of the unnamed DB are the names of the named sub-DBs. Sub-DB
    // names are stored NUL-padded; trim trailing NULs before reopening.
    for entry in main.iter(rtxn)? {
        let (key, _value) = entry?;
        let mut end = key.len();
        while end > 0 && key[end - 1] == 0 {
            end -= 1;
        }
        if end == 0 {
            continue;
        }
        let Ok(name) = std::str::from_utf8(&key[..end]) else {
            continue;
        };
        // A key that does not resolve to a sub-DB is ordinary data in the
        // unnamed DB (already counted in `main.stat`); open_database returns
        // Ok(None) / Err there, both of which we skip.
        match env.open_database::<Bytes, Bytes>(rtxn, Some(name)) {
            Ok(Some(db)) => {
                if let Ok(stat) = db.stat(rtxn) {
                    total = total.saturating_add(page_pages(&stat));
                }
            }
            Ok(None) | Err(_) => {}
        }
    }

    Ok(total)
}

/// Bounded budget (ms) to drain in-flight holders of a collection's storage
/// Arc down to unique before a compaction swap. A serving instance always has
/// the manager's cached Arc (removed first) plus possibly short-lived gateway
/// handler clones; those drain in milliseconds. If quiescence is not reached
/// within this budget the compaction aborts cost-free (no copy paid) and the
/// collection keeps serving. Default 5 s.
fn auto_compact_drain_ms() -> u64 {
    std::env::var("HELIX_COMPACT_DRAIN_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(5_000)
}

/// Maximum consecutive-failure cooldown backoff multiplier (cap). The cooldown
/// doubles per consecutive busy/restore failure up to this cap so a
/// pathologically hot collection cannot drive repeated full-file copies.
const AUTO_COMPACT_MAX_BACKOFF_MULT: u64 = 16;

/// Warn if a single compact-copy holds the per-collection `open_gate` (and thus
/// stalls new writers to that collection) longer than this. New upserts to the
/// collection being compacted block on `open_gate` for the copy duration; this
/// surfaces the stall in logs so operators can tune size/idle gates or off-peak.
const AUTO_COMPACT_SLOW_COPY_WARN: Duration = Duration::from_secs(2);

/// fsync a file's data + metadata so its bytes (and length) are durable. Used
/// in the compaction swap to guarantee the renamed `data.mdb` is on disk before
/// the WAL (the only other recovery source) is deleted.
fn sync_file_durable(path: &Path) -> Result<(), GraphError> {
    let file = std::fs::File::open(path)?;
    file.sync_all()?;
    Ok(())
}

/// fsync a directory so a rename within it (the new `data.mdb` directory entry)
/// is journaled. Without this barrier a crash after `rename` can leave the
/// directory entry pointing at an inode whose data/length never reached disk.
/// On platforms where opening a directory for fsync is unsupported this is a
/// best-effort no-op (the file-level fsync above still bounds the exposure).
fn sync_dir_durable(dir: &Path) -> Result<(), GraphError> {
    match std::fs::File::open(dir) {
        Ok(handle) => {
            // Directory fsync can legitimately fail on some filesystems; treat
            // a failure as best-effort (the data.mdb fsync already ran).
            let _ = handle.sync_all();
            Ok(())
        }
        Err(_) => Ok(()),
    }
}

/// Global single-compaction permit. A compaction copies an entire env to disk
/// and spikes RSS / page-cache, so at most one runs process-wide at a time.
/// `try_lock` is used so a sweep never blocks waiting for an in-flight compact.
static AUTO_COMPACT_SEMAPHORE: std::sync::LazyLock<Mutex<()>> =
    std::sync::LazyLock::new(|| Mutex::new(()));

/// Collections currently in their compaction drain/copy/swap window.
///
/// The eviction-drain waits for the storage `Arc` to reach `strong_count == 1`,
/// but Helion's OWN background loops (optimizer/backlog sweep, idle sweeper,
/// the auto-compact sweep's snapshot Vec, page-cache scans) periodically clone
/// Arcs to loaded collections. With a drain window comparable to the sweep
/// cadence there is almost always an internal holder in-window, so the drain
/// never reaches quiescence even on a fully idle box (the v2 smoke symptom:
/// "deferred ... no copy paid" every cycle, forever). The marker breaks that:
/// it is set BEFORE the drain begins and every INTERNAL Arc-taker skips a
/// marked collection for the duration. Client `get_collection` is unaffected
/// (it still blocks on the `closing` weak-ref). Bounded by the number of
/// collections compacting at once (≤1 via the global permit).
static AUTO_COMPACT_IN_PROGRESS: std::sync::LazyLock<Mutex<HashSet<String>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashSet::new()));

/// True while `name` is inside its compaction window. INTERNAL background
/// Arc-takers consult this and SKIP marked collections so their transient
/// `Arc::clone` cannot pin the storage above `strong_count == 1` and starve the
/// drain. A poisoned lock fails OPEN (returns false → do not skip) so a poison
/// can never wedge normal maintenance; the global permit still bounds risk.
pub(crate) fn auto_compact_in_progress(name: &str) -> bool {
    AUTO_COMPACT_IN_PROGRESS
        .lock()
        .map(|set| set.contains(name))
        .unwrap_or(false)
}

/// RAII guard that marks `name` as compacting for its lifetime and clears the
/// mark on drop — including on panic or any early-return error path, so the
/// marker can never leak and permanently exclude a collection from maintenance.
struct CompactionInProgressGuard {
    name: String,
}

impl CompactionInProgressGuard {
    /// Mark `name` and return the guard. The mark is cleared when the guard
    /// drops.
    fn new(name: &str) -> Self {
        if let Ok(mut set) = AUTO_COMPACT_IN_PROGRESS.lock() {
            set.insert(name.to_string());
        }
        Self {
            name: name.to_string(),
        }
    }
}

impl Drop for CompactionInProgressGuard {
    fn drop(&mut self) {
        if let Ok(mut set) = AUTO_COMPACT_IN_PROGRESS.lock() {
            set.remove(&self.name);
        }
    }
}

/// Per-collection consecutive auto-compaction failure count, for cooldown
/// backoff. Reset to 0 on a successful compaction.
static AUTO_COMPACT_FAILURES: std::sync::LazyLock<Mutex<HashMap<String, u32>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// Per-collection cold-open failure state, for open-failure backoff. A
/// collection whose storage is corrupt (e.g. manifest referencing GC'd SSTs)
/// fails every cold open, and each attempt costs object-store round-trips,
/// bumps the SlateDB writer epoch (fencing any prior handle), and spins up a
/// GC/compactor that immediately dies. Without a negative cache, request
/// traffic turns that into an S3-hammering open loop. Cleared on successful
/// open and on create/drop so recreate-based repair is never throttled.
struct OpenFailureState {
    failures: u32,
    last_attempt: Instant,
    last_error: String,
    sticky_quarantine: bool,
}

static OPEN_FAILURES: std::sync::LazyLock<Mutex<HashMap<String, OpenFailureState>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

const OPEN_FAILURE_BACKOFF_MAX_MS: u64 = 60_000;

fn open_failure_backoff_base_ms() -> u64 {
    std::env::var("HELIX_COLLECTION_OPEN_FAILURE_BACKOFF_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2_000)
}

/// If `name` is inside its open-failure cooldown, returns the error from the
/// last real attempt so callers fail fast with the same classification
/// upstream repair logic keys on, without touching object storage.
fn open_failure_backoff_error(name: &str) -> Option<GraphError> {
    let base = open_failure_backoff_base_ms();
    if base == 0 {
        return None;
    }
    let map = OPEN_FAILURES.lock().ok()?;
    let state = map.get(name)?;
    if state.sticky_quarantine {
        return Some(GraphError::New(state.last_error.clone()));
    }
    let mult = 1u64
        .checked_shl(state.failures.saturating_sub(1).min(20))
        .unwrap_or(u64::MAX);
    let cooldown =
        Duration::from_millis(base.saturating_mul(mult).min(OPEN_FAILURE_BACKOFF_MAX_MS));
    if state.last_attempt.elapsed() < cooldown {
        Some(GraphError::New(state.last_error.clone()))
    } else {
        None
    }
}

/// SlateDB phrases an open against a prefix with no manifest as "failed to
/// find latest transactional object (e.g. manifest) version". No manifest
/// means the collection does not exist in object storage — dropped/purged
/// elsewhere (readers never see the writer's drop tombstones), or never
/// created. Surface the same not-found error the registry paths raise so it
/// maps to 404 rather than a replica-health 500. The phrasing must keep
/// clear of `lsm_object_reference_is_missing` so a not-found can never
/// quarantine the name.
///
/// The vendored SlateDB fork's `DbReader` instead reports a missing manifest
/// as the typed `ErrorCode::DatabaseMissing`; `LsmReader` tags that case with
/// [`LSM_READER_DATABASE_MISSING`] before the error is stringified, so the
/// reader replica matches that marker rather than SlateDB's display text.
fn missing_manifest_open_error(name: &str, error: GraphError) -> GraphError {
    let message = error.to_string();
    if message
        .to_ascii_lowercase()
        .contains("latest transactional object")
        || message.contains(LSM_READER_DATABASE_MISSING)
    {
        GraphError::New(format!(
            "Collection '{name}' not found {MISSING_MANIFEST_NOT_FOUND_SUFFIX}"
        ))
    } else {
        error
    }
}

const MISSING_MANIFEST_NOT_FOUND_SUFFIX: &str = "(no manifest in object store)";

fn open_failure_record(name: &str, error: &GraphError) {
    open_failure_record_inner(name, error, false);
}

fn open_failure_record_quarantine(name: &str, error: &GraphError) {
    open_failure_record_inner(name, error, true);
}

fn fenced_writer_quarantine_error(name: &str, reason: CloseReason) -> GraphError {
    GraphError::New(format!(
        "Collection '{name}' LSM writer is fenced ({reason:?}); collection is quarantined until process replacement"
    ))
}

fn record_fenced_writer_quarantine(name: &str, reason: CloseReason) -> GraphError {
    let error = fenced_writer_quarantine_error(name, reason);
    open_failure_record_quarantine(name, &error);
    metrics::counter!("helix_collection_quarantine_total").increment(1);
    error
}

fn open_failure_record_inner(name: &str, error: &GraphError, sticky_quarantine: bool) {
    if let Ok(mut map) = OPEN_FAILURES.lock() {
        let entry = map.entry(name.to_string()).or_insert(OpenFailureState {
            failures: 0,
            last_attempt: Instant::now(),
            last_error: String::new(),
            sticky_quarantine,
        });
        let last_error = error.to_string();
        // A missing manifest is a cheap LIST, not an S3-hammering corrupt open:
        // keep the base cooldown so a collection the writer creates right after
        // is not hidden behind an escalated (up to 60s) not-found.
        entry.failures = if last_error.contains(MISSING_MANIFEST_NOT_FOUND_SUFFIX) {
            1
        } else {
            entry.failures.saturating_add(1)
        };
        entry.last_attempt = Instant::now();
        entry.last_error = last_error;
        entry.sticky_quarantine |= sticky_quarantine;
    }
}

fn open_failure_clear(name: &str) {
    if let Ok(mut map) = OPEN_FAILURES.lock() {
        map.remove(name);
    }
}

fn open_failure_clear_after_successful_open(name: &str) {
    if let Ok(mut map) = OPEN_FAILURES.lock() {
        if map.get(name).is_some_and(|state| !state.sticky_quarantine) {
            map.remove(name);
        }
    }
}

/// Record one consecutive auto-compaction failure for `name` (busy/copy/
/// restore). Lengthens the effective cooldown via [`auto_compact_cooldown_elapsed`].
fn auto_compact_record_failure(name: &str) {
    if let Ok(mut map) = AUTO_COMPACT_FAILURES.lock() {
        let entry = map.entry(name.to_string()).or_insert(0);
        *entry = entry.saturating_add(1);
    }
}

/// Clear the consecutive-failure count for `name` after a successful compaction.
fn auto_compact_clear_failures(name: &str) {
    if let Ok(mut map) = AUTO_COMPACT_FAILURES.lock() {
        map.remove(name);
    }
}

/// Effective cooldown backoff multiplier for `name`: `2^failures`, capped at
/// [`AUTO_COMPACT_MAX_BACKOFF_MULT`]. 0 failures → 1× (base cooldown).
fn auto_compact_backoff_mult(name: &str) -> u64 {
    let failures = AUTO_COMPACT_FAILURES
        .lock()
        .ok()
        .and_then(|map| map.get(name).copied())
        .unwrap_or(0);
    let mult = 1u64.checked_shl(failures.min(20)).unwrap_or(u64::MAX);
    mult.min(AUTO_COMPACT_MAX_BACKOFF_MULT)
}

/// Exclusive window over a collection during compaction: carries the now-unique
/// storage `Arc` and the collection dir path. The per-collection open_gate is
/// held by the CALLER (a `MutexGuard` cannot be stored alongside the `Arc` it
/// borrows from), so this struct stays lifetime-free. While the caller's gate
/// guard is alive and the entry is published in `closing`, new opens block on
/// the weak ref and no concurrent writer can advance the env. Produced by
/// `evict_and_drain_for_compaction`, consumed by `swap_in_compacted_file`.
struct CompactionWindow {
    storage: Arc<HelixGraphStorage>,
    path: PathBuf,
}

/// Per-collection last-compaction-start instant, for cooldown enforcement.
static AUTO_COMPACT_LAST_START: std::sync::LazyLock<Mutex<HashMap<String, Instant>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// True if `name` is outside its compaction cooldown (or has never compacted).
///
/// The base cooldown is multiplied by the consecutive-failure backoff
/// (`2^failures`, capped) so a collection that repeatedly fails to quiesce or
/// swap is retried exponentially less often instead of paying I/O every cycle.
fn auto_compact_cooldown_elapsed(name: &str) -> bool {
    let base = auto_compact_cooldown_secs();
    let cooldown = Duration::from_secs(base.saturating_mul(auto_compact_backoff_mult(name)));
    match AUTO_COMPACT_LAST_START.lock() {
        Ok(map) => map
            .get(name)
            .map(|last| last.elapsed() >= cooldown)
            .unwrap_or(true),
        // Poisoned lock: fail closed (do not compact) rather than risk a loop.
        Err(_) => false,
    }
}

/// Record that a compaction for `name` has just started (arms the cooldown).
fn auto_compact_mark_started(name: &str) {
    if let Ok(mut map) = AUTO_COMPACT_LAST_START.lock() {
        map.insert(name.to_string(), Instant::now());
    }
}

/// Best-effort free-bytes on the filesystem backing `path`. Returns `None`
/// when the platform statvfs call is unavailable or fails; callers treat
/// `None` as "skip the preflight" (do not block compaction on an unknown).
#[cfg(unix)]
fn available_disk_bytes(path: &Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let c_path = CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };
    if rc != 0 {
        return None;
    }
    // f_bavail = blocks available to non-root; f_frsize = fragment size.
    Some((stat.f_bavail as u64).saturating_mul(stat.f_frsize as u64))
}

#[cfg(not(unix))]
fn available_disk_bytes(_path: &Path) -> Option<u64> {
    None
}

fn cache_drop_hint_paths(collection_path: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let Ok(entries) = fs::read_dir(collection_path) else {
        return paths;
    };

    for entry in entries.filter_map(|entry| entry.ok()) {
        let path = entry.path();
        if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
            continue;
        }
        let file_name = path.file_name().and_then(|value| value.to_str());
        let extension = path.extension().and_then(|value| value.to_str());
        if file_name == Some("data.mdb") || matches!(extension, Some("hvec" | "hvs8" | "hvtq")) {
            paths.push(path);
        }
    }
    paths
}

fn remove_lsm_object_cache_dirs_not_in(
    cache_root: &Path,
    retained: &HashSet<PathBuf>,
    reason: &'static str,
) -> u64 {
    let Ok(entries) = fs::read_dir(cache_root) else {
        return 0;
    };
    let mut removed = 0u64;
    for entry in entries.filter_map(|entry| entry.ok()) {
        let path = entry.path();
        if retained.contains(&path) {
            continue;
        }
        if !entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
            continue;
        }
        match fs::remove_dir_all(&path) {
            Ok(()) => {
                removed += 1;
                metrics::counter!(
                    "helix_collection_lsm_cache_subtree_removals_total",
                    "outcome" => "removed",
                    "reason" => reason
                )
                .increment(1);
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                metrics::counter!(
                    "helix_collection_lsm_cache_subtree_removals_total",
                    "outcome" => "missing",
                    "reason" => reason
                )
                .increment(1);
            }
            Err(err) => {
                metrics::counter!(
                    "helix_collection_lsm_cache_subtree_removals_total",
                    "outcome" => "error",
                    "reason" => reason
                )
                .increment(1);
                warn!(
                    cache_subtree = %path.display(),
                    error = %err,
                    reason,
                    "Failed to remove orphaned LSM object-store cache subtree"
                );
            }
        }
    }
    removed
}

fn lsm_object_cache_retained_dirs(
    collections: &Arc<RwLock<HashMap<String, CachedCollection>>>,
    collections_dir: &Path,
) -> Result<HashSet<PathBuf>, String> {
    let mut retained = HashSet::new();
    match collections.read() {
        Ok(collections) => {
            retained.extend(collections.keys().filter_map(|name| {
                backend_any::cache_subtree_for_collection_path(&collections_dir.join(name))
            }));
        }
        Err(err) => {
            return Err(err.to_string());
        }
    }

    if let Ok(entries) = fs::read_dir(collections_dir) {
        retained.extend(entries.filter_map(|entry| {
            let entry = entry.ok()?;
            if !entry.file_type().ok()?.is_dir() {
                return None;
            }
            backend_any::cache_subtree_for_collection_path(&entry.path())
        }));
    }

    Ok(retained)
}

fn sweep_lsm_orphan_object_cache_dirs(
    collections: &Arc<RwLock<HashMap<String, CachedCollection>>>,
    collections_dir: &Path,
    reason: &'static str,
    storage_backend: StorageBackendConfig,
) {
    if !storage_backend.is_lsm() {
        return;
    }
    let Some(cache_root) = backend_any::lsm_cache_root_from_env() else {
        return;
    };
    let retained = match lsm_object_cache_retained_dirs(collections, collections_dir) {
        Ok(retained) => retained,
        Err(err) => {
            warn!(error = %err, "LSM object-cache orphan sweep skipped after lock poison");
            return;
        }
    };
    let removed = remove_lsm_object_cache_dirs_not_in(&cache_root, &retained, reason);
    if removed > 0 {
        info!(
            removed,
            retained = retained.len(),
            cache_root = %cache_root.display(),
            "Removed orphaned LSM object-store cache subtrees"
        );
    }
}

#[cfg(target_os = "linux")]
fn advise_file_cache_dropped(path: &Path) -> std::io::Result<u64> {
    use std::os::fd::AsRawFd;

    let file = fs::File::open(path)?;
    let len = file.metadata()?.len();
    let rc = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    if rc == 0 {
        Ok(len)
    } else {
        Err(std::io::Error::from_raw_os_error(rc))
    }
}

#[cfg(not(target_os = "linux"))]
fn advise_file_cache_dropped(path: &Path) -> std::io::Result<u64> {
    let _ = fs::metadata(path)?;
    Ok(0)
}

fn advise_collection_cache_dropped(collection_path: &Path, collection: &str, reason: &'static str) {
    if !cache_drop_hint_enabled() {
        metrics::counter!(
            "helix_collection_cache_drop_hint_total",
            "outcome" => "disabled",
            "reason" => reason
        )
        .increment(1);
        return;
    }

    let started = Instant::now();
    let paths = cache_drop_hint_paths(collection_path);
    if paths.is_empty() {
        metrics::counter!(
            "helix_collection_cache_drop_hint_total",
            "outcome" => "empty",
            "reason" => reason
        )
        .increment(1);
        return;
    }

    let mut files = 0u64;
    let mut bytes = 0u64;
    let mut errors = 0u64;
    for path in paths {
        match advise_file_cache_dropped(&path) {
            Ok(len) => {
                files += 1;
                bytes = bytes.saturating_add(len);
            }
            Err(err) => {
                errors += 1;
                debug!(
                    collection,
                    reason,
                    path = %path.display(),
                    error = %err,
                    "Collection cache drop hint failed for file"
                );
            }
        }
    }

    if bytes > 0 {
        metrics::counter!(
            "helix_collection_cache_drop_hint_bytes_total",
            "reason" => reason
        )
        .increment(bytes);
    }
    metrics::histogram!(
        "helix_collection_cache_drop_hint_duration_ms",
        "reason" => reason
    )
    .record(started.elapsed().as_secs_f64() * 1000.0);
    metrics::counter!(
        "helix_collection_cache_drop_hint_total",
        "outcome" => if errors == 0 { "completed" } else { "partial" },
        "reason" => reason
    )
    .increment(1);
    debug!(
        collection,
        reason,
        files,
        bytes,
        errors,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "Collection cache drop hint completed"
    );
}

fn schedule_cache_drop_after_close(
    weak: Weak<HelixGraphStorage>,
    collection_path: PathBuf,
    collection: String,
    reason: &'static str,
) {
    if !cache_drop_hint_enabled() {
        return;
    }

    metrics::counter!(
        "helix_collection_cache_drop_hint_total",
        "outcome" => "scheduled",
        "reason" => reason
    )
    .increment(1);

    let thread_name = format!("helix-cache-drop-{}", collection);
    let spawn_collection = collection.clone();
    if let Err(err) = thread::Builder::new().name(thread_name).spawn(move || {
        let started = Instant::now();
        let budget = Duration::from_millis(cache_drop_hint_wait_ms());
        let mut poll = Duration::from_millis(50);
        loop {
            if weak.upgrade().is_none() {
                metrics::histogram!(
                    "helix_collection_cache_drop_hint_wait_ms",
                    "outcome" => "closed",
                    "reason" => reason
                )
                .record(started.elapsed().as_secs_f64() * 1000.0);
                advise_collection_cache_dropped(&collection_path, &collection, reason);
                return;
            }

            if started.elapsed() >= budget {
                metrics::histogram!(
                    "helix_collection_cache_drop_hint_wait_ms",
                    "outcome" => "timeout",
                    "reason" => reason
                )
                .record(started.elapsed().as_secs_f64() * 1000.0);
                metrics::counter!(
                    "helix_collection_cache_drop_hint_total",
                    "outcome" => "timeout",
                    "reason" => reason
                )
                .increment(1);
                debug!(
                    collection,
                    reason,
                    budget_ms = budget.as_millis() as u64,
                    "Collection cache drop hint timed out waiting for env close"
                );
                return;
            }

            thread::sleep(poll);
            poll = (poll * 2).min(Duration::from_millis(500));
        }
    }) {
        metrics::counter!(
            "helix_collection_cache_drop_hint_total",
            "outcome" => "spawn_error",
            "reason" => reason
        )
        .increment(1);
        warn!(
            collection = %spawn_collection,
            reason,
            error = %err,
            "Failed to spawn collection cache drop hint worker"
        );
    }
}

/// Object-store prefixes whose purge failed during `drop_collection` and has
/// not yet been confirmed, keyed by collection path (the S3 prefix is derived
/// from it, so a recreate of the same name reuses the same prefix). The value
/// is the token of the retry worker that owns the purge. While an entry exists
/// the name must not be (re)opened: the old manifest would resurrect dropped
/// data, and a later retry would delete the live writer's new objects.
static LSM_PURGE_PENDING: std::sync::LazyLock<Mutex<HashMap<PathBuf, u64>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));
static LSM_PURGE_TOKEN: AtomicU64 = AtomicU64::new(1);

fn lsm_purge_pending_register(path: &Path) -> u64 {
    let token = LSM_PURGE_TOKEN.fetch_add(1, Ordering::Relaxed);
    if let Ok(mut pending) = LSM_PURGE_PENDING.lock() {
        pending.insert(path.to_path_buf(), token);
    }
    token
}

fn lsm_purge_pending(path: &Path) -> bool {
    LSM_PURGE_PENDING
        .lock()
        .map(|pending| pending.contains_key(path))
        // A poisoned registry cannot prove the prefix is clean; fail closed.
        .unwrap_or(true)
}

fn lsm_purge_pending_clear(path: &Path) {
    if let Ok(mut pending) = LSM_PURGE_PENDING.lock() {
        pending.remove(path);
    }
}

#[derive(Debug, PartialEq, Eq)]
enum LsmPurgeRetryStep {
    Purged,
    Failed,
    /// The pending entry is gone or owned by a newer drop: the prefix was
    /// purged synchronously (e.g. by a recreate) and may now hold a live
    /// collection. Purging it would destroy that collection's objects.
    Superseded,
}

/// One retry attempt. Runs under the collection's open gate so it cannot
/// interleave with a create/open of the same name, and only purges while
/// this worker's token still owns the pending entry.
fn lsm_purge_retry_attempt(
    path: &Path,
    token: u64,
    open_gate: &Mutex<()>,
    tombstone_exists: impl FnOnce() -> Result<bool, super::backend::BackendError>,
    purge: impl FnOnce() -> Result<(), super::backend::BackendError>,
    clear_tombstone: impl FnOnce() -> Result<(), super::backend::BackendError>,
    name: &str,
    attempt: u32,
) -> LsmPurgeRetryStep {
    let _guard = match open_gate.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let owns_entry = LSM_PURGE_PENDING
        .lock()
        .map(|pending| pending.get(path) == Some(&token))
        .unwrap_or(false);
    if !owns_entry {
        return LsmPurgeRetryStep::Superseded;
    }
    match tombstone_exists() {
        Ok(true) => {}
        Ok(false) => return LsmPurgeRetryStep::Superseded,
        Err(error) => {
            warn!(
                collection = %name,
                attempt,
                error = %error,
                "LSM object-store drop tombstone lookup failed before purge retry"
            );
            return LsmPurgeRetryStep::Failed;
        }
    }
    match purge() {
        Ok(()) => {
            if let Err(error) = clear_tombstone() {
                warn!(
                    collection = %name,
                    attempt,
                    error = %error,
                    "LSM object-store drop tombstone cleanup failed after purge retry"
                );
                return LsmPurgeRetryStep::Failed;
            }
            if let Ok(mut pending) = LSM_PURGE_PENDING.lock() {
                if pending.get(path) == Some(&token) {
                    pending.remove(path);
                }
            }
            LsmPurgeRetryStep::Purged
        }
        Err(error) => {
            warn!(
                collection = %name,
                attempt,
                error = %error,
                "LSM object-store purge retry failed"
            );
            LsmPurgeRetryStep::Failed
        }
    }
}

/// Retry a failed drop-time prefix purge in the background. The caller must
/// have registered `token` in `LSM_PURGE_PENDING` for `path`. On exhaustion the
/// entry is left in place so the name stays unopenable until a later
/// create/drop purges the prefix synchronously.
fn spawn_lsm_purge_retry(
    path: PathBuf,
    name: String,
    storage_backend: StorageBackendConfig,
    token: u64,
    open_gate: Arc<Mutex<()>>,
) {
    metrics::counter!(
        "helix_collection_drop_purge_retry_scheduled_total",
        "collection" => name.clone()
    )
    .increment(1);
    let spawned = thread::Builder::new()
        .name(format!("helix-purge-retry-{}", name))
        .spawn({
            let name = name.clone();
            move || {
                let mut backoff = Duration::from_secs(10);
                for attempt in 1..=6u32 {
                    thread::sleep(backoff);
                    let step = lsm_purge_retry_attempt(
                        &path,
                        token,
                        &open_gate,
                        || backend_any::lsm_drop_tombstone_exists_from_env(&path, storage_backend),
                        || backend_any::purge_lsm_prefix_from_env(&path, storage_backend),
                        || backend_any::clear_lsm_drop_tombstone_from_env(&path, storage_backend),
                        &name,
                        attempt,
                    );
                    match step {
                        LsmPurgeRetryStep::Purged => {
                            info!(
                                collection = %name,
                                attempt,
                                "LSM object-store purge retry succeeded"
                            );
                            metrics::counter!(
                                "helix_collection_drop_purge_retry_success_total",
                                "collection" => name.clone()
                            )
                            .increment(1);
                            return;
                        }
                        LsmPurgeRetryStep::Superseded => {
                            info!(
                                collection = %name,
                                attempt,
                                "LSM object-store purge retry superseded (prefix already purged or recreated); not purging"
                            );
                            return;
                        }
                        LsmPurgeRetryStep::Failed => {}
                    }
                    backoff = (backoff * 2).min(Duration::from_secs(300));
                }
                warn!(
                    collection = %name,
                    "LSM object-store purge retries exhausted; prefix is orphaned and the name stays blocked until a create/drop purges it"
                );
                metrics::counter!(
                    "helix_collection_drop_purge_retry_exhausted_total",
                    "collection" => name.clone()
                )
                .increment(1);
            }
        });
    if let Err(error) = spawned {
        warn!(
            collection = %name,
            error = %error,
            "Failed to spawn LSM purge retry worker; prefix is orphaned"
        );
    }
}

/// Worker body for deferred collection cleanup. Spins until the moved-in
/// `Arc<HelixGraphStorage>` is unique, drops it to close the LMDB env
/// (releases the path from heed's process-wide `OPENED_ENV` registry), then
/// removes the on-disk directory. If the Arc never becomes unique within
/// `HELIX_DROP_DEFERRED_CLEANUP_MS`, logs and exits; the Arc drops at scope
/// end so the env eventually closes but directory removal is skipped (the
/// dir is orphaned — `list_collections` will still report it; the operator
/// can reclaim it, or the next restart will clean up).
fn deferred_drop_cleanup(storage: Arc<HelixGraphStorage>, path: PathBuf, name: String) {
    let mut arc = Some(storage);
    let started = Instant::now();
    let budget = Duration::from_millis(drop_deferred_cleanup_ms());
    let mut poll = Duration::from_millis(50);
    loop {
        if Arc::strong_count(arc.as_ref().expect("arc present")) == 1 {
            drop(arc.take());
            if path.exists() {
                if let Err(err) = fs::remove_dir_all(&path) {
                    warn!(
                        collection = %name,
                        error = %err,
                        "Deferred drop: remove_dir_all failed"
                    );
                } else {
                    debug!(
                        collection = %name,
                        waited_ms = started.elapsed().as_millis() as u64,
                        "Deferred drop cleanup completed"
                    );
                    metrics::counter!(
                        "helix_collection_drop_deferred_completed_total",
                        "collection" => name.clone()
                    )
                    .increment(1);
                }
            }
            return;
        }
        if started.elapsed() >= budget {
            warn!(
                collection = %name,
                budget_ms = budget.as_millis() as u64,
                strong_count = Arc::strong_count(arc.as_ref().expect("arc present")),
                "Deferred drop cleanup exceeded budget; abandoning directory removal"
            );
            metrics::counter!(
                "helix_collection_drop_deferred_abandoned_total",
                "collection" => name.clone()
            )
            .increment(1);
            return;
        }
        thread::sleep(poll);
        poll = (poll * 2).min(Duration::from_millis(500));
    }
}

/// Open an LMDB-backed `HelixGraphStorage` with bounded retry on
/// `GraphError::EnvAlreadyOpen`. Handles the drop→recreate race where a
/// concurrent caller still holds an `Arc` to the prior env for the same
/// path. Total wait ≤ `HELIX_OPEN_RETRY_MS` (default 2 s).
fn open_storage_with_retry(
    path: &Path,
    config: Config,
    collection_name: &str,
) -> Result<HelixGraphStorage, GraphError> {
    let path_str = path
        .to_str()
        .ok_or_else(|| GraphError::New("Invalid path".into()))?;
    let budget = Duration::from_millis(open_retry_budget_ms());
    let started = Instant::now();
    let path_exists = if path.exists() { "true" } else { "false" };
    let mut attempt: u32 = 0;
    let mut backoff = Duration::from_millis(20);
    loop {
        match HelixGraphStorage::new(path_str, config.clone()) {
            Ok(storage) => {
                metrics::histogram!(
                    "helix_collection_open_storage_ms",
                    "outcome" => "ok",
                    "path_exists" => path_exists,
                    "retried" => if attempt > 0 { "true" } else { "false" }
                )
                .record(started.elapsed().as_secs_f64() * 1000.0);
                if attempt > 0 {
                    debug!(
                        collection = %collection_name,
                        attempts = attempt,
                        waited_ms = started.elapsed().as_millis() as u64,
                        "Opened LMDB env after EnvAlreadyOpen retry"
                    );
                    metrics::counter!(
                        "helix_collection_open_retry_success_total",
                        "collection" => collection_name.to_string()
                    )
                    .increment(1);
                }
                if orphan_sidecar_scan_on_open_enabled() {
                    match maintenance_process_resource_admission() {
                        MaintenanceAdmission::Admitted => {
                            let scan_started = Instant::now();
                            let stats = storage
                                .named_vectors
                                .cleanup_inactive_sidecar_files(path, false);
                            metrics::histogram!("helix_orphan_sidecar_scan_on_open_ms")
                                .record(scan_started.elapsed().as_secs_f64() * 1000.0);
                            metrics::counter!(
                                "helix_orphan_sidecar_scan_on_open_total",
                                "outcome" => "ran",
                                "reason" => "admitted"
                            )
                            .increment(1);
                            if stats.removed_files > 0 || stats.error_count > 0 {
                                info!(
                                    collection = %collection_name,
                                    removed_files = stats.removed_files,
                                    removed_bytes = stats.removed_bytes,
                                    skipped_active = stats.skipped_active_files,
                                    errors = stats.error_count,
                                    scanned_ms = scan_started.elapsed().as_millis() as u64,
                                    "orphan sidecar scan on open"
                                );
                            }
                        }
                        MaintenanceAdmission::Rejected(reason) => {
                            metrics::counter!(
                                "helix_orphan_sidecar_scan_on_open_total",
                                "outcome" => "skipped",
                                "reason" => reason
                            )
                            .increment(1);
                            debug!(
                                collection = %collection_name,
                                reason,
                                "Skipping orphan sidecar scan on open under maintenance pressure"
                            );
                        }
                    }
                }
                return Ok(storage);
            }
            Err(GraphError::EnvAlreadyOpen) => {
                if started.elapsed() >= budget {
                    metrics::histogram!(
                        "helix_collection_open_storage_ms",
                        "outcome" => "env_already_open_exhausted",
                        "path_exists" => path_exists,
                        "retried" => if attempt > 0 { "true" } else { "false" }
                    )
                    .record(started.elapsed().as_secs_f64() * 1000.0);
                    warn!(
                        collection = %collection_name,
                        budget_ms = budget.as_millis() as u64,
                        attempts = attempt,
                        "Giving up on EnvAlreadyOpen retry"
                    );
                    metrics::counter!(
                        "helix_collection_open_retry_exhausted_total",
                        "collection" => collection_name.to_string()
                    )
                    .increment(1);
                    return Err(GraphError::EnvAlreadyOpen);
                }
                thread::sleep(backoff);
                backoff = (backoff * 2).min(Duration::from_millis(200));
                attempt += 1;
            }
            Err(e) => {
                metrics::histogram!(
                    "helix_collection_open_storage_ms",
                    "outcome" => "error",
                    "path_exists" => path_exists,
                    "retried" => if attempt > 0 { "true" } else { "false" }
                )
                .record(started.elapsed().as_secs_f64() * 1000.0);
                return Err(e);
            }
        }
    }
}

/// Default maximum open LMDB environments kept in the collection cache.
/// Override with `HELIX_MAX_OPEN_COLLECTIONS` env var for EKS tuning.
///
/// Budget at 64 MB initial map per collection:
///   256 × 64 MB = 16 GB VA — comfortable on 8 GB pods.
///   Evicted collections are re-opened from disk on next access (~2 ms).
fn max_open_collections() -> usize {
    std::env::var("HELIX_MAX_OPEN_COLLECTIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256)
        .max(1)
}

fn collection_idle_eviction_ms() -> u64 {
    std::env::var("HELIX_COLLECTION_IDLE_EVICTION_MS")
        .or_else(|_| std::env::var("HELIX_IDLE_EVICTION_MS"))
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300_000)
}

fn count_dense_sidecars_in_dir(dir: &Path) -> usize {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    entries
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "hvec" || ext == "hvs8" || ext == "hvtq")
                .unwrap_or(false)
        })
        .count()
}

/// Scan collection directory for orphan `.hvec` / `.hvs8` / `.hvtq` sidecar files on
/// env open and unlink any whose segment is no longer in the named-vector
/// metadata. Rebalances disk footprint after historical reaper lag or
/// process crash between segment retire and sidecar unlink.
///
/// Enabled by default. Override with `HELIX_ORPHAN_SIDECAR_SCAN_ON_OPEN=0`
/// to disable (e.g., for diagnostic purposes — keeps orphans visible).
fn orphan_sidecar_scan_on_open_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("HELIX_ORPHAN_SIDECAR_SCAN_ON_OPEN")
            .ok()
            .map(|v| !matches!(v.as_str(), "0" | "false" | "FALSE" | "no" | "off"))
            .unwrap_or(true)
    })
}

fn maintenance_memory_high_pct() -> usize {
    std::env::var("HELIX_MAINTENANCE_MEM_HIGH_PCT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| (1..=100).contains(v))
        .unwrap_or(70)
}

fn maintenance_memory_high_bytes() -> Option<usize> {
    if let Ok(value) = std::env::var("HELIX_MAINTENANCE_MEM_HIGH_MB") {
        if let Ok(mb) = value.parse::<usize>() {
            if mb == 0 {
                return None;
            }
            return Some(mb.saturating_mul(1024).saturating_mul(1024));
        }
    }
    memory_ceiling_bytes().map(|ceiling| ceiling * maintenance_memory_high_pct() / 100)
}

fn maintenance_open_max_loaded_pct() -> usize {
    std::env::var("HELIX_MAINTENANCE_OPEN_MAX_LOADED_PCT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| (1..=100).contains(v))
        .unwrap_or(85)
}

fn maintenance_open_loaded_limit(max_open: usize, pct: usize) -> usize {
    let pct = pct.clamp(1, 100);
    max_open.saturating_mul(pct).div_ceil(100).max(1)
}

fn maintenance_fd_high() -> Option<usize> {
    std::env::var("HELIX_MAINTENANCE_FD_HIGH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .and_then(|v| if v == 0 { None } else { Some(v) })
        .or(Some(4096))
}

fn process_fd_count() -> Option<usize> {
    fs::read_dir("/proc/self/fd")
        .ok()
        .map(|entries| entries.filter_map(|entry| entry.ok()).count())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaintenanceAdmission {
    Admitted,
    Rejected(&'static str),
}

impl MaintenanceAdmission {
    pub fn is_admitted(self) -> bool {
        matches!(self, MaintenanceAdmission::Admitted)
    }

    pub fn reason(self) -> &'static str {
        match self {
            MaintenanceAdmission::Admitted => "admitted",
            MaintenanceAdmission::Rejected(reason) => reason,
        }
    }
}

fn record_maintenance_process_snapshot(memory: CollectionMemorySnapshot, fd_count: Option<usize>) {
    record_memory_snapshot(memory);
    if let Some(value) = fd_count {
        metrics::gauge!("helix_maintenance_fd_count").set(value as f64);
    }
}

/// Reclaim-aware memory gate for maintenance admission. Uses the snapshot's
/// `pressure_bytes()` (current minus reclaimable file cache) so a warm mmap
/// page cache — the steady state on every node — does not read as pressure.
fn maintenance_memory_pressure_exceeds(
    memory: CollectionMemorySnapshot,
    high_bytes: Option<usize>,
) -> bool {
    match (memory.pressure_bytes(), high_bytes) {
        (Some(pressure), Some(high)) => pressure >= high,
        _ => false,
    }
}

pub(crate) fn maintenance_process_resource_admission() -> MaintenanceAdmission {
    let memory = collection_memory_snapshot();
    let fd_count = process_fd_count();
    record_maintenance_process_snapshot(memory, fd_count);

    if maintenance_memory_pressure_exceeds(memory, maintenance_memory_high_bytes()) {
        return MaintenanceAdmission::Rejected("memory");
    }

    if let (Some(count), Some(high)) = (fd_count, maintenance_fd_high()) {
        if count >= high {
            return MaintenanceAdmission::Rejected("fds");
        }
    }

    MaintenanceAdmission::Admitted
}

fn page_cache_sweep_ms() -> u64 {
    std::env::var("HELIX_PAGE_CACHE_SWEEP_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| if cfg!(test) { 0 } else { 10_000 })
}

fn page_cache_sweep_batch() -> usize {
    std::env::var("HELIX_PAGE_CACHE_SWEEP_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16)
}

fn page_cache_min_reclaimable_bytes() -> usize {
    std::env::var("HELIX_PAGE_CACHE_MIN_RECLAIMABLE_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1024)
        .saturating_mul(1024)
        .saturating_mul(1024)
}

fn collection_idle_sweep_ms() -> u64 {
    std::env::var("HELIX_COLLECTION_IDLE_SWEEP_MS")
        .or_else(|_| std::env::var("HELIX_IDLE_EVICTION_SWEEP_MS"))
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| if cfg!(test) { 0 } else { 30_000 })
}

fn collection_idle_eviction_min_loaded() -> usize {
    std::env::var("HELIX_COLLECTION_IDLE_EVICTION_MIN_LOADED")
        .or_else(|_| std::env::var("HELIX_IDLE_EVICTION_MIN_LOADED"))
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

fn collection_idle_eviction_per_sweep() -> usize {
    std::env::var("HELIX_COLLECTION_IDLE_EVICTION_PER_SWEEP")
        .or_else(|_| std::env::var("HELIX_IDLE_EVICTION_BATCH"))
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16)
}

/// Minimum cache size before idle-age eviction runs when memory is low.
/// Below this threshold cold opens are cheaper than the open/evict thrash.
fn collection_idle_min_cache_size() -> usize {
    std::env::var("HELIX_COLLECTION_IDLE_MIN_CACHE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20)
}

/// Maximum time a cold collection open may wait for cache/memory capacity.
/// Hot-cache hits never pass through this path. A bounded wait is safer than
/// letting a 500-search cold miss storm open hundreds of LMDB envs and get the
/// process OOM-killed.
fn collection_open_wait_ms() -> u64 {
    std::env::var("HELIX_COLLECTION_OPEN_WAIT_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5_000)
}

fn collection_open_poll_ms() -> u64 {
    std::env::var("HELIX_COLLECTION_OPEN_POLL_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(25)
        .max(1)
}

fn collection_open_memory_high_pct() -> usize {
    std::env::var("HELIX_COLLECTION_OPEN_MEM_HIGH_PCT")
        .or_else(|_| std::env::var("HELIX_MEM_HIGH_PCT"))
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| (1..=100).contains(v))
        .unwrap_or(85)
}

fn collection_open_memory_low_pct() -> usize {
    std::env::var("HELIX_COLLECTION_OPEN_MEM_LOW_PCT")
        .or_else(|_| std::env::var("HELIX_MEM_LOW_PCT"))
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| (1..=100).contains(v))
        .unwrap_or(70)
}

fn collection_open_memory_high_bytes() -> Option<usize> {
    if let Ok(value) = std::env::var("HELIX_COLLECTION_OPEN_MEM_HIGH_MB") {
        if let Ok(mb) = value.parse::<usize>() {
            return Some(mb.saturating_mul(1024).saturating_mul(1024));
        }
    }
    memory_ceiling_bytes().map(|ceiling| ceiling * collection_open_memory_high_pct() / 100)
}

fn collection_open_current_high_pct() -> usize {
    std::env::var("HELIX_COLLECTION_OPEN_CURRENT_HIGH_PCT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| (1..=100).contains(v))
        .unwrap_or(90)
}

fn collection_open_current_high_bytes() -> Option<usize> {
    if let Ok(value) = std::env::var("HELIX_COLLECTION_OPEN_CURRENT_HIGH_MB") {
        if let Ok(mb) = value.parse::<usize>() {
            if mb == 0 {
                return None;
            }
            return Some(mb.saturating_mul(1024).saturating_mul(1024));
        }
    }
    memory_ceiling_bytes().map(|ceiling| ceiling * collection_open_current_high_pct() / 100)
}

/// Recovery percentage of the current-high watermark for idle-sweeper
/// hysteresis. Once the sweeper starts evicting for `current >= high`, it
/// keeps evicting until `current` drops below this percentage of `high`, so
/// the evict decision cannot oscillate while `current` hovers at `high`.
/// Must be below 90 (the `collection_open_current_high_pct` default) to
/// create a band.
fn collection_open_current_low_recovery_pct() -> usize {
    std::env::var("HELIX_COLLECTION_OPEN_CURRENT_LOW_RECOVERY_PCT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| (50..=89).contains(v))
        .unwrap_or(85)
}

fn collection_open_memory_low_bytes() -> Option<usize> {
    if let Ok(value) = std::env::var("HELIX_COLLECTION_OPEN_MEM_LOW_MB") {
        if let Ok(mb) = value.parse::<usize>() {
            return Some(mb.saturating_mul(1024).saturating_mul(1024));
        }
    }
    memory_ceiling_bytes().map(|ceiling| ceiling * collection_open_memory_low_pct() / 100)
}

fn parse_meminfo_total_bytes(contents: &str) -> Option<usize> {
    contents.lines().find_map(|line| {
        let mut parts = line.split_whitespace();
        match (parts.next(), parts.next()) {
            (Some("MemTotal:"), Some(kb)) => kb
                .parse::<usize>()
                .ok()
                .map(|value| value.saturating_mul(1024)),
            _ => None,
        }
    })
}

fn meminfo_total_bytes() -> Option<usize> {
    fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|contents| parse_meminfo_total_bytes(&contents))
}

fn memory_ceiling_bytes() -> Option<usize> {
    if let Ok(value) = std::env::var("HELIX_MEMORY_LIMIT_MB") {
        if let Ok(mb) = value.parse::<usize>() {
            return Some(mb.saturating_mul(1024).saturating_mul(1024));
        }
    }
    if let Ok(value) = fs::read_to_string("/sys/fs/cgroup/memory.max") {
        let value = value.trim();
        if value != "max" {
            if let Ok(bytes) = value.parse::<usize>() {
                return Some(bytes);
            }
        }
    }
    if let Ok(value) = fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes") {
        if let Ok(bytes) = value.trim().parse::<usize>() {
            if bytes < (1usize << 62) {
                return Some(bytes);
            }
        }
    }
    meminfo_total_bytes().or_else(|| Some(8usize.saturating_mul(1024 * 1024 * 1024)))
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CollectionMemorySnapshot {
    pub current_bytes: Option<usize>,
    pub inactive_file_bytes: Option<usize>,
    pub active_file_bytes: Option<usize>,
    pub file_bytes: Option<usize>,
    pub shmem_bytes: Option<usize>,
    pub file_dirty_bytes: Option<usize>,
    pub file_writeback_bytes: Option<usize>,
    pub reclaimable_file_bytes: Option<usize>,
    pub working_set_bytes: Option<usize>,
    pub admission_pressure_bytes: Option<usize>,
    pub ceiling_bytes: Option<usize>,
    pub low_bytes: Option<usize>,
    pub high_bytes: Option<usize>,
}

/// The single memory-pressure arbiter's verdict for one snapshot, evaluated
/// against a `current` watermark, an (optional) `pressure` high watermark, and
/// a reclaimable-file-cache floor.
///
/// Every consumer that asks "is this collection cache under memory pressure?"
/// (idle sweeper, page-cache sweeper, cold-open guard, build admission) reads
/// its decision from one of the accessors below so the evict / skip / block /
/// defer choice lives in exactly one place. Before this struct existed the
/// same intent was reimplemented in three predicates that drifted apart and
/// caused the c8d90da3 prod bug.
#[derive(Debug, Clone, Copy)]
struct PressureDecision {
    /// `Some((current, high))` when `current >= high`, else `None`. This is the
    /// raw current-vs-high comparison with no exemption applied.
    current_high: Option<(usize, usize)>,
    /// Pressure (the reclaim-aware figure) is below its high watermark.
    pressure_is_safe: bool,
    /// Reclaimable file cache is at or above the supplied floor, i.e. there is
    /// enough page cache that the kernel can drop to relieve `current` without
    /// us evicting / blocking application state.
    reclaimable_file_is_large: bool,
}

impl PressureDecision {
    /// True when the reclaimable-file-cache exemption applies: pressure is safe
    /// AND there is a large reclaimable file cache the kernel can drop instead.
    fn reclaimable_exempt(self) -> bool {
        self.pressure_is_safe && self.reclaimable_file_is_large
    }

    /// THE arbiter result: `Some((current, high))` when memory pressure
    /// genuinely requires action — `current >= high` AND the exemption does not
    /// apply — and `None` otherwise. Callers that evict / block / defer consume
    /// this; callers that only want the raw `current >= high` fact read
    /// [`Self::current_high`].
    fn requires_action(self) -> Option<(usize, usize)> {
        self.current_high.filter(|_| !self.reclaimable_exempt())
    }
}

impl CollectionMemorySnapshot {
    fn pressure_bytes(self) -> Option<usize> {
        self.admission_pressure_bytes
            .or(self.working_set_bytes)
            .or(self.current_bytes)
    }

    fn current_above_high(self, high: Option<usize>) -> Option<(usize, usize)> {
        let current = self.current_bytes?;
        let high = high?;
        (current >= high).then_some((current, high))
    }

    /// Sole pressure arbiter. Evaluates the current-vs-high comparison and the
    /// reclaimable-file-cache exemption once; all pressure consumers derive
    /// their evict / skip / block / defer decision from the returned
    /// [`PressureDecision`] so the logic exists in exactly one place.
    fn pressure_decision(
        self,
        current_high_bytes: Option<usize>,
        pressure_high_bytes: Option<usize>,
        reclaimable_file_floor_bytes: usize,
    ) -> PressureDecision {
        let pressure_is_safe = match (self.pressure_bytes(), pressure_high_bytes) {
            (Some(pressure), Some(pressure_high)) => pressure < pressure_high,
            _ => false,
        };
        let reclaimable_file_is_large =
            self.reclaimable_file_bytes.unwrap_or(0) >= reclaimable_file_floor_bytes;

        PressureDecision {
            current_high: self.current_above_high(current_high_bytes),
            pressure_is_safe,
            reclaimable_file_is_large,
        }
    }

    /// Test-only thin wrapper over the arbiter, retained so the existing
    /// reclaimable-exemption regression tests keep asserting end-to-end
    /// `current_high + exemption` semantics. Production code calls
    /// `pressure_decision(..).requires_action()` (or `build_admission_under_pressure`)
    /// directly.
    #[cfg(test)]
    fn current_high_requires_block_at(
        self,
        current_high_bytes: Option<usize>,
        pressure_high_bytes: Option<usize>,
        reclaimable_file_floor_bytes: usize,
    ) -> Option<(usize, usize)> {
        self.pressure_decision(
            current_high_bytes,
            pressure_high_bytes,
            reclaimable_file_floor_bytes,
        )
        .requires_action()
    }
}

#[derive(Debug, Clone, Copy)]
struct MemoryUsage {
    current_bytes: usize,
    inactive_file_bytes: usize,
    active_file_bytes: usize,
    file_bytes: usize,
    shmem_bytes: usize,
    file_dirty_bytes: usize,
    file_writeback_bytes: usize,
}

impl MemoryUsage {
    fn working_set_bytes(self) -> usize {
        self.current_bytes.saturating_sub(self.inactive_file_bytes)
    }

    fn reclaimable_file_bytes(self) -> usize {
        if self.file_bytes == 0 {
            return self.inactive_file_bytes;
        }

        let unreclaimable_file = self
            .shmem_bytes
            .saturating_add(self.file_dirty_bytes)
            .saturating_add(self.file_writeback_bytes);
        self.file_bytes.saturating_sub(unreclaimable_file)
    }

    fn pressure_bytes(self) -> usize {
        self.current_bytes
            .saturating_sub(self.reclaimable_file_bytes())
    }
}

fn memory_stat_value(contents: &str, keys: &[&str]) -> usize {
    for key in keys {
        if let Some(value) = contents.lines().find_map(|line| {
            let mut parts = line.split_whitespace();
            match (parts.next(), parts.next()) {
                (Some(found), Some(value)) if found == *key => value.parse::<usize>().ok(),
                _ => None,
            }
        }) {
            return value;
        }
    }
    0
}

fn memory_usage_from_stat(current_bytes: usize, contents: &str) -> MemoryUsage {
    let inactive_file_bytes =
        memory_stat_value(contents, &["inactive_file", "total_inactive_file"]);
    let active_file_bytes = memory_stat_value(contents, &["active_file", "total_active_file"]);
    let file_bytes = match memory_stat_value(contents, &["file", "total_cache", "cache"]) {
        0 => active_file_bytes.saturating_add(inactive_file_bytes),
        value => value,
    };

    MemoryUsage {
        current_bytes,
        inactive_file_bytes,
        active_file_bytes,
        file_bytes,
        shmem_bytes: memory_stat_value(contents, &["shmem", "total_shmem"]),
        file_dirty_bytes: memory_stat_value(contents, &["file_dirty", "total_dirty", "dirty"]),
        file_writeback_bytes: memory_stat_value(
            contents,
            &["file_writeback", "total_writeback", "writeback"],
        ),
    }
}

fn current_memory_usage() -> Option<MemoryUsage> {
    if let Ok(current_raw) = fs::read_to_string("/sys/fs/cgroup/memory.current") {
        if let Ok(current) = current_raw.trim().parse::<usize>() {
            return fs::read_to_string("/sys/fs/cgroup/memory.stat")
                .map(|contents| memory_usage_from_stat(current, &contents))
                .ok();
        }
    }
    if let Ok(current_raw) = fs::read_to_string("/sys/fs/cgroup/memory/memory.usage_in_bytes") {
        if let Ok(current) = current_raw.trim().parse::<usize>() {
            return fs::read_to_string("/sys/fs/cgroup/memory/memory.stat")
                .map(|contents| memory_usage_from_stat(current, &contents))
                .ok();
        }
    }
    None
}

fn collection_memory_snapshot() -> CollectionMemorySnapshot {
    let usage = current_memory_usage();
    CollectionMemorySnapshot {
        current_bytes: usage.map(|value| value.current_bytes),
        inactive_file_bytes: usage.map(|value| value.inactive_file_bytes),
        active_file_bytes: usage.map(|value| value.active_file_bytes),
        file_bytes: usage.map(|value| value.file_bytes),
        shmem_bytes: usage.map(|value| value.shmem_bytes),
        file_dirty_bytes: usage.map(|value| value.file_dirty_bytes),
        file_writeback_bytes: usage.map(|value| value.file_writeback_bytes),
        reclaimable_file_bytes: usage.map(|value| value.reclaimable_file_bytes()),
        working_set_bytes: usage.map(|value| value.working_set_bytes()),
        admission_pressure_bytes: usage.map(|value| value.pressure_bytes()),
        ceiling_bytes: memory_ceiling_bytes(),
        low_bytes: collection_open_memory_low_bytes(),
        high_bytes: collection_open_memory_high_bytes(),
    }
}

/// True when a fresh memory snapshot is above the hard-high watermark per the
/// single pressure arbiter — i.e. `current >= high` and the reclaimable-file
/// cache exemption does not apply.
///
/// This is the build-admission entry point for `vector_core`: new HNSW build
/// starts consult it through the same arbiter the idle sweeper, page-cache
/// sweeper, and cold-open guard use, so build deferral cannot drift from the
/// rest of the memory-pressure machinery. It mirrors the cold-open guard's
/// watermarks (`collection_open_current_high_bytes` for current,
/// `collection_open_memory_high_bytes` for the reclaim-aware pressure high,
/// `page_cache_min_reclaimable_bytes` for the exemption floor).
pub(crate) fn build_admission_under_pressure() -> bool {
    collection_memory_snapshot()
        .pressure_decision(
            collection_open_current_high_bytes(),
            collection_open_memory_high_bytes(),
            page_cache_min_reclaimable_bytes(),
        )
        .requires_action()
        .is_some()
}

/// Idle-sweeper evict decision with hysteresis. Eviction starts on the
/// arbiter's `requires_action` (`current >= high` AND the reclaimable-file
/// exemption does not apply). Once started, eviction continues until
/// `current` drops below `recovery_pct` of `high` or the exemption starts to
/// apply, so the decision cannot oscillate while `current` hovers at `high`.
fn idle_sweeper_should_evict(
    decision: PressureDecision,
    current_bytes: Option<usize>,
    current_high_bytes: Option<usize>,
    recovery_pct: usize,
    was_evicting: bool,
) -> bool {
    if decision.requires_action().is_some() {
        return true;
    }
    if !was_evicting || decision.reclaimable_exempt() {
        return false;
    }
    match (current_bytes, current_high_bytes) {
        (Some(current), Some(high)) => {
            let recovery = ((high as u128 * recovery_pct as u128) / 100) as usize;
            current >= recovery
        }
        _ => false,
    }
}

fn record_memory_snapshot(snapshot: CollectionMemorySnapshot) {
    if let Some(value) = snapshot.current_bytes {
        metrics::gauge!("helix_collection_open_memory_current_bytes").set(value as f64);
    }
    if let Some(value) = snapshot.inactive_file_bytes {
        metrics::gauge!("helix_collection_open_memory_inactive_file_bytes").set(value as f64);
    }
    if let Some(value) = snapshot.active_file_bytes {
        metrics::gauge!("helix_collection_open_memory_active_file_bytes").set(value as f64);
    }
    if let Some(value) = snapshot.file_bytes {
        metrics::gauge!("helix_collection_open_memory_file_bytes").set(value as f64);
    }
    if let Some(value) = snapshot.shmem_bytes {
        metrics::gauge!("helix_collection_open_memory_shmem_bytes").set(value as f64);
    }
    if let Some(value) = snapshot.file_dirty_bytes {
        metrics::gauge!("helix_collection_open_memory_file_dirty_bytes").set(value as f64);
    }
    if let Some(value) = snapshot.file_writeback_bytes {
        metrics::gauge!("helix_collection_open_memory_file_writeback_bytes").set(value as f64);
    }
    if let Some(value) = snapshot.reclaimable_file_bytes {
        metrics::gauge!("helix_collection_open_memory_reclaimable_file_bytes").set(value as f64);
    }
    if let Some(value) = snapshot.working_set_bytes {
        metrics::gauge!("helix_collection_open_memory_working_set_bytes").set(value as f64);
    }
    if let Some(value) = snapshot.admission_pressure_bytes {
        metrics::gauge!("helix_collection_open_memory_pressure_bytes").set(value as f64);
    }
    if let Some(value) = snapshot.ceiling_bytes {
        metrics::gauge!("helix_collection_open_memory_ceiling_bytes").set(value as f64);
    }
    if let Some(value) = snapshot.low_bytes {
        metrics::gauge!("helix_collection_open_memory_low_bytes").set(value as f64);
    }
    if let Some(value) = snapshot.high_bytes {
        metrics::gauge!("helix_collection_open_memory_high_bytes").set(value as f64);
    }
    if let Some(value) = collection_open_current_high_bytes() {
        metrics::gauge!("helix_collection_open_memory_current_high_bytes").set(value as f64);
    }
}

/// Worst-case virtual-address-space budget per LMDB env, in bytes. Mirrors
/// `HelixGraphStorage::max_map_size()` — default 48 GB. An env's VA footprint
/// can grow up to this cap via `grow_map()` increments. At 48 GB × 256 cached
/// envs, projected VA is 12 TB — comfortably within Linux's 128 TB budget.
fn max_map_size_bytes() -> u64 {
    std::env::var("HELIX_MAX_MAP_SIZE_GB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(48)
        .saturating_mul(1024 * 1024 * 1024)
}

/// VA exhaustion warning threshold in bytes. Defaults to 32 TB, which leaves
/// ample headroom on 48-bit Linux (~128 TB usable) and macOS (~18 TB).
/// Override with `HELIX_VA_WARN_THRESHOLD_GB`.
fn va_warn_threshold_bytes() -> u64 {
    std::env::var("HELIX_VA_WARN_THRESHOLD_GB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(32 * 1024)
        .saturating_mul(1024 * 1024 * 1024)
}

/// Count collection directories currently on disk. Cheap fs scan; only
/// called at startup. Returns 0 on any IO error — this is observability,
/// not correctness, so we never fail construction over it.
fn count_collections_on_disk(collections_dir: &Path) -> usize {
    fs::read_dir(collections_dir)
        .map(|iter| {
            iter.filter_map(|e| e.ok())
                .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                .count()
        })
        .unwrap_or(0)
}

/// Emit a one-shot log summary of projected VA consumption at startup so
/// operators can correlate pod OOM / map-full incidents with capacity plans.
/// Warns when projected VA crosses the configured threshold.
fn audit_virtual_address_budget(max_open: usize, on_disk: usize) {
    let per_env = max_map_size_bytes();
    // Only the currently-cacheable envs hold VA; everything else sits on disk
    // until opened. Projected VA is therefore bounded by `min(on_disk, max_open)`.
    let concurrent = on_disk.min(max_open) as u64;
    let projected_bytes = concurrent.saturating_mul(per_env);
    let projected_gb = projected_bytes / (1024 * 1024 * 1024);
    let threshold = va_warn_threshold_bytes();
    info!(
        projected_va_gb = projected_gb,
        max_open,
        on_disk,
        per_env_max_gb = per_env / (1024 * 1024 * 1024),
        "LMDB virtual-address budget"
    );
    if projected_bytes >= threshold {
        warn!(
            projected_va_gb = projected_gb,
            threshold_gb = threshold / (1024 * 1024 * 1024),
            "Projected LMDB virtual-address usage exceeds safety threshold; \
             consider lowering HELIX_MAX_OPEN_COLLECTIONS or HELIX_MAX_MAP_SIZE_GB"
        );
    }
}

/// Cached collection entry with LRU tracking.
struct CachedCollection {
    storage: Arc<HelixGraphStorage>,
    path: PathBuf,
    /// Monotonic access counter for LRU eviction.
    last_access: AtomicU64,
    /// Wall-clock touch time for idle-age eviction.
    last_access_millis: AtomicU64,
}

struct OpenSlotReservation<'a> {
    opening: Option<&'a AtomicUsize>,
}

impl Drop for OpenSlotReservation<'_> {
    fn drop(&mut self) {
        if let Some(opening) = self.opening {
            let remaining = opening.fetch_sub(1, Ordering::AcqRel).saturating_sub(1);
            metrics::gauge!("helix_collection_open_inflight").set(remaining as f64);
        }
    }
}

/// Global monotonic counter for LRU ordering.
static LRU_CLOCK: AtomicU64 = AtomicU64::new(0);

fn current_time_millis() -> Option<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u64::MAX as u128) as u64)
        .ok()
}

fn touch_cached_collection(entry: &CachedCollection) {
    entry
        .last_access
        .store(LRU_CLOCK.fetch_add(1, Ordering::Relaxed), Ordering::Relaxed);
    if let Some(now_millis) = current_time_millis() {
        entry
            .last_access_millis
            .store(now_millis, Ordering::Relaxed);
    }
}

fn record_cold_open_duration(started: Instant, outcome: &'static str) {
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    metrics::histogram!("helix_collection_cold_open_duration_ms", "outcome" => outcome)
        .record(elapsed_ms);
    metrics::counter!("helix_collection_cold_open_total", "outcome" => outcome).increment(1);
}

fn cold_collection_cache_drop_candidates(
    collections_dir: &Path,
    loaded: &HashSet<String>,
    cursor: &mut usize,
    batch: usize,
) -> Vec<(String, PathBuf)> {
    if batch == 0 {
        return Vec::new();
    }

    let Ok(entries) = fs::read_dir(collections_dir) else {
        return Vec::new();
    };
    let mut candidates: Vec<(String, PathBuf)> = entries
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
        .filter_map(|entry| {
            let name = entry.file_name().to_str()?.to_string();
            (!loaded.contains(&name)).then_some((name, entry.path()))
        })
        .collect();
    candidates.sort_by(|a, b| a.0.cmp(&b.0));

    if candidates.is_empty() {
        *cursor = 0;
        return Vec::new();
    }

    if *cursor >= candidates.len() {
        *cursor %= candidates.len();
    }
    let take = batch.min(candidates.len());
    let mut selected = Vec::with_capacity(take);
    for offset in 0..take {
        selected.push(candidates[(*cursor + offset) % candidates.len()].clone());
    }
    *cursor = (*cursor + take) % candidates.len();
    selected
}

fn spawn_idle_collection_sweeper(collections: &Arc<RwLock<HashMap<String, CachedCollection>>>) {
    let sweep_ms = collection_idle_sweep_ms();
    let idle_ms = collection_idle_eviction_ms();
    let per_sweep = collection_idle_eviction_per_sweep();
    if sweep_ms == 0 || idle_ms == 0 || per_sweep == 0 {
        metrics::counter!(
            "helix_collection_idle_sweeper_total",
            "outcome" => "disabled"
        )
        .increment(1);
        return;
    }

    let collections = Arc::downgrade(collections);
    if let Err(err) = thread::Builder::new()
        .name("helix-idle-collections".to_string())
        .spawn(move || {
            let sweep = Duration::from_millis(sweep_ms);
            let mut evicting = false;
            loop {
                thread::sleep(sweep);
                let Some(collections) = collections.upgrade() else {
                    return;
                };
                let started = Instant::now();
                let evicted = match collections.write() {
                    Ok(mut collections) => {
                        let memory = collection_memory_snapshot();
                        record_memory_snapshot(memory);
                        let mut evicted =
                            CollectionManager::evict_unhealthy_writers(&mut collections);
                        let reclaimable_file_floor = page_cache_min_reclaimable_bytes();
                        // Single arbiter with the reclaim-aware pressure high: when
                        // `current` is inflated by reclaimable page cache (e.g. a
                        // backfill streaming LMDB pages) and pressure is safe, the
                        // exemption applies and the idle sweeper leaves the cache
                        // alone — the page-cache sweeper relieves `current` instead.
                        // `idle_sweeper_should_evict` adds hysteresis on top.
                        let decision = memory.pressure_decision(
                            collection_open_current_high_bytes(),
                            collection_open_memory_high_bytes(),
                            reclaimable_file_floor,
                        );
                        evicting = idle_sweeper_should_evict(
                            decision,
                            memory.current_bytes,
                            collection_open_current_high_bytes(),
                            collection_open_current_low_recovery_pct(),
                            evicting,
                        );
                        if evicting {
                            let memory_evicted = CollectionManager::evict_idle_for_memory(
                                &mut collections,
                                "current_memory",
                            );
                            evicted += memory_evicted;
                            // Wedge guard: `current` is whole-cgroup memory.
                            // If nothing was left to evict, holding the
                            // evicting state cannot move `current` (it may be
                            // pinned by mmaps/anon the env reaper doesn't
                            // own) and would only churn freshly-loaded
                            // collections on every sweep. Drop the hold;
                            // re-entry still requires a fresh
                            // `requires_action` trigger at `high`.
                            if memory_evicted == 0 {
                                evicting = false;
                            }
                            evicted += CollectionManager::evict_idle_for_age(&mut collections);
                        } else if collections.len() >= collection_idle_min_cache_size() {
                            // Memory is low but the cache is large enough that idle
                            // cleanup is worthwhile — below this threshold cold opens
                            // are cheaper than the open/evict thrash.
                            evicted += CollectionManager::evict_idle_for_age(&mut collections);
                        }
                        metrics::gauge!("helix_collection_cache_pinned").set(
                            collections
                                .values()
                                .filter(|entry| Arc::strong_count(&entry.storage) > 1)
                                .count() as f64,
                        );
                        metrics::gauge!("helix_collections_loaded").set(collections.len() as f64);
                        evicted
                    }
                    Err(err) => {
                        metrics::counter!(
                            "helix_collection_idle_sweeper_total",
                            "outcome" => "lock_poisoned"
                        )
                        .increment(1);
                        warn!(error = %err, "Idle collection sweeper lock poisoned");
                        0
                    }
                };
                metrics::histogram!("helix_collection_idle_sweeper_duration_ms")
                    .record(started.elapsed().as_secs_f64() * 1000.0);
                metrics::counter!(
                    "helix_collection_idle_sweeper_total",
                    "outcome" => if evicted > 0 { "evicted" } else { "scanned" }
                )
                .increment(1);
                if evicted > 0 {
                    debug!(
                        evicted,
                        elapsed_ms = started.elapsed().as_millis() as u64,
                        "Idle collection sweeper closed old collections"
                    );
                }
            }
        })
    {
        metrics::counter!(
            "helix_collection_idle_sweeper_total",
            "outcome" => "spawn_error"
        )
        .increment(1);
        warn!(error = %err, "Failed to spawn idle collection sweeper");
    }
}

/// Minimal HTTP/1.1 GET for the in-cluster change feed: plain HTTP, tiny JSON
/// body, one request per poll tick. `Connection: close` so the body ends at
/// EOF — no HTTP client dependency needed (reqwest is feature-gated behind
/// `ingestion`).
fn fetch_feed_changes(
    url: &str,
) -> Result<crate::helix_engine::storage_core::change_feed::FeedChanges, String> {
    use std::io::{Read, Write};
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| "feed URL must be http://".to_string())?;
    let (host_port, path) = match rest.split_once('/') {
        Some((host_port, path)) => (host_port, format!("/{path}")),
        None => (rest, "/".to_string()),
    };
    let authority = if host_port.contains(':') {
        host_port.to_string()
    } else {
        format!("{host_port}:80")
    };
    let timeout = Duration::from_secs(5);
    let addr = std::net::ToSocketAddrs::to_socket_addrs(&authority.as_str())
        .map_err(|e| e.to_string())?
        .next()
        .ok_or_else(|| format!("no address for {authority}"))?;
    let mut stream =
        std::net::TcpStream::connect_timeout(&addr, timeout).map_err(|e| e.to_string())?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: {host_port}\r\nConnection: close\r\n\r\n"
    )
    .map_err(|e| e.to_string())?;
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|e| e.to_string())?;
    let header_end = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response".to_string())?;
    let head = String::from_utf8_lossy(&response[..header_end]);
    let status_line = head.lines().next().unwrap_or("");
    if !status_line.contains(" 200 ") {
        return Err(format!("feed returned non-200: {status_line}"));
    }
    serde_json::from_slice(&response[header_end + 4..]).map_err(|e| e.to_string())
}

/// Reader-replica change-feed poller (see `storage_core::change_feed`): polls
/// the writer's `/internal/changes` over in-cluster HTTP and reopens ONLY the
/// resident collections that actually changed — promoting them to the active
/// (short-poll) `DbReader` tier and demoting them back to the idle tier after
/// a quiet period. This replaces the O(residents × readers) per-collection S3
/// manifest/WAL polling floor with S3 traffic proportional to write activity.
/// Enabled only on LSM reader replicas with `HELIX_LSM_WRITER_FEED_URL` set;
/// unset keeps the previous per-collection S3 polling behavior.
fn spawn_reader_change_feed_poller(
    collections: &Arc<RwLock<HashMap<String, CachedCollection>>>,
    storage_backend: StorageBackendConfig,
) {
    use crate::helix_engine::storage_core::backend_lsm::{
        reader_active_poll_interval_from_env, reader_active_ttl_from_env,
        reader_feed_poll_interval_from_env, reader_idle_poll_interval_from_env,
        writer_feed_url_from_env,
    };
    if !lsm_reader_cold_open_enabled(storage_backend) {
        return;
    }
    let Some(feed_url) = writer_feed_url_from_env() else {
        return;
    };
    let poll = reader_feed_poll_interval_from_env();
    let active_interval = reader_active_poll_interval_from_env();
    let idle_interval = reader_idle_poll_interval_from_env();
    let active_ttl = reader_active_ttl_from_env();
    let collections = Arc::downgrade(collections);
    if let Err(err) = thread::Builder::new()
        .name("helix-reader-feed".to_string())
        .spawn(move || {
            let mut since: u64 = 0;
            let mut epoch: u64 = 0;
            let mut initialized = false;
            let mut active: HashMap<String, Instant> = HashMap::new();
            loop {
                thread::sleep(poll);
                let Some(collections) = collections.upgrade() else {
                    return;
                };
                let url = format!("{feed_url}/internal/changes?since={since}");
                let snapshot = match fetch_feed_changes(&url) {
                    Ok(snapshot) => snapshot,
                    Err(error) => {
                        metrics::counter!(
                            "helix_reader_feed_polls_total",
                            "outcome" => "error"
                        )
                        .increment(1);
                        debug!(error, "reader change-feed poll failed");
                        continue;
                    }
                };
                metrics::counter!(
                    "helix_reader_feed_polls_total",
                    "outcome" => "ok"
                )
                .increment(1);
                if snapshot.epoch != epoch {
                    // New writer process: its cursor restarted, so ours is
                    // meaningless. On reader boot (first poll) the fresh
                    // DbReaders are already current — just adopt the cursor.
                    // On a live epoch flip, re-poll from 0 so everything the
                    // new writer has written gets refreshed.
                    epoch = snapshot.epoch;
                    since = if initialized { 0 } else { snapshot.cursor };
                    initialized = true;
                    continue;
                }
                let now = Instant::now();
                for name in &snapshot.changes {
                    let storage = match collections.read() {
                        Ok(guard) => guard.get(name).map(|entry| Arc::clone(&entry.storage)),
                        Err(_) => break,
                    };
                    // Not resident: nothing to refresh — a later cold open
                    // reads current state by construction.
                    let Some(storage) = storage else { continue };
                    let already_active = active.insert(name.clone(), now).is_some();
                    if already_active {
                        // Short-poll tier is already riding this write burst.
                        continue;
                    }
                    match storage
                        .backend
                        .refresh_lsm_reader_with_poll_interval(active_interval)
                    {
                        Ok(true) => {
                            // Stamps write-promotion recency (not just the
                            // tier flag) so the read-gated sweeper knows this
                            // collection was just promoted for a WRITE reason
                            // and won't demote it out from under the burst.
                            storage.note_write_feed_promotion();
                            metrics::counter!(
                                "helix_reader_feed_refreshes_total",
                                "tier" => "active"
                            )
                            .increment(1);
                        }
                        Ok(false) => {}
                        Err(error) => {
                            active.remove(name.as_str());
                            warn!(
                                collection = name.as_str(),
                                error = %error,
                                "reader change-feed promotion refresh failed"
                            );
                        }
                    }
                }
                since = snapshot.cursor;
                let read_idle_after =
                    crate::helix_engine::storage_core::backend_lsm::reader_idle_after_from_env();
                let expired: Vec<String> = active
                    .iter()
                    .filter(|(_, last)| now.duration_since(**last) > active_ttl)
                    .map(|(name, _)| name.clone())
                    .collect();
                for name in expired {
                    // A collection still being actively read must not be
                    // demoted out from under it just because its last WRITE
                    // signal went quiet — reads own their own freshness via
                    // `note_reader_read_and_maybe_promote`. Leave it in
                    // `active` so this check runs again next tick.
                    if !read_idle_after.is_zero() {
                        let recently_read = collections
                            .read()
                            .ok()
                            .and_then(|guard| {
                                guard.get(&name).map(|entry| Arc::clone(&entry.storage))
                            })
                            .map(|storage| storage.reader_read_recently(read_idle_after))
                            .unwrap_or(false);
                        if recently_read {
                            continue;
                        }
                    }
                    active.remove(&name);
                    let storage = collections
                        .read()
                        .ok()
                        .and_then(|guard| guard.get(&name).map(|entry| Arc::clone(&entry.storage)));
                    let Some(storage) = storage else { continue };
                    match storage
                        .backend
                        .refresh_lsm_reader_with_poll_interval(idle_interval)
                    {
                        Ok(true) => {
                            storage.set_reader_poll_tier_fast(false);
                            metrics::counter!(
                                "helix_reader_feed_refreshes_total",
                                "tier" => "idle"
                            )
                            .increment(1);
                        }
                        Ok(false) => {}
                        Err(error) => {
                            warn!(
                                collection = name.as_str(),
                                error = %error,
                                "reader change-feed demotion refresh failed"
                            );
                        }
                    }
                }
            }
        })
    {
        warn!(error = %err, "Failed to spawn reader change-feed poller");
    }
}

/// Pure decision logic for the read-gated poll-tier demotion sweep: should a
/// collection with these three signals be demoted right now? Kept free of
/// `HelixGraphStorage` so it's unit-testable with plain bools.
/// `write_recent` is always `false` when the write-feed isn't configured
/// (nothing ever calls `note_write_feed_promotion`), so this single rule
/// naturally reduces to the original read-only demotion rule outside feed
/// mode — no separate feed-mode branch needed.
fn reader_poll_tier_sweeper_should_demote(
    is_fast: bool,
    read_recent: bool,
    write_recent: bool,
) -> bool {
    is_fast && !read_recent && !write_recent
}

/// Read-gated reader poll-tier demotion sweep (S3 LIST-cost reduction):
/// most resident reader-replica collections are idle from a READ standpoint
/// at any instant, independent of whether the writer is touching them (see
/// `storage_core::HelixGraphStorage::note_reader_read_and_maybe_promote` for
/// the promotion half, which runs synchronously on the read path). This sweep
/// only ever demotes: periodically walks resident collections and, for any
/// reader-replica collection believed to be on the fast poll tier that hasn't
/// served a read in `HELIX_LSM_READER_IDLE_AFTER_MS` AND wasn't promoted by
/// the write-feed poller within that same window
/// (`reader_poll_tier_sweeper_should_demote`), reopens its `DbReader` at the
/// idle poll interval. One process-wide thread, not one per collection.
///
/// Disabled (never spawned) only when idle-after is `0` (kill switch) or
/// outside LSM reader-replica cold-open mode. Runs in BOTH write-feed modes:
/// standing down entirely when the feed is configured (the earlier design)
/// left read-promoted-then-abandoned collections stuck on the fast tier
/// forever, because the feed poller only ever demotes entries in its OWN
/// write-driven active set — it has no notion of "was this promoted for a
/// read that stopped happening". Per-collection coordination with the feed
/// poller is via `write_promoted_recently`, not by disabling the sweep.
fn spawn_reader_poll_tier_sweeper(
    collections: &Arc<RwLock<HashMap<String, CachedCollection>>>,
    storage_backend: StorageBackendConfig,
) {
    use crate::helix_engine::storage_core::backend_lsm::{
        reader_idle_after_from_env, reader_idle_poll_interval_from_env,
        reader_poll_tier_sweeper_enabled,
    };
    if !lsm_reader_cold_open_enabled(storage_backend) {
        return;
    }
    let idle_after = reader_idle_after_from_env();
    if !reader_poll_tier_sweeper_enabled(idle_after) {
        metrics::counter!("helix_reader_poll_tier_sweeper_total", "outcome" => "disabled")
            .increment(1);
        return;
    }
    let idle_interval = reader_idle_poll_interval_from_env();
    // Granular enough to demote promptly after the idle threshold without
    // spinning; capped so a very small idle-after still gets timely sweeps.
    let tick = (idle_after / 4)
        .max(Duration::from_secs(1))
        .min(Duration::from_secs(30));
    let collections = Arc::downgrade(collections);
    if let Err(err) = thread::Builder::new()
        .name("helix-reader-poll-tier".to_string())
        .spawn(move || loop {
            thread::sleep(tick);
            let Some(collections) = collections.upgrade() else {
                return;
            };
            let snapshot: Vec<Arc<HelixGraphStorage>> = match collections.read() {
                Ok(guard) => guard
                    .values()
                    .map(|entry| Arc::clone(&entry.storage))
                    .collect(),
                Err(_) => continue,
            };
            for storage in snapshot {
                if !storage.backend.is_reader_replica() {
                    continue;
                }
                let is_fast = storage.reader_poll_tier_is_fast();
                let read_recent = storage.reader_read_recently(idle_after);
                let write_recent = storage.write_promoted_recently(idle_after);
                if !reader_poll_tier_sweeper_should_demote(is_fast, read_recent, write_recent) {
                    // Attribute the skip specifically to write-recency only
                    // when it's the sole blocking reason, so the metric
                    // reflects genuine feed-poller/sweeper coordination
                    // rather than every ordinary "not fast"/"read recently"
                    // no-op.
                    if is_fast && !read_recent && write_recent {
                        metrics::counter!(
                            "helix_reader_poll_tier_sweeper_total",
                            "outcome" => "skipped_write_recent"
                        )
                        .increment(1);
                    }
                    continue;
                }
                if !storage.try_begin_poll_tier_transition() {
                    // A promotion just started (e.g. a racing read); let it
                    // win instead of contending for the swap.
                    continue;
                }
                // Re-check every signal under the guard: a read or a
                // write-feed promotion may have landed while this collection
                // sat in the snapshot above.
                if !reader_poll_tier_sweeper_should_demote(
                    true,
                    storage.reader_read_recently(idle_after),
                    storage.write_promoted_recently(idle_after),
                ) {
                    storage.end_poll_tier_transition();
                    continue;
                }
                match storage
                    .backend
                    .refresh_lsm_reader_with_poll_interval(idle_interval)
                {
                    Ok(true) => {
                        storage.set_reader_poll_tier_fast(false);
                        metrics::counter!("helix_reader_poll_tier_total", "direction" => "demote")
                            .increment(1);
                    }
                    Ok(false) => {}
                    Err(error) => {
                        warn!(error = %error, "reader poll-tier demotion refresh failed");
                    }
                }
                storage.end_poll_tier_transition();
            }
        })
    {
        warn!(error = %err, "Failed to spawn reader poll-tier sweeper");
    }
}

fn spawn_page_cache_sweeper(
    collections: &Arc<RwLock<HashMap<String, CachedCollection>>>,
    collections_dir: PathBuf,
) {
    let sweep_ms = page_cache_sweep_ms();
    let batch = page_cache_sweep_batch();
    if sweep_ms == 0 || batch == 0 || !cache_drop_hint_enabled() {
        metrics::counter!(
            "helix_page_cache_sweeper_total",
            "outcome" => "disabled"
        )
        .increment(1);
        return;
    }

    let collections = Arc::downgrade(collections);
    if let Err(err) = thread::Builder::new()
        .name("helix-page-cache".to_string())
        .spawn(move || {
            let sweep = Duration::from_millis(sweep_ms);
            let mut cursor = 0usize;
            loop {
                thread::sleep(sweep);
                let memory = collection_memory_snapshot();
                record_memory_snapshot(memory);

                // Derive both gating facts from the single arbiter so the
                // page-cache sweeper and the idle sweeper can never disagree on
                // the current-vs-high comparison or the reclaimable-file figure.
                // The sweeper drops file cache, so it acts on the opposite side
                // of the exemption: it runs precisely when `current >= high` AND
                // there is a large reclaimable file cache to drop. `pressure_high
                // = None` matches the historical (separate-check) behavior.
                let decision = memory.pressure_decision(
                    collection_open_current_high_bytes(),
                    None,
                    page_cache_min_reclaimable_bytes(),
                );

                if decision.current_high.is_none() {
                    metrics::counter!(
                        "helix_page_cache_sweeper_total",
                        "outcome" => "skipped",
                        "reason" => "current_below_high"
                    )
                    .increment(1);
                    continue;
                }

                if !decision.reclaimable_file_is_large {
                    metrics::counter!(
                        "helix_page_cache_sweeper_total",
                        "outcome" => "skipped",
                        "reason" => "low_reclaimable_file"
                    )
                    .increment(1);
                    continue;
                }

                let Some(collections) = collections.upgrade() else {
                    return;
                };
                let loaded: HashSet<String> = match collections.read() {
                    Ok(collections) => collections.keys().cloned().collect(),
                    Err(err) => {
                        metrics::counter!(
                            "helix_page_cache_sweeper_total",
                            "outcome" => "lock_poisoned"
                        )
                        .increment(1);
                        warn!(error = %err, "Page-cache sweeper lock poisoned");
                        continue;
                    }
                };

                let started = Instant::now();
                let candidates = cold_collection_cache_drop_candidates(
                    &collections_dir,
                    &loaded,
                    &mut cursor,
                    batch,
                );
                if candidates.is_empty() {
                    metrics::counter!(
                        "helix_page_cache_sweeper_total",
                        "outcome" => "skipped",
                        "reason" => "no_cold_collections"
                    )
                    .increment(1);
                    continue;
                }

                let count = candidates.len() as u64;
                for (name, path) in candidates {
                    advise_collection_cache_dropped(&path, &name, "page_cache_pressure");
                }
                metrics::counter!("helix_page_cache_sweeper_collections_total").increment(count);
                metrics::histogram!("helix_page_cache_sweeper_duration_ms")
                    .record(started.elapsed().as_secs_f64() * 1000.0);
                metrics::counter!(
                    "helix_page_cache_sweeper_total",
                    "outcome" => "swept",
                    "reason" => "current_high"
                )
                .increment(1);
                debug!(
                    cold_collections = count,
                    current_mb = memory.current_bytes.unwrap_or(0) / (1024 * 1024),
                    reclaimable_file_mb =
                        memory.reclaimable_file_bytes.unwrap_or(0) / (1024 * 1024),
                    elapsed_ms = started.elapsed().as_millis() as u64,
                    "Page-cache sweeper advised cold collection files"
                );
            }
        })
    {
        metrics::counter!(
            "helix_page_cache_sweeper_total",
            "outcome" => "spawn_error"
        )
        .increment(1);
        warn!(error = %err, "Failed to spawn page-cache sweeper");
    }
}

/// Manages per-collection LMDB environments and collection aliases.
///
/// Each collection gets its own directory and LMDB env under `{data_dir}/collections/{name}/`.
/// Collections are lazily opened on first access and cached with LRU eviction.
/// When the cache exceeds `MAX_OPEN_COLLECTIONS`, the least-recently-accessed
/// environments are closed. The OS page cache ensures only active collections
/// consume RAM; evicted collections are re-opened transparently on next access.
///
/// **Aliases** map a symbolic name to an actual collection name. They're persisted
/// to `{data_dir}/aliases.json` and resolved transparently in `get_collection()`.
/// This enables zero-downtime reindexing via atomic staging→serving swap:
///   1. Index into `myrepo_v2` (staging collection)
///   2. Atomically swap alias `myrepo` from `myrepo_v1` → `myrepo_v2`
///   3. Drop `myrepo_v1` when ready
pub struct CollectionManager {
    data_dir: PathBuf,
    collections: Arc<RwLock<HashMap<String, CachedCollection>>>,
    /// Per-collection open gates. Opening LMDB can take tens of milliseconds
    /// and must be serialized per path because LMDB forbids duplicate opens in
    /// one process. Do not hold the global collection map write lock while
    /// opening: that couples unrelated tenants and lets one cold collection
    /// stall hot read traffic for another collection.
    open_gates: RwLock<HashMap<String, Arc<Mutex<()>>>>,
    /// Collections recently dropped from the cache whose LMDB env may still
    /// be alive through in-flight `Arc<HelixGraphStorage>` holders.
    ///
    /// `Weak` is used so tracking the closing collection does not keep the env
    /// alive on its own. A later reopen waits for the weak ref to expire before
    /// calling `mdb_env_open` on the same path again.
    closing: RwLock<HashMap<String, Weak<HelixGraphStorage>>>,
    /// Collections with a drop in progress (or a drop that failed partway
    /// through teardown). While a name is tombstoned here, every open path
    /// (`create_collection`, cold `get_collection`, `get_or_create_collection`)
    /// returns "Collection not found" instead of waiting on the closing env or
    /// rediscovering the (possibly orphaned) object-store prefix. Without this,
    /// a drop that quiesced SlateDB but failed the S3 purge leaves a zombie:
    /// writers auto-recreate the collection from the orphaned prefix, trip the
    /// process-wide LMDB open registry, and every request loops on
    /// `EnvAlreadyOpen` until the pod restarts.
    ///
    /// Entries are removed when a drop completes successfully. A drop that
    /// errors leaves its tombstone in place so the name keeps returning 404
    /// (honest, non-wedging) until a retried `drop_collection` succeeds.
    dropping: RwLock<HashMap<String, Instant>>,
    /// alias_name → collection_name. Persisted to `{data_dir}/aliases.json`.
    aliases: RwLock<HashMap<String, String>>,
    /// Per-collection snapshot gates: serialise concurrent snapshot requests
    /// for the same collection and cache the most recent successful snapshot
    /// so repeat calls with no intervening writes are cheap. Bounded by the
    /// number of collections ever snapshotted since process start; entries
    /// are never removed (memory is a few hundred bytes per collection).
    snapshot_gates: RwLock<HashMap<String, Arc<Mutex<SnapshotState>>>>,
    /// Per-collection point mutation gates. Upsert/delete requests may be
    /// chunked into many LMDB transactions, but request ordering is scoped to
    /// the whole point mutation for Qdrant-compatible behavior.
    point_mutation_gates: RwLock<HashMap<String, Arc<Mutex<()>>>>,
    write_gates: RwLock<HashMap<String, Arc<Mutex<()>>>>,
    /// Cold LMDB opens currently in progress but not yet visible in the
    /// loaded collection cache. This closes the gap where many concurrent
    /// searches could each pass the cache-size check, then all open envs at
    /// once before LRU eviction had a chance to run.
    open_reservations: AtomicUsize,
    config: Config,
}

/// State held inside a per-collection snapshot mutex.
#[derive(Default)]
struct SnapshotState {
    /// Most recent successful snapshot info. `None` until the first snapshot
    /// completes in this process; we do **not** hydrate from disk on startup
    /// because the cost is a single extra copy after restart.
    last: Option<SnapshotInfo>,
    /// LMDB `last_txn_id` observed immediately after the most recent
    /// successful snapshot completed. Any subsequent LMDB write commit —
    /// whether via the Raft apply pipeline or a direct write transaction —
    /// bumps this counter, so comparing it against the live `last_txn_id` is
    /// how we detect "no writes since last snapshot". Covers both WAL-backed
    /// and bypass-WAL writers uniformly.
    last_post_snapshot_txn_id: usize,
}

/// Stats for a single collection.
#[derive(Debug, serde::Serialize)]
pub struct CollectionStats {
    pub name: String,
    pub schema_version: u32,
    pub node_count: u64,
    pub edge_count: u64,
    pub vector_count: u64,
    pub disk_bytes: u64,
}

#[derive(Debug, serde::Serialize)]
pub struct CollectionStorageBytes {
    pub name: String,
    pub storage_bytes: u64,
    pub source: &'static str,
}

/// Result of an LSM counter recount/repair for a single collection.
#[derive(Debug, serde::Serialize)]
pub struct RecountResult {
    pub name: String,
    pub counters: RecountCounters,
}

/// Result of a ghost payload-index GC run for a single collection.
#[derive(Debug, serde::Serialize)]
pub struct PayloadIndexGcResult {
    pub name: String,
    pub fields: HashMap<String, PayloadIndexGcFieldResult>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SnapshotInfo {
    pub name: String,
    pub lsn: u64,
    pub path: String,
    pub disk_bytes: u64,
    pub created_at_millis: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SnapshotManifest {
    info: SnapshotInfo,
    /// LMDB `last_txn_id` observed immediately after the compact-copy
    /// completed. Persisted so skip-if-no-advance works across process
    /// restarts: the next snapshot call can compare the live txn id against
    /// this value without needing a full copy. `Option` with `#[serde(default)]`
    /// keeps old manifests (written before this field existed) readable; those
    /// cause one cold-start copy and then the new format takes over.
    #[serde(default)]
    txn_id_at_snapshot: Option<u64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct RaftCollectionSnapshot {
    name: String,
    bytes: Vec<u8>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct RaftClusterSnapshot {
    collections: Vec<RaftCollectionSnapshot>,
}

static LAST_SNAPSHOT_MILLIS: AtomicI64 = AtomicI64::new(0);

fn next_snapshot_millis() -> i64 {
    loop {
        let now_millis = chrono::Utc::now().timestamp_millis();
        let last_millis = LAST_SNAPSHOT_MILLIS.load(Ordering::Relaxed);
        let next_millis = now_millis.max(last_millis.saturating_add(1));
        if LAST_SNAPSHOT_MILLIS
            .compare_exchange(
                last_millis,
                next_millis,
                Ordering::SeqCst,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            return next_millis;
        }
    }
}

impl CollectionManager {
    pub fn new(data_dir: PathBuf, config: Config) -> Result<Self, GraphError> {
        let collections_dir = data_dir.join("collections");
        fs::create_dir_all(&collections_dir)?;
        let storage_backend = config.storage_backend;

        // LSM catalog rediscovery: a writer that starts with an empty /data
        // (fresh node, PVC loss, or a reader replica) only knows about
        // collections from the local registry. Reconcile it against the object
        // store so collections that live in S3 but not on local disk are
        // re-registered (lazily — just a marker dir) and become visible to
        // describe / list / cold-open. Best-effort: a listing failure is logged
        // and never blocks startup.
        if storage_backend.is_lsm() {
            Self::reconcile_lsm_catalog(&collections_dir, storage_backend);
        }

        // Load persisted aliases
        let aliases_path = data_dir.join("aliases.json");
        let aliases = if aliases_path.exists() {
            let data = fs::read(&aliases_path)?;
            sonic_rs::from_slice::<HashMap<String, String>>(&data).unwrap_or_default()
        } else {
            HashMap::new()
        };

        let lsm_cache_unbounded = lsm_collection_cache_unbounded(storage_backend);
        let max = max_open_collections();
        let on_disk = count_collections_on_disk(&collections_dir);
        if !lsm_cache_unbounded {
            audit_virtual_address_budget(max, on_disk);
        }
        info!(
            max_open_collections = max,
            on_disk,
            aliases = aliases.len(),
            lsm_cache_unbounded,
            "CollectionManager ready"
        );

        let collections = Arc::new(RwLock::new(HashMap::new()));
        sweep_lsm_orphan_object_cache_dirs(
            &collections,
            &collections_dir,
            "startup_orphan",
            storage_backend,
        );
        // The idle collection sweeper reclaims cold/idle collections — closing the
        // SlateDB handle and freeing the in-memory HNSW/segment state (eviction
        // also removes the object-store cache subtree). It must run even when
        // residency is "unbounded": unbounded only removes the hard MAX_OPEN
        // *count* cap (and its active-set thrash), NOT idle/memory-pressure
        // reclamation. Without it an LSM writer accumulates every accessed
        // collection's RAM until OOM (252 resident => ~40 GiB observed). Idle
        // eviction is governed by HELIX_COLLECTION_IDLE_EVICTION_MS (set 0 to
        // disable), so cold opens stay rare when the TTL sits above the active
        // re-access interval. The page-cache sweeper only advises dropping LMDB
        // mmap pages and has nothing to reclaim on the SlateDB object-store path,
        // so it stays off under LSM.
        spawn_idle_collection_sweeper(&collections);
        spawn_reader_change_feed_poller(&collections, storage_backend);
        spawn_reader_poll_tier_sweeper(&collections, storage_backend);
        if lsm_cache_unbounded {
            metrics::counter!(
                "helix_page_cache_sweeper_total",
                "outcome" => "disabled",
                "reason" => "lsm"
            )
            .increment(1);
        } else {
            spawn_page_cache_sweeper(&collections, collections_dir);
        }

        let manager = Self {
            data_dir,
            collections,
            open_gates: RwLock::new(HashMap::new()),
            closing: RwLock::new(HashMap::new()),
            dropping: RwLock::new(HashMap::new()),
            aliases: RwLock::new(aliases),
            snapshot_gates: RwLock::new(HashMap::new()),
            point_mutation_gates: RwLock::new(HashMap::new()),
            write_gates: RwLock::new(HashMap::new()),
            open_reservations: AtomicUsize::new(0),
            config,
        };
        Ok(manager)
    }

    /// Best-effort: register a local marker dir for every collection that exists
    /// in the LSM object store but not yet in the local registry. Registering the
    /// NAME (an empty `collections/<name>` dir) is enough for `list_collections`,
    /// `collection_exists`, and `get_collection` cold-open to find it; the Db is
    /// not opened here. Never fails: listing or `mkdir` errors are logged and the
    /// node continues (collections are still found lazily on first touch).
    fn reconcile_lsm_catalog(collections_dir: &Path, storage_backend: StorageBackendConfig) {
        match backend_any::list_collection_prefixes_from_env(storage_backend) {
            Ok(names) => {
                let mut registered = 0usize;
                for name in names {
                    if name.is_empty() {
                        continue;
                    }
                    let dir = collections_dir.join(&name);
                    if dir.exists() {
                        continue;
                    }
                    match fs::create_dir_all(&dir) {
                        Ok(()) => registered += 1,
                        Err(e) => warn!(
                            collection = %name,
                            error = %e,
                            "LSM catalog rediscovery: failed to register collection marker dir"
                        ),
                    }
                }
                if registered > 0 {
                    info!(
                        registered,
                        "LSM catalog rediscovery: registered collections from object store"
                    );
                }
            }
            Err(e) => warn!(
                error = %e,
                "LSM catalog rediscovery skipped (object-store listing failed); \
                 collections will be found lazily on first touch"
            ),
        }
    }

    /// On the LSM backend, check whether `name` exists in the object store and, if
    /// so, register a local marker dir at `path` so the normal cold-open path can
    /// serve it (a describe/get of an un-loaded-but-in-S3 collection returns its
    /// config instead of 404). Returns `false` on LMDB, when the collection is
    /// absent in S3, or when the listing fails (treated as not found).
    fn rediscover_lsm_collection(&self, name: &str, path: &Path) -> bool {
        if !self.config.storage_backend.is_lsm() {
            return false;
        }
        match backend_any::lsm_drop_tombstone_exists_from_env(path, self.config.storage_backend) {
            Ok(false) => {}
            Ok(true) => return false,
            Err(e) => {
                warn!(
                    collection = %name,
                    error = %e,
                    "LSM rediscovery: drop tombstone lookup failed; treating as not found"
                );
                return false;
            }
        }
        match backend_any::list_collection_prefixes_from_env(self.config.storage_backend) {
            Ok(names) => {
                if !names.iter().any(|n| n == name) {
                    return false;
                }
                // The collection exists in S3. Register the marker dir so the
                // cold-open path proceeds; even if mkdir fails, cold-open still
                // recreates it (`HelixGraphStorage::new` calls `create_dir_all`).
                if let Err(e) = fs::create_dir_all(path) {
                    warn!(
                        collection = %name,
                        error = %e,
                        "LSM rediscovery: failed to register marker dir for in-S3 collection"
                    );
                }
                true
            }
            Err(e) => {
                warn!(
                    collection = %name,
                    error = %e,
                    "LSM rediscovery: object-store listing failed; treating as not found"
                );
                false
            }
        }
    }

    fn open_gate(&self, name: &str) -> Arc<Mutex<()>> {
        if let Ok(gates) = self.open_gates.read() {
            if let Some(gate) = gates.get(name) {
                return Arc::clone(gate);
            }
        }
        let mut gates = match self.open_gates.write() {
            Ok(gates) => gates,
            Err(_) => return Arc::new(Mutex::new(())),
        };
        Arc::clone(
            gates
                .entry(name.to_string())
                .or_insert_with(|| Arc::new(Mutex::new(()))),
        )
    }

    pub fn point_mutation_gate(&self, name: &str) -> Arc<Mutex<()>> {
        if let Ok(gates) = self.point_mutation_gates.read() {
            if let Some(gate) = gates.get(name) {
                return Arc::clone(gate);
            }
        }
        let mut gates = match self.point_mutation_gates.write() {
            Ok(gates) => gates,
            Err(_) => return Arc::new(Mutex::new(())),
        };
        Arc::clone(
            gates
                .entry(name.to_string())
                .or_insert_with(|| Arc::new(Mutex::new(()))),
        )
    }

    fn write_gate(&self, name: &str) -> Arc<Mutex<()>> {
        if let Ok(gates) = self.write_gates.read() {
            if let Some(gate) = gates.get(name) {
                return Arc::clone(gate);
            }
        }
        let mut gates = match self.write_gates.write() {
            Ok(gates) => gates,
            Err(_) => return Arc::new(Mutex::new(())),
        };
        Arc::clone(
            gates
                .entry(name.to_string())
                .or_insert_with(|| Arc::new(Mutex::new(()))),
        )
    }

    fn open_collection_storage(
        &self,
        path: &Path,
        name: &str,
    ) -> Result<HelixGraphStorage, GraphError> {
        let mut storage = open_storage_with_retry(path, self.config.clone(), name)?;
        storage.set_write_txn_gate(self.write_gate(name));
        storage.seed_lsm_counter_keys()?;
        Ok(storage)
    }

    /// Called under the open gate before (re)creating `name`. If a previous
    /// drop left its object-store prefix un-purged, purge it synchronously now
    /// (the prefix is name-derived, so the new incarnation would otherwise open
    /// the old manifest and a later background retry would delete its objects).
    /// Refuses with a retryable error while the prefix cannot be purged.
    fn resolve_pending_lsm_purge(&self, name: &str, path: &Path) -> Result<(), GraphError> {
        let process_pending = lsm_purge_pending(path);
        let durable_pending = match backend_any::lsm_drop_tombstone_exists_from_env(
            path,
            self.config.storage_backend,
        ) {
            Ok(pending) => pending,
            Err(e) => {
                return Err(GraphError::PurgePending(format!(
                    "Collection '{}' cannot be created yet: object-store drop tombstone \
                         status is unavailable ({}); retry later",
                    name, e
                )));
            }
        };
        if !process_pending && !durable_pending {
            return Ok(());
        }
        match backend_any::purge_lsm_prefix_from_env(path, self.config.storage_backend) {
            Ok(()) => {
                backend_any::clear_lsm_drop_tombstone_from_env(path, self.config.storage_backend)
                    .map_err(|e| {
                    GraphError::PurgePending(format!(
                        "Collection '{}' cannot be created yet: object-store purge completed \
                             but drop tombstone cleanup failed ({}); retry later",
                        name, e
                    ))
                })?;
                lsm_purge_pending_clear(path);
                info!(
                    collection = %name,
                    "Purged pending LSM object-store prefix before recreate"
                );
                Ok(())
            }
            Err(e) => {
                metrics::counter!(
                    "helix_collection_create_refused_purge_pending_total",
                    "collection" => name.to_string()
                )
                .increment(1);
                Err(GraphError::PurgePending(format!(
                    "Collection '{}' cannot be created yet: object-store purge of the \
                     previously dropped collection is still pending ({}); retry later",
                    name, e
                )))
            }
        }
    }

    fn mark_dropping(&self, name: &str) {
        if let Ok(mut dropping) = self.dropping.write() {
            dropping.insert(name.to_string(), Instant::now());
        }
    }

    fn clear_dropping(&self, name: &str) {
        if let Ok(mut dropping) = self.dropping.write() {
            dropping.remove(name);
        }
    }

    fn is_dropping(&self, name: &str) -> bool {
        self.dropping
            .read()
            .map(|dropping| dropping.contains_key(name))
            .unwrap_or(false)
    }

    fn dropping_not_found_error(name: &str) -> GraphError {
        GraphError::New(format!(
            "Collection '{}' not found (drop in progress)",
            name
        ))
    }

    /// Wait for a recently dropped collection env to actually close before
    /// reopening the same path in this process. This avoids LMDB/heed's
    /// process-wide `EnvAlreadyOpened` error on drop→recreate races.
    fn wait_for_closing_collection(&self, name: &str) -> Result<(), GraphError> {
        let budget = Duration::from_millis(open_retry_budget_ms());
        let started = Instant::now();
        let mut backoff = Duration::from_millis(20);
        loop {
            let weak = {
                let closing = self
                    .closing
                    .read()
                    .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
                closing.get(name).cloned()
            };

            let Some(weak) = weak else {
                return Ok(());
            };

            if weak.upgrade().is_none() {
                if let Ok(mut closing) = self.closing.write() {
                    closing.remove(name);
                }
                return Ok(());
            }

            if started.elapsed() >= budget {
                if self.is_dropping(name) {
                    warn!(
                        collection = %name,
                        budget_ms = budget.as_millis() as u64,
                        "Closing env still held for a dropped collection; returning not-found"
                    );
                    return Err(Self::dropping_not_found_error(name));
                }
                warn!(
                    collection = %name,
                    budget_ms = budget.as_millis() as u64,
                    "Timed out waiting for closing collection env to release"
                );
                return Err(GraphError::EnvAlreadyOpen);
            }

            thread::sleep(backoff);
            backoff = (backoff * 2).min(Duration::from_millis(200));
        }
    }

    /// Fetch (or lazily insert) the per-collection snapshot gate. A poisoned
    /// map lock falls back to an orphan gate so snapshot requests still make
    /// progress while ops investigates.
    fn snapshot_gate(&self, name: &str) -> Arc<Mutex<SnapshotState>> {
        if let Ok(map) = self.snapshot_gates.read() {
            if let Some(gate) = map.get(name) {
                return Arc::clone(gate);
            }
        }
        let mut map = match self.snapshot_gates.write() {
            Ok(g) => g,
            Err(_) => return Arc::new(Mutex::new(SnapshotState::default())),
        };
        Arc::clone(
            map.entry(name.to_string())
                .or_insert_with(|| Arc::new(Mutex::new(SnapshotState::default()))),
        )
    }

    fn collection_path(&self, name: &str) -> PathBuf {
        self.data_dir.join("collections").join(name)
    }

    fn aliases_path(&self) -> PathBuf {
        self.data_dir.join("aliases.json")
    }

    /// Persist the current alias map to disk.
    fn persist_aliases(&self) -> Result<(), GraphError> {
        let aliases = self
            .aliases
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        let data = sonic_rs::to_vec(&*aliases)?;
        fs::write(self.aliases_path(), data)?;
        Ok(())
    }

    /// Resolve a name through the alias map. If `name` is an alias, returns
    /// the target collection name. Otherwise returns `name` unchanged.
    pub fn resolve_alias(&self, name: &str) -> String {
        self.aliases
            .read()
            .ok()
            .and_then(|a| a.get(name).cloned())
            .unwrap_or_else(|| name.to_string())
    }

    /// Return true when a collection exists in memory or on disk without
    /// opening its LMDB environment. This is for cheap API probes such as
    /// idempotent creates and Qdrant `/exists` checks.
    pub fn collection_exists(&self, name: &str) -> Result<bool, GraphError> {
        let resolved = self.resolve_alias(name);
        let name = resolved.as_str();

        {
            let collections = self
                .collections
                .read()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            if collections.contains_key(name) {
                return Ok(true);
            }
        }

        Ok(self.collection_path(name).exists())
    }

    /// Fetch a collection only if it is already loaded. Unlike
    /// `get_collection`, this never opens storage and never participates in cold
    /// open admission. Use it for optional optimizations/diagnostics that must
    /// not turn metadata traffic into storage pressure.
    pub fn get_loaded_collection(
        &self,
        name: &str,
    ) -> Result<Option<Arc<HelixGraphStorage>>, GraphError> {
        let resolved = self.resolve_alias(name);
        let name = resolved.as_str();

        let collections = self
            .collections
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        if let Some(entry) = collections.get(name) {
            if matches!(
                entry.storage.unhealthy_lsm_writer_close_reason(),
                Some(CloseReason::Fenced)
            ) {
                return Err(record_fenced_writer_quarantine(name, CloseReason::Fenced));
            }
            touch_cached_collection(entry);
            return Ok(Some(Arc::clone(&entry.storage)));
        }
        Ok(None)
    }

    /// Like [`get_loaded_collection`] but does NOT touch the LRU/idle timer.
    /// Background scans (optimizer debt sweep, diagnostics) must use this: a
    /// periodic scan of every loaded collection through `get_loaded_collection`
    /// resets each idle timer on every pass, so collections never reach the
    /// idle-age threshold and `evict_idle_for_age` can never reclaim them.
    /// Only genuine foreground access should refresh the idle timer.
    pub fn peek_loaded_collection(
        &self,
        name: &str,
    ) -> Result<Option<Arc<HelixGraphStorage>>, GraphError> {
        let resolved = self.resolve_alias(name);
        let name = resolved.as_str();

        let collections = self
            .collections
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        Ok(collections
            .get(name)
            .map(|entry| Arc::clone(&entry.storage)))
    }

    /// Snapshot currently loaded collections without opening cold LMDB envs.
    /// Intended for background diagnostics/optimizer scheduling; unlike
    /// `get_loaded_collection`, this does not touch LRU state.
    pub fn loaded_collections_snapshot(
        &self,
    ) -> Result<Vec<(String, Arc<HelixGraphStorage>)>, GraphError> {
        let collections = self
            .collections
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        Ok(collections
            .iter()
            .map(|(name, entry)| (name.clone(), Arc::clone(&entry.storage)))
            .collect())
    }

    pub fn unhealthy_lsm_writer_close_reasons(
        &self,
    ) -> Result<Vec<(String, CloseReason)>, GraphError> {
        let collections = self
            .collections
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        Ok(collections
            .iter()
            .filter_map(|(name, entry)| {
                entry
                    .storage
                    .unhealthy_lsm_writer_close_reason()
                    .map(|reason| (name.clone(), reason))
            })
            .collect())
    }

    /// Names of currently-loaded collections WITHOUT cloning their storage
    /// `Arc`s. Used by the auto-compaction sweep: holding an `Arc` clone of the
    /// collection it is about to compact would pin it above `strong_count == 1`
    /// and starve the eviction-drain (the v2 smoke deadlock). Re-fetch per name
    /// via `get_collection` only for the brief measurement, then drop.
    pub fn loaded_collection_names(&self) -> Result<Vec<String>, GraphError> {
        let collections = self
            .collections
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        Ok(collections.keys().cloned().collect())
    }

    /// Names of currently-loaded collections, most-recently-accessed first.
    /// Like [`loaded_collection_names`] this clones no storage `Arc`s and does
    /// not touch LRU state; used to persist the reader warm hot-list.
    pub fn loaded_collection_names_by_recency(&self) -> Result<Vec<String>, GraphError> {
        let collections = self
            .collections
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        let mut names: Vec<(u64, String)> = collections
            .iter()
            .map(|(name, entry)| (entry.last_access.load(Ordering::Relaxed), name.clone()))
            .collect();
        drop(collections);
        names.sort_by(|a, b| b.0.cmp(&a.0));
        Ok(names.into_iter().map(|(_, name)| name).collect())
    }

    /// Admission for automatic maintenance that would open a cold collection.
    ///
    /// Foreground traffic still uses `get_collection` and can wait/retry through
    /// normal open admission. Background optimizers are lower priority: when
    /// the process is already high on loaded envs, memory, or file descriptors,
    /// they skip this sweep instead of competing with tenant requests.
    pub fn maintenance_cold_open_admission(&self, name: &str) -> MaintenanceAdmission {
        let resolved = self.resolve_alias(name);
        let name = resolved.as_str();
        if lsm_collection_cache_unbounded(self.config.storage_backend) {
            metrics::counter!(
                "helix_maintenance_cold_open_admission_total",
                "outcome" => "admitted",
                "reason" => "lsm"
            )
            .increment(1);
            return MaintenanceAdmission::Admitted;
        }
        let max_open = max_open_collections();
        let loaded_limit =
            maintenance_open_loaded_limit(max_open, maintenance_open_max_loaded_pct());
        let opening = self.open_reservations.load(Ordering::Acquire);
        let loaded = match self.collections.read() {
            Ok(collections) => {
                if collections.contains_key(name) {
                    metrics::counter!(
                        "helix_maintenance_cold_open_admission_total",
                        "outcome" => "admitted",
                        "reason" => "already_loaded"
                    )
                    .increment(1);
                    return MaintenanceAdmission::Admitted;
                }
                collections.len()
            }
            Err(err) => {
                warn!(
                    collection = %name,
                    error = %err,
                    "maintenance cold-open admission lock poisoned"
                );
                metrics::counter!(
                    "helix_maintenance_cold_open_admission_total",
                    "outcome" => "rejected",
                    "reason" => "lock_poisoned"
                )
                .increment(1);
                return MaintenanceAdmission::Rejected("lock_poisoned");
            }
        };

        metrics::gauge!("helix_maintenance_loaded_collections").set(loaded as f64);
        metrics::gauge!("helix_maintenance_opening_collections").set(opening as f64);
        metrics::gauge!("helix_maintenance_loaded_collection_limit").set(loaded_limit as f64);

        if loaded.saturating_add(opening) >= loaded_limit {
            metrics::counter!(
                "helix_maintenance_cold_open_admission_total",
                "outcome" => "rejected",
                "reason" => "open_capacity"
            )
            .increment(1);
            return MaintenanceAdmission::Rejected("open_capacity");
        }

        match maintenance_process_resource_admission() {
            MaintenanceAdmission::Admitted => {
                metrics::counter!(
                    "helix_maintenance_cold_open_admission_total",
                    "outcome" => "admitted",
                    "reason" => "resources"
                )
                .increment(1);
                MaintenanceAdmission::Admitted
            }
            MaintenanceAdmission::Rejected(reason) => {
                metrics::counter!(
                    "helix_maintenance_cold_open_admission_total",
                    "outcome" => "rejected",
                    "reason" => reason
                )
                .increment(1);
                MaintenanceAdmission::Rejected(reason)
            }
        }
    }

    pub fn collection_metadata_sidecar(
        &self,
        name: &str,
    ) -> Result<Option<StorageMetadataSidecar>, GraphError> {
        let resolved = self.resolve_alias(name);
        let path = self.collection_path(&resolved);
        HelixGraphStorage::read_metadata_sidecar_from_path_or_lsm(
            &path,
            self.config.storage_backend,
        )
    }

    pub fn collection_storage_bytes(
        &self,
        name: &str,
    ) -> Result<CollectionStorageBytes, GraphError> {
        let resolved = self.resolve_alias(name);
        if let Some(storage) = self.get_loaded_collection(&resolved)? {
            if let Some(storage_bytes) = storage
                .backend
                .lsm_storage_bytes()
                .map_err(graph_error_from_backend_error)?
            {
                return Ok(CollectionStorageBytes {
                    name: name.to_string(),
                    storage_bytes,
                    source: "lsm_object_prefix",
                });
            }
        }

        let stats = self.collection_stats(&resolved)?;
        Ok(CollectionStorageBytes {
            name: name.to_string(),
            storage_bytes: stats.disk_bytes,
            source: "collection_stats",
        })
    }

    /// Cheap on-disk check for the persistent degraded marker without opening
    /// the LMDB env. Mirrors `collection_metadata_sidecar`: lets the idempotent
    /// create fast-path refuse to ACK success for a collection that a fatal
    /// storage error quarantined while it is evicted/cold.
    pub fn collection_degraded_marker(&self, name: &str) -> Option<(&'static str, String)> {
        let resolved = self.resolve_alias(name);
        let path = self.collection_path(&resolved);
        HelixGraphStorage::read_degraded_marker(path.to_str()?)
    }

    /// Create an alias pointing to a collection. The collection must exist.
    /// If the alias already exists, it is overwritten (atomic re-point).
    pub fn create_alias(&self, alias_name: &str, collection_name: &str) -> Result<(), GraphError> {
        // Validate: target collection must exist (on disk).
        let target_path = self.collection_path(collection_name);
        if !target_path.exists() {
            return Err(GraphError::New(format!(
                "Cannot create alias '{}': collection '{}' does not exist",
                alias_name, collection_name
            )));
        }
        // An alias name must not collide with an actual collection directory.
        let alias_as_collection = self.collection_path(alias_name);
        if alias_as_collection.exists() {
            return Err(GraphError::New(format!(
                "Cannot create alias '{}': a collection with that name already exists",
                alias_name
            )));
        }

        {
            let mut aliases = self
                .aliases
                .write()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            aliases.insert(alias_name.to_string(), collection_name.to_string());
        }
        self.persist_aliases()
    }

    /// Delete an alias. Returns error if the alias doesn't exist.
    pub fn delete_alias(&self, alias_name: &str) -> Result<(), GraphError> {
        {
            let mut aliases = self
                .aliases
                .write()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            if aliases.remove(alias_name).is_none() {
                return Err(GraphError::New(format!(
                    "Alias '{}' does not exist",
                    alias_name
                )));
            }
        }
        self.persist_aliases()
    }

    /// List all aliases as (alias_name, collection_name) pairs.
    pub fn list_aliases(&self) -> Result<Vec<(String, String)>, GraphError> {
        let aliases = self
            .aliases
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        let mut list: Vec<(String, String)> = aliases
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        list.sort();
        Ok(list)
    }

    /// List aliases that point to a specific collection.
    pub fn aliases_for_collection(&self, collection_name: &str) -> Result<Vec<String>, GraphError> {
        let aliases = self
            .aliases
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        let mut result: Vec<String> = aliases
            .iter()
            .filter(|(_, v)| v.as_str() == collection_name)
            .map(|(k, _)| k.clone())
            .collect();
        result.sort();
        Ok(result)
    }

    fn snapshot_path(&self, name: &str, created_at_millis: i64) -> PathBuf {
        self.collection_path(name)
            .join("snapshots")
            .join(format!("{:020}.mdb", created_at_millis))
    }

    fn snapshot_manifest_path(snapshot_path: &Path) -> PathBuf {
        snapshot_path.with_extension("json")
    }

    fn evict_idle_for_capacity(
        collections: &mut HashMap<String, CachedCollection>,
        opening: usize,
    ) -> u64 {
        let max = max_open_collections();
        if collections.len().saturating_add(opening) < max {
            return 0;
        }

        let needed = collections
            .len()
            .saturating_add(opening)
            .saturating_sub(max)
            .saturating_add(1);
        // Evict at least what is needed, but batch up to 25% of the cache to
        // avoid open/evict thrash on many-tenant workloads.
        let evict_count = (max / 4).max(needed).max(1);
        if evict_count > 0 {
            debug!(
                evicted = evict_count,
                cached = collections.len(),
                opening,
                max,
                "LRU evicting cold collections"
            );
        }

        Self::evict_lru_idle(collections, evict_count, "lru")
    }

    fn evict_lru_idle(
        collections: &mut HashMap<String, CachedCollection>,
        count: usize,
        reason: &'static str,
    ) -> u64 {
        if count == 0 {
            return 0;
        }

        let mut candidates: Vec<(String, u64)> = collections
            .iter()
            .filter(|(_, entry)| Arc::strong_count(&entry.storage) <= 1)
            .map(|(k, entry)| (k.clone(), entry.last_access.load(Ordering::Relaxed)))
            .collect();
        candidates.sort_by_key(|(_, ts)| *ts);

        let mut evicted = 0u64;
        for (key, _) in candidates.into_iter().take(count) {
            if let Some(entry) = collections.remove(&key) {
                let weak = Arc::downgrade(&entry.storage);
                let path = entry.path.clone();
                drop(entry);
                schedule_cache_drop_after_close(weak, path, key, reason);
                evicted += 1;
            }
        }
        if evicted > 0 {
            metrics::counter!("helix_collection_cache_evictions_total", "reason" => reason)
                .increment(evicted);
        }
        evicted
    }

    fn evict_idle_for_memory(
        collections: &mut HashMap<String, CachedCollection>,
        reason: &'static str,
    ) -> u64 {
        let evict_count = (collections.len() / 4).max(1);
        Self::evict_lru_idle(collections, evict_count, reason)
    }

    fn evict_idle_by_age(
        collections: &mut HashMap<String, CachedCollection>,
        now_millis: u64,
        idle_ms: u64,
        count: usize,
        min_loaded: usize,
        reason: &'static str,
    ) -> u64 {
        if idle_ms == 0 || count == 0 || collections.len() <= min_loaded {
            return 0;
        }

        let count = count.min(collections.len().saturating_sub(min_loaded));
        let mut candidates: Vec<(String, u64, u64)> = collections
            .iter()
            .filter(|(_, entry)| Arc::strong_count(&entry.storage) <= 1)
            .filter_map(|(key, entry)| {
                let last_touch = entry.last_access_millis.load(Ordering::Relaxed);
                let idle_for = now_millis.saturating_sub(last_touch);
                if idle_for >= idle_ms {
                    Some((
                        key.clone(),
                        last_touch,
                        entry.last_access.load(Ordering::Relaxed),
                    ))
                } else {
                    None
                }
            })
            .collect();
        candidates.sort_by_key(|(_, last_touch, lru)| (*last_touch, *lru));

        let mut evicted = 0u64;
        for (key, _, _) in candidates.into_iter().take(count) {
            if let Some(entry) = collections.remove(&key) {
                let weak = Arc::downgrade(&entry.storage);
                let path = entry.path.clone();
                drop(entry);
                schedule_cache_drop_after_close(weak, path, key, reason);
                evicted += 1;
            }
        }
        if evicted > 0 {
            metrics::counter!("helix_collection_cache_evictions_total", "reason" => reason)
                .increment(evicted);
        }
        evicted
    }

    fn evict_idle_for_age(collections: &mut HashMap<String, CachedCollection>) -> u64 {
        let Some(now_millis) = current_time_millis() else {
            metrics::counter!(
                "helix_collection_idle_sweeper_total",
                "outcome" => "clock_error"
            )
            .increment(1);
            return 0;
        };

        Self::evict_idle_by_age(
            collections,
            now_millis,
            collection_idle_eviction_ms(),
            collection_idle_eviction_per_sweep(),
            collection_idle_eviction_min_loaded(),
            "idle",
        )
    }

    /// Evict cached collections whose LSM writer backend was closed by an
    /// internal panic. A genuinely fenced writer is different: reopening in the
    /// same process would claim a new writer epoch and could fence the active
    /// writer, so fenced entries stay resident, unhealthy, and quarantined until
    /// process replacement.
    fn evict_unhealthy_writers(collections: &mut HashMap<String, CachedCollection>) -> u64 {
        let unhealthy: Vec<(String, CloseReason)> = collections
            .iter()
            .filter_map(|(name, entry)| {
                entry
                    .storage
                    .unhealthy_lsm_writer_close_reason()
                    .map(|reason| (name.clone(), reason))
            })
            .collect();

        let mut evicted = 0u64;
        for (name, reason) in unhealthy {
            if matches!(reason, CloseReason::Fenced) {
                let error = record_fenced_writer_quarantine(&name, reason);
                warn!(
                    collection = %name,
                    close_reason = ?reason,
                    error = %error,
                    "Quarantined fenced LSM writer without reopening in this process"
                );
                continue;
            }
            if let Some(entry) = collections.remove(&name) {
                let weak = Arc::downgrade(&entry.storage);
                let path = entry.path.clone();
                drop(entry);
                warn!(
                    collection = %name,
                    close_reason = ?reason,
                    "Evicted collection with dead LSM writer; next access reopens under a fresh epoch"
                );
                schedule_cache_drop_after_close(weak, path, name, "unhealthy_writer");
                evicted += 1;
            }
        }
        if evicted > 0 {
            metrics::counter!(
                "helix_collection_cache_evictions_total",
                "reason" => "unhealthy_writer"
            )
            .increment(evicted);
        }
        evicted
    }

    /// Insert a collection into the cache, evicting cold entries if full.
    fn cache_insert(
        collections: &mut HashMap<String, CachedCollection>,
        name: String,
        storage: Arc<HelixGraphStorage>,
        opening: usize,
    ) {
        if storage.backend.kind() != BackendKind::Lsm || !lsm_collection_cache_unbounded_enabled() {
            Self::evict_idle_for_capacity(collections, opening);
        }
        let tick = LRU_CLOCK.fetch_add(1, Ordering::Relaxed);
        let now_millis = current_time_millis().unwrap_or(u64::MAX);
        let path = storage
            .lmdb_env()
            .map(|env| env.path().to_path_buf())
            .unwrap_or_else(|_| storage.path().to_path_buf());
        collections.insert(
            name,
            CachedCollection {
                storage,
                path,
                last_access: AtomicU64::new(tick),
                last_access_millis: AtomicU64::new(now_millis),
            },
        );
    }

    fn reserve_open_slot(&self, name: &str) -> Result<OpenSlotReservation<'_>, GraphError> {
        if lsm_collection_cache_unbounded(self.config.storage_backend) {
            metrics::counter!(
                "helix_collection_open_admission_total",
                "outcome" => "admitted",
                "reason" => "lsm"
            )
            .increment(1);
            return Ok(OpenSlotReservation { opening: None });
        }
        let started = Instant::now();
        let budget = Duration::from_millis(collection_open_wait_ms());
        let poll = Duration::from_millis(collection_open_poll_ms());
        let max = max_open_collections();
        let high_mem = collection_open_memory_high_bytes();
        let mut wait_recorded = false;

        loop {
            let memory = collection_memory_snapshot();
            record_memory_snapshot(memory);
            let reclaimable_file_floor = page_cache_min_reclaimable_bytes();
            // Single arbiter: one decision feeds both the block path
            // (`requires_action`) and the reclaimable-file-cache bypass log
            // (`current_high`), so the cold-open guard cannot disagree with
            // itself or the sweepers on the current-vs-high comparison.
            let decision = memory.pressure_decision(
                collection_open_current_high_bytes(),
                high_mem,
                reclaimable_file_floor,
            );
            let current_high = decision.current_high;
            if let Some((current, high)) = decision.requires_action() {
                let evicted = {
                    let mut collections = self
                        .collections
                        .write()
                        .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
                    let evicted = Self::evict_idle_for_memory(&mut collections, "current_memory");
                    metrics::gauge!("helix_collection_cache_pinned").set(
                        collections
                            .values()
                            .filter(|entry| Arc::strong_count(&entry.storage) > 1)
                            .count() as f64,
                    );
                    evicted
                };
                if evicted > 0 {
                    metrics::counter!(
                        "helix_collection_open_admission_total",
                        "outcome" => "evicted",
                        "reason" => "current_memory"
                    )
                    .increment(evicted);
                    thread::sleep(poll);
                    continue;
                }
                if !wait_recorded {
                    metrics::counter!(
                        "helix_collection_open_admission_total",
                        "outcome" => "waiting",
                        "reason" => "current_memory"
                    )
                    .increment(1);
                    wait_recorded = true;
                }
                if started.elapsed() >= budget {
                    metrics::histogram!(
                        "helix_collection_open_reserve_wait_ms",
                        "outcome" => "rejected",
                        "reason" => "current_memory"
                    )
                    .record(started.elapsed().as_secs_f64() * 1000.0);
                    metrics::counter!(
                        "helix_collection_open_admission_total",
                        "outcome" => "rejected",
                        "reason" => "current_memory"
                    )
                    .increment(1);
                    warn!(
                        collection = %name,
                        current_mb = current / (1024 * 1024),
                        high_mb = high / (1024 * 1024),
                        pressure_mb = memory.pressure_bytes().unwrap_or(0) / (1024 * 1024),
                        working_set_mb = memory.working_set_bytes.unwrap_or(0) / (1024 * 1024),
                        file_mb = memory.file_bytes.unwrap_or(0) / (1024 * 1024),
                        reclaimable_file_mb = memory.reclaimable_file_bytes.unwrap_or(0) / (1024 * 1024),
                        waited_ms = started.elapsed().as_millis() as u64,
                        "Rejecting cold collection open near cgroup memory ceiling"
                    );
                    return Err(GraphError::StorageError(format!(
                        "collection open current memory high: current={}MB high={}MB pressure={}MB working_set={}MB file={}MB reclaimable_file={}MB",
                        current / (1024 * 1024),
                        high / (1024 * 1024),
                        memory.pressure_bytes().unwrap_or(0) / (1024 * 1024),
                        memory.working_set_bytes.unwrap_or(0) / (1024 * 1024),
                        memory.file_bytes.unwrap_or(0) / (1024 * 1024),
                        memory.reclaimable_file_bytes.unwrap_or(0) / (1024 * 1024)
                    )));
                }
                thread::sleep(poll);
                continue;
            } else if let Some((current, high)) = current_high {
                metrics::counter!(
                    "helix_collection_open_admission_total",
                    "outcome" => "bypassed",
                    "reason" => "reclaimable_file_cache"
                )
                .increment(1);
                debug!(
                    collection = %name,
                    current_mb = current / (1024 * 1024),
                    high_mb = high / (1024 * 1024),
                    pressure_mb = memory.pressure_bytes().unwrap_or(0) / (1024 * 1024),
                    reclaimable_file_mb =
                        memory.reclaimable_file_bytes.unwrap_or(0) / (1024 * 1024),
                    "Bypassing hard current-memory guard because pressure is reclaimable file cache"
                );
            }
            if let (Some(pressure), Some(high)) = (memory.pressure_bytes(), high_mem) {
                if pressure >= high {
                    let evicted = {
                        let mut collections = self
                            .collections
                            .write()
                            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
                        let evicted = Self::evict_idle_for_memory(&mut collections, "memory");
                        metrics::gauge!("helix_collection_cache_pinned").set(
                            collections
                                .values()
                                .filter(|entry| Arc::strong_count(&entry.storage) > 1)
                                .count() as f64,
                        );
                        evicted
                    };
                    if evicted > 0 {
                        metrics::counter!(
                            "helix_collection_open_admission_total",
                            "outcome" => "evicted",
                            "reason" => "memory"
                        )
                        .increment(evicted);
                        thread::sleep(poll);
                        continue;
                    }
                    if !wait_recorded {
                        metrics::counter!(
                            "helix_collection_open_admission_total",
                            "outcome" => "waiting",
                            "reason" => "memory"
                        )
                        .increment(1);
                        wait_recorded = true;
                    }
                    if started.elapsed() >= budget {
                        metrics::histogram!(
                            "helix_collection_open_reserve_wait_ms",
                            "outcome" => "rejected",
                            "reason" => "memory"
                        )
                        .record(started.elapsed().as_secs_f64() * 1000.0);
                        metrics::counter!(
                            "helix_collection_open_admission_total",
                            "outcome" => "rejected",
                            "reason" => "memory"
                        )
                        .increment(1);
                        warn!(
                            collection = %name,
                            pressure_mb = pressure / (1024 * 1024),
                            current_mb = memory.current_bytes.unwrap_or(0) / (1024 * 1024),
                            working_set_mb = memory.working_set_bytes.unwrap_or(0) / (1024 * 1024),
                            file_mb = memory.file_bytes.unwrap_or(0) / (1024 * 1024),
                            active_file_mb = memory.active_file_bytes.unwrap_or(0) / (1024 * 1024),
                            inactive_file_mb = memory.inactive_file_bytes.unwrap_or(0) / (1024 * 1024),
                            reclaimable_file_mb = memory.reclaimable_file_bytes.unwrap_or(0) / (1024 * 1024),
                            dirty_file_mb = memory.file_dirty_bytes.unwrap_or(0) / (1024 * 1024),
                            writeback_file_mb = memory.file_writeback_bytes.unwrap_or(0) / (1024 * 1024),
                            high_mb = high / (1024 * 1024),
                            waited_ms = started.elapsed().as_millis() as u64,
                            "Rejecting cold collection open under memory pressure"
                        );
                        return Err(GraphError::StorageError(format!(
                            "collection open memory pressure: pressure={}MB current={}MB working_set={}MB file={}MB reclaimable_file={}MB high={}MB",
                            pressure / (1024 * 1024),
                            memory.current_bytes.unwrap_or(0) / (1024 * 1024),
                            memory.working_set_bytes.unwrap_or(0) / (1024 * 1024),
                            memory.file_bytes.unwrap_or(0) / (1024 * 1024),
                            memory.reclaimable_file_bytes.unwrap_or(0) / (1024 * 1024),
                            high / (1024 * 1024)
                        )));
                    }
                    thread::sleep(poll);
                    continue;
                }
            }

            {
                let opening = self.open_reservations.load(Ordering::Acquire);
                let mut collections = self
                    .collections
                    .write()
                    .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
                Self::evict_idle_for_capacity(&mut collections, opening);
            }

            let loaded = self
                .collections
                .read()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?
                .len();
            let mut opening = self.open_reservations.load(Ordering::Acquire);
            while loaded.saturating_add(opening) < max {
                match self.open_reservations.compare_exchange(
                    opening,
                    opening + 1,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        metrics::gauge!("helix_collection_open_inflight").set((opening + 1) as f64);
                        metrics::histogram!(
                            "helix_collection_open_reserve_wait_ms",
                            "outcome" => "admitted",
                            "reason" => "capacity"
                        )
                        .record(started.elapsed().as_secs_f64() * 1000.0);
                        metrics::counter!(
                            "helix_collection_open_admission_total",
                            "outcome" => "admitted",
                            "reason" => "capacity"
                        )
                        .increment(1);
                        return Ok(OpenSlotReservation {
                            opening: Some(&self.open_reservations),
                        });
                    }
                    Err(actual) => opening = actual,
                }
            }

            if !wait_recorded {
                metrics::counter!(
                    "helix_collection_open_admission_total",
                    "outcome" => "waiting",
                    "reason" => "capacity"
                )
                .increment(1);
                wait_recorded = true;
            }
            if started.elapsed() >= budget {
                let pinned = self
                    .collections
                    .read()
                    .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?
                    .values()
                    .filter(|entry| Arc::strong_count(&entry.storage) > 1)
                    .count();
                metrics::histogram!(
                    "helix_collection_open_reserve_wait_ms",
                    "outcome" => "rejected",
                    "reason" => "capacity"
                )
                .record(started.elapsed().as_secs_f64() * 1000.0);
                metrics::counter!(
                    "helix_collection_open_admission_total",
                    "outcome" => "rejected",
                    "reason" => "capacity"
                )
                .increment(1);
                metrics::gauge!("helix_collection_cache_pinned").set(pinned as f64);
                warn!(
                    collection = %name,
                    loaded,
                    opening,
                    pinned,
                    max,
                    waited_ms = started.elapsed().as_millis() as u64,
                    "Rejecting cold collection open because cache capacity is saturated"
                );
                return Err(GraphError::StorageError(format!(
                    "collection open capacity saturated: loaded={} opening={} max={}",
                    loaded, opening, max
                )));
            }

            thread::sleep(poll);
        }
    }

    /// Create a new collection. Returns error if it already exists.
    ///
    /// Serializes opens per collection name, but does not hold the global
    /// collections write lock while opening LMDB.
    pub fn create_collection(&self, name: &str) -> Result<Arc<HelixGraphStorage>, GraphError> {
        let open_gate = self.open_gate(name);
        let _open_guard = open_gate
            .lock()
            .map_err(|e| GraphError::New(format!("Open gate poisoned: {}", e)))?;

        if self.is_dropping(name) {
            return Err(Self::dropping_not_found_error(name));
        }

        // Wait for any pending close on this name to drain BEFORE taking the
        // collections write lock. Holding the write lock across a multi-second
        // wait would stall every other collection operation in the pod.
        self.wait_for_closing_collection(name)?;

        {
            let collections = self
                .collections
                .read()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            if collections.contains_key(name) {
                return Err(GraphError::New(format!(
                    "Collection '{}' already exists",
                    name
                )));
            }
        }

        let path = self.collection_path(name);
        if path.exists() {
            return Err(GraphError::New(format!(
                "Collection '{}' already exists",
                name
            )));
        }
        self.resolve_pending_lsm_purge(name, &path)?;

        let open_slot = self.reserve_open_slot(name)?;
        let storage = self.open_collection_storage(&path, name)?;
        let storage = Arc::new(storage);
        super::replication::attach_dirty_retired_reaper_hook(&storage);
        super::replication::submit_post_open_vector_maintenance(&storage, &self.config, name)?;

        let mut collections = self
            .collections
            .write()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        if collections.contains_key(name) {
            return Err(GraphError::New(format!(
                "Collection '{}' already exists",
                name
            )));
        }
        Self::cache_insert(
            &mut collections,
            name.to_string(),
            Arc::clone(&storage),
            self.open_reservations.load(Ordering::Acquire),
        );
        drop(open_slot);
        open_failure_clear(name);

        Ok(storage)
    }

    /// Get a collection by name or alias. Resolves aliases transparently,
    /// then lazy-opens the underlying collection if the dir exists but isn't loaded.
    ///
    /// The LMDB environment open is performed **inside** the write lock to prevent
    /// two threads from racing to open the same environment concurrently
    /// (LMDB forbids duplicate `mdb_env_open` on the same path within a process).
    pub fn get_collection(&self, name: &str) -> Result<Arc<HelixGraphStorage>, GraphError> {
        // Resolve alias → real collection name.
        let resolved = self.resolve_alias(name);
        let name = resolved.as_str();

        // Fast path: check read lock, bump LRU counter.
        {
            let collections = self
                .collections
                .read()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            if let Some(entry) = collections.get(name) {
                if matches!(
                    entry.storage.unhealthy_lsm_writer_close_reason(),
                    Some(CloseReason::Fenced)
                ) {
                    metrics::counter!("helix_collection_cache_hits_total").increment(1);
                    return Err(record_fenced_writer_quarantine(name, CloseReason::Fenced));
                }
                touch_cached_collection(entry);
                entry
                    .storage
                    .maybe_refresh_reader_view(reader_refresh_ttl_ms());
                let storage = Arc::clone(&entry.storage);
                metrics::counter!("helix_collection_cache_hits_total").increment(1);
                // Drop the collections-map read guard before scheduling
                // poll-tier promotion so a slow S3 reopen cannot
                // head-of-line-block cold-opens/evictions waiting on the
                // writer side of this lock.
                drop(collections);
                storage.note_reader_read_and_maybe_promote();
                return Ok(storage);
            }
        }

        let cold_open_started = Instant::now();
        let open_gate = self.open_gate(name);
        let _open_guard = match open_gate.lock() {
            Ok(guard) => guard,
            Err(e) => {
                record_cold_open_duration(cold_open_started, "open_gate_poisoned");
                return Err(GraphError::New(format!("Open gate poisoned: {}", e)));
            }
        };

        if self.is_dropping(name) {
            record_cold_open_duration(cold_open_started, "dropping");
            return Err(Self::dropping_not_found_error(name));
        }

        // Wait for any pending close on this name to drain while holding only
        // this collection's open gate, not the global collection map lock.
        if let Err(err) = self.wait_for_closing_collection(name) {
            record_cold_open_duration(cold_open_started, "closing_wait_error");
            return Err(err);
        }

        // Double-check: another thread may have opened it while we waited.
        {
            let collections = self
                .collections
                .read()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            if let Some(existing) = collections.get(name) {
                if matches!(
                    existing.storage.unhealthy_lsm_writer_close_reason(),
                    Some(CloseReason::Fenced)
                ) {
                    metrics::counter!("helix_collection_cache_hits_total").increment(1);
                    return Err(record_fenced_writer_quarantine(name, CloseReason::Fenced));
                }
                touch_cached_collection(existing);
                metrics::counter!("helix_collection_cache_hits_total").increment(1);
                record_cold_open_duration(cold_open_started, "race_hit_after_gate");
                return Ok(Arc::clone(&existing.storage));
            }
        }
        metrics::counter!("helix_collection_cache_misses_total").increment(1);

        if let Some(err) = open_failure_backoff_error(name) {
            metrics::counter!(
                "helix_collection_cold_open_total",
                "outcome" => "open_failure_backoff"
            )
            .increment(1);
            return Err(err);
        }

        let path = self.collection_path(name);
        // A dropped collection whose prefix purge is still pending is
        // logically gone; rediscovery would reopen its old manifest.
        let durable_purge_pending = match backend_any::lsm_drop_tombstone_exists_from_env(
            &path,
            self.config.storage_backend,
        ) {
            Ok(pending) => pending,
            Err(e) => {
                record_cold_open_duration(cold_open_started, "purge_pending_read_error");
                return Err(GraphError::PurgePending(format!(
                    "Collection '{}' cannot be opened yet: object-store drop tombstone \
                         status is unavailable ({}); retry later",
                    name, e
                )));
            }
        };
        if lsm_purge_pending(&path) || durable_purge_pending {
            record_cold_open_duration(cold_open_started, "purge_pending");
            return Err(GraphError::New(format!(
                "Collection '{}' not found (object-store purge pending)",
                name
            )));
        }
        if !path.exists()
            && !lsm_reader_cold_open_enabled(self.config.storage_backend)
            && !self.rediscover_lsm_collection(name, &path)
        {
            record_cold_open_duration(cold_open_started, "not_found");
            return Err(GraphError::New(format!("Collection '{}' not found", name)));
        }

        let open_slot = match self.reserve_open_slot(name) {
            Ok(open_slot) => open_slot,
            Err(err) => {
                record_cold_open_duration(cold_open_started, "admission_error");
                return Err(err);
            }
        };
        {
            let collections = self
                .collections
                .read()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            if let Some(existing) = collections.get(name) {
                if matches!(
                    existing.storage.unhealthy_lsm_writer_close_reason(),
                    Some(CloseReason::Fenced)
                ) {
                    metrics::counter!("helix_collection_cache_hits_total").increment(1);
                    return Err(record_fenced_writer_quarantine(name, CloseReason::Fenced));
                }
                touch_cached_collection(existing);
                metrics::counter!("helix_collection_cache_hits_total").increment(1);
                record_cold_open_duration(cold_open_started, "race_hit_after_admission");
                return Ok(Arc::clone(&existing.storage));
            }
        }

        metrics::counter!("helix_collection_cache_load_misses_total").increment(1);
        debug!(collection = name, "Lazy-opening collection from disk");

        // SlateDB writer and reader builds have side effects and cannot be
        // aborted safely. Once a build starts, defer request cancellation
        // through storage initialization and cache publication so the completed
        // handle is not discarded before it becomes resident.
        // The mask constructor atomically rejects requests already cancelled
        // while queued behind `open_gate`; dropping it after `cache_insert`
        // restores the token for the handler's first ordinary read.
        let cold_open_cancellation_mask =
            if lsm_cold_open_needs_cancellation_mask(self.config.storage_backend) {
                match mask_lsm_read_cancellation_for_cold_open() {
                    Ok(mask) => Some(mask),
                    Err(error) => {
                        record_cold_open_duration(cold_open_started, "cancelled_before_open");
                        return Err(graph_error_from_backend_error(error));
                    }
                }
            } else {
                None
            };
        let storage = match self.open_collection_storage(&path, name) {
            Ok(storage) => storage,
            Err(err) => {
                if err.to_string().contains("LSM read request cancelled") {
                    record_cold_open_duration(cold_open_started, "cancelled_during_open");
                    return Err(err);
                }
                let err = missing_manifest_open_error(name, err);
                open_failure_record(name, &err);
                record_cold_open_duration(cold_open_started, "open_error");
                return Err(err);
            }
        };
        open_failure_clear_after_successful_open(name);
        let storage = Arc::new(storage);
        super::replication::attach_dirty_retired_reaper_hook(&storage);
        if let Err(err) =
            super::replication::submit_post_open_vector_maintenance(&storage, &self.config, name)
        {
            open_failure_record(name, &err);
            record_cold_open_duration(cold_open_started, "post_open_error");
            return Err(err);
        }

        let mut collections = self
            .collections
            .write()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        if let Some(existing) = collections.get(name) {
            touch_cached_collection(existing);
            metrics::counter!("helix_collection_cache_hits_total").increment(1);
            record_cold_open_duration(cold_open_started, "race_hit_after_open");
            return Ok(Arc::clone(&existing.storage));
        }
        Self::cache_insert(
            &mut collections,
            name.to_string(),
            Arc::clone(&storage),
            self.open_reservations.load(Ordering::Acquire),
        );
        drop(cold_open_cancellation_mask);
        drop(open_slot);
        record_cold_open_duration(cold_open_started, "loaded");

        Ok(storage)
    }

    /// Evict a collection from the cache so the next access re-opens it from
    /// disk. Use this to recover from corrupted LMDB env handles (e.g., after
    /// a failed resize leaves the env in EINVAL state).
    ///
    /// Note: LMDB keeps a process-wide registry of opened env paths, so a
    /// subsequent `get_collection` may trip `EnvAlreadyOpened` until every
    /// outstanding `Arc` to the old storage is dropped. The open helper
    /// retries transparently via `open_storage_with_retry`.
    pub fn evict_collection(&self, name: &str) {
        if let Ok(mut collections) = self.collections.write() {
            if let Some(entry) = collections.remove(name) {
                if let Ok(mut closing) = self.closing.write() {
                    closing.insert(name.to_string(), Arc::downgrade(&entry.storage));
                }
                let weak = Arc::downgrade(&entry.storage);
                let path = entry.path.clone();
                drop(entry);
                schedule_cache_drop_after_close(weak, path, name.to_string(), "explicit");
                metrics::counter!(
                    "helix_collection_cache_evictions_total",
                    "reason" => "explicit"
                )
                .increment(1);
                tracing::warn!(collection = name, "Evicted corrupted collection from cache");
            }
        }
    }

    pub fn quarantine_collection_after_storage_error(&self, name: &str, error: &GraphError) {
        open_failure_record_quarantine(name, error);
        self.evict_collection(name);
        metrics::counter!("helix_collection_quarantine_total").increment(1);
        warn!(
            collection = %name,
            error = %error,
            "Quarantined collection after fatal storage read error"
        );
    }

    /// Get or create a collection. Idempotent.
    ///
    /// Takes the write lock once and either returns an existing collection
    /// or creates a new one, avoiding TOCTOU races on the path check.
    pub fn get_or_create_collection(
        &self,
        name: &str,
    ) -> Result<Arc<HelixGraphStorage>, GraphError> {
        // Fast path: read lock
        {
            let collections = self
                .collections
                .read()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            if let Some(entry) = collections.get(name) {
                if matches!(
                    entry.storage.unhealthy_lsm_writer_close_reason(),
                    Some(CloseReason::Fenced)
                ) {
                    return Err(record_fenced_writer_quarantine(name, CloseReason::Fenced));
                }
                touch_cached_collection(entry);
                return Ok(Arc::clone(&entry.storage));
            }
        }

        let open_gate = self.open_gate(name);
        let _open_guard = open_gate
            .lock()
            .map_err(|e| GraphError::New(format!("Open gate poisoned: {}", e)))?;

        if self.is_dropping(name) {
            return Err(Self::dropping_not_found_error(name));
        }

        // Wait for any pending close on this name to drain while holding only
        // this collection's open gate.
        self.wait_for_closing_collection(name)?;

        {
            let collections = self
                .collections
                .read()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            if let Some(entry) = collections.get(name) {
                if matches!(
                    entry.storage.unhealthy_lsm_writer_close_reason(),
                    Some(CloseReason::Fenced)
                ) {
                    return Err(record_fenced_writer_quarantine(name, CloseReason::Fenced));
                }
                touch_cached_collection(entry);
                return Ok(Arc::clone(&entry.storage));
            }
        }

        if let Some(err) = open_failure_backoff_error(name) {
            return Err(err);
        }

        let path = self.collection_path(name);
        self.resolve_pending_lsm_purge(name, &path)?;
        let open_slot = self.reserve_open_slot(name)?;
        let storage = self.open_collection_storage(&path, name)?;
        open_failure_clear_after_successful_open(name);
        let storage = Arc::new(storage);
        super::replication::attach_dirty_retired_reaper_hook(&storage);
        super::replication::submit_post_open_vector_maintenance(&storage, &self.config, name)?;

        let mut collections = self
            .collections
            .write()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        if let Some(entry) = collections.get(name) {
            touch_cached_collection(entry);
            return Ok(Arc::clone(&entry.storage));
        }
        Self::cache_insert(
            &mut collections,
            name.to_string(),
            Arc::clone(&storage),
            self.open_reservations.load(Ordering::Acquire),
        );
        drop(open_slot);

        Ok(storage)
    }

    /// Drop a collection: remove from cache, delete the on-disk directory,
    /// and clean up any aliases that pointed to this collection.
    ///
    /// The directory is removed synchronously even if other threads still
    /// hold an `Arc<HelixGraphStorage>` for this name. On Linux/macOS this
    /// is safe — `unlink` succeeds on files with open fds, and the old
    /// env keeps operating on the unlinked inodes until its last holder
    /// drops. A subsequent `create_collection` on the same name may still
    /// trip `EnvAlreadyOpened` because LMDB/heed keys its process-wide
    /// open-env registry by canonical path; `open_storage_with_retry`
    /// handles that transparently.
    ///
    /// If the synchronous removal fails (unusual — e.g. an NFS mount that
    /// refuses unlink of open files), fall back to a best-effort background
    /// worker that retries once the Arc becomes unique.
    pub fn drop_collection(&self, name: &str) -> Result<(), GraphError> {
        let open_gate = self.open_gate(name);
        let _open_guard = open_gate
            .lock()
            .map_err(|e| GraphError::New(format!("Open gate poisoned: {}", e)))?;

        let path = self.collection_path(name);
        let is_object_store_writer = self.config.storage_backend.is_lsm()
            && !self.config.storage_backend.is_lsm_in_memory()
            && !self.config.storage_backend.is_reader();
        let should_persist_lsm_tombstone = is_object_store_writer
            && (path.exists()
                || self
                    .collections
                    .read()
                    .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?
                    .contains_key(name)
                || backend_any::lsm_collection_prefix_exists_from_env(
                    &path,
                    self.config.storage_backend,
                )
                .map_err(|e| {
                    GraphError::PurgePending(format!(
                        "Collection '{}' cannot be dropped: object-store prefix status \
                         is unavailable before teardown ({})",
                        name, e
                    ))
                })?);
        if should_persist_lsm_tombstone {
            backend_any::persist_lsm_drop_tombstone_from_env(&path, self.config.storage_backend)
                .map_err(|e| {
                    GraphError::PurgePending(format!(
                        "Collection '{}' cannot be dropped: failed to persist object-store \
                         drop tombstone before teardown ({})",
                        name, e
                    ))
                })?;
        }

        self.mark_dropping(name);

        // Take the Arc out of the cache while holding the write lock.
        let removed = {
            let mut collections = self
                .collections
                .write()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            collections.remove(name)
        };

        // Purge per-collection snapshot state so dropped/recreated tenants
        // don't leave behind entries that grow the map forever.
        if let Ok(mut gates) = self.snapshot_gates.write() {
            gates.remove(name);
        }
        // Publish the Weak into `closing` BEFORE removing the on-disk
        // directory. Otherwise a concurrent get_collection that passed the
        // cache-miss check sees no `closing` entry AND no path, and returns
        // NotFound instead of waiting on the env to fully close.
        if let Some(entry) = &removed {
            if Arc::strong_count(&entry.storage) > 1 {
                if let Ok(mut closing) = self.closing.write() {
                    closing.insert(name.to_string(), Arc::downgrade(&entry.storage));
                }
            }
        }

        // LSM-only: close SlateDB (quiesce compactor/GC) then purge the
        // object-store prefix and SSD cache subtree.  Must run BEFORE
        // `remove_dir_all` so the safe ordering is:
        //   evict cache → close SlateDB → S3 purge + cache clear → local rm
        //
        // A purge failure MUST NOT abort the drop: `destroy()` has already
        // closed SlateDB by then, so bailing out here would leave a zombie
        // (gone from the cache, env stuck in `closing`, prefix orphaned in
        // S3) that resurrects through rediscovery and wedges every write on
        // `EnvAlreadyOpen`. Finish the local teardown and retry the prefix
        // purge in the background instead.
        //
        // LMDB / LsmReader arms are no-ops; the existing `remove_dir_all`
        // below handles the local LMDB env directory on all backends.
        //
        // A failed purge registers the prefix as pending (see
        // `LSM_PURGE_PENDING`) so the name cannot be reopened/recreated on the
        // old manifest, and so the retry worker aborts once a later
        // create/drop has purged the prefix itself.
        let purge_result = if let Some(ref entry) = removed {
            entry.storage.backend.destroy_lsm().map(|()| true)
        } else if self.config.storage_backend.is_lsm()
            && (!is_object_store_writer || should_persist_lsm_tombstone)
        {
            backend_any::purge_lsm_prefix_from_env(&path, self.config.storage_backend)
                .map(|()| true)
        } else {
            Ok(false)
        };
        match purge_result {
            Ok(true) => {
                if let Err(e) = backend_any::clear_lsm_drop_tombstone_from_env(
                    &path,
                    self.config.storage_backend,
                ) {
                    warn!(
                        collection = %name,
                        error = %e,
                        "LSM object-store drop tombstone cleanup failed after purge"
                    );
                    let token = lsm_purge_pending_register(&path);
                    spawn_lsm_purge_retry(
                        path.clone(),
                        name.to_string(),
                        self.config.storage_backend,
                        token,
                        Arc::clone(&open_gate),
                    );
                } else {
                    lsm_purge_pending_clear(&path);
                }
            }
            Ok(false) => {}
            Err(e) => {
                warn!(
                    collection = %name,
                    error = %e,
                    loaded = removed.is_some(),
                    "LSM object-store purge failed during drop; retrying in background"
                );
                let token = lsm_purge_pending_register(&path);
                spawn_lsm_purge_retry(
                    path.clone(),
                    name.to_string(),
                    self.config.storage_backend,
                    token,
                    Arc::clone(&open_gate),
                );
            }
        }

        // Remove the on-disk directory. Do this before attempting a clean
        // Arc-unique drop so a concurrent create_collection sees the path
        // as absent and its retry loop is only ever gated on the heed
        // OPENED_ENV registry.
        let mut sync_remove_err: Option<std::io::Error> = None;
        if path.exists() {
            if let Err(e) = fs::remove_dir_all(&path) {
                sync_remove_err = Some(e);
            }
        }

        if let Some(entry) = removed {
            let arc = entry.storage;
            if sync_remove_err.is_none() && Arc::strong_count(&arc) == 1 {
                // Unique: drop closes the env, releasing the path from
                // heed's OPENED_ENV registry immediately.
                drop(arc);
                if let Ok(mut closing) = self.closing.write() {
                    closing.remove(name);
                }
            } else if let Some(err) = sync_remove_err.take() {
                // Sync removal failed (rare). Defer to a worker that waits
                // for the Arc to become unique (env close) and retries the
                // directory removal.
                metrics::counter!(
                    "helix_collection_drop_deferred_total",
                    "collection" => name.to_string()
                )
                .increment(1);
                let path_for_worker = path.clone();
                let name_for_worker = name.to_string();
                if let Err(spawn_err) = thread::Builder::new()
                    .name(format!("helix-drop-{}", name))
                    .spawn(move || {
                        deferred_drop_cleanup(arc, path_for_worker, name_for_worker);
                    })
                {
                    warn!(
                        collection = %name,
                        remove_error = %err,
                        spawn_error = %spawn_err,
                        "Failed to spawn deferred drop worker after remove_dir_all failure"
                    );
                    return Err(GraphError::from(err));
                }
            }
            // else: sync remove succeeded but Arc is still shared. The env
            // closes when the last holder drops; the dir is already gone.
        } else if let Some(err) = sync_remove_err {
            return Err(GraphError::from(err));
        }

        // Remove any aliases that pointed to this collection.
        let stale_aliases = self.aliases_for_collection(name)?;
        if !stale_aliases.is_empty() {
            let mut aliases = self
                .aliases
                .write()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            for alias in &stale_aliases {
                aliases.remove(alias);
            }
            drop(aliases);
            self.persist_aliases()?;
        }
        self.cleanup_open_gate_if_idle(name, &open_gate);
        open_failure_clear(name);
        self.clear_dropping(name);
        Ok(())
    }

    fn cleanup_open_gate_if_idle(&self, name: &str, gate: &Arc<Mutex<()>>) {
        if Arc::strong_count(gate) != 2 {
            return;
        }
        if let Ok(mut gates) = self.open_gates.write() {
            if gates
                .get(name)
                .is_some_and(|current| Arc::ptr_eq(current, gate))
            {
                gates.remove(name);
            }
        }
    }

    /// List all collections (by scanning the collections directory).
    pub fn list_collections(&self) -> Result<Vec<String>, GraphError> {
        let collections_dir = self.data_dir.join("collections");
        let mut names: HashSet<String> = HashSet::new();
        if collections_dir.exists() {
            for entry in fs::read_dir(&collections_dir)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    if let Some(name) = entry.file_name().to_str() {
                        names.insert(name.to_string());
                    }
                }
            }
        }
        // On LSM, include collections that exist in the object store but are not
        // yet in the local registry (e.g. created before a PVC loss, or not yet
        // touched since a fresh start). Best-effort: a listing failure falls back
        // to the local-only view rather than erroring the request.
        if self.config.storage_backend.is_lsm() {
            match backend_any::list_collection_prefixes_from_env(self.config.storage_backend) {
                Ok(s3_names) => {
                    for name in s3_names {
                        if !name.is_empty() {
                            names.insert(name);
                        }
                    }
                }
                Err(e) => warn!(
                    error = %e,
                    "list_collections: object-store listing failed; returning local view only"
                ),
            }
            let mut visible = HashSet::with_capacity(names.len());
            for name in names {
                let path = self.collection_path(&name);
                match backend_any::lsm_drop_tombstone_exists_from_env(
                    &path,
                    self.config.storage_backend,
                ) {
                    Ok(false) => {
                        visible.insert(name);
                    }
                    Ok(true) => {}
                    Err(e) => {
                        return Err(GraphError::PurgePending(format!(
                            "Collection '{}' cannot be listed: object-store drop tombstone \
                             status is unavailable ({}); retry later",
                            name, e
                        )));
                    }
                }
            }
            names = visible;
        }
        let mut names: Vec<String> = names.into_iter().collect();
        names.sort();
        Ok(names)
    }

    /// Return cold (currently unloaded) collections whose on-disk dense
    /// sidecar file count exceeds `min_sidecars`. Pure filesystem scan —
    /// never opens an LMDB env. Intended for the optimizer backlog sweeper
    /// to discover cold tenants with merge work without keeping them
    /// loaded: the caller submits a merge which transiently admits the env
    /// through the LRU; idle eviction reclaims it after the merge.
    pub fn cold_collections_with_sidecar_backlog(
        &self,
        min_sidecars: usize,
    ) -> Result<Vec<(String, usize)>, GraphError> {
        let loaded: HashSet<String> = {
            let collections = self
                .collections
                .read()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            collections.keys().cloned().collect()
        };
        let collections_dir = self.data_dir.join("collections");
        if !collections_dir.exists() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for entry in fs::read_dir(&collections_dir)? {
            let Ok(entry) = entry else { continue };
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if !file_type.is_dir() {
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(|s| s.to_string()) else {
                continue;
            };
            if loaded.contains(&name) {
                continue;
            }
            let sidecars = count_dense_sidecars_in_dir(&entry.path());
            if sidecars > min_sidecars {
                out.push((name, sidecars));
            }
        }
        Ok(out)
    }

    /// Get stats for a collection.
    pub fn collection_stats(&self, name: &str) -> Result<CollectionStats, GraphError> {
        let resolved = self.resolve_alias(name);
        if let Some(storage) = self.get_loaded_collection(&resolved)? {
            let metadata = storage.metadata_snapshot()?;
            let disk_bytes = match storage.lmdb_env() {
                Ok(env) => HelixGraphStorage::data_mdb_bytes(env.path())?,
                Err(_) => 0,
            };

            return Ok(CollectionStats {
                name: name.to_string(),
                schema_version: metadata.schema_version,
                node_count: metadata.stats.node_count,
                edge_count: metadata.stats.edge_count,
                vector_count: metadata.stats.vector_count,
                disk_bytes,
            });
        }

        let path = self.collection_path(&resolved);
        let sidecar = HelixGraphStorage::read_metadata_sidecar_from_path_or_lsm(
            &path,
            self.config.storage_backend,
        )?;
        let (metadata, disk_bytes) = match sidecar {
            Some(sidecar) => (sidecar.metadata, sidecar.data_mdb_bytes),
            None => {
                if !path.exists() {
                    return Err(GraphError::New(format!(
                        "Collection '{}' does not exist",
                        name
                    )));
                }
                return Err(GraphError::New(format!(
                    "Collection '{}' metadata sidecar is not available",
                    name
                )));
            }
        };

        Ok(CollectionStats {
            name: name.to_string(),
            schema_version: metadata.schema_version,
            node_count: metadata.stats.node_count,
            edge_count: metadata.stats.edge_count,
            vector_count: metadata.stats.vector_count,
            disk_bytes,
        })
    }

    /// Recount a collection's LSM counter keys from scan truth. Unlike
    /// `collection_stats`, this always opens the collection (cold or not) since
    /// a repair must be able to run on a collection nobody has touched since
    /// the corruption occurred.
    pub fn recount_collection(&self, name: &str) -> Result<RecountResult, GraphError> {
        let resolved = self.resolve_alias(name);
        let storage = self.get_collection(&resolved)?;
        let counters = storage.recount_lsm_counters()?;
        Ok(RecountResult {
            name: name.to_string(),
            counters,
        })
    }

    /// Repair the ghost payload-index class for a collection: always opens
    /// the collection (cold or not), same rationale as `recount_collection` —
    /// the corrupted index may belong to a collection nobody has touched
    /// since the corruption occurred.
    pub fn gc_payload_index_for_collection(
        &self,
        name: &str,
        field: Option<&str>,
    ) -> Result<PayloadIndexGcResult, GraphError> {
        let resolved = self.resolve_alias(name);
        let storage = self.get_collection(&resolved)?;
        let gate = self.point_mutation_gate(&resolved);
        let fields = storage.gc_payload_index(field, Some(&gate))?;
        Ok(PayloadIndexGcResult {
            name: name.to_string(),
            fields,
        })
    }

    /// Number of currently loaded (open) collections.
    pub fn loaded_count(&self) -> usize {
        self.collections.read().map(|c| c.len()).unwrap_or(0)
    }

    pub fn max_open_collections_limit(&self) -> usize {
        max_open_collections()
    }

    pub fn memory_snapshot(&self) -> CollectionMemorySnapshot {
        let snapshot = collection_memory_snapshot();
        record_memory_snapshot(snapshot);
        snapshot
    }

    /// Sum of LMDB `map_size` across currently-loaded collections, in bytes.
    /// Approximates virtual-address-space consumption of the database layer.
    /// Returns `None` if the cache lock is poisoned (should never happen in
    /// practice; metrics scrape endpoint handles the None quietly).
    pub fn total_lmdb_va_bytes(&self) -> Option<u64> {
        let guard = self.collections.read().ok()?;
        let total: u64 = guard
            .values()
            .map(|c| c.storage.lmdb_map_size_bytes().unwrap_or(0))
            .sum();
        Some(total)
    }

    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn backend_kind(&self) -> BackendKind {
        self.config.storage_backend.kind()
    }

    pub fn snapshot_interval_secs(&self) -> u64 {
        self.config.snapshot_interval_secs()
    }

    pub fn snapshot_keep_last(&self) -> usize {
        self.config.snapshot_keep_last()
    }

    /// Create a snapshot of `name`. Concurrent callers targeting the same
    /// collection are serialised by a per-collection mutex so only one LMDB
    /// compact-copy runs at a time (prevents long reader txns from piling up
    /// and bloating the free-list). If the LMDB write txn id has not
    /// advanced since the last successful snapshot — from either the current
    /// process or a previous one hydrated from the on-disk manifest — and
    /// the snapshot file is still on disk, the cached `SnapshotInfo` is
    /// returned without doing any I/O.
    pub fn snapshot_collection(&self, name: &str) -> Result<SnapshotInfo, GraphError> {
        let storage = self.get_collection(name)?;
        if storage.backend.kind() == BackendKind::Lsm {
            return Err(GraphError::StorageError(
                "LMDB snapshots are unavailable on the LSM backend; use object-store backup/checkpointing instead"
                    .to_string(),
            ));
        }
        let gate = self.snapshot_gate(name);
        let mut state = gate
            .lock()
            .map_err(|e| GraphError::New(format!("snapshot gate poisoned: {}", e)))?;

        // First call for this collection in the current process: try to
        // recover prior snapshot state from the most recent on-disk manifest
        // so we can skip-if-no-advance across restarts. Failure is non-fatal;
        // we just fall through to a fresh copy.
        if state.last.is_none() {
            self.hydrate_snapshot_state(name, &mut state);
        }

        // Skip-if-no-advance: compare the live LMDB `last_txn_id` against
        // the value seen right after the previous snapshot. Any commit since
        // then (Raft apply *or* direct write) will have advanced this
        // counter. We only trust the cache when the snapshot file is still
        // on disk — retention pruning or operator deletion can have removed
        // it, in which case we must fall through and rebuild.
        let current_txn_id = storage.lmdb_last_txn_id()?;
        if let Some(last) = &state.last {
            if current_txn_id == state.last_post_snapshot_txn_id
                && state.last_post_snapshot_txn_id != 0
                && Path::new(&last.path).exists()
            {
                metrics::counter!(
                    "helix_snapshot_skipped_total",
                    "collection" => name.to_string(),
                    "reason" => "no_advance"
                )
                .increment(1);
                debug!(
                    collection = %name,
                    lmdb_txn_id = current_txn_id,
                    snapshot_lsn = last.lsn,
                    path = %last.path,
                    "Skipping snapshot; no writes since last snapshot"
                );
                return Ok(last.clone());
            }
        }

        let created_at_millis = next_snapshot_millis();
        let snapshot_path = self.snapshot_path(name, created_at_millis);
        let snapshot_dir = snapshot_path
            .parent()
            .ok_or_else(|| GraphError::New("snapshot path missing parent".into()))?;

        fs::create_dir_all(snapshot_dir)?;
        let started = std::time::Instant::now();
        let (lsn, snapshot_file) = storage.snapshot_to(name, &snapshot_path)?;
        let elapsed = started.elapsed();
        let disk_bytes = snapshot_file.metadata()?.len();
        // Sample the post-copy txn id *before* writing the manifest so the
        // persisted value matches what we cache in memory.
        let post_txn_id = storage.lmdb_last_txn_id()?;
        let snapshot = SnapshotInfo {
            name: snapshot_path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default()
                .to_string(),
            lsn,
            path: snapshot_path.display().to_string(),
            disk_bytes,
            created_at_millis,
        };
        fs::write(
            Self::snapshot_manifest_path(&snapshot_path),
            sonic_rs::to_vec(&SnapshotManifest {
                info: snapshot.clone(),
                txn_id_at_snapshot: Some(post_txn_id as u64),
            })?,
        )?;
        prune_snapshots(snapshot_dir, self.snapshot_keep_last())?;

        metrics::counter!(
            "helix_snapshot_created_total",
            "collection" => name.to_string()
        )
        .increment(1);
        metrics::histogram!(
            "helix_snapshot_duration_seconds",
            "collection" => name.to_string()
        )
        .record(elapsed.as_secs_f64());

        state.last = Some(snapshot.clone());
        state.last_post_snapshot_txn_id = post_txn_id;
        Ok(snapshot)
    }

    /// Asynchronous snapshot: returns immediately with a placeholder
    /// `SnapshotInfo`, then spawns a background thread to do the actual
    /// LMDB copy.  The background thread updates the in-memory cache with
    /// the real `disk_bytes` once the copy completes.
    ///
    /// This avoids blocking the HTTP handler's tokio `spawn_blocking` thread
    /// for the (potentially long) `copy_to_path` duration.
    pub fn snapshot_collection_async(&self, name: &str) -> Result<SnapshotInfo, GraphError> {
        use std::thread;

        let storage = self.get_collection(name)?;
        if storage.backend.kind() == BackendKind::Lsm {
            return Err(GraphError::StorageError(
                "LMDB snapshots are unavailable on the LSM backend; use object-store backup/checkpointing instead"
                    .to_string(),
            ));
        }
        let gate = self.snapshot_gate(name);
        let mut state = gate
            .lock()
            .map_err(|e| GraphError::New(format!("snapshot gate poisoned: {}", e)))?;

        // Hydrate state on first call in this process
        if state.last.is_none() {
            self.hydrate_snapshot_state(name, &mut state);
        }

        // Skip-if-no-advance: cache hit — return immediately
        let current_txn_id = storage.lmdb_last_txn_id()?;
        if let Some(last) = &state.last {
            if current_txn_id == state.last_post_snapshot_txn_id
                && state.last_post_snapshot_txn_id != 0
                && Path::new(&last.path).exists()
            {
                return Ok(last.clone());
            }
        }

        // Fast path: init snapshot (WAL flush + sync + WAL op), create dir,
        // write manifest, return placeholder immediately.
        let created_at_millis = next_snapshot_millis();
        let snapshot_path = self.snapshot_path(name, created_at_millis);
        let snapshot_dir = snapshot_path
            .parent()
            .ok_or_else(|| GraphError::New("snapshot path missing parent".into()))?;
        fs::create_dir_all(snapshot_dir)?;

        let lsn = storage.snapshot_init(name)?;
        let post_txn_id = storage.lmdb_last_txn_id()?;

        // Placeholder response — disk_bytes = 0, real value filled by background thread
        let placeholder = SnapshotInfo {
            name: snapshot_path
                .file_name()
                .and_then(|v| v.to_str())
                .unwrap_or_default()
                .to_string(),
            lsn,
            path: snapshot_path.display().to_string(),
            disk_bytes: 0,
            created_at_millis,
        };

        // Write manifest with placeholder
        fs::write(
            Self::snapshot_manifest_path(&snapshot_path),
            sonic_rs::to_vec(&SnapshotManifest {
                info: placeholder.clone(),
                txn_id_at_snapshot: Some(post_txn_id as u64),
            })?,
        )?;

        // Update cache with placeholder so subsequent calls see the snapshot
        state.last = Some(placeholder.clone());
        state.last_post_snapshot_txn_id = post_txn_id;

        // Background copy phase — dedicated thread holds resize_guard during copy
        let storage = Arc::clone(&storage);
        let gate = Arc::clone(&gate);
        let snapshot_path = snapshot_path.clone();
        let snapshot_keep_last = self.snapshot_keep_last();
        let collection_name = name.to_string();

        thread::spawn(move || {
            let result = (|| -> Result<(), GraphError> {
                let _resize_guard = storage.read_resize_guard_if_needed()?;
                let file = storage
                    .lmdb_env()?
                    .copy_to_path(&snapshot_path, CompactionOption::Enabled)?;
                file.sync_data()?;
                let disk_bytes = file.metadata()?.len();

                // Prune old snapshots
                prune_snapshots(snapshot_path.parent().unwrap(), snapshot_keep_last)?;

                metrics::counter!(
                    "helix_snapshot_created_total",
                    "collection" => collection_name.clone()
                )
                .increment(1);

                // Update cache with real disk_bytes
                let mut state = gate
                    .lock()
                    .map_err(|e| GraphError::New(format!("snapshot gate poisoned: {}", e)))?;
                if let Some(info) = &mut state.last {
                    if info.path == snapshot_path.display().to_string() {
                        info.disk_bytes = disk_bytes;
                    }
                }

                // Rewrite manifest on disk so list_snapshots (which reads from disk)
                // returns the real disk_bytes value. Preserve txn_id_at_snapshot
                // from the existing manifest so skip-if-no-advance survives restarts.
                let manifest_path = Self::snapshot_manifest_path(&snapshot_path);
                if manifest_path.exists() {
                    if let Ok(existing) =
                        sonic_rs::from_slice::<SnapshotManifest>(&fs::read(&manifest_path)?)
                    {
                        let mut updated = existing.info.clone();
                        updated.disk_bytes = disk_bytes;
                        if let Ok(manifest_bytes) = sonic_rs::to_vec(&SnapshotManifest {
                            info: updated,
                            txn_id_at_snapshot: existing.txn_id_at_snapshot,
                        }) {
                            let _ = fs::write(&manifest_path, manifest_bytes);
                        }
                    }
                }
                Ok(())
            })();
            if let Err(e) = result {
                warn!(
                    error = %e,
                    path = %snapshot_path.display(),
                    "Background snapshot copy failed"
                );
            }
        });

        Ok(placeholder)
    }

    /// Populate `state` from the most recent on-disk snapshot manifest for
    /// `name`, if any. Called lazily on the first `snapshot_collection` call
    /// per collection per process, so restarts don't force a redundant copy
    /// when the data hasn't actually changed.
    ///
    /// Silently does nothing when:
    ///   * the snapshots directory doesn't exist yet (first-ever snapshot);
    ///   * no manifest has a `txn_id_at_snapshot` (pre-upgrade manifests);
    ///   * any IO or parse error — we prefer a redundant copy to a wrong skip.
    fn hydrate_snapshot_state(&self, name: &str, state: &mut SnapshotState) {
        let snapshots_dir = self.collection_path(name).join("snapshots");
        let read_iter = match fs::read_dir(&snapshots_dir) {
            Ok(it) => it,
            Err(_) => return,
        };
        let mut newest: Option<SnapshotManifest> = None;
        for entry in read_iter.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let bytes = match fs::read(&path) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let manifest: SnapshotManifest = match sonic_rs::from_slice(&bytes) {
                Ok(m) => m,
                Err(_) => continue,
            };
            if !Path::new(&manifest.info.path).exists() {
                continue;
            }
            match &newest {
                Some(cur) if cur.info.created_at_millis >= manifest.info.created_at_millis => {}
                _ => newest = Some(manifest),
            }
        }
        if let Some(manifest) = newest {
            if let Some(txn_id) = manifest.txn_id_at_snapshot {
                state.last_post_snapshot_txn_id = txn_id as usize;
            }
            state.last = Some(manifest.info);
        }
    }

    pub fn snapshot_loaded_collections(&self) -> Result<Vec<SnapshotInfo>, GraphError> {
        let loaded_names: Vec<String> = self
            .collections
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?
            .keys()
            .cloned()
            .collect();

        let mut snapshots = Vec::with_capacity(loaded_names.len());
        for name in loaded_names {
            if let Some(storage) = self.get_loaded_collection(&name)? {
                if skip_periodic_snapshot_for_backend(storage.backend.kind()) {
                    metrics::counter!(
                        "helix_snapshot_skipped_total",
                        "collection" => name.to_string(),
                        "reason" => "lsm_backend"
                    )
                    .increment(1);
                    debug!(
                        collection = %name,
                        "Skipping periodic LMDB snapshot for LSM-backed collection"
                    );
                    continue;
                }
            }
            snapshots.push(self.snapshot_collection(&name)?);
        }

        Ok(snapshots)
    }

    /// Flush the WAL for all currently loaded collections. Call on graceful
    /// shutdown to ensure all pending writes are durable on disk.
    pub fn flush_all_wals(&self) -> Result<usize, GraphError> {
        let loaded: Vec<(String, Arc<HelixGraphStorage>)> = self
            .collections
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?
            .iter()
            .map(|(k, v)| (k.clone(), Arc::clone(&v.storage)))
            .collect();

        let mut flushed = 0usize;
        for (name, storage) in &loaded {
            match storage.wal.flush() {
                Ok(_lsn) => flushed += 1,
                Err(e) => {
                    warn!(collection = %name, error = %e, "WAL flush failed");
                }
            }
        }
        Ok(flushed)
    }

    /// Iterate every loaded collection and call `env.force_sync()` on its
    /// LMDB env. Used by the background LMDB sync thread when
    /// `HELIX_LMDB_NOSYNC=1` is set: per-commit fsync is disabled, but
    /// the kernel page cache must be force-synced before the
    /// corresponding /data/write-queue/{uuid}.json record is allowed to
    /// be deleted, otherwise a kernel panic between commit and
    /// auto-flush would lose data permanently.
    ///
    /// On a clean (no dirty pages) env, `force_sync` is essentially a
    /// no-op at the LMDB level — heed3 calls `mdb_env_sync` which
    /// short-circuits when the env's `me_dirty_root` is unset. So
    /// iterating all loaded envs every 100 ms is cheap.
    ///
    /// Returns the number of envs successfully synced.
    pub fn force_sync_all(&self) -> usize {
        let loaded: Vec<(String, Arc<HelixGraphStorage>)> = match self.collections.read() {
            Ok(g) => g
                .iter()
                .map(|(k, v)| (k.clone(), Arc::clone(&v.storage)))
                .collect(),
            Err(_) => return 0,
        };

        let mut synced = 0usize;
        for (name, storage) in &loaded {
            // heed3's force_sync calls mdb_env_sync(env, force=1) which
            // is a no-op on a clean env. Errors are non-fatal here:
            // the durable JSON spool will replay on next startup if the
            // sync didn't actually persist before a crash.
            if let Err(e) = storage.force_sync_lmdb() {
                warn!(collection = %name, error = ?e, "LMDB force_sync failed");
            } else {
                synced += 1;
            }
        }
        synced
    }

    pub fn list_snapshots(&self, name: &str) -> Result<Vec<SnapshotInfo>, GraphError> {
        let collection_path = self.collection_path(name);
        if !collection_path.exists() {
            return Err(GraphError::New(format!("Collection '{}' not found", name)));
        }

        let snapshot_dir = collection_path.join("snapshots");
        if !snapshot_dir.exists() {
            return Ok(Vec::new());
        }

        let mut snapshots: Vec<SnapshotInfo> = fs::read_dir(&snapshot_dir)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("mdb"))
            .map(|path| self.read_snapshot_info(name, &path))
            .collect::<Result<Vec<_>, _>>()?;

        snapshots.sort_by(|lhs, rhs| rhs.created_at_millis.cmp(&lhs.created_at_millis));
        Ok(snapshots)
    }

    pub fn restore_collection_from_snapshot(
        &self,
        name: &str,
        snapshot: &str,
    ) -> Result<SnapshotInfo, GraphError> {
        let snapshot_path = self.resolve_snapshot_path(name, snapshot)?;
        self.restore_collection_from_snapshot_path(name, &snapshot_path)
    }

    pub(crate) fn create_raft_cluster_snapshot(&self) -> Result<Vec<u8>, GraphError> {
        let staging_dir = self
            .data_dir
            .join("raft")
            .join("snapshot-build")
            .join(format!("{}", next_snapshot_millis()));
        fs::create_dir_all(&staging_dir)?;

        let result = (|| {
            let mut collections = Vec::new();
            for (index, name) in self.list_collections()?.into_iter().enumerate() {
                let storage = self.get_collection(&name)?;
                let snapshot_path = staging_dir.join(format!("{:04}.mdb", index));
                let (_, file) = storage.snapshot_to(&name, &snapshot_path)?;
                drop(file);
                collections.push(RaftCollectionSnapshot {
                    name,
                    bytes: fs::read(&snapshot_path)?,
                });
            }

            bincode::serialize(&RaftClusterSnapshot { collections }).map_err(GraphError::from)
        })();

        let _ = fs::remove_dir_all(&staging_dir);
        result
    }

    pub(crate) fn restore_raft_cluster_snapshot(
        &self,
        snapshot_bytes: &[u8],
    ) -> Result<(), GraphError> {
        let snapshot: RaftClusterSnapshot = bincode::deserialize(snapshot_bytes)?;
        let desired_collections: HashSet<String> = snapshot
            .collections
            .iter()
            .map(|collection| collection.name.clone())
            .collect();

        for existing in self.list_collections()? {
            if !desired_collections.contains(&existing) {
                self.drop_collection(&existing)?;
            }
        }

        for collection in snapshot.collections {
            self.restore_collection_from_bytes(&collection.name, &collection.bytes)?;
        }

        Ok(())
    }

    pub(crate) fn restore_collection_from_bytes(
        &self,
        name: &str,
        snapshot_bytes: &[u8],
    ) -> Result<(), GraphError> {
        let collection_path = self.collection_path(name);
        fs::create_dir_all(&collection_path)?;
        let snapshot_path = collection_path.join("data.restore.source.mdb");
        fs::write(&snapshot_path, snapshot_bytes)?;
        let result = self.restore_collection_from_snapshot_path(name, &snapshot_path);
        let _ = fs::remove_file(&snapshot_path);
        result.map(|_| ())
    }

    fn restore_collection_from_snapshot_path(
        &self,
        name: &str,
        snapshot_path: &Path,
    ) -> Result<SnapshotInfo, GraphError> {
        let snapshot_info = self.read_snapshot_info(name, snapshot_path)?;
        let collection_path = self.collection_path(name);

        let removed = {
            let mut collections = self
                .collections
                .write()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            collections.remove(name)
        };

        if let Some(entry) = removed {
            if Arc::strong_count(&entry.storage) > 1 {
                self.collections
                    .write()
                    .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?
                    .insert(name.to_string(), entry);
                return Err(GraphError::New(format!(
                    "Collection '{}' is busy and cannot be restored while in use",
                    name
                )));
            }
        }

        fs::create_dir_all(&collection_path)?;
        let data_path = collection_path.join("data.mdb");
        let temp_restore = collection_path.join("data.restore.tmp");
        let lock_path = collection_path.join("lock.mdb");
        let wal_path = collection_path.join("wal");

        fs::copy(&snapshot_path, &temp_restore)?;
        if data_path.exists() {
            fs::remove_file(&data_path)?;
        }
        fs::rename(&temp_restore, &data_path)?;
        if lock_path.exists() {
            fs::remove_file(lock_path)?;
        }
        if wal_path.exists() {
            fs::remove_dir_all(&wal_path)?;
        }

        self.get_collection(name)?;
        Ok(snapshot_info)
    }

    /// Ratio-triggered copy-compaction.
    ///
    /// LMDB never returns freed pages to the OS: between compactions `data.mdb`
    /// is a high-water mark. The only shrink path is a compact-copy
    /// (`copy_to_path(CompactionOption::Enabled)`) + restore-in-place. This is
    /// the *automatic* trigger for that path: when a loaded collection's
    /// `data.mdb` is sufficiently larger than its live (non-free) bytes, write-
    /// idle, and outside its cooldown, run the compaction so dead space does not
    /// sit for weeks waiting for a manual snapshot.
    ///
    /// Returns `Ok(true)` when a compaction actually ran (success or not),
    /// `Ok(false)` when the trigger conditions were not met (no-op). All gating
    /// is cheap and txn-free; the heavy copy only happens after every guard
    /// passes and the global single-compaction permit is held.
    ///
    /// Safety / atomicity: the compacted env is written to a temp file first;
    /// only `restore_collection_from_snapshot_path` swaps it into place, and
    /// that swap copies temp→`data.restore.tmp`→rename and only removes the
    /// original `data.mdb` *after* the temp restore copy succeeds. A failed
    /// compact therefore leaves the collection serving from the original file.
    /// The restore path additionally rejects when the storage `Arc` is still in
    /// use (`strong_count > 1`), so this method must drop its own measurement
    /// `Arc` before invoking restore — it does.
    pub fn auto_compact_collection_if_needed(&self, name: &str) -> Result<bool, GraphError> {
        if !auto_compact_enabled() {
            return Ok(false);
        }
        let resolved = self.resolve_alias(name);
        let name = resolved.as_str();

        // Measure with a short-lived Arc, then DROP it before any restore: the
        // restore path rejects collections whose storage Arc is still aliased
        // (strong_count > 1). Scope the measurement so the clone is gone before
        // we attempt the swap.
        let (file_bytes, live_bytes, write_idle) = {
            let storage = self.get_collection(name)?;
            if storage.backend.kind() == BackendKind::Lsm {
                return Ok(false);
            }

            // File size = LMDB high-water mark. Prefer the OS file length over
            // the env's reported map size; that is what actually occupies disk.
            let env = storage.lmdb_env()?;
            let file_bytes = match env.real_disk_size() {
                Ok(bytes) => bytes,
                Err(_) => HelixGraphStorage::data_mdb_bytes(&self.collection_path(name))?,
            };

            // Cheap gate first: skip the (read-txn-opening) live-bytes scan when
            // the file is below the absolute minimum — tiny collections are
            // never worth compacting regardless of ratio.
            if file_bytes < auto_compact_min_bytes() {
                return Ok(false);
            }

            // Upper-bound gate (H2): a compaction holds the per-collection
            // open_gate for the FULL copy duration, stalling new writers to this
            // collection for that whole window. Operators can cap the eligible
            // file size so very large envs are excluded from automatic
            // compaction (deferred to an off-peak/manual window). 0 = unlimited.
            let max_file = auto_compact_max_file_bytes();
            if max_file > 0 && file_bytes > max_file {
                metrics::counter!(
                    "helix_auto_compact_total",
                    "result" => "too_large"
                )
                .increment(1);
                return Ok(false);
            }

            // Live bytes = sum over the env's DBs of
            // (leaf + branch + overflow pages) × page_size (see
            // `env_live_bytes`). NOT `last_page_number × page_size` (that is the
            // high-water mark / file size again). Use the storage read wrapper
            // so this probe is counted by the resize fence.
            let live_bytes = storage.with_read_txn(|rtxn| env_live_bytes(env, rtxn))?;

            // Write-idle: reuse the same recency signal the optimizer quiesce
            // loop uses (`last_upsert_at`, epoch millis). 0 means "no write
            // observed this process" → treat as idle.
            let now_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            let last_upsert = storage.last_upsert_at.load(Ordering::Acquire);
            let write_idle = last_upsert == 0
                || now_ms.saturating_sub(last_upsert) >= auto_compact_write_idle_ms();

            (file_bytes, live_bytes, write_idle)
            // `storage` Arc dropped here.
        };

        metrics::gauge!(
            "helix_collection_data_file_bytes",
            "collection" => name.to_string()
        )
        .set(file_bytes as f64);
        metrics::gauge!(
            "helix_collection_live_bytes",
            "collection" => name.to_string()
        )
        .set(live_bytes as f64);

        // Ratio gate. live_bytes can legitimately be 0 for a freshly created /
        // fully emptied env; guard the divide and treat 0-live as "compactable
        // if the file is large" (a large file with no live data is pure waste).
        let ratio_exceeded = if live_bytes == 0 {
            file_bytes >= auto_compact_min_bytes()
        } else {
            (file_bytes as f64 / live_bytes as f64) > auto_compact_ratio()
        };
        if !ratio_exceeded || !write_idle {
            return Ok(false);
        }

        // Cooldown: skip if we compacted this collection too recently. Checked
        // before taking the global permit so a hot loop cannot spin on the
        // permit.
        if !auto_compact_cooldown_elapsed(name) {
            metrics::counter!(
                "helix_auto_compact_total",
                "result" => "cooldown"
            )
            .increment(1);
            return Ok(false);
        }

        // Global single-compaction permit: never run two compactions at once
        // (each copies an entire env to disk and spikes RSS/page-cache).
        let _permit = match AUTO_COMPACT_SEMAPHORE.try_lock() {
            Ok(guard) => guard,
            Err(_) => {
                metrics::counter!(
                    "helix_auto_compact_total",
                    "result" => "busy"
                )
                .increment(1);
                return Ok(false);
            }
        };

        // Mark this collection as compacting for the rest of the call. Every
        // INTERNAL background Arc-taker (optimizer/backlog sweep, idle sweeper,
        // the auto-compact sweep's own iteration of OTHER collections) skips a
        // marked collection, so their transient `Arc::clone` cannot pin the
        // storage above strong_count==1 and starve the eviction-drain below.
        // RAII: the mark is cleared on EVERY exit path (success, busy-abort,
        // error, panic), so it can never leak and permanently exclude a
        // collection from maintenance. Set after the permit so exactly the one
        // collection that will actually drain is marked (no churn).
        let _in_progress = CompactionInProgressGuard::new(name);

        // Disk preflight: require free space >= current file size before
        // starting (the compacted copy is at most the original size). Abort
        // cleanly otherwise — never start a compaction that could ENOSPC mid-
        // copy.
        let collection_path = self.collection_path(name);
        if let Some(free) = available_disk_bytes(&collection_path) {
            if free < file_bytes {
                metrics::counter!(
                    "helix_auto_compact_total",
                    "result" => "insufficient_disk"
                )
                .increment(1);
                warn!(
                    collection = %name,
                    free_bytes = free,
                    needed_bytes = file_bytes,
                    "auto-compaction skipped: insufficient free disk"
                );
                return Ok(false);
            }
        }

        // Acquire the per-collection open_gate for the ENTIRE compaction (copy
        // + swap). It serializes against opens/closes/other compactions on this
        // name and is held by this stack frame (a MutexGuard cannot live in the
        // CompactionWindow struct alongside the Arc it conceptually protects).
        let open_gate = self.open_gate(name);
        let open_guard = match open_gate.lock() {
            Ok(g) => g,
            Err(_) => {
                metrics::counter!(
                    "helix_auto_compact_total",
                    "result" => "evict_error"
                )
                .increment(1);
                auto_compact_record_failure(name);
                return Ok(true);
            }
        };

        // Acquire the eviction/quiesce window BEFORE paying the copy. On a
        // serving instance the manager's own cache always holds an Arc (plus
        // any in-flight gateway handler), so the legacy
        // `restore_collection_from_snapshot_path` strong_count>1 check rejects
        // every time AFTER the full compact-copy I/O is already spent. Here we
        // evict-and-drain to a unique Arc FIRST: if the collection does not
        // quiesce within the bounded window, we abort with result="busy"
        // WITHOUT having copied anything (cost-free), and record the failure so
        // the cooldown backs off. While `open_guard` is held and the entry is
        // published in `closing`, new opens block and in-flight requests drain.
        let window = match self.evict_and_drain_for_compaction(name, &open_guard) {
            Ok(Some(window)) => window,
            Ok(None) => {
                // Did not quiesce in the bounded drain → busy. No copy paid.
                metrics::counter!(
                    "helix_auto_compact_total",
                    "result" => "busy"
                )
                .increment(1);
                auto_compact_record_failure(name);
                tracing::info!(
                    collection = %name,
                    "auto-compaction deferred: collection did not quiesce within drain window (no copy paid)"
                );
                return Ok(false);
            }
            Err(e) => {
                metrics::counter!(
                    "helix_auto_compact_total",
                    "result" => "evict_error"
                )
                .increment(1);
                auto_compact_record_failure(name);
                warn!(collection = %name, error = %e, "auto-compaction evict/drain failed; original retained");
                return Ok(true);
            }
        };

        // Mark the cooldown now (we hold the quiesced window) so a crash
        // mid-compact does not immediately re-arm on restart.
        auto_compact_mark_started(name);

        let started = Instant::now();
        info!(
            collection = %name,
            file_bytes,
            live_bytes,
            ratio = (file_bytes as f64 / live_bytes.max(1) as f64),
            "auto-compaction starting (collection quiesced)"
        );

        // Compact-copy the (now-private) env to a temp file under the
        // collection dir. The copy holds the read-resize guard so no map resize
        // races the copy, mirroring snapshot_collection_async. The env is the
        // unique Arc held by `window`, so no concurrent writer can advance it.
        let temp_compacted = collection_path.join("data.autocompact.tmp.mdb");
        let _ = fs::remove_file(&temp_compacted);

        let copy_started = Instant::now();
        let copy_result = (|| -> Result<u64, GraphError> {
            let _resize_guard = window.storage.read_resize_guard_if_needed()?;
            let file = window
                .storage
                .lmdb_env()?
                .copy_to_path(&temp_compacted, CompactionOption::Enabled)?;
            file.sync_data()?;
            let bytes = file.metadata()?.len();
            Ok(bytes)
        })();
        let copy_elapsed = copy_started.elapsed();
        metrics::histogram!("helix_auto_compact_copy_duration_ms")
            .record(copy_elapsed.as_secs_f64() * 1000.0);
        // H2: the open_gate is held for this whole copy, so any new writer to
        // this collection stalled for `copy_elapsed`. Surface slow copies so
        // operators can tune size/idle gates or schedule off-peak.
        if copy_result.is_ok() && copy_elapsed >= AUTO_COMPACT_SLOW_COPY_WARN {
            warn!(
                collection = %name,
                copy_ms = copy_elapsed.as_millis() as u64,
                file_bytes,
                "auto-compaction copy held the collection write-gate for >2s (new writers to this collection stalled for the copy duration); consider HELIX_COMPACT_MAX_FILE_BYTES or a tighter idle window"
            );
        }

        let compacted_bytes = match copy_result {
            Ok(bytes) => bytes,
            Err(e) => {
                let _ = fs::remove_file(&temp_compacted);
                // `window` drops below → env closes, `closing` cleared, next
                // access lazily reloads from the untouched original file.
                drop(window);
                metrics::counter!(
                    "helix_auto_compact_total",
                    "result" => "copy_error"
                )
                .increment(1);
                auto_compact_record_failure(name);
                warn!(collection = %name, error = %e, "auto-compaction copy failed; original retained");
                return Ok(true);
            }
        };

        // Swap the compacted file into place. We hold the exclusive window, so
        // the env can be closed (drop the unique Arc) and the files swapped
        // atomically: temp→data.restore.tmp→rename(data.mdb). The original is
        // only removed AFTER the temp restore copy succeeds, so any failure
        // here leaves the original `data.mdb` intact for the lazy reload.
        let swap_result = self.swap_in_compacted_file(name, window, &temp_compacted);
        let _ = fs::remove_file(&temp_compacted);

        match swap_result {
            Ok(()) => {
                auto_compact_clear_failures(name);
                let reclaimed = file_bytes.saturating_sub(compacted_bytes);
                metrics::counter!(
                    "helix_auto_compact_total",
                    "result" => "ok"
                )
                .increment(1);
                metrics::gauge!(
                    "helix_auto_compact_reclaimed_bytes",
                    "collection" => name.to_string()
                )
                .set(reclaimed as f64);
                info!(
                    collection = %name,
                    before_bytes = file_bytes,
                    after_bytes = compacted_bytes,
                    reclaimed_bytes = reclaimed,
                    elapsed_ms = started.elapsed().as_millis() as u64,
                    "auto-compaction complete"
                );
                Ok(true)
            }
            Err(e) => {
                metrics::counter!(
                    "helix_auto_compact_total",
                    "result" => "restore_error"
                )
                .increment(1);
                auto_compact_record_failure(name);
                warn!(
                    collection = %name,
                    error = %e,
                    "auto-compaction swap failed; original retained"
                );
                Ok(true)
            }
        }
    }

    /// Evict `name` from the manager's cache and drain in-flight holders until
    /// the storage `Arc` is unique, returning an exclusive [`CompactionWindow`]
    /// that holds the open_gate and that unique `Arc`.
    ///
    /// Returns `Ok(None)` when the collection does not quiesce within the
    /// bounded drain budget (in-flight requests still hold clones) — the entry
    /// is re-inserted and the caller must NOT proceed (cost-free busy abort).
    ///
    /// Mechanism (reuses the existing close/eviction quiesce model):
    ///   * hold the per-collection `open_gate` so no concurrent open/close/
    ///     compaction races this one;
    ///   * remove the entry from the cache and publish a `Weak` into `closing`
    ///     so NEW `get_collection` calls block in `wait_for_closing_collection`
    ///     rather than racing the swap;
    ///   * poll `Arc::strong_count` down to 1 (existing clones drain) within a
    ///     bounded budget;
    ///   * on success return the unique `Arc` in the window; on timeout
    ///     re-insert, clear `closing`, and return `None`.
    ///
    /// Readers are never blocked indefinitely: the drain budget is bounded and
    /// new opens wait at most `open_retry_budget_ms` (the same ceiling the
    /// normal close path uses) before the swap completes and they reload.
    ///
    /// The caller MUST hold this collection's `open_gate` guard for the entire
    /// compaction (copy + swap); it is passed in as `_open_guard` to prove that
    /// at the type level without this struct having to store a self-referential
    /// `MutexGuard`.
    fn evict_and_drain_for_compaction(
        &self,
        name: &str,
        _open_guard: &std::sync::MutexGuard<'_, ()>,
    ) -> Result<Option<CompactionWindow>, GraphError> {
        // Remove from the cache; if absent, the collection is cold — nothing to
        // compact through this serving path (the cold sweeper loads then
        // compacts on a later pass).
        let entry = {
            let mut collections = self
                .collections
                .write()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            collections.remove(name)
        };
        let Some(entry) = entry else {
            return Ok(None);
        };

        // Publish into `closing` so concurrent get_collection() blocks on the
        // weak ref instead of NotFound / racing the swap.
        if let Ok(mut closing) = self.closing.write() {
            closing.insert(name.to_string(), Arc::downgrade(&entry.storage));
        }

        let storage = entry.storage;
        let path = entry.path;

        // Drain in-flight clones to a unique Arc within a bounded budget.
        let budget = Duration::from_millis(auto_compact_drain_ms());
        let started = Instant::now();
        let mut backoff = Duration::from_millis(20);
        loop {
            if Arc::strong_count(&storage) == 1 {
                return Ok(Some(CompactionWindow { storage, path }));
            }
            if started.elapsed() >= budget {
                // Did not quiesce. Re-insert so the collection keeps serving
                // from the cache, clear `closing`, and report busy. No copy
                // was paid.
                if let Ok(mut closing) = self.closing.write() {
                    closing.remove(name);
                }
                if let Ok(mut collections) = self.collections.write() {
                    if !collections.contains_key(name) {
                        Self::cache_insert(
                            &mut collections,
                            name.to_string(),
                            Arc::clone(&storage),
                            self.open_reservations.load(Ordering::Acquire),
                        );
                    }
                }
                drop(storage);
                return Ok(None);
            }
            thread::sleep(backoff);
            backoff = (backoff * 2).min(Duration::from_millis(200));
        }
    }

    /// Swap a successfully-compacted temp file into `data.mdb`, consuming the
    /// exclusive [`CompactionWindow`]. The unique env Arc is dropped first
    /// (closing the env and releasing the file/mmap), then the files are
    /// swapped temp→`data.restore.tmp`→rename. The original `data.mdb` is only
    /// removed AFTER the temp restore copy succeeds, so any IO failure leaves
    /// the original intact. Finally `closing` is cleared so the next access
    /// lazily reopens the compacted file.
    fn swap_in_compacted_file(
        &self,
        name: &str,
        window: CompactionWindow,
        compacted_path: &Path,
    ) -> Result<(), GraphError> {
        let CompactionWindow {
            storage,
            path: collection_path,
        } = window;

        // Drop the unique Arc → close the env so the OS releases data.mdb's
        // mmap/fd before we rename over it. Wait (bounded) for heed's
        // process-wide OPENED_ENV registry to release the path.
        drop(storage);
        let release_budget = Duration::from_millis(open_retry_budget_ms());
        let release_started = Instant::now();
        loop {
            let still_open = self
                .closing
                .read()
                .ok()
                .and_then(|closing| closing.get(name).and_then(|w| w.upgrade()))
                .is_some();
            if !still_open || release_started.elapsed() >= release_budget {
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }

        let data_path = collection_path.join("data.mdb");
        let temp_restore = collection_path.join("data.restore.tmp");
        let lock_path = collection_path.join("lock.mdb");
        let wal_path = collection_path.join("wal");

        // Crash-safe swap. The compacted env captured ALL committed state
        // (copy_to_path snapshots the env up to its last commit, including the
        // committed LSN), so once `data.mdb` is the compacted file and durable
        // it is safe to delete the WAL. The ordering below guarantees that if a
        // crash occurs at ANY point the next restart either sees the original
        // intact (with its WAL) or the fully-durable compacted file:
        //
        //   1. copy compacted → data.restore.tmp, then fsync the temp file
        //      (its bytes must hit disk before it can become data.mdb).
        //   2. remove the original data.mdb, then rename temp → data.mdb.
        //   3. fsync the NEW data.mdb, then fsync the CONTAINING DIRECTORY so
        //      the rename's directory entry is journaled (without this, a crash
        //      can leave data.mdb pointing at an unflushed/zero-length inode).
        //   4. ONLY THEN delete lock.mdb and the WAL dir, and clear `closing`.
        //
        // `compacted_path` itself was already `sync_data()`'d by the copy phase;
        // the fresh `fs::copy` here produces new bytes that must be synced again.
        fs::copy(compacted_path, &temp_restore)?;
        sync_file_durable(&temp_restore)?;

        if data_path.exists() {
            fs::remove_file(&data_path)?;
        }
        fs::rename(&temp_restore, &data_path)?;

        // Durability barrier: the renamed data.mdb AND its directory entry must
        // be on disk before we destroy the WAL (the only other recovery source).
        sync_file_durable(&data_path)?;
        sync_dir_durable(&collection_path)?;

        // Now the compacted data.mdb is durable — safe to drop the stale lock
        // and the WAL (its contents are already folded into the compacted env).
        if lock_path.exists() {
            let _ = fs::remove_file(lock_path);
        }
        if wal_path.exists() {
            let _ = fs::remove_dir_all(&wal_path);
        }

        // Clear `closing` so the next get_collection reopens the compacted env
        // (lazy reload). The caller's open_gate guard is released when
        // `auto_compact_collection_if_needed` returns.
        if let Ok(mut closing) = self.closing.write() {
            closing.remove(name);
        }
        Ok(())
    }

    fn resolve_snapshot_path(&self, name: &str, snapshot: &str) -> Result<PathBuf, GraphError> {
        let snapshot_name = Path::new(snapshot);
        if snapshot_name.is_absolute()
            || snapshot_name.components().count() != 1
            || snapshot_name.file_name().and_then(|value| value.to_str()) != Some(snapshot)
        {
            return Err(GraphError::New(format!(
                "Invalid snapshot '{}' for collection '{}': expected a snapshot filename",
                snapshot, name
            )));
        }

        let snapshots_dir = self.collection_path(name).join("snapshots");
        let resolved = snapshots_dir.join(snapshot);
        if !resolved.exists() {
            return Err(GraphError::New(format!(
                "Snapshot '{}' not found for collection '{}'",
                snapshot, name
            )));
        }

        let canonical_snapshots_dir = fs::canonicalize(&snapshots_dir)?;
        let canonical_resolved = fs::canonicalize(&resolved)?;
        if !canonical_resolved.starts_with(&canonical_snapshots_dir) {
            return Err(GraphError::New(format!(
                "Invalid snapshot '{}' for collection '{}': path escapes snapshots directory",
                snapshot, name
            )));
        }

        Ok(canonical_resolved)
    }

    fn read_snapshot_info(
        &self,
        collection: &str,
        snapshot_path: &Path,
    ) -> Result<SnapshotInfo, GraphError> {
        let manifest_path = Self::snapshot_manifest_path(snapshot_path);
        if manifest_path.exists() {
            let manifest: SnapshotManifest = sonic_rs::from_slice(&fs::read(&manifest_path)?)?;
            return Ok(manifest.info);
        }

        let metadata = fs::metadata(snapshot_path)?;
        let created_at_millis = snapshot_path
            .file_stem()
            .and_then(|value| value.to_str())
            .and_then(|value| value.parse::<i64>().ok())
            .or_else(|| {
                metadata
                    .modified()
                    .ok()
                    .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
                    .map(|duration| duration.as_millis() as i64)
            })
            .unwrap_or_default();

        Ok(SnapshotInfo {
            name: snapshot_path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or(collection)
                .to_string(),
            lsn: 0,
            path: snapshot_path.display().to_string(),
            disk_bytes: metadata.len(),
            created_at_millis,
        })
    }
}

fn prune_snapshots(path: &Path, keep_last: usize) -> Result<(), GraphError> {
    let mut files: Vec<PathBuf> = fs::read_dir(path)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().map(|ft| ft.is_file()).unwrap_or(false))
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("mdb"))
        .collect();

    files.sort();
    if files.len() <= keep_last {
        return Ok(());
    }

    let remove_count = files.len() - keep_last;
    for old in files.into_iter().take(remove_count) {
        fs::remove_file(&old)?;
        let manifest = CollectionManager::snapshot_manifest_path(&old);
        if manifest.exists() {
            fs::remove_file(manifest)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helix_engine::graph_core::config::{GraphConfig, VectorConfig};
    use crate::helix_engine::storage_core::backend::{LsmRole, LsmStorage};
    use crate::helix_engine::storage_core::backend_lsm::{
        allow_lsm_blocking_cancellable, LsmReadCancellation,
    };
    use crate::helix_engine::storage_core::storage_methods::StorageMethods;
    use crate::protocol::value::Value;
    use std::sync::LazyLock;
    use tempfile::TempDir;

    static ENV_TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    #[test]
    fn missing_manifest_open_error_maps_to_not_found() {
        let err = missing_manifest_open_error(
            "org_repo-abc123",
            GraphError::StorageError(
                "io error: Data error: failed to find latest transactional object (e.g. manifest) version"
                    .into(),
            ),
        );
        let message = err.to_string();
        assert!(
            message.contains("Collection 'org_repo-abc123' not found"),
            "missing manifest must surface as not-found (404), got: {message}"
        );
        assert!(
            !err.should_quarantine_collection(),
            "not-found must never quarantine the collection name"
        );

        let other = missing_manifest_open_error(
            "org_repo-abc123",
            GraphError::StorageError("io error: connection reset by peer".into()),
        );
        assert!(
            other.to_string().contains("connection reset"),
            "unrelated open errors must pass through unchanged"
        );
    }

    #[test]
    fn lsm_reader_database_missing_open_error_maps_to_not_found() {
        // Exact shape `new_lsm` produces from `LsmReader::open_with_store` when
        // SlateDB reports `ErrorCode::DatabaseMissing`.
        let err = missing_manifest_open_error(
            "codebase",
            GraphError::StorageError(format!(
                "io error: {LSM_READER_DATABASE_MISSING}: Data error: database does not exist"
            )),
        );
        assert!(
            matches!(&err, GraphError::New(m)
                if m == "Collection 'codebase' not found (no manifest in object store)"),
            "missing reader database must surface as not-found (404), got: {err}"
        );
        assert!(
            !err.should_quarantine_collection(),
            "not-found must never quarantine the collection name"
        );

        // Other reader-open storage failures (S3, corruption, other SlateDB
        // data errors) must stay unchanged so they still surface as 500.
        for message in [
            "io error: Generic S3 error: request failed: 503 Slow Down",
            "io error: Data error: invalid DB state error",
            "io error: Data error: database does not exist",
        ] {
            let other =
                missing_manifest_open_error("codebase", GraphError::StorageError(message.into()));
            assert!(
                matches!(&other, GraphError::StorageError(m) if m == message),
                "unrelated reader-open errors must pass through unchanged, got: {other}"
            );
        }
    }

    struct EnvRestore {
        key: &'static str,
        value: Option<String>,
    }

    impl EnvRestore {
        fn set(key: &'static str, value: &str) -> Self {
            let original = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self {
                key,
                value: original,
            }
        }

        fn unset(key: &'static str) -> Self {
            let original = std::env::var(key).ok();
            std::env::remove_var(key);
            Self {
                key,
                value: original,
            }
        }
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            match &self.value {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    fn test_config() -> Config {
        Config::new(16, 128, 768, 1) // 1GB max for tests
    }

    /// Pure decision table for the read-gated poll-tier demotion sweep — the
    /// MEDIUM finding fix from the poll-tiering review's final round.
    /// `write_recent` always being `false` outside feed mode must reduce the
    /// rule to the original read-only behavior (sweeper demotes a fast,
    /// read-idle collection); `write_recent = true` must block demotion even
    /// though the collection is fast and read-idle (the feed-mode gap the fix
    /// closes) — the sweeper defers to the write-feed poller's own recency
    /// instead of stomping a collection it just promoted for a write burst.
    #[test]
    fn reader_poll_tier_sweeper_should_demote_gates_on_all_three_signals() {
        // Fast, read-idle, write-idle: demote. Non-feed-mode shape too, since
        // write_recent is always false there.
        assert!(reader_poll_tier_sweeper_should_demote(true, false, false));

        // Not fast: nothing to demote.
        assert!(!reader_poll_tier_sweeper_should_demote(false, false, false));

        // Read-recently: the read path owns freshness, sweeper must not demote.
        assert!(!reader_poll_tier_sweeper_should_demote(true, true, false));

        // Write-feed promoted this collection recently (feed mode): the
        // sweeper must stand down even though it's fast and read-idle — this
        // is exactly the case that was broken before this fix (the sweeper
        // didn't run in feed mode at all, so it never even reached this
        // decision; now it reaches it and must say "no").
        assert!(!reader_poll_tier_sweeper_should_demote(true, false, true));

        // Both recent: still no.
        assert!(!reader_poll_tier_sweeper_should_demote(true, true, true));
    }

    #[test]
    fn lsm_reader_cold_open_is_enabled_only_for_reader_role() {
        let reader = StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role: LsmRole::Reader,
        };
        let writer = StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role: LsmRole::Writer,
        };

        assert!(lsm_reader_cold_open_enabled(reader));
        assert!(!lsm_reader_cold_open_enabled(writer));
        assert!(!lsm_reader_cold_open_enabled(StorageBackendConfig::Lmdb));
    }

    #[test]
    fn lsm_cold_open_masks_cancellation_for_reader_and_writer_roles() {
        let reader = StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role: LsmRole::Reader,
        };
        let writer = StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role: LsmRole::Writer,
        };

        assert!(lsm_cold_open_needs_cancellation_mask(reader));
        assert!(lsm_cold_open_needs_cancellation_mask(writer));
        assert!(!lsm_cold_open_needs_cancellation_mask(
            StorageBackendConfig::Lmdb
        ));
    }

    #[test]
    fn lsm_opened_handles_for_one_collection_share_write_gate() {
        let tmp = TempDir::new().unwrap();
        let mgr =
            CollectionManager::new(tmp.path().to_path_buf(), test_config().with_lsm_in_memory())
                .unwrap();

        let path = mgr.collection_path("same");
        let first = mgr.open_collection_storage(&path, "same").unwrap();
        let second = mgr.open_collection_storage(&path, "same").unwrap();
        assert!(Arc::ptr_eq(&first.write_txn_gate, &second.write_txn_gate));

        let other_path = mgr.collection_path("other");
        let other = mgr.open_collection_storage(&other_path, "other").unwrap();
        assert!(!Arc::ptr_eq(&first.write_txn_gate, &other.write_txn_gate));
    }

    #[test]
    fn cancelled_writer_cold_open_skips_cache_and_failure_backoff() {
        let tmp = TempDir::new().unwrap();
        let mgr =
            CollectionManager::new(tmp.path().to_path_buf(), test_config().with_lsm_in_memory())
                .unwrap();
        let name = "cancelled_writer_cold_open";
        let storage = mgr.create_collection(name).unwrap();
        drop(storage);
        mgr.evict_collection(name);
        assert_eq!(mgr.loaded_count(), 0);

        let cancellation = LsmReadCancellation::new();
        cancellation.cancel();
        let cancelled = allow_lsm_blocking_cancellable(cancellation, || mgr.get_collection(name));
        assert!(
            matches!(cancelled, Err(ref error) if error.to_string().contains("LSM read request cancelled")),
            "a request cancelled behind the open gate must not cold-open the writer"
        );
        assert_eq!(mgr.loaded_count(), 0);

        // Cancellation is not an unhealthy open and must not poison the
        // collection's open-failure backoff; the next live request can load it.
        let reopened = mgr
            .get_collection(name)
            .expect("live request should cold-open immediately after cancellation");
        assert_eq!(mgr.loaded_count(), 1);
        drop(reopened);
    }

    #[test]
    fn lsm_collection_cache_obeys_open_limit_by_default() {
        let _lock = ENV_TEST_LOCK.lock().unwrap();
        let _max_open = EnvRestore::set("HELIX_MAX_OPEN_COLLECTIONS", "1");
        let _unbounded = EnvRestore::unset("HELIX_LSM_COLLECTION_CACHE_UNBOUNDED");

        let tmp = TempDir::new().unwrap();
        let mgr =
            CollectionManager::new(tmp.path().to_path_buf(), test_config().with_lsm_in_memory())
                .unwrap();

        mgr.create_collection("first").unwrap();
        mgr.create_collection("second").unwrap();

        assert_eq!(mgr.loaded_count(), 1);
        assert_eq!(mgr.open_reservations.load(Ordering::Acquire), 0);
        assert_eq!(
            mgr.maintenance_cold_open_admission("third"),
            MaintenanceAdmission::Rejected("open_capacity")
        );
    }

    #[test]
    fn lsm_collection_cache_unbounded_when_explicitly_enabled() {
        let _lock = ENV_TEST_LOCK.lock().unwrap();
        let _max_open = EnvRestore::set("HELIX_MAX_OPEN_COLLECTIONS", "1");
        let _unbounded = EnvRestore::set("HELIX_LSM_COLLECTION_CACHE_UNBOUNDED", "1");

        let tmp = TempDir::new().unwrap();
        let mgr =
            CollectionManager::new(tmp.path().to_path_buf(), test_config().with_lsm_in_memory())
                .unwrap();

        mgr.create_collection("first").unwrap();
        mgr.create_collection("second").unwrap();

        assert_eq!(mgr.loaded_count(), 2);
        assert_eq!(mgr.open_reservations.load(Ordering::Acquire), 0);
        assert_eq!(
            mgr.maintenance_cold_open_admission("third"),
            MaintenanceAdmission::Admitted
        );
    }

    #[test]
    fn periodic_snapshots_skip_lsm_backends() {
        assert!(skip_periodic_snapshot_for_backend(BackendKind::Lsm));
        assert!(!skip_periodic_snapshot_for_backend(BackendKind::Lmdb));
    }

    #[test]
    fn test_create_and_get_collection() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();

        let storage = mgr.create_collection("test_coll").unwrap();
        // Should be able to get it back
        let storage2 = mgr.get_collection("test_coll").unwrap();
        assert!(Arc::ptr_eq(&storage, &storage2));
    }

    #[test]
    fn point_mutation_gate_serializes_same_collection_only() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();

        let first = mgr.point_mutation_gate("test_coll");
        let second = mgr.point_mutation_gate("test_coll");
        let other = mgr.point_mutation_gate("other_coll");

        assert!(Arc::ptr_eq(&first, &second));
        assert!(!Arc::ptr_eq(&first, &other));

        let guard = first.lock().unwrap();
        assert!(second.try_lock().is_err());
        assert!(other.try_lock().is_ok());
        drop(guard);
    }

    #[test]
    fn meminfo_total_parser_reads_kilobytes() {
        assert_eq!(
            parse_meminfo_total_bytes("MemTotal:       18000000 kB\nMemFree: 1 kB\n"),
            Some(18_000_000usize * 1024)
        );
    }

    #[test]
    fn cache_drop_hint_paths_include_lmdb_and_vector_sidecars_only() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("data.mdb"), b"lmdb").unwrap();
        fs::write(tmp.path().join("dense__seg_000001.hvec"), b"hvec").unwrap();
        fs::write(tmp.path().join("dense__seg_000001.hvs8"), b"hvs8").unwrap();
        fs::write(tmp.path().join("dense__seg_000001.hvtq"), b"hvtq").unwrap();
        fs::write(tmp.path().join("metadata.json"), b"{}").unwrap();
        fs::write(tmp.path().join("lock.mdb"), b"lock").unwrap();
        fs::create_dir(tmp.path().join("snapshots")).unwrap();
        fs::write(tmp.path().join("snapshots").join("0001.mdb"), b"snapshot").unwrap();

        let mut names: Vec<String> = cache_drop_hint_paths(tmp.path())
            .into_iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        names.sort();

        assert_eq!(
            names,
            vec![
                "data.mdb".to_string(),
                "dense__seg_000001.hvec".to_string(),
                "dense__seg_000001.hvs8".to_string(),
                "dense__seg_000001.hvtq".to_string(),
            ]
        );
    }

    #[test]
    fn cache_drop_hint_preserves_lsm_object_store_cache_subtree() {
        let _lock = ENV_TEST_LOCK.lock().unwrap();
        let _hint = EnvRestore::set("HELIX_CACHE_DROP_HINT", "1");
        let _prefix = EnvRestore::set("HELIX_LSM_PREFIX", "prod");
        let cache_root = TempDir::new().unwrap();
        let _cache = EnvRestore::set("HELIX_LSM_CACHE_DIR", cache_root.path().to_str().unwrap());
        let collections = TempDir::new().unwrap();
        let collection_path = collections.path().join("my_collection");
        fs::create_dir(&collection_path).unwrap();
        let cache_subtree = cache_root.path().join("prod_my_collection");
        fs::create_dir_all(&cache_subtree).unwrap();
        fs::write(cache_subtree.join("sst.bin"), b"cached").unwrap();

        advise_collection_cache_dropped(&collection_path, "my_collection", "test");

        assert!(cache_subtree.exists());
    }

    #[test]
    fn cache_drop_hint_disabled_preserves_lsm_object_store_cache_subtree() {
        let _lock = ENV_TEST_LOCK.lock().unwrap();
        let _hint = EnvRestore::set("HELIX_CACHE_DROP_HINT", "0");
        let _prefix = EnvRestore::set("HELIX_LSM_PREFIX", "prod");
        let cache_root = TempDir::new().unwrap();
        let _cache = EnvRestore::set("HELIX_LSM_CACHE_DIR", cache_root.path().to_str().unwrap());
        let collections = TempDir::new().unwrap();
        let collection_path = collections.path().join("disabled_hint");
        fs::create_dir(&collection_path).unwrap();
        let cache_subtree = cache_root.path().join("prod_disabled_hint");
        fs::create_dir_all(&cache_subtree).unwrap();
        fs::write(cache_subtree.join("sst.bin"), b"cached").unwrap();

        advise_collection_cache_dropped(&collection_path, "disabled_hint", "test");

        assert!(cache_subtree.exists());
    }

    #[test]
    fn lsm_orphan_object_cache_sweep_removes_only_non_retained_dirs() {
        let cache_root = TempDir::new().unwrap();
        let retained_dir = cache_root.path().join("prod_loaded");
        let orphan_dir = cache_root.path().join("prod_orphan");
        let plain_file = cache_root.path().join("not_a_cache_dir");
        fs::create_dir_all(&retained_dir).unwrap();
        fs::create_dir_all(&orphan_dir).unwrap();
        fs::write(retained_dir.join("sst.bin"), b"retained").unwrap();
        fs::write(orphan_dir.join("sst.bin"), b"orphan").unwrap();
        fs::write(&plain_file, b"file").unwrap();
        let retained = HashSet::from([retained_dir.clone()]);

        let removed = remove_lsm_object_cache_dirs_not_in(cache_root.path(), &retained, "test");

        assert_eq!(removed, 1);
        assert!(retained_dir.exists());
        assert!(!orphan_dir.exists());
        assert!(plain_file.exists());
    }

    #[test]
    fn lsm_startup_object_cache_sweep_retains_catalog_marker_dirs() {
        let _lock = ENV_TEST_LOCK.lock().unwrap();
        let _prefix = EnvRestore::set("HELIX_LSM_PREFIX", "prod");
        let cache_root = TempDir::new().unwrap();
        let _cache = EnvRestore::set("HELIX_LSM_CACHE_DIR", cache_root.path().to_str().unwrap());
        let collections_root = TempDir::new().unwrap();
        let collections_dir = collections_root.path().join("collections");
        fs::create_dir(&collections_dir).unwrap();
        fs::create_dir(collections_dir.join("loaded")).unwrap();

        let retained_dir = cache_root.path().join("prod_loaded");
        let orphan_dir = cache_root.path().join("prod_orphan");
        fs::create_dir_all(&retained_dir).unwrap();
        fs::create_dir_all(&orphan_dir).unwrap();
        fs::write(retained_dir.join("sst.bin"), b"retained").unwrap();
        fs::write(orphan_dir.join("sst.bin"), b"orphan").unwrap();
        let collections = Arc::new(RwLock::new(HashMap::new()));

        sweep_lsm_orphan_object_cache_dirs(
            &collections,
            &collections_dir,
            "startup_orphan",
            StorageBackendConfig::Lsm {
                storage: LsmStorage::ObjectStore,
                role: LsmRole::Writer,
            },
        );

        assert!(retained_dir.exists());
        assert!(!orphan_dir.exists());
    }

    #[test]
    fn page_cache_sweeper_candidates_skip_loaded_and_advance_cursor() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir(tmp.path().join("alpha")).unwrap();
        fs::create_dir(tmp.path().join("beta")).unwrap();
        fs::create_dir(tmp.path().join("gamma")).unwrap();
        fs::write(tmp.path().join("not_a_collection"), b"file").unwrap();

        let loaded = HashSet::from(["beta".to_string()]);
        let mut cursor = 0;
        let first = cold_collection_cache_drop_candidates(tmp.path(), &loaded, &mut cursor, 1);
        let second = cold_collection_cache_drop_candidates(tmp.path(), &loaded, &mut cursor, 1);
        let third = cold_collection_cache_drop_candidates(tmp.path(), &loaded, &mut cursor, 1);

        assert_eq!(first[0].0, "alpha");
        assert_eq!(second[0].0, "gamma");
        assert_eq!(third[0].0, "alpha");
    }

    #[test]
    fn memory_pressure_eviction_removes_idle_lru_entries() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();

        mgr.create_collection("a").unwrap();
        mgr.create_collection("b").unwrap();
        mgr.create_collection("c").unwrap();
        mgr.create_collection("d").unwrap();
        assert_eq!(mgr.loaded_count(), 4);

        let evicted = {
            let mut collections = mgr.collections.write().unwrap();
            CollectionManager::evict_idle_for_memory(&mut collections, "memory")
        };

        assert_eq!(evicted, 1);
        assert_eq!(mgr.loaded_count(), 3);
    }

    #[test]
    fn idle_age_eviction_skips_pinned_entries_and_keeps_warm_floor() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();

        mgr.create_collection("a").unwrap();
        mgr.create_collection("b").unwrap();
        mgr.create_collection("c").unwrap();
        mgr.create_collection("d").unwrap();
        let _pinned = mgr.get_collection("b").unwrap();

        let now_millis = 10_000;
        {
            let collections = mgr.collections.write().unwrap();
            for entry in collections.values() {
                entry.last_access_millis.store(1_000, Ordering::Relaxed);
            }
        }

        let evicted = {
            let mut collections = mgr.collections.write().unwrap();
            CollectionManager::evict_idle_by_age(&mut collections, now_millis, 5_000, 8, 2, "idle")
        };

        assert_eq!(evicted, 2);
        assert_eq!(mgr.loaded_count(), 2);

        let collections = mgr.collections.read().unwrap();
        assert!(collections.contains_key("b"));
    }

    #[test]
    fn idle_age_eviction_respects_budget_and_oldest_touch_order() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();

        mgr.create_collection("a").unwrap();
        mgr.create_collection("b").unwrap();
        mgr.create_collection("c").unwrap();

        {
            let collections = mgr.collections.write().unwrap();
            collections
                .get("a")
                .unwrap()
                .last_access_millis
                .store(1_000, Ordering::Relaxed);
            collections
                .get("b")
                .unwrap()
                .last_access_millis
                .store(2_000, Ordering::Relaxed);
            collections
                .get("c")
                .unwrap()
                .last_access_millis
                .store(3_000, Ordering::Relaxed);
        }

        let evicted = {
            let mut collections = mgr.collections.write().unwrap();
            CollectionManager::evict_idle_by_age(&mut collections, 10_000, 5_000, 1, 0, "idle")
        };

        assert_eq!(evicted, 1);
        let collections = mgr.collections.read().unwrap();
        assert!(!collections.contains_key("a"));
        assert!(collections.contains_key("b"));
        assert!(collections.contains_key("c"));
    }

    #[test]
    fn memory_usage_working_set_subtracts_inactive_file() {
        let usage = MemoryUsage {
            current_bytes: 18 * 1024 * 1024 * 1024,
            inactive_file_bytes: 17 * 1024 * 1024 * 1024,
            active_file_bytes: 0,
            file_bytes: 0,
            shmem_bytes: 0,
            file_dirty_bytes: 0,
            file_writeback_bytes: 0,
        };

        assert_eq!(usage.working_set_bytes(), 1024 * 1024 * 1024);
    }

    #[test]
    fn memory_usage_pressure_subtracts_clean_active_file_cache() {
        let gib = 1024 * 1024 * 1024;
        let usage = MemoryUsage {
            current_bytes: 24 * gib,
            inactive_file_bytes: 1 * gib,
            active_file_bytes: 19 * gib,
            file_bytes: 20 * gib,
            shmem_bytes: 1 * gib,
            file_dirty_bytes: 512 * 1024 * 1024,
            file_writeback_bytes: 512 * 1024 * 1024,
        };

        assert_eq!(usage.working_set_bytes(), 23 * gib);
        assert_eq!(usage.reclaimable_file_bytes(), 18 * gib);
        assert_eq!(usage.pressure_bytes(), 6 * gib);
    }

    #[test]
    fn memory_usage_from_stat_supports_v2_and_v1_names() {
        let gib = 1024 * 1024 * 1024;
        let v2 = "\
anon 3221225472
file 21474836480
inactive_file 1073741824
active_file 20401094656
shmem 0
file_dirty 0
file_writeback 0
";
        let usage = memory_usage_from_stat(24 * gib, v2);
        assert_eq!(usage.file_bytes, 20 * gib);
        assert_eq!(usage.inactive_file_bytes, gib);
        assert_eq!(usage.active_file_bytes, 19 * gib);
        assert_eq!(usage.pressure_bytes(), 4 * gib);

        let v1 = "\
total_cache 21474836480
total_inactive_file 1073741824
total_active_file 20401094656
total_shmem 1073741824
total_dirty 0
total_writeback 0
";
        let usage = memory_usage_from_stat(24 * gib, v1);
        assert_eq!(usage.reclaimable_file_bytes(), 19 * gib);
        assert_eq!(usage.pressure_bytes(), 5 * gib);
    }

    #[test]
    fn collection_open_pressure_prefers_reclaim_aware_pressure() {
        let snapshot = CollectionMemorySnapshot {
            current_bytes: Some(18 * 1024 * 1024 * 1024),
            inactive_file_bytes: Some(17 * 1024 * 1024 * 1024),
            admission_pressure_bytes: Some(2 * 1024 * 1024 * 1024),
            working_set_bytes: Some(1024 * 1024 * 1024),
            ceiling_bytes: Some(24 * 1024 * 1024 * 1024),
            low_bytes: Some(14 * 1024 * 1024 * 1024),
            high_bytes: Some(18 * 1024 * 1024 * 1024),
            ..Default::default()
        };

        assert_eq!(snapshot.pressure_bytes(), Some(2 * 1024 * 1024 * 1024));
    }

    #[test]
    fn collection_open_pressure_falls_back_to_current_without_working_set() {
        let snapshot = CollectionMemorySnapshot {
            current_bytes: Some(18 * 1024 * 1024 * 1024),
            inactive_file_bytes: None,
            working_set_bytes: None,
            ceiling_bytes: Some(24 * 1024 * 1024 * 1024),
            low_bytes: Some(14 * 1024 * 1024 * 1024),
            high_bytes: Some(18 * 1024 * 1024 * 1024),
            ..Default::default()
        };

        assert_eq!(snapshot.pressure_bytes(), Some(18 * 1024 * 1024 * 1024));
    }

    #[test]
    fn current_high_allows_reclaimable_file_cache_when_pressure_is_low() {
        let gib = 1024 * 1024 * 1024;
        let snapshot = CollectionMemorySnapshot {
            current_bytes: Some(31 * gib),
            file_bytes: Some(29 * gib),
            reclaimable_file_bytes: Some(29 * gib),
            working_set_bytes: Some(2 * gib),
            admission_pressure_bytes: Some(2 * gib),
            ceiling_bytes: Some(32 * gib),
            ..Default::default()
        };

        assert_eq!(
            snapshot.current_above_high(Some(28 * gib)),
            Some((31 * gib, 28 * gib))
        );
        assert_eq!(
            snapshot.current_high_requires_block_at(Some(28 * gib), Some(24 * gib), gib),
            None
        );
    }

    #[test]
    fn current_high_blocks_when_reclaim_aware_pressure_is_high() {
        let gib = 1024 * 1024 * 1024;
        let snapshot = CollectionMemorySnapshot {
            current_bytes: Some(31 * gib),
            file_bytes: Some(8 * gib),
            reclaimable_file_bytes: Some(8 * gib),
            working_set_bytes: Some(25 * gib),
            admission_pressure_bytes: Some(25 * gib),
            ceiling_bytes: Some(32 * gib),
            ..Default::default()
        };

        assert_eq!(
            snapshot.current_high_requires_block_at(Some(28 * gib), Some(24 * gib), gib),
            Some((31 * gib, 28 * gib))
        );
    }

    /// Locks the consolidation in place: both sweepers must derive their
    /// decision from the single arbiter (`pressure_decision`) so they cannot
    /// drift apart again (the c8d90da3 prod bug). We build a grid over
    /// current-above/below-high, reclaimable-file-cache large/small, and
    /// pressure safe/unsafe, then assert the idle-sweeper (evict) and
    /// page-cache-sweeper (sweep) decisions are mutually consistent and
    /// complementary: when `current >= high` is pure reclaimable page cache
    /// the page-cache sweeper acts (drops file cache), otherwise the idle
    /// sweeper acts (evicts collection envs).
    #[test]
    fn sweepers_agree_via_single_arbiter() {
        let gib = 1024 * 1024 * 1024;
        let current_high = Some(28 * gib);
        let pressure_high = Some(24 * gib);
        let reclaim_floor = gib; // 1 GiB

        // (label, current, reclaimable_file, pressure)
        let grid = [
            ("below_high", 20 * gib, 4 * gib, 5 * gib),
            ("at_high_large_reclaim_safe", 28 * gib, 4 * gib, 5 * gib),
            ("above_high_large_reclaim_safe", 31 * gib, 29 * gib, 2 * gib),
            (
                "above_high_large_reclaim_unsafe",
                31 * gib,
                29 * gib,
                25 * gib,
            ),
            ("above_high_small_reclaim_safe", 31 * gib, 0, 2 * gib),
            ("above_high_small_reclaim_unsafe", 31 * gib, 0, 25 * gib),
        ];

        for (label, current, reclaim, pressure) in grid {
            let snapshot = CollectionMemorySnapshot {
                current_bytes: Some(current),
                reclaimable_file_bytes: Some(reclaim),
                admission_pressure_bytes: Some(pressure),
                ceiling_bytes: Some(32 * gib),
                ..Default::default()
            };

            // Idle sweeper call site: reclaim-aware pressure high, so the
            // exemption applies when pressure is safe and the file cache is
            // large — a backfill's page cache no longer triggers env eviction.
            let idle_decision =
                snapshot.pressure_decision(current_high, pressure_high, reclaim_floor);
            let idle_evicts = idle_decision.requires_action().is_some();

            // Page-cache sweeper call site: same arbiter, acts only when
            // `current >= high` AND there is a large reclaimable file cache.
            let pc_decision = snapshot.pressure_decision(current_high, None, reclaim_floor);
            let pc_sweeps =
                pc_decision.current_high.is_some() && pc_decision.reclaimable_file_is_large;

            // Both decisions share one `current_high` fact.
            assert_eq!(
                idle_decision.current_high, pc_decision.current_high,
                "[{label}] sweepers must agree on current-vs-high"
            );

            // Idle sweeper evicts iff current >= high AND the exemption does
            // not apply (pressure unsafe OR no large reclaimable file cache).
            let exempt = pressure < pressure_high.unwrap() && reclaim >= reclaim_floor;
            assert_eq!(
                idle_evicts,
                snapshot.current_above_high(current_high).is_some() && !exempt,
                "[{label}] idle sweeper must act on current >= high minus the exemption"
            );

            // Above the high watermark at least one sweeper must act: either
            // the idle sweeper evicts envs or the page-cache sweeper drops
            // reclaimable file cache.
            if snapshot.current_above_high(current_high).is_some() {
                assert!(
                    idle_evicts || pc_sweeps,
                    "[{label}] no sweeper acted above high"
                );
            }

            // Neither sweeper may act below the high watermark.
            if snapshot.current_above_high(current_high).is_none() {
                assert!(!idle_evicts, "[{label}] idle sweeper acted below high");
                assert!(!pc_sweeps, "[{label}] page-cache sweeper acted below high");
            }
        }
    }

    /// Hysteresis: the idle sweeper enters eviction on `requires_action`,
    /// holds while `current` sits in the [recovery, high) band, and exits
    /// when `current` drops below recovery or the reclaimable-file exemption
    /// starts to apply.
    #[test]
    fn idle_sweeper_hysteresis_band() {
        let gib: usize = 1024 * 1024 * 1024;
        let high = Some(28 * gib);
        let reclaim_floor = gib;
        let recovery_pct = 85; // recovery = 23.8 GiB

        let snapshot_at =
            |current: usize, reclaim: usize, pressure: usize| CollectionMemorySnapshot {
                current_bytes: Some(current),
                reclaimable_file_bytes: Some(reclaim),
                admission_pressure_bytes: Some(pressure),
                ceiling_bytes: Some(32 * gib),
                ..Default::default()
            };
        let decide = |current: usize, reclaim: usize, pressure: usize, was: bool| {
            let snapshot = snapshot_at(current, reclaim, pressure);
            let decision = snapshot.pressure_decision(high, Some(24 * gib), reclaim_floor);
            idle_sweeper_should_evict(decision, Some(current), high, recovery_pct, was)
        };

        // Not evicting, below high: stays off.
        assert!(!decide(20 * gib, 0, 20 * gib, false));
        // Crosses high with no exemption: turns on.
        assert!(decide(29 * gib, 0, 29 * gib, false));
        // Drops into the band [recovery, high): holds on.
        assert!(decide(26 * gib, 0, 26 * gib, true));
        // Same band without prior eviction: stays off (no fresh trigger).
        assert!(!decide(26 * gib, 0, 26 * gib, false));
        // Drops below recovery (23.8 GiB): turns off.
        assert!(!decide(23 * gib, 0, 23 * gib, true));
        // In the band but the exemption now applies (safe pressure + large
        // reclaimable file cache): turns off — evicting envs is pointless.
        assert!(!decide(26 * gib, 10 * gib, 5 * gib, true));
        // Above high but exempt: never turns on.
        assert!(!decide(31 * gib, 29 * gib, 2 * gib, false));
    }

    #[test]
    fn maintenance_loaded_limit_rounds_up_and_stays_in_range() {
        assert_eq!(maintenance_open_loaded_limit(256, 85), 218);
        assert_eq!(maintenance_open_loaded_limit(64, 85), 55);
        assert_eq!(maintenance_open_loaded_limit(3, 1), 1);
        assert_eq!(maintenance_open_loaded_limit(3, 100), 3);
        assert_eq!(maintenance_open_loaded_limit(0, 85), 1);
    }

    #[test]
    fn maintenance_admission_reason_is_stable() {
        assert!(MaintenanceAdmission::Admitted.is_admitted());
        assert_eq!(MaintenanceAdmission::Admitted.reason(), "admitted");
        assert!(!MaintenanceAdmission::Rejected("memory").is_admitted());
        assert_eq!(MaintenanceAdmission::Rejected("memory").reason(), "memory");
    }

    /// The maintenance memory gate must be reclaim-aware: a node whose
    /// `current` is pinned at the cgroup limit by clean mmap page cache is
    /// healthy and must stay admitted; only genuine anonymous-memory pressure
    /// rejects maintenance.
    #[test]
    fn maintenance_memory_gate_ignores_reclaimable_page_cache() {
        let gib = 1024 * 1024 * 1024;
        let high = Some(24 * gib);

        // Live-incident shape: current at the 32GiB ceiling, but nearly all of
        // it reclaimable file cache — pressure is ~2GiB. Must NOT reject.
        let warm_cache = CollectionMemorySnapshot {
            current_bytes: Some(34 * gib),
            reclaimable_file_bytes: Some(32 * gib),
            admission_pressure_bytes: Some(2 * gib),
            ..Default::default()
        };
        assert!(!maintenance_memory_pressure_exceeds(warm_cache, high));

        // Genuine pressure: anonymous memory above the watermark. Must reject.
        let anon_pressure = CollectionMemorySnapshot {
            current_bytes: Some(30 * gib),
            reclaimable_file_bytes: Some(2 * gib),
            admission_pressure_bytes: Some(28 * gib),
            ..Default::default()
        };
        assert!(maintenance_memory_pressure_exceeds(anon_pressure, high));

        // Degraded snapshot (no memory.stat): falls back to current_bytes via
        // pressure_bytes(), preserving the old conservative behavior.
        let no_stat = CollectionMemorySnapshot {
            current_bytes: Some(30 * gib),
            ..Default::default()
        };
        assert!(maintenance_memory_pressure_exceeds(no_stat, high));

        // Missing data on either side never rejects.
        assert!(!maintenance_memory_pressure_exceeds(
            CollectionMemorySnapshot::default(),
            high
        ));
        assert!(!maintenance_memory_pressure_exceeds(warm_cache, None));
    }

    #[test]
    fn test_create_duplicate_fails() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();

        mgr.create_collection("dupe").unwrap();
        let err = mgr.create_collection("dupe");
        assert!(err.is_err());
    }

    #[test]
    fn test_get_nonexistent_fails() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();

        let err = mgr.get_collection("nope");
        assert!(err.is_err());
    }

    #[test]
    fn test_get_or_create_idempotent() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();

        let s1 = mgr.get_or_create_collection("idem").unwrap();
        let s2 = mgr.get_or_create_collection("idem").unwrap();
        assert!(Arc::ptr_eq(&s1, &s2));
    }

    #[test]
    fn test_list_collections() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();

        mgr.create_collection("alpha").unwrap();
        mgr.create_collection("beta").unwrap();
        mgr.create_collection("gamma").unwrap();

        let list = mgr.list_collections().unwrap();
        assert_eq!(list, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn list_and_exists_do_not_open_cold_collections() {
        let tmp = TempDir::new().unwrap();
        {
            let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
            mgr.create_collection("alpha").unwrap();
            mgr.create_collection("beta").unwrap();
            assert_eq!(mgr.loaded_count(), 2);
        }

        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        assert_eq!(mgr.loaded_count(), 0);

        assert_eq!(
            mgr.list_collections().unwrap(),
            vec!["alpha".to_string(), "beta".to_string()]
        );
        assert_eq!(mgr.loaded_count(), 0);

        assert!(mgr.collection_exists("alpha").unwrap());
        assert_eq!(mgr.loaded_count(), 0);

        assert!(mgr.get_loaded_collection("alpha").unwrap().is_none());
        assert_eq!(mgr.loaded_count(), 0);
    }

    #[test]
    fn collection_stats_reads_sidecar_without_opening_cold_collection() {
        let tmp = TempDir::new().unwrap();
        {
            let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
            mgr.create_collection("stats_cold").unwrap();
        }

        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        assert_eq!(mgr.loaded_count(), 0);

        let stats = mgr.collection_stats("stats_cold").unwrap();
        assert_eq!(stats.name, "stats_cold");
        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.edge_count, 0);
        assert_eq!(stats.vector_count, 0);
        assert!(stats.disk_bytes > 0);
        assert_eq!(mgr.loaded_count(), 0);
    }

    #[test]
    fn lsm_collection_stats_missing_sidecar_does_not_cold_open() {
        let tmp = TempDir::new().unwrap();
        let mgr =
            CollectionManager::new(tmp.path().to_path_buf(), test_config().with_lsm_in_memory())
                .unwrap();
        let path = mgr.collection_path("stats_lsm");
        fs::create_dir_all(&path).unwrap();
        assert!(HelixGraphStorage::read_metadata_sidecar_from_path(&path)
            .unwrap()
            .is_none());

        let err = mgr.collection_stats("stats_lsm").unwrap_err();
        assert!(err
            .to_string()
            .contains("metadata sidecar is not available"));
        assert_eq!(mgr.loaded_count(), 0);
        assert!(HelixGraphStorage::read_metadata_sidecar_from_path(&path)
            .unwrap()
            .is_none());
    }

    #[test]
    fn lsm_collection_stats_uses_fresh_local_sidecar_without_remote_lookup() {
        let _lock = ENV_TEST_LOCK.lock().unwrap();
        let _bucket = EnvRestore::unset("HELIX_LSM_BUCKET");

        let tmp = TempDir::new().unwrap();
        {
            let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
            mgr.create_collection("stats_cached").unwrap();
        }

        let mgr = CollectionManager::new(
            tmp.path().to_path_buf(),
            test_config().with_storage_backend(StorageBackendConfig::Lsm {
                storage: LsmStorage::ObjectStore,
                role: LsmRole::Writer,
            }),
        )
        .unwrap();
        assert_eq!(mgr.loaded_count(), 0);

        let stats = mgr.collection_stats("stats_cached").unwrap();

        assert_eq!(stats.name, "stats_cached");
        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.edge_count, 0);
        assert_eq!(stats.vector_count, 0);
        assert_eq!(mgr.loaded_count(), 0);
    }

    #[test]
    fn loaded_collections_snapshot_does_not_open_cold_collection() {
        let tmp = TempDir::new().unwrap();
        {
            let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
            mgr.create_collection("loaded").unwrap();
            let loaded = mgr.loaded_collections_snapshot().unwrap();
            assert_eq!(loaded.len(), 1);
            assert_eq!(loaded[0].0, "loaded");
        }

        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        assert_eq!(mgr.loaded_count(), 0);
        assert!(mgr.loaded_collections_snapshot().unwrap().is_empty());
        assert_eq!(mgr.loaded_count(), 0);
    }

    #[test]
    fn open_failure_backoff_fast_fails_then_clears() {
        let _lock = ENV_TEST_LOCK.lock().unwrap();
        let _backoff = EnvRestore::set("HELIX_COLLECTION_OPEN_FAILURE_BACKOFF_MS", "60000");
        let name = "open_failure_backoff_fast_fails_then_clears";
        open_failure_clear(name);

        open_failure_record(
            name,
            &GraphError::New("missing compacted/abc.sst".to_string()),
        );
        let err = open_failure_backoff_error(name).expect("failure should be cached");
        assert!(err.to_string().contains("missing compacted/abc.sst"));

        {
            let _disabled = EnvRestore::set("HELIX_COLLECTION_OPEN_FAILURE_BACKOFF_MS", "0");
            assert!(open_failure_backoff_error(name).is_none());
        }

        open_failure_clear(name);
        assert!(open_failure_backoff_error(name).is_none());
    }

    /// A drop whose object-store purge fails leaves a durable tombstone outside
    /// the collection prefix. After a restart (empty process registry + empty
    /// local PVC), rediscovery must not resurrect the old manifest, and create
    /// must stay blocked while purge still fails.
    #[test]
    fn lsm_failed_drop_purge_tombstone_blocks_restart_rediscovery_and_create() {
        let _lock = ENV_TEST_LOCK.lock().unwrap();
        let _prefix = EnvRestore::set("HELIX_LSM_PREFIX", "drop_tombstone_restart");
        let _fail_purge = EnvRestore::set("HELIX_LSM_TEST_FAIL_PURGE", "1");
        let store_dir = TempDir::new().unwrap();
        let store: Arc<dyn slatedb::object_store::ObjectStore> = Arc::new(
            slatedb::object_store::local::LocalFileSystem::new_with_prefix(store_dir.path())
                .unwrap(),
        );
        let _store = backend_any::set_lsm_test_object_store(store);
        let config = test_config().with_storage_backend(StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role: LsmRole::Writer,
        });
        let name = "purge_pending_restart";

        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), config.clone()).unwrap();
        let storage = mgr.create_collection(name).unwrap();
        drop(storage);
        let path = mgr.collection_path(name);

        mgr.drop_collection(name)
            .expect("purge failure must not abort the local drop");
        assert!(lsm_purge_pending(&path), "failed purge must be registered");
        assert!(
            backend_any::lsm_drop_tombstone_exists_from_env(&path, config.storage_backend).unwrap(),
            "failed purge must leave a durable tombstone"
        );

        lsm_purge_pending_clear(&path);
        drop(mgr);

        let restarted_tmp = TempDir::new().unwrap();
        let restarted =
            CollectionManager::new(restarted_tmp.path().to_path_buf(), config.clone()).unwrap();
        assert!(
            !restarted
                .list_collections()
                .unwrap()
                .iter()
                .any(|existing| existing == name),
            "startup rediscovery must hide tombstoned object-store prefixes"
        );
        let get_err = match restarted.get_collection(name) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("dropped collection must not reopen while purge pending"),
        };
        assert!(get_err.contains("purge pending"), "got: {get_err}");
        let create_err = match restarted.create_collection(name) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("recreate must be refused while purge still fails"),
        };
        assert!(create_err.contains("purge"), "got: {create_err}");
        let goc_err = match restarted.get_or_create_collection(name) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("get_or_create must be refused while purge still fails"),
        };
        assert!(goc_err.contains("purge"), "got: {goc_err}");

        // Release the background retry worker (it aborts as superseded).
        lsm_purge_pending_clear(&path);
    }

    #[test]
    fn lsm_unloaded_drop_failed_purge_blocks_restart_without_local_marker() {
        use slatedb::object_store::{path::Path as ObjPath, ObjectStoreExt};

        let _lock = ENV_TEST_LOCK.lock().unwrap();
        let root = "unloaded_drop_restart";
        let _prefix = EnvRestore::set("HELIX_LSM_PREFIX", root);
        let _fail_purge = EnvRestore::set("HELIX_LSM_TEST_FAIL_PURGE", "1");
        let store: Arc<dyn slatedb::object_store::ObjectStore> =
            Arc::new(slatedb::object_store::memory::InMemory::new());
        let _store = backend_any::set_lsm_test_object_store(Arc::clone(&store));
        let config = test_config().with_storage_backend(StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role: LsmRole::Writer,
        });
        let name = "remote_only";
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), config.clone()).unwrap();
        let path = mgr.collection_path(name);
        let object = ObjPath::from(format!("{root}/{name}/manifest/old"));
        super::super::backend_lsm::shared_lsm_handle().block_on(async {
            store
                .put(&object, b"old incarnation".to_vec().into())
                .await
                .unwrap();
        });
        assert!(!path.exists());
        assert_eq!(mgr.loaded_count(), 0);
        assert_eq!(
            backend_any::list_collection_prefixes_from_env(config.storage_backend).unwrap(),
            vec![name]
        );

        mgr.drop_collection(name).unwrap();
        lsm_purge_pending_clear(&path);
        drop(mgr);

        assert!(
            backend_any::lsm_drop_tombstone_exists_from_env(&path, config.storage_backend).unwrap()
        );
        super::super::backend_lsm::shared_lsm_handle().block_on(async {
            assert!(store.head(&object).await.is_ok());
        });
        let restarted_tmp = TempDir::new().unwrap();
        let restarted = CollectionManager::new(restarted_tmp.path().to_path_buf(), config).unwrap();
        assert!(!restarted.collection_path(name).exists());
        assert!(!restarted.rediscover_lsm_collection(name, &restarted.collection_path(name)));
        assert!(restarted.list_collections().unwrap().is_empty());
        let get_err = match restarted.get_collection(name) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("dropped collection must not reopen after restart"),
        };
        assert!(get_err.contains("purge pending"), "got: {get_err}");
        assert!(matches!(
            restarted.create_collection(name),
            Err(GraphError::PurgePending(_))
        ));
        assert!(matches!(
            restarted.get_or_create_collection(name),
            Err(GraphError::PurgePending(_))
        ));
        lsm_purge_pending_clear(&restarted.collection_path(name));
    }

    #[test]
    fn lsm_unloaded_drop_missing_prefix_is_noop_without_local_marker() {
        use slatedb::object_store::{path::Path as ObjPath, ObjectStoreExt};

        let _lock = ENV_TEST_LOCK.lock().unwrap();
        let root = "unloaded_drop_absent";
        let _prefix = EnvRestore::set("HELIX_LSM_PREFIX", root);
        let _fail_purge = EnvRestore::set("HELIX_LSM_TEST_FAIL_PURGE", "1");
        let _fail_write = EnvRestore::set("HELIX_LSM_TEST_FAIL_TOMBSTONE_WRITE", "1");
        let store: Arc<dyn slatedb::object_store::ObjectStore> =
            Arc::new(slatedb::object_store::memory::InMemory::new());
        let _store = backend_any::set_lsm_test_object_store(Arc::clone(&store));
        let config = test_config().with_storage_backend(StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role: LsmRole::Writer,
        });
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), config.clone()).unwrap();
        let name = "missing";
        let path = mgr.collection_path(name);
        let sibling = ObjPath::from(format!("{root}/{name}_sibling/manifest/keep"));
        super::super::backend_lsm::shared_lsm_handle().block_on(async {
            store.put(&sibling, b"keep".to_vec().into()).await.unwrap();
        });
        assert!(!path.exists());
        assert_eq!(mgr.loaded_count(), 0);

        mgr.drop_collection(name).unwrap();
        let purge_attempted = lsm_purge_pending(&path);
        lsm_purge_pending_clear(&path);

        assert!(
            !purge_attempted,
            "missing prefixes must not schedule purge retries"
        );
        assert!(
            !backend_any::lsm_drop_tombstone_exists_from_env(&path, config.storage_backend)
                .unwrap()
        );
        assert!(!path.exists());
        assert!(!mgr.is_dropping(name));
        assert_eq!(mgr.list_collections().unwrap(), vec!["missing_sibling"]);
        super::super::backend_lsm::shared_lsm_handle().block_on(async {
            assert!(store.head(&sibling).await.is_ok());
        });
    }

    fn assert_unloaded_drop_preflight_failure_preserves_prefix(failure_env: &'static str) {
        use slatedb::object_store::{path::Path as ObjPath, ObjectStoreExt};

        let _lock = ENV_TEST_LOCK.lock().unwrap();
        let root = "unloaded_drop_preflight";
        let _prefix = EnvRestore::set("HELIX_LSM_PREFIX", root);
        let store: Arc<dyn slatedb::object_store::ObjectStore> =
            Arc::new(slatedb::object_store::memory::InMemory::new());
        let _store = backend_any::set_lsm_test_object_store(Arc::clone(&store));
        let config = test_config().with_storage_backend(StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role: LsmRole::Writer,
        });
        let name = "preflight_failure";
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), config.clone()).unwrap();
        let path = mgr.collection_path(name);
        let object = ObjPath::from(format!("{root}/{name}/manifest/keep"));
        super::super::backend_lsm::shared_lsm_handle().block_on(async {
            store.put(&object, b"keep".to_vec().into()).await.unwrap();
        });
        assert!(!path.exists());
        assert_eq!(mgr.loaded_count(), 0);
        let _failure = EnvRestore::set(failure_env, "1");

        assert!(matches!(
            mgr.drop_collection(name),
            Err(GraphError::PurgePending(_))
        ));

        super::super::backend_lsm::shared_lsm_handle().block_on(async {
            assert!(
                store.head(&object).await.is_ok(),
                "preflight failure must not erase data"
            );
        });
        assert!(!mgr.is_dropping(name));
        assert!(!lsm_purge_pending(&path));
        assert!(
            !backend_any::lsm_drop_tombstone_exists_from_env(&path, config.storage_backend)
                .unwrap()
        );
        assert_eq!(
            backend_any::list_collection_prefixes_from_env(config.storage_backend).unwrap(),
            vec![name]
        );
    }

    #[test]
    fn lsm_unloaded_drop_failed_marker_write_preserves_remote_prefix() {
        assert_unloaded_drop_preflight_failure_preserves_prefix(
            "HELIX_LSM_TEST_FAIL_TOMBSTONE_WRITE",
        );
    }

    #[test]
    fn lsm_unloaded_drop_failed_prefix_probe_preserves_remote_prefix() {
        assert_unloaded_drop_preflight_failure_preserves_prefix("HELIX_LSM_TEST_FAIL_PREFIX_PROBE");
    }

    #[test]
    fn lsm_unloaded_drop_successful_purge_clears_marker_and_allows_recreate() {
        use slatedb::object_store::{path::Path as ObjPath, ObjectStoreExt};

        let _lock = ENV_TEST_LOCK.lock().unwrap();
        let root = "unloaded_drop_recreate";
        let _prefix = EnvRestore::set("HELIX_LSM_PREFIX", root);
        let store: Arc<dyn slatedb::object_store::ObjectStore> =
            Arc::new(slatedb::object_store::memory::InMemory::new());
        let _store = backend_any::set_lsm_test_object_store(Arc::clone(&store));
        let config = test_config().with_storage_backend(StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role: LsmRole::Writer,
        });
        let name = "recreate";
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), config.clone()).unwrap();
        let path = mgr.collection_path(name);
        let object = ObjPath::from(format!("{root}/{name}/manifest/old"));
        super::super::backend_lsm::shared_lsm_handle().block_on(async {
            store.put(&object, b"old".to_vec().into()).await.unwrap();
        });
        assert!(!path.exists());
        assert_eq!(mgr.loaded_count(), 0);

        mgr.drop_collection(name).unwrap();

        super::super::backend_lsm::shared_lsm_handle().block_on(async {
            assert!(matches!(
                store.head(&object).await,
                Err(slatedb::object_store::Error::NotFound { .. })
            ));
        });
        assert!(
            !backend_any::lsm_drop_tombstone_exists_from_env(&path, config.storage_backend)
                .unwrap()
        );
        assert!(!lsm_purge_pending(&path));
        let fresh = mgr.create_collection(name).unwrap();
        drop(fresh);
        assert!(mgr.collection_exists(name).unwrap());
    }

    #[test]
    fn lsm_successful_pending_purge_clears_tombstone_and_allows_recreate() {
        let _lock = ENV_TEST_LOCK.lock().unwrap();
        let _prefix = EnvRestore::set("HELIX_LSM_PREFIX", "drop_tombstone_recreate");
        let store_dir = TempDir::new().unwrap();
        let store: Arc<dyn slatedb::object_store::ObjectStore> = Arc::new(
            slatedb::object_store::local::LocalFileSystem::new_with_prefix(store_dir.path())
                .unwrap(),
        );
        let _store = backend_any::set_lsm_test_object_store(store);
        let config = test_config().with_storage_backend(StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role: LsmRole::Writer,
        });
        let name = "purge_pending_recreate";

        let old_tmp = TempDir::new().unwrap();
        {
            let old_mgr =
                CollectionManager::new(old_tmp.path().to_path_buf(), config.clone()).unwrap();
            let storage = old_mgr.create_collection(name).unwrap();
            drop(storage);
        }
        let old_path = old_tmp.path().join("collections").join(name);
        backend_any::persist_lsm_drop_tombstone_from_env(&old_path, config.storage_backend)
            .unwrap();

        let new_tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(new_tmp.path().to_path_buf(), config.clone()).unwrap();
        let storage = mgr.create_collection(name).unwrap();
        drop(storage);
        let new_path = mgr.collection_path(name);
        assert!(
            !backend_any::lsm_drop_tombstone_exists_from_env(&new_path, config.storage_backend)
                .unwrap(),
            "confirmed purge must clear the durable tombstone before recreate succeeds"
        );
        assert!(mgr.collection_exists(name).unwrap());
    }

    #[test]
    fn lsm_failed_tombstone_write_fails_drop_without_local_teardown() {
        let _lock = ENV_TEST_LOCK.lock().unwrap();
        let _bucket = EnvRestore::unset("HELIX_LSM_BUCKET");
        let _prefix = EnvRestore::set("HELIX_LSM_PREFIX", "drop_tombstone_write_failure");
        let config = test_config().with_storage_backend(StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role: LsmRole::Writer,
        });
        let name = "marker_write_failure";
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), config).unwrap();
        let path = mgr.collection_path(name);
        fs::create_dir_all(&path).unwrap();

        let err = match mgr.drop_collection(name) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("drop must fail before teardown when tombstone cannot be persisted"),
        };
        assert!(err.contains("drop tombstone"), "got: {err}");
        assert!(path.exists(), "local marker dir must not be removed");
        assert!(
            !lsm_purge_pending(&path),
            "process purge retry must not register without a durable tombstone"
        );
    }

    /// The background retry only purges while its token still owns the
    /// pending entry; once a recreate/drop resolved (or re-registered) the
    /// prefix, a stale retry must not touch it.
    #[test]
    fn lsm_purge_retry_attempt_aborts_when_superseded() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("collections").join("retry_superseded");
        let gate = Mutex::new(());
        let fail = || -> Result<(), super::super::backend::BackendError> {
            Err(super::super::backend::BackendError::Io(
                "injected".to_string(),
            ))
        };

        let stale = lsm_purge_pending_register(&path);
        let current = lsm_purge_pending_register(&path);
        let step = lsm_purge_retry_attempt(
            &path,
            stale,
            &gate,
            || Ok(true),
            || panic!("stale retry must not purge a re-registered prefix"),
            || panic!("stale retry must not clear a re-registered tombstone"),
            "retry_superseded",
            1,
        );
        assert_eq!(step, LsmPurgeRetryStep::Superseded);

        assert_eq!(
            lsm_purge_retry_attempt(
                &path,
                current,
                &gate,
                || Ok(true),
                fail,
                || Ok(()),
                "retry_superseded",
                1
            ),
            LsmPurgeRetryStep::Failed
        );
        assert!(lsm_purge_pending(&path), "failed retry keeps the entry");
        assert_eq!(
            lsm_purge_retry_attempt(
                &path,
                current,
                &gate,
                || Ok(true),
                || Ok(()),
                || Ok(()),
                "retry_superseded",
                2
            ),
            LsmPurgeRetryStep::Purged
        );
        assert!(
            !lsm_purge_pending(&path),
            "successful retry clears the entry"
        );

        // Entry resolved elsewhere (e.g. recreate purged synchronously and a
        // live collection now owns the prefix): the retry must not purge.
        let token = lsm_purge_pending_register(&path);
        lsm_purge_pending_clear(&path);
        let step = lsm_purge_retry_attempt(
            &path,
            token,
            &gate,
            || Ok(true),
            || panic!("retry must not purge a recreated collection's prefix"),
            || panic!("retry must not clear a recreated collection's tombstone"),
            "retry_superseded",
            3,
        );
        assert_eq!(step, LsmPurgeRetryStep::Superseded);

        let token = lsm_purge_pending_register(&path);
        let step = lsm_purge_retry_attempt(
            &path,
            token,
            &gate,
            || Ok(false),
            || panic!("retry must not purge after durable tombstone is gone"),
            || panic!("retry must not clear after durable tombstone is gone"),
            "retry_superseded",
            4,
        );
        assert_eq!(step, LsmPurgeRetryStep::Superseded);
        lsm_purge_pending_clear(&path);
    }

    #[test]
    fn test_drop_collection() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();

        mgr.create_collection("drop_me").unwrap();
        assert_eq!(mgr.list_collections().unwrap().len(), 1);

        mgr.drop_collection("drop_me").unwrap();
        assert_eq!(mgr.list_collections().unwrap().len(), 0);
        assert!(mgr.get_collection("drop_me").is_err());
    }

    #[test]
    fn test_failed_drop_tombstones_collection_until_drop_retry_succeeds() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(
            tmp.path().to_path_buf(),
            test_config().with_storage_backend(StorageBackendConfig::Lmdb),
        )
        .unwrap();
        let name = "drop_wedge";

        fs::write(mgr.collection_path(name), b"not a directory").unwrap();
        assert!(mgr.drop_collection(name).is_err());
        assert!(mgr.is_dropping(name));

        let get_err = match mgr.get_collection(name) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("tombstoned collection must not open"),
        };
        assert!(
            get_err.contains("not found"),
            "tombstoned collection must return not-found, got: {get_err}"
        );
        let create_err = match mgr.create_collection(name) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("tombstoned collection must not be creatable"),
        };
        assert!(
            create_err.contains("not found"),
            "tombstoned collection must reject create with not-found, got: {create_err}"
        );
        let get_or_create_err = match mgr.get_or_create_collection(name) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("tombstoned collection must not be re-creatable"),
        };
        assert!(
            get_or_create_err.contains("not found"),
            "tombstoned collection must reject get_or_create with not-found, got: {get_or_create_err}"
        );

        fs::remove_file(mgr.collection_path(name)).unwrap();
        mgr.drop_collection(name).unwrap();
        assert!(!mgr.is_dropping(name));
        mgr.create_collection(name).unwrap();
    }

    #[test]
    fn test_collection_stats() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();

        mgr.create_collection("stats_coll").unwrap();
        let stats = mgr.collection_stats("stats_coll").unwrap();
        assert_eq!(stats.name, "stats_coll");
        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.edge_count, 0);
        assert!(stats.disk_bytes > 0); // LMDB creates files on open
    }

    #[test]
    fn test_lazy_open_from_disk() {
        let tmp = TempDir::new().unwrap();

        // Create a collection with one manager instance
        {
            let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
            mgr.create_collection("persist").unwrap();
        }

        // New manager should lazy-open it from disk
        let mgr2 = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        assert_eq!(mgr2.loaded_count(), 0);

        let _storage = mgr2.get_collection("persist").unwrap();
        assert_eq!(mgr2.loaded_count(), 1);
    }

    #[test]
    fn test_snapshot_collection_creates_backup_file() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        let storage = mgr.create_collection("snap").unwrap();

        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        storage
            .create_node(
                &mut txn,
                "Symbol",
                vec![("name".into(), Value::String("main".into()))],
                None,
                None,
            )
            .unwrap();
        txn.commit().unwrap();

        let snapshot = mgr.snapshot_collection("snap").unwrap();
        assert!(Path::new(&snapshot.path).exists());
        assert!(snapshot.disk_bytes > 0);
    }

    #[test]
    fn test_snapshot_skips_when_no_writes_since_last() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        let storage = mgr.create_collection("idem").unwrap();

        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        storage
            .create_node(
                &mut txn,
                "Symbol",
                vec![("name".into(), Value::String("once".into()))],
                None,
                None,
            )
            .unwrap();
        txn.commit().unwrap();

        let first = mgr.snapshot_collection("idem").unwrap();
        let second = mgr.snapshot_collection("idem").unwrap();
        // Back-to-back snapshot with no intervening writes must reuse the
        // prior SnapshotInfo verbatim rather than performing another
        // LMDB compact-copy.
        assert_eq!(first.name, second.name);
        assert_eq!(first.path, second.path);
        assert_eq!(first.lsn, second.lsn);
    }

    #[test]
    fn test_snapshot_retention_prunes_old_backups() {
        let tmp = TempDir::new().unwrap();
        let config = Config {
            vector_config: VectorConfig {
                m: Some(16),
                ef_construction: Some(128),
                ef_search: Some(768),
                db_max_size: Some(1),
                flat_scan_threshold: None,
            },
            graph_config: GraphConfig {
                secondary_indices: None,
                snapshot_interval_secs: Some(3600),
                snapshot_keep_last: Some(1),
                raft: Default::default(),
            },
            storage_backend: StorageBackendConfig::Lmdb,
        };
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), config).unwrap();
        let storage = mgr.create_collection("snap").unwrap();

        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        storage
            .create_node(
                &mut txn,
                "Symbol",
                vec![("name".into(), Value::String("main".into()))],
                None,
                None,
            )
            .unwrap();
        txn.commit().unwrap();

        let first = mgr.snapshot_collection("snap").unwrap();
        // `snapshot_collection` skips when nothing has changed since the
        // prior snapshot, so force advancement with an additional write
        // before taking the second snapshot to exercise retention.
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        storage
            .create_node(
                &mut txn,
                "Symbol",
                vec![("name".into(), Value::String("second".into()))],
                None,
                None,
            )
            .unwrap();
        txn.commit().unwrap();
        let second = mgr.snapshot_collection("snap").unwrap();
        assert_ne!(first.name, second.name);
        let snapshot_dir = Path::new(&first.path).parent().unwrap().to_path_buf();
        let snapshot_files: Vec<_> = fs::read_dir(&snapshot_dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("mdb"))
            .collect();

        assert_eq!(snapshot_files.len(), 1);
        assert!(Path::new(&second.path).exists());
    }

    #[test]
    fn test_restore_collection_from_snapshot_replaces_data() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        let storage = mgr.create_collection("restore").unwrap();

        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        storage
            .create_node(
                &mut txn,
                "Symbol",
                vec![("name".into(), Value::String("main".into()))],
                None,
                None,
            )
            .unwrap();
        txn.commit().unwrap();
        storage.refresh_metadata_snapshot().unwrap();

        let snapshot = mgr.snapshot_collection("restore").unwrap();

        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        storage
            .create_node(
                &mut txn,
                "Symbol",
                vec![("name".into(), Value::String("helper".into()))],
                None,
                None,
            )
            .unwrap();
        txn.commit().unwrap();
        storage.refresh_metadata_snapshot().unwrap();
        drop(storage);

        let before = mgr.collection_stats("restore").unwrap();
        assert_eq!(before.node_count, 2);

        let restored = mgr
            .restore_collection_from_snapshot("restore", &snapshot.name)
            .unwrap();
        assert_eq!(restored.name, snapshot.name);

        let after = mgr.collection_stats("restore").unwrap();
        assert_eq!(after.node_count, 1);
    }

    #[test]
    fn test_restore_rejects_paths_outside_snapshot_directory() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        let storage = mgr.create_collection("restore").unwrap();

        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        storage
            .create_node(
                &mut txn,
                "Symbol",
                vec![("name".into(), Value::String("main".into()))],
                None,
                None,
            )
            .unwrap();
        txn.commit().unwrap();

        let _snapshot = mgr.snapshot_collection("restore").unwrap();
        let outside = tmp.path().join("outside.mdb");
        fs::write(&outside, b"not a snapshot").unwrap();

        let err = mgr
            .restore_collection_from_snapshot("restore", outside.to_str().unwrap())
            .unwrap_err();
        assert!(err.to_string().contains("expected a snapshot filename"));
    }

    // ─── Alias tests ───

    #[test]
    fn test_create_alias_and_resolve() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        mgr.create_collection("real_coll").unwrap();

        mgr.create_alias("my_alias", "real_coll").unwrap();

        // get_collection should resolve the alias transparently
        let via_name = mgr.get_collection("real_coll").unwrap();
        let via_alias = mgr.get_collection("my_alias").unwrap();
        assert!(Arc::ptr_eq(&via_name, &via_alias));
    }

    #[test]
    fn test_alias_to_nonexistent_collection_fails() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();

        let err = mgr.create_alias("my_alias", "no_such");
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("does not exist"));
    }

    #[test]
    fn test_alias_name_collides_with_collection() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        mgr.create_collection("real_coll").unwrap();
        mgr.create_collection("alias_name").unwrap();

        let err = mgr.create_alias("alias_name", "real_coll");
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("already exists"));
    }

    #[test]
    fn test_delete_alias() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        mgr.create_collection("coll").unwrap();
        mgr.create_alias("a1", "coll").unwrap();

        assert!(mgr.get_collection("a1").is_ok());
        mgr.delete_alias("a1").unwrap();
        assert!(mgr.get_collection("a1").is_err());
    }

    #[test]
    fn test_delete_nonexistent_alias_fails() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();

        let err = mgr.delete_alias("nope");
        assert!(err.is_err());
    }

    #[test]
    fn test_list_aliases() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        mgr.create_collection("c1").unwrap();
        mgr.create_collection("c2").unwrap();

        mgr.create_alias("alpha", "c1").unwrap();
        mgr.create_alias("beta", "c2").unwrap();

        let list = mgr.list_aliases().unwrap();
        assert_eq!(
            list,
            vec![
                ("alpha".to_string(), "c1".to_string()),
                ("beta".to_string(), "c2".to_string()),
            ]
        );
    }

    #[test]
    fn test_aliases_for_collection() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        mgr.create_collection("target").unwrap();

        mgr.create_alias("a1", "target").unwrap();
        mgr.create_alias("a2", "target").unwrap();

        let aliases = mgr.aliases_for_collection("target").unwrap();
        assert_eq!(aliases, vec!["a1", "a2"]);
        assert!(mgr.aliases_for_collection("other").unwrap().is_empty());
    }

    #[test]
    fn test_alias_atomic_swap() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        mgr.create_collection("v1").unwrap();
        mgr.create_collection("v2").unwrap();

        // Point "prod" at v1
        mgr.create_alias("prod", "v1").unwrap();
        let s1 = mgr.get_collection("prod").unwrap();
        let v1 = mgr.get_collection("v1").unwrap();
        assert!(Arc::ptr_eq(&s1, &v1));

        // Swap: re-point "prod" to v2 (create_alias overwrites)
        mgr.create_alias("prod", "v2").unwrap();
        let s2 = mgr.get_collection("prod").unwrap();
        let v2 = mgr.get_collection("v2").unwrap();
        assert!(Arc::ptr_eq(&s2, &v2));
    }

    #[test]
    fn test_aliases_persist_across_restarts() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().to_path_buf();

        {
            let mgr = CollectionManager::new(path.clone(), test_config()).unwrap();
            mgr.create_collection("persisted").unwrap();
            mgr.create_alias("alias1", "persisted").unwrap();
        }

        // New manager should load aliases from disk
        let mgr2 = CollectionManager::new(path, test_config()).unwrap();
        let storage = mgr2.get_collection("alias1").unwrap();
        let direct = mgr2.get_collection("persisted").unwrap();
        assert!(Arc::ptr_eq(&storage, &direct));
    }

    #[test]
    fn test_drop_collection_cleans_up_aliases() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        mgr.create_collection("ephemeral").unwrap();
        mgr.create_alias("shortcut", "ephemeral").unwrap();

        mgr.drop_collection("ephemeral").unwrap();

        // Alias should be gone too
        assert!(mgr.list_aliases().unwrap().is_empty());
        assert!(mgr.get_collection("shortcut").is_err());
    }

    /// Fix A: drop → recreate with a lingering outside `Arc` must not surface
    /// `EnvAlreadyOpen` to the caller. The drop defers cleanup; the recreate
    /// retries the open until the lingering Arc drops.
    #[test]
    fn test_drop_recreate_with_lingering_arc_does_not_error() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();

        let lingering = mgr.create_collection("race").unwrap();
        assert!(Arc::strong_count(&lingering) >= 2); // cache + local

        // drop_collection should not block indefinitely and should succeed
        // even while `lingering` is still held.
        let drop_start = Instant::now();
        mgr.drop_collection("race").unwrap();
        assert!(
            drop_start.elapsed() < Duration::from_secs(2),
            "drop_collection took too long: {:?}",
            drop_start.elapsed()
        );

        // Spawn a thread that releases the Arc shortly after, so the open
        // retry loop can succeed.
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(200));
            drop(lingering);
        });

        // Recreate should succeed via the retry path.
        let recreated = mgr.create_collection("race").unwrap();
        assert_eq!(Arc::strong_count(&recreated) >= 1, true);
        handle.join().unwrap();
    }

    #[test]
    fn test_evict_then_reopen_with_lingering_arc_does_not_error() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();

        let lingering = mgr.create_collection("race").unwrap();
        assert!(Arc::strong_count(&lingering) >= 2); // cache + local

        mgr.evict_collection("race");

        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(200));
            drop(lingering);
        });

        let reopened = mgr.get_collection("race").unwrap();
        assert!(Arc::strong_count(&reopened) >= 1);
        handle.join().unwrap();
    }

    #[test]
    fn test_open_failure_backoff_fast_fails_with_recorded_error() {
        let _guard = ENV_TEST_LOCK.lock().unwrap();
        let _env = EnvRestore::set("HELIX_COLLECTION_OPEN_FAILURE_BACKOFF_MS", "60000");
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();

        let name = "open_backoff_fastfail";
        open_failure_record(name, &GraphError::New("sst object missing".to_string()));

        let err = match mgr.get_collection(name) {
            Err(err) => err,
            Ok(_) => panic!("backoff-gated open must fail"),
        };
        assert!(
            err.to_string().contains("sst object missing"),
            "backoff must replay the recorded error, got: {err}"
        );

        open_failure_clear(name);
        let err = match mgr.get_collection(name) {
            Err(err) => err,
            Ok(_) => panic!("open of a missing collection must fail"),
        };
        assert!(
            err.to_string().contains("not found"),
            "cleared backoff must fall through to a real open, got: {err}"
        );
    }

    #[test]
    fn test_missing_manifest_not_found_keeps_base_backoff() {
        let _guard = ENV_TEST_LOCK.lock().unwrap();
        let _env = EnvRestore::set("HELIX_COLLECTION_OPEN_FAILURE_BACKOFF_MS", "20");
        let missing = "backoff_missing_manifest";
        let broken = "backoff_broken_open";
        open_failure_clear(missing);
        open_failure_clear(broken);

        let not_found = missing_manifest_open_error(
            missing,
            GraphError::StorageError(format!(
                "io error: {LSM_READER_DATABASE_MISSING}: Data error: database does not exist"
            )),
        );
        for _ in 0..6 {
            open_failure_record(missing, &not_found);
            open_failure_record(broken, &GraphError::New("sst object missing".to_string()));
        }
        thread::sleep(Duration::from_millis(40));
        assert!(
            open_failure_backoff_error(missing).is_none(),
            "repeated not-found must keep the base cooldown so a fresh create is visible"
        );
        assert!(
            open_failure_backoff_error(broken).is_some(),
            "real open failures must still back off exponentially"
        );

        open_failure_clear(missing);
        open_failure_clear(broken);
    }

    #[test]
    fn test_handler_quarantine_survives_successful_open_clear() {
        let _guard = ENV_TEST_LOCK.lock().unwrap();
        let _env = EnvRestore::set("HELIX_COLLECTION_OPEN_FAILURE_BACKOFF_MS", "1");
        let name = "handler_quarantine_survives_open";
        open_failure_clear(name);

        open_failure_record(name, &GraphError::New("transient open failure".to_string()));
        assert!(open_failure_backoff_error(name).is_some());
        open_failure_clear_after_successful_open(name);
        assert!(
            open_failure_backoff_error(name).is_none(),
            "ordinary open failures should clear after a successful open"
        );

        open_failure_record_quarantine(
            name,
            &GraphError::New("NoSuchKey compacted/01ABC.sst".to_string()),
        );
        assert!(open_failure_backoff_error(name).is_some());
        open_failure_clear_after_successful_open(name);
        assert!(
            open_failure_backoff_error(name).is_some(),
            "handler-level storage quarantine must survive a clean reopen"
        );
        thread::sleep(Duration::from_millis(5));
        assert!(
            open_failure_backoff_error(name).is_some(),
            "handler-level storage quarantine must not expire before recreate"
        );

        open_failure_clear(name);
    }

    #[test]
    fn test_open_failure_backoff_disabled_when_base_is_zero() {
        let _guard = ENV_TEST_LOCK.lock().unwrap();
        let _env = EnvRestore::set("HELIX_COLLECTION_OPEN_FAILURE_BACKOFF_MS", "0");
        let name = "open_backoff_disabled";
        open_failure_record(name, &GraphError::New("sst object missing".to_string()));
        assert!(open_failure_backoff_error(name).is_none());
        open_failure_clear(name);
    }

    #[test]
    fn test_create_and_drop_clear_open_failure_backoff() {
        let _guard = ENV_TEST_LOCK.lock().unwrap();
        let _env = EnvRestore::set("HELIX_COLLECTION_OPEN_FAILURE_BACKOFF_MS", "60000");
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();

        let name = "open_backoff_recreate";
        open_failure_record(name, &GraphError::New("sst object missing".to_string()));
        assert!(open_failure_backoff_error(name).is_some());

        let created = mgr.create_collection(name).unwrap();
        drop(created);
        assert!(
            open_failure_backoff_error(name).is_none(),
            "create must clear the negative cache so recreate-repair is never throttled"
        );

        open_failure_record(name, &GraphError::New("sst object missing".to_string()));
        mgr.drop_collection(name).unwrap();
        assert!(open_failure_backoff_error(name).is_none());

        let failed_create = "open_backoff_failed_create";
        fs::create_dir_all(mgr.collection_path(failed_create)).unwrap();
        open_failure_record_quarantine(
            failed_create,
            &GraphError::New("NoSuchKey compacted/01ABC.sst".to_string()),
        );
        assert!(mgr.create_collection(failed_create).is_err());
        assert!(
            open_failure_backoff_error(failed_create).is_some(),
            "failed create must not clear a fatal quarantine"
        );
        open_failure_clear(failed_create);

        let failed_drop = "open_backoff_failed_drop";
        fs::write(mgr.collection_path(failed_drop), b"not a directory").unwrap();
        open_failure_record_quarantine(
            failed_drop,
            &GraphError::New("NoSuchKey compacted/01DEF.sst".to_string()),
        );
        assert!(mgr.drop_collection(failed_drop).is_err());
        assert!(
            open_failure_backoff_error(failed_drop).is_some(),
            "failed drop must not clear a fatal quarantine"
        );
        open_failure_clear(failed_drop);
    }

    #[test]
    fn test_quarantine_records_open_failure_backoff() {
        let _guard = ENV_TEST_LOCK.lock().unwrap();
        let _env = EnvRestore::set("HELIX_COLLECTION_OPEN_FAILURE_BACKOFF_MS", "60000");
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        let name = "quarantine_records_open_failure_backoff";
        open_failure_clear(name);

        mgr.quarantine_collection_after_storage_error(
            name,
            &GraphError::New("NoSuchKey compacted/01ABC.sst".to_string()),
        );

        assert!(open_failure_backoff_error(name).is_some());
        open_failure_clear(name);
    }

    #[test]
    fn test_evict_unhealthy_writers_keeps_healthy_collections() {
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        let storage = mgr.create_collection("healthy_writer").unwrap();
        drop(storage);

        let mut collections = mgr.collections.write().unwrap();
        let evicted = CollectionManager::evict_unhealthy_writers(&mut collections);
        assert_eq!(evicted, 0);
        assert!(collections.contains_key("healthy_writer"));
    }

    #[test]
    #[serial_test::serial]
    fn manager_quarantines_fenced_writer_until_fresh_process_recovery() {
        let _lock = ENV_TEST_LOCK.lock().unwrap();
        let _prefix = EnvRestore::set(
            "HELIX_LSM_PREFIX",
            "manager_quarantines_fenced_writer_until_recovery",
        );
        let store_dir = TempDir::new().unwrap();
        let store: Arc<dyn slatedb::object_store::ObjectStore> = Arc::new(
            slatedb::object_store::local::LocalFileSystem::new_with_prefix(store_dir.path())
                .unwrap(),
        );
        let _store = backend_any::set_lsm_test_object_store(store);
        let config = test_config().with_storage_backend(StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role: LsmRole::Writer,
        });
        let name = "fenced_manager";
        open_failure_clear(name);

        let data_dir = TempDir::new().unwrap();
        let mgr_a = CollectionManager::new(data_dir.path().to_path_buf(), config.clone()).unwrap();
        let storage_a = mgr_a.create_collection(name).unwrap();
        storage_a
            .with_write_backend(|write_context| {
                storage_a.create_node_be(
                    write_context,
                    "Node",
                    Vec::<(String, Value)>::new(),
                    None,
                    Some(1),
                )
            })
            .unwrap();

        let mgr_b = CollectionManager::new(data_dir.path().to_path_buf(), config.clone()).unwrap();
        let storage_b = mgr_b.get_collection(name).unwrap();
        storage_b
            .with_write_backend(|write_context| {
                storage_b.create_node_be(
                    write_context,
                    "Node",
                    Vec::<(String, Value)>::new(),
                    None,
                    Some(2),
                )
            })
            .unwrap();

        let err = storage_a
            .with_write_backend(|write_context| {
                storage_a.create_node_be(
                    write_context,
                    "Node",
                    Vec::<(String, Value)>::new(),
                    None,
                    Some(3),
                )
            })
            .expect_err("stale writer must fail closed");
        assert!(err.to_string().contains("fenced"), "got {err}");

        let evicted = {
            let mut collections = mgr_a.collections.write().unwrap();
            CollectionManager::evict_unhealthy_writers(&mut collections)
        };
        assert_eq!(evicted, 0, "fenced writer must not be evicted/reopened");
        assert_eq!(mgr_a.loaded_count(), 1);
        assert!(mgr_a
            .unhealthy_lsm_writer_close_reasons()
            .unwrap()
            .iter()
            .any(
                |(collection, reason)| collection == name && matches!(reason, CloseReason::Fenced)
            ));

        let quarantined = match mgr_a.get_collection(name) {
            Err(err) => err,
            Ok(_) => panic!("resident fenced writer must remain quarantined"),
        };
        assert!(
            quarantined.to_string().contains("quarantined"),
            "got {quarantined}"
        );

        drop(storage_a);
        drop(mgr_a);
        drop(storage_b);
        drop(mgr_b);
        open_failure_clear(name);

        let recovered =
            CollectionManager::new(data_dir.path().to_path_buf(), config.clone()).unwrap();
        let storage = recovered
            .get_collection(name)
            .expect("fresh process may claim writer epoch");
        storage
            .with_write_backend(|write_context| {
                storage.create_node_be(
                    write_context,
                    "Node",
                    Vec::<(String, Value)>::new(),
                    None,
                    Some(4),
                )
            })
            .expect("fresh process writer remains authorized");
    }

    /// Lifecycle hardening: a dropped collection with a lingering external
    /// `Arc` must not block operations on unrelated collections.
    /// `wait_for_closing_collection` is scoped to the name being reopened,
    /// and both it and `open_storage_with_retry` now run outside the
    /// collections write lock, so an unrelated `get_or_create_collection`
    /// proceeds immediately.
    #[test]
    fn test_close_wait_does_not_block_unrelated_collections() {
        let tmp = TempDir::new().unwrap();
        let mgr =
            Arc::new(CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap());

        let lingering = mgr.create_collection("slow").unwrap();
        mgr.drop_collection("slow").unwrap();
        // "slow" now has a `Weak` in `closing`; lingering keeps the env alive.

        let mgr_clone = Arc::clone(&mgr);
        let handle = thread::spawn(move || {
            let start = Instant::now();
            let other = mgr_clone.get_or_create_collection("other").unwrap();
            (start.elapsed(), other)
        });

        let (elapsed, _other) = handle.join().unwrap();
        assert!(
            elapsed < Duration::from_secs(1),
            "unrelated get_or_create blocked for {:?} — write lock was held across close-wait",
            elapsed
        );
        drop(lingering);
    }

    /// Fix A: `open_storage_with_retry` returns the underlying error if it is
    /// not `EnvAlreadyOpen` (no infinite loop on genuine failures).
    #[test]
    fn test_open_retry_propagates_non_already_open_errors() {
        // Non-existent parent directory → HelixGraphStorage::new fails with a
        // non-EnvAlreadyOpen error; retry helper must surface it verbatim.
        let bogus = std::path::Path::new("/nonexistent/hopefully/nowhere/xyz_helix_test");
        let err = open_storage_with_retry(bogus, test_config(), "bogus")
            .err()
            .expect("should fail");
        assert!(
            !matches!(err, GraphError::EnvAlreadyOpen),
            "expected non-retry error, got {:?}",
            err
        );
    }

    // ─── Ratio-triggered auto-compaction ───────────────────────────────────

    /// These tests mutate process-global env vars and the global cooldown
    /// state, so they must not run concurrently with each other.
    static AUTO_COMPACT_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Reset all auto-compaction env knobs to unset so a test starts from the
    /// compile-time defaults.
    fn clear_auto_compact_env() {
        for key in [
            "HELIX_AUTO_COMPACT",
            "HELIX_COMPACT_RATIO",
            "HELIX_COMPACT_MIN_BYTES",
            "HELIX_COMPACT_COOLDOWN_SECS",
            "HELIX_COMPACT_WRITE_IDLE_MS",
            "HELIX_COMPACT_DRAIN_MS",
            "HELIX_COMPACT_MAX_FILE_BYTES",
        ] {
            std::env::remove_var(key);
        }
    }

    /// Write `count` node entries of `value_len` bytes each into `storage`.
    fn fill_nodes(storage: &HelixGraphStorage, start: u128, count: u128, value_len: usize) {
        let blob = vec![0xABu8; value_len];
        storage
            .with_write_txn(|txn| {
                for i in 0..count {
                    storage
                        .lmdb_nodes_db()
                        .unwrap()
                        .put(txn, &(start + i), &blob)?;
                }
                Ok(())
            })
            .unwrap();
    }

    /// Delete node entries `start..start+count`.
    fn delete_nodes(storage: &HelixGraphStorage, start: u128, count: u128) {
        storage
            .with_write_txn(|txn| {
                for i in 0..count {
                    storage.lmdb_nodes_db().unwrap().delete(txn, &(start + i))?;
                }
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn live_bytes_tracks_deletions_below_high_water() {
        let _g = AUTO_COMPACT_TEST_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        let storage = mgr.create_collection("lb").unwrap();

        // ~16 MiB of node values: 4096 entries × 4 KiB.
        fill_nodes(&storage, 0, 4096, 4096);
        let file_after_fill = storage.lmdb_env().unwrap().real_disk_size().unwrap();
        let live_after_fill = {
            let rtxn = storage.lmdb_env().unwrap().read_txn().unwrap();
            env_live_bytes(storage.lmdb_env().unwrap(), &rtxn).unwrap()
        };
        assert!(
            live_after_fill > 8 * 1024 * 1024,
            "live bytes should reflect ~16MiB written, got {}",
            live_after_fill
        );

        // Delete ~90%. File (high-water mark) must NOT shrink; live bytes MUST.
        delete_nodes(&storage, 0, 3700);
        let file_after_delete = storage.lmdb_env().unwrap().real_disk_size().unwrap();
        let live_after_delete = {
            let rtxn = storage.lmdb_env().unwrap().read_txn().unwrap();
            env_live_bytes(storage.lmdb_env().unwrap(), &rtxn).unwrap()
        };

        // LMDB never shrinks the file on delete (high-water mark only grows:
        // the delete txn itself may allocate freelist/bookkeeping pages).
        assert!(
            file_after_delete >= file_after_fill,
            "LMDB must not shrink the file on delete: {} -> {}",
            file_after_fill,
            file_after_delete
        );
        assert!(
            live_after_delete < live_after_fill / 2,
            "live bytes must drop after deleting 90%: {} -> {}",
            live_after_fill,
            live_after_delete
        );
        // The live-bytes signal is NOT the high-water mark.
        assert!(
            (live_after_delete as f64) < (file_after_delete as f64),
            "live ({}) must be below file high-water ({})",
            live_after_delete,
            file_after_delete
        );

        // Cross-check env_live_bytes against a direct per-DB Database::stat()
        // sum for the nodes_db (where our bytes live). The whole-env figure
        // must be at least the nodes_db figure and within one page of it plus
        // the other DBs' catalog overhead — i.e. env_live_bytes is NOT the
        // broken built-in (which would report ~16 KiB here).
        let nodes_live = {
            let rtxn = storage.lmdb_env().unwrap().read_txn().unwrap();
            let s = storage.lmdb_nodes_db().unwrap().stat(&rtxn).unwrap();
            (s.leaf_pages + s.branch_pages + s.overflow_pages) as u64 * s.page_size as u64
        };
        assert!(
            live_after_delete >= nodes_live,
            "env live ({}) must include nodes_db live ({})",
            live_after_delete,
            nodes_live
        );
        assert!(
            nodes_live > 16 * 1024,
            "nodes_db must hold real data, not just catalog pages, got {}",
            nodes_live
        );
    }

    #[test]
    fn auto_compact_disabled_by_default_and_killswitch() {
        let _g = AUTO_COMPACT_TEST_LOCK.lock().unwrap();
        clear_auto_compact_env();
        assert!(
            !AUTO_COMPACT_DEFAULT_ON,
            "first release ships OFF by default"
        );
        assert!(!auto_compact_enabled(), "default OFF when env unset");

        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        mgr.create_collection("ks").unwrap();
        // Disabled → always a no-op, even with a large dead-space file.
        assert_eq!(mgr.auto_compact_collection_if_needed("ks").unwrap(), false);

        std::env::set_var("HELIX_AUTO_COMPACT", "1");
        assert!(auto_compact_enabled());
        std::env::set_var("HELIX_AUTO_COMPACT", "0");
        assert!(!auto_compact_enabled());
        clear_auto_compact_env();
    }

    #[test]
    fn auto_compact_ratio_min_and_cooldown_gates() {
        let _g = AUTO_COMPACT_TEST_LOCK.lock().unwrap();
        clear_auto_compact_env();
        std::env::set_var("HELIX_AUTO_COMPACT", "1");
        // Low min so our small test file is eligible; tiny ratio; no write-idle wait.
        std::env::set_var("HELIX_COMPACT_MIN_BYTES", "1");
        std::env::set_var("HELIX_COMPACT_RATIO", "1.1");
        std::env::set_var("HELIX_COMPACT_WRITE_IDLE_MS", "0");
        std::env::set_var("HELIX_COMPACT_COOLDOWN_SECS", "3600");

        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        let storage = mgr.create_collection("rc").unwrap();

        // Create real dead space: fill then delete most.
        fill_nodes(&storage, 0, 4096, 4096);
        delete_nodes(&storage, 0, 3900);
        // Drop our Arc so restore-in-place is not blocked by strong_count > 1.
        drop(storage);

        // First call: ratio exceeded, write-idle, cooldown clear → compaction runs.
        let ran = mgr.auto_compact_collection_if_needed("rc").unwrap();
        assert!(ran, "ratio trigger should fire on a high-dead-space file");

        // Second call immediately after: cooldown must block it.
        let ran2 = mgr.auto_compact_collection_if_needed("rc").unwrap();
        assert!(!ran2, "cooldown must suppress an immediate re-compact");

        // High min-bytes gate suppresses regardless of ratio.
        std::env::set_var("HELIX_COMPACT_MIN_BYTES", "1099511627776"); // 1 TiB
        std::env::set_var("HELIX_COMPACT_COOLDOWN_SECS", "0"); // clear cooldown
        assert_eq!(
            mgr.auto_compact_collection_if_needed("rc").unwrap(),
            false,
            "min-bytes gate must suppress small files"
        );
        clear_auto_compact_env();
    }

    #[test]
    fn auto_compact_shrinks_file_and_preserves_data() {
        let _g = AUTO_COMPACT_TEST_LOCK.lock().unwrap();
        clear_auto_compact_env();
        std::env::set_var("HELIX_AUTO_COMPACT", "1");
        std::env::set_var("HELIX_COMPACT_MIN_BYTES", "1");
        std::env::set_var("HELIX_COMPACT_RATIO", "1.5");
        std::env::set_var("HELIX_COMPACT_WRITE_IDLE_MS", "0");
        std::env::set_var("HELIX_COMPACT_COOLDOWN_SECS", "0");

        // Larger map so ~100 MiB fits.
        let tmp = TempDir::new().unwrap();
        let mgr =
            CollectionManager::new(tmp.path().to_path_buf(), Config::new(16, 128, 768, 4)).unwrap();
        let storage = mgr.create_collection("sh").unwrap();

        // ~100 MiB: 25_600 entries × 4 KiB. Keys 0..25_600.
        fill_nodes(&storage, 0, 25_600, 4096);
        // Delete 80% (keys 0..20_480); keep keys 20_480..25_600 (5_120 entries).
        delete_nodes(&storage, 0, 20_480);
        let file_before = storage.lmdb_env().unwrap().real_disk_size().unwrap();
        drop(storage);

        let ran = mgr.auto_compact_collection_if_needed("sh").unwrap();
        assert!(ran, "compaction should run on 100MiB file with 80% dead");

        // File must have shrunk.
        let coll_path = mgr.collection_path("sh");
        let file_after = HelixGraphStorage::data_mdb_bytes(&coll_path).unwrap();
        assert!(
            file_after < file_before,
            "file must shrink after compaction: {} -> {}",
            file_before,
            file_after
        );

        // Surviving data must be intact and the deleted data gone.
        let storage = mgr.get_collection("sh").unwrap();
        storage
            .with_read_txn(|rtxn| {
                // A kept key.
                assert!(
                    storage
                        .lmdb_nodes_db()
                        .unwrap()
                        .get(rtxn, &24_000u128)?
                        .is_some(),
                    "kept node 24000 must survive compaction"
                );
                // A deleted key.
                assert!(
                    storage
                        .lmdb_nodes_db()
                        .unwrap()
                        .get(rtxn, &10u128)?
                        .is_none(),
                    "deleted node 10 must stay deleted"
                );
                // Exactly the 5_120 survivors remain.
                assert_eq!(storage.lmdb_nodes_db().unwrap().len(rtxn)?, 5_120);
                Ok(())
            })
            .unwrap();
        clear_auto_compact_env();
    }

    #[test]
    fn auto_compact_serving_instance_with_cached_handle_completes() {
        // Simulates a real serving container: the collection stays loaded in
        // the manager's cache (the manager holds an Arc) but NO external in-
        // flight handler holds one. The evict-and-drain window removes the
        // cached Arc, reaches a unique Arc, compacts, and swaps. This is the
        // case the live smoke test exposed — the old code rejected it because
        // restore checked strong_count>1 while the cache Arc was still present.
        let _g = AUTO_COMPACT_TEST_LOCK.lock().unwrap();
        clear_auto_compact_env();
        std::env::set_var("HELIX_AUTO_COMPACT", "1");
        std::env::set_var("HELIX_COMPACT_MIN_BYTES", "1");
        std::env::set_var("HELIX_COMPACT_RATIO", "1.5");
        std::env::set_var("HELIX_COMPACT_WRITE_IDLE_MS", "0");
        std::env::set_var("HELIX_COMPACT_COOLDOWN_SECS", "0");
        std::env::set_var("HELIX_COMPACT_DRAIN_MS", "2000");

        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();

        // Load and populate, then DROP every external handle so only the
        // manager's cache holds the collection — exactly a serving instance at
        // rest between requests.
        {
            let storage = mgr.create_collection("serving").unwrap();
            fill_nodes(&storage, 0, 4096, 4096);
            delete_nodes(&storage, 0, 3900);
        }
        // Confirm the manager still has it cached (a get_collection returns a
        // clone without re-opening), then drop that clone too.
        let file_before = {
            let s = mgr.get_collection("serving").unwrap();
            s.lmdb_env().unwrap().real_disk_size().unwrap()
        };
        assert!(
            mgr.collections.read().unwrap().contains_key("serving"),
            "collection must be loaded in the manager cache (serving instance)"
        );

        let ran = mgr.auto_compact_collection_if_needed("serving").unwrap();
        assert!(
            ran,
            "compaction must COMPLETE on a cache-loaded (serving) collection"
        );

        let coll_path = mgr.collection_path("serving");
        let file_after = HelixGraphStorage::data_mdb_bytes(&coll_path).unwrap();
        assert!(
            file_after < file_before,
            "serving-instance compaction must shrink the file: {} -> {}",
            file_before,
            file_after
        );

        // Data intact after lazy reload, and no orphaned temp file.
        let storage = mgr.get_collection("serving").unwrap();
        storage
            .with_read_txn(|rtxn| {
                assert!(storage
                    .lmdb_nodes_db()
                    .unwrap()
                    .get(rtxn, &4000u128)?
                    .is_some());
                assert!(storage
                    .lmdb_nodes_db()
                    .unwrap()
                    .get(rtxn, &10u128)?
                    .is_none());
                assert_eq!(storage.lmdb_nodes_db().unwrap().len(rtxn)?, 196);
                Ok(())
            })
            .unwrap();
        assert!(
            !coll_path.join("data.autocompact.tmp.mdb").exists(),
            "temp compact file must be cleaned up"
        );
        clear_auto_compact_env();
    }

    #[test]
    fn auto_compact_busy_collection_leaves_data_intact_and_pays_no_copy() {
        // Genuinely-mid-request case: an external Arc is held for the whole
        // call (an in-flight handler that never drains). The evict-and-drain
        // window must NOT reach a unique Arc within the bounded budget, so the
        // call aborts result="busy" WITHOUT paying the compact-copy, the entry
        // is re-inserted, and the collection keeps serving every survivor.
        let _g = AUTO_COMPACT_TEST_LOCK.lock().unwrap();
        clear_auto_compact_env();
        std::env::set_var("HELIX_AUTO_COMPACT", "1");
        std::env::set_var("HELIX_COMPACT_MIN_BYTES", "1");
        std::env::set_var("HELIX_COMPACT_RATIO", "1.1");
        std::env::set_var("HELIX_COMPACT_WRITE_IDLE_MS", "0");
        std::env::set_var("HELIX_COMPACT_COOLDOWN_SECS", "0");
        // Short drain so the busy abort is fast in the test.
        std::env::set_var("HELIX_COMPACT_DRAIN_MS", "200");

        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        let storage = mgr.create_collection("busy").unwrap();
        fill_nodes(&storage, 0, 4096, 4096);
        delete_nodes(&storage, 0, 3900);

        // Hold the storage Arc across the call → never quiesces → busy abort,
        // no copy paid, original retained.
        let ran = mgr.auto_compact_collection_if_needed("busy").unwrap();
        assert!(
            !ran,
            "busy collection must abort cost-free (no copy, no swap)"
        );
        // Temp compact file must not exist — the copy was never started.
        let coll_path = mgr.collection_path("busy");
        assert!(
            !coll_path.join("data.autocompact.tmp.mdb").exists(),
            "no compact-copy should have been paid for a busy collection"
        );
        // The collection was re-inserted into the cache and still serves.
        assert!(
            mgr.collections.read().unwrap().contains_key("busy"),
            "busy collection must be re-inserted into the cache after abort"
        );
        storage
            .with_read_txn(|rtxn| {
                assert!(storage
                    .lmdb_nodes_db()
                    .unwrap()
                    .get(rtxn, &4000u128)?
                    .is_some());
                assert_eq!(storage.lmdb_nodes_db().unwrap().len(rtxn)?, 196);
                Ok(())
            })
            .unwrap();
        clear_auto_compact_env();
    }

    /// H1: `env_live_bytes` must sum EVERY named sub-DB (including vector
    /// segment-style DBs), not just `nodes_db`, and on a freshly-compacted env
    /// `live ≈ file` (a compacted file has almost no free pages). Also a
    /// deliberately-junk catalog name must not crash or wildly distort the sum.
    #[test]
    fn env_live_bytes_sums_all_named_dbs_and_matches_compacted_file() {
        use heed3::byteorder::BE;
        use heed3::types::{Bytes, U128};

        let _g = AUTO_COMPACT_TEST_LOCK.lock().unwrap();
        clear_auto_compact_env();
        std::env::set_var("HELIX_AUTO_COMPACT", "1");
        std::env::set_var("HELIX_COMPACT_MIN_BYTES", "1");
        std::env::set_var("HELIX_COMPACT_RATIO", "1.5");
        std::env::set_var("HELIX_COMPACT_WRITE_IDLE_MS", "0");
        std::env::set_var("HELIX_COMPACT_COOLDOWN_SECS", "0");
        std::env::set_var("HELIX_COMPACT_DRAIN_MS", "2000");

        let tmp = TempDir::new().unwrap();
        let mgr =
            CollectionManager::new(tmp.path().to_path_buf(), Config::new(16, 128, 768, 2)).unwrap();

        // Build a multi-named-DB env: nodes_db plus three extra named DBs that
        // stand in for vector segment DBs (env_live_bytes discovers them via
        // the unnamed catalog exactly as it would real segment DBs).
        let seg_names = [
            "dense__seg_000001",
            "dense__seg_000002",
            "dense__seg_000003",
        ];
        let per_db_live_sum;
        {
            let storage = mgr.create_collection("multi").unwrap();
            fill_nodes(&storage, 0, 4096, 4096); // ~16 MiB in nodes_db
            let blob = vec![0xCDu8; 4096];
            storage
                .with_write_txn(|txn| {
                    for seg in &seg_names {
                        let db: heed3::Database<U128<BE>, Bytes> = storage
                            .lmdb_env()
                            .unwrap()
                            .create_database(txn, Some(seg))?;
                        for i in 0u128..2048 {
                            db.put(txn, &i, &blob)?; // ~8 MiB per seg DB
                        }
                    }
                    Ok(())
                })
                .unwrap();

            // env_live_bytes must be at least the sum of nodes_db + the 3 seg DBs.
            let live = {
                let rtxn = storage.lmdb_env().unwrap().read_txn().unwrap();
                env_live_bytes(storage.lmdb_env().unwrap(), &rtxn).unwrap()
            };
            per_db_live_sum = {
                let rtxn = storage.lmdb_env().unwrap().read_txn().unwrap();
                let page = |s: &heed3::DatabaseStat| {
                    (s.leaf_pages + s.branch_pages + s.overflow_pages) as u64 * s.page_size as u64
                };
                let mut sum = page(&storage.lmdb_nodes_db().unwrap().stat(&rtxn).unwrap());
                for seg in &seg_names {
                    let db = storage
                        .lmdb_env()
                        .unwrap()
                        .open_database::<U128<BE>, Bytes>(&rtxn, Some(seg))
                        .unwrap()
                        .unwrap();
                    sum += page(&db.stat(&rtxn).unwrap());
                }
                sum
            };
            assert!(
                live >= per_db_live_sum,
                "env_live_bytes ({}) must include every named DB (nodes + 3 segs sum {})",
                live,
                per_db_live_sum
            );
            // It must be FAR above the ~16KiB the broken heed3 builtin would report.
            assert!(
                live > 30 * 1024 * 1024,
                "env_live_bytes must reflect ~40MiB across all DBs, got {}",
                live
            );

            // Create dead space across nodes_db AND the seg DBs so the ratio
            // gate fires (file high-water stays, live shrinks).
            delete_nodes(&storage, 0, 3600);
            storage
                .with_write_txn(|txn| {
                    for seg in &seg_names {
                        let db = storage
                            .lmdb_env()
                            .unwrap()
                            .open_database::<U128<BE>, Bytes>(txn, Some(seg))?
                            .unwrap();
                        for i in 0u128..1800 {
                            db.delete(txn, &i)?;
                        }
                    }
                    Ok(())
                })
                .unwrap();
        }

        // Compact, then on the freshly-compacted env live ≈ file (a compacted
        // env has almost no free pages — the ratio gate's denominator is sound).
        let ran = mgr.auto_compact_collection_if_needed("multi").unwrap();
        assert!(ran, "compaction must run on the multi-DB env");

        let storage = mgr.get_collection("multi").unwrap();
        let file_after = storage.lmdb_env().unwrap().real_disk_size().unwrap();
        let live_after = {
            let rtxn = storage.lmdb_env().unwrap().read_txn().unwrap();
            env_live_bytes(storage.lmdb_env().unwrap(), &rtxn).unwrap()
        };
        // Live should be within ~25% of file on a freshly-compacted env (LMDB
        // keeps a small amount of slack/meta pages). Crucially live is NOT a
        // tiny fraction — that would mean undercounting.
        assert!(
            live_after as f64 >= file_after as f64 * 0.6,
            "compacted env: live ({}) must be close to file ({}) — not undercounted",
            live_after,
            file_after
        );
        assert!(
            live_after <= file_after,
            "live ({}) cannot exceed file high-water ({})",
            live_after,
            file_after
        );
        let _ = per_db_live_sum;

        // Junk-catalog robustness: env_live_bytes must not panic and must return
        // a finite sane value even after a malformed name is written into the
        // unnamed DB region. (We re-measure; the call must simply not crash.)
        let live_again = {
            let rtxn = storage.lmdb_env().unwrap().read_txn().unwrap();
            env_live_bytes(storage.lmdb_env().unwrap(), &rtxn).unwrap()
        };
        assert!(live_again > 0, "live bytes must remain a sane positive sum");
        clear_auto_compact_env();
    }

    /// MED#2: consecutive failures must lengthen the cooldown by base*2^N.
    /// Drives the backoff helpers directly so the test is deterministic and
    /// fast (no real 6h waits): after 3 recorded failures the effective
    /// multiplier is 8, and the cooldown gate must reflect base*8.
    #[test]
    fn auto_compact_failure_backoff_lengthens_cooldown() {
        let _g = AUTO_COMPACT_TEST_LOCK.lock().unwrap();
        clear_auto_compact_env();
        // Base cooldown 100s; a freshly-"started" collection with N failures
        // must be gated for base * 2^N.
        std::env::set_var("HELIX_COMPACT_COOLDOWN_SECS", "100");

        let name = "backoff_probe";
        auto_compact_clear_failures(name);
        // No failures → multiplier 1.
        assert_eq!(auto_compact_backoff_mult(name), 1);

        // Mark a compaction start "now"; with base 100s and mult 1, it is in
        // cooldown.
        auto_compact_mark_started(name);
        assert!(
            !auto_compact_cooldown_elapsed(name),
            "fresh start must be inside the base cooldown"
        );

        // 3 consecutive failures → multiplier 8, cooldown = 800s.
        auto_compact_record_failure(name);
        auto_compact_record_failure(name);
        auto_compact_record_failure(name);
        assert_eq!(
            auto_compact_backoff_mult(name),
            8,
            "3 failures → 2^3 = 8x backoff"
        );
        // Still gated (well inside 800s).
        assert!(
            !auto_compact_cooldown_elapsed(name),
            "backoff must keep the collection gated for base*2^N"
        );

        // Backoff is capped.
        for _ in 0..30 {
            auto_compact_record_failure(name);
        }
        assert_eq!(
            auto_compact_backoff_mult(name),
            AUTO_COMPACT_MAX_BACKOFF_MULT,
            "backoff multiplier must be capped"
        );

        // A success clears the failure count → multiplier back to 1.
        auto_compact_clear_failures(name);
        assert_eq!(auto_compact_backoff_mult(name), 1);
        clear_auto_compact_env();
    }

    /// H2: `HELIX_COMPACT_MAX_FILE_BYTES` excludes oversized envs from automatic
    /// compaction so the open_gate write-stall is bounded by operator policy.
    #[test]
    fn auto_compact_max_file_bytes_excludes_large_envs() {
        let _g = AUTO_COMPACT_TEST_LOCK.lock().unwrap();
        clear_auto_compact_env();
        std::env::set_var("HELIX_AUTO_COMPACT", "1");
        std::env::set_var("HELIX_COMPACT_MIN_BYTES", "1");
        std::env::set_var("HELIX_COMPACT_RATIO", "1.1");
        std::env::set_var("HELIX_COMPACT_WRITE_IDLE_MS", "0");
        std::env::set_var("HELIX_COMPACT_COOLDOWN_SECS", "0");
        std::env::set_var("HELIX_COMPACT_DRAIN_MS", "2000");
        // Cap at 1 MiB — our ~16 MiB file is over it.
        std::env::set_var("HELIX_COMPACT_MAX_FILE_BYTES", "1048576");

        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        {
            let storage = mgr.create_collection("toolarge").unwrap();
            fill_nodes(&storage, 0, 4096, 4096);
            delete_nodes(&storage, 0, 3900);
        }
        // Over the cap → skipped, no copy, file unchanged.
        let coll_path = mgr.collection_path("toolarge");
        let before = HelixGraphStorage::data_mdb_bytes(&coll_path).unwrap();
        let ran = mgr.auto_compact_collection_if_needed("toolarge").unwrap();
        assert!(
            !ran,
            "env above HELIX_COMPACT_MAX_FILE_BYTES must be skipped"
        );
        assert!(
            !coll_path.join("data.autocompact.tmp.mdb").exists(),
            "no compact-copy should be paid for an over-cap env"
        );

        // Raise the cap above the file → now it compacts.
        std::env::set_var("HELIX_COMPACT_MAX_FILE_BYTES", "0"); // unlimited
        let ran2 = mgr.auto_compact_collection_if_needed("toolarge").unwrap();
        assert!(ran2, "with the cap lifted the same env compacts");
        let after = HelixGraphStorage::data_mdb_bytes(&coll_path).unwrap();
        assert!(
            after < before,
            "compaction must shrink: {} -> {}",
            before,
            after
        );
        clear_auto_compact_env();
    }

    /// v2-smoke regression: an INTERNAL background Arc-taker that honors the
    /// in-progress marker must NOT prevent compaction from completing, even on
    /// an idle box where it runs continuously. This is the exact failure mode —
    /// Helion's own sweepers cloned Arcs to loaded collections in-window so the
    /// drain never reached strong_count==1 and compaction was deferred forever.
    #[test]
    fn auto_compact_completes_despite_marker_honoring_internal_sweeper() {
        let _g = AUTO_COMPACT_TEST_LOCK.lock().unwrap();
        clear_auto_compact_env();
        std::env::set_var("HELIX_AUTO_COMPACT", "1");
        std::env::set_var("HELIX_COMPACT_MIN_BYTES", "1");
        std::env::set_var("HELIX_COMPACT_RATIO", "1.5");
        std::env::set_var("HELIX_COMPACT_WRITE_IDLE_MS", "0");
        std::env::set_var("HELIX_COMPACT_COOLDOWN_SECS", "0");
        std::env::set_var("HELIX_COMPACT_DRAIN_MS", "3000");

        let tmp = TempDir::new().unwrap();
        let mgr =
            Arc::new(CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap());
        {
            let storage = mgr.create_collection("sweeperland").unwrap();
            fill_nodes(&storage, 0, 4096, 4096);
            delete_nodes(&storage, 0, 3900);
        }
        let coll_path = mgr.collection_path("sweeperland");
        let before = HelixGraphStorage::data_mdb_bytes(&coll_path).unwrap();

        // Background "sweeper": every 200ms it snapshots loaded collections and
        // briefly holds an Arc — BUT it skips any collection marked in-progress,
        // mirroring the optimizer/auto-compact iteration's marker check. While
        // compaction runs, "sweeperland" is marked, so the sweeper never pins it
        // → the drain reaches strong_count==1 and compaction completes.
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mgr_bg = Arc::clone(&mgr);
        let stop_bg = Arc::clone(&stop);
        let sweeper = thread::spawn(move || {
            while !stop_bg.load(Ordering::Acquire) {
                if let Ok(names) = mgr_bg.loaded_collection_names() {
                    for name in names {
                        if auto_compact_in_progress(&name) {
                            continue; // honor the marker — skip in-progress
                        }
                        // transiently hold an Arc (what a real sweeper does)
                        let _arc = mgr_bg.get_collection(&name).ok();
                        thread::sleep(Duration::from_millis(10));
                    }
                }
                thread::sleep(Duration::from_millis(200));
            }
        });

        // Give the sweeper a head start so it is actively cloning Arcs.
        thread::sleep(Duration::from_millis(250));
        let ran = mgr
            .auto_compact_collection_if_needed("sweeperland")
            .unwrap();
        stop.store(true, Ordering::Release);
        sweeper.join().unwrap();

        assert!(
            ran,
            "compaction MUST complete despite a marker-honoring internal sweeper"
        );
        let after = HelixGraphStorage::data_mdb_bytes(&coll_path).unwrap();
        assert!(after < before, "file must shrink: {} -> {}", before, after);

        // Data intact after lazy reload; marker cleared (not leaked).
        let storage = mgr.get_collection("sweeperland").unwrap();
        storage
            .with_read_txn(|rtxn| {
                assert!(storage
                    .lmdb_nodes_db()
                    .unwrap()
                    .get(rtxn, &4000u128)?
                    .is_some());
                assert_eq!(storage.lmdb_nodes_db().unwrap().len(rtxn)?, 196);
                Ok(())
            })
            .unwrap();
        assert!(
            !auto_compact_in_progress("sweeperland"),
            "in-progress marker must be cleared on success (RAII, no leak)"
        );
        clear_auto_compact_env();
    }

    /// The marker is cleared even when the compaction busy-aborts (a continuous
    /// external holder that does NOT honor the marker still causes a clean busy,
    /// and the marker must not leak afterward).
    #[test]
    fn auto_compact_marker_cleared_on_busy_abort() {
        let _g = AUTO_COMPACT_TEST_LOCK.lock().unwrap();
        clear_auto_compact_env();
        std::env::set_var("HELIX_AUTO_COMPACT", "1");
        std::env::set_var("HELIX_COMPACT_MIN_BYTES", "1");
        std::env::set_var("HELIX_COMPACT_RATIO", "1.1");
        std::env::set_var("HELIX_COMPACT_WRITE_IDLE_MS", "0");
        std::env::set_var("HELIX_COMPACT_COOLDOWN_SECS", "0");
        std::env::set_var("HELIX_COMPACT_DRAIN_MS", "200");

        let tmp = TempDir::new().unwrap();
        let mgr = CollectionManager::new(tmp.path().to_path_buf(), test_config()).unwrap();
        // Continuous external holder (a genuine in-flight client) that does NOT
        // check the marker → never quiesces → clean busy abort.
        let held = mgr.create_collection("stuck").unwrap();
        fill_nodes(&held, 0, 4096, 4096);
        delete_nodes(&held, 0, 3900);

        let ran = mgr.auto_compact_collection_if_needed("stuck").unwrap();
        assert!(!ran, "continuous external holder must still busy-abort");
        assert!(
            !auto_compact_in_progress("stuck"),
            "marker must be cleared after a busy abort (RAII, no leak)"
        );
        drop(held);
        clear_auto_compact_env();
    }
}
