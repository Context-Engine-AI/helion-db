use crate::{
    helix_engine::{
        graph_core::config::{Config, DEFAULT_DB_INITIAL_MAP_MB},
        storage_core::{
            metadata::{
                counter_label, decode_lsm_counter_value, encode_lsm_counter_value, lsm_counter_key,
                MetadataCounter, PayloadIndexSchema, StorageMetadata, StorageMetadataSidecar,
                StorageStats, STORAGE_METADATA_SIDECAR_FILE,
            },
            storage_methods::StorageMethods,
            wal::{SyncPolicy, WalOp, WalWriter},
        },
        types::{graph_error_from_backend_error, GraphError},
        vector_core::{
            named_vectors::{
                dead_segment_names_enabled, DenseReadTxnProvider, NamedVectorConfig,
                NamedVectorManager, DEAD_SEGMENT_KEY_PREFIX,
            },
            vector_core::{HNSWConfig, HnswOverrides, VectorCore},
        },
    },
    protocol::{
        filterable::Filterable,
        items::{v6_uuid, SerializedEdge, SerializedNode},
        label_hash::hash_label,
    },
    protocol::{
        items::{Edge, Node},
        value::Value,
    },
};

use heed3::byteorder::BE;
use heed3::{
    types::*, CompactionOption, Database, DatabaseFlags, Env, EnvFlags, EnvOpenOptions, RoTxn,
    RwTxn, WithTls,
};
use parking_lot;
use slatedb::CloseReason;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::hash::Hasher;
use std::ops::{Bound, Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Arc, Condvar, LazyLock, Mutex, MutexGuard, RwLock, Weak,
};
use std::time::{Duration, Instant, SystemTime};
use twox_hash::XxHash64;

use super::backend::{
    BackendError, BackendKind, KeyRange, Namespace, StorageBackend, StorageBackendConfig,
};
use super::backend_any::{
    lsm_collection_prefix_exists_from_env, read_metadata_sidecar_from_lsm_env, AnyBackend, AnyRead,
    AnyWrite, LSM_READER_READONLY,
};
use super::storage_methods::{BasicStorageMethods, DBMethods};
use super::text_walk_cache::{TextWalk, TextWalkKey};
use super::upsert::{EdgeUpsert, NodeUpsert};
use crate::protocol::deterministic_id;

#[path = "payload_marker_backfill.rs"]
pub(crate) mod payload_marker_backfill;

const DEFAULT_LMDB_MAX_DBS: u32 = 65_536;

static TXN_DIAG_SEQ: AtomicU64 = AtomicU64::new(1);
static PAYLOAD_INDEX_JOB_SEQ: AtomicU64 = AtomicU64::new(1);

/// Best-effort metadata refreshes run after the commit has already succeeded.
/// A request-scoped LSM cancellation therefore only means the caller abandoned
/// the follow-up read; other failures can still indicate an unhealthy backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MetadataRefreshErrorLogClass {
    CooperativeCancellation,
    BackendFailure,
}

fn metadata_refresh_error_log_class(error: &GraphError) -> MetadataRefreshErrorLogClass {
    if error.to_string().contains("LSM read request cancelled") {
        MetadataRefreshErrorLogClass::CooperativeCancellation
    } else {
        MetadataRefreshErrorLogClass::BackendFailure
    }
}

/// Guard for the LSM counter recount/repair path: it must run on the LSM
/// writer only (an LMDB collection has no separate counter keys to repair,
/// and a reader replica must never write).
fn ensure_lsm_writer(backend: &AnyBackend) -> Result<(), GraphError> {
    if backend.kind() != BackendKind::Lsm {
        return Err(GraphError::New(
            "recount is only supported on the LSM backend".to_string(),
        ));
    }
    if backend.is_reader_replica() {
        return Err(GraphError::New(LSM_READER_READONLY.to_string()));
    }
    Ok(())
}

/// Before/after values for one counter, produced by [`HelixGraphStorage::recount_lsm_counters`].
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct RecountCounterResult {
    pub before: u64,
    pub after: u64,
}

/// Response payload for the `/v1/collections/recount` admin endpoint.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct RecountCounters {
    pub nodes: RecountCounterResult,
    pub edges: RecountCounterResult,
    pub vectors: RecountCounterResult,
}

/// Per-field result for one [`HelixGraphStorage::gc_payload_index`] run,
/// produced by the `/v1/collections/gc_payload_index` admin endpoint.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct PayloadIndexGcFieldResult {
    pub scanned: usize,
    pub removed: usize,
}

fn diag_warn_ms(env: &str, default: u128) -> u128 {
    std::env::var(env)
        .ok()
        .and_then(|value| value.parse::<u128>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn env_usize_at_least(name: &str, default: usize, min: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= min)
        .unwrap_or(default)
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn env_mb(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
        .saturating_mul(1024)
        .saturating_mul(1024)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .and_then(|value| match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
        .unwrap_or(default)
}

fn payload_index_workers() -> usize {
    env_usize("HELIX_PAYLOAD_INDEX_WORKERS")
        .or_else(|| env_usize("HELIX_INDEX_EXECUTOR_WORKERS"))
        .unwrap_or(2)
        .clamp(1, 32)
}

fn payload_index_queue_cap() -> usize {
    env_usize("HELIX_PAYLOAD_INDEX_QUEUE_CAP")
        .or_else(|| env_usize("HELIX_INDEX_EXECUTOR_QUEUE_CAP"))
        .unwrap_or(8192)
        .max(1)
}

fn payload_index_submit_timeout() -> Duration {
    Duration::from_millis(
        env_usize_at_least("HELIX_PAYLOAD_INDEX_SUBMIT_TIMEOUT_MS", 120_000, 0) as u64,
    )
}

fn payload_index_chunk_size() -> usize {
    env_usize_at_least("HELIX_PAYLOAD_INDEX_CHUNK_SIZE", 2_000, 1)
}

fn payload_index_chunk_pause() -> Duration {
    Duration::from_millis(env_usize_at_least("HELIX_PAYLOAD_INDEX_CHUNK_PAUSE_MS", 0, 0) as u64)
}

fn wal_keep_segments() -> usize {
    env_usize_at_least("HELIX_WAL_KEEP_SEGMENTS", 2, 0)
}

#[inline]
fn observe_write_txn_phase(
    kind: &'static str,
    caller_label: &str,
    phase: &'static str,
    elapsed: std::time::Duration,
) {
    if !crate::telemetry::hot_path_metrics_enabled() {
        return;
    }
    metrics::histogram!(
        "helix_lmdb_write_txn_phase_ms",
        "kind" => kind,
        "phase" => phase,
        "caller" => caller_label.to_string()
    )
    .record(elapsed.as_secs_f64() * 1000.0);
}

struct TxnDiagSpan {
    id: u64,
    kind: &'static str,
    operation: &'static str,
    /// Source location of the caller of `with_write_txn` /
    /// `with_exclusive_write_txn`. Captured via `#[track_caller]` so the
    /// long-txn WARN tells us exactly which call site held the LMDB writer
    /// without needing to add a `site` parameter to every caller.
    caller: &'static std::panic::Location<'static>,
    started: Instant,
    warn_ms: u128,
}

impl TxnDiagSpan {
    fn new_at(
        kind: &'static str,
        operation: &'static str,
        caller: &'static std::panic::Location<'static>,
    ) -> Self {
        let id = TXN_DIAG_SEQ.fetch_add(1, Ordering::Relaxed);
        metrics::gauge!("helix_lmdb_txn_active", "kind" => kind, "operation" => operation)
            .increment(1.0);
        Self {
            id,
            kind,
            operation,
            caller,
            started: Instant::now(),
            warn_ms: diag_warn_ms("HELIX_TXN_HOLD_WARN_MS", 30_000),
        }
    }
}

impl Drop for TxnDiagSpan {
    fn drop(&mut self) {
        let elapsed = self.started.elapsed();
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        metrics::histogram!(
            "helix_lmdb_txn_scope_ms",
            "kind" => self.kind,
            "operation" => self.operation
        )
        .record(elapsed_ms);
        metrics::gauge!(
            "helix_lmdb_txn_active",
            "kind" => self.kind,
            "operation" => self.operation
        )
        .decrement(1.0);
        if elapsed.as_millis() >= self.warn_ms {
            tracing::warn!(
                txn_id = self.id,
                kind = self.kind,
                operation = self.operation,
                caller_file = self.caller.file(),
                caller_line = self.caller.line(),
                elapsed_ms = elapsed.as_millis(),
                "LMDB transaction scope held for a long time"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Global write backpressure based on memory usage
// ---------------------------------------------------------------------------

/// Detect the memory ceiling for this process (cgroup or system RAM).
///
/// Priority:
/// 1. cgroup v2 limit (`/sys/fs/cgroup/memory.max`)
/// 2. cgroup v1 limit (`/sys/fs/cgroup/memory/memory.limit_in_bytes`)
/// 3. `/proc/meminfo` MemTotal
/// 4. Conservative 8 GB default (macOS / unknown)
fn detect_memory_ceiling() -> usize {
    // cgroup v2
    if let Ok(val) = std::fs::read_to_string("/sys/fs/cgroup/memory.max") {
        let val = val.trim();
        if val != "max" {
            if let Ok(bytes) = val.parse::<usize>() {
                return bytes;
            }
        }
    }
    // cgroup v1
    if let Ok(val) = std::fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes") {
        if let Ok(bytes) = val.trim().parse::<usize>() {
            // cgroup v1 returns a very large number when unlimited
            if bytes < 1 << 62 {
                return bytes;
            }
        }
    }
    // Fallback: /proc/meminfo
    if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
        for line in contents.lines() {
            if line.starts_with("MemTotal:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<usize>() {
                        return kb * 1024;
                    }
                }
            }
        }
    }
    // macOS / other
    8 * 1024 * 1024 * 1024
}

/// Read current RSS (resident set size) of this process.
///
/// Returns 0 on platforms where /proc/self/statm is unavailable (macOS),
/// disabling backpressure gracefully.
fn current_rss_bytes() -> usize {
    // Prefer cgroup pressure inside containers. `memory.current` includes
    // clean LMDB mmap pages in file cache, including active_file pages that
    // Kubernetes working-set accounting keeps. Those pages are reclaimable by
    // the kernel; dirty/writeback/shmem are not, so keep those as pressure.
    if let Ok(current_raw) = std::fs::read_to_string("/sys/fs/cgroup/memory.current") {
        if let Ok(current) = current_raw.trim().parse::<usize>() {
            return std::fs::read_to_string("/sys/fs/cgroup/memory.stat")
                .map(|contents| reclaim_aware_cgroup_pressure_bytes(current, &contents))
                .unwrap_or(current);
        }
    }
    if let Ok(current_raw) = std::fs::read_to_string("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    {
        if let Ok(current) = current_raw.trim().parse::<usize>() {
            return std::fs::read_to_string("/sys/fs/cgroup/memory/memory.stat")
                .map(|contents| reclaim_aware_cgroup_pressure_bytes(current, &contents))
                .unwrap_or(current);
        }
    }

    // Linux: /proc/self/statm — field 1 is RSS in pages
    if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
        let parts: Vec<&str> = statm.split_whitespace().collect();
        if parts.len() >= 2 {
            if let Ok(pages) = parts[1].parse::<usize>() {
                return pages * page_size::get();
            }
        }
    }
    0
}

fn cgroup_stat_value(contents: &str, keys: &[&str]) -> usize {
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

fn reclaim_aware_cgroup_pressure_bytes(current: usize, stat_contents: &str) -> usize {
    let inactive_file = cgroup_stat_value(stat_contents, &["inactive_file", "total_inactive_file"]);
    let active_file = cgroup_stat_value(stat_contents, &["active_file", "total_active_file"]);
    let file = match cgroup_stat_value(stat_contents, &["file", "total_cache", "cache"]) {
        0 => active_file.saturating_add(inactive_file),
        value => value,
    };

    if file == 0 {
        return current.saturating_sub(inactive_file);
    }

    let unreclaimable_file = cgroup_stat_value(stat_contents, &["shmem", "total_shmem"])
        .saturating_add(cgroup_stat_value(
            stat_contents,
            &["file_dirty", "total_dirty", "dirty"],
        ))
        .saturating_add(cgroup_stat_value(
            stat_contents,
            &["file_writeback", "total_writeback", "writeback"],
        ));

    current.saturating_sub(file.saturating_sub(unreclaimable_file))
}

/// Memory watermark configuration for write backpressure.
///
/// - Below `low_watermark`: writes proceed at full speed.
/// - Between `low` and `high`: proportional throttle (1–100 ms based on
///   how close RSS is to the high mark).
/// - Above `high_watermark`: writes pause until memory drops below.
///
/// **Writes are never rejected.** The caller blocks (with a timeout) and
/// retries. This matches Milvus's approach and avoids data loss.
///
/// RSS is sampled at most once per `SAMPLE_INTERVAL` to amortize the
/// `/proc/self/statm` syscall across thousands of concurrent writers.
struct MemoryWatermarks {
    ceiling: usize,
    low: usize,  // 70% of ceiling
    high: usize, // 85% of ceiling
    /// Cached RSS and timestamp for amortized sampling.
    /// Layout: upper 44 bits = RSS in KB, lower 20 bits = timestamp in
    /// (epoch_ms >> 4) masked to 20 bits (~16 ms granularity, wraps ~17 min).
    /// This packs into a single AtomicU64 for lock-free reads.
    cached_rss: std::sync::atomic::AtomicU64,
}

/// How often to re-read /proc/self/statm (milliseconds).
const SAMPLE_INTERVAL_MS: u64 = 500;
/// Base cooldown after a failed reader poll-tier promotion before the next
/// attempt. Doubles per consecutive failure (see
/// `reader_poll_promote_cooldown_ms`) so a collection that reliably cannot
/// promote stops burning a thread + S3 open + WARN every 30s.
const READER_POLL_PROMOTE_FAILURE_COOLDOWN_MS: u64 = 30_000;
/// Cap for the exponential promotion-failure backoff
/// (30s → 1m → 2m → 4m → 8m → 15m).
const READER_POLL_PROMOTE_FAILURE_COOLDOWN_MAX_MS: u64 = 900_000;
/// How long the read that starts an in-place poll-tier promotion waits for
/// the promotion's manifest poll. A poll is one LIST, the manifest and the WAL
/// objects written since the last one, tens of milliseconds as a rule; a read
/// is not held longer than this for a slow one.
/// `helix_reader_poll_tier_total{direction="promote_wait_timeout"}` counts
/// the reads that reached it.
pub(crate) const READER_PROMOTE_POLL_WAIT: Duration = Duration::from_millis(250);

impl MemoryWatermarks {
    fn detect() -> Self {
        let ceiling_override = std::env::var("HELIX_MEMORY_LIMIT_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|mb| mb * 1024 * 1024);

        let ceiling = ceiling_override.unwrap_or_else(detect_memory_ceiling);

        let low_pct: usize = std::env::var("HELIX_MEM_LOW_PCT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(70);
        let high_pct: usize = std::env::var("HELIX_MEM_HIGH_PCT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(85);

        let wm = Self {
            ceiling,
            low: ceiling * low_pct / 100,
            high: ceiling * high_pct / 100,
            cached_rss: std::sync::atomic::AtomicU64::new(0),
        };
        tracing::info!(
            ceiling_mb = ceiling / (1024 * 1024),
            low_mb = wm.low / (1024 * 1024),
            high_mb = wm.high / (1024 * 1024),
            "Memory watermarks initialized"
        );
        wm
    }

    /// Get RSS, sampling at most once per SAMPLE_INTERVAL_MS.
    /// Multiple threads may redundantly sample — that's fine, it's idempotent.
    fn sampled_rss(&self) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        use std::time::{SystemTime, UNIX_EPOCH};

        let now_bits = (SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
            / (SAMPLE_INTERVAL_MS.max(1)))
            & 0xFFFFF; // 20-bit epoch slot

        let packed = self.cached_rss.load(Relaxed);
        let cached_slot = packed & 0xFFFFF;
        let cached_rss_kb = (packed >> 20) as usize;

        if cached_slot == now_bits && cached_rss_kb > 0 {
            return cached_rss_kb * 1024;
        }

        // Stale — re-sample
        let rss = current_rss_bytes();
        let rss_kb = (rss / 1024) as u64;
        let new_packed = (rss_kb << 20) | now_bits;
        self.cached_rss.store(new_packed, Relaxed);
        rss
    }

    /// Apply request-path backpressure based on current RSS.
    ///
    /// This intentionally uses short bounded sleeps. Long waits here sit
    /// directly in front of the LMDB writer gate and amplify one pressured
    /// request into a queue-wide latency cliff.
    fn apply_backpressure(&self) -> Result<(), GraphError> {
        let rss = self.sampled_rss();
        if rss == 0 || rss < self.low {
            return Ok(()); // No pressure or unsupported platform
        }

        if rss >= self.high {
            let max_pause_ms = env_u64("HELIX_MEM_HIGH_MAX_PAUSE_MS", 100);
            let reject_on_timeout = env_bool("HELIX_MEM_BACKPRESSURE_REJECT", false);

            tracing::warn!(
                rss_mb = rss / (1024 * 1024),
                high_mb = self.high / (1024 * 1024),
                max_pause_ms,
                reject_on_timeout,
                "Memory high watermark — bounded write pause"
            );

            let started = Instant::now();
            while started.elapsed() < Duration::from_millis(max_pause_ms) {
                std::thread::sleep(Duration::from_millis(25));
                let new_rss = current_rss_bytes(); // Direct read, not cached
                if new_rss < self.high {
                    tracing::info!(
                        rss_mb = new_rss / (1024 * 1024),
                        "Memory dropped below high watermark — resuming writes"
                    );
                    return Ok(());
                }
            }

            let current = current_rss_bytes();
            metrics::counter!("helix_memory_backpressure_high_total").increment(1);
            if reject_on_timeout {
                metrics::counter!("helix_memory_backpressure_rejected_total").increment(1);
                return Err(GraphError::StorageError(format!(
                    "memory high watermark exceeded: rss={}MB high={}MB",
                    current / (1024 * 1024),
                    self.high / (1024 * 1024)
                )));
            }

            tracing::error!(
                rss_mb = current / (1024 * 1024),
                ceiling_mb = self.ceiling / (1024 * 1024),
                max_pause_ms,
                "Memory still high after bounded backpressure — proceeding"
            );
        } else {
            // Between low and high — proportional throttle.
            // At low: 0 ms. At high: HELIX_MEM_LOW_MAX_PAUSE_MS.
            let max_pause_ms = env_u64("HELIX_MEM_LOW_MAX_PAUSE_MS", 25);
            let range = self.high - self.low;
            let over = rss - self.low;
            let delay_ms = if range > 0 {
                (over as u64 * max_pause_ms) / range as u64
            } else {
                max_pause_ms / 2
            };
            if delay_ms > 10 {
                tracing::debug!(
                    rss_mb = rss / (1024 * 1024),
                    delay_ms = delay_ms,
                    "Memory backpressure — throttling write"
                );
            }
            if delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
            }
        }
        Ok(())
    }
}

static MEMORY_WATERMARKS: std::sync::LazyLock<MemoryWatermarks> =
    std::sync::LazyLock::new(MemoryWatermarks::detect);

// Database names for different stores
const DB_NODES: &str = "nodes"; // For node data (n:)
const DB_EDGES: &str = "edges"; // For edge data (e:)
                                //const DB_NODE_LABELS: &str = "node_labels"; // For node label indices (nl:)
                                //const DB_EDGE_LABELS: &str = "edge_labels"; // For edge label indices (el:)
const DB_OUT_EDGES: &str = "out_edges"; // For outgoing edge indices (o:)
const DB_IN_EDGES: &str = "in_edges"; // For incoming edge indices (i:)
const DB_EDGE_PATH_IDX: &str = "edge_path_idx"; // [dir][path][0][edge_id] -> []
const DB_METADATA: &str = "metadata";
const METADATA_CURRENT_KEY: &str = "current";
const COMMITTED_LSN_KEY: &str = "committed_lsn";
const COLLECTION_DEGRADED_MARKER_FILE: &str = ".helix_collection_degraded";
const EDGE_PATH_IDX_CURSOR_KEY: &str = "edge_path_idx:cursor";
const EDGE_PATH_IDX_COMPLETE_KEY: &str = "edge_path_idx:backfill_complete";
const ADJ_BACKFILL_FROM_POINTS_CURSOR_KEY: &str = "adj_backfill_from_points:cursor";
const ADJ_BACKFILL_FROM_POINTS_COMPLETE_KEY: &str = "adj_backfill_from_points:complete";
const HNSW_OVERRIDES_KEY: &str = "hnsw_overrides";
const MAX_RAW_LMDB_KEY_BYTES: usize = 480;

/// The adjacency backfill's mark in one byte: 0 not known, 1 not complete,
/// 2 complete.
fn adjacency_mark_code(mark: Option<bool>) -> u8 {
    match mark {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    }
}

/// A metadata sidecar as built for one put: its bytes, the fingerprint of
/// what it says, and the adjacency backfill's mark it carries.
struct BuiltSidecar {
    bytes: Vec<u8>,
    fingerprint: Option<u64>,
    mark: Option<bool>,
}

/// Stored value of an LSM keyword payload-index row indexed from a top-level
/// (non-array) value. Array-element rows, and rows older binaries wrote,
/// store nothing: their key alone cannot tell `"x"` from `["x"]`, which
/// `match.text` treats differently.
const PAYLOAD_INDEX_PLAIN_ROW: &[u8] = &[1];

#[cfg(test)]
thread_local! {
    /// Test-only: `get_node_be` calls (payload node reads) on this thread.
    static NODE_READS_FOR_TEST: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Test-only: payload node reads issued by the current thread so far.
#[cfg(test)]
pub(crate) fn node_reads_for_test() -> usize {
    NODE_READS_FOR_TEST.with(std::cell::Cell::get)
}

#[cfg(test)]
thread_local! {
    /// Test-only: run once, on this thread, by an async payload index build
    /// between its drain and its publish.
    static BEFORE_PAYLOAD_INDEX_PUBLISH_FOR_TEST: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}
const MAX_EDGE_PATH_COMPONENT_BYTES: usize = 400;
const COMPACT_INDEX_KEY_PREFIX: &[u8] = b"\xffhxk1";

// Key prefixes for different types of data

#[derive(Clone)]
struct PayloadIndexJob {
    collection: String,
    field_name: String,
    schema: PayloadIndexSchema,
    db: Option<Database<Bytes, Bytes>>,
    job_id: String,
    /// Weak so a job queued behind a backlog does not pin the collection's
    /// LMDB env after a drop — a pinned env blocks drop→recreate with
    /// `EnvAlreadyOpen` until the queue drains. Upgraded at dequeue; jobs
    /// whose collection is gone are skipped.
    storage: Weak<HelixGraphStorage>,
}

struct PayloadIndexExecutor {
    sender: flume::Sender<PayloadIndexJob>,
}

/// A worker's hold of the storage whose build it has taken. Dropped, it
/// gives back the count the build took when it was queued
/// (`HelixGraphStorage::counted_as_pending`): the build is over, however
/// the worker left it.
struct PendingPayloadIndexBuild(Arc<HelixGraphStorage>);

impl Drop for PendingPayloadIndexBuild {
    fn drop(&mut self) {
        self.0
            .payload_index_builds_pending
            .fetch_sub(1, Ordering::AcqRel);
    }
}

impl PayloadIndexExecutor {
    fn start() -> Self {
        let workers = payload_index_workers();
        let (sender, receiver): (
            flume::Sender<PayloadIndexJob>,
            flume::Receiver<PayloadIndexJob>,
        ) = flume::bounded(payload_index_queue_cap());
        for worker_id in 0..workers {
            let receiver = receiver.clone();
            if let Err(e) = std::thread::Builder::new()
                .name(format!("helix-payload-index-{}", worker_id))
                .spawn(move || {
                    for job in receiver {
                        #[cfg(test)]
                        keep_while_held_for_test(&job.collection);
                        let Some(storage) = job.storage.upgrade() else {
                            metrics::counter!(
                                "helix_payload_index_job_skipped_total",
                                "reason" => "collection_dropped"
                            )
                            .increment(1);
                            tracing::debug!(
                                collection = %job.collection,
                                field = %job.field_name,
                                job_id = %job.job_id,
                                "payload index job skipped: collection dropped while queued"
                            );
                            continue;
                        };
                        // Whatever way this turn of the loop ends, the
                        // storage's count of pending builds comes back.
                        let _pending = PendingPayloadIndexBuild(Arc::clone(&storage));
                        // A panicking job must NOT kill the worker thread — that
                        // closes the submit channel and bricks ALL future
                        // payload-index builds process-wide ("sending on a closed
                        // channel"). Isolate each job with catch_unwind.
                        let (collection, field, job_id) = (
                            job.collection.clone(),
                            job.field_name.clone(),
                            job.job_id.clone(),
                        );
                        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            storage.run_payload_index_job(job)
                        }))
                        .is_err()
                        {
                            metrics::counter!("helix_payload_index_job_panicked_total")
                                .increment(1);
                            tracing::error!(
                                collection = %collection,
                                field = %field,
                                job_id = %job_id,
                                "payload index job panicked; worker continuing"
                            );
                        }
                    }
                })
            {
                tracing::error!(
                    worker_id,
                    error = %e,
                    "failed to start payload index background worker"
                );
            }
        }
        Self { sender }
    }

    fn submit(&self, job: PayloadIndexJob) -> Result<(), GraphError> {
        let timeout = payload_index_submit_timeout();
        let started = Instant::now();
        match self.sender.send_timeout(job, timeout) {
            Ok(()) => {
                let waited = started.elapsed();
                if waited >= Duration::from_millis(250) {
                    tracing::warn!(
                        waited_ms = waited.as_millis() as u64,
                        timeout_ms = timeout.as_millis() as u64,
                        "payload index executor admission waited for queue capacity"
                    );
                }
                Ok(())
            }
            Err(flume::SendTimeoutError::Timeout(_job)) => {
                Err(GraphError::ResizeBackpressure(format!(
                    "payload index executor saturated for {} ms; retry later",
                    timeout.as_millis()
                )))
            }
            Err(flume::SendTimeoutError::Disconnected(_job)) => Err(GraphError::New(
                "payload index executor queue is unavailable: disconnected".to_string(),
            )),
        }
    }
}

static PAYLOAD_INDEX_EXECUTOR: LazyLock<PayloadIndexExecutor> =
    LazyLock::new(PayloadIndexExecutor::start);

/// Test-only: a worker that takes a job of a collection named here keeps it,
/// not yet looked at, for as long as the name stands: a job still in the
/// queue, for all that anything but that worker can tell.
#[cfg(test)]
pub(crate) static PAYLOAD_INDEX_JOBS_HELD_FOR_TEST: Mutex<Vec<String>> = Mutex::new(Vec::new());

#[cfg(test)]
fn keep_while_held_for_test(collection: &str) {
    while PAYLOAD_INDEX_JOBS_HELD_FOR_TEST
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .any(|held| held == collection)
    {
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[derive(Clone, Debug)]
pub enum PayloadIndexState {
    Building {
        job_id: String,
        started_at_millis: i64,
        updated_at_millis: i64,
        cursor: Option<u128>,
        indexed_nodes: u64,
    },
    /// Build finished. `indexed_nodes` carries the final backfilled count from the
    /// `Building` state it was promoted from (0 when reconstructed at collection
    /// open, where the count is not persisted). In-memory only — never serialized.
    /// `since` is taken after the build's last commit returned (or at handle
    /// creation): a snapshot opened later holds every index row.
    Ready { indexed_nodes: u64, since: Instant },
    Failed {
        job_id: String,
        error: String,
        updated_at_millis: i64,
    },
    Cancelled {
        job_id: Option<String>,
        updated_at_millis: i64,
    },
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PayloadIndexStatus {
    pub field_name: String,
    pub schema: PayloadIndexSchema,
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    pub indexed_nodes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at_millis: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at_millis: Option<i64>,
}

/// Counts the rows a keyword index was read for: the whole list of a value
/// (`list`), a count of it (`count`), or a scan page's window of it
/// (`window`). What a filter's planning costs is these rows, and nothing
/// else said how many they were.
fn payload_index_rows_read(read: &'static str, rows: usize) {
    metrics::counter!("helix_payload_index_rows_read_total", "read" => read).increment(rows as u64);
}

/// One keyword value's rows in a field's index, named once for a scan page
/// that reads them a window at a time
/// ([`HelixGraphStorage::payload_value_rows`]). It holds no handle of the
/// index: the page reads on from its own snapshot whatever becomes of the
/// index while it runs.
pub(crate) struct PayloadValueRows {
    db_name: String,
    keys: Vec<Vec<u8>>,
}

#[derive(Clone)]
pub struct PayloadIndexHandle {
    pub schema: PayloadIndexSchema,
    pub db: Option<Database<Bytes, Bytes>>,
    pub state: Arc<RwLock<PayloadIndexState>>,
}

impl PayloadIndexHandle {
    fn ready(schema: PayloadIndexSchema, db: Database<Bytes, Bytes>) -> Self {
        Self {
            schema,
            db: Some(db),
            state: Arc::new(RwLock::new(PayloadIndexState::Ready {
                indexed_nodes: 0,
                since: Instant::now(),
            })),
        }
    }

    fn ready_lsm(schema: PayloadIndexSchema) -> Self {
        Self {
            schema,
            db: None,
            state: Arc::new(RwLock::new(PayloadIndexState::Ready {
                indexed_nodes: 0,
                since: Instant::now(),
            })),
        }
    }

    fn building(schema: PayloadIndexSchema, db: Database<Bytes, Bytes>, job_id: String) -> Self {
        let now = chrono::Utc::now().timestamp_millis();
        Self {
            schema,
            db: Some(db),
            state: Arc::new(RwLock::new(PayloadIndexState::Building {
                job_id,
                started_at_millis: now,
                updated_at_millis: now,
                cursor: None,
                indexed_nodes: 0,
            })),
        }
    }

    fn building_lsm(schema: PayloadIndexSchema, job_id: String) -> Self {
        let now = chrono::Utc::now().timestamp_millis();
        Self {
            schema,
            db: None,
            state: Arc::new(RwLock::new(PayloadIndexState::Building {
                job_id,
                started_at_millis: now,
                updated_at_millis: now,
                cursor: None,
                indexed_nodes: 0,
            })),
        }
    }

    pub fn lmdb_db(&self) -> Result<&Database<Bytes, Bytes>, GraphError> {
        self.db.as_ref().ok_or_else(|| {
            GraphError::StorageError(
                "payload index has no LMDB DB handle on the LSM backend".to_string(),
            )
        })
    }

    pub fn is_ready(&self) -> bool {
        self.state
            .read()
            .map(|state| matches!(*state, PayloadIndexState::Ready { .. }))
            .unwrap_or(false)
    }

    /// Ready, and complete in a snapshot opened at `opened_at`. A build flips
    /// to Ready only after its last commit returned, so a snapshot opened
    /// before the flip may lack rows; such a request must treat the field as
    /// unindexed. `None` (no orderable open time: LMDB, reader replicas whose
    /// handles come from the metadata they opened with) keeps `is_ready`.
    pub fn is_ready_for(&self, opened_at: Option<Instant>) -> bool {
        self.state
            .read()
            .map(|state| match &*state {
                PayloadIndexState::Ready { since, .. } => {
                    opened_at.is_none_or(|opened_at| opened_at > *since)
                }
                _ => false,
            })
            .unwrap_or(false)
    }

    /// True while a build job owns this index (foreground writes dual-write it
    /// AND must record build deltas so the LSM backfill scan cannot resurrect
    /// stale rows — see [`Namespace::PayloadIndexBuild`]).
    pub fn is_building(&self) -> bool {
        self.state
            .read()
            .map(|state| matches!(*state, PayloadIndexState::Building { .. }))
            .unwrap_or(false)
    }

    pub fn accepts_writes(&self) -> bool {
        self.state
            .read()
            .map(|state| {
                matches!(
                    *state,
                    PayloadIndexState::Ready { .. } | PayloadIndexState::Building { .. }
                )
            })
            .unwrap_or(false)
    }

    fn is_building_job(&self, job_id: &str) -> bool {
        self.state
            .read()
            .map(|state| match &*state {
                PayloadIndexState::Building {
                    job_id: active_job_id,
                    ..
                } => active_job_id == job_id,
                _ => false,
            })
            .unwrap_or(false)
    }

    fn update_build_progress(
        &self,
        job_id: &str,
        cursor: Option<u128>,
        indexed_nodes: u64,
    ) -> bool {
        match self.state.write() {
            Ok(mut state) => match &mut *state {
                PayloadIndexState::Building {
                    job_id: active_job_id,
                    updated_at_millis,
                    cursor: active_cursor,
                    indexed_nodes: active_indexed_nodes,
                    ..
                } if active_job_id == job_id => {
                    *updated_at_millis = chrono::Utc::now().timestamp_millis();
                    *active_cursor = cursor;
                    *active_indexed_nodes = indexed_nodes;
                    true
                }
                _ => false,
            },
            Err(_) => false,
        }
    }

    fn mark_ready(&self, job_id: &str) -> bool {
        match self.state.write() {
            Ok(mut state) => match &*state {
                PayloadIndexState::Building {
                    job_id: active_job_id,
                    indexed_nodes,
                    ..
                } if active_job_id == job_id => {
                    // Promote the final backfilled count into Ready instead of
                    // discarding it — a completed index must report the number of
                    // nodes it indexed, not 0 (which is indistinguishable from a
                    // hollow index in the status / collection-info API).
                    *state = PayloadIndexState::Ready {
                        indexed_nodes: *indexed_nodes,
                        since: Instant::now(),
                    };
                    true
                }
                _ => false,
            },
            Err(_) => false,
        }
    }

    fn mark_failed(&self, job_id: &str, error: String) -> bool {
        match self.state.write() {
            Ok(mut state) => match &*state {
                PayloadIndexState::Building {
                    job_id: active_job_id,
                    ..
                } if active_job_id == job_id => {
                    *state = PayloadIndexState::Failed {
                        job_id: job_id.to_string(),
                        error,
                        updated_at_millis: chrono::Utc::now().timestamp_millis(),
                    };
                    true
                }
                _ => false,
            },
            Err(_) => false,
        }
    }

    fn mark_cancelled(&self) {
        let job_id = self.state.read().ok().and_then(|state| match &*state {
            PayloadIndexState::Building { job_id, .. } => Some(job_id.clone()),
            PayloadIndexState::Failed { job_id, .. } => Some(job_id.clone()),
            PayloadIndexState::Cancelled { job_id, .. } => job_id.clone(),
            PayloadIndexState::Ready { .. } => None,
        });
        if let Ok(mut state) = self.state.write() {
            *state = PayloadIndexState::Cancelled {
                job_id,
                updated_at_millis: chrono::Utc::now().timestamp_millis(),
            };
        }
    }

    pub fn status(&self, field_name: &str) -> PayloadIndexStatus {
        match self.state.read() {
            Ok(state) => self.status_from_state(field_name, &state),
            Err(_) => self.poisoned_status(field_name),
        }
    }

    fn try_status(&self, field_name: &str) -> Option<PayloadIndexStatus> {
        match self.state.try_read() {
            Ok(state) => Some(self.status_from_state(field_name, &state)),
            Err(std::sync::TryLockError::WouldBlock) => None,
            Err(std::sync::TryLockError::Poisoned(_)) => Some(self.poisoned_status(field_name)),
        }
    }

    fn status_from_state(&self, field_name: &str, state: &PayloadIndexState) -> PayloadIndexStatus {
        match state {
            PayloadIndexState::Ready { indexed_nodes, .. } => PayloadIndexStatus {
                field_name: field_name.to_string(),
                schema: self.schema.clone(),
                status: "ready",
                job_id: None,
                cursor: None,
                indexed_nodes: *indexed_nodes,
                error: None,
                started_at_millis: None,
                updated_at_millis: None,
            },
            PayloadIndexState::Building {
                job_id,
                started_at_millis,
                updated_at_millis,
                cursor,
                indexed_nodes,
            } => PayloadIndexStatus {
                field_name: field_name.to_string(),
                schema: self.schema.clone(),
                status: "building",
                job_id: Some(job_id.clone()),
                cursor: cursor.map(|id| id.to_string()),
                indexed_nodes: *indexed_nodes,
                error: None,
                started_at_millis: Some(*started_at_millis),
                updated_at_millis: Some(*updated_at_millis),
            },
            PayloadIndexState::Failed {
                job_id,
                error,
                updated_at_millis,
            } => PayloadIndexStatus {
                field_name: field_name.to_string(),
                schema: self.schema.clone(),
                status: "failed",
                job_id: Some(job_id.clone()),
                cursor: None,
                indexed_nodes: 0,
                error: Some(error.clone()),
                started_at_millis: None,
                updated_at_millis: Some(*updated_at_millis),
            },
            PayloadIndexState::Cancelled {
                job_id,
                updated_at_millis,
            } => PayloadIndexStatus {
                field_name: field_name.to_string(),
                schema: self.schema.clone(),
                status: "cancelled",
                job_id: job_id.clone(),
                cursor: None,
                indexed_nodes: 0,
                error: None,
                started_at_millis: None,
                updated_at_millis: Some(*updated_at_millis),
            },
        }
    }

    fn poisoned_status(&self, field_name: &str) -> PayloadIndexStatus {
        PayloadIndexStatus {
            field_name: field_name.to_string(),
            schema: self.schema.clone(),
            status: "failed",
            job_id: None,
            cursor: None,
            indexed_nodes: 0,
            error: Some("payload index state lock poisoned".to_string()),
            started_at_millis: None,
            updated_at_millis: Some(chrono::Utc::now().timestamp_millis()),
        }
    }
}

/// What a reader replica's reads coordinate on so that a search runs with the
/// dense segment list of the data view it reads.
#[derive(Default)]
struct ReaderViewReconcile {
    /// Held for the length of one reconcile, whoever started it.
    run: Mutex<()>,
    state: Mutex<ReaderViewReconcileState>,
    /// Signalled when a reconcile records a view or a started one ends.
    finished: Condvar,
}

/// Views are `(reader generation, sequence)` and compare as tuples.
#[derive(Default)]
struct ReaderViewReconcileState {
    /// The view the last complete reconcile is sure to cover: one that read
    /// the metadata afresh and left every listed segment with a core.
    covered: Option<(u64, u64)>,
    /// The newest view a read started a reconcile for. One attempt per view:
    /// if it fails, the timed refresh retries, not every read behind it.
    attempted: Option<(u64, u64)>,
    /// A reconcile started by a read has not ended yet.
    running: bool,
    /// A read reached its bound on the running one; nobody queues behind it.
    gave_up: bool,
}

/// What a reader-replica read found when it checked its dense segment list
/// against the data view it is about to read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReaderViewWait {
    /// The list covers the view already, or this is not a reader replica.
    Current,
    /// Waited for a reconcile, and the list covers the view now.
    Reconciled,
    /// The reconcile outran the bound: the read goes on with the old list.
    TimedOut,
    /// A reconcile another read gave up on is still running: no wait.
    BehindSlow,
    /// The reconcile for this view ended without covering it (it failed,
    /// left a listed segment without a core, or could not start). The timed
    /// refresh retries.
    Unreconciled,
}

pub struct HelixGraphStorage {
    collection_path: PathBuf,
    pub graph_env: Option<Env<WithTls>>,
    pub nodes_db: Option<Database<U128<BE>, Bytes>>,
    pub edges_db: Option<Database<U128<BE>, Bytes>>,
    pub out_edges_db: Option<Database<Bytes, Bytes>>,
    pub in_edges_db: Option<Database<Bytes, Bytes>>,
    pub edge_path_idx: Option<Database<Bytes, Bytes>>,
    pub metadata_db: Option<Database<Str, Bytes>>,
    pub metadata_snapshot: RwLock<StorageMetadata>,
    /// Sequence of the read behind `metadata_snapshot` and the local sidecar.
    /// Held while a refresh stores both, so they change in the order the
    /// refreshes read, and never across the object-store put.
    metadata_snapshot_read_at: parking_lot::Mutex<Option<u64>>,
    last_lsm_sidecar_upload_ms: AtomicU64,
    last_lsm_sidecar_fingerprint: AtomicU64,
    /// The put interval held back the newest sidecar: the object store is
    /// one change behind until [`Self::upload_held_metadata_sidecar`] sends it.
    lsm_sidecar_put_held: std::sync::atomic::AtomicBool,
    /// One put of the sidecar at a time, and each with the mark below as it
    /// is when the put is made. A regular put only tries it, and is held
    /// for the sweeper where it is taken; a backfill call's put of the mark
    /// ([`Self::send_adjacency_backfill_mark`]) is the one that waits.
    sidecar_put: parking_lot::Mutex<()>,
    /// Whether the adjacency backfill from points has run to its end
    /// ([`adjacency_mark_code`]): read from the store at an LSM open and at
    /// a reader's view refresh, and set by a backfill call on its writer. It
    /// is what the sidecar says and what the collection's info answers, so
    /// that neither reads the store.
    adjacency_backfill_mark: std::sync::atomic::AtomicU8,
    /// The mark in the last sidecar this process put on the object store.
    adjacency_mark_sent: std::sync::atomic::AtomicU8,
    pub secondary_indices: HashMap<String, Option<Database<Bytes, Bytes>>>,
    /// Multi-value indices (1:N) using DUP_SORT. Key = serialized Value, Values = node IDs (16 bytes each).
    pub multi_indices: HashMap<String, Option<Database<Bytes, Bytes>>>,
    pub payload_indices: RwLock<HashMap<String, PayloadIndexHandle>>,
    /// Completed `match.text` walks keyed by index handle and snapshot sequence.
    text_walk_cache: parking_lot::Mutex<super::text_walk_cache::TextWalkCache>,
    pub vectors: VectorCore,
    /// Named vector indexes for Qdrant-compatible multi-vector support.
    pub named_vectors: NamedVectorManager,
    pub wal: WalWriter,
    /// Guards against concurrent background optimizer threads for this collection.
    pub optimizer_running: std::sync::atomic::AtomicBool,
    /// Epoch-millis timestamp of the end of the most recent point upsert or
    /// delete batch. The background optimizer uses this to decide when ingest
    /// has quiesced and the mutable tail should be sealed even if it hasn't
    /// reached `flat_scan_threshold`.
    pub last_upsert_at: std::sync::atomic::AtomicU64,
    /// Point upsert/delete batches started, bumped under the point-mutation
    /// gate. A regraph compares it across its export so a write it missed
    /// stands its publish down.
    pub point_mutations: std::sync::atomic::AtomicU64,
    /// Point upsert/delete batches being applied right now. A sub-threshold
    /// dense tail is sealed for quiet only while this is zero and
    /// `last_upsert_at` is older than the quiet period.
    pub point_mutations_in_flight: AtomicUsize,
    /// Legacy coordination gate for callers/tests that explicitly gate LMDB
    /// transaction creation. The authoritative resize fence is
    /// `active_lmdb_txns`: resize-safe transaction wrappers increment that
    /// counter for the full LMDB transaction lifetime, and `env.resize()` waits
    /// until the counter drains.
    pub resize_gate: parking_lot::RwLock<()>,
    /// Number of resize writers waiting to acquire `resize_gate` exclusively.
    ///
    /// `std::sync::RwLock` can admit fresh readers while a writer is queued on
    /// some platforms. Under probe/search load that can starve an LMDB resize
    /// indefinitely. Readers check this counter before taking the shared gate
    /// so a pending resize can block new transaction creation and proceed.
    resize_writers_pending: AtomicUsize,
    /// Count of resize writers currently holding the exclusive gate.
    ///
    /// This complements `resize_writers_pending`: admission must stay closed
    /// until the queued writer has both acquired and released the exclusive
    /// resize gate. Otherwise HTTP requests can leave async admission during
    /// the active resize window and pin Tokio blocking workers on
    /// `resize_gate.read()`.
    resize_writers_active: AtomicUsize,
    /// Park primitive for readers waiting on a pending or active resize.
    ///
    /// Replaces the former 1 ms busy-spin in `read_resize_guard_for`. A
    /// reader that observes resize admission as blocked waits on
    /// `resize_wait_cv` instead of burning CPU in a sleep loop. Gateway
    /// admission keeps normal HTTP requests out of this blocking wait before
    /// they enter `spawn_blocking`; this condvar is the synchronous fallback
    /// for non-HTTP callers and races. `write_resize_guard_for` notifies all
    /// waiters once the active writer drops the exclusive gate and both resize
    /// counters return to zero. This pair is intentionally NOT nested with
    /// `resize_gate`:
    /// the mutex is dropped before `resize_gate.read()` is acquired, so the
    /// existing lock order is unchanged.
    resize_wait_lock: Mutex<()>,
    resize_wait_cv: Condvar,
    /// Number of in-process LMDB transactions currently active through the
    /// resize-safe helpers.
    ///
    /// Resizing may remap the environment. Transaction admission increments this
    /// before creating the LMDB transaction and the wrapper decrements only
    /// after the transaction has been dropped/committed/aborted.
    active_lmdb_txns: AtomicUsize,
    /// Serializes entry into write transactions for this in-process collection.
    ///
    /// LMDB itself only permits one active writer. LSM has the same logical
    /// need because collection metadata is a monolithic read-modify-write record:
    /// concurrent batches can otherwise clobber each other's payload-index,
    /// vector, or counter fields. `CollectionManager` replaces the default gate
    /// with a per-collection shared `Arc` so cache eviction/cold-open races keep
    /// one process-wide writer gate for the same collection name.
    pub write_txn_gate: Arc<Mutex<()>>,
    /// Serializes payload-index create/delete administration without blocking
    /// ordinary upserts from reading the current ready/building handle map.
    payload_index_admin_gate: Mutex<()>,
    /// Payload index builds of this collection that are queued or running:
    /// [`Self::payload_index_build_pending`].
    payload_index_builds_pending: AtomicUsize,
    /// Held by an async build job across its active-check + chunk/drain
    /// commit, and by `delete_payload_index` while it cancels the job, so a
    /// cancelled job never commits rows into a later build of the field.
    payload_index_build_lock: Mutex<()>,
    /// Set true after the first metadata-corruption event is logged for this
    /// collection. Prevents log spam on read-heavy paths where
    /// `get_metadata()` repeatedly hits the same bad bytes every request.
    pub metadata_corruption_logged: std::sync::atomic::AtomicBool,
    pub collection_degraded: std::sync::atomic::AtomicBool,
    /// Set before the shutdown step closes this storage's writer under
    /// whatever still runs on it. An operation cut by that close may fail
    /// with an error that reads like corruption, and a collection marked
    /// degraded for it would keep the marker in its directory past the
    /// restart and refuse writes.
    pub(crate) closing_for_shutdown: std::sync::atomic::AtomicBool,
    degraded_reason: RwLock<Option<(&'static str, String)>>,
    /// Set true when dense layout changes (new mutable segment created) but
    /// metadata hasn't been flushed to LMDB yet. The IndexExecutor clears this
    /// during its periodic cycle, avoiding a per-upsert LMDB write.
    pub dense_metadata_dirty: std::sync::atomic::AtomicBool,
    /// Epoch-ms wall-clock time of the last reader-replica dense-view
    /// reconcile. Used by `maybe_refresh_reader_view` to rate-limit reconciles
    /// to at most once per `HELIX_LSM_READER_REFRESH_MS` per collection.
    last_reader_refresh_ms: std::sync::atomic::AtomicU64,
    /// True while a reader-replica dense-view refresh is already running in the
    /// background. Foreground read requests must not stack duplicate refreshes or
    /// block on object-store metadata / sidecar reopen work.
    reader_refresh_inflight: std::sync::atomic::AtomicBool,
    /// Epoch-ms of the last read served against this reader-replica
    /// collection. Consulted by the read-gated `DbReader` poll-tier
    /// promotion/demotion (S3 LIST-cost reduction): a collection nobody is
    /// reading has no reason to keep its manifest poll on the fast/serving
    /// cadence. `0` means "never read since this handle opened".
    last_reader_read_at_ms: std::sync::atomic::AtomicU64,
    /// Believed-current poll tier for this reader-replica's `DbReader`: `true`
    /// = fast/serving tier, `false` = idle tier. Seeded at construction from
    /// [`backend_lsm::reader_cold_open_starts_fast`] — the SAME predicate
    /// [`backend_lsm::reader_options_from_env`] uses to pick the actual
    /// cold-open poll interval, so this starts in sync with reality (only the
    /// write-feed with the sweeper kill switch cold-opens IDLE). Every
    /// successful tier change — read-gated promotion/demotion AND the
    /// write-feed poller's own promote/demote — flows through
    /// `set_reader_poll_tier_fast` as the single source of truth, so this is
    /// only ever stale for the width of one in-flight transition: a stale
    /// `true` just skips a redundant promotion (the next read re-checks
    /// freshness via `last_reader_read_at_ms`), and a stale `false` costs one
    /// extra harmless promotion.
    ///
    /// This is the tier only where a tier change is a reopen. A reader that
    /// changes tiers in place is asked instead (`reader_poll_tier_is_fast`).
    reader_poll_tier_fast: std::sync::atomic::AtomicBool,
    /// Mutual exclusion between a poll-tier promotion and demotion for this
    /// collection: only one `refresh_with_poll_interval` swap in flight at a
    /// time. Readers that lose the race serve the current view rather than
    /// wait — one transition in flight is enough, no waiter queue.
    reader_poll_tier_transition: std::sync::atomic::AtomicBool,
    last_reader_promote_failed_at_ms: AtomicU64,
    /// Consecutive failed poll-tier promotions for this collection; drives the
    /// exponential retry backoff in `reader_poll_promote_retry_due`. Reset to
    /// zero on a successful promotion.
    reader_promote_consecutive_failures: std::sync::atomic::AtomicU32,
    /// Epoch-ms of the last write-feed change the poller saw for this
    /// collection while it was on the fast poll tier: the promoting change
    /// (`note_write_feed_promotion`) and every later one of the same burst
    /// (`note_write_feed_change`).
    /// Distinct from `last_reader_read_at_ms`: this is a WRITE-driven signal,
    /// consulted only by the read-gated demotion sweep so it doesn't demote a
    /// collection out from under an active write burst nobody has read yet —
    /// the sweeper otherwise has no visibility into write-feed activity.
    /// `0` means "never write-feed-promoted since this handle opened" (also
    /// true for good on writer/Lmdb backends and whenever the write-feed
    /// isn't configured, since nothing ever calls the setter).
    last_write_promoted_at_ms: std::sync::atomic::AtomicU64,
    /// Reader replicas: the data view the dense segment list and payload
    /// index handles were last reconciled against, and what a read waits on
    /// when its own view is past that (`reconcile_reader_view_if_advanced`).
    reader_view_reconcile: ReaderViewReconcile,
    /// CAS guard serializing `recount_lsm_counters` per collection: the scan
    /// (read) + overwrite (write) pair is not atomic against concurrent
    /// writers, so two concurrent recounts racing each other could each read
    /// a mid-flight scan value and stomp the other's result. Acquired at
    /// entry, released via an RAII guard on every exit path (success or
    /// error) so a panic or early `?` return can't leak it stuck `true`.
    recount_inflight: std::sync::atomic::AtomicBool,
    /// CAS guard serializing `gc_payload_index` per collection, same
    /// rationale as `recount_inflight`: the scan (read) + dup-delete (write)
    /// pass is not atomic against concurrent writers, so two concurrent GC
    /// runs on the same collection could race each other's deletes.
    payload_index_gc_inflight: std::sync::atomic::AtomicBool,
    /// Pluggable storage backend (US-004 integration point). Wraps a clone of
    /// `graph_env` (heed3 `Env` is `Arc`-backed, so this shares the same env);
    /// LMDB by default. Call sites migrate onto this incrementally — behavior is
    /// unchanged until they do.
    ///
    /// `Arc`-wrapped (US-006) so the same backend handle can be shared into the
    /// `NamedVectorManager` and every `VectorCore` without re-creating it —
    /// `AnyBackend` is not `Clone` and the `Lsm` variant cannot be rebuilt from
    /// an env. The 54 `self.backend.*` sites are `&self` method calls and reach
    /// through the `Arc` deref transparently.
    pub backend: Arc<AnyBackend>,
}

impl Drop for HelixGraphStorage {
    fn drop(&mut self) {
        if let Err(error) = self.backend.close_lsm_for_cache_eviction() {
            tracing::warn!(
                path = ?self.path(),
                error = %error,
                "failed to close LSM collection writer during storage drop"
            );
        }
        self.named_vectors.unlink_retired_sidecars(self.path());
        tracing::info!(path = ?self.path(), "closing collection storage");
    }
}

type ResizeReadGuard<'a> = parking_lot::lock_api::RwLockReadGuard<'a, parking_lot::RawRwLock, ()>;
type ResizeRawWriteGuard<'a> =
    parking_lot::lock_api::RwLockWriteGuard<'a, parking_lot::RawRwLock, ()>;

pub(crate) struct ResizeWriteGuard<'a> {
    storage: &'a HelixGraphStorage,
    guard: Option<ResizeRawWriteGuard<'a>>,
}

impl Drop for ResizeWriteGuard<'_> {
    fn drop(&mut self) {
        drop(self.guard.take());
        self.storage.resize_writer_finished();
    }
}

struct LmdbTxnPermit<'a> {
    storage: &'a HelixGraphStorage,
    operation: &'static str,
    started: Instant,
}

impl Drop for LmdbTxnPermit<'_> {
    fn drop(&mut self) {
        self.storage
            .finish_lmdb_txn(self.operation, self.started.elapsed());
    }
}

pub struct ResizeSafeReadTxn<'a> {
    txn: Option<RoTxn<'a, WithTls>>,
    _permit: LmdbTxnPermit<'a>,
}

pub struct NestedDenseReadTxnProvider<'a, 'env> {
    storage: &'a HelixGraphStorage,
    _outer: std::marker::PhantomData<fn(&'a ResizeSafeReadTxn<'env>)>,
}

impl<'a> Deref for ResizeSafeReadTxn<'a> {
    type Target = RoTxn<'a, WithTls>;

    fn deref(&self) -> &Self::Target {
        self.txn.as_ref().expect("read txn already dropped")
    }
}

impl Drop for ResizeSafeReadTxn<'_> {
    fn drop(&mut self) {
        drop(self.txn.take());
    }
}

pub struct ResizeSafeWriteTxn<'a> {
    txn: Option<RwTxn<'a>>,
    _permit: LmdbTxnPermit<'a>,
}

impl ResizeSafeWriteTxn<'_> {
    pub fn commit(mut self) -> Result<(), heed3::Error> {
        self.txn
            .take()
            .expect("write txn already consumed")
            .commit()
    }
}

impl<'a> ResizeSafeWriteTxn<'a> {
    /// Move the inner heed write txn out, leaving the resize permit to drop with
    /// `self`. The field-flip write-view entry uses this to put the `RwTxn` into
    /// `AnyWrite::Lmdb` while keeping the permit (and therefore the resize
    /// fence) alive for the txn's whole lifetime.
    pub(crate) fn take_txn(&mut self) -> RwTxn<'a> {
        self.txn.take().expect("write txn already taken")
    }
}

impl<'a> Deref for ResizeSafeWriteTxn<'a> {
    type Target = RwTxn<'a>;

    fn deref(&self) -> &Self::Target {
        self.txn.as_ref().expect("write txn already dropped")
    }
}

impl DerefMut for ResizeSafeWriteTxn<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.txn.as_mut().expect("write txn already dropped")
    }
}

impl Drop for ResizeSafeWriteTxn<'_> {
    fn drop(&mut self) {
        drop(self.txn.take());
    }
}

/// Owns the resize-safe write batch for one write query (traversal field flip).
///
/// On LMDB it holds the writer gate + the resize-safe wrapper (which holds the
/// resize permit) + the heed `RwTxn` wrapped in `AnyWrite::Lmdb`; on LSM it
/// holds the same writer gate + a SlateDB write batch. A whole write query threads
/// `&mut view.write_mut()` into `G::new_mut`, so every op shares ONE batch that
/// `commit()` applies atomically — the backend-neutral analogue of
/// `begin_resize_safe_write_txn` + `txn.commit()`.
pub struct WriteView<'a> {
    storage: &'a HelixGraphStorage,
    _gate: Option<MutexGuard<'a, ()>>,
    // Kept alive (with its inner txn taken out into `write`) so the resize
    // permit drops only AFTER `commit`, exactly like the manual write path.
    _resize: Option<ResizeSafeWriteTxn<'a>>,
    write: Option<AnyWrite<'a>>,
}

impl<'a> WriteView<'a> {
    /// Mutable access to the shared write batch for `G::new_mut`.
    pub fn write_mut(&mut self) -> &mut AnyWrite<'a> {
        self.write.as_mut().expect("write view already committed")
    }

    /// Read-your-writes view of the live batch: a read view that observes the
    /// batch's own buffered (uncommitted) writes, so a write query can read a
    /// node/edge it wrote earlier in the same query.
    ///
    /// LMDB: borrows the batch's `RwTxn` (which reads its own writes for free) as
    /// a read view — byte-identical read-your-writes. LSM: takes a fresh
    /// committed snapshot and overlays a clone of the batch's `pending` map, so
    /// point reads see buffered puts/deletes too.
    ///
    /// Borrowing `self` for the returned view's lifetime enforces that the read
    /// view is dropped before the next `write_mut()` — callers scope it.
    pub fn read_view(&self) -> AnyRead<'_> {
        let w = self.write.as_ref().expect("write view already committed");
        match self.storage.backend.kind() {
            BackendKind::Lmdb => {
                let rwtxn = w.lmdb_ro().expect("LMDB write view must hold an RwTxn");
                self.storage.backend.read_borrowed(rwtxn)
            }
            BackendKind::Lsm => {
                let pending = w
                    .lsm_pending()
                    .expect("LSM write view must hold a pending overlay")
                    .clone();
                self.storage
                    .backend
                    .lsm_read_with_pending(pending)
                    .expect("LSM begin_read should not fail for write-view read")
            }
        }
    }

    /// Commit the batch atomically. On LMDB this commits the heed `RwTxn`; the
    /// resize permit + writer gate then drop (after the commit), matching the
    /// manual `ResizeSafeWriteTxn::commit` ordering. On LSM it durably writes
    /// the SlateDB batch.
    pub fn commit(mut self) -> Result<(), GraphError> {
        let w = self.write.take().expect("write view already committed");
        self.storage
            .backend
            .commit(w)
            .map_err(|e| GraphError::New(e.to_string()))
    }
}

/// A chunk-level edge candidate found while rebuilding `_graph` adjacency.
/// Unlike the Symbol-node edges `relationship_upserts_from_node` derives,
/// both endpoints here are Qdrant point ids (rendered as decimal strings
/// by the client) belonging to the BASE collection, not this `_graph` collection.
/// `handle_rebuild_adjacency_for_paths` resolves and applies these to the
/// base collection when the request carries a `base_collection`.
pub(crate) struct ChunkEdgeCandidate {
    pub(crate) edge_type: String,
    pub(crate) caller_point_id: String,
    pub(crate) callee_point_id: String,
    pub(crate) caller_path: String,
    pub(crate) callee_path: String,
}

/// What [`HelixGraphStorage::rebuild_adjacency_for_paths`] wrote.
#[derive(Default)]
pub(crate) struct PathsRebuild {
    pub(crate) nodes_upserted: usize,
    /// The edges derived, written or not.
    pub(crate) edges_upserted: usize,
    /// Of them, the edges that were left: stored exactly as derived (one
    /// that two rows derive, as the later derives it).
    pub(crate) edges_unchanged: usize,
    pub(crate) chunk_candidates: Vec<ChunkEdgeCandidate>,
    /// The edges dropped and the nodes they left without an edge, when the
    /// paths' outgoing edges were replaced; `None` when they were not.
    pub(crate) replaced: Option<(usize, usize)>,
}

impl HelixGraphStorage {
    pub(crate) fn path(&self) -> &Path {
        &self.collection_path
    }

    pub(crate) fn lmdb_env(&self) -> Result<&Env<WithTls>, GraphError> {
        self.graph_env.as_ref().ok_or_else(|| {
            GraphError::StorageError("LMDB env handle is unavailable on this backend".to_string())
        })
    }

    /// (entries, hits) of the text-walk cache.
    #[cfg(test)]
    pub(crate) fn text_walk_cache_stats(&self) -> (usize, u64) {
        let cache = self.text_walk_cache.lock();
        (cache.len(), cache.hits())
    }

    pub(crate) fn collection_path(&self) -> &Path {
        &self.collection_path
    }

    pub(crate) fn lmdb_nodes_db(&self) -> Result<Database<U128<BE>, Bytes>, GraphError> {
        self.nodes_db.ok_or_else(|| {
            GraphError::StorageError(
                "LMDB node DB handle is unavailable on this backend".to_string(),
            )
        })
    }

    pub(crate) fn lmdb_edges_db(&self) -> Result<Database<U128<BE>, Bytes>, GraphError> {
        self.edges_db.ok_or_else(|| {
            GraphError::StorageError(
                "LMDB edge DB handle is unavailable on this backend".to_string(),
            )
        })
    }

    pub(crate) fn lmdb_metadata_db(&self) -> Result<Database<Str, Bytes>, GraphError> {
        self.metadata_db.ok_or_else(|| {
            GraphError::StorageError(
                "LMDB metadata DB handle is unavailable on this backend".to_string(),
            )
        })
    }

    pub(crate) fn lmdb_edge_path_idx(&self) -> Result<Database<Bytes, Bytes>, GraphError> {
        self.edge_path_idx.ok_or_else(|| {
            GraphError::StorageError(
                "LMDB edge-path index DB handle is unavailable on this backend".to_string(),
            )
        })
    }

    fn degraded_code_from_marker(code: &str) -> &'static str {
        match code {
            "mdb_problem" => "mdb_problem",
            "mdb_page_not_found" => "mdb_page_not_found",
            "mdb_corrupted" => "mdb_corrupted",
            "mdb_panic" => "mdb_panic",
            _ => "collection_degraded",
        }
    }

    fn degraded_marker_path(path: &Path) -> PathBuf {
        path.join(COLLECTION_DEGRADED_MARKER_FILE)
    }

    pub(crate) fn read_degraded_marker(path: &str) -> Option<(&'static str, String)> {
        let marker = Self::degraded_marker_path(Path::new(path));
        let content = fs::read_to_string(marker).ok()?;
        let mut lines = content.lines();
        let code = Self::degraded_code_from_marker(lines.next().unwrap_or_default());
        let message = lines.collect::<Vec<_>>().join("\n");
        let message = if message.trim().is_empty() {
            "collection marked degraded; rebuild or quarantine required".to_string()
        } else {
            message
        };
        Some((code, message))
    }

    fn persist_degraded_marker(&self, code: &'static str, message: &str) {
        let marker = Self::degraded_marker_path(self.path());
        let body = format!("{code}\n{message}\n");
        if let Err(err) = fs::write(&marker, body) {
            tracing::warn!(
                path = ?marker,
                error = %err,
                "failed to persist collection degraded marker"
            );
        }
    }

    pub fn mark_collection_degraded(&self, error: &GraphError, context: &'static str) {
        let GraphError::FatalCollectionStorage { code, message } = error else {
            return;
        };
        if self.closing_for_shutdown.load(Ordering::Acquire) {
            tracing::warn!(
                code = *code,
                message = %message,
                context,
                path = ?self.path(),
                "storage error under the shutdown close; collection not marked degraded"
            );
            return;
        }
        let first = !self.collection_degraded.swap(true, Ordering::AcqRel);
        if let Ok(mut reason) = self.degraded_reason.write() {
            *reason = Some((*code, message.clone()));
        }
        if first {
            metrics::counter!(
                "helix_collection_degraded_total",
                "code" => (*code).to_string(),
                "context" => context
            )
            .increment(1);
            self.persist_degraded_marker(*code, message);
            tracing::error!(
                code = *code,
                message = %message,
                context,
                path = ?self.path(),
                "marked collection degraded after fatal LMDB storage error"
            );
        }
    }

    pub fn degraded_collection_error(&self) -> Option<GraphError> {
        if !self.collection_degraded.load(Ordering::Acquire) {
            return None;
        }
        let (code, message) = self
            .degraded_reason
            .read()
            .ok()
            .and_then(|reason| reason.clone())
            .unwrap_or((
                "collection_degraded",
                "collection marked degraded; rebuild or quarantine required".to_string(),
            ));
        Some(GraphError::FatalCollectionStorage { code, message })
    }

    pub fn ensure_not_degraded(&self) -> Result<(), GraphError> {
        match self.degraded_collection_error() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    pub fn unhealthy_lsm_writer_close_reason(&self) -> Option<CloseReason> {
        self.backend.unhealthy_lsm_writer_close_reason()
    }

    fn compact_index_bytes(bytes: &[u8], max_raw_len: usize) -> Vec<u8> {
        if bytes.len() <= max_raw_len {
            return bytes.to_vec();
        }

        let mut h1 = XxHash64::with_seed(0x4845_4c49_585f_4b31);
        h1.write(bytes);
        let mut h2 = XxHash64::with_seed(0x4345_5f4b_4559_5f32);
        h2.write(bytes);

        let mut out = Vec::with_capacity(COMPACT_INDEX_KEY_PREFIX.len() + 24);
        out.extend_from_slice(COMPACT_INDEX_KEY_PREFIX);
        out.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
        out.extend_from_slice(&h1.finish().to_be_bytes());
        out.extend_from_slice(&h2.finish().to_be_bytes());
        out
    }

    pub(crate) fn stable_index_key_for_value(value: &Value) -> Result<Vec<u8>, GraphError> {
        let encoded = bincode::serialize(value)?;
        Ok(Self::compact_index_bytes(&encoded, MAX_RAW_LMDB_KEY_BYTES))
    }

    fn edge_path_index_component(path: &str) -> Vec<u8> {
        Self::compact_index_bytes(path.as_bytes(), MAX_EDGE_PATH_COMPONENT_BYTES)
    }

    fn edge_path_index_prefix(dir: u8, path: &str) -> Vec<u8> {
        let path = Self::edge_path_index_component(path);
        let mut key = Vec::with_capacity(1 + path.len() + 1);
        key.push(dir);
        key.extend_from_slice(&path);
        key.push(0);
        key
    }

    fn edge_path_index_key(dir: u8, path: &str, edge_id: &u128) -> Vec<u8> {
        let mut key = Self::edge_path_index_prefix(dir, path);
        key.extend_from_slice(&edge_id.to_be_bytes());
        key
    }

    /// Index directions to read: 1 holds an edge under its `from_path`, 2
    /// under its `to_path`.
    fn edge_path_index_dirs(from_only: bool) -> &'static [u8] {
        if from_only {
            &[1]
        } else {
            &[1, 2]
        }
    }

    fn edge_path_property(edge: &Edge, key: &str) -> Option<String> {
        match edge.properties.get(key) {
            Some(Value::String(path)) if !path.is_empty() => Some(path.clone()),
            _ => None,
        }
    }

    pub fn edge_ids_for_path(
        &self,
        txn: &RoTxn,
        path: &str,
        max_edges: usize,
    ) -> Result<Vec<u128>, GraphError> {
        self.edge_ids_for_path_ends(txn, path, max_edges, false)
    }

    /// The edges with an end in `path`: both ends, or with `from_only` only
    /// the edges that start there (index direction 1 is `from_path`).
    pub fn edge_ids_for_path_ends(
        &self,
        txn: &RoTxn,
        path: &str,
        max_edges: usize,
        from_only: bool,
    ) -> Result<Vec<u128>, GraphError> {
        let mut seen = std::collections::HashSet::new();
        let mut ids = Vec::new();
        for &dir in Self::edge_path_index_dirs(from_only) {
            let prefix = Self::edge_path_index_prefix(dir, path);
            let edge_path_idx = self.lmdb_edge_path_idx()?;
            let iter = edge_path_idx.prefix_iter(txn, &prefix)?;
            for item in iter {
                let (key, _) = item?;
                if key.len() < 16 {
                    continue;
                }
                let start = key.len() - 16;
                let mut raw = [0u8; 16];
                raw.copy_from_slice(&key[start..]);
                let edge_id = u128::from_be_bytes(raw);
                if seen.insert(edge_id) {
                    ids.push(edge_id);
                    if ids.len() >= max_edges {
                        return Ok(ids);
                    }
                }
            }
        }
        Ok(ids)
    }

    pub fn edge_ids_for_path_be(
        &self,
        r: &AnyRead<'_>,
        path: &str,
        max_edges: usize,
    ) -> Result<Vec<u128>, GraphError> {
        self.edge_ids_for_path_ends_be(r, path, max_edges, false)
    }

    /// Backend-routed twin of [`Self::edge_ids_for_path_ends`].
    pub fn edge_ids_for_path_ends_be(
        &self,
        r: &AnyRead<'_>,
        path: &str,
        max_edges: usize,
        from_only: bool,
    ) -> Result<Vec<u128>, GraphError> {
        let mut seen = std::collections::HashSet::new();
        let mut ids = Vec::new();
        for &dir in Self::edge_path_index_dirs(from_only) {
            let prefix = Self::edge_path_index_prefix(dir, path);
            self.backend
                .scan(
                    r,
                    Namespace::EdgePathIdx,
                    KeyRange::prefix(&prefix),
                    |key, _| {
                        if key.len() < 16 {
                            return true;
                        }
                        let start = key.len() - 16;
                        let mut raw = [0u8; 16];
                        raw.copy_from_slice(&key[start..]);
                        let edge_id = u128::from_be_bytes(raw);
                        if seen.insert(edge_id) {
                            ids.push(edge_id);
                            if ids.len() >= max_edges {
                                return false;
                            }
                        }
                        true
                    },
                )
                .map_err(|e| GraphError::New(e.to_string()))?;
            if ids.len() >= max_edges {
                return Ok(ids);
            }
        }
        Ok(ids)
    }

    /// The edges that start in `paths` and are not among `derived`, each with
    /// its two ends: through the path index once its backfill is complete,
    /// and by reading every edge until then. An edge that does not decode
    /// cannot be told to start there and is left out.
    fn stale_edges_starting_in_paths_be(
        &self,
        r: &AnyRead<'_>,
        paths: &std::collections::HashSet<&str>,
        derived: &std::collections::HashSet<u128>,
    ) -> Result<std::collections::BTreeMap<u128, (u128, u128)>, GraphError> {
        /// Edge rows held in memory by one batched read.
        const EDGE_ROWS_PER_READ: usize = 5_000;

        let mut stale = std::collections::BTreeMap::new();
        let mut note = |edge_id: u128, data: &[u8]| {
            if let Ok(edge) = SerializedEdge::decode_edge(data, edge_id) {
                if matches!(
                    edge.properties.get("from_path"),
                    Some(Value::String(path)) if paths.contains(path.as_str())
                ) {
                    stale.insert(edge_id, (edge.from_node, edge.to_node));
                }
            }
        };
        if self.edge_path_index_backfill_complete_be(r)? {
            let mut edge_ids = Vec::new();
            for path in paths {
                edge_ids.extend(
                    self.edge_ids_for_path_ends_be(r, path, usize::MAX, true)?
                        .into_iter()
                        .filter(|edge_id| !derived.contains(edge_id)),
                );
            }
            // An index row is only a hint: the edge itself says where it starts.
            for edge_ids in edge_ids.chunks(EDGE_ROWS_PER_READ) {
                let keys: Vec<Vec<u8>> = edge_ids
                    .iter()
                    .map(|edge_id| edge_id.to_be_bytes().to_vec())
                    .collect();
                let rows = self
                    .backend
                    .get_many(r, Namespace::Edges, &keys)
                    .map_err(|e| GraphError::New(e.to_string()))?;
                for (edge_id, row) in edge_ids.iter().zip(&rows) {
                    if let Some(data) = row {
                        note(*edge_id, data);
                    }
                }
            }
        } else {
            self.backend
                .scan(r, Namespace::Edges, KeyRange::all(), |key, data| {
                    if let Ok(raw) = <[u8; 16]>::try_from(key) {
                        let edge_id = u128::from_be_bytes(raw);
                        if !derived.contains(&edge_id) {
                            note(edge_id, data);
                        }
                    }
                    true
                })
                .map_err(|e| GraphError::New(e.to_string()))?;
        }
        Ok(stale)
    }

    pub fn edge_path_index_backfill_complete(&self, txn: &RoTxn) -> Result<bool, GraphError> {
        // Routed through the backend seam (heed-txn scaffold; Namespace::Metadata
        // resolves to the same metadata_db on LMDB). Byte-identical: metadata_db
        // is Str-keyed and heed `Str` stores raw UTF-8 with no prefix, so the
        // key bytes match `EDGE_PATH_IDX_COMPLETE_KEY.as_bytes()`.
        self.backend
            .get_with_heed(
                txn,
                Namespace::Metadata,
                EDGE_PATH_IDX_COMPLETE_KEY.as_bytes(),
                |v| matches!(v, Some(b"1")),
            )
            .map_err(|e| GraphError::New(e.to_string()))
    }

    pub fn edge_path_index_backfill_complete_be(
        &self,
        r: &AnyRead<'_>,
    ) -> Result<bool, GraphError> {
        self.backend
            .get_with(
                r,
                Namespace::Metadata,
                EDGE_PATH_IDX_COMPLETE_KEY.as_bytes(),
                |v| matches!(v, Some(b"1")),
            )
            .map_err(|e| GraphError::New(e.to_string()))
    }

    pub fn backfill_edge_path_index_batch(
        &self,
        batch_size: usize,
    ) -> Result<(usize, bool, Option<u128>), GraphError> {
        use super::backend::KeyRange;

        // LSM has no local LMDB env: the heed path below cannot even open its
        // read txn there, and the cursor/index writes require LMDB handles that
        // are `None` on LSM pods. Route through the backend-write twin.
        if self.backend.kind() == BackendKind::Lsm {
            return self.backfill_edge_path_index_batch_be(batch_size);
        }

        let batch_size = batch_size.max(1);
        let (edge_ids, last_edge_id, exhausted) = self.with_read_txn(|rtxn| {
            if self.edge_path_index_backfill_complete(rtxn)? {
                return Ok((Vec::new(), None, true));
            }

            // Cursor read routed through the seam: Namespace::Metadata resolves to
            // the same metadata_db (Str-keyed, raw UTF-8) so the key bytes match
            // `EDGE_PATH_IDX_CURSOR_KEY.as_bytes()`; the stored value is the raw
            // 16-byte big-endian edge id, decoded inside the closure-scoped borrow.
            let cursor = self
                .backend
                .get_with_heed(
                    rtxn,
                    Namespace::Metadata,
                    EDGE_PATH_IDX_CURSOR_KEY.as_bytes(),
                    |v| {
                        v.and_then(|raw| {
                            if raw.len() == 16 {
                                let mut bytes = [0u8; 16];
                                bytes.copy_from_slice(raw);
                                Some(u128::from_be_bytes(bytes))
                            } else {
                                None
                            }
                        })
                    },
                )
                .map_err(|e| GraphError::New(e.to_string()))?;

            let mut edge_ids = Vec::new();
            let mut last_edge_id = None;
            let mut exhausted = true;

            // Forward scan of Namespace::Edges from the cursor (exclusive) to the
            // end, early-stopping after `batch_size`. Byte-identical to the prior
            // `edges_db.range(rtxn, &(Excluded(cursor), Unbounded))`: edge ids are
            // 16-byte big-endian keys (matching `Database<U128<BE>, Bytes>`), and
            // the seam's `scan_raw` breaks when the visitor returns `false`.
            let start = cursor
                .map(|id| Bound::Excluded(id.to_be_bytes().to_vec()))
                .unwrap_or(Bound::Unbounded);
            let range = KeyRange {
                start,
                end: Bound::Unbounded,
            };
            self.backend
                .scan_heed(rtxn, Namespace::Edges, range, |k, _v| {
                    let mut bytes = [0u8; 16];
                    bytes.copy_from_slice(k);
                    let edge_id = u128::from_be_bytes(bytes);
                    edge_ids.push(edge_id);
                    last_edge_id = Some(edge_id);
                    if edge_ids.len() >= batch_size {
                        exhausted = false;
                        false
                    } else {
                        true
                    }
                })
                .map_err(|e| GraphError::New(e.to_string()))?;

            Ok((edge_ids, last_edge_id, exhausted))
        })?;

        self.with_write_txn(|wtxn| {
            let mut indexed = 0usize;
            for edge_id in &edge_ids {
                // Read-your-writes through the write txn, routed via the seam.
                // Decode inside the visitor closure (the borrow is closure-scoped).
                let decoded = self
                    .backend
                    .get_for_update_heed(wtxn, Namespace::Edges, &edge_id.to_be_bytes(), |v| {
                        v.map(|bytes| SerializedEdge::decode_edge(bytes, *edge_id))
                    })
                    .map_err(|e| GraphError::New(e.to_string()))?;
                let edge = match decoded {
                    None => continue,
                    Some(Ok(edge)) => edge,
                    Some(Err(err)) => {
                        tracing::warn!(
                            "edge_path_idx backfill skipped undecodable edge {}: {}",
                            edge_id,
                            err
                        );
                        continue;
                    }
                };
                // Not skipped on failure, as in the backend-routed twin.
                self.index_edge_paths(wtxn, &edge)?;
                indexed += 1;
            }

            if let Some(last) = last_edge_id {
                self.lmdb_metadata_db()?.put(
                    wtxn,
                    EDGE_PATH_IDX_CURSOR_KEY,
                    &last.to_be_bytes(),
                )?;
            }
            if exhausted {
                self.lmdb_metadata_db()?
                    .put(wtxn, EDGE_PATH_IDX_COMPLETE_KEY, b"1")?;
            }

            Ok((indexed, exhausted, last_edge_id))
        })
    }

    /// Backend-routed twin of `backfill_edge_path_index_batch` for LSM pods
    /// (no local LMDB env, so the heed cursor/index writes above are
    /// unavailable). Same two-phase shape: committed-snapshot scan for the
    /// batch of edge ids, then one write batch that indexes the edges and
    /// advances the cursor / complete marker in `Namespace::Metadata`.
    fn backfill_edge_path_index_batch_be(
        &self,
        batch_size: usize,
    ) -> Result<(usize, bool, Option<u128>), GraphError> {
        use super::backend::KeyRange;

        let batch_size = batch_size.max(1);
        let (edge_ids, last_edge_id, exhausted) = {
            let r = self
                .backend
                .begin_read()
                .map_err(|e| GraphError::New(e.to_string()))?;
            if self.edge_path_index_backfill_complete_be(&r)? {
                return Ok((0, true, None));
            }

            let cursor = self
                .backend
                .get_with(
                    &r,
                    Namespace::Metadata,
                    EDGE_PATH_IDX_CURSOR_KEY.as_bytes(),
                    |v| {
                        v.and_then(|raw| {
                            if raw.len() == 16 {
                                let mut bytes = [0u8; 16];
                                bytes.copy_from_slice(raw);
                                Some(u128::from_be_bytes(bytes))
                            } else {
                                None
                            }
                        })
                    },
                )
                .map_err(|e| GraphError::New(e.to_string()))?;

            let mut edge_ids = Vec::new();
            let mut last_edge_id = None;
            let mut exhausted = true;
            let start = cursor
                .map(|id| Bound::Excluded(id.to_be_bytes().to_vec()))
                .unwrap_or(Bound::Unbounded);
            let range = KeyRange {
                start,
                end: Bound::Unbounded,
            };
            self.backend
                .scan(&r, Namespace::Edges, range, |k, _v| {
                    if k.len() != 16 {
                        return true;
                    }
                    let mut bytes = [0u8; 16];
                    bytes.copy_from_slice(k);
                    let edge_id = u128::from_be_bytes(bytes);
                    edge_ids.push(edge_id);
                    last_edge_id = Some(edge_id);
                    if edge_ids.len() >= batch_size {
                        exhausted = false;
                        return false;
                    }
                    true
                })
                .map_err(|e| GraphError::New(e.to_string()))?;

            (edge_ids, last_edge_id, exhausted)
        };

        self.with_write_backend(|w| {
            let mut indexed = 0usize;
            for edge_id in &edge_ids {
                let decoded = self
                    .backend
                    .get_for_update(w, Namespace::Edges, &edge_id.to_be_bytes(), |v| {
                        v.map(|bytes| SerializedEdge::decode_edge(bytes, *edge_id))
                    })
                    .map_err(|e| GraphError::New(e.to_string()))?;
                let edge = match decoded {
                    None => continue,
                    Some(Ok(edge)) => edge,
                    Some(Err(err)) => {
                        tracing::warn!(
                            "edge_path_idx backfill skipped undecodable edge {}: {}",
                            edge_id,
                            err
                        );
                        continue;
                    }
                };
                // Not skipped on failure: the cursor below would pass an edge
                // without its rows, and once the marker is set nothing looks
                // for that edge by path again.
                self.index_edge_paths_be(w, &edge)?;
                indexed += 1;
            }

            if let Some(last) = last_edge_id {
                self.backend
                    .put(
                        w,
                        Namespace::Metadata,
                        EDGE_PATH_IDX_CURSOR_KEY.as_bytes(),
                        &last.to_be_bytes(),
                    )
                    .map_err(|e| GraphError::New(e.to_string()))?;
            }
            if exhausted {
                self.backend
                    .put(
                        w,
                        Namespace::Metadata,
                        EDGE_PATH_IDX_COMPLETE_KEY.as_bytes(),
                        b"1",
                    )
                    .map_err(|e| GraphError::New(e.to_string()))?;
            }

            Ok((indexed, exhausted, last_edge_id))
        })
    }

    /// Scan a batch of relationship points (nodes carrying `caller_symbol` and
    /// `callee_symbol`) and rebuild the native graph adjacency that the forward
    /// the client → Helix ingest path would have written. For every relationship point we
    /// upsert the caller/callee `Symbol` nodes and a deterministic edge, which
    /// populates `OutEdges`/`InEdges` and the edge counter.
    ///
    /// Returns `(nodes_upserted, edges_upserted, complete, next_cursor,
    /// edges_unchanged)`. Callers should repeat the endpoint with the returned
    /// cursor until `complete` is true. The operation is idempotent: re-running
    /// on the same point set derives the same deterministic node/edge rows.
    ///
    /// On LSM an edge that is stored exactly as its point derives it, with
    /// its adjacency and path keys, is left
    /// ([`Self::upsert_derived_edges_be`]) and counted in `edges_unchanged`:
    /// a pass puts back every edge that is absent, changed or short of a key,
    /// and, as before, every absent node row of an edge's end. A call still
    /// commits its cursor, so a pass over a whole graph walks to its end and
    /// sets its mark. `rewrite_unchanged_edges` has the call write every edge
    /// it derives, as every call did before: for a store whose rows or keys
    /// are not what a read of them says. It does not write an equal node row
    /// again, as no call did.
    ///
    /// `force` starts the pass over (mark and cursor cleared, the scan from
    /// the first node) and nothing more.
    pub fn backfill_adjacency_from_points_batch(
        &self,
        collection: &str,
        batch_size: usize,
        force: bool,
        rewrite_unchanged_edges: bool,
    ) -> Result<(usize, usize, bool, Option<u128>, usize), GraphError> {
        let batch_size = batch_size.clamp(1, 10_000);

        // A finished pass is only reported: a caller asks once per process
        // whether a collection is done, and that must not cost a commit.
        if !force && self.with_read_backend(|r| self.adjacency_backfill_marked_be(r))? {
            // The sidecar says so from here on, so that a reader answers the
            // next caller, who then need not open the collection on its
            // writer: one put, where this process's last sidecar did not say it.
            if let Err(error) = self.send_adjacency_backfill_mark(true) {
                tracing::warn!(
                    collection = self.collection_display_name(),
                    error = %error,
                    "a finished adjacency backfill's mark was not put in the metadata sidecar"
                );
            }
            return Ok((0, 0, true, None, 0));
        }

        let (node_upserts, edge_upserts, last_node_id, exhausted) =
            self.with_read_backend(|r| {
                let cursor = if force {
                    None
                } else {
                    self.backend
                        .get_with(
                            r,
                            Namespace::Metadata,
                            ADJ_BACKFILL_FROM_POINTS_CURSOR_KEY.as_bytes(),
                            |v| {
                                v.and_then(|raw| {
                                    if raw.len() == 16 {
                                        let mut bytes = [0u8; 16];
                                        bytes.copy_from_slice(raw);
                                        Some(u128::from_be_bytes(bytes))
                                    } else {
                                        None
                                    }
                                })
                            },
                        )
                        .map_err(|e| GraphError::New(e.to_string()))?
                };

                let start = cursor
                    .map(|id| Bound::Excluded(id.to_be_bytes().to_vec()))
                    .unwrap_or(Bound::Unbounded);
                let range = KeyRange {
                    start,
                    end: Bound::Unbounded,
                };

                let mut node_upserts = Vec::new();
                let mut edge_upserts = Vec::new();
                let mut last_node_id = None;
                let mut relationship_count = 0usize;
                let mut exhausted = true;

                // Streamed: a pass reads every node of the graph once, and
                // must not push the blocks searches read out of the cache.
                self.backend
                    .scan_streaming(r, Namespace::Nodes, range, |key, value| {
                        let id = match <[u8; 16]>::try_from(key) {
                            Ok(bytes) => u128::from_be_bytes(bytes),
                            Err(_) => return true,
                        };
                        last_node_id = Some(id);

                        let node = match SerializedNode::decode_node(value, id) {
                            Ok(n) => n,
                            Err(_) => return true,
                        };

                        if let Some((caller, callee, edge)) =
                            Self::relationship_upserts_from_node(collection, &node.properties)
                        {
                            node_upserts.push(caller);
                            node_upserts.push(callee);
                            edge_upserts.push(edge);
                            relationship_count += 1;
                            if relationship_count >= batch_size {
                                exhausted = false;
                                return false;
                            }
                        }

                        true
                    })
                    .map_err(|e| GraphError::New(e.to_string()))?;

                Ok((node_upserts, edge_upserts, last_node_id, exhausted))
            })?;

        let nodes_upserted = node_upserts.len();
        let edges_upserted = edge_upserts.len();
        let leave_as_derived = !rewrite_unchanged_edges && self.leaves_edges_stored_as_derived();

        // Before the write that clears the mark, and whatever the put
        // interval: the object store must not say "complete" of a pass that
        // has started over. A put that fails fails the call, unwritten.
        if force {
            self.send_adjacency_backfill_mark(false)?;
        }

        let edges_unchanged = self.with_write_backend(|w| {
            if force {
                self.backend
                    .delete(
                        w,
                        Namespace::Metadata,
                        ADJ_BACKFILL_FROM_POINTS_COMPLETE_KEY.as_bytes(),
                    )
                    .map_err(|e| GraphError::New(e.to_string()))?;
                self.backend
                    .delete(
                        w,
                        Namespace::Metadata,
                        ADJ_BACKFILL_FROM_POINTS_CURSOR_KEY.as_bytes(),
                    )
                    .map_err(|e| GraphError::New(e.to_string()))?;
            }
            for node in &node_upserts {
                self.upsert_node_be(w, node)?;
            }
            let edges_unchanged = if leave_as_derived {
                self.upsert_derived_edges_be(w, &edge_upserts)?
            } else {
                for edge in &edge_upserts {
                    self.upsert_edge_be(w, edge)?;
                }
                0
            };

            if let Some(last) = last_node_id {
                self.backend
                    .put(
                        w,
                        Namespace::Metadata,
                        ADJ_BACKFILL_FROM_POINTS_CURSOR_KEY.as_bytes(),
                        &last.to_be_bytes(),
                    )
                    .map_err(|e| GraphError::New(e.to_string()))?;
            }
            if exhausted {
                self.backend
                    .put(
                        w,
                        Namespace::Metadata,
                        ADJ_BACKFILL_FROM_POINTS_COMPLETE_KEY.as_bytes(),
                        b"1",
                    )
                    .map_err(|e| GraphError::New(e.to_string()))?;
                self.backend
                    .delete(
                        w,
                        Namespace::Metadata,
                        ADJ_BACKFILL_FROM_POINTS_CURSOR_KEY.as_bytes(),
                    )
                    .map_err(|e| GraphError::New(e.to_string()))?;
            }

            Ok(edges_unchanged)
        })?;

        // Only now that the store holds the pass's mark.
        if exhausted {
            if let Err(error) = self.send_adjacency_backfill_mark(true) {
                tracing::warn!(
                    collection = self.collection_display_name(),
                    error = %error,
                    "a finished adjacency backfill's mark was not put in the metadata sidecar"
                );
            }
        }

        Ok((
            nodes_upserted,
            edges_upserted,
            exhausted,
            last_node_id,
            edges_unchanged,
        ))
    }

    /// The adjacency backfill's mark as `r` reads it from the store.
    fn adjacency_backfill_marked_be(&self, r: &AnyRead<'_>) -> Result<bool, GraphError> {
        self.backend
            .get_with(
                r,
                Namespace::Metadata,
                ADJ_BACKFILL_FROM_POINTS_COMPLETE_KEY.as_bytes(),
                |v| matches!(v, Some(b"1")),
            )
            .map_err(|e| GraphError::New(e.to_string()))
    }

    /// Incrementally rebuild native adjacency for a delta of changed source
    /// paths, deriving edges from the `_graph` relationship points already
    /// stored on S3.
    ///
    /// This is the incremental counterpart to
    /// [`backfill_adjacency_from_points_batch`]: instead of a full one-shot scan
    /// gated by a completion marker (which forces a `force` full rebuild to pick
    /// up later writes), the client calls this after re-indexing a delta of changed
    /// files. It pairs with `delete_by_path(s)` — which the client already calls to wipe
    /// stale edges before re-ingesting: delete clears the old adjacency for the
    /// changed paths, then this re-derives the current adjacency for just those
    /// paths from the relationship points. Edge ids are deterministic, so the
    /// re-derive is idempotent. Cost is O(scanned nodes) read + O(changed edges)
    /// write per re-index event — no full rebuild, no per-edge dual-write — and
    /// the adjacency lands in the same SlateDB namespaces on S3 that readers
    /// already traverse and cache, exactly like the vector index.
    ///
    /// Also collects chunk-edge candidates (see [`ChunkEdgeCandidate`]) from
    /// the same scan; the caller applies them to a base collection. Unlike the
    /// native Symbol-node edges above, these chunk edges are NOT wiped by the
    /// `delete_by_path(s)` this function pairs with — they clean up via the
    /// base collection's point-delete cascade instead, so a stale chunk edge
    /// can briefly outlive a re-indexed path until that cascade catches up.
    ///
    /// With `replace_outgoing` the delete is part of this call: the edges
    /// that start in `paths` and that no relationship point derives any more
    /// are dropped in the write batch that upserts the derived ones, with the
    /// nodes those drops leave without an edge. The batch is stored whole or
    /// not at all, so a reader sees the paths' old outgoing edges and then
    /// the new ones, never none. Edges that other files have into the paths
    /// are derived from those files' points and stay. LSM backend only: on
    /// LMDB the call rebuilds as it does without the option. With no path
    /// nothing is read or written, and `replaced` still says whether the
    /// option is honoured here: how a caller asks before it relies on it.
    ///
    /// The nodes are worked out in the write batch, once its drops are staged
    /// ([`Self::nodes_orphaned_by_dropping_be`]), on what other writers have
    /// committed by then.
    ///
    /// On LSM an edge that is stored exactly as its point derives it is left
    /// ([`Self::upsert_derived_edges_be`]), with or without `replace_outgoing`:
    /// a rebuild of rows that did not change stages nothing for them, and one
    /// that changes nothing writes no batch. `edges_unchanged` counts them.
    ///
    /// shortcut: the edges to drop and to upsert are chosen on the snapshot
    /// the points are read from, and written later. A rebuild of the same
    /// paths that commits in between is written over with what the older
    /// points derive: an edge it derived again is dropped. The client writes a
    /// collection's graph under one lock; closing it here needs a
    /// per-collection write gate.
    pub(crate) fn rebuild_adjacency_for_paths(
        &self,
        collection: &str,
        paths: &[String],
        replace_outgoing: bool,
    ) -> Result<PathsRebuild, GraphError> {
        let replace_outgoing = replace_outgoing && self.backend.kind() == BackendKind::Lsm;
        if paths.is_empty() {
            return Ok(PathsRebuild {
                replaced: replace_outgoing.then_some((0, 0)),
                ..Default::default()
            });
        }
        let path_set: std::collections::HashSet<&str> = paths.iter().map(|s| s.as_str()).collect();

        let (node_upserts, edge_upserts, chunk_candidates, stale, ends) =
            self.with_read_backend(|r| {
                let mut node_upserts: Vec<NodeUpsert> = Vec::new();
                let mut edge_upserts: Vec<EdgeUpsert> = Vec::new();
                let mut chunk_candidates: Vec<ChunkEdgeCandidate> = Vec::new();

                let mut derive = |key: &[u8], value: &[u8]| {
                    let id = match <[u8; 16]>::try_from(key) {
                        Ok(bytes) => u128::from_be_bytes(bytes),
                        Err(_) => return,
                    };
                    let node = match SerializedNode::decode_node(value, id) {
                        Ok(n) => n,
                        Err(_) => return,
                    };

                    // Only re-derive relationship points whose source file is in
                    // the changed set; every other node keeps its existing
                    // adjacency untouched (no rewrite, no churn).
                    let caller_path = match node.properties.get("caller_path") {
                        Some(Value::String(s)) => s.as_str(),
                        _ => return,
                    };
                    if !path_set.contains(caller_path) {
                        return;
                    }

                    if let Some((caller, callee, edge)) =
                        Self::relationship_upserts_from_node(collection, &node.properties)
                    {
                        node_upserts.push(caller);
                        node_upserts.push(callee);
                        edge_upserts.push(edge);
                    }
                    if let Some(candidate) = Self::chunk_edge_candidate_from_node(&node.properties)
                    {
                        chunk_candidates.push(candidate);
                    }
                };

                // A keyword index on `caller_path` names the paths' relationship
                // points; without one every node is read to find them. The ids
                // come back sorted, the order the scan meets them in, and each
                // node is checked against the paths again, so both ways derive
                // the same rows in the same order.
                if self.has_payload_index_at("caller_path", r) == Some(PayloadIndexSchema::Keyword)
                {
                    let mut ids = Vec::new();
                    for path in &path_set {
                        ids.extend(self.get_nodes_by_payload_value_be(
                            r,
                            "caller_path",
                            &Value::String((*path).to_string()),
                        )?);
                    }
                    ids.sort_unstable();
                    ids.dedup();
                    for id in ids {
                        let key = id.to_be_bytes();
                        self.backend
                            .get_with(r, Namespace::Nodes, &key, |value| {
                                if let Some(value) = value {
                                    derive(&key[..], value);
                                }
                            })
                            .map_err(|e| GraphError::New(e.to_string()))?;
                    }
                } else {
                    self.backend
                        .scan(r, Namespace::Nodes, KeyRange::all(), |key, value| {
                            derive(key, value);
                            true
                        })
                        .map_err(|e| GraphError::New(e.to_string()))?;
                }

                // What the derived edges replace: the paths' other outgoing
                // edges. Of their ends, the write batch drops those that no
                // edge is left at.
                let mut stale = std::collections::BTreeMap::new();
                let mut ends: Vec<u128> = Vec::new();
                if replace_outgoing {
                    let derived: std::collections::HashSet<u128> =
                        edge_upserts.iter().map(|edge| edge.id).collect();
                    stale = self.stale_edges_starting_in_paths_be(r, &path_set, &derived)?;
                    // A derived edge keeps both its ends.
                    let kept: std::collections::HashSet<u128> = edge_upserts
                        .iter()
                        .flat_map(|edge| [edge.from_node, edge.to_node])
                        .collect();
                    ends = stale
                        .values()
                        .flat_map(|(from, to)| [*from, *to])
                        .filter(|id| !kept.contains(id))
                        .collect();
                    ends.sort_unstable();
                    ends.dedup();
                }

                Ok((node_upserts, edge_upserts, chunk_candidates, stale, ends))
            })?;

        let nodes_upserted = node_upserts.len();
        let edges_upserted = edge_upserts.len();
        let leave_as_derived = self.leaves_edges_stored_as_derived();

        let (edges_unchanged, edges_deleted, nodes_deleted) = self.with_write_backend(|w| {
            let mut edges_deleted = 0usize;
            for edge_id in stale.keys() {
                match self.drop_edge_be(w, edge_id) {
                    Ok(()) => edges_deleted += 1,
                    // Gone since it was read.
                    Err(GraphError::EdgeNotFound) => {}
                    Err(error) => return Err(error),
                }
            }
            for node in &node_upserts {
                self.upsert_node_be(w, node)?;
            }
            let edges_unchanged = if leave_as_derived {
                self.upsert_derived_edges_be(w, &edge_upserts)?
            } else {
                for edge in &edge_upserts {
                    self.upsert_edge_be(w, edge)?;
                }
                0
            };
            let mut orphans = self.nodes_orphaned_by_dropping_be(w, &ends, &stale)?;
            if !orphans.is_empty() {
                // A node whose row does not decode cannot be dropped with
                // its index entries: it stays, as `delete_by_paths` leaves it.
                let r = self
                    .backend
                    .begin_read()
                    .map_err(|e| GraphError::New(e.to_string()))?;
                orphans.retain(|id| {
                    !matches!(
                        self.get_node_be(&r, id),
                        Err(GraphError::ConversionError(_))
                    )
                });
            }
            let nodes_deleted = self.drop_node_rows_be(w, &orphans)?;
            Ok((edges_unchanged, edges_deleted, nodes_deleted))
        })?;

        Ok(PathsRebuild {
            nodes_upserted,
            edges_upserted,
            edges_unchanged,
            chunk_candidates,
            replaced: replace_outgoing.then_some((edges_deleted, nodes_deleted)),
        })
    }

    /// Build a chunk-edge candidate from a relationship point payload carrying
    /// `caller_point_id`/`callee_point_id` — the client-resolved Qdrant point ids
    /// of the caller/callee chunks. Returns `None` unless BOTH ids are present
    /// and non-empty; the client only sets `callee_point_id` when it resolved the
    /// callee to a repo-internal chunk.
    fn chunk_edge_candidate_from_node(
        properties: &HashMap<String, Value>,
    ) -> Option<ChunkEdgeCandidate> {
        let caller_point_id = match properties.get("caller_point_id") {
            Some(Value::String(s)) if !s.is_empty() => s.clone(),
            _ => return None,
        };
        let callee_point_id = match properties.get("callee_point_id") {
            Some(Value::String(s)) if !s.is_empty() => s.clone(),
            _ => return None,
        };
        let caller_path = match properties.get("caller_path") {
            Some(Value::String(s)) => s.clone(),
            _ => String::new(),
        };
        let callee_path = match properties.get("callee_path") {
            Some(Value::String(s)) => s.clone(),
            _ => String::new(),
        };

        Some(ChunkEdgeCandidate {
            edge_type: Self::normalize_relationship_edge_type(properties),
            caller_point_id,
            callee_point_id,
            caller_path,
            callee_path,
        })
    }

    /// Normalize a relationship point's `edge_type` property to one of
    /// `CALLS`/`IMPORTS`/`INHERITS_FROM` (case-insensitive), defaulting to
    /// `CALLS` when absent or unrecognized.
    fn normalize_relationship_edge_type(properties: &HashMap<String, Value>) -> String {
        let edge_type = match properties.get("edge_type") {
            Some(Value::String(s)) => s.to_uppercase(),
            _ => "CALLS".to_string(),
        };
        match edge_type.as_str() {
            "CALLS" | "IMPORTS" | "INHERITS_FROM" => edge_type,
            _ => "CALLS".to_string(),
        }
    }

    /// Build caller/callee Symbol node upserts and a deterministic edge upsert
    /// from a relationship point payload. Returns `None` if the payload does not
    /// describe a relationship (no `caller_symbol`/`callee_symbol`).
    fn relationship_upserts_from_node(
        collection: &str,
        properties: &HashMap<String, Value>,
    ) -> Option<(NodeUpsert, NodeUpsert, EdgeUpsert)> {
        let caller_symbol = match properties.get("caller_symbol") {
            Some(Value::String(s)) if !s.is_empty() => s.clone(),
            _ => return None,
        };
        let callee_symbol = match properties.get("callee_symbol") {
            Some(Value::String(s)) if !s.is_empty() => s.clone(),
            _ => return None,
        };
        let caller_path = match properties.get("caller_path") {
            Some(Value::String(s)) => s.clone(),
            _ => String::new(),
        };
        let callee_path = match properties.get("callee_path") {
            Some(Value::String(s)) => s.clone(),
            _ => String::new(),
        };

        let edge_type = Self::normalize_relationship_edge_type(properties);

        let from_node =
            deterministic_id::node_id(collection, "Symbol", &caller_symbol, &caller_path);
        let to_node = deterministic_id::node_id(collection, "Symbol", &callee_symbol, &callee_path);
        let id = deterministic_id::edge_id(
            collection,
            &edge_type,
            &caller_symbol,
            &callee_symbol,
            &caller_path,
            &callee_path,
        );

        let caller = NodeUpsert {
            id: from_node,
            label: "Symbol".to_string(),
            properties: HashMap::from([
                ("name".to_string(), Value::String(caller_symbol.clone())),
                ("path".to_string(), Value::String(caller_path.clone())),
                ("label".to_string(), Value::String("Symbol".to_string())),
                (
                    "collection".to_string(),
                    Value::String(collection.to_string()),
                ),
            ]),
        };

        let callee = NodeUpsert {
            id: to_node,
            label: "Symbol".to_string(),
            properties: HashMap::from([
                ("name".to_string(), Value::String(callee_symbol.clone())),
                ("path".to_string(), Value::String(callee_path.clone())),
                ("label".to_string(), Value::String("Symbol".to_string())),
                (
                    "collection".to_string(),
                    Value::String(collection.to_string()),
                ),
            ]),
        };

        let mut edge_properties = properties.clone();
        edge_properties.insert("edge_type".to_string(), Value::String(edge_type.clone()));
        edge_properties.insert("from_name".to_string(), Value::String(caller_symbol));
        edge_properties.insert("to_name".to_string(), Value::String(callee_symbol));
        edge_properties.insert("from_path".to_string(), Value::String(caller_path));
        edge_properties.insert("to_path".to_string(), Value::String(callee_path));

        Some((
            caller,
            callee,
            EdgeUpsert {
                id,
                label: edge_type,
                from_node,
                to_node,
                properties: edge_properties,
            },
        ))
    }

    pub(crate) fn index_edge_paths(&self, txn: &mut RwTxn, edge: &Edge) -> Result<(), GraphError> {
        if let Some(path) = Self::edge_path_property(edge, "from_path") {
            self.lmdb_edge_path_idx()?.put(
                txn,
                &Self::edge_path_index_key(1, &path, &edge.id),
                &edge.id.to_be_bytes(),
            )?;
        }
        if let Some(path) = Self::edge_path_property(edge, "to_path") {
            self.lmdb_edge_path_idx()?.put(
                txn,
                &Self::edge_path_index_key(2, &path, &edge.id),
                &edge.id.to_be_bytes(),
            )?;
        }
        Ok(())
    }

    pub(crate) fn delete_edge_paths(&self, txn: &mut RwTxn, edge: &Edge) -> Result<(), GraphError> {
        if let Some(path) = Self::edge_path_property(edge, "from_path") {
            self.lmdb_edge_path_idx()?
                .delete(txn, &Self::edge_path_index_key(1, &path, &edge.id))?;
        }
        if let Some(path) = Self::edge_path_property(edge, "to_path") {
            self.lmdb_edge_path_idx()?
                .delete(txn, &Self::edge_path_index_key(2, &path, &edge.id))?;
        }
        Ok(())
    }

    pub(crate) fn payload_index_db_name(field: &str, schema: &PayloadIndexSchema) -> String {
        let kind = match schema {
            PayloadIndexSchema::Keyword => "kw",
            PayloadIndexSchema::Integer => "i64",
            PayloadIndexSchema::Float => "f64",
        };
        let encoded_field: String = field
            .as_bytes()
            .iter()
            .map(|byte| format!("{:02x}", byte))
            .collect();
        format!("pidx_{}_{}", kind, encoded_field)
    }

    /// Reader replicas only. A dense core opens its segment's local files of
    /// vectors by the segment's name (`VectorCore::new_named_lsm_with_dir`),
    /// and the names start again in a collection that is dropped and created
    /// again: the files a reader kept would score the new collection's
    /// points by the dropped one's vectors. So they are unlinked unless the
    /// local copy of the metadata, written at this reader's last open of
    /// the name, is of the database being opened: the same creation time.
    /// No copy says nothing, and is what the eviction of a reader that lost
    /// its database leaves.
    ///
    /// A reader that unlinked for nothing builds the files again from rows,
    /// as a new pod does.
    ///
    /// The marker of a fatal read error ([`Self::read_degraded_marker`]) goes
    /// with them: it was written for the database these files were of, the
    /// open below reads it, and nothing else on a reader takes it away.
    ///
    /// An error is a file that is still there, or a directory that could not
    /// be listed. The caller must not open then: the cores would open what
    /// stayed by its name.
    fn unlink_dense_files_of_another_database(
        path: &Path,
        metadata: &StorageMetadata,
    ) -> std::io::Result<()> {
        let same = Self::read_metadata_sidecar_from_path(path)
            .ok()
            .flatten()
            .is_some_and(|local| local.metadata.created_at_millis == metadata.created_at_millis);
        if same {
            return Ok(());
        }
        let entries = match fs::read_dir(path) {
            Ok(entries) => entries,
            // A reader's first open of the name: no directory, no files.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error),
        };
        let (mut unlinked, mut marker) = (0u64, false);
        let remove_stale = || -> std::io::Result<()> {
            for entry in entries {
                let file = entry?.path();
                let dense = matches!(
                    file.extension().and_then(|ext| ext.to_str()),
                    Some("hvec" | "hvs8" | "hvtq" | "hspn")
                );
                let is_marker = file.file_name().and_then(|name| name.to_str())
                    == Some(COLLECTION_DEGRADED_MARKER_FILE);
                if !dense && !is_marker {
                    continue;
                }
                match fs::remove_file(&file) {
                    Ok(()) if dense => unlinked += 1,
                    Ok(()) => marker = true,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error),
                }
            }
            Ok(())
        };
        let removed = remove_stale();
        if unlinked > 0 || marker {
            metrics::counter!("helix_reader_stale_dense_files_unlinked_total").increment(unlinked);
            tracing::info!(
                path = ?path,
                unlinked,
                degraded_marker = marker,
                "Unlinked a reader's local vector files: they are not of the collection that is at this name now"
            );
        }
        removed
    }

    fn new_lsm(
        path: &str,
        config: Config,
        wal: WalWriter,
        configured_secondary_indices: Vec<String>,
    ) -> Result<HelixGraphStorage, GraphError> {
        let open_start = std::time::Instant::now();
        let phase_start = std::time::Instant::now();
        let backend = Arc::new(
            AnyBackend::open_selected_for_collection(config.storage_backend, None, Path::new(path))
                .map_err(|e| GraphError::StorageError(e.to_string()))?,
        );
        metrics::histogram!("helix_lsm_collection_open_phase_ms", "phase" => "backend_open")
            .record(phase_start.elapsed().as_secs_f64() * 1000.0);
        tracing::info!(backend = ?backend.kind(), path = ?path, "storage backend selected");

        let phase_start = std::time::Instant::now();
        let read = backend
            .begin_read()
            .map_err(|e| GraphError::StorageError(e.to_string()))?;
        let overrides_started = std::time::Instant::now();
        let hnsw_overrides = backend
            .get_with(
                &read,
                Namespace::Metadata,
                HNSW_OVERRIDES_KEY.as_bytes(),
                |bytes| bytes.and_then(|b| bincode::deserialize::<HnswOverrides>(b).ok()),
            )
            .map_err(|e| GraphError::StorageError(e.to_string()))?;
        metrics::histogram!(
            "helix_lsm_metadata_read_step_ms",
            "step" => "hnsw_overrides"
        )
        .record(overrides_started.elapsed().as_secs_f64() * 1000.0);
        let metadata_started = std::time::Instant::now();
        let metadata = backend
            .get_with(
                &read,
                Namespace::Metadata,
                METADATA_CURRENT_KEY.as_bytes(),
                |bytes| bytes.map(Self::try_deserialize_metadata).transpose(),
            )
            .map_err(|e| GraphError::StorageError(e.to_string()))??
            .unwrap_or_else(|| StorageMetadata::new(configured_secondary_indices.clone()));
        metrics::histogram!(
            "helix_lsm_metadata_read_step_ms",
            "step" => "metadata_current"
        )
        .record(metadata_started.elapsed().as_secs_f64() * 1000.0);
        // One key more beside the metadata row: the collection's info and
        // its sidecar say it from memory from here on.
        let mark_started = std::time::Instant::now();
        let adjacency_backfill_complete = Self::adjacency_mark_read_at_open(backend.get_with(
            &read,
            Namespace::Metadata,
            ADJ_BACKFILL_FROM_POINTS_COMPLETE_KEY.as_bytes(),
            |value| matches!(value, Some(b"1")),
        ));
        metrics::histogram!(
            "helix_lsm_metadata_read_step_ms",
            "step" => "adjacency_backfill_mark"
        )
        .record(mark_started.elapsed().as_secs_f64() * 1000.0);
        // A writer keeps the record of the dense segments that left a stored
        // layout (`DEAD_SEGMENT_KEY_PREFIX`), from the read its layout is
        // from: a few keys beside the metadata row, read a block at a time.
        // A reader replica allocates and stores nothing and reads none.
        let dead_segments = if backend.is_reader_replica() || !dead_segment_names_enabled() {
            None
        } else {
            let dead_started = std::time::Instant::now();
            let mut rows = Vec::new();
            backend
                .scan_few(
                    &read,
                    Namespace::Metadata,
                    KeyRange::prefix(DEAD_SEGMENT_KEY_PREFIX),
                    |key, value| {
                        rows.push((key.to_vec(), value.to_vec()));
                        true
                    },
                )
                .map_err(|e| GraphError::StorageError(e.to_string()))?;
            metrics::histogram!(
                "helix_lsm_metadata_read_step_ms",
                "step" => "dense_dead_names"
            )
            .record(dead_started.elapsed().as_secs_f64() * 1000.0);
            Some(rows)
        };
        drop(read);
        metrics::histogram!("helix_lsm_collection_open_phase_ms", "phase" => "metadata_read")
            .record(phase_start.elapsed().as_secs_f64() * 1000.0);

        let phase_start = std::time::Instant::now();
        // Before the local copy of the metadata is written over, and before
        // the dense cores open what the directory holds. An open that fails
        // here leaves that copy as it was, so the next one tries again.
        if backend.is_reader_replica() {
            Self::unlink_dense_files_of_another_database(Path::new(path), &metadata).map_err(
                |error| {
                    tracing::warn!(
                        path = ?path,
                        %error,
                        "A reader could not remove local files of another collection at this name: the open fails, and the next one tries again"
                    );
                    GraphError::StorageError(format!(
                        "reader could not remove local files left by another collection at this name: {error}"
                    ))
                },
            )?;
        }
        if let Err(err) = Self::persist_metadata_sidecar_at_path(
            Path::new(path),
            &metadata,
            None,
            Some(adjacency_backfill_complete),
        ) {
            metrics::counter!("helix_metadata_sidecar_write_errors_total").increment(1);
            tracing::warn!(
                path = ?path,
                error = ?err,
                "Failed to persist LSM collection metadata sidecar during open"
            );
        }
        metrics::histogram!(
            "helix_lsm_collection_open_phase_ms",
            "phase" => "metadata_sidecar"
        )
        .record(phase_start.elapsed().as_secs_f64() * 1000.0);

        let hnsw_config = HNSWConfig::with_overrides(
            config.vector_config.m,
            config.vector_config.ef_construction,
            config.vector_config.ef_search,
            hnsw_overrides.as_ref(),
        );
        let phase_start = std::time::Instant::now();
        let vectors = VectorCore::new_lsm(hnsw_config.clone(), Arc::clone(&backend))?;
        metrics::histogram!("helix_lsm_collection_open_phase_ms", "phase" => "base_vector_core")
            .record(phase_start.elapsed().as_secs_f64() * 1000.0);

        let collection_name = Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        let named_vectors = NamedVectorManager::new(collection_name);
        named_vectors.attach_backend(Arc::clone(&backend));
        if let Some(threshold_override) = metadata.indexing_threshold_override {
            named_vectors.set_indexing_threshold(threshold_override);
        }

        let phase_start = std::time::Instant::now();
        for (name, vector_config) in metadata.named_vectors.clone() {
            named_vectors.load_vector_index_lsm(
                Path::new(path),
                &name,
                vector_config,
                hnsw_config.clone(),
                metadata.dense_vector_spaces.get(&name).cloned(),
            )?;
        }
        if let Some(rows) = &dead_segments {
            named_vectors.load_dead_segments(rows)?;
        }
        metrics::histogram!("helix_lsm_collection_open_phase_ms", "phase" => "dense_vectors")
            .record(phase_start.elapsed().as_secs_f64() * 1000.0);

        let phase_start = std::time::Instant::now();
        for (name, sparse_config) in metadata.sparse_vectors.clone() {
            named_vectors.load_sparse_index_lsm(&name, sparse_config)?;
        }
        metrics::histogram!("helix_lsm_collection_open_phase_ms", "phase" => "sparse_vectors")
            .record(phase_start.elapsed().as_secs_f64() * 1000.0);

        let phase_start = std::time::Instant::now();
        let mut payload_indices = HashMap::new();
        for (field, schema) in metadata.payload_indices.clone() {
            payload_indices.insert(field, PayloadIndexHandle::ready_lsm(schema));
        }

        let mut multi_indices = HashMap::new();
        multi_indices.insert("name".to_string(), None);
        metrics::histogram!("helix_lsm_collection_open_phase_ms", "phase" => "payload_indices")
            .record(phase_start.elapsed().as_secs_f64() * 1000.0);

        let phase_start = std::time::Instant::now();
        let degraded_reason = Self::read_degraded_marker(path);
        let collection_degraded = degraded_reason.is_some();
        if collection_degraded {
            tracing::error!(
                path = ?path,
                "opening collection already marked degraded; background vector maintenance will be skipped"
            );
        }
        metrics::histogram!("helix_lsm_collection_open_phase_ms", "phase" => "degraded_marker")
            .record(phase_start.elapsed().as_secs_f64() * 1000.0);

        tracing::debug!(
            path = ?path,
            elapsed_ms = %open_start.elapsed().as_secs_f64() * 1000.0,
            "LSM collection open complete without local LMDB env"
        );

        Ok(Self {
            collection_path: PathBuf::from(path),
            backend,
            graph_env: None,
            nodes_db: None,
            edges_db: None,
            out_edges_db: None,
            in_edges_db: None,
            edge_path_idx: None,
            metadata_db: None,
            metadata_snapshot: RwLock::new(metadata),
            metadata_snapshot_read_at: parking_lot::Mutex::new(None),
            last_lsm_sidecar_upload_ms: AtomicU64::new(0),
            last_lsm_sidecar_fingerprint: AtomicU64::new(0),
            lsm_sidecar_put_held: std::sync::atomic::AtomicBool::new(false),
            sidecar_put: parking_lot::Mutex::new(()),
            adjacency_backfill_mark: std::sync::atomic::AtomicU8::new(adjacency_mark_code(Some(
                adjacency_backfill_complete,
            ))),
            adjacency_mark_sent: std::sync::atomic::AtomicU8::new(0),
            secondary_indices: configured_secondary_indices
                .into_iter()
                .map(|name| (name, None))
                .collect(),
            multi_indices,
            payload_indices: RwLock::new(payload_indices),
            text_walk_cache: parking_lot::Mutex::new(
                super::text_walk_cache::TextWalkCache::from_env(),
            ),
            vectors,
            named_vectors,
            wal,
            optimizer_running: std::sync::atomic::AtomicBool::new(false),
            last_upsert_at: std::sync::atomic::AtomicU64::new(0),
            point_mutations: std::sync::atomic::AtomicU64::new(0),
            point_mutations_in_flight: AtomicUsize::new(0),
            resize_gate: parking_lot::RwLock::new(()),
            resize_writers_pending: AtomicUsize::new(0),
            resize_writers_active: AtomicUsize::new(0),
            resize_wait_lock: Mutex::new(()),
            resize_wait_cv: Condvar::new(),
            active_lmdb_txns: AtomicUsize::new(0),
            write_txn_gate: Arc::new(Mutex::new(())),
            payload_index_admin_gate: Mutex::new(()),
            payload_index_builds_pending: AtomicUsize::new(0),
            payload_index_build_lock: Mutex::new(()),
            metadata_corruption_logged: std::sync::atomic::AtomicBool::new(false),
            collection_degraded: std::sync::atomic::AtomicBool::new(collection_degraded),
            closing_for_shutdown: std::sync::atomic::AtomicBool::new(false),
            degraded_reason: RwLock::new(degraded_reason),
            dense_metadata_dirty: std::sync::atomic::AtomicBool::new(false),
            last_reader_refresh_ms: std::sync::atomic::AtomicU64::new(0),
            reader_refresh_inflight: std::sync::atomic::AtomicBool::new(false),
            last_reader_read_at_ms: std::sync::atomic::AtomicU64::new(0),
            reader_poll_tier_fast: std::sync::atomic::AtomicBool::new(
                crate::helix_engine::storage_core::backend_lsm::reader_cold_open_starts_fast(),
            ),
            reader_poll_tier_transition: std::sync::atomic::AtomicBool::new(false),
            last_reader_promote_failed_at_ms: AtomicU64::new(0),
            reader_promote_consecutive_failures: std::sync::atomic::AtomicU32::new(0),
            last_write_promoted_at_ms: std::sync::atomic::AtomicU64::new(0),
            reader_view_reconcile: ReaderViewReconcile::default(),
            recount_inflight: std::sync::atomic::AtomicBool::new(false),
            payload_index_gc_inflight: std::sync::atomic::AtomicBool::new(false),
        })
    }

    pub fn new(path: &str, config: Config) -> Result<HelixGraphStorage, GraphError> {
        let _open_start = Instant::now();
        fs::create_dir_all(path)?;
        let wal = WalWriter::open(Path::new(path).join("wal"), SyncPolicy::default())?;
        let configured_secondary_indices = config
            .graph_config
            .secondary_indices
            .clone()
            .unwrap_or_default();

        let backend_kind = config.storage_backend.kind();
        if backend_kind == BackendKind::Lsm {
            return Self::new_lsm(path, config, wal, configured_secondary_indices);
        }

        // Configure and open LMDB environment.
        //
        // Start small and grow under resize_gate protection. Opening every
        // collection at the ceiling makes data.mdb physically large on ext4
        // and turns a 6 GB compact database into a 32+ GB live file. Existing
        // collections keep at least their current data.mdb length so restarts
        // never try to open an env below its on-disk map size.
        let initial_map_bytes = Self::initial_map_size(path, &config);
        let max_readers = std::env::var("HELIX_MAX_READERS")
            .ok()
            .and_then(|value| value.trim().parse::<u32>().ok())
            .filter(|value| *value >= 2048)
            .unwrap_or(2048);
        // HELIX_LMDB_NOSYNC=1 enables MDB_NOSYNC + MDB_NOMETASYNC, which
        // skips LMDB's per-commit fsync and meta-page flush. That is only
        // safe when an independent fsynced recovery journal is enabled.
        // The old write-queue spool is intentionally disabled on the direct
        // submit path, so HELIX_WRITE_QUEUE_DIR alone is not proof that
        // acknowledged writes are recoverable after a kernel panic.
        let lmdb_nosync_requested = std::env::var("HELIX_LMDB_NOSYNC")
            .map(|v| matches!(v.trim(), "1" | "true" | "True" | "TRUE"))
            .unwrap_or(false);
        let durable_journal_enabled = std::env::var("HELIX_DURABLE_WRITE_JOURNAL")
            .ok()
            .map(|v| matches!(v.trim(), "1" | "true" | "True" | "TRUE"))
            .unwrap_or(false);
        let lmdb_nosync = lmdb_nosync_requested && durable_journal_enabled;
        if lmdb_nosync_requested && !durable_journal_enabled {
            tracing::warn!(
                path = ?path,
                "HELIX_LMDB_NOSYNC=1 ignored: no fsynced durable write journal is enabled. \
                 Per-commit fsync remains enabled to prevent silent data-loss-on-crash."
            );
        }
        let mut env_flags = EnvFlags::empty();
        if lmdb_nosync {
            env_flags |= EnvFlags::NO_SYNC | EnvFlags::NO_META_SYNC;
        }
        let max_dbs = Self::max_dbs();

        let open_start = std::time::Instant::now();
        let graph_env = unsafe {
            let mut opts = EnvOpenOptions::new();
            opts.map_size(initial_map_bytes)
                // Dense vector segmenting opens several LMDB databases per physical
                // segment (~5 per segment × named-vectors + base DBs + payload indices).
                // Retired segment cleanup clears those DBs in chunks instead of deleting
                // and closing their handles, so DBI slots are reused only when segment
                // names are reused. Keep enough headroom for churned collections.
                .max_dbs(max_dbs)
                .max_readers(max_readers);
            if !env_flags.is_empty() {
                opts.flags(env_flags);
            }
            opts.open(Path::new(path))?
        };
        if lmdb_nosync {
            tracing::info!(
                "LMDB env opened with NO_SYNC + NO_META_SYNC (HELIX_DURABLE_WRITE_JOURNAL=1 — operator opt-in)"
            );
        }
        tracing::debug!(path = ?path, elapsed_ms = %open_start.elapsed().as_secs_f64() * 1000.0, "LMDB env open complete");

        // Build the shared backend handle once (US-006). It is `Arc`-shared
        // into the base VectorCore, the NamedVectorManager, and every dense
        // core created below, because `AnyBackend` is not `Clone` and the
        // `Lsm` variant cannot be re-created from an env. LMDB remains opened
        // during cutover because older call sites still carry heed txns/handles;
        // `HELIX_STORAGE_BACKEND=lsm` selects SlateDB for backend-routed paths.
        let backend = Arc::new(
            AnyBackend::open_selected_for_collection(
                config.storage_backend,
                Some(graph_env.clone()),
                Path::new(path),
            )
            .map_err(|e| GraphError::StorageError(e.to_string()))?,
        );
        tracing::info!(backend = ?backend.kind(), path = ?path, "storage backend selected");

        let mut wtxn = graph_env.write_txn()?;

        // Create/open all necessary databases
        let nodes_db = graph_env
            .database_options()
            .types::<U128<BE>, Bytes>()
            //.flags(DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED) // NOTE: commend out because add_n gave error upon inserting
            .name(DB_NODES)
            .create(&mut wtxn)?;
        let edges_db = graph_env
            .database_options()
            .types::<U128<BE>, Bytes>()
            // .flags(DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED)
            .name(DB_EDGES)
            .create(&mut wtxn)?;
        let out_edges_db: Database<Bytes, Bytes> = graph_env
            .database_options()
            .types::<Bytes, Bytes>()
            .flags(DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED)
            .name(DB_OUT_EDGES)
            .create(&mut wtxn)?;
        let in_edges_db: Database<Bytes, Bytes> = graph_env
            .database_options()
            .types::<Bytes, Bytes>()
            .flags(DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED)
            .name(DB_IN_EDGES)
            .create(&mut wtxn)?;
        let edge_path_idx: Database<Bytes, Bytes> = graph_env
            .database_options()
            .types::<Bytes, Bytes>()
            .name(DB_EDGE_PATH_IDX)
            .create(&mut wtxn)?;
        let metadata_db: Database<Str, Bytes> = graph_env
            .database_options()
            .types::<Str, Bytes>()
            .name(DB_METADATA)
            .create(&mut wtxn)?;
        // Create secondary indices
        let mut secondary_indices = HashMap::new();
        for index in &configured_secondary_indices {
            secondary_indices.insert(
                index.clone(),
                Some(graph_env.create_database(&mut wtxn, Some(index.as_str()))?),
            );
        }
        tracing::debug!(path = ?path, elapsed_ms = %open_start.elapsed().as_secs_f64() * 1000.0, "LMDB DB handles + metadata loaded");

        let (backend_hnsw_overrides, backend_metadata) = if backend.kind() == BackendKind::Lsm {
            let read = backend
                .begin_read()
                .map_err(|e| GraphError::StorageError(e.to_string()))?;
            let hnsw = backend
                .get_with(
                    &read,
                    Namespace::Metadata,
                    HNSW_OVERRIDES_KEY.as_bytes(),
                    |bytes| bytes.and_then(|b| bincode::deserialize::<HnswOverrides>(b).ok()),
                )
                .map_err(|e| GraphError::StorageError(e.to_string()))?;
            let metadata = backend
                .get_with(
                    &read,
                    Namespace::Metadata,
                    METADATA_CURRENT_KEY.as_bytes(),
                    |bytes| bytes.map(Self::try_deserialize_metadata).transpose(),
                )
                .map_err(|e| GraphError::StorageError(e.to_string()))??;
            (hnsw, metadata)
        } else {
            (None, None)
        };

        // Read per-collection HNSW overrides (if any) before constructing
        // vector indexes. Missing key = fall back to global config. Decode
        // errors are treated as "no overrides" rather than fatal, matching
        // the self-healing metadata convention.
        let hnsw_overrides: Option<HnswOverrides> = if let Some(overrides) = backend_hnsw_overrides
        {
            let bytes = bincode::serialize(&overrides)?;
            metadata_db.put(&mut wtxn, HNSW_OVERRIDES_KEY, &bytes)?;
            Some(overrides)
        } else {
            metadata_db
                .get(&wtxn, HNSW_OVERRIDES_KEY)?
                .and_then(|bytes| bincode::deserialize::<HnswOverrides>(bytes).ok())
        };

        let vectors = VectorCore::new(
            &graph_env,
            &mut wtxn,
            HNSWConfig::with_overrides(
                config.vector_config.m,
                config.vector_config.ef_construction,
                config.vector_config.ef_search,
                hnsw_overrides.as_ref(),
            ),
            Arc::clone(&backend),
        )?;

        // Initialize or heal metadata. If bytes are missing OR fail strict
        // decode (e.g. bincode schema drift from a previous build), persist a
        // fresh default within this same write txn. Fixing the on-disk bytes
        // at open time means subsequent reads never trigger corruption
        // detection on the hot path.
        let mut metadata: StorageMetadata = if let Some(meta) = backend_metadata {
            metadata_db.put(
                &mut wtxn,
                METADATA_CURRENT_KEY,
                &Self::serialize_metadata(&meta)?,
            )?;
            meta
        } else {
            match metadata_db.get(&wtxn, METADATA_CURRENT_KEY)? {
                None => {
                    let fresh = StorageMetadata::new(configured_secondary_indices);
                    metadata_db.put(
                        &mut wtxn,
                        METADATA_CURRENT_KEY,
                        &Self::serialize_metadata(&fresh)?,
                    )?;
                    fresh
                }
                Some(bytes) => match Self::try_deserialize_metadata(bytes) {
                    Ok(meta) => meta,
                    Err(e) => {
                        tracing::warn!(
                            "Corrupted metadata at collection open ({:?}); healing to defaults: {}",
                            path,
                            e
                        );
                        let fresh = StorageMetadata::new(Vec::new());
                        metadata_db.put(
                            &mut wtxn,
                            METADATA_CURRENT_KEY,
                            &Self::serialize_metadata(&fresh)?,
                        )?;
                        fresh
                    }
                },
            }
        };

        tracing::debug!(path = ?path, elapsed_ms = %open_start.elapsed().as_secs_f64() * 1000.0, "LMDB vectors + metadata healed");

        let mut payload_indices = HashMap::new();
        for (field, schema) in metadata.payload_indices.clone() {
            let handle = if backend.kind() == BackendKind::Lsm {
                PayloadIndexHandle::ready_lsm(schema)
            } else {
                let db = graph_env
                    .database_options()
                    .types::<Bytes, Bytes>()
                    .flags(DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED)
                    .name(&Self::payload_index_db_name(&field, &schema))
                    .create(&mut wtxn)?;
                PayloadIndexHandle::ready(schema, db)
            };
            payload_indices.insert(field, handle);
        }
        tracing::debug!(path = ?path, elapsed_ms = %open_start.elapsed().as_secs_f64() * 1000.0, "LMDB DB vectors + payload indices added");

        // Derive a collection identity from the trailing path component.
        // Used only for diagnostics (missing-segment log + metric labels),
        // so a lossy conversion / "unknown" fallback is fine and never
        // affects correctness.
        let collection_name = Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        let named_vectors = NamedVectorManager::new(collection_name);
        // Wire the graph adjacency DB into the manager before any dense
        // cores are created so graph-entangled HNSW (gated by
        // HELIX_GRAPH_ENTANGLED_HNSW) can use it on the search/insert hot
        // paths without opening a new LMDB env. `create_dense_core_attached`
        // clones this handle into every subsequently-built VectorCore.
        named_vectors.attach_graph_out_edges_db(out_edges_db);
        // Share the backend handle into the manager before any dense core is
        // created so `create_dense_core_attached`/`load_vector_index` thread it
        // into every VectorCore (US-006 plumbing).
        named_vectors.attach_backend(Arc::clone(&backend));
        if let Some(threshold_override) = metadata.indexing_threshold_override {
            named_vectors.set_indexing_threshold(threshold_override);
        }
        let mut dense_vector_spaces_repaired = false;
        for (name, vector_config) in metadata.named_vectors.clone() {
            dense_vector_spaces_repaired |= named_vectors.load_vector_index(
                &graph_env,
                &mut wtxn,
                &name,
                vector_config,
                HNSWConfig::with_overrides(
                    config.vector_config.m,
                    config.vector_config.ef_construction,
                    config.vector_config.ef_search,
                    hnsw_overrides.as_ref(),
                ),
                metadata.dense_vector_spaces.get(&name).cloned(),
            )?;
        }
        tracing::debug!(path = ?path, elapsed_ms = %open_start.elapsed().as_secs_f64() * 1000.0, "LMDB vectors + dense vectors");

        if dense_vector_spaces_repaired {
            let mut repaired_metadata = metadata.clone();
            repaired_metadata.set_dense_vector_spaces(named_vectors.list_dense_vector_spaces());
            metadata_db.put(
                &mut wtxn,
                METADATA_CURRENT_KEY,
                &Self::serialize_metadata(&repaired_metadata)?,
            )?;
            metadata = repaired_metadata;
            tracing::debug!(path = ?path, elapsed_ms = %open_start.elapsed().as_secs_f64() * 1000.0, "LMDB DB vectors + dense vectors re[aired");
        }

        for (name, sparse_config) in metadata.sparse_vectors.clone() {
            named_vectors.load_sparse_index(&graph_env, &mut wtxn, &name, sparse_config)?;
        }
        tracing::debug!(path = ?path, elapsed_ms = %open_start.elapsed().as_secs_f64() * 1000.0, "Vector indexes loaded");

        // Re-upsert tombstones: rebuild the superseded-copy set + in-memory
        // counts from live data now that every dense space + core is loaded.
        // This is the crash-safe reconstruction the tombstone design relies on
        // — a torn/empty `dense_tombstones` DB is fully repaired here, so the
        // on-disk format never has to change to survive a crash. No-op when the
        // kill switch is off. Failures are non-fatal: a missing reconstruction
        // only delays reclamation (search correctness never depends on it).
        if let Err(e) = named_vectors.reconstruct_tombstones(&mut wtxn) {
            tracing::warn!(error = %e, "reconstruct_tombstones on open failed (non-fatal)");
        }

        // Deferred-repair delete tombstones (HELIX_LSM_DELETE_TOMBSTONES,
        // issue #29 "fix 2"): rebuild each dense core's in-memory deleted-id
        // set from the durable keyspace. Unlike the re-upsert reconstruction
        // above, this runs unconditionally (independent of the flag's
        // current value) because the set gates search correctness, not just
        // reclamation — a flag flip to OFF after deletes exist must still
        // keep those ids out of search results. Failures are logged as a
        // warning (not just debug) for the same reason: unlike the re-upsert
        // path, a missed reconstruction here can surface already-deleted
        // vectors until the next successful open.
        if let Err(e) = named_vectors.reconstruct_delete_tombstones(&mut wtxn) {
            tracing::warn!(error = %e, "reconstruct_delete_tombstones on open failed (non-fatal, may surface deleted vectors until next open)");
        }

        // Auto-provision the `name` multi-index so graph queries can resolve
        // symbols by name without requiring a file path. This is a DUP_SORT
        // index mapping serialized Value::String(name) → [node_id, ...].
        let name_idx_db = graph_env
            .database_options()
            .types::<Bytes, Bytes>()
            .flags(DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED)
            .name("midx_name")
            .create(&mut wtxn)?;
        let mut multi_indices = HashMap::new();
        multi_indices.insert("name".to_string(), Some(name_idx_db));

        wtxn.commit()?;
        let degraded_reason = Self::read_degraded_marker(path);
        let collection_degraded = degraded_reason.is_some();
        if collection_degraded {
            tracing::error!(
                path = ?path,
                "opening collection already marked degraded; background vector maintenance will be skipped"
            );
        }

        let storage_result: Result<Self, GraphError> = Ok(Self {
            collection_path: PathBuf::from(path),
            backend,
            graph_env: Some(graph_env),
            nodes_db: Some(nodes_db),
            edges_db: Some(edges_db),
            out_edges_db: Some(out_edges_db),
            in_edges_db: Some(in_edges_db),
            edge_path_idx: Some(edge_path_idx),
            metadata_db: Some(metadata_db),
            metadata_snapshot: RwLock::new(metadata.clone()),
            metadata_snapshot_read_at: parking_lot::Mutex::new(None),
            last_lsm_sidecar_upload_ms: AtomicU64::new(0),
            last_lsm_sidecar_fingerprint: AtomicU64::new(0),
            lsm_sidecar_put_held: std::sync::atomic::AtomicBool::new(false),
            sidecar_put: parking_lot::Mutex::new(()),
            adjacency_backfill_mark: std::sync::atomic::AtomicU8::new(0),
            adjacency_mark_sent: std::sync::atomic::AtomicU8::new(0),
            secondary_indices,
            multi_indices,
            payload_indices: RwLock::new(payload_indices),
            text_walk_cache: parking_lot::Mutex::new(
                super::text_walk_cache::TextWalkCache::from_env(),
            ),
            vectors,
            named_vectors,
            wal,
            optimizer_running: std::sync::atomic::AtomicBool::new(false),
            last_upsert_at: std::sync::atomic::AtomicU64::new(0),
            point_mutations: std::sync::atomic::AtomicU64::new(0),
            point_mutations_in_flight: AtomicUsize::new(0),
            resize_gate: parking_lot::RwLock::new(()),
            resize_writers_pending: AtomicUsize::new(0),
            resize_writers_active: AtomicUsize::new(0),
            resize_wait_lock: Mutex::new(()),
            resize_wait_cv: Condvar::new(),
            active_lmdb_txns: AtomicUsize::new(0),
            write_txn_gate: Arc::new(Mutex::new(())),
            payload_index_admin_gate: Mutex::new(()),
            payload_index_builds_pending: AtomicUsize::new(0),
            payload_index_build_lock: Mutex::new(()),
            metadata_corruption_logged: std::sync::atomic::AtomicBool::new(false),
            collection_degraded: std::sync::atomic::AtomicBool::new(collection_degraded),
            closing_for_shutdown: std::sync::atomic::AtomicBool::new(false),
            degraded_reason: RwLock::new(degraded_reason),
            dense_metadata_dirty: std::sync::atomic::AtomicBool::new(false),
            last_reader_refresh_ms: std::sync::atomic::AtomicU64::new(0),
            reader_refresh_inflight: std::sync::atomic::AtomicBool::new(false),
            last_reader_read_at_ms: std::sync::atomic::AtomicU64::new(0),
            reader_poll_tier_fast: std::sync::atomic::AtomicBool::new(
                crate::helix_engine::storage_core::backend_lsm::reader_cold_open_starts_fast(),
            ),
            reader_poll_tier_transition: std::sync::atomic::AtomicBool::new(false),
            last_reader_promote_failed_at_ms: AtomicU64::new(0),
            reader_promote_consecutive_failures: std::sync::atomic::AtomicU32::new(0),
            last_write_promoted_at_ms: std::sync::atomic::AtomicU64::new(0),
            reader_view_reconcile: ReaderViewReconcile::default(),
            recount_inflight: std::sync::atomic::AtomicBool::new(false),
            payload_index_gc_inflight: std::sync::atomic::AtomicBool::new(false),
        });

        // WAL crash recovery: replay any committed transactions that may
        // not have been flushed to LMDB before a crash.
        'recover: {
            let storage = match &storage_result {
                Ok(s) => s,
                Err(_) => break 'recover,
            };
            let wal_dir = Path::new(path).join("wal");
            if !wal_dir.exists() {
                break 'recover;
            }
            let committed_lsn = match storage.with_read_txn(|rtxn| {
                Ok(storage
                    .lmdb_metadata_db()?
                    .get(rtxn, COMMITTED_LSN_KEY)?
                    .and_then(|bytes: &[u8]| {
                        if bytes.len() == 8 {
                            let mut buf = [0u8; 8];
                            buf.copy_from_slice(&bytes[..8]);
                            Some(u64::from_le_bytes(buf))
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0))
            }) {
                Ok(lsn) => lsn,
                Err(e) => {
                    // Skipping WAL replay is safer than aborting the pod
                    // here — the WAL itself is durable and the next
                    // successful read txn will pick up the same state.
                    tracing::error!(
                        error = %e,
                        "Failed to open read txn for WAL recovery; skipping replay"
                    );
                    break 'recover;
                }
            };
            match crate::helix_engine::storage_core::wal::recover(&wal_dir, committed_lsn) {
                Ok((report, batches)) => {
                    if report.replayed > 0 {
                        tracing::info!(
                            replayed = report.replayed,
                            skipped = report.skipped,
                            from_lsn = committed_lsn,
                            "WAL recovery"
                        );
                        for batch in batches {
                            if let Err(e) = storage.replay_wal_batch(&batch) {
                                tracing::error!(error = %e, "WAL replay error");
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "WAL recovery failed (non-fatal)");
                }
            }
            match storage.truncate_wal_at_committed_lsn() {
                Ok(removed) if removed > 0 => {
                    tracing::info!(removed, "WAL checkpoint truncation after recovery");
                    metrics::counter!("helix_wal_segments_truncated_total")
                        .increment(removed as u64);
                }
                Ok(_) => {}
                Err(e) => tracing::warn!(error = %e, "WAL checkpoint truncation failed"),
            }
        }

        if let Ok(storage) = &storage_result {
            match storage.metadata_snapshot() {
                Ok(metadata) => storage.persist_metadata_sidecar_best_effort(&metadata),
                Err(err) => tracing::warn!(
                    path = ?path,
                    error = ?err,
                    "Failed to read metadata snapshot for sidecar hydration"
                ),
            }
        }

        let elapsed = _open_start.elapsed();
        match &storage_result {
            Ok(_) => {
                tracing::info!(
                    path = ?path,
                    elapsed_ms = elapsed.as_secs_f64() * 1000.0,
                    "LMDB collection open complete"
                );
            }
            Err(e) => {
                tracing::warn!(
                    path = ?path,
                    elapsed_ms = elapsed.as_secs_f64() * 1000.0,
                    error = %e,
                    "LMDB collection open failed"
                );
            }
        }

        storage_result
    }

    /// Persist the committed WAL LSN in LMDB metadata so crash recovery
    /// knows where to start replaying from.
    pub fn set_committed_lsn(&self, txn: &mut RwTxn, lsn: u64) -> Result<(), GraphError> {
        self.lmdb_metadata_db()?
            .put(txn, COMMITTED_LSN_KEY, &lsn.to_le_bytes())?;
        // Gauge exposes the most recent durable LSN per process. Collection
        // label is omitted deliberately to bound cardinality — in practice
        // operators compare this against the WAL's current LSN to detect
        // fsync stalls. Monotonic, but we use `set` so restarts reset cleanly.
        metrics::gauge!("helix_wal_durable_lsn").set(lsn as f64);
        Ok(())
    }

    fn committed_lsn(&self) -> Result<u64, GraphError> {
        self.with_read_txn(|rtxn| {
            Ok(self
                .lmdb_metadata_db()?
                .get(rtxn, COMMITTED_LSN_KEY)?
                .and_then(|bytes: &[u8]| {
                    if bytes.len() == 8 {
                        let mut buf = [0u8; 8];
                        buf.copy_from_slice(&bytes[..8]);
                        Some(u64::from_le_bytes(buf))
                    } else {
                        None
                    }
                })
                .unwrap_or(0))
        })
    }

    fn checkpoint_wal_to_lsn(&self, lsn: u64) -> Result<u32, GraphError> {
        if lsn == 0 {
            return Ok(0);
        }
        if self.backend.kind() == BackendKind::Lsm {
            return Ok(0);
        }
        self.with_write_txn(|txn| self.set_committed_lsn(txn, lsn))?;
        self.lmdb_env()?.force_sync()?;
        self.wal.truncate(lsn, wal_keep_segments())
    }

    fn truncate_wal_at_committed_lsn(&self) -> Result<u32, GraphError> {
        if self.backend.kind() == BackendKind::Lsm {
            return Ok(0);
        }
        let committed_lsn = self.committed_lsn()?;
        if committed_lsn == 0 {
            return Ok(0);
        }
        self.lmdb_env()?.force_sync()?;
        self.wal.truncate(committed_lsn, wal_keep_segments())
    }

    /// Replay a batch of WAL entries into storage. Used during crash recovery.
    fn replay_wal_batch(
        &self,
        batch: &[crate::helix_engine::storage_core::wal::WalEntry],
    ) -> Result<(), GraphError> {
        use crate::helix_engine::storage_core::upsert::{EdgeUpsert, NodeUpsert};
        use crate::helix_engine::storage_core::wal::WalOp;
        let mut replayed_vectors = false;
        self.with_write_txn(|txn| {
            for entry in batch {
                match &entry.op {
                    WalOp::CreateNode {
                        id,
                        label,
                        properties_json,
                    }
                    | WalOp::UpsertNode {
                        id,
                        label,
                        properties_json,
                    } => {
                        let props: HashMap<String, Value> = if properties_json.is_empty() {
                            HashMap::new()
                        } else {
                            sonic_rs::from_slice(properties_json).unwrap_or_default()
                        };
                        let upsert = NodeUpsert {
                            id: *id,
                            label: label.clone(),
                            properties: props,
                        };
                        self.upsert_node(txn, &upsert)?;
                    }
                    WalOp::DropNode { id } => {
                        // drop_node returns Ok(()) when the node is absent (idempotent),
                        // so propagate all errors — none of them mean "not found".
                        self.drop_node(txn, id)?;
                    }
                    WalOp::CreateEdge {
                        id,
                        label,
                        from_node,
                        to_node,
                        properties_json,
                    }
                    | WalOp::UpsertEdge {
                        id,
                        label,
                        from_node,
                        to_node,
                        properties_json,
                    } => {
                        let props: HashMap<String, Value> = if properties_json.is_empty() {
                            HashMap::new()
                        } else {
                            sonic_rs::from_slice(properties_json).unwrap_or_default()
                        };
                        let upsert = EdgeUpsert {
                            id: *id,
                            label: label.clone(),
                            from_node: *from_node,
                            to_node: *to_node,
                            properties: props,
                        };
                        self.upsert_edge(txn, &upsert)?;
                    }
                    WalOp::DropEdge { id } => {
                        // Ignore EdgeNotFound — idempotent replay of an already-deleted edge.
                        // Propagate all other errors (e.g. MapFull) so they are not silently lost.
                        match self.drop_edge(txn, id) {
                            Ok(()) | Err(GraphError::EdgeNotFound) => {}
                            Err(e) => return Err(e),
                        }
                    }
                    WalOp::InsertVector {
                        id,
                        data,
                        fields_json,
                        named_index,
                    } => {
                        let fields: HashMap<String, Value> = fields_json
                            .as_deref()
                            .filter(|bytes| !bytes.is_empty())
                            .map(sonic_rs::from_slice)
                            .transpose()
                            .map_err(|e| {
                                GraphError::StorageError(format!("vector WAL fields decode: {}", e))
                            })?
                            .unwrap_or_default();

                        if let Some(vector_name) = named_index {
                            let config = Config::default();
                            let hnsw_overrides = self.get_hnsw_overrides(txn)?;
                            let hnsw_config = HNSWConfig::with_overrides(
                                config.vector_config.m,
                                config.vector_config.ef_construction,
                                config.vector_config.ef_search,
                                hnsw_overrides.as_ref(),
                            );
                            let flat_scan_threshold = self
                                .named_vectors
                                .effective_indexing_threshold(config.vector_flat_scan_threshold());
                            if self
                                .named_vectors
                                .dense_append_to_mutable(
                                    self.lmdb_env()?,
                                    txn,
                                    vector_name,
                                    data,
                                    *id,
                                    fields,
                                    hnsw_config,
                                    flat_scan_threshold,
                                )
                                .map_err(GraphError::from)?
                            {
                                self.set_dense_vector_spaces_metadata(
                                    txn,
                                    self.named_vectors.list_dense_vector_spaces(),
                                )?;
                            }
                        } else {
                            self.vectors
                                .insert_flat(txn, data, Some(*id), Some(fields))
                                .map_err(GraphError::from)?;
                        }
                        replayed_vectors = true;
                    }
                    WalOp::TxBegin { .. } | WalOp::TxCommit { .. } | WalOp::Snapshot { .. } => {}
                }
            }
            if let Some(last) = batch.last() {
                self.set_committed_lsn(txn, last.lsn)?;
            }
            Ok(())
        })?;
        if replayed_vectors {
            self.named_vectors.flush_mmap_stores();
        }
        Ok(())
    }

    /// Initialize the snapshot: flush WAL, sync LMDB, write snapshot WAL op.
    /// Does NOT perform the copy — caller should do the copy separately,
    /// optionally in a background thread, to avoid blocking the HTTP handler
    /// thread for the duration of the (potentially slow) file copy.
    pub fn snapshot_init(&self, collection: &str) -> Result<u64, GraphError> {
        let snapshot_lsn = self.wal.flush()?;
        if self.backend.kind() != BackendKind::Lsm {
            self.lmdb_env()?.force_sync()?;
        }
        self.wal.append(
            collection,
            WalOp::Snapshot {
                lsn_at_snapshot: snapshot_lsn,
            },
        )?;
        match self.checkpoint_wal_to_lsn(snapshot_lsn) {
            Ok(removed) if removed > 0 => {
                tracing::info!(
                    removed,
                    snapshot_lsn,
                    "WAL checkpoint truncation after snapshot"
                );
                metrics::counter!("helix_wal_segments_truncated_total").increment(removed as u64);
            }
            Ok(_) => {}
            Err(e) => tracing::warn!(
                error = %e,
                snapshot_lsn,
                "WAL checkpoint after snapshot failed"
            ),
        }
        Ok(snapshot_lsn)
    }

    pub fn snapshot_to<P: AsRef<Path>>(
        &self,
        collection: &str,
        snapshot_path: P,
    ) -> Result<(u64, File), GraphError> {
        if self.backend.kind() == BackendKind::Lsm {
            return Err(GraphError::StorageError(
                "LMDB snapshot export is unavailable on the LSM backend; use object-store backup/checkpointing instead"
                    .to_string(),
            ));
        }
        let snapshot_lsn = self.snapshot_init(collection)?;
        let _resize_guard = self.read_resize_guard_if_needed()?;

        let snapshot_path = snapshot_path.as_ref();
        let file = self
            .lmdb_env()?
            .copy_to_path(snapshot_path, CompactionOption::Enabled)?;
        file.sync_data()?;

        Ok((snapshot_lsn, file))
    }

    pub fn force_sync_lmdb(&self) -> Result<(), GraphError> {
        if self.backend.kind() == BackendKind::Lsm {
            return Ok(());
        }
        let _shared = self.read_resize_guard_if_needed()?;
        self.lmdb_env()?.force_sync().map_err(GraphError::from)
    }

    pub fn lmdb_map_size_bytes(&self) -> Result<u64, GraphError> {
        let _shared = self.read_resize_guard_if_needed()?;
        Ok(self.lmdb_env()?.info().map_size as u64)
    }

    pub fn lmdb_last_txn_id(&self) -> Result<usize, GraphError> {
        let _shared = self.read_resize_guard_if_needed()?;
        Ok(self.lmdb_env()?.info().last_txn_id)
    }

    pub fn get_metadata(&self, txn: &RoTxn) -> Result<StorageMetadata, GraphError> {
        // Routed through the backend seam (heed-txn scaffold). Byte-identical to
        // the prior `metadata_db.get(txn, METADATA_CURRENT_KEY)`: Namespace::Metadata
        // resolves to the same metadata_db (Str-keyed; heed `Str` stores raw UTF-8
        // so the key bytes match `METADATA_CURRENT_KEY.as_bytes()`). Deserialize +
        // corruption-log happen inside the visitor (the borrow is closure-scoped).
        self.backend
            .get_with_heed(txn, Namespace::Metadata, METADATA_CURRENT_KEY.as_bytes(), |v| {
                let bytes = v.ok_or_else(|| {
                    GraphError::StorageError("storage metadata missing".to_string())
                })?;
                match Self::try_deserialize_metadata(bytes) {
                    Ok(meta) => Ok(meta),
                    Err(e) => {
                        let error = GraphError::StorageError(format!(
                            "corrupted collection metadata: {}",
                            e
                        ));

                        if !self
                            .metadata_corruption_logged
                            .swap(true, std::sync::atomic::Ordering::Relaxed)
                        {
                            tracing::warn!(
                                error = ?error,
                                "Corrupted collection metadata detected; refusing to return defaults"
                            );
                        }

                        Err(error)
                    }
                }
            })
            .map_err(|e| GraphError::New(e.to_string()))?
    }

    pub fn get_metadata_be(&self, r: &AnyRead<'_>) -> Result<StorageMetadata, GraphError> {
        let missing_metadata_is_default = self.backend.kind() == BackendKind::Lsm;
        let mut metadata = self
            .backend
            .get_with(r, Namespace::Metadata, METADATA_CURRENT_KEY.as_bytes(), |v| {
                let Some(bytes) = v else {
                    if missing_metadata_is_default {
                        return Ok(StorageMetadata::new(Vec::new()));
                    }
                    return Err(GraphError::StorageError(
                        "storage metadata missing".to_string(),
                    ));
                };
                match Self::try_deserialize_metadata(bytes) {
                    Ok(meta) => Ok(meta),
                    Err(e) => {
                        let error = GraphError::StorageError(format!(
                            "corrupted collection metadata: {}",
                            e
                        ));

                        if !self
                            .metadata_corruption_logged
                            .swap(true, std::sync::atomic::Ordering::Relaxed)
                        {
                            tracing::warn!(
                                error = ?error,
                                "Corrupted collection metadata detected; refusing to return defaults"
                            );
                        }

                        Err(error)
                    }
                }
            })
            .map_err(|e| GraphError::New(e.to_string()))??;
        self.overlay_lsm_counter_stats(r, &mut metadata)?;
        Ok(metadata)
    }

    fn overlay_lsm_counter_stats(
        &self,
        r: &AnyRead<'_>,
        metadata: &mut StorageMetadata,
    ) -> Result<(), GraphError> {
        if self.backend.kind() != BackendKind::Lsm {
            return Ok(());
        }
        let mut stats = metadata.stats.clone();
        for counter in [
            MetadataCounter::Nodes,
            MetadataCounter::Edges,
            MetadataCounter::Vectors,
        ] {
            let value = self
                .backend
                .get_with(r, Namespace::Metadata, lsm_counter_key(counter), |v| {
                    v.map(decode_lsm_counter_value).transpose()
                })
                .map_err(|e| GraphError::New(e.to_string()))??;
            let Some(value) = value else {
                continue;
            };
            match counter {
                MetadataCounter::Nodes => stats.node_count = value,
                MetadataCounter::Edges => stats.edge_count = value,
                MetadataCounter::Vectors => stats.vector_count = value,
            }
        }
        metadata.stats = stats;
        Ok(())
    }

    pub(crate) fn seed_lsm_counter_keys(&self) -> Result<(), GraphError> {
        if self.backend.kind() != BackendKind::Lsm || self.backend.is_reader_replica() {
            return Ok(());
        }

        let read = self
            .backend
            .begin_read()
            .map_err(|e| self.backend_error(e, "seed_lsm_counter_keys_read"))?;
        let metadata = match self.get_metadata_be(&read) {
            Ok(metadata) => metadata,
            Err(GraphError::StorageError(message)) if message == "storage metadata missing" => {
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        let stats = metadata.stats;
        let missing = self.missing_lsm_counter_keys(&read, &stats)?;
        if missing.is_empty() {
            return Ok(());
        }

        let mut write = self
            .backend
            .begin_write()
            .map_err(|e| self.backend_error(e, "seed_lsm_counter_keys_begin"))?;
        for (counter, value) in missing {
            self.backend
                .put(
                    &mut write,
                    Namespace::Metadata,
                    lsm_counter_key(counter),
                    &encode_lsm_counter_value(value),
                )
                .map_err(|e| self.backend_error(e, "seed_lsm_counter_keys_put"))?;
        }
        self.backend
            .commit(write)
            .map_err(|e| self.backend_error(e, "seed_lsm_counter_keys_commit"))?;
        Ok(())
    }

    fn missing_lsm_counter_keys(
        &self,
        r: &AnyRead<'_>,
        stats: &StorageStats,
    ) -> Result<Vec<(MetadataCounter, u64)>, GraphError> {
        let mut missing = Vec::new();
        for (counter, blob_baseline) in [
            (MetadataCounter::Nodes, stats.node_count),
            (MetadataCounter::Edges, stats.edge_count),
            (MetadataCounter::Vectors, stats.vector_count),
        ] {
            let exists = self
                .backend
                .get_with(r, Namespace::Metadata, lsm_counter_key(counter), |v| {
                    v.is_some()
                })
                .map_err(|e| GraphError::New(e.to_string()))?;
            if exists {
                continue;
            }
            let value = self.seed_value_from_scan_or_blob(r, counter, blob_baseline, "open");
            missing.push((counter, value));
        }
        Ok(missing)
    }

    /// Short collection name for tracing/log context, derived the same way as
    /// `NamedVectorManager::new`'s `collection_name` (the path's final component).
    pub(crate) fn collection_display_name(&self) -> &str {
        self.path()
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
    }

    /// The true value of `counter` computed by scanning the namespace it
    /// mirrors, ignoring the (possibly stale/corrupted) counter key and the
    /// frozen metadata blob. Nodes/Edges are a plain key count; Vectors mirrors
    /// the count of `Namespace::Nodes` rows labeled `"point"` (see the
    /// `existing_is_point` call sites in `replication.rs` that drive it), not a
    /// separate namespace, so its scan must decode each node's label.
    pub(crate) fn scan_counter_truth(
        &self,
        r: &AnyRead<'_>,
        counter: MetadataCounter,
    ) -> Result<u64, GraphError> {
        let mut count: u64 = 0;
        let ns = match counter {
            MetadataCounter::Nodes | MetadataCounter::Vectors => Namespace::Nodes,
            MetadataCounter::Edges => Namespace::Edges,
        };
        self.backend
            .scan(r, ns, KeyRange::all(), |_key, value| {
                let counts = match counter {
                    MetadataCounter::Vectors => SerializedNode::decode_node(value, 0)
                        .map(|node| node.label == "point")
                        .unwrap_or(false),
                    MetadataCounter::Nodes | MetadataCounter::Edges => true,
                };
                if counts {
                    count += 1;
                }
                true
            })
            .map_err(|e| GraphError::New(e.to_string()))?;
        Ok(count)
    }

    /// Resolve the seed value for an absent LSM counter key: scan truth when
    /// the scan succeeds (self-healing even under a false-negative existence
    /// check), the stale blob baseline only if the scan itself errors. Emits
    /// the warn/metric contract for `helix_lsm_counter_seed_total` whenever a
    /// seed fires against non-empty data.
    pub(crate) fn seed_value_from_scan_or_blob(
        &self,
        r: &AnyRead<'_>,
        counter: MetadataCounter,
        blob_baseline: u64,
        context: &'static str,
    ) -> u64 {
        let started = Instant::now();
        match self.scan_counter_truth(r, counter) {
            Ok(scan_value) => {
                if scan_value > 0 {
                    tracing::warn!(
                        collection = self.collection_display_name(),
                        counter = counter_label(counter),
                        context,
                        blob_baseline,
                        scan_value,
                        elapsed_ms = started.elapsed().as_millis() as u64,
                        "LSM counter key absent; seeding from scan truth"
                    );
                    metrics::counter!(
                        "helix_lsm_counter_seed_total",
                        "counter" => counter_label(counter),
                        "context" => context
                    )
                    .increment(1);
                }
                scan_value
            }
            Err(error) => {
                tracing::warn!(
                    collection = self.collection_display_name(),
                    counter = counter_label(counter),
                    context,
                    source = "blob_fallback",
                    blob_baseline,
                    error = %error,
                    "LSM counter scan failed; falling back to stale blob baseline"
                );
                blob_baseline
            }
        }
    }

    /// Fleet-repair tool for the LSM counter-corruption class: unconditionally
    /// recompute each counter from scan truth and overwrite the merge-key,
    /// regardless of whether the key was present, absent, or wrong. Unlike
    /// `seed_lsm_counter_keys` (only fires on a missing key), this always
    /// writes, so it also fixes a key that exists but drifted. The scan and
    /// the overwrite are two separate transactions, not one atomic unit: a
    /// write that commits in between can still be lost from the counter, so
    /// run repairs during quiescence or verify the result afterward.
    pub fn recount_lsm_counters(&self) -> Result<RecountCounters, GraphError> {
        if !self.try_begin_recount() {
            return Err(GraphError::New(
                "recount already in progress for this collection".to_string(),
            ));
        }
        // Releases the guard on drop so every exit path below — success, an
        // early `?` return, or a panic — clears it rather than leaving this
        // collection stuck permanently rejecting recounts.
        struct RecountGuard<'a>(&'a HelixGraphStorage);
        impl Drop for RecountGuard<'_> {
            fn drop(&mut self) {
                self.0.end_recount();
            }
        }
        let _guard = RecountGuard(self);

        ensure_lsm_writer(&self.backend)?;

        let read = self
            .backend
            .begin_read()
            .map_err(|e| self.backend_error(e, "recount_scan"))?;
        let mut write = self
            .backend
            .begin_write()
            .map_err(|e| self.backend_error(e, "recount_begin"))?;

        let mut rows = [
            (MetadataCounter::Nodes, 0u64, 0u64),
            (MetadataCounter::Edges, 0u64, 0u64),
            (MetadataCounter::Vectors, 0u64, 0u64),
        ];
        for (counter, before, after) in &mut rows {
            *before = self
                .backend
                .get_for_update(
                    &write,
                    Namespace::Metadata,
                    lsm_counter_key(*counter),
                    |v| v.map(decode_lsm_counter_value),
                )
                .map_err(|e| GraphError::New(e.to_string()))?
                .transpose()?
                .unwrap_or(0);
            *after = self.scan_counter_truth(&read, *counter)?;
            self.backend
                .put(
                    &mut write,
                    Namespace::Metadata,
                    lsm_counter_key(*counter),
                    &encode_lsm_counter_value(*after),
                )
                .map_err(|e| GraphError::New(e.to_string()))?;
        }
        drop(read);

        self.backend
            .commit(write)
            .map_err(|e| self.backend_error(e, "recount_commit"))?;
        self.refresh_metadata_snapshot_best_effort();

        for (counter, before, after) in rows {
            tracing::warn!(
                collection = self.collection_display_name(),
                counter = counter_label(counter),
                context = "recount",
                before,
                after,
                "LSM counter recounted from scan truth via admin endpoint"
            );
            metrics::counter!(
                "helix_lsm_counter_seed_total",
                "counter" => counter_label(counter),
                "context" => "recount"
            )
            .increment(1);
        }

        Ok(RecountCounters {
            nodes: RecountCounterResult {
                before: rows[0].1,
                after: rows[0].2,
            },
            edges: RecountCounterResult {
                before: rows[1].1,
                after: rows[1].2,
            },
            vectors: RecountCounterResult {
                before: rows[2].1,
                after: rows[2].2,
            },
        })
    }

    /// Fleet-repair tool for the ghost payload-index class (issue: `drop_node`
    /// / `drop_node_be`'s de-index step used a flat property lookup instead of
    /// the nested-aware `payload_value_for_key`, so any node indexed on a
    /// dotted field name — e.g. `metadata.repo` — silently failed to
    /// de-index on delete, leaking a dup entry pointing at a node id that no
    /// longer exists). Run this AFTER the resolver fix ships, or it will
    /// re-poison: this only repairs already-corrupted state, it does not
    /// change how future deletes de-index.
    ///
    /// For each targeted payload index, scans every dup entry, checks whether
    /// its node id still exists in `Namespace::Nodes`, and `delete_dup`s the
    /// entries whose node is gone. `field = None` runs every registered
    /// index; `field = Some(name)` scopes to one. One read pass + one write
    /// batch per field (mirrors `recount_lsm_counters`'s single-pass shape;
    /// scoped per-field so one huge poisoned index doesn't hold a write
    /// batch open across every other index too).
    pub fn gc_payload_index(
        &self,
        field: Option<&str>,
        point_mutation_gate: Option<&Mutex<()>>,
    ) -> Result<HashMap<String, PayloadIndexGcFieldResult>, GraphError> {
        if !self.try_begin_payload_index_gc() {
            return Err(GraphError::New(
                "payload index GC already in progress for this collection".to_string(),
            ));
        }
        // Releases the guard on drop so every exit path — success, an early
        // `?` return, or a panic — clears it rather than leaving this
        // collection stuck permanently rejecting GC runs.
        struct GcGuard<'a>(&'a HelixGraphStorage);
        impl Drop for GcGuard<'_> {
            fn drop(&mut self) {
                self.0.end_payload_index_gc();
            }
        }
        let _guard = GcGuard(self);

        ensure_lsm_writer(&self.backend)?;

        let targets: Vec<(String, PayloadIndexSchema)> = {
            let payload_indices = self
                .payload_indices
                .read()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            match field {
                Some(name) => {
                    let handle = payload_indices.get(name).ok_or_else(|| {
                        GraphError::New(format!("Payload index '{}' not found", name))
                    })?;
                    vec![(name.to_string(), handle.schema.clone())]
                }
                None => payload_indices
                    .iter()
                    .filter(|(_, handle)| handle.is_ready())
                    .map(|(name, handle)| (name.clone(), handle.schema.clone()))
                    .collect(),
            }
        };

        let mut results = HashMap::with_capacity(targets.len());
        for (field_name, schema) in targets {
            let db_name = Self::payload_index_db_name(&field_name, &schema);

            // Read phase: every (key, node_id) dup entry in this field's index.
            let read = self
                .backend
                .begin_read()
                .map_err(|e| self.backend_error(e, "gc_payload_index_scan"))?;
            let mut entries: Vec<(Vec<u8>, [u8; 16])> = Vec::new();
            let mut scan_err = false;
            self.backend
                .scan(
                    &read,
                    Namespace::PayloadIndex(&db_name),
                    KeyRange::all(),
                    |key, val_bytes| {
                        let Ok(id_bytes) = <[u8; 16]>::try_from(val_bytes) else {
                            scan_err = true;
                            return false;
                        };
                        entries.push((key.to_vec(), id_bytes));
                        true
                    },
                )
                .map_err(|e| GraphError::New(e.to_string()))?;
            if scan_err {
                return Err(GraphError::New(format!(
                    "payload index '{}' has an undecodable dup value",
                    field_name
                )));
            }
            let scanned = entries.len();

            // Resolve which entries are ghosts (node no longer exists),
            // still against the read snapshot taken above.
            let mut ghosts: Vec<(Vec<u8>, [u8; 16])> = Vec::new();
            for (key, id_bytes) in entries {
                let exists = self
                    .backend
                    .get_with(&read, Namespace::Nodes, &id_bytes, |v| v.is_some())
                    .map_err(|e| GraphError::New(e.to_string()))?;
                if !exists {
                    ghosts.push((key, id_bytes));
                }
            }
            drop(read);

            // Deleting from the snapshot's ghost list alone is racy: the client uses
            // deterministic point ids, so a delete→reindex of the same file
            // between the scan above and the write below recreates the same
            // (payload_key, node_id) entry, and an unguarded delete_dup would
            // remove the now-live index entry. Hold the collection's
            // point-mutation gate (the same gate upserts/deletes take) across
            // re-verify + delete + commit, and re-check each ghost against a
            // fresh snapshot taken under the gate.
            let mut removed = 0usize;
            if !ghosts.is_empty() {
                let _mutation_gate = point_mutation_gate.map(|gate| {
                    gate.lock().unwrap_or_else(|poisoned| {
                        tracing::error!(
                            collection = self.collection_display_name(),
                            "point mutation gate poisoned; recovering lock for payload-index GC"
                        );
                        poisoned.into_inner()
                    })
                });
                let reverify = self
                    .backend
                    .begin_read()
                    .map_err(|e| self.backend_error(e, "gc_payload_index_reverify"))?;
                let mut confirmed: Vec<(Vec<u8>, [u8; 16])> = Vec::with_capacity(ghosts.len());
                for (key, id_bytes) in ghosts {
                    let exists = self
                        .backend
                        .get_with(&reverify, Namespace::Nodes, &id_bytes, |v| v.is_some())
                        .map_err(|e| GraphError::New(e.to_string()))?;
                    if !exists {
                        confirmed.push((key, id_bytes));
                    }
                }
                drop(reverify);

                removed = confirmed.len();
                if removed > 0 {
                    let mut write = self
                        .backend
                        .begin_write()
                        .map_err(|e| self.backend_error(e, "gc_payload_index_write"))?;
                    for (key, id_bytes) in &confirmed {
                        self.backend
                            .delete_dup(
                                &mut write,
                                Namespace::PayloadIndex(&db_name),
                                key,
                                id_bytes,
                            )
                            .map_err(|e| GraphError::New(e.to_string()))?;
                    }
                    self.backend
                        .commit(write)
                        .map_err(|e| self.backend_error(e, "gc_payload_index_commit"))?;
                    tracing::warn!(
                        collection = self.collection_display_name(),
                        field = %field_name,
                        scanned,
                        removed,
                        "ghost payload-index entries reclaimed via admin GC endpoint"
                    );
                    metrics::counter!(
                        "helix_payload_index_gc_removed_total",
                        "field" => field_name.clone(),
                    )
                    .increment(removed as u64);
                }
            }

            results.insert(field_name, PayloadIndexGcFieldResult { scanned, removed });
        }

        Ok(results)
    }

    pub fn put_metadata(
        &self,
        txn: &mut RwTxn,
        metadata: &StorageMetadata,
    ) -> Result<(), GraphError> {
        let bytes = Self::serialize_metadata(metadata)?;
        // Delete-before-put: force a fresh insert instead of an in-place
        // overwrite. For collections with many vector spaces / payload indices
        // the serialized metadata exceeds LMDB's overflow threshold (~2 KB) and
        // is stored in an overflow page. lmdb-master3's in-place overflow
        // overwrite (`_mdb_cursor_put`, mdb.c:8636-8693) can memcpy directly
        // into an overflow page that was *spilled* mid-transaction by the large
        // segment-seal write; in a non-writemap env a spilled page lives only in
        // the read-only data mmap, so the in-place write is a write-permission
        // fault (SIGSEGV) on the index thread. Deleting the key first routes the
        // write through the fresh-allocation path, which always targets a
        // writable dirty page and never the spilled/read-only overflow page.
        // The metadata write is infrequent (one per segment seal), so the extra
        // delete is negligible.
        // SIGSEGV workaround PRESERVED: delete-first forces heed's
        // fresh-allocation path (never an in-place overwrite onto a spilled
        // read-only overflow page) THEN put. delete_raw/put_raw map 1:1 to heed
        // delete/put on the same metadata DBI, so the workaround and the on-disk
        // bytes are unchanged on LMDB.
        self.backend
            .delete_heed(txn, Namespace::Metadata, METADATA_CURRENT_KEY.as_bytes())
            .map_err(|e| GraphError::New(e.to_string()))?;
        self.backend
            .put_heed(
                txn,
                Namespace::Metadata,
                METADATA_CURRENT_KEY.as_bytes(),
                &bytes,
            )
            .map_err(|e| GraphError::New(e.to_string()))?;
        Ok(())
    }

    pub fn metadata_sidecar_path(path: &Path) -> PathBuf {
        path.join(STORAGE_METADATA_SIDECAR_FILE)
    }

    fn metadata_sidecar_miss_path(path: &Path) -> PathBuf {
        let mut hasher = XxHash64::default();
        hasher.write(path.to_string_lossy().as_bytes());
        let miss_name = format!(".metadata.lsm.miss.{:016x}", hasher.finish());
        path.parent()
            .unwrap_or_else(|| Path::new("."))
            .join(miss_name)
    }

    fn lsm_metadata_sidecar_put_interval_ms() -> u64 {
        std::env::var("HELIX_LSM_METADATA_SIDECAR_PUT_INTERVAL_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(10_000)
    }

    fn lsm_metadata_sidecar_miss_ttl_ms() -> u64 {
        std::env::var("HELIX_LSM_METADATA_SIDECAR_MISS_TTL_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(60_000)
    }

    fn lsm_metadata_sidecar_cache_ttl_ms() -> u64 {
        std::env::var("HELIX_LSM_METADATA_SIDECAR_CACHE_TTL_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(300_000)
    }

    fn now_epoch_millis() -> u64 {
        chrono::Utc::now().timestamp_millis().max(0) as u64
    }

    /// Cooldown before the next promotion attempt after `consecutive_failures`
    /// failures in a row: base 30s doubling per failure, capped at 15m.
    fn reader_poll_promote_cooldown_ms(consecutive_failures: u32) -> u64 {
        let shift = consecutive_failures.saturating_sub(1).min(5);
        READER_POLL_PROMOTE_FAILURE_COOLDOWN_MS
            .saturating_mul(1u64 << shift)
            .min(READER_POLL_PROMOTE_FAILURE_COOLDOWN_MAX_MS)
    }

    fn reader_poll_promote_retry_due(
        last_failed_ms: u64,
        now_ms: u64,
        consecutive_failures: u32,
    ) -> bool {
        last_failed_ms == 0
            || now_ms.saturating_sub(last_failed_ms)
                >= Self::reader_poll_promote_cooldown_ms(consecutive_failures)
    }

    fn metadata_sidecar_fingerprint(
        metadata: &StorageMetadata,
        data_mdb_bytes: u64,
        map_size_bytes: Option<u64>,
        adjacency_backfill_complete: Option<bool>,
    ) -> Option<u64> {
        let mut hasher = XxHash64::default();
        let bytes = sonic_rs::to_vec(metadata).ok()?;
        hasher.write(&bytes);
        hasher.write(&data_mdb_bytes.to_le_bytes());
        hasher.write(&map_size_bytes.unwrap_or(0).to_le_bytes());
        hasher.write(&[adjacency_mark_code(adjacency_backfill_complete)]);
        Some(hasher.finish())
    }

    fn should_upload_lsm_metadata_sidecar(&self, fingerprint: u64) -> bool {
        if self.backend.kind() != BackendKind::Lsm {
            return false;
        }
        if self.last_lsm_sidecar_fingerprint.load(Ordering::Acquire) == fingerprint {
            self.lsm_sidecar_put_held.store(false, Ordering::Release);
            return false;
        }
        let now = Self::now_epoch_millis();
        let last = self.last_lsm_sidecar_upload_ms.load(Ordering::Acquire);
        let min_interval = Self::lsm_metadata_sidecar_put_interval_ms();
        let held = min_interval > 0 && last > 0 && now.saturating_sub(last) < min_interval;
        self.lsm_sidecar_put_held.store(held, Ordering::Release);
        if held {
            return false;
        }
        self.last_lsm_sidecar_upload_ms
            .store(now, Ordering::Release);
        true
    }

    fn mark_lsm_metadata_sidecar_uploaded(&self, fingerprint: u64) {
        self.last_lsm_sidecar_fingerprint
            .store(fingerprint, Ordering::Release);
    }

    fn lsm_metadata_sidecar_miss_is_fresh(path: &Path) -> bool {
        let Ok(bytes) = fs::read(Self::metadata_sidecar_miss_path(path)) else {
            return false;
        };
        let Ok(text) = std::str::from_utf8(&bytes) else {
            return false;
        };
        let Ok(last_miss) = text.trim().parse::<u64>() else {
            return false;
        };
        Self::now_epoch_millis().saturating_sub(last_miss)
            < Self::lsm_metadata_sidecar_miss_ttl_ms()
    }

    fn remember_lsm_metadata_sidecar_miss(path: &Path) {
        let miss_path = Self::metadata_sidecar_miss_path(path);
        if let Some(parent) = miss_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if miss_path
            .parent()
            .map(|parent| parent.exists())
            .unwrap_or(true)
        {
            let _ = fs::write(miss_path, Self::now_epoch_millis().to_string());
        }
    }

    fn metadata_sidecar_cache_is_fresh(path: &Path) -> bool {
        let ttl_ms = Self::lsm_metadata_sidecar_cache_ttl_ms();
        if ttl_ms == 0 {
            return false;
        }
        let Ok(metadata) = fs::metadata(Self::metadata_sidecar_path(path)) else {
            return false;
        };
        let Ok(modified) = metadata.modified() else {
            return false;
        };
        let Ok(age) = SystemTime::now().duration_since(modified) else {
            return true;
        };
        age.as_millis() < u128::from(ttl_ms)
    }

    pub fn data_mdb_bytes(path: &Path) -> Result<u64, GraphError> {
        match fs::metadata(path.join("data.mdb")) {
            Ok(metadata) => Ok(metadata.len()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(0),
            Err(err) => Err(GraphError::from(err)),
        }
    }

    pub fn read_metadata_sidecar_from_path(
        path: &Path,
    ) -> Result<Option<StorageMetadataSidecar>, GraphError> {
        match fs::read(Self::metadata_sidecar_path(path)) {
            Ok(bytes) => Ok(Some(sonic_rs::from_slice(&bytes)?)),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(GraphError::from(err)),
        }
    }

    pub fn read_metadata_sidecar_from_path_or_lsm(
        path: &Path,
        storage_backend: StorageBackendConfig,
    ) -> Result<Option<StorageMetadataSidecar>, GraphError> {
        let local_sidecar = Self::read_metadata_sidecar_from_path(path)?;
        if !storage_backend.is_lsm() || storage_backend.is_lsm_in_memory() {
            return Ok(local_sidecar);
        }
        if std::env::var_os("HELIX_LSM_BUCKET").is_none() {
            return Ok(local_sidecar);
        }
        if local_sidecar.is_some() && Self::metadata_sidecar_cache_is_fresh(path) {
            return Ok(local_sidecar);
        }
        if Self::lsm_metadata_sidecar_miss_is_fresh(path) {
            return Ok(local_sidecar);
        }
        match read_metadata_sidecar_from_lsm_env(path, storage_backend) {
            Ok(Some(bytes)) => {
                let sidecar: StorageMetadataSidecar = sonic_rs::from_slice(&bytes)?;
                // Readers: skip local persist to avoid `create_dir_all` that
                // creates a local registry dir for every name ever queried,
                // leaving dirs that survive drops (bug b).
                if !storage_backend.is_reader() {
                    Self::persist_metadata_sidecar_bytes_at_path(path, &bytes)?;
                }
                let _ = fs::remove_file(Self::metadata_sidecar_miss_path(path));
                Ok(Some(sidecar))
            }
            Ok(None) => {
                Self::remember_lsm_metadata_sidecar_miss(path);
                // No S3 sidecar is not proof of a drop: collections untouched
                // since sidecars moved to S3 never uploaded one. A reader drops
                // its local copy only once the prefix itself is gone (one LIST,
                // at most once per miss TTL); deleting the file keeps the
                // remembered-miss path above from serving it again. Writers keep
                // the local fallback: a just-created sidecar lands in S3 later.
                if storage_backend.is_reader()
                    && local_sidecar.is_some()
                    && matches!(
                        lsm_collection_prefix_exists_from_env(path, storage_backend),
                        Ok(false)
                    )
                {
                    let _ = fs::remove_file(Self::metadata_sidecar_path(path));
                    return Ok(None);
                }
                Ok(local_sidecar)
            }
            Err(err) if local_sidecar.is_some() => {
                tracing::debug!(
                    path = %path.display(),
                    error = %err,
                    "serving stale local LSM metadata sidecar after remote refresh failed"
                );
                Ok(local_sidecar)
            }
            Err(err) => Err(GraphError::StorageError(err.to_string())),
        }
    }

    fn persist_metadata_sidecar_bytes_at_path(path: &Path, bytes: &[u8]) -> Result<(), GraphError> {
        fs::create_dir_all(path)?;
        let sidecar_path = Self::metadata_sidecar_path(path);
        let tmp_path = sidecar_path.with_extension("json.tmp");
        fs::write(&tmp_path, bytes)?;
        fs::rename(tmp_path, sidecar_path)?;
        Ok(())
    }

    pub fn persist_metadata_sidecar_at_path(
        path: &Path,
        metadata: &StorageMetadata,
        map_size_bytes: Option<u64>,
        adjacency_backfill_complete: Option<bool>,
    ) -> Result<(), GraphError> {
        let mut sidecar = StorageMetadataSidecar::new(
            metadata.clone(),
            Self::data_mdb_bytes(path)?,
            map_size_bytes,
        );
        sidecar.adjacency_backfill_complete = adjacency_backfill_complete;
        let bytes = sonic_rs::to_vec(&sidecar)?;
        Self::persist_metadata_sidecar_bytes_at_path(path, &bytes)?;
        Ok(())
    }

    fn persist_metadata_sidecar_best_effort(&self, metadata: &StorageMetadata) {
        if let Some(sidecar) = self.persist_metadata_sidecar_locally(metadata) {
            self.upload_metadata_sidecar(sidecar);
        }
    }

    /// Writes the local sidecar, and hands it back when the object store is
    /// due it.
    fn persist_metadata_sidecar_locally(&self, metadata: &StorageMetadata) -> Option<BuiltSidecar> {
        let built = self.build_metadata_sidecar(metadata)?;
        self.should_upload_lsm_metadata_sidecar(built.fingerprint?)
            .then_some(built)
    }

    /// The sidecar of `metadata` with the adjacency backfill's mark as it is
    /// now, written to the local file. `None` where it does not serialize.
    fn build_metadata_sidecar(&self, metadata: &StorageMetadata) -> Option<BuiltSidecar> {
        let path = self.path();
        let map_size_bytes = self.lmdb_env().ok().map(|env| env.info().map_size as u64);
        let data_mdb_bytes = Self::data_mdb_bytes(path).unwrap_or(0);
        let mark = self.adjacency_backfill_complete();
        let fingerprint =
            Self::metadata_sidecar_fingerprint(metadata, data_mdb_bytes, map_size_bytes, mark);
        let mut sidecar =
            StorageMetadataSidecar::new(metadata.clone(), data_mdb_bytes, map_size_bytes);
        sidecar.adjacency_backfill_complete = mark;
        let bytes = match sonic_rs::to_vec(&sidecar) {
            Ok(bytes) => bytes,
            Err(err) => {
                metrics::counter!("helix_metadata_sidecar_write_errors_total").increment(1);
                tracing::warn!(
                    path = ?path,
                    error = ?err,
                    "Failed to serialize collection metadata sidecar"
                );
                return None;
            }
        };
        if let Err(err) = Self::persist_metadata_sidecar_bytes_at_path(path, &bytes) {
            metrics::counter!("helix_metadata_sidecar_write_errors_total").increment(1);
            tracing::warn!(
                path = ?path,
                error = ?err,
                "Failed to persist collection metadata sidecar"
            );
        }
        Some(BuiltSidecar {
            bytes,
            fingerprint,
            mark,
        })
    }

    fn upload_metadata_sidecar(&self, sidecar: BuiltSidecar) {
        // No wait on this path, which is a commit's or the sweeper's. The
        // sidecar is held for the sweeper instead where another put of this
        // collection is on its way, and where it was built before a backfill
        // call changed the mark: that call puts the sidecar that says so,
        // and this one must not land after it.
        let one_at_a_time = self.sidecar_put.try_lock();
        if one_at_a_time.is_none() || sidecar.mark != self.adjacency_backfill_complete() {
            self.lsm_sidecar_put_held.store(true, Ordering::Release);
            return;
        }
        if let Err(err) = self.put_metadata_sidecar(sidecar) {
            // Nothing else would send this state again: the sweeper does.
            self.lsm_sidecar_put_held.store(true, Ordering::Release);
            metrics::counter!("helix_metadata_sidecar_write_errors_total").increment(1);
            tracing::warn!(
                path = ?self.path(),
                error = ?err,
                "Failed to persist LSM collection metadata sidecar to object store"
            );
        }
    }

    /// Puts `sidecar` on the object store and records what it carried. The
    /// caller holds `sidecar_put`.
    fn put_metadata_sidecar(&self, sidecar: BuiltSidecar) -> Result<(), BackendError> {
        if let Err(error) = self.backend.persist_lsm_metadata_sidecar(sidecar.bytes) {
            // A put that answers an error may have arrived all the same.
            self.adjacency_mark_sent.store(0, Ordering::Release);
            return Err(error);
        }
        if let Some(fingerprint) = sidecar.fingerprint {
            self.mark_lsm_metadata_sidecar_uploaded(fingerprint);
        }
        self.adjacency_mark_sent
            .store(adjacency_mark_code(sidecar.mark), Ordering::Release);
        Ok(())
    }

    /// Whether the adjacency backfill from points has run to its end, as
    /// this process knows it without a read: `None` on a backend that does
    /// not read the mark at its open.
    pub fn adjacency_backfill_complete(&self) -> Option<bool> {
        match self.adjacency_backfill_mark.load(Ordering::Acquire) {
            1 => Some(false),
            2 => Some(true),
            _ => None,
        }
    }

    /// The mark becomes `complete`, here and in the object store's sidecar,
    /// which is put now, whatever the put interval, unless the last sidecar
    /// this process put already says so. A backfill call on the writer calls
    /// it under the collection's graph gate: with `false` BEFORE the write
    /// that starts a pass over, and fails without that write when the put
    /// fails; with `true` only once the store holds the pass's mark. So the
    /// object store never says "complete" of a pass that is not, and a
    /// reader that is behind it says "not complete" or nothing.
    fn send_adjacency_backfill_mark(&self, complete: bool) -> Result<(), GraphError> {
        let code = adjacency_mark_code(Some(complete));
        // Nothing to say, and no wait to find that out: the backfill calls
        // of one collection are serial under its graph gate, and a regular
        // put only ever records the mark as it is.
        if self.adjacency_backfill_mark.load(Ordering::Acquire) == code
            && self.adjacency_mark_sent.load(Ordering::Acquire) == code
        {
            return Ok(());
        }
        // The one wait on this lock: for a regular put that is on its way,
        // so that this put is the later of the two.
        let _one_at_a_time = self.sidecar_put.lock();
        self.adjacency_backfill_mark.store(code, Ordering::Release);
        if self.adjacency_mark_sent.load(Ordering::Acquire) == code {
            return Ok(());
        }
        let sidecar = {
            // The local file is written in the order the refreshes write it.
            let _in_order = self.metadata_snapshot_read_at.lock();
            let metadata = self.metadata_snapshot()?;
            self.build_metadata_sidecar(&metadata)
        }
        .ok_or_else(|| {
            GraphError::StorageError("the metadata sidecar did not serialize".to_string())
        })?;
        // An error no caller classifies by its text: a sidecar that does not
        // arrive is no fault of the collection, which is neither quarantined
        // nor degraded for it, whatever the transfer's own error says.
        self.put_metadata_sidecar(sidecar).map_err(|error| {
            GraphError::Io(std::io::Error::other(format!(
                "the metadata sidecar was not put: {error}"
            )))
        })?;
        self.last_lsm_sidecar_upload_ms
            .store(Self::now_epoch_millis(), Ordering::Release);
        Ok(())
    }

    /// The adjacency backfill's mark as an open read it. An open does not
    /// need the mark, so a read of it that fails does not fail the open: it
    /// is counted and reads as no mark, and the pass is due again.
    fn adjacency_mark_read_at_open(read: Result<bool, BackendError>) -> bool {
        read.unwrap_or_else(|error| {
            metrics::counter!("helix_adjacency_backfill_mark_unread_total").increment(1);
            tracing::warn!(
                %error,
                "the adjacency backfill's mark was not read at an open; it reads as not complete"
            );
            false
        })
    }

    /// Sends the sidecar a put interval held back. The last change of a
    /// burst has no later refresh to send it, so the optimizer's backlog
    /// sweeper calls this for every loaded collection; it costs two atomic
    /// reads when nothing is held.
    /// shortcut: a collection evicted between a held put and the next sweep
    /// keeps the older object until its next change; send from the eviction
    /// if stale sidecars of evicted collections are seen.
    pub(crate) fn upload_held_metadata_sidecar(&self) {
        if !self.lsm_sidecar_put_held.load(Ordering::Acquire) {
            return;
        }
        let upload = {
            let _in_order = self.metadata_snapshot_read_at.lock();
            self.metadata_snapshot()
                .ok()
                .and_then(|metadata| self.persist_metadata_sidecar_locally(&metadata))
        };
        if let Some(sidecar) = upload {
            self.upload_metadata_sidecar(sidecar);
        }
    }

    pub fn metadata_snapshot(&self) -> Result<StorageMetadata, GraphError> {
        self.metadata_snapshot
            .read()
            .map(|snapshot| snapshot.clone())
            .map_err(|e| GraphError::StorageError(format!("metadata snapshot poisoned: {}", e)))
    }

    pub fn refresh_metadata_snapshot(&self) -> Result<(), GraphError> {
        self.reread_metadata_snapshot().map(|_| ())
    }

    /// [`Self::refresh_metadata_snapshot`], handing back the metadata it read
    /// (another refresh can replace the shared snapshot right after this one
    /// stored it) and the sequence it was read at, when the read was pinned.
    fn reread_metadata_snapshot(&self) -> Result<(StorageMetadata, Option<u64>), GraphError> {
        let (metadata, read_at) =
            self.with_read_backend(|r| Ok((self.get_metadata_be(r)?, r.snapshot_seq())))?;
        self.store_metadata_snapshot(&metadata, read_at)?;
        Ok((metadata, read_at))
    }

    /// The second half of a refresh: `metadata`, read at `read_at`, becomes
    /// the snapshot and the local sidecar, unless a refresh that read later
    /// stored first. Refreshes overlap when commits end together, and the one
    /// that read earlier must not put either back.
    fn store_metadata_snapshot(
        &self,
        metadata: &StorageMetadata,
        read_at: Option<u64>,
    ) -> Result<(), GraphError> {
        let sidecar_missing = !matches!(
            Self::read_metadata_sidecar_from_path(self.path()),
            Ok(Some(_))
        );
        let upload = {
            let mut stored_at = self.metadata_snapshot_read_at.lock();
            // A reader's sequence goes back when its `DbReader` is swapped for
            // one at an older state, so only a writer's reads are put in order.
            let read_at = read_at.filter(|_| !self.backend.is_reader_replica());
            if matches!((read_at, *stored_at), (Some(read), Some(stored)) if read < stored) {
                metrics::counter!("helix_metadata_refresh_skipped_older_total").increment(1);
                return Ok(());
            }
            *stored_at = read_at.or(*stored_at);
            let changed = {
                let mut snapshot = self.metadata_snapshot.write().map_err(|e| {
                    GraphError::StorageError(format!("metadata snapshot poisoned: {}", e))
                })?;
                let changed = *snapshot != *metadata;
                *snapshot = metadata.clone();
                changed
            };
            if changed || sidecar_missing {
                self.persist_metadata_sidecar_locally(metadata)
            } else {
                None
            }
        };
        // A network round trip: after the lock.
        if let Some(sidecar) = upload {
            self.upload_metadata_sidecar(sidecar);
        }
        Ok(())
    }

    pub(crate) fn refresh_metadata_snapshot_best_effort(&self) {
        self.reread_metadata_snapshot_best_effort();
    }

    /// The metadata just read and the sequence it was read at, or `None` when
    /// the read failed and the snapshot is what it was.
    fn reread_metadata_snapshot_best_effort(&self) -> Option<(StorageMetadata, Option<u64>)> {
        match self.reread_metadata_snapshot() {
            Ok(read) => Some(read),
            Err(e) => {
                match metadata_refresh_error_log_class(&e) {
                    MetadataRefreshErrorLogClass::CooperativeCancellation => {
                        tracing::debug!(
                            error = ?e,
                            "Metadata snapshot refresh cancelled after commit"
                        );
                    }
                    MetadataRefreshErrorLogClass::BackendFailure => {
                        tracing::warn!(
                            error = ?e,
                            "Failed to refresh metadata snapshot after commit"
                        );
                    }
                }
                None
            }
        }
    }

    /// Reader-replica only: re-read committed metadata from the object store
    /// via `DbReader` and reconcile the in-memory dense segment list and HNSW
    /// cores so searches see segments the writer created after this replica
    /// first opened the collection.
    ///
    /// No-op on writer / LMDB replicas (gated on `is_reader_replica()`). LSM
    /// readers do not own a local LMDB env, so dense cores are re-opened through
    /// the LSM vector path and local sidecar cache directory only.
    pub(crate) fn refresh_reader_view_best_effort(&self) {
        if !self.backend.is_reader_replica() {
            return;
        }
        // One at a time: each reconcile swaps in the whole segment list of
        // the metadata it read, so an overlapping older one would put the
        // list back behind the view recorded below.
        let _one_at_a_time = self
            .reader_view_reconcile
            .run
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Before the metadata read: a reader swapped in meanwhile leaves the
        // generation recorded below one behind, never ahead.
        let view = self.backend.lsm_reader_view();
        // The metadata this reconcile read itself, not the shared snapshot:
        // another refresh may have replaced that with an older read since.
        let reread = self.reread_metadata_snapshot_best_effort();
        let metadata_fresh = reread.is_some();
        let (meta, read_at) = match reread {
            Some(read) => read,
            None => match self.metadata_snapshot() {
                Ok(m) => (m, None),
                Err(e) => {
                    tracing::warn!(error = ?e, "reader reconcile: metadata snapshot unavailable");
                    return;
                }
            },
        };
        // The list ends up at exactly the sequence its metadata was read at.
        // `DbReader` announces a sequence a moment before its reads can see
        // it, so the announced one stands in only for an unpinned read.
        let covers = view.map(|(generation, announced)| (generation, read_at.unwrap_or(announced)));
        // With them, the adjacency backfill's mark as this view holds it:
        // what the collection's info answers on a reader, without a read.
        let (hnsw_overrides, backfill_complete) = self
            .with_read_backend(|r| {
                Ok((
                    self.get_hnsw_overrides_be(r)?,
                    // A mark that is not read leaves the overrides as read.
                    self.adjacency_backfill_marked_be(r).ok(),
                ))
            })
            .unwrap_or((None, None));
        if let Some(complete) = backfill_complete {
            self.adjacency_backfill_mark
                .store(adjacency_mark_code(Some(complete)), Ordering::Release);
        }
        let base_config = Config::default();
        let base_hnsw = HNSWConfig::with_overrides(
            base_config.vector_config.m,
            base_config.vector_config.ef_construction,
            base_config.vector_config.ef_search,
            hnsw_overrides.as_ref(),
        );
        if self.backend.kind() == BackendKind::Lsm {
            self.adopt_registered_payload_indexes(&meta.payload_indices);
            match self.named_vectors.refresh_dense_view_from_metadata_lsm(
                self.path(),
                &meta.named_vectors,
                &meta.dense_vector_spaces,
                base_hnsw,
                read_at,
            ) {
                // A failed metadata read reconciled against the old snapshot:
                // the list is no newer than it was.
                Ok(0) if metadata_fresh => self.note_reader_view_reconciled(covers),
                Ok(0) => {}
                // The list names a segment no search can reach. Not recorded
                // as reconciled: the timed refresh retries the open.
                Ok(_) => {
                    metrics::counter!("helix_reader_view_reconcile_incomplete_total").increment(1);
                }
                Err(e) => {
                    tracing::warn!(error = ?e, "reader reconcile: dense view refresh failed (non-fatal)");
                }
            }
            return;
        }
        let env = match self.lmdb_env() {
            Ok(env) => env,
            Err(e) => {
                tracing::warn!(error = ?e, "reader reconcile: local LMDB env unavailable");
                return;
            }
        };
        let mut wtxn = match env.write_txn() {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(error = ?e, "reader reconcile: failed to open local write txn");
                return;
            }
        };
        if let Err(e) = self.named_vectors.refresh_dense_view_from_metadata(
            env,
            &mut wtxn,
            &meta.named_vectors,
            &meta.dense_vector_spaces,
            base_hnsw,
        ) {
            tracing::warn!(error = ?e, "reader reconcile: dense view refresh failed (non-fatal)");
            let _ = wtxn.abort();
            return;
        }
        if let Err(e) = wtxn.commit() {
            tracing::warn!(error = ?e, "reader reconcile: local txn commit failed (non-fatal)");
        }
    }

    /// Give a reader replica handles for the indexes the writer has registered
    /// since it opened the collection. Its handles came only from the metadata
    /// it opened with, so a replica that opened a collection before its indexes
    /// were built scanned those fields for the life of the handle.
    /// `has_payload_index_at` still requires each read's own metadata to
    /// register the index, so a handle adopted here cannot serve a read that
    /// predates the index.
    fn adopt_registered_payload_indexes(&self, registered: &HashMap<String, PayloadIndexSchema>) {
        let stale = |handles: &HashMap<String, PayloadIndexHandle>| {
            registered.iter().any(|(field, schema)| {
                handles
                    .get(field)
                    .is_none_or(|handle| &handle.schema != schema)
            })
        };
        if !self
            .payload_indices
            .read()
            .is_ok_and(|handles| stale(&handles))
        {
            return;
        }
        // try_write: a writer waiting on this lock would block every query's
        // read of it. A busy round is picked up by the next refresh.
        let Ok(mut handles) = self.payload_indices.try_write() else {
            return;
        };
        for (field, schema) in registered {
            if handles
                .get(field)
                .is_none_or(|handle| &handle.schema != schema)
            {
                handles.insert(field.clone(), PayloadIndexHandle::ready_lsm(schema.clone()));
            }
        }
    }

    /// Rate-limited entry point for `refresh_reader_view_best_effort`. At most
    /// one reconcile is initiated per `ttl_ms` window per collection regardless
    /// of QPS. Refresh runs on a detached thread: object-store metadata reads and
    /// dense sidecar reopens must not stall foreground reader-replica requests.
    pub(crate) fn maybe_refresh_reader_view(self: &Arc<Self>, ttl_ms: u64) {
        if !self.backend.is_reader_replica() {
            return;
        }
        let now = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            Ok(d) => d.as_millis().min(u64::MAX as u128) as u64,
            Err(_) => return,
        };
        if self
            .reader_refresh_inflight
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Relaxed,
            )
            .is_err()
        {
            metrics::counter!(
                "helix_reader_view_refresh_total",
                "outcome" => "already_inflight"
            )
            .increment(1);
            return;
        }
        let last = self
            .last_reader_refresh_ms
            .load(std::sync::atomic::Ordering::Relaxed);
        if now.saturating_sub(last) < ttl_ms {
            self.reader_refresh_inflight
                .store(false, std::sync::atomic::Ordering::Release);
            return;
        }
        if self
            .last_reader_refresh_ms
            .compare_exchange(
                last,
                now,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Relaxed,
            )
            .is_err()
        {
            self.reader_refresh_inflight
                .store(false, std::sync::atomic::Ordering::Release);
            return;
        }
        let storage = Arc::clone(self);
        metrics::counter!(
            "helix_reader_view_refresh_total",
            "outcome" => "scheduled"
        )
        .increment(1);
        if let Err(err) = std::thread::Builder::new()
            .name("helix-reader-refresh".to_string())
            .spawn(move || {
                let started = Instant::now();
                storage.refresh_reader_view_best_effort();
                storage
                    .reader_refresh_inflight
                    .store(false, std::sync::atomic::Ordering::Release);
                metrics::histogram!("helix_reader_view_refresh_duration_ms")
                    .record(started.elapsed().as_secs_f64() * 1000.0);
                metrics::counter!(
                    "helix_reader_view_refresh_total",
                    "outcome" => "completed"
                )
                .increment(1);
            })
        {
            self.reader_refresh_inflight
                .store(false, std::sync::atomic::Ordering::Release);
            metrics::counter!(
                "helix_reader_view_refresh_total",
                "outcome" => "spawn_failed"
            )
            .increment(1);
            tracing::warn!(error = %err, "failed to spawn reader view refresh");
        }
    }

    fn note_reader_view_reconciled(&self, covers: Option<(u64, u64)>) {
        let Some(covers) = covers else {
            return;
        };
        let sync = &self.reader_view_reconcile;
        sync.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .covered = Some(covers);
        sync.finished.notify_all();
    }

    /// Tests: keep reconciles from running, as a slow one would.
    #[cfg(test)]
    pub(crate) fn hold_reader_view_reconcile(&self) -> MutexGuard<'_, ()> {
        self.reader_view_reconcile
            .run
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Reader replicas: a read whose data view is past the one the dense
    /// segment list and payload index handles were last reconciled against
    /// reconciles first, and waits up to `wait` for it. The timed refresh
    /// alone left a read after a quiet spell on the list of an older view:
    /// dense search missed the points of a tail the writer had opened since,
    /// and the points of segments a merge had replaced and reaped.
    ///
    /// Detection is in memory (`AnyBackend::lsm_reader_view`). The reconcile
    /// runs on its own thread, so the bound holds whatever it is doing: a read
    /// that reaches it goes on with the list it has, exactly as stale as
    /// before this existed, and the reconcile finishes for the next read.
    /// Reads arriving while that slow one still runs do not wait again, and a
    /// view gets one attempt, so a failing reconcile cannot be retried by
    /// every read behind it.
    pub(crate) fn reconcile_reader_view_if_advanced(
        self: &Arc<Self>,
        wait: Duration,
    ) -> ReaderViewWait {
        let Some(view) = self.backend.lsm_reader_view() else {
            return ReaderViewWait::Current;
        };
        let sync = &self.reader_view_reconcile;
        let mut state = sync
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.covered >= Some(view) {
            return ReaderViewWait::Current;
        }
        let started = Instant::now();
        let outcome = loop {
            if state.covered >= Some(view) {
                break ReaderViewWait::Reconciled;
            }
            if !state.running {
                if state.attempted >= Some(view) {
                    break ReaderViewWait::Unreconciled;
                }
                state.attempted = Some(view);
                let storage = Arc::clone(self);
                let spawned = std::thread::Builder::new()
                    .name("helix-reader-reconcile".to_string())
                    .spawn(move || {
                        let started = Instant::now();
                        storage.refresh_reader_view_best_effort();
                        metrics::histogram!("helix_reader_view_refresh_duration_ms")
                            .record(started.elapsed().as_secs_f64() * 1000.0);
                        let sync = &storage.reader_view_reconcile;
                        let mut state = sync
                            .state
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        state.running = false;
                        state.gave_up = false;
                        drop(state);
                        sync.finished.notify_all();
                    });
                if let Err(error) = spawned {
                    tracing::warn!(error = %error, "failed to spawn reader view reconcile");
                    break ReaderViewWait::Unreconciled;
                }
                state.running = true;
                // The timed refresh is not due again right behind this one.
                self.last_reader_refresh_ms.store(
                    Self::now_epoch_millis(),
                    std::sync::atomic::Ordering::Relaxed,
                );
            } else if state.gave_up {
                break ReaderViewWait::BehindSlow;
            }
            let remaining = wait.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                state.gave_up = true;
                break ReaderViewWait::TimedOut;
            }
            state = sync
                .finished
                .wait_timeout(state, remaining)
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .0;
        };
        drop(state);
        let label = match outcome {
            ReaderViewWait::Current | ReaderViewWait::Reconciled => "reconciled",
            ReaderViewWait::TimedOut => {
                // The others waiting on the same reconcile give up with it.
                sync.finished.notify_all();
                "timeout"
            }
            ReaderViewWait::BehindSlow => "behind_slow",
            ReaderViewWait::Unreconciled => "unreconciled",
        };
        metrics::counter!("helix_reader_view_reconcile_wait_total", "outcome" => label)
            .increment(1);
        metrics::histogram!("helix_reader_view_reconcile_wait_ms", "outcome" => label)
            .record(started.elapsed().as_secs_f64() * 1000.0);
        outcome
    }

    /// Read-gated `DbReader` poll-tier promotion (S3 LIST-cost reduction):
    /// stamps this read as activity and, if this reader-replica collection is
    /// believed to be on the idle poll tier, schedules a single-flight background
    /// promotion to the active/serving tier. Where the promotion is a reopen,
    /// the foreground read serves the current checkpoint instead of taking
    /// part in it. Where it is done in place it is one manifest poll, and the
    /// read that started it waits for that poll, at most
    /// [`READER_PROMOTE_POLL_WAIT`], so that it reads what was written while
    /// the collection sat on the idle poll; past the bound it goes on with
    /// what it has, and so do the reads that find a promotion in flight.
    /// No-op on writer/LMDB backends and when `HELIX_LSM_READER_IDLE_AFTER_MS=0`
    /// (kill switch, preserves prior behavior exactly).
    ///
    /// Demotion is the mirror image, driven by the periodic sweep in
    /// `collection_manager` (see `reader_read_recently`) rather than from the
    /// read path.
    pub(crate) fn note_reader_read_and_maybe_promote(self: &Arc<Self>) {
        if !self.backend.is_reader_replica() {
            return;
        }
        let idle_after =
            crate::helix_engine::storage_core::backend_lsm::reader_idle_after_from_env();
        if idle_after.is_zero() {
            return;
        }
        let now = Self::now_epoch_millis();
        self.last_reader_read_at_ms
            .store(now, std::sync::atomic::Ordering::Relaxed);
        if self.reader_poll_tier_is_fast() {
            return;
        }
        let last_failed = self
            .last_reader_promote_failed_at_ms
            .load(std::sync::atomic::Ordering::Relaxed);
        let consecutive_failures = self
            .reader_promote_consecutive_failures
            .load(std::sync::atomic::Ordering::Relaxed);
        if !Self::reader_poll_promote_retry_due(last_failed, now, consecutive_failures) {
            metrics::counter!(
                "helix_reader_poll_tier_total",
                "direction" => "promote_retry_cooldown"
            )
            .increment(1);
            return;
        }
        if !self.try_begin_poll_tier_transition() {
            // A transition (promote or demote) is already in flight for this
            // collection; serve the current view rather than wait.
            return;
        }
        let active_interval =
            crate::helix_engine::storage_core::backend_lsm::reader_active_poll_interval_from_env();
        let storage = Arc::clone(self);
        // Dropped by the thread when the promotion is over, whatever came of it.
        let (promotion_over, wait_for_promotion) = std::sync::mpsc::channel::<()>();
        let waits = self.backend.lsm_reader_in_place_tier_is_fast().is_some();
        if let Err(error) = std::thread::Builder::new()
            .name("helix-reader-promote".to_string())
            .spawn(move || {
                let _promotion_over = promotion_over;
                match storage.backend.retier_lsm_reader(active_interval) {
                    Ok(_) => {
                        storage
                            .last_reader_promote_failed_at_ms
                            .store(0, std::sync::atomic::Ordering::Relaxed);
                        storage
                            .reader_promote_consecutive_failures
                            .store(0, std::sync::atomic::Ordering::Relaxed);
                        storage.set_reader_poll_tier_fast(true);
                        metrics::counter!("helix_reader_poll_tier_total", "direction" => "promote")
                            .increment(1);
                    }
                    Err(error) => {
                        metrics::counter!(
                            "helix_reader_poll_tier_total",
                            "direction" => "promote_failed"
                        )
                        .increment(1);
                        storage.last_reader_promote_failed_at_ms.store(
                            HelixGraphStorage::now_epoch_millis(),
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        let failures = storage
                            .reader_promote_consecutive_failures
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                            .saturating_add(1);
                        tracing::warn!(
                            error = %error,
                            consecutive_failures = failures,
                            retry_backoff_ms =
                                HelixGraphStorage::reader_poll_promote_cooldown_ms(failures),
                            "reader poll-tier promotion failed; serving from existing checkpoint"
                        );
                    }
                }
                storage.end_poll_tier_transition();
            })
        {
            metrics::counter!(
                "helix_reader_poll_tier_total",
                "direction" => "promote_spawn_failed"
            )
            .increment(1);
            self.last_reader_promote_failed_at_ms
                .store(now, std::sync::atomic::Ordering::Relaxed);
            let failures = self
                .reader_promote_consecutive_failures
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                .saturating_add(1);
            tracing::warn!(
                error = %error,
                consecutive_failures = failures,
                retry_backoff_ms = HelixGraphStorage::reader_poll_promote_cooldown_ms(failures),
                "reader poll-tier promotion spawn failed; serving from existing checkpoint"
            );
            self.end_poll_tier_transition();
        } else if waits
            && wait_for_promotion.recv_timeout(READER_PROMOTE_POLL_WAIT)
                == Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        {
            metrics::counter!(
                "helix_reader_poll_tier_total",
                "direction" => "promote_wait_timeout"
            )
            .increment(1);
        }
    }

    /// True if a read landed on this collection within `window`. Consulted by
    /// the read-gated poll-tier demotion sweep (`collection_manager`) — and by
    /// the write-feed change poller's own demotion — so neither ever demotes a
    /// collection an active reader still needs at the fast cadence.
    pub(crate) fn reader_read_recently(&self, window: std::time::Duration) -> bool {
        let last = self
            .last_reader_read_at_ms
            .load(std::sync::atomic::Ordering::Relaxed);
        if last == 0 {
            return false;
        }
        let now = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            Ok(d) => d.as_millis().min(u64::MAX as u128) as u64,
            Err(_) => return true,
        };
        now.saturating_sub(last) <= window.as_millis().min(u64::MAX as u128) as u64
    }

    /// Which poll tier this collection's `DbReader` is on. A reader that
    /// changes tiers in place says so itself, from the interval it polls at.
    /// A stored flag is written after the change it records, so two threads
    /// that change the tier at once can leave it saying fast over a reader on
    /// the idle poll, and then no read promotes it; asking the reader has no
    /// such state. Where a tier change is a reopen, the stored flag is the
    /// answer, as before.
    pub(crate) fn reader_poll_tier_is_fast(&self) -> bool {
        self.backend
            .lsm_reader_in_place_tier_is_fast()
            .unwrap_or_else(|| {
                self.reader_poll_tier_fast
                    .load(std::sync::atomic::Ordering::Acquire)
            })
    }

    /// Single source of truth for "which poll tier does this collection's
    /// `DbReader` actually think it's on". Every caller that successfully
    /// changes the underlying poll interval — this storage's own read-gated
    /// promotion/demotion AND the write-feed change poller's promote/demote —
    /// must call this afterward, so neither subsystem's belief about the
    /// current tier drifts from what the other one just did.
    pub(crate) fn set_reader_poll_tier_fast(&self, fast: bool) {
        self.reader_poll_tier_fast
            .store(fast, std::sync::atomic::Ordering::Release);
    }

    /// Acquire the poll-tier transition guard; `false` means a promotion or
    /// demotion is already in flight for this collection.
    pub(crate) fn try_begin_poll_tier_transition(&self) -> bool {
        self.reader_poll_tier_transition
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Relaxed,
            )
            .is_ok()
    }

    pub(crate) fn end_poll_tier_transition(&self) {
        self.reader_poll_tier_transition
            .store(false, std::sync::atomic::Ordering::Release);
    }

    /// True while a promotion or demotion holds the transition guard: the
    /// tier flag still names the tier that transition is leaving.
    pub(crate) fn poll_tier_transition_in_flight(&self) -> bool {
        self.reader_poll_tier_transition
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Called by the write-feed change poller's promote success arm instead
    /// of `set_reader_poll_tier_fast(true)` directly: marks the tier fast AND
    /// stamps write-promotion recency, so the read-gated demotion sweep (which
    /// has no other visibility into write-feed activity) knows this
    /// collection was just promoted for a WRITE reason and shouldn't be
    /// demoted out from under an active write burst even if nobody has read
    /// it yet.
    pub(crate) fn note_write_feed_promotion(&self) {
        self.note_write_feed_change();
        self.set_reader_poll_tier_fast(true);
    }

    /// Called by the write-feed change poller for every change that finds the
    /// collection on the short poll: moves the write stamp only. A burst that
    /// outlasts the sweeper's window would otherwise be demoted to the idle
    /// tier while its writes continue. Never touches the tier flag: the
    /// sweeper may have demoted between the poller's check and this call, and
    /// the flag must keep saying so for the next change to promote again.
    pub(crate) fn note_write_feed_change(&self) {
        let now = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            Ok(d) => d.as_millis().min(u64::MAX as u128) as u64,
            Err(_) => 0,
        };
        if now != 0 {
            self.last_write_promoted_at_ms
                .store(now, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// True if the write-feed poller saw a change of this collection within
    /// `window`, whether that change promoted it or found it on the short
    /// poll. Naturally always `false` when the write-feed isn't configured
    /// (nothing ever calls `note_write_feed_change`), so callers don't need
    /// to separately check whether the feed is enabled.
    pub(crate) fn write_promoted_recently(&self, window: std::time::Duration) -> bool {
        let last = self
            .last_write_promoted_at_ms
            .load(std::sync::atomic::Ordering::Relaxed);
        if last == 0 {
            return false;
        }
        let now = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            Ok(d) => d.as_millis().min(u64::MAX as u128) as u64,
            Err(_) => return true,
        };
        now.saturating_sub(last) <= window.as_millis().min(u64::MAX as u128) as u64
    }

    /// Acquire the per-collection recount guard; `false` means a recount is
    /// already in progress for this collection. `recount_lsm_counters` is the
    /// only production caller (via an RAII guard so release can't be
    /// forgotten on an early exit); exposed as a named method — rather than
    /// leaving the CAS inline — so tests can simulate an in-flight recount
    /// directly.
    pub(crate) fn try_begin_recount(&self) -> bool {
        self.recount_inflight
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Relaxed,
            )
            .is_ok()
    }

    pub(crate) fn end_recount(&self) {
        self.recount_inflight
            .store(false, std::sync::atomic::Ordering::Release);
    }

    /// Acquire the per-collection payload-index-GC guard; `false` means a GC
    /// run is already in progress for this collection. Mirrors
    /// `try_begin_recount`.
    pub(crate) fn try_begin_payload_index_gc(&self) -> bool {
        self.payload_index_gc_inflight
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Relaxed,
            )
            .is_ok()
    }

    pub(crate) fn end_payload_index_gc(&self) {
        self.payload_index_gc_inflight
            .store(false, std::sync::atomic::Ordering::Release);
    }

    /// Read the persisted per-collection HNSW overrides. Missing key or decode
    /// error returns `None` (caller falls back to global config).
    pub fn get_hnsw_overrides(&self, txn: &RoTxn) -> Result<Option<HnswOverrides>, GraphError> {
        self.backend
            .get_with_heed(
                txn,
                Namespace::Metadata,
                HNSW_OVERRIDES_KEY.as_bytes(),
                |v| v.and_then(|bytes| bincode::deserialize::<HnswOverrides>(bytes).ok()),
            )
            .map_err(|e| GraphError::New(e.to_string()))
    }

    pub(crate) fn get_hnsw_overrides_be(
        &self,
        r: &AnyRead<'_>,
    ) -> Result<Option<HnswOverrides>, GraphError> {
        self.backend
            .get_with(r, Namespace::Metadata, HNSW_OVERRIDES_KEY.as_bytes(), |v| {
                v.and_then(|bytes| bincode::deserialize::<HnswOverrides>(bytes).ok())
            })
            .map_err(|e| GraphError::New(e.to_string()))
    }

    /// Persist per-collection HNSW overrides. Caller owns the write txn and
    /// must commit. Empty overrides are deleted rather than stored so the
    /// on-disk footprint stays zero when the caller didn't opt in.
    pub fn set_hnsw_overrides(
        &self,
        txn: &mut RwTxn,
        overrides: &HnswOverrides,
    ) -> Result<(), GraphError> {
        if overrides.is_empty() {
            self.backend
                .delete_heed(txn, Namespace::Metadata, HNSW_OVERRIDES_KEY.as_bytes())
                .map_err(|e| GraphError::New(e.to_string()))?;
            return Ok(());
        }
        let bytes = bincode::serialize(overrides)
            .map_err(|e| GraphError::StorageError(format!("hnsw overrides serialize: {}", e)))?;
        self.backend
            .put_heed(
                txn,
                Namespace::Metadata,
                HNSW_OVERRIDES_KEY.as_bytes(),
                &bytes,
            )
            .map_err(|e| GraphError::New(e.to_string()))?;
        Ok(())
    }

    /// Acquire the shared resize gate before creating an LMDB transaction.
    ///
    /// The guard is held across `read_txn()` / `write_txn()` creation, then
    /// dropped. Resizes take the exclusive guard so `env.resize()` cannot remap
    /// while a transaction handle is being initialized.
    pub fn read_resize_guard_if_needed(&self) -> Result<Option<ResizeReadGuard<'_>>, GraphError> {
        self.read_resize_guard_for("generic_read")
    }

    /// True while at least one resize writer is queued or currently holding
    /// the exclusive `resize_gate`. The async gateway uses this as an
    /// admission signal so requests for the resizing collection wait before
    /// taking a Tokio blocking worker.
    pub(crate) fn resize_admission_blocked(&self) -> bool {
        self.resize_writers_pending.load(Ordering::Acquire) != 0
            || self.resize_writers_active.load(Ordering::Acquire) != 0
    }

    fn wait_on_resize_cv<'a>(&self, wait_guard: MutexGuard<'a, ()>) -> MutexGuard<'a, ()> {
        let (guard, _timeout) = match self
            .resize_wait_cv
            .wait_timeout(wait_guard, std::time::Duration::from_millis(250))
        {
            Ok(pair) => pair,
            Err(e) => {
                let (g, t) = e.into_inner();
                (g, t)
            }
        };
        guard
    }

    fn begin_lmdb_txn(&self, operation: &'static str) -> Result<LmdbTxnPermit<'_>, GraphError> {
        let started = Instant::now();
        let mut wait_guard = match self.resize_wait_lock.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        while self.resize_admission_blocked() {
            wait_guard = self.wait_on_resize_cv(wait_guard);
        }
        let active = self.active_lmdb_txns.fetch_add(1, Ordering::AcqRel) + 1;
        metrics::gauge!("helix_lmdb_active_tracked_txns").set(active as f64);
        drop(wait_guard);

        let waited = started.elapsed();
        metrics::histogram!("helix_lmdb_txn_admission_wait_ms", "operation" => operation)
            .record(waited.as_secs_f64() * 1000.0);
        if waited.as_millis() >= diag_warn_ms("HELIX_RESIZE_GATE_WAIT_WARN_MS", 5_000) {
            tracing::warn!(
                operation,
                waited_ms = waited.as_millis(),
                active_txns = active,
                "LMDB transaction admission waited for resize"
            );
        }
        Ok(LmdbTxnPermit {
            storage: self,
            operation,
            started: Instant::now(),
        })
    }

    fn begin_nested_lmdb_txn(
        &self,
        operation: &'static str,
    ) -> Result<LmdbTxnPermit<'_>, GraphError> {
        let started = Instant::now();
        let wait_guard = match self.resize_wait_lock.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        let active_before = self.active_lmdb_txns.load(Ordering::Acquire);
        if active_before == 0 {
            return Err(GraphError::StorageError(format!(
                "{} requires an active outer resize-safe transaction",
                operation
            )));
        }
        let active = self.active_lmdb_txns.fetch_add(1, Ordering::AcqRel) + 1;
        metrics::gauge!("helix_lmdb_active_tracked_txns").set(active as f64);
        drop(wait_guard);

        metrics::histogram!("helix_lmdb_nested_txn_admission_ms", "operation" => operation)
            .record(started.elapsed().as_secs_f64() * 1000.0);
        Ok(LmdbTxnPermit {
            storage: self,
            operation,
            started: Instant::now(),
        })
    }

    fn finish_lmdb_txn(&self, operation: &'static str, held: Duration) {
        let wait_guard = match self.resize_wait_lock.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        let previous = self.active_lmdb_txns.fetch_sub(1, Ordering::AcqRel);
        let active = previous.saturating_sub(1);
        metrics::gauge!("helix_lmdb_active_tracked_txns").set(active as f64);
        metrics::histogram!("helix_lmdb_tracked_txn_lifetime_ms", "operation" => operation)
            .record(held.as_secs_f64() * 1000.0);
        if previous <= 1 {
            self.resize_wait_cv.notify_all();
        }
        drop(wait_guard);
    }

    fn wait_for_lmdb_txns_to_drain(&self, operation: &'static str) {
        let started = Instant::now();
        let mut wait_guard = match self.resize_wait_lock.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        while self.active_lmdb_txns.load(Ordering::Acquire) != 0 {
            wait_guard = self.wait_on_resize_cv(wait_guard);
        }
        drop(wait_guard);
        let waited = started.elapsed();
        metrics::histogram!("helix_resize_active_txn_drain_wait_ms", "operation" => operation)
            .record(waited.as_secs_f64() * 1000.0);
        if waited.as_millis() >= diag_warn_ms("HELIX_RESIZE_GATE_WAIT_WARN_MS", 5_000) {
            tracing::warn!(
                operation,
                waited_ms = waited.as_millis(),
                "resize waited for active LMDB transactions to drain"
            );
        }
    }

    fn read_resize_guard_for(
        &self,
        operation: &'static str,
    ) -> Result<Option<ResizeReadGuard<'_>>, GraphError> {
        let started = Instant::now();
        // Park (not spin) while a resize writer is pending or active. The former
        // `while pending != 0 { sleep(1ms) }` loop burned CPU on any caller
        // that reached the storage layer. HTTP traffic should normally wait in
        // async gateway admission before entering this synchronous path; this
        // fallback still matters for direct storage callers and races.
        //
        // Correctness/safety: this only changes HOW a reader waits BEFORE it
        // attempts `resize_gate.read()`. It does not touch resize timing or
        // guard scope. The cv-mutex is released before `resize_gate.read()` is
        // taken, preserving the existing lock order (no nesting with
        // `resize_gate`, no inversion). Lost-wakeup safe: writer state changes
        // and `notify_all()` happen under this same mutex, and we re-check the
        // predicate in a `while` loop (spurious- and multi-writer-safe). The
        // waiting write still completes after the resize finishes — no request
        // is ever rejected.
        if self.resize_admission_blocked() {
            let mut wait_guard = match self.resize_wait_lock.lock() {
                Ok(g) => g,
                Err(e) => e.into_inner(),
            };
            while self.resize_admission_blocked() {
                // Bounded re-check interval is a safety net against a missed
                // wakeup; correctness does not depend on the timeout firing.
                let (g, _timeout) = match self
                    .resize_wait_cv
                    .wait_timeout(wait_guard, std::time::Duration::from_millis(250))
                {
                    Ok(pair) => pair,
                    Err(e) => {
                        let (g, t) = e.into_inner();
                        (g, t)
                    }
                };
                wait_guard = g;
            }
            drop(wait_guard);
        }
        let guard = self.resize_gate.read();
        let waited = started.elapsed();
        metrics::histogram!("helix_resize_gate_wait_ms", "mode" => "read", "operation" => operation)
            .record(waited.as_secs_f64() * 1000.0);
        if waited.as_millis() >= diag_warn_ms("HELIX_RESIZE_GATE_WAIT_WARN_MS", 5_000) {
            tracing::warn!(
                operation,
                waited_ms = waited.as_millis(),
                "resize_gate read lock acquisition is slow"
            );
        }
        Ok(Some(guard))
    }

    pub(crate) fn lock_write_txn_gate(&self) -> Result<MutexGuard<'_, ()>, GraphError> {
        self.lock_write_txn_gate_for("generic_write")
    }

    pub(crate) fn set_write_txn_gate(&mut self, gate: Arc<Mutex<()>>) {
        self.write_txn_gate = gate;
    }

    pub(crate) fn lock_write_txn_gate_for(
        &self,
        operation: &'static str,
    ) -> Result<MutexGuard<'_, ()>, GraphError> {
        let started = Instant::now();
        let guard = match self.write_txn_gate.lock() {
            Ok(guard) => guard,
            Err(e) => {
                tracing::error!(
                    operation,
                    error = %e,
                    "write_txn_gate lock poisoned; recovering coordination lock"
                );
                e.into_inner()
            }
        };
        let waited = started.elapsed();
        metrics::histogram!("helix_write_txn_gate_wait_ms", "operation" => operation)
            .record(waited.as_secs_f64() * 1000.0);
        if waited.as_millis() >= diag_warn_ms("HELIX_WRITE_TXN_GATE_WAIT_WARN_MS", 5_000) {
            tracing::warn!(
                operation,
                waited_ms = waited.as_millis(),
                "write_txn_gate acquisition is slow"
            );
        }
        Ok(guard)
    }

    /// Transition a queued resize writer into the active exclusive window.
    ///
    /// The active increment happens before the pending decrement so async
    /// pollers never observe a false gap between the two atomics.
    fn resize_writer_acquired(&self) {
        let _wake = match self.resize_wait_lock.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        self.resize_writers_active.fetch_add(1, Ordering::AcqRel);
        self.resize_writers_pending.fetch_sub(1, Ordering::AcqRel);
        self.resize_wait_cv.notify_all();
    }

    /// Decrement the active-resize counter and wake any parked readers.
    ///
    /// Must run when the exclusive resize guard drops. The decrement and
    /// `notify_all()` happen under `resize_wait_lock` so a reader cannot slip
    /// between its predicate re-check and `cv.wait()` and miss the wakeup.
    fn resize_writer_finished(&self) {
        let _wake = match self.resize_wait_lock.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        self.resize_writers_active.fetch_sub(1, Ordering::AcqRel);
        self.resize_wait_cv.notify_all();
    }

    pub(crate) fn write_resize_guard_for(
        &self,
        operation: &'static str,
    ) -> Result<ResizeWriteGuard<'_>, GraphError> {
        let started = Instant::now();
        {
            let _wake = match self.resize_wait_lock.lock() {
                Ok(g) => g,
                Err(e) => e.into_inner(),
            };
            self.resize_writers_pending.fetch_add(1, Ordering::AcqRel);
            self.resize_wait_cv.notify_all();
        }
        self.wait_for_lmdb_txns_to_drain(operation);
        let guard = self.resize_gate.write();
        self.resize_writer_acquired();
        let waited = started.elapsed();
        metrics::histogram!("helix_resize_gate_wait_ms", "mode" => "write", "operation" => operation)
            .record(waited.as_secs_f64() * 1000.0);
        if waited.as_millis() >= diag_warn_ms("HELIX_RESIZE_GATE_WAIT_WARN_MS", 5_000) {
            tracing::warn!(
                operation,
                waited_ms = waited.as_millis(),
                "resize_gate write lock acquisition is slow"
            );
        }
        Ok(ResizeWriteGuard {
            storage: self,
            guard: Some(guard),
        })
    }

    pub(crate) fn try_write_resize_guard_for(
        &self,
        operation: &'static str,
    ) -> Result<Option<ResizeWriteGuard<'_>>, GraphError> {
        let started = Instant::now();
        {
            let _wake = match self.resize_wait_lock.lock() {
                Ok(g) => g,
                Err(e) => e.into_inner(),
            };
            if self.resize_admission_blocked() || self.active_lmdb_txns.load(Ordering::Acquire) != 0
            {
                metrics::counter!(
                    "helix_resize_gate_try_write_total",
                    "operation" => operation,
                    "outcome" => "active_txns"
                )
                .increment(1);
                return Ok(None);
            }
            self.resize_writers_active.fetch_add(1, Ordering::AcqRel);
            self.resize_wait_cv.notify_all();
        }
        match self.resize_gate.try_write() {
            Some(guard) => {
                metrics::counter!(
                    "helix_resize_gate_try_write_total",
                    "operation" => operation,
                    "outcome" => "acquired"
                )
                .increment(1);
                let waited = started.elapsed();
                metrics::histogram!(
                    "helix_resize_gate_wait_ms",
                    "mode" => "write_try",
                    "operation" => operation
                )
                .record(waited.as_secs_f64() * 1000.0);
                Ok(Some(ResizeWriteGuard {
                    storage: self,
                    guard: Some(guard),
                }))
            }
            None => {
                self.resize_writer_finished();
                metrics::counter!(
                    "helix_resize_gate_try_write_total",
                    "operation" => operation,
                    "outcome" => "busy"
                )
                .increment(1);
                Ok(None)
            }
        }
    }

    /// Maximum number of auto-resize retries before giving up.
    ///
    /// With linear growth (`HELIX_MAP_GROW_MB` per step, default 1024 MB) and a
    /// 48 GB per-env cap, 48 retries let a single write reach the cap in one
    /// shot. Previously capped at 12 retries × 256 MB = ~3.1 GB, which surfaced
    /// MapFull at a fraction of the 8 GB env cap — callers saw 500s even though
    /// the env could have grown further.
    const MAX_RESIZE_RETRIES: usize = 48;

    /// Maximum map size cap per collection.
    ///
    /// With 5 000+ collections and LRU capping active envs to ~256, the worst
    /// case is 256 × 48 GB = 12 TB VA, well within Linux's 128 TB limit.
    /// Override via `HELIX_MAX_MAP_SIZE_GB` env var for large single-tenant
    /// deployments where a few collections hold tens of millions of vectors.
    fn max_map_size() -> usize {
        std::env::var("HELIX_MAX_MAP_SIZE_GB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(48)
            * 1024
            * 1024
            * 1024
    }

    /// Pre-grow margin in bytes. If available headroom drops below this margin
    /// plus the requested additional bytes, `ensure_map_headroom()` triggers
    /// a resize early to avoid `MapFull` stalls during ingest spikes.
    ///
    /// Default is 0 (proactive growth disabled unless headroom is exhausted).
    /// Configure via `HELIX_PRE_GROW_MB` env var.
    fn pre_grow_bytes() -> usize {
        env_mb("HELIX_PRE_GROW_MB", 0)
    }

    /// Maximum named LMDB databases per collection env.
    ///
    /// Dense vector segmenting creates multiple named DBs per segment. The
    /// segment reaper uses chunked clear instead of delete-and-close, so DBI
    /// slot pressure grows unless segment names are reused. The default leaves
    /// enough room for churn while costing only a few hundred KiB per open env.
    fn max_dbs() -> u32 {
        std::env::var("HELIX_MAX_DBS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|v| *v >= 8192)
            .unwrap_or(DEFAULT_LMDB_MAX_DBS)
    }

    fn initial_map_size(path: &str, config: &Config) -> usize {
        let requested_mb = std::env::var("HELIX_INITIAL_MAP_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .or_else(|| {
                config
                    .vector_config
                    .db_max_size
                    .map(|gb| gb.saturating_mul(1024))
            })
            .unwrap_or(DEFAULT_DB_INITIAL_MAP_MB)
            .max(1);
        let requested = requested_mb.saturating_mul(1024).saturating_mul(1024);
        let cap = Self::max_map_size();
        let existing = fs::metadata(Path::new(path).join("data.mdb"))
            .ok()
            .map(|metadata| metadata.len() as usize)
            .unwrap_or(0);
        if existing == 0 {
            return requested.min(cap);
        }

        let reopened_with_headroom = existing.saturating_add(Self::reopen_map_headroom_bytes());
        requested
            .max(existing)
            .max(reopened_with_headroom)
            .min(cap)
            .max(existing)
    }

    /// Extra LMDB map slack when reopening an existing `data.mdb`.
    ///
    /// Without this, a collection whose file grew beyond `HELIX_INITIAL_MAP_MB`
    /// reopens with `map_size == data.mdb size`, immediately hits `MapFull`,
    /// and has to resize on the first write after every restart.
    fn reopen_map_headroom_bytes() -> usize {
        std::env::var("HELIX_REOPEN_MAP_HEADROOM_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|mb| mb.saturating_mul(1024).saturating_mul(1024))
            .unwrap_or_else(Self::map_grow_bytes)
    }
    /// Version byte prefix for metadata serialization. Enables future
    /// codec migration without breaking existing data.
    const METADATA_CODEC_V1: u8 = 0x01;

    /// Serialize StorageMetadata with version prefix for forward compatibility.
    fn serialize_metadata(meta: &StorageMetadata) -> Result<Vec<u8>, GraphError> {
        let mut buf = vec![Self::METADATA_CODEC_V1];
        let encoded = bincode::serialize(meta)?;
        buf.extend_from_slice(&encoded);
        Ok(buf)
    }

    /// Strict deserialize — returns an error on any parse failure.
    /// Handles both legacy (no prefix) and versioned formats. Legacy data
    /// starts with a u32 schema_version (bincode LE), whose first byte is
    /// 0x02. Our version prefix 0x01 is distinguishable.
    fn try_deserialize_metadata(bytes: &[u8]) -> Result<StorageMetadata, GraphError> {
        if bytes.is_empty() {
            return Err(GraphError::StorageError("empty metadata bytes".to_string()));
        }
        let result = if bytes[0] == Self::METADATA_CODEC_V1 {
            bincode::deserialize(&bytes[1..])
        } else {
            bincode::deserialize(bytes)
        };
        result.map_err(|e| GraphError::StorageError(format!("metadata decode: {}", e)))
    }

    /// Execute a closure within a resize-safe read transaction.
    ///
    /// Begin a read transaction that participates in the LMDB resize fence.
    /// The legacy `resize_gate` is not held for the query lifetime; instead an
    /// active-transaction permit is held until the transaction drops. Resize
    /// marks itself pending, parks new transaction admission, then waits for
    /// active permits to drain before calling `env.resize()`.
    ///
    /// Use this instead of `graph_env.read_txn()` directly whenever the
    /// read may run concurrently with write transactions that could trigger
    /// auto-resize.
    pub fn with_read_txn<F, T>(&self, f: F) -> Result<T, GraphError>
    where
        F: FnOnce(&RoTxn) -> Result<T, GraphError>,
    {
        self.ensure_not_degraded()?;
        let rtxn = self.begin_resize_safe_read_txn()?;
        let result = f(&rtxn);
        drop(rtxn);
        if let Err(error) = &result {
            self.mark_collection_degraded(error, "with_read_txn");
        }
        result
    }

    /// Begin a manually-managed read transaction. Prefer `with_read_txn`;
    /// this exists for code generators that need a named `txn` variable in
    /// emitted Rust.
    pub fn begin_resize_safe_read_txn(&self) -> Result<ResizeSafeReadTxn<'_>, GraphError> {
        self.ensure_not_degraded()?;
        let permit = self.begin_lmdb_txn("read_txn")?;
        match self.lmdb_env()?.read_txn() {
            Ok(txn) => Ok(ResizeSafeReadTxn {
                txn: Some(txn),
                _permit: permit,
            }),
            Err(err) => {
                drop(permit);
                let error = GraphError::from(err);
                self.mark_collection_degraded(&error, "begin_resize_safe_read_txn");
                Err(error)
            }
        }
    }

    fn begin_nested_resize_safe_read_txn(&self) -> Result<ResizeSafeReadTxn<'_>, GraphError> {
        self.ensure_not_degraded()?;
        let permit = self.begin_nested_lmdb_txn("nested_read_txn")?;
        match self.lmdb_env()?.read_txn() {
            Ok(txn) => Ok(ResizeSafeReadTxn {
                txn: Some(txn),
                _permit: permit,
            }),
            Err(err) => {
                drop(permit);
                let error = GraphError::from(err);
                self.mark_collection_degraded(&error, "begin_nested_resize_safe_read_txn");
                Err(error)
            }
        }
    }

    pub fn nested_dense_read_provider<'a, 'env>(
        &'a self,
        _outer: &'a ResizeSafeReadTxn<'env>,
    ) -> NestedDenseReadTxnProvider<'a, 'env> {
        NestedDenseReadTxnProvider {
            storage: self,
            _outer: std::marker::PhantomData,
        }
    }

    /// Begin a manually-managed write transaction that participates in the
    /// writer gate and the active-transaction resize fence.
    /// Prefer `with_write_txn` because it retries on `MapFull`; this exists for
    /// generated handlers that still own the transaction/commit shape.
    pub fn begin_resize_safe_write_txn(
        &self,
    ) -> Result<(MutexGuard<'_, ()>, ResizeSafeWriteTxn<'_>), GraphError> {
        self.ensure_not_degraded()?;
        MEMORY_WATERMARKS.apply_backpressure()?;
        let write_txn_gate = self.lock_write_txn_gate()?;
        let wtxn = self.begin_tracked_write_txn("manual_write_txn")?;
        Ok((write_txn_gate, wtxn))
    }

    fn begin_tracked_write_txn(
        &self,
        operation: &'static str,
    ) -> Result<ResizeSafeWriteTxn<'_>, GraphError> {
        let permit = self.begin_lmdb_txn(operation)?;
        match self.lmdb_env()?.write_txn() {
            Ok(txn) => Ok(ResizeSafeWriteTxn {
                txn: Some(txn),
                _permit: permit,
            }),
            Err(err) => {
                drop(permit);
                Err(GraphError::from(err))
            }
        }
    }

    /// Execute a write operation with automatic LMDB map resize on retryable
    /// LMDB capacity errors.
    ///
    /// The closure receives a mutable `RwTxn`. If any `put`/write inside the
    /// closure or the final `commit()` triggers `MDB_MAP_FULL` (surfaced as
    /// `GraphError::MapFull`, or the Linux/LMDB `Invalid argument (os error
    /// 22)` remap race), the transaction is dropped/consumed, the map is grown,
    /// and the closure is retried.
    ///
    /// The closure **must not** commit the transaction — `with_write_txn` commits
    /// automatically on success.
    ///
    /// # Idempotency requirement
    ///
    /// Because `f` is `FnMut`, it **may be called more than once** when a
    /// resize retry occurs. The closure must therefore be safe to re-execute:
    ///
    /// * Avoid side-effects that cannot be repeated (e.g. sending a network
    ///   request, appending to a log outside the transaction).
    /// * All LMDB writes inside `f` are inherently idempotent because the
    ///   previous transaction is aborted before retry — only external
    ///   side-effects need care.
    #[track_caller]
    pub fn with_write_txn<F, T>(&self, mut f: F) -> Result<T, GraphError>
    where
        F: FnMut(&mut RwTxn) -> Result<T, GraphError>,
    {
        // Capture the caller's source location so the long-txn WARN below
        // identifies WHICH call site is holding the LMDB writer, not just
        // that *some* `with_write_txn` was slow. Free at runtime — `Location`
        // is a `&'static` set by `#[track_caller]`.
        let caller = std::panic::Location::caller();
        // Coarse caller label for histogram routing. Uses the file basename +
        // line so long-write-txn warnings can be attributed to upsert vs delete
        // vs optimizer vs index_* call sites without exploding cardinality.
        let caller_label = {
            let f = caller.file();
            let base = f.rsplit('/').next().unwrap_or(f);
            format!("{}:{}", base, caller.line())
        };

        // Backpressure: if RSS is above the low watermark, slow down
        // or pause this write to give the OS time to page out cold data.
        // This is the primary defense against OOM — never rejects, only delays.
        self.ensure_not_degraded()?;
        if self.backend.kind() == BackendKind::Lsm {
            return Err(GraphError::StorageError(
                "with_write_txn is LMDB-only; LSM code must use SlateDB WriteView/AnyWrite"
                    .to_string(),
            ));
        }
        MEMORY_WATERMARKS.apply_backpressure()?;

        crate::diag::log("enter", "with_write_txn", "");
        // Do not let multiple Helix write workers hold shared resize guards
        // while queued on LMDB's single-writer mutex. A MapFull retry needs an
        // exclusive resize gate; queued shared holders can otherwise wedge the
        // collection with all blocking-admission slots occupied and no CPU use.
        let gate_wait_start = Instant::now();
        let _write_txn_gate = self.lock_write_txn_gate_for("with_write_txn")?;
        observe_write_txn_phase(
            "shared",
            &caller_label,
            "write_gate_wait",
            gate_wait_start.elapsed(),
        );
        crate::diag::log("got_gate", "with_write_txn", "");

        for attempt in 0..Self::MAX_RESIZE_RETRIES {
            if attempt > 0 {
                metrics::counter!(
                    "helix_lmdb_write_txn_retry_total",
                    "kind" => "shared",
                    "caller" => caller_label.to_string()
                )
                .increment(1);
            }
            crate::diag::log(
                "getting_shared",
                "with_write_txn",
                &format!("attempt={}", attempt),
            );
            let mut wtxn = self.begin_tracked_write_txn("with_write_txn")?;
            crate::diag::log(
                "got_tracked_txn",
                "with_write_txn",
                &format!("attempt={}", attempt),
            );
            let lmdb_writer_wait_start = Instant::now();
            observe_write_txn_phase(
                "shared",
                &caller_label,
                "lmdb_writer_wait",
                lmdb_writer_wait_start.elapsed(),
            );
            let txn_diag = TxnDiagSpan::new_at("write", "with_write_txn", caller);
            crate::diag::log(
                "got_wtxn",
                "with_write_txn",
                &format!("attempt={}", attempt),
            );
            let observe_hot_metrics = crate::telemetry::hot_path_metrics_enabled();
            let txn_start = observe_hot_metrics.then(std::time::Instant::now);
            let mutation_start = Instant::now();
            let mutation_result = f(&mut wtxn);
            observe_write_txn_phase(
                "shared",
                &caller_label,
                "mutation",
                mutation_start.elapsed(),
            );
            match mutation_result {
                Ok(val) => match {
                    // LMDB commit latency. Includes msync of dirty pages.
                    // High p99 here = LMDB I/O bound, not CPU. Compare
                    // against wal_fsync_duration_ms to attribute.
                    let commit_start = Instant::now();
                    let res = wtxn.commit().map_err(GraphError::from);
                    let commit_elapsed = commit_start.elapsed();
                    observe_write_txn_phase(
                        "shared",
                        &caller_label,
                        "commit_fsync",
                        commit_elapsed,
                    );
                    if let Some(txn_start) = txn_start {
                        metrics::histogram!(
                            "helix_lmdb_write_txn_commit_ms",
                            "kind" => "shared",
                            "caller" => caller_label.clone()
                        )
                        .record(commit_elapsed.as_secs_f64() * 1000.0);
                        metrics::histogram!(
                            "helix_lmdb_write_txn_total_ms",
                            "kind" => "shared",
                            "caller" => caller_label.clone()
                        )
                        .record(txn_start.elapsed().as_secs_f64() * 1000.0);
                    }
                    res
                } {
                    Ok(()) => {
                        drop(txn_diag);
                        crate::diag::log(
                            "exit_ok",
                            "with_write_txn",
                            &format!("attempt={}", attempt),
                        );
                        self.refresh_metadata_snapshot_best_effort();
                        return Ok(val);
                    }
                    Err(e) if Self::should_retry_after_lmdb_resize_error(&e) => {
                        // LMDB can report MDB_MAP_FULL during commit after
                        // all individual puts appeared to fit. Some remap
                        // races surface as EINVAL instead. In both cases the
                        // transaction is already consumed, so grow the map and
                        // rerun f.
                        drop(txn_diag);
                        Self::record_lmdb_resize_retry("shared", &caller_label, &e);
                        self.grow_map_inner()?;
                        if attempt + 1 < Self::MAX_RESIZE_RETRIES {
                            continue;
                        }
                        return Err(Self::lmdb_resize_retry_exhausted());
                    }
                    Err(e) => {
                        self.mark_collection_degraded(&e, "with_write_txn_commit");
                        drop(txn_diag);
                        return Err(e);
                    }
                },
                Err(e) if Self::should_retry_after_lmdb_resize_error(&e) => {
                    // Must drop the txn before resizing.
                    drop(wtxn);
                    drop(txn_diag);
                    Self::record_lmdb_resize_retry("shared", &caller_label, &e);
                    self.grow_map_inner()?;
                    if attempt + 1 < Self::MAX_RESIZE_RETRIES {
                        continue;
                    }
                    return Err(Self::lmdb_resize_retry_exhausted());
                }
                Err(e) => {
                    self.mark_collection_degraded(&e, "with_write_txn_mutation");
                    drop(txn_diag);
                    return Err(e);
                }
            }
        }
        Err(Self::lmdb_resize_retry_exhausted())
    }

    /// Execute a write transaction that is created under the exclusive resize gate.
    ///
    /// Prefer this for writes that need stronger admission control while LMDB
    /// creates the transaction. The exclusive gate is released once
    /// `write_txn()` returns, so long mutation/commit work does not block
    /// resize-safe readers from creating their own transactions.
    #[track_caller]
    pub fn with_exclusive_write_txn<F, T>(&self, mut f: F) -> Result<T, GraphError>
    where
        F: FnMut(&mut RwTxn) -> Result<T, GraphError>,
    {
        // See `with_write_txn` for rationale on `#[track_caller]` capture.
        let caller = std::panic::Location::caller();
        let caller_label = {
            let f = caller.file();
            let base = f.rsplit('/').next().unwrap_or(f);
            format!("{}:{}", base, caller.line())
        };

        self.ensure_not_degraded()?;
        MEMORY_WATERMARKS.apply_backpressure()?;

        crate::diag::log("enter", "with_exclusive_write_txn", "");
        let gate_wait_start = Instant::now();
        let _write_txn_gate = self.lock_write_txn_gate_for("with_exclusive_write_txn")?;
        observe_write_txn_phase(
            "exclusive",
            &caller_label,
            "write_gate_wait",
            gate_wait_start.elapsed(),
        );
        crate::diag::log("got_gate", "with_exclusive_write_txn", "");

        for attempt in 0..Self::MAX_RESIZE_RETRIES {
            let mut wtxn = self.begin_tracked_write_txn("with_exclusive_write_txn")?;
            crate::diag::log(
                "got_tracked_txn",
                "with_exclusive_write_txn",
                &format!("attempt={}", attempt),
            );
            let lmdb_writer_wait_start = Instant::now();
            observe_write_txn_phase(
                "exclusive",
                &caller_label,
                "lmdb_writer_wait",
                lmdb_writer_wait_start.elapsed(),
            );
            let txn_diag = TxnDiagSpan::new_at("write", "with_exclusive_write_txn", caller);
            let observe_hot_metrics = crate::telemetry::hot_path_metrics_enabled();
            let txn_start = observe_hot_metrics.then(std::time::Instant::now);
            let mutation_start = Instant::now();
            let mutation_result = f(&mut wtxn);
            observe_write_txn_phase(
                "exclusive",
                &caller_label,
                "mutation",
                mutation_start.elapsed(),
            );
            match mutation_result {
                Ok(val) => match {
                    let commit_start = Instant::now();
                    let res = wtxn.commit().map_err(GraphError::from);
                    let commit_elapsed = commit_start.elapsed();
                    observe_write_txn_phase(
                        "exclusive",
                        &caller_label,
                        "commit_fsync",
                        commit_elapsed,
                    );
                    if let Some(txn_start) = txn_start {
                        metrics::histogram!(
                            "helix_lmdb_write_txn_commit_ms",
                            "kind" => "exclusive",
                            "caller" => caller_label.clone()
                        )
                        .record(commit_elapsed.as_secs_f64() * 1000.0);
                        metrics::histogram!(
                            "helix_lmdb_write_txn_total_ms",
                            "kind" => "exclusive",
                            "caller" => caller_label.clone()
                        )
                        .record(txn_start.elapsed().as_secs_f64() * 1000.0);
                    }
                    res
                } {
                    Ok(()) => {
                        drop(txn_diag);
                        crate::diag::log(
                            "exit_ok",
                            "with_exclusive_write_txn",
                            &format!("attempt={}", attempt),
                        );
                        self.refresh_metadata_snapshot_best_effort();
                        return Ok(val);
                    }
                    Err(e) if Self::should_retry_after_lmdb_resize_error(&e) => {
                        drop(txn_diag);
                        Self::record_lmdb_resize_retry("exclusive", &caller_label, &e);
                        self.grow_map_inner()?;
                        if attempt + 1 < Self::MAX_RESIZE_RETRIES {
                            continue;
                        }
                        return Err(Self::lmdb_resize_retry_exhausted());
                    }
                    Err(e) => {
                        self.mark_collection_degraded(&e, "with_exclusive_write_txn_commit");
                        drop(txn_diag);
                        return Err(e);
                    }
                },
                Err(e) if Self::should_retry_after_lmdb_resize_error(&e) => {
                    drop(wtxn);
                    drop(txn_diag);
                    Self::record_lmdb_resize_retry("exclusive", &caller_label, &e);
                    self.grow_map_inner()?;
                    if attempt + 1 < Self::MAX_RESIZE_RETRIES {
                        continue;
                    }
                    return Err(Self::lmdb_resize_retry_exhausted());
                }
                Err(e) => {
                    self.mark_collection_degraded(&e, "with_exclusive_write_txn_mutation");
                    drop(txn_diag);
                    return Err(e);
                }
            }
        }
        Err(Self::lmdb_resize_retry_exhausted())
    }

    fn should_retry_after_lmdb_resize_error(error: &GraphError) -> bool {
        matches!(error, GraphError::MapFull) || error.is_retryable_lmdb_invalid_argument()
    }

    fn record_lmdb_resize_retry(kind: &'static str, caller_label: &str, error: &GraphError) {
        let reason = if matches!(error, GraphError::MapFull) {
            "map_full"
        } else {
            "invalid_argument"
        };
        metrics::counter!(
            "helix_lmdb_write_txn_resize_retry_total",
            "kind" => kind,
            "caller" => caller_label.to_string(),
            "reason" => reason
        )
        .increment(1);
        if reason == "invalid_argument" {
            tracing::warn!(
                kind,
                caller = caller_label,
                error = %error,
                "retrying LMDB write transaction after retryable invalid argument"
            );
        }
    }

    fn lmdb_resize_retry_exhausted() -> GraphError {
        GraphError::StorageError(
            "LMDB write transaction still needed resize after maximum retries".to_string(),
        )
    }

    /// Grow step size for LMDB map auto-resize.
    ///
    /// Instead of doubling (which causes exponential memory blowup when
    /// many collections resize concurrently), we grow by a fixed increment.
    /// Default: 1024 MB per step. Override with `HELIX_MAP_GROW_MB`.
    ///
    /// At 1024 MB per step with MAX_RESIZE_RETRIES=48, a single collection
    /// can grow from 64 MB to ~48 GB before failing — matching the default
    /// `HELIX_MAX_MAP_SIZE_GB=48` cap. For tenants beyond that, bump
    /// `HELIX_MAX_MAP_SIZE_GB` per-deployment.
    fn map_grow_bytes() -> usize {
        env_mb("HELIX_MAP_GROW_MB", 1024)
    }

    fn giant_map_min_used_bytes() -> usize {
        env_mb("HELIX_GIANT_MAP_MIN_USED_MB", 32 * 1024)
    }

    fn giant_segment_threshold() -> Option<usize> {
        std::env::var("HELIX_GIANT_SEGMENT_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .or(Some(64))
            .filter(|value| *value > 0)
    }

    fn giant_pre_grow_bytes() -> usize {
        env_mb("HELIX_GIANT_PRE_GROW_MB", 32 * 1024)
    }

    fn giant_map_grow_bytes() -> usize {
        env_mb("HELIX_GIANT_MAP_GROW_MB", 32 * 1024)
    }

    fn resize_hard_headroom_bytes() -> usize {
        env_mb("HELIX_RESIZE_HARD_HEADROOM_MB", 1024)
    }

    /// Deadline for the opportunistic giant pre-grow poll before it escalates
    /// to the blocking (reader-parking) gate acquire. `0` disables polling.
    fn resize_try_poll_ms() -> u64 {
        env_u64("HELIX_RESIZE_TRY_POLL_MS", 10_000)
    }

    fn is_giant_for_resize(&self, used: usize) -> bool {
        used >= Self::giant_map_min_used_bytes()
            || Self::giant_segment_threshold()
                .map(|threshold| self.named_vectors.indexed_dense_segment_count() >= threshold)
                .unwrap_or(false)
    }

    fn resize_margin_bytes_for_giant(is_giant: bool) -> usize {
        let normal = Self::pre_grow_bytes();
        if is_giant {
            normal.max(Self::giant_pre_grow_bytes())
        } else {
            normal
        }
    }

    fn map_grow_bytes_for_giant(is_giant: bool) -> usize {
        let normal = Self::map_grow_bytes();
        if is_giant {
            normal.max(Self::giant_map_grow_bytes())
        } else {
            normal
        }
    }

    fn round_to_page_size(size: usize, page_size: usize) -> usize {
        ((size + page_size - 1) / page_size) * page_size
    }

    fn resize_target(
        current: usize,
        available: usize,
        additional_bytes: usize,
        margin: usize,
        min_step: usize,
        page_size: usize,
        cap: usize,
    ) -> usize {
        let want = additional_bytes.saturating_add(additional_bytes / 4);
        let deficit = want.saturating_add(margin).saturating_sub(available);
        let step = deficit.max(min_step);
        Self::round_to_page_size(current.saturating_add(step).min(cap), page_size)
    }

    fn resize_backpressure_error(
        operation: &'static str,
        used: usize,
        available: usize,
        required: usize,
    ) -> GraphError {
        GraphError::ResizeBackpressure(format!(
            "{} needs LMDB resize runway but the resize gate is busy (used_mb={}, available_mb={}, required_mb={})",
            operation,
            used / (1024 * 1024),
            available / (1024 * 1024),
            required / (1024 * 1024)
        ))
    }

    /// Grow the LMDB map size by a fixed increment.
    ///
    /// Before resizing, applies write backpressure if the process RSS is
    /// above the low watermark. This slows down concurrent ingest instead
    /// of rejecting writes, preventing OOM kills without losing data.
    ///
    /// LMDB map_size is virtual address space, not committed RAM. The
    /// kernel pages in/out as needed — large maps are safe. The real
    /// memory cost comes from dirty pages and HNSW builds, which are
    /// throttled separately by the build semaphore.
    ///
    /// Returns `MapFull` only if the per-collection cap is reached.
    pub(crate) fn grow_map(&self) -> Result<(), GraphError> {
        // Apply bounded backpressure before announcing resize intent. Doing
        // even a short sleep under either write coordination gate would block
        // unrelated writers exactly when memory pressure is highest.
        MEMORY_WATERMARKS.apply_backpressure()?;
        self.grow_map_inner()
    }

    fn grow_map_inner(&self) -> Result<(), GraphError> {
        crate::diag::log("enter", "grow_map_inner", "");
        let env = self.lmdb_env()?;
        let pre_info = env.info();
        let pre_used = pre_info.last_page_number.saturating_mul(page_size::get());
        let is_giant = self.is_giant_for_resize(pre_used);
        // Giant collections use try-write so a MapFull retry cannot create a
        // minutes-long pending writer that blocks every fresh reader.
        let _exclusive = if is_giant {
            match self.try_write_resize_guard_for("grow_map_inner_giant")? {
                Some(guard) => guard,
                None => {
                    metrics::counter!(
                        "helix_resize_runway_backpressure_total",
                        "operation" => "grow_map_inner",
                        "reason" => "gate_busy"
                    )
                    .increment(1);
                    let available = pre_info.map_size.saturating_sub(pre_used);
                    return Err(Self::resize_backpressure_error(
                        "grow_map_inner",
                        pre_used,
                        available,
                        Self::resize_hard_headroom_bytes(),
                    ));
                }
            }
        } else {
            // Acquire exclusive lock — blocks until all shared holders have
            // dropped their LMDB transaction + guard.
            self.write_resize_guard_for("grow_map_inner")?
        };
        crate::diag::log("got_exclusive", "grow_map_inner", "");

        let current = env.info().map_size;
        let page_size = page_size::get();
        let cap = Self::max_map_size();
        let grow = Self::map_grow_bytes_for_giant(is_giant);
        let new_size = current.saturating_add(grow).min(cap);
        // Round up to page boundary
        let new_size = Self::round_to_page_size(new_size, page_size);
        if new_size <= current {
            return Err(GraphError::StorageError(format!(
                "LMDB map at maximum size ({} MB), cannot grow further",
                current / (1024 * 1024)
            )));
        }

        tracing::info!(
            from_mb = current / (1024 * 1024),
            to_mb = new_size / (1024 * 1024),
            grow_mb = grow / (1024 * 1024),
            rss_mb = current_rss_bytes() / (1024 * 1024),
            "LMDB auto-resize"
        );

        // The backend's catalog thread opens a write transaction on this env
        // outside the resize gate: wait for a pass in hand, hold off the next.
        let _catalog = self.backend.pause_lmdb_catalog();
        // SAFETY: exclusive resize_gate lock prevents guarded callers from
        // creating new LMDB transactions while the map is being resized, and
        // `_catalog` the catalog thread. The retrying caller already dropped
        // its RwTxn before calling grow_map.
        unsafe { env.resize(new_size)? };
        crate::diag::log(
            "resize_done",
            "grow_map_inner",
            &format!(
                "from_mb={} to_mb={}",
                current / (1024 * 1024),
                new_size / (1024 * 1024)
            ),
        );
        Ok(())
    }

    fn map_headroom_satisfied(
        current: usize,
        used: usize,
        available: usize,
        additional: usize,
        margin: usize,
    ) -> bool {
        let required = additional.saturating_add(margin);
        if available >= required {
            return true;
        }
        if margin == 0 {
            return false;
        }

        // Fresh LMDB envs spend a few MiB on metapages, DB metadata, and early
        // small writes before the first bulk-sized write arrives. When the
        // operator intentionally sizes the initial map to `additional + margin`,
        // keep the invariant that the next write fits but let the pre-grow
        // margin absorb a bounded fresh-map overhead instead of growing every
        // tiny collection by a full map step.
        let fresh_overhead_budget = (margin / 4).clamp(1024 * 1024, 16 * 1024 * 1024);
        used <= fresh_overhead_budget
            && current >= required
            && available >= additional
            && margin > used
    }

    /// Ensure the LMDB map has at least `additional_bytes` of free headroom
    /// above the currently used pages before the next write. Sizes the grow
    /// step to the caller's estimate instead of the fixed `HELIX_MAP_GROW_MB`
    /// increment, so single-shot writes that cannot be retried (e.g. the
    /// optimizer merge flush, which consumes `PreparedMerge` by value) don't
    /// surface `MapFull` part-way through the commit.
    ///
    /// Computes headroom as `map_size - (last_page_number * page_size)`. If
    /// that is already at least `additional_bytes + pre_grow_bytes()`, the
    /// call is a no-op. Otherwise grows to cover `used + additional_bytes`
    /// (over-provisioned by 25 % and rounded up to at least `map_grow_bytes()`),
    /// capped at the per-env ceiling.
    ///
    /// Caller must not hold an active write transaction.
    pub fn ensure_map_headroom(&self, additional_bytes: usize) -> Result<(), GraphError> {
        let page_size = page_size::get();
        let (probe_used, probe_available, margin, is_giant) = {
            // Quick shared-gated check avoids taking the writer gate on the
            // common path while still preventing `mdb_env_info` from racing a
            // concurrent map remap.
            let _shared = self.read_resize_guard_if_needed()?;
            let info = self.lmdb_env()?.info();
            let used = info.last_page_number.saturating_mul(page_size);
            let available = info.map_size.saturating_sub(used);
            let is_giant = self.is_giant_for_resize(used);
            let margin = Self::resize_margin_bytes_for_giant(is_giant);
            if Self::map_headroom_satisfied(
                info.map_size,
                used,
                available,
                additional_bytes,
                margin,
            ) {
                return Ok(());
            }
            (used, available, margin, is_giant)
        };

        if is_giant {
            let hard_required = additional_bytes.saturating_add(Self::resize_hard_headroom_bytes());
            match self.try_write_resize_guard_for("ensure_map_headroom_giant")? {
                Some(exclusive) => {
                    return self.ensure_map_headroom_with_exclusive(
                        exclusive,
                        additional_bytes,
                        margin,
                        Self::map_grow_bytes_for_giant(true),
                        "ensure_map_headroom_giant",
                    );
                }
                None if probe_available >= hard_required => {
                    metrics::counter!(
                        "helix_resize_runway_deferred_total",
                        "operation" => "ensure_map_headroom",
                        "reason" => "gate_busy_above_hard"
                    )
                    .increment(1);
                    tracing::debug!(
                        requested_mb = additional_bytes / (1024 * 1024),
                        margin_mb = margin / (1024 * 1024),
                        used_mb = probe_used / (1024 * 1024),
                        available_mb = probe_available / (1024 * 1024),
                        hard_required_mb = hard_required / (1024 * 1024),
                        "giant LMDB pre-grow deferred because resize gate is busy and write runway remains"
                    );
                    return Ok(());
                }
                // Giant maps with low headroom that couldn't acquire the try-lock
                // poll for it below before escalating to the blocking path. The
                // map is well under its cap and the resize would succeed —
                // waiting is safer than returning 503 to the caller.
                None => {}
            }
        }

        MEMORY_WATERMARKS.apply_backpressure()?;

        // A blocking writer parks every new reader on this env for its full
        // wait (writer priority), so camping behind one long in-flight read
        // stalls all traffic until that read drains. While the requested
        // write still fits in the current map, retry the non-queuing try-lock
        // instead and let readers flow; the grow slips in the moment the gate
        // goes quiet.
        if is_giant && self.poll_giant_headroom_grow(additional_bytes, margin)? {
            return Ok(());
        }

        crate::diag::log(
            "enter",
            "ensure_map_headroom",
            &format!(
                "requested_mb={} margin_mb={}",
                additional_bytes / (1024 * 1024),
                margin / (1024 * 1024)
            ),
        );
        let _exclusive = self.write_resize_guard_for("ensure_map_headroom")?;
        crate::diag::log("got_exclusive", "ensure_map_headroom", "");

        self.ensure_map_headroom_after_exclusive(
            additional_bytes,
            margin,
            Self::map_grow_bytes_for_giant(false),
            "ensure_map_headroom",
        )
    }

    /// Bounded opportunistic wait for the exclusive resize gate on giant maps
    /// with low runway. Returns `Ok(true)` once headroom is ensured — either
    /// the gate was acquired and the map grown here, or a concurrent resize
    /// already satisfied the margin. Returns `Ok(false)` to escalate to the
    /// blocking, reader-parking acquire: when the poll deadline expires, or
    /// when concurrent commits erode the remaining runway below the requested
    /// write (at that point `MapFull` is imminent and stalling readers is the
    /// lesser evil — the merge flush caller cannot survive `MapFull`).
    fn poll_giant_headroom_grow(
        &self,
        additional_bytes: usize,
        margin: usize,
    ) -> Result<bool, GraphError> {
        let poll_ms = Self::resize_try_poll_ms();
        if poll_ms == 0 {
            return Ok(false);
        }
        let started = Instant::now();
        let deadline = std::time::Duration::from_millis(poll_ms);
        let page_size = page_size::get();
        loop {
            if let Some(exclusive) =
                self.try_write_resize_guard_for("ensure_map_headroom_giant_poll")?
            {
                self.ensure_map_headroom_with_exclusive(
                    exclusive,
                    additional_bytes,
                    margin,
                    Self::map_grow_bytes_for_giant(true),
                    "ensure_map_headroom_giant_poll",
                )?;
                metrics::counter!(
                    "helix_resize_try_poll_total",
                    "outcome" => "acquired"
                )
                .increment(1);
                let waited_ms = started.elapsed().as_millis();
                if waited_ms >= diag_warn_ms("HELIX_RESIZE_GATE_WAIT_WARN_MS", 5_000) {
                    tracing::info!(
                        waited_ms,
                        "giant LMDB pre-grow acquired resize gate via poll without parking readers"
                    );
                }
                return Ok(true);
            }
            {
                // Re-probe under the shared gate: a concurrent resize may have
                // grown the map already, or concurrent commits may have eaten
                // the remaining runway.
                let _shared = self.read_resize_guard_if_needed()?;
                let info = self.lmdb_env()?.info();
                let used = info.last_page_number.saturating_mul(page_size);
                let available = info.map_size.saturating_sub(used);
                if Self::map_headroom_satisfied(
                    info.map_size,
                    used,
                    available,
                    additional_bytes,
                    margin,
                ) {
                    metrics::counter!(
                        "helix_resize_try_poll_total",
                        "outcome" => "satisfied_concurrently"
                    )
                    .increment(1);
                    return Ok(true);
                }
                if available < additional_bytes {
                    metrics::counter!(
                        "helix_resize_try_poll_total",
                        "outcome" => "escalated_low_runway"
                    )
                    .increment(1);
                    tracing::warn!(
                        polled_ms = started.elapsed().as_millis(),
                        available_mb = available / (1024 * 1024),
                        requested_mb = additional_bytes / (1024 * 1024),
                        "giant LMDB pre-grow runway eroded during poll; escalating to blocking resize"
                    );
                    return Ok(false);
                }
            }
            if started.elapsed() >= deadline {
                metrics::counter!(
                    "helix_resize_try_poll_total",
                    "outcome" => "escalated_deadline"
                )
                .increment(1);
                tracing::warn!(
                    polled_ms = started.elapsed().as_millis(),
                    "giant LMDB pre-grow poll deadline expired; escalating to blocking resize"
                );
                return Ok(false);
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
    }

    fn ensure_map_headroom_with_exclusive(
        &self,
        _exclusive: ResizeWriteGuard<'_>,
        additional_bytes: usize,
        margin: usize,
        min_step: usize,
        operation: &'static str,
    ) -> Result<(), GraphError> {
        crate::diag::log("got_exclusive", operation, "");
        self.ensure_map_headroom_after_exclusive(additional_bytes, margin, min_step, operation)
    }

    fn ensure_map_headroom_after_exclusive(
        &self,
        additional_bytes: usize,
        margin: usize,
        min_step: usize,
        operation: &'static str,
    ) -> Result<(), GraphError> {
        let page_size = page_size::get();
        // Recompute under the exclusive lock. Another thread may have grown
        // the map while we were queued, and the pre-lock `info` snapshot is
        // stale. Computing `target` from stale `current` can undersize the
        // resize or no-op incorrectly.
        let env = self.lmdb_env()?;
        let info = env.info();
        let current = info.map_size;
        let used = info.last_page_number.saturating_mul(page_size);
        let available = current.saturating_sub(used);
        if Self::map_headroom_satisfied(current, used, available, additional_bytes, margin) {
            return Ok(());
        }

        let cap = Self::max_map_size();
        let target = Self::resize_target(
            current,
            available,
            additional_bytes,
            margin,
            min_step,
            page_size,
            cap,
        );
        if target <= current {
            // Already at (or above) cap — caller will surface `MapFull`
            // if the write doesn't fit, which is the correct signal that
            // the tenant has outgrown its cap.
            return Ok(());
        }

        tracing::info!(
            from_mb = current / (1024 * 1024),
            to_mb = target / (1024 * 1024),
            requested_mb = additional_bytes / (1024 * 1024),
            margin_mb = margin / (1024 * 1024),
            used_mb = used / (1024 * 1024),
            rss_mb = current_rss_bytes() / (1024 * 1024),
            "LMDB sized pre-grow (ensure_map_headroom)"
        );

        // The backend's catalog thread opens a write transaction on this env
        // outside the resize gate: wait for a pass in hand, hold off the next.
        let _catalog = self.backend.pause_lmdb_catalog();
        // SAFETY: exclusive resize_gate lock prevents guarded callers from
        // creating new LMDB transactions while the map is being resized, and
        // `_catalog` the catalog thread. The caller contract forbids an
        // active RwTxn.
        unsafe { env.resize(target)? };
        crate::diag::log(
            "resize_done",
            operation,
            &format!(
                "from_mb={} to_mb={}",
                current / (1024 * 1024),
                target / (1024 * 1024)
            ),
        );
        Ok(())
    }

    pub fn update_metadata<F>(
        &self,
        txn: &mut RwTxn,
        update: F,
    ) -> Result<StorageMetadata, GraphError>
    where
        F: FnOnce(&mut StorageMetadata) -> Result<(), GraphError>,
    {
        let mut metadata = self
            .lmdb_metadata_db()?
            .get(txn, METADATA_CURRENT_KEY)?
            .map(Self::try_deserialize_metadata)
            .transpose()?
            .unwrap_or_else(|| StorageMetadata::new(Vec::new()));
        update(&mut metadata)?;
        self.put_metadata(txn, &metadata)?;
        Ok(metadata)
    }

    pub fn set_secondary_indices_metadata(
        &self,
        txn: &mut RwTxn,
        secondary_indices: Vec<String>,
    ) -> Result<StorageMetadata, GraphError> {
        self.update_metadata(txn, |metadata| {
            metadata.set_secondary_indices(secondary_indices);
            Ok(())
        })
    }

    pub fn set_payload_indices_metadata(
        &self,
        txn: &mut RwTxn,
        payload_indices: HashMap<String, PayloadIndexSchema>,
    ) -> Result<StorageMetadata, GraphError> {
        self.update_metadata(txn, |metadata| {
            metadata.set_payload_indices(payload_indices);
            Ok(())
        })
    }

    pub fn set_named_vectors_metadata(
        &self,
        txn: &mut RwTxn,
        named_vectors: HashMap<String, NamedVectorConfig>,
    ) -> Result<StorageMetadata, GraphError> {
        self.update_metadata(txn, |metadata| {
            metadata.set_named_vectors(named_vectors);
            Ok(())
        })
    }

    pub fn set_dense_vector_spaces_metadata(
        &self,
        txn: &mut RwTxn,
        dense_vector_spaces: HashMap<
            String,
            crate::helix_engine::vector_core::named_vectors::DenseVectorSpaceMetadata,
        >,
    ) -> Result<StorageMetadata, GraphError> {
        self.update_metadata(txn, |metadata| {
            metadata.set_dense_vector_spaces(dense_vector_spaces);
            Ok(())
        })
    }

    /// Mark that dense layout changed (new mutable segment created) and
    /// metadata needs to be flushed to LMDB. Called from the upsert hot path
    /// instead of writing metadata directly — the IndexExecutor flushes it
    /// later, avoiding a per-upsert LMDB write.
    pub fn mark_dense_layout_dirty(&self) {
        self.dense_metadata_dirty
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Write the current in-memory dense layout to LMDB. Unconditional: the
    /// caller owns the dirty-flag lifecycle (swap before, restore on failure)
    /// because `with_write_txn` re-runs its closure on MapFull retries — a
    /// flag cleared inside the closure would turn the retry into a silent
    /// no-op that commits nothing while reporting success.
    pub fn flush_dense_metadata(&self, txn: &mut RwTxn) -> Result<(), GraphError> {
        self.set_dense_vector_spaces_metadata(txn, self.named_vectors.list_dense_vector_spaces())
            .map(|_| ())
    }

    pub fn set_sparse_vectors_metadata(
        &self,
        txn: &mut RwTxn,
        sparse_vectors: HashMap<
            String,
            crate::helix_engine::vector_core::sparse::SparseVectorConfig,
        >,
    ) -> Result<StorageMetadata, GraphError> {
        self.update_metadata(txn, |metadata| {
            metadata.set_sparse_vectors(sparse_vectors);
            Ok(())
        })
    }

    pub fn set_indexing_threshold_override_metadata(
        &self,
        txn: &mut RwTxn,
        indexing_threshold_override: Option<usize>,
    ) -> Result<StorageMetadata, GraphError> {
        self.update_metadata(txn, |metadata| {
            metadata.set_indexing_threshold_override(indexing_threshold_override);
            Ok(())
        })
    }

    pub fn adjust_metadata_counter(
        &self,
        txn: &mut RwTxn,
        counter: MetadataCounter,
        delta: i64,
    ) -> Result<StorageMetadata, GraphError> {
        self.update_metadata(txn, |metadata| metadata.adjust_counter(counter, delta))
    }

    #[inline(always)]
    pub fn new_node(label: &str, properties: impl IntoIterator<Item = (String, Value)>) -> Node {
        Node {
            id: v6_uuid(),
            label: label.to_string(),
            properties: HashMap::from_iter(properties),
        }
    }

    #[inline(always)]
    pub fn new_edge(
        label: &str,
        from_node: u128,
        to_node: u128,
        properties: impl IntoIterator<Item = (String, Value)>,
    ) -> Edge {
        Edge {
            id: v6_uuid(),
            label: label.to_string(),
            from_node,
            to_node,
            properties: HashMap::from_iter(properties),
        }
    }

    /// Returns the first node in key order (lexicographically smallest ID).
    /// Not actually random — name kept for backward compatibility with callers.
    pub fn get_first_node(&self, txn: &RoTxn) -> Result<Node, GraphError> {
        // Routed through the backend seam (heed-txn scaffold). `nodes_db.first`
        // returned the lexicographically-smallest key; scanning Namespace::Nodes
        // over the full keyspace and taking the first visited row is byte-identical
        // (the visitor returns `false` to stop after the first key). Decode happens
        // inside the closure-scoped borrow.
        use super::backend::KeyRange;

        let mut first: Option<Result<Node, GraphError>> = None;
        self.backend
            .scan_heed(txn, Namespace::Nodes, KeyRange::all(), |_k, v| {
                first = Some(bincode::deserialize(v).map_err(GraphError::from));
                false
            })
            .map_err(|e| GraphError::New(e.to_string()))?;
        match first {
            Some(node) => node,
            None => Err(GraphError::NodeNotFound),
        }
    }

    // todo look into using a shorter hash for space efficiency
    // #[inline(always)]
    // pub fn hash_label(label: &str) -> [u8; 4] {
    //     let mut hash = twox_hash::XxHash32::with_seed(0);
    //     hash.write(label.as_bytes());
    //     hash.finish_32().to_le_bytes()
    // }

    // #[inline(always)]
    // pub fn node_key(id: &u128) -> [u8; 16] {
    //     id.to_be_bytes()
    // }

    // #[inline(always)]
    // pub fn edge_key(id: &u128) -> [u8; 16] {
    //     id.to_be_bytes()
    // }

    #[inline(always)]
    pub fn node_key(id: &u128) -> &u128 {
        id
    }

    #[inline(always)]
    pub fn edge_key(id: &u128) -> &u128 {
        id
    }

    #[inline(always)]
    pub fn node_label_key(label: &[u8; 4], id: &u128) -> [u8; 20] {
        let mut key = [0u8; 20];
        key[0..4].copy_from_slice(label);
        key[4..20].copy_from_slice(&id.to_be_bytes());
        key
    }

    #[inline(always)]
    pub fn edge_label_key(label: &[u8; 4], id: &u128) -> [u8; 20] {
        let mut key = [0u8; 20];
        key[0..4].copy_from_slice(label);
        key[4..20].copy_from_slice(&id.to_be_bytes());
        key
    }

    // key = from-node(16) | label-id(4) | chunk-no(2)   ← 22 B
    // val = to-node(16)  | edge-id(16)                  ← 32 B (DUPFIXED)
    #[inline(always)]
    pub fn out_edge_key(from_node_id: &u128, label: &[u8; 4]) -> [u8; 20] {
        // 2 end bytes for chunk number
        let mut key = [0u8; 20];
        key[0..16].copy_from_slice(&from_node_id.to_be_bytes());
        key[16..20].copy_from_slice(label);
        key
    }

    #[inline(always)]
    pub fn in_edge_key(to_node_id: &u128, label: &[u8; 4]) -> [u8; 20] {
        // 2 end bytes for chunk number
        let mut key = [0u8; 20];
        key[0..16].copy_from_slice(&to_node_id.to_be_bytes());
        key[16..20].copy_from_slice(label);
        key
    }

    #[inline(always)]
    pub fn pack_edge_data(node_id: &u128, edge_id: &u128) -> [u8; 32] {
        let mut key = [0u8; 32];
        key[0..16].copy_from_slice(&edge_id.to_be_bytes());
        key[16..32].copy_from_slice(&node_id.to_be_bytes());
        key
    }

    #[inline(always)]
    pub fn unpack_adj_edge_data(data: &[u8]) -> Result<(u128, u128), GraphError> {
        let edge_id = u128::from_be_bytes(
            data[0..16]
                .try_into()
                .map_err(|_| GraphError::SliceLengthError)?,
        );
        let node_id = u128::from_be_bytes(
            data[16..32]
                .try_into()
                .map_err(|_| GraphError::SliceLengthError)?,
        );
        Ok((node_id, edge_id))
    }

    pub fn get_u128_from_bytes(bytes: &[u8]) -> Result<u128, GraphError> {
        let mut arr = [0u8; 16];
        arr.copy_from_slice(bytes);
        let res = u128::from_be_bytes(arr);
        Ok(res)
    }
}

impl DenseReadTxnProvider for NestedDenseReadTxnProvider<'_, '_> {
    fn with_dense_read_txn<T, F>(&self, f: F) -> Result<T, crate::helix_engine::types::VectorError>
    where
        F: FnOnce(&RoTxn) -> Result<T, crate::helix_engine::types::VectorError>,
    {
        let txn = self
            .storage
            .begin_nested_resize_safe_read_txn()
            .map_err(|err| {
                crate::helix_engine::types::VectorError::VectorCoreError(format!(
                    "nested resize-safe read txn failed: {}",
                    err
                ))
            })?;
        f(&txn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helix_engine::graph_core::config::Config;
    use crate::helix_engine::vector_core::hnsw::HNSW;
    use crate::helix_engine::vector_core::named_vectors::{DistanceMetric, NamedVectorConfig};
    use crate::helix_engine::vector_core::spindle::{SpindleConfig, SpindleMode};
    use crate::helix_engine::vector_core::vector::HVector;
    use serial_test::serial;
    use tempfile::TempDir;

    type Filter = fn(&HVector) -> bool;

    #[test]
    #[serial]
    fn lsm_open_records_metadata_steps_without_duplicating_aggregate() {
        let directory = TempDir::new().unwrap();
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();

        let _storage = metrics::with_local_recorder(&recorder, || {
            HelixGraphStorage::new(
                directory.path().to_str().unwrap(),
                test_config().with_lsm_in_memory(),
            )
            .unwrap()
        });

        let rendered = handle.render();
        let mut counts = rendered
            .lines()
            .filter(|line| {
                line.starts_with("helix_lsm_metadata_read_step_ms_count")
                    || line.starts_with(
                        "helix_lsm_collection_open_phase_ms_count{phase=\"metadata_read\"}",
                    )
            })
            .collect::<Vec<_>>();
        counts.sort_unstable();
        // Every open reads the adjacency backfill's mark as a step of its
        // own, and a writer's also scans the dead dense segment names: both
        // inside the aggregate like the first two.
        assert_eq!(
            counts,
            [
                "helix_lsm_collection_open_phase_ms_count{phase=\"metadata_read\"} 1",
                "helix_lsm_metadata_read_step_ms_count{step=\"adjacency_backfill_mark\"} 1",
                "helix_lsm_metadata_read_step_ms_count{step=\"dense_dead_names\"} 1",
                "helix_lsm_metadata_read_step_ms_count{step=\"hnsw_overrides\"} 1",
                "helix_lsm_metadata_read_step_ms_count{step=\"metadata_current\"} 1",
            ]
        );
    }

    #[test]
    fn metadata_refresh_logging_only_downgrades_cooperative_cancellation() {
        let cancelled = GraphError::New("LSM I/O: LSM read request cancelled".into());
        assert_eq!(
            metadata_refresh_error_log_class(&cancelled),
            MetadataRefreshErrorLogClass::CooperativeCancellation
        );

        for error in [
            GraphError::StorageError("object store unavailable".into()),
            GraphError::New("LSM I/O: checksum mismatch".into()),
        ] {
            assert_eq!(
                metadata_refresh_error_log_class(&error),
                MetadataRefreshErrorLogClass::BackendFailure,
                "real metadata refresh failure must remain warning-level: {error}"
            );
        }
    }

    struct TestEnvRestore {
        key: &'static str,
        previous: Option<String>,
    }

    impl TestEnvRestore {
        fn set(key: &'static str, value: &str) -> Self {
            let restore = Self {
                key,
                previous: std::env::var(key).ok(),
            };
            unsafe {
                std::env::set_var(key, value);
            }
            restore
        }

        fn unset(key: &'static str) -> Self {
            let restore = Self {
                key,
                previous: std::env::var(key).ok(),
            };
            unsafe {
                std::env::remove_var(key);
            }
            restore
        }
    }

    impl Drop for TestEnvRestore {
        fn drop(&mut self) {
            unsafe {
                match &self.previous {
                    Some(value) => std::env::set_var(self.key, value),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    #[test]
    fn reader_poll_promote_retry_waits_after_recent_failure() {
        assert!(HelixGraphStorage::reader_poll_promote_retry_due(0, 100, 0));
        assert!(!HelixGraphStorage::reader_poll_promote_retry_due(
            100,
            100 + READER_POLL_PROMOTE_FAILURE_COOLDOWN_MS - 1,
            1
        ));
        assert!(HelixGraphStorage::reader_poll_promote_retry_due(
            100,
            100 + READER_POLL_PROMOTE_FAILURE_COOLDOWN_MS,
            1
        ));
        assert!(!HelixGraphStorage::reader_poll_promote_retry_due(
            200, 100, 1
        ));
    }

    #[test]
    fn reader_poll_promote_cooldown_backs_off_exponentially() {
        assert_eq!(
            HelixGraphStorage::reader_poll_promote_cooldown_ms(0),
            30_000
        );
        assert_eq!(
            HelixGraphStorage::reader_poll_promote_cooldown_ms(1),
            30_000
        );
        assert_eq!(
            HelixGraphStorage::reader_poll_promote_cooldown_ms(2),
            60_000
        );
        assert_eq!(
            HelixGraphStorage::reader_poll_promote_cooldown_ms(3),
            120_000
        );
        assert_eq!(
            HelixGraphStorage::reader_poll_promote_cooldown_ms(5),
            480_000
        );
        assert_eq!(
            HelixGraphStorage::reader_poll_promote_cooldown_ms(6),
            900_000
        );
        assert_eq!(
            HelixGraphStorage::reader_poll_promote_cooldown_ms(60),
            900_000
        );
        // Backoff never allows an earlier retry than the base cooldown.
        assert!(!HelixGraphStorage::reader_poll_promote_retry_due(
            100,
            100 + READER_POLL_PROMOTE_FAILURE_COOLDOWN_MS,
            2
        ));
        assert!(HelixGraphStorage::reader_poll_promote_retry_due(
            100,
            100 + 2 * READER_POLL_PROMOTE_FAILURE_COOLDOWN_MS,
            2
        ));
    }

    #[test]
    #[serial]
    fn payload_index_workers_falls_back_to_generic_index_executor_workers() {
        let _payload = TestEnvRestore::unset("HELIX_PAYLOAD_INDEX_WORKERS");
        let _generic = TestEnvRestore::set("HELIX_INDEX_EXECUTOR_WORKERS", "8");
        assert_eq!(payload_index_workers(), 8);
    }

    #[test]
    #[serial]
    fn payload_index_submit_reports_backpressure_after_bounded_wait() {
        let _timeout = TestEnvRestore::set("HELIX_PAYLOAD_INDEX_SUBMIT_TIMEOUT_MS", "1");
        let (sender, _receiver) = flume::bounded(1);
        sender.send(dummy_payload_index_job("queued")).unwrap();
        let executor = PayloadIndexExecutor { sender };

        let error = executor
            .submit(dummy_payload_index_job("blocked"))
            .expect_err("full payload index queue must backpressure");
        assert!(
            matches!(error, GraphError::ResizeBackpressure(_)),
            "expected retryable backpressure, got {error:?}"
        );
    }

    fn dummy_payload_index_job(field_name: &str) -> PayloadIndexJob {
        PayloadIndexJob {
            collection: "test".to_string(),
            field_name: field_name.to_string(),
            schema: PayloadIndexSchema::Keyword,
            db: None,
            job_id: format!("job-{field_name}"),
            storage: Weak::new(),
        }
    }

    fn force_giant_resize_env(hard_headroom_mb: &str) -> Vec<TestEnvRestore> {
        vec![
            TestEnvRestore::set("HELIX_INITIAL_MAP_MB", "128"),
            TestEnvRestore::set("HELIX_MAX_MAP_SIZE_GB", "1"),
            TestEnvRestore::set("HELIX_PRE_GROW_MB", "0"),
            TestEnvRestore::set("HELIX_GIANT_MAP_MIN_USED_MB", "0"),
            TestEnvRestore::set("HELIX_GIANT_SEGMENT_THRESHOLD", "999999"),
            TestEnvRestore::set("HELIX_GIANT_PRE_GROW_MB", "256"),
            TestEnvRestore::set("HELIX_GIANT_MAP_GROW_MB", "256"),
            TestEnvRestore::set("HELIX_RESIZE_HARD_HEADROOM_MB", hard_headroom_mb),
        ]
    }

    /// End-to-end proof that a real graph created through the engine's write
    /// twins survives an LMDB -> LSM migration and reads back correctly through
    /// the read twins against the SlateDB-backed LSM engine — i.e. the graph
    /// data layer works on the LSM backend, not just on LMDB.
    #[test]
    fn graph_round_trips_lmdb_to_lsm_via_read_twins() {
        use super::super::backend::{Namespace, StorageBackend};
        use super::super::backend_any::AnyBackend;
        use super::super::migrate::copy_namespaces;
        use crate::protocol::value::Value;

        let dir = tempfile::TempDir::new().unwrap();
        let mut storage =
            HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();

        // Build a tiny graph via the WRITE twins on the LMDB backend:
        //   main --CALLS--> helper
        let from: u128 = 0xA1;
        let to: u128 = 0xA2;
        // Nodes in their own committed batch: create_edge_be's node-existence
        // check reads a committed snapshot (it does not read-your-writes within
        // the batch), so the endpoints must be committed before the edge — the
        // same ordering the create_edge_be unit tests use.
        {
            let mut w = storage.backend.begin_write().unwrap();
            storage
                .create_node_be(
                    &mut w,
                    "Sym",
                    [("name".to_string(), Value::String("main".into()))],
                    None,
                    Some(from),
                )
                .unwrap();
            storage
                .create_node_be(
                    &mut w,
                    "Sym",
                    [("name".to_string(), Value::String("helper".into()))],
                    None,
                    Some(to),
                )
                .unwrap();
            storage.backend.commit(w).unwrap();
        }
        let edge_id = {
            let mut w = storage.backend.begin_write().unwrap();
            let e = storage
                .create_edge_be(&mut w, "CALLS", &from, &to, Vec::<(String, Value)>::new())
                .unwrap();
            storage.backend.commit(w).unwrap();
            e.id
        };

        // Snapshot the read twins against LMDB (the reference results).
        let (exp_from, exp_edge, exp_out) = {
            let r = storage.backend.begin_read().unwrap();
            (
                storage.get_node_be(&r, &from).unwrap(),
                storage.get_edge_be(&r, &edge_id).unwrap(),
                storage.out_edges_be(&r, &from, "CALLS").unwrap(),
            )
        };

        // Migrate every graph namespace LMDB -> in-memory LSM (dups included).
        let lsm = AnyBackend::open_lsm_in_memory("/graph-lsm-demo").unwrap();
        copy_namespaces(
            &*storage.backend,
            &lsm,
            &[
                Namespace::Nodes,
                Namespace::Edges,
                Namespace::OutEdges,
                Namespace::InEdges,
                Namespace::Metadata,
            ],
        )
        .unwrap();

        // Swap the backend to LSM. Test-only: the heed Database handle fields go
        // stale, but the read twins use ONLY self.backend + Namespace, so node,
        // edge and adjacency reads now resolve against the LSM engine.
        storage.backend = Arc::new(lsm);

        // Read the SAME graph back through the twins — now against LSM.
        let r = storage.backend.begin_read().unwrap();
        let got_from = storage.get_node_be(&r, &from).unwrap();
        let got_edge = storage.get_edge_be(&r, &edge_id).unwrap();
        let got_out = storage.out_edges_be(&r, &from, "CALLS").unwrap();

        assert_eq!(got_from.id, exp_from.id);
        assert_eq!(got_from.label, exp_from.label);
        assert_eq!(
            got_from.properties.get("name"),
            Some(&Value::String("main".into())),
            "node property decodes correctly from LSM"
        );
        assert_eq!(
            got_from.properties.get("name"),
            exp_from.properties.get("name")
        );
        assert_eq!(got_edge.id, exp_edge.id);
        assert_eq!(got_edge.from_node, from);
        assert_eq!(got_edge.to_node, to);
        assert_eq!(got_out, exp_out, "out-adjacency edge ids identical on LSM");
        assert_eq!(
            got_out,
            vec![edge_id],
            "LSM out_edges resolves the migrated edge"
        );
    }

    /// `backfill_edge_path_index_batch` on the LSM backend: routes through the
    /// backend-seam twin (no local LMDB env), indexes existing edges' from/to
    /// paths, advances the cursor, and sets the complete marker — the heed path
    /// would fail on LSM because the cursor/index writes need LMDB handles.
    #[test]
    #[serial]
    fn lsm_backfill_edge_path_index_batch_indexes_and_completes() {
        use super::super::backend::StorageBackend;
        use crate::helix_engine::storage_core::upsert::EdgeUpsert;
        use crate::protocol::value::Value;

        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(
            dir.path().to_str().unwrap(),
            test_config().with_lsm_in_memory(),
        )
        .unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);

        let from: u128 = 0xB1;
        let to: u128 = 0xB2;
        let edge_id: u128 = 0xB150;
        let mut w = storage.backend.begin_write().unwrap();
        storage
            .create_node_be(&mut w, "Sym", [], None, Some(from))
            .unwrap();
        storage
            .create_node_be(&mut w, "Sym", [], None, Some(to))
            .unwrap();
        storage
            .upsert_edge_be(
                &mut w,
                &EdgeUpsert {
                    id: edge_id,
                    label: "CALLS".to_string(),
                    from_node: from,
                    to_node: to,
                    properties: HashMap::from([
                        (
                            "from_path".to_string(),
                            Value::String("src/a.rs".to_string()),
                        ),
                        ("to_path".to_string(), Value::String("src/b.rs".to_string())),
                    ]),
                },
            )
            .unwrap();
        storage.backend.commit(w).unwrap();

        // Simulate a pre-index deployment: remove the path-index rows the
        // upsert wrote, so backfill has real work to do.
        let mut w = storage.backend.begin_write().unwrap();
        let edge = {
            let r = storage.backend.begin_read().unwrap();
            storage.get_edge_be(&r, &edge_id).unwrap()
        };
        storage.delete_edge_paths_be(&mut w, &edge).unwrap();
        storage.backend.commit(w).unwrap();
        {
            let r = storage.backend.begin_read().unwrap();
            assert!(
                storage
                    .edge_ids_for_path_be(&r, "src/a.rs", 16)
                    .unwrap()
                    .is_empty(),
                "path index rows removed before backfill"
            );
        }

        let (indexed, complete, cursor) = storage.backfill_edge_path_index_batch(1000).unwrap();
        assert_eq!(indexed, 1, "one edge re-indexed");
        assert!(complete, "single batch drains the edge keyspace");
        assert_eq!(cursor, Some(edge_id));

        let r = storage.backend.begin_read().unwrap();
        assert!(
            storage
                .edge_ids_for_path_be(&r, "src/a.rs", 16)
                .unwrap()
                .contains(&edge_id),
            "from_path indexed by LSM backfill"
        );
        assert!(
            storage
                .edge_ids_for_path_be(&r, "src/b.rs", 16)
                .unwrap()
                .contains(&edge_id),
            "to_path indexed by LSM backfill"
        );
        assert!(
            storage.edge_path_index_backfill_complete_be(&r).unwrap(),
            "complete marker persisted through the seam"
        );

        // Second call is a no-op short-circuit on the complete marker.
        let (indexed, complete, cursor) = storage.backfill_edge_path_index_batch(1000).unwrap();
        assert_eq!((indexed, complete, cursor), (0, true, None));
    }

    /// Root-cause regression: on LSM the metadata blob's `stats.node_count` is
    /// frozen at 0 (never updated after write — only the merge-key counter is),
    /// so an absent counter key must reseed from a scan of `Namespace::Nodes`,
    /// not from that stale frozen blob value.
    #[test]
    #[serial]
    fn missing_lsm_counter_key_reseeds_from_scan_not_stale_blob() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(
            dir.path().to_str().unwrap(),
            test_config().with_lsm_in_memory(),
        )
        .unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);

        for _ in 0..3 {
            let mut w = storage.backend.begin_write().unwrap();
            storage
                .create_node_be(&mut w, "L", Vec::<(String, Value)>::new(), None, None)
                .unwrap();
            storage.backend.commit(w).unwrap();
        }

        // Simulate the corruption precondition: the counter key is absent.
        let mut w = storage.backend.begin_write().unwrap();
        storage
            .backend
            .delete(
                &mut w,
                Namespace::Metadata,
                lsm_counter_key(MetadataCounter::Nodes),
            )
            .unwrap();
        storage.backend.commit(w).unwrap();

        storage.seed_lsm_counter_keys().unwrap();

        let r = storage.backend.begin_read().unwrap();
        let seeded = storage
            .backend
            .get_with(
                &r,
                Namespace::Metadata,
                lsm_counter_key(MetadataCounter::Nodes),
                |v| v.map(decode_lsm_counter_value),
            )
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(
            seeded, 3,
            "must reseed from scan truth (3 real nodes), not the stale frozen blob baseline (0)"
        );
    }

    #[test]
    fn ensure_lsm_writer_rejects_non_lsm_and_reader_backends() {
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use crate::helix_engine::storage_core::backend_lsm_reader::LsmReader;
        use object_store::memory::InMemory;
        use object_store::ObjectStore;

        let dir = tempfile::TempDir::new().unwrap();
        let lmdb_storage =
            HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
        assert!(
            ensure_lsm_writer(&lmdb_storage.backend).is_err(),
            "LMDB backend has no separate counter keys to repair"
        );

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = "ensure-lsm-writer-guard-test";
        let writer_backend =
            AnyBackend::Lsm(LsmBackend::open_with_store(path, store.clone()).unwrap());
        assert!(
            ensure_lsm_writer(&writer_backend).is_ok(),
            "LSM writer must be accepted"
        );

        let reader_backend =
            AnyBackend::LsmReader(LsmReader::open_with_store(path, store).unwrap());
        let error = ensure_lsm_writer(&reader_backend).unwrap_err();
        assert!(
            error.to_string().contains("read-only"),
            "reader replica must be rejected: {error}"
        );
    }

    /// `WriteView::read_view` gives read-your-writes on the LSM backend: a node
    /// buffered into the batch (uncommitted) is visible through the read view
    /// before commit, then via a fresh committed snapshot after commit.
    #[test]
    #[serial]
    fn lsm_write_view_with_pending_writes_is_not_pinned_to_a_sequence() {
        use crate::protocol::value::Value;

        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(
            dir.path().to_str().unwrap(),
            test_config().with_lsm_in_memory(),
        )
        .unwrap();
        let committed = storage.backend.begin_read().unwrap();
        assert!(committed.snapshot_seq().is_some());
        let mut wv = storage.begin_write_view().unwrap();
        storage
            .create_node_be(
                wv.write_mut(),
                "Sym",
                [("k".to_string(), Value::String("v".into()))],
                None,
                Some(0xAB),
            )
            .unwrap();
        assert_eq!(
            wv.read_view().snapshot_seq(),
            None,
            "a view overlaying uncommitted writes must never key a cache"
        );
    }

    #[test]
    #[serial]
    fn lsm_write_view_read_your_writes() {
        use super::super::backend::StorageBackend;
        use crate::protocol::value::Value;

        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(
            dir.path().to_str().unwrap(),
            test_config().with_lsm_in_memory(),
        )
        .unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);

        let id: u128 = 0xAA;
        let mut wv = storage.begin_write_view().unwrap();
        storage
            .create_node_be(
                wv.write_mut(),
                "Sym",
                [("k".to_string(), Value::String("v".into()))],
                None,
                Some(id),
            )
            .unwrap();

        // BEFORE commit: the uncommitted write is visible through the read view.
        {
            let r = wv.read_view();
            assert!(
                storage.check_exists_be(&r, &id).unwrap(),
                "uncommitted node must be visible through the write-view read view (LSM)"
            );
            let n = storage.get_node_be(&r, &id).unwrap();
            assert_eq!(n.id, id);
        }

        // A fresh committed snapshot must NOT yet see it (still uncommitted).
        {
            let r = storage.backend.begin_read().unwrap();
            assert!(
                !storage.check_exists_be(&r, &id).unwrap(),
                "uncommitted node must not be visible through a fresh committed snapshot"
            );
        }

        wv.commit().unwrap();

        // AFTER commit: visible via a fresh committed snapshot too.
        let r = storage.backend.begin_read().unwrap();
        assert!(storage.check_exists_be(&r, &id).unwrap());
        assert_eq!(storage.get_node_be(&r, &id).unwrap().id, id);
    }

    /// Reader-replica reconcile coverage on the LSM backend: `get_metadata_be`
    /// (the backend-seam metadata read introduced for the reconcile fix) must
    /// read committed metadata through `with_read_backend` WITHOUT touching the
    /// local LMDB env, and `refresh_metadata_snapshot` (now routed through
    /// `get_metadata_be`) must succeed on LSM rather than failing with the
    /// "LMDB env handle is unavailable on this backend" error. The dense-view
    /// reconcile leg is exercised by the `_lsm` cold-open paths.
    #[test]
    #[serial]
    fn lsm_get_metadata_be_matches_snapshot_without_lmdb_env() {
        use super::super::metadata::PayloadIndexSchema;

        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(
            dir.path().to_str().unwrap(),
            test_config().with_lsm_in_memory(),
        )
        .unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);

        // No local heed env on LSM: the old reconcile path failed here.
        assert!(
            storage.lmdb_env().is_err(),
            "LSM collection must not expose a local LMDB env"
        );

        // Mutate committed metadata through the LSM write seam.
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

        // `get_metadata_be` reads the committed value through the backend seam.
        let via_be = storage
            .with_read_backend(|r| storage.get_metadata_be(r))
            .expect("get_metadata_be must succeed on LSM without a local LMDB env");
        assert_eq!(
            via_be.payload_indices.get("repo"),
            Some(&PayloadIndexSchema::Keyword)
        );
        let sidecar_path = HelixGraphStorage::metadata_sidecar_path(dir.path());
        std::fs::remove_file(&sidecar_path).expect("test should remove sidecar");
        assert!(
            HelixGraphStorage::read_metadata_sidecar_from_path(dir.path())
                .unwrap()
                .is_none(),
            "sidecar should be absent before refresh"
        );

        // `refresh_metadata_snapshot` (the fixed reconcile entry point) must
        // succeed on LSM and publish the same metadata `get_metadata_be` reads.
        storage
            .refresh_metadata_snapshot()
            .expect("refresh_metadata_snapshot must succeed on LSM reader/writer");
        let snapshot = storage.metadata_snapshot().unwrap();
        assert_eq!(snapshot.payload_indices, via_be.payload_indices);
        assert!(
            HelixGraphStorage::read_metadata_sidecar_from_path(dir.path())
                .unwrap()
                .is_some(),
            "refresh must recreate a missing metadata sidecar"
        );
    }

    /// Two refreshes overlap when two commits end together. The one that read
    /// earlier and stores last must leave the snapshot and the local sidecar
    /// at what the later read saw.
    #[test]
    #[serial]
    fn lsm_metadata_snapshot_keeps_the_later_read_when_an_earlier_refresh_stores_last() {
        use super::super::metadata::PayloadIndexSchema;

        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(
            dir.path().to_str().unwrap(),
            test_config().with_lsm_in_memory(),
        )
        .unwrap();

        // The first refresh has read and has not stored yet.
        let (older, older_read_at) = storage
            .with_read_backend(|r| Ok((storage.get_metadata_be(r)?, r.snapshot_seq())))
            .unwrap();
        assert!(older_read_at.is_some(), "a writer's read is pinned");

        // A commit ends meanwhile: its refresh reads later and stores first.
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
        let newer = storage.metadata_snapshot().unwrap();
        assert_ne!(newer, older);

        storage
            .store_metadata_snapshot(&older, older_read_at)
            .unwrap();

        assert_eq!(storage.metadata_snapshot().unwrap(), newer);
        let sidecar = HelixGraphStorage::read_metadata_sidecar_from_path(dir.path())
            .unwrap()
            .expect("the commit's refresh wrote the sidecar");
        assert_eq!(sidecar.metadata, newer);

        // A read at the stored sequence is the same data, and is not refused.
        storage.refresh_metadata_snapshot().unwrap();
        assert_eq!(storage.metadata_snapshot().unwrap(), newer);
    }

    /// The object store gets a sidecar at most once per put interval. The
    /// change that falls inside the interval must still arrive: held first,
    /// sent by the sweeper's call once the interval allows.
    #[test]
    #[serial]
    fn lsm_metadata_sidecar_held_back_by_the_put_interval_is_sent_later() {
        use super::super::metadata::PayloadIndexSchema;

        let _interval =
            TestEnvRestore::set("HELIX_LSM_METADATA_SIDECAR_PUT_INTERVAL_MS", "3600000");
        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(
            dir.path().to_str().unwrap(),
            test_config().with_lsm_in_memory(),
        )
        .unwrap();
        let index = |field: &str| {
            storage
                .with_write_backend(|w| {
                    storage
                        .update_metadata_be(w, |metadata| {
                            let mut indices = metadata.payload_indices.clone();
                            indices.insert(field.to_string(), PayloadIndexSchema::Keyword);
                            metadata.set_payload_indices(indices);
                            Ok(())
                        })
                        .map(|_| ())
                })
                .unwrap();
        };
        let sent = || storage.last_lsm_sidecar_fingerprint.load(Ordering::Acquire);
        let held = || storage.lsm_sidecar_put_held.load(Ordering::Acquire);

        // The first change is put at once.
        index("repo");
        let first = sent();
        assert!(first != 0 && !held());

        // The second falls inside the interval: the object store keeps the first.
        index("path");
        assert!(held());
        assert_eq!(sent(), first);
        storage.upload_held_metadata_sidecar();
        assert!(held(), "still inside the interval");
        assert_eq!(sent(), first);

        // Once the interval allows, the sweeper's call sends the newest.
        let _interval = TestEnvRestore::set("HELIX_LSM_METADATA_SIDECAR_PUT_INTERVAL_MS", "0");
        storage.upload_held_metadata_sidecar();
        assert!(!held());
        assert_ne!(sent(), first);
        let newest = sent();
        storage.upload_held_metadata_sidecar();
        assert_eq!(sent(), newest);
    }

    /// A sidecar is put with the adjacency backfill's mark as it is when the
    /// put is made. One that was built while the mark said "complete" is not
    /// put once a backfill call has said "not complete": it would land after
    /// that call's own put and say "complete" of a pass that has started
    /// over. It is held, and the sweeper's call sends the sidecar as it is
    /// now. A change of the mark alone is a change of the fingerprint.
    #[test]
    #[serial]
    fn lsm_metadata_sidecar_built_before_the_mark_changed_is_not_put() {
        let _interval = TestEnvRestore::set("HELIX_LSM_METADATA_SIDECAR_PUT_INTERVAL_MS", "0");
        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(
            dir.path().to_str().unwrap(),
            test_config().with_lsm_in_memory(),
        )
        .unwrap();
        assert_eq!(
            storage.adjacency_backfill_complete(),
            Some(false),
            "an LSM open reads the mark"
        );
        let sent = || storage.last_lsm_sidecar_fingerprint.load(Ordering::Acquire);
        let sent_mark = || storage.adjacency_mark_sent.load(Ordering::Acquire);
        let held = || storage.lsm_sidecar_put_held.load(Ordering::Acquire);
        let local_mark = || {
            HelixGraphStorage::read_metadata_sidecar_from_path(storage.path())
                .unwrap()
                .unwrap()
                .adjacency_backfill_complete
        };
        let metadata = storage.metadata_snapshot().unwrap();

        storage.send_adjacency_backfill_mark(true).unwrap();
        let complete = sent();
        assert_eq!((sent_mark(), local_mark()), (2, Some(true)));
        let built_complete = storage.build_metadata_sidecar(&metadata).unwrap();

        storage.send_adjacency_backfill_mark(false).unwrap();
        let not_complete = sent();
        assert_ne!(
            not_complete, complete,
            "the mark alone changes the fingerprint"
        );
        assert_eq!((sent_mark(), local_mark()), (1, Some(false)));

        storage.upload_metadata_sidecar(built_complete);
        assert_eq!(
            (sent(), sent_mark(), held()),
            (not_complete, 1, true),
            "a sidecar that says complete was put after the mark was cleared"
        );
        storage.upload_held_metadata_sidecar();
        assert_eq!((sent(), sent_mark(), held()), (not_complete, 1, false));
        assert_eq!(local_mark(), Some(false));

        // With the mark as it was built and no other put on its way, a
        // sidecar is put as it always was.
        let forget_the_last_put = || {
            storage
                .last_lsm_sidecar_fingerprint
                .store(7, Ordering::Release)
        };
        forget_the_last_put();
        storage.upload_metadata_sidecar(storage.build_metadata_sidecar(&metadata).unwrap());
        assert_eq!((sent(), sent_mark(), held()), (not_complete, 1, false));

        // Another put of this collection is on its way: this one does not
        // wait for it, and is held for the sweeper.
        forget_the_last_put();
        let second = storage.build_metadata_sidecar(&metadata).unwrap();
        assert!(
            returns_beside_a_put(&storage, || storage.upload_metadata_sidecar(second)),
            "a regular put waited for the one on its way"
        );
        assert_eq!((sent(), held()), (7, true), "it was put all the same");
        storage.upload_held_metadata_sidecar();
        assert_eq!((sent(), sent_mark(), held()), (not_complete, 1, false));

        // A second call with the same mark puts nothing, and does not wait
        // for a put on its way to find that out.
        let puts_nothing = |complete: bool| {
            storage
                .last_lsm_sidecar_fingerprint
                .store(7, Ordering::Release);
            storage.send_adjacency_backfill_mark(complete).unwrap();
            sent() == 7
        };
        assert!(puts_nothing(false));
        assert!(!puts_nothing(true));
        assert!(puts_nothing(true));
        assert!(
            returns_beside_a_put(&storage, || {
                storage.send_adjacency_backfill_mark(true).unwrap()
            }),
            "a call with nothing to say waited for the put lock"
        );
    }

    /// Whether `call` returns while another holds the collection's sidecar
    /// put lock, as a put on its way does. Five seconds, then the lock is
    /// given up so that a call that waits ends and the test fails.
    fn returns_beside_a_put(storage: &HelixGraphStorage, call: impl FnOnce() + Send) -> bool {
        let on_its_way = storage.sidecar_put.lock();
        std::thread::scope(|scope| {
            let call = scope.spawn(call);
            let started = std::time::Instant::now();
            while !call.is_finished() && started.elapsed() < std::time::Duration::from_secs(5) {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            let returned = call.is_finished();
            drop(on_its_way);
            returned
        })
    }

    /// An open does not need the adjacency backfill's mark: a read of it that
    /// fails is counted and reads as no mark, so the pass is due again.
    #[test]
    fn an_adjacency_backfill_mark_that_an_open_cannot_read_is_no_mark() {
        assert!(HelixGraphStorage::adjacency_mark_read_at_open(Ok(true)));
        assert!(!HelixGraphStorage::adjacency_mark_read_at_open(Ok(false)));

        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let unread = BackendError::Io("a read that fails".to_string());
        let read = metrics::with_local_recorder(&recorder, || {
            HelixGraphStorage::adjacency_mark_read_at_open(Err(unread))
        });
        assert!(!read, "a mark that was not read reads as complete");
        let counted = handle.render();
        assert!(
            counted
                .lines()
                .any(|line| line == "helix_adjacency_backfill_mark_unread_total 1"),
            "{counted}"
        );
    }

    #[test]
    #[serial]
    fn lsm_metadata_transactions_survive_stale_write_view_without_gate() {
        use super::super::metadata::PayloadIndexSchema;
        use std::sync::mpsc;
        use std::time::Duration;

        let dir = tempfile::TempDir::new().unwrap();
        let storage = Arc::new(
            HelixGraphStorage::new(
                dir.path().to_str().unwrap(),
                test_config().with_lsm_in_memory(),
            )
            .unwrap(),
        );
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);

        let (stale_ready_tx, stale_ready_rx) = mpsc::channel();
        let (allow_stale_commit_tx, allow_stale_commit_rx) = mpsc::channel();
        let stale_storage = Arc::clone(&storage);
        let stale_writer = std::thread::spawn(move || {
            let mut view = stale_storage.begin_write_view().unwrap();
            stale_storage
                .update_metadata_be(view.write_mut(), |metadata| {
                    metadata.set_secondary_indices(vec!["stale-write".to_string()]);
                    Ok(())
                })
                .unwrap();
            stale_ready_tx.send(()).unwrap();
            allow_stale_commit_rx.recv().unwrap();
            view.commit().unwrap();
        });

        stale_ready_rx.recv_timeout(Duration::from_secs(2)).unwrap();

        let (index_done_tx, index_done_rx) = mpsc::channel();
        let index_storage = Arc::clone(&storage);
        let indexer = std::thread::spawn(move || {
            let result = index_storage.create_payload_index("repo", PayloadIndexSchema::Keyword);
            index_done_tx.send(()).unwrap();
            result
        });

        let index_finished_before_stale_commit =
            index_done_rx.recv_timeout(Duration::from_secs(2)).is_ok();
        allow_stale_commit_tx.send(()).unwrap();
        stale_writer.join().unwrap();
        indexer.join().unwrap().unwrap();

        assert!(
            index_finished_before_stale_commit,
            "payload-index metadata publish must not wait on the removed LSM write gate"
        );
        let metadata = storage
            .with_read_backend(|r| storage.get_metadata_be(r))
            .expect("metadata must be readable after serialized writes");
        assert_eq!(
            metadata.payload_indices.get("repo"),
            Some(&PayloadIndexSchema::Keyword)
        );
        assert_eq!(metadata.secondary_indices, vec!["stale-write".to_string()]);
    }

    #[test]
    #[serial]
    fn lsm_counter_merges_bypass_write_gate_and_survive_metadata_publish() {
        use super::super::metadata::PayloadIndexSchema;
        use std::sync::mpsc;
        use std::time::Duration;

        let dir = tempfile::TempDir::new().unwrap();
        let storage = Arc::new(
            HelixGraphStorage::new(
                dir.path().to_str().unwrap(),
                test_config().with_lsm_in_memory(),
            )
            .unwrap(),
        );
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);

        let gate = storage.write_txn_gate.lock().unwrap();
        let (done_tx, done_rx) = mpsc::channel();
        let writer_storage = Arc::clone(&storage);
        let writer = std::thread::spawn(move || {
            let result = writer_storage.with_write_backend(|w| {
                writer_storage.adjust_metadata_counter_be(w, MetadataCounter::Nodes, 1)?;
                writer_storage.adjust_metadata_counter_be(w, MetadataCounter::Nodes, 1)?;
                Ok(())
            });
            done_tx.send(result.is_ok()).unwrap();
            result
        });

        let finished_while_gate_held = done_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap_or(false);
        drop(gate);
        writer.join().unwrap().unwrap();
        assert!(
            finished_while_gate_held,
            "LSM counter writes must not wait on the LMDB write_txn_gate"
        );

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

        let metadata = storage
            .with_read_backend(|r| storage.get_metadata_be(r))
            .expect("metadata must be readable after counter merge and metadata publish");
        assert_eq!(metadata.stats.node_count, 2);
        assert_eq!(
            metadata.payload_indices.get("repo"),
            Some(&PayloadIndexSchema::Keyword)
        );
    }

    /// Companion to `missing_lsm_counter_key_reseeds_from_scan_not_stale_blob`,
    /// exercising the OTHER seeding call site: `seed_lsm_counter_key_for_write`,
    /// invoked mid-write from `adjust_metadata_counter_be` when the counter key
    /// is absent. Must seed from a scan of already-committed nodes, then apply
    /// this write's own +1 delta — not reseed from the frozen blob (0) + 1.
    #[test]
    #[serial]
    fn write_time_counter_reseed_uses_scan_truth_not_stale_blob() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(
            dir.path().to_str().unwrap(),
            test_config().with_lsm_in_memory(),
        )
        .unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);

        for _ in 0..3 {
            let mut w = storage.backend.begin_write().unwrap();
            storage
                .create_node_be(&mut w, "L", Vec::<(String, Value)>::new(), None, None)
                .unwrap();
            storage.backend.commit(w).unwrap();
        }

        let mut w = storage.backend.begin_write().unwrap();
        storage
            .backend
            .delete(
                &mut w,
                Namespace::Metadata,
                lsm_counter_key(MetadataCounter::Nodes),
            )
            .unwrap();
        storage.backend.commit(w).unwrap();

        let mut w = storage.backend.begin_write().unwrap();
        storage
            .create_node_be(&mut w, "L", Vec::<(String, Value)>::new(), None, None)
            .unwrap();
        storage.backend.commit(w).unwrap();

        let r = storage.backend.begin_read().unwrap();
        let counter = storage
            .backend
            .get_with(
                &r,
                Namespace::Metadata,
                lsm_counter_key(MetadataCounter::Nodes),
                |v| v.map(decode_lsm_counter_value),
            )
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(
            counter, 4,
            "must seed from scan truth (3 pre-existing nodes) plus this write's delta, not stale blob (0) plus delta"
        );
    }

    /// Building a payload index on the LSM backend must NOT panic and must produce
    /// a queryable index. Before the fix the payload-index build loop called the
    /// LMDB-only `get_for_update_heed`, hitting `unreachable!("heed write path is
    /// LMDB-only")` under `HELIX_STORAGE_BACKEND=lsm` (breaking the client's `repo` filter
    /// on the cloud backend). This drives the real `create_payload_index` build path
    /// and then the (already-routed) query path end-to-end on an in-memory LSM.
    #[test]
    #[serial]
    fn lsm_create_payload_index_builds_and_queries() {
        use super::super::metadata::PayloadIndexSchema;
        use crate::protocol::value::Value;

        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(
            dir.path().to_str().unwrap(),
            test_config().with_lsm_in_memory(),
        )
        .unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);

        // Seed nodes carrying a "repo" payload field (the client's filter field).
        let (n1, n2, n3): (u128, u128, u128) = (0x101, 0x102, 0x103);
        {
            let mut wv = storage.begin_write_view().unwrap();
            for (id, repo) in [(n1, "alpha"), (n2, "alpha"), (n3, "beta")] {
                storage
                    .create_node_be(
                        wv.write_mut(),
                        "Sym",
                        [("repo".to_string(), Value::String(repo.into()))],
                        None,
                        Some(id),
                    )
                    .unwrap();
            }
            wv.commit().unwrap();
        }

        // Build the index — this is the path that previously panicked on LSM.
        storage
            .create_payload_index("repo", PayloadIndexSchema::Keyword)
            .unwrap();

        // Query via the already-routed read path; both 'alpha' nodes must resolve.
        let r = storage.backend.begin_read().unwrap();
        let mut alpha = storage
            .get_nodes_by_payload_value_be(&r, "repo", &Value::String("alpha".into()))
            .unwrap();
        alpha.sort_unstable();
        assert_eq!(
            alpha,
            vec![n1, n2],
            "payload index built on LSM must resolve both 'alpha' nodes"
        );
        let beta = storage
            .get_nodes_by_payload_value_be(&r, "repo", &Value::String("beta".into()))
            .unwrap();
        assert_eq!(beta, vec![n3], "and the single 'beta' node");
    }

    /// The count-only probe readers must abort the index walk once the count
    /// exceeds `cap` (returning `cap + 1`), instead of counting every entry.
    #[test]
    #[serial]
    fn count_probe_aborts_at_cap() {
        use super::super::backend_any::AnyBackend;
        use super::super::metadata::PayloadIndexSchema;
        use crate::protocol::value::Value;

        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(
            dir.path().to_str().unwrap(),
            test_config().with_lsm_in_memory(),
        )
        .unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);

        {
            let mut wv = storage.begin_write_view().unwrap();
            for id in 1..=50_u128 {
                storage
                    .create_node_be(
                        wv.write_mut(),
                        "Sym",
                        [
                            ("repo".to_string(), Value::String("alpha".into())),
                            ("rank".to_string(), Value::I64(id as i64)),
                        ],
                        None,
                        Some(id),
                    )
                    .unwrap();
            }
            wv.commit().unwrap();
        }

        storage
            .create_payload_index("repo", PayloadIndexSchema::Keyword)
            .unwrap();
        storage
            .create_payload_index("rank", PayloadIndexSchema::Integer)
            .unwrap();

        let r = storage.backend.begin_read().unwrap();
        let value = Value::String("alpha".into());
        assert_eq!(
            storage
                .count_nodes_by_payload_value_be(&r, "repo", &value, 10)
                .unwrap(),
            11,
            "keyword count must abort at cap + 1, not walk all 50 entries"
        );
        assert_eq!(
            storage
                .count_nodes_by_payload_value_be(&r, "repo", &value, 100)
                .unwrap(),
            50,
            "an uncapped keyword count must see every entry"
        );
        assert_eq!(
            storage
                .count_nodes_by_payload_range_be(&r, "rank", Some((1.0, true)), None, 10)
                .unwrap(),
            11,
            "range count must abort at cap + 1"
        );
        assert_eq!(
            storage
                .count_nodes_by_payload_range_be(&r, "rank", Some((1.0, true)), None, 100)
                .unwrap(),
            50,
            "an uncapped range count must see every entry"
        );

        let AnyBackend::Lsm(lsm) = storage.backend.as_ref() else {
            panic!("fixture must use LSM");
        };
        lsm.reset_read_count();
        assert_eq!(
            storage
                .count_nodes_by_payload_range_be(
                    &r,
                    "rank",
                    Some((50.0, true)),
                    Some((50.0, true)),
                    10
                )
                .unwrap(),
            1
        );
        assert_eq!(
            lsm.read_count(),
            1,
            "exact count must not scan unrelated ranks"
        );
        lsm.reset_read_count();
        assert_eq!(
            storage
                .get_nodes_by_payload_range_be(&r, "rank", Some((50.0, true)), Some((50.0, true)))
                .unwrap(),
            vec![50]
        );
        assert_eq!(
            lsm.read_count(),
            1,
            "exact lookup must not scan unrelated ranks"
        );
        lsm.reset_read_count();
        assert_eq!(
            storage
                .count_nodes_by_payload_range_be(
                    &r,
                    "rank",
                    Some((51.0, true)),
                    Some((51.0, true)),
                    10
                )
                .unwrap(),
            0
        );
        assert_eq!(
            lsm.read_count(),
            0,
            "absent exact keys must not scan the index"
        );
    }

    #[test]
    #[serial]
    fn lsm_empty_enqueue_payload_index_publishes_ready_without_background_job() {
        use super::super::backend_any::AnyBackend;
        use super::super::metadata::PayloadIndexSchema;

        let dir = tempfile::TempDir::new().unwrap();
        let mut storage =
            HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
        storage.backend = Arc::new(
            AnyBackend::open_lsm_in_memory(dir.path().join("lsm").to_str().unwrap()).unwrap(),
        );
        let storage = Arc::new(storage);
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);

        let status = storage
            .enqueue_payload_index_build(
                "repo",
                "metadata.symbol",
                PayloadIndexSchema::Keyword,
                true,
            )
            .unwrap();
        assert_eq!(status.status, "ready");

        let metadata = storage
            .with_read_backend(|r| storage.get_metadata_be(r))
            .unwrap();
        assert_eq!(
            metadata.payload_indices.get("metadata.symbol"),
            Some(&PayloadIndexSchema::Keyword)
        );
        assert!(
            !storage.collection_degraded.load(Ordering::Acquire),
            "empty LSM index registration must not mark the collection degraded"
        );
    }

    /// Regression: building a payload index over an EXISTING, SST-resident,
    /// file-backed LSM collection through the real async executor path must finish
    /// `ready` AND report the backfilled node count (`indexed_nodes`), not 0.
    ///
    /// Root cause this guards: `PayloadIndexHandle::mark_ready` used to drop the
    /// `Building` state's `indexed_nodes` when promoting to `Ready`, and
    /// `status()` hard-coded `indexed_nodes: 0` for `Ready`. A fully-built index
    /// therefore reported `indexed_nodes=0`, indistinguishable from a hollow index
    /// (the production symptom: "status=ready but indexed_nodes=0" on a 165k-point
    /// collection). The backfill itself is correct — this asserts the count is
    /// surfaced AND the index is genuinely queryable (proving it is NOT hollow).
    ///
    /// Faithful to prod: file-backed LSM, rows made durable then read back through
    /// a freshly reopened handle (SST-resident, memtable empty), seeded via the
    /// real upsert seam with a nested `metadata.repo` payload (the client's filter field),
    /// built via `enqueue_payload_index_build` on the background executor, across
    /// multiple chunks (small chunk size exercises the cumulative count).
    #[test]
    #[serial]
    fn lsm_payload_index_build_reports_indexed_count_over_ssts() {
        use super::super::backend_any::AnyBackend;
        use super::super::backend_lsm::LsmBackend;
        use super::super::metadata::PayloadIndexSchema;
        use crate::helix_engine::storage_core::upsert::NodeUpsert;
        use crate::protocol::value::Value;
        use slatedb::object_store::local::LocalFileSystem;
        use slatedb::object_store::ObjectStore;
        use std::collections::HashMap;

        // Small chunk size -> multiple backfill chunks (cursor advancement +
        // cumulative indexed_nodes), mirroring a large prod backfill.
        let _chunk = TestEnvRestore::set("HELIX_PAYLOAD_INDEX_CHUNK_SIZE", "200");

        let dir = tempfile::TempDir::new().unwrap();
        let mut storage = HelixGraphStorage::new(
            dir.path().to_str().unwrap(),
            test_config().with_lsm_in_memory(),
        )
        .unwrap();

        // Swap in a FILE-BACKED LSM (LocalFileSystem object store), not in-memory.
        let data_dir = tempfile::TempDir::new().unwrap();
        let store: Arc<dyn ObjectStore> =
            Arc::new(LocalFileSystem::new_with_prefix(data_dir.path()).unwrap());
        let lsm = LsmBackend::open_with_store("helion/repro", store.clone()).unwrap();
        storage.backend = Arc::new(AnyBackend::Lsm(lsm));
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);

        // Seed N points through the REAL upsert seam (buffered group-commit + a
        // durability flush per batch — exactly the upsert path), with a NESTED
        // `metadata.repo` payload like live client/qdrant points.
        let n: u128 = 800;
        let batch = 10u128;
        let mut i = 0u128;
        while i < n {
            storage
                .with_write_backend_buffered(|w| {
                    for j in 0..batch {
                        let id = i + j;
                        let repo = if id % 2 == 0 { "alpha" } else { "beta" };
                        let mut md = HashMap::new();
                        md.insert("repo".to_string(), Value::String(repo.into()));
                        let mut props = HashMap::new();
                        props.insert("metadata".to_string(), Value::Object(md));
                        storage.upsert_node_be(
                            w,
                            &NodeUpsert {
                                id: 0x1000 + id,
                                label: "point".into(),
                                properties: props,
                            },
                        )?;
                    }
                    Ok(())
                })
                .unwrap();
            storage.backend.flush_durable().unwrap();
            i += batch;
        }

        // CLEAN CLOSE: flush the memtable to L0 SSTs and truncate the WAL, then
        // REOPEN a fresh handle — the build reads the rows from durable SSTs with an
        // empty memtable (existing-populated-collection / process-restart scenario).
        if let AnyBackend::Lsm(b) = storage.backend.as_ref() {
            b.close().unwrap();
        }
        storage.backend = Arc::new(AnyBackend::open_lsm_in_memory("/discard").unwrap());
        let lsm2 = LsmBackend::open_with_store("helion/repro", store.clone()).unwrap();
        storage.backend = Arc::new(AnyBackend::Lsm(lsm2));
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);

        // Build the payload index over the SST-resident collection via the real
        // background executor path (Raft apply -> enqueue_payload_index_build).
        let storage = Arc::new(storage);
        storage
            .enqueue_payload_index_build(
                "repro",
                "metadata.repo",
                PayloadIndexSchema::Keyword,
                true,
            )
            .unwrap();

        // Wait for the background job to leave "building".
        let mut status = None;
        for _ in 0..600 {
            let s = storage
                .payload_index_status("metadata.repo")
                .unwrap()
                .unwrap();
            if s.status != "building" {
                status = Some(s);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let status = status.expect("payload index build did not finish");
        assert_eq!(
            status.status, "ready",
            "build must reach ready: {:?}",
            status
        );

        // The fix: a completed index reports its backfilled node count, not 0.
        assert_eq!(
            status.indexed_nodes, n as u64,
            "ready payload index must report the backfilled count, not 0 \
             (a 0 here is the hollow-index regression)"
        );

        // And the index is genuinely populated (not hollow): both reads resolve.
        let r = storage.backend.begin_read().unwrap();
        let alpha = storage
            .get_nodes_by_payload_value_be(&r, "metadata.repo", &Value::String("alpha".into()))
            .unwrap();
        assert_eq!(
            alpha.len() as u128,
            n / 2,
            "payload-index query must resolve all 'alpha' points"
        );
    }

    #[test]
    #[serial]
    fn lsm_write_view_edge_scans_read_your_writes() {
        use super::super::backend::StorageBackend;
        use crate::protocol::value::Value;

        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(
            dir.path().to_str().unwrap(),
            test_config().with_lsm_in_memory(),
        )
        .unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);

        let from: u128 = 0xCA;
        let to: u128 = 0xCB;
        {
            let mut seed = storage.begin_write_view().unwrap();
            storage
                .create_node_be(
                    seed.write_mut(),
                    "Sym",
                    [("name".to_string(), Value::String("main".into()))],
                    None,
                    Some(from),
                )
                .unwrap();
            storage
                .create_node_be(
                    seed.write_mut(),
                    "Sym",
                    [("name".to_string(), Value::String("helper".into()))],
                    None,
                    Some(to),
                )
                .unwrap();
            seed.commit().unwrap();
        }

        let mut wv = storage.begin_write_view().unwrap();
        let edge = storage
            .create_edge_be(
                wv.write_mut(),
                "CALLS",
                &from,
                &to,
                Vec::<(String, Value)>::new(),
            )
            .unwrap();

        {
            let r = wv.read_view();
            let edges = storage
                .scan_all_edges_be(&r)
                .unwrap()
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(edges.len(), 1);
            assert_eq!(edges[0].id, edge.id);
            assert_eq!(
                storage.out_edges_be(&r, &from, "CALLS").unwrap(),
                vec![edge.id]
            );
        }

        {
            let r = storage.backend.begin_read().unwrap();
            let edges = storage.scan_all_edges_be(&r).unwrap();
            assert!(
                edges.is_empty(),
                "fresh committed snapshot must not see the uncommitted edge"
            );
        }

        wv.commit().unwrap();

        let r = storage.backend.begin_read().unwrap();
        let edges = storage
            .scan_all_edges_be(&r)
            .unwrap()
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].id, edge.id);
        assert_eq!(
            storage.out_edges_be(&r, &from, "CALLS").unwrap(),
            vec![edge.id]
        );
    }

    #[test]
    #[serial]
    fn lsm_native_graph_scales_on_slatedb_mvcc() {
        use crate::protocol::value::Value;
        use std::collections::HashMap;

        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(
            dir.path().to_str().unwrap(),
            test_config().with_lsm_in_memory(),
        )
        .unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);

        const NODE_COUNT: u128 = 2_048;
        const EDGE_COUNT: u128 = 8_192;
        const BATCH: u128 = 256;

        let mut next = 0u128;
        while next < NODE_COUNT {
            let end = (next + BATCH).min(NODE_COUNT);
            let mut wv = storage.begin_write_view().unwrap();
            for id in next..end {
                storage
                    .create_node_be(
                        wv.write_mut(),
                        "Symbol",
                        [("ordinal".to_string(), Value::U64(id as u64))],
                        None,
                        Some(0x1000_0000 + id),
                    )
                    .unwrap();
            }
            wv.commit().unwrap();
            next = end;
        }

        let mut expected_out: HashMap<u128, Vec<u128>> = HashMap::new();
        let mut expected_in: HashMap<u128, Vec<u128>> = HashMap::new();
        let mut edge_idx = 0u128;
        while edge_idx < EDGE_COUNT {
            let end = (edge_idx + BATCH).min(EDGE_COUNT);
            let mut wv = storage.begin_write_view().unwrap();
            for i in edge_idx..end {
                let from = 0x1000_0000 + (i % NODE_COUNT);
                let to = 0x1000_0000 + ((i.wrapping_mul(17) + 23) % NODE_COUNT);
                let edge = storage
                    .create_edge_be(
                        wv.write_mut(),
                        "CALLS",
                        &from,
                        &to,
                        [("ordinal".to_string(), Value::U64(i as u64))],
                    )
                    .unwrap();
                expected_out.entry(from).or_default().push(edge.id);
                expected_in.entry(to).or_default().push(edge.id);

                if i == edge_idx {
                    let r = wv.read_view();
                    assert!(
                        storage
                            .out_edges_be(&r, &from, "CALLS")
                            .unwrap()
                            .contains(&edge.id),
                        "SlateDB write-view read must see pending out adjacency"
                    );
                    assert!(
                        storage
                            .in_edges_be(&r, &to, "CALLS")
                            .unwrap()
                            .contains(&edge.id),
                        "SlateDB write-view read must see pending in adjacency"
                    );
                }
            }
            {
                let r = storage.backend.begin_read().unwrap();
                let committed_edges = storage.scan_all_edges_be(&r).unwrap().len() as u128;
                assert_eq!(
                    committed_edges, edge_idx,
                    "fresh SlateDB snapshot must not see an uncommitted graph batch"
                );
            }
            wv.commit().unwrap();
            edge_idx = end;
        }

        let r = storage.backend.begin_read().unwrap();
        let nodes = storage
            .scan_all_nodes_be(&r)
            .unwrap()
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let edges = storage
            .scan_all_edges_be(&r)
            .unwrap()
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(nodes.len(), NODE_COUNT as usize);
        assert_eq!(edges.len(), EDGE_COUNT as usize);

        for node in [0x1000_0000, 0x1000_0001, 0x1000_0042, 0x1000_0400] {
            let mut got = storage.out_edges_be(&r, &node, "CALLS").unwrap();
            let mut expected = expected_out.remove(&node).unwrap_or_default();
            got.sort_unstable();
            expected.sort_unstable();
            assert_eq!(got, expected, "out adjacency mismatch for {node:#x}");
        }
        for node in [0x1000_0017, 0x1000_0028, 0x1000_0300, 0x1000_0700] {
            let mut got = storage.in_edges_be(&r, &node, "CALLS").unwrap();
            let mut expected = expected_in.remove(&node).unwrap_or_default();
            got.sort_unstable();
            expected.sort_unstable();
            assert_eq!(got, expected, "in adjacency mismatch for {node:#x}");
        }
    }

    /// LMDB parity for `WriteView::read_view`: the `RwTxn` reads its own buffered
    /// writes, so the same read-your-writes guarantee holds on the default
    /// backend (no env override).
    #[test]
    fn lmdb_write_view_read_your_writes() {
        use super::super::backend::StorageBackend;
        use crate::protocol::value::Value;

        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lmdb);

        let id: u128 = 0xBB;
        let mut wv = storage.begin_write_view().unwrap();
        storage
            .create_node_be(
                wv.write_mut(),
                "Sym",
                [("k".to_string(), Value::String("v".into()))],
                None,
                Some(id),
            )
            .unwrap();

        {
            let r = wv.read_view();
            assert!(
                storage.check_exists_be(&r, &id).unwrap(),
                "uncommitted node must be visible through the write-view read view (LMDB)"
            );
            assert_eq!(storage.get_node_be(&r, &id).unwrap().id, id);
        }

        wv.commit().unwrap();

        let r = storage.backend.begin_read().unwrap();
        assert!(storage.check_exists_be(&r, &id).unwrap());
        assert_eq!(storage.get_node_be(&r, &id).unwrap().id, id);
    }

    #[test]
    fn get_node_be_matches_get_node() {
        use super::super::backend::StorageBackend;
        use super::super::storage_methods::StorageMethods;
        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
        let id: u128 = 0xABCD;
        storage
            .with_write_txn(|wtxn| {
                storage.create_node(
                    wtxn,
                    "Person",
                    vec![("name".to_string(), Value::String("alice".to_string()))],
                    None,
                    Some(id),
                )?;
                Ok(())
            })
            .unwrap();

        // Old heed path and new backend path must agree, and the backend path
        // must read what the heed write produced (byte-identical encoding).
        let old = storage
            .with_read_txn(|rtxn| storage.get_node(rtxn, &id))
            .unwrap();
        let r = storage.backend.begin_read().unwrap();
        let new = storage.get_node_be(&r, &id).unwrap();
        assert_eq!(old.id, new.id);
        assert_eq!(old.label, new.label);
        assert!(storage.check_exists_be(&r, &id).unwrap());
        assert!(storage.get_node_be(&r, &0xDEADu128).is_err());
    }

    #[test]
    fn storage_backend_field_round_trips_in_struct() {
        use crate::helix_engine::storage_core::backend::{
            BackendKind, KeyRange, Namespace, StorageBackend,
        };
        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();

        // The production `backend` field (LMDB, wrapping a clone of graph_env)
        // round-trips through the StorageBackend trait inside the real struct.
        assert_eq!(storage.backend.kind(), BackendKind::Lmdb);

        let mut w = storage.backend.begin_write().unwrap();
        storage
            .backend
            .put(&mut w, Namespace::Metadata, b"bk1", b"bv1")
            .unwrap();
        storage
            .backend
            .put(&mut w, Namespace::Metadata, b"bk2", b"bv2")
            .unwrap();
        storage.backend.commit(w).unwrap();

        let r = storage.backend.begin_read().unwrap();
        let got = storage
            .backend
            .get_with(&r, Namespace::Metadata, b"bk1", |v| v.map(|b| b.to_vec()))
            .unwrap();
        assert_eq!(got, Some(b"bv1".to_vec()));

        let mut keys = Vec::new();
        storage
            .backend
            .scan(&r, Namespace::Metadata, KeyRange::all(), |k, _v| {
                keys.push(k.to_vec());
                true
            })
            .unwrap();
        // `Namespace::Metadata` resolves to the SAME LMDB database the struct
        // uses, so this scan also sees pre-existing entries the struct wrote
        // (e.g. the metadata "current" key) — proof the backend reads/writes
        // real Helion data through the shared env, not a private copy.
        assert!(
            keys.contains(&b"bk1".to_vec()),
            "backend write bk1 not visible via scan"
        );
        assert!(
            keys.contains(&b"bk2".to_vec()),
            "backend write bk2 not visible via scan"
        );
    }

    fn test_config() -> Config {
        Config::new(8, 32, 64, 1)
    }

    #[test]
    fn cgroup_pressure_subtracts_clean_active_file_cache() {
        let gib = 1024 * 1024 * 1024;
        let stat = "\
anon 3221225472
file 21474836480
inactive_file 1073741824
active_file 20401094656
shmem 1073741824
file_dirty 536870912
file_writeback 536870912
";

        assert_eq!(reclaim_aware_cgroup_pressure_bytes(24 * gib, stat), 6 * gib);
    }

    #[test]
    fn cgroup_pressure_falls_back_to_inactive_file_when_file_stat_missing() {
        let gib = 1024 * 1024 * 1024;
        let stat = "inactive_file 17179869184\n";

        assert_eq!(reclaim_aware_cgroup_pressure_bytes(18 * gib, stat), 2 * gib);
    }

    #[test]
    #[serial]
    fn max_dbs_respects_env_override_with_floor() {
        let key = "HELIX_MAX_DBS";
        let previous = std::env::var(key).ok();

        struct EnvRestore {
            key: &'static str,
            previous: Option<String>,
        }

        impl Drop for EnvRestore {
            fn drop(&mut self) {
                unsafe {
                    match &self.previous {
                        Some(value) => std::env::set_var(self.key, value),
                        None => std::env::remove_var(self.key),
                    }
                }
            }
        }

        let _restore = EnvRestore { key, previous };

        unsafe {
            std::env::set_var(key, "16384");
        }
        assert_eq!(HelixGraphStorage::max_dbs(), 16_384);

        unsafe {
            std::env::set_var(key, "1024");
        }
        assert_eq!(HelixGraphStorage::max_dbs(), DEFAULT_LMDB_MAX_DBS);
    }

    #[test]
    fn initial_map_size_adds_reopen_headroom_for_large_existing_file() {
        let temp_dir = TempDir::new().unwrap();
        let existing_size = 2usize * 1024 * 1024 * 1024;
        File::create(temp_dir.path().join("data.mdb"))
            .unwrap()
            .set_len(existing_size as u64)
            .unwrap();

        let size =
            HelixGraphStorage::initial_map_size(temp_dir.path().to_str().unwrap(), &test_config());
        let expected = existing_size
            .saturating_add(HelixGraphStorage::reopen_map_headroom_bytes())
            .min(HelixGraphStorage::max_map_size())
            .max(existing_size);

        assert_eq!(size, expected);
        assert!(size >= existing_size);
    }

    #[test]
    fn reloads_named_vector_spindle_config_and_index() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let spindle = SpindleConfig {
            mode: SpindleMode::ScalarInt8,
            keep_original: true,
            rescore: true,
            oversampling: 3,
            binary_dims: 32,
            turbo_dims: 64,
        };

        {
            let storage = HelixGraphStorage::new(&path, test_config()).unwrap();
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

            storage
                .named_vectors
                .create_vector_index(
                    storage.lmdb_env().unwrap(),
                    &mut txn,
                    "dense",
                    NamedVectorConfig {
                        size: 4,
                        distance: DistanceMetric::Cosine,
                        spindle: spindle.clone(),
                    },
                    HNSWConfig::new(Some(8), Some(32), Some(64)),
                )
                .unwrap();
            storage
                .set_named_vectors_metadata(&mut txn, storage.named_vectors.list_vectors())
                .unwrap();
            storage
                .set_dense_vector_spaces_metadata(
                    &mut txn,
                    storage.named_vectors.list_dense_vector_spaces(),
                )
                .unwrap();

            storage
                .named_vectors
                .with_core("dense", |core| {
                    core.insert::<Filter>(&mut txn, &[0.25, 0.5, 0.75, 1.0], Some(7), None)?;
                    Ok(())
                })
                .unwrap();

            txn.commit().unwrap();
        }

        let storage = HelixGraphStorage::new(&path, test_config()).unwrap();
        let config = storage.named_vectors.get_config("dense").unwrap();
        assert_eq!(config.size, 4);
        assert_eq!(config.distance, DistanceMetric::Cosine);
        assert_eq!(config.spindle, spindle);

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let results = storage
            .named_vectors
            .with_core("dense", |core| {
                core.search::<Filter>(
                    &core.backend.read_borrowed(&txn),
                    &[0.25, 0.5, 0.75, 1.0],
                    1,
                    None,
                    false,
                )
            })
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 7);
    }

    /// A build the executor refuses was never queued: the count taken before
    /// it was sent is given back, and no eviction waits for it.
    #[test]
    fn a_payload_index_build_the_executor_refuses_is_not_pending() {
        let temp_dir = TempDir::new().unwrap();
        let storage =
            HelixGraphStorage::new(temp_dir.path().to_str().unwrap(), test_config()).unwrap();

        let refused = storage.counted_as_pending(|| Err(GraphError::New("refused".into())));
        assert!(refused.is_err());
        assert!(!storage.payload_index_build_pending());

        storage.counted_as_pending(|| Ok(())).unwrap();
        assert!(storage.payload_index_build_pending());
    }

    #[test]
    fn reloads_indexing_threshold_override_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();

        {
            let storage = HelixGraphStorage::new(&path, test_config()).unwrap();
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            storage
                .set_indexing_threshold_override_metadata(&mut txn, Some(0))
                .unwrap();
            txn.commit().unwrap();
        }

        let storage = HelixGraphStorage::new(&path, test_config()).unwrap();
        assert_eq!(storage.named_vectors.effective_indexing_threshold(1024), 0);
    }

    #[test]
    fn corrupted_metadata_read_returns_error_not_defaults() {
        let temp_dir = TempDir::new().unwrap();
        let storage =
            HelixGraphStorage::new(temp_dir.path().to_str().unwrap(), test_config()).unwrap();

        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        storage
            .lmdb_metadata_db()
            .unwrap()
            .put(&mut txn, METADATA_CURRENT_KEY, b"not-valid-metadata")
            .unwrap();
        txn.commit().unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let err = storage.get_metadata(&txn).unwrap_err();
        assert!(matches!(err, GraphError::StorageError(_)));
        assert!(err.to_string().contains("corrupted collection metadata"));
    }

    #[test]
    fn collection_degraded_marker_fast_fails_writes() {
        let temp_dir = TempDir::new().unwrap();
        let storage =
            HelixGraphStorage::new(temp_dir.path().to_str().unwrap(), test_config()).unwrap();
        let fatal = GraphError::FatalCollectionStorage {
            code: "mdb_page_not_found",
            message: "LMDB MDB_PAGE_NOTFOUND: requested page not found".to_string(),
        };

        storage.mark_collection_degraded(&fatal, "test");

        let err = storage.with_write_txn(|_| Ok(())).unwrap_err();
        assert!(err.is_fatal_collection_storage());
        assert!(temp_dir
            .path()
            .join(COLLECTION_DEGRADED_MARKER_FILE)
            .exists());
    }

    #[test]
    fn lsm_write_conflict_does_not_mark_collection_degraded() {
        let temp_dir = TempDir::new().unwrap();
        let storage =
            HelixGraphStorage::new(temp_dir.path().to_str().unwrap(), test_config()).unwrap();

        let err = storage.backend_error(
            BackendError::Conflict("Closed error: detected newer DB client".to_string()),
            "test_lsm_conflict",
        );

        assert!(!err.is_fatal_collection_storage());
        assert!(!temp_dir
            .path()
            .join(COLLECTION_DEGRADED_MARKER_FILE)
            .exists());
    }

    #[test]
    #[serial]
    fn collection_degraded_marker_survives_reopen() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        {
            let storage = HelixGraphStorage::new(&path, test_config()).unwrap();
            let fatal = GraphError::FatalCollectionStorage {
                code: "mdb_page_not_found",
                message: "LMDB MDB_PAGE_NOTFOUND: requested page not found".to_string(),
            };
            storage.mark_collection_degraded(&fatal, "test");
        }

        let storage = HelixGraphStorage::new(&path, test_config()).unwrap();
        let err = storage.ensure_not_degraded().unwrap_err();
        assert!(err.is_fatal_collection_storage());
        assert!(err.to_string().contains("mdb_page_not_found"));
    }

    #[test]
    fn payload_numeric_value_rejects_non_finite_floats() {
        assert_eq!(
            HelixGraphStorage::payload_numeric_value(&Value::F64(42.0)),
            Some(42.0)
        );
        assert!(HelixGraphStorage::payload_numeric_value(&Value::F32(f32::NAN)).is_none());
        assert!(HelixGraphStorage::payload_numeric_value(&Value::F64(f64::INFINITY)).is_none());
    }

    #[test]
    fn with_write_txn_auto_resizes_on_map_full() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();

        // Use a tiny map_size (1 page = ~16KB) so writes fill it fast.
        let page_sz = page_size::get();
        let tiny_map = page_sz * 4; // 4 pages — enough for metadata but fills quickly
        let env = unsafe {
            heed3::EnvOpenOptions::new()
                .map_size(tiny_map)
                .max_dbs(64)
                .max_readers(200)
                .open(std::path::Path::new(&path))
                .unwrap()
        };

        let mut wtxn = env.write_txn().unwrap();
        let nodes_db = env
            .database_options()
            .types::<U128<BE>, Bytes>()
            .name(DB_NODES)
            .create(&mut wtxn)
            .unwrap();
        let edges_db = env
            .database_options()
            .types::<U128<BE>, Bytes>()
            .name(DB_EDGES)
            .create(&mut wtxn)
            .unwrap();
        let out_edges_db = env
            .database_options()
            .types::<Bytes, Bytes>()
            .flags(DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED)
            .name(DB_OUT_EDGES)
            .create(&mut wtxn)
            .unwrap();
        let in_edges_db = env
            .database_options()
            .types::<Bytes, Bytes>()
            .flags(DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED)
            .name(DB_IN_EDGES)
            .create(&mut wtxn)
            .unwrap();
        let edge_path_idx = env
            .database_options()
            .types::<Bytes, Bytes>()
            .name(DB_EDGE_PATH_IDX)
            .create(&mut wtxn)
            .unwrap();
        let metadata_db = env
            .database_options()
            .types::<Str, Bytes>()
            .name(DB_METADATA)
            .create(&mut wtxn)
            .unwrap();
        let wal = WalWriter::open(
            std::path::Path::new(&path).join("wal"),
            SyncPolicy::default(),
        )
        .unwrap();
        let backend = Arc::new(AnyBackend::Lmdb(
            crate::helix_engine::storage_core::backend_lmdb::LmdbBackend::from_env(env.clone()),
        ));
        let vectors = VectorCore::new(
            &env,
            &mut wtxn,
            HNSWConfig::new(None, None, None),
            Arc::clone(&backend),
        )
        .unwrap();
        wtxn.commit().unwrap();

        let named_vectors = NamedVectorManager::new(
            std::path::Path::new(&path)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
        );
        named_vectors.attach_backend(Arc::clone(&backend));

        let storage = HelixGraphStorage {
            collection_path: PathBuf::from(&path),
            backend,
            graph_env: Some(env),
            nodes_db: Some(nodes_db),
            edges_db: Some(edges_db),
            out_edges_db: Some(out_edges_db),
            in_edges_db: Some(in_edges_db),
            edge_path_idx: Some(edge_path_idx),
            metadata_db: Some(metadata_db),
            metadata_snapshot: RwLock::new(StorageMetadata::new(Vec::new())),
            metadata_snapshot_read_at: parking_lot::Mutex::new(None),
            last_lsm_sidecar_upload_ms: AtomicU64::new(0),
            last_lsm_sidecar_fingerprint: AtomicU64::new(0),
            lsm_sidecar_put_held: std::sync::atomic::AtomicBool::new(false),
            sidecar_put: parking_lot::Mutex::new(()),
            adjacency_backfill_mark: std::sync::atomic::AtomicU8::new(0),
            adjacency_mark_sent: std::sync::atomic::AtomicU8::new(0),
            secondary_indices: HashMap::new(),
            multi_indices: HashMap::new(),
            payload_indices: RwLock::new(HashMap::new()),
            text_walk_cache: parking_lot::Mutex::new(
                crate::helix_engine::storage_core::text_walk_cache::TextWalkCache::from_env(),
            ),
            vectors,
            named_vectors,
            wal,
            optimizer_running: std::sync::atomic::AtomicBool::new(false),
            last_upsert_at: std::sync::atomic::AtomicU64::new(0),
            point_mutations: std::sync::atomic::AtomicU64::new(0),
            point_mutations_in_flight: AtomicUsize::new(0),
            resize_gate: parking_lot::RwLock::new(()),
            resize_writers_pending: AtomicUsize::new(0),
            resize_writers_active: AtomicUsize::new(0),
            resize_wait_lock: Mutex::new(()),
            resize_wait_cv: Condvar::new(),
            active_lmdb_txns: AtomicUsize::new(0),
            write_txn_gate: Arc::new(Mutex::new(())),
            payload_index_admin_gate: Mutex::new(()),
            payload_index_builds_pending: AtomicUsize::new(0),
            payload_index_build_lock: Mutex::new(()),
            metadata_corruption_logged: std::sync::atomic::AtomicBool::new(false),
            collection_degraded: std::sync::atomic::AtomicBool::new(false),
            closing_for_shutdown: std::sync::atomic::AtomicBool::new(false),
            degraded_reason: RwLock::new(None),
            dense_metadata_dirty: std::sync::atomic::AtomicBool::new(false),
            last_reader_refresh_ms: std::sync::atomic::AtomicU64::new(0),
            reader_refresh_inflight: std::sync::atomic::AtomicBool::new(false),
            last_reader_read_at_ms: std::sync::atomic::AtomicU64::new(0),
            reader_poll_tier_fast: std::sync::atomic::AtomicBool::new(
                crate::helix_engine::storage_core::backend_lsm::reader_cold_open_starts_fast(),
            ),
            reader_poll_tier_transition: std::sync::atomic::AtomicBool::new(false),
            last_reader_promote_failed_at_ms: AtomicU64::new(0),
            reader_promote_consecutive_failures: std::sync::atomic::AtomicU32::new(0),
            last_write_promoted_at_ms: std::sync::atomic::AtomicU64::new(0),
            reader_view_reconcile: ReaderViewReconcile::default(),
            recount_inflight: std::sync::atomic::AtomicBool::new(false),
            payload_index_gc_inflight: std::sync::atomic::AtomicBool::new(false),
        };

        let initial_map_size = storage.lmdb_env().unwrap().info().map_size;
        assert_eq!(initial_map_size, tiny_map);

        // Write enough data to trigger MapFull + auto-resize.
        // A large payload will fill 4 pages quickly.
        let big_payload = "x".repeat(page_sz);
        let result = storage.with_write_txn(|txn| {
            for i in 0u128..20 {
                let serialized = big_payload.as_bytes();
                storage.lmdb_nodes_db().unwrap().put(txn, &i, serialized)?;
            }
            Ok(())
        });

        assert!(
            result.is_ok(),
            "with_write_txn should auto-resize: {:?}",
            result.err()
        );

        let final_map_size = storage.lmdb_env().unwrap().info().map_size;
        assert!(
            final_map_size > initial_map_size,
            "Map should have grown: initial={}, final={}",
            initial_map_size,
            final_map_size
        );

        // Verify data is actually persisted after resize.
        let rtxn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let count = storage
            .lmdb_nodes_db()
            .unwrap()
            .iter(&rtxn)
            .unwrap()
            .count();
        assert_eq!(count, 20);
    }

    #[test]
    #[serial]
    fn write_txns_retry_retryable_lmdb_invalid_argument() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = HelixGraphStorage::new(&path, test_config()).unwrap();
        let retryable = || {
            GraphError::VectorError(
                "VectorError: Vector core error: heed error: Invalid argument (os error 22)"
                    .to_string(),
            )
        };

        let mut shared_attempts = 0usize;
        storage
            .with_write_txn(|txn| {
                shared_attempts += 1;
                if shared_attempts == 1 {
                    return Err(retryable());
                }
                storage
                    .lmdb_metadata_db()
                    .unwrap()
                    .put(txn, "shared_retry", b"ok")?;
                Ok(())
            })
            .unwrap();

        let mut exclusive_attempts = 0usize;
        storage
            .with_exclusive_write_txn(|txn| {
                exclusive_attempts += 1;
                if exclusive_attempts == 1 {
                    return Err(retryable());
                }
                storage
                    .lmdb_metadata_db()
                    .unwrap()
                    .put(txn, "exclusive_retry", b"ok")?;
                Ok(())
            })
            .unwrap();

        assert_eq!(shared_attempts, 2);
        assert_eq!(exclusive_attempts, 2);

        let rtxn = storage.lmdb_env().unwrap().read_txn().unwrap();
        assert_eq!(
            storage
                .lmdb_metadata_db()
                .unwrap()
                .get(&rtxn, "shared_retry")
                .unwrap(),
            Some(b"ok".as_slice())
        );
        assert_eq!(
            storage
                .lmdb_metadata_db()
                .unwrap()
                .get(&rtxn, "exclusive_retry")
                .unwrap(),
            Some(b"ok".as_slice())
        );
    }

    #[test]
    fn with_write_txn_releases_resize_shared_guard_after_txn_creation() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = std::sync::Arc::new(HelixGraphStorage::new(&path, test_config()).unwrap());
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let writer_storage = std::sync::Arc::clone(&storage);

        let writer = std::thread::spawn(move || {
            writer_storage
                .with_write_txn(|txn| {
                    writer_storage.lmdb_metadata_db().unwrap().put(
                        txn,
                        "resize_guard_test",
                        b"1",
                    )?;
                    entered_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Ok(())
                })
                .unwrap();
        });

        entered_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap();
        let exclusive = storage.resize_gate.try_write();
        assert!(
            exclusive.is_some(),
            "active with_write_txn must not hold shared resize_gate after txn creation"
        );
        drop(exclusive);

        release_tx.send(()).unwrap();
        writer.join().unwrap();
    }

    #[test]
    fn manual_resize_safe_txn_helpers_release_resize_guard_after_txn_creation() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = HelixGraphStorage::new(&path, test_config()).unwrap();

        {
            let txn = storage.begin_resize_safe_read_txn().unwrap();
            assert!(
                storage.resize_gate.try_write().is_some(),
                "begin_resize_safe_read_txn must not hold shared resize gate after txn creation"
            );
            drop(txn);
        }
        assert!(
            storage.resize_gate.try_write().is_some(),
            "resize gate should remain open after manual read transaction drops"
        );

        {
            let (_write_txn_gate, mut txn) = storage.begin_resize_safe_write_txn().unwrap();
            storage
                .lmdb_metadata_db()
                .unwrap()
                .put(&mut txn, "manual_resize_gate_test", b"1")
                .unwrap();
            assert!(
                storage.resize_gate.try_write().is_some(),
                "begin_resize_safe_write_txn must not hold shared resize gate after txn creation"
            );
            txn.commit().unwrap();
        }
        assert!(
            storage.resize_gate.try_write().is_some(),
            "resize gate should remain open after manual write transaction commits"
        );
    }

    #[test]
    fn active_tracked_read_transaction_blocks_resize_without_holding_resize_gate() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = std::sync::Arc::new(HelixGraphStorage::new(&path, test_config()).unwrap());
        let txn = storage.begin_resize_safe_read_txn().unwrap();

        assert!(
            storage.resize_gate.try_write().is_some(),
            "tracked read txn must not hold the legacy resize_gate"
        );
        assert_eq!(storage.active_lmdb_txns.load(Ordering::Acquire), 1);

        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let resize_storage = std::sync::Arc::clone(&storage);
        let resize_thread = std::thread::spawn(move || {
            let _guard = resize_storage
                .write_resize_guard_for("active_tracked_read_transaction_blocks_resize")
                .unwrap();
            entered_tx.send(()).unwrap();
        });

        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(
            entered_rx.try_recv().is_err(),
            "resize must wait for the active tracked read transaction to drop"
        );
        assert_eq!(storage.resize_writers_pending.load(Ordering::Acquire), 1);

        drop(txn);
        entered_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap();
        resize_thread.join().unwrap();
        assert_eq!(storage.active_lmdb_txns.load(Ordering::Acquire), 0);
        assert_eq!(storage.resize_writers_pending.load(Ordering::Acquire), 0);
        assert_eq!(storage.resize_writers_active.load(Ordering::Acquire), 0);
    }

    #[test]
    fn pending_resize_blocks_new_tracked_read_transaction_admission() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = std::sync::Arc::new(HelixGraphStorage::new(&path, test_config()).unwrap());
        let legacy_reader = storage.resize_gate.read();

        let (resize_started_tx, resize_started_rx) = std::sync::mpsc::channel();
        let (release_resize_tx, release_resize_rx) = std::sync::mpsc::channel();
        let resize_storage = std::sync::Arc::clone(&storage);
        let resize_thread = std::thread::spawn(move || {
            resize_started_tx.send(()).unwrap();
            let _guard = resize_storage
                .write_resize_guard_for("pending_resize_blocks_new_tracked_read")
                .unwrap();
            release_resize_rx.recv().unwrap();
        });

        resize_started_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while storage.resize_writers_pending.load(Ordering::Acquire) == 0 {
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        let (reader_entered_tx, reader_entered_rx) = std::sync::mpsc::channel();
        let reader_storage = std::sync::Arc::clone(&storage);
        let reader_thread = std::thread::spawn(move || {
            let txn = reader_storage.begin_resize_safe_read_txn().unwrap();
            reader_entered_tx.send(()).unwrap();
            drop(txn);
        });

        assert!(
            reader_entered_rx
                .recv_timeout(std::time::Duration::from_millis(100))
                .is_err(),
            "pending resize must park new tracked transaction admission"
        );
        drop(legacy_reader);
        release_resize_tx.send(()).unwrap();
        reader_entered_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap();
        resize_thread.join().unwrap();
        reader_thread.join().unwrap();
        assert_eq!(storage.resize_writers_pending.load(Ordering::Acquire), 0);
        assert_eq!(storage.resize_writers_active.load(Ordering::Acquire), 0);
    }

    #[test]
    fn nested_dense_read_admission_does_not_park_behind_outer_read_during_pending_resize() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = std::sync::Arc::new(HelixGraphStorage::new(&path, test_config()).unwrap());
        let outer = storage.begin_resize_safe_read_txn().unwrap();

        let (resize_acquired_tx, resize_acquired_rx) = std::sync::mpsc::channel();
        let (release_resize_tx, release_resize_rx) = std::sync::mpsc::channel();
        let resize_storage = std::sync::Arc::clone(&storage);
        let resize_thread = std::thread::spawn(move || {
            let _guard = resize_storage
                .write_resize_guard_for("nested_dense_read_admission_test")
                .unwrap();
            resize_acquired_tx.send(()).unwrap();
            release_resize_rx.recv().unwrap();
        });

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while storage.resize_writers_pending.load(Ordering::Acquire) == 0 {
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        let provider = storage.nested_dense_read_provider(&outer);
        std::thread::scope(|scope| {
            let nested = scope.spawn(|| {
                provider.with_dense_read_txn(|rtxn| {
                    assert!(storage
                        .lmdb_metadata_db()
                        .unwrap()
                        .get(rtxn, "missing")
                        .unwrap()
                        .is_none());
                    assert!(
                        storage.active_lmdb_txns.load(Ordering::Acquire) >= 2,
                        "nested read should be counted alongside the outer read"
                    );
                    Ok::<(), crate::helix_engine::types::VectorError>(())
                })
            });
            nested.join().unwrap().unwrap();
        });
        assert!(
            resize_acquired_rx
                .recv_timeout(std::time::Duration::from_millis(100))
                .is_err(),
            "resize must still wait for the outer read transaction"
        );

        drop(provider);
        drop(outer);
        resize_acquired_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap();
        release_resize_tx.send(()).unwrap();
        resize_thread.join().unwrap();
        assert_eq!(storage.active_lmdb_txns.load(Ordering::Acquire), 0);
        assert_eq!(storage.resize_writers_pending.load(Ordering::Acquire), 0);
        assert_eq!(storage.resize_writers_active.load(Ordering::Acquire), 0);
    }

    #[test]
    fn with_write_txn_waiting_for_write_gate_does_not_hold_resize_guard() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = std::sync::Arc::new(HelixGraphStorage::new(&path, test_config()).unwrap());
        let gate = storage.write_txn_gate.lock().unwrap();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let writer_storage = std::sync::Arc::clone(&storage);

        let writer = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            writer_storage
                .with_write_txn(|txn| {
                    writer_storage.lmdb_metadata_db().unwrap().put(
                        txn,
                        "write_gate_wait_test",
                        b"1",
                    )?;
                    Ok(())
                })
                .unwrap();
        });

        started_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        let exclusive = storage.resize_gate.try_write();
        assert!(
            exclusive.is_some(),
            "writer blocked on write_txn_gate must not hold shared resize_gate"
        );
        drop(exclusive);
        drop(gate);
        writer.join().unwrap();
    }

    #[test]
    fn write_txn_gate_recovers_after_poisoned_coordination_lock() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = std::sync::Arc::new(HelixGraphStorage::new(&path, test_config()).unwrap());
        let poison_storage = std::sync::Arc::clone(&storage);

        let _ = std::thread::spawn(move || {
            let _guard = poison_storage.write_txn_gate.lock().unwrap();
            panic!("poison write_txn_gate for recovery test");
        })
        .join();

        let guard = storage.lock_write_txn_gate().unwrap();
        drop(guard);
    }

    #[test]
    fn resize_gate_recovers_after_poisoned_coordination_lock() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = std::sync::Arc::new(HelixGraphStorage::new(&path, test_config()).unwrap());

        let read_storage = std::sync::Arc::clone(&storage);
        let _ = std::thread::spawn(move || {
            let _guard = read_storage.resize_gate.write();
            panic!("poison resize_gate writer side for recovery test");
        })
        .join();

        let read_guard = storage.read_resize_guard_for("recovery_test").unwrap();
        drop(read_guard);
        let write_guard = storage.write_resize_guard_for("recovery_test").unwrap();
        drop(write_guard);
    }

    #[test]
    fn pending_resize_blocks_new_read_resize_guards() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = std::sync::Arc::new(HelixGraphStorage::new(&path, test_config()).unwrap());
        let existing_reader = storage.resize_gate.read();

        let (writer_started_tx, writer_started_rx) = std::sync::mpsc::channel();
        let (writer_acquired_tx, writer_acquired_rx) = std::sync::mpsc::channel();
        let (release_writer_tx, release_writer_rx) = std::sync::mpsc::channel();
        let writer_storage = std::sync::Arc::clone(&storage);
        let writer = std::thread::spawn(move || {
            writer_started_tx.send(()).unwrap();
            let _exclusive = writer_storage
                .write_resize_guard_for("pending_resize_blocks_new_read_resize_guards")
                .unwrap();
            writer_acquired_tx.send(()).unwrap();
            release_writer_rx.recv().unwrap();
        });

        writer_started_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while storage.resize_writers_pending.load(Ordering::Acquire) == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "writer did not mark resize as pending"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        let (reader_acquired_tx, reader_acquired_rx) = std::sync::mpsc::channel();
        let reader_storage = std::sync::Arc::clone(&storage);
        let reader = std::thread::spawn(move || {
            let _shared = reader_storage.read_resize_guard_if_needed().unwrap();
            reader_acquired_tx.send(()).unwrap();
        });

        assert!(
            reader_acquired_rx
                .recv_timeout(std::time::Duration::from_millis(50))
                .is_err(),
            "new readers must wait while an exclusive resize is pending"
        );
        drop(existing_reader);
        writer_acquired_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap();
        assert!(
            reader_acquired_rx
                .recv_timeout(std::time::Duration::from_millis(50))
                .is_err(),
            "new readers must wait while the resize holds the exclusive gate"
        );

        release_writer_tx.send(()).unwrap();
        reader_acquired_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap();
        writer.join().unwrap();
        reader.join().unwrap();
    }

    #[test]
    fn parked_reader_completes_after_pending_resize_clears_without_error() {
        // W1 regression guard: a reader that arrives while a resize writer is
        // pending must (a) not return an error, (b) not spin, and (c) acquire
        // its shared guard and complete once the resize finishes. This is the
        // property that turns a multi-minute resize from a pod-wide worker
        // pool starvation + liveness kill into bounded per-collection latency.
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = std::sync::Arc::new(HelixGraphStorage::new(&path, test_config()).unwrap());

        // Hold an existing shared reader so the resize writer must queue.
        let existing_reader = storage.resize_gate.read();

        let (release_writer_tx, release_writer_rx) = std::sync::mpsc::channel();
        let writer_storage = std::sync::Arc::clone(&storage);
        let writer = std::thread::spawn(move || {
            let _exclusive = writer_storage
                .write_resize_guard_for("parked_reader_test")
                .unwrap();
            release_writer_rx.recv().unwrap();
        });

        // Wait until the writer has marked the resize pending.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while storage.resize_writers_pending.load(Ordering::Acquire) == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "writer did not mark resize as pending"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        let (reader_done_tx, reader_done_rx) = std::sync::mpsc::channel();
        let reader_storage = std::sync::Arc::clone(&storage);
        let reader = std::thread::spawn(move || {
            // Must return Ok, never an error, even though it has to wait.
            let guard = reader_storage.read_resize_guard_if_needed();
            reader_done_tx.send(guard.is_ok()).unwrap();
        });

        // Reader is parked: it must not have completed while the resize is
        // still pending and the existing shared reader still holds the gate.
        assert!(
            reader_done_rx
                .recv_timeout(std::time::Duration::from_millis(100))
                .is_err(),
            "reader must park while resize is pending, not proceed or error"
        );

        // Let the writer acquire the exclusive gate, but keep it active. The
        // reader must stay parked in the condvar path instead of falling into
        // `resize_gate.read()` on a blocking worker.
        drop(existing_reader);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while storage.resize_writers_active.load(Ordering::Acquire) == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "writer did not enter active resize window"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(
            reader_done_rx
                .recv_timeout(std::time::Duration::from_millis(100))
                .is_err(),
            "reader must park while resize is active, not block on resize_gate.read()"
        );

        // Let the resize finish.
        release_writer_tx.send(()).unwrap();
        writer.join().unwrap();

        // The parked reader now wakes, acquires its shared guard, and the
        // call returns Ok — no error, no rejection.
        let reader_ok = reader_done_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("parked reader did not complete after resize cleared");
        assert!(reader_ok, "read_resize_guard_if_needed must return Ok");
        reader.join().unwrap();
    }

    #[test]
    fn ensure_map_headroom_does_not_wait_for_write_txn_gate() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = std::sync::Arc::new(HelixGraphStorage::new(&path, test_config()).unwrap());

        let info = storage.lmdb_env().unwrap().info();
        let used = info.last_page_number.saturating_mul(page_size::get());
        let available = info.map_size.saturating_sub(used);
        let requested = available.saturating_add(1024 * 1024);

        let gate = storage.write_txn_gate.lock().unwrap();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let resize_storage = std::sync::Arc::clone(&storage);
        let resize = std::thread::spawn(move || {
            done_tx
                .send(resize_storage.ensure_map_headroom(requested))
                .unwrap();
        });

        let while_write_gate_held = done_rx.recv_timeout(std::time::Duration::from_secs(2));

        drop(gate);
        resize.join().unwrap();

        assert!(
            matches!(while_write_gate_held, Ok(Ok(()))),
            "resize growth should not queue behind write_txn_gate: {while_write_gate_held:?}"
        );
    }

    /// The backend's catalog thread opens a write transaction on the env
    /// outside the resize gate, at open and every 250 ms. A resize that met
    /// one was refused by LMDB (`Invalid argument`), and a pass that began
    /// while the map was swapped read the meta page through the old mapping
    /// and killed the process (SIGBUS). The resize waits for the pass now.
    #[test]
    #[serial]
    fn ensure_map_headroom_waits_for_a_catalog_pass_in_hand() {
        use std::sync::mpsc;
        use std::time::Duration;

        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = std::sync::Arc::new(HelixGraphStorage::new(&path, test_config()).unwrap());
        let info = storage.lmdb_env().unwrap().info();
        let initial = info.map_size;
        let used = info.last_page_number.saturating_mul(page_size::get());
        let requested = initial.saturating_sub(used).saturating_add(1024 * 1024);
        assert!(
            initial < HelixGraphStorage::max_map_size(),
            "the test storage opens below its map cap, so the request has to grow it"
        );

        // Given a catalog pass with its write transaction open.
        let (opened_tx, opened_rx) = mpsc::channel();
        let (finish_tx, finish_rx) = mpsc::channel::<()>();
        let pass_storage = std::sync::Arc::clone(&storage);
        let pass = std::thread::spawn(move || {
            let AnyBackend::Lmdb(backend) = &*pass_storage.backend else {
                panic!("the default test storage is LMDB");
            };
            backend.hold_catalog_txn_for_test(|| {
                opened_tx.send(()).unwrap();
                // Bounded, so that a failed assertion below ends the test.
                let _ = finish_rx.recv_timeout(Duration::from_secs(30));
            })
        });
        opened_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("the pass did not open its transaction");

        // When the map has to grow.
        let (done_tx, done_rx) = mpsc::channel();
        let resize_storage = std::sync::Arc::clone(&storage);
        let resize = std::thread::spawn(move || {
            done_tx
                .send(resize_storage.ensure_map_headroom(requested))
                .unwrap();
        });

        // Then the resize waits for the pass,
        let under_the_pass = done_rx.recv_timeout(Duration::from_millis(500));
        // and grows the map once the pass has committed.
        finish_tx.send(()).unwrap();
        pass.join().unwrap().unwrap();
        let after_the_pass = match under_the_pass {
            Err(mpsc::RecvTimeoutError::Timeout) => done_rx.recv_timeout(Duration::from_secs(30)),
            early => panic!("the resize did not wait for the catalog pass: {early:?}"),
        };
        resize.join().unwrap();
        assert!(
            matches!(after_the_pass, Ok(Ok(()))),
            "the resize after the pass: {after_the_pass:?}"
        );
        let grown = storage.lmdb_env().unwrap().info().map_size;
        assert!(grown > initial, "map {initial} -> {grown}");
    }

    #[test]
    fn async_payload_index_build_is_hidden_until_ready_and_catches_concurrent_writes() {
        use crate::helix_engine::storage_core::upsert::NodeUpsert;

        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = std::sync::Arc::new(HelixGraphStorage::new(&path, test_config()).unwrap());
        let field = "repo";
        let schema = PayloadIndexSchema::Keyword;
        let job_id = "test-job".to_string();

        let db = storage
            .with_write_txn(|txn| {
                storage
                    .lmdb_env()
                    .unwrap()
                    .database_options()
                    .types::<Bytes, Bytes>()
                    .flags(DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED)
                    .name(&HelixGraphStorage::payload_index_db_name(field, &schema))
                    .create(txn)
                    .map_err(GraphError::from)
            })
            .unwrap();
        let handle = PayloadIndexHandle::building(schema.clone(), db, job_id.clone());
        storage
            .payload_indices
            .write()
            .unwrap()
            .insert(field.to_string(), handle.clone());

        assert_eq!(storage.has_payload_index(field), None);

        storage
            .with_write_txn(|txn| {
                for id in 1_u128..=3 {
                    storage.upsert_node(
                        txn,
                        &NodeUpsert {
                            id,
                            label: "Symbol".into(),
                            properties: HashMap::from([(
                                field.into(),
                                Value::String("context-engine".into()),
                            )]),
                        },
                    )?;
                }
                Ok(())
            })
            .unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        assert!(storage
            .get_nodes_by_payload_value(&txn, field, &Value::String("context-engine".into()))
            .is_err());
        drop(txn);

        assert!(handle.mark_ready(&job_id));
        assert_eq!(storage.has_payload_index(field), Some(schema));

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let hits = storage
            .get_nodes_by_payload_value(&txn, field, &Value::String("context-engine".into()))
            .unwrap();
        assert_eq!(hits, vec![1, 2, 3]);
    }

    #[test]
    fn try_list_payload_index_statuses_returns_without_waiting_on_busy_locks() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = std::sync::Arc::new(HelixGraphStorage::new(&path, test_config()).unwrap());
        let field = "repo";
        let schema = PayloadIndexSchema::Keyword;
        let job_id = "busy-job".to_string();

        let db = storage
            .with_write_txn(|txn| {
                storage
                    .lmdb_env()
                    .unwrap()
                    .database_options()
                    .types::<Bytes, Bytes>()
                    .flags(DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED)
                    .name(&HelixGraphStorage::payload_index_db_name(field, &schema))
                    .create(txn)
                    .map_err(GraphError::from)
            })
            .unwrap();
        let handle = PayloadIndexHandle::building(schema, db, job_id);
        storage
            .payload_indices
            .write()
            .unwrap()
            .insert(field.to_string(), handle.clone());

        let state_guard = handle.state.write().unwrap();
        let statuses = storage.try_list_payload_index_statuses().unwrap().unwrap();
        assert!(statuses.is_empty());
        drop(state_guard);

        let map_guard = storage.payload_indices.write().unwrap();
        assert!(storage.try_list_payload_index_statuses().unwrap().is_none());
        drop(map_guard);
    }

    #[test]
    fn quiesced_empty_payload_index_build_publishes_ready_atomically() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = std::sync::Arc::new(HelixGraphStorage::new(&path, test_config()).unwrap());
        let field = "repo";

        let status = storage
            .enqueue_payload_index_build("test", field, PayloadIndexSchema::Keyword, true)
            .unwrap();

        assert_eq!(status.status, "ready");
        assert_eq!(
            storage.has_payload_index(field),
            Some(PayloadIndexSchema::Keyword)
        );

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let metadata = storage.get_metadata(&txn).unwrap();
        assert_eq!(
            metadata.payload_indices.get(field),
            Some(&PayloadIndexSchema::Keyword)
        );
    }

    /// The sweep of leftover build-delta rows takes what the sidecar held when
    /// it began. While an index is Building every foreground write records a
    /// row, so a sweep that scanned until it found nothing followed the
    /// writers, a commit a batch, and the build under it did not start.
    #[test]
    fn lsm_build_delta_sweep_takes_only_the_rows_it_began_with() {
        use crate::helix_engine::storage_core::backend_lsm::lsm_round_trips_for_test;
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        use std::time::{Duration, Instant};

        let dir = TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(
            dir.path().to_str().unwrap(),
            test_config().with_lsm_in_memory(),
        )
        .unwrap();
        let db_name =
            HelixGraphStorage::payload_index_db_name("repo", &PayloadIndexSchema::Keyword);
        // As a foreground write records its row: in a buffered commit, which
        // the next read sees at once.
        let record = |node: u128| {
            storage
                .with_write_backend_buffered(|w| {
                    storage.record_payload_index_build_delta(w, &db_name, node, None)
                })
                .unwrap()
        };
        let recorded = || {
            let read = storage.backend.begin_read().unwrap();
            let mut nodes = Vec::new();
            storage
                .backend
                .scan(
                    &read,
                    Namespace::PayloadIndexBuild(&db_name),
                    KeyRange::all(),
                    |key, _| {
                        nodes.push(u128::from_be_bytes(key.try_into().unwrap()));
                        true
                    },
                )
                .unwrap();
            nodes
        };
        let sweep_round_trips = || {
            let before = lsm_round_trips_for_test();
            storage.clear_payload_index_build_deltas(&db_name).unwrap();
            lsm_round_trips_for_test() - before
        };

        // What the sweep costs on 150 rows while nobody writes.
        for node in 1..=150u128 {
            record(node);
        }
        assert_eq!(recorded().len(), 150);
        let quiet = sweep_round_trips();
        assert!(quiet > 0, "the sweep's reads and commits are counted");
        assert!(recorded().is_empty());

        // Given 100 leftover rows, and a writer that records a row every
        // millisecond and is 50 rows in. It gives up after ten seconds, so
        // that a sweep that follows it returns as well.
        for node in 1..=100u128 {
            record(node);
        }
        let stop = AtomicBool::new(false);
        let written = AtomicU64::new(0);
        let busy = std::thread::scope(|scope| {
            scope.spawn(|| {
                let began = Instant::now();
                let mut node = 1_000u128;
                while !stop.load(Ordering::Acquire) && began.elapsed() < Duration::from_secs(10) {
                    record(node);
                    node += 1;
                    written.fetch_add(1, Ordering::Release);
                    std::thread::sleep(Duration::from_millis(1));
                }
            });
            while written.load(Ordering::Acquire) < 50 {
                std::thread::yield_now();
            }

            // When the sweep runs under that writer.
            let busy = sweep_round_trips();
            stop.store(true, Ordering::Release);
            busy
        });

        // Then it cost what its own rows cost, not a scan and a commit for
        // every batch the writer added meanwhile, and no leftover remains.
        assert!(
            busy <= 4 * quiet + 16,
            "{busy} LSM round trips under a writer, {quiet} without"
        );
        assert!(recorded().iter().all(|node| *node >= 1_000));
    }

    /// The async build sweeps the build-delta rows after its Ready flip only
    /// when it is the job that flipped it, and under the admin gate. A job
    /// that lost the index between its drain and its publish swept anyway,
    /// after the gate: it deleted the rows a newer build of the field had
    /// recorded for its own drain.
    #[test]
    fn lsm_async_build_sweeps_deltas_only_when_it_published() {
        fn building(storage: &HelixGraphStorage, job_id: &str) {
            storage.payload_indices.write().unwrap().insert(
                "repo".to_string(),
                PayloadIndexHandle::building_lsm(PayloadIndexSchema::Keyword, job_id.to_string()),
            );
        }
        // As a foreground write records its row while the index is Building.
        fn record(storage: &HelixGraphStorage, db_name: &str, node: u128) {
            storage
                .with_write_backend_buffered(|w| {
                    storage.record_payload_index_build_delta(w, db_name, node, None)
                })
                .unwrap();
        }
        fn before_publish(then: Box<dyn FnOnce()>) {
            BEFORE_PAYLOAD_INDEX_PUBLISH_FOR_TEST.with(|hook| *hook.borrow_mut() = Some(then));
        }

        let dir = TempDir::new().unwrap();
        let storage = Arc::new(
            HelixGraphStorage::new(
                dir.path().to_str().unwrap(),
                test_config().with_lsm_in_memory(),
            )
            .unwrap(),
        );
        let schema = PayloadIndexSchema::Keyword;
        let db_name = HelixGraphStorage::payload_index_db_name("repo", &schema);
        let job = dummy_payload_index_job("repo");
        let recorded = || {
            let read = storage.backend.begin_read().unwrap();
            let mut nodes = Vec::new();
            storage
                .backend
                .scan(
                    &read,
                    Namespace::PayloadIndexBuild(&db_name),
                    KeyRange::all(),
                    |key, _| {
                        nodes.push(u128::from_be_bytes(key.try_into().unwrap()));
                        true
                    },
                )
                .unwrap();
            nodes
        };

        // Given a build past its drain, and a write that records one more
        // row before the publish.
        building(&storage, &job.job_id);
        let (late, name) = (Arc::clone(&storage), db_name.clone());
        before_publish(Box::new(move || record(&late, &name, 5)));

        // When the job publishes.
        storage.build_payload_index_job(&job).unwrap();

        // Then the index is Ready and the row, garbage by then, is gone.
        assert_eq!(storage.has_payload_index("repo"), Some(schema));
        assert!(recorded().is_empty());

        // Given the same job on a new build, and the index dropped and
        // created again between its drain and its publish, with a write
        // under the newer build.
        building(&storage, &job.job_id);
        let (newer, name) = (Arc::clone(&storage), db_name.clone());
        before_publish(Box::new(move || {
            building(&newer, "job-newer");
            record(&newer, &name, 7);
        }));

        // When the job goes on to its publish.
        storage.build_payload_index_job(&job).unwrap();

        // Then it published nothing, and the newer build's row is still
        // there for that build's drain.
        assert!(storage
            .payload_index_job_active("repo", "job-newer")
            .unwrap());
        assert_eq!(recorded(), vec![7]);
    }

    /// The pre-Ready drain repairs a row a backfill chunk resurrected and
    /// consumes its delta in one checked pass, and writes nothing once its
    /// job no longer owns the index.
    #[test]
    fn lsm_build_drain_repairs_resurrected_rows_in_one_pass() {
        use crate::helix_engine::storage_core::upsert::NodeUpsert;

        let dir = TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(
            dir.path().to_str().unwrap(),
            test_config().with_lsm_in_memory(),
        )
        .unwrap();
        let schema = PayloadIndexSchema::Keyword;
        let db_name = HelixGraphStorage::payload_index_db_name("repo", &schema);
        let key = |s: &str| {
            HelixGraphStorage::stable_index_key_for_value(&Value::String(s.into())).unwrap()
        };
        storage
            .with_write_backend(|w| {
                for id in 1..=3u128 {
                    storage.upsert_node_be(
                        w,
                        &NodeUpsert {
                            id,
                            label: "point".into(),
                            properties: HashMap::from([(
                                "repo".to_string(),
                                Value::String("a".into()),
                            )]),
                        },
                    )?;
                }
                Ok(())
            })
            .unwrap();
        storage
            .create_payload_index("repo", schema.clone())
            .unwrap();
        // A write moved node 1 off "b" (recording it) and a chunk that had
        // read the old value put the row back.
        storage
            .with_write_backend(|w| {
                storage
                    .backend
                    .put_dup_stored(
                        w,
                        Namespace::PayloadIndex(&db_name),
                        &key("b"),
                        &1u128.to_be_bytes(),
                        PAYLOAD_INDEX_PLAIN_ROW,
                    )
                    .map_err(|e| GraphError::New(e.to_string()))?;
                storage.record_payload_index_build_delta(w, &db_name, 1, Some(&key("b")))
            })
            .unwrap();
        storage.payload_indices.write().unwrap().insert(
            "repo".to_string(),
            PayloadIndexHandle::building_lsm(schema.clone(), "job".to_string()),
        );
        let rows = |ns| {
            let read = storage.backend.begin_read().unwrap();
            let mut rows = Vec::new();
            storage
                .backend
                .scan_dup_stored(&read, ns, KeyRange::all(), |key, id, stored| {
                    rows.push((key.to_vec(), id.to_vec(), stored.to_vec()));
                    true
                })
                .unwrap();
            rows
        };
        let deltas = || rows(Namespace::PayloadIndexBuild(&db_name)).len();

        assert!(!storage
            .drain_payload_index_build_deltas("repo", &schema, &db_name, "older")
            .unwrap());
        assert_eq!(deltas(), 1, "a job that lost the index wrote");
        assert!(storage
            .drain_payload_index_build_deltas("repo", &schema, &db_name, "job")
            .unwrap());
        assert_eq!(deltas(), 0);
        let current: Vec<_> = (1..=3u128)
            .map(|id| {
                (
                    key("a"),
                    id.to_be_bytes().to_vec(),
                    PAYLOAD_INDEX_PLAIN_ROW.to_vec(),
                )
            })
            .collect();
        assert_eq!(rows(Namespace::PayloadIndex(&db_name)), current);
    }

    /// `upsert_node_be` and `drop_node_be` hold `payload_indices.read()`
    /// across their per-index loop, so the index/deindex calls inside it must
    /// not read-lock the map again: std's RwLock parks a new reader behind a
    /// queued writer (`delete_payload_index`), so a nested read deadlocks the
    /// write and the delete.
    #[test]
    fn lsm_payload_index_writes_do_not_relock_the_index_map() {
        let dir = TempDir::new().unwrap();
        let storage = std::sync::Arc::new(
            HelixGraphStorage::new(
                dir.path().to_str().unwrap(),
                test_config().with_lsm_in_memory(),
            )
            .unwrap(),
        );
        let schema = PayloadIndexSchema::Keyword;
        let db_name = HelixGraphStorage::payload_index_db_name("repo", &schema);
        let handle = PayloadIndexHandle::building_lsm(schema.clone(), "job".to_string());
        storage
            .payload_indices
            .write()
            .unwrap()
            .insert("repo".to_string(), handle.clone());

        // The loop's guard, then a writer queued behind it.
        let outer = storage.payload_indices.read().unwrap();
        let writer = {
            let storage = std::sync::Arc::clone(&storage);
            std::thread::spawn(move || drop(storage.payload_indices.write().unwrap()))
        };
        let queued = std::time::Instant::now();
        while storage.payload_indices.try_read().is_ok()
            && queued.elapsed() < std::time::Duration::from_secs(2)
        {
            std::thread::yield_now();
        }
        assert!(
            storage.payload_indices.try_read().is_err(),
            "writer never queued"
        );
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let index_write = {
            let storage = std::sync::Arc::clone(&storage);
            let handle = handle.clone();
            std::thread::spawn(move || {
                let value = Value::String("a".into());
                let result = storage.with_write_backend(|w| {
                    storage.index_node_payload_field_be(w, "repo", &handle, 1, Some(&value))?;
                    storage.deindex_node_payload_field_be(w, "repo", &handle, 1, Some(&value))
                });
                done_tx.send(result).unwrap();
            })
        };
        let finished = done_rx.recv_timeout(std::time::Duration::from_secs(5));
        drop(outer);
        writer.join().unwrap();
        index_write.join().unwrap();
        assert!(
            matches!(finished, Ok(Ok(()))),
            "index write under a held map guard: {finished:?}"
        );

        let deltas = || {
            let read = storage.backend.begin_read().unwrap();
            let mut deltas = 0;
            storage
                .backend
                .scan_dup_stored(
                    &read,
                    Namespace::PayloadIndexBuild(&db_name),
                    KeyRange::all(),
                    |_, _, _| {
                        deltas += 1;
                        true
                    },
                )
                .unwrap();
            deltas
        };
        // Still recorded while building: one marker, one deleted row.
        assert_eq!(deltas(), 2);
        // And not once the index is Ready.
        assert!(handle.mark_ready("job"));
        storage
            .with_write_backend(|w| {
                let value = Value::String("b".into());
                storage.index_node_payload_field_be(w, "repo", &handle, 2, Some(&value))
            })
            .unwrap();
        assert_eq!(deltas(), 2);
    }

    #[test]
    #[serial]
    fn sync_payload_index_build_catches_concurrent_writes_between_chunks() {
        use crate::helix_engine::storage_core::upsert::NodeUpsert;

        struct EnvRestore {
            key: &'static str,
            previous: Option<String>,
        }

        impl Drop for EnvRestore {
            fn drop(&mut self) {
                unsafe {
                    match &self.previous {
                        Some(value) => std::env::set_var(self.key, value),
                        None => std::env::remove_var(self.key),
                    }
                }
            }
        }

        let chunk_key = "HELIX_PAYLOAD_INDEX_CHUNK_SIZE";
        let pause_key = "HELIX_PAYLOAD_INDEX_CHUNK_PAUSE_MS";
        let _chunk_restore = EnvRestore {
            key: chunk_key,
            previous: std::env::var(chunk_key).ok(),
        };
        let _pause_restore = EnvRestore {
            key: pause_key,
            previous: std::env::var(pause_key).ok(),
        };
        unsafe {
            std::env::set_var(chunk_key, "1");
            std::env::set_var(pause_key, "20");
        }

        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = std::sync::Arc::new(HelixGraphStorage::new(&path, test_config()).unwrap());
        let field = "repo";

        storage
            .with_write_txn(|txn| {
                for id in 1_u128..=32 {
                    storage.upsert_node(
                        txn,
                        &NodeUpsert {
                            id,
                            label: "Symbol".into(),
                            properties: HashMap::from([(
                                field.into(),
                                Value::String("old".into()),
                            )]),
                        },
                    )?;
                }
                Ok(())
            })
            .unwrap();

        let builder_storage = std::sync::Arc::clone(&storage);
        let builder = std::thread::spawn(move || {
            builder_storage
                .create_payload_index(field, PayloadIndexSchema::Keyword)
                .unwrap();
        });

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let status = storage.payload_index_status(field).unwrap();
            if status
                .as_ref()
                .is_some_and(|status| status.status == "building" && status.indexed_nodes >= 1)
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "payload index did not enter building state before test deadline"
            );
            std::thread::sleep(Duration::from_millis(1));
        }

        assert_eq!(storage.has_payload_index(field), None);
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        assert!(storage
            .get_nodes_by_payload_value(&txn, field, &Value::String("old".into()))
            .is_err());
        drop(txn);

        storage
            .with_write_txn(|txn| {
                storage.upsert_node(
                    txn,
                    &NodeUpsert {
                        id: 1,
                        label: "Symbol".into(),
                        properties: HashMap::from([(field.into(), Value::String("new".into()))]),
                    },
                )?;
                Ok(())
            })
            .unwrap();

        builder.join().unwrap();
        assert_eq!(
            storage.has_payload_index(field),
            Some(PayloadIndexSchema::Keyword)
        );

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let old_hits = storage
            .get_nodes_by_payload_value(&txn, field, &Value::String("old".into()))
            .unwrap();
        let new_hits = storage
            .get_nodes_by_payload_value(&txn, field, &Value::String("new".into()))
            .unwrap();
        assert!(!old_hits.contains(&1));
        assert_eq!(new_hits, vec![1]);
    }

    /// Fix B: `ensure_map_headroom` grows the map when the requested
    /// headroom exceeds the currently-available map bytes, and is a no-op
    /// on subsequent small requests that already fit.
    #[test]
    fn ensure_map_headroom_grows_and_is_idempotent() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = HelixGraphStorage::new(&path, test_config()).unwrap();

        let info = storage.lmdb_env().unwrap().info();
        let initial = info.map_size;
        let cap = HelixGraphStorage::max_map_size();
        let page_sz = page_size::get();
        let used = info.last_page_number.saturating_mul(page_sz);
        let available = initial.saturating_sub(used);

        // Request more than the currently available headroom → must grow.
        let want = available.saturating_add(128 * 1024 * 1024);
        storage.ensure_map_headroom(want).unwrap();
        let after_first = storage.lmdb_env().unwrap().info().map_size;
        if initial < cap {
            assert!(
                after_first > initial,
                "expected growth: initial={}, after_first={}",
                initial,
                after_first
            );
            let new_avail = storage
                .lmdb_env()
                .unwrap()
                .info()
                .map_size
                .saturating_sub(used);
            assert!(
                new_avail >= want,
                "post-grow headroom {} below request {}",
                new_avail,
                want
            );
        } else {
            assert_eq!(after_first, initial, "map already opened at cap");
        }

        // Small follow-up request already fits → no-op.
        storage.ensure_map_headroom(1024).unwrap();
        let after_second = storage.lmdb_env().unwrap().info().map_size;
        assert_eq!(
            after_first, after_second,
            "second call should not resize when headroom already sufficient"
        );
    }

    #[test]
    #[serial]
    fn ensure_map_headroom_does_not_grow_empty_map_at_exact_margin_boundary() {
        struct EnvRestore {
            key: &'static str,
            previous: Option<String>,
        }

        impl Drop for EnvRestore {
            fn drop(&mut self) {
                match &self.previous {
                    Some(value) => std::env::set_var(self.key, value),
                    None => std::env::remove_var(self.key),
                }
            }
        }

        let _initial_restore = EnvRestore {
            key: "HELIX_INITIAL_MAP_MB",
            previous: std::env::var("HELIX_INITIAL_MAP_MB").ok(),
        };
        let _margin_restore = EnvRestore {
            key: "HELIX_PRE_GROW_MB",
            previous: std::env::var("HELIX_PRE_GROW_MB").ok(),
        };
        std::env::set_var("HELIX_INITIAL_MAP_MB", "128");
        std::env::set_var("HELIX_PRE_GROW_MB", "64");

        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = HelixGraphStorage::new(&path, test_config()).unwrap();
        let initial = storage.lmdb_env().unwrap().info().map_size;

        storage.ensure_map_headroom(64 * 1024 * 1024).unwrap();

        assert_eq!(
            storage.lmdb_env().unwrap().info().map_size,
            initial,
            "fresh map should not pre-grow when initial size equals request plus margin"
        );
    }

    #[test]
    #[serial]
    fn giant_ensure_map_headroom_defers_without_pending_writer_when_gate_busy_with_runway() {
        let _env = force_giant_resize_env("1");
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = HelixGraphStorage::new(&path, test_config()).unwrap();
        let initial = storage.lmdb_env().unwrap().info().map_size;

        let _reader = storage.resize_gate.read();
        storage.ensure_map_headroom(1024).unwrap();

        assert_eq!(
            storage.resize_writers_pending.load(Ordering::Acquire),
            0,
            "giant deferred resize must not queue a pending writer"
        );
        assert_eq!(
            storage.resize_writers_active.load(Ordering::Acquire),
            0,
            "try-write miss must not mark resize active"
        );
        assert_eq!(
            storage.lmdb_env().unwrap().info().map_size,
            initial,
            "busy try-write path should defer instead of resizing"
        );
    }

    #[test]
    #[serial]
    fn giant_ensure_map_headroom_falls_through_to_blocking_when_gate_busy() {
        let _env = force_giant_resize_env("512");
        // Disable the opportunistic poll so this test still exercises the
        // blocking escalation path deterministically.
        let _poll = TestEnvRestore::set("HELIX_RESIZE_TRY_POLL_MS", "0");
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = std::sync::Arc::new(HelixGraphStorage::new(&path, test_config()).unwrap());

        // Hold the write gate so the try-lock fails and we fall through to
        // the blocking path.
        let _gate = storage.resize_gate.write();

        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let resize_storage = std::sync::Arc::clone(&storage);
        let t = std::thread::spawn(move || {
            let result = resize_storage.ensure_map_headroom(1024);
            done_tx.send(result).unwrap();
        });

        // The blocking path should be waiting on the gate.
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(
            !done_rx
                .recv_timeout(std::time::Duration::from_secs(1))
                .is_ok(),
            "blocking path should still be waiting on the gate"
        );

        // Release the gate — the blocking path should proceed and resize.
        drop(_gate);
        let result = done_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("blocking path should have completed the resize");
        result.unwrap();

        let after = storage.lmdb_env().unwrap().info().map_size;
        assert!(
            after >= 512 * 1024 * 1024,
            "expected growth after blocking resize: after={after}"
        );
        assert!(t.join().is_ok());
    }

    #[test]
    #[serial]
    fn giant_ensure_map_headroom_polls_gate_without_parking_readers() {
        let _env = force_giant_resize_env("512");
        let _poll = TestEnvRestore::set("HELIX_RESIZE_TRY_POLL_MS", "5000");
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = std::sync::Arc::new(HelixGraphStorage::new(&path, test_config()).unwrap());
        let initial = storage.lmdb_env().unwrap().info().map_size;

        // Simulate a long-running read holding the shared gate.
        let reader = storage.resize_gate.read();

        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let resize_storage = std::sync::Arc::clone(&storage);
        let t = std::thread::spawn(move || {
            done_tx
                .send(resize_storage.ensure_map_headroom(1024))
                .unwrap();
        });

        // While the pre-grow polls it must not queue a pending writer (which
        // would park every new reader), and new readers must still get in.
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert!(
            done_rx.try_recv().is_err(),
            "pre-grow should still be polling while the reader holds the gate"
        );
        assert_eq!(
            storage.resize_writers_pending.load(Ordering::Acquire),
            0,
            "polling pre-grow must not park readers behind a pending writer"
        );
        assert!(
            storage.resize_gate.try_read().is_some(),
            "new readers must acquire the gate while the pre-grow polls"
        );

        drop(reader);
        let result = done_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("poll should acquire the gate after the reader releases");
        result.unwrap();
        assert!(
            storage.lmdb_env().unwrap().info().map_size > initial,
            "expected growth via polled acquisition"
        );
        assert_eq!(storage.resize_writers_pending.load(Ordering::Acquire), 0);
        assert_eq!(storage.resize_writers_active.load(Ordering::Acquire), 0);
        assert!(t.join().is_ok());
    }

    #[test]
    #[serial]
    fn giant_ensure_map_headroom_escalates_to_blocking_after_poll_deadline() {
        let _env = force_giant_resize_env("512");
        let _poll = TestEnvRestore::set("HELIX_RESIZE_TRY_POLL_MS", "100");
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = std::sync::Arc::new(HelixGraphStorage::new(&path, test_config()).unwrap());

        let reader = storage.resize_gate.read();

        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let resize_storage = std::sync::Arc::clone(&storage);
        let t = std::thread::spawn(move || {
            done_tx
                .send(resize_storage.ensure_map_headroom(1024))
                .unwrap();
        });

        // After the deadline expires the pre-grow must queue as a blocking
        // writer (reader parking resumes — bounded politeness, not starvation).
        let queued = std::time::Instant::now();
        while storage.resize_writers_pending.load(Ordering::Acquire) == 0
            && queued.elapsed() < std::time::Duration::from_secs(2)
        {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(
            storage.resize_writers_pending.load(Ordering::Acquire),
            1,
            "expired poll must escalate to a queued blocking writer"
        );
        assert!(done_rx.try_recv().is_err());

        drop(reader);
        let result = done_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("blocking escalation should complete after the reader releases");
        result.unwrap();
        assert!(t.join().is_ok());
    }

    #[test]
    #[serial]
    fn giant_ensure_map_headroom_resizes_with_large_step_when_gate_free() {
        let _env = force_giant_resize_env("1");
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = HelixGraphStorage::new(&path, test_config()).unwrap();
        let initial = storage.lmdb_env().unwrap().info().map_size;

        storage.ensure_map_headroom(1024).unwrap();

        let after = storage.lmdb_env().unwrap().info().map_size;
        assert!(
            after >= initial.saturating_add(256 * 1024 * 1024),
            "expected giant step growth: initial={initial}, after={after}"
        );
        assert_eq!(storage.resize_writers_pending.load(Ordering::Acquire), 0);
        assert_eq!(storage.resize_writers_active.load(Ordering::Acquire), 0);
    }

    #[test]
    fn map_headroom_satisfied_allows_bounded_fresh_map_overhead() {
        let mib = 1024 * 1024;
        let current = 128 * mib;
        let additional = 64 * mib;
        let margin = 64 * mib;
        let used = 10 * mib;
        let available = current - used;

        assert!(
            HelixGraphStorage::map_headroom_satisfied(current, used, available, additional, margin),
            "fresh 10MiB LMDB overhead should be absorbed by the pre-grow margin"
        );

        let used = 20 * mib;
        let available = current - used;
        assert!(
            !HelixGraphStorage::map_headroom_satisfied(
                current, used, available, additional, margin
            ),
            "larger used maps should still pre-grow to preserve margin"
        );
    }

    /// Fix B: `ensure_map_headroom` is capped at the per-env max and does
    /// not return an error when the cap is reached.
    #[test]
    fn ensure_map_headroom_caps_at_max() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = HelixGraphStorage::new(&path, test_config()).unwrap();

        let cap = HelixGraphStorage::max_map_size();
        // Ask for far more than the cap; call should still succeed.
        let absurd = cap.saturating_mul(4);
        storage.ensure_map_headroom(absurd).unwrap();
        let after = storage.lmdb_env().unwrap().info().map_size;
        assert!(
            after <= cap,
            "map_size {} must not exceed cap {}",
            after,
            cap
        );
    }

    /// Fix B: `ensure_map_headroom` respects the `HELIX_PRE_GROW_MB` margin.
    /// It should grow the map if (available < additional_bytes + margin),
    /// even if (available >= additional_bytes).
    #[test]
    #[serial]
    fn ensure_map_headroom_respects_pre_grow_margin() {
        // Set a 256MB margin for the test.
        std::env::set_var("HELIX_PRE_GROW_MB", "256");

        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = HelixGraphStorage::new(&path, test_config()).unwrap();

        let info = storage.lmdb_env().unwrap().info();
        let initial = info.map_size;
        let page_sz = page_size::get();
        let used = info.last_page_number.saturating_mul(page_sz);
        let available = initial.saturating_sub(used);

        // Request exactly what is available.
        // Without margin, this would be a no-op.
        // With a 256MB margin, this must trigger growth.
        storage.ensure_map_headroom(available).unwrap();

        let after = storage.lmdb_env().unwrap().info().map_size;
        let cap = HelixGraphStorage::max_map_size();

        if initial < cap {
            assert!(
                after > initial,
                "expected growth due to margin: initial={}, after={}",
                initial,
                after
            );
        }

        std::env::remove_var("HELIX_PRE_GROW_MB");
    }

    #[test]
    fn payload_index_accepts_oversized_keyword_values() {
        use crate::helix_engine::storage_core::upsert::NodeUpsert;

        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = HelixGraphStorage::new(&path, test_config()).unwrap();
        storage
            .create_payload_index("path", PayloadIndexSchema::Keyword)
            .unwrap();

        let long_path = format!("src/{}", "very-long-component/".repeat(128));
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        storage
            .upsert_node(
                &mut txn,
                &NodeUpsert {
                    id: 42,
                    label: "Symbol".to_string(),
                    properties: HashMap::from([(
                        "path".to_string(),
                        Value::String(long_path.clone()),
                    )]),
                },
            )
            .unwrap();
        txn.commit().unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let ids = storage
            .get_nodes_by_payload_value(&txn, "path", &Value::String(long_path))
            .unwrap();
        assert_eq!(ids, vec![42]);
    }

    #[test]
    fn edge_path_index_accepts_oversized_paths() {
        use crate::helix_engine::storage_core::upsert::{EdgeUpsert, NodeUpsert};

        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = HelixGraphStorage::new(&path, test_config()).unwrap();
        let long_path = format!("src/{}", "nested-module/".repeat(128));

        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        storage
            .upsert_node(
                &mut txn,
                &NodeUpsert {
                    id: 1,
                    label: "Symbol".to_string(),
                    properties: HashMap::from([(
                        "path".to_string(),
                        Value::String(long_path.clone()),
                    )]),
                },
            )
            .unwrap();
        storage
            .upsert_node(
                &mut txn,
                &NodeUpsert {
                    id: 2,
                    label: "Symbol".to_string(),
                    properties: HashMap::from([(
                        "path".to_string(),
                        Value::String("src/callee.rs".to_string()),
                    )]),
                },
            )
            .unwrap();
        storage
            .upsert_edge(
                &mut txn,
                &EdgeUpsert {
                    id: 99,
                    label: "CALLS".to_string(),
                    from_node: 1,
                    to_node: 2,
                    properties: HashMap::from([
                        ("from_path".to_string(), Value::String(long_path.clone())),
                        (
                            "to_path".to_string(),
                            Value::String("src/callee.rs".to_string()),
                        ),
                    ]),
                },
            )
            .unwrap();
        txn.commit().unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        assert_eq!(
            storage.edge_ids_for_path(&txn, &long_path, 10).unwrap(),
            vec![99]
        );
    }
}

impl HelixGraphStorage {
    pub(crate) fn payload_value_for_key<'a>(
        properties: &'a HashMap<String, Value>,
        key: &str,
    ) -> Option<&'a Value> {
        let parts: Vec<&str> = key.split('.').collect();
        if parts.is_empty() {
            return None;
        }

        let mut current = properties.get(parts[0])?;
        for &part in &parts[1..] {
            match current {
                Value::Object(map) => {
                    current = map.get(part)?;
                }
                _ => return None,
            }
        }
        Some(current)
    }

    /// Create a multi-value (1:N) index. Uses DUP_SORT | DUP_FIXED so one key maps to many 16-byte node IDs.
    pub fn create_multi_index(&mut self, name: &str) -> Result<(), GraphError> {
        let idx_name = format!("midx_{}", name);
        let db = self.with_write_txn(|wtxn| {
            let db = self
                .lmdb_env()?
                .database_options()
                .types::<Bytes, Bytes>()
                .flags(DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED)
                .name(&idx_name)
                .create(wtxn)?;
            Ok(db)
        })?;
        self.multi_indices.insert(name.to_string(), Some(db));
        Ok(())
    }

    /// Register (insert) one payload index in storage metadata. Routed: heed RMW
    /// on LMDB, backend RMW (`update_metadata_be`) on LSM. The heed `put_metadata`
    /// uses delete_heed/put_heed which hit the `unreachable!("heed write path is
    /// LMDB-only")` arm under LSM — and because the payload-index BUILD runs on a
    /// background executor thread, that panic kills the executor thread AND poisons
    /// the `payload_index_admin_gate` mutex, bricking ALL subsequent index builds
    /// with "sending on a closed channel". Every payload-index metadata write MUST
    /// go through here. Used by create_payload_index (sync) and
    /// publish_payload_index_ready (async executor completion).
    fn register_payload_index_metadata(
        &self,
        name: &str,
        schema: &PayloadIndexSchema,
    ) -> Result<(), GraphError> {
        if self.backend.kind() == BackendKind::Lsm {
            return self.with_write_backend(|w| {
                self.update_metadata_be(w, |metadata| {
                    let mut indices = metadata.payload_indices.clone();
                    indices.insert(name.to_string(), schema.clone());
                    metadata.set_payload_indices(indices);
                    Ok(())
                })?;
                if *schema == PayloadIndexSchema::Keyword {
                    // Built by this binary: its rows already carry markers.
                    self.put_payload_marker_backfill_done(w, name)?;
                }
                Ok(())
            });
        }
        self.with_write_txn(|wtxn| {
            let mut indices = self.get_metadata(wtxn)?.payload_indices;
            indices.insert(name.to_string(), schema.clone());
            self.set_payload_indices_metadata(wtxn, indices)?;
            Ok(())
        })
    }

    /// [`Self::register_payload_index_metadata`] on LSM as ONE write that is
    /// flushed here: the metadata key and the marker record go into the same
    /// transaction, it commits buffered, and one flush makes it durable
    /// before this returns. The two durable commits of the plain form each
    /// wait for the next WAL flush tick (500 ms apart in production) instead,
    /// which made every index create on an empty collection take a second.
    fn register_payload_index_metadata_in_one_flush(
        &self,
        name: &str,
        schema: &PayloadIndexSchema,
    ) -> Result<(), GraphError> {
        self.ensure_not_degraded()?;
        // Built by this binary: its rows already carry markers.
        let marker = (*schema == PayloadIndexSchema::Keyword)
            .then(|| payload_marker_backfill::marker_backfill_done_row(name));
        let also: Vec<(&[u8], &[u8])> = marker
            .iter()
            .map(|(key, done)| (key.as_slice(), *done))
            .collect();
        self.update_metadata_be_buffered(&also, |metadata| {
            let mut indices = metadata.payload_indices.clone();
            indices.insert(name.to_string(), schema.clone());
            metadata.set_payload_indices(indices);
            Ok(())
        })?;
        self.backend
            .flush_durable()
            .map_err(|e| self.backend_error(e, "payload_index_register_flush_durable"))?;
        // After the flush: the snapshot is written out as the metadata sidecar, which
        // must not name an index the store does not hold yet.
        self.refresh_metadata_snapshot_best_effort();
        Ok(())
    }

    pub fn create_payload_index(
        &self,
        name: &str,
        schema: PayloadIndexSchema,
    ) -> Result<(), GraphError> {
        let _admin_guard = self
            .payload_index_admin_gate
            .lock()
            .map_err(|e| GraphError::New(format!("Payload index admin gate poisoned: {}", e)))?;
        {
            let payload_indices = self
                .payload_indices
                .read()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            if let Some(existing) = payload_indices.get(name) {
                if existing.schema == schema && existing.is_ready() {
                    return Ok(());
                }
                if existing.schema == schema {
                    return Err(GraphError::New(format!(
                        "Payload index '{}' is currently {}",
                        name,
                        existing.status(name).status
                    )));
                }
                return Err(GraphError::New(format!(
                    "Payload index '{}' already exists with schema {:?}",
                    name, existing.schema
                )));
            }
        }

        if self.backend.kind() == BackendKind::Lsm {
            let job_id = format!(
                "pidx-sync-{}-{}",
                chrono::Utc::now().timestamp_millis(),
                PAYLOAD_INDEX_JOB_SEQ.fetch_add(1, Ordering::Relaxed)
            );
            let handle = PayloadIndexHandle::building_lsm(schema.clone(), job_id.clone());
            {
                let mut payload_indices = self
                    .payload_indices
                    .write()
                    .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
                payload_indices.insert(name.to_string(), handle.clone());
            }

            let build_result = (|| -> Result<(), GraphError> {
                let db_name = Self::payload_index_db_name(name, &schema);
                // A failed prior job may have left build-delta rows behind;
                // this build's drain must only see its own deltas.
                self.clear_payload_index_build_deltas(&db_name)?;
                self.clear_payload_index_rows(&db_name)?;
                let mut cursor: Option<u128> = None;
                let mut indexed_nodes = 0_u64;
                let chunk_size = payload_index_chunk_size();
                let pause = payload_index_chunk_pause();
                loop {
                    let ids = self.next_payload_index_node_ids(cursor, chunk_size)?;
                    if ids.is_empty() {
                        break;
                    }

                    let indexed = self.index_payload_chunk(&ids, name, &schema, None)?;

                    cursor = ids.last().copied();
                    indexed_nodes = indexed_nodes.saturating_add(indexed);
                    self.update_payload_index_progress(name, &job_id, cursor, indexed_nodes)?;

                    if !pause.is_zero() {
                        std::thread::sleep(pause);
                    }
                }

                // The backfill scan raced foreground writes (LSM write batches
                // take no shared gate): repair what the build-delta sidecar
                // recorded so no chunk-resurrected stale row survives
                // publication.
                if !self.drain_payload_index_build_deltas(name, &schema, &db_name, &job_id)? {
                    return Err(GraphError::New(format!(
                        "payload index '{name}' build was cancelled"
                    )));
                }

                self.register_payload_index_metadata(name, &schema)?;
                handle.mark_ready(&job_id);
                // Foreground writes racing the Ready flip maintained the live
                // index themselves; their leftover delta rows are garbage.
                self.clear_payload_index_build_deltas(&db_name)?;
                Ok(())
            })();

            if let Err(e) = build_result {
                handle.mark_failed(&job_id, e.to_string());
                if let Ok(mut payload_indices) = self.payload_indices.write() {
                    if payload_indices
                        .get(name)
                        .map(|existing| existing.status(name).job_id.as_deref() == Some(&job_id))
                        .unwrap_or(false)
                    {
                        payload_indices.remove(name);
                    }
                }
                return Err(e);
            }

            return Ok(());
        }

        // Create the index database (single small transaction). Clear it so a
        // retry after a crash/failed attempt cannot publish stale partial data.
        let db = self.with_write_txn(|wtxn| {
            let db = self
                .lmdb_env()?
                .database_options()
                .types::<Bytes, Bytes>()
                .flags(DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED)
                .name(&Self::payload_index_db_name(name, &schema))
                .create(wtxn)?;
            db.clear(wtxn)?;
            Ok(db)
        })?;

        let job_id = format!(
            "pidx-sync-{}-{}",
            chrono::Utc::now().timestamp_millis(),
            PAYLOAD_INDEX_JOB_SEQ.fetch_add(1, Ordering::Relaxed)
        );
        let handle = PayloadIndexHandle::building(schema.clone(), db, job_id.clone());
        {
            let mut payload_indices = self
                .payload_indices
                .write()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            payload_indices.insert(name.to_string(), handle.clone());
        }

        // Index nodes in chunks — each chunk gets its own write transaction,
        // releasing write_txn_gate between batches.
        let build_result = (|| -> Result<(), GraphError> {
            let mut cursor: Option<u128> = None;
            let mut indexed_nodes = 0_u64;
            let chunk_size = payload_index_chunk_size();
            let pause = payload_index_chunk_pause();
            loop {
                let ids = self.next_payload_index_node_ids(cursor, chunk_size)?;
                if ids.is_empty() {
                    break;
                }

                let indexed = self.index_payload_chunk(&ids, name, &schema, Some(&db))?;

                cursor = ids.last().copied();
                indexed_nodes = indexed_nodes.saturating_add(indexed);
                self.update_payload_index_progress(name, &job_id, cursor, indexed_nodes)?;

                if !pause.is_zero() {
                    std::thread::sleep(pause);
                }
            }

            // Register metadata now that indexing is complete (routed for LSM).
            self.register_payload_index_metadata(name, &schema)?;

            handle.mark_ready(&job_id);
            Ok(())
        })();

        if let Err(e) = build_result {
            handle.mark_failed(&job_id, e.to_string());
            if let Ok(mut payload_indices) = self.payload_indices.write() {
                if payload_indices
                    .get(name)
                    .map(|existing| existing.status(name).job_id.as_deref() == Some(&job_id))
                    .unwrap_or(false)
                {
                    payload_indices.remove(name);
                }
            }
            return Err(e);
        }

        Ok(())
    }

    /// `name`'s status when it is already ready or building with `schema`;
    /// an error when it exists with another schema.
    fn ready_or_building_payload_index(
        &self,
        name: &str,
        schema: &PayloadIndexSchema,
    ) -> Result<Option<PayloadIndexStatus>, GraphError> {
        let payload_indices = self
            .payload_indices
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        let Some(existing) = payload_indices.get(name) else {
            return Ok(None);
        };
        if existing.schema != *schema {
            return Err(GraphError::New(format!(
                "Payload index '{}' already exists with schema {:?}",
                name, existing.schema
            )));
        }
        let status = existing.status(name);
        Ok(matches!(status.status, "ready" | "building").then_some(status))
    }

    /// Whether a payload index build of this collection is queued or
    /// running. A queued job has only a `Weak` to the storage, so that it
    /// pins no LMDB env across a drop: an eviction that closed the collection
    /// under it dropped the build its caller was told is `building`, and the
    /// next open knew no such index.
    pub(crate) fn payload_index_build_pending(&self) -> bool {
        self.payload_index_builds_pending.load(Ordering::Acquire) > 0
    }

    /// Counts a build as pending from before `submit` hands it to the
    /// executor, since a worker may take it at once. The worker that takes
    /// it gives the count back by a guard (`PendingPayloadIndexBuild`). A
    /// job that `submit` refuses was never queued: its count is given back
    /// here.
    fn counted_as_pending(
        &self,
        submit: impl FnOnce() -> Result<(), GraphError>,
    ) -> Result<(), GraphError> {
        self.payload_index_builds_pending
            .fetch_add(1, Ordering::AcqRel);
        let queued = submit();
        if queued.is_err() {
            self.payload_index_builds_pending
                .fetch_sub(1, Ordering::AcqRel);
        }
        queued
    }

    pub fn enqueue_payload_index_build(
        self: &Arc<Self>,
        collection: &str,
        name: &str,
        schema: PayloadIndexSchema,
        point_writes_quiesced: bool,
    ) -> Result<PayloadIndexStatus, GraphError> {
        // Re-sent creates for an existing index (the client sends them) answer without
        // the admin gate: an async build holds it while clearing a dropped
        // index's rows, and the caller holds the point-mutation gate.
        if let Some(status) = self.ready_or_building_payload_index(name, &schema)? {
            return Ok(status);
        }
        let _admin_guard = self
            .payload_index_admin_gate
            .lock()
            .map_err(|e| GraphError::New(format!("Payload index admin gate poisoned: {}", e)))?;

        if let Some(status) = self.ready_or_building_payload_index(name, &schema)? {
            return Ok(status);
        }

        let job_id = format!(
            "pidx-{}-{}",
            chrono::Utc::now().timestamp_millis(),
            PAYLOAD_INDEX_JOB_SEQ.fetch_add(1, Ordering::Relaxed)
        );
        if self.backend.kind() == BackendKind::Lsm {
            if point_writes_quiesced {
                let r = self
                    .backend
                    .begin_read()
                    .map_err(|e| GraphError::New(e.to_string()))?;
                let metadata = self
                    .get_metadata_be(&r)
                    .or_else(|_| self.metadata_snapshot())?;
                if metadata.stats.node_count == 0 {
                    drop(r);
                    self.clear_payload_index_rows(&Self::payload_index_db_name(name, &schema))?;
                    let handle = PayloadIndexHandle::ready_lsm(schema.clone());
                    let replaced = {
                        let mut payload_indices = self
                            .payload_indices
                            .write()
                            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
                        payload_indices.insert(name.to_string(), handle.clone())
                    };
                    let registered = if super::replication::lsm_point_write_flush_enabled() {
                        self.register_payload_index_metadata_in_one_flush(name, &schema)
                    } else {
                        self.register_payload_index_metadata(name, &schema)
                    };
                    if let Err(e) = registered {
                        // Not registered: a retry has to write it, so it must not be
                        // answered "ready" from this handle. What it replaced comes back.
                        let mut payload_indices =
                            self.payload_indices.write().map_err(|lock_err| {
                                GraphError::New(format!("Lock poisoned: {}", lock_err))
                            })?;
                        match replaced {
                            Some(replaced) => payload_indices.insert(name.to_string(), replaced),
                            None => payload_indices.remove(name),
                        };
                        return Err(e);
                    }
                    metrics::counter!("helix_payload_index_empty_fast_path_total").increment(1);
                    return Ok(handle.status(name));
                }
            }

            let handle = PayloadIndexHandle::building_lsm(schema.clone(), job_id.clone());
            {
                let mut payload_indices = self
                    .payload_indices
                    .write()
                    .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
                payload_indices.insert(name.to_string(), handle.clone());
            }

            let job = PayloadIndexJob {
                collection: collection.to_string(),
                field_name: name.to_string(),
                schema,
                db: None,
                job_id: job_id.clone(),
                storage: Arc::downgrade(self),
            };
            if let Err(e) = self.counted_as_pending(|| PAYLOAD_INDEX_EXECUTOR.submit(job)) {
                let mut payload_indices = self
                    .payload_indices
                    .write()
                    .map_err(|lock_err| GraphError::New(format!("Lock poisoned: {}", lock_err)))?;
                if payload_indices
                    .get(name)
                    .map(|existing| existing.is_building_job(&job_id))
                    .unwrap_or(false)
                {
                    payload_indices.remove(name);
                }
                return Err(e);
            }

            return Ok(handle.status(name));
        }
        let (db, ready_without_backfill) = self.with_write_txn(|wtxn| {
            let db = self
                .lmdb_env()?
                .database_options()
                .types::<Bytes, Bytes>()
                .flags(DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED)
                .name(&Self::payload_index_db_name(name, &schema))
                .create(wtxn)?;
            // Metadata is published only after the background pass completes.
            // Clear any abandoned partial DB from a previous crash/retry first.
            db.clear(wtxn)?;
            // Empty-collection fast path writes metadata via heed here, which
            // panics under LSM. On LSM skip it and let the executor handle the
            // (empty) build + publish metadata through the routed path.
            if point_writes_quiesced && self.backend.kind() != BackendKind::Lsm {
                let metadata = self.get_metadata(wtxn)?;
                if metadata.stats.node_count == 0 {
                    let mut payload_index_metadata = metadata.payload_indices;
                    payload_index_metadata.insert(name.to_string(), schema.clone());
                    self.set_payload_indices_metadata(wtxn, payload_index_metadata)?;
                    return Ok((db, true));
                }
            }
            Ok((db, false))
        })?;

        if ready_without_backfill {
            let handle = PayloadIndexHandle::ready(schema.clone(), db);
            {
                let mut payload_indices = self
                    .payload_indices
                    .write()
                    .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
                payload_indices.insert(name.to_string(), handle.clone());
            }
            metrics::counter!("helix_payload_index_empty_fast_path_total").increment(1);
            return Ok(handle.status(name));
        }

        let handle = PayloadIndexHandle::building(schema.clone(), db, job_id.clone());
        {
            let mut payload_indices = self
                .payload_indices
                .write()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            payload_indices.insert(name.to_string(), handle.clone());
        }

        let job = PayloadIndexJob {
            collection: collection.to_string(),
            field_name: name.to_string(),
            schema,
            db: Some(db),
            job_id: job_id.clone(),
            storage: Arc::downgrade(self),
        };
        if let Err(e) = self.counted_as_pending(|| PAYLOAD_INDEX_EXECUTOR.submit(job)) {
            let mut payload_indices = self
                .payload_indices
                .write()
                .map_err(|lock_err| GraphError::New(format!("Lock poisoned: {}", lock_err)))?;
            if payload_indices
                .get(name)
                .map(|existing| existing.is_building_job(&job_id))
                .unwrap_or(false)
            {
                payload_indices.remove(name);
            }
            return Err(e);
        }

        Ok(handle.status(name))
    }

    fn run_payload_index_job(self: Arc<Self>, job: PayloadIndexJob) {
        if let Err(e) = self.build_payload_index_job(&job) {
            self.mark_payload_index_failed(&job.field_name, &job.job_id, e.to_string());
            tracing::error!(
                collection = %job.collection,
                field = %job.field_name,
                job_id = %job.job_id,
                error = %e,
                "payload index background build failed"
            );
        }
    }

    fn build_payload_index_job(&self, job: &PayloadIndexJob) -> Result<(), GraphError> {
        let chunk_size = payload_index_chunk_size();
        let pause = payload_index_chunk_pause();
        let mut cursor = None;
        let mut indexed_nodes = 0_u64;
        let lsm = self.backend.kind() == BackendKind::Lsm;
        let db_name = Self::payload_index_db_name(&job.field_name, &job.schema);
        if lsm {
            // Mirror the sync LSM build. Under the admin gate, so a job that is
            // no longer current never clears a newer build's rows.
            let _admin_guard = self.payload_index_admin_gate.lock().map_err(|e| {
                GraphError::New(format!("Payload index admin gate poisoned: {}", e))
            })?;
            if !self.payload_index_job_active(&job.field_name, &job.job_id)? {
                return Ok(());
            }
            self.clear_payload_index_build_deltas(&db_name)?;
            self.clear_payload_index_rows(&db_name)?;
        }

        loop {
            if !self.payload_index_job_active(&job.field_name, &job.job_id)? {
                return Ok(());
            }

            let ids = self.next_payload_index_node_ids(cursor, chunk_size)?;
            if ids.is_empty() {
                break;
            }
            let next_cursor = ids.last().copied();
            let indexed = {
                let _build = self.payload_index_build_lock()?;
                if !self.payload_index_job_active(&job.field_name, &job.job_id)? {
                    return Ok(());
                }
                self.write_payload_index_chunk(job, &ids)?
            };

            indexed_nodes = indexed_nodes.saturating_add(indexed);
            self.update_payload_index_progress(
                &job.field_name,
                &job.job_id,
                next_cursor,
                indexed_nodes,
            )?;
            cursor = next_cursor;

            if !pause.is_zero() {
                std::thread::sleep(pause);
            }
        }

        // Backfill chunks raced foreground writes: repair any row a chunk
        // resurrected before the index is published Ready.
        if lsm
            && !self.drain_payload_index_build_deltas(
                &job.field_name,
                &job.schema,
                &db_name,
                &job.job_id,
            )?
        {
            return Ok(());
        }
        #[cfg(test)]
        if let Some(hook) =
            BEFORE_PAYLOAD_INDEX_PUBLISH_FOR_TEST.with(|hook| hook.borrow_mut().take())
        {
            hook();
        }
        self.publish_payload_index_ready(job)
    }

    fn next_payload_index_node_ids(
        &self,
        cursor: Option<u128>,
        limit: usize,
    ) -> Result<Vec<u128>, GraphError> {
        let start = cursor
            .map(|id| Bound::Excluded(id.to_be_bytes().to_vec()))
            .unwrap_or(Bound::Unbounded);
        let range = KeyRange {
            start,
            end: Bound::Unbounded,
        };

        if self.backend.kind() == BackendKind::Lsm {
            let r = self
                .backend
                .begin_read()
                .map_err(|e| GraphError::New(e.to_string()))?;
            let mut ids = Vec::with_capacity(limit);
            self.backend
                .scan(&r, Namespace::Nodes, range, |k, _v| {
                    let mut bytes = [0u8; 16];
                    bytes.copy_from_slice(k);
                    ids.push(u128::from_be_bytes(bytes));
                    ids.len() < limit
                })
                .map_err(|e| GraphError::New(e.to_string()))?;
            return Ok(ids);
        }

        self.with_read_txn(|txn| {
            // Forward scan of Namespace::Nodes from the cursor (exclusive) to the
            // end, early-stopping after `limit`. Byte-identical to the prior
            // `nodes_db.range(txn, &(Excluded(cursor), Unbounded))`: node ids are
            // 16-byte big-endian keys (matching `Database<U128<BE>, Bytes>`), and
            // the seam's `scan_raw` breaks when the visitor returns `false`.
            let mut ids = Vec::with_capacity(limit);
            self.backend
                .scan_heed(txn, Namespace::Nodes, range, |k, _v| {
                    let mut bytes = [0u8; 16];
                    bytes.copy_from_slice(k);
                    ids.push(u128::from_be_bytes(bytes));
                    ids.len() < limit
                })
                .map_err(|e| GraphError::New(e.to_string()))?;
            Ok(ids)
        })
    }

    /// Index one chunk of node ids for a payload field, routed by backend.
    ///
    /// On **LMDB** this runs the original heed loop (byte-identical). On **LSM** it
    /// runs through `with_write_backend` + `get_for_update` + the `_be` payload
    /// helpers, writing to SlateDB. Previously both payload-index build paths called
    /// the LMDB-only `get_for_update_heed` unconditionally, which hits the
    /// `unreachable!("heed write path is LMDB-only")` arm and panics under
    /// `HELIX_STORAGE_BACKEND=lsm` — breaking payload-index builds (e.g. the client's `repo`
    /// filter) on the LSM backend. The payload-index READ path was already routed
    /// (`get_nodes_by_payload_value` / `get_nodes_by_payload_range`); this closes
    /// the write side.
    fn index_payload_chunk(
        &self,
        ids: &[u128],
        field: &str,
        schema: &PayloadIndexSchema,
        db: Option<&Database<Bytes, Bytes>>,
    ) -> Result<u64, GraphError> {
        if self.backend.kind() == BackendKind::Lsm {
            return self.with_write_backend(|w| {
                let mut indexed = 0_u64;
                for id in ids {
                    let node = match self
                        .backend
                        .get_for_update(w, Namespace::Nodes, &id.to_be_bytes(), |v| {
                            v.map(|bytes| SerializedNode::decode_node(bytes, *id))
                                .transpose()
                        })
                        .map_err(|e| GraphError::New(e.to_string()))??
                    {
                        None => continue,
                        Some(node) => node,
                    };
                    let value = Self::payload_value_for_key(&node.properties, field);
                    // Unrecorded: the build job's own writes must not mark
                    // themselves dirty in the build-delta sidecar.
                    self.deindex_node_payload_field_be_unrecorded(w, field, schema, *id, value)?;
                    self.index_node_payload_field_be_unrecorded(w, field, schema, *id, value)?;
                    indexed = indexed.saturating_add(1);
                }
                Ok(indexed)
            });
        }

        self.with_write_txn(|wtxn| {
            let mut indexed = 0_u64;
            for id in ids {
                let node = match self
                    .backend
                    .get_for_update_heed(wtxn, Namespace::Nodes, &id.to_be_bytes(), |v| {
                        v.map(|data| SerializedNode::decode_node(data, *id))
                    })
                    .map_err(|e| GraphError::New(e.to_string()))?
                {
                    None => continue,
                    Some(res) => res?,
                };
                let value = Self::payload_value_for_key(&node.properties, field);
                let db = db.ok_or_else(|| {
                    GraphError::StorageError("LMDB payload-index DB missing".to_string())
                })?;
                self.deindex_node_payload_field(wtxn, db, schema, *id, value)?;
                self.index_node_payload_field(wtxn, db, schema, *id, value)?;
                indexed = indexed.saturating_add(1);
            }
            Ok(indexed)
        })
    }

    fn write_payload_index_chunk(
        &self,
        job: &PayloadIndexJob,
        ids: &[u128],
    ) -> Result<u64, GraphError> {
        self.index_payload_chunk(ids, &job.field_name, &job.schema, job.db.as_ref())
    }

    /// Repair, before an LSM build publishes Ready, every node the build-delta
    /// sidecar recorded (see [`Namespace::PayloadIndexBuild`]): a backfill
    /// chunk that read a node before a foreground write and committed after
    /// it resurrected the old value's row. One pass over the sidecar as of a
    /// snapshot taken after the last chunk: rows recorded later come from
    /// writes that maintained the index themselves, with no chunk left to
    /// undo them. Each batch commits only if none of its nodes was written
    /// after the batch began ([`LsmCheckedTxn`]), so the repair resurrects
    /// nothing either; a refused batch is retried in halves, a refused single
    /// node until it commits. The build lock is held per commit only.
    /// Returns false once `job_id` no longer owns the index.
    fn drain_payload_index_build_deltas(
        &self,
        field: &str,
        schema: &PayloadIndexSchema,
        db_name: &str,
        job_id: &str,
    ) -> Result<bool, GraphError> {
        const BATCH_NODES: usize = 512;
        let read = self
            .backend
            .begin_read()
            .map_err(|e| self.backend_error(e, "payload_index_build_delta_scan"))?;
        let mut start = Bound::Unbounded;
        loop {
            let mut batch: Vec<(u128, Vec<Vec<u8>>)> = Vec::new();
            self.backend
                .scan(
                    &read,
                    Namespace::PayloadIndexBuild(db_name),
                    KeyRange {
                        start: start.clone(),
                        end: Bound::Unbounded,
                    },
                    |k, v| {
                        let Ok(id_bytes) = <[u8; 16]>::try_from(k) else {
                            return true;
                        };
                        let id = u128::from_be_bytes(id_bytes);
                        // Rows sort by (node_id, value): whole row sets per batch.
                        if let Some((_, values)) = batch.last_mut().filter(|(last, _)| *last == id)
                        {
                            values.push(v.to_vec());
                        } else if batch.len() >= BATCH_NODES {
                            return false;
                        } else {
                            batch.push((id, vec![v.to_vec()]));
                        }
                        true
                    },
                )
                .map_err(|e| self.backend_error(e, "payload_index_build_delta_scan"))?;
            let Some((last, _)) = batch.last() else {
                return Ok(true);
            };
            start = Bound::Excluded(last.to_be_bytes().to_vec());
            let mut todo = vec![batch.as_slice()];
            let mut refusals = 0u32;
            while let Some(nodes) = todo.pop() {
                match self.repair_payload_index_nodes(field, schema, db_name, job_id, nodes)? {
                    None => return Ok(false),
                    Some(true) => refusals = 0,
                    Some(false) if nodes.len() > 1 => {
                        let (head, tail) = nodes.split_at(nodes.len() / 2);
                        todo.extend([tail, head]);
                    }
                    Some(false) => {
                        // Lost to a write of this node: retry once it pauses.
                        refusals = refusals.saturating_add(1);
                        std::thread::sleep(Duration::from_millis(u64::from(refusals.min(20)) * 5));
                        todo.push(nodes);
                    }
                }
            }
        }
    }

    /// One checked drain commit for `nodes` (id, recorded delta rows):
    /// re-delete the rows a chunk may have resurrected, reindex each node
    /// from its current value, and consume the rows. `Some(false)`: refused,
    /// a node was written meanwhile; `None`: `job_id` no longer owns the
    /// index.
    fn repair_payload_index_nodes(
        &self,
        field: &str,
        schema: &PayloadIndexSchema,
        db_name: &str,
        job_id: &str,
        nodes: &[(u128, Vec<Vec<u8>>)],
    ) -> Result<Option<bool>, GraphError> {
        let AnyBackend::Lsm(lsm) = self.backend.as_ref() else {
            return Err(GraphError::New(
                "payload index build deltas are LSM-only".to_string(),
            ));
        };
        let error = |e| self.backend_error(e, "payload_index_build_delta_drain");
        // Begun before the reads below, so a write landing after any of them
        // refuses the commit.
        let txn = lsm.begin_checked().map_err(error)?;
        let mut w = self.backend.begin_write().map_err(error)?;
        for (id, recorded) in nodes {
            let id_be = id.to_be_bytes();
            // Re-delete rows a racing backfill chunk may have resurrected.
            // The reindex below re-adds any that are still current (put
            // after delete on the same composite key wins in the batch).
            // Row values are `[seq: u64 BE][old_index_key...]` — an empty
            // tail is a marker row (reindex only, nothing to re-delete).
            for old_key in recorded
                .iter()
                .filter_map(|value| value.get(8..))
                .filter(|old_key| !old_key.is_empty())
            {
                self.backend
                    .delete_dup(&mut w, Namespace::PayloadIndex(db_name), old_key, &id_be)
                    .map_err(error)?;
            }
            let node = self
                .backend
                .get_for_update(&w, Namespace::Nodes, &id_be, |v| {
                    v.map(|bytes| SerializedNode::decode_node(bytes, *id))
                        .transpose()
                })
                .map_err(error)??;
            if let Some(node) = node {
                let value = Self::payload_value_for_key(&node.properties, field);
                self.deindex_node_payload_field_be_unrecorded(&mut w, field, schema, *id, value)?;
                self.index_node_payload_field_be_unrecorded(&mut w, field, schema, *id, value)?;
            }
            for value in recorded {
                self.backend
                    .delete_dup(&mut w, Namespace::PayloadIndexBuild(db_name), &id_be, value)
                    .map_err(error)?;
            }
            txn.mark_read(Namespace::Nodes, &id_be).map_err(error)?;
        }
        let AnyWrite::Lsm(w) = w else {
            unreachable!("an LSM backend begins LSM writes")
        };
        let _build = self.payload_index_build_lock()?;
        if !self.payload_index_job_active(field, job_id)? {
            return Ok(None);
        }
        txn.commit_write(w).map(Some).map_err(error)
    }

    /// Delete the rows an LSM payload index left behind before a build
    /// backfills it (LMDB clears its DB the same way). Dropping an index
    /// leaves its rows, and nothing maintains them while no handle exists, so
    /// a rebuild would otherwise publish rows for values the points no longer
    /// hold. One snapshot drives the sweep: rows foreground writes add after
    /// it are current and survive, and the backfill that follows rewrites
    /// every current row the sweep deleted. Returns the rows deleted.
    fn clear_payload_index_rows(&self, db_name: &str) -> Result<usize, GraphError> {
        self.clear_dup_rows_as_of_now(Namespace::PayloadIndex(db_name), "payload_index_rows_clear")
    }

    /// Delete every row of the dup namespace `ns` that one snapshot taken
    /// here sees, in bounded batches. Rows written after the snapshot are
    /// left: a sweep that scans again until it finds nothing does not end
    /// while writers keep adding rows. Returns the rows deleted.
    fn clear_dup_rows_as_of_now(
        &self,
        ns: Namespace<'_>,
        context: &'static str,
    ) -> Result<usize, GraphError> {
        const BATCH_ROWS: usize = 4096;
        let read = self
            .backend
            .begin_read()
            .map_err(|e| self.backend_error(e, context))?;
        let mut start = Bound::Unbounded;
        let mut cleared = 0usize;
        loop {
            // Whole keys per scan (the next scan resumes after the last one);
            // a key's values are kept once, and deletes commit in bounded
            // batches however many points share the key.
            let mut keys: Vec<(Vec<u8>, Vec<Vec<u8>>)> = Vec::new();
            let mut rows = 0usize;
            self.backend
                .scan(
                    &read,
                    ns,
                    KeyRange {
                        start: start.clone(),
                        end: Bound::Unbounded,
                    },
                    |k, v| {
                        match keys.last_mut() {
                            Some((last, values)) if last.as_slice() == k => values.push(v.to_vec()),
                            _ if rows >= BATCH_ROWS => return false,
                            _ => keys.push((k.to_vec(), vec![v.to_vec()])),
                        }
                        rows += 1;
                        true
                    },
                )
                .map_err(|e| self.backend_error(e, context))?;
            let Some((last_key, _)) = keys.last() else {
                return Ok(cleared);
            };
            start = Bound::Excluded(last_key.clone());
            let mut pending: Vec<(&[u8], &[u8])> = keys
                .iter()
                .flat_map(|(key, values)| {
                    values.iter().map(move |v| (key.as_slice(), v.as_slice()))
                })
                .collect();
            while !pending.is_empty() {
                let batch: Vec<_> = pending.drain(..pending.len().min(BATCH_ROWS)).collect();
                self.with_write_backend(|w| {
                    for (key, value) in &batch {
                        self.backend
                            .delete_dup(w, ns, key, value)
                            .map_err(|e| GraphError::New(e.to_string()))?;
                    }
                    Ok(())
                })?;
            }
            cleared += rows;
        }
    }

    /// Delete the rows the build-delta sidecar for `db_name` holds when this
    /// is called. Used before a fresh build (a failed prior job may have left
    /// rows) and after the Ready flip (post-drain foreground rows are garbage).
    /// One snapshot, as for the index rows: before a build the handle is
    /// already Building, so every foreground write records a row, and those
    /// belong to the build that follows, not to the leftovers. Scanning again
    /// until a scan came back empty, the sweep chased the writers for up to
    /// 10,000 commits and the build under it did not write its first chunk.
    fn clear_payload_index_build_deltas(&self, db_name: &str) -> Result<(), GraphError> {
        self.clear_dup_rows_as_of_now(
            Namespace::PayloadIndexBuild(db_name),
            "payload_index_build_delta_clear",
        )
        .map(|_| ())
    }

    fn payload_index_build_lock(&self) -> Result<std::sync::MutexGuard<'_, ()>, GraphError> {
        self.payload_index_build_lock
            .lock()
            .map_err(|e| GraphError::New(format!("Payload index build lock poisoned: {}", e)))
    }

    fn payload_index_job_active(&self, field_name: &str, job_id: &str) -> Result<bool, GraphError> {
        let payload_indices = self
            .payload_indices
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        Ok(payload_indices
            .get(field_name)
            .map(|handle| handle.is_building_job(job_id))
            .unwrap_or(false))
    }

    fn update_payload_index_progress(
        &self,
        field_name: &str,
        job_id: &str,
        cursor: Option<u128>,
        indexed_nodes: u64,
    ) -> Result<(), GraphError> {
        let payload_indices = self
            .payload_indices
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        if let Some(handle) = payload_indices.get(field_name) {
            handle.update_build_progress(job_id, cursor, indexed_nodes);
        }
        Ok(())
    }

    fn mark_payload_index_failed(&self, field_name: &str, job_id: &str, error: String) {
        if let Ok(payload_indices) = self.payload_indices.read() {
            if let Some(handle) = payload_indices.get(field_name) {
                handle.mark_failed(job_id, error);
            }
        }
    }

    fn publish_payload_index_ready(&self, job: &PayloadIndexJob) -> Result<(), GraphError> {
        let _admin_guard = self
            .payload_index_admin_gate
            .lock()
            .map_err(|e| GraphError::New(format!("Payload index admin gate poisoned: {}", e)))?;
        let handle = {
            let payload_indices = self
                .payload_indices
                .read()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            let Some(handle) = payload_indices.get(&job.field_name) else {
                return Ok(());
            };
            if !handle.is_building_job(&job.job_id) {
                return Ok(());
            }
            handle.clone()
        };

        // Routed for LSM — this runs on the background executor thread; an
        // unrouted heed metadata write here panics under LSM, kills the executor
        // thread, and poisons the admin gate (closed-channel for all later builds).
        self.register_payload_index_metadata(&job.field_name, &job.schema)?;
        handle.mark_ready(&job.job_id);
        if self.backend.kind() == BackendKind::Lsm {
            // Delta rows foreground writes recorded after the drain are
            // garbage once Ready. Swept here, by the job that flipped it and
            // still under the gate: once the gate is free a newer build of
            // the field may be recording rows for its own drain.
            self.clear_payload_index_build_deltas(&Self::payload_index_db_name(
                &job.field_name,
                &job.schema,
            ))?;
        }
        Ok(())
    }

    pub fn delete_payload_index(&self, name: &str) -> Result<(), GraphError> {
        let _admin_guard = self
            .payload_index_admin_gate
            .lock()
            .map_err(|e| GraphError::New(format!("Payload index admin gate poisoned: {}", e)))?;
        let handle = {
            // Waits out an in-flight async chunk or drain of this index.
            let _build = self.payload_index_build_lock()?;
            let mut payload_indices = self
                .payload_indices
                .write()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            let handle = payload_indices.remove(name);
            if let Some(handle) = &handle {
                handle.mark_cancelled();
            }
            handle
        };

        let Some(handle) = handle else {
            return Ok(());
        };

        // LSM: the heed write path (`set_payload_indices_metadata` / `db.clear`
        // via `with_write_txn`) hits `unreachable!("heed write path is LMDB-only")`
        // and PANICS on the LSM backend. Because delete runs on a request worker
        // (and builds on the executor), that panic poisons the
        // `payload_index_admin_gate` mutex and bricks ALL subsequent payload-index
        // ops on the collection (create returns "gate poisoned"; builds stall at
        // indexed_nodes=0). Route the metadata removal through the backend seam,
        // mirroring `register_payload_index_metadata`. The in-memory handle is
        // already removed above so the index is no longer consulted; orphaned
        // `PayloadIndex` keys left in SlateDB are never read without a handle;
        // a rebuild clears them before its backfill (`clear_payload_index_rows`).
        if self.backend.kind() == BackendKind::Lsm {
            self.with_write_backend(|w| {
                self.update_metadata_be(w, |metadata| {
                    let mut indices = metadata.payload_indices.clone();
                    indices.remove(name);
                    metadata.set_payload_indices(indices);
                    Ok(())
                })
                .map(|_| ())
            })?;
            return Ok(());
        }

        self.with_write_txn(|wtxn| {
            handle.lmdb_db()?.clear(wtxn)?;
            let mut payload_index_metadata = self.get_metadata(wtxn)?.payload_indices;
            payload_index_metadata.remove(name);
            self.set_payload_indices_metadata(wtxn, payload_index_metadata)?;
            Ok(())
        })?;

        Ok(())
    }

    pub fn has_payload_index(&self, name: &str) -> Option<PayloadIndexSchema> {
        self.payload_indices.read().ok().and_then(|indices| {
            indices
                .get(name)
                .filter(|handle| handle.is_ready())
                .map(|handle| handle.schema.clone())
        })
    }

    /// [`Self::has_payload_index`] judged for the state `r` reads: an index
    /// published after `r` was opened is incomplete there. A reader replica's
    /// handles come from the metadata it opened with and are never refreshed,
    /// while the writer may drop the index, clear its rows and rebuild it, so
    /// a replica also requires the metadata `r` reads to register the index
    /// (it is registered only once the rebuild's rows are complete).
    pub fn has_payload_index_at(&self, name: &str, r: &AnyRead<'_>) -> Option<PayloadIndexSchema> {
        let opened_at = r.snapshot_opened_at();
        let schema = self.payload_indices.read().ok().and_then(|indices| {
            indices
                .get(name)
                .filter(|handle| handle.is_ready_for(opened_at))
                .map(|handle| handle.schema.clone())
        })?;
        if matches!(r, AnyRead::LsmReader(_)) && !self.payload_index_registered_be(r, name, &schema)
        {
            return None;
        }
        Some(schema)
    }

    /// The metadata `r` reads registers `name` with `schema`; any read or
    /// decode failure counts as unregistered (the field is then unindexed).
    fn payload_index_registered_be(
        &self,
        r: &AnyRead<'_>,
        name: &str,
        schema: &PayloadIndexSchema,
    ) -> bool {
        self.backend
            .get_with(
                r,
                Namespace::Metadata,
                METADATA_CURRENT_KEY.as_bytes(),
                |v| {
                    v.and_then(|bytes| Self::try_deserialize_metadata(bytes).ok())
                        .is_some_and(|metadata| metadata.payload_indices.get(name) == Some(schema))
                },
            )
            .unwrap_or(false)
    }

    /// The dense space `name` as the metadata `r` reads lists it, for a dense
    /// read pinned at another sequence than the replica's shared segment list
    /// (`NamedVectorManager::dense_segment_infos_at`). `None` on any read or
    /// decode failure, and when that metadata has no such space.
    pub(crate) fn dense_space_at(
        backend: &AnyBackend,
        r: &AnyRead<'_>,
        name: &str,
    ) -> Option<crate::helix_engine::vector_core::named_vectors::DenseVectorSpaceMetadata> {
        backend
            .get_with(
                r,
                Namespace::Metadata,
                METADATA_CURRENT_KEY.as_bytes(),
                |v| {
                    Self::try_deserialize_metadata(v?)
                        .ok()?
                        .dense_vector_spaces
                        .remove(name)
                },
            )
            .ok()
            .flatten()
    }

    pub fn list_payload_indexes(&self) -> Result<HashMap<String, PayloadIndexSchema>, GraphError> {
        let payload_indices = self
            .payload_indices
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        Ok(payload_indices
            .iter()
            .filter(|(_, handle)| handle.is_ready())
            .map(|(name, handle)| (name.clone(), handle.schema.clone()))
            .collect())
    }

    pub fn payload_index_status(
        &self,
        name: &str,
    ) -> Result<Option<PayloadIndexStatus>, GraphError> {
        let payload_indices = self
            .payload_indices
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        Ok(payload_indices.get(name).map(|handle| handle.status(name)))
    }

    pub fn list_payload_index_statuses(&self) -> Result<Vec<PayloadIndexStatus>, GraphError> {
        let payload_indices = self
            .payload_indices
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        Ok(payload_indices
            .iter()
            .map(|(name, handle)| handle.status(name))
            .collect())
    }

    pub fn try_list_payload_index_statuses(
        &self,
    ) -> Result<Option<Vec<PayloadIndexStatus>>, GraphError> {
        let payload_indices = match self.payload_indices.try_read() {
            Ok(payload_indices) => payload_indices,
            Err(std::sync::TryLockError::WouldBlock) => {
                metrics::counter!(
                    "helix_payload_index_status_try_list_total",
                    "outcome" => "map_busy"
                )
                .increment(1);
                return Ok(None);
            }
            Err(std::sync::TryLockError::Poisoned(e)) => {
                return Err(GraphError::New(format!("Lock poisoned: {}", e)));
            }
        };

        let mut statuses = Vec::with_capacity(payload_indices.len());
        let mut skipped = 0_u64;
        for (name, handle) in payload_indices.iter() {
            match handle.try_status(name) {
                Some(status) => statuses.push(status),
                None => skipped = skipped.saturating_add(1),
            }
        }
        if skipped > 0 {
            metrics::counter!(
                "helix_payload_index_status_try_list_total",
                "outcome" => "state_busy"
            )
            .increment(skipped);
        }
        metrics::counter!(
            "helix_payload_index_status_try_list_total",
            "outcome" => "ok"
        )
        .increment(1);
        Ok(Some(statuses))
    }

    pub fn get_nodes_by_payload_value(
        &self,
        txn: &RoTxn,
        field: &str,
        value: &Value,
    ) -> Result<Vec<u128>, GraphError> {
        // Before the map guard: the `_be` path takes its own, and a nested
        // read deadlocks behind a queued writer.
        if self.backend.kind() == BackendKind::Lsm {
            let r = self.backend.read_borrowed(txn);
            return self.get_nodes_by_payload_value_be(&r, field, value);
        }
        let payload_indices = self
            .payload_indices
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        let handle = payload_indices
            .get(field)
            .filter(|handle| handle.is_ready())
            .ok_or_else(|| GraphError::New(format!("Payload index '{}' not found", field)))?;
        if handle.schema != PayloadIndexSchema::Keyword {
            return Err(GraphError::New(format!(
                "Payload index '{}' is not a keyword index",
                field
            )));
        }

        let mut ids = Vec::new();
        for encoded in Self::payload_keyword_lookup_keys(value) {
            if let Some(iter) = handle.lmdb_db()?.get_duplicates(txn, &encoded)? {
                for item in iter {
                    let (_, val_bytes) = item?;
                    ids.push(u128::from_be_bytes(
                        val_bytes
                            .try_into()
                            .map_err(|_| GraphError::SliceLengthError)?,
                    ));
                }
            }
        }
        ids.sort_unstable();
        ids.dedup();
        Ok(ids)
    }

    pub(crate) fn get_nodes_by_payload_value_be(
        &self,
        r: &AnyRead<'_>,
        field: &str,
        value: &Value,
    ) -> Result<Vec<u128>, GraphError> {
        let payload_indices = self
            .payload_indices
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        let handle = payload_indices
            .get(field)
            .filter(|handle| handle.is_ready())
            .ok_or_else(|| GraphError::New(format!("Payload index '{}' not found", field)))?;
        if handle.schema != PayloadIndexSchema::Keyword {
            return Err(GraphError::New(format!(
                "Payload index '{}' is not a keyword index",
                field
            )));
        }

        let db_name = Self::payload_index_db_name(field, &handle.schema);
        let mut ids = Vec::new();
        for encoded in Self::payload_keyword_lookup_keys(value) {
            self.backend
                .for_each_dup(
                    r,
                    Namespace::PayloadIndex(&db_name),
                    &encoded,
                    |val_bytes| {
                        if let Ok(arr) = <[u8; 16]>::try_from(val_bytes) {
                            ids.push(u128::from_be_bytes(arr));
                        }
                        true
                    },
                )
                .map_err(|e| GraphError::New(e.to_string()))?;
        }
        payload_index_rows_read("list", ids.len());
        ids.sort_unstable();
        ids.dedup();
        Ok(ids)
    }

    /// Count-only variant of [`Self::get_nodes_by_payload_value_be`] for the
    /// filter cardinality probe: walks the same duplicate entries but never
    /// collects ids, and stops iterating as soon as the running count exceeds
    /// `cap` (returning `cap + 1`, meaning "exceeds cap"). Entries are counted
    /// without the reader's sort/dedup pass, so a node indexed under multiple
    /// lookup keys can be counted more than once — acceptable for an
    /// upper-bound estimate.
    pub(crate) fn count_nodes_by_payload_value_be(
        &self,
        r: &AnyRead<'_>,
        field: &str,
        value: &Value,
        cap: usize,
    ) -> Result<usize, GraphError> {
        let payload_indices = self
            .payload_indices
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        let handle = payload_indices
            .get(field)
            .filter(|handle| handle.is_ready())
            .ok_or_else(|| GraphError::New(format!("Payload index '{}' not found", field)))?;
        if handle.schema != PayloadIndexSchema::Keyword {
            return Err(GraphError::New(format!(
                "Payload index '{}' is not a keyword index",
                field
            )));
        }

        let db_name = Self::payload_index_db_name(field, &handle.schema);
        let mut count = 0usize;
        for encoded in Self::payload_keyword_lookup_keys(value) {
            if count > cap {
                break;
            }
            self.backend
                .for_each_dup(
                    r,
                    Namespace::PayloadIndex(&db_name),
                    &encoded,
                    |_val_bytes| {
                        count += 1;
                        count <= cap
                    },
                )
                .map_err(|e| GraphError::New(e.to_string()))?;
        }
        payload_index_rows_read("count", count);
        Ok(count)
    }

    /// The rows the keyword index of `field` lists under `value`, named for
    /// [`Self::payload_value_ids_from_be`]. Fails as
    /// [`Self::get_nodes_by_payload_value_be`] does where the index is not
    /// ready or is no keyword index.
    pub(crate) fn payload_value_rows(
        &self,
        field: &str,
        value: &Value,
    ) -> Result<PayloadValueRows, GraphError> {
        let payload_indices = self
            .payload_indices
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        let handle = payload_indices
            .get(field)
            .filter(|handle| handle.is_ready())
            .ok_or_else(|| GraphError::New(format!("Payload index '{}' not found", field)))?;
        if handle.schema != PayloadIndexSchema::Keyword {
            return Err(GraphError::New(format!(
                "Payload index '{}' is not a keyword index",
                field
            )));
        }
        Ok(PayloadValueRows {
            db_name: Self::payload_index_db_name(field, &handle.schema),
            keys: Self::payload_keyword_lookup_keys(value),
        })
    }

    /// The first `max` ids of `rows` from `from` on, in id order: a scan
    /// page's window of the list [`Self::get_nodes_by_payload_value_be`]
    /// reads whole. One value's rows are stored in id order, so the read
    /// starts at `from` and ends after `max` rows of each lookup key.
    pub(crate) fn payload_value_ids_from_be(
        &self,
        r: &AnyRead<'_>,
        rows: &PayloadValueRows,
        from: u128,
        max: usize,
    ) -> Result<Vec<u128>, GraphError> {
        let from = from.to_be_bytes();
        let mut ids = Vec::new();
        for key in &rows.keys {
            let mut taken = 0usize;
            self.backend
                .for_each_dup_from(
                    r,
                    Namespace::PayloadIndex(&rows.db_name),
                    key,
                    &from,
                    |val_bytes| {
                        if let Ok(arr) = <[u8; 16]>::try_from(val_bytes) {
                            ids.push(u128::from_be_bytes(arr));
                            taken += 1;
                        }
                        taken < max
                    },
                )
                .map_err(|e| GraphError::New(e.to_string()))?;
        }
        payload_index_rows_read("window", ids.len());
        ids.sort_unstable();
        ids.dedup();
        ids.truncate(max);
        Ok(ids)
    }

    /// Resolve a Qdrant `match.text` (case-insensitive substring) condition by
    /// walking the keyword index terms. Returns `Ok(None)` — "treat as
    /// unindexed", the caller falls back to a recheck scan — when a term can't
    /// be decoded or the walk would visit more than `walk_max` index entries
    /// (`0` = unbounded).
    pub(crate) fn get_nodes_by_payload_text(
        &self,
        txn: &RoTxn,
        field: &str,
        text: &str,
        walk_max: usize,
    ) -> Result<Option<Vec<u128>>, GraphError> {
        // Before the map guard, as in `get_nodes_by_payload_value`.
        if self.backend.kind() == BackendKind::Lsm {
            let r = self.backend.read_borrowed(txn);
            return self.get_nodes_by_payload_text_be(&r, field, text, walk_max);
        }
        let payload_indices = self
            .payload_indices
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        let handle = payload_indices
            .get(field)
            .filter(|handle| handle.is_ready())
            .ok_or_else(|| GraphError::New(format!("Payload index '{}' not found", field)))?;
        if handle.schema != PayloadIndexSchema::Keyword {
            return Err(GraphError::New(format!(
                "Payload index '{}' is not a keyword index",
                field
            )));
        }

        // No exact-value shortcut: Qdrant `match.text` is a case-insensitive
        // substring match, so the exact keyword hits are only a subset of the
        // answer. The planner treats these ids as the complete candidate set,
        // so returning just the exact hits silently dropped every point whose
        // value merely contains `text` from scroll/count/search results.
        let needle = text.to_lowercase();
        let db = handle.lmdb_db()?;
        let mut ids = Vec::new();
        let mut visited = 0usize;
        // One cursor step per DISTINCT term (MDB_NEXT_NODUP); duplicates (ids)
        // are only read for matching terms, so low-cardinality fields scale
        // with their term count rather than their point count.
        for item in db.iter(txn)?.move_between_keys() {
            let (key, _) = item?;
            visited += 1;
            if Self::payload_text_walk_capped(visited, walk_max) {
                return Ok(None);
            }
            match Self::payload_index_key_matches_text(key, &needle) {
                Some(true) => {
                    if let Some(dups) = db.get_duplicates(txn, key)? {
                        for dup in dups {
                            let (_, val_bytes) = dup?;
                            visited += 1;
                            if Self::payload_text_walk_capped(visited, walk_max) {
                                return Ok(None);
                            }
                            ids.push(u128::from_be_bytes(
                                val_bytes
                                    .try_into()
                                    .map_err(|_| GraphError::SliceLengthError)?,
                            ));
                        }
                    }
                }
                Some(false) => {}
                None => return Ok(None),
            }
        }
        ids.sort_unstable();
        ids.dedup();
        metrics::counter!("helix_payload_text_walk_total", "outcome" => "walked").increment(1);
        Ok(Some(ids))
    }

    /// True (and counted) once a text walk has visited more than `walk_max`
    /// index entries; `walk_max == 0` disables the cap.
    fn payload_text_walk_capped(visited: usize, walk_max: usize) -> bool {
        let capped = walk_max > 0 && visited > walk_max;
        if capped {
            metrics::counter!("helix_payload_text_walk_total", "outcome" => "capped").increment(1);
        }
        capped
    }

    /// The key a walk of `field` for `text` is cached under in `r`. `None`
    /// for a read that is not pinned to a sequence, and for an index that is
    /// not ready.
    fn text_walk_key(
        handle: &PayloadIndexHandle,
        r: &AnyRead<'_>,
        field: &str,
        text: &str,
    ) -> Option<TextWalkKey> {
        let seq = r.snapshot_seq()?;
        match &*handle.state.read().ok()? {
            PayloadIndexState::Ready { since, .. } => Some(TextWalkKey::new(
                field,
                text,
                Arc::as_ptr(&handle.state) as usize,
                *since,
                seq,
            )),
            _ => None,
        }
    }

    /// Whether a text walk of `field` is known to give up in `r`: the mark
    /// [`Self::get_nodes_by_payload_text_exact_be`] leaves when one does,
    /// read with no walk. `false` where there is no mark to read, which is
    /// every read that is not pinned to a sequence.
    pub(crate) fn payload_text_walk_gave_up_be(
        &self,
        r: &AnyRead<'_>,
        field: &str,
        walk_max: usize,
    ) -> bool {
        let Ok(payload_indices) = self.payload_indices.read() else {
            return false;
        };
        let Some(key) = payload_indices
            .get(field)
            .filter(|handle| handle.is_ready() && handle.schema == PayloadIndexSchema::Keyword)
            .and_then(|handle| Self::text_walk_key(handle, r, field, ""))
        else {
            return false;
        };
        let gave_up = self.text_walk_cache.lock().gave_up(&key.of_field());
        gave_up.is_some_and(|visited| Self::payload_text_walk_capped(visited, walk_max))
    }

    /// Backend twin of [`Self::get_nodes_by_payload_text`]; same `walk_max`
    /// and `Ok(None)` contract, except that a hashed key is kept as a
    /// candidate (see [`Self::get_nodes_by_payload_text_exact_be`]).
    pub(crate) fn get_nodes_by_payload_text_be(
        &self,
        r: &AnyRead<'_>,
        field: &str,
        text: &str,
        walk_max: usize,
    ) -> Result<Option<Vec<u128>>, GraphError> {
        Ok(self
            .get_nodes_by_payload_text_exact_be(r, field, text, walk_max)?
            .map(|(ids, _)| ids))
    }

    /// [`Self::get_nodes_by_payload_text_be`] plus whether the ids are exactly
    /// the `match.text` matches. The walk applies `match.text`'s test (lower
    /// case, substring) to each decoded term but also matches terms indexed
    /// from array elements, which `match.text` rejects. When every matching
    /// row is a plain string row ([`PAYLOAD_INDEX_PLAIN_ROW`]: a top-level
    /// string value), no hit came from an array, so the set is exact.
    ///
    /// A value too long for a raw index key is indexed under a hash of it
    /// ([`Self::compact_index_bytes`]), so the walk cannot test its text: its
    /// ids are returned as candidates and the set is not exact.
    pub(crate) fn get_nodes_by_payload_text_exact_be(
        &self,
        r: &AnyRead<'_>,
        field: &str,
        text: &str,
        walk_max: usize,
    ) -> Result<Option<(Vec<u128>, bool)>, GraphError> {
        let payload_indices = self
            .payload_indices
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        let handle = payload_indices
            .get(field)
            .filter(|handle| handle.is_ready())
            .ok_or_else(|| GraphError::New(format!("Payload index '{}' not found", field)))?;
        if handle.schema != PayloadIndexSchema::Keyword {
            return Err(GraphError::New(format!(
                "Payload index '{}' is not a keyword index",
                field
            )));
        }

        let cache_key = Self::text_walk_key(handle, r, field, text);
        if let Some(key) = &cache_key {
            // The rows read do not depend on the text: a walk of this field
            // that gave up at this snapshot gives up for this text too.
            let gave_up = self.text_walk_cache.lock().gave_up(&key.of_field());
            if gave_up.is_some_and(|visited| Self::payload_text_walk_capped(visited, walk_max)) {
                return Ok(None);
            }
            if let Some(hit) = self.text_walk_cache.lock().get(key) {
                if Self::payload_text_walk_capped(hit.visited, walk_max) {
                    return Ok(None);
                }
                if hit.exact {
                    metrics::counter!("helix_payload_text_walk_exact_total").increment(1);
                }
                return Ok(Some((hit.ids.clone(), hit.exact)));
            }
        }

        // No exact-value shortcut — see `get_nodes_by_payload_text`: the exact
        // hits are a strict subset of the substring match the filter applies.
        let needle = text.to_lowercase();
        let db_name = Self::payload_index_db_name(field, &handle.schema);
        let mut ids = Vec::new();
        let mut undecodable = false;
        let mut capped = false;
        let mut exact = true;
        let mut visited = 0usize;
        // Entries arrive grouped by term; decode a term once and reuse the
        // verdict (a memcmp) for each of its duplicate ids.
        let mut last_term: Vec<u8> = Vec::new();
        let mut last_matched: Option<(bool, bool)> = None;
        self.backend
            .scan_dup_stored(
                r,
                Namespace::PayloadIndex(&db_name),
                KeyRange::all(),
                |key, val_bytes, stored| {
                    visited += 1;
                    if Self::payload_text_walk_capped(visited, walk_max) {
                        capped = true;
                        return false;
                    }
                    let (matched, is_string) = match last_matched {
                        Some(verdict) if key == last_term.as_slice() => verdict,
                        // Hashed long value: a candidate the recheck decides.
                        // Giving up here instead sent every text lookup on a
                        // field with one long value to a paged scan.
                        _ if key.starts_with(COMPACT_INDEX_KEY_PREFIX) => (true, false),
                        _ => match bincode::deserialize::<Value>(key) {
                            Ok(value) => {
                                let verdict = (
                                    Self::payload_value_matches_text(&value, &needle),
                                    matches!(value, Value::String(_)),
                                );
                                last_term.clear();
                                last_term.extend_from_slice(key);
                                last_matched = Some(verdict);
                                verdict
                            }
                            Err(_) => {
                                undecodable = true;
                                return false;
                            }
                        },
                    };
                    if matched {
                        exact &= is_string && stored == PAYLOAD_INDEX_PLAIN_ROW;
                        if let Ok(arr) = <[u8; 16]>::try_from(val_bytes) {
                            ids.push(u128::from_be_bytes(arr));
                        }
                    }
                    true
                },
            )
            .map_err(|e| GraphError::New(e.to_string()))?;
        if undecodable {
            metrics::counter!("helix_payload_text_walk_total", "outcome" => "undecodable")
                .increment(1);
        }
        if capped {
            if let Some(key) = &cache_key {
                self.text_walk_cache.lock().insert(
                    key.of_field(),
                    TextWalk {
                        ids: Vec::new(),
                        exact: false,
                        visited,
                    },
                );
            }
        }
        if undecodable || capped {
            return Ok(None);
        }
        ids.sort_unstable();
        ids.dedup();
        metrics::counter!("helix_payload_text_walk_total", "outcome" => "walked").increment(1);
        if exact {
            metrics::counter!("helix_payload_text_walk_exact_total").increment(1);
        }
        if let Some(key) = cache_key {
            self.text_walk_cache.lock().insert(
                key,
                TextWalk {
                    ids: ids.clone(),
                    exact,
                    visited,
                },
            );
        }
        Ok(Some((ids, exact)))
    }

    fn payload_index_key_matches_text(key: &[u8], needle_lower: &str) -> Option<bool> {
        match bincode::deserialize::<Value>(key) {
            Ok(value) => Some(Self::payload_value_matches_text(&value, needle_lower)),
            Err(_) => None,
        }
    }

    fn payload_value_matches_text(value: &Value, needle_lower: &str) -> bool {
        match value {
            Value::String(value) => value.to_lowercase().contains(needle_lower),
            Value::Array(values) => values
                .iter()
                .any(|value| Self::payload_value_matches_text(value, needle_lower)),
            _ => false,
        }
    }

    pub(crate) fn get_nodes_by_payload_range_be(
        &self,
        r: &AnyRead<'_>,
        field: &str,
        lower: Option<(f64, bool)>,
        upper: Option<(f64, bool)>,
    ) -> Result<Vec<u128>, GraphError> {
        use std::ops::Bound;

        let payload_indices = self
            .payload_indices
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        let handle = payload_indices
            .get(field)
            .filter(|handle| handle.is_ready())
            .ok_or_else(|| GraphError::New(format!("Payload index '{}' not found", field)))?;
        if !matches!(
            handle.schema,
            PayloadIndexSchema::Integer | PayloadIndexSchema::Float
        ) {
            return Err(GraphError::New(format!(
                "Payload index '{}' is not a numeric range index",
                field
            )));
        }

        let lower_key = lower.map(|(value, _)| Self::ordered_f64_bytes(value));
        let upper_key = upper.map(|(value, _)| Self::ordered_f64_bytes(value));
        let range = KeyRange {
            start: lower_key
                .as_ref()
                .map(
                    |key| match lower.as_ref().map(|(_, inclusive)| *inclusive) {
                        Some(true) => Bound::Included(key.clone()),
                        Some(false) => Bound::Excluded(key.clone()),
                        None => Bound::Unbounded,
                    },
                )
                .unwrap_or(Bound::Unbounded),
            end: upper_key
                .as_ref()
                .map(
                    |key| match upper.as_ref().map(|(_, inclusive)| *inclusive) {
                        Some(true) => Bound::Included(key.clone()),
                        Some(false) => Bound::Excluded(key.clone()),
                        None => Bound::Unbounded,
                    },
                )
                .unwrap_or(Bound::Unbounded),
        };

        let db_name = Self::payload_index_db_name(field, &handle.schema);
        let mut ids = Vec::new();
        self.backend
            .scan(
                r,
                Namespace::PayloadIndex(&db_name),
                range,
                |_key, val_bytes| {
                    if let Ok(arr) = <[u8; 16]>::try_from(val_bytes) {
                        ids.push(u128::from_be_bytes(arr));
                    }
                    true
                },
            )
            .map_err(|e| GraphError::New(e.to_string()))?;
        ids.sort_unstable();
        ids.dedup();
        Ok(ids)
    }

    /// Count-only variant of [`Self::get_nodes_by_payload_range_be`] for the
    /// filter cardinality probe: runs the same `KeyRange` scan but never
    /// collects ids, and stops as soon as the running count exceeds `cap`
    /// (returning `cap + 1`, meaning "exceeds cap").
    pub(crate) fn count_nodes_by_payload_range_be(
        &self,
        r: &AnyRead<'_>,
        field: &str,
        lower: Option<(f64, bool)>,
        upper: Option<(f64, bool)>,
        cap: usize,
    ) -> Result<usize, GraphError> {
        use std::ops::Bound;

        let payload_indices = self
            .payload_indices
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        let handle = payload_indices
            .get(field)
            .filter(|handle| handle.is_ready())
            .ok_or_else(|| GraphError::New(format!("Payload index '{}' not found", field)))?;
        if !matches!(
            handle.schema,
            PayloadIndexSchema::Integer | PayloadIndexSchema::Float
        ) {
            return Err(GraphError::New(format!(
                "Payload index '{}' is not a numeric range index",
                field
            )));
        }

        let lower_key = lower.map(|(value, _)| Self::ordered_f64_bytes(value));
        let upper_key = upper.map(|(value, _)| Self::ordered_f64_bytes(value));
        let range = KeyRange {
            start: lower_key
                .as_ref()
                .map(
                    |key| match lower.as_ref().map(|(_, inclusive)| *inclusive) {
                        Some(true) => Bound::Included(key.clone()),
                        Some(false) => Bound::Excluded(key.clone()),
                        None => Bound::Unbounded,
                    },
                )
                .unwrap_or(Bound::Unbounded),
            end: upper_key
                .as_ref()
                .map(
                    |key| match upper.as_ref().map(|(_, inclusive)| *inclusive) {
                        Some(true) => Bound::Included(key.clone()),
                        Some(false) => Bound::Excluded(key.clone()),
                        None => Bound::Unbounded,
                    },
                )
                .unwrap_or(Bound::Unbounded),
        };

        let db_name = Self::payload_index_db_name(field, &handle.schema);
        let mut count = 0usize;
        self.backend
            .scan(
                r,
                Namespace::PayloadIndex(&db_name),
                range,
                |_key, _val_bytes| {
                    count += 1;
                    count <= cap
                },
            )
            .map_err(|e| GraphError::New(e.to_string()))?;
        Ok(count)
    }

    pub fn get_nodes_by_payload_range(
        &self,
        txn: &RoTxn,
        field: &str,
        lower: Option<(f64, bool)>,
        upper: Option<(f64, bool)>,
    ) -> Result<Vec<u128>, GraphError> {
        use std::ops::Bound;

        let payload_indices = self
            .payload_indices
            .read()
            .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
        let handle = payload_indices
            .get(field)
            .filter(|handle| handle.is_ready())
            .ok_or_else(|| GraphError::New(format!("Payload index '{}' not found", field)))?;
        if !matches!(
            handle.schema,
            PayloadIndexSchema::Integer | PayloadIndexSchema::Float
        ) {
            return Err(GraphError::New(format!(
                "Payload index '{}' is not a numeric range index",
                field
            )));
        }

        let lower_key = lower.map(|(value, _)| Self::ordered_f64_bytes(value));
        let upper_key = upper.map(|(value, _)| Self::ordered_f64_bytes(value));
        if self.backend.kind() == BackendKind::Lsm {
            let db_name = Self::payload_index_db_name(field, &handle.schema);
            let r = self.backend.read_borrowed(txn);
            let range = KeyRange {
                start: lower_key
                    .as_ref()
                    .map(
                        |key| match lower.as_ref().map(|(_, inclusive)| *inclusive) {
                            Some(true) => Bound::Included(key.clone()),
                            Some(false) => Bound::Excluded(key.clone()),
                            None => Bound::Unbounded,
                        },
                    )
                    .unwrap_or(Bound::Unbounded),
                end: upper_key
                    .as_ref()
                    .map(
                        |key| match upper.as_ref().map(|(_, inclusive)| *inclusive) {
                            Some(true) => Bound::Included(key.clone()),
                            Some(false) => Bound::Excluded(key.clone()),
                            None => Bound::Unbounded,
                        },
                    )
                    .unwrap_or(Bound::Unbounded),
            };
            let mut ids = Vec::new();
            self.backend
                .scan(
                    &r,
                    Namespace::PayloadIndex(&db_name),
                    range,
                    |_key, val_bytes| {
                        if let Ok(arr) = <[u8; 16]>::try_from(val_bytes) {
                            ids.push(u128::from_be_bytes(arr));
                        }
                        true
                    },
                )
                .map_err(|e| GraphError::New(e.to_string()))?;
            ids.sort_unstable();
            ids.dedup();
            return Ok(ids);
        }

        let lower_bound = match (lower.as_ref(), lower_key.as_ref()) {
            (Some((_, true)), Some(key)) => Bound::Included(key.as_slice()),
            (Some((_, false)), Some(key)) => Bound::Excluded(key.as_slice()),
            _ => Bound::Unbounded,
        };
        let upper_bound = match (upper.as_ref(), upper_key.as_ref()) {
            (Some((_, true)), Some(key)) => Bound::Included(key.as_slice()),
            (Some((_, false)), Some(key)) => Bound::Excluded(key.as_slice()),
            _ => Bound::Unbounded,
        };

        let mut ids = Vec::new();
        let iter = handle.lmdb_db()?.range(txn, &(lower_bound, upper_bound))?;
        for item in iter {
            let (_, val_bytes) = item?;
            ids.push(u128::from_be_bytes(
                val_bytes
                    .try_into()
                    .map_err(|_| GraphError::SliceLengthError)?,
            ));
        }
        ids.sort_unstable();
        ids.dedup();
        Ok(ids)
    }

    fn payload_keyword_index_keys(value: &Value) -> Vec<Vec<u8>> {
        match value {
            Value::Array(values) => values
                .iter()
                .filter_map(|value| Self::stable_index_key_for_value(value).ok())
                .collect(),
            _ => Self::stable_index_key_for_value(value)
                .map(|bytes| vec![bytes])
                .unwrap_or_default(),
        }
    }

    fn payload_keyword_lookup_keys(value: &Value) -> Vec<Vec<u8>> {
        let mut keys = Vec::new();
        Self::push_payload_keyword_lookup_keys(value, &mut keys);
        keys
    }

    fn push_payload_keyword_lookup_keys(value: &Value, keys: &mut Vec<Vec<u8>>) {
        match value {
            Value::Array(values) => {
                for value in values {
                    Self::push_payload_keyword_lookup_keys(value, keys);
                }
            }
            _ => {
                for value in Self::numeric_equivalent_values(value) {
                    if let Ok(key) = Self::stable_index_key_for_value(&value) {
                        if !keys.iter().any(|existing| existing == &key) {
                            keys.push(key);
                        }
                    }
                }
            }
        }
    }

    fn numeric_equivalent_values(value: &Value) -> Vec<Value> {
        let mut values = Vec::new();
        match value {
            Value::I8(v) => Self::push_integer_equivalent_values(*v as i128, &mut values),
            Value::I16(v) => Self::push_integer_equivalent_values(*v as i128, &mut values),
            Value::I32(v) => Self::push_integer_equivalent_values(*v as i128, &mut values),
            Value::I64(v) => Self::push_integer_equivalent_values(*v as i128, &mut values),
            Value::U8(v) => Self::push_integer_equivalent_values(*v as i128, &mut values),
            Value::U16(v) => Self::push_integer_equivalent_values(*v as i128, &mut values),
            Value::U32(v) => Self::push_integer_equivalent_values(*v as i128, &mut values),
            Value::U64(v) if *v <= i128::MAX as u64 => {
                Self::push_integer_equivalent_values(*v as i128, &mut values)
            }
            Value::U64(v) => values.push(Value::U64(*v)),
            Value::U128(v) if *v <= i128::MAX as u128 => {
                Self::push_integer_equivalent_values(*v as i128, &mut values)
            }
            Value::U128(v) => values.push(Value::U128(*v)),
            Value::F32(v) => Self::push_float_equivalent_values(*v as f64, &mut values),
            Value::F64(v) => Self::push_float_equivalent_values(*v, &mut values),
            _ => values.push(value.clone()),
        }
        values
    }

    fn push_integer_equivalent_values(value: i128, values: &mut Vec<Value>) {
        if value >= i8::MIN as i128 && value <= i8::MAX as i128 {
            values.push(Value::I8(value as i8));
        }
        if value >= i16::MIN as i128 && value <= i16::MAX as i128 {
            values.push(Value::I16(value as i16));
        }
        if value >= i32::MIN as i128 && value <= i32::MAX as i128 {
            values.push(Value::I32(value as i32));
        }
        if value >= i64::MIN as i128 && value <= i64::MAX as i128 {
            values.push(Value::I64(value as i64));
        }
        if value >= 0 {
            if value <= u8::MAX as i128 {
                values.push(Value::U8(value as u8));
            }
            if value <= u16::MAX as i128 {
                values.push(Value::U16(value as u16));
            }
            if value <= u32::MAX as i128 {
                values.push(Value::U32(value as u32));
            }
            if value <= u64::MAX as i128 {
                values.push(Value::U64(value as u64));
            }
            values.push(Value::U128(value as u128));
        }

        let float = value as f64;
        if float.is_finite() && float as i128 == value {
            values.push(Value::F64(float));
            let float32 = float as f32;
            if (float32 as f64) == float {
                values.push(Value::F32(float32));
            }
        }
    }

    fn push_float_equivalent_values(value: f64, values: &mut Vec<Value>) {
        values.push(Value::F64(value));
        let float32 = value as f32;
        if (float32 as f64) == value || (float32.is_nan() && value.is_nan()) {
            values.push(Value::F32(float32));
        }

        if !value.is_finite()
            || value.fract() != 0.0
            || value < i128::MIN as f64
            || value > i128::MAX as f64
        {
            return;
        }

        let integer = value as i128;
        if integer as f64 == value {
            Self::push_integer_equivalent_values(integer, values);
        }
    }

    fn ordered_f64_bytes(value: f64) -> Vec<u8> {
        let bits = value.to_bits();
        let sortable = if (bits >> 63) == 0 {
            bits | (1 << 63)
        } else {
            !bits
        };
        sortable.to_be_bytes().to_vec()
    }

    pub(crate) fn index_node_payload_field(
        &self,
        txn: &mut RwTxn,
        db: &Database<Bytes, Bytes>,
        schema: &PayloadIndexSchema,
        node_id: u128,
        value: Option<&Value>,
    ) -> Result<(), GraphError> {
        let Some(value) = value else {
            return Ok(());
        };

        match schema {
            PayloadIndexSchema::Keyword => {
                for encoded in Self::payload_keyword_index_keys(value) {
                    db.put(txn, &encoded, &node_id.to_be_bytes())?;
                }
            }
            PayloadIndexSchema::Integer | PayloadIndexSchema::Float => {
                if let Some(number) = Self::payload_numeric_value(value) {
                    db.put(
                        txn,
                        &Self::ordered_f64_bytes(number),
                        &node_id.to_be_bytes(),
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Record one build-delta row for an in-flight LSM payload-index build:
    /// `old_index_key = Some(k)` remembers an index row `(k, node_id)` a
    /// foreground write just deleted (so the drain can re-delete it if the
    /// backfill scan resurrects it); `None` is the marker row ("reindex this
    /// node from its current value"). Replay order is irrelevant.
    ///
    /// Row value encoding: `[seq: u64 BE][old_index_key bytes...]` (empty tail
    /// = marker). The per-process `seq` makes EVERY recorded row unique: a
    /// drain batch deletes exactly the rows its snapshot observed, so a
    /// foreground write landing between the drain's snapshot and its commit —
    /// even one re-recording byte-identical evidence (delete-then-recreate,
    /// value oscillation) — always leaves fresh rows behind that force one
    /// more drain pass. Without the seq, such a write's evidence would share
    /// a composite key with an observed row and be consumed by the drain's
    /// delete, letting the drain's own stale writes go unrepaired.
    fn record_payload_index_build_delta(
        &self,
        w: &mut AnyWrite<'_>,
        db_name: &str,
        node_id: u128,
        old_index_key: Option<&[u8]>,
    ) -> Result<(), GraphError> {
        static PAYLOAD_INDEX_BUILD_DELTA_SEQ: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(1);
        let seq = PAYLOAD_INDEX_BUILD_DELTA_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let old_index_key = old_index_key.unwrap_or(&[]);
        let mut value = Vec::with_capacity(8 + old_index_key.len());
        value.extend_from_slice(&seq.to_be_bytes());
        value.extend_from_slice(old_index_key);
        self.backend
            .put_dup(
                w,
                Namespace::PayloadIndexBuild(db_name),
                &node_id.to_be_bytes(),
                &value,
            )
            .map_err(|e| GraphError::New(e.to_string()))
    }

    /// Foreground index write; records build-delta rows while `handle` is
    /// owned by a build job (LSM online builds only — see
    /// [`Namespace::PayloadIndexBuild`]). Takes the caller's `handle` instead
    /// of looking `field` up: callers hold `payload_indices.read()`, and a
    /// second read of it deadlocks behind a queued writer.
    pub(crate) fn index_node_payload_field_be(
        &self,
        w: &mut AnyWrite<'_>,
        field: &str,
        handle: &PayloadIndexHandle,
        node_id: u128,
        value: Option<&Value>,
    ) -> Result<(), GraphError> {
        let record = handle.is_building();
        self.index_node_payload_field_be_inner(w, field, &handle.schema, node_id, value, record)
    }

    /// [`Self::index_node_payload_field_be`] WITHOUT build-delta recording —
    /// for the build job's own backfill/drain writes, which must not mark
    /// themselves dirty.
    pub(crate) fn index_node_payload_field_be_unrecorded(
        &self,
        w: &mut AnyWrite<'_>,
        field: &str,
        schema: &PayloadIndexSchema,
        node_id: u128,
        value: Option<&Value>,
    ) -> Result<(), GraphError> {
        self.index_node_payload_field_be_inner(w, field, schema, node_id, value, false)
    }

    fn index_node_payload_field_be_inner(
        &self,
        w: &mut AnyWrite<'_>,
        field: &str,
        schema: &PayloadIndexSchema,
        node_id: u128,
        value: Option<&Value>,
        record_build_delta: bool,
    ) -> Result<(), GraphError> {
        let Some(value) = value else {
            return Ok(());
        };
        let db_name = Self::payload_index_db_name(field, schema);
        if record_build_delta && self.backend.kind() == BackendKind::Lsm {
            // Identity marker: the drain re-derives this node's index rows
            // from its then-current value, in the same batch as the node row.
            self.record_payload_index_build_delta(w, &db_name, node_id, None)?;
        }
        match schema {
            PayloadIndexSchema::Keyword => {
                let stored: &[u8] = match value {
                    Value::Array(_) => &[],
                    _ => PAYLOAD_INDEX_PLAIN_ROW,
                };
                for encoded in Self::payload_keyword_index_keys(value) {
                    self.backend
                        .put_dup_stored(
                            w,
                            Namespace::PayloadIndex(&db_name),
                            &encoded,
                            &node_id.to_be_bytes(),
                            stored,
                        )
                        .map_err(|e| GraphError::New(e.to_string()))?;
                }
            }
            PayloadIndexSchema::Integer | PayloadIndexSchema::Float => {
                if let Some(number) = Self::payload_numeric_value(value) {
                    self.backend
                        .put_dup(
                            w,
                            Namespace::PayloadIndex(&db_name),
                            &Self::ordered_f64_bytes(number),
                            &node_id.to_be_bytes(),
                        )
                        .map_err(|e| GraphError::New(e.to_string()))?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn deindex_node_payload_field(
        &self,
        txn: &mut RwTxn,
        db: &Database<Bytes, Bytes>,
        schema: &PayloadIndexSchema,
        node_id: u128,
        value: Option<&Value>,
    ) -> Result<(), GraphError> {
        let Some(value) = value else {
            return Ok(());
        };

        match schema {
            PayloadIndexSchema::Keyword => {
                for encoded in Self::payload_keyword_index_keys(value) {
                    db.delete_one_duplicate(txn, &encoded, &node_id.to_be_bytes())?;
                }
            }
            PayloadIndexSchema::Integer | PayloadIndexSchema::Float => {
                if let Some(number) = Self::payload_numeric_value(value) {
                    db.delete_one_duplicate(
                        txn,
                        &Self::ordered_f64_bytes(number),
                        &node_id.to_be_bytes(),
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Foreground de-index; see [`Self::index_node_payload_field_be`] for
    /// why it takes the caller's `handle`.
    pub(crate) fn deindex_node_payload_field_be(
        &self,
        w: &mut AnyWrite<'_>,
        field: &str,
        handle: &PayloadIndexHandle,
        node_id: u128,
        value: Option<&Value>,
    ) -> Result<(), GraphError> {
        let record = handle.is_building();
        self.deindex_node_payload_field_be_inner(w, field, &handle.schema, node_id, value, record)
    }

    /// [`Self::deindex_node_payload_field_be`] WITHOUT build-delta recording —
    /// for the build job's own backfill/drain writes.
    pub(crate) fn deindex_node_payload_field_be_unrecorded(
        &self,
        w: &mut AnyWrite<'_>,
        field: &str,
        schema: &PayloadIndexSchema,
        node_id: u128,
        value: Option<&Value>,
    ) -> Result<(), GraphError> {
        self.deindex_node_payload_field_be_inner(w, field, schema, node_id, value, false)
    }

    fn deindex_node_payload_field_be_inner(
        &self,
        w: &mut AnyWrite<'_>,
        field: &str,
        schema: &PayloadIndexSchema,
        node_id: u128,
        value: Option<&Value>,
        record_build_delta: bool,
    ) -> Result<(), GraphError> {
        let Some(value) = value else {
            return Ok(());
        };
        let db_name = Self::payload_index_db_name(field, schema);
        let record = record_build_delta && self.backend.kind() == BackendKind::Lsm;
        match schema {
            PayloadIndexSchema::Keyword => {
                for encoded in Self::payload_keyword_index_keys(value) {
                    if record {
                        // Remember the deleted row so the drain can re-delete
                        // it if the backfill scan resurrects it.
                        self.record_payload_index_build_delta(
                            w,
                            &db_name,
                            node_id,
                            Some(&encoded),
                        )?;
                    }
                    self.backend
                        .delete_dup(
                            w,
                            Namespace::PayloadIndex(&db_name),
                            &encoded,
                            &node_id.to_be_bytes(),
                        )
                        .map_err(|e| GraphError::New(e.to_string()))?;
                }
            }
            PayloadIndexSchema::Integer | PayloadIndexSchema::Float => {
                if let Some(number) = Self::payload_numeric_value(value) {
                    let encoded = Self::ordered_f64_bytes(number);
                    if record {
                        self.record_payload_index_build_delta(
                            w,
                            &db_name,
                            node_id,
                            Some(&encoded),
                        )?;
                    }
                    self.backend
                        .delete_dup(
                            w,
                            Namespace::PayloadIndex(&db_name),
                            &encoded,
                            &node_id.to_be_bytes(),
                        )
                        .map_err(|e| GraphError::New(e.to_string()))?;
                }
            }
        }
        Ok(())
    }

    fn payload_numeric_value(value: &Value) -> Option<f64> {
        let number = match value {
            Value::F32(v) => Some(*v as f64),
            Value::F64(v) => Some(*v),
            Value::I8(v) => Some(*v as f64),
            Value::I16(v) => Some(*v as f64),
            Value::I32(v) => Some(*v as f64),
            Value::I64(v) => Some(*v as f64),
            Value::U8(v) => Some(*v as f64),
            Value::U16(v) => Some(*v as f64),
            Value::U32(v) => Some(*v as f64),
            Value::U64(v) => Some(*v as f64),
            Value::U128(v) => Some(*v as f64),
            _ => None,
        };

        number.filter(|value| value.is_finite())
    }
}

impl DBMethods for HelixGraphStorage {
    fn create_secondary_index(&mut self, name: &str) -> Result<(), GraphError> {
        let db = self.with_write_txn(|wtxn| {
            let db = self.lmdb_env()?.create_database(wtxn, Some(name))?;
            let mut secondary_indices: Vec<String> =
                self.secondary_indices.keys().cloned().collect();
            secondary_indices.push(name.to_string());
            self.set_secondary_indices_metadata(wtxn, secondary_indices)?;
            Ok(db)
        })?;
        self.secondary_indices.insert(name.to_string(), Some(db));
        Ok(())
    }

    fn drop_secondary_index(&mut self, name: &str) -> Result<(), GraphError> {
        let db = self
            .secondary_indices
            .get(name)
            .ok_or(GraphError::New(format!(
                "Secondary Index {} not found",
                name
            )))?;
        self.with_write_txn(|wtxn| {
            db.ok_or_else(|| {
                GraphError::StorageError(
                    "LMDB secondary-index DB handle is unavailable on this backend".to_string(),
                )
            })?
            .clear(wtxn)?;
            let secondary_indices: Vec<String> = self
                .secondary_indices
                .keys()
                .filter(|index_name| index_name.as_str() != name)
                .cloned()
                .collect();
            self.set_secondary_indices_metadata(wtxn, secondary_indices)?;
            Ok(())
        })?;
        self.secondary_indices.remove(name);
        Ok(())
    }
}

impl BasicStorageMethods for HelixGraphStorage {
    #[inline(always)]
    fn get_temp_node<'a>(&self, txn: &'a RoTxn, id: &u128) -> Result<&'a [u8], GraphError> {
        match self.lmdb_nodes_db()?.get(&txn, Self::node_key(id))? {
            Some(data) => Ok(data),
            None => Err(GraphError::NodeNotFound),
        }
    }

    #[inline(always)]
    fn get_temp_edge<'a>(&self, txn: &'a RoTxn, id: &u128) -> Result<&'a [u8], GraphError> {
        match self.lmdb_edges_db()?.get(&txn, Self::edge_key(id))? {
            Some(data) => Ok(data),
            None => Err(GraphError::EdgeNotFound),
        }
    }
}

/// Backend-routed (`_be`) variants of the core storage reads (US-004 migration).
/// These read the SAME databases with byte-identical keys (node/edge id as
/// 16-byte big-endian, matching `Database<U128<BE>, Bytes>`) but go through the
/// pluggable `StorageBackend`. They coexist with the heed methods during the
/// call-site migration and read each other's data, so call sites can flip one at
/// a time while the build stays green.
impl HelixGraphStorage {
    #[inline]
    pub fn check_exists_be(&self, r: &AnyRead<'_>, id: &u128) -> Result<bool, GraphError> {
        self.backend
            .get_with(r, Namespace::Nodes, &id.to_be_bytes(), |v| v.is_some())
            .map_err(|e| GraphError::New(e.to_string()))
    }

    #[inline]
    pub fn get_node_be(&self, r: &AnyRead<'_>, id: &u128) -> Result<Node, GraphError> {
        #[cfg(test)]
        NODE_READS_FOR_TEST.with(|reads| reads.set(reads.get() + 1));
        self.backend
            .get_with(r, Namespace::Nodes, &id.to_be_bytes(), |v| match v {
                Some(data) => SerializedNode::decode_node(data, *id),
                None => Err(GraphError::NodeNotFound),
            })
            .map_err(|e| GraphError::New(e.to_string()))?
    }

    /// [`Self::get_node_be`] for each of `ids` in one batched read, in order;
    /// `None` for an id with no node, or with one that does not decode. On an
    /// LSM store the reads are in flight together, so the waits of cold rows
    /// overlap instead of adding up. `Ok(None)`: this backend has no batched
    /// read and the caller reads one node at a time.
    pub(crate) fn get_nodes_many_be(
        &self,
        r: &AnyRead<'_>,
        ids: &[u128],
    ) -> Result<Option<Vec<Option<Node>>>, BackendError> {
        Ok(self.get_nodes_many_checked_be(r, ids)?.map(|rows| {
            rows.into_iter()
                .map(|row| row.and_then(Result::ok))
                .collect()
        }))
    }

    /// [`Self::get_nodes_many_be`] with a node that does not decode told
    /// apart from an id with no node: `Some(Err(_))`, where a caller that
    /// read one node at a time had [`Self::get_node_be`]'s error.
    pub(crate) fn get_nodes_many_checked_be(
        &self,
        r: &AnyRead<'_>,
        ids: &[u128],
    ) -> Result<Option<Vec<Option<Result<Node, GraphError>>>>, BackendError> {
        let keys: Vec<Vec<u8>> = ids.iter().map(|id| id.to_be_bytes().to_vec()).collect();
        let rows = match (&*self.backend, r) {
            (AnyBackend::Lsm(writer), AnyRead::Lsm(read)) => {
                writer.collect_values_many_with(read, Namespace::Nodes, &keys)?
            }
            (AnyBackend::LsmReader(reader), AnyRead::LsmReader(snapshot)) => {
                reader.collect_values_many_with_at(snapshot.as_ref(), Namespace::Nodes, &keys)?
            }
            _ => return Ok(None),
        };
        #[cfg(test)]
        NODE_READS_FOR_TEST.with(|reads| reads.set(reads.get() + ids.len()));
        Ok(Some(
            ids.iter()
                .zip(rows)
                .map(|(id, row)| row.map(|bytes| SerializedNode::decode_node(&bytes, *id)))
                .collect(),
        ))
    }

    #[inline]
    pub fn get_edge_be(&self, r: &AnyRead<'_>, id: &u128) -> Result<Edge, GraphError> {
        self.backend
            .get_with(r, Namespace::Edges, &id.to_be_bytes(), |v| match v {
                Some(data) => SerializedEdge::decode_edge(data, *id),
                None => Err(GraphError::EdgeNotFound),
            })
            .map_err(|e| GraphError::New(e.to_string()))?
    }

    /// Build a backend read handle (`AnyRead`) from an existing heed read txn,
    /// for the traversal field flip. On LMDB this borrows the txn (zero-copy,
    /// resize-safe when the txn came from `begin_resize_safe_read_txn`); on LSM
    /// it ignores the txn and opens a SlateDB snapshot. Callers (query entry
    /// points and tests) hold the returned handle and pass `&handle` to
    /// `G::new`, so the whole traversal shares ONE snapshot.
    #[inline]
    pub fn read_view<'a>(&self, txn: &'a RoTxn<'a>) -> AnyRead<'a> {
        self.backend.read_borrowed(txn)
    }

    /// Begin a resize-safe write batch view for a whole write query (field flip).
    ///
    /// LMDB: acquires the writer gate + a tracked `RwTxn` (resize permit held by
    /// the retained wrapper) and wraps the txn as `AnyWrite::Lmdb`. LSM: opens a
    /// SlateDB write batch. Callers thread `view.write_mut()` into `G::new_mut`
    /// and finish with `view.commit()`, so the whole query is ONE atomic batch.
    pub fn begin_write_view(&self) -> Result<WriteView<'_>, GraphError> {
        match self.backend.kind() {
            BackendKind::Lmdb => {
                let (gate, mut resize) = self.begin_resize_safe_write_txn()?;
                let rwtxn = resize.take_txn();
                Ok(WriteView {
                    storage: self,
                    _gate: Some(gate),
                    _resize: Some(resize),
                    write: Some(AnyWrite::from_lmdb_txn(rwtxn)),
                })
            }
            BackendKind::Lsm => {
                let write = self
                    .backend
                    .begin_write()
                    .map_err(|e| GraphError::New(e.to_string()))?;
                Ok(WriteView {
                    storage: self,
                    _gate: None,
                    _resize: None,
                    write: Some(write),
                })
            }
        }
    }

    /// Map a backend error to a `GraphError`, marking the collection degraded
    /// when the failure is fatal (epoch fencing / CAS conflict, corruption, or
    /// clearly-fatal I/O). This is the backend-routed analogue of the
    /// `mark_collection_degraded` calls the heed txn helpers make on LMDB
    /// errors, so a fenced or corrupted LSM collection is quarantined the same
    /// way an `MDB_PAGE_NOTFOUND`/`MDB_CORRUPTED` LMDB collection is.
    pub(crate) fn backend_error(&self, error: BackendError, context: &'static str) -> GraphError {
        let err = graph_error_from_backend_error(error);
        self.mark_collection_degraded(&err, context);
        err
    }

    /// Run `f` with a backend read snapshot — the backend-routed analogue of
    /// `with_read_txn`. Call sites flip from
    /// `with_read_txn(|t| self.get_node(t, id))` to
    /// `with_read_backend(|r| self.get_node_be(r, id))` one at a time.
    #[inline]
    pub fn with_read_backend<F, T>(&self, f: F) -> Result<T, GraphError>
    where
        F: FnOnce(&AnyRead<'_>) -> Result<T, GraphError>,
    {
        self.ensure_not_degraded()?;
        let r = self
            .backend
            .begin_read()
            .map_err(|e| self.backend_error(e, "with_read_backend"))?;
        let result = f(&r);
        if let Err(error) = &result {
            self.mark_collection_degraded(error, "with_read_backend");
        }
        result
    }

    /// Run `f` with a backend write batch, committing on success — the
    /// backend-routed analogue of `with_write_txn`.
    #[inline]
    pub fn with_write_backend<F, T>(&self, f: F) -> Result<T, GraphError>
    where
        F: FnOnce(&mut AnyWrite<'_>) -> Result<T, GraphError>,
    {
        self.ensure_not_degraded()?;
        let _gate = if self.backend.kind() == BackendKind::Lsm {
            None
        } else {
            Some(self.lock_write_txn_gate_for("backend_write")?)
        };
        let mut w = self
            .backend
            .begin_write()
            .map_err(|e| self.backend_error(e, "with_write_backend_begin"))?;
        let out = match f(&mut w) {
            Ok(out) => out,
            Err(e) => {
                self.mark_collection_degraded(&e, "with_write_backend_mutation");
                return Err(e);
            }
        };
        self.backend
            .commit(w)
            .map_err(|e| self.backend_error(e, "with_write_backend_commit"))?;
        // Mirror the heed `with_write_txn` path: refresh the in-memory metadata
        // snapshot after a successful commit so `collection_stats` /
        // `metadata_snapshot()` reflect the node/edge/vector counters this batch
        // just bumped in the backend. Without this the LSM write path persists
        // the counters durably but `collection_stats` keeps reading a stale
        // snapshot (e.g. node_count=0/edge_count=0 right after a fresh ingest).
        self.refresh_metadata_snapshot_best_effort();
        Ok(out)
    }

    /// Like [`with_write_backend`] but commits the batch BUFFERED — visible to
    /// later reads, NOT yet durable on the object store (group-commit, #1). The
    /// caller MUST invoke `self.backend.flush_durable()` before acking so many
    /// buffered commits collapse to ONE durable flush while never acking a
    /// non-durable write. On LMDB `commit_buffered` is a fully-durable commit, so
    /// this is identical to [`with_write_backend`] there.
    #[inline]
    pub fn with_write_backend_buffered<F, T>(&self, f: F) -> Result<T, GraphError>
    where
        F: FnOnce(&mut AnyWrite<'_>) -> Result<T, GraphError>,
    {
        self.ensure_not_degraded()?;
        let _gate = if self.backend.kind() == BackendKind::Lsm {
            None
        } else {
            Some(self.lock_write_txn_gate_for("backend_write_buffered")?)
        };
        let mut w = self
            .backend
            .begin_write()
            .map_err(|e| self.backend_error(e, "with_write_backend_buffered_begin"))?;
        let out = match f(&mut w) {
            Ok(out) => out,
            Err(e) => {
                self.mark_collection_degraded(&e, "with_write_backend_buffered_mutation");
                return Err(e);
            }
        };
        self.backend
            .commit_buffered(w)
            .map_err(|e| self.backend_error(e, "with_write_backend_buffered_commit"))?;
        // Buffered commits are immediately visible to backend reads (group-commit
        // makes them durable later), so the snapshot refresh here observes this
        // batch's counters — same post-commit refresh the heed `with_write_txn`
        // and `with_write_backend` paths perform, keeping `collection_stats`
        // honest after LSM ingest/points batches.
        self.refresh_metadata_snapshot_best_effort();
        Ok(out)
    }

    /// Collect the full edge records incident to `node` on the given adjacency
    /// namespace (`OutEdges` / `InEdges`), reading through the live write txn
    /// (read-your-writes). Backend-`_raw` twin of `collect_incident_edges`,
    /// mirroring `drop_node`'s `prefix_iter` over the DUP_SORT adjacency DB.
    ///
    /// The on-disk keys are 20 bytes (`[node:16 | label_hash:4]`); scanning with
    /// the node's 16-byte id as a prefix therefore covers every label variant for
    /// that node, and the DUP_SORT scan yields one item per dup value. Each value
    /// is unpacked to its edge id, then the edge bytes are decoded. Edges whose
    /// row is missing are skipped, exactly as the heed path's
    /// `if let Some(edge_data)` guard.
    ///
    /// Takes the live `&RwTxn` (not a separate read snapshot): the `_raw` scan
    /// reads through the write txn so it observes this operation's own buffered
    /// writes (read-your-writes), matching the heed `drop_node` which scans inside
    /// its `&mut RwTxn`.
    fn collect_incident_edges_raw(
        &self,
        txn: &RwTxn,
        ns: Namespace<'_>,
        node: &u128,
    ) -> Result<Vec<Edge>, GraphError> {
        use super::backend::KeyRange;

        let mut edge_ids: Vec<u128> = Vec::new();
        let mut decode_err: Option<GraphError> = None;
        self.backend
            .scan_heed(txn, ns, KeyRange::prefix(&node.to_be_bytes()), |_k, v| {
                match Self::unpack_adj_edge_data(v) {
                    Ok((_peer, edge_id)) => {
                        edge_ids.push(edge_id);
                        true
                    }
                    Err(e) => {
                        decode_err = Some(e);
                        false
                    }
                }
            })
            .map_err(|e| GraphError::New(e.to_string()))?;
        if let Some(e) = decode_err {
            return Err(e);
        }

        let mut edges = Vec::with_capacity(edge_ids.len());
        for edge_id in edge_ids {
            let edge = self
                .backend
                .get_with_heed(txn, Namespace::Edges, &edge_id.to_be_bytes(), |v| {
                    v.map(|data| SerializedEdge::decode_edge(data, edge_id))
                })
                .map_err(|e| GraphError::New(e.to_string()))?;
            if let Some(edge) = edge {
                edges.push(edge?);
            }
        }
        Ok(edges)
    }
}

impl StorageMethods for HelixGraphStorage {
    #[inline(always)]
    fn check_exists(&self, txn: &RoTxn, id: &u128) -> Result<bool, GraphError> {
        // Routed through the backend seam (heed-txn scaffold). Byte-identical to
        // the prior nodes_db.get(...).is_some() on LMDB.
        self.backend
            .get_with_heed(txn, Namespace::Nodes, &id.to_be_bytes(), |v| v.is_some())
            .map_err(|e| GraphError::New(e.to_string()))
    }

    #[inline(always)]
    fn get_node(&self, txn: &RoTxn, id: &u128) -> Result<Node, GraphError> {
        // Routed through the backend seam (heed-txn scaffold). Byte-identical to
        // the prior nodes_db.get on LMDB: Namespace::Nodes -> nodes_db and
        // id.to_be_bytes() == node_key(id) (U128<BE> identity, proven by the
        // get_node_be_matches_get_node equivalence test).
        self.backend
            .get_with_heed(txn, Namespace::Nodes, &id.to_be_bytes(), |v| match v {
                Some(data) => SerializedNode::decode_node(data, *id),
                None => Err(GraphError::NodeNotFound),
            })
            .map_err(|e| GraphError::New(e.to_string()))?
    }

    #[inline(always)]
    fn get_edge(&self, txn: &RoTxn, id: &u128) -> Result<Edge, GraphError> {
        // Routed through the backend seam (heed-txn scaffold). Byte-identical to
        // the prior edges_db.get on LMDB: Namespace::Edges -> edges_db and
        // id.to_be_bytes() == edge_key(id) (U128<BE> identity).
        self.backend
            .get_with_heed(txn, Namespace::Edges, &id.to_be_bytes(), |v| match v {
                Some(data) => SerializedEdge::decode_edge(data, *id),
                None => Err(GraphError::EdgeNotFound),
            })
            .map_err(|e| GraphError::New(e.to_string()))?
    }

    fn get_node_by_secondary_index(
        &self,
        txn: &RoTxn,
        index: &str,
        key: &Value,
    ) -> Result<Node, GraphError> {
        // Preserve the "index not configured" error exactly (checked on the
        // in-memory map), then route the value read through the seam:
        // Namespace::SecondaryIndex(index) resolves to the same DB the
        // secondary_indices map holds (proven by create_node's index writes).
        // Byte-identical key (stable_index_key_for_value) + value (node id as
        // 16-byte big-endian, decoded by get_u128_from_bytes inside the visitor).
        if !self.secondary_indices.contains_key(index) {
            return Err(GraphError::New(format!(
                "Secondary Index {} not found",
                index
            )));
        }
        let node_id = self
            .backend
            .get_with_heed(
                txn,
                Namespace::SecondaryIndex(index),
                &Self::stable_index_key_for_value(key)?,
                |v| match v {
                    Some(data) => Self::get_u128_from_bytes(data),
                    None => Err(GraphError::NodeNotFound),
                },
            )
            .map_err(|e| GraphError::New(e.to_string()))??;
        self.get_node(txn, &node_id)
    }

    fn drop_node(&self, txn: &mut RwTxn, id: &u128) -> Result<(), GraphError> {
        if self.backend.kind() == BackendKind::Lsm {
            return self.with_write_backend(|w| self.drop_node_be(w, id));
        }

        // Snapshot the node row first (read-your-writes through the write txn) so
        // its multi-index / payload-index entries can be de-indexed after the
        // delete, and so the node counter is gated on whether the row existed.
        // Same decode + key (node_key == id.to_be_bytes()) as the prior
        // nodes_db.get; routed through the seam's get_with_raw.
        let existing_node = self
            .backend
            .get_with_heed(txn, Namespace::Nodes, &id.to_be_bytes(), |v| {
                v.map(|data| SerializedNode::decode_node(data, *id))
            })
            .map_err(|e| GraphError::New(e.to_string()))?
            .transpose()?;

        // Collect outgoing then incoming edge records by prefix-scanning the
        // adjacency DBs under this node's id, mirroring drop_node's two
        // prefix_iter passes + edge-data lookups (now via collect_incident_edges_raw).
        let out_edges = self.collect_incident_edges_raw(txn, Namespace::OutEdges, id)?;
        let in_edges = self.collect_incident_edges_raw(txn, Namespace::InEdges, id)?;

        let removed_edge_ids = out_edges
            .iter()
            .chain(in_edges.iter())
            .map(|edge| edge.id)
            .collect::<std::collections::HashSet<_>>();

        // Delete all related edge data: edge bytes + BOTH adjacency sides.
        //
        // FIX (matches the drop_edge port): the prior out/in_edges_db.delete(key)
        // removed EVERY dup under the (node|label) key. On the cross-endpoint
        // deletes that wiped adjacency belonging to OTHER nodes — dropping node A
        // deleted every incoming edge to a peer P that shared A's edge label, not
        // just A→P. delete_dup_raw removes only this edge's packed value, leaving
        // the peers' unrelated same-label adjacency intact.
        for edge in out_edges.iter().chain(in_edges.iter()) {
            let label_hash = hash_label(&edge.label, None);
            self.backend
                .delete_heed(txn, Namespace::Edges, &edge.id.to_be_bytes())
                .map_err(|e| GraphError::New(e.to_string()))?;
            self.backend
                .delete_dup_heed(
                    txn,
                    Namespace::OutEdges,
                    &Self::out_edge_key(&edge.from_node, &label_hash),
                    &Self::pack_edge_data(&edge.to_node, &edge.id),
                )
                .map_err(|e| GraphError::New(e.to_string()))?;
            self.backend
                .delete_dup_heed(
                    txn,
                    Namespace::InEdges,
                    &Self::in_edge_key(&edge.to_node, &label_hash),
                    &Self::pack_edge_data(&edge.from_node, &edge.id),
                )
                .map_err(|e| GraphError::New(e.to_string()))?;
            // Cascaded edges must leave the edge-path index too (as drop_edge
            // does), or delete-by-path later resolves to a missing edge.
            self.delete_edge_paths(txn, edge)?;
        }

        // Delete node bytes, then de-index — only when the row existed. Because
        // nothing between the snapshot above and here mutates the node row,
        // `existing_node.is_some()` equals heed `nodes_db.delete`'s returned bool,
        // so it is the same counter/de-index gate as before. Multi-index and
        // payload-index de-indexing stay on the heed handles (the backend
        // `Namespace` enum cannot address payload-index DBs).
        self.backend
            .delete_heed(txn, Namespace::Nodes, &id.to_be_bytes())
            .map_err(|e| GraphError::New(e.to_string()))?;
        if let Some(node) = &existing_node {
            // Single-value secondary indices written by create_node. Delete only
            // an entry that still points at THIS node (the index is unique by
            // value, so another node may own the key now).
            let id_bytes = id.to_be_bytes();
            for index in self.secondary_indices.keys() {
                if let Some(value) = node.properties.get(index) {
                    let key = Self::stable_index_key_for_value(value)?;
                    let points_to_node = self
                        .backend
                        .get_for_update_heed(txn, Namespace::SecondaryIndex(index), &key, |v| {
                            v.is_some_and(|bytes| bytes == id_bytes.as_slice())
                        })
                        .map_err(|e| GraphError::New(e.to_string()))?;
                    if points_to_node {
                        self.backend
                            .delete_heed(txn, Namespace::SecondaryIndex(index), &key)
                            .map_err(|e| GraphError::New(e.to_string()))?;
                    }
                }
            }
            // Multi-index de-indexing routed through the seam: Namespace::MultiIndex
            // (idx_name) resolves to the same midx_{name} DUP_SORT DB the
            // multi_indices map holds; delete_dup_raw maps 1:1 to heed
            // delete_one_duplicate (removes the single (key,value) duplicate),
            // byte-identical key + value (node id as 16-byte big-endian).
            for idx_name in self.multi_indices.keys() {
                if let Some(value) = Self::payload_value_for_key(&node.properties, idx_name) {
                    let key = Self::stable_index_key_for_value(value)?;
                    self.backend
                        .delete_dup_heed(
                            txn,
                            Namespace::MultiIndex(idx_name),
                            &key,
                            &id.to_be_bytes(),
                        )
                        .map_err(|e| GraphError::New(e.to_string()))?;
                }
            }
            let payload_indices = self
                .payload_indices
                .read()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            for (idx_name, handle) in payload_indices.iter() {
                if !handle.accepts_writes() {
                    continue;
                }
                self.deindex_node_payload_field(
                    txn,
                    handle.lmdb_db()?,
                    &handle.schema,
                    *id,
                    Self::payload_value_for_key(&node.properties, idx_name),
                )?;
            }
            self.adjust_metadata_counter(txn, MetadataCounter::Nodes, -1)?;
        }
        if !removed_edge_ids.is_empty() {
            self.adjust_metadata_counter(
                txn,
                MetadataCounter::Edges,
                -(removed_edge_ids.len() as i64),
            )?;
        }

        Ok(())
    }

    fn drop_edge(&self, txn: &mut RwTxn, edge_id: &u128) -> Result<(), GraphError> {
        if self.backend.kind() == BackendKind::Lsm {
            return self.with_write_backend(|w| self.drop_edge_be(w, edge_id));
        }

        // Load + decode the edge first (read-your-writes through the write txn),
        // EdgeNotFound if absent — same as the prior edges_db.get(&txn, ..).
        let edge = match self
            .backend
            .get_for_update_heed(txn, Namespace::Edges, &edge_id.to_be_bytes(), |v| {
                v.map(|b| b.to_vec())
            })
            .map_err(|e| GraphError::New(e.to_string()))?
        {
            Some(bytes) => SerializedEdge::decode_edge(&bytes, *edge_id)?,
            None => return Err(GraphError::EdgeNotFound),
        };
        let label_hash = hash_label(&edge.label, None);

        // edge_path_idx deletes (kept on its own method).
        self.delete_edge_paths(txn, &edge)?;

        // Edge bytes.
        self.backend
            .delete_heed(txn, Namespace::Edges, &edge_id.to_be_bytes())
            .map_err(|e| GraphError::New(e.to_string()))?;

        // Adjacency: delete ONLY this edge's packed dup value. FIX (user-approved
        // behavior change): the prior out/in_edges_db.delete(key) removed EVERY
        // dup under the (from|label) / (to|label) key, wiping adjacency for ALL
        // same-label edges of the node. delete_dup_raw removes just this
        // (key,value) pair, matching the corrected drop_edge_be.
        self.backend
            .delete_dup_heed(
                txn,
                Namespace::OutEdges,
                &Self::out_edge_key(&edge.from_node, &label_hash),
                &Self::pack_edge_data(&edge.to_node, edge_id),
            )
            .map_err(|e| GraphError::New(e.to_string()))?;
        self.backend
            .delete_dup_heed(
                txn,
                Namespace::InEdges,
                &Self::in_edge_key(&edge.to_node, &label_hash),
                &Self::pack_edge_data(&edge.from_node, edge_id),
            )
            .map_err(|e| GraphError::New(e.to_string()))?;

        self.adjust_metadata_counter(txn, MetadataCounter::Edges, -1)?;

        Ok(())
    }

    fn create_node(
        &self,
        txn: &mut RwTxn,
        label: &str,
        properties: impl IntoIterator<Item = (String, Value)>,
        secondary_indices: Option<&[String]>,
        id: Option<u128>,
    ) -> Result<Node, GraphError> {
        if self.backend.kind() == BackendKind::Lsm {
            return self.with_write_backend(|w| {
                self.create_node_be(w, label, properties, secondary_indices, id)
            });
        }

        let node = Node {
            id: id.unwrap_or(v6_uuid()),
            label: label.to_string(),
            properties: HashMap::from_iter(properties),
        };

        // Store node data through the seam (plain put_raw == heed nodes_db.put on
        // the same DBI; key id.to_be_bytes() == node_key). Mirrors the
        // equivalence-tested create_node_be, but on the threaded &mut RwTxn.
        self.backend
            .put_heed(
                txn,
                Namespace::Nodes,
                &node.id.to_be_bytes(),
                &SerializedNode::encode_node(&node)?,
            )
            .map_err(|e| GraphError::New(e.to_string()))?;

        // Single-value secondary indices. Same checks + identical error messages
        // as before; Namespace::SecondaryIndex(index) resolves to the same DB the
        // secondary_indices map holds (proven by create_node_be_index_validation).
        for index in secondary_indices.unwrap_or(&[]) {
            if !self.secondary_indices.contains_key(index) {
                return Err(GraphError::New(format!(
                    "Secondary Index {} not found",
                    index
                )));
            }
            let key = match node.check_property(index) {
                Some(value) => value,
                None => {
                    return Err(GraphError::New(format!(
                        "Secondary Index {} not found",
                        index
                    )))
                }
            };
            self.backend
                .put_heed(
                    txn,
                    Namespace::SecondaryIndex(index),
                    &Self::stable_index_key_for_value(key)?,
                    &node.id.to_be_bytes(),
                )
                .map_err(|e| GraphError::New(e.to_string()))?;
        }

        self.adjust_metadata_counter(txn, MetadataCounter::Nodes, 1)?;
        Ok(node)
    }

    fn create_edge(
        &self,
        txn: &mut RwTxn,
        label: &str,
        from_node: &u128,
        to_node: &u128,
        properties: impl IntoIterator<Item = (String, Value)>,
    ) -> Result<Edge, GraphError> {
        if self.backend.kind() == BackendKind::Lsm {
            return self.with_write_backend(|w| {
                self.create_edge_be(w, label, from_node, to_node, properties)
            });
        }

        // Check if nodes exist

        // if self.check_exists(from_node)? || self.check_exists(to_node)? {
        //     return Err(GraphError::New(
        //         "One or both nodes do not exist".to_string(),
        //     ));
        // }
        // Endpoint existence check through the seam, read-your-writes via the
        // write txn — byte-identical to the prior nodes_db.get(txn, ..) checks
        // (and intentionally keeps read-your-writes rather than the committed
        // snapshot node_missing_be that create_edge_be uses).
        if self
            .backend
            .get_for_update_heed(txn, Namespace::Nodes, &from_node.to_be_bytes(), |v| {
                v.is_none()
            })
            .map_err(|e| GraphError::New(e.to_string()))?
            || self
                .backend
                .get_for_update_heed(txn, Namespace::Nodes, &to_node.to_be_bytes(), |v| {
                    v.is_none()
                })
                .map_err(|e| GraphError::New(e.to_string()))?
        {
            return Err(GraphError::NodeNotFound);
        }

        let edge = Edge {
            id: v6_uuid(),
            label: label.to_string(),
            from_node: *from_node,
            to_node: *to_node,
            properties: HashMap::from_iter(properties),
        };

        // Edge bytes (plain put_raw == heed edges_db.put; key == edge_key).
        self.backend
            .put_heed(
                txn,
                Namespace::Edges,
                &edge.id.to_be_bytes(),
                &SerializedEdge::encode_edge(&edge)?,
            )
            .map_err(|e| GraphError::New(e.to_string()))?;

        let label_hash = hash_label(label, None);

        // Adjacency (DUP_SORT): resolve_write opens OutEdges/InEdges with
        // DatabaseFlags::DUP_SORT, so put_dup_raw ADDS a duplicate (never
        // overwrites) — byte-identical to the heed out/in_edges_db.put.
        self.backend
            .put_dup_heed(
                txn,
                Namespace::OutEdges,
                &Self::out_edge_key(from_node, &label_hash),
                &Self::pack_edge_data(to_node, &edge.id),
            )
            .map_err(|e| GraphError::New(e.to_string()))?;
        self.backend
            .put_dup_heed(
                txn,
                Namespace::InEdges,
                &Self::in_edge_key(to_node, &label_hash),
                &Self::pack_edge_data(from_node, &edge.id),
            )
            .map_err(|e| GraphError::New(e.to_string()))?;

        self.index_edge_paths(txn, &edge)?;
        self.adjust_metadata_counter(txn, MetadataCounter::Edges, 1)?;
        Ok(edge)
    }
}

// impl SearchMethods for HelixGraphStorage {
//     fn shortest_path(
//         &self,
//         txn: &RoTxn,
//         edge_label: &str,
//         from_id: &u128,
//         to_id: &u128,
//     ) -> Result<(Vec<Node>, Vec<Edge>), GraphError> {
//         let mut queue = VecDeque::with_capacity(32);
//         let mut visited = HashSet::with_capacity(64);
//         let mut parent: HashMap<u128, (u128, Edge)> = HashMap::with_capacity(32);
//         queue.push_back(*from_id);
//         visited.insert(*from_id);

//         let reconstruct_path = |parent: &HashMap<u128, (u128, Edge)>,
//                                 start_id: &u128,
//                                 end_id: &u128|
//          -> Result<(Vec<Node>, Vec<Edge>), GraphError> {
//             let mut nodes = Vec::with_capacity(parent.len());
//             let mut edges = Vec::with_capacity(parent.len() - 1);

//             let mut current = end_id;

//             while current != start_id {
//                 nodes.push(self.get_node(txn, current)?);

//                 let (prev_node, edge) = &parent[current];
//                 edges.push(edge.clone());
//                 current = prev_node;
//             }

//             nodes.push(self.get_node(txn, start_id)?);

//             Ok((nodes, edges))
//         };

//         while let Some(current_id) = queue.pop_front() {
//             let out_prefix = Self::out_edge_key(&current_id, edge_label, None);
//             let iter = self
//                 .out_edges_db
//                 .lazily_decode_data()
//                 .prefix_iter(&txn, &out_prefix)?;

//             for result in iter {
//                 let (key, value) = result?;
//                 let to_node = Self::get_u128_from_bytes(&key[out_prefix.len()..])?;

//                 if !visited.contains(&to_node) {
//                     visited.insert(to_node);
//                     let edge_id = decode_u128!(value);
//                     let edge = self.get_edge(&txn, &edge_id)?;
//                     parent.insert(to_node, (current_id, edge));

//                     if to_node == *to_id {
//                         return reconstruct_path(&parent, from_id, to_id);
//                     }

//                     queue.push_back(to_node);
//                 }
//             }
//         }

//         Err(GraphError::from(format!(
//             "No path found between {} and {}",
//             from_id, to_id
//         )))
//     }

//     fn shortest_mutual_path(
//         &self,
//         txn: &RoTxn,
//         edge_label: &str,
//         from_id: &u128,
//         to_id: &u128,
//     ) -> Result<(Vec<Node>, Vec<Edge>), GraphError> {
//         let mut queue = VecDeque::with_capacity(32);
//         let mut visited = HashSet::with_capacity(64);
//         let mut parent = HashMap::with_capacity(32);

//         queue.push_back(*from_id);
//         visited.insert(*from_id);

//         let reconstruct_path = |parent: &HashMap<u128, (u128, Edge)>,
//                                 start_id: &u128,
//                                 end_id: &u128|
//          -> Result<(Vec<Node>, Vec<Edge>), GraphError> {
//             let mut nodes = Vec::with_capacity(parent.len());
//             let mut edges = Vec::with_capacity(parent.len() - 1);

//             let mut current = end_id;

//             while current != start_id {
//                 nodes.push(self.get_node(txn, current)?);

//                 let (prev_node, edge) = &parent[current];
//                 edges.push(edge.clone());
//                 current = prev_node;
//             }
//             nodes.push(self.get_node(txn, start_id)?);
//             Ok((nodes, edges))
//         };

//         while let Some(current_id) = queue.pop_front() {
//             let out_prefix = Self::out_edge_key(&current_id, edge_label, None);
//             let iter = self
//                 .out_edges_db
//                 .lazily_decode_data()
//                 .prefix_iter(&txn, &out_prefix)?;

//             for result in iter {
//                 let (key, value) = result?;
//                 let to_node = Self::get_u128_from_bytes(&key[out_prefix.len()..])?;

//                 println!("To Node: {}", to_node);
//                 println!("Current: {}", current_id);
//                 // Check if there's a reverse edge
//                 let reverse_edge_key = Self::out_edge_key(&to_node, edge_label, Some(&current_id));

//                 let has_reverse_edge = self.out_edges_db.as_ref().unwrap().get(&txn, &reverse_edge_key)?.is_some();

//                 // Only proceed if there's a mutual connection
//                 if has_reverse_edge && !visited.contains(&to_node) {
//                     visited.insert(to_node);
//                     let edge_id = decode_u128!(value);
//                     let edge = self.get_edge(&txn, &edge_id)?;
//                     parent.insert(to_node, (current_id, edge));

//                     if to_node == *to_id {
//                         return reconstruct_path(&parent, from_id, to_id);
//                     }

//                     queue.push_back(to_node);
//                 }
//             }
//         }

//         Err(GraphError::from(format!(
//             "No mutual path found between {} and {}",
//             from_id, to_id
//         )))
//     }
// }
