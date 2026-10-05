use crate::helix_engine::{
    storage_core::backend::{BackendKind, KeyRange, Namespace, SegmentDb, StorageBackend},
    storage_core::backend_any::{AnyBackend, AnyRead, AnyWrite},
    types::VectorError,
    vector_core::{
        arena_heap::ArenaHeap,
        hnsw::HNSW,
        named_vectors::DistanceMetric,
        spindle::{
            decode_vector, encode_vector, prepare_query, project_for_search, score_encoded,
            ApproximateInnerProduct, PreparedSpindleQuery, SpindleConfig, SpindleMode,
        },
        vector::HVector,
    },
};
use crate::protocol::value::Value;
use bumpalo::Bump;
use heed3::{
    types::{Bytes, Unit},
    Database, Env, RoTxn, RwTxn,
};
use itertools::Itertools;
use rand::prelude::Rng;
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, HashSet},
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AtomicOrdering},
        Arc, Condvar, Mutex, RwLock as StdRwLock,
    },
};

use lru::LruCache;

/// Global semaphore that limits concurrent HNSW builds.
/// Each build loads all vectors + adjacency into memory (~150MB per 16k×768d segment).
/// Unbounded concurrency causes OOM under heavy ingest. Permits scale with CPU count
/// since each build uses rayon internally.
struct BuildSemaphore {
    mu: Mutex<usize>,
    cv: Condvar,
    permits: usize,
}

impl BuildSemaphore {
    fn new(permits: usize) -> Self {
        Self {
            mu: Mutex::new(0),
            cv: Condvar::new(),
            permits,
        }
    }

    fn acquire(&self) -> Result<(), VectorError> {
        let started = std::time::Instant::now();
        let mut warned = false;
        let timeout = hnsw_build_permit_timeout();
        // Recover from poison instead of panicking: the guarded state is just
        // the active-build count, and a panicking build still releases its
        // permit via `BuildPermit::drop`, so the count stays consistent. An
        // `.unwrap()` here would turn one crashed build into a panic cascade
        // across every subsequent build that needs a permit.
        let mut active = self.mu.lock().unwrap_or_else(|p| p.into_inner());
        while *active >= self.permits {
            // Wait in 5s slices so we can emit a tracing warning when a
            // build queue stalls. `HELIX_HNSW_BUILD_PERMIT_TIMEOUT_MS=0`
            // restores the old unbounded wait; otherwise timed-out
            // background jobs fail and get retried by the optimizer.
            let wait_for = timeout
                .map(|limit| limit.saturating_sub(started.elapsed()))
                .filter(|remaining| !remaining.is_zero())
                .map(|remaining| remaining.min(std::time::Duration::from_secs(5)))
                .unwrap_or_else(|| std::time::Duration::from_secs(5));
            let (next, _) = self
                .cv
                .wait_timeout(active, wait_for)
                .unwrap_or_else(|p| p.into_inner());
            active = next;
            if let Some(limit) = timeout {
                if started.elapsed() >= limit && *active >= self.permits {
                    metrics::histogram!(
                        "helix_hnsw_build_permit_wait_ms",
                        "outcome" => "timeout"
                    )
                    .record(started.elapsed().as_secs_f64() * 1000.0);
                    metrics::counter!("helix_hnsw_build_permit_timeout_total").increment(1);
                    return Err(VectorError::VectorCoreError(format!(
                        "HNSW build permit timed out after {} ms",
                        limit.as_millis()
                    )));
                }
            }
            if !warned && started.elapsed() >= std::time::Duration::from_secs(30) {
                tracing::warn!(
                    waited_secs = started.elapsed().as_secs(),
                    permits = self.permits,
                    active = *active,
                    "HNSW build permit acquire is slow"
                );
                warned = true;
            }
        }
        *active += 1;
        metrics::histogram!(
            "helix_hnsw_build_permit_wait_ms",
            "outcome" => "acquired"
        )
        .record(started.elapsed().as_secs_f64() * 1000.0);
        metrics::gauge!("helix_hnsw_builds_in_flight").set(*active as f64);
        Ok(())
    }

    fn release(&self) {
        // Poison-tolerant for the same reason as `acquire`: release runs from
        // `BuildPermit::drop` during panic unwinding, and panicking inside a
        // Drop while unwinding aborts the whole process.
        let mut active = self.mu.lock().unwrap_or_else(|p| p.into_inner());
        *active = active.saturating_sub(1);
        metrics::gauge!("helix_hnsw_builds_in_flight").set(*active as f64);
        self.cv.notify_one();
    }
}

/// RAII guard for `BUILD_SEMAPHORE` so the permit is always released, even
/// if the build path panics. Without this, a panic inside `build_index_inner`
/// would permanently consume a permit and starve future builds.
pub(crate) struct BuildPermit;
impl Drop for BuildPermit {
    fn drop(&mut self) {
        BUILD_SEMAPHORE.release();
    }
}
pub(crate) fn acquire_build_permit() -> Result<BuildPermit, VectorError> {
    // Concurrency semaphore ONLY — no memory-pressure deferral here. Several
    // callers hold a live LMDB txn (`build_index_from_flat`,
    // `build_dense_segment`, segment seal, legacy merge), and LMDB is
    // single-writer per env: sleeping here stalls every write and map-resize
    // for the collection, including the flushes that would relieve pressure.
    // Pressure deferral happens at txn-free admission points instead — see
    // `defer_build_start_under_pressure` callers in the optimizer/index job.
    // Write txns are independently backpressured at open by
    // `MEMORY_WATERMARKS.apply_backpressure()` in `with_write_txn`.
    BUILD_SEMAPHORE.acquire()?;
    Ok(BuildPermit)
}

static BUILD_SEMAPHORE: std::sync::LazyLock<BuildSemaphore> = std::sync::LazyLock::new(|| {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    // Each build uses significant heap/RSS and then fans out through rayon.
    // Default to at most two concurrent builds; operators can raise this
    // explicitly on large boxes.
    let default_permits = (cpus / 2).clamp(1, 2);
    // Override via HELIX_MAX_CONCURRENT_BUILDS env var if needed.
    let permits = std::env::var("HELIX_MAX_CONCURRENT_BUILDS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default_permits)
        .max(1);
    eprintln!(
        "[INIT] build semaphore: {} permits ({} cpus)",
        permits, cpus
    );
    BuildSemaphore::new(permits)
});

/// Optional operator override for the IVF centroid count (`HELIX_IVF_K`);
/// default is `ivf::default_k(n)` = clamp(isqrt(n), 1, 4096).
fn ivf_k_override() -> Option<usize> {
    static K: std::sync::LazyLock<Option<usize>> = std::sync::LazyLock::new(|| {
        std::env::var("HELIX_IVF_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|k| *k >= 1)
    });
    *K
}

/// Optional operator override for the number of probed IVF posting lists
/// (`HELIX_IVF_NPROBE`); default is the `default_nprobe` persisted at build.
fn ivf_nprobe_override() -> Option<usize> {
    static NPROBE: std::sync::LazyLock<Option<usize>> = std::sync::LazyLock::new(|| {
        std::env::var("HELIX_IVF_NPROBE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n >= 1)
    });
    *NPROBE
}

fn hnsw_build_permit_timeout() -> Option<std::time::Duration> {
    static TIMEOUT: std::sync::LazyLock<Option<std::time::Duration>> =
        std::sync::LazyLock::new(|| {
            let ms = std::env::var("HELIX_HNSW_BUILD_PERMIT_TIMEOUT_MS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(60_000);
            (ms > 0).then(|| std::time::Duration::from_millis(ms))
        });
    *TIMEOUT
}

/// Whether new HNSW build starts should be deferred while memory is above the
/// hard-high watermark. Enabled by default; set `HELIX_BUILD_PRESSURE_GATE` to
/// `0/false/off/no` to restore the old "start builds regardless of pressure"
/// behavior (the fixed semaphore still bounds concurrency).
fn build_pressure_gate_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("HELIX_BUILD_PRESSURE_GATE")
            .map(|value| {
                let value = value.trim().to_ascii_lowercase();
                !matches!(value.as_str(), "0" | "false" | "off" | "no")
            })
            .unwrap_or(true)
    });
    *ENABLED
}

/// Upper bound on how long a single build start may be deferred under memory
/// pressure before it PROCEEDS anyway with a warning. Builds are what DRAIN
/// segment backlogs (they spike RSS short-term but reduce memory long-term), so
/// the deferral is intentionally bounded — never an indefinite block — to keep
/// recovery live. Default 300s; `0` disables deferral (proceed immediately).
fn build_pressure_max_wait() -> std::time::Duration {
    static MAX_WAIT: std::sync::LazyLock<std::time::Duration> = std::sync::LazyLock::new(|| {
        let secs = std::env::var("HELIX_BUILD_PRESSURE_MAX_WAIT_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(300);
        std::time::Duration::from_secs(secs)
    });
    *MAX_WAIT
}

/// Poll interval between pressure re-checks while a build start is deferred.
const BUILD_PRESSURE_POLL: std::time::Duration = std::time::Duration::from_secs(5);

/// Defer a new HNSW build start while memory is above the hard-high watermark,
/// re-checking every [`BUILD_PRESSURE_POLL`] until pressure clears or the
/// bounded budget ([`build_pressure_max_wait`]) is exhausted, after which the
/// build PROCEEDS anyway (logged at `warn!`). This consults the same memory
/// arbiter used by the cache sweepers and cold-open guard
/// (`build_admission_under_pressure`) so build deferral cannot drift from the
/// rest of the pressure machinery. Liveness: an unconditional block could
/// deadlock recovery because builds are what reclaim memory, so we always
/// proceed after the budget.
///
/// MUST only be called from txn-free contexts (no RwTxn — single-writer
/// stall — and no RoTxn — long-lived readers pin freelist pages). The
/// admission points are in the optimizer/index job, before any txn opens.
pub(crate) fn defer_build_start_under_pressure() {
    use crate::helix_engine::storage_core::collection_manager::build_admission_under_pressure;

    if !build_pressure_gate_enabled() {
        return;
    }
    if !build_admission_under_pressure() {
        return;
    }

    let started = std::time::Instant::now();
    let budget = build_pressure_max_wait();
    metrics::counter!("helix_hnsw_build_pressure_deferred_total").increment(1);
    let mut deferred = false;
    while started.elapsed() < budget {
        if !build_admission_under_pressure() {
            if deferred {
                metrics::histogram!(
                    "helix_hnsw_build_pressure_wait_ms",
                    "outcome" => "cleared"
                )
                .record(started.elapsed().as_secs_f64() * 1000.0);
                tracing::info!(
                    waited_secs = started.elapsed().as_secs(),
                    "HNSW build start resumed after memory pressure cleared"
                );
            }
            return;
        }
        deferred = true;
        let remaining = budget.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            break;
        }
        std::thread::sleep(remaining.min(BUILD_PRESSURE_POLL));
    }

    metrics::histogram!(
        "helix_hnsw_build_pressure_wait_ms",
        "outcome" => "budget_exhausted"
    )
    .record(started.elapsed().as_secs_f64() * 1000.0);
    tracing::warn!(
        waited_secs = started.elapsed().as_secs(),
        max_wait_secs = budget.as_secs(),
        "HNSW build start proceeding under sustained memory pressure after deferral budget exhausted"
    );
}

const DB_VECTORS: &str = "vectors"; // for vector data (v:)
const DB_VECTOR_DATA: &str = "vector_data"; // for vector data (v:)

const DB_HNSW_OUT_EDGES: &str = "hnsw_out_nodes"; // for hnsw out node data
const DB_HNSW_NEIGHBOR_LISTS: &str = "hnsw_neighbors"; // packed adjacency lists
const VECTOR_PREFIX: &[u8] = b"v:";
const ENTRY_POINT_KEY: &str = "entry_point";
const NEIGHBOR_BLOCK_MAGIC: &[u8; 4] = b"NBR2";
/// IVF build metadata key (under `SegmentDb::IvfCentroids`). Written last by
/// the IVF build, so its presence marks a complete index — the runtime
/// analogue of `ENTRY_POINT_KEY` for HNSW.
const IVF_META_KEY: &str = "ivf:meta";
/// IVF centroid-table blob key (under `SegmentDb::IvfCentroids`).
const IVF_CENTROIDS_KEY: &str = "ivf:centroids";

/// Per-VectorCore in-process cache capacities. Each open collection has
/// 6 named-vector spaces × N segments worth of cores, so these caps are
/// HARD upper bounds per core and per-pod memory scales linearly with
/// `caps × cores_per_collection × loaded_collections`.
///
/// Defaults match the prior `evict_if_full` thresholds so the LRU swap
/// is a pure semantic upgrade (true LRU eviction instead of FIFO-ish
/// HashMap iteration order) without changing the per-core memory
/// envelope. To reduce memory at the cost of cache hit rate — useful
/// once we're confident in the multi-tenant scaling story — set the
/// `HELIX_*_CACHE_CAP` env vars per-pod. There is no recompile needed.
const NEIGHBOR_CACHE_DEFAULT_ENTRIES: usize = 8_192;
const VECTOR_CACHE_DEFAULT_ENTRIES: usize = 4_096;
const NAV_CACHE_DEFAULT_ENTRIES: usize = 32_768;
const NAV_CACHE_WARM_DEFAULT_ENTRIES: usize = 0;

#[inline]
fn env_cache_cap(name: &str, default: usize) -> NonZeroUsize {
    let v = std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v >= 1)
        .unwrap_or(default);
    NonZeroUsize::new(v).unwrap_or_else(|| NonZeroUsize::new(1).unwrap())
}

#[inline]
fn neighbor_cache_cap() -> NonZeroUsize {
    env_cache_cap("HELIX_NEIGHBOR_CACHE_CAP", NEIGHBOR_CACHE_DEFAULT_ENTRIES)
}

#[inline]
fn vector_cache_cap() -> NonZeroUsize {
    env_cache_cap("HELIX_VECTOR_CACHE_CAP", VECTOR_CACHE_DEFAULT_ENTRIES)
}

#[inline]
fn nav_cache_cap() -> NonZeroUsize {
    env_cache_cap("HELIX_NAV_CACHE_CAP", NAV_CACHE_DEFAULT_ENTRIES)
}

#[inline]
fn nav_cache_warm_cap() -> usize {
    static CAP: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        nav_cache_warm_cap_from_raw(
            std::env::var("HELIX_NAV_CACHE_WARM_CAP").ok().as_deref(),
            NAV_CACHE_WARM_DEFAULT_ENTRIES,
        )
    });
    *CAP
}

fn nav_cache_warm_cap_from_raw(raw: Option<&str>, default: usize) -> usize {
    raw.and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

/// Master switch for the pod-global sharded caches. When off, each
/// VectorCore uses its own private LruCaches (the pre-PR behavior).
/// Read once per process via LazyLock so the decision can't drift.
fn shared_caches_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("HELIX_SHARED_VECTOR_CACHES")
            .ok()
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
            .unwrap_or(false)
    });
    *ENABLED
}

/// Experimental exact HVTQ block-transposed scanner. The current row scorer is
/// faster in local benches, so keep this opt-in until the SIMD LUT kernel wins.
fn hvtq_fastscan_enabled() -> bool {
    std::env::var("HELIX_HVTQ_FASTSCAN")
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

fn hvtq_fastscan_rerank_multiplier() -> usize {
    std::env::var("HELIX_HVTQ_FASTSCAN_RERANK_MULT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(16)
}

/// Lazily-constructed pod-global cache. Every VectorCore that enables
/// shared caches gets the same Arc. Caps are read once at construction;
/// changing `HELIX_*_CACHE_CAP` requires a pod restart to take effect.
fn shared_caches_handle() -> std::sync::Arc<super::shared_cache::SharedVectorCaches> {
    static HANDLE: std::sync::LazyLock<std::sync::Arc<super::shared_cache::SharedVectorCaches>> =
        std::sync::LazyLock::new(|| {
            std::sync::Arc::new(super::shared_cache::SharedVectorCaches::new(
                neighbor_cache_cap(),
                vector_cache_cap(),
                nav_cache_cap(),
            ))
        });
    std::sync::Arc::clone(&HANDLE)
}

#[inline]
fn patch_reverse_edges_on_batch_delete() -> bool {
    std::env::var("HELIX_DENSE_DELETE_PATCH_REVERSE_EDGES")
        .map(|value| {
            matches!(
                value.trim(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(false)
}

const DENSE_DELETE_READ_SNAPSHOT_CHUNK_DEFAULT: usize = 64;
const DENSE_DELETE_READ_SNAPSHOT_CHUNK_ENV: &str = "HELIX_DENSE_DELETE_READ_SNAPSHOT_CHUNK";

#[cfg(test)]
static DENSE_DELETE_READ_SNAPSHOT_OPENS_FOR_TEST: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn dense_delete_read_snapshot_chunk_from_raw(raw: Option<&str>) -> usize {
    raw.and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DENSE_DELETE_READ_SNAPSHOT_CHUNK_DEFAULT)
}

#[inline]
fn dense_delete_read_snapshot_chunk() -> usize {
    dense_delete_read_snapshot_chunk_from_raw(
        std::env::var(DENSE_DELETE_READ_SNAPSHOT_CHUNK_ENV)
            .ok()
            .as_deref(),
    )
}

/// Process-wide feature gate for tombstone-only dense deletes with repair
/// deferred to the next merge (issue #29 "fix 2"). Default OFF: the delete
/// path hard-removes rows exactly as before. When on, a delete only records
/// a delete-tombstone (see `delete_tombstone_key`) and updates the
/// in-memory `VectorCore::delete_tombstones` set; the physical rows are
/// dropped later when the owning segment's export loop excludes them at
/// merge time. Search must reject a tombstoned id regardless of the flag's
/// CURRENT value — see `VectorCore::is_delete_tombstoned` — so flipping
/// this off after deletes exist does not resurrect them.
#[cfg(not(test))]
pub(crate) fn lsm_delete_tombstones_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("HELIX_LSM_DELETE_TOMBSTONES")
            .ok()
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    });
    *ENABLED
}

#[cfg(test)]
pub(crate) fn lsm_delete_tombstones_enabled() -> bool {
    std::env::var("HELIX_LSM_DELETE_TOMBSTONES")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// Fraction of a segment's live rows that must be delete-tombstoned before
/// the merge scheduler forces that segment into a merge candidate set even
/// when the space is below its normal segment-count merge trigger
/// (`HELIX_LSM_DELETE_VACUUM_FRACTION`, default 0.20 — mirrors the dormant
/// `VACUUM_DELETED_FRACTION` policy in `segments.rs`).
pub(crate) fn delete_vacuum_fraction() -> f64 {
    static FRACTION: std::sync::LazyLock<f64> = std::sync::LazyLock::new(|| {
        std::env::var("HELIX_LSM_DELETE_VACUUM_FRACTION")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0 && *v < 1.0)
            .unwrap_or(0.20)
    });
    *FRACTION
}

/// Encode a DELETE-tombstone key for `Namespace::DenseTombstones`. Distinct
/// from the re-upsert tombstone keyspace (`tombstone_key` in
/// `named_vectors.rs`, `physical_name ++ 0x00 ++ id`) via a leading NUL
/// marker byte that no physical segment name can produce (segment names are
/// NUL-free ASCII), so the two tombstone families share one namespace
/// without any risk of key collision.
fn delete_tombstone_key(physical_name: &str, id: u128) -> Vec<u8> {
    debug_assert!(
        !physical_name.as_bytes().contains(&0u8),
        "delete_tombstone_key requires NUL-free segment names; got {physical_name:?}"
    );
    let mut key = Vec::with_capacity(1 + physical_name.len() + 1 + 16);
    key.push(0u8);
    key.extend_from_slice(physical_name.as_bytes());
    key.push(0u8);
    key.extend_from_slice(&id.to_be_bytes());
    key
}

/// Prefix for iterating every delete-tombstone of one segment.
fn delete_tombstone_segment_prefix(physical_name: &str) -> Vec<u8> {
    let mut p = Vec::with_capacity(1 + physical_name.len() + 1);
    p.push(0u8);
    p.extend_from_slice(physical_name.as_bytes());
    p.push(0u8);
    p
}

/// Recover the id encoded in a `delete_tombstone_key`. Used by
/// reconstruction, which only knows the segment's own prefix was matched by
/// the scan and needs the trailing id back out.
fn decode_delete_tombstone_id(physical_name: &str, key: &[u8]) -> Option<u128> {
    let prefix = delete_tombstone_segment_prefix(physical_name);
    let id_bytes = key.strip_prefix(prefix.as_slice())?;
    let id_bytes: [u8; 16] = id_bytes.try_into().ok()?;
    Some(u128::from_be_bytes(id_bytes))
}

/// Minimum fraction of level-0 neighbors to fetch after approximate ranking.
/// BinarySign recall is only ~35%, so we must keep a large share of candidates
/// to avoid catastrophic recall loss. 75% is conservative enough to benefit from
/// the ranking (worst candidates are pruned) without killing recall.
const LEVEL0_APPROX_KEEP_FRACTION: f64 = 0.75;
const MIN_HNSW_M: usize = 2;
/// Create-time upper bound for client-supplied `m`; very large `m` only burns
/// memory and build time without recall benefit.
const MAX_HNSW_M: usize = 128;
const MAX_HNSW_LEVEL: usize = 64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWConfig {
    pub m: usize,            // max num of bi-directional links per element
    pub m_max_0: usize,      // max num of links for lower layers
    pub ef_construct: usize, // size of the dynamic candidate list for construction
    pub m_l: f64,            // level generation factor
    pub ef: usize,           // search param, num of cands to search
    #[serde(default = "default_adaptive_ef")]
    pub adaptive_ef_enabled: bool, // auto-increase ef for selective filters
}

fn default_adaptive_ef() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct StoredVectorData {
    #[serde(default)]
    fields: HashMap<String, Value>,
    #[serde(default)]
    original_vector: Option<Vec<f32>>,
}

const EXTERNALIZED_VECTOR_MARKER: &[u8] = b"";
pub(crate) const HVTQ_SIDECAR_BLOB_KEY: &[u8] = b"__helix_sidecar:hvtq";
pub(crate) const HVEC_SIDECAR_BLOB_KEY: &[u8] = b"__helix_sidecar:hvec";
pub(crate) const HVEC_SIDECAR_ORDINALS_BLOB_KEY: &[u8] = b"__helix_sidecar:hvec_ordinals";
const HVEC_SIDECAR_ORDINALS_MAGIC: &[u8; 4] = b"HORD";

/// Pre-computed HNSW graph that can be flushed to LMDB in a separate write
/// transaction. This allows the expensive graph construction to happen outside
/// the write lock, dramatically reducing write-lock contention during ingest.
pub struct PreparedIndex {
    pub(crate) point_ids: Vec<u128>,
    pub(crate) original_data: Vec<Option<Vec<f32>>>,
    pub(crate) encoded_data: Vec<Option<Vec<u8>>>,
    pub(crate) point_fields: Vec<HashMap<String, Value>>,
    pub(crate) levels: Vec<usize>,
    pub(crate) adjacency: Vec<Option<Vec<Vec<u32>>>>,
    pub(crate) entry_ord: usize,
    pub(crate) level_zero_preflushed: bool,
}

impl PreparedIndex {
    pub(crate) fn raw_len_at(&self, index: usize) -> usize {
        self.original_data
            .get(index)
            .and_then(Option::as_ref)
            .map(|raw| raw.len())
            .unwrap_or(0)
    }

    pub(crate) fn raw_at(&self, index: usize) -> Result<&[f32], VectorError> {
        self.original_data
            .get(index)
            .and_then(Option::as_deref)
            .ok_or_else(|| {
                VectorError::VectorCoreError(format!(
                    "prepared index row {} raw vector already drained",
                    index
                ))
            })
    }

    pub(crate) fn live_raw_slices(&self) -> Result<Vec<&[f32]>, VectorError> {
        let mut rows = Vec::with_capacity(self.point_ids.len());
        for index in 0..self.point_ids.len() {
            rows.push(self.raw_at(index)?);
        }
        Ok(rows)
    }

    pub(crate) fn live_encoded_slices(&self) -> Option<Vec<&[u8]>> {
        if self.encoded_data.len() != self.point_ids.len() {
            return None;
        }
        let mut rows = Vec::with_capacity(self.point_ids.len());
        for row in &self.encoded_data {
            rows.push(row.as_deref()?);
        }
        Some(rows)
    }

    pub(crate) fn adjacency_level_count_at(&self, index: usize) -> usize {
        self.adjacency
            .get(index)
            .and_then(Option::as_ref)
            .map(Vec::len)
            .unwrap_or(0)
    }

    pub(crate) fn neighbor_ref_count_at(&self, index: usize) -> usize {
        self.adjacency
            .get(index)
            .and_then(Option::as_ref)
            .map(|levels| levels.iter().map(Vec::len).sum())
            .unwrap_or(0)
    }

    fn neighbor_ids_at(&self, index: usize, level: usize) -> Result<Vec<u128>, VectorError> {
        let id = self
            .point_ids
            .get(index)
            .copied()
            .ok_or(VectorError::InvalidVectorData)?;
        let neighbors = self
            .adjacency
            .get(index)
            .and_then(Option::as_ref)
            .and_then(|levels| levels.get(level))
            .ok_or_else(|| {
                VectorError::VectorCoreError(format!(
                    "prepared index row {} level {} adjacency already drained",
                    index, level
                ))
            })?;
        let mut out = Vec::with_capacity(neighbors.len());
        for &ord in neighbors {
            let neighbor_id = self
                .point_ids
                .get(ord as usize)
                .copied()
                .ok_or(VectorError::InvalidVectorData)?;
            if neighbor_id != id {
                out.push(neighbor_id);
            }
        }
        Ok(out)
    }

    fn drain_row(&mut self, index: usize) {
        if let Some(slot) = self.original_data.get_mut(index) {
            *slot = None;
        }
        if let Some(slot) = self.encoded_data.get_mut(index) {
            *slot = None;
        }
        if let Some(slot) = self.adjacency.get_mut(index) {
            *slot = None;
        }
    }

    pub(crate) fn mark_level_zero_preflushed(&mut self) {
        self.level_zero_preflushed = true;
    }
}

#[derive(Debug, Clone, Default)]
pub struct NeighborBlock {
    ids: Vec<u128>,
    approx_code_len: usize,
    approx_codes: Vec<u8>,
}

impl NeighborBlock {
    fn from_ids(ids: Vec<u128>) -> Self {
        Self {
            ids,
            approx_code_len: 0,
            approx_codes: Vec::new(),
        }
    }

    fn has_approx_codes(&self) -> bool {
        self.approx_code_len > 0
            && !self.approx_codes.is_empty()
            && self.approx_codes.len() == self.ids.len().saturating_mul(self.approx_code_len)
    }

    fn code_at(&self, index: usize) -> Option<&[u8]> {
        if !self.has_approx_codes() {
            return None;
        }
        let start = index.checked_mul(self.approx_code_len)?;
        let end = start.checked_add(self.approx_code_len)?;
        self.approx_codes.get(start..end)
    }

    fn retain_without(&self, removed_id: u128) -> Self {
        if !self.has_approx_codes() {
            return Self::from_ids(
                self.ids
                    .iter()
                    .copied()
                    .filter(|neighbor_id| *neighbor_id != removed_id)
                    .collect(),
            );
        }

        let mut ids = Vec::with_capacity(self.ids.len());
        let mut approx_codes =
            Vec::with_capacity(self.ids.len().saturating_mul(self.approx_code_len));
        for (index, neighbor_id) in self.ids.iter().copied().enumerate() {
            if neighbor_id == removed_id {
                continue;
            }
            ids.push(neighbor_id);
            if let Some(code) = self.code_at(index) {
                approx_codes.extend_from_slice(code);
            }
        }
        Self {
            ids,
            approx_code_len: self.approx_code_len,
            approx_codes,
        }
    }

    /// Batched variant of `retain_without`. Removes every id present in
    /// `removed` in one pass over the block. The removed slice is usually
    /// tiny, so linear membership avoids a per-neighbor-block HashSet
    /// allocation inside `VectorCore::delete_vectors_batch`.
    fn retain_without_slice(&self, removed: &[u128]) -> Self {
        match removed {
            [] => self.clone(),
            [removed_id] => self.retain_without(*removed_id),
            _ if !self.has_approx_codes() => Self::from_ids(
                self.ids
                    .iter()
                    .copied()
                    .filter(|nid| !removed.contains(nid))
                    .collect(),
            ),
            _ => {
                let mut ids = Vec::with_capacity(self.ids.len());
                let mut approx_codes =
                    Vec::with_capacity(self.ids.len().saturating_mul(self.approx_code_len));
                for (index, neighbor_id) in self.ids.iter().copied().enumerate() {
                    if removed.contains(&neighbor_id) {
                        continue;
                    }
                    ids.push(neighbor_id);
                    if let Some(code) = self.code_at(index) {
                        approx_codes.extend_from_slice(code);
                    }
                }
                Self {
                    ids,
                    approx_code_len: self.approx_code_len,
                    approx_codes,
                }
            }
        }
    }
}

/// Reject vectors with NaN/inf components at the engine boundary: they yield
/// NaN distances that corrupt heap ordering and graph construction.
#[inline]
fn ensure_finite_vector(data: &[f32]) -> Result<(), VectorError> {
    if data.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(VectorError::InvalidVectorData)
    }
}

impl HNSWConfig {
    pub fn new(m: Option<usize>, ef_construct: Option<usize>, ef: Option<usize>) -> Self {
        let m = m.unwrap_or(16).max(MIN_HNSW_M);
        Self {
            m,
            m_max_0: 2 * m,
            ef_construct: ef_construct.unwrap_or(128),
            m_l: 1.0 / (m as f64).ln(),
            ef: ef.unwrap_or(128),
            adaptive_ef_enabled: true,
        }
    }

    /// Build an `HNSWConfig` by layering optional per-collection overrides on
    /// top of global defaults. Any field left unset in `overrides` falls back
    /// to the global `HELIX_HNSW_*` / env-driven value.
    pub fn with_overrides(
        global_m: Option<usize>,
        global_ef_construct: Option<usize>,
        global_ef: Option<usize>,
        overrides: Option<&HnswOverrides>,
    ) -> Self {
        let m = overrides.and_then(|o| o.m).or(global_m);
        let ef_construct = overrides
            .and_then(|o| o.ef_construction)
            .or(global_ef_construct);
        let ef = overrides.and_then(|o| o.ef).or(global_ef);
        Self::new(m, ef_construct, ef)
    }
}

fn assign_hnsw_level(sample: f64, m_l: f64) -> usize {
    if !m_l.is_finite() || m_l <= 0.0 {
        return 0;
    }
    let normalized_sample = if sample.is_finite() && sample > 0.0 {
        sample
    } else {
        f64::MIN_POSITIVE
    };
    let level = (-normalized_sample.ln() * m_l).floor();
    if !level.is_finite() || level < 0.0 {
        0
    } else {
        (level as usize).min(MAX_HNSW_LEVEL)
    }
}

/// Per-collection HNSW tuning overrides. Persisted in `metadata_db` under
/// the `HNSW_OVERRIDES_KEY` for collections whose create request supplied a
/// `hnsw_config` block. Missing fields fall back to global `HELIX_HNSW_*`
/// env defaults at `HNSWConfig::with_overrides` time.
///
/// `m` takes effect at initial graph build (topology is baked into LMDB
/// neighbor lists on first insert). `ef_construction` affects insert-time
/// quality. `ef` is the default search-time dynamic candidate-list size;
/// callers can still override per-request via `params.hnsw_ef`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct HnswOverrides {
    #[serde(default)]
    pub m: Option<usize>,
    #[serde(default)]
    pub ef_construction: Option<usize>,
    #[serde(default)]
    pub ef: Option<usize>,
}

impl HnswOverrides {
    pub fn is_empty(&self) -> bool {
        self.m.is_none() && self.ef_construction.is_none() && self.ef.is_none()
    }

    /// Reject client-supplied HNSW parameters that would degenerate the
    /// graph. Collection-create entry points should call this before
    /// persisting overrides; `HNSWConfig::new` additionally clamps `m` up to
    /// `MIN_HNSW_M` so already-persisted bad values cannot hang a build.
    pub fn validate(&self) -> Result<(), VectorError> {
        if let Some(m) = self.m {
            if m < MIN_HNSW_M {
                return Err(VectorError::VectorCoreError(format!(
                    "hnsw_config.m must be at least {MIN_HNSW_M}"
                )));
            }
            if m > MAX_HNSW_M {
                return Err(VectorError::VectorCoreError(format!(
                    "hnsw_config.m must be at most {MAX_HNSW_M}"
                )));
            }
        }
        if self.ef_construction == Some(0) {
            return Err(VectorError::VectorCoreError(
                "invalid hnsw_config.ef_construct=0: must be at least 1".into(),
            ));
        }
        if self.ef == Some(0) {
            return Err(VectorError::VectorCoreError(
                "invalid hnsw_config.ef=0: must be at least 1".into(),
            ));
        }
        Ok(())
    }
}

/// Labels supplied by the named-vector layer for low-level search metrics.
#[derive(Clone, Copy)]
pub struct VectorSearchMetrics<'a> {
    pub collection: &'a str,
    pub vector: &'a str,
    pub segment_count: usize,
    pub segment_index: usize,
    pub filtered: bool,
}

#[derive(PartialEq)]
struct Candidate {
    id: u128,
    distance: f32,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// ANN index mode of one dense segment, derived at runtime from stored keys
/// (see `VectorCore::index_mode`): HNSW when an entry point exists, IVF when
/// an `ivf:meta` blob exists, flat scan otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexMode {
    Hnsw,
    Ivf,
    Flat,
}

#[derive(Clone, Copy, PartialEq)]
struct FlatCandidate {
    id: u128,
    level: usize,
    distance: f32,
}

impl Eq for FlatCandidate {}

impl PartialOrd for FlatCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for FlatCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

#[derive(Clone, Copy, PartialEq)]
struct MmapFlatCandidate {
    id: u128,
    ordinal: u64,
    distance: f32,
}

impl Eq for MmapFlatCandidate {}

impl PartialOrd for MmapFlatCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for MmapFlatCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

fn search_layer_ord<DF>(
    query_ord: usize,
    entry_ord: usize,
    ef: usize,
    level: usize,
    adjacency: &[StdRwLock<Vec<Vec<u32>>>],
    dist_fn: &DF,
) -> Vec<(usize, f32)>
where
    DF: Fn(usize, usize) -> f32,
{
    // Note: leaving as Vec::new() / HashSet::new() / BinaryHeap::new() —
    // an earlier `with_capacity(ef)` swap perturbed HNSW build topology
    // enough to flip a unit test (delete_removes_asymmetric_neighbor_references
    // depends on which nodes 1's neighbor list contains, which is rebuild
    // sensitive). The micro-win is small enough that reverting is the
    // correct call until we have a topology-stable build path.
    let mut visited = HashSet::new();
    let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();
    let mut results: Vec<(usize, f32)> = Vec::new();

    let ep_dist = dist_fn(query_ord, entry_ord);
    visited.insert(entry_ord);
    candidates.push(Candidate {
        id: entry_ord as u128,
        distance: ep_dist,
    });
    results.push((entry_ord, ep_dist));
    let mut worst_dist = ep_dist;

    while let Some(curr) = candidates.pop() {
        if results.len() >= ef && curr.distance > worst_dist {
            break;
        }

        let curr_ord = curr.id as usize;
        let neighbors = {
            let guard = adjacency[curr_ord].read().unwrap();
            if level < guard.len() {
                guard[level]
                    .iter()
                    .map(|&ord| ord as usize)
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        };

        for neighbor_ord in neighbors {
            if !visited.insert(neighbor_ord) {
                continue;
            }
            let d = dist_fn(query_ord, neighbor_ord);
            if results.len() < ef || d < worst_dist {
                candidates.push(Candidate {
                    id: neighbor_ord as u128,
                    distance: d,
                });
                results.push((neighbor_ord, d));
                if results.len() > ef {
                    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                    results.truncate(ef);
                }
                worst_dist = results
                    .iter()
                    .map(|r| r.1)
                    .fold(f32::NEG_INFINITY, f32::max);
            }
        }
    }

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    results
}

fn select_neighbor_ords(
    candidates: &[(usize, f32)],
    max_neighbors: usize,
    point_count: usize,
) -> Vec<u32> {
    candidates
        .iter()
        .take(max_neighbors)
        .filter_map(|&(ord, _)| (ord < point_count).then_some(ord as u32))
        .collect()
}

/// Construction distance for the in-memory merge and split builds between
/// ordinals `a` and `b`. Cosine uses the SQ8 codes (`codes`, `dim` bytes per point) as it
/// always has. Dot and Euclid use the raw f32 vectors with the same distances
/// as the monolithic build and serving search: SQ8's per-dimension min-shift
/// and scale does not preserve inner products or L2 geometry (in 1D Dot,
/// [-2, -1, +1] all become non-negative codes, so -1 links to +1 instead of
/// -2), and `codes` is left empty for them.
fn merge_build_distance(
    distance_metric: &DistanceMetric,
    raw: &[Vec<f32>],
    codes: &[u8],
    dim: usize,
    a: usize,
    b: usize,
) -> f32 {
    use super::simd;
    match distance_metric {
        DistanceMetric::Cosine => simd::cosine_u8(
            &codes[a * dim..(a + 1) * dim],
            &codes[b * dim..(b + 1) * dim],
        ),
        DistanceMetric::Dot => simd::dot_f32(&raw[a], &raw[b]),
        DistanceMetric::Euclid => simd::euclid_f32(&raw[a], &raw[b]).sqrt(),
    }
}

fn insert_point_ord<DF>(
    point_ord: usize,
    current_entry_ord: usize,
    current_max_level: usize,
    levels: &[usize],
    adjacency: &[StdRwLock<Vec<Vec<u32>>>],
    ef_construction: usize,
    m: usize,
    m_max_0: usize,
    dist_fn: &DF,
) where
    DF: Fn(usize, usize) -> f32,
{
    let point_level = levels[point_ord];

    let mut ep = current_entry_ord;
    for level in (point_level + 1..=current_max_level).rev() {
        let nearest = search_layer_ord(point_ord, ep, 1, level, adjacency, dist_fn);
        if let Some(&(best_ord, _)) = nearest.first() {
            ep = best_ord;
        }
    }

    let top = point_level.min(current_max_level);
    for level in (0..=top).rev() {
        let nearest = search_layer_ord(point_ord, ep, ef_construction, level, adjacency, dist_fn);

        if let Some(&(best_ord, _)) = nearest.first() {
            ep = best_ord;
        }

        let max_neighbors = if level == 0 { m_max_0 } else { m };
        let selected = select_neighbor_ords(&nearest, max_neighbors, levels.len());

        {
            let mut guard = adjacency[point_ord].write().unwrap();
            if level < guard.len() {
                guard[level] = selected.clone();
            }
        }

        for &neighbor_ord_u32 in &selected {
            let neighbor_ord = neighbor_ord_u32 as usize;
            if neighbor_ord == point_ord {
                continue;
            }
            if neighbor_ord < adjacency.len() {
                let mut guard = adjacency[neighbor_ord].write().unwrap();
                if level < guard.len() {
                    let point_ord_u32 = point_ord as u32;
                    if !guard[level].contains(&point_ord_u32) {
                        guard[level].push(point_ord_u32);
                    }
                    if guard[level].len() > max_neighbors {
                        let mut scored: Vec<(u32, f32)> = guard[level]
                            .iter()
                            .copied()
                            .map(|nord| {
                                let nord_usize = nord as usize;
                                (nord, dist_fn(neighbor_ord, nord_usize))
                            })
                            .collect();
                        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                        scored.truncate(max_neighbors);
                        guard[level] = scored.into_iter().map(|(id, _)| id).collect();
                    }
                }
            }
        }
    }
}

pub trait HeapOps<T> {
    /// Extend the heap with another heap
    /// Used because using `.extend()` does not keep the order
    fn extend_inord(&mut self, other: BinaryHeap<T>)
    where
        T: Ord;

    /// Take the top k elements from the heap
    /// Used because using `.iter()` does not keep the order
    fn take_inord(&mut self, k: usize) -> BinaryHeap<T>
    where
        T: Ord;

    /// Take the top k elements from the heap and return a vector
    fn to_vec(&mut self, k: usize) -> Vec<T>
    where
        T: Ord;

    /// Get the maximum element from the heap
    /// Returns the farthest element (worst candidate) in the heap.
    /// Uses iter().min() because Candidate has reversed Ord (min-distance heap).
    fn get_farthest(&self) -> Option<&T>
    where
        T: Ord;

    fn to_vec_with_filter<F>(&mut self, k: usize, filter: Option<&[F]>) -> Vec<T>
    where
        T: Ord,
        F: Fn(&T) -> bool;
}

impl<T> HeapOps<T> for BinaryHeap<T> {
    #[inline(always)]
    fn extend_inord(&mut self, mut other: BinaryHeap<T>)
    where
        T: Ord,
    {
        self.reserve(other.len());
        for item in other.drain() {
            self.push(item);
        }
    }

    #[inline(always)]
    fn take_inord(&mut self, k: usize) -> BinaryHeap<T>
    where
        T: Ord,
    {
        let mut result = BinaryHeap::with_capacity(k);
        for _ in 0..k {
            if let Some(item) = self.pop() {
                result.push(item);
            } else {
                break;
            }
        }
        result
    }

    #[inline(always)]
    fn to_vec(&mut self, k: usize) -> Vec<T>
    where
        T: Ord,
    {
        let mut result = Vec::with_capacity(k);
        for _ in 0..k {
            if let Some(item) = self.pop() {
                result.push(item);
            } else {
                break;
            }
        }
        result
    }

    #[inline(always)]
    /// Returns the farthest element (worst candidate) in the heap.
    /// Uses iter().min() because Candidate has reversed Ord (min-distance heap).
    fn get_farthest(&self) -> Option<&T>
    where
        T: Ord,
    {
        self.iter().min()
    }

    #[inline(always)]
    fn to_vec_with_filter<F>(&mut self, k: usize, filter: Option<&[F]>) -> Vec<T>
    where
        T: Ord,
        F: Fn(&T) -> bool,
    {
        let mut result = Vec::with_capacity(k);
        for _ in 0..k {
            // while pop check filters and pop until one passes
            while let Some(item) = self.pop() {
                if filter.is_none() || filter.unwrap().iter().all(|f| f(&item)) {
                    result.push(item);
                    break;
                }
            }
        }
        result
    }
}

enum SidecarOrdinals {
    Partial(HashMap<u128, u64>),
    Complete(HashMap<u128, u64>),
}

impl SidecarOrdinals {
    fn entries(&self) -> &HashMap<u128, u64> {
        match self {
            Self::Partial(entries) | Self::Complete(entries) => entries,
        }
    }

    fn entries_mut(&mut self) -> &mut HashMap<u128, u64> {
        match self {
            Self::Partial(entries) | Self::Complete(entries) => entries,
        }
    }

    fn is_complete(&self) -> bool {
        matches!(self, Self::Complete(_))
    }
}

const REQUEST_SIDECAR_ORDINAL_CACHE_MAX: usize = 4096;

#[derive(Default)]
struct SidecarOrdinalRequestCache {
    entries: HashMap<u128, Option<u64>>,
}

impl SidecarOrdinalRequestCache {
    fn new() -> Self {
        Self::default()
    }

    fn get(&self, id: u128) -> Option<Option<u64>> {
        self.entries.get(&id).copied()
    }

    fn insert(&mut self, id: u128, ordinal: Option<u64>) {
        if self.entries.contains_key(&id) || self.entries.len() < REQUEST_SIDECAR_ORDINAL_CACHE_MAX
        {
            self.entries.insert(id, ordinal);
        }
    }

    fn is_full(&self) -> bool {
        self.entries.len() >= REQUEST_SIDECAR_ORDINAL_CACHE_MAX
    }
}

pub struct VectorCore {
    pub vectors_db: Option<Database<Bytes, Bytes>>,
    pub vector_data_db: Option<Database<Bytes, Bytes>>,
    pub out_edges_db: Option<Database<Bytes, Unit>>,
    pub neighbor_lists_db: Option<Database<Bytes, Bytes>>,
    /// id → ordinal (u64 LE) mapping for mmap vector lookup.
    pub ordinals_db: Option<Database<Bytes, Bytes>>,
    pub config: HNSWConfig,
    pub distance_metric: DistanceMetric,
    pub spindle: SpindleConfig,
    /// Sidecar file stem (`{data_dir}/{segment_name}`) used to derive `.hvec`
    /// and `.hvs8` paths. Present even when compact mutable segments do not
    /// create an appendable HVEC sidecar.
    mmap_sidecar_stem: Option<PathBuf>,
    /// Flat mmap'd vector file for O(1) reads on the search hot path.
    /// The inner option is empty when running without a data directory or when
    /// compact mutable segments intentionally defer sidecar creation until the
    /// immutable HVS8 publish boundary.
    mmap_store: StdRwLock<Option<super::mmap_vectors::MmapBackend>>,
    /// In-process id -> sidecar ordinal map. When present, HNSW scoring avoids
    /// a SlateDB point-get for every neighbor before reading the mmap row.
    mmap_ordinals: StdRwLock<Option<SidecarOrdinals>>,
    /// Graph out-edges index from the owning collection's storage core.
    ///
    /// Populated when the VectorCore is constructed for a segment that lives
    /// inside a HelixGraphStorage (i.e. the normal production path); `None`
    /// for standalone VectorCores (legacy tests, bulk-build micro-benches)
    /// that don't have a graph attached.
    ///
    /// This is the plumbing hook for graph-entangled HNSW: both HNSW search
    /// and insert can consult graph adjacency through this handle using the
    /// same RoTxn, without crossing into a separate LMDB env. Behavior is
    /// gated by `HELIX_GRAPH_ENTANGLED_HNSW=1`; when the flag is off this
    /// field is read but never dereferenced on the hot path.
    ///
    /// Keyed by `HelixGraphStorage::out_edge_key(from_id, label_hash)` (20
    /// bytes). A `prefix_iter` on the first 16 bytes yields every outgoing
    /// edge regardless of label — see `drop_node` in `storage_core.rs` for
    /// the same pattern.
    graph_out_edges_db: Option<Database<Bytes, Bytes>>,
    /// LRU caches for the search hot path. `Mutex` instead of `RwLock`
    /// because `LruCache::get` mutates (promotes to MRU). Lock contention
    /// is bounded — per-search lock acquires are <1 µs and search visits
    /// at most a few hundred nodes.
    ///
    /// When `shared_caches` is `Some` (HELIX_SHARED_VECTOR_CACHES=1), the
    /// hot path consults the pod-global cache instead and these private
    /// fields stay empty. Kept around as the legacy fallback so we can A/B
    /// the rollout and revert by flipping the env flag.
    neighbor_cache: Mutex<LruCache<(u128, usize), NeighborBlock>>,
    vector_cache: Mutex<LruCache<(u128, usize), Vec<f32>>>,
    nav_neighbor_cache: Mutex<LruCache<(u128, usize), NeighborBlock>>,
    nav_vector_cache: Mutex<LruCache<(u128, usize), Vec<f32>>>,
    nav_cache_root: StdRwLock<Option<(u128, usize)>>,
    /// Pod-global sharded cache handle. When `Some`, all cache ops route
    /// here keyed by `(namespace, id, level)` instead of the private LRUs
    /// above. `namespace` is derived from the owning segment's physical
    /// name so two cores on the same collection share cache entries where
    /// their id spaces overlap (rare) without stomping each other.
    shared_caches: Option<std::sync::Arc<super::shared_cache::SharedVectorCaches>>,
    /// Namespace id for `shared_caches` keys. Unused when `shared_caches`
    /// is `None`. Stable across process restarts (SipHash of the physical
    /// segment name) so cache-age metrics stay comparable.
    cache_namespace: u64,
    /// Set by read-only scan/search paths when an LMDB externalized-vector
    /// marker points to missing mmap sidecar bytes. The index worker uses this
    /// to run a bounded write-side purge only on affected segment cores.
    externalized_marker_repair_needed: AtomicBool,
    /// Shared storage backend handle from the owning `HelixGraphStorage`
    /// (US-006 seam). Shared via `Arc` because `AnyBackend` is not `Clone`
    /// and the `Lsm` variant cannot be re-created from an env. Threaded in
    /// at construction so each core can later address its per-segment DBs
    /// via `Namespace::Segment { physical_name: &self.physical_name, db }`.
    /// Plumbing only in 6a — no KV access routes through it yet (6c-6g wire
    /// the per-segment DBs onto it), hence `#[allow(dead_code)]`.
    #[allow(dead_code)]
    pub(crate) backend: Arc<AnyBackend>,
    /// Physical segment name (e.g. `{base}__seg_{id:06}`) used to address
    /// this core's per-segment databases through the backend namespace.
    /// Empty for the legacy unnamed base core created via `VectorCore::new`.
    #[allow(dead_code)]
    pub(crate) physical_name: String,
    /// Caches a completed lazy sidecar probe that found no materializable
    /// sidecar, so later query-path reads skip the write lock and LSM scan.
    /// Stores the probe time as unix seconds (0 = no cached miss). On the
    /// writer the miss is authoritative until a local materialization clears
    /// it; on a reader replica the writer can seal the sidecar upstream at
    /// any time, so the cached miss expires after
    /// [`lsm_sidecar_miss_ttl_secs`] and the next read re-probes.
    lsm_sidecar_miss_cached_at_secs: AtomicU64,
    /// In-memory set of ids tombstone-deleted on this segment
    /// (`HELIX_LSM_DELETE_TOMBSTONES`): the row is still physically present
    /// (repair is deferred to the next merge of this segment), so every
    /// search read path must reject a member of this set the same way it
    /// already rejects `VectorNotFound`. Populated by `tombstone_delete_batch_be`
    /// and rebuilt on open by `reconstruct_delete_tombstones_be`. Checked
    /// unconditionally (not gated on the env flag) so a flag flip to OFF
    /// after deletes exist does not resurrect them in search.
    delete_tombstones: StdRwLock<HashSet<u128>>,
}

/// Unix-epoch seconds, floored to 1 so `0` stays reserved for "no cached miss".
fn unix_secs_now_nonzero() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(1)
        .max(1)
}

/// TTL (seconds) for a reader replica's cached "sidecar missing" probe
/// (`HELIX_LSM_SIDECAR_MISS_TTL_SECS`, default 30, 0 = never expires).
fn lsm_sidecar_miss_ttl_secs() -> u64 {
    std::env::var("HELIX_LSM_SIDECAR_MISS_TTL_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(30)
}

fn lsm_sidecar_miss_expired(cached_at_secs: u64, now_secs: u64, ttl_secs: u64) -> bool {
    ttl_secs > 0 && now_secs.saturating_sub(cached_at_secs) >= ttl_secs
}

/// Process-wide count of in-flight first-touch LSM sidecar materializations.
static SIDECAR_MATERIALIZE_INFLIGHT: AtomicUsize = AtomicUsize::new(0);

/// Cap on concurrent first-touch sidecar scans
/// (`HELIX_LSM_SIDECAR_MATERIALIZE_CONCURRENCY`, default 4, 0 = unbounded).
fn sidecar_materialize_concurrency() -> usize {
    std::env::var("HELIX_LSM_SIDECAR_MATERIALIZE_CONCURRENCY")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map(|v| if v == 0 { usize::MAX } else { v })
        .unwrap_or(4)
}

/// RAII permit bounding concurrent sidecar materializations process-wide, so a
/// cold search wave over many collections cannot fan out into N simultaneous
/// SlateDB segment scans (S3 GET burst + memory spike) through the query path.
/// Capped probes defer: the query serves from the full-vector fallback and a
/// later read retries the materialization.
struct SidecarMaterializePermit {
    counter: &'static AtomicUsize,
}

impl SidecarMaterializePermit {
    fn try_acquire() -> Option<Self> {
        let counter = &SIDECAR_MATERIALIZE_INFLIGHT;
        let cap = sidecar_materialize_concurrency();
        counter
            .fetch_update(AtomicOrdering::AcqRel, AtomicOrdering::Acquire, |cur| {
                (cur < cap).then_some(cur + 1)
            })
            .ok()
            .map(|_| Self { counter })
    }
}

impl Drop for SidecarMaterializePermit {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, AtomicOrdering::Release);
    }
}

impl VectorCore {
    fn neighbor_code_config(dim: usize) -> Option<SpindleConfig> {
        if dim == 0 {
            return None;
        }

        Some(SpindleConfig {
            mode: SpindleMode::BinarySign,
            keep_original: false,
            rescore: false,
            oversampling: 1,
            binary_dims: dim.min(256),
            turbo_dims: dim,
        })
    }

    pub fn new(
        env: &Env,
        txn: &mut RwTxn,
        config: HNSWConfig,
        backend: Arc<AnyBackend>,
    ) -> Result<Self, VectorError> {
        let create_lmdb_dbs = backend.kind() == BackendKind::Lmdb;
        let vectors_db = if create_lmdb_dbs {
            Some(env.create_database(txn, Some(DB_VECTORS))?)
        } else {
            None
        };
        let vector_data_db = if create_lmdb_dbs {
            Some(env.create_database(txn, Some(DB_VECTOR_DATA))?)
        } else {
            None
        };
        let out_edges_db = if create_lmdb_dbs {
            Some(env.create_database(txn, Some(DB_HNSW_OUT_EDGES))?)
        } else {
            None
        };
        let neighbor_lists_db = if create_lmdb_dbs {
            Some(env.create_database(txn, Some(DB_HNSW_NEIGHBOR_LISTS))?)
        } else {
            None
        };
        let ordinals_db = if create_lmdb_dbs {
            Some(env.create_database(txn, Some("vector_ordinals"))?)
        } else {
            None
        };

        Ok(Self {
            vectors_db,
            vector_data_db,
            out_edges_db,
            neighbor_lists_db,
            ordinals_db,
            config,
            distance_metric: DistanceMetric::Cosine,
            spindle: SpindleConfig {
                mode: SpindleMode::None,
                keep_original: false,
                ..SpindleConfig::default()
            },
            mmap_sidecar_stem: None,
            mmap_store: StdRwLock::new(None), // legacy path — no mmap
            mmap_ordinals: StdRwLock::new(None),
            graph_out_edges_db: None, // legacy path — no graph
            neighbor_cache: Mutex::new(LruCache::new(neighbor_cache_cap())),
            vector_cache: Mutex::new(LruCache::new(vector_cache_cap())),
            nav_neighbor_cache: Mutex::new(LruCache::new(nav_cache_cap())),
            nav_vector_cache: Mutex::new(LruCache::new(nav_cache_cap())),
            nav_cache_root: StdRwLock::new(None),
            shared_caches: None,
            cache_namespace: 0,
            externalized_marker_repair_needed: AtomicBool::new(false),
            backend,
            physical_name: String::new(),
            lsm_sidecar_miss_cached_at_secs: AtomicU64::new(0),
            delete_tombstones: StdRwLock::new(HashSet::new()),
        })
    }

    /// Create the legacy unnamed/base vector core without opening any heed DBIs.
    /// Used by the SlateDB backend where segment storage is addressed entirely
    /// through [`Namespace::Segment`] prefixes.
    pub fn new_lsm(config: HNSWConfig, backend: Arc<AnyBackend>) -> Result<Self, VectorError> {
        Ok(Self {
            vectors_db: None,
            vector_data_db: None,
            out_edges_db: None,
            neighbor_lists_db: None,
            ordinals_db: None,
            config,
            distance_metric: DistanceMetric::Cosine,
            spindle: SpindleConfig {
                mode: SpindleMode::None,
                keep_original: false,
                ..SpindleConfig::default()
            },
            mmap_sidecar_stem: None,
            mmap_store: StdRwLock::new(None),
            mmap_ordinals: StdRwLock::new(None),
            graph_out_edges_db: None,
            neighbor_cache: Mutex::new(LruCache::new(neighbor_cache_cap())),
            vector_cache: Mutex::new(LruCache::new(vector_cache_cap())),
            nav_neighbor_cache: Mutex::new(LruCache::new(nav_cache_cap())),
            nav_vector_cache: Mutex::new(LruCache::new(nav_cache_cap())),
            nav_cache_root: StdRwLock::new(None),
            shared_caches: None,
            cache_namespace: 0,
            externalized_marker_repair_needed: AtomicBool::new(false),
            backend,
            physical_name: String::new(),
            lsm_sidecar_miss_cached_at_secs: AtomicU64::new(0),
            delete_tombstones: StdRwLock::new(HashSet::new()),
        })
    }

    /// Create a named VectorCore for SlateDB without creating local heed DBIs.
    pub fn new_named_lsm_with_dir(
        name: &str,
        config: HNSWConfig,
        distance_metric: DistanceMetric,
        spindle: SpindleConfig,
        data_dir: Option<&Path>,
        _dim: usize,
        backend: Arc<AnyBackend>,
    ) -> Result<Self, VectorError> {
        let mmap_sidecar_stem = data_dir.map(|dir| dir.join(name));
        let (mmap_store, mmap_ordinals) = if let Some(stem) = mmap_sidecar_stem.as_ref() {
            Self::open_lsm_mmap_sidecar(&backend, name, stem, &spindle, false)?
        } else {
            (None, None)
        };

        let shared_caches = if shared_caches_enabled() {
            Some(shared_caches_handle())
        } else {
            None
        };
        let cache_namespace = super::shared_cache::namespace_id(name, "");

        Ok(Self {
            vectors_db: None,
            vector_data_db: None,
            out_edges_db: None,
            neighbor_lists_db: None,
            ordinals_db: None,
            config,
            distance_metric,
            spindle,
            mmap_sidecar_stem,
            mmap_store: StdRwLock::new(mmap_store),
            mmap_ordinals: StdRwLock::new(mmap_ordinals.map(SidecarOrdinals::Complete)),
            graph_out_edges_db: None,
            neighbor_cache: Mutex::new(LruCache::new(neighbor_cache_cap())),
            vector_cache: Mutex::new(LruCache::new(vector_cache_cap())),
            nav_neighbor_cache: Mutex::new(LruCache::new(nav_cache_cap())),
            nav_vector_cache: Mutex::new(LruCache::new(nav_cache_cap())),
            nav_cache_root: StdRwLock::new(None),
            shared_caches,
            cache_namespace,
            externalized_marker_repair_needed: AtomicBool::new(false),
            backend,
            physical_name: name.to_string(),
            lsm_sidecar_miss_cached_at_secs: AtomicU64::new(0),
            delete_tombstones: StdRwLock::new(HashSet::new()),
        })
    }

    fn open_lsm_mmap_sidecar(
        backend: &AnyBackend,
        physical_name: &str,
        stem: &Path,
        spindle: &SpindleConfig,
        allow_backend_materialization: bool,
    ) -> Result<
        (
            Option<super::mmap_vectors::MmapBackend>,
            Option<HashMap<u128, u64>>,
        ),
        VectorError,
    > {
        let hvtq_path = stem.with_extension("hvtq");
        let hspn_path = stem.with_extension("hspn");
        let hvs8_path = stem.with_extension("hvs8");
        let hvec_path = stem.with_extension("hvec");
        let compact_spindle = Self::compact_spindle_sidecar_supported(spindle);

        if allow_backend_materialization {
            Self::restore_hvtq_sidecar_blob(backend, physical_name, &hvtq_path)?;
        }

        if hvtq_path.exists() {
            match super::mmap_vectors::MmapTurboQuantStore::open(&hvtq_path) {
                Ok(store) => {
                    let _ = fs::remove_file(&hspn_path);
                    let _ = fs::remove_file(&hvs8_path);
                    let _ = fs::remove_file(&hvec_path);
                    return Ok((Some(super::mmap_vectors::MmapBackend::Hvtq(store)), None));
                }
                Err(_) => {
                    let _ = fs::remove_file(&hvtq_path);
                }
            }
        }

        if hspn_path.exists() {
            match super::mmap_vectors::MmapSpindleStore::open(&hspn_path) {
                Ok(store) => {
                    if store.count() == 0 {
                        if allow_backend_materialization {
                            let _ = fs::remove_file(&hspn_path);
                        } else {
                            return Ok((None, None));
                        }
                    } else {
                        let _ = fs::remove_file(&hvs8_path);
                        let _ = fs::remove_file(&hvec_path);
                        return Ok((Some(super::mmap_vectors::MmapBackend::Hspn(store)), None));
                    }
                }
                Err(_) if compact_spindle && allow_backend_materialization => {
                    let _ = fs::remove_file(&hspn_path);
                }
                Err(_) => return Ok((None, None)),
            }
        }

        if compact_spindle && allow_backend_materialization {
            if let Some((store, ordinals)) = Self::materialize_spindle_sidecar_from_lsm_rows(
                backend,
                physical_name,
                &hspn_path,
                spindle,
            )? {
                let _ = fs::remove_file(&hvs8_path);
                let _ = fs::remove_file(&hvec_path);
                return Ok((Some(store), Some(ordinals)));
            }
        }

        if hvs8_path.exists() {
            match super::mmap_vectors::MmapQuantizedStore::open(&hvs8_path) {
                Ok(store) => {
                    return Ok((Some(super::mmap_vectors::MmapBackend::Hvs8(store)), None));
                }
                Err(_) => {
                    let _ = fs::remove_file(&hvs8_path);
                }
            }
        }

        if allow_backend_materialization && spindle.mode == SpindleMode::None {
            Self::restore_hvec_sidecar_blob(backend, physical_name, &hvec_path)?;
        }

        if hvec_path.exists() {
            match super::mmap_vectors::MmapVectorStore::open(&hvec_path) {
                Ok(store) => {
                    if store.count() == 0 {
                        if allow_backend_materialization {
                            let _ = fs::remove_file(&hvec_path);
                        } else {
                            return Ok((None, None));
                        }
                    } else {
                        let ordinals = if allow_backend_materialization {
                            Self::load_hvec_sidecar_ordinals_blob(backend, physical_name)?
                        } else {
                            None
                        };
                        return Ok((
                            Some(super::mmap_vectors::MmapBackend::Hvec(store)),
                            ordinals,
                        ));
                    }
                }
                Err(_) if allow_backend_materialization => {
                    let _ = fs::remove_file(&hvec_path);
                }
                Err(_) => return Ok((None, None)),
            }
        }

        if allow_backend_materialization && spindle.mode == SpindleMode::None {
            if let Some((store, ordinals)) = Self::materialize_exact_hvec_sidecar_from_lsm_rows(
                backend,
                physical_name,
                &hvec_path,
                spindle,
            )? {
                return Ok((Some(store), Some(ordinals)));
            }
        }

        Ok((None, None))
    }

    /// Create a VectorCore with namespaced database names for multi-vector support.
    /// E.g., name="dense" → databases "vectors_dense", "vector_data_dense", "hnsw_out_dense".
    pub fn new_named(
        env: &Env,
        txn: &mut RwTxn,
        name: &str,
        config: HNSWConfig,
        distance_metric: DistanceMetric,
        spindle: SpindleConfig,
        backend: Arc<AnyBackend>,
    ) -> Result<Self, VectorError> {
        Self::new_named_with_dir(
            env,
            txn,
            name,
            config,
            distance_metric,
            spindle,
            None,
            0,
            None,
            backend,
        )
    }

    /// Create a named VectorCore with an optional data directory for mmap vector storage.
    /// When `data_dir` is Some, vectors are also written to a flat mmap file for O(1) reads.
    ///
    /// `graph_out_edges_db` is the owning collection's graph adjacency DB
    /// (see `HelixGraphStorage::out_edges_db`). Pass `None` for standalone
    /// VectorCores without a graph attached (legacy tests, bulk builders).
    /// See the `graph_out_edges_db` field on `VectorCore` for usage details.
    pub fn new_named_with_dir(
        env: &Env,
        txn: &mut RwTxn,
        name: &str,
        config: HNSWConfig,
        distance_metric: DistanceMetric,
        spindle: SpindleConfig,
        data_dir: Option<&std::path::Path>,
        dim: usize,
        graph_out_edges_db: Option<Database<Bytes, Bytes>>,
        backend: Arc<AnyBackend>,
    ) -> Result<Self, VectorError> {
        let create_lmdb_dbs = backend.kind() == BackendKind::Lmdb;
        let vectors_db = if create_lmdb_dbs {
            Some(env.create_database(txn, Some(&format!("vectors_{}", name)))?)
        } else {
            None
        };
        let vector_data_db = if create_lmdb_dbs {
            Some(env.create_database(txn, Some(&format!("vector_data_{}", name)))?)
        } else {
            None
        };
        let out_edges_db = if create_lmdb_dbs {
            Some(env.create_database(txn, Some(&format!("hnsw_out_{}", name)))?)
        } else {
            None
        };
        let neighbor_lists_db = if create_lmdb_dbs {
            Some(env.create_database(txn, Some(&format!("hnsw_neighbors_{}", name)))?)
        } else {
            None
        };
        let ordinals_db = if create_lmdb_dbs {
            Some(env.create_database(txn, Some(&format!("vector_ordinals_{}", name)))?)
        } else {
            None
        };

        // Open or create mmap vector file for O(1) vector reads.
        //
        // Format priority on reopen:
        //   1. {name}.hvtq — read-only TurboProd payloads for compact
        //      TurboProd merge targets.
        //   2. {name}.hvs8 — read-only scalar quantized sidecar.
        //   3. {name}.hvec — legacy mutable f32 layout (default).
        let mmap_sidecar_stem = data_dir.map(|dir| dir.join(name));
        let mmap_store = if let Some(stem) = mmap_sidecar_stem.as_ref() {
            let hvtq_path = stem.with_extension("hvtq");
            let hvs8_path = stem.with_extension("hvs8");
            let hvec_path = stem.with_extension("hvec");
            Self::restore_hvtq_sidecar_blob(&backend, name, &hvtq_path)?;
            if hvtq_path.exists() {
                match super::mmap_vectors::MmapTurboQuantStore::open(&hvtq_path) {
                    Ok(store) => {
                        let _ = fs::remove_file(&hvs8_path);
                        let _ = fs::remove_file(&hvec_path);
                        Some(super::mmap_vectors::MmapBackend::Hvtq(store))
                    }
                    Err(_) => match super::mmap_vectors::MmapQuantizedStore::open(&hvs8_path) {
                        Ok(store) => Some(super::mmap_vectors::MmapBackend::Hvs8(store)),
                        Err(_) => match super::mmap_vectors::MmapVectorStore::open(&hvec_path) {
                            Ok(store) => Some(super::mmap_vectors::MmapBackend::Hvec(store)),
                            Err(_) => None,
                        },
                    },
                }
            } else if hvs8_path.exists() {
                match super::mmap_vectors::MmapQuantizedStore::open(&hvs8_path) {
                    Ok(store) => Some(super::mmap_vectors::MmapBackend::Hvs8(store)),
                    Err(_) => match super::mmap_vectors::MmapVectorStore::open(&hvec_path) {
                        Ok(store) => Some(super::mmap_vectors::MmapBackend::Hvec(store)),
                        Err(_) => None,
                    },
                }
            } else if hvec_path.exists() {
                match super::mmap_vectors::MmapVectorStore::open(&hvec_path) {
                    Ok(store) => Some(super::mmap_vectors::MmapBackend::Hvec(store)),
                    Err(_) => None,
                }
            } else if spindle.is_enabled() && !spindle.keep_original {
                None
            } else {
                match super::mmap_vectors::MmapVectorStore::open_or_create(&hvec_path, dim) {
                    Ok(store) => Some(super::mmap_vectors::MmapBackend::Hvec(store)),
                    Err(_) => None, // fall back to LMDB-only
                }
            }
        } else {
            None
        };

        // Shared-cache wire-up: only the "with_dir" path has a physical
        // segment name, so this is where we derive the namespace id and
        // (if enabled) grab the pod-global cache handle. When the flag
        // is off, `shared_caches` stays None and the private LRUs above
        // handle everything — exactly the pre-PR behavior.
        let shared_caches = if shared_caches_enabled() {
            Some(shared_caches_handle())
        } else {
            None
        };
        let cache_namespace = super::shared_cache::namespace_id(name, "");

        Ok(Self {
            vectors_db,
            vector_data_db,
            out_edges_db,
            neighbor_lists_db,
            ordinals_db,
            config,
            distance_metric,
            spindle,
            mmap_sidecar_stem,
            mmap_store: StdRwLock::new(mmap_store),
            mmap_ordinals: StdRwLock::new(None),
            graph_out_edges_db,
            neighbor_cache: Mutex::new(LruCache::new(neighbor_cache_cap())),
            vector_cache: Mutex::new(LruCache::new(vector_cache_cap())),
            nav_neighbor_cache: Mutex::new(LruCache::new(nav_cache_cap())),
            nav_vector_cache: Mutex::new(LruCache::new(nav_cache_cap())),
            nav_cache_root: StdRwLock::new(None),
            shared_caches,
            cache_namespace,
            externalized_marker_repair_needed: AtomicBool::new(false),
            backend,
            physical_name: name.to_string(),
            lsm_sidecar_miss_cached_at_secs: AtomicU64::new(0),
            delete_tombstones: StdRwLock::new(HashSet::new()),
        })
    }

    #[inline(always)]
    fn vector_key(id: u128, level: usize) -> Vec<u8> {
        [VECTOR_PREFIX, &id.to_be_bytes(), &level.to_be_bytes()].concat()
    }

    pub(crate) fn restore_hvtq_sidecar_blob(
        backend: &AnyBackend,
        physical_name: &str,
        hvtq_path: &std::path::Path,
    ) -> Result<bool, VectorError> {
        if hvtq_path.exists() {
            return Ok(false);
        }

        let r = backend
            .begin_read()
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        let bytes = backend
            .get_with(
                &r,
                Namespace::Segment {
                    physical_name,
                    db: SegmentDb::VectorData,
                },
                HVTQ_SIDECAR_BLOB_KEY,
                |opt| opt.map(|b| b.to_vec()),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        drop(r);

        let Some(bytes) = bytes else {
            return Ok(false);
        };
        if bytes.is_empty() {
            return Ok(false);
        }

        if let Some(parent) = hvtq_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                VectorError::VectorCoreError(format!("hvtq restore mkdir failed: {}", e))
            })?;
        }
        let tmp_path = hvtq_path.with_extension("hvtq.tmp");
        fs::write(&tmp_path, bytes).map_err(|e| {
            VectorError::VectorCoreError(format!("hvtq restore write failed: {}", e))
        })?;
        fs::rename(&tmp_path, hvtq_path).map_err(|e| {
            VectorError::VectorCoreError(format!("hvtq restore rename failed: {}", e))
        })?;
        Ok(true)
    }

    pub(crate) fn restore_hvec_sidecar_blob(
        backend: &AnyBackend,
        physical_name: &str,
        hvec_path: &std::path::Path,
    ) -> Result<bool, VectorError> {
        if hvec_path.exists() {
            return Ok(false);
        }

        let r = backend
            .begin_read()
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        let bytes = backend
            .get_with(
                &r,
                Namespace::Segment {
                    physical_name,
                    db: SegmentDb::VectorData,
                },
                HVEC_SIDECAR_BLOB_KEY,
                |opt| opt.map(|b| b.to_vec()),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        drop(r);

        let Some(bytes) = bytes else {
            return Ok(false);
        };
        if bytes.is_empty() {
            return Ok(false);
        }

        if let Some(parent) = hvec_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                VectorError::VectorCoreError(format!("hvec restore mkdir failed: {}", e))
            })?;
        }
        let tmp_path = hvec_path.with_extension("hvec.tmp");
        fs::write(&tmp_path, bytes).map_err(|e| {
            VectorError::VectorCoreError(format!("hvec restore write failed: {}", e))
        })?;
        fs::rename(&tmp_path, hvec_path).map_err(|e| {
            VectorError::VectorCoreError(format!("hvec restore rename failed: {}", e))
        })?;
        Ok(true)
    }

    fn encode_hvec_sidecar_ordinals(entries: &[(u128, u64)]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(12 + entries.len().saturating_mul(24));
        bytes.extend_from_slice(HVEC_SIDECAR_ORDINALS_MAGIC);
        bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (id, ordinal) in entries {
            bytes.extend_from_slice(&id.to_be_bytes());
            bytes.extend_from_slice(&ordinal.to_le_bytes());
        }
        bytes
    }

    fn decode_hvec_sidecar_ordinals(bytes: &[u8]) -> Result<HashMap<u128, u64>, VectorError> {
        if bytes.len() < 12 || &bytes[0..4] != HVEC_SIDECAR_ORDINALS_MAGIC {
            return Err(VectorError::InvalidVectorData);
        }
        let count = u64::from_le_bytes(bytes[4..12].try_into().unwrap()) as usize;
        let expected = 12usize
            .checked_add(
                count
                    .checked_mul(24)
                    .ok_or(VectorError::InvalidVectorData)?,
            )
            .ok_or(VectorError::InvalidVectorData)?;
        if bytes.len() != expected {
            return Err(VectorError::InvalidVectorData);
        }

        let mut ordinals = HashMap::with_capacity(count);
        for chunk in bytes[12..].chunks_exact(24) {
            let id = u128::from_be_bytes(chunk[0..16].try_into().unwrap());
            let ordinal = u64::from_le_bytes(chunk[16..24].try_into().unwrap());
            ordinals.insert(id, ordinal);
        }
        Ok(ordinals)
    }

    fn load_hvec_sidecar_ordinals_blob(
        backend: &AnyBackend,
        physical_name: &str,
    ) -> Result<Option<HashMap<u128, u64>>, VectorError> {
        if backend.kind() != BackendKind::Lsm {
            return Ok(None);
        }

        let r = backend
            .begin_read()
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        let bytes = backend
            .get_with(
                &r,
                Namespace::Segment {
                    physical_name,
                    db: SegmentDb::VectorData,
                },
                HVEC_SIDECAR_ORDINALS_BLOB_KEY,
                |opt| opt.map(|b| b.to_vec()),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        drop(r);

        let Some(bytes) = bytes else {
            return Ok(None);
        };
        if bytes.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self::decode_hvec_sidecar_ordinals(&bytes)?))
    }

    fn build_sidecar_ordinal_map(point_ids: &[u128]) -> Result<HashMap<u128, u64>, VectorError> {
        let mut ordinals = HashMap::with_capacity(point_ids.len());
        for (ordinal, id) in point_ids.iter().enumerate() {
            let ordinal = u64::try_from(ordinal)
                .map_err(|_| VectorError::VectorCoreError("sidecar ordinal exceeds u64".into()))?;
            ordinals.insert(*id, ordinal);
        }
        Ok(ordinals)
    }

    fn clear_lsm_sidecar_miss_cache(&self) {
        self.lsm_sidecar_miss_cached_at_secs
            .store(0, AtomicOrdering::Relaxed);
    }

    fn record_lsm_sidecar_miss(&self) {
        self.lsm_sidecar_miss_cached_at_secs
            .store(unix_secs_now_nonzero(), AtomicOrdering::Relaxed);
    }

    /// True while a cached "sidecar missing" probe is still authoritative. On
    /// the writer a miss never expires (only local materialization changes the
    /// answer, and those paths clear the cache). On a reader replica the
    /// writer can seal the sidecar upstream, so the miss expires after the TTL
    /// and the next read re-probes; an expired entry is cleared so only one
    /// re-probe pays the scan.
    fn lsm_sidecar_miss_is_cached(&self) -> bool {
        let cached_at = self
            .lsm_sidecar_miss_cached_at_secs
            .load(AtomicOrdering::Relaxed);
        if cached_at == 0 {
            return false;
        }
        if !self.backend.is_reader_replica() {
            return true;
        }
        let ttl = lsm_sidecar_miss_ttl_secs();
        if lsm_sidecar_miss_expired(cached_at, unix_secs_now_nonzero(), ttl) {
            metrics::counter!("helix_lsm_sidecar_materialize_total", "outcome" => "miss_expired")
                .increment(1);
            // Best-effort: only reset if still the entry we judged expired.
            let _ = self.lsm_sidecar_miss_cached_at_secs.compare_exchange(
                cached_at,
                0,
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
            );
            return false;
        }
        true
    }

    fn collect_exact_hvec_rows_from_lsm(
        backend: &AnyBackend,
        physical_name: &str,
        spindle: &SpindleConfig,
    ) -> Result<Vec<(u128, Vec<f32>)>, VectorError> {
        if backend.kind() != BackendKind::Lsm || spindle.mode != SpindleMode::None {
            return Ok(Vec::new());
        }

        let r = backend
            .begin_read()
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        let mut encoded_rows = Vec::new();
        let id_start = VECTOR_PREFIX.len();
        let id_end = id_start + std::mem::size_of::<u128>();
        let level_end = id_end + std::mem::size_of::<usize>();
        backend
            .scan_streaming(
                &r,
                Namespace::Segment {
                    physical_name,
                    db: SegmentDb::Vectors,
                },
                KeyRange::prefix(VECTOR_PREFIX),
                |key, value| {
                    if key.len() < level_end || Self::is_vector_marker(value) {
                        return true;
                    }
                    let Ok(level_arr) = key[id_end..level_end].try_into() else {
                        return true;
                    };
                    if usize::from_be_bytes(level_arr) != 0 {
                        return true;
                    }
                    let Ok(id_arr) = key[id_start..id_end].try_into() else {
                        return true;
                    };
                    encoded_rows.push((u128::from_be_bytes(id_arr), value.to_vec()));
                    true
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;

        let mut rows = Vec::with_capacity(encoded_rows.len());
        for (id, encoded) in encoded_rows {
            rows.push((id, decode_vector(&encoded)?));
        }
        rows.sort_unstable_by_key(|(id, _)| *id);
        Ok(rows)
    }

    fn compact_spindle_sidecar_supported(spindle: &SpindleConfig) -> bool {
        spindle.is_enabled() && !spindle.keep_original && spindle.mode != SpindleMode::TurboProd
    }

    fn collect_spindle_encoded_rows_from_lsm(
        backend: &AnyBackend,
        physical_name: &str,
        spindle: &SpindleConfig,
    ) -> Result<Vec<(u128, Vec<u8>)>, VectorError> {
        if backend.kind() != BackendKind::Lsm || !Self::compact_spindle_sidecar_supported(spindle) {
            return Ok(Vec::new());
        }

        let r = backend
            .begin_read()
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        let mut rows = Vec::new();
        let id_start = VECTOR_PREFIX.len();
        let id_end = id_start + std::mem::size_of::<u128>();
        let level_end = id_end + std::mem::size_of::<usize>();
        backend
            .scan(
                &r,
                Namespace::Segment {
                    physical_name,
                    db: SegmentDb::Vectors,
                },
                KeyRange::prefix(VECTOR_PREFIX),
                |key, value| {
                    if key.len() < level_end || Self::is_vector_marker(value) {
                        return true;
                    }
                    let Ok(level_arr) = key[id_end..level_end].try_into() else {
                        return true;
                    };
                    if usize::from_be_bytes(level_arr) != 0 {
                        return true;
                    }
                    let Ok(id_arr) = key[id_start..id_end].try_into() else {
                        return true;
                    };
                    rows.push((u128::from_be_bytes(id_arr), value.to_vec()));
                    true
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;

        rows.sort_unstable_by_key(|(id, _)| *id);
        Ok(rows)
    }

    fn materialize_exact_hvec_sidecar_from_lsm_rows(
        backend: &AnyBackend,
        physical_name: &str,
        hvec_path: &Path,
        spindle: &SpindleConfig,
    ) -> Result<Option<(super::mmap_vectors::MmapBackend, HashMap<u128, u64>)>, VectorError> {
        let rows = Self::collect_exact_hvec_rows_from_lsm(backend, physical_name, spindle)?;
        if rows.is_empty() {
            return Ok(None);
        }
        let point_ids = rows.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        let slices = rows
            .iter()
            .map(|(_, row)| row.as_slice())
            .collect::<Vec<_>>();
        let store = super::mmap_vectors::MmapVectorStore::create_from_slices(hvec_path, &slices)?;
        let ordinals = Self::build_sidecar_ordinal_map(&point_ids)?;
        Ok(Some((
            super::mmap_vectors::MmapBackend::Hvec(store),
            ordinals,
        )))
    }

    fn materialize_spindle_sidecar_from_lsm_rows(
        backend: &AnyBackend,
        physical_name: &str,
        hspn_path: &Path,
        spindle: &SpindleConfig,
    ) -> Result<Option<(super::mmap_vectors::MmapBackend, HashMap<u128, u64>)>, VectorError> {
        let rows = Self::collect_spindle_encoded_rows_from_lsm(backend, physical_name, spindle)?;
        if rows.is_empty() {
            return Ok(None);
        }
        let point_ids = rows.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        let slices = rows
            .iter()
            .map(|(_, row)| row.as_slice())
            .collect::<Vec<_>>();
        let store = super::mmap_vectors::MmapSpindleStore::create_from_encoded(hspn_path, &slices)?;
        let ordinals = Self::build_sidecar_ordinal_map(&point_ids)?;
        Ok(Some((
            super::mmap_vectors::MmapBackend::Hspn(store),
            ordinals,
        )))
    }

    pub(crate) fn ensure_exact_hvec_sidecar_for_write(
        &self,
        dim: usize,
    ) -> Result<(), VectorError> {
        if self.backend.kind() != BackendKind::Lsm || dim == 0 {
            return Ok(());
        }
        let Some(stem) = self.mmap_sidecar_stem.as_ref() else {
            return Ok(());
        };
        if self
            .mmap_store
            .read()
            .ok()
            .and_then(|slot| slot.as_ref().map(|_| ()))
            .is_some()
        {
            return Ok(());
        }

        let hvec_path = stem.with_extension("hvec");
        let mut slot = self
            .mmap_store
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        if slot.is_none() {
            let store = super::mmap_vectors::MmapVectorStore::open_or_create(&hvec_path, dim)?;
            *slot = Some(super::mmap_vectors::MmapBackend::Hvec(store));
            self.clear_lsm_sidecar_miss_cache();
        }
        Ok(())
    }

    #[inline(always)]
    fn out_edges_key(source_id: u128, level: usize, sink_id: Option<u128>) -> Vec<u8> {
        match sink_id {
            Some(sink_id) => [
                source_id.to_be_bytes().as_slice(),
                level.to_be_bytes().as_slice(),
                sink_id.to_be_bytes().as_slice(),
            ]
            .concat()
            .to_vec(),
            None => [
                source_id.to_be_bytes().as_slice(),
                level.to_be_bytes().as_slice(),
            ]
            .concat()
            .to_vec(),
        }
    }

    #[inline(always)]
    fn neighbor_list_key(source_id: u128, level: usize) -> Vec<u8> {
        [
            source_id.to_be_bytes().as_slice(),
            level.to_be_bytes().as_slice(),
        ]
        .concat()
        .to_vec()
    }

    #[inline(always)]
    fn encode_neighbor_block(block: &NeighborBlock) -> Vec<u8> {
        if !block.has_approx_codes() {
            let mut encoded = Vec::with_capacity(block.ids.len() * std::mem::size_of::<u128>());
            for neighbor in &block.ids {
                encoded.extend_from_slice(&neighbor.to_be_bytes());
            }
            return encoded;
        }

        let count = u32::try_from(block.ids.len()).unwrap_or(0);
        let code_len = u32::try_from(block.approx_code_len).unwrap_or(0);
        let mut encoded = Vec::with_capacity(
            12 + block.ids.len() * std::mem::size_of::<u128>() + block.approx_codes.len(),
        );
        encoded.extend_from_slice(NEIGHBOR_BLOCK_MAGIC);
        encoded.extend_from_slice(&count.to_be_bytes());
        encoded.extend_from_slice(&code_len.to_be_bytes());
        for neighbor in &block.ids {
            encoded.extend_from_slice(&neighbor.to_be_bytes());
        }
        encoded.extend_from_slice(&block.approx_codes);
        encoded
    }

    fn decode_neighbor_block(bytes: &[u8]) -> Result<NeighborBlock, VectorError> {
        if bytes.starts_with(NEIGHBOR_BLOCK_MAGIC) {
            if bytes.len() < 12 {
                return Err(VectorError::InvalidVectorData);
            }

            let count = u32::from_be_bytes(
                bytes[4..8]
                    .try_into()
                    .map_err(|_| VectorError::InvalidVectorData)?,
            ) as usize;
            let code_len = u32::from_be_bytes(
                bytes[8..12]
                    .try_into()
                    .map_err(|_| VectorError::InvalidVectorData)?,
            ) as usize;
            let ids_bytes_len = count
                .checked_mul(std::mem::size_of::<u128>())
                .ok_or(VectorError::InvalidVectorData)?;
            let codes_bytes_len = count
                .checked_mul(code_len)
                .ok_or(VectorError::InvalidVectorData)?;
            if bytes.len() != 12 + ids_bytes_len + codes_bytes_len {
                return Err(VectorError::InvalidVectorData);
            }

            let mut ids = Vec::with_capacity(count);
            for chunk in bytes[12..12 + ids_bytes_len].chunks_exact(std::mem::size_of::<u128>()) {
                let id = u128::from_be_bytes(
                    chunk
                        .try_into()
                        .map_err(|_| VectorError::InvalidVectorData)?,
                );
                ids.push(id);
            }

            let approx_codes = bytes[12 + ids_bytes_len..].to_vec();
            return Ok(NeighborBlock {
                ids,
                approx_code_len: code_len,
                approx_codes,
            });
        }

        if bytes.len() % std::mem::size_of::<u128>() != 0 {
            return Err(VectorError::InvalidVectorData);
        }

        let mut neighbors = Vec::with_capacity(bytes.len() / std::mem::size_of::<u128>());
        for chunk in bytes.chunks_exact(std::mem::size_of::<u128>()) {
            let id = u128::from_be_bytes(
                chunk
                    .try_into()
                    .map_err(|_| VectorError::InvalidVectorData)?,
            );
            neighbors.push(id);
        }
        Ok(NeighborBlock::from_ids(neighbors))
    }

    #[inline]
    fn invalidate_neighbor_cache_entry(&self, id: u128, level: usize) {
        if let Some(shared) = self.shared_caches.as_ref() {
            let key = super::shared_cache::CacheKey::new(self.cache_namespace, id, level);
            shared.neighbor.pop(&key);
            shared.nav_neighbor.pop(&key);
            return;
        }
        if let Ok(mut cache) = self.neighbor_cache.lock() {
            cache.pop(&(id, level));
        }
        if let Ok(mut cache) = self.nav_neighbor_cache.lock() {
            cache.pop(&(id, level));
        }
    }

    #[inline]
    fn invalidate_vector_cache_entry(&self, id: u128, level: usize) {
        if let Some(shared) = self.shared_caches.as_ref() {
            let key = super::shared_cache::CacheKey::new(self.cache_namespace, id, level);
            shared.vector.pop(&key);
            shared.nav_vector.pop(&key);
            return;
        }
        if let Ok(mut cache) = self.vector_cache.lock() {
            cache.pop(&(id, level));
        }
        if let Ok(mut cache) = self.nav_vector_cache.lock() {
            cache.pop(&(id, level));
        }
    }

    #[inline]
    fn clear_neighbor_cache(&self) {
        // When shared, clearing on `self` is intentionally a no-op: the
        // pod-global cache is cleared via `invalidate_namespace` at
        // collection drop, and individual segment clears would stomp
        // peer cores sharing the same namespace. Other cores' entries
        // survive LRU eviction instead.
        if self.shared_caches.is_some() {
            return;
        }
        if let Ok(mut cache) = self.neighbor_cache.lock() {
            cache.clear();
        }
        if let Ok(mut cache) = self.nav_neighbor_cache.lock() {
            cache.clear();
        }
        if let Ok(mut root) = self.nav_cache_root.write() {
            *root = None;
        }
    }

    #[inline]
    fn clear_vector_cache(&self) {
        if self.shared_caches.is_some() {
            return;
        }
        if let Ok(mut cache) = self.vector_cache.lock() {
            cache.clear();
        }
        if let Ok(mut cache) = self.nav_vector_cache.lock() {
            cache.clear();
        }
    }

    #[inline]
    fn cache_vector(&self, id: u128, level: usize, data: &[f32]) {
        if let Some(shared) = self.shared_caches.as_ref() {
            let key = super::shared_cache::CacheKey::new(self.cache_namespace, id, level);
            if level > 0 {
                shared.nav_vector.put(key, data.to_vec());
            }
            shared.vector.put(key, data.to_vec());
            return;
        }
        if level > 0 {
            if let Ok(mut cache) = self.nav_vector_cache.lock() {
                cache.put((id, level), data.to_vec());
            }
        }
        if let Ok(mut cache) = self.vector_cache.lock() {
            cache.put((id, level), data.to_vec());
        }
    }

    #[inline]
    fn can_use_global_vector_cache(read_context: &AnyRead<'_>) -> bool {
        !matches!(read_context, AnyRead::LsmReader(Some(_)))
    }

    #[inline]
    fn cache_vector_for_read(
        &self,
        read_context: &AnyRead<'_>,
        id: u128,
        level: usize,
        data: &[f32],
    ) {
        if Self::can_use_global_vector_cache(read_context) {
            self.cache_vector(id, level, data);
        }
    }

    /// Namespace for one of this segment's five sub-DBs, addressed by the
    /// segment's physical name. Replaces the raw `heed3::Database` handle so
    /// access routes through the backend seam (`self.backend`).
    #[inline]
    fn seg_ns(&self, db: SegmentDb) -> Namespace<'_> {
        Namespace::Segment {
            physical_name: &self.physical_name,
            db,
        }
    }

    #[inline]
    fn backend_label(&self) -> &'static str {
        match self.backend.kind() {
            BackendKind::Lmdb => "lmdb",
            BackendKind::Lsm => "lsm",
        }
    }

    #[inline]
    fn observe_hnsw_build_phase(
        &self,
        path: &'static str,
        phase: &'static str,
        points: usize,
        elapsed: std::time::Duration,
    ) {
        metrics::histogram!(
            "helix_vector_core_hnsw_build_phase_ms",
            "backend" => self.backend_label(),
            "path" => path,
            "phase" => phase
        )
        .record(elapsed.as_secs_f64() * 1000.0);
        if phase == "total" {
            metrics::histogram!(
                "helix_vector_core_hnsw_build_points",
                "backend" => self.backend_label(),
                "path" => path
            )
            .record(points as f64);
        }
    }

    #[inline]
    fn observe_flat_search(
        &self,
        labels: VectorSearchMetrics<'_>,
        mode: &'static str,
        target_k: usize,
        result_count: usize,
        elapsed: std::time::Duration,
    ) {
        metrics::histogram!(
            "helix_vector_core_search_stage_ms",
            "collection" => labels.collection.to_string(),
            "vector" => labels.vector.to_string(),
            "stage" => "flat_total",
            "segment_count" => labels.segment_count.to_string(),
            "segment_index" => labels.segment_index.to_string(),
            "ef" => target_k.to_string(),
            "filtered" => if labels.filtered { "true" } else { "false" }.to_string(),
            "sidecar_format" => self.sidecar_format_label().to_string(),
        )
        .record(elapsed.as_secs_f64() * 1000.0);
        metrics::histogram!(
            "helix_vector_core_flat_search_results",
            "collection" => labels.collection.to_string(),
            "vector" => labels.vector.to_string(),
            "mode" => mode,
            "segment_count" => labels.segment_count.to_string(),
            "segment_index" => labels.segment_index.to_string(),
        )
        .record(result_count as f64);
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn observe_hnsw_level_counts(
        &self,
        labels: Option<VectorSearchMetrics<'_>>,
        mode: &'static str,
        level: usize,
        ef: usize,
        visited: usize,
        neighbor_blocks: usize,
        neighbor_edges_seen: usize,
        neighbor_edges_scored: usize,
        approx_blocks: usize,
        approx_candidates_kept: usize,
        filtered_out: usize,
        result_candidates: usize,
        hydrated_results: usize,
    ) {
        let Some(labels) = labels else {
            return;
        };
        let segment_count = labels.segment_count.to_string();
        let segment_index = labels.segment_index.to_string();
        let level = level.to_string();
        let ef = ef.to_string();
        let filtered = if labels.filtered { "true" } else { "false" }.to_string();
        let sidecar_format = self.sidecar_format_label().to_string();
        let record_count = |measurement: &'static str, value: usize| {
            metrics::histogram!(
                "helix_vector_core_search_items",
                "collection" => labels.collection.to_string(),
                "vector" => labels.vector.to_string(),
                "mode" => mode,
                "measurement" => measurement,
                "level" => level.clone(),
                "segment_count" => segment_count.clone(),
                "segment_index" => segment_index.clone(),
                "ef" => ef.clone(),
                "filtered" => filtered.clone(),
                "sidecar_format" => sidecar_format.clone(),
            )
            .record(value as f64);
        };
        record_count("visited", visited);
        record_count("neighbor_blocks", neighbor_blocks);
        record_count("neighbor_edges_seen", neighbor_edges_seen);
        record_count("neighbor_edges_scored", neighbor_edges_scored);
        record_count("approx_blocks", approx_blocks);
        record_count("approx_candidates_kept", approx_candidates_kept);
        record_count("filtered_out", filtered_out);
        record_count("result_candidates", result_candidates);
        record_count("hydrated_results", hydrated_results);
    }

    /// True when the given segment namespace holds no keys. Early-stops on the
    /// first key, mirroring `Database::len(txn)? == 0` without a full count.
    fn ns_empty(&self, txn: &RoTxn, db: SegmentDb) -> Result<bool, VectorError> {
        let mut empty = true;
        self.backend
            .scan_heed(txn, self.seg_ns(db), KeyRange::all(), |_k, _v| {
                empty = false;
                false
            })
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        Ok(empty)
    }

    fn ns_empty_be(
        &self,
        r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
        db: SegmentDb,
    ) -> Result<bool, VectorError> {
        let mut empty = true;
        self.backend
            .scan(r, self.seg_ns(db), KeyRange::all(), |_k, _v| {
                empty = false;
                false
            })
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        Ok(empty)
    }

    /// Delete every key in this segment namespace. Seam-routed replacement for
    /// `Database::clear` (`mdb_drop`): collect all keys, then `delete_raw` each.
    /// Byte-identical end state (empty keyspace).
    fn ns_clear_all(&self, txn: &mut RwTxn, db: SegmentDb) -> Result<(), VectorError> {
        let mut keys: Vec<Vec<u8>> = Vec::new();
        self.backend
            .scan_heed(txn, self.seg_ns(db), KeyRange::all(), |k, _v| {
                keys.push(k.to_vec());
                true
            })
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        for k in keys {
            self.backend
                .delete_heed(txn, self.seg_ns(db), &k)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        }
        Ok(())
    }

    /// Drain up to `max_records` keys from this segment namespace and report
    /// whether it is now empty. Seam-routed replacement for the chunked
    /// `iter_mut`/`del_current` drain used by the segment reaper: it deletes a
    /// bounded number of records per call so the writer gate releases between
    /// chunks, then scans for any remainder.
    fn ns_drain_chunk(
        &self,
        txn: &mut RwTxn,
        db: SegmentDb,
        max_records: usize,
    ) -> Result<bool, VectorError> {
        let max_records = max_records.max(1);
        let mut keys: Vec<Vec<u8>> = Vec::new();
        self.backend
            .scan_heed(txn, self.seg_ns(db), KeyRange::all(), |k, _v| {
                keys.push(k.to_vec());
                keys.len() < max_records
            })
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        for k in keys {
            self.backend
                .delete_heed(txn, self.seg_ns(db), &k)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        }
        self.ns_empty(txn, db)
    }

    pub fn clear_chunk_db_be(
        &self,
        w: &mut AnyWrite<'_>,
        db_index: usize,
        max_records: usize,
    ) -> Result<bool, VectorError> {
        let db = match db_index {
            0 => SegmentDb::Vectors,
            1 => SegmentDb::Ordinals,
            2 => SegmentDb::VectorData,
            3 => SegmentDb::HnswOut,
            4 => SegmentDb::HnswNeighbors,
            5 => SegmentDb::IvfCentroids,
            6 => SegmentDb::IvfPostings,
            7 => SegmentDb::SimHash,
            _ => {
                return Err(VectorError::VectorCoreError(format!(
                    "clear_chunk_db_be: db_index {db_index} out of range"
                )))
            }
        };
        let max_records = max_records.max(1);
        let mut keys: Vec<Vec<u8>> = Vec::new();
        {
            let r = self
                .backend
                .begin_read()
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            self.backend
                .scan(&r, self.seg_ns(db), KeyRange::all(), |k, _v| {
                    keys.push(k.to_vec());
                    keys.len() < max_records
                })
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        }
        for k in keys {
            self.backend
                .delete(w, self.seg_ns(db), &k)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        }
        let r = self
            .backend
            .begin_read()
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        self.ns_empty_be(&r, db)
    }

    fn get_legacy_neighbor_ids(
        &self,
        r: &AnyRead<'_>,
        id: u128,
        level: usize,
    ) -> Result<NeighborBlock, VectorError> {
        let out_key = Self::out_edges_key(id, level, None);
        let mut neighbors = Vec::with_capacity(self.config.m_max_0.min(512));

        let prefix_len = out_key.len();

        self.backend
            .scan(
                r,
                self.seg_ns(SegmentDb::HnswOut),
                KeyRange::prefix(&out_key),
                |key, _v| {
                    if key.len() < prefix_len + std::mem::size_of::<u128>() {
                        return true;
                    }
                    if let Ok(arr) =
                        key[prefix_len..prefix_len + std::mem::size_of::<u128>()].try_into()
                    {
                        let neighbor_id = u128::from_be_bytes(arr);
                        if neighbor_id != id {
                            neighbors.push(neighbor_id);
                        }
                    }
                    true
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;

        Ok(NeighborBlock::from_ids(neighbors))
    }

    fn get_neighbor_block_uncached(
        &self,
        r: &AnyRead<'_>,
        id: u128,
        level: usize,
    ) -> Result<NeighborBlock, VectorError> {
        let key = Self::neighbor_list_key(id, level);
        let decoded = self
            .backend
            .get_with(
                r,
                self.seg_ns(SegmentDb::HnswNeighbors),
                key.as_ref(),
                |opt| opt.map(Self::decode_neighbor_block),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        if let Some(block) = decoded {
            return block;
        }
        self.get_legacy_neighbor_ids(r, id, level)
    }

    fn build_neighbor_block(
        &self,
        r: &AnyRead<'_>,
        level: usize,
        neighbors: &[u128],
    ) -> Result<NeighborBlock, VectorError> {
        let mut block = NeighborBlock::from_ids(neighbors.to_vec());
        if level != 0 || neighbors.is_empty() {
            return Ok(block);
        }

        let Some(first) = neighbors.first().copied() else {
            return Ok(block);
        };
        let sample = match self.get_vector(r, first, level, true) {
            Ok(sample) => sample,
            Err(_) => return Ok(block),
        };
        let Some(config) = Self::neighbor_code_config(sample.get_data().len()) else {
            return Ok(block);
        };

        let mut approx_codes = Vec::new();
        let mut approx_code_len = None;
        for neighbor_id in neighbors {
            let vector = match self.get_vector(r, *neighbor_id, level, true) {
                Ok(vector) => vector,
                Err(_) => return Ok(NeighborBlock::from_ids(neighbors.to_vec())),
            };
            let encoded = encode_vector(vector.get_data(), &config)?;
            match approx_code_len {
                Some(expected) if expected != encoded.len() => {
                    return Ok(NeighborBlock::from_ids(neighbors.to_vec()));
                }
                None => approx_code_len = Some(encoded.len()),
                _ => {}
            }
            approx_codes.extend_from_slice(&encoded);
        }

        block.approx_code_len = approx_code_len.unwrap_or_default();
        block.approx_codes = approx_codes;
        Ok(block)
    }

    fn put_neighbor_ids(
        &self,
        txn: &mut RwTxn,
        id: u128,
        level: usize,
        neighbors: &[u128],
    ) -> Result<(), VectorError> {
        let block = {
            let r = self.backend.read_borrowed(&*txn);
            self.build_neighbor_block(&r, level, neighbors)?
        };
        self.put_neighbor_block(txn, id, level, &block)
    }

    fn put_neighbor_block(
        &self,
        txn: &mut RwTxn,
        id: u128,
        level: usize,
        block: &NeighborBlock,
    ) -> Result<(), VectorError> {
        let key = Self::neighbor_list_key(id, level);
        let encoded = Self::encode_neighbor_block(block);
        self.backend
            .put_heed(
                txn,
                self.seg_ns(SegmentDb::HnswNeighbors),
                &key,
                encoded.as_slice(),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;

        let legacy_prefix = Self::out_edges_key(id, level, None);
        let mut legacy_keys: Vec<Vec<u8>> = Vec::new();
        self.backend
            .scan_heed(
                txn,
                self.seg_ns(SegmentDb::HnswOut),
                KeyRange::prefix(legacy_prefix.as_ref()),
                |stored_key, _v| {
                    legacy_keys.push(stored_key.to_vec());
                    true
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        for legacy_key in legacy_keys {
            self.backend
                .delete_heed(txn, self.seg_ns(SegmentDb::HnswOut), &legacy_key)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        }

        self.invalidate_neighbor_cache_entry(id, level);
        Ok(())
    }

    #[inline]
    fn get_new_level(&self) -> usize {
        // TODO: look at using the XOR shift algorithm for random number generation
        // Storing global rng will not be threadsafe or possible as thread rng needs to be mutable
        // Should instead using an atomic mutable seed and the XOR shift algorithm
        let mut rng = rand::rng();
        let r: f64 = rng.random::<f64>();
        assign_hnsw_level(r, self.config.m_l)
    }

    fn get_highest_level(&self, r: &AnyRead<'_>, id: u128) -> Result<usize, VectorError> {
        let prefix = [VECTOR_PREFIX, &id.to_be_bytes()].concat();
        let level_offset = prefix.len();
        let level_end = level_offset + std::mem::size_of::<usize>();
        let mut highest_level = None;

        self.backend
            .scan(
                r,
                self.seg_ns(SegmentDb::Vectors),
                KeyRange::prefix(prefix.as_slice()),
                |key, _v| {
                    if key.len() < level_end {
                        return true;
                    }
                    if let Ok(arr) = key[level_offset..level_end].try_into() {
                        let level = usize::from_be_bytes(arr);
                        highest_level =
                            Some(highest_level.map_or(level, |curr: usize| curr.max(level)));
                    }
                    true
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;

        highest_level.ok_or_else(|| VectorError::VectorNotFound(id.to_string()))
    }

    #[inline]
    pub(crate) fn get_entry_point(&self, r: &AnyRead<'_>) -> Result<HVector, VectorError> {
        let ep_id = self
            .backend
            .get_with(
                r,
                self.seg_ns(SegmentDb::Vectors),
                ENTRY_POINT_KEY.as_bytes(),
                |opt| opt.map(|b| b.to_vec()),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        if let Some(ep_id) = ep_id {
            let mut arr = [0u8; 16];
            let len = std::cmp::min(ep_id.len(), 16);
            arr[..len].copy_from_slice(&ep_id[..len]);
            let id = u128::from_be_bytes(arr);
            let level = self
                .get_highest_level(r, id)
                .map_err(|_| VectorError::EntryPointNotFound)?;

            let ep = self
                .get_vector(r, id, level, true)
                .map_err(|_| VectorError::EntryPointNotFound)?;
            Ok(ep)
        } else {
            Err(VectorError::EntryPointNotFound)
        }
    }

    /// Returns true when this vector space has built HNSW entrypoint state and
    /// can serve indexed search. Collections below the flat-scan threshold keep
    /// level-0 vectors stored but intentionally skip entrypoint/edge creation.
    pub fn has_index(&self, r: &AnyRead<'_>) -> Result<bool, VectorError> {
        let Some(ep_id) = self
            .backend
            .get_with(
                r,
                self.seg_ns(SegmentDb::Vectors),
                ENTRY_POINT_KEY.as_bytes(),
                |opt| opt.map(|b| b.to_vec()),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?
        else {
            return Ok(false);
        };

        let mut arr = [0u8; 16];
        let len = std::cmp::min(ep_id.len(), 16);
        arr[..len].copy_from_slice(&ep_id[..len]);
        Ok(self
            .get_vector(r, u128::from_be_bytes(arr), 0, false)
            .is_ok())
    }

    /// Runtime-derived ANN index mode for this segment, probed the same way
    /// `has_index` derives HNSW state (no metadata fields — bincode-positional
    /// metadata structs must not grow). HNSW is probed first so the production
    /// hot path pays nothing for the IVF branch; the extra `IVF_META_KEY` point
    /// read only happens on segments without an HNSW entry point.
    pub fn index_mode(&self, r: &AnyRead<'_>) -> Result<IndexMode, VectorError> {
        if self.has_index(r)? {
            return Ok(IndexMode::Hnsw);
        }
        let has_ivf = self
            .backend
            .get_with(
                r,
                self.seg_ns(SegmentDb::IvfCentroids),
                IVF_META_KEY.as_bytes(),
                |opt| opt.is_some(),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        Ok(if has_ivf {
            IndexMode::Ivf
        } else {
            IndexMode::Flat
        })
    }

    /// Count stored level-0 vectors for this named vector space. This is the
    /// stable per-space point count regardless of whether higher HNSW layers
    /// exist yet.
    pub fn level_zero_count(&self, r: &AnyRead<'_>) -> Result<u64, VectorError> {
        let level_offset = VECTOR_PREFIX.len() + std::mem::size_of::<u128>();
        let level_end = level_offset + std::mem::size_of::<usize>();
        let mut count = 0u64;

        self.backend
            .scan(
                r,
                self.seg_ns(SegmentDb::Vectors),
                KeyRange::prefix(VECTOR_PREFIX),
                |key, _v| {
                    if key.len() < level_end {
                        return true;
                    }
                    if let Ok(arr) = key[level_offset..level_end].try_into() {
                        if usize::from_be_bytes(arr) == 0 {
                            count += 1;
                        }
                    }
                    true
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;

        Ok(count)
    }

    /// Collect every level-0 vector id owned by this segment. Used by the
    /// re-upsert tombstone reconstruction path to find ids that live in more
    /// than one segment. O(level-0 rows) — single prefix scan, no payload
    /// reads.
    pub fn level_zero_ids(&self, r: &AnyRead<'_>) -> Result<Vec<u128>, VectorError> {
        let level_offset = VECTOR_PREFIX.len() + std::mem::size_of::<u128>();
        let level_end = level_offset + std::mem::size_of::<usize>();
        let mut ids = Vec::new();
        self.backend
            .scan(
                r,
                self.seg_ns(SegmentDb::Vectors),
                KeyRange::prefix(VECTOR_PREFIX),
                |key, _v| {
                    if key.len() < level_end {
                        return true;
                    }
                    let Ok(level_arr) = key[level_offset..level_end].try_into() else {
                        return true;
                    };
                    if usize::from_be_bytes(level_arr) != 0 {
                        return true;
                    }
                    if let Ok(id_arr) = key[VECTOR_PREFIX.len()..level_offset].try_into() {
                        ids.push(u128::from_be_bytes(id_arr));
                    }
                    true
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        Ok(ids)
    }

    /// True only when all LMDB databases owned by this segment are empty.
    /// Segment-name reuse depends on this stronger check, not just
    /// `level_zero_count`, because stale HNSW/payload rows would be inherited
    /// by the next segment that reopens the same named DBs.
    pub fn is_empty(&self, r: &AnyRead<'_>) -> Result<bool, VectorError> {
        Ok(self.ns_empty_be(r, SegmentDb::Vectors)?
            && self.ns_empty_be(r, SegmentDb::Ordinals)?
            && self.ns_empty_be(r, SegmentDb::VectorData)?
            && self.ns_empty_be(r, SegmentDb::HnswOut)?
            && self.ns_empty_be(r, SegmentDb::HnswNeighbors)?)
    }

    pub fn contains_id(&self, r: &AnyRead<'_>, id: u128) -> Result<bool, VectorError> {
        self.backend
            .get_with(
                r,
                self.seg_ns(SegmentDb::Vectors),
                Self::vector_key(id, 0).as_ref(),
                |opt| opt.is_some(),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    pub fn contains_id_be(&self, w: &AnyWrite<'_>, id: u128) -> Result<bool, VectorError> {
        self.backend
            .get_for_update(
                w,
                self.seg_ns(SegmentDb::Vectors),
                Self::vector_key(id, 0).as_ref(),
                |opt| opt.is_some(),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    #[inline]
    fn set_entry_point(&self, txn: &mut RwTxn, entry: &HVector) -> Result<(), VectorError> {
        let entry_key = ENTRY_POINT_KEY.as_bytes().to_vec();
        self.backend
            .put_heed(
                txn,
                self.seg_ns(SegmentDb::Vectors),
                &entry_key,
                &entry.get_id().to_be_bytes(),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;

        Ok(())
    }

    // #[inline(always)]
    // fn get_vector_(&self, txn: &RoTxn, id: u128) -> Result<Vec<f64>, VectorError> {
    #[inline(always)]
    fn put_raw_vector(
        &self,
        txn: &mut RwTxn,
        id: u128,
        level: usize,
        data: &[f32],
    ) -> Result<(), VectorError> {
        if self.can_externalize_turbo_quant_vector(txn, id, data.len())? {
            self.put_vector_marker(txn, id, level)?;
            self.invalidate_vector_cache_entry(id, level);
            return Ok(());
        }

        if self.can_externalize_raw_vectors() {
            if level == 0 {
                if self.put_externalized_level_zero_vector(txn, id, data)? {
                    self.invalidate_vector_cache_entry(id, level);
                    return Ok(());
                }
            } else if self
                .backend
                .get_for_update_heed(
                    txn,
                    self.seg_ns(SegmentDb::Ordinals),
                    &id.to_be_bytes(),
                    |opt| opt.is_some(),
                )
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?
            {
                self.put_vector_marker(txn, id, level)?;
                self.invalidate_vector_cache_entry(id, level);
                return Ok(());
            }
        }

        let encoded = encode_vector(data, &self.spindle)?;
        self.backend
            .put_heed(
                txn,
                self.seg_ns(SegmentDb::Vectors),
                &Self::vector_key(id, level),
                encoded.as_ref(),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        self.invalidate_vector_cache_entry(id, level);

        // Append raw f32 sidecars only when we need exact level-0 storage.
        // Compact spindle collections keep vectors compressed in LMDB; writing
        // the raw sidecar too would defeat the storage win while mutable
        // segments are waiting to merge into HVS8.
        let should_append_raw_sidecar = !self.spindle.is_enabled() || self.spindle.keep_original;
        if level == 0 && should_append_raw_sidecar {
            self.ensure_exact_hvec_sidecar_for_write(data.len())?;
            if let Ok(mut slot) = self.mmap_store.write() {
                if let Some(store) = slot.as_mut() {
                    if store.dim() == data.len() {
                        let ordinal = store.append(data)?;
                        store.refresh_read_mmap()?;
                        self.backend
                            .put_heed(
                                txn,
                                self.seg_ns(SegmentDb::Ordinals),
                                &id.to_be_bytes(),
                                &ordinal.to_le_bytes(),
                            )
                            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
                        self.cache_sidecar_ordinal(id, ordinal);
                    }
                }
            }
        }
        Ok(())
    }

    #[inline(always)]
    fn can_externalize_raw_vectors(&self) -> bool {
        matches!(self.spindle.mode, SpindleMode::None)
            && self
                .mmap_store
                .read()
                .map(|slot| slot.is_some())
                .unwrap_or(false)
    }

    #[inline(always)]
    fn can_externalize_turbo_quant_vector(
        &self,
        txn: &RoTxn,
        id: u128,
        expected_dim: usize,
    ) -> Result<bool, VectorError> {
        if self.spindle.mode != SpindleMode::TurboProd || self.spindle.keep_original {
            return Ok(false);
        }
        let Some(ordinal_bytes) = self
            .backend
            .get_with_heed(
                txn,
                self.seg_ns(SegmentDb::Ordinals),
                &id.to_be_bytes(),
                |opt| opt.map(|b| b.to_vec()),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?
        else {
            return Ok(false);
        };
        if ordinal_bytes.len() != 8 {
            return Err(VectorError::InvalidVectorData);
        }
        let ordinal = u64::from_le_bytes(
            ordinal_bytes
                .as_slice()
                .try_into()
                .map_err(|_| VectorError::InvalidVectorData)?,
        );
        let slot = self
            .mmap_store
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("mmap lock poisoned: {}", e)))?;
        let Some(store) = slot.as_ref() else {
            return Ok(false);
        };
        Ok(store.is_hvtq() && store.dim() == expected_dim && ordinal < store.count())
    }

    fn cached_turbo_quant_payload_matches(
        &self,
        id: u128,
        data: &[f32],
    ) -> Result<Option<bool>, VectorError> {
        if self.spindle.mode != SpindleMode::TurboProd || self.spindle.keep_original {
            return Ok(None);
        }
        let Some(ordinal) = self.cached_sidecar_ordinal(id)? else {
            return Ok(None);
        };
        let slot = self
            .mmap_store
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("mmap lock poisoned: {}", e)))?;
        let Some(store) = slot.as_ref() else {
            return Ok(None);
        };
        if !store.is_hvtq() {
            return Ok(None);
        }
        let encoded = encode_vector(data, &self.spindle)?;
        Ok(Some(
            store.dim() == data.len() && store.hvtq_encoded_matches(ordinal, &encoded),
        ))
    }

    fn can_externalize_cached_turbo_quant_vector(
        &self,
        id: u128,
        expected_dim: usize,
    ) -> Result<bool, VectorError> {
        if self.spindle.mode != SpindleMode::TurboProd || self.spindle.keep_original {
            return Ok(false);
        }
        let Some(ordinal) = self.cached_sidecar_ordinal(id)? else {
            return Ok(false);
        };
        let slot = self
            .mmap_store
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("mmap lock poisoned: {}", e)))?;
        let Some(store) = slot.as_ref() else {
            return Ok(false);
        };
        Ok(store.is_hvtq() && store.dim() == expected_dim && ordinal < store.count())
    }

    #[inline(always)]
    fn is_vector_marker(bytes: &[u8]) -> bool {
        bytes.is_empty()
    }

    #[inline(always)]
    fn put_vector_marker(
        &self,
        txn: &mut RwTxn,
        id: u128,
        level: usize,
    ) -> Result<(), VectorError> {
        self.backend
            .put_heed(
                txn,
                self.seg_ns(SegmentDb::Vectors),
                &Self::vector_key(id, level),
                EXTERNALIZED_VECTOR_MARKER,
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    fn put_externalized_level_zero_vector(
        &self,
        txn: &mut RwTxn,
        id: u128,
        data: &[f32],
    ) -> Result<bool, VectorError> {
        let mut slot = self
            .mmap_store
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("mmap lock poisoned: {}", e)))?;
        let Some(store) = slot.as_mut() else {
            return Ok(false);
        };
        if store.dim() != data.len() {
            return Ok(false);
        }

        let ordinal = store.append(data)?;
        store.flush()?;
        drop(slot);

        self.backend
            .put_heed(
                txn,
                self.seg_ns(SegmentDb::Ordinals),
                &id.to_be_bytes(),
                &ordinal.to_le_bytes(),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        self.cache_sidecar_ordinal(id, ordinal);
        self.put_vector_marker(txn, id, 0)?;
        Ok(true)
    }

    #[inline(always)]
    fn put_vector_data(
        &self,
        txn: &mut RwTxn,
        id: u128,
        data: &StoredVectorData,
    ) -> Result<(), VectorError> {
        let encoded = bincode::serialize(data)?;
        self.backend
            .put_heed(
                txn,
                self.seg_ns(SegmentDb::VectorData),
                &id.to_be_bytes(),
                encoded.as_slice(),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        Ok(())
    }

    fn maybe_put_vector_data(
        &self,
        txn: &mut RwTxn,
        id: u128,
        fields: Option<HashMap<String, Value>>,
        data: &[f32],
    ) -> Result<(), VectorError> {
        let keep_original = self.spindle.is_enabled() && self.spindle.keep_original;
        let exact_sidecar_has_original = if keep_original {
            self.backend
                .get_for_update_heed(
                    txn,
                    self.seg_ns(SegmentDb::Ordinals),
                    &id.to_be_bytes(),
                    |opt| match opt {
                        Some(ordinal_bytes) => {
                            Self::ordinal_from_bytes(ordinal_bytes).and_then(|ordinal| {
                                self.has_exact_sidecar_vector(ordinal, Some(data.len()))
                            })
                        }
                        None => Ok(false),
                    },
                )
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))??
        } else {
            false
        };
        let store_original_in_lmdb = keep_original && !exact_sidecar_has_original;

        if fields.is_some() || store_original_in_lmdb {
            let stored = StoredVectorData {
                fields: fields.unwrap_or_default(),
                original_vector: if store_original_in_lmdb {
                    Some(data.to_vec())
                } else {
                    None
                },
            };
            self.put_vector_data(txn, id, &stored)?;
        }
        Ok(())
    }

    /// Flush the mmap sidecar to disk (header count + fsync).
    /// Call after a batch of inserts.
    pub fn flush_mmap(&self) -> Result<(), VectorError> {
        if let Ok(mut slot) = self.mmap_store.write() {
            if let Some(store) = slot.as_mut() {
                store.flush()?;
            }
        }
        Ok(())
    }

    /// Convert this segment's mmap sidecar from HVEC (f32) to HVS8
    /// (scalar-quantized u8). Idempotent: returns `Ok(false)` if the
    /// backend is already HVS8 or there is no mmap store.
    ///
    /// Intended for use during the merge-publish phase, when the
    /// segment is in `Building` role and not yet referenced from
    /// `space.segments` — no concurrent readers can race the swap.
    /// Outside that window the caller must guarantee exclusivity.
    pub fn quantize_mmap(&self) -> Result<bool, VectorError> {
        if self.spindle.is_enabled() && self.spindle.keep_original {
            return Ok(false);
        }
        let mut slot = self
            .mmap_store
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("mmap lock poisoned: {}", e)))?;
        let Some(store) = slot.as_mut() else {
            return Ok(false);
        };
        store.convert_to_hvs8()
    }

    /// Materialize this immutable segment's level-0 sidecar directly as HVS8
    /// from the prepared rows used to publish the segment.
    ///
    /// This is the compact-spindle publish path: mutable rows remain LMDB-only,
    /// then the sealed segment gets one complete quantized mmap file. Ordinals
    /// are intentionally not written here; callers must invoke
    /// `write_prepared_sidecar_ordinals` in a write transaction only after this
    /// file has been created and synced.
    pub fn materialize_hvs8_sidecar_from_prepared(
        &self,
        prepared: &PreparedIndex,
    ) -> Result<bool, VectorError> {
        if !self.spindle.is_enabled() || self.spindle.keep_original {
            return Ok(false);
        }
        if prepared.point_ids.is_empty() || prepared.original_data.is_empty() {
            return Ok(false);
        }
        if prepared.point_ids.len() != prepared.original_data.len() {
            return Err(VectorError::VectorCoreError(format!(
                "prepared sidecar row mismatch: {} ids, {} vectors",
                prepared.point_ids.len(),
                prepared.original_data.len()
            )));
        }
        let rows = prepared.live_raw_slices()?;
        let Some(stem) = self.mmap_sidecar_stem.as_ref() else {
            return Ok(false);
        };

        let hvs8_path = stem.with_extension("hvs8");
        let hvtq_path = stem.with_extension("hvtq");
        let hvec_path = stem.with_extension("hvec");
        let quantized =
            match super::mmap_vectors::MmapQuantizedStore::create_from_slices(&hvs8_path, &rows) {
                Ok(store) => store,
                Err(err) => {
                    let _ = fs::remove_file(&hvs8_path);
                    return Err(err);
                }
            };

        let mut slot = self
            .mmap_store
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("mmap lock poisoned: {}", e)))?;
        *slot = Some(super::mmap_vectors::MmapBackend::Hvs8(quantized));
        self.clear_lsm_sidecar_miss_cache();
        let _ = fs::remove_file(hvec_path);
        let _ = fs::remove_file(hvtq_path);
        self.clear_vector_cache();
        Ok(true)
    }

    /// Materialize this immutable compact TurboProd segment directly as HVTQ
    /// from the prepared rows used to publish the segment.
    ///
    /// HVTQ stores the encoded TurboProd payloads in one mmap sidecar. Once
    /// ordinals are attached, LMDB vector rows can be markers instead of
    /// duplicating the same payload bytes per point.
    pub fn materialize_tq_sidecar_from_prepared(
        &self,
        prepared: &PreparedIndex,
    ) -> Result<bool, VectorError> {
        if self.spindle.mode != SpindleMode::TurboProd || self.spindle.keep_original {
            return Ok(false);
        }
        if prepared.point_ids.is_empty() {
            return Ok(false);
        }
        if let Some(encoded_rows) = prepared.live_encoded_slices() {
            return self.materialize_tq_sidecar_from_encoded_rows(&encoded_rows);
        }
        if prepared.original_data.is_empty() {
            return Ok(false);
        }
        if prepared.point_ids.len() != prepared.original_data.len() {
            return Err(VectorError::VectorCoreError(format!(
                "prepared sidecar row mismatch: {} ids, {} vectors",
                prepared.point_ids.len(),
                prepared.original_data.len()
            )));
        }
        let Some(stem) = self.mmap_sidecar_stem.as_ref() else {
            return Ok(false);
        };

        let hvtq_path = stem.with_extension("hvtq");
        let hvs8_path = stem.with_extension("hvs8");
        let hvec_path = stem.with_extension("hvec");
        let rows = prepared.live_raw_slices()?;
        let turbo_quant = super::mmap_vectors::MmapTurboQuantStore::create_from_slices(
            &hvtq_path,
            &rows,
            &self.spindle,
        );
        let turbo_quant = match turbo_quant {
            Ok(store) => store,
            Err(err) => {
                let _ = fs::remove_file(&hvtq_path);
                return Err(err);
            }
        };

        let mut slot = self
            .mmap_store
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("mmap lock poisoned: {}", e)))?;
        *slot = Some(super::mmap_vectors::MmapBackend::Hvtq(turbo_quant));
        self.clear_lsm_sidecar_miss_cache();
        let _ = fs::remove_file(hvec_path);
        let _ = fs::remove_file(hvs8_path);
        self.clear_vector_cache();
        Ok(true)
    }

    fn materialize_tq_sidecar_from_encoded_rows(
        &self,
        encoded_rows: &[&[u8]],
    ) -> Result<bool, VectorError> {
        if self.spindle.mode != SpindleMode::TurboProd || self.spindle.keep_original {
            return Ok(false);
        }
        if encoded_rows.is_empty() {
            return Ok(false);
        }
        let Some(stem) = self.mmap_sidecar_stem.as_ref() else {
            return Ok(false);
        };

        let hvtq_path = stem.with_extension("hvtq");
        let hvs8_path = stem.with_extension("hvs8");
        let hvec_path = stem.with_extension("hvec");
        let turbo_quant =
            super::mmap_vectors::MmapTurboQuantStore::create_from_encoded(&hvtq_path, encoded_rows);
        let turbo_quant = match turbo_quant {
            Ok(store) => store,
            Err(err) => {
                let _ = fs::remove_file(&hvtq_path);
                return Err(err);
            }
        };

        let mut slot = self
            .mmap_store
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("mmap lock poisoned: {}", e)))?;
        *slot = Some(super::mmap_vectors::MmapBackend::Hvtq(turbo_quant));
        self.clear_lsm_sidecar_miss_cache();
        let _ = fs::remove_file(hvec_path);
        let _ = fs::remove_file(hvs8_path);
        self.clear_vector_cache();
        Ok(true)
    }

    fn hvtq_sidecar_blob_exists_be(&self, r: &AnyRead<'_>) -> Result<bool, VectorError> {
        let bytes = self
            .backend
            .get_with(
                r,
                self.seg_ns(SegmentDb::VectorData),
                HVTQ_SIDECAR_BLOB_KEY,
                |opt| opt.map(|b| !b.is_empty()),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        Ok(bytes.unwrap_or(false))
    }

    fn has_level_zero_encoded_row_be(&self, r: &AnyRead<'_>) -> Result<bool, VectorError> {
        let id_start = VECTOR_PREFIX.len();
        let id_end = id_start + std::mem::size_of::<u128>();
        let level_end = id_end + std::mem::size_of::<usize>();
        let mut found = false;
        self.backend
            .scan(
                r,
                self.seg_ns(SegmentDb::Vectors),
                KeyRange::prefix(VECTOR_PREFIX),
                |key, value| {
                    if key.len() < level_end {
                        return true;
                    }
                    let Ok(level_arr) = key[id_end..level_end].try_into() else {
                        return true;
                    };
                    if usize::from_be_bytes(level_arr) != 0 || Self::is_vector_marker(value) {
                        return true;
                    }
                    found = true;
                    false
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        Ok(found)
    }

    pub(crate) fn hvtq_sidecar_backfill_needed_be(
        &self,
        r: &AnyRead<'_>,
    ) -> Result<bool, VectorError> {
        if self.spindle.mode != SpindleMode::TurboProd || self.spindle.keep_original {
            return Ok(false);
        }
        Ok(!self.hvtq_sidecar_blob_exists_be(r)? && self.has_level_zero_encoded_row_be(r)?)
    }

    pub(crate) fn prepare_hvtq_sidecar_backfill_be(
        &self,
        r: &AnyRead<'_>,
    ) -> Result<Option<Vec<u128>>, VectorError> {
        if !self.hvtq_sidecar_backfill_needed_be(r)? {
            return Ok(None);
        }

        let mut rows = self
            .collect_level_zero_encoded_map(r)?
            .into_iter()
            .collect::<Vec<_>>();
        if rows.is_empty() {
            return Ok(None);
        }
        rows.sort_unstable_by_key(|(id, _)| *id);
        let encoded_rows = rows
            .iter()
            .map(|(_, bytes)| bytes.as_slice())
            .collect::<Vec<_>>();
        if !self.materialize_tq_sidecar_from_encoded_rows(&encoded_rows)? {
            return Ok(None);
        }
        Ok(Some(rows.into_iter().map(|(id, _)| id).collect()))
    }

    fn read_hvtq_sidecar_blob(&self) -> Result<Option<Vec<u8>>, VectorError> {
        let Some(stem) = self.mmap_sidecar_stem.as_ref() else {
            return Ok(None);
        };
        let hvtq_path = stem.with_extension("hvtq");
        if !hvtq_path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&hvtq_path)
            .map_err(|e| VectorError::VectorCoreError(format!("hvtq read failed: {}", e)))?;
        if bytes.is_empty() {
            return Ok(None);
        }
        Ok(Some(bytes))
    }

    fn read_hvec_sidecar_blob(&self) -> Result<Option<Vec<u8>>, VectorError> {
        if self.backend.kind() != BackendKind::Lsm || self.spindle.mode != SpindleMode::None {
            return Ok(None);
        }

        let path = {
            let mut slot = self
                .mmap_store
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("mmap lock poisoned: {}", e)))?;
            let Some(store) = slot.as_mut() else {
                return Ok(None);
            };
            if !store.is_exact() || store.count() == 0 {
                return Ok(None);
            }
            store.flush()?;
            store.path().to_path_buf()
        };

        let bytes = fs::read(&path)
            .map_err(|e| VectorError::VectorCoreError(format!("hvec blob read failed: {}", e)))?;
        if bytes.is_empty() {
            return Ok(None);
        }
        Ok(Some(bytes))
    }

    fn persist_hvec_sidecar_blob_put(
        &self,
        put: impl FnOnce(&[u8]) -> Result<(), VectorError>,
    ) -> Result<bool, VectorError> {
        let Some(bytes) = self.read_hvec_sidecar_blob()? else {
            return Ok(false);
        };
        put(bytes.as_slice())?;
        Ok(true)
    }

    pub fn persist_hvec_sidecar_blob(&self, txn: &mut RwTxn) -> Result<bool, VectorError> {
        self.persist_hvec_sidecar_blob_put(|bytes| {
            self.backend
                .put_heed(
                    txn,
                    self.seg_ns(SegmentDb::VectorData),
                    HVEC_SIDECAR_BLOB_KEY,
                    bytes,
                )
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))
        })
    }

    pub fn persist_hvec_sidecar_blob_be(&self, w: &mut AnyWrite<'_>) -> Result<bool, VectorError> {
        self.persist_hvec_sidecar_blob_put(|bytes| {
            self.backend
                .put(
                    w,
                    self.seg_ns(SegmentDb::VectorData),
                    HVEC_SIDECAR_BLOB_KEY,
                    bytes,
                )
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))
        })
    }

    fn hvec_sidecar_ordinals_blob_be(
        &self,
        w: &AnyWrite<'_>,
        point_ids: &[u128],
    ) -> Result<Option<Vec<u8>>, VectorError> {
        if self.backend.kind() != BackendKind::Lsm
            || self.spindle.mode != SpindleMode::None
            || point_ids.is_empty()
        {
            return Ok(None);
        }

        let mut entries = Vec::with_capacity(point_ids.len());
        for id in point_ids {
            let key = id.to_be_bytes();
            let ordinal = self
                .backend
                .get_for_update(w, self.seg_ns(SegmentDb::Ordinals), &key, |opt| {
                    opt.map(|bytes| Self::ordinal_from_bytes(bytes))
                })
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?
                .transpose()?;
            let Some(ordinal) = ordinal else {
                return Ok(None);
            };
            entries.push((*id, ordinal));
        }
        entries.sort_unstable_by_key(|(id, _)| *id);
        Ok(Some(Self::encode_hvec_sidecar_ordinals(&entries)))
    }

    pub fn persist_hvec_sidecar_ordinals_blob_be(
        &self,
        w: &mut AnyWrite<'_>,
        point_ids: &[u128],
    ) -> Result<bool, VectorError> {
        let Some(bytes) = self.hvec_sidecar_ordinals_blob_be(w, point_ids)? else {
            return Ok(false);
        };
        self.backend
            .put(
                w,
                self.seg_ns(SegmentDb::VectorData),
                HVEC_SIDECAR_ORDINALS_BLOB_KEY,
                bytes.as_slice(),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        Ok(true)
    }

    fn persist_hvtq_sidecar_blob_put(
        &self,
        put: impl FnOnce(&[u8]) -> Result<(), VectorError>,
    ) -> Result<bool, VectorError> {
        let Some(bytes) = self.read_hvtq_sidecar_blob()? else {
            return Ok(false);
        };
        put(bytes.as_slice())?;
        Ok(true)
    }

    pub fn persist_hvtq_sidecar_blob(&self, txn: &mut RwTxn) -> Result<bool, VectorError> {
        self.persist_hvtq_sidecar_blob_put(|bytes| {
            self.backend
                .put_heed(
                    txn,
                    self.seg_ns(SegmentDb::VectorData),
                    HVTQ_SIDECAR_BLOB_KEY,
                    bytes,
                )
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))
        })
    }

    pub fn persist_hvtq_sidecar_blob_be(&self, w: &mut AnyWrite<'_>) -> Result<bool, VectorError> {
        self.persist_hvtq_sidecar_blob_put(|bytes| {
            self.backend
                .put(
                    w,
                    self.seg_ns(SegmentDb::VectorData),
                    HVTQ_SIDECAR_BLOB_KEY,
                    bytes,
                )
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))
        })
    }

    /// Attach point-id to HVS8 ordinal mappings after the HVS8 file has been
    /// fully materialized. Must run in the same LMDB env as the segment.
    pub fn write_prepared_sidecar_ordinals(
        &self,
        txn: &mut RwTxn,
        prepared: &PreparedIndex,
    ) -> Result<(), VectorError> {
        self.persist_hvtq_sidecar_blob(txn)?;
        self.persist_hvec_sidecar_blob(txn)?;
        let externalize_turbo_quant = self.mmap_is_turbo_quantized();
        for (ordinal, id) in prepared.point_ids.iter().enumerate() {
            let ordinal = u64::try_from(ordinal)
                .map_err(|_| VectorError::VectorCoreError("sidecar ordinal exceeds u64".into()))?;
            self.backend
                .put_heed(
                    txn,
                    self.seg_ns(SegmentDb::Ordinals),
                    &id.to_be_bytes(),
                    &ordinal.to_le_bytes(),
                )
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            if externalize_turbo_quant {
                self.put_vector_marker(txn, *id, 0)?;
            }
        }
        self.replace_cached_sidecar_ordinals(&prepared.point_ids)?;
        Ok(())
    }

    pub fn write_prepared_sidecar_ordinals_be(
        &self,
        w: &mut AnyWrite<'_>,
        prepared: &PreparedIndex,
    ) -> Result<(), VectorError> {
        self.write_sidecar_ordinals_be(w, &prepared.point_ids)
    }

    pub(crate) fn write_sidecar_ordinals_be(
        &self,
        w: &mut AnyWrite<'_>,
        point_ids: &[u128],
    ) -> Result<(), VectorError> {
        self.persist_hvtq_sidecar_blob_be(w)?;
        self.persist_hvec_sidecar_blob_be(w)?;
        let externalize_turbo_quant = self.mmap_is_turbo_quantized();
        for (ordinal, id) in point_ids.iter().enumerate() {
            let ordinal = u64::try_from(ordinal)
                .map_err(|_| VectorError::VectorCoreError("sidecar ordinal exceeds u64".into()))?;
            self.backend
                .put(
                    w,
                    self.seg_ns(SegmentDb::Ordinals),
                    &id.to_be_bytes(),
                    &ordinal.to_le_bytes(),
                )
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            if externalize_turbo_quant {
                self.backend
                    .put(
                        w,
                        self.seg_ns(SegmentDb::Vectors),
                        &Self::vector_key(*id, 0),
                        EXTERNALIZED_VECTOR_MARKER,
                    )
                    .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            }
        }
        self.persist_hvec_sidecar_ordinals_blob_be(w, point_ids)?;
        self.replace_cached_sidecar_ordinals(point_ids)?;
        Ok(())
    }

    /// True if the mmap sidecar is the read-only HVS8 format.
    pub fn mmap_is_quantized(&self) -> bool {
        self.mmap_store
            .read()
            .ok()
            .and_then(|slot| slot.as_ref().map(|s| s.is_hvs8()))
            .unwrap_or(false)
    }

    /// True if the mmap sidecar is the read-only HVTQ TurboProd format.
    pub fn mmap_is_turbo_quantized(&self) -> bool {
        self.mmap_store
            .read()
            .ok()
            .and_then(|slot| slot.as_ref().map(|s| s.is_hvtq()))
            .unwrap_or(false)
    }

    pub fn mmap_is_spindle_encoded(&self) -> bool {
        self.mmap_store
            .read()
            .ok()
            .and_then(|slot| slot.as_ref().map(|s| s.is_hspn()))
            .unwrap_or(false)
    }

    fn sidecar_format_label(&self) -> &'static str {
        self.mmap_store
            .read()
            .ok()
            .and_then(|slot| slot.as_ref().map(|s| s.format_label()))
            .unwrap_or("none")
    }

    #[cfg(test)]
    fn has_populated_mmap_sidecar(&self) -> bool {
        self.mmap_store
            .read()
            .ok()
            .and_then(|slot| slot.as_ref().map(|store| store.count() > 0))
            .unwrap_or(false)
    }

    /// Best-effort unlink of this segment's mmap sidecar file.
    ///
    /// The retired segment may still have live mmap readers for a short
    /// metadata-swap transition window, so this removes the directory entry
    /// without mutating the backend. Unix keeps existing mappings valid until
    /// the core is later dropped; new opens will not see the retired file.
    pub fn unlink_mmap_sidecar(&self) -> Result<bool, VectorError> {
        let started = std::time::Instant::now();
        let path = {
            let slot = self
                .mmap_store
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("mmap lock poisoned: {}", e)))?;
            let Some(store) = slot.as_ref() else {
                return Ok(false);
            };
            store.path().to_path_buf()
        };
        match fs::metadata(&path) {
            Ok(metadata) => match fs::remove_file(&path) {
                Ok(()) => {
                    metrics::counter!("helix_sidecar_files_unlinked_total").increment(1);
                    metrics::counter!("helix_sidecar_bytes_unlinked_total")
                        .increment(metadata.len());
                    metrics::histogram!("helix_sidecar_unlink_duration_ms", "mode" => "core")
                        .record(started.elapsed().as_secs_f64() * 1000.0);
                    Ok(true)
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
                Err(e) => Err(VectorError::VectorCoreError(format!(
                    "unlink mmap sidecar {} failed: {}",
                    path.display(),
                    e
                ))),
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(VectorError::VectorCoreError(format!(
                "stat mmap sidecar {} failed: {}",
                path.display(),
                e
            ))),
        }
    }

    /// Store a vector without building HNSW links yet. This is used for fresh
    /// collections below the flat-scan threshold so ingest avoids per-point graph
    /// rewiring while the collection is still small.
    pub fn insert_flat(
        &self,
        txn: &mut RwTxn,
        data: &[f32],
        nid: Option<u128>,
        fields: Option<HashMap<String, Value>>,
    ) -> Result<HVector, VectorError> {
        ensure_finite_vector(data)?;
        let id = nid.unwrap_or(uuid::Uuid::new_v4().as_u128());
        let projected = project_for_search(data, &self.spindle)?;
        self.put_raw_vector(txn, id, 0, data)?;
        self.maybe_put_vector_data(txn, id, fields, data)?;

        let mut vector = HVector::from_slice(id, 0, projected);
        vector.set_distance(0.0);
        Ok(vector)
    }

    /// Backend-native flat insert for the LSM cutover path.
    ///
    /// This intentionally stores the vector bytes in the backend namespace, not
    /// in local mmap sidecars. HNSW build/merge remains an explicit follow-up
    /// because those paths still carry `RwTxn`.
    pub fn insert_flat_be(
        &self,
        w: &mut AnyWrite<'_>,
        data: &[f32],
        nid: Option<u128>,
        fields: Option<HashMap<String, Value>>,
    ) -> Result<HVector, VectorError> {
        ensure_finite_vector(data)?;
        let id = nid.unwrap_or(uuid::Uuid::new_v4().as_u128());
        let projected = project_for_search(data, &self.spindle)?;
        let externalize = self.cached_turbo_quant_payload_matches(id, data)?;
        if matches!(externalize, Some(false)) {
            self.backend
                .delete(w, self.seg_ns(SegmentDb::Ordinals), &id.to_be_bytes())
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            self.remove_cached_sidecar_ordinals(&[id]);
        }
        let encoded;
        let stored = if matches!(externalize, Some(true)) {
            EXTERNALIZED_VECTOR_MARKER
        } else {
            encoded = encode_vector(data, &self.spindle)?;
            encoded.as_slice()
        };
        self.backend
            .put(
                w,
                self.seg_ns(SegmentDb::Vectors),
                &Self::vector_key(id, 0),
                stored,
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;

        // Flag-gated SimHash row for the new vector (LSM only): the storage
        // foundation for locality-aware layout / Hamming-prefix probing.
        // Computed from the full-precision input, same batch as the vector
        // row. Segment-scoped: dropped with the segment, absent for vectors
        // that predate the flag (readers must treat missing rows as no-code).
        if self.backend.kind() == BackendKind::Lsm
            && crate::helix_engine::vector_core::simhash::simhash_rows_enabled()
        {
            let row = crate::helix_engine::vector_core::simhash::encode_simhash_row(
                crate::helix_engine::vector_core::simhash::simhash64(
                    data,
                    crate::helix_engine::vector_core::simhash::SIMHASH_SEED,
                ),
            );
            self.backend
                .put(w, self.seg_ns(SegmentDb::SimHash), &id.to_be_bytes(), &row)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        }

        if self.backend.kind() == BackendKind::Lsm && !self.spindle.is_enabled() {
            self.ensure_exact_hvec_sidecar_for_write(data.len())?;
            if let Ok(mut slot) = self.mmap_store.write() {
                if let Some(store) = slot.as_mut() {
                    if store.dim() == data.len() {
                        let ordinal = store.append(data)?;
                        store.refresh_read_mmap()?;
                        self.backend
                            .put(
                                w,
                                self.seg_ns(SegmentDb::Ordinals),
                                &id.to_be_bytes(),
                                &ordinal.to_le_bytes(),
                            )
                            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
                        self.cache_sidecar_ordinal(id, ordinal);
                    }
                }
            }
        }

        // Persist the original (full-precision) vector when the spindle keeps it,
        // mirroring the LMDB `maybe_put_vector_data` path. Without this, the LSM
        // flat-insert path stored only the lossy spindle payload, so TurboProd's
        // exact rescore (`rescore_results` → `get_original_vector`) silently
        // became a no-op and indexed search ranked by approximate codes only —
        // collapsing recall. The original must live in the KV `VectorData`
        // namespace; this remains gated to the LSM backend so the LMDB path
        // (which stores the original via
        // `maybe_put_vector_data`) stays byte-identical.
        let keep_original = self.backend.kind() == BackendKind::Lsm
            && self.spindle.is_enabled()
            && self.spindle.keep_original;
        if fields.is_some() || keep_original {
            let stored = StoredVectorData {
                fields: fields.unwrap_or_default(),
                original_vector: if keep_original {
                    Some(data.to_vec())
                } else {
                    None
                },
            };
            let encoded = bincode::serialize(&stored)?;
            self.backend
                .put(
                    w,
                    self.seg_ns(SegmentDb::VectorData),
                    &id.to_be_bytes(),
                    encoded.as_slice(),
                )
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        }

        self.invalidate_vector_cache_entry(id, 0);
        let mut vector = HVector::from_slice(id, 0, projected);
        vector.set_distance(0.0);
        Ok(vector)
    }

    pub fn insert_write_context<F>(
        &self,
        w: &mut AnyWrite<'_>,
        data: &[f32],
        nid: Option<u128>,
        fields: Option<HashMap<String, Value>>,
    ) -> Result<HVector, VectorError>
    where
        F: Fn(&HVector) -> bool,
    {
        match w {
            AnyWrite::Lmdb(lmdb) => {
                <Self as HNSW>::insert::<F>(self, lmdb.txn_mut(), data, nid, fields)
            }
            AnyWrite::Lsm(_) => self.insert_flat_be(w, data, nid, fields),
        }
    }

    /// Choose a replacement HNSW entry point for a deleted `old_ep`.
    ///
    /// Prefers the surviving neighbor with the highest level, walking
    /// `old_ep`'s neighbor lists from its top layer down (a neighbor at layer
    /// L has level >= L, so the first non-empty layer yields the best
    /// candidates). If `old_ep` had no surviving neighbors at all, falls back
    /// to a key-only scan for the highest-level surviving row — rare (only on
    /// an isolated entry point) and still cheaper than losing the index.
    /// Returns `None` only when no level-0 row survives, i.e. the space is
    /// genuinely empty and the entry point may be cleared.
    fn replacement_entry_point(
        &self,
        r: &AnyRead<'_>,
        old_ep: &HVector,
        is_deleted: impl Fn(u128) -> bool,
    ) -> Result<Option<HVector>, VectorError> {
        let old_id = old_ep.get_id();
        let mut best: Option<(usize, u128)> = None;
        for level in (0..=old_ep.get_level()).rev() {
            let Ok(neighbor_ids) = self.get_neighbor_ids(r, old_id, level) else {
                continue;
            };
            for neighbor_id in neighbor_ids {
                if neighbor_id == old_id || is_deleted(neighbor_id) {
                    continue;
                }
                if let Ok(neighbor_level) = self.get_highest_level(r, neighbor_id) {
                    if best.is_none_or(|(best_level, _)| neighbor_level > best_level) {
                        best = Some((neighbor_level, neighbor_id));
                    }
                }
            }
            if best.is_some() {
                break;
            }
        }

        if best.is_none() {
            let id_offset = VECTOR_PREFIX.len();
            let level_offset = id_offset + std::mem::size_of::<u128>();
            let level_end = level_offset + std::mem::size_of::<usize>();
            self.backend
                .scan(
                    r,
                    self.seg_ns(SegmentDb::Vectors),
                    KeyRange::prefix(VECTOR_PREFIX),
                    |key, _v| {
                        if key.len() < level_end {
                            return true;
                        }
                        let (Ok(id_arr), Ok(level_arr)) = (
                            key[id_offset..level_offset].try_into(),
                            key[level_offset..level_end].try_into(),
                        ) else {
                            return true;
                        };
                        let id = u128::from_be_bytes(id_arr);
                        let level = usize::from_be_bytes(level_arr);
                        if id != old_id
                            && !is_deleted(id)
                            && best.is_none_or(|(best_level, _)| level > best_level)
                        {
                            best = Some((level, id));
                        }
                        true
                    },
                )
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        }

        let Some((level, id)) = best else {
            return Ok(None);
        };
        match self.get_vector(r, id, level, false) {
            Ok(vector) => Ok(Some(vector)),
            Err(VectorError::VectorNotFound(_)) => self.get_vector(r, id, 0, false).map(Some),
            Err(e) => Err(e),
        }
    }

    /// Remove a vector from the HNSW index: delete vector data, stored properties/original,
    /// and all HNSW edges (both outgoing and incoming references to this id).
    /// If the deleted vector is the entry point, it is replaced by the
    /// highest-level survivor (see `replacement_entry_point`).
    pub fn delete_vector(&self, txn: &mut RwTxn, id: u128) -> Result<(), VectorError> {
        self.clear_neighbor_cache();
        self.clear_vector_cache();
        // Check if this is the entry point and replace it before removing edges
        let entry_point = {
            let r = self.backend.read_borrowed(&*txn);
            self.get_entry_point(&r).ok()
        };
        if let Some(ep) = entry_point {
            if ep.get_id() == id {
                let replacement = {
                    let r = self.backend.read_borrowed(&*txn);
                    self.replacement_entry_point(&r, &ep, |other| other == id)?
                };
                match replacement {
                    Some(new_ep) => self.set_entry_point(txn, &new_ep)?,
                    // Last vector — clear entry point
                    None => {
                        if let Some(db) = self.vectors_db {
                            let _ = db.delete(txn, ENTRY_POINT_KEY.as_bytes());
                        }
                    }
                }
            }
        }

        // Delete vector data at all levels
        let key_prefix = [VECTOR_PREFIX, &id.to_be_bytes()].concat();
        let mut keys_to_delete: Vec<Vec<u8>> = Vec::new();
        self.backend
            .scan_heed(
                txn,
                self.seg_ns(SegmentDb::Vectors),
                KeyRange::prefix(&key_prefix),
                |k, _v| {
                    keys_to_delete.push(k.to_vec());
                    true
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        for key in keys_to_delete {
            self.backend
                .delete_heed(txn, self.seg_ns(SegmentDb::Vectors), &key)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        }

        let _ = self
            .backend
            .delete_heed(txn, self.seg_ns(SegmentDb::Ordinals), &id.to_be_bytes());

        // Delete stored vector data (properties + original vector)
        let _ =
            self.backend
                .delete_heed(txn, self.seg_ns(SegmentDb::VectorData), &id.to_be_bytes());

        // Read the deleted vector's neighbor lists to get its outgoing neighbors
        // BEFORE deleting them. This lets us do O(degree) targeted reverse-
        // reference removal instead of the old O(N) full-table scan.
        let packed_prefix = id.to_be_bytes().to_vec();
        let mut reverse_targets: Vec<(u128, usize)> = Vec::new(); // (neighbor_id, level)
        {
            let id_len = std::mem::size_of::<u128>();
            let level_len = std::mem::size_of::<usize>();
            self.backend
                .scan_heed(
                    txn,
                    self.seg_ns(SegmentDb::HnswNeighbors),
                    KeyRange::prefix(&packed_prefix),
                    |key, value| {
                        // Extract level from key: id(16) + level(8)
                        let level = if key.len() == id_len + level_len {
                            usize::from_be_bytes(
                                key[id_len..id_len + level_len].try_into().unwrap_or([0; 8]),
                            )
                        } else {
                            0
                        };
                        if let Ok(block) = Self::decode_neighbor_block(value) {
                            for &neighbor_id in &block.ids {
                                reverse_targets.push((neighbor_id, level));
                            }
                        }
                        true
                    },
                )
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        }

        // Delete the deleted vector's own neighbor blocks
        let mut packed_keys: Vec<Vec<u8>> = Vec::new();
        self.backend
            .scan_heed(
                txn,
                self.seg_ns(SegmentDb::HnswNeighbors),
                KeyRange::prefix(&packed_prefix),
                |key, _v| {
                    packed_keys.push(key.to_vec());
                    true
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        for key in packed_keys {
            self.backend
                .delete_heed(txn, self.seg_ns(SegmentDb::HnswNeighbors), &key)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        }

        // O(degree) targeted reverse-reference removal: for each neighbor that
        // the deleted vector pointed to, look up that neighbor's block at the
        // same level and remove the deleted ID.
        for (neighbor_id, level) in &reverse_targets {
            let neighbor_key = Self::neighbor_list_key(*neighbor_id, *level);
            let encoded = self
                .backend
                .get_for_update_heed(
                    txn,
                    self.seg_ns(SegmentDb::HnswNeighbors),
                    &neighbor_key,
                    |opt| {
                        let bytes = opt?;
                        let block = Self::decode_neighbor_block(bytes).ok()?;
                        let updated = block.retain_without(id);
                        if updated.ids.len() != block.ids.len() {
                            Some(Self::encode_neighbor_block(&updated))
                        } else {
                            None
                        }
                    },
                )
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            if let Some(encoded) = encoded {
                self.backend
                    .put_heed(
                        txn,
                        self.seg_ns(SegmentDb::HnswNeighbors),
                        &neighbor_key,
                        encoded.as_slice(),
                    )
                    .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            }
        }

        // Asymmetric reverse-reference fallback. The targeted loop above
        // only catches neighbors that `id` itself pointed to — the
        // symmetric case. In production, partial-failure scenarios (a
        // crash mid-insert, an interrupted segment merge, or a hand-set
        // neighbor list) can leave a block where another node points at
        // `id` but `id` does not point back. Without this scan such
        // dangling refs survive deletion and surface as stale ids in
        // search results.
        //
        // Cost: full prefix-iter over `neighbor_lists_db`. delete_vector
        // is rare (explicit user delete + tombstone GC + replication
        // re-upsert) and the per-block work is decode → cheap
        // `retain_without` → conditional put-back. Compared to the
        // mmap_vectors / out_edges_db scans this function already does,
        // this is one more linear pass — acceptable for correctness.
        let mut asymmetric_updates: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        let id_bytes = id.to_be_bytes();
        self.backend
            .scan_heed(
                txn,
                self.seg_ns(SegmentDb::HnswNeighbors),
                KeyRange::all(),
                |key, bytes| {
                    // Skip blocks we already handled: id's own deleted blocks
                    // start with id_bytes, and the targeted loop above
                    // re-encoded each (neighbor_id, level) we walked. The
                    // `retain_without` below is idempotent so we'd just produce
                    // a no-op write — gate on the prefix to avoid that.
                    if key.starts_with(&id_bytes) {
                        return true;
                    }
                    let block = match Self::decode_neighbor_block(bytes) {
                        Ok(b) => b,
                        Err(_) => return true,
                    };
                    if !block.ids.contains(&id) {
                        return true;
                    }
                    let updated = block.retain_without(id);
                    if updated.ids.len() != block.ids.len() {
                        let encoded = Self::encode_neighbor_block(&updated);
                        asymmetric_updates.push((key.to_vec(), encoded));
                    }
                    true
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        for (key, encoded) in asymmetric_updates {
            self.backend
                .put_heed(
                    txn,
                    self.seg_ns(SegmentDb::HnswNeighbors),
                    &key,
                    encoded.as_slice(),
                )
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        }

        // Delete all HNSW edges involving this id.
        // Edge key format: source_id(16 bytes) + level(8 bytes) + sink_id(16 bytes)
        let edge_prefix = id.to_be_bytes().to_vec();
        let mut edge_keys: Vec<Vec<u8>> = Vec::new();
        self.backend
            .scan_heed(
                txn,
                self.seg_ns(SegmentDb::HnswOut),
                KeyRange::prefix(&edge_prefix),
                |k, _v| {
                    edge_keys.push(k.to_vec());
                    true
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        for key in &edge_keys {
            self.backend
                .delete_heed(txn, self.seg_ns(SegmentDb::HnswOut), key)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            // Delete reverse edge: neighbor(16) + level(8) + id(16)
            let id_len = std::mem::size_of::<u128>();
            let level_len = std::mem::size_of::<usize>();
            if key.len() == id_len + level_len + id_len {
                let neighbor_id = &key[id_len..id_len + level_len + id_len][level_len..];
                let level_bytes = &key[id_len..id_len + level_len];
                let reverse_key = [neighbor_id, level_bytes, &id.to_be_bytes()].concat();
                let _ =
                    self.backend
                        .delete_heed(txn, self.seg_ns(SegmentDb::HnswOut), &reverse_key);
            }
        }

        self.remove_cached_sidecar_ordinals(&[id]);
        Ok(())
    }

    /// Batched delete for high-throughput mutation paths.
    ///
    /// Synchronous deletes remove vector rows and the deleted ids' own neighbor
    /// blocks. Reverse neighbor references are left lazy by default: search
    /// already skips stale neighbor ids whose vector row is gone, and the
    /// optimizer/merge path rebuilds topology. Strict reverse patching can be
    /// re-enabled with `HELIX_DENSE_DELETE_PATCH_REVERSE_EDGES=1`, but it is
    /// too expensive for large live collections under overwrite pressure.
    ///
    /// Background: per-step instrumentation showed `delete_vector` at
    /// p50 174ms / p99 688ms per call. With chunks of 64 re-upserts on a
    /// hot collection that's 11s p99 just for the delete pass — the
    /// dominant cost in `apply_upsert_points_in_txn` and the source of
    /// the 14.5s p99 on `lmdb_write_txn_total_ms{shared}`. Aggregating
    /// the reverse-edge updates removes the quadratic blow-up.
    pub fn delete_vectors_batch(&self, txn: &mut RwTxn, ids: &[u128]) -> Result<(), VectorError> {
        if ids.is_empty() {
            return Ok(());
        }
        let observe_metrics = crate::telemetry::hot_path_metrics_enabled();
        let record_step = |step: &'static str, started: std::time::Instant| {
            if observe_metrics {
                metrics::histogram!("helix_dense_delete_batch_step_ms", "step" => step)
                    .record(started.elapsed().as_secs_f64() * 1000.0);
            }
        };
        let record_items = |kind: &'static str, count: usize| {
            if observe_metrics {
                metrics::histogram!("helix_dense_delete_batch_items", "kind" => kind)
                    .record(count as f64);
            }
        };
        record_items("ids", ids.len());

        self.clear_neighbor_cache();
        self.clear_vector_cache();

        let id_set: std::collections::HashSet<u128> = ids.iter().copied().collect();

        // Entry-point handling. If the current entry point is in the batch,
        // walk every deleted id's level-0 neighbors looking for a survivor
        // (one not in id_set). Falls back to clearing if none exists.
        let step_started = std::time::Instant::now();
        // Resolve the replacement entry point (or lack thereof) through a
        // borrowed read view first, then drop the view before mutating `txn`.
        let entry_point_replacement: Option<Option<HVector>> = {
            let r = self.backend.read_borrowed(&*txn);
            match self.get_entry_point(&r) {
                Ok(ep) if id_set.contains(&ep.get_id()) => {
                    Some(self.replacement_entry_point(&r, &ep, |other| id_set.contains(&other))?)
                }
                _ => None,
            }
        };
        if let Some(survivor) = entry_point_replacement {
            match survivor {
                Some(v) => {
                    self.set_entry_point(txn, &v)?;
                }
                None => {
                    if let Some(db) = self.vectors_db {
                        let _ = db.delete(txn, ENTRY_POINT_KEY.as_bytes());
                    }
                }
            }
        }
        record_step("entry_point", step_started);

        // Aggregate reverse-edge updates: (neighbor_id, level) -> {ids to remove}.
        // Skip neighbors that are themselves in the batch (their block will be
        // dropped wholesale below, so any update would just be discarded).
        let patch_reverse_edges = patch_reverse_edges_on_batch_delete();
        let id_len = std::mem::size_of::<u128>();
        let level_len = std::mem::size_of::<usize>();
        // Lower-bound capacity: each id contributes at least one (id, level) entry.
        // Real fan-out is ids.len() × levels-with-edges (typically ~2x), but a
        // single-rehash margin past ids.len() is enough to remove most growth.
        let mut updates = patch_reverse_edges.then(|| {
            std::collections::HashMap::<(u128, usize), Vec<u128>>::with_capacity(ids.len())
        });
        let mut own_neighbor_keys: Vec<Vec<u8>> = Vec::with_capacity(ids.len());
        let step_started = std::time::Instant::now();
        for &id in ids {
            let id_bytes = id.to_be_bytes();
            self.backend
                .scan_heed(
                    txn,
                    self.seg_ns(SegmentDb::HnswNeighbors),
                    KeyRange::prefix(&id_bytes),
                    |key, value| {
                        own_neighbor_keys.push(key.to_vec());
                        if let Some(updates) = updates.as_mut() {
                            let level = if key.len() == id_len + level_len {
                                usize::from_be_bytes(
                                    key[id_len..id_len + level_len].try_into().unwrap_or([0; 8]),
                                )
                            } else {
                                0
                            };
                            if let Ok(block) = Self::decode_neighbor_block(value) {
                                for &neighbor_id in &block.ids {
                                    if !id_set.contains(&neighbor_id) {
                                        updates.entry((neighbor_id, level)).or_default().push(id);
                                    }
                                }
                            }
                        }
                        true
                    },
                )
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        }
        record_step("collect_neighbor_keys", step_started);
        record_items("own_neighbor_keys", own_neighbor_keys.len());
        if let Some(updates) = updates.as_ref() {
            record_items("reverse_update_groups", updates.len());
        }

        // Per-id payload deletion: vector data, ordinals, vector_data,
        // own neighbor blocks, own out_edges (forward + reverse).
        let step_started = std::time::Instant::now();
        let mut vector_rows_deleted = 0usize;
        let mut edge_rows_deleted = 0usize;
        for &id in ids {
            let id_bytes = id.to_be_bytes();

            let key_prefix = [VECTOR_PREFIX, &id_bytes].concat();
            let mut keys: Vec<Vec<u8>> = Vec::new();
            self.backend
                .scan_heed(
                    txn,
                    self.seg_ns(SegmentDb::Vectors),
                    KeyRange::prefix(&key_prefix),
                    |k, _v| {
                        keys.push(k.to_vec());
                        true
                    },
                )
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            for key in keys {
                self.backend
                    .delete_heed(txn, self.seg_ns(SegmentDb::Vectors), &key)
                    .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
                vector_rows_deleted += 1;
            }

            let _ = self
                .backend
                .delete_heed(txn, self.seg_ns(SegmentDb::Ordinals), &id_bytes);
            let _ = self
                .backend
                .delete_heed(txn, self.seg_ns(SegmentDb::VectorData), &id_bytes);

            let mut edge_keys: Vec<Vec<u8>> = Vec::new();
            self.backend
                .scan_heed(
                    txn,
                    self.seg_ns(SegmentDb::HnswOut),
                    KeyRange::prefix(&id_bytes),
                    |k, _v| {
                        edge_keys.push(k.to_vec());
                        true
                    },
                )
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            for key in &edge_keys {
                self.backend
                    .delete_heed(txn, self.seg_ns(SegmentDb::HnswOut), key)
                    .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
                edge_rows_deleted += 1;
                if patch_reverse_edges && key.len() == id_len + level_len + id_len {
                    let level_bytes = &key[id_len..id_len + level_len];
                    let neighbor_id = &key[id_len + level_len..];
                    let reverse_key = [neighbor_id, level_bytes, &id_bytes].concat();
                    if let Some(db) = self.out_edges_db {
                        let _ = db.delete(txn, &reverse_key);
                    }
                }
            }
        }
        record_step("payload_delete", step_started);
        record_items("vector_rows_deleted", vector_rows_deleted);
        record_items("edge_rows_deleted", edge_rows_deleted);

        let own_neighbor_key_count = own_neighbor_keys.len();
        let step_started = std::time::Instant::now();
        for key in own_neighbor_keys {
            self.backend
                .delete_heed(txn, self.seg_ns(SegmentDb::HnswNeighbors), &key)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        }
        record_step("own_neighbor_delete", step_started);

        // Single r-m-w per (neighbor, level), even if dozens of deleted
        // ids shared this neighbor. This is the load-bearing optimization
        // versus `delete_vector` called N times.
        if let Some(updates) = updates {
            let step_started = std::time::Instant::now();
            for ((neighbor_id, level), ids_to_remove) in updates {
                let key = Self::neighbor_list_key(neighbor_id, level);
                let encoded = self
                    .backend
                    .get_for_update_heed(txn, self.seg_ns(SegmentDb::HnswNeighbors), &key, |opt| {
                        let bytes = opt?;
                        let block = Self::decode_neighbor_block(bytes).ok()?;
                        let updated = block.retain_without_slice(&ids_to_remove);
                        if updated.ids.len() != block.ids.len() {
                            Some(Self::encode_neighbor_block(&updated))
                        } else {
                            None
                        }
                    })
                    .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
                if let Some(encoded) = encoded {
                    self.backend
                        .put_heed(
                            txn,
                            self.seg_ns(SegmentDb::HnswNeighbors),
                            &key,
                            encoded.as_slice(),
                        )
                        .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
                }
            }
            record_step("reverse_patch", step_started);
        } else if crate::telemetry::hot_path_metrics_enabled() {
            metrics::counter!("helix_dense_delete_lazy_reverse_edges_total")
                .increment(own_neighbor_key_count as u64);
        }

        self.remove_cached_sidecar_ordinals(ids);
        Ok(())
    }

    pub fn delete_vectors_batch_be(
        &self,
        w: &mut AnyWrite<'_>,
        ids: &[u128],
    ) -> Result<(), VectorError> {
        if ids.is_empty() {
            return Ok(());
        }
        self.clear_neighbor_cache();
        self.clear_vector_cache();

        for chunk in ids.chunks(dense_delete_read_snapshot_chunk()) {
            let r = self
                .backend
                .begin_read()
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            #[cfg(test)]
            DENSE_DELETE_READ_SNAPSHOT_OPENS_FOR_TEST
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            for &id in chunk {
                let id_bytes = id.to_be_bytes();
                let key_prefix = [VECTOR_PREFIX, &id_bytes].concat();

                let mut vector_keys = Vec::new();
                self.backend
                    .scan(
                        &r,
                        self.seg_ns(SegmentDb::Vectors),
                        KeyRange::prefix(&key_prefix),
                        |k, _v| {
                            vector_keys.push(k.to_vec());
                            true
                        },
                    )
                    .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
                for key in vector_keys {
                    self.backend
                        .delete(w, self.seg_ns(SegmentDb::Vectors), &key)
                        .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
                }

                self.backend
                    .delete(w, self.seg_ns(SegmentDb::Ordinals), &id_bytes)
                    .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
                self.backend
                    .delete(w, self.seg_ns(SegmentDb::VectorData), &id_bytes)
                    .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;

                for db in [SegmentDb::HnswNeighbors, SegmentDb::HnswOut] {
                    let mut keys = Vec::new();
                    self.backend
                        .scan(&r, self.seg_ns(db), KeyRange::prefix(&id_bytes), |k, _v| {
                            keys.push(k.to_vec());
                            true
                        })
                        .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
                    for key in keys {
                        self.backend
                            .delete(w, self.seg_ns(db), &key)
                            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
                    }
                }
            }
        }
        self.remove_cached_sidecar_ordinals(ids);
        Ok(())
    }

    // ── Deferred-repair delete tombstones (issue #29 "fix 2") ──────────
    //
    // `HELIX_LSM_DELETE_TOMBSTONES`: an alternative to `delete_vectors_batch_be`
    // that records a delete-tombstone per id instead of removing the Vectors /
    // Ordinals / VectorData / HnswNeighbors / HnswOut rows immediately. The
    // rows are reclaimed later when this segment is next selected for merge
    // (`run_chunked_merge_publish`'s export loop drops tombstoned ids, mirroring
    // the existing re-upsert tombstone reclamation in `named_vectors.rs`).
    // Search must never surface a tombstoned id: `is_delete_tombstoned` is
    // consulted at every point a candidate id resolves to a result (see the
    // `score_neighbor_distance` / `get_vector` call sites below).

    /// O(1) in-memory membership check — zero backend/S3 cost, safe to call
    /// once per HNSW candidate on the search hot path. Checked unconditionally
    /// (independent of `HELIX_LSM_DELETE_TOMBSTONES`'s current value): the gate
    /// is keyed on set contents, not the flag, so a flag flip to OFF after
    /// deletes exist still excludes those ids from search.
    #[inline]
    pub(crate) fn is_delete_tombstoned(&self, id: u128) -> bool {
        self.delete_tombstones
            .read()
            .map(|set| set.contains(&id))
            .unwrap_or(false)
    }

    /// Count of ids delete-tombstoned on this segment. In-memory, O(1) —
    /// used for the merge-debt fold and the deleted-ratio vacuum ceiling.
    pub(crate) fn deleted_count(&self) -> usize {
        self.delete_tombstones
            .read()
            .map(|set| set.len())
            .unwrap_or(0)
    }

    /// Record a delete-tombstone for every id in `ids` instead of removing
    /// their rows. Idempotent: an id already tombstoned (repeat delete) is
    /// skipped, matching the no-op-on-repeat-delete contract
    /// `delete_vectors_batch_be` provides today. Returns the ids newly
    /// recorded so the caller can apply them to the in-memory set via
    /// `apply_delete_tombstones` AFTER the batch commits — mutating the set
    /// here, before `w` is durable, would suppress live vectors from search
    /// if the commit later fails.
    pub(crate) fn tombstone_delete_batch_be(
        &self,
        w: &mut AnyWrite<'_>,
        ids: &[u128],
    ) -> Result<Vec<u128>, VectorError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut staged: Vec<u128> = Vec::new();
        for &id in ids {
            if self.is_delete_tombstoned(id) || staged.contains(&id) {
                continue;
            }
            let key = delete_tombstone_key(&self.physical_name, id);
            self.backend
                .put(w, Namespace::DenseTombstones, &key, &[])
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            staged.push(id);
        }
        Ok(staged)
    }

    /// Insert ids returned by `tombstone_delete_batch_be` into the in-memory
    /// set. Call only after the write batch that staged them has committed.
    pub(crate) fn apply_delete_tombstones(&self, ids: &[u128]) {
        if ids.is_empty() {
            return;
        }
        if let Ok(mut set) = self.delete_tombstones.write() {
            set.extend(ids.iter().copied());
        }
    }

    /// Rebuild the in-memory delete-tombstone set from the durable keyspace.
    /// Unlike the re-upsert tombstone reconstruction (which infers
    /// supersession from cross-segment analysis of live data),
    /// delete-tombstones are written durably by the delete path itself — this
    /// only replays what is already there. Called on collection open (both
    /// backends) and, on a reader replica, when a newly-discovered segment's
    /// core is opened. Runs unconditionally, independent of the flag's
    /// current value, for the same reason `is_delete_tombstoned` is
    /// unconditional. Returns the number of ids loaded.
    pub(crate) fn reconstruct_delete_tombstones_be(
        &self,
        r: &AnyRead<'_>,
    ) -> Result<usize, VectorError> {
        let prefix = delete_tombstone_segment_prefix(&self.physical_name);
        let mut loaded: HashSet<u128> = HashSet::new();
        self.backend
            .scan(
                r,
                Namespace::DenseTombstones,
                KeyRange::prefix(&prefix),
                |k, _v| {
                    if let Some(id) = decode_delete_tombstone_id(&self.physical_name, k) {
                        loaded.insert(id);
                    }
                    true
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        let count = loaded.len();
        if let Ok(mut set) = self.delete_tombstones.write() {
            *set = loaded;
        }
        Ok(count)
    }

    /// Drop this segment's durable delete-tombstones after its rows have
    /// been reclaimed by a merge/retire. Mirrors
    /// `NamedVectorManager::clear_tombstones_for_segment_be` for the
    /// re-upsert keyspace. Idempotent; a segment with no delete-tombstones is
    /// a cheap no-op scan. Returns the number of keys removed.
    ///
    /// The in-memory set is deliberately left intact: this runs inside the
    /// caller's uncommitted write batch, and clearing the set before commit
    /// would resurrect deleted ids in this process if the commit fails. A
    /// retiring core is dropped after the commit lands, so the stale
    /// in-memory entries are unreachable on success.
    pub(crate) fn clear_delete_tombstones_be(
        &self,
        r: &AnyRead<'_>,
        w: &mut AnyWrite<'_>,
    ) -> Result<usize, VectorError> {
        let prefix = delete_tombstone_segment_prefix(&self.physical_name);
        let mut keys: Vec<Vec<u8>> = Vec::new();
        self.backend
            .scan(
                r,
                Namespace::DenseTombstones,
                KeyRange::prefix(&prefix),
                |k, _v| {
                    keys.push(k.to_vec());
                    true
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        let removed = keys.len();
        for k in &keys {
            self.backend
                .delete(w, Namespace::DenseTombstones, k)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        }
        Ok(removed)
    }

    /// Heed-`RwTxn` counterpart of `clear_delete_tombstones_be`, for retire
    /// paths that only hold a raw LMDB write txn (no `AnyWrite` bridge exists
    /// from one — see `NamedVectorManager::clear_tombstones_for_segment`,
    /// which is in the same position for the re-upsert keyspace). Routes
    /// through the heed-only backend seam (`scan_heed`/`delete_heed`), which
    /// is only reachable when the active backend is `AnyBackend::Lmdb` — the
    /// only backend this is ever called against.
    pub(crate) fn clear_delete_tombstones(&self, txn: &mut RwTxn) -> Result<usize, VectorError> {
        let prefix = delete_tombstone_segment_prefix(&self.physical_name);
        let mut keys: Vec<Vec<u8>> = Vec::new();
        self.backend
            .scan_heed(
                txn,
                Namespace::DenseTombstones,
                KeyRange::prefix(&prefix),
                |k, _v| {
                    keys.push(k.to_vec());
                    true
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        let removed = keys.len();
        for k in &keys {
            self.backend
                .delete_heed(txn, Namespace::DenseTombstones, k)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        }
        if let Ok(mut set) = self.delete_tombstones.write() {
            set.clear();
        }
        Ok(removed)
    }

    fn get_vector_data(&self, r: &AnyRead<'_>, id: u128) -> Result<StoredVectorData, VectorError> {
        self.backend
            .get_with(
                r,
                self.seg_ns(SegmentDb::VectorData),
                &id.to_be_bytes(),
                |opt| {
                    let Some(bytes) = opt else {
                        return Ok(StoredVectorData::default());
                    };

                    if let Ok(data) = bincode::deserialize::<StoredVectorData>(bytes) {
                        return Ok(data);
                    }

                    if let Ok(fields) = bincode::deserialize::<HashMap<String, Value>>(bytes) {
                        return Ok(StoredVectorData {
                            fields,
                            original_vector: None,
                        });
                    }

                    Err(VectorError::InvalidVectorData)
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?
    }

    /// Sequentially scan this segment's `VectorData` namespace once, returning an
    /// id -> [`StoredVectorData`] map.
    ///
    /// The per-point build loop (`prepare_index_from_flat`) otherwise issues one
    /// random point-get per vector against `VectorData`. On the LSM/SlateDB
    /// backend a giant segment's `VectorData` spans gigabytes of SSTs, so N
    /// scattered point-gets thrash the (capped) object-store cache and storm S3
    /// (the "evictor queue full" symptom: builds crawl and never publish). One
    /// ordered prefix scan reads the SSTs in key order (evict-as-you-go,
    /// cache-friendly) and is the bounded-working-set replacement. Rows that fail
    /// to deserialize are skipped, matching `get_vector_data`'s self-healing
    /// fallback (a missing/garbled entry is treated as `StoredVectorData::default`).
    fn collect_vector_data_map(
        &self,
        r: &AnyRead<'_>,
    ) -> Result<HashMap<u128, StoredVectorData>, VectorError> {
        let mut out: HashMap<u128, StoredVectorData> = HashMap::new();
        self.backend
            .scan(
                r,
                self.seg_ns(SegmentDb::VectorData),
                KeyRange::all(),
                |k, v| {
                    let Ok(id_arr) = k.try_into() else {
                        return true;
                    };
                    let id = u128::from_be_bytes(id_arr);
                    if let Ok(data) = bincode::deserialize::<StoredVectorData>(v) {
                        out.insert(id, data);
                    } else if let Ok(fields) = bincode::deserialize::<HashMap<String, Value>>(v) {
                        out.insert(
                            id,
                            StoredVectorData {
                                fields,
                                original_vector: None,
                            },
                        );
                    }
                    true
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        Ok(out)
    }

    /// Sequentially scan this segment's level-0 `Vectors` namespace once,
    /// returning an id -> stored-bytes map for every non-marker row.
    ///
    /// Same rationale as [`collect_vector_data_map`]: it replaces the per-point
    /// random `get_encoded_vector` point-get in the build loop with one ordered
    /// pass. Marker rows (externalized to the local mmap sidecar) are omitted
    /// here; the caller resolves their encoded payload from the local sidecar by
    /// ordinal, which is a local-mmap read, not an S3 read.
    fn collect_level_zero_encoded_map(
        &self,
        r: &AnyRead<'_>,
    ) -> Result<HashMap<u128, Vec<u8>>, VectorError> {
        let mut out: HashMap<u128, Vec<u8>> = HashMap::new();
        let id_start = VECTOR_PREFIX.len();
        let id_end = id_start + std::mem::size_of::<u128>();
        let level_end = id_end + std::mem::size_of::<usize>();
        self.backend
            .scan(
                r,
                self.seg_ns(SegmentDb::Vectors),
                KeyRange::prefix(VECTOR_PREFIX),
                |key, value| {
                    if key.len() < level_end {
                        return true;
                    }
                    let Ok(level_arr) = key[id_end..level_end].try_into() else {
                        return true;
                    };
                    if usize::from_be_bytes(level_arr) != 0 {
                        return true;
                    }
                    if Self::is_vector_marker(value) {
                        return true;
                    }
                    if let Ok(id_arr) = key[id_start..id_end].try_into() {
                        out.insert(u128::from_be_bytes(id_arr), value.to_vec());
                    }
                    true
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        Ok(out)
    }

    #[cfg(test)]
    pub fn debug_fields(
        &self,
        r: &AnyRead<'_>,
        id: u128,
    ) -> Result<HashMap<String, Value>, VectorError> {
        Ok(self.get_vector_data(r, id)?.fields)
    }

    pub fn export_level_zero(
        &self,
        r: &AnyRead<'_>,
    ) -> Result<Vec<(u128, Vec<f32>, HashMap<String, Value>)>, VectorError> {
        let mut exported = Vec::new();
        for vector in self.get_all_vectors(r, Some(0))? {
            let stored = self.get_vector_data(r, vector.get_id())?;
            let raw = stored
                .original_vector
                .clone()
                .map(Ok)
                .or_else(|| {
                    self.read_sidecar_vector_if_available(
                        r,
                        vector.get_id(),
                        Some(vector.get_data().len()),
                    )
                    .transpose()
                })
                .transpose()?
                .unwrap_or_else(|| vector.get_data().to_vec());
            exported.push((vector.get_id(), raw, stored.fields));
        }
        Ok(exported)
    }

    pub fn clear(&self, txn: &mut RwTxn) -> Result<(), VectorError> {
        self.ns_clear_all(txn, SegmentDb::Vectors)?;
        self.ns_clear_all(txn, SegmentDb::Ordinals)?;
        self.ns_clear_all(txn, SegmentDb::VectorData)?;
        self.ns_clear_all(txn, SegmentDb::HnswOut)?;
        self.ns_clear_all(txn, SegmentDb::HnswNeighbors)?;
        self.ns_clear_all(txn, SegmentDb::IvfCentroids)?;
        self.ns_clear_all(txn, SegmentDb::IvfPostings)?;
        self.clear_neighbor_cache();
        self.clear_vector_cache();
        Ok(())
    }

    /// Delete up to `max_per_db` records from each of this segment's five
    /// LMDB databases. Returns `Ok(true)` once everything is drained
    /// (caller can stop), `Ok(false)` if any database still has remaining
    /// records (caller must invoke again in a fresh write txn).
    ///
    /// This is the chunked replacement for `clear()`. The full `clear`
    /// path calls `mdb_drop` on five B-trees inside a single write txn —
    /// O(pages) per tree. For a retired HNSW segment with 50 k+ vectors
    /// (≈ 300 k records across the five DBs) the drop took ~39 s and
    /// monopolised the per-collection LMDB writer for that entire window,
    /// blocking every concurrent upsert worker. The chunked variant lets
    /// `SegmentReaper` commit between chunks so the writer gate is
    /// released ~every 100-200 ms — normal writes interleave instead of
    /// queueing behind the cleanup.
    ///
    /// Caller must invoke `clear_caches` once after the final
    /// `Ok(true)`. The cache flush is not invoked here so the per-chunk
    /// path stays purely LMDB-bound.
    pub fn clear_chunk(&self, txn: &mut RwTxn, max_per_db: usize) -> Result<bool, VectorError> {
        let max_per_db = max_per_db.max(1);
        let mut more = false;

        // Each DB is drained up to `max_per_db` records through the backend
        // seam; `ns_drain_chunk` returns `true` when that DB is now empty.
        for db in [
            SegmentDb::Vectors,
            SegmentDb::Ordinals,
            SegmentDb::VectorData,
            SegmentDb::HnswOut,
            SegmentDb::HnswNeighbors,
            SegmentDb::IvfCentroids,
            SegmentDb::IvfPostings,
        ] {
            if !self.ns_drain_chunk(txn, db, max_per_db)? {
                more = true;
            }
        }

        // Caller contract: `true` = drained (caller stops), `false` = more
        // chunks needed. See doc comment for rationale; flipping internal
        // `more` keeps the per-DB loop reasoning natural.
        Ok(!more)
    }

    /// Number of LMDB databases drained by `clear_chunk_db`. Exposed so
    /// the segment reaper can iterate `0..DB_COUNT` without hard-coding it.
    pub const REAPER_DB_COUNT: usize = 8;

    /// Single-database variant of `clear_chunk`. Drains up to
    /// `max_records` records from the database identified by `db_index`
    /// and returns `Ok(true)` when that database is empty, `Ok(false)`
    /// when more chunks are needed.
    ///
    /// `clear_chunk` bundles all five DBs into one writer-held txn; on a
    /// retired HNSW segment the `neighbor_lists_db` rows are 5–50× larger
    /// than the others so the combined hold ran 50–66 s in production.
    /// Splitting per-DB lets the reaper open one short `with_write_txn`
    /// per database and yield between them, capping the worst-case
    /// writer hold to one DB's chunk.
    ///
    /// `db_index` mapping (matches the order in `clear_chunk`):
    ///   0 → vectors_db          (small key/value rows)
    ///   1 → ordinals_db         (small key/value rows)
    ///   2 → vector_data_db      (per-row vector blob)
    ///   3 → out_edges_db        (small rows, dupsort)
    ///   4 → neighbor_lists_db   (largest values: M·u32 per level)
    ///   5 → ivf_centroids_db    (≤2 blobs: centroid table + meta)
    ///   6 → ivf_postings_db     (one id blob per centroid)
    ///   7 → hnsw_simhash_db     (16-byte SimHash row per vector)
    pub fn clear_chunk_db(
        &self,
        txn: &mut RwTxn,
        db_index: usize,
        max_records: usize,
    ) -> Result<bool, VectorError> {
        let db = match db_index {
            0 => SegmentDb::Vectors,
            1 => SegmentDb::Ordinals,
            2 => SegmentDb::VectorData,
            3 => SegmentDb::HnswOut,
            4 => SegmentDb::HnswNeighbors,
            5 => SegmentDb::IvfCentroids,
            6 => SegmentDb::IvfPostings,
            7 => SegmentDb::SimHash,
            _ => {
                return Err(VectorError::VectorCoreError(format!(
                    "clear_chunk_db: db_index {db_index} out of range (0..{})",
                    Self::REAPER_DB_COUNT
                )))
            }
        };
        self.ns_drain_chunk(txn, db, max_records)
    }

    /// Drop the in-memory caches that `clear` flushes on completion.
    /// `clear_chunk` callers invoke this once after the final chunk so
    /// stale neighbor / vector entries don't survive the reaper drain.
    pub fn clear_caches(&self) {
        self.clear_neighbor_cache();
        self.clear_vector_cache();
    }

    /// Read stored fields for a vector. Used internally by segment merge/vacuum
    /// to preserve payload fields when rebuilding segments.
    pub(crate) fn debug_fields_internal(
        &self,
        r: &AnyRead<'_>,
        id: u128,
    ) -> HashMap<String, Value> {
        self.get_vector_data(r, id)
            .map(|d| d.fields)
            .unwrap_or_default()
    }

    pub(crate) fn get_original_vector(
        &self,
        r: &AnyRead<'_>,
        id: u128,
    ) -> Result<Option<Vec<f32>>, VectorError> {
        let data = self.get_vector_data(r, id)?;
        if data.original_vector.is_some() {
            return Ok(data.original_vector);
        }
        self.read_exact_sidecar_vector(r, id, None)
    }

    fn original_vector_for_rebuild(
        &self,
        r: &AnyRead<'_>,
        vector: &HVector,
        stored: &StoredVectorData,
    ) -> Result<Vec<f32>, VectorError> {
        if let Some(original) = stored.original_vector.clone() {
            return Ok(original);
        }
        Ok(self
            .read_exact_sidecar_vector(r, vector.get_id(), None)?
            .unwrap_or_else(|| vector.get_data().to_vec()))
    }

    fn get_encoded_vector(
        &self,
        r: &AnyRead<'_>,
        id: u128,
        level: usize,
    ) -> Result<Vec<u8>, VectorError> {
        let key = Self::vector_key(id, level);
        let stored = self
            .backend
            .get_with(r, self.seg_ns(SegmentDb::Vectors), key.as_ref(), |opt| {
                opt.map(|b| b.to_vec())
            })
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        match stored {
            Some(bytes) if Self::is_vector_marker(&bytes) => {
                if let Some(encoded) = self.read_encoded_vector_from_sidecar(r, id)? {
                    return Ok(encoded);
                }
                Ok(encode_vector(
                    &self.resolve_marker_vector(r, id, level)?,
                    &self.spindle,
                )?)
            }
            Some(bytes) => Ok(bytes),
            None if level > 0 => self.get_encoded_vector(r, id, 0),
            None => Err(VectorError::VectorNotFound(id.to_string())),
        }
    }

    fn read_encoded_vector_from_sidecar(
        &self,
        r: &AnyRead<'_>,
        id: u128,
    ) -> Result<Option<Vec<u8>>, VectorError> {
        let Some(ordinal) = self.sidecar_ordinal_from_store(r, id)? else {
            return Ok(None);
        };
        let slot = self
            .mmap_store
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("mmap lock poisoned: {}", e)))?;
        Ok(slot
            .as_ref()
            .and_then(|store| store.get_encoded_vec(ordinal)))
    }

    fn read_vector_from_sidecar(&self, r: &AnyRead<'_>, id: u128) -> Result<Vec<f32>, VectorError> {
        let ordinal = self.sidecar_ordinal(r, id)?.ok_or_else(|| {
            VectorError::VectorCoreError(format!(
                "externalized vector {} missing ordinal mapping",
                id
            ))
        })?;
        let data = self.sidecar_vec(ordinal)?.ok_or_else(|| {
            VectorError::VectorCoreError(format!(
                "externalized vector {} missing sidecar bytes at ordinal {}",
                id, ordinal
            ))
        })?;
        Ok(data)
    }

    fn read_vector_from_sidecar_durable(
        &self,
        r: &AnyRead<'_>,
        id: u128,
    ) -> Result<Vec<f32>, VectorError> {
        let ordinal = self.sidecar_ordinal_from_store(r, id)?.ok_or_else(|| {
            VectorError::VectorCoreError(format!(
                "externalized vector {} missing ordinal mapping",
                id
            ))
        })?;
        let data = self.sidecar_vec(ordinal)?.ok_or_else(|| {
            VectorError::VectorCoreError(format!(
                "externalized vector {} missing sidecar bytes at ordinal {}",
                id, ordinal
            ))
        })?;
        Ok(data)
    }

    fn has_exact_sidecar_vector(
        &self,
        ordinal: u64,
        expected_dim: Option<usize>,
    ) -> Result<bool, VectorError> {
        let slot = self
            .mmap_store
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("mmap lock poisoned: {}", e)))?;
        let Some(store) = slot.as_ref() else {
            return Ok(false);
        };
        if !store.is_exact() {
            return Ok(false);
        }
        if let Some(dim) = expected_dim {
            if store.dim() != dim {
                return Ok(false);
            }
        }
        Ok(ordinal < store.count())
    }

    fn read_exact_sidecar_vector(
        &self,
        r: &AnyRead<'_>,
        id: u128,
        expected_dim: Option<usize>,
    ) -> Result<Option<Vec<f32>>, VectorError> {
        let Some(ordinal) = self.sidecar_ordinal(r, id)? else {
            return Ok(None);
        };
        if !self.has_exact_sidecar_vector(ordinal, expected_dim)? {
            return Ok(None);
        }
        match self.read_vector_from_sidecar(r, id) {
            Ok(data) => Ok(Some(data)),
            Err(VectorError::VectorCoreError(message))
                if message.contains("missing sidecar bytes") =>
            {
                Ok(None)
            }
            Err(err) => Err(err),
        }
    }

    fn read_sidecar_vector_if_available(
        &self,
        r: &AnyRead<'_>,
        id: u128,
        expected_dim: Option<usize>,
    ) -> Result<Option<Vec<f32>>, VectorError> {
        if self.sidecar_ordinal(r, id)?.is_none() {
            return Ok(None);
        }
        match self.read_vector_from_sidecar(r, id) {
            Ok(data) if expected_dim.is_none_or(|dim| data.len() == dim) => Ok(Some(data)),
            Ok(_) => Ok(None),
            Err(VectorError::VectorCoreError(message))
                if Self::is_externalized_vector_unavailable(&message) =>
            {
                Ok(None)
            }
            Err(VectorError::VectorNotFound(_)) => Ok(None),
            Err(err) => Err(err),
        }
    }

    fn is_externalized_vector_unavailable(message: &str) -> bool {
        message.contains("missing mmap store")
            || message.contains("missing sidecar bytes")
            || message.contains("missing ordinal mapping")
    }

    pub fn externalized_marker_repair_needed(&self) -> bool {
        self.externalized_marker_repair_needed
            .load(AtomicOrdering::Acquire)
    }

    pub fn mmap_store_count(&self) -> Option<u64> {
        self.mmap_store
            .read()
            .ok()
            .and_then(|slot| slot.as_ref().map(|store| store.count()))
    }

    pub(crate) fn mark_externalized_marker_repair_needed(&self) {
        self.externalized_marker_repair_needed
            .store(true, AtomicOrdering::Release);
    }

    pub(crate) fn clear_externalized_marker_repair_needed(&self) {
        self.externalized_marker_repair_needed
            .store(false, AtomicOrdering::Release);
    }

    pub fn unavailable_externalized_marker_ids(
        &self,
        r: &AnyRead<'_>,
        limit: usize,
    ) -> Result<Vec<u128>, VectorError> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let mut ids = Vec::new();
        let mut seen = HashSet::new();
        // Collect marker (id, level) candidates first; the resolve loop below
        // re-reads the DB through the seam, so it cannot run inside the scan
        // closure (which already borrows `self.backend`).
        let mut markers: Vec<(u128, usize)> = Vec::new();
        let mut scan_err: Option<VectorError> = None;
        self.backend
            .scan(
                r,
                self.seg_ns(SegmentDb::Vectors),
                KeyRange::prefix(VECTOR_PREFIX),
                |key, value| {
                    if !Self::is_vector_marker(value) {
                        return true;
                    }
                    if key.len() < VECTOR_PREFIX.len() + 16 + std::mem::size_of::<usize>() {
                        scan_err = Some(VectorError::InvalidVectorData);
                        return false;
                    }
                    let id_start = VECTOR_PREFIX.len();
                    let id_end = id_start + 16;
                    let level_end = id_end + std::mem::size_of::<usize>();
                    let Ok(id_arr) = key[id_start..id_end].try_into() else {
                        scan_err = Some(VectorError::InvalidVectorData);
                        return false;
                    };
                    let Ok(raw_level) = key[id_end..level_end].try_into() else {
                        scan_err = Some(VectorError::InvalidVectorData);
                        return false;
                    };
                    markers.push((u128::from_be_bytes(id_arr), usize::from_be_bytes(raw_level)));
                    true
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        if let Some(err) = scan_err {
            return Err(err);
        }
        for (id, vector_level) in markers {
            match self.resolve_marker_vector(r, id, vector_level) {
                Ok(_) => {}
                Err(VectorError::VectorCoreError(message))
                    if Self::is_externalized_vector_unavailable(&message) =>
                {
                    if seen.insert(id) {
                        ids.push(id);
                    }
                    if ids.len() >= limit {
                        break;
                    }
                }
                Err(VectorError::VectorNotFound(_)) => {
                    if seen.insert(id) {
                        ids.push(id);
                    }
                    if ids.len() >= limit {
                        break;
                    }
                }
                Err(err) => return Err(err),
            }
        }

        Ok(ids)
    }

    pub fn purge_unavailable_externalized_markers(
        &self,
        txn: &mut RwTxn,
        limit: usize,
    ) -> Result<usize, VectorError> {
        let ids = {
            let r = self.backend.read_borrowed(&*txn);
            self.unavailable_externalized_marker_ids(&r, limit)?
        };
        self.purge_externalized_marker_ids(txn, &ids, limit)
    }

    pub fn purge_externalized_marker_ids(
        &self,
        txn: &mut RwTxn,
        ids: &[u128],
        scan_limit: usize,
    ) -> Result<usize, VectorError> {
        if ids.is_empty() {
            self.clear_externalized_marker_repair_needed();
            return Ok(0);
        }

        let purge_ids = {
            let r = self.backend.read_borrowed(&*txn);
            self.still_unavailable_externalized_marker_ids(&r, ids)?
        };
        if purge_ids.is_empty() {
            if ids.len() < scan_limit {
                self.clear_externalized_marker_repair_needed();
            } else {
                self.mark_externalized_marker_repair_needed();
            }
            return Ok(0);
        }

        self.delete_vectors_batch(txn, &purge_ids)?;
        metrics::counter!("helix_externalized_marker_repaired_total")
            .increment(purge_ids.len() as u64);

        if ids.len() < scan_limit {
            self.clear_externalized_marker_repair_needed();
        } else {
            self.mark_externalized_marker_repair_needed();
        }

        Ok(purge_ids.len())
    }

    fn still_unavailable_externalized_marker_ids(
        &self,
        r: &AnyRead<'_>,
        ids: &[u128],
    ) -> Result<Vec<u128>, VectorError> {
        let mut purge_ids = Vec::new();
        for &id in ids {
            let key_prefix = [VECTOR_PREFIX, &id.to_be_bytes()].concat();
            // Collect marker levels first; `resolve_marker_vector` re-reads the
            // DB through the seam and cannot run inside the scan closure.
            let mut levels: Vec<usize> = Vec::new();
            let mut scan_err: Option<VectorError> = None;
            self.backend
                .scan(
                    r,
                    self.seg_ns(SegmentDb::Vectors),
                    KeyRange::prefix(&key_prefix),
                    |key, value| {
                        if !Self::is_vector_marker(value) {
                            return true;
                        }
                        if key.len() < VECTOR_PREFIX.len() + 16 + std::mem::size_of::<usize>() {
                            scan_err = Some(VectorError::InvalidVectorData);
                            return false;
                        }
                        let id_end = VECTOR_PREFIX.len() + 16;
                        let level_end = id_end + std::mem::size_of::<usize>();
                        let Ok(raw_level) = key[id_end..level_end].try_into() else {
                            scan_err = Some(VectorError::InvalidVectorData);
                            return false;
                        };
                        levels.push(usize::from_be_bytes(raw_level));
                        true
                    },
                )
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            if let Some(err) = scan_err {
                return Err(err);
            }
            let mut unavailable = false;
            for vector_level in levels {
                match self.resolve_marker_vector(r, id, vector_level) {
                    Ok(_) => {}
                    Err(VectorError::VectorCoreError(message))
                        if Self::is_externalized_vector_unavailable(&message) =>
                    {
                        unavailable = true;
                        break;
                    }
                    Err(VectorError::VectorNotFound(_)) => {
                        unavailable = true;
                        break;
                    }
                    Err(err) => return Err(err),
                }
            }
            if unavailable {
                purge_ids.push(id);
            }
        }
        Ok(purge_ids)
    }

    pub fn probe_unavailable_externalized_markers(
        &self,
        r: &AnyRead<'_>,
        limit: usize,
    ) -> Result<usize, VectorError> {
        let unavailable = self.unavailable_externalized_marker_ids(r, limit)?;
        if !unavailable.is_empty() {
            self.mark_externalized_marker_repair_needed();
        }
        Ok(unavailable.len())
    }

    fn resolve_marker_vector(
        &self,
        r: &AnyRead<'_>,
        id: u128,
        level: usize,
    ) -> Result<Vec<f32>, VectorError> {
        let sidecar_err = match self.read_vector_from_sidecar_durable(r, id) {
            Ok(data) => return Ok(data),
            Err(err) => err,
        };
        if level > 0 {
            let level_zero_key = Self::vector_key(id, 0);
            let level_zero_bytes = self
                .backend
                .get_with(
                    r,
                    self.seg_ns(SegmentDb::Vectors),
                    level_zero_key.as_ref(),
                    |opt| opt.map(|b| b.to_vec()),
                )
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            if let Some(level_zero_bytes) = level_zero_bytes {
                if !Self::is_vector_marker(&level_zero_bytes) {
                    return decode_vector(&level_zero_bytes);
                }
            }
        }
        Err(sidecar_err)
    }

    fn distance_between(&self, lhs: &[f32], rhs: &[f32]) -> Result<f32, VectorError> {
        if lhs.len() != rhs.len() {
            return Err(VectorError::InvalidVectorLength);
        }

        use super::simd;
        Ok(match self.distance_metric {
            DistanceMetric::Cosine => simd::cosine_f32(lhs, rhs),
            DistanceMetric::Dot => simd::dot_f32(lhs, rhs),
            // euclid_f32 returns squared L2; take sqrt for consistency with
            // distance_from_approximate which also returns actual L2 distance.
            DistanceMetric::Euclid => simd::euclid_f32(lhs, rhs).sqrt(),
        })
    }

    #[inline]
    fn mmap_distance_metric(&self) -> super::mmap_vectors::MmapDistanceMetric {
        match self.distance_metric {
            DistanceMetric::Cosine => super::mmap_vectors::MmapDistanceMetric::Cosine,
            DistanceMetric::Dot => super::mmap_vectors::MmapDistanceMetric::Dot,
            DistanceMetric::Euclid => super::mmap_vectors::MmapDistanceMetric::Euclid,
        }
    }

    #[inline]
    fn ordinal_from_bytes(bytes: &[u8]) -> Result<u64, VectorError> {
        if bytes.len() != 8 {
            return Err(VectorError::InvalidVectorData);
        }
        let mut ordinal = [0; 8];
        ordinal.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(ordinal))
    }

    fn cached_sidecar_ordinal(&self, id: u128) -> Result<Option<u64>, VectorError> {
        let ordinals = self.mmap_ordinals.read().map_err(|e| {
            VectorError::VectorCoreError(format!("mmap ordinal lock poisoned: {e}"))
        })?;
        Ok(ordinals
            .as_ref()
            .and_then(|cache| cache.entries().get(&id).copied()))
    }

    fn cache_sidecar_ordinal(&self, id: u128, ordinal: u64) {
        if let Ok(mut ordinals) = self.mmap_ordinals.write() {
            ordinals
                .get_or_insert_with(|| SidecarOrdinals::Partial(HashMap::new()))
                .entries_mut()
                .insert(id, ordinal);
        }
    }

    fn sidecar_ordinals_loaded(&self) -> Result<bool, VectorError> {
        let ordinals = self.mmap_ordinals.read().map_err(|e| {
            VectorError::VectorCoreError(format!("mmap ordinal lock poisoned: {e}"))
        })?;
        Ok(ordinals.as_ref().is_some_and(SidecarOrdinals::is_complete))
    }

    fn has_mmap_sidecar(&self) -> bool {
        self.mmap_store
            .read()
            .ok()
            .and_then(|slot| slot.as_ref().map(|_| ()))
            .is_some()
    }

    fn should_try_lsm_sidecar(&self) -> bool {
        self.backend.kind() == BackendKind::Lsm && self.mmap_sidecar_stem.is_some()
    }

    fn ensure_lsm_mmap_sidecar(&self) -> Result<bool, VectorError> {
        self.ensure_lsm_mmap_sidecar_with_lock_observer(|| {})
    }

    fn ensure_lsm_mmap_sidecar_with_lock_observer<F>(
        &self,
        on_lock_attempt: F,
    ) -> Result<bool, VectorError>
    where
        F: FnOnce(),
    {
        if self.has_mmap_sidecar() {
            return Ok(true);
        }
        if !self.should_try_lsm_sidecar() {
            return Ok(false);
        }
        if self.lsm_sidecar_miss_is_cached() {
            metrics::counter!("helix_lsm_sidecar_materialize_total", "outcome" => "cached_missing")
                .increment(1);
            return Ok(false);
        }
        let Some(stem) = self.mmap_sidecar_stem.as_ref().cloned() else {
            return Ok(false);
        };
        on_lock_attempt();
        let mut slot = self
            .mmap_store
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("mmap lock poisoned: {}", e)))?;
        if slot.is_some() {
            return Ok(true);
        }
        if self.lsm_sidecar_miss_is_cached() {
            metrics::counter!("helix_lsm_sidecar_materialize_total", "outcome" => "cached_missing")
                .increment(1);
            return Ok(false);
        }
        // Only the request that won this core's write lock and still needs a
        // scan consumes a process-wide permit. Duplicate requests wait here
        // without starving unrelated segments of materialization capacity.
        let Some(_permit) = SidecarMaterializePermit::try_acquire() else {
            metrics::counter!("helix_lsm_sidecar_materialize_total", "outcome" => "deferred")
                .increment(1);
            return Ok(false);
        };

        let started = std::time::Instant::now();
        let (store, ordinals) = Self::open_lsm_mmap_sidecar(
            &self.backend,
            &self.physical_name,
            &stem,
            &self.spindle,
            true,
        )?;
        let loaded = store.is_some();
        if let Some(store) = store {
            *slot = Some(store);
            self.clear_lsm_sidecar_miss_cache();
        } else {
            self.record_lsm_sidecar_miss();
        }
        drop(slot);
        if let Some(ordinals) = ordinals {
            let mut slot = self.mmap_ordinals.write().map_err(|e| {
                VectorError::VectorCoreError(format!("mmap ordinal lock poisoned: {}", e))
            })?;
            *slot = Some(SidecarOrdinals::Complete(ordinals));
        }
        metrics::histogram!(
            "helix_lsm_sidecar_materialize_ms",
            "outcome" => if loaded { "loaded" } else { "missing" }
        )
        .record(started.elapsed().as_secs_f64() * 1000.0);
        if loaded {
            metrics::counter!("helix_lsm_sidecar_materialize_total", "outcome" => "loaded")
                .increment(1);
        } else {
            metrics::counter!("helix_lsm_sidecar_materialize_total", "outcome" => "missing")
                .increment(1);
        }
        Ok(loaded)
    }

    fn ensure_sidecar_ordinals_loaded(&self, read: &AnyRead<'_>) -> Result<(), VectorError> {
        if matches!(read, AnyRead::LsmReader(_))
            || self.sidecar_ordinals_loaded()?
            || !self.has_mmap_sidecar()
        {
            return Ok(());
        }

        let started = std::time::Instant::now();
        let mut loaded = HashMap::new();
        let mut scan_err: Option<VectorError> = None;
        self.backend
            .scan(
                read,
                self.seg_ns(SegmentDb::Ordinals),
                KeyRange::all(),
                |key, value| {
                    if key.len() != std::mem::size_of::<u128>() {
                        scan_err = Some(VectorError::InvalidVectorData);
                        return false;
                    }
                    let Ok(id_bytes) = <[u8; 16]>::try_from(key) else {
                        scan_err = Some(VectorError::InvalidVectorData);
                        return false;
                    };
                    let ordinal = match Self::ordinal_from_bytes(value) {
                        Ok(ordinal) => ordinal,
                        Err(err) => {
                            scan_err = Some(err);
                            return false;
                        }
                    };
                    loaded.insert(u128::from_be_bytes(id_bytes), ordinal);
                    true
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        if let Some(err) = scan_err {
            return Err(err);
        }

        let loaded_count = loaded.len();
        let outcome = if loaded_count == 0 { "empty" } else { "loaded" };
        let mut ordinals = self.mmap_ordinals.write().map_err(|e| {
            VectorError::VectorCoreError(format!("mmap ordinal lock poisoned: {e}"))
        })?;
        if !ordinals.as_ref().is_some_and(SidecarOrdinals::is_complete) {
            *ordinals = Some(SidecarOrdinals::Complete(loaded));
            metrics::histogram!("helix_lsm_sidecar_ordinals_load_ms", "outcome" => outcome)
                .record(started.elapsed().as_secs_f64() * 1000.0);
            metrics::histogram!("helix_lsm_sidecar_ordinals_items", "outcome" => outcome)
                .record(loaded_count as f64);
            metrics::counter!("helix_lsm_sidecar_ordinals_load_total", "outcome" => outcome)
                .increment(1);
        }
        Ok(())
    }

    fn refresh_mmap_sidecar_reads(&self) -> Result<(), VectorError> {
        let mut slot = self
            .mmap_store
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("mmap lock poisoned: {}", e)))?;
        if let Some(store) = slot.as_mut() {
            store.refresh_read_mmap()?;
        }
        Ok(())
    }

    fn sidecar_vec(&self, ordinal: u64) -> Result<Option<Vec<f32>>, VectorError> {
        self.ensure_lsm_mmap_sidecar()?;
        for attempt in 0..2 {
            let found = {
                let slot = self.mmap_store.read().map_err(|e| {
                    VectorError::VectorCoreError(format!("mmap lock poisoned: {}", e))
                })?;
                slot.as_ref().and_then(|store| store.get_vec(ordinal))
            };
            if found.is_some() {
                return Ok(found);
            }
            if attempt == 0 {
                self.refresh_mmap_sidecar_reads()?;
            }
        }
        Ok(None)
    }

    fn sidecar_distance(&self, ordinal: u64, query: &[f32]) -> Result<Option<f32>, VectorError> {
        self.ensure_lsm_mmap_sidecar()?;
        for attempt in 0..2 {
            let found = {
                let slot = self.mmap_store.read().map_err(|e| {
                    VectorError::VectorCoreError(format!("mmap lock poisoned: {}", e))
                })?;
                slot.as_ref().and_then(|store| {
                    store.distance_to(ordinal, query, self.mmap_distance_metric())
                })
            };
            if let Some(distance) = found {
                return distance.map(Some);
            }
            if attempt == 0 {
                self.refresh_mmap_sidecar_reads()?;
            }
        }
        Ok(None)
    }

    fn replace_cached_sidecar_ordinals(&self, point_ids: &[u128]) -> Result<(), VectorError> {
        let map = Self::build_sidecar_ordinal_map(point_ids)?;
        let mut ordinals = self.mmap_ordinals.write().map_err(|e| {
            VectorError::VectorCoreError(format!("mmap ordinal lock poisoned: {e}"))
        })?;
        *ordinals = Some(SidecarOrdinals::Complete(map));
        Ok(())
    }

    fn remove_cached_sidecar_ordinals(&self, point_ids: &[u128]) {
        if let Ok(mut ordinals) = self.mmap_ordinals.write() {
            if let Some(map) = ordinals.as_mut() {
                for id in point_ids {
                    map.entries_mut().remove(id);
                }
            }
        }
    }

    fn sidecar_ordinal(&self, read: &AnyRead<'_>, id: u128) -> Result<Option<u64>, VectorError> {
        if matches!(read, AnyRead::LsmReader(_)) {
            return self.sidecar_ordinal_from_store(read, id);
        }
        let needs_initial_load = {
            let ordinals = self.mmap_ordinals.read().map_err(|e| {
                VectorError::VectorCoreError(format!("mmap ordinal lock poisoned: {e}"))
            })?;
            match ordinals.as_ref() {
                Some(SidecarOrdinals::Complete(entries)) => return Ok(entries.get(&id).copied()),
                Some(SidecarOrdinals::Partial(entries)) => {
                    if let Some(ordinal) = entries.get(&id) {
                        return Ok(Some(*ordinal));
                    }
                    false
                }
                None => true,
            }
        };
        if needs_initial_load {
            self.ensure_sidecar_ordinals_loaded(read)?;
            let ordinals = self.mmap_ordinals.read().map_err(|e| {
                VectorError::VectorCoreError(format!("mmap ordinal lock poisoned: {e}"))
            })?;
            if let Some(SidecarOrdinals::Complete(entries)) = ordinals.as_ref() {
                return Ok(entries.get(&id).copied());
            }
        }
        self.sidecar_ordinal_from_store(read, id)
    }

    fn sidecar_ordinal_with_request_cache(
        &self,
        read_context: &AnyRead<'_>,
        id: u128,
        cache: &mut SidecarOrdinalRequestCache,
    ) -> Result<Option<u64>, VectorError> {
        if !read_context.is_snapshot_pinned() {
            return self.sidecar_ordinal(read_context, id);
        }
        if let Some(ordinal) = cache.get(id) {
            return Ok(ordinal);
        }
        let ordinal = self.sidecar_ordinal(read_context, id)?;
        cache.insert(id, ordinal);
        Ok(ordinal)
    }

    fn prefetch_sidecar_ordinals(
        &self,
        read_context: &AnyRead<'_>,
        ids: &[u128],
        visited: &HashSet<u128>,
        cache: &mut SidecarOrdinalRequestCache,
    ) -> Result<(), VectorError> {
        if cache.is_full()
            || !read_context.is_snapshot_pinned()
            || !(self.has_mmap_sidecar() || self.should_try_lsm_sidecar())
            || !self.ensure_lsm_mmap_sidecar()?
        {
            return Ok(());
        }

        let mut keys = Vec::new();
        let mut key_ids = Vec::new();
        let mut unique = HashSet::new();
        for &id in ids {
            if visited.contains(&id) || cache.get(id).is_some() || !unique.insert(id) {
                continue;
            }
            keys.push(id.to_be_bytes().to_vec());
            key_ids.push(id);
            if key_ids.len() >= REQUEST_SIDECAR_ORDINAL_CACHE_MAX {
                break;
            }
        }
        if keys.is_empty() {
            return Ok(());
        }

        let rows = match (&*self.backend, read_context) {
            (AnyBackend::Lsm(writer), AnyRead::Lsm(snapshot)) => {
                writer.collect_values_many_with(snapshot, self.seg_ns(SegmentDb::Ordinals), &keys)
            }
            (AnyBackend::LsmReader(reader), AnyRead::LsmReader(snapshot)) => reader
                .collect_values_many_with_at(
                    snapshot.as_ref(),
                    self.seg_ns(SegmentDb::Ordinals),
                    &keys,
                ),
            _ => return Ok(()),
        }
        .map_err(|error| VectorError::VectorCoreError(error.to_string()))?;
        metrics::counter!("helix_hnsw_ordinal_prefetch_batches_total").increment(1);
        metrics::counter!("helix_hnsw_ordinal_prefetch_rows_total").increment(keys.len() as u64);

        for (id, row) in key_ids.into_iter().zip(rows) {
            let ordinal = match row {
                Some(bytes) => Some(Self::ordinal_from_bytes(&bytes)?),
                None => None,
            };
            cache.insert(id, ordinal);
        }
        Ok(())
    }

    fn sidecar_ordinal_from_store(
        &self,
        read: &AnyRead<'_>,
        id: u128,
    ) -> Result<Option<u64>, VectorError> {
        let Some(ordinal_bytes) = self
            .backend
            .get_with(
                read,
                self.seg_ns(SegmentDb::Ordinals),
                &id.to_be_bytes(),
                |opt| opt.map(|bytes| bytes.to_vec()),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?
        else {
            return Ok(None);
        };
        let ordinal = Self::ordinal_from_bytes(&ordinal_bytes)?;
        if !matches!(read, AnyRead::LsmReader(_)) {
            self.cache_sidecar_ordinal(id, ordinal);
        }
        Ok(Some(ordinal))
    }

    pub fn public_score(&self, distance: f32) -> f32 {
        match self.distance_metric {
            DistanceMetric::Cosine => 1.0 - distance,
            DistanceMetric::Dot => -distance,
            DistanceMetric::Euclid => -distance,
        }
    }

    /// Score directly from encoded bytes when a prepared query is available,
    /// falling back to decoded-vector cosine otherwise. Avoids redundant norm
    /// computation and gives the asymmetric scoring path for TurboProd.
    #[inline]
    fn scored_distance(
        &self,
        r: &AnyRead<'_>,
        prepared: &PreparedSpindleQuery,
        neighbor_id: u128,
        neighbor_level: usize,
        neighbor_data: &[f32],
        query_data: &[f32],
    ) -> Result<f32, VectorError> {
        if !matches!(prepared, PreparedSpindleQuery::None) {
            if let Ok(bytes) = self.get_encoded_vector(r, neighbor_id, neighbor_level) {
                if let Ok(Some(approx)) = score_encoded(prepared, &bytes) {
                    return Ok(self.distance_from_approximate(&approx));
                }
            }
        }
        self.distance_between(neighbor_data, query_data)
    }

    #[inline]
    fn score_sidecar_approx_distance_with_cache(
        &self,
        read_context: &AnyRead<'_>,
        prepared: &PreparedSpindleQuery,
        neighbor_id: u128,
        mut ordinal_cache: Option<&mut SidecarOrdinalRequestCache>,
    ) -> Result<Option<f32>, VectorError> {
        if matches!(prepared, PreparedSpindleQuery::None) {
            return Ok(None);
        }
        let slot = self
            .mmap_store
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("mmap lock poisoned: {}", e)))?;
        let Some(store) = slot.as_ref() else {
            return Ok(None);
        };
        let ordinal = if let Some(cache) = ordinal_cache.as_deref_mut() {
            self.sidecar_ordinal_with_request_cache(read_context, neighbor_id, cache)?
        } else {
            self.sidecar_ordinal(read_context, neighbor_id)?
        };
        let Some(ordinal) = ordinal else {
            return Ok(None);
        };
        match store.score_encoded_to(ordinal, prepared) {
            Some(Ok(approx)) => Ok(Some(self.distance_from_approximate(&approx))),
            Some(Err(err)) => Err(err),
            None => Ok(None),
        }
    }

    /// Score a single neighbor for the HNSW search frontier without
    /// materializing its vector data into a `Vec<f32>`.
    ///
    /// Path selection (in order of preference):
    ///   1. **Spindle approx** — if `prepared` is non-`None`, score via
    ///      encoded codes; no full-vector load needed regardless.
    ///   2. **Mmap direct scoring** — with a sidecar ordinal, run distance
    ///      against the mmap row without allocating: HVEC borrows the `&[f32]`
    ///      slice and HVS8 streams quantized bytes through min/scale. Sidecar
    ///      ordinals are keyed by point id, so the same level-0 vector row can
    ///      score HNSW neighbors from any level.
    ///   3. **Fallback** — full `get_vector(with_data=true)` then
    ///      `distance_between`. This is what the search loop did before; kept
    ///      for level > 0, no-mmap paths, or transient remap races.
    ///
    /// Returns:
    ///   - `Ok(Some(distance))` on success.
    ///   - `Ok(None)` when the neighbor's vector row is missing
    ///     (`VectorNotFound`) — the search loop should treat that as
    ///     "skip this candidate" rather than fail the whole query.
    ///   - `Err(_)` for any other error.
    #[inline]
    fn score_neighbor_distance(
        &self,
        r: &AnyRead<'_>,
        prepared: &PreparedSpindleQuery,
        neighbor_id: u128,
        level: usize,
        query_data: &[f32],
    ) -> Result<Option<f32>, VectorError> {
        self.score_neighbor_distance_with_cache(r, prepared, neighbor_id, level, query_data, None)
    }

    #[inline]
    fn score_neighbor_distance_with_cache(
        &self,
        read_context: &AnyRead<'_>,
        prepared: &PreparedSpindleQuery,
        neighbor_id: u128,
        level: usize,
        query_data: &[f32],
        ordinal_cache: Option<&mut SidecarOrdinalRequestCache>,
    ) -> Result<Option<f32>, VectorError> {
        // Deferred-repair deletes (HELIX_LSM_DELETE_TOMBSTONES): a tombstoned
        // id's row is still physically present until this segment's next
        // merge, so every resolution tier below (spindle approx / mmap direct
        // / full fallback) would otherwise still score it. One check up front
        // covers all three — consistent with this function's existing
        // "Ok(None) means skip" contract every caller already implements for
        // `VectorNotFound`.
        if self.is_delete_tombstoned(neighbor_id) {
            return Ok(None);
        }
        self.score_neighbor_distance_for_traversal_with_cache(
            read_context,
            prepared,
            neighbor_id,
            level,
            query_data,
            ordinal_cache,
        )
    }

    /// `score_neighbor_distance` without the delete-tombstone gate. Only for
    /// HNSW frontier expansion: a tombstoned node's row and links are still
    /// present until merge, so it must stay traversable as a bridge (otherwise
    /// each delete punches a hole in the graph) — callers keep it out of the
    /// result set themselves.
    #[inline]
    fn score_neighbor_distance_for_traversal(
        &self,
        r: &AnyRead<'_>,
        prepared: &PreparedSpindleQuery,
        neighbor_id: u128,
        level: usize,
        query_data: &[f32],
    ) -> Result<Option<f32>, VectorError> {
        self.score_neighbor_distance_for_traversal_with_cache(
            r,
            prepared,
            neighbor_id,
            level,
            query_data,
            None,
        )
    }

    #[inline]
    fn score_neighbor_distance_for_traversal_with_cache(
        &self,
        read_context: &AnyRead<'_>,
        prepared: &PreparedSpindleQuery,
        neighbor_id: u128,
        level: usize,
        query_data: &[f32],
        mut ordinal_cache: Option<&mut SidecarOrdinalRequestCache>,
    ) -> Result<Option<f32>, VectorError> {
        // 1. Spindle approx path: no full-vector load needed.
        if !matches!(prepared, PreparedSpindleQuery::None) {
            if let Some(distance) = self.score_sidecar_approx_distance_with_cache(
                read_context,
                prepared,
                neighbor_id,
                ordinal_cache.as_deref_mut(),
            )? {
                return Ok(Some(distance));
            }
            if let Ok(bytes) = self.get_encoded_vector(read_context, neighbor_id, level) {
                if let Ok(Some(approx)) = score_encoded(prepared, &bytes) {
                    return Ok(Some(self.distance_from_approximate(&approx)));
                }
            }
            // approx miss falls through to full-vector path
        }

        // 2. Mmap direct path: ordinal known. This eliminates the per-neighbor
        // owned vector load; for HVS8 it also avoids building a temporary
        // dequantized Vec before distance computation.
        if (self.has_mmap_sidecar() || self.should_try_lsm_sidecar())
            && self.ensure_lsm_mmap_sidecar()?
        {
            let ordinal = if let Some(cache) = ordinal_cache.as_deref_mut() {
                self.sidecar_ordinal_with_request_cache(read_context, neighbor_id, cache)?
            } else {
                self.sidecar_ordinal(read_context, neighbor_id)?
            };
            if let Some(ordinal) = ordinal {
                if let Some(distance) = self.sidecar_distance(ordinal, query_data)? {
                    return Ok(Some(distance));
                }
            }
        }

        let known_ordinal_miss = ordinal_cache
            .as_deref()
            .and_then(|cache| cache.get(neighbor_id))
            == Some(None);

        // 3. Fallback: full vector load + distance compute. Same allocation
        // profile as the pre-mmap-slice code path; preserves correctness for
        // HVS8, level > 0, and any race where the mmap is between remaps.
        let neighbor = match if known_ordinal_miss {
            self.get_vector_from_row(read_context, neighbor_id, level, true, false)
        } else if let Some(cache) = ordinal_cache.as_deref_mut() {
            self.get_vector_with_ordinal_cache(read_context, neighbor_id, level, true, cache)
        } else {
            self.get_vector(read_context, neighbor_id, level, true)
        } {
            Ok(n) => n,
            Err(VectorError::VectorNotFound(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
        let distance = self.scored_distance(
            read_context,
            prepared,
            neighbor.get_id(),
            neighbor.get_level(),
            neighbor.get_data(),
            query_data,
        )?;
        Ok(Some(distance))
    }

    fn prefetch_neighbor_vectors(
        &self,
        read_context: &AnyRead<'_>,
        prepared: &PreparedSpindleQuery,
        neighbors: &[u128],
        visited: &HashSet<u128>,
        level: usize,
    ) -> HashMap<u128, HVector> {
        let mut prefetched = HashMap::new();
        if !read_context.is_snapshot_pinned()
            || !matches!(prepared, PreparedSpindleQuery::None)
            || self.backend.kind() != BackendKind::Lsm
            || self.has_mmap_sidecar()
            || (self.should_try_lsm_sidecar() && !self.lsm_sidecar_miss_is_cached())
        {
            return prefetched;
        }
        let mut missing = Vec::new();
        let mut unique = HashSet::new();
        for &id in neighbors {
            if visited.contains(&id) || !unique.insert(id) {
                continue;
            }
            let cached = if let Some(shared) = self.shared_caches.as_ref() {
                let key = super::shared_cache::CacheKey::new(self.cache_namespace, id, level);
                (level > 0 && shared.nav_vector.get(&key).is_some())
                    || shared.vector.get(&key).is_some()
            } else {
                (level > 0
                    && self
                        .nav_vector_cache
                        .lock()
                        .is_ok_and(|cache| cache.peek(&(id, level)).is_some()))
                    || self
                        .vector_cache
                        .lock()
                        .is_ok_and(|cache| cache.peek(&(id, level)).is_some())
            };
            if !cached {
                missing.push(id);
            }
        }
        for ids in missing.chunks(64) {
            let keys: Vec<Vec<u8>> = ids.iter().map(|&id| Self::vector_key(id, level)).collect();
            let fetched = match (&*self.backend, read_context) {
                (AnyBackend::Lsm(writer), AnyRead::Lsm(read)) => {
                    writer.collect_values_many_with(read, self.seg_ns(SegmentDb::Vectors), &keys)
                }
                (AnyBackend::LsmReader(reader), AnyRead::LsmReader(snapshot)) => reader
                    .collect_values_many_with_at(
                        snapshot.as_ref(),
                        self.seg_ns(SegmentDb::Vectors),
                        &keys,
                    ),
                _ => return prefetched,
            };
            metrics::counter!("helix_hnsw_vector_prefetch_batches_total").increment(1);
            metrics::counter!("helix_hnsw_vector_prefetch_rows_total").increment(ids.len() as u64);
            let Ok(rows) = fetched else {
                metrics::counter!("helix_hnsw_vector_prefetch_fallback_total").increment(1);
                continue;
            };
            for (&id, row) in ids.iter().zip(rows) {
                let Some(bytes) = row else {
                    continue;
                };
                if Self::is_vector_marker(&bytes) {
                    continue;
                }
                if let Ok(decoded) = decode_vector(&bytes) {
                    prefetched.insert(id, HVector::from_slice(id, level, decoded));
                }
            }
        }
        prefetched
    }

    fn get_vector_from_row(
        &self,
        read_context: &AnyRead<'_>,
        id: u128,
        level: usize,
        with_data: bool,
        resolve_marker_sidecar: bool,
    ) -> Result<HVector, VectorError> {
        let key = Self::vector_key(id, level);
        let stored = self
            .backend
            .get_with(
                read_context,
                self.seg_ns(SegmentDb::Vectors),
                key.as_ref(),
                |opt| opt.map(|bytes| bytes.to_vec()),
            )
            .map_err(|error| VectorError::VectorCoreError(error.to_string()))?;
        match stored {
            Some(bytes) => {
                let vector = if with_data {
                    let decoded = if Self::is_vector_marker(&bytes) {
                        if !resolve_marker_sidecar {
                            self.resolve_marker_vector_after_known_ordinal_miss(
                                read_context,
                                id,
                                level,
                            )?
                        } else {
                            self.resolve_marker_vector(read_context, id, level)?
                        }
                    } else {
                        decode_vector(&bytes)?
                    };
                    self.cache_vector_for_read(read_context, id, level, &decoded);
                    HVector::from_slice(id, level, decoded)
                } else {
                    HVector::from_slice(id, level, Vec::new())
                };
                Ok(vector)
            }
            None if level > 0 => {
                self.get_vector_from_row(read_context, id, 0, with_data, resolve_marker_sidecar)
            }
            None => Err(VectorError::VectorNotFound(id.to_string())),
        }
    }

    fn resolve_marker_vector_after_known_ordinal_miss(
        &self,
        read_context: &AnyRead<'_>,
        id: u128,
        level: usize,
    ) -> Result<Vec<f32>, VectorError> {
        if level > 0 {
            let level_zero_key = Self::vector_key(id, 0);
            let level_zero_bytes = self
                .backend
                .get_with(
                    read_context,
                    self.seg_ns(SegmentDb::Vectors),
                    level_zero_key.as_ref(),
                    |opt| opt.map(|bytes| bytes.to_vec()),
                )
                .map_err(|error| VectorError::VectorCoreError(error.to_string()))?;
            if let Some(level_zero_bytes) = level_zero_bytes {
                if !Self::is_vector_marker(&level_zero_bytes) {
                    return decode_vector(&level_zero_bytes);
                }
            }
        }
        Err(VectorError::VectorCoreError(format!(
            "externalized vector {} missing ordinal mapping",
            id
        )))
    }

    fn get_vector_with_ordinal_cache(
        &self,
        read_context: &AnyRead<'_>,
        id: u128,
        level: usize,
        with_data: bool,
        ordinal_cache: &mut SidecarOrdinalRequestCache,
    ) -> Result<HVector, VectorError> {
        let mut known_ordinal = None;
        if with_data
            && (self.has_mmap_sidecar() || self.should_try_lsm_sidecar())
            && self.ensure_lsm_mmap_sidecar()?
        {
            known_ordinal =
                Some(self.sidecar_ordinal_with_request_cache(read_context, id, ordinal_cache)?);
            if let Some(ordinal) = known_ordinal.flatten() {
                if let Some(vector_data) = self.sidecar_vec(ordinal)? {
                    self.cache_vector_for_read(read_context, id, level, &vector_data);
                    return Ok(HVector::from_slice(id, level, vector_data));
                }
            }
        }
        if known_ordinal.is_some() {
            return self.get_vector_from_row(
                read_context,
                id,
                level,
                with_data,
                known_ordinal != Some(None),
            );
        }
        self.get_vector(read_context, id, level, with_data)
    }

    /// Compute distance from an ApproximateInnerProduct. Internal math stays f64, cast at boundary.
    fn distance_from_approximate(&self, approx: &ApproximateInnerProduct) -> f32 {
        let unit_dot = approx.unit_dot.clamp(-1.0, 1.0);
        let result = match self.distance_metric {
            DistanceMetric::Cosine => 1.0 - unit_dot,
            DistanceMetric::Dot => -(unit_dot * approx.query_norm * approx.doc_norm),
            DistanceMetric::Euclid => {
                let squared = approx.query_norm * approx.query_norm
                    + approx.doc_norm * approx.doc_norm
                    - 2.0 * unit_dot * approx.query_norm * approx.doc_norm;
                squared.max(0.0).sqrt()
            }
        };
        result as f32
    }

    fn rescore_results(
        &self,
        r: &AnyRead<'_>,
        raw_query: &[f32],
        mut results: Vec<HVector>,
    ) -> Result<Vec<HVector>, VectorError> {
        if self.spindle.keep_original {
            for result in &mut results {
                if let Some(original) = self.get_original_vector(r, result.get_id())? {
                    let distance = self.distance_between(raw_query, &original)?;
                    result.replace_data(original);
                    result.set_distance(distance);
                }
            }
        } else {
            let prepared = prepare_query(raw_query, &self.spindle)?;
            for result in &mut results {
                let bytes = self.get_encoded_vector(r, result.get_id(), result.get_level())?;
                if let Some(approx) = score_encoded(&prepared, &bytes)? {
                    result.set_distance(self.distance_from_approximate(&approx));
                }
            }
        }

        results.sort_by(|lhs, rhs| {
            lhs.get_distance()
                .partial_cmp(&rhs.get_distance())
                .unwrap_or(Ordering::Equal)
        });
        Ok(results)
    }

    fn search_flat<F>(
        &self,
        r: &AnyRead<'_>,
        raw_query: &[f32],
        k: usize,
        filter: Option<&[F]>,
    ) -> Result<Vec<HVector>, VectorError>
    where
        F: Fn(&HVector) -> bool,
    {
        let projected_query = project_for_search(raw_query, &self.spindle)?;
        let query = HVector::from_slice(0, 0, projected_query);
        let prepared = prepare_query(raw_query, &self.spindle)?;
        let target_k = self.rerank_target_k(k, None);
        let mut results = Vec::new();

        if filter.is_none() {
            results =
                self.search_flat_unfiltered_direct(r, query.get_data(), &prepared, target_k)?;
            if self.spindle.is_enabled() && self.spindle.rescore {
                results = self.rescore_results(r, raw_query, results)?;
            }
            results.truncate(k);
            return Ok(results);
        }

        for mut candidate in self.get_all_vectors(r, Some(0))? {
            // Deferred-repair deletes: this payload-filtered flat/mutable-tail
            // scan reads raw rows directly (no `score_neighbor_distance` /
            // `get_vector` in between), so it needs its own tombstone check.
            if self.is_delete_tombstoned(candidate.get_id()) {
                continue;
            }
            let distance = self.scored_distance(
                r,
                &prepared,
                candidate.get_id(),
                candidate.get_level(),
                candidate.get_data(),
                query.get_data(),
            )?;
            candidate.set_distance(distance);
            if filter.is_none() || filter.unwrap().iter().all(|f| f(&candidate)) {
                results.push(candidate);
            }
        }

        results.sort_by(|lhs, rhs| {
            lhs.get_distance()
                .partial_cmp(&rhs.get_distance())
                .unwrap_or(Ordering::Equal)
        });
        results.truncate(target_k);

        if self.spindle.is_enabled() && self.spindle.rescore {
            results = self.rescore_results(r, raw_query, results)?;
        }

        results.truncate(k);
        Ok(results)
    }

    /// IVF posting-list search (SPANN-style) for segments built by
    /// `build_ivf_from_flat_with_permit`: rank every centroid in memory, then
    /// read and score only the posting lists of the `nprobe` nearest
    /// centroids. Pure reads over `Namespace::Segment` KV — no lazy builds or
    /// write-backs — so it serves identically on the writer, the LSM backend,
    /// and read-only `LsmReader` replicas.
    ///
    /// `nprobe` resolution: explicit override, else `HELIX_IVF_NPROBE`, else
    /// the `default_nprobe` persisted at build; always clamped to
    /// `[1, centroid_count]` so tiny segments stay searchable.
    fn search_ivf<F>(
        &self,
        r: &AnyRead<'_>,
        raw_query: &[f32],
        k: usize,
        filter: Option<&[F]>,
        selectivity_hint: Option<f32>,
        nprobe_override: Option<usize>,
    ) -> Result<Vec<HVector>, VectorError>
    where
        F: Fn(&HVector) -> bool,
    {
        if k == 0 {
            return Ok(Vec::new());
        }
        let meta_bytes = self
            .backend
            .get_with(
                r,
                self.seg_ns(SegmentDb::IvfCentroids),
                IVF_META_KEY.as_bytes(),
                |opt| opt.map(|b| b.to_vec()),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        let centroid_bytes = self
            .backend
            .get_with(
                r,
                self.seg_ns(SegmentDb::IvfCentroids),
                IVF_CENTROIDS_KEY.as_bytes(),
                |opt| opt.map(|b| b.to_vec()),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        let (Some(meta_bytes), Some(centroid_bytes)) = (meta_bytes, centroid_bytes) else {
            // Racing a segment clear between `index_mode` and here: the
            // level-0 rows are still authoritative, fall back to exact flat.
            return self.search_flat(r, raw_query, k, filter);
        };
        let meta = super::ivf::decode_meta(&meta_bytes)?;
        let (_dim, centroids) = super::ivf::decode_centroids(&centroid_bytes)?;
        if centroids.is_empty() {
            return self.search_flat(r, raw_query, k, filter);
        }

        let projected_query = project_for_search(raw_query, &self.spindle)?;

        // Rank all centroids in memory (k ≤ 4096, cheap relative to IO).
        let mut order: Vec<(f32, u32)> = Vec::with_capacity(centroids.len());
        for (centroid_id, centroid) in centroids.iter().enumerate() {
            order.push((
                self.distance_between(&projected_query, centroid)?,
                centroid_id as u32,
            ));
        }
        order.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal));
        let nprobe = nprobe_override
            .or_else(ivf_nprobe_override)
            .unwrap_or(meta.default_nprobe as usize)
            .clamp(1, centroids.len());

        // Sequential posting-list reads for the probed centroids.
        let mut candidate_ids: Vec<u128> = Vec::new();
        for (_, centroid_id) in order.iter().take(nprobe) {
            let posting = self
                .backend
                .get_with(
                    r,
                    self.seg_ns(SegmentDb::IvfPostings),
                    &centroid_id.to_be_bytes(),
                    |opt| opt.map(|b| b.to_vec()),
                )
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            if let Some(blob) = posting {
                candidate_ids.extend(super::ivf::decode_posting(&blob)?);
            }
        }

        let Some(filters) = filter else {
            // Unfiltered: same heap/hydrate/rescore tail as exact-id search.
            return self.search_exact_ids_with_selectivity(
                r,
                raw_query,
                candidate_ids,
                k,
                selectivity_hint,
            );
        };

        // Filtered: hydrate each probed candidate, score, then apply the
        // predicate — `search_flat`'s filtered branch restricted to postings.
        let prepared = prepare_query(raw_query, &self.spindle)?;
        let target_k = self.rerank_target_k(k, selectivity_hint);
        let mut results: Vec<HVector> = Vec::new();
        for id in candidate_ids {
            let mut candidate = match self.get_vector(r, id, 0, true) {
                Ok(candidate) => candidate,
                // Stale posting id (vector deleted after the IVF build).
                Err(VectorError::VectorNotFound(_)) => continue,
                Err(VectorError::VectorCoreError(message))
                    if Self::is_externalized_vector_unavailable(&message) =>
                {
                    self.mark_externalized_marker_repair_needed();
                    metrics::counter!(
                        "helix_externalized_marker_unavailable_total",
                        "level" => "0"
                    )
                    .increment(1);
                    tracing::warn!(
                        vector_id = %id,
                        level = 0,
                        error = %message,
                        "skipping unavailable externalized vector marker"
                    );
                    continue;
                }
                Err(err) => return Err(err),
            };
            let distance = self.scored_distance(
                r,
                &prepared,
                candidate.get_id(),
                candidate.get_level(),
                candidate.get_data(),
                &projected_query,
            )?;
            candidate.set_distance(distance);
            if filters.iter().all(|f| f(&candidate)) {
                results.push(candidate);
            }
        }
        results.sort_by(|lhs, rhs| {
            lhs.get_distance()
                .partial_cmp(&rhs.get_distance())
                .unwrap_or(Ordering::Equal)
        });
        results.truncate(target_k);
        if self.spindle.is_enabled() && self.spindle.rescore {
            results = self.rescore_results(r, raw_query, results)?;
        }
        results.truncate(k);
        Ok(results)
    }

    fn search_flat_unfiltered_direct(
        &self,
        r: &AnyRead<'_>,
        projected_query: &[f32],
        prepared: &PreparedSpindleQuery,
        target_k: usize,
    ) -> Result<Vec<HVector>, VectorError> {
        self.search_flat_id_filter_direct::<fn(u128) -> bool>(
            r,
            projected_query,
            prepared,
            target_k,
            None,
        )
    }

    fn search_flat_mmap_ordinals_direct<F>(
        &self,
        r: &AnyRead<'_>,
        projected_query: &[f32],
        target_k: usize,
        filter: Option<&[F]>,
    ) -> Result<Option<Vec<HVector>>, VectorError>
    where
        F: Fn(u128) -> bool,
    {
        let slot = self
            .mmap_store
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("mmap lock poisoned: {}", e)))?;
        let Some(store) = slot.as_ref() else {
            return Ok(None);
        };
        if target_k == 0 {
            return Ok(Some(Vec::new()));
        }
        if store.count() == 0 {
            return Ok(Some(Vec::new()));
        }

        let metric = self.mmap_distance_metric();
        let mut nearest: BinaryHeap<MmapFlatCandidate> = BinaryHeap::with_capacity(target_k);
        let mut scan_err: Option<VectorError> = None;
        // Deferred-repair deletes: tombstoned rows are still physically
        // present. Probe the set once so the common no-delete case pays
        // nothing per row.
        let has_tombstones = self.deleted_count() > 0;
        let mut consider = |id: u128, ordinal: u64| -> Result<(), VectorError> {
            if has_tombstones && self.is_delete_tombstoned(id) {
                return Ok(());
            }
            if let Some(filters) = filter {
                if !filters.iter().all(|f| f(id)) {
                    return Ok(());
                }
            }
            let Some(distance) = store.distance_to(ordinal, projected_query, metric) else {
                return Ok(());
            };
            let candidate = MmapFlatCandidate {
                id,
                ordinal,
                distance: distance?,
            };
            if nearest.len() < target_k {
                nearest.push(candidate);
            } else if nearest
                .peek()
                .is_some_and(|worst| candidate.distance < worst.distance)
            {
                nearest.pop();
                nearest.push(candidate);
            }
            Ok(())
        };

        let used_cached_ordinals = {
            let ordinals = self.mmap_ordinals.read().map_err(|e| {
                VectorError::VectorCoreError(format!("mmap ordinal lock poisoned: {e}"))
            })?;
            match ordinals.as_ref() {
                Some(SidecarOrdinals::Complete(map)) if !matches!(r, AnyRead::LsmReader(_)) => {
                    for (&id, &ordinal) in map {
                        consider(id, ordinal)?;
                    }
                    true
                }
                Some(SidecarOrdinals::Partial(_)) | Some(SidecarOrdinals::Complete(_)) | None => {
                    false
                }
            }
        };
        if !used_cached_ordinals {
            let mut scanned_ordinals = Vec::new();
            self.backend
                .scan(
                    r,
                    self.seg_ns(SegmentDb::Ordinals),
                    KeyRange::all(),
                    |key, value| {
                        if key.len() != std::mem::size_of::<u128>() {
                            scan_err = Some(VectorError::InvalidVectorData);
                            return false;
                        }
                        let Ok(id_bytes) = key.try_into() else {
                            scan_err = Some(VectorError::InvalidVectorData);
                            return false;
                        };
                        let id = u128::from_be_bytes(id_bytes);
                        let ordinal = match Self::ordinal_from_bytes(value) {
                            Ok(ordinal) => ordinal,
                            Err(err) => {
                                scan_err = Some(err);
                                return false;
                            }
                        };
                        scanned_ordinals.push((id, ordinal));
                        true
                    },
                )
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            if let Some(err) = scan_err {
                return Err(err);
            }
            for (id, ordinal) in scanned_ordinals {
                consider(id, ordinal)?;
            }
        }
        if let Some(err) = scan_err {
            return Err(err);
        }

        let mut scored = nearest.into_vec();
        scored.sort_by(|lhs, rhs| {
            lhs.distance
                .partial_cmp(&rhs.distance)
                .unwrap_or(Ordering::Equal)
        });
        let mut results = Vec::with_capacity(scored.len());
        for candidate in scored {
            if let Some(data) = store.get_vec(candidate.ordinal) {
                let mut vector = HVector::from_slice(candidate.id, 0, data);
                vector.set_distance(candidate.distance);
                results.push(vector);
            }
        }
        Ok(Some(results))
    }

    fn search_flat_id_filter_direct<F>(
        &self,
        r: &AnyRead<'_>,
        projected_query: &[f32],
        prepared: &PreparedSpindleQuery,
        target_k: usize,
        filter: Option<&[F]>,
    ) -> Result<Vec<HVector>, VectorError>
    where
        F: Fn(u128) -> bool,
    {
        if target_k == 0 {
            return Ok(Vec::new());
        }
        if let Some(results) =
            self.search_flat_hvtq_fastscan_direct(r, prepared, target_k, filter)?
        {
            return Ok(results);
        }
        if let Some(results) =
            self.search_flat_mmap_ordinals_direct(r, projected_query, target_k, filter)?
        {
            return Ok(results);
        }

        let level_offset = VECTOR_PREFIX.len() + std::mem::size_of::<u128>();
        let level_end = level_offset + std::mem::size_of::<usize>();
        let mut nearest: BinaryHeap<FlatCandidate> = BinaryHeap::with_capacity(target_k);
        let mut ordinal_cache = SidecarOrdinalRequestCache::new();

        // Collect filtered level-0 ids first; the scoring loop re-reads the DB
        // through the seam and cannot run inside the scan closure (which already
        // borrows `self.backend`). The filter has no DB dependency, so it stays
        // inside the scan to preserve the original ordering.
        let mut candidate_ids: Vec<u128> = Vec::new();
        let mut scan_err: Option<VectorError> = None;
        self.backend
            .scan(
                r,
                self.seg_ns(SegmentDb::Vectors),
                KeyRange::prefix(VECTOR_PREFIX),
                |key, _v| {
                    if key.len() < level_end {
                        scan_err = Some(VectorError::InvalidVectorData);
                        return false;
                    }
                    let Ok(id_arr) = key[VECTOR_PREFIX.len()..level_offset].try_into() else {
                        scan_err = Some(VectorError::InvalidVectorData);
                        return false;
                    };
                    let Ok(level_arr) = key[level_offset..level_end].try_into() else {
                        scan_err = Some(VectorError::InvalidVectorData);
                        return false;
                    };
                    let id = u128::from_be_bytes(id_arr);
                    if usize::from_be_bytes(level_arr) != 0 {
                        return true;
                    }
                    if let Some(fs) = filter {
                        if !fs.iter().all(|f| f(id)) {
                            return true;
                        }
                    }
                    candidate_ids.push(id);
                    true
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        if let Some(err) = scan_err {
            return Err(err);
        }

        for id in candidate_ids {
            let level = 0usize;

            let distance = match self.score_neighbor_distance_with_cache(
                r,
                prepared,
                id,
                level,
                projected_query,
                Some(&mut ordinal_cache),
            ) {
                Ok(Some(distance)) => distance,
                Ok(None) | Err(VectorError::VectorNotFound(_)) => continue,
                Err(VectorError::VectorCoreError(message))
                    if Self::is_externalized_vector_unavailable(&message) =>
                {
                    self.mark_externalized_marker_repair_needed();
                    metrics::counter!(
                        "helix_externalized_marker_unavailable_total",
                        "level" => level.to_string()
                    )
                    .increment(1);
                    tracing::warn!(
                        vector_id = %id,
                        level,
                        error = %message,
                        "skipping unavailable externalized vector marker"
                    );
                    continue;
                }
                Err(err) => return Err(err),
            };

            let candidate = FlatCandidate {
                id,
                level,
                distance,
            };
            if nearest.len() < target_k {
                nearest.push(candidate);
            } else if nearest
                .peek()
                .is_some_and(|worst| candidate.distance < worst.distance)
            {
                nearest.pop();
                nearest.push(candidate);
            }
        }

        let mut scored = nearest.into_vec();
        scored.sort_by(|lhs, rhs| {
            lhs.distance
                .partial_cmp(&rhs.distance)
                .unwrap_or(Ordering::Equal)
        });
        let mut results = Vec::with_capacity(scored.len());
        for candidate in scored {
            match self.get_vector_with_ordinal_cache(
                r,
                candidate.id,
                candidate.level,
                true,
                &mut ordinal_cache,
            ) {
                Ok(mut hydrated) => {
                    hydrated.set_distance(candidate.distance);
                    results.push(hydrated);
                }
                Err(VectorError::VectorNotFound(_)) => continue,
                Err(VectorError::VectorCoreError(message))
                    if Self::is_externalized_vector_unavailable(&message) =>
                {
                    self.mark_externalized_marker_repair_needed();
                    metrics::counter!(
                        "helix_externalized_marker_unavailable_total",
                        "level" => candidate.level.to_string()
                    )
                    .increment(1);
                    tracing::warn!(
                        vector_id = %candidate.id,
                        level = candidate.level,
                        error = %message,
                        "skipping unavailable externalized vector marker"
                    );
                    continue;
                }
                Err(err) => return Err(err),
            }
        }

        Ok(results)
    }

    pub fn search_exact_ids_with_selectivity<I>(
        &self,
        r: &AnyRead<'_>,
        raw_query: &[f32],
        candidate_ids: I,
        k: usize,
        selectivity_hint: Option<f32>,
    ) -> Result<Vec<HVector>, VectorError>
    where
        I: IntoIterator<Item = u128>,
    {
        if k == 0 {
            return Ok(Vec::new());
        }

        let projected_query = project_for_search(raw_query, &self.spindle)?;
        let prepared = prepare_query(raw_query, &self.spindle)?;
        let target_k = self.rerank_target_k(k, selectivity_hint);
        let mut nearest: BinaryHeap<FlatCandidate> = BinaryHeap::with_capacity(target_k);
        let mut ordinal_cache = SidecarOrdinalRequestCache::new();

        for id in candidate_ids {
            let distance = match self.score_neighbor_distance_with_cache(
                r,
                &prepared,
                id,
                0,
                &projected_query,
                Some(&mut ordinal_cache),
            ) {
                Ok(Some(distance)) => distance,
                Ok(None) | Err(VectorError::VectorNotFound(_)) => continue,
                Err(VectorError::VectorCoreError(message))
                    if Self::is_externalized_vector_unavailable(&message) =>
                {
                    self.mark_externalized_marker_repair_needed();
                    metrics::counter!(
                        "helix_externalized_marker_unavailable_total",
                        "level" => "0"
                    )
                    .increment(1);
                    tracing::warn!(
                        vector_id = %id,
                        level = 0,
                        error = %message,
                        "skipping unavailable externalized vector marker"
                    );
                    continue;
                }
                Err(err) => return Err(err),
            };

            let candidate = FlatCandidate {
                id,
                level: 0,
                distance,
            };
            if nearest.len() < target_k {
                nearest.push(candidate);
            } else if nearest
                .peek()
                .is_some_and(|worst| candidate.distance < worst.distance)
            {
                nearest.pop();
                nearest.push(candidate);
            }
        }

        let mut scored = nearest.into_vec();
        scored.sort_by(|lhs, rhs| {
            lhs.distance
                .partial_cmp(&rhs.distance)
                .unwrap_or(Ordering::Equal)
        });
        let mut results = Vec::with_capacity(scored.len());
        for candidate in scored {
            match self.get_vector_with_ordinal_cache(r, candidate.id, 0, true, &mut ordinal_cache) {
                Ok(mut hydrated) => {
                    hydrated.set_distance(candidate.distance);
                    results.push(hydrated);
                }
                Err(VectorError::VectorNotFound(_)) => continue,
                Err(VectorError::VectorCoreError(message))
                    if Self::is_externalized_vector_unavailable(&message) =>
                {
                    self.mark_externalized_marker_repair_needed();
                    metrics::counter!(
                        "helix_externalized_marker_unavailable_total",
                        "level" => "0"
                    )
                    .increment(1);
                    tracing::warn!(
                        vector_id = %candidate.id,
                        level = 0,
                        error = %message,
                        "skipping unavailable externalized vector marker"
                    );
                }
                Err(err) => return Err(err),
            }
        }

        if self.spindle.is_enabled() && self.spindle.rescore {
            results = self.rescore_results(r, raw_query, results)?;
        }
        results.truncate(k);
        Ok(results)
    }

    fn search_flat_hvtq_fastscan_direct<F>(
        &self,
        r: &AnyRead<'_>,
        prepared: &PreparedSpindleQuery,
        target_k: usize,
        filter: Option<&[F]>,
    ) -> Result<Option<Vec<HVector>>, VectorError>
    where
        F: Fn(u128) -> bool,
    {
        if !hvtq_fastscan_enabled() {
            return Ok(None);
        }
        if prepared.as_turbo_prod().is_none() {
            return Ok(None);
        }
        let (count, scores) = {
            let slot = self
                .mmap_store
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("mmap lock poisoned: {}", e)))?;
            let Some(store) = slot.as_ref() else {
                return Ok(None);
            };
            let Some(scores) = store.score_all_mse_code_i8_lut_neon(prepared) else {
                return Ok(None);
            };
            (store.count() as usize, scores?)
        };
        if count == 0 {
            return Ok(Some(Vec::new()));
        }

        let mut ordinal_to_id = vec![None; count];
        let has_tombstones = self.deleted_count() > 0;
        self.backend
            .scan(
                r,
                self.seg_ns(SegmentDb::Ordinals),
                KeyRange::all(),
                |id_bytes, ordinal_bytes| {
                    if id_bytes.len() != std::mem::size_of::<u128>()
                        || ordinal_bytes.len() != std::mem::size_of::<u64>()
                    {
                        return true;
                    }
                    let Ok(id_arr) = id_bytes.try_into() else {
                        return true;
                    };
                    let id = u128::from_be_bytes(id_arr);
                    // Deferred-repair deletes: unmapped ordinals are skipped
                    // by both the fast scan and the exact rerank below.
                    if has_tombstones && self.is_delete_tombstoned(id) {
                        return true;
                    }
                    if let Some(fs) = filter {
                        if !fs.iter().all(|f| f(id)) {
                            return true;
                        }
                    }
                    let Ok(ordinal_arr) = ordinal_bytes.try_into() else {
                        return true;
                    };
                    let ordinal = u64::from_le_bytes(ordinal_arr) as usize;
                    if ordinal < count {
                        ordinal_to_id[ordinal] = Some(id);
                    }
                    true
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;

        let candidate_k = target_k
            .saturating_mul(hvtq_fastscan_rerank_multiplier())
            .max(target_k)
            .min(count);
        let mut fast_nearest: BinaryHeap<FlatCandidate> = BinaryHeap::with_capacity(candidate_k);
        for (ordinal, maybe_id) in ordinal_to_id.iter().enumerate() {
            if maybe_id.is_none() {
                continue;
            }
            let distance = self.distance_from_approximate(&scores[ordinal]);
            let candidate = FlatCandidate {
                id: ordinal as u128,
                level: 0,
                distance,
            };
            if fast_nearest.len() < candidate_k {
                fast_nearest.push(candidate);
            } else if fast_nearest
                .peek()
                .is_some_and(|worst| candidate.distance < worst.distance)
            {
                fast_nearest.pop();
                fast_nearest.push(candidate);
            }
        }

        let mut exact_nearest: BinaryHeap<FlatCandidate> = BinaryHeap::with_capacity(target_k);
        {
            let slot = self
                .mmap_store
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("mmap lock poisoned: {}", e)))?;
            let Some(store) = slot.as_ref() else {
                return Ok(None);
            };
            for fast_candidate in fast_nearest.into_vec() {
                let ordinal = fast_candidate.id as usize;
                let Some(id) = ordinal_to_id.get(ordinal).and_then(|value| *value) else {
                    continue;
                };
                let distance = match store.score_encoded_to(ordinal as u64, prepared) {
                    Some(Ok(approx)) => self.distance_from_approximate(&approx),
                    Some(Err(err)) => return Err(err),
                    None => continue,
                };
                let candidate = FlatCandidate {
                    id,
                    level: 0,
                    distance,
                };
                if exact_nearest.len() < target_k {
                    exact_nearest.push(candidate);
                } else if exact_nearest
                    .peek()
                    .is_some_and(|worst| candidate.distance < worst.distance)
                {
                    exact_nearest.pop();
                    exact_nearest.push(candidate);
                }
            }
        }

        let mut scored = exact_nearest.into_vec();
        scored.sort_by(|lhs, rhs| {
            lhs.distance
                .partial_cmp(&rhs.distance)
                .unwrap_or(Ordering::Equal)
        });
        let mut results = Vec::with_capacity(scored.len());
        for candidate in scored {
            match self.get_vector(r, candidate.id, 0, true) {
                Ok(mut hydrated) => {
                    hydrated.set_distance(candidate.distance);
                    results.push(hydrated);
                }
                Err(VectorError::VectorNotFound(_)) => continue,
                Err(VectorError::VectorCoreError(message))
                    if Self::is_externalized_vector_unavailable(&message) =>
                {
                    self.mark_externalized_marker_repair_needed();
                    metrics::counter!(
                        "helix_externalized_marker_unavailable_total",
                        "level" => "0"
                    )
                    .increment(1);
                    tracing::warn!(
                        vector_id = %candidate.id,
                        error = %message,
                        "skipping unavailable externalized vector marker"
                    );
                    continue;
                }
                Err(err) => return Err(err),
            }
        }

        Ok(Some(results))
    }

    /// Build the HNSW graph once from previously stored level-0 vectors.
    /// This is intentionally used only after the collection outgrows the
    /// bounded flat-scan window.
    ///
    /// Unlike the incremental `insert()` path, this method:
    /// 1. Reads all flat-stored vectors into memory in one pass
    /// 2. Assigns HNSW levels to all points upfront
    /// 3. Precomputes compact build-time codes for approximate traversal
    /// 4. Bootstraps the first 256 points serially
    /// 5. Builds the remainder in parallel via rayon
    /// 6. Flushes all edges to LMDB in a single write pass
    pub fn build_index_from_flat(&self, txn: &mut RwTxn) -> Result<(), VectorError> {
        let permit = acquire_build_permit()?;
        self.build_index_from_flat_with_permit(txn, &permit)
    }

    pub(crate) fn build_index_from_flat_with_permit(
        &self,
        txn: &mut RwTxn,
        _permit: &BuildPermit,
    ) -> Result<(), VectorError> {
        let load_start = std::time::Instant::now();
        let vectors = {
            let rd = self.backend.read_borrowed(&*txn);
            if self.has_index(&rd)? {
                metrics::counter!(
                    "helix_vector_core_hnsw_build_skipped_total",
                    "backend" => self.backend_label(),
                    "reason" => "already_indexed"
                )
                .increment(1);
                return Ok(());
            }
            self.get_all_vectors(&rd, Some(0))?
        };
        self.observe_hnsw_build_phase(
            "monolithic",
            "load_flat_vectors",
            vectors.len(),
            load_start.elapsed(),
        );
        if vectors.is_empty() {
            metrics::counter!(
                "helix_vector_core_hnsw_build_skipped_total",
                "backend" => self.backend_label(),
                "reason" => "empty"
            )
            .increment(1);
            return Ok(());
        }

        self.build_index_inner(txn, vectors)
    }

    /// Build an IVF posting-list index (SPANN-style) from previously stored
    /// level-0 vectors, as the opt-in alternative to
    /// `build_index_from_flat_with_permit`. Runs deterministic k-means over
    /// the projected vectors and persists three write-once artifacts under
    /// `Namespace::Segment` KV: the centroid table, one posting blob per
    /// centroid, and — last, as the completion marker `index_mode` probes —
    /// the `ivf:meta` blob. Level-0 rows are untouched, so every non-IVF read
    /// path still sees a correct (flat) segment.
    ///
    /// Deletes after the build leave stale ids in posting lists; `search_ivf`
    /// skips ids whose vector rows are gone.
    pub(crate) fn build_ivf_from_flat_with_permit(
        &self,
        txn: &mut RwTxn,
        _permit: &BuildPermit,
    ) -> Result<(), VectorError> {
        let load_start = std::time::Instant::now();
        let vectors = {
            let rd = self.backend.read_borrowed(&*txn);
            if self.index_mode(&rd)? != IndexMode::Flat {
                metrics::counter!(
                    "helix_vector_core_hnsw_build_skipped_total",
                    "backend" => self.backend_label(),
                    "reason" => "already_indexed"
                )
                .increment(1);
                return Ok(());
            }
            self.get_all_vectors(&rd, Some(0))?
        };
        self.observe_hnsw_build_phase(
            "ivf",
            "load_flat_vectors",
            vectors.len(),
            load_start.elapsed(),
        );
        if vectors.is_empty() {
            metrics::counter!(
                "helix_vector_core_hnsw_build_skipped_total",
                "backend" => self.backend_label(),
                "reason" => "empty"
            )
            .increment(1);
            return Ok(());
        }

        // Hydrate projected vectors (same projection the search path scores
        // against — mirrors the hydrate step of `build_index_inner`).
        let build_start = std::time::Instant::now();
        let n = vectors.len();
        let mut point_ids: Vec<u128> = Vec::with_capacity(n);
        let mut point_data: Vec<Vec<f32>> = Vec::with_capacity(n);
        let phase_start = std::time::Instant::now();
        {
            let rd = self.backend.read_borrowed(&*txn);
            for vector in &vectors {
                let id = vector.get_id();
                let stored = self.get_vector_data(&rd, id)?;
                let raw = self.original_vector_for_rebuild(&rd, vector, &stored)?;
                let projected = project_for_search(&raw, &self.spindle)?;
                point_ids.push(id);
                point_data.push(projected);
            }
        }
        self.observe_hnsw_build_phase("ivf", "hydrate_vectors", n, phase_start.elapsed());

        let dim = point_data[0].len();
        let k = ivf_k_override().unwrap_or_else(|| super::ivf::default_k(n));

        // Same distance closure as the HNSW bulk build (`build_index_inner`).
        let dist_fn = |a: &[f32], b: &[f32]| -> f32 {
            use super::simd;
            match self.distance_metric {
                DistanceMetric::Cosine => simd::cosine_f32(a, b),
                DistanceMetric::Dot => simd::dot_f32(a, b),
                DistanceMetric::Euclid => simd::euclid_f32(a, b).sqrt(),
            }
        };
        let phase_start = std::time::Instant::now();
        let clustered = super::ivf::kmeans(&point_data, k, dist_fn);
        self.observe_hnsw_build_phase("ivf", "kmeans", n, phase_start.elapsed());

        let phase_start = std::time::Instant::now();
        let k = clustered.centroids.len();
        let mut postings: Vec<Vec<u128>> = vec![Vec::new(); k];
        for (id, assignment) in point_ids.iter().zip(clustered.assignments.iter()) {
            postings[*assignment as usize].push(*id);
        }
        for (centroid_id, ids) in postings.iter().enumerate() {
            if ids.is_empty() {
                continue;
            }
            self.backend
                .put_heed(
                    txn,
                    self.seg_ns(SegmentDb::IvfPostings),
                    &(centroid_id as u32).to_be_bytes(),
                    &super::ivf::encode_posting(ids),
                )
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        }
        self.backend
            .put_heed(
                txn,
                self.seg_ns(SegmentDb::IvfCentroids),
                IVF_CENTROIDS_KEY.as_bytes(),
                &super::ivf::encode_centroids(dim, &clustered.centroids)?,
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        // Meta last: its presence is the "index complete" marker.
        self.backend
            .put_heed(
                txn,
                self.seg_ns(SegmentDb::IvfCentroids),
                IVF_META_KEY.as_bytes(),
                &super::ivf::encode_meta(&super::ivf::IvfMeta {
                    k: k as u32,
                    dim: dim as u32,
                    default_nprobe: super::ivf::default_nprobe(k) as u32,
                }),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        self.observe_hnsw_build_phase("ivf", "flush", n, phase_start.elapsed());
        self.observe_hnsw_build_phase("ivf", "total", n, build_start.elapsed());
        Ok(())
    }

    /// Inner build logic, called with build permit held.
    fn build_index_inner(&self, txn: &mut RwTxn, vectors: Vec<HVector>) -> Result<(), VectorError> {
        let build_start = std::time::Instant::now();
        // ── Step 1: Load all vectors into memory ──
        let n = vectors.len();
        let mut point_ids: Vec<u128> = Vec::with_capacity(n);
        let mut point_data: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut original_data: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut point_fields: Vec<HashMap<String, Value>> = Vec::with_capacity(n);

        let phase_start = std::time::Instant::now();
        {
            let rd = self.backend.read_borrowed(&*txn);
            for vector in &vectors {
                let id = vector.get_id();
                let stored = self.get_vector_data(&rd, id)?;
                let raw = self.original_vector_for_rebuild(&rd, vector, &stored)?;
                let projected = project_for_search(&raw, &self.spindle)?;
                point_ids.push(id);
                point_data.push(projected);
                original_data.push(raw);
                point_fields.push(stored.fields);
            }
        }
        self.observe_hnsw_build_phase("monolithic", "hydrate_vectors", n, phase_start.elapsed());

        // ── Step 2: Assign levels upfront ──
        let m = self.config.m;
        let m_max_0 = self.config.m_max_0;
        let m_l = self.config.m_l;
        // Use lower ef for bulk builds: half of configured ef_construct
        // (minimum 32). This is 2-3x faster per build without meaningful
        // recall loss since we're building the full graph at once.
        let ef_construction = (self.config.ef_construct / 2).max(32);

        let mut rng = rand::rng();
        let phase_start = std::time::Instant::now();
        let mut levels: Vec<usize> = Vec::with_capacity(n);
        let mut max_level: usize = 0;
        let mut entry_ord: usize = 0;
        for i in 0..n {
            let r: f64 = rng.random::<f64>();
            let level = assign_hnsw_level(r, m_l);
            levels.push(level);
            if level > max_level {
                max_level = level;
                entry_ord = i;
            }
        }
        self.observe_hnsw_build_phase("monolithic", "assign_levels", n, phase_start.elapsed());

        // ── Step 3: Build in-memory adjacency structure ──
        // adjacency[point_ordinal] = per-level neighbor lists
        // Each point has (max_level_for_that_point + 1) levels
        let phase_start = std::time::Instant::now();
        let adjacency: Vec<StdRwLock<Vec<Vec<u32>>>> = (0..n)
            .map(|i| StdRwLock::new(vec![Vec::new(); levels[i] + 1]))
            .collect();
        self.observe_hnsw_build_phase("monolithic", "allocate_adjacency", n, phase_start.elapsed());

        // Distance function closure (captures self.distance_metric)
        let dist_fn = |a: &[f32], b: &[f32]| -> f32 {
            use super::simd;
            match self.distance_metric {
                DistanceMetric::Cosine => simd::cosine_f32(a, b),
                DistanceMetric::Dot => simd::dot_f32(a, b),
                DistanceMetric::Euclid => simd::euclid_f32(a, b).sqrt(),
            }
        };

        let dist_fn_ord =
            |a_ord: usize, b_ord: usize| -> f32 { dist_fn(&point_data[a_ord], &point_data[b_ord]) };

        // ── Step 4: Serial bootstrap (first 256 points) ──
        let phase_start = std::time::Instant::now();
        let bootstrap_count = n.min(256);
        let mut current_entry_ord = entry_ord;
        let mut current_max_level = levels[entry_ord];

        // Insert entry point first (initialize its adjacency)
        // Then insert remaining bootstrap points
        for i in 0..bootstrap_count {
            if i == entry_ord {
                continue; // entry point is already "inserted" (it's the seed)
            }
            insert_point_ord(
                i,
                current_entry_ord,
                current_max_level,
                &levels,
                &adjacency,
                ef_construction,
                m,
                m_max_0,
                &dist_fn_ord,
            );
            if levels[i] > current_max_level {
                current_max_level = levels[i];
                current_entry_ord = i;
            }
        }
        self.observe_hnsw_build_phase("monolithic", "bootstrap", n, phase_start.elapsed());

        // ── Step 5: Parallel build for remaining points ──
        let phase_start = std::time::Instant::now();
        if bootstrap_count < n {
            // Snapshot the entry state for the parallel phase
            let par_entry_ord = current_entry_ord;
            let par_max_level = current_max_level;

            use rayon::prelude::*;
            (bootstrap_count..n).into_par_iter().for_each(|i| {
                if i == entry_ord {
                    return; // entry point already handled
                }
                insert_point_ord(
                    i,
                    par_entry_ord,
                    par_max_level,
                    &levels,
                    &adjacency,
                    ef_construction,
                    m,
                    m_max_0,
                    &dist_fn_ord,
                );
            });

            // Update entry point if any parallel point got a higher level
            for i in bootstrap_count..n {
                if levels[i] > current_max_level {
                    current_max_level = levels[i];
                    current_entry_ord = i;
                }
            }
        }
        self.observe_hnsw_build_phase("monolithic", "parallel_build", n, phase_start.elapsed());

        // ── Step 6: Single LMDB flush ──
        // Re-write all vector data first, then adjacency. Packed neighbor blocks
        // may read neighbor vectors while encoding their compact sidecars.
        let phase_start = std::time::Instant::now();
        for i in 0..n {
            let id = point_ids[i];
            let level = levels[i];
            let raw = &original_data[i];

            // Write vector at level 0
            self.put_raw_vector(txn, id, 0, raw)?;
            // Write vector at assigned level if > 0
            if level > 0 {
                self.put_raw_vector(txn, id, level, raw)?;
            }

            // Write stored data (fields + original vector for rescoring)
            let fields = std::mem::take(&mut point_fields[i]);
            self.maybe_put_vector_data(txn, id, Some(fields), raw)?;
        }
        self.observe_hnsw_build_phase("monolithic", "flush_vectors", n, phase_start.elapsed());

        // Write packed adjacency directly — no legacy cleanup needed (flat
        // vectors never had out_edges), no cache invalidation (caches are empty).
        let phase_start = std::time::Instant::now();
        for i in 0..n {
            let id = point_ids[i];
            let guard = adjacency[i].read().unwrap();
            for lvl in 0..guard.len() {
                let neighbors: Vec<u128> = guard[lvl]
                    .iter()
                    .filter_map(|&ord| point_ids.get(ord as usize).copied())
                    .filter(|neighbor_id| *neighbor_id != id)
                    .collect();
                let block = NeighborBlock::from_ids(neighbors);
                let key = Self::neighbor_list_key(id, lvl);
                let encoded = Self::encode_neighbor_block(&block);
                self.backend
                    .put_heed(
                        txn,
                        self.seg_ns(SegmentDb::HnswNeighbors),
                        &key,
                        encoded.as_slice(),
                    )
                    .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            }
        }
        self.observe_hnsw_build_phase("monolithic", "flush_neighbors", n, phase_start.elapsed());

        // Set the entry point (highest-level point)
        let phase_start = std::time::Instant::now();
        let entry_id = point_ids[current_entry_ord];
        let entry_vector = HVector::from_slice(entry_id, levels[current_entry_ord], vec![]);
        self.set_entry_point(txn, &entry_vector)?;
        self.observe_hnsw_build_phase("monolithic", "set_entry", n, phase_start.elapsed());
        self.observe_hnsw_build_phase("monolithic", "total", n, build_start.elapsed());

        Ok(())
    }

    /// Build an HNSW graph entirely in memory from pre-exported vectors.
    ///
    /// This is the merge-path companion to `prepare_index_from_flat`:
    /// it takes vectors already exported from source segments (no LMDB
    /// read needed) and returns a `PreparedIndex` for flushing.
    ///
    /// Used by `NamedVectorManager::prepare_merge` for split-phase merge.
    pub fn build_hnsw_in_memory(
        exported: &[(u128, Vec<f32>, HashMap<String, Value>)],
        hnsw_config: &HNSWConfig,
        distance_metric: DistanceMetric,
    ) -> Result<PreparedIndex, VectorError> {
        let permit = acquire_build_permit()?;
        let owned = exported
            .iter()
            .map(|(id, data, fields)| (*id, data.clone(), fields.clone()))
            .collect();
        Self::build_hnsw_in_memory_owned_with_permit(owned, hnsw_config, distance_metric, &permit)
    }

    /// Borrowing compatibility wrapper. Merge callers should prefer
    /// `build_hnsw_in_memory_owned_with_permit` so exported vectors become the
    /// prepared rows instead of being cloned into a second heap copy.
    #[allow(dead_code)]
    pub(crate) fn build_hnsw_in_memory_with_permit(
        exported: &[(u128, Vec<f32>, HashMap<String, Value>)],
        hnsw_config: &HNSWConfig,
        distance_metric: DistanceMetric,
        permit: &BuildPermit,
    ) -> Result<PreparedIndex, VectorError> {
        let owned = exported
            .iter()
            .map(|(id, data, fields)| (*id, data.clone(), fields.clone()))
            .collect();
        Self::build_hnsw_in_memory_owned_with_permit(owned, hnsw_config, distance_metric, permit)
    }

    /// Same as `build_hnsw_in_memory`, but consumes exported rows. This is the
    /// production merge path: the raw vectors live once inside `PreparedIndex`
    /// and are progressively drained as publish chunks complete.
    pub(crate) fn build_hnsw_in_memory_owned_with_permit(
        exported: Vec<(u128, Vec<f32>, HashMap<String, Value>)>,
        hnsw_config: &HNSWConfig,
        distance_metric: DistanceMetric,
        _permit: &BuildPermit,
    ) -> Result<PreparedIndex, VectorError> {
        if exported.is_empty() {
            return Err(VectorError::VectorCoreError(
                "Cannot build HNSW from empty vector set".into(),
            ));
        }

        let n = exported.len();
        if n > u32::MAX as usize {
            return Err(VectorError::VectorCoreError(
                "HNSW prepared segment exceeds u32 ordinal capacity".into(),
            ));
        }
        let mut point_ids = Vec::with_capacity(n);
        let mut original_data = Vec::with_capacity(n);
        let mut point_fields = Vec::with_capacity(n);
        for (id, data, fields) in exported {
            point_ids.push(id);
            original_data.push(data);
            point_fields.push(fields);
        }

        let m = hnsw_config.m;
        let m_max_0 = hnsw_config.m_max_0;
        let m_l = hnsw_config.m_l;
        let ef_construction = (hnsw_config.ef_construct / 2).max(32);

        let mut rng = rand::rng();
        let mut levels: Vec<usize> = Vec::with_capacity(n);
        let mut max_level: usize = 0;
        let mut entry_ord: usize = 0;
        for i in 0..n {
            let r: f64 = rng.random::<f64>();
            let level = assign_hnsw_level(r, m_l);
            levels.push(level);
            if level > max_level {
                max_level = level;
                entry_ord = i;
            }
        }

        // Cosine: SQ8 quantized construction, fit + quantized directly from
        // `original_data` (no spindle projection on the merge path). Dot and
        // Euclid construct on the raw f32 rows; see `merge_build_distance`.
        use super::simd::SQ8Params;
        let dim = original_data[0].len();
        let quantized_flat = if matches!(distance_metric, DistanceMetric::Cosine) {
            SQ8Params::fit(&original_data)
                .quantize_bulk(&original_data)
                .0
        } else {
            Vec::new()
        };

        let adjacency: Vec<StdRwLock<Vec<Vec<u32>>>> = (0..n)
            .map(|i| StdRwLock::new(vec![Vec::new(); levels[i] + 1]))
            .collect();

        let dist_fn_q = |a_ord: usize, b_ord: usize| -> f32 {
            merge_build_distance(
                &distance_metric,
                &original_data,
                &quantized_flat,
                dim,
                a_ord,
                b_ord,
            )
        };

        // Serial bootstrap
        let bootstrap_count = n.min(256);
        let mut current_entry_ord = entry_ord;
        let mut current_max_level = levels[entry_ord];
        for i in 0..bootstrap_count {
            if i == entry_ord {
                continue;
            }
            insert_point_ord(
                i,
                current_entry_ord,
                current_max_level,
                &levels,
                &adjacency,
                ef_construction,
                m,
                m_max_0,
                &dist_fn_q,
            );
            if levels[i] > current_max_level {
                current_max_level = levels[i];
                current_entry_ord = i;
            }
        }

        // Parallel build
        if bootstrap_count < n {
            let par_entry_ord = current_entry_ord;
            let par_max_level = current_max_level;
            use rayon::prelude::*;
            (bootstrap_count..n).into_par_iter().for_each(|i| {
                if i == entry_ord {
                    return;
                }
                insert_point_ord(
                    i,
                    par_entry_ord,
                    par_max_level,
                    &levels,
                    &adjacency,
                    ef_construction,
                    m,
                    m_max_0,
                    &dist_fn_q,
                );
            });
            for i in bootstrap_count..n {
                if levels[i] > current_max_level {
                    current_max_level = levels[i];
                    current_entry_ord = i;
                }
            }
        }

        let adj_plain: Vec<Option<Vec<Vec<u32>>>> = adjacency
            .into_iter()
            .map(|rw| Some(rw.into_inner().unwrap()))
            .collect();

        Ok(PreparedIndex {
            point_ids,
            original_data: original_data.into_iter().map(Some).collect(),
            encoded_data: vec![None; n],
            point_fields,
            levels,
            adjacency: adj_plain,
            entry_ord: current_entry_ord,
            level_zero_preflushed: false,
        })
    }

    /// Phase 1 of split index build: read vectors from a read-only transaction
    /// and build the HNSW graph entirely in memory. Returns a `PreparedIndex`
    /// that can later be flushed to LMDB via `flush_prepared_index`.
    ///
    /// This allows the expensive O(n log n) graph construction to happen
    /// without holding a write lock, dramatically reducing contention during
    /// concurrent ingest.
    pub fn prepare_index_from_flat(
        &self,
        r: &AnyRead<'_>,
    ) -> Result<Option<PreparedIndex>, VectorError> {
        if self.has_index(r)? {
            metrics::counter!(
                "helix_vector_core_hnsw_build_skipped_total",
                "backend" => self.backend_label(),
                "reason" => "already_indexed"
            )
            .increment(1);
            return Ok(None);
        }

        let prepare_start = std::time::Instant::now();
        let phase_start = std::time::Instant::now();
        let vectors = self.get_all_vectors(r, Some(0))?;
        self.observe_hnsw_build_phase(
            "split_prepare",
            "load_flat_vectors",
            vectors.len(),
            phase_start.elapsed(),
        );
        if vectors.is_empty() {
            metrics::counter!(
                "helix_vector_core_hnsw_build_skipped_total",
                "backend" => self.backend_label(),
                "reason" => "empty"
            )
            .increment(1);
            return Ok(None);
        }

        let n = vectors.len();
        let mut point_ids: Vec<u128> = Vec::with_capacity(n);
        let mut point_data: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut original_data: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut encoded_data: Vec<Option<Vec<u8>>> = Vec::with_capacity(n);
        let mut point_fields: Vec<HashMap<String, Value>> = Vec::with_capacity(n);

        // On the LSM/SlateDB backend, prefetch the two namespaces the per-point
        // loop would otherwise hit with one random point-get per vector
        // (`VectorData` via `get_vector_data`, level-0 `Vectors` via
        // `get_encoded_vector`). For a giant segment those N×2 scattered reads
        // span gigabytes of SSTs and thrash the per-collection object-store
        // cache, storming S3 and stalling the build. A single ordered prefix
        // scan per namespace replaces them with cache-friendly sequential reads.
        // Gated to LSM so the LMDB path (local mmap/heed, no S3) stays
        // byte-for-byte unchanged.
        let lsm = self.backend.kind() == BackendKind::Lsm;
        let phase_start = std::time::Instant::now();
        let prefetched_vector_data = if lsm {
            Some(self.collect_vector_data_map(r)?)
        } else {
            None
        };
        let prefetched_encoded = if lsm {
            Some(self.collect_level_zero_encoded_map(r)?)
        } else {
            None
        };
        self.observe_hnsw_build_phase("split_prepare", "prefetch", n, phase_start.elapsed());

        let phase_start = std::time::Instant::now();
        for vector in &vectors {
            let id = vector.get_id();
            let stored = match prefetched_vector_data.as_ref() {
                // Sequential-scan hit: reuse the prefetched row. A point missing
                // from `VectorData` (no fields, no externalized original) has no
                // entry, exactly as `get_vector_data` would return the default.
                Some(map) => map.get(&id).cloned().unwrap_or_default(),
                None => self.get_vector_data(r, id)?,
            };
            let raw = self.original_vector_for_rebuild(r, vector, &stored)?;
            let projected = project_for_search(&raw, &self.spindle)?;
            point_ids.push(id);
            point_data.push(projected);
            original_data.push(raw);
            let encoded = match prefetched_encoded.as_ref() {
                // Non-marker level-0 rows come straight from the sequential scan.
                // Marker rows (externalized to the local sidecar) are absent from
                // the map; resolve them via `get_encoded_vector`, which reads the
                // encoded payload from the LOCAL mmap by ordinal — a local read,
                // not an S3 read — so it does not reintroduce cache thrash.
                Some(map) => match map.get(&id) {
                    Some(bytes) => Some(bytes.clone()),
                    None => self.get_encoded_vector(r, id, 0).ok(),
                },
                None => self.get_encoded_vector(r, id, 0).ok(),
            };
            encoded_data.push(encoded);
            point_fields.push(stored.fields);
        }
        self.observe_hnsw_build_phase("split_prepare", "prepare_rows", n, phase_start.elapsed());

        let m = self.config.m;
        let m_max_0 = self.config.m_max_0;
        let m_l = self.config.m_l;
        let ef_construction = (self.config.ef_construct / 2).max(32);

        let mut rng = rand::rng();
        let phase_start = std::time::Instant::now();
        let mut levels: Vec<usize> = Vec::with_capacity(n);
        let mut max_level: usize = 0;
        let mut entry_ord: usize = 0;
        for i in 0..n {
            let r: f64 = rng.random::<f64>();
            let level = assign_hnsw_level(r, m_l);
            levels.push(level);
            if level > max_level {
                max_level = level;
                entry_ord = i;
            }
        }
        self.observe_hnsw_build_phase("split_prepare", "assign_levels", n, phase_start.elapsed());

        // ── Flash build: SQ8 quantization for construction distances ──
        //
        // Quantize all projected vectors to u8 for use during graph
        // construction. This reduces memory bandwidth per distance call
        // from 2×dim×4 bytes (f32) to 2×dim bytes (u8) and enables
        // faster NEON/AVX2 integer SIMD paths.
        //
        // The full-precision f32 vectors are retained for the final
        // stored index; only the graph topology (neighbor selection)
        // uses approximate SQ8 distances.
        //
        // Cosine only: SQ8's per-dimension min-shift/scale distorts inner
        // products and L2, so Dot/Euclid construct on the projected f32
        // rows exactly like the monolithic build (see `merge_build_distance`).
        use super::simd::SQ8Params;
        let phase_start = std::time::Instant::now();
        let dim = point_data[0].len();
        let (quantized_flat, point_data) = if matches!(self.distance_metric, DistanceMetric::Cosine)
        {
            let (quantized_flat, _qdim) = SQ8Params::fit(&point_data).quantize_bulk(&point_data);
            // Drop the f32 working copies — only u8 needed for construction
            (quantized_flat, Vec::new())
        } else {
            (Vec::new(), point_data)
        };
        self.observe_hnsw_build_phase("split_prepare", "quantize", n, phase_start.elapsed());

        if n > u32::MAX as usize {
            return Err(VectorError::VectorCoreError(
                "HNSW prepared segment exceeds u32 ordinal capacity".into(),
            ));
        }

        let phase_start = std::time::Instant::now();
        let adjacency: Vec<StdRwLock<Vec<Vec<u32>>>> = (0..n)
            .map(|i| StdRwLock::new(vec![Vec::new(); levels[i] + 1]))
            .collect();
        self.observe_hnsw_build_phase(
            "split_prepare",
            "allocate_adjacency",
            n,
            phase_start.elapsed(),
        );

        let dist_fn_q = |a_ord: usize, b_ord: usize| -> f32 {
            merge_build_distance(
                &self.distance_metric,
                &point_data,
                &quantized_flat,
                dim,
                a_ord,
                b_ord,
            )
        };

        // Serial bootstrap
        let phase_start = std::time::Instant::now();
        let bootstrap_count = n.min(256);
        let mut current_entry_ord = entry_ord;
        let mut current_max_level = levels[entry_ord];

        for i in 0..bootstrap_count {
            if i == entry_ord {
                continue;
            }
            insert_point_ord(
                i,
                current_entry_ord,
                current_max_level,
                &levels,
                &adjacency,
                ef_construction,
                m,
                m_max_0,
                &dist_fn_q,
            );
            if levels[i] > current_max_level {
                current_max_level = levels[i];
                current_entry_ord = i;
            }
        }
        self.observe_hnsw_build_phase("split_prepare", "bootstrap", n, phase_start.elapsed());

        // Parallel build
        let phase_start = std::time::Instant::now();
        if bootstrap_count < n {
            let par_entry_ord = current_entry_ord;
            let par_max_level = current_max_level;

            use rayon::prelude::*;
            (bootstrap_count..n).into_par_iter().for_each(|i| {
                if i == entry_ord {
                    return;
                }
                insert_point_ord(
                    i,
                    par_entry_ord,
                    par_max_level,
                    &levels,
                    &adjacency,
                    ef_construction,
                    m,
                    m_max_0,
                    &dist_fn_q,
                );
            });

            for i in bootstrap_count..n {
                if levels[i] > current_max_level {
                    current_max_level = levels[i];
                    current_entry_ord = i;
                }
            }
        }
        self.observe_hnsw_build_phase("split_prepare", "parallel_build", n, phase_start.elapsed());

        // Extract adjacency into plain vecs (drop the RwLocks)
        let phase_start = std::time::Instant::now();
        let adj_plain: Vec<Option<Vec<Vec<u32>>>> = adjacency
            .into_iter()
            .map(|rw| Some(rw.into_inner().unwrap()))
            .collect();
        self.observe_hnsw_build_phase(
            "split_prepare",
            "extract_adjacency",
            n,
            phase_start.elapsed(),
        );
        self.observe_hnsw_build_phase("split_prepare", "total", n, prepare_start.elapsed());

        Ok(Some(PreparedIndex {
            point_ids,
            original_data: original_data.into_iter().map(Some).collect(),
            encoded_data,
            point_fields,
            levels,
            adjacency: adj_plain,
            entry_ord: current_entry_ord,
            level_zero_preflushed: true,
        }))
    }

    /// Phase 2 of split index build: flush a pre-computed HNSW graph to LMDB.
    /// This only does LMDB writes (no graph construction), so the write lock
    /// is held for a much shorter time than `build_index_from_flat`.
    pub fn flush_prepared_index(
        &self,
        txn: &mut RwTxn,
        prepared: PreparedIndex,
    ) -> Result<(), VectorError> {
        {
            let rd = self.backend.read_borrowed(&*txn);
            if self.has_index(&rd)? {
                return Ok(());
            }
        }

        let mut prepared = prepared;
        let n = prepared.point_ids.len();
        // The merge path leaves `point_fields` empty (payloads were written
        // by `insert_flat` upstream). Single-segment build path populates
        // them from LMDB and we write them through here.
        let write_fields = prepared.point_fields.len() == n;

        // Write vector data + fields
        let flush_start = std::time::Instant::now();
        let phase_start = std::time::Instant::now();
        for i in 0..n {
            let id = prepared.point_ids[i];
            let level = prepared.levels[i];
            let fields = if write_fields && !prepared.level_zero_preflushed {
                Some(std::mem::take(&mut prepared.point_fields[i]))
            } else {
                None
            };
            let raw = prepared.raw_at(i)?;

            if !prepared.level_zero_preflushed {
                self.put_raw_vector(txn, id, 0, raw)?;
            }
            if level > 0 {
                self.put_raw_vector(txn, id, level, raw)?;
            }

            if let Some(fields) = fields {
                self.maybe_put_vector_data(txn, id, Some(fields), raw)?;
            }
        }
        self.observe_hnsw_build_phase("split_flush", "flush_vectors", n, phase_start.elapsed());

        // Write packed adjacency
        let phase_start = std::time::Instant::now();
        for i in 0..n {
            for lvl in 0..prepared.adjacency_level_count_at(i) {
                let neighbors = prepared.neighbor_ids_at(i, lvl)?;
                let block = NeighborBlock::from_ids(neighbors);
                let key = Self::neighbor_list_key(prepared.point_ids[i], lvl);
                let encoded = Self::encode_neighbor_block(&block);
                self.backend
                    .put_heed(
                        txn,
                        self.seg_ns(SegmentDb::HnswNeighbors),
                        &key,
                        encoded.as_slice(),
                    )
                    .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            }
        }
        self.observe_hnsw_build_phase("split_flush", "flush_neighbors", n, phase_start.elapsed());

        // Set entry point
        let phase_start = std::time::Instant::now();
        let entry_id = prepared.point_ids[prepared.entry_ord];
        let entry_vector =
            HVector::from_slice(entry_id, prepared.levels[prepared.entry_ord], vec![]);
        self.set_entry_point(txn, &entry_vector)?;
        self.observe_hnsw_build_phase("split_flush", "set_entry", n, phase_start.elapsed());
        self.observe_hnsw_build_phase("split_flush", "total", n, flush_start.elapsed());

        Ok(())
    }

    /// Prewrite level-0 vector rows and payload fields for a merge target from
    /// the prepared artifact itself. This avoids a duplicate merge-export vector
    /// buffer while preserving the publish order: all level-0 rows exist before
    /// HNSW neighbor blocks are encoded.
    pub fn flush_prepared_flat_chunk(
        &self,
        txn: &mut RwTxn,
        prepared: &mut PreparedIndex,
        start: usize,
        end: usize,
    ) -> Result<(), VectorError> {
        let n = prepared.point_ids.len();
        let end = end.min(n);
        if start >= end {
            return Ok(());
        }
        let write_fields = prepared.point_fields.len() == n;
        for i in start..end {
            let id = prepared.point_ids[i];
            let fields = if write_fields {
                Some(std::mem::take(&mut prepared.point_fields[i]))
            } else {
                None
            };
            let raw = prepared.raw_at(i)?;
            self.put_raw_vector(txn, id, 0, raw)?;
            self.maybe_put_vector_data(txn, id, fields, raw)?;
        }
        prepared.mark_level_zero_preflushed();
        Ok(())
    }

    pub fn flush_prepared_flat_chunk_be(
        &self,
        w: &mut AnyWrite<'_>,
        prepared: &mut PreparedIndex,
        start: usize,
        end: usize,
    ) -> Result<(), VectorError> {
        let n = prepared.point_ids.len();
        let end = end.min(n);
        if start >= end {
            return Ok(());
        }
        let write_fields = prepared.point_fields.len() == n;
        for i in start..end {
            let id = prepared.point_ids[i];
            let fields = if write_fields {
                Some(std::mem::take(&mut prepared.point_fields[i]))
            } else {
                None
            };
            let raw = prepared.raw_at(i)?;
            self.insert_flat_be(w, raw, Some(id), fields)?;
        }
        prepared.mark_level_zero_preflushed();
        Ok(())
    }

    /// Flush a slice `[start, end)` of a `PreparedIndex` to LMDB.
    ///
    /// `point_fields` are taken (mutated to empty `HashMap`s) so successive
    /// calls don't double-write. Caller invokes `finalize_prepared_index_entry`
    /// once after the final chunk to set the entry point and bail-out gate.
    ///
    /// Used by chunked-publish to keep each exclusive write txn under ~250 ms
    /// even when merging 50k+ vectors into a new segment. The new segment is
    /// only "published" (made visible to readers) by a separate metadata-swap
    /// txn after all chunks finish, so partial writes are invisible.
    pub fn flush_prepared_index_chunk(
        &self,
        txn: &mut RwTxn,
        prepared: &mut PreparedIndex,
        start: usize,
        end: usize,
    ) -> Result<(), VectorError> {
        let n = prepared.point_ids.len();
        let end = end.min(n);
        if start >= end {
            return Ok(());
        }
        let flush_start = std::time::Instant::now();
        // Merge path leaves `point_fields` empty (payloads written by
        // `insert_flat` upstream); single-segment build populates them
        // and writes them through here.
        let write_fields = prepared.point_fields.len() == n && !prepared.level_zero_preflushed;

        for i in start..end {
            let id = prepared.point_ids[i];
            let level = prepared.levels[i];
            let fields = if write_fields {
                Some(std::mem::take(&mut prepared.point_fields[i]))
            } else {
                None
            };
            let raw = prepared.raw_at(i)?;

            if !prepared.level_zero_preflushed {
                self.put_raw_vector(txn, id, 0, raw)?;
            }
            if level > 0 {
                self.put_raw_vector(txn, id, level, raw)?;
            }
            if let Some(fields) = fields {
                self.maybe_put_vector_data(txn, id, Some(fields), raw)?;
            }
        }

        for i in start..end {
            for lvl in 0..prepared.adjacency_level_count_at(i) {
                let neighbors = prepared.neighbor_ids_at(i, lvl)?;
                let block = NeighborBlock::from_ids(neighbors);
                let key = Self::neighbor_list_key(prepared.point_ids[i], lvl);
                let encoded = Self::encode_neighbor_block(&block);
                self.backend
                    .put_heed(
                        txn,
                        self.seg_ns(SegmentDb::HnswNeighbors),
                        &key,
                        encoded.as_slice(),
                    )
                    .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            }
            prepared.drain_row(i);
        }

        self.observe_hnsw_build_phase("split_chunk", "total", end - start, flush_start.elapsed());
        Ok(())
    }

    pub fn flush_prepared_index_chunk_be(
        &self,
        w: &mut AnyWrite<'_>,
        prepared: &mut PreparedIndex,
        start: usize,
        end: usize,
    ) -> Result<(), VectorError> {
        let n = prepared.point_ids.len();
        let end = end.min(n);
        if start >= end {
            return Ok(());
        }
        let flush_start = std::time::Instant::now();
        let write_fields = prepared.point_fields.len() == n && !prepared.level_zero_preflushed;

        for i in start..end {
            let id = prepared.point_ids[i];
            let level = prepared.levels[i];
            let fields = if write_fields {
                Some(std::mem::take(&mut prepared.point_fields[i]))
            } else {
                None
            };
            let raw = prepared.raw_at(i)?;
            if !prepared.level_zero_preflushed {
                self.insert_flat_be(w, raw, Some(id), fields)?;
            }
            if level > 0 {
                let externalize = self.can_externalize_cached_turbo_quant_vector(id, raw.len())?;
                let encoded;
                let stored = if externalize {
                    EXTERNALIZED_VECTOR_MARKER
                } else {
                    encoded = encode_vector(raw, &self.spindle)?;
                    encoded.as_slice()
                };
                self.backend
                    .put(
                        w,
                        self.seg_ns(SegmentDb::Vectors),
                        &Self::vector_key(id, level),
                        stored,
                    )
                    .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            }
        }

        for i in start..end {
            for lvl in 0..prepared.adjacency_level_count_at(i) {
                let neighbors = prepared.neighbor_ids_at(i, lvl)?;
                let block = NeighborBlock::from_ids(neighbors);
                let key = Self::neighbor_list_key(prepared.point_ids[i], lvl);
                let encoded = Self::encode_neighbor_block(&block);
                self.backend
                    .put(
                        w,
                        self.seg_ns(SegmentDb::HnswNeighbors),
                        &key,
                        encoded.as_slice(),
                    )
                    .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            }
            prepared.drain_row(i);
        }
        self.observe_hnsw_build_phase(
            "split_chunk_be",
            "total",
            end - start,
            flush_start.elapsed(),
        );
        Ok(())
    }

    /// Finalize a chunked `PreparedIndex` by writing its entry point.
    /// Must be called in the same publish txn (or earlier) but after all
    /// `flush_prepared_index_chunk` calls. Idempotent if `has_index`.
    pub fn finalize_prepared_index_entry(
        &self,
        txn: &mut RwTxn,
        prepared: &PreparedIndex,
    ) -> Result<(), VectorError> {
        {
            let rd = self.backend.read_borrowed(&*txn);
            if self.has_index(&rd)? {
                return Ok(());
            }
        }
        if prepared.point_ids.is_empty() {
            return Ok(());
        }
        let entry_id = prepared.point_ids[prepared.entry_ord];
        let entry_vector =
            HVector::from_slice(entry_id, prepared.levels[prepared.entry_ord], vec![]);
        self.set_entry_point(txn, &entry_vector)?;
        Ok(())
    }

    pub fn finalize_prepared_index_entry_be(
        &self,
        w: &mut AnyWrite<'_>,
        prepared: &PreparedIndex,
    ) -> Result<(), VectorError> {
        if prepared.point_ids.is_empty() {
            return Ok(());
        }
        self.persist_hvec_sidecar_blob_be(w)?;
        self.persist_hvec_sidecar_ordinals_blob_be(w, &prepared.point_ids)?;
        let entry_id = prepared.point_ids[prepared.entry_ord];
        self.backend
            .put(
                w,
                self.seg_ns(SegmentDb::Vectors),
                ENTRY_POINT_KEY.as_bytes(),
                &entry_id.to_be_bytes(),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    #[inline(always)]
    fn get_neighbor_ids(
        &self,
        r: &AnyRead<'_>,
        id: u128,
        level: usize,
    ) -> Result<Vec<u128>, VectorError> {
        Ok(self.get_neighbor_block(r, id, level)?.ids)
    }

    fn cache_neighbor_block(&self, id: u128, level: usize, block: &NeighborBlock) {
        if let Some(shared) = self.shared_caches.as_ref() {
            let key = super::shared_cache::CacheKey::new(self.cache_namespace, id, level);
            if level > 0 {
                shared.nav_neighbor.put(key, block.clone());
            }
            shared.neighbor.put(key, block.clone());
            return;
        }
        if level > 0 {
            if let Ok(mut cache) = self.nav_neighbor_cache.lock() {
                cache.put((id, level), block.clone());
            }
        }
        if let Ok(mut cache) = self.neighbor_cache.lock() {
            cache.put((id, level), block.clone());
        }
    }

    fn get_neighbor_block(
        &self,
        r: &AnyRead<'_>,
        id: u128,
        level: usize,
    ) -> Result<NeighborBlock, VectorError> {
        // Cache instrumentation: per-cache hit/miss counters by tier.
        // `tier="nav"` = level>0 HNSW navigation layer (small, hot, reused across queries).
        // `tier="level0"` = level=0 neighbor blocks (bulk of search work, cold set huge).
        // Signal used to tune HELIX_NEIGHBOR_CACHE_CAP / HELIX_NAV_CACHE_CAP.
        if let Some(shared) = self.shared_caches.as_ref() {
            let key = super::shared_cache::CacheKey::new(self.cache_namespace, id, level);
            if level > 0 {
                if let Some(block) = shared.nav_neighbor.get(&key) {
                    metrics::counter!("helix_vector_cache_hits_total",
                        "cache" => "neighbor", "tier" => "nav")
                    .increment(1);
                    return Ok(block);
                }
            }
            if let Some(block) = shared.neighbor.get(&key) {
                metrics::counter!("helix_vector_cache_hits_total",
                    "cache" => "neighbor", "tier" => "level0")
                .increment(1);
                return Ok(block);
            }
            metrics::counter!("helix_vector_cache_misses_total",
                "cache" => "neighbor",
                "tier" => if level > 0 { "nav" } else { "level0" })
            .increment(1);
            let block = self.get_neighbor_block_uncached(r, id, level)?;
            self.cache_neighbor_block(id, level, &block);
            return Ok(block);
        }
        if level > 0 {
            if let Ok(mut cache) = self.nav_neighbor_cache.lock() {
                if let Some(block) = cache.get(&(id, level)) {
                    metrics::counter!("helix_vector_cache_hits_total",
                        "cache" => "neighbor", "tier" => "nav")
                    .increment(1);
                    return Ok(block.clone());
                }
            }
        }
        if let Ok(mut cache) = self.neighbor_cache.lock() {
            if let Some(block) = cache.get(&(id, level)) {
                metrics::counter!("helix_vector_cache_hits_total",
                    "cache" => "neighbor", "tier" => "level0")
                .increment(1);
                return Ok(block.clone());
            }
        }
        metrics::counter!("helix_vector_cache_misses_total",
            "cache" => "neighbor",
            "tier" => if level > 0 { "nav" } else { "level0" })
        .increment(1);
        let block = self.get_neighbor_block_uncached(r, id, level)?;
        self.cache_neighbor_block(id, level, &block);
        Ok(block)
    }

    fn ensure_navigation_cache(&self, r: &AnyRead<'_>, entry_point: &HVector) {
        if entry_point.get_level() == 0 {
            return;
        }
        if let Ok(root) = self.nav_cache_root.read() {
            if root.as_ref() == Some(&(entry_point.get_id(), entry_point.get_level())) {
                return;
            }
        }

        let warm_started = std::time::Instant::now();
        let max_warm_entries = nav_cache_warm_cap();
        if max_warm_entries == 0 {
            metrics::histogram!("helix_nav_cache_warm_items", "outcome" => "disabled").record(0.0);
            metrics::histogram!("helix_nav_cache_warm_duration_ms", "outcome" => "disabled")
                .record(warm_started.elapsed().as_secs_f64() * 1000.0);
            return;
        }
        let mut truncated = false;
        let mut stack = vec![(entry_point.get_id(), entry_point.get_level())];
        let mut visited: HashSet<(u128, usize)> = HashSet::new();

        while let Some((id, level)) = stack.pop() {
            if level == 0 {
                continue;
            }
            if visited.len() >= max_warm_entries {
                truncated = true;
                break;
            }
            if !visited.insert((id, level)) {
                continue;
            }
            if let Ok(block) = self.get_neighbor_block_uncached(r, id, level) {
                self.cache_neighbor_block(id, level, &block);
                for neighbor_id in &block.ids {
                    stack.push((*neighbor_id, level));
                }
            }
            if let Ok(vector) = self.get_vector(r, id, level, true) {
                self.cache_vector(id, level, vector.get_data());
            }
            stack.push((id, level.saturating_sub(1)));
        }

        let outcome = if truncated { "truncated" } else { "complete" };
        metrics::histogram!("helix_nav_cache_warm_items", "outcome" => outcome)
            .record(visited.len() as f64);
        metrics::histogram!("helix_nav_cache_warm_duration_ms", "outcome" => outcome)
            .record(warm_started.elapsed().as_secs_f64() * 1000.0);
        if truncated {
            metrics::counter!("helix_nav_cache_warm_truncated_total").increment(1);
        }

        if let Ok(mut root) = self.nav_cache_root.write() {
            *root = Some((entry_point.get_id(), entry_point.get_level()));
        }
    }

    #[inline(always)]
    fn set_neighbours<'a>(
        &self,
        txn: &mut RwTxn,
        id: u128,
        neighbors: &'a BinaryHeap<HVector>,
        level: usize,
    ) -> Result<(), VectorError> {
        let desired_neighbors: Vec<u128> = neighbors
            .iter()
            .map(HVector::get_id)
            .filter(|neighbor_id| *neighbor_id != id)
            .unique()
            .collect();
        let existing_neighbors: HashSet<u128> = {
            let r = self.backend.read_borrowed(&*txn);
            self.get_neighbor_block_uncached(&r, id, level)?
                .ids
                .into_iter()
                .collect()
        };
        let desired_set: HashSet<u128> = desired_neighbors.iter().copied().collect();

        self.put_neighbor_ids(txn, id, level, &desired_neighbors)?;

        for neighbor_id in existing_neighbors.difference(&desired_set) {
            let mut reverse_neighbors = {
                let r = self.backend.read_borrowed(&*txn);
                self.get_neighbor_block_uncached(&r, *neighbor_id, level)?
                    .ids
            };
            reverse_neighbors.retain(|other| *other != id);
            self.put_neighbor_ids(txn, *neighbor_id, level, &reverse_neighbors)?;
        }

        for neighbor_id in desired_set.difference(&existing_neighbors) {
            let mut reverse_neighbors = {
                let r = self.backend.read_borrowed(&*txn);
                self.get_neighbor_block_uncached(&r, *neighbor_id, level)?
                    .ids
            };
            if !reverse_neighbors.contains(&id) {
                reverse_neighbors.push(id);
            }
            self.put_neighbor_ids(txn, *neighbor_id, level, &reverse_neighbors)?;
        }

        Ok(())
    }

    fn select_neighbors<'a, F>(
        &'a self,
        r: &AnyRead<'_>,
        query: &'a HVector,
        mut cands: BinaryHeap<HVector>,
        level: usize,
        should_extend: bool,
        filter: Option<&[F]>,
    ) -> Result<BinaryHeap<HVector>, VectorError>
    where
        F: Fn(&HVector) -> bool,
    {
        // Same caps as the bulk builder (`insert_point_ord`): the dense base
        // layer keeps up to `m_max_0` (2*m) links, upper layers `m`.
        let m: usize = if level == 0 {
            self.config.m_max_0
        } else {
            self.config.m
        };
        let mut visited: HashSet<u128> = HashSet::new();
        if should_extend {
            let mut ranked: Vec<(u128, f32)> = Vec::with_capacity(m * cands.len() + cands.len());
            for candidate in cands.iter() {
                let distance = match candidate.distance {
                    Some(distance) => distance,
                    None => self.distance_between(candidate.get_data(), query.get_data())?,
                };
                ranked.push((candidate.get_id(), distance));
            }
            for candidate in cands.iter() {
                for neighbor_id in self.get_neighbor_ids(r, candidate.get_id(), level)? {
                    if !visited.insert(neighbor_id) {
                        continue;
                    }
                    let distance = if let Some(fs) = filter {
                        let mut neighbor = self.get_vector(r, neighbor_id, level, true)?;
                        let distance =
                            self.distance_between(neighbor.get_data(), query.get_data())?;
                        neighbor.set_distance(distance);
                        if !fs.iter().all(|f| f(&neighbor)) {
                            continue;
                        }
                        distance
                    } else {
                        match self.score_neighbor_distance(
                            r,
                            &PreparedSpindleQuery::None,
                            neighbor_id,
                            level,
                            query.get_data(),
                        )? {
                            Some(distance) => distance,
                            None => continue,
                        }
                    };
                    ranked.push((neighbor_id, distance));
                }
            }
            ranked.sort_by(|lhs, rhs| lhs.1.partial_cmp(&rhs.1).unwrap_or(Ordering::Equal));
            let mut selected = HashSet::with_capacity(ranked.len());
            ranked.retain(|(id, _)| selected.insert(*id));
            ranked.truncate(m);

            let mut result = BinaryHeap::with_capacity(ranked.len());
            for (neighbor_id, distance) in ranked {
                let mut neighbor = match self.get_vector(r, neighbor_id, level, true) {
                    Ok(vector) => vector,
                    Err(VectorError::VectorNotFound(_)) => continue,
                    Err(e) => return Err(e),
                };
                neighbor.set_distance(distance);
                result.push(neighbor);
            }
            Ok(result)
        } else {
            Ok(cands.take_inord(m))
        }
    }

    /// Graph-entangled HNSW bridge (Phase 2).
    ///
    /// Called when the HNSW frontier dead-ends (`candidates` empty, `results`
    /// short of `ef`) on a level-0 search. Expands via graph out-edges of
    /// already-visited HNSW nodes: every graph neighbor that passes the
    /// caller's filter gets scored and pushed back into the candidate heap
    /// and result set, exactly as if HNSW had reached it organically.
    ///
    /// Returns the number of newly-admitted results whose id was *first*
    /// discovered through graph adjacency, so callers can emit
    /// `helix_graph_bridge_new_results_total`. No-ops (returns 0) when the
    /// env flag is off, no graph handle is attached, `level != 0`, the
    /// frontier isn't actually empty, or `visited` is empty.
    #[allow(clippy::too_many_arguments)]
    fn graph_bridge_level0<ScoreFilter, IdFilter>(
        &self,
        r: &AnyRead<'_>,
        query_data: &[f32],
        prepared: &PreparedSpindleQuery,
        visited: &mut HashSet<u128>,
        candidates: &mut ArenaHeap<Candidate>,
        results: &mut bumpalo::collections::Vec<'_, (u128, f32)>,
        worst_distance: &mut f32,
        ef: usize,
        level: usize,
        labels: Option<VectorSearchMetrics<'_>>,
        score_filter: Option<&[ScoreFilter]>,
        id_filter: Option<&[IdFilter]>,
    ) -> Result<usize, VectorError>
    where
        ScoreFilter: Fn(&HVector) -> bool,
        IdFilter: Fn(u128) -> bool,
    {
        use super::named_vectors::NamedVectorManager;

        // Skip condition checks. Return early and emit the skip counter so
        // dashboards can see why the bridge is (or isn't) firing. These
        // conditions are also enforced by callers, but checking here keeps
        // the bridge safe to call from any search loop without re-auditing
        // the caller each time.
        if level != 0 {
            return Ok(0);
        }
        if !NamedVectorManager::graph_entangled_enabled() {
            self.bridge_skip(labels, "flag_off");
            return Ok(0);
        }
        let Some(edges_db) = self.graph_out_edges_db else {
            self.bridge_skip(labels, "no_edges_db");
            return Ok(0);
        };
        // The graph adjacency DB is a heed `Database`, so it needs the heed
        // `RoTxn` underneath `r`. Graph-entanglement only runs on the LMDB
        // backend (the legacy graph path); on LSM `lmdb_ro()` is `None` and the
        // bridge is a no-op.
        let Some(txn) = r.lmdb_ro() else {
            self.bridge_skip(labels, "no_edges_db");
            return Ok(0);
        };
        if !candidates.is_empty() {
            self.bridge_skip(labels, "frontier_nonempty");
            return Ok(0);
        }
        if results.len() >= ef {
            self.bridge_skip(labels, "ef_satisfied");
            return Ok(0);
        }
        if visited.is_empty() {
            self.bridge_skip(labels, "no_visited");
            return Ok(0);
        }

        let max_seeds = NamedVectorManager::graph_bridge_max_seeds();
        let max_expansions = NamedVectorManager::graph_bridge_max_expansions();
        let bridge_start = std::time::Instant::now();

        // Snapshot seeds before we start inserting new ids into `visited`;
        // iterating `visited` directly while we grow it would double-count
        // and risk inflated metrics.
        let seeds: Vec<u128> = visited.iter().copied().take(max_seeds).collect();
        let seeds_examined = seeds.len();

        let mut lmdb_reads: u64 = 0;
        let mut filter_passed: u64 = 0;
        let mut filter_total: u64 = 0;
        let mut new_results: usize = 0;
        let mut expansions: usize = 0;

        'seeds: for seed_id in seeds {
            // Prefix-iterate the first 16 bytes of the out-edge key to get
            // every outgoing edge regardless of label. See
            // HelixGraphStorage::drop_node for the same pattern. Values
            // (edge_id || to_node) are unpacked via `unpack_adj_edge_data`.
            let iter = match edges_db.prefix_iter(txn, &seed_id.to_be_bytes()) {
                Ok(iter) => iter,
                Err(e) => return Err(VectorError::from(e)),
            };
            for item in iter {
                let (_, value) = item?;
                lmdb_reads += 1;
                // Layout: [edge_id(16) | to_node(16)]; see
                // HelixGraphStorage::pack_edge_data.
                if value.len() < 32 {
                    continue;
                }
                let neighbor_id = u128::from_be_bytes(match value[16..32].try_into() {
                    Ok(bytes) => bytes,
                    Err(_) => continue,
                });
                if !visited.insert(neighbor_id) {
                    continue;
                }
                if self.is_delete_tombstoned(neighbor_id) {
                    continue;
                }

                filter_total += 1;
                let distance = if let Some(score_fs) = score_filter {
                    let neighbor = match self.get_vector(r, neighbor_id, level, true) {
                        Ok(n) => n,
                        Err(VectorError::VectorNotFound(_)) => {
                            self.bridge_outcome(labels, "scoring_failed");
                            continue;
                        }
                        Err(e) => return Err(e),
                    };
                    if !score_fs.iter().all(|f| f(&neighbor)) {
                        self.bridge_outcome(labels, "filter_rejected");
                        continue;
                    }
                    filter_passed += 1;
                    self.scored_distance(
                        r,
                        prepared,
                        neighbor.get_id(),
                        neighbor.get_level(),
                        neighbor.get_data(),
                        query_data,
                    )?
                } else if let Some(id_fs) = id_filter {
                    if !id_fs.iter().all(|f| f(neighbor_id)) {
                        self.bridge_outcome(labels, "filter_rejected");
                        continue;
                    }
                    filter_passed += 1;
                    match self.score_neighbor_distance(
                        r,
                        prepared,
                        neighbor_id,
                        level,
                        query_data,
                    )? {
                        Some(d) => d,
                        None => {
                            self.bridge_outcome(labels, "scoring_failed");
                            continue;
                        }
                    }
                } else {
                    filter_passed += 1;
                    match self.score_neighbor_distance(
                        r,
                        prepared,
                        neighbor_id,
                        level,
                        query_data,
                    )? {
                        Some(d) => d,
                        None => {
                            self.bridge_outcome(labels, "scoring_failed");
                            continue;
                        }
                    }
                };

                if results.len() < ef || distance < *worst_distance {
                    candidates.push(Candidate {
                        id: neighbor_id,
                        distance,
                    });
                    results.push((neighbor_id, distance));
                    new_results += 1;
                    self.bridge_outcome(labels, "candidate_added");

                    if results.len() > ef {
                        let worst_idx = results
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1 .1.partial_cmp(&b.1 .1).unwrap_or(Ordering::Equal))
                            .map(|(idx, _)| idx)
                            .unwrap();
                        results.swap_remove(worst_idx);
                    }
                    *worst_distance = results
                        .iter()
                        .map(|r| r.1)
                        .fold(f32::NEG_INFINITY, f32::max);
                } else {
                    self.bridge_outcome(labels, "duplicate");
                }

                expansions += 1;
                if expansions >= max_expansions {
                    break 'seeds;
                }
            }
        }

        if let Some(labels) = labels {
            let collection = labels.collection.to_string();
            let vector = labels.vector.to_string();
            metrics::histogram!(
                "helix_graph_bridge_seeds_examined",
                "collection" => collection.clone(),
                "vector" => vector.clone(),
            )
            .record(seeds_examined as f64);
            metrics::counter!(
                "helix_graph_bridge_lmdb_reads_total",
                "collection" => collection.clone(),
                "vector" => vector.clone(),
            )
            .increment(lmdb_reads);
            metrics::counter!(
                "helix_graph_bridge_new_results_total",
                "collection" => collection.clone(),
                "vector" => vector.clone(),
            )
            .increment(new_results as u64);
            metrics::histogram!(
                "helix_graph_bridge_duration_ms",
                "collection" => collection.clone(),
                "vector" => vector.clone(),
            )
            .record(bridge_start.elapsed().as_secs_f64() * 1000.0);
            if filter_total > 0 {
                metrics::gauge!(
                    "helix_graph_bridge_filter_pass_rate",
                    "collection" => collection,
                    "vector" => vector,
                )
                .set(filter_passed as f64 / filter_total as f64);
            }
        }

        Ok(new_results)
    }

    /// Graph-entangled insert seeding (Phase 3).
    ///
    /// Returns up to `HELIX_GRAPH_BRIDGE_MAX_SEEDS` out-edge targets of
    /// `id` from the attached graph. When the point being inserted has
    /// pre-existing graph edges (CE's normal ingest order: symbols and
    /// edges first, embeddings later), those targets are strong semantic
    /// priors for HNSW neighbor candidates. Returns empty when the flag
    /// is off, no graph DB is attached, or the node has no outgoing edges.
    fn collect_graph_seed_ids(&self, r: &AnyRead<'_>, id: u128) -> Vec<u128> {
        use super::named_vectors::NamedVectorManager;

        let Some(edges_db) = self.graph_out_edges_db else {
            return Vec::new();
        };
        // The graph adjacency DB needs the heed `RoTxn` under `r`; graph-
        // entangled seeding is LMDB-only, so on LSM (`lmdb_ro() == None`)
        // there are no seeds.
        let Some(txn) = r.lmdb_ro() else {
            return Vec::new();
        };
        let max_seeds = NamedVectorManager::graph_bridge_max_seeds();
        let iter = match edges_db.prefix_iter(txn, &id.to_be_bytes()) {
            Ok(iter) => iter,
            Err(_) => return Vec::new(),
        };
        let mut out = Vec::with_capacity(max_seeds.min(16));
        for item in iter {
            let Ok((_, value)) = item else {
                break;
            };
            if value.len() < 32 {
                continue;
            }
            let Ok(bytes) = value[16..32].try_into() else {
                continue;
            };
            out.push(u128::from_be_bytes(bytes));
            if out.len() >= max_seeds {
                break;
            }
        }
        out
    }

    #[inline]
    fn bridge_skip(&self, labels: Option<VectorSearchMetrics<'_>>, reason: &'static str) {
        if let Some(labels) = labels {
            metrics::counter!(
                "helix_graph_bridge_skipped_total",
                "collection" => labels.collection.to_string(),
                "vector" => labels.vector.to_string(),
                "reason" => reason.to_string(),
            )
            .increment(1);
        }
    }

    #[inline]
    fn bridge_outcome(&self, labels: Option<VectorSearchMetrics<'_>>, outcome: &'static str) {
        if let Some(labels) = labels {
            metrics::counter!(
                "helix_graph_bridge_expansions_total",
                "collection" => labels.collection.to_string(),
                "vector" => labels.vector.to_string(),
                "bridge_outcome" => outcome.to_string(),
            )
            .increment(1);
        }
    }

    fn search_level<'a, F>(
        &'a self,
        r: &AnyRead<'_>,
        query: &'a HVector,
        entry_point: &'a mut HVector,
        ef: usize,
        level: usize,
        filter: Option<&[F]>,
        arena: &'a Bump,
        prepared: &PreparedSpindleQuery,
        metrics_labels: Option<VectorSearchMetrics<'_>>,
        seed_ids: &[u128],
        ordinal_cache: &mut SidecarOrdinalRequestCache,
    ) -> Result<BinaryHeap<HVector>, VectorError>
    where
        F: Fn(&HVector) -> bool,
    {
        // ── Worst-distance tracking ──
        // Use a simple sorted Vec<(u128, f32)> for the result set.
        // Worst distance is tracked incrementally: we only recompute
        // after eviction (O(ef) but amortized across ef insertions).
        // This avoids the previous approach of storing full HVector
        // objects (each ~3KB at 768-dim) in the results buffer.
        let mut visited: HashSet<u128> = HashSet::with_capacity(ef * 2);
        let mut candidates: ArenaHeap<Candidate> = ArenaHeap::with_capacity(arena, ef.max(1));
        let mut results = bumpalo::collections::Vec::with_capacity_in(ef + 1, arena);
        let mut neighbor_blocks = 0usize;
        let mut neighbor_edges_seen = 0usize;
        let mut neighbor_edges_scored = 0usize;
        let mut approx_blocks = 0usize;
        let mut approx_candidates_kept = 0usize;
        let mut filtered_out = 0usize;
        let neighbor_prepared = if level == 0 {
            Self::neighbor_code_config(query.get_data().len())
                .and_then(|config| prepare_query(query.get_data(), &config).ok())
        } else {
            None
        };
        let ep_dist = self.scored_distance(
            r,
            prepared,
            entry_point.get_id(),
            entry_point.get_level(),
            entry_point.get_data(),
            query.get_data(),
        )?;
        candidates.push(Candidate {
            id: entry_point.get_id(),
            distance: ep_dist,
        });
        results.push((entry_point.get_id(), ep_dist));
        visited.insert(entry_point.get_id());
        let mut worst_distance = ep_dist;

        // Phase 3: graph-entangled insert seeding. Seed the frontier with
        // caller-supplied ids (graph out-edges of the inserting point) so
        // HNSW construction starts from known semantic neighbors rather
        // than having to walk the graph from the entry point. Each seed is
        // scored, pushed into candidates, and recorded as visited; filter
        // is not applied since insert-time search passes None anyway.
        if level == 0 && !seed_ids.is_empty() {
            let mut seeded: u64 = 0;
            for &seed_id in seed_ids {
                if !visited.insert(seed_id) {
                    continue;
                }
                let distance = match self.score_neighbor_distance_with_cache(
                    r,
                    prepared,
                    seed_id,
                    level,
                    query.get_data(),
                    Some(ordinal_cache),
                ) {
                    Ok(Some(d)) => d,
                    Ok(None) => continue,
                    Err(_) => continue,
                };
                candidates.push(Candidate {
                    id: seed_id,
                    distance,
                });
                results.push((seed_id, distance));
                if distance > worst_distance {
                    worst_distance = distance;
                }
                seeded += 1;
            }
            if let Some(labels) = metrics_labels {
                metrics::histogram!(
                    "helix_graph_seed_candidates_used",
                    "collection" => labels.collection.to_string(),
                    "vector" => labels.vector.to_string(),
                )
                .record(seeded as f64);
                if seeded > 0 {
                    metrics::counter!(
                        "helix_graph_seed_insertions_total",
                        "collection" => labels.collection.to_string(),
                        "vector" => labels.vector.to_string(),
                    )
                    .increment(1);
                }
            }
        }

        while let Some(curr_cand) = candidates.pop() {
            if results.len() >= ef && curr_cand.distance > worst_distance {
                break;
            }

            let block = self.get_neighbor_block(r, curr_cand.id, level)?;
            neighbor_blocks += 1;
            neighbor_edges_seen += block.ids.len();

            // Approximate pruning via neighbor codes (avoids full vector fetch)
            let use_approx = results.len() >= ef && level == 0 && block.has_approx_codes();
            let approx_limit = if use_approx {
                if let Some(prepared_neighbor) = neighbor_prepared.as_ref() {
                    let mut approx_ranked = block
                        .ids
                        .iter()
                        .enumerate()
                        .filter_map(|(index, &neighbor_id)| {
                            block.code_at(index).and_then(|code| {
                                score_encoded(prepared_neighbor, code).ok().flatten().map(
                                    |approx| (neighbor_id, self.distance_from_approximate(&approx)),
                                )
                            })
                        })
                        .collect::<Vec<_>>();
                    approx_ranked
                        .sort_by(|lhs, rhs| lhs.1.partial_cmp(&rhs.1).unwrap_or(Ordering::Equal));
                    let proportional = ((approx_ranked.len() as f64 * LEVEL0_APPROX_KEEP_FRACTION)
                        .ceil() as usize)
                        .max(self.config.m);
                    let fetch_limit = proportional.min(approx_ranked.len());
                    if fetch_limit > 0 {
                        Some(
                            approx_ranked
                                .into_iter()
                                .take(fetch_limit)
                                .map(|(neighbor_id, _)| neighbor_id)
                                .collect::<Vec<u128>>(),
                        )
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let neighbor_iter: &[u128] = match approx_limit.as_ref() {
                Some(ids) => {
                    approx_blocks += 1;
                    approx_candidates_kept += ids.len();
                    ids.as_slice()
                }
                None => &block.ids,
            };

            let mut prefetched =
                self.prefetch_neighbor_vectors(r, prepared, neighbor_iter, &visited, level);
            self.prefetch_sidecar_ordinals(r, neighbor_iter, &visited, ordinal_cache)?;

            for &neighbor_id in neighbor_iter {
                if !visited.insert(neighbor_id) {
                    continue;
                }
                // Deferred-repair deletes: a tombstoned id's row and links
                // are still present until this segment's next merge. Keep it
                // in the frontier as a traversal bridge (like a filtered-out
                // node) but never admit it to `results`.
                let tombstoned = self.is_delete_tombstoned(neighbor_id);

                let (distance, filter_passes) =
                    if let Some(neighbor) = prefetched.remove(&neighbor_id) {
                        (
                            self.distance_between(neighbor.get_data(), query.get_data())?,
                            !tombstoned && filter.is_none_or(|fs| fs.iter().all(|f| f(&neighbor))),
                        )
                    } else if tombstoned {
                        let Some(distance) = self.score_neighbor_distance_for_traversal(
                            r,
                            prepared,
                            neighbor_id,
                            level,
                            query.get_data(),
                        )?
                        else {
                            continue;
                        };
                        (distance, false)
                    } else if let Some(fs) = filter {
                        // Preserve the generic HNSW filter contract: callers
                        // receive the same full HVector they received before
                        // this optimization. Qdrant/CE id-only filters can use
                        // the no-filter branch when no payload filter is active.
                        let neighbor = match self.get_vector_with_ordinal_cache(
                            r,
                            neighbor_id,
                            level,
                            true,
                            ordinal_cache,
                        ) {
                            Ok(n) => n,
                            Err(VectorError::VectorNotFound(_)) => continue,
                            Err(e) => return Err(e),
                        };
                        (
                            self.scored_distance(
                                r,
                                prepared,
                                neighbor.get_id(),
                                neighbor.get_level(),
                                neighbor.get_data(),
                                query.get_data(),
                            )?,
                            fs.iter().all(|f| f(&neighbor)),
                        )
                    } else {
                        // Score directly from the mmap-borrowed slice when
                        // possible (HVEC, level 0). Falls back to full vector
                        // load + score for HVS8 / level > 0 / no-mmap. `None`
                        // means the vector row is missing.
                        let Some(distance) = self.score_neighbor_distance_with_cache(
                            r,
                            prepared,
                            neighbor_id,
                            level,
                            query.get_data(),
                            Some(ordinal_cache),
                        )?
                        else {
                            continue;
                        };
                        (distance, true)
                    };
                neighbor_edges_scored += 1;

                if results.len() < ef || distance < worst_distance {
                    candidates.push(Candidate {
                        id: neighbor_id,
                        distance,
                    });
                    if !filter_passes {
                        filtered_out += 1;
                        continue;
                    }
                    results.push((neighbor_id, distance));

                    if results.len() > ef {
                        // Evict the worst: find max distance, swap_remove it
                        let worst_idx = results
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1 .1.partial_cmp(&b.1 .1).unwrap_or(Ordering::Equal))
                            .map(|(idx, _)| idx)
                            .unwrap();
                        results.swap_remove(worst_idx);
                    }
                    worst_distance = results
                        .iter()
                        .map(|r| r.1)
                        .fold(f32::NEG_INFINITY, f32::max);
                }
            }
        }

        // Graph-entangled bridge: if the HNSW frontier dead-ended before
        // we collected `ef` results (common on filtered search at low
        // selectivity), expand via graph out-edges of visited nodes. No-op
        // when the flag is off, no graph handle is attached, or the
        // frontier still has candidates to explore.
        let _ = self.graph_bridge_level0::<F, fn(u128) -> bool>(
            r,
            query.get_data(),
            prepared,
            &mut visited,
            &mut candidates,
            &mut results,
            &mut worst_distance,
            ef,
            level,
            metrics_labels,
            filter,
            None,
        )?;

        // Hydrate final results: fetch full HVector data only for the
        // ef results we're keeping (not the ~ef×M neighbors we visited).
        let mut hvec_results: Vec<HVector> = Vec::with_capacity(results.len());
        for (id, dist) in &results {
            if self.is_delete_tombstoned(*id) {
                continue;
            }
            let mut v = match self.get_vector_with_ordinal_cache(r, *id, level, true, ordinal_cache)
            {
                Ok(vector) => vector,
                Err(VectorError::VectorNotFound(_)) => continue,
                Err(e) => return Err(e),
            };
            v.set_distance(*dist);
            hvec_results.push(v);
        }
        self.observe_hnsw_level_counts(
            metrics_labels,
            "hvector",
            level,
            ef,
            visited.len(),
            neighbor_blocks,
            neighbor_edges_seen,
            neighbor_edges_scored,
            approx_blocks,
            approx_candidates_kept,
            filtered_out,
            results.len(),
            hvec_results.len(),
        );

        Ok(BinaryHeap::from(hvec_results))
    }

    fn search_level_id_filter<'a, F>(
        &'a self,
        r: &AnyRead<'_>,
        query: &'a HVector,
        entry_point: &'a mut HVector,
        ef: usize,
        level: usize,
        filter: Option<&[F]>,
        arena: &'a Bump,
        prepared: &PreparedSpindleQuery,
        metrics_labels: Option<VectorSearchMetrics<'_>>,
        _seed_ids: &[u128],
        ordinal_cache: &mut SidecarOrdinalRequestCache,
    ) -> Result<BinaryHeap<HVector>, VectorError>
    where
        F: Fn(u128) -> bool,
    {
        let mut visited: HashSet<u128> = HashSet::with_capacity(ef * 2);
        let mut candidates: ArenaHeap<Candidate> = ArenaHeap::with_capacity(arena, ef.max(1));
        let mut results = bumpalo::collections::Vec::with_capacity_in(ef + 1, arena);
        let mut neighbor_blocks = 0usize;
        let mut neighbor_edges_seen = 0usize;
        let mut neighbor_edges_scored = 0usize;
        let mut approx_blocks = 0usize;
        let mut approx_candidates_kept = 0usize;
        let mut filtered_out = 0usize;
        let neighbor_prepared = if level == 0 {
            Self::neighbor_code_config(query.get_data().len())
                .and_then(|config| prepare_query(query.get_data(), &config).ok())
        } else {
            None
        };

        let ep_dist = self.scored_distance(
            r,
            prepared,
            entry_point.get_id(),
            entry_point.get_level(),
            entry_point.get_data(),
            query.get_data(),
        )?;
        candidates.push(Candidate {
            id: entry_point.get_id(),
            distance: ep_dist,
        });
        let entry_id = entry_point.get_id();
        if filter.is_none_or(|fs| fs.iter().all(|f| f(entry_id))) {
            results.push((entry_id, ep_dist));
        }
        visited.insert(entry_id);
        let mut worst_distance = if results.is_empty() {
            f32::INFINITY
        } else {
            ep_dist
        };

        while let Some(curr_cand) = candidates.pop() {
            if results.len() >= ef && curr_cand.distance > worst_distance {
                break;
            }

            let block = self.get_neighbor_block(r, curr_cand.id, level)?;
            neighbor_blocks += 1;
            neighbor_edges_seen += block.ids.len();

            let use_approx = results.len() >= ef && level == 0 && block.has_approx_codes();
            let approx_limit = if use_approx {
                if let Some(prepared_neighbor) = neighbor_prepared.as_ref() {
                    let mut approx_ranked = block
                        .ids
                        .iter()
                        .enumerate()
                        .filter_map(|(index, &neighbor_id)| {
                            block.code_at(index).and_then(|code| {
                                score_encoded(prepared_neighbor, code).ok().flatten().map(
                                    |approx| (neighbor_id, self.distance_from_approximate(&approx)),
                                )
                            })
                        })
                        .collect::<Vec<_>>();
                    approx_ranked
                        .sort_by(|lhs, rhs| lhs.1.partial_cmp(&rhs.1).unwrap_or(Ordering::Equal));
                    let proportional = ((approx_ranked.len() as f64 * LEVEL0_APPROX_KEEP_FRACTION)
                        .ceil() as usize)
                        .max(self.config.m);
                    let fetch_limit = proportional.min(approx_ranked.len());
                    if fetch_limit > 0 {
                        Some(
                            approx_ranked
                                .into_iter()
                                .take(fetch_limit)
                                .map(|(neighbor_id, _)| neighbor_id)
                                .collect::<Vec<u128>>(),
                        )
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let neighbor_iter: &[u128] = match approx_limit.as_ref() {
                Some(ids) => {
                    approx_blocks += 1;
                    approx_candidates_kept += ids.len();
                    ids.as_slice()
                }
                None => &block.ids,
            };

            let mut prefetched =
                self.prefetch_neighbor_vectors(r, prepared, neighbor_iter, &visited, level);
            self.prefetch_sidecar_ordinals(r, neighbor_iter, &visited, ordinal_cache)?;

            for &neighbor_id in neighbor_iter {
                if !visited.insert(neighbor_id) {
                    continue;
                }
                // Tombstoned ids stay traversable (graph bridge) but never
                // enter `results`; see `search_level`.
                let filter_passes = !self.is_delete_tombstoned(neighbor_id)
                    && filter.is_none_or(|fs| fs.iter().all(|f| f(neighbor_id)));

                let distance = if let Some(neighbor) = prefetched.remove(&neighbor_id) {
                    self.distance_between(neighbor.get_data(), query.get_data())?
                } else {
                    match self.score_neighbor_distance_for_traversal_with_cache(
                        r,
                        prepared,
                        neighbor_id,
                        level,
                        query.get_data(),
                        Some(ordinal_cache),
                    )? {
                        Some(d) => d,
                        None => continue,
                    }
                };
                neighbor_edges_scored += 1;

                if results.len() < ef || distance < worst_distance {
                    // ACORN-lite: a filtered-out neighbor is still useful as a
                    // graph bridge in HNSW, but it must not enter the result
                    // heap. Previously we skipped these ids before enqueueing,
                    // so selective repo/branch filters could disconnect the
                    // traversal and under-return even when matching points were
                    // reachable behind nonmatching neighbors.
                    candidates.push(Candidate {
                        id: neighbor_id,
                        distance,
                    });
                    if !filter_passes {
                        filtered_out += 1;
                        continue;
                    }

                    results.push((neighbor_id, distance));

                    if results.len() > ef {
                        let worst_idx = results
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1 .1.partial_cmp(&b.1 .1).unwrap_or(Ordering::Equal))
                            .map(|(idx, _)| idx)
                            .unwrap();
                        results.swap_remove(worst_idx);
                    }
                    worst_distance = if results.is_empty() {
                        f32::INFINITY
                    } else {
                        results
                            .iter()
                            .map(|r| r.1)
                            .fold(f32::NEG_INFINITY, f32::max)
                    };
                }
            }
        }

        // Graph-entangled bridge for id-filtered HNSW search. Same shape as
        // the `search_level` bridge — on frontier dead-end, consult graph
        // adjacency of visited nodes. The id-filtered variant uses the
        // caller's id-only predicate so we don't pay a full vector fetch
        // per graph neighbor just to filter it out.
        let _ = self.graph_bridge_level0::<fn(&HVector) -> bool, F>(
            r,
            query.get_data(),
            prepared,
            &mut visited,
            &mut candidates,
            &mut results,
            &mut worst_distance,
            ef,
            level,
            metrics_labels,
            None,
            filter,
        )?;

        let mut hvec_results: Vec<HVector> = Vec::with_capacity(results.len());
        for (id, dist) in &results {
            if self.is_delete_tombstoned(*id) {
                continue;
            }
            let mut v = match self.get_vector_with_ordinal_cache(r, *id, level, true, ordinal_cache)
            {
                Ok(vector) => vector,
                Err(VectorError::VectorNotFound(_)) => continue,
                Err(e) => return Err(e),
            };
            v.set_distance(*dist);
            hvec_results.push(v);
        }
        self.observe_hnsw_level_counts(
            metrics_labels,
            "id_filter",
            level,
            ef,
            visited.len(),
            neighbor_blocks,
            neighbor_edges_seen,
            neighbor_edges_scored,
            approx_blocks,
            approx_candidates_kept,
            filtered_out,
            results.len(),
            hvec_results.len(),
        );

        Ok(BinaryHeap::from(hvec_results))
    }

    /// Compute effective ef for adaptive search. When a filter is highly selective
    /// (few candidates pass), we increase ef so the HNSW traversal explores more
    /// of the graph, compensating for filtered-out candidates.
    ///
    /// `selectivity` is the fraction of total points that pass the filter (0.0..=1.0).
    /// A selectivity of 0.01 means only 1% of points pass, requiring higher ef.
    /// The multiplier is clamped to [1.0, 8.0] so ef never decreases and never
    /// exceeds 8x the base value.
    #[inline]
    fn compute_adaptive_ef(&self, base_ef: usize, selectivity: Option<f32>) -> usize {
        if !self.config.adaptive_ef_enabled {
            return base_ef;
        }
        match selectivity {
            Some(sel) if sel < 1.0 => {
                let multiplier = (1.0_f32 / sel.max(0.01)).sqrt().clamp(1.0, 8.0);
                let adaptive = (base_ef as f32 * multiplier) as usize;
                adaptive.max(base_ef) // never decrease below base
            }
            _ => base_ef,
        }
    }

    /// Compute how many approximate candidates to keep for exact/secondary
    /// reranking. The configured Spindle oversampling remains the quality cap;
    /// this runtime policy avoids paying the worst-case multiplier for large
    /// unfiltered top-k requests while preserving the old default for small k
    /// and boosting selective filters back toward the cap.
    #[inline]
    fn compute_adaptive_rerank_oversampling(&self, k: usize, selectivity: Option<f32>) -> usize {
        if !(self.spindle.is_enabled() && self.spindle.rescore) {
            return 1;
        }

        let configured = self.spindle.oversampling();
        if configured <= 1 || k == 0 {
            return configured.max(1);
        }

        let base = match k {
            0..=50 => configured,
            51..=200 => configured.min(8),
            201..=1000 => configured.min(4),
            _ => configured.min(2),
        };

        match selectivity {
            Some(sel) if sel < 1.0 => {
                let boost = (1.0_f32 / sel.max(0.01)).sqrt().clamp(1.0, 8.0);
                ((base as f32 * boost).ceil() as usize).clamp(base, configured)
            }
            _ => base,
        }
    }

    #[inline]
    fn rerank_target_k(&self, k: usize, selectivity: Option<f32>) -> usize {
        k.saturating_mul(self.compute_adaptive_rerank_oversampling(k, selectivity))
            .max(k)
    }

    /// Search with an optional selectivity hint for adaptive ef adjustment.
    ///
    /// This wraps the same search logic as the HNSW trait `search()` but allows
    /// the caller to provide filter selectivity information so the ef parameter
    /// can be increased for highly selective filters.
    ///
    /// `selectivity_hint`: fraction of points expected to pass the filter (0.0..1.0).
    /// `None` or `Some(1.0)` means no adaptive adjustment (unfiltered search).
    pub fn search_with_selectivity<F>(
        &self,
        r: &AnyRead<'_>,
        query: &[f32],
        k: usize,
        filter: Option<&[F]>,
        should_trickle: bool,
        selectivity_hint: Option<f32>,
    ) -> Result<Vec<HVector>, VectorError>
    where
        F: Fn(&HVector) -> bool,
    {
        self.search_with_selectivity_ef(r, query, k, filter, should_trickle, selectivity_hint, None)
    }

    /// Like `search_with_selectivity` but accepts an optional per-request ef override.
    /// When `ef_override` is `Some(n)`, it replaces `self.config.ef` as the base ef.
    pub fn search_with_selectivity_ef<F>(
        &self,
        r: &AnyRead<'_>,
        query: &[f32],
        k: usize,
        filter: Option<&[F]>,
        should_trickle: bool,
        selectivity_hint: Option<f32>,
        ef_override: Option<usize>,
    ) -> Result<Vec<HVector>, VectorError>
    where
        F: Fn(&HVector) -> bool,
    {
        self.search_with_selectivity_ef_observed(
            r,
            query,
            k,
            filter,
            should_trickle,
            selectivity_hint,
            ef_override,
            None,
        )
    }

    /// Like `search_with_selectivity_ef`, with optional metric labels supplied
    /// by the named-vector layer. Keeping labels out of `VectorCore` itself
    /// avoids baking tenant names into every physical segment core.
    pub fn search_with_selectivity_ef_observed<F>(
        &self,
        r: &AnyRead<'_>,
        query: &[f32],
        k: usize,
        filter: Option<&[F]>,
        should_trickle: bool,
        selectivity_hint: Option<f32>,
        ef_override: Option<usize>,
        metrics_labels: Option<VectorSearchMetrics<'_>>,
    ) -> Result<Vec<HVector>, VectorError>
    where
        F: Fn(&HVector) -> bool,
    {
        match self.index_mode(r)? {
            IndexMode::Flat => {
                let flat_start = std::time::Instant::now();
                let results = self.search_flat(r, query, k, filter)?;
                if let Some(labels) = metrics_labels {
                    self.observe_flat_search(
                        labels,
                        "hvector",
                        self.rerank_target_k(k, selectivity_hint),
                        results.len(),
                        flat_start.elapsed(),
                    );
                }
                return Ok(results);
            }
            IndexMode::Ivf => {
                let ivf_start = std::time::Instant::now();
                let results = self.search_ivf(r, query, k, filter, selectivity_hint, None)?;
                if let Some(labels) = metrics_labels {
                    self.observe_flat_search(
                        labels,
                        "ivf",
                        self.rerank_target_k(k, selectivity_hint),
                        results.len(),
                        ivf_start.elapsed(),
                    );
                }
                return Ok(results);
            }
            // Fall through to the HNSW graph search below.
            IndexMode::Hnsw => {}
        }
        let raw_query = query.to_vec();
        let projected_query = project_for_search(&raw_query, &self.spindle)?;
        let query = HVector::from_slice(0, 0, projected_query);
        let prepared = prepare_query(&raw_query, &self.spindle)?;
        let arena = Bump::new();
        let mut ordinal_cache = SidecarOrdinalRequestCache::new();
        let target_k = self.rerank_target_k(k, selectivity_hint);

        let t0 = std::time::Instant::now();
        let mut entry_point = self.get_entry_point(r)?;
        self.ensure_navigation_cache(r, &entry_point);
        let t_nav = t0.elapsed();

        let base_ef = ef_override.unwrap_or(self.config.ef).max(target_k);
        // When the caller provides an explicit ef_override AND a selectivity
        // hint, treat the override as a hard cap so adaptive inflation cannot
        // push ef to pathological levels (e.g. 3200) for filtered queries.
        // We cap at the raw override value (not base_ef which includes the
        // max(target_k) floor) because callers like the ReFRAG gate
        // intentionally choose ef < k for speed over recall on filtered paths.
        //
        // For *unfiltered* queries we must not apply this cap: ef_override is
        // commonly < target_k (e.g. ReFRAG), and chopping below target_k
        // would silently collapse recall — the search would return < k
        // results because the candidate pool is smaller than k.
        let effective_ef = match (ef_override, selectivity_hint) {
            (Some(cap), Some(_)) => self.compute_adaptive_ef(base_ef, selectivity_hint).min(cap),
            _ => self.compute_adaptive_ef(base_ef, selectivity_hint),
        };
        let curr_level = entry_point.get_level();

        let t_upper_start = std::time::Instant::now();
        for level in (1..=curr_level).rev() {
            let mut nearest = self.search_level(
                r,
                &query,
                &mut entry_point,
                1,
                level,
                match should_trickle {
                    true => filter,
                    false => None,
                },
                &arena,
                &prepared,
                None,
                &[],
                &mut ordinal_cache,
            )?;
            if let Some(closest) = nearest.pop() {
                entry_point = closest;
            }
        }
        let t_upper = t_upper_start.elapsed();

        let t_l0_start = std::time::Instant::now();
        let mut candidates = self.search_level(
            r,
            &query,
            &mut entry_point,
            effective_ef,
            0,
            match should_trickle {
                true => filter,
                false => None,
            },
            &arena,
            &prepared,
            metrics_labels,
            &[],
            &mut ordinal_cache,
        )?;
        let t_l0 = t_l0_start.elapsed();
        let total_elapsed = t0.elapsed();
        if let Some(labels) = metrics_labels {
            let segment_count = labels.segment_count.to_string();
            let segment_index = labels.segment_index.to_string();
            let ef = effective_ef.to_string();
            let filtered = if labels.filtered { "true" } else { "false" }.to_string();
            let sidecar_format = self.sidecar_format_label().to_string();
            let record_stage = |stage: &'static str, elapsed: std::time::Duration| {
                metrics::histogram!(
                    "helix_vector_core_search_stage_ms",
                    "collection" => labels.collection.to_string(),
                    "vector" => labels.vector.to_string(),
                    "stage" => stage.to_string(),
                    "segment_count" => segment_count.clone(),
                    "segment_index" => segment_index.clone(),
                    "ef" => ef.clone(),
                    "filtered" => filtered.clone(),
                    "sidecar_format" => sidecar_format.clone(),
                )
                .record(elapsed.as_secs_f64() * 1000.0);
            };
            record_stage("nav", t_nav);
            record_stage("upper", t_upper);
            record_stage("level0", t_l0);
            record_stage("total", total_elapsed);
            if total_elapsed.as_millis() > 2 {
                metrics::counter!(
                    "helix_vector_core_search_slow_total",
                    "collection" => labels.collection.to_string(),
                    "vector" => labels.vector.to_string(),
                    "segment_count" => segment_count,
                    "segment_index" => segment_index,
                    "ef" => ef,
                    "filtered" => filtered,
                    "sidecar_format" => sidecar_format,
                )
                .increment(1);
            }
        }

        let mut results = candidates.to_vec_with_filter(target_k, filter);
        if self.spindle.is_enabled() && self.spindle.rescore {
            results = self.rescore_results(r, &raw_query, results)?;
        }
        results.truncate(k);
        Ok(results)
    }

    pub fn search_with_id_filter_ef_observed<F>(
        &self,
        r: &AnyRead<'_>,
        query: &[f32],
        k: usize,
        filter: Option<&[F]>,
        should_trickle: bool,
        selectivity_hint: Option<f32>,
        ef_override: Option<usize>,
        metrics_labels: Option<VectorSearchMetrics<'_>>,
    ) -> Result<Vec<HVector>, VectorError>
    where
        F: Fn(u128) -> bool,
    {
        if !self.has_index(r)? {
            let flat_start = std::time::Instant::now();
            let raw_query = query.to_vec();
            let projected_query = project_for_search(&raw_query, &self.spindle)?;
            let query = HVector::from_slice(0, 0, projected_query);
            let prepared = prepare_query(&raw_query, &self.spindle)?;
            let target_k = self.rerank_target_k(k, selectivity_hint);
            let mut results = self.search_flat_id_filter_direct(
                r,
                query.get_data(),
                &prepared,
                target_k,
                filter,
            )?;
            if self.spindle.is_enabled() && self.spindle.rescore {
                results = self.rescore_results(r, &raw_query, results)?;
            }
            results.truncate(k);
            if let Some(labels) = metrics_labels {
                self.observe_flat_search(
                    labels,
                    "id_filter",
                    target_k,
                    results.len(),
                    flat_start.elapsed(),
                );
            }
            return Ok(results);
        }
        let raw_query = query.to_vec();
        let projected_query = project_for_search(&raw_query, &self.spindle)?;
        let query = HVector::from_slice(0, 0, projected_query);
        let prepared = prepare_query(&raw_query, &self.spindle)?;
        let arena = Bump::new();
        let mut ordinal_cache = SidecarOrdinalRequestCache::new();
        let target_k = self.rerank_target_k(k, selectivity_hint);

        let t0 = std::time::Instant::now();
        let mut entry_point = self.get_entry_point(r)?;
        self.ensure_navigation_cache(r, &entry_point);
        let t_nav = t0.elapsed();

        let base_ef = ef_override.unwrap_or(self.config.ef).max(target_k);
        let effective_ef = match (ef_override, selectivity_hint) {
            (Some(cap), Some(_)) => self.compute_adaptive_ef(base_ef, selectivity_hint).min(cap),
            _ => self.compute_adaptive_ef(base_ef, selectivity_hint),
        };
        let curr_level = entry_point.get_level();

        let t_upper_start = std::time::Instant::now();
        for level in (1..=curr_level).rev() {
            let mut nearest = self.search_level_id_filter(
                r,
                &query,
                &mut entry_point,
                1,
                level,
                match should_trickle {
                    true => filter,
                    false => None,
                },
                &arena,
                &prepared,
                None,
                &[],
                &mut ordinal_cache,
            )?;
            if let Some(closest) = nearest.pop() {
                entry_point = closest;
            }
        }
        let t_upper = t_upper_start.elapsed();

        let t_l0_start = std::time::Instant::now();
        let mut candidates = self.search_level_id_filter(
            r,
            &query,
            &mut entry_point,
            effective_ef,
            0,
            match should_trickle {
                true => filter,
                false => None,
            },
            &arena,
            &prepared,
            metrics_labels,
            &[],
            &mut ordinal_cache,
        )?;
        let t_l0 = t_l0_start.elapsed();
        let total_elapsed = t0.elapsed();
        if let Some(labels) = metrics_labels {
            let segment_count = labels.segment_count.to_string();
            let segment_index = labels.segment_index.to_string();
            let ef = effective_ef.to_string();
            let filtered = if labels.filtered { "true" } else { "false" }.to_string();
            let sidecar_format = self.sidecar_format_label().to_string();
            let record_stage = |stage: &'static str, elapsed: std::time::Duration| {
                metrics::histogram!(
                    "helix_vector_core_search_stage_ms",
                    "collection" => labels.collection.to_string(),
                    "vector" => labels.vector.to_string(),
                    "stage" => stage.to_string(),
                    "segment_count" => segment_count.clone(),
                    "segment_index" => segment_index.clone(),
                    "ef" => ef.clone(),
                    "filtered" => filtered.clone(),
                    "sidecar_format" => sidecar_format.clone(),
                )
                .record(elapsed.as_secs_f64() * 1000.0);
            };
            record_stage("nav", t_nav);
            record_stage("upper", t_upper);
            record_stage("level0", t_l0);
            record_stage("total", total_elapsed);
            if total_elapsed.as_millis() > 2 {
                metrics::counter!(
                    "helix_vector_core_search_slow_total",
                    "collection" => labels.collection.to_string(),
                    "vector" => labels.vector.to_string(),
                    "segment_count" => segment_count,
                    "segment_index" => segment_index,
                    "ef" => ef,
                    "filtered" => filtered,
                    "sidecar_format" => sidecar_format,
                )
                .increment(1);
            }
        }

        let mut results = if let Some(fs) = filter {
            let mut out = Vec::with_capacity(target_k);
            while out.len() < target_k {
                let Some(candidate) = candidates.pop() else {
                    break;
                };
                if fs.iter().all(|f| f(candidate.id)) {
                    out.push(candidate);
                }
            }
            out
        } else {
            candidates.to_vec(target_k)
        };
        if self.spindle.is_enabled() && self.spindle.rescore {
            results = self.rescore_results(r, &raw_query, results)?;
        }
        results.truncate(k);
        Ok(results)
    }
}

impl HNSW for VectorCore {
    #[inline(always)]
    fn get_vector(
        &self,
        read_context: &AnyRead<'_>,
        id: u128,
        level: usize,
        with_data: bool,
    ) -> Result<HVector, VectorError> {
        if with_data {
            // Check in-memory caches first (upper-level nav cache, then general cache).
            if !Self::can_use_global_vector_cache(read_context) {
                metrics::counter!("helix_vector_cache_misses_total",
                    "cache" => "vector",
                    "tier" => if level > 0 { "nav" } else { "level0" })
                .increment(1);
            } else if let Some(shared) = self.shared_caches.as_ref() {
                let key = super::shared_cache::CacheKey::new(self.cache_namespace, id, level);
                if level > 0 {
                    if let Some(data) = shared.nav_vector.get(&key) {
                        metrics::counter!("helix_vector_cache_hits_total",
                            "cache" => "vector", "tier" => "nav")
                        .increment(1);
                        return Ok(HVector::from_slice(id, level, data));
                    }
                }
                if let Some(data) = shared.vector.get(&key) {
                    metrics::counter!("helix_vector_cache_hits_total",
                        "cache" => "vector", "tier" => "level0")
                    .increment(1);
                    return Ok(HVector::from_slice(id, level, data));
                }
                metrics::counter!("helix_vector_cache_misses_total",
                    "cache" => "vector",
                    "tier" => if level > 0 { "nav" } else { "level0" })
                .increment(1);
            } else {
                if level > 0 {
                    if let Ok(mut cache) = self.nav_vector_cache.lock() {
                        if let Some(data) = cache.get(&(id, level)) {
                            metrics::counter!("helix_vector_cache_hits_total",
                                "cache" => "vector", "tier" => "nav")
                            .increment(1);
                            return Ok(HVector::from_slice(id, level, data.clone()));
                        }
                    }
                }
                if let Ok(mut cache) = self.vector_cache.lock() {
                    if let Some(data) = cache.get(&(id, level)) {
                        metrics::counter!("helix_vector_cache_hits_total",
                            "cache" => "vector", "tier" => "level0")
                        .increment(1);
                        return Ok(HVector::from_slice(id, level, data.clone()));
                    }
                }
                metrics::counter!("helix_vector_cache_misses_total",
                    "cache" => "vector",
                    "tier" => if level > 0 { "nav" } else { "level0" })
                .increment(1);
            }

            // Mmap fast path: O(1) pointer arithmetic instead of B-tree lookup.
            // Only for level 0 (where all vector data lives).
            if level == 0
                && (self.has_mmap_sidecar() || self.should_try_lsm_sidecar())
                && self.ensure_lsm_mmap_sidecar()?
            {
                if let Some(ordinal) = self.sidecar_ordinal(read_context, id)? {
                    if let Some(vector_data) = self.sidecar_vec(ordinal)? {
                        self.cache_vector_for_read(read_context, id, level, &vector_data);
                        return Ok(HVector::from_slice(id, level, vector_data));
                    }
                }
            }
        }

        // Fallback: LMDB B-tree lookup (for level > 0 or when mmap not available).
        let key = Self::vector_key(id, level);
        let stored = self
            .backend
            .get_with(
                read_context,
                self.seg_ns(SegmentDb::Vectors),
                key.as_ref(),
                |opt| opt.map(|bytes| bytes.to_vec()),
            )
            .map_err(|error| VectorError::VectorCoreError(error.to_string()))?;
        match stored {
            Some(bytes) => {
                let vector = if with_data {
                    let decoded = if Self::is_vector_marker(&bytes) {
                        self.resolve_marker_vector(read_context, id, level)?
                    } else {
                        decode_vector(&bytes)?
                    };
                    self.cache_vector_for_read(read_context, id, level, &decoded);
                    HVector::from_slice(id, level, decoded)
                } else {
                    HVector::from_slice(id, level, vec![])
                };
                Ok(vector)
            }
            None if level > 0 => self.get_vector(read_context, id, 0, with_data),
            None => Err(VectorError::VectorNotFound(id.to_string())),
        }
    }

    fn search<F>(
        &self,
        r: &AnyRead<'_>,
        query: &[f32],
        k: usize,
        filter: Option<&[F]>,
        should_trickle: bool,
    ) -> Result<Vec<HVector>, VectorError>
    where
        F: Fn(&HVector) -> bool,
    {
        if !self.has_index(r)? {
            return self.search_flat(r, query, k, filter);
        }

        let raw_query = query.to_vec();
        let projected_query = project_for_search(&raw_query, &self.spindle)?;
        let query = HVector::from_slice(0, 0, projected_query);
        let prepared = prepare_query(&raw_query, &self.spindle)?;
        let arena = Bump::new();
        let mut ordinal_cache = SidecarOrdinalRequestCache::new();
        let target_k = self.rerank_target_k(k, None);

        let mut entry_point = self.get_entry_point(r)?;
        self.ensure_navigation_cache(r, &entry_point);

        let ef = self.config.ef.max(target_k);
        let curr_level = entry_point.get_level();

        for level in (1..=curr_level).rev() {
            let mut nearest = self.search_level(
                r,
                &query,
                &mut entry_point,
                1,
                level,
                match should_trickle {
                    true => filter,
                    false => None,
                },
                &arena,
                &prepared,
                None,
                &[],
                &mut ordinal_cache,
            )?;
            if let Some(closest) = nearest.pop() {
                entry_point = closest;
            }
        }

        let mut candidates = self.search_level(
            r,
            &query,
            &mut entry_point,
            ef,
            0,
            match should_trickle {
                true => filter,
                false => None,
            },
            &arena,
            &prepared,
            None,
            &[],
            &mut ordinal_cache,
        )?;

        let mut results = candidates.to_vec_with_filter(target_k, filter);
        if self.spindle.is_enabled() && self.spindle.rescore {
            results = self.rescore_results(r, &raw_query, results)?;
        }
        results.truncate(k);
        Ok(results)
    }

    fn insert<F>(
        &self,
        txn: &mut RwTxn,
        data: &[f32],
        nid: Option<u128>,
        fields: Option<HashMap<String, Value>>,
    ) -> Result<HVector, VectorError>
    where
        F: Fn(&HVector) -> bool,
    {
        ensure_finite_vector(data)?;
        let id = nid.unwrap_or(uuid::Uuid::new_v4().as_u128());
        let new_level = self.get_new_level();
        let projected = project_for_search(data, &self.spindle)?;

        let mut query = HVector::from_slice(id, 0, projected);
        self.put_raw_vector(txn, query.get_id(), 0, data)?;

        query.level = new_level;
        if new_level > 0 {
            self.put_raw_vector(txn, query.get_id(), new_level, data)?;
        }

        let entry_point = {
            let rd = self.backend.read_borrowed(&*txn);
            self.get_entry_point(&rd)
        };
        let entry_point = match entry_point {
            Ok(ep) => ep,
            Err(_) => {
                self.maybe_put_vector_data(txn, query.get_id(), fields, data)?;
                self.set_entry_point(txn, &query)?;
                query.set_distance(0.0);
                return Ok(query);
            }
        };

        let l = entry_point.get_level();
        let mut curr_ep = entry_point;
        let no_prepared = PreparedSpindleQuery::None;
        let arena = Bump::new();
        let mut ordinal_cache = SidecarOrdinalRequestCache::new();
        {
            let rd = self.backend.read_borrowed(&*txn);
            for level in (new_level + 1..=l).rev() {
                let nearest = self.search_level::<F>(
                    &rd,
                    &query,
                    &mut curr_ep,
                    1,
                    level,
                    None,
                    &arena,
                    &no_prepared,
                    None,
                    &[],
                    &mut ordinal_cache,
                )?;
                // Empty only when the sole candidate was a tombstoned entry
                // point (excluded from results); keep descending from it.
                if let Some(closest) = nearest.peek() {
                    curr_ep = closest.clone();
                }
            }
        }

        // Phase 3: graph-entangled insert seeding. If the point being
        // inserted already has graph out-edges, feed those neighbor ids as
        // bonus seeds into the level-0 HNSW walk. This is common in CE:
        // symbols are ingested as graph nodes with callers/callees edges
        // before their embeddings are computed, so the graph already
        // carries a strong semantic prior by the time we build the HNSW
        // connections. No-op when the flag is off or no graph DB is wired.
        let graph_seeds: Vec<u128> =
            if super::named_vectors::NamedVectorManager::graph_entangled_enabled() {
                let rd = self.backend.read_borrowed(&*txn);
                self.collect_graph_seed_ids(&rd, query.get_id())
            } else {
                Vec::new()
            };

        for level in (0..=l.min(new_level)).rev() {
            let seeds: &[u128] = if level == 0 { &graph_seeds } else { &[] };
            let (next_ep, neighbors) = {
                let rd = self.backend.read_borrowed(&*txn);
                let nearest = self.search_level::<F>(
                    &rd,
                    &query,
                    &mut curr_ep,
                    self.config.ef_construct,
                    level,
                    None,
                    &arena,
                    &no_prepared,
                    None,
                    seeds,
                    &mut ordinal_cache,
                )?;
                // Capture the closest candidate before `nearest` is consumed by
                // `select_neighbors` (preserves the original entry-point update).
                let next_ep = nearest.peek().cloned().unwrap_or_else(|| curr_ep.clone());
                let neighbors =
                    self.select_neighbors::<F>(&rd, &query, nearest, level, true, None)?;
                (next_ep, neighbors)
            };

            curr_ep = next_ep;

            self.set_neighbours(txn, query.get_id(), &neighbors, level)?;

            let max_conns = if level == 0 {
                self.config.m_max_0
            } else {
                self.config.m
            };
            for e in neighbors {
                let id = e.get_id();
                // Overflow prune of `e`'s own list: rank e's existing links by
                // distance to `e` (not to the inserted point) and keep the
                // closest `max_conns`. Only e's list is rewritten — dropped
                // nodes keep their link to `e` (HNSW links are directed after
                // pruning), matching the bulk builder.
                let pruned: Option<Vec<u128>> = {
                    let rd = self.backend.read_borrowed(&*txn);
                    let conn_ids = self.get_neighbor_ids(&rd, id, level)?;
                    if conn_ids.len() > max_conns {
                        let mut scored: Vec<(u128, f32)> = Vec::with_capacity(conn_ids.len());
                        for neighbor_id in conn_ids {
                            // Level-0 row always exists and carries the same
                            // data; upper-level rows may not be materialized.
                            if let Ok(conn) = self.get_vector(&rd, neighbor_id, 0, true) {
                                let distance =
                                    self.distance_between(conn.get_data(), e.get_data())?;
                                scored.push((neighbor_id, distance));
                            }
                        }
                        scored.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1));
                        scored.truncate(max_conns);
                        Some(
                            scored
                                .into_iter()
                                .map(|(neighbor_id, _)| neighbor_id)
                                .collect(),
                        )
                    } else {
                        None
                    }
                };
                if let Some(pruned) = pruned {
                    self.put_neighbor_ids(txn, id, level, &pruned)?;
                }
            }
        }

        if new_level > l {
            self.set_entry_point(txn, &query)?;
        }

        self.maybe_put_vector_data(txn, query.get_id(), fields, data)?;
        Ok(query)
    }

    fn get_all_vectors(
        &self,
        r: &AnyRead<'_>,
        level: Option<usize>,
    ) -> Result<Vec<HVector>, VectorError> {
        let mut vectors = Vec::new();
        // Collect rows first: non-marker values decode in-place (no DB access);
        // marker rows defer to `resolve_marker_vector` below, which re-reads the
        // DB through the seam and cannot run inside the scan closure.
        // `decoded` is `Some(vec)` for ready rows, `None` for markers.
        let mut rows: Vec<(u128, usize, Option<Vec<f32>>)> = Vec::new();
        let mut scan_err: Option<VectorError> = None;
        self.backend
            .scan(
                r,
                self.seg_ns(SegmentDb::Vectors),
                KeyRange::prefix(VECTOR_PREFIX),
                |key, value| {
                    if key.len() < VECTOR_PREFIX.len() + 16 + std::mem::size_of::<usize>() {
                        scan_err = Some(VectorError::InvalidVectorData);
                        return false;
                    }
                    let id_start = VECTOR_PREFIX.len();
                    let id_end = id_start + 16;
                    let level_end = id_end + std::mem::size_of::<usize>();
                    let Ok(id_arr) = key[id_start..id_end].try_into() else {
                        scan_err = Some(VectorError::InvalidVectorData);
                        return false;
                    };
                    let Ok(raw_level) = key[id_end..level_end].try_into() else {
                        scan_err = Some(VectorError::InvalidVectorData);
                        return false;
                    };
                    let id = u128::from_be_bytes(id_arr);
                    let vector_level = usize::from_be_bytes(raw_level);
                    if !level.map_or(true, |l| vector_level == l) {
                        return true;
                    }
                    if Self::is_vector_marker(value) {
                        rows.push((id, vector_level, None));
                    } else {
                        match decode_vector(value) {
                            Ok(decoded) => rows.push((id, vector_level, Some(decoded))),
                            Err(e) => {
                                scan_err = Some(e);
                                return false;
                            }
                        }
                    }
                    true
                },
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        if let Some(err) = scan_err {
            return Err(err);
        }
        for (id, vector_level, decoded) in rows {
            let decoded = match decoded {
                Some(decoded) => decoded,
                None => match self.resolve_marker_vector(r, id, vector_level) {
                    Ok(decoded) => decoded,
                    Err(VectorError::VectorCoreError(message))
                        if Self::is_externalized_vector_unavailable(&message) =>
                    {
                        self.mark_externalized_marker_repair_needed();
                        metrics::counter!(
                            "helix_externalized_marker_unavailable_total",
                            "level" => vector_level.to_string()
                        )
                        .increment(1);
                        tracing::warn!(
                            vector_id = %id,
                            level = vector_level,
                            error = %message,
                            "skipping unavailable externalized vector marker"
                        );
                        continue;
                    }
                    Err(VectorError::VectorNotFound(_)) => continue,
                    Err(err) => return Err(err),
                },
            };
            vectors.push(HVector::from_slice(id, vector_level, decoded));
        }
        Ok(vectors)
    }

    fn load<F>(&self, txn: &mut RwTxn, data: Vec<&[f32]>) -> Result<(), VectorError>
    where
        F: Fn(&HVector) -> bool,
    {
        for v in data.iter() {
            let _ = self.insert::<F>(txn, v, None, None);
        }

        // NOTE: need to txn.commit() outside of call

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helix_engine::vector_core::mmap_vectors::MmapTurboQuantStore;
    use crate::helix_engine::vector_core::spindle::{SpindleConfig, SpindleMode};
    use rand::SeedableRng;
    use tempfile::TempDir;

    type VF = fn(&HVector) -> bool;

    fn prefetch_test_core(backend: Arc<AnyBackend>, name: &str) -> VectorCore {
        VectorCore::new_named_lsm_with_dir(
            name,
            HNSWConfig::new(Some(8), Some(16), Some(32)),
            DistanceMetric::Euclid,
            SpindleConfig {
                mode: SpindleMode::None,
                ..SpindleConfig::default()
            },
            None,
            2,
            backend,
        )
        .unwrap()
    }

    #[test]
    fn hnsw_prefetch_matches_scalar_pending_overlay_and_bounds_batches() {
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("hnsw-prefetch-{}", uuid::Uuid::new_v4())).unwrap(),
        ));
        let core = prefetch_test_core(Arc::clone(&backend), "prefetch_overlay");
        let mut write = backend.begin_write().unwrap();
        for id in 1..=130 {
            let data = encode_vector(&[id as f32, 1.0], &core.spindle).unwrap();
            backend
                .put(
                    &mut write,
                    core.seg_ns(SegmentDb::Vectors),
                    &VectorCore::vector_key(id, 0),
                    &data,
                )
                .unwrap();
        }
        backend.commit(write).unwrap();
        let committed = backend.begin_read().unwrap();
        let mut write = backend.begin_write().unwrap();
        let replacement = encode_vector(&[999.0, 2.0], &core.spindle).unwrap();
        backend
            .put(
                &mut write,
                core.seg_ns(SegmentDb::Vectors),
                &VectorCore::vector_key(2, 0),
                &replacement,
            )
            .unwrap();
        backend
            .delete(
                &mut write,
                core.seg_ns(SegmentDb::Vectors),
                &VectorCore::vector_key(3, 0),
            )
            .unwrap();
        backend
            .put(
                &mut write,
                core.seg_ns(SegmentDb::Vectors),
                &VectorCore::vector_key(131, 0),
                &[1, 2, 3],
            )
            .unwrap();
        let overlay = backend
            .lsm_read_with_pending(write.lsm_pending().unwrap().clone())
            .unwrap();
        let neighbors: Vec<u128> = (1..=131).chain([2, 1]).collect();
        let visited = HashSet::from([1]);
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let rows = metrics::with_local_recorder(&recorder, || {
            core.prefetch_neighbor_vectors(
                &overlay,
                &PreparedSpindleQuery::None,
                &neighbors,
                &visited,
                0,
            )
        });
        assert_eq!(rows.len(), 128);
        assert_eq!(rows.get(&2).unwrap().get_data(), &[999.0, 2.0]);
        assert!(!rows.contains_key(&1));
        assert!(!rows.contains_key(&3));
        assert!(!rows.contains_key(&131));
        for (&id, row) in &rows {
            let scalar = core.get_vector(&overlay, id, 0, true).unwrap();
            assert_eq!(row.get_data(), scalar.get_data());
            assert_eq!(row.get_level(), scalar.get_level());
        }
        assert!(matches!(
            core.get_vector(&overlay, 3, 0, true),
            Err(VectorError::VectorNotFound(_))
        ));
        assert!(core.get_vector(&overlay, 131, 0, true).is_err());
        let metrics = handle.render();
        assert!(
            metrics.contains("helix_hnsw_vector_prefetch_batches_total 3"),
            "{metrics}"
        );
        assert!(
            metrics.contains("helix_hnsw_vector_prefetch_rows_total 130"),
            "{metrics}"
        );
        let isolated = prefetch_test_core(Arc::clone(&backend), "prefetch_overlay");
        assert_eq!(
            isolated
                .get_vector(&committed, 2, 0, true)
                .unwrap()
                .get_data(),
            &[2.0, 1.0]
        );
    }

    #[test]
    fn hnsw_prefetch_preserves_reader_snapshot_and_shared_cache_isolation() {
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use crate::helix_engine::storage_core::backend_lsm_reader::LsmReader;
        use object_store::memory::InMemory;
        use object_store::ObjectStore;
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = format!("hnsw-prefetch-reader-{}", uuid::Uuid::new_v4());
        let writer = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_with_store(&path, Arc::clone(&store)).unwrap(),
        ));
        let core = prefetch_test_core(Arc::clone(&writer), "prefetch_reader");
        let mut write = writer.begin_write().unwrap();
        for id in [1, 2] {
            let data = encode_vector(&[id as f32, 0.0], &core.spindle).unwrap();
            writer
                .put(
                    &mut write,
                    core.seg_ns(SegmentDb::Vectors),
                    &VectorCore::vector_key(id, 0),
                    &data,
                )
                .unwrap();
        }
        writer.commit(write).unwrap();
        let reader = Arc::new(AnyBackend::LsmReader(
            LsmReader::open_with_store(&path, Arc::clone(&store)).unwrap(),
        ));
        let mut reader_core = prefetch_test_core(Arc::clone(&reader), "prefetch_reader");
        let shared = Arc::new(super::super::shared_cache::SharedVectorCaches::new(
            NonZeroUsize::new(16).unwrap(),
            NonZeroUsize::new(16).unwrap(),
            NonZeroUsize::new(16).unwrap(),
        ));
        reader_core.shared_caches = Some(Arc::clone(&shared));
        let AnyBackend::LsmReader(lsm_reader) = &*reader else {
            unreachable!()
        };
        let pinned = AnyRead::LsmReader(Some(lsm_reader.begin_snapshot().unwrap()));
        let mut write = writer.begin_write().unwrap();
        let data = encode_vector(&[99.0, 0.0], &core.spindle).unwrap();
        writer
            .put(
                &mut write,
                core.seg_ns(SegmentDb::Vectors),
                &VectorCore::vector_key(1, 0),
                &data,
            )
            .unwrap();
        writer.commit(write).unwrap();
        reader
            .refresh_lsm_reader_with_poll_interval(std::time::Duration::from_millis(100))
            .unwrap();
        let rows = reader_core.prefetch_neighbor_vectors(
            &pinned,
            &PreparedSpindleQuery::None,
            &[1, 2],
            &HashSet::new(),
            0,
        );
        assert_eq!(rows.get(&1).unwrap().get_data(), &[1.0, 0.0]);
        assert_eq!(
            rows.get(&1).unwrap().get_data(),
            reader_core
                .get_vector(&pinned, 1, 0, true)
                .unwrap()
                .get_data()
        );
        assert!(shared
            .vector
            .get(&super::super::shared_cache::CacheKey::new(
                reader_core.cache_namespace,
                2,
                0
            ))
            .is_none());
        assert!(reader_core
            .prefetch_neighbor_vectors(
                &AnyRead::LsmReader(None),
                &PreparedSpindleQuery::None,
                &[2],
                &HashSet::new(),
                0
            )
            .is_empty());
    }

    #[test]
    fn hnsw_prefetch_traversal_matches_scalar_cache_with_filters_and_stale_links() {
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("hnsw-prefetch-search-{}", uuid::Uuid::new_v4()))
                .unwrap(),
        ));
        let mut batch_core = prefetch_test_core(Arc::clone(&backend), "prefetch_search");
        let mut scalar_core = prefetch_test_core(Arc::clone(&backend), "prefetch_search");
        batch_core.shared_caches = None;
        scalar_core.shared_caches = None;
        let mut write = backend.begin_write().unwrap();
        for id in 1..=48 {
            let data = [id as f32, 49.0 - id as f32];
            let encoded = encode_vector(&data, &batch_core.spindle).unwrap();
            backend
                .put(
                    &mut write,
                    batch_core.seg_ns(SegmentDb::Vectors),
                    &VectorCore::vector_key(id, 0),
                    &encoded,
                )
                .unwrap();
            scalar_core.cache_vector(id, 0, &data);
            let neighbors = if id == 1 {
                NeighborBlock::from_ids((2..=48).chain([2, 999]).collect())
            } else {
                NeighborBlock::from_ids(vec![1])
            };
            backend
                .put(
                    &mut write,
                    batch_core.seg_ns(SegmentDb::HnswNeighbors),
                    &VectorCore::neighbor_list_key(id, 0),
                    &VectorCore::encode_neighbor_block(&neighbors),
                )
                .unwrap();
        }
        backend.commit(write).unwrap();
        let read = backend.begin_read().unwrap();
        let query = HVector::new(0, vec![37.0, 12.0]);
        let vector_filter: [VF; 1] = [|vector| vector.get_id() % 2 == 0];
        let id_filter: [fn(u128) -> bool; 1] = [|id| id % 2 == 0];
        let summarize = |heap: BinaryHeap<HVector>| {
            heap.into_sorted_vec()
                .into_iter()
                .map(|row| (row.get_id(), row.get_distance()))
                .collect::<Vec<_>>()
        };
        for metric in [
            DistanceMetric::Euclid,
            DistanceMetric::Dot,
            DistanceMetric::Cosine,
        ] {
            batch_core.distance_metric = metric.clone();
            scalar_core.distance_metric = metric;
            let mut expected_entry = HVector::new(1, vec![1.0, 48.0]);
            let mut actual_entry = expected_entry.clone();
            let expected_arena = Bump::new();
            let actual_arena = Bump::new();
            let mut expected_ordinals = SidecarOrdinalRequestCache::new();
            let mut actual_ordinals = SidecarOrdinalRequestCache::new();
            let expected = scalar_core
                .search_level(
                    &read,
                    &query,
                    &mut expected_entry,
                    16,
                    0,
                    Some(&vector_filter),
                    &expected_arena,
                    &PreparedSpindleQuery::None,
                    None,
                    &[],
                    &mut expected_ordinals,
                )
                .unwrap();
            let actual = batch_core
                .search_level(
                    &read,
                    &query,
                    &mut actual_entry,
                    16,
                    0,
                    Some(&vector_filter),
                    &actual_arena,
                    &PreparedSpindleQuery::None,
                    None,
                    &[],
                    &mut actual_ordinals,
                )
                .unwrap();
            assert_eq!(summarize(actual), summarize(expected));
            batch_core.clear_vector_cache();
            let mut expected_entry = HVector::new(1, vec![1.0, 48.0]);
            let mut actual_entry = expected_entry.clone();
            let expected_arena = Bump::new();
            let actual_arena = Bump::new();
            let mut expected_ordinals = SidecarOrdinalRequestCache::new();
            let mut actual_ordinals = SidecarOrdinalRequestCache::new();
            let expected = scalar_core
                .search_level_id_filter(
                    &read,
                    &query,
                    &mut expected_entry,
                    16,
                    0,
                    Some(&id_filter),
                    &expected_arena,
                    &PreparedSpindleQuery::None,
                    None,
                    &[],
                    &mut expected_ordinals,
                )
                .unwrap();
            let actual = batch_core
                .search_level_id_filter(
                    &read,
                    &query,
                    &mut actual_entry,
                    16,
                    0,
                    Some(&id_filter),
                    &actual_arena,
                    &PreparedSpindleQuery::None,
                    None,
                    &[],
                    &mut actual_ordinals,
                )
                .unwrap();
            assert_eq!(summarize(actual), summarize(expected));
            batch_core.clear_vector_cache();
        }
    }

    struct EnvGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, val: &str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::set_var(key, val);
            Self { key, prev }
        }

        fn unset(key: &'static str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::remove_var(key);
            Self { key, prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    #[test]
    fn nav_cache_warm_cap_allows_zero_to_disable_warming() {
        assert_eq!(nav_cache_warm_cap_from_raw(Some("64"), 512), 64);
        assert_eq!(nav_cache_warm_cap_from_raw(Some("0"), 512), 0);
        assert_eq!(nav_cache_warm_cap_from_raw(Some("nope"), 512), 512);
        assert_eq!(nav_cache_warm_cap_from_raw(None, 0), 0);
    }

    #[test]
    fn hnsw_config_clamps_m_below_two() {
        let config = HNSWConfig::new(Some(1), Some(32), Some(64));

        assert_eq!(config.m, MIN_HNSW_M);
        assert_eq!(config.m_max_0, MIN_HNSW_M * 2);
        assert!(config.m_l.is_finite());
    }

    #[test]
    fn hnsw_level_assignment_caps_non_finite_samples() {
        assert_eq!(assign_hnsw_level(1.0, 1.0), 0);
        assert_eq!(assign_hnsw_level(0.0, 1.0), MAX_HNSW_LEVEL);
        assert_eq!(assign_hnsw_level(f64::NAN, 1.0), MAX_HNSW_LEVEL);
        assert_eq!(assign_hnsw_level(0.5, f64::INFINITY), 0);
    }

    #[test]
    fn merge_build_distance_uses_raw_geometry_for_dot_and_euclid() {
        // 1D Dot counterexample: under SQ8 codes -1 would rank +1 nearer
        // than -2; the raw inner product ranks -2 nearer.
        let dot_raw = vec![vec![-2.0f32], vec![-1.0], vec![1.0]];
        let d = |a, b| merge_build_distance(&DistanceMetric::Dot, &dot_raw, &[], 1, a, b);
        assert!(d(1, 0) < d(1, 2));

        // Unequal ranges: the wide dimension decides L2 nearness.
        let l2_raw = vec![vec![0.0f32, 0.0], vec![1.0, 1.0], vec![100.0, 0.0]];
        let e = |a, b| merge_build_distance(&DistanceMetric::Euclid, &l2_raw, &[], 2, a, b);
        assert!(e(0, 1) < e(0, 2));
        assert!((e(0, 2) - 100.0).abs() < 1e-4);

        // Cosine keeps using SQ8 codes.
        let codes = [255u8, 0, 128, 0, 0, 255];
        let c = |a, b| merge_build_distance(&DistanceMetric::Cosine, &[], &codes, 2, a, b);
        assert_eq!(c(0, 0), c(0, 1));
        assert!(c(0, 1) < c(0, 2));
    }

    fn setup() -> (heed3::Env, TempDir) {
        let dir = TempDir::new().unwrap();
        let env = unsafe {
            heed3::EnvOpenOptions::new()
                .map_size(256 * 1024 * 1024)
                .max_dbs(32)
                .open(dir.path())
                .unwrap()
        };
        (env, dir)
    }

    /// Test-only backend handle sharing the same env as the VectorCore under
    /// test. 6a is pure plumbing — the cores hold this Arc but no KV access
    /// routes through it yet.
    fn test_backend(env: &heed3::Env) -> Arc<AnyBackend> {
        use crate::helix_engine::storage_core::backend_lmdb::LmdbBackend;
        Arc::new(AnyBackend::Lmdb(LmdbBackend::from_env(env.clone())))
    }

    fn core_with_mode(env: &heed3::Env, name: &str, mode: SpindleMode) -> VectorCore {
        let mut txn = env.write_txn().unwrap();
        let spindle = SpindleConfig {
            mode,
            ..SpindleConfig::default()
        };
        let core = VectorCore::new_named(
            env,
            &mut txn,
            name,
            HNSWConfig::new(Some(8), Some(32), Some(64)),
            DistanceMetric::Cosine,
            spindle,
            test_backend(env),
        )
        .unwrap();
        txn.commit().unwrap();
        core
    }

    fn core_with_mode_and_mmap(
        env: &heed3::Env,
        data_dir: &std::path::Path,
        name: &str,
        mode: SpindleMode,
        dim: usize,
    ) -> VectorCore {
        core_with_spindle_and_mmap(
            env,
            data_dir,
            name,
            SpindleConfig {
                mode,
                ..SpindleConfig::default()
            },
            dim,
        )
    }

    fn core_with_spindle_and_mmap(
        env: &heed3::Env,
        data_dir: &std::path::Path,
        name: &str,
        spindle: SpindleConfig,
        dim: usize,
    ) -> VectorCore {
        let mut txn = env.write_txn().unwrap();
        let core = VectorCore::new_named_with_dir(
            env,
            &mut txn,
            name,
            HNSWConfig::new(Some(8), Some(32), Some(64)),
            DistanceMetric::Cosine,
            spindle,
            Some(data_dir),
            dim,
            None,
            test_backend(env),
        )
        .unwrap();
        txn.commit().unwrap();
        core
    }

    fn insert_n(env: &heed3::Env, core: &VectorCore, n: usize, dim: usize) {
        let mut txn = env.write_txn().unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for i in 0..n {
            let data: Vec<f32> = (0..dim)
                .map(|_| rand::Rng::random_range(&mut rng, -1.0f32..1.0f32))
                .collect();
            core.insert::<VF>(&mut txn, &data, Some((i + 1) as u128), None)
                .unwrap();
        }
        txn.commit().unwrap();
    }

    #[test]
    fn dense_delete_read_snapshot_chunk_uses_default_for_invalid_values() {
        assert_eq!(dense_delete_read_snapshot_chunk_from_raw(Some("128")), 128);
        assert_eq!(
            dense_delete_read_snapshot_chunk_from_raw(Some("0")),
            DENSE_DELETE_READ_SNAPSHOT_CHUNK_DEFAULT
        );
        assert_eq!(
            dense_delete_read_snapshot_chunk_from_raw(Some("nope")),
            DENSE_DELETE_READ_SNAPSHOT_CHUNK_DEFAULT
        );
        assert_eq!(
            dense_delete_read_snapshot_chunk_from_raw(None),
            DENSE_DELETE_READ_SNAPSHOT_CHUNK_DEFAULT
        );
    }

    #[test]
    #[serial_test::serial]
    fn delete_vectors_batch_be_refreshes_read_snapshot_after_chunk_limit() {
        use crate::helix_engine::storage_core::backend::StorageBackend;

        let _chunk = EnvGuard::set("HELIX_DENSE_DELETE_READ_SNAPSHOT_CHUNK", "2");
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "delete_be_snapshot_refresh", SpindleMode::None);

        let mut write = core.backend.begin_write().unwrap();
        for id in 1..=5u128 {
            let data = [id as f32, 0.0, 0.0, 0.0];
            core.insert_flat_be(&mut write, &data, Some(id), None)
                .unwrap();
        }
        core.backend.commit(write).unwrap();

        DENSE_DELETE_READ_SNAPSHOT_OPENS_FOR_TEST.store(0, std::sync::atomic::Ordering::Relaxed);
        let mut write = core.backend.begin_write().unwrap();
        core.delete_vectors_batch_be(&mut write, &[1, 2, 3, 4, 5])
            .unwrap();
        core.backend.commit(write).unwrap();

        assert_eq!(
            DENSE_DELETE_READ_SNAPSHOT_OPENS_FOR_TEST.load(std::sync::atomic::Ordering::Relaxed),
            3,
            "five ids with a chunk size of two must open one read snapshot per chunk"
        );

        let read = core.backend.begin_read().unwrap();
        for id in 1..=5u128 {
            assert!(
                core.get_vector(&read, id, 0, false).is_err(),
                "id {id} should be deleted"
            );
        }
    }

    #[test]
    fn sidecar_partial_ordinals_flat_filter_reads_share_snapshot() {
        let directory = TempDir::new().unwrap();
        let backend = Arc::new(
            AnyBackend::open_lsm_in_memory(&format!("ordinal-filter-{}", uuid::Uuid::new_v4()))
                .unwrap(),
        );
        let core = VectorCore::new_named_lsm_with_dir(
            "ordinal_filter",
            HNSWConfig::new(Some(8), Some(16), Some(32)),
            DistanceMetric::Euclid,
            SpindleConfig {
                mode: SpindleMode::None,
                ..SpindleConfig::default()
            },
            Some(directory.path()),
            2,
            Arc::clone(&backend),
        )
        .unwrap();
        let rows = [[1.0, 0.0], [0.0, 1.0]];
        let mut write = backend.begin_write().unwrap();
        for (id, row) in [(11u128, &rows[0]), (22, &rows[1])] {
            core.insert_flat_be(&mut write, row, Some(id), None)
                .unwrap();
            backend
                .put(&mut write, Namespace::Nodes, &id.to_be_bytes(), b"visible")
                .unwrap();
        }
        backend.commit(write).unwrap();
        assert!(!core.sidecar_ordinals_loaded().unwrap());

        let old_read = backend.begin_read().unwrap();
        let mut pending = backend.begin_write().unwrap();
        backend
            .delete(&mut pending, Namespace::Nodes, &22u128.to_be_bytes())
            .unwrap();
        let pending_read = backend
            .lsm_read_with_pending(pending.lsm_pending().unwrap().clone())
            .unwrap();
        for (read, expected_ids) in [(&old_read, vec![22, 11]), (&pending_read, vec![11])] {
            let read_errors = std::cell::Cell::new(0usize);
            let filter = |id: u128| match backend.get_with(
                read,
                Namespace::Nodes,
                &id.to_be_bytes(),
                |value| value.is_some(),
            ) {
                Ok(visible) => visible,
                Err(_) => {
                    read_errors.set(read_errors.get() + 1);
                    false
                }
            };
            let results = core
                .search_flat_mmap_ordinals_direct(read, &rows[1], 2, Some(&[filter]))
                .unwrap()
                .unwrap();
            assert_eq!(read_errors.get(), 0);
            assert_eq!(
                results.iter().map(HVector::get_id).collect::<Vec<_>>(),
                expected_ids
            );
            for result in results {
                let row = if result.get_id() == 11 {
                    &rows[0]
                } else {
                    &rows[1]
                };
                assert_eq!(
                    result.get_distance(),
                    core.distance_between(row, &rows[1]).unwrap()
                );
            }
        }
        assert!(!core.sidecar_ordinals_loaded().unwrap());
    }

    #[test]
    fn sidecar_partial_ordinals_preserve_exact_scoring_and_flat_results() {
        use crate::helix_engine::vector_core::mmap_vectors::MmapVectorStore;

        let directory = TempDir::new().unwrap();
        let segment = "partial_ordinals";
        let rows = [vec![1.0, 0.0], vec![0.0, 1.0]];
        let slices = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        drop(
            MmapVectorStore::create_from_slices(
                &directory.path().join(segment).with_extension("hvec"),
                &slices,
            )
            .unwrap(),
        );
        let backend = Arc::new(
            AnyBackend::open_lsm_in_memory(&format!("partial-ordinals-{}", uuid::Uuid::new_v4()))
                .unwrap(),
        );
        let mut write = backend.begin_write().unwrap();
        for (id, ordinal) in [(11u128, 0u64), (22, 1)] {
            backend
                .put(
                    &mut write,
                    Namespace::Segment {
                        physical_name: segment,
                        db: SegmentDb::Ordinals,
                    },
                    &id.to_be_bytes(),
                    &ordinal.to_le_bytes(),
                )
                .unwrap();
        }
        backend.commit(write).unwrap();
        let core = VectorCore::new_named_lsm_with_dir(
            segment,
            HNSWConfig::new(Some(8), Some(16), Some(32)),
            DistanceMetric::Euclid,
            SpindleConfig {
                mode: SpindleMode::None,
                ..SpindleConfig::default()
            },
            Some(directory.path()),
            2,
            Arc::clone(&backend),
        )
        .unwrap();
        core.cache_sidecar_ordinal(11, 0);
        let read = backend.begin_read().unwrap();
        let results = core
            .search_flat_mmap_ordinals_direct::<fn(u128) -> bool>(&read, &rows[1], 2, None)
            .unwrap()
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|result| result.get_id() == 22));
        assert_eq!(core.sidecar_ordinal(&read, 22).unwrap(), Some(1));
        for (id, expected_row) in [(11, &rows[0]), (22, &rows[1])] {
            let distance = core
                .score_neighbor_distance(&read, &PreparedSpindleQuery::None, id, 0, &rows[1])
                .unwrap()
                .unwrap();
            assert_eq!(
                distance,
                core.distance_between(expected_row, &rows[1]).unwrap()
            );
        }
    }

    #[test]
    fn sidecar_request_ordinals_batch_and_memoize_reads() {
        use crate::helix_engine::vector_core::mmap_vectors::MmapVectorStore;

        let directory = TempDir::new().unwrap();
        let segment = "request_ordinals";
        let rows = [vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let raw_fallback = encode_vector(&[2.0, 2.0], &SpindleConfig::default()).unwrap();
        let decoded_fallback = decode_vector(&raw_fallback).unwrap();
        let marker_fallback = encode_vector(&[3.0, 3.0], &SpindleConfig::default()).unwrap();
        let decoded_marker_fallback = decode_vector(&marker_fallback).unwrap();
        let slices = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        drop(
            MmapVectorStore::create_from_slices(
                &directory.path().join(segment).with_extension("hvec"),
                &slices,
            )
            .unwrap(),
        );
        let backend = Arc::new(
            AnyBackend::open_lsm_in_memory(&format!("request-ordinals-{}", uuid::Uuid::new_v4()))
                .unwrap(),
        );
        let mut write = backend.begin_write().unwrap();
        for (id, ordinal) in [(11u128, 0u64), (22, 1), (33, 2)] {
            backend
                .put(
                    &mut write,
                    Namespace::Segment {
                        physical_name: segment,
                        db: SegmentDb::Ordinals,
                    },
                    &id.to_be_bytes(),
                    &ordinal.to_le_bytes(),
                )
                .unwrap();
        }
        backend
            .put(
                &mut write,
                Namespace::Segment {
                    physical_name: segment,
                    db: SegmentDb::Vectors,
                },
                &VectorCore::vector_key(44, 0),
                &raw_fallback,
            )
            .unwrap();
        backend
            .put(
                &mut write,
                Namespace::Segment {
                    physical_name: segment,
                    db: SegmentDb::Vectors,
                },
                &VectorCore::vector_key(55, 0),
                &marker_fallback,
            )
            .unwrap();
        backend
            .put(
                &mut write,
                Namespace::Segment {
                    physical_name: segment,
                    db: SegmentDb::Vectors,
                },
                &VectorCore::vector_key(55, 1),
                EXTERNALIZED_VECTOR_MARKER,
            )
            .unwrap();
        backend
            .put(
                &mut write,
                Namespace::Segment {
                    physical_name: segment,
                    db: SegmentDb::Vectors,
                },
                &VectorCore::vector_key(66, 0),
                EXTERNALIZED_VECTOR_MARKER,
            )
            .unwrap();
        backend.commit(write).unwrap();
        let core = VectorCore::new_named_lsm_with_dir(
            segment,
            HNSWConfig::new(Some(8), Some(16), Some(32)),
            DistanceMetric::Euclid,
            SpindleConfig {
                mode: SpindleMode::None,
                ..SpindleConfig::default()
            },
            Some(directory.path()),
            2,
            Arc::clone(&backend),
        )
        .unwrap();
        let AnyBackend::Lsm(lsm) = &*backend else {
            unreachable!("backend is LSM in this test");
        };
        let read = backend.begin_read().unwrap();
        let repeated_ids = [11u128, 22, 33, 22, 44, 33, 11, 44, 22];

        lsm.reset_read_count();
        let uncached = repeated_ids
            .iter()
            .map(|id| core.sidecar_ordinal_from_store(&read, *id).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            uncached,
            vec![
                Some(0),
                Some(1),
                Some(2),
                Some(1),
                None,
                Some(2),
                Some(0),
                None,
                Some(1)
            ]
        );
        let uncached_reads = lsm.read_count();
        assert_eq!(uncached_reads, repeated_ids.len());

        lsm.reset_read_count();
        let mut request_cache = SidecarOrdinalRequestCache::new();
        core.prefetch_sidecar_ordinals(&read, &repeated_ids, &HashSet::new(), &mut request_cache)
            .unwrap();
        let cached = repeated_ids
            .iter()
            .map(|id| {
                core.sidecar_ordinal_with_request_cache(&read, *id, &mut request_cache)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(cached, uncached);
        let request_cached_reads = lsm.read_count();
        assert_eq!(request_cached_reads, 4);
        eprintln!(
            "ordinal_request_cache_evidence uncached_reads={uncached_reads} prefetched_reads={request_cached_reads}"
        );
        assert!(
            request_cached_reads < uncached_reads,
            "request cache should reduce repeated ordinal point reads: uncached={uncached_reads}, request_cached={request_cached_reads}"
        );

        lsm.reset_read_count();
        let vector = core
            .get_vector_with_ordinal_cache(&read, 44, 0, true, &mut request_cache)
            .unwrap();
        assert_eq!(vector.get_data(), decoded_fallback.as_slice());
        assert_eq!(
            lsm.read_count(),
            1,
            "known ordinal miss should read only the raw vector row"
        );
        eprintln!("known_ordinal_miss_get_vector_reads={}", lsm.read_count());

        core.clear_vector_cache();
        lsm.reset_read_count();
        let distance = core
            .score_neighbor_distance_with_cache(
                &read,
                &PreparedSpindleQuery::None,
                44,
                0,
                decoded_fallback.as_slice(),
                Some(&mut request_cache),
            )
            .unwrap()
            .unwrap();
        assert!(distance < 1e-6);
        assert_eq!(
            lsm.read_count(),
            1,
            "known ordinal miss scoring should not re-probe the ordinal key"
        );
        eprintln!("known_ordinal_miss_scoring_reads={}", lsm.read_count());

        core.sidecar_ordinal_with_request_cache(&read, 55, &mut request_cache)
            .unwrap();
        lsm.reset_read_count();
        let vector = core
            .get_vector_with_ordinal_cache(&read, 55, 1, true, &mut request_cache)
            .unwrap();
        assert_eq!(vector.get_level(), 1);
        assert_eq!(vector.get_data(), decoded_marker_fallback.as_slice());
        assert_eq!(
            lsm.read_count(),
            2,
            "known ordinal miss on an upper-level marker should read marker + raw level-0 row, not re-probe the ordinal"
        );

        core.clear_vector_cache();
        lsm.reset_read_count();
        let distance = core
            .score_neighbor_distance_with_cache(
                &read,
                &PreparedSpindleQuery::None,
                55,
                1,
                decoded_marker_fallback.as_slice(),
                Some(&mut request_cache),
            )
            .unwrap()
            .unwrap();
        assert!(distance < 1e-6);
        assert_eq!(
            lsm.read_count(),
            2,
            "known ordinal miss scoring for an upper-level marker should fall back to raw level-0 without another ordinal read"
        );

        core.sidecar_ordinal_with_request_cache(&read, 66, &mut request_cache)
            .unwrap();
        lsm.reset_read_count();
        let error = core
            .get_vector_with_ordinal_cache(&read, 66, 0, true, &mut request_cache)
            .unwrap_err()
            .to_string();
        assert!(error.contains("externalized vector 66 missing ordinal mapping"));
        assert_eq!(
            lsm.read_count(),
            1,
            "known ordinal miss on a level-0 marker should read only the marker row before returning the repairable error"
        );
        core.clear_externalized_marker_repair_needed();
        let repaired = core
            .search_exact_ids_with_selectivity(
                &read,
                decoded_marker_fallback.as_slice(),
                [66],
                1,
                None,
            )
            .unwrap();
        assert!(repaired.is_empty());
        assert!(
            core.externalized_marker_repair_needed(),
            "search callers must still mark missing-ordinal markers for repair"
        );

        let results = core
            .search_exact_ids_with_selectivity(&read, &rows[1], [11u128, 22, 33], 3, None)
            .unwrap();
        assert_eq!(
            results.iter().map(HVector::get_id).collect::<Vec<_>>(),
            vec![22, 33, 11]
        );
        let distances = results
            .iter()
            .map(HVector::get_distance)
            .collect::<Vec<_>>();
        assert!((distances[0] - 0.0).abs() < 1e-6);
        assert!((distances[1] - 1.0).abs() < 1e-6);
        assert!((distances[2] - 2.0_f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    #[serial_test::serial]
    fn sidecar_request_ordinals_prefetch_file_backed_reader_snapshot() {
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use crate::helix_engine::storage_core::backend_lsm_reader::LsmReader;
        use crate::helix_engine::vector_core::mmap_vectors::MmapVectorStore;
        use object_store::local::LocalFileSystem;
        use object_store::ObjectStore;

        let _cache_dir = EnvGuard::unset("HELIX_LSM_CACHE_DIR");
        let _multi_get = EnvGuard::set("HELIX_LSM_MULTI_GET", "1");
        let directory = TempDir::new().unwrap();
        let segment = "request_ordinals_reader";
        let rows = (1..=96)
            .map(|id| vec![id as f32, 96.0 - id as f32])
            .collect::<Vec<_>>();
        let slices = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        drop(
            MmapVectorStore::create_from_slices(
                &directory.path().join(segment).with_extension("hvec"),
                &slices,
            )
            .unwrap(),
        );

        let store_directory = TempDir::new().unwrap();
        let store: Arc<dyn ObjectStore> =
            Arc::new(LocalFileSystem::new_with_prefix(store_directory.path()).unwrap());
        let path = format!("reader-ordinal-prefetch-{}", uuid::Uuid::new_v4());
        let writer = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_with_store(&path, Arc::clone(&store)).unwrap(),
        ));
        let mut write = writer.begin_write().unwrap();
        for id in 1..=96u128 {
            let ordinal = (id - 1) as u64;
            writer
                .put(
                    &mut write,
                    Namespace::Segment {
                        physical_name: segment,
                        db: SegmentDb::Ordinals,
                    },
                    &id.to_be_bytes(),
                    &ordinal.to_le_bytes(),
                )
                .unwrap();
        }
        writer.commit(write).unwrap();
        let AnyBackend::Lsm(lsm_writer) = &*writer else {
            unreachable!("writer backend is LSM in this test");
        };
        lsm_writer.flush_durable().unwrap();
        let mut pending_directories = vec![store_directory.path().to_path_buf()];
        let mut sst_files = 0usize;
        while let Some(directory) = pending_directories.pop() {
            for entry in std::fs::read_dir(directory).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    pending_directories.push(path);
                } else if path.extension().is_some_and(|extension| extension == "sst") {
                    sst_files += 1;
                }
            }
        }
        assert!(sst_files > 0, "fixture must place ordinals in SST files");

        let reader = Arc::new(AnyBackend::LsmReader(
            LsmReader::open_with_store(&path, Arc::clone(&store)).unwrap(),
        ));
        reader
            .refresh_lsm_reader_with_poll_interval(std::time::Duration::from_millis(100))
            .unwrap();
        let AnyBackend::LsmReader(lsm_reader) = &*reader else {
            unreachable!("reader backend is LsmReader in this test");
        };
        let read = AnyRead::LsmReader(Some(lsm_reader.begin_snapshot().unwrap()));
        let core = VectorCore::new_named_lsm_with_dir(
            segment,
            HNSWConfig::new(Some(8), Some(16), Some(32)),
            DistanceMetric::Euclid,
            SpindleConfig {
                mode: SpindleMode::None,
                ..SpindleConfig::default()
            },
            Some(directory.path()),
            2,
            Arc::clone(&reader),
        )
        .unwrap();
        let frontier_ids = (1..=96u128).chain([17, 31, 95]).collect::<Vec<_>>();
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let mut request_cache = SidecarOrdinalRequestCache::new();

        metrics::with_local_recorder(&recorder, || {
            core.prefetch_sidecar_ordinals(
                &read,
                &frontier_ids,
                &HashSet::new(),
                &mut request_cache,
            )
            .unwrap();
        });
        let metrics = handle.render();
        assert!(
            metrics.contains("helix_hnsw_ordinal_prefetch_batches_total 1"),
            "{metrics}"
        );
        assert!(
            metrics.contains("helix_hnsw_ordinal_prefetch_rows_total 96"),
            "{metrics}"
        );
        eprintln!(
            "reader_snapshot_ordinal_prefetch_evidence sst_files={sst_files} batches=1 rows=96"
        );
        assert_eq!(request_cache.entries.len(), 96);
        assert_eq!(request_cache.get(17), Some(Some(16)));

        for id in frontier_ids {
            assert!(core
                .sidecar_ordinal_with_request_cache(&read, id, &mut request_cache)
                .unwrap()
                .is_some());
        }
    }

    #[test]
    fn sidecar_partial_ordinals_preserve_pinned_reader_after_refresh() {
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use crate::helix_engine::storage_core::backend_lsm_reader::LsmReader;
        use crate::helix_engine::vector_core::mmap_vectors::MmapVectorStore;
        use object_store::memory::InMemory;
        use object_store::ObjectStore;

        let directory = TempDir::new().unwrap();
        let segment = "partial_reader_ordinals";
        let rows = [vec![1.0, 0.0], vec![0.0, 1.0], vec![9.0, 0.0]];
        let slices = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        drop(
            MmapVectorStore::create_from_slices(
                &directory.path().join(segment).with_extension("hvec"),
                &slices,
            )
            .unwrap(),
        );
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = format!("partial-reader-ordinals-{}", uuid::Uuid::new_v4());
        let writer = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_with_store(&path, Arc::clone(&store)).unwrap(),
        ));
        let namespace = Namespace::Segment {
            physical_name: segment,
            db: SegmentDb::Ordinals,
        };
        let mut write = writer.begin_write().unwrap();
        for (id, ordinal) in [(11u128, 0u64), (22, 1)] {
            writer
                .put(
                    &mut write,
                    namespace,
                    &id.to_be_bytes(),
                    &ordinal.to_le_bytes(),
                )
                .unwrap();
        }
        writer.commit(write).unwrap();
        let reader = Arc::new(AnyBackend::LsmReader(
            LsmReader::open_with_store(&path, Arc::clone(&store)).unwrap(),
        ));
        let core = VectorCore::new_named_lsm_with_dir(
            segment,
            HNSWConfig::new(Some(8), Some(16), Some(32)),
            DistanceMetric::Euclid,
            SpindleConfig {
                mode: SpindleMode::None,
                ..SpindleConfig::default()
            },
            Some(directory.path()),
            2,
            Arc::clone(&reader),
        )
        .unwrap();
        core.cache_sidecar_ordinal(11, 0);
        let AnyBackend::LsmReader(lsm_reader) = &*reader else {
            unreachable!()
        };
        let old_read = AnyRead::LsmReader(Some(lsm_reader.begin_snapshot().unwrap()));
        let mut write = writer.begin_write().unwrap();
        writer
            .put(
                &mut write,
                namespace,
                &22u128.to_be_bytes(),
                &2u64.to_le_bytes(),
            )
            .unwrap();
        writer.commit(write).unwrap();
        reader
            .refresh_lsm_reader_with_poll_interval(std::time::Duration::from_millis(100))
            .unwrap();
        let new_read = AnyRead::LsmReader(Some(lsm_reader.begin_snapshot().unwrap()));
        assert_eq!(
            core.sidecar_ordinal_from_store(&new_read, 22).unwrap(),
            Some(2)
        );
        assert_eq!(core.sidecar_ordinal(&old_read, 22).unwrap(), Some(1));
        let mut old_request_cache = SidecarOrdinalRequestCache::new();
        assert_eq!(
            core.sidecar_ordinal_with_request_cache(&old_read, 22, &mut old_request_cache)
                .unwrap(),
            Some(1)
        );
        assert_eq!(
            core.sidecar_ordinal_with_request_cache(&old_read, 22, &mut old_request_cache)
                .unwrap(),
            Some(1)
        );
        let mut new_request_cache = SidecarOrdinalRequestCache::new();
        assert_eq!(
            core.sidecar_ordinal_with_request_cache(&new_read, 22, &mut new_request_cache)
                .unwrap(),
            Some(2)
        );
        core.clear_vector_cache();
        let old_hydrated = core
            .get_vector_with_ordinal_cache(&old_read, 22, 0, true, &mut old_request_cache)
            .unwrap();
        assert_eq!(old_hydrated.get_data(), rows[1].as_slice());
        let old_then_new_new = core.get_vector(&new_read, 22, 0, true).unwrap();
        assert_eq!(old_then_new_new.get_data(), rows[2].as_slice());

        core.clear_vector_cache();
        let new_hydrated = core
            .get_vector_with_ordinal_cache(&new_read, 22, 0, true, &mut new_request_cache)
            .unwrap();
        assert_eq!(new_hydrated.get_data(), rows[2].as_slice());
        let new_then_old_old = core.get_vector(&old_read, 22, 0, true).unwrap();
        assert_eq!(new_then_old_old.get_data(), rows[1].as_slice());
        for (read, expected_row) in [(&old_read, &rows[1]), (&new_read, &rows[2])] {
            let distance = core
                .score_neighbor_distance(read, &PreparedSpindleQuery::None, 22, 0, &rows[1])
                .unwrap()
                .unwrap();
            assert_eq!(
                distance,
                core.distance_between(expected_row, &rows[1]).unwrap()
            );
            let results = core
                .search_flat_mmap_ordinals_direct::<fn(u128) -> bool>(read, &rows[1], 2, None)
                .unwrap()
                .unwrap();
            let result = results.iter().find(|result| result.get_id() == 22).unwrap();
            assert_eq!(result.get_data(), expected_row.as_slice());
        }
        assert_eq!(core.cached_sidecar_ordinal(22).unwrap(), None);
    }

    #[test]
    fn lsm_named_core_defers_sidecar_ordinal_scan_on_open() {
        use crate::helix_engine::storage_core::backend::StorageBackend;

        let dir = TempDir::new().unwrap();
        let physical_name = "lazy_ordinals";
        let hvtq_path = dir.path().join(physical_name).with_extension("hvtq");
        let vectors = [vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let slices = vectors.iter().map(Vec::as_slice).collect::<Vec<&[f32]>>();
        let spindle = SpindleConfig {
            mode: SpindleMode::TurboProd,
            keep_original: false,
            ..SpindleConfig::default()
        };
        let store = MmapTurboQuantStore::create_from_slices(&hvtq_path, &slices, &spindle).unwrap();
        drop(store);

        let backend =
            Arc::new(AnyBackend::open_lsm_in_memory("/lsm-named-core-defers-ordinals").unwrap());
        let mut write = backend.begin_write().unwrap();
        backend
            .put(
                &mut write,
                Namespace::Segment {
                    physical_name,
                    db: SegmentDb::Ordinals,
                },
                &1u128.to_be_bytes(),
                &0u64.to_le_bytes(),
            )
            .unwrap();
        backend
            .put(
                &mut write,
                Namespace::Segment {
                    physical_name,
                    db: SegmentDb::Ordinals,
                },
                &2u128.to_be_bytes(),
                &1u64.to_le_bytes(),
            )
            .unwrap();
        backend.commit(write).unwrap();

        let core = VectorCore::new_named_lsm_with_dir(
            physical_name,
            HNSWConfig::new(Some(8), Some(32), Some(64)),
            DistanceMetric::Cosine,
            spindle,
            Some(dir.path()),
            4,
            backend,
        )
        .unwrap();

        assert!(core.mmap_store.read().unwrap().is_some());
        assert!(core.mmap_ordinals.read().unwrap().is_none());

        let read = core.backend.begin_read().unwrap();
        assert_eq!(core.sidecar_ordinal(&read, 1).unwrap(), Some(0));
        let ordinals = core.mmap_ordinals.read().unwrap();
        let ordinals = ordinals.as_ref().unwrap();
        assert_eq!(ordinals.entries().get(&1), Some(&0));
        assert_eq!(ordinals.entries().get(&2), Some(&1));
    }

    #[test]
    fn delete_removes_vector_from_search_results() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "del_test", SpindleMode::None);
        insert_n(&env, &core, 50, 32);

        // Verify vector 1 is findable
        let txn = env.read_txn().unwrap();
        let query: Vec<f32> = vec![0.5; 32];
        let results = core
            .search::<VF>(&core.backend.read_borrowed(&txn), &query, 50, None, false)
            .unwrap();
        assert!(results.iter().any(|r| r.get_id() == 1));
        drop(txn);

        // Delete vector 1
        let mut wtxn = env.write_txn().unwrap();
        core.delete_vector(&mut wtxn, 1).unwrap();
        wtxn.commit().unwrap();

        // Vector 1 should no longer appear
        let txn = env.read_txn().unwrap();
        let results = core
            .search::<VF>(&core.backend.read_borrowed(&txn), &query, 50, None, false)
            .unwrap();
        assert!(!results.iter().any(|r| r.get_id() == 1));
    }

    #[test]
    fn delete_nonexistent_vector_is_idempotent() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "idempotent", SpindleMode::None);
        insert_n(&env, &core, 10, 16);

        let mut wtxn = env.write_txn().unwrap();
        // Delete a vector that was never inserted — should not error
        let result = core.delete_vector(&mut wtxn, 99999);
        assert!(result.is_ok());
        wtxn.commit().unwrap();

        // Index still works
        let txn = env.read_txn().unwrap();
        let query: Vec<f32> = vec![0.5; 16];
        let results = core
            .search::<VF>(&core.backend.read_borrowed(&txn), &query, 10, None, false)
            .unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn search_skips_stale_neighbor_reference() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "stale_neighbor", SpindleMode::None);
        insert_n(&env, &core, 30, 16);

        let entry_id = {
            let txn = env.read_txn().unwrap();
            let rd = core.backend.read_borrowed(&txn);
            core.get_entry_point(&rd).unwrap().get_id()
        };

        let stale_id = 999_999_u128;
        let mut wtxn = env.write_txn().unwrap();
        let mut neighbors = core
            .get_neighbor_ids(&core.backend.read_borrowed(&wtxn), entry_id, 0)
            .unwrap();
        neighbors.push(stale_id);
        core.put_neighbor_ids(&mut wtxn, entry_id, 0, &neighbors)
            .unwrap();
        wtxn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        let query: Vec<f32> = vec![0.5; 16];
        let results = core
            .search::<VF>(&core.backend.read_borrowed(&txn), &query, 10, None, false)
            .unwrap();
        assert!(!results.is_empty());
        assert!(!results.iter().any(|r| r.get_id() == stale_id));
    }

    #[test]
    fn delete_removes_asymmetric_neighbor_references() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "asymmetric_delete", SpindleMode::None);
        insert_n(&env, &core, 30, 16);

        let mut wtxn = env.write_txn().unwrap();
        core.put_neighbor_ids(&mut wtxn, 2, 0, &[1]).unwrap();
        core.delete_vector(&mut wtxn, 1).unwrap();
        wtxn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        assert!(!core
            .get_neighbor_ids(&core.backend.read_borrowed(&txn), 2, 0)
            .unwrap()
            .contains(&1));
        let query: Vec<f32> = vec![0.5; 16];
        let results = core
            .search::<VF>(&core.backend.read_borrowed(&txn), &query, 10, None, false)
            .unwrap();
        assert!(!results.iter().any(|r| r.get_id() == 1));
    }

    #[test]
    fn flat_insert_searches_before_and_after_index_build() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "flat_mode", SpindleMode::None);

        let mut wtxn = env.write_txn().unwrap();
        core.insert_flat(&mut wtxn, &[1.0, 0.0], Some(1), None)
            .unwrap();
        core.insert_flat(&mut wtxn, &[0.0, 1.0], Some(2), None)
            .unwrap();
        core.insert_flat(&mut wtxn, &[0.8, 0.2], Some(3), None)
            .unwrap();
        wtxn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        assert!(!core.has_index(&core.backend.read_borrowed(&txn)).unwrap());
        let results = core
            .search::<VF>(
                &core.backend.read_borrowed(&txn),
                &[1.0, 0.0],
                2,
                None,
                false,
            )
            .unwrap();
        assert_eq!(results[0].get_id(), 1);
        drop(txn);

        let mut wtxn = env.write_txn().unwrap();
        core.build_index_from_flat(&mut wtxn).unwrap();
        wtxn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        assert!(core.has_index(&core.backend.read_borrowed(&txn)).unwrap());
        let results = core
            .search::<VF>(
                &core.backend.read_borrowed(&txn),
                &[1.0, 0.0],
                2,
                None,
                false,
            )
            .unwrap();
        assert_eq!(results[0].get_id(), 1);
    }

    #[test]
    fn delete_entry_point_keeps_index_searchable() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "ep_del", SpindleMode::None);
        insert_n(&env, &core, 20, 16);

        // Find the entry point
        let txn = env.read_txn().unwrap();
        let ep = core
            .get_entry_point(&core.backend.read_borrowed(&txn))
            .unwrap();
        let ep_id = ep.get_id();
        drop(txn);

        // Delete the entry point
        let mut wtxn = env.write_txn().unwrap();
        core.delete_vector(&mut wtxn, ep_id).unwrap();
        wtxn.commit().unwrap();

        // Search should still work with remaining 19 vectors
        let txn = env.read_txn().unwrap();
        let query: Vec<f32> = vec![0.5; 16];
        let results = core
            .search::<VF>(&core.backend.read_borrowed(&txn), &query, 10, None, false)
            .unwrap();
        assert!(!results.is_empty());
        assert!(!results.iter().any(|r| r.get_id() == ep_id));
    }

    #[test]
    fn entry_point_uses_highest_stored_level() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "ep_level", SpindleMode::None);

        let mut txn = env.write_txn().unwrap();
        core.put_raw_vector(&mut txn, 42, 0, &[1.0, 0.0, 0.0, 0.0])
            .unwrap();
        core.put_raw_vector(&mut txn, 42, 3, &[1.0, 0.0, 0.0, 0.0])
            .unwrap();
        let ep = HVector::from_slice(42, 3, vec![1.0, 0.0, 0.0, 0.0]);
        core.set_entry_point(&mut txn, &ep).unwrap();
        txn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        let loaded = core
            .get_entry_point(&core.backend.read_borrowed(&txn))
            .unwrap();
        assert_eq!(loaded.get_id(), 42);
        assert_eq!(loaded.get_level(), 3);
    }

    #[test]
    fn packed_neighbor_lists_roundtrip() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "packed_neighbors", SpindleMode::None);

        let mut txn = env.write_txn().unwrap();
        core.put_neighbor_ids(&mut txn, 11, 0, &[22, 33, 44])
            .unwrap();
        txn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        assert_eq!(
            core.get_neighbor_ids(&core.backend.read_borrowed(&txn), 11, 0)
                .unwrap(),
            vec![22, 33, 44]
        );
        assert_eq!(
            core.out_edges_db
                .as_ref()
                .unwrap()
                .iter(&txn)
                .unwrap()
                .count(),
            0
        );
    }

    #[test]
    fn packed_neighbor_lists_include_compact_sidecars_when_vectors_exist() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "packed_neighbors_codes", SpindleMode::None);

        let mut txn = env.write_txn().unwrap();
        core.put_raw_vector(&mut txn, 22, 0, &[1.0, 0.0, 0.0, 0.0])
            .unwrap();
        core.put_raw_vector(&mut txn, 33, 0, &[0.0, 1.0, 0.0, 0.0])
            .unwrap();
        core.put_raw_vector(&mut txn, 44, 0, &[0.0, 0.0, 1.0, 0.0])
            .unwrap();
        core.put_neighbor_ids(&mut txn, 11, 0, &[22, 33, 44])
            .unwrap();
        txn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        let block = core
            .get_neighbor_block(&core.backend.read_borrowed(&txn), 11, 0)
            .unwrap();
        assert_eq!(block.ids, vec![22, 33, 44]);
        assert!(block.has_approx_codes());
        assert!(block.code_at(0).is_some());
        assert_eq!(
            core.out_edges_db
                .as_ref()
                .unwrap()
                .iter(&txn)
                .unwrap()
                .count(),
            0
        );
    }

    #[test]
    fn legacy_neighbor_edges_still_read() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "legacy_neighbors", SpindleMode::None);

        let mut txn = env.write_txn().unwrap();
        core.out_edges_db
            .as_ref()
            .unwrap()
            .put(&mut txn, &VectorCore::out_edges_key(7, 0, Some(8)), &())
            .unwrap();
        core.out_edges_db
            .as_ref()
            .unwrap()
            .put(&mut txn, &VectorCore::out_edges_key(7, 0, Some(9)), &())
            .unwrap();
        txn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        assert_eq!(
            core.get_neighbor_ids(&core.backend.read_borrowed(&txn), 7, 0)
                .unwrap(),
            vec![8, 9]
        );
    }

    #[test]
    fn delete_vector_removes_reverse_packed_neighbors() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "packed_delete", SpindleMode::None);

        let mut txn = env.write_txn().unwrap();
        core.put_neighbor_ids(&mut txn, 1, 0, &[2]).unwrap();
        core.put_neighbor_ids(&mut txn, 2, 0, &[1]).unwrap();
        txn.commit().unwrap();

        let mut txn = env.write_txn().unwrap();
        core.delete_vector(&mut txn, 1).unwrap();
        txn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        assert!(core
            .get_neighbor_ids(&core.backend.read_borrowed(&txn), 2, 0)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn filtered_search_excludes_non_matching_vectors() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "filter_test", SpindleMode::None);

        // Insert 100 vectors with varied data so HNSW has enough graph connectivity
        let mut txn = env.write_txn().unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(77);
        for i in 1..=100u128 {
            let data: Vec<f32> = (0..16)
                .map(|_| rand::Rng::random_range(&mut rng, -1.0f32..1.0f32))
                .collect();
            core.insert::<VF>(&mut txn, &data, Some(i), None).unwrap();
        }
        txn.commit().unwrap();

        // Search with filter: only ids <= 50
        let txn = env.read_txn().unwrap();
        let query: Vec<f32> = vec![0.5; 16];
        let filter_fn = |v: &HVector| -> bool { v.get_id() <= 50 };
        let results = core
            .search(
                &core.backend.read_borrowed(&txn),
                &query,
                10,
                Some(&[filter_fn]),
                true,
            )
            .unwrap();

        // All results should have id <= 50
        assert!(!results.is_empty(), "Filtered search returned no results");
        for r in &results {
            assert!(r.get_id() <= 50, "Expected id <= 50, got {}", r.get_id());
        }
    }

    // ── Adaptive ef tests ──

    #[test]
    fn compute_adaptive_ef_unfiltered_uses_base() {
        let config = HNSWConfig::new(Some(8), Some(32), Some(64));
        assert!(config.adaptive_ef_enabled);

        let (env, _dir) = setup();
        let core = core_with_mode(&env, "adaptive_base", SpindleMode::None);

        // No selectivity hint (unfiltered) → base ef unchanged
        assert_eq!(core.compute_adaptive_ef(64, None), 64);
        // Selectivity 1.0 (all pass) → base ef unchanged
        assert_eq!(core.compute_adaptive_ef(64, Some(1.0)), 64);
    }

    #[test]
    fn compute_adaptive_ef_selective_filter_increases_ef() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "adaptive_sel", SpindleMode::None);

        // 1% selectivity → multiplier = sqrt(1/0.01) = 10.0, clamped to 8.0
        let ef_1pct = core.compute_adaptive_ef(64, Some(0.01));
        assert_eq!(ef_1pct, (64.0_f32 * 8.0) as usize); // 512

        // 10% selectivity → multiplier = sqrt(1/0.1) = ~3.16
        let ef_10pct = core.compute_adaptive_ef(64, Some(0.1));
        let expected_10 = (64.0_f32 * (1.0_f32 / 0.1_f32).sqrt().clamp(1.0, 8.0)) as usize;
        assert_eq!(ef_10pct, expected_10);
        assert!(
            ef_10pct > 64,
            "10% selectivity should increase ef above base"
        );

        // 50% selectivity → multiplier = sqrt(2) ≈ 1.41
        let ef_50pct = core.compute_adaptive_ef(64, Some(0.5));
        let expected_50 = (64.0_f32 * (1.0_f32 / 0.5_f32).sqrt().clamp(1.0, 8.0)) as usize;
        assert_eq!(ef_50pct, expected_50);
        assert!(ef_50pct > 64, "50% selectivity should still increase ef");
        assert!(ef_50pct < ef_10pct, "50% should use lower ef than 10%");
    }

    #[test]
    fn compute_adaptive_ef_never_decreases_below_base() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "adaptive_floor", SpindleMode::None);

        // Even with selectivity > 1.0 (impossible in practice, but defensive)
        assert_eq!(core.compute_adaptive_ef(64, Some(2.0)), 64);
        assert_eq!(core.compute_adaptive_ef(64, Some(100.0)), 64);
    }

    #[test]
    fn compute_adaptive_ef_disabled_uses_base() {
        let (env, _dir) = setup();
        let mut txn = env.write_txn().unwrap();
        let mut config = HNSWConfig::new(Some(8), Some(32), Some(64));
        config.adaptive_ef_enabled = false;
        let core = VectorCore::new_named(
            &env,
            &mut txn,
            "adaptive_off",
            config,
            DistanceMetric::Cosine,
            SpindleConfig {
                mode: SpindleMode::None,
                ..SpindleConfig::default()
            },
            test_backend(&env),
        )
        .unwrap();
        txn.commit().unwrap();

        // Even with very selective filter, ef stays at base when disabled
        assert_eq!(core.compute_adaptive_ef(64, Some(0.01)), 64);
    }

    #[test]
    fn compute_adaptive_ef_capped_by_ef_override() {
        // When ef_override is provided, effective_ef must not exceed the
        // override value — even if adaptive EF would inflate it.
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "adaptive_cap", SpindleMode::None);

        // Without override, 1% selectivity inflates 64 -> 512 (8x clamp)
        let uncapped = core.compute_adaptive_ef(64, Some(0.01));
        assert_eq!(uncapped, 512);

        // The cap logic lives in search_with_selectivity_ef_observed, not in
        // compute_adaptive_ef directly, so we verify via the search entry
        // point. Insert enough vectors to build an HNSW index.
        insert_n(&env, &core, 50, 16);
        let txn = env.read_txn().unwrap();
        let query: Vec<f32> = vec![0.5; 16];

        // With ef_override=96, effective_ef should be min(adaptive, 96) = 96
        let results = core
            .search_with_selectivity_ef::<VF>(
                &core.backend.read_borrowed(&txn),
                &query,
                5,
                None,
                false,
                Some(0.01), // very selective
                Some(96),   // explicit cap
            )
            .unwrap();
        // The search should succeed (the cap doesn't break anything)
        assert!(!results.is_empty());
    }

    #[test]
    fn adaptive_rerank_preserves_small_top_k_and_tapers_large_k() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "adaptive_rerank_taper", SpindleMode::TurboProd);

        assert_eq!(core.compute_adaptive_rerank_oversampling(10, None), 16);
        assert_eq!(core.compute_adaptive_rerank_oversampling(50, None), 16);
        assert_eq!(core.compute_adaptive_rerank_oversampling(51, None), 8);
        assert_eq!(core.compute_adaptive_rerank_oversampling(201, None), 4);
        assert_eq!(core.compute_adaptive_rerank_oversampling(1001, None), 2);
    }

    #[test]
    fn adaptive_rerank_boosts_selective_filters_within_configured_cap() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "adaptive_rerank_selective", SpindleMode::TurboProd);

        assert_eq!(core.compute_adaptive_rerank_oversampling(100, None), 8);
        assert_eq!(
            core.compute_adaptive_rerank_oversampling(100, Some(0.5)),
            12
        );
        assert_eq!(
            core.compute_adaptive_rerank_oversampling(100, Some(0.1)),
            16
        );
        assert_eq!(
            core.compute_adaptive_rerank_oversampling(100, Some(0.01)),
            16
        );
    }

    #[test]
    fn adaptive_rerank_respects_rescore_and_configured_oversampling() {
        let (env, _dir) = setup();
        let mut txn = env.write_txn().unwrap();
        let capped = VectorCore::new_named(
            &env,
            &mut txn,
            "adaptive_rerank_capped",
            HNSWConfig::new(Some(8), Some(32), Some(64)),
            DistanceMetric::Cosine,
            SpindleConfig {
                mode: SpindleMode::TurboProd,
                oversampling: 4,
                ..SpindleConfig::default()
            },
            test_backend(&env),
        )
        .unwrap();
        let no_rescore = VectorCore::new_named(
            &env,
            &mut txn,
            "adaptive_rerank_no_rescore",
            HNSWConfig::new(Some(8), Some(32), Some(64)),
            DistanceMetric::Cosine,
            SpindleConfig {
                mode: SpindleMode::TurboProd,
                rescore: false,
                ..SpindleConfig::default()
            },
            test_backend(&env),
        )
        .unwrap();
        txn.commit().unwrap();

        assert_eq!(
            capped.compute_adaptive_rerank_oversampling(100, Some(0.01)),
            4
        );
        assert_eq!(capped.compute_adaptive_rerank_oversampling(500, None), 4);
        assert_eq!(
            no_rescore.compute_adaptive_rerank_oversampling(10, Some(0.01)),
            1
        );
    }

    #[test]
    fn unfiltered_ef_override_keeps_k_floor() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "adaptive_unfiltered_cap", SpindleMode::None);
        insert_n(&env, &core, 50, 16);

        let txn = env.read_txn().unwrap();
        let query: Vec<f32> = vec![0.5; 16];
        let results = core
            .search_with_selectivity_ef::<VF>(
                &core.backend.read_borrowed(&txn),
                &query,
                10,
                None,
                false,
                None,
                Some(4),
            )
            .unwrap();

        assert_eq!(results.len(), 10);
    }

    #[test]
    fn id_filter_search_matches_hvector_filter() {
        let (env, dir) = setup();
        let core =
            core_with_mode_and_mmap(&env, dir.path(), "id_filter_parity", SpindleMode::None, 16);
        insert_n(&env, &core, 80, 16);
        core.flush_mmap().unwrap();

        let txn = env.read_txn().unwrap();
        let query: Vec<f32> = vec![0.5; 16];
        let hvector_filter = |v: &HVector| v.id % 2 == 0;
        let id_filter = |id: u128| id % 2 == 0;

        let hvector_results = core
            .search_with_selectivity_ef(
                &core.backend.read_borrowed(&txn),
                &query,
                10,
                Some(&[hvector_filter]),
                true,
                Some(0.5),
                None,
            )
            .unwrap();
        let id_results = core
            .search_with_id_filter_ef_observed(
                &core.backend.read_borrowed(&txn),
                &query,
                10,
                Some(&[id_filter]),
                true,
                Some(0.5),
                None,
                None,
            )
            .unwrap();

        assert!(!id_results.is_empty());
        assert!(id_results.iter().all(|v| v.id % 2 == 0));
        assert_eq!(
            hvector_results
                .iter()
                .map(HVector::get_id)
                .collect::<Vec<_>>(),
            id_results.iter().map(HVector::get_id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn id_filter_hnsw_traverses_rejected_neighbors_as_bridges() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "id_filter_bridge", SpindleMode::None);

        let mut txn = env.write_txn().unwrap();
        let v1 = core
            .insert_flat(&mut txn, &[1.0, 0.0, 0.0], Some(1), None)
            .unwrap();
        let _v2 = core
            .insert_flat(&mut txn, &[0.8, 0.2, 0.0], Some(2), None)
            .unwrap();
        let _v3 = core
            .insert_flat(&mut txn, &[0.0, 1.0, 0.0], Some(3), None)
            .unwrap();
        core.set_entry_point(&mut txn, &v1).unwrap();
        core.put_neighbor_ids(&mut txn, 1, 0, &[2]).unwrap();
        core.put_neighbor_ids(&mut txn, 2, 0, &[3]).unwrap();
        core.put_neighbor_ids(&mut txn, 3, 0, &[]).unwrap();
        txn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        assert!(core.has_index(&core.backend.read_borrowed(&txn)).unwrap());
        let only_three = |id: u128| id == 3;
        let results = core
            .search_with_id_filter_ef_observed(
                &core.backend.read_borrowed(&txn),
                &[0.0, 1.0, 0.0],
                1,
                Some(&[only_three]),
                true,
                Some(1.0 / 3.0),
                Some(4),
                None,
            )
            .unwrap();

        assert_eq!(results.iter().map(HVector::get_id).collect::<Vec<_>>(), [3]);
    }

    #[test]
    fn flat_id_filter_search_matches_hvector_filter_without_index() {
        let (env, dir) = setup();
        let core = core_with_mode_and_mmap(
            &env,
            dir.path(),
            "flat_id_filter_parity",
            SpindleMode::None,
            4,
        );

        let mut txn = env.write_txn().unwrap();
        for id in 1..=12u128 {
            let value = id as f32;
            let data = [value, (id % 5) as f32, 1.0, -1.0];
            core.insert_flat(&mut txn, &data, Some(id), None).unwrap();
        }
        txn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        assert!(
            !core.has_index(&core.backend.read_borrowed(&txn)).unwrap(),
            "insert_flat should leave this core in flat-scan mode"
        );
        let query = [4.5f32, 2.0, 1.0, -1.0];
        let hvector_filter = |v: &HVector| v.id % 3 == 0;
        let id_filter = |id: u128| id % 3 == 0;

        let hvector_results = core
            .search_flat(
                &core.backend.read_borrowed(&txn),
                &query,
                4,
                Some(&[hvector_filter]),
            )
            .unwrap();
        let id_results = core
            .search_with_id_filter_ef_observed(
                &core.backend.read_borrowed(&txn),
                &query,
                4,
                Some(&[id_filter]),
                true,
                Some(1.0 / 3.0),
                None,
                None,
            )
            .unwrap();

        assert!(!id_results.is_empty());
        assert!(id_results.iter().all(|v| v.id % 3 == 0));
        assert_eq!(
            hvector_results
                .iter()
                .map(HVector::get_id)
                .collect::<Vec<_>>(),
            id_results.iter().map(HVector::get_id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn search_with_selectivity_returns_results() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "sel_search", SpindleMode::None);
        insert_n(&env, &core, 50, 16);

        let txn = env.read_txn().unwrap();
        let query: Vec<f32> = vec![0.5; 16];

        // Unfiltered search with no selectivity hint
        let results_base = core
            .search_with_selectivity::<VF>(
                &core.backend.read_borrowed(&txn),
                &query,
                10,
                None,
                false,
                None,
            )
            .unwrap();
        assert_eq!(results_base.len(), 10);

        // Same search with 1% selectivity hint (should still return results,
        // just with higher internal ef)
        let results_adaptive = core
            .search_with_selectivity::<VF>(
                &core.backend.read_borrowed(&txn),
                &query,
                10,
                None,
                false,
                Some(0.01),
            )
            .unwrap();
        assert_eq!(results_adaptive.len(), 10);

        // Results should be identical for unfiltered (no filter applied, only ef differs)
        for (a, b) in results_base.iter().zip(results_adaptive.iter()) {
            assert_eq!(a.get_id(), b.get_id());
        }
    }

    #[test]
    fn search_with_selectivity_filtered_returns_valid_results() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "sel_filtered", SpindleMode::None);

        // Insert 200 vectors — enough for HNSW to reliably reach filtered nodes.
        let mut txn = env.write_txn().unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        for i in 1..=200u128 {
            let data: Vec<f32> = (0..16)
                .map(|_| rand::Rng::random_range(&mut rng, -1.0f32..1.0f32))
                .collect();
            core.insert::<VF>(&mut txn, &data, Some(i), None).unwrap();
        }
        txn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        let query: Vec<f32> = vec![0.5; 16];

        // Filter: only even ids (50% selectivity) — large enough for HNSW to find matches.
        let filter_fn = |v: &HVector| -> bool { v.get_id() % 2 == 0 };
        let results = core
            .search_with_selectivity(
                &core.backend.read_borrowed(&txn),
                &query,
                5,
                Some(&[filter_fn]),
                true,
                Some(0.5),
            )
            .unwrap();

        assert!(
            !results.is_empty(),
            "Filtered adaptive search returned no results"
        );
        for r in &results {
            assert!(r.get_id() % 2 == 0, "Expected even id, got {}", r.get_id());
        }
    }

    // ── Bulk HNSW builder tests ──

    fn insert_flat_n(env: &heed3::Env, core: &VectorCore, n: usize, dim: usize) {
        let mut txn = env.write_txn().unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for i in 0..n {
            let data: Vec<f32> = (0..dim)
                .map(|_| rand::Rng::random_range(&mut rng, -1.0f32..1.0f32))
                .collect();
            core.insert_flat(&mut txn, &data, Some((i + 1) as u128), None)
                .unwrap();
        }
        txn.commit().unwrap();
    }

    #[test]
    fn bulk_build_produces_searchable_index() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "bulk_search", SpindleMode::None);

        // Insert 500 vectors flat (no HNSW links)
        insert_flat_n(&env, &core, 500, 32);

        let txn = env.read_txn().unwrap();
        assert!(
            !core.has_index(&core.backend.read_borrowed(&txn)).unwrap(),
            "Should not have index before build"
        );
        drop(txn);

        // Build the bulk index
        let mut wtxn = env.write_txn().unwrap();
        core.build_index_from_flat(&mut wtxn).unwrap();
        wtxn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        assert!(
            core.has_index(&core.backend.read_borrowed(&txn)).unwrap(),
            "Should have index after build"
        );

        // Search and verify results are reasonable
        let query: Vec<f32> = vec![0.5; 32];
        let results = core
            .search::<VF>(&core.backend.read_borrowed(&txn), &query, 10, None, false)
            .unwrap();
        assert_eq!(results.len(), 10, "Should return exactly 10 results");

        // Verify results are sorted by distance (ascending)
        for i in 1..results.len() {
            assert!(
                results[i].get_distance() >= results[i - 1].get_distance(),
                "Results should be sorted by distance"
            );
        }

        // Verify we can find a specific vector by searching near it
        // Vector 1 was the first inserted; retrieve its data and search for it
        let v1 = core
            .get_vector(&core.backend.read_borrowed(&txn), 1, 0, true)
            .unwrap();
        let v1_results = core
            .search::<VF>(
                &core.backend.read_borrowed(&txn),
                v1.get_data(),
                5,
                None,
                false,
            )
            .unwrap();
        assert!(
            v1_results.iter().any(|r| r.get_id() == 1),
            "Searching for vector 1's own data should return vector 1"
        );
    }

    #[test]
    fn bulk_build_matches_incremental_quality() {
        let dim = 32;
        let n = 200;
        let k = 10;
        let query: Vec<f32> = vec![0.5; dim];

        // Build an incremental index
        let (env_inc, _dir_inc) = setup();
        let core_inc = core_with_mode(&env_inc, "inc", SpindleMode::None);
        insert_n(&env_inc, &core_inc, n, dim);

        let txn_inc = env_inc.read_txn().unwrap();
        let results_inc = core_inc
            .search::<VF>(
                &core_inc.backend.read_borrowed(&txn_inc),
                &query,
                k,
                None,
                false,
            )
            .unwrap();
        let inc_ids: HashSet<u128> = results_inc.iter().map(|r| r.get_id()).collect();
        drop(txn_inc);

        // Build a bulk index with the same data
        let (env_bulk, _dir_bulk) = setup();
        let core_bulk = core_with_mode(&env_bulk, "bulk", SpindleMode::None);
        insert_flat_n(&env_bulk, &core_bulk, n, dim);

        let mut wtxn = env_bulk.write_txn().unwrap();
        core_bulk.build_index_from_flat(&mut wtxn).unwrap();
        wtxn.commit().unwrap();

        let txn_bulk = env_bulk.read_txn().unwrap();
        let results_bulk = core_bulk
            .search::<VF>(
                &core_bulk.backend.read_borrowed(&txn_bulk),
                &query,
                k,
                None,
                false,
            )
            .unwrap();
        let bulk_ids: HashSet<u128> = results_bulk.iter().map(|r| r.get_id()).collect();

        // Both should return k results
        assert_eq!(results_inc.len(), k);
        assert_eq!(results_bulk.len(), k);

        // Recall: at least 50% overlap (HNSW is approximate, different build
        // orders yield slightly different graphs, but quality should be comparable)
        let overlap = inc_ids.intersection(&bulk_ids).count();
        assert!(
            overlap >= k / 2,
            "Bulk build recall too low: only {}/{} overlap with incremental",
            overlap,
            k
        );
    }

    #[test]
    fn bulk_build_matches_incremental_quality_with_compressed_vectors() {
        let dim = 64;
        let n = 240;
        let k = 10;
        let query: Vec<f32> = vec![0.25; dim];

        let (env_inc, _dir_inc) = setup();
        let core_inc = core_with_mode(&env_inc, "inc_turbo", SpindleMode::TurboProd);
        insert_n(&env_inc, &core_inc, n, dim);

        let txn_inc = env_inc.read_txn().unwrap();
        let results_inc = core_inc
            .search::<VF>(
                &core_inc.backend.read_borrowed(&txn_inc),
                &query,
                k,
                None,
                false,
            )
            .unwrap();
        let inc_ids: HashSet<u128> = results_inc.iter().map(|r| r.get_id()).collect();
        drop(txn_inc);

        let (env_bulk, _dir_bulk) = setup();
        let core_bulk = core_with_mode(&env_bulk, "bulk_turbo", SpindleMode::TurboProd);
        insert_flat_n(&env_bulk, &core_bulk, n, dim);

        let mut wtxn = env_bulk.write_txn().unwrap();
        core_bulk.build_index_from_flat(&mut wtxn).unwrap();
        wtxn.commit().unwrap();

        let txn_bulk = env_bulk.read_txn().unwrap();
        let results_bulk = core_bulk
            .search::<VF>(
                &core_bulk.backend.read_borrowed(&txn_bulk),
                &query,
                k,
                None,
                false,
            )
            .unwrap();
        let bulk_ids: HashSet<u128> = results_bulk.iter().map(|r| r.get_id()).collect();

        assert_eq!(results_inc.len(), k);
        assert_eq!(results_bulk.len(), k);

        let overlap = inc_ids.intersection(&bulk_ids).count();
        assert!(
            overlap >= k / 2,
            "Compressed bulk build recall too low: only {}/{} overlap with incremental",
            overlap,
            k
        );
    }

    #[test]
    fn bulk_build_empty_is_noop() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "bulk_empty", SpindleMode::None);

        // No vectors inserted — build should succeed without crash
        let mut wtxn = env.write_txn().unwrap();
        core.build_index_from_flat(&mut wtxn).unwrap();
        wtxn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        assert!(
            !core.has_index(&core.backend.read_borrowed(&txn)).unwrap(),
            "Empty build should not create index"
        );
    }

    #[test]
    fn bulk_build_idempotent() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "bulk_idem", SpindleMode::None);
        insert_flat_n(&env, &core, 100, 16);

        // First build
        let mut wtxn = env.write_txn().unwrap();
        core.build_index_from_flat(&mut wtxn).unwrap();
        wtxn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        let query: Vec<f32> = vec![0.5; 16];
        let results1 = core
            .search::<VF>(&core.backend.read_borrowed(&txn), &query, 10, None, false)
            .unwrap();
        let ids1: Vec<u128> = results1.iter().map(|r| r.get_id()).collect();
        drop(txn);

        // Second build — should be a no-op (has_index returns true)
        let mut wtxn = env.write_txn().unwrap();
        core.build_index_from_flat(&mut wtxn).unwrap();
        wtxn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        let results2 = core
            .search::<VF>(&core.backend.read_borrowed(&txn), &query, 10, None, false)
            .unwrap();
        let ids2: Vec<u128> = results2.iter().map(|r| r.get_id()).collect();

        // Results should be identical
        assert_eq!(ids1, ids2, "Second build should not change search results");
    }

    /// Split-build topology for `rows` (id i+1 -> rows[i]) under `metric`.
    fn split_build_level0(
        name: &str,
        metric: DistanceMetric,
        m: usize,
        rows: &[Vec<f32>],
    ) -> Vec<(u128, Vec<u128>)> {
        let (env, _dir) = setup();
        let core = core_with_config(
            &env,
            name,
            HNSWConfig::new(Some(m), Some(64), Some(64)),
            metric,
        );
        let mut txn = env.write_txn().unwrap();
        for (i, row) in rows.iter().enumerate() {
            core.insert_flat(&mut txn, row, Some((i + 1) as u128), None)
                .unwrap();
        }
        txn.commit().unwrap();
        let txn = env.read_txn().unwrap();
        let prepared = core
            .prepare_index_from_flat(&core.backend.read_borrowed(&txn))
            .unwrap()
            .unwrap();
        prepared
            .adjacency
            .iter()
            .enumerate()
            .map(|(ord, adjacency)| {
                let neighbors = adjacency.as_ref().unwrap()[0]
                    .iter()
                    .map(|&other| prepared.point_ids[other as usize])
                    .collect();
                (prepared.point_ids[ord], neighbors)
            })
            .collect()
    }

    #[test]
    fn split_build_uses_raw_geometry_for_dot_and_euclid() {
        // Dot on the unit circle (see `dot_circle_rows`).
        let n = 200usize;
        let (dot_positions, rows) = dot_circle_rows(29, n);
        let level0 = split_build_level0("split_dot", DistanceMetric::Dot, 4, &rows)
            .into_iter()
            .map(|(id, neighbors)| {
                let pos = |id: u128| dot_positions[id as usize - 1];
                (pos(id), neighbors.into_iter().map(pos).collect())
            });
        let pct = circle_locality_percent(n, 16, level0);
        assert!(
            pct >= 90,
            "split Dot build must follow inner products: {pct}% local"
        );
        let mut rng = rand::rngs::StdRng::seed_from_u64(31);

        // Euclid with unequal ranges: dim 0 spans [0, 2000), dim 1 [0, 1).
        let mut positions: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rand::Rng::random_range(&mut rng, 0..=i);
            positions.swap(i, j);
        }
        let rows: Vec<Vec<f32>> = positions
            .iter()
            .map(|&x| {
                vec![
                    x as f32 * 10.0,
                    rand::Rng::random_range(&mut rng, 0.0f32..1.0f32),
                ]
            })
            .collect();
        let pos_of = |id: u128| positions[id as usize - 1] as i64;
        let bound = 16i64;
        let local = split_build_level0("split_l2", DistanceMetric::Euclid, 4, &rows)
            .into_iter()
            .filter(|(id, neighbors)| {
                neighbors
                    .iter()
                    .all(|&other| (pos_of(other) - pos_of(*id)).abs() <= bound)
            })
            .count();
        // Level assignment is unseeded, so locality varies a few points run
        // to run (89% seen in a full-suite run).
        assert!(
            local * 100 >= n * 85,
            "split Euclid build must follow raw L2 geometry: {local}/{n} local"
        );
    }

    #[test]
    fn split_phase_build_produces_searchable_index() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "split_build", SpindleMode::None);
        insert_flat_n(&env, &core, 500, 32);

        let txn = env.read_txn().unwrap();
        assert!(!core.has_index(&core.backend.read_borrowed(&txn)).unwrap());

        // Phase 1: prepare index from read txn (no write lock)
        let prepared = core
            .prepare_index_from_flat(&core.backend.read_borrowed(&txn))
            .unwrap();
        assert!(prepared.is_some(), "Should produce a prepared index");
        drop(txn);

        // Phase 2: flush to LMDB (write lock, but fast)
        let mut wtxn = env.write_txn().unwrap();
        core.flush_prepared_index(&mut wtxn, prepared.unwrap())
            .unwrap();
        wtxn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        assert!(
            core.has_index(&core.backend.read_borrowed(&txn)).unwrap(),
            "Should have index after flush"
        );

        let query: Vec<f32> = vec![0.5; 32];
        let results = core
            .search::<VF>(&core.backend.read_borrowed(&txn), &query, 10, None, false)
            .unwrap();
        assert_eq!(results.len(), 10, "Should return 10 results");

        // Verify results sorted by distance
        for i in 1..results.len() {
            assert!(results[i].get_distance() >= results[i - 1].get_distance());
        }
    }

    #[test]
    fn split_phase_build_matches_monolithic_build_quality() {
        let dim = 32;
        let n = 200;
        let k = 10;
        let query: Vec<f32> = vec![0.5; dim];

        // Monolithic build
        let (env_mono, _dir_mono) = setup();
        let core_mono = core_with_mode(&env_mono, "mono", SpindleMode::None);
        insert_flat_n(&env_mono, &core_mono, n, dim);
        let mut wtxn = env_mono.write_txn().unwrap();
        core_mono.build_index_from_flat(&mut wtxn).unwrap();
        wtxn.commit().unwrap();

        let txn = env_mono.read_txn().unwrap();
        let results_mono = core_mono
            .search::<VF>(
                &core_mono.backend.read_borrowed(&txn),
                &query,
                k,
                None,
                false,
            )
            .unwrap();
        let mono_ids: HashSet<u128> = results_mono.iter().map(|r| r.get_id()).collect();
        drop(txn);

        // Split-phase build
        let (env_split, _dir_split) = setup();
        let core_split = core_with_mode(&env_split, "split", SpindleMode::None);
        insert_flat_n(&env_split, &core_split, n, dim);

        let rtxn = env_split.read_txn().unwrap();
        let prepared = core_split
            .prepare_index_from_flat(&core_split.backend.read_borrowed(&rtxn))
            .unwrap()
            .unwrap();
        drop(rtxn);

        let mut wtxn = env_split.write_txn().unwrap();
        core_split
            .flush_prepared_index(&mut wtxn, prepared)
            .unwrap();
        wtxn.commit().unwrap();

        let txn = env_split.read_txn().unwrap();
        let results_split = core_split
            .search::<VF>(
                &core_split.backend.read_borrowed(&txn),
                &query,
                k,
                None,
                false,
            )
            .unwrap();
        let split_ids: HashSet<u128> = results_split.iter().map(|r| r.get_id()).collect();

        assert_eq!(results_mono.len(), k);
        assert_eq!(results_split.len(), k);

        // Both are approximate, so we just check reasonable overlap
        // (different RNG seeds for level assignment means graphs differ)
        let overlap = mono_ids.intersection(&split_ids).count();
        assert!(
            overlap >= k / 3,
            "Split build recall too low vs monolithic: {}/{} overlap",
            overlap,
            k
        );
    }

    #[test]
    fn build_hnsw_in_memory_returns_well_formed_prepared_index() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(999);
        let exported: Vec<(u128, Vec<f32>, HashMap<String, Value>)> = (1..=64u128)
            .map(|id| {
                let data: Vec<f32> = (0..16)
                    .map(|_| rand::Rng::random_range(&mut rng, -1.0f32..1.0f32))
                    .collect();
                (id, data, HashMap::new())
            })
            .collect();

        let config = HNSWConfig::new(Some(8), Some(32), Some(64));
        let prepared =
            VectorCore::build_hnsw_in_memory(&exported, &config, DistanceMetric::Cosine).unwrap();

        assert_eq!(prepared.point_ids.len(), exported.len());
        assert_eq!(prepared.original_data.len(), exported.len());
        assert_eq!(prepared.point_fields.len(), exported.len());
        assert_eq!(prepared.levels.len(), exported.len());
        assert_eq!(prepared.adjacency.len(), exported.len());
        assert!(prepared.entry_ord < exported.len());

        for i in 0..prepared.point_ids.len() {
            let adjacency = prepared.adjacency[i].as_ref().unwrap();
            assert_eq!(adjacency.len(), prepared.levels[i] + 1);
            for (level, neighbors) in adjacency.iter().enumerate() {
                let max_neighbors = if level == 0 { config.m_max_0 } else { config.m };
                assert!(neighbors.len() <= max_neighbors);
                assert!(!neighbors.contains(&(i as u32)));
            }
        }
    }

    #[test]
    fn flash_sq8_build_recall_vs_brute_force() {
        // Flash build (SQ8) at realistic dimension, verify recall against brute-force
        let dim = 128; // representative; 768 would be slow in debug mode
        let n = 500;
        let k = 10;
        let num_queries = 20;

        let (env, _dir) = setup();
        let core = core_with_mode(&env, "flash_recall", SpindleMode::None);
        insert_flat_n(&env, &core, n, dim);

        // Build using split-phase (which now uses SQ8 internally)
        let rtxn = env.read_txn().unwrap();
        let prepared = core
            .prepare_index_from_flat(&core.backend.read_borrowed(&rtxn))
            .unwrap()
            .unwrap();
        // Collect all vectors for brute-force
        let all_vectors: Vec<(u128, Vec<f32>)> = core
            .get_all_vectors(&core.backend.read_borrowed(&rtxn), Some(0))
            .unwrap()
            .into_iter()
            .map(|v| {
                let data = core
                    .get_vector_data(&core.backend.read_borrowed(&rtxn), v.get_id())
                    .unwrap();
                let raw = data
                    .original_vector
                    .unwrap_or_else(|| v.get_data().to_vec());
                (v.get_id(), raw)
            })
            .collect();
        drop(rtxn);

        let mut wtxn = env.write_txn().unwrap();
        core.flush_prepared_index(&mut wtxn, prepared).unwrap();
        wtxn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        assert!(core.has_index(&core.backend.read_borrowed(&txn)).unwrap());

        let mut total_overlap = 0;
        let mut total_k = 0;
        for qi in 0..num_queries {
            let query: Vec<f32> = (0..dim)
                .map(|j| ((qi * 17 + j * 31) as f32 * 0.01).sin())
                .collect();
            let brute = brute_force_top_k(&all_vectors, &query, k);
            let results = core
                .search::<VF>(&core.backend.read_borrowed(&txn), &query, k, None, false)
                .unwrap();
            let hnsw_ids: HashSet<u128> = results.iter().map(|r| r.get_id()).collect();
            let brute_ids: HashSet<u128> = brute.into_iter().collect();
            total_overlap += hnsw_ids.intersection(&brute_ids).count();
            total_k += k;
        }
        let recall = total_overlap as f64 / total_k as f64;
        assert!(
            recall >= 0.5,
            "Flash SQ8 build recall too low vs brute-force: {:.1}% ({}/{})",
            recall * 100.0,
            total_overlap,
            total_k
        );
    }

    // ── Recall quality tests ──

    /// Brute-force top-k by cosine distance for ground-truth computation.
    fn brute_force_top_k(vectors: &[(u128, Vec<f32>)], query: &[f32], k: usize) -> Vec<u128> {
        use crate::helix_engine::vector_core::simd;
        let mut scored: Vec<(u128, f32)> = vectors
            .iter()
            .map(|(id, data)| (*id, simd::cosine_f32(query, data)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        scored.into_iter().take(k).map(|(id, _)| id).collect()
    }

    #[test]
    fn hnsw_recall_vs_brute_force_above_threshold() {
        let dim = 64;
        let n = 500;
        let k = 10;
        let num_queries = 20;
        let min_recall = 0.70; // HNSW should get at least 70% recall

        let (env, _dir) = setup();
        let core = core_with_mode(&env, "recall_test", SpindleMode::None);

        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let mut all_vectors: Vec<(u128, Vec<f32>)> = Vec::with_capacity(n);

        let mut txn = env.write_txn().unwrap();
        for i in 1..=n {
            let data: Vec<f32> = (0..dim)
                .map(|_| rand::Rng::random_range(&mut rng, -1.0f32..1.0f32))
                .collect();
            core.insert::<VF>(&mut txn, &data, Some(i as u128), None)
                .unwrap();
            all_vectors.push((i as u128, data));
        }
        txn.commit().unwrap();

        let queries: Vec<Vec<f32>> = (0..num_queries)
            .map(|_| {
                (0..dim)
                    .map(|_| rand::Rng::random_range(&mut rng, -1.0f32..1.0f32))
                    .collect()
            })
            .collect();

        let txn = env.read_txn().unwrap();
        let mut total_recall = 0.0;
        for query in &queries {
            let gt: HashSet<u128> = brute_force_top_k(&all_vectors, query, k)
                .into_iter()
                .collect();
            let results = core
                .search::<VF>(&core.backend.read_borrowed(&txn), query, k, None, false)
                .unwrap();
            let result_ids: HashSet<u128> = results.iter().map(|r| r.get_id()).collect();
            let overlap = gt.intersection(&result_ids).count() as f64 / k as f64;
            total_recall += overlap;
        }
        let avg_recall = total_recall / num_queries as f64;
        assert!(
            avg_recall >= min_recall,
            "HNSW recall {avg_recall:.3} below threshold {min_recall:.3}"
        );
    }

    #[test]
    fn bulk_build_recall_vs_brute_force_above_threshold() {
        let dim = 64;
        let n = 500;
        let k = 10;
        let num_queries = 20;
        let min_recall = 0.70;

        let (env, _dir) = setup();
        let core = core_with_mode(&env, "bulk_recall", SpindleMode::None);

        let mut rng = rand::rngs::StdRng::seed_from_u64(456);
        let mut all_vectors: Vec<(u128, Vec<f32>)> = Vec::with_capacity(n);

        let mut txn = env.write_txn().unwrap();
        for i in 1..=n {
            let data: Vec<f32> = (0..dim)
                .map(|_| rand::Rng::random_range(&mut rng, -1.0f32..1.0f32))
                .collect();
            core.insert_flat(&mut txn, &data, Some(i as u128), None)
                .unwrap();
            all_vectors.push((i as u128, data));
        }
        core.build_index_from_flat(&mut txn).unwrap();
        txn.commit().unwrap();

        let queries: Vec<Vec<f32>> = (0..num_queries)
            .map(|_| {
                (0..dim)
                    .map(|_| rand::Rng::random_range(&mut rng, -1.0f32..1.0f32))
                    .collect()
            })
            .collect();

        let txn = env.read_txn().unwrap();
        let mut total_recall = 0.0;
        for query in &queries {
            let gt: HashSet<u128> = brute_force_top_k(&all_vectors, query, k)
                .into_iter()
                .collect();
            let results = core
                .search::<VF>(&core.backend.read_borrowed(&txn), query, k, None, false)
                .unwrap();
            let result_ids: HashSet<u128> = results.iter().map(|r| r.get_id()).collect();
            let overlap = gt.intersection(&result_ids).count() as f64 / k as f64;
            total_recall += overlap;
        }
        let avg_recall = total_recall / num_queries as f64;
        assert!(
            avg_recall >= min_recall,
            "Bulk-build recall {avg_recall:.3} below threshold {min_recall:.3}"
        );
    }

    #[test]
    fn vsag_pruning_preserves_recall_vs_ids_only_same_graph() {
        // Build one graph with level-0 sidecar codes enabled, then rewrite the same
        // level-0 neighbor blocks to ids-only form. This compares pruning-on versus
        // pruning-off on the exact same graph instead of conflating pruning quality
        // with differences in graph construction order.
        let dim = 128; // high enough for neighbor_code_config to activate
        let n = 300;
        let k = 10;
        let num_queries = 15;
        let max_recall_drop = 0.10;

        let (env, _dir) = setup();
        let mut txn = env.write_txn().unwrap();
        let core = VectorCore::new_named(
            &env,
            &mut txn,
            "vsag_prune",
            HNSWConfig::new(Some(16), Some(64), Some(128)),
            DistanceMetric::Cosine,
            SpindleConfig {
                mode: SpindleMode::None,
                ..SpindleConfig::default()
            },
            test_backend(&env),
        )
        .unwrap();
        txn.commit().unwrap();

        let mut rng = rand::rngs::StdRng::seed_from_u64(789);
        let mut all_vectors: Vec<(u128, Vec<f32>)> = Vec::with_capacity(n);
        let mut txn = env.write_txn().unwrap();
        for i in 1..=n {
            let data: Vec<f32> = (0..dim)
                .map(|_| rand::Rng::random_range(&mut rng, -1.0f32..1.0f32))
                .collect();
            core.insert::<VF>(&mut txn, &data, Some(i as u128), None)
                .unwrap();
            all_vectors.push((i as u128, data));
        }
        txn.commit().unwrap();

        let queries: Vec<Vec<f32>> = (0..num_queries)
            .map(|_| {
                (0..dim)
                    .map(|_| rand::Rng::random_range(&mut rng, -1.0f32..1.0f32))
                    .collect()
            })
            .collect();

        let measure_avg_recall = |txn: &RoTxn| -> f64 {
            let mut total_recall = 0.0;
            for query in &queries {
                let gt: HashSet<u128> = brute_force_top_k(&all_vectors, query, k)
                    .into_iter()
                    .collect();
                let results = core
                    .search::<VF>(&core.backend.read_borrowed(&txn), query, k, None, false)
                    .unwrap();
                let result_ids: HashSet<u128> = results.iter().map(|r| r.get_id()).collect();
                total_recall += gt.intersection(&result_ids).count() as f64 / k as f64;
            }
            total_recall / num_queries as f64
        };

        let avg_recall_with_pruning = {
            let txn = env.read_txn().unwrap();
            measure_avg_recall(&txn)
        };

        let ids_only_blocks: Vec<(u128, Vec<u128>)> = {
            let txn = env.read_txn().unwrap();
            (1..=n as u128)
                .filter_map(|id| {
                    let block = core
                        .get_neighbor_block(&core.backend.read_borrowed(&txn), id, 0)
                        .unwrap();
                    if block.ids.is_empty() || !block.has_approx_codes() {
                        None
                    } else {
                        Some((id, block.ids))
                    }
                })
                .collect()
        };
        assert!(
            !ids_only_blocks.is_empty(),
            "Expected at least one level-0 block with sidecar codes"
        );

        let mut txn = env.write_txn().unwrap();
        for (id, neighbors) in &ids_only_blocks {
            core.put_neighbor_block(
                &mut txn,
                *id,
                0,
                &NeighborBlock::from_ids(neighbors.clone()),
            )
            .unwrap();
        }
        txn.commit().unwrap();

        let avg_recall_without_pruning = {
            let txn = env.read_txn().unwrap();
            measure_avg_recall(&txn)
        };

        assert!(
            avg_recall_with_pruning + max_recall_drop >= avg_recall_without_pruning,
            "VSAG pruning recall regressed too far: pruning={avg_recall_with_pruning:.3}, ids_only={avg_recall_without_pruning:.3}, max_drop={max_recall_drop:.3}"
        );
        assert!(
            avg_recall_with_pruning >= 0.60,
            "VSAG pruning killed recall: {avg_recall_with_pruning:.3} < 0.60"
        );
    }

    #[test]
    fn nav_cache_invalidated_after_delete() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "nav_cache_del", SpindleMode::None);
        insert_n(&env, &core, 30, 16);

        // Prime the nav cache
        let txn = env.read_txn().unwrap();
        let ep = core
            .get_entry_point(&core.backend.read_borrowed(&txn))
            .unwrap();
        core.ensure_navigation_cache(&core.backend.read_borrowed(&txn), &ep);
        drop(txn);

        // Delete entry point — caches should be cleared
        let ep_id = {
            let txn = env.read_txn().unwrap();
            let rd = core.backend.read_borrowed(&txn);
            core.get_entry_point(&rd).unwrap().get_id()
        };
        let mut wtxn = env.write_txn().unwrap();
        core.delete_vector(&mut wtxn, ep_id).unwrap();
        wtxn.commit().unwrap();

        // Search should still work — nav cache must not serve stale ep
        let txn = env.read_txn().unwrap();
        let query: Vec<f32> = vec![0.5; 16];
        let results = core
            .search::<VF>(&core.backend.read_borrowed(&txn), &query, 5, None, false)
            .unwrap();
        assert!(!results.is_empty());
        assert!(
            !results.iter().any(|r| r.get_id() == ep_id),
            "Deleted entry point still in results after cache should be cleared"
        );
    }

    #[test]
    fn bulk_build_skips_sidecars_for_speed() {
        // Bulk build writes plain neighbor blocks (no sidecar codes) to keep
        // ingest fast. Sidecars are only generated during incremental insert.
        let dim = 64;
        let n = 50;

        let (env, _dir) = setup();
        let core = core_with_mode(&env, "sidecar_check", SpindleMode::None);

        let mut rng = rand::rngs::StdRng::seed_from_u64(321);
        let mut txn = env.write_txn().unwrap();
        for i in 1..=n {
            let data: Vec<f32> = (0..dim)
                .map(|_| rand::Rng::random_range(&mut rng, -1.0f32..1.0f32))
                .collect();
            core.insert_flat(&mut txn, &data, Some(i as u128), None)
                .unwrap();
        }
        core.build_index_from_flat(&mut txn).unwrap();
        txn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        // Bulk-built blocks should NOT have sidecar codes (speed optimization)
        for id in 1..=n as u128 {
            let block = core
                .get_neighbor_block(&core.backend.read_borrowed(&txn), id, 0)
                .unwrap();
            if block.ids.is_empty() {
                continue;
            }
            assert!(
                !block.has_approx_codes(),
                "Bulk-built block for node {id} should not have sidecar codes"
            );
        }
        // But search should still work fine without sidecars
        let query: Vec<f32> = vec![0.5; dim];
        let results = core
            .search::<VF>(&core.backend.read_borrowed(&txn), &query, 10, None, false)
            .unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn incremental_insert_generates_sidecars() {
        // Incremental insert (via put_neighbor_ids → build_neighbor_block)
        // should generate sidecar codes when vectors are available.
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "incr_sidecars", SpindleMode::None);

        let mut txn = env.write_txn().unwrap();
        core.put_raw_vector(&mut txn, 22, 0, &[1.0, 0.0, 0.0, 0.0])
            .unwrap();
        core.put_raw_vector(&mut txn, 33, 0, &[0.0, 1.0, 0.0, 0.0])
            .unwrap();
        core.put_raw_vector(&mut txn, 44, 0, &[0.0, 0.0, 1.0, 0.0])
            .unwrap();
        core.put_neighbor_ids(&mut txn, 11, 0, &[22, 33, 44])
            .unwrap();
        txn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        let block = core
            .get_neighbor_block(&core.backend.read_borrowed(&txn), 11, 0)
            .unwrap();
        assert_eq!(block.ids, vec![22, 33, 44]);
        assert!(
            block.has_approx_codes(),
            "Incremental insert should generate sidecar codes"
        );
        for i in 0..block.ids.len() {
            assert!(block.code_at(i).is_some());
        }
    }

    #[test]
    fn mmap_marker_roundtrip_reads_real_vector_data() {
        let (env, dir) = setup();
        let core = core_with_mode_and_mmap(&env, dir.path(), "mmap_marker", SpindleMode::None, 4);

        let data = vec![1.0f32, -2.0, 3.0, -4.0];
        let mut txn = env.write_txn().unwrap();
        core.insert_flat(&mut txn, &data, Some(7), None).unwrap();
        txn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        let stored = core
            .vectors_db
            .as_ref()
            .unwrap()
            .get(&txn, VectorCore::vector_key(7, 0).as_ref())
            .unwrap()
            .unwrap();
        assert!(
            stored.is_empty(),
            "externalized vectors should store an LMDB marker"
        );
        assert!(core
            .ordinals_db
            .as_ref()
            .unwrap()
            .get(&txn, &7u128.to_be_bytes())
            .unwrap()
            .is_some());
        assert_eq!(
            core.get_vector(&core.backend.read_borrowed(&txn), 7, 0, true)
                .unwrap()
                .get_data(),
            &data[..]
        );
        assert_eq!(
            decode_vector(
                &core
                    .get_encoded_vector(&core.backend.read_borrowed(&txn), 7, 0)
                    .unwrap()
            )
            .unwrap(),
            data
        );
        let all = core
            .get_all_vectors(&core.backend.read_borrowed(&txn), Some(0))
            .unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].get_data(), &data[..]);
    }

    #[test]
    fn compact_spindle_does_not_create_raw_hvec_sidecar() {
        let (env, dir) = setup();
        let core = core_with_spindle_and_mmap(
            &env,
            dir.path(),
            "compact_mmap",
            SpindleConfig {
                mode: SpindleMode::ScalarInt8,
                keep_original: false,
                rescore: false,
                oversampling: 1,
                ..SpindleConfig::default()
            },
            4,
        );

        let data = vec![1.0f32, -2.0, 3.0, -4.0];
        let mut txn = env.write_txn().unwrap();
        core.insert_flat(&mut txn, &data, Some(7), None).unwrap();
        txn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        assert!(core
            .ordinals_db
            .as_ref()
            .unwrap()
            .get(&txn, &7u128.to_be_bytes())
            .unwrap()
            .is_none());
        assert_eq!(
            core.get_vector(&core.backend.read_borrowed(&txn), 7, 0, true)
                .unwrap()
                .get_data()
                .len(),
            4
        );

        assert!(
            !dir.path().join("compact_mmap.hvec").exists(),
            "compact spindle must not create a raw f32 sidecar"
        );
        assert!(
            !dir.path().join("compact_mmap.hvs8").exists(),
            "mutable compact spindle segments defer HVS8 until publish"
        );
    }

    #[test]
    fn compact_spindle_materializes_hvs8_sidecar_from_prepared_index() {
        let (env, dir) = setup();
        let spindle = SpindleConfig {
            mode: SpindleMode::ScalarInt8,
            keep_original: false,
            rescore: false,
            oversampling: 1,
            ..SpindleConfig::default()
        };
        let core =
            core_with_spindle_and_mmap(&env, dir.path(), "compact_publish", spindle.clone(), 4);

        let rows = [
            (11u128, vec![0.10, 0.20, 0.30, 0.40]),
            (12u128, vec![0.90, 0.80, 0.70, 0.60]),
            (13u128, vec![-0.20, 0.15, -0.35, 0.55]),
            (14u128, vec![0.05, -0.75, 0.25, -0.45]),
        ];
        let mut txn = env.write_txn().unwrap();
        for (id, data) in &rows {
            core.insert_flat(&mut txn, data, Some(*id), None).unwrap();
        }
        txn.commit().unwrap();

        let hvec = dir.path().join("compact_publish.hvec");
        let hvs8 = dir.path().join("compact_publish.hvs8");
        assert!(
            !hvec.exists(),
            "compact mutable writes should not create HVEC"
        );
        assert!(!hvs8.exists(), "HVS8 is created only at publish time");

        let rtxn = env.read_txn().unwrap();
        let mut prepared = core
            .prepare_index_from_flat(&core.backend.read_borrowed(&rtxn))
            .unwrap()
            .unwrap();
        drop(rtxn);

        assert!(core
            .materialize_hvs8_sidecar_from_prepared(&prepared)
            .unwrap());
        assert!(
            hvs8.exists(),
            "publish should create an immutable HVS8 sidecar"
        );
        assert!(
            !hvec.exists(),
            "publish must not leave behind an HVEC sidecar"
        );
        assert!(core.mmap_is_quantized());

        let total = prepared.point_ids.len();
        let mut wtxn = env.write_txn().unwrap();
        core.flush_prepared_index_chunk(&mut wtxn, &mut prepared, 0, total)
            .unwrap();
        core.finalize_prepared_index_entry(&mut wtxn, &prepared)
            .unwrap();
        wtxn.commit().unwrap();

        let mut wtxn = env.write_txn().unwrap();
        core.write_prepared_sidecar_ordinals(&mut wtxn, &prepared)
            .unwrap();
        wtxn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        assert!(core.has_index(&core.backend.read_borrowed(&txn)).unwrap());
        for (id, _) in &rows {
            assert!(core
                .ordinals_db
                .as_ref()
                .unwrap()
                .get(&txn, &id.to_be_bytes())
                .unwrap()
                .is_some());
            assert_eq!(
                core.get_vector(&core.backend.read_borrowed(&txn), *id, 0, true)
                    .unwrap()
                    .get_data()
                    .len(),
                4
            );
        }
        assert_eq!(
            std::fs::metadata(&hvs8).unwrap().len(),
            (24 + 4 * 8 + rows.len() * 4) as u64
        );
        let results = core
            .search::<VF>(
                &core.backend.read_borrowed(&txn),
                &[0.10, 0.20, 0.30, 0.40],
                2,
                None,
                false,
            )
            .unwrap();
        assert_eq!(results.len(), 2);
        drop(txn);

        let reopened = core_with_spindle_and_mmap(&env, dir.path(), "compact_publish", spindle, 4);
        assert!(reopened.mmap_is_quantized());
        let txn = env.read_txn().unwrap();
        assert_eq!(
            reopened
                .get_vector(&reopened.backend.read_borrowed(&txn), 11, 0, true)
                .unwrap()
                .get_data()
                .len(),
            4
        );
        assert_eq!(
            reopened
                .search::<VF>(
                    &reopened.backend.read_borrowed(&txn),
                    &[0.10, 0.20, 0.30, 0.40],
                    2,
                    None,
                    false
                )
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn compact_turbo_prod_materializes_hvtq_sidecar_from_prepared_index() {
        let (env, dir) = setup();
        let spindle = SpindleConfig::turbo_prod_compact(4);
        let core =
            core_with_spindle_and_mmap(&env, dir.path(), "compact_tq_publish", spindle.clone(), 4);

        let rows = [
            (11u128, vec![0.10, 0.20, 0.30, 0.40]),
            (12u128, vec![0.90, 0.80, 0.70, 0.60]),
            (13u128, vec![-0.20, 0.15, -0.35, 0.55]),
            (14u128, vec![0.05, -0.75, 0.25, -0.45]),
        ];
        let mut txn = env.write_txn().unwrap();
        for (id, data) in &rows {
            core.insert_flat(&mut txn, data, Some(*id), None).unwrap();
        }
        txn.commit().unwrap();

        let hvec = dir.path().join("compact_tq_publish.hvec");
        let hvs8 = dir.path().join("compact_tq_publish.hvs8");
        let hvtq = dir.path().join("compact_tq_publish.hvtq");
        assert!(!hvec.exists(), "compact TurboProd should not create HVEC");
        assert!(!hvs8.exists(), "TQ path should not pre-create HVS8");
        assert!(!hvtq.exists(), "HVTQ is created only at publish time");

        let rtxn = env.read_txn().unwrap();
        let encoded_before = core
            .get_encoded_vector(&core.backend.read_borrowed(&rtxn), 11, 0)
            .unwrap();
        let mut prepared = core
            .prepare_index_from_flat(&core.backend.read_borrowed(&rtxn))
            .unwrap()
            .unwrap();
        drop(rtxn);

        assert!(core
            .materialize_tq_sidecar_from_prepared(&prepared)
            .unwrap());
        assert!(hvtq.exists(), "publish should create an HVTQ sidecar");
        assert!(!hvs8.exists(), "TQ publish must not add an HVS8 sidecar");
        assert!(!hvec.exists(), "TQ publish must not add an HVEC sidecar");
        assert!(core.mmap_is_turbo_quantized());

        let total = prepared.point_ids.len();
        let mut wtxn = env.write_txn().unwrap();
        core.write_prepared_sidecar_ordinals(&mut wtxn, &prepared)
            .unwrap();
        wtxn.commit().unwrap();

        let mut wtxn = env.write_txn().unwrap();
        core.flush_prepared_index_chunk(&mut wtxn, &mut prepared, 0, total)
            .unwrap();
        core.finalize_prepared_index_entry(&mut wtxn, &prepared)
            .unwrap();
        wtxn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        assert!(core.has_index(&core.backend.read_borrowed(&txn)).unwrap());
        assert_eq!(
            core.vectors_db
                .as_ref()
                .unwrap()
                .get(&txn, VectorCore::vector_key(11, 0).as_ref())
                .unwrap()
                .unwrap(),
            EXTERNALIZED_VECTOR_MARKER,
            "LMDB vector row should be a marker once HVTQ owns the payload"
        );
        assert_eq!(
            core.get_encoded_vector(&core.backend.read_borrowed(&txn), 11, 0)
                .unwrap(),
            encoded_before,
            "HVTQ should copy existing TurboProd payload bytes, not decode/re-encode"
        );
        for (id, _) in &rows {
            assert!(core
                .ordinals_db
                .as_ref()
                .unwrap()
                .get(&txn, &id.to_be_bytes())
                .unwrap()
                .is_some());
            assert_eq!(
                core.get_vector(&core.backend.read_borrowed(&txn), *id, 0, true)
                    .unwrap()
                    .get_data()
                    .len(),
                4
            );
        }
        assert_eq!(
            std::fs::metadata(&hvtq).unwrap().len(),
            (32 + rows.len() * encoded_before.len()) as u64
        );
        let results = core
            .search::<VF>(
                &core.backend.read_borrowed(&txn),
                &[0.10, 0.20, 0.30, 0.40],
                2,
                None,
                false,
            )
            .unwrap();
        assert_eq!(results.len(), 2);
        drop(txn);

        let reopened =
            core_with_spindle_and_mmap(&env, dir.path(), "compact_tq_publish", spindle, 4);
        assert!(reopened.mmap_is_turbo_quantized());
        let txn = env.read_txn().unwrap();
        assert_eq!(
            reopened
                .get_encoded_vector(&reopened.backend.read_borrowed(&txn), 11, 0)
                .unwrap(),
            encoded_before
        );
        assert_eq!(
            reopened
                .search::<VF>(
                    &reopened.backend.read_borrowed(&txn),
                    &[0.10, 0.20, 0.30, 0.40],
                    2,
                    None,
                    false
                )
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn spindle_originals_use_exact_sidecar_without_lmdb_vector_copy() {
        let (env, dir) = setup();
        let core = core_with_spindle_and_mmap(
            &env,
            dir.path(),
            "spindle_sidecar_original",
            SpindleConfig {
                mode: SpindleMode::TurboProd,
                keep_original: true,
                rescore: true,
                turbo_dims: 8,
                ..SpindleConfig::default()
            },
            8,
        );

        let data = vec![0.13, -0.27, 0.39, -0.41, 0.58, -0.62, 0.77, -0.91];
        let mut fields = HashMap::new();
        fields.insert("label".to_string(), Value::String("sidecar".to_string()));

        let mut txn = env.write_txn().unwrap();
        core.insert_flat(&mut txn, &data, Some(101), Some(fields.clone()))
            .unwrap();
        txn.commit().unwrap();
        core.flush_mmap().unwrap();

        let txn = env.read_txn().unwrap();
        let stored = core
            .get_vector_data(&core.backend.read_borrowed(&txn), 101)
            .unwrap();
        assert_eq!(stored.fields, fields);
        assert!(
            stored.original_vector.is_none(),
            "exact sidecar should avoid a duplicate raw vector in LMDB"
        );
        assert_eq!(
            core.get_original_vector(&core.backend.read_borrowed(&txn), 101)
                .unwrap()
                .unwrap(),
            data
        );
    }

    #[test]
    fn compact_spindle_export_uses_decoded_payload_without_raw_sidecar() {
        let (env, dir) = setup();
        let core = core_with_spindle_and_mmap(
            &env,
            dir.path(),
            "spindle_export_sidecar",
            SpindleConfig {
                mode: SpindleMode::TurboProd,
                keep_original: false,
                rescore: false,
                turbo_dims: 8,
                ..SpindleConfig::default()
            },
            8,
        );

        let data = vec![0.13, -0.27, 0.39, -0.41, 0.58, -0.62, 0.77, -0.91];
        let mut txn = env.write_txn().unwrap();
        core.insert_flat(&mut txn, &data, Some(202), None).unwrap();
        txn.commit().unwrap();
        core.flush_mmap().unwrap();

        let txn = env.read_txn().unwrap();
        let exported = core
            .export_level_zero(&core.backend.read_borrowed(&txn))
            .unwrap();
        assert_eq!(exported.len(), 1);
        assert_eq!(exported[0].0, 202);
        assert_ne!(
            exported[0].1, data,
            "compact spindle with keep_original=false should not require raw sidecar storage"
        );
        assert_eq!(exported[0].1.len(), data.len());
    }

    #[test]
    fn keep_original_skips_lossy_hvs8_sidecar_conversion() {
        let (env, dir) = setup();
        let core = core_with_spindle_and_mmap(
            &env,
            dir.path(),
            "spindle_hvs8_guard",
            SpindleConfig {
                mode: SpindleMode::TurboProd,
                keep_original: true,
                rescore: true,
                turbo_dims: 4,
                ..SpindleConfig::default()
            },
            4,
        );

        let mut txn = env.write_txn().unwrap();
        core.insert_flat(&mut txn, &[1.0, -2.0, 3.0, -4.0], Some(303), None)
            .unwrap();
        txn.commit().unwrap();
        core.flush_mmap().unwrap();

        assert!(!core.quantize_mmap().unwrap());
        assert!(!core.mmap_is_quantized());
    }

    #[test]
    fn externalized_marker_missing_ordinal_is_repairable_error() {
        let (env, dir) = setup();
        let core = core_with_mode_and_mmap(&env, dir.path(), "mmap_missing", SpindleMode::None, 4);

        let mut txn = env.write_txn().unwrap();
        core.insert_flat(&mut txn, &[1.0, 2.0, 3.0, 4.0], Some(9), None)
            .unwrap();
        txn.commit().unwrap();

        let mut txn = env.write_txn().unwrap();
        core.ordinals_db
            .as_ref()
            .unwrap()
            .delete(&mut txn, &9u128.to_be_bytes())
            .unwrap();
        txn.commit().unwrap();
        core.remove_cached_sidecar_ordinals(&[9]);

        let txn = env.read_txn().unwrap();
        let err = core
            .get_vector(&core.backend.read_borrowed(&txn), 9, 0, true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("externalized vector 9"));
        let err = core
            .get_encoded_vector(&core.backend.read_borrowed(&txn), 9, 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("externalized vector 9"));
        assert!(
            core.get_all_vectors(&core.backend.read_borrowed(&txn), Some(0))
                .unwrap()
                .is_empty(),
            "bulk scans should skip unavailable externalized markers"
        );
        assert!(core.externalized_marker_repair_needed());
    }

    #[test]
    fn purge_unavailable_externalized_marker_deletes_stale_rows() {
        let (env, dir) = setup();
        let core = core_with_mode_and_mmap(&env, dir.path(), "mmap_repair", SpindleMode::None, 4);

        let mut txn = env.write_txn().unwrap();
        core.insert_flat(&mut txn, &[1.0, 2.0, 3.0, 4.0], Some(9), None)
            .unwrap();
        core.ordinals_db
            .as_ref()
            .unwrap()
            .put(&mut txn, &9u128.to_be_bytes(), &99u64.to_le_bytes())
            .unwrap();
        txn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        assert!(
            core.get_all_vectors(&core.backend.read_borrowed(&txn), Some(0))
                .unwrap()
                .is_empty(),
            "scan should skip stale marker before repair"
        );
        assert!(core.externalized_marker_repair_needed());
        drop(txn);

        let mut txn = env.write_txn().unwrap();
        let repaired = core
            .purge_unavailable_externalized_markers(&mut txn, 64)
            .unwrap();
        txn.commit().unwrap();

        assert_eq!(repaired, 1);
        assert!(!core.externalized_marker_repair_needed());

        let txn = env.read_txn().unwrap();
        assert!(!core
            .contains_id(&core.backend.read_borrowed(&txn), 9)
            .unwrap());
        assert!(core
            .ordinals_db
            .as_ref()
            .unwrap()
            .get(&txn, &9u128.to_be_bytes())
            .unwrap()
            .is_none());
        assert!(core
            .get_all_vectors(&core.backend.read_borrowed(&txn), Some(0))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn externalized_delete_clears_ordinal_mapping() {
        let (env, dir) = setup();
        let core = core_with_mode_and_mmap(&env, dir.path(), "mmap_delete", SpindleMode::None, 4);

        let mut txn = env.write_txn().unwrap();
        core.insert_flat(&mut txn, &[1.0, 0.0, 0.0, 0.0], Some(11), None)
            .unwrap();
        txn.commit().unwrap();

        let mut txn = env.write_txn().unwrap();
        core.delete_vector(&mut txn, 11).unwrap();
        txn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        assert!(core
            .ordinals_db
            .as_ref()
            .unwrap()
            .get(&txn, &11u128.to_be_bytes())
            .unwrap()
            .is_none());
        assert!(!core
            .contains_id(&core.backend.read_borrowed(&txn), 11)
            .unwrap());
    }

    #[test]
    fn mixed_legacy_and_marker_vectors_export_and_build() {
        let (env, dir) = setup();
        let core = core_with_mode_and_mmap(&env, dir.path(), "mmap_mixed", SpindleMode::None, 4);

        let legacy = vec![9.0f32, 8.0, 7.0, 6.0];
        let marker = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut txn = env.write_txn().unwrap();
        let legacy_bytes = encode_vector(&legacy, &core.spindle).unwrap();
        core.vectors_db
            .as_ref()
            .unwrap()
            .put(
                &mut txn,
                VectorCore::vector_key(1, 0).as_ref(),
                legacy_bytes.as_slice(),
            )
            .unwrap();
        core.insert_flat(&mut txn, &marker, Some(2), None).unwrap();
        txn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        let mut exported = core
            .export_level_zero(&core.backend.read_borrowed(&txn))
            .unwrap();
        exported.sort_by_key(|(id, _, _)| *id);
        assert_eq!(exported.len(), 2);
        assert_eq!(exported[0].0, 1);
        assert_eq!(exported[0].1, legacy);
        assert_eq!(exported[1].0, 2);
        assert_eq!(exported[1].1, marker);
    }

    #[test]
    fn externalized_vectors_build_index_from_flat() {
        let (env, dir) = setup();
        let core = core_with_mode_and_mmap(&env, dir.path(), "mmap_build", SpindleMode::None, 4);

        let mut txn = env.write_txn().unwrap();
        core.insert_flat(&mut txn, &[1.0, 0.0, 0.0, 0.0], Some(1), None)
            .unwrap();
        core.insert_flat(&mut txn, &[0.0, 1.0, 0.0, 0.0], Some(2), None)
            .unwrap();
        core.insert_flat(&mut txn, &[0.9, 0.1, 0.0, 0.0], Some(3), None)
            .unwrap();
        txn.commit().unwrap();

        let mut txn = env.write_txn().unwrap();
        core.build_index_from_flat(&mut txn).unwrap();
        txn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        let results = core
            .search::<VF>(
                &core.backend.read_borrowed(&txn),
                &[1.0, 2.0, 3.0, 4.0],
                2,
                None,
                false,
            )
            .unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|vector| vector.get_id() == 2));
    }

    // ── mmap zero-copy scoring path coverage ──
    //
    // These tests build an mmap-backed VectorCore (HVEC) and run searches
    // through `search()`, which walks `search_level` and exercises the
    // `score_neighbor_distance` → `MmapBackend::with_vec_slice` zero-copy
    // path for distance computation. The asserts are end-to-end correctness
    // (right ids, right ordering, filter respected) — there's no in-process
    // introspection to confirm "the slice path ran", but by construction:
    //   - mmap_store is set (via `new_named_with_dir`),
    //   - level 0 search visits neighbors,
    //   - HVEC backend → `with_vec_slice` returns Some,
    //   - so `score_neighbor_distance` returns at the mmap branch.
    // Removing the mmap branch from the helper would make these fail.

    #[test]
    fn mmap_search_unfiltered_finds_self_at_top() {
        // Inserting a vector and then searching for it exactly should put it
        // at distance 0 (or very close). This exercises the unfiltered code
        // path: search_level → score_neighbor_distance → mmap-slice → SIMD
        // distance kernel. No filter, no metadata-only get_vector.
        let (env, dir) = setup();
        let core =
            core_with_mode_and_mmap(&env, dir.path(), "mmap_search_self", SpindleMode::None, 8);

        let mut txn = env.write_txn().unwrap();
        core.insert_flat(
            &mut txn,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Some(1),
            None,
        )
        .unwrap();
        core.insert_flat(
            &mut txn,
            &[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            Some(2),
            None,
        )
        .unwrap();
        core.insert_flat(
            &mut txn,
            &[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            Some(3),
            None,
        )
        .unwrap();
        core.build_index_from_flat(&mut txn).unwrap();
        txn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        // Query for vector 1's exact data: it should be the closest hit.
        let results = core
            .search::<VF>(
                &core.backend.read_borrowed(&txn),
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                3,
                None,
                false,
            )
            .unwrap();
        assert!(
            !results.is_empty(),
            "mmap-backed search must return results"
        );
        assert_eq!(
            results[0].get_id(),
            1,
            "self-query through the mmap-slice path should rank the queried \
             vector first (got {:?})",
            results.iter().map(|h| h.get_id()).collect::<Vec<_>>()
        );

        // Distance ordering must hold across the result list.
        for w in results.windows(2) {
            assert!(
                w[1].get_distance() >= w[0].get_distance(),
                "results not in ascending distance order: {} then {}",
                w[0].get_distance(),
                w[1].get_distance()
            );
        }
        drop(txn);

        let reopened =
            core_with_mode_and_mmap(&env, dir.path(), "mmap_search_self", SpindleMode::None, 8);
        assert!(dir.path().join("mmap_search_self.hvec").exists());
        let txn = env.read_txn().unwrap();
        let reopened_results = reopened
            .search::<VF>(
                &reopened.backend.read_borrowed(&txn),
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                3,
                None,
                false,
            )
            .unwrap();
        assert_eq!(reopened_results[0].get_id(), 1);
    }

    #[test]
    fn mmap_search_with_filter_returns_only_filter_passing_ids() {
        // Exercises the filter branch: get_vector(with_data=false) for the
        // filter check, then score_neighbor_distance for the distance via
        // the mmap-slice. Filter excludes specific ids; results must
        // contain only ids that pass.
        let (env, dir) = setup();
        let core =
            core_with_mode_and_mmap(&env, dir.path(), "mmap_search_filter", SpindleMode::None, 8);

        let mut txn = env.write_txn().unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xfeed_beef);
        // 60 vectors is plenty for HNSW to navigate to filtered candidates
        // without selectivity drift dominating.
        for i in 1..=60u128 {
            let data: Vec<f32> = (0..8)
                .map(|_| rand::Rng::random_range(&mut rng, -1.0f32..1.0f32))
                .collect();
            core.insert_flat(&mut txn, &data, Some(i), None).unwrap();
        }
        core.build_index_from_flat(&mut txn).unwrap();
        txn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        let query: Vec<f32> = vec![0.0; 8];

        // Filter: only even ids (50 % selectivity).
        let only_even: VF = |v: &HVector| v.get_id() % 2 == 0;
        let results = core
            .search::<VF>(
                &core.backend.read_borrowed(&txn),
                &query,
                5,
                Some(&[only_even]),
                true,
            )
            .unwrap();

        assert!(
            !results.is_empty(),
            "filtered search through mmap-slice path must return results"
        );
        for r in &results {
            assert_eq!(
                r.get_id() % 2,
                0,
                "filter rejected id {} should not appear in results",
                r.get_id()
            );
        }
        // Distance ordering must still hold.
        for w in results.windows(2) {
            assert!(w[1].get_distance() >= w[0].get_distance());
        }
    }

    #[test]
    fn mmap_search_returns_distinct_topk_with_distance_ordering() {
        // Sanity test for the unfiltered mmap-slice path with enough
        // vectors that the HNSW graph actually has to navigate. Catches
        // bugs where the slice-path returns wrong distances (which would
        // collapse top-k to the same vector or break ordering).
        let (env, dir) = setup();
        let core =
            core_with_mode_and_mmap(&env, dir.path(), "mmap_search_topk", SpindleMode::None, 16);

        let mut txn = env.write_txn().unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xdead_beef);
        for i in 1..=120u128 {
            let data: Vec<f32> = (0..16)
                .map(|_| rand::Rng::random_range(&mut rng, -1.0f32..1.0f32))
                .collect();
            core.insert_flat(&mut txn, &data, Some(i), None).unwrap();
        }
        core.build_index_from_flat(&mut txn).unwrap();
        txn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        let query: Vec<f32> = vec![0.5; 16];
        let results = core
            .search::<VF>(&core.backend.read_borrowed(&txn), &query, 10, None, false)
            .unwrap();

        assert_eq!(results.len(), 10, "expected exactly 10 hits");

        // No duplicate ids — each candidate is visited once.
        let mut ids: Vec<u128> = results.iter().map(|h| h.get_id()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 10, "results contained duplicate ids");

        // Strict distance ordering.
        for w in results.windows(2) {
            assert!(
                w[1].get_distance() >= w[0].get_distance(),
                "mmap-slice path returned out-of-order results: {} > {}",
                w[0].get_distance(),
                w[1].get_distance()
            );
        }
    }

    #[test]
    #[ignore]
    fn lsm_minio_flat_hnsw_publish_search_round_trip() {
        use crate::helix_engine::storage_core::backend::StorageBackend;
        use crate::helix_engine::storage_core::backend_any::AnyBackend;
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use object_store::aws::AmazonS3Builder;
        use std::sync::Arc;
        use std::time::{SystemTime, UNIX_EPOCH};

        let (env, _dir) = setup();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = format!("vector-hnsw-{nanos}");
        let store = AmazonS3Builder::new()
            .with_endpoint("http://localhost:9000")
            .with_region("us-east-1")
            .with_bucket_name("helion-test")
            .with_access_key_id("test")
            .with_secret_access_key("testtest123")
            .with_allow_http(true)
            .build()
            .expect("build MinIO S3 store");
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_with_store(&path, Arc::new(store)).expect("open LSM backend"),
        ));

        let mut txn = env.write_txn().unwrap();
        let core = VectorCore::new_named_with_dir(
            &env,
            &mut txn,
            "dense",
            HNSWConfig::new(Some(8), Some(32), Some(64)),
            DistanceMetric::Cosine,
            SpindleConfig::default(),
            None,
            4,
            None,
            Arc::clone(&backend),
        )
        .unwrap();
        txn.commit().unwrap();

        {
            let mut w = backend.begin_write().unwrap();
            core.insert_flat_be(&mut w, &[1.0, 0.0, 0.0, 0.0], Some(1), None)
                .unwrap();
            core.insert_flat_be(&mut w, &[0.0, 1.0, 0.0, 0.0], Some(2), None)
                .unwrap();
            core.insert_flat_be(&mut w, &[0.9, 0.1, 0.0, 0.0], Some(3), None)
                .unwrap();
            backend.commit(w).unwrap();
        }

        let ro = env.read_txn().unwrap();
        let mut prepared = core
            .prepare_index_from_flat(&core.backend.read_borrowed(&ro))
            .unwrap()
            .expect("prepared HNSW index");
        drop(ro);

        {
            let mut w = backend.begin_write().unwrap();
            let total = prepared.point_ids.len();
            core.flush_prepared_index_chunk_be(&mut w, &mut prepared, 0, total)
                .unwrap();
            core.finalize_prepared_index_entry_be(&mut w, &prepared)
                .unwrap();
            backend.commit(w).unwrap();
        }

        let ro = env.read_txn().unwrap();
        assert!(
            core.has_index(&core.backend.read_borrowed(&ro)).unwrap(),
            "LSM HNSW entry point present"
        );
        let results = core
            .search::<VF>(
                &core.backend.read_borrowed(&ro),
                &[1.0, 0.0, 0.0, 0.0],
                2,
                None,
                false,
            )
            .unwrap();
        assert!(
            results.iter().any(|vector| vector.get_id() == 1),
            "LSM HNSW search should find the closest inserted vector; got {:?}",
            results.iter().map(|h| h.get_id()).collect::<Vec<_>>()
        );
    }

    /// Proves that on the SlateDB-LSM backend a TurboProd-spindle dense search,
    /// once the HNSW index is built through the real `_be` optimizer path, is
    /// HNSW-bounded (it does NOT degrade to a full O(N) namespace scan) and is
    /// scored over compressed TurboProd payloads — not a full-f32 decode of every
    /// stored vector.
    ///
    /// Bounded proof: the `LsmBackend` read counter is reset immediately before
    /// the indexed query and asserted to be `<< N` afterwards. A flat scan would
    /// register ≥ N reads (one per KV the `scan` iterator yields), so this test
    /// fails the instant search falls back to `search_flat`/`get_all_vectors`.
    ///
    /// Runs on an in-memory SlateDB (no MinIO required) like the other in-memory
    /// LSM tests. No `.hvtq` sidecar is materialized here (data_dir=None), so
    /// TurboProd scoring runs via bounded backend point reads of the encoded
    /// payload + `score_encoded` (the compact TurboProd scorer), which is exactly
    /// the LSM hot path when the sidecar is absent.
    #[test]
    fn lsm_hnsw_indexed_turboprod_search_is_bounded() {
        use crate::helix_engine::storage_core::backend_any::AnyBackend;
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::sync::Arc;
        use std::time::{SystemTime, UNIX_EPOCH};

        const N: usize = 8000;
        const DIM: usize = 768;

        let (env, _dir) = setup();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = format!("vector-hnsw-bounded-{nanos}");
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&path).expect("open in-memory LSM backend"),
        ));

        // TurboProd spindle: stored payloads are TurboProd-compressed bytes and
        // the prepared query is `PreparedSpindleQuery::TurboProd`, so the search
        // hot path scores compact codes (score_encoded → score_turbo_prod_payload).
        let spindle = SpindleConfig::turbo_prod(DIM);
        assert_eq!(
            spindle.mode,
            SpindleMode::TurboProd,
            "test must exercise the TurboProd compact scorer"
        );

        let mut txn = env.write_txn().unwrap();
        let core = VectorCore::new_named_with_dir(
            &env,
            &mut txn,
            "dense",
            HNSWConfig::new(Some(16), Some(32), Some(64)),
            DistanceMetric::Cosine,
            spindle,
            None, // data_dir=None → no .hvtq sidecar on LSM (point-read scoring path)
            DIM,
            None,
            Arc::clone(&backend),
        )
        .unwrap();
        txn.commit().unwrap();

        // Deterministic corpus of independent random vectors (id 1..=N). The
        // The query (chosen after the build, below) is the HNSW *entry point's*
        // own original vector. The entry point is where every search starts and is
        // scored first, so it is guaranteed reachable regardless of the build's
        // (deliberately unseeded) level-assignment RNG — making recall
        // deterministic. Its true self-distance is the global minimum (~0).
        // TurboProd's compact codes are lossy, but the indexed search reranks the
        // surfaced candidates against the stored originals (rescore), recovering
        // the exact self-match — provided the original is persisted on the LSM
        // flat path, which is the fix this test also exercises.
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        {
            let mut w = backend.begin_write().unwrap();
            for id in 1..=N as u128 {
                let data: Vec<f32> = (0..DIM)
                    .map(|_| rand::Rng::random_range(&mut rng, -1.0f32..1.0f32))
                    .collect();
                core.insert_flat_be(&mut w, &data, Some(id), None).unwrap();
            }
            backend.commit(w).unwrap();
        }

        // Build the HNSW index through the LSM arm of the real optimizer path:
        // prepare_index_from_flat → flush_prepared_index_chunk_be →
        // finalize_prepared_index_entry_be, committed via the backend.
        {
            let ro = backend.begin_read().unwrap();
            let mut prepared = core
                .prepare_index_from_flat(&ro)
                .unwrap()
                .expect("prepared HNSW index over the flat LSM corpus");
            drop(ro);

            let mut w = backend.begin_write().unwrap();
            let total = prepared.point_ids.len();
            core.flush_prepared_index_chunk_be(&mut w, &mut prepared, 0, total)
                .unwrap();
            core.finalize_prepared_index_entry_be(&mut w, &prepared)
                .unwrap();
            backend.commit(w).unwrap();
        }

        // (4) Index is consulted on LSM.
        let ro = backend.begin_read().unwrap();
        assert!(
            core.has_index(&ro).unwrap(),
            "HNSW entry point must be present on LSM after the _be build"
        );

        let AnyBackend::Lsm(lsm) = &*backend else {
            unreachable!("backend is LSM in this test");
        };

        // Deterministic target: the HNSW entry point. It is the search start (and
        // scored first in search_level), so it is always reachable; its stored
        // original is the exact query, giving a guaranteed ~0 self-match.
        let entry = core.get_entry_point(&ro).expect("entry point after build");
        let target_id = entry.get_id();
        let query = core
            .get_original_vector(&ro, target_id)
            .expect("original vector lookup")
            .expect("entry-point original must be persisted on the LSM flat path");

        // (7) TurboProd compact scoring is on the hot path: the stored level-0
        // payload is the TurboProd-compressed encoding (spindle magic + tag),
        // strictly smaller than a raw f32 vector. The frontier scorer
        // (scored_distance → score_encoded → score_turbo_prod_payload) reads and
        // scores exactly these bytes; it never full-f32-decodes every candidate.
        let stored_payload = core
            .get_encoded_vector(&ro, target_id, 0)
            .expect("encoded TurboProd payload for target");
        assert!(
            crate::helix_engine::vector_core::spindle::is_spindle_payload(&stored_payload),
            "stored level-0 payload must be a spindle (TurboProd) encoding, not raw f32"
        );
        assert!(
            stored_payload.len() < DIM * std::mem::size_of::<f32>(),
            "TurboProd payload ({} bytes) must be smaller than a raw f32 vector ({} bytes)",
            stored_payload.len(),
            DIM * std::mem::size_of::<f32>()
        );

        // (6) BOUNDED: reset the LSM read counter, run the indexed query, and
        // assert the per-query backend reads are sub-linear in N. A degraded
        // full scan (search_flat → get_all_vectors) yields ≥ N reads; an
        // HNSW-bounded query touches ~O(ef·M) graph nodes, independent of N.
        lsm.reset_read_count();
        let results = core.search::<VF>(&ro, &query, 5, None, false).unwrap();
        let reads = lsm.read_count();

        // (5) Correctness: rescore against the stored original recovers the exact
        // self-match, so the entry-point target is the nearest neighbor (dist ~0).
        let target_hit = results.iter().find(|v| v.get_id() == target_id);
        assert!(
            target_hit.is_some(),
            "indexed TurboProd search must return the nearest neighbor (entry id={}); got {:?}",
            target_id,
            results
                .iter()
                .map(|h| (h.get_id(), h.get_distance()))
                .collect::<Vec<_>>()
        );
        assert!(
            target_hit.unwrap().get_distance() < 1e-3,
            "rescored self-match must have ~0 distance; got {}",
            target_hit.unwrap().get_distance()
        );

        // The crux: the indexed query touches a bounded ~O(ef·M) working set,
        // NOT the whole collection.
        //
        // (a) Hard "not a full scan" guard: a degraded search_flat path scans the
        //     entire Vectors namespace, registering ≥ N backend reads. The indexed
        //     query must stay strictly below N.
        assert!(
            reads < N,
            "indexed LSM search degraded to a full scan: {} backend reads for N={} \
             (≥ N means search fell back to search_flat/get_all_vectors)",
            reads,
            N
        );
        // (b) Scalability guard: the HNSW working set is governed by ef and M, not
        //     by N. With ef=64 / m_max_0=32 the per-query envelope is a few
        //     thousand point reads regardless of corpus size, and it grows
        //     LOGARITHMICALLY rather than linearly. Calibrated across scales (same
        //     query, same ef/M, only N changing):
        //         N=4000 → ~3.4k reads      N=8000 → ~4.5k reads
        //     i.e. a 2× corpus costs ~1.33× reads, not 2×. Under a true O(N) scan
        //     the count would double with N; here it does not. A generous fixed
        //     fraction of N (well above the measured ~4.3–4.9k band, well below N)
        //     captures this without flaking on the build's RNG nondeterminism: the
        //     instant search degrades to O(N) this fails, while honest log-growth
        //     stays comfortably under the bound for N ≫ the ef·M envelope.
        let envelope = N * 7 / 8; // 7000 at N=8000; the working set is ~half this.
        assert!(
            reads < envelope,
            "indexed LSM search exceeded its sub-linear HNSW envelope: {} reads for N={} \
             (cap {}); a degraded O(N) scan would not satisfy this",
            reads,
            N,
            envelope
        );

        eprintln!(
            "lsm_hnsw_indexed_turboprod_search_is_bounded: N={N} DIM={DIM} \
             per-query backend reads={reads} (bounded by ~O(ef·M), N-independent; \
             a full scan would be ≥ {N}); \
             turboprod_payload_bytes={} vs raw_f32_bytes={}",
            stored_payload.len(),
            DIM * std::mem::size_of::<f32>()
        );
    }

    #[test]
    fn lsm_hvtq_backfill_persists_blob_for_existing_indexed_segment() {
        use crate::helix_engine::storage_core::backend_any::AnyBackend;
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::sync::Arc;
        use std::time::{SystemTime, UNIX_EPOCH};

        const DIM: usize = 4;

        let (_env, dir) = setup();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("hvtq-backfill-{nanos}"))
                .expect("open in-memory LSM backend"),
        ));
        let segment = "dense__seg_000000";
        let core = VectorCore::new_named_lsm_with_dir(
            segment,
            HNSWConfig::new(Some(8), Some(16), Some(32)),
            DistanceMetric::Cosine,
            SpindleConfig::turbo_prod_compact(DIM),
            Some(dir.path()),
            DIM,
            Arc::clone(&backend),
        )
        .expect("create LSM vector core with sidecar dir");

        {
            let mut w = backend.begin_write().unwrap();
            for (id, data) in [
                (11u128, [0.10, 0.20, 0.30, 0.40]),
                (12u128, [0.90, 0.80, 0.70, 0.60]),
                (13u128, [-0.20, 0.15, -0.35, 0.55]),
                (14u128, [0.05, -0.75, 0.25, -0.45]),
            ] {
                core.insert_flat_be(&mut w, &data, Some(id), None).unwrap();
            }
            backend.commit(w).unwrap();
        }

        {
            let ro = backend.begin_read().unwrap();
            let mut prepared = core
                .prepare_index_from_flat(&ro)
                .unwrap()
                .expect("prepared HNSW index");
            drop(ro);
            let total = prepared.point_ids.len();
            let mut w = backend.begin_write().unwrap();
            core.flush_prepared_index_chunk_be(&mut w, &mut prepared, 0, total)
                .unwrap();
            core.finalize_prepared_index_entry_be(&mut w, &prepared)
                .unwrap();
            backend.commit(w).unwrap();
        }

        let hvtq_path = dir.path().join(segment).with_extension("hvtq");
        assert!(
            !hvtq_path.exists(),
            "fixture must start as an indexed segment without a local sidecar"
        );
        let ro = backend.begin_read().unwrap();
        assert!(core.has_index(&ro).unwrap());
        assert!(core.hvtq_sidecar_backfill_needed_be(&ro).unwrap());
        let point_ids = core
            .prepare_hvtq_sidecar_backfill_be(&ro)
            .unwrap()
            .expect("missing HVTQ sidecar should produce a backfill plan");
        drop(ro);
        assert!(hvtq_path.exists(), "backfill should materialize local HVTQ");

        {
            let mut w = backend.begin_write().unwrap();
            core.write_sidecar_ordinals_be(&mut w, &point_ids).unwrap();
            backend.commit(w).unwrap();
        }

        {
            let mut w = backend.begin_write().unwrap();
            core.insert_flat_be(&mut w, &[0.10, 0.20, 0.30, 0.40], Some(11), None)
                .unwrap();
            backend.commit(w).unwrap();
        }
        let ro = backend.begin_read().unwrap();
        let same_payload = backend
            .get_with(
                &ro,
                Namespace::Segment {
                    physical_name: segment,
                    db: SegmentDb::Vectors,
                },
                VectorCore::vector_key(11, 0).as_ref(),
                |value| value.map(|bytes| bytes.to_vec()),
            )
            .unwrap()
            .unwrap();
        assert!(
            same_payload.is_empty(),
            "an unchanged HVTQ vector may keep using its sidecar marker"
        );
        drop(ro);

        let replacement = [-0.80, 0.15, 0.45, 0.70];
        {
            let mut w = backend.begin_write().unwrap();
            core.insert_flat_be(&mut w, &replacement, Some(11), None)
                .unwrap();
            backend.commit(w).unwrap();
        }
        let ro = backend.begin_read().unwrap();
        let changed_payload = backend
            .get_with(
                &ro,
                Namespace::Segment {
                    physical_name: segment,
                    db: SegmentDb::Vectors,
                },
                VectorCore::vector_key(11, 0).as_ref(),
                |value| value.map(|bytes| bytes.to_vec()),
            )
            .unwrap()
            .unwrap();
        assert!(
            !changed_payload.is_empty(),
            "a changed vector must not resolve through a stale HVTQ ordinal"
        );
        let expected = encode_vector(&replacement, &core.spindle).unwrap();
        assert_eq!(changed_payload, expected);
        assert_eq!(
            core.get_encoded_vector(&ro, 11, 0).unwrap(),
            expected,
            "reads must observe the replacement payload"
        );
        assert_eq!(
            core.get_vector(&ro, 11, 0, true).unwrap().get_data(),
            decode_vector(&expected).unwrap(),
            "decoded reads must not use the stale HVTQ sidecar row"
        );
        let ordinal_exists = backend
            .get_with(
                &ro,
                Namespace::Segment {
                    physical_name: segment,
                    db: SegmentDb::Ordinals,
                },
                &11u128.to_be_bytes(),
                |value| value.is_some(),
            )
            .unwrap();
        assert!(
            !ordinal_exists,
            "replacement payload must detach its stale sidecar ordinal"
        );
        assert_eq!(core.cached_sidecar_ordinal(11).unwrap(), None);
        drop(ro);

        let ro = backend.begin_read().unwrap();
        assert!(
            !core.hvtq_sidecar_backfill_needed_be(&ro).unwrap(),
            "persisted blob should clear HVTQ backfill debt"
        );
        drop(ro);
        std::fs::remove_file(&hvtq_path).unwrap();

        let reopened = VectorCore::new_named_lsm_with_dir(
            segment,
            HNSWConfig::new(Some(8), Some(16), Some(32)),
            DistanceMetric::Cosine,
            SpindleConfig::turbo_prod_compact(DIM),
            Some(dir.path()),
            DIM,
            Arc::clone(&backend),
        )
        .expect("reopen LSM vector core");
        assert!(
            !hvtq_path.exists(),
            "reopen should not restore HVTQ before a real read"
        );

        let ro = backend.begin_read().unwrap();
        let _ = reopened.get_vector(&ro, 11, 0, true).unwrap();
        drop(ro);
        assert!(hvtq_path.exists(), "first read should restore HVTQ");
        assert!(reopened.mmap_is_turbo_quantized());
    }

    #[test]
    fn lsm_exact_hvec_opens_sidecar_on_first_raw_write() {
        use crate::helix_engine::storage_core::backend_any::AnyBackend;
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::sync::Arc;
        use std::time::{SystemTime, UNIX_EPOCH};

        const DIM: usize = 4;

        let (_env, dir) = setup();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("exact-hvec-first-write-{nanos}"))
                .expect("open in-memory LSM backend"),
        ));
        let segment = "dense__seg_000000";
        let core = VectorCore::new_named_lsm_with_dir(
            segment,
            HNSWConfig::new(Some(8), Some(16), Some(32)),
            DistanceMetric::Cosine,
            SpindleConfig {
                mode: SpindleMode::None,
                ..SpindleConfig::default()
            },
            Some(dir.path()),
            DIM,
            Arc::clone(&backend),
        )
        .expect("create empty exact LSM vector core");

        assert_eq!(core.mmap_store_count(), None);
        let hvec_path = dir.path().join(segment).with_extension("hvec");
        assert!(
            !hvec_path.exists(),
            "empty exact LSM cores should not create a cold sidecar at construction"
        );

        {
            let mut w = backend.begin_write().unwrap();
            core.insert_flat_be(&mut w, &[0.0, 1.0, 0.0, 0.0], Some(42), None)
                .unwrap();
            backend.commit(w).unwrap();
        }

        assert_eq!(core.sidecar_format_label(), "hvec");
        assert_eq!(core.mmap_store_count(), Some(1));
        assert!(
            hvec_path.exists(),
            "first level-0 raw write should open the exact HVEC sidecar"
        );
        assert_eq!(core.cached_sidecar_ordinal(42).unwrap(), Some(0));

        let AnyBackend::Lsm(lsm) = &*backend else {
            unreachable!("backend is LSM in this test");
        };
        let ro = backend.begin_read().unwrap();
        lsm.reset_read_count();
        let distance = core
            .score_neighbor_distance(
                &ro,
                &PreparedSpindleQuery::None,
                42,
                0,
                &[0.0, 1.0, 0.0, 0.0],
            )
            .unwrap()
            .expect("fresh sidecar should score the inserted vector");

        assert!(distance < 1e-6);
        assert_eq!(
            lsm.read_count(),
            0,
            "fresh sidecar scoring should use the cached ordinal map and mmap row"
        );
    }

    #[test]
    fn lsm_exact_hvec_materializes_from_slate_rows_and_scores_without_backend_reads() {
        use crate::helix_engine::storage_core::backend_any::AnyBackend;
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::sync::Arc;
        use std::time::{SystemTime, UNIX_EPOCH};

        const DIM: usize = 4;

        let (_env, dir) = setup();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("exact-hvec-materialize-{nanos}"))
                .expect("open in-memory LSM backend"),
        ));
        let segment = "dense__seg_000000";
        let core = VectorCore::new_named_lsm_with_dir(
            segment,
            HNSWConfig::new(Some(8), Some(16), Some(32)),
            DistanceMetric::Cosine,
            SpindleConfig {
                mode: SpindleMode::None,
                ..SpindleConfig::default()
            },
            Some(dir.path()),
            DIM,
            Arc::clone(&backend),
        )
        .expect("create LSM vector core");

        {
            let mut w = backend.begin_write().unwrap();
            core.insert_flat_be(&mut w, &[1.0, 0.0, 0.0, 0.0], Some(41), None)
                .unwrap();
            core.insert_flat_be(&mut w, &[0.0, 1.0, 0.0, 0.0], Some(42), None)
                .unwrap();
            core.insert_flat_be(&mut w, &[0.0, 0.0, 1.0, 0.0], Some(43), None)
                .unwrap();
            backend.commit(w).unwrap();
        }

        let hvec_path = dir.path().join(segment).with_extension("hvec");
        let _ = std::fs::remove_file(&hvec_path);
        assert!(
            !hvec_path.exists(),
            "fixture must reopen from SlateDB rows, not a pre-existing sidecar"
        );

        let reopened = VectorCore::new_named_lsm_with_dir(
            segment,
            HNSWConfig::new(Some(8), Some(16), Some(32)),
            DistanceMetric::Cosine,
            SpindleConfig {
                mode: SpindleMode::None,
                ..SpindleConfig::default()
            },
            Some(dir.path()),
            DIM,
            Arc::clone(&backend),
        )
        .expect("reopen LSM vector core");

        assert_eq!(reopened.sidecar_format_label(), "none");
        assert_eq!(reopened.mmap_store_count(), None);
        assert!(
            !hvec_path.exists(),
            "reopen should not create an exact HVEC sidecar"
        );

        let AnyBackend::Lsm(lsm) = &*backend else {
            unreachable!("backend is LSM in this test");
        };
        let ro = backend.begin_read().unwrap();
        lsm.reset_read_count();
        let distance = reopened
            .score_neighbor_distance(
                &ro,
                &PreparedSpindleQuery::None,
                42,
                0,
                &[0.0, 1.0, 0.0, 0.0],
            )
            .unwrap()
            .expect("materialized sidecar should score the neighbor");

        assert!(
            distance < 1e-6,
            "self-score should stay exact from HVEC; got {distance}"
        );
        assert_eq!(reopened.sidecar_format_label(), "hvec");
        assert_eq!(reopened.mmap_store_count(), Some(3));
        assert!(hvec_path.exists(), "first score should create HVEC sidecar");
        assert!(
            lsm.read_count() > 0,
            "first score should materialize from LSM"
        );

        lsm.reset_read_count();
        let distance = reopened
            .score_neighbor_distance(
                &ro,
                &PreparedSpindleQuery::None,
                42,
                0,
                &[0.0, 1.0, 0.0, 0.0],
            )
            .unwrap()
            .expect("materialized sidecar should score the neighbor");
        assert!(
            distance < 1e-6,
            "self-score should stay exact from HVEC; got {distance}"
        );
        assert_eq!(
            lsm.read_count(),
            0,
            "sidecar scoring should use the in-process ordinal map and mmap row"
        );

        lsm.reset_read_count();
        let distance = reopened
            .score_neighbor_distance(
                &ro,
                &PreparedSpindleQuery::None,
                42,
                1,
                &[0.0, 1.0, 0.0, 0.0],
            )
            .unwrap()
            .expect("upper-level HNSW scoring should reuse the sidecar row");
        assert!(
            distance < 1e-6,
            "upper-level self-score should stay exact from HVEC; got {distance}"
        );
        assert_eq!(
            lsm.read_count(),
            0,
            "upper-level HNSW scoring should not read vector payloads from LSM"
        );

        fn only_id_42(id: u128) -> bool {
            id == 42
        }
        let filters: [fn(u128) -> bool; 1] = [only_id_42];
        lsm.reset_read_count();
        let filtered = reopened
            .search_flat_id_filter_direct(
                &ro,
                &[0.0, 1.0, 0.0, 0.0],
                &PreparedSpindleQuery::None,
                1,
                Some(&filters),
            )
            .expect("flat id-filter search should use materialized sidecar");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].get_id(), 42);
        assert_eq!(
            lsm.read_count(),
            0,
            "flat id-filter search should use cached ordinals and mmap rows"
        );
    }

    #[test]
    fn lsm_indexed_hvec_search_uses_hnsw_path_with_populated_sidecar() {
        use crate::helix_engine::storage_core::backend_any::AnyBackend;
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::sync::Arc;
        use std::time::{SystemTime, UNIX_EPOCH};

        const DIM: usize = 4;

        let (_env, dir) = setup();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("indexed-hvec-search-{nanos}"))
                .expect("open in-memory LSM backend"),
        ));
        let segment = "dense__seg_indexed";
        let core = VectorCore::new_named_lsm_with_dir(
            segment,
            HNSWConfig::new(Some(8), Some(16), Some(32)),
            DistanceMetric::Cosine,
            SpindleConfig {
                mode: SpindleMode::None,
                ..SpindleConfig::default()
            },
            Some(dir.path()),
            DIM,
            Arc::clone(&backend),
        )
        .expect("create LSM vector core");

        {
            let mut w = backend.begin_write().unwrap();
            for id in 1..=32u128 {
                let scale = id as f32 / 32.0;
                let data = [scale, 1.0 - scale, (id % 3) as f32, (id % 5) as f32];
                core.insert_flat_be(&mut w, &data, Some(id), None).unwrap();
            }
            backend.commit(w).unwrap();
        }

        {
            let ro = backend.begin_read().unwrap();
            let mut prepared = core
                .prepare_index_from_flat(&ro)
                .unwrap()
                .expect("prepared HNSW index over flat LSM rows");
            drop(ro);

            let mut w = backend.begin_write().unwrap();
            let total = prepared.point_ids.len();
            core.flush_prepared_index_chunk_be(&mut w, &mut prepared, 0, total)
                .unwrap();
            core.finalize_prepared_index_entry_be(&mut w, &prepared)
                .unwrap();
            backend.commit(w).unwrap();
        }

        let ro = backend.begin_read().unwrap();
        assert!(core.has_index(&ro).unwrap());
        assert_eq!(core.sidecar_format_label(), "hvec");
        assert!(core.has_populated_mmap_sidecar());

        let AnyBackend::Lsm(lsm) = &*backend else {
            unreachable!("backend is LSM in this test");
        };
        lsm.reset_read_count();
        let results = core
            .search_with_id_filter_ef_observed::<fn(u128) -> bool>(
                &ro,
                &[1.0, 0.0, 0.0, 0.0],
                5,
                None,
                false,
                None,
                None,
                None,
            )
            .unwrap();

        assert!(!results.is_empty());
        assert!(
            lsm.read_count() > 2,
            "indexed sidecar search should read HNSW graph state; read_count={} suggests a flat sidecar scan bypass",
            lsm.read_count()
        );
    }

    #[test]
    fn lsm_indexed_hvec_reopens_from_persisted_sidecar_blob_without_row_scan() {
        use crate::helix_engine::storage_core::backend_any::AnyBackend;
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::sync::Arc;
        use std::time::{SystemTime, UNIX_EPOCH};

        const DIM: usize = 4;

        let (_env, dir) = setup();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("indexed-hvec-blob-restore-{nanos}"))
                .expect("open in-memory LSM backend"),
        ));
        let segment = "dense__seg_indexed_blob";
        let core = VectorCore::new_named_lsm_with_dir(
            segment,
            HNSWConfig::new(Some(8), Some(16), Some(32)),
            DistanceMetric::Cosine,
            SpindleConfig {
                mode: SpindleMode::None,
                ..SpindleConfig::default()
            },
            Some(dir.path()),
            DIM,
            Arc::clone(&backend),
        )
        .expect("create LSM vector core");

        {
            let mut w = backend.begin_write().unwrap();
            for id in 1..=32u128 {
                let scale = id as f32 / 32.0;
                let data = [scale, 1.0 - scale, (id % 3) as f32, (id % 5) as f32];
                core.insert_flat_be(&mut w, &data, Some(id), None).unwrap();
            }
            backend.commit(w).unwrap();
        }

        {
            let ro = backend.begin_read().unwrap();
            let mut prepared = core
                .prepare_index_from_flat(&ro)
                .unwrap()
                .expect("prepared HNSW index over flat LSM rows");
            drop(ro);

            let mut w = backend.begin_write().unwrap();
            let total = prepared.point_ids.len();
            core.flush_prepared_index_chunk_be(&mut w, &mut prepared, 0, total)
                .unwrap();
            core.finalize_prepared_index_entry_be(&mut w, &prepared)
                .unwrap();
            backend.commit(w).unwrap();
        }

        let ro = backend.begin_read().unwrap();
        let persisted_hvec_blob = backend
            .get_with(
                &ro,
                core.seg_ns(SegmentDb::VectorData),
                HVEC_SIDECAR_BLOB_KEY,
                |opt| opt.map(|bytes| bytes.to_vec()),
            )
            .unwrap()
            .expect("indexed LSM segment should persist its HVEC sidecar blob");
        drop(ro);
        assert!(
            persisted_hvec_blob.len() >= 16,
            "persisted HVEC blob should include the header"
        );
        let persisted_dim =
            u32::from_le_bytes(persisted_hvec_blob[4..8].try_into().unwrap()) as usize;
        let persisted_count = u64::from_le_bytes(persisted_hvec_blob[8..16].try_into().unwrap());
        assert_eq!(persisted_dim, DIM);
        assert_eq!(persisted_count, 32);

        let hvec_path = dir.path().join(segment).with_extension("hvec");
        assert!(
            hvec_path.exists(),
            "writer should have an exact HVEC sidecar"
        );
        std::fs::remove_file(&hvec_path).unwrap();

        let reopened = VectorCore::new_named_lsm_with_dir(
            segment,
            HNSWConfig::new(Some(8), Some(16), Some(32)),
            DistanceMetric::Cosine,
            SpindleConfig {
                mode: SpindleMode::None,
                ..SpindleConfig::default()
            },
            Some(dir.path()),
            DIM,
            Arc::clone(&backend),
        )
        .expect("reopen LSM vector core");
        assert_eq!(reopened.sidecar_format_label(), "none");

        let AnyBackend::Lsm(lsm) = &*backend else {
            unreachable!("backend is LSM in this test");
        };
        let ro = backend.begin_read().unwrap();
        lsm.reset_read_count();
        let distance = reopened
            .score_neighbor_distance(
                &ro,
                &PreparedSpindleQuery::None,
                16,
                1,
                &[0.5, 0.5, 1.0, 1.0],
            )
            .unwrap()
            .expect("restored sidecar should score upper-level neighbor");

        assert!(distance.is_finite());
        assert_eq!(reopened.sidecar_format_label(), "hvec");
        assert!(
            hvec_path.exists(),
            "first score should restore the HVEC blob"
        );
        assert!(
            lsm.read_count() <= 3,
            "restoring from the sidecar blob should avoid scanning all vector rows; read_count={}",
            lsm.read_count()
        );
    }

    #[test]
    fn lsm_scalar_int8_materializes_hspn_from_slate_rows_and_scores_without_backend_reads() {
        use crate::helix_engine::storage_core::backend_any::AnyBackend;
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::sync::Arc;
        use std::time::{SystemTime, UNIX_EPOCH};

        const DIM: usize = 4;

        let (_env, dir) = setup();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("scalar-hspn-materialize-{nanos}"))
                .expect("open in-memory LSM backend"),
        ));
        let segment = "dense__seg_000000";
        let spindle = SpindleConfig::scalar_int8();
        let core = VectorCore::new_named_lsm_with_dir(
            segment,
            HNSWConfig::new(Some(8), Some(16), Some(32)),
            DistanceMetric::Cosine,
            spindle.clone(),
            Some(dir.path()),
            DIM,
            Arc::clone(&backend),
        )
        .expect("create compact scalar LSM vector core");

        {
            let mut w = backend.begin_write().unwrap();
            core.insert_flat_be(&mut w, &[1.0, 0.0, 0.0, 0.0], Some(41), None)
                .unwrap();
            core.insert_flat_be(&mut w, &[0.0, 1.0, 0.0, 0.0], Some(42), None)
                .unwrap();
            core.insert_flat_be(&mut w, &[0.0, 0.0, 1.0, 0.0], Some(43), None)
                .unwrap();
            backend.commit(w).unwrap();
        }

        let hspn_path = dir.path().join(segment).with_extension("hspn");
        assert!(
            !hspn_path.exists(),
            "fixture must reopen from SlateDB rows, not a pre-existing sidecar"
        );

        let reopened = VectorCore::new_named_lsm_with_dir(
            segment,
            HNSWConfig::new(Some(8), Some(16), Some(32)),
            DistanceMetric::Cosine,
            spindle.clone(),
            Some(dir.path()),
            DIM,
            Arc::clone(&backend),
        )
        .expect("reopen compact scalar LSM vector core");

        assert_eq!(reopened.sidecar_format_label(), "none");
        assert!(!reopened.mmap_is_spindle_encoded());
        assert_eq!(reopened.mmap_store_count(), None);
        assert!(
            !hspn_path.exists(),
            "reopen should not create a compact encoded HSPN sidecar"
        );

        let AnyBackend::Lsm(lsm) = &*backend else {
            unreachable!("backend is LSM in this test");
        };
        let ro = backend.begin_read().unwrap();
        let query = project_for_search(&[0.0, 1.0, 0.0, 0.0], &spindle).unwrap();
        lsm.reset_read_count();
        let distance = reopened
            .score_neighbor_distance(&ro, &PreparedSpindleQuery::None, 42, 0, &query)
            .unwrap()
            .expect("materialized HSPN should score the neighbor");

        assert!(
            distance < 1e-6,
            "self-score should stay exact for ScalarInt8 decoded values; got {distance}"
        );
        assert_eq!(reopened.sidecar_format_label(), "hspn");
        assert!(reopened.mmap_is_spindle_encoded());
        assert_eq!(reopened.mmap_store_count(), Some(3));
        assert!(hspn_path.exists(), "first score should create HSPN sidecar");
        assert!(
            lsm.read_count() > 0,
            "first score should materialize from LSM"
        );

        lsm.reset_read_count();
        let distance = reopened
            .score_neighbor_distance(&ro, &PreparedSpindleQuery::None, 42, 0, &query)
            .unwrap()
            .expect("materialized HSPN should score the neighbor");
        assert!(
            distance < 1e-6,
            "self-score should stay exact for ScalarInt8 decoded values; got {distance}"
        );
        assert_eq!(
            lsm.read_count(),
            0,
            "ScalarInt8 sidecar scoring should use cached ordinals and mmap rows"
        );
    }

    #[test]
    fn lsm_exact_hvec_rebuilds_empty_existing_sidecar_from_slate_rows() {
        use crate::helix_engine::storage_core::backend_any::AnyBackend;
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::sync::Arc;
        use std::time::{SystemTime, UNIX_EPOCH};

        const DIM: usize = 4;

        let (_env, dir) = setup();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("empty-hvec-rebuild-{nanos}"))
                .expect("open in-memory LSM backend"),
        ));
        let segment = "dense__seg_000000";
        let core = VectorCore::new_named_lsm_with_dir(
            segment,
            HNSWConfig::new(Some(8), Some(16), Some(32)),
            DistanceMetric::Cosine,
            SpindleConfig {
                mode: SpindleMode::None,
                ..SpindleConfig::default()
            },
            Some(dir.path()),
            DIM,
            Arc::clone(&backend),
        )
        .expect("create LSM vector core");

        {
            let mut w = backend.begin_write().unwrap();
            core.insert_flat_be(&mut w, &[1.0, 0.0, 0.0, 0.0], Some(51), None)
                .unwrap();
            core.insert_flat_be(&mut w, &[0.0, 1.0, 0.0, 0.0], Some(52), None)
                .unwrap();
            backend.commit(w).unwrap();
        }

        let hvec_path = dir.path().join(segment).with_extension("hvec");
        let empty =
            crate::helix_engine::vector_core::mmap_vectors::MmapVectorStore::open_or_create(
                &hvec_path, DIM,
            )
            .expect("create empty legacy HVEC sidecar");
        assert_eq!(empty.count(), 0);
        drop(empty);

        let reopened = VectorCore::new_named_lsm_with_dir(
            segment,
            HNSWConfig::new(Some(8), Some(16), Some(32)),
            DistanceMetric::Cosine,
            SpindleConfig {
                mode: SpindleMode::None,
                ..SpindleConfig::default()
            },
            Some(dir.path()),
            DIM,
            Arc::clone(&backend),
        )
        .expect("reopen LSM vector core with empty sidecar");

        assert_eq!(reopened.sidecar_format_label(), "none");
        assert_eq!(reopened.mmap_store_count(), None);

        let AnyBackend::Lsm(lsm) = &*backend else {
            unreachable!("backend is LSM in this test");
        };
        let ro = backend.begin_read().unwrap();
        lsm.reset_read_count();
        let distance = reopened
            .score_neighbor_distance(
                &ro,
                &PreparedSpindleQuery::None,
                52,
                0,
                &[0.0, 1.0, 0.0, 0.0],
            )
            .unwrap()
            .expect("rebuilt sidecar should score the neighbor");

        assert!(distance < 1e-6);
        assert_eq!(reopened.sidecar_format_label(), "hvec");
        assert_eq!(reopened.mmap_store_count(), Some(2));
        assert!(lsm.read_count() > 0, "first score should rebuild from LSM");

        lsm.reset_read_count();
        let distance = reopened
            .score_neighbor_distance(
                &ro,
                &PreparedSpindleQuery::None,
                52,
                0,
                &[0.0, 1.0, 0.0, 0.0],
            )
            .unwrap()
            .expect("rebuilt sidecar should score the neighbor");
        assert!(distance < 1e-6);
        assert_eq!(
            lsm.read_count(),
            0,
            "rebuilt empty sidecar should use cached ordinals and mmap rows"
        );
    }

    #[test]
    fn lsm_no_sidecar_level0_reads_skip_ordinal_probe() {
        use crate::helix_engine::storage_core::backend_any::AnyBackend;
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::sync::Arc;
        use std::time::{SystemTime, UNIX_EPOCH};

        const DIM: usize = 4;

        let (env, _dir) = setup();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = format!("vector-no-sidecar-read-budget-{nanos}");
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&path).expect("open in-memory LSM backend"),
        ));

        let mut txn = env.write_txn().unwrap();
        let core = VectorCore::new_named_with_dir(
            &env,
            &mut txn,
            "dense",
            HNSWConfig::new(Some(8), Some(32), Some(64)),
            DistanceMetric::Cosine,
            SpindleConfig {
                mode: SpindleMode::None,
                ..SpindleConfig::default()
            },
            None,
            DIM,
            None,
            Arc::clone(&backend),
        )
        .unwrap();
        txn.commit().unwrap();

        {
            let mut w = backend.begin_write().unwrap();
            core.insert_flat_be(&mut w, &[1.0, 0.0, 0.0, 0.0], Some(41), None)
                .unwrap();
            core.insert_flat_be(&mut w, &[0.0, 1.0, 0.0, 0.0], Some(42), None)
                .unwrap();
            backend.commit(w).unwrap();
        }

        let AnyBackend::Lsm(lsm) = &*backend else {
            unreachable!("backend is LSM in this test");
        };
        let ro = backend.begin_read().unwrap();

        lsm.reset_read_count();
        let vector = core.get_vector(&ro, 41, 0, true).unwrap();
        assert_eq!(vector.get_data(), &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            lsm.read_count(),
            1,
            "no-sidecar get_vector should read only the vector row, not an ordinal miss first"
        );

        let mut request_cache = SidecarOrdinalRequestCache::new();
        lsm.reset_read_count();
        let vector = core
            .get_vector_with_ordinal_cache(&ro, 41, 0, true, &mut request_cache)
            .unwrap();
        assert_eq!(vector.get_data(), &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            lsm.read_count(),
            0,
            "no-sidecar cache-aware hydration should preserve plain get_vector's global cache hit"
        );

        lsm.reset_read_count();
        let distance = core
            .score_neighbor_distance(
                &ro,
                &PreparedSpindleQuery::None,
                42,
                0,
                &[0.0, 1.0, 0.0, 0.0],
            )
            .unwrap()
            .expect("stored vector should score");
        assert!(
            distance < 1e-6,
            "self-score should stay exact after skipping the ordinal probe; got {distance}"
        );
        assert_eq!(
            lsm.read_count(),
            1,
            "no-sidecar scorer should read only the vector row, not ordinal misses before fallback"
        );
    }

    #[test]
    fn lsm_missing_sidecar_materialization_is_cached() {
        use crate::helix_engine::storage_core::backend_any::AnyBackend;
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::sync::Arc;
        use std::time::{SystemTime, UNIX_EPOCH};

        const DIM: usize = 4;

        let (_env, dir) = setup();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("missing-sidecar-cache-{nanos}"))
                .expect("open in-memory LSM backend"),
        ));
        let core = VectorCore::new_named_lsm_with_dir(
            "dense__seg_000000",
            HNSWConfig::new(Some(8), Some(16), Some(32)),
            DistanceMetric::Cosine,
            SpindleConfig {
                mode: SpindleMode::None,
                ..SpindleConfig::default()
            },
            Some(dir.path()),
            DIM,
            Arc::clone(&backend),
        )
        .unwrap();

        let AnyBackend::Lsm(lsm) = &*backend else {
            unreachable!("backend is LSM in this test");
        };
        let ro = backend.begin_read().unwrap();

        lsm.reset_read_count();
        assert!(matches!(
            core.get_vector(&ro, 99, 0, true),
            Err(VectorError::VectorNotFound(_))
        ));
        let first_reads = lsm.read_count();
        assert!(
            first_reads > 1,
            "first miss should prove absence through materialization attempt"
        );

        lsm.reset_read_count();
        assert!(matches!(
            core.get_vector(&ro, 99, 0, true),
            Err(VectorError::VectorNotFound(_))
        ));
        assert_eq!(
            lsm.read_count(),
            1,
            "cached missing sidecar should skip repeated scans and ordinal probes"
        );
    }

    /// Item-1 regression: a cached sidecar miss must be permanent on the
    /// writer (0 TTL semantics via `is_reader_replica() == false`) but expire
    /// on reader replicas, where the writer can seal the sidecar upstream at
    /// any time.
    #[test]
    fn lsm_sidecar_miss_ttl_expiry_semantics() {
        // Pure TTL math: expired only when a TTL is set and enough time passed.
        assert!(
            !lsm_sidecar_miss_expired(100, 500, 0),
            "ttl 0 never expires"
        );
        assert!(!lsm_sidecar_miss_expired(100, 129, 30), "within ttl");
        assert!(lsm_sidecar_miss_expired(100, 130, 30), "at ttl boundary");
        assert!(lsm_sidecar_miss_expired(100, 500, 30), "past ttl");
        // Clock skew backwards must not underflow.
        assert!(!lsm_sidecar_miss_expired(500, 100, 30));
    }

    /// Item-1 regression: on a reader-replica backend the cached miss expires,
    /// so `lsm_sidecar_miss_is_cached` flips false and clears the entry;
    /// on the writer the same aged entry stays authoritative.
    #[test]
    fn lsm_sidecar_miss_cache_expires_only_on_reader() {
        use crate::helix_engine::storage_core::backend_any::AnyBackend;
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use crate::helix_engine::storage_core::backend_lsm_reader::LsmReader;
        use object_store::memory::InMemory;
        use std::sync::Arc;
        use std::time::{SystemTime, UNIX_EPOCH};

        const DIM: usize = 4;
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let store: Arc<dyn object_store::ObjectStore> = Arc::new(InMemory::new());
        let path = format!("helion/miss-ttl-{nanos}");
        // Writer must exist first so the reader can resolve a manifest.
        let writer_backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_with_store(&path, store.clone()).expect("open LSM writer"),
        ));
        let reader_backend = Arc::new(AnyBackend::LsmReader(
            LsmReader::open_with_store(&path, store).expect("open LSM reader"),
        ));

        let mk_core = |backend: &Arc<AnyBackend>, dir: &std::path::Path| {
            VectorCore::new_named_lsm_with_dir(
                "dense__seg_000000",
                HNSWConfig::new(Some(8), Some(16), Some(32)),
                DistanceMetric::Cosine,
                SpindleConfig {
                    mode: SpindleMode::None,
                    ..SpindleConfig::default()
                },
                Some(dir),
                DIM,
                Arc::clone(backend),
            )
            .unwrap()
        };

        let (_env_w, dir_w) = setup();
        let writer_core = mk_core(&writer_backend, dir_w.path());
        let (_env_r, dir_r) = setup();
        let reader_core = mk_core(&reader_backend, dir_r.path());

        // Age the cached miss far beyond any TTL.
        let stale = unix_secs_now_nonzero().saturating_sub(86_400).max(1);
        writer_core
            .lsm_sidecar_miss_cached_at_secs
            .store(stale, AtomicOrdering::Relaxed);
        reader_core
            .lsm_sidecar_miss_cached_at_secs
            .store(stale, AtomicOrdering::Relaxed);

        assert!(
            writer_core.lsm_sidecar_miss_is_cached(),
            "writer miss never expires: local materialization is the only clearer"
        );
        assert!(
            !reader_core.lsm_sidecar_miss_is_cached(),
            "reader miss must expire so the sealed-upstream sidecar is re-probed"
        );
        assert_eq!(
            reader_core
                .lsm_sidecar_miss_cached_at_secs
                .load(AtomicOrdering::Relaxed),
            0,
            "expired reader entry is cleared so only one re-probe pays the scan"
        );
    }

    /// Item-3 regression: the process-wide materialization permit caps
    /// concurrent first-touch scans; capped probes defer instead of queueing.
    #[test]
    #[serial_test::serial]
    fn sidecar_materialize_permit_caps_and_releases() {
        // The cap is read from env at acquire time; hold permits and verify
        // acquisition fails at the cap and recovers after drop.
        let cap = sidecar_materialize_concurrency();
        assert!(cap >= 1, "default cap must admit at least one probe");
        let mut held = Vec::new();
        for _ in 0..cap {
            held.push(
                SidecarMaterializePermit::try_acquire()
                    .expect("under-cap acquisition must succeed"),
            );
        }
        assert!(
            SidecarMaterializePermit::try_acquire().is_none(),
            "at-cap acquisition must defer, not block"
        );
        held.pop();
        assert!(
            SidecarMaterializePermit::try_acquire().is_some(),
            "released permit must be reusable"
        );
    }

    #[test]
    #[serial_test::serial]
    fn lsm_sidecar_lock_waiter_does_not_consume_materialization_permit() {
        use crate::helix_engine::storage_core::backend_any::AnyBackend;
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::sync::{mpsc, Arc};
        use std::time::{Duration, SystemTime, UNIX_EPOCH};

        let cap = sidecar_materialize_concurrency();
        assert!(
            (1..=64).contains(&cap),
            "test requires a finite materialization cap, got {cap}"
        );
        assert_eq!(
            SIDECAR_MATERIALIZE_INFLIGHT.load(AtomicOrdering::Acquire),
            0,
            "serial permit test must start without an in-flight materialization"
        );

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("sidecar-permit-order-{nanos}"))
                .expect("open in-memory LSM backend"),
        ));
        let (_env, dir) = setup();
        let core = VectorCore::new_named_lsm_with_dir(
            "dense__seg_000000",
            HNSWConfig::new(Some(8), Some(16), Some(32)),
            DistanceMetric::Cosine,
            SpindleConfig {
                mode: SpindleMode::None,
                ..SpindleConfig::default()
            },
            Some(dir.path()),
            4,
            backend,
        )
        .unwrap();

        let held = (1..cap)
            .map(|_| {
                SidecarMaterializePermit::try_acquire()
                    .expect("cap-minus-one setup acquisition must succeed")
            })
            .collect::<Vec<_>>();
        let inflight_while_blocked = std::thread::scope(|scope| {
            let (lock_attempt_tx, lock_attempt_rx) = mpsc::sync_channel(0);
            let (continue_tx, continue_rx) = mpsc::sync_channel(0);
            let core_ref = &core;
            let worker = scope.spawn(move || {
                core_ref.ensure_lsm_mmap_sidecar_with_lock_observer(|| {
                    lock_attempt_tx
                        .send(())
                        .expect("test receiver must remain connected");
                    let _ = continue_rx.recv();
                })
            });

            lock_attempt_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("materialization worker must attempt the mmap write lock");
            let mmap_guard = core.mmap_store.write().unwrap();
            let inflight = SIDECAR_MATERIALIZE_INFLIGHT.load(AtomicOrdering::Acquire);

            continue_tx
                .send(())
                .expect("materialization worker must remain connected");
            drop(mmap_guard);
            assert!(
                !worker
                    .join()
                    .expect("materialization worker must not panic")
                    .expect("missing-sidecar probe must succeed"),
                "empty LSM backend should not materialize a sidecar"
            );
            inflight
        });
        assert_eq!(
            inflight_while_blocked,
            cap - 1,
            "a duplicate request blocked on the per-core lock must not consume the final global permit"
        );

        drop(held);
        assert_eq!(
            SIDECAR_MATERIALIZE_INFLIGHT.load(AtomicOrdering::Acquire),
            0,
            "all materialization permits must be released after the test"
        );
    }

    /// Regression: a GIANT dense segment's HNSW build must complete and return
    /// correct neighbors on a FILE-BACKED, SST-resident LSM collection whose
    /// object-store cache is far smaller than the corpus — i.e. the
    /// cache-thrash scenario that left `indexed_vectors_count` stuck at 0 in
    /// prod.
    ///
    /// Setup mirrors prod: a `LocalFileSystem` object store (reads come from
    /// on-disk SSTs, not RAM) with a deliberately tiny `HELIX_LSM_CACHE_*` cap so
    /// the corpus cannot stay cache-resident — every build read is a real SST
    /// read under cache pressure, exactly the prod giant-build scenario.
    ///
    /// Bounded-working-set proof: the build's backend reads are counted. The fix
    /// replaced the per-point random point-get loop (`get_vector_data` +
    /// `get_encoded_vector`, ~2 random gets/point = ~2N scattered SST fetches)
    /// with at most THREE ordered prefix scans (`get_all_vectors`,
    /// `collect_vector_data_map`, `collect_level_zero_encoded_map`). So the build
    /// must issue ZERO per-point `get_with` point reads — its entire read budget
    /// is sequential-scan KVs (~3N), and crucially never a per-point random read.
    /// We assert the build read count stays within a tight multiple of N (a
    /// sequential ceiling) AND that the result is correct, which together prove
    /// the build neither thrashed (it finished) nor corrupted results.
    #[test]
    #[serial_test::serial]
    fn lsm_giant_segment_build_bounded_working_set_sst_resident() {
        use crate::helix_engine::storage_core::backend_any::AnyBackend;
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use slatedb::object_store::local::LocalFileSystem;
        use slatedb::object_store::ObjectStore;
        use std::sync::Arc;
        use std::time::{SystemTime, UNIX_EPOCH};

        /// Restore-on-drop env guard (local to this test; the storage_core
        /// `TestEnvRestore` is private to that module).
        struct EnvGuard {
            key: &'static str,
            prev: Option<String>,
        }
        impl EnvGuard {
            fn set(key: &'static str, val: &str) -> Self {
                let prev = std::env::var(key).ok();
                std::env::set_var(key, val);
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

        const N: usize = 4000;
        const DIM: usize = 256;

        // SST-resident store + a tiny object-store cache (1 MiB total across the
        // default 256-collection divisor → a few KiB per collection), far smaller
        // than the corpus (N×DIM×4 ≈ 4 MiB of raw vectors plus SST overhead), so
        // it cannot stay cache-resident: every build read is a real SST/cold read,
        // exactly the prod giant-build pressure.
        let cache_dir = TempDir::new().unwrap();
        let _g_dir = EnvGuard::set("HELIX_LSM_CACHE_DIR", cache_dir.path().to_str().unwrap());
        let _g_cap = EnvGuard::set("HELIX_LSM_CACHE_MAX_BYTES", "1048576");

        let store_dir = TempDir::new().unwrap();
        let store: Arc<dyn ObjectStore> =
            Arc::new(LocalFileSystem::new_with_prefix(store_dir.path()).unwrap());

        let (env, _dir) = setup();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = format!("helion/giant-build-{nanos}");
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_with_store(&path, store.clone())
                .expect("open file-backed LSM backend"),
        ));

        // Raw spindle (no compression): exact f32 rows live in `Vectors`, so KNN
        // self-distance is exactly 0 and the `Vectors` sequential-scan prefetch
        // (`collect_level_zero_encoded_map`) carries the real encoded payloads.
        // `keep_original=true` persists the original into `VectorData`, exercising
        // the `collect_vector_data_map` prefetch as well. Mirrors a giant raw
        // dense segment — the prod cache-thrash case.
        let spindle = SpindleConfig {
            mode: SpindleMode::None,
            keep_original: true,
            ..SpindleConfig::default()
        };
        let mut txn = env.write_txn().unwrap();
        let core = VectorCore::new_named_with_dir(
            &env,
            &mut txn,
            "dense",
            // ef_construct=200 / ef=200: high build + search quality so recall on
            // a random corpus is robust (the test asserts correctness, not the
            // approximation envelope — that is covered by the bounded test above).
            HNSWConfig::new(Some(16), Some(200), Some(200)),
            DistanceMetric::Cosine,
            spindle,
            None,
            DIM,
            None,
            Arc::clone(&backend),
        )
        .unwrap();
        txn.commit().unwrap();

        // Deterministic corpus of independent random vectors. In high dimension
        // distinct random vectors are near-orthogonal (cosine ~1), so each point's
        // own vector is its STRICT unique nearest neighbor (self cosine distance
        // ~0) and HNSW recalls it reliably — no near-parallel cluster to get lost
        // in. This makes the brute-force ground-truth check below deterministic.
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let mut originals: Vec<Vec<f32>> = Vec::with_capacity(N);
        {
            let mut w = backend.begin_write().unwrap();
            for id in 1..=N as u128 {
                let data: Vec<f32> = (0..DIM)
                    .map(|_| rand::Rng::random_range(&mut rng, -1.0f32..1.0f32))
                    .collect();
                core.insert_flat_be(&mut w, &data, Some(id), None).unwrap();
                originals.push(data);
            }
            backend.commit(w).unwrap();
        }

        // Force the rows out to durable SSTs and reopen with an empty memtable so
        // the build reads from SSTs (cold), not from an in-memory write buffer.
        if let AnyBackend::Lsm(b) = backend.as_ref() {
            b.flush_durable().unwrap();
        }

        let AnyBackend::Lsm(lsm) = &*backend else {
            unreachable!("backend is LSM in this test");
        };

        // ── BUILD under cache pressure, counting backend reads ──
        lsm.reset_read_count();
        let prepared = {
            let ro = backend.begin_read().unwrap();
            let prepared = core
                .prepare_index_from_flat(&ro)
                .unwrap()
                .expect("prepared HNSW index over the SST-resident LSM corpus");
            drop(ro);
            prepared
        };
        let build_reads = lsm.read_count();

        // Bounded-working-set proof. The fix prefetches with at most three
        // ordered scans (get_all_vectors + two collect_* helpers), each yielding
        // ~N KVs, so the build budget is ~3N sequential reads. The OLD per-point
        // loop did the same ~3N but as ~2N RANDOM point-gets scattered across the
        // SSTs (the thrash). A defensive ceiling of 6N catches any accidental
        // reintroduction of per-point random reads or an O(N^2) blowup while
        // staying clear of sequential-scan overhead.
        assert!(
            build_reads <= 6 * N,
            "build read budget exceeded the bounded sequential-scan ceiling: \
             {build_reads} reads for N={N} (cap {}); the fix must prefetch via \
             ordered scans, not per-point random gets",
            6 * N
        );

        // The build actually produced a full index (not a stalled/empty one).
        assert_eq!(
            prepared.point_ids.len(),
            N,
            "prepared index must cover every seeded point"
        );

        // Flush + finalize through the real LSM optimizer arm, then promote.
        {
            let mut prepared = prepared;
            let mut w = backend.begin_write().unwrap();
            let total = prepared.point_ids.len();
            core.flush_prepared_index_chunk_be(&mut w, &mut prepared, 0, total)
                .unwrap();
            core.finalize_prepared_index_entry_be(&mut w, &prepared)
                .unwrap();
            backend.commit(w).unwrap();
        }
        if let AnyBackend::Lsm(b) = backend.as_ref() {
            b.flush_durable().unwrap();
        }

        // ── CORRECTNESS: the build is INDEXED and KNN is valid ──
        let ro = backend.begin_read().unwrap();
        assert!(
            core.has_index(&ro).unwrap(),
            "giant segment must publish an HNSW entry point (indexed), not stall at 0"
        );

        // Deterministic exactness guarantee: the HNSW entry point is the search
        // start (scored first in every query), so querying with its own vector
        // ALWAYS surfaces it, and its exact self-distance is ~0. A corrupt graph
        // or mis-prefetched vector row would break this self-match. (Same
        // deterministic anchor the existing bounded test uses.)
        let entry = core.get_entry_point(&ro).expect("entry point after build");
        let entry_id = entry.get_id();
        let entry_query = originals[(entry_id - 1) as usize].clone();
        let entry_results = core
            .search::<VF>(&ro, &entry_query, 5, None, false)
            .unwrap();
        let entry_hit = entry_results
            .iter()
            .find(|v| v.get_id() == entry_id)
            .unwrap_or_else(|| {
                panic!(
                    "entry-point self-query must recall the entry point (id={entry_id}); got {:?}",
                    entry_results.iter().map(|h| h.get_id()).collect::<Vec<_>>()
                )
            });
        assert!(
            entry_hit.get_distance() < 1e-3,
            "entry-point self-match must have ~0 distance; got {}",
            entry_hit.get_distance()
        );

        let cosine = |a: &[f32], b: &[f32]| -> f32 {
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;
            for i in 0..a.len() {
                dot += a[i] as f64 * b[i] as f64;
                na += (a[i] as f64).powi(2);
                nb += (b[i] as f64).powi(2);
            }
            if na == 0.0 || nb == 0.0 {
                return 1.0;
            }
            (1.0 - dot / (na.sqrt() * nb.sqrt())) as f32
        };

        // Broader correctness over a deterministic sample. Two corruption-sensitive
        // checks that do NOT depend on HNSW's approximate recall:
        //
        //  (1) DATA INTEGRITY — for every returned neighbor, recompute the cosine
        //      distance from the brute-force originals and assert it matches the
        //      distance the index reported. This is the direct guard for THIS fix:
        //      the change is purely about HOW the build prefetches each point's
        //      vector row; a mis-prefetched/corrupt row would make the indexed
        //      vector (and thus the reported distance) diverge from the original.
        //  (2) RECALL not collapsed — track how often the exact self-match is in
        //      the top-k. HNSW is approximate so this need not be perfect, but a
        //      corrupt graph would collapse it toward zero; require a solid floor.
        let sample_ids: [u128; 8] = [1, 7, 101, 500, 999, 1500, 2750, N as u128];
        let mut self_recall = 0usize;
        for &id in &sample_ids {
            let query = &originals[(id - 1) as usize];
            let results = core.search::<VF>(&ro, query, 5, None, false).unwrap();
            assert!(
                !results.is_empty(),
                "indexed search for id={id} returned nothing"
            );
            for hit in &results {
                let d = hit.get_distance();
                assert!(
                    d.is_finite() && d >= -1e-4,
                    "indexed search returned an invalid distance {d} (id={id}); a \
                     mis-prefetched/corrupt vector row would produce this"
                );
                // (1) The reported distance must equal cosine(query, the stored
                // original for that id). Divergence == corrupted/mis-read vector.
                let truth = cosine(query, &originals[(hit.get_id() - 1) as usize]);
                assert!(
                    (d - truth).abs() < 1e-3,
                    "neighbor id={} for query id={id} reported distance {d} but its \
                     stored original scores {truth}: the indexed vector diverged from \
                     the source (mis-prefetch/corruption is the failure mode of this fix)",
                    hit.get_id()
                );
            }
            if let Some(hit) = results.iter().find(|v| v.get_id() == id) {
                assert!(
                    hit.get_distance() < 1e-3,
                    "self-match for id={id} must have ~0 distance; got {}",
                    hit.get_distance()
                );
                self_recall += 1;
            }
        }
        // (2) A navigable, uncorrupted graph recalls most exact self-matches.
        // (Approximate, so not all; corruption would crater this toward 0.)
        assert!(
            self_recall >= 5,
            "self-recall collapsed to {self_recall}/{} — the bounded build produced a \
             corrupt or unnavigable graph",
            sample_ids.len()
        );

        eprintln!(
            "lsm_giant_segment_build_bounded_working_set_sst_resident: N={N} DIM={DIM} \
             build_reads={build_reads} (<= {} sequential-scan ceiling; the fix turns \
             ~2N random point-gets into ordered scans), entry-point self-match exact, \
             every returned distance == brute-force ground truth, self_recall={self_recall}/{}",
            6 * N,
            sample_ids.len()
        );
    }

    // ───────────────────────── IVF posting-list index ─────────────────────────

    /// Seeded LCG so IVF test data is deterministic without touching the
    /// process-global rand state.
    fn ivf_lcg_vectors(seed: u64, n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (u32::MAX >> 1) as f32) - 1.0
        };
        (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
    }

    /// Flat-insert `vectors` with ids `1..=n` and commit.
    fn ivf_insert_flat(env: &heed3::Env, core: &VectorCore, vectors: &[Vec<f32>]) {
        let mut txn = env.write_txn().unwrap();
        for (i, data) in vectors.iter().enumerate() {
            core.insert_flat(&mut txn, data, Some((i + 1) as u128), None)
                .unwrap();
        }
        txn.commit().unwrap();
    }

    fn ivf_build(env: &heed3::Env, core: &VectorCore) {
        let mut txn = env.write_txn().unwrap();
        let permit = acquire_build_permit().unwrap();
        core.build_ivf_from_flat_with_permit(&mut txn, &permit)
            .unwrap();
        txn.commit().unwrap();
    }

    fn ivf_ns_key_count(env: &heed3::Env, core: &VectorCore, db: SegmentDb) -> usize {
        let txn = env.read_txn().unwrap();
        let rd = core.backend.read_borrowed(&txn);
        let mut count = 0usize;
        core.backend
            .scan(&rd, core.seg_ns(db), KeyRange::all(), |_k, _v| {
                count += 1;
                true
            })
            .unwrap();
        count
    }

    /// Cosine distance matching `DistanceMetric::Cosine` semantics for the
    /// brute-force ground truth.
    fn ivf_cosine_dist(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        1.0 - dot / (na * nb)
    }

    #[test]
    fn ivf_recall_vs_bruteforce() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "ivf_recall", SpindleMode::None);
        let vectors = ivf_lcg_vectors(1, 2000, 16);
        ivf_insert_flat(&env, &core, &vectors);
        ivf_build(&env, &core);

        let txn = env.read_txn().unwrap();
        let rd = core.backend.read_borrowed(&txn);
        assert_eq!(core.index_mode(&rd).unwrap(), IndexMode::Ivf);

        let queries = ivf_lcg_vectors(2, 20, 16);
        let mut hits = 0usize;
        for query in &queries {
            let mut exact: Vec<(f32, u128)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (ivf_cosine_dist(query, v), (i + 1) as u128))
                .collect();
            exact.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal));
            let exact_top: Vec<u128> = exact.iter().take(10).map(|(_, id)| *id).collect();

            // Uniform random vectors are the worst case for cluster
            // separation, so probe wider than the persisted default; this
            // asserts posting-list quality at a fixed nprobe.
            let got = core
                .search_ivf::<VF>(&rd, query, 10, None, None, Some(16))
                .unwrap();
            assert!(got.len() <= 10);
            hits += got
                .iter()
                .filter(|v| exact_top.contains(&v.get_id()))
                .count();
        }
        let recall = hits as f32 / (queries.len() * 10) as f32;
        assert!(
            recall >= 0.9,
            "ivf recall@10 vs brute force = {recall} (< 0.9)"
        );
    }

    #[test]
    fn ivf_persistence_roundtrip() {
        let (env, _dir) = setup();
        let vectors = ivf_lcg_vectors(3, 400, 16);
        let query = ivf_lcg_vectors(4, 1, 16).remove(0);

        let before: Vec<(u128, f32)> = {
            let core = core_with_mode(&env, "ivf_persist", SpindleMode::None);
            ivf_insert_flat(&env, &core, &vectors);
            ivf_build(&env, &core);
            let txn = env.read_txn().unwrap();
            let rd = core.backend.read_borrowed(&txn);
            core.search_with_selectivity::<VF>(&rd, &query, 10, None, false, None)
                .unwrap()
                .iter()
                .map(|v| (v.get_id(), v.get_distance()))
                .collect()
        };
        assert!(!before.is_empty());

        // Reopen a fresh core over the same backend/name: the IVF artifacts
        // must be discovered from storage alone.
        let reopened = core_with_mode(&env, "ivf_persist", SpindleMode::None);
        let txn = env.read_txn().unwrap();
        let rd = reopened.backend.read_borrowed(&txn);
        assert_eq!(reopened.index_mode(&rd).unwrap(), IndexMode::Ivf);
        let after: Vec<(u128, f32)> = reopened
            .search_with_selectivity::<VF>(&rd, &query, 10, None, false, None)
            .unwrap()
            .iter()
            .map(|v| (v.get_id(), v.get_distance()))
            .collect();
        assert_eq!(before, after);
    }

    #[test]
    fn default_off_zero_change() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "ivf_default_off", SpindleMode::None);
        let vectors = ivf_lcg_vectors(5, 300, 16);
        ivf_insert_flat(&env, &core, &vectors);

        // Default (hnsw / unset) build path: no ivf_* keys may appear.
        let mut txn = env.write_txn().unwrap();
        core.build_index_from_flat(&mut txn).unwrap();
        txn.commit().unwrap();

        {
            let txn = env.read_txn().unwrap();
            let rd = core.backend.read_borrowed(&txn);
            assert!(core.has_index(&rd).unwrap());
            assert_eq!(core.index_mode(&rd).unwrap(), IndexMode::Hnsw);
        }
        assert_eq!(ivf_ns_key_count(&env, &core, SegmentDb::IvfCentroids), 0);
        assert_eq!(ivf_ns_key_count(&env, &core, SegmentDb::IvfPostings), 0);
    }

    #[test]
    fn ivf_respects_filter_predicate() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "ivf_filter", SpindleMode::None);
        let vectors = ivf_lcg_vectors(6, 300, 16);
        ivf_insert_flat(&env, &core, &vectors);
        ivf_build(&env, &core);

        let txn = env.read_txn().unwrap();
        let rd = core.backend.read_borrowed(&txn);
        let even_only: VF = |v| v.get_id() % 2 == 0;
        let results = core
            .search_with_selectivity(&rd, &vectors[0], 10, Some(&[even_only]), false, None)
            .unwrap();
        assert!(!results.is_empty());
        assert!(
            results.iter().all(|v| v.get_id() % 2 == 0),
            "filtered ivf search leaked excluded ids: {:?}",
            results.iter().map(|v| v.get_id()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn ivf_reaper_drains_keys() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "ivf_reaper", SpindleMode::None);
        let vectors = ivf_lcg_vectors(7, 200, 16);
        ivf_insert_flat(&env, &core, &vectors);
        ivf_build(&env, &core);
        assert!(ivf_ns_key_count(&env, &core, SegmentDb::IvfCentroids) > 0);
        assert!(ivf_ns_key_count(&env, &core, SegmentDb::IvfPostings) > 0);

        // Drive the chunked reaper drain over every per-segment DB index.
        let mut txn = env.write_txn().unwrap();
        for db_index in 0..VectorCore::REAPER_DB_COUNT {
            while !core.clear_chunk_db(&mut txn, db_index, 64).unwrap() {}
        }
        txn.commit().unwrap();
        core.clear_caches();

        assert_eq!(ivf_ns_key_count(&env, &core, SegmentDb::IvfCentroids), 0);
        assert_eq!(ivf_ns_key_count(&env, &core, SegmentDb::IvfPostings), 0);
        let txn = env.read_txn().unwrap();
        let rd = core.backend.read_borrowed(&txn);
        assert_eq!(core.index_mode(&rd).unwrap(), IndexMode::Flat);
    }

    #[test]
    fn ivf_small_n_edge() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "ivf_small", SpindleMode::None);
        let vectors = ivf_lcg_vectors(8, 5, 8);
        ivf_insert_flat(&env, &core, &vectors);
        ivf_build(&env, &core);

        let txn = env.read_txn().unwrap();
        let rd = core.backend.read_borrowed(&txn);
        assert_eq!(core.index_mode(&rd).unwrap(), IndexMode::Ivf);
        let results = core
            .search_with_selectivity::<VF>(&rd, &vectors[2], 3, None, false, None)
            .unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 3);
        assert!(results.iter().all(|v| v.get_distance().is_finite()));
        // The query is a stored vector: its own id must be the top hit.
        assert_eq!(results[0].get_id(), 3);
    }

    // ── Review-fix regressions (HNSW params, metric, pruning, deletes) ──

    fn core_with_config(
        env: &heed3::Env,
        name: &str,
        config: HNSWConfig,
        metric: DistanceMetric,
    ) -> VectorCore {
        let mut txn = env.write_txn().unwrap();
        let core = VectorCore::new_named(
            env,
            &mut txn,
            name,
            config,
            metric,
            SpindleConfig {
                mode: SpindleMode::None,
                ..SpindleConfig::default()
            },
            test_backend(env),
        )
        .unwrap();
        txn.commit().unwrap();
        core
    }

    #[test]
    fn hnsw_overrides_validate_rejects_degenerate_params() {
        for m in [0usize, 1, MAX_HNSW_M + 1] {
            let overrides = HnswOverrides {
                m: Some(m),
                ..HnswOverrides::default()
            };
            assert!(overrides.validate().is_err(), "m={m} must be rejected");
        }
        for m in [MIN_HNSW_M, 16, MAX_HNSW_M] {
            let overrides = HnswOverrides {
                m: Some(m),
                ..HnswOverrides::default()
            };
            assert!(overrides.validate().is_ok(), "m={m} must be accepted");
        }
        assert!(HnswOverrides {
            ef_construction: Some(0),
            ..HnswOverrides::default()
        }
        .validate()
        .is_err());
        assert!(HnswOverrides {
            ef: Some(0),
            ..HnswOverrides::default()
        }
        .validate()
        .is_err());
        assert!(HnswOverrides::default().validate().is_ok());

        // Already-persisted bad overrides are clamped, never yielding inf m_l.
        for m in [0usize, 1] {
            let config = HNSWConfig::new(Some(m), None, None);
            assert_eq!(config.m, MIN_HNSW_M);
            assert!(config.m_l.is_finite() && config.m_l > 0.0);
        }
    }

    #[test]
    fn in_memory_merge_build_dispatches_distance_metric() {
        // Points on the diagonal ray, positions shuffled relative to
        // ordinals: under Euclid each point's true neighbors are its
        // positional neighbors. Every SQ8 code is a multiple of [1,1,1,1], so a
        // cosine-built graph sees all-equal distances and is non-local.
        let n = 200usize;
        let mut positions: Vec<usize> = (0..n).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        for i in (1..n).rev() {
            let j = rand::Rng::random_range(&mut rng, 0..=i);
            positions.swap(i, j);
        }
        let exported: Vec<(u128, Vec<f32>, HashMap<String, Value>)> = positions
            .iter()
            .enumerate()
            .map(|(i, &x)| (i as u128, vec![(x + 1) as f32; 4], HashMap::new()))
            .collect();
        let config = HNSWConfig::new(Some(4), Some(64), Some(64));
        let permit = acquire_build_permit().unwrap();
        let prepared = VectorCore::build_hnsw_in_memory_owned_with_permit(
            exported,
            &config,
            DistanceMetric::Euclid,
            &permit,
        )
        .unwrap();

        let bound = (2 * config.m_max_0) as i64;
        let mut local = 0usize;
        for (ord, adjacency) in prepared.adjacency.iter().enumerate() {
            let level0 = &adjacency.as_ref().unwrap()[0];
            if level0.iter().all(|&other| {
                (positions[other as usize] as i64 - positions[ord] as i64).abs() <= bound
            }) {
                local += 1;
            }
        }
        assert!(
            local * 100 >= n * 90,
            "euclid build should link points to nearby points: {local}/{n} local"
        );
    }

    /// Points on the unit circle at shuffled angular positions `0..n`. Raw
    /// Dot equals cosine here, so true neighbors are angular neighbors. SQ8
    /// shifts both dimensions to non-negative codes, which makes every point
    /// prefer partners toward the 45-degree direction instead.
    fn dot_circle_rows(seed: u64, n: usize) -> (Vec<usize>, Vec<Vec<f32>>) {
        let mut positions: Vec<usize> = (0..n).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        for i in (1..n).rev() {
            let j = rand::Rng::random_range(&mut rng, 0..=i);
            positions.swap(i, j);
        }
        let rows = positions
            .iter()
            .map(|&p| {
                let theta = p as f32 * std::f32::consts::TAU / n as f32;
                vec![theta.cos(), theta.sin()]
            })
            .collect();
        (positions, rows)
    }

    /// Fraction (percent) of points whose level-0 neighbors are all within
    /// `bound` angular positions on a circle of `n`.
    fn circle_locality_percent(
        n: usize,
        bound: usize,
        level0: impl Iterator<Item = (usize, Vec<usize>)>,
    ) -> usize {
        let local = level0
            .filter(|(p, neighbors)| {
                neighbors.iter().all(|&q| {
                    let d = p.abs_diff(q);
                    d.min(n - d) <= bound
                })
            })
            .count();
        local * 100 / n
    }

    #[test]
    fn dot_merge_build_links_by_inner_product_not_sq8_codes() {
        let n = 200usize;
        let (positions, rows) = dot_circle_rows(17, n);
        let exported: Vec<(u128, Vec<f32>, HashMap<String, Value>)> = rows
            .into_iter()
            .enumerate()
            .map(|(i, row)| (i as u128, row, HashMap::new()))
            .collect();
        let config = HNSWConfig::new(Some(4), Some(64), Some(64));
        let permit = acquire_build_permit().unwrap();
        let prepared = VectorCore::build_hnsw_in_memory_owned_with_permit(
            exported,
            &config,
            DistanceMetric::Dot,
            &permit,
        )
        .unwrap();
        let level0 = prepared.adjacency.iter().enumerate().map(|(ord, adj)| {
            let neighbors = adj.as_ref().unwrap()[0]
                .iter()
                .map(|&o| positions[o as usize])
                .collect();
            (positions[ord], neighbors)
        });
        let pct = circle_locality_percent(n, 2 * config.m_max_0, level0);
        assert!(
            pct >= 90,
            "Dot merge build must follow inner products: {pct}% local"
        );
    }

    #[test]
    fn euclid_merge_build_respects_unequal_dimension_ranges() {
        // dim 0 spans [0, 2000), dim 1 only [0, 1): true L2 neighbors are
        // decided by dim 0. SQ8 normalizes each dimension to 0..=255, which
        // inflates the dim-1 noise to the same weight and breaks locality.
        let n = 200usize;
        let mut positions: Vec<usize> = (0..n).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(23);
        for i in (1..n).rev() {
            let j = rand::Rng::random_range(&mut rng, 0..=i);
            positions.swap(i, j);
        }
        let exported: Vec<(u128, Vec<f32>, HashMap<String, Value>)> = positions
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let noise = rand::Rng::random_range(&mut rng, 0.0f32..1.0f32);
                (i as u128, vec![x as f32 * 10.0, noise], HashMap::new())
            })
            .collect();
        let config = HNSWConfig::new(Some(4), Some(64), Some(64));
        let permit = acquire_build_permit().unwrap();
        let prepared = VectorCore::build_hnsw_in_memory_owned_with_permit(
            exported,
            &config,
            DistanceMetric::Euclid,
            &permit,
        )
        .unwrap();

        let bound = (2 * config.m_max_0) as i64;
        let mut local = 0usize;
        for (ord, adjacency) in prepared.adjacency.iter().enumerate() {
            let level0 = &adjacency.as_ref().unwrap()[0];
            if level0.iter().all(|&other| {
                (positions[other as usize] as i64 - positions[ord] as i64).abs() <= bound
            }) {
                local += 1;
            }
        }
        assert!(
            local * 100 >= n * 90,
            "euclid build must follow raw L2 geometry: {local}/{n} local"
        );
    }

    #[test]
    fn incremental_insert_respects_caps_and_keeps_true_neighbors() {
        let (env, _dir) = setup();
        let config = HNSWConfig::new(Some(4), Some(64), Some(64));
        let (m, m_max_0) = (config.m, config.m_max_0);
        let core = core_with_config(&env, "prune_caps", config, DistanceMetric::Euclid);

        let n = 300usize;
        let dim = 8usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let points: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| rand::Rng::random_range(&mut rng, -1.0f32..1.0f32))
                    .collect()
            })
            .collect();
        let mut txn = env.write_txn().unwrap();
        for (i, point) in points.iter().enumerate() {
            core.insert::<VF>(&mut txn, point, Some(i as u128 + 1), None)
                .unwrap();
        }
        txn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        let r = core.backend.read_borrowed(&txn);
        let mut has_true_nearest = 0usize;
        for (i, point) in points.iter().enumerate() {
            let id = i as u128 + 1;
            let top = core.get_highest_level(&r, id).unwrap();
            for level in 0..=top {
                let ids = core.get_neighbor_ids(&r, id, level).unwrap();
                let cap = if level == 0 { m_max_0 } else { m };
                assert!(
                    ids.len() <= cap,
                    "id {id} level {level} has {} links > cap {cap}",
                    ids.len()
                );
            }
            let nearest = (0..n)
                .filter(|&j| j != i)
                .min_by(|&a, &b| {
                    let da = core.distance_between(&points[a], point).unwrap();
                    let db = core.distance_between(&points[b], point).unwrap();
                    da.total_cmp(&db)
                })
                .unwrap() as u128
                + 1;
            if core.get_neighbor_ids(&r, id, 0).unwrap().contains(&nearest) {
                has_true_nearest += 1;
            }
        }
        assert!(
            has_true_nearest * 100 >= n * 90,
            "overflow pruning must keep each node's true nearest: {has_true_nearest}/{n}"
        );
    }

    #[test]
    fn flat_mmap_fast_path_excludes_delete_tombstoned_ids() {
        let (env, dir) = setup();
        let core =
            core_with_mode_and_mmap(&env, dir.path(), "flat_tomb_mmap", SpindleMode::None, 4);
        let mut txn = env.write_txn().unwrap();
        for id in 1..=8u128 {
            let v = id as f32;
            core.insert_flat(&mut txn, &[v, 1.0, 0.5, -0.5], Some(id), None)
                .unwrap();
        }
        core.build_index_from_flat(&mut txn).unwrap();
        txn.commit().unwrap();

        let query = [3.0f32, 1.0, 0.5, -0.5];
        let txn = env.read_txn().unwrap();
        let r = core.backend.read_borrowed(&txn);
        let before = core
            .search_flat_mmap_ordinals_direct::<fn(u128) -> bool>(&r, &query, 8, None)
            .unwrap()
            .expect("mmap ordinal fast path must be active for this core");
        assert!(before.iter().any(|v| v.get_id() == 3));

        core.apply_delete_tombstones(&[3]);
        let after = core
            .search_flat_mmap_ordinals_direct::<fn(u128) -> bool>(&r, &query, 8, None)
            .unwrap()
            .unwrap();
        assert!(!after.is_empty());
        assert!(!after.iter().any(|v| v.get_id() == 3));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    #[serial_test::serial]
    fn flat_hvtq_fastscan_excludes_delete_tombstoned_ids() {
        use crate::helix_engine::storage_core::backend::StorageBackend;

        let _fastscan = EnvGuard::set("HELIX_HVTQ_FASTSCAN", "1");
        let dir = TempDir::new().unwrap();
        let physical_name = "flat_tomb_hvtq";
        let hvtq_path = dir.path().join(physical_name).with_extension("hvtq");
        let vectors: Vec<Vec<f32>> = (0..8)
            .map(|i| {
                let t = i as f32 * 0.2;
                vec![t.cos(), t.sin(), 0.25, -0.25]
            })
            .collect();
        let slices = vectors.iter().map(Vec::as_slice).collect::<Vec<&[f32]>>();
        let spindle = SpindleConfig {
            mode: SpindleMode::TurboProd,
            keep_original: false,
            ..SpindleConfig::default()
        };
        drop(MmapTurboQuantStore::create_from_slices(&hvtq_path, &slices, &spindle).unwrap());

        let backend = Arc::new(AnyBackend::open_lsm_in_memory("/flat-tomb-hvtq").unwrap());
        let mut write = backend.begin_write().unwrap();
        for ordinal in 0..vectors.len() as u64 {
            let id = ordinal as u128 + 1;
            let ns = |db| Namespace::Segment { physical_name, db };
            backend
                .put(
                    &mut write,
                    ns(SegmentDb::Ordinals),
                    &id.to_be_bytes(),
                    &ordinal.to_le_bytes(),
                )
                .unwrap();
            backend
                .put(
                    &mut write,
                    ns(SegmentDb::Vectors),
                    &VectorCore::vector_key(id, 0),
                    EXTERNALIZED_VECTOR_MARKER,
                )
                .unwrap();
        }
        backend.commit(write).unwrap();

        let core = VectorCore::new_named_lsm_with_dir(
            physical_name,
            HNSWConfig::new(Some(8), Some(32), Some(64)),
            DistanceMetric::Cosine,
            spindle,
            Some(dir.path()),
            4,
            backend,
        )
        .unwrap();
        let prepared = prepare_query(&vectors[2], &core.spindle).unwrap();
        let read = core.backend.begin_read().unwrap();
        let before = core
            .search_flat_hvtq_fastscan_direct::<fn(u128) -> bool>(&read, &prepared, 8, None)
            .unwrap()
            .expect("hvtq fastscan path must be active");
        assert!(before.iter().any(|v| v.get_id() == 3));

        core.apply_delete_tombstones(&[3]);
        let after = core
            .search_flat_hvtq_fastscan_direct::<fn(u128) -> bool>(&read, &prepared, 8, None)
            .unwrap()
            .unwrap();
        assert!(!after.is_empty());
        assert!(!after.iter().any(|v| v.get_id() == 3));
    }

    /// A -> B -> C chain at level 0 with B delete-tombstoned: C is reachable
    /// only through B, so B must stay traversable while never being returned.
    fn tombstone_bridge_core(env: &heed3::Env, name: &str) -> VectorCore {
        let core = core_with_mode(env, name, SpindleMode::None);
        let mut txn = env.write_txn().unwrap();
        core.put_raw_vector(&mut txn, 1, 0, &[1.0, 0.0, 0.0, 0.0])
            .unwrap();
        core.put_raw_vector(&mut txn, 2, 0, &[0.7, 0.7, 0.0, 0.0])
            .unwrap();
        core.put_raw_vector(&mut txn, 3, 0, &[0.0, 1.0, 0.0, 0.0])
            .unwrap();
        core.put_neighbor_ids(&mut txn, 1, 0, &[2]).unwrap();
        core.put_neighbor_ids(&mut txn, 2, 0, &[1, 3]).unwrap();
        core.put_neighbor_ids(&mut txn, 3, 0, &[2]).unwrap();
        core.set_entry_point(
            &mut txn,
            &HVector::from_slice(1, 0, vec![1.0, 0.0, 0.0, 0.0]),
        )
        .unwrap();
        txn.commit().unwrap();
        core.apply_delete_tombstones(&[2]);
        core
    }

    #[test]
    fn hnsw_search_traverses_through_delete_tombstoned_nodes() {
        let (env, _dir) = setup();
        let core = tombstone_bridge_core(&env, "tomb_bridge");
        let txn = env.read_txn().unwrap();
        let r = core.backend.read_borrowed(&txn);
        let query = [0.0f32, 1.0, 0.0, 0.0];

        let results = core.search::<VF>(&r, &query, 2, None, false).unwrap();
        let ids: Vec<u128> = results.iter().map(|v| v.get_id()).collect();
        assert!(
            ids.contains(&3),
            "C must be reachable via tombstoned B: {ids:?}"
        );
        assert!(
            !ids.contains(&2),
            "tombstoned B must not be returned: {ids:?}"
        );

        let id_filter = |_id: u128| true;
        let results = core
            .search_with_id_filter_ef_observed(
                &r,
                &query,
                2,
                Some(&[id_filter]),
                false,
                None,
                None,
                None,
            )
            .unwrap();
        let ids: Vec<u128> = results.iter().map(|v| v.get_id()).collect();
        assert!(ids.contains(&3), "id-filter search must reach C: {ids:?}");
        assert!(!ids.contains(&2), "id-filter search returned tombstoned B");
    }

    #[test]
    fn delete_entry_point_promotes_highest_level_survivor() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "ep_promote", SpindleMode::None);
        insert_n(&env, &core, 300, 8);

        let (ep_id, max_survivor_level) = {
            let txn = env.read_txn().unwrap();
            let r = core.backend.read_borrowed(&txn);
            let ep = core.get_entry_point(&r).unwrap();
            let max_level = (1..=300u128)
                .filter(|&id| id != ep.get_id())
                .map(|id| core.get_highest_level(&r, id).unwrap())
                .max()
                .unwrap();
            (ep.get_id(), max_level)
        };

        let mut wtxn = env.write_txn().unwrap();
        core.delete_vector(&mut wtxn, ep_id).unwrap();
        wtxn.commit().unwrap();

        let txn = env.read_txn().unwrap();
        let r = core.backend.read_borrowed(&txn);
        assert!(core.has_index(&r).unwrap());
        let new_ep = core.get_entry_point(&r).unwrap();
        assert_ne!(new_ep.get_id(), ep_id);
        assert_eq!(
            new_ep.get_level(),
            max_survivor_level,
            "replacement entry point must keep the upper layers reachable"
        );
    }

    #[test]
    fn delete_isolated_entry_point_keeps_has_index_while_rows_remain() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "ep_isolated", SpindleMode::None);
        let mut txn = env.write_txn().unwrap();
        core.put_raw_vector(&mut txn, 1, 0, &[1.0, 0.0, 0.0, 0.0])
            .unwrap();
        core.put_raw_vector(&mut txn, 2, 0, &[0.0, 1.0, 0.0, 0.0])
            .unwrap();
        core.set_entry_point(
            &mut txn,
            &HVector::from_slice(1, 0, vec![1.0, 0.0, 0.0, 0.0]),
        )
        .unwrap();
        txn.commit().unwrap();

        let mut txn = env.write_txn().unwrap();
        core.delete_vector(&mut txn, 1).unwrap();
        txn.commit().unwrap();
        {
            let txn = env.read_txn().unwrap();
            let r = core.backend.read_borrowed(&txn);
            assert!(
                core.has_index(&r).unwrap(),
                "row 2 remains, index must stay"
            );
            assert_eq!(core.get_entry_point(&r).unwrap().get_id(), 2);
        }

        let mut txn = env.write_txn().unwrap();
        core.delete_vectors_batch(&mut txn, &[2]).unwrap();
        txn.commit().unwrap();
        let txn = env.read_txn().unwrap();
        assert!(!core.has_index(&core.backend.read_borrowed(&txn)).unwrap());
    }

    #[test]
    fn insert_rejects_non_finite_components() {
        let (env, _dir) = setup();
        let core = core_with_mode(&env, "non_finite", SpindleMode::None);
        let mut txn = env.write_txn().unwrap();
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let data = [1.0, bad, 0.0, 0.0];
            assert!(matches!(
                core.insert::<VF>(&mut txn, &data, Some(1), None),
                Err(VectorError::InvalidVectorData)
            ));
            assert!(matches!(
                core.insert_flat(&mut txn, &data, Some(1), None),
                Err(VectorError::InvalidVectorData)
            ));
        }
    }

    #[test]
    fn heap_orders_treat_nan_distance_as_farthest() {
        use super::super::heap_utils::{Candidate, MaxByDistance};
        let mut heap = BinaryHeap::new();
        for (id, distance) in [(1u128, 0.5f32), (2, f32::NAN), (3, 0.1), (4, 0.9)] {
            let mut v = HVector::from_slice(id, 0, vec![0.0]);
            v.set_distance(distance);
            heap.push(v);
        }
        let popped: Vec<u128> = std::iter::from_fn(|| heap.pop().map(|v| v.get_id())).collect();
        assert_eq!(popped, vec![3, 1, 4, 2]);

        let mut candidates: BinaryHeap<Candidate> = [0.5f32, f32::NAN, 0.1]
            .iter()
            .enumerate()
            .map(|(i, &distance)| Candidate {
                id: i as u128,
                distance,
            })
            .collect();
        assert_eq!(candidates.pop().unwrap().id, 2);
        assert_eq!(candidates.pop().unwrap().id, 0);
        assert_eq!(candidates.pop().unwrap().id, 1);

        let mut worst: BinaryHeap<MaxByDistance> = [0.5f32, f32::NAN, 0.1]
            .iter()
            .enumerate()
            .map(|(i, &distance)| {
                MaxByDistance(Candidate {
                    id: i as u128,
                    distance,
                })
            })
            .collect();
        assert_eq!(worst.pop().unwrap().0.id, 1, "NaN must be evicted first");
    }
}
