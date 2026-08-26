use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, LazyLock, Mutex, RwLock};
use std::time::Instant;

use heed3::{types::Bytes, Database, Env, RoTxn, RwTxn, WithTls};
use serde::{Deserialize, Serialize};

use crate::helix_engine::storage_core::backend::{KeyRange, Namespace, StorageBackend};
use crate::helix_engine::storage_core::backend_any::{AnyBackend, AnyRead, AnyWrite};
use crate::helix_engine::types::VectorError;
use crate::protocol::value::Value;

use super::{
    hnsw::HNSW,
    sparse::{SparseMetadataFlushStats, SparseVector, SparseVectorConfig, SparseVectorCore},
    spindle::SpindleConfig,
    vector::HVector,
    vector_core::{
        acquire_build_permit, delete_vacuum_fraction, lsm_delete_tombstones_enabled, HNSWConfig,
        VectorCore, VectorSearchMetrics,
    },
};

const LSM_CORE_OPEN_MAX_CONCURRENCY: usize = 8;

pub trait DenseReadTxnProvider: Sync {
    fn with_dense_read_txn<T, F>(&self, f: F) -> Result<T, VectorError>
    where
        F: FnOnce(&RoTxn) -> Result<T, VectorError>;
}

#[cfg(test)]
impl DenseReadTxnProvider for Env<WithTls> {
    fn with_dense_read_txn<T, F>(&self, f: F) -> Result<T, VectorError>
    where
        F: FnOnce(&RoTxn) -> Result<T, VectorError>,
    {
        let txn = self.read_txn()?;
        f(&txn)
    }
}

/// Configuration for a single named vector space.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NamedVectorConfig {
    pub size: usize,
    pub distance: DistanceMetric,
    #[serde(default)]
    pub spindle: SpindleConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetric {
    Cosine,
    Dot,
    Euclid,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum DenseVectorSegmentRole {
    #[default]
    Mutable,
    Building,
    Indexed,
}

/// ANN index algorithm used by `build_dense_segment` for newly built dense
/// segments. Derived at runtime from the environment — never stored in the
/// bincode-positional metadata structs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DenseIndexMode {
    Hnsw,
    Ivf,
}

/// Testable parser behind [`dense_index_mode`].
fn dense_index_mode_from_raw(raw: Option<&str>) -> DenseIndexMode {
    match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("ivf") => DenseIndexMode::Ivf,
        _ => DenseIndexMode::Hnsw,
    }
}

/// Env-gated dense index mode: `HELIX_DENSE_INDEX_MODE=ivf` opts a deployment
/// into SPANN/IVF posting-list builds; unset or any other value keeps HNSW
/// (default — zero behavior change). Resolved once per process.
pub(crate) fn dense_index_mode() -> DenseIndexMode {
    static MODE: std::sync::OnceLock<DenseIndexMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        dense_index_mode_from_raw(std::env::var("HELIX_DENSE_INDEX_MODE").ok().as_deref())
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct DenseVectorSegmentMetadata {
    pub physical_name: String,
    #[serde(default)]
    pub role: DenseVectorSegmentRole,
}

#[derive(Debug, Clone, Default)]
pub struct DenseDeletePlan {
    pub ids: Vec<u128>,
    active_segments: HashSet<String>,
    by_segment: Vec<(String, Vec<u128>)>,
}

impl DenseDeletePlan {
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// Ids written to `Namespace::DenseTombstones` inside a not-yet-committed
/// write batch, grouped by segment. Produced by
/// `delete_vectors_batch_with_plan_be`; must be handed back to
/// `apply_delete_tombstones` only after that batch commits.
#[derive(Debug, Default)]
pub struct StagedDeleteTombstones {
    by_segment: Vec<(String, Vec<u128>)>,
}

impl StagedDeleteTombstones {
    pub fn is_empty(&self) -> bool {
        self.by_segment.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct ExternalizedMarkerRepairPlan {
    physical_name: String,
    role: DenseVectorSegmentRole,
    // Candidate ids captured under a read txn. The write txn revalidates the
    // segment before purging because intervening writes can repair these ids.
    ids: Vec<u128>,
    limit_per_segment: usize,
}

impl ExternalizedMarkerRepairPlan {
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct HvtqSidecarBackfillPlan {
    physical_name: String,
    point_ids: Vec<u128>,
}

impl HvtqSidecarBackfillPlan {
    pub(crate) fn physical_name(&self) -> &str {
        &self.physical_name
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct DenseVectorSpaceMetadata {
    #[serde(default)]
    pub next_segment_id: u64,
    #[serde(default)]
    pub segments: Vec<DenseVectorSegmentMetadata>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DenseSegmentDebt {
    pub indexed_segments: usize,
    pub active_segments: usize,
    pub merge_debt: usize,
    pub gate_debt: usize,
    pub dirty_retired_segments: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DenseMergeCandidateLimits {
    max_indexed_segments: usize,
    max_fan_in: usize,
    bounded_prefix: bool,
}

impl DenseMergeCandidateLimits {
    pub(crate) const fn new(max_indexed_segments: usize, max_fan_in: usize) -> Self {
        Self {
            max_indexed_segments,
            max_fan_in,
            bounded_prefix: false,
        }
    }

    pub(crate) const fn bounded_prefix(max_indexed_segments: usize, max_fan_in: usize) -> Self {
        Self {
            max_indexed_segments,
            max_fan_in,
            bounded_prefix: true,
        }
    }

    pub(crate) fn max_fan_in(self) -> usize {
        self.max_fan_in.max(2)
    }

    pub(crate) fn bounded_scan_len(self) -> usize {
        self.max_fan_in().saturating_mul(8)
    }

    pub(crate) const fn uses_bounded_prefix(self) -> bool {
        self.bounded_prefix
    }
}

/// `DenseSegmentDebt` augmented with the re-upsert tombstone backlog.
///
/// Carried as a separate type (not a new field on `DenseSegmentDebt`) so the
/// existing struct literal in `replication.rs` keeps compiling — adding a
/// field there is the freshness/optimizer owner's call, not ours. The
/// optimizer calls `dense_segment_debt_with_tombstones` (which has a read txn)
/// to get the folded-in merge pressure; the txn-free `dense_segment_debt`
/// path is unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DenseSegmentDebtWithTombstones {
    pub debt: DenseSegmentDebt,
    /// Count of superseded vector copies awaiting reclamation.
    pub tombstoned_copies: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct SidecarCleanupStats {
    pub dry_run: bool,
    pub candidate_files: usize,
    pub skipped_active_files: usize,
    pub removed_files: usize,
    pub removed_bytes: u64,
    pub duration_ms: u64,
    pub error_count: usize,
    pub errors: Vec<String>,
}

impl SidecarCleanupStats {
    pub fn merge(&mut self, other: SidecarCleanupStats) {
        self.dry_run |= other.dry_run;
        self.candidate_files += other.candidate_files;
        self.skipped_active_files += other.skipped_active_files;
        self.removed_files += other.removed_files;
        self.removed_bytes += other.removed_bytes;
        self.duration_ms = self.duration_ms.saturating_add(other.duration_ms);
        self.error_count += other.error_count;
        self.errors.extend(other.errors);
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct SidecarQuantizeStats {
    pub checked_segments: usize,
    pub converted_segments: usize,
    pub skipped_segments: usize,
    pub duration_ms: u64,
    pub error_count: usize,
    pub errors: Vec<String>,
}

impl SidecarQuantizeStats {
    pub fn merge(&mut self, other: SidecarQuantizeStats) {
        self.checked_segments += other.checked_segments;
        self.converted_segments += other.converted_segments;
        self.skipped_segments += other.skipped_segments;
        self.duration_ms = self.duration_ms.saturating_add(other.duration_ms);
        self.error_count += other.error_count;
        self.errors.extend(other.errors);
    }
}

fn finish_sidecar_cleanup_stats(
    stats: &mut SidecarCleanupStats,
    started: Instant,
    mode: &'static str,
) {
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    stats.duration_ms = elapsed_ms.round() as u64;
    metrics::histogram!(
        "helix_sidecar_cleanup_duration_ms",
        "mode" => mode,
        "dry_run" => if stats.dry_run { "true" } else { "false" }
    )
    .record(elapsed_ms);
    if stats.error_count > 0 {
        metrics::counter!("helix_sidecar_cleanup_errors_total", "mode" => mode)
            .increment(stats.error_count as u64);
    }
}

/// Pre-computed merge result. Built outside the write lock by
/// `prepare_merge`, then flushed under a short write txn by
/// `flush_prepared_merge`. This avoids holding the write lock
/// during O(n log n) HNSW construction.
pub struct PreparedMerge {
    pub merge_targets: Vec<String>,
    pub prepared_index: super::vector_core::PreparedIndex,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseVectorStats {
    pub vectors_count: u64,
    pub indexed_vectors_count: u64,
    pub segments_count: usize,
    pub missing_segments_count: usize,
    pub missing_segments: Vec<String>,
}

fn next_segment_name(base: &str, next_id: u64) -> String {
    format!("{base}__seg_{next_id:06}")
}

/// Allocate a segment ID by reusing gaps in the active segment list.
/// With bounded merge (≤4 indexed + 1 mutable), most IDs below
/// `next_segment_id` are unused and their LMDB DB slots can be
/// reused via `create_database` (reopens the cleared DB, no new slot).
/// Only increments `next_segment_id` if no gaps exist.
fn segment_id_from_name(logical_name: &str, physical_name: &str) -> Option<u64> {
    if physical_name == logical_name {
        return Some(0);
    }
    let suffix = physical_name.strip_prefix(logical_name)?;
    let suffix = suffix.strip_prefix("__seg_")?;
    suffix.parse().ok()
}

fn reserved_segment_ids<'a>(
    space: &DenseVectorSpaceMetadata,
    logical_name: &str,
    reserved_physical_names: impl IntoIterator<Item = &'a str>,
) -> HashSet<u64> {
    let mut ids: HashSet<u64> = space
        .segments
        .iter()
        .filter_map(|s| segment_id_from_name(logical_name, &s.physical_name))
        .collect();
    ids.extend(
        reserved_physical_names
            .into_iter()
            .filter_map(|name| segment_id_from_name(logical_name, name)),
    );
    ids
}

#[allow(dead_code)]
fn alloc_segment_id<'a>(
    space: &mut DenseVectorSpaceMetadata,
    logical_name: &str,
    reserved_physical_names: impl IntoIterator<Item = &'a str>,
) -> u64 {
    let active_ids = reserved_segment_ids(space, logical_name, reserved_physical_names);
    // Reuse the lowest unused ID (its LMDB DB already exists but was cleared)
    for id in 0..space.next_segment_id {
        if !active_ids.contains(&id) {
            return id;
        }
    }
    // All IDs in [0..next_segment_id) are active, allocate new
    let id = space.next_segment_id;
    space.next_segment_id += 1;
    id
}

fn active_segment_name(space: &DenseVectorSpaceMetadata) -> Option<String> {
    space
        .segments
        .iter()
        .find(|segment| segment.role == DenseVectorSegmentRole::Mutable)
        .or_else(|| space.segments.first())
        .map(|segment| segment.physical_name.clone())
}

fn mutable_segment_name(space: &DenseVectorSpaceMetadata) -> Option<String> {
    space
        .segments
        .iter()
        .find(|segment| segment.role == DenseVectorSegmentRole::Mutable)
        .map(|segment| segment.physical_name.clone())
}

/// Order a space's physical segment names newest-first. The mutable tail is
/// always newest (re-upserts append there). Among non-mutable segments the
/// highest segment id is newest (ids are monotonic per `next_segment_id`).
/// Used by tombstone reconstruction to pick the surviving copy of each id.
fn segments_newest_first(logical_name: &str, space: &DenseVectorSpaceMetadata) -> Vec<String> {
    let mut segs: Vec<(&DenseVectorSegmentMetadata, u64)> = space
        .segments
        .iter()
        .map(|s| {
            let id = segment_id_from_name(logical_name, &s.physical_name).unwrap_or(0);
            (s, id)
        })
        .collect();
    // Mutable first, then by descending segment id. Tie-break on the physical
    // name so two unparseable names (both id 0) order deterministically rather
    // than by HashMap/Vec happenstance — keeps reconstruct winner selection
    // stable.
    segs.sort_by(|a, b| {
        let a_mut = a.0.role == DenseVectorSegmentRole::Mutable;
        let b_mut = b.0.role == DenseVectorSegmentRole::Mutable;
        b_mut
            .cmp(&a_mut)
            .then(b.1.cmp(&a.1))
            .then_with(|| a.0.physical_name.cmp(&b.0.physical_name))
    });
    segs.into_iter()
        .map(|(s, _)| s.physical_name.clone())
        .collect()
}

fn dense_segments_by_recency<'a>(
    logical_name: &str,
    segment_infos: &'a [(String, DenseVectorSegmentRole)],
) -> Vec<&'a (String, DenseVectorSegmentRole)> {
    let mut ordered: Vec<&(String, DenseVectorSegmentRole)> = segment_infos.iter().collect();
    ordered.sort_by(|a, b| {
        let a_mut = a.1 == DenseVectorSegmentRole::Mutable;
        let b_mut = b.1 == DenseVectorSegmentRole::Mutable;
        let a_id = segment_id_from_name(logical_name, &a.0).unwrap_or(0);
        let b_id = segment_id_from_name(logical_name, &b.0).unwrap_or(0);
        b_mut
            .cmp(&a_mut)
            .then(b_id.cmp(&a_id))
            .then_with(|| a.0.cmp(&b.0))
    });
    ordered
}

/// Recency rank for each `(physical_name, role)` in a search's segment set,
/// newest = smallest rank. Mutable tail ranks above all Indexed; among Indexed
/// the highest segment id is newest. Used by `dedup_keep_newest` so a
/// re-upserted id surfaces its NEWEST copy, never a stale sealed copy that
/// happens to sit closer to the query. Pure in-memory; no LMDB / tombstone
/// read on the search path.
fn segment_recency_ranks(
    logical_name: &str,
    segment_infos: &[(String, DenseVectorSegmentRole)],
) -> HashMap<String, u32> {
    let ordered = dense_segments_by_recency(logical_name, segment_infos);
    ordered
        .into_iter()
        .enumerate()
        .map(|(rank, (name, _))| (name.clone(), rank as u32))
        .collect()
}

fn unique_candidate_ids(candidate_ids: Vec<u128>) -> Vec<u128> {
    let mut seen = HashSet::with_capacity(candidate_ids.len());
    candidate_ids
        .into_iter()
        .filter(|id| seen.insert(*id))
        .collect()
}

fn exact_candidate_ids_present_in_segment(
    core: &VectorCore,
    r: &AnyRead<'_>,
    candidate_ids: &[u128],
) -> Result<Vec<u128>, VectorError> {
    let mut routed = Vec::new();
    for &id in candidate_ids {
        match core.get_vector(r, id, 0, false) {
            Ok(_) => routed.push(id),
            Err(VectorError::VectorNotFound(_)) => {}
            Err(err) => return Err(err),
        }
    }
    Ok(routed)
}

/// Collapse duplicate ids in a merged multi-segment result, keeping the copy
/// from the NEWEST segment (lowest recency rank); ties broken by smaller
/// distance. Then orders survivors by distance and truncates to `k`. This is
/// the read-path recency fix: while old and re-upserted copies of an id coexist
/// (the entire pre-merge window) a query nearer the OLD vector must still return
/// the NEW one. Cost: one `HashMap<u128,(rank,dist)>` pass over `combined`
/// (already O(results)); no extra LMDB reads.
fn dedup_keep_newest(mut combined: Vec<(u32, HVector)>, k: usize) -> Vec<HVector> {
    // best_rank[id] = (rank, distance) of the winning copy so far.
    let mut best: HashMap<u128, (u32, f32)> = HashMap::with_capacity(combined.len());
    for (rank, hv) in &combined {
        let id = hv.id;
        let dist = hv.get_distance();
        match best.get(&id) {
            Some(&(br, bd)) if br < *rank || (br == *rank && bd <= dist) => {}
            _ => {
                best.insert(id, (*rank, dist));
            }
        }
    }
    // Retain only the winning (rank, distance) copy per id.
    let mut taken: HashSet<u128> = HashSet::with_capacity(best.len());
    combined.retain(|(rank, hv)| {
        let id = hv.id;
        match best.get(&id) {
            Some(&(br, bd)) if br == *rank && bd == hv.get_distance() && taken.insert(id) => true,
            _ => false,
        }
    });
    let mut survivors: Vec<HVector> = combined.into_iter().map(|(_, hv)| hv).collect();
    survivors.sort_by(|lhs, rhs| {
        lhs.get_distance()
            .partial_cmp(&rhs.get_distance())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    survivors.truncate(k);
    survivors
}

/// Manages multiple named VectorCore instances within a single LMDB environment.
///
/// Dense vector spaces are logical names such as `dense` or `mini`. Each logical
/// space can now own multiple physical segment cores:
///
/// - one mutable flat segment for fresh writes
/// - zero or more building/indexed HNSW segments
///
/// Sparse vector spaces remain one core per logical name.
/// Sentinel: use global config threshold (no per-collection override).
const INDEXING_THRESHOLD_USE_GLOBAL: usize = usize::MAX;

/// Process-wide dedup set for "missing dense segment" warnings. Keyed by
/// `(collection, vector, segment)` so each stale segment is logged exactly
/// once per process lifetime. Subsequent observations still bump the
/// `helix_dense_segment_missing_total` counter so operators can see rate
/// via metrics, but stop spamming the log. Deliberately keeps the signal
/// observable per commit 55ca5c56 — missing segment refs must remain
/// visible to operators for targeted reindex decisions.
static MISSING_SEGMENT_REGISTRY: LazyLock<Mutex<HashSet<(String, String, String)>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// Record an observation of a missing dense segment. Always increments the
/// metric counter; emits a `warn!` only on the first observation of each
/// `(collection, vector, segment)` tuple in this process.
fn note_missing_segment(collection: &str, vector: &str, segment: &str, site: &'static str) {
    metrics::counter!(
        "helix_dense_segment_missing_total",
        "collection" => collection.to_string(),
        "vector" => vector.to_string(),
        "segment" => segment.to_string(),
    )
    .increment(1);
    let key = (
        collection.to_string(),
        vector.to_string(),
        segment.to_string(),
    );
    let first_sighting = match MISSING_SEGMENT_REGISTRY.lock() {
        Ok(mut set) => set.insert(key),
        // Poisoned mutex: fall through and log anyway rather than go silent.
        Err(_) => true,
    };
    if first_sighting {
        tracing::warn!(
            collection = %collection,
            vector = %vector,
            segment = %segment,
            site = %site,
            "Skipping missing dense segment (metadata references a segment with no live core; reindex the collection to repair)"
        );
    }
}

pub struct NamedVectorManager {
    /// Owning collection name (trailing path component of the LMDB env
    /// directory). Threaded through so missing-segment diagnostics can
    /// identify which tenant needs repair.
    collection_name: String,
    configs: RwLock<HashMap<String, NamedVectorConfig>>,
    dense_spaces: RwLock<HashMap<String, DenseVectorSpaceMetadata>>,
    cores: RwLock<HashMap<String, VectorCore>>,
    dirty_retired_segments: RwLock<HashSet<String>>,
    dirty_retired_segment_hook: RwLock<Option<Arc<dyn Fn(String) + Send + Sync + 'static>>>,
    sparse_configs: RwLock<HashMap<String, SparseVectorConfig>>,
    sparse_cores: RwLock<HashMap<String, SparseVectorCore>>,
    /// Per-collection indexing threshold override.
    /// - `usize::MAX` → use global config (default)
    /// - `0` → bulk mode: defer all indexing, inserts go flat
    /// - `N` → override threshold to N
    indexing_threshold_override: AtomicUsize,
    /// Graph out-edges DB handle from the owning `HelixGraphStorage`.
    ///
    /// Set via `attach_graph_out_edges_db` during storage construction and
    /// then cloned into every subsequently-created `VectorCore`. Previously
    /// constructed cores are left alone; in practice this is fine because
    /// all cores are created *after* storage finishes wiring up the graph
    /// DBs. Standalone test harnesses that build a `NamedVectorManager`
    /// without a graph simply leave this as `None`.
    graph_out_edges_db: RwLock<Option<Database<Bytes, Bytes>>>,
    /// Shared storage backend handle from the owning `HelixGraphStorage`
    /// (US-006 seam). Set via `attach_backend` during storage construction
    /// before any dense core is created, then cloned into every subsequently
    /// built `VectorCore`. Shared via `Arc` because `AnyBackend` is not
    /// `Clone` and the `Lsm` variant cannot be re-created from an env. Wrapped
    /// in `RwLock<Option<…>>` (like `graph_out_edges_db`) because the setter
    /// takes `&self`. Plumbing only in 6a — no KV access routes through it yet.
    backend: RwLock<Option<Arc<AnyBackend>>>,
    /// In-memory per-segment tombstone counts, maintained on record / clear /
    /// reconstruct. The merge scheduler and debt scorer read these instead of
    /// scanning LMDB, so neither the txn-free debt path nor the hot merge-
    /// candidate selection pays a `prefix_iter`. Rebuilt from LMDB on
    /// `reconstruct_tombstones` (i.e. on open), so it is always consistent with
    /// the durable set after a restart. Best-effort: a poisoned lock degrades to
    /// "0 tombstones" (no merge pressure), never incorrect search.
    tombstone_counts: RwLock<HashMap<String, usize>>,
}

/// Process-wide feature gate for re-upsert tombstoning. Default OFF while the
/// production crash regression is isolated; set `HELIX_REUPSERT_TOMBSTONES=1`
/// to enable the reclamation path after staging validation.
#[cfg(not(test))]
fn reupsert_tombstones_enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        std::env::var("HELIX_REUPSERT_TOMBSTONES")
            .ok()
            // Match the collection-manager env-parse style: trim + lowercase
            // so " 1", "On", "TRUE" all enable consistently across modules.
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
fn reupsert_tombstones_enabled() -> bool {
    std::env::var("HELIX_REUPSERT_TOMBSTONES")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// Encode a tombstone key: `physical_name` ++ 0x00 ++ id(BE). The 0x00
/// separator is safe because physical segment names are ASCII
/// (`{base}__seg_{id:06}`) and never contain a NUL.
fn tombstone_key(physical_name: &str, id: u128) -> Vec<u8> {
    debug_assert!(
        !physical_name.as_bytes().contains(&0u8),
        "tombstone_key requires NUL-free segment names; got {physical_name:?}"
    );
    let mut key = Vec::with_capacity(physical_name.len() + 1 + 16);
    key.extend_from_slice(physical_name.as_bytes());
    key.push(0u8);
    key.extend_from_slice(&id.to_be_bytes());
    key
}

/// Prefix for iterating every tombstone of one segment.
fn tombstone_segment_prefix(physical_name: &str) -> Vec<u8> {
    let mut p = Vec::with_capacity(physical_name.len() + 1);
    p.extend_from_slice(physical_name.as_bytes());
    p.push(0u8);
    p
}

impl NamedVectorManager {
    /// Target maximum indexed segment fanout per dense vector space.
    ///
    /// Default preserves the current background optimizer ceiling. Operators
    /// can lower this gradually after watching merge CPU and p99 search
    /// latency, without requiring a binary change.
    pub fn max_indexed_segments_target() -> usize {
        static TARGET: LazyLock<usize> = LazyLock::new(|| {
            std::env::var("HELIX_DENSE_MAX_INDEXED_SEGMENTS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(16)
        });
        *TARGET
    }

    fn search_segment_fanout_target() -> usize {
        // Indexed target + one mutable tail + one building segment during
        // split-phase optimization before the next metrics scrape observes it.
        Self::max_indexed_segments_target().saturating_add(2)
    }

    fn dense_parallel_search_enabled() -> bool {
        static ENABLED: LazyLock<bool> = LazyLock::new(|| {
            std::env::var("HELIX_DENSE_PARALLEL_SEARCH")
                .ok()
                .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
                .unwrap_or(false)
        });
        *ENABLED
    }

    fn lsm_dense_parallel_search_enabled() -> bool {
        static ENABLED: LazyLock<bool> = LazyLock::new(|| {
            std::env::var("HELIX_LSM_DENSE_PARALLEL_SEARCH")
                .ok()
                .or_else(|| std::env::var("HELIX_DENSE_PARALLEL_SEARCH").ok())
                .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "off"))
                .unwrap_or(true)
        });
        *ENABLED
    }

    /// Graph-entangled HNSW master switch. When off, the graph bridge on the
    /// search path and graph seeding on the insert path are both no-ops; only
    /// metric cardinality (the `entangled="false"` label) is affected.
    ///
    /// Checked once at process start via `LazyLock` so it can't drift between
    /// queries inside one run.
    pub fn graph_entangled_enabled() -> bool {
        static ENABLED: LazyLock<bool> = LazyLock::new(|| {
            std::env::var("HELIX_GRAPH_ENTANGLED_HNSW")
                .ok()
                .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
                .unwrap_or(false)
        });
        *ENABLED
    }

    /// Per-invocation cap on how many visited HNSW nodes we consult for graph
    /// out-edges when the frontier dead-ends. Keeps the bridge's worst-case
    /// LMDB read budget bounded.
    pub fn graph_bridge_max_seeds() -> usize {
        static V: LazyLock<usize> = LazyLock::new(|| {
            std::env::var("HELIX_GRAPH_BRIDGE_MAX_SEEDS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(32)
        });
        *V
    }

    /// Per-invocation cap on total graph neighbors added as new candidates.
    pub fn graph_bridge_max_expansions() -> usize {
        static V: LazyLock<usize> = LazyLock::new(|| {
            std::env::var("HELIX_GRAPH_BRIDGE_MAX_EXPANSIONS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(64)
        });
        *V
    }

    /// Phase 4 master switch: graph-affinity reordering at merge time.
    ///
    /// When enabled, `prepare_merge` reorders the exported vector set via
    /// BFS over graph out-edges before building HNSW and the mmap sidecar.
    /// Graph-adjacent vectors then share disk pages, so HNSW traversal on
    /// the merged segment incurs fewer page faults. Separate from the
    /// search/insert flag because merge reordering is permanent — changes
    /// the on-disk layout — whereas search bridging is runtime-only.
    pub fn graph_affinity_merge_enabled() -> bool {
        static ENABLED: LazyLock<bool> = LazyLock::new(|| {
            std::env::var("HELIX_GRAPH_AFFINITY_MERGE")
                .ok()
                .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
                .unwrap_or(false)
        });
        *ENABLED
    }

    /// Phase 5: reorder segments so ones likely to contain graph neighbors
    /// of the query's top early candidate are probed first.
    ///
    /// Does a single cheap probe on the first segment (k=1), looks up graph
    /// out-edges of that top-1 id, then partitions remaining segments into
    /// "owns at least one graph-neighbor id" (go first) and "doesn't" (tail).
    /// Tail segments still get searched, but serial early-termination on
    /// `worst_distance` kicks in faster because priority segments have
    /// already populated the result heap with good distances.
    ///
    /// No-op path when the first segment produces no result or the top
    /// candidate has no graph out-edges.
    #[allow(clippy::too_many_arguments)]
    fn graph_directed_segment_order<F>(
        &self,
        txn: &RoTxn,
        cores: &HashMap<String, VectorCore>,
        _name: &str,
        query: &[f32],
        filter: Option<&[F]>,
        should_trickle: bool,
        selectivity_hint: Option<f32>,
        ef_override: Option<usize>,
        segments: Vec<(String, DenseVectorSegmentRole)>,
    ) -> Result<(Vec<(String, DenseVectorSegmentRole)>, u64), VectorError>
    where
        F: Fn(&HVector) -> bool + Send + Sync,
    {
        let edges_db = match self
            .graph_out_edges_db
            .read()
            .ok()
            .and_then(|slot| slot.as_ref().copied())
        {
            Some(db) => db,
            None => return Ok((segments, 0)),
        };
        if segments.len() < 2 {
            return Ok((segments, 0));
        }

        // Pre-probe the first segment with k=1 for the top candidate. Use
        // a fresh txn so the probe doesn't hold onto caller state longer
        // than necessary.
        let Some((probe_name, _)) = segments.first() else {
            return Ok((segments, 0));
        };
        let Some(probe_core) = cores.get(probe_name) else {
            return Ok((segments, 0));
        };
        let probe_result = {
            let rd = probe_core.backend.read_borrowed(txn);
            probe_core.search_with_selectivity_ef_observed(
                &rd,
                query,
                1,
                filter,
                should_trickle,
                selectivity_hint,
                ef_override,
                None,
            )?
        };
        let Some(top) = probe_result.first() else {
            return Ok((segments, 0));
        };
        let top_id = top.get_id();

        // Collect up to max_seeds graph neighbors of top_id.
        let max_seeds = Self::graph_bridge_max_seeds();
        let mut neighbors: HashSet<u128> = HashSet::with_capacity(max_seeds.min(32));
        let iter = match edges_db.prefix_iter(txn, &top_id.to_be_bytes()) {
            Ok(iter) => iter,
            Err(_) => return Ok((segments, 0)),
        };
        for item in iter {
            let Ok((_, value)) = item else { break };
            if value.len() < 32 {
                continue;
            }
            let Ok(bytes) = value[16..32].try_into() else {
                continue;
            };
            neighbors.insert(u128::from_be_bytes(bytes));
            if neighbors.len() >= max_seeds {
                break;
            }
        }
        if neighbors.is_empty() {
            return Ok((segments, 0));
        }

        // Partition: segments owning at least one neighbor id → priority;
        // others → tail. The probe segment stays first to keep the result
        // heap warm; we only reorder segments[1..].
        let mut priority: Vec<(String, DenseVectorSegmentRole)> = Vec::new();
        let mut tail: Vec<(String, DenseVectorSegmentRole)> = Vec::new();
        let mut hits: u64 = 0;
        for (seg_name, role) in segments.iter().skip(1) {
            let Some(core) = cores.get(seg_name) else {
                tail.push((seg_name.clone(), *role));
                continue;
            };
            let mut overlaps = false;
            {
                let rd = core.backend.read_borrowed(txn);
                for nid in &neighbors {
                    if core.contains_id(&rd, *nid).unwrap_or(false) {
                        overlaps = true;
                        hits += 1;
                        break;
                    }
                }
            }
            if overlaps {
                priority.push((seg_name.clone(), *role));
            } else {
                tail.push((seg_name.clone(), *role));
            }
        }

        let mut out: Vec<(String, DenseVectorSegmentRole)> = Vec::with_capacity(segments.len());
        out.push(segments[0].clone());
        out.extend(priority);
        out.extend(tail);
        Ok((out, hits))
    }

    /// Reorder exported vectors via BFS over graph out-edges so graph-adjacent
    /// vectors share disk pages in the merged segment's mmap sidecar.
    ///
    /// Algorithm: build a membership set of exported ids, then run BFS starting
    /// from the highest-degree node within that set (approximated by iteration
    /// order). Every visited id is emitted to the output in BFS order;
    /// unreachable ids are appended at the end to preserve completeness.
    ///
    /// When no graph DB is attached or the exported set has fewer than 2
    /// elements, returns the input unchanged with edges_followed=0.
    ///
    /// Returns `(reordered, edges_followed)` so the caller can emit the
    /// `helix_graph_affinity_edges_followed_total` counter.
    pub(crate) fn graph_affinity_reorder(
        &self,
        txn: &RoTxn,
        exported: Vec<(u128, Vec<f32>, HashMap<String, Value>)>,
    ) -> (Vec<(u128, Vec<f32>, HashMap<String, Value>)>, u64) {
        if exported.len() < 2 {
            return (exported, 0);
        }
        let edges_db = match self
            .graph_out_edges_db
            .read()
            .ok()
            .and_then(|slot| slot.as_ref().copied())
        {
            Some(db) => db,
            None => return (exported, 0),
        };

        // Index-by-id so BFS can pluck rows out without scanning.
        let mut remaining: HashMap<u128, (Vec<f32>, HashMap<String, Value>)> =
            HashMap::with_capacity(exported.len());
        let mut insertion_order: Vec<u128> = Vec::with_capacity(exported.len());
        for (id, data, fields) in exported {
            insertion_order.push(id);
            remaining.insert(id, (data, fields));
        }

        let mut reordered: Vec<(u128, Vec<f32>, HashMap<String, Value>)> =
            Vec::with_capacity(remaining.len());
        let mut edges_followed: u64 = 0;
        let mut queue: std::collections::VecDeque<u128> = std::collections::VecDeque::new();

        // Walk the full insertion order as BFS roots so disconnected
        // components all get covered without one arbitrary first node
        // dominating the layout.
        for seed in &insertion_order {
            if !remaining.contains_key(seed) {
                continue;
            }
            queue.push_back(*seed);
            while let Some(id) = queue.pop_front() {
                let Some((data, fields)) = remaining.remove(&id) else {
                    continue;
                };
                reordered.push((id, data, fields));

                let iter = match edges_db.prefix_iter(txn, &id.to_be_bytes()) {
                    Ok(iter) => iter,
                    Err(_) => continue,
                };
                for item in iter {
                    let Ok((_, value)) = item else {
                        continue;
                    };
                    edges_followed = edges_followed.saturating_add(1);
                    if value.len() < 32 {
                        continue;
                    }
                    let Ok(bytes) = value[16..32].try_into() else {
                        continue;
                    };
                    let neighbor_id = u128::from_be_bytes(bytes);
                    if remaining.contains_key(&neighbor_id) {
                        queue.push_back(neighbor_id);
                    }
                }
            }
        }

        // Any ids unreachable via graph edges (no out-edges or disconnected)
        // fall through to the tail in original insertion order.
        for id in insertion_order {
            if let Some((data, fields)) = remaining.remove(&id) {
                reordered.push((id, data, fields));
            }
        }

        (reordered, edges_followed)
    }

    /// Phase 5 master switch: graph-directed segment fanout ordering.
    ///
    /// When enabled, `dense_search_with_selectivity_ef` orders the serial
    /// per-segment walk by graph prior — segments likely to contain
    /// graph neighbors of the top early candidate are probed first, so
    /// `worst_distance`-based early termination kicks in sooner on the
    /// tail segments.
    pub fn graph_directed_fanout_enabled() -> bool {
        static ENABLED: LazyLock<bool> = LazyLock::new(|| {
            std::env::var("HELIX_GRAPH_DIRECTED_FANOUT")
                .ok()
                .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
                .unwrap_or(false)
        });
        *ENABLED
    }

    /// Hard cap on `Indexed + Building` dense segments per named vector.
    /// When the count is at or above this value, every Mutable→Building seal
    /// site refuses to transition: the active Mutable stays open, no new
    /// Building segment is created, and writes continue to land in the
    /// existing Mutable. Search latency on the affected vector may degrade
    /// (Mutable has no HNSW), but writes never fail. The merge optimizer
    /// drains existing Indexed segments back below the cap, at which point
    /// seals resume normally.
    ///
    /// This makes runaway segment counts (the 5404-segment incident) an
    /// architectural impossibility — independent of which write path the
    /// caller hits or whether the upsert breaker happens to fire.
    ///
    /// Default 32 = 4× the merge target (HELIX_DENSE_MAX_INDEXED_SEGMENTS=8).
    /// Industry calibration: upstream HelixDB has no segmentation (cap=1),
    /// Qdrant defaults `default_segment_number=5`, Milvus ~10-30, ES tiered
    /// merge ~20-40. 32 is comfortably inside the norm with headroom for
    /// in-flight Building segments and transient bursts.
    pub(crate) fn segment_creation_gate_cap() -> usize {
        static CAP: LazyLock<usize> = LazyLock::new(|| {
            std::env::var("HELIX_PER_COLLECTION_SEGMENT_CREATION_GATE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| n > 0)
                .unwrap_or(32)
        });
        *CAP
    }

    fn should_block_mutable_seal(
        &self,
        space: &DenseVectorSpaceMetadata,
        logical_name: &str,
    ) -> bool {
        let cap = Self::segment_creation_gate_cap();
        let active = space
            .segments
            .iter()
            .filter(|s| {
                matches!(
                    s.role,
                    DenseVectorSegmentRole::Indexed | DenseVectorSegmentRole::Building
                )
            })
            .count();
        if active < cap {
            return false;
        }
        metrics::counter!(
            "helix_segment_creation_gate_blocked_total",
            "collection" => self.collection_name.clone(),
            "vector" => logical_name.to_string(),
        )
        .increment(1);
        tracing::warn!(
            collection = %self.collection_name,
            vector = logical_name,
            active_indexed_plus_building = active,
            cap = cap,
            "Mutable->Building seal blocked by segment creation gate; writes continue accreting in active Mutable until merge optimizer drains backlog"
        );
        true
    }

    pub fn new(collection_name: String) -> Self {
        Self {
            collection_name,
            configs: RwLock::new(HashMap::new()),
            dense_spaces: RwLock::new(HashMap::new()),
            cores: RwLock::new(HashMap::new()),
            dirty_retired_segments: RwLock::new(HashSet::new()),
            dirty_retired_segment_hook: RwLock::new(None),
            sparse_configs: RwLock::new(HashMap::new()),
            sparse_cores: RwLock::new(HashMap::new()),
            indexing_threshold_override: AtomicUsize::new(INDEXING_THRESHOLD_USE_GLOBAL),
            graph_out_edges_db: RwLock::new(None),
            backend: RwLock::new(None),
            tombstone_counts: RwLock::new(HashMap::new()),
        }
    }

    /// Total in-memory tombstone count across all segments. O(segments), no
    /// LMDB. Used by the txn-free debt path.
    fn tombstone_total_cached(&self) -> usize {
        self.tombstone_counts
            .read()
            .map(|m| m.values().sum())
            .unwrap_or(0)
    }

    /// Cached per-segment tombstone count. O(1), no LMDB. Used by the merge-
    /// candidate scorer (H4: replaces the per-pass prefix scan).
    fn tombstone_count_for_segment_cached(&self, physical_name: &str) -> usize {
        self.tombstone_counts
            .read()
            .ok()
            .and_then(|m| m.get(physical_name).copied())
            .unwrap_or(0)
    }

    fn bump_tombstone_count(&self, physical_name: &str, delta: usize) {
        if delta == 0 {
            return;
        }
        if let Ok(mut m) = self.tombstone_counts.write() {
            *m.entry(physical_name.to_string()).or_insert(0) += delta;
        }
    }

    fn reset_tombstone_count(&self, physical_name: &str) {
        if let Ok(mut m) = self.tombstone_counts.write() {
            m.remove(physical_name);
        }
    }

    /// Wire up the owning storage's graph out-edges DB.
    ///
    /// Called by `HelixGraphStorage::new` after both the storage-level
    /// graph DBs and this manager have been constructed but before any
    /// dense cores are created. Subsequently-created cores will pick up
    /// this handle via `current_graph_out_edges_db()`; cores built before
    /// this call keep `None` and the graph bridge is a no-op for them.
    pub fn attach_graph_out_edges_db(&self, db: Database<Bytes, Bytes>) {
        if let Ok(mut slot) = self.graph_out_edges_db.write() {
            *slot = Some(db);
        }
    }

    /// Wire up the owning storage's shared backend handle (US-006 seam).
    ///
    /// Called by `HelixGraphStorage::new` before any dense core is created,
    /// so subsequently-built cores receive a clone of this `Arc` via
    /// `current_backend()`. Mirrors `attach_graph_out_edges_db`. Plumbing
    /// only in 6a — no KV access routes through the backend yet.
    pub fn attach_backend(&self, backend: Arc<AnyBackend>) {
        if let Ok(mut slot) = self.backend.write() {
            *slot = Some(backend);
        }
    }

    /// Current shared backend handle, if attached. Cloned into every core
    /// built after `attach_backend`. Errors if used before attach — the
    /// dense-core builders normally run after wiring, but a use-before-attach
    /// race must fail the one request, not panic the serving thread.
    fn current_backend(&self) -> Result<Arc<AnyBackend>, VectorError> {
        self.backend
            .read()
            .ok()
            .and_then(|slot| slot.clone())
            .ok_or_else(|| {
                VectorError::VectorCoreError(
                    "NamedVectorManager backend not attached before core creation".to_string(),
                )
            })
    }

    pub fn attach_dirty_retired_segment_hook<F>(&self, hook: F)
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        if let Ok(mut slot) = self.dirty_retired_segment_hook.write() {
            *slot = Some(Arc::new(hook));
        }
    }

    fn current_graph_out_edges_db(&self) -> Option<Database<Bytes, Bytes>> {
        self.graph_out_edges_db
            .read()
            .ok()
            .and_then(|slot| slot.as_ref().copied())
    }

    // ── Re-upsert tombstones ──────────────────────────────────────────
    //
    // A re-upsert of an existing `id` appends a fresh copy into the mutable
    // tail (`dense_append_to_mutable`). The older copies in sealed/Indexed
    // segments remain bytes-on-disk until their segment happens to merge.
    // These helpers record, per `(segment, id)`, that a copy is *superseded*
    // so `prepare_merge` drops it deterministically at the next merge —
    // regardless of which segment it lives in — and so tombstone-heavy
    // segments raise merge debt and become preferred candidates. The store is
    // additive (standalone LMDB DB) and fully reconstructable from live data.

    /// Whether the re-upsert tombstone path is active. Routes through the
    /// `HELIX_REUPSERT_TOMBSTONES` kill switch only: when off, every tombstone
    /// read/write helper short-circuits so the `Namespace::DenseTombstones`
    /// keyspace is never touched (`resolve_write` only creates it on first
    /// write, so reads before any write return `None`/empty through the seam —
    /// the lazy-create-on-first-write contract is preserved by the backend).
    fn tombstones_active(&self) -> bool {
        reupsert_tombstones_enabled()
    }

    /// True if `(physical_name, id)` is recorded as a superseded copy. Reads via
    /// the backend seam (`Namespace::DenseTombstones`); a never-written keyspace
    /// resolves as empty and returns `false`.
    fn is_tombstoned(
        &self,
        txn: &RoTxn,
        physical_name: &str,
        id: u128,
    ) -> Result<bool, VectorError> {
        self.current_backend()?
            .get_with_heed(
                txn,
                Namespace::DenseTombstones,
                &tombstone_key(physical_name, id),
                |v| v.is_some(),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    /// Record that every `OTHER` segment holding `id` now has a superseded
    /// copy (the winner is the freshly-appended copy in `winner_segment`).
    /// Writes via the backend seam — `put_raw` lazily creates the
    /// `dense_tombstones` keyspace on the first write.
    fn record_supersession(
        &self,
        txn: &mut RwTxn,
        owner_segments: &[String],
        winner_segment: &str,
        id: u128,
    ) -> Result<u64, VectorError> {
        let backend = self.current_backend()?;
        let mut recorded = 0u64;
        for seg in owner_segments {
            if seg == winner_segment {
                continue;
            }
            let key = tombstone_key(seg, id);
            let exists = backend
                .get_with_heed(txn, Namespace::DenseTombstones, &key, |v| v.is_some())
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            if !exists {
                backend
                    .put_heed(txn, Namespace::DenseTombstones, &key, &[])
                    .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
                recorded += 1;
                // Maintain the in-memory per-segment count so debt + merge
                // selection never scan LMDB.
                self.bump_tombstone_count(seg, 1);
            }
        }
        if recorded > 0 {
            metrics::counter!(
                "helix_reupsert_tombstones_recorded_total",
                "collection" => self.collection_name.clone(),
            )
            .increment(recorded);
        }
        Ok(recorded)
    }

    /// On a re-upsert append of `id` into `winner_segment`, probe every other
    /// segment of the same space for a surviving copy of `id` and tombstone it.
    /// `other_segments` is the caller's already-snapshotted list of the space's
    /// physical segment names; `cores` is the caller's already-held read guard,
    /// avoiding a second lock acquisition on the hot path. No-op (and no DB
    /// open) unless a stale copy is actually found, so steady-state ingest of
    /// brand-new ids never touches the tombstone DB.
    fn tombstone_superseded_on_append(
        &self,
        txn: &mut RwTxn,
        cores: &HashMap<String, VectorCore>,
        other_segments: &[String],
        winner_segment: &str,
        id: u128,
    ) -> Result<(), VectorError> {
        if !reupsert_tombstones_enabled() {
            return Ok(());
        }
        // Find which other segments actually own a copy of this id. A fresh id
        // (the overwhelmingly common steady-state case) owns nothing elsewhere,
        // so we never open the DB.
        let mut owners: Vec<String> = Vec::new();
        for seg in other_segments {
            if seg == winner_segment {
                continue;
            }
            let Some(core) = cores.get(seg) else {
                continue;
            };
            // contains_id only reads vectors_db; borrow a read view off the txn.
            let owns = {
                let rd = core.backend.read_borrowed(&*txn);
                core.contains_id(&rd, id)?
            };
            if owns {
                owners.push(seg.clone());
            }
        }
        if owners.is_empty() {
            return Ok(());
        }
        self.record_supersession(txn, &owners, winner_segment, id)?;
        Ok(())
    }

    /// Drop all tombstones owned by `physical_name`. Called after a segment's
    /// bytes are reclaimed (merge retire) so the set never grows unbounded.
    /// Routes through the backend seam: a prefix `scan_raw` collects the keys,
    /// then `delete_raw` removes each one (byte-identical to the prior
    /// `prefix_iter` + `delete` on `dense_tombstones`).
    fn clear_segment_tombstones(
        &self,
        txn: &mut RwTxn,
        physical_name: &str,
    ) -> Result<usize, VectorError> {
        let backend = self.current_backend()?;
        let prefix = tombstone_segment_prefix(physical_name);
        let mut keys: Vec<Vec<u8>> = Vec::new();
        backend
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
        for k in keys {
            backend
                .delete_heed(txn, Namespace::DenseTombstones, &k)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        }
        // Retired segment: drop its cached count entirely.
        self.reset_tombstone_count(physical_name);
        Ok(removed)
    }

    /// Aggregate tombstone count across all dense segments. Prefers the
    /// O(segments) in-memory cache (no LMDB scan); falls back to `db.len` only
    /// if the cache is empty but the durable DB is open (e.g. immediately after
    /// open, before reconstruct populates the cache).
    pub fn dense_tombstone_count(&self, txn: &RoTxn) -> usize {
        let cached = self.tombstone_total_cached();
        if cached > 0 {
            return cached;
        }
        if !self.tombstones_active() {
            return 0;
        }
        // Fall back to a full scan-count of the durable keyspace. A never-written
        // `dense_tombstones` keyspace resolves as empty through the seam, so this
        // returns 0 exactly like the prior `db.len` on an unopened DB.
        let Ok(backend) = self.current_backend() else {
            return 0;
        };
        let mut count = 0usize;
        let scanned = backend.scan_heed(
            txn,
            Namespace::DenseTombstones,
            KeyRange::all(),
            |_k, _v| {
                count += 1;
                true
            },
        );
        match scanned {
            Ok(()) => count,
            Err(_) => 0,
        }
    }

    /// Public read-path check for the production chunked-merge export
    /// (`run_chunked_merge_publish`). True if `(physical_name, id)` is a
    /// superseded copy that the merge should drop. Returns `Ok(false)` when the
    /// tombstone DB has never been opened (kill switch off / no re-upserts yet).
    pub fn is_tombstoned_for_merge(
        &self,
        txn: &RoTxn,
        physical_name: &str,
        id: u128,
    ) -> Result<bool, VectorError> {
        if !self.tombstones_active() {
            return Ok(false);
        }
        self.is_tombstoned(txn, physical_name, id)
    }

    pub fn is_tombstoned_for_merge_be(
        &self,
        r: &AnyRead<'_>,
        physical_name: &str,
        id: u128,
    ) -> Result<bool, VectorError> {
        if !self.tombstones_active() {
            return Ok(false);
        }
        self.current_backend()?
            .get_with(
                r,
                Namespace::DenseTombstones,
                &tombstone_key(physical_name, id),
                |v| v.is_some(),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    /// Public tombstone-clear for retired segments, called by the production
    /// retire paths (chunked-merge Phase E reaper finalize, segment breaker,
    /// collection delete). Idempotent; no-op if the DB is unopened. Opens the
    /// DB lazily under the caller's write txn so a retire that runs before any
    /// re-upsert still cleans up correctly.
    pub fn clear_tombstones_for_segment(
        &self,
        txn: &mut RwTxn,
        physical_name: &str,
    ) -> Result<usize, VectorError> {
        // Deferred-repair delete-tombstones (issue #29 "fix 2"): always
        // clear, independent of the re-upsert flag checked below — see
        // `clear_tombstones_for_segment_be`'s identical rationale. Reported
        // under its own metric, not folded into the re-upsert return value.
        if let Ok(cores) = self.cores.read() {
            if let Some(core) = cores.get(physical_name) {
                match core.clear_delete_tombstones(txn) {
                    Ok(n) if n > 0 => {
                        metrics::counter!(
                            "helix_lsm_delete_tombstones_cleared_on_retire_total",
                            "collection" => self.collection_name.clone(),
                        )
                        .increment(n as u64);
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::debug!(
                            error = %e,
                            segment = %physical_name,
                            "delete-tombstone clear on retire failed (non-fatal)"
                        );
                    }
                }
            }
        }

        if !self.tombstones_active() {
            return Ok(0);
        }
        self.clear_segment_tombstones(txn, physical_name)
    }

    pub fn clear_tombstones_for_segment_be(
        &self,
        r: &AnyRead<'_>,
        w: &mut AnyWrite<'_>,
        physical_name: &str,
    ) -> Result<usize, VectorError> {
        // Deferred-repair delete-tombstones (issue #29 "fix 2"): always clear
        // here too, independent of the re-upsert flag checked below — this is
        // the universal retire chokepoint (merge Phase E, segment breaker,
        // collection delete) for BOTH tombstone families, and a retiring
        // segment's delete-tombstones must be reclaimed whenever it runs.
        // Reported under its own metric, not folded into the re-upsert
        // return value below.
        if let Ok(cores) = self.cores.read() {
            if let Some(core) = cores.get(physical_name) {
                match core.clear_delete_tombstones_be(r, w) {
                    Ok(n) if n > 0 => {
                        metrics::counter!(
                            "helix_lsm_delete_tombstones_cleared_on_retire_total",
                            "collection" => self.collection_name.clone(),
                        )
                        .increment(n as u64);
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::debug!(
                            error = %e,
                            segment = %physical_name,
                            "delete-tombstone clear on retire failed (non-fatal)"
                        );
                    }
                }
            }
        }

        if !self.tombstones_active() {
            return Ok(0);
        }
        let backend = self.current_backend()?;
        let prefix = tombstone_segment_prefix(physical_name);
        let mut keys: Vec<Vec<u8>> = Vec::new();
        backend
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
        for k in keys {
            backend
                .delete(w, Namespace::DenseTombstones, &k)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        }
        self.reset_tombstone_count(physical_name);
        Ok(removed)
    }

    /// Rebuild the tombstone set from live data. For every id present in ≥2
    /// segments of a space, tombstone every copy except the one in the newest
    /// segment (the mutable tail, else the highest segment id). Deterministic
    /// and crash-safe: a torn/missing tombstone DB is fully repaired here, so
    /// the format itself never has to change to survive a crash. Returns the
    /// number of tombstones written.
    pub fn reconstruct_tombstones(&self, txn: &mut RwTxn) -> Result<usize, VectorError> {
        if !self.tombstones_active() {
            return Ok(0);
        }
        let backend = self.current_backend()?;

        let spaces_segments: Vec<(String, Vec<String>)> = {
            let dense_spaces = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            dense_spaces
                .iter()
                .map(|(logical, space)| (logical.clone(), segments_newest_first(logical, space)))
                .collect()
        };

        let mut written = 0usize;
        // Rebuild the in-memory per-segment count from the durable set as we go,
        // so it is consistent with LMDB after a restart. Start clean.
        let mut rebuilt: HashMap<String, usize> = HashMap::new();
        for (_logical, ordered) in spaces_segments {
            let mut winner: HashMap<u128, String> = HashMap::new();
            let cores = self
                .cores
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            for phys in &ordered {
                let Some(core) = cores.get(phys) else {
                    continue;
                };
                let ids = {
                    let rd = core.backend.read_borrowed(&*txn);
                    match core.level_zero_ids(&rd) {
                        Ok(ids) => ids,
                        Err(_) => continue,
                    }
                };
                for id in ids {
                    if winner.contains_key(&id) {
                        // A newer segment already owns this id: tombstone this
                        // (superseded) copy. Routed through the seam — `put_raw`
                        // lazily creates the keyspace on first write.
                        let key = tombstone_key(phys, id);
                        let exists = backend
                            .get_with_heed(txn, Namespace::DenseTombstones, &key, |v| v.is_some())
                            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
                        if !exists {
                            backend
                                .put_heed(txn, Namespace::DenseTombstones, &key, &[])
                                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
                            written += 1;
                        }
                        // Count every superseded copy (pre-existing or newly
                        // written) so the cache reflects the full durable set.
                        *rebuilt.entry(phys.clone()).or_insert(0) += 1;
                    } else {
                        winner.insert(id, phys.clone());
                    }
                }
            }
        }
        if let Ok(mut counts) = self.tombstone_counts.write() {
            *counts = rebuilt;
        }
        if written > 0 {
            metrics::counter!(
                "helix_reupsert_tombstones_reconstructed_total",
                "collection" => self.collection_name.clone(),
            )
            .increment(written as u64);
        }
        Ok(written)
    }

    /// Rebuild every dense core's in-memory delete-tombstone set from the
    /// durable keyspace (`HELIX_LSM_DELETE_TOMBSTONES`, issue #29 "fix 2").
    /// Unlike `reconstruct_tombstones` (which infers re-upsert supersession
    /// from cross-segment analysis of live data), delete-tombstones are
    /// written durably by the delete path itself — each core's
    /// `reconstruct_delete_tombstones_be` just replays what is already there.
    /// Runs unconditionally, independent of the flag's current value, for the
    /// same reason `VectorCore::is_delete_tombstoned` is unconditional: a
    /// flag flip to OFF after deletes exist must not resurrect them. Called
    /// on collection open (both backends); reader replicas additionally call
    /// the per-core variant when `refresh_dense_view_from_metadata_lsm` opens
    /// a newly-discovered segment. Returns the total ids loaded.
    pub fn reconstruct_delete_tombstones(&self, txn: &mut RwTxn) -> Result<usize, VectorError> {
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let mut total = 0usize;
        for core in cores.values() {
            let r = core.backend.read_borrowed(&*txn);
            total += core.reconstruct_delete_tombstones_be(&r)?;
        }
        Ok(total)
    }

    fn mark_dirty_retired_segment(&self, physical_name: String) -> Result<bool, VectorError> {
        let inserted = {
            let mut dirty = self
                .dirty_retired_segments
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            dirty.insert(physical_name.clone())
        };

        metrics::counter!(
            "helix_segment_dirty_retired_discovered_total",
            "collection" => self.collection_name.clone(),
            "outcome" => if inserted { "new" } else { "duplicate" },
        )
        .increment(1);

        if inserted {
            let hook = self
                .dirty_retired_segment_hook
                .read()
                .ok()
                .and_then(|slot| slot.clone());
            if let Some(hook) = hook {
                hook(physical_name);
            }
        }

        Ok(inserted)
    }

    /// Set the per-collection indexing threshold override.
    /// Pass `0` to enter bulk mode (defer all indexing).
    /// Pass `usize::MAX` to clear the override and use the global config.
    pub fn set_indexing_threshold(&self, threshold: usize) {
        self.indexing_threshold_override
            .store(threshold, AtomicOrdering::Relaxed);
    }

    /// Resolve the effective indexing threshold for this collection.
    /// Returns the per-collection override if set, otherwise the global fallback.
    pub fn effective_indexing_threshold(&self, global_threshold: usize) -> usize {
        let override_val = self
            .indexing_threshold_override
            .load(AtomicOrdering::Relaxed);
        if override_val == INDEXING_THRESHOLD_USE_GLOBAL {
            global_threshold
        } else {
            override_val
        }
    }

    #[cfg(test)]
    pub(crate) fn create_dense_core(
        env: &Env<WithTls>,
        txn: &mut RwTxn,
        physical_name: &str,
        config: &NamedVectorConfig,
        hnsw_config: HNSWConfig,
        backend: Arc<AnyBackend>,
    ) -> Result<VectorCore, VectorError> {
        Self::create_dense_core_with_graph(
            env,
            txn,
            physical_name,
            config,
            hnsw_config,
            None,
            backend,
        )
    }

    /// Instance-level core builder that automatically threads the attached
    /// graph out-edges DB (if any) into the new `VectorCore`. Prefer this
    /// over the static `create_dense_core` from within `NamedVectorManager`
    /// methods so graph-entangled HNSW sees the graph on cores created
    /// after `attach_graph_out_edges_db`.
    fn create_dense_core_attached(
        &self,
        env: &Env<WithTls>,
        txn: &mut RwTxn,
        physical_name: &str,
        config: &NamedVectorConfig,
        hnsw_config: HNSWConfig,
    ) -> Result<VectorCore, VectorError> {
        Self::create_dense_core_with_graph(
            env,
            txn,
            physical_name,
            config,
            hnsw_config,
            self.current_graph_out_edges_db(),
            self.current_backend()?,
        )
    }

    fn create_dense_core_attached_lsm(
        &self,
        data_dir: &Path,
        physical_name: &str,
        config: &NamedVectorConfig,
        hnsw_config: HNSWConfig,
    ) -> Result<VectorCore, VectorError> {
        VectorCore::new_named_lsm_with_dir(
            physical_name,
            hnsw_config,
            config.distance.clone(),
            config.spindle.clone(),
            Some(data_dir),
            config.size,
            self.current_backend()?,
        )
    }

    fn infer_legacy_space_be(
        &self,
        physical_name: &str,
        core: &VectorCore,
    ) -> DenseVectorSpaceMetadata {
        DenseVectorSpaceMetadata {
            next_segment_id: 1,
            segments: vec![DenseVectorSegmentMetadata {
                physical_name: physical_name.to_string(),
                role: match core.backend.begin_read() {
                    Ok(read) if core.has_index(&read).unwrap_or(false) => {
                        DenseVectorSegmentRole::Indexed
                    }
                    _ => DenseVectorSegmentRole::Mutable,
                },
            }],
        }
    }

    /// Like `create_dense_core` but threads the owning collection's graph
    /// out-edges DB handle into the VectorCore so graph-entangled HNSW
    /// (gated by `HELIX_GRAPH_ENTANGLED_HNSW=1`) can consult adjacency on
    /// the search/insert hot paths without opening a new LMDB env.
    ///
    /// Callers that don't have a graph attached (legacy single-core tests,
    /// bulk micro-benches) should use `create_dense_core` which passes
    /// `None`; the field is then `None` on the core and the graph bridge
    /// is a no-op regardless of the env flag.
    pub(crate) fn create_dense_core_with_graph(
        env: &Env<WithTls>,
        txn: &mut RwTxn,
        physical_name: &str,
        config: &NamedVectorConfig,
        hnsw_config: HNSWConfig,
        graph_out_edges_db: Option<Database<Bytes, Bytes>>,
        backend: Arc<AnyBackend>,
    ) -> Result<VectorCore, VectorError> {
        VectorCore::new_named_with_dir(
            env,
            txn,
            physical_name,
            hnsw_config,
            config.distance.clone(),
            config.spindle.clone(),
            Some(env.path()),
            config.size,
            graph_out_edges_db,
            backend,
        )
    }

    fn observe_segment_ids_high_water(&self, logical_name: &str, space: &DenseVectorSpaceMetadata) {
        metrics::gauge!(
            "helix_segment_ids_high_water",
            "collection" => self.collection_name.clone(),
            "vector" => logical_name.to_string(),
        )
        .set(space.next_segment_id as f64);
    }

    fn allocate_dense_core_locked(
        &self,
        env: &Env<WithTls>,
        txn: &mut RwTxn,
        logical_name: &str,
        config: &NamedVectorConfig,
        hnsw_config: HNSWConfig,
        space: &mut DenseVectorSpaceMetadata,
        cores: &HashMap<String, VectorCore>,
    ) -> Result<(String, VectorCore), VectorError> {
        loop {
            let dirty_names: Vec<String> = self
                .dirty_retired_segments
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?
                .iter()
                .cloned()
                .collect();
            let reserved_names: Vec<&str> = cores
                .keys()
                .map(String::as_str)
                .chain(dirty_names.iter().map(String::as_str))
                .collect();
            let reserved_ids = reserved_segment_ids(space, logical_name, reserved_names);

            for id in 0..space.next_segment_id {
                if reserved_ids.contains(&id) {
                    continue;
                }

                let physical_name = next_segment_name(logical_name, id);
                let core = self.create_dense_core_attached(
                    env,
                    txn,
                    &physical_name,
                    config,
                    hnsw_config.clone(),
                )?;
                let is_empty = {
                    let rd = core.backend.read_borrowed(&*txn);
                    core.is_empty(&rd)?
                };
                if is_empty {
                    self.observe_segment_ids_high_water(logical_name, space);
                    return Ok((physical_name, core));
                }

                metrics::counter!(
                    "helix_segment_dirty_reuse_skipped_total",
                    "collection" => self.collection_name.clone(),
                    "vector" => logical_name.to_string(),
                )
                .increment(1);
                let inserted = self.mark_dirty_retired_segment(physical_name.clone())?;
                if inserted {
                    tracing::info!(
                        collection = %self.collection_name,
                        vector = %logical_name,
                        segment = %physical_name,
                        "Discovered dirty retired segment ID during allocation; queued reaper before name reuse"
                    );
                } else {
                    tracing::debug!(
                        collection = %self.collection_name,
                        vector = %logical_name,
                        segment = %physical_name,
                        "Dirty retired segment ID still awaiting reaper before name reuse"
                    );
                }
                break;
            }

            let id = space.next_segment_id;
            space.next_segment_id += 1;
            let physical_name = next_segment_name(logical_name, id);
            let core = self.create_dense_core_attached(
                env,
                txn,
                &physical_name,
                config,
                hnsw_config.clone(),
            )?;
            let is_empty = {
                let rd = core.backend.read_borrowed(&*txn);
                core.is_empty(&rd)?
            };
            if is_empty {
                self.observe_segment_ids_high_water(logical_name, space);
                return Ok((physical_name, core));
            }

            metrics::counter!(
                "helix_segment_dirty_fresh_skipped_total",
                "collection" => self.collection_name.clone(),
                "vector" => logical_name.to_string(),
            )
            .increment(1);
            let inserted = self.mark_dirty_retired_segment(physical_name.clone())?;
            if inserted {
                tracing::info!(
                    collection = %self.collection_name,
                    vector = %logical_name,
                    segment = %physical_name,
                    "Fresh segment ID opened dirty LMDB databases; queued reaper before name reuse"
                );
            } else {
                tracing::debug!(
                    collection = %self.collection_name,
                    vector = %logical_name,
                    segment = %physical_name,
                    "Fresh dirty segment ID still awaiting reaper before name reuse"
                );
            }
        }
    }

    fn allocate_dense_core_locked_lsm(
        &self,
        data_dir: &Path,
        logical_name: &str,
        config: &NamedVectorConfig,
        hnsw_config: HNSWConfig,
        space: &mut DenseVectorSpaceMetadata,
        cores: &HashMap<String, VectorCore>,
    ) -> Result<(String, VectorCore), VectorError> {
        loop {
            let dirty_names: Vec<String> = self
                .dirty_retired_segments
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?
                .iter()
                .cloned()
                .collect();
            let reserved_names: Vec<&str> = cores
                .keys()
                .map(String::as_str)
                .chain(dirty_names.iter().map(String::as_str))
                .collect();
            let reserved_ids = reserved_segment_ids(space, logical_name, reserved_names);

            for id in 0..space.next_segment_id {
                if reserved_ids.contains(&id) {
                    continue;
                }
                let physical_name = next_segment_name(logical_name, id);
                let core = self.create_dense_core_attached_lsm(
                    data_dir,
                    &physical_name,
                    config,
                    hnsw_config.clone(),
                )?;
                let is_empty = {
                    let read = core
                        .backend
                        .begin_read()
                        .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
                    core.is_empty(&read)?
                };
                if is_empty {
                    self.observe_segment_ids_high_water(logical_name, space);
                    return Ok((physical_name, core));
                }
                let _ = self.mark_dirty_retired_segment(physical_name)?;
                break;
            }

            let id = space.next_segment_id;
            space.next_segment_id += 1;
            let physical_name = next_segment_name(logical_name, id);
            let core = self.create_dense_core_attached_lsm(
                data_dir,
                &physical_name,
                config,
                hnsw_config.clone(),
            )?;
            let is_empty = {
                let read = core
                    .backend
                    .begin_read()
                    .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
                core.is_empty(&read)?
            };
            if is_empty {
                self.observe_segment_ids_high_water(logical_name, space);
                return Ok((physical_name, core));
            }
            let _ = self.mark_dirty_retired_segment(physical_name)?;
        }
    }

    fn infer_legacy_space(
        &self,
        txn: &RoTxn,
        physical_name: &str,
        core: &VectorCore,
    ) -> DenseVectorSpaceMetadata {
        DenseVectorSpaceMetadata {
            next_segment_id: 1,
            segments: vec![DenseVectorSegmentMetadata {
                physical_name: physical_name.to_string(),
                role: if {
                    let rd = core.backend.read_borrowed(txn);
                    core.has_index(&rd).unwrap_or(false)
                } {
                    DenseVectorSegmentRole::Indexed
                } else {
                    DenseVectorSegmentRole::Mutable
                },
            }],
        }
    }

    /// Create a logical named vector space with an initial mutable segment.
    pub fn create_vector_index(
        &self,
        env: &Env<WithTls>,
        txn: &mut RwTxn,
        name: &str,
        config: NamedVectorConfig,
        hnsw_config: HNSWConfig,
    ) -> Result<(), VectorError> {
        let core = self.create_dense_core_attached(env, txn, name, &config, hnsw_config)?;

        self.configs
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?
            .insert(name.to_string(), config);

        self.cores
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?
            .insert(name.to_string(), core);

        self.dense_spaces
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?
            .insert(
                name.to_string(),
                DenseVectorSpaceMetadata {
                    next_segment_id: 1,
                    segments: vec![DenseVectorSegmentMetadata {
                        physical_name: name.to_string(),
                        role: DenseVectorSegmentRole::Mutable,
                    }],
                },
            );

        Ok(())
    }

    pub fn create_vector_index_lsm(
        &self,
        data_dir: &Path,
        name: &str,
        config: NamedVectorConfig,
        hnsw_config: HNSWConfig,
    ) -> Result<(), VectorError> {
        self.load_vector_index_lsm(data_dir, name, config, hnsw_config, None)
            .map(|_| ())
    }

    /// Rehydrate a vector space from persisted metadata. If dense segment
    /// metadata is absent, fall back to the legacy single-core layout.
    pub fn load_vector_index(
        &self,
        env: &Env<WithTls>,
        txn: &mut RwTxn,
        name: &str,
        config: NamedVectorConfig,
        hnsw_config: HNSWConfig,
        segment_meta: Option<DenseVectorSpaceMetadata>,
    ) -> Result<bool, VectorError> {
        self.configs
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?
            .insert(name.to_string(), config.clone());

        let mut cores = self
            .cores
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

        let space = if let Some(space) = segment_meta.filter(|space| !space.segments.is_empty()) {
            for segment in &space.segments {
                let core = self.create_dense_core_attached(
                    env,
                    txn,
                    &segment.physical_name,
                    &config,
                    hnsw_config.clone(),
                )?;
                core.mark_externalized_marker_repair_needed();
                cores.insert(segment.physical_name.clone(), core);
            }
            space
        } else {
            let core = self.create_dense_core_attached(env, txn, name, &config, hnsw_config)?;
            let legacy_space = self.infer_legacy_space(txn, name, &core);
            core.mark_externalized_marker_repair_needed();
            cores.insert(name.to_string(), core);
            legacy_space
        };
        drop(cores);

        self.dense_spaces
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?
            .insert(name.to_string(), space);

        Ok(false)
    }

    /// Rehydrate a vector space on SlateDB without opening local heed DBIs.
    pub fn load_vector_index_lsm(
        &self,
        data_dir: &Path,
        name: &str,
        config: NamedVectorConfig,
        hnsw_config: HNSWConfig,
        segment_meta: Option<DenseVectorSpaceMetadata>,
    ) -> Result<bool, VectorError> {
        self.configs
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?
            .insert(name.to_string(), config.clone());

        let mut opened_cores = Vec::new();
        let space = if let Some(space) = segment_meta.filter(|space| !space.segments.is_empty()) {
            let segment_names: Vec<String> = space
                .segments
                .iter()
                .map(|segment| segment.physical_name.clone())
                .collect();
            let open_segment =
                |physical_name: &String| -> Result<(String, VectorCore), VectorError> {
                    let open_start = Instant::now();
                    let core = self.create_dense_core_attached_lsm(
                        data_dir,
                        physical_name,
                        &config,
                        hnsw_config.clone(),
                    )?;
                    metrics::histogram!(
                        "helix_lsm_dense_core_cold_open_ms",
                        "outcome" => "ok"
                    )
                    .record(open_start.elapsed().as_secs_f64() * 1000.0);
                    Ok((physical_name.clone(), core))
                };
            if segment_names.len() > 1 {
                let threads = segment_names.len().min(LSM_CORE_OPEN_MAX_CONCURRENCY);
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|i| format!("helix-lsm-core-open-{i}"))
                    .build()
                    .map_err(|e| {
                        VectorError::VectorCoreError(format!(
                            "failed to build LSM core-open pool: {}",
                            e
                        ))
                    })?;
                opened_cores = pool.install(|| {
                    use rayon::prelude::*;
                    segment_names
                        .par_iter()
                        .map(open_segment)
                        .collect::<Result<Vec<_>, _>>()
                })?;
            } else {
                for segment_name in &segment_names {
                    opened_cores.push(open_segment(segment_name)?);
                }
            }
            space
        } else {
            let core = self.create_dense_core_attached_lsm(data_dir, name, &config, hnsw_config)?;
            let legacy_space = self.infer_legacy_space_be(name, &core);
            opened_cores.push((name.to_string(), core));
            legacy_space
        };

        let mut cores = self
            .cores
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        for (physical_name, core) in opened_cores {
            core.mark_externalized_marker_repair_needed();
            cores.insert(physical_name, core);
        }
        drop(cores);

        self.dense_spaces
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?
            .insert(name.to_string(), space);

        Ok(false)
    }

    /// Reader-replica reconcile: bring the in-memory dense segment list and
    /// HNSW cores up to the view reflected in committed metadata, without
    /// mutating any committed KV state. Opens a [`VectorCore`] for every
    /// segment present in `dense_spaces_meta` but not yet in `cores`, then
    /// publishes the updated `dense_spaces` map.
    ///
    /// Lock order: `configs` write → `cores` write (held together) → released,
    /// then `dense_spaces` write. Cores are inserted BEFORE `dense_spaces` is
    /// swapped so a concurrent search never sees a listed segment without a
    /// backing core.
    ///
    /// `env`/`txn` are the **local** graph env — no committed KV state is
    /// written.
    pub fn refresh_dense_view_from_metadata(
        &self,
        env: &Env<WithTls>,
        txn: &mut RwTxn,
        named_vector_configs: &HashMap<String, NamedVectorConfig>,
        dense_spaces_meta: &HashMap<String, DenseVectorSpaceMetadata>,
        base_hnsw: HNSWConfig,
    ) -> Result<(), VectorError> {
        {
            let mut configs = self
                .configs
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let mut cores = self
                .cores
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            for (logical_name, space) in dense_spaces_meta {
                let Some(config) = named_vector_configs.get(logical_name) else {
                    continue;
                };
                configs
                    .entry(logical_name.clone())
                    .or_insert_with(|| config.clone());
                for seg in &space.segments {
                    if !cores.contains_key(&seg.physical_name) {
                        match self.create_dense_core_attached(
                            env,
                            txn,
                            &seg.physical_name,
                            config,
                            base_hnsw.clone(),
                        ) {
                            Ok(core) => {
                                core.mark_externalized_marker_repair_needed();
                                cores.insert(seg.physical_name.clone(), core);
                            }
                            Err(e) => {
                                tracing::warn!(
                                    collection = %self.collection_name,
                                    segment = %seg.physical_name,
                                    error = %e,
                                    "reader reconcile: failed to open core for segment (non-fatal)"
                                );
                            }
                        }
                    }
                }
            }
        } // drop configs + cores write locks before dense_spaces write
        *self
            .dense_spaces
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))? =
            dense_spaces_meta.clone();
        Ok(())
    }

    /// Reader-replica reconcile for SlateDB: same in-memory refresh as
    /// `refresh_dense_view_from_metadata`, but without opening local heed DBIs.
    pub fn refresh_dense_view_from_metadata_lsm(
        &self,
        data_dir: &Path,
        named_vector_configs: &HashMap<String, NamedVectorConfig>,
        dense_spaces_meta: &HashMap<String, DenseVectorSpaceMetadata>,
        base_hnsw: HNSWConfig,
    ) -> Result<(), VectorError> {
        let mut missing_segments = Vec::new();
        {
            let mut configs = self
                .configs
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let cores = self
                .cores
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            for (logical_name, space) in dense_spaces_meta {
                let Some(config) = named_vector_configs.get(logical_name) else {
                    continue;
                };
                configs
                    .entry(logical_name.clone())
                    .or_insert_with(|| config.clone());
                for seg in &space.segments {
                    if !cores.contains_key(&seg.physical_name) {
                        missing_segments.push((seg.physical_name.clone(), config.clone()));
                    }
                }
            }
        }
        let mut opened_cores = Vec::with_capacity(missing_segments.len());
        for (physical_name, config) in missing_segments {
            let open_start = Instant::now();
            match self.create_dense_core_attached_lsm(
                data_dir,
                &physical_name,
                &config,
                base_hnsw.clone(),
            ) {
                Ok(core) => {
                    metrics::histogram!(
                        "helix_lsm_dense_core_refresh_open_ms",
                        "outcome" => "ok"
                    )
                    .record(open_start.elapsed().as_secs_f64() * 1000.0);
                    core.mark_externalized_marker_repair_needed();
                    // Deferred-repair delete tombstones (issue #29 "fix 2"): a
                    // segment this reader is discovering for the first time
                    // may already carry delete-tombstones from a writer-side
                    // delete. Best-effort, scoped to newly-opened cores only
                    // (not a periodic re-scan of already-open ones) to avoid
                    // adding backend scan cost to every reader poll cycle — a
                    // known limitation: a delete on a segment this reader
                    // already has open is not picked up until the reader
                    // process restarts. See collection open's unconditional
                    // `reconstruct_delete_tombstones` for the initial-boot
                    // case this covers on top of.
                    if let Ok(backend) = self.current_backend() {
                        if let Ok(r) = backend.begin_read() {
                            if let Err(e) = core.reconstruct_delete_tombstones_be(&r) {
                                tracing::warn!(
                                    collection = %self.collection_name,
                                    segment = %physical_name,
                                    error = %e,
                                    "reader reconcile: delete-tombstone reconstruction failed (non-fatal)"
                                );
                            }
                        }
                    }
                    opened_cores.push((physical_name, core));
                }
                Err(e) => {
                    metrics::histogram!(
                        "helix_lsm_dense_core_refresh_open_ms",
                        "outcome" => "error"
                    )
                    .record(open_start.elapsed().as_secs_f64() * 1000.0);
                    tracing::warn!(
                        collection = %self.collection_name,
                        segment = %physical_name,
                        error = %e,
                        "reader reconcile: failed to open LSM core for segment (non-fatal)"
                    );
                }
            }
        }
        if !opened_cores.is_empty() {
            let mut cores = self
                .cores
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            for (physical_name, core) in opened_cores {
                cores.entry(physical_name).or_insert(core);
            }
        }
        *self
            .dense_spaces
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))? =
            dense_spaces_meta.clone();
        Ok(())
    }

    fn validate_dense_segment_publishable(
        &self,
        txn: &RoTxn,
        logical_name: &str,
        physical_name: &str,
    ) -> Result<(), VectorError> {
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let Some(core) = cores.get(physical_name) else {
            return Err(VectorError::VectorCoreError(format!(
                "Cannot publish dense segment '{}': core missing",
                physical_name
            )));
        };
        let unavailable = {
            let rd = core.backend.read_borrowed(txn);
            core.probe_unavailable_externalized_markers(&rd, 1)?
        };
        if unavailable == 0 {
            return Ok(());
        }

        metrics::counter!(
            "helix_dense_segment_publish_rejected_total",
            "collection" => self.collection_name.clone(),
            "vector" => logical_name.to_string()
        )
        .increment(1);
        Err(VectorError::VectorCoreError(format!(
            "Refusing to publish dense segment '{}' for '{}': externalized vector markers reference unavailable sidecar rows",
            physical_name, logical_name
        )))
    }

    fn validate_dense_segment_publishable_be(
        &self,
        r: &AnyRead<'_>,
        logical_name: &str,
        physical_name: &str,
    ) -> Result<(), VectorError> {
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let Some(core) = cores.get(physical_name) else {
            return Err(VectorError::VectorCoreError(format!(
                "Cannot publish dense segment '{}': core missing",
                physical_name
            )));
        };
        let unavailable = core.probe_unavailable_externalized_markers(r, 1)?;
        if unavailable == 0 {
            return Ok(());
        }

        metrics::counter!(
            "helix_dense_segment_publish_rejected_total",
            "collection" => self.collection_name.clone(),
            "vector" => logical_name.to_string()
        )
        .increment(1);
        Err(VectorError::VectorCoreError(format!(
            "Refusing to publish dense segment '{}' for '{}': externalized vector markers reference unavailable sidecar rows",
            physical_name, logical_name
        )))
    }

    fn resolve_core_name_locked(
        &self,
        name: &str,
        dense_spaces: &HashMap<String, DenseVectorSpaceMetadata>,
        cores: &HashMap<String, VectorCore>,
    ) -> Option<String> {
        if cores.contains_key(name) {
            return Some(name.to_string());
        }
        dense_spaces
            .get(name)
            .and_then(active_segment_name)
            .filter(|physical_name| cores.contains_key(physical_name))
    }

    fn ensure_mutable_dense_segment_locked(
        &self,
        env: &Env<WithTls>,
        txn: &mut RwTxn,
        logical_name: &str,
        config: &NamedVectorConfig,
        hnsw_config: HNSWConfig,
        space: &mut DenseVectorSpaceMetadata,
        cores: &mut HashMap<String, VectorCore>,
    ) -> Result<String, VectorError> {
        if let Some(active_name) = mutable_segment_name(space) {
            return Ok(active_name);
        }

        let (physical_name, core) = self.allocate_dense_core_locked(
            env,
            txn,
            logical_name,
            config,
            hnsw_config,
            space,
            cores,
        )?;
        cores.insert(physical_name.clone(), core);
        space.segments.push(DenseVectorSegmentMetadata {
            physical_name: physical_name.clone(),
            role: DenseVectorSegmentRole::Mutable,
        });
        Ok(physical_name)
    }

    fn seal_active_dense_segment_locked(
        &self,
        env: &Env<WithTls>,
        txn: &mut RwTxn,
        logical_name: &str,
        config: &NamedVectorConfig,
        hnsw_config: HNSWConfig,
        space: &mut DenseVectorSpaceMetadata,
        cores: &mut HashMap<String, VectorCore>,
    ) -> Result<Option<String>, VectorError> {
        let active_name = self.ensure_mutable_dense_segment_locked(
            env,
            txn,
            logical_name,
            config,
            hnsw_config.clone(),
            space,
            cores,
        )?;
        let active_core = cores.get(&active_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Segment core '{}' missing", active_name))
        })?;
        let active_count = {
            let rd = active_core.backend.read_borrowed(&*txn);
            active_core.level_zero_count(&rd)?
        };
        if active_count == 0 {
            return Ok(None);
        }

        // Segment creation gate: if Indexed + Building is already at the cap,
        // refuse to seal. The active Mutable stays open and continues to
        // accept writes; the merge optimizer drains existing Indexed below
        // the cap, at which point seals resume on the next call.
        if self.should_block_mutable_seal(space, logical_name) {
            return Ok(None);
        }

        for segment in &mut space.segments {
            if segment.physical_name == active_name {
                segment.role = DenseVectorSegmentRole::Building;
            }
        }

        let (next_physical_name, next_core) = self.allocate_dense_core_locked(
            env,
            txn,
            logical_name,
            config,
            hnsw_config,
            space,
            cores,
        )?;
        cores.insert(next_physical_name.clone(), next_core);
        space.segments.push(DenseVectorSegmentMetadata {
            physical_name: next_physical_name,
            role: DenseVectorSegmentRole::Mutable,
        });

        Ok(Some(active_name))
    }

    /// Get a reference to a named VectorCore. This is a compatibility helper
    /// for code paths that still expect a single core; it resolves to the active
    /// mutable segment when a logical space has multiple segments.
    pub fn get_core(&self, name: &str) -> Result<VectorCoreRef<'_>, VectorError> {
        let dense_spaces = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

        let resolved = self
            .resolve_core_name_locked(name, &dense_spaces, &cores)
            .ok_or_else(|| {
                VectorError::VectorNotFound(format!("Named vector '{}' not found", name))
            })?;

        Ok(VectorCoreRef {
            name: resolved,
            _manager: self,
        })
    }

    /// Execute a function with access to a single resolved VectorCore. Prefer
    /// the dense-space methods below for runtime search/insert/stat behavior.
    pub fn with_core<F, R>(&self, name: &str, f: F) -> Result<R, VectorError>
    where
        F: FnOnce(&VectorCore) -> Result<R, VectorError>,
    {
        let dense_spaces = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

        let resolved = self
            .resolve_core_name_locked(name, &dense_spaces, &cores)
            .ok_or_else(|| {
                VectorError::VectorCoreError(format!("Named vector '{}' not found", name))
            })?;
        let core = cores.get(&resolved).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", name))
        })?;

        f(core)
    }

    pub fn list_vectors(&self) -> HashMap<String, NamedVectorConfig> {
        self.configs.read().map(|c| c.clone()).unwrap_or_default()
    }

    /// Read-locked access to segment cores, for callers that need to
    /// interact with individual segments (e.g. split-phase merge export).
    pub fn cores_read(
        &self,
    ) -> Result<std::sync::RwLockReadGuard<'_, HashMap<String, VectorCore>>, VectorError> {
        self.cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))
    }

    pub fn try_core_names(&self) -> Result<Option<HashSet<String>>, VectorError> {
        match self.cores.try_read() {
            Ok(cores) => Ok(Some(cores.keys().cloned().collect())),
            Err(std::sync::TryLockError::WouldBlock) => Ok(None),
            Err(std::sync::TryLockError::Poisoned(e)) => Err(VectorError::VectorCoreError(
                format!("Lock poisoned: {}", e),
            )),
        }
    }

    pub fn list_dense_vector_spaces(&self) -> HashMap<String, DenseVectorSpaceMetadata> {
        self.dense_spaces
            .read()
            .map(|spaces| spaces.clone())
            .unwrap_or_default()
    }

    fn active_dense_segment_names(&self) -> Result<HashSet<String>, VectorError> {
        let spaces = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        Ok(spaces
            .values()
            .flat_map(|space| {
                space
                    .segments
                    .iter()
                    .map(|segment| segment.physical_name.clone())
            })
            .collect())
    }

    fn sidecar_path(dir: &Path, physical_name: &str, extension: &str) -> std::path::PathBuf {
        dir.join(format!("{physical_name}.{extension}"))
    }

    fn is_sidecar_path(path: &Path) -> bool {
        matches!(
            path.extension().and_then(|ext| ext.to_str()),
            Some("hvec" | "hvs8" | "hvtq")
        )
    }

    fn remove_sidecar_path(path: &Path, dry_run: bool, stats: &mut SidecarCleanupStats) {
        stats.candidate_files += 1;
        let metadata = match fs::metadata(path) {
            Ok(metadata) => metadata,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
            Err(e) => {
                stats.error_count += 1;
                stats
                    .errors
                    .push(format!("metadata {}: {}", path.display(), e));
                return;
            }
        };
        let bytes = metadata.len();
        if !dry_run {
            match fs::remove_file(path) {
                Ok(()) => {
                    metrics::counter!("helix_sidecar_files_unlinked_total").increment(1);
                    metrics::counter!("helix_sidecar_bytes_unlinked_total").increment(bytes);
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    stats.error_count += 1;
                    stats
                        .errors
                        .push(format!("remove {}: {}", path.display(), e));
                    return;
                }
            }
        }
        stats.removed_files += 1;
        stats.removed_bytes = stats.removed_bytes.saturating_add(bytes);
    }

    pub fn unlink_sidecar_files_for_segments(
        &self,
        dir: &Path,
        physical_names: &[String],
    ) -> SidecarCleanupStats {
        let started = Instant::now();
        let mut stats = SidecarCleanupStats::default();
        let active = match self.active_dense_segment_names() {
            Ok(active) => active,
            Err(e) => {
                stats.error_count += 1;
                stats.errors.push(e.to_string());
                finish_sidecar_cleanup_stats(&mut stats, started, "explicit");
                return stats;
            }
        };
        for physical_name in physical_names {
            if active.contains(physical_name) {
                stats.skipped_active_files += 3;
                continue;
            }
            Self::remove_sidecar_path(
                &Self::sidecar_path(dir, physical_name, "hvec"),
                false,
                &mut stats,
            );
            Self::remove_sidecar_path(
                &Self::sidecar_path(dir, physical_name, "hvs8"),
                false,
                &mut stats,
            );
            Self::remove_sidecar_path(
                &Self::sidecar_path(dir, physical_name, "hvtq"),
                false,
                &mut stats,
            );
        }
        finish_sidecar_cleanup_stats(&mut stats, started, "explicit");
        stats
    }

    pub fn cleanup_inactive_sidecar_files(&self, dir: &Path, dry_run: bool) -> SidecarCleanupStats {
        let started = Instant::now();
        let mut stats = SidecarCleanupStats {
            dry_run,
            ..SidecarCleanupStats::default()
        };
        let active = match self.active_dense_segment_names() {
            Ok(active) => active,
            Err(e) => {
                stats.error_count += 1;
                stats.errors.push(e.to_string());
                finish_sidecar_cleanup_stats(&mut stats, started, "scan");
                return stats;
            }
        };
        let entries = match fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(e) => {
                stats.error_count += 1;
                stats
                    .errors
                    .push(format!("read_dir {}: {}", dir.display(), e));
                finish_sidecar_cleanup_stats(&mut stats, started, "scan");
                return stats;
            }
        };
        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(e) => {
                    stats.error_count += 1;
                    stats.errors.push(format!("read_dir entry: {}", e));
                    continue;
                }
            };
            let path = entry.path();
            if !Self::is_sidecar_path(&path) {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            if active.contains(stem) {
                stats.skipped_active_files += 1;
                continue;
            }
            Self::remove_sidecar_path(&path, dry_run, &mut stats);
        }
        finish_sidecar_cleanup_stats(&mut stats, started, "scan");
        stats
    }

    /// Flush mmap sidecar files for all active dense segments.
    /// Call after a batch of inserts to ensure mmap reads reflect new data.
    pub fn flush_mmap_stores(&self) {
        if let Ok(cores) = self.cores.read() {
            for core in cores.values() {
                let _ = core.flush_mmap();
            }
        }
    }

    pub fn get_config(&self, name: &str) -> Option<NamedVectorConfig> {
        self.configs
            .read()
            .ok()
            .and_then(|configs| configs.get(name).cloned())
    }

    pub fn get_dense_space_stats(
        &self,
        txn: &RoTxn,
        name: &str,
    ) -> Result<DenseVectorStats, VectorError> {
        let dense_spaces = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let space = dense_spaces.get(name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", name))
        })?;
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

        let mut vectors_count = 0u64;
        let mut indexed_vectors_count = 0u64;
        let mut segments_count = 0usize;
        let mut missing_segments_count = 0usize;
        let mut missing_segments = Vec::new();
        for segment in &space.segments {
            let Some(core) = cores.get(&segment.physical_name) else {
                note_missing_segment(&self.collection_name, name, &segment.physical_name, "stats");
                missing_segments_count += 1;
                missing_segments.push(segment.physical_name.clone());
                continue;
            };
            let rd = core.backend.read_borrowed(txn);
            let count = core.level_zero_count(&rd)?;
            vectors_count += count;
            if core.has_index(&rd)? {
                indexed_vectors_count += count;
            }
            segments_count += 1;
        }

        Ok(DenseVectorStats {
            vectors_count,
            indexed_vectors_count,
            segments_count,
            missing_segments_count,
            missing_segments,
        })
    }

    pub fn get_dense_space_stats_be(
        &self,
        r: &AnyRead<'_>,
        name: &str,
    ) -> Result<DenseVectorStats, VectorError> {
        let dense_spaces = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let space = dense_spaces.get(name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", name))
        })?;
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

        let mut vectors_count = 0u64;
        let mut indexed_vectors_count = 0u64;
        let mut segments_count = 0usize;
        let mut missing_segments_count = 0usize;
        let mut missing_segments = Vec::new();
        for segment in &space.segments {
            let Some(core) = cores.get(&segment.physical_name) else {
                note_missing_segment(&self.collection_name, name, &segment.physical_name, "stats");
                missing_segments_count += 1;
                missing_segments.push(segment.physical_name.clone());
                continue;
            };
            let count = core.level_zero_count(r)?;
            vectors_count += count;
            if core.has_index(r)? {
                indexed_vectors_count += count;
            }
            segments_count += 1;
        }

        Ok(DenseVectorStats {
            vectors_count,
            indexed_vectors_count,
            segments_count,
            missing_segments_count,
            missing_segments,
        })
    }

    pub fn public_score(&self, name: &str, distance: f32) -> Result<f32, VectorError> {
        let config = self.get_config(name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", name))
        })?;
        Ok(match config.distance {
            DistanceMetric::Cosine => 1.0 - distance,
            DistanceMetric::Dot => -distance,
            DistanceMetric::Euclid => -distance,
        })
    }

    #[cfg(test)]
    pub fn dense_search_with_selectivity<F>(
        &self,
        env: &Env<WithTls>,
        txn: &RoTxn,
        name: &str,
        query: &[f32],
        k: usize,
        filter: Option<&[F]>,
        should_trickle: bool,
        selectivity_hint: Option<f32>,
    ) -> Result<Vec<HVector>, VectorError>
    where
        F: Fn(&HVector) -> bool + Send + Sync,
    {
        self.dense_search_with_selectivity_ef_with_provider(
            env,
            txn,
            name,
            query,
            k,
            filter,
            should_trickle,
            selectivity_hint,
            None,
        )
    }

    /// Like `dense_search_with_selectivity` but accepts an optional per-request ef override.
    ///
    /// `env` is required so per-segment workers can open their own thread-local
    /// read transactions; `txn` is used as the fallback for the single-segment
    /// fast path (where the rayon dispatch would be pure overhead).
    #[cfg(test)]
    pub fn dense_search_with_selectivity_ef<F>(
        &self,
        env: &Env<WithTls>,
        txn: &RoTxn,
        name: &str,
        query: &[f32],
        k: usize,
        filter: Option<&[F]>,
        should_trickle: bool,
        selectivity_hint: Option<f32>,
        ef_override: Option<usize>,
    ) -> Result<Vec<HVector>, VectorError>
    where
        F: Fn(&HVector) -> bool + Send + Sync,
    {
        self.dense_search_with_selectivity_ef_with_provider(
            env,
            txn,
            name,
            query,
            k,
            filter,
            should_trickle,
            selectivity_hint,
            ef_override,
        )
    }

    /// Dense search variant for production call sites that need nested worker
    /// transactions to participate in the storage resize fence.
    #[allow(clippy::too_many_arguments)]
    pub fn dense_search_with_selectivity_ef_with_provider<F, P>(
        &self,
        read_txns: &P,
        txn: &RoTxn,
        name: &str,
        query: &[f32],
        k: usize,
        filter: Option<&[F]>,
        should_trickle: bool,
        selectivity_hint: Option<f32>,
        ef_override: Option<usize>,
    ) -> Result<Vec<HVector>, VectorError>
    where
        F: Fn(&HVector) -> bool + Send + Sync,
        P: DenseReadTxnProvider,
    {
        let dense_spaces = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let space = dense_spaces.get(name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", name))
        })?;
        let all_segment_infos: Vec<(String, DenseVectorSegmentRole)> = space
            .segments
            .iter()
            .map(|segment| (segment.physical_name.clone(), segment.role))
            .collect();
        let building_count = all_segment_infos
            .iter()
            .filter(|(_, role)| *role == DenseVectorSegmentRole::Building)
            .count();
        let segment_infos: Vec<(String, DenseVectorSegmentRole)> = all_segment_infos
            .into_iter()
            .filter(|(_, role)| *role != DenseVectorSegmentRole::Building)
            .collect();
        // Recency rank per segment so duplicate ids (old sealed copy + fresh
        // re-upserted copy) resolve to the NEWEST copy at dedup, not the one
        // closest to the query.
        let recency_ranks = segment_recency_ranks(name, &segment_infos);
        let segment_count = segment_infos.len();
        let indexed_count = segment_infos
            .iter()
            .filter(|(_, role)| *role == DenseVectorSegmentRole::Indexed)
            .count();
        let mutable_count = segment_infos
            .iter()
            .filter(|(_, role)| *role == DenseVectorSegmentRole::Mutable)
            .count();
        drop(dense_spaces);

        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

        let observe_hot_metrics = crate::telemetry::hot_path_metrics_enabled();
        let search_start = observe_hot_metrics.then(std::time::Instant::now);
        let filtered = filter.is_some();
        let filtered_label = if filtered { "true" } else { "false" }.to_string();
        let segment_count_label = segment_count.to_string();
        let target_label = Self::search_segment_fanout_target().to_string();
        // Graph-entangled A/B label — same panel, two series.
        let entangled_label = if Self::graph_entangled_enabled() {
            "true"
        } else {
            "false"
        }
        .to_string();
        if observe_hot_metrics {
            metrics::gauge!(
                "helix_dense_search_segments",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "role" => "total".to_string(),
            )
            .set(segment_count as f64);
            metrics::gauge!(
                "helix_dense_search_segments",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "role" => "indexed".to_string(),
            )
            .set(indexed_count as f64);
            metrics::gauge!(
                "helix_dense_search_segments",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "role" => "building".to_string(),
            )
            .set(building_count as f64);
            metrics::gauge!(
                "helix_dense_search_segments",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "role" => "mutable".to_string(),
            )
            .set(mutable_count as f64);
            if segment_count > Self::search_segment_fanout_target() {
                metrics::counter!(
                    "helix_dense_search_segment_fanout_over_target_total",
                    "collection" => self.collection_name.clone(),
                    "vector" => name.to_string(),
                    "segment_count" => segment_count_label.clone(),
                    "target" => target_label,
                )
                .increment(1);
            }
        }

        // Phase 5: graph-directed segment fanout. When enabled and worth
        // it (segment count above threshold), probe the first segment for a
        // single top candidate, look up its graph out-edges, then reorder
        // remaining segments so ones owning those neighbor ids go first.
        // Segments without any graph-neighbor overlap fall through to the
        // tail; early termination in the serial loop then avoids fully
        // searching them when the worst-distance bound is already met.
        //
        // Gated behind HELIX_GRAPH_DIRECTED_FANOUT=1 because the pre-probe
        // costs one extra segment search — only worth it at high fanout.
        let segment_infos = if Self::graph_directed_fanout_enabled()
            && segment_count >= 4
            && !Self::dense_parallel_search_enabled()
        {
            let reorder_start = observe_hot_metrics.then(std::time::Instant::now);
            let (reordered, hits) = self.graph_directed_segment_order(
                txn,
                &cores,
                name,
                query,
                filter,
                should_trickle,
                selectivity_hint,
                ef_override,
                segment_infos,
            )?;
            if let Some(start) = reorder_start {
                metrics::histogram!(
                    "helix_graph_fanout_reorder_duration_ms",
                    "collection" => self.collection_name.clone(),
                    "vector" => name.to_string(),
                )
                .record(start.elapsed().as_secs_f64() * 1000.0);
                metrics::counter!(
                    "helix_graph_fanout_segment_hits_total",
                    "collection" => self.collection_name.clone(),
                    "vector" => name.to_string(),
                )
                .increment(hits);
            }
            reordered
        } else {
            segment_infos
        };

        // Per-segment search. Parallel fanout is gated because each worker
        // opens its own read txn over mmap-backed segment state; keep the
        // conservative serial path unless explicitly enabled.
        let mut combined: Vec<(u32, HVector)> = Vec::with_capacity(k * segment_count);
        if segment_count >= 2 && Self::dense_parallel_search_enabled() {
            use rayon::prelude::*;
            let cores_ref: &HashMap<String, VectorCore> = &*cores;
            let collected: Result<Vec<Vec<HVector>>, VectorError> = segment_infos
                .par_iter()
                .enumerate()
                .map(
                    |(seg_idx, (segment_name, role))| -> Result<Vec<HVector>, VectorError> {
                        let Some(core) = cores_ref.get(segment_name) else {
                            note_missing_segment(
                                &self.collection_name,
                                name,
                                segment_name,
                                "search",
                            );
                            return Ok(Vec::new());
                        };
                        let seg_start = observe_hot_metrics.then(std::time::Instant::now);
                        let metrics_labels = observe_hot_metrics.then(|| VectorSearchMetrics {
                            collection: &self.collection_name,
                            vector: name,
                            segment_count,
                            segment_index: seg_idx,
                            filtered,
                        });
                        let results = read_txns.with_dense_read_txn(|local_txn| {
                            let rd = core.backend.read_borrowed(local_txn);
                            core.search_with_selectivity_ef_observed(
                                &rd,
                                query,
                                k,
                                filter,
                                should_trickle,
                                selectivity_hint,
                                ef_override,
                                metrics_labels,
                            )
                        })?;
                        if let Some(seg_start) = seg_start {
                            let seg_elapsed = seg_start.elapsed();
                            metrics::histogram!(
                                "helix_dense_segment_search_duration_ms",
                                "collection" => self.collection_name.clone(),
                                "vector" => name.to_string(),
                                "segment_count" => segment_count_label.clone(),
                                "segment_index" => seg_idx.to_string(),
                                "role" => match role {
                                    DenseVectorSegmentRole::Mutable => "mutable",
                                    DenseVectorSegmentRole::Building => "building",
                                    DenseVectorSegmentRole::Indexed => "indexed",
                                }
                                .to_string(),
                                "filtered" => filtered_label.clone(),
                                "entangled" => entangled_label.clone(),
                            )
                            .record(seg_elapsed.as_secs_f64() * 1000.0);
                            metrics::histogram!(
                                "helix_dense_segment_search_results",
                                "collection" => self.collection_name.clone(),
                                "vector" => name.to_string(),
                                "segment_count" => segment_count_label.clone(),
                                "segment_index" => seg_idx.to_string(),
                                "filtered" => filtered_label.clone(),
                            )
                            .record(results.len() as f64);
                        }
                        Ok(results)
                    },
                )
                .collect();
            // `collected` preserves `segment_infos` order (par_iter + collect),
            // so tag each segment's results with that segment's recency rank.
            for ((segment_name, _), results) in segment_infos.iter().zip(collected?) {
                let rank = recency_ranks.get(segment_name).copied().unwrap_or(u32::MAX);
                combined.extend(results.into_iter().map(|hv| (rank, hv)));
            }
        } else {
            // Single-segment (or empty) — keep the serial path; rayon
            // dispatch overhead is pure cost when there's nothing to fan out.
            for (seg_idx, (segment_name, role)) in segment_infos.iter().enumerate() {
                let Some(core) = cores.get(segment_name) else {
                    note_missing_segment(&self.collection_name, name, segment_name, "search");
                    continue;
                };
                let seg_start = observe_hot_metrics.then(std::time::Instant::now);
                let metrics_labels = observe_hot_metrics.then(|| VectorSearchMetrics {
                    collection: &self.collection_name,
                    vector: name,
                    segment_count,
                    segment_index: seg_idx,
                    filtered,
                });
                let rd = core.backend.read_borrowed(txn);
                let results = core.search_with_selectivity_ef_observed(
                    &rd,
                    query,
                    k,
                    filter,
                    should_trickle,
                    selectivity_hint,
                    ef_override,
                    metrics_labels,
                )?;
                if let Some(seg_start) = seg_start {
                    let seg_elapsed = seg_start.elapsed();
                    metrics::histogram!(
                        "helix_dense_segment_search_duration_ms",
                        "collection" => self.collection_name.clone(),
                        "vector" => name.to_string(),
                        "segment_count" => segment_count_label.clone(),
                        "segment_index" => seg_idx.to_string(),
                        "role" => match role {
                            DenseVectorSegmentRole::Mutable => "mutable",
                            DenseVectorSegmentRole::Building => "building",
                            DenseVectorSegmentRole::Indexed => "indexed",
                        }
                        .to_string(),
                        "filtered" => filtered_label.clone(),
                        "entangled" => entangled_label.clone(),
                    )
                    .record(seg_elapsed.as_secs_f64() * 1000.0);
                    metrics::histogram!(
                        "helix_dense_segment_search_results",
                        "collection" => self.collection_name.clone(),
                        "vector" => name.to_string(),
                        "segment_count" => segment_count_label.clone(),
                        "segment_index" => seg_idx.to_string(),
                        "filtered" => filtered_label.clone(),
                    )
                    .record(results.len() as f64);
                }
                let rank = recency_ranks.get(segment_name).copied().unwrap_or(u32::MAX);
                combined.extend(results.into_iter().map(|hv| (rank, hv)));
            }
        }
        if let Some(search_start) = search_start {
            let total_elapsed = search_start.elapsed();
            metrics::histogram!(
                "helix_dense_search_duration_ms",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "segment_count" => segment_count_label.clone(),
                "filtered" => filtered_label.clone(),
                "entangled" => entangled_label.clone(),
            )
            .record(total_elapsed.as_secs_f64() * 1000.0);
            if total_elapsed.as_millis() > 5 {
                metrics::counter!(
                    "helix_dense_search_slow_total",
                    "collection" => self.collection_name.clone(),
                    "vector" => name.to_string(),
                    "segment_count" => segment_count_label.clone(),
                    "filtered" => filtered_label,
                )
                .increment(1);
            }
        }
        drop(cores);

        // Recency-aware dedup: keep the NEWEST copy of each id (lowest segment
        // rank), then order survivors by distance and truncate. This closes the
        // re-upsert staleness window — a query nearer an old sealed copy still
        // returns the freshly re-upserted vector.
        Ok(dedup_keep_newest(combined, k))
    }

    /// Dense search variant for Qdrant-compatible filters that only need the
    /// point id. This keeps the public `Fn(&HVector)` search contract intact
    /// while allowing Qdrant payload/index filters to run before vector
    /// materialization inside HNSW traversal.
    pub fn dense_search_with_id_filter_ef<F>(
        &self,
        txn: &RoTxn,
        name: &str,
        query: &[f32],
        k: usize,
        filter: Option<&[F]>,
        should_trickle: bool,
        selectivity_hint: Option<f32>,
        ef_override: Option<usize>,
    ) -> Result<Vec<HVector>, VectorError>
    where
        F: Fn(u128) -> bool,
    {
        let dense_spaces = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let space = dense_spaces.get(name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", name))
        })?;
        let all_segment_infos: Vec<(String, DenseVectorSegmentRole)> = space
            .segments
            .iter()
            .map(|segment| (segment.physical_name.clone(), segment.role))
            .collect();
        let building_count = all_segment_infos
            .iter()
            .filter(|(_, role)| *role == DenseVectorSegmentRole::Building)
            .count();
        let segment_infos: Vec<(String, DenseVectorSegmentRole)> = all_segment_infos
            .into_iter()
            .filter(|(_, role)| *role != DenseVectorSegmentRole::Building)
            .collect();
        // Recency rank per segment so duplicate ids (old sealed copy + fresh
        // re-upserted copy) resolve to the NEWEST copy at dedup, not the one
        // closest to the query.
        let recency_ranks = segment_recency_ranks(name, &segment_infos);
        let segment_count = segment_infos.len();
        let indexed_count = segment_infos
            .iter()
            .filter(|(_, role)| *role == DenseVectorSegmentRole::Indexed)
            .count();
        let mutable_count = segment_infos
            .iter()
            .filter(|(_, role)| *role == DenseVectorSegmentRole::Mutable)
            .count();
        drop(dense_spaces);

        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

        let observe_hot_metrics = crate::telemetry::hot_path_metrics_enabled();
        let search_start = observe_hot_metrics.then(std::time::Instant::now);
        let filtered = filter.is_some();
        let filtered_label = if filtered { "true" } else { "false" }.to_string();
        let segment_count_label = segment_count.to_string();
        let target_label = Self::search_segment_fanout_target().to_string();
        // Graph-entangled A/B label — same panel, two series.
        let entangled_label = if Self::graph_entangled_enabled() {
            "true"
        } else {
            "false"
        }
        .to_string();
        if observe_hot_metrics {
            metrics::gauge!(
                "helix_dense_search_segments",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "role" => "total".to_string(),
            )
            .set(segment_count as f64);
            metrics::gauge!(
                "helix_dense_search_segments",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "role" => "indexed".to_string(),
            )
            .set(indexed_count as f64);
            metrics::gauge!(
                "helix_dense_search_segments",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "role" => "building".to_string(),
            )
            .set(building_count as f64);
            metrics::gauge!(
                "helix_dense_search_segments",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "role" => "mutable".to_string(),
            )
            .set(mutable_count as f64);
            if segment_count > Self::search_segment_fanout_target() {
                metrics::counter!(
                    "helix_dense_search_segment_fanout_over_target_total",
                    "collection" => self.collection_name.clone(),
                    "vector" => name.to_string(),
                    "segment_count" => segment_count_label.clone(),
                    "target" => target_label,
                )
                .increment(1);
            }
        }

        let mut combined: Vec<(u32, HVector)> = Vec::with_capacity(k * segment_count);
        for (seg_idx, (segment_name, role)) in segment_infos.iter().enumerate() {
            let Some(core) = cores.get(segment_name) else {
                note_missing_segment(&self.collection_name, name, segment_name, "search");
                continue;
            };
            let seg_start = observe_hot_metrics.then(std::time::Instant::now);
            let metrics_labels = observe_hot_metrics.then(|| VectorSearchMetrics {
                collection: &self.collection_name,
                vector: name,
                segment_count,
                segment_index: seg_idx,
                filtered,
            });
            let rd = core.backend.read_borrowed(txn);
            let results = core.search_with_id_filter_ef_observed(
                &rd,
                query,
                k,
                filter,
                should_trickle,
                selectivity_hint,
                ef_override,
                metrics_labels,
            )?;
            if let Some(seg_start) = seg_start {
                let seg_elapsed = seg_start.elapsed();
                metrics::histogram!(
                    "helix_dense_segment_search_duration_ms",
                    "collection" => self.collection_name.clone(),
                    "vector" => name.to_string(),
                    "segment_count" => segment_count_label.clone(),
                    "segment_index" => seg_idx.to_string(),
                    "role" => match role {
                        DenseVectorSegmentRole::Mutable => "mutable",
                        DenseVectorSegmentRole::Building => "building",
                        DenseVectorSegmentRole::Indexed => "indexed",
                    }
                    .to_string(),
                    "filtered" => filtered_label.clone(),
                    "entangled" => entangled_label.clone(),
                )
                .record(seg_elapsed.as_secs_f64() * 1000.0);
                metrics::histogram!(
                    "helix_dense_segment_search_results",
                    "collection" => self.collection_name.clone(),
                    "vector" => name.to_string(),
                    "segment_count" => segment_count_label.clone(),
                    "segment_index" => seg_idx.to_string(),
                    "filtered" => filtered_label.clone(),
                )
                .record(results.len() as f64);
            }
            let rank = recency_ranks.get(segment_name).copied().unwrap_or(u32::MAX);
            combined.extend(results.into_iter().map(|hv| (rank, hv)));
        }

        if let Some(search_start) = search_start {
            let total_elapsed = search_start.elapsed();
            metrics::histogram!(
                "helix_dense_search_duration_ms",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "segment_count" => segment_count_label.clone(),
                "filtered" => filtered_label.clone(),
                "entangled" => entangled_label.clone(),
            )
            .record(total_elapsed.as_secs_f64() * 1000.0);
            if total_elapsed.as_millis() > 5 {
                metrics::counter!(
                    "helix_dense_search_slow_total",
                    "collection" => self.collection_name.clone(),
                    "vector" => name.to_string(),
                    "segment_count" => segment_count_label.clone(),
                    "filtered" => filtered_label,
                )
                .increment(1);
            }
        }
        drop(cores);

        // Recency-aware dedup: keep the NEWEST copy of each id (lowest segment
        // rank), then order survivors by distance and truncate. This closes the
        // re-upsert staleness window — a query nearer an old sealed copy still
        // returns the freshly re-upserted vector.
        Ok(dedup_keep_newest(combined, k))
    }

    pub fn dense_search_with_id_filter_ef_be<F>(
        &self,
        r: &AnyRead<'_>,
        name: &str,
        query: &[f32],
        k: usize,
        filter: Option<&[F]>,
        should_trickle: bool,
        selectivity_hint: Option<f32>,
        ef_override: Option<usize>,
    ) -> Result<Vec<HVector>, VectorError>
    where
        F: Fn(u128) -> bool,
    {
        let dense_spaces = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let space = dense_spaces.get(name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", name))
        })?;
        let all_segment_infos: Vec<(String, DenseVectorSegmentRole)> = space
            .segments
            .iter()
            .map(|segment| (segment.physical_name.clone(), segment.role))
            .collect();
        let building_count = all_segment_infos
            .iter()
            .filter(|(_, role)| *role == DenseVectorSegmentRole::Building)
            .count();
        let segment_infos: Vec<(String, DenseVectorSegmentRole)> = all_segment_infos
            .into_iter()
            .filter(|(_, role)| *role != DenseVectorSegmentRole::Building)
            .collect();
        let recency_ranks = segment_recency_ranks(name, &segment_infos);
        let segment_count = segment_infos.len();
        let indexed_count = segment_infos
            .iter()
            .filter(|(_, role)| *role == DenseVectorSegmentRole::Indexed)
            .count();
        let mutable_count = segment_infos
            .iter()
            .filter(|(_, role)| *role == DenseVectorSegmentRole::Mutable)
            .count();
        drop(dense_spaces);

        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

        let observe_hot_metrics = crate::telemetry::hot_path_metrics_enabled();
        let search_start = observe_hot_metrics.then(std::time::Instant::now);
        let filtered = filter.is_some();
        let filtered_label = if filtered { "true" } else { "false" }.to_string();
        let segment_count_label = segment_count.to_string();
        let target_label = Self::search_segment_fanout_target().to_string();
        let entangled_label = if Self::graph_entangled_enabled() {
            "true"
        } else {
            "false"
        }
        .to_string();
        if observe_hot_metrics {
            metrics::gauge!(
                "helix_dense_search_segments",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "role" => "total".to_string(),
            )
            .set(segment_count as f64);
            metrics::gauge!(
                "helix_dense_search_segments",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "role" => "indexed".to_string(),
            )
            .set(indexed_count as f64);
            metrics::gauge!(
                "helix_dense_search_segments",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "role" => "building".to_string(),
            )
            .set(building_count as f64);
            metrics::gauge!(
                "helix_dense_search_segments",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "role" => "mutable".to_string(),
            )
            .set(mutable_count as f64);
            if segment_count > Self::search_segment_fanout_target() {
                metrics::counter!(
                    "helix_dense_search_segment_fanout_over_target_total",
                    "collection" => self.collection_name.clone(),
                    "vector" => name.to_string(),
                    "segment_count" => segment_count_label.clone(),
                    "target" => target_label,
                )
                .increment(1);
            }
        }

        let mut combined: Vec<(u32, HVector)> = Vec::with_capacity(k * segment_count);
        let can_parallelize_read = matches!(r, AnyRead::Lsm(_) | AnyRead::LsmReader(_));
        if segment_count >= 2
            && Self::lsm_dense_parallel_search_enabled()
            && can_parallelize_read
            && filter.is_none()
        {
            use rayon::prelude::*;
            let cores_ref: &HashMap<String, VectorCore> = &*cores;
            let (lsm_read, reader_snap) = match r {
                AnyRead::Lsm(lsm_read) => (Some(lsm_read.clone()), None),
                // Share the outer read txn's snapshot (when pinned) so every
                // parallel segment worker observes the same committed state.
                AnyRead::LsmReader(snap) => (None, snap.clone()),
                AnyRead::Lmdb(_) | AnyRead::Failed(_) => {
                    unreachable!("parallel dense LSM search is gated to LSM read handles")
                }
            };
            let collected: Result<Vec<Vec<HVector>>, VectorError> = segment_infos
                .par_iter()
                .enumerate()
                .map(
                    |(seg_idx, (segment_name, role))| -> Result<Vec<HVector>, VectorError> {
                        let Some(core) = cores_ref.get(segment_name) else {
                            note_missing_segment(
                                &self.collection_name,
                                name,
                                segment_name,
                                "search",
                            );
                            return Ok(Vec::new());
                        };
                        let seg_start = observe_hot_metrics.then(std::time::Instant::now);
                        let metrics_labels = observe_hot_metrics.then(|| VectorSearchMetrics {
                            collection: &self.collection_name,
                            vector: name,
                            segment_count,
                            segment_index: seg_idx,
                            filtered,
                        });
                        let results = match &lsm_read {
                            Some(lsm_read) => {
                                let local_read = AnyRead::Lsm(lsm_read.clone());
                                core.search_with_id_filter_ef_observed(
                                    &local_read,
                                    query,
                                    k,
                                    None::<&[fn(u128) -> bool]>,
                                    should_trickle,
                                    selectivity_hint,
                                    ef_override,
                                    metrics_labels,
                                )
                            }
                            None => {
                                let local_read = AnyRead::LsmReader(reader_snap.clone());
                                core.search_with_id_filter_ef_observed(
                                    &local_read,
                                    query,
                                    k,
                                    None::<&[fn(u128) -> bool]>,
                                    should_trickle,
                                    selectivity_hint,
                                    ef_override,
                                    metrics_labels,
                                )
                            }
                        }?;
                        if let Some(seg_start) = seg_start {
                            let seg_elapsed = seg_start.elapsed();
                            metrics::histogram!(
                                "helix_dense_segment_search_duration_ms",
                                "collection" => self.collection_name.clone(),
                                "vector" => name.to_string(),
                                "segment_count" => segment_count_label.clone(),
                                "segment_index" => seg_idx.to_string(),
                                "role" => match role {
                                    DenseVectorSegmentRole::Mutable => "mutable",
                                    DenseVectorSegmentRole::Building => "building",
                                    DenseVectorSegmentRole::Indexed => "indexed",
                                }
                                .to_string(),
                                "filtered" => filtered_label.clone(),
                                "entangled" => entangled_label.clone(),
                            )
                            .record(seg_elapsed.as_secs_f64() * 1000.0);
                            metrics::histogram!(
                                "helix_dense_segment_search_results",
                                "collection" => self.collection_name.clone(),
                                "vector" => name.to_string(),
                                "segment_count" => segment_count_label.clone(),
                                "segment_index" => seg_idx.to_string(),
                                "filtered" => filtered_label.clone(),
                            )
                            .record(results.len() as f64);
                        }
                        Ok(results)
                    },
                )
                .collect();
            for ((segment_name, _), results) in segment_infos.iter().zip(collected?) {
                let rank = recency_ranks.get(segment_name).copied().unwrap_or(u32::MAX);
                combined.extend(results.into_iter().map(|hv| (rank, hv)));
            }
        } else {
            for (seg_idx, (segment_name, role)) in segment_infos.iter().enumerate() {
                let Some(core) = cores.get(segment_name) else {
                    note_missing_segment(&self.collection_name, name, segment_name, "search");
                    continue;
                };
                let seg_start = observe_hot_metrics.then(std::time::Instant::now);
                let metrics_labels = observe_hot_metrics.then(|| VectorSearchMetrics {
                    collection: &self.collection_name,
                    vector: name,
                    segment_count,
                    segment_index: seg_idx,
                    filtered,
                });
                let results = core.search_with_id_filter_ef_observed(
                    r,
                    query,
                    k,
                    filter,
                    should_trickle,
                    selectivity_hint,
                    ef_override,
                    metrics_labels,
                )?;
                if let Some(seg_start) = seg_start {
                    let seg_elapsed = seg_start.elapsed();
                    metrics::histogram!(
                        "helix_dense_segment_search_duration_ms",
                        "collection" => self.collection_name.clone(),
                        "vector" => name.to_string(),
                        "segment_count" => segment_count_label.clone(),
                        "segment_index" => seg_idx.to_string(),
                        "role" => match role {
                            DenseVectorSegmentRole::Mutable => "mutable",
                            DenseVectorSegmentRole::Building => "building",
                            DenseVectorSegmentRole::Indexed => "indexed",
                        }
                        .to_string(),
                        "filtered" => filtered_label.clone(),
                        "entangled" => entangled_label.clone(),
                    )
                    .record(seg_elapsed.as_secs_f64() * 1000.0);
                    metrics::histogram!(
                        "helix_dense_segment_search_results",
                        "collection" => self.collection_name.clone(),
                        "vector" => name.to_string(),
                        "segment_count" => segment_count_label.clone(),
                        "segment_index" => seg_idx.to_string(),
                        "filtered" => filtered_label.clone(),
                    )
                    .record(results.len() as f64);
                }
                let rank = recency_ranks.get(segment_name).copied().unwrap_or(u32::MAX);
                combined.extend(results.into_iter().map(|hv| (rank, hv)));
            }
        }

        if let Some(search_start) = search_start {
            let total_elapsed = search_start.elapsed();
            metrics::histogram!(
                "helix_dense_search_duration_ms",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "segment_count" => segment_count_label.clone(),
                "filtered" => filtered_label.clone(),
                "entangled" => entangled_label.clone(),
            )
            .record(total_elapsed.as_secs_f64() * 1000.0);
            if total_elapsed.as_millis() > 5 {
                metrics::counter!(
                    "helix_dense_search_slow_total",
                    "collection" => self.collection_name.clone(),
                    "vector" => name.to_string(),
                    "segment_count" => segment_count_label,
                    "filtered" => filtered_label,
                )
                .increment(1);
            }
        }
        drop(cores);

        Ok(dedup_keep_newest(combined, k))
    }

    pub fn dense_search_candidate_set_filter_ef_be(
        &self,
        r: &AnyRead<'_>,
        name: &str,
        query: &[f32],
        k: usize,
        candidate_ids: &HashSet<u128>,
        should_trickle: bool,
        selectivity_hint: Option<f32>,
        ef_override: Option<usize>,
    ) -> Result<Vec<HVector>, VectorError> {
        if k == 0 || candidate_ids.is_empty() {
            return Ok(Vec::new());
        }

        let dense_spaces = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let space = dense_spaces.get(name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", name))
        })?;
        let segment_infos: Vec<(String, DenseVectorSegmentRole)> = space
            .segments
            .iter()
            .filter(|segment| segment.role != DenseVectorSegmentRole::Building)
            .map(|segment| (segment.physical_name.clone(), segment.role))
            .collect();
        drop(dense_spaces);

        let can_parallelize_read = matches!(r, AnyRead::Lsm(_) | AnyRead::LsmReader(_));
        if segment_infos.len() < 2
            || !Self::lsm_dense_parallel_search_enabled()
            || !can_parallelize_read
        {
            let candidate_filter = |id: u128| candidate_ids.contains(&id);
            return self.dense_search_with_id_filter_ef_be(
                r,
                name,
                query,
                k,
                Some(&[candidate_filter][..]),
                should_trickle,
                selectivity_hint,
                ef_override,
            );
        }

        use rayon::prelude::*;

        let recency_ranks = segment_recency_ranks(name, &segment_infos);
        let segment_count = segment_infos.len();
        let segment_count_label = segment_count.to_string();
        let observe_hot_metrics = crate::telemetry::hot_path_metrics_enabled();
        let filtered_label = "true".to_string();
        let entangled_label = if Self::graph_entangled_enabled() {
            "true"
        } else {
            "false"
        }
        .to_string();
        let search_start = observe_hot_metrics.then(std::time::Instant::now);

        if observe_hot_metrics {
            metrics::histogram!(
                "helix_dense_candidate_filter_items",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "measurement" => "candidate_ids",
                "segment_count" => segment_count_label.clone(),
            )
            .record(candidate_ids.len() as f64);
        }

        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let cores_ref: &HashMap<String, VectorCore> = &cores;
        let (lsm_read, reader_snap) = match r {
            AnyRead::Lsm(lsm_read) => (Some(lsm_read.clone()), None),
            // Share the outer read txn's snapshot (when pinned) so every
            // parallel segment worker observes the same committed state.
            AnyRead::LsmReader(snap) => (None, snap.clone()),
            AnyRead::Lmdb(_) | AnyRead::Failed(_) => {
                unreachable!("parallel dense candidate filter is gated to LSM read handles")
            }
        };

        let collected: Result<Vec<Vec<HVector>>, VectorError> = segment_infos
            .par_iter()
            .enumerate()
            .map(
                |(seg_idx, (segment_name, role))| -> Result<Vec<HVector>, VectorError> {
                    let Some(core) = cores_ref.get(segment_name) else {
                        note_missing_segment(
                            &self.collection_name,
                            name,
                            segment_name,
                            "candidate_filter_search",
                        );
                        return Ok(Vec::new());
                    };
                    let candidate_filter = |id: u128| candidate_ids.contains(&id);
                    let filters = [candidate_filter];
                    let seg_start = observe_hot_metrics.then(std::time::Instant::now);
                    let metrics_labels = observe_hot_metrics.then(|| VectorSearchMetrics {
                        collection: &self.collection_name,
                        vector: name,
                        segment_count,
                        segment_index: seg_idx,
                        filtered: true,
                    });
                    let results = match &lsm_read {
                        Some(lsm_read) => {
                            let local_read = AnyRead::Lsm(lsm_read.clone());
                            core.search_with_id_filter_ef_observed(
                                &local_read,
                                query,
                                k,
                                Some(&filters[..]),
                                should_trickle,
                                selectivity_hint,
                                ef_override,
                                metrics_labels,
                            )
                        }
                        None => {
                            let local_read = AnyRead::LsmReader(reader_snap.clone());
                            core.search_with_id_filter_ef_observed(
                                &local_read,
                                query,
                                k,
                                Some(&filters[..]),
                                should_trickle,
                                selectivity_hint,
                                ef_override,
                                metrics_labels,
                            )
                        }
                    }?;
                    if let Some(seg_start) = seg_start {
                        let seg_elapsed = seg_start.elapsed();
                        metrics::histogram!(
                            "helix_dense_segment_search_duration_ms",
                            "collection" => self.collection_name.clone(),
                            "vector" => name.to_string(),
                            "segment_count" => segment_count_label.clone(),
                            "segment_index" => seg_idx.to_string(),
                            "role" => match role {
                                DenseVectorSegmentRole::Mutable => "mutable",
                                DenseVectorSegmentRole::Building => "building",
                                DenseVectorSegmentRole::Indexed => "indexed",
                            }
                            .to_string(),
                            "filtered" => filtered_label.clone(),
                            "entangled" => entangled_label.clone(),
                        )
                        .record(seg_elapsed.as_secs_f64() * 1000.0);
                        metrics::histogram!(
                            "helix_dense_segment_search_results",
                            "collection" => self.collection_name.clone(),
                            "vector" => name.to_string(),
                            "segment_count" => segment_count_label.clone(),
                            "segment_index" => seg_idx.to_string(),
                            "filtered" => filtered_label.clone(),
                        )
                        .record(results.len() as f64);
                    }
                    Ok(results)
                },
            )
            .collect();

        let mut combined: Vec<(u32, HVector)> = Vec::with_capacity(k * segment_count);
        for ((segment_name, _), results) in segment_infos.iter().zip(collected?) {
            let rank = recency_ranks.get(segment_name).copied().unwrap_or(u32::MAX);
            combined.extend(results.into_iter().map(|hv| (rank, hv)));
        }
        drop(cores);

        if let Some(search_start) = search_start {
            let total_elapsed = search_start.elapsed();
            metrics::histogram!(
                "helix_dense_search_duration_ms",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "segment_count" => segment_count_label.clone(),
                "filtered" => filtered_label.clone(),
                "entangled" => entangled_label,
            )
            .record(total_elapsed.as_secs_f64() * 1000.0);
            if total_elapsed.as_millis() > 5 {
                metrics::counter!(
                    "helix_dense_search_slow_total",
                    "collection" => self.collection_name.clone(),
                    "vector" => name.to_string(),
                    "segment_count" => segment_count_label,
                    "filtered" => filtered_label,
                )
                .increment(1);
            }
        }

        Ok(dedup_keep_newest(combined, k))
    }

    /// Exact-score a caller-supplied ID set across all readable dense segments.
    ///
    /// Used by Qdrant indexed filters when the payload-index candidate set is
    /// small enough that exact scoring is both cheaper and more reliable than
    /// asking HNSW to rediscover a selective filtered subset.
    pub fn dense_search_candidate_ids_exact<I>(
        &self,
        txn: &RoTxn,
        name: &str,
        query: &[f32],
        k: usize,
        candidate_ids: I,
        selectivity_hint: Option<f32>,
    ) -> Result<Vec<HVector>, VectorError>
    where
        I: IntoIterator<Item = u128>,
    {
        if k == 0 {
            return Ok(Vec::new());
        }

        let candidate_ids = unique_candidate_ids(candidate_ids.into_iter().collect());
        if candidate_ids.is_empty() {
            return Ok(Vec::new());
        }

        let dense_spaces = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let space = dense_spaces.get(name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", name))
        })?;
        let segment_infos: Vec<(String, DenseVectorSegmentRole)> = space
            .segments
            .iter()
            .filter(|segment| segment.role != DenseVectorSegmentRole::Building)
            .map(|segment| (segment.physical_name.clone(), segment.role))
            .collect();
        drop(dense_spaces);

        let recency_ranks = segment_recency_ranks(name, &segment_infos);
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let mut combined: Vec<(u32, HVector)> = Vec::with_capacity(k * segment_infos.len());

        for (segment_name, _) in dense_segments_by_recency(name, &segment_infos) {
            let Some(core) = cores.get(segment_name) else {
                note_missing_segment(&self.collection_name, name, segment_name, "exact_id_search");
                continue;
            };
            let rank = recency_ranks.get(segment_name).copied().unwrap_or(u32::MAX);
            let rd = core.backend.read_borrowed(txn);
            let segment_candidate_ids =
                exact_candidate_ids_present_in_segment(core, &rd, &candidate_ids)?;
            if segment_candidate_ids.is_empty() {
                continue;
            }
            let hits = core.search_exact_ids_with_selectivity(
                &rd,
                query,
                segment_candidate_ids.iter().copied(),
                k,
                selectivity_hint,
            )?;
            combined.extend(hits.into_iter().map(|hit| (rank, hit)));
        }

        Ok(dedup_keep_newest(combined, k))
    }

    pub fn dense_search_candidate_ids_exact_be<I>(
        &self,
        r: &AnyRead<'_>,
        name: &str,
        query: &[f32],
        k: usize,
        candidate_ids: I,
        selectivity_hint: Option<f32>,
    ) -> Result<Vec<HVector>, VectorError>
    where
        I: IntoIterator<Item = u128>,
    {
        if k == 0 {
            return Ok(Vec::new());
        }

        let candidate_ids = unique_candidate_ids(candidate_ids.into_iter().collect());
        if candidate_ids.is_empty() {
            return Ok(Vec::new());
        }

        let dense_spaces = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let space = dense_spaces.get(name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", name))
        })?;
        let segment_infos: Vec<(String, DenseVectorSegmentRole)> = space
            .segments
            .iter()
            .filter(|segment| segment.role != DenseVectorSegmentRole::Building)
            .map(|segment| (segment.physical_name.clone(), segment.role))
            .collect();
        drop(dense_spaces);

        let recency_ranks = segment_recency_ranks(name, &segment_infos);
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let mut combined: Vec<(u32, HVector)> = Vec::with_capacity(k * segment_infos.len());
        metrics::histogram!(
            "helix_named_vector_exact_search_items",
            "collection" => self.collection_name.clone(),
            "vector" => name.to_string(),
            "measurement" => "candidate_ids",
            "segment_index" => "all",
            "segment_count" => segment_infos.len().to_string(),
        )
        .record(candidate_ids.len() as f64);
        metrics::histogram!(
            "helix_named_vector_exact_search_items",
            "collection" => self.collection_name.clone(),
            "vector" => name.to_string(),
            "measurement" => "segment_count",
            "segment_index" => "all",
            "segment_count" => segment_infos.len().to_string(),
        )
        .record(segment_infos.len() as f64);

        for (segment_index, (segment_name, _)) in dense_segments_by_recency(name, &segment_infos)
            .into_iter()
            .enumerate()
        {
            let Some(core) = cores.get(segment_name) else {
                note_missing_segment(&self.collection_name, name, segment_name, "exact_id_search");
                continue;
            };
            let rank = recency_ranks.get(segment_name).copied().unwrap_or(u32::MAX);
            let candidate_start = Instant::now();
            let segment_candidate_ids =
                exact_candidate_ids_present_in_segment(core, r, &candidate_ids)?;
            metrics::histogram!(
                "helix_named_vector_exact_search_stage_ms",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "stage" => "segment_candidate_filter",
                "segment_index" => segment_index.to_string(),
                "segment_count" => segment_infos.len().to_string(),
            )
            .record(candidate_start.elapsed().as_secs_f64() * 1000.0);
            metrics::histogram!(
                "helix_named_vector_exact_search_items",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "measurement" => "segment_candidate_ids",
                "segment_index" => segment_index.to_string(),
                "segment_count" => segment_infos.len().to_string(),
            )
            .record(segment_candidate_ids.len() as f64);
            if segment_candidate_ids.is_empty() {
                continue;
            }
            let search_start = Instant::now();
            let hits = core.search_exact_ids_with_selectivity(
                r,
                query,
                segment_candidate_ids.iter().copied(),
                k,
                selectivity_hint,
            )?;
            metrics::histogram!(
                "helix_named_vector_exact_search_stage_ms",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "stage" => "segment_exact_score",
                "segment_index" => segment_index.to_string(),
                "segment_count" => segment_infos.len().to_string(),
            )
            .record(search_start.elapsed().as_secs_f64() * 1000.0);
            metrics::histogram!(
                "helix_named_vector_exact_search_items",
                "collection" => self.collection_name.clone(),
                "vector" => name.to_string(),
                "measurement" => "segment_hits",
                "segment_index" => segment_index.to_string(),
                "segment_count" => segment_infos.len().to_string(),
            )
            .record(hits.len() as f64);
            combined.extend(hits.into_iter().map(|hit| (rank, hit)));
        }

        let dedup_start = Instant::now();
        let results = dedup_keep_newest(combined, k);
        metrics::histogram!(
            "helix_named_vector_exact_search_stage_ms",
            "collection" => self.collection_name.clone(),
            "vector" => name.to_string(),
            "stage" => "dedup_keep_newest",
            "segment_index" => "all",
            "segment_count" => segment_infos.len().to_string(),
        )
        .record(dedup_start.elapsed().as_secs_f64() * 1000.0);
        metrics::histogram!(
            "helix_named_vector_exact_search_items",
            "collection" => self.collection_name.clone(),
            "vector" => name.to_string(),
            "measurement" => "results_returned",
            "segment_index" => "all",
            "segment_count" => segment_infos.len().to_string(),
        )
        .record(results.len() as f64);
        Ok(results)
    }

    pub fn dense_insert(
        &self,
        env: &Env<WithTls>,
        txn: &mut RwTxn,
        name: &str,
        data: &[f32],
        id: u128,
        fields: HashMap<String, Value>,
        hnsw_config: HNSWConfig,
        flat_scan_threshold: usize,
    ) -> Result<Option<String>, VectorError> {
        let config = self.get_config(name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", name))
        })?;

        let mut dense_spaces = self
            .dense_spaces
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let space = dense_spaces.get_mut(name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", name))
        })?;
        let mut cores = self
            .cores
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

        if flat_scan_threshold == 0 {
            let active_name = active_segment_name(space).ok_or_else(|| {
                VectorError::VectorCoreError(format!(
                    "Named vector '{}' has no active segment",
                    name
                ))
            })?;
            let core = cores.get(&active_name).ok_or_else(|| {
                VectorError::VectorCoreError(format!("Segment core '{}' missing", active_name))
            })?;
            core.insert_flat(txn, data, Some(id), Some(fields))?;
            return Ok(None);
        }

        let active_name = self.ensure_mutable_dense_segment_locked(
            env,
            txn,
            name,
            &config,
            hnsw_config.clone(),
            space,
            &mut cores,
        )?;

        let core = cores.get(&active_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Segment core '{}' missing", active_name))
        })?;
        core.insert_flat(txn, data, Some(id), Some(fields))?;
        let active_count = {
            let rd = core.backend.read_borrowed(&*txn);
            core.level_zero_count(&rd)? as usize
        };
        if active_count < flat_scan_threshold {
            return Ok(None);
        }

        self.seal_active_dense_segment_locked(
            env,
            txn,
            name,
            &config,
            hnsw_config,
            space,
            &mut cores,
        )
    }

    /// Append a dense vector to the active mutable segment without sealing it.
    ///
    /// The production upsert path uses this so hot writes only persist the
    /// flat tail under the write transaction. Threshold-based sealing and HNSW
    /// construction are owned by the background index executor.
    ///
    /// Returns true when a new mutable segment had to be created and dense
    /// segment metadata must be persisted by the caller.
    pub fn dense_append_to_mutable(
        &self,
        env: &Env<WithTls>,
        txn: &mut RwTxn,
        name: &str,
        data: &[f32],
        id: u128,
        fields: HashMap<String, Value>,
        hnsw_config: HNSWConfig,
        flat_scan_threshold: usize,
    ) -> Result<bool, VectorError> {
        // Fast path: a mutable segment already exists for this named
        // vector. The common case for re-upsert / steady-state ingest is
        // that we're appending to an existing segment with no layout
        // change. Acquire `dense_spaces` and `cores` as READ locks so
        // concurrent fast-path appends do not serialize on each other —
        // only on LMDB's per-env writer mutex (which they'd hit anyway).
        //
        // Phase 0 instrumentation showed dense_append_all_vectors p99
        // jumped to ~50 s under high concurrency once NOSYNC removed
        // fsync backpressure: the prior `dense_spaces.write()` +
        // `cores.write()` exclusive locks were the new bottleneck. The
        // slow path below still takes WRITE locks because creating a
        // mutable segment mutates `space.segments` and `cores`.
        if flat_scan_threshold > 0 {
            let dense_spaces_read = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            if let Some(space) = dense_spaces_read.get(name) {
                if let Some(active_name) = mutable_segment_name(space) {
                    // Snapshot the other (non-mutable) segment names so we can
                    // tombstone superseded copies after the append. Cheap clone
                    // of a handful of short strings; only non-empty once the
                    // space has sealed/Indexed segments (i.e. once re-upserts
                    // can actually shadow an older copy).
                    let other_segments: Vec<String> = space
                        .segments
                        .iter()
                        .filter(|s| s.physical_name != active_name)
                        .map(|s| s.physical_name.clone())
                        .collect();
                    let cores_read = self.cores.read().map_err(|e| {
                        VectorError::VectorCoreError(format!("Lock poisoned: {}", e))
                    })?;
                    if let Some(core) = cores_read.get(&active_name) {
                        core.insert_flat(txn, data, Some(id), Some(fields))?;
                        // Re-upsert tombstoning: if this id already lives in any
                        // sealed segment, mark those copies superseded so the
                        // next merge reclaims them deterministically. Search is
                        // unaffected (it already dedups via seen_ids, newest
                        // wins). No-op for brand-new ids.
                        if !other_segments.is_empty() {
                            self.tombstone_superseded_on_append(
                                txn,
                                &cores_read,
                                &other_segments,
                                &active_name,
                                id,
                            )?;
                        }
                        // No layout change — caller need not refresh
                        // dense-space metadata.
                        return Ok(false);
                    }
                }
            }
        }

        // Slow path: no mutable segment exists yet (or
        // flat_scan_threshold == 0 forces resolve via active_segment).
        // We may need to create a segment, which mutates `space.segments`
        // and `cores` — must hold write locks.
        let config = self.get_config(name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", name))
        })?;

        let mut dense_spaces = self
            .dense_spaces
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let space = dense_spaces.get_mut(name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", name))
        })?;
        let mut cores = self
            .cores
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

        let before_mutable = mutable_segment_name(space);
        let active_name = if flat_scan_threshold == 0 {
            active_segment_name(space).ok_or_else(|| {
                VectorError::VectorCoreError(format!(
                    "Named vector '{}' has no active segment",
                    name
                ))
            })?
        } else {
            self.ensure_mutable_dense_segment_locked(
                env,
                txn,
                name,
                &config,
                hnsw_config,
                space,
                &mut cores,
            )?
        };

        // Snapshot other segment names before the append (after any segment
        // creation above) so we can tombstone superseded copies.
        let other_segments: Vec<String> = space
            .segments
            .iter()
            .filter(|s| s.physical_name != active_name)
            .map(|s| s.physical_name.clone())
            .collect();

        let core = cores.get(&active_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Segment core '{}' missing", active_name))
        })?;
        core.insert_flat(txn, data, Some(id), Some(fields))?;

        // Re-upsert tombstoning (slow path). See fast-path comment above.
        if !other_segments.is_empty() {
            self.tombstone_superseded_on_append(txn, &cores, &other_segments, &active_name, id)?;
        }

        Ok(before_mutable.is_none() && mutable_segment_name(space).is_some())
    }

    /// LSM cutover variant of [`dense_append_to_mutable`].
    ///
    /// This supports the already-initialized mutable-tail case. Segment creation,
    /// sealing, and HNSW build still depend on the old heed transaction shape and
    /// must be ported before LMDB can be removed.
    pub fn dense_append_to_mutable_be(
        &self,
        w: &mut AnyWrite<'_>,
        name: &str,
        data: &[f32],
        id: u128,
        fields: HashMap<String, Value>,
    ) -> Result<bool, VectorError> {
        let dense_spaces = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let space = dense_spaces.get(name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", name))
        })?;
        let active_name = mutable_segment_name(space).ok_or_else(|| {
            VectorError::VectorCoreError(format!(
                "Named vector '{}' has no mutable segment initialized for LSM append",
                name
            ))
        })?;
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let core = cores.get(&active_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Segment core '{}' missing", active_name))
        })?;
        core.insert_flat_be(w, data, Some(id), Some(fields))?;
        Ok(false)
    }

    /// Return true when a dense upsert batch can append into already-open
    /// mutable segment DBs without changing dense-space layout metadata.
    ///
    /// Unlike `dense_batch_fits_existing_segments`, this deliberately ignores
    /// the flat-scan threshold. The background optimizer, not the hot write
    /// transaction, is responsible for threshold sealing and HNSW build.
    pub fn dense_batch_can_append_existing_segments(
        &self,
        batch_counts: &HashMap<String, usize>,
        flat_scan_threshold: usize,
    ) -> Result<bool, VectorError> {
        if batch_counts.is_empty() {
            return Ok(true);
        }

        let dense_spaces = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

        for name in batch_counts.keys() {
            let space = dense_spaces.get(name).ok_or_else(|| {
                VectorError::VectorCoreError(format!("Named vector '{}' not found", name))
            })?;
            let active_name = if flat_scan_threshold == 0 {
                active_segment_name(space)
            } else {
                mutable_segment_name(space)
            };
            let Some(active_name) = active_name else {
                return Ok(false);
            };
            if !cores.contains_key(&active_name) {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Return true when a dense upsert batch can write into already-open
    /// mutable segment DBs without publishing new segment handles or changing
    /// dense-space layout metadata.
    ///
    /// This lets ordinary point upserts use the shared resize gate. Callers
    /// must fall back to an exclusive write transaction when this returns
    /// false, because `dense_insert` may create a mutable segment or seal the
    /// active segment once the flat-scan threshold is reached.
    pub fn dense_batch_fits_existing_segments(
        &self,
        txn: &RoTxn,
        batch_counts: &HashMap<String, usize>,
        flat_scan_threshold: usize,
    ) -> Result<bool, VectorError> {
        if batch_counts.is_empty() {
            return Ok(true);
        }

        let dense_spaces = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

        for (name, incoming) in batch_counts {
            let space = dense_spaces.get(name).ok_or_else(|| {
                VectorError::VectorCoreError(format!("Named vector '{}' not found", name))
            })?;

            if flat_scan_threshold == 0 {
                let Some(active_name) = active_segment_name(space) else {
                    return Ok(false);
                };
                if !cores.contains_key(&active_name) {
                    return Ok(false);
                }
                continue;
            }

            let Some(mutable_name) = mutable_segment_name(space) else {
                return Ok(false);
            };
            let Some(core) = cores.get(&mutable_name) else {
                return Ok(false);
            };
            let active_count = {
                let rd = core.backend.read_borrowed(txn);
                core.level_zero_count(&rd)? as usize
            };
            if active_count.saturating_add(*incoming) >= flat_scan_threshold {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Seal the current mutable segment even if it has not crossed the normal
    /// threshold yet. This is used after a large bulk upsert so the trailing
    /// mutable tail can be indexed in the background instead of remaining on
    /// the flat path indefinitely.
    pub fn finalize_dense_tail(
        &self,
        env: &Env<WithTls>,
        txn: &mut RwTxn,
        name: &str,
        hnsw_config: HNSWConfig,
    ) -> Result<Option<String>, VectorError> {
        let config = self.get_config(name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", name))
        })?;
        let mut dense_spaces = self
            .dense_spaces
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let space = dense_spaces.get_mut(name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", name))
        })?;
        let mut cores = self
            .cores
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

        self.seal_active_dense_segment_locked(
            env,
            txn,
            name,
            &config,
            hnsw_config,
            space,
            &mut cores,
        )
    }

    /// Synchronous post-batch compaction: seal the mutable tail, build HNSW
    /// for un-indexed segments, then compact using a bounded merge policy.
    ///
    /// Unlike merge-to-one, this targets at most
    /// `HELIX_DENSE_MAX_INDEXED_SEGMENTS` indexed segments. Only small
    /// segments of similar size are merged (LSM-style tiered compaction).
    /// Large segments are left alone.
    ///
    /// This scales to 400K-1M vectors per collection without O(N²) rebuild
    /// cascades, while keeping search fanout bounded.
    pub fn finalize_and_compact<P>(
        &self,
        read_txns: &P,
        env: &Env<WithTls>,
        txn: &mut RwTxn,
        logical_name: &str,
        hnsw_config: HNSWConfig,
    ) -> Result<bool, VectorError>
    where
        P: DenseReadTxnProvider,
    {
        let config = self.get_config(logical_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", logical_name))
        })?;

        // Step 1: Seal the mutable tail (if non-empty).
        {
            let mut dense_spaces = self
                .dense_spaces
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let space = dense_spaces.get_mut(logical_name).ok_or_else(|| {
                VectorError::VectorCoreError(format!("Named vector '{}' not found", logical_name))
            })?;
            let mut cores = self
                .cores
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

            if let Some(mutable_name) = mutable_segment_name(space) {
                let count = cores
                    .get(&mutable_name)
                    .map(|c| {
                        let rd = c.backend.read_borrowed(&*txn);
                        c.level_zero_count(&rd).unwrap_or(0)
                    })
                    .unwrap_or(0);
                if count > 0 && !self.should_block_mutable_seal(space, logical_name) {
                    // Mark as Building and open a fresh mutable
                    for seg in &mut space.segments {
                        if seg.physical_name == mutable_name {
                            seg.role = DenseVectorSegmentRole::Building;
                        }
                    }
                    let (next_name, next_core) = self.allocate_dense_core_locked(
                        env,
                        txn,
                        logical_name,
                        &config,
                        hnsw_config.clone(),
                        space,
                        &cores,
                    )?;
                    cores.insert(next_name.clone(), next_core);
                    space.segments.push(DenseVectorSegmentMetadata {
                        physical_name: next_name,
                        role: DenseVectorSegmentRole::Mutable,
                    });
                }
            }
        }

        // Step 2: Build HNSW for every Building segment (synchronously).
        // Collect names first and drop the read locks before acquiring the
        // build permit. Holding `cores.read()` / `dense_spaces.read()` while
        // waiting on a saturated build semaphore would block every
        // write-side path (segment seal, new segment, merge).
        let to_build: HashSet<String> = {
            let dense_spaces = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let space = dense_spaces.get(logical_name).ok_or_else(|| {
                VectorError::VectorCoreError(format!("Named vector '{}' not found", logical_name))
            })?;
            space
                .segments
                .iter()
                .filter(|seg| seg.role == DenseVectorSegmentRole::Building)
                .map(|seg| seg.physical_name.clone())
                .collect()
        };
        for physical_name in &to_build {
            let permit = acquire_build_permit()?;
            let cores = self
                .cores
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            if let Some(core) = cores.get(physical_name) {
                core.build_index_from_flat_with_permit(txn, &permit)?;
            }
        }
        // Promote Building → Indexed
        {
            let mut dense_spaces = self
                .dense_spaces
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            if let Some(space) = dense_spaces.get_mut(logical_name) {
                for seg in &mut space.segments {
                    if seg.role == DenseVectorSegmentRole::Building
                        && to_build.contains(&seg.physical_name)
                    {
                        seg.role = DenseVectorSegmentRole::Indexed;
                    }
                }
            }
        }

        // Step 3: Bounded merge - only merge when above the configured target.
        // Each merge pass combines the 2 smallest segments (tiered policy).
        // Stop once we're at or below the target.
        loop {
            let merged = self.maybe_merge_dense_segments(
                read_txns,
                env,
                txn,
                logical_name,
                hnsw_config.clone(),
                Self::max_indexed_segments_target(),
            )?;
            if !merged {
                break;
            }
        }

        // Clean up any empty segments left over.
        self.cleanup_empty_dense_segments(txn, logical_name)?;

        Ok(true)
    }

    /// Collect the names of Building segments for a given logical space.
    pub fn building_segment_names(&self, logical_name: &str) -> Vec<String> {
        let dense_spaces = match self.dense_spaces.read() {
            Ok(ds) => ds,
            Err(_) => return Vec::new(),
        };
        let Some(space) = dense_spaces.get(logical_name) else {
            return Vec::new();
        };
        space
            .segments
            .iter()
            .filter(|seg| seg.role == DenseVectorSegmentRole::Building)
            .map(|seg| seg.physical_name.clone())
            .collect()
    }

    /// Prepare HNSW indices for Building segments using only a read
    /// transaction (no write lock needed). Returns a map from physical
    /// segment name to the prepared index data.
    pub fn prepare_building_indices(
        &self,
        txn: &RoTxn,
        logical_name: &str,
    ) -> Result<Vec<(String, super::vector_core::PreparedIndex)>, VectorError> {
        self.prepare_building_indices_limited(txn, logical_name, usize::MAX)
    }

    /// Variant used by the background optimizer to keep one job from
    /// monopolizing CPU/RSS by preparing every Building segment at once.
    pub fn prepare_building_indices_limited(
        &self,
        txn: &RoTxn,
        logical_name: &str,
        max_segments: usize,
    ) -> Result<Vec<(String, super::vector_core::PreparedIndex)>, VectorError> {
        let building_names = self
            .building_segment_names(logical_name)
            .into_iter()
            .take(max_segments)
            .collect::<Vec<_>>();
        let mut prepared = Vec::new();
        for seg_name in building_names {
            let build_permit = acquire_build_permit()?;
            let cores = self
                .cores
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            if let Some(core) = cores.get(&seg_name) {
                let idx = {
                    let rd = core.backend.read_borrowed(txn);
                    core.prepare_index_from_flat(&rd)?
                };
                if let Some(idx) = idx {
                    prepared.push((seg_name, idx));
                }
            }
            drop(cores);
            drop(build_permit);
        }
        Ok(prepared)
    }

    pub fn prepare_building_indices_limited_be(
        &self,
        r: &AnyRead<'_>,
        logical_name: &str,
        max_segments: usize,
    ) -> Result<Vec<(String, super::vector_core::PreparedIndex)>, VectorError> {
        let building_names = self
            .building_segment_names(logical_name)
            .into_iter()
            .take(max_segments)
            .collect::<Vec<_>>();
        let mut prepared = Vec::new();
        for seg_name in building_names {
            let build_permit = acquire_build_permit()?;
            let cores = self
                .cores
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            if let Some(core) = cores.get(&seg_name) {
                if let Some(idx) = core.prepare_index_from_flat(r)? {
                    prepared.push((seg_name, idx));
                }
            }
            drop(cores);
            drop(build_permit);
        }
        Ok(prepared)
    }

    /// Flush pre-computed HNSW indices into LMDB and promote the segments
    /// from Building to Indexed. This is the write-lock phase of the split
    /// build and should be fast (linear in vector count, no graph construction).
    pub fn flush_prepared_indices(
        &self,
        txn: &mut RwTxn,
        logical_name: &str,
        prepared: Vec<(String, super::vector_core::PreparedIndex)>,
    ) -> Result<bool, VectorError> {
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

        let mut flushed_names = HashSet::new();
        for (seg_name, idx) in prepared {
            if let Some(core) = cores.get(&seg_name) {
                core.flush_prepared_index(txn, idx)?;
                flushed_names.insert(seg_name);
            }
        }
        drop(cores);

        if !flushed_names.is_empty() {
            let mut dense_spaces = self
                .dense_spaces
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            if let Some(space) = dense_spaces.get_mut(logical_name) {
                for seg in &mut space.segments {
                    if seg.role == DenseVectorSegmentRole::Building
                        && flushed_names.contains(&seg.physical_name)
                    {
                        seg.role = DenseVectorSegmentRole::Indexed;
                    }
                }
            }
        }

        Ok(!flushed_names.is_empty())
    }

    pub fn flush_prepared_indices_be(
        &self,
        w: &mut AnyWrite<'_>,
        logical_name: &str,
        prepared: Vec<(String, super::vector_core::PreparedIndex)>,
    ) -> Result<bool, VectorError> {
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let mut flushed_names = HashSet::new();
        for (seg_name, mut idx) in prepared {
            if let Some(core) = cores.get(&seg_name) {
                let total = idx.point_ids.len();
                core.flush_prepared_index_chunk_be(w, &mut idx, 0, total)?;
                core.finalize_prepared_index_entry_be(w, &idx)?;
                flushed_names.insert(seg_name);
            }
        }
        drop(cores);
        if !flushed_names.is_empty() {
            let mut dense_spaces = self
                .dense_spaces
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            if let Some(space) = dense_spaces.get_mut(logical_name) {
                for seg in &mut space.segments {
                    if seg.role == DenseVectorSegmentRole::Building
                        && flushed_names.contains(&seg.physical_name)
                    {
                        seg.role = DenseVectorSegmentRole::Indexed;
                    }
                }
            }
        }
        Ok(!flushed_names.is_empty())
    }

    /// Split-phase compaction: seal tail in a write txn, build HNSW in
    /// memory without any transaction, flush results in a write txn.
    ///
    /// This dramatically reduces write-lock hold time compared to
    /// `finalize_and_compact` because the O(n log n) graph construction
    /// happens outside of any LMDB transaction.
    pub fn seal_mutable_tail(
        &self,
        env: &Env<WithTls>,
        txn: &mut RwTxn,
        logical_name: &str,
        hnsw_config: HNSWConfig,
    ) -> Result<bool, VectorError> {
        let config = self.get_config(logical_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", logical_name))
        })?;

        let mut dense_spaces = self
            .dense_spaces
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let space = dense_spaces.get_mut(logical_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", logical_name))
        })?;
        let mut cores = self
            .cores
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

        if let Some(mutable_name) = mutable_segment_name(space) {
            let has_vectors = cores
                .get(&mutable_name)
                .map(|c| {
                    let rd = c.backend.read_borrowed(&*txn);
                    c.level_zero_count(&rd).unwrap_or(0) > 0
                })
                .unwrap_or(false);
            if has_vectors && !self.should_block_mutable_seal(space, logical_name) {
                for seg in &mut space.segments {
                    if seg.physical_name == mutable_name {
                        seg.role = DenseVectorSegmentRole::Building;
                    }
                }
                let (next_name, next_core) = self.allocate_dense_core_locked(
                    env,
                    txn,
                    logical_name,
                    &config,
                    hnsw_config,
                    space,
                    &cores,
                )?;
                cores.insert(next_name.clone(), next_core);
                space.segments.push(DenseVectorSegmentMetadata {
                    physical_name: next_name,
                    role: DenseVectorSegmentRole::Mutable,
                });
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn seal_mutable_tail_lsm(
        &self,
        r: &AnyRead<'_>,
        data_dir: &Path,
        logical_name: &str,
        hnsw_config: HNSWConfig,
    ) -> Result<bool, VectorError> {
        let config = self.get_config(logical_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", logical_name))
        })?;

        let mut dense_spaces = self
            .dense_spaces
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let space = dense_spaces.get_mut(logical_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", logical_name))
        })?;
        let mut cores = self
            .cores
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

        if let Some(mutable_name) = mutable_segment_name(space) {
            let has_vectors = cores
                .get(&mutable_name)
                .and_then(|c| c.level_zero_count(r).ok().map(|count| count > 0))
                .unwrap_or(false);
            if has_vectors && !self.should_block_mutable_seal(space, logical_name) {
                for seg in &mut space.segments {
                    if seg.physical_name == mutable_name {
                        seg.role = DenseVectorSegmentRole::Building;
                    }
                }
                let (next_name, next_core) = self.allocate_dense_core_locked_lsm(
                    data_dir,
                    logical_name,
                    &config,
                    hnsw_config,
                    space,
                    &cores,
                )?;
                cores.insert(next_name.clone(), next_core);
                space.segments.push(DenseVectorSegmentMetadata {
                    physical_name: next_name,
                    role: DenseVectorSegmentRole::Mutable,
                });
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn build_dense_segment(
        &self,
        txn: &mut RwTxn,
        logical_name: &str,
        physical_name: &str,
    ) -> Result<(), VectorError> {
        {
            let build_permit = acquire_build_permit()?;
            let cores = self
                .cores
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let core = cores.get(physical_name).ok_or_else(|| {
                VectorError::VectorCoreError(format!("Segment core '{}' missing", physical_name))
            })?;
            // Experimental increment: only this build hook honors the env-gated
            // IVF mode. Seal and merge paths (`seal_mutable_tail`,
            // `prepare_merge`/`build_hnsw_in_memory`, `prepare_building_indices*`)
            // stay HNSW even in ivf mode — per-segment `index_mode` detection at
            // search time keeps such mixed collections correct.
            match dense_index_mode() {
                DenseIndexMode::Ivf => core.build_ivf_from_flat_with_permit(txn, &build_permit)?,
                DenseIndexMode::Hnsw => {
                    core.build_index_from_flat_with_permit(txn, &build_permit)?
                }
            }
        }

        let mut dense_spaces = self
            .dense_spaces
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        if let Some(space) = dense_spaces.get_mut(logical_name) {
            if let Some(segment) = space
                .segments
                .iter_mut()
                .find(|segment| segment.physical_name == physical_name)
            {
                segment.role = DenseVectorSegmentRole::Indexed;
            }
        }
        Ok(())
    }

    /// Tiered size-class boundaries. Segments in the same tier are merge
    /// candidates when the tier has more than `TIER_MAX_PER_CLASS` members.
    /// Tier boundaries: [0, 8k), [8k, 64k), [64k, 512k), [512k, ∞).
    const TIER_BOUNDARIES: &'static [usize] = &[8_192, 65_536, 524_288];
    /// Maximum segments allowed in a single size tier before merge.
    const TIER_MAX_PER_CLASS: usize = 4;

    pub(crate) fn merge_max_fan_in() -> usize {
        static FAN_IN: LazyLock<usize> = LazyLock::new(|| {
            std::env::var("HELIX_DENSE_MERGE_MAX_FANIN")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|value| *value >= 2)
                .unwrap_or(8)
        });
        *FAN_IN
    }

    /// Select merge candidates using tiered size-class compaction.
    ///
    /// Returns `None` if no merge is needed, or `Some(targets)` with the
    /// physical names of segments to merge (all from the same size tier).
    pub fn select_merge_candidates(
        &self,
        txn: &RoTxn,
        logical_name: &str,
        max_indexed_segments: usize,
    ) -> Result<Option<Vec<String>>, VectorError> {
        self.select_merge_candidates_with_limits(
            txn,
            logical_name,
            DenseMergeCandidateLimits::new(max_indexed_segments, Self::merge_max_fan_in()),
        )
    }

    pub(crate) fn select_merge_candidates_with_limits(
        &self,
        txn: &RoTxn,
        logical_name: &str,
        limits: DenseMergeCandidateLimits,
    ) -> Result<Option<Vec<String>>, VectorError> {
        let max_fan_in = limits.max_fan_in();
        let dense_spaces = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let space = dense_spaces.get(logical_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", logical_name))
        })?;
        if limits.uses_bounded_prefix() {
            let indexed_count = space
                .segments
                .iter()
                .filter(|segment| segment.role == DenseVectorSegmentRole::Indexed)
                .count();
            if indexed_count <= limits.max_indexed_segments {
                return Ok(None);
            }

            let cores = self
                .cores
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let mut indexed = Vec::with_capacity(limits.bounded_scan_len());
            let mut scanned = 0usize;
            for segment in &space.segments {
                if segment.role != DenseVectorSegmentRole::Indexed {
                    continue;
                }
                scanned += 1;
                if let Some(core) = cores.get(&segment.physical_name) {
                    let count = {
                        let rd = core.backend.read_borrowed(txn);
                        core.level_zero_count(&rd).unwrap_or(0) as usize
                    };
                    let tombstoned =
                        self.tombstone_count_for_segment_cached(&segment.physical_name);
                    let effective = count.saturating_sub(tombstoned);
                    indexed.push((segment.physical_name.clone(), effective));
                } else {
                    note_missing_segment(
                        &self.collection_name,
                        logical_name,
                        &segment.physical_name,
                        "bounded_merge_selection",
                    );
                }
                if scanned >= limits.bounded_scan_len() {
                    break;
                }
            }
            if indexed.len() < 2 {
                return Ok(None);
            }
            indexed.sort_by_key(|(_, count)| *count);
            let take = indexed.len().min(max_fan_in);
            return if take >= 2 {
                Ok(Some(
                    indexed[..take]
                        .iter()
                        .map(|(name, _)| name.clone())
                        .collect(),
                ))
            } else {
                Ok(None)
            };
        }
        // Merges operate only on Indexed segments (filtered below). A
        // concurrent Building segment in the same space does not share
        // physical state with an Indexed merge source: the build path
        // promotes Building -> Indexed under the same exclusive write txn
        // that publishes the merge target, so there is no race.
        // The previous "any Building" early-return starved continuously-
        // ingesting tenants whose tail is always Building (segments would
        // accumulate to hundreds/thousands while the optimizer rescheduled
        // forever without ever selecting a merge candidate).

        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

        // Tombstone-aware sizing: a segment's *effective* size for tier
        // bucketing and smallest-first selection is its live count minus the
        // copies a re-upsert has superseded, AND minus any deferred-repair
        // delete-tombstones (issue #29 "fix 2") — both leave rows physically
        // present until the next merge. A segment that is mostly stale
        // therefore sinks into a smaller tier and is merged (reclaimed) sooner,
        // which is exactly the debt the optimizer wants to pay down.
        // (H4) Read the per-segment tombstone count from the O(1) in-memory
        // cache — never a `prefix_iter` on this scheduling hot path.
        let mut indexed: Vec<(String, usize)> = Vec::new();
        let mut delete_ratio_breach = false;
        for seg in &space.segments {
            if seg.role != DenseVectorSegmentRole::Indexed {
                continue;
            }
            let Some(core) = cores.get(&seg.physical_name) else {
                note_missing_segment(
                    &self.collection_name,
                    logical_name,
                    &seg.physical_name,
                    "merge_selection",
                );
                continue;
            };
            let count = {
                let rd = core.backend.read_borrowed(txn);
                core.level_zero_count(&rd).unwrap_or(0) as usize
            };
            let tombstoned = self.tombstone_count_for_segment_cached(&seg.physical_name);
            let deleted = core.deleted_count();
            if count > 0 && (deleted as f64 / count as f64) > delete_vacuum_fraction() {
                delete_ratio_breach = true;
            }
            let effective = count.saturating_sub(tombstoned).saturating_sub(deleted);
            indexed.push((seg.physical_name.clone(), effective));
        }

        // A segment past the deleted-ratio ceiling schedules a merge even
        // when the space is below its normal segment-count trigger — the
        // fallback below picks it up (it always sorts to the smallest
        // effective size).
        if indexed.len() <= limits.max_indexed_segments
            && !(delete_ratio_breach && indexed.len() >= 2)
        {
            return Ok(None);
        }

        // Bucket segments into size tiers.
        let tier_of = |count: usize| -> usize {
            Self::TIER_BOUNDARIES
                .iter()
                .position(|&b| count < b)
                .unwrap_or(Self::TIER_BOUNDARIES.len())
        };

        // Find the lowest tier that has > TIER_MAX_PER_CLASS members.
        let mut tier_buckets: HashMap<usize, Vec<(String, usize)>> = HashMap::new();
        for (name, count) in &indexed {
            tier_buckets
                .entry(tier_of(*count))
                .or_default()
                .push((name.clone(), *count));
        }

        // Pick the tier with the most segments above the per-tier cap.
        // Within that tier, merge the smallest entries (up to TIER_MAX_PER_CLASS).
        let mut best_tier: Option<(usize, Vec<(String, usize)>)> = None;
        for (tier, mut members) in tier_buckets {
            if members.len() <= Self::TIER_MAX_PER_CLASS {
                continue;
            }
            members.sort_by_key(|(_, c)| *c);
            let overflow = members.len() - Self::TIER_MAX_PER_CLASS;
            // Merge the smallest `overflow` segments (at least 2).
            let take = overflow.max(2).min(max_fan_in).min(members.len());
            if best_tier
                .as_ref()
                .map_or(true, |(_, prev)| members.len() > prev.len())
            {
                best_tier = Some((tier, members[..take].to_vec()));
            }
        }

        // Fallback: if no single tier overflows but total exceeds cap, or a
        // segment breached the deleted-ratio ceiling, merge the 2 smallest
        // overall. The ratio-breach case is what actually turns the vacuum
        // ceiling into a scheduled merge when the space is otherwise under
        // its normal segment-count trigger.
        if best_tier.is_none()
            && indexed.len() >= 2
            && (indexed.len() > limits.max_indexed_segments || delete_ratio_breach)
        {
            let mut sorted = indexed.clone();
            sorted.sort_by_key(|(_, c)| *c);
            let take = sorted.len().min(max_fan_in).max(2).min(sorted.len());
            best_tier = Some((0, sorted[..take].to_vec()));
        }

        Ok(best_tier.map(|(_, targets)| targets.into_iter().map(|(name, _)| name).collect()))
    }

    pub(crate) fn select_merge_candidates_with_limits_be(
        &self,
        r: &AnyRead<'_>,
        logical_name: &str,
        limits: DenseMergeCandidateLimits,
    ) -> Result<Option<Vec<String>>, VectorError> {
        let max_fan_in = limits.max_fan_in();
        let dense_spaces = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let space = dense_spaces.get(logical_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", logical_name))
        })?;

        if limits.uses_bounded_prefix() {
            let indexed_count = space
                .segments
                .iter()
                .filter(|segment| segment.role == DenseVectorSegmentRole::Indexed)
                .count();
            if indexed_count <= limits.max_indexed_segments {
                return Ok(None);
            }

            let cores = self
                .cores
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let mut indexed = Vec::with_capacity(limits.bounded_scan_len());
            let mut scanned = 0usize;
            for segment in &space.segments {
                if segment.role != DenseVectorSegmentRole::Indexed {
                    continue;
                }
                scanned += 1;
                if let Some(core) = cores.get(&segment.physical_name) {
                    let count = core.level_zero_count(r).unwrap_or(0) as usize;
                    let tombstoned =
                        self.tombstone_count_for_segment_cached(&segment.physical_name);
                    let deleted = core.deleted_count();
                    indexed.push((
                        segment.physical_name.clone(),
                        count.saturating_sub(tombstoned).saturating_sub(deleted),
                    ));
                } else {
                    note_missing_segment(
                        &self.collection_name,
                        logical_name,
                        &segment.physical_name,
                        "bounded_merge_selection_lsm",
                    );
                }
                if scanned >= limits.bounded_scan_len() {
                    break;
                }
            }
            if indexed.len() < 2 {
                return Ok(None);
            }
            indexed.sort_by_key(|(_, count)| *count);
            let take = indexed.len().min(max_fan_in);
            return Ok((take >= 2).then(|| {
                indexed[..take]
                    .iter()
                    .map(|(name, _)| name.clone())
                    .collect()
            }));
        }

        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let mut indexed: Vec<(String, usize)> = Vec::new();
        let mut delete_ratio_breach = false;
        for seg in &space.segments {
            if seg.role != DenseVectorSegmentRole::Indexed {
                continue;
            }
            let Some(core) = cores.get(&seg.physical_name) else {
                note_missing_segment(
                    &self.collection_name,
                    logical_name,
                    &seg.physical_name,
                    "merge_selection_lsm",
                );
                continue;
            };
            let count = core.level_zero_count(r).unwrap_or(0) as usize;
            let tombstoned = self.tombstone_count_for_segment_cached(&seg.physical_name);
            let deleted = core.deleted_count();
            if count > 0 && (deleted as f64 / count as f64) > delete_vacuum_fraction() {
                delete_ratio_breach = true;
            }
            indexed.push((
                seg.physical_name.clone(),
                count.saturating_sub(tombstoned).saturating_sub(deleted),
            ));
        }

        // A segment past the deleted-ratio ceiling (HELIX_LSM_DELETE_VACUUM_FRACTION,
        // issue #29 "fix 2") schedules a merge even when the space is below its
        // normal segment-count trigger — the fallback below picks it up (it
        // always sorts to the smallest effective size).
        if indexed.len() <= limits.max_indexed_segments
            && !(delete_ratio_breach && indexed.len() >= 2)
        {
            return Ok(None);
        }

        let tier_of = |count: usize| -> usize {
            Self::TIER_BOUNDARIES
                .iter()
                .position(|&b| count < b)
                .unwrap_or(Self::TIER_BOUNDARIES.len())
        };
        let mut tier_buckets: HashMap<usize, Vec<(String, usize)>> = HashMap::new();
        for (name, count) in &indexed {
            tier_buckets
                .entry(tier_of(*count))
                .or_default()
                .push((name.clone(), *count));
        }

        let mut best_tier: Option<(usize, Vec<(String, usize)>)> = None;
        for (tier, mut members) in tier_buckets {
            if members.len() <= Self::TIER_MAX_PER_CLASS {
                continue;
            }
            members.sort_by_key(|(_, c)| *c);
            let overflow = members.len() - Self::TIER_MAX_PER_CLASS;
            let take = overflow.max(2).min(max_fan_in).min(members.len());
            if best_tier
                .as_ref()
                .map_or(true, |(_, prev)| members.len() > prev.len())
            {
                best_tier = Some((tier, members[..take].to_vec()));
            }
        }

        if best_tier.is_none()
            && indexed.len() >= 2
            && (indexed.len() > limits.max_indexed_segments || delete_ratio_breach)
        {
            let mut sorted = indexed.clone();
            sorted.sort_by_key(|(_, c)| *c);
            let take = sorted.len().min(max_fan_in).max(2).min(sorted.len());
            best_tier = Some((0, sorted[..take].to_vec()));
        }

        Ok(best_tier.map(|(_, targets)| targets.into_iter().map(|(name, _)| name).collect()))
    }

    /// Split-phase merge — read-only phase.
    ///
    /// Exports vectors from the given segments and builds an HNSW index
    /// entirely in memory. No write lock is held. Returns `PreparedMerge`
    /// which can be flushed under a short write txn via `flush_prepared_merge`.
    pub fn prepare_merge(
        &self,
        txn: &RoTxn,
        _logical_name: &str,
        merge_targets: &[String],
        hnsw_config: &HNSWConfig,
    ) -> Result<PreparedMerge, VectorError> {
        // Hold the global HNSW build permit across export + graph build. Export
        // itself can be a multi-GB working set on large merges, so acquiring
        // only inside `build_hnsw_in_memory` would let queued jobs retain those
        // vectors while waiting and still spike RSS.
        let build_permit = acquire_build_permit()?;
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

        // Export all vectors from the merge targets, skipping any copy a
        // re-upsert has tombstoned. A tombstoned (segment, id) copy is a
        // SUPERSEDED version — the live copy lives in a newer segment (usually
        // the mutable tail, outside merge_targets). Dropping it here is the
        // deterministic reclamation point: the merged segment never carries a
        // stale version regardless of which segment it lived in. Search is
        // unaffected (seen_ids already makes newest win on read).
        let tombstones_active = self.tombstones_active();
        let mut exported = Vec::new();
        let mut seen_ids = HashSet::new();
        let mut dropped_tombstoned = 0u64;
        for phys in merge_targets {
            let core = cores.get(phys).ok_or_else(|| {
                VectorError::VectorCoreError(format!("Segment core '{}' missing", phys))
            })?;
            let exported_rows = {
                let rd = core.backend.read_borrowed(txn);
                core.export_level_zero(&rd)?
            };
            for (id, data, fields) in exported_rows {
                if tombstones_active && self.is_tombstoned(txn, phys, id)? {
                    dropped_tombstoned += 1;
                    continue;
                }
                if seen_ids.insert(id) {
                    exported.push((id, data, fields));
                }
            }
        }
        drop(cores);
        if dropped_tombstoned > 0 {
            metrics::counter!(
                "helix_reupsert_tombstones_dropped_on_merge_total",
                "collection" => self.collection_name.clone(),
            )
            .increment(dropped_tombstoned);
        }

        // Phase 4: graph-affinity reordering. BFS over graph out-edges,
        // restricted to the exported id set, produces a layout where
        // graph-adjacent vectors share disk pages in the mmap sidecar.
        // HNSW traversal on the merged segment then hits fewer distinct
        // 4KB pages per query — the VeloANN affinity co-placement trick,
        // but using our unique graph edges as the affinity signal instead
        // of inferring it from HNSW.
        if Self::graph_affinity_merge_enabled() {
            let reorder_start = std::time::Instant::now();
            let (reordered, edges_followed) = self.graph_affinity_reorder(txn, exported);
            exported = reordered;
            metrics::histogram!(
                "helix_graph_affinity_reorder_duration_ms",
                "collection" => self.collection_name.clone(),
            )
            .record(reorder_start.elapsed().as_secs_f64() * 1000.0);
            metrics::counter!(
                "helix_graph_affinity_edges_followed_total",
                "collection" => self.collection_name.clone(),
            )
            .increment(edges_followed);
        }

        // Build HNSW in memory using the same logic as prepare_index_from_flat
        // but operating on the exported vector set.
        let prepared_index = VectorCore::build_hnsw_in_memory_owned_with_permit(
            exported,
            hnsw_config,
            &build_permit,
        )?;

        Ok(PreparedMerge {
            merge_targets: merge_targets.to_vec(),
            prepared_index,
        })
    }

    /// Split-phase merge — write phase.
    ///
    /// Creates a new segment, inserts exported vectors flat, flushes the
    /// pre-built HNSW index, and swaps out the old segments. The write
    /// lock is held only for linear I/O (no graph construction).
    pub fn flush_prepared_merge(
        &self,
        env: &Env<WithTls>,
        txn: &mut RwTxn,
        logical_name: &str,
        hnsw_config: HNSWConfig,
        merge: PreparedMerge,
    ) -> Result<bool, VectorError> {
        let config = self.get_config(logical_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", logical_name))
        })?;

        // Allocate a clean segment name and open its DBs.
        let (merged_name, merged_core) = {
            let mut dense_spaces = self
                .dense_spaces
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let space = dense_spaces.get_mut(logical_name).ok_or_else(|| {
                VectorError::VectorCoreError(format!("Named vector '{}' not found", logical_name))
            })?;
            let cores = self
                .cores
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            self.allocate_dense_core_locked(
                env,
                txn,
                logical_name,
                &config,
                hnsw_config,
                space,
                &cores,
            )?
        };

        // Insert flat rows from the prepared artifact, then flush the pre-built
        // graph. The raw rows are held once in PreparedIndex instead of also in
        // a separate exported merge buffer.
        let mut prepared_index = merge.prepared_index;
        let total = prepared_index.point_ids.len();
        merged_core.flush_prepared_flat_chunk(txn, &mut prepared_index, 0, total)?;
        merged_core.flush_prepared_index(txn, prepared_index)?;

        // Insert the new merged core first so readers can never observe segment
        // metadata that points to a missing core.
        {
            let mut cores = self
                .cores
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            cores.insert(merged_name.clone(), merged_core);
        }

        // Publish the metadata swap while the old cores are still present.
        let mut dense_spaces = self
            .dense_spaces
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        if let Some(space) = dense_spaces.get_mut(logical_name) {
            space
                .segments
                .retain(|s| !merge.merge_targets.contains(&s.physical_name));
            space.segments.push(DenseVectorSegmentMetadata {
                physical_name: merged_name.clone(),
                role: DenseVectorSegmentRole::Indexed,
            });
        }
        drop(dense_spaces);

        // Only after the metadata points at the merged segment do we clear and
        // remove the retired cores. Readers that snapped the old metadata can
        // still resolve the old cores during the transition window.
        let mut cores = self
            .cores
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        for phys in &merge.merge_targets {
            if let Some(core) = cores.get(phys) {
                let _ = core.clear(txn);
            }
            cores.remove(phys);
        }
        drop(cores);

        // Retired segments are gone — drop their tombstone entries so the
        // tombstone DB tracks only live segments. (The merged segment's own
        // rows are already deduped/non-superseded.) Best-effort: a leftover
        // tombstone is harmless and will be reconstructed/ignored.
        if self.tombstones_active() {
            for phys in &merge.merge_targets {
                let _ = self.clear_segment_tombstones(txn, phys);
            }
        }

        Ok(true)
    }

    // ─── Chunked merge publish ────────────────────────────────────────
    //
    // The non-chunked `flush_prepared_merge` writes the entire merged
    // segment + the publish swap inside ONE exclusive write txn. For
    // 50k+ vector merges this can hold the LMDB writer for 30+ seconds,
    // wedging every other operation on the env.
    //
    // The helpers below split that work into phases that each fit in
    // ~250 ms. The orchestrator (`run_chunked_merge_publish` in
    // `storage_core/replication.rs`) runs each helper inside its own
    // `with_exclusive_write_txn`. Phases B and C write to a *new*
    // segment name that no reader knows about until phase D commits the
    // metadata swap, so partial chunks are invisible.

    /// Phase A: allocate a fresh segment name, create its LMDB databases, and
    /// register it in `space.segments` as `Building`.
    ///
    /// Recording the `Building` entry in the persisted metadata (caller must
    /// call `set_dense_vector_spaces_metadata` in the same txn) means that a
    /// crash between phase A and phase D leaves an orphan that
    /// `gc_orphan_building_segments_deferred` can detect and hand off to
    /// `SegmentReaper` on the next index cycle. Without this registration,
    /// `alloc_segment_id` could silently reuse the same ID and inherit
    /// the partial data written during phases B/C.
    ///
    /// Returns the merged-segment physical name. Caller commits the txn
    /// before invoking subsequent phases.
    pub fn create_merge_target(
        &self,
        env: &Env<WithTls>,
        txn: &mut RwTxn,
        logical_name: &str,
        hnsw_config: HNSWConfig,
    ) -> Result<String, VectorError> {
        let config = self.get_config(logical_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", logical_name))
        })?;
        let (merged_name, core) = {
            let mut dense_spaces = self
                .dense_spaces
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let space = dense_spaces.get_mut(logical_name).ok_or_else(|| {
                VectorError::VectorCoreError(format!("Named vector '{}' not found", logical_name))
            })?;
            let cores = self
                .cores
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let (name, core) = self.allocate_dense_core_locked(
                env,
                txn,
                logical_name,
                &config,
                hnsw_config,
                space,
                &cores,
            )?;
            // Register immediately as Building so the ID is reserved in
            // persisted metadata and gc_orphan_building_segments_deferred
            // can find crash orphans from the A→D window.
            space.segments.push(DenseVectorSegmentMetadata {
                physical_name: name.clone(),
                role: DenseVectorSegmentRole::Building,
            });
            (name, core)
        };
        let mut cores = self
            .cores
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        cores.insert(merged_name.clone(), core);
        Ok(merged_name)
    }

    pub fn create_merge_target_lsm(
        &self,
        data_dir: &Path,
        logical_name: &str,
        hnsw_config: HNSWConfig,
    ) -> Result<String, VectorError> {
        let config = self.get_config(logical_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", logical_name))
        })?;
        let (merged_name, core) = {
            let mut dense_spaces = self
                .dense_spaces
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let space = dense_spaces.get_mut(logical_name).ok_or_else(|| {
                VectorError::VectorCoreError(format!("Named vector '{}' not found", logical_name))
            })?;
            let cores = self
                .cores
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let (name, core) = self.allocate_dense_core_locked_lsm(
                data_dir,
                logical_name,
                &config,
                hnsw_config,
                space,
                &cores,
            )?;
            space.segments.push(DenseVectorSegmentMetadata {
                physical_name: name.clone(),
                role: DenseVectorSegmentRole::Building,
            });
            (name, core)
        };
        let mut cores = self
            .cores
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        cores.insert(merged_name.clone(), core);
        Ok(merged_name)
    }

    /// Phase B: append a chunk of prepared vectors to the merge target's
    /// flat storage. The merged segment is unobserved (not yet in the
    /// dense_spaces metadata), so this can run in a tight write txn
    /// without blocking concurrent readers of the live segments.
    pub fn flush_merge_flat_chunk(
        &self,
        txn: &mut RwTxn,
        merged_name: &str,
        prepared: &mut super::vector_core::PreparedIndex,
        start: usize,
        end: usize,
    ) -> Result<(), VectorError> {
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let core = cores.get(merged_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Merge target core '{}' not found", merged_name))
        })?;
        core.flush_prepared_flat_chunk(txn, prepared, start, end)
    }

    pub fn flush_merge_flat_chunk_be(
        &self,
        w: &mut AnyWrite<'_>,
        merged_name: &str,
        prepared: &mut super::vector_core::PreparedIndex,
        start: usize,
        end: usize,
    ) -> Result<(), VectorError> {
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let core = cores.get(merged_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Merge target core '{}' not found", merged_name))
        })?;
        core.flush_prepared_flat_chunk_be(w, prepared, start, end)
    }

    /// Phase C: write a slice of the prepared HNSW into the merge target.
    /// Mutates `prepared.point_fields[start..end]` to take the field maps,
    /// so successive calls do not double-write.
    pub fn flush_merge_index_chunk(
        &self,
        txn: &mut RwTxn,
        merged_name: &str,
        prepared: &mut super::vector_core::PreparedIndex,
        start: usize,
        end: usize,
    ) -> Result<(), VectorError> {
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let core = cores.get(merged_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Merge target core '{}' not found", merged_name))
        })?;
        core.flush_prepared_index_chunk(txn, prepared, start, end)
    }

    pub fn flush_merge_index_chunk_be(
        &self,
        w: &mut AnyWrite<'_>,
        merged_name: &str,
        prepared: &mut super::vector_core::PreparedIndex,
        start: usize,
        end: usize,
    ) -> Result<(), VectorError> {
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let core = cores.get(merged_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Merge target core '{}' not found", merged_name))
        })?;
        core.flush_prepared_index_chunk_be(w, prepared, start, end)
    }

    /// Phase C-fin: write the entry point on the merge target. Idempotent
    /// under `has_index`. Must run after all `flush_merge_index_chunk`
    /// calls and before the publish swap.
    pub fn finalize_merge_index(
        &self,
        txn: &mut RwTxn,
        merged_name: &str,
        prepared: &super::vector_core::PreparedIndex,
    ) -> Result<(), VectorError> {
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let core = cores.get(merged_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Merge target core '{}' not found", merged_name))
        })?;
        core.finalize_prepared_index_entry(txn, prepared)
    }

    pub fn finalize_merge_index_be(
        &self,
        w: &mut AnyWrite<'_>,
        merged_name: &str,
        prepared: &super::vector_core::PreparedIndex,
    ) -> Result<(), VectorError> {
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let core = cores.get(merged_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Merge target core '{}' not found", merged_name))
        })?;
        core.finalize_prepared_index_entry_be(w, prepared)
    }

    /// Phase C-quantize (optional): convert the merged segment's mmap
    /// sidecar from HVEC to HVS8. Called between `finalize_merge_index`
    /// (Phase C-fin) and `publish_merge_target` (Phase D) so the
    /// quantization is invisible to readers — the segment is still in
    /// `Building` role with no metadata pointer.
    ///
    /// Returns `Ok(true)` if a conversion happened, `Ok(false)` if the
    /// segment was already quantized or had no mmap store. Errors leave
    /// the segment unchanged (still HVEC); the caller can either retry
    /// or proceed to publish without quantization.
    pub fn quantize_merge_target(&self, merged_name: &str) -> Result<bool, VectorError> {
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let core = cores.get(merged_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Merge target core '{}' not found", merged_name))
        })?;
        core.quantize_mmap()
    }

    /// Materialize the merge target's immutable sidecar directly as HVS8 from
    /// the prepared rows. This is used for compact spindle segments that never
    /// created a mutable HVEC sidecar, so the old HVEC→HVS8 conversion path has
    /// nothing to convert.
    pub fn materialize_merge_target_hvs8_sidecar(
        &self,
        merged_name: &str,
        prepared: &super::vector_core::PreparedIndex,
    ) -> Result<bool, VectorError> {
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let core = cores.get(merged_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Merge target core '{}' not found", merged_name))
        })?;
        core.materialize_hvs8_sidecar_from_prepared(prepared)
    }

    /// Materialize the merge target's immutable sidecar directly as HVTQ from
    /// the prepared rows. Used for compact TurboProd segments so the merged
    /// segment stores one sidecar copy of encoded payloads instead of one LMDB
    /// value per vector plus a separate sidecar.
    pub fn materialize_merge_target_tq_sidecar(
        &self,
        merged_name: &str,
        prepared: &super::vector_core::PreparedIndex,
    ) -> Result<bool, VectorError> {
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let core = cores.get(merged_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Merge target core '{}' not found", merged_name))
        })?;
        core.materialize_tq_sidecar_from_prepared(prepared)
    }

    /// Write point-id → sidecar ordinal mappings for a merge target after its
    /// sidecar file has already been fully materialized and synced.
    pub fn attach_merge_target_sidecar_ordinals(
        &self,
        txn: &mut RwTxn,
        merged_name: &str,
        prepared: &super::vector_core::PreparedIndex,
    ) -> Result<(), VectorError> {
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let core = cores.get(merged_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Merge target core '{}' not found", merged_name))
        })?;
        core.write_prepared_sidecar_ordinals(txn, prepared)
    }

    pub fn attach_merge_target_sidecar_ordinals_be(
        &self,
        w: &mut AnyWrite<'_>,
        merged_name: &str,
        prepared: &super::vector_core::PreparedIndex,
    ) -> Result<(), VectorError> {
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let core = cores.get(merged_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Merge target core '{}' not found", merged_name))
        })?;
        core.write_prepared_sidecar_ordinals_be(w, prepared)
    }

    pub(crate) fn prepare_next_hvtq_sidecar_backfill_be(
        &self,
        r: &AnyRead<'_>,
        logical_name: &str,
        max_segments: usize,
    ) -> Result<Option<HvtqSidecarBackfillPlan>, VectorError> {
        if max_segments == 0 {
            return Ok(None);
        }
        let indexed_names = {
            let dense_spaces = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let Some(space) = dense_spaces.get(logical_name) else {
                return Ok(None);
            };
            space
                .segments
                .iter()
                .filter(|segment| segment.role == DenseVectorSegmentRole::Indexed)
                .map(|segment| segment.physical_name.clone())
                .collect::<Vec<_>>()
        };

        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        for physical_name in indexed_names {
            let Some(core) = cores.get(&physical_name) else {
                continue;
            };
            if let Some(point_ids) = core.prepare_hvtq_sidecar_backfill_be(r)? {
                return Ok(Some(HvtqSidecarBackfillPlan {
                    physical_name,
                    point_ids,
                }));
            }
        }
        Ok(None)
    }

    pub(crate) fn has_hvtq_sidecar_backfill_debt_be(
        &self,
        r: &AnyRead<'_>,
        logical_name: &str,
    ) -> Result<bool, VectorError> {
        let indexed_names = {
            let dense_spaces = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let Some(space) = dense_spaces.get(logical_name) else {
                return Ok(false);
            };
            space
                .segments
                .iter()
                .filter(|segment| segment.role == DenseVectorSegmentRole::Indexed)
                .map(|segment| segment.physical_name.clone())
                .collect::<Vec<_>>()
        };

        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        for physical_name in indexed_names {
            let Some(core) = cores.get(&physical_name) else {
                continue;
            };
            if core.hvtq_sidecar_backfill_needed_be(r)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(crate) fn apply_hvtq_sidecar_backfill_be(
        &self,
        w: &mut AnyWrite<'_>,
        plan: &HvtqSidecarBackfillPlan,
    ) -> Result<(), VectorError> {
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let core = cores.get(&plan.physical_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!(
                "HVTQ sidecar backfill core '{}' not found",
                plan.physical_name
            ))
        })?;
        core.write_sidecar_ordinals_be(w, &plan.point_ids)
    }

    /// Convert existing indexed dense sidecars from HVEC to HVS8.
    ///
    /// This is an explicit maintenance path for already-published segments.
    /// It only touches `Indexed` segments because mutable segments must remain
    /// appendable. The per-core mmap write lock excludes concurrent vector
    /// readers while a single sidecar is swapped, and failures leave that
    /// segment in its original HVEC/HVS8 state.
    pub fn quantize_indexed_mmap_sidecars(
        &self,
        max_segments: Option<usize>,
    ) -> SidecarQuantizeStats {
        let started = Instant::now();
        let mut stats = SidecarQuantizeStats::default();
        let limit = max_segments.unwrap_or(usize::MAX);

        let segment_names: Vec<String> = match self.dense_spaces.read() {
            Ok(spaces) => spaces
                .values()
                .flat_map(|space| {
                    space
                        .segments
                        .iter()
                        .filter(|segment| segment.role == DenseVectorSegmentRole::Indexed)
                        .map(|segment| segment.physical_name.clone())
                })
                .take(limit)
                .collect(),
            Err(e) => {
                stats.error_count += 1;
                stats
                    .errors
                    .push(format!("dense_spaces lock poisoned: {}", e));
                stats.duration_ms = started.elapsed().as_millis() as u64;
                return stats;
            }
        };

        for segment_name in segment_names {
            stats.checked_segments += 1;
            let converted = {
                let cores = match self.cores.read() {
                    Ok(cores) => cores,
                    Err(e) => {
                        stats.error_count += 1;
                        stats.errors.push(format!("cores lock poisoned: {}", e));
                        break;
                    }
                };
                let Some(core) = cores.get(&segment_name) else {
                    stats.skipped_segments += 1;
                    continue;
                };
                core.quantize_mmap()
            };
            match converted {
                Ok(true) => stats.converted_segments += 1,
                Ok(false) => stats.skipped_segments += 1,
                Err(e) => {
                    stats.error_count += 1;
                    stats.errors.push(format!("{}: {}", segment_name, e));
                }
            }
        }

        stats.duration_ms = started.elapsed().as_millis() as u64;
        metrics::histogram!("helix_sidecar_quantize_duration_ms")
            .record(started.elapsed().as_secs_f64() * 1000.0);
        if stats.converted_segments > 0 {
            metrics::counter!("helix_sidecar_quantized_segments_total")
                .increment(stats.converted_segments as u64);
        }
        if stats.error_count > 0 {
            metrics::counter!("helix_sidecar_quantize_errors_total")
                .increment(stats.error_count as u64);
        }
        stats
    }

    /// Phase D: atomic publish swap. Removes `merge_targets` from the
    /// space metadata and promotes the merged segment to `Indexed`.
    ///
    /// `create_merge_target` now registers `merged_name` in `space.segments`
    /// as `Building` so the ID is reserved across crashes. Phase D therefore
    /// promotes the existing entry instead of pushing a new one. A push-new
    /// fallback handles collections created by an older binary that never
    /// wrote the `Building` entry.
    ///
    /// The caller is responsible for persisting the metadata blob via
    /// `set_dense_vector_spaces_metadata` in the same txn.
    pub fn publish_merge_target(
        &self,
        txn: &RoTxn,
        logical_name: &str,
        merged_name: &str,
        merge_targets: &[String],
    ) -> Result<(), VectorError> {
        self.validate_dense_segment_publishable(txn, logical_name, merged_name)?;

        let mut dense_spaces = self
            .dense_spaces
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let Some(space) = dense_spaces.get_mut(logical_name) else {
            return Err(VectorError::VectorCoreError(format!(
                "Named vector '{}' not found",
                logical_name
            )));
        };
        space
            .segments
            .retain(|s| !merge_targets.contains(&s.physical_name));
        // Promote Building → Indexed if create_merge_target already registered
        // the entry; otherwise push new (backward-compat with older binaries).
        if let Some(seg) = space
            .segments
            .iter_mut()
            .find(|s| s.physical_name == merged_name)
        {
            seg.role = DenseVectorSegmentRole::Indexed;
        } else {
            space.segments.push(DenseVectorSegmentMetadata {
                physical_name: merged_name.to_string(),
                role: DenseVectorSegmentRole::Indexed,
            });
        }
        Ok(())
    }

    pub fn publish_merge_target_be(
        &self,
        r: &AnyRead<'_>,
        logical_name: &str,
        merged_name: &str,
        merge_targets: &[String],
    ) -> Result<(), VectorError> {
        self.validate_dense_segment_publishable_be(r, logical_name, merged_name)?;

        let mut dense_spaces = self
            .dense_spaces
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let Some(space) = dense_spaces.get_mut(logical_name) else {
            return Err(VectorError::VectorCoreError(format!(
                "Named vector '{}' not found",
                logical_name
            )));
        };
        space
            .segments
            .retain(|s| !merge_targets.contains(&s.physical_name));
        if let Some(seg) = space
            .segments
            .iter_mut()
            .find(|s| s.physical_name == merged_name)
        {
            seg.role = DenseVectorSegmentRole::Indexed;
        } else {
            space.segments.push(DenseVectorSegmentMetadata {
                physical_name: merged_name.to_string(),
                role: DenseVectorSegmentRole::Indexed,
            });
        }
        Ok(())
    }

    /// Phase E: clear and remove one retired source segment. Runs in
    /// its own short txn after the publish swap, so readers that
    /// snapped the old metadata can still resolve it during the
    /// transition window before this runs.
    pub fn drop_retired_segment(
        &self,
        txn: &mut RwTxn,
        physical_name: &str,
    ) -> Result<(), VectorError> {
        let mut cores = self
            .cores
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        if let Some(core) = cores.get(physical_name) {
            let _ = core.clear(txn);
            let _ = core.unlink_mmap_sidecar();
        }
        cores.remove(physical_name);
        self.dirty_retired_segments
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?
            .remove(physical_name);
        Ok(())
    }

    /// Incrementally clear one retired segment. Each call deletes at most
    /// `chunk_size` records per database and returns `Ok(true)` when all
    /// five databases are fully drained and the segment has been removed
    /// from `cores`. Returns `Ok(false)` when more work remains — caller
    /// must call again in a fresh write transaction.
    ///
    /// This replaces `drop_retired_segment` on the `SegmentReaper` path
    /// to avoid the 30-60 s write-lock hold that `mdb_drop` causes on
    /// large segments.
    pub fn drop_retired_segment_chunk(
        &self,
        txn: &mut RwTxn,
        physical_name: &str,
        chunk_size: usize,
    ) -> Result<bool, VectorError> {
        let drained = {
            let cores = self
                .cores
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let Some(core) = cores.get(physical_name) else {
                return Ok(true); // Already removed — nothing to do.
            };
            core.clear_chunk(txn, chunk_size)?
        };

        if drained {
            // All five databases confirmed empty — release caches and evict core.
            let mut cores = self
                .cores
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            if let Some(core) = cores.get(physical_name) {
                let _ = core.unlink_mmap_sidecar();
                core.clear_caches();
            }
            cores.remove(physical_name);
            self.dirty_retired_segments
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?
                .remove(physical_name);
        }

        // Return true = fully done, false = more chunks needed.
        Ok(drained)
    }

    /// Drain up to `max_records` rows from a single LMDB database of the
    /// retired segment identified by `physical_name`. `db_index` selects
    /// which database (see `VectorCore::clear_chunk_db`). Returns
    /// `Ok(true)` when that database is empty.
    ///
    /// Exposed so `SegmentReaper` can open one short `with_write_txn` per
    /// database instead of bundling all five into a single 50-66 s
    /// writer-held txn. The caller is responsible for invoking
    /// `finalize_drained_retired_segment` once every database has been
    /// confirmed empty.
    pub fn drop_retired_segment_db_step(
        &self,
        env: &Env<WithTls>,
        txn: &mut RwTxn,
        physical_name: &str,
        db_index: usize,
        max_records: usize,
    ) -> Result<bool, VectorError> {
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        if let Some(core) = cores.get(physical_name) {
            return core.clear_chunk_db(txn, db_index, max_records);
        }
        drop(cores);

        let Some(core) = self.open_unregistered_retired_core(env, txn, physical_name)? else {
            // Unknown legacy segment name. Treat as drained so the reaper does
            // not spin forever; allocation will rediscover it if the DBs are
            // still dirty under an active vector prefix.
            return Ok(true);
        };
        core.clear_chunk_db(txn, db_index, max_records)
    }

    pub fn drop_retired_segment_db_step_be(
        &self,
        w: &mut AnyWrite<'_>,
        physical_name: &str,
        db_index: usize,
        max_records: usize,
    ) -> Result<bool, VectorError> {
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        if let Some(core) = cores.get(physical_name) {
            return core.clear_chunk_db_be(w, db_index, max_records);
        }
        Ok(true)
    }

    fn open_unregistered_retired_core(
        &self,
        env: &Env<WithTls>,
        txn: &mut RwTxn,
        physical_name: &str,
    ) -> Result<Option<VectorCore>, VectorError> {
        let config = {
            let configs = self
                .configs
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            configs
                .iter()
                .filter(|(logical_name, _)| {
                    segment_id_from_name(logical_name, physical_name).is_some()
                })
                .max_by_key(|(logical_name, _)| logical_name.len())
                .map(|(_, config)| config.clone())
        };

        let Some(config) = config else {
            tracing::warn!(
                collection = %self.collection_name,
                segment = %physical_name,
                "Retired dense segment has no matching named-vector config; skipping LMDB drain"
            );
            metrics::counter!(
                "helix_segment_reaper_unknown_config_total",
                "collection" => self.collection_name.clone()
            )
            .increment(1);
            return Ok(None);
        };

        VectorCore::new_named(
            env,
            txn,
            physical_name,
            HNSWConfig::new(None, None, None),
            config.distance,
            config.spindle,
            self.current_backend()?,
        )
        .map(Some)
    }

    /// Companion to `drop_retired_segment_db_step`: after every database
    /// has been confirmed empty, flush the segment's in-memory caches
    /// and remove the core from the registry. Idempotent — safe to call
    /// when the core has already been evicted.
    pub fn finalize_drained_retired_segment(&self, physical_name: &str) -> Result<(), VectorError> {
        let mut cores = self
            .cores
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        if let Some(core) = cores.get(physical_name) {
            let _ = core.unlink_mmap_sidecar();
            core.clear_caches();
        }
        cores.remove(physical_name);
        self.dirty_retired_segments
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?
            .remove(physical_name);
        Ok(())
    }

    /// Total number of dense segments currently in `Indexed` role across
    /// every named vector space on this collection. Used by the point
    /// delete path to scale its per-chunk size: HNSW neighbor patching
    /// in `delete_vectors_batch` is `O(indexed_segments)` per id, so the
    /// chunk size must shrink as fanout grows or the writer hold balloons.
    pub fn indexed_dense_segment_count(&self) -> usize {
        let dense_spaces = match self.dense_spaces.read() {
            Ok(g) => g,
            Err(_) => return 0,
        };
        dense_spaces
            .values()
            .map(|space| {
                space
                    .segments
                    .iter()
                    .filter(|s| s.role == DenseVectorSegmentRole::Indexed)
                    .count()
            })
            .sum()
    }

    /// Current dense-segment pressure used by the optimizer scheduler.
    ///
    /// `gate_debt` becomes non-zero as soon as any named vector reaches the
    /// Mutable→Building seal gate (`Indexed + Building >= cap`). That is the
    /// critical scale signal: once it fires, writes keep appending into the
    /// existing Mutable and search degrades until the optimizer/reaper drains
    /// enough segment debt to reopen sealing.
    pub fn dense_segment_debt(&self) -> DenseSegmentDebt {
        let cap = Self::segment_creation_gate_cap();
        let target = Self::max_indexed_segments_target();
        let mut debt = DenseSegmentDebt::default();

        let dense_spaces = match self.dense_spaces.read() {
            Ok(g) => g,
            Err(_) => return debt,
        };

        for space in dense_spaces.values() {
            let mut indexed = 0usize;
            let mut active = 0usize;
            let mut has_mutable = false;
            for segment in &space.segments {
                match segment.role {
                    DenseVectorSegmentRole::Indexed => {
                        indexed = indexed.saturating_add(1);
                        active = active.saturating_add(1);
                    }
                    DenseVectorSegmentRole::Building => {
                        active = active.saturating_add(1);
                    }
                    DenseVectorSegmentRole::Mutable => {
                        has_mutable = true;
                    }
                }
            }
            debt.indexed_segments = debt.indexed_segments.saturating_add(indexed);
            debt.active_segments = debt.active_segments.saturating_add(active);
            debt.merge_debt = debt
                .merge_debt
                .saturating_add(indexed.saturating_sub(target));
            debt.gate_debt = debt
                .gate_debt
                .saturating_add(active.saturating_sub(cap.saturating_sub(1)));
            // INITIAL-BUILD priority (LSM): a space with 0 Indexed segments but
            // data sitting in Mutable segments has never been HNSW-built. The
            // LMDB-era debt calc ignores Mutable, so such a space scores 0 debt
            // and the optimizer starves its first build behind perpetual merge
            // work on already-indexed collections — bulk-migrated giants stay at
            // indexed_vectors_count=0 forever and vector KNN brute-forces. Count
            // it as gate debt so the initial build outranks merges (gate_debt is
            // weighted above merge_debt in optimizer_debt_priority_score and
            // expands the per-job build budget). An empty Mutable tail below the
            // seal threshold simply no-ops on its turn, so this is safe.
            if indexed == 0 && has_mutable {
                debt.gate_debt = debt.gate_debt.saturating_add(1);
            }
        }

        debt.dirty_retired_segments = self
            .dirty_retired_segments
            .read()
            .map(|segments| segments.len())
            .unwrap_or(0);

        // Fold the re-upsert tombstone backlog into merge pressure here, using
        // the O(segments) in-memory cache (no txn, no LMDB scan), so EVERY debt
        // site — including the txn-free priority scorers — schedules merges that
        // reclaim superseded bytes. Each `TOMBSTONES_PER_MERGE_DEBT_UNIT`
        // superseded copies add one unit, so a handful of stale copies does not
        // force a merge but a sustained re-upsert backlog does.
        const TOMBSTONES_PER_MERGE_DEBT_UNIT: usize = 1024;
        let tombstones = self.tombstone_total_cached();
        debt.merge_debt = debt
            .merge_debt
            .saturating_add(tombstones / TOMBSTONES_PER_MERGE_DEBT_UNIT);

        // Fold the deferred-repair delete-tombstone backlog in the same way
        // (issue #29 "fix 2"): `deleted_total_cached` sums each core's
        // in-memory delete-tombstone set (O(segments), no txn), so this stays
        // on the same txn-free path as everything else in this function. The
        // deleted-ratio vacuum ceiling (per-segment, needs a live row count)
        // lives in `select_merge_candidates_with_limits[_be]` instead, where a
        // read handle is already available.
        let deleted = self.deleted_total_cached();
        debt.merge_debt = debt
            .merge_debt
            .saturating_add(deleted / TOMBSTONES_PER_MERGE_DEBT_UNIT);

        debt
    }

    /// Sum of `VectorCore::deleted_count()` across every currently-open dense
    /// core. In-memory only (no txn) — mirrors `tombstone_total_cached` for
    /// the deferred-repair delete-tombstone set.
    fn deleted_total_cached(&self) -> usize {
        self.cores
            .read()
            .map(|cores| cores.values().map(|core| core.deleted_count()).sum())
            .unwrap_or(0)
    }

    /// `dense_segment_debt` plus the raw superseded-copy count, for callers
    /// that want to observe/log the backlog. The merge-pressure fold now lives
    /// in `dense_segment_debt` itself (cache-backed), so this is purely
    /// additive reporting.
    pub fn dense_segment_debt_with_tombstones(
        &self,
        _txn: &RoTxn,
    ) -> DenseSegmentDebtWithTombstones {
        DenseSegmentDebtWithTombstones {
            debt: self.dense_segment_debt(),
            tombstoned_copies: self.tombstone_total_cached(),
        }
    }

    /// Promote selected `Building` segments of `logical_name` to `Indexed`.
    /// In-memory only; caller persists `dense_spaces_metadata` via the
    /// caller's own write txn after invoking this.
    pub fn promote_building_segments_to_indexed(
        &self,
        txn: &RoTxn,
        logical_name: &str,
        physical_names: &HashSet<String>,
    ) -> Result<bool, VectorError> {
        if physical_names.is_empty() {
            return Ok(false);
        }

        for physical_name in physical_names {
            self.validate_dense_segment_publishable(txn, logical_name, physical_name)?;
        }

        let mut dense_spaces = self
            .dense_spaces
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let Some(space) = dense_spaces.get_mut(logical_name) else {
            return Ok(false);
        };
        let mut changed = false;
        for seg in &mut space.segments {
            if seg.role == DenseVectorSegmentRole::Building
                && physical_names.contains(&seg.physical_name)
            {
                seg.role = DenseVectorSegmentRole::Indexed;
                changed = true;
            }
        }
        Ok(changed)
    }

    pub fn promote_building_segments_to_indexed_be(
        &self,
        r: &AnyRead<'_>,
        logical_name: &str,
        physical_names: &HashSet<String>,
    ) -> Result<bool, VectorError> {
        if physical_names.is_empty() {
            return Ok(false);
        }

        for physical_name in physical_names {
            self.validate_dense_segment_publishable_be(r, logical_name, physical_name)?;
        }

        let mut dense_spaces = self
            .dense_spaces
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let Some(space) = dense_spaces.get_mut(logical_name) else {
            return Ok(false);
        };
        let mut changed = false;
        for seg in &mut space.segments {
            if seg.role == DenseVectorSegmentRole::Building
                && physical_names.contains(&seg.physical_name)
            {
                seg.role = DenseVectorSegmentRole::Indexed;
                changed = true;
            }
        }
        Ok(changed)
    }

    /// Garbage-collect any dense segment stuck in `Building` role with
    /// zero level-zero count. This handles crashes between phases B and
    /// D of the chunked merge: a partially-written merge target is
    /// `Building` (or stays unindexed because phase D never ran), and
    /// can be safely dropped because no metadata pointer references it.
    /// Idempotent — safe to call on every startup.
    pub fn gc_orphan_building_segments(
        &self,
        txn: &mut RwTxn,
        logical_name: &str,
    ) -> Result<usize, VectorError> {
        let orphans: Vec<String> = {
            let dense_spaces = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let Some(space) = dense_spaces.get(logical_name) else {
                return Ok(0);
            };
            let cores = self
                .cores
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            space
                .segments
                .iter()
                .filter(|seg| seg.role == DenseVectorSegmentRole::Building)
                .filter_map(|seg| {
                    let core = cores.get(&seg.physical_name)?;
                    let count = {
                        let rd = core.backend.read_borrowed(&*txn);
                        core.level_zero_count(&rd).unwrap_or(0)
                    };
                    if count == 0 {
                        Some(seg.physical_name.clone())
                    } else {
                        None
                    }
                })
                .collect()
        };
        if orphans.is_empty() {
            return Ok(0);
        }
        for name in &orphans {
            let mut cores = self
                .cores
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            if let Some(core) = cores.get(name) {
                let _ = core.clear(txn);
            }
            cores.remove(name);
        }
        let mut dense_spaces = self
            .dense_spaces
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        if let Some(space) = dense_spaces.get_mut(logical_name) {
            space
                .segments
                .retain(|s| !orphans.contains(&s.physical_name));
        }
        Ok(orphans.len())
    }

    /// Legacy synchronous merge (used by finalize_and_compact).
    /// Prefer prepare_merge + flush_prepared_merge for concurrent workloads.
    pub fn maybe_merge_dense_segments<P>(
        &self,
        read_txns: &P,
        env: &Env<WithTls>,
        txn: &mut RwTxn,
        logical_name: &str,
        hnsw_config: HNSWConfig,
        max_indexed_segments: usize,
    ) -> Result<bool, VectorError>
    where
        P: DenseReadTxnProvider,
    {
        let candidates = read_txns.with_dense_read_txn(|ro_txn| {
            self.select_merge_candidates(ro_txn, logical_name, max_indexed_segments)
        })?;

        let Some(targets) = candidates else {
            return Ok(false);
        };

        // For the legacy path, do prepare + flush sequentially in the
        // caller's write txn. The prepare phase builds HNSW in memory
        // (fast) and flush writes linearly (fast). Still better than
        // the old build_index_from_flat-in-write-txn path.
        let merge = read_txns.with_dense_read_txn(|ro_txn| {
            self.prepare_merge(ro_txn, logical_name, &targets, &hnsw_config)
        })?;

        self.flush_prepared_merge(env, txn, logical_name, hnsw_config, merge)?;
        Ok(true)
    }

    pub fn cleanup_empty_dense_segments(
        &self,
        txn: &mut heed3::RwTxn,
        logical_name: &str,
    ) -> Result<bool, VectorError> {
        let mut dense_spaces = self
            .dense_spaces
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let Some(space) = dense_spaces.get_mut(logical_name) else {
            return Ok(false);
        };

        let mut empty_segments = Vec::new();
        {
            let cores = self
                .cores
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            for segment in &space.segments {
                if segment.role == DenseVectorSegmentRole::Mutable {
                    continue;
                }
                let Some(core) = cores.get(&segment.physical_name) else {
                    continue;
                };
                let is_empty = {
                    let rd = core.backend.read_borrowed(&*txn);
                    core.level_zero_count(&rd)? == 0
                };
                if is_empty {
                    empty_segments.push(segment.physical_name.clone());
                }
            }
        }

        if empty_segments.is_empty() {
            return Ok(false);
        }

        let mut cores = self
            .cores
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        for physical_name in &empty_segments {
            if let Some(core) = cores.get(physical_name) {
                core.clear(txn)?;
            }
        }
        for physical_name in &empty_segments {
            cores.remove(physical_name);
        }
        space.segments.retain(|segment| {
            !empty_segments
                .iter()
                .any(|physical_name| physical_name == &segment.physical_name)
        });

        Ok(true)
    }

    /// Garbage-collect `Building` segments with zero level-zero count by
    /// removing them from `space.segments` only, leaving the `cores` entry
    /// and LMDB pages for `SegmentReaper` to clear asynchronously. Returns
    /// the list of orphan physical names for submission to the reaper.
    ///
    /// Called once per `run_index_job` pass (before the seal step) to clean
    /// up merge targets that were registered as `Building` in phase A but
    /// whose phase D never committed (crash between A and D). Those targets
    /// have zero level-zero vectors because phases B/C wrote nothing (or
    /// partial data was present but is now stale).
    ///
    /// Idempotent — safe to call on every index pass.
    pub fn gc_orphan_building_segments_deferred(
        &self,
        txn: &RoTxn,
        logical_name: &str,
    ) -> Result<Vec<String>, VectorError> {
        let orphans: Vec<String> = {
            let dense_spaces = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let Some(space) = dense_spaces.get(logical_name) else {
                return Ok(Vec::new());
            };
            let cores = self
                .cores
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            space
                .segments
                .iter()
                .filter(|seg| seg.role == DenseVectorSegmentRole::Building)
                .filter_map(|seg| {
                    let core = cores.get(&seg.physical_name)?;
                    let count = {
                        let rd = core.backend.read_borrowed(&*txn);
                        core.level_zero_count(&rd).unwrap_or(0)
                    };
                    if count == 0 {
                        Some(seg.physical_name.clone())
                    } else {
                        None
                    }
                })
                .collect()
        };
        if orphans.is_empty() {
            return Ok(Vec::new());
        }
        {
            let mut dense_spaces = self
                .dense_spaces
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            if let Some(space) = dense_spaces.get_mut(logical_name) {
                space
                    .segments
                    .retain(|s| !orphans.contains(&s.physical_name));
            }
        }
        Ok(orphans)
    }

    pub fn gc_orphan_building_segments_deferred_be(
        &self,
        r: &AnyRead<'_>,
        logical_name: &str,
    ) -> Result<Vec<String>, VectorError> {
        let orphans: Vec<String> = {
            let dense_spaces = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let Some(space) = dense_spaces.get(logical_name) else {
                return Ok(Vec::new());
            };
            let cores = self
                .cores
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            space
                .segments
                .iter()
                .filter(|seg| seg.role == DenseVectorSegmentRole::Building)
                .filter_map(|seg| {
                    let core = cores.get(&seg.physical_name)?;
                    let count = core.level_zero_count(r).unwrap_or(0);
                    if count == 0 {
                        Some(seg.physical_name.clone())
                    } else {
                        None
                    }
                })
                .collect()
        };
        if orphans.is_empty() {
            return Ok(Vec::new());
        }
        {
            let mut dense_spaces = self
                .dense_spaces
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            if let Some(space) = dense_spaces.get_mut(logical_name) {
                space
                    .segments
                    .retain(|s| !orphans.contains(&s.physical_name));
            }
        }
        Ok(orphans)
    }

    /// Identify empty (non-`Mutable`, level-zero count 0) segments and
    /// remove them from `space.segments` only — leaving the entries in
    /// `cores` and the LMDB pages untouched. Returns the list of physical
    /// names that the caller should hand to `SegmentReaper` after the
    /// metadata-mutation txn commits.
    ///
    /// Splitting "remove the metadata pointer" from "clear the LMDB
    /// pages" lets the heavy `core.clear(txn)` work — five `db.clear`
    /// calls per segment, each O(pages) — happen in a fresh background
    /// txn instead of holding the per-collection write gate. Phase E of
    /// `run_chunked_merge_publish` and every call site of the original
    /// synchronous variant previously caused 30-263 s exclusive write
    /// txns under load.
    ///
    /// Safety: once a segment is no longer in `space.segments`, no
    /// `cleanup_empty_dense_segments` / search / scroll / count code path
    /// can resolve it (they all enumerate via `space.segments` first).
    /// The `cores` entry remains so the reaper can find the
    /// `VectorCore` to call `clear` on; the reaper removes it after
    /// clearing.
    pub fn cleanup_empty_dense_segments_deferred(
        &self,
        txn: &mut heed3::RwTxn,
        logical_name: &str,
    ) -> Result<Vec<String>, VectorError> {
        let mut dense_spaces = self
            .dense_spaces
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let Some(space) = dense_spaces.get_mut(logical_name) else {
            return Ok(Vec::new());
        };

        let mut empty_segments = Vec::new();
        {
            let cores = self
                .cores
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            for segment in &space.segments {
                if segment.role == DenseVectorSegmentRole::Mutable {
                    continue;
                }
                let Some(core) = cores.get(&segment.physical_name) else {
                    continue;
                };
                let is_empty = {
                    let rd = core.backend.read_borrowed(&*txn);
                    core.level_zero_count(&rd)? == 0
                };
                if is_empty {
                    empty_segments.push(segment.physical_name.clone());
                }
            }
        }

        if empty_segments.is_empty() {
            return Ok(Vec::new());
        }

        space.segments.retain(|segment| {
            !empty_segments
                .iter()
                .any(|physical_name| physical_name == &segment.physical_name)
        });

        Ok(empty_segments)
    }

    /// Read-txn variant of `cleanup_empty_dense_segments_deferred`'s
    /// "locate" phase: identify empty segments via `level_zero_count`
    /// (the O(N) prefix scan that turns into a 100s+ exclusive write-txn
    /// hold under load when called inside `with_exclusive_write_txn`).
    /// Returns the list of physical names that are candidates for removal;
    /// the caller commits the actual `space.segments` mutation in a
    /// separate, brief exclusive write txn via
    /// `apply_dense_segment_removal`.
    ///
    /// Idempotent. Safe to invoke from any read-txn context — does not
    /// hold the writer gate.
    pub fn cleanup_empty_dense_segments_locate(
        &self,
        rtxn: &RoTxn,
        logical_name: &str,
    ) -> Result<Vec<String>, VectorError> {
        let dense_spaces = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let Some(space) = dense_spaces.get(logical_name) else {
            return Ok(Vec::new());
        };
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let mut empty_segments = Vec::new();
        for segment in &space.segments {
            if segment.role == DenseVectorSegmentRole::Mutable {
                continue;
            }
            let Some(core) = cores.get(&segment.physical_name) else {
                continue;
            };
            let is_empty = {
                let rd = core.backend.read_borrowed(rtxn);
                core.level_zero_count(&rd)? == 0
            };
            if is_empty {
                empty_segments.push(segment.physical_name.clone());
            }
        }
        Ok(empty_segments)
    }

    pub fn cleanup_empty_dense_segments_locate_be(
        &self,
        r: &AnyRead<'_>,
        logical_name: &str,
    ) -> Result<Vec<String>, VectorError> {
        let dense_spaces = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let Some(space) = dense_spaces.get(logical_name) else {
            return Ok(Vec::new());
        };
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let mut empty_segments = Vec::new();
        for segment in &space.segments {
            if segment.role == DenseVectorSegmentRole::Mutable {
                continue;
            }
            let Some(core) = cores.get(&segment.physical_name) else {
                continue;
            };
            if core.level_zero_count(r)? == 0 {
                empty_segments.push(segment.physical_name.clone());
            }
        }
        Ok(empty_segments)
    }

    pub fn repair_unavailable_externalized_markers(
        &self,
        txn: &mut RwTxn,
        logical_name: &str,
        limit_per_segment: usize,
    ) -> Result<usize, VectorError> {
        if limit_per_segment == 0 {
            return Ok(0);
        }

        let segments: Vec<(String, DenseVectorSegmentRole)> = {
            let dense_spaces = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let Some(space) = dense_spaces.get(logical_name) else {
                return Ok(0);
            };
            space
                .segments
                .iter()
                .map(|segment| (segment.physical_name.clone(), segment.role))
                .collect()
        };

        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let mut repaired = 0usize;
        for (physical_name, role) in segments {
            let Some(core) = cores.get(&physical_name) else {
                continue;
            };
            if !core.externalized_marker_repair_needed() {
                continue;
            }

            let purged = core.purge_unavailable_externalized_markers(txn, limit_per_segment)?;
            if purged == 0 {
                continue;
            }

            repaired = repaired.saturating_add(purged);
            metrics::counter!(
                "helix_externalized_marker_segment_repair_total",
                "collection" => self.collection_name.clone(),
                "vector" => logical_name.to_string(),
                "role" => format!("{role:?}")
            )
            .increment(purged as u64);
            tracing::warn!(
                collection = %self.collection_name,
                vector = logical_name,
                segment = %physical_name,
                role = ?role,
                purged,
                limit_per_segment,
                "Purged unavailable externalized vector markers from dense segment"
            );
        }

        Ok(repaired)
    }

    pub fn repair_next_unavailable_externalized_marker_segment(
        &self,
        txn: &mut RwTxn,
        logical_name: &str,
        limit_per_segment: usize,
    ) -> Result<Option<usize>, VectorError> {
        if limit_per_segment == 0 {
            return Ok(None);
        }

        let segments: Vec<(String, DenseVectorSegmentRole)> = {
            let dense_spaces = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let Some(space) = dense_spaces.get(logical_name) else {
                return Ok(None);
            };
            space
                .segments
                .iter()
                .map(|segment| (segment.physical_name.clone(), segment.role))
                .collect()
        };

        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        for (physical_name, role) in segments {
            let Some(core) = cores.get(&physical_name) else {
                continue;
            };
            if !core.externalized_marker_repair_needed() {
                continue;
            }

            let purged = core.purge_unavailable_externalized_markers(txn, limit_per_segment)?;
            if purged > 0 {
                metrics::counter!(
                    "helix_externalized_marker_segment_repair_total",
                    "collection" => self.collection_name.clone(),
                    "vector" => logical_name.to_string(),
                    "role" => format!("{role:?}")
                )
                .increment(purged as u64);
                tracing::warn!(
                    collection = %self.collection_name,
                    vector = logical_name,
                    segment = %physical_name,
                    role = ?role,
                    purged,
                    limit_per_segment,
                    "Purged unavailable externalized vector markers from dense segment"
                );
            }
            return Ok(Some(purged));
        }

        Ok(None)
    }

    pub fn plan_next_unavailable_externalized_marker_segment(
        &self,
        txn: &RoTxn,
        logical_name: &str,
        limit_per_segment: usize,
    ) -> Result<Option<ExternalizedMarkerRepairPlan>, VectorError> {
        if limit_per_segment == 0 {
            return Ok(None);
        }

        let segments: Vec<(String, DenseVectorSegmentRole)> = {
            let dense_spaces = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let Some(space) = dense_spaces.get(logical_name) else {
                return Ok(None);
            };
            space
                .segments
                .iter()
                .map(|segment| (segment.physical_name.clone(), segment.role))
                .collect()
        };

        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        for (physical_name, role) in segments {
            let Some(core) = cores.get(&physical_name) else {
                continue;
            };
            if !core.externalized_marker_repair_needed() {
                continue;
            }

            let ids = {
                let rd = core.backend.read_borrowed(txn);
                core.unavailable_externalized_marker_ids(&rd, limit_per_segment)?
            };
            if ids.is_empty() {
                core.clear_externalized_marker_repair_needed();
            }
            return Ok(Some(ExternalizedMarkerRepairPlan {
                physical_name,
                role,
                ids,
                limit_per_segment,
            }));
        }

        Ok(None)
    }

    pub fn apply_externalized_marker_repair_plan(
        &self,
        txn: &mut RwTxn,
        logical_name: &str,
        plan: &ExternalizedMarkerRepairPlan,
    ) -> Result<usize, VectorError> {
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let Some(core) = cores.get(&plan.physical_name) else {
            return Ok(0);
        };

        let purged = core.purge_externalized_marker_ids(txn, &plan.ids, plan.limit_per_segment)?;
        if purged > 0 {
            metrics::counter!(
                "helix_externalized_marker_segment_repair_total",
                "collection" => self.collection_name.clone(),
                "vector" => logical_name.to_string(),
                "role" => format!("{:?}", plan.role)
            )
            .increment(purged as u64);
            tracing::warn!(
                collection = %self.collection_name,
                vector = logical_name,
                segment = %plan.physical_name,
                role = ?plan.role,
                purged,
                limit_per_segment = plan.limit_per_segment,
                "Purged unavailable externalized vector markers from dense segment"
            );
        }
        Ok(purged)
    }

    pub fn has_externalized_marker_repair_debt(
        &self,
        logical_name: &str,
    ) -> Result<bool, VectorError> {
        let segments: Vec<String> = {
            let dense_spaces = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let Some(space) = dense_spaces.get(logical_name) else {
                return Ok(false);
            };
            space
                .segments
                .iter()
                .map(|segment| segment.physical_name.clone())
                .collect()
        };

        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        Ok(segments.iter().any(|physical_name| {
            cores
                .get(physical_name)
                .map(|core| core.externalized_marker_repair_needed())
                .unwrap_or(false)
        }))
    }

    /// Mutate `space.segments` to drop the segments named in `empties`.
    /// Caller has already located them via
    /// `cleanup_empty_dense_segments_locate` (in a read txn) and is
    /// responsible for persisting the metadata blob via
    /// `set_dense_vector_spaces_metadata` in the same exclusive write txn.
    /// The `cores` entries stay so `SegmentReaper` can drop them
    /// asynchronously.
    pub fn apply_dense_segment_removal(
        &self,
        logical_name: &str,
        empties: &[String],
    ) -> Result<bool, VectorError> {
        if empties.is_empty() {
            return Ok(false);
        }
        let mut dense_spaces = self
            .dense_spaces
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let Some(space) = dense_spaces.get_mut(logical_name) else {
            return Ok(false);
        };
        let before = space.segments.len();
        space.segments.retain(|segment| {
            !empties
                .iter()
                .any(|physical_name| physical_name == &segment.physical_name)
        });
        Ok(space.segments.len() != before)
    }

    pub fn debug_fields(
        &self,
        txn: &RoTxn,
        logical_name: &str,
        id: u128,
    ) -> Result<HashMap<String, Value>, VectorError> {
        let dense_spaces = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let space = dense_spaces.get(logical_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", logical_name))
        })?;
        let segment_names: Vec<String> = space
            .segments
            .iter()
            .map(|segment| segment.physical_name.clone())
            .collect();
        drop(dense_spaces);

        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        for segment_name in &segment_names {
            let core = cores.get(segment_name).ok_or_else(|| {
                VectorError::VectorCoreError(format!("Segment core '{}' missing", segment_name))
            })?;
            let rd = core.backend.read_borrowed(txn);
            if core.contains_id(&rd, id)? {
                return Ok(core.debug_fields_internal(&rd, id));
            }
        }
        Err(VectorError::VectorNotFound(id.to_string()))
    }

    /// Delete a vector from all named dense indexes by point id.
    /// Not-found is silently skipped (point may not exist in every segment),
    /// but LMDB/IO errors are propagated.
    ///
    /// Fast path: probe each core's `contains_id` (single LMDB get on
    /// `vectors_db`) before invoking the full `core.delete_vector` walk
    /// (two `prefix_iter` scans + neighbor-edge cleanup). On collections
    /// with N segments this turns the per-id delete from O(N) heavy
    /// scans into O(N) cheap probes + 1 heavy scan for the segment that
    /// actually owns the id. Phase 0 instrumentation showed `delete_vector`
    /// at p99 ≈ 2.2 s per call as the dominant cost in
    /// `apply_upsert_points_in_txn` for re-upsert workloads.
    pub fn delete_vector(&self, txn: &mut heed3::RwTxn, id: u128) -> Result<(), VectorError> {
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        for core in cores.values() {
            // Cheap O(log n) probe — most segments don't have any given id.
            let owns = {
                let rd = core.backend.read_borrowed(&*txn);
                core.contains_id(&rd, id)?
            };
            if !owns {
                continue;
            }
            match core.delete_vector(txn, id) {
                Ok(()) => {}
                Err(VectorError::VectorNotFound(_)) => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Batched delete across every owning core. Each core sees only the
    /// subset of ids it actually contains (cheap `contains_id` probe per
    /// core × per id), then runs one `delete_vectors_batch` so the
    /// per-(neighbor,level) reverse-edge updates fold across the chunk.
    /// Used by `apply_upsert_points_in_txn` on the re-upsert path to
    /// replace N per-point `delete_vector` calls with one batched delete per
    /// active segment. Retired cores stay open while the segment reaper drains
    /// them, but they are no longer search-visible and must not inflate delete
    /// fanout.
    pub fn delete_vectors_batch(
        &self,
        txn: &mut heed3::RwTxn,
        ids: &[u128],
    ) -> Result<(), VectorError> {
        if ids.is_empty() {
            return Ok(());
        }
        let active_segment_names: HashSet<String> = {
            let dense_spaces = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            dense_spaces
                .values()
                .flat_map(|space| {
                    space
                        .segments
                        .iter()
                        .map(|segment| segment.physical_name.clone())
                })
                .collect()
        };
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        if crate::telemetry::hot_path_metrics_enabled() {
            metrics::histogram!("helix_delete_dense_cores_active")
                .record(active_segment_names.len() as f64);
            metrics::histogram!("helix_delete_dense_cores_open").record(cores.len() as f64);
        }
        for segment_name in active_segment_names {
            let Some(core) = cores.get(&segment_name) else {
                continue;
            };
            let mut owned: Vec<u128> = Vec::with_capacity(ids.len());
            {
                let rd = core.backend.read_borrowed(&*txn);
                for &id in ids {
                    if core.contains_id(&rd, id)? {
                        owned.push(id);
                    }
                }
            }
            if !owned.is_empty() {
                core.delete_vectors_batch(txn, &owned)?;
            }
        }
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
        let active_segment_names: HashSet<String> = {
            let dense_spaces = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            dense_spaces
                .values()
                .flat_map(|space| {
                    space
                        .segments
                        .iter()
                        .map(|segment| segment.physical_name.clone())
                })
                .collect()
        };
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        for segment_name in active_segment_names {
            let Some(core) = cores.get(&segment_name) else {
                continue;
            };
            let mut owned = Vec::with_capacity(ids.len());
            for &id in ids {
                if core.contains_id_be(w, id)? {
                    owned.push(id);
                }
            }
            if !owned.is_empty() {
                core.delete_vectors_batch_be(w, &owned)?;
            }
        }
        Ok(())
    }

    /// Plan dense deletes with a read transaction so the expensive
    /// active-segment ownership probe does not hold the LMDB writer.
    pub fn plan_delete_vectors_batch(
        &self,
        txn: &RoTxn,
        ids: &[u128],
    ) -> Result<DenseDeletePlan, VectorError> {
        if ids.is_empty() {
            return Ok(DenseDeletePlan::default());
        }

        let started = Instant::now();
        let active_segments: HashSet<String> = {
            let dense_spaces = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            dense_spaces
                .values()
                .flat_map(|space| {
                    space
                        .segments
                        .iter()
                        .map(|segment| segment.physical_name.clone())
                })
                .collect()
        };

        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let mut by_segment = Vec::new();
        for segment_name in &active_segments {
            let Some(core) = cores.get(segment_name) else {
                continue;
            };
            let mut owned = Vec::new();
            {
                let rd = core.backend.read_borrowed(&*txn);
                for &id in ids {
                    if core.contains_id(&rd, id)? {
                        owned.push(id);
                    }
                }
            }
            if !owned.is_empty() {
                by_segment.push((segment_name.clone(), owned));
            }
        }

        if crate::telemetry::hot_path_metrics_enabled() {
            metrics::histogram!("helix_delete_dense_plan_ms")
                .record(started.elapsed().as_secs_f64() * 1000.0);
            metrics::histogram!("helix_delete_dense_plan_segments")
                .record(active_segments.len() as f64);
            metrics::histogram!("helix_delete_dense_plan_ids").record(ids.len() as f64);
        }

        Ok(DenseDeletePlan {
            ids: ids.to_vec(),
            active_segments,
            by_segment,
        })
    }

    /// Apply a precomputed dense delete plan. If the active segment layout
    /// changed after planning, fall back to the in-writer scan for correctness.
    pub fn delete_vectors_batch_with_plan(
        &self,
        txn: &mut heed3::RwTxn,
        plan: &DenseDeletePlan,
    ) -> Result<(), VectorError> {
        if plan.is_empty() {
            return Ok(());
        }

        let active_segments: HashSet<String> = {
            let dense_spaces = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            dense_spaces
                .values()
                .flat_map(|space| {
                    space
                        .segments
                        .iter()
                        .map(|segment| segment.physical_name.clone())
                })
                .collect()
        };

        if active_segments != plan.active_segments {
            metrics::counter!("helix_delete_dense_plan_stale_total").increment(1);
            return self.delete_vectors_batch(txn, &plan.ids);
        }

        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        for (segment_name, owned) in &plan.by_segment {
            let Some(core) = cores.get(segment_name) else {
                metrics::counter!("helix_delete_dense_plan_stale_total").increment(1);
                drop(cores);
                return self.delete_vectors_batch(txn, &plan.ids);
            };
            core.delete_vectors_batch(txn, owned)?;
        }
        Ok(())
    }

    /// Plan dense deletes with a backend read handle so the expensive
    /// active-segment ownership probe does not hold the LSM writer. Mirrors
    /// `plan_delete_vectors_batch`, but probes ownership through `contains_id`
    /// against an `AnyRead` handle instead of an LMDB `RoTxn`.
    pub fn plan_delete_vectors_batch_be(
        &self,
        r: &AnyRead<'_>,
        ids: &[u128],
    ) -> Result<DenseDeletePlan, VectorError> {
        if ids.is_empty() {
            return Ok(DenseDeletePlan::default());
        }

        let started = Instant::now();
        let active_segments: HashSet<String> = {
            let dense_spaces = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            dense_spaces
                .values()
                .flat_map(|space| {
                    space
                        .segments
                        .iter()
                        .map(|segment| segment.physical_name.clone())
                })
                .collect()
        };

        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let mut by_segment = Vec::new();
        for segment_name in &active_segments {
            let Some(core) = cores.get(segment_name) else {
                continue;
            };
            let mut owned = Vec::new();
            for &id in ids {
                if core.contains_id(r, id)? {
                    owned.push(id);
                }
            }
            if !owned.is_empty() {
                by_segment.push((segment_name.clone(), owned));
            }
        }

        if crate::telemetry::hot_path_metrics_enabled() {
            metrics::histogram!("helix_delete_dense_plan_ms")
                .record(started.elapsed().as_secs_f64() * 1000.0);
            metrics::histogram!("helix_delete_dense_plan_segments")
                .record(active_segments.len() as f64);
            metrics::histogram!("helix_delete_dense_plan_ids").record(ids.len() as f64);
        }

        Ok(DenseDeletePlan {
            ids: ids.to_vec(),
            active_segments,
            by_segment,
        })
    }

    /// Apply a precomputed dense delete plan. If the active segment layout
    /// changed after planning, fall back to the in-writer scan for
    /// correctness. Mirrors `delete_vectors_batch_with_plan`'s staleness
    /// guard for the backend-agnostic (LSM) path.
    ///
    /// Returns the per-segment ids staged as delete-tombstones in `w`. The
    /// caller MUST pass them to `apply_delete_tombstones` after the batch
    /// commits — applying them to the in-memory sets before the commit is
    /// durable would suppress live vectors from search on commit failure.
    pub fn delete_vectors_batch_with_plan_be(
        &self,
        w: &mut AnyWrite<'_>,
        plan: &DenseDeletePlan,
    ) -> Result<StagedDeleteTombstones, VectorError> {
        if plan.is_empty() {
            return Ok(StagedDeleteTombstones::default());
        }

        let active_segments: HashSet<String> = {
            let dense_spaces = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            dense_spaces
                .values()
                .flat_map(|space| {
                    space
                        .segments
                        .iter()
                        .map(|segment| segment.physical_name.clone())
                })
                .collect()
        };

        if active_segments != plan.active_segments {
            metrics::counter!("helix_delete_dense_plan_stale_total").increment(1);
            self.delete_vectors_batch_be(w, &plan.ids)?;
            return Ok(StagedDeleteTombstones::default());
        }

        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let tombstones_enabled = lsm_delete_tombstones_enabled();
        let mut staged = StagedDeleteTombstones::default();
        for (segment_name, owned) in &plan.by_segment {
            let Some(core) = cores.get(segment_name) else {
                metrics::counter!("helix_delete_dense_plan_stale_total").increment(1);
                drop(cores);
                self.delete_vectors_batch_be(w, &plan.ids)?;
                return Ok(StagedDeleteTombstones::default());
            };
            if tombstones_enabled {
                let ids = core.tombstone_delete_batch_be(w, owned)?;
                if !ids.is_empty() {
                    staged.by_segment.push((segment_name.clone(), ids));
                }
            } else {
                core.delete_vectors_batch_be(w, owned)?;
            }
        }
        Ok(staged)
    }

    /// Post-commit companion of `delete_vectors_batch_with_plan_be`: applies
    /// the staged tombstones to each segment core's in-memory set and records
    /// the metric. Call only after the staging write batch committed.
    pub fn apply_delete_tombstones(&self, staged: &StagedDeleteTombstones) {
        if staged.is_empty() {
            return;
        }
        let Ok(cores) = self.cores.read() else {
            return;
        };
        let mut applied = 0u64;
        for (segment_name, ids) in &staged.by_segment {
            if let Some(core) = cores.get(segment_name) {
                core.apply_delete_tombstones(ids);
                applied += ids.len() as u64;
            }
        }
        if applied > 0 {
            metrics::counter!(
                "helix_lsm_delete_tombstones_recorded_total",
                "collection" => self.collection_name.clone(),
            )
            .increment(applied);
        }
    }

    pub fn has_vector(&self, name: &str) -> bool {
        self.configs
            .read()
            .map(|configs| configs.contains_key(name))
            .unwrap_or(false)
    }

    // ── Sparse vector management ──

    pub fn create_sparse_index(
        &self,
        env: &Env<WithTls>,
        txn: &mut RwTxn,
        name: &str,
        config: SparseVectorConfig,
    ) -> Result<(), VectorError> {
        let core = SparseVectorCore::new(env, txn, name, config.clone(), self.current_backend()?)?;

        self.sparse_configs
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?
            .insert(name.to_string(), config);

        self.sparse_cores
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?
            .insert(name.to_string(), core);

        Ok(())
    }

    pub fn load_sparse_index(
        &self,
        env: &Env<WithTls>,
        txn: &mut RwTxn,
        name: &str,
        config: SparseVectorConfig,
    ) -> Result<(), VectorError> {
        self.create_sparse_index(env, txn, name, config)
    }

    pub fn load_sparse_index_lsm(
        &self,
        name: &str,
        config: SparseVectorConfig,
    ) -> Result<(), VectorError> {
        let core = SparseVectorCore::new_lsm(name, config.clone(), self.current_backend()?)?;

        self.sparse_configs
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?
            .insert(name.to_string(), config);

        self.sparse_cores
            .write()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?
            .insert(name.to_string(), core);

        Ok(())
    }

    pub fn create_sparse_index_lsm(
        &self,
        name: &str,
        config: SparseVectorConfig,
    ) -> Result<(), VectorError> {
        self.load_sparse_index_lsm(name, config)
    }

    pub fn with_sparse_core<F, R>(&self, name: &str, f: F) -> Result<R, VectorError>
    where
        F: FnOnce(&SparseVectorCore) -> Result<R, VectorError>,
    {
        let cores = self
            .sparse_cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

        let core = cores.get(name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Sparse vector '{}' not found", name))
        })?;

        f(core)
    }

    /// Term-major batched sparse upsert. Mirrors `delete_vectors_batch`
    /// on the dense side: the caller buckets the chunk's points by
    /// sparse-vector name and we issue ONE call per name with all docs
    /// belonging to that name, letting the core sort term-major and
    /// keep the LMDB DUPSORT cursor warm. Returns the number of fresh
    /// inserts processed (diff/unchanged docs aren't counted).
    pub fn upsert_sparse_batch(
        &self,
        txn: &mut heed3::RwTxn,
        name: &str,
        items: &[(u128, &SparseVector)],
    ) -> Result<usize, VectorError> {
        if items.is_empty() {
            return Ok(0);
        }
        let cores = self
            .sparse_cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let core = cores.get(name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Sparse vector '{}' not found", name))
        })?;
        core.upsert_batch(txn, items)
    }

    pub fn upsert_sparse_batch_be(
        &self,
        w: &mut AnyWrite<'_>,
        name: &str,
        items: &[(u128, &SparseVector)],
    ) -> Result<usize, VectorError> {
        if items.is_empty() {
            return Ok(0);
        }
        let cores = self
            .sparse_cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let core = cores.get(name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Sparse vector '{}' not found", name))
        })?;
        core.upsert_batch_be(w, items)
    }

    pub fn delete_sparse_vectors(
        &self,
        txn: &mut heed3::RwTxn,
        id: u128,
    ) -> Result<(), VectorError> {
        let cores = self
            .sparse_cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        for core in cores.values() {
            core.delete(txn, id)?;
        }
        Ok(())
    }

    pub fn delete_sparse_vectors_be(
        &self,
        w: &mut AnyWrite<'_>,
        id: u128,
    ) -> Result<(), VectorError> {
        let cores = self
            .sparse_cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        for core in cores.values() {
            core.delete_be(w, id)?;
        }
        Ok(())
    }

    pub fn has_sparse_vector(&self, name: &str) -> bool {
        self.sparse_cores
            .read()
            .map(|c| c.contains_key(name))
            .unwrap_or(false)
    }

    pub fn list_sparse_vectors(&self) -> HashMap<String, SparseVectorConfig> {
        self.sparse_configs
            .read()
            .map(|c| c.clone())
            .unwrap_or_default()
    }

    pub fn get_sparse_config(&self, name: &str) -> Option<SparseVectorConfig> {
        self.sparse_configs
            .read()
            .ok()
            .and_then(|configs| configs.get(name).cloned())
    }

    /// Drain pending df + max metadata for every sparse core into LMDB
    /// in `txn`. Called from `apply_upsert_points_in_txn` at the end of
    /// each chunk so the metadata writes that used to happen per-point
    /// happen once per chunk instead. Returns the total number of
    /// metadata entries flushed across all sparse cores.
    pub fn flush_sparse_pending_metadata(&self, txn: &mut RwTxn) -> Result<usize, VectorError> {
        Ok(self
            .flush_sparse_pending_metadata_budgeted(txn, None, None)?
            .flushed)
    }

    /// Budgeted variant of `flush_sparse_pending_metadata`.
    ///
    /// `budget` is shared across sparse cores in this collection for one
    /// chunk-end write transaction. HashMap iteration order is sufficient
    /// for v1 because df deltas are commutative and max updates are
    /// order-independent; future durable coalescing can add oldest-first
    /// fairness if staleness bounds need to be tighter.
    pub fn flush_sparse_pending_metadata_budgeted(
        &self,
        txn: &mut RwTxn,
        budget: Option<usize>,
        max_duration: Option<std::time::Duration>,
    ) -> Result<SparseMetadataFlushStats, VectorError> {
        let cores = self
            .sparse_cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let started = std::time::Instant::now();
        let mut aggregate = SparseMetadataFlushStats::default();
        let mut remaining_budget = budget.unwrap_or(usize::MAX);

        for core in cores.values() {
            let elapsed = started.elapsed();
            if let Some(limit) = max_duration {
                if elapsed >= limit {
                    aggregate.pending_after = cores
                        .values()
                        .map(|remaining_core| remaining_core.pending_metadata_count())
                        .sum();
                    aggregate.time_exhausted = aggregate.pending_after > 0;
                    break;
                }
            }

            let core_budget = budget.map(|_| remaining_budget);
            let core_duration = max_duration.map(|limit| limit.saturating_sub(elapsed));
            let stats = core.flush_metadata_pending_budgeted(txn, core_budget, core_duration)?;
            aggregate.pending_before += stats.pending_before;
            aggregate.processed += stats.processed;
            aggregate.flushed += stats.flushed;
            aggregate.pending_after += stats.pending_after;
            aggregate.budget_exhausted |= stats.budget_exhausted;
            aggregate.time_exhausted |= stats.time_exhausted;

            if budget.is_some() {
                remaining_budget = remaining_budget.saturating_sub(stats.processed);
            }

            if (budget.is_some() && remaining_budget == 0) || stats.time_exhausted {
                aggregate.pending_after = cores
                    .values()
                    .map(|remaining_core| remaining_core.pending_metadata_count())
                    .sum();
                aggregate.budget_exhausted =
                    budget.is_some() && remaining_budget == 0 && aggregate.pending_after > 0;
                aggregate.time_exhausted |= stats.time_exhausted && aggregate.pending_after > 0;
                break;
            }
        }

        Ok(aggregate)
    }

    /// Drain pending df/max for every sparse core on the LSM backend. Each core
    /// flushes in its own self-contained, commit-safe batch (see
    /// [`SparseVectorCore::flush_metadata_pending_budgeted_be`]), so this takes
    /// no caller batch and must be invoked AFTER the data batch has committed.
    pub fn flush_sparse_pending_metadata_budgeted_be(
        &self,
        budget: Option<usize>,
        max_duration: Option<std::time::Duration>,
    ) -> Result<SparseMetadataFlushStats, VectorError> {
        let cores = self
            .sparse_cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let started = std::time::Instant::now();
        let mut aggregate = SparseMetadataFlushStats::default();
        let mut remaining_budget = budget.unwrap_or(usize::MAX);

        for core in cores.values() {
            let elapsed = started.elapsed();
            if let Some(limit) = max_duration {
                if elapsed >= limit {
                    aggregate.pending_after = cores
                        .values()
                        .map(|remaining_core| remaining_core.pending_metadata_count())
                        .sum();
                    aggregate.time_exhausted = aggregate.pending_after > 0;
                    break;
                }
            }

            let core_budget = budget.map(|_| remaining_budget);
            let core_duration = max_duration.map(|limit| limit.saturating_sub(elapsed));
            let stats = core.flush_metadata_pending_budgeted_be(core_budget, core_duration)?;
            aggregate.pending_before += stats.pending_before;
            aggregate.processed += stats.processed;
            aggregate.flushed += stats.flushed;
            aggregate.pending_after += stats.pending_after;
            aggregate.budget_exhausted |= stats.budget_exhausted;
            aggregate.time_exhausted |= stats.time_exhausted;

            if budget.is_some() {
                remaining_budget = remaining_budget.saturating_sub(stats.processed);
            }
            if (budget.is_some() && remaining_budget == 0) || stats.time_exhausted {
                aggregate.pending_after = cores
                    .values()
                    .map(|remaining_core| remaining_core.pending_metadata_count())
                    .sum();
                aggregate.budget_exhausted =
                    budget.is_some() && remaining_budget == 0 && aggregate.pending_after > 0;
                aggregate.time_exhausted |= stats.time_exhausted && aggregate.pending_after > 0;
                break;
            }
        }
        Ok(aggregate)
    }

    /// Approximate count of pending sparse metadata entries across all
    /// cores. Useful as a backpressure signal — if this stays high,
    /// `flush_sparse_pending_metadata` is being called too rarely.
    pub fn pending_sparse_metadata_count(&self) -> usize {
        self.sparse_cores
            .read()
            .ok()
            .map(|cores| cores.values().map(|c| c.pending_metadata_count()).sum())
            .unwrap_or(0)
    }
}

/// Reference to a named vector core (used for type-safe access).
pub struct VectorCoreRef<'a> {
    pub name: String,
    _manager: &'a NamedVectorManager,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helix_engine::graph_core::config::Config;
    use crate::helix_engine::storage_core::storage_core::HelixGraphStorage;
    use crate::helix_engine::vector_core::spindle::SpindleMode;
    use rand::SeedableRng;
    use tempfile::TempDir;

    static TOMBSTONE_ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    struct EnvVarGuard {
        name: &'static str,
        previous: Option<String>,
    }

    impl EnvVarGuard {
        fn set(name: &'static str, value: &str) -> Self {
            let previous = std::env::var(name).ok();
            std::env::set_var(name, value);
            Self { name, previous }
        }

        fn unset(name: &'static str) -> Self {
            let previous = std::env::var(name).ok();
            std::env::remove_var(name);
            Self { name, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => std::env::set_var(self.name, value),
                None => std::env::remove_var(self.name),
            }
        }
    }

    fn setup_storage() -> (HelixGraphStorage, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let storage =
            HelixGraphStorage::new(temp_dir.path().to_str().unwrap(), Config::new(8, 32, 64, 1))
                .unwrap();
        (storage, temp_dir)
    }

    #[test]
    fn dense_index_mode_defaults_to_hnsw() {
        assert_eq!(dense_index_mode_from_raw(None), DenseIndexMode::Hnsw);
        assert_eq!(
            dense_index_mode_from_raw(Some("hnsw")),
            DenseIndexMode::Hnsw
        );
        assert_eq!(dense_index_mode_from_raw(Some("")), DenseIndexMode::Hnsw);
        assert_eq!(
            dense_index_mode_from_raw(Some("garbage")),
            DenseIndexMode::Hnsw
        );
        assert_eq!(dense_index_mode_from_raw(Some("ivf")), DenseIndexMode::Ivf);
        assert_eq!(
            dense_index_mode_from_raw(Some(" IVF ")),
            DenseIndexMode::Ivf
        );
    }

    fn create_dense(storage: &HelixGraphStorage, name: &str) {
        create_dense_with_spindle(storage, name, SpindleConfig::default());
    }

    fn raw_hvec_spindle() -> SpindleConfig {
        SpindleConfig {
            mode: SpindleMode::None,
            keep_original: false,
            ..SpindleConfig::default()
        }
    }

    fn create_dense_with_spindle(storage: &HelixGraphStorage, name: &str, spindle: SpindleConfig) {
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        storage
            .named_vectors
            .create_vector_index(
                storage.lmdb_env().unwrap(),
                &mut txn,
                name,
                NamedVectorConfig {
                    size: 3,
                    distance: DistanceMetric::Cosine,
                    spindle,
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
        txn.commit().unwrap();
    }

    fn create_indexed_segments_for_selection(
        storage: &HelixGraphStorage,
        logical_name: &str,
        count: usize,
    ) {
        let config = storage.named_vectors.get_config(logical_name).unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let mut segments = Vec::with_capacity(count);

        for idx in 1..=count {
            let physical_name = format!("{logical_name}__seg_{idx:06}");
            let core = NamedVectorManager::create_dense_core(
                storage.lmdb_env().unwrap(),
                &mut txn,
                &physical_name,
                &config,
                hnsw.clone(),
                Arc::clone(&storage.backend),
            )
            .unwrap();
            core.insert_flat(&mut txn, &[idx as f32, 0.0, 0.0], Some(idx as u128), None)
                .unwrap();
            storage
                .named_vectors
                .cores
                .write()
                .unwrap()
                .insert(physical_name.clone(), core);
            segments.push(DenseVectorSegmentMetadata {
                physical_name,
                role: DenseVectorSegmentRole::Indexed,
            });
        }

        storage.named_vectors.dense_spaces.write().unwrap().insert(
            logical_name.to_string(),
            DenseVectorSpaceMetadata {
                next_segment_id: count as u64 + 1,
                segments,
            },
        );
        storage
            .set_dense_vector_spaces_metadata(
                &mut txn,
                storage.named_vectors.list_dense_vector_spaces(),
            )
            .unwrap();
        txn.commit().unwrap();
    }

    fn write_header_only_hvec(path: &Path, dim: usize) {
        let mut header = Vec::with_capacity(16);
        header.extend_from_slice(b"HVEC");
        header.extend_from_slice(&(dim as u32).to_le_bytes());
        header.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(path, header).unwrap();
    }

    fn write_hvec_rows(path: &Path, dim: usize, rows: &[&[f32]]) {
        let mut bytes = Vec::with_capacity(16 + rows.len() * dim * 4);
        bytes.extend_from_slice(b"HVEC");
        bytes.extend_from_slice(&(dim as u32).to_le_bytes());
        bytes.extend_from_slice(&(rows.len() as u64).to_le_bytes());
        for row in rows {
            assert_eq!(row.len(), dim);
            for &value in *row {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn inactive_sidecar_cleanup_skips_active_segments() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let dir = storage.lmdb_env().unwrap().path();
        let active = dir.join("dense.hvec");
        if !active.exists() {
            std::fs::write(&active, b"active").unwrap();
        }
        let stale_hvec = dir.join("dense__seg_999999.hvec");
        let stale_hvs8 = dir.join("dense__seg_999998.hvs8");
        std::fs::write(&stale_hvec, b"stale-hvec").unwrap();
        std::fs::write(&stale_hvs8, b"stale-hvs8").unwrap();

        let dry = storage
            .named_vectors
            .cleanup_inactive_sidecar_files(dir, true);
        assert_eq!(dry.removed_files, 2);
        assert_eq!(dry.skipped_active_files, 1);
        assert!(active.exists());
        assert!(stale_hvec.exists());
        assert!(stale_hvs8.exists());

        let cleaned = storage
            .named_vectors
            .cleanup_inactive_sidecar_files(dir, false);
        assert_eq!(cleaned.removed_files, 2);
        assert_eq!(cleaned.error_count, 0);
        assert!(active.exists());
        assert!(!stale_hvec.exists());
        assert!(!stale_hvs8.exists());
    }

    #[test]
    fn quantize_indexed_mmap_sidecars_converts_hvec_once() {
        let (storage, _tmp) = setup_storage();
        create_dense_with_spindle(
            &storage,
            "dense",
            SpindleConfig {
                mode: SpindleMode::None,
                keep_original: false,
                rescore: false,
                oversampling: 1,
                binary_dims: 1,
                turbo_dims: 1,
            },
        );
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            storage
                .named_vectors
                .dense_insert(
                    storage.lmdb_env().unwrap(),
                    &mut txn,
                    "dense",
                    &[1.0, 0.0, 0.0],
                    1,
                    HashMap::new(),
                    hnsw,
                    10,
                )
                .unwrap();
            txn.commit().unwrap();
        }

        {
            let mut spaces = storage.named_vectors.dense_spaces.write().unwrap();
            spaces.get_mut("dense").unwrap().segments[0].role = DenseVectorSegmentRole::Indexed;
        }

        let hvec = storage.lmdb_env().unwrap().path().join("dense.hvec");
        let hvs8 = storage.lmdb_env().unwrap().path().join("dense.hvs8");
        assert!(hvec.exists());
        assert!(!hvs8.exists());

        let stats = storage.named_vectors.quantize_indexed_mmap_sidecars(None);
        assert_eq!(stats.checked_segments, 1);
        assert_eq!(stats.converted_segments, 1);
        assert_eq!(stats.error_count, 0);
        assert!(!hvec.exists());
        assert!(hvs8.exists());

        let second = storage.named_vectors.quantize_indexed_mmap_sidecars(None);
        assert_eq!(second.checked_segments, 1);
        assert_eq!(second.converted_segments, 0);
        assert_eq!(second.skipped_segments, 1);
    }

    #[test]
    fn dense_insert_rolls_over_per_space_threshold() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));

        let job = storage
            .named_vectors
            .dense_insert(
                storage.lmdb_env().unwrap(),
                &mut txn,
                "dense",
                &[1.0, 0.0, 0.0],
                1,
                HashMap::new(),
                hnsw.clone(),
                2,
            )
            .unwrap();
        assert!(job.is_none());

        let job = storage
            .named_vectors
            .dense_insert(
                storage.lmdb_env().unwrap(),
                &mut txn,
                "dense",
                &[0.0, 1.0, 0.0],
                2,
                HashMap::new(),
                hnsw,
                2,
            )
            .unwrap();
        assert_eq!(job.as_deref(), Some("dense"));

        let spaces = storage.named_vectors.list_dense_vector_spaces();
        let space = spaces.get("dense").unwrap();
        assert_eq!(space.segments.len(), 2);
        assert_eq!(
            space
                .segments
                .iter()
                .filter(|segment| segment.role == DenseVectorSegmentRole::Building)
                .count(),
            1
        );
        assert_eq!(
            space
                .segments
                .iter()
                .filter(|segment| segment.role == DenseVectorSegmentRole::Mutable)
                .count(),
            1
        );
    }

    #[test]
    fn dense_batch_fit_detects_layout_boundary_before_insert() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));

        {
            let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
            let mut batch_counts = HashMap::new();
            batch_counts.insert("dense".to_string(), 1);
            assert!(storage
                .named_vectors
                .dense_batch_fits_existing_segments(&txn, &batch_counts, 2)
                .unwrap());
        }

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            storage
                .named_vectors
                .dense_insert(
                    storage.lmdb_env().unwrap(),
                    &mut txn,
                    "dense",
                    &[1.0, 0.0, 0.0],
                    1,
                    HashMap::new(),
                    hnsw,
                    10,
                )
                .unwrap();
            txn.commit().unwrap();
        }

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let mut batch_counts = HashMap::new();
        batch_counts.insert("dense".to_string(), 1);
        assert!(!storage
            .named_vectors
            .dense_batch_fits_existing_segments(&txn, &batch_counts, 2)
            .unwrap());
    }

    #[test]
    fn dense_append_to_mutable_does_not_roll_over_threshold() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));

        for id in 1..=3 {
            let layout_changed = storage
                .named_vectors
                .dense_append_to_mutable(
                    storage.lmdb_env().unwrap(),
                    &mut txn,
                    "dense",
                    &[id as f32, 0.0, 0.0],
                    id,
                    HashMap::new(),
                    hnsw.clone(),
                    2,
                )
                .unwrap();
            assert!(!layout_changed);
        }
        txn.commit().unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let stats = storage
            .named_vectors
            .get_dense_space_stats(&txn, "dense")
            .unwrap();
        assert_eq!(stats.vectors_count, 3);
        assert_eq!(stats.indexed_vectors_count, 0);

        let spaces = storage.named_vectors.list_dense_vector_spaces();
        let space = spaces.get("dense").unwrap();
        assert_eq!(space.segments.len(), 1);
        assert_eq!(space.segments[0].role, DenseVectorSegmentRole::Mutable);
    }

    #[test]
    fn promote_building_segments_only_promotes_flushed_names() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let config = storage.named_vectors.get_config("dense").unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let core_1 = NamedVectorManager::create_dense_core(
            storage.lmdb_env().unwrap(),
            &mut txn,
            "dense__seg_000001",
            &config,
            hnsw.clone(),
            Arc::clone(&storage.backend),
        )
        .unwrap();
        let core_2 = NamedVectorManager::create_dense_core(
            storage.lmdb_env().unwrap(),
            &mut txn,
            "dense__seg_000002",
            &config,
            hnsw,
            Arc::clone(&storage.backend),
        )
        .unwrap();
        {
            let mut cores = storage.named_vectors.cores.write().unwrap();
            cores.insert("dense__seg_000001".to_string(), core_1);
            cores.insert("dense__seg_000002".to_string(), core_2);
        }
        {
            let mut dense_spaces = storage.named_vectors.dense_spaces.write().unwrap();
            dense_spaces.insert(
                "dense".to_string(),
                DenseVectorSpaceMetadata {
                    next_segment_id: 3,
                    segments: vec![
                        DenseVectorSegmentMetadata {
                            physical_name: "dense__seg_000001".to_string(),
                            role: DenseVectorSegmentRole::Building,
                        },
                        DenseVectorSegmentMetadata {
                            physical_name: "dense__seg_000002".to_string(),
                            role: DenseVectorSegmentRole::Building,
                        },
                    ],
                },
            );
        }
        txn.commit().unwrap();

        let mut flushed_names = HashSet::new();
        flushed_names.insert("dense__seg_000001".to_string());

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        assert!(storage
            .named_vectors
            .promote_building_segments_to_indexed(&txn, "dense", &flushed_names)
            .unwrap());

        let spaces = storage.named_vectors.list_dense_vector_spaces();
        let segments = &spaces.get("dense").unwrap().segments;
        assert_eq!(segments[0].role, DenseVectorSegmentRole::Indexed);
        assert_eq!(segments[1].role, DenseVectorSegmentRole::Building);
    }

    #[test]
    fn promote_building_segment_rejects_unavailable_sidecar_marker() {
        let (storage, _tmp) = setup_storage();
        create_dense_with_spindle(&storage, "dense", raw_hvec_spindle());
        let config = storage.named_vectors.get_config("dense").unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let poisoned_name = "dense__seg_000001".to_string();

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            let core = NamedVectorManager::create_dense_core(
                storage.lmdb_env().unwrap(),
                &mut txn,
                &poisoned_name,
                &config,
                hnsw.clone(),
                Arc::clone(&storage.backend),
            )
            .unwrap();
            core.insert_flat(&mut txn, &[1.0, 0.0, 0.0], Some(42), None)
                .unwrap();
            txn.commit().unwrap();
        }
        write_header_only_hvec(
            &storage
                .lmdb_env()
                .unwrap()
                .path()
                .join("dense__seg_000001.hvec"),
            3,
        );

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            let reopened_core = NamedVectorManager::create_dense_core(
                storage.lmdb_env().unwrap(),
                &mut txn,
                &poisoned_name,
                &config,
                hnsw,
                Arc::clone(&storage.backend),
            )
            .unwrap();
            assert_eq!(
                reopened_core
                    .unavailable_externalized_marker_ids(
                        &reopened_core.backend.read_borrowed(&txn),
                        16,
                    )
                    .unwrap()
                    .len(),
                1,
                "test fixture must contain one marker beyond the header-only sidecar"
            );
            storage
                .named_vectors
                .cores
                .write()
                .unwrap()
                .insert(poisoned_name.clone(), reopened_core);
            storage.named_vectors.dense_spaces.write().unwrap().insert(
                "dense".to_string(),
                DenseVectorSpaceMetadata {
                    next_segment_id: 2,
                    segments: vec![DenseVectorSegmentMetadata {
                        physical_name: poisoned_name.clone(),
                        role: DenseVectorSegmentRole::Building,
                    }],
                },
            );
            txn.commit().unwrap();
        }

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let flushed_names = HashSet::from([poisoned_name]);
        let err = storage
            .named_vectors
            .promote_building_segments_to_indexed(&txn, "dense", &flushed_names)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Refusing to publish dense segment"));
    }

    #[test]
    fn open_rehydrates_indexed_segment_without_sidecar_marker_repair() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = HelixGraphStorage::new(&path, Config::new(8, 32, 64, 1)).unwrap();
        create_dense_with_spindle(&storage, "dense", raw_hvec_spindle());
        let config = storage.named_vectors.get_config("dense").unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let poisoned_name = "dense__seg_000001".to_string();

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            let core = NamedVectorManager::create_dense_core(
                storage.lmdb_env().unwrap(),
                &mut txn,
                &poisoned_name,
                &config,
                hnsw,
                Arc::clone(&storage.backend),
            )
            .unwrap();
            core.insert_flat(&mut txn, &[1.0, 0.0, 0.0], Some(42), None)
                .unwrap();
            storage
                .named_vectors
                .cores
                .write()
                .unwrap()
                .insert(poisoned_name.clone(), core);
            storage.named_vectors.dense_spaces.write().unwrap().insert(
                "dense".to_string(),
                DenseVectorSpaceMetadata {
                    next_segment_id: 2,
                    segments: vec![DenseVectorSegmentMetadata {
                        physical_name: poisoned_name.clone(),
                        role: DenseVectorSegmentRole::Indexed,
                    }],
                },
            );
            storage
                .set_dense_vector_spaces_metadata(
                    &mut txn,
                    storage.named_vectors.list_dense_vector_spaces(),
                )
                .unwrap();
            txn.commit().unwrap();
        }
        write_header_only_hvec(
            &storage
                .lmdb_env()
                .unwrap()
                .path()
                .join("dense__seg_000001.hvec"),
            3,
        );
        drop(storage);

        let reopened = HelixGraphStorage::new(&path, Config::new(8, 32, 64, 1)).unwrap();
        let spaces = reopened.named_vectors.list_dense_vector_spaces();
        let dense = spaces.get("dense").unwrap();
        assert!(
            dense
                .segments
                .iter()
                .any(|segment| segment.physical_name == poisoned_name),
            "collection open must not scan and deactivate dense segments"
        );

        let txn = reopened.lmdb_env().unwrap().read_txn().unwrap();
        let metadata = reopened.get_metadata(&txn).unwrap();
        assert!(
            metadata
                .dense_vector_spaces
                .get("dense")
                .unwrap()
                .segments
                .iter()
                .any(|segment| segment.physical_name == poisoned_name),
            "collection open must not rewrite dense metadata for marker repair"
        );
        let cores = reopened.named_vectors.cores.read().unwrap();
        let core = cores.get(&poisoned_name).unwrap();
        assert!(core.externalized_marker_repair_needed());
        assert_eq!(
            core.unavailable_externalized_marker_ids(&core.backend.read_borrowed(&txn), 16)
                .unwrap()
                .len(),
            1,
            "marker repair should be deferred out of the collection-open path"
        );
    }

    #[test]
    fn open_rehydrates_partial_sidecar_markers_without_purging() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = HelixGraphStorage::new(&path, Config::new(8, 32, 64, 1)).unwrap();
        create_dense_with_spindle(&storage, "dense", raw_hvec_spindle());
        let config = storage.named_vectors.get_config("dense").unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let poisoned_name = "dense__seg_000001".to_string();

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            let core = NamedVectorManager::create_dense_core(
                storage.lmdb_env().unwrap(),
                &mut txn,
                &poisoned_name,
                &config,
                hnsw,
                Arc::clone(&storage.backend),
            )
            .unwrap();
            core.insert_flat(&mut txn, &[1.0, 0.0, 0.0], Some(42), None)
                .unwrap();
            core.insert_flat(&mut txn, &[0.0, 1.0, 0.0], Some(43), None)
                .unwrap();
            storage
                .named_vectors
                .cores
                .write()
                .unwrap()
                .insert(poisoned_name.clone(), core);
            storage.named_vectors.dense_spaces.write().unwrap().insert(
                "dense".to_string(),
                DenseVectorSpaceMetadata {
                    next_segment_id: 2,
                    segments: vec![DenseVectorSegmentMetadata {
                        physical_name: poisoned_name.clone(),
                        role: DenseVectorSegmentRole::Indexed,
                    }],
                },
            );
            storage
                .set_dense_vector_spaces_metadata(
                    &mut txn,
                    storage.named_vectors.list_dense_vector_spaces(),
                )
                .unwrap();
            txn.commit().unwrap();
        }

        write_hvec_rows(
            &storage
                .lmdb_env()
                .unwrap()
                .path()
                .join("dense__seg_000001.hvec"),
            3,
            &[&[1.0, 0.0, 0.0]],
        );
        drop(storage);

        let reopened = HelixGraphStorage::new(&path, Config::new(8, 32, 64, 1)).unwrap();
        let spaces = reopened.named_vectors.list_dense_vector_spaces();
        let dense = spaces.get("dense").unwrap();
        assert!(
            dense
                .segments
                .iter()
                .any(|segment| segment.physical_name == poisoned_name),
            "segment with partial sidecar damage should stay active on open"
        );

        let txn = reopened.lmdb_env().unwrap().read_txn().unwrap();
        let cores = reopened.named_vectors.cores.read().unwrap();
        let core = cores.get(&poisoned_name).unwrap();
        assert!(core.externalized_marker_repair_needed());
        assert_eq!(
            core.unavailable_externalized_marker_ids(&core.backend.read_borrowed(&txn), 16)
                .unwrap()
                .len(),
            1,
            "marker purge should be deferred out of the collection-open path"
        );
        assert_eq!(
            core.level_zero_count(&core.backend.read_borrowed(&txn))
                .unwrap(),
            2,
            "collection open must not rewrite vector rows while rehydrating handles"
        );
        drop(cores);
        drop(txn);

        let mut txn = reopened.lmdb_env().unwrap().write_txn().unwrap();
        assert_eq!(
            reopened
                .named_vectors
                .repair_unavailable_externalized_markers(&mut txn, "dense", 16)
                .unwrap(),
            1,
            "bounded deferred repair should purge the damaged marker after open"
        );
        txn.commit().unwrap();

        let txn = reopened.lmdb_env().unwrap().read_txn().unwrap();
        let cores = reopened.named_vectors.cores.read().unwrap();
        let core = cores.get(&poisoned_name).unwrap();
        assert_eq!(
            core.unavailable_externalized_marker_ids(&core.backend.read_borrowed(&txn), 16)
                .unwrap()
                .len(),
            0,
            "deferred marker repair should clear damaged marker rows"
        );
        assert_eq!(
            core.level_zero_count(&core.backend.read_borrowed(&txn))
                .unwrap(),
            1,
            "the valid externalized row should remain after deferred repair"
        );
    }

    #[test]
    fn externalized_marker_repair_next_attempts_one_segment_per_txn() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = HelixGraphStorage::new(&path, Config::new(8, 32, 64, 1)).unwrap();
        create_dense_with_spindle(&storage, "dense", raw_hvec_spindle());
        let config = storage.named_vectors.get_config("dense").unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let first_name = "dense__seg_000001".to_string();
        let second_name = "dense__seg_000002".to_string();

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            for (name, id) in [(&first_name, 42), (&second_name, 43)] {
                let core = NamedVectorManager::create_dense_core(
                    storage.lmdb_env().unwrap(),
                    &mut txn,
                    name,
                    &config,
                    hnsw.clone(),
                    Arc::clone(&storage.backend),
                )
                .unwrap();
                core.insert_flat(&mut txn, &[1.0, 0.0, 0.0], Some(id), None)
                    .unwrap();
                storage
                    .named_vectors
                    .cores
                    .write()
                    .unwrap()
                    .insert(name.clone(), core);
            }
            storage.named_vectors.dense_spaces.write().unwrap().insert(
                "dense".to_string(),
                DenseVectorSpaceMetadata {
                    next_segment_id: 3,
                    segments: vec![
                        DenseVectorSegmentMetadata {
                            physical_name: first_name.clone(),
                            role: DenseVectorSegmentRole::Indexed,
                        },
                        DenseVectorSegmentMetadata {
                            physical_name: second_name.clone(),
                            role: DenseVectorSegmentRole::Indexed,
                        },
                    ],
                },
            );
            storage
                .set_dense_vector_spaces_metadata(
                    &mut txn,
                    storage.named_vectors.list_dense_vector_spaces(),
                )
                .unwrap();
            txn.commit().unwrap();
        }
        write_header_only_hvec(
            &storage
                .lmdb_env()
                .unwrap()
                .path()
                .join("dense__seg_000001.hvec"),
            3,
        );
        write_header_only_hvec(
            &storage
                .lmdb_env()
                .unwrap()
                .path()
                .join("dense__seg_000002.hvec"),
            3,
        );
        drop(storage);

        let reopened = HelixGraphStorage::new(&path, Config::new(8, 32, 64, 1)).unwrap();
        let mut txn = reopened.lmdb_env().unwrap().write_txn().unwrap();
        assert_eq!(
            reopened
                .named_vectors
                .repair_next_unavailable_externalized_marker_segment(&mut txn, "dense", 16)
                .unwrap(),
            Some(1)
        );
        txn.commit().unwrap();

        let txn = reopened.lmdb_env().unwrap().read_txn().unwrap();
        let cores = reopened.named_vectors.cores.read().unwrap();
        assert_eq!(
            {
                let core = cores.get(&first_name).unwrap();
                core.unavailable_externalized_marker_ids(&core.backend.read_borrowed(&txn), 16)
                    .unwrap()
                    .len()
            },
            0
        );
        assert_eq!(
            {
                let core = cores.get(&second_name).unwrap();
                core.unavailable_externalized_marker_ids(&core.backend.read_borrowed(&txn), 16)
                    .unwrap()
                    .len()
            },
            1,
            "one repair step must not scan and purge every flagged segment"
        );
    }

    #[test]
    fn externalized_marker_repair_plan_revalidates_stale_ids() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = HelixGraphStorage::new(&path, Config::new(8, 32, 64, 1)).unwrap();
        create_dense_with_spindle(&storage, "dense", raw_hvec_spindle());
        let config = storage.named_vectors.get_config("dense").unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let segment_name = "dense__seg_000001".to_string();

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            let core = NamedVectorManager::create_dense_core(
                storage.lmdb_env().unwrap(),
                &mut txn,
                &segment_name,
                &config,
                hnsw,
                Arc::clone(&storage.backend),
            )
            .unwrap();
            core.insert_flat(&mut txn, &[1.0, 0.0, 0.0], Some(42), None)
                .unwrap();
            storage
                .named_vectors
                .cores
                .write()
                .unwrap()
                .insert(segment_name.clone(), core);
            storage.named_vectors.dense_spaces.write().unwrap().insert(
                "dense".to_string(),
                DenseVectorSpaceMetadata {
                    next_segment_id: 2,
                    segments: vec![DenseVectorSegmentMetadata {
                        physical_name: segment_name.clone(),
                        role: DenseVectorSegmentRole::Indexed,
                    }],
                },
            );
            storage
                .set_dense_vector_spaces_metadata(
                    &mut txn,
                    storage.named_vectors.list_dense_vector_spaces(),
                )
                .unwrap();
            txn.commit().unwrap();
        }
        write_header_only_hvec(
            &storage
                .lmdb_env()
                .unwrap()
                .path()
                .join("dense__seg_000001.hvec"),
            3,
        );
        drop(storage);

        let reopened = HelixGraphStorage::new(&path, Config::new(8, 32, 64, 1)).unwrap();
        let plan = {
            let txn = reopened.lmdb_env().unwrap().read_txn().unwrap();
            reopened
                .named_vectors
                .plan_next_unavailable_externalized_marker_segment(&txn, "dense", 16)
                .unwrap()
                .expect("damaged marker should produce a repair plan")
        };

        {
            let mut txn = reopened.lmdb_env().unwrap().write_txn().unwrap();
            assert_eq!(
                reopened
                    .named_vectors
                    .repair_unavailable_externalized_markers(&mut txn, "dense", 16)
                    .unwrap(),
                1
            );
            txn.commit().unwrap();
        }

        let mut txn = reopened.lmdb_env().unwrap().write_txn().unwrap();
        assert_eq!(
            reopened
                .named_vectors
                .apply_externalized_marker_repair_plan(&mut txn, "dense", &plan)
                .unwrap(),
            0,
            "stale repair plans must revalidate before reporting purged ids"
        );
        txn.commit().unwrap();
    }

    #[test]
    fn dense_segment_debt_counts_gate_pressure_and_dirty_retired() {
        let manager = NamedVectorManager::new("test".to_string());
        let cap = NamedVectorManager::segment_creation_gate_cap();
        let mut segments: Vec<DenseVectorSegmentMetadata> = (0..cap)
            .map(|idx| DenseVectorSegmentMetadata {
                physical_name: format!("dense__seg_{idx:06}"),
                role: DenseVectorSegmentRole::Indexed,
            })
            .collect();
        segments.push(DenseVectorSegmentMetadata {
            physical_name: "dense__seg_mutable".to_string(),
            role: DenseVectorSegmentRole::Mutable,
        });

        {
            let mut dense_spaces = manager.dense_spaces.write().unwrap();
            dense_spaces.insert(
                "dense".to_string(),
                DenseVectorSpaceMetadata {
                    next_segment_id: cap as u64 + 1,
                    segments,
                },
            );
        }
        {
            let mut dirty = manager.dirty_retired_segments.write().unwrap();
            dirty.insert("dense__seg_900001".to_string());
            dirty.insert("dense__seg_900002".to_string());
        }

        let debt = manager.dense_segment_debt();
        assert_eq!(debt.indexed_segments, cap);
        assert_eq!(debt.active_segments, cap);
        assert_eq!(
            debt.merge_debt,
            cap.saturating_sub(NamedVectorManager::max_indexed_segments_target())
        );
        assert_eq!(debt.gate_debt, 1);
        assert_eq!(debt.dirty_retired_segments, 2);
    }

    #[test]
    fn lsm_metadata_refresh_does_not_restore_hvtq_sidecars_on_open() {
        use crate::helix_engine::storage_core::backend::BackendKind;
        use crate::helix_engine::storage_core::backend::{SegmentDb, StorageBackend};
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use crate::helix_engine::vector_core::vector_core::HVTQ_SIDECAR_BLOB_KEY;
        use std::sync::Arc;
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("refresh-no-hvtq-restore-{nanos}"))
                .expect("open in-memory LSM backend"),
        ));
        assert_eq!(backend.kind(), BackendKind::Lsm);

        let dir = TempDir::new().unwrap();
        let manager = NamedVectorManager::new("refresh_test".to_string());
        manager.attach_backend(Arc::clone(&backend));

        let names = ["pf__seg_000000", "pf__seg_000001", "pf__seg_000002"];
        let blob: &[u8] = b"hvtq-sidecar-blob-deterministic-bytes";

        {
            let mut w = backend.begin_write().unwrap();
            for name in names {
                backend
                    .put(
                        &mut w,
                        Namespace::Segment {
                            physical_name: name,
                            db: SegmentDb::VectorData,
                        },
                        HVTQ_SIDECAR_BLOB_KEY,
                        blob,
                    )
                    .unwrap();
            }
            backend.commit(w).unwrap();
        }

        let mut configs = HashMap::new();
        configs.insert(
            "dense".to_string(),
            NamedVectorConfig {
                size: 4,
                distance: DistanceMetric::Cosine,
                spindle: SpindleConfig::turbo_prod_compact(4),
            },
        );
        let mut dense_spaces = HashMap::new();
        dense_spaces.insert(
            "dense".to_string(),
            DenseVectorSpaceMetadata {
                next_segment_id: 3,
                segments: names
                    .iter()
                    .map(|name| DenseVectorSegmentMetadata {
                        physical_name: (*name).to_string(),
                        role: DenseVectorSegmentRole::Indexed,
                    })
                    .collect(),
            },
        );

        manager
            .refresh_dense_view_from_metadata_lsm(
                dir.path(),
                &configs,
                &dense_spaces,
                HNSWConfig::new(Some(8), Some(16), Some(32)),
            )
            .unwrap();
        for name in names {
            assert!(
                !dir.path().join(name).with_extension("hvtq").exists(),
                "metadata refresh should not restore {name}"
            );
        }
    }

    #[test]
    fn dense_batch_append_fit_ignores_threshold_boundary() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));

        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        storage
            .named_vectors
            .dense_insert(
                storage.lmdb_env().unwrap(),
                &mut txn,
                "dense",
                &[1.0, 0.0, 0.0],
                1,
                HashMap::new(),
                hnsw,
                10,
            )
            .unwrap();
        txn.commit().unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let mut batch_counts = HashMap::new();
        batch_counts.insert("dense".to_string(), 1);
        assert!(!storage
            .named_vectors
            .dense_batch_fits_existing_segments(&txn, &batch_counts, 2)
            .unwrap());
        assert!(storage
            .named_vectors
            .dense_batch_can_append_existing_segments(&batch_counts, 2)
            .unwrap());
    }

    #[test]
    fn dense_search_spans_indexed_and_mutable_segments() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));

        storage
            .named_vectors
            .dense_insert(
                storage.lmdb_env().unwrap(),
                &mut txn,
                "dense",
                &[1.0, 0.0, 0.0],
                1,
                HashMap::new(),
                hnsw.clone(),
                2,
            )
            .unwrap();
        let build_job = storage
            .named_vectors
            .dense_insert(
                storage.lmdb_env().unwrap(),
                &mut txn,
                "dense",
                &[0.0, 1.0, 0.0],
                2,
                HashMap::new(),
                hnsw.clone(),
                2,
            )
            .unwrap()
            .unwrap();
        storage
            .named_vectors
            .build_dense_segment(&mut txn, "dense", &build_job)
            .unwrap();
        storage
            .named_vectors
            .dense_insert(
                storage.lmdb_env().unwrap(),
                &mut txn,
                "dense",
                &[0.9, 0.0, 0.0],
                3,
                HashMap::new(),
                hnsw,
                2,
            )
            .unwrap();
        txn.commit().unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let results = storage
            .named_vectors
            .dense_search_with_selectivity::<fn(&HVector) -> bool>(
                storage.lmdb_env().unwrap(),
                &txn,
                "dense",
                &[1.0, 0.0, 0.0],
                3,
                None,
                true,
                None,
            )
            .unwrap();

        let ids: Vec<u128> = results.iter().map(|vector| vector.id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
    }

    #[test]
    fn dense_search_skips_building_segments() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));

        let build_job = storage
            .named_vectors
            .dense_insert(
                storage.lmdb_env().unwrap(),
                &mut txn,
                "dense",
                &[1.0, 0.0, 0.0],
                1,
                HashMap::new(),
                hnsw.clone(),
                1,
            )
            .unwrap()
            .unwrap();
        storage
            .named_vectors
            .build_dense_segment(&mut txn, "dense", &build_job)
            .unwrap();

        storage
            .named_vectors
            .dense_insert(
                storage.lmdb_env().unwrap(),
                &mut txn,
                "dense",
                &[0.0, 1.0, 0.0],
                2,
                HashMap::new(),
                hnsw,
                1,
            )
            .unwrap()
            .expect("second segment should remain building");
        txn.commit().unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let results = storage
            .named_vectors
            .dense_search_with_selectivity::<fn(&HVector) -> bool>(
                storage.lmdb_env().unwrap(),
                &txn,
                "dense",
                &[0.0, 1.0, 0.0],
                2,
                None,
                true,
                None,
            )
            .unwrap();
        assert!(!results.iter().any(|vector| vector.id == 2));

        let filtered_results = storage
            .named_vectors
            .dense_search_with_id_filter_ef::<fn(u128) -> bool>(
                &txn,
                "dense",
                &[0.0, 1.0, 0.0],
                2,
                None,
                true,
                None,
                None,
            )
            .unwrap();
        assert!(!filtered_results.iter().any(|vector| vector.id == 2));
    }

    #[test]
    fn merge_dense_segments_preserves_search_results() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));

        for (id, vec) in [(1u128, vec![1.0, 0.0, 0.0]), (2u128, vec![0.9, 0.0, 0.0])] {
            let build_job = storage
                .named_vectors
                .dense_insert(
                    storage.lmdb_env().unwrap(),
                    &mut txn,
                    "dense",
                    &vec,
                    id,
                    HashMap::new(),
                    hnsw.clone(),
                    1,
                )
                .unwrap()
                .unwrap();
            storage
                .named_vectors
                .build_dense_segment(&mut txn, "dense", &build_job)
                .unwrap();
        }

        txn.commit().unwrap();

        // Split-phase merge: select candidates, prepare, flush
        let rtxn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let targets = storage
            .named_vectors
            .select_merge_candidates(&rtxn, "dense", 1)
            .unwrap()
            .expect("should have merge candidates");
        let merge = storage
            .named_vectors
            .prepare_merge(&rtxn, "dense", &targets, &hnsw)
            .unwrap();
        drop(rtxn);

        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let merged = storage
            .named_vectors
            .flush_prepared_merge(storage.lmdb_env().unwrap(), &mut txn, "dense", hnsw, merge)
            .unwrap();
        assert!(merged);
        txn.commit().unwrap();

        let spaces = storage.named_vectors.list_dense_vector_spaces();
        let space = spaces.get("dense").unwrap();
        assert_eq!(
            space
                .segments
                .iter()
                .filter(|segment| segment.role == DenseVectorSegmentRole::Indexed)
                .count(),
            1
        );

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let results = storage
            .named_vectors
            .dense_search_with_selectivity::<fn(&HVector) -> bool>(
                storage.lmdb_env().unwrap(),
                &txn,
                "dense",
                &[1.0, 0.0, 0.0],
                2,
                None,
                true,
                None,
            )
            .unwrap();
        let ids: Vec<u128> = results.iter().map(|vector| vector.id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }

    #[test]
    fn select_merge_candidates_with_fan_in_limit_bounds_backlog_slice() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        create_indexed_segments_for_selection(&storage, "dense", 30);

        let rtxn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let normal_targets = storage
            .named_vectors
            .select_merge_candidates_with_limits(
                &rtxn,
                "dense",
                DenseMergeCandidateLimits::new(1, 24),
            )
            .unwrap()
            .expect("normal selection should see merge debt");
        assert_eq!(
            normal_targets.len(),
            24,
            "normal optimizer fan-in should still allow throughput merges"
        );

        let drain_targets = storage
            .named_vectors
            .select_merge_candidates_with_limits(
                &rtxn,
                "dense",
                DenseMergeCandidateLimits::new(1, 2),
            )
            .unwrap()
            .expect("drain selection should see merge debt");
        assert_eq!(
            drain_targets,
            vec![
                "dense__seg_000001".to_string(),
                "dense__seg_000002".to_string()
            ],
            "breaker drain must pick the smallest bounded prefix"
        );
    }

    #[test]
    fn select_merge_candidates_with_bounded_prefix_picks_smallest_sampled_pair() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let config = storage.named_vectors.get_config("dense").unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let mut segments = Vec::new();
        for (idx, rows) in [(1usize, 8usize), (2, 7), (3, 1), (4, 1), (5, 1)] {
            let physical_name = format!("dense__seg_{idx:06}");
            let core = NamedVectorManager::create_dense_core(
                storage.lmdb_env().unwrap(),
                &mut txn,
                &physical_name,
                &config,
                hnsw.clone(),
                Arc::clone(&storage.backend),
            )
            .unwrap();
            for row in 0..rows {
                core.insert_flat(
                    &mut txn,
                    &[idx as f32, row as f32, 0.0],
                    Some(((idx as u128) << 32) | row as u128),
                    None,
                )
                .unwrap();
            }
            storage
                .named_vectors
                .cores
                .write()
                .unwrap()
                .insert(physical_name.clone(), core);
            segments.push(DenseVectorSegmentMetadata {
                physical_name,
                role: DenseVectorSegmentRole::Indexed,
            });
        }
        storage.named_vectors.dense_spaces.write().unwrap().insert(
            "dense".to_string(),
            DenseVectorSpaceMetadata {
                next_segment_id: 6,
                segments,
            },
        );
        storage
            .set_dense_vector_spaces_metadata(
                &mut txn,
                storage.named_vectors.list_dense_vector_spaces(),
            )
            .unwrap();
        txn.commit().unwrap();

        let rtxn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let normal_targets = storage
            .named_vectors
            .select_merge_candidates_with_limits(
                &rtxn,
                "dense",
                DenseMergeCandidateLimits::new(1, 2),
            )
            .unwrap()
            .expect("normal selection should see merge debt");
        assert_eq!(
            normal_targets,
            vec![
                "dense__seg_000003".to_string(),
                "dense__seg_000004".to_string()
            ]
        );

        let drain_targets = storage
            .named_vectors
            .select_merge_candidates_with_limits(
                &rtxn,
                "dense",
                DenseMergeCandidateLimits::bounded_prefix(1, 2),
            )
            .unwrap()
            .expect("bounded drain selection should see merge debt");
        assert_eq!(
            drain_targets,
            vec![
                "dense__seg_000003".to_string(),
                "dense__seg_000004".to_string()
            ]
        );
    }

    #[test]
    fn repeated_merge_converges_to_segment_cap() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));

        for id in 1u128..=4 {
            let build_job = storage
                .named_vectors
                .dense_insert(
                    storage.lmdb_env().unwrap(),
                    &mut txn,
                    "dense",
                    &[id as f32, 0.0, 0.0],
                    id,
                    HashMap::new(),
                    hnsw.clone(),
                    1,
                )
                .unwrap()
                .unwrap();
            storage
                .named_vectors
                .build_dense_segment(&mut txn, "dense", &build_job)
                .unwrap();
        }

        txn.commit().unwrap();

        // Use the split-phase merge API
        loop {
            let rtxn = storage.lmdb_env().unwrap().read_txn().unwrap();
            let candidates = storage
                .named_vectors
                .select_merge_candidates(&rtxn, "dense", 2)
                .unwrap();
            let Some(targets) = candidates else {
                drop(rtxn);
                break;
            };
            let merge = storage
                .named_vectors
                .prepare_merge(&rtxn, "dense", &targets, &hnsw)
                .unwrap();
            drop(rtxn);

            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            storage
                .named_vectors
                .flush_prepared_merge(
                    storage.lmdb_env().unwrap(),
                    &mut txn,
                    "dense",
                    hnsw.clone(),
                    merge,
                )
                .unwrap();
            txn.commit().unwrap();
        }

        let spaces = storage.named_vectors.list_dense_vector_spaces();
        let space = spaces.get("dense").unwrap();
        assert!(
            space.segments.len() <= 3,
            "expected <= 2 indexed segments plus mutable tail, got {}",
            space.segments.len()
        );
        assert!(
            space
                .segments
                .iter()
                .filter(|segment| segment.role == DenseVectorSegmentRole::Indexed)
                .count()
                <= 2
        );

        // Regression for the live EKS race: every segment published in
        // metadata must still have a live core after split-phase merge.
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let stats = storage
            .named_vectors
            .get_dense_space_stats(&txn, "dense")
            .unwrap();
        assert_eq!(stats.vectors_count, 4);
    }

    // ── Re-upsert tombstone tests ─────────────────────────────────────

    #[test]
    fn reupsert_tombstones_default_off_and_opt_in() {
        let _lock = TOMBSTONE_ENV_LOCK.lock().unwrap();
        let _unset = EnvVarGuard::unset("HELIX_REUPSERT_TOMBSTONES");
        assert!(
            !reupsert_tombstones_enabled(),
            "re-upsert tombstones must default off"
        );

        std::env::set_var("HELIX_REUPSERT_TOMBSTONES", "1");
        assert!(reupsert_tombstones_enabled());

        std::env::set_var("HELIX_REUPSERT_TOMBSTONES", "true");
        assert!(reupsert_tombstones_enabled());

        std::env::set_var("HELIX_REUPSERT_TOMBSTONES", "0");
        assert!(!reupsert_tombstones_enabled());
    }

    /// Seal one Indexed segment holding id=1, then re-upsert id=1 with a fresh
    /// vector into the mutable tail. The tombstone must be recorded for the
    /// sealed copy, search must return the NEW vector both before and after a
    /// merge, and the merge must drop the superseded copy.
    #[test]
    fn reupsert_tombstones_old_version_and_merge_drops_it() {
        let _lock = TOMBSTONE_ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::set("HELIX_REUPSERT_TOMBSTONES", "1");
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));

        // Seal id=1 into an Indexed segment via the explicit build path.
        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            let job = storage
                .named_vectors
                .dense_insert(
                    storage.lmdb_env().unwrap(),
                    &mut txn,
                    "dense",
                    &[1.0, 0.0, 0.0],
                    1,
                    HashMap::new(),
                    hnsw.clone(),
                    1,
                )
                .unwrap()
                .unwrap();
            storage
                .named_vectors
                .build_dense_segment(&mut txn, "dense", &job)
                .unwrap();
            txn.commit().unwrap();
        }

        // Re-upsert id=1 with a DIFFERENT vector into the mutable tail.
        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            storage
                .named_vectors
                .dense_append_to_mutable(
                    storage.lmdb_env().unwrap(),
                    &mut txn,
                    "dense",
                    &[0.0, 1.0, 0.0],
                    1,
                    HashMap::new(),
                    hnsw.clone(),
                    1_000_000, // high threshold: stay in mutable, do not seal
                )
                .unwrap();
            txn.commit().unwrap();
        }

        // A tombstone for the sealed copy must now exist.
        {
            let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
            let total = storage.named_vectors.dense_tombstone_count(&txn);
            assert_eq!(total, 1, "re-upsert must tombstone the sealed copy");
        }

        // Search returns the NEW vector (closest to [0,1,0]) before merge.
        let nearest = |storage: &HelixGraphStorage| -> u128 {
            let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
            let results = storage
                .named_vectors
                .dense_search_with_selectivity::<fn(&HVector) -> bool>(
                    storage.lmdb_env().unwrap(),
                    &txn,
                    "dense",
                    &[0.0, 1.0, 0.0],
                    1,
                    None,
                    true,
                    None,
                )
                .unwrap();
            results.first().map(|v| v.id).unwrap()
        };
        assert_eq!(
            nearest(&storage),
            1,
            "search returns id=1 (new version) pre-merge"
        );

        // Build the mutable tail into an Indexed segment, then merge the two.
        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            // Force the mutable tail to seal+build.
            let job = storage
                .named_vectors
                .dense_insert(
                    storage.lmdb_env().unwrap(),
                    &mut txn,
                    "dense",
                    &[0.0, 1.0, 0.0],
                    1,
                    HashMap::new(),
                    hnsw.clone(),
                    1,
                )
                .unwrap()
                .unwrap();
            storage
                .named_vectors
                .build_dense_segment(&mut txn, "dense", &job)
                .unwrap();
            txn.commit().unwrap();
        }

        let rtxn = storage.lmdb_env().unwrap().read_txn().unwrap();
        if let Some(targets) = storage
            .named_vectors
            .select_merge_candidates(&rtxn, "dense", 1)
            .unwrap()
        {
            let merge = storage
                .named_vectors
                .prepare_merge(&rtxn, "dense", &targets, &hnsw)
                .unwrap();
            drop(rtxn);
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            storage
                .named_vectors
                .flush_prepared_merge(
                    storage.lmdb_env().unwrap(),
                    &mut txn,
                    "dense",
                    hnsw.clone(),
                    merge,
                )
                .unwrap();
            txn.commit().unwrap();
        } else {
            drop(rtxn);
        }

        // After merge: search still returns id=1's NEW vector, and the
        // collection holds exactly one logical copy of id=1.
        assert_eq!(
            nearest(&storage),
            1,
            "search returns id=1 (new version) post-merge"
        );
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let stats = storage
            .named_vectors
            .get_dense_space_stats(&txn, "dense")
            .unwrap();
        assert_eq!(stats.vectors_count, 1, "superseded copy reclaimed by merge");
    }

    /// Tombstones are not persisted as a format — they are reconstructed from
    /// live data. After a re-upsert leaves two copies of an id across two
    /// segments, clearing the in-memory handle and calling
    /// `reconstruct_tombstones` must re-derive the tombstone.
    #[test]
    fn tombstones_reconstruct_from_live_data() {
        let _lock = TOMBSTONE_ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::set("HELIX_REUPSERT_TOMBSTONES", "1");
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));

        // Two Indexed segments, both holding id=7 (simulating a re-upsert that
        // landed in a second segment before any merge).
        for v in [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            let job = storage
                .named_vectors
                .dense_insert(
                    storage.lmdb_env().unwrap(),
                    &mut txn,
                    "dense",
                    &v,
                    7,
                    HashMap::new(),
                    hnsw.clone(),
                    1,
                )
                .unwrap()
                .unwrap();
            storage
                .named_vectors
                .build_dense_segment(&mut txn, "dense", &job)
                .unwrap();
            txn.commit().unwrap();
        }

        // Simulate a fresh open: drop the in-memory per-segment count cache so
        // reconstruct must re-derive the tombstone set from live data (the
        // durable rows now live in the `dense_tombstones` seam keyspace).
        {
            let mut counts = storage.named_vectors.tombstone_counts.write().unwrap();
            counts.clear();
        }

        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let written = storage
            .named_vectors
            .reconstruct_tombstones(&mut txn)
            .unwrap();
        txn.commit().unwrap();
        assert_eq!(written, 1, "id=7 present in two segments → one tombstone");

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        assert_eq!(storage.named_vectors.dense_tombstone_count(&txn), 1);
    }

    /// Seam round-trip (US-006 6b): the tombstone record / read / clear cycle is
    /// byte-identical through `Namespace::DenseTombstones`. Recording a
    /// supersession makes `is_tombstoned_for_merge` true; clearing the segment's
    /// tombstones makes it false again. Exercises `record_supersession` (put_raw
    /// → lazy-create), `is_tombstoned_for_merge` (get_with_raw) and
    /// `clear_tombstones_for_segment` (scan_raw prefix + delete_raw) all routed
    /// through the backend seam, not a raw heed `Database` handle.
    #[test]
    fn tombstone_record_read_clear_round_trip_through_seam() {
        let _lock = TOMBSTONE_ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::set("HELIX_REUPSERT_TOMBSTONES", "1");
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");

        let superseded_seg = "dense__seg_000001".to_string();
        let winner_seg = "dense__seg_000002";
        let id: u128 = 42;

        // Before any record: reads through the seam see an empty (never-created)
        // keyspace and return false.
        {
            let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
            assert!(
                !storage
                    .named_vectors
                    .is_tombstoned_for_merge(&txn, &superseded_seg, id)
                    .unwrap(),
                "no tombstone recorded yet"
            );
        }

        // Record a supersession: the older segment's copy of `id` is tombstoned
        // (winner is the freshly-appended copy in `winner_seg`). put_raw lazily
        // creates `dense_tombstones` on this first write.
        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            let recorded = storage
                .named_vectors
                .record_supersession(&mut txn, &[superseded_seg.clone()], winner_seg, id)
                .unwrap();
            txn.commit().unwrap();
            assert_eq!(recorded, 1, "exactly one superseded copy recorded");
        }

        // Read back through the seam: the tombstone is now visible.
        {
            let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
            assert!(
                storage
                    .named_vectors
                    .is_tombstoned_for_merge(&txn, &superseded_seg, id)
                    .unwrap(),
                "recorded tombstone must read back true through the seam"
            );
        }

        // Clear the segment's tombstones (retire path) and confirm the read flips
        // back to false.
        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            let cleared = storage
                .named_vectors
                .clear_tombstones_for_segment(&mut txn, &superseded_seg)
                .unwrap();
            txn.commit().unwrap();
            assert_eq!(cleared, 1, "the one recorded tombstone is cleared");
        }
        {
            let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
            assert!(
                !storage
                    .named_vectors
                    .is_tombstoned_for_merge(&txn, &superseded_seg, id)
                    .unwrap(),
                "cleared tombstone must read back false through the seam"
            );
        }
    }

    /// C2 read-path recency: when an id exists in an OLD sealed segment and a
    /// NEWER segment, search must return the NEWEST copy even when the query is
    /// nearer the OLD copy. Without recency dedup the min-distance copy (old)
    /// would win — returning a stale embedding for an edited point.
    #[test]
    fn search_returns_newest_copy_even_when_old_is_closer() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));

        // Segment seg_000001 (older): id=5 = [1,0,0].
        // Segment seg_000002 (newer): id=5 = [0,1,0].
        // Build them in id order so the second is the newer segment.
        for v in [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            let job = storage
                .named_vectors
                .dense_insert(
                    storage.lmdb_env().unwrap(),
                    &mut txn,
                    "dense",
                    &v,
                    5,
                    HashMap::new(),
                    hnsw.clone(),
                    1,
                )
                .unwrap()
                .unwrap();
            storage
                .named_vectors
                .build_dense_segment(&mut txn, "dense", &job)
                .unwrap();
            txn.commit().unwrap();
        }

        // Query nearer the OLD copy ([1,0,0]).
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let results = storage
            .named_vectors
            .dense_search_with_selectivity_ef::<fn(&HVector) -> bool>(
                storage.lmdb_env().unwrap(),
                &txn,
                "dense",
                &[1.0, 0.0, 0.0],
                3,
                None,
                true,
                None,
                None,
            )
            .unwrap();
        let id5: Vec<&HVector> = results.iter().filter(|v| v.id == 5).collect();
        assert_eq!(id5.len(), 1, "id=5 must be deduped to a single copy");
        // The surviving copy must be the NEWEST one ([0,1,0]), not the old
        // [1,0,0] that sits closer to the query.
        let data = id5[0].get_data();
        assert!(
            data[1] > data[0],
            "expected newest copy [0,1,0], got {:?}",
            data
        );
    }

    #[test]
    fn exact_candidate_id_search_scores_only_candidate_ids() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));

        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        for (id, vector) in [
            (1u128, [1.0f32, 0.0, 0.0]),
            (2u128, [0.9, 0.1, 0.0]),
            (3u128, [0.0, 1.0, 0.0]),
        ] {
            storage
                .named_vectors
                .dense_append_to_mutable(
                    storage.lmdb_env().unwrap(),
                    &mut txn,
                    "dense",
                    &vector,
                    id,
                    HashMap::new(),
                    hnsw.clone(),
                    1_000_000,
                )
                .unwrap();
        }
        txn.commit().unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let results = storage
            .named_vectors
            .dense_search_candidate_ids_exact(
                &txn,
                "dense",
                &[0.0, 1.0, 0.0],
                2,
                [1u128, 3u128],
                Some(2.0 / 3.0),
            )
            .unwrap();

        assert_eq!(
            results.iter().map(HVector::get_id).collect::<Vec<_>>(),
            [3, 1]
        );
    }

    #[test]
    fn exact_candidate_segment_routing_keeps_only_present_ids() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        create_indexed_segments_for_selection(&storage, "dense", 3);

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let cores = storage.named_vectors.cores.read().unwrap();
        let core = cores.get("dense__seg_000002").unwrap();
        let rd = core.backend.read_borrowed(&txn);
        let routed =
            exact_candidate_ids_present_in_segment(core, &rd, &[1u128, 2u128, 3u128, 404u128])
                .unwrap();

        assert_eq!(routed, [2u128]);
    }

    #[test]
    fn exact_candidate_id_search_dedups_reupsert_to_newest_segment() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));

        for vector in [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            let job = storage
                .named_vectors
                .dense_insert(
                    storage.lmdb_env().unwrap(),
                    &mut txn,
                    "dense",
                    &vector,
                    5,
                    HashMap::new(),
                    hnsw.clone(),
                    1,
                )
                .unwrap()
                .unwrap();
            storage
                .named_vectors
                .build_dense_segment(&mut txn, "dense", &job)
                .unwrap();
            txn.commit().unwrap();
        }

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let results = storage
            .named_vectors
            .dense_search_candidate_ids_exact(
                &txn,
                "dense",
                &[1.0, 0.0, 0.0],
                3,
                [5u128],
                Some(1.0),
            )
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 5);
        let data = results[0].get_data();
        assert!(
            data[1] > data[0],
            "expected newest copy [0,1,0], got {:?}",
            data
        );
    }

    /// Debt accounting: the txn-free `dense_segment_debt` path (which every
    /// optimizer priority scorer uses) must reflect the tombstone backlog via
    /// the in-memory cache — no LMDB read. Build a backlog through the same
    /// `bump_tombstone_count` the record path uses and assert merge_debt rises.
    #[test]
    fn tombstone_backlog_raises_merge_debt() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");

        let base = storage.named_vectors.dense_segment_debt();

        // Simulate the per-segment counts the record path maintains (4096
        // superseded copies = 4 merge-debt units at 1024/unit).
        storage
            .named_vectors
            .bump_tombstone_count("dense__seg_000001", 4096);

        // Txn-free path now sees the backlog (no LMDB scan).
        let with_tomb = storage.named_vectors.dense_segment_debt();
        assert!(
            with_tomb.merge_debt > base.merge_debt,
            "tombstone backlog must raise merge debt ({} !> {})",
            with_tomb.merge_debt,
            base.merge_debt
        );
        assert_eq!(
            with_tomb.merge_debt - base.merge_debt,
            4,
            "4096 tombstones / 1024 = 4 merge-debt units"
        );
        // Reporting variant exposes the raw count.
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        assert_eq!(
            storage
                .named_vectors
                .dense_segment_debt_with_tombstones(&txn)
                .tombstoned_copies,
            4096
        );
    }

    #[test]
    fn many_dense_segments_do_not_exhaust_lmdb_dbs() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));

        for id in 1u128..=20 {
            let build_job = storage
                .named_vectors
                .dense_insert(
                    storage.lmdb_env().unwrap(),
                    &mut txn,
                    "dense",
                    &[id as f32, 0.0, 0.0],
                    id,
                    HashMap::new(),
                    hnsw.clone(),
                    1,
                )
                .unwrap()
                .unwrap();
            storage
                .named_vectors
                .build_dense_segment(&mut txn, "dense", &build_job)
                .unwrap();
        }
    }

    #[test]
    fn reaped_retired_segment_becomes_reusable_gap() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let config = storage.named_vectors.get_config("dense").unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let retired_name = "dense__seg_000001".to_string();

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            let retired_core = NamedVectorManager::create_dense_core(
                storage.lmdb_env().unwrap(),
                &mut txn,
                &retired_name,
                &config,
                hnsw,
                Arc::clone(&storage.backend),
            )
            .unwrap();
            retired_core
                .insert_flat(&mut txn, &[1.0, 0.0, 0.0], Some(42), None)
                .unwrap();

            storage
                .named_vectors
                .cores
                .write()
                .unwrap()
                .insert(retired_name.clone(), retired_core);

            let mut dense_spaces = storage.named_vectors.dense_spaces.write().unwrap();
            let space = dense_spaces.get_mut("dense").unwrap();
            space.next_segment_id = 3;
            space.segments.push(DenseVectorSegmentMetadata {
                physical_name: retired_name.clone(),
                role: DenseVectorSegmentRole::Indexed,
            });
            space.segments.push(DenseVectorSegmentMetadata {
                physical_name: "dense__seg_000002".to_string(),
                role: DenseVectorSegmentRole::Mutable,
            });
            txn.commit().unwrap();
        }

        {
            let mut dense_spaces = storage.named_vectors.dense_spaces.write().unwrap();
            let space = dense_spaces.get_mut("dense").unwrap();
            space
                .segments
                .retain(|segment| segment.physical_name != retired_name);
        }

        {
            let dense_spaces = storage.named_vectors.dense_spaces.read().unwrap();
            let mut probe = dense_spaces.get("dense").unwrap().clone();
            let cores = storage.named_vectors.cores.read().unwrap();
            let before_reap = next_segment_name(
                "dense",
                alloc_segment_id(&mut probe, "dense", cores.keys().map(String::as_str)),
            );
            assert_eq!(before_reap, "dense__seg_000003");
        }

        let mut drained = false;
        for _ in 0..32 {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            drained = storage
                .named_vectors
                .drop_retired_segment_chunk(&mut txn, &retired_name, 1)
                .unwrap();
            txn.commit().unwrap();
            if drained {
                break;
            }
        }
        assert!(drained, "retired segment must eventually drain");
        assert!(!storage
            .named_vectors
            .cores
            .read()
            .unwrap()
            .contains_key(&retired_name));

        let mut dense_spaces = storage.named_vectors.dense_spaces.write().unwrap();
        let space = dense_spaces.get_mut("dense").unwrap();
        let cores = storage.named_vectors.cores.read().unwrap();
        let reused = next_segment_name(
            "dense",
            alloc_segment_id(space, "dense", cores.keys().map(String::as_str)),
        );
        assert_eq!(reused, retired_name);
        assert_eq!(space.next_segment_id, 3);
    }

    #[test]
    fn dirty_retired_gap_is_not_reused() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let config = storage.named_vectors.get_config("dense").unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let dirty_name = "dense__seg_000001";
        let discovered = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        storage.named_vectors.attach_dirty_retired_segment_hook({
            let discovered = std::sync::Arc::clone(&discovered);
            move |physical_name| {
                discovered.lock().unwrap().push(physical_name);
            }
        });

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            let dirty_core = NamedVectorManager::create_dense_core(
                storage.lmdb_env().unwrap(),
                &mut txn,
                dirty_name,
                &config,
                hnsw.clone(),
                Arc::clone(&storage.backend),
            )
            .unwrap();
            dirty_core
                .insert_flat(&mut txn, &[1.0, 0.0, 0.0], Some(7), None)
                .unwrap();

            let mut dense_spaces = storage.named_vectors.dense_spaces.write().unwrap();
            let space = dense_spaces.get_mut("dense").unwrap();
            space.next_segment_id = 3;
            space.segments.push(DenseVectorSegmentMetadata {
                physical_name: "dense__seg_000002".to_string(),
                role: DenseVectorSegmentRole::Mutable,
            });
            txn.commit().unwrap();
        }

        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let mut dense_spaces = storage.named_vectors.dense_spaces.write().unwrap();
        let space = dense_spaces.get_mut("dense").unwrap();
        let cores = storage.named_vectors.cores.read().unwrap();
        let (allocated_name, allocated_core) = storage
            .named_vectors
            .allocate_dense_core_locked(
                storage.lmdb_env().unwrap(),
                &mut txn,
                "dense",
                &config,
                hnsw,
                space,
                &cores,
            )
            .unwrap();

        assert_eq!(allocated_name, "dense__seg_000003");
        assert!(allocated_core
            .is_empty(&allocated_core.backend.read_borrowed(&txn))
            .unwrap());
        assert_eq!(space.next_segment_id, 4);
        assert!(storage
            .named_vectors
            .dirty_retired_segments
            .read()
            .unwrap()
            .contains(dirty_name));
        assert_eq!(
            discovered.lock().unwrap().as_slice(),
            &[dirty_name.to_string()]
        );
    }

    #[test]
    fn dirty_retired_gap_probe_stops_after_first_dirty_slot() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let config = storage.named_vectors.get_config("dense").unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let first_dirty = "dense__seg_000001";
        let second_dirty = "dense__seg_000002";
        let discovered = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        storage.named_vectors.attach_dirty_retired_segment_hook({
            let discovered = std::sync::Arc::clone(&discovered);
            move |physical_name| {
                discovered.lock().unwrap().push(physical_name);
            }
        });

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            for dirty_name in [first_dirty, second_dirty] {
                let dirty_core = NamedVectorManager::create_dense_core(
                    storage.lmdb_env().unwrap(),
                    &mut txn,
                    dirty_name,
                    &config,
                    hnsw.clone(),
                    Arc::clone(&storage.backend),
                )
                .unwrap();
                dirty_core
                    .insert_flat(&mut txn, &[1.0, 0.0, 0.0], Some(7), None)
                    .unwrap();
            }

            let mut dense_spaces = storage.named_vectors.dense_spaces.write().unwrap();
            let space = dense_spaces.get_mut("dense").unwrap();
            space.next_segment_id = 4;
            space.segments.push(DenseVectorSegmentMetadata {
                physical_name: "dense__seg_000003".to_string(),
                role: DenseVectorSegmentRole::Mutable,
            });
            txn.commit().unwrap();
        }

        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let mut dense_spaces = storage.named_vectors.dense_spaces.write().unwrap();
        let space = dense_spaces.get_mut("dense").unwrap();
        let cores = storage.named_vectors.cores.read().unwrap();
        let (allocated_name, allocated_core) = storage
            .named_vectors
            .allocate_dense_core_locked(
                storage.lmdb_env().unwrap(),
                &mut txn,
                "dense",
                &config,
                hnsw,
                space,
                &cores,
            )
            .unwrap();

        assert_eq!(allocated_name, "dense__seg_000004");
        assert!(allocated_core
            .is_empty(&allocated_core.backend.read_borrowed(&txn))
            .unwrap());
        assert_eq!(space.next_segment_id, 5);
        let dirty_retired = storage.named_vectors.dirty_retired_segments.read().unwrap();
        assert!(dirty_retired.contains(first_dirty));
        assert!(!dirty_retired.contains(second_dirty));
        assert_eq!(
            discovered.lock().unwrap().as_slice(),
            &[first_dirty.to_string()]
        );
    }

    #[test]
    fn unregistered_dirty_retired_segment_drains_before_reuse() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let config = storage.named_vectors.get_config("dense").unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let dirty_name = "dense__seg_000001";

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            let dirty_core = NamedVectorManager::create_dense_core(
                storage.lmdb_env().unwrap(),
                &mut txn,
                dirty_name,
                &config,
                hnsw.clone(),
                Arc::clone(&storage.backend),
            )
            .unwrap();
            dirty_core
                .insert_flat(&mut txn, &[1.0, 0.0, 0.0], Some(7), None)
                .unwrap();
            txn.commit().unwrap();
        }
        assert!(!storage
            .named_vectors
            .cores
            .read()
            .unwrap()
            .contains_key(dirty_name));
        storage
            .named_vectors
            .dirty_retired_segments
            .write()
            .unwrap()
            .insert(dirty_name.to_string());

        for db_index in 0..5 {
            let mut drained = false;
            for _ in 0..32 {
                let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
                drained = storage
                    .named_vectors
                    .drop_retired_segment_db_step(
                        storage.lmdb_env().unwrap(),
                        &mut txn,
                        dirty_name,
                        db_index,
                        1,
                    )
                    .unwrap();
                txn.commit().unwrap();
                if drained {
                    break;
                }
            }
            assert!(drained, "db index {db_index} did not drain");
        }
        storage.named_vectors.unlink_sidecar_files_for_segments(
            storage.lmdb_env().unwrap().path(),
            &[dirty_name.to_string()],
        );
        storage
            .named_vectors
            .finalize_drained_retired_segment(dirty_name)
            .unwrap();
        assert!(!storage
            .named_vectors
            .dirty_retired_segments
            .read()
            .unwrap()
            .contains(dirty_name));

        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let mut dense_spaces = storage.named_vectors.dense_spaces.write().unwrap();
        let space = dense_spaces.get_mut("dense").unwrap();
        space.next_segment_id = 2;
        let cores = storage.named_vectors.cores.read().unwrap();
        let (allocated_name, allocated_core) = storage
            .named_vectors
            .allocate_dense_core_locked(
                storage.lmdb_env().unwrap(),
                &mut txn,
                "dense",
                &config,
                hnsw,
                space,
                &cores,
            )
            .unwrap();

        assert_eq!(allocated_name, dirty_name);
        assert!(allocated_core
            .is_empty(&allocated_core.backend.read_borrowed(&txn))
            .unwrap());
    }

    #[test]
    fn delete_vectors_batch_skips_retired_cores() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let config = storage.named_vectors.get_config("dense").unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let retired_name = "dense__seg_000001".to_string();

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            {
                let cores = storage.named_vectors.cores.read().unwrap();
                cores
                    .get("dense")
                    .unwrap()
                    .insert_flat(&mut txn, &[1.0, 0.0, 0.0], Some(8), None)
                    .unwrap();
            }

            let retired_core = NamedVectorManager::create_dense_core(
                storage.lmdb_env().unwrap(),
                &mut txn,
                &retired_name,
                &config,
                hnsw,
                Arc::clone(&storage.backend),
            )
            .unwrap();
            retired_core
                .insert_flat(&mut txn, &[0.0, 1.0, 0.0], Some(7), None)
                .unwrap();
            storage
                .named_vectors
                .cores
                .write()
                .unwrap()
                .insert(retired_name.clone(), retired_core);
            txn.commit().unwrap();
        }

        {
            let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
            let plan = storage
                .named_vectors
                .plan_delete_vectors_batch(&txn, &[7, 8])
                .unwrap();
            assert_eq!(plan.by_segment.len(), 1);
            assert_eq!(plan.by_segment[0].1, vec![8]);
        }

        {
            let plan = {
                let rtxn = storage.lmdb_env().unwrap().read_txn().unwrap();
                storage
                    .named_vectors
                    .plan_delete_vectors_batch(&rtxn, &[7, 8])
                    .unwrap()
            };
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            storage
                .named_vectors
                .delete_vectors_batch_with_plan(&mut txn, &plan)
                .unwrap();
            txn.commit().unwrap();
        }

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let cores = storage.named_vectors.cores.read().unwrap();
        assert!(!cores
            .get("dense")
            .unwrap()
            .contains_id(&cores.get("dense").unwrap().backend.read_borrowed(&txn), 8)
            .unwrap());
        assert!({
            let core = cores.get(&retired_name).unwrap();
            core.contains_id(&core.backend.read_borrowed(&txn), 7)
                .unwrap()
        });
    }

    /// Backend-agnostic (LSM) counterpart of the LMDB
    /// `plan_delete_vectors_batch` / `delete_vectors_batch_with_plan` test
    /// above: the ownership probe must run against a read handle and the
    /// resulting plan must delete only the ids each segment actually owns.
    #[test]
    fn plan_delete_vectors_batch_be_removes_exactly_owned_ids_across_segments() {
        // `delete_vectors_batch_with_plan_be` branches on
        // `HELIX_LSM_DELETE_TOMBSTONES`; hold the shared lock and force it
        // off so this test's hard-delete assertions aren't racy against the
        // `delete_tombstones_*` tests that flip the same process-global var.
        let _lock = TOMBSTONE_ENV_LOCK.lock().unwrap();
        let _unset = EnvVarGuard::unset("HELIX_LSM_DELETE_TOMBSTONES");
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::sync::Arc;
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("plan-delete-be-multi-{nanos}"))
                .expect("open in-memory LSM backend"),
        ));

        let manager = NamedVectorManager::new("plan_delete_multi_test".to_string());
        manager.attach_backend(Arc::clone(&backend));

        let dir = TempDir::new().unwrap();
        let config = NamedVectorConfig {
            size: 3,
            distance: DistanceMetric::Cosine,
            spindle: SpindleConfig::default(),
        };
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));

        let seg_names = [
            "dense__seg_000000",
            "dense__seg_000001",
            "dense__seg_000002",
        ];
        let segments: Vec<DenseVectorSegmentMetadata> = seg_names
            .iter()
            .map(|name| DenseVectorSegmentMetadata {
                physical_name: (*name).to_string(),
                role: DenseVectorSegmentRole::Indexed,
            })
            .collect();
        {
            let mut cores = manager.cores.write().unwrap();
            for name in seg_names {
                let core = manager
                    .create_dense_core_attached_lsm(dir.path(), name, &config, hnsw.clone())
                    .unwrap();
                cores.insert(name.to_string(), core);
            }
        }
        manager.dense_spaces.write().unwrap().insert(
            "dense".to_string(),
            DenseVectorSpaceMetadata {
                next_segment_id: 3,
                segments,
            },
        );

        // ids 1 and 2 live on segment 0, id 3 on segment 1, id 4 on segment 2.
        // id 999 never exists anywhere.
        {
            let mut w = backend.begin_write().unwrap();
            let cores = manager.cores.read().unwrap();
            cores
                .get(seg_names[0])
                .unwrap()
                .insert_flat_be(&mut w, &[1.0, 0.0, 0.0], Some(1), None)
                .unwrap();
            cores
                .get(seg_names[0])
                .unwrap()
                .insert_flat_be(&mut w, &[0.0, 1.0, 0.0], Some(2), None)
                .unwrap();
            cores
                .get(seg_names[1])
                .unwrap()
                .insert_flat_be(&mut w, &[0.0, 0.0, 1.0], Some(3), None)
                .unwrap();
            cores
                .get(seg_names[2])
                .unwrap()
                .insert_flat_be(&mut w, &[1.0, 1.0, 0.0], Some(4), None)
                .unwrap();
            backend.commit(w).unwrap();
        }

        let plan = {
            let r = backend.begin_read().unwrap();
            manager
                .plan_delete_vectors_batch_be(&r, &[1, 3, 4, 999])
                .unwrap()
        };
        assert_eq!(plan.ids, vec![1, 3, 4, 999]);
        assert_eq!(
            plan.by_segment.len(),
            3,
            "id 999 owns no segment; ids 1/3/4 own exactly one segment each"
        );
        let owned_by = |seg: &str| -> Option<&Vec<u128>> {
            plan.by_segment
                .iter()
                .find(|(name, _)| name == seg)
                .map(|(_, ids)| ids)
        };
        assert_eq!(owned_by(seg_names[0]), Some(&vec![1u128]));
        assert_eq!(owned_by(seg_names[1]), Some(&vec![3u128]));
        assert_eq!(owned_by(seg_names[2]), Some(&vec![4u128]));

        {
            let mut w = backend.begin_write().unwrap();
            let staged = manager
                .delete_vectors_batch_with_plan_be(&mut w, &plan)
                .unwrap();
            backend.commit(w).unwrap();
            manager.apply_delete_tombstones(&staged);
        }

        let r = backend.begin_read().unwrap();
        let cores = manager.cores.read().unwrap();
        assert!(
            !cores.get(seg_names[0]).unwrap().contains_id(&r, 1).unwrap(),
            "id 1 must be deleted"
        );
        assert!(
            cores.get(seg_names[0]).unwrap().contains_id(&r, 2).unwrap(),
            "id 2 was not targeted and must remain"
        );
        assert!(
            !cores.get(seg_names[1]).unwrap().contains_id(&r, 3).unwrap(),
            "id 3 must be deleted"
        );
        assert!(
            !cores.get(seg_names[2]).unwrap().contains_id(&r, 4).unwrap(),
            "id 4 must be deleted"
        );
    }

    /// `plan_delete_vectors_batch_be` must type-check and succeed against a
    /// read-only `AnyRead` handle with no write handle open concurrently —
    /// this is what lets the LSM delete path move the ownership probe out
    /// of the write hold.
    #[test]
    fn plan_delete_vectors_batch_be_runs_on_read_only_handle() {
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::sync::Arc;
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("plan-delete-be-readonly-{nanos}"))
                .expect("open in-memory LSM backend"),
        ));

        let manager = NamedVectorManager::new("plan_delete_readonly_test".to_string());
        manager.attach_backend(Arc::clone(&backend));

        let dir = TempDir::new().unwrap();
        let config = NamedVectorConfig {
            size: 3,
            distance: DistanceMetric::Cosine,
            spindle: SpindleConfig::default(),
        };
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let seg_name = "dense__seg_000000";
        {
            let core = manager
                .create_dense_core_attached_lsm(dir.path(), seg_name, &config, hnsw)
                .unwrap();
            manager
                .cores
                .write()
                .unwrap()
                .insert(seg_name.to_string(), core);
        }
        manager.dense_spaces.write().unwrap().insert(
            "dense".to_string(),
            DenseVectorSpaceMetadata {
                next_segment_id: 1,
                segments: vec![DenseVectorSegmentMetadata {
                    physical_name: seg_name.to_string(),
                    role: DenseVectorSegmentRole::Indexed,
                }],
            },
        );
        {
            let mut w = backend.begin_write().unwrap();
            manager
                .cores
                .read()
                .unwrap()
                .get(seg_name)
                .unwrap()
                .insert_flat_be(&mut w, &[1.0, 0.0, 0.0], Some(1), None)
                .unwrap();
            backend.commit(w).unwrap();
        }

        // No write handle is open here — only a read handle exists when
        // planning runs.
        let r = backend.begin_read().unwrap();
        let plan = manager.plan_delete_vectors_batch_be(&r, &[1]).unwrap();
        assert_eq!(plan.by_segment.len(), 1);
        assert_eq!(plan.by_segment[0].1, vec![1u128]);
    }

    /// Re-running plan + execute on ids that were already deleted by a prior
    /// round must be a no-op: no error, and the already-deleted ids stay
    /// gone. Guards the same concurrent-disappearance tolerance the LMDB
    /// `delete_vectors_batch_with_plan` path relies on.
    #[test]
    fn delete_vectors_batch_with_plan_be_is_idempotent() {
        // See the lock rationale on
        // `plan_delete_vectors_batch_be_removes_exactly_owned_ids_across_segments`.
        let _lock = TOMBSTONE_ENV_LOCK.lock().unwrap();
        let _unset = EnvVarGuard::unset("HELIX_LSM_DELETE_TOMBSTONES");
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::sync::Arc;
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("plan-delete-be-idempotent-{nanos}"))
                .expect("open in-memory LSM backend"),
        ));

        let manager = NamedVectorManager::new("plan_delete_idempotent_test".to_string());
        manager.attach_backend(Arc::clone(&backend));

        let dir = TempDir::new().unwrap();
        let config = NamedVectorConfig {
            size: 3,
            distance: DistanceMetric::Cosine,
            spindle: SpindleConfig::default(),
        };
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let seg_name = "dense__seg_000000";
        {
            let core = manager
                .create_dense_core_attached_lsm(dir.path(), seg_name, &config, hnsw)
                .unwrap();
            manager
                .cores
                .write()
                .unwrap()
                .insert(seg_name.to_string(), core);
        }
        manager.dense_spaces.write().unwrap().insert(
            "dense".to_string(),
            DenseVectorSpaceMetadata {
                next_segment_id: 1,
                segments: vec![DenseVectorSegmentMetadata {
                    physical_name: seg_name.to_string(),
                    role: DenseVectorSegmentRole::Indexed,
                }],
            },
        );
        {
            let mut w = backend.begin_write().unwrap();
            manager
                .cores
                .read()
                .unwrap()
                .get(seg_name)
                .unwrap()
                .insert_flat_be(&mut w, &[1.0, 0.0, 0.0], Some(1), None)
                .unwrap();
            backend.commit(w).unwrap();
        }

        // First round: id 1 is owned and gets deleted.
        let plan = {
            let r = backend.begin_read().unwrap();
            manager.plan_delete_vectors_batch_be(&r, &[1]).unwrap()
        };
        assert_eq!(plan.by_segment.len(), 1);
        {
            let mut w = backend.begin_write().unwrap();
            let staged = manager
                .delete_vectors_batch_with_plan_be(&mut w, &plan)
                .unwrap();
            backend.commit(w).unwrap();
            manager.apply_delete_tombstones(&staged);
        }
        {
            let r = backend.begin_read().unwrap();
            assert!(!manager
                .cores
                .read()
                .unwrap()
                .get(seg_name)
                .unwrap()
                .contains_id(&r, 1)
                .unwrap());
        }

        // Second round: id 1 is already gone from every segment, so the plan
        // owns nothing and execute must be a safe no-op.
        let plan_again = {
            let r = backend.begin_read().unwrap();
            manager.plan_delete_vectors_batch_be(&r, &[1]).unwrap()
        };
        assert!(
            plan_again.by_segment.is_empty(),
            "no segment should still own the already-deleted id"
        );
        {
            let mut w = backend.begin_write().unwrap();
            let staged = manager
                .delete_vectors_batch_with_plan_be(&mut w, &plan_again)
                .unwrap();
            backend.commit(w).unwrap();
            manager.apply_delete_tombstones(&staged);
        }
    }

    // ── Deferred-repair delete tombstones (issue #29 "fix 2") ──────────

    #[test]
    fn lsm_delete_tombstones_default_off_and_opt_in() {
        let _lock = TOMBSTONE_ENV_LOCK.lock().unwrap();
        let _unset = EnvVarGuard::unset("HELIX_LSM_DELETE_TOMBSTONES");
        assert!(
            !lsm_delete_tombstones_enabled(),
            "delete tombstones must default off"
        );

        std::env::set_var("HELIX_LSM_DELETE_TOMBSTONES", "1");
        assert!(lsm_delete_tombstones_enabled());

        std::env::set_var("HELIX_LSM_DELETE_TOMBSTONES", "true");
        assert!(lsm_delete_tombstones_enabled());

        std::env::set_var("HELIX_LSM_DELETE_TOMBSTONES", "0");
        assert!(!lsm_delete_tombstones_enabled());
    }

    /// Test rig: a `NamedVectorManager` over an in-memory LSM backend with one
    /// Indexed segment holding `ids_and_vectors` (flat-inserted, no HNSW
    /// build). Mirrors the setup used by the `plan_delete_vectors_batch_be_*`
    /// tests above.
    fn setup_lsm_manager_with_flat_segment(
        test_name: &str,
        seg_name: &str,
        ids_and_vectors: &[(u128, [f32; 3])],
    ) -> (Arc<AnyBackend>, NamedVectorManager, TempDir) {
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("{test_name}-{nanos}"))
                .expect("open in-memory LSM backend"),
        ));
        let manager = NamedVectorManager::new(format!("{test_name}_collection"));
        manager.attach_backend(Arc::clone(&backend));
        let dir = TempDir::new().unwrap();
        let config = NamedVectorConfig {
            size: 3,
            distance: DistanceMetric::Cosine,
            spindle: SpindleConfig::default(),
        };
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        {
            let core = manager
                .create_dense_core_attached_lsm(dir.path(), seg_name, &config, hnsw)
                .unwrap();
            let mut w = backend.begin_write().unwrap();
            for (id, v) in ids_and_vectors {
                core.insert_flat_be(&mut w, v, Some(*id), None).unwrap();
            }
            backend.commit(w).unwrap();
            manager
                .cores
                .write()
                .unwrap()
                .insert(seg_name.to_string(), core);
        }
        manager.dense_spaces.write().unwrap().insert(
            "dense".to_string(),
            DenseVectorSpaceMetadata {
                next_segment_id: 1,
                segments: vec![DenseVectorSegmentMetadata {
                    physical_name: seg_name.to_string(),
                    role: DenseVectorSegmentRole::Indexed,
                }],
            },
        );
        (backend, manager, dir)
    }

    /// flag OFF (the default): the with-plan execute path must be
    /// byte-identical to today — rows are hard-deleted, and no
    /// delete-tombstone bookkeeping (durable key or in-memory set) happens at
    /// all.
    #[test]
    fn delete_tombstones_flag_off_hard_deletes_exactly_as_today() {
        let _lock = TOMBSTONE_ENV_LOCK.lock().unwrap();
        let _unset = EnvVarGuard::unset("HELIX_LSM_DELETE_TOMBSTONES");
        assert!(!lsm_delete_tombstones_enabled());

        let seg_name = "dense__seg_000000";
        let (backend, manager, _dir) = setup_lsm_manager_with_flat_segment(
            "delete_tombstones_off",
            seg_name,
            &[(1, [1.0, 0.0, 0.0]), (2, [0.0, 1.0, 0.0])],
        );

        let plan = {
            let r = backend.begin_read().unwrap();
            manager.plan_delete_vectors_batch_be(&r, &[1]).unwrap()
        };
        {
            let mut w = backend.begin_write().unwrap();
            let staged = manager
                .delete_vectors_batch_with_plan_be(&mut w, &plan)
                .unwrap();
            backend.commit(w).unwrap();
            manager.apply_delete_tombstones(&staged);
        }

        let r = backend.begin_read().unwrap();
        let cores = manager.cores.read().unwrap();
        let core = cores.get(seg_name).unwrap();
        assert!(
            !core.contains_id(&r, 1).unwrap(),
            "flag off must hard-delete the row exactly as before"
        );
        assert!(
            !core.is_delete_tombstoned(1),
            "flag off must not record a delete-tombstone"
        );
        assert_eq!(core.deleted_count(), 0);
    }

    /// flag ON: ids staged by the with-plan execute must NOT appear in the
    /// in-memory set until `apply_delete_tombstones` runs post-commit — a
    /// failed commit would otherwise suppress live vectors from search.
    #[test]
    fn delete_tombstones_apply_is_deferred_until_after_commit() {
        let _lock = TOMBSTONE_ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::set("HELIX_LSM_DELETE_TOMBSTONES", "1");

        let seg_name = "dense__seg_000000";
        let (backend, manager, _dir) = setup_lsm_manager_with_flat_segment(
            "delete_tombstones_deferred_apply",
            seg_name,
            &[(1, [1.0, 0.0, 0.0]), (2, [0.0, 1.0, 0.0])],
        );
        let plan = {
            let r = backend.begin_read().unwrap();
            manager.plan_delete_vectors_batch_be(&r, &[1]).unwrap()
        };
        let mut w = backend.begin_write().unwrap();
        let staged = manager
            .delete_vectors_batch_with_plan_be(&mut w, &plan)
            .unwrap();

        let tombstoned_pre_commit = {
            let cores = manager.cores.read().unwrap();
            cores.get(seg_name).unwrap().is_delete_tombstoned(1)
        };
        assert!(
            !tombstoned_pre_commit,
            "staging must not touch the in-memory set before commit"
        );

        backend.commit(w).unwrap();
        manager.apply_delete_tombstones(&staged);
        let cores = manager.cores.read().unwrap();
        assert!(cores.get(seg_name).unwrap().is_delete_tombstoned(1));
        assert!(!cores.get(seg_name).unwrap().is_delete_tombstoned(2));
    }

    /// flag ON: the with-plan execute records a delete-tombstone instead of
    /// removing the row. The raw row stays present (repair is deferred to
    /// merge), the in-memory set reflects it, and re-running the same delete
    /// is a no-op (idempotent, matching the flag-off contract).
    #[test]
    fn delete_tombstones_flag_on_leaves_row_and_is_idempotent() {
        let _lock = TOMBSTONE_ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::set("HELIX_LSM_DELETE_TOMBSTONES", "1");

        let seg_name = "dense__seg_000000";
        let (backend, manager, _dir) = setup_lsm_manager_with_flat_segment(
            "delete_tombstones_on",
            seg_name,
            &[(1, [1.0, 0.0, 0.0]), (2, [0.0, 1.0, 0.0])],
        );

        for _round in 0..2 {
            let plan = {
                let r = backend.begin_read().unwrap();
                manager.plan_delete_vectors_batch_be(&r, &[1]).unwrap()
            };
            {
                let mut w = backend.begin_write().unwrap();
                let staged = manager
                    .delete_vectors_batch_with_plan_be(&mut w, &plan)
                    .unwrap();
                backend.commit(w).unwrap();
                manager.apply_delete_tombstones(&staged);
            }
        }

        let r = backend.begin_read().unwrap();
        let cores = manager.cores.read().unwrap();
        let core = cores.get(seg_name).unwrap();
        assert!(
            core.contains_id(&r, 1).unwrap(),
            "flag on must leave the raw row in place; repair is deferred to merge"
        );
        assert!(core.is_delete_tombstoned(1));
        assert_eq!(
            core.deleted_count(),
            1,
            "repeat delete of the same id must not double-count"
        );
        assert!(!core.is_delete_tombstoned(2), "id 2 was never targeted");
    }

    /// flag ON, flat/mutable-tail search (no HNSW build): a delete-tombstoned
    /// id must never be returned, on both the unfiltered and id-filtered
    /// search paths.
    #[test]
    fn delete_tombstones_flag_on_flat_search_excludes_deleted_id() {
        let _lock = TOMBSTONE_ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::set("HELIX_LSM_DELETE_TOMBSTONES", "1");

        let seg_name = "dense__seg_000000";
        let (backend, manager, _dir) = setup_lsm_manager_with_flat_segment(
            "delete_tombstones_flat_search",
            seg_name,
            &[
                (1, [1.0, 0.0, 0.0]),
                (2, [0.9, 0.1, 0.0]),
                (3, [0.0, 0.0, 1.0]),
            ],
        );

        // Tombstone-delete id=1, the closest vector to the query below.
        {
            let mut w = backend.begin_write().unwrap();
            let cores = manager.cores.read().unwrap();
            let core = cores.get(seg_name).unwrap();
            let staged = core.tombstone_delete_batch_be(&mut w, &[1]).unwrap();
            backend.commit(w).unwrap();
            core.apply_delete_tombstones(&staged);
        }

        let r = backend.begin_read().unwrap();
        let results = manager
            .dense_search_with_id_filter_ef_be::<fn(u128) -> bool>(
                &r,
                "dense",
                &[1.0, 0.0, 0.0],
                3,
                None,
                false,
                None,
                None,
            )
            .unwrap();
        assert!(
            !results.iter().any(|v| v.get_id() == 1),
            "tombstoned id must never be a search result: got {:?}",
            results.iter().map(|v| v.get_id()).collect::<Vec<_>>()
        );
        assert!(results.iter().any(|v| v.get_id() == 2));
        assert!(results.iter().any(|v| v.get_id() == 3));
    }

    /// flag ON, HNSW-indexed segment: build a real HNSW graph (mirrors the
    /// production LSM build path: `prepare_index_from_flat` →
    /// `flush_prepared_index_chunk_be` → `finalize_prepared_index_entry_be`),
    /// tombstone-delete the HNSW entry point (guaranteed reachable — it is
    /// scored first in every traversal), and confirm search never returns it.
    #[test]
    fn delete_tombstones_flag_on_hnsw_search_excludes_deleted_id() {
        let _lock = TOMBSTONE_ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::set("HELIX_LSM_DELETE_TOMBSTONES", "1");

        let seg_name = "dense__seg_000000";
        let mut ids_and_vectors = Vec::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        for id in 1u128..=24 {
            let v: [f32; 3] = [
                rand::Rng::random_range(&mut rng, -1.0f32..1.0f32),
                rand::Rng::random_range(&mut rng, -1.0f32..1.0f32),
                rand::Rng::random_range(&mut rng, -1.0f32..1.0f32),
            ];
            ids_and_vectors.push((id, v));
        }
        let (backend, manager, _dir) = setup_lsm_manager_with_flat_segment(
            "delete_tombstones_hnsw_search",
            seg_name,
            &ids_and_vectors,
        );

        let (entry_id, entry_query) = {
            let cores = manager.cores.read().unwrap();
            let core = cores.get(seg_name).unwrap();
            {
                let ro = backend.begin_read().unwrap();
                let mut prepared = core
                    .prepare_index_from_flat(&ro)
                    .unwrap()
                    .expect("prepared HNSW index over the flat corpus");
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
            assert!(
                core.has_index(&ro).unwrap(),
                "HNSW build must publish an entry point"
            );
            let entry = core.get_entry_point(&ro).unwrap();
            let entry_id = entry.get_id();
            let entry_query = core
                .get_original_vector(&ro, entry_id)
                .unwrap()
                .expect("entry point original vector");
            (entry_id, entry_query)
        };

        // Tombstone-delete the entry point through the manager's dense-delete
        // path (not directly on the core) so the plan/execute wiring is
        // exercised too.
        let plan = {
            let r = backend.begin_read().unwrap();
            manager
                .plan_delete_vectors_batch_be(&r, &[entry_id])
                .unwrap()
        };
        {
            let mut w = backend.begin_write().unwrap();
            let staged = manager
                .delete_vectors_batch_with_plan_be(&mut w, &plan)
                .unwrap();
            backend.commit(w).unwrap();
            manager.apply_delete_tombstones(&staged);
        }
        {
            let r = backend.begin_read().unwrap();
            let cores = manager.cores.read().unwrap();
            let core = cores.get(seg_name).unwrap();
            assert!(
                core.contains_id(&r, entry_id).unwrap(),
                "row must still be physically present"
            );
        }

        let r = backend.begin_read().unwrap();
        let results = manager
            .dense_search_with_id_filter_ef_be::<fn(u128) -> bool>(
                &r,
                "dense",
                &entry_query,
                24,
                None,
                false,
                None,
                None,
            )
            .unwrap();
        assert!(
            !results.iter().any(|v| v.get_id() == entry_id),
            "deleted HNSW entry point must never be a search result"
        );
        assert!(
            !results.is_empty(),
            "the other 23 live vectors must still be searchable"
        );
    }

    /// Flipping the flag OFF after tombstones exist must not resurrect them:
    /// the search gate is keyed on the in-memory set's contents, not the
    /// flag's current value.
    #[test]
    fn delete_tombstones_flag_flipped_off_still_excludes_previously_deleted() {
        let _lock = TOMBSTONE_ENV_LOCK.lock().unwrap();
        let seg_name = "dense__seg_000000";
        let (backend, manager, _dir) = {
            let _env = EnvVarGuard::set("HELIX_LSM_DELETE_TOMBSTONES", "1");
            let rig = setup_lsm_manager_with_flat_segment(
                "delete_tombstones_flip_off",
                seg_name,
                &[(1, [1.0, 0.0, 0.0]), (2, [0.0, 1.0, 0.0])],
            );
            let plan = {
                let r = rig.0.begin_read().unwrap();
                rig.1.plan_delete_vectors_batch_be(&r, &[1]).unwrap()
            };
            {
                let mut w = rig.0.begin_write().unwrap();
                let staged = rig
                    .1
                    .delete_vectors_batch_with_plan_be(&mut w, &plan)
                    .unwrap();
                rig.0.commit(w).unwrap();
                rig.1.apply_delete_tombstones(&staged);
            }
            rig
        };

        // Flag is now OFF (the `EnvVarGuard` above restored the previous
        // value on drop) — a fresh delete of id=2 must hard-delete, but the
        // earlier tombstone on id=1 must still gate search.
        assert!(!lsm_delete_tombstones_enabled());
        let plan = {
            let r = backend.begin_read().unwrap();
            manager.plan_delete_vectors_batch_be(&r, &[2]).unwrap()
        };
        {
            let mut w = backend.begin_write().unwrap();
            let staged = manager
                .delete_vectors_batch_with_plan_be(&mut w, &plan)
                .unwrap();
            backend.commit(w).unwrap();
            manager.apply_delete_tombstones(&staged);
        }

        let r = backend.begin_read().unwrap();
        let cores = manager.cores.read().unwrap();
        let core = cores.get(seg_name).unwrap();
        assert!(
            core.is_delete_tombstoned(1),
            "id 1's tombstone must survive the flag flip"
        );
        assert!(
            !core.contains_id(&r, 2).unwrap(),
            "id 2 must be hard-deleted now that the flag is off"
        );

        let results = manager
            .dense_search_with_id_filter_ef_be::<fn(u128) -> bool>(
                &r,
                "dense",
                &[1.0, 0.0, 0.0],
                5,
                None,
                false,
                None,
                None,
            )
            .unwrap();
        assert!(
            !results.iter().any(|v| v.get_id() == 1),
            "search must still exclude id 1 even though the flag is now off"
        );
    }

    /// Deleted-ratio ceiling: once a segment's delete-tombstoned fraction
    /// crosses `HELIX_LSM_DELETE_VACUUM_FRACTION`, the merge candidate
    /// selector must schedule its merge even though the space is well below
    /// the normal segment-count trigger.
    #[test]
    fn delete_tombstones_ratio_ceiling_forces_merge_candidate_below_segment_cap() {
        let _lock = TOMBSTONE_ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::set("HELIX_LSM_DELETE_TOMBSTONES", "1");
        let _fraction = EnvVarGuard::set("HELIX_LSM_DELETE_VACUUM_FRACTION", "0.2");

        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("delete-ratio-ceiling-{nanos}"))
                .expect("open in-memory LSM backend"),
        ));
        let manager = NamedVectorManager::new("delete_ratio_ceiling_test".to_string());
        manager.attach_backend(Arc::clone(&backend));
        let dir = TempDir::new().unwrap();
        let config = NamedVectorConfig {
            size: 3,
            distance: DistanceMetric::Cosine,
            spindle: SpindleConfig::default(),
        };
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));

        // Two Indexed segments (well below any segment-count merge trigger,
        // which requires 3+ by default). Segment 0 gets 4 ids, 2 of which are
        // tombstoned (50% > the 20% ceiling); segment 1 stays untouched.
        let seg_names = ["dense__seg_000000", "dense__seg_000001"];
        {
            let mut cores = manager.cores.write().unwrap();
            for name in seg_names {
                let core = manager
                    .create_dense_core_attached_lsm(dir.path(), name, &config, hnsw.clone())
                    .unwrap();
                cores.insert(name.to_string(), core);
            }
        }
        manager.dense_spaces.write().unwrap().insert(
            "dense".to_string(),
            DenseVectorSpaceMetadata {
                next_segment_id: 2,
                segments: seg_names
                    .iter()
                    .map(|name| DenseVectorSegmentMetadata {
                        physical_name: (*name).to_string(),
                        role: DenseVectorSegmentRole::Indexed,
                    })
                    .collect(),
            },
        );
        {
            let mut w = backend.begin_write().unwrap();
            let cores = manager.cores.read().unwrap();
            let seg0 = cores.get(seg_names[0]).unwrap();
            for (id, v) in [
                (1u128, [1.0f32, 0.0, 0.0]),
                (2u128, [0.0, 1.0, 0.0]),
                (3u128, [0.0, 0.0, 1.0]),
                (4u128, [1.0, 1.0, 0.0]),
            ] {
                seg0.insert_flat_be(&mut w, &v, Some(id), None).unwrap();
            }
            cores
                .get(seg_names[1])
                .unwrap()
                .insert_flat_be(&mut w, &[1.0, 1.0, 1.0], Some(5), None)
                .unwrap();
            let staged = seg0.tombstone_delete_batch_be(&mut w, &[1, 2]).unwrap();
            backend.commit(w).unwrap();
            seg0.apply_delete_tombstones(&staged);
        }

        assert_eq!(
            manager
                .cores
                .read()
                .unwrap()
                .get(seg_names[0])
                .unwrap()
                .deleted_count(),
            2
        );

        let r = backend.begin_read().unwrap();
        let candidates = manager
            .select_merge_candidates_with_limits_be(
                &r,
                "dense",
                DenseMergeCandidateLimits::new(4, 4),
            )
            .unwrap();
        let targets = candidates.expect(
            "a segment past the deleted-ratio ceiling must be selected even \
             though only 2 segments exist (well under the segment-count cap)",
        );
        assert!(targets.contains(&seg_names[0].to_string()));
    }

    /// Reader replica: the delete-tombstone set reconstructs from the durable
    /// keyspace, not just from the writer's own in-memory bookkeeping. A
    /// second `VectorCore` opened against the same backend + physical name
    /// (standing in for a reader that opened this segment after the delete)
    /// must load the tombstone via `reconstruct_delete_tombstones_be` and
    /// exclude the id from search.
    #[test]
    fn delete_tombstones_reconstruct_on_open_for_reader_replica() {
        let _lock = TOMBSTONE_ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::set("HELIX_LSM_DELETE_TOMBSTONES", "1");

        let seg_name = "dense__seg_000000";
        let (backend, manager, dir) = setup_lsm_manager_with_flat_segment(
            "delete_tombstones_reader_reconstruct",
            seg_name,
            &[(1, [1.0, 0.0, 0.0]), (2, [0.0, 1.0, 0.0])],
        );
        {
            let mut w = backend.begin_write().unwrap();
            manager
                .cores
                .read()
                .unwrap()
                .get(seg_name)
                .unwrap()
                .tombstone_delete_batch_be(&mut w, &[1])
                .unwrap();
            backend.commit(w).unwrap();
        }

        // A fresh core over the SAME backend + physical name, with an empty
        // in-memory set — stands in for a reader replica that has not yet
        // seen this delete.
        let config = NamedVectorConfig {
            size: 3,
            distance: DistanceMetric::Cosine,
            spindle: SpindleConfig::default(),
        };
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let reader_core = manager
            .create_dense_core_attached_lsm(dir.path(), seg_name, &config, hnsw)
            .unwrap();
        assert!(
            !reader_core.is_delete_tombstoned(1),
            "a freshly-opened core must not see the tombstone before reconstruction"
        );

        let r = backend.begin_read().unwrap();
        let loaded = reader_core.reconstruct_delete_tombstones_be(&r).unwrap();
        assert_eq!(loaded, 1);
        assert!(reader_core.is_delete_tombstoned(1));
        assert!(!reader_core.is_delete_tombstoned(2));
    }

    #[test]
    fn finalize_dense_tail_seals_remaining_mutable_segment() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));

        storage
            .named_vectors
            .dense_insert(
                storage.lmdb_env().unwrap(),
                &mut txn,
                "dense",
                &[1.0, 0.0, 0.0],
                1,
                HashMap::new(),
                hnsw.clone(),
                2,
            )
            .unwrap();
        let first_build = storage
            .named_vectors
            .dense_insert(
                storage.lmdb_env().unwrap(),
                &mut txn,
                "dense",
                &[0.0, 1.0, 0.0],
                2,
                HashMap::new(),
                hnsw.clone(),
                2,
            )
            .unwrap()
            .unwrap();
        storage
            .named_vectors
            .dense_insert(
                storage.lmdb_env().unwrap(),
                &mut txn,
                "dense",
                &[0.0, 0.0, 1.0],
                3,
                HashMap::new(),
                hnsw.clone(),
                2,
            )
            .unwrap();

        let tail_build = storage
            .named_vectors
            .finalize_dense_tail(storage.lmdb_env().unwrap(), &mut txn, "dense", hnsw.clone())
            .unwrap()
            .unwrap();

        assert_eq!(first_build, "dense");
        // The legacy unsuffixed root segment occupies logical id 0, so the
        // first numbered mutable segment is `__seg_000001`.
        assert_eq!(tail_build, "dense__seg_000001");

        storage
            .named_vectors
            .build_dense_segment(&mut txn, "dense", &first_build)
            .unwrap();
        storage
            .named_vectors
            .build_dense_segment(&mut txn, "dense", &tail_build)
            .unwrap();
        txn.commit().unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let stats = storage
            .named_vectors
            .get_dense_space_stats(&txn, "dense")
            .unwrap();
        assert_eq!(stats.vectors_count, 3);
        assert_eq!(stats.indexed_vectors_count, 3);

        let spaces = storage.named_vectors.list_dense_vector_spaces();
        let space = spaces.get("dense").unwrap();
        assert_eq!(
            space
                .segments
                .iter()
                .filter(|segment| segment.role == DenseVectorSegmentRole::Indexed)
                .count(),
            2
        );
        assert_eq!(
            space
                .segments
                .iter()
                .filter(|segment| segment.role == DenseVectorSegmentRole::Mutable)
                .count(),
            1
        );
    }

    #[test]
    fn dense_insert_bulk_mode_threshold_zero_stays_flat() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        storage.named_vectors.set_indexing_threshold(0);

        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let job = storage
            .named_vectors
            .dense_insert(
                storage.lmdb_env().unwrap(),
                &mut txn,
                "dense",
                &[1.0, 0.0, 0.0],
                1,
                HashMap::new(),
                hnsw,
                0,
            )
            .unwrap();
        assert!(job.is_none());
        txn.commit().unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let stats = storage
            .named_vectors
            .get_dense_space_stats(&txn, "dense")
            .unwrap();
        assert_eq!(stats.vectors_count, 1);
        assert_eq!(stats.indexed_vectors_count, 0);

        let spaces = storage.named_vectors.list_dense_vector_spaces();
        let space = spaces.get("dense").unwrap();
        assert_eq!(space.segments.len(), 1);
        assert_eq!(space.segments[0].role, DenseVectorSegmentRole::Mutable);
    }

    #[test]
    fn alloc_segment_id_reuses_gaps() {
        // Simulate: next_segment_id=5, only segments 2 and 4 active
        let mut space = DenseVectorSpaceMetadata {
            next_segment_id: 5,
            segments: vec![
                DenseVectorSegmentMetadata {
                    physical_name: "lex__seg_000002".to_string(),
                    role: DenseVectorSegmentRole::Indexed,
                },
                DenseVectorSegmentMetadata {
                    physical_name: "lex__seg_000004".to_string(),
                    role: DenseVectorSegmentRole::Mutable,
                },
            ],
        };

        // Should reuse 0 (first gap)
        assert_eq!(
            alloc_segment_id(&mut space, "lex", std::iter::empty::<&str>()),
            0
        );
        // next_segment_id unchanged since we reused
        assert_eq!(space.next_segment_id, 5);

        // Simulate adding seg 0 as active
        space.segments.push(DenseVectorSegmentMetadata {
            physical_name: "lex__seg_000000".to_string(),
            role: DenseVectorSegmentRole::Indexed,
        });

        // Should reuse 1 (next gap)
        assert_eq!(
            alloc_segment_id(&mut space, "lex", std::iter::empty::<&str>()),
            1
        );

        // Add 1 and 3 as active — now 0,1,2,3,4 all in use
        space.segments.push(DenseVectorSegmentMetadata {
            physical_name: "lex__seg_000001".to_string(),
            role: DenseVectorSegmentRole::Indexed,
        });
        space.segments.push(DenseVectorSegmentMetadata {
            physical_name: "lex__seg_000003".to_string(),
            role: DenseVectorSegmentRole::Indexed,
        });

        // No gaps — must allocate new, incrementing next_segment_id
        assert_eq!(
            alloc_segment_id(&mut space, "lex", std::iter::empty::<&str>()),
            5
        );
        assert_eq!(space.next_segment_id, 6);
    }

    #[test]
    fn alloc_segment_id_fresh_space() {
        // Fresh space: next_segment_id=0, no segments
        let mut space = DenseVectorSpaceMetadata::default();
        assert_eq!(
            alloc_segment_id(&mut space, "lex", std::iter::empty::<&str>()),
            0
        );
        assert_eq!(space.next_segment_id, 1);
        // Second alloc with no active segments still reuses 0
        // (because we didn't add seg_0 to segments)
        assert_eq!(
            alloc_segment_id(&mut space, "lex", std::iter::empty::<&str>()),
            0
        );
    }

    #[test]
    fn alloc_segment_id_treats_legacy_unsuffixed_root_as_zero() {
        let mut space = DenseVectorSpaceMetadata {
            next_segment_id: 3,
            segments: vec![
                DenseVectorSegmentMetadata {
                    physical_name: "mini".to_string(),
                    role: DenseVectorSegmentRole::Indexed,
                },
                DenseVectorSegmentMetadata {
                    physical_name: "mini__seg_000001".to_string(),
                    role: DenseVectorSegmentRole::Mutable,
                },
            ],
        };

        // The legacy base segment occupies logical id 0, so the first reusable
        // gap is 2, not 0.
        assert_eq!(
            alloc_segment_id(&mut space, "mini", std::iter::empty::<&str>()),
            2
        );
        assert_eq!(space.next_segment_id, 3);
    }

    #[test]
    fn note_missing_segment_dedups_per_tuple_within_process() {
        // Use unique strings so this test does not collide with any other
        // tuple that the process may have already recorded. `insert` returns
        // `true` on first insertion and `false` if the tuple was already
        // present — that's the contract we rely on inside `note_missing_segment`.
        let col = "test_collection_dedup_a";
        let vec = "dense_dedup_a";
        let seg = "dense_dedup_a__seg_0001";

        // First call records the tuple.
        note_missing_segment(col, vec, seg, "unit_test");
        // Second call with the same tuple must not add it again.
        note_missing_segment(col, vec, seg, "unit_test");

        let set = MISSING_SEGMENT_REGISTRY
            .lock()
            .expect("registry lock not poisoned");
        let key = (col.to_string(), vec.to_string(), seg.to_string());
        assert!(
            set.contains(&key),
            "tuple must be recorded after first observation"
        );

        // A different segment in the same collection is a new tuple.
        drop(set);
        let seg_other = "dense_dedup_a__seg_0002";
        note_missing_segment(col, vec, seg_other, "unit_test");
        let set = MISSING_SEGMENT_REGISTRY
            .lock()
            .expect("registry lock not poisoned");
        assert!(set.contains(&(col.to_string(), vec.to_string(), seg_other.to_string())));
    }
}
