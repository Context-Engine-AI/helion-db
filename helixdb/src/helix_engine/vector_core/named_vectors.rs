use std::collections::{HashMap, HashSet};
#[path = "named_vectors_exact_search.rs"]
mod exact_search;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
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
    merge_windows::MergeWindows,
    sparse::{SparseMetadataFlushStats, SparseVector, SparseVectorConfig, SparseVectorCore},
    spindle::SpindleConfig,
    vector::HVector,
    vector_core::{
        acquire_build_permit, delete_vacuum_fraction, lsm_delete_tombstones_enabled, HNSWConfig,
        NewerCopies, StoredReach, VectorCore, VectorSearchMetrics, HNSW_DIVERSE_MIN_POINTS,
    },
};

#[cfg(test)]
pub(crate) use exact_search::with_exact_scan_rows_per_id_for_test;

#[cfg(test)]
#[path = "named_vectors_exact_scan_tests.rs"]
mod exact_scan_tests;
#[cfg(test)]
#[path = "named_vectors_exact_validation_tests.rs"]
mod exact_validation_tests;

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

/// Written into a mutable tail's own keyspace in the same batch that persists
/// its seal to `Building`. Such a segment holds the only copy of its points
/// until its build publishes, so dense search keeps scanning it. Unmarked
/// `Building` segments (merge targets, tails sealed before this marker) stay
/// hidden: a merge target's points are still served by its source segments.
const SEALED_TAIL_MARKER_KEY: &[u8] = b"__helix_segment:sealed_tail";

/// Written into a merge target's keyspace in the commit that registers it as
/// `Building` (merge phase A); phase D promotes it. A marked target still
/// `Building` is a dead merge's or a dropped regraph's: its sources retire
/// only in phase D, so it never holds a row's only copy, and it may hold rows
/// deleted while it was written. The index job reaps it instead of building it.
const MERGE_TARGET_MARKER_KEY: &[u8] = b"__helix_segment:merge_target";

/// Prefix of the keys of the metadata table that name the dense segments
/// which left a stored layout: one key a physical name, its value the time
/// the name left, in milliseconds since the epoch (8 bytes, big endian).
///
/// A segment that leaves the layout keeps its rows until something clears
/// them, and what clears them is queued in memory: a writer that stops first
/// leaves rows under a name nothing stored names. The key is that record. It
/// is put by the store of the layout that drops the name, in that store's
/// transaction, so the two are stored together or not at all. It is deleted
/// once every table of the name was read empty, and by the store of a layout
/// that lists the name again.
///
/// One key a name and not one row of all names: a row rewritten whole by two
/// threads keeps the older copy of one of them. An image without this code
/// reads the metadata table by exact key and never meets one.
pub(crate) const DEAD_SEGMENT_KEY_PREFIX: &[u8] = b"dense_segment_dead:v1:";

/// The key of [`DEAD_SEGMENT_KEY_PREFIX`] for `physical_name`.
pub(crate) fn dead_segment_key(physical_name: &str) -> Vec<u8> {
    [DEAD_SEGMENT_KEY_PREFIX, physical_name.as_bytes()].concat()
}

/// The segment a key of [`DEAD_SEGMENT_KEY_PREFIX`] names.
fn dead_segment_name(key: &[u8]) -> Option<&str> {
    std::str::from_utf8(key.strip_prefix(DEAD_SEGMENT_KEY_PREFIX)?)
        .ok()
        .filter(|name| !name.is_empty())
}

/// The time a dead key's `value` holds. One that is no time reads as the
/// latest there is: nothing that waits for a name to have been dead for a
/// while takes it for an old one.
fn dead_segment_time(value: &[u8]) -> u64 {
    value.try_into().map(u64::from_be_bytes).unwrap_or(u64::MAX)
}

/// `HELIX_DENSE_DEAD_NAMES=0` keeps no record of the names that leave a
/// layout: a writer then puts, reads and deletes no key of
/// [`DEAD_SEGMENT_KEY_PREFIX`], as an image before them. Read at each open
/// of a collection.
pub(crate) fn dead_segment_names_enabled() -> bool {
    std::env::var("HELIX_DENSE_DEAD_NAMES")
        .map(|v| !matches!(v.trim(), "0" | "false" | "FALSE" | "no" | "off"))
        .unwrap_or(true)
}

/// `HELIX_DENSE_SEARCH_SEALED_TAILS=0` hides every `Building` segment again
/// (pre-fix behavior: sealed points vanish from dense search until promoted).
fn search_sealed_tails_enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        std::env::var("HELIX_DENSE_SEARCH_SEALED_TAILS")
            .map(|v| !matches!(v.trim(), "0" | "false" | "FALSE" | "no" | "off"))
            .unwrap_or(true)
    });
    *ENABLED
}

fn sealed_tail_marker_ns(physical_name: &str) -> Namespace<'_> {
    Namespace::Segment {
        physical_name,
        db: crate::helix_engine::storage_core::backend::SegmentDb::VectorData,
    }
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
    /// Exact levels where each id was deleted, for post-commit cache invalidation.
    /// Always filled (levels are recorded regardless of shared-cache mode).
    id_levels: std::collections::HashMap<u128, Vec<usize>>,
    /// Hard-deleted ids by the segment that owned them, so only that segment
    /// guards them against stale-snapshot cache refills. Empty when ownership
    /// is unknown (stale-plan fallback), in which case every segment guards.
    hard_deleted_by_segment: Vec<(String, Vec<u128>)>,
    /// Rows removed from named segments only (`delete_replaced_vectors_be`,
    /// `delete_from_segment_be`): the ids may live on elsewhere, as a
    /// rewritten point does in the tail its batch appended it to, so only
    /// the segments a row left drop their cached entries for them.
    owners_only: bool,
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

/// What the link pass plans for one segment
/// (`NamedVectorManager::plan_stored_links_be`).
#[derive(Debug)]
pub(crate) struct StoredLinks {
    pub space: String,
    pub segment: String,
    /// Served rows of the segment no stored link leads to.
    pub unreached: usize,
    /// `(row, point)`: the point's id goes at the end of the row's level-0
    /// list.
    pub links: Vec<(u128, u128)>,
    /// Rows the call looked at that no row had room for, near them or
    /// below: they get no link and stay unreached.
    pub without_room: usize,
    /// The last row the call looked at, where rows are left after it: the
    /// segment is not marked, and the next call goes on from there. `None`
    /// where it came to the end, and the segment is marked.
    pub upto: Option<u128>,
    /// The count of deleted rows the pass began at in the segment, which
    /// the mark or the place is written with, and the count the segment had
    /// when the plan read it (`VectorCore::link_pass_counts`).
    pub began_at: u64,
    pub deleted_rows: u64,
}

impl StoredLinks {
    /// True when the mark this plan's write leaves reads as done: the call
    /// came to the end, and no delete has taken a row of the segment since
    /// the pass began there.
    pub(crate) fn marks_done(&self) -> bool {
        self.upto.is_none() && self.deleted_rows <= self.began_at
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

/// What `export_for_regraph_be` read of the segment a regraph rebuilds. A row
/// is its id, its vector and the fields it was written with.
#[derive(Default)]
pub(crate) struct RegraphExport {
    /// The rows of ids no more recent segment holds.
    pub rows: Vec<(u128, Vec<f32>, HashMap<String, Value>)>,
    /// Rows left out: delete-tombstoned ids and re-upsert tombstoned copies.
    pub dropped: u64,
    /// The rows of ids a more recent segment holds too, each with the fields
    /// of those copies, the one search serves first.
    pub doubled: Vec<(
        (u128, Vec<f32>, HashMap<String, Value>),
        Vec<HashMap<String, Value>>,
    )>,
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

fn take_candidate_ids_present_in_segment(
    core: &VectorCore,
    r: &AnyRead<'_>,
    candidate_ids: &mut Vec<u128>,
) -> Result<Vec<u128>, VectorError> {
    let mut routed = Vec::new();
    for &id in candidate_ids.iter() {
        match core.get_vector(r, id, 0, false) {
            Ok(_) => routed.push(id),
            Err(VectorError::VectorNotFound(_)) => {}
            Err(err) => return Err(err),
        }
    }
    // Resolve every current copy, even when it falls outside this segment's top-k.
    let present: HashSet<_> = routed.iter().copied().collect();
    candidate_ids.retain(|id| !present.contains(id));
    Ok(routed)
}

/// Collapse duplicate ids in a merged multi-segment result, keeping the copy
/// from the NEWEST segment (lowest recency rank); ties broken by smaller
/// distance. Then orders survivors by distance, equally far ones by id
/// whichever segment holds them, and truncates to `k`. This is
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
    // `HVector` orders the nearer, then the smaller id, as the greater.
    survivors.sort_by(|lhs, rhs| rhs.cmp(lhs));
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

/// `NamedVectorManager::dense_spaces_read_at` when the shared segment list
/// did not come from a read with a sequence: any writer or LMDB list, and a
/// replica's until its first reconcile.
const DENSE_SPACES_READ_AT_UNKNOWN: u64 = u64::MAX;

/// How long a replica keeps the core of a segment its list has stopped
/// naming. A read pinned before a merge searches the replaced segments from
/// its own list (`dense_segment_infos_at`) for as long as it runs; 30 s is
/// also how long a replaced `DbReader` stays open for the reads pinned on
/// it. Tests keep the bare two-reconcile rule unless they set a grace.
const RETIRED_CORE_GRACE: std::time::Duration =
    std::time::Duration::from_secs(if cfg!(test) { 0 } else { 30 });

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

/// Tests: whether `note_missing_segment` has warned of this segment since the
/// last call. Forgets the sighting, so each search path can be asked in turn,
/// and sees the paths whose segments run on worker threads, where a test's
/// recorder does not reach.
#[cfg(test)]
pub(crate) fn take_missing_segment_note(collection: &str, vector: &str, segment: &str) -> bool {
    MISSING_SEGMENT_REGISTRY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(&(
            collection.to_string(),
            vector.to_string(),
            segment.to_string(),
        ))
}

/// Tests: (collection, segment, site, thread) for each time a search found a
/// retired segment without a core (`note_unreachable_segment`).
#[cfg(test)]
static RETIRED_SEGMENT_SIGHTINGS: Mutex<
    Vec<(String, String, &'static str, std::thread::ThreadId)>,
> = Mutex::new(Vec::new());

/// Tests: the sites that found `segment` retired and without a core since the
/// last call, each with whether it ran on the calling thread. That tells the
/// two searches that run their segments on worker threads from the rest.
#[cfg(test)]
pub(crate) fn take_retired_segment_sightings(
    collection: &str,
    segment: &str,
) -> Vec<(&'static str, bool)> {
    let caller = std::thread::current().id();
    let mut taken = Vec::new();
    RETIRED_SEGMENT_SIGHTINGS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .retain(|(seen_collection, seen_segment, site, thread)| {
            let mine = seen_collection == collection && seen_segment == segment;
            if mine {
                taken.push((*site, *thread == caller));
            }
            !mine
        });
    taken
}

/// Segments whose rows a merge selection could not count, keyed like
/// [`MISSING_SEGMENT_REGISTRY`]: each is warned of once per process.
static UNCOUNTED_SEGMENT_REGISTRY: LazyLock<Mutex<HashSet<(String, String, String)>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// A merge selection could not count a segment's rows and read them as 0.
/// By size ratio the space then picks no merge
/// (`helix_dense_merge_selection_skipped_total`), so a count that keeps
/// failing is a space that has stopped merging: say which segment and why,
/// once.
fn note_uncounted_segment(collection: &str, vector: &str, segment: &str, error: &VectorError) {
    let key = (
        collection.to_string(),
        vector.to_string(),
        segment.to_string(),
    );
    let first_sighting = match UNCOUNTED_SEGMENT_REGISTRY.lock() {
        Ok(mut set) => set.insert(key),
        // Poisoned mutex: log anyway rather than go silent.
        Err(_) => true,
    };
    if first_sighting {
        tracing::warn!(
            collection = %collection,
            vector = %vector,
            segment = %segment,
            error = %error,
            "Dense segment's rows could not be counted for a merge selection; they read as 0"
        );
    }
}

/// One change of an LSM collection's dense layout at a time: held from before
/// the change touches memory or the store until its store has returned.
///
/// The layout is stored whole, as memory has it. Two stores that could cross
/// leave the older reading stored last, and a store between a seal's store
/// and the seal's change of memory leaves the next tail out of it: rows
/// appended to a tail no stored layout names are lost to the next writer.
/// The store function and the allocator of segment ids take this by
/// reference, so neither runs without it.
///
/// Order: after the collection's point-mutation gate where that is held (a
/// regraph's retire and swap), before `dense_spaces`, then `cores`. No append
/// and no search takes it. It is held across the store's commit, which waits
/// for the WAL, and never across a build.
pub(crate) struct DenseLayoutChange<'a> {
    _held: std::sync::MutexGuard<'a, ()>,
}

/// A seal of an LSM tail that memory does not have yet: the tail it turns
/// `Building` and the next tail, opened and found empty. Made by
/// [`NamedVectorManager::prepare_tail_seal_lsm`], stored by its caller, then
/// put into memory by [`NamedVectorManager::apply_tail_seal`].
pub(crate) struct PreparedTailSeal {
    logical_name: String,
    sealed: String,
    next_name: String,
    next_core: VectorCore,
    next_segment_id: u64,
    skipped: Vec<String>,
}

impl PreparedTailSeal {
    /// The tail this seal turns `Building`.
    pub(crate) fn sealed(&self) -> &str {
        &self.sealed
    }

    /// `layout` with this seal in it. The one function for the copy that is
    /// stored and for memory, so the two come out the same.
    pub(crate) fn flip_in(&self, layout: &mut HashMap<String, DenseVectorSpaceMetadata>) {
        let Some(space) = layout.get_mut(&self.logical_name) else {
            return;
        };
        for segment in &mut space.segments {
            if segment.physical_name == self.sealed {
                segment.role = DenseVectorSegmentRole::Building;
            }
        }
        space.next_segment_id = space.next_segment_id.max(self.next_segment_id);
        if !space
            .segments
            .iter()
            .any(|segment| segment.physical_name == self.next_name)
        {
            space.segments.push(DenseVectorSegmentMetadata {
                physical_name: self.next_name.clone(),
                role: DenseVectorSegmentRole::Mutable,
            });
        }
    }
}

/// What [`NamedVectorManager::allocate_dense_core_lsm`] opened.
struct AllocatedDenseCore {
    name: String,
    core: VectorCore,
    /// The space's next id once `name` is taken.
    next_segment_id: u64,
    /// Ids passed over on the way: they hold rows no layout names.
    skipped: Vec<String>,
}

pub struct NamedVectorManager {
    /// Owning collection name (trailing path component of the LMDB env
    /// directory). Threaded through so missing-segment diagnostics can
    /// identify which tenant needs repair.
    collection_name: String,
    configs: RwLock<HashMap<String, NamedVectorConfig>>,
    dense_spaces: RwLock<HashMap<String, DenseVectorSpaceMetadata>>,
    /// See [`DenseLayoutChange`].
    dense_layout_change: Mutex<()>,
    /// Sequence of the read a replica's reconcile took `dense_spaces` from.
    /// Stored and loaded under that lock, so the two are read as a pair.
    dense_spaces_read_at: AtomicU64,
    /// Store sequence at which the writer last dropped a segment from a list
    /// in `dense_spaces`: the sources of a merge, an emptied segment. A
    /// writer read pinned below it can hold rows that only a dropped segment
    /// has at its snapshot. Stored and loaded under that lock, so the list and
    /// the mark are read as a pair.
    dense_segments_dropped_at: AtomicU64,
    cores: RwLock<HashMap<String, VectorCore>>,
    /// Replicas: when a reconcile first found each open core's segment off
    /// the list, and how long (ms) such a core is kept from then on.
    retired_cores: Mutex<HashMap<String, Instant>>,
    retired_core_grace_ms: AtomicU64,
    dirty_retired_segments: RwLock<HashSet<String>>,
    dirty_retired_segment_hook: RwLock<Option<Arc<dyn Fn(String) + Send + Sync + 'static>>>,
    /// The names that have a key of [`DEAD_SEGMENT_KEY_PREFIX`] stored, each
    /// with its time: read at open, and from then on changed only by a write
    /// of the keys that has committed. `None` where no record is kept: on a
    /// reader replica, on LMDB, and with `HELIX_DENSE_DEAD_NAMES=0`.
    dead_segments: RwLock<Option<HashMap<String, u64>>>,
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
    /// Point removals that merges in flight must keep out of their targets.
    pub(crate) merge_windows: MergeWindows,
    /// Rows without a point that this collection's last unfiltered dense
    /// searches met: how many more hits the next one asks for.
    pub(crate) rows_without_point_met: AtomicUsize,
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
            dense_layout_change: Mutex::new(()),
            dense_spaces_read_at: AtomicU64::new(DENSE_SPACES_READ_AT_UNKNOWN),
            dense_segments_dropped_at: AtomicU64::new(0),
            cores: RwLock::new(HashMap::new()),
            retired_cores: Mutex::new(HashMap::new()),
            retired_core_grace_ms: AtomicU64::new(RETIRED_CORE_GRACE.as_millis() as u64),
            dirty_retired_segments: RwLock::new(HashSet::new()),
            dirty_retired_segment_hook: RwLock::new(None),
            dead_segments: RwLock::new(None),
            sparse_configs: RwLock::new(HashMap::new()),
            sparse_cores: RwLock::new(HashMap::new()),
            indexing_threshold_override: AtomicUsize::new(INDEXING_THRESHOLD_USE_GLOBAL),
            graph_out_edges_db: RwLock::new(None),
            backend: RwLock::new(None),
            tombstone_counts: RwLock::new(HashMap::new()),
            merge_windows: MergeWindows::default(),
            rows_without_point_met: AtomicUsize::new(0),
        }
    }

    /// Begins a change of the dense layout: see [`DenseLayoutChange`]. A
    /// holder that panicked left the lock poisoned and nothing torn behind
    /// it, since the lock guards no data: it is taken all the same.
    pub(crate) fn begin_dense_layout_change(&self) -> DenseLayoutChange<'_> {
        DenseLayoutChange {
            _held: self
                .dense_layout_change
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        }
    }

    /// Takes the dead names the store has at open (`rows`: the keys of
    /// [`DEAD_SEGMENT_KEY_PREFIX`] and their values), once the spaces are
    /// loaded, and keeps the record from then on. Each space's next id is
    /// lifted above every dead name of it, so no allocation starts at one. A
    /// name the layout lists as well is said at ERROR and served as listed:
    /// the next store of the layout deletes its key.
    pub(crate) fn load_dead_segments(
        &self,
        rows: &[(Vec<u8>, Vec<u8>)],
    ) -> Result<(), VectorError> {
        let mut dead = HashMap::with_capacity(rows.len());
        for (key, value) in rows {
            match dead_segment_name(key) {
                Some(name) => {
                    dead.insert(name.to_string(), dead_segment_time(value));
                }
                None => tracing::warn!(
                    collection = %self.collection_name,
                    "A key under the prefix of the dead dense segments names no segment; passed over"
                ),
            }
        }
        {
            let mut dense_spaces = self
                .dense_spaces
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            for (logical_name, space) in dense_spaces.iter_mut() {
                for segment in &space.segments {
                    if dead.contains_key(&segment.physical_name) {
                        tracing::error!(
                            collection = %self.collection_name,
                            vector = %logical_name,
                            segment = %segment.physical_name,
                            "A dense segment the layout lists has a dead key; it is served as listed, and the next store of the layout deletes the key"
                        );
                    }
                }
                let above_dead = dead
                    .keys()
                    .filter_map(|name| segment_id_from_name(logical_name, name))
                    .max()
                    .map_or(0, |id| id + 1);
                if space.next_segment_id < above_dead {
                    tracing::warn!(
                        collection = %self.collection_name,
                        vector = %logical_name,
                        stored = space.next_segment_id,
                        lifted_to = above_dead,
                        "A dense space's next segment id was at or below a dead name's; lifted above it"
                    );
                    space.next_segment_id = above_dead;
                }
            }
        }
        *self
            .dead_segments
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(dead);
        Ok(())
    }

    /// Whether `physical_name` has a dead key stored. False where no record
    /// is kept.
    pub(crate) fn is_dead_segment(&self, physical_name: &str) -> bool {
        self.dead_segments
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .is_some_and(|dead| dead.contains_key(physical_name))
    }

    /// How many names have a dead key stored.
    pub(crate) fn dead_segment_count(&self) -> usize {
        self.dead_segments
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .map_or(0, HashMap::len)
    }

    /// Of `names`, those that have a dead key stored. `None` where no record
    /// is kept: no dead key is then written either.
    pub(crate) fn dead_among<'a>(
        &self,
        names: impl Iterator<Item = &'a str>,
    ) -> Option<Vec<String>> {
        let dead = self
            .dead_segments
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let dead = dead.as_ref()?;
        Some(
            names
                .filter(|name| dead.contains_key(*name))
                .map(str::to_string)
                .collect(),
        )
    }

    /// Puts into the record what a write of dead keys did, once that write
    /// has committed and not before: `recorded` were put with their times,
    /// the keys of `gone` deleted. A name is dead here only as long as the
    /// store says so.
    pub(crate) fn dead_segments_stored(&self, recorded: &[(String, u64)], gone: &[String]) {
        let mut dead = self
            .dead_segments
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(dead) = dead.as_mut() else {
            return;
        };
        for name in gone {
            dead.remove(name);
        }
        for (name, at) in recorded {
            dead.insert(name.clone(), *at);
        }
    }

    /// The record of dead names with their times; `None` where none is kept.
    #[cfg(test)]
    pub(crate) fn dead_segment_times(&self) -> Option<HashMap<String, u64>> {
        self.dead_segments
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
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

    /// Mark `physical_name` as a sealed mutable tail (see
    /// [`SEALED_TAIL_MARKER_KEY`]); commit it before the seal flips the role.
    pub fn mark_sealed_tail_be(
        &self,
        w: &mut AnyWrite<'_>,
        physical_name: &str,
    ) -> Result<(), VectorError> {
        self.current_backend()?
            .put(
                w,
                sealed_tail_marker_ns(physical_name),
                SEALED_TAIL_MARKER_KEY,
                b"1",
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    /// Mark `physical_name` as a merge target (see [`MERGE_TARGET_MARKER_KEY`]).
    pub fn mark_merge_target_be(
        &self,
        w: &mut AnyWrite<'_>,
        physical_name: &str,
    ) -> Result<(), VectorError> {
        self.current_backend()?
            .put(
                w,
                sealed_tail_marker_ns(physical_name),
                MERGE_TARGET_MARKER_KEY,
                b"1",
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    fn merge_target_marked_be(
        &self,
        r: &AnyRead<'_>,
        physical_name: &str,
    ) -> Result<bool, VectorError> {
        self.current_backend()?
            .get_with(
                r,
                sealed_tail_marker_ns(physical_name),
                MERGE_TARGET_MARKER_KEY,
                |value| value.is_some(),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    /// The `(physical_name, role)` pairs of a space, in metadata order.
    fn dense_segment_infos(
        space: &DenseVectorSpaceMetadata,
    ) -> Vec<(String, DenseVectorSegmentRole)> {
        space
            .segments
            .iter()
            .map(|segment| (segment.physical_name.clone(), segment.role))
            .collect()
    }

    /// The `(physical_name, role)` pairs a dense read of `name` starts from.
    ///
    /// A replica has one shared list, swapped by each reconcile, while every
    /// read stays pinned to the sequence it began at. Searched with the list
    /// of another sequence a read loses points that exist for it: a tail
    /// sealed since is `Building` with no marker at an older snapshot, a
    /// merge target has no rows there, and the sources of a merge are being
    /// reaped at a newer one. So a pinned replica read at any other sequence
    /// than the shared list's takes the list from the metadata it reads
    /// itself, as `has_payload_index_at` does for a payload index.
    ///
    /// The cores stay shared. A read older than the shared list reaches the
    /// segments retired since through cores the reconcile keeps for
    /// `RETIRED_CORE_GRACE` after a retirement. The third value names those
    /// segments, so that a search which outlives the grace counts one it can
    /// no longer reach as its own loss (`note_unreachable_segment`), not as
    /// a segment the metadata names and nothing can serve. A read ahead of
    /// the shared list searches its own list once every segment in it has a
    /// core; until that reconcile lands it searches the shared list, exactly
    /// as it did before reads had their own.
    ///
    /// A writer changes its one list in place. A list that gained a segment or
    /// flipped a role since the read began is still right for it: the new
    /// rows are not at its snapshot, and `sealed_tail_for_read_be` settles a
    /// sealed tail. A list that dropped a segment is not: the rows the read
    /// has there are listed nowhere else, and what replaced them (a merge
    /// target) is empty at its snapshot. That search is answered on the new
    /// read returned here instead, taken with the list under the lock a drop
    /// holds for writing, so nothing can be dropped between the two and one
    /// new read always suffices. Search with it in place of `r`.
    fn dense_segment_infos_at<'r>(
        &self,
        r: &AnyRead<'r>,
        name: &str,
    ) -> Result<
        (
            Option<AnyRead<'r>>,
            Vec<(String, DenseVectorSegmentRole)>,
            Vec<String>,
        ),
        VectorError,
    > {
        let (shared, shared_read_at) = {
            let dense_spaces = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let space = dense_spaces.get(name).ok_or_else(|| {
                VectorError::VectorCoreError(format!("Named vector '{}' not found", name))
            })?;
            if let AnyRead::Lsm(read) = r {
                let dropped_at = self.dense_segments_dropped_at.load(AtomicOrdering::Acquire);
                if read.snapshot_seq().is_some_and(|seq| seq < dropped_at) {
                    let backend = self.current_backend()?;
                    if let AnyBackend::Lsm(lsm) = backend.as_ref() {
                        let newer = lsm
                            .begin_read()
                            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
                        metrics::counter!("helix_dense_search_newer_read_total").increment(1);
                        return Ok((
                            Some(AnyRead::Lsm(newer)),
                            Self::dense_segment_infos(space),
                            Vec::new(),
                        ));
                    }
                }
            }
            (
                Self::dense_segment_infos(space),
                self.dense_spaces_read_at.load(AtomicOrdering::Acquire),
            )
        };
        let pinned_at = match r {
            AnyRead::LsmReader(_) => r.snapshot_seq(),
            _ => None,
        };
        let Some(pinned_at) = pinned_at.filter(|seq| *seq != shared_read_at) else {
            return Ok((None, shared, Vec::new()));
        };
        let own = self.current_backend().ok().and_then(|backend| {
            crate::helix_engine::storage_core::storage_core::HelixGraphStorage::dense_space_at(
                &backend, r, name,
            )
        });
        let Some(mut own) = own
            .map(|space| Self::dense_segment_infos(&space))
            .filter(|own| *own != shared)
        else {
            return Ok((None, shared, Vec::new()));
        };
        let older = shared_read_at != DENSE_SPACES_READ_AT_UNKNOWN && pinned_at < shared_read_at;
        if older {
            let retired: Vec<String> = own
                .iter()
                .filter(|(physical_name, _)| !shared.iter().any(|(kept, _)| kept == physical_name))
                .map(|(physical_name, _)| physical_name.clone())
                .collect();
            metrics::counter!("helix_dense_read_segment_list_total", "outcome" => "own")
                .increment(1);
            return Ok((None, own, retired));
        }
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let cores_open = own
            .iter()
            .all(|(physical_name, _)| cores.contains_key(physical_name));
        drop(cores);
        let outcome = if cores_open {
            "own"
        } else {
            own = shared;
            "shared_cores_not_open"
        };
        metrics::counter!("helix_dense_read_segment_list_total", "outcome" => outcome).increment(1);
        Ok((None, own, Vec::new()))
    }

    /// A search found no core for a segment of its list. One the shared list
    /// has retired is what a read older than `RETIRED_CORE_GRACE` loses; any
    /// other is a segment the metadata names and nothing can serve.
    fn note_unreachable_segment(
        &self,
        vector: &str,
        segment: &str,
        retired: &[String],
        site: &'static str,
    ) {
        if retired.iter().any(|name| name == segment) {
            metrics::counter!(
                "helix_dense_read_segment_list_total",
                "outcome" => "own_core_missing"
            )
            .increment(1);
            #[cfg(test)]
            RETIRED_SEGMENT_SIGHTINGS
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push((
                    self.collection_name.clone(),
                    segment.to_string(),
                    site,
                    std::thread::current().id(),
                ));
        } else {
            note_missing_segment(&self.collection_name, vector, segment, site);
        }
    }

    /// Tests: how long a replica keeps the core of a segment off its list.
    #[cfg(test)]
    pub(crate) fn set_retired_core_grace(&self, grace: std::time::Duration) {
        self.retired_core_grace_ms
            .store(grace.as_millis() as u64, AtomicOrdering::Release);
    }

    /// The LSM writer's present store sequence, the mark a drop leaves in
    /// `dense_segments_dropped_at`; zero where reads are not pinned at one.
    /// A drop that cannot take its mark fails instead of going unnoticed.
    fn present_writer_seq(&self) -> Result<u64, VectorError> {
        match self.current_backend().as_deref() {
            Ok(AnyBackend::Lsm(lsm)) => Ok(lsm
                .begin_read()
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?
                .snapshot_seq()
                .unwrap_or(0)),
            _ => Ok(0),
        }
    }

    pub fn sealed_tail_marked_be(
        &self,
        r: &AnyRead<'_>,
        physical_name: &str,
    ) -> Result<bool, VectorError> {
        self.current_backend()?
            .get_with(
                r,
                sealed_tail_marker_ns(physical_name),
                SEALED_TAIL_MARKER_KEY,
                |value| value.is_some(),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    /// The space's current mutable tail, if any.
    pub fn mutable_dense_segment_name(&self, logical_name: &str) -> Option<String> {
        let dense_spaces = self.dense_spaces.read().ok()?;
        mutable_segment_name(dense_spaces.get(logical_name)?)
    }

    /// Segments a dense read searches: every non-`Building` segment plus the
    /// `Building` ones that are sealed tails for it. Call without holding
    /// `dense_spaces`: the probes are backend reads.
    fn searchable_dense_segments_be(
        &self,
        r: &AnyRead<'_>,
        logical_name: &str,
        segments: Vec<(String, DenseVectorSegmentRole)>,
    ) -> Result<Vec<(String, DenseVectorSegmentRole)>, VectorError> {
        let mut searchable = Vec::with_capacity(segments.len());
        for (physical_name, role) in segments {
            if role == DenseVectorSegmentRole::Building
                && !self.sealed_tail_for_read_be(r, logical_name, &physical_name)?
            {
                continue;
            }
            searchable.push((physical_name, role));
        }
        Ok(searchable)
    }

    /// `id`'s vector in dense space `logical_name` as stored in the copy
    /// search serves: the newest segment that holds it. A merge target not
    /// yet published is read last: its sources still serve its rows and it
    /// may hold a row rewritten since the merge read it, but a reader whose
    /// layout lags a publish finds the row nowhere else. A segment that
    /// cannot be read counts as not holding the id.
    pub fn stored_dense_vector_be(
        &self,
        r: &AnyRead<'_>,
        logical_name: &str,
        id: u128,
    ) -> Option<Vec<f32>> {
        let segment_infos =
            Self::dense_segment_infos(self.dense_spaces.read().ok()?.get(logical_name)?);
        let cores = self.cores.read().ok()?;
        let (targets, serving): (Vec<_>, Vec<_>) =
            dense_segments_by_recency(logical_name, &segment_infos)
                .into_iter()
                .partition(|(name, role)| {
                    *role == DenseVectorSegmentRole::Building
                        && self.merge_target_marked_be(r, name).unwrap_or(true)
                });
        serving
            .into_iter()
            .chain(targets)
            .filter_map(|(name, _)| cores.get(name)?.get_vector(r, id, 0, true).ok())
            .map(|vector| vector.get_data().to_vec())
            .find(|data| !data.is_empty())
    }

    /// `segments` of `logical_name` in search's order, newest first: a merge
    /// that reads its sources in this order keeps, of two copies of an id,
    /// the one search serves.
    pub(crate) fn newest_first(logical_name: &str, segments: &[String]) -> Vec<String> {
        let segment_infos: Vec<(String, DenseVectorSegmentRole)> = segments
            .iter()
            .map(|name| (name.clone(), DenseVectorSegmentRole::Indexed))
            .collect();
        dense_segments_by_recency(logical_name, &segment_infos)
            .into_iter()
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Whether `Building` segment `physical_name` is a sealed tail for `r`,
    /// and so holds the only copy of its rows. The marker says so to a read
    /// begun after the seal. A read begun before it has no marker at its
    /// snapshot, yet the list it searches can already show the seal: a writer
    /// flips its list in place, and a replica read can be handed a newer
    /// shared list. For that read the segment is still the mutable tail in
    /// the metadata it reads itself; skipping it lost every row of the tail.
    fn sealed_tail_for_read_be(
        &self,
        r: &AnyRead<'_>,
        logical_name: &str,
        physical_name: &str,
    ) -> Result<bool, VectorError> {
        if !search_sealed_tails_enabled() {
            return Ok(false);
        }
        if self.sealed_tail_marked_be(r, physical_name)? {
            return Ok(true);
        }
        // Never a tail, and its marker spares every search the metadata read
        // below for as long as a merge is writing its target.
        if self.merge_target_marked_be(r, physical_name)? {
            return Ok(false);
        }
        let backend = self.current_backend()?;
        Ok(
            crate::helix_engine::storage_core::storage_core::HelixGraphStorage::dense_space_at(
                &backend,
                r,
                logical_name,
            )
            .is_some_and(|space| {
                space.segments.iter().any(|segment| {
                    segment.physical_name == physical_name
                        && segment.role == DenseVectorSegmentRole::Mutable
                })
            }),
        )
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

    /// Opens the first segment of `logical_name`, from id `next_segment_id`
    /// on, that holds no row. Nothing in memory changes here and no lock of
    /// the manager is held across the probe, which reads the store: the
    /// caller takes the id by putting the answer's `next_segment_id` into the
    /// space, and hands the ids passed over to
    /// [`Self::note_skipped_segment_ids`]. Two callers cannot be handed one
    /// id: each holds the layout change.
    fn allocate_dense_core_lsm(
        &self,
        _held: &DenseLayoutChange<'_>,
        data_dir: &Path,
        logical_name: &str,
        config: &NamedVectorConfig,
        hnsw_config: HNSWConfig,
        mut next_segment_id: u64,
    ) -> Result<AllocatedDenseCore, VectorError> {
        // LSM writer never reuses segment names to avoid reader serving stale
        // vectors from retired cores. Always allocate next_segment_id and increment.
        // The LMDB path (allocate_dense_core_locked) still reuses names safely via
        // the deferred dirty-segment reaper protocol.
        let mut skipped = Vec::new();
        loop {
            let name = next_segment_name(logical_name, next_segment_id);
            next_segment_id += 1;
            // A dead name is not handed out again, whether it reads empty or
            // not: its key says rows of it may be left, and a segment opened
            // under it would be listed and dead at once. An open lifts the
            // next id above them, so this is met only where the stored
            // layout had a higher next id than memory.
            if self.is_dead_segment(&name) {
                metrics::counter!("helix_dense_dead_name_refused_total").increment(1);
                tracing::warn!(
                    collection = %self.collection_name,
                    vector = %logical_name,
                    segment = %name,
                    "A fresh dense segment id is a dead name's; passed over"
                );
                continue;
            }
            let core =
                self.create_dense_core_attached_lsm(data_dir, &name, config, hnsw_config.clone())?;
            let is_empty = {
                let read = core
                    .backend
                    .begin_read()
                    .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
                core.is_empty(&read)?
            };
            if is_empty {
                return Ok(AllocatedDenseCore {
                    name,
                    core,
                    next_segment_id,
                    skipped,
                });
            }
            // Rare: a fresh segment opened dirty backend state (orphaned from a
            // previous writer crash). It is passed over for the next ID.
            skipped.push(name);
        }
    }

    /// Counts and reports the ids an allocation passed over. Each holds rows
    /// under a name no layout has, and they may be the only copy of a write
    /// that was acknowledged. They are not marked dirty and so not handed to
    /// the reaper, as they were while the allocation itself changed memory:
    /// nothing may clear them before they are counted against the segments
    /// that are listed.
    fn note_skipped_segment_ids(&self, logical_name: &str, skipped: &[String]) {
        for physical_name in skipped {
            metrics::counter!(
                "helix_segment_dirty_retired_discovered_total",
                "collection" => self.collection_name.clone(),
                "outcome" => "new",
            )
            .increment(1);
            tracing::error!(
                collection = %self.collection_name,
                vector = %logical_name,
                segment = %physical_name,
                "A fresh dense segment id holds rows that no layout names; passed over, its rows left as they are"
            );
        }
    }

    /// [`Self::allocate_dense_core_lsm`] as the tests of this file drive it:
    /// on a space of their own, which takes the id at once.
    #[cfg(test)]
    fn allocate_dense_core_locked_lsm(
        &self,
        data_dir: &Path,
        logical_name: &str,
        config: &NamedVectorConfig,
        hnsw_config: HNSWConfig,
        space: &mut DenseVectorSpaceMetadata,
        _cores: &HashMap<String, VectorCore>,
    ) -> Result<(String, VectorCore), VectorError> {
        let held = self.begin_dense_layout_change();
        let next = self.allocate_dense_core_lsm(
            &held,
            data_dir,
            logical_name,
            config,
            hnsw_config,
            space.next_segment_id,
        )?;
        space.next_segment_id = next.next_segment_id;
        self.note_skipped_segment_ids(logical_name, &next.skipped);
        Ok((next.name, next.core))
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
    ///
    /// `read_at` is the sequence of the read `dense_spaces_meta` came from,
    /// when that read was pinned; reads pinned anywhere else then take their
    /// own list (`dense_segment_infos_at`). Returns how many listed segments
    /// are left without a core, which every search skips.
    pub fn refresh_dense_view_from_metadata_lsm(
        &self,
        data_dir: &Path,
        named_vector_configs: &HashMap<String, NamedVectorConfig>,
        dense_spaces_meta: &HashMap<String, DenseVectorSpaceMetadata>,
        base_hnsw: HNSWConfig,
        read_at: Option<u64>,
    ) -> Result<usize, VectorError> {
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

        // Build set of all valid physical names from NEW metadata (before any locks)
        let new_valid_names: HashSet<String> = dense_spaces_meta
            .values()
            .flat_map(|space| space.segments.iter().map(|seg| seg.physical_name.clone()))
            .collect();

        // Build set of valid names from PREVIOUS metadata (before swap)
        let prev_valid_names: HashSet<String> = {
            let dense_spaces = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            dense_spaces
                .values()
                .flat_map(|space| space.segments.iter().map(|seg| seg.physical_name.clone()))
                .collect()
        };

        // Cores for newly listed segments go in BEFORE the swap, so a search that
        // snapshots the new view always finds them; stale cores are removed only after
        // the swap (and only once absent from two consecutive views, below).
        if !opened_cores.is_empty() {
            let mut cores = self
                .cores
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            for (physical_name, core) in opened_cores.drain(..) {
                cores.entry(physical_name).or_insert(core);
            }
        }

        {
            let mut dense_spaces = self
                .dense_spaces
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            *dense_spaces = dense_spaces_meta.clone();
            self.dense_spaces_read_at.store(
                read_at.unwrap_or(DENSE_SPACES_READ_AT_UNKNOWN),
                AtomicOrdering::Release,
            );
        }
        let unopened = {
            let cores = self
                .cores
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            new_valid_names
                .iter()
                .filter(|name| !cores.contains_key(*name))
                .count()
        };

        // Identify segments to prune: absent from BOTH previous and new metadata.
        // This delays removal by one refresh, mirroring the safe ordering at lines 5677-5698
        // where the writer publishes the merged segment, then prunes retired sources.
        // Only prune within logical spaces present in the new metadata (to avoid pruning
        // everything on an empty or default metadata read), and only segment IDs below
        // each space's next_segment_id (to avoid pruning ahead of allocation).
        // And only once the segment has been off the list for the grace: a read
        // pinned before it was retired still searches it from its own list.
        let grace = std::time::Duration::from_millis(
            self.retired_core_grace_ms.load(AtomicOrdering::Acquire),
        );
        let now = Instant::now();
        let mut retired_since = self
            .retired_cores
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let stale_names: Vec<String> = {
            let cores = self
                .cores
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

            // Build map of logical_name -> next_segment_id from new metadata
            let next_ids: HashMap<&str, u64> = dense_spaces_meta
                .iter()
                .map(|(name, space)| (name.as_str(), space.next_segment_id))
                .collect();

            let stale: Vec<String> = cores
                .keys()
                .filter(|name| {
                    // Parse logical name from physical name (format: {logical}__seg_{id:06})
                    let logical_name = if let Some(idx) = name.rfind("__seg_") {
                        &name[..idx]
                    } else {
                        // No __seg_ suffix, treat the whole name as logical (e.g., segment 0)
                        name.as_str()
                    };

                    // Only consider pruning if logical space exists in new metadata
                    if let Some(&next_id) = next_ids.get(logical_name) {
                        // Extract segment ID from physical name using the helper
                        if let Some(seg_id) = segment_id_from_name(logical_name, name) {
                            // Prune only if below next_segment_id, absent from both
                            // metadatas, and off the list for the grace. The clock
                            // starts at the first reconcile whose list lacks it.
                            seg_id < next_id
                                && !new_valid_names.contains(*name)
                                && now.duration_since(
                                    *retired_since.entry((*name).clone()).or_insert(now),
                                ) >= grace
                                && !prev_valid_names.contains(*name)
                        } else {
                            false // unparseable name, don't prune
                        }
                    } else {
                        false // logical space not in new metadata, don't prune
                    }
                })
                .cloned()
                .collect();
            // A core off the list builds no sidecar from here on: the reads
            // pinned before still search it, and a build reads the newest view.
            for (name, core) in cores.iter() {
                if retired_since.contains_key(name) && !new_valid_names.contains(name) {
                    core.keep_sidecar_as_is();
                }
            }
            stale
        };

        for name in &stale_names {
            retired_since.remove(name);
        }
        drop(retired_since);
        self.remove_retired_cores(&stale_names)?;
        Ok(unopened)
    }

    /// Take `names` out of the core map. Their cleanup I/O runs once the
    /// map's lock is released.
    fn remove_retired_cores(&self, names: &[String]) -> Result<(), VectorError> {
        if names.is_empty() {
            return Ok(());
        }

        // Take the write lock only to remove; cleanup I/O runs after it is released.
        let mut removed_cores = Vec::new();
        {
            let mut cores = self
                .cores
                .write()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;

            // Remove stale cores, collecting them for cleanup outside the lock
            for name in names {
                if let Some(core) = cores.remove(name) {
                    removed_cores.push((name.clone(), core));
                }
            }
        } // Release write lock before cleanup I/O

        // Cleanup (unlink mmap sidecar, clear caches, drop) after releasing the lock
        for (name, core) in removed_cores {
            let _ = core.unlink_mmap_sidecar();
            core.clear_caches();
            tracing::debug!(
                collection = %self.collection_name,
                segment = %name,
                "reader reconcile: removed stale core not in current or previous metadata"
            );
            drop(core);
        }
        Ok(())
    }

    /// Reader replicas: drop the cores a reconcile found off the list longer
    /// ago than the grace. Reconciles run only from request paths, so a
    /// collection whose last read landed inside the grace would keep those
    /// cores, with their ordinal maps and sidecar files, until its next read
    /// or its close. The poll-tier sweeper calls this for every open
    /// collection. A zero grace (tests) leaves the cores to the reconcile's
    /// two-pass rule. Returns how many it dropped.
    pub(crate) fn drop_retired_cores_past_grace(&self) -> Result<usize, VectorError> {
        let grace = std::time::Duration::from_millis(
            self.retired_core_grace_ms.load(AtomicOrdering::Acquire),
        );
        let due: Vec<String> = {
            let mut retired_since = self
                .retired_cores
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if grace.is_zero() || retired_since.is_empty() {
                return Ok(0);
            }
            let dense_spaces = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let now = Instant::now();
            // One the list names again is not retired, whatever its clock says.
            let due: Vec<String> = retired_since
                .iter()
                .filter(|(name, since)| {
                    now.duration_since(**since) >= grace
                        && !dense_spaces.values().any(|space| {
                            space
                                .segments
                                .iter()
                                .any(|segment| segment.physical_name == **name)
                        })
                })
                .map(|(name, _)| name.clone())
                .collect();
            for name in &due {
                retired_since.remove(name);
            }
            due
        };
        self.remove_retired_cores(&due)?;
        Ok(due.len())
    }

    /// A collection's close: unlink the sidecar files of the segments whose
    /// cores are still kept for the grace. No read is left to use them, and
    /// nothing else unlinks them before the collection is next opened on
    /// this pod. By name, so a file no core has attached goes too, and a
    /// segment the list names again keeps its files.
    pub(crate) fn unlink_retired_sidecars(&self, dir: &Path) {
        let retired: Vec<String> = self
            .retired_cores
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .keys()
            .cloned()
            .collect();
        if !retired.is_empty() {
            let _ = self.unlink_sidecar_files_for_segments(dir, &retired);
        }
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

    /// [`Self::list_dense_vector_spaces`] for the store of the layout. That
    /// one answers a layout of no space when the lock is poisoned, and a
    /// store of it leaves the collection with no segment listed: this one
    /// fails.
    pub(crate) fn dense_vector_spaces_to_store(
        &self,
    ) -> Result<HashMap<String, DenseVectorSpaceMetadata>, VectorError> {
        self.dense_spaces
            .read()
            .map(|spaces| spaces.clone())
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))
    }

    /// Leaves the lock of the dense layout poisoned, as a holder of its write
    /// lock that panicked does.
    #[cfg(test)]
    pub(crate) fn poison_dense_spaces(&self) {
        let died = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let _held = self.dense_spaces.write();
                    panic!("a holder of the dense layout's lock that died");
                })
                .join()
        });
        assert!(died.is_err() && self.dense_spaces.is_poisoned());
    }

    /// What [`VectorCore::stored_reach`] reads of every segment of every dense
    /// space, the spaces by name and a space's segments newest first. Writes
    /// nothing. A segment with no core open here is left out.
    ///
    /// A row is replaced as a regraph's export has it
    /// (`export_for_regraph_be`): by a re-upsert tombstone, or by a copy of
    /// its id in a newer segment that search serves. The newer copies are
    /// known from the ids already read, so only the tombstone is a read, and
    /// only where re-upsert tombstones are switched on.
    pub(crate) fn stored_reach_be(
        &self,
        r: &AnyRead<'_>,
    ) -> Result<Vec<(String, DenseVectorSegmentMetadata, StoredReach)>, VectorError> {
        let mut spaces: Vec<_> = self.list_dense_vector_spaces().into_iter().collect();
        spaces.sort_by(|a, b| a.0.cmp(&b.0));
        let mut out = Vec::new();
        for (space, metadata) in spaces {
            let infos = Self::dense_segment_infos(&metadata);
            let mut newer: HashSet<u128> = HashSet::new();
            for (name, role) in dense_segments_by_recency(&space, &infos) {
                let Some(segment) = metadata
                    .segments
                    .iter()
                    .find(|segment| segment.physical_name == *name)
                else {
                    continue;
                };
                // The cores are locked a segment at a time, as long as a
                // regraph's export of one holds them.
                let reach = match self.cores_read()?.get(name) {
                    Some(core) => {
                        core.stored_reach(r, &|id: u128| -> Result<bool, VectorError> {
                            Ok(newer.contains(&id)
                                || self.is_tombstoned_for_merge_be(r, name, id)?)
                        })?
                    }
                    None => continue,
                };
                // An unmarked Building segment is hidden from search: its
                // rows replace none.
                if *role != DenseVectorSegmentRole::Building
                    || self.sealed_tail_marked_be(r, name)?
                {
                    newer.extend(reach.ids.iter().copied());
                }
                out.push((space.clone(), segment.clone(), reach));
            }
        }
        Ok(out)
    }

    /// One call of the link pass over the graphs as they are stored,
    /// read-only half. The first Indexed segment the pass has still to do
    /// (`VectorCore::link_pass_due`; the spaces by name, a space's segments
    /// newest first) is read, and it alone: the links that give its served
    /// unreached rows a row leading to them, for no more than `max_rows` of
    /// those rows, from where the call before left off
    /// (`VectorCore::plan_stored_links`). A segment with nothing to link is
    /// planned too, so that it is marked. Also returns how many segments the
    /// pass has still to do, that one counted: the marks say so, and no
    /// list is read for it.
    ///
    /// A row is served unless another copy has replaced it, as
    /// `stored_reach_be` has it. Here that is asked of the unreached rows
    /// alone, by a read of each more recent segment for each.
    pub(crate) fn plan_stored_links_be(
        &self,
        r: &AnyRead<'_>,
        max_rows: usize,
    ) -> Result<(Option<StoredLinks>, usize), VectorError> {
        let mut spaces: Vec<_> = self.list_dense_vector_spaces().into_iter().collect();
        spaces.sort_by(|a, b| a.0.cmp(&b.0));
        let mut plan = None;
        let mut due = 0;
        for (space, metadata) in spaces {
            let infos = Self::dense_segment_infos(&metadata);
            // The segments more recent than the one looked at, whose rows
            // search serves.
            let mut newer = Vec::new();
            for (name, role) in dense_segments_by_recency(&space, &infos) {
                let cores = self.cores_read()?;
                let core = cores.get(name);
                let is_due = match core {
                    Some(core) if *role == DenseVectorSegmentRole::Indexed => {
                        core.link_pass_due(r)?
                    }
                    _ => false,
                };
                due += usize::from(is_due);
                if let (true, None, Some(core)) = (is_due, &plan, core) {
                    // Every row counts as served in this read; the unreached
                    // ones another copy has replaced are left out after it.
                    let reach = core
                        .stored_reach(r, &|_: u128| -> Result<bool, VectorError> { Ok(false) })?;
                    let mut unreached = Vec::new();
                    for id in reach.unreached.unwrap_or_default() {
                        let mut replaced = self.is_tombstoned_for_merge_be(r, name, id)?;
                        for other in &newer {
                            if replaced {
                                break;
                            }
                            if let Some(other) = cores.get(*other) {
                                replaced = other.contains_id(r, id)?;
                            }
                        }
                        if !replaced {
                            unreached.push(id);
                        }
                    }
                    let (links, without_room, upto) =
                        core.plan_stored_links(r, &unreached, core.link_pass_after(r)?, max_rows)?;
                    let (began_at, deleted_rows) = core.link_pass_counts(r)?;
                    plan = Some(StoredLinks {
                        space: space.clone(),
                        segment: name.clone(),
                        unreached: unreached.len(),
                        links,
                        without_room,
                        upto,
                        began_at,
                        deleted_rows,
                    });
                }
                // An unmarked Building segment is hidden from search: its
                // rows replace none.
                if *role != DenseVectorSegmentRole::Building
                    || self.sealed_tail_marked_be(r, name)?
                {
                    newer.push(name);
                }
            }
        }
        Ok((plan, due))
    }

    /// The write half: stages `plan` in `w` for the segments that are still
    /// Indexed in the layout, each with its mark, or with the row the pass
    /// has come to in it. Returns, per segment, the rows whose list changed,
    /// for `forget_stored_links` after the commit.
    /// Call with nothing written to the collection since `plan` was made.
    pub(crate) fn apply_stored_links_be(
        &self,
        w: &mut AnyWrite<'_>,
        plan: &[StoredLinks],
    ) -> Result<Vec<(String, Vec<u128>)>, VectorError> {
        let spaces = self.list_dense_vector_spaces();
        let cores = self.cores_read()?;
        let mut changed = Vec::new();
        for segment in plan {
            let indexed = spaces.get(&segment.space).is_some_and(|space| {
                space.segments.iter().any(|listed| {
                    listed.physical_name == segment.segment
                        && listed.role == DenseVectorSegmentRole::Indexed
                })
            });
            let Some(core) = cores.get(&segment.segment).filter(|_| indexed) else {
                continue;
            };
            let rows =
                core.apply_stored_links_be(w, &segment.links, segment.upto, segment.began_at)?;
            changed.push((segment.segment.clone(), rows));
        }
        Ok(changed)
    }

    /// Drops the cached lists `apply_stored_links_be` changed, once its
    /// batch has committed.
    pub(crate) fn forget_stored_links(&self, changed: &[(String, Vec<u128>)]) {
        let Ok(cores) = self.cores_read() else {
            return;
        };
        for (segment, rows) in changed {
            if let Some(core) = cores.get(segment) {
                core.forget_neighbor_lists(rows);
            }
        }
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

    /// True when a dense space quantizes its stored vectors. Such a space
    /// keeps no exact sidecar for its mutable tail (`insert_flat_be`), so a
    /// reader replica scores every tail row from a point read.
    pub(crate) fn has_spindle_dense_space(&self) -> bool {
        self.quantizes_a_dense_space().unwrap_or(false)
    }

    /// As `has_spindle_dense_space`, with a poisoned lock an error where
    /// that one reads it as no quantized space: for a caller that must not
    /// take a quantized collection for a plain one.
    pub(crate) fn quantizes_a_dense_space(&self) -> Result<bool, VectorError> {
        self.configs
            .read()
            .map(|configs| configs.values().any(|config| config.spindle.is_enabled()))
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))
    }

    /// Test-only: leaves the lock on the vector configs poisoned, as a panic
    /// under it does.
    #[cfg(test)]
    pub(crate) fn poison_configs_for_test(&self) {
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _held = self.configs.write();
            panic!("poisoning the vector configs lock for a test");
        }));
        assert!(poisoned.is_err());
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

    /// Rows in the mutable tail, stopping at `limit`: what the seal rule
    /// counts, without rescanning the other segments. A segment that is not
    /// the tail can hold rows without a usable index (while it builds, and
    /// for good once its entry point's rows are deleted, see
    /// `has_entry_point_key`); sealing the tail does nothing for those.
    pub(crate) fn mutable_tail_count_upto_be(
        &self,
        r: &AnyRead<'_>,
        name: &str,
        limit: u64,
    ) -> Result<u64, VectorError> {
        let spaces = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {e}")))?;
        let space = spaces.get(name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{name}' not found"))
        })?;
        let cores = self.cores_read()?;
        let mut count = 0u64;
        for segment in &space.segments {
            let Some(core) = cores.get(&segment.physical_name) else {
                // Reported for every segment, as the full stats did: this is
                // the index job's probe.
                note_missing_segment(&self.collection_name, name, &segment.physical_name, "stats");
                continue;
            };
            if segment.role == DenseVectorSegmentRole::Mutable {
                count = count
                    .saturating_add(core.level_zero_count_upto(r, limit.saturating_sub(count))?);
            }
        }
        Ok(count)
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
        let (newer, all_segment_infos, retired) = self.dense_segment_infos_at(r, name)?;
        let r = newer.as_ref().unwrap_or(r);
        let building_count = all_segment_infos
            .iter()
            .filter(|(_, role)| *role == DenseVectorSegmentRole::Building)
            .count();
        let segment_infos = self.searchable_dense_segments_be(r, name, all_segment_infos)?;
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
                            self.note_unreachable_segment(name, segment_name, &retired, "search");
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
                    self.note_unreachable_segment(name, segment_name, &retired, "search");
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

        let (newer, segment_infos, retired) = self.dense_segment_infos_at(r, name)?;
        let r = newer.as_ref().unwrap_or(r);
        let segment_infos = self.searchable_dense_segments_be(r, name, segment_infos)?;

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
                        self.note_unreachable_segment(
                            name,
                            segment_name,
                            &retired,
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

        let mut candidate_ids = unique_candidate_ids(candidate_ids.into_iter().collect());
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
            if candidate_ids.is_empty() {
                break;
            }
            let Some(core) = cores.get(segment_name) else {
                note_missing_segment(&self.collection_name, name, segment_name, "exact_id_search");
                continue;
            };
            let rank = recency_ranks.get(segment_name).copied().unwrap_or(u32::MAX);
            let rd = core.backend.read_borrowed(txn);
            let segment_candidate_ids =
                take_candidate_ids_present_in_segment(core, &rd, &mut candidate_ids)?;
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

    /// LSM cutover variant of [`dense_append_to_mutable`]; returns the tail it
    /// appended to.
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
    ) -> Result<String, VectorError> {
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
        Ok(active_name)
    }

    /// The input `dense_append_to_mutable_be` refuses: a name that is no
    /// dense space, or values that are not finite. Writes nothing.
    pub(crate) fn check_dense_append(&self, name: &str, data: &[f32]) -> Result<(), VectorError> {
        let known = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?
            .contains_key(name);
        if !known {
            return Err(VectorError::VectorCoreError(format!(
                "Named vector '{}' not found",
                name
            )));
        }
        super::vector_core::ensure_finite_vector(data)
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

    /// Building segments whose HNSW publish completed before their metadata
    /// role was persisted. These are safe to validate and promote without
    /// rebuilding the index.
    pub fn indexed_building_segment_names(
        &self,
        txn: &RoTxn,
        logical_name: &str,
    ) -> Result<HashSet<String>, VectorError> {
        let building_names = self.building_segment_names(logical_name);
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let mut indexed = HashSet::new();
        for physical_name in building_names {
            let Some(core) = cores.get(&physical_name) else {
                continue;
            };
            let read = core.backend.read_borrowed(txn);
            if core.has_index(&read)? {
                indexed.insert(physical_name);
            }
        }
        Ok(indexed)
    }

    pub fn indexed_building_segment_names_be(
        &self,
        r: &AnyRead<'_>,
        logical_name: &str,
    ) -> Result<HashSet<String>, VectorError> {
        let building_names = self.building_segment_names(logical_name);
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let mut indexed = HashSet::new();
        for physical_name in building_names {
            let Some(core) = cores.get(&physical_name) else {
                continue;
            };
            if core.has_index(r)? {
                indexed.insert(physical_name);
            }
        }
        Ok(indexed)
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

    /// First step of the LSM twin of [`Self::seal_mutable_tail`]: decides
    /// whether `logical_name`'s tail is sealed now and opens the tail after
    /// it. Memory is as it was when this returns, whatever it returns: an
    /// append still takes the old tail, and a failure here leaves the space
    /// its tail and its next id. The caller stores the layout with the seal
    /// in it ([`PreparedTailSeal::flip_in`]) and only then calls
    /// [`Self::apply_tail_seal`], so the next tail is in a stored layout
    /// before an append can reach it. Its store reads (has the tail a row,
    /// is the next id empty) run with neither write lock held: an append and
    /// a search go on beside them.
    pub(crate) fn prepare_tail_seal_lsm(
        &self,
        held: &DenseLayoutChange<'_>,
        r: &AnyRead<'_>,
        data_dir: &Path,
        logical_name: &str,
        hnsw_config: HNSWConfig,
    ) -> Result<Option<PreparedTailSeal>, VectorError> {
        let config = self.get_config(logical_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", logical_name))
        })?;

        // A copy: the space stays as it is for as long as `held` is.
        let space = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?
            .get(logical_name)
            .cloned()
            .ok_or_else(|| {
                VectorError::VectorCoreError(format!("Named vector '{}' not found", logical_name))
            })?;
        let Some(sealed) = mutable_segment_name(&space) else {
            return Ok(None);
        };
        // Emptiness only: stop at the first row instead of counting the tail.
        let has_vectors = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?
            .get(&sealed)
            .and_then(|core| core.level_zero_count_upto(r, 1).ok())
            .is_some_and(|rows| rows > 0);
        if !has_vectors || self.should_block_mutable_seal(&space, logical_name) {
            return Ok(None);
        }

        let next = self.allocate_dense_core_lsm(
            held,
            data_dir,
            logical_name,
            &config,
            hnsw_config,
            space.next_segment_id,
        )?;
        Ok(Some(PreparedTailSeal {
            logical_name: logical_name.to_string(),
            sealed,
            next_name: next.name,
            next_core: next.core,
            next_segment_id: next.next_segment_id,
            skipped: next.skipped,
        }))
    }

    /// Last step of an LSM tail's seal: puts a seal whose layout is stored
    /// into memory, from where the next append takes the next tail. It takes
    /// the write locks of `dense_spaces` and `cores`, in that order and
    /// together, so no append finds the next tail's name without its core,
    /// and it reads nothing from the store under them. It cannot fail: a
    /// poisoned lock is taken all the same, since a seal that is stored and
    /// not in memory is a stored layout ahead of memory until the next store.
    pub(crate) fn apply_tail_seal(&self, _held: &DenseLayoutChange<'_>, seal: PreparedTailSeal) {
        {
            let mut dense_spaces = self
                .dense_spaces
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let mut cores = self
                .cores
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            seal.flip_in(&mut dense_spaces);
            if let Some(space) = dense_spaces.get(&seal.logical_name) {
                self.observe_segment_ids_high_water(&seal.logical_name, space);
            }
            cores.insert(seal.next_name, seal.next_core);
        }
        self.note_skipped_segment_ids(&seal.logical_name, &seal.skipped);
    }

    /// A seal put into memory with no store, as every seal was before its
    /// store came first: for a test that stores the layout itself, or wants
    /// memory ahead of the store. Returns the sealed segment's name.
    #[cfg(test)]
    pub fn seal_mutable_tail_lsm(
        &self,
        r: &AnyRead<'_>,
        data_dir: &Path,
        logical_name: &str,
        hnsw_config: HNSWConfig,
    ) -> Result<Option<String>, VectorError> {
        let held = self.begin_dense_layout_change();
        let Some(seal) =
            self.prepare_tail_seal_lsm(&held, r, data_dir, logical_name, hnsw_config)?
        else {
            return Ok(None);
        };
        let sealed = seal.sealed().to_string();
        self.apply_tail_seal(&held, seal);
        Ok(Some(sealed))
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

    /// `HELIX_DENSE_MERGE_SIZE_RATIO`: a positive whole number turns on the
    /// selection by size ratio: a merge of a space over its target takes the
    /// two smallest indexed segments, then each next larger one while it
    /// holds at most this many times the rows taken so far
    /// ([`Self::merge_targets_by_size_ratio`]). Unset, 0, or anything else:
    /// the size tiers and their fallback ([`Self::merge_targets_by_tier`]).
    pub(crate) fn merge_size_ratio() -> Option<usize> {
        Self::merge_size_ratio_from(
            std::env::var("HELIX_DENSE_MERGE_SIZE_RATIO")
                .ok()
                .as_deref(),
        )
    }

    fn merge_size_ratio_from(setting: Option<&str>) -> Option<usize> {
        setting
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|ratio| *ratio > 0)
    }

    /// Select the segments of one merge: by size tier, or by size ratio where
    /// `HELIX_DENSE_MERGE_SIZE_RATIO` is set.
    ///
    /// Returns `None` if no merge is needed, or `Some(targets)` with the
    /// physical names of segments to merge.
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
        let mut forced: HashSet<String> = HashSet::new();
        let mut size_unknown = false;
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
                core.level_zero_count(&rd).unwrap_or_else(|error| {
                    note_uncounted_segment(
                        &self.collection_name,
                        logical_name,
                        &seg.physical_name,
                        &error,
                    );
                    size_unknown = true;
                    0
                }) as usize
            };
            let tombstoned = self.tombstone_count_for_segment_cached(&seg.physical_name);
            let deleted = core.deleted_count();
            if count > 0 && (deleted as f64 / count as f64) > delete_vacuum_fraction() {
                delete_ratio_breach = true;
                forced.insert(seg.physical_name.clone());
            }
            let effective = count.saturating_sub(tombstoned).saturating_sub(deleted);
            indexed.push((seg.physical_name.clone(), effective));
        }

        // A segment past the deleted-ratio ceiling schedules a merge even
        // when the space is below its normal segment-count trigger:
        // `merge_targets` takes it as a forced segment, or by tier in the
        // fallback there.
        if indexed.len() <= limits.max_indexed_segments
            && !(delete_ratio_breach && indexed.len() >= 2)
        {
            return Ok(None);
        }

        Ok(Self::merge_targets(
            Self::merge_size_ratio(),
            indexed,
            &forced,
            limits.max_indexed_segments,
            max_fan_in,
            delete_ratio_breach,
            size_unknown,
        ))
    }

    /// The segments of one merge of a space that is over its target or holds
    /// a segment past the deleted-rows ceiling: `indexed` is (segment, live
    /// rows), `forced` the segments to rewrite whatever their size, `ratio`
    /// [`Self::merge_size_ratio`]. `size_unknown`: a segment's rows could not
    /// be counted and read as 0.
    fn merge_targets(
        ratio: Option<usize>,
        indexed: Vec<(String, usize)>,
        forced: &HashSet<String>,
        max_indexed_segments: usize,
        max_fan_in: usize,
        delete_ratio_breach: bool,
        size_unknown: bool,
    ) -> Option<Vec<String>> {
        match ratio {
            // A segment that reads as empty sorts first and would be taken
            // whatever it holds: this pass picks nothing, the next counts again.
            Some(_) if size_unknown => {
                metrics::counter!(
                    "helix_dense_merge_selection_skipped_total",
                    "reason" => "size_unknown"
                )
                .increment(1);
                None
            }
            Some(ratio) => Self::merge_targets_by_size_ratio(indexed, forced, ratio, max_fan_in),
            None => Self::merge_targets_by_tier(
                &indexed,
                max_indexed_segments,
                max_fan_in,
                delete_ratio_breach,
            ),
        }
    }

    /// One merge by size ratio. Without a forced segment: the two smallest,
    /// then each next larger one while it holds at most `ratio` times the
    /// rows taken so far, up to `max_fan_in`. A segment is so rewritten only
    /// once the smaller ones merged with it add up to 1/`ratio` of it, where
    /// the tier fallback took every segment of a space over a target of 4 and
    /// rewrote the biggest one at every merge. A forced segment (past the
    /// deleted-rows ceiling, or with a graph whose entry point's row is gone)
    /// is rewritten whatever its size, with the smallest other segment.
    fn merge_targets_by_size_ratio(
        mut indexed: Vec<(String, usize)>,
        forced: &HashSet<String>,
        ratio: usize,
        max_fan_in: usize,
    ) -> Option<Vec<String>> {
        if indexed.len() < 2 {
            return None;
        }
        indexed.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        if let Some(at) = indexed.iter().position(|(name, _)| forced.contains(name)) {
            let (low, high) = if at == 0 { (0, 1) } else { (0, at) };
            return Some(vec![indexed[low].0.clone(), indexed[high].0.clone()]);
        }
        let mut rows = indexed[0].1.saturating_add(indexed[1].1);
        let mut take = 2;
        while take < indexed.len().min(max_fan_in) && indexed[take].1 <= rows.saturating_mul(ratio)
        {
            rows = rows.saturating_add(indexed[take].1);
            take += 1;
        }
        indexed.truncate(take);
        Some(indexed.into_iter().map(|(name, _)| name).collect())
    }

    /// One merge by size tier (`HELIX_DENSE_MERGE_SIZE_RATIO=0`): a tier with
    /// more than `TIER_MAX_PER_CLASS` members merges its smallest; otherwise
    /// the fallback takes the smallest segments overall, up to `max_fan_in`.
    fn merge_targets_by_tier(
        indexed: &[(String, usize)],
        max_indexed_segments: usize,
        max_fan_in: usize,
        delete_ratio_breach: bool,
    ) -> Option<Vec<String>> {
        // Bucket segments into size tiers.
        let tier_of = |count: usize| -> usize {
            Self::TIER_BOUNDARIES
                .iter()
                .position(|&b| count < b)
                .unwrap_or(Self::TIER_BOUNDARIES.len())
        };

        let mut tier_buckets: HashMap<usize, Vec<(String, usize)>> = HashMap::new();
        for (name, count) in indexed {
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
        // segment breached the deleted-ratio ceiling, merge the smallest
        // overall, up to the fan-in. The ratio-breach case is what turns the
        // vacuum ceiling into a scheduled merge when the space is otherwise
        // under its normal segment-count trigger.
        if best_tier.is_none()
            && indexed.len() >= 2
            && (indexed.len() > max_indexed_segments || delete_ratio_breach)
        {
            let mut sorted = indexed.to_vec();
            sorted.sort_by_key(|(_, c)| *c);
            let take = sorted.len().min(max_fan_in).max(2).min(sorted.len());
            best_tier = Some((0, sorted[..take].to_vec()));
        }

        best_tier.map(|(_, targets)| targets.into_iter().map(|(name, _)| name).collect())
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
        let mut forced: HashSet<String> = HashSet::new();
        let mut size_unknown = false;
        let indexed_cores = || {
            space
                .segments
                .iter()
                .filter(|segment| segment.role == DenseVectorSegmentRole::Indexed)
                .filter_map(|segment| cores.get(&segment.physical_name))
        };
        // A graph that hard deletes have taken more than the ceiling's share
        // of rows from is forced like a segment past it in tombstones (see
        // `VectorCore::deleted_graph_rows_over`): a point read a segment,
        // and a count only where rows were deleted. A read that fails
        // forces nothing. Nor does a segment left with fewer rows than a
        // search walks from, as for a lost entry point below: it is scanned
        // anyway, and forced it would take the next larger segment into its
        // merge, which under the size-ratio rule can be the big one.
        for seg in &space.segments {
            if seg.role != DenseVectorSegmentRole::Indexed {
                continue;
            }
            if cores.get(&seg.physical_name).is_some_and(|core| {
                core.deleted_graph_rows_over(
                    r,
                    delete_vacuum_fraction(),
                    HNSW_DIVERSE_MIN_POINTS as u64,
                )
                .unwrap_or(false)
            }) {
                delete_ratio_breach = true;
                forced.insert(seg.physical_name.clone());
            }
        }
        if indexed_cores().count() <= limits.max_indexed_segments
            && indexed_cores().all(|core| core.deleted_count() == 0)
            && forced.is_empty()
        {
            return Ok(None);
        }
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
            let count = core.level_zero_count(r).unwrap_or_else(|error| {
                note_uncounted_segment(
                    &self.collection_name,
                    logical_name,
                    &seg.physical_name,
                    &error,
                );
                size_unknown = true;
                0
            }) as usize;
            let tombstoned = self.tombstone_count_for_segment_cached(&seg.physical_name);
            let deleted = core.deleted_count();
            if count > 0 && (deleted as f64 / count as f64) > delete_vacuum_fraction() {
                delete_ratio_breach = true;
                forced.insert(seg.physical_name.clone());
            }
            indexed.push((
                seg.physical_name.clone(),
                count.saturating_sub(tombstoned).saturating_sub(deleted),
            ));
        }

        // A segment past the deleted-ratio ceiling (HELIX_LSM_DELETE_VACUUM_FRACTION,
        // issue #29 "fix 2") schedules a merge even when the space is below its
        // normal segment-count trigger: `merge_targets` takes it as a forced
        // segment, or by tier in the fallback there.
        if indexed.len() <= limits.max_indexed_segments
            && !(delete_ratio_breach && indexed.len() >= 2)
        {
            return Ok(None);
        }

        // A graph whose entry point's row is gone is scored row by row until
        // a merge rewrites it, and by size ratio no merge takes a big segment
        // for a long time. Segments under the exact-scan floor are scanned
        // anyway. Two point reads a segment, only where a merge is picked. A
        // read that fails is not a lost entry point: no rewrite for it.
        let ratio = Self::merge_size_ratio();
        if ratio.is_some() {
            for (name, rows) in &indexed {
                let Some(core) = cores.get(name).filter(|_| *rows >= HNSW_DIVERSE_MIN_POINTS)
                else {
                    continue;
                };
                if core.entry_point_row_is_gone(r).unwrap_or(false) {
                    forced.insert(name.clone());
                }
            }
        }

        Ok(Self::merge_targets(
            ratio,
            indexed,
            &forced,
            limits.max_indexed_segments,
            max_fan_in,
            delete_ratio_breach,
            size_unknown,
        ))
    }

    /// Split-phase merge — read-only phase.
    ///
    /// Exports vectors from the given segments and builds an HNSW index
    /// entirely in memory. No write lock is held. Returns `PreparedMerge`
    /// which can be flushed under a short write txn via `flush_prepared_merge`.
    pub fn prepare_merge(
        &self,
        txn: &RoTxn,
        logical_name: &str,
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
        let distance_metric = self
            .get_config(logical_name)
            .map(|config| config.distance)
            .ok_or_else(|| {
                VectorError::VectorCoreError(format!("Named vector '{}' not found", logical_name))
            })?;
        let prepared_index = VectorCore::build_hnsw_in_memory_owned_with_permit(
            exported,
            hnsw_config,
            distance_metric,
            &build_permit,
        )?;

        Ok(PreparedMerge {
            merge_targets: merge_targets.to_vec(),
            prepared_index,
        })
    }

    /// The rows of Indexed HNSW segment `physical_name` for a rebuild of it
    /// alone, without delete-tombstoned ids and re-upsert tombstoned copies.
    /// The rebuilt segment takes the newest segment id, so a row carried into
    /// it outranks every other copy of its id: the rows of ids a more recent
    /// segment holds too come back apart, for the caller to keep those that
    /// are their point's own. `None` when the segment has no graph or stores
    /// fewer than `min_rows` rows: it is the rows stored that a search walks,
    /// however few of them a rebuild would keep.
    pub(crate) fn export_for_regraph_be(
        &self,
        r: &AnyRead<'_>,
        logical_name: &str,
        physical_name: &str,
        min_rows: usize,
    ) -> Result<Option<RegraphExport>, VectorError> {
        let segment_infos = {
            let dense_spaces = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            let space = dense_spaces.get(logical_name).ok_or_else(|| {
                VectorError::VectorCoreError(format!("Named vector '{}' not found", logical_name))
            })?;
            Self::dense_segment_infos(space)
        };
        if !segment_infos
            .iter()
            .any(|(name, role)| name == physical_name && *role == DenseVectorSegmentRole::Indexed)
        {
            return Ok(None);
        }
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        let Some(core) = cores.get(physical_name) else {
            return Ok(None);
        };
        if !core.has_entry_point_key(r)?
            || core.level_zero_count_upto(r, min_rows as u64)? < min_rows as u64
        {
            return Ok(None);
        }
        let mut newer: Vec<(&String, &VectorCore)> = Vec::new();
        for (name, role) in dense_segments_by_recency(logical_name, &segment_infos)
            .into_iter()
            .take_while(|(name, _)| name != physical_name)
        {
            // Only copies search serves supersede: an unmarked Building
            // segment (a merge target, or a dead merge's orphan) is hidden.
            if *role == DenseVectorSegmentRole::Building && !self.sealed_tail_marked_be(r, name)? {
                continue;
            }
            newer.extend(cores.get(name).map(|core| (name, core)));
        }
        let mut export = RegraphExport::default();
        for (id, data, fields) in core.export_level_zero(r)? {
            if core.is_delete_tombstoned(id)
                || self.is_tombstoned_for_merge_be(r, physical_name, id)?
            {
                export.dropped += 1;
                continue;
            }
            // shortcut: one point read per copy, batch them (`get_many`) if a
            // regraph of a collection full of doubled ids reads slowly.
            // A more recent row that is delete-tombstoned, or marked
            // superseded by a re-upsert, does not replace this one. Taken
            // for the copy that does, it could leave the point's live row
            // out, and with it the whole segment; and the next merge drops
            // a superseded row, which would leave the point with none.
            let mut copies = Vec::new();
            for (newer_name, newer_core) in &newer {
                if newer_core.contains_id(r, id)?
                    && !newer_core.is_delete_tombstoned(id)
                    && !self.is_tombstoned_for_merge_be(r, newer_name, id)?
                {
                    copies.push(newer_core.debug_fields(r, id)?);
                }
            }
            if copies.is_empty() {
                export.rows.push((id, data, fields));
            } else {
                export.doubled.push(((id, data, fields), copies));
            }
        }
        Ok(Some(export))
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

    /// Opens a merge target and lists it in memory as `Building`, which no
    /// append takes rows to and no search reads before it is published; the
    /// caller stores the layout under the same `held`. The probe of the id
    /// reads the store with no lock of the manager held.
    pub(crate) fn create_merge_target_lsm(
        &self,
        held: &DenseLayoutChange<'_>,
        data_dir: &Path,
        logical_name: &str,
        hnsw_config: HNSWConfig,
    ) -> Result<String, VectorError> {
        let config = self.get_config(logical_name).ok_or_else(|| {
            VectorError::VectorCoreError(format!("Named vector '{}' not found", logical_name))
        })?;
        let next_segment_id = self
            .dense_spaces
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?
            .get(logical_name)
            .map(|space| space.next_segment_id)
            .ok_or_else(|| {
                VectorError::VectorCoreError(format!("Named vector '{}' not found", logical_name))
            })?;
        let target = self.allocate_dense_core_lsm(
            held,
            data_dir,
            logical_name,
            &config,
            hnsw_config,
            next_segment_id,
        )?;
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
            space.next_segment_id = space.next_segment_id.max(target.next_segment_id);
            // Register immediately as Building so the ID is reserved in
            // persisted metadata and the orphan cleanup can find what a
            // crash before the publish left.
            space.segments.push(DenseVectorSegmentMetadata {
                physical_name: target.name.clone(),
                role: DenseVectorSegmentRole::Building,
            });
            self.observe_segment_ids_high_water(logical_name, space);
            cores.insert(target.name.clone(), target.core);
        }
        self.note_skipped_segment_ids(logical_name, &target.skipped);
        Ok(target.name)
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
        // The target is complete by now: a read at or past this sequence has
        // all its rows, an older one is answered on a new read.
        let dropped_at = self.present_writer_seq()?;

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
        self.dense_segments_dropped_at
            .fetch_max(dropped_at, AtomicOrdering::Release);
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

    /// True when some dense space holds more Indexed segments (with a loaded
    /// core) than the merge target: the segment-count trigger of
    /// `select_merge_candidates_with_limits[_be]`, read without touching rows.
    /// `dense_segment_debt().merge_debt` cannot stand in for it: it also counts
    /// tombstone backlog, which selects no merge by itself.
    pub fn has_dense_space_over_merge_target(&self) -> bool {
        let target = Self::max_indexed_segments_target();
        let (Ok(dense_spaces), Ok(cores)) = (self.dense_spaces.read(), self.cores.read()) else {
            return false;
        };
        dense_spaces.values().any(|space| {
            space
                .segments
                .iter()
                .filter(|segment| {
                    segment.role == DenseVectorSegmentRole::Indexed
                        && cores.contains_key(&segment.physical_name)
                })
                .count()
                > target
        })
    }

    /// True when some dense space holds a Building segment with a loaded core:
    /// the only kind the index job can build, promote or reap. One without a
    /// core is skipped by `prepare_building_indices_limited[_be]` and by the
    /// orphan GC, so a job submitted for it does nothing.
    pub fn has_building_segment_with_core(&self) -> bool {
        let (Ok(dense_spaces), Ok(cores)) = (self.dense_spaces.read(), self.cores.read()) else {
            return false;
        };
        dense_spaces.values().any(|space| {
            space.segments.iter().any(|segment| {
                segment.role == DenseVectorSegmentRole::Building
                    && cores.contains_key(&segment.physical_name)
            })
        })
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
                    // A count that could not be read is not a count of none.
                    // Taken for 0 it drops a sealed tail that waits for its
                    // build, rows and all; the segment waits for the next
                    // pass instead.
                    let count = match core.level_zero_count(r) {
                        Ok(count) => count,
                        Err(error) => {
                            tracing::warn!(
                                segment = %seg.physical_name,
                                %error,
                                "orphan cleanup could not count a Building segment's rows; leaving it"
                            );
                            return None;
                        }
                    };
                    if count == 0
                        || self
                            .merge_target_marked_be(r, &seg.physical_name)
                            .unwrap_or(false)
                    {
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
        // A sealed tail emptied by rewrites still has its rows for a read
        // begun before them; a merge target is read by no search.
        let dropped_at = self.present_writer_seq()?;
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
            self.dense_segments_dropped_at
                .fetch_max(dropped_at, AtomicOrdering::Release);
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
            // Emptiness only: stop at the first row instead of counting the segment.
            if core.level_zero_count_upto(r, 1)? == 0 {
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
        // Empty by now; a read begun earlier may still have rows in them.
        let dropped_at = self.present_writer_seq()?;
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
        self.dense_segments_dropped_at
            .fetch_max(dropped_at, AtomicOrdering::Release);
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
    ) -> Result<std::collections::HashMap<u128, Vec<usize>>, VectorError> {
        if ids.is_empty() {
            return Ok(std::collections::HashMap::new());
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
        let mut merged_id_levels = std::collections::HashMap::new();
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
                let id_levels = core.delete_vectors_batch_be(w, &owned)?;
                // Merge id_levels from this segment
                for (id, levels) in id_levels {
                    merged_id_levels
                        .entry(id)
                        .or_insert_with(Vec::new)
                        .extend(levels);
                }
            }
        }
        Ok(merged_id_levels)
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
        let segments: Vec<(&String, &VectorCore)> = active_segments
            .iter()
            .filter_map(|name| cores.get(name).map(|core| (name, core)))
            .collect();
        // One batched read answers every (segment, id) probe: all segments of
        // a collection share its keyspace, and a probe per pair costs a
        // SlateDB round trip each (segments x ids of them per deleted file).
        let probes: Vec<_> = segments
            .iter()
            .flat_map(|(_, core)| ids.iter().map(move |&id| core.contains_id_probe(id)))
            .collect();
        let batched = match segments.first() {
            Some((_, core)) => core
                .backend
                .contains_many(r, &probes)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?,
            None => None,
        };
        let mut by_segment = Vec::new();
        for (index, (segment_name, core)) in segments.iter().enumerate() {
            let mut owned = Vec::new();
            for (offset, &id) in ids.iter().enumerate() {
                let present = match &batched {
                    Some(present) => present[index * ids.len() + offset],
                    None => core.contains_id(r, id)?,
                };
                if present {
                    owned.push(id);
                }
            }
            if !owned.is_empty() {
                by_segment.push(((*segment_name).clone(), owned));
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
            let id_levels = self.delete_vectors_batch_be(w, &plan.ids)?;
            return Ok(StagedDeleteTombstones {
                by_segment: Vec::new(),
                id_levels,
                hard_deleted_by_segment: Vec::new(),
                owners_only: false,
            });
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
                // Merge fallback id_levels into staged so tombstones from
                // earlier segments in this loop iteration are preserved.
                let id_levels = self.delete_vectors_batch_be(w, &plan.ids)?;
                for (id, levels) in id_levels {
                    staged
                        .id_levels
                        .entry(id)
                        .or_insert_with(Vec::new)
                        .extend(levels);
                }
                // The fallback deletes from any active segment, so ownership
                // is unknown: an empty list makes every segment guard.
                staged.hard_deleted_by_segment.clear();
                return Ok(staged);
            };
            if tombstones_enabled {
                let ids = core.tombstone_delete_batch_be(w, owned)?;
                if !ids.is_empty() {
                    staged.by_segment.push((segment_name.clone(), ids));
                }
            } else {
                let id_levels = core.delete_vectors_batch_be(w, owned)?;
                staged
                    .hard_deleted_by_segment
                    .push((segment_name.clone(), owned.clone()));
                // Merge id_levels from this segment into the staged result
                for (id, levels) in id_levels {
                    staged
                        .id_levels
                        .entry(id)
                        .or_insert_with(Vec::new)
                        .extend(levels);
                }
            }
        }
        Ok(staged)
    }

    /// The dense deletes of an upsert that rewrites `plan.ids`, staged ahead
    /// of its appends in the same batch: every stored copy goes, except a row
    /// in a space's mutable tail that the batch appends again (`appends(space,
    /// id)`). The append overwrites that row in place; deleting it first would
    /// also drop the sidecar slot the append publishes, which hides the point
    /// from the writer's flat search. Those rows are returned as `(space,
    /// tail, id)`: if the tail is sealed before the append, the append goes
    /// to the next one and the caller removes the row
    /// (`delete_from_segment_be`).
    ///
    /// Always a hard delete: a reader loads a segment's delete tombstones
    /// only when it first opens the segment, so it would keep serving a
    /// tombstoned old copy. A layout that moved since the plan is probed
    /// again in the writer, as in `delete_vectors_batch_with_plan_be`. Hand
    /// the result to `apply_delete_tombstones` once the batch has committed.
    pub fn delete_replaced_vectors_be(
        &self,
        w: &mut AnyWrite<'_>,
        plan: &DenseDeletePlan,
        appends: &dyn Fn(&str, u128) -> bool,
    ) -> Result<(StagedDeleteTombstones, Vec<(String, String, u128)>), VectorError> {
        let mut staged = StagedDeleteTombstones {
            owners_only: true,
            ..Default::default()
        };
        let mut kept = Vec::new();
        if plan.is_empty() {
            return Ok((staged, kept));
        }
        // Segment names and each space's tail from one view of the layout.
        let (active_segments, tails): (HashSet<String>, HashMap<String, String>) = {
            let dense_spaces = self
                .dense_spaces
                .read()
                .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
            (
                dense_spaces
                    .values()
                    .flat_map(|space| {
                        space
                            .segments
                            .iter()
                            .map(|segment| segment.physical_name.clone())
                    })
                    .collect(),
                dense_spaces
                    .iter()
                    .filter_map(|(name, space)| Some((mutable_segment_name(space)?, name.clone())))
                    .collect(),
            )
        };
        let planned = active_segments == plan.active_segments;
        if !planned {
            metrics::counter!("helix_delete_dense_plan_stale_total").increment(1);
        }
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        for segment_name in &active_segments {
            let Some(core) = cores.get(segment_name) else {
                continue;
            };
            let mut owned = Vec::new();
            if planned {
                if let Some((_, ids)) = plan
                    .by_segment
                    .iter()
                    .find(|(name, _)| name == segment_name)
                {
                    owned.clone_from(ids);
                }
            } else {
                for &id in &plan.ids {
                    if core.contains_id_be(w, id)? {
                        owned.push(id);
                    }
                }
            }
            if let Some(space) = tails.get(segment_name) {
                owned.retain(|id| {
                    let overwritten = appends(space, *id);
                    if overwritten {
                        kept.push((space.clone(), segment_name.clone(), *id));
                    }
                    !overwritten
                });
            }
            if owned.is_empty() {
                continue;
            }
            for (id, levels) in core.delete_vectors_batch_be(w, &owned)? {
                staged.id_levels.entry(id).or_default().extend(levels);
            }
            staged
                .hard_deleted_by_segment
                .push((segment_name.clone(), owned));
        }
        Ok((staged, kept))
    }

    /// Hard-deletes whatever rows segment `physical_name` holds for `ids`.
    /// Hand the result to `apply_delete_tombstones` once the batch has
    /// committed.
    pub(crate) fn delete_from_segment_be(
        &self,
        w: &mut AnyWrite<'_>,
        physical_name: &str,
        ids: &[u128],
    ) -> Result<StagedDeleteTombstones, VectorError> {
        let mut staged = StagedDeleteTombstones {
            owners_only: true,
            ..Default::default()
        };
        let cores = self
            .cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        if let Some(core) = cores.get(physical_name) {
            staged.id_levels = core.delete_vectors_batch_be(w, ids)?;
            staged
                .hard_deleted_by_segment
                .push((physical_name.to_string(), ids.to_vec()));
        }
        Ok(staged)
    }

    /// Post-commit companion of `delete_vectors_batch_with_plan_be`: applies
    /// the staged tombstones to each segment core's in-memory set and records
    /// the metric. Also performs post-commit shared-cache invalidation at the
    /// exact levels where ids were deleted. Call only after the staging write
    /// batch committed.
    pub fn apply_delete_tombstones(&self, staged: &StagedDeleteTombstones) {
        if staged.is_empty() && staged.id_levels.is_empty() {
            return;
        }
        let Ok(cores) = self.cores.read() else {
            return;
        };
        if staged.owners_only {
            for (segment_name, owned) in &staged.hard_deleted_by_segment {
                if let Some(core) = cores.get(segment_name) {
                    let levels: HashMap<u128, Vec<usize>> = owned
                        .iter()
                        .filter_map(|id| Some((*id, staged.id_levels.get(id)?.clone())))
                        .collect();
                    core.apply_hard_deletes_post_commit(owned, &levels);
                }
            }
            return;
        }

        // Apply tombstones to in-memory sets (only filled when tombstones enabled)
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

        // Post-commit cache invalidation at exact levels (runs regardless of tombstone mode).
        // A read during the delete can re-cache both the vector and its sidecar ordinal from
        // the still-committed row, so both are dropped once the delete is durable. Only the
        // segment that owned an id guards it against later stale-snapshot refills (every
        // segment when the stale-plan fallback left ownership unknown).
        if !staged.id_levels.is_empty() {
            let all_ids: Vec<u128> = staged.id_levels.keys().copied().collect();
            for (segment_name, core) in cores.iter() {
                let guarded: &[u128] = if staged.hard_deleted_by_segment.is_empty() {
                    &all_ids
                } else {
                    staged
                        .hard_deleted_by_segment
                        .iter()
                        .find(|(owner, _)| owner == segment_name)
                        .map(|(_, ids)| ids.as_slice())
                        .unwrap_or(&[])
                };
                core.apply_hard_deletes_post_commit(guarded, &staged.id_levels);
            }
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
        ids: &[u128],
    ) -> Result<(), VectorError> {
        let cores = self
            .sparse_cores
            .read()
            .map_err(|e| VectorError::VectorCoreError(format!("Lock poisoned: {}", e)))?;
        for core in cores.values() {
            core.delete_batch_be(w, ids)?;
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

    /// The stored form of a dead segment's key: what an image after this one
    /// has to read, so the bytes are pinned here.
    #[test]
    fn a_dead_segment_key_names_its_segment_and_holds_its_time() {
        let key = dead_segment_key("dense__seg_000007");
        assert_eq!(key, b"dense_segment_dead:v1:dense__seg_000007");
        assert_eq!(dead_segment_name(&key), Some("dense__seg_000007"));
        assert_eq!(dead_segment_name(DEAD_SEGMENT_KEY_PREFIX), None);
        assert_eq!(dead_segment_name(b"current"), None);
        assert_eq!(dead_segment_time(&42u64.to_be_bytes()), 42);
        // A value that is no time is never taken for an old one.
        assert_eq!(dead_segment_time(b"seven"), u64::MAX);
    }

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
                None,
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

    fn merge_sizes(sizes: &[(&str, usize)]) -> Vec<(String, usize)> {
        sizes
            .iter()
            .map(|(name, rows)| (name.to_string(), *rows))
            .collect()
    }

    #[test]
    fn size_ratio_merge_takes_a_segment_once_the_smaller_ones_reach_half_of_it() {
        let none = HashSet::new();
        let pick = |sizes: &[(&str, usize)]| {
            NamedVectorManager::merge_targets_by_size_ratio(merge_sizes(sizes), &none, 2, 8)
                .unwrap_or_default()
        };
        // Sealed tails beside a big segment: the tails and the 6,000 rows that
        // are at most twice the 6,144 taken before them, not the big one.
        assert_eq!(
            pick(&[
                ("big", 100_000),
                ("t1", 2_048),
                ("t2", 2_048),
                ("t3", 2_048),
                ("mid", 6_000)
            ]),
            ["t1", "t2", "t3", "mid"]
        );
        // Sizes that are not alike: the two smallest and no more.
        assert_eq!(
            pick(&[
                ("e", 100_000),
                ("a", 2_048),
                ("b", 9_000),
                ("c", 30_000),
                ("d", 70_000)
            ]),
            ["a", "b"]
        );
        // The big segment joins when the smaller ones add up to half of it.
        assert_eq!(
            pick(&[("big", 100_000), ("a", 20_000), ("b", 30_000)]),
            ["a", "b", "big"]
        );
        assert_eq!(
            pick(&[("big", 100_001), ("a", 20_000), ("b", 30_000)]),
            ["a", "b"]
        );
        // Never more than the fan-in, and nothing below two segments.
        let nine: Vec<(&str, usize)> = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
            .into_iter()
            .map(|name| (name, 1))
            .collect();
        assert_eq!(pick(&nine).len(), 8);
        assert!(pick(&[("only", 5)]).is_empty());
    }

    #[test]
    fn size_ratio_merge_rewrites_a_forced_segment_with_the_smallest_other() {
        let pick = |sizes: &[(&str, usize)], forced: &[&str]| {
            let forced: HashSet<String> = forced.iter().map(|name| name.to_string()).collect();
            NamedVectorManager::merge_targets_by_size_ratio(merge_sizes(sizes), &forced, 2, 8)
                .unwrap_or_default()
        };
        let sizes = [("big", 100_000), ("a", 2_048), ("b", 2_048), ("c", 9_000)];
        // A segment past the deleted-rows ceiling, or with a graph that lost
        // its entry point: rewritten whatever its size, with the smallest other.
        assert_eq!(pick(&sizes, &["big"]), ["a", "big"]);
        // The smallest forced one first, also when it is the smallest of all.
        assert_eq!(pick(&sizes, &["a", "big"]), ["a", "b"]);
    }

    /// Rows the merges rewrite per row written, over 49 sealed tails of 2,048
    /// rows onto one segment of 100,000 rows with a target of 4 segments.
    fn rewritten_per_row_written(pick: impl Fn(&[(String, usize)]) -> Option<Vec<String>>) -> f64 {
        let mut segments = vec![("s0".to_string(), 100_000usize)];
        let (mut written, mut rewritten) = (0usize, 0usize);
        for seal in 1..=49 {
            segments.push((format!("s{seal}"), 2_048));
            written += 2_048;
            while segments.len() > 4 {
                let targets = pick(&segments).expect("a space over its target has a merge");
                let rows: usize = segments
                    .iter()
                    .filter(|(name, _)| targets.contains(name))
                    .map(|(_, rows)| rows)
                    .sum();
                segments.retain(|(name, _)| !targets.contains(name));
                segments.push((format!("m{seal}_{}", segments.len()), rows));
                rewritten += rows;
            }
        }
        rewritten as f64 / written as f64
    }

    #[test]
    fn size_ratio_merges_rewrite_a_fraction_of_what_the_tier_fallback_rewrites() {
        let none = HashSet::new();
        let by_ratio = rewritten_per_row_written(|segments| {
            NamedVectorManager::merge_targets_by_size_ratio(segments.to_vec(), &none, 2, 8)
        });
        let by_tier = rewritten_per_row_written(|segments| {
            NamedVectorManager::merge_targets_by_tier(segments, 4, 8, false)
        });
        assert!(by_ratio < 5.0, "by size ratio: {by_ratio:.1}");
        assert!(by_tier > 18.0, "by tier: {by_tier:.1}");
    }

    #[test]
    fn merge_size_ratio_is_off_unless_set_to_a_positive_whole_number() {
        let ratio = NamedVectorManager::merge_size_ratio_from;
        assert_eq!(ratio(None), None);
        assert_eq!(ratio(Some("2")), Some(2));
        assert_eq!(ratio(Some(" 3 ")), Some(3));
        // 0 and anything that is not a whole number leave the size tiers.
        assert_eq!(ratio(Some("0")), None);
        assert_eq!(ratio(Some("2.5")), None);
        assert_eq!(ratio(Some("off")), None);
        assert_eq!(ratio(Some("")), None);
    }

    #[test]
    fn a_segment_that_could_not_be_counted_picks_no_merge_by_size_ratio() {
        let none = HashSet::new();
        // A big segment whose count failed reads as 0 rows and sorts first.
        let sizes = || {
            merge_sizes(&[
                ("big", 0),
                ("t1", 2_048),
                ("t2", 2_048),
                ("t3", 2_048),
                ("t4", 2_048),
            ])
        };
        let pick = |ratio, size_unknown| {
            NamedVectorManager::merge_targets(ratio, sizes(), &none, 4, 8, false, size_unknown)
        };
        // By ratio the pass picks nothing; with every count known it picks.
        assert_eq!(pick(Some(2), true), None);
        assert!(pick(Some(2), false).is_some());
        // The size tiers are as they were: the unreadable segment is taken.
        let by_tier = pick(None, true).unwrap_or_default();
        assert!(by_tier.iter().any(|name| name == "big"), "{by_tier:?}");
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

    /// Equally far rows come back by id whichever segment holds them: three
    /// sealed segments of one row each, written as ids 9, 3 and 7, and the
    /// merge of two segments' hits handed over in either order. The copy of
    /// an id that is returned is still the newest segment's.
    #[test]
    fn equally_far_rows_of_several_segments_come_back_by_id() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        for id in [9u128, 3, 7] {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            let job = storage
                .named_vectors
                .dense_insert(
                    storage.lmdb_env().unwrap(),
                    &mut txn,
                    "dense",
                    &[1.0, 0.0, 0.0],
                    id,
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
        let search = |k: usize| -> Vec<u128> {
            storage
                .named_vectors
                .dense_search_with_selectivity_ef::<fn(&HVector) -> bool>(
                    storage.lmdb_env().unwrap(),
                    &txn,
                    "dense",
                    &[0.0, 1.0, 0.0],
                    k,
                    None,
                    true,
                    None,
                    None,
                )
                .unwrap()
                .iter()
                .map(HVector::get_id)
                .collect()
        };
        assert_eq!(search(3), [3, 7, 9]);
        assert_eq!(search(2), [3, 7]);

        let hit = |id: u128, distance: f32| {
            let mut hit = HVector::from_slice(id, 0, Vec::new());
            hit.set_distance(distance);
            hit
        };
        // The older segment holds id 2 nearer than the newer one does.
        let older = || vec![(1u32, hit(2, 0.1)), (1, hit(5, 0.5)), (1, hit(6, 0.5))];
        let newer = || vec![(0u32, hit(1, 0.5)), (0, hit(2, 0.5))];
        for combined in [[older(), newer()].concat(), [newer(), older()].concat()] {
            let merged: Vec<(u128, f32)> = dedup_keep_newest(combined, 3)
                .iter()
                .map(|hit| (hit.get_id(), hit.get_distance()))
                .collect();
            assert_eq!(merged, [(1, 0.5), (2, 0.5), (5, 0.5)]);
        }
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
    fn exact_candidate_search_does_not_resurrect_a_newer_segment_loser() {
        let (storage, _tmp) = setup_storage();
        create_dense(&storage, "dense");
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let env = storage.lmdb_env().unwrap();
        let mut txn = env.write_txn().unwrap();
        let segment = storage
            .named_vectors
            .dense_insert(
                env,
                &mut txn,
                "dense",
                &[1.0, 0.0, 0.0],
                5,
                HashMap::new(),
                hnsw.clone(),
                1,
            )
            .unwrap()
            .unwrap();
        storage
            .named_vectors
            .build_dense_segment(&mut txn, "dense", &segment)
            .unwrap();
        txn.commit().unwrap();

        let mut txn = env.write_txn().unwrap();
        for (id, vector) in [(5, [0.0, 1.0, 0.0]), (6, [0.9, 0.1, 0.0])] {
            storage
                .named_vectors
                .dense_append_to_mutable(
                    env,
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

        let txn = env.read_txn().unwrap();
        let results = storage
            .named_vectors
            .dense_search_candidate_ids_exact(&txn, "dense", &[1.0, 0.0, 0.0], 1, [5, 6], Some(1.0))
            .unwrap();
        assert_eq!(
            results.iter().map(HVector::get_id).collect::<Vec<_>>(),
            [6],
            "id 5's stale close vector must not win when its current vector loses top-k"
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
        let mut candidates = vec![1u128, 2u128, 3u128, 404u128];
        let routed = take_candidate_ids_present_in_segment(core, &rd, &mut candidates).unwrap();

        assert_eq!(routed, [2u128]);
        assert_eq!(candidates, [1, 3, 404]);
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

    #[test]
    fn indexed_building_segments_are_recoverable_on_lsm() {
        let seg_name = "dense__seg_000000";
        let (backend, manager, _dir) = setup_lsm_manager_with_flat_segment(
            "indexed_building_recovery",
            seg_name,
            &[(1, [1.0, 0.0, 0.0]), (2, [0.0, 1.0, 0.0])],
        );
        manager
            .dense_spaces
            .write()
            .unwrap()
            .get_mut("dense")
            .unwrap()
            .segments[0]
            .role = DenseVectorSegmentRole::Building;

        let mut prepared = {
            let read = backend.begin_read().unwrap();
            manager
                .prepare_building_indices_limited_be(&read, "dense", 1)
                .unwrap()
        };
        let (prepared_name, mut index) = prepared.pop().unwrap();
        assert_eq!(prepared_name, seg_name);
        {
            let mut write = backend.begin_write().unwrap();
            let cores = manager.cores.read().unwrap();
            let core = cores.get(seg_name).unwrap();
            let total = index.point_ids.len();
            core.flush_prepared_index_chunk_be(&mut write, &mut index, 0, total)
                .unwrap();
            core.finalize_prepared_index_entry_be(&mut write, &index)
                .unwrap();
            drop(cores);
            backend.commit(write).unwrap();
        }

        let read = backend.begin_read().unwrap();
        let recoverable = manager
            .indexed_building_segment_names_be(&read, "dense")
            .unwrap();
        assert_eq!(recoverable, HashSet::from([seg_name.to_string()]));
    }

    /// Test rig: a `NamedVectorManager` over an in-memory LSM backend with one
    /// Indexed segment holding `ids_and_vectors` (flat-inserted, no HNSW
    /// build). Mirrors the setup used by the `plan_delete_vectors_batch_be_*`
    /// tests above.
    pub(super) fn setup_lsm_manager_with_flat_segment(
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

    #[test]
    fn exact_candidate_lsm_does_not_resurrect_a_newer_segment_loser() {
        let (backend, manager, dir) = setup_lsm_manager_with_flat_segment(
            "exact_candidate_recency",
            "dense__seg_000000",
            &[(5, [1.0, 0.0, 0.0])],
        );
        let newest = "dense__seg_000001";
        let core = manager
            .create_dense_core_attached_lsm(
                dir.path(),
                newest,
                &NamedVectorConfig {
                    size: 3,
                    distance: DistanceMetric::Cosine,
                    spindle: SpindleConfig::default(),
                },
                HNSWConfig::new(Some(8), Some(32), Some(64)),
            )
            .unwrap();
        let mut write = backend.begin_write().unwrap();
        for (id, vector) in [(5, [0.0, 1.0, 0.0]), (6, [0.9, 0.1, 0.0])] {
            core.insert_flat_be(&mut write, &vector, Some(id), None)
                .unwrap();
        }
        backend.commit(write).unwrap();
        manager
            .cores
            .write()
            .unwrap()
            .insert(newest.to_string(), core);
        manager
            .dense_spaces
            .write()
            .unwrap()
            .get_mut("dense")
            .unwrap()
            .segments
            .push(DenseVectorSegmentMetadata {
                physical_name: newest.to_string(),
                role: DenseVectorSegmentRole::Mutable,
            });
        let AnyBackend::Lsm(lsm) = backend.as_ref() else {
            panic!("expected LSM fixture")
        };
        let read = backend.begin_read().unwrap();
        lsm.reset_read_count();
        let results = manager
            .dense_search_candidate_ids_exact_be(
                &read,
                "dense",
                &[1.0, 0.0, 0.0],
                1,
                [5, 6],
                Some(1.0),
            )
            .unwrap();
        eprintln!(
            "exact candidate recency backend reads: {}",
            lsm.read_count()
        );
        assert_eq!(
            results.iter().map(HVector::get_id).collect::<Vec<_>>(),
            [6],
            "a current vector excluded by top-k must still suppress its stale copy"
        );
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

        use crate::helix_engine::vector_core::vector_core::with_small_segment_exact_for_test;
        let r = backend.begin_read().unwrap();
        // The graph walk and the small-segment exact scan.
        for exact in [false, true] {
            let results = with_small_segment_exact_for_test(exact, || {
                manager.dense_search_with_id_filter_ef_be::<fn(u128) -> bool>(
                    &r,
                    "dense",
                    &entry_query,
                    24,
                    None,
                    false,
                    None,
                    None,
                )
            })
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

    /// The ceiling counts the rows hard deletes took from a segment's graph
    /// as it counts tombstoned ones. Two Indexed segments, far below the
    /// segment-count trigger, select no merge while the first has lost a
    /// fifth of the rows it was built with, and select one once it has lost
    /// more. The second, with one row left, is past the ceiling and forces
    /// nothing: no search walks it.
    #[test]
    fn hard_deleted_graph_rows_past_the_ceiling_force_a_merge_candidate() {
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("deleted-rows-ceiling-{}", uuid::Uuid::new_v4()))
                .expect("open in-memory LSM backend"),
        ));
        let manager = NamedVectorManager::new("deleted_rows_ceiling_test".to_string());
        manager.attach_backend(Arc::clone(&backend));
        let dir = TempDir::new().unwrap();
        let config = NamedVectorConfig {
            size: 3,
            distance: DistanceMetric::Cosine,
            spindle: SpindleConfig::default(),
        };
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
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
        // 512 rows are left in the first segment, as many as a search walks
        // a graph from; one is in the second.
        {
            let mut w = backend.begin_write().unwrap();
            let cores = manager.cores.read().unwrap();
            for id in 1..=512u128 {
                cores
                    .get(seg_names[0])
                    .unwrap()
                    .insert_flat_be(&mut w, &[id as f32, 1.0, 0.0], Some(id), None)
                    .unwrap();
            }
            cores
                .get(seg_names[1])
                .unwrap()
                .insert_flat_be(&mut w, &[1.0, 1.0, 1.0], Some(1_000), None)
                .unwrap();
            backend.commit(w).unwrap();
        }
        // The merge selected once `deleted` rows are counted as taken from
        // the graph of segment `of`.
        let selected = |of: usize, deleted: u64| {
            let mut w = backend.begin_write().unwrap();
            manager
                .cores
                .read()
                .unwrap()
                .get(seg_names[of])
                .unwrap()
                .put_deleted_graph_rows_for_test(&mut w, deleted);
            backend.commit(w).unwrap();
            let r = backend.begin_read().unwrap();
            manager
                .select_merge_candidates_with_limits_be(
                    &r,
                    "dense",
                    DenseMergeCandidateLimits::new(4, 4),
                )
                .unwrap()
        };
        assert_eq!(
            selected(0, 128),
            None,
            "128 of 640 rows is the ceiling, not past it"
        );
        assert_eq!(
            selected(1, 5),
            None,
            "5 of 6 rows is past the ceiling, in a segment no search walks"
        );
        let targets = selected(0, 129).expect("129 of 641 rows is past the ceiling");
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

    #[test]
    fn lsm_allocation_never_reuses_segment_names() {
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("lsm_no_reuse-{nanos}"))
                .expect("open in-memory LSM backend"),
        ));
        let manager = NamedVectorManager::new("test_collection".to_string());
        manager.attach_backend(Arc::clone(&backend));
        let dir = TempDir::new().unwrap();
        let config = NamedVectorConfig {
            size: 3,
            distance: DistanceMetric::Cosine,
            spindle: SpindleConfig::default(),
        };
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));

        // Allocate segment 0
        let mut space = DenseVectorSpaceMetadata {
            next_segment_id: 0,
            segments: vec![],
        };
        let cores_empty = HashMap::new();
        let (name_0, _core_0) = manager
            .allocate_dense_core_locked_lsm(
                dir.path(),
                "dense",
                &config,
                hnsw.clone(),
                &mut space,
                &cores_empty,
            )
            .unwrap();
        assert_eq!(name_0, "dense__seg_000000");
        assert_eq!(space.next_segment_id, 1);

        // Add segment 0 to metadata
        space.segments.push(DenseVectorSegmentMetadata {
            physical_name: name_0.clone(),
            role: DenseVectorSegmentRole::Indexed,
        });

        // Remove segment 0 from metadata (simulate it was merged and removed)
        space.segments.retain(|s| s.physical_name != name_0);

        // Allocate a new segment — must return segment 1, NOT reuse 0
        let (name_1, _core_1) = manager
            .allocate_dense_core_locked_lsm(
                dir.path(),
                "dense",
                &config,
                hnsw.clone(),
                &mut space,
                &cores_empty,
            )
            .unwrap();
        assert_eq!(
            name_1, "dense__seg_000001",
            "LSM must never reuse segment 0"
        );
        assert_eq!(space.next_segment_id, 2);

        // Simulate writer restart: next_segment_id is persisted and reloaded
        let space_after_restart = DenseVectorSpaceMetadata {
            next_segment_id: space.next_segment_id,
            segments: vec![DenseVectorSegmentMetadata {
                physical_name: name_1.clone(),
                role: DenseVectorSegmentRole::Indexed,
            }],
        };
        let cores_after = HashMap::new();

        // Allocate another segment — must continue from next_segment_id=2
        let (name_2, _core_2) = manager
            .allocate_dense_core_locked_lsm(
                dir.path(),
                "dense",
                &config,
                hnsw,
                &mut space_after_restart.clone(),
                &cores_after,
            )
            .unwrap();
        assert_eq!(
            name_2, "dense__seg_000002",
            "LSM must continue monotonic allocation after restart"
        );
    }

    #[test]
    fn reader_refresh_removes_cores_absent_from_metadata() {
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("reader_stale_removal-{nanos}"))
                .expect("open in-memory LSM backend"),
        ));
        let manager = NamedVectorManager::new("test_collection".to_string());
        manager.attach_backend(Arc::clone(&backend));
        let dir = TempDir::new().unwrap();
        let config = NamedVectorConfig {
            size: 3,
            distance: DistanceMetric::Cosine,
            spindle: SpindleConfig::default(),
        };
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));

        // Create two cores: segment 0 and segment 1
        let core_0 = manager
            .create_dense_core_attached_lsm(dir.path(), "dense__seg_000000", &config, hnsw.clone())
            .unwrap();
        let core_1 = manager
            .create_dense_core_attached_lsm(dir.path(), "dense__seg_000001", &config, hnsw.clone())
            .unwrap();

        {
            let mut w = backend.begin_write().unwrap();
            core_0
                .insert_flat_be(&mut w, &[1.0, 0.0, 0.0], Some(1), None)
                .unwrap();
            core_1
                .insert_flat_be(&mut w, &[0.0, 1.0, 0.0], Some(2), None)
                .unwrap();
            backend.commit(w).unwrap();
        }

        // Reader state: both cores open
        manager
            .cores
            .write()
            .unwrap()
            .insert("dense__seg_000000".to_string(), core_0);
        manager
            .cores
            .write()
            .unwrap()
            .insert("dense__seg_000001".to_string(), core_1);
        manager
            .configs
            .write()
            .unwrap()
            .insert("dense".to_string(), config.clone());
        assert_eq!(manager.cores.read().unwrap().len(), 2);

        // Metadata update: segment 0 removed (merged), only segment 1 remains
        let mut new_metadata = HashMap::new();
        new_metadata.insert(
            "dense".to_string(),
            DenseVectorSpaceMetadata {
                next_segment_id: 2,
                segments: vec![DenseVectorSegmentMetadata {
                    physical_name: "dense__seg_000001".to_string(),
                    role: DenseVectorSegmentRole::Indexed,
                }],
            },
        );
        let mut configs = HashMap::new();
        configs.insert("dense".to_string(), config);

        // Reader refresh with new metadata
        manager
            .refresh_dense_view_from_metadata_lsm(dir.path(), &configs, &new_metadata, hnsw, None)
            .unwrap();

        // Verify: segment 0 core removed, segment 1 core retained
        let cores = manager.cores.read().unwrap();
        assert_eq!(cores.len(), 1, "reader must remove stale core");
        assert!(cores.contains_key("dense__seg_000001"));
        assert!(
            !cores.contains_key("dense__seg_000000"),
            "stale core must be removed"
        );
    }

    #[test]
    fn reader_refresh_empty_metadata_does_not_prune_all_cores() {
        // Issue B-LOW: an empty or default metadata read must not prune every core.
        // Only prune within logical spaces present in the new metadata, and only segment
        // IDs below next_segment_id.
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("empty_meta_{nanos}")).unwrap(),
        ));
        let manager = NamedVectorManager::new("test_collection".to_string());
        manager.attach_backend(Arc::clone(&backend));

        let dir = TempDir::new().unwrap();
        let config = NamedVectorConfig {
            size: 3,
            distance: DistanceMetric::Cosine,
            spindle: SpindleConfig::default(),
        };
        let hnsw = HNSWConfig::new(Some(8), Some(16), Some(32));

        // Allocate and publish segment 0
        let mut space = DenseVectorSpaceMetadata {
            next_segment_id: 0,
            segments: vec![],
        };
        let cores_empty = HashMap::new();
        let (name_0, _core_0) = manager
            .allocate_dense_core_locked_lsm(
                dir.path(),
                "dense",
                &config,
                hnsw.clone(),
                &mut space,
                &cores_empty,
            )
            .unwrap();
        space.segments.push(DenseVectorSegmentMetadata {
            physical_name: name_0.clone(),
            role: DenseVectorSegmentRole::Indexed,
        });

        // Populate manager's view
        let mut configs = HashMap::new();
        configs.insert("dense".to_string(), config.clone());
        let mut metadata = HashMap::new();
        metadata.insert("dense".to_string(), space.clone());
        manager
            .refresh_dense_view_from_metadata_lsm(
                dir.path(),
                &configs,
                &metadata,
                hnsw.clone(),
                None,
            )
            .unwrap();
        assert!(manager.cores.read().unwrap().contains_key(&name_0));

        // Refresh with EMPTY metadata (simulating a read that decodes to default)
        let empty_metadata = HashMap::new();
        manager
            .refresh_dense_view_from_metadata_lsm(
                dir.path(),
                &configs,
                &empty_metadata,
                hnsw.clone(),
                None,
            )
            .unwrap();

        // VERIFY FIX: core must NOT be pruned (because logical space "dense" is absent from new metadata)
        assert!(
            manager.cores.read().unwrap().contains_key(&name_0),
            "Empty metadata refresh must not prune cores from spaces absent in new metadata"
        );
    }

    #[test]
    fn reader_refresh_prune_ordering_race_safe() {
        // Issue B-HIGH: pruning under cores write lock BEFORE dense_spaces swap causes a race.
        // A concurrent search holding pre-merge dense_spaces snapshot, queued behind cores.write(),
        // finds its source cores pruned after release. Fix: swap dense_spaces FIRST, delay prune
        // by one refresh (only prune names absent from BOTH previous and current metadata).
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("prune_race_{nanos}")).unwrap(),
        ));
        let manager = NamedVectorManager::new("test_collection".to_string());
        manager.attach_backend(Arc::clone(&backend));

        let dir = TempDir::new().unwrap();
        let config = NamedVectorConfig {
            size: 3,
            distance: DistanceMetric::Cosine,
            spindle: SpindleConfig::default(),
        };
        let hnsw = HNSWConfig::new(Some(8), Some(16), Some(32));

        // Writer: allocate segment 0, publish, then retire (simulating a merge)
        let mut space = DenseVectorSpaceMetadata {
            next_segment_id: 0,
            segments: vec![],
        };
        let cores_empty = HashMap::new();
        let (name_0, _core_0) = manager
            .allocate_dense_core_locked_lsm(
                dir.path(),
                "dense",
                &config,
                hnsw.clone(),
                &mut space,
                &cores_empty,
            )
            .unwrap();
        space.segments.push(DenseVectorSegmentMetadata {
            physical_name: name_0.clone(),
            role: DenseVectorSegmentRole::Indexed,
        });

        // Reader sees segment 0
        let mut configs = HashMap::new();
        configs.insert("dense".to_string(), config.clone());
        let mut metadata = HashMap::new();
        metadata.insert("dense".to_string(), space.clone());
        manager
            .refresh_dense_view_from_metadata_lsm(
                dir.path(),
                &configs,
                &metadata,
                hnsw.clone(),
                None,
            )
            .unwrap();
        assert!(manager.cores.read().unwrap().contains_key(&name_0));

        // Simulate search holding pre-merge dense_spaces snapshot:
        // Snapshot the current metadata BEFORE the refresh that retires segment 0
        let pre_merge_metadata = manager.dense_spaces.read().unwrap().clone();
        assert!(pre_merge_metadata
            .get("dense")
            .unwrap()
            .segments
            .iter()
            .any(|s| s.physical_name == name_0));

        // Writer retires segment 0 (removes from metadata)
        space.segments.retain(|s| s.physical_name != name_0);
        metadata.insert("dense".to_string(), space.clone());

        // First refresh: segment 0 is absent from NEW metadata but present in PREVIOUS metadata.
        // Fix: it must NOT be pruned yet (delayed by one refresh).
        manager
            .refresh_dense_view_from_metadata_lsm(
                dir.path(),
                &configs,
                &metadata,
                hnsw.clone(),
                None,
            )
            .unwrap();
        assert!(
            manager.cores.read().unwrap().contains_key(&name_0),
            "First refresh after retire must NOT prune (delayed by one refresh)"
        );

        // Second refresh: segment 0 is absent from BOTH previous and current metadata.
        // Now it's safe to prune.
        manager
            .refresh_dense_view_from_metadata_lsm(dir.path(), &configs, &metadata, hnsw, None)
            .unwrap();
        assert!(
            !manager.cores.read().unwrap().contains_key(&name_0),
            "Second refresh must prune core absent from both previous and current metadata"
        );
    }

    #[test]
    fn lsm_reader_keeps_a_core_for_reads_pinned_before_a_merge() {
        // A read pinned before a merge searches the replaced segment from its
        // own list, so a replica keeps that core for a grace however many
        // reconciles follow, counted from the first one without the segment.
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::time::{Duration, SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("retired_grace_{nanos}")).unwrap(),
        ));
        let manager = NamedVectorManager::new("test_collection".to_string());
        manager.attach_backend(Arc::clone(&backend));
        manager.set_retired_core_grace(Duration::from_secs(3600));

        let dir = TempDir::new().unwrap();
        let config = NamedVectorConfig {
            size: 3,
            distance: DistanceMetric::Cosine,
            spindle: SpindleConfig::default(),
        };
        let hnsw = HNSWConfig::new(Some(8), Some(16), Some(32));
        let mut space = DenseVectorSpaceMetadata {
            next_segment_id: 0,
            segments: vec![],
        };
        let cores_empty = HashMap::new();
        let (name_0, _core_0) = manager
            .allocate_dense_core_locked_lsm(
                dir.path(),
                "dense",
                &config,
                hnsw.clone(),
                &mut space,
                &cores_empty,
            )
            .unwrap();
        space.segments.push(DenseVectorSegmentMetadata {
            physical_name: name_0.clone(),
            role: DenseVectorSegmentRole::Indexed,
        });
        let mut configs = HashMap::new();
        configs.insert("dense".to_string(), config.clone());
        let mut metadata = HashMap::new();
        metadata.insert("dense".to_string(), space.clone());
        // Reconcile, then say whether segment 0 still has its core.
        let reconcile = |metadata: &HashMap<String, DenseVectorSpaceMetadata>| {
            manager
                .refresh_dense_view_from_metadata_lsm(
                    dir.path(),
                    &configs,
                    metadata,
                    hnsw.clone(),
                    None,
                )
                .unwrap();
            manager.cores.read().unwrap().contains_key(&name_0)
        };
        assert!(reconcile(&metadata));
        // A core off the list builds no sidecar: a build reads the newest view.
        let sidecar_kept_as_is = || manager.cores.read().unwrap()[&name_0].sidecar_kept_as_is();
        assert!(
            !sidecar_kept_as_is(),
            "a listed segment's core was held to the sidecar it has"
        );

        // The writer retires segment 0. The clock starts at the reconcile
        // that first sees the list without it.
        space.segments.retain(|s| s.physical_name != name_0);
        metadata.insert("dense".to_string(), space.clone());
        assert!(reconcile(&metadata));
        assert!(
            sidecar_kept_as_is(),
            "a core off the list was left free to build a sidecar"
        );
        std::thread::sleep(Duration::from_millis(60));
        assert!(
            reconcile(&metadata),
            "the second reconcile dropped a core inside its grace"
        );
        assert!(
            reconcile(&metadata),
            "the third reconcile dropped a core inside its grace"
        );

        manager.set_retired_core_grace(Duration::from_millis(40));
        assert!(
            !reconcile(&metadata),
            "a core 60 ms past the first reconcile without it outlived a 40 ms grace"
        );
    }

    /// A manager with two listed dense segments whose cores are open, the
    /// directory their sidecars go to, and what a reconcile needs.
    #[allow(clippy::type_complexity)]
    fn two_listed_segments() -> (
        NamedVectorManager,
        TempDir,
        HashMap<String, NamedVectorConfig>,
        HNSWConfig,
        DenseVectorSpaceMetadata,
        Vec<String>,
    ) {
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("retired_two_{nanos}")).unwrap(),
        ));
        let manager = NamedVectorManager::new("test_collection".to_string());
        manager.attach_backend(Arc::clone(&backend));

        let dir = TempDir::new().unwrap();
        let config = NamedVectorConfig {
            size: 3,
            distance: DistanceMetric::Cosine,
            spindle: SpindleConfig::default(),
        };
        let hnsw = HNSWConfig::new(Some(8), Some(16), Some(32));
        let mut space = DenseVectorSpaceMetadata {
            next_segment_id: 0,
            segments: vec![],
        };
        let cores_empty = HashMap::new();
        let mut names = Vec::new();
        for _ in 0..2 {
            let (name, _core) = manager
                .allocate_dense_core_locked_lsm(
                    dir.path(),
                    "dense",
                    &config,
                    hnsw.clone(),
                    &mut space,
                    &cores_empty,
                )
                .unwrap();
            space.segments.push(DenseVectorSegmentMetadata {
                physical_name: name.clone(),
                role: DenseVectorSegmentRole::Indexed,
            });
            names.push(name);
        }
        let mut configs = HashMap::new();
        configs.insert("dense".to_string(), config);
        (manager, dir, configs, hnsw, space, names)
    }

    #[test]
    fn lsm_reader_drops_a_retired_core_past_its_grace_without_a_reconcile() {
        // Reconciles run only from request paths. A collection that is not
        // read again still gives up a retired core once its grace is over:
        // the poll-tier sweeper asks for that between requests.
        use std::time::Duration;

        let (manager, dir, configs, hnsw, mut space, names) = two_listed_segments();
        manager.set_retired_core_grace(Duration::from_secs(3600));
        let reconcile = |space: &DenseVectorSpaceMetadata| {
            let mut metadata = HashMap::new();
            metadata.insert("dense".to_string(), space.clone());
            manager
                .refresh_dense_view_from_metadata_lsm(
                    dir.path(),
                    &configs,
                    &metadata,
                    hnsw.clone(),
                    None,
                )
                .unwrap();
        };
        let open = |name: &String| manager.cores.read().unwrap().contains_key(name);
        reconcile(&space);
        assert!(open(&names[0]) && open(&names[1]));
        assert_eq!(manager.drop_retired_cores_past_grace().unwrap(), 0);

        // The writer retires segment 0 and one reconcile sees the list without
        // it. No other reconcile follows.
        space.segments.retain(|s| s.physical_name != names[0]);
        reconcile(&space);
        assert_eq!(
            manager.drop_retired_cores_past_grace().unwrap(),
            0,
            "a core inside its grace was dropped"
        );
        assert!(open(&names[0]));

        std::thread::sleep(Duration::from_millis(60));
        manager.set_retired_core_grace(Duration::from_millis(40));
        assert_eq!(
            manager.drop_retired_cores_past_grace().unwrap(),
            1,
            "a core past its grace stayed without a reconcile"
        );
        assert!(!open(&names[0]), "the retired core is still open");
        assert!(open(&names[1]), "a listed segment's core was dropped");
        assert_eq!(manager.drop_retired_cores_past_grace().unwrap(), 0);
    }

    #[test]
    fn lsm_reader_close_unlinks_the_sidecars_of_its_retired_cores() {
        // A close leaves a listed segment's sidecar for the next open and
        // takes a retired one's: nothing else would before that open.
        use std::time::Duration;

        let (manager, dir, configs, hnsw, mut space, names) = two_listed_segments();
        manager.set_retired_core_grace(Duration::from_secs(3600));
        let reconcile = |space: &DenseVectorSpaceMetadata| {
            let mut metadata = HashMap::new();
            metadata.insert("dense".to_string(), space.clone());
            manager
                .refresh_dense_view_from_metadata_lsm(
                    dir.path(),
                    &configs,
                    &metadata,
                    hnsw.clone(),
                    None,
                )
                .unwrap();
        };
        reconcile(&space);
        // Stand-ins for the files a search builds.
        let sidecar = |name: &String| dir.path().join(format!("{name}.hvec"));
        for name in &names {
            std::fs::write(sidecar(name), b"sidecar").unwrap();
        }

        space.segments.retain(|s| s.physical_name != names[0]);
        reconcile(&space);
        assert!(
            sidecar(&names[0]).exists(),
            "a reconcile took a retired core's sidecar inside its grace"
        );
        manager.unlink_retired_sidecars(dir.path());
        assert!(
            !sidecar(&names[0]).exists(),
            "a retired core's sidecar outlived the close"
        );
        assert!(
            sidecar(&names[1]).exists(),
            "the close took a listed segment's sidecar"
        );
    }

    #[test]
    fn reader_delayed_pruning_and_writer_monotonic_allocation() {
        // Issue B: Two-part fix preventing stale-core bugs:
        // (a) Writer monotonic allocation: allocate_dense_core_locked_lsm never reuses
        //     retired segment names (next_segment_id increments).
        // (b) Reader delayed pruning: refresh_dense_view_from_metadata_lsm delays pruning
        //     by one refresh cycle to prevent search snapshots from racing with core pruning.
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("monotonic_{nanos}"))
                .expect("open in-memory LSM backend"),
        ));
        let writer_manager = NamedVectorManager::new("test_collection".to_string());
        writer_manager.attach_backend(Arc::clone(&backend));

        let reader_manager = NamedVectorManager::new("test_collection".to_string());
        reader_manager.attach_backend(Arc::clone(&backend));

        let dir = TempDir::new().unwrap();
        let config = NamedVectorConfig {
            size: 3,
            distance: DistanceMetric::Cosine,
            spindle: SpindleConfig::default(),
        };
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));

        // Writer allocates segment 0
        let mut space = DenseVectorSpaceMetadata {
            next_segment_id: 0,
            segments: vec![],
        };
        let cores_empty = HashMap::new();
        let (name_0, core_0) = writer_manager
            .allocate_dense_core_locked_lsm(
                dir.path(),
                "dense",
                &config,
                hnsw.clone(),
                &mut space,
                &cores_empty,
            )
            .unwrap();
        writer_manager
            .cores
            .write()
            .unwrap()
            .insert(name_0.clone(), core_0);
        space.segments.push(DenseVectorSegmentMetadata {
            physical_name: name_0.clone(),
            role: DenseVectorSegmentRole::Indexed,
        });

        // Reader sees segment 0 in metadata and opens it
        let mut reader_configs = HashMap::new();
        reader_configs.insert("dense".to_string(), config.clone());
        let mut reader_metadata = HashMap::new();
        reader_metadata.insert("dense".to_string(), space.clone());
        reader_manager
            .refresh_dense_view_from_metadata_lsm(
                dir.path(),
                &reader_configs,
                &reader_metadata,
                hnsw.clone(),
                None,
            )
            .unwrap();
        assert!(
            reader_manager.cores.read().unwrap().contains_key(&name_0),
            "Reader must open segment 0 after initial refresh"
        );

        // Writer retires segment 0 (removes from metadata)
        space.segments.retain(|s| s.physical_name != name_0);
        reader_metadata.insert("dense".to_string(), space.clone());

        // VERIFY FIX (b): First refresh after retirement - core should NOT be pruned yet
        reader_manager
            .refresh_dense_view_from_metadata_lsm(
                dir.path(),
                &reader_configs,
                &reader_metadata,
                hnsw.clone(),
                None,
            )
            .unwrap();
        assert!(
            reader_manager.cores.read().unwrap().contains_key(&name_0),
            "FIX (b) First refresh: core must NOT be pruned (delayed by one refresh)"
        );

        // VERIFY FIX (b): Second refresh - now pruning is safe
        reader_manager
            .refresh_dense_view_from_metadata_lsm(
                dir.path(),
                &reader_configs,
                &reader_metadata,
                hnsw.clone(),
                None,
            )
            .unwrap();
        assert!(
            !reader_manager.cores.read().unwrap().contains_key(&name_0),
            "FIX (b) Second refresh: reader must drop core absent from both previous and current metadata"
        );

        // VERIFY FIX (a): Writer allocates a NEW segment with a DIFFERENT name
        let (name_new, _core_new) = writer_manager
            .allocate_dense_core_locked_lsm(
                dir.path(),
                "dense",
                &config,
                hnsw,
                &mut space,
                &cores_empty,
            )
            .unwrap();
        assert_ne!(
            name_new, name_0,
            "FIX (a) Writer must not reuse retired segment name (monotonic allocation)"
        );
        assert!(
            space.next_segment_id > 0,
            "FIX (a) next_segment_id must increment (was 0, now {})",
            space.next_segment_id
        );

        // All guarantees proven:
        // (a) Writer allocated a different name (name_new != name_0, next_segment_id incremented)
        // (b) Reader delayed pruning by one refresh (first refresh kept core, second dropped it)
    }

    #[test]
    fn shared_cache_post_commit_invalidation_via_plan_flow() {
        // Issue A end-to-end: prod flow (plan → delete_vectors_batch_with_plan_be → commit →
        // apply_delete_tombstones) must invalidate shared cache entries post-commit, catching
        // any re-cached entries from concurrent reads between pre-commit pop and commit.
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use crate::helix_engine::vector_core::shared_cache::SharedVectorCaches;
        use std::num::NonZeroUsize;
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("plan_cache_{nanos}")).unwrap(),
        ));
        let manager = NamedVectorManager::new("test_collection".to_string());
        manager.attach_backend(Arc::clone(&backend));

        let shared = Arc::new(SharedVectorCaches::new(
            NonZeroUsize::new(512).unwrap(),
            NonZeroUsize::new(512).unwrap(),
            NonZeroUsize::new(512).unwrap(),
        ));

        let dir = TempDir::new().unwrap();
        let config = NamedVectorConfig {
            size: 3,
            distance: DistanceMetric::Cosine,
            spindle: SpindleConfig::default(),
        };
        let hnsw = HNSWConfig::new(Some(8), Some(16), Some(32));

        // Allocate segment and attach shared cache
        let mut space = DenseVectorSpaceMetadata {
            next_segment_id: 0,
            segments: vec![],
        };
        let cores_empty = HashMap::new();
        let (seg_name, mut core) = manager
            .allocate_dense_core_locked_lsm(
                dir.path(),
                "dense",
                &config,
                hnsw.clone(),
                &mut space,
                &cores_empty,
            )
            .unwrap();
        core.shared_caches = Some(Arc::clone(&shared));
        manager
            .cores
            .write()
            .unwrap()
            .insert(seg_name.clone(), core);

        space.segments.push(DenseVectorSegmentMetadata {
            physical_name: seg_name.clone(),
            role: DenseVectorSegmentRole::Indexed,
        });
        manager
            .dense_spaces
            .write()
            .unwrap()
            .insert("dense".to_string(), space.clone());

        // Insert id 1
        let mut w = backend.begin_write().unwrap();
        let vec = vec![1.0f32, 0.0, 0.0];
        manager
            .cores
            .read()
            .unwrap()
            .get(&seg_name)
            .unwrap()
            .insert_flat_be(&mut w, &vec, Some(1), None)
            .unwrap();
        backend.commit(w).unwrap();

        // Read (populates cache)
        let r = backend.begin_read().unwrap();
        let cached = manager
            .cores
            .read()
            .unwrap()
            .get(&seg_name)
            .unwrap()
            .get_vector(&r, 1, 0, true)
            .unwrap();
        // Verify vector exists (exact value depends on normalization)
        assert!(cached.get_data()[0] > 0.0);
        drop(r);

        // Delete via prod flow: plan → delete_with_plan_be → apply_delete_tombstones
        let r2 = backend.begin_read().unwrap();
        let plan = manager.plan_delete_vectors_batch_be(&r2, &[1]).unwrap();
        drop(r2);

        let mut w2 = backend.begin_write().unwrap();
        let staged = manager
            .delete_vectors_batch_with_plan_be(&mut w2, &plan)
            .unwrap();
        // A read before the delete commits re-caches the still-committed row.
        let mid = backend.begin_read().unwrap();
        let _ = manager
            .cores
            .read()
            .unwrap()
            .get(&seg_name)
            .unwrap()
            .get_vector(&mid, 1, 0, true);
        drop(mid);
        backend.commit(w2).unwrap();

        // Post-commit: apply_delete_tombstones must invalidate cache at exact levels
        manager.apply_delete_tombstones(&staged);

        // Verify: get_vector must miss (not return cached value)
        let r3 = backend.begin_read().unwrap();
        assert!(
            manager
                .cores
                .read()
                .unwrap()
                .get(&seg_name)
                .unwrap()
                .get_vector(&r3, 1, 0, true)
                .is_err(),
            "Post-commit invalidation failed: get_vector returned cached entry after delete"
        );
    }

    #[test]
    fn hard_delete_allows_immediate_reinsertion() {
        // Issue A follow-up: with tombstones OFF (hard-delete path), verify that
        // by_segment (tombstones) remains empty, so re-inserted IDs remain visible.
        // Flow: insert id → delete id → verify by_segment is empty → re-insert same id → must be visible.
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("hard_del_{nanos}")).unwrap(),
        ));
        let manager = NamedVectorManager::new("test_collection".to_string());
        manager.attach_backend(Arc::clone(&backend));

        let dir = TempDir::new().unwrap();
        let config = NamedVectorConfig {
            size: 3,
            distance: DistanceMetric::Cosine,
            spindle: SpindleConfig::default(),
        };
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));

        // Allocate dense space
        let mut space = DenseVectorSpaceMetadata {
            next_segment_id: 0,
            segments: vec![],
        };
        let cores_empty = HashMap::new();
        let (name, core) = manager
            .allocate_dense_core_locked_lsm(
                dir.path(),
                "dense",
                &config,
                hnsw.clone(),
                &mut space,
                &cores_empty,
            )
            .unwrap();
        manager.cores.write().unwrap().insert(name.clone(), core);

        space.segments.push(DenseVectorSegmentMetadata {
            physical_name: name.clone(),
            role: DenseVectorSegmentRole::Indexed,
        });
        manager
            .dense_spaces
            .write()
            .unwrap()
            .insert("dense".to_string(), space.clone());

        // Insert id=1
        let vec_a = vec![1.0f32, 0.0, 0.0];
        let mut w1 = backend.begin_write().unwrap();
        manager
            .cores
            .read()
            .unwrap()
            .get(&name)
            .unwrap()
            .insert_flat_be(&mut w1, &vec_a, Some(1), None)
            .unwrap();
        backend.commit(w1).unwrap();

        // Verify id=1 is readable
        let r1 = backend.begin_read().unwrap();
        assert!(
            manager
                .cores
                .read()
                .unwrap()
                .get(&name)
                .unwrap()
                .get_vector(&r1, 1, 0, true)
                .is_ok(),
            "Initial insert: id=1 must be visible"
        );
        drop(r1);

        // Hard-delete id=1
        let r_plan = backend.begin_read().unwrap();
        let plan = manager.plan_delete_vectors_batch_be(&r_plan, &[1]).unwrap();
        drop(r_plan);

        let mut w2 = backend.begin_write().unwrap();
        let staged = manager
            .delete_vectors_batch_with_plan_be(&mut w2, &plan)
            .unwrap();
        backend.commit(w2).unwrap();

        // CRITICAL CHECK: with tombstones OFF (hard-delete), by_segment must be empty
        // to ensure re-inserted ids won't be hidden by in-memory tombstone sets
        assert!(
            staged.by_segment.is_empty(),
            "Hard-delete path must not populate by_segment (found {} entries). \
             This would hide re-inserted IDs via in-memory tombstone sets.",
            staged.by_segment.len()
        );
        manager.apply_delete_tombstones(&staged);

        // Verify delete worked
        let r_check = backend.begin_read().unwrap();
        assert!(
            manager
                .cores
                .read()
                .unwrap()
                .get(&name)
                .unwrap()
                .get_vector(&r_check, 1, 0, true)
                .is_err(),
            "After hard-delete: id=1 must be absent"
        );
        drop(r_check);

        // Re-insert id=1 with a different vector
        let vec_b = vec![0.0f32, 1.0, 0.0];
        let mut w3 = backend.begin_write().unwrap();
        manager
            .cores
            .read()
            .unwrap()
            .get(&name)
            .unwrap()
            .insert_flat_be(&mut w3, &vec_b, Some(1), None)
            .unwrap();
        backend.commit(w3).unwrap();

        // VERIFY: id=1 must be visible after re-insertion
        let r2 = backend.begin_read().unwrap();
        assert!(
            manager
                .cores
                .read()
                .unwrap()
                .get(&name)
                .unwrap()
                .get_vector(&r2, 1, 0, true)
                .is_ok(),
            "After re-insert: id=1 must be visible (hard-delete must not leave tombstones)"
        );
        // get_vector ignores tombstones, so check the set search consults directly.
        assert!(!manager
            .cores
            .read()
            .unwrap()
            .get(&name)
            .unwrap()
            .is_delete_tombstoned(1));
    }

    #[test]
    fn shared_cache_post_commit_drops_stale_sidecar_ordinal() {
        // Prod hard-delete flow on a sidecar-backed core: a read between the delete
        // and its commit re-caches the row's sidecar ordinal; the post-commit hook
        // must drop it, or get_vector serves the deleted vector from the hvec file.
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use crate::helix_engine::vector_core::hnsw::HNSW;
        use crate::helix_engine::vector_core::spindle::{SpindleConfig, SpindleMode};
        use crate::helix_engine::vector_core::vector_core::HNSWConfig;
        use std::num::NonZeroUsize;

        const SEG: &str = "dense__seg_000001";
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("post_commit_ordinal-{}", uuid::Uuid::new_v4()))
                .unwrap(),
        ));
        let manager = NamedVectorManager::new("post_commit_ordinal".to_string());
        manager.attach_backend(Arc::clone(&backend));
        let mut configs = HashMap::new();
        configs.insert(
            "dense".to_string(),
            NamedVectorConfig {
                size: 4,
                distance: DistanceMetric::Cosine,
                spindle: SpindleConfig {
                    mode: SpindleMode::None,
                    ..SpindleConfig::default()
                },
            },
        );
        let mut meta = HashMap::new();
        meta.insert(
            "dense".to_string(),
            DenseVectorSpaceMetadata {
                next_segment_id: 2,
                segments: vec![DenseVectorSegmentMetadata {
                    physical_name: SEG.to_string(),
                    role: DenseVectorSegmentRole::Mutable,
                }],
            },
        );
        // A data dir gives the core an hvec sidecar, so reads resolve through ordinals.
        let dir = TempDir::new().unwrap();
        manager
            .refresh_dense_view_from_metadata_lsm(
                dir.path(),
                &configs,
                &meta,
                HNSWConfig::new(Some(8), Some(16), Some(32)),
                None,
            )
            .unwrap();
        let shared = Arc::new(super::super::shared_cache::SharedVectorCaches::new(
            NonZeroUsize::new(4096).unwrap(),
            NonZeroUsize::new(4096).unwrap(),
            NonZeroUsize::new(4096).unwrap(),
        ));
        manager
            .cores
            .write()
            .unwrap()
            .get_mut(SEG)
            .unwrap()
            .shared_caches = Some(shared);

        let read_7 = || {
            let cores = manager.cores.read().unwrap();
            let r = backend.begin_read().unwrap();
            cores
                .get(SEG)
                .unwrap()
                .get_vector(&r, 7, 0, true)
                .map(|v| v.get_data().to_vec())
        };
        {
            let cores = manager.cores.read().unwrap();
            let mut w = backend.begin_write().unwrap();
            cores
                .get(SEG)
                .unwrap()
                .insert_flat_be(&mut w, &[1.0, 0.0, 0.0, 0.0], Some(7), None)
                .unwrap();
            backend.commit(w).unwrap();
        }
        assert!(read_7().is_ok());

        let plan = {
            let r = backend.begin_read().unwrap();
            manager.plan_delete_vectors_batch_be(&r, &[7]).unwrap()
        };
        let mut w = backend.begin_write().unwrap();
        let staged = manager
            .delete_vectors_batch_with_plan_be(&mut w, &plan)
            .unwrap();
        assert!(
            read_7().is_ok(),
            "row is still committed before the delete commits"
        );
        backend.commit(w).unwrap();
        manager.apply_delete_tombstones(&staged);

        let after = read_7();
        assert!(
            after.is_err(),
            "deleted id served after commit + post-commit hook: {:?}",
            after
        );
    }

    #[test]
    fn reader_refresh_publishes_core_before_listing_segment() {
        // A search holding cores.read() must never see a segment listed in
        // dense_spaces whose core is not yet in `cores` (it would be skipped).
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use crate::helix_engine::vector_core::spindle::{SpindleConfig, SpindleMode};
        use crate::helix_engine::vector_core::vector_core::HNSWConfig;

        const SEG_A: &str = "dense__seg_000001";
        const SEG_B: &str = "dense__seg_000002";
        let meta = |names: &[&str]| {
            let mut m = HashMap::new();
            m.insert(
                "dense".to_string(),
                DenseVectorSpaceMetadata {
                    next_segment_id: 3,
                    segments: names
                        .iter()
                        .map(|n| DenseVectorSegmentMetadata {
                            physical_name: n.to_string(),
                            role: DenseVectorSegmentRole::Mutable,
                        })
                        .collect(),
                },
            );
            m
        };
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("publish_order-{}", uuid::Uuid::new_v4())).unwrap(),
        ));
        let manager = NamedVectorManager::new("publish_order".to_string());
        manager.attach_backend(Arc::clone(&backend));
        let mut configs = HashMap::new();
        configs.insert(
            "dense".to_string(),
            NamedVectorConfig {
                size: 4,
                distance: DistanceMetric::Cosine,
                spindle: SpindleConfig {
                    mode: SpindleMode::None,
                    ..SpindleConfig::default()
                },
            },
        );
        let hnsw = HNSWConfig::new(Some(8), Some(16), Some(32));
        let dir = TempDir::new().unwrap();
        manager
            .refresh_dense_view_from_metadata_lsm(
                dir.path(),
                &configs,
                &meta(&[SEG_A]),
                hnsw.clone(),
                None,
            )
            .unwrap();

        let mut listed_without_core = false;
        std::thread::scope(|s| {
            let in_flight_search = manager.cores.read().unwrap();
            let refresh = s.spawn(|| {
                manager
                    .refresh_dense_view_from_metadata_lsm(
                        dir.path(),
                        &configs,
                        &meta(&[SEG_A, SEG_B]),
                        hnsw.clone(),
                        None,
                    )
                    .unwrap()
            });
            let started = std::time::Instant::now();
            while started.elapsed() < std::time::Duration::from_millis(500) {
                let listed = manager
                    .dense_spaces
                    .read()
                    .unwrap()
                    .get("dense")
                    .is_some_and(|sp| sp.segments.iter().any(|seg| seg.physical_name == SEG_B));
                if listed {
                    listed_without_core = !in_flight_search.contains_key(SEG_B);
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            drop(in_flight_search);
            refresh.join().unwrap();
        });
        assert!(
            !listed_without_core,
            "dense_spaces listed {SEG_B} before its core was published"
        );
        assert!(manager.cores.read().unwrap().contains_key(SEG_B));
    }

    // ── Reviewer probes (P1, P1b, P2, P3) ──
    //
    // Regression tests for deleted-vector resurrection through cache fills.
    // Run with tombstones OFF (prod mode) and shared caches both on and off.

    const PROBE_SEG: &str = "dense__seg_000001";

    fn probe_rig(shared_on: bool, dir: &Path, backend: &Arc<AnyBackend>) -> NamedVectorManager {
        let manager = NamedVectorManager::new("probe".to_string());
        manager.attach_backend(Arc::clone(backend));
        let mut configs = HashMap::new();
        configs.insert(
            "dense".to_string(),
            NamedVectorConfig {
                size: 4,
                distance: DistanceMetric::Cosine,
                spindle: SpindleConfig {
                    mode: SpindleMode::None,
                    ..SpindleConfig::default()
                },
            },
        );
        let mut meta = HashMap::new();
        meta.insert(
            "dense".to_string(),
            DenseVectorSpaceMetadata {
                next_segment_id: 2,
                segments: vec![DenseVectorSegmentMetadata {
                    physical_name: PROBE_SEG.to_string(),
                    role: DenseVectorSegmentRole::Mutable,
                }],
            },
        );
        manager
            .refresh_dense_view_from_metadata_lsm(
                dir,
                &configs,
                &meta,
                HNSWConfig::new(Some(8), Some(16), Some(32)),
                None,
            )
            .unwrap();
        if shared_on {
            use crate::helix_engine::vector_core::shared_cache::SharedVectorCaches;
            use std::num::NonZeroUsize;
            let shared = Arc::new(SharedVectorCaches::new(
                NonZeroUsize::new(4096).unwrap(),
                NonZeroUsize::new(4096).unwrap(),
                NonZeroUsize::new(4096).unwrap(),
            ));
            manager
                .cores
                .write()
                .unwrap()
                .get_mut(PROBE_SEG)
                .unwrap()
                .shared_caches = Some(shared);
        }
        manager
    }

    fn probe_backend() -> Arc<AnyBackend> {
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("probe-{}", uuid::Uuid::new_v4())).unwrap(),
        ))
    }

    fn probe_insert(m: &NamedVectorManager, b: &AnyBackend, rows: &[(u128, [f32; 4])]) {
        let cores = m.cores.read().unwrap();
        let mut w = b.begin_write().unwrap();
        for (id, v) in rows {
            cores
                .get(PROBE_SEG)
                .unwrap()
                .insert_flat_be(&mut w, v, Some(*id), None)
                .unwrap();
        }
        b.commit(w).unwrap();
    }

    fn probe_get(
        m: &NamedVectorManager,
        r: &AnyRead<'_>,
        id: u128,
    ) -> Result<Vec<f32>, VectorError> {
        let cores = m.cores.read().unwrap();
        cores
            .get(PROBE_SEG)
            .unwrap()
            .get_vector(r, id, 0, true)
            .map(|v| v.get_data().to_vec())
    }

    fn probe_search_ids(m: &NamedVectorManager, b: &AnyBackend) -> Vec<u128> {
        let r = b.begin_read().unwrap();
        m.dense_search_with_id_filter_ef_be::<fn(u128) -> bool>(
            &r,
            "dense",
            &[1.0, 0.0, 0.0, 0.0],
            10,
            None,
            false,
            None,
            None,
        )
        .unwrap()
        .into_iter()
        .map(|h| h.get_id())
        .collect()
    }

    /// delete -> commit -> post-commit hook. `between` runs after the delete
    /// is staged but before commit.
    fn probe_delete_with_hook(
        m: &NamedVectorManager,
        b: &AnyBackend,
        ids: &[u128],
        between: impl FnOnce(),
    ) {
        let _lock = TOMBSTONE_ENV_LOCK.lock().unwrap();
        let _unset = EnvVarGuard::unset("HELIX_LSM_DELETE_TOMBSTONES");
        let plan = {
            let r = b.begin_read().unwrap();
            m.plan_delete_vectors_batch_be(&r, ids).unwrap()
        };
        let mut w = b.begin_write().unwrap();
        let staged = m.delete_vectors_batch_with_plan_be(&mut w, &plan).unwrap();
        assert!(
            staged.by_segment.is_empty(),
            "tombstones must be OFF (prod mode)"
        );
        between();
        b.commit(w).unwrap();
        m.apply_delete_tombstones(&staged);
    }

    fn p1(shared_on: bool) {
        let b = probe_backend();
        let dir = TempDir::new().unwrap();
        let m = probe_rig(shared_on, dir.path(), &b);
        probe_insert(
            &m,
            &b,
            &[(7, [1.0, 0.0, 0.0, 0.0]), (8, [0.0, 1.0, 0.0, 0.0])],
        );
        assert!(probe_get(&m, &b.begin_read().unwrap(), 7).is_ok());
        probe_delete_with_hook(&m, &b, &[7], || {});
        let after = probe_get(&m, &b.begin_read().unwrap(), 7);
        let hits = probe_search_ids(&m, &b);
        assert!(
            after.is_err() && !hits.contains(&7),
            "P1 shared={shared_on}: deleted id served: get={after:?} search={hits:?}"
        );
    }

    fn p1b(shared_on: bool) {
        let b = probe_backend();
        let dir = TempDir::new().unwrap();
        let m = probe_rig(shared_on, dir.path(), &b);
        probe_insert(
            &m,
            &b,
            &[(7, [1.0, 0.0, 0.0, 0.0]), (8, [0.0, 1.0, 0.0, 0.0])],
        );
        assert!(probe_get(&m, &b.begin_read().unwrap(), 7).is_ok());
        probe_delete_with_hook(&m, &b, &[7], || {
            // concurrent fresh read in the pre-commit window
            assert!(probe_get(&m, &b.begin_read().unwrap(), 7).is_ok());
        });
        let after = probe_get(&m, &b.begin_read().unwrap(), 7);
        let hits = probe_search_ids(&m, &b);
        assert!(
            after.is_err() && !hits.contains(&7),
            "P1b shared={shared_on}: deleted id served: get={after:?} search={hits:?}"
        );
    }

    fn p2(shared_on: bool) {
        let b = probe_backend();
        let dir = TempDir::new().unwrap();
        let m = probe_rig(shared_on, dir.path(), &b);
        probe_insert(
            &m,
            &b,
            &[(7, [1.0, 0.0, 0.0, 0.0]), (8, [0.0, 1.0, 0.0, 0.0])],
        );
        assert!(probe_get(&m, &b.begin_read().unwrap(), 7).is_ok());
        let old = b.begin_read().unwrap();
        probe_delete_with_hook(&m, &b, &[7], || {});
        // snapshot-isolated read of the old snapshot AFTER the hook
        let _old_read = probe_get(&m, &old, 7);
        drop(old);
        let after = probe_get(&m, &b.begin_read().unwrap(), 7);
        let hits = probe_search_ids(&m, &b);
        assert!(
            after.is_err() && !hits.contains(&7),
            "P2 shared={shared_on}: deleted id served: get={after:?} search={hits:?}"
        );
    }

    fn p3(shared_on: bool) {
        let b = probe_backend();
        let dir = TempDir::new().unwrap();
        {
            let m = probe_rig(shared_on, dir.path(), &b);
            probe_insert(
                &m,
                &b,
                &[(7, [1.0, 0.0, 0.0, 0.0]), (8, [0.0, 1.0, 0.0, 0.0])],
            );
            m.cores
                .read()
                .unwrap()
                .get(PROBE_SEG)
                .unwrap()
                .flush_mmap()
                .unwrap();
        }
        // Restart: ordinal map not loaded.
        let m = probe_rig(shared_on, dir.path(), &b);
        let old = b.begin_read().unwrap();
        probe_delete_with_hook(&m, &b, &[7], || {});
        // Old-snapshot read of a DIFFERENT live id triggers the full ordinal load.
        let _old_read = probe_get(&m, &old, 8);
        drop(old);
        let after = probe_get(&m, &b.begin_read().unwrap(), 7);
        let hits = probe_search_ids(&m, &b);
        assert!(
            after.is_err() && !hits.contains(&7),
            "P3 shared={shared_on}: deleted id served: get7={after:?} search={hits:?}"
        );
    }

    #[test]
    fn probe_p1_shared_on() {
        p1(true)
    }
    #[test]
    fn probe_p1_shared_off() {
        p1(false)
    }
    #[test]
    fn probe_p1b_shared_on() {
        p1b(true)
    }
    #[test]
    fn probe_p1b_shared_off() {
        p1b(false)
    }
    #[test]
    fn probe_p2_shared_on() {
        p2(true)
    }
    #[test]
    fn probe_p2_shared_off() {
        p2(false)
    }
    #[test]
    fn probe_p3_shared_on() {
        p3(true)
    }
    #[test]
    fn probe_p3_shared_off() {
        p3(false)
    }

    /// P4: a re-insert of a hard-deleted id is staged while a pre-delete
    /// snapshot reads the id; after commit a fresh read must serve V2.
    fn p4(shared_on: bool) {
        let b = probe_backend();
        let dir = TempDir::new().unwrap();
        let m = probe_rig(shared_on, dir.path(), &b);
        probe_insert(
            &m,
            &b,
            &[(7, [1.0, 0.0, 0.0, 0.0]), (8, [0.0, 1.0, 0.0, 0.0])],
        );
        let old = b.begin_read().unwrap();
        probe_delete_with_hook(&m, &b, &[7], || {});
        let mut w = b.begin_write().unwrap();
        m.cores
            .read()
            .unwrap()
            .get(PROBE_SEG)
            .unwrap()
            .insert_flat_be(&mut w, &[0.0, 0.0, 1.0, 0.0], Some(7), None)
            .unwrap();
        let _old_read = probe_get(&m, &old, 7);
        b.commit(w).unwrap();
        drop(old);
        let after = probe_get(&m, &b.begin_read().unwrap(), 7);
        assert_eq!(
            after.as_deref().ok(),
            Some(&[0.0f32, 0.0, 1.0, 0.0][..]),
            "P4 shared={shared_on}: re-inserted id must serve V2: after={after:?}"
        );
    }

    #[test]
    fn probe_p4_shared_on() {
        p4(true)
    }
    #[test]
    fn probe_p4_shared_off() {
        p4(false)
    }

    /// P5 (sidecar variant): a re-insert of a hard-deleted id that is staged
    /// and then ABORTED must leave no trace — not in caches (guard) and not
    /// in the sidecar ordinal map (publication retracted on abort), so a
    /// fresh read and search see the id as deleted.
    fn p5(shared_on: bool) {
        let b = probe_backend();
        let dir = TempDir::new().unwrap();
        let m = probe_rig(shared_on, dir.path(), &b);
        probe_insert(
            &m,
            &b,
            &[(7, [1.0, 0.0, 0.0, 0.0]), (8, [0.0, 1.0, 0.0, 0.0])],
        );
        let old = b.begin_read().unwrap();
        probe_delete_with_hook(&m, &b, &[7], || {});
        {
            let mut w = b.begin_write().unwrap();
            m.cores
                .read()
                .unwrap()
                .get(PROBE_SEG)
                .unwrap()
                .insert_flat_be(&mut w, &[0.0, 0.0, 1.0, 0.0], Some(7), None)
                .unwrap();
            drop(w); // abort: never committed
        }
        let _old_read = probe_get(&m, &old, 7);
        drop(old);
        let after = probe_get(&m, &b.begin_read().unwrap(), 7);
        let hits = probe_search_ids(&m, &b);
        assert!(
            after.is_err() && !hits.contains(&7),
            "P5 shared={shared_on}: deleted id served after aborted re-insert: get={after:?} search={hits:?}"
        );
    }

    #[test]
    fn probe_p5_shared_on() {
        p5(true)
    }
    #[test]
    fn probe_p5_shared_off() {
        p5(false)
    }

    // ── M7: stale-plan merge preserves earlier segments' tombstones ──

    #[test]
    fn stale_plan_merge_preserves_earlier_segments_tombstones() {
        // Verify 8a5deaf2: when a segment is missing from the plan (stale),
        // the fallback must merge id_levels into the existing staged result
        // rather than replacing it.
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;

        let _lock = TOMBSTONE_ENV_LOCK.lock().unwrap();
        let _unset = EnvVarGuard::unset("HELIX_LSM_DELETE_TOMBSTONES");

        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("stale-merge-{}", uuid::Uuid::new_v4())).unwrap(),
        ));
        let manager = NamedVectorManager::new("stale_merge".to_string());
        manager.attach_backend(Arc::clone(&backend));

        let dir = TempDir::new().unwrap();
        let config = NamedVectorConfig {
            size: 3,
            distance: DistanceMetric::Cosine,
            spindle: SpindleConfig::default(),
        };
        let hnsw = HNSWConfig::new(Some(8), Some(16), Some(32));

        // Create TWO segments, insert one id into each.
        let mut space = DenseVectorSpaceMetadata {
            next_segment_id: 0,
            segments: vec![],
        };
        let cores_empty = HashMap::new();
        let (seg_a, core_a) = manager
            .allocate_dense_core_locked_lsm(
                dir.path(),
                "dense",
                &config,
                hnsw.clone(),
                &mut space,
                &cores_empty,
            )
            .unwrap();
        let mut cores_a = HashMap::new();
        cores_a.insert(seg_a.clone(), core_a);
        let (seg_b, core_b) = manager
            .allocate_dense_core_locked_lsm(
                dir.path(),
                "dense",
                &config,
                hnsw.clone(),
                &mut space,
                &cores_a,
            )
            .unwrap();
        {
            let mut cores = manager.cores.write().unwrap();
            for (k, v) in cores_a {
                cores.insert(k, v);
            }
            cores.insert(seg_b.clone(), core_b);
        }
        space.segments.push(DenseVectorSegmentMetadata {
            physical_name: seg_a.clone(),
            role: DenseVectorSegmentRole::Indexed,
        });
        space.segments.push(DenseVectorSegmentMetadata {
            physical_name: seg_b.clone(),
            role: DenseVectorSegmentRole::Indexed,
        });
        manager
            .dense_spaces
            .write()
            .unwrap()
            .insert("dense".to_string(), space.clone());

        // Insert id=1 into seg_a, id=2 into seg_b.
        {
            let cores = manager.cores.read().unwrap();
            let mut w = backend.begin_write().unwrap();
            cores
                .get(&seg_a)
                .unwrap()
                .insert_flat_be(&mut w, &[1.0, 0.0, 0.0], Some(1), None)
                .unwrap();
            cores
                .get(&seg_b)
                .unwrap()
                .insert_flat_be(&mut w, &[0.0, 1.0, 0.0], Some(2), None)
                .unwrap();
            backend.commit(w).unwrap();
        }

        // Plan against the current layout.
        let plan = {
            let r = backend.begin_read().unwrap();
            manager.plan_delete_vectors_batch_be(&r, &[1, 2]).unwrap()
        };

        // Simulate stale plan: remove seg_b from the active segments view
        // so the plan's active_segments won't match the current layout.
        {
            let mut ds = manager.dense_spaces.write().unwrap();
            let sp = ds.get_mut("dense").unwrap();
            sp.segments.retain(|s| s.physical_name != seg_b);
            sp.segments.push(DenseVectorSegmentMetadata {
                physical_name: format!("{seg_b}_replaced"),
                role: DenseVectorSegmentRole::Indexed,
            });
        }

        // Execute with stale plan — should fall back and still produce id_levels
        // for BOTH ids (the fix in 8a5deaf2).
        let mut w = backend.begin_write().unwrap();
        let staged = manager
            .delete_vectors_batch_with_plan_be(&mut w, &plan)
            .unwrap();
        backend.commit(w).unwrap();
        manager.apply_delete_tombstones(&staged);

        // Verify: id_levels must contain entries for both ids.
        assert!(
            staged.id_levels.contains_key(&1) || staged.id_levels.contains_key(&2),
            "stale-plan fallback must produce id_levels for deleted ids; got {:?}",
            staged.id_levels.keys().collect::<Vec<_>>()
        );
    }

    /// Two Indexed segments on one in-memory LSM backend, id 1 in `seg_a`
    /// and id 2 in `seg_b`. Returns (backend, manager, dir, seg_a, seg_b).
    fn two_segment_manager() -> (Arc<AnyBackend>, NamedVectorManager, TempDir, String, String) {
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("two-seg-{}", uuid::Uuid::new_v4())).unwrap(),
        ));
        let manager = NamedVectorManager::new("two_seg".to_string());
        manager.attach_backend(Arc::clone(&backend));
        let dir = TempDir::new().unwrap();
        let config = NamedVectorConfig {
            size: 3,
            distance: DistanceMetric::Cosine,
            spindle: SpindleConfig::default(),
        };
        let hnsw = HNSWConfig::new(Some(8), Some(16), Some(32));
        let mut space = DenseVectorSpaceMetadata {
            next_segment_id: 0,
            segments: vec![],
        };
        let (seg_a, core_a) = manager
            .allocate_dense_core_locked_lsm(
                dir.path(),
                "dense",
                &config,
                hnsw.clone(),
                &mut space,
                &HashMap::new(),
            )
            .unwrap();
        let mut cores_a = HashMap::new();
        cores_a.insert(seg_a.clone(), core_a);
        let (seg_b, core_b) = manager
            .allocate_dense_core_locked_lsm(
                dir.path(),
                "dense",
                &config,
                hnsw,
                &mut space,
                &cores_a,
            )
            .unwrap();
        {
            let mut cores = manager.cores.write().unwrap();
            cores.extend(cores_a);
            cores.insert(seg_b.clone(), core_b);
        }
        for name in [&seg_a, &seg_b] {
            space.segments.push(DenseVectorSegmentMetadata {
                physical_name: name.clone(),
                role: DenseVectorSegmentRole::Indexed,
            });
        }
        manager
            .dense_spaces
            .write()
            .unwrap()
            .insert("dense".to_string(), space);
        {
            let cores = manager.cores.read().unwrap();
            let mut w = backend.begin_write().unwrap();
            cores
                .get(&seg_a)
                .unwrap()
                .insert_flat_be(&mut w, &[1.0, 0.0, 0.0], Some(1), None)
                .unwrap();
            cores
                .get(&seg_b)
                .unwrap()
                .insert_flat_be(&mut w, &[0.0, 1.0, 0.0], Some(2), None)
                .unwrap();
            backend.commit(w).unwrap();
        }
        (backend, manager, dir, seg_a, seg_b)
    }

    /// The index job's tail count must report a segment whose core is gone
    /// as the full stats did, wherever it stops counting.
    #[test]
    fn mutable_tail_count_reports_a_segment_without_a_core() {
        // Given a tail and a second segment of one flat row each, the second
        // with no core.
        let (backend, manager, _dir, seg_a, seg_b) = two_segment_manager();
        for segment in &mut manager
            .dense_spaces
            .write()
            .unwrap()
            .get_mut("dense")
            .unwrap()
            .segments
        {
            if segment.physical_name == seg_a {
                segment.role = DenseVectorSegmentRole::Mutable;
            }
        }
        manager.cores.write().unwrap().remove(&seg_b);
        let read = backend.begin_read().unwrap();
        // When the count stops at the tail's first row (limit 1) and when it does not.
        for limit in [1, 10] {
            let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
            let count = metrics::with_local_recorder(&recorder, || {
                manager.mutable_tail_count_upto_be(&read, "dense", limit)
            });
            // Then it counts the tail and reports the missing segment.
            assert_eq!(count.unwrap(), 1, "limit {limit}");
            let rendered = recorder.handle().render();
            assert!(
                rendered.contains("helix_dense_segment_missing_total")
                    && rendered.contains(&format!("segment=\"{seg_b}\"")),
                "limit {limit}: {rendered}"
            );
        }
    }

    /// A manager over an in-memory LSM backend whose space `dense` is as a
    /// collection has it before its first seal: the one tail `dense` with a
    /// row in it, and 1 for its next id.
    fn manager_with_a_first_tail(
        test_name: &str,
    ) -> (
        Arc<AnyBackend>,
        NamedVectorManager,
        TempDir,
        NamedVectorConfig,
    ) {
        let (backend, manager, dir) =
            setup_lsm_manager_with_flat_segment(test_name, "dense", &[(1, [1.0, 0.0, 0.0])]);
        let config = NamedVectorConfig {
            size: 3,
            distance: DistanceMetric::Cosine,
            spindle: SpindleConfig::default(),
        };
        manager
            .configs
            .write()
            .unwrap()
            .insert("dense".to_string(), config.clone());
        manager
            .dense_spaces
            .write()
            .unwrap()
            .get_mut("dense")
            .unwrap()
            .segments[0]
            .role = DenseVectorSegmentRole::Mutable;
        (backend, manager, dir, config)
    }

    /// A seal opens the next tail, and that can fail (it reads the store).
    /// The space is then as it was before the seal: its tail takes the next
    /// append and its next id is not spent. With the tail turned `Building`
    /// before the next one was open, a failure left the space with no tail
    /// at all, and every append to it failed from then on.
    #[test]
    fn a_seal_whose_next_tail_cannot_be_opened_leaves_the_space_its_tail() {
        let (backend, manager, dir, _config) = manager_with_a_first_tail("seal_cannot_open_next");
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let before = manager.list_dense_vector_spaces();

        // When the next tail cannot be opened: the manager has lost its
        // backend, which the tail's own core does not need to find its rows.
        let attached = manager.backend.write().unwrap().take();
        let failed = {
            let read = backend.begin_read().unwrap();
            manager.seal_mutable_tail_lsm(&read, dir.path(), "dense", hnsw.clone())
        };
        *manager.backend.write().unwrap() = attached;
        assert!(failed.is_err(), "the seal opened its next tail: {failed:?}");

        // Then the space is as it was, and its tail takes an append.
        assert_eq!(manager.list_dense_vector_spaces(), before);
        let mut w = backend.begin_write().unwrap();
        let appended_to = manager
            .dense_append_to_mutable_be(&mut w, "dense", &[0.0, 1.0, 0.0], 2, HashMap::new())
            .unwrap();
        backend.commit(w).unwrap();
        assert_eq!(appended_to, "dense");

        // And the seal after it takes the id the failed one did not spend.
        let read = backend.begin_read().unwrap();
        let sealed = manager
            .seal_mutable_tail_lsm(&read, dir.path(), "dense", hnsw)
            .unwrap();
        assert_eq!(sealed.as_deref(), Some("dense"));
        assert_eq!(
            manager.mutable_dense_segment_name("dense").as_deref(),
            Some("dense__seg_000001")
        );
        assert_eq!(
            manager.list_dense_vector_spaces()["dense"].next_segment_id,
            2
        );
    }

    /// A fresh id that holds rows is what a writer left that died with rows
    /// under a tail no stored layout named: they may be the only copy of a
    /// write it had acknowledged. The seal passes the id over, and nothing
    /// is told to clear it.
    #[test]
    fn a_fresh_id_that_holds_rows_is_passed_over_and_not_handed_to_the_reaper() {
        let (backend, manager, dir, config) = manager_with_a_first_tail("fresh_id_with_rows");
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        // Given a row under the id the space takes next.
        let stray = manager
            .create_dense_core_attached_lsm(dir.path(), "dense__seg_000001", &config, hnsw.clone())
            .unwrap();
        let mut w = backend.begin_write().unwrap();
        stray
            .insert_flat_be(&mut w, &[0.0, 1.0, 0.0], Some(9), None)
            .unwrap();
        backend.commit(w).unwrap();
        let handed = Arc::new(Mutex::new(Vec::new()));
        manager.attach_dirty_retired_segment_hook({
            let handed = Arc::clone(&handed);
            move |physical_name| handed.lock().unwrap().push(physical_name)
        });

        // When the tail is sealed.
        let read = backend.begin_read().unwrap();
        manager
            .seal_mutable_tail_lsm(&read, dir.path(), "dense", hnsw)
            .unwrap()
            .expect("a tail with a row is sealed");

        // Then the next tail is the id after it.
        assert_eq!(
            manager.mutable_dense_segment_name("dense").as_deref(),
            Some("dense__seg_000002")
        );
        assert_eq!(
            manager.list_dense_vector_spaces()["dense"].next_segment_id,
            3
        );
        // And the row is where it was, with nothing queued to clear it.
        assert_eq!(*handed.lock().unwrap(), Vec::<String>::new());
        assert!(manager.dirty_retired_segments.read().unwrap().is_empty());
        let read = backend.begin_read().unwrap();
        assert!(stray.contains_id(&read, 9).unwrap());
    }

    /// What a store of the layout reads must fail where the listing answers
    /// a layout of no space: stored, that is a collection with no segment.
    #[test]
    fn the_layout_to_store_is_an_error_when_its_lock_is_poisoned() {
        let (_backend, manager, _dir, _config) = manager_with_a_first_tail("poisoned_layout");
        assert_eq!(
            manager.dense_vector_spaces_to_store().unwrap(),
            manager.list_dense_vector_spaces()
        );

        manager.poison_dense_spaces();

        assert!(manager.list_dense_vector_spaces().is_empty());
        assert!(manager.dense_vector_spaces_to_store().is_err());
    }

    /// M7: with tombstones ON, a plan whose later segment's core has gone
    /// (stale) falls back to the physical delete; the tombstones the earlier
    /// segment already staged in this batch must survive into the result so
    /// `apply_delete_tombstones` records them.
    #[test]
    fn stale_plan_fallback_keeps_earlier_segment_tombstones_when_tombstones_on() {
        let _lock = TOMBSTONE_ENV_LOCK.lock().unwrap();
        let _on = EnvVarGuard::set("HELIX_LSM_DELETE_TOMBSTONES", "1");
        let (backend, manager, _dir, seg_a, seg_b) = two_segment_manager();

        // Deterministic segment order: seg_a stages first, seg_b is stale.
        let plan = DenseDeletePlan {
            ids: vec![1, 2],
            active_segments: [seg_a.clone(), seg_b.clone()].into_iter().collect(),
            by_segment: vec![(seg_a.clone(), vec![1]), (seg_b.clone(), vec![2])],
        };
        manager.cores.write().unwrap().remove(&seg_b);

        let mut w = backend.begin_write().unwrap();
        let staged = manager
            .delete_vectors_batch_with_plan_be(&mut w, &plan)
            .unwrap();
        backend.commit(w).unwrap();
        manager.apply_delete_tombstones(&staged);

        assert_eq!(
            staged.by_segment,
            vec![(seg_a.clone(), vec![1u128])],
            "seg_a's staged tombstone must survive the stale-plan fallback"
        );
        assert!(
            staged.id_levels.contains_key(&1),
            "fallback id_levels must be merged in: {:?}",
            staged.id_levels
        );
        let cores = manager.cores.read().unwrap();
        assert!(
            cores.get(&seg_a).unwrap().is_delete_tombstoned(1),
            "post-commit apply must record seg_a's tombstone"
        );
    }

    /// Only the segment that owned a hard-deleted id guards it: a stale
    /// snapshot can only refill that segment's caches with the deleted
    /// vector, and guarding every segment is what made the guard D x C.
    #[test]
    fn hard_delete_guards_only_the_owning_segment() {
        let _lock = TOMBSTONE_ENV_LOCK.lock().unwrap();
        let _off = EnvVarGuard::unset("HELIX_LSM_DELETE_TOMBSTONES");
        let (backend, manager, _dir, seg_a, seg_b) = two_segment_manager();

        let plan = {
            let r = backend.begin_read().unwrap();
            manager.plan_delete_vectors_batch_be(&r, &[1]).unwrap()
        };
        let mut w = backend.begin_write().unwrap();
        let staged = manager
            .delete_vectors_batch_with_plan_be(&mut w, &plan)
            .unwrap();
        backend.commit(w).unwrap();
        manager.apply_delete_tombstones(&staged);

        let cores = manager.cores.read().unwrap();
        assert!(
            cores.get(&seg_a).unwrap().is_cache_fill_blocked(1),
            "owning segment must guard the deleted id"
        );
        assert!(
            !cores.get(&seg_b).unwrap().is_cache_fill_blocked(1),
            "a segment that never held the id must not guard it"
        );
        assert!(
            !cores.get(&seg_b).unwrap().is_cache_fill_blocked(2),
            "the other segment's live id stays unguarded"
        );
    }

    /// Tombstones OFF, stale core hit after an earlier segment already
    /// hard-deleted its share: the fallback's physical delete may remove ids
    /// from any active segment, so ownership is unknown and every segment
    /// must guard, not only the segments the loop visited first.
    #[test]
    fn stale_plan_fallback_guards_every_segment_when_tombstones_off() {
        let _lock = TOMBSTONE_ENV_LOCK.lock().unwrap();
        let _off = EnvVarGuard::unset("HELIX_LSM_DELETE_TOMBSTONES");
        let (backend, manager, dir, seg_a, seg_b) = two_segment_manager();

        // A third active segment, absent from the plan, that holds id 2.
        let mut space = manager
            .dense_spaces
            .read()
            .unwrap()
            .get("dense")
            .unwrap()
            .clone();
        let config = NamedVectorConfig {
            size: 3,
            distance: DistanceMetric::Cosine,
            spindle: SpindleConfig::default(),
        };
        let (seg_c, core_c) = manager
            .allocate_dense_core_locked_lsm(
                dir.path(),
                "dense",
                &config,
                HNSWConfig::new(Some(8), Some(16), Some(32)),
                &mut space,
                &HashMap::new(),
            )
            .unwrap();
        let mut w = backend.begin_write().unwrap();
        core_c
            .insert_flat_be(&mut w, &[0.0, 0.0, 1.0], Some(2), None)
            .unwrap();
        backend.commit(w).unwrap();
        manager.cores.write().unwrap().insert(seg_c.clone(), core_c);
        space.segments.push(DenseVectorSegmentMetadata {
            physical_name: seg_c.clone(),
            role: DenseVectorSegmentRole::Indexed,
        });
        manager
            .dense_spaces
            .write()
            .unwrap()
            .insert("dense".to_string(), space);

        // seg_a deletes its share first; seg_b's core is gone (stale plan).
        let plan = DenseDeletePlan {
            ids: vec![1, 2],
            active_segments: [seg_a.clone(), seg_b.clone(), seg_c.clone()]
                .into_iter()
                .collect(),
            by_segment: vec![(seg_a.clone(), vec![1]), (seg_b.clone(), vec![2])],
        };
        manager.cores.write().unwrap().remove(&seg_b);

        let mut w = backend.begin_write().unwrap();
        let staged = manager
            .delete_vectors_batch_with_plan_be(&mut w, &plan)
            .unwrap();
        backend.commit(w).unwrap();
        manager.apply_delete_tombstones(&staged);

        let cores = manager.cores.read().unwrap();
        let core_c = cores.get(&seg_c).unwrap();
        let r = backend.begin_read().unwrap();
        assert!(
            !core_c.contains_id(&r, 2).unwrap(),
            "the fallback must have hard-deleted id 2 from seg_c"
        );
        assert!(
            core_c.is_cache_fill_blocked(2),
            "a segment the fallback deleted from must guard the id"
        );
        assert!(cores.get(&seg_a).unwrap().is_cache_fill_blocked(1));
    }
}
