//! Dense segment abstraction for vector indexing.
//!
//! Segments decouple ingest from indexing:
//! - **MutableSegment**: flat storage, appendable, brute-force search
//! - **IndexedSegment**: HNSW index, searchable via graph traversal
//! - **SegmentManager**: lifecycle manager per named vector space
//! - **SegmentOptimizer**: synchronous seal / merge / vacuum triggers
//!
//! **Copy-on-write safety (Phase E)**: Merge and vacuum operations collect
//! vector data from old segments via a read-only snapshot, then build the
//! replacement segment and swap it in atomically. LMDB MVCC guarantees that
//! concurrent readers on existing `RoTxn` snapshots see the old data until
//! they open a new transaction, so the swap is invisible to in-flight queries.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

use heed3::{Env, RoTxn, RwTxn};

use crate::helix_engine::storage_core::backend_any::{AnyBackend, AnyRead};
use crate::helix_engine::types::VectorError;
use crate::helix_engine::vector_core::hnsw::HNSW;
use crate::helix_engine::vector_core::named_vectors::DistanceMetric;
use crate::helix_engine::vector_core::spindle::SpindleConfig;
use crate::helix_engine::vector_core::vector::HVector;
use crate::helix_engine::vector_core::vector_core::{HNSWConfig, VectorCore};
use crate::protocol::value::Value;

// ─── MutableSegment ─────────────────────────────────────────────────────────

/// Flat vector storage. Appendable, searched via brute-force scan.
pub struct MutableSegment {
    /// Underlying VectorCore with flat-only storage (no HNSW entry point).
    pub(crate) core: VectorCore,
    /// Monotonic segment id for unique DB prefixes.
    pub(crate) seg_id: u64,
}

impl MutableSegment {
    /// Create a new mutable segment with namespaced LMDB databases.
    pub fn new(
        env: &Env,
        txn: &mut RwTxn,
        name: &str,
        seg_id: u64,
        config: HNSWConfig,
        distance_metric: DistanceMetric,
        spindle: SpindleConfig,
        backend: Arc<AnyBackend>,
    ) -> Result<Self, VectorError> {
        let prefix = format!("{}_seg_mut_{}", name, seg_id);
        let core =
            VectorCore::new_named(env, txn, &prefix, config, distance_metric, spindle, backend)?;
        Ok(Self { core, seg_id })
    }

    /// Insert a vector into flat storage (no HNSW links).
    pub fn insert(
        &self,
        txn: &mut RwTxn,
        data: &[f32],
        id: Option<u128>,
        fields: Option<HashMap<String, Value>>,
    ) -> Result<HVector, VectorError> {
        self.core.insert_flat(txn, data, id, fields)
    }

    /// Brute-force search over all vectors in this segment.
    pub fn search<F>(
        &self,
        r: &AnyRead<'_>,
        query: &[f32],
        k: usize,
        filter: Option<&[F]>,
    ) -> Result<Vec<HVector>, VectorError>
    where
        F: Fn(&HVector) -> bool,
    {
        self.core
            .search_with_selectivity(r, query, k, filter, false, None)
    }

    /// Number of vectors stored in this mutable segment.
    pub fn count(&self, r: &AnyRead<'_>) -> Result<u64, VectorError> {
        self.core.level_zero_count(r)
    }

    /// Seal this mutable segment: bulk-build HNSW and return an IndexedSegment.
    /// After sealing, the mutable segment's VectorCore becomes an indexed core.
    pub fn seal(self, txn: &mut RwTxn) -> Result<IndexedSegment, VectorError> {
        self.core.build_index_from_flat(txn)?;
        Ok(IndexedSegment {
            core: self.core,
            seg_id: self.seg_id,
            deleted_count: 0,
        })
    }
}

// ─── IndexedSegment ─────────────────────────────────────────────────────────

/// HNSW-indexed segment. Searchable via graph traversal.
pub struct IndexedSegment {
    /// Underlying VectorCore with built HNSW graph.
    pub(crate) core: VectorCore,
    /// Segment id for tracking.
    #[allow(dead_code)]
    pub(crate) seg_id: u64,
    /// Number of tombstoned (deleted) vectors in this segment.
    pub(crate) deleted_count: u64,
}

impl IndexedSegment {
    /// HNSW search with optional selectivity hint.
    pub fn search<F>(
        &self,
        r: &AnyRead<'_>,
        query: &[f32],
        k: usize,
        filter: Option<&[F]>,
        should_trickle: bool,
        selectivity: Option<f32>,
    ) -> Result<Vec<HVector>, VectorError>
    where
        F: Fn(&HVector) -> bool,
    {
        self.core
            .search_with_selectivity(r, query, k, filter, should_trickle, selectivity)
    }

    /// Incremental insert into the existing HNSW graph.
    pub fn insert(
        &self,
        txn: &mut RwTxn,
        data: &[f32],
        id: Option<u128>,
        fields: Option<HashMap<String, Value>>,
    ) -> Result<HVector, VectorError> {
        self.core
            .insert::<fn(&HVector) -> bool>(txn, data, id, fields)
    }

    /// Delete a vector from this indexed segment.
    pub fn delete(&mut self, txn: &mut RwTxn, id: u128) -> Result<(), VectorError> {
        match self.core.delete_vector(txn, id) {
            Ok(()) => {
                self.deleted_count += 1;
                Ok(())
            }
            Err(VectorError::VectorNotFound(_)) => Ok(()), // not in this segment
            Err(e) => Err(e),
        }
    }

    /// Total stored points (including deleted/tombstoned).
    pub fn point_count(&self, r: &AnyRead<'_>) -> Result<u64, VectorError> {
        self.core.level_zero_count(r)
    }

    /// Number of tombstoned points.
    pub fn deleted_count(&self) -> u64 {
        self.deleted_count
    }

    /// Check if this segment has an HNSW index.
    pub fn has_index(&self, r: &AnyRead<'_>) -> Result<bool, VectorError> {
        self.core.has_index(r)
    }

    /// Public score conversion (distance -> user-facing score).
    pub fn public_score(&self, distance: f32) -> f32 {
        self.core.public_score(distance)
    }
}

// ─── DenseSegment ───────────────────────────────────────────────────────────

/// A dense vector segment is either mutable (flat, appendable) or indexed (HNSW).
pub enum DenseSegment {
    Mutable(MutableSegment),
    Indexed(IndexedSegment),
}

// ─── SegmentManager ─────────────────────────────────────────────────────────

/// Manages the lifecycle of dense segments for a single named vector space.
///
/// Writes go to the active mutable segment. When it crosses the threshold,
/// the optimizer seals it (builds HNSW) and opens a fresh mutable segment.
/// Search fuses results from all segments (indexed via HNSW + mutable via flat scan).
pub struct SegmentManager {
    /// The vector space name (e.g., "dense").
    name: String,
    /// Current write target (flat storage).
    active_mutable: Option<MutableSegment>,
    /// Sealed, HNSW-indexed segments.
    indexed_segments: Vec<IndexedSegment>,
    /// When to seal the mutable segment (point count threshold).
    segment_threshold: usize,
    /// Monotonic counter for segment IDs.
    next_seg_id: u64,
    /// HNSW config for building new segments.
    hnsw_config: HNSWConfig,
    /// Distance metric for this vector space.
    distance_metric: DistanceMetric,
    /// Compression config.
    spindle: SpindleConfig,
    /// Shared storage backend handle threaded into every segment core
    /// (US-006 plumbing). Shared via `Arc` because `AnyBackend` is not
    /// `Clone`.
    backend: Arc<AnyBackend>,
}

impl SegmentManager {
    /// Create a new SegmentManager for a named vector space.
    pub fn new(
        env: &Env,
        txn: &mut RwTxn,
        name: &str,
        segment_threshold: usize,
        hnsw_config: HNSWConfig,
        distance_metric: DistanceMetric,
        spindle: SpindleConfig,
        backend: Arc<AnyBackend>,
    ) -> Result<Self, VectorError> {
        let mutable = MutableSegment::new(
            env,
            txn,
            name,
            0,
            hnsw_config.clone(),
            distance_metric.clone(),
            spindle.clone(),
            Arc::clone(&backend),
        )?;

        Ok(Self {
            name: name.to_string(),
            active_mutable: Some(mutable),
            indexed_segments: Vec::new(),
            segment_threshold,
            next_seg_id: 1,
            hnsw_config,
            distance_metric,
            spindle,
            backend,
        })
    }

    /// Insert a vector. Routes to the active mutable segment.
    pub fn insert(
        &mut self,
        env: &Env,
        txn: &mut RwTxn,
        data: &[f32],
        id: Option<u128>,
        fields: Option<HashMap<String, Value>>,
    ) -> Result<HVector, VectorError> {
        // Ensure we have an active mutable segment
        if self.active_mutable.is_none() {
            self.open_new_mutable(env, txn)?;
        }

        let mutable = self.active_mutable.as_ref().unwrap();
        let result = mutable.insert(txn, data, id, fields)?;

        // Check if we should seal (optimizer trigger)
        let count = {
            let rd = self.backend.read_borrowed(&*txn);
            mutable.count(&rd)?
        };
        if count as usize >= self.segment_threshold && self.segment_threshold > 0 {
            self.seal_active(env, txn)?;
        }

        Ok(result)
    }

    /// Insert into flat storage without triggering seal.
    /// Used for bulk ingest below the threshold.
    pub fn insert_flat(
        &mut self,
        env: &Env,
        txn: &mut RwTxn,
        data: &[f32],
        id: Option<u128>,
        fields: Option<HashMap<String, Value>>,
    ) -> Result<HVector, VectorError> {
        if self.active_mutable.is_none() {
            self.open_new_mutable(env, txn)?;
        }
        self.active_mutable
            .as_ref()
            .unwrap()
            .insert(txn, data, id, fields)
    }

    /// Multi-segment search: fuse results from all indexed segments (HNSW)
    /// and the active mutable segment (flat scan), return top-k.
    pub fn search<F>(
        &self,
        txn: &RoTxn,
        query: &[f32],
        k: usize,
        filter: Option<&[F]>,
        should_trickle: bool,
        selectivity_hint: Option<f32>,
    ) -> Result<Vec<HVector>, VectorError>
    where
        F: Fn(&HVector) -> bool,
    {
        let search_start = std::time::Instant::now();
        let mut all_candidates: Vec<HVector> = Vec::with_capacity(
            self.indexed_segments
                .len()
                .saturating_add(if self.active_mutable.is_some() { 1 } else { 0 })
                .saturating_mul(k),
        );
        let rd = self.backend.read_borrowed(txn);

        // Search each indexed segment via HNSW
        let indexed_start = std::time::Instant::now();
        for segment in &self.indexed_segments {
            let results =
                segment.search(&rd, query, k, filter, should_trickle, selectivity_hint)?;
            all_candidates.extend(results);
        }
        metrics::histogram!("helix_segment_manager_search_phase_ms", "phase" => "indexed")
            .record(indexed_start.elapsed().as_secs_f64() * 1000.0);

        // Search the active mutable segment via flat scan
        let mutable_start = std::time::Instant::now();
        if let Some(mutable) = &self.active_mutable {
            let count = mutable.count(&rd)?;
            if count > 0 {
                let results = mutable.search(&rd, query, k, filter)?;
                all_candidates.extend(results);
            }
        }
        metrics::histogram!("helix_segment_manager_search_phase_ms", "phase" => "mutable")
            .record(mutable_start.elapsed().as_secs_f64() * 1000.0);
        metrics::histogram!("helix_segment_manager_search_candidates")
            .record(all_candidates.len() as f64);
        metrics::histogram!("helix_segment_manager_search_indexed_segments")
            .record(self.indexed_segments.len() as f64);

        // Merge by distance (ascending — lower distance = better match)
        let merge_start = std::time::Instant::now();
        all_candidates.sort_by(|a, b| {
            a.get_distance()
                .partial_cmp(&b.get_distance())
                .unwrap_or(Ordering::Equal)
        });
        all_candidates.truncate(k);
        metrics::histogram!("helix_segment_manager_search_phase_ms", "phase" => "merge")
            .record(merge_start.elapsed().as_secs_f64() * 1000.0);
        metrics::histogram!("helix_segment_manager_search_phase_ms", "phase" => "total")
            .record(search_start.elapsed().as_secs_f64() * 1000.0);

        Ok(all_candidates)
    }

    /// Delete a vector from all segments.
    pub fn delete(&mut self, txn: &mut RwTxn, id: u128) -> Result<(), VectorError> {
        // Delete from indexed segments
        for segment in &mut self.indexed_segments {
            segment.delete(txn, id)?;
        }

        // Delete from active mutable segment
        if let Some(mutable) = &self.active_mutable {
            match mutable.core.delete_vector(txn, id) {
                Ok(()) => {}
                Err(VectorError::VectorNotFound(_)) => {}
                Err(e) => return Err(e),
            }
        }

        Ok(())
    }

    /// Seal the active mutable segment: build HNSW, move to indexed, open new mutable.
    pub fn seal_active(&mut self, env: &Env, txn: &mut RwTxn) -> Result<(), VectorError> {
        if let Some(mutable) = self.active_mutable.take() {
            let count = {
                let rd = self.backend.read_borrowed(&*txn);
                mutable.count(&rd)?
            };
            if count > 0 {
                let indexed = mutable.seal(txn)?;
                self.indexed_segments.push(indexed);
            }
            // Open a fresh mutable segment
            self.open_new_mutable(env, txn)?;
        }
        Ok(())
    }

    /// Open a new mutable segment.
    fn open_new_mutable(&mut self, env: &Env, txn: &mut RwTxn) -> Result<(), VectorError> {
        let seg_id = self.next_seg_id;
        self.next_seg_id += 1;
        let mutable = MutableSegment::new(
            env,
            txn,
            &self.name,
            seg_id,
            self.hnsw_config.clone(),
            self.distance_metric.clone(),
            self.spindle.clone(),
            Arc::clone(&self.backend),
        )?;
        self.active_mutable = Some(mutable);
        Ok(())
    }

    /// Returns true if any indexed segment has HNSW state.
    pub fn has_index(&self, txn: &RoTxn) -> Result<bool, VectorError> {
        let rd = self.backend.read_borrowed(txn);
        for segment in &self.indexed_segments {
            if segment.has_index(&rd)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Total level-0 vector count across all segments.
    pub fn total_count(&self, txn: &RoTxn) -> Result<u64, VectorError> {
        let rd = self.backend.read_borrowed(txn);
        let mut total = 0u64;
        for segment in &self.indexed_segments {
            total += segment.point_count(&rd)?;
        }
        if let Some(mutable) = &self.active_mutable {
            total += mutable.count(&rd)?;
        }
        Ok(total)
    }

    /// Number of indexed segments.
    pub fn indexed_segment_count(&self) -> usize {
        self.indexed_segments.len()
    }

    /// Public score conversion.
    pub fn public_score(&self, distance: f32) -> f32 {
        // All segments share the same distance metric, so use the mutable core's conversion.
        // If no mutable exists, use the first indexed segment.
        if let Some(mutable) = &self.active_mutable {
            return mutable.core.public_score(distance);
        }
        if let Some(indexed) = self.indexed_segments.first() {
            return indexed.public_score(distance);
        }
        // Fallback: identity
        distance
    }

    /// Access the segment threshold.
    pub fn segment_threshold(&self) -> usize {
        self.segment_threshold
    }
}

// ─── SegmentOptimizer ───────────────────────────────────────────────────────

/// Copy-on-write segment optimizer.
///
/// Merge and vacuum follow a two-phase pattern for safety:
/// 1. **Read phase**: collect vector data from old segments (can use the
///    existing `RwTxn` as a reader — LMDB readers never block).
/// 2. **Write phase**: build replacement segment from collected data and
///    swap it in atomically.
///
/// Because LMDB uses MVCC, concurrent readers on existing `RoTxn` snapshots
/// continue to see the old segments until they open a new transaction.
/// This means the swap is invisible to in-flight queries — no proxy needed.
pub struct SegmentOptimizer;

/// Maximum number of indexed segments before merge is triggered.
const MAX_INDEXED_SEGMENTS: usize = 5;

/// Fraction of deleted points that triggers vacuum.
const VACUUM_DELETED_FRACTION: f64 = 0.20;

/// Collected vector data for COW rebuild (read phase output).
struct CollectedVectors {
    vectors: Vec<(u128, Vec<f32>, HashMap<String, Value>)>,
}

impl SegmentOptimizer {
    /// Run all optimizer checks on the segment manager.
    /// This is synchronous — intended to be called at the end of upsert batches.
    /// Returns the number of optimization actions taken.
    pub fn optimize(
        manager: &mut SegmentManager,
        env: &Env,
        txn: &mut RwTxn,
    ) -> Result<usize, VectorError> {
        let mut actions = 0;

        // 1. Seal trigger: mutable segment over threshold
        if let Some(mutable) = &manager.active_mutable {
            let count = {
                let rd = manager.backend.read_borrowed(&*txn);
                mutable.count(&rd)?
            };
            if count as usize >= manager.segment_threshold && manager.segment_threshold > 0 {
                manager.seal_active(env, txn)?;
                actions += 1;
            }
        }

        // 2. Merge trigger: too many indexed segments
        if manager.indexed_segments.len() > MAX_INDEXED_SEGMENTS {
            Self::merge_smallest_segments(manager, env, txn)?;
            actions += 1;
        }

        // 3. Vacuum trigger: indexed segment with >20% deleted points
        let mut vacuum_indices = Vec::new();
        {
            let rd = manager.backend.read_borrowed(&*txn);
            for (i, segment) in manager.indexed_segments.iter().enumerate() {
                let total = segment.point_count(&rd)?;
                if total > 0 {
                    let deleted = segment.deleted_count();
                    let fraction = deleted as f64 / total as f64;
                    if fraction > VACUUM_DELETED_FRACTION {
                        vacuum_indices.push(i);
                    }
                }
            }
        }

        // Vacuum in reverse order so indices stay valid
        for &i in vacuum_indices.iter().rev() {
            Self::vacuum_segment(manager, env, txn, i)?;
            actions += 1;
        }

        Ok(actions)
    }

    // ── COW Read Phase ──

    /// Collect all live vectors from the given segment indices.
    /// This is the read phase — only reads LMDB, does not mutate.
    fn collect_from_segments(
        manager: &SegmentManager,
        txn: &RwTxn,
        indices: &[usize],
    ) -> Result<CollectedVectors, VectorError> {
        let mut vectors = Vec::new();
        for &idx in indices {
            let seg = &manager.indexed_segments[idx];
            let rd = seg.core.backend.read_borrowed(txn);
            let seg_vectors = seg.core.get_all_vectors(&rd, Some(0))?;
            for v in seg_vectors {
                let id = v.get_id();
                let data = v.get_data().to_vec();
                let fields = seg.core.debug_fields_internal(&rd, id);
                vectors.push((id, data, fields));
            }
        }
        Ok(CollectedVectors { vectors })
    }

    // ── COW Write Phase ──

    /// Build a new indexed segment from collected vectors and return it.
    /// This is the write phase — creates new LMDB databases and builds HNSW.
    fn build_replacement(
        manager: &mut SegmentManager,
        env: &Env,
        txn: &mut RwTxn,
        collected: &CollectedVectors,
    ) -> Result<IndexedSegment, VectorError> {
        let seg_id = manager.next_seg_id;
        manager.next_seg_id += 1;
        let new_segment = MutableSegment::new(
            env,
            txn,
            &manager.name,
            seg_id,
            manager.hnsw_config.clone(),
            manager.distance_metric.clone(),
            manager.spindle.clone(),
            Arc::clone(&manager.backend),
        )?;

        for (id, data, fields) in &collected.vectors {
            let f = if fields.is_empty() {
                None
            } else {
                Some(fields.clone())
            };
            new_segment.insert(txn, data, Some(*id), f)?;
        }

        new_segment.seal(txn)
    }

    /// Merge the two smallest indexed segments into one.
    /// Uses COW: read from old segments, build replacement, swap atomically.
    fn merge_smallest_segments(
        manager: &mut SegmentManager,
        env: &Env,
        txn: &mut RwTxn,
    ) -> Result<(), VectorError> {
        if manager.indexed_segments.len() < 2 {
            return Ok(());
        }

        // Find the two smallest segments by point count
        let mut sizes: Vec<(usize, u64)> = {
            let rd = manager.backend.read_borrowed(&*txn);
            manager
                .indexed_segments
                .iter()
                .enumerate()
                .map(|(i, seg)| {
                    let count = seg.point_count(&rd).unwrap_or(0);
                    (i, count)
                })
                .collect()
        };
        sizes.sort_by_key(|&(_, count)| count);

        if sizes.len() < 2 {
            return Ok(());
        }

        let idx_a = sizes[0].0;
        let idx_b = sizes[1].0;

        // Phase 1 (read): collect vectors from old segments
        let collected = Self::collect_from_segments(manager, txn, &[idx_a, idx_b])?;

        // Phase 2 (write): build replacement segment
        let indexed = Self::build_replacement(manager, env, txn, &collected)?;

        // Phase 3 (swap): remove old segments, add new one — atomic in this txn
        let (first, second) = if idx_a > idx_b {
            (idx_a, idx_b)
        } else {
            (idx_b, idx_a)
        };
        manager.indexed_segments.remove(first);
        manager.indexed_segments.remove(second);
        manager.indexed_segments.push(indexed);

        Ok(())
    }

    /// Vacuum a segment: rebuild without deleted vectors.
    /// Uses COW: read live vectors, build replacement, swap atomically.
    fn vacuum_segment(
        manager: &mut SegmentManager,
        env: &Env,
        txn: &mut RwTxn,
        segment_idx: usize,
    ) -> Result<(), VectorError> {
        if segment_idx >= manager.indexed_segments.len() {
            return Ok(());
        }

        // Phase 1 (read): collect live vectors from the segment
        let collected = Self::collect_from_segments(manager, txn, &[segment_idx])?;

        if collected.vectors.is_empty() {
            // Segment is empty after vacuum — just remove it
            manager.indexed_segments.remove(segment_idx);
            return Ok(());
        }

        // Phase 2 (write): build replacement segment from live vectors only
        let indexed = Self::build_replacement(manager, env, txn, &collected)?;

        // Phase 3 (swap): replace the old segment atomically
        manager.indexed_segments[segment_idx] = indexed;

        Ok(())
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helix_engine::vector_core::spindle::{SpindleConfig, SpindleMode};
    use rand::SeedableRng;
    use tempfile::TempDir;

    fn setup() -> (heed3::Env, TempDir) {
        let dir = TempDir::new().unwrap();
        let env = unsafe {
            heed3::EnvOpenOptions::new()
                .map_size(256 * 1024 * 1024)
                .max_dbs(128)
                .open(dir.path())
                .unwrap()
        };
        (env, dir)
    }

    /// Test-only backend handle sharing the segment env. 6a is pure
    /// plumbing — segment cores hold this Arc but no KV access routes
    /// through it yet.
    fn test_backend(env: &heed3::Env) -> Arc<AnyBackend> {
        use crate::helix_engine::storage_core::backend_lmdb::LmdbBackend;
        Arc::new(AnyBackend::Lmdb(LmdbBackend::from_env(env.clone())))
    }

    fn default_spindle() -> SpindleConfig {
        SpindleConfig {
            mode: SpindleMode::None,
            ..SpindleConfig::default()
        }
    }

    fn default_hnsw() -> HNSWConfig {
        HNSWConfig::new(Some(8), Some(32), Some(64))
    }

    fn random_vector(rng: &mut impl rand::Rng, dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|_| rand::Rng::random_range(rng, -1.0f32..1.0f32))
            .collect()
    }

    #[test]
    fn segment_seal_and_search() {
        let (env, _dir) = setup();
        let threshold = 20;

        let mut txn = env.write_txn().unwrap();
        let mut manager = SegmentManager::new(
            &env,
            &mut txn,
            "test_seal",
            threshold,
            default_hnsw(),
            DistanceMetric::Cosine,
            default_spindle(),
            test_backend(&env),
        )
        .unwrap();
        txn.commit().unwrap();

        // Insert more than threshold vectors
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let dim = 16;
        let n = 30;

        let mut txn = env.write_txn().unwrap();
        for i in 0..n {
            let data = random_vector(&mut rng, dim);
            manager
                .insert(&env, &mut txn, &data, Some((i + 1) as u128), None)
                .unwrap();
        }
        txn.commit().unwrap();

        // Should have sealed: at least 1 indexed segment
        assert!(
            manager.indexed_segment_count() >= 1,
            "Expected at least 1 indexed segment after inserting {} vectors with threshold {}",
            n,
            threshold
        );

        // Search should find results across segments
        let txn = env.read_txn().unwrap();
        let query = random_vector(&mut rng, dim);
        let results: Vec<HVector> = manager
            .search::<fn(&HVector) -> bool>(&txn, &query, 10, None, false, None)
            .unwrap();
        assert!(
            !results.is_empty(),
            "Search should return results after seal"
        );
        assert!(results.len() <= 10);
    }

    #[test]
    fn multi_segment_search_fuses_results() {
        let (env, _dir) = setup();
        // Use a small threshold so we get multiple segments
        let threshold = 10;

        let mut txn = env.write_txn().unwrap();
        let mut manager = SegmentManager::new(
            &env,
            &mut txn,
            "test_fuse",
            threshold,
            default_hnsw(),
            DistanceMetric::Cosine,
            default_spindle(),
            test_backend(&env),
        )
        .unwrap();
        txn.commit().unwrap();

        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let dim = 16;

        // Insert 25 vectors → should create 2 indexed segments + some in mutable
        let mut txn = env.write_txn().unwrap();
        for i in 0..25 {
            let data = random_vector(&mut rng, dim);
            manager
                .insert(&env, &mut txn, &data, Some((i + 1) as u128), None)
                .unwrap();
        }
        txn.commit().unwrap();

        // Verify we have multiple segments
        assert!(
            manager.indexed_segment_count() >= 2,
            "Expected at least 2 indexed segments, got {}",
            manager.indexed_segment_count()
        );

        // Search should return results from all segments
        let txn = env.read_txn().unwrap();
        let query = random_vector(&mut rng, dim);
        let results: Vec<HVector> = manager
            .search::<fn(&HVector) -> bool>(&txn, &query, 25, None, false, None)
            .unwrap();

        // Should find results (from both indexed and possibly mutable)
        assert!(
            results.len() >= 20,
            "Expected at least 20 results from multi-segment search, got {}",
            results.len()
        );

        // Results should be sorted by distance (ascending)
        for i in 1..results.len() {
            assert!(
                results[i].get_distance() >= results[i - 1].get_distance(),
                "Results should be sorted by distance (ascending)"
            );
        }
    }

    #[test]
    fn segment_delete_removes_from_all() {
        let (env, _dir) = setup();
        let threshold = 15;

        let mut txn = env.write_txn().unwrap();
        let mut manager = SegmentManager::new(
            &env,
            &mut txn,
            "test_del",
            threshold,
            default_hnsw(),
            DistanceMetric::Cosine,
            default_spindle(),
            test_backend(&env),
        )
        .unwrap();
        txn.commit().unwrap();

        let mut rng = rand::rngs::StdRng::seed_from_u64(77);
        let dim = 16;

        // Insert vectors
        let mut txn = env.write_txn().unwrap();
        for i in 0..20 {
            let data = random_vector(&mut rng, dim);
            manager
                .insert(&env, &mut txn, &data, Some((i + 1) as u128), None)
                .unwrap();
        }
        txn.commit().unwrap();

        // Delete vector 5
        let mut txn = env.write_txn().unwrap();
        manager.delete(&mut txn, 5).unwrap();
        txn.commit().unwrap();

        // Search should not return vector 5
        let txn = env.read_txn().unwrap();
        let query = random_vector(&mut rng, dim);
        let results: Vec<HVector> = manager
            .search::<fn(&HVector) -> bool>(&txn, &query, 20, None, false, None)
            .unwrap();
        assert!(
            !results.iter().any(|r| r.get_id() == 5),
            "Deleted vector 5 should not appear in search results"
        );
    }

    #[test]
    fn segment_merge_reduces_count() {
        let (env, _dir) = setup();
        // Very small threshold to create many segments
        let threshold = 5;

        let mut txn = env.write_txn().unwrap();
        let mut manager = SegmentManager::new(
            &env,
            &mut txn,
            "test_merge",
            threshold,
            default_hnsw(),
            DistanceMetric::Cosine,
            default_spindle(),
            test_backend(&env),
        )
        .unwrap();
        txn.commit().unwrap();

        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let dim = 16;

        // Insert enough vectors to create >5 segments (need >30 with threshold 5)
        let mut txn = env.write_txn().unwrap();
        for i in 0..35 {
            let data = random_vector(&mut rng, dim);
            manager
                .insert(&env, &mut txn, &data, Some((i + 1) as u128), None)
                .unwrap();
        }
        txn.commit().unwrap();

        let before_count = manager.indexed_segment_count();
        assert!(
            before_count > MAX_INDEXED_SEGMENTS,
            "Expected more than {} segments before merge, got {}",
            MAX_INDEXED_SEGMENTS,
            before_count
        );

        // Run optimizer to trigger merge
        let mut txn = env.write_txn().unwrap();
        let actions = SegmentOptimizer::optimize(&mut manager, &env, &mut txn).unwrap();
        txn.commit().unwrap();

        assert!(actions > 0, "Optimizer should have taken at least 1 action");
        let after_count = manager.indexed_segment_count();
        assert!(
            after_count < before_count,
            "Merge should reduce segment count: before={}, after={}",
            before_count,
            after_count
        );

        // Search should still work
        let txn = env.read_txn().unwrap();
        let query = random_vector(&mut rng, dim);
        let results: Vec<HVector> = manager
            .search::<fn(&HVector) -> bool>(&txn, &query, 10, None, false, None)
            .unwrap();
        assert!(!results.is_empty(), "Search should still work after merge");
    }

    #[test]
    fn segment_vacuum_removes_deleted() {
        let (env, _dir) = setup();
        let threshold = 20;

        let mut txn = env.write_txn().unwrap();
        let mut manager = SegmentManager::new(
            &env,
            &mut txn,
            "test_vacuum",
            threshold,
            default_hnsw(),
            DistanceMetric::Cosine,
            default_spindle(),
            test_backend(&env),
        )
        .unwrap();
        txn.commit().unwrap();

        let mut rng = rand::rngs::StdRng::seed_from_u64(55);
        let dim = 16;

        // Insert vectors and seal
        let mut txn = env.write_txn().unwrap();
        for i in 0..20 {
            let data = random_vector(&mut rng, dim);
            manager
                .insert(&env, &mut txn, &data, Some((i + 1) as u128), None)
                .unwrap();
        }
        txn.commit().unwrap();

        // Should have at least 1 indexed segment
        assert!(manager.indexed_segment_count() >= 1);

        // Delete >20% of vectors (delete 7 of 20 = 35%)
        let mut txn = env.write_txn().unwrap();
        for id in 1..=7u128 {
            manager.delete(&mut txn, id).unwrap();
        }
        txn.commit().unwrap();

        // Check that at least one segment has high deletion ratio
        let txn = env.read_txn().unwrap();
        let has_high_deletion = manager.indexed_segments.iter().any(|seg| {
            let total = seg
                .point_count(&seg.core.backend.read_borrowed(&txn))
                .unwrap_or(0);
            total > 0 && (seg.deleted_count() as f64 / total as f64) > VACUUM_DELETED_FRACTION
        });
        drop(txn);

        assert!(
            has_high_deletion,
            "At least one segment should have >20% deleted vectors"
        );

        // Run optimizer (should vacuum)
        let mut txn = env.write_txn().unwrap();
        let actions = SegmentOptimizer::optimize(&mut manager, &env, &mut txn).unwrap();
        txn.commit().unwrap();

        assert!(actions > 0, "Vacuum should have been triggered");

        // After vacuum, deleted_count should be reset on rebuilt segments
        let total_deleted: u64 = manager
            .indexed_segments
            .iter()
            .map(|s| s.deleted_count())
            .sum();
        // The vacuum rebuilt the segment without deleted vectors,
        // so deleted count should now be 0
        assert_eq!(
            total_deleted, 0,
            "After vacuum, deleted count should be 0, got {}",
            total_deleted
        );

        // Search should still work and not return deleted vectors
        let txn = env.read_txn().unwrap();
        let query = random_vector(&mut rng, dim);
        let results: Vec<HVector> = manager
            .search::<fn(&HVector) -> bool>(&txn, &query, 20, None, false, None)
            .unwrap();
        for r in &results {
            assert!(
                r.get_id() > 7,
                "Deleted vector {} should not appear in results",
                r.get_id()
            );
        }
    }
}
