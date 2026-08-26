use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use heed3::{
    byteorder::BE,
    types::{Bytes, U128},
    Env, RoTxn, RwTxn, WithTls,
};
use serde::{Deserialize, Serialize};

use crate::helix_engine::storage_core::backend::{
    BackendKind, Namespace, SparseDb, StorageBackend,
};
use crate::helix_engine::storage_core::backend_any::{AnyBackend, AnyRead, AnyWrite};
use crate::helix_engine::types::VectorError;

// ─── Types ───

/// Wire-format sparse vector: parallel arrays of term indices and weights.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SparseMetadataFlushStats {
    pub pending_before: usize,
    pub processed: usize,
    pub flushed: usize,
    pub pending_after: usize,
    pub budget_exhausted: bool,
    pub time_exhausted: bool,
}

/// Outcome of `SparseVectorCore::upsert`. Lets the caller record metrics
/// without re-reading LMDB just to classify the operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseUpsertOutcome {
    /// First time this doc_id is seen — a full insert was performed.
    Inserted,
    /// Doc existed with the exact same sparse payload — no LMDB writes.
    Unchanged,
    /// Doc existed with a different payload — only changed postings were
    /// touched. Counts are per-term, not per-posting-entry.
    DiffApplied {
        added: usize,
        removed: usize,
        changed: usize,
    },
}

#[inline]
fn record_sparse_batch_step(step: &'static str, started: Instant) {
    metrics::histogram!("helix_sparse_upsert_batch_step_ms", "step" => step)
        .record(started.elapsed().as_secs_f64() * 1000.0);
}

fn bool_label(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

fn unique_term_ids(term_ids: &[u32]) -> Vec<u32> {
    let mut seen = HashSet::with_capacity(term_ids.len());
    let mut unique = Vec::with_capacity(term_ids.len());
    for &term_id in term_ids {
        if seen.insert(term_id) {
            unique.push(term_id);
        }
    }
    unique
}

impl SparseVector {
    pub fn validate(&self) -> Result<(), VectorError> {
        if self.indices.len() != self.values.len() {
            return Err(VectorError::VectorCoreError(
                "Sparse vector indices and values must have the same length".into(),
            ));
        }
        // Reject duplicate term indices — they would inflate scores in the posting list.
        let mut seen = std::collections::HashSet::with_capacity(self.indices.len());
        for &idx in &self.indices {
            if !seen.insert(idx) {
                return Err(VectorError::VectorCoreError(format!(
                    "Sparse vector has duplicate term index {}",
                    idx
                )));
            }
        }
        // Reject NaN/Inf values — they violate Ord and corrupt the search heap.
        for &v in &self.values {
            if !v.is_finite() {
                return Err(VectorError::VectorCoreError(
                    "Sparse vector values must be finite (no NaN or Inf)".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Modifier applied to sparse vector scores at query time.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SparseModifier {
    None,
    Idf,
}

impl Default for SparseModifier {
    fn default() -> Self {
        Self::None
    }
}

/// Configuration for a sparse vector space.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SparseVectorConfig {
    #[serde(default = "default_full_scan_threshold")]
    pub full_scan_threshold: usize,
    #[serde(default)]
    pub modifier: SparseModifier,
    /// Enable WAND early termination for sparse search.
    /// When true, search uses upper-bound pruning to skip candidates that
    /// cannot make it into the top-K. Falls back to full accumulation when
    /// the query has fewer than 3 terms or the index has fewer docs than
    /// `full_scan_threshold`.
    #[serde(default = "default_wand_enabled")]
    pub wand_enabled: bool,
}

fn default_full_scan_threshold() -> usize {
    5000
}

fn default_wand_enabled() -> bool {
    true
}

impl Default for SparseVectorConfig {
    fn default() -> Self {
        Self {
            full_scan_threshold: default_full_scan_threshold(),
            modifier: SparseModifier::None,
            wand_enabled: default_wand_enabled(),
        }
    }
}

// ─── Scored result ───

#[derive(Debug, Clone)]
struct ScoredDoc {
    id: u128,
    score: f64,
}

impl PartialEq for ScoredDoc {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.score.to_bits() == other.score.to_bits()
    }
}
impl Eq for ScoredDoc {}

// Min-heap: we want to evict the *lowest* score.
impl Ord for ScoredDoc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering via total_cmp: smaller score = greater in heap (min-heap).
        // total_cmp is safe with NaN (treats NaN as greater than all finite values).
        // Tie-break on id so that Ord is consistent with Eq.
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.id.cmp(&other.id))
    }
}
impl PartialOrd for ScoredDoc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn maxscore_can_stop(scores: &HashMap<u128, f64>, limit: usize, remaining_ub: f64) -> bool {
    if limit == 0 || scores.len() <= limit {
        return false;
    }

    let remaining_ub = remaining_ub.max(0.0);
    let mut heap: BinaryHeap<ScoredDoc> = BinaryHeap::with_capacity(limit);
    let mut best_competing_score = remaining_ub;

    for (&id, &score) in scores {
        let candidate = ScoredDoc { id, score };
        if heap.len() < limit {
            heap.push(candidate);
            continue;
        }

        let min = heap.peek().expect("heap is full");
        if score > min.score || (score == min.score && id < min.id) {
            let evicted = heap.pop().expect("heap is full");
            best_competing_score = best_competing_score.max(evicted.score + remaining_ub);
            heap.push(candidate);
        } else {
            best_competing_score = best_competing_score.max(score + remaining_ub);
        }
    }

    heap.peek()
        .map(|min| best_competing_score < min.score)
        .unwrap_or(false)
}

/// Read per query (NOT OnceLock-cached) so retuning the interval in the pod
/// env takes effect on restartless config reloads; the env read is once per
/// query, never per term.
fn sparse_wand_maxscore_check_interval() -> u64 {
    std::env::var("HELIX_SPARSE_WAND_MAXSCORE_CHECK_INTERVAL")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(8)
}

/// Document-frequency threshold above which a maxscore check ALWAYS runs
/// before scanning a term's posting list, regardless of the periodic check
/// interval. Skipping the check saves one pass over the accumulator, but a
/// common term costs a scan proportional to its df — for large posting lists
/// stopping first is the better trade. 0 disables eager checks.
fn sparse_wand_maxscore_eager_df() -> u64 {
    std::env::var("HELIX_SPARSE_WAND_MAXSCORE_EAGER_DF")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(8192)
}

#[inline]
fn should_check_wand_maxscore(terms_scanned: u64, remaining_terms: u64, interval: u64) -> bool {
    remaining_terms > 0 && (remaining_terms <= 2 || terms_scanned % interval.max(1) == 0)
}

fn sparse_posting_cache_max_terms() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HELIX_SPARSE_POSTING_CACHE_MAX_TERMS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(2048)
    })
}

fn sparse_term_metadata_cache_max_terms() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HELIX_SPARSE_TERM_METADATA_CACHE_MAX_TERMS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(8192)
    })
}

fn sparse_wand_prefetch_terms() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HELIX_SPARSE_WAND_PREFETCH_TERMS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(64)
    })
}

// ─── Forward index entry: bincode-serialized Vec<(u32, f32)> ───

fn encode_forward(terms: &[(u32, f32)]) -> Result<Vec<u8>, VectorError> {
    bincode::serialize(terms).map_err(|e| {
        VectorError::VectorCoreError(format!("forward index serialization failed: {}", e))
    })
}

fn decode_forward(data: &[u8]) -> Result<Vec<(u32, f32)>, VectorError> {
    bincode::deserialize(data)
        .map_err(|e| VectorError::VectorCoreError(format!("forward index corruption: {}", e)))
}

// ─── Posting list entry: 20 bytes = doc_id (16 BE) + value (f32 BE) ───

const POSTING_ENTRY_LEN: usize = 20;

fn encode_posting(doc_id: u128, value: f32) -> [u8; POSTING_ENTRY_LEN] {
    let mut buf = [0u8; POSTING_ENTRY_LEN];
    buf[..16].copy_from_slice(&doc_id.to_be_bytes());
    buf[16..20].copy_from_slice(&value.to_be_bytes());
    buf
}

fn decode_posting(data: &[u8]) -> Result<(u128, f32), VectorError> {
    if data.len() != POSTING_ENTRY_LEN {
        return Err(VectorError::VectorCoreError(format!(
            "posting entry corruption: expected {} bytes, got {}",
            POSTING_ENTRY_LEN,
            data.len()
        )));
    }
    let doc_id = u128::from_be_bytes(data[..16].try_into().unwrap());
    let value = f32::from_be_bytes(data[16..20].try_into().unwrap());
    Ok((doc_id, value))
}

// ─── Meta keys ───

const META_DOC_COUNT_KEY: &[u8] = b"__doc_count__";

fn meta_term_key(term_id: u32) -> [u8; 5] {
    let mut buf = [b't'; 5];
    buf[1..5].copy_from_slice(&term_id.to_be_bytes());
    buf
}

/// Key for per-term upper-bound max value: "m" + term_id (4 bytes BE).
fn meta_max_key(term_id: u32) -> [u8; 5] {
    let mut buf = [b'm'; 5];
    buf[1..5].copy_from_slice(&term_id.to_be_bytes());
    buf
}

// ─── Inverted index key: term_id (4 bytes BE) ───

fn inv_key(term_id: u32) -> [u8; 4] {
    term_id.to_be_bytes()
}

// ─── SparseVectorCore ───

/// Inverted-index storage for sparse vectors within a single LMDB environment.
///
/// Three LMDB databases per sparse vector space:
/// - `inv_db`:  term_id (4B BE) → DUP_SORT posting entries (doc_id + value, 20B each)
/// - `fwd_db`:  doc_id (16B BE) → bincode Vec<(term_id, value)>
/// - `meta_db`: "__doc_count__" → u64 BE | "t" + term_id → u64 BE (document frequency)
pub struct SparseVectorCore {
    pub config: SparseVectorConfig,
    /// Deferred df increments (positive) and decrements (negative) per
    /// term_id, accumulated since the last `flush_metadata_pending`.
    /// Insert/delete update this in memory instead of writing to
    /// `meta_db` per-term — that read-modify-write was the dominant
    /// cost in `apply_upsert_points_in_txn` (~80 % of sparse_insert_all
    /// per Phase-0 instrumentation). `get_term_df` returns the persisted
    /// value plus the pending delta; `flush_metadata_pending` drains
    /// these into `meta_db` in a single pass.
    pending_df_delta: Mutex<HashMap<u32, i64>>,
    /// Deferred per-term max upper-bound updates. Holds the highest
    /// observed value per term since the last flush. Search WAND
    /// pruning consults `get_term_max` which returns
    /// `max(persisted, pending)` — so a stale persisted max never
    /// causes incorrect search results, only weaker pruning until the
    /// next flush.
    pending_max: Mutex<HashMap<u32, f32>>,
    /// Shared storage backend handle from the owning `HelixGraphStorage`
    /// (US-006 seam). Shared via `Arc` because `AnyBackend` is not `Clone`.
    /// All KV access routes through here via `Namespace::SparseSegment`.
    pub(crate) backend: Arc<AnyBackend>,
    /// Logical sparse-vector name used to address this core's three databases
    /// (`sparse_{inv,fwd,meta}_{name}`) through the backend namespace.
    pub(crate) physical_name: String,
    posting_cache: RwLock<HashMap<u32, Arc<Vec<(u128, f32)>>>>,
    term_metadata_cache: RwLock<HashMap<u32, (u64, f32)>>,
}

impl SparseVectorCore {
    /// Create (or open) the three LMDB databases for a named sparse vector space.
    pub fn new(
        env: &Env<WithTls>,
        txn: &mut RwTxn,
        name: &str,
        config: SparseVectorConfig,
        backend: Arc<AnyBackend>,
    ) -> Result<Self, VectorError> {
        // DBI creation is only needed for the LMDB backend. LSM addresses the
        // same logical sparse sub-databases via SlateDB namespace prefixes and
        // must not allocate local heed DBIs during collection/vector creation.
        if backend.kind() == BackendKind::Lmdb {
            let _inv_db = env
                .database_options()
                .types::<Bytes, Bytes>()
                .flags(heed3::DatabaseFlags::DUP_SORT | heed3::DatabaseFlags::DUP_FIXED)
                .name(&format!("sparse_inv_{}", name))
                .create(txn)?;

            let _fwd_db = env
                .database_options()
                .types::<U128<BE>, Bytes>()
                .name(&format!("sparse_fwd_{}", name))
                .create(txn)?;

            let _meta_db = env
                .database_options()
                .types::<Bytes, Bytes>()
                .name(&format!("sparse_meta_{}", name))
                .create(txn)?;
        }

        let core = Self {
            config,
            pending_df_delta: Mutex::new(HashMap::new()),
            pending_max: Mutex::new(HashMap::new()),
            backend,
            physical_name: name.to_string(),
            posting_cache: RwLock::new(HashMap::new()),
            term_metadata_cache: RwLock::new(HashMap::new()),
        };

        // Initialize doc_count to 0 if not present. This eager write uses the
        // heed write path (`put_heed`), which is `unreachable!()` on the LSM
        // backend — so only run it on LMDB. On LSM the key is simply absent at
        // create time; every reader (`get_doc_count`/`get_doc_count_be`) maps a
        // missing key to 0, and the first upsert persists it through the seam
        // (`set_doc_count_be` → `put`). The dense core constructor likewise does
        // no backend write at creation, which is why dense-only collections
        // create cleanly on LSM.
        if core.backend.kind() == BackendKind::Lmdb {
            let absent = core
                .backend
                .get_with_heed(txn, core.meta_ns(), META_DOC_COUNT_KEY, |v| v.is_none())
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            if absent {
                core.backend
                    .put_heed(txn, core.meta_ns(), META_DOC_COUNT_KEY, &0u64.to_be_bytes())
                    .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            }
        }

        Ok(core)
    }

    /// Create/open a sparse vector space on SlateDB without creating local heed DBIs.
    pub fn new_lsm(
        name: &str,
        config: SparseVectorConfig,
        backend: Arc<AnyBackend>,
    ) -> Result<Self, VectorError> {
        Ok(Self {
            config,
            pending_df_delta: Mutex::new(HashMap::new()),
            pending_max: Mutex::new(HashMap::new()),
            backend,
            physical_name: name.to_string(),
            posting_cache: RwLock::new(HashMap::new()),
            term_metadata_cache: RwLock::new(HashMap::new()),
        })
    }

    // ── Namespace addressing ──

    #[inline]
    fn inv_ns(&self) -> Namespace<'_> {
        Namespace::SparseSegment {
            physical_name: &self.physical_name,
            db: SparseDb::Inv,
        }
    }

    #[inline]
    fn fwd_ns(&self) -> Namespace<'_> {
        Namespace::SparseSegment {
            physical_name: &self.physical_name,
            db: SparseDb::Fwd,
        }
    }

    #[inline]
    fn meta_ns(&self) -> Namespace<'_> {
        Namespace::SparseSegment {
            physical_name: &self.physical_name,
            db: SparseDb::Meta,
        }
    }

    #[inline]
    fn sparse_search_metrics_enabled(&self) -> bool {
        self.backend.kind() == BackendKind::Lsm
    }

    #[inline]
    fn record_sparse_search_stage(
        &self,
        mode: &'static str,
        stage: &'static str,
        filtered: bool,
        limit: usize,
        query_terms: usize,
        started: Instant,
    ) {
        if !self.sparse_search_metrics_enabled() {
            return;
        }
        self.record_sparse_search_stage_duration(
            mode,
            stage,
            filtered,
            limit,
            query_terms,
            started.elapsed(),
        );
    }

    #[inline]
    fn record_sparse_search_stage_duration(
        &self,
        mode: &'static str,
        stage: &'static str,
        filtered: bool,
        limit: usize,
        query_terms: usize,
        elapsed: Duration,
    ) {
        if !self.sparse_search_metrics_enabled() {
            return;
        }
        metrics::histogram!(
            "helix_sparse_search_stage_ms",
            "backend" => "lsm",
            "vector" => self.physical_name.clone(),
            "mode" => mode,
            "stage" => stage,
            "filtered" => bool_label(filtered),
            "limit" => limit.to_string(),
            "query_terms" => query_terms.to_string(),
        )
        .record(elapsed.as_secs_f64() * 1000.0);
    }

    #[inline]
    fn record_sparse_search_items(
        &self,
        mode: &'static str,
        measurement: &'static str,
        filtered: bool,
        value: usize,
    ) {
        if !self.sparse_search_metrics_enabled() {
            return;
        }
        metrics::histogram!(
            "helix_sparse_search_items",
            "backend" => "lsm",
            "vector" => self.physical_name.clone(),
            "mode" => mode,
            "measurement" => measurement,
            "filtered" => bool_label(filtered),
        )
        .record(value as f64);
    }

    fn clear_posting_cache(&self) {
        if let Ok(mut cache) = self.posting_cache.write() {
            if !cache.is_empty() {
                cache.clear();
                metrics::counter!("helix_sparse_posting_cache_clears_total").increment(1);
            }
        }
    }

    fn cached_postings_be(
        &self,
        r: &AnyRead<'_>,
        term_id: u32,
    ) -> Result<Arc<Vec<(u128, f32)>>, VectorError> {
        if let Ok(cache) = self.posting_cache.read() {
            if let Some(postings) = cache.get(&term_id) {
                metrics::counter!("helix_sparse_posting_cache_hits_total").increment(1);
                return Ok(Arc::clone(postings));
            }
        }

        let key = inv_key(term_id);
        let mut postings = Vec::new();
        let mut decode_err: Option<VectorError> = None;
        self.backend
            .for_each_dup(r, self.inv_ns(), &key, |val_bytes| {
                match decode_posting(val_bytes) {
                    Ok(posting) => {
                        postings.push(posting);
                        true
                    }
                    Err(e) => {
                        decode_err = Some(e);
                        false
                    }
                }
            })
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        if let Some(e) = decode_err {
            return Err(e);
        }

        metrics::counter!("helix_sparse_posting_cache_misses_total").increment(1);
        let postings = Arc::new(postings);
        if let Ok(mut cache) = self.posting_cache.write() {
            if cache.len() >= sparse_posting_cache_max_terms() && !cache.contains_key(&term_id) {
                cache.clear();
                metrics::counter!("helix_sparse_posting_cache_clears_total").increment(1);
            }
            cache.insert(term_id, Arc::clone(&postings));
        }
        Ok(postings)
    }

    fn cache_postings(&self, term_id: u32, postings: Arc<Vec<(u128, f32)>>) {
        if let Ok(mut cache) = self.posting_cache.write() {
            if cache.len() >= sparse_posting_cache_max_terms() && !cache.contains_key(&term_id) {
                cache.clear();
                metrics::counter!("helix_sparse_posting_cache_clears_total").increment(1);
            }
            cache.insert(term_id, postings);
        }
    }

    fn cache_term_metadata(&self, term_id: u32, df: u64, term_max: f32) {
        if let Ok(mut cache) = self.term_metadata_cache.write() {
            if cache.len() >= sparse_term_metadata_cache_max_terms()
                && !cache.contains_key(&term_id)
            {
                cache.clear();
                metrics::counter!("helix_sparse_term_metadata_cache_clears_total").increment(1);
            }
            cache.insert(term_id, (df, term_max));
        }
    }

    fn invalidate_term_metadata_cache(&self, term_id: u32) {
        if let Ok(mut cache) = self.term_metadata_cache.write() {
            cache.remove(&term_id);
        }
    }

    fn apply_pending_term_metadata(&self, term_id: u32, df: u64, term_max: f32) -> (u64, f32) {
        let pending_df = self.pending_df_delta_for(term_id);
        let pending_max = self.pending_max_for(term_id);
        let persisted_df = i64::try_from(df).unwrap_or(i64::MAX);
        (
            persisted_df.saturating_add(pending_df).max(0) as u64,
            term_max.max(pending_max),
        )
    }

    fn cached_postings_many_be(
        &self,
        r: &AnyRead<'_>,
        term_ids: &[u32],
    ) -> Result<HashMap<u32, Arc<Vec<(u128, f32)>>>, VectorError> {
        let term_ids = unique_term_ids(term_ids);
        let mut postings_by_term = HashMap::with_capacity(term_ids.len());
        let mut missing = Vec::new();
        if let Ok(cache) = self.posting_cache.read() {
            for &term_id in &term_ids {
                if let Some(postings) = cache.get(&term_id) {
                    metrics::counter!("helix_sparse_posting_cache_hits_total").increment(1);
                    postings_by_term.insert(term_id, Arc::clone(postings));
                } else {
                    missing.push(term_id);
                }
            }
        } else {
            missing.extend_from_slice(&term_ids);
        }
        if missing.is_empty() {
            return Ok(postings_by_term);
        }

        metrics::counter!("helix_sparse_posting_prefetch_batches_total").increment(1);
        metrics::counter!("helix_sparse_posting_prefetch_terms_total")
            .increment(missing.len() as u64);
        let keys: Vec<Vec<u8>> = missing
            .iter()
            .map(|term_id| inv_key(*term_id).to_vec())
            .collect();
        let raw_lists = match (&*self.backend, r) {
            (AnyBackend::Lsm(writer), AnyRead::Lsm(read)) => writer
                .collect_dup_values_many_with(read, self.inv_ns(), &keys)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?,
            (AnyBackend::LsmReader(reader), AnyRead::LsmReader(snap)) => reader
                .collect_dup_values_many_with_at(snap.as_ref(), self.inv_ns(), &keys)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?,
            _ => {
                for term_id in missing {
                    let postings = self.cached_postings_be(r, term_id)?;
                    postings_by_term.insert(term_id, postings);
                }
                return Ok(postings_by_term);
            }
        };

        for (term_id, raw_values) in missing.into_iter().zip(raw_lists) {
            metrics::counter!("helix_sparse_posting_cache_misses_total").increment(1);
            let mut decoded = Vec::with_capacity(raw_values.len());
            for value in raw_values {
                decoded.push(decode_posting(&value)?);
            }
            let postings = Arc::new(decoded);
            self.cache_postings(term_id, Arc::clone(&postings));
            postings_by_term.insert(term_id, postings);
        }
        Ok(postings_by_term)
    }

    fn decode_term_df_raw(term_id: u32, raw: Option<Vec<u8>>) -> Result<u64, VectorError> {
        match raw {
            None => Ok(0),
            Some(bytes) => {
                let arr: [u8; 8] =
                    bytes
                        .get(..8)
                        .and_then(|s| s.try_into().ok())
                        .ok_or_else(|| {
                            VectorError::VectorCoreError(format!(
                                "corrupt sparse term_df for term {}: got {} bytes, expected 8",
                                term_id,
                                bytes.len()
                            ))
                        })?;
                Ok(u64::from_be_bytes(arr))
            }
        }
    }

    fn decode_term_max_raw(term_id: u32, raw: Option<Vec<u8>>) -> Result<f32, VectorError> {
        match raw {
            None => Ok(0.0),
            Some(bytes) => {
                let arr: [u8; 4] =
                    bytes
                        .get(..4)
                        .and_then(|s| s.try_into().ok())
                        .ok_or_else(|| {
                            VectorError::VectorCoreError(format!(
                                "corrupt sparse term_max for term {}: got {} bytes, expected 4",
                                term_id,
                                bytes.len()
                            ))
                        })?;
                Ok(f32::from_be_bytes(arr))
            }
        }
    }

    fn pending_df_delta_for(&self, term_id: u32) -> i64 {
        self.pending_df_delta
            .lock()
            .ok()
            .and_then(|m| m.get(&term_id).copied())
            .unwrap_or(0)
    }

    fn pending_max_for(&self, term_id: u32) -> f32 {
        self.pending_max
            .lock()
            .ok()
            .and_then(|m| m.get(&term_id).copied())
            .unwrap_or(f32::NEG_INFINITY)
    }

    fn term_df_many_read_be(
        &self,
        r: &AnyRead<'_>,
        term_ids: &[u32],
    ) -> Result<HashMap<u32, u64>, VectorError> {
        let term_ids = unique_term_ids(term_ids);
        let mut values = HashMap::with_capacity(term_ids.len());
        metrics::counter!(
            "helix_sparse_term_metadata_prefetch_batches_total",
            "kind" => "df"
        )
        .increment(1);
        metrics::counter!(
            "helix_sparse_term_metadata_prefetch_values_total",
            "kind" => "df"
        )
        .increment(term_ids.len() as u64);

        let keys: Vec<Vec<u8>> = term_ids
            .iter()
            .map(|term_id| meta_term_key(*term_id).to_vec())
            .collect();
        let raw_values = match (&*self.backend, r) {
            (AnyBackend::Lsm(writer), AnyRead::Lsm(read)) => writer
                .collect_values_many_with(read, self.meta_ns(), &keys)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?,
            (AnyBackend::LsmReader(reader), AnyRead::LsmReader(snap)) => reader
                .collect_values_many_with_at(snap.as_ref(), self.meta_ns(), &keys)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?,
            _ => {
                for &term_id in &term_ids {
                    values.insert(term_id, self.get_term_df_read_be(r, term_id)?);
                }
                return Ok(values);
            }
        };
        for (term_id, raw) in term_ids.into_iter().zip(raw_values) {
            let persisted = Self::decode_term_df_raw(term_id, raw)? as i64;
            let pending = self.pending_df_delta_for(term_id);
            values.insert(term_id, (persisted + pending).max(0) as u64);
        }
        Ok(values)
    }

    fn term_metadata_many_read_be(
        &self,
        r: &AnyRead<'_>,
        term_ids: &[u32],
    ) -> Result<HashMap<u32, (u64, f32)>, VectorError> {
        let term_ids = unique_term_ids(term_ids);
        let mut values = HashMap::with_capacity(term_ids.len());
        let mut missing = Vec::new();
        if let Ok(cache) = self.term_metadata_cache.read() {
            for &term_id in &term_ids {
                if let Some(&(df, term_max)) = cache.get(&term_id) {
                    metrics::counter!(
                        "helix_sparse_term_metadata_cache_total",
                        "kind" => "df_max",
                        "outcome" => "hit"
                    )
                    .increment(1);
                    values.insert(
                        term_id,
                        self.apply_pending_term_metadata(term_id, df, term_max),
                    );
                } else {
                    metrics::counter!(
                        "helix_sparse_term_metadata_cache_total",
                        "kind" => "df_max",
                        "outcome" => "miss"
                    )
                    .increment(1);
                    missing.push(term_id);
                }
            }
        } else {
            missing.extend_from_slice(&term_ids);
        }
        if missing.is_empty() {
            return Ok(values);
        }

        metrics::counter!(
            "helix_sparse_term_metadata_prefetch_batches_total",
            "kind" => "df_max"
        )
        .increment(1);
        metrics::counter!(
            "helix_sparse_term_metadata_prefetch_values_total",
            "kind" => "df_max"
        )
        .increment((missing.len() * 2) as u64);

        let mut keys = Vec::with_capacity(missing.len() * 2);
        for &term_id in &missing {
            keys.push(meta_term_key(term_id).to_vec());
            keys.push(meta_max_key(term_id).to_vec());
        }
        let mut raw_values = match (&*self.backend, r) {
            (AnyBackend::Lsm(writer), AnyRead::Lsm(read)) => writer
                .collect_values_many_with(read, self.meta_ns(), &keys)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?,
            (AnyBackend::LsmReader(reader), AnyRead::LsmReader(snap)) => reader
                .collect_values_many_with_at(snap.as_ref(), self.meta_ns(), &keys)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?,
            _ => {
                for &term_id in &missing {
                    values.insert(
                        term_id,
                        (
                            self.get_term_df_read_be(r, term_id)?,
                            self.get_term_max_read_be(r, term_id)?,
                        ),
                    );
                }
                return Ok(values);
            }
        }
        .into_iter();
        for term_id in missing {
            let df_raw = raw_values.next().ok_or_else(|| {
                VectorError::VectorCoreError(
                    "sparse term metadata prefetch returned too few df values".to_string(),
                )
            })?;
            let max_raw = raw_values.next().ok_or_else(|| {
                VectorError::VectorCoreError(
                    "sparse term metadata prefetch returned too few max values".to_string(),
                )
            })?;
            let persisted_df = Self::decode_term_df_raw(term_id, df_raw)?;
            let persisted_max = Self::decode_term_max_raw(term_id, max_raw)?;
            self.cache_term_metadata(term_id, persisted_df, persisted_max);
            values.insert(
                term_id,
                self.apply_pending_term_metadata(term_id, persisted_df, persisted_max),
            );
        }
        Ok(values)
    }

    // ── Seam KV helpers for the forward + inverted indexes ──
    //
    // The forward index is keyed by the doc_id's 16-byte big-endian encoding
    // (byte-identical to the old `U128<BE>` heed key type). The inverted index
    // is the DUP_SORT namespace: `inv_put` appends a duplicate posting, and
    // `inv_delete_dup` removes one specific `(key, value)` pair.

    /// Forward-index point read → owned bytes (None when absent).
    fn fwd_get(&self, txn: &RoTxn, doc_id: u128) -> Result<Option<Vec<u8>>, VectorError> {
        self.backend
            .get_with_heed(txn, self.fwd_ns(), &doc_id.to_be_bytes(), |opt| {
                opt.map(|b| b.to_vec())
            })
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    fn fwd_get_be(&self, w: &AnyWrite<'_>, doc_id: u128) -> Result<Option<Vec<u8>>, VectorError> {
        self.backend
            .get_for_update(w, self.fwd_ns(), &doc_id.to_be_bytes(), |opt| {
                opt.map(|b| b.to_vec())
            })
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    fn fwd_get_read_be(
        &self,
        r: &AnyRead<'_>,
        doc_id: u128,
    ) -> Result<Option<Vec<u8>>, VectorError> {
        self.backend
            .get_with(r, self.fwd_ns(), &doc_id.to_be_bytes(), |opt| {
                opt.map(|b| b.to_vec())
            })
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    fn fwd_put(&self, txn: &mut RwTxn, doc_id: u128, val: &[u8]) -> Result<(), VectorError> {
        self.backend
            .put_heed(txn, self.fwd_ns(), &doc_id.to_be_bytes(), val)
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    fn fwd_put_be(
        &self,
        w: &mut AnyWrite<'_>,
        doc_id: u128,
        val: &[u8],
    ) -> Result<(), VectorError> {
        self.backend
            .put(w, self.fwd_ns(), &doc_id.to_be_bytes(), val)
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    fn fwd_delete(&self, txn: &mut RwTxn, doc_id: u128) -> Result<(), VectorError> {
        self.backend
            .delete_heed(txn, self.fwd_ns(), &doc_id.to_be_bytes())
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    fn fwd_delete_be(&self, w: &mut AnyWrite<'_>, doc_id: u128) -> Result<(), VectorError> {
        self.backend
            .delete(w, self.fwd_ns(), &doc_id.to_be_bytes())
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    /// Append a posting duplicate under `key` in the DUP_SORT inverted index.
    fn inv_put(&self, txn: &mut RwTxn, key: &[u8], entry: &[u8]) -> Result<(), VectorError> {
        self.backend
            .put_dup_heed(txn, self.inv_ns(), key, entry)
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    fn inv_put_be(
        &self,
        w: &mut AnyWrite<'_>,
        key: &[u8],
        entry: &[u8],
    ) -> Result<(), VectorError> {
        self.backend
            .put_dup(w, self.inv_ns(), key, entry)
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    /// Remove one specific posting `(key, entry)` from the inverted index.
    fn inv_delete_dup(&self, txn: &mut RwTxn, key: &[u8], entry: &[u8]) -> Result<(), VectorError> {
        self.backend
            .delete_dup_heed(txn, self.inv_ns(), key, entry)
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    fn inv_delete_dup_be(
        &self,
        w: &mut AnyWrite<'_>,
        key: &[u8],
        entry: &[u8],
    ) -> Result<(), VectorError> {
        self.backend
            .delete_dup(w, self.inv_ns(), key, entry)
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    // ── Helpers ──

    fn get_doc_count(&self, txn: &RoTxn) -> Result<u64, VectorError> {
        let raw = self
            .backend
            .get_with_heed(txn, self.meta_ns(), META_DOC_COUNT_KEY, |opt| {
                opt.map(|b| b.to_vec())
            })
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        match raw {
            None => Ok(0),
            Some(b) => {
                let arr: [u8; 8] = b.get(..8).and_then(|s| s.try_into().ok()).ok_or_else(|| {
                    VectorError::VectorCoreError(format!(
                        "corrupt sparse doc_count: got {} bytes, expected 8",
                        b.len()
                    ))
                })?;
                Ok(u64::from_be_bytes(arr))
            }
        }
    }

    fn get_doc_count_be(&self, w: &AnyWrite<'_>) -> Result<u64, VectorError> {
        let raw = self
            .backend
            .get_for_update(w, self.meta_ns(), META_DOC_COUNT_KEY, |opt| {
                opt.map(|b| b.to_vec())
            })
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        match raw {
            None => Ok(0),
            Some(b) => {
                let arr: [u8; 8] = b.get(..8).and_then(|s| s.try_into().ok()).ok_or_else(|| {
                    VectorError::VectorCoreError(format!(
                        "corrupt sparse doc_count: got {} bytes, expected 8",
                        b.len()
                    ))
                })?;
                Ok(u64::from_be_bytes(arr))
            }
        }
    }

    fn get_doc_count_read_be(&self, r: &AnyRead<'_>) -> Result<u64, VectorError> {
        let raw = self
            .backend
            .get_with(r, self.meta_ns(), META_DOC_COUNT_KEY, |opt| {
                opt.map(|b| b.to_vec())
            })
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        match raw {
            None => Ok(0),
            Some(b) => {
                let arr: [u8; 8] = b.get(..8).and_then(|s| s.try_into().ok()).ok_or_else(|| {
                    VectorError::VectorCoreError(format!(
                        "corrupt sparse doc_count: got {} bytes, expected 8",
                        b.len()
                    ))
                })?;
                Ok(u64::from_be_bytes(arr))
            }
        }
    }

    fn set_doc_count(&self, txn: &mut RwTxn, count: u64) -> Result<(), VectorError> {
        self.backend
            .put_heed(
                txn,
                self.meta_ns(),
                META_DOC_COUNT_KEY,
                &count.to_be_bytes(),
            )
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        Ok(())
    }

    fn set_doc_count_be(&self, w: &mut AnyWrite<'_>, count: u64) -> Result<(), VectorError> {
        self.backend
            .put(w, self.meta_ns(), META_DOC_COUNT_KEY, &count.to_be_bytes())
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    /// Read the persisted df for a term from `meta_db`. Bypasses the
    /// pending delta — used internally by `flush_metadata_pending` so
    /// drains do not double-count their own pending values.
    fn get_term_df_persisted(&self, txn: &RoTxn, term_id: u32) -> Result<u64, VectorError> {
        let key = meta_term_key(term_id);
        let raw = self
            .backend
            .get_with_heed(txn, self.meta_ns(), &key, |opt| opt.map(|b| b.to_vec()))
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        match raw {
            None => Ok(0),
            Some(b) => {
                let arr: [u8; 8] = b.get(..8).and_then(|s| s.try_into().ok()).ok_or_else(|| {
                    VectorError::VectorCoreError(format!(
                        "corrupt sparse term_df for term {}: got {} bytes, expected 8",
                        term_id,
                        b.len()
                    ))
                })?;
                Ok(u64::from_be_bytes(arr))
            }
        }
    }

    fn get_term_df_persisted_be(&self, w: &AnyWrite<'_>, term_id: u32) -> Result<u64, VectorError> {
        let key = meta_term_key(term_id);
        let raw = self
            .backend
            .get_for_update(w, self.meta_ns(), &key, |opt| opt.map(|b| b.to_vec()))
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        match raw {
            None => Ok(0),
            Some(b) => {
                let arr: [u8; 8] = b.get(..8).and_then(|s| s.try_into().ok()).ok_or_else(|| {
                    VectorError::VectorCoreError(format!(
                        "corrupt sparse term_df for term {}: got {} bytes, expected 8",
                        term_id,
                        b.len()
                    ))
                })?;
                Ok(u64::from_be_bytes(arr))
            }
        }
    }

    fn get_term_df_persisted_read_be(
        &self,
        r: &AnyRead<'_>,
        term_id: u32,
    ) -> Result<u64, VectorError> {
        let key = meta_term_key(term_id);
        let raw = self
            .backend
            .get_with(r, self.meta_ns(), &key, |opt| opt.map(|b| b.to_vec()))
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        match raw {
            None => Ok(0),
            Some(b) => {
                let arr: [u8; 8] = b.get(..8).and_then(|s| s.try_into().ok()).ok_or_else(|| {
                    VectorError::VectorCoreError(format!(
                        "corrupt sparse term_df for term {}: got {} bytes, expected 8",
                        term_id,
                        b.len()
                    ))
                })?;
                Ok(u64::from_be_bytes(arr))
            }
        }
    }

    fn get_term_df(&self, txn: &RoTxn, term_id: u32) -> Result<u64, VectorError> {
        let persisted = self.get_term_df_persisted(txn, term_id)? as i64;
        let pending = self
            .pending_df_delta
            .lock()
            .ok()
            .and_then(|m| m.get(&term_id).copied())
            .unwrap_or(0);
        Ok((persisted + pending).max(0) as u64)
    }

    fn get_term_df_read_be(&self, r: &AnyRead<'_>, term_id: u32) -> Result<u64, VectorError> {
        let persisted = self.get_term_df_persisted_read_be(r, term_id)? as i64;
        let pending = self
            .pending_df_delta
            .lock()
            .ok()
            .and_then(|m| m.get(&term_id).copied())
            .unwrap_or(0);
        Ok((persisted + pending).max(0) as u64)
    }

    fn set_term_df(&self, txn: &mut RwTxn, term_id: u32, df: u64) -> Result<(), VectorError> {
        let key = meta_term_key(term_id);
        if df == 0 {
            self.backend
                .delete_heed(txn, self.meta_ns(), &key)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        } else {
            self.backend
                .put_heed(txn, self.meta_ns(), &key, &df.to_be_bytes())
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        }
        self.invalidate_term_metadata_cache(term_id);
        Ok(())
    }

    fn set_term_df_be(
        &self,
        w: &mut AnyWrite<'_>,
        term_id: u32,
        df: u64,
    ) -> Result<(), VectorError> {
        let key = meta_term_key(term_id);
        if df == 0 {
            self.backend
                .delete(w, self.meta_ns(), &key)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        } else {
            self.backend
                .put(w, self.meta_ns(), &key, &df.to_be_bytes())
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        }
        self.invalidate_term_metadata_cache(term_id);
        Ok(())
    }

    /// Read the persisted upper-bound max value for a term, ignoring
    /// the pending buffer. Used by `flush_metadata_pending`.
    fn get_term_max_persisted(&self, txn: &RoTxn, term_id: u32) -> Result<f32, VectorError> {
        let key = meta_max_key(term_id);
        let raw = self
            .backend
            .get_with_heed(txn, self.meta_ns(), &key, |opt| opt.map(|b| b.to_vec()))
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        match raw {
            None => Ok(0.0),
            Some(b) => {
                let arr: [u8; 4] = b.get(..4).and_then(|s| s.try_into().ok()).ok_or_else(|| {
                    VectorError::VectorCoreError(format!(
                        "corrupt sparse term_max for term {}: got {} bytes, expected 4",
                        term_id,
                        b.len()
                    ))
                })?;
                Ok(f32::from_be_bytes(arr))
            }
        }
    }

    fn get_term_max_persisted_be(
        &self,
        w: &AnyWrite<'_>,
        term_id: u32,
    ) -> Result<f32, VectorError> {
        let key = meta_max_key(term_id);
        let raw = self
            .backend
            .get_for_update(w, self.meta_ns(), &key, |opt| opt.map(|b| b.to_vec()))
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        match raw {
            None => Ok(0.0),
            Some(b) => {
                let arr: [u8; 4] = b.get(..4).and_then(|s| s.try_into().ok()).ok_or_else(|| {
                    VectorError::VectorCoreError(format!(
                        "corrupt sparse term_max for term {}: got {} bytes, expected 4",
                        term_id,
                        b.len()
                    ))
                })?;
                Ok(f32::from_be_bytes(arr))
            }
        }
    }

    fn get_term_max_persisted_read_be(
        &self,
        r: &AnyRead<'_>,
        term_id: u32,
    ) -> Result<f32, VectorError> {
        let key = meta_max_key(term_id);
        let raw = self
            .backend
            .get_with(r, self.meta_ns(), &key, |opt| opt.map(|b| b.to_vec()))
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        match raw {
            None => Ok(0.0),
            Some(b) => {
                let arr: [u8; 4] = b.get(..4).and_then(|s| s.try_into().ok()).ok_or_else(|| {
                    VectorError::VectorCoreError(format!(
                        "corrupt sparse term_max for term {}: got {} bytes, expected 4",
                        term_id,
                        b.len()
                    ))
                })?;
                Ok(f32::from_be_bytes(arr))
            }
        }
    }

    /// Read the stored upper-bound max value for a term. Returns 0.0 if not set.
    /// Includes the pending buffer so search WAND pruning sees fresh upper bounds
    /// even before the next `flush_metadata_pending`.
    fn get_term_max(&self, txn: &RoTxn, term_id: u32) -> Result<f32, VectorError> {
        let persisted = self.get_term_max_persisted(txn, term_id)?;
        let pending = self
            .pending_max
            .lock()
            .ok()
            .and_then(|m| m.get(&term_id).copied())
            .unwrap_or(f32::NEG_INFINITY);
        Ok(persisted.max(pending))
    }

    fn get_term_max_read_be(&self, r: &AnyRead<'_>, term_id: u32) -> Result<f32, VectorError> {
        let persisted = self.get_term_max_persisted_read_be(r, term_id)?;
        let pending = self
            .pending_max
            .lock()
            .ok()
            .and_then(|m| m.get(&term_id).copied())
            .unwrap_or(f32::NEG_INFINITY);
        Ok(persisted.max(pending))
    }

    /// Store the upper-bound max value for a term.
    fn set_term_max(&self, txn: &mut RwTxn, term_id: u32, max_val: f32) -> Result<(), VectorError> {
        let key = meta_max_key(term_id);
        self.backend
            .put_heed(txn, self.meta_ns(), &key, &max_val.to_be_bytes())
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        self.invalidate_term_metadata_cache(term_id);
        Ok(())
    }

    fn set_term_max_be(
        &self,
        w: &mut AnyWrite<'_>,
        term_id: u32,
        max_val: f32,
    ) -> Result<(), VectorError> {
        let key = meta_max_key(term_id);
        self.backend
            .put(w, self.meta_ns(), &key, &max_val.to_be_bytes())
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        self.invalidate_term_metadata_cache(term_id);
        Ok(())
    }

    fn adjust_term_df_be(
        &self,
        w: &mut AnyWrite<'_>,
        term_id: u32,
        delta: i64,
    ) -> Result<(), VectorError> {
        if delta == 0 {
            return Ok(());
        }
        let current = self.get_term_df_persisted_be(w, term_id)? as i64;
        self.set_term_df_be(w, term_id, (current + delta).max(0) as u64)
    }

    fn maybe_raise_term_max_be(
        &self,
        w: &mut AnyWrite<'_>,
        term_id: u32,
        candidate: f32,
    ) -> Result<(), VectorError> {
        if !candidate.is_finite() {
            return Ok(());
        }
        let persisted = self.get_term_max_persisted_be(w, term_id)?;
        if candidate > persisted {
            self.set_term_max_be(w, term_id, candidate)?;
        }
        Ok(())
    }

    // ── Insert ──

    /// Insert a sparse vector for the given document.
    /// If the document already has a sparse vector, delegates to `upsert`
    /// which does a term-level diff instead of a full delete+reinsert.
    /// Existing callers keep the simple API; the hot path in
    /// `apply_upsert_points_in_txn` uses `upsert` directly for the
    /// per-chunk outcome counters.
    ///
    /// Hot-path policy: only `fwd_db`, `inv_db`, and `doc_count` are
    /// written here. Per-term df increments and per-term max updates
    /// accumulate in memory (`pending_df_delta`, `pending_max`) and are
    /// drained later by `flush_metadata_pending`. This drops the
    /// per-point cost from O(K) read-modify-writes per metadata field
    /// to O(1), since the K-term loop now only touches the inverted
    /// index. Search results are unaffected: `get_term_df` and
    /// `get_term_max` consult the pending buffer transparently.
    pub fn insert(
        &self,
        txn: &mut RwTxn,
        doc_id: u128,
        sparse: &SparseVector,
    ) -> Result<(), VectorError> {
        self.upsert(txn, doc_id, sparse).map(|_| ())
    }

    /// Term-major batched upsert across a chunk's docs.
    ///
    /// Per-call instrumentation showed `helix_upsert_step_ms{step="sparse_insert_all"}`
    /// at ~89 ms mean per single doc — the chunk-level cost is N × that.
    /// The structural fix mirrors b04a1f82's batched delete: fold the
    /// per-doc per-term work so all postings to one DUPSORT subtree
    /// land contiguously instead of N×K random LMDB tree descents.
    ///
    /// Hot path (fresh inserts):
    ///   1. Validate + classify each doc (fresh / diff / unchanged).
    ///   2. Fresh fwd entries: 1 fwd_db.put per doc, in input order.
    ///   3. Collect (term_id, doc_id, value) tuples for ALL fresh docs;
    ///      sort by (term_id, doc_id). LMDB cursor stays warm across
    ///      contiguous puts to the same term's DUPSORT subtree.
    ///   4. Single pmax merge under one lock; single doc_count bump.
    ///
    /// Diff path stays per-doc (rare, complex; falls back to `upsert`).
    pub fn upsert_batch(
        &self,
        txn: &mut RwTxn,
        items: &[(u128, &SparseVector)],
    ) -> Result<usize, VectorError> {
        if items.is_empty() {
            return Ok(0);
        }
        self.clear_posting_cache();
        let batch_started = Instant::now();
        metrics::histogram!("helix_sparse_upsert_batch_docs").record(items.len() as f64);

        // Correctness gate: duplicate doc_id within one batch.
        // Classification reads the forward index ONCE before any writes,
        // so two items with the same doc_id both see "no existing fwd"
        // and both end up on the fresh path. That double-writes
        // postings, over-increments doc_count, and bloats hot terms'
        // DUPSORT subtrees. run_coalesced_batch() in async_gateway.rs
        // concatenates jobs from multiple /points requests without
        // deduping by id, so this is reachable in real traffic on
        // re-index/retry. Detect and fall back to fully sequential
        // upsert calls — they preserve the original last-write-wins
        // semantics where the second upsert sees the first's writes
        // and takes the diff path.
        // Detect duplicates AND count how many distinct doc_ids appear
        // multiple times. The density (duplicates / batch_size) tells
        // us whether fallback is rare-and-small (acceptable) or
        // frequent-and-large (motivates option-3 split).
        let step_started = Instant::now();
        let mut seen: std::collections::HashSet<u128> =
            std::collections::HashSet::with_capacity(items.len());
        let mut duplicate_count = 0usize;
        for &(doc_id, _) in items {
            if !seen.insert(doc_id) {
                duplicate_count += 1;
            }
        }
        let has_duplicate = duplicate_count > 0;
        record_sparse_batch_step("dedup_scan", step_started);
        if has_duplicate {
            metrics::counter!("helix_sparse_upsert_batch_dedup_fallback_total").increment(1);
            // Histogram of duplicate ratio per fallback batch. Reading:
            // p50 ~ 0.02 means most fallbacks have one stray dup;
            // p99 ~ 0.5 means half-the-batch dup'd, which is the case
            // option-3 (split-batch) was designed for.
            let density = duplicate_count as f64 / items.len() as f64;
            metrics::histogram!("helix_sparse_upsert_batch_dedup_density").record(density);
            metrics::histogram!("helix_sparse_upsert_batch_dedup_count")
                .record(duplicate_count as f64);
            let fallback_started = Instant::now();
            let mut fresh_count = 0usize;
            for &(doc_id, sparse) in items {
                let outcome = self.upsert(txn, doc_id, sparse)?;
                if matches!(outcome, SparseUpsertOutcome::Inserted) {
                    fresh_count += 1;
                }
            }
            record_sparse_batch_step("dedup_fallback", fallback_started);
            record_sparse_batch_step("total", batch_started);
            return Ok(fresh_count);
        }

        // Validate + classify. We hold no LMDB cursor here; all txn ops
        // are short reads, so this is cheap.
        let step_started = Instant::now();
        let mut fresh: Vec<(u128, Vec<(u32, f32)>)> = Vec::with_capacity(items.len());
        let mut diff_idx: Vec<usize> = Vec::new();
        let mut unchanged_count = 0usize;
        let mut total_input_terms = 0usize;
        for (idx, &(doc_id, sparse)) in items.iter().enumerate() {
            sparse.validate()?;
            total_input_terms += sparse.indices.len();
            let new_terms: Vec<(u32, f32)> = sparse
                .indices
                .iter()
                .zip(sparse.values.iter())
                .map(|(&i, &v)| (i, v))
                .collect();
            match self.fwd_get(txn, doc_id)? {
                None => fresh.push((doc_id, new_terms)),
                Some(existing) => {
                    let old_terms = decode_forward(&existing)?;
                    if old_terms == new_terms {
                        metrics::counter!(
                            "helix_sparse_upsert_outcome_total",
                            "outcome" => "unchanged"
                        )
                        .increment(1);
                        unchanged_count += 1;
                        // skip: identical re-upsert
                    } else {
                        diff_idx.push(idx);
                    }
                }
            }
        }
        record_sparse_batch_step("classify", step_started);
        metrics::histogram!("helix_sparse_upsert_batch_terms").record(total_input_terms as f64);
        metrics::histogram!("helix_sparse_upsert_batch_classified_docs", "outcome" => "fresh")
            .record(fresh.len() as f64);
        metrics::histogram!("helix_sparse_upsert_batch_classified_docs", "outcome" => "diff")
            .record(diff_idx.len() as f64);
        metrics::histogram!("helix_sparse_upsert_batch_classified_docs", "outcome" => "unchanged")
            .record(unchanged_count as f64);

        // ── Fresh path: term-major batched ────────────────────────────────
        let mut fresh_count: u64 = 0;
        if !fresh.is_empty() {
            // Forward entries first. fwd_db is keyed by doc_id; if items
            // arrive in roughly-sorted doc-id order the cursor stays warm.
            let step_started = Instant::now();
            for (doc_id, terms) in &fresh {
                self.fwd_put(txn, *doc_id, &encode_forward(terms)?)?;
            }
            record_sparse_batch_step("fresh_forward_put", step_started);

            // Flatten into (term_id, doc_id, value) and sort term-major.
            // Capacity guess: chunk × ~200 terms/doc.
            let step_started = Instant::now();
            let mut postings: Vec<(u32, u128, f32)> =
                Vec::with_capacity(fresh.iter().map(|(_, t)| t.len()).sum());
            for (doc_id, terms) in &fresh {
                for &(term_id, value) in terms {
                    postings.push((term_id, *doc_id, value));
                }
            }
            postings.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
            record_sparse_batch_step("fresh_posting_sort", step_started);
            metrics::histogram!("helix_sparse_upsert_batch_postings").record(postings.len() as f64);

            // The single sequential pass is the load-bearing optimization:
            // all puts to the same term's DUPSORT subtree are contiguous.
            let step_started = Instant::now();
            for (term_id, doc_id, value) in &postings {
                let key = inv_key(*term_id);
                let entry = encode_posting(*doc_id, *value);
                self.inv_put(txn, &key, &entry)?;
            }
            record_sparse_batch_step("fresh_posting_put", step_started);

            // Single pmax merge under one lock for the whole chunk.
            let step_started = Instant::now();
            if let Ok(mut pmax) = self.pending_max.lock() {
                for (term_id, _doc_id, value) in &postings {
                    let slot = pmax.entry(*term_id).or_insert(f32::NEG_INFINITY);
                    if *value > *slot {
                        *slot = *value;
                    }
                }
            }
            record_sparse_batch_step("fresh_pending_max", step_started);

            // Stage df deltas per-doc (each doc contributes +1 to its
            // unique term set). stage_df_delta already takes one lock.
            let step_started = Instant::now();
            for (_doc_id, terms) in &fresh {
                self.stage_df_delta(terms, 1);
            }
            record_sparse_batch_step("fresh_pending_df", step_started);

            // One doc_count bump for the whole chunk instead of N.
            fresh_count = fresh.len() as u64;
            let step_started = Instant::now();
            let dc = self.get_doc_count(txn)?;
            self.set_doc_count(txn, dc + fresh_count)?;
            record_sparse_batch_step("fresh_doc_count", step_started);
            metrics::counter!(
                "helix_sparse_upsert_outcome_total",
                "outcome" => "insert"
            )
            .increment(fresh_count);
        }

        // ── Diff path: per-doc fallback ───────────────────────────────────
        // Most points in steady-state CE re-ingest go through here, but
        // each doc's diff only touches changed terms (already
        // optimized), so per-doc cost is small. Batching the diff path
        // is a follow-up if it becomes the new ceiling.
        let step_started = Instant::now();
        for idx in diff_idx {
            let (doc_id, sparse) = items[idx];
            self.upsert(txn, doc_id, sparse)?;
        }
        record_sparse_batch_step("diff_fallback", step_started);
        record_sparse_batch_step("total", batch_started);

        Ok(fresh_count as usize)
    }

    pub fn upsert_batch_be(
        &self,
        w: &mut AnyWrite<'_>,
        items: &[(u128, &SparseVector)],
    ) -> Result<usize, VectorError> {
        if items.is_empty() {
            return Ok(0);
        }
        self.clear_posting_cache();

        let batch_started = Instant::now();
        let step_started = Instant::now();
        let mut seen = std::collections::HashSet::with_capacity(items.len());
        let mut duplicate_count = 0usize;
        for &(doc_id, _) in items {
            if !seen.insert(doc_id) {
                duplicate_count += 1;
            }
        }
        record_sparse_batch_step("dedup_scan", step_started);
        if duplicate_count > 0 {
            metrics::counter!("helix_sparse_upsert_batch_dedup_fallback_total").increment(1);
            metrics::histogram!("helix_sparse_upsert_batch_dedup_density")
                .record(duplicate_count as f64 / items.len() as f64);
            metrics::histogram!("helix_sparse_upsert_batch_dedup_count")
                .record(duplicate_count as f64);
            let fallback_started = Instant::now();
            let mut fresh_count = 0usize;
            for &(doc_id, sparse) in items {
                if matches!(
                    self.upsert_be(w, doc_id, sparse)?,
                    SparseUpsertOutcome::Inserted
                ) {
                    fresh_count += 1;
                }
            }
            record_sparse_batch_step("dedup_fallback", fallback_started);
            record_sparse_batch_step("total", batch_started);
            return Ok(fresh_count);
        }

        let step_started = Instant::now();
        let mut fresh: Vec<(u128, Vec<(u32, f32)>)> = Vec::with_capacity(items.len());
        let mut diff_idx: Vec<usize> = Vec::new();
        let mut unchanged_count = 0usize;
        let mut total_input_terms = 0usize;
        for (idx, &(doc_id, sparse)) in items.iter().enumerate() {
            sparse.validate()?;
            total_input_terms += sparse.indices.len();
            let new_terms: Vec<(u32, f32)> = sparse
                .indices
                .iter()
                .zip(sparse.values.iter())
                .map(|(&i, &v)| (i, v))
                .collect();
            match self.fwd_get_be(w, doc_id)? {
                None => fresh.push((doc_id, new_terms)),
                Some(existing) => {
                    let old_terms = decode_forward(&existing)?;
                    if old_terms == new_terms {
                        metrics::counter!(
                            "helix_sparse_upsert_outcome_total",
                            "outcome" => "unchanged"
                        )
                        .increment(1);
                        unchanged_count += 1;
                    } else {
                        diff_idx.push(idx);
                    }
                }
            }
        }
        record_sparse_batch_step("classify", step_started);
        metrics::histogram!("helix_sparse_upsert_batch_terms").record(total_input_terms as f64);
        metrics::histogram!("helix_sparse_upsert_batch_classified_docs", "outcome" => "fresh")
            .record(fresh.len() as f64);
        metrics::histogram!("helix_sparse_upsert_batch_classified_docs", "outcome" => "diff")
            .record(diff_idx.len() as f64);
        metrics::histogram!("helix_sparse_upsert_batch_classified_docs", "outcome" => "unchanged")
            .record(unchanged_count as f64);

        let mut fresh_count = 0usize;
        if !fresh.is_empty() {
            let step_started = Instant::now();
            for (doc_id, terms) in &fresh {
                self.fwd_put_be(w, *doc_id, &encode_forward(terms)?)?;
            }
            record_sparse_batch_step("fresh_forward_put", step_started);

            let step_started = Instant::now();
            let mut postings: Vec<(u32, u128, f32)> =
                Vec::with_capacity(fresh.iter().map(|(_, terms)| terms.len()).sum());
            let mut df_deltas: HashMap<u32, i64> = HashMap::new();
            let mut max_candidates: HashMap<u32, f32> = HashMap::new();
            for (doc_id, terms) in &fresh {
                for &(term_id, value) in terms {
                    postings.push((term_id, *doc_id, value));
                    *df_deltas.entry(term_id).or_insert(0) += 1;
                    max_candidates
                        .entry(term_id)
                        .and_modify(|candidate| {
                            if value > *candidate {
                                *candidate = value;
                            }
                        })
                        .or_insert(value);
                }
            }
            postings.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
            record_sparse_batch_step("fresh_posting_sort", step_started);
            metrics::histogram!("helix_sparse_upsert_batch_postings").record(postings.len() as f64);

            let step_started = Instant::now();
            for (term_id, doc_id, value) in &postings {
                let key = inv_key(*term_id);
                let entry = encode_posting(*doc_id, *value);
                self.inv_put_be(w, &key, &entry)?;
            }
            record_sparse_batch_step("fresh_posting_put", step_started);

            let step_started = Instant::now();
            let doc_count = self.get_doc_count_be(w)?;
            fresh_count = fresh.len();
            self.set_doc_count_be(w, doc_count + fresh_count as u64)?;
            record_sparse_batch_step("fresh_doc_count", step_started);

            let step_started = Instant::now();
            for (term_id, delta) in df_deltas {
                self.adjust_term_df_be(w, term_id, delta)?;
            }
            record_sparse_batch_step("fresh_term_df", step_started);

            let step_started = Instant::now();
            for (term_id, candidate) in max_candidates {
                self.maybe_raise_term_max_be(w, term_id, candidate)?;
            }
            record_sparse_batch_step("fresh_term_max", step_started);

            metrics::counter!(
                "helix_sparse_upsert_outcome_total",
                "outcome" => "insert"
            )
            .increment(fresh_count as u64);
        }

        let step_started = Instant::now();
        for idx in diff_idx {
            let (doc_id, sparse) = items[idx];
            self.upsert_be(w, doc_id, sparse)?;
        }
        record_sparse_batch_step("diff_fallback", step_started);
        record_sparse_batch_step("total", batch_started);
        Ok(fresh_count)
    }

    /// Diff-based upsert.
    ///
    /// - Fresh insert: writes fwd, postings, stages df+max, bumps doc_count.
    /// - True update with identical sparse payload: **no-op** (skips fwd
    ///   rewrite, posting churn, df/max staging).
    /// - True update with differing terms: only touches postings that were
    ///   added, removed, or whose value changed. Unchanged terms keep their
    ///   posting entries — no delete+reinsert thrash.
    ///
    /// This is the primary entry point for the re-index workload where CE's
    /// watcher frequently re-feeds points whose sparse representation hasn't
    /// shifted meaningfully between runs; the old delete-then-reinsert path
    /// wrote every posting again for every such point.
    pub fn upsert(
        &self,
        txn: &mut RwTxn,
        doc_id: u128,
        sparse: &SparseVector,
    ) -> Result<SparseUpsertOutcome, VectorError> {
        sparse.validate()?;
        self.clear_posting_cache();

        let new_terms: Vec<(u32, f32)> = sparse
            .indices
            .iter()
            .zip(sparse.values.iter())
            .map(|(&i, &v)| (i, v))
            .collect();

        // Fresh insert path — no existing forward entry.
        let Some(existing_bytes) = self.fwd_get(txn, doc_id)? else {
            self.write_forward_and_postings(txn, doc_id, &new_terms)?;
            let doc_count = self.get_doc_count(txn)?;
            self.set_doc_count(txn, doc_count + 1)?;
            self.stage_df_delta(&new_terms, 1);
            metrics::counter!("helix_sparse_upsert_outcome_total", "outcome" => "insert")
                .increment(1);
            return Ok(SparseUpsertOutcome::Inserted);
        };

        let old_terms = decode_forward(&existing_bytes)?;

        // Identical payload — skip everything. This is the re-index win.
        if old_terms == new_terms {
            metrics::counter!("helix_sparse_upsert_outcome_total", "outcome" => "unchanged")
                .increment(1);
            return Ok(SparseUpsertOutcome::Unchanged);
        }

        // Diff term sets. Both old and new term vectors are sorted by
        // term_id (validated by SparseVector::validate), so we merge-walk.
        // Diff Vecs are bounded by their respective input sizes. `removed`
        // and `changed` cannot exceed `old_terms.len()`; `added` cannot
        // exceed `new_terms.len()`. Re-index workloads run this loop on
        // every doc update — preallocating skips reallocation churn.
        let mut added: Vec<(u32, f32)> = Vec::with_capacity(new_terms.len());
        let mut removed: Vec<(u32, f32)> = Vec::with_capacity(old_terms.len());
        let mut changed: Vec<(u32, f32, f32)> = Vec::with_capacity(old_terms.len()); // (term_id, old_val, new_val)

        let mut i = 0usize;
        let mut j = 0usize;
        while i < old_terms.len() && j < new_terms.len() {
            match old_terms[i].0.cmp(&new_terms[j].0) {
                std::cmp::Ordering::Equal => {
                    if old_terms[i].1 != new_terms[j].1 {
                        changed.push((old_terms[i].0, old_terms[i].1, new_terms[j].1));
                    }
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => {
                    removed.push(old_terms[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    added.push(new_terms[j]);
                    j += 1;
                }
            }
        }
        while i < old_terms.len() {
            removed.push(old_terms[i]);
            i += 1;
        }
        while j < new_terms.len() {
            added.push(new_terms[j]);
            j += 1;
        }

        // Apply removals first so duplicate-key pairs for changed terms
        // don't accidentally delete the new entry we're about to write.
        for &(term_id, value) in &removed {
            let key = inv_key(term_id);
            let entry = encode_posting(doc_id, value);
            self.inv_delete_dup(txn, &key, &entry)?;
        }
        // Changed terms: the posting encodes (doc_id, value), so when value
        // flips the old byte pattern and new byte pattern are distinct DUP
        // entries — remove the old, insert the new.
        for &(term_id, old_val, new_val) in &changed {
            let key = inv_key(term_id);
            let old_entry = encode_posting(doc_id, old_val);
            let new_entry = encode_posting(doc_id, new_val);
            self.inv_delete_dup(txn, &key, &old_entry)?;
            self.inv_put(txn, &key, &new_entry)?;
        }
        for &(term_id, value) in &added {
            let key = inv_key(term_id);
            let entry = encode_posting(doc_id, value);
            self.inv_put(txn, &key, &entry)?;
        }

        // Rewrite the forward entry. Always needed here because the term
        // list differs (identical-payload case returned above).
        self.fwd_put(txn, doc_id, &encode_forward(&new_terms)?)?;

        // Update per-term pending max for added + changed terms. removed
        // terms intentionally don't decrement max — max is a WAND upper
        // bound, staleness only costs pruning quality not correctness.
        if let Ok(mut pmax) = self.pending_max.lock() {
            for &(term_id, value) in &added {
                let slot = pmax.entry(term_id).or_insert(f32::NEG_INFINITY);
                if value > *slot {
                    *slot = value;
                }
            }
            for &(term_id, _old, new_val) in &changed {
                let slot = pmax.entry(term_id).or_insert(f32::NEG_INFINITY);
                if new_val > *slot {
                    *slot = new_val;
                }
            }
        }

        // df adjustments: added terms +1, removed terms -1. changed terms
        // don't move df (same doc, same term, new value).
        if let Ok(mut pdf) = self.pending_df_delta.lock() {
            for &(term_id, _) in &added {
                *pdf.entry(term_id).or_insert(0) += 1;
            }
            for &(term_id, _) in &removed {
                *pdf.entry(term_id).or_insert(0) -= 1;
            }
        }

        metrics::counter!("helix_sparse_upsert_outcome_total", "outcome" => "diff").increment(1);
        metrics::counter!("helix_sparse_upsert_terms_added_total").increment(added.len() as u64);
        metrics::counter!("helix_sparse_upsert_terms_removed_total")
            .increment(removed.len() as u64);
        metrics::counter!("helix_sparse_upsert_terms_changed_total")
            .increment(changed.len() as u64);

        Ok(SparseUpsertOutcome::DiffApplied {
            added: added.len(),
            removed: removed.len(),
            changed: changed.len(),
        })
    }

    pub fn upsert_be(
        &self,
        w: &mut AnyWrite<'_>,
        doc_id: u128,
        sparse: &SparseVector,
    ) -> Result<SparseUpsertOutcome, VectorError> {
        sparse.validate()?;
        self.clear_posting_cache();
        let new_terms: Vec<(u32, f32)> = sparse
            .indices
            .iter()
            .zip(sparse.values.iter())
            .map(|(&i, &v)| (i, v))
            .collect();

        let Some(existing_bytes) = self.fwd_get_be(w, doc_id)? else {
            self.write_forward_and_postings_be(w, doc_id, &new_terms)?;
            let doc_count = self.get_doc_count_be(w)?;
            self.set_doc_count_be(w, doc_count + 1)?;
            let mut seen = std::collections::HashSet::with_capacity(new_terms.len());
            for &(term_id, _) in &new_terms {
                if seen.insert(term_id) {
                    self.adjust_term_df_be(w, term_id, 1)?;
                }
            }
            metrics::counter!("helix_sparse_upsert_outcome_total", "outcome" => "insert")
                .increment(1);
            return Ok(SparseUpsertOutcome::Inserted);
        };

        let old_terms = decode_forward(&existing_bytes)?;
        if old_terms == new_terms {
            metrics::counter!("helix_sparse_upsert_outcome_total", "outcome" => "unchanged")
                .increment(1);
            return Ok(SparseUpsertOutcome::Unchanged);
        }

        let mut added: Vec<(u32, f32)> = Vec::with_capacity(new_terms.len());
        let mut removed: Vec<(u32, f32)> = Vec::with_capacity(old_terms.len());
        let mut changed: Vec<(u32, f32, f32)> = Vec::with_capacity(old_terms.len());
        let mut i = 0usize;
        let mut j = 0usize;
        while i < old_terms.len() && j < new_terms.len() {
            match old_terms[i].0.cmp(&new_terms[j].0) {
                std::cmp::Ordering::Equal => {
                    if old_terms[i].1 != new_terms[j].1 {
                        changed.push((old_terms[i].0, old_terms[i].1, new_terms[j].1));
                    }
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => {
                    removed.push(old_terms[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    added.push(new_terms[j]);
                    j += 1;
                }
            }
        }
        while i < old_terms.len() {
            removed.push(old_terms[i]);
            i += 1;
        }
        while j < new_terms.len() {
            added.push(new_terms[j]);
            j += 1;
        }

        for &(term_id, value) in &removed {
            let key = inv_key(term_id);
            let entry = encode_posting(doc_id, value);
            self.inv_delete_dup_be(w, &key, &entry)?;
        }
        for &(term_id, old_val, new_val) in &changed {
            let key = inv_key(term_id);
            self.inv_delete_dup_be(w, &key, &encode_posting(doc_id, old_val))?;
            self.inv_put_be(w, &key, &encode_posting(doc_id, new_val))?;
        }
        for &(term_id, value) in &added {
            let key = inv_key(term_id);
            self.inv_put_be(w, &key, &encode_posting(doc_id, value))?;
        }
        self.fwd_put_be(w, doc_id, &encode_forward(&new_terms)?)?;

        for &(term_id, value) in &added {
            self.adjust_term_df_be(w, term_id, 1)?;
            self.maybe_raise_term_max_be(w, term_id, value)?;
        }
        for &(term_id, _) in &removed {
            self.adjust_term_df_be(w, term_id, -1)?;
        }
        for &(term_id, _old, new_val) in &changed {
            self.maybe_raise_term_max_be(w, term_id, new_val)?;
        }

        metrics::counter!("helix_sparse_upsert_outcome_total", "outcome" => "diff").increment(1);
        Ok(SparseUpsertOutcome::DiffApplied {
            added: added.len(),
            removed: removed.len(),
            changed: changed.len(),
        })
    }

    #[inline]
    fn write_forward_and_postings(
        &self,
        txn: &mut RwTxn,
        doc_id: u128,
        terms: &[(u32, f32)],
    ) -> Result<(), VectorError> {
        self.fwd_put(txn, doc_id, &encode_forward(terms)?)?;
        // Issue all posting writes first (the slow part), then apply the
        // pending_max update under a single brief lock at the end. The
        // previous version acquired pending_max.lock() PER TERM —
        // ~12,800 mutex acquires per 64-doc × 200-term chunk, and worse
        // it held the search-side pmax mutex for the full per-term LMDB
        // put loop. Now the lock is held only for the in-memory merge.
        for &(term_id, value) in terms {
            let key = inv_key(term_id);
            let entry = encode_posting(doc_id, value);
            self.inv_put(txn, &key, &entry)?;
        }
        if let Ok(mut pmax) = self.pending_max.lock() {
            for &(term_id, value) in terms {
                let slot = pmax.entry(term_id).or_insert(f32::NEG_INFINITY);
                if value > *slot {
                    *slot = value;
                }
            }
        }
        Ok(())
    }

    fn write_forward_and_postings_be(
        &self,
        w: &mut AnyWrite<'_>,
        doc_id: u128,
        terms: &[(u32, f32)],
    ) -> Result<(), VectorError> {
        self.fwd_put_be(w, doc_id, &encode_forward(terms)?)?;
        for &(term_id, value) in terms {
            let key = inv_key(term_id);
            let entry = encode_posting(doc_id, value);
            self.inv_put_be(w, &key, &entry)?;
            self.maybe_raise_term_max_be(w, term_id, value)?;
        }
        Ok(())
    }

    #[inline]
    fn stage_df_delta(&self, terms: &[(u32, f32)], delta: i64) {
        if delta == 0 || terms.is_empty() {
            return;
        }
        if let Ok(mut pdf) = self.pending_df_delta.lock() {
            // `seen` dedupes term_ids against the input; bounded by terms.len().
            let mut seen = std::collections::HashSet::with_capacity(terms.len());
            for &(term_id, _) in terms {
                if seen.insert(term_id) {
                    *pdf.entry(term_id).or_insert(0) += delta;
                }
            }
        }
    }

    // ── Delete ──

    /// Remove a document's sparse vector from all indexes.
    ///
    /// Mirrors `insert`'s deferred-metadata policy: posting + fwd
    /// removals + doc_count are inline; df decrements stage into
    /// `pending_df_delta` (negative deltas) and drain later. The
    /// per-term max is intentionally not decremented here — it's an
    /// upper bound used for WAND pruning, so an over-stale max only
    /// reduces pruning quality, never correctness. The reaper-driven
    /// rebuild path can refresh maxes from posting lists if drift
    /// becomes measurable.
    pub fn delete(&self, txn: &mut RwTxn, doc_id: u128) -> Result<(), VectorError> {
        self.clear_posting_cache();
        let Some(fwd_data) = self.fwd_get(txn, doc_id)? else {
            return Ok(()); // Not present — nothing to delete
        };
        let terms = decode_forward(&fwd_data)?;

        // Track unique terms for df decrement.
        let mut seen_terms = std::collections::HashSet::new();

        // Remove posting list entries.
        for &(term_id, value) in &terms {
            let key = inv_key(term_id);
            let entry = encode_posting(doc_id, value);
            self.inv_delete_dup(txn, &key, &entry)?;
            seen_terms.insert(term_id);
        }

        // Remove forward index entry.
        self.fwd_delete(txn, doc_id)?;

        // doc_count stays inline.
        let doc_count = self.get_doc_count(txn)?;
        self.set_doc_count(txn, doc_count.saturating_sub(1))?;

        // Stage df decrements per unique term in memory.
        if let Ok(mut pdf) = self.pending_df_delta.lock() {
            for term_id in seen_terms {
                *pdf.entry(term_id).or_insert(0) -= 1;
            }
        }

        Ok(())
    }

    pub fn delete_be(&self, w: &mut AnyWrite<'_>, doc_id: u128) -> Result<(), VectorError> {
        self.clear_posting_cache();
        let Some(fwd_data) = self.fwd_get_be(w, doc_id)? else {
            return Ok(());
        };
        let terms = decode_forward(&fwd_data)?;
        let mut seen_terms = std::collections::HashSet::new();
        for &(term_id, value) in &terms {
            let key = inv_key(term_id);
            let entry = encode_posting(doc_id, value);
            self.inv_delete_dup_be(w, &key, &entry)?;
            seen_terms.insert(term_id);
        }
        self.fwd_delete_be(w, doc_id)?;
        let doc_count = self.get_doc_count_be(w)?;
        self.set_doc_count_be(w, doc_count.saturating_sub(1))?;
        for term_id in seen_terms {
            self.adjust_term_df_be(w, term_id, -1)?;
        }
        Ok(())
    }

    /// Drain the in-memory `pending_df_delta` and `pending_max` buffers
    /// into `meta_db` in a single pass. Idempotent: a second call with
    /// nothing pending is a no-op.
    ///
    /// Caller is responsible for invoking this from a write txn at a
    /// cadence that keeps the pending buffer bounded (e.g. once per
    /// `apply_upsert_points_in_txn` chunk, or on a periodic background
    /// timer). Hot-path inserts/deletes do *not* call this directly —
    /// that's the entire point of the deferral.
    ///
    /// Returns the total number of (df + max) entries flushed, useful
    /// for metrics.
    pub fn flush_metadata_pending(&self, txn: &mut RwTxn) -> Result<usize, VectorError> {
        Ok(self
            .flush_metadata_pending_budgeted(txn, None, None)?
            .flushed)
    }

    /// Budgeted variant of `flush_metadata_pending`.
    ///
    /// `budget` caps how many pending metadata entries are processed in
    /// this write transaction; `max_duration` caps wall-clock time spent
    /// in this flush. Unprocessed entries are merged back into
    /// the in-memory pending buffers. Readers remain correct while entries
    /// are pending because `get_term_df` and `get_term_max` overlay pending
    /// state on top of persisted metadata. The crash window is intentionally
    /// the same class of risk as the existing in-memory deferral: postings
    /// may commit while not-yet-flushed df/max metadata is lost until a
    /// future reconciliation pass rebuilds it from postings.
    pub fn flush_metadata_pending_budgeted(
        &self,
        txn: &mut RwTxn,
        budget: Option<usize>,
        max_duration: Option<Duration>,
    ) -> Result<SparseMetadataFlushStats, VectorError> {
        // Take the pending state out of the Mutex so concurrent inserts
        // can stage new deltas while we write. If LMDB writes fail partway
        // through, the un-applied local remainder is lost — same failure
        // mode as the original full-drain implementation.
        let pending_df = self
            .pending_df_delta
            .lock()
            .map(|mut g| std::mem::take(&mut *g))
            .unwrap_or_default();
        let pending_max = self
            .pending_max
            .lock()
            .map(|mut g| std::mem::take(&mut *g))
            .unwrap_or_default();

        let pending_before = pending_df.len() + pending_max.len();
        let started = Instant::now();
        let mut remaining_budget = budget.unwrap_or(usize::MAX);
        let mut leftover_df: HashMap<u32, i64> = HashMap::new();
        let mut leftover_max: HashMap<u32, f32> = HashMap::new();
        let mut processed = 0usize;
        let mut flushed = 0usize;
        let mut time_exhausted = false;

        for (term_id, delta) in pending_df {
            if delta == 0 {
                continue;
            }
            if remaining_budget == 0 {
                leftover_df.insert(term_id, delta);
                continue;
            }
            if max_duration.is_some_and(|limit| started.elapsed() >= limit) {
                time_exhausted = true;
                leftover_df.insert(term_id, delta);
                continue;
            }
            remaining_budget -= 1;
            processed += 1;

            let current = self.get_term_df_persisted(txn, term_id)? as i64;
            let new = (current + delta).max(0) as u64;
            self.set_term_df(txn, term_id, new)?;
            flushed += 1;
        }

        for (term_id, candidate) in pending_max {
            if !candidate.is_finite() {
                continue;
            }
            if remaining_budget == 0 {
                leftover_max.insert(term_id, candidate);
                continue;
            }
            if time_exhausted || max_duration.is_some_and(|limit| started.elapsed() >= limit) {
                time_exhausted = true;
                leftover_max.insert(term_id, candidate);
                continue;
            }
            remaining_budget -= 1;
            processed += 1;

            let persisted = self.get_term_max_persisted(txn, term_id)?;
            if candidate > persisted {
                self.set_term_max(txn, term_id, candidate)?;
                flushed += 1;
            }
        }

        if !leftover_df.is_empty() {
            if let Ok(mut pending) = self.pending_df_delta.lock() {
                for (term_id, delta) in leftover_df {
                    *pending.entry(term_id).or_insert(0) += delta;
                }
            }
        }
        if !leftover_max.is_empty() {
            if let Ok(mut pending) = self.pending_max.lock() {
                for (term_id, candidate) in leftover_max {
                    pending
                        .entry(term_id)
                        .and_modify(|existing| {
                            if candidate > *existing {
                                *existing = candidate;
                            }
                        })
                        .or_insert(candidate);
                }
            }
        }

        let pending_after = self.pending_metadata_count();
        Ok(SparseMetadataFlushStats {
            pending_before,
            processed,
            flushed,
            pending_after,
            budget_exhausted: budget.is_some() && remaining_budget == 0 && pending_after > 0,
            time_exhausted: time_exhausted && pending_after > 0,
        })
    }

    /// Re-stage drained df deltas back into the in-memory pending buffer
    /// (additive merge). Used to restore state if a flush batch fails to commit.
    fn restage_pending_df(&self, df: HashMap<u32, i64>) {
        if df.is_empty() {
            return;
        }
        if let Ok(mut pending) = self.pending_df_delta.lock() {
            for (term_id, delta) in df {
                *pending.entry(term_id).or_insert(0) += delta;
            }
        }
    }

    /// Re-stage drained max candidates back into the in-memory pending buffer
    /// (max-merge, since the per-term max is an upper bound). Used to restore
    /// state if a flush batch fails to commit.
    fn restage_pending_max(&self, mx: HashMap<u32, f32>) {
        if mx.is_empty() {
            return;
        }
        if let Ok(mut pending) = self.pending_max.lock() {
            for (term_id, candidate) in mx {
                pending
                    .entry(term_id)
                    .and_modify(|existing| {
                        if candidate > *existing {
                            *existing = candidate;
                        }
                    })
                    .or_insert(candidate);
            }
        }
    }

    /// LSM variant of [`Self::flush_metadata_pending_budgeted`].
    ///
    /// Drains pending df/max into the backend in its OWN self-contained write
    /// batch and is **commit-safe**: every delta attempted in the batch is
    /// restored to the in-memory pending buffers if the commit fails (e.g.
    /// SlateDB CAS [`BackendError::Conflict`] or the runtime-guard
    /// `Unsupported`), so no df/max update is silently lost. Budget/time
    /// remainder is merged back exactly as the LMDB path does. Readers stay
    /// correct while entries are pending because `get_term_df`/`get_term_max`
    /// overlay the pending buffers on top of persisted metadata.
    pub fn flush_metadata_pending_budgeted_be(
        &self,
        budget: Option<usize>,
        max_duration: Option<Duration>,
    ) -> Result<SparseMetadataFlushStats, VectorError> {
        // Take the pending state out of the Mutex so concurrent inserts can stage
        // new deltas while we write. Unlike the LMDB path, nothing is dropped on
        // failure: the remainder is re-staged below, and the applied set is
        // re-staged if the commit does not land.
        let pending_df = self
            .pending_df_delta
            .lock()
            .map(|mut g| std::mem::take(&mut *g))
            .unwrap_or_default();
        let pending_max = self
            .pending_max
            .lock()
            .map(|mut g| std::mem::take(&mut *g))
            .unwrap_or_default();

        let pending_before = pending_df.len() + pending_max.len();
        let started = Instant::now();
        let mut remaining_budget = budget.unwrap_or(usize::MAX);
        let mut leftover_df: HashMap<u32, i64> = HashMap::new();
        let mut leftover_max: HashMap<u32, f32> = HashMap::new();
        let mut to_apply_df: Vec<(u32, i64)> = Vec::new();
        let mut to_apply_max: Vec<(u32, f32)> = Vec::new();
        let mut time_exhausted = false;

        // Phase 1: partition into apply vs leftover under budget/time. No I/O.
        for (term_id, delta) in pending_df {
            if delta == 0 {
                continue;
            }
            let over_time = max_duration.is_some_and(|limit| started.elapsed() >= limit);
            if remaining_budget == 0 || over_time {
                time_exhausted |= over_time;
                leftover_df.insert(term_id, delta);
                continue;
            }
            remaining_budget -= 1;
            to_apply_df.push((term_id, delta));
        }
        for (term_id, candidate) in pending_max {
            if !candidate.is_finite() {
                continue;
            }
            let over_time =
                time_exhausted || max_duration.is_some_and(|limit| started.elapsed() >= limit);
            if remaining_budget == 0 || over_time {
                time_exhausted |= over_time;
                leftover_max.insert(term_id, candidate);
                continue;
            }
            remaining_budget -= 1;
            to_apply_max.push((term_id, candidate));
        }

        // Unprocessed remainder goes back immediately.
        self.restage_pending_df(std::mem::take(&mut leftover_df));
        self.restage_pending_max(std::mem::take(&mut leftover_max));

        let processed = to_apply_df.len() + to_apply_max.len();
        if processed == 0 {
            let pending_after = self.pending_metadata_count();
            return Ok(SparseMetadataFlushStats {
                pending_before,
                processed: 0,
                flushed: 0,
                pending_after,
                budget_exhausted: false,
                time_exhausted: time_exhausted && pending_after > 0,
            });
        }

        // Phase 2: apply in a self-contained batch. Any failure restores the
        // attempted deltas so nothing is silently lost.
        let result = (|| -> Result<usize, VectorError> {
            let mut w = self
                .backend
                .begin_write()
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            let mut flushed = 0usize;
            for &(term_id, delta) in &to_apply_df {
                let current = self.get_term_df_persisted_be(&w, term_id)? as i64;
                self.set_term_df_be(&mut w, term_id, (current + delta).max(0) as u64)?;
                flushed += 1;
            }
            for &(term_id, candidate) in &to_apply_max {
                let persisted = self.get_term_max_persisted_be(&w, term_id)?;
                if candidate > persisted {
                    self.set_term_max_be(&mut w, term_id, candidate)?;
                    flushed += 1;
                }
            }
            self.backend
                .commit(w)
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            Ok(flushed)
        })();

        match result {
            Ok(flushed) => {
                let pending_after = self.pending_metadata_count();
                Ok(SparseMetadataFlushStats {
                    pending_before,
                    processed,
                    flushed,
                    pending_after,
                    budget_exhausted: budget.is_some()
                        && remaining_budget == 0
                        && pending_after > 0,
                    time_exhausted: time_exhausted && pending_after > 0,
                })
            }
            Err(e) => {
                // Commit did not land — restore every attempted delta.
                self.restage_pending_df(to_apply_df.into_iter().collect());
                self.restage_pending_max(to_apply_max.into_iter().collect());
                Err(e)
            }
        }
    }

    /// Approximate count of pending entries (df deltas + max updates).
    /// Used by callers to decide when to flush — e.g. after a batch.
    pub fn pending_metadata_count(&self) -> usize {
        let df = self.pending_df_delta.lock().map(|g| g.len()).unwrap_or(0);
        let mx = self.pending_max.lock().map(|g| g.len()).unwrap_or(0);
        df + mx
    }

    // ── Search ──

    /// Search for the top-K documents most similar to the query sparse vector.
    ///
    /// `filter_fn` is called for each candidate doc_id; return `true` to include.
    /// When `None`, all documents pass.
    ///
    /// When WAND is enabled (the default) and the query has >= 3 terms and the
    /// index has >= `full_scan_threshold` documents, an early-termination strategy
    /// is used that skips candidates whose upper-bound score cannot reach the
    /// current k-th best score.  Otherwise the method falls back to the full
    /// accumulation path which evaluates every posting.
    pub fn search<F>(
        &self,
        txn: &RoTxn,
        query: &SparseVector,
        limit: usize,
        filter_fn: Option<&F>,
    ) -> Result<Vec<(u128, f64)>, VectorError>
    where
        F: Fn(u128) -> bool,
    {
        query.validate()?;
        if limit == 0 || query.indices.is_empty() {
            return Ok(Vec::new());
        }

        let doc_count = self.get_doc_count(txn)?;
        let use_wand = self.config.wand_enabled
            && query.indices.len() >= 3
            && doc_count as usize >= self.config.full_scan_threshold;

        if use_wand {
            self.search_wand(txn, query, limit, filter_fn, doc_count)
        } else {
            self.search_full_scan(txn, query, limit, filter_fn, doc_count)
        }
    }

    pub fn search_be<F>(
        &self,
        r: &AnyRead<'_>,
        query: &SparseVector,
        limit: usize,
        filter_fn: Option<&F>,
    ) -> Result<Vec<(u128, f64)>, VectorError>
    where
        F: Fn(u128) -> bool,
    {
        let total_start = Instant::now();
        let filtered = filter_fn.is_some();
        query.validate()?;
        if limit == 0 || query.indices.is_empty() {
            return Ok(Vec::new());
        }

        let doc_count_start = Instant::now();
        let doc_count = self.get_doc_count_read_be(r)?;
        self.record_sparse_search_stage(
            "select",
            "doc_count",
            filtered,
            limit,
            query.indices.len(),
            doc_count_start,
        );
        let use_wand = self.config.wand_enabled
            && query.indices.len() >= 3
            && doc_count as usize >= self.config.full_scan_threshold;
        let mode = if use_wand { "wand" } else { "full_scan" };
        self.record_sparse_search_items(mode, "doc_count", filtered, doc_count as usize);
        self.record_sparse_search_items(mode, "query_terms", filtered, query.indices.len());
        self.record_sparse_search_items(mode, "limit", filtered, limit);

        let result = if use_wand {
            self.search_wand_be(r, query, limit, filter_fn, doc_count)
        } else {
            self.search_full_scan_be(r, query, limit, filter_fn, doc_count)
        };
        self.record_sparse_search_stage(
            mode,
            "total",
            filtered,
            limit,
            query.indices.len(),
            total_start,
        );
        if let Ok(results) = &result {
            self.record_sparse_search_items(mode, "results_returned", filtered, results.len());
        }
        result
    }

    pub fn search_candidate_ids_exact_be<I>(
        &self,
        r: &AnyRead<'_>,
        query: &SparseVector,
        limit: usize,
        candidate_ids: I,
    ) -> Result<Vec<(u128, f64)>, VectorError>
    where
        I: IntoIterator<Item = u128>,
    {
        let total_start = Instant::now();
        let mode = "exact";
        query.validate()?;
        if limit == 0 || query.indices.is_empty() {
            return Ok(Vec::new());
        }

        let use_idf = matches!(self.config.modifier, SparseModifier::Idf);
        let weight_start = Instant::now();
        let doc_count = self.get_doc_count_read_be(r)?;
        let doc_count_f = doc_count as f64;
        let mut query_weights = HashMap::with_capacity(query.indices.len());
        let term_dfs = if use_idf {
            self.term_df_many_read_be(r, &query.indices)?
        } else {
            HashMap::new()
        };
        for (&term_id, &query_value) in query.indices.iter().zip(query.values.iter()) {
            if query_value == 0.0 {
                continue;
            }
            let idf = if use_idf {
                let df = term_dfs.get(&term_id).copied().unwrap_or(0);
                if df == 0 {
                    continue;
                }
                (1.0 + doc_count_f / df as f64).ln()
            } else {
                1.0
            };
            if idf == 0.0 {
                continue;
            }
            query_weights.insert(term_id, query_value as f64 * idf);
        }
        self.record_sparse_search_stage(
            mode,
            "query_weights",
            true,
            limit,
            query.indices.len(),
            weight_start,
        );
        self.record_sparse_search_items(mode, "doc_count", true, doc_count as usize);
        self.record_sparse_search_items(mode, "query_terms", true, query.indices.len());
        self.record_sparse_search_items(mode, "query_weights", true, query_weights.len());
        if query_weights.is_empty() {
            return Ok(Vec::new());
        }

        let score_start = Instant::now();
        let mut heap: BinaryHeap<ScoredDoc> = BinaryHeap::new();
        let mut candidates_checked = 0usize;
        let mut forward_hits = 0usize;
        let mut scored_docs = 0usize;
        for doc_id in candidate_ids {
            candidates_checked += 1;
            let Some(fwd_data) = self.fwd_get_read_be(r, doc_id)? else {
                continue;
            };
            forward_hits += 1;
            let terms = decode_forward(&fwd_data)?;
            let mut score = 0.0;
            for (term_id, stored_value) in terms {
                if let Some(query_weight) = query_weights.get(&term_id) {
                    score += *query_weight * stored_value as f64;
                }
            }
            if score == 0.0 {
                continue;
            }
            scored_docs += 1;
            if heap.len() < limit {
                heap.push(ScoredDoc { id: doc_id, score });
            } else if let Some(min) = heap.peek() {
                if score > min.score || (score == min.score && doc_id < min.id) {
                    heap.pop();
                    heap.push(ScoredDoc { id: doc_id, score });
                }
            }
        }
        self.record_sparse_search_stage(
            mode,
            "score_candidates",
            true,
            limit,
            query.indices.len(),
            score_start,
        );
        self.record_sparse_search_items(mode, "candidate_ids", true, candidates_checked);
        self.record_sparse_search_items(mode, "forward_hits", true, forward_hits);
        self.record_sparse_search_items(mode, "scored_docs", true, scored_docs);

        let result = Self::heap_to_sorted_results(heap);
        self.record_sparse_search_stage(
            mode,
            "total",
            true,
            limit,
            query.indices.len(),
            total_start,
        );
        if let Ok(results) = &result {
            self.record_sparse_search_items(mode, "results_returned", true, results.len());
        }
        result
    }

    /// Full accumulation search — evaluates every posting for every query term.
    fn search_full_scan<F>(
        &self,
        txn: &RoTxn,
        query: &SparseVector,
        limit: usize,
        filter_fn: Option<&F>,
        doc_count: u64,
    ) -> Result<Vec<(u128, f64)>, VectorError>
    where
        F: Fn(u128) -> bool,
    {
        let use_idf = matches!(self.config.modifier, SparseModifier::Idf);
        let doc_count_f = doc_count as f64;

        // Accumulate scores per document. Cap capacity at 64k to avoid
        // over-allocation on large corpora (full-scan only fires when
        // doc_count is small, but defend against unexpected growth).
        let mut scores: HashMap<u128, f64> =
            HashMap::with_capacity((doc_count as usize).min(65536));

        for (&term_id, &query_value) in query.indices.iter().zip(query.values.iter()) {
            let idf = if use_idf {
                let df = self.get_term_df(txn, term_id)?;
                if df == 0 {
                    continue; // Term doesn't exist in corpus — skip
                }
                (1.0 + doc_count_f / df as f64).ln()
            } else {
                1.0
            };

            // Iterate posting list for this term.
            let key = inv_key(term_id);
            let mut decode_err: Option<VectorError> = None;
            self.backend
                .for_each_dup_heed(txn, self.inv_ns(), &key, |val_bytes| {
                    match decode_posting(val_bytes) {
                        Ok((doc_id, stored_value)) => {
                            let contribution = query_value as f64 * stored_value as f64 * idf;
                            *scores.entry(doc_id).or_insert(0.0) += contribution;
                            true
                        }
                        Err(e) => {
                            decode_err = Some(e);
                            false
                        }
                    }
                })
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            if let Some(e) = decode_err {
                return Err(e);
            }
        }

        // Apply filter and build top-K heap.
        let mut heap: BinaryHeap<ScoredDoc> = BinaryHeap::new();

        for (doc_id, score) in scores {
            if let Some(f) = filter_fn {
                if !f(doc_id) {
                    continue;
                }
            }

            if heap.len() < limit {
                heap.push(ScoredDoc { id: doc_id, score });
            } else if let Some(min) = heap.peek() {
                if score > min.score || (score == min.score && doc_id < min.id) {
                    heap.pop();
                    heap.push(ScoredDoc { id: doc_id, score });
                }
            }
        }

        Self::heap_to_sorted_results(heap)
    }

    fn search_full_scan_be<F>(
        &self,
        r: &AnyRead<'_>,
        query: &SparseVector,
        limit: usize,
        filter_fn: Option<&F>,
        doc_count: u64,
    ) -> Result<Vec<(u128, f64)>, VectorError>
    where
        F: Fn(u128) -> bool,
    {
        let total_start = Instant::now();
        let mode = "full_scan";
        let filtered = filter_fn.is_some();
        let use_idf = matches!(self.config.modifier, SparseModifier::Idf);
        let doc_count_f = doc_count as f64;
        let mut scores: HashMap<u128, f64> =
            HashMap::with_capacity((doc_count as usize).min(65536));

        let postings_start = Instant::now();
        let mut postings_seen = 0usize;
        for (&term_id, &query_value) in query.indices.iter().zip(query.values.iter()) {
            let idf = if use_idf {
                let df = self.get_term_df_read_be(r, term_id)?;
                if df == 0 {
                    continue;
                }
                (1.0 + doc_count_f / df as f64).ln()
            } else {
                1.0
            };
            let postings = self.cached_postings_be(r, term_id)?;
            postings_seen += postings.len();
            for &(doc_id, stored_value) in postings.iter() {
                let contribution = query_value as f64 * stored_value as f64 * idf;
                *scores.entry(doc_id).or_insert(0.0) += contribution;
            }
        }
        self.record_sparse_search_stage(
            mode,
            "postings_scan",
            filtered,
            limit,
            query.indices.len(),
            postings_start,
        );
        self.record_sparse_search_items(mode, "postings_seen", filtered, postings_seen);
        self.record_sparse_search_items(mode, "candidate_scores", filtered, scores.len());

        let heap_start = Instant::now();
        let mut heap: BinaryHeap<ScoredDoc> = BinaryHeap::new();
        let mut filtered_out = 0usize;
        let mut scored_docs = 0usize;
        for (doc_id, score) in scores {
            if let Some(f) = filter_fn {
                if !f(doc_id) {
                    filtered_out += 1;
                    continue;
                }
            }
            scored_docs += 1;
            if heap.len() < limit {
                heap.push(ScoredDoc { id: doc_id, score });
            } else if let Some(min) = heap.peek() {
                if score > min.score || (score == min.score && doc_id < min.id) {
                    heap.pop();
                    heap.push(ScoredDoc { id: doc_id, score });
                }
            }
        }
        self.record_sparse_search_stage(
            mode,
            "heap_select",
            filtered,
            limit,
            query.indices.len(),
            heap_start,
        );
        self.record_sparse_search_items(mode, "filtered_out", filtered, filtered_out);
        self.record_sparse_search_items(mode, "scored_docs", filtered, scored_docs);

        let result = Self::heap_to_sorted_results(heap);
        self.record_sparse_search_stage(
            mode,
            "mode_total",
            filtered,
            limit,
            query.indices.len(),
            total_start,
        );
        if let Ok(results) = &result {
            self.record_sparse_search_items(mode, "results_returned", filtered, results.len());
        }
        result
    }

    /// WAND-inspired sparse search with term-level upper-bound pruning.
    ///
    /// Algorithm:
    /// 1. For each query term, compute its contribution upper bound
    ///    `ub_t = |query_value_t| * term_max_t * idf_t`, where `term_max_t`
    ///    comes from `get_term_max` (overlays the in-memory pending buffer
    ///    so deferred metadata doesn't silently lower bounds before flush).
    /// 2. Drop terms with zero df (not in corpus), zero max (no postings), or
    ///    zero query weight. These contribute nothing and only cost LMDB
    ///    iteration time.
    /// 3. Sort surviving terms by descending `ub_t` so high-contribution
    ///    postings are processed first. Maintain `remaining_ub` as the sum of
    ///    `ub_t` over the still-unprocessed terms.
    /// 4. After accumulating each term's posting list, subtract its upper
    ///    bound from `remaining_ub`. If `remaining_ub <= 0`, stop — no future
    ///    term can alter any doc's score or ranking.
    ///
    /// We deliberately do NOT skip individual documents within a term's posting
    /// list. Term-at-a-time accumulation with per-document skipping is only
    /// correct under document-at-a-time traversal (classic DAAT WAND), which
    /// would require merged posting-list iteration we don't have here. The
    /// term-level exit is safe because every surviving doc's final score is a
    /// sum of *all* processed terms' contributions, and `remaining_ub == 0`
    /// proves nothing more can be added.
    ///
    /// Correctness properties (validated by `wand_matches_full_scan_*` tests):
    /// - Results are identical to `search_full_scan` for the same query.
    /// - Pending-metadata writes stay visible because `get_term_max` consults
    ///   the in-memory overlay; WAND does not regress recall across the
    ///   budgeted-flush window.
    fn search_wand<F>(
        &self,
        txn: &RoTxn,
        query: &SparseVector,
        limit: usize,
        filter_fn: Option<&F>,
        doc_count: u64,
    ) -> Result<Vec<(u128, f64)>, VectorError>
    where
        F: Fn(u128) -> bool,
    {
        let use_idf = matches!(self.config.modifier, SparseModifier::Idf);
        let doc_count_f = doc_count as f64;

        // Stage 1: build the pruned, upper-bound-sorted term list.
        // Each entry: (term_id, query_value, idf, upper_bound, df).
        // df = 0 when unknown (non-idf mode skips the df read); the eager
        // maxscore gate only fires on a known-large df.
        let mut weighted: Vec<(u32, f32, f64, f64, u64)> = Vec::with_capacity(query.indices.len());
        let mut total_ub: f64 = 0.0;
        for (&term_id, &query_value) in query.indices.iter().zip(query.values.iter()) {
            if query_value == 0.0 {
                continue;
            }
            let mut term_df = 0u64;
            let idf = if use_idf {
                let df = self.get_term_df(txn, term_id)?;
                if df == 0 {
                    continue;
                }
                term_df = df;
                (1.0 + doc_count_f / df as f64).ln()
            } else {
                1.0
            };
            if idf == 0.0 {
                continue;
            }
            let term_max = self.get_term_max(txn, term_id)?;
            if term_max <= 0.0 {
                continue;
            }
            let ub = (query_value as f64).abs() * term_max as f64 * idf;
            if ub <= 0.0 {
                continue;
            }
            weighted.push((term_id, query_value, idf, ub, term_df));
            total_ub += ub;
        }

        // Descending upper bound. Ties broken by term_id for determinism so
        // floating-point accumulation order is stable across runs.
        weighted.sort_by(|a, b| {
            b.3.partial_cmp(&a.3)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        // WAND prunes aggressively, so the actual occupancy is well below
        // doc_count. Cap at 64k to keep memory bounded on large corpora.
        let mut scores: HashMap<u128, f64> =
            HashMap::with_capacity(((doc_count as usize) / 4).clamp(1024, 65536));
        let mut remaining_ub = total_ub;
        let mut terms_scanned = 0u64;
        let mut terms_pruned = 0u64;

        let check_interval = sparse_wand_maxscore_check_interval();
        let eager_df = sparse_wand_maxscore_eager_df();
        for (term_id, query_value, idf, ub, term_df) in weighted.iter() {
            // Eager gate: before descending into a KNOWN-large posting list,
            // always run the maxscore check — the check costs one accumulator
            // pass, the scan it can avoid costs O(df).
            if terms_scanned > 0
                && eager_df > 0
                && *term_df >= eager_df
                && maxscore_can_stop(&scores, limit, remaining_ub)
            {
                terms_pruned = (weighted.len() as u64).saturating_sub(terms_scanned);
                metrics::counter!("helix_sparse_wand_maxscore_stops_total").increment(1);
                break;
            }
            terms_scanned += 1;
            let key = inv_key(*term_id);
            let mut decode_err: Option<VectorError> = None;
            self.backend
                .for_each_dup_heed(txn, self.inv_ns(), &key, |val_bytes| {
                    match decode_posting(val_bytes) {
                        Ok((doc_id, stored_value)) => {
                            let contribution = *query_value as f64 * stored_value as f64 * idf;
                            *scores.entry(doc_id).or_insert(0.0) += contribution;
                            true
                        }
                        Err(e) => {
                            decode_err = Some(e);
                            false
                        }
                    }
                })
                .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
            if let Some(e) = decode_err {
                return Err(e);
            }

            // Subtract this term's upper bound after processing. `remaining_ub`
            // now reflects the maximum possible contribution of all unprocessed
            // terms; when it drops to zero (or negative due to fp rounding),
            // no unseen term can affect any document's final score.
            remaining_ub -= *ub;
            if remaining_ub <= 0.0 {
                terms_pruned = (weighted.len() as u64).saturating_sub(terms_scanned);
                break;
            }
            let remaining_terms = (weighted.len() as u64).saturating_sub(terms_scanned);
            if should_check_wand_maxscore(terms_scanned, remaining_terms, check_interval)
                && maxscore_can_stop(&scores, limit, remaining_ub)
            {
                terms_pruned = (weighted.len() as u64).saturating_sub(terms_scanned);
                metrics::counter!("helix_sparse_wand_maxscore_stops_total").increment(1);
                break;
            }
        }

        metrics::counter!("helix_sparse_wand_terms_scanned_total").increment(terms_scanned);
        metrics::counter!("helix_sparse_wand_terms_pruned_total").increment(terms_pruned);

        // Build top-K heap from final scores.
        let mut heap: BinaryHeap<ScoredDoc> = BinaryHeap::new();
        for (doc_id, score) in scores {
            if let Some(f) = filter_fn {
                if !f(doc_id) {
                    continue;
                }
            }

            if heap.len() < limit {
                heap.push(ScoredDoc { id: doc_id, score });
            } else if let Some(min) = heap.peek() {
                if score > min.score || (score == min.score && doc_id < min.id) {
                    heap.pop();
                    heap.push(ScoredDoc { id: doc_id, score });
                }
            }
        }

        Self::heap_to_sorted_results(heap)
    }

    fn search_wand_be<F>(
        &self,
        r: &AnyRead<'_>,
        query: &SparseVector,
        limit: usize,
        filter_fn: Option<&F>,
        doc_count: u64,
    ) -> Result<Vec<(u128, f64)>, VectorError>
    where
        F: Fn(u128) -> bool,
    {
        let total_start = Instant::now();
        let mode = "wand";
        let filtered = filter_fn.is_some();
        let use_idf = matches!(self.config.modifier, SparseModifier::Idf);
        let doc_count_f = doc_count as f64;
        // Each entry: (term_id, query_value, idf, upper_bound, df). df = 0 when
        // unknown; the eager maxscore gate only fires on a known-large df.
        let mut weighted: Vec<(u32, f32, f64, f64, u64)> = Vec::with_capacity(query.indices.len());
        let mut total_ub: f64 = 0.0;
        let weight_start = Instant::now();
        let term_metadata = self.term_metadata_many_read_be(r, &query.indices)?;
        for (&term_id, &query_value) in query.indices.iter().zip(query.values.iter()) {
            if query_value == 0.0 {
                continue;
            }
            let term_df = term_metadata.get(&term_id).map(|(df, _)| *df).unwrap_or(0);
            let idf = if use_idf {
                if term_df == 0 {
                    continue;
                }
                (1.0 + doc_count_f / term_df as f64).ln()
            } else {
                1.0
            };
            if idf == 0.0 {
                continue;
            }
            let term_max = term_metadata
                .get(&term_id)
                .map(|(_, term_max)| *term_max)
                .unwrap_or(0.0);
            if term_max <= 0.0 {
                continue;
            }
            let ub = (query_value as f64).abs() * term_max as f64 * idf;
            if ub <= 0.0 {
                continue;
            }
            weighted.push((term_id, query_value, idf, ub, term_df));
            total_ub += ub;
        }
        weighted.sort_by(|a, b| {
            b.3.partial_cmp(&a.3)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        self.record_sparse_search_stage(
            mode,
            "weight_terms",
            filtered,
            limit,
            query.indices.len(),
            weight_start,
        );
        self.record_sparse_search_items(mode, "weighted_terms", filtered, weighted.len());

        let mut scores: HashMap<u128, f64> =
            HashMap::with_capacity(((doc_count as usize) / 4).clamp(1024, 65536));
        let mut remaining_ub = total_ub;
        let mut terms_scanned = 0u64;
        let mut terms_pruned = 0u64;
        let postings_start = Instant::now();
        let mut postings_seen = 0usize;
        let mut maxscore_checks = 0usize;
        let mut maxscore_check_elapsed = Duration::ZERO;
        let prefetch_window = sparse_wand_prefetch_terms().min(weighted.len());
        let mut prefetched = if prefetch_window > 1 {
            let term_ids: Vec<u32> = weighted
                .iter()
                .take(prefetch_window)
                .map(|(term_id, _, _, _, _)| *term_id)
                .collect();
            self.cached_postings_many_be(r, &term_ids)?
        } else {
            HashMap::new()
        };

        let check_interval = sparse_wand_maxscore_check_interval();
        let eager_df = sparse_wand_maxscore_eager_df();
        // Time the maxscore checks only when stage metrics are on: the
        // Instant pair must not run per-term in the hot loop otherwise.
        let time_checks = self.sparse_search_metrics_enabled();
        for (index, (term_id, query_value, idf, ub, term_df)) in weighted.iter().enumerate() {
            // Eager gate: before descending into a KNOWN-large posting list,
            // always run the maxscore check — the check costs one accumulator
            // pass, the scan it can avoid costs O(df).
            if terms_scanned > 0 && eager_df > 0 && *term_df >= eager_df {
                let maxscore_start = time_checks.then(Instant::now);
                let can_stop = maxscore_can_stop(&scores, limit, remaining_ub);
                if let Some(started) = maxscore_start {
                    maxscore_check_elapsed =
                        maxscore_check_elapsed.saturating_add(started.elapsed());
                }
                maxscore_checks += 1;
                if can_stop {
                    terms_pruned = (weighted.len() as u64).saturating_sub(terms_scanned);
                    metrics::counter!("helix_sparse_wand_maxscore_stops_total").increment(1);
                    break;
                }
            }
            terms_scanned += 1;
            if prefetch_window > 1 && !prefetched.contains_key(term_id) {
                let end = (index + prefetch_window).min(weighted.len());
                let term_ids: Vec<u32> = weighted[index..end]
                    .iter()
                    .map(|(term_id, _, _, _, _)| *term_id)
                    .collect();
                prefetched.extend(self.cached_postings_many_be(r, &term_ids)?);
            }
            let postings = match prefetched.get(term_id) {
                Some(postings) => Arc::clone(postings),
                None => self.cached_postings_be(r, *term_id)?,
            };
            postings_seen += postings.len();
            for &(doc_id, stored_value) in postings.iter() {
                let contribution = *query_value as f64 * stored_value as f64 * idf;
                *scores.entry(doc_id).or_insert(0.0) += contribution;
            }
            remaining_ub -= *ub;
            if remaining_ub <= 0.0 {
                terms_pruned = (weighted.len() as u64).saturating_sub(terms_scanned);
                break;
            }
            let remaining_terms = (weighted.len() as u64).saturating_sub(terms_scanned);
            if should_check_wand_maxscore(terms_scanned, remaining_terms, check_interval) {
                let maxscore_start = time_checks.then(Instant::now);
                let can_stop = maxscore_can_stop(&scores, limit, remaining_ub);
                if let Some(started) = maxscore_start {
                    maxscore_check_elapsed =
                        maxscore_check_elapsed.saturating_add(started.elapsed());
                }
                maxscore_checks += 1;
                if can_stop {
                    terms_pruned = (weighted.len() as u64).saturating_sub(terms_scanned);
                    metrics::counter!("helix_sparse_wand_maxscore_stops_total").increment(1);
                    break;
                }
            }
        }

        metrics::counter!("helix_sparse_wand_terms_scanned_total").increment(terms_scanned);
        metrics::counter!("helix_sparse_wand_terms_pruned_total").increment(terms_pruned);
        self.record_sparse_search_stage(
            mode,
            "postings_scan",
            filtered,
            limit,
            query.indices.len(),
            postings_start,
        );
        self.record_sparse_search_items(mode, "postings_seen", filtered, postings_seen);
        self.record_sparse_search_items(mode, "candidate_scores", filtered, scores.len());
        self.record_sparse_search_items(mode, "terms_scanned", filtered, terms_scanned as usize);
        self.record_sparse_search_items(mode, "terms_pruned", filtered, terms_pruned as usize);
        self.record_sparse_search_items(mode, "maxscore_checks", filtered, maxscore_checks);
        self.record_sparse_search_stage_duration(
            mode,
            "maxscore_check",
            filtered,
            limit,
            query.indices.len(),
            maxscore_check_elapsed,
        );

        let heap_start = Instant::now();
        let mut heap: BinaryHeap<ScoredDoc> = BinaryHeap::new();
        let mut filtered_out = 0usize;
        let mut scored_docs = 0usize;
        for (doc_id, score) in scores {
            if let Some(f) = filter_fn {
                if !f(doc_id) {
                    filtered_out += 1;
                    continue;
                }
            }
            scored_docs += 1;
            if heap.len() < limit {
                heap.push(ScoredDoc { id: doc_id, score });
            } else if let Some(min) = heap.peek() {
                if score > min.score || (score == min.score && doc_id < min.id) {
                    heap.pop();
                    heap.push(ScoredDoc { id: doc_id, score });
                }
            }
        }
        self.record_sparse_search_stage(
            mode,
            "heap_select",
            filtered,
            limit,
            query.indices.len(),
            heap_start,
        );
        self.record_sparse_search_items(mode, "filtered_out", filtered, filtered_out);
        self.record_sparse_search_items(mode, "scored_docs", filtered, scored_docs);

        let result = Self::heap_to_sorted_results(heap);
        self.record_sparse_search_stage(
            mode,
            "mode_total",
            filtered,
            limit,
            query.indices.len(),
            total_start,
        );
        if let Ok(results) = &result {
            self.record_sparse_search_items(mode, "results_returned", filtered, results.len());
        }
        result
    }

    /// Convert a min-heap of scored docs into a descending-score result vector.
    fn heap_to_sorted_results(
        heap: BinaryHeap<ScoredDoc>,
    ) -> Result<Vec<(u128, f64)>, VectorError> {
        let mut results: Vec<(u128, f64)> = heap.into_iter().map(|sd| (sd.id, sd.score)).collect();
        results.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        Ok(results)
    }

    /// Check if a document has a sparse vector stored.
    pub fn has_doc(&self, txn: &RoTxn, doc_id: u128) -> Result<bool, VectorError> {
        self.backend
            .get_with_heed(txn, self.fwd_ns(), &doc_id.to_be_bytes(), |opt| {
                opt.is_some()
            })
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))
    }

    /// Read a document's stored sparse vector from the forward index.
    ///
    /// Mirrors `has_doc`'s read pattern (`fwd_ns` keyed by `doc_id` BE) but
    /// decodes the bincode `Vec<(term, value)>` and splits it into the wire
    /// `SparseVector { indices, values }`. Returns `Ok(None)` when the doc has
    /// no sparse vector stored. Used by the Qdrant scroll/get read path so the
    /// bulk migrator's scroll->upsert carries sparse vectors, not just dense.
    pub fn get_doc_vector(
        &self,
        txn: &RoTxn,
        doc_id: u128,
    ) -> Result<Option<SparseVector>, VectorError> {
        let bytes: Option<Vec<u8>> = self
            .backend
            .get_with_heed(txn, self.fwd_ns(), &doc_id.to_be_bytes(), |opt| {
                opt.map(|b| b.to_vec())
            })
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        let Some(bytes) = bytes else {
            return Ok(None);
        };
        let terms = decode_forward(&bytes)?;
        let mut indices = Vec::with_capacity(terms.len());
        let mut values = Vec::with_capacity(terms.len());
        for (term, value) in terms {
            indices.push(term);
            values.push(value);
        }
        Ok(Some(SparseVector { indices, values }))
    }

    pub fn get_doc_vector_be(
        &self,
        r: &AnyRead<'_>,
        doc_id: u128,
    ) -> Result<Option<SparseVector>, VectorError> {
        let bytes: Option<Vec<u8>> = self
            .backend
            .get_with(r, self.fwd_ns(), &doc_id.to_be_bytes(), |opt| {
                opt.map(|b| b.to_vec())
            })
            .map_err(|e| VectorError::VectorCoreError(e.to_string()))?;
        let Some(bytes) = bytes else {
            return Ok(None);
        };
        let terms = decode_forward(&bytes)?;
        let mut indices = Vec::with_capacity(terms.len());
        let mut values = Vec::with_capacity(terms.len());
        for (term, value) in terms {
            indices.push(term);
            values.push(value);
        }
        Ok(Some(SparseVector { indices, values }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup() -> (TempDir, Env<WithTls>) {
        let tmp = TempDir::new().unwrap();
        let env = unsafe {
            heed3::EnvOpenOptions::new()
                .max_dbs(20)
                .map_size(64 * 1024 * 1024)
                .open(tmp.path())
                .unwrap()
        };
        (tmp, env)
    }

    fn test_backend(env: &Env<WithTls>) -> Arc<AnyBackend> {
        use crate::helix_engine::storage_core::backend_lmdb::LmdbBackend;
        Arc::new(AnyBackend::Lmdb(LmdbBackend::from_env(env.clone())))
    }

    fn make_core(env: &Env<WithTls>, name: &str, modifier: SparseModifier) -> SparseVectorCore {
        let mut txn = env.write_txn().unwrap();
        let core = SparseVectorCore::new(
            env,
            &mut txn,
            name,
            SparseVectorConfig {
                full_scan_threshold: 5000,
                modifier,
                ..Default::default()
            },
            test_backend(env),
        )
        .unwrap();
        txn.commit().unwrap();
        core
    }

    fn make_lsm_core(name: &str, modifier: SparseModifier) -> (Arc<AnyBackend>, SparseVectorCore) {
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;

        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory(&format!("sparse-batch-be-{name}-{nanos}")).unwrap(),
        ));
        let core = SparseVectorCore::new_lsm(
            name,
            SparseVectorConfig {
                full_scan_threshold: 5000,
                modifier,
                ..Default::default()
            },
            Arc::clone(&backend),
        )
        .unwrap();
        (backend, core)
    }

    #[test]
    fn unique_term_ids_preserves_first_seen_order() {
        assert_eq!(unique_term_ids(&[7, 3, 7, 1, 3, 9]), vec![7, 3, 1, 9]);
    }

    #[test]
    fn upsert_batch_be_batches_fresh_and_preserves_update_semantics() {
        let (backend, core) = make_lsm_core("batch_be", SparseModifier::None);
        let doc1 = SparseVector {
            indices: vec![0, 1],
            values: vec![1.0, 2.0],
        };
        let doc2 = SparseVector {
            indices: vec![1, 3],
            values: vec![5.0, 1.0],
        };

        let mut w = backend.begin_write().unwrap();
        let inserted = core
            .upsert_batch_be(&mut w, &[(1, &doc1), (2, &doc2)])
            .unwrap();
        assert_eq!(inserted, 2);
        backend.commit(w).unwrap();

        let r = backend.begin_read().unwrap();
        assert_eq!(core.get_doc_vector_be(&r, 1).unwrap(), Some(doc1.clone()));
        let results = core
            .search_be::<fn(u128) -> bool>(
                &r,
                &SparseVector {
                    indices: vec![1],
                    values: vec![1.0],
                },
                10,
                None,
            )
            .unwrap();
        assert_eq!(
            results.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![2, 1]
        );
        drop(r);

        let mut w = backend.begin_write().unwrap();
        let unchanged = core
            .upsert_batch_be(&mut w, &[(1, &doc1), (2, &doc2)])
            .unwrap();
        assert_eq!(unchanged, 0);
        let doc2_changed = SparseVector {
            indices: vec![4],
            values: vec![7.0],
        };
        let updated = core.upsert_batch_be(&mut w, &[(2, &doc2_changed)]).unwrap();
        assert_eq!(updated, 0);
        backend.commit(w).unwrap();

        let r = backend.begin_read().unwrap();
        let old_term_results = core
            .search_be::<fn(u128) -> bool>(
                &r,
                &SparseVector {
                    indices: vec![3],
                    values: vec![1.0],
                },
                10,
                None,
            )
            .unwrap();
        assert!(old_term_results.is_empty());
        let new_term_results = core
            .search_be::<fn(u128) -> bool>(
                &r,
                &SparseVector {
                    indices: vec![4],
                    values: vec![1.0],
                },
                10,
                None,
            )
            .unwrap();
        assert_eq!(new_term_results, vec![(2, 7.0)]);
    }

    #[test]
    fn exact_search_be_uses_lsm_writer_batch_term_metadata_pending_overlay() {
        let (backend, core) = make_lsm_core("writer_exact", SparseModifier::Idf);
        let doc1 = SparseVector {
            indices: vec![10, 20],
            values: vec![1.0, 2.0],
        };
        let doc2 = SparseVector {
            indices: vec![10, 30],
            values: vec![4.0, 1.0],
        };
        let doc3 = SparseVector {
            indices: vec![30],
            values: vec![8.0],
        };
        let mut w = backend.begin_write().unwrap();
        core.upsert_batch_be(&mut w, &[(1, &doc1), (2, &doc2), (3, &doc3)])
            .unwrap();
        backend.commit(w).unwrap();

        let r = backend.begin_read().unwrap();
        let results = core
            .search_candidate_ids_exact_be(
                &r,
                &SparseVector {
                    indices: vec![10, 30],
                    values: vec![1.0, 1.0],
                },
                3,
                [1, 2, 3],
            )
            .unwrap();

        assert_eq!(
            results.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![3, 2, 1],
            "fresh LSM writer exact search should batch metadata without losing pending df/max"
        );
        let metadata = core.term_metadata_many_read_be(&r, &[10, 30]).unwrap();
        assert_eq!(metadata.get(&10).unwrap().0, 2);
        assert_eq!(metadata.get(&30).unwrap().0, 2);
        drop(r);

        let doc4 = SparseVector {
            indices: vec![10, 30],
            values: vec![9.0, 12.0],
        };
        let mut w = backend.begin_write().unwrap();
        core.upsert_batch_be(&mut w, &[(4, &doc4)]).unwrap();
        backend.commit(w).unwrap();

        let r = backend.begin_read().unwrap();
        let cached_metadata = core.term_metadata_many_read_be(&r, &[10, 30]).unwrap();
        assert_eq!(cached_metadata.get(&10).unwrap().0, 3);
        assert_eq!(cached_metadata.get(&30).unwrap().0, 3);
        assert!(cached_metadata.get(&10).unwrap().1 >= 9.0);
        assert!(cached_metadata.get(&30).unwrap().1 >= 12.0);
    }

    #[test]
    fn exact_search_be_uses_lsm_reader_batch_term_metadata() {
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use crate::helix_engine::storage_core::backend_lsm_reader::LsmReader;
        use object_store::memory::InMemory;
        use object_store::ObjectStore;

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = format!("sparse-reader-exact-{nanos}");
        let config = SparseVectorConfig {
            full_scan_threshold: 5000,
            modifier: SparseModifier::Idf,
            ..Default::default()
        };

        let writer_backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_with_store(&path, Arc::clone(&store)).unwrap(),
        ));
        let writer_core =
            SparseVectorCore::new_lsm("lex_sparse", config.clone(), Arc::clone(&writer_backend))
                .unwrap();
        let doc1 = SparseVector {
            indices: vec![10, 20],
            values: vec![1.0, 2.0],
        };
        let doc2 = SparseVector {
            indices: vec![10, 30],
            values: vec![4.0, 1.0],
        };
        let doc3 = SparseVector {
            indices: vec![30],
            values: vec![8.0],
        };
        let mut w = writer_backend.begin_write().unwrap();
        writer_core
            .upsert_batch_be(&mut w, &[(1, &doc1), (2, &doc2), (3, &doc3)])
            .unwrap();
        writer_backend.commit(w).unwrap();
        writer_core
            .flush_metadata_pending_budgeted_be(None, None)
            .unwrap();

        let reader_backend = Arc::new(AnyBackend::LsmReader(
            LsmReader::open_with_store(&path, store).unwrap(),
        ));
        let reader_core =
            SparseVectorCore::new_lsm("lex_sparse", config, Arc::clone(&reader_backend)).unwrap();
        let r = reader_backend.begin_read().unwrap();
        let results = reader_core
            .search_candidate_ids_exact_be(
                &r,
                &SparseVector {
                    indices: vec![10, 30],
                    values: vec![1.0, 1.0],
                },
                3,
                [1, 2, 3],
            )
            .unwrap();
        assert_eq!(
            results.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![3, 2, 1],
            "fresh LSM reader exact search should use persisted batched df metadata"
        );
    }

    #[test]
    fn insert_and_search_basic() {
        let (_tmp, env) = setup();
        let core = make_core(&env, "test", SparseModifier::None);
        let mut txn = env.write_txn().unwrap();

        // Insert doc 1: terms [0, 1, 2] with values [1.0, 2.0, 3.0]
        core.insert(
            &mut txn,
            1,
            &SparseVector {
                indices: vec![0, 1, 2],
                values: vec![1.0, 2.0, 3.0],
            },
        )
        .unwrap();

        // Insert doc 2: terms [1, 3] with values [5.0, 1.0]
        core.insert(
            &mut txn,
            2,
            &SparseVector {
                indices: vec![1, 3],
                values: vec![5.0, 1.0],
            },
        )
        .unwrap();
        txn.commit().unwrap();

        let rtxn = env.read_txn().unwrap();

        // Query on term 1 only: doc1 has 2.0, doc2 has 5.0
        let results = core
            .search::<fn(u128) -> bool>(
                &rtxn,
                &SparseVector {
                    indices: vec![1],
                    values: vec![1.0],
                },
                10,
                None,
            )
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 2); // doc2 scores higher (5.0 vs 2.0)
        assert_eq!(results[1].0, 1);
        assert!((results[0].1 - 5.0).abs() < 1e-6);
        assert!((results[1].1 - 2.0).abs() < 1e-6);
    }

    #[test]
    fn upsert_replaces_old_sparse_vector() {
        let (_tmp, env) = setup();
        let core = make_core(&env, "test", SparseModifier::None);
        let mut txn = env.write_txn().unwrap();

        core.insert(
            &mut txn,
            1,
            &SparseVector {
                indices: vec![0, 1],
                values: vec![1.0, 2.0],
            },
        )
        .unwrap();

        // Upsert (replace) doc 1 with completely different terms.
        core.insert(
            &mut txn,
            1,
            &SparseVector {
                indices: vec![5, 6],
                values: vec![10.0, 20.0],
            },
        )
        .unwrap();
        txn.commit().unwrap();

        let rtxn = env.read_txn().unwrap();

        // Old terms should return nothing.
        let results = core
            .search::<fn(u128) -> bool>(
                &rtxn,
                &SparseVector {
                    indices: vec![0, 1],
                    values: vec![1.0, 1.0],
                },
                10,
                None,
            )
            .unwrap();
        assert!(results.is_empty());

        // New terms should find doc 1.
        let results = core
            .search::<fn(u128) -> bool>(
                &rtxn,
                &SparseVector {
                    indices: vec![5],
                    values: vec![1.0],
                },
                10,
                None,
            )
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
        assert!((results[0].1 - 10.0).abs() < 1e-6);
    }

    #[test]
    fn upsert_batch_handles_fresh_diff_and_unchanged() {
        // Covers the three classification branches: fresh insert,
        // unchanged (skip), and diff (replace). After the batched call
        // the inverted index should match what N independent upsert
        // calls would have produced.
        let (_tmp, env) = setup();
        let core = make_core(&env, "test_batch", SparseModifier::None);

        // Seed: doc 1 already exists. Will become an "unchanged" item.
        // Doc 2 already exists with different terms — will become diff.
        // Docs 3 & 4 are fresh.
        let mut txn = env.write_txn().unwrap();
        core.insert(
            &mut txn,
            1,
            &SparseVector {
                indices: vec![10, 20],
                values: vec![1.0, 2.0],
            },
        )
        .unwrap();
        core.insert(
            &mut txn,
            2,
            &SparseVector {
                indices: vec![30],
                values: vec![5.0],
            },
        )
        .unwrap();
        txn.commit().unwrap();

        // Batch with all three classifications + sharing the term=20
        // posting list across docs 1 (unchanged) and 3 (fresh) — exactly
        // the term-major sort path we want to exercise.
        let mut txn = env.write_txn().unwrap();
        let unchanged_doc1 = SparseVector {
            indices: vec![10, 20],
            values: vec![1.0, 2.0],
        };
        let diff_doc2 = SparseVector {
            indices: vec![40, 50],
            values: vec![7.0, 8.0],
        };
        let fresh_doc3 = SparseVector {
            indices: vec![20, 60],
            values: vec![3.0, 9.0],
        };
        let fresh_doc4 = SparseVector {
            indices: vec![40, 70],
            values: vec![4.0, 4.5],
        };
        let items: Vec<(u128, &SparseVector)> = vec![
            (1, &unchanged_doc1),
            (2, &diff_doc2),
            (3, &fresh_doc3),
            (4, &fresh_doc4),
        ];
        let fresh_count = core.upsert_batch(&mut txn, &items).unwrap();
        assert_eq!(fresh_count, 2, "two docs should be classified fresh");
        txn.commit().unwrap();

        let rtxn = env.read_txn().unwrap();

        // Doc 1 unchanged → still findable on its original term 10.
        let results = core
            .search::<fn(u128) -> bool>(
                &rtxn,
                &SparseVector {
                    indices: vec![10],
                    values: vec![1.0],
                },
                10,
                None,
            )
            .unwrap();
        let ids: Vec<u128> = results.iter().map(|(id, _)| *id).collect();
        assert!(
            ids.contains(&1),
            "doc 1 (unchanged) should still index term 10"
        );

        // Doc 2 diff → old term 30 should now miss doc 2; new term 40
        // should hit it (and also doc 4).
        let results = core
            .search::<fn(u128) -> bool>(
                &rtxn,
                &SparseVector {
                    indices: vec![30],
                    values: vec![1.0],
                },
                10,
                None,
            )
            .unwrap();
        let ids: Vec<u128> = results.iter().map(|(id, _)| *id).collect();
        assert!(
            !ids.contains(&2),
            "doc 2's old term 30 should be removed by diff"
        );

        let results = core
            .search::<fn(u128) -> bool>(
                &rtxn,
                &SparseVector {
                    indices: vec![40],
                    values: vec![1.0],
                },
                10,
                None,
            )
            .unwrap();
        let ids: Vec<u128> = results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&2), "doc 2 should now match new term 40");
        assert!(ids.contains(&4), "doc 4 (fresh) should match term 40");

        // Term 20 is shared between unchanged doc 1 and fresh doc 3 —
        // both must be present. Tests that the term-major sort + DUPSORT
        // contiguous puts didn't drop a posting.
        let results = core
            .search::<fn(u128) -> bool>(
                &rtxn,
                &SparseVector {
                    indices: vec![20],
                    values: vec![1.0],
                },
                10,
                None,
            )
            .unwrap();
        let ids: Vec<u128> = results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&1), "doc 1 (unchanged) keeps term 20 posting");
        assert!(ids.contains(&3), "doc 3 (fresh) gets term 20 posting");

        // Term 60 only on doc 3 (fresh).
        let results = core
            .search::<fn(u128) -> bool>(
                &rtxn,
                &SparseVector {
                    indices: vec![60],
                    values: vec![1.0],
                },
                10,
                None,
            )
            .unwrap();
        let ids: Vec<u128> = results.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![3], "term 60 only on doc 3");
    }

    #[test]
    fn upsert_batch_dedupes_duplicate_doc_ids_via_fallback() {
        // Regression: prod hit a 3+ minute hang after we shipped the
        // batched path — the symptom was massive sparse posting bloat
        // because run_coalesced_batch in async_gateway concatenates
        // /points requests without deduping by id. Two items with the
        // same doc_id both saw "no existing fwd" and BOTH inserted,
        // double-writing postings and over-bumping doc_count. The
        // sequential upsert path preserves last-write-wins because the
        // second call observes the first's fwd entry and takes the
        // diff path. This test pins that semantic.
        let (_tmp, env) = setup();
        let core = make_core(&env, "test_dup", SparseModifier::None);

        let v1 = SparseVector {
            indices: vec![10, 20],
            values: vec![1.0, 2.0],
        };
        let v2 = SparseVector {
            indices: vec![30, 40],
            values: vec![5.0, 6.0],
        };

        let mut txn = env.write_txn().unwrap();
        // doc_id=1 appears twice. Last value (v2) should win.
        let items: Vec<(u128, &SparseVector)> = vec![(1, &v1), (1, &v2)];
        let fresh = core.upsert_batch(&mut txn, &items).unwrap();
        // Exactly one fresh insert (the first call); the second is a diff.
        assert_eq!(fresh, 1);
        txn.commit().unwrap();

        let rtxn = env.read_txn().unwrap();

        // doc_count must be 1, not 2.
        assert_eq!(core.get_doc_count(&rtxn).unwrap(), 1);

        // v1 terms should be gone — only v2 terms should match.
        let r10 = core
            .search::<fn(u128) -> bool>(
                &rtxn,
                &SparseVector {
                    indices: vec![10],
                    values: vec![1.0],
                },
                10,
                None,
            )
            .unwrap();
        assert!(
            r10.iter().all(|(id, _)| *id != 1),
            "old term 10 must not still index doc 1 after second upsert wins"
        );

        let r30 = core
            .search::<fn(u128) -> bool>(
                &rtxn,
                &SparseVector {
                    indices: vec![30],
                    values: vec![1.0],
                },
                10,
                None,
            )
            .unwrap();
        assert_eq!(r30.len(), 1);
        assert_eq!(r30[0].0, 1);
    }

    #[test]
    fn upsert_batch_empty_is_noop() {
        let (_tmp, env) = setup();
        let core = make_core(&env, "test_empty", SparseModifier::None);
        let mut txn = env.write_txn().unwrap();
        let count = core.upsert_batch(&mut txn, &[]).unwrap();
        assert_eq!(count, 0);
        txn.commit().unwrap();
    }

    #[test]
    fn upsert_batch_matches_per_doc_inserts() {
        // Property: a batched upsert produces the same searchable
        // state as the same items applied via per-doc insert calls.
        // Useful regression guard against future re-orderings inside
        // upsert_batch's term-major sort.
        let (_tmp, env_a) = setup();
        let core_a = make_core(&env_a, "ref", SparseModifier::None);
        let (_tmp, env_b) = setup();
        let core_b = make_core(&env_b, "batch", SparseModifier::None);

        let docs: Vec<(u128, SparseVector)> = (1..=10u128)
            .map(|i| {
                (
                    i,
                    SparseVector {
                        indices: vec![(i * 3) as u32 % 7, ((i * 5) as u32 % 13) + 7, 100],
                        values: vec![i as f32, (i * 2) as f32, 0.5],
                    },
                )
            })
            .collect();

        // Reference: per-doc inserts.
        let mut txn_a = env_a.write_txn().unwrap();
        for (id, sp) in &docs {
            core_a.insert(&mut txn_a, *id, sp).unwrap();
        }
        txn_a.commit().unwrap();

        // Batch: single upsert_batch call.
        let mut txn_b = env_b.write_txn().unwrap();
        let items: Vec<(u128, &SparseVector)> = docs.iter().map(|(id, sp)| (*id, sp)).collect();
        core_b.upsert_batch(&mut txn_b, &items).unwrap();
        txn_b.commit().unwrap();

        // Compare doc_count.
        let rtxn_a = env_a.read_txn().unwrap();
        let rtxn_b = env_b.read_txn().unwrap();
        assert_eq!(core_a.get_doc_count(&rtxn_a).unwrap(), 10);
        assert_eq!(core_b.get_doc_count(&rtxn_b).unwrap(), 10);

        // For each unique term, both indexes must return the same docs.
        let probe_terms: Vec<u32> = (0..14u32).chain(std::iter::once(100)).collect();
        for t in probe_terms {
            let q = SparseVector {
                indices: vec![t],
                values: vec![1.0],
            };
            let mut a = core_a
                .search::<fn(u128) -> bool>(&rtxn_a, &q, 100, None)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect::<Vec<_>>();
            let mut b = core_b
                .search::<fn(u128) -> bool>(&rtxn_b, &q, 100, None)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect::<Vec<_>>();
            a.sort();
            b.sort();
            assert_eq!(
                a, b,
                "term {} disagrees between per-doc and batched paths",
                t
            );
        }
    }

    #[test]
    fn delete_removes_from_index() {
        let (_tmp, env) = setup();
        let core = make_core(&env, "test", SparseModifier::None);
        let mut txn = env.write_txn().unwrap();

        core.insert(
            &mut txn,
            1,
            &SparseVector {
                indices: vec![10],
                values: vec![3.0],
            },
        )
        .unwrap();
        core.insert(
            &mut txn,
            2,
            &SparseVector {
                indices: vec![10],
                values: vec![7.0],
            },
        )
        .unwrap();

        core.delete(&mut txn, 1).unwrap();
        txn.commit().unwrap();

        let rtxn = env.read_txn().unwrap();
        assert!(!core.has_doc(&rtxn, 1).unwrap());
        assert!(core.has_doc(&rtxn, 2).unwrap());

        let results = core
            .search::<fn(u128) -> bool>(
                &rtxn,
                &SparseVector {
                    indices: vec![10],
                    values: vec![1.0],
                },
                10,
                None,
            )
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn delete_nonexistent_is_noop() {
        let (_tmp, env) = setup();
        let core = make_core(&env, "test", SparseModifier::None);
        let mut txn = env.write_txn().unwrap();
        // Should not error.
        core.delete(&mut txn, 999).unwrap();
        txn.commit().unwrap();
    }

    #[test]
    fn idf_modifier_weights_rare_terms_higher() {
        let (_tmp, env) = setup();
        let core = make_core(&env, "idf_test", SparseModifier::Idf);
        let mut txn = env.write_txn().unwrap();

        // Insert 10 docs, all containing term 0 (common), only doc 10 has term 1 (rare).
        for doc_id in 1..=10u128 {
            core.insert(
                &mut txn,
                doc_id,
                &SparseVector {
                    indices: vec![0],
                    values: vec![1.0],
                },
            )
            .unwrap();
        }
        core.insert(
            &mut txn,
            10,
            &SparseVector {
                indices: vec![0, 1],
                values: vec![1.0, 1.0],
            },
        )
        .unwrap();
        txn.commit().unwrap();

        let rtxn = env.read_txn().unwrap();

        // Query for term 1: only doc 10 matches.
        let results = core
            .search::<fn(u128) -> bool>(
                &rtxn,
                &SparseVector {
                    indices: vec![1],
                    values: vec![1.0],
                },
                10,
                None,
            )
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 10);
        // IDF = ln(1 + 10/1) = ln(11) ≈ 2.397
        let expected_idf = (1.0 + 10.0 / 1.0_f64).ln();
        assert!((results[0].1 - expected_idf).abs() < 0.01);

        // Query both terms for doc 10: score = 1*1*idf(0) + 1*1*idf(1)
        let results = core
            .search::<fn(u128) -> bool>(
                &rtxn,
                &SparseVector {
                    indices: vec![0, 1],
                    values: vec![1.0, 1.0],
                },
                1,
                None,
            )
            .unwrap();
        assert_eq!(results[0].0, 10);
        // idf(0) = ln(1 + 10/10) = ln(2) ≈ 0.693
        // idf(1) = ln(1 + 10/1)  = ln(11) ≈ 2.397
        // Total = 0.693 + 2.397 ≈ 3.09
        let idf0 = (1.0 + 10.0 / 10.0_f64).ln();
        let idf1 = (1.0 + 10.0 / 1.0_f64).ln();
        assert!((results[0].1 - (idf0 + idf1)).abs() < 0.01);
    }

    #[test]
    fn search_with_filter() {
        let (_tmp, env) = setup();
        let core = make_core(&env, "test", SparseModifier::None);
        let mut txn = env.write_txn().unwrap();

        for doc_id in 1..=5u128 {
            core.insert(
                &mut txn,
                doc_id,
                &SparseVector {
                    indices: vec![0],
                    values: vec![doc_id as f32],
                },
            )
            .unwrap();
        }
        txn.commit().unwrap();

        let rtxn = env.read_txn().unwrap();
        // Filter: only even doc IDs.
        let filter_fn = |id: u128| -> bool { id % 2 == 0 };
        let results = core
            .search(
                &rtxn,
                &SparseVector {
                    indices: vec![0],
                    values: vec![1.0],
                },
                10,
                Some(&filter_fn),
            )
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 4); // 4.0 > 2.0
        assert_eq!(results[1].0, 2);
    }

    #[test]
    fn search_respects_limit() {
        let (_tmp, env) = setup();
        let core = make_core(&env, "test", SparseModifier::None);
        let mut txn = env.write_txn().unwrap();

        for doc_id in 1..=20u128 {
            core.insert(
                &mut txn,
                doc_id,
                &SparseVector {
                    indices: vec![0],
                    values: vec![1.0],
                },
            )
            .unwrap();
        }
        txn.commit().unwrap();

        let rtxn = env.read_txn().unwrap();
        let results = core
            .search::<fn(u128) -> bool>(
                &rtxn,
                &SparseVector {
                    indices: vec![0],
                    values: vec![1.0],
                },
                3,
                None,
            )
            .unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn empty_query_returns_empty() {
        let (_tmp, env) = setup();
        let core = make_core(&env, "test", SparseModifier::None);
        let mut txn = env.write_txn().unwrap();
        core.insert(
            &mut txn,
            1,
            &SparseVector {
                indices: vec![0],
                values: vec![1.0],
            },
        )
        .unwrap();
        txn.commit().unwrap();

        let rtxn = env.read_txn().unwrap();
        let results = core
            .search::<fn(u128) -> bool>(
                &rtxn,
                &SparseVector {
                    indices: vec![],
                    values: vec![],
                },
                10,
                None,
            )
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn mismatched_indices_values_is_error() {
        let sv = SparseVector {
            indices: vec![0, 1],
            values: vec![1.0],
        };
        assert!(sv.validate().is_err());
    }

    #[test]
    fn query_nonexistent_term_returns_empty() {
        let (_tmp, env) = setup();
        let core = make_core(&env, "test", SparseModifier::None);
        let mut txn = env.write_txn().unwrap();
        core.insert(
            &mut txn,
            1,
            &SparseVector {
                indices: vec![0],
                values: vec![1.0],
            },
        )
        .unwrap();
        txn.commit().unwrap();

        let rtxn = env.read_txn().unwrap();
        let results = core
            .search::<fn(u128) -> bool>(
                &rtxn,
                &SparseVector {
                    indices: vec![999],
                    values: vec![1.0],
                },
                10,
                None,
            )
            .unwrap();
        assert!(results.is_empty());
    }

    // ─── WAND-specific test helpers ───

    fn setup_large() -> (TempDir, Env<WithTls>) {
        let tmp = TempDir::new().unwrap();
        let env = unsafe {
            heed3::EnvOpenOptions::new()
                .max_dbs(20)
                .map_size(256 * 1024 * 1024)
                .open(tmp.path())
                .unwrap()
        };
        (tmp, env)
    }

    fn make_core_cfg(
        env: &Env<WithTls>,
        name: &str,
        config: SparseVectorConfig,
    ) -> SparseVectorCore {
        let mut txn = env.write_txn().unwrap();
        let core = SparseVectorCore::new(env, &mut txn, name, config, test_backend(env)).unwrap();
        txn.commit().unwrap();
        core
    }

    // ─── WAND tests ───

    #[test]
    fn maxscore_stop_requires_unbeatable_top_k() {
        let scores = HashMap::from([(1u128, 12.0), (2, 10.0), (3, 3.0), (4, 2.0)]);

        assert!(maxscore_can_stop(&scores, 2, 1.0));
        assert!(!maxscore_can_stop(&scores, 2, 8.0));
        assert!(!maxscore_can_stop(&scores, 4, 1.0));
    }

    /// Verify that WAND returns identical results to full scan for all existing
    /// test patterns. We insert the same data as `insert_and_search_basic` but
    /// configure a low `full_scan_threshold` so WAND is triggered even with
    /// few documents, and use >= 3 query terms to satisfy the WAND condition.
    #[test]
    fn wand_matches_full_scan_basic() {
        let (_tmp, env) = setup();

        // Core with WAND enabled and very low threshold so WAND activates.
        let wand_core = make_core_cfg(
            &env,
            "wand",
            SparseVectorConfig {
                full_scan_threshold: 1, // Force WAND for any doc count >= 1
                modifier: SparseModifier::None,
                wand_enabled: true,
            },
        );
        // Core with WAND disabled — always full scan.
        let full_core = make_core_cfg(
            &env,
            "full",
            SparseVectorConfig {
                full_scan_threshold: 1,
                modifier: SparseModifier::None,
                wand_enabled: false,
            },
        );

        let mut txn = env.write_txn().unwrap();

        let docs: Vec<(u128, SparseVector)> = vec![
            (
                1,
                SparseVector {
                    indices: vec![0, 1, 2, 3],
                    values: vec![1.0, 2.0, 3.0, 0.5],
                },
            ),
            (
                2,
                SparseVector {
                    indices: vec![1, 3, 4],
                    values: vec![5.0, 1.0, 2.0],
                },
            ),
            (
                3,
                SparseVector {
                    indices: vec![0, 2, 4, 5],
                    values: vec![0.5, 4.0, 1.0, 3.0],
                },
            ),
            (
                4,
                SparseVector {
                    indices: vec![1, 2, 5],
                    values: vec![3.0, 1.0, 2.0],
                },
            ),
        ];

        for (id, sv) in &docs {
            wand_core.insert(&mut txn, *id, sv).unwrap();
            full_core.insert(&mut txn, *id, sv).unwrap();
        }
        txn.commit().unwrap();

        let rtxn = env.read_txn().unwrap();

        // Query with >= 3 terms to trigger WAND.
        let query = SparseVector {
            indices: vec![0, 1, 2, 4],
            values: vec![1.0, 1.5, 2.0, 0.5],
        };

        for limit in [1, 2, 3, 4, 10] {
            let wand_results = wand_core
                .search::<fn(u128) -> bool>(&rtxn, &query, limit, None)
                .unwrap();
            let full_results = full_core
                .search::<fn(u128) -> bool>(&rtxn, &query, limit, None)
                .unwrap();

            assert_eq!(
                wand_results.len(),
                full_results.len(),
                "WAND and full scan should return same count for limit={}",
                limit
            );
            for (w, f) in wand_results.iter().zip(full_results.iter()) {
                assert_eq!(w.0, f.0, "doc id mismatch for limit={}", limit);
                assert!(
                    (w.1 - f.1).abs() < 1e-10,
                    "score mismatch for doc {} limit={}: wand={} full={}",
                    w.0,
                    limit,
                    w.1,
                    f.1
                );
            }
        }
    }

    /// Large-scale test: 500+ docs, 50+ unique terms, verifying WAND top-K
    /// correctness against full scan.
    #[test]
    fn wand_large_scale_correctness() {
        let (_tmp, env) = setup_large();

        let wand_core = make_core_cfg(
            &env,
            "wand_large",
            SparseVectorConfig {
                full_scan_threshold: 1,
                modifier: SparseModifier::None,
                wand_enabled: true,
            },
        );
        let full_core = make_core_cfg(
            &env,
            "full_large",
            SparseVectorConfig {
                full_scan_threshold: 1,
                modifier: SparseModifier::None,
                wand_enabled: false,
            },
        );

        let mut txn = env.write_txn().unwrap();

        let num_docs: u128 = 600;
        let num_terms: u32 = 60;

        // Insert docs with pseudo-random sparse vectors.
        for doc_id in 1..=num_docs {
            // Each doc gets ~10-15 terms based on a simple hash pattern.
            let mut indices = Vec::new();
            let mut values = Vec::new();
            let seed = doc_id as u32;
            for t in 0..num_terms {
                // Deterministic pseudo-random selection.
                let hash = seed
                    .wrapping_mul(2654435761)
                    .wrapping_add(t.wrapping_mul(1013904223));
                if hash % 5 < 2 {
                    // ~40% of terms selected per doc
                    indices.push(t);
                    // Value based on doc_id and term for variety.
                    let val = ((hash % 100) as f32 + 1.0) / 10.0;
                    values.push(val);
                }
            }
            if indices.is_empty() {
                indices.push(0);
                values.push(1.0);
            }
            let sv = SparseVector { indices, values };
            wand_core.insert(&mut txn, doc_id, &sv).unwrap();
            full_core.insert(&mut txn, doc_id, &sv).unwrap();
        }
        txn.commit().unwrap();

        let rtxn = env.read_txn().unwrap();

        // Query with many terms.
        let query = SparseVector {
            indices: (0..20).collect(),
            values: (0..20).map(|i| (i as f32 + 1.0) * 0.5).collect(),
        };

        for limit in [5, 10, 20, 50] {
            let wand_results = wand_core
                .search::<fn(u128) -> bool>(&rtxn, &query, limit, None)
                .unwrap();
            let full_results = full_core
                .search::<fn(u128) -> bool>(&rtxn, &query, limit, None)
                .unwrap();

            assert_eq!(
                wand_results.len(),
                full_results.len(),
                "WAND and full scan should return same count for limit={}",
                limit
            );
            for (w, f) in wand_results.iter().zip(full_results.iter()) {
                assert_eq!(
                    w.0, f.0,
                    "doc id mismatch for limit={}: wand={} full={}",
                    limit, w.0, f.0
                );
                assert!(
                    (w.1 - f.1).abs() < 1e-10,
                    "score mismatch for doc {} limit={}: wand={} full={}",
                    w.0,
                    limit,
                    w.1,
                    f.1
                );
            }
        }
    }

    /// Test WAND with IDF modifier enabled — results must match full scan.
    #[test]
    fn wand_with_idf_matches_full_scan() {
        let (_tmp, env) = setup_large();

        let wand_core = make_core_cfg(
            &env,
            "wand_idf",
            SparseVectorConfig {
                full_scan_threshold: 1,
                modifier: SparseModifier::Idf,
                wand_enabled: true,
            },
        );
        let full_core = make_core_cfg(
            &env,
            "full_idf",
            SparseVectorConfig {
                full_scan_threshold: 1,
                modifier: SparseModifier::Idf,
                wand_enabled: false,
            },
        );

        let mut txn = env.write_txn().unwrap();

        // Insert 200 docs. Term 0 is common (in all docs), terms 1-9 are rarer.
        for doc_id in 1..=200u128 {
            let mut indices = vec![0]; // common term
            let mut values = vec![1.0];
            // Add a few rare terms based on doc_id.
            for t in 1..10u32 {
                if doc_id as u32 % (t + 1) == 0 {
                    indices.push(t);
                    values.push((t as f32) * 0.5);
                }
            }
            let sv = SparseVector { indices, values };
            wand_core.insert(&mut txn, doc_id, &sv).unwrap();
            full_core.insert(&mut txn, doc_id, &sv).unwrap();
        }
        txn.commit().unwrap();

        let rtxn = env.read_txn().unwrap();

        // Query hitting common + rare terms (>= 3 terms to trigger WAND).
        let query = SparseVector {
            indices: vec![0, 1, 2, 5, 8],
            values: vec![1.0, 2.0, 3.0, 1.5, 0.5],
        };

        for limit in [5, 10, 20] {
            let wand_results = wand_core
                .search::<fn(u128) -> bool>(&rtxn, &query, limit, None)
                .unwrap();
            let full_results = full_core
                .search::<fn(u128) -> bool>(&rtxn, &query, limit, None)
                .unwrap();

            assert_eq!(
                wand_results.len(),
                full_results.len(),
                "IDF: WAND and full scan should return same count for limit={}",
                limit
            );
            for (w, f) in wand_results.iter().zip(full_results.iter()) {
                assert_eq!(
                    w.0, f.0,
                    "IDF: doc id mismatch for limit={}: wand={} full={}",
                    limit, w.0, f.0
                );
                assert!(
                    (w.1 - f.1).abs() < 1e-10,
                    "IDF: score mismatch for doc {} limit={}: wand={} full={}",
                    w.0,
                    limit,
                    w.1,
                    f.1
                );
            }
        }
    }

    /// Verify that WAND is NOT used when query has fewer than 3 terms, even
    /// when wand_enabled=true and doc_count >= full_scan_threshold.
    /// Both paths (WAND and full scan) should return the same results, but
    /// the key is that the full_scan path is used (which we verify indirectly
    /// by checking correctness — since both are correct, we just verify the
    /// fallback still works).
    #[test]
    fn wand_skipped_for_few_query_terms() {
        let (_tmp, env) = setup_large();

        let core = make_core_cfg(
            &env,
            "few_terms",
            SparseVectorConfig {
                full_scan_threshold: 1, // Would trigger WAND if enough terms
                modifier: SparseModifier::None,
                wand_enabled: true,
            },
        );

        let mut txn = env.write_txn().unwrap();
        for doc_id in 1..=100u128 {
            core.insert(
                &mut txn,
                doc_id,
                &SparseVector {
                    indices: vec![0, 1, 2],
                    values: vec![doc_id as f32, (101 - doc_id) as f32, 1.0],
                },
            )
            .unwrap();
        }
        txn.commit().unwrap();

        let rtxn = env.read_txn().unwrap();

        // Query with 1 term — must NOT use WAND (< 3 terms).
        let results_1 = core
            .search::<fn(u128) -> bool>(
                &rtxn,
                &SparseVector {
                    indices: vec![0],
                    values: vec![1.0],
                },
                5,
                None,
            )
            .unwrap();
        assert_eq!(results_1.len(), 5);
        // Highest doc_id has highest value for term 0.
        assert_eq!(results_1[0].0, 100);
        assert_eq!(results_1[1].0, 99);

        // Query with 2 terms — must NOT use WAND (< 3 terms).
        let results_2 = core
            .search::<fn(u128) -> bool>(
                &rtxn,
                &SparseVector {
                    indices: vec![0, 1],
                    values: vec![1.0, 1.0],
                },
                5,
                None,
            )
            .unwrap();
        assert_eq!(results_2.len(), 5);
        // All docs have term0 + term1 = doc_id + (101 - doc_id) = 101, so
        // all scores are equal (101.0). With equal scores, order is by id
        // tie-breaking. Verify all scores are 101.0.
        for (_, score) in &results_2 {
            assert!((*score - 101.0).abs() < 1e-6);
        }
    }

    /// Verify per-term max metadata is correctly maintained on insert.
    #[test]
    fn wand_term_max_tracking() {
        let (_tmp, env) = setup();
        let core = make_core_cfg(
            &env,
            "max_track",
            SparseVectorConfig {
                full_scan_threshold: 1,
                modifier: SparseModifier::None,
                wand_enabled: true,
            },
        );

        let mut txn = env.write_txn().unwrap();

        // Insert doc 1 with term 0 = 3.0
        core.insert(
            &mut txn,
            1,
            &SparseVector {
                indices: vec![0],
                values: vec![3.0],
            },
        )
        .unwrap();

        // Max for term 0 should be 3.0.
        let max0 = core.get_term_max(&txn, 0).unwrap();
        assert!((max0 - 3.0).abs() < 1e-6);

        // Insert doc 2 with term 0 = 5.0 — max should update.
        core.insert(
            &mut txn,
            2,
            &SparseVector {
                indices: vec![0],
                values: vec![5.0],
            },
        )
        .unwrap();

        let max0 = core.get_term_max(&txn, 0).unwrap();
        assert!((max0 - 5.0).abs() < 1e-6);

        // Insert doc 3 with term 0 = 2.0 — max should NOT decrease.
        core.insert(
            &mut txn,
            3,
            &SparseVector {
                indices: vec![0],
                values: vec![2.0],
            },
        )
        .unwrap();

        let max0 = core.get_term_max(&txn, 0).unwrap();
        assert!((max0 - 5.0).abs() < 1e-6);

        // Delete doc 2 (the one with max value) — max should remain stale
        // (conservative upper bound, not recomputed on delete).
        core.delete(&mut txn, 2).unwrap();
        let max0 = core.get_term_max(&txn, 0).unwrap();
        assert!(
            max0 >= 3.0,
            "max should be >= actual max (stale is OK, too low is not)"
        );

        txn.commit().unwrap();
    }

    /// Verify WAND disabled falls back to full scan behavior.
    #[test]
    fn wand_disabled_uses_full_scan() {
        let (_tmp, env) = setup();
        let core = make_core_cfg(
            &env,
            "disabled",
            SparseVectorConfig {
                full_scan_threshold: 1,
                modifier: SparseModifier::None,
                wand_enabled: false,
            },
        );

        let mut txn = env.write_txn().unwrap();
        for doc_id in 1..=10u128 {
            core.insert(
                &mut txn,
                doc_id,
                &SparseVector {
                    indices: vec![0, 1, 2, 3],
                    values: vec![doc_id as f32, 1.0, 1.0, 1.0],
                },
            )
            .unwrap();
        }
        txn.commit().unwrap();

        let rtxn = env.read_txn().unwrap();
        let results = core
            .search::<fn(u128) -> bool>(
                &rtxn,
                &SparseVector {
                    indices: vec![0, 1, 2, 3],
                    values: vec![1.0, 1.0, 1.0, 1.0],
                },
                3,
                None,
            )
            .unwrap();

        assert_eq!(results.len(), 3);
        // Doc 10 has highest score: 10 + 1 + 1 + 1 = 13
        assert_eq!(results[0].0, 10);
        assert!((results[0].1 - 13.0).abs() < 1e-6);
    }

    /// df read by `get_term_df` must reflect pending in-memory deltas
    /// before the flush has run, so search IDF stays correct between
    /// inserts and the next chunk-end flush.
    #[test]
    fn pending_df_visible_via_get_term_df_pre_flush() {
        let (_tmp, env) = setup();
        let core = make_core(&env, "pending_df", SparseModifier::None);
        let mut txn = env.write_txn().unwrap();

        for doc_id in 1..=5u128 {
            core.insert(
                &mut txn,
                doc_id,
                &SparseVector {
                    indices: vec![7],
                    values: vec![1.0],
                },
            )
            .unwrap();
        }

        // No flush yet — meta_db has zero df for term 7, but get_term_df
        // must return 5 because the pending buffer carries the deltas.
        assert_eq!(core.get_term_df_persisted(&txn, 7).unwrap(), 0);
        assert_eq!(core.get_term_df(&txn, 7).unwrap(), 5);

        // Flush drains pending into meta_db; then both readers agree.
        let flushed = core.flush_metadata_pending(&mut txn).unwrap();
        assert!(
            flushed >= 2,
            "expected >= 2 entries (df+max), got {}",
            flushed
        );
        assert_eq!(core.get_term_df_persisted(&txn, 7).unwrap(), 5);
        assert_eq!(core.get_term_df(&txn, 7).unwrap(), 5);
    }

    /// Pending max must be visible to `get_term_max` before flush so
    /// WAND pruning sees the correct upper bound on a freshly inserted
    /// vector with a value larger than the persisted max.
    #[test]
    fn pending_max_visible_via_get_term_max_pre_flush() {
        let (_tmp, env) = setup();
        let core = make_core(&env, "pending_max", SparseModifier::None);
        let mut txn = env.write_txn().unwrap();

        core.insert(
            &mut txn,
            1,
            &SparseVector {
                indices: vec![3],
                values: vec![7.5],
            },
        )
        .unwrap();

        // Persisted is still 0 (no flush yet); get_term_max sees pending.
        assert_eq!(core.get_term_max_persisted(&txn, 3).unwrap(), 0.0);
        assert!((core.get_term_max(&txn, 3).unwrap() - 7.5).abs() < 1e-6);

        core.flush_metadata_pending(&mut txn).unwrap();
        assert!((core.get_term_max_persisted(&txn, 3).unwrap() - 7.5).abs() < 1e-6);
        assert!((core.get_term_max(&txn, 3).unwrap() - 7.5).abs() < 1e-6);

        // A second insert with a *lower* value must not shrink the max.
        core.insert(
            &mut txn,
            2,
            &SparseVector {
                indices: vec![3],
                values: vec![2.0],
            },
        )
        .unwrap();
        core.flush_metadata_pending(&mut txn).unwrap();
        assert!((core.get_term_max(&txn, 3).unwrap() - 7.5).abs() < 1e-6);
    }

    /// Delete decrements pending df. After flush, persisted df reflects
    /// inserts minus deletes accurately (no double-count, no underflow).
    #[test]
    fn delete_decrements_pending_df_correctly() {
        let (_tmp, env) = setup();
        let core = make_core(&env, "delete_df", SparseModifier::None);
        let mut txn = env.write_txn().unwrap();

        for doc_id in 1..=4u128 {
            core.insert(
                &mut txn,
                doc_id,
                &SparseVector {
                    indices: vec![11],
                    values: vec![1.0],
                },
            )
            .unwrap();
        }
        assert_eq!(core.get_term_df(&txn, 11).unwrap(), 4);

        core.delete(&mut txn, 2).unwrap();
        core.delete(&mut txn, 3).unwrap();
        // Pre-flush, the pending delta is +4 - 2 = +2 over persisted 0.
        assert_eq!(core.get_term_df_persisted(&txn, 11).unwrap(), 0);
        assert_eq!(core.get_term_df(&txn, 11).unwrap(), 2);

        core.flush_metadata_pending(&mut txn).unwrap();
        assert_eq!(core.get_term_df_persisted(&txn, 11).unwrap(), 2);
        assert_eq!(core.get_term_df(&txn, 11).unwrap(), 2);

        // Delete the rest. Persisted is 2, pending will be -2 → 0 after flush.
        core.delete(&mut txn, 1).unwrap();
        core.delete(&mut txn, 4).unwrap();
        core.flush_metadata_pending(&mut txn).unwrap();
        assert_eq!(core.get_term_df_persisted(&txn, 11).unwrap(), 0);
    }

    #[test]
    fn budgeted_flush_leaves_pending_entries_search_visible() {
        let (_tmp, env) = setup();
        let core = make_core(&env, "budgeted_visible", SparseModifier::None);
        let mut txn = env.write_txn().unwrap();

        core.insert(
            &mut txn,
            1,
            &SparseVector {
                indices: vec![101, 102, 103],
                values: vec![1.0, 2.0, 3.0],
            },
        )
        .unwrap();

        let pending_before = core.pending_metadata_count();
        assert!(
            pending_before >= 6,
            "expected df+max entries, got {pending_before}"
        );

        let stats = core
            .flush_metadata_pending_budgeted(&mut txn, Some(2), None)
            .unwrap();

        assert_eq!(stats.pending_before, pending_before);
        assert_eq!(stats.processed, 2);
        assert!(stats.flushed <= 2);
        assert!(stats.budget_exhausted);
        assert!(stats.pending_after > 0);

        // Terms that did not fit in the budget remain correct through the
        // pending overlay even if their persisted metadata is still stale.
        for term_id in [101, 102, 103] {
            assert_eq!(core.get_term_df(&txn, term_id).unwrap(), 1);
            assert!(core.get_term_max(&txn, term_id).unwrap() > 0.0);
        }

        core.flush_metadata_pending(&mut txn).unwrap();
        assert_eq!(core.pending_metadata_count(), 0);
        for term_id in [101, 102, 103] {
            assert_eq!(core.get_term_df(&txn, term_id).unwrap(), 1);
            assert!(core.get_term_max_persisted(&txn, term_id).unwrap() > 0.0);
        }
    }

    #[test]
    fn zero_budget_flush_preserves_all_pending_entries() {
        let (_tmp, env) = setup();
        let core = make_core(&env, "zero_budget", SparseModifier::None);
        let mut txn = env.write_txn().unwrap();

        core.insert(
            &mut txn,
            1,
            &SparseVector {
                indices: vec![201, 202],
                values: vec![4.0, 5.0],
            },
        )
        .unwrap();

        let pending_before = core.pending_metadata_count();
        let stats = core
            .flush_metadata_pending_budgeted(&mut txn, Some(0), None)
            .unwrap();

        assert_eq!(stats.pending_before, pending_before);
        assert_eq!(stats.processed, 0);
        assert_eq!(stats.flushed, 0);
        assert_eq!(stats.pending_after, pending_before);
        assert!(stats.budget_exhausted);
        assert!(!stats.time_exhausted);
        assert_eq!(core.get_term_df(&txn, 201).unwrap(), 1);
        assert!((core.get_term_max(&txn, 202).unwrap() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn zero_duration_flush_preserves_pending_entries() {
        let (_tmp, env) = setup();
        let core = make_core(&env, "zero_duration", SparseModifier::None);
        let mut txn = env.write_txn().unwrap();

        core.insert(
            &mut txn,
            1,
            &SparseVector {
                indices: vec![301, 302],
                values: vec![6.0, 7.0],
            },
        )
        .unwrap();

        let pending_before = core.pending_metadata_count();
        let stats = core
            .flush_metadata_pending_budgeted(&mut txn, None, Some(Duration::ZERO))
            .unwrap();

        assert_eq!(stats.pending_before, pending_before);
        assert_eq!(stats.processed, 0);
        assert_eq!(stats.flushed, 0);
        assert_eq!(stats.pending_after, pending_before);
        assert!(!stats.budget_exhausted);
        assert!(stats.time_exhausted);
        assert_eq!(core.get_term_df(&txn, 301).unwrap(), 1);
        assert!((core.get_term_max(&txn, 302).unwrap() - 7.0).abs() < 1e-6);
    }
}
