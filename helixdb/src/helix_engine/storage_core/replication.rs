use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, LazyLock, Mutex, Weak,
};
use std::thread;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use flume::{Receiver, Sender};
use heed3::RwTxn;
use prost::Message as ProstMessage;
use raft::prelude::Message;
use serde::{Deserialize, Serialize};

use crate::helix_engine::graph_core::{config::Config, traversals::algorithms};
use crate::helix_engine::storage_core::backend::{BackendKind, Namespace, StorageBackend};
use crate::helix_engine::storage_core::collection_manager::{
    maintenance_process_resource_admission, CollectionManager, MaintenanceAdmission,
};
use crate::helix_engine::storage_core::gateway_buffer::{
    GatewayBufferAck, GatewayBufferConfig, GatewayDurableBuffer,
};
use crate::helix_engine::storage_core::metadata::{MetadataCounter, PayloadIndexSchema};
use crate::helix_engine::storage_core::raft::{AppliedProposal, Proposal, RaftNode};
use crate::helix_engine::storage_core::storage_core::HelixGraphStorage;
use crate::helix_engine::storage_core::storage_methods::StorageMethods;
use crate::helix_engine::storage_core::upsert::{EdgeUpsert, NodeUpsert};
use crate::helix_engine::types::GraphError;
use crate::helix_engine::vector_core::named_vectors::{
    DenseDeletePlan, DenseMergeCandidateLimits, DenseSegmentDebt, NamedVectorConfig,
    NamedVectorManager,
};
use crate::helix_engine::vector_core::sparse::{SparseVector, SparseVectorConfig};
use crate::helix_engine::vector_core::vector_core::{
    acquire_build_permit, HNSWConfig, HnswOverrides, VectorCore,
};
use crate::helix_gateway::thread_pool::thread_pool::{route_class, RouteClass};
use crate::protocol::items::SerializedNode;
use crate::protocol::request::Request;
use crate::protocol::response::Response;
use crate::protocol::value::Value;

const INTERNAL_RAFT_MESSAGE_PATH: &str = "/_raft/message";
const INTERNAL_RAFT_PROPOSE_PATH: &str = "/_raft/propose";
const PROPOSE_TIMEOUT: Duration = Duration::from_secs(15);
const LEADER_WAIT_TIMEOUT: Duration = Duration::from_secs(5);
const READ_TIMEOUT: Duration = Duration::from_secs(15);
const GATEWAY_READER_EVICT_MS: u64 = 5_000;
const GATEWAY_READER_PROXY_TIMEOUT_MS: u64 = 15_000;
const GATEWAY_WRITER_PROXY_TIMEOUT_MS: u64 = 30_000;
const GATEWAY_BYPASS_HEADER: &str = "x-helix-gateway-bypass";
const GATEWAY_BYPASS_VALUE: &str = "1";
const GATEWAY_BUFFER_MAX_ENTRIES: usize = 10_000;
const GATEWAY_BUFFER_MAX_BYTES: u64 = 1024 * 1024 * 1024;
const GATEWAY_BUFFER_REPLAY_MS: u64 = 1_000;
static CONTENT_HASH_MISMATCH_DIAG_COUNT: AtomicU64 = AtomicU64::new(0);
static PROXY_HTTP_CLIENT: LazyLock<reqwest::blocking::Client> = LazyLock::new(|| {
    reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(2))
        .pool_idle_timeout(Duration::from_secs(90))
        .build()
        .expect("gateway proxy HTTP client must build")
});

#[derive(Clone)]
struct IndexJob {
    collection: String,
    storage: Weak<HelixGraphStorage>,
    spaces_to_check: Vec<String>,
    hnsw_config: HNSWConfig,
    flat_scan_threshold: usize,
}

impl IndexJob {
    fn merge_from(&mut self, other: Self) {
        let mut seen: HashSet<String> = self.spaces_to_check.iter().cloned().collect();
        for name in other.spaces_to_check {
            if seen.insert(name.clone()) {
                self.spaces_to_check.push(name);
            }
        }
        self.hnsw_config = other.hnsw_config;
        self.flat_scan_threshold = other.flat_scan_threshold;
        self.storage = other.storage;
    }
}

fn index_job_backlog_ratio_millis(job: &IndexJob) -> usize {
    let target = NamedVectorManager::max_indexed_segments_target().max(1);
    job.storage
        .upgrade()
        .map(|storage| {
            storage
                .named_vectors
                .dense_segment_debt()
                .indexed_segments
                .saturating_mul(1_000)
                / target
        })
        .unwrap_or(0)
}

/// Per-collection segment-breaker recovery floor: `HELIX_PER_COLLECTION_SEGMENT_LIMIT`
/// × `HELIX_SEGMENT_BREAKER_RECOVERY_RATIO`, mirroring the gateway upsert breaker.
/// A collection at/over this many indexed dense segments is returning (or about
/// to return) client-facing 503s on upserts. `None` when the breaker is disabled
/// (limit unset).
fn segment_breaker_drain_floor() -> Option<usize> {
    let limit = std::env::var("HELIX_PER_COLLECTION_SEGMENT_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)?;
    let ratio = std::env::var("HELIX_SEGMENT_BREAKER_RECOVERY_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|r| *r > 0.0 && *r < 1.0)
        .unwrap_or(0.75);
    Some(((limit as f64) * ratio).floor() as usize)
}

fn dense_drain_merge_max_fan_in() -> usize {
    env_usize_or("HELIX_DENSE_DRAIN_MERGE_MAX_FANIN", 2, 2)
}

fn breaker_drain_merge_fan_in_limit(
    debt: DenseSegmentDebt,
    drain_floor: Option<usize>,
) -> Option<usize> {
    drain_floor
        .filter(|floor| debt.indexed_segments >= *floor)
        .map(|_| dense_drain_merge_max_fan_in())
}

fn should_reserve_index_job_turn_for_merges(
    debt: DenseSegmentDebt,
    _breaker_drain_merge_fan_in: Option<usize>,
) -> bool {
    debt.merge_debt > 0
}

fn optimizer_debt_priority_score(debt: DenseSegmentDebt, drain_floor: Option<usize>) -> usize {
    let target = NamedVectorManager::max_indexed_segments_target().max(1);
    let backlog_score = debt.indexed_segments.saturating_mul(1_000) / target;
    // A collection at/over the breaker recovery floor is 503ing client upserts.
    // That outage must outrank gate-debt freshness work so its drain actually
    // wins a build slot — otherwise tripping the breaker zeroes the collection's
    // own gate_debt and it starves behind healthy collections forever. Scale by
    // overage so the most-over (longest-stuck) collection drains first.
    let breaker_pressure = drain_floor
        .filter(|floor| debt.indexed_segments >= *floor)
        .map(|floor| {
            debt.indexed_segments
                .saturating_sub(floor)
                .saturating_add(1)
                .saturating_mul(1_000_000_000_000)
        })
        .unwrap_or(0);
    // Dirty-retired cleanup is scheduled by SegmentReaper. Counting it here
    // lets cleanup-heavy collections monopolize merge/build workers.
    breaker_pressure
        .saturating_add(debt.gate_debt.saturating_mul(1_000_000_000))
        .saturating_add(debt.merge_debt.saturating_mul(1_000))
        .saturating_add(backlog_score)
}

fn reaper_debt_priority_score(debt: DenseSegmentDebt, drain_floor: Option<usize>) -> usize {
    optimizer_debt_priority_score(debt, drain_floor)
        .saturating_add(debt.dirty_retired_segments.saturating_mul(100_000_000))
}

fn index_job_debt(job: &IndexJob) -> DenseSegmentDebt {
    job.storage
        .upgrade()
        .map(|storage| storage.named_vectors.dense_segment_debt())
        .unwrap_or_default()
}

#[derive(Default)]
struct IndexExecutorState {
    queued_order: VecDeque<String>,
    queued_jobs: HashMap<String, IndexJob>,
    active: HashSet<String>,
    /// Subset of `active` whose jobs were selected as breaker drains (indexed
    /// segments at/over the breaker recovery floor). Used to reserve workers for
    /// non-breaker indexing.
    active_drains: HashSet<String>,
}

#[derive(Debug)]
struct IndexExecutorSubmitError;

impl std::fmt::Display for IndexExecutorSubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "index executor queue full")
    }
}

impl std::error::Error for IndexExecutorSubmitError {}

struct IndexExecutor {
    state: Arc<(Mutex<IndexExecutorState>, std::sync::Condvar)>,
    queue_cap: usize,
}

fn index_subthreshold_quiesced_tails() -> bool {
    std::env::var("HELIX_INDEX_SUBTHRESHOLD_QUIESCED_TAILS")
        .ok()
        .and_then(|value| match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
        .unwrap_or(true)
}

fn env_usize_or(name: &str, default: usize, min: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= min)
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn scaled_index_job_max_merges(indexed_segments: usize, base_max_merges: usize) -> usize {
    let target = NamedVectorManager::max_indexed_segments_target().max(1);
    if indexed_segments > target.saturating_mul(10) {
        base_max_merges.saturating_mul(2)
    } else {
        base_max_merges
    }
}

struct IndexJobBudget {
    started: Instant,
    max_runtime: Option<Duration>,
    max_build_segments: usize,
    max_merges: usize,
    build_segments: usize,
    merges: usize,
    merge_after_build: bool,
}

impl IndexJobBudget {
    fn new(indexed_segments: usize) -> Self {
        let runtime_ms = env_u64("HELIX_INDEX_JOB_MAX_RUNTIME_MS", 5_000);
        let max_merges = scaled_index_job_max_merges(
            indexed_segments,
            env_usize_or("HELIX_INDEX_JOB_MAX_MERGES", 1, 1),
        );
        Self {
            started: Instant::now(),
            max_runtime: if runtime_ms == 0 {
                None
            } else {
                Some(Duration::from_millis(runtime_ms))
            },
            max_build_segments: env_usize_or("HELIX_INDEX_JOB_MAX_BUILD_SEGMENTS", 1, 1),
            max_merges,
            build_segments: 0,
            merges: 0,
            merge_after_build: false,
        }
    }

    fn remaining_build_segments(&self) -> usize {
        self.max_build_segments.saturating_sub(self.build_segments)
    }

    fn can_merge(&self) -> bool {
        self.merges < self.max_merges
    }

    fn record_build_segments(&mut self, count: usize) {
        self.build_segments = self.build_segments.saturating_add(count);
        if count > 0 {
            metrics::counter!("helix_index_job_build_segments_total").increment(count as u64);
        }
    }

    fn record_merge(&mut self) {
        self.merges = self.merges.saturating_add(1);
        metrics::counter!("helix_index_job_merges_total").increment(1);
    }

    fn elapsed_exceeded(&self) -> bool {
        self.max_runtime
            .map(|limit| self.started.elapsed() >= limit)
            .unwrap_or(false)
    }

    fn should_yield(&self) -> bool {
        self.elapsed_exceeded()
            || (!self.merge_after_build
                && self.max_build_segments > 0
                && self.build_segments >= self.max_build_segments)
            || self.merges >= self.max_merges
    }

    fn reserve_turn_for_merges(&mut self) {
        self.max_build_segments = 0;
        self.merge_after_build = true;
    }

    fn is_merge_only_turn(&self) -> bool {
        self.merge_after_build && self.max_build_segments == 0
    }

    fn expand_builds_for_gate_debt(&mut self, gate_debt: usize) {
        self.max_build_segments = self
            .max_build_segments
            .max(gate_debt_build_segments_for(gate_debt));
        self.merge_after_build = true;
    }
}

fn index_executor_worker_count() -> usize {
    std::env::var("HELIX_INDEX_EXECUTOR_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2)
}

/// Maximum workers that may run breaker-drain jobs concurrently. Reserves at
/// least one worker for normal (non-breaker) index maintenance so a wave of
/// circuit-breaker drains can't fully starve freshness indexing. Override with
/// `HELIX_INDEX_EXECUTOR_MAX_DRAIN_WORKERS` (clamped to `[1, workers]`).
fn index_executor_max_drain_workers() -> usize {
    let workers = index_executor_worker_count();
    let default_cap = workers.saturating_sub(1).max(1);
    std::env::var("HELIX_INDEX_EXECUTOR_MAX_DRAIN_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 1 && *value <= workers)
        .unwrap_or(default_cap)
}

impl IndexExecutor {
    fn new() -> Self {
        let workers = index_executor_worker_count();
        let queue_cap = std::env::var("HELIX_INDEX_EXECUTOR_QUEUE_CAP")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(1024);
        let state = Arc::new((
            Mutex::new(IndexExecutorState::default()),
            std::sync::Condvar::new(),
        ));
        for id in 0..workers {
            let state = Arc::clone(&state);
            let _ = thread::Builder::new()
                .name(format!("helix-index-executor-{id}"))
                .spawn(move || loop {
                    let (key, job) = IndexExecutor::next_job(&state);
                    metrics::gauge!("helix_index_executor_active").increment(1.0);
                    run_index_job(job);
                    metrics::gauge!("helix_index_executor_active").decrement(1.0);
                    IndexExecutor::finish_job(&state, &key);
                });
        }
        Self { state, queue_cap }
    }

    fn next_job(
        state: &Arc<(Mutex<IndexExecutorState>, std::sync::Condvar)>,
    ) -> (String, IndexJob) {
        let (lock, cv) = &**state;
        let mut guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        let drain_floor = segment_breaker_drain_floor();
        let max_drain_workers = index_executor_max_drain_workers();
        loop {
            // Single pass: track the best job overall and the best *non-drain*
            // job, plus whether each is a breaker drain (indexed segments at/over
            // the breaker recovery floor).
            let mut best_any: Option<(usize, String, usize, bool)> = None;
            let mut best_nondrain: Option<(usize, String, usize)> = None;
            for (idx, key) in guard.queued_order.iter().enumerate() {
                let Some(job) = guard.queued_jobs.get(key) else {
                    continue;
                };
                let debt = index_job_debt(job);
                let is_drain = drain_floor.map_or(false, |floor| debt.indexed_segments >= floor);
                let score = optimizer_debt_priority_score(debt, drain_floor);
                if best_any
                    .as_ref()
                    .map(|(_, _, best_score, _)| score > *best_score)
                    .unwrap_or(true)
                {
                    best_any = Some((idx, key.clone(), score, is_drain));
                }
                if !is_drain
                    && best_nondrain
                        .as_ref()
                        .map(|(_, _, best_score)| score > *best_score)
                        .unwrap_or(true)
                {
                    best_nondrain = Some((idx, key.clone(), score));
                }
            }
            // Reservation: if the top job is a breaker drain and we're already at
            // the drain-worker cap, run the best freshness job instead so normal
            // indexing never fully starves. Fall back to the drain only when no
            // freshness work is queued (never idle a worker).
            let chosen = match best_any {
                Some((_, _, _, true))
                    if guard.active_drains.len() >= max_drain_workers
                        && best_nondrain.is_some() =>
                {
                    let (idx, key, score) = best_nondrain.unwrap();
                    (idx, key, score, false)
                }
                Some(best) => best,
                None => {
                    guard = cv.wait(guard).unwrap_or_else(|e| e.into_inner());
                    continue;
                }
            };
            {
                let (idx, key, score, is_drain) = chosen;
                guard.queued_order.remove(idx);
                if let Some(job) = guard.queued_jobs.remove(&key) {
                    let debt = index_job_debt(&job);
                    let backlog_ratio = index_job_backlog_ratio_millis(&job);
                    guard.active.insert(key.clone());
                    if is_drain {
                        guard.active_drains.insert(key.clone());
                    }
                    metrics::gauge!(
                        "helix_index_executor_selected_backlog_ratio",
                        "collection" => key.clone()
                    )
                    .set(backlog_ratio as f64 / 1_000.0);
                    metrics::gauge!(
                        "helix_index_executor_selected_segments",
                        "collection" => key.clone()
                    )
                    .set(debt.indexed_segments as f64);
                    metrics::gauge!(
                        "helix_index_executor_selected_active_segments",
                        "collection" => key.clone()
                    )
                    .set(debt.active_segments as f64);
                    metrics::gauge!(
                        "helix_index_executor_selected_gate_debt",
                        "collection" => key.clone()
                    )
                    .set(debt.gate_debt as f64);
                    metrics::gauge!(
                        "helix_index_executor_selected_merge_debt",
                        "collection" => key.clone()
                    )
                    .set(debt.merge_debt as f64);
                    metrics::gauge!(
                        "helix_index_executor_selected_dirty_retired_segments",
                        "collection" => key.clone()
                    )
                    .set(debt.dirty_retired_segments as f64);
                    metrics::gauge!(
                        "helix_index_executor_selected_priority_score",
                        "collection" => key.clone()
                    )
                    .set(score as f64);
                    metrics::gauge!("helix_index_executor_queue_depth")
                        .set(guard.queued_jobs.len() as f64);
                    metrics::gauge!("helix_index_executor_active_drains")
                        .set(guard.active_drains.len() as f64);
                    return (key, job);
                }
            }
            guard = cv.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn finish_job(state: &Arc<(Mutex<IndexExecutorState>, std::sync::Condvar)>, key: &str) {
        let (lock, cv) = &**state;
        let mut guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        guard.active.remove(key);
        guard.active_drains.remove(key);
        if guard.queued_jobs.contains_key(key) {
            guard.queued_order.push_back(key.to_string());
            cv.notify_one();
        }
        metrics::gauge!("helix_index_executor_queue_depth").set(guard.queued_jobs.len() as f64);
        metrics::gauge!("helix_index_executor_active_drains").set(guard.active_drains.len() as f64);
    }

    fn submit(&self, job: IndexJob) -> Result<(), IndexExecutorSubmitError> {
        let (lock, cv) = &*self.state;
        let mut guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        let key = job.collection.clone();

        if let Some(existing) = guard.queued_jobs.get_mut(&key) {
            existing.merge_from(job);
            metrics::counter!("helix_index_executor_coalesced_total").increment(1);
        } else if guard.active.contains(&key) {
            if guard.queued_jobs.len() >= self.queue_cap {
                return Err(IndexExecutorSubmitError);
            }
            guard.queued_jobs.insert(key, job);
            metrics::counter!("helix_index_executor_coalesced_total").increment(1);
        } else {
            if guard.queued_jobs.len() >= self.queue_cap {
                return Err(IndexExecutorSubmitError);
            }
            guard.queued_order.push_back(key.clone());
            guard.queued_jobs.insert(key, job);
            cv.notify_one();
        }

        metrics::gauge!("helix_index_executor_queue_depth").set(guard.queued_jobs.len() as f64);
        if self.queue_cap > 0 {
            metrics::gauge!("helix_index_executor_queue_capacity").set(self.queue_cap as f64);
        }
        Ok(())
    }
}

static INDEX_EXECUTOR: LazyLock<IndexExecutor> = LazyLock::new(IndexExecutor::new);

fn optimizer_backlog_sweep_ms() -> u64 {
    env_u64(
        "HELIX_OPTIMIZER_BACKLOG_SWEEP_MS",
        if cfg!(test) { 0 } else { 60_000 },
    )
}

fn optimizer_backlog_min_ratio() -> usize {
    env_usize_or("HELIX_OPTIMIZER_BACKLOG_MIN_RATIO", 2, 1)
}

fn optimizer_backlog_per_sweep() -> usize {
    env_usize_or("HELIX_OPTIMIZER_BACKLOG_PER_SWEEP", 8, 1)
}

fn optimizer_backlog_pressure_per_sweep() -> usize {
    env_usize_or("HELIX_OPTIMIZER_BACKLOG_PRESSURE_PER_SWEEP", 1, 0)
}

fn optimizer_backlog_effective_per_sweep(
    configured: usize,
    pressure_budget: usize,
    admission: MaintenanceAdmission,
) -> usize {
    if admission.is_admitted() {
        configured
    } else {
        configured.min(pressure_budget)
    }
}

fn optimizer_backlog_submit_cooldown_ms() -> u64 {
    env_u64("HELIX_OPTIMIZER_BACKLOG_SUBMIT_COOLDOWN_MS", 60_000)
}

fn optimizer_backlog_urgent_cooldown_ms() -> u64 {
    env_u64("HELIX_OPTIMIZER_BACKLOG_URGENT_COOLDOWN_MS", 5_000)
}

fn index_job_gate_debt_build_segments() -> usize {
    env_usize_or("HELIX_INDEX_JOB_GATE_DEBT_BUILD_SEGMENTS", 8, 1)
}

fn index_job_gate_debt_build_segments_max() -> usize {
    let base = index_job_gate_debt_build_segments();
    env_usize_or(
        "HELIX_INDEX_JOB_GATE_DEBT_BUILD_SEGMENTS_MAX",
        base.max(128),
        1,
    )
    .max(base)
}

fn index_job_build_flush_batch_segments() -> usize {
    env_usize_or("HELIX_INDEX_JOB_BUILD_FLUSH_BATCH_SEGMENTS", 16, 1)
}

fn gate_debt_build_segments_for(gate_debt: usize) -> usize {
    let base = index_job_gate_debt_build_segments();
    if gate_debt == 0 {
        return base;
    }
    base.max(gate_debt.min(index_job_gate_debt_build_segments_max()))
}

fn optimizer_backlog_cold_loads_per_sweep() -> usize {
    env_usize_or("HELIX_OPTIMIZER_BACKLOG_COLD_LOADS_PER_SWEEP", 2, 0)
}

fn optimizer_backlog_giant_cold_loads_per_sweep() -> usize {
    env_usize_or("HELIX_OPTIMIZER_BACKLOG_GIANT_COLD_LOADS_PER_SWEEP", 0, 0)
}

fn optimizer_backlog_cold_sidecar_max() -> usize {
    env_usize_or("HELIX_OPTIMIZER_BACKLOG_COLD_SIDECAR_MAX", 512, 0)
}

fn optimizer_backlog_cold_min_ratio() -> usize {
    env_usize_or("HELIX_OPTIMIZER_BACKLOG_COLD_MIN_RATIO", 8, 1)
}

fn optimizer_backlog_cold_cooldown_multiplier() -> u64 {
    env_u64("HELIX_OPTIMIZER_BACKLOG_COLD_COOLDOWN_MULT", 3).max(1)
}

struct OptimizerDebtCandidate {
    name: String,
    debt: DenseSegmentDebt,
    priority_score: usize,
}

impl OptimizerDebtCandidate {
    fn is_gate_blocked(&self) -> bool {
        self.debt.gate_debt > 0
    }
}

fn optimizer_debt_has_submit_work(debt: DenseSegmentDebt, min_segments: usize) -> bool {
    debt.indexed_segments > min_segments
        || (debt.gate_debt > 0
            && debt.active_segments >= NamedVectorManager::segment_creation_gate_cap())
}

fn optimizer_debt_can_expand_build_budget(debt: DenseSegmentDebt) -> bool {
    debt.gate_debt > 0 && debt.active_segments >= NamedVectorManager::segment_creation_gate_cap()
}

/// True when this node is an LSM read-only replica (`HELIX_LSM_ROLE=reader`).
/// Reader replicas serve reads from the writer's committed S3 state and reject
/// all writes; the HNSW build/merge optimizer is a WRITE path, so running it on a
/// reader just fails every cycle with "read-only LSM reader replica" (error spam
/// + wasted CPU). The optimizer must run on the writer only.
fn lsm_role_is_reader() -> bool {
    std::env::var("HELIX_LSM_ROLE")
        .map(|role| role.eq_ignore_ascii_case("reader"))
        .unwrap_or(false)
}

fn lsm_role_is_writer() -> bool {
    std::env::var("HELIX_LSM_ROLE")
        .map(|role| role.eq_ignore_ascii_case("writer"))
        .unwrap_or(false)
}

fn spawn_optimizer_backlog_sweeper(collections: Arc<CollectionManager>, config: Config) {
    if lsm_role_is_reader() {
        metrics::counter!("helix_optimizer_backlog_sweeper_total", "outcome" => "reader_disabled")
            .increment(1);
        return;
    }
    let sweep_ms = optimizer_backlog_sweep_ms();
    if sweep_ms == 0 {
        metrics::counter!("helix_optimizer_backlog_sweeper_total", "outcome" => "disabled")
            .increment(1);
        return;
    }

    let collections = Arc::downgrade(&collections);
    if let Err(err) = thread::Builder::new()
        .name("helix-optimizer-backlog".to_string())
        .spawn(move || {
            let sweep = Duration::from_millis(sweep_ms);
            let cooldown = Duration::from_millis(optimizer_backlog_submit_cooldown_ms());
            let mut last_submit: HashMap<String, Instant> = HashMap::new();
            loop {
                thread::sleep(sweep);
                let Some(collections) = collections.upgrade() else {
                    return;
                };
                let started = Instant::now();
                let target = NamedVectorManager::max_indexed_segments_target().max(1);
                let min_segments = target.saturating_mul(optimizer_backlog_min_ratio());
                let mut candidates: Vec<OptimizerDebtCandidate> = match collections.loaded_collection_names() {
                    Ok(names) => names
                        .into_iter()
                        .filter_map(|name| {
                            // Snapshot names only, then briefly clone one loaded
                            // storage Arc at a time. A full `loaded_collections_snapshot`
                            // clones every loaded Arc and can make the cache appear
                            // entirely pinned while this sweeper scans debt, starving
                            // foreground cold opens when loaded == max.
                            // Use the NON-touching peek: this background debt scan
                            // visits every loaded collection each sweep, so touching
                            // the idle timer here would perpetually reset it and
                            // starve idle-age eviction (writer RAM never bounds).
                            let storage = match collections.peek_loaded_collection(&name) {
                                Ok(Some(storage)) => storage,
                                Ok(None) => return None,
                                Err(err) => {
                                    metrics::counter!(
                                        "helix_optimizer_backlog_sweeper_total",
                                        "outcome" => "loaded_lookup_error"
                                    )
                                    .increment(1);
                                    tracing::debug!(
                                        collection = %name,
                                        error = %err,
                                        "optimizer backlog sweeper skipped loaded collection after lookup error"
                                    );
                                    return None;
                                }
                            };
                            let debt = storage.named_vectors.dense_segment_debt();
                            let priority_score = optimizer_debt_priority_score(debt, segment_breaker_drain_floor());
                            metrics::gauge!(
                                "helix_collection_indexed_dense_segments",
                                "collection" => name.clone()
                            )
                            .set(debt.indexed_segments as f64);
                            metrics::gauge!(
                                "helix_collection_active_dense_segments",
                                "collection" => name.clone()
                            )
                            .set(debt.active_segments as f64);
                            metrics::gauge!(
                                "helix_collection_dense_segment_gate_debt",
                                "collection" => name.clone()
                            )
                            .set(debt.gate_debt as f64);
                            metrics::gauge!(
                                "helix_collection_dense_segment_merge_debt",
                                "collection" => name.clone()
                            )
                            .set(debt.merge_debt as f64);
                            metrics::gauge!(
                                "helix_collection_dirty_retired_segments",
                                "collection" => name.clone()
                            )
                            .set(debt.dirty_retired_segments as f64);
                            metrics::gauge!(
                                "helix_collection_optimizer_priority_score",
                                "collection" => name.clone()
                            )
                            .set(priority_score as f64);
                            metrics::gauge!(
                                "helix_collection_indexed_dense_segment_target",
                                "collection" => name.clone()
                            )
                            .set(target as f64);
                            // Skip a collection that is mid-compaction: submitting
                            // an optimizer job would re-fetch and clone its
                            // storage Arc, racing the compaction eviction-drain
                            // (which needs strong_count==1). It loses nothing by
                            // waiting one sweep; the marker clears on compaction
                            // exit (RAII).
                            if optimizer_debt_has_submit_work(debt, min_segments)
                                && !storage.optimizer_running.load(Ordering::Acquire)
                                && !crate::helix_engine::storage_core::collection_manager::auto_compact_in_progress(&name)
                            {
                                Some(OptimizerDebtCandidate {
                                    name,
                                    debt,
                                    priority_score,
                                })
                            } else {
                                None
                            }
                        })
                        .collect(),
                    Err(err) => {
                        metrics::counter!(
                            "helix_optimizer_backlog_sweeper_total",
                            "outcome" => "name_snapshot_error"
                        )
                        .increment(1);
                        tracing::warn!(error = %err, "optimizer backlog sweeper name snapshot failed");
                        continue;
                    }
                };

                candidates.sort_by(|a, b| {
                    b.priority_score
                        .cmp(&a.priority_score)
                        .then_with(|| a.name.cmp(&b.name))
                });
                let process_admission = maintenance_process_resource_admission();
                let configured_per_sweep = optimizer_backlog_per_sweep();
                let warm_per_sweep = optimizer_backlog_effective_per_sweep(
                    configured_per_sweep,
                    optimizer_backlog_pressure_per_sweep(),
                    process_admission,
                );
                if !process_admission.is_admitted() {
                    metrics::counter!(
                        "helix_optimizer_backlog_sweeper_total",
                        "outcome" => "throttled",
                        "reason" => process_admission.reason()
                    )
                    .increment(1);
                    metrics::gauge!("helix_optimizer_backlog_sweeper_pressure_per_sweep")
                        .set(warm_per_sweep as f64);
                }

                let mut submitted = 0usize;
                for candidate in candidates.into_iter().take(warm_per_sweep) {
                    let gate_blocked = candidate.is_gate_blocked();
                    let priority_score = candidate.priority_score;
                    let name = candidate.name;
                    let debt = candidate.debt;
                    let effective_cooldown = if gate_blocked {
                        Duration::from_millis(optimizer_backlog_urgent_cooldown_ms())
                    } else {
                        cooldown
                    };
                    if last_submit
                        .get(&name)
                        .map(|last| last.elapsed() < effective_cooldown)
                        .unwrap_or(false)
                    {
                        metrics::counter!(
                            "helix_optimizer_backlog_sweeper_total",
                            "outcome" => "throttled",
                            "reason" => if gate_blocked { "gate_debt" } else { "backlog" }
                        )
                        .increment(1);
                        continue;
                    }
                    match submit_collection_optimizer(&collections, &config, &name) {
                        Ok(_) => {
                            submitted = submitted.saturating_add(1);
                            last_submit.insert(name.clone(), Instant::now());
                            metrics::counter!(
                                "helix_optimizer_backlog_sweeper_total",
                                "outcome" => "submitted"
                            )
                            .increment(1);
                            tracing::debug!(
                                collection = %name,
                                indexed_segments = debt.indexed_segments,
                                active_segments = debt.active_segments,
                                gate_debt = debt.gate_debt,
                                dirty_retired_segments = debt.dirty_retired_segments,
                                priority_score,
                                target,
                                "optimizer backlog sweeper submitted collection"
                            );
                        }
                        Err(err) => {
                            metrics::counter!(
                                "helix_optimizer_backlog_sweeper_total",
                                "outcome" => "submit_error"
                            )
                            .increment(1);
                            tracing::warn!(
                                collection = %name,
                                indexed_segments = debt.indexed_segments,
                                active_segments = debt.active_segments,
                                gate_debt = debt.gate_debt,
                                dirty_retired_segments = debt.dirty_retired_segments,
                                priority_score,
                                error = %err,
                                "optimizer backlog sweeper submit failed"
                            );
                        }
                    }
                }
                let cold_loads_per_sweep = if process_admission.is_admitted() {
                    optimizer_backlog_cold_loads_per_sweep()
                } else {
                    metrics::counter!(
                        "helix_optimizer_backlog_sweeper_total",
                        "outcome" => "cold_skipped",
                        "reason" => process_admission.reason()
                    )
                    .increment(1);
                    0
                };
                let mut cold_submitted = 0usize;
                if cold_loads_per_sweep > 0 {
                    let giant_cold_loads_per_sweep =
                        optimizer_backlog_giant_cold_loads_per_sweep();
                    let mut giant_cold_submitted = 0usize;
                    let cold_threshold = target.saturating_mul(optimizer_backlog_cold_min_ratio());
                    let cold_sidecar_max = optimizer_backlog_cold_sidecar_max();
                    match collections.cold_collections_with_sidecar_backlog(cold_threshold) {
                        Ok(mut cold) => {
                            cold.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                            let cold_cooldown = cooldown
                                .saturating_mul(optimizer_backlog_cold_cooldown_multiplier() as u32);
                            for (name, sidecar_count) in cold {
                                if cold_submitted >= cold_loads_per_sweep {
                                    break;
                                }
                                let giant_cold = cold_sidecar_max > 0
                                    && sidecar_count > cold_sidecar_max;
                                if giant_cold
                                    && giant_cold_submitted >= giant_cold_loads_per_sweep
                                {
                                    metrics::counter!(
                                        "helix_optimizer_backlog_sweeper_total",
                                        "outcome" => "cold_skipped",
                                        "reason" => "giant_quota"
                                    )
                                    .increment(1);
                                    tracing::warn!(
                                        collection = %name,
                                        sidecar_files = sidecar_count,
                                        cold_sidecar_max,
                                        giant_cold_loads_per_sweep,
                                        "optimizer backlog sweeper deferred giant cold tenant"
                                    );
                                    continue;
                                }
                                if last_submit
                                    .get(&name)
                                    .map(|last| last.elapsed() < cold_cooldown)
                                    .unwrap_or(false)
                                {
                                    metrics::counter!(
                                        "helix_optimizer_backlog_sweeper_total",
                                        "outcome" => "cold_throttled"
                                    )
                                    .increment(1);
                                    continue;
                                }
                                if giant_cold && giant_cold_loads_per_sweep == 0 {
                                    metrics::counter!(
                                        "helix_optimizer_backlog_sweeper_total",
                                        "outcome" => "cold_skipped",
                                        "reason" => "sidecar_backlog"
                                    )
                                    .increment(1);
                                    tracing::warn!(
                                        collection = %name,
                                        sidecar_files = sidecar_count,
                                        cold_sidecar_max,
                                        "optimizer backlog sweeper skipped cold tenant with oversized sidecar backlog"
                                    );
                                    continue;
                                }
                                if let MaintenanceAdmission::Rejected(reason) =
                                    collections.maintenance_cold_open_admission(&name)
                                {
                                    metrics::counter!(
                                        "helix_optimizer_backlog_sweeper_total",
                                        "outcome" => "cold_skipped",
                                        "reason" => reason
                                    )
                                    .increment(1);
                                    tracing::debug!(
                                        collection = %name,
                                        sidecar_files = sidecar_count,
                                        reason,
                                        "optimizer backlog sweeper skipped cold tenant under maintenance pressure"
                                    );
                                    continue;
                                }
                                match submit_collection_optimizer(
                                    &collections,
                                    &config,
                                    &name,
                                ) {
                                    Ok(_) => {
                                        cold_submitted = cold_submitted.saturating_add(1);
                                        if giant_cold {
                                            giant_cold_submitted =
                                                giant_cold_submitted.saturating_add(1);
                                        }
                                        last_submit.insert(name.clone(), Instant::now());
                                        metrics::counter!(
                                            "helix_optimizer_backlog_sweeper_total",
                                            "outcome" => if giant_cold {
                                                "giant_cold_submitted"
                                            } else {
                                                "cold_submitted"
                                            }
                                        )
                                        .increment(1);
                                        tracing::info!(
                                            collection = %name,
                                            sidecar_files = sidecar_count,
                                            cold_threshold,
                                            giant_cold,
                                            "optimizer backlog sweeper loaded cold tenant for merge"
                                        );
                                    }
                                    Err(err) => {
                                        metrics::counter!(
                                            "helix_optimizer_backlog_sweeper_total",
                                            "outcome" => "cold_submit_error"
                                        )
                                        .increment(1);
                                        tracing::warn!(
                                            collection = %name,
                                            sidecar_files = sidecar_count,
                                            error = %err,
                                            "optimizer backlog sweeper cold submit failed"
                                        );
                                    }
                                }
                            }
                        }
                        Err(err) => {
                            metrics::counter!(
                                "helix_optimizer_backlog_sweeper_total",
                                "outcome" => "cold_scan_error"
                            )
                            .increment(1);
                            tracing::warn!(
                                error = %err,
                                "optimizer backlog sweeper cold scan failed"
                            );
                        }
                    }
                }
                // Ratio-triggered auto-compaction. Runs at this txn-free
                // admission point (no LMDB txn or build permit held).
                //
                // CRITICAL: iterate NAMES only (loaded_collection_names), never
                // a snapshot that clones storage Arcs. Holding an Arc clone of
                // the collection being compacted pins it above strong_count==1
                // and the eviction-drain can never reach quiescence — the v2
                // smoke deadlock ("deferred ... no copy paid" forever on an idle
                // box). auto_compact_collection_if_needed re-fetches per name,
                // measures cheaply, marks the collection in-progress (so this
                // very loop and other internal Arc-takers skip it), and bails
                // before any heavy work unless every gate passes. At most one
                // collection is compacted per sweep (global permit also enforces
                // process-wide exclusivity).
                if let Ok(names) = collections.loaded_collection_names() {
                    for name in names {
                        match collections.auto_compact_collection_if_needed(&name) {
                            Ok(true) => {
                                // A compaction ran (or was attempted). Stop this
                                // sweep here so we never copy two large envs in
                                // one pass; the next sweep picks up the rest.
                                break;
                            }
                            Ok(false) => {}
                            Err(err) => {
                                tracing::warn!(
                                    collection = %name,
                                    error = %err,
                                    "auto-compaction check failed (non-fatal)"
                                );
                            }
                        }
                    }
                }

                metrics::histogram!("helix_optimizer_backlog_sweeper_duration_ms")
                    .record(started.elapsed().as_secs_f64() * 1000.0);
                metrics::gauge!("helix_optimizer_backlog_sweeper_submitted")
                    .set(submitted as f64);
                metrics::gauge!("helix_optimizer_backlog_sweeper_cold_submitted")
                    .set(cold_submitted as f64);
            }
        })
    {
        metrics::counter!("helix_optimizer_backlog_sweeper_total", "outcome" => "spawn_error")
            .increment(1);
        tracing::warn!(error = %err, "failed to spawn optimizer backlog sweeper");
    }
}

fn record_dense_merge_publish_phase(phase: &'static str, started: Instant) {
    metrics::histogram!("helix_dense_merge_publish_phase_ms", "phase" => phase)
        .record(started.elapsed().as_secs_f64() * 1000.0);
}

pub fn submit_collection_optimizer(
    collections: &Arc<CollectionManager>,
    config: &Config,
    collection: &str,
) -> Result<usize, GraphError> {
    let started = Instant::now();
    // Read-only replicas never run the optimizer (it writes; the writer owns it).
    // Guard here so post-open maintenance, the segment-breaker path, and any manual
    // submit all no-op on a reader instead of failing every cycle with the
    // "read-only LSM reader replica" write rejection.
    if lsm_role_is_reader() {
        metrics::counter!("helix_index_executor_submit_skipped_total", "reason" => "reader_replica")
            .increment(1);
        return Ok(0);
    }
    let resolved_collection = collections.resolve_alias(collection);
    let storage = collections.get_collection(&resolved_collection)?;
    if let Some(error) = storage.degraded_collection_error() {
        metrics::counter!("helix_index_executor_submit_skipped_total", "reason" => "collection_degraded")
            .increment(1);
        tracing::warn!(
            collection = %resolved_collection,
            error = %error,
            "skipping optimizer submit because collection is degraded"
        );
        return Ok(0);
    }
    let spaces_to_check: Vec<String> = storage
        .named_vectors
        .list_dense_vector_spaces()
        .into_keys()
        .collect();
    let submitted_spaces = spaces_to_check.len();
    if submitted_spaces == 0 {
        metrics::histogram!(
            "helix_index_executor_submit_duration_ms",
            "source" => "maintenance",
            "status" => "empty"
        )
        .record(started.elapsed().as_secs_f64() * 1000.0);
        return Ok(0);
    }

    let flat_scan_threshold = storage
        .named_vectors
        .effective_indexing_threshold(config.vector_flat_scan_threshold());
    let overrides = if storage.backend.kind() == BackendKind::Lsm {
        let r = storage
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))?;
        storage.get_hnsw_overrides_be(&r)?
    } else {
        let rtxn = storage.begin_resize_safe_read_txn()?;
        storage.get_hnsw_overrides(&rtxn)?
    };
    let hnsw_config = HNSWConfig::with_overrides(
        config.vector_config.m,
        config.vector_config.ef_construction,
        config.vector_config.ef_search,
        overrides.as_ref(),
    );

    if !optimizer_has_pending_work(
        &storage,
        &spaces_to_check,
        flat_scan_threshold,
        index_subthreshold_quiesced_tails(),
    ) {
        metrics::histogram!(
            "helix_index_executor_submit_duration_ms",
            "source" => "maintenance",
            "status" => "no_pending_work"
        )
        .record(started.elapsed().as_secs_f64() * 1000.0);
        metrics::counter!(
            "helix_index_executor_submit_skipped_total",
            "reason" => "no_pending_work"
        )
        .increment(1);
        tracing::debug!(
            collection = %resolved_collection,
            spaces = submitted_spaces,
            "skipping optimizer submit because stats show no pending vector work"
        );
        return Ok(0);
    }

    if let Err(e) = INDEX_EXECUTOR.submit(IndexJob {
        collection: resolved_collection,
        storage: Arc::downgrade(&storage),
        spaces_to_check,
        hnsw_config,
        flat_scan_threshold,
    }) {
        metrics::histogram!(
            "helix_index_executor_submit_duration_ms",
            "source" => "maintenance",
            "status" => "error"
        )
        .record(started.elapsed().as_secs_f64() * 1000.0);
        metrics::counter!("helix_index_executor_submit_rejected_total").increment(1);
        return Err(GraphError::New(format!(
            "failed to submit optimizer job: {}",
            e
        )));
    }
    metrics::histogram!(
        "helix_index_executor_submit_duration_ms",
        "source" => "maintenance",
        "status" => "ok"
    )
    .record(started.elapsed().as_secs_f64() * 1000.0);
    metrics::counter!("helix_index_executor_manual_submissions_total")
        .increment(submitted_spaces as u64);
    Ok(submitted_spaces)
}

pub(crate) fn submit_post_open_vector_maintenance(
    storage: &Arc<HelixGraphStorage>,
    config: &Config,
    collection: &str,
) -> Result<usize, GraphError> {
    // Read-only replicas open collections to serve reads; they must NOT submit
    // the (write-path) optimizer job here, or every collection opened on a reader
    // logs "background optimizer failed: read-only LSM reader replica" each cycle.
    // The writer owns all optimizer work. (This path submits an IndexJob directly,
    // so it needs its own guard separate from submit_collection_optimizer.)
    if lsm_role_is_reader() {
        metrics::counter!(
            "helix_post_open_vector_maintenance_total",
            "outcome" => "reader_replica"
        )
        .increment(1);
        return Ok(0);
    }
    if let Some(error) = storage.degraded_collection_error() {
        metrics::counter!(
            "helix_post_open_vector_maintenance_total",
            "outcome" => "collection_degraded"
        )
        .increment(1);
        tracing::warn!(
            collection,
            error = %error,
            "skipping post-open vector maintenance because collection is degraded"
        );
        return Ok(0);
    }
    let spaces_to_check: Vec<String> = storage
        .named_vectors
        .list_dense_vector_spaces()
        .into_keys()
        .collect();
    let submitted_spaces = spaces_to_check.len();
    if submitted_spaces == 0 {
        metrics::counter!(
            "helix_post_open_vector_maintenance_total",
            "outcome" => "empty"
        )
        .increment(1);
        return Ok(0);
    }

    let flat_scan_threshold = storage
        .named_vectors
        .effective_indexing_threshold(config.vector_flat_scan_threshold());
    let overrides = if storage.backend.kind() == BackendKind::Lsm {
        let r = storage
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))?;
        storage.get_hnsw_overrides_be(&r)?
    } else {
        let rtxn = storage.begin_resize_safe_read_txn()?;
        storage.get_hnsw_overrides(&rtxn)?
    };
    let hnsw_config = HNSWConfig::with_overrides(
        config.vector_config.m,
        config.vector_config.ef_construction,
        config.vector_config.ef_search,
        overrides.as_ref(),
    );

    if let Err(err) = INDEX_EXECUTOR.submit(IndexJob {
        collection: collection.to_string(),
        storage: Arc::downgrade(storage),
        spaces_to_check,
        hnsw_config,
        flat_scan_threshold,
    }) {
        metrics::counter!(
            "helix_post_open_vector_maintenance_total",
            "outcome" => "queue_full"
        )
        .increment(1);
        tracing::warn!(
            collection = collection,
            error = %err,
            "Post-open vector maintenance queue is full; loaded collection will be retried by optimizer backlog sweeper"
        );
        return Ok(0);
    }

    metrics::counter!(
        "helix_post_open_vector_maintenance_total",
        "outcome" => "submitted"
    )
    .increment(submitted_spaces as u64);
    Ok(submitted_spaces)
}

// ─── Segment Reaper ────────────────────────────────────────────────────────
//
// Background drainer for retired/empty dense vector segments. Phase E of
// `run_chunked_merge_publish` (drop_retired_segment) and every call site
// of `cleanup_empty_dense_segments` previously held the per-collection
// LMDB write gate for tens to hundreds of seconds while `core.clear(txn)`
// walked five LMDB B-trees per segment. Track-caller diagnostics confirmed
// these sites were responsible for 30-263 s `with_exclusive_write_txn`
// holds in production.
//
// Once a segment is removed from `space.segments` (via the metadata-swap
// txn that publishes the merge target or deletes the empty entry), no
// reader can ever resolve it again. Clearing its LMDB pages is pure disk
// reclamation and can run in any later txn. Background workers open a fresh
// `with_write_txn` per database step so each clear competes only with normal
// writes — never with the merge publish that produced the work.

const REAPER_DB_COUNT: usize = VectorCore::REAPER_DB_COUNT;

struct ReaperJob {
    key: String,
    storage: Arc<HelixGraphStorage>,
    physical_name: String,
    priority_score: usize,
    gate_debt: usize,
    dirty_retired_segments: usize,
    reason: &'static str,
    attempts: usize,
    next_db_index: usize,
    drained_dbs: [bool; REAPER_DB_COUNT],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReaperJobOutcome {
    Succeeded,
    Yielded,
    Failed,
}

#[derive(Default)]
struct ReaperQueueState {
    queued_jobs: Vec<ReaperJob>,
}

struct SegmentReaper {
    state: Arc<(Mutex<ReaperQueueState>, std::sync::Condvar)>,
    queue_depth: Arc<std::sync::atomic::AtomicUsize>,
    queued_or_running: Arc<Mutex<HashSet<String>>>,
}

impl SegmentReaper {
    fn new() -> Self {
        let state = Arc::new((
            Mutex::new(ReaperQueueState::default()),
            std::sync::Condvar::new(),
        ));
        let queue_depth = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let queued_or_running = Arc::new(Mutex::new(HashSet::new()));
        let worker_count = segment_reaper_threads();
        for worker_id in 0..worker_count {
            let state_for_thread = Arc::clone(&state);
            let queue_depth_for_thread = Arc::clone(&queue_depth);
            let queued_for_thread = Arc::clone(&queued_or_running);
            thread::Builder::new()
                .name(format!("helix-segment-reaper-{worker_id}"))
                .spawn(move || {
                    Self::run_worker(state_for_thread, queue_depth_for_thread, queued_for_thread)
                })
                .expect("spawn helix-segment-reaper thread");
        }
        metrics::gauge!("helix_segment_reaper_workers").set(worker_count as f64);
        Self {
            state,
            queue_depth,
            queued_or_running,
        }
    }

    fn run_worker(
        state: Arc<(Mutex<ReaperQueueState>, std::sync::Condvar)>,
        queue_depth: Arc<std::sync::atomic::AtomicUsize>,
        queued_or_running: Arc<Mutex<HashSet<String>>>,
    ) {
        loop {
            let mut job = Self::take_next_job(&state, &queue_depth);
            metrics::counter!(
                "helix_segment_reaper_turns_total",
                "reason" => job.reason
            )
            .increment(1);
            if is_fresh_reaper_job(&job) {
                metrics::counter!(
                    "helix_segment_reaper_started_total",
                    "reason" => job.reason
                )
                .increment(1);
            }
            let selected_key = reaper_job_selection_key(&job);
            metrics::gauge!("helix_segment_reaper_selected_priority_score")
                .set(selected_key.priority_score as f64);
            metrics::gauge!("helix_segment_reaper_selected_gate_debt")
                .set(selected_key.gate_debt as f64);
            metrics::gauge!("helix_segment_reaper_selected_dirty_retired_segments")
                .set(selected_key.dirty_retired_segments as f64);
            let started = Instant::now();

            let yield_ms = reaper_yield_ms();
            let scale = reaper_chunk_scale();
            let outcome = Self::run_job_turn(&mut job, yield_ms, scale);

            let elapsed_ms = started.elapsed().as_millis() as f64;
            metrics::histogram!("helix_segment_reaper_clear_ms").record(elapsed_ms);
            match outcome {
                ReaperJobOutcome::Succeeded => {
                    metrics::counter!("helix_segment_reaper_succeeded_total").increment(1);
                    Self::finish_or_retry_job(&state, &queue_depth, &queued_or_running, job, true);
                }
                ReaperJobOutcome::Yielded => {
                    Self::requeue_yielded_job(&state, &queue_depth, job);
                }
                ReaperJobOutcome::Failed => {
                    Self::finish_or_retry_job(&state, &queue_depth, &queued_or_running, job, false);
                }
            }
        }
    }

    fn run_job_turn(job: &mut ReaperJob, yield_ms: u64, scale: f32) -> ReaperJobOutcome {
        let Some(db_index) = next_reaper_db_index(job) else {
            return Self::finalize_reaper_job(job);
        };
        let (base_cap, label) = REAPER_DB_CAPS[db_index];
        let cap = scale_cap(base_cap, scale);
        metrics::gauge!("helix_segment_reaper_selected_db_index").set(db_index as f64);
        let db_started = Instant::now();
        let result = if job.storage.backend.kind() == BackendKind::Lsm {
            job.storage.with_write_backend(|w| {
                job.storage
                    .named_vectors
                    .drop_retired_segment_db_step_be(w, &job.physical_name, db_index, cap)
                    .map_err(GraphError::from)
            })
        } else {
            job.storage.with_write_txn(|txn| {
                job.storage
                    .named_vectors
                    .drop_retired_segment_db_step(
                        job.storage.lmdb_env()?,
                        txn,
                        &job.physical_name,
                        db_index,
                        cap,
                    )
                    .map_err(GraphError::from)
            })
        };
        metrics::histogram!(
            "helix_segment_reaper_db_clear_ms",
            "db" => label,
        )
        .record(db_started.elapsed().as_millis() as f64);
        metrics::counter!(
            "helix_segment_reaper_db_steps_total",
            "db" => label,
        )
        .increment(1);
        metrics::counter!("helix_segment_reaper_chunks_total").increment(1);
        match result {
            Ok(drained) => {
                mark_reaper_db_step(job, db_index, drained);
                if all_reaper_dbs_drained(job) {
                    Self::finalize_reaper_job(job)
                } else {
                    thread::sleep(Duration::from_millis(yield_ms));
                    ReaperJobOutcome::Yielded
                }
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    segment = %job.physical_name,
                    db = label,
                    "segment reaper db step failed; aborting this job"
                );
                metrics::counter!("helix_segment_reaper_failed_total").increment(1);
                ReaperJobOutcome::Failed
            }
        }
    }

    fn finalize_reaper_job(job: &ReaperJob) -> ReaperJobOutcome {
        match job
            .storage
            .named_vectors
            .finalize_drained_retired_segment(&job.physical_name)
        {
            Ok(()) => {
                let sidecars = job.storage.named_vectors.unlink_sidecar_files_for_segments(
                    job.storage.path(),
                    std::slice::from_ref(&job.physical_name),
                );
                if sidecars.error_count > 0 {
                    tracing::warn!(
                        errors = sidecars.error_count,
                        segment = %job.physical_name,
                        "segment reaper sidecar unlink had errors"
                    );
                }
                // Re-upsert tombstones: this is the universal retire chokepoint
                // (merge Phase E, segment breaker, collection delete), so clear
                // the retired segment's tombstones in one place. Best-effort:
                // leftover keys are harmless and reconstructed/ignored.
                let cleared = if job.storage.backend.kind() == BackendKind::Lsm {
                    job.storage
                        .backend
                        .begin_read()
                        .map_err(|e| GraphError::New(e.to_string()))
                        .and_then(|r| {
                            job.storage.with_write_backend(|w| {
                                job.storage
                                    .named_vectors
                                    .clear_tombstones_for_segment_be(&r, w, &job.physical_name)
                                    .map_err(GraphError::from)
                            })
                        })
                } else {
                    job.storage.with_write_txn(|txn| {
                        job.storage
                            .named_vectors
                            .clear_tombstones_for_segment(txn, &job.physical_name)
                            .map_err(GraphError::from)
                    })
                };
                match cleared {
                    Ok(n) if n > 0 => {
                        metrics::counter!("helix_reupsert_tombstones_cleared_on_retire_total")
                            .increment(n as u64);
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::debug!(
                            error = %e,
                            segment = %job.physical_name,
                            "tombstone clear on retire failed (non-fatal)"
                        );
                    }
                }
                ReaperJobOutcome::Succeeded
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    segment = %job.physical_name,
                    "segment reaper finalize failed"
                );
                metrics::counter!("helix_segment_reaper_failed_total").increment(1);
                ReaperJobOutcome::Failed
            }
        }
    }

    fn take_next_job(
        state: &Arc<(Mutex<ReaperQueueState>, std::sync::Condvar)>,
        queue_depth: &Arc<std::sync::atomic::AtomicUsize>,
    ) -> ReaperJob {
        let (lock, cvar) = &**state;
        let mut guard = match lock.lock() {
            Ok(guard) => guard,
            Err(err) => {
                tracing::warn!(error = %err, "segment reaper queue lock poisoned");
                err.into_inner()
            }
        };
        while guard.queued_jobs.is_empty() {
            guard = match cvar.wait(guard) {
                Ok(guard) => guard,
                Err(err) => {
                    tracing::warn!(error = %err, "segment reaper queue wait lock poisoned");
                    err.into_inner()
                }
            };
        }
        let best_idx = select_reaper_job_index(&guard.queued_jobs).unwrap_or(0);
        let job = guard.queued_jobs.remove(best_idx);
        record_segment_reaper_queue_depth(queue_depth, guard.queued_jobs.len());
        job
    }

    fn submit(&self, storage: Arc<HelixGraphStorage>, physical_name: String) {
        let debt = storage.named_vectors.dense_segment_debt();
        let priority_score = reaper_debt_priority_score(debt, segment_breaker_drain_floor());
        self.submit_with_priority(
            storage,
            physical_name,
            priority_score,
            debt.gate_debt,
            debt.dirty_retired_segments,
            "retired",
        );
    }

    fn submit_dirty_discovered(&self, storage: Arc<HelixGraphStorage>, physical_name: String) {
        let debt = DenseSegmentDebt {
            indexed_segments: 0,
            active_segments: 0,
            merge_debt: 0,
            gate_debt: 1,
            dirty_retired_segments: 1,
        };
        self.submit_with_priority(
            storage,
            physical_name,
            reaper_debt_priority_score(debt, segment_breaker_drain_floor()),
            debt.gate_debt,
            debt.dirty_retired_segments,
            "dirty_discovered",
        );
    }

    fn submit_with_priority(
        &self,
        storage: Arc<HelixGraphStorage>,
        physical_name: String,
        priority_score: usize,
        gate_debt: usize,
        dirty_retired_segments: usize,
        reason: &'static str,
    ) {
        let key = reaper_job_key(&storage, &physical_name);
        {
            let mut queued = match self.queued_or_running.lock() {
                Ok(queued) => queued,
                Err(err) => {
                    tracing::warn!(error = %err, "segment reaper dedupe lock poisoned");
                    return;
                }
            };
            if !queued.insert(key.clone()) {
                metrics::counter!(
                    "helix_segment_reaper_deduped_total",
                    "reason" => reason
                )
                .increment(1);
                return;
            }
        }

        metrics::gauge!("helix_segment_reaper_submitted_priority_score").set(priority_score as f64);
        metrics::counter!(
            "helix_segment_reaper_submitted_total",
            "reason" => reason
        )
        .increment(1);
        enqueue_reaper_job(
            &self.state,
            &self.queue_depth,
            ReaperJob {
                key,
                storage,
                physical_name,
                priority_score,
                gate_debt,
                dirty_retired_segments,
                reason,
                attempts: 0,
                next_db_index: 0,
                drained_dbs: [false; REAPER_DB_COUNT],
            },
        );
    }

    fn requeue_yielded_job(
        state: &Arc<(Mutex<ReaperQueueState>, std::sync::Condvar)>,
        queue_depth: &Arc<std::sync::atomic::AtomicUsize>,
        job: ReaperJob,
    ) {
        metrics::counter!(
            "helix_segment_reaper_yielded_total",
            "reason" => job.reason
        )
        .increment(1);
        enqueue_reaper_job(state, queue_depth, job);
    }

    fn finish_or_retry_job(
        state: &Arc<(Mutex<ReaperQueueState>, std::sync::Condvar)>,
        queue_depth: &Arc<std::sync::atomic::AtomicUsize>,
        queued_or_running: &Arc<Mutex<HashSet<String>>>,
        mut job: ReaperJob,
        succeeded: bool,
    ) {
        if succeeded {
            if let Ok(mut queued) = queued_or_running.lock() {
                queued.remove(&job.key);
            }
            return;
        }

        if job.attempts >= segment_reaper_retry_attempts() {
            if let Ok(mut queued) = queued_or_running.lock() {
                queued.remove(&job.key);
            }
            metrics::counter!(
                "helix_segment_reaper_abandoned_total",
                "reason" => job.reason
            )
            .increment(1);
            return;
        }

        job.attempts = job.attempts.saturating_add(1);
        metrics::counter!(
            "helix_segment_reaper_requeued_total",
            "reason" => job.reason
        )
        .increment(1);
        enqueue_reaper_job(state, queue_depth, job);
    }
}

static SEGMENT_REAPER: LazyLock<SegmentReaper> = LazyLock::new(SegmentReaper::new);

fn enqueue_reaper_job(
    state: &Arc<(Mutex<ReaperQueueState>, std::sync::Condvar)>,
    queue_depth: &Arc<std::sync::atomic::AtomicUsize>,
    job: ReaperJob,
) {
    let (lock, cvar) = &**state;
    let mut guard = match lock.lock() {
        Ok(guard) => guard,
        Err(err) => {
            tracing::warn!(error = %err, "segment reaper queue lock poisoned on enqueue");
            err.into_inner()
        }
    };
    guard.queued_jobs.push(job);
    record_segment_reaper_queue_depth(queue_depth, guard.queued_jobs.len());
    cvar.notify_all();
}

fn record_segment_reaper_queue_depth(
    queue_depth: &Arc<std::sync::atomic::AtomicUsize>,
    depth: usize,
) {
    queue_depth.store(depth, Ordering::Relaxed);
    metrics::gauge!("helix_segment_reaper_queue_depth").set(depth as f64);
    metrics::gauge!("helix_segment_reaper_queue_actual_depth").set(depth as f64);
}

fn select_reaper_job_index(jobs: &[ReaperJob]) -> Option<usize> {
    let mut best: Option<(usize, ReaperJobSelectionKey)> = None;
    for (idx, job) in jobs.iter().enumerate() {
        let key = reaper_job_selection_key(job);
        if best.map(|(_, best_key)| key > best_key).unwrap_or(true) {
            best = Some((idx, key));
        }
    }
    best.map(|(idx, _)| idx)
}

fn is_fresh_reaper_job(job: &ReaperJob) -> bool {
    job.attempts == 0 && job.next_db_index == 0 && job.drained_dbs.iter().all(|drained| !*drained)
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ReaperJobSelectionKey {
    priority_score: usize,
    dirty_retired_segments: usize,
    gate_debt: usize,
}

fn reaper_job_selection_key(job: &ReaperJob) -> ReaperJobSelectionKey {
    ReaperJobSelectionKey {
        priority_score: job.priority_score,
        dirty_retired_segments: job.dirty_retired_segments,
        gate_debt: job.gate_debt,
    }
}

fn next_reaper_db_index(job: &ReaperJob) -> Option<usize> {
    for offset in 0..REAPER_DB_COUNT {
        let db_index = (job.next_db_index + offset) % REAPER_DB_COUNT;
        if !job.drained_dbs[db_index] {
            return Some(db_index);
        }
    }
    None
}

fn mark_reaper_db_step(job: &mut ReaperJob, db_index: usize, drained: bool) {
    if drained {
        job.drained_dbs[db_index] = true;
    }
    job.next_db_index = (db_index + 1) % REAPER_DB_COUNT;
}

fn all_reaper_dbs_drained(job: &ReaperJob) -> bool {
    job.drained_dbs.iter().all(|drained| *drained)
}

fn reaper_job_key(storage: &HelixGraphStorage, physical_name: &str) -> String {
    format!("{}::{physical_name}", storage.path().display())
}

pub(crate) fn attach_dirty_retired_reaper_hook(storage: &Arc<HelixGraphStorage>) {
    let storage_weak = Arc::downgrade(storage);
    storage
        .named_vectors
        .attach_dirty_retired_segment_hook(move |physical_name| {
            let Some(storage) = storage_weak.upgrade() else {
                metrics::counter!(
                    "helix_segment_dirty_retired_enqueue_total",
                    "outcome" => "storage_gone"
                )
                .increment(1);
                return;
            };
            SEGMENT_REAPER.submit_dirty_discovered(storage, physical_name);
            metrics::counter!(
                "helix_segment_dirty_retired_enqueue_total",
                "outcome" => "submitted"
            )
            .increment(1);
        });
}

/// Chunk size (vectors per exclusive write txn) for chunked merge publish.
/// Tunable via `HELIX_MERGE_CHUNK_SIZE`. Default 8192 keeps each phase B/C
/// txn under ~250 ms even on 1536-dim vectors.
fn merge_chunk_size() -> usize {
    std::env::var("HELIX_MERGE_CHUNK_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v >= 64)
        .unwrap_or(8192)
}

const MERGE_FLAT_BYTE_BUDGET: usize = 2 * 1024 * 1024;
const MERGE_FLAT_ROW_CAP: usize = 512;
const MERGE_INDEX_BYTE_BUDGET: usize = 2 * 1024 * 1024;
const MERGE_INDEX_HNSW_EDGE_BUDGET: usize = 4 * 1024;
const MERGE_CHUNK_WARN_MS: u128 = 5_000;

fn merge_flat_byte_budget() -> usize {
    env_usize_or(
        "HELIX_MERGE_FLAT_BYTE_BUDGET",
        MERGE_FLAT_BYTE_BUDGET,
        64 * 1024,
    )
}

fn merge_flat_row_cap() -> usize {
    env_usize_or("HELIX_MERGE_FLAT_ROW_CAP", MERGE_FLAT_ROW_CAP, 1)
}

fn merge_index_byte_budget() -> usize {
    env_usize_or(
        "HELIX_MERGE_INDEX_BYTE_BUDGET",
        MERGE_INDEX_BYTE_BUDGET,
        64 * 1024,
    )
}

fn merge_index_hnsw_edge_budget() -> usize {
    env_usize_or(
        "HELIX_MERGE_INDEX_HNSW_EDGE_BUDGET",
        MERGE_INDEX_HNSW_EDGE_BUDGET,
        128,
    )
}

#[derive(Clone, Copy, Debug, Default)]
struct MergeChunkStats {
    estimated_bytes: usize,
    hnsw_edges: usize,
}

impl MergeChunkStats {
    fn saturating_add(self, other: Self) -> Self {
        Self {
            estimated_bytes: self.estimated_bytes.saturating_add(other.estimated_bytes),
            hnsw_edges: self.hnsw_edges.saturating_add(other.hnsw_edges),
        }
    }
}

fn value_estimated_bytes(value: &Value) -> usize {
    match value {
        Value::String(s) => s.len(),
        Value::F32(_) | Value::I32(_) | Value::U32(_) => 4,
        Value::F64(_) | Value::I64(_) | Value::U64(_) => 8,
        Value::I8(_) | Value::U8(_) | Value::Boolean(_) => 1,
        Value::I16(_) | Value::U16(_) => 2,
        Value::U128(_) => 16,
        Value::Array(values) => values.iter().map(value_estimated_bytes).sum(),
        Value::Object(map) => map
            .iter()
            .map(|(k, v)| k.len().saturating_add(value_estimated_bytes(v)))
            .sum(),
        Value::Empty => 0,
    }
}

fn fields_estimated_bytes(fields: &HashMap<String, Value>) -> usize {
    fields
        .iter()
        .map(|(k, v)| k.len().saturating_add(value_estimated_bytes(v)))
        .sum()
}

#[cfg(test)]
fn flat_merge_item_stats(item: &(u128, Vec<f32>, HashMap<String, Value>)) -> MergeChunkStats {
    MergeChunkStats {
        estimated_bytes: item
            .1
            .len()
            .saturating_mul(std::mem::size_of::<f32>())
            .saturating_add(fields_estimated_bytes(&item.2))
            .saturating_add(64),
        hnsw_edges: 0,
    }
}

#[cfg(test)]
fn budgeted_merge_flat_chunk_len(
    items: &[(u128, Vec<f32>, HashMap<String, Value>)],
    max_items: usize,
) -> (usize, MergeChunkStats) {
    if items.is_empty() {
        return (0, MergeChunkStats::default());
    }
    // Flat publish cost is not purely byte-proportional: payload index puts,
    // LMDB B-tree page splits, and sidecar appends all add per-row work. Keep
    // an independent row cap so low-byte _edge merges cannot pin the writer for
    // thousands of rows in one transaction.
    let max_items = max_items.max(1).min(merge_flat_row_cap()).min(items.len());
    let mut len = 0usize;
    let mut stats = MergeChunkStats::default();
    for item in items.iter().take(max_items) {
        let candidate = stats.saturating_add(flat_merge_item_stats(item));
        if len > 0 && candidate.estimated_bytes > merge_flat_byte_budget() {
            break;
        }
        len += 1;
        stats = candidate;
    }
    (len.max(1), stats)
}

fn prepared_flat_item_stats(
    prepared: &crate::helix_engine::vector_core::vector_core::PreparedIndex,
    index: usize,
) -> MergeChunkStats {
    let fields_bytes = prepared
        .point_fields
        .get(index)
        .map(fields_estimated_bytes)
        .unwrap_or(0);
    MergeChunkStats {
        estimated_bytes: prepared
            .raw_len_at(index)
            .saturating_mul(std::mem::size_of::<f32>())
            .saturating_add(fields_bytes)
            .saturating_add(64),
        hnsw_edges: 0,
    }
}

fn budgeted_prepared_flat_chunk_len(
    prepared: &crate::helix_engine::vector_core::vector_core::PreparedIndex,
    start: usize,
    max_items: usize,
) -> (usize, MergeChunkStats) {
    let total = prepared.point_ids.len().saturating_sub(start);
    if total == 0 {
        return (0, MergeChunkStats::default());
    }
    let max_items = max_items.max(1).min(merge_flat_row_cap()).min(total);
    let mut len = 0usize;
    let mut stats = MergeChunkStats::default();
    for offset in 0..max_items {
        let candidate = stats.saturating_add(prepared_flat_item_stats(prepared, start + offset));
        if len > 0 && candidate.estimated_bytes > merge_flat_byte_budget() {
            break;
        }
        len += 1;
        stats = candidate;
    }
    (len.max(1), stats)
}

fn prepared_index_item_stats(
    prepared: &crate::helix_engine::vector_core::vector_core::PreparedIndex,
    index: usize,
) -> MergeChunkStats {
    let raw_bytes = prepared
        .raw_len_at(index)
        .saturating_mul(std::mem::size_of::<f32>());
    let level = prepared.levels.get(index).copied().unwrap_or(0);
    let vector_bytes = raw_bytes.saturating_mul(if level > 0 { 2 } else { 1 });
    let neighbor_refs = prepared.neighbor_ref_count_at(index);
    let level_count = prepared.adjacency_level_count_at(index);
    MergeChunkStats {
        estimated_bytes: vector_bytes
            .saturating_add(neighbor_refs.saturating_mul(std::mem::size_of::<u32>()))
            .saturating_add(level_count.saturating_mul(16))
            .saturating_add(64),
        hnsw_edges: neighbor_refs,
    }
}

fn budgeted_merge_index_chunk_len(
    prepared: &crate::helix_engine::vector_core::vector_core::PreparedIndex,
    start: usize,
    max_items: usize,
) -> (usize, MergeChunkStats) {
    let total = prepared.point_ids.len().saturating_sub(start);
    if total == 0 {
        return (0, MergeChunkStats::default());
    }
    let max_items = max_items.max(1).min(total);
    let mut len = 0usize;
    let mut stats = MergeChunkStats::default();
    for offset in 0..max_items {
        let candidate = stats.saturating_add(prepared_index_item_stats(prepared, start + offset));
        if len > 0
            && (candidate.estimated_bytes > merge_index_byte_budget()
                || candidate.hnsw_edges > merge_index_hnsw_edge_budget())
        {
            break;
        }
        len += 1;
        stats = candidate;
    }
    (len.max(1), stats)
}

fn record_dense_merge_publish_chunk(
    phase: &'static str,
    vec_name: &str,
    merged_name: &str,
    chunk_len: usize,
    stats: MergeChunkStats,
    elapsed: Duration,
) {
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    metrics::histogram!("helix_dense_merge_publish_chunk_ms", "phase" => phase).record(elapsed_ms);
    metrics::histogram!("helix_dense_merge_publish_chunk_len", "phase" => phase)
        .record(chunk_len as f64);
    metrics::histogram!("helix_dense_merge_publish_chunk_estimated_bytes", "phase" => phase)
        .record(stats.estimated_bytes as f64);
    metrics::histogram!("helix_dense_merge_publish_chunk_hnsw_edges", "phase" => phase)
        .record(stats.hnsw_edges as f64);
    if elapsed.as_millis() >= MERGE_CHUNK_WARN_MS {
        tracing::warn!(
            phase,
            vec_name,
            merged_name,
            chunk_len,
            estimated_bytes = stats.estimated_bytes,
            hnsw_edges = stats.hnsw_edges,
            elapsed_ms,
            "dense merge publish chunk slow"
        );
    }
}

/// Records to delete per database per write txn during segment reaping.
/// Legacy single-txn knob. Retained for back-compat with the old
/// `drop_retired_segment_chunk` API (still used by tests). The live
/// reaper path uses `REAPER_DB_CAPS` + `reaper_chunk_scale` instead so
/// each LMDB database has its own per-row-cost-aware cap.
/// Override via `HELIX_REAPER_CHUNK_SIZE`.
#[allow(dead_code)]
fn reaper_chunk_size() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HELIX_REAPER_CHUNK_SIZE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 64)
            .unwrap_or(1_000)
    })
}

/// Per-database deletion caps for the segment reaper, sized by row width.
/// Index matches `VectorCore::clear_chunk_db` / `drop_retired_segment_db_step`:
///   0 vectors_db        — small key/value rows
///   1 ordinals_db       — small key/value rows
///   2 vector_data_db    — per-row vector blob (medium)
///   3 out_edges_db      — small dupsort rows
///   4 neighbor_lists_db — largest values (M·u32 per level)
///   5 ivf_centroids_db  — ≤2 blobs (centroid table + meta)
///   6 ivf_postings_db   — one id blob per centroid (≤4096 rows)
///   7 hnsw_simhash_db   — 16-byte SimHash row per vector
///
/// All caps are scaled uniformly by `reaper_chunk_scale()`, so an
/// operator can shrink the worst-case writer hold by setting a single
/// env var without losing the relative sizing between DBs.
const REAPER_DB_CAPS: [(usize, &str); REAPER_DB_COUNT] = [
    (4096, "vectors"),
    (4096, "ordinals"),
    (512, "vector_data"),
    (4096, "out_edges"),
    (128, "neighbor_lists"),
    (64, "ivf_centroids"),
    (512, "ivf_postings"),
    (4096, "hnsw_simhash"),
];

/// Uniform scale factor applied to every entry in `REAPER_DB_CAPS`.
/// Default 1.0 (use the row-width-tuned caps verbatim). Override via
/// `HELIX_REAPER_CHUNK_SCALE` — e.g. `0.5` halves every cap when the
/// per-row cost on a workload is higher than expected.
fn reaper_chunk_scale() -> f32 {
    static CACHED: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HELIX_REAPER_CHUNK_SCALE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(1.0)
    })
}

#[inline]
fn scale_cap(base: usize, scale: f32) -> usize {
    let scaled = (base as f32 * scale).round() as i64;
    scaled.max(1) as usize
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DenseSegmentFormat {
    Legacy,
    Hvs8,
    TurboQuant,
}

fn parse_dense_segment_format(value: Option<&str>) -> DenseSegmentFormat {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("hvs8") => DenseSegmentFormat::Hvs8,
        Some("tq" | "hvtq" | "turbo_prod" | "turboquant" | "turbo_quant") => {
            DenseSegmentFormat::TurboQuant
        }
        _ => DenseSegmentFormat::Legacy,
    }
}

/// Segment sidecar format selected by `HELIX_SEGMENT_FORMAT`.
///
/// `hvs8` preserves the existing scalar-quantized publish path. `tq` makes
/// compact TurboProd merge targets publish a single HVTQ payload sidecar; older
/// non-TurboProd compact collections still fall back to HVS8 so they do not
/// lose their existing sidecar behavior when the deployment default changes.
fn dense_segment_format() -> DenseSegmentFormat {
    parse_dense_segment_format(std::env::var("HELIX_SEGMENT_FORMAT").ok().as_deref())
}

fn externalized_marker_repair_limit_per_segment() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| env_usize_or("HELIX_EXTERNALIZED_MARKER_REPAIR_LIMIT", 256, 0))
}

/// Override via `HELIX_REAPER_YIELD_MS`.
fn reaper_yield_ms() -> u64 {
    static CACHED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HELIX_REAPER_YIELD_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(50)
    })
}

/// Number of background segment-reaper worker threads. Default 4.
///
/// The reaper drains retired segments one `with_write_txn`-per-DB-chunk
/// at a time. On hot tenants a single worker cannot keep up with merge
/// retire throughput (observed ~8 s per segment × thousand-segment
/// backlog = multi-hour drain). Multiple workers share the flume MPMC
/// channel so retired segments drain in parallel across tenants.
/// Per-env LMDB still serialises their writes, so parallelism wins
/// across tenants rather than within one tenant.
///
/// Override with `HELIX_SEGMENT_REAPER_THREADS`.
fn segment_reaper_threads() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HELIX_SEGMENT_REAPER_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(4)
    })
}

fn segment_reaper_retry_attempts() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| env_usize_or("HELIX_SEGMENT_REAPER_RETRY_ATTEMPTS", 3, 0))
}

/// Chunked replacement for the giant `flush_prepared_merge` exclusive
/// write txn. Splits the merge publish into 5 phases, each running
/// inside its own short `with_exclusive_write_txn`:
///
///   A. allocate name + create LMDB databases for the new segment
///   B. insert flat vectors in deterministic, cost-budgeted chunks
///   C. flush the prepared HNSW in deterministic, cost-budgeted chunks; finalize the entry point
///   D. atomic publish swap (metadata-only)
///   E. drop retired source segments, one txn per segment
///
/// Phases B and C write to a segment that no metadata pointer
/// references yet, so concurrent readers and writers on other
/// segments are not blocked beyond the per-chunk write txn (~250 ms).
/// If the worker dies between phases A and D the partially-written
/// segment has no metadata reference; `gc_orphan_building_segments`
/// reaps it on next startup or next index pass.
fn run_chunked_merge_publish(
    storage: &Arc<HelixGraphStorage>,
    vec_name: &str,
    hnsw_config: &HNSWConfig,
    prepared_merge: crate::helix_engine::vector_core::named_vectors::PreparedMerge,
) -> Result<(), GraphError> {
    let publish_started = Instant::now();
    use crate::helix_engine::vector_core::named_vectors::PreparedMerge;
    let PreparedMerge {
        merge_targets,
        mut prepared_index,
    } = prepared_merge;

    let max_chunk_size = merge_chunk_size();
    let segment_format = dense_segment_format();

    // Phase A: allocate + create LMDB databases. `create_merge_target` now
    // registers the segment in `space.segments` as `Building` so the ID is
    // reserved in persisted metadata. Persisting the metadata blob here means
    // a crash between A and D leaves an orphan that
    // `gc_orphan_building_segments_deferred` can detect and queue to
    // `SEGMENT_REAPER` on the next index cycle.
    let phase_started = Instant::now();
    let merged_name = if storage.backend.kind() == BackendKind::Lsm {
        let name = storage
            .named_vectors
            .create_merge_target_lsm(storage.collection_path(), vec_name, hnsw_config.clone())
            .map_err(GraphError::from)?;
        storage.with_write_backend(|w| {
            storage.set_dense_vector_spaces_metadata_be(
                w,
                storage.named_vectors.list_dense_vector_spaces(),
            )?;
            Ok(())
        })?;
        name
    } else {
        let env = storage.lmdb_env()?.clone();
        storage.with_exclusive_write_txn(|txn| {
            let name = storage
                .named_vectors
                .create_merge_target(&env, txn, vec_name, hnsw_config.clone())
                .map_err(GraphError::from)?;
            storage.set_dense_vector_spaces_metadata(
                txn,
                storage.named_vectors.list_dense_vector_spaces(),
            )?;
            Ok(name)
        })?
    };
    record_dense_merge_publish_phase("allocate", phase_started);
    crate::diag::log("phase_a_done", "chunked_merge", &merged_name);

    // Phase A-quantize-direct (optional, gated): compact-spindle merge targets
    // do not write a raw HVEC sidecar during flat publish. Build the immutable
    // HVS8 sidecar directly from PreparedIndex while raw rows are still live;
    // Phase C drains those rows after adjacency is flushed.
    let mut direct_hvs8_materialized = false;
    let mut direct_tq_materialized = false;
    if segment_format == DenseSegmentFormat::Hvs8 {
        let phase_started = Instant::now();
        match storage
            .named_vectors
            .materialize_merge_target_hvs8_sidecar(&merged_name, &prepared_index)
        {
            Ok(true) => {
                direct_hvs8_materialized = true;
                crate::diag::log(
                    "phase_a_quantize_direct_done",
                    "chunked_merge",
                    &merged_name,
                );
            }
            Ok(false) => {}
            Err(e) => crate::diag::log(
                "phase_a_quantize_direct_skip",
                "chunked_merge",
                &format!("{}: {}", merged_name, e),
            ),
        }
        record_dense_merge_publish_phase("hvs8_materialize", phase_started);
    } else if segment_format == DenseSegmentFormat::TurboQuant {
        let phase_started = Instant::now();
        match storage
            .named_vectors
            .materialize_merge_target_tq_sidecar(&merged_name, &prepared_index)
        {
            Ok(true) => {
                direct_tq_materialized = true;
                crate::diag::log("phase_a_tq_direct_done", "chunked_merge", &merged_name);
            }
            Ok(false) => {
                match storage
                    .named_vectors
                    .materialize_merge_target_hvs8_sidecar(&merged_name, &prepared_index)
                {
                    Ok(true) => {
                        direct_hvs8_materialized = true;
                        crate::diag::log(
                            "phase_a_quantize_direct_done",
                            "chunked_merge",
                            &merged_name,
                        );
                    }
                    Ok(false) => {}
                    Err(e) => crate::diag::log(
                        "phase_a_quantize_direct_skip",
                        "chunked_merge",
                        &format!("{}: {}", merged_name, e),
                    ),
                }
            }
            Err(e) => {
                crate::diag::log(
                    "phase_a_tq_direct_error",
                    "chunked_merge",
                    &format!("{}: {}", merged_name, e),
                );
                return Err(GraphError::from(e));
            }
        }
        record_dense_merge_publish_phase("tq_materialize", phase_started);
    }

    // HVTQ needs ordinals before Phase B so level-0 and upper-level vector rows
    // can be written as LMDB markers instead of duplicating encoded payloads.
    if direct_tq_materialized {
        let phase_started = Instant::now();
        if storage.backend.kind() == BackendKind::Lsm {
            storage.with_write_backend(|w| {
                storage
                    .named_vectors
                    .attach_merge_target_sidecar_ordinals_be(w, &merged_name, &prepared_index)
                    .map_err(GraphError::from)
            })?;
        } else {
            storage.with_write_txn(|txn| {
                storage
                    .named_vectors
                    .attach_merge_target_sidecar_ordinals(txn, &merged_name, &prepared_index)
                    .map_err(GraphError::from)
            })?;
        }
        crate::diag::log(
            "phase_a_tq_direct_ordinals_done",
            "chunked_merge",
            &merged_name,
        );
        record_dense_merge_publish_phase("tq_ordinals", phase_started);
    }

    // Phase B: insert flat vectors in chunks. Uses the *shared* write
    // gate (`with_write_txn`) because the merge target is not yet
    // referenced from `space.segments` — no reader can resolve it, so
    // there is no need to block resize-safe readers via the exclusive
    // resize guard. Layer 3 swap: previously held the exclusive guard
    // for the entire phase, which combined with phase C dominated the
    // 30-263 s `with_exclusive_write_txn` warns we observed in prod.
    let phase_started = Instant::now();
    let total = prepared_index.point_ids.len();
    let mut flat_start = 0usize;
    while flat_start < total {
        let (chunk_len, chunk_stats) =
            budgeted_prepared_flat_chunk_len(&prepared_index, flat_start, max_chunk_size);
        let flat_end = (flat_start + chunk_len).min(total);
        let merged_ref = merged_name.as_str();
        let prepared_ref = &mut prepared_index;
        let chunk_started = Instant::now();
        if storage.backend.kind() == BackendKind::Lsm {
            storage.with_write_backend(|w| {
                storage
                    .named_vectors
                    .flush_merge_flat_chunk_be(w, merged_ref, prepared_ref, flat_start, flat_end)
                    .map_err(GraphError::from)
            })?;
        } else {
            storage.with_write_txn(|txn| {
                storage
                    .named_vectors
                    .flush_merge_flat_chunk(txn, merged_ref, prepared_ref, flat_start, flat_end)
                    .map_err(GraphError::from)
            })?;
        }
        let elapsed = chunk_started.elapsed();
        record_dense_merge_publish_chunk(
            "flat_flush",
            vec_name,
            &merged_name,
            chunk_len,
            chunk_stats,
            elapsed,
        );
        flat_start = flat_end;
    }
    record_dense_merge_publish_phase("flat_flush", phase_started);
    crate::diag::log(
        "phase_b_done",
        "chunked_merge",
        &format!("{}_{}", merged_name, total),
    );

    // Phase C: flush the prepared HNSW in chunks. Final chunk also
    // writes the entry point. Same shared-guard rationale as phase B —
    // the segment is invisible until phase D's metadata swap.
    let phase_started = Instant::now();
    let mut start = 0;
    while start < total {
        let (chunk_len, chunk_stats) =
            budgeted_merge_index_chunk_len(&prepared_index, start, max_chunk_size);
        let end = (start + chunk_len).min(total);
        let merged_ref = merged_name.as_str();
        let prepared_ref = &mut prepared_index;
        let chunk_started = Instant::now();
        if storage.backend.kind() == BackendKind::Lsm {
            storage.with_write_backend(|w| {
                storage
                    .named_vectors
                    .flush_merge_index_chunk_be(w, merged_ref, prepared_ref, start, end)
                    .map_err(GraphError::from)
            })?;
        } else {
            storage.with_write_txn(|txn| {
                storage
                    .named_vectors
                    .flush_merge_index_chunk(txn, merged_ref, prepared_ref, start, end)
                    .map_err(GraphError::from)
            })?;
        }
        let elapsed = chunk_started.elapsed();
        record_dense_merge_publish_chunk(
            "index_flush",
            vec_name,
            &merged_name,
            chunk_len,
            chunk_stats,
            elapsed,
        );
        start = end;
    }
    if storage.backend.kind() == BackendKind::Lsm {
        storage.with_write_backend(|w| {
            storage
                .named_vectors
                .finalize_merge_index_be(w, &merged_name, &prepared_index)
                .map_err(GraphError::from)
        })?;
    } else {
        storage.with_write_txn(|txn| {
            storage
                .named_vectors
                .finalize_merge_index(txn, &merged_name, &prepared_index)
                .map_err(GraphError::from)
        })?;
    }
    record_dense_merge_publish_phase("index_flush", phase_started);
    crate::diag::log("phase_c_done", "chunked_merge", &merged_name);

    // Phase C-quantize (optional, gated): attach ordinals for direct HVS8
    // materialization, or convert a legacy HVEC sidecar after the graph flush.
    // Runs before Phase D so the segment is still invisible to readers. Failure
    // is non-fatal: the segment can publish with LMDB vectors or HVEC fallback.
    if segment_format != DenseSegmentFormat::Legacy {
        let phase_started = Instant::now();
        if direct_hvs8_materialized {
            let attach_result = if storage.backend.kind() == BackendKind::Lsm {
                storage.with_write_backend(|w| {
                    storage
                        .named_vectors
                        .attach_merge_target_sidecar_ordinals_be(w, &merged_name, &prepared_index)
                        .map_err(GraphError::from)
                })
            } else {
                storage.with_write_txn(|txn| {
                    storage
                        .named_vectors
                        .attach_merge_target_sidecar_ordinals(txn, &merged_name, &prepared_index)
                        .map_err(GraphError::from)
                })
            };
            match attach_result {
                Ok(()) => crate::diag::log(
                    "phase_c_quantize_direct_ordinals_done",
                    "chunked_merge",
                    &merged_name,
                ),
                Err(e) => crate::diag::log(
                    "phase_c_quantize_direct_ordinals_skip",
                    "chunked_merge",
                    &format!("{}: {}", merged_name, e),
                ),
            }
        } else if direct_tq_materialized {
            crate::diag::log(
                "phase_c_tq_direct_ordinals_already_attached",
                "chunked_merge",
                &merged_name,
            );
        } else {
            match storage.named_vectors.quantize_merge_target(&merged_name) {
                Ok(true) => {
                    crate::diag::log("phase_c_quantize_done", "chunked_merge", &merged_name)
                }
                Ok(false) => {}
                Err(e) => crate::diag::log(
                    "phase_c_quantize_skip",
                    "chunked_merge",
                    &format!("{}: {}", merged_name, e),
                ),
            }
        }
        record_dense_merge_publish_phase("hvs8_quantize", phase_started);
    }
    drop(prepared_index);

    // Phase D: atomic metadata swap + persist metadata blob in one
    // small write txn. Must hold the exclusive resize guard so readers
    // walking `space.segments` see the new merged segment + the absence
    // of the retired ones in a single coherent snapshot. Cleanup of
    // empty segments is *deferred*: we remove them from `space.segments`
    // here (cheap), and queue the actual `core.clear` work to the
    // background reaper so this txn stays under a few ms. The slow
    // `level_zero_count` scan that locates empties runs in a *read* txn
    // BEFORE we enter the exclusive publish window — only the metadata
    // mutation (publish + retain) holds the writer gate.
    let phase_started = Instant::now();
    let deferred_empties = if storage.backend.kind() == BackendKind::Lsm {
        let r = storage
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))?;
        let deferred_empties = storage
            .named_vectors
            .cleanup_empty_dense_segments_locate_be(&r, vec_name)
            .map_err(GraphError::from)?;
        storage
            .named_vectors
            .publish_merge_target_be(&r, vec_name, &merged_name, &merge_targets)
            .map_err(GraphError::from)?;
        storage
            .named_vectors
            .apply_dense_segment_removal(vec_name, &deferred_empties)
            .map_err(GraphError::from)?;
        storage.with_write_backend(|w| {
            storage.set_dense_vector_spaces_metadata_be(
                w,
                storage.named_vectors.list_dense_vector_spaces(),
            )?;
            Ok(())
        })?;
        deferred_empties
    } else {
        let deferred_empties = storage.with_read_txn(|rtxn| {
            storage
                .named_vectors
                .cleanup_empty_dense_segments_locate(rtxn, vec_name)
                .map_err(GraphError::from)
        })?;
        storage.with_exclusive_write_txn(|txn| {
            storage
                .named_vectors
                .publish_merge_target(txn, vec_name, &merged_name, &merge_targets)
                .map_err(GraphError::from)?;
            storage
                .named_vectors
                .apply_dense_segment_removal(vec_name, &deferred_empties)
                .map_err(GraphError::from)?;
            storage.set_dense_vector_spaces_metadata(
                txn,
                storage.named_vectors.list_dense_vector_spaces(),
            )?;
            Ok(())
        })?;
        deferred_empties
    };
    storage.refresh_metadata_snapshot_best_effort();
    record_dense_merge_publish_phase("metadata_swap", phase_started);
    crate::diag::log("phase_d_done", "chunked_merge", &merged_name);

    // Sidecar vector files are outside LMDB, so remove retired file entries
    // immediately after the metadata swap commits. The LMDB page drain can
    // continue later, but disk-visible .hvec/.hvs8 bloat should not wait on it.
    let phase_started = Instant::now();
    let retired_sidecars = storage
        .named_vectors
        .unlink_sidecar_files_for_segments(storage.path(), &merge_targets);
    let empty_sidecars = storage
        .named_vectors
        .unlink_sidecar_files_for_segments(storage.path(), &deferred_empties);
    if retired_sidecars.error_count > 0 || empty_sidecars.error_count > 0 {
        tracing::warn!(
            retired_errors = retired_sidecars.error_count,
            empty_errors = empty_sidecars.error_count,
            "sidecar unlink after merge publish had errors"
        );
    }
    record_dense_merge_publish_phase("sidecar_unlink", phase_started);

    // Phase E: hand retired source segments + empty segments off to the
    // background reaper. Both sets are no longer referenced from
    // `space.segments` after phase D's commit, so the actual
    // `core.clear(txn)` work can run in any later txn — it competes only
    // with normal writes, not with the merge publish that produced it.
    // Failures here are non-fatal: the metadata pointer no longer
    // references these segments, so any leftover LMDB pages are wasted
    // space, not correctness issues.
    let phase_started = Instant::now();
    for phys in merge_targets {
        SEGMENT_REAPER.submit(Arc::clone(storage), phys);
    }
    for phys in deferred_empties {
        SEGMENT_REAPER.submit(Arc::clone(storage), phys);
    }
    record_dense_merge_publish_phase("reaper_submit", phase_started);
    metrics::histogram!("helix_dense_merge_publish_duration_ms")
        .record(publish_started.elapsed().as_secs_f64() * 1000.0);
    crate::diag::log("phase_e_done", "chunked_merge", &merged_name);

    Ok(())
}

/// Chunked replacement for `flush_prepared_indices`. The non-merge index
/// flush path (post-seal_mutable_tail, before merge consideration)
/// previously held the LMDB exclusive write txn for 1-5s on big
/// `Building` segments, blocking every other operation. Replace with
/// per-segment chunked flush in ~250ms slices.
///
/// Each `(seg_name, prepared)` pair gets:
///   - many small txns flushing slices of the prepared HNSW
///   - one small txn finalizing the entry point + flipping
///     Building→Indexed
///
/// The segment is in `Building` role throughout, so search excludes it
/// until the final flip. Worker death between chunks leaves a Building
/// segment with partial HNSW; gc_orphan_building_segments / next pass
/// re-prepares from level zero (idempotent).
fn run_chunked_indices_publish(
    storage: &Arc<HelixGraphStorage>,
    vec_name: &str,
    prepared: Vec<(
        String,
        crate::helix_engine::vector_core::vector_core::PreparedIndex,
    )>,
) -> Result<bool, GraphError> {
    let max_chunk_size = merge_chunk_size();
    let mut any_flushed = false;
    let mut flushed_names = HashSet::new();

    // The segment is in `Building` role throughout — search excludes it
    // until `promote_building_segments_to_indexed` runs at the end. So
    // there is no need to hold the exclusive resize guard during chunked
    // flush; shared `with_write_txn` is sufficient. Layer 3 swap removes
    // the per-chunk exclusive hold that was the second-largest source of
    // long `with_exclusive_write_txn` warns after merge phase E.
    for (seg_name, mut idx) in prepared {
        let total = idx.point_ids.len();
        if total == 0 {
            continue;
        }
        let seg_ref = seg_name.as_str();
        if storage.backend.kind() == BackendKind::Lsm {
            match storage
                .named_vectors
                .materialize_merge_target_tq_sidecar(seg_ref, &idx)
            {
                Ok(true) => {
                    storage.with_write_backend(|w| {
                        storage
                            .named_vectors
                            .attach_merge_target_sidecar_ordinals_be(w, seg_ref, &idx)
                            .map_err(GraphError::from)
                    })?;
                    metrics::counter!(
                        "helix_lsm_hvtq_sidecar_published_total",
                        "collection" => storage
                            .path()
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("unknown")
                            .to_string(),
                        "vector" => vec_name.to_string(),
                        "source" => "build"
                    )
                    .increment(1);
                    crate::diag::log("hvtq_sidecar_published", "chunked_indices", seg_ref);
                }
                Ok(false) => {}
                Err(e) => return Err(GraphError::from(e)),
            }
        }
        let mut start = 0;
        while start < total {
            let (chunk_len, chunk_stats) =
                budgeted_merge_index_chunk_len(&idx, start, max_chunk_size);
            let end = (start + chunk_len).min(total);
            let idx_ref = &mut idx;
            let chunk_started = Instant::now();
            if storage.backend.kind() == BackendKind::Lsm {
                storage.with_write_backend(|w| {
                    storage
                        .named_vectors
                        .flush_merge_index_chunk_be(w, seg_ref, idx_ref, start, end)
                        .map_err(GraphError::from)
                })?;
            } else {
                storage.with_write_txn(|txn| {
                    storage
                        .named_vectors
                        .flush_merge_index_chunk(txn, seg_ref, idx_ref, start, end)
                        .map_err(GraphError::from)
                })?;
            }
            let elapsed = chunk_started.elapsed();
            record_dense_merge_publish_chunk(
                "indices_flush",
                vec_name,
                seg_ref,
                chunk_len,
                chunk_stats,
                elapsed,
            );
            start = end;
        }
        let seg_ref = seg_name.as_str();
        let idx_ref = &idx;
        if storage.backend.kind() == BackendKind::Lsm {
            storage.with_write_backend(|w| {
                storage
                    .named_vectors
                    .finalize_merge_index_be(w, seg_ref, idx_ref)
                    .map_err(GraphError::from)
            })?;
        } else {
            storage.with_write_txn(|txn| {
                storage
                    .named_vectors
                    .finalize_merge_index(txn, seg_ref, idx_ref)
                    .map_err(GraphError::from)
            })?;
        }
        any_flushed = true;
        flushed_names.insert(seg_name.clone());
        crate::diag::log("indices_chunk_done", "chunked_indices", &seg_name);
    }

    if any_flushed {
        if storage.backend.kind() == BackendKind::Lsm {
            let r = storage
                .backend
                .begin_read()
                .map_err(|e| GraphError::New(e.to_string()))?;
            storage
                .named_vectors
                .promote_building_segments_to_indexed_be(&r, vec_name, &flushed_names)
                .map_err(GraphError::from)?;
            storage.with_write_backend(|w| {
                storage.set_dense_vector_spaces_metadata_be(
                    w,
                    storage.named_vectors.list_dense_vector_spaces(),
                )?;
                Ok(())
            })?;
        } else {
            storage.with_read_txn(|rtxn| {
                storage
                    .named_vectors
                    .promote_building_segments_to_indexed(rtxn, vec_name, &flushed_names)
                    .map_err(GraphError::from)
            })?;
        }
    }

    Ok(any_flushed)
}

fn run_index_job(job: IndexJob) {
    let IndexJob {
        collection,
        storage,
        spaces_to_check,
        hnsw_config,
        flat_scan_threshold,
    } = job;
    let Some(storage) = storage.upgrade() else {
        metrics::counter!("helix_index_job_skipped_total", "reason" => "collection_evicted")
            .increment(1);
        tracing::debug!(
            collection = %collection,
            "skipping optimizer job because collection storage was evicted before execution"
        );
        return;
    };
    let job_started = Instant::now();
    if let Some(error) = storage.degraded_collection_error() {
        metrics::counter!("helix_index_job_skipped_total", "reason" => "collection_degraded")
            .increment(1);
        tracing::warn!(
            collection = %collection,
            error = %error,
            "skipping optimizer job because collection is degraded"
        );
        return;
    }

    // Flush any pending dense layout metadata from the upsert path. Take the
    // dirty mark first and restore it on failure: `with_write_txn` re-runs
    // its closure on MapFull, so the flag must not be cleared inside the
    // transaction (a cleared flag would make the retry a silent no-op).
    // Marks set concurrently during the flush stay set, so the next index
    // job re-flushes. Runs on a background thread so a failed flush doesn't
    // affect the index job itself.
    if storage
        .dense_metadata_dirty
        .swap(false, std::sync::atomic::Ordering::AcqRel)
    {
        let flush_result = if storage.backend.kind() == BackendKind::Lsm {
            storage.with_write_backend(|w| storage.flush_dense_metadata_be(w))
        } else {
            storage.with_write_txn(|wtxn| storage.flush_dense_metadata(wtxn))
        };
        if let Err(e) = flush_result {
            storage.mark_collection_degraded(&e, "index_job_dirty_metadata_flush");
            storage
                .dense_metadata_dirty
                .store(true, std::sync::atomic::Ordering::Release);
            tracing::warn!(error = %e, "Failed to flush dirty dense metadata in index job");
            if e.is_fatal_collection_storage() {
                return;
            }
        }
    }

    metrics::histogram!("helix_index_job_spaces").record(spaces_to_check.len() as f64);

    struct OptimizerGuard(Arc<HelixGraphStorage>);
    impl Drop for OptimizerGuard {
        fn drop(&mut self) {
            self.0
                .optimizer_running
                .store(false, std::sync::atomic::Ordering::Release);
            metrics::gauge!("helix_optimizer_running").decrement(1.0);
        }
    }

    storage
        .optimizer_running
        .store(true, std::sync::atomic::Ordering::Release);
    metrics::gauge!("helix_optimizer_running").increment(1.0);
    let _guard = OptimizerGuard(Arc::clone(&storage));

    let result = (|| -> Result<(), GraphError> {
        const QUIESCE_MS: u64 = if cfg!(test) { 100 } else { 1000 };
        const MAX_PASSES: usize = 20;
        let index_quiesced_tails = index_subthreshold_quiesced_tails();
        let start_debt = storage.named_vectors.dense_segment_debt();
        let indexed_segments = start_debt.indexed_segments;
        let breaker_drain_merge_fan_in =
            breaker_drain_merge_fan_in_limit(start_debt, segment_breaker_drain_floor());
        let mut budget = IndexJobBudget::new(indexed_segments);
        if let Some(max_fan_in) = breaker_drain_merge_fan_in {
            metrics::gauge!(
                "helix_index_job_breaker_drain_merge_fan_in",
                "collection" => collection.to_string()
            )
            .set(max_fan_in as f64);
            tracing::info!(
                collection = %collection,
                indexed_segments = start_debt.indexed_segments,
                merge_debt = start_debt.merge_debt,
                max_fan_in,
                "optimizer bounding breaker-drain merge fan-in"
            );
        }
        if should_reserve_index_job_turn_for_merges(start_debt, breaker_drain_merge_fan_in) {
            budget.reserve_turn_for_merges();
            if start_debt.gate_debt > 0 {
                metrics::counter!(
                    "helix_index_job_gate_debt_merge_first_total",
                    "collection" => collection.to_string()
                )
                .increment(1);
                tracing::info!(
                    collection = %collection,
                    gate_debt = start_debt.gate_debt,
                    merge_debt = start_debt.merge_debt,
                    indexed_segments = start_debt.indexed_segments,
                    active_segments = start_debt.active_segments,
                    "optimizer reserving cap-hit job turn for merges"
                );
            } else if breaker_drain_merge_fan_in.is_some() {
                metrics::counter!(
                    "helix_index_job_breaker_drain_merge_first_total",
                    "collection" => collection.to_string()
                )
                .increment(1);
                tracing::info!(
                    collection = %collection,
                    merge_debt = start_debt.merge_debt,
                    indexed_segments = start_debt.indexed_segments,
                    active_segments = start_debt.active_segments,
                    "optimizer reserving breaker-drain job turn for merges"
                );
            } else {
                metrics::counter!(
                    "helix_index_job_merge_debt_merge_first_total",
                    "collection" => collection.to_string()
                )
                .increment(1);
                tracing::info!(
                    collection = %collection,
                    merge_debt = start_debt.merge_debt,
                    indexed_segments = start_debt.indexed_segments,
                    active_segments = start_debt.active_segments,
                    "optimizer reserving merge-debt job turn for merges"
                );
            }
        } else if optimizer_debt_can_expand_build_budget(start_debt) {
            budget.expand_builds_for_gate_debt(start_debt.gate_debt);
            metrics::counter!(
                "helix_index_job_gate_debt_build_first_total",
                "collection" => collection.to_string()
            )
            .increment(1);
            metrics::gauge!(
                "helix_index_job_gate_debt_build_budget",
                "collection" => collection.to_string()
            )
            .set(budget.max_build_segments as f64);
            tracing::info!(
                collection = %collection,
                gate_debt = start_debt.gate_debt,
                indexed_segments = start_debt.indexed_segments,
                active_segments = start_debt.active_segments,
                max_build_segments = budget.max_build_segments,
                "optimizer expanding cap-hit job build budget"
            );
        }
        metrics::gauge!(
            "helix_index_job_start_segments",
            "collection" => collection.to_string()
        )
        .set(indexed_segments as f64);
        metrics::gauge!(
            "helix_index_job_start_gate_debt",
            "collection" => collection.to_string()
        )
        .set(start_debt.gate_debt as f64);
        metrics::gauge!(
            "helix_index_job_start_merge_debt",
            "collection" => collection.to_string()
        )
        .set(start_debt.merge_debt as f64);
        let mut needs_reschedule = false;

        for _pass in 0..MAX_PASSES {
            if budget.should_yield() {
                needs_reschedule = true;
                break;
            }

            if !budget.is_merge_only_turn() {
                // GC: clean up any Building-role, zero-vector orphans left behind
                // by a crash between phase A and phase D of a previous merge run.
                // `create_merge_target` now registers the merge target as Building
                // in persisted metadata (phase A), so these are detectable across
                // restarts. Deferred to SEGMENT_REAPER — no inline core.clear on
                // the index-job path.
                for vec_name in &spaces_to_check {
                    let orphans = if storage.backend.kind() == BackendKind::Lsm {
                        storage
                            .backend
                            .begin_read()
                            .map_err(|e| GraphError::New(e.to_string()))
                            .and_then(|r| {
                                storage
                                    .named_vectors
                                    .gc_orphan_building_segments_deferred_be(&r, vec_name)
                                    .map_err(GraphError::from)
                            })
                    } else {
                        storage.with_read_txn(|rtxn| {
                            storage
                                .named_vectors
                                .gc_orphan_building_segments_deferred(rtxn, vec_name)
                                .map_err(GraphError::from)
                        })
                    };
                    match orphans {
                        Ok(names) if !names.is_empty() => {
                            // `gc_orphan_building_segments_deferred` already
                            // dropped the orphan entries from the in-memory dense
                            // layout; only the metadata persist differs by
                            // backend (heed metadata writes are `unreachable!()`
                            // on LSM, so route through the backend seam there).
                            let deferred = if storage.backend.kind() == BackendKind::Lsm {
                                storage
                                    .with_write_backend(|w| {
                                        storage.set_dense_vector_spaces_metadata_be(
                                            w,
                                            storage.named_vectors.list_dense_vector_spaces(),
                                        )?;
                                        Ok(())
                                    })
                                    .map(|_| names.clone())
                            } else {
                                storage.with_exclusive_write_txn(|txn| {
                                    storage.set_dense_vector_spaces_metadata(
                                        txn,
                                        storage.named_vectors.list_dense_vector_spaces(),
                                    )?;
                                    Ok(names.clone())
                                })
                            };
                            if let Ok(phys_names) = deferred {
                                let sidecars = storage
                                    .named_vectors
                                    .unlink_sidecar_files_for_segments(storage.path(), &phys_names);
                                if sidecars.error_count > 0 {
                                    tracing::warn!(
                                        errors = sidecars.error_count,
                                        "sidecar unlink for orphan building segments had errors"
                                    );
                                }
                                for phys in phys_names {
                                    SEGMENT_REAPER.submit(Arc::clone(&storage), phys);
                                }
                            }
                        }
                        Ok(_) => {}
                        Err(e) => {
                            tracing::warn!(error = %e, vec_name = %vec_name,
                            "gc_orphan_building_segments_deferred failed (non-fatal)");
                        }
                    }
                }

                if storage.backend.kind() == BackendKind::Lsm {
                    let mut backfilled_hvtq = false;
                    for vec_name in &spaces_to_check {
                        let r = storage
                            .backend
                            .begin_read()
                            .map_err(|e| GraphError::New(e.to_string()))?;
                        let plan = storage
                            .named_vectors
                            .prepare_next_hvtq_sidecar_backfill_be(&r, vec_name, 1)
                            .map_err(GraphError::from)?;
                        drop(r);
                        let Some(plan) = plan else {
                            continue;
                        };
                        let physical_name = plan.physical_name().to_string();
                        storage.with_write_backend(|w| {
                            storage
                                .named_vectors
                                .apply_hvtq_sidecar_backfill_be(w, &plan)
                                .map_err(GraphError::from)
                        })?;
                        metrics::counter!(
                            "helix_lsm_hvtq_sidecar_published_total",
                            "collection" => collection.to_string(),
                            "vector" => vec_name.to_string(),
                            "source" => "backfill"
                        )
                        .increment(1);
                        tracing::info!(
                            collection = %collection,
                            vector = %vec_name,
                            segment = %physical_name,
                            "backfilled missing HVTQ sidecar blob for indexed LSM segment"
                        );
                        backfilled_hvtq = true;
                        needs_reschedule = true;
                        break;
                    }
                    if backfilled_hvtq && budget.should_yield() {
                        break;
                    }
                }

                let repair_limit = externalized_marker_repair_limit_per_segment();
                if repair_limit > 0 && storage.backend.kind() != BackendKind::Lsm {
                    let mut attempted_marker_repair = false;
                    for vec_name in &spaces_to_check {
                        let plan = storage.with_read_txn(|txn| {
                            storage
                                .named_vectors
                                .plan_next_unavailable_externalized_marker_segment(
                                    txn,
                                    vec_name,
                                    repair_limit,
                                )
                                .map_err(GraphError::from)
                        })?;
                        let Some(plan) = plan else {
                            continue;
                        };
                        attempted_marker_repair = true;

                        let repaired = if plan.is_empty() {
                            0
                        } else {
                            storage.with_write_txn(|txn| {
                                storage
                                    .named_vectors
                                    .apply_externalized_marker_repair_plan(txn, vec_name, &plan)
                                    .map_err(GraphError::from)
                            })?
                        };
                        if repaired > 0 {
                            metrics::counter!(
                                "helix_index_job_externalized_marker_repair_total",
                                "collection" => collection.to_string(),
                                "vector" => vec_name.to_string()
                            )
                            .increment(repaired as u64);
                        }
                    }
                    if attempted_marker_repair
                        && spaces_to_check.iter().any(|name| {
                            storage
                                .named_vectors
                                .has_externalized_marker_repair_debt(name)
                                .unwrap_or(false)
                        })
                    {
                        needs_reschedule = true;
                    }
                }

                let quiesced = loop {
                    let now_ms = {
                        use std::time::{SystemTime, UNIX_EPOCH};
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64
                    };
                    let last_upsert = storage
                        .last_upsert_at
                        .load(std::sync::atomic::Ordering::Acquire);
                    let elapsed = now_ms.saturating_sub(last_upsert);
                    if elapsed >= QUIESCE_MS {
                        break true;
                    }
                    let above_threshold = spaces_to_check.iter().any(|name| {
                        let stats = if storage.backend.kind() == BackendKind::Lsm {
                            storage.backend.begin_read().ok().and_then(|r| {
                                storage
                                    .named_vectors
                                    .get_dense_space_stats_be(&r, name)
                                    .ok()
                            })
                        } else {
                            storage.begin_resize_safe_read_txn().ok().and_then(|rtxn| {
                                storage
                                    .named_vectors
                                    .get_dense_space_stats(&rtxn, name)
                                    .ok()
                            })
                        };
                        if let Some(stats) = stats {
                            let unindexed = stats
                                .vectors_count
                                .saturating_sub(stats.indexed_vectors_count);
                            return unindexed as usize >= flat_scan_threshold;
                        }
                        false
                    });
                    if above_threshold {
                        break false;
                    }
                    thread::sleep(std::time::Duration::from_millis(100));
                };

                // Determine which spaces need sealing using a *read* txn so that
                // the O(vectors) prefix scan in get_dense_space_stats never holds
                // the LMDB write lock. Enter the exclusive write txn only when
                // there is actually work to do.
                let spaces_to_seal: Vec<String> = if storage.backend.kind() == BackendKind::Lsm {
                    let r = storage
                        .backend
                        .begin_read()
                        .map_err(|e| GraphError::New(e.to_string()))?;
                    let mut to_seal = Vec::new();
                    for vec_name in &spaces_to_check {
                        if let Ok(stats) =
                            storage.named_vectors.get_dense_space_stats_be(&r, vec_name)
                        {
                            let unindexed = stats
                                .vectors_count
                                .saturating_sub(stats.indexed_vectors_count);
                            let above_threshold = unindexed as usize >= flat_scan_threshold;
                            let quiesced_tail = index_quiesced_tails && quiesced && unindexed > 0;
                            if above_threshold || quiesced_tail {
                                to_seal.push(vec_name.clone());
                            }
                        }
                    }
                    to_seal
                } else {
                    storage.with_read_txn(|rtxn| {
                        let mut to_seal = Vec::new();
                        for vec_name in &spaces_to_check {
                            if let Ok(stats) =
                                storage.named_vectors.get_dense_space_stats(rtxn, vec_name)
                            {
                                let unindexed = stats
                                    .vectors_count
                                    .saturating_sub(stats.indexed_vectors_count);
                                let above_threshold = unindexed as usize >= flat_scan_threshold;
                                let quiesced_tail =
                                    index_quiesced_tails && quiesced && unindexed > 0;
                                if above_threshold || quiesced_tail {
                                    to_seal.push(vec_name.clone());
                                }
                            }
                        }
                        Ok(to_seal)
                    })?
                };

                for vec_name in &spaces_to_seal {
                    // The exclusive write txn registers the new mutable
                    // segment's DBI handles (heed env) and flips the sealed
                    // tail's role in memory — both correct on LSM, where
                    // segment registration still rides the graph env. Only the
                    // dense-layout METADATA persistence differs: on LSM it must
                    // go through the backend seam (the heed metadata write path
                    // is `unreachable!()` on the LSM arm), so persist it via
                    // `with_write_backend` after the heed txn commits.
                    let _sealed = if storage.backend.kind() == BackendKind::Lsm {
                        let r = storage
                            .backend
                            .begin_read()
                            .map_err(|e| GraphError::New(e.to_string()))?;
                        let sealed = storage
                            .named_vectors
                            .seal_mutable_tail_lsm(
                                &r,
                                storage.collection_path(),
                                vec_name,
                                hnsw_config.clone(),
                            )
                            .map_err(GraphError::from)?;
                        if sealed {
                            storage.with_write_backend(|w| {
                                storage.set_dense_vector_spaces_metadata_be(
                                    w,
                                    storage.named_vectors.list_dense_vector_spaces(),
                                )?;
                                Ok(())
                            })?;
                        }
                        sealed
                    } else {
                        storage.with_exclusive_write_txn(|txn| {
                            let sealed = storage
                                .named_vectors
                                .seal_mutable_tail(
                                    storage.lmdb_env()?,
                                    txn,
                                    vec_name,
                                    hnsw_config.clone(),
                                )
                                .map_err(GraphError::from)?;
                            if sealed {
                                storage.set_dense_vector_spaces_metadata(
                                    txn,
                                    storage.named_vectors.list_dense_vector_spaces(),
                                )?;
                            }
                            Ok(sealed)
                        })?
                    };

                    if budget.should_yield() || budget.elapsed_exceeded() {
                        needs_reschedule = true;
                        break;
                    }
                }

                if needs_reschedule {
                    break;
                }

                let spaces_needing_build: Vec<String> = spaces_to_check
                    .iter()
                    .filter(|name| {
                        !storage
                            .named_vectors
                            .building_segment_names(name)
                            .is_empty()
                    })
                    .cloned()
                    .collect();

                let mut build_layout_changed = false;
                let mut build_prepared_any = false;
                if !spaces_needing_build.is_empty() {
                    'build_spaces: for vec_name in &spaces_needing_build {
                        loop {
                            let remaining_builds = budget.remaining_build_segments();
                            if remaining_builds == 0 {
                                needs_reschedule = true;
                                break 'build_spaces;
                            }
                            let batch_limit =
                                remaining_builds.min(index_job_build_flush_batch_segments());
                            // Background-build admission: defer under memory
                            // pressure HERE, before any txn opens. The deferral
                            // must never run inside acquire_build_permit — several
                            // of its callers hold the LMDB writer.
                            crate::helix_engine::vector_core::vector_core::defer_build_start_under_pressure();
                            let prepared = if storage.backend.kind() == BackendKind::Lsm {
                                let r = storage
                                    .backend
                                    .begin_read()
                                    .map_err(|e| GraphError::New(e.to_string()))?;
                                storage
                                    .named_vectors
                                    .prepare_building_indices_limited_be(&r, vec_name, batch_limit)
                                    .map_err(GraphError::from)?
                            } else {
                                storage.with_read_txn(|rtxn| {
                                    storage
                                        .named_vectors
                                        .prepare_building_indices_limited(
                                            rtxn,
                                            vec_name,
                                            batch_limit,
                                        )
                                        .map_err(GraphError::from)
                                })?
                            };
                            let prepared_count = prepared.len();
                            if prepared_count == 0 {
                                break;
                            }

                            budget.record_build_segments(prepared_count);
                            let flushed =
                                run_chunked_indices_publish(&storage, vec_name, prepared)?;
                            build_layout_changed |= flushed;
                            build_prepared_any = true;

                            if prepared_count < batch_limit {
                                break;
                            }
                            if budget.elapsed_exceeded() {
                                needs_reschedule = true;
                                break 'build_spaces;
                            }
                            if budget.should_yield() {
                                needs_reschedule = true;
                                break 'build_spaces;
                            }
                        }
                    }
                }

                if build_prepared_any {
                    // Layer 3: defer the heavy `core.clear` work to the
                    // background reaper. The slow `level_zero_count` scan
                    // runs in a *read* txn so it never holds the writer
                    // gate; only the small metadata-mutation enters the
                    // exclusive write txn.
                    let mut empties_per_space: Vec<(String, Vec<String>)> = Vec::new();
                    for vec_name in &spaces_to_check {
                        let names = if storage.backend.kind() == BackendKind::Lsm {
                            let r = storage
                                .backend
                                .begin_read()
                                .map_err(|e| GraphError::New(e.to_string()))?;
                            storage
                                .named_vectors
                                .cleanup_empty_dense_segments_locate_be(&r, vec_name)
                                .map_err(GraphError::from)?
                        } else {
                            storage.with_read_txn(|rtxn| {
                                storage
                                    .named_vectors
                                    .cleanup_empty_dense_segments_locate(rtxn, vec_name)
                                    .map_err(GraphError::from)
                            })?
                        };
                        if !names.is_empty() {
                            empties_per_space.push((vec_name.clone(), names));
                        }
                    }
                    let any_empties = empties_per_space.iter().any(|(_, n)| !n.is_empty());
                    if any_empties || build_layout_changed {
                        // `apply_dense_segment_removal` only mutates the
                        // in-memory dense layout (no txn), so the empties can be
                        // removed identically on both backends. Only the dense
                        // metadata persistence differs: the heed metadata write
                        // path is `unreachable!()` on the LSM arm, so route the
                        // persist through the backend seam there.
                        if storage.backend.kind() == BackendKind::Lsm {
                            let mut removal_layout_changed = false;
                            for (vec_name, empties) in &empties_per_space {
                                let removed = storage
                                    .named_vectors
                                    .apply_dense_segment_removal(vec_name, empties)
                                    .map_err(GraphError::from)?;
                                removal_layout_changed |= removed;
                            }
                            if build_layout_changed || removal_layout_changed {
                                storage.with_write_backend(|w| {
                                    storage.set_dense_vector_spaces_metadata_be(
                                        w,
                                        storage.named_vectors.list_dense_vector_spaces(),
                                    )?;
                                    Ok(())
                                })?;
                            }
                        } else {
                            storage.with_exclusive_write_txn(|txn| {
                                let mut removal_layout_changed = false;
                                for (vec_name, empties) in &empties_per_space {
                                    let removed = storage
                                        .named_vectors
                                        .apply_dense_segment_removal(vec_name, empties)
                                        .map_err(GraphError::from)?;
                                    removal_layout_changed |= removed;
                                }
                                if build_layout_changed || removal_layout_changed {
                                    storage.set_dense_vector_spaces_metadata(
                                        txn,
                                        storage.named_vectors.list_dense_vector_spaces(),
                                    )?;
                                }
                                Ok(())
                            })?;
                        }
                    }
                    for (_, empties) in empties_per_space {
                        let sidecars = storage
                            .named_vectors
                            .unlink_sidecar_files_for_segments(storage.path(), &empties);
                        if sidecars.error_count > 0 {
                            tracing::warn!(
                                errors = sidecars.error_count,
                                "sidecar unlink for empty dense segments had errors"
                            );
                        }
                        for phys in empties {
                            SEGMENT_REAPER.submit(Arc::clone(&storage), phys);
                        }
                    }
                    if budget.should_yield() {
                        needs_reschedule = true;
                    }
                }
            }

            let merge_ceiling = NamedVectorManager::max_indexed_segments_target();
            for vec_name in &spaces_to_check {
                if budget.should_yield() {
                    needs_reschedule = true;
                    break;
                }
                loop {
                    if !budget.can_merge() || budget.elapsed_exceeded() {
                        needs_reschedule = true;
                        break;
                    }
                    let candidates = if storage.backend.kind() == BackendKind::Lsm {
                        let r = storage
                            .backend
                            .begin_read()
                            .map_err(|e| GraphError::New(e.to_string()))?;
                        let limits = if let Some(max_fan_in) = breaker_drain_merge_fan_in {
                            DenseMergeCandidateLimits::bounded_prefix(merge_ceiling, max_fan_in)
                        } else {
                            DenseMergeCandidateLimits::new(
                                merge_ceiling,
                                NamedVectorManager::merge_max_fan_in(),
                            )
                        };
                        storage
                            .named_vectors
                            .select_merge_candidates_with_limits_be(&r, vec_name, limits)
                            .map_err(GraphError::from)?
                    } else {
                        storage.with_read_txn(|rtxn| {
                            if let Some(max_fan_in) = breaker_drain_merge_fan_in {
                                storage
                                    .named_vectors
                                    .select_merge_candidates_with_limits(
                                        rtxn,
                                        vec_name,
                                        DenseMergeCandidateLimits::bounded_prefix(
                                            merge_ceiling,
                                            max_fan_in,
                                        ),
                                    )
                                    .map_err(GraphError::from)
                            } else {
                                storage
                                    .named_vectors
                                    .select_merge_candidates(rtxn, vec_name, merge_ceiling)
                                    .map_err(GraphError::from)
                            }
                        })?
                    };
                    let Some(targets) = candidates else {
                        break;
                    };
                    metrics::counter!(
                        "helix_dense_segment_merge_selected_total",
                        "collection" => collection.to_string(),
                        "vector" => vec_name.to_string(),
                        "target" => merge_ceiling.to_string(),
                        "segments" => targets.len().to_string(),
                    )
                    .increment(1);

                    // Background-merge admission: defer under memory pressure
                    // before taking the permit or opening any txn (the export
                    // below is the merge build's peak-RSS phase).
                    crate::helix_engine::vector_core::vector_core::defer_build_start_under_pressure(
                    );
                    // Hold the HNSW build permit before exporting source
                    // vectors. The exported Vec is part of the merge build's
                    // peak RSS; if workers wait for the permit after export,
                    // queued merge jobs can still stack multi-GB working sets.
                    let build_permit = acquire_build_permit().map_err(GraphError::from)?;
                    let exported = if storage.backend.kind() == BackendKind::Lsm {
                        let r = storage
                            .backend
                            .begin_read()
                            .map_err(|e| GraphError::New(e.to_string()))?;
                        let cores = storage
                            .named_vectors
                            .cores_read()
                            .map_err(GraphError::from)?;
                        let mut exported = Vec::new();
                        let mut seen = std::collections::HashSet::new();
                        let mut dropped_tombstoned = 0u64;
                        for phys in &targets {
                            if let Some(core) = cores.get(phys) {
                                for (id, data, fields) in
                                    core.export_level_zero(&r).map_err(GraphError::from)?
                                {
                                    // Deferred-repair deletes (issue #29 "fix 2"):
                                    // a delete-tombstoned id's row is still
                                    // physically present until this merge, so
                                    // this is where it actually gets reclaimed.
                                    if core.is_delete_tombstoned(id) {
                                        dropped_tombstoned += 1;
                                        continue;
                                    }
                                    if storage
                                        .named_vectors
                                        .is_tombstoned_for_merge_be(&r, phys, id)
                                        .map_err(GraphError::from)?
                                    {
                                        dropped_tombstoned += 1;
                                        continue;
                                    }
                                    if seen.insert(id) {
                                        exported.push((id, data, fields));
                                    }
                                }
                            }
                        }
                        drop(cores);
                        if dropped_tombstoned > 0 {
                            metrics::counter!(
                                "helix_reupsert_tombstones_dropped_on_merge_total",
                                "collection" => collection.to_string(),
                            )
                            .increment(dropped_tombstoned);
                        }
                        exported
                    } else {
                        storage.with_read_txn(|rtxn| {
                            let cores = storage
                                .named_vectors
                                .cores_read()
                                .map_err(GraphError::from)?;
                            // Re-upsert tombstones: a tombstoned (segment, id) copy
                            // is superseded — drop it here so the merged segment
                            // never carries a stale version. The live copy lives in
                            // a newer segment (usually the mutable tail, outside
                            // `targets`). Search already prefers the newest copy
                            // (recency dedup); this is the storage-reclamation half.
                            let mut exported = Vec::new();
                            let mut seen = std::collections::HashSet::new();
                            let mut dropped_tombstoned = 0u64;
                            for phys in &targets {
                                if let Some(core) = cores.get(phys) {
                                    let rd = core.backend.read_borrowed(rtxn);
                                    for (id, data, fields) in
                                        core.export_level_zero(&rd).map_err(GraphError::from)?
                                    {
                                        // Deferred-repair deletes (issue #29
                                        // "fix 2"): the LMDB path never sets
                                        // HELIX_LSM_DELETE_TOMBSTONES, so this
                                        // set is normally empty here, but the
                                        // check is unconditional the same way
                                        // the search gate is.
                                        if core.is_delete_tombstoned(id) {
                                            dropped_tombstoned += 1;
                                            continue;
                                        }
                                        if storage
                                            .named_vectors
                                            .is_tombstoned_for_merge(rtxn, phys, id)
                                            .map_err(GraphError::from)?
                                        {
                                            dropped_tombstoned += 1;
                                            continue;
                                        }
                                        if seen.insert(id) {
                                            exported.push((id, data, fields));
                                        }
                                    }
                                }
                            }
                            drop(cores);
                            if dropped_tombstoned > 0 {
                                metrics::counter!(
                                    "helix_reupsert_tombstones_dropped_on_merge_total",
                                    "collection" => collection.to_string(),
                                )
                                .increment(dropped_tombstoned);
                            }

                            // Phase 4: graph-affinity reorder.
                            // The chunked merge path (this function) constructs
                            // PreparedMerge inline below instead of going through
                            // NamedVectorManager::prepare_merge, so the affinity
                            // reorder hook inside prepare_merge never fires here.
                            // Mirror it so HELIX_GRAPH_AFFINITY_MERGE=1 actually
                            // changes on-disk layout for production merges.
                            if NamedVectorManager::graph_affinity_merge_enabled() {
                                let reorder_start = std::time::Instant::now();
                                let (reordered, edges_followed) =
                                    storage.named_vectors.graph_affinity_reorder(rtxn, exported);
                                exported = reordered;
                                metrics::histogram!(
                                    "helix_graph_affinity_reorder_duration_ms",
                                    "collection" => collection.to_string(),
                                )
                                .record(reorder_start.elapsed().as_secs_f64() * 1000.0);
                                metrics::counter!(
                                    "helix_graph_affinity_edges_followed_total",
                                    "collection" => collection.to_string(),
                                )
                                .increment(edges_followed);
                            }
                            Ok(exported)
                        })?
                    };
                    let prepared_index = VectorCore::build_hnsw_in_memory_owned_with_permit(
                        exported,
                        &hnsw_config,
                        &build_permit,
                    )
                    .map_err(GraphError::from)?;
                    let prepared_merge =
                        crate::helix_engine::vector_core::named_vectors::PreparedMerge {
                            merge_targets: targets,
                            prepared_index,
                        };

                    let estimated_bytes = estimate_merge_flush_bytes(&prepared_merge);
                    if let Err(e) = storage.ensure_map_headroom(estimated_bytes) {
                        tracing::debug!(
                            error = %e,
                            estimated_bytes,
                            collection = %collection,
                            "ensure_map_headroom failed before merge flush; proceeding"
                        );
                    }
                    let flush_outcome =
                        run_chunked_merge_publish(&storage, vec_name, &hnsw_config, prepared_merge);
                    match flush_outcome {
                        Ok(()) => {
                            budget.record_merge();
                            if budget.should_yield() {
                                needs_reschedule = true;
                                break;
                            }
                        }
                        Err(GraphError::MapFull) => {
                            tracing::warn!(
                                vec_name = %vec_name,
                                collection = %collection,
                                estimated_bytes,
                                "merge flush hit MapFull; growing map and re-preparing merge"
                            );
                            storage.grow_map()?;
                            break;
                        }
                        Err(e) => return Err(e),
                    }
                }
            }

            let has_building = spaces_to_check.iter().any(|name| {
                !storage
                    .named_vectors
                    .building_segment_names(name)
                    .is_empty()
            });

            if has_building {
                if budget.remaining_build_segments() == 0 {
                    needs_reschedule = true;
                    break;
                }
                thread::sleep(std::time::Duration::from_millis(50));
                continue;
            }

            let has_unindexed_tail = spaces_to_check.iter().any(|name| {
                if storage.backend.kind() == BackendKind::Lsm {
                    if let Ok(r) = storage.backend.begin_read() {
                        if let Ok(stats) = storage.named_vectors.get_dense_space_stats_be(&r, name)
                        {
                            let unindexed = stats
                                .vectors_count
                                .saturating_sub(stats.indexed_vectors_count);
                            return unindexed > 0;
                        }
                    }
                } else if let Ok(rtxn) = storage.begin_resize_safe_read_txn() {
                    if let Ok(stats) = storage.named_vectors.get_dense_space_stats(&rtxn, name) {
                        let unindexed = stats
                            .vectors_count
                            .saturating_sub(stats.indexed_vectors_count);
                        return unindexed > 0;
                    }
                }
                false
            });

            if !has_unindexed_tail {
                break;
            }

            if !index_quiesced_tails {
                break;
            }

            let now_check = {
                use std::time::{SystemTime, UNIX_EPOCH};
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64
            };
            let lu = storage
                .last_upsert_at
                .load(std::sync::atomic::Ordering::Acquire);
            if now_check.saturating_sub(lu) >= QUIESCE_MS {
                continue;
            }

            thread::sleep(std::time::Duration::from_millis(200));
        }

        // Final convergence cleanup — same Layer 3 deferred pattern as
        // the post-indices-publish cleanup above. The slow
        // `level_zero_count` scan runs in a read txn so it doesn't hold
        // the writer gate; only the metadata-mutation enters exclusive.
        let mut empties_per_space: Vec<(String, Vec<String>)> = Vec::new();
        for vec_name in &spaces_to_check {
            let names = if storage.backend.kind() == BackendKind::Lsm {
                let r = storage
                    .backend
                    .begin_read()
                    .map_err(|e| GraphError::New(e.to_string()))?;
                storage
                    .named_vectors
                    .cleanup_empty_dense_segments_locate_be(&r, vec_name)
                    .map_err(GraphError::from)?
            } else {
                storage.with_read_txn(|rtxn| {
                    storage
                        .named_vectors
                        .cleanup_empty_dense_segments_locate(rtxn, vec_name)
                        .map_err(GraphError::from)
                })?
            };
            if !names.is_empty() {
                empties_per_space.push((vec_name.clone(), names));
            }
        }
        if !empties_per_space.is_empty() {
            if storage.backend.kind() == BackendKind::Lsm {
                let mut layout_changed = false;
                for (vec_name, empties) in &empties_per_space {
                    let removed = storage
                        .named_vectors
                        .apply_dense_segment_removal(vec_name, empties)
                        .map_err(GraphError::from)?;
                    layout_changed |= removed;
                }
                if layout_changed {
                    storage.with_write_backend(|w| {
                        storage.set_dense_vector_spaces_metadata_be(
                            w,
                            storage.named_vectors.list_dense_vector_spaces(),
                        )?;
                        Ok(())
                    })?;
                }
            } else {
                storage.with_exclusive_write_txn(|txn| {
                    let mut layout_changed = false;
                    for (vec_name, empties) in &empties_per_space {
                        let removed = storage
                            .named_vectors
                            .apply_dense_segment_removal(vec_name, empties)
                            .map_err(GraphError::from)?;
                        layout_changed |= removed;
                    }
                    if layout_changed {
                        storage.set_dense_vector_spaces_metadata(
                            txn,
                            storage.named_vectors.list_dense_vector_spaces(),
                        )?;
                    }
                    Ok(())
                })?;
            }
        }
        for (_, empties) in empties_per_space {
            let sidecars = storage
                .named_vectors
                .unlink_sidecar_files_for_segments(storage.path(), &empties);
            if sidecars.error_count > 0 {
                tracing::warn!(
                    errors = sidecars.error_count,
                    "sidecar unlink for final empty dense segments had errors"
                );
            }
            for phys in empties {
                SEGMENT_REAPER.submit(Arc::clone(&storage), phys);
            }
        }

        if needs_reschedule
            && optimizer_has_pending_work(
                &storage,
                &spaces_to_check,
                flat_scan_threshold,
                index_quiesced_tails,
            )
        {
            metrics::counter!("helix_index_job_rescheduled_total").increment(1);
            if let Err(e) = INDEX_EXECUTOR.submit(IndexJob {
                collection: collection.clone(),
                storage: Arc::downgrade(&storage),
                spaces_to_check: spaces_to_check.clone(),
                hnsw_config: hnsw_config.clone(),
                flat_scan_threshold,
            }) {
                tracing::warn!(
                    error = %e,
                    collection = %collection,
                    "index job could not reschedule bounded follow-up"
                );
            }
        }

        Ok(())
    })();

    let status = if result.is_ok() { "ok" } else { "error" };
    if let Err(e) = result {
        storage.mark_collection_degraded(&e, "index_job");
        if matches!(e, GraphError::MapFull) {
            tracing::warn!(
                error = %e,
                collection = %collection,
                "background optimizer deferred cycle (MapFull); next cycle retries on larger map"
            );
        } else {
            tracing::error!(error = %e, collection = %collection, "background optimizer failed");
        }
    }
    metrics::histogram!("helix_index_job_duration_ms", "status" => status)
        .record(job_started.elapsed().as_secs_f64() * 1000.0);
}

fn optimizer_has_pending_work(
    storage: &Arc<HelixGraphStorage>,
    spaces_to_check: &[String],
    flat_scan_threshold: usize,
    index_quiesced_tails: bool,
) -> bool {
    if storage.degraded_collection_error().is_some() {
        return false;
    }

    if spaces_to_check.iter().any(|name| {
        !storage
            .named_vectors
            .building_segment_names(name)
            .is_empty()
    }) {
        return true;
    }

    if storage.backend.kind() == BackendKind::Lsm {
        let has_hvtq_backfill_debt = (|| -> Result<bool, GraphError> {
            let r = storage
                .backend
                .begin_read()
                .map_err(|e| GraphError::New(e.to_string()))?;
            for vec_name in spaces_to_check {
                if storage
                    .named_vectors
                    .has_hvtq_sidecar_backfill_debt_be(&r, vec_name)
                    .map_err(GraphError::from)?
                {
                    return Ok(true);
                }
            }
            Ok(false)
        })()
        .unwrap_or(false);
        if has_hvtq_backfill_debt {
            return true;
        }
    }

    if externalized_marker_repair_limit_per_segment() > 0
        && spaces_to_check.iter().any(|name| {
            storage
                .named_vectors
                .has_externalized_marker_repair_debt(name)
                .unwrap_or(false)
        })
    {
        return true;
    }

    let merge_ceiling = NamedVectorManager::max_indexed_segments_target();
    if storage.backend.kind() == BackendKind::Lsm {
        (|| -> Result<bool, GraphError> {
            let r = storage
                .backend
                .begin_read()
                .map_err(|e| GraphError::New(e.to_string()))?;
            for vec_name in spaces_to_check {
                if let Ok(stats) = storage.named_vectors.get_dense_space_stats_be(&r, vec_name) {
                    let unindexed = stats
                        .vectors_count
                        .saturating_sub(stats.indexed_vectors_count);
                    if (unindexed as usize) >= flat_scan_threshold
                        || (index_quiesced_tails && unindexed > 0)
                    {
                        return Ok(true);
                    }
                }
                if storage
                    .named_vectors
                    .select_merge_candidates_with_limits_be(
                        &r,
                        vec_name,
                        DenseMergeCandidateLimits::new(
                            merge_ceiling,
                            NamedVectorManager::merge_max_fan_in(),
                        ),
                    )
                    .map_err(GraphError::from)?
                    .is_some()
                {
                    return Ok(true);
                }
            }
            Ok(false)
        })()
        .unwrap_or(false)
    } else {
        storage
            .with_read_txn(|rtxn| {
                for vec_name in spaces_to_check {
                    if let Ok(stats) = storage.named_vectors.get_dense_space_stats(rtxn, vec_name) {
                        let unindexed = stats
                            .vectors_count
                            .saturating_sub(stats.indexed_vectors_count);
                        if (unindexed as usize) >= flat_scan_threshold
                            || (index_quiesced_tails && unindexed > 0)
                        {
                            return Ok(true);
                        }
                    }
                    if storage
                        .named_vectors
                        .select_merge_candidates(rtxn, vec_name, merge_ceiling)
                        .map_err(GraphError::from)?
                        .is_some()
                    {
                        return Ok(true);
                    }
                }
                Ok(false)
            })
            .unwrap_or(false)
    }
}

/// Rough per-vector overhead for the HNSW graph serialized under the merge
/// flush transaction: node links (M × 2 layers × 8 bytes per link id) plus
/// per-node metadata headers. A conservative ~512 bytes covers M≤32 with
/// multi-layer nodes; the `ensure_map_headroom` over-provisioning margin
/// absorbs the tail.
const HNSW_PER_VECTOR_OVERHEAD_BYTES: usize = 512;

/// Estimate the LMDB bytes a `flush_prepared_merge` commit will write:
/// dense vector data + HNSW links + per-point payload/fields. Used to size
/// the pre-grow so a single-attempt write txn has headroom.
fn estimate_merge_flush_bytes(
    merge: &crate::helix_engine::vector_core::named_vectors::PreparedMerge,
) -> usize {
    let mut vector_bytes: usize = 0;
    let mut payload_bytes: usize = 0;
    let prepared = &merge.prepared_index;
    let n = prepared.point_ids.len();
    for i in 0..n {
        vector_bytes = vector_bytes.saturating_add(prepared.raw_len_at(i).saturating_mul(4));
        let Some(fields) = prepared.point_fields.get(i) else {
            continue;
        };
        // Cheap upper bound for serialized payload: 32 bytes per key plus the
        // stringified value length. Exact value encoding isn't worth walking.
        let mut entry = 0usize;
        for (k, v) in fields {
            entry = entry
                .saturating_add(k.len().saturating_add(32))
                .saturating_add(match v {
                    Value::String(s) => s.len().saturating_add(8),
                    Value::Array(arr) => arr.len().saturating_mul(16).saturating_add(16),
                    _ => 32,
                });
        }
        payload_bytes = payload_bytes.saturating_add(entry);
    }
    let hnsw_bytes = n.saturating_mul(HNSW_PER_VECTOR_OVERHEAD_BYTES);
    vector_bytes
        .saturating_add(payload_bytes)
        .saturating_add(hnsw_bytes)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReplicatedPoint {
    pub id: u128,
    pub vectors: HashMap<String, Vec<f32>>,
    #[serde(default)]
    pub sparse_vectors: HashMap<String, SparseVector>,
    pub payload: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReplicatedIngestOp {
    Node(NodeUpsert),
    Edge(EdgeUpsert),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReplicatedMutation {
    CreateCollection {
        name: String,
        vectors: HashMap<String, NamedVectorConfig>,
        #[serde(default)]
        sparse_vectors: HashMap<String, SparseVectorConfig>,
        /// Optional per-collection HNSW overrides (m / ef_construction / ef).
        /// Trailing `#[serde(default)]` keeps older bincode payloads
        /// decodable: when absent, falls back to global `HELIX_HNSW_*` env.
        #[serde(default)]
        hnsw_overrides: Option<HnswOverrides>,
    },
    DeleteCollection {
        name: String,
    },
    UpsertPoints {
        collection: String,
        points: Vec<ReplicatedPoint>,
    },
    DeletePoints {
        collection: String,
        ids: Vec<u128>,
    },
    CreatePayloadIndex {
        collection: String,
        field_name: String,
        schema: PayloadIndexSchema,
    },
    DeletePayloadIndex {
        collection: String,
        field_name: String,
    },
    IngestBatch {
        collection: String,
        ops: Vec<ReplicatedIngestOp>,
    },
    UpdateCollection {
        name: String,
        #[serde(default)]
        vectors: HashMap<String, NamedVectorConfig>,
        #[serde(default)]
        sparse_vectors: HashMap<String, SparseVectorConfig>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplicationStatus {
    pub enabled: bool,
    pub node_id: Option<u64>,
    pub leader_id: Option<u64>,
    pub term: u64,
    pub is_leader: bool,
    pub commit_index: u64,
    pub applied_index: u64,
    pub first_index: u64,
    pub last_index: u64,
    pub snapshot_index: u64,
}

enum RuntimeCommand {
    Propose {
        proposal: Proposal,
        response: Sender<RuntimeProposalResult>,
    },
    ReadBarrier {
        context: Vec<u8>,
        response: Sender<RuntimeReadResult>,
    },
    Step {
        message: Message,
    },
    Status {
        response: Sender<ReplicationStatus>,
    },
    Shutdown,
}

enum RuntimeProposalResult {
    Applied(Result<(), String>),
    NotLeader { leader_id: Option<u64> },
}

enum RuntimeReadResult {
    Ready(Result<(), String>),
    NotLeader { leader_id: Option<u64> },
}

struct PendingRead {
    required_index: Option<u64>,
    response: Sender<RuntimeReadResult>,
}

pub(crate) trait RaftTransport: Send + Sync {
    fn send_message(&self, target: &str, message: &Message) -> Result<(), String>;
    fn forward_proposal(&self, target: &str, mutation: &ReplicatedMutation) -> Result<(), String>;
}

pub struct ReplicationManager {
    collections: Arc<CollectionManager>,
    config: Config,
    runtime: Option<Arc<RuntimeHandle>>,
    transport: Arc<dyn RaftTransport>,
    cloud_gateway: Option<Arc<CloudGatewayRouter>>,
}

#[derive(Clone, Debug)]
struct CloudGatewayConfig {
    writer_url: String,
    reader_urls: Vec<String>,
    max_reader_attempts: usize,
    reader_eviction: Duration,
    reader_proxy_timeout: Duration,
    writer_proxy_timeout: Duration,
    writer_is_local: bool,
    include_writer_for_reads: bool,
    buffer: Option<GatewayBufferConfig>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GatewayTargetKind {
    Writer,
    Reader,
}

impl GatewayTargetKind {
    fn as_label(self) -> &'static str {
        match self {
            Self::Writer => "writer",
            Self::Reader => "reader",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GatewayTarget {
    kind: GatewayTargetKind,
    base_url: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReaderProxyFailureLogClass {
    LocalWriterFallback,
    PotentialClientFailure,
}

/// A reader transport failure is diagnostic when this writer process can
/// satisfy the same read locally later in the target plan. Without that
/// guaranteed local fallback, retain warning visibility because the request
/// may still propagate an upstream failure to the client.
fn reader_proxy_failure_log_class(
    writer_is_local: bool,
    targets: &[GatewayTarget],
    failed_index: usize,
) -> ReaderProxyFailureLogClass {
    if writer_is_local
        && targets
            .iter()
            .skip(failed_index.saturating_add(1))
            .any(|target| target.kind == GatewayTargetKind::Writer)
    {
        ReaderProxyFailureLogClass::LocalWriterFallback
    } else {
        ReaderProxyFailureLogClass::PotentialClientFailure
    }
}

struct CloudGatewayRouter {
    config: CloudGatewayConfig,
    next_reader: AtomicU64,
    unhealthy_readers: Mutex<HashMap<String, Instant>>,
    buffer: Option<Arc<GatewayDurableBuffer>>,
}

impl CloudGatewayConfig {
    fn from_env() -> Option<Self> {
        // Reader replicas must execute requests against their local DbReader.
        // Enabling the shared gateway config on a reader sends the request back
        // through the reader Service and recursively proxies it until timeout.
        if lsm_role_is_reader() {
            return None;
        }
        let writer_url = gateway_env_first(&[
            "HELIX_GATEWAY_WRITER_URL",
            "HELIX_CLOUD_WRITER_URL",
            "HELIX_WRITER_URL",
        ])?;
        let reader_urls = gateway_env_list(&[
            "HELIX_GATEWAY_READER_URLS",
            "HELIX_CLOUD_READER_URLS",
            "HELIX_READER_URLS",
        ]);
        let max_reader_attempts =
            gateway_env_usize("HELIX_GATEWAY_MAX_READER_ATTEMPTS", usize::MAX);
        let reader_eviction = Duration::from_millis(
            std::env::var("HELIX_GATEWAY_READER_EVICT_MS")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(GATEWAY_READER_EVICT_MS),
        );
        let reader_proxy_timeout = Duration::from_millis(gateway_env_u64(
            "HELIX_GATEWAY_READER_TIMEOUT_MS",
            GATEWAY_READER_PROXY_TIMEOUT_MS,
        ));
        let writer_proxy_timeout = Duration::from_millis(gateway_env_u64(
            "HELIX_GATEWAY_WRITER_TIMEOUT_MS",
            GATEWAY_WRITER_PROXY_TIMEOUT_MS,
        ));
        let include_writer_for_reads =
            gateway_env_bool("HELIX_GATEWAY_READS_INCLUDE_WRITER", false)
                || gateway_env_bool("HELIX_CLOUD_READS_INCLUDE_WRITER", false);
        let buffer = gateway_buffer_config_from_env();

        Some(Self {
            writer_url,
            reader_urls,
            max_reader_attempts,
            reader_eviction,
            reader_proxy_timeout,
            writer_proxy_timeout,
            writer_is_local: lsm_role_is_writer(),
            include_writer_for_reads,
            buffer,
        })
    }
}

fn gateway_env_first(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        std::env::var(name)
            .ok()
            .and_then(|value| normalize_gateway_url(&value))
    })
}

fn gateway_env_list(names: &[&str]) -> Vec<String> {
    let mut urls = Vec::new();
    for name in names {
        let Ok(value) = std::env::var(name) else {
            continue;
        };
        for raw in value.split(',') {
            if let Some(url) = normalize_gateway_url(raw) {
                if !urls.contains(&url) {
                    urls.push(url);
                }
            }
        }
        if !urls.is_empty() {
            break;
        }
    }
    urls
}

fn normalize_gateway_url(value: &str) -> Option<String> {
    let trimmed = value.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn gateway_env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
        .unwrap_or(default)
}

fn gateway_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn gateway_env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn gateway_buffer_config_from_env() -> Option<GatewayBufferConfig> {
    let dir = gateway_env_first(&["HELIX_GATEWAY_BUFFER_DIR", "HELIX_WRITE_QUEUE_DIR"])?;
    let max_entries = gateway_env_usize(
        "HELIX_GATEWAY_BUFFER_MAX_ENTRIES",
        GATEWAY_BUFFER_MAX_ENTRIES,
    );
    let max_bytes = gateway_env_u64("HELIX_GATEWAY_BUFFER_MAX_BYTES", GATEWAY_BUFFER_MAX_BYTES);
    let max_request_bytes = gateway_env_usize(
        "HELIX_GATEWAY_BUFFER_MAX_REQUEST_BYTES",
        crate::protocol::request::max_body_bytes(),
    );
    let replay_interval = Duration::from_millis(gateway_env_u64(
        "HELIX_GATEWAY_BUFFER_REPLAY_MS",
        GATEWAY_BUFFER_REPLAY_MS,
    ));

    Some(GatewayBufferConfig {
        dir: PathBuf::from(dir),
        max_entries,
        max_bytes,
        max_request_bytes,
        replay_interval,
    })
}

fn gateway_retryable_reader_status(status: u16) -> bool {
    // A newly-created collection can legitimately be absent on an eventually
    // fresh reader for a few seconds. Fall through to another target/writer so
    // first search after onboarding does not surface the replica's 404.
    status == 404 || status == 429 || status >= 500
}

/// Statuses that also mark the reader replica unhealthy, on top of falling
/// through to the next target. Only replica-scoped conditions qualify: 404s
/// and generic 500s are usually collection-scoped (absent-on-fresh-reader,
/// broken storage for one name) and repeat identically on every replica, so
/// ejecting the reader for them just drains the healthy pool and shifts all
/// reads onto the writer.
fn gateway_ejectable_reader_status(status: u16) -> bool {
    matches!(status, 429 | 502 | 503 | 504)
}

fn gateway_retryable_writer_status(status: u16) -> bool {
    matches!(status, 429 | 502 | 503 | 504) || status >= 500
}

fn has_gateway_bypass_header(request: &Request) -> bool {
    request.headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case(GATEWAY_BYPASS_HEADER) && value.trim() == GATEWAY_BYPASS_VALUE
    })
}

impl CloudGatewayRouter {
    fn new(config: CloudGatewayConfig) -> Self {
        let buffer = config
            .buffer
            .clone()
            .map(GatewayDurableBuffer::new)
            .map(Arc::new);
        let router = Self {
            config,
            next_reader: AtomicU64::new(0),
            unhealthy_readers: Mutex::new(HashMap::new()),
            buffer,
        };
        router.spawn_buffer_replay_worker();
        router
    }

    fn spawn_buffer_replay_worker(&self) {
        if cfg!(test) {
            return;
        }
        let Some(buffer) = self.buffer.as_ref().map(Arc::clone) else {
            return;
        };
        let writer_url = self.config.writer_url.clone();
        let spawn_result = thread::Builder::new()
            .name("helix-gateway-buffer-replay".to_string())
            .spawn(move || loop {
                match buffer.drain_once(|request| proxy_http_request(&writer_url, request)) {
                    Ok(report) => {
                        if report.delivered > 0 || report.retained > 0 {
                            metrics::counter!("helix_gateway_buffer_replay_total", "result" => "delivered")
                                .increment(report.delivered as u64);
                            metrics::counter!("helix_gateway_buffer_replay_total", "result" => "retained")
                                .increment(report.retained as u64);
                        }
                    }
                    Err(err) => {
                        tracing::warn!(error = %err, "cloud gateway buffer replay scan failed");
                    }
                }
                thread::sleep(buffer.replay_interval());
            });
        if let Err(err) = spawn_result {
            tracing::warn!(error = %err, "failed to spawn cloud gateway buffer replay worker");
        }
    }

    fn from_env() -> Option<Self> {
        CloudGatewayConfig::from_env().map(Self::new)
    }

    fn target_plan(&self, request: &Request, now: Instant) -> Option<Vec<GatewayTarget>> {
        if has_gateway_bypass_header(request) {
            return None;
        }
        if !is_cluster_managed_request(request) {
            return None;
        }

        let class = route_class(&request.method, &request.path);
        if class == RouteClass::Probe {
            return None;
        }
        if class.is_write_like() {
            return Some(vec![self.writer_target()]);
        }

        let readers = self.healthy_reader_urls(now);
        metrics::gauge!("helix_gateway_healthy_readers").set(readers.len() as f64);
        let mut targets = Vec::new();
        if !readers.is_empty() {
            let start = self.next_reader.fetch_add(1, Ordering::Relaxed) as usize;
            for offset in 0..readers.len().min(self.config.max_reader_attempts) {
                let url = readers[(start + offset) % readers.len()].clone();
                targets.push(GatewayTarget {
                    kind: GatewayTargetKind::Reader,
                    base_url: url,
                });
            }
            if self.config.include_writer_for_reads {
                targets.push(self.writer_target());
            }
        }
        if targets.is_empty() {
            targets.push(self.writer_target());
        }
        Some(targets)
    }

    fn mark_reader_unhealthy(&self, base_url: &str, now: Instant) {
        let unhealthy_until = now + self.config.reader_eviction;
        let mut guard = self
            .unhealthy_readers
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        guard.insert(base_url.to_string(), unhealthy_until);
    }

    fn healthy_reader_urls(&self, now: Instant) -> Vec<String> {
        let mut guard = self
            .unhealthy_readers
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        guard.retain(|_, unhealthy_until| *unhealthy_until > now);
        self.config
            .reader_urls
            .iter()
            .filter(|url| !guard.contains_key(*url))
            .cloned()
            .collect()
    }

    fn writer_target(&self) -> GatewayTarget {
        GatewayTarget {
            kind: GatewayTargetKind::Writer,
            base_url: self.config.writer_url.clone(),
        }
    }

    fn proxy_timeout(&self, kind: GatewayTargetKind) -> Duration {
        match kind {
            GatewayTargetKind::Reader => self.config.reader_proxy_timeout,
            GatewayTargetKind::Writer => self.config.writer_proxy_timeout,
        }
    }

    fn buffered_write_response(
        &self,
        request: &Request,
        reason: &'static str,
    ) -> Result<Option<Response>, GraphError> {
        let Some(buffer) = &self.buffer else {
            return Ok(None);
        };
        let ack = buffer.enqueue(request)?;
        metrics::counter!(
            "helix_gateway_buffer_enqueue_total",
            "reason" => reason,
            "duplicate" => ack.duplicate.to_string()
        )
        .increment(1);
        Ok(Some(gateway_buffered_response(ack, reason)))
    }
}

fn gateway_buffered_response(ack: GatewayBufferAck, reason: &'static str) -> Response {
    let mut response = Response::new();
    response.status = 202;
    response
        .headers
        .insert("Content-Type".to_string(), "application/json".to_string());
    response
        .headers
        .insert("X-Helix-Gateway-Buffered".to_string(), "true".to_string());
    response
        .headers
        .insert("X-Helix-Gateway-Buffer-Id".to_string(), ack.id.clone());
    response.headers.insert(
        "X-Helix-Gateway-Buffer-Duplicate".to_string(),
        ack.duplicate.to_string(),
    );
    response.body = format!(
        "{{\"status\":\"queued\",\"id\":\"{}\",\"duplicate\":{},\"reason\":\"{}\"}}",
        ack.id, ack.duplicate, reason
    )
    .into_bytes();
    response
}

struct RuntimeHandle {
    peer_addresses: HashMap<u64, String>,
    proposal_ids: AtomicU64,
    commands: Sender<RuntimeCommand>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

/// Header used for inter-node Raft authentication.
pub(crate) const RAFT_SECRET_HEADER: &str = "X-Raft-Secret";

struct HttpRaftTransport {
    client: reqwest::blocking::Client,
    /// Shared secret sent on every outgoing Raft RPC (if configured).
    secret: Option<String>,
}

impl HttpRaftTransport {
    fn new(secret: Option<String>) -> Self {
        Self {
            client: reqwest::blocking::Client::builder()
                .connect_timeout(Duration::from_millis(200))
                .timeout(Duration::from_millis(500))
                .build()
                .expect("reqwest blocking client must build"),
            secret,
        }
    }

    /// Build a request with optional auth header.
    fn authed_post(&self, url: String, body: Vec<u8>) -> reqwest::blocking::RequestBuilder {
        let req = self
            .client
            .post(url)
            .header("Content-Type", "application/octet-stream")
            .body(body);
        match &self.secret {
            Some(s) => req.header(RAFT_SECRET_HEADER, s.as_str()),
            None => req,
        }
    }
}

impl RaftTransport for HttpRaftTransport {
    fn send_message(&self, target: &str, message: &Message) -> Result<(), String> {
        // Fire-and-forget: spawn a thread so the Raft tick loop isn't blocked
        // waiting for slow/unreachable peers. Raft retries on the next tick
        // anyway, so losing one message is harmless.
        let url = format!(
            "{}{}",
            target.trim_end_matches('/'),
            INTERNAL_RAFT_MESSAGE_PATH
        );
        let body = message.encode_to_vec();
        let client = self.client.clone();
        let secret = self.secret.clone();
        thread::spawn(move || {
            let req = client
                .post(&url)
                .header("Content-Type", "application/octet-stream")
                .body(body);
            let req = match &secret {
                Some(s) => req.header(RAFT_SECRET_HEADER, s.as_str()),
                None => req,
            };
            if let Err(err) = req.send() {
                tracing::error!(target = %url, error = %err, "raft async send failed");
            }
        });
        Ok(())
    }

    fn forward_proposal(&self, target: &str, mutation: &ReplicatedMutation) -> Result<(), String> {
        // forward_proposal must be synchronous: the caller needs to know
        // whether the leader accepted the proposal.
        let url = format!(
            "{}{}",
            target.trim_end_matches('/'),
            INTERNAL_RAFT_PROPOSE_PATH
        );
        let body = bincode::serialize(mutation).map_err(|e| e.to_string())?;
        let response = self
            .authed_post(url, body)
            .send()
            .map_err(|e| e.to_string())?;
        if response.status().is_success() {
            Ok(())
        } else {
            Err(format!(
                "raft forward failed with status {}",
                response.status()
            ))
        }
    }
}

impl RuntimeHandle {
    fn new(
        collections: Arc<CollectionManager>,
        config: Config,
        transport: Arc<dyn RaftTransport>,
    ) -> Result<Self, GraphError> {
        let raft_cfg = &config.graph_config.raft;
        let node_id = raft_cfg.node_id.ok_or_else(|| {
            GraphError::New("Raft enabled but graph_config.raft.node_id is missing".into())
        })?;
        let peer_addresses: HashMap<u64, String> = raft_cfg
            .peers
            .iter()
            .map(|peer| (peer.id, peer.address.clone()))
            .collect();
        if !peer_addresses.contains_key(&node_id) {
            return Err(GraphError::New(format!(
                "Raft peer list does not include local node {}",
                node_id
            )));
        }

        let (tx, rx) = flume::unbounded();
        let peers: Vec<u64> = peer_addresses.keys().copied().collect();
        let state_path = raft_state_path(collections.data_dir(), node_id);
        let thread_transport = Arc::clone(&transport);
        let thread_collections = Arc::clone(&collections);
        let thread_config = config.clone();
        let thread_peer_addresses = peer_addresses.clone();
        let handle = thread::Builder::new()
            .name(format!("raft-node-{}", node_id))
            .spawn(move || {
                run_raft_loop(
                    node_id,
                    peers,
                    state_path,
                    rx,
                    thread_peer_addresses,
                    thread_transport,
                    thread_collections,
                    thread_config,
                )
            })
            .map_err(|e| GraphError::New(format!("failed to spawn raft thread: {}", e)))?;

        Ok(Self {
            peer_addresses,
            proposal_ids: AtomicU64::new(1),
            commands: tx,
            thread: Mutex::new(Some(handle)),
        })
    }

    fn next_proposal_id(&self) -> u64 {
        self.proposal_ids.fetch_add(1, Ordering::SeqCst)
    }

    fn propose_local(&self, mutation: ReplicatedMutation) -> Result<(), String> {
        let (tx, rx) = flume::bounded(1);
        let proposal = Proposal {
            id: self.next_proposal_id(),
            data: bincode::serialize(&mutation).map_err(|e| e.to_string())?,
        };
        self.commands
            .send(RuntimeCommand::Propose {
                proposal,
                response: tx,
            })
            .map_err(|e| e.to_string())?;
        match rx
            .recv_timeout(PROPOSE_TIMEOUT)
            .map_err(|e| e.to_string())?
        {
            RuntimeProposalResult::Applied(result) => result,
            RuntimeProposalResult::NotLeader { leader_id } => Err(match leader_id {
                Some(id) => format!("not leader; current leader is {}", id),
                None => "not leader; current leader unknown".to_string(),
            }),
        }
    }

    fn linearizable_read(&self) -> Result<(), String> {
        let (tx, rx) = flume::bounded(1);
        self.commands
            .send(RuntimeCommand::ReadBarrier {
                context: self.next_proposal_id().to_be_bytes().to_vec(),
                response: tx,
            })
            .map_err(|e| e.to_string())?;
        match rx.recv_timeout(READ_TIMEOUT).map_err(|e| e.to_string())? {
            RuntimeReadResult::Ready(result) => result,
            RuntimeReadResult::NotLeader { leader_id } => Err(match leader_id {
                Some(id) => format!("not leader; current leader is {}", id),
                None => "not leader; current leader unknown".to_string(),
            }),
        }
    }

    fn receive_message(&self, message: Message) -> Result<(), String> {
        self.commands
            .send(RuntimeCommand::Step { message })
            .map_err(|e| e.to_string())
    }

    fn status(&self) -> Result<ReplicationStatus, String> {
        let (tx, rx) = flume::bounded(1);
        self.commands
            .send(RuntimeCommand::Status { response: tx })
            .map_err(|e| e.to_string())?;
        rx.recv_timeout(Duration::from_secs(2))
            .map_err(|e| e.to_string())
    }
}

impl Drop for RuntimeHandle {
    fn drop(&mut self) {
        let _ = self.commands.send(RuntimeCommand::Shutdown);
        if let Ok(mut handle) = self.thread.lock() {
            if let Some(handle) = handle.take() {
                let _ = handle.join();
            }
        }
    }
}

/// A no-op transport used when Raft is disabled. Avoids constructing a
/// reqwest client so that `ReplicationManager` can be safely dropped
/// inside an async context.
struct NoopRaftTransport;

impl RaftTransport for NoopRaftTransport {
    fn send_message(&self, _target: &str, _message: &Message) -> Result<(), String> {
        Err("raft is not enabled".into())
    }
    fn forward_proposal(
        &self,
        _target: &str,
        _mutation: &ReplicatedMutation,
    ) -> Result<(), String> {
        Err("raft is not enabled".into())
    }
}

impl ReplicationManager {
    pub fn new(collections: Arc<CollectionManager>, config: Config) -> Result<Self, GraphError> {
        // Only create the HTTP transport when Raft is enabled. The blocking
        // reqwest client owns an internal tokio Runtime; constructing it inside
        // an async context can panic on drop.
        let transport: Arc<dyn RaftTransport> = if config.graph_config.raft.enabled {
            Arc::new(HttpRaftTransport::new(config.raft_secret()))
        } else {
            Arc::new(NoopRaftTransport)
        };
        Self::new_with_transport(collections, config, transport)
    }

    /// Returns the configured Raft shared secret (if any).
    /// Used by Raft endpoint handlers to validate incoming requests.
    pub fn raft_secret(&self) -> Option<String> {
        self.config.raft_secret()
    }

    pub(crate) fn new_with_transport(
        collections: Arc<CollectionManager>,
        config: Config,
        transport: Arc<dyn RaftTransport>,
    ) -> Result<Self, GraphError> {
        spawn_optimizer_backlog_sweeper(Arc::clone(&collections), config.clone());
        let runtime = if config.graph_config.raft.enabled {
            Some(Arc::new(RuntimeHandle::new(
                Arc::clone(&collections),
                config.clone(),
                Arc::clone(&transport),
            )?))
        } else {
            None
        };
        Ok(Self {
            collections,
            config,
            runtime,
            transport,
            cloud_gateway: CloudGatewayRouter::from_env().map(Arc::new),
        })
    }

    pub fn status(&self) -> Result<ReplicationStatus, GraphError> {
        match &self.runtime {
            Some(runtime) => runtime.status().map_err(GraphError::from),
            None => Ok(ReplicationStatus {
                enabled: false,
                node_id: None,
                leader_id: None,
                term: 0,
                is_leader: true,
                commit_index: 0,
                applied_index: 0,
                first_index: 0,
                last_index: 0,
                snapshot_index: 0,
            }),
        }
    }

    pub fn prepare_cluster_request(
        &self,
        request: &Request,
    ) -> Result<Option<Response>, GraphError> {
        if let Some(response) = self.prepare_cloud_gateway_request(request)? {
            return Ok(Some(response));
        }

        let Some(runtime) = &self.runtime else {
            return Ok(None);
        };
        if !is_cluster_managed_request(request) {
            return Ok(None);
        }

        if let Ok(status) = runtime.status() {
            if status.is_leader {
                if is_linearizable_read_request(request) {
                    return match runtime.linearizable_read() {
                        Ok(()) => Ok(None),
                        Err(err) => Ok(Some(service_unavailable_response(&err))),
                    };
                }
                return Ok(None);
            }
            if let Some(leader_id) = status.leader_id {
                if let Some(target) = runtime.peer_addresses.get(&leader_id) {
                    return proxy_http_request(target, request).map(Some);
                }
            }
        }

        let deadline = Instant::now() + LEADER_WAIT_TIMEOUT;
        while Instant::now() < deadline {
            thread::sleep(Duration::from_millis(100));
            let status = runtime.status().map_err(GraphError::from)?;
            if status.is_leader {
                if is_linearizable_read_request(request) {
                    return match runtime.linearizable_read() {
                        Ok(()) => Ok(None),
                        Err(err) => Ok(Some(service_unavailable_response(&err))),
                    };
                }
                return Ok(None);
            }
            if let Some(leader_id) = status.leader_id {
                if let Some(target) = runtime.peer_addresses.get(&leader_id) {
                    return proxy_http_request(target, request).map(Some);
                }
            }
        }

        Ok(Some(service_unavailable_response(
            "raft leader unavailable",
        )))
    }

    fn prepare_cloud_gateway_request(
        &self,
        request: &Request,
    ) -> Result<Option<Response>, GraphError> {
        let Some(gateway) = &self.cloud_gateway else {
            return Ok(None);
        };
        let Some(targets) = gateway.target_plan(request, Instant::now()) else {
            return Ok(None);
        };
        let class = route_class(&request.method, &request.path);

        let last_index = targets.len().saturating_sub(1);
        for (idx, target) in targets.iter().enumerate() {
            // The production gateway is embedded in the writer process. Serve
            // writer targets locally instead of blocking a gateway thread on
            // an HTTP call back into the same bounded server thread pool.
            if target.kind == GatewayTargetKind::Writer && gateway.config.writer_is_local {
                metrics::counter!(
                    "helix_gateway_local_writer_total",
                    "reason" => if class.is_write_like() {
                        "write"
                    } else {
                        "read_fallback"
                    }
                )
                .increment(1);
                return Ok(None);
            }
            let started = Instant::now();
            match proxy_http_request_with_timeout(
                &target.base_url,
                request,
                gateway.proxy_timeout(target.kind),
            ) {
                Ok(mut response) => {
                    let retryable_reader = target.kind == GatewayTargetKind::Reader
                        && gateway_retryable_reader_status(response.status);
                    observe_gateway_proxy_attempt(
                        target.kind,
                        if retryable_reader {
                            "retryable_status"
                        } else {
                            "response"
                        },
                        started.elapsed(),
                    );
                    if target.kind == GatewayTargetKind::Writer
                        && class.is_write_like()
                        && gateway_retryable_writer_status(response.status)
                    {
                        if let Some(buffered) =
                            gateway.buffered_write_response(request, "writer_retryable_status")?
                        {
                            tracing::warn!(
                                target = %target.base_url,
                                status = response.status,
                                "cloud gateway writer returned retryable status; request buffered"
                            );
                            return Ok(Some(buffered));
                        }
                    }
                    if retryable_reader && idx < last_index {
                        if gateway_ejectable_reader_status(response.status) {
                            gateway.mark_reader_unhealthy(&target.base_url, Instant::now());
                            metrics::counter!(
                                "helix_gateway_reader_ejections_total",
                                "reason" => "retryable_status"
                            )
                            .increment(1);
                            tracing::warn!(
                                target = %target.base_url,
                                status = response.status,
                                "cloud gateway reader returned retryable status; evicting temporarily"
                            );
                        } else {
                            tracing::debug!(
                                target = %target.base_url,
                                status = response.status,
                                "cloud gateway reader returned collection-scoped status; trying next target"
                            );
                        }
                        continue;
                    }
                    response.headers.insert(
                        "X-Helix-Gateway-Target-Role".to_string(),
                        target.kind.as_label().to_string(),
                    );
                    return Ok(Some(response));
                }
                Err(err) => {
                    observe_gateway_proxy_attempt(
                        target.kind,
                        "transport_error",
                        started.elapsed(),
                    );
                    if target.kind == GatewayTargetKind::Writer && class.is_write_like() {
                        if let Some(buffered) =
                            gateway.buffered_write_response(request, "writer_proxy_error")?
                        {
                            tracing::warn!(
                                target = %target.base_url,
                                error = %err,
                                "cloud gateway writer proxy failed; request buffered"
                            );
                            return Ok(Some(buffered));
                        }
                    }
                    if target.kind == GatewayTargetKind::Reader && idx < last_index {
                        gateway.mark_reader_unhealthy(&target.base_url, Instant::now());
                        metrics::counter!(
                            "helix_gateway_reader_ejections_total",
                            "reason" => "transport_error"
                        )
                        .increment(1);
                        match reader_proxy_failure_log_class(
                            gateway.config.writer_is_local,
                            &targets,
                            idx,
                        ) {
                            ReaderProxyFailureLogClass::LocalWriterFallback => {
                                tracing::debug!(
                                    target = %target.base_url,
                                    error = %err,
                                    "cloud gateway reader proxy failed; evicting temporarily before local writer fallback"
                                );
                            }
                            ReaderProxyFailureLogClass::PotentialClientFailure => {
                                tracing::warn!(
                                    target = %target.base_url,
                                    error = %err,
                                    "cloud gateway reader proxy failed; evicting temporarily"
                                );
                            }
                        }
                        continue;
                    }
                    tracing::warn!(
                        target = %target.base_url,
                        role = target.kind.as_label(),
                        error = %err,
                        "cloud gateway proxy failed"
                    );
                    return Ok(Some(service_unavailable_response(
                        "helix gateway upstream unavailable",
                    )));
                }
            }
        }

        Ok(Some(service_unavailable_response(
            "helix gateway upstream unavailable",
        )))
    }

    pub fn apply(&self, mutation: ReplicatedMutation) -> Result<(), GraphError> {
        if let Some(runtime) = &self.runtime {
            self.apply_via_raft(runtime, mutation)
        } else {
            apply_mutation(&self.collections, &self.config, &mutation)
        }
    }

    /// Apply a graph-ingest batch and return the number of records actually
    /// persisted. The batch is atomic: a returned `Ok(n)` means all `n` ops
    /// committed; any failure rolls the batch back and returns `Err` (callers
    /// MUST treat that as zero applied, not a partial success). This lets the
    /// ingest handlers report a verified applied count rather than blindly
    /// echoing the request length.
    ///
    /// When Raft is active the proposal round-trip cannot carry a count back, so
    /// a successful proposal (whole-batch atomic apply on the leader) reports
    /// `ops.len()`; the local (non-Raft) path returns the count straight from
    /// the apply.
    pub fn apply_ingest(
        &self,
        collection: String,
        ops: Vec<ReplicatedIngestOp>,
    ) -> Result<usize, GraphError> {
        if let Some(runtime) = &self.runtime {
            let op_count = ops.len();
            self.apply_via_raft(runtime, ReplicatedMutation::IngestBatch { collection, ops })?;
            Ok(op_count)
        } else {
            apply_ingest_batch(&self.collections, &collection, &ops)
        }
    }

    pub fn propose_internal(&self, mutation: ReplicatedMutation) -> Result<(), GraphError> {
        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| GraphError::New("raft is not enabled".into()))?;
        runtime.propose_local(mutation).map_err(GraphError::from)
    }

    pub fn receive_raft_message(&self, message: Message) -> Result<(), GraphError> {
        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| GraphError::New("raft is not enabled".into()))?;
        runtime.receive_message(message).map_err(GraphError::from)
    }

    fn apply_via_raft(
        &self,
        runtime: &Arc<RuntimeHandle>,
        mutation: ReplicatedMutation,
    ) -> Result<(), GraphError> {
        if let Ok(status) = runtime.status() {
            if status.is_leader {
                return runtime.propose_local(mutation).map_err(GraphError::from);
            }
            if let Some(leader_id) = status.leader_id {
                if let Some(target) = runtime.peer_addresses.get(&leader_id) {
                    return self
                        .transport
                        .forward_proposal(target, &mutation)
                        .map_err(GraphError::from);
                }
            }
        }

        let deadline = Instant::now() + LEADER_WAIT_TIMEOUT;
        while Instant::now() < deadline {
            thread::sleep(Duration::from_millis(100));
            let status = runtime.status().map_err(GraphError::from)?;
            if status.is_leader {
                return runtime.propose_local(mutation).map_err(GraphError::from);
            }
            if let Some(leader_id) = status.leader_id {
                if let Some(target) = runtime.peer_addresses.get(&leader_id) {
                    return self
                        .transport
                        .forward_proposal(target, &mutation)
                        .map_err(GraphError::from);
                }
            }
        }

        Err(GraphError::New("raft leader unavailable".into()))
    }
}

fn observe_gateway_proxy_attempt(
    kind: GatewayTargetKind,
    outcome: &'static str,
    elapsed: Duration,
) {
    let role = kind.as_label();
    metrics::counter!(
        "helix_gateway_proxy_attempts_total",
        "role" => role,
        "outcome" => outcome
    )
    .increment(1);
    metrics::histogram!(
        "helix_gateway_proxy_duration_ms",
        "role" => role,
        "outcome" => outcome
    )
    .record(elapsed.as_secs_f64() * 1000.0);
}

fn run_raft_loop(
    node_id: u64,
    peers: Vec<u64>,
    state_path: PathBuf,
    commands: Receiver<RuntimeCommand>,
    peer_addresses: HashMap<u64, String>,
    transport: Arc<dyn RaftTransport>,
    collections: Arc<CollectionManager>,
    config: Config,
) {
    let mut node = match RaftNode::restore_from_path(node_id, peers.clone(), &state_path) {
        Ok(node) => node,
        Err(err) => {
            tracing::error!(node_id, error = %err, "raft restore failed");
            return;
        }
    };
    let apply_collections = Arc::clone(&collections);
    let apply_config = config.clone();
    node.set_apply_fn(move |payload| {
        let mutation: ReplicatedMutation =
            bincode::deserialize(payload).map_err(|e| e.to_string())?;
        apply_mutation(&apply_collections, &apply_config, &mutation).map_err(|e| e.to_string())
    });
    let snapshot_collections = Arc::clone(&collections);
    node.set_snapshot_fn(move |snapshot_data| {
        if snapshot_data.is_empty() {
            return Ok(());
        }
        snapshot_collections
            .restore_raft_cluster_snapshot(snapshot_data)
            .map_err(|e| e.to_string())
    });

    if peers.len() == 1 {
        let _ = node.campaign();
    }

    let mut pending = HashMap::<u64, Sender<RuntimeProposalResult>>::new();
    let mut pending_reads = HashMap::<Vec<u8>, PendingRead>::new();
    let snapshot_entries = config.raft_snapshot_entries();
    let snapshot_catchup_entries = config.raft_snapshot_catchup_entries();

    loop {
        match commands.recv_timeout(Duration::from_millis(50)) {
            Ok(RuntimeCommand::Propose { proposal, response }) => {
                if node.is_leader() {
                    match node.propose(proposal.clone()) {
                        Ok(_) => {
                            pending.insert(proposal.id, response);
                        }
                        Err(err) => {
                            let _ = response.send(RuntimeProposalResult::Applied(Err(err)));
                        }
                    }
                } else {
                    let leader_id = match node.leader_id() {
                        0 => None,
                        id => Some(id),
                    };
                    let _ = response.send(RuntimeProposalResult::NotLeader { leader_id });
                }
            }
            Ok(RuntimeCommand::ReadBarrier { context, response }) => {
                if node.is_leader() {
                    match node.request_read_index(context.clone()) {
                        Ok(_) => {
                            pending_reads.insert(
                                context,
                                PendingRead {
                                    required_index: None,
                                    response,
                                },
                            );
                        }
                        Err(err) => {
                            let _ = response.send(RuntimeReadResult::Ready(Err(err)));
                        }
                    }
                } else {
                    let leader_id = match node.leader_id() {
                        0 => None,
                        id => Some(id),
                    };
                    let _ = response.send(RuntimeReadResult::NotLeader { leader_id });
                }
            }
            Ok(RuntimeCommand::Step { message }) => {
                if let Err(err) = node.step(message) {
                    tracing::error!(node_id, error = %err, "raft step failed");
                }
            }
            Ok(RuntimeCommand::Status { response }) => {
                let _ = response.send(ReplicationStatus {
                    enabled: true,
                    node_id: Some(node_id),
                    leader_id: match node.leader_id() {
                        0 => None,
                        id => Some(id),
                    },
                    term: node.term(),
                    is_leader: node.is_leader(),
                    commit_index: node.commit_index(),
                    applied_index: node.last_applied,
                    first_index: node.first_index(),
                    last_index: node.last_index(),
                    snapshot_index: node.snapshot_index(),
                });
            }
            Ok(RuntimeCommand::Shutdown) => break,
            Err(flume::RecvTimeoutError::Timeout) => {}
            Err(flume::RecvTimeoutError::Disconnected) => break,
        }

        if node.should_tick() {
            node.tick();
            node.mark_ticked();
        }

        let outcome = match node.process_ready() {
            Ok(outcome) => outcome,
            Err(err) => {
                tracing::error!(node_id, error = %err, "raft process_ready failed");
                break;
            }
        };
        if outcome.should_persist {
            if let Err(err) = node.persist_to_path(&state_path) {
                tracing::error!(node_id, error = %err, "raft persist failed");
            }
        }
        for message in outcome.messages {
            if message.to == node_id {
                if let Err(err) = node.step(message) {
                    tracing::error!(node_id, error = %err, "raft self-step failed");
                }
                continue;
            }
            if let Some(target) = peer_addresses.get(&message.to) {
                if let Err(err) = transport.send_message(target, &message) {
                    tracing::error!(
                        node_id,
                        to = message.to,
                        target = %target,
                        error = %err,
                        "raft send failed"
                    );
                }
            }
        }
        for AppliedProposal { id, result, .. } in outcome.applied {
            if let Some(response) = pending.remove(&id) {
                let _ = response.send(RuntimeProposalResult::Applied(result));
            }
        }
        for read_state in outcome.read_states {
            if let Some(pending_read) = pending_reads.get_mut(read_state.request_ctx.as_slice()) {
                pending_read.required_index = Some(read_state.index);
            }
        }
        fail_pending_reads_if_not_leader(&node, &mut pending_reads);
        drain_ready_reads(&node, &mut pending_reads);

        if maybe_snapshot_and_compact(
            &mut node,
            &collections,
            snapshot_entries,
            snapshot_catchup_entries,
        ) {
            if let Err(err) = node.persist_to_path(&state_path) {
                tracing::error!(
                    node_id,
                    error = %err,
                    "raft persist failed after snapshot"
                );
            }
        }
    }
}

fn maybe_snapshot_and_compact(
    node: &mut RaftNode,
    collections: &Arc<CollectionManager>,
    snapshot_entries: u64,
    snapshot_catchup_entries: u64,
) -> bool {
    let next_snapshot_index = node.snapshot_index().saturating_add(snapshot_entries);
    if node.last_applied < next_snapshot_index {
        return false;
    }

    let snapshot_data = match collections.create_raft_cluster_snapshot() {
        Ok(snapshot_data) => snapshot_data,
        Err(err) => {
            tracing::error!(error = %err, "raft cluster snapshot creation failed");
            return false;
        }
    };
    if let Err(err) = node.create_snapshot(node.last_applied, snapshot_data) {
        tracing::error!(error = %err, "raft snapshot creation failed");
        return false;
    }

    let compact_index = node
        .last_applied
        .saturating_sub(snapshot_catchup_entries)
        .max(node.first_index().saturating_sub(1));
    if compact_index >= node.first_index() {
        node.compact_to(compact_index);
    }

    true
}

fn drain_ready_reads(node: &RaftNode, pending_reads: &mut HashMap<Vec<u8>, PendingRead>) {
    let ready_contexts: Vec<Vec<u8>> = pending_reads
        .iter()
        .filter_map(|(context, pending)| {
            pending
                .required_index
                .filter(|required| node.last_applied >= *required)
                .map(|_| context.clone())
        })
        .collect();

    for context in ready_contexts {
        if let Some(pending) = pending_reads.remove(&context) {
            let _ = pending.response.send(RuntimeReadResult::Ready(Ok(())));
        }
    }
}

fn fail_pending_reads_if_not_leader(
    node: &RaftNode,
    pending_reads: &mut HashMap<Vec<u8>, PendingRead>,
) {
    if node.is_leader() {
        return;
    }

    let leader_id = match node.leader_id() {
        0 => None,
        id => Some(id),
    };
    for (_, pending) in pending_reads.drain() {
        let _ = pending
            .response
            .send(RuntimeReadResult::NotLeader { leader_id });
    }
}

fn is_linearizable_read_request(request: &Request) -> bool {
    let path = request
        .path
        .split_once('?')
        .map(|(path, _)| path)
        .unwrap_or(request.path.as_str());

    match request.method.as_str() {
        "GET" => path == "/collections" || path.starts_with("/collections/"),
        "POST" => {
            path == "/v1/collections/list"
                || path == "/v1/collections/stats"
                || path.starts_with("/v1/graph/")
                || path.ends_with("/points/search")
                || path.ends_with("/points/scroll")
                || path.ends_with("/points/query")
                || path.ends_with("/snapshots")
        }
        _ => false,
    }
}

fn is_cluster_managed_request(request: &Request) -> bool {
    let path = request
        .path
        .split_once('?')
        .map(|(path, _)| path)
        .unwrap_or(request.path.as_str());

    if path == "/health" || path == "/ready" || path.starts_with("/_raft/") {
        return false;
    }

    path == "/collections" || path.starts_with("/collections/") || path.starts_with("/v1/")
}

fn proxy_http_request(target: &str, request: &Request) -> Result<Response, GraphError> {
    proxy_http_request_with_timeout(target, request, Duration::from_secs(30))
}

fn proxy_http_request_with_timeout(
    target: &str,
    request: &Request,
    timeout: Duration,
) -> Result<Response, GraphError> {
    let method = reqwest::Method::from_bytes(request.method.as_bytes()).map_err(|e| {
        GraphError::New(format!(
            "invalid request method '{}': {}",
            request.method, e
        ))
    })?;
    let url = format!("{}{}", target.trim_end_matches('/'), request.path);

    let mut builder = PROXY_HTTP_CLIENT.request(method, url).timeout(timeout);
    for (header, value) in &request.headers {
        if header.eq_ignore_ascii_case("host") || header.eq_ignore_ascii_case("content-length") {
            continue;
        }
        builder = builder.header(header, value);
    }
    builder = builder.header(GATEWAY_BYPASS_HEADER, GATEWAY_BYPASS_VALUE);
    if !request.body.is_empty() {
        builder = builder.body(request.body.clone());
    }

    let upstream = builder
        .send()
        .map_err(|e| GraphError::New(format!("failed to proxy request to leader: {}", e)))?;
    let status = upstream.status().as_u16();
    let headers = upstream.headers().clone();
    let body = upstream
        .bytes()
        .map_err(|e| GraphError::New(format!("failed reading proxied response: {}", e)))?;

    let mut response = Response::new();
    response.status = status;
    response.headers.clear();
    // Filter hop-by-hop headers that would conflict with our own framing.
    // Response::send() always writes its own Content-Length, so passing through
    // the upstream's would produce duplicate/conflicting headers (invalid HTTP).
    const HOP_BY_HOP: &[&str] = &[
        "content-length",
        "transfer-encoding",
        "connection",
        "keep-alive",
    ];
    for (header, value) in headers.iter() {
        let name = header.as_str().to_ascii_lowercase();
        if HOP_BY_HOP.contains(&name.as_str()) {
            continue;
        }
        if let Ok(value) = value.to_str() {
            response
                .headers
                .insert(header.as_str().to_string(), value.to_string());
        }
    }
    response.body = body.to_vec();
    Ok(response)
}

fn service_unavailable_response(message: &str) -> Response {
    let mut response = Response::new();
    response.status = 503;
    response.body = message.as_bytes().to_vec();
    response
}

fn raft_state_path(data_dir: &std::path::Path, node_id: u64) -> PathBuf {
    data_dir.join("raft").join(format!("node-{}.bin", node_id))
}

fn apply_mutation(
    collections: &Arc<CollectionManager>,
    config: &Config,
    mutation: &ReplicatedMutation,
) -> Result<(), GraphError> {
    let result = match mutation {
        ReplicatedMutation::CreateCollection {
            name,
            vectors,
            sparse_vectors,
            hnsw_overrides,
        } => apply_create_collection(
            collections,
            config,
            name,
            vectors,
            sparse_vectors,
            hnsw_overrides.as_ref(),
        ),
        ReplicatedMutation::DeleteCollection { name } => collections.drop_collection(name),
        ReplicatedMutation::UpsertPoints { collection, points } => {
            apply_upsert_points(collections, config, collection, points)
        }
        ReplicatedMutation::DeletePoints { collection, ids } => {
            apply_delete_points(collections, collection, ids)
        }
        ReplicatedMutation::CreatePayloadIndex {
            collection,
            field_name,
            schema,
        } => {
            let resolved_collection = collections.resolve_alias(collection);
            let point_mutation_gate = collections.point_mutation_gate(&resolved_collection);
            let _point_mutation_guard = lock_point_mutation_gate(
                &point_mutation_gate,
                &resolved_collection,
                "create_payload_index",
            );
            let storage = collections.get_collection(&resolved_collection)?;
            storage
                .enqueue_payload_index_build(&resolved_collection, field_name, schema.clone(), true)
                .map(|_| ())
        }
        ReplicatedMutation::DeletePayloadIndex {
            collection,
            field_name,
        } => {
            let storage = collections.get_collection(collection)?;
            storage.delete_payload_index(field_name)
        }
        ReplicatedMutation::IngestBatch { collection, ops } => {
            apply_ingest_batch(collections, collection, ops).map(|_| ())
        }
        ReplicatedMutation::UpdateCollection {
            name,
            vectors,
            sparse_vectors,
        } => apply_update_collection(collections, config, name, vectors, sparse_vectors),
    };

    if result.is_ok() {
        if let Some(collection) = algorithm_cache_collection_for_mutation(mutation) {
            algorithms::invalidate_algorithm_cache(collection);
        }
    }

    result
}

fn algorithm_cache_collection_for_mutation(mutation: &ReplicatedMutation) -> Option<&str> {
    match mutation {
        ReplicatedMutation::CreateCollection { name, .. }
        | ReplicatedMutation::DeleteCollection { name }
        | ReplicatedMutation::UpdateCollection { name, .. } => Some(name),
        ReplicatedMutation::UpsertPoints { collection, .. }
        | ReplicatedMutation::DeletePoints { collection, .. }
        | ReplicatedMutation::IngestBatch { collection, .. } => Some(collection),
        ReplicatedMutation::CreatePayloadIndex { .. }
        | ReplicatedMutation::DeletePayloadIndex { .. } => None,
    }
}

fn apply_create_collection(
    collections: &Arc<CollectionManager>,
    config: &Config,
    name: &str,
    vectors: &HashMap<String, NamedVectorConfig>,
    sparse_vectors: &HashMap<String, SparseVectorConfig>,
    hnsw_overrides: Option<&HnswOverrides>,
) -> Result<(), GraphError> {
    let storage = collections.get_or_create_collection(name)?;
    // On the LSM backend the metadata/config write path (`put_metadata` →
    // `put_heed`) `unreachable!`s, so the heed `set_*_metadata` /
    // `set_hnsw_overrides` calls below must instead be routed through the
    // backend seam (`*_be`). The segment-DB registration
    // (`create_vector_index` / `create_sparse_index`) still runs against the
    // heed `graph_env` on LSM (the dense-segment lifecycle is heed-backed even
    // on an LSM collection), so we keep the exclusive heed txn for it and only
    // divert the metadata persistence. The merged HNSW overrides are computed
    // in-memory and passed to index creation by value, so deferring their
    // *persistence* to after the heed txn commits does not change index
    // creation. The LMDB path is unchanged (writes happen inside the heed txn).
    let on_lsm = storage.backend.kind() == BackendKind::Lsm;

    if on_lsm {
        let merged_overrides = hnsw_overrides.cloned();
        let hnsw_config = HNSWConfig::with_overrides(
            config.vector_config.m,
            config.vector_config.ef_construction,
            config.vector_config.ef_search,
            merged_overrides.as_ref(),
        );

        for (vec_name, vector_config) in vectors {
            if let Some(existing) = storage.named_vectors.get_config(vec_name) {
                if existing != *vector_config {
                    return Err(GraphError::New(format!(
                        "Named vector '{}' already exists with a different config",
                        vec_name
                    )));
                }
                continue;
            }
            storage
                .named_vectors
                .load_vector_index_lsm(
                    storage.collection_path(),
                    vec_name,
                    vector_config.clone(),
                    hnsw_config.clone(),
                    None,
                )
                .map_err(GraphError::from)?;
        }

        for (sp_name, sp_config) in sparse_vectors {
            if let Some(existing) = storage.named_vectors.get_sparse_config(sp_name) {
                if existing != *sp_config {
                    return Err(GraphError::New(format!(
                        "Sparse vector '{}' already exists with a different config",
                        sp_name
                    )));
                }
                continue;
            }
            storage
                .named_vectors
                .load_sparse_index_lsm(sp_name, sp_config.clone())
                .map_err(GraphError::from)?;
        }

        storage.with_write_backend(|w| {
            if let Some(merged) = merged_overrides.as_ref() {
                storage.set_hnsw_overrides_be(w, merged)?;
            }
            if !vectors.is_empty() {
                storage.set_named_vectors_metadata_be(w, storage.named_vectors.list_vectors())?;
                storage.set_dense_vector_spaces_metadata_be(
                    w,
                    storage.named_vectors.list_dense_vector_spaces(),
                )?;
            }
            if !sparse_vectors.is_empty() {
                storage.set_sparse_vectors_metadata_be(
                    w,
                    storage.named_vectors.list_sparse_vectors(),
                )?;
            }
            Ok(())
        })?;
        return Ok(());
    }

    let (dense_changed, sparse_changed, merged_overrides, overrides_changed) = storage
        .with_exclusive_write_txn(|txn| {
            let mut dense_changed = false;
            let mut sparse_changed = false;
            let mut overrides_changed = false;

            // Persist overrides up-front so they apply to ALL index creations in
            // this call AND survive restart via the normal open path. Merge with
            // whatever's already on disk so callers can supply partial overrides
            // (e.g. just `ef`) without clobbering previously-set `m`. The read
            // (`get_hnsw_overrides`) is backend-routed and works on LSM.
            let merged_overrides = if let Some(new_ov) = hnsw_overrides {
                let mut merged = storage.get_hnsw_overrides(txn)?.unwrap_or_default();
                if new_ov.m.is_some() {
                    merged.m = new_ov.m;
                }
                if new_ov.ef_construction.is_some() {
                    merged.ef_construction = new_ov.ef_construction;
                }
                if new_ov.ef.is_some() {
                    merged.ef = new_ov.ef;
                }
                if on_lsm {
                    overrides_changed = true;
                } else {
                    storage.set_hnsw_overrides(txn, &merged)?;
                }
                Some(merged)
            } else {
                storage.get_hnsw_overrides(txn)?
            };

            for (vec_name, vector_config) in vectors {
                if let Some(existing) = storage.named_vectors.get_config(vec_name) {
                    if existing != *vector_config {
                        return Err(GraphError::New(format!(
                            "Named vector '{}' already exists with a different config",
                            vec_name
                        )));
                    }
                    continue;
                }
                storage
                    .named_vectors
                    .create_vector_index(
                        storage.lmdb_env()?,
                        txn,
                        vec_name,
                        vector_config.clone(),
                        HNSWConfig::with_overrides(
                            config.vector_config.m,
                            config.vector_config.ef_construction,
                            config.vector_config.ef_search,
                            merged_overrides.as_ref(),
                        ),
                    )
                    .map_err(GraphError::from)?;
                dense_changed = true;
            }

            for (sp_name, sp_config) in sparse_vectors {
                if let Some(existing) = storage.named_vectors.get_sparse_config(sp_name) {
                    if existing != *sp_config {
                        return Err(GraphError::New(format!(
                            "Sparse vector '{}' already exists with a different config",
                            sp_name
                        )));
                    }
                    continue;
                }
                storage
                    .named_vectors
                    .create_sparse_index(storage.lmdb_env()?, txn, sp_name, sp_config.clone())
                    .map_err(GraphError::from)?;
                sparse_changed = true;
            }

            if !on_lsm {
                if dense_changed {
                    storage
                        .set_named_vectors_metadata(txn, storage.named_vectors.list_vectors())?;
                    storage.set_dense_vector_spaces_metadata(
                        txn,
                        storage.named_vectors.list_dense_vector_spaces(),
                    )?;
                }
                if sparse_changed {
                    storage.set_sparse_vectors_metadata(
                        txn,
                        storage.named_vectors.list_sparse_vectors(),
                    )?;
                }
            }
            Ok((
                dense_changed,
                sparse_changed,
                merged_overrides,
                overrides_changed,
            ))
        })?;

    let publish_dense_metadata = on_lsm && (!vectors.is_empty() || dense_changed);
    let publish_sparse_metadata = on_lsm && (!sparse_vectors.is_empty() || sparse_changed);

    // LSM: persist the config/metadata through the backend seam now that the
    // heed txn (segment-DB registration) has committed.
    if overrides_changed || publish_dense_metadata || publish_sparse_metadata {
        storage.with_write_backend(|w| {
            if overrides_changed {
                if let Some(merged) = merged_overrides.as_ref() {
                    storage.set_hnsw_overrides_be(w, merged)?;
                }
            }
            if publish_dense_metadata {
                storage.set_named_vectors_metadata_be(w, storage.named_vectors.list_vectors())?;
                storage.set_dense_vector_spaces_metadata_be(
                    w,
                    storage.named_vectors.list_dense_vector_spaces(),
                )?;
            }
            if publish_sparse_metadata {
                storage.set_sparse_vectors_metadata_be(
                    w,
                    storage.named_vectors.list_sparse_vectors(),
                )?;
            }
            Ok(())
        })?;
    }
    Ok(())
}

/// PATCH /collections/{name} — add new named vectors to an existing collection.
/// Reuses the same logic as create for individual vector spaces but requires the collection to already exist.
fn apply_update_collection(
    collections: &Arc<CollectionManager>,
    config: &Config,
    name: &str,
    vectors: &HashMap<String, NamedVectorConfig>,
    sparse_vectors: &HashMap<String, SparseVectorConfig>,
) -> Result<(), GraphError> {
    let storage = collections.get_collection(name)?;
    if storage.backend.kind() == BackendKind::Lsm {
        let overrides = {
            let r = storage
                .backend
                .begin_read()
                .map_err(|e| GraphError::New(e.to_string()))?;
            storage.get_hnsw_overrides_be(&r)?
        };
        let mut dense_changed = false;
        let mut sparse_changed = false;

        for (vec_name, vector_config) in vectors {
            if let Some(existing) = storage.named_vectors.get_config(vec_name) {
                if existing != *vector_config {
                    return Err(GraphError::New(format!(
                        "Cannot change config of existing vector '{}'; dimension or distance mismatch",
                        vec_name
                    )));
                }
                continue;
            }
            storage
                .named_vectors
                .create_vector_index_lsm(
                    storage.collection_path(),
                    vec_name,
                    vector_config.clone(),
                    HNSWConfig::with_overrides(
                        config.vector_config.m,
                        config.vector_config.ef_construction,
                        config.vector_config.ef_search,
                        overrides.as_ref(),
                    ),
                )
                .map_err(GraphError::from)?;
            dense_changed = true;
        }

        for (sp_name, sp_config) in sparse_vectors {
            if let Some(existing) = storage.named_vectors.get_sparse_config(sp_name) {
                if existing != *sp_config {
                    return Err(GraphError::New(format!(
                        "Cannot change config of existing sparse vector '{}'",
                        sp_name
                    )));
                }
                continue;
            }
            storage
                .named_vectors
                .create_sparse_index_lsm(sp_name, sp_config.clone())
                .map_err(GraphError::from)?;
            sparse_changed = true;
        }

        if dense_changed || sparse_changed {
            storage.with_write_backend(|w| {
                if dense_changed {
                    storage
                        .set_named_vectors_metadata_be(w, storage.named_vectors.list_vectors())?;
                    storage.set_dense_vector_spaces_metadata_be(
                        w,
                        storage.named_vectors.list_dense_vector_spaces(),
                    )?;
                }
                if sparse_changed {
                    storage.set_sparse_vectors_metadata_be(
                        w,
                        storage.named_vectors.list_sparse_vectors(),
                    )?;
                }
                Ok(())
            })?;
        }
        return Ok(());
    }

    storage.with_exclusive_write_txn(|txn| {
        let mut dense_changed = false;
        let mut sparse_changed = false;

        let overrides = storage.get_hnsw_overrides(txn)?;

        for (vec_name, vector_config) in vectors {
            if let Some(existing) = storage.named_vectors.get_config(vec_name) {
                if existing != *vector_config {
                    return Err(GraphError::New(format!(
                        "Cannot change config of existing vector '{}'; dimension or distance mismatch",
                        vec_name
                    )));
                }
                continue; // idempotent
            }
            storage
                .named_vectors
                .create_vector_index(
                    storage.lmdb_env()?,
                    txn,
                    vec_name,
                    vector_config.clone(),
                    HNSWConfig::with_overrides(
                        config.vector_config.m,
                        config.vector_config.ef_construction,
                        config.vector_config.ef_search,
                        overrides.as_ref(),
                    ),
                )
                .map_err(GraphError::from)?;
            dense_changed = true;
        }

        for (sp_name, sp_config) in sparse_vectors {
            if let Some(existing) = storage.named_vectors.get_sparse_config(sp_name) {
                if existing != *sp_config {
                    return Err(GraphError::New(format!(
                        "Cannot change config of existing sparse vector '{}'",
                        sp_name
                    )));
                }
                continue;
            }
            storage
                .named_vectors
                .create_sparse_index(storage.lmdb_env()?, txn, sp_name, sp_config.clone())
                .map_err(GraphError::from)?;
            sparse_changed = true;
        }

        if dense_changed {
            storage.set_named_vectors_metadata(txn, storage.named_vectors.list_vectors())?;
            storage.set_dense_vector_spaces_metadata(
                txn,
                storage.named_vectors.list_dense_vector_spaces(),
            )?;
        }
        if sparse_changed {
            storage.set_sparse_vectors_metadata(txn, storage.named_vectors.list_sparse_vectors())?;
        }
        Ok(())
    })
}

enum UpsertTxnResult {
    Applied(HashMap<String, usize>),
    NeedsExclusive,
}

fn dense_batch_counts(points: &[ReplicatedPoint]) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for point in points {
        for vec_name in point.vectors.keys() {
            *counts.entry(vec_name.clone()).or_default() += 1;
        }
    }
    counts
}

/// Record per-step latency for `apply_upsert_points_in_txn` into the
/// `helix_upsert_step_ms` histogram. The 80-556 s holds we logged via
/// track-caller diagnostics tell us *the chunk* is slow but not which
/// of the per-point sub-ops (existence lookup, graph node upsert,
/// vector delete, dense append, sparse insert) is the burner. Phase 0
/// of the Layer 2 spec calls for exactly this instrumentation so we can
/// either fix a single bad sub-op or confirm sharding is necessary.
#[inline]
fn record_upsert_step(step: &'static str, start: Instant) {
    if crate::telemetry::hot_path_metrics_enabled() {
        metrics::histogram!("helix_upsert_step_ms", "step" => step)
            .record(start.elapsed().as_millis() as f64);
    }
}

/// Record per-step latency for `apply_delete_points` so live long-txn warnings
/// can be attributed to dense-vector cleanup, sparse cleanup, node/payload-index
/// deletion, or LMDB commit/reclaim pressure instead of only the whole chunk.
#[inline]
fn record_delete_step(step: &'static str, start: Option<Instant>) {
    if let Some(start) = start {
        metrics::histogram!("helix_delete_step_ms", "step" => step)
            .record(start.elapsed().as_millis() as f64);
    }
}

fn sparse_metadata_flush_budget() -> usize {
    std::env::var("HELIX_SPARSE_METADATA_FLUSH_BUDGET")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(1024)
}

fn sparse_metadata_flush_hard_budget() -> usize {
    std::env::var("HELIX_SPARSE_METADATA_FLUSH_HARD_BUDGET")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(32_768)
        .max(sparse_metadata_flush_budget())
}

fn sparse_metadata_flush_hard_pending() -> usize {
    std::env::var("HELIX_SPARSE_METADATA_FLUSH_HARD_PENDING")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(50_000)
}

fn sparse_metadata_flush_max_duration() -> Duration {
    let millis = std::env::var("HELIX_SPARSE_METADATA_FLUSH_MAX_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(150);
    Duration::from_millis(millis)
}

fn sparse_metadata_flush_hard_max_duration() -> Duration {
    let millis = std::env::var("HELIX_SPARSE_METADATA_FLUSH_HARD_MAX_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(500);
    Duration::from_millis(millis).max(sparse_metadata_flush_max_duration())
}

/// Reserved payload key that holds the xxh64 content hash of the
/// parts of a point that affect vector/sparse index state. Presence
/// lets a repeat upsert short-circuit index rewrites. Payload-only
/// fields may still be rewritten on the node without deleting or
/// re-appending vectors. Absence on stored nodes (legacy writes) is
/// handled by treating the point as a mismatch and proceeding with a
/// normal upsert; the new hash is stamped on the way out so the next
/// call can short-circuit.
const CONTENT_HASH_KEY: &str = "__helix_content_hash__";
const CONTENT_HASH_PAYLOAD_ONLY_KEYS: &[&str] = &[
    "caller_point_id",
    "_pending_prune",
    "_pending_prune_at",
    // Display / graph-coordinate fields. CE can refresh these without
    // changing the dense/sparse index content for a point.
    "start_line",
    "end_line",
    "start_col",
    "end_col",
];
const CONTENT_HASH_METADATA_KEY: &str = "metadata";
/// Subkeys of the top-level `metadata` payload value that are written by
/// CE ingest/backfill jobs but are not used as part of the index-relevant
/// surface (they get refreshed on every upsert via the payload-only update
/// path, so consumers still see fresh values). Stripping them from the
/// content hash lets unchanged-document re-ingests skip the dense delete +
/// dense append + sparse upsert + HNSW submission.
const CONTENT_HASH_METADATA_PAYLOAD_ONLY_SUBKEYS: &[&str] = &[
    "ingested_at",
    "churn_count",
    "author_count",
    "_graph_backfilled",
    "_graph_backfilled_version",
    "_pseudo_backfilled",
    // host_path is the developer's local filesystem path
    // (e.g. /Users/jopad/Downloads/...) and container_path embeds a
    // content-addressed slug that flips every ingest run. Both are
    // unindexed per CE's tests/test_qdrant_delete_paths.py:57 — display
    // only, never used in filter clauses, never mutated by set_payload.
    "host_path",
    "container_path",
    // File/branch bookkeeping and display coordinates. These are filter or
    // presentation metadata, not vector/sparse content.
    "file_hash",
    "file_size_bytes",
    "last_modified_at",
    "indexed_branch",
    "git_branches",
    "start_line",
    "end_line",
    "start_col",
    "end_col",
];

fn content_hash_skip_enabled() -> bool {
    std::env::var("HELIX_CONTENT_HASH_SKIP")
        .ok()
        .map(|v| !matches!(v.as_str(), "0" | "false" | "FALSE" | "off" | ""))
        .unwrap_or(true)
}

/// Stable 64-bit hash of the parts of a `ReplicatedPoint` that actually
/// affect stored index state (index-relevant payload, dense vectors,
/// sparse vectors).
/// The doc_id is intentionally excluded — we only use this hash to decide
/// whether to skip a write, and two upserts against the same doc_id with
/// identical index content *are* equivalent. Payload-only operational
/// fields are excluded from this hash and handled by a node-only update
/// path when they change.
///
/// Determinism is load-bearing: the same logical input must yield the
/// same hash across processes and HashMap insertion orders. We hash by
/// sorting keys / term ids before feeding bytes into xxh64. Float bits
/// go through `f32::to_bits` / `f64::to_bits` so +0.0/-0.0 and NaN
/// payloads hash consistently (NaN is already rejected at sparse
/// ingress validation, so the NaN-bit-pattern corner is theoretical).
/// Quantization mode applied to dense and sparse float values before
/// hashing. Embedders (fastembed, ONNX, CPU-fp32) are known to produce
/// bit-different floats across runs on identical input — a ULP or two
/// of drift is enough to flip the sha bits and make every re-upsert
/// look like a content change. Quantizing to f16 keeps ~11 bits of
/// mantissa, which is more precision than the downstream int8 storage
/// anyway, so hash equality under f16 matches semantic equality for
/// CE's purposes.
///
/// `None` keeps full f32 bit precision (old behavior); use when
/// debugging determinism. Default is `F16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HashQuantization {
    /// Bit-exact f32 hashing. No tolerance for embedder drift.
    Bits,
    /// Round each float to f16, hash the 16-bit pattern. ~3-4 decimal
    /// digits of precision retained. Absorbs embedder noise.
    F16,
}

fn content_hash_quantization() -> HashQuantization {
    match std::env::var("HELIX_CONTENT_HASH_QUANTIZE")
        .ok()
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("bits") | Some("f32") | Some("exact") => HashQuantization::Bits,
        _ => HashQuantization::F16,
    }
}

#[inline]
fn f32_to_quantized_bytes(f: f32, mode: HashQuantization) -> [u8; 4] {
    match mode {
        HashQuantization::Bits => f.to_bits().to_le_bytes(),
        HashQuantization::F16 => {
            // IEEE 754 binary16 encode with round-to-nearest-even.
            // Subnormals and NaN are bucketed to 0 and a canonical NaN
            // bit pattern so embedder noise around zero collapses and
            // NaN flips (SparseVector::validate already rejects NaN but
            // dense vectors don't validate) can't produce different
            // hashes for equivalent nonsense.
            let bits16 = f32_to_f16_bits(f);
            // Widen to 4 bytes so the hash stream length stays stable
            // across modes (otherwise switching modes breaks comparison
            // with previously stamped hashes).
            [bits16 as u8, (bits16 >> 8) as u8, 0, 0]
        }
    }
}

/// Pure f32 -> f16 bit converter. Avoid pulling in a crate dep when the
/// IEEE encoding is ~15 lines. Based on the standard round-to-nearest-
/// even conversion used in half-precision codecs.
#[inline]
fn f32_to_f16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32;
    let mut mant = (bits & 0x007f_ffff) as i32;

    if exp == 0xff {
        // NaN or Inf -> canonical f16 NaN/Inf
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    exp = exp - 127 + 15;
    if exp >= 0x1f {
        // Overflow -> Inf
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            // Too small -> signed zero
            return sign;
        }
        // Subnormal
        mant |= 0x0080_0000;
        let shift = 14 - exp;
        let round_bit = 1 << (shift - 1);
        let mut result = mant >> shift;
        if mant & round_bit != 0 && (mant & ((round_bit << 1) - 1)) != 0 {
            result += 1;
        }
        return sign | result as u16;
    }
    // Normal: round-to-nearest-even on the low 13 mantissa bits
    let round_bit: i32 = 1 << 12;
    let low_mask: i32 = (round_bit << 1) - 1;
    if mant & round_bit != 0 && (mant & low_mask) != 0 {
        mant += round_bit;
        if mant & 0x0080_0000 != 0 {
            mant = 0;
            exp += 1;
            if exp >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

fn compute_content_hash(point: &ReplicatedPoint) -> u64 {
    use std::hash::Hasher;
    use twox_hash::XxHash64;
    let mut hasher = XxHash64::with_seed(0);
    let q = content_hash_quantization();

    // Payload: sort keys for determinism. Nested Values (Array, Object)
    // recurse with the same rules.
    fn hash_payload_map<H: Hasher>(h: &mut H, m: &HashMap<String, crate::protocol::value::Value>) {
        let mut keys: Vec<&String> = m.keys().collect();
        keys.sort();
        h.write_u8(0xA1);
        h.write_u64(keys.len() as u64);
        for k in keys {
            h.write_u8(0xA2);
            h.write_u64(k.len() as u64);
            h.write(k.as_bytes());
            hash_value(h, &m[k]);
        }
    }

    fn hash_value<H: Hasher>(h: &mut H, v: &crate::protocol::value::Value) {
        use crate::protocol::value::Value;
        match v {
            Value::Empty => h.write_u8(0x00),
            Value::Boolean(b) => {
                h.write_u8(0x01);
                h.write_u8(u8::from(*b));
            }
            Value::String(s) => {
                h.write_u8(0x02);
                h.write_u64(s.len() as u64);
                h.write(s.as_bytes());
            }
            Value::F32(f) => {
                h.write_u8(0x03);
                h.write_u32(f.to_bits());
            }
            Value::F64(f) => {
                h.write_u8(0x04);
                h.write_u64(f.to_bits());
            }
            Value::I8(i) => {
                h.write_u8(0x05);
                h.write_i8(*i);
            }
            Value::I16(i) => {
                h.write_u8(0x06);
                h.write_i16(*i);
            }
            Value::I32(i) => {
                h.write_u8(0x07);
                h.write_i32(*i);
            }
            Value::I64(i) => {
                h.write_u8(0x08);
                h.write_i64(*i);
            }
            Value::U8(u) => {
                h.write_u8(0x09);
                h.write_u8(*u);
            }
            Value::U16(u) => {
                h.write_u8(0x0A);
                h.write_u16(*u);
            }
            Value::U32(u) => {
                h.write_u8(0x0B);
                h.write_u32(*u);
            }
            Value::U64(u) => {
                h.write_u8(0x0C);
                h.write_u64(*u);
            }
            Value::U128(u) => {
                h.write_u8(0x0D);
                h.write_u128(*u);
            }
            Value::Array(arr) => {
                h.write_u8(0x0E);
                h.write_u64(arr.len() as u64);
                for item in arr {
                    hash_value(h, item);
                }
            }
            Value::Object(obj) => {
                h.write_u8(0x0F);
                hash_payload_map(h, obj);
            }
        }
    }

    // Exclude payload-only operational fields from the index hash. They
    // must still be persisted on the node, but changing them alone
    // should not delete/rewrite dense or sparse vectors.
    let payload_without_marker = payload_without_content_hash_payload_only_keys(&point.payload);
    hash_payload_map(&mut hasher, &payload_without_marker);

    // Dense vectors: sort by vector name for determinism. Floats go
    // through `f32_to_quantized_bytes(q)` so embedder ULP noise
    // (common with CPU-fp32 ONNX/fastembed re-runs) doesn't flip the
    // hash when the semantic vector is effectively identical.
    let mut dense_names: Vec<&String> = point.vectors.keys().collect();
    dense_names.sort();
    hasher.write_u8(0xD1);
    hasher.write_u64(dense_names.len() as u64);
    for name in dense_names {
        hasher.write_u8(0xD2);
        hasher.write_u64(name.len() as u64);
        hasher.write(name.as_bytes());
        let v = &point.vectors[name];
        hasher.write_u64(v.len() as u64);
        for &f in v {
            hasher.write(&f32_to_quantized_bytes(f, q));
        }
    }

    // Sparse vectors: sort by name; terms inside SparseVector are
    // expected sorted by index (validate() enforces no duplicates but
    // doesn't enforce order; sort defensively). Sparse values go
    // through the same quantization — they're also f32 and subject to
    // the same embedder ULP noise on re-runs.
    let mut sparse_names: Vec<&String> = point.sparse_vectors.keys().collect();
    sparse_names.sort();
    hasher.write_u8(0xE1);
    hasher.write_u64(sparse_names.len() as u64);
    for name in sparse_names {
        hasher.write_u8(0xE2);
        hasher.write_u64(name.len() as u64);
        hasher.write(name.as_bytes());
        let sv = &point.sparse_vectors[name];
        let mut pairs: Vec<(u32, f32)> = sv
            .indices
            .iter()
            .copied()
            .zip(sv.values.iter().copied())
            .collect();
        pairs.sort_by_key(|p| p.0);
        hasher.write_u64(pairs.len() as u64);
        for (term, val) in &pairs {
            hasher.write_u32(*term);
            hasher.write(&f32_to_quantized_bytes(*val, q));
        }
    }

    hasher.finish()
}

fn payload_without_content_hash_marker(payload: &HashMap<String, Value>) -> HashMap<String, Value> {
    payload
        .iter()
        .filter(|(k, _)| k.as_str() != CONTENT_HASH_KEY)
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

fn is_content_hash_payload_only_key(key: &str) -> bool {
    key == CONTENT_HASH_KEY || CONTENT_HASH_PAYLOAD_ONLY_KEYS.contains(&key)
}

fn is_metadata_payload_only_subkey(key: &str) -> bool {
    CONTENT_HASH_METADATA_PAYLOAD_ONLY_SUBKEYS.contains(&key)
}

/// Strip operational subkeys from a `metadata` Value::Object so the content
/// hash ignores fields that are refreshed on every upsert (timestamps,
/// backfill markers, derived stats). Leaves non-Object metadata values
/// untouched so we don't change hashing for collections that happen to use
/// `metadata` as a non-object field.
fn metadata_without_payload_only_subkeys(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let filtered: HashMap<String, Value> = map
                .iter()
                .filter(|(k, _)| !is_metadata_payload_only_subkey(k.as_str()))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            Value::Object(filtered)
        }
        other => other.clone(),
    }
}

fn payload_without_content_hash_payload_only_keys(
    payload: &HashMap<String, Value>,
) -> HashMap<String, Value> {
    payload
        .iter()
        .filter(|(k, _)| !is_content_hash_payload_only_key(k.as_str()))
        .map(|(k, v)| {
            if k.as_str() == CONTENT_HASH_METADATA_KEY {
                (k.clone(), metadata_without_payload_only_subkeys(v))
            } else {
                (k.clone(), v.clone())
            }
        })
        .collect()
}

fn payloads_equal_without_content_hash_marker(
    stored_payload: &HashMap<String, Value>,
    incoming_payload: &HashMap<String, Value>,
) -> bool {
    payload_without_content_hash_marker(stored_payload)
        == payload_without_content_hash_marker(incoming_payload)
}

fn content_hash_from_properties(properties: &HashMap<String, Value>) -> Option<u64> {
    match properties.get(CONTENT_HASH_KEY) {
        Some(Value::U64(h)) => Some(*h),
        _ => None,
    }
}

fn content_hash_mismatch_log_every() -> u64 {
    env_u64("HELIX_CONTENT_HASH_MISMATCH_LOG_EVERY", 256).max(1)
}

fn first_payload_diff_key(
    stored_payload: &HashMap<String, Value>,
    incoming_payload: &HashMap<String, Value>,
) -> Option<String> {
    let stored = payload_without_content_hash_payload_only_keys(stored_payload);
    let incoming = payload_without_content_hash_payload_only_keys(incoming_payload);
    let mut keys: Vec<&String> = stored.keys().chain(incoming.keys()).collect();
    keys.sort();
    keys.dedup();
    keys.into_iter().find_map(|key| {
        let stored_value = stored.get(key);
        let incoming_value = incoming.get(key);
        if stored_value == incoming_value {
            return None;
        }
        if key == CONTENT_HASH_METADATA_KEY {
            return first_metadata_diff_key(stored_value, incoming_value);
        }
        Some(key.clone())
    })
}

fn first_metadata_diff_key(stored: Option<&Value>, incoming: Option<&Value>) -> Option<String> {
    match (stored, incoming) {
        (Some(Value::Object(stored_metadata)), Some(Value::Object(incoming_metadata))) => {
            let mut keys: Vec<&String> = stored_metadata
                .keys()
                .chain(incoming_metadata.keys())
                .collect();
            keys.sort();
            keys.dedup();
            keys.into_iter()
                .find(|key| stored_metadata.get(*key) != incoming_metadata.get(*key))
                .map(|key| format!("{CONTENT_HASH_METADATA_KEY}.{key}"))
        }
        _ => Some(CONTENT_HASH_METADATA_KEY.to_string()),
    }
}

fn observe_content_hash_mismatch(
    point: &ReplicatedPoint,
    stored_payload: Option<&HashMap<String, Value>>,
    stored_hash: Option<u64>,
    incoming_hash: u64,
) {
    let incoming_payload = payload_without_content_hash_payload_only_keys(&point.payload);
    let (reason, payload_diff_key) = match (stored_hash, stored_payload) {
        (None, _) => ("missing_hash", None),
        (Some(_), Some(stored_payload)) => {
            let stored_without_payload_only =
                payload_without_content_hash_payload_only_keys(stored_payload);
            if stored_without_payload_only == incoming_payload {
                ("payload_same", None)
            } else {
                (
                    "payload_changed",
                    first_payload_diff_key(stored_payload, &point.payload),
                )
            }
        }
        (Some(_), None) => ("missing_stored_payload", None),
    };

    metrics::counter!("helix_upsert_content_hash_mismatch_reason_total", "reason" => reason)
        .increment(1);
    metrics::histogram!("helix_upsert_content_hash_mismatch_payload_keys", "reason" => reason)
        .record(incoming_payload.len() as f64);
    metrics::histogram!("helix_upsert_content_hash_mismatch_dense_vectors", "reason" => reason)
        .record(point.vectors.len() as f64);
    metrics::histogram!("helix_upsert_content_hash_mismatch_dense_dims", "reason" => reason)
        .record(point.vectors.values().map(Vec::len).sum::<usize>() as f64);
    metrics::histogram!("helix_upsert_content_hash_mismatch_sparse_vectors", "reason" => reason)
        .record(point.sparse_vectors.len() as f64);
    metrics::histogram!("helix_upsert_content_hash_mismatch_sparse_terms", "reason" => reason)
        .record(
            point
                .sparse_vectors
                .values()
                .map(|vector| vector.indices.len())
                .sum::<usize>() as f64,
        );

    let seq = CONTENT_HASH_MISMATCH_DIAG_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    let every = content_hash_mismatch_log_every();
    if seq % every == 0 {
        tracing::info!(
            point = %format!("{:032x}", point.id),
            reason,
            stored_hash = ?stored_hash,
            incoming_hash,
            payload_keys = incoming_payload.len(),
            payload_diff_key = payload_diff_key.as_deref().unwrap_or(""),
            dense_vectors = point.vectors.len(),
            dense_dims = point.vectors.values().map(Vec::len).sum::<usize>(),
            sparse_vectors = point.sparse_vectors.len(),
            sparse_terms = point
                .sparse_vectors
                .values()
                .map(|vector| vector.indices.len())
                .sum::<usize>(),
            "content hash mismatch diagnostic"
        );
    }
}

/// Pull the stamped content hash off a stored node's payload. Returns
/// `None` when the node predates the hash stamping (treated as mismatch
/// by callers; the next upsert restamps).
fn stored_content_hash(
    storage: &HelixGraphStorage,
    txn: &RwTxn,
    point_id: u128,
) -> Result<Option<u64>, GraphError> {
    // Read through the backend seam, observing this write txn's own buffered
    // writes (read-your-writes) exactly as the prior `nodes_db.get(txn, ..)`
    // through the RwTxn did. Byte-identical key on LMDB: Namespace::Nodes
    // resolves to nodes_db, keyed by the 16-byte big-endian point id.
    storage
        .backend
        .get_for_update_heed(
            txn,
            crate::helix_engine::storage_core::backend::Namespace::Nodes,
            &point_id.to_be_bytes(),
            |v| match v {
                Some(bytes) => content_hash_from_node_bytes(bytes, point_id),
                None => Ok(None),
            },
        )
        .map_err(|e| GraphError::New(e.to_string()))?
}

fn content_hash_from_node_bytes(bytes: &[u8], point_id: u128) -> Result<Option<u64>, GraphError> {
    let node = SerializedNode::decode_node(bytes, point_id)?;
    if node.label != "point" {
        return Ok(None);
    }
    Ok(content_hash_from_properties(&node.properties))
}

fn plan_dense_delete_for_upsert(
    storage: &HelixGraphStorage,
    points: &[ReplicatedPoint],
    hash_skip_on: bool,
) -> Result<DenseDeletePlan, GraphError> {
    let txn = storage.begin_resize_safe_read_txn()?;
    let mut delete_batch_ids = Vec::with_capacity(points.len());

    for point in points {
        // Read through the backend seam (RoTxn visitor). Byte-identical key on
        // LMDB: Namespace::Nodes resolves to nodes_db, keyed by the 16-byte
        // big-endian point id (Database<U128<BE>, Bytes>). Decode inside the
        // closure since the borrowed bytes cannot escape the visitor.
        let existing_node = storage
            .backend
            .get_with_heed(
                &txn,
                crate::helix_engine::storage_core::backend::Namespace::Nodes,
                &point.id.to_be_bytes(),
                |v| {
                    v.map(|bytes| SerializedNode::decode_node(bytes, point.id))
                        .transpose()
                },
            )
            .map_err(|e| GraphError::New(e.to_string()))??;
        let Some(existing_node) = existing_node else {
            continue;
        };
        if existing_node.label != "point" {
            continue;
        }
        if hash_skip_on {
            let incoming_hash = compute_content_hash(point);
            let stored = content_hash_from_properties(&existing_node.properties);
            if stored == Some(incoming_hash) {
                continue;
            }
            observe_content_hash_mismatch(
                point,
                Some(&existing_node.properties),
                stored,
                incoming_hash,
            );
        }
        delete_batch_ids.push(point.id);
    }

    storage
        .named_vectors
        .plan_delete_vectors_batch(&txn, &delete_batch_ids)
        .map_err(GraphError::from)
}

fn plan_dense_delete_for_delete(
    storage: &HelixGraphStorage,
    ids: &[u128],
) -> Result<(HashSet<u128>, DenseDeletePlan), GraphError> {
    let txn = storage.begin_resize_safe_read_txn()?;
    let mut existing_point_ids = HashSet::with_capacity(ids.len());

    for id in ids {
        // Read through the backend seam (RoTxn visitor). Byte-identical key on
        // LMDB: Namespace::Nodes -> nodes_db, 16-byte big-endian id. Decode in
        // the closure (borrowed bytes cannot escape the visitor).
        let existing_is_point = storage
            .backend
            .get_with_heed(
                &txn,
                crate::helix_engine::storage_core::backend::Namespace::Nodes,
                &id.to_be_bytes(),
                |v| {
                    v.map(|bytes| SerializedNode::decode_node(bytes, *id))
                        .transpose()
                },
            )
            .map_err(|e| GraphError::New(e.to_string()))??
            .map(|node| node.label == "point")
            .unwrap_or(false);
        if existing_is_point {
            existing_point_ids.insert(*id);
        }
    }

    let dense_delete_plan = storage
        .named_vectors
        .plan_delete_vectors_batch(&txn, ids)
        .map_err(GraphError::from)?;
    Ok((existing_point_ids, dense_delete_plan))
}

fn apply_upsert_points_in_txn(
    storage: &HelixGraphStorage,
    txn: &mut RwTxn,
    points: &[ReplicatedPoint],
    hnsw_config: HNSWConfig,
    flat_scan_threshold: usize,
    dense_delete_plan: Option<&DenseDeletePlan>,
) -> Result<HashMap<String, usize>, GraphError> {
    let mut dense_batch_counts: HashMap<String, usize> = HashMap::new();
    let mut dense_layout_changed = false;
    let hash_skip_on = content_hash_skip_enabled();
    // Chunk-wide sparse buckets keyed by sparse-vector name. Built up
    // inside the per-point loop and drained term-major in one batched
    // upsert per name AFTER the loop. See upsert_batch in
    // SparseVectorCore for the LMDB DUPSORT cursor-locality story.
    let mut sparse_buckets: HashMap<String, Vec<(u128, &SparseVector)>> = HashMap::new();

    // Batched dense-vector delete pre-pass.
    //
    // Without this, the per-point loop below would call
    // `named_vectors.delete_vector(point.id)` once per re-upserted
    // point. Per-step instrumentation showed that single call at p50
    // 174 ms / p99 688 ms — multiplied by chunk_size=64, the delete
    // pass alone consumed up to 11 s of the 14.5 s p99 we observed on
    // `lmdb_write_txn_total_ms{shared}`. The cost is the per-id
    // O(degree × levels) reverse-edge update on each neighbor block.
    // When N points in the same chunk share a neighbor (very common in
    // HNSW clusters) the per-block read-modify-write happens N times.
    //
    // The batched API folds those updates: one r-m-w per (neighbor,
    // level) regardless of how many deleted ids reference it. We do
    // the existing-point lookup + content-hash skip decision once up
    // front so we can collect only the ids that actually need deletion
    // (skip identical re-upserts, never delete a point that isn't
    // there). The per-point loop below repeats the same lookup, but
    // it's ~0 ms p99 — negligible compared to what we save.
    if let Some(plan) = dense_delete_plan {
        if !plan.is_empty() {
            let t = Instant::now();
            storage
                .named_vectors
                .delete_vectors_batch_with_plan(txn, plan)
                .map_err(GraphError::from)?;
            record_upsert_step("delete_vectors_batch", t);
        }
    } else {
        let mut delete_batch_ids: Vec<u128> = Vec::with_capacity(points.len());
        for point in points {
            // Read through the backend seam, observing this write txn's own
            // buffered writes (read-your-writes) exactly as the prior
            // `nodes_db.get(txn, ..)` through the RwTxn did. Byte-identical key:
            // Namespace::Nodes -> nodes_db, 16-byte big-endian id.
            let existing_is_point = storage
                .backend
                .get_for_update_heed(
                    txn,
                    crate::helix_engine::storage_core::backend::Namespace::Nodes,
                    &point.id.to_be_bytes(),
                    |v| {
                        v.map(|bytes| SerializedNode::decode_node(bytes, point.id))
                            .transpose()
                    },
                )
                .map_err(|e| GraphError::New(e.to_string()))??
                .map(|node| node.label == "point")
                .unwrap_or(false);
            if !existing_is_point {
                continue;
            }
            if hash_skip_on {
                let incoming_hash = compute_content_hash(point);
                let stored = stored_content_hash(storage, txn, point.id)?;
                if stored == Some(incoming_hash) {
                    // Identical re-upsert: per-point loop will continue without
                    // touching anything for this id, so don't delete its vector.
                    continue;
                }
            }
            delete_batch_ids.push(point.id);
        }
        if !delete_batch_ids.is_empty() {
            let t = Instant::now();
            storage
                .named_vectors
                .delete_vectors_batch(txn, &delete_batch_ids)
                .map_err(GraphError::from)?;
            record_upsert_step("delete_vectors_batch", t);
        }
    }

    for point in points {
        let t = Instant::now();
        // Read through the backend seam with read-your-writes (RwTxn) so the
        // decoded node observes this txn's buffered writes, exactly as the prior
        // `nodes_db.get(txn, ..)` did. Byte-identical key: Namespace::Nodes ->
        // nodes_db, 16-byte big-endian id. Decode in the closure (borrow cannot
        // escape) and return the owned SerializedNode.
        let existing_node = storage
            .backend
            .get_for_update_heed(
                txn,
                crate::helix_engine::storage_core::backend::Namespace::Nodes,
                &point.id.to_be_bytes(),
                |v| {
                    v.map(|bytes| SerializedNode::decode_node(bytes, point.id))
                        .transpose()
                },
            )
            .map_err(|e| GraphError::New(e.to_string()))??;
        let existing_is_point = existing_node
            .as_ref()
            .map(|node| node.label == "point")
            .unwrap_or(false);
        record_upsert_step("lookup_existing", t);

        // Content-hash short-circuit. When the stored point's stamped hash
        // matches the incoming point's hash, the entire upsert — node
        // rewrite, dense delete+append, sparse diff, HNSW submission — is
        // a no-op for index state. This is the biggest re-index win by
        // far once it lands: the sparse diff already skips sparse writes
        // on unchanged content, but it still pays upsert_node + the dense
        // path; content-hash skip walks away before any of it.
        let incoming_hash = if hash_skip_on {
            let t = Instant::now();
            let h = compute_content_hash(point);
            record_upsert_step("compute_content_hash", t);
            Some(h)
        } else {
            None
        };
        let mut payload_only_update = false;
        if hash_skip_on && existing_is_point {
            let t = Instant::now();
            let stored = existing_node
                .as_ref()
                .and_then(|node| content_hash_from_properties(&node.properties));
            record_upsert_step("lookup_content_hash", t);
            if stored == incoming_hash {
                if payloads_equal_without_content_hash_marker(
                    &existing_node.as_ref().unwrap().properties,
                    &point.payload,
                ) {
                    metrics::counter!("helix_upsert_content_hash_skipped_total").increment(1);
                    continue;
                }
                payload_only_update = true;
                metrics::counter!("helix_upsert_content_hash_payload_only_update_total")
                    .increment(1);
            } else {
                metrics::counter!("helix_upsert_content_hash_mismatch_total").increment(1);
            }
        } else if hash_skip_on {
            metrics::counter!("helix_upsert_content_hash_new_total").increment(1);
        }

        // Clone the payload and stamp the content hash so the next upsert
        // of this doc_id can short-circuit. Stamping goes on a fresh copy
        // to keep the hash out of `fields` (which feeds dense vector
        // metadata) — we don't want a volatile optimization key leaking
        // into payload indexes or dense sidecar props.
        let fields = point.payload.clone();
        let mut props = point.payload.clone();
        if let Some(h) = incoming_hash {
            props.insert(
                CONTENT_HASH_KEY.to_string(),
                crate::protocol::value::Value::U64(h),
            );
        }
        let node_upsert = NodeUpsert {
            id: point.id,
            label: "point".into(),
            properties: props,
        };
        let t = Instant::now();
        storage.upsert_node(txn, &node_upsert)?;
        record_upsert_step("upsert_node", t);
        if payload_only_update {
            continue;
        }

        // Per-point dense delete is now handled by the batched
        // delete_vectors_batch pre-pass above. Sparse delete still
        // happens implicitly via the sparse core's diff-aware upsert
        // below, which reads the old forward entry and only touches
        // postings that changed.

        // Insert dense vectors into the mutable flat tail only. Threshold
        // sealing and HNSW construction are handled by the background index
        // executor so hot upserts never build HNSW in the write transaction.
        let t = Instant::now();
        for (vec_name, vec_data) in &point.vectors {
            if storage
                .named_vectors
                .dense_append_to_mutable(
                    storage.lmdb_env()?,
                    txn,
                    vec_name,
                    vec_data,
                    point.id,
                    fields.clone(),
                    hnsw_config.clone(),
                    flat_scan_threshold,
                )
                .map_err(GraphError::from)?
            {
                dense_layout_changed = true;
            }
            *dense_batch_counts.entry(vec_name.clone()).or_default() += 1;
        }
        record_upsert_step("dense_append_all_vectors", t);

        // Sparse vectors are inserted in a batched, term-major pass at
        // the END of the chunk loop (after this per-point loop) — not
        // here. This lets us fold every doc's per-term puts on one
        // DUPSORT cursor descent per term instead of per (doc × term).
        // Accumulate the references; the actual upsert runs once per
        // sparse-vector name post-loop.
        for (sp_name, sp_vec) in &point.sparse_vectors {
            sparse_buckets
                .entry(sp_name.clone())
                .or_default()
                .push((point.id, sp_vec));
        }

        if !existing_is_point {
            let t = Instant::now();
            storage.adjust_metadata_counter(txn, MetadataCounter::Vectors, 1)?;
            record_upsert_step("counter_increment", t);
        }
    }

    // Batched sparse upsert pass. ONE call per sparse-vector name
    // covers every doc in the chunk that has that sparse type. Inside
    // upsert_batch the fresh-insert posts are sorted (term_id, doc_id)
    // so all puts to the same DUPSORT subtree are contiguous; pmax +
    // doc_count are merged once for the whole chunk. Diff-class docs
    // fall back to per-doc upsert inside the same call.
    if !sparse_buckets.is_empty() {
        let t = Instant::now();
        for (name, items) in &sparse_buckets {
            storage
                .named_vectors
                .upsert_sparse_batch(txn, name, items)
                .map_err(GraphError::from)?;
        }
        record_upsert_step("sparse_insert_all", t);
    }

    if dense_layout_changed {
        storage.mark_dense_layout_dirty();
    }

    // Drain per-sparse-core pending df + max into LMDB. Per-point
    // inserts/deletes now stage these in memory; this call amortizes
    // the previously per-point metadata writes across the chunk. The
    // normal path is budgeted to bound write-txn tail latency; if pending
    // metadata crosses the hard ceiling, force a full drain so pathological
    // sparse workloads cannot starve metadata indefinitely.
    let pending_sparse = storage.named_vectors.pending_sparse_metadata_count();
    metrics::gauge!("helix_sparse_metadata_pending_entries").set(pending_sparse as f64);
    let (budget, max_duration) = if pending_sparse >= sparse_metadata_flush_hard_pending() {
        metrics::counter!("helix_sparse_metadata_flush_hard_budget_total", "reason" => "hard_pending")
            .increment(1);
        (
            Some(sparse_metadata_flush_hard_budget()),
            Some(sparse_metadata_flush_hard_max_duration()),
        )
    } else {
        (
            Some(sparse_metadata_flush_budget()),
            Some(sparse_metadata_flush_max_duration()),
        )
    };

    let t = Instant::now();
    let sparse_flush = storage
        .named_vectors
        .flush_sparse_pending_metadata_budgeted(txn, budget, max_duration)
        .map_err(GraphError::from)?;
    metrics::histogram!("helix_sparse_metadata_flush_duration_ms")
        .record(t.elapsed().as_millis() as f64);
    metrics::counter!("helix_sparse_metadata_flushed_entries_total")
        .increment(sparse_flush.flushed as u64);
    metrics::gauge!("helix_sparse_metadata_pending_entries").set(sparse_flush.pending_after as f64);
    if sparse_flush.budget_exhausted {
        metrics::counter!("helix_sparse_metadata_flush_budget_exhausted_total").increment(1);
    }
    if sparse_flush.time_exhausted {
        metrics::counter!("helix_sparse_metadata_flush_time_exhausted_total").increment(1);
    }
    record_upsert_step("flush_sparse_metadata", t);
    Ok(dense_batch_counts)
}

/// Group-commit (#1) kill-switch. Enabled by default: the LSM upsert path commits
/// each chunk BUFFERED and `apply_upsert_points` issues ONE durability barrier per
/// apply (collapses `ceil(N/64)` object-store flushes to ~1). Set
/// `HELIX_LSM_GROUP_COMMIT=0` (or `false`/`off`) to fall back to a fully-durable
/// commit per chunk — an operational safety valve and the A/B benchmark control.
fn lsm_group_commit_enabled() -> bool {
    // Case/whitespace-insensitive so an operator reaching for the kill-switch
    // during an incident gets the same result from "0"/"off"/"OFF"/"False"/" no ".
    !matches!(
        std::env::var("HELIX_LSM_GROUP_COMMIT")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref(),
        Some("0") | Some("false") | Some("off") | Some("no")
    )
}

fn apply_upsert_points_lsm_chunk(
    storage: &HelixGraphStorage,
    points: &[ReplicatedPoint],
    buffered: bool,
) -> Result<HashMap<String, usize>, GraphError> {
    let mut dense_batch_counts: HashMap<String, usize> = HashMap::new();
    let hash_skip_on = content_hash_skip_enabled();

    // Group-commit (#1): when `buffered`, commit this chunk visible-but-not-yet-
    // durable; `apply_upsert_points` issues ONE `flush_durable` barrier after the
    // whole batch, so N chunks collapse to ONE durable flush and we never ack a
    // non-durable write. When disabled, commit fully durable per chunk.
    let run = |w: &mut super::backend_any::AnyWrite<'_>| -> Result<(), GraphError> {
        let mut sparse_buckets: HashMap<String, Vec<(u128, &SparseVector)>> = HashMap::new();
        for point in points {
            let existing_node = storage
                .backend
                .get_for_update(w, Namespace::Nodes, &point.id.to_be_bytes(), |v| {
                    v.map(|bytes| SerializedNode::decode_node(bytes, point.id))
                        .transpose()
                })
                .map_err(|e| GraphError::New(e.to_string()))??;
            let existing_is_point = existing_node
                .as_ref()
                .map(|node| node.label == "point")
                .unwrap_or(false);

            let incoming_hash = if hash_skip_on {
                Some(compute_content_hash(point))
            } else {
                None
            };
            let mut payload_only_update = false;
            if hash_skip_on && existing_is_point {
                let stored = existing_node
                    .as_ref()
                    .and_then(|node| content_hash_from_properties(&node.properties));
                if stored == incoming_hash {
                    if payloads_equal_without_content_hash_marker(
                        &existing_node.as_ref().unwrap().properties,
                        &point.payload,
                    ) {
                        metrics::counter!("helix_upsert_content_hash_skipped_total").increment(1);
                        continue;
                    }
                    payload_only_update = true;
                    metrics::counter!("helix_upsert_content_hash_payload_only_update_total")
                        .increment(1);
                }
            }

            let fields = point.payload.clone();
            let mut props = point.payload.clone();
            if let Some(h) = incoming_hash {
                props.insert(CONTENT_HASH_KEY.to_string(), Value::U64(h));
            }
            let node_upsert = NodeUpsert {
                id: point.id,
                label: "point".into(),
                properties: props,
            };
            storage.upsert_node_be_with_existing(w, &node_upsert, existing_node.clone())?;

            if payload_only_update {
                continue;
            }

            for (vec_name, vec_data) in &point.vectors {
                storage
                    .named_vectors
                    .dense_append_to_mutable_be(w, vec_name, vec_data, point.id, fields.clone())
                    .map_err(GraphError::from)?;
                *dense_batch_counts.entry(vec_name.clone()).or_default() += 1;
            }

            for (sp_name, sp_vec) in &point.sparse_vectors {
                sparse_buckets
                    .entry(sp_name.clone())
                    .or_default()
                    .push((point.id, sp_vec));
            }

            if !existing_is_point {
                storage.adjust_metadata_counter_be(w, MetadataCounter::Vectors, 1)?;
            }
        }

        for (name, items) in &sparse_buckets {
            storage
                .named_vectors
                .upsert_sparse_batch_be(w, name, items)
                .map_err(GraphError::from)?;
        }

        Ok(())
    };
    if buffered {
        storage.with_write_backend_buffered(run)?;
    } else {
        storage.with_write_backend(run)?;
    }

    // Drain pending sparse df/max metadata AFTER the postings batch above has
    // been committed (visible immediately; made durable by the end-of-apply
    // group-commit barrier, or per-chunk when the kill-switch is off), in its
    // own self-contained batch. Keeping it separate
    // makes it commit-safe: a failed metadata commit (e.g. SlateDB CAS conflict)
    // re-stages the deltas instead of losing them, and readers stay correct via
    // the pending overlay until the next flush succeeds.
    let pending_sparse = storage.named_vectors.pending_sparse_metadata_count();
    let (budget, max_duration) = if pending_sparse >= sparse_metadata_flush_hard_pending() {
        (
            Some(sparse_metadata_flush_hard_budget()),
            Some(sparse_metadata_flush_hard_max_duration()),
        )
    } else {
        (
            Some(sparse_metadata_flush_budget()),
            Some(sparse_metadata_flush_max_duration()),
        )
    };
    storage
        .named_vectors
        .flush_sparse_pending_metadata_budgeted_be(budget, max_duration)
        .map_err(GraphError::from)?;

    Ok(dense_batch_counts)
}

/// Per-batch upsert chunk size: how many points commit per write txn.
/// Tunable via `HELIX_UPSERT_APPLY_CHUNK`. Default 64 keeps each
/// `apply_upsert_points_in_txn` under ~250 ms for typical CE workloads
/// (1536-dim vectors + payload index puts).
fn upsert_apply_chunk_size() -> usize {
    env_usize_at_least("HELIX_UPSERT_APPLY_CHUNK", 1).unwrap_or(64)
}

/// Cap on total sparse posting count per upsert chunk. CE workloads write
/// inverted-index postings via DUPSORT during `apply_upsert_points_in_txn`,
/// and live measurements show `fresh_posting_put` averaging ~4.5 s even
/// with 16-doc chunks — sparse cost scales with total postings, not docs.
/// When a chunk's cumulative posting count would exceed this budget, the
/// chunk is split early. Input order is preserved, and at least one doc
/// always lands in the chunk so an oversized single doc doesn't stall.
///
/// `0` disables posting-budget chunking (doc-count cap only). Tunable via
/// `HELIX_UPSERT_SPARSE_POSTING_CHUNK_BUDGET`. Default 4096 caps the
/// heaviest chunks at a fraction of the prod tail observed pre-patch.
fn upsert_apply_sparse_posting_budget() -> usize {
    std::env::var("HELIX_UPSERT_SPARSE_POSTING_CHUNK_BUDGET")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4096)
}

/// Cap estimated LMDB writes per upsert write transaction. The sparse posting
/// budget bounds incoming vector volume; this budget also accounts for the
/// in-LMDB work each point can trigger: payload index puts, dense vector appends,
/// sparse DUPSORT writes, and dense delete reverse-edge patching.
///
/// `0` disables write-budget chunking. Tunable via
/// `HELIX_UPSERT_WRITE_BUDGET`. Default 32K is intentionally above normal CE
/// chunks while splitting payload-index-heavy or sparse-frontier-heavy chunks
/// before they can pin LMDB's single writer for minutes.
fn upsert_apply_write_budget() -> usize {
    std::env::var("HELIX_UPSERT_WRITE_BUDGET")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(32 * 1024)
}

#[inline]
fn point_sparse_posting_count(point: &ReplicatedPoint) -> usize {
    point
        .sparse_vectors
        .values()
        .map(|sv| sv.indices.len())
        .sum()
}

fn points_sparse_posting_count(points: &[ReplicatedPoint]) -> usize {
    points
        .iter()
        .map(point_sparse_posting_count)
        .fold(0usize, usize::saturating_add)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UpsertChunkSplitReason {
    None,
    PostingBudget,
    WriteBudget,
}

const UPSERT_PAYLOAD_INDEX_WRITE_FANOUT: usize = 2;
const UPSERT_DENSE_VECTOR_WRITE_FANOUT: usize = 4;

fn payload_index_write_count(storage: &HelixGraphStorage) -> usize {
    storage
        .payload_indices
        .read()
        .map(|indices| {
            indices
                .values()
                .filter(|handle| handle.accepts_writes())
                .count()
        })
        .unwrap_or(0)
}

fn point_estimated_write_count(
    point: &ReplicatedPoint,
    payload_index_count: usize,
    indexed_segments: usize,
) -> usize {
    // Payload indexes can remove an old entry and insert a new one. Dense flat
    // appends touch several stores (LMDB vector metadata plus sidecar data), so
    // count each named vector as a small write fanout instead of a single put.
    let node_writes = 1usize
        .saturating_add(payload_index_count.saturating_mul(UPSERT_PAYLOAD_INDEX_WRITE_FANOUT));
    let dense_inserts = point
        .vectors
        .len()
        .saturating_mul(UPSERT_DENSE_VECTOR_WRITE_FANOUT);
    let sparse_writes = point_sparse_posting_count(point).saturating_mul(2);
    let dense_delete_writes = indexed_segments
        .max(1)
        .saturating_mul(4)
        .saturating_mul(point.vectors.len().max(1));

    node_writes
        .saturating_add(dense_inserts)
        .saturating_add(sparse_writes)
        .saturating_add(dense_delete_writes)
}

/// Return the next upsert chunk length bounded by `max_docs`, sparse posting
/// count, and an estimated LMDB write budget. Always returns at least one doc
/// for non-empty input so an oversized single doc terminates the caller's loop
/// without requiring partial-point writes.
fn budgeted_upsert_chunk_len(
    points: &[ReplicatedPoint],
    max_docs: usize,
    max_postings: usize,
    max_writes: usize,
    payload_index_count: usize,
    indexed_segments: usize,
) -> (usize, UpsertChunkSplitReason) {
    if points.is_empty() {
        return (0, UpsertChunkSplitReason::None);
    }

    let max_docs = max_docs.max(1);
    let mut docs = 0usize;
    let mut postings = 0usize;
    let mut estimated_writes = 0usize;
    for (i, p) in points.iter().enumerate() {
        let p_postings = point_sparse_posting_count(p);
        let p_writes = point_estimated_write_count(p, payload_index_count, indexed_segments);
        if docs > 0 && docs + 1 > max_docs {
            return (i, UpsertChunkSplitReason::None);
        }
        if max_postings > 0 && docs > 0 && postings.saturating_add(p_postings) > max_postings {
            return (i, UpsertChunkSplitReason::PostingBudget);
        }
        if max_writes > 0 && docs > 0 && estimated_writes.saturating_add(p_writes) > max_writes {
            return (i, UpsertChunkSplitReason::WriteBudget);
        }
        docs += 1;
        postings = postings.saturating_add(p_postings);
        estimated_writes = estimated_writes.saturating_add(p_writes);
    }
    (points.len(), UpsertChunkSplitReason::None)
}

fn env_usize_at_least(name: &str, min: usize) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v >= 1)
        .map(|v| v.max(min))
}

fn delete_segment_fanout_scale(indexed_segments: usize) -> usize {
    if indexed_segments <= 1 {
        return 1;
    }

    // Scale by sqrt(fanout), not fanout. The dense delete implementation
    // batches reverse-edge updates across ids; reducing to one id per chunk
    // loses that batching and creates thousands of tiny commits on collections
    // with large retired/active segment fanout.
    let mut scale = 1usize;
    while scale.saturating_mul(scale) < indexed_segments {
        scale = scale.saturating_add(1);
    }
    scale
}

/// Per-batch delete chunk size: how many point ids are processed per chunk.
///
/// `HELIX_DELETE_APPLY_CHUNK` is an explicit final chunk-size override. When
/// unset, derive from the upsert chunk with sqrt(segment fanout) scaling plus a
/// floor so pathological segment counts do not collapse into hundreds of tiny
/// write transactions and defeat `delete_vectors_batch` coalescing.
fn delete_apply_chunk_size(indexed_segments: usize) -> usize {
    if let Some(chunk) = env_usize_at_least("HELIX_DELETE_APPLY_CHUNK", 1) {
        return chunk;
    }

    let base = upsert_apply_chunk_size();
    let floor = env_usize_at_least("HELIX_DELETE_APPLY_MIN_CHUNK", 1).unwrap_or(64);
    let scale = delete_segment_fanout_scale(indexed_segments);
    let scaled = (base / scale).max(1);
    scaled.max(floor).min(base.max(floor))
}

fn lock_point_mutation_gate<'a>(
    gate: &'a Arc<Mutex<()>>,
    collection: &str,
    operation: &'static str,
) -> std::sync::MutexGuard<'a, ()> {
    match gate.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::error!(
                collection = %collection,
                operation,
                "point mutation gate poisoned; recovering lock"
            );
            metrics::counter!(
                "helix_point_mutation_gate_poison_recovered_total",
                "operation" => operation.to_string()
            )
            .increment(1);
            poisoned.into_inner()
        }
    }
}

fn apply_upsert_points(
    collections: &Arc<CollectionManager>,
    config: &Config,
    collection: &str,
    points: &[ReplicatedPoint],
) -> Result<(), GraphError> {
    // Phase-level instrumentation: chunk_commit_ms (in async_gateway.rs) shows
    // ~30 s p50≈p99 per chunk while inner LMDB txn work measures ~150 ms.
    // The 99.5% gap lives in this function outside the txn — pin down which
    // phase by emitting `helix_apply_upsert_phase_ms{phase}`.
    let resolved_collection = collections.resolve_alias(collection);
    let point_mutation_gate = collections.point_mutation_gate(&resolved_collection);
    let _point_mutation_guard =
        lock_point_mutation_gate(&point_mutation_gate, &resolved_collection, "upsert");

    let phase_start = Instant::now();
    let storage = collections.get_collection(&resolved_collection)?;
    storage.ensure_not_degraded()?;
    let observe_hot_metrics = crate::telemetry::hot_path_metrics_enabled();
    if observe_hot_metrics {
        metrics::histogram!("helix_apply_upsert_phase_ms", "phase" => "get_collection")
            .record(phase_start.elapsed().as_secs_f64() * 1000.0);
    }

    let phase_start = Instant::now();
    let global_threshold = config.vector_flat_scan_threshold();
    let flat_scan_threshold = storage
        .named_vectors
        .effective_indexing_threshold(global_threshold);
    // Read per-collection HNSW overrides so that the upsert path uses the same
    // `m`/`ef_construction`/`ef` the index was built with. Falls back to global
    // `HELIX_HNSW_*` when no overrides set.
    let overrides = if storage.backend.kind() == BackendKind::Lsm {
        let r = storage
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))?;
        storage.get_hnsw_overrides_be(&r)?
    } else {
        let rtxn = storage.begin_resize_safe_read_txn()?;
        storage.get_hnsw_overrides(&rtxn)?
    };
    let hnsw_config = HNSWConfig::with_overrides(
        config.vector_config.m,
        config.vector_config.ef_construction,
        config.vector_config.ef_search,
        overrides.as_ref(),
    );
    if observe_hot_metrics {
        metrics::histogram!("helix_apply_upsert_phase_ms", "phase" => "setup")
            .record(phase_start.elapsed().as_secs_f64() * 1000.0);
    }

    // Chunk the batch into many short write txns. Spool batches can pack
    // dozens of upsert chunks into one apply call; committing them all in
    // a single write txn holds the LMDB writer for tens of seconds and
    // wedges every other operation on the env. With chunking, each commit
    // is bounded and the writer is released between chunks so other
    // workers (chunked merge phases, other collections' upserts) can
    // interleave.
    let chunk_size = upsert_apply_chunk_size();
    let posting_budget = upsert_apply_sparse_posting_budget();
    let write_budget = upsert_apply_write_budget();
    let payload_index_count = payload_index_write_count(&storage);
    let indexed_segments = storage.named_vectors.indexed_dense_segment_count();
    let mut combined_dense_counts: HashMap<String, usize> = HashMap::new();

    let phase_start = Instant::now();
    let hash_skip_on = content_hash_skip_enabled();
    // #1 group-commit: on LSM (and unless the kill-switch disables it) chunks
    // commit BUFFERED and one durability barrier runs after the loop.
    let lsm_group_commit = storage.backend.kind() == BackendKind::Lsm && lsm_group_commit_enabled();
    let mut chunk_start = 0usize;
    while chunk_start < points.len() {
        let (chunk_len, split_reason) = budgeted_upsert_chunk_len(
            &points[chunk_start..],
            chunk_size,
            posting_budget,
            write_budget,
            payload_index_count,
            indexed_segments,
        );
        match split_reason {
            UpsertChunkSplitReason::PostingBudget => {
                metrics::counter!("helix_upsert_chunk_split_by_posting_budget_total").increment(1);
            }
            UpsertChunkSplitReason::WriteBudget => {
                metrics::counter!("helix_upsert_chunk_split_by_write_budget_total").increment(1);
            }
            UpsertChunkSplitReason::None => {}
        }
        let chunk_end = chunk_start + chunk_len;
        let chunk = &points[chunk_start..chunk_end];
        if observe_hot_metrics {
            metrics::histogram!("helix_upsert_chunk_docs").record(chunk.len() as f64);
            metrics::histogram!("helix_upsert_chunk_sparse_postings")
                .record(points_sparse_posting_count(chunk) as f64);
            metrics::histogram!("helix_upsert_chunk_estimated_writes").record(
                chunk
                    .iter()
                    .map(|point| {
                        point_estimated_write_count(point, payload_index_count, indexed_segments)
                    })
                    .fold(0usize, usize::saturating_add) as f64,
            );
        }
        if storage.backend.kind() == BackendKind::Lsm {
            let chunk_counts = apply_upsert_points_lsm_chunk(&storage, chunk, lsm_group_commit)?;
            for (k, v) in chunk_counts {
                *combined_dense_counts.entry(k).or_default() += v;
            }
            chunk_start = chunk_end;
            continue;
        }
        let requested_chunk_counts = dense_batch_counts(chunk);
        let dense_delete_plan = plan_dense_delete_for_upsert(&storage, chunk, hash_skip_on)?;
        let chunk_outcome = storage.with_write_txn(|txn| {
            if !storage
                .named_vectors
                .dense_batch_can_append_existing_segments(
                    &requested_chunk_counts,
                    flat_scan_threshold,
                )
                .map_err(GraphError::from)?
            {
                metrics::counter!(
                    "helix_upsert_retry_exclusive_total",
                    "collection" => collection.to_string(),
                    "reason" => "dense_mutable_missing",
                )
                .increment(1);
                return Ok(UpsertTxnResult::NeedsExclusive);
            }
            apply_upsert_points_in_txn(
                &storage,
                txn,
                chunk,
                hnsw_config.clone(),
                flat_scan_threshold,
                Some(&dense_delete_plan),
            )
            .map(UpsertTxnResult::Applied)
        })?;

        let chunk_counts = match chunk_outcome {
            UpsertTxnResult::Applied(counts) => counts,
            UpsertTxnResult::NeedsExclusive => storage.with_exclusive_write_txn(|txn| {
                apply_upsert_points_in_txn(
                    &storage,
                    txn,
                    chunk,
                    hnsw_config.clone(),
                    flat_scan_threshold,
                    Some(&dense_delete_plan),
                )
            })?,
        };
        for (k, v) in chunk_counts {
            *combined_dense_counts.entry(k).or_default() += v;
        }
        chunk_start = chunk_end;
    }
    // Group-commit barrier (#1): when group-commit is on, the LSM chunk path
    // committed each chunk BUFFERED (visible, not yet durable). One flush makes
    // every buffered chunk durable in a single object-store flush BEFORE we ack —
    // N chunk commits collapse to ONE durable flush, and we never ack a
    // non-durable write. Skipped when the kill-switch is off (chunks were already
    // durable) and on LMDB.
    if lsm_group_commit {
        let durable_start = Instant::now();
        storage
            .backend
            .flush_durable()
            .map_err(|e| GraphError::New(e.to_string()))?;
        if observe_hot_metrics {
            metrics::histogram!("helix_apply_upsert_phase_ms", "phase" => "lsm_flush_durable")
                .record(durable_start.elapsed().as_secs_f64() * 1000.0);
        }
    }
    if observe_hot_metrics {
        metrics::histogram!("helix_apply_upsert_phase_ms", "phase" => "txn_loop")
            .record(phase_start.elapsed().as_secs_f64() * 1000.0);
    }
    let dense_batch_counts = combined_dense_counts;

    // Flush mmap sidecar files so search reads reflect the new vectors.
    let phase_start = Instant::now();
    storage.named_vectors.flush_mmap_stores();
    if observe_hot_metrics {
        metrics::histogram!("helix_apply_upsert_phase_ms", "phase" => "flush_mmap_stores")
            .record(phase_start.elapsed().as_secs_f64() * 1000.0);
    }

    // Record upsert timestamp for quiescence-based tail sealing.
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        storage
            .last_upsert_at
            .store(now_ms, std::sync::atomic::Ordering::Release);
    }

    // Background optimizer: split-phase, convergence-looping.
    //
    // Loops until fully convergent:
    //   Phase 1 – Seal: seal tail if above threshold OR quiesced for >1s
    //   Phase 2 – Build: prepare HNSW for all Building segments (parallel)
    //   Phase 3 – Flush + merge: brief write txn
    //   Re-check: if work remains, loop
    if flat_scan_threshold > 0 && storage.degraded_collection_error().is_none() {
        let spaces_to_check: Vec<String> = dense_batch_counts.into_keys().collect();
        if !spaces_to_check.is_empty() {
            let submit_result = INDEX_EXECUTOR.submit(IndexJob {
                collection: collection.to_string(),
                storage: Arc::downgrade(&storage),
                spaces_to_check,
                hnsw_config: hnsw_config.clone(),
                flat_scan_threshold,
            });
            if let Err(e) = submit_result {
                tracing::error!(error = %e, "failed to submit optimizer job");
                metrics::counter!("helix_index_executor_submit_rejected_total").increment(1);
            }
        }
    }

    Ok(())
}

fn apply_delete_points(
    collections: &Arc<CollectionManager>,
    collection: &str,
    ids: &[u128],
) -> Result<(), GraphError> {
    let resolved_collection = collections.resolve_alias(collection);
    let point_mutation_gate = collections.point_mutation_gate(&resolved_collection);
    let _point_mutation_guard =
        lock_point_mutation_gate(&point_mutation_gate, &resolved_collection, "delete");
    let storage = collections.get_collection(&resolved_collection)?;
    storage.ensure_not_degraded()?;

    // Chunk the per-id delete work.
    //
    // A re-index batch can include thousands of ids; the original code
    // did them all in one exclusive write txn, which held the LMDB
    // writer for tens of seconds and starved every concurrent reader on
    // this collection. Two structural fixes here:
    //
    //   (1) adaptive chunk size — `delete_vectors_batch` patches HNSW
    //       neighbors in *every* indexed dense segment, so the per-id
    //       cost is `O(indexed_segments)`. Scale the upsert-derived
    //       chunk down by segment count so the writer hold is bounded
    //       regardless of fanout.
    //
    //   (2) split the per-chunk work into two short write txns:
    //         Txn A — dense HNSW neighbor patching (one batched call)
    //         Txn B — per-id sparse delete + graph drop_node + counter
    //       Each txn releases the writer between them so queue workers
    //       interleave instead of waiting for one combined hold.
    let metrics_enabled = crate::telemetry::hot_path_metrics_enabled();
    let segs = storage.named_vectors.indexed_dense_segment_count();
    let chunk_size = delete_apply_chunk_size(segs);
    if metrics_enabled {
        metrics::histogram!("helix_delete_indexed_segments").record(segs as f64);
        metrics::histogram!("helix_delete_chunk_size_effective").record(chunk_size as f64);
    }

    for chunk in ids.chunks(chunk_size) {
        let chunk_start = metrics_enabled.then(Instant::now);
        if metrics_enabled {
            metrics::histogram!("helix_delete_chunk_ids").record(chunk.len() as f64);
        }

        // Dedupe outside any txn — pure local CPU work.
        let t = metrics_enabled.then(Instant::now);
        let mut seen_ids = HashSet::with_capacity(chunk.len());
        let mut delete_ids = Vec::with_capacity(chunk.len());
        for id in chunk {
            if seen_ids.insert(*id) {
                delete_ids.push(*id);
            }
        }
        record_delete_step("dedupe_ids", t);

        // ── Read plan: lookup_existing + dense ownership probe ──
        //
        // `lookup_existing` only reads `nodes_db`, but it must observe
        // the graph state before `drop_node` runs in Txn B. The dense
        // active-segment ownership probe is also read-only and expensive
        // on high-fanout collections, so keep both out of the LMDB writer.
        //
        // The point-mutation gate held for this call (taken above) serializes
        // this plan→execute window only against concurrent upserts/deletes on
        // the same collection — it is NOT held by the background segment
        // optimizer (`run_index_job`, which seals/builds/merges segments on
        // its own schedule and never touches this gate). A merge landing in
        // that window can retire a planned segment; `delete_vectors_batch_with_plan_be`
        // detects the active-segment mismatch and falls back to the unplanned
        // rescan, which is what actually keeps this correct rather than the gate.
        if storage.backend.kind() == BackendKind::Lsm {
            let plan_start = metrics_enabled.then(Instant::now);
            let r = storage
                .backend
                .begin_read()
                .map_err(|e| GraphError::New(e.to_string()))?;
            let mut existing_point_ids = HashSet::with_capacity(delete_ids.len());
            for id in &delete_ids {
                let existing_is_point = storage
                    .backend
                    .get_with(&r, Namespace::Nodes, &id.to_be_bytes(), |v| {
                        v.map(|bytes| SerializedNode::decode_node(bytes, *id))
                            .transpose()
                    })
                    .map_err(|e| GraphError::New(e.to_string()))??
                    .map(|node| node.label == "point")
                    .unwrap_or(false);
                if existing_is_point {
                    existing_point_ids.insert(*id);
                }
            }
            if metrics_enabled {
                metrics::histogram!("helix_delete_existing_point_ids")
                    .record(existing_point_ids.len() as f64);
            }
            let dense_delete_plan = storage
                .named_vectors
                .plan_delete_vectors_batch_be(&r, &delete_ids)
                .map_err(GraphError::from)?;
            drop(r);
            record_delete_step("lookup_existing", plan_start);

            let txn_start = metrics_enabled.then(Instant::now);
            let mut staged_tombstones = None;
            storage.with_write_backend(|w| {
                staged_tombstones = Some(
                    storage
                        .named_vectors
                        .delete_vectors_batch_with_plan_be(w, &dense_delete_plan)
                        .map_err(GraphError::from)?,
                );
                for id in &delete_ids {
                    storage
                        .named_vectors
                        .delete_sparse_vectors_be(w, *id)
                        .map_err(GraphError::from)?;
                    storage.drop_node_be(w, id)?;
                    if existing_point_ids.contains(id) {
                        storage.adjust_metadata_counter_be(w, MetadataCounter::Vectors, -1)?;
                    }
                }
                Ok(())
            })?;
            if let Some(staged) = staged_tombstones {
                storage.named_vectors.apply_delete_tombstones(&staged);
            }
            record_delete_step("chunk_lsm_delete_txn", txn_start);
            continue;
        }

        let plan_start = metrics_enabled.then(Instant::now);
        let (existing_point_ids, dense_delete_plan) =
            plan_dense_delete_for_delete(&storage, &delete_ids)?;
        if metrics_enabled {
            metrics::histogram!("helix_delete_existing_point_ids")
                .record(existing_point_ids.len() as f64);
        }
        record_delete_step("lookup_existing", plan_start);

        // ── Txn A: apply precomputed HNSW neighbor patching ──
        let txn_a_start = metrics_enabled.then(Instant::now);
        storage.with_write_txn(|txn| {
            let t = metrics_enabled.then(Instant::now);
            storage
                .named_vectors
                .delete_vectors_batch_with_plan(txn, &dense_delete_plan)
                .map_err(GraphError::from)?;
            record_delete_step("dense_delete_vectors_batch", t);
            Ok(())
        })?;
        record_delete_step("chunk_txn_a", txn_a_start);

        // ── Txn B: per-id sparse + graph drop_node + counter ──
        //
        // Idempotent w.r.t. Txn A: re-running this on a chunk whose
        // dense vectors were already removed is safe because the sparse
        // / graph / counter work is keyed on `id` and tolerates missing
        // rows. A failure in B leaves a transient state where the dense
        // index is patched but the graph node still exists; the
        // optimizer's convergence loop will re-emit the delete, or a
        // retry from the caller will land on the same id again.
        let txn_b_start = metrics_enabled.then(Instant::now);
        storage.with_write_txn(|txn| {
            for id in &delete_ids {
                let t = metrics_enabled.then(Instant::now);
                storage
                    .named_vectors
                    .delete_sparse_vectors(txn, *id)
                    .map_err(GraphError::from)?;
                record_delete_step("delete_sparse_vectors", t);

                let t = metrics_enabled.then(Instant::now);
                storage.drop_node(txn, id)?;
                record_delete_step("drop_node", t);

                if existing_point_ids.contains(id) {
                    let t = metrics_enabled.then(Instant::now);
                    storage.adjust_metadata_counter(txn, MetadataCounter::Vectors, -1)?;
                    record_delete_step("adjust_vector_counter", t);
                }
            }
            Ok(())
        })?;
        record_delete_step("chunk_txn_b", txn_b_start);

        record_delete_step("chunk_write_txn", chunk_start);
        if let Some(start) = chunk_start {
            metrics::histogram!("helix_point_delete_chunk_total_ms")
                .record(start.elapsed().as_secs_f64() * 1000.0);
        }
    }

    // The optimizer's convergence loop already runs cleanup_empty_dense_segments_deferred
    // and defers reclamation to SEGMENT_REAPER. Running it here too means every
    // delete batch holds an exclusive write txn for O(total_vectors) prefix scans
    // across all segments — measured at 100-115 s on large collections.
    // Omit it: the optimizer handles cleanup without blocking the write queue.
    Ok(())
}

/// Apply a graph-ingest batch, returning the number of ops actually persisted.
/// The whole batch is one atomic write: on success every op landed, so the
/// returned count equals `ops.len()`; on any error the batch is rolled back and
/// an `Err` is returned (callers MUST NOT report success). Surfacing the count
/// lets the ingest handlers echo a verified applied count instead of a blind
/// `req.len()`.
fn apply_ingest_batch(
    collections: &Arc<CollectionManager>,
    collection: &str,
    ops: &[ReplicatedIngestOp],
) -> Result<usize, GraphError> {
    let storage = collections.get_or_create_collection(collection)?;
    storage.ensure_not_degraded()?;
    if storage.backend.kind() == BackendKind::Lsm {
        storage.with_write_backend(|w| {
            for op in ops {
                match op {
                    ReplicatedIngestOp::Node(node) => {
                        storage.upsert_node_be(w, node)?;
                    }
                    ReplicatedIngestOp::Edge(edge) => {
                        storage.upsert_edge_be(w, edge)?;
                    }
                }
            }
            Ok(())
        })?;
        return Ok(ops.len());
    }

    storage.with_write_txn(|txn| {
        for op in ops {
            match op {
                ReplicatedIngestOp::Node(node) => {
                    storage.upsert_node(txn, node)?;
                }
                ReplicatedIngestOp::Edge(edge) => {
                    storage.upsert_edge(txn, edge)?;
                }
            }
        }
        Ok(())
    })?;
    Ok(ops.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helix_engine::graph_core::config::{
        Config, GraphConfig, RaftConfig, RaftPeerConfig, VectorConfig,
    };
    use crate::helix_engine::storage_core::storage_methods::StorageMethods;
    use std::sync::Weak;
    use tempfile::TempDir;

    static ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    fn gateway_request(method: &str, path: &str) -> Request {
        Request {
            method: method.to_string(),
            headers: HashMap::new(),
            path: path.to_string(),
            body: Vec::new(),
        }
    }

    fn gateway_router_for_tests(include_writer_for_reads: bool) -> CloudGatewayRouter {
        CloudGatewayRouter::new(CloudGatewayConfig {
            writer_url: "http://writer:8080".to_string(),
            reader_urls: vec![
                "http://reader-a:8080".to_string(),
                "http://reader-b:8080".to_string(),
            ],
            max_reader_attempts: usize::MAX,
            reader_eviction: Duration::from_secs(60),
            reader_proxy_timeout: Duration::from_secs(15),
            writer_proxy_timeout: Duration::from_secs(30),
            writer_is_local: false,
            include_writer_for_reads,
            buffer: None,
        })
    }

    fn writer_target() -> GatewayTarget {
        GatewayTarget {
            kind: GatewayTargetKind::Writer,
            base_url: "http://writer:8080".to_string(),
        }
    }

    fn reader_target(name: &str) -> GatewayTarget {
        GatewayTarget {
            kind: GatewayTargetKind::Reader,
            base_url: format!("http://{name}:8080"),
        }
    }

    #[test]
    fn reader_proxy_failure_logging_requires_a_later_local_writer_fallback() {
        let targets = vec![
            reader_target("reader-a"),
            reader_target("reader-b"),
            writer_target(),
        ];

        assert_eq!(
            reader_proxy_failure_log_class(true, &targets, 0),
            ReaderProxyFailureLogClass::LocalWriterFallback
        );
        assert_eq!(
            reader_proxy_failure_log_class(true, &targets, 1),
            ReaderProxyFailureLogClass::LocalWriterFallback
        );

        // A remote writer can still fail its own proxy attempt, so the reader
        // failure remains warning-level until success is known.
        assert_eq!(
            reader_proxy_failure_log_class(false, &targets, 0),
            ReaderProxyFailureLogClass::PotentialClientFailure
        );
        // A plan containing only later readers has no guaranteed fallback.
        assert_eq!(
            reader_proxy_failure_log_class(true, &targets[..2], 0),
            ReaderProxyFailureLogClass::PotentialClientFailure
        );
        // A writer earlier in the plan cannot rescue a later failed target.
        assert_eq!(
            reader_proxy_failure_log_class(true, &[writer_target(), reader_target("reader-a")], 1),
            ReaderProxyFailureLogClass::PotentialClientFailure
        );
    }

    #[test]
    fn cloud_gateway_routes_mutations_to_writer_only() {
        let router = gateway_router_for_tests(false);
        let request = gateway_request("PUT", "/collections/repo/points");

        let plan = router.target_plan(&request, Instant::now()).unwrap();

        assert_eq!(plan, vec![writer_target()]);
    }

    #[test]
    fn cloud_gateway_retries_every_reader_server_error() {
        assert!(gateway_retryable_reader_status(429));
        assert!(gateway_retryable_reader_status(404));
        assert!(gateway_retryable_reader_status(500));
        assert!(gateway_retryable_reader_status(502));
        assert!(gateway_retryable_reader_status(504));
        assert!(!gateway_retryable_reader_status(400));
    }

    #[test]
    fn cloud_gateway_ejects_readers_only_for_replica_scoped_statuses() {
        assert!(gateway_ejectable_reader_status(429));
        assert!(gateway_ejectable_reader_status(502));
        assert!(gateway_ejectable_reader_status(503));
        assert!(gateway_ejectable_reader_status(504));
        // Collection-scoped: identical on every replica, must not drain the
        // healthy-reader pool.
        assert!(!gateway_ejectable_reader_status(404));
        assert!(!gateway_ejectable_reader_status(500));
    }

    #[test]
    fn cloud_gateway_routes_reads_to_healthy_readers_round_robin() {
        let router = gateway_router_for_tests(false);
        let request = gateway_request("POST", "/collections/repo/points/search");
        let now = Instant::now();

        let first_plan = router.target_plan(&request, now).unwrap();
        let second_plan = router.target_plan(&request, now).unwrap();

        assert_eq!(
            first_plan,
            vec![reader_target("reader-a"), reader_target("reader-b")]
        );
        assert_eq!(
            second_plan,
            vec![reader_target("reader-b"), reader_target("reader-a")]
        );
    }

    #[test]
    fn cloud_gateway_skips_unhealthy_readers_and_falls_back_to_writer() {
        let router = gateway_router_for_tests(false);
        let request = gateway_request("POST", "/collections/repo/points/search");
        let now = Instant::now();

        router.mark_reader_unhealthy("http://reader-a:8080", now);
        let one_reader_plan = router.target_plan(&request, now).unwrap();
        assert_eq!(one_reader_plan, vec![reader_target("reader-b")]);

        router.mark_reader_unhealthy("http://reader-b:8080", now);
        let fallback_plan = router.target_plan(&request, now).unwrap();
        assert_eq!(fallback_plan, vec![writer_target()]);
    }

    #[test]
    fn cloud_gateway_can_include_writer_in_read_plan() {
        let router = gateway_router_for_tests(true);
        let request = gateway_request("GET", "/collections/repo");

        let plan = router.target_plan(&request, Instant::now()).unwrap();

        assert_eq!(
            plan,
            vec![
                reader_target("reader-a"),
                reader_target("reader-b"),
                writer_target()
            ]
        );
    }

    #[test]
    fn cloud_gateway_can_bound_reader_attempts_before_writer_fallback() {
        let mut router = gateway_router_for_tests(true);
        router.config.max_reader_attempts = 1;
        let request = gateway_request("GET", "/collections/repo");

        let plan = router.target_plan(&request, Instant::now()).unwrap();

        assert_eq!(plan, vec![reader_target("reader-a"), writer_target()]);
    }

    #[test]
    fn cloud_gateway_is_disabled_inside_reader_replicas() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _writer_url = EnvVarGuard::set("HELIX_GATEWAY_WRITER_URL", "http://writer:8080");
        let _role = EnvVarGuard::set("HELIX_LSM_ROLE", "reader");

        assert!(CloudGatewayConfig::from_env().is_none());
    }

    #[test]
    fn cloud_gateway_leaves_probes_and_raft_local() {
        let router = gateway_router_for_tests(false);

        assert!(router
            .target_plan(&gateway_request("GET", "/health"), Instant::now())
            .is_none());
        assert!(router
            .target_plan(&gateway_request("POST", "/_raft/propose"), Instant::now())
            .is_none());
    }

    #[test]
    fn cloud_gateway_bypass_header_leaves_request_local() {
        let router = gateway_router_for_tests(false);
        let mut request = gateway_request("POST", "/collections/repo/points/search");
        request.headers.insert(
            GATEWAY_BYPASS_HEADER.to_string(),
            GATEWAY_BYPASS_VALUE.to_string(),
        );

        let plan = router.target_plan(&request, Instant::now());

        assert!(plan.is_none());
    }

    #[test]
    fn cloud_gateway_buffers_writer_mutation_when_enabled() {
        let dir = TempDir::new().unwrap();
        let router = CloudGatewayRouter::new(CloudGatewayConfig {
            writer_url: "http://writer:8080".to_string(),
            reader_urls: Vec::new(),
            max_reader_attempts: usize::MAX,
            reader_eviction: Duration::from_secs(60),
            reader_proxy_timeout: Duration::from_secs(15),
            writer_proxy_timeout: Duration::from_secs(30),
            writer_is_local: false,
            include_writer_for_reads: false,
            buffer: Some(GatewayBufferConfig {
                dir: dir.path().to_path_buf(),
                max_entries: 8,
                max_bytes: 1024 * 1024,
                max_request_bytes: 1024,
                replay_interval: Duration::from_millis(10),
            }),
        });
        let request = Request {
            method: "PUT".to_string(),
            headers: HashMap::new(),
            path: "/collections/repo/points".to_string(),
            body: br#"{"points":[{"id":1}]}"#.to_vec(),
        };

        let response = router
            .buffered_write_response(&request, "writer_proxy_error")
            .unwrap()
            .unwrap();

        assert_eq!(response.status, 202);
        assert_eq!(
            response
                .headers
                .get("X-Helix-Gateway-Buffered")
                .map(String::as_str),
            Some("true")
        );
        let buffer = router.buffer.as_ref().unwrap();
        assert_eq!(buffer.queue_depth().unwrap().entries, 1);
    }

    #[test]
    fn scaled_merge_budget_doubles_only_for_severe_backlog() {
        let target = NamedVectorManager::max_indexed_segments_target().max(1);
        assert_eq!(scaled_index_job_max_merges(target.saturating_mul(10), 4), 4);
        assert_eq!(
            scaled_index_job_max_merges(target.saturating_mul(10).saturating_add(1), 4),
            8
        );
    }

    #[test]
    fn queued_index_jobs_do_not_pin_collection_storage() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = Arc::new(HelixGraphStorage::new(&path, Config::new(8, 32, 64, 1)).unwrap());
        let job = IndexJob {
            collection: "repo".to_string(),
            storage: Arc::downgrade(&storage),
            spaces_to_check: vec!["dense".to_string()],
            hnsw_config: HNSWConfig::new(Some(8), Some(32), Some(64)),
            flat_scan_threshold: 1,
        };

        assert_eq!(Arc::strong_count(&storage), 1);
        assert!(job.storage.upgrade().is_some());
        drop(storage);
        assert!(job.storage.upgrade().is_none());
        assert_eq!(index_job_debt(&job), DenseSegmentDebt::default());
        assert_eq!(index_job_backlog_ratio_millis(&job), 0);
    }

    #[test]
    fn reaper_selection_is_global_priority_not_fifo_windowed() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = Arc::new(HelixGraphStorage::new(&path, Config::new(8, 32, 64, 1)).unwrap());
        let state = Arc::new((
            Mutex::new(ReaperQueueState::default()),
            std::sync::Condvar::new(),
        ));
        let queue_depth = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        for idx in 0..256 {
            enqueue_reaper_job(
                &state,
                &queue_depth,
                test_reaper_job(
                    &storage,
                    &format!("dirty__seg_{idx:06}"),
                    1_100_000_000,
                    "dirty_discovered",
                ),
            );
        }
        enqueue_reaper_job(
            &state,
            &queue_depth,
            test_reaper_job(
                &storage,
                "admin__seg_high_priority",
                39_200_049_875,
                "retired",
            ),
        );

        let selected = SegmentReaper::take_next_job(&state, &queue_depth);

        assert_eq!(selected.physical_name, "admin__seg_high_priority");
        assert_eq!(selected.reason, "retired");
        assert_eq!(queue_depth.load(Ordering::Relaxed), 256);
    }

    #[test]
    fn reaper_selection_round_robins_equal_priority_jobs_after_yield() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = Arc::new(HelixGraphStorage::new(&path, Config::new(8, 32, 64, 1)).unwrap());
        let state = Arc::new((
            Mutex::new(ReaperQueueState::default()),
            std::sync::Condvar::new(),
        ));
        let queue_depth = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        for physical_name in ["dirty__seg_a", "dirty__seg_b", "dirty__seg_c"] {
            enqueue_reaper_job(
                &state,
                &queue_depth,
                test_reaper_job(&storage, physical_name, 1_100_000_000, "dirty_discovered"),
            );
        }

        let first = SegmentReaper::take_next_job(&state, &queue_depth);
        SegmentReaper::requeue_yielded_job(&state, &queue_depth, first);
        let second = SegmentReaper::take_next_job(&state, &queue_depth);

        assert_eq!(second.physical_name, "dirty__seg_b");
        assert_eq!(queue_depth.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn reaper_selection_key_uses_submitted_priority_snapshot() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = Arc::new(HelixGraphStorage::new(&path, Config::new(8, 32, 64, 1)).unwrap());
        let job = test_reaper_job(
            &storage,
            "dirty__seg_000001",
            1_100_000_000,
            "dirty_discovered",
        );

        let selected = reaper_job_selection_key(&job);

        assert_eq!(selected.priority_score, job.priority_score);
        assert_eq!(selected.dirty_retired_segments, job.dirty_retired_segments);
        assert_eq!(selected.gate_debt, job.gate_debt);
    }

    #[test]
    fn reaper_selection_key_keeps_gate_debt_tiebreaker_snapshot() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = Arc::new(HelixGraphStorage::new(&path, Config::new(8, 32, 64, 1)).unwrap());
        let mut job = test_reaper_job(
            &storage,
            "dirty__seg_000001",
            1_100_000_000,
            "dirty_discovered",
        );
        job.gate_debt = 4;
        job.dirty_retired_segments = 9;

        let selected = reaper_job_selection_key(&job);

        assert_eq!(selected.priority_score, job.priority_score);
        assert_eq!(selected.dirty_retired_segments, job.dirty_retired_segments);
        assert_eq!(selected.gate_debt, job.gate_debt);
    }

    #[test]
    fn reaper_job_turn_advances_one_database_at_a_time() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = Arc::new(HelixGraphStorage::new(&path, Config::new(8, 32, 64, 1)).unwrap());
        let mut job = test_reaper_job(
            &storage,
            "dirty__seg_000001",
            1_100_000_000,
            "dirty_discovered",
        );

        assert!(is_fresh_reaper_job(&job));
        assert_eq!(next_reaper_db_index(&job), Some(0));

        mark_reaper_db_step(&mut job, 0, false);

        assert!(!is_fresh_reaper_job(&job));
        assert_eq!(job.next_db_index, 1);
        assert!(!job.drained_dbs[0]);
        assert_eq!(next_reaper_db_index(&job), Some(1));

        for db_index in 1..REAPER_DB_COUNT {
            mark_reaper_db_step(&mut job, db_index, true);
        }

        assert_eq!(next_reaper_db_index(&job), Some(0));
        assert!(!all_reaper_dbs_drained(&job));

        mark_reaper_db_step(&mut job, 0, true);

        assert!(all_reaper_dbs_drained(&job));
        assert_eq!(next_reaper_db_index(&job), None);
    }

    #[test]
    fn yielded_reaper_job_requeues_without_retry_attempt() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = Arc::new(HelixGraphStorage::new(&path, Config::new(8, 32, 64, 1)).unwrap());
        let state = Arc::new((
            Mutex::new(ReaperQueueState::default()),
            std::sync::Condvar::new(),
        ));
        let queue_depth = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut job = test_reaper_job(
            &storage,
            "dirty__seg_000001",
            1_100_000_000,
            "dirty_discovered",
        );
        job.attempts = 2;
        mark_reaper_db_step(&mut job, 0, false);

        SegmentReaper::requeue_yielded_job(&state, &queue_depth, job);

        assert_eq!(queue_depth.load(Ordering::Relaxed), 1);
        let guard = state.0.lock().unwrap();
        assert_eq!(guard.queued_jobs.len(), 1);
        assert_eq!(guard.queued_jobs[0].attempts, 2);
        assert_eq!(guard.queued_jobs[0].next_db_index, 1);
    }

    fn test_reaper_job(
        storage: &Arc<HelixGraphStorage>,
        physical_name: &str,
        priority_score: usize,
        reason: &'static str,
    ) -> ReaperJob {
        ReaperJob {
            key: format!(
                "{}::{physical_name}",
                storage.lmdb_env().unwrap().path().display()
            ),
            storage: Arc::clone(storage),
            physical_name: physical_name.to_string(),
            priority_score,
            gate_debt: usize::from(reason == "dirty_discovered"),
            dirty_retired_segments: 1,
            reason,
            attempts: 0,
            next_db_index: 0,
            drained_dbs: [false; REAPER_DB_COUNT],
        }
    }

    #[test]
    fn dense_segment_format_parser_accepts_turbo_quant_aliases() {
        assert_eq!(parse_dense_segment_format(None), DenseSegmentFormat::Legacy);
        assert_eq!(
            parse_dense_segment_format(Some("")),
            DenseSegmentFormat::Legacy
        );
        assert_eq!(
            parse_dense_segment_format(Some(" hvs8 ")),
            DenseSegmentFormat::Hvs8
        );
        for value in ["tq", "hvtq", "turbo_prod", "turboquant", "turbo_quant"] {
            assert_eq!(
                parse_dense_segment_format(Some(value)),
                DenseSegmentFormat::TurboQuant
            );
        }
    }

    #[test]
    fn chunked_merge_publish_tq_materializes_hvtq_and_writes_markers() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _format = EnvVarGuard::set("HELIX_SEGMENT_FORMAT", "tq");
        let spindle =
            crate::helix_engine::vector_core::spindle::SpindleConfig::turbo_prod_compact(4);
        let (temp_dir, storage, hnsw_config) = storage_with_dense_vector(spindle.clone());
        let rows = vec![
            (101u128, vec![0.10, 0.20, 0.30, 0.40]),
            (102u128, vec![0.90, 0.80, 0.70, 0.60]),
        ];
        let merge = prepared_merge_for_rows(&rows, Some(&spindle));

        run_chunked_merge_publish(&storage, TEST_DENSE_VECTOR, &hnsw_config, merge).unwrap();

        let hvtq_path = tq_sidecar_path(&temp_dir);
        assert!(hvtq_path.exists());
        assert!(!hvs8_sidecar_path(&temp_dir).exists());
        assert!(!temp_dir
            .path()
            .join(TEST_MERGE_TARGET)
            .with_extension("hvec")
            .exists());
        let hvtq_blob = std::fs::read(&hvtq_path).unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let cores = storage.named_vectors.cores_read().unwrap();
        let core = cores.get(TEST_MERGE_TARGET).unwrap();
        assert!(core.mmap_is_turbo_quantized());
        let durable_blob = core
            .vector_data_db
            .as_ref()
            .unwrap()
            .get(
                &txn,
                crate::helix_engine::vector_core::vector_core::HVTQ_SIDECAR_BLOB_KEY,
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            durable_blob,
            hvtq_blob.as_slice(),
            "HVTQ sidecar must be persisted in the durable segment namespace"
        );
        let row = core
            .vectors_db
            .as_ref()
            .unwrap()
            .get(&txn, vector_key_for_test(101, 0).as_slice())
            .unwrap()
            .unwrap();
        assert!(
            row.is_empty(),
            "HVTQ publish must attach ordinals before Phase B so LMDB stores markers"
        );
        drop(cores);
        drop(txn);

        std::fs::remove_file(&hvtq_path).unwrap();
        drop(storage);

        let reopened = Arc::new(
            HelixGraphStorage::new(temp_dir.path().to_str().unwrap(), Config::new(8, 32, 64, 1))
                .unwrap(),
        );
        assert_eq!(
            std::fs::read(&hvtq_path).unwrap(),
            hvtq_blob,
            "reopen should rehydrate a missing local HVTQ sidecar from durable storage"
        );
        let reopened_cores = reopened.named_vectors.cores_read().unwrap();
        let reopened_core = reopened_cores.get(TEST_MERGE_TARGET).unwrap();
        assert!(reopened_core.mmap_is_turbo_quantized());
    }

    #[test]
    fn chunked_merge_publish_tq_attaches_ordinals_on_lsm_backend() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _backend = EnvVarGuard::set("HELIX_STORAGE_BACKEND", "lsm");
        let _in_mem = EnvVarGuard::set("HELIX_LSM_IN_MEMORY", "1");
        let _format = EnvVarGuard::set("HELIX_SEGMENT_FORMAT", "tq");
        let spindle =
            crate::helix_engine::vector_core::spindle::SpindleConfig::turbo_prod_compact(4);
        let temp_dir = TempDir::new().unwrap();
        let config = Config::new(8, 32, 64, 1);
        let collections =
            Arc::new(CollectionManager::new(temp_dir.path().join("data"), config.clone()).unwrap());
        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::CreateCollection {
                name: "repo".into(),
                vectors: HashMap::from([(
                    TEST_DENSE_VECTOR.into(),
                    NamedVectorConfig {
                        size: 4,
                        distance:
                            crate::helix_engine::vector_core::named_vectors::DistanceMetric::Cosine,
                        spindle: spindle.clone(),
                    },
                )]),
                sparse_vectors: HashMap::new(),
                hnsw_overrides: None,
            },
        )
        .unwrap();
        let storage = collections.get_collection("repo").unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);
        let hnsw_config = HNSWConfig::new(Some(8), Some(32), Some(64));
        let rows = vec![
            (401u128, vec![0.10, 0.20, 0.30, 0.40]),
            (402u128, vec![0.90, 0.80, 0.70, 0.60]),
        ];
        let merge = prepared_merge_for_rows(&rows, Some(&spindle));

        run_chunked_merge_publish(&storage, TEST_DENSE_VECTOR, &hnsw_config, merge).unwrap();

        let hvtq_path = storage
            .collection_path()
            .join(TEST_MERGE_TARGET)
            .with_extension("hvtq");
        let hvtq_blob = std::fs::read(&hvtq_path).unwrap();
        let r = storage.backend.begin_read().unwrap();
        let ordinal = storage
            .backend
            .get_with(
                &r,
                Namespace::Segment {
                    physical_name: TEST_MERGE_TARGET,
                    db: crate::helix_engine::storage_core::backend::SegmentDb::Ordinals,
                },
                &401u128.to_be_bytes(),
                |opt| opt.map(|b| b.to_vec()),
            )
            .unwrap()
            .unwrap();
        assert!(
            ordinal.len() == 8,
            "HVTQ LSM publish must write sidecar ordinals through AnyWrite"
        );
        let durable_blob = storage
            .backend
            .get_with(
                &r,
                Namespace::Segment {
                    physical_name: TEST_MERGE_TARGET,
                    db: crate::helix_engine::storage_core::backend::SegmentDb::VectorData,
                },
                crate::helix_engine::vector_core::vector_core::HVTQ_SIDECAR_BLOB_KEY,
                |opt| opt.map(|b| b.to_vec()),
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            durable_blob, hvtq_blob,
            "HVTQ LSM publish must store the sidecar bytes in the durable backend"
        );
    }

    #[test]
    fn chunked_merge_publish_tq_falls_back_to_hvs8_for_non_turbo_prod() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _format = EnvVarGuard::set("HELIX_SEGMENT_FORMAT", "tq");
        let spindle = crate::helix_engine::vector_core::spindle::SpindleConfig::scalar_int8();
        let (temp_dir, storage, hnsw_config) = storage_with_dense_vector(spindle);
        let rows = vec![
            (201u128, vec![0.10, 0.20, 0.30, 0.40]),
            (202u128, vec![0.90, 0.80, 0.70, 0.60]),
        ];
        let merge = prepared_merge_for_rows(&rows, None);

        run_chunked_merge_publish(&storage, TEST_DENSE_VECTOR, &hnsw_config, merge).unwrap();

        assert!(
            hvs8_sidecar_path(&temp_dir).exists(),
            "tq format should preserve HVS8 sidecars for non-TurboProd compact collections"
        );
        assert!(!tq_sidecar_path(&temp_dir).exists());
        let cores = storage.named_vectors.cores_read().unwrap();
        let core = cores.get(TEST_MERGE_TARGET).unwrap();
        assert!(core.mmap_is_quantized());
        assert!(!core.mmap_is_turbo_quantized());
    }

    #[test]
    fn chunked_merge_publish_tq_materialize_error_is_fatal() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _format = EnvVarGuard::set("HELIX_SEGMENT_FORMAT", "tq");
        let spindle =
            crate::helix_engine::vector_core::spindle::SpindleConfig::turbo_prod_compact(4);
        let (temp_dir, storage, hnsw_config) = storage_with_dense_vector(spindle.clone());
        let encoded = crate::helix_engine::vector_core::spindle::encode_vector(
            &[0.10, 0.20, 0.30, 0.40],
            &spindle,
        )
        .unwrap();
        let merge = crate::helix_engine::vector_core::named_vectors::PreparedMerge {
            merge_targets: Vec::new(),
            prepared_index: crate::helix_engine::vector_core::vector_core::PreparedIndex {
                point_ids: vec![301, 302],
                original_data: vec![Some(vec![0.10, 0.20, 0.30, 0.40])],
                encoded_data: vec![Some(encoded)],
                point_fields: vec![HashMap::new(), HashMap::new()],
                levels: vec![0, 0],
                adjacency: vec![Some(vec![Vec::new()]), Some(vec![Vec::new()])],
                entry_ord: 0,
                level_zero_preflushed: false,
            },
        };

        let err = run_chunked_merge_publish(&storage, TEST_DENSE_VECTOR, &hnsw_config, merge)
            .unwrap_err();

        assert!(
            err.to_string().contains("prepared sidecar row mismatch"),
            "unexpected error: {err}"
        );
        assert!(!tq_sidecar_path(&temp_dir).exists());
        assert!(
            !hvs8_sidecar_path(&temp_dir).exists(),
            "HVTQ materialize errors must not fall back to HVS8 and publish degraded state"
        );
    }

    #[test]
    fn gate_debt_budget_can_skip_builds_without_yielding() {
        let mut budget = IndexJobBudget::new(NamedVectorManager::max_indexed_segments_target() + 1);
        budget.reserve_turn_for_merges();

        assert_eq!(budget.remaining_build_segments(), 0);
        assert!(budget.can_merge());
        assert!(budget.is_merge_only_turn());
        assert!(!budget.should_yield());
    }

    #[test]
    fn gate_debt_build_budget_can_continue_to_merge_phase() {
        let mut budget = IndexJobBudget::new(NamedVectorManager::max_indexed_segments_target());
        budget.expand_builds_for_gate_debt(1);
        let build_cap = budget.max_build_segments;

        budget.record_build_segments(build_cap);

        assert_eq!(budget.remaining_build_segments(), 0);
        assert!(budget.can_merge());
        assert!(!budget.should_yield());
    }

    #[test]
    fn gate_debt_build_budget_scales_for_giant_backlog() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _base = EnvVarGuard::set("HELIX_INDEX_JOB_GATE_DEBT_BUILD_SEGMENTS", "8");
        let _max = EnvVarGuard::set("HELIX_INDEX_JOB_GATE_DEBT_BUILD_SEGMENTS_MAX", "256");

        assert_eq!(gate_debt_build_segments_for(1), 8);
        assert_eq!(gate_debt_build_segments_for(64), 64);
        assert_eq!(gate_debt_build_segments_for(10_000), 256);
    }

    #[test]
    fn optimizer_debt_submits_gate_work_only_at_segment_cap() {
        let cap = NamedVectorManager::segment_creation_gate_cap();
        let gate_only = DenseSegmentDebt {
            indexed_segments: 0,
            active_segments: 0,
            merge_debt: 0,
            gate_debt: 6,
            dirty_retired_segments: 0,
        };
        let below_cap_gate = DenseSegmentDebt {
            active_segments: cap.saturating_sub(1),
            ..gate_only
        };
        let cap_hit_gate = DenseSegmentDebt {
            active_segments: cap,
            ..gate_only
        };
        let merge_backlog = DenseSegmentDebt {
            indexed_segments: 3,
            gate_debt: 0,
            ..gate_only
        };

        assert!(!optimizer_debt_has_submit_work(gate_only, 1));
        assert!(!optimizer_debt_has_submit_work(below_cap_gate, 1));
        assert!(optimizer_debt_has_submit_work(cap_hit_gate, 1));
        assert!(optimizer_debt_has_submit_work(merge_backlog, 1));
        assert!(!optimizer_debt_can_expand_build_budget(gate_only));
        assert!(!optimizer_debt_can_expand_build_budget(below_cap_gate));
        assert!(optimizer_debt_can_expand_build_budget(cap_hit_gate));
        assert!(!optimizer_debt_can_expand_build_budget(merge_backlog));
    }

    #[test]
    fn optimizer_priority_prefers_gate_blocked_collections() {
        let target = NamedVectorManager::max_indexed_segments_target().max(1);
        let heavy_backlog = DenseSegmentDebt {
            indexed_segments: target.saturating_mul(999),
            active_segments: target.saturating_mul(999),
            merge_debt: target.saturating_mul(998),
            gate_debt: 0,
            dirty_retired_segments: 0,
        };
        let gate_blocked = DenseSegmentDebt {
            indexed_segments: 1,
            active_segments: 1,
            merge_debt: 0,
            gate_debt: 1,
            dirty_retired_segments: 0,
        };

        assert!(
            optimizer_debt_priority_score(gate_blocked, None)
                > optimizer_debt_priority_score(heavy_backlog, None)
        );
    }

    #[test]
    fn optimizer_priority_keeps_dirty_cleanup_from_starving_merge_debt() {
        let admin_like_merge_backlog = DenseSegmentDebt {
            indexed_segments: 73,
            active_segments: 80,
            merge_debt: 32,
            gate_debt: 0,
            dirty_retired_segments: 6,
        };
        let dirty_cleanup = DenseSegmentDebt {
            indexed_segments: 46,
            active_segments: 46,
            merge_debt: 0,
            gate_debt: 0,
            dirty_retired_segments: 2_000,
        };

        assert!(
            optimizer_debt_priority_score(admin_like_merge_backlog, None)
                > optimizer_debt_priority_score(dirty_cleanup, None)
        );
    }

    #[test]
    fn reaper_priority_counts_dirty_retired_segments() {
        let target = NamedVectorManager::max_indexed_segments_target().max(1);
        let normal_backlog = DenseSegmentDebt {
            indexed_segments: target.saturating_mul(8),
            active_segments: target.saturating_mul(8),
            merge_debt: target.saturating_mul(7),
            gate_debt: 0,
            dirty_retired_segments: 0,
        };
        let dirty_cleanup = DenseSegmentDebt {
            indexed_segments: 0,
            active_segments: 0,
            merge_debt: 0,
            gate_debt: 0,
            dirty_retired_segments: 1,
        };

        assert!(
            reaper_debt_priority_score(dirty_cleanup, None)
                > reaper_debt_priority_score(normal_backlog, None)
        );
    }

    #[test]
    fn optimizer_priority_prefers_breaker_tripped_collections() {
        // Breaker recovery floor = 128 × 0.75 = 96. A collection at/over the
        // floor is 503ing client upserts and must outrank gate-debt freshness
        // work, even though tripping the breaker has zeroed its own gate_debt.
        let drain_floor = Some(96);

        let breaker_tripped = DenseSegmentDebt {
            indexed_segments: 115,
            active_segments: 115,
            merge_debt: 67,
            gate_debt: 0,
            dirty_retired_segments: 0,
        };
        let gate_blocked = DenseSegmentDebt {
            indexed_segments: 1,
            active_segments: 1,
            merge_debt: 0,
            gate_debt: 1,
            dirty_retired_segments: 1,
        };
        assert!(
            optimizer_debt_priority_score(breaker_tripped, drain_floor)
                > optimizer_debt_priority_score(gate_blocked, drain_floor)
        );

        // More-over-the-floor drains first (overage-scaled).
        let barely_over = DenseSegmentDebt {
            indexed_segments: 97,
            ..breaker_tripped
        };
        assert!(
            optimizer_debt_priority_score(breaker_tripped, drain_floor)
                > optimizer_debt_priority_score(barely_over, drain_floor)
        );

        // Below the floor the breaker term is inert: gate-debt still wins.
        let below_floor = DenseSegmentDebt {
            indexed_segments: 50,
            active_segments: 50,
            merge_debt: 40,
            gate_debt: 0,
            dirty_retired_segments: 0,
        };
        assert!(
            optimizer_debt_priority_score(gate_blocked, drain_floor)
                > optimizer_debt_priority_score(below_floor, drain_floor)
        );

        // With the breaker disabled (floor None) the term never engages.
        assert!(
            optimizer_debt_priority_score(gate_blocked, None)
                > optimizer_debt_priority_score(breaker_tripped, None)
        );
    }

    #[test]
    fn breaker_drain_merge_fan_in_limit_engages_only_at_recovery_floor() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::unset("HELIX_DENSE_DRAIN_MERGE_MAX_FANIN");
        let debt = DenseSegmentDebt {
            indexed_segments: 96,
            active_segments: 96,
            merge_debt: 80,
            gate_debt: 0,
            dirty_retired_segments: 0,
        };

        assert_eq!(breaker_drain_merge_fan_in_limit(debt, Some(96)), Some(2));
        assert_eq!(breaker_drain_merge_fan_in_limit(debt, Some(97)), None);
        assert_eq!(breaker_drain_merge_fan_in_limit(debt, None), None);

        {
            let _env = EnvVarGuard::set("HELIX_DENSE_DRAIN_MERGE_MAX_FANIN", "3");
            assert_eq!(breaker_drain_merge_fan_in_limit(debt, Some(96)), Some(3));
        }

        {
            let _env = EnvVarGuard::set("HELIX_DENSE_DRAIN_MERGE_MAX_FANIN", "1");
            assert_eq!(breaker_drain_merge_fan_in_limit(debt, Some(96)), Some(2));
        }
    }

    #[test]
    fn merge_debt_reserves_merge_turn_without_gate_or_breaker_debt() {
        let debt = DenseSegmentDebt {
            indexed_segments: 73,
            active_segments: 81,
            merge_debt: 32,
            gate_debt: 0,
            dirty_retired_segments: 0,
        };

        assert!(should_reserve_index_job_turn_for_merges(debt, Some(2)));
        assert!(should_reserve_index_job_turn_for_merges(debt, None));
    }

    #[test]
    fn optimizer_pending_work_includes_externalized_marker_repair_debt() {
        let _guard = ENV_LOCK.lock().unwrap();
        let spindle = crate::helix_engine::vector_core::spindle::SpindleConfig::default();
        let (_temp_dir, storage, _hnsw_config) = storage_with_dense_vector(spindle);

        {
            let cores = storage.named_vectors.cores_read().unwrap();
            cores
                .get(TEST_DENSE_VECTOR)
                .unwrap()
                .mark_externalized_marker_repair_needed();
        }

        assert!(optimizer_has_pending_work(
            &storage,
            &[TEST_DENSE_VECTOR.to_string()],
            usize::MAX,
            true,
        ));
    }

    #[test]
    fn optimizer_backlog_budget_respects_process_pressure() {
        assert_eq!(
            optimizer_backlog_effective_per_sweep(16, 1, MaintenanceAdmission::Admitted),
            16
        );
        assert_eq!(
            optimizer_backlog_effective_per_sweep(16, 1, MaintenanceAdmission::Rejected("memory")),
            1
        );
        assert_eq!(
            optimizer_backlog_effective_per_sweep(4, 0, MaintenanceAdmission::Rejected("fds")),
            0
        );
    }

    struct EnvVarGuard {
        name: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn set(name: &'static str, value: &'static str) -> Self {
            let guard = Self {
                name,
                previous: std::env::var_os(name),
            };
            std::env::set_var(name, value);
            guard
        }

        fn unset(name: &'static str) -> Self {
            let guard = Self {
                name,
                previous: std::env::var_os(name),
            };
            std::env::remove_var(name);
            guard
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

    const TEST_DENSE_VECTOR: &str = "dense";
    const TEST_MERGE_TARGET: &str = "dense__seg_000001";

    fn vector_key_for_test(id: u128, level: usize) -> Vec<u8> {
        [b"v:".as_slice(), &id.to_be_bytes(), &level.to_be_bytes()].concat()
    }

    fn storage_with_dense_vector(
        spindle: crate::helix_engine::vector_core::spindle::SpindleConfig,
    ) -> (TempDir, Arc<HelixGraphStorage>, HNSWConfig) {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();
        let storage = Arc::new(HelixGraphStorage::new(&path, Config::new(8, 32, 64, 1)).unwrap());
        let hnsw_config = HNSWConfig::new(Some(8), Some(32), Some(64));
        let vector_config = NamedVectorConfig {
            size: 4,
            distance: crate::helix_engine::vector_core::named_vectors::DistanceMetric::Cosine,
            spindle,
        };

        storage
            .with_exclusive_write_txn(|txn| {
                storage
                    .named_vectors
                    .create_vector_index(
                        storage.lmdb_env().unwrap(),
                        txn,
                        TEST_DENSE_VECTOR,
                        vector_config.clone(),
                        hnsw_config.clone(),
                    )
                    .map_err(GraphError::from)?;
                storage.set_named_vectors_metadata(txn, storage.named_vectors.list_vectors())?;
                storage.set_dense_vector_spaces_metadata(
                    txn,
                    storage.named_vectors.list_dense_vector_spaces(),
                )?;
                Ok(())
            })
            .unwrap();

        (temp_dir, storage, hnsw_config)
    }

    fn prepared_merge_for_rows(
        rows: &[(u128, Vec<f32>)],
        spindle: Option<&crate::helix_engine::vector_core::spindle::SpindleConfig>,
    ) -> crate::helix_engine::vector_core::named_vectors::PreparedMerge {
        use crate::helix_engine::vector_core::named_vectors::PreparedMerge;
        use crate::helix_engine::vector_core::vector_core::PreparedIndex;

        let encoded_data = rows
            .iter()
            .map(|(_, data)| {
                spindle.map(|config| {
                    crate::helix_engine::vector_core::spindle::encode_vector(data, config).unwrap()
                })
            })
            .collect::<Vec<_>>();
        let n = rows.len();
        PreparedMerge {
            merge_targets: Vec::new(),
            prepared_index: PreparedIndex {
                point_ids: rows.iter().map(|(id, _)| *id).collect(),
                original_data: rows.iter().map(|(_, data)| Some(data.clone())).collect(),
                encoded_data,
                point_fields: vec![HashMap::new(); n],
                levels: vec![0; n],
                adjacency: vec![Some(vec![Vec::new()]); n],
                entry_ord: 0,
                level_zero_preflushed: false,
            },
        }
    }

    fn tq_sidecar_path(temp_dir: &TempDir) -> PathBuf {
        temp_dir
            .path()
            .join(TEST_MERGE_TARGET)
            .with_extension("hvtq")
    }

    fn hvs8_sidecar_path(temp_dir: &TempDir) -> PathBuf {
        temp_dir
            .path()
            .join(TEST_MERGE_TARGET)
            .with_extension("hvs8")
    }

    fn test_point(id: u128, sparse_terms: usize) -> ReplicatedPoint {
        let mut sparse_vectors = HashMap::new();
        if sparse_terms > 0 {
            sparse_vectors.insert(
                "text".to_string(),
                SparseVector {
                    indices: (0..sparse_terms).map(|i| i as u32).collect(),
                    values: vec![1.0; sparse_terms],
                },
            );
        }
        ReplicatedPoint {
            id,
            vectors: HashMap::new(),
            sparse_vectors,
            payload: HashMap::new(),
        }
    }

    fn collect_budgeted_chunk_ids(
        points: &[ReplicatedPoint],
        max_docs: usize,
        max_postings: usize,
        max_writes: usize,
        payload_index_count: usize,
        indexed_segments: usize,
    ) -> Vec<Vec<u128>> {
        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < points.len() {
            let (len, _) = budgeted_upsert_chunk_len(
                &points[start..],
                max_docs,
                max_postings,
                max_writes,
                payload_index_count,
                indexed_segments,
            );
            let end = start + len;
            chunks.push(points[start..end].iter().map(|p| p.id).collect());
            start = end;
        }
        chunks
    }

    struct InMemoryTransport {
        registry: Arc<Mutex<HashMap<String, Weak<ReplicationManager>>>>,
    }

    impl InMemoryTransport {
        fn new(registry: Arc<Mutex<HashMap<String, Weak<ReplicationManager>>>>) -> Self {
            Self { registry }
        }

        fn lookup(&self, target: &str) -> Result<Arc<ReplicationManager>, String> {
            self.registry
                .lock()
                .map_err(|e| e.to_string())?
                .get(target)
                .and_then(Weak::upgrade)
                .ok_or_else(|| format!("unknown target {}", target))
        }
    }

    impl RaftTransport for InMemoryTransport {
        fn send_message(&self, target: &str, message: &Message) -> Result<(), String> {
            self.lookup(target)?
                .receive_raft_message(message.clone())
                .map_err(|e| e.to_string())
        }

        fn forward_proposal(
            &self,
            target: &str,
            mutation: &ReplicatedMutation,
        ) -> Result<(), String> {
            self.lookup(target)?
                .propose_internal(mutation.clone())
                .map_err(|e| e.to_string())
        }
    }

    #[test]
    fn delete_apply_chunk_uses_explicit_override() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("HELIX_UPSERT_APPLY_CHUNK", "128");
        std::env::set_var("HELIX_DELETE_APPLY_CHUNK", "32");
        std::env::remove_var("HELIX_DELETE_APPLY_MIN_CHUNK");

        assert_eq!(delete_apply_chunk_size(948), 32);

        std::env::remove_var("HELIX_DELETE_APPLY_CHUNK");
        std::env::remove_var("HELIX_UPSERT_APPLY_CHUNK");
    }

    #[test]
    fn delete_apply_chunk_uses_sqrt_fanout_with_floor() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("HELIX_UPSERT_APPLY_CHUNK", "128");
        std::env::remove_var("HELIX_DELETE_APPLY_CHUNK");
        std::env::remove_var("HELIX_DELETE_APPLY_MIN_CHUNK");

        assert_eq!(delete_apply_chunk_size(1), 128);
        assert_eq!(delete_apply_chunk_size(16), 64);
        assert_eq!(delete_apply_chunk_size(948), 64);

        std::env::remove_var("HELIX_UPSERT_APPLY_CHUNK");
    }

    #[test]
    fn delete_apply_chunk_floor_is_tunable() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("HELIX_UPSERT_APPLY_CHUNK", "128");
        std::env::set_var("HELIX_DELETE_APPLY_MIN_CHUNK", "32");
        std::env::remove_var("HELIX_DELETE_APPLY_CHUNK");

        assert_eq!(delete_apply_chunk_size(948), 32);

        std::env::remove_var("HELIX_DELETE_APPLY_MIN_CHUNK");
        std::env::remove_var("HELIX_UPSERT_APPLY_CHUNK");
    }

    #[test]
    fn budgeted_upsert_chunk_len_respects_doc_cap() {
        let points: Vec<_> = (0..5).map(|id| test_point(id, 1)).collect();

        assert_eq!(
            budgeted_upsert_chunk_len(&points, 2, 100, 0, 0, 0),
            (2, UpsertChunkSplitReason::None)
        );
        assert_eq!(
            collect_budgeted_chunk_ids(&points, 2, 100, 0, 0, 0),
            vec![vec![0, 1], vec![2, 3], vec![4]]
        );
    }

    #[test]
    fn budgeted_upsert_chunk_len_respects_sparse_posting_budget() {
        let points = vec![test_point(10, 2), test_point(11, 3), test_point(12, 4)];

        assert_eq!(
            budgeted_upsert_chunk_len(&points, 10, 5, 0, 0, 0),
            (2, UpsertChunkSplitReason::PostingBudget)
        );
        assert_eq!(
            collect_budgeted_chunk_ids(&points, 10, 5, 0, 0, 0),
            vec![vec![10, 11], vec![12]]
        );
    }

    #[test]
    fn budgeted_upsert_chunk_len_preserves_order_across_budget_splits() {
        let points = vec![
            test_point(20, 2),
            test_point(21, 4),
            test_point(22, 1),
            test_point(23, 4),
        ];

        assert_eq!(
            collect_budgeted_chunk_ids(&points, 10, 5, 0, 0, 0),
            vec![vec![20], vec![21, 22], vec![23]]
        );
    }

    #[test]
    fn budgeted_upsert_chunk_len_can_disable_sparse_budget() {
        let points = vec![test_point(30, 10_000), test_point(31, 10_000)];

        assert_eq!(
            budgeted_upsert_chunk_len(&points, 10, 0, 0, 0, 0),
            (2, UpsertChunkSplitReason::None)
        );
    }

    #[test]
    fn budgeted_upsert_chunk_len_allows_oversized_single_doc() {
        let points = vec![test_point(40, 10_000), test_point(41, 1)];

        assert_eq!(
            budgeted_upsert_chunk_len(&points, 10, 4096, 0, 0, 0),
            (1, UpsertChunkSplitReason::PostingBudget)
        );
        assert_eq!(
            collect_budgeted_chunk_ids(&points, 10, 4096, 0, 0, 0),
            vec![vec![40], vec![41]]
        );
    }

    #[test]
    fn budgeted_upsert_chunk_len_respects_write_budget() {
        let points: Vec<_> = (50..55).map(|id| test_point(id, 3)).collect();

        assert_eq!(
            budgeted_upsert_chunk_len(&points, 10, 0, 27, 1, 1),
            (2, UpsertChunkSplitReason::WriteBudget)
        );
        assert_eq!(
            collect_budgeted_chunk_ids(&points, 10, 0, 27, 1, 1),
            vec![vec![50, 51], vec![52, 53], vec![54]]
        );
    }

    #[test]
    fn budgeted_upsert_chunk_len_write_budget_accounts_for_dense_fanout() {
        let mut points: Vec<_> = (60..64).map(|id| test_point(id, 0)).collect();
        for point in &mut points {
            point.vectors.insert("dense".to_string(), vec![0.0, 1.0]);
        }

        assert_eq!(
            budgeted_upsert_chunk_len(&points, 10, 0, 30, 0, 3),
            (1, UpsertChunkSplitReason::WriteBudget)
        );
        assert_eq!(
            budgeted_upsert_chunk_len(&points, 10, 0, 30, 0, 0),
            (3, UpsertChunkSplitReason::WriteBudget)
        );
    }

    #[test]
    fn budgeted_upsert_chunk_len_allows_oversized_single_write_budget_doc() {
        let points = vec![test_point(70, 100)];

        assert_eq!(
            budgeted_upsert_chunk_len(&points, 10, 0, 10, 1, 1),
            (1, UpsertChunkSplitReason::None)
        );
    }

    #[test]
    fn upsert_apply_write_budget_respects_override() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::unset("HELIX_UPSERT_WRITE_BUDGET");

        assert_eq!(upsert_apply_write_budget(), 32 * 1024);

        std::env::set_var("HELIX_UPSERT_WRITE_BUDGET", "2048");
        assert_eq!(upsert_apply_write_budget(), 2048);

        std::env::set_var("HELIX_UPSERT_WRITE_BUDGET", "not-a-number");
        assert_eq!(upsert_apply_write_budget(), 32 * 1024);
    }

    #[test]
    fn sparse_metadata_hard_flush_is_bounded_and_at_least_normal_budget() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("HELIX_SPARSE_METADATA_FLUSH_BUDGET", "8192");
        std::env::remove_var("HELIX_SPARSE_METADATA_FLUSH_HARD_BUDGET");
        std::env::set_var("HELIX_SPARSE_METADATA_FLUSH_MAX_MS", "150");
        std::env::remove_var("HELIX_SPARSE_METADATA_FLUSH_HARD_MAX_MS");

        assert_eq!(sparse_metadata_flush_hard_budget(), 32_768);
        assert_eq!(
            sparse_metadata_flush_hard_max_duration(),
            Duration::from_millis(500)
        );

        std::env::remove_var("HELIX_SPARSE_METADATA_FLUSH_BUDGET");
        std::env::remove_var("HELIX_SPARSE_METADATA_FLUSH_MAX_MS");
    }

    #[test]
    fn sparse_metadata_hard_flush_respects_overrides() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("HELIX_SPARSE_METADATA_FLUSH_BUDGET", "8192");
        std::env::set_var("HELIX_SPARSE_METADATA_FLUSH_HARD_BUDGET", "4096");
        std::env::set_var("HELIX_SPARSE_METADATA_FLUSH_MAX_MS", "750");
        std::env::set_var("HELIX_SPARSE_METADATA_FLUSH_HARD_MAX_MS", "250");

        assert_eq!(sparse_metadata_flush_hard_budget(), 8192);
        assert_eq!(
            sparse_metadata_flush_hard_max_duration(),
            Duration::from_millis(750)
        );

        std::env::remove_var("HELIX_SPARSE_METADATA_FLUSH_BUDGET");
        std::env::remove_var("HELIX_SPARSE_METADATA_FLUSH_HARD_BUDGET");
        std::env::remove_var("HELIX_SPARSE_METADATA_FLUSH_MAX_MS");
        std::env::remove_var("HELIX_SPARSE_METADATA_FLUSH_HARD_MAX_MS");
    }

    #[test]
    fn point_mutation_gate_recovers_after_poison() {
        let gate = Arc::new(Mutex::new(()));
        let poison_gate = Arc::clone(&gate);
        let _ = thread::spawn(move || {
            let _guard = poison_gate.lock().unwrap();
            panic!("poison point mutation gate for recovery test");
        })
        .join();

        let guard = lock_point_mutation_gate(&gate, "repo", "test");
        drop(guard);

        let guard = lock_point_mutation_gate(&gate, "repo", "test");
        drop(guard);
    }

    struct NodeHarness {
        _path: PathBuf,
        collections: Arc<CollectionManager>,
        replication: Arc<ReplicationManager>,
        addr: String,
    }

    fn test_config(node_id: u64, peers: Vec<(u64, String)>) -> Config {
        Config {
            vector_config: VectorConfig {
                m: Some(16),
                ef_construction: Some(128),
                ef_search: Some(256),
                db_max_size: Some(1),
                flat_scan_threshold: None,
            },
            graph_config: GraphConfig {
                secondary_indices: None,
                snapshot_interval_secs: Some(3600),
                snapshot_keep_last: Some(3),
                raft: RaftConfig {
                    enabled: true,
                    node_id: Some(node_id),
                    bind_address: peers
                        .iter()
                        .find(|(id, _)| *id == node_id)
                        .map(|(_, addr)| addr.clone()),
                    snapshot_entries: Some(8),
                    snapshot_catchup_entries: Some(2),
                    raft_secret: None,
                    peers: peers
                        .into_iter()
                        .map(|(id, address)| RaftPeerConfig { id, address })
                        .collect(),
                },
            },
        }
    }

    fn make_node(
        registry: Arc<Mutex<HashMap<String, Weak<ReplicationManager>>>>,
        node_id: u64,
        peers: Vec<(u64, String)>,
        path: PathBuf,
    ) -> NodeHarness {
        let config = test_config(node_id, peers.clone());
        std::fs::create_dir_all(&path).unwrap();
        let collections = Arc::new(CollectionManager::new(path.join("data"), config).unwrap());
        make_node_with_collections(registry, node_id, peers, path, collections)
    }

    fn make_node_with_collections(
        registry: Arc<Mutex<HashMap<String, Weak<ReplicationManager>>>>,
        node_id: u64,
        peers: Vec<(u64, String)>,
        path: PathBuf,
        collections: Arc<CollectionManager>,
    ) -> NodeHarness {
        let config = test_config(node_id, peers.clone());
        let addr = peers
            .iter()
            .find(|(id, _)| *id == node_id)
            .unwrap()
            .1
            .clone();
        let transport: Arc<dyn RaftTransport> =
            Arc::new(InMemoryTransport::new(Arc::clone(&registry)));
        let replication = Arc::new(
            ReplicationManager::new_with_transport(Arc::clone(&collections), config, transport)
                .unwrap(),
        );
        registry
            .lock()
            .unwrap()
            .insert(addr.clone(), Arc::downgrade(&replication));
        NodeHarness {
            _path: path,
            collections,
            replication,
            addr,
        }
    }

    fn wait_for_leader(nodes: &[NodeHarness]) -> u64 {
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            for node in nodes {
                let status = node.replication.status().unwrap();
                if status.is_leader {
                    return status.node_id.unwrap();
                }
            }
            thread::sleep(Duration::from_millis(100));
        }
        panic!("leader election timed out");
    }

    #[test]
    fn delete_points_removes_dense_sparse_and_nodes() {
        let temp_dir = TempDir::new().unwrap();
        let config = Config::new(8, 32, 64, 1);
        let collections =
            Arc::new(CollectionManager::new(temp_dir.path().join("data"), config.clone()).unwrap());

        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::CreateCollection {
                name: "repo".into(),
                vectors: HashMap::from([(
                    "dense".into(),
                    NamedVectorConfig {
                        size: 3,
                        distance:
                            crate::helix_engine::vector_core::named_vectors::DistanceMetric::Cosine,
                        spindle: crate::helix_engine::vector_core::spindle::SpindleConfig::default(
                        ),
                    },
                )]),
                sparse_vectors: HashMap::from([(
                    "lex_sparse".into(),
                    SparseVectorConfig {
                        full_scan_threshold: 5000,
                        modifier: crate::helix_engine::vector_core::sparse::SparseModifier::Idf,
                        ..Default::default()
                    },
                )]),
                hnsw_overrides: None,
            },
        )
        .unwrap();

        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::UpsertPoints {
                collection: "repo".into(),
                points: (1u128..=3)
                    .map(|id| ReplicatedPoint {
                        id,
                        vectors: HashMap::from([("dense".into(), vec![id as f32, 0.0, 0.0])]),
                        sparse_vectors: HashMap::from([(
                            "lex_sparse".into(),
                            SparseVector {
                                indices: vec![id as u32],
                                values: vec![1.0],
                            },
                        )]),
                        payload: HashMap::from([(
                            "path".into(),
                            Value::String(format!("f{id}.rs")),
                        )]),
                    })
                    .collect(),
            },
        )
        .unwrap();

        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::DeletePoints {
                collection: "repo".into(),
                ids: vec![1, 1, 2],
            },
        )
        .unwrap();

        let storage = collections.get_collection("repo").unwrap();
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        assert!(storage
            .lmdb_nodes_db()
            .unwrap()
            .get(&txn, &1)
            .unwrap()
            .is_none());
        assert!(storage
            .lmdb_nodes_db()
            .unwrap()
            .get(&txn, &2)
            .unwrap()
            .is_none());
        assert!(storage
            .lmdb_nodes_db()
            .unwrap()
            .get(&txn, &3)
            .unwrap()
            .is_some());

        let stats = storage
            .named_vectors
            .get_dense_space_stats(&txn, "dense")
            .unwrap();
        assert_eq!(stats.vectors_count, 1);

        let has_deleted_sparse = storage
            .named_vectors
            .with_sparse_core("lex_sparse", |core| core.has_doc(&txn, 1))
            .unwrap_or(false);
        let has_remaining_sparse = storage
            .named_vectors
            .with_sparse_core("lex_sparse", |core| core.has_doc(&txn, 3))
            .unwrap_or(false);
        assert!(!has_deleted_sparse);
        assert!(has_remaining_sparse);
    }

    /// LSM counterpart of `delete_points_removes_dense_sparse_and_nodes`,
    /// driven through the real production LSM delete path
    /// (`apply_mutation(DeletePoints)` → `apply_delete_points`'s
    /// `BackendKind::Lsm` branch → `plan_delete_vectors_batch_be` in the
    /// read phase → `delete_vectors_batch_with_plan_be` inside the write
    /// hold). Builds two active segments (one sealed/Indexed via the real
    /// optimizer, one Mutable tail) so the planned ownership probe has to
    /// span multiple segments, then deletes a mix of ids owned by each
    /// segment plus an id that never existed.
    #[test]
    fn apply_delete_points_lsm_removes_ids_across_indexed_and_mutable_segments() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _backend = EnvVarGuard::set("HELIX_STORAGE_BACKEND", "lsm");
        let _in_mem = EnvVarGuard::set("HELIX_LSM_IN_MEMORY", "1");

        let temp_dir = TempDir::new().unwrap();
        let mut config = Config::new(8, 32, 64, 1);
        config.vector_config.flat_scan_threshold = Some(3);
        let collections =
            Arc::new(CollectionManager::new(temp_dir.path().join("data"), config.clone()).unwrap());

        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::CreateCollection {
                name: "repo".into(),
                vectors: HashMap::from([(
                    "dense".into(),
                    NamedVectorConfig {
                        size: 4,
                        distance:
                            crate::helix_engine::vector_core::named_vectors::DistanceMetric::Cosine,
                        spindle: crate::helix_engine::vector_core::spindle::SpindleConfig::default(
                        ),
                    },
                )]),
                sparse_vectors: HashMap::new(),
                hnsw_overrides: None,
            },
        )
        .unwrap();

        let storage = collections.get_collection("repo").unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);

        // 4 points > threshold(3): sealed and built into an Indexed segment
        // by the real optimizer below.
        let upsert_points = |ids: std::ops::RangeInclusive<u128>| {
            apply_mutation(
                &collections,
                &config,
                &ReplicatedMutation::UpsertPoints {
                    collection: "repo".into(),
                    points: ids
                        .map(|id| {
                            let mut v = [0.0f32; 4];
                            v[(id as usize - 1) % 4] = 1.0;
                            v[0] += 0.001 * id as f32;
                            ReplicatedPoint {
                                id,
                                vectors: HashMap::from([("dense".into(), v.to_vec())]),
                                sparse_vectors: HashMap::new(),
                                payload: HashMap::new(),
                            }
                        })
                        .collect(),
                },
            )
            .unwrap();
        };
        upsert_points(1..=4);

        let flat_scan_threshold = storage
            .named_vectors
            .effective_indexing_threshold(config.vector_flat_scan_threshold());
        run_index_job(IndexJob {
            collection: "repo".to_string(),
            storage: Arc::downgrade(&storage),
            spaces_to_check: vec!["dense".to_string()],
            hnsw_config: HNSWConfig::new(Some(8), Some(32), Some(64)),
            flat_scan_threshold,
        });

        // 2 more points land on a fresh Mutable tail segment behind the
        // now-Indexed segment.
        upsert_points(5..=6);

        let pre_stats = {
            let r = storage.backend.begin_read().unwrap();
            storage
                .named_vectors
                .get_dense_space_stats_be(&r, "dense")
                .unwrap()
        };
        assert_eq!(pre_stats.vectors_count, 6);
        assert!(
            pre_stats.segments_count >= 2,
            "fixture must span an Indexed segment and a Mutable tail, got {} segment(s)",
            pre_stats.segments_count
        );

        // id 1 lives in the Indexed segment, id 5 in the Mutable tail,
        // id 999 never existed anywhere.
        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::DeletePoints {
                collection: "repo".into(),
                ids: vec![1, 5, 999],
            },
        )
        .unwrap();

        let r = storage.backend.begin_read().unwrap();
        for (id, should_exist) in [
            (1u128, false),
            (2, true),
            (3, true),
            (4, true),
            (5, false),
            (6, true),
        ] {
            let node_exists = storage
                .backend
                .get_with(&r, Namespace::Nodes, &id.to_be_bytes(), |v| v.is_some())
                .unwrap();
            assert_eq!(
                node_exists, should_exist,
                "node {id} existence mismatch after delete"
            );
        }
        let post_stats = storage
            .named_vectors
            .get_dense_space_stats_be(&r, "dense")
            .unwrap();
        assert_eq!(
            post_stats.vectors_count, 4,
            "2 of 6 dense vectors must be removed across both segments"
        );

        // Directly confirm the dense delete landed on the segment that owns
        // each id, not just on whichever segment happened to be probed
        // first.
        let cores = storage.named_vectors.cores_read().unwrap();
        let any_contains = |id: u128| {
            cores
                .values()
                .any(|core| core.contains_id(&r, id).unwrap_or(false))
        };
        assert!(!any_contains(1), "id 1's dense vector must be gone");
        assert!(!any_contains(5), "id 5's dense vector must be gone");
        assert!(any_contains(2), "id 2's dense vector must remain");
        assert!(any_contains(6), "id 6's dense vector must remain");
    }

    #[test]
    fn three_node_cluster_replicates_points_and_indexes() {
        let registry = Arc::new(Mutex::new(HashMap::new()));
        let peers = vec![
            (1, "n1".to_string()),
            (2, "n2".to_string()),
            (3, "n3".to_string()),
        ];

        let nodes = vec![
            make_node(
                Arc::clone(&registry),
                1,
                peers.clone(),
                TempDir::new().unwrap().into_path(),
            ),
            make_node(
                Arc::clone(&registry),
                2,
                peers.clone(),
                TempDir::new().unwrap().into_path(),
            ),
            make_node(
                Arc::clone(&registry),
                3,
                peers.clone(),
                TempDir::new().unwrap().into_path(),
            ),
        ];

        let _leader_id = wait_for_leader(&nodes);

        let vectors = HashMap::from([(
            "dense".to_string(),
            NamedVectorConfig {
                size: 3,
                distance: crate::helix_engine::vector_core::named_vectors::DistanceMetric::Cosine,
                spindle: crate::helix_engine::vector_core::spindle::SpindleConfig::default(),
            },
        )]);
        nodes[1]
            .replication
            .apply(ReplicatedMutation::CreateCollection {
                name: "repo".into(),
                vectors,
                sparse_vectors: HashMap::new(),
                hnsw_overrides: None,
            })
            .unwrap();
        nodes[2]
            .replication
            .apply(ReplicatedMutation::UpsertPoints {
                collection: "repo".into(),
                points: vec![ReplicatedPoint {
                    id: 42,
                    vectors: HashMap::from([("dense".into(), vec![1.0, 0.0, 0.0])]),
                    sparse_vectors: HashMap::new(),
                    payload: HashMap::from([
                        ("repo".into(), Value::String("context-engine".into())),
                        ("stars".into(), Value::I64(10)),
                    ]),
                }],
            })
            .unwrap();
        nodes[0]
            .replication
            .apply(ReplicatedMutation::CreatePayloadIndex {
                collection: "repo".into(),
                field_name: "repo".into(),
                schema: PayloadIndexSchema::Keyword,
            })
            .unwrap();

        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            let mut all_ready = true;
            for node in &nodes {
                let storage = node.collections.get_collection("repo").unwrap();
                let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
                if storage.get_node(&txn, &42).is_err()
                    || storage.has_payload_index("repo").is_none()
                {
                    all_ready = false;
                    break;
                }
            }
            if all_ready {
                break;
            }
            thread::sleep(Duration::from_millis(100));
        }

        for node in &nodes {
            let storage = node.collections.get_collection("repo").unwrap();
            let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
            let point = storage.get_node(&txn, &42).unwrap();
            assert_eq!(
                point.properties.get("repo"),
                Some(&Value::String("context-engine".into()))
            );
            assert_eq!(
                storage.has_payload_index("repo"),
                Some(PayloadIndexSchema::Keyword)
            );
            let candidates = storage
                .get_nodes_by_payload_value(&txn, "repo", &Value::String("context-engine".into()))
                .unwrap();
            assert_eq!(candidates, vec![42]);
        }
    }

    #[test]
    fn follower_restart_recovers_and_catches_up() {
        let registry = Arc::new(Mutex::new(HashMap::new()));
        let peers = vec![
            (1, "m1".to_string()),
            (2, "m2".to_string()),
            (3, "m3".to_string()),
        ];
        let node1_path = TempDir::new().unwrap().into_path();
        let node2_path = TempDir::new().unwrap().into_path();
        let node3_path = TempDir::new().unwrap().into_path();

        let node1 = make_node(Arc::clone(&registry), 1, peers.clone(), node1_path);
        let node2 = make_node(Arc::clone(&registry), 2, peers.clone(), node2_path.clone());
        let node3 = make_node(Arc::clone(&registry), 3, peers.clone(), node3_path);
        let mut nodes = vec![node1, node2, node3];
        let _leader_id = wait_for_leader(&nodes);

        nodes[0]
            .replication
            .apply(ReplicatedMutation::CreateCollection {
                name: "repo".into(),
                vectors: HashMap::new(),
                sparse_vectors: HashMap::new(),
                hnsw_overrides: None,
            })
            .unwrap();
        nodes[0]
            .replication
            .apply(ReplicatedMutation::IngestBatch {
                collection: "repo".into(),
                ops: vec![ReplicatedIngestOp::Node(NodeUpsert {
                    id: 1,
                    label: "Symbol".into(),
                    properties: HashMap::from([("name".into(), Value::String("main".into()))]),
                })],
            })
            .unwrap();

        let addr = nodes[1].addr.clone();
        let collections = Arc::clone(&nodes[1].collections);
        drop(nodes.remove(1));

        // Recreate the Raft runtime against the same on-disk state and the same
        // collection manager. In production the process would restart; in-test
        // LMDB cannot reopen the same environment path inside the same process.
        let replacement = make_node_with_collections(
            Arc::clone(&registry),
            2,
            peers.clone(),
            node2_path,
            collections,
        );
        registry
            .lock()
            .unwrap()
            .insert(addr, Arc::downgrade(&replacement.replication));
        nodes.insert(1, replacement);
        let _leader_id = wait_for_leader(&nodes);

        nodes[2]
            .replication
            .apply(ReplicatedMutation::IngestBatch {
                collection: "repo".into(),
                ops: vec![ReplicatedIngestOp::Node(NodeUpsert {
                    id: 2,
                    label: "Symbol".into(),
                    properties: HashMap::from([("name".into(), Value::String("helper".into()))]),
                })],
            })
            .unwrap();

        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            let storage = nodes[1].collections.get_collection("repo").unwrap();
            let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
            if storage.get_node(&txn, &1).is_ok() && storage.get_node(&txn, &2).is_ok() {
                return;
            }
            thread::sleep(Duration::from_millis(100));
        }
        panic!("restarted follower did not catch up");
    }

    #[test]
    fn sparse_vectors_replicate_across_cluster() {
        let registry = Arc::new(Mutex::new(HashMap::new()));
        let peers = vec![
            (1, "sp1".to_string()),
            (2, "sp2".to_string()),
            (3, "sp3".to_string()),
        ];

        let nodes = vec![
            make_node(
                Arc::clone(&registry),
                1,
                peers.clone(),
                TempDir::new().unwrap().into_path(),
            ),
            make_node(
                Arc::clone(&registry),
                2,
                peers.clone(),
                TempDir::new().unwrap().into_path(),
            ),
            make_node(
                Arc::clone(&registry),
                3,
                peers.clone(),
                TempDir::new().unwrap().into_path(),
            ),
        ];

        let _leader_id = wait_for_leader(&nodes);

        // Create collection with both dense + sparse vectors.
        let sparse_cfg = HashMap::from([(
            "lex_sparse".to_string(),
            SparseVectorConfig {
                full_scan_threshold: 5000,
                modifier: crate::helix_engine::vector_core::sparse::SparseModifier::Idf,
                ..Default::default()
            },
        )]);
        let vectors = HashMap::from([(
            "dense".to_string(),
            NamedVectorConfig {
                size: 3,
                distance: crate::helix_engine::vector_core::named_vectors::DistanceMetric::Cosine,
                spindle: crate::helix_engine::vector_core::spindle::SpindleConfig::default(),
            },
        )]);
        nodes[0]
            .replication
            .apply(ReplicatedMutation::CreateCollection {
                name: "sp_repo".into(),
                vectors,
                sparse_vectors: sparse_cfg,
                hnsw_overrides: None,
            })
            .unwrap();

        // Upsert a point with both dense and sparse vectors.
        use crate::helix_engine::vector_core::sparse::SparseVector;
        nodes[1]
            .replication
            .apply(ReplicatedMutation::UpsertPoints {
                collection: "sp_repo".into(),
                points: vec![ReplicatedPoint {
                    id: 100,
                    vectors: HashMap::from([("dense".into(), vec![0.5, 0.5, 0.0])]),
                    sparse_vectors: HashMap::from([(
                        "lex_sparse".into(),
                        SparseVector {
                            indices: vec![42, 99],
                            values: vec![3.0, 7.0],
                        },
                    )]),
                    payload: HashMap::from([("lang".into(), Value::String("rust".into()))]),
                }],
            })
            .unwrap();

        // Wait for replication to all nodes.
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            let mut all_ready = true;
            for node in &nodes {
                let storage = match node.collections.get_collection("sp_repo") {
                    Ok(s) => s,
                    Err(_) => {
                        all_ready = false;
                        break;
                    }
                };
                let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
                if storage.get_node(&txn, &100).is_err() {
                    all_ready = false;
                    break;
                }
                // Check sparse vector is searchable.
                let search_ok = storage
                    .named_vectors
                    .with_sparse_core("lex_sparse", |core| core.has_doc(&txn, 100));
                if search_ok.unwrap_or(false) != true {
                    all_ready = false;
                    break;
                }
            }
            if all_ready {
                break;
            }
            thread::sleep(Duration::from_millis(100));
        }

        // Verify sparse search returns results on every node.
        for (i, node) in nodes.iter().enumerate() {
            let storage = node.collections.get_collection("sp_repo").unwrap();
            let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
            let results = storage
                .named_vectors
                .with_sparse_core("lex_sparse", |core| {
                    core.search::<fn(u128) -> bool>(
                        &txn,
                        &SparseVector {
                            indices: vec![42],
                            values: vec![1.0],
                        },
                        10,
                        None,
                    )
                })
                .unwrap();
            assert!(
                !results.is_empty(),
                "node {} sparse search returned empty",
                i
            );
            assert_eq!(results[0].0, 100);
        }
    }

    #[test]
    fn bulk_upsert_finalizes_dense_tail_after_large_batch() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = Config::new(8, 32, 64, 1);
        config.vector_config.flat_scan_threshold = Some(2);
        let collections =
            Arc::new(CollectionManager::new(temp_dir.path().join("data"), config.clone()).unwrap());

        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::CreateCollection {
                name: "repo".into(),
                vectors: HashMap::from([(
                    "dense".into(),
                    NamedVectorConfig {
                        size: 3,
                        distance:
                            crate::helix_engine::vector_core::named_vectors::DistanceMetric::Cosine,
                        spindle: crate::helix_engine::vector_core::spindle::SpindleConfig::default(
                        ),
                    },
                )]),
                sparse_vectors: HashMap::new(),
                hnsw_overrides: None,
            },
        )
        .unwrap();

        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::UpsertPoints {
                collection: "repo".into(),
                points: vec![
                    ReplicatedPoint {
                        id: 1,
                        vectors: HashMap::from([("dense".into(), vec![1.0, 0.0, 0.0])]),
                        sparse_vectors: HashMap::new(),
                        payload: HashMap::new(),
                    },
                    ReplicatedPoint {
                        id: 2,
                        vectors: HashMap::from([("dense".into(), vec![0.9, 0.1, 0.0])]),
                        sparse_vectors: HashMap::new(),
                        payload: HashMap::new(),
                    },
                    ReplicatedPoint {
                        id: 3,
                        vectors: HashMap::from([("dense".into(), vec![0.0, 1.0, 0.0])]),
                        sparse_vectors: HashMap::new(),
                        payload: HashMap::new(),
                    },
                ],
            },
        )
        .unwrap();

        // Background optimizer runs in a separate thread; poll until indexed.
        // The split-phase optimizer may briefly expose segment metadata before
        // the LMDB write transaction commits, causing transient read errors.
        // Tolerate those during polling.
        let storage = collections.get_collection("repo").unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut stats = None;
        while Instant::now() < deadline {
            if let Ok(txn) = storage.lmdb_env().unwrap().read_txn() {
                if let Ok(current) = storage.named_vectors.get_dense_space_stats(&txn, "dense") {
                    if current.indexed_vectors_count == 3 {
                        stats = Some(current);
                        break;
                    }
                }
            }
            thread::sleep(Duration::from_millis(50));
        }
        let stats = stats.expect("background optimizer did not index all vectors in time");
        assert_eq!(stats.vectors_count, 3);
        assert_eq!(stats.indexed_vectors_count, 3);

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let results = storage
            .named_vectors
            .dense_search_with_selectivity::<fn(&crate::helix_engine::vector_core::vector::HVector) -> bool>(
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
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
    }

    #[test]
    fn optimizer_converges_all_vectors_indexed() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("HELIX_INDEX_SUBTHRESHOLD_QUIESCED_TAILS");

        // Insert vectors across multiple batches including a sub-threshold tail,
        // then verify the convergence-looping optimizer indexes everything.
        let temp_dir = TempDir::new().unwrap();
        let mut config = Config::new(8, 32, 64, 1);
        // Low threshold so we get seal + tail behavior with few vectors
        config.vector_config.flat_scan_threshold = Some(3);
        let collections =
            Arc::new(CollectionManager::new(temp_dir.path().join("data"), config.clone()).unwrap());

        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::CreateCollection {
                name: "repo".into(),
                vectors: HashMap::from([(
                    "dense".into(),
                    NamedVectorConfig {
                        size: 4,
                        distance:
                            crate::helix_engine::vector_core::named_vectors::DistanceMetric::Cosine,
                        spindle: crate::helix_engine::vector_core::spindle::SpindleConfig::default(
                        ),
                    },
                )]),
                sparse_vectors: HashMap::new(),
                hnsw_overrides: None,
            },
        )
        .unwrap();

        // Batch 1: 3 vectors (hits threshold, triggers seal)
        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::UpsertPoints {
                collection: "repo".into(),
                points: (1..=3)
                    .map(|i| ReplicatedPoint {
                        id: i,
                        vectors: HashMap::from([("dense".into(), vec![i as f32, 0.0, 0.0, 0.0])]),
                        sparse_vectors: HashMap::new(),
                        payload: HashMap::new(),
                    })
                    .collect(),
            },
        )
        .unwrap();

        // Batch 2: 2 more vectors (sub-threshold tail — tests quiescence seal)
        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::UpsertPoints {
                collection: "repo".into(),
                points: (4..=5)
                    .map(|i| ReplicatedPoint {
                        id: i,
                        vectors: HashMap::from([("dense".into(), vec![0.0, i as f32, 0.0, 0.0])]),
                        sparse_vectors: HashMap::new(),
                        payload: HashMap::new(),
                    })
                    .collect(),
            },
        )
        .unwrap();

        // Wait for convergence: all 5 vectors should be indexed.
        // The quiescence seal triggers after 1s of no upserts.
        let storage = collections.get_collection("repo").unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut final_stats = None;
        while Instant::now() < deadline {
            if let Ok(txn) = storage.lmdb_env().unwrap().read_txn() {
                if let Ok(stats) = storage.named_vectors.get_dense_space_stats(&txn, "dense") {
                    if stats.indexed_vectors_count == 5 {
                        final_stats = Some(stats);
                        break;
                    }
                }
            }
            thread::sleep(Duration::from_millis(100));
        }
        let stats = final_stats
            .expect("optimizer did not converge: not all 5 vectors indexed within deadline");
        assert_eq!(stats.vectors_count, 5);
        assert_eq!(stats.indexed_vectors_count, 5);

        // With bounded multi-segment lifecycle, we no longer merge to 1
        // after quiescence. Segments stay separate. Just verify we're
        // within HELIX_DENSE_MAX_INDEXED_SEGMENTS + 1 mutable tail.
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let final_s = storage
            .named_vectors
            .get_dense_space_stats(&txn, "dense")
            .unwrap();
        let max_segments = NamedVectorManager::max_indexed_segments_target().saturating_add(1);
        assert!(
            final_s.segments_count <= max_segments,
            "Segments should be bounded, got {} > {}",
            final_s.segments_count,
            max_segments
        );
        drop(txn);

        // Verify search works correctly over the fully-indexed collection
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let results = storage
            .named_vectors
            .dense_search_with_selectivity::<fn(&crate::helix_engine::vector_core::vector::HVector) -> bool>(
                storage.lmdb_env().unwrap(),
                &txn,
                "dense",
                &[1.0, 0.0, 0.0, 0.0],
                5,
                None,
                true,
                None,
            )
            .unwrap();
        assert_eq!(results.len(), 5, "Should return all 5 vectors");
        let ids: Vec<u128> = results.iter().map(|v| v.id).collect();
        for i in 1u128..=5 {
            assert!(ids.contains(&i), "Missing vector {}", i);
        }
    }

    /// END-TO-END LSM optimizer proof: vectors are written into a LIVE
    /// collection through the *production* LSM upsert path
    /// (`apply_mutation(UpsertPoints)` → `apply_upsert_points_lsm_chunk` →
    /// `dense_append_to_mutable_be`, all on `BackendKind::Lsm`), then the
    /// background segment optimizer is driven through its real synchronous
    /// build entry (`run_index_job` with a production-shaped `IndexJob`).
    ///
    /// This asserts the optimizer actually drives the Mutable→Building→Indexed
    /// lifecycle to completion on LSM (`indexed_vectors_count > 0`, i.e.
    /// `has_index()` becomes true) and that an indexed search over the now-built
    /// HNSW returns the correct nearest neighbor. The existing
    /// `lsm_hnsw_indexed_turboprod_search_is_bounded` proves only the manual
    /// `prepare/flush/finalize` mechanics on a bare `VectorCore`; this proves the
    /// optimizer that production relies on does NOT silently skip the build on
    /// LSM (which would leave collections permanently flat / O(n)).
    ///
    /// The build is driven by calling `run_index_job` directly (the synchronous
    /// entry the executor worker invokes), so the test is deterministic rather
    /// than racing the async `INDEX_EXECUTOR` thread pool.
    #[test]
    fn optimizer_indexes_lsm_collection_end_to_end() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _backend = EnvVarGuard::set("HELIX_STORAGE_BACKEND", "lsm");
        let _in_mem = EnvVarGuard::set("HELIX_LSM_IN_MEMORY", "1");

        let temp_dir = TempDir::new().unwrap();
        let mut config = Config::new(8, 32, 64, 1);
        // Low indexing threshold so a handful of inserts crosses it and the
        // optimizer seals + builds instead of staying flat.
        config.vector_config.flat_scan_threshold = Some(3);
        let collections =
            Arc::new(CollectionManager::new(temp_dir.path().join("data"), config.clone()).unwrap());

        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::CreateCollection {
                name: "repo".into(),
                vectors: HashMap::from([(
                    "dense".into(),
                    NamedVectorConfig {
                        size: 4,
                        distance:
                            crate::helix_engine::vector_core::named_vectors::DistanceMetric::Cosine,
                        spindle: crate::helix_engine::vector_core::spindle::SpindleConfig::default(
                        ),
                    },
                )]),
                sparse_vectors: HashMap::new(),
                hnsw_overrides: None,
            },
        )
        .unwrap();

        let storage = collections.get_collection("repo").unwrap();
        assert_eq!(
            storage.backend.kind(),
            BackendKind::Lsm,
            "test harness must run on the LSM backend"
        );

        // Production LSM write path: route the upsert through the real
        // `apply_mutation` dispatch, which on LSM lands in
        // `apply_upsert_points_lsm_chunk` → `dense_append_to_mutable_be`.
        // 6 vectors > threshold(3): the mutable tail must be sealed + built.
        const N: u128 = 6;
        let vectors: Vec<[f32; 4]> = (1..=N)
            .map(|i| {
                // Well-separated directions so the nearest neighbor of query[0] is
                // UNAMBIGUOUSLY id=1 even under hvtq quantization. The old scheme
                // `v[(i-1)%4]=1.0` wrapped id=5 back onto +x (same axis as id=1) and
                // leaned on a 0.001 tilt to break the tie — but that tilt is below
                // quantization resolution, so id=1 vs id=5 became a non-deterministic
                // quantized tie (flaky [5,1,6] vs [1,..]). Put ids 1-4 on +axes and
                // ids 5-6 on -axes: id=1[+x] is closest to its own query and id=5[-x]
                // is the FARTHEST (cosine -1), so no two collide in quantized space.
                let mut v = [0.0f32; 4];
                let axis = (i as usize - 1) % 4;
                v[axis] = if i <= 4 { 1.0 } else { -1.0 };
                v[0] += 0.001 * i as f32; // tiny tilt to keep ids strictly distinct
                v
            })
            .collect();
        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::UpsertPoints {
                collection: "repo".into(),
                points: (1..=N)
                    .map(|i| ReplicatedPoint {
                        id: i,
                        vectors: HashMap::from([(
                            "dense".into(),
                            vectors[(i - 1) as usize].to_vec(),
                        )]),
                        sparse_vectors: HashMap::new(),
                        payload: HashMap::new(),
                    })
                    .collect(),
            },
        )
        .unwrap();

        // Before the optimizer runs, the vectors live on the flat tail: stats
        // count them but none are indexed yet. (The async IndexJob submitted by
        // apply_mutation may or may not have started; assert on the storage
        // state we control rather than racing it.)
        let pre_stats = {
            let r = storage.backend.begin_read().unwrap();
            storage
                .named_vectors
                .get_dense_space_stats_be(&r, "dense")
                .unwrap()
        };
        assert_eq!(
            pre_stats.vectors_count, N as u64,
            "all {} flat vectors must be present on LSM before indexing",
            N
        );

        // Drive the optimizer DETERMINISTICALLY through its real synchronous
        // build entry. This is exactly the job the INDEX_EXECUTOR worker (and
        // the backlog sweeper) runs; constructing it here removes the async
        // race while exercising the production seal→build→promote path.
        let flat_scan_threshold = storage
            .named_vectors
            .effective_indexing_threshold(config.vector_flat_scan_threshold());
        let job = IndexJob {
            collection: "repo".to_string(),
            storage: Arc::downgrade(&storage),
            spaces_to_check: vec!["dense".to_string()],
            hnsw_config: HNSWConfig::new(Some(8), Some(32), Some(64)),
            flat_scan_threshold,
        };
        run_index_job(job);

        // After the optimizer: the Mutable→Building→Indexed lifecycle must have
        // completed on LSM. has_index() is true for the built segment(s), so
        // indexed_vectors_count is non-zero. A backend that silently skipped the
        // build on LSM would leave this at 0 (collection permanently flat).
        let post_stats = {
            let r = storage.backend.begin_read().unwrap();
            storage
                .named_vectors
                .get_dense_space_stats_be(&r, "dense")
                .unwrap()
        };
        assert!(
            post_stats.indexed_vectors_count > 0,
            "optimizer did NOT index the LSM collection: indexed_vectors_count=0 \
             (vectors_count={}, segments={}); the Mutable→Indexed lifecycle was \
             skipped on the LSM backend",
            post_stats.vectors_count,
            post_stats.segments_count
        );
        assert_eq!(
            post_stats.vectors_count, N as u64,
            "indexing must not drop or duplicate flat vectors on LSM"
        );

        // Directly confirm has_index() on the built segment core(s) via the
        // backend (not a heed txn): at least one segment carries an HNSW entry
        // point on the LSM backend.
        let any_segment_has_index = {
            let cores = storage.named_vectors.cores_read().unwrap();
            cores.values().any(|core| {
                let r = core.backend.begin_read().unwrap();
                core.has_index(&r).unwrap_or(false)
            })
        };
        assert!(
            any_segment_has_index,
            "no dense segment carries an HNSW index on LSM after the optimizer ran"
        );
    }

    #[test]
    fn optimizer_can_leave_subthreshold_quiesced_tail_flat() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("HELIX_INDEX_SUBTHRESHOLD_QUIESCED_TAILS", "0");

        let temp_dir = TempDir::new().unwrap();
        let mut config = Config::new(8, 32, 64, 1);
        config.vector_config.flat_scan_threshold = Some(3);
        let collections =
            Arc::new(CollectionManager::new(temp_dir.path().join("data"), config.clone()).unwrap());

        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::CreateCollection {
                name: "repo".into(),
                vectors: HashMap::from([(
                    "dense".into(),
                    NamedVectorConfig {
                        size: 4,
                        distance:
                            crate::helix_engine::vector_core::named_vectors::DistanceMetric::Cosine,
                        spindle: crate::helix_engine::vector_core::spindle::SpindleConfig::default(
                        ),
                    },
                )]),
                sparse_vectors: HashMap::new(),
                hnsw_overrides: None,
            },
        )
        .unwrap();

        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::UpsertPoints {
                collection: "repo".into(),
                points: (1..=2)
                    .map(|i| ReplicatedPoint {
                        id: i,
                        vectors: HashMap::from([("dense".into(), vec![i as f32, 0.0, 0.0, 0.0])]),
                        sparse_vectors: HashMap::new(),
                        payload: HashMap::new(),
                    })
                    .collect(),
            },
        )
        .unwrap();

        let storage = collections.get_collection("repo").unwrap();
        thread::sleep(Duration::from_millis(500));
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let stats = storage
            .named_vectors
            .get_dense_space_stats(&txn, "dense")
            .unwrap();
        assert_eq!(stats.vectors_count, 2);
        assert_eq!(stats.indexed_vectors_count, 0);

        std::env::remove_var("HELIX_INDEX_SUBTHRESHOLD_QUIESCED_TAILS");
    }

    /// Fix B: `estimate_merge_flush_bytes` returns a lower bound that at
    /// least covers the raw vector bytes plus per-vector HNSW overhead.
    #[test]
    fn estimate_merge_flush_bytes_covers_vectors_and_hnsw() {
        use crate::helix_engine::vector_core::named_vectors::PreparedMerge;
        use crate::helix_engine::vector_core::vector_core::PreparedIndex;

        // 100 vectors, 384 dims each.
        let n = 100usize;
        let dims = 384usize;
        let point_ids: Vec<u128> = (0..n).map(|i| i as u128).collect();
        let original_data = (0..n).map(|_| Some(vec![0.0f32; dims])).collect();
        let point_fields = (0..n)
            .map(|i| {
                let mut fields = std::collections::HashMap::new();
                fields.insert("name".to_string(), Value::String("hello".to_string()));
                fields.insert("row".to_string(), Value::I32(i as i32));
                fields
            })
            .collect();
        let merge = PreparedMerge {
            merge_targets: Vec::new(),
            prepared_index: PreparedIndex {
                point_ids,
                original_data,
                encoded_data: vec![None; n],
                point_fields,
                levels: vec![0; n],
                adjacency: vec![Some(vec![Vec::new()]); n],
                entry_ord: 0,
                level_zero_preflushed: false,
            },
        };

        let est = super::estimate_merge_flush_bytes(&merge);
        let raw_vector_bytes = n * dims * 4;
        let hnsw_floor = n * super::HNSW_PER_VECTOR_OVERHEAD_BYTES;
        assert!(
            est >= raw_vector_bytes + hnsw_floor,
            "estimate {} below floor vectors({}) + hnsw({})",
            est,
            raw_vector_bytes,
            hnsw_floor
        );
    }

    #[test]
    fn budgeted_merge_flat_chunk_len_caps_estimated_bytes() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::unset("HELIX_MERGE_FLAT_BYTE_BUDGET");
        let items = vec![
            (1, vec![0.0f32; 300_000], HashMap::new()),
            (2, vec![0.0f32; 300_000], HashMap::new()),
            (3, vec![0.0f32; 1], HashMap::new()),
        ];

        let (len, stats) = budgeted_merge_flat_chunk_len(&items, 8192);

        assert_eq!(
            len, 1,
            "two 1.2MiB vectors would exceed the 2MiB flat budget"
        );
        assert!(stats.estimated_bytes <= merge_flat_byte_budget());
        assert_eq!(stats.hnsw_edges, 0);
    }

    #[test]
    fn budgeted_merge_flat_chunk_len_caps_row_count() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::unset("HELIX_MERGE_FLAT_ROW_CAP");
        let items: Vec<_> = (0..600u128)
            .map(|id| (id, vec![0.0f32; 1], HashMap::new()))
            .collect();

        let (len, stats) = budgeted_merge_flat_chunk_len(&items, 8192);

        assert_eq!(len, MERGE_FLAT_ROW_CAP);
        assert!(stats.estimated_bytes <= merge_flat_byte_budget());
        assert_eq!(stats.hnsw_edges, 0);
    }

    #[test]
    fn budgeted_merge_flat_chunk_len_respects_byte_budget_override() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::set("HELIX_MERGE_FLAT_BYTE_BUDGET", "350000");
        let items = vec![
            (1, vec![0.0f32; 80_000], HashMap::new()),
            (2, vec![0.0f32; 80_000], HashMap::new()),
            (3, vec![0.0f32; 1], HashMap::new()),
        ];

        let (len, stats) = budgeted_merge_flat_chunk_len(&items, 8192);

        assert_eq!(len, 1);
        assert!(stats.estimated_bytes <= merge_flat_byte_budget());
    }

    #[test]
    fn budgeted_merge_flat_chunk_len_respects_row_cap_override() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::unset("HELIX_MERGE_FLAT_ROW_CAP");
        let items: Vec<_> = (0..10u128)
            .map(|id| (id, vec![0.0f32; 1], HashMap::new()))
            .collect();

        std::env::set_var("HELIX_MERGE_FLAT_ROW_CAP", "3");
        let (len, _) = budgeted_merge_flat_chunk_len(&items, 8192);
        assert_eq!(len, 3);

        std::env::set_var("HELIX_MERGE_FLAT_ROW_CAP", "0");
        let (len, _) = budgeted_merge_flat_chunk_len(&items, 8192);
        assert_eq!(len, MERGE_FLAT_ROW_CAP.min(items.len()));
    }

    #[test]
    fn budgeted_merge_index_chunk_len_caps_hnsw_edges() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _byte_env = EnvVarGuard::unset("HELIX_MERGE_INDEX_BYTE_BUDGET");
        let _edge_env = EnvVarGuard::unset("HELIX_MERGE_INDEX_HNSW_EDGE_BUDGET");
        use crate::helix_engine::vector_core::vector_core::PreparedIndex;

        let prepared = PreparedIndex {
            point_ids: vec![1, 2, 3],
            original_data: vec![
                Some(vec![0.0f32; 4]),
                Some(vec![0.0f32; 4]),
                Some(vec![0.0f32; 4]),
            ],
            encoded_data: vec![None; 3],
            point_fields: Vec::new(),
            levels: vec![0, 0, 0],
            adjacency: vec![
                Some(vec![(0..3_000).map(|i| i as u32).collect()]),
                Some(vec![(0..3_000).map(|i| (3_000 + i) as u32).collect()]),
                Some(vec![vec![2]]),
            ],
            entry_ord: 0,
            level_zero_preflushed: false,
        };

        let (len, stats) = budgeted_merge_index_chunk_len(&prepared, 0, 8192);

        assert_eq!(
            len, 1,
            "two 3K-edge points would exceed the 4K HNSW edge budget"
        );
        assert!(stats.hnsw_edges <= merge_index_hnsw_edge_budget());
    }

    #[test]
    fn budgeted_merge_index_chunk_len_respects_budget_overrides() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _byte_env = EnvVarGuard::set("HELIX_MERGE_INDEX_BYTE_BUDGET", "700000");
        let _edge_env = EnvVarGuard::set("HELIX_MERGE_INDEX_HNSW_EDGE_BUDGET", "1024");
        use crate::helix_engine::vector_core::vector_core::PreparedIndex;

        let prepared = PreparedIndex {
            point_ids: vec![1, 2, 3],
            original_data: vec![
                Some(vec![0.0f32; 80_000]),
                Some(vec![0.0f32; 80_000]),
                Some(vec![0.0f32; 1]),
            ],
            encoded_data: vec![None; 3],
            point_fields: Vec::new(),
            levels: vec![0, 0, 0],
            adjacency: vec![
                Some(vec![(0..700).map(|i| i as u32).collect()]),
                Some(vec![(700..1_400).map(|i| i as u32).collect()]),
                Some(vec![vec![2]]),
            ],
            entry_ord: 0,
            level_zero_preflushed: false,
        };

        let (len, stats) = budgeted_merge_index_chunk_len(&prepared, 0, 8192);

        assert_eq!(len, 1);
        assert!(stats.estimated_bytes <= merge_index_byte_budget());
        assert!(stats.hnsw_edges <= merge_index_hnsw_edge_budget());
    }

    /// The content hash is load-bearing for the re-upsert skip path:
    /// - Identical payload + vectors → same hash (skip)
    /// - Any material change → different hash (no skip)
    /// - HashMap insertion order must not change the hash
    /// - The reserved CONTENT_HASH_KEY itself is excluded from the hash
    ///   so stamping it into stored payload doesn't feed back into the
    ///   next comparison
    #[test]
    fn content_hash_matches_on_identical_payload() {
        use crate::protocol::value::Value;
        let mut payload_a = HashMap::new();
        payload_a.insert("alpha".into(), Value::String("one".into()));
        payload_a.insert("beta".into(), Value::U32(42));
        payload_a.insert("gamma".into(), Value::F32(1.5));

        let mut payload_b = HashMap::new();
        payload_b.insert("gamma".into(), Value::F32(1.5));
        payload_b.insert("alpha".into(), Value::String("one".into()));
        payload_b.insert("beta".into(), Value::U32(42));

        let mut dense = HashMap::new();
        dense.insert("default".to_string(), vec![0.1_f32, 0.2, 0.3]);

        let mut sparse = HashMap::new();
        sparse.insert(
            "lex".to_string(),
            SparseVector {
                indices: vec![3, 1, 2],
                values: vec![0.3, 0.1, 0.2],
            },
        );

        let point_a = ReplicatedPoint {
            id: 7,
            vectors: dense.clone(),
            sparse_vectors: sparse.clone(),
            payload: payload_a,
        };
        let point_b = ReplicatedPoint {
            id: 7,
            vectors: dense,
            sparse_vectors: sparse,
            payload: payload_b,
        };
        assert_eq!(
            compute_content_hash(&point_a),
            compute_content_hash(&point_b),
            "hash must be stable across HashMap insertion orders"
        );
    }

    #[test]
    fn content_hash_changes_on_payload_edit() {
        use crate::protocol::value::Value;
        let dense = HashMap::from([("d".to_string(), vec![0.1_f32])]);
        let sparse = HashMap::new();

        let base = ReplicatedPoint {
            id: 11,
            vectors: dense.clone(),
            sparse_vectors: sparse.clone(),
            payload: HashMap::from([("k".to_string(), Value::U32(1))]),
        };
        let edited = ReplicatedPoint {
            id: 11,
            vectors: dense,
            sparse_vectors: sparse,
            payload: HashMap::from([("k".to_string(), Value::U32(2))]),
        };
        assert_ne!(
            compute_content_hash(&base),
            compute_content_hash(&edited),
            "hash must change when a payload value changes"
        );
    }

    #[test]
    fn content_hash_ignores_payload_only_operational_keys() {
        use crate::protocol::value::Value;
        let dense = HashMap::from([("d".to_string(), vec![0.1_f32])]);
        let sparse = HashMap::new();

        let base = ReplicatedPoint {
            id: 12,
            vectors: dense.clone(),
            sparse_vectors: sparse.clone(),
            payload: HashMap::from([("path".to_string(), Value::String("src/lib.rs".into()))]),
        };
        let with_payload_only_keys = ReplicatedPoint {
            id: 12,
            vectors: dense,
            sparse_vectors: sparse,
            payload: HashMap::from([
                ("path".to_string(), Value::String("src/lib.rs".into())),
                (
                    "caller_point_id".to_string(),
                    Value::String("old-id".into()),
                ),
                ("_pending_prune".to_string(), Value::Boolean(true)),
                ("_pending_prune_at".to_string(), Value::U64(123)),
            ]),
        };

        assert_eq!(
            compute_content_hash(&base),
            compute_content_hash(&with_payload_only_keys),
            "payload-only operational fields must not force vector/sparse rewrites"
        );
    }

    #[test]
    fn content_hash_ignores_top_level_line_coordinates() {
        use crate::protocol::value::Value;
        let dense = HashMap::from([("d".to_string(), vec![0.1_f32])]);
        let sparse = HashMap::new();

        let base = ReplicatedPoint {
            id: 14,
            vectors: dense.clone(),
            sparse_vectors: sparse.clone(),
            payload: HashMap::from([("path".to_string(), Value::String("src/lib.rs".into()))]),
        };
        let with_line_coords = ReplicatedPoint {
            id: 14,
            vectors: dense,
            sparse_vectors: sparse,
            payload: HashMap::from([
                ("path".to_string(), Value::String("src/lib.rs".into())),
                ("start_line".to_string(), Value::U64(10)),
                ("end_line".to_string(), Value::U64(42)),
                ("start_col".to_string(), Value::U64(4)),
                ("end_col".to_string(), Value::U64(12)),
            ]),
        };

        assert_eq!(
            compute_content_hash(&base),
            compute_content_hash(&with_line_coords),
            "top-level line coordinates must not force vector/sparse rewrites"
        );
    }

    #[test]
    fn content_hash_ignores_metadata_host_and_container_path() {
        use crate::protocol::value::Value;
        let dense = HashMap::from([("d".to_string(), vec![0.1_f32])]);
        let sparse = HashMap::new();

        let metadata_v1 = Value::Object(HashMap::from([
            ("path".to_string(), Value::String("src/lib.rs".into())),
            (
                "host_path".to_string(),
                Value::String("/Users/alice/repo/src/lib.rs".into()),
            ),
            (
                "container_path".to_string(),
                Value::String("/work/repo-aaaaaaaaaaaaaaaa/src/lib.rs".into()),
            ),
        ]));
        let metadata_v2 = Value::Object(HashMap::from([
            ("path".to_string(), Value::String("src/lib.rs".into())),
            (
                "host_path".to_string(),
                Value::String("/Users/bob/Downloads/repo/src/lib.rs".into()),
            ),
            (
                "container_path".to_string(),
                Value::String("/work/repo-bbbbbbbbbbbbbbbb/src/lib.rs".into()),
            ),
        ]));

        let p_v1 = ReplicatedPoint {
            id: 31,
            vectors: dense.clone(),
            sparse_vectors: sparse.clone(),
            payload: HashMap::from([("metadata".to_string(), metadata_v1)]),
        };
        let p_v2 = ReplicatedPoint {
            id: 31,
            vectors: dense,
            sparse_vectors: sparse,
            payload: HashMap::from([("metadata".to_string(), metadata_v2)]),
        };

        assert_eq!(
            compute_content_hash(&p_v1),
            compute_content_hash(&p_v2),
            "host_path and container_path are unindexed per CE contract; \
             different developer/container prefixes must not change the hash"
        );
    }

    #[test]
    fn content_hash_ignores_metadata_payload_only_subkeys() {
        use crate::protocol::value::Value;
        let dense = HashMap::from([("d".to_string(), vec![0.1_f32])]);
        let sparse = HashMap::new();

        let metadata_v1 = Value::Object(HashMap::from([
            ("path".to_string(), Value::String("src/lib.rs".into())),
            ("ingested_at".to_string(), Value::U64(1_700_000_000)),
            ("churn_count".to_string(), Value::U64(3)),
            ("author_count".to_string(), Value::U64(2)),
            ("_graph_backfilled".to_string(), Value::Boolean(false)),
            (
                "_graph_backfilled_version".to_string(),
                Value::String("calls-v1".into()),
            ),
            ("_pseudo_backfilled".to_string(), Value::Boolean(false)),
        ]));
        let metadata_v2 = Value::Object(HashMap::from([
            ("path".to_string(), Value::String("src/lib.rs".into())),
            ("ingested_at".to_string(), Value::U64(1_900_000_000)),
            ("churn_count".to_string(), Value::U64(7)),
            ("author_count".to_string(), Value::U64(5)),
            ("_graph_backfilled".to_string(), Value::Boolean(true)),
            (
                "_graph_backfilled_version".to_string(),
                Value::String("calls-v2".into()),
            ),
            ("_pseudo_backfilled".to_string(), Value::Boolean(true)),
        ]));

        let p_v1 = ReplicatedPoint {
            id: 21,
            vectors: dense.clone(),
            sparse_vectors: sparse.clone(),
            payload: HashMap::from([("metadata".to_string(), metadata_v1)]),
        };
        let p_v2 = ReplicatedPoint {
            id: 21,
            vectors: dense,
            sparse_vectors: sparse,
            payload: HashMap::from([("metadata".to_string(), metadata_v2)]),
        };

        assert_eq!(
            compute_content_hash(&p_v1),
            compute_content_hash(&p_v2),
            "operational metadata subkeys must not change the content hash"
        );
    }

    #[test]
    fn content_hash_ignores_metadata_bookkeeping_subkeys() {
        use crate::protocol::value::Value;
        let dense = HashMap::from([("d".to_string(), vec![0.1_f32])]);
        let sparse = HashMap::new();

        let metadata_v1 = Value::Object(HashMap::from([
            ("path".to_string(), Value::String("src/lib.rs".into())),
            (
                "file_hash".to_string(),
                Value::String("old-file-hash".into()),
            ),
            ("file_size_bytes".to_string(), Value::U64(1024)),
            ("last_modified_at".to_string(), Value::U64(1_700_000_000)),
            ("indexed_branch".to_string(), Value::String("main".into())),
            (
                "git_branches".to_string(),
                Value::Array(vec![Value::String("main".into())]),
            ),
            ("start_line".to_string(), Value::U64(1)),
            ("end_line".to_string(), Value::U64(12)),
            ("start_col".to_string(), Value::U64(0)),
            ("end_col".to_string(), Value::U64(8)),
        ]));
        let metadata_v2 = Value::Object(HashMap::from([
            ("path".to_string(), Value::String("src/lib.rs".into())),
            (
                "file_hash".to_string(),
                Value::String("new-file-hash".into()),
            ),
            ("file_size_bytes".to_string(), Value::U64(2048)),
            ("last_modified_at".to_string(), Value::U64(1_900_000_000)),
            (
                "indexed_branch".to_string(),
                Value::String("feature".into()),
            ),
            (
                "git_branches".to_string(),
                Value::Array(vec![
                    Value::String("main".into()),
                    Value::String("feature".into()),
                ]),
            ),
            ("start_line".to_string(), Value::U64(2)),
            ("end_line".to_string(), Value::U64(15)),
            ("start_col".to_string(), Value::U64(1)),
            ("end_col".to_string(), Value::U64(10)),
        ]));

        let p_v1 = ReplicatedPoint {
            id: 22,
            vectors: dense.clone(),
            sparse_vectors: sparse.clone(),
            payload: HashMap::from([("metadata".to_string(), metadata_v1)]),
        };
        let p_v2 = ReplicatedPoint {
            id: 22,
            vectors: dense,
            sparse_vectors: sparse,
            payload: HashMap::from([("metadata".to_string(), metadata_v2)]),
        };

        assert_eq!(
            compute_content_hash(&p_v1),
            compute_content_hash(&p_v2),
            "metadata bookkeeping fields must not force vector/sparse rewrites"
        );
    }

    #[test]
    fn content_hash_changes_on_index_relevant_metadata_subkey() {
        use crate::protocol::value::Value;
        let dense = HashMap::from([("d".to_string(), vec![0.1_f32])]);
        let sparse = HashMap::new();

        let metadata_v1 = Value::Object(HashMap::from([
            ("path".to_string(), Value::String("src/lib.rs".into())),
            ("ingested_at".to_string(), Value::U64(1_700_000_000)),
        ]));
        let metadata_v2 = Value::Object(HashMap::from([
            ("path".to_string(), Value::String("src/main.rs".into())),
            ("ingested_at".to_string(), Value::U64(1_700_000_000)),
        ]));

        let p_v1 = ReplicatedPoint {
            id: 22,
            vectors: dense.clone(),
            sparse_vectors: sparse.clone(),
            payload: HashMap::from([("metadata".to_string(), metadata_v1)]),
        };
        let p_v2 = ReplicatedPoint {
            id: 22,
            vectors: dense,
            sparse_vectors: sparse,
            payload: HashMap::from([("metadata".to_string(), metadata_v2)]),
        };

        assert_ne!(
            compute_content_hash(&p_v1),
            compute_content_hash(&p_v2),
            "index-relevant metadata subkeys (e.g. path) must change the content hash"
        );
    }

    #[test]
    fn first_payload_diff_key_surfaces_index_relevant_metadata_subkey() {
        use crate::protocol::value::Value;
        let stored = HashMap::from([(
            "metadata".to_string(),
            Value::Object(HashMap::from([
                ("path".to_string(), Value::String("src/lib.rs".into())),
                ("ingested_at".to_string(), Value::U64(1_700_000_000)),
            ])),
        )]);
        let incoming = HashMap::from([(
            "metadata".to_string(),
            Value::Object(HashMap::from([
                ("path".to_string(), Value::String("src/main.rs".into())),
                ("ingested_at".to_string(), Value::U64(1_900_000_000)),
            ])),
        )]);
        assert_eq!(
            first_payload_diff_key(&stored, &incoming),
            Some("metadata.path".into()),
            "metadata diff with index-relevant subkey change must report the nested key"
        );
    }

    #[test]
    fn first_payload_diff_key_surfaces_added_metadata_subkey() {
        use crate::protocol::value::Value;
        let stored = HashMap::from([(
            "metadata".to_string(),
            Value::Object(HashMap::from([(
                "path".to_string(),
                Value::String("src/lib.rs".into()),
            )])),
        )]);
        let incoming = HashMap::from([(
            "metadata".to_string(),
            Value::Object(HashMap::from([
                ("path".to_string(), Value::String("src/lib.rs".into())),
                ("symbol".to_string(), Value::String("compute".into())),
            ])),
        )]);

        assert_eq!(
            first_payload_diff_key(&stored, &incoming),
            Some("metadata.symbol".into()),
            "metadata diff should report added index-relevant nested keys"
        );
    }

    #[test]
    fn first_payload_diff_key_reports_metadata_for_shape_change() {
        use crate::protocol::value::Value;
        let stored = HashMap::from([(
            "metadata".to_string(),
            Value::String("legacy-metadata".into()),
        )]);
        let incoming = HashMap::from([(
            "metadata".to_string(),
            Value::Object(HashMap::from([(
                "path".to_string(),
                Value::String("src/lib.rs".into()),
            )])),
        )]);

        assert_eq!(
            first_payload_diff_key(&stored, &incoming),
            Some("metadata".into()),
            "metadata object/type changes should fall back to the top-level key"
        );
    }

    #[test]
    fn first_payload_diff_key_ignores_metadata_payload_only_subkeys() {
        use crate::protocol::value::Value;
        let stored = HashMap::from([(
            "metadata".to_string(),
            Value::Object(HashMap::from([
                ("path".to_string(), Value::String("src/lib.rs".into())),
                ("ingested_at".to_string(), Value::U64(1_700_000_000)),
                ("churn_count".to_string(), Value::U64(3)),
            ])),
        )]);
        let incoming = HashMap::from([(
            "metadata".to_string(),
            Value::Object(HashMap::from([
                ("path".to_string(), Value::String("src/lib.rs".into())),
                ("ingested_at".to_string(), Value::U64(1_900_000_000)),
                ("churn_count".to_string(), Value::U64(8)),
            ])),
        )]);
        assert_eq!(
            first_payload_diff_key(&stored, &incoming),
            None,
            "metadata diff caused only by operational subkeys must not surface as a diff"
        );
    }

    #[test]
    fn first_payload_diff_key_ignores_payload_only_keys() {
        use crate::protocol::value::Value;
        let stored = HashMap::from([
            ("path".to_string(), Value::String("src/lib.rs".into())),
            (
                "caller_point_id".to_string(),
                Value::String("old-id".into()),
            ),
        ]);
        let incoming = HashMap::from([
            ("path".to_string(), Value::String("src/lib.rs".into())),
            (
                "caller_point_id".to_string(),
                Value::String("new-id".into()),
            ),
        ]);
        assert_eq!(
            first_payload_diff_key(&stored, &incoming),
            None,
            "diff key must skip allowlisted payload-only fields"
        );
    }

    #[test]
    fn first_payload_diff_key_returns_index_relevant_diff() {
        use crate::protocol::value::Value;
        let stored = HashMap::from([
            ("chunk_text".to_string(), Value::String("v1".into())),
            (
                "caller_point_id".to_string(),
                Value::String("old-id".into()),
            ),
        ]);
        let incoming = HashMap::from([
            ("chunk_text".to_string(), Value::String("v2".into())),
            (
                "caller_point_id".to_string(),
                Value::String("new-id".into()),
            ),
        ]);
        assert_eq!(
            first_payload_diff_key(&stored, &incoming),
            Some("chunk_text".into()),
            "diff key must surface the actual index-relevant field, not the allowlisted one"
        );
    }

    #[test]
    fn first_payload_diff_key_ignores_content_hash_marker() {
        use crate::protocol::value::Value;
        let stored = HashMap::from([
            ("path".to_string(), Value::String("src/lib.rs".into())),
            (CONTENT_HASH_KEY.to_string(), Value::U64(123)),
        ]);
        let incoming = HashMap::from([
            ("path".to_string(), Value::String("src/lib.rs".into())),
            (CONTENT_HASH_KEY.to_string(), Value::U64(456)),
        ]);
        assert_eq!(
            first_payload_diff_key(&stored, &incoming),
            None,
            "stamped content hash must not be reported as a diff"
        );
    }

    #[test]
    fn content_hash_changes_on_dense_vector_edit() {
        let payload = HashMap::new();
        let sparse = HashMap::new();

        let base = ReplicatedPoint {
            id: 13,
            vectors: HashMap::from([("d".to_string(), vec![0.1_f32, 0.2])]),
            sparse_vectors: sparse.clone(),
            payload: payload.clone(),
        };
        let edited = ReplicatedPoint {
            id: 13,
            vectors: HashMap::from([("d".to_string(), vec![0.1_f32, 0.21])]),
            sparse_vectors: sparse,
            payload,
        };
        assert_ne!(
            compute_content_hash(&base),
            compute_content_hash(&edited),
            "hash must change when dense vector floats change"
        );
    }

    #[test]
    fn content_hash_changes_on_sparse_term_edit() {
        let payload = HashMap::new();
        let dense = HashMap::new();

        let base = ReplicatedPoint {
            id: 17,
            vectors: dense.clone(),
            sparse_vectors: HashMap::from([(
                "lex".to_string(),
                SparseVector {
                    indices: vec![1, 2],
                    values: vec![0.1, 0.2],
                },
            )]),
            payload: payload.clone(),
        };
        let edited = ReplicatedPoint {
            id: 17,
            vectors: dense,
            sparse_vectors: HashMap::from([(
                "lex".to_string(),
                SparseVector {
                    indices: vec![1, 2],
                    values: vec![0.1, 0.25],
                },
            )]),
            payload,
        };
        assert_ne!(
            compute_content_hash(&base),
            compute_content_hash(&edited),
            "hash must change when a sparse value changes"
        );
    }

    /// Direct unit coverage for the f32 -> f16 bit conversion. Avoids
    /// the process-wide env var hazard: parallel cargo tests that flip
    /// HELIX_CONTENT_HASH_QUANTIZE race each other. Testing the pure
    /// helper is deterministic and doesn't touch global state.
    #[test]
    fn f16_conversion_absorbs_ulp_noise() {
        // Fp32 ULP at 0.5 is ~6e-8. f16 step at 0.5 is ~4.88e-4, so
        // dozens of fp32 ULPs must round to the same f16 bucket. This
        // is the load-bearing property: embedder re-runs producing
        // floats that differ by a few ULPs must hash equal.
        let base = 0.5_f32;
        let noisy = f32::from_bits(base.to_bits() + 1);
        assert_eq!(
            f32_to_f16_bits(base),
            f32_to_f16_bits(noisy),
            "f16 must absorb a single fp32 ULP"
        );
        // 1e-7 scale drift is well below the f16 step size at 0.5, so
        // it must also round to the same bucket.
        assert_eq!(
            f32_to_f16_bits(0.5),
            f32_to_f16_bits(0.5_f32 + 1e-7),
            "f16 must absorb 1e-7 drift"
        );
    }

    #[test]
    fn f16_conversion_detects_real_change() {
        assert_ne!(
            f32_to_f16_bits(0.5),
            f32_to_f16_bits(0.6),
            "f16 must distinguish 10% differences"
        );
        assert_ne!(
            f32_to_f16_bits(0.0),
            f32_to_f16_bits(1.0),
            "f16 must distinguish zero from one"
        );
        assert_ne!(
            f32_to_f16_bits(-0.5),
            f32_to_f16_bits(0.5),
            "f16 must distinguish sign flips"
        );
    }

    #[test]
    fn f16_quantized_bytes_padding_stable_width() {
        // Width stays 4 bytes regardless of mode so the hash stream
        // length doesn't change between modes. Important so a future
        // mode switch doesn't silently invalidate every stored hash by
        // shifting byte positions downstream.
        let a = f32_to_quantized_bytes(0.5, HashQuantization::Bits);
        let b = f32_to_quantized_bytes(0.5, HashQuantization::F16);
        assert_eq!(a.len(), 4);
        assert_eq!(b.len(), 4);
    }

    /// End-to-end proof via the hash function with quantization forced
    /// to f16 at call time rather than via env. We drive mode by
    /// inspecting the default path (env unset = F16) which is what
    /// prod will run with.
    #[test]
    fn content_hash_default_mode_absorbs_embedder_noise() {
        // Ensure env is unset so we exercise the F16 default.
        std::env::remove_var("HELIX_CONTENT_HASH_QUANTIZE");
        let base_vec = vec![0.5_f32, -0.25, 0.125];
        let noisy_vec: Vec<f32> = base_vec
            .iter()
            .map(|&f| f32::from_bits(f.to_bits() + 1))
            .collect();
        let a = ReplicatedPoint {
            id: 1,
            vectors: HashMap::from([("d".to_string(), base_vec)]),
            sparse_vectors: HashMap::new(),
            payload: HashMap::new(),
        };
        let b = ReplicatedPoint {
            id: 1,
            vectors: HashMap::from([("d".to_string(), noisy_vec)]),
            sparse_vectors: HashMap::new(),
            payload: HashMap::new(),
        };
        assert_eq!(
            compute_content_hash(&a),
            compute_content_hash(&b),
            "default (f16) quantization must absorb fp32 ULP noise end-to-end"
        );
    }

    #[test]
    fn content_hash_ignores_stamped_marker_key() {
        use crate::protocol::value::Value;
        // If the reserved marker key leaks into an incoming payload, the
        // hash must not change — otherwise a stored-hash round-trip would
        // never re-match.
        let dense = HashMap::from([("d".to_string(), vec![0.1_f32])]);
        let clean = ReplicatedPoint {
            id: 19,
            vectors: dense.clone(),
            sparse_vectors: HashMap::new(),
            payload: HashMap::from([("k".to_string(), Value::String("v".into()))]),
        };
        let clean_hash = compute_content_hash(&clean);

        let mut stamped_payload = HashMap::new();
        stamped_payload.insert("k".to_string(), Value::String("v".into()));
        stamped_payload.insert(CONTENT_HASH_KEY.to_string(), Value::U64(0xDEAD_BEEF));
        let stamped = ReplicatedPoint {
            id: 19,
            vectors: dense,
            sparse_vectors: HashMap::new(),
            payload: stamped_payload,
        };
        assert_eq!(
            clean_hash,
            compute_content_hash(&stamped),
            "stamped hash marker must be excluded from hashing"
        );
    }

    #[test]
    fn content_hash_payload_only_change_updates_node_payload() {
        use crate::protocol::value::Value;
        let temp_dir = TempDir::new().unwrap();
        let config = Config::new(8, 32, 64, 1);
        let collections =
            Arc::new(CollectionManager::new(temp_dir.path().join("data"), config.clone()).unwrap());

        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::CreateCollection {
                name: "repo".into(),
                vectors: HashMap::from([(
                    "dense".into(),
                    NamedVectorConfig {
                        size: 3,
                        distance:
                            crate::helix_engine::vector_core::named_vectors::DistanceMetric::Cosine,
                        spindle: crate::helix_engine::vector_core::spindle::SpindleConfig::default(
                        ),
                    },
                )]),
                sparse_vectors: HashMap::new(),
                hnsw_overrides: None,
            },
        )
        .unwrap();

        let storage = collections.get_collection("repo").unwrap();
        storage
            .create_payload_index("end_line", PayloadIndexSchema::Integer)
            .unwrap();
        storage
            .create_payload_index("metadata.file_hash", PayloadIndexSchema::Keyword)
            .unwrap();
        storage
            .create_payload_index("metadata.file_size_bytes", PayloadIndexSchema::Integer)
            .unwrap();

        let base_point = ReplicatedPoint {
            id: 31,
            vectors: HashMap::from([("dense".into(), vec![1.0, 0.0, 0.0])]),
            sparse_vectors: HashMap::new(),
            payload: HashMap::from([
                ("path".into(), Value::String("src/lib.rs".into())),
                ("end_line".into(), Value::U64(7)),
                (
                    "metadata".into(),
                    Value::Object(HashMap::from([
                        ("path".to_string(), Value::String("src/lib.rs".into())),
                        (
                            "file_hash".to_string(),
                            Value::String("old-file-hash".into()),
                        ),
                        ("file_size_bytes".to_string(), Value::U64(1024)),
                    ])),
                ),
            ]),
        };
        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::UpsertPoints {
                collection: "repo".into(),
                points: vec![base_point.clone()],
            },
        )
        .unwrap();

        {
            let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
            assert_eq!(
                storage
                    .get_nodes_by_payload_range(
                        &txn,
                        "end_line",
                        Some((7.0, true)),
                        Some((7.0, true)),
                    )
                    .unwrap(),
                vec![31],
                "test precondition: old top-level payload value must start indexed"
            );
            assert_eq!(
                storage
                    .get_nodes_by_payload_value(
                        &txn,
                        "metadata.file_hash",
                        &Value::String("old-file-hash".into()),
                    )
                    .unwrap(),
                vec![31],
                "test precondition: old nested metadata value must start indexed"
            );
            assert_eq!(
                storage
                    .get_nodes_by_payload_range(
                        &txn,
                        "metadata.file_size_bytes",
                        Some((1024.0, true)),
                        Some((1024.0, true)),
                    )
                    .unwrap(),
                vec![31],
                "test precondition: old nested numeric metadata value must start indexed"
            );
        }

        let first_hash = {
            let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
            let node = storage.get_node(&txn, &31).unwrap();
            content_hash_from_properties(&node.properties).unwrap()
        };

        let mut payload_only_point = base_point;
        payload_only_point
            .payload
            .insert("_pending_prune".into(), Value::Boolean(true));
        payload_only_point
            .payload
            .insert("_pending_prune_at".into(), Value::U64(123));
        payload_only_point
            .payload
            .insert("caller_point_id".into(), Value::String("caller-31".into()));
        payload_only_point
            .payload
            .insert("end_line".into(), Value::U64(42));
        payload_only_point.payload.insert(
            "metadata".into(),
            Value::Object(HashMap::from([
                ("path".to_string(), Value::String("src/lib.rs".into())),
                (
                    "file_hash".to_string(),
                    Value::String("new-file-hash".into()),
                ),
                ("file_size_bytes".to_string(), Value::U64(2048)),
                ("last_modified_at".to_string(), Value::U64(1_900_000_000)),
            ])),
        );
        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::UpsertPoints {
                collection: "repo".into(),
                points: vec![payload_only_point],
            },
        )
        .unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let node = storage.get_node(&txn, &31).unwrap();
        assert_eq!(
            content_hash_from_properties(&node.properties),
            Some(first_hash),
            "payload-only changes must preserve the index-content hash"
        );
        assert_eq!(
            node.properties.get("_pending_prune"),
            Some(&Value::Boolean(true)),
            "payload-only update must still rewrite the node payload"
        );
        assert_eq!(
            node.properties.get("caller_point_id"),
            Some(&Value::String("caller-31".into()))
        );
        assert_eq!(node.properties.get("end_line"), Some(&Value::U64(42)));
        match node.properties.get("metadata") {
            Some(Value::Object(metadata)) => {
                assert_eq!(
                    metadata.get("file_hash"),
                    Some(&Value::String("new-file-hash".into()))
                );
                assert_eq!(metadata.get("file_size_bytes"), Some(&Value::U64(2048)));
            }
            other => panic!("expected metadata object, got {other:?}"),
        }
        assert!(
            storage
                .get_nodes_by_payload_range(&txn, "end_line", Some((7.0, true)), Some((7.0, true)))
                .unwrap()
                .is_empty(),
            "payload-only updates must remove stale indexed top-level payload values"
        );
        assert_eq!(
            storage
                .get_nodes_by_payload_range(
                    &txn,
                    "end_line",
                    Some((42.0, true)),
                    Some((42.0, true))
                )
                .unwrap(),
            vec![31],
            "payload-only updates must index fresh top-level payload values"
        );
        assert!(
            storage
                .get_nodes_by_payload_value(
                    &txn,
                    "metadata.file_hash",
                    &Value::String("old-file-hash".into())
                )
                .unwrap()
                .is_empty(),
            "payload-only updates must remove stale indexed nested metadata values"
        );
        assert_eq!(
            storage
                .get_nodes_by_payload_value(
                    &txn,
                    "metadata.file_hash",
                    &Value::String("new-file-hash".into())
                )
                .unwrap(),
            vec![31],
            "payload-only updates must index fresh nested metadata values"
        );
        assert!(
            storage
                .get_nodes_by_payload_range(
                    &txn,
                    "metadata.file_size_bytes",
                    Some((1024.0, true)),
                    Some((1024.0, true))
                )
                .unwrap()
                .is_empty(),
            "payload-only updates must remove stale indexed nested numeric metadata values"
        );
        assert_eq!(
            storage
                .get_nodes_by_payload_range(
                    &txn,
                    "metadata.file_size_bytes",
                    Some((2048.0, true)),
                    Some((2048.0, true))
                )
                .unwrap(),
            vec![31],
            "payload-only updates must index fresh nested numeric metadata values"
        );
    }

    /// PRODUCTION-path C1 test: a tombstoned (superseded) copy is dropped by the
    /// real chunked-merge export, and the tombstone is cleared on retire — using
    /// the exact methods `run_collection_optimizer` calls
    /// (`is_tombstoned_for_merge`, `run_chunked_merge_publish`,
    /// `clear_tombstones_for_segment`), NOT the test-only `prepare_merge` path.
    #[test]
    fn chunked_merge_drops_tombstoned_copy_and_clears_on_retire() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _tombstones = EnvVarGuard::set("HELIX_REUPSERT_TOMBSTONES", "1");
        let spindle = crate::helix_engine::vector_core::spindle::SpindleConfig::default();
        let (_temp_dir, storage, hnsw_config) = storage_with_dense_vector(spindle);

        // Build two real Indexed segments, both holding id=42 (old + new copy),
        // plus a unique id in each so the merge has >1 surviving row.
        for (uniq, v) in [
            (1u128, [1.0f32, 0.0, 0.0, 0.0]),
            (2u128, [0.0, 1.0, 0.0, 0.0]),
        ] {
            storage
                .with_exclusive_write_txn(|txn| {
                    for (id, data) in [(42u128, v), (uniq, v)] {
                        let phys = storage
                            .named_vectors
                            .dense_insert(
                                storage.lmdb_env().unwrap(),
                                txn,
                                TEST_DENSE_VECTOR,
                                &data,
                                id,
                                HashMap::new(),
                                hnsw_config.clone(),
                                1,
                            )
                            .map_err(GraphError::from)?
                            .unwrap();
                        storage
                            .named_vectors
                            .build_dense_segment(txn, TEST_DENSE_VECTOR, &phys)
                            .map_err(GraphError::from)?;
                    }
                    Ok(())
                })
                .unwrap();
        }

        // Reconstruct tombstones from live data (id=42 in two segments → the
        // older copy is tombstoned). This is the same call collection-open makes.
        let dropped_targets: Vec<String> = storage
            .with_exclusive_write_txn(|txn| {
                storage
                    .named_vectors
                    .reconstruct_tombstones(txn)
                    .map_err(GraphError::from)?;
                Ok(())
            })
            .map(|_| {
                storage
                    .named_vectors
                    .list_dense_vector_spaces()
                    .get(TEST_DENSE_VECTOR)
                    .map(|s| {
                        s.segments
                            .iter()
                            .map(|seg| seg.physical_name.clone())
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .unwrap();
        assert!(dropped_targets.len() >= 2, "expected two indexed segments");

        // Find which segment holds the tombstoned (older) copy of id=42.
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let tombstoned_seg = dropped_targets
            .iter()
            .find(|phys| {
                storage
                    .named_vectors
                    .is_tombstoned_for_merge(&txn, phys, 42)
                    .unwrap_or(false)
            })
            .cloned()
            .expect("one segment must hold the tombstoned copy of id=42");
        drop(txn);

        // Run the PRODUCTION export-skip (mirrors run_collection_optimizer's
        // inline export at REPL ~2497) then the real chunked publish.
        let targets = dropped_targets.clone();
        let exported = storage
            .with_read_txn(|rtxn| {
                let cores = storage
                    .named_vectors
                    .cores_read()
                    .map_err(GraphError::from)?;
                let mut exported = Vec::new();
                let mut seen = std::collections::HashSet::new();
                for phys in &targets {
                    if let Some(core) = cores.get(phys) {
                        for (id, data, fields) in core
                            .export_level_zero(&core.backend.read_borrowed(&rtxn))
                            .map_err(GraphError::from)?
                        {
                            if storage
                                .named_vectors
                                .is_tombstoned_for_merge(rtxn, phys, id)
                                .map_err(GraphError::from)?
                            {
                                continue;
                            }
                            if seen.insert(id) {
                                exported.push((id, data, fields));
                            }
                        }
                    }
                }
                Ok(exported)
            })
            .unwrap();

        // The merged set holds id=42 exactly once (the non-tombstoned copy)
        // plus the two unique ids.
        let count_42 = exported.iter().filter(|(id, _, _)| *id == 42).count();
        assert_eq!(count_42, 1, "merged export keeps exactly one copy of id=42");
        assert!(exported.iter().any(|(id, _, _)| *id == 1));
        assert!(exported.iter().any(|(id, _, _)| *id == 2));

        let build_permit = acquire_build_permit().unwrap();
        let prepared_index = VectorCore::build_hnsw_in_memory_owned_with_permit(
            exported,
            &hnsw_config,
            &build_permit,
        )
        .unwrap();
        drop(build_permit);
        let merge = crate::helix_engine::vector_core::named_vectors::PreparedMerge {
            merge_targets: targets.clone(),
            prepared_index,
        };
        run_chunked_merge_publish(&storage, TEST_DENSE_VECTOR, &hnsw_config, merge).unwrap();

        // Production retire-clear (what the SEGMENT_REAPER finalize block calls).
        let cleared = storage
            .with_write_txn(|txn| {
                storage
                    .named_vectors
                    .clear_tombstones_for_segment(txn, &tombstoned_seg)
                    .map_err(GraphError::from)
            })
            .unwrap();
        assert!(cleared >= 1, "retire must clear the segment's tombstone");

        // The merged collection holds id=42 exactly once.
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let stats = storage
            .named_vectors
            .get_dense_space_stats(&txn, TEST_DENSE_VECTOR)
            .unwrap();
        assert_eq!(
            stats.vectors_count, 3,
            "superseded copy reclaimed: id 42,1,2 = 3 logical points"
        );
    }

    /// Deferred-repair delete-tombstones (issue #29 "fix 2"), PRODUCTION-path
    /// counterpart of `chunked_merge_drops_tombstoned_copy_and_clears_on_retire`:
    /// a delete-tombstoned id is dropped by the real chunked-merge export
    /// (the `core.is_delete_tombstoned(id)` check added to `run_index_job`'s
    /// export loop), and the segment's delete-tombstones are cleared on
    /// retire via `clear_tombstones_for_segment`'s generalized clear path.
    #[test]
    fn chunked_merge_drops_delete_tombstoned_copy_and_clears_on_retire() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _tombstones = EnvVarGuard::set("HELIX_LSM_DELETE_TOMBSTONES", "1");
        let spindle = crate::helix_engine::vector_core::spindle::SpindleConfig::default();
        let (_temp_dir, storage, hnsw_config) = storage_with_dense_vector(spindle);

        // Three real Indexed segments (threshold=1 seals a new one per
        // insert, mirroring `chunked_merge_drops_tombstoned_copy_and_clears_on_retire`
        // above): id=42, plus two unique ids, so the merge has other
        // surviving rows once id=42 is dropped.
        for (id, v) in [
            (42u128, [1.0f32, 0.0, 0.0, 0.0]),
            (1u128, [0.0, 1.0, 0.0, 0.0]),
            (2u128, [0.0, 0.0, 1.0, 0.0]),
        ] {
            storage
                .with_exclusive_write_txn(|txn| {
                    let phys = storage
                        .named_vectors
                        .dense_insert(
                            storage.lmdb_env().unwrap(),
                            txn,
                            TEST_DENSE_VECTOR,
                            &v,
                            id,
                            HashMap::new(),
                            hnsw_config.clone(),
                            1,
                        )
                        .map_err(GraphError::from)?
                        .unwrap();
                    storage
                        .named_vectors
                        .build_dense_segment(txn, TEST_DENSE_VECTOR, &phys)
                        .map_err(GraphError::from)?;
                    Ok(())
                })
                .unwrap();
        }
        let targets: Vec<String> = storage
            .named_vectors
            .list_dense_vector_spaces()
            .get(TEST_DENSE_VECTOR)
            .map(|s| {
                s.segments
                    .iter()
                    .map(|seg| seg.physical_name.clone())
                    .collect()
            })
            .unwrap_or_default();
        assert!(
            targets.len() >= 3,
            "expected at least three indexed segments"
        );

        // Find which segment actually holds id=42 (threshold=1 gives each id
        // its own segment, but rely on ownership, not insertion order).
        let tombstoned_seg = {
            let cores = storage.named_vectors.cores_read().unwrap();
            let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
            targets
                .iter()
                .find(|phys| {
                    cores
                        .get(*phys)
                        .map(|core| {
                            core.contains_id(&core.backend.read_borrowed(&txn), 42)
                                .unwrap_or(false)
                        })
                        .unwrap_or(false)
                })
                .cloned()
                .expect("one segment must hold id=42")
        };

        // Delete-tombstone id=42 on its owning segment through the real
        // backend-agnostic write path (same call `delete_vectors_batch_with_plan_be`
        // makes when the flag is on).
        let staged = storage
            .with_write_backend(|w| {
                let cores = storage
                    .named_vectors
                    .cores_read()
                    .map_err(GraphError::from)?;
                let core = cores.get(&tombstoned_seg).unwrap();
                core.tombstone_delete_batch_be(w, &[42])
                    .map_err(GraphError::from)
            })
            .unwrap();
        {
            let cores = storage.named_vectors.cores_read().unwrap();
            cores
                .get(&tombstoned_seg)
                .unwrap()
                .apply_delete_tombstones(&staged);
        }
        {
            let cores = storage.named_vectors.cores_read().unwrap();
            let core = cores.get(&tombstoned_seg).unwrap();
            assert!(core.is_delete_tombstoned(42));
            assert_eq!(core.deleted_count(), 1);
            let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
            assert!(
                core.contains_id(&core.backend.read_borrowed(&txn), 42)
                    .unwrap(),
                "row must still be physically present until merge"
            );
        }

        // Run the PRODUCTION export-skip (mirrors run_index_job's inline
        // export), now including the delete-tombstone check.
        let exported = storage
            .with_read_txn(|rtxn| {
                let cores = storage
                    .named_vectors
                    .cores_read()
                    .map_err(GraphError::from)?;
                let mut exported = Vec::new();
                let mut seen = std::collections::HashSet::new();
                for phys in &targets {
                    if let Some(core) = cores.get(phys) {
                        for (id, data, fields) in core
                            .export_level_zero(&core.backend.read_borrowed(&rtxn))
                            .map_err(GraphError::from)?
                        {
                            if core.is_delete_tombstoned(id) {
                                continue;
                            }
                            if seen.insert(id) {
                                exported.push((id, data, fields));
                            }
                        }
                    }
                }
                Ok(exported)
            })
            .unwrap();

        assert!(
            !exported.iter().any(|(id, _, _)| *id == 42),
            "delete-tombstoned id=42 must be dropped by the merge export"
        );
        assert!(exported.iter().any(|(id, _, _)| *id == 1));
        assert!(exported.iter().any(|(id, _, _)| *id == 2));

        let build_permit = acquire_build_permit().unwrap();
        let prepared_index = VectorCore::build_hnsw_in_memory_owned_with_permit(
            exported,
            &hnsw_config,
            &build_permit,
        )
        .unwrap();
        drop(build_permit);
        let merge = crate::helix_engine::vector_core::named_vectors::PreparedMerge {
            merge_targets: targets.clone(),
            prepared_index,
        };
        run_chunked_merge_publish(&storage, TEST_DENSE_VECTOR, &hnsw_config, merge).unwrap();

        // Production retire-clear (what the SEGMENT_REAPER finalize block
        // calls) must also clear the delete-tombstones now, via the
        // generalized `clear_tombstones_for_segment`.
        storage
            .with_write_txn(|txn| {
                storage
                    .named_vectors
                    .clear_tombstones_for_segment(txn, &tombstoned_seg)
                    .map_err(GraphError::from)
            })
            .unwrap();
        {
            let cores = storage.named_vectors.cores_read().unwrap();
            if let Some(core) = cores.get(&tombstoned_seg) {
                assert_eq!(
                    core.deleted_count(),
                    0,
                    "retire must clear the segment's delete-tombstones"
                );
            }
        }

        // The merged collection holds id=1 and id=2 but not the deleted id=42.
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let stats = storage
            .named_vectors
            .get_dense_space_stats(&txn, TEST_DENSE_VECTOR)
            .unwrap();
        assert_eq!(
            stats.vectors_count, 2,
            "deleted id reclaimed: only id 1 and 2 remain"
        );
    }

    /// REGRESSION: the background-optimizer read helpers must succeed on an
    /// LSM-backed collection.
    ///
    /// Before the fix, the optimizer cycle on an LSM collection errored with
    /// `GraphError::StorageError("LMDB env handle is unavailable on this
    /// backend")` because the read-path helpers it drives
    /// (`get_dense_space_stats_be`, `cleanup_empty_dense_segments_locate_be`,
    /// and the optimizer's pending-work probe) reached for the LMDB `Env`
    /// handle (`storage.lmdb_env()`), which is `None` on the LSM backend.
    ///
    /// This builds a REAL LSM-backed collection (the same production
    /// `apply_mutation` → `apply_upsert_points_lsm_chunk` →
    /// `dense_append_to_mutable_be` write path used by
    /// `optimizer_indexes_lsm_collection_end_to_end`), opens an `AnyRead`
    /// through `storage.backend.begin_read()`, and asserts each helper the
    /// optimizer relies on returns `Ok` rather than the LMDB-handle error:
    ///   * `get_dense_space_stats_be`            (run_index_job read probe)
    ///   * `cleanup_empty_dense_segments_locate_be`
    ///   * `gc_orphan_building_segments_deferred_be`   (NEW in the fix)
    ///   * `optimizer_has_pending_work`          (module-private optimizer gate)
    /// and that the collection is not left degraded afterward.
    #[test]
    fn optimizer_read_helpers_succeed_on_lsm_backend() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _backend = EnvVarGuard::set("HELIX_STORAGE_BACKEND", "lsm");
        let _in_mem = EnvVarGuard::set("HELIX_LSM_IN_MEMORY", "1");

        let temp_dir = TempDir::new().unwrap();
        let mut config = Config::new(8, 32, 64, 1);
        config.vector_config.flat_scan_threshold = Some(3);
        let collections =
            Arc::new(CollectionManager::new(temp_dir.path().join("data"), config.clone()).unwrap());

        // Create an LSM-backed collection with one dense vector space.
        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::CreateCollection {
                name: "repo".into(),
                vectors: HashMap::from([(
                    "dense".into(),
                    NamedVectorConfig {
                        size: 4,
                        distance:
                            crate::helix_engine::vector_core::named_vectors::DistanceMetric::Cosine,
                        spindle: crate::helix_engine::vector_core::spindle::SpindleConfig::default(
                        ),
                    },
                )]),
                sparse_vectors: HashMap::new(),
                hnsw_overrides: None,
            },
        )
        .unwrap();

        let storage = collections.get_collection("repo").unwrap();
        assert_eq!(
            storage.backend.kind(),
            BackendKind::Lsm,
            "test harness must run on the LSM backend"
        );

        // Land a few vectors through the production LSM upsert path so the
        // "dense" space actually exists with segments to stat/scan.
        const N: u128 = 4;
        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::UpsertPoints {
                collection: "repo".into(),
                points: (1..=N)
                    .map(|i| {
                        let mut v = [0.0f32; 4];
                        v[(i as usize - 1) % 4] = 1.0;
                        ReplicatedPoint {
                            id: i,
                            vectors: HashMap::from([("dense".into(), v.to_vec())]),
                            sparse_vectors: HashMap::new(),
                            payload: HashMap::new(),
                        }
                    })
                    .collect(),
            },
        )
        .unwrap();

        // The actual pre-fix failure points: the optimizer's read-path helpers,
        // driven against a real AnyRead view of the LSM backend. Pre-fix these
        // reached for the (absent) LMDB Env and returned
        // "LMDB env handle is unavailable on this backend".
        let r = storage.backend.begin_read().unwrap();

        let stats = storage.named_vectors.get_dense_space_stats_be(&r, "dense");
        assert!(
            stats.is_ok(),
            "get_dense_space_stats_be must succeed on LSM, got: {:?}",
            stats.err()
        );
        assert_eq!(
            stats.unwrap().vectors_count,
            N as u64,
            "all flat vectors must be visible to the optimizer read probe on LSM"
        );

        let empty = storage
            .named_vectors
            .cleanup_empty_dense_segments_locate_be(&r, "dense");
        assert!(
            empty.is_ok(),
            "cleanup_empty_dense_segments_locate_be must succeed on LSM, got: {:?}",
            empty.err()
        );

        // The helper added by the fix (named_vectors.rs).
        let orphans = storage
            .named_vectors
            .gc_orphan_building_segments_deferred_be(&r, "dense");
        assert!(
            orphans.is_ok(),
            "gc_orphan_building_segments_deferred_be must succeed on LSM, got: {:?}",
            orphans.err()
        );

        drop(r);

        // The module-private optimizer gate must run its LSM branch without
        // touching the LMDB Env. It returns a bool (errors are swallowed), so
        // the real assertion is that the collection is NOT degraded by the
        // attempt below.
        let _pending = optimizer_has_pending_work(
            &storage,
            &["dense".to_string()],
            config.vector_flat_scan_threshold(),
            true,
        );

        // None of the read helpers above may have flipped the collection into
        // the degraded state via the LMDB-handle error path.
        assert!(
            storage.degraded_collection_error().is_none(),
            "optimizer read helpers degraded the LSM collection: {:?}",
            storage.degraded_collection_error()
        );
    }
}
