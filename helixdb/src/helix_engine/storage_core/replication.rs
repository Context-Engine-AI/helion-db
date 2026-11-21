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
use crate::helix_engine::storage_core::backend::{
    BackendKind, Namespace, StorageBackend, StorageBackendConfig,
};
use crate::helix_engine::storage_core::collection_manager::{
    is_one_path_segment, maintenance_process_resource_admission, spawn_idle_gc_sweeper,
    CollectionManager, MaintenanceAdmission,
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
#[cfg(test)]
use crate::helix_engine::vector_core::named_vectors::DistanceMetric;
use crate::helix_engine::vector_core::named_vectors::{
    DenseDeletePlan, DenseLayoutChange, DenseMergeCandidateLimits, DenseSegmentDebt,
    DenseVectorSpaceMetadata, NamedVectorConfig, NamedVectorManager, PreparedTailSeal,
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
/// Replica-shed 503s from one reader inside this window before the gateway
/// ejects the whole reader; fewer sheds only spill the shed request.
const GATEWAY_READER_SHED_WINDOW: Duration = Duration::from_secs(10);
const GATEWAY_READER_SHED_EJECT_COUNT: u32 = 8;
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

/// How long a collection goes without a point mutation before its
/// sub-threshold dense tails are sealed (`HELIX_INDEX_TAIL_QUIET_MS`, default
/// 30000). A tail sealed at every short pause of an index run is a segment
/// built, and later merged, per pause.
fn index_tail_quiet_ms() -> u64 {
    env_u64(
        "HELIX_INDEX_TAIL_QUIET_MS",
        if cfg!(test) { 100 } else { 30_000 },
    )
}

/// Default of `index_tail_seal_min_vectors`. A tail is sealed by the job of
/// the upsert that took it there, with the rows that came in until then (up
/// to 252 more, measured), and a sealed segment is scored exactly up to
/// 2,048 rows (`SMALL_SEGMENT_EXACT_MAX_ROWS`): 256 under that.
const INDEX_TAIL_SEAL_MIN_VECTORS: usize = 1_792;

/// Unindexed vectors at which a dense tail is sealed without waiting for
/// quiet (`HELIX_INDEX_TAIL_SEAL_MIN_VECTORS`, default 1792, never above the
/// flat-scan threshold; 0 = a sub-threshold tail is sealed by quiet only).
/// `HELIX_INDEX_TAIL_QUIET_MS=1000` with 0 here is the rule before the debounce.
fn index_tail_seal_min_vectors(flat_scan_threshold: usize) -> usize {
    env_usize_or(
        "HELIX_INDEX_TAIL_SEAL_MIN_VECTORS",
        if cfg!(test) {
            0
        } else {
            INDEX_TAIL_SEAL_MIN_VECTORS
        },
        0,
    )
    .min(flat_scan_threshold)
}

/// An index job waits out a quiet period up to this long, as it did before
/// the debounce. A longer one would hold an index worker for the wait: the
/// job ends instead, and the backlog sweeper submits the collection once it
/// is quiet.
const INDEX_TAIL_QUIET_WAIT_IN_JOB_MAX_MS: u64 = 2_000;

fn epoch_millis_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// No point mutation is being applied and the last one ended at least
/// `quiet_ms` ago, or none was applied since the collection was opened.
fn point_mutations_quiet_for(storage: &HelixGraphStorage, quiet_ms: u64) -> bool {
    storage.point_mutations_in_flight.load(Ordering::Acquire) == 0
        && epoch_millis_now().saturating_sub(storage.last_upsert_at.load(Ordering::Acquire))
            >= quiet_ms
}

/// `index_tail_quiet_ms` for this collection. One with a quantized dense
/// space keeps the second of quiet it had before the debounce, and no seal
/// minimum: its tail has no sidecar, so every tail row a reader replica
/// scores is a point read, and a longer tail is a slower search. One
/// difference remains: a mutation being applied is not quiet, where a job
/// that ran over a second into a long upsert used to seal the tail under it.
fn collection_tail_quiet_ms(storage: &HelixGraphStorage) -> u64 {
    let quiet_ms = index_tail_quiet_ms();
    if storage.named_vectors.has_spindle_dense_space() {
        quiet_ms.min(1_000)
    } else {
        quiet_ms
    }
}

/// `index_tail_seal_min_vectors` for this collection: 0 with a quantized
/// dense space (see `collection_tail_quiet_ms`) and on LMDB, where the rows
/// counted are every unindexed row and not the tail's alone.
fn collection_tail_seal_min(storage: &HelixGraphStorage, flat_scan_threshold: usize) -> usize {
    if storage.backend.kind() != BackendKind::Lsm || storage.named_vectors.has_spindle_dense_space()
    {
        0
    } else {
        index_tail_seal_min_vectors(flat_scan_threshold)
    }
}

/// Whether a dense tail of `tail_rows` vectors below the flat-scan threshold
/// is sealed now: the collection is quiet, or the tail reached the seal
/// minimum.
fn subthreshold_tail_is_sealable(tail_rows: u64, quiesced: bool, seal_min: usize) -> bool {
    tail_rows > 0 && (quiesced || (seal_min > 0 && tail_rows as usize >= seal_min))
}

/// Waits until every point mutation that was being applied when this was
/// called has ended; false when `give_up` says to stop waiting. A tail sealed
/// under a mutation still takes the rows that mutation staged before the
/// seal, and they commit after it: a read begun after this wait sees them.
/// A collection's mutations run one at a time under its point-mutation gate,
/// so a later one having started means the earlier one is over.
fn point_mutation_in_flight_has_ended(
    storage: &HelixGraphStorage,
    give_up: impl Fn() -> bool,
) -> bool {
    let started = storage.point_mutations.load(Ordering::Acquire);
    while storage.point_mutations_in_flight.load(Ordering::Acquire) > 0
        && storage.point_mutations.load(Ordering::Acquire) == started
    {
        if give_up() {
            return false;
        }
        thread::sleep(Duration::from_millis(5));
    }
    true
}

/// Counts a point mutation as being applied until it is dropped, and stamps
/// the time the tail-seal quiet period is measured from when it ends.
struct PointMutationInFlight<'a>(&'a HelixGraphStorage);

impl<'a> PointMutationInFlight<'a> {
    fn begin(storage: &'a HelixGraphStorage) -> Self {
        storage
            .point_mutations_in_flight
            .fetch_add(1, Ordering::AcqRel);
        Self(storage)
    }
}

impl Drop for PointMutationInFlight<'_> {
    fn drop(&mut self) {
        self.0
            .last_upsert_at
            .store(epoch_millis_now(), Ordering::Release);
        self.0
            .point_mutations_in_flight
            .fetch_sub(1, Ordering::AcqRel);
    }
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

/// What a panic a worker caught said, for the one line it writes of it.
fn panic_text(panic: &(dyn std::any::Any + Send)) -> &str {
    if let Some(text) = panic.downcast_ref::<&'static str>() {
        *text
    } else if let Some(text) = panic.downcast_ref::<String>() {
        text.as_str()
    } else {
        "a panic that carried no text"
    }
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
                    // Uncaught, a panic ended this thread with the gauge
                    // raised and the collection still in `active`: every
                    // later submit for it was parked where `next_job` does
                    // not look, and every collection had a worker fewer.
                    let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        run_index_job(job)
                    }));
                    if let Err(panic) = ran {
                        metrics::counter!("helix_index_executor_panics_total").increment(1);
                        tracing::error!(
                            collection = %key,
                            panic = panic_text(&*panic),
                            "index job panicked; its worker goes on and the collection's next job runs"
                        );
                    }
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

/// True while the index executor holds a queued or running job for
/// `collection`. A queued job has only a `Weak` to the storage, so nothing
/// else keeps the collection loaded until the job runs.
pub(crate) fn index_job_queued_or_active(collection: &str) -> bool {
    let guard = INDEX_EXECUTOR
        .state
        .0
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    guard.queued_jobs.contains_key(collection) || guard.active.contains(collection)
}

/// Every collection the backlog sweeper loop handed to its submit.
#[cfg(test)]
static BACKLOG_SWEEPER_SUBMITS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Test-only: an index job of a collection named here, or a reaper turn of a
/// segment named here, panics as it starts. Each entry is one panic.
#[cfg(test)]
static WORKER_PANICS_FOR_TEST: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Test-only: panics if `name` has a panic left in [`WORKER_PANICS_FOR_TEST`],
/// and takes that one out.
#[cfg(test)]
fn panic_once_for_test(name: &str) {
    let due = {
        let mut names = WORKER_PANICS_FOR_TEST
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let listed = names.iter().position(|listed| listed == name);
        listed.map(|at| names.swap_remove(at)).is_some()
    };
    if due {
        panic!("{name} panics for the test");
    }
}

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

/// How far down its list a sweep looks for candidates past their cooldown.
/// Admitted, to the end: a cooling candidate takes none of its places. Under
/// pressure, at as many candidates as it has places and no further, as it did
/// before that rule. Going on past the cooling ones there submits a job at
/// every sweep, each for another collection, where the first candidate's
/// cooldown let one through an hour: on a pod that is short of memory, and
/// the index job asks no admission of its own.
fn optimizer_backlog_candidates_looked_at(
    warm_per_sweep: usize,
    admission: MaintenanceAdmission,
) -> usize {
    if admission.is_admitted() {
        usize::MAX
    } else {
        warm_per_sweep
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
    /// The segment debt alone makes it a candidate.
    debt_work: bool,
    /// `last_upsert_at` of a collection that has gone quiet since: the index
    /// job its last upsert submitted did not wait the quiet period out.
    quiet_since: Option<u64>,
}

impl OptimizerDebtCandidate {
    fn is_gate_blocked(&self) -> bool {
        self.debt.gate_debt > 0
    }
}

/// `debt.indexed_segments` sums every dense space, so on its own it is over
/// `min_segments` for a fully merged multi-space collection. Submit a backlog
/// only when a space is over the merge target (`space_over_merge_target`, see
/// `has_dense_space_over_merge_target`), which is what the index job merges on.
/// A Building segment the job can build (`buildable_segment`, see
/// `has_building_segment_with_core`) is submitted too: this is its retry when
/// a job ended on an error. A job that starts with merge debt takes a
/// merge-only turn first, so the build can take more than one turn.
/// shortcut: the deleted-ratio vacuum merge (HELIX_LSM_DELETE_TOMBSTONES, off
/// by default) needs a row count and is not swept; add it here if enabled.
fn optimizer_debt_has_submit_work(
    debt: DenseSegmentDebt,
    min_segments: usize,
    space_over_merge_target: bool,
    buildable_segment: bool,
) -> bool {
    (debt.indexed_segments > min_segments && space_over_merge_target)
        || buildable_segment
        || (debt.gate_debt > 0
            && debt.active_segments >= NamedVectorManager::segment_creation_gate_cap())
}

fn optimizer_debt_can_expand_build_budget(debt: DenseSegmentDebt) -> bool {
    debt.gate_debt > 0 && debt.active_segments >= NamedVectorManager::segment_creation_gate_cap()
}

/// True when this node is configured as an LSM read-only replica.
/// Reader replicas serve reads from the writer's committed S3 state and reject
/// all writes; the HNSW build/merge optimizer is a WRITE path, so running it on a
/// reader just fails every cycle with "read-only LSM reader replica" (error spam
/// + wasted CPU). The optimizer must run on the writer only.
fn lsm_config_is_reader(config: &Config) -> bool {
    config.storage_backend.is_reader()
}

fn spawn_optimizer_backlog_sweeper(collections: Arc<CollectionManager>, config: Config) {
    if lsm_config_is_reader(&config) {
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
            // Per collection, the `last_upsert_at` whose quiet period was
            // found to have no tail left to seal.
            let mut quiet_settled: HashMap<String, u64> = HashMap::new();
            loop {
                thread::sleep(sweep);
                let Some(collections) = collections.upgrade() else {
                    return;
                };
                let started = Instant::now();
                // Here because this thread wakes anyway: it reads counters.
                crate::helix_engine::storage_core::backend_lsm::export_shared_block_cache_usage();
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
                            // The last sidecar of a burst of changes waits
                            // here for the put interval to pass.
                            storage.upload_held_metadata_sidecar();
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
                                "helix_collection_dense_dead_names",
                                "collection" => name.clone()
                            )
                            .set(storage.named_vectors.dead_segment_count() as f64);
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
                            let debt_work = optimizer_debt_has_submit_work(
                                debt,
                                min_segments,
                                storage.named_vectors.has_dense_space_over_merge_target(),
                                storage.named_vectors.has_building_segment_with_core(),
                            );
                            let last_mutation = storage.last_upsert_at.load(Ordering::Acquire);
                            let quiet_ms = collection_tail_quiet_ms(&storage);
                            // Indexing threshold 0 asks for a flat tail: an
                            // upsert submits no job for it, and neither does quiet.
                            let quiet_since = (quiet_ms > INDEX_TAIL_QUIET_WAIT_IN_JOB_MAX_MS
                                && last_mutation != 0
                                && quiet_settled.get(&name) != Some(&last_mutation)
                                && storage
                                    .named_vectors
                                    .effective_indexing_threshold(config.vector_flat_scan_threshold())
                                    > 0
                                && point_mutations_quiet_for(&storage, quiet_ms))
                            .then_some(last_mutation);
                            if (debt_work || quiet_since.is_some())
                                && !storage.optimizer_running.load(Ordering::Acquire)
                                && !crate::helix_engine::storage_core::collection_manager::auto_compact_in_progress(&name)
                            {
                                Some(OptimizerDebtCandidate {
                                    name,
                                    debt,
                                    priority_score,
                                    debt_work,
                                    quiet_since,
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
                // A candidate in its cooldown takes none of the sweep's
                // places: as many of those as a sweep takes, with debt that
                // stands, would keep every collection after them from its
                // submit, a quiet one from the one that seals its tail.
                let mut past_cooldown = 0usize;
                for candidate in
                    candidates
                        .into_iter()
                        .take(optimizer_backlog_candidates_looked_at(
                            warm_per_sweep,
                            process_admission,
                        ))
                {
                    if past_cooldown >= warm_per_sweep {
                        break;
                    }
                    let gate_blocked = candidate.is_gate_blocked();
                    let priority_score = candidate.priority_score;
                    let name = candidate.name;
                    let debt = candidate.debt;
                    let debt_work = candidate.debt_work;
                    let quiet_since = candidate.quiet_since;
                    let effective_cooldown = if gate_blocked || quiet_since.is_some() {
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
                    past_cooldown += 1;
                    #[cfg(test)]
                    BACKLOG_SWEEPER_SUBMITS
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(name.clone());
                    match submit_loaded_collection_optimizer(&collections, &config, &name) {
                        Ok(None) => {}
                        Ok(Some(spaces)) => {
                            // One submit answers a quiet period, whether it
                            // found a tail to seal or not: the next mutation
                            // starts another. Work a job cannot finish must
                            // not bring a job back every sweep.
                            if let Some(last_mutation) = quiet_since {
                                quiet_settled.insert(name.clone(), last_mutation);
                            }
                            if spaces == 0 && !debt_work {
                                continue;
                            }
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
    if lsm_config_is_reader(config) {
        metrics::counter!("helix_index_executor_submit_skipped_total", "reason" => "reader_replica")
            .increment(1);
        return Ok(0);
    }
    let resolved_collection = collections.resolve_alias(collection);
    let storage = collections.get_collection(&resolved_collection)?;
    submit_optimizer_for_storage(storage, config, resolved_collection, started)
}

/// Backlog-sweeper submit. The sweeper picks its candidates from the loaded
/// set, and one may be evicted before its turn: `get_collection` would reopen
/// it and reset its idle timer. `Ok(None)` when there is nothing to submit to,
/// counted under its own sweeper outcome: `not_loaded` or `fenced_writer`.
fn submit_loaded_collection_optimizer(
    collections: &Arc<CollectionManager>,
    config: &Config,
    collection: &str,
) -> Result<Option<usize>, GraphError> {
    let started = Instant::now();
    let Some(storage) = collections.peek_loaded_collection(collection)? else {
        metrics::counter!("helix_optimizer_backlog_sweeper_total", "outcome" => "not_loaded")
            .increment(1);
        return Ok(None);
    };
    // `get_collection` refuses a fenced writer. It stays loaded and quarantined
    // until the process restarts, and the optimizer cannot write through it.
    if matches!(
        storage.unhealthy_lsm_writer_close_reason(),
        Some(slatedb::CloseReason::Fenced)
    ) {
        metrics::counter!("helix_optimizer_backlog_sweeper_total", "outcome" => "fenced_writer")
            .increment(1);
        return Ok(None);
    }
    submit_optimizer_for_storage(storage, config, collection.to_string(), started).map(Some)
}

fn submit_optimizer_for_storage(
    storage: Arc<HelixGraphStorage>,
    config: &Config,
    resolved_collection: String,
    started: Instant,
) -> Result<usize, GraphError> {
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
    let hnsw_config = collection_hnsw_config(&storage, config)?;

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

fn collection_hnsw_config(
    storage: &HelixGraphStorage,
    config: &Config,
) -> Result<HNSWConfig, GraphError> {
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
    Ok(HNSWConfig::with_overrides(
        config.vector_config.m,
        config.vector_config.ef_construction,
        config.vector_config.ef_search,
        overrides.as_ref(),
    ))
}

// ─── Regraph ────────────────────────────────────────────────────────────────
//
// Rebuilds dense graphs built before diverse neighbors (2d2891c7) from their
// stored vectors: each Indexed HNSW segment keeping at least
// `HNSW_DIVERSE_MIN_POINTS` live rows is rebuilt alone into a new segment
// through the merge publish (`write_merge_target`, `swap_in_merge_target`),
// inside the collection's index job.
// `POST /collections/{name}/regraph` queues it; the GET reports progress.

/// `HELIX_REGRAPH_ENABLED=0` refuses regraph requests.
fn regraph_enabled() -> bool {
    !matches!(
        std::env::var("HELIX_REGRAPH_ENABLED")
            .map(|value| value.trim().to_ascii_lowercase())
            .as_deref(),
        Ok("0" | "false" | "no" | "off")
    )
}

/// `HELIX_REGRAPH_SKIP_CURRENT=0` has a requested regraph rebuild the
/// segments the current build made too.
fn regraph_skips_current() -> bool {
    !matches!(
        std::env::var("HELIX_REGRAPH_SKIP_CURRENT")
            .map(|value| value.trim().to_ascii_lowercase())
            .as_deref(),
        Ok("0" | "false" | "no" | "off")
    )
}

/// Progress of a collection's regraph: `queued`, `running`, `done`, `busy`
/// (writes landed during a rebuild, retry later) or `failed`.
#[derive(Debug, Clone, Default, Serialize)]
pub struct RegraphStatus {
    pub state: &'static str,
    pub segments_rebuilt: usize,
    /// Left alone: their graph was made by the current build.
    pub segments_current: usize,
    /// Left alone: no graph, or fewer rows stored than a search walks.
    pub segments_small: usize,
    /// Taken out with nothing in their place: a rebuild would have kept no
    /// row of them.
    pub segments_retired: usize,
    pub segments_busy: usize,
    pub points_rebuilt: usize,
    pub superseded_dropped: u64,
    /// Rows left out because their point is gone.
    pub nodeless_dropped: u64,
    /// Milliseconds summed over the run's segments: export and graph build,
    /// target write (no write gate), and the swap the write gate is held for.
    pub build_ms: u64,
    pub write_ms: u64,
    pub swap_ms: u64,
    pub error: Option<String>,
}

#[derive(Debug)]
pub enum RegraphRefusal {
    Disabled,
    ReaderReplica,
    NotLsm,
    /// The link pass alone: a named vector of the collection is quantized.
    Quantized,
    InProgress(&'static str),
    Storage(GraphError),
}

/// Status, point-mutation gate and dense spaces per collection, in memory
/// only: a restart forgets `done`, which the driver's state file covers.
type RegraphRequest = (RegraphStatus, Arc<Mutex<()>>, Vec<String>);
static REGRAPH_REQUESTS: LazyLock<Mutex<HashMap<String, RegraphRequest>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn set_regraph_status(collection: &str, status: RegraphStatus) {
    let mut requests = REGRAPH_REQUESTS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(entry) = requests.get_mut(collection) {
        entry.0 = status;
    }
}

/// Fails a regraph its index job left queued or running (early return,
/// error or panic), so the next request can queue it again.
struct RegraphJobGuard(String);

impl Drop for RegraphJobGuard {
    fn drop(&mut self) {
        let mut requests = REGRAPH_REQUESTS.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((status, ..)) = requests.get_mut(&self.0) {
            if matches!(status.state, "queued" | "running") {
                status.state = "failed";
                status.error = Some("index job ended before the regraph finished".into());
            }
        }
    }
}

pub fn regraph_status(collections: &CollectionManager, collection: &str) -> Option<RegraphStatus> {
    let resolved = collections.resolve_alias(collection);
    REGRAPH_REQUESTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&resolved)
        .map(|(status, ..)| status.clone())
}

/// Queue a regraph of `collection` on the writer. A collection already done
/// since this process started returns its status without new work.
pub fn submit_collection_regraph(
    collections: &Arc<CollectionManager>,
    config: &Config,
    collection: &str,
) -> Result<RegraphStatus, RegraphRefusal> {
    if !regraph_enabled() {
        return Err(RegraphRefusal::Disabled);
    }
    if lsm_config_is_reader(config) {
        return Err(RegraphRefusal::ReaderReplica);
    }
    let resolved = collections.resolve_alias(collection);
    let storage = collections
        .get_collection(&resolved)
        .map_err(RegraphRefusal::Storage)?;
    if storage.backend.is_reader_replica() {
        return Err(RegraphRefusal::ReaderReplica);
    }
    if storage.backend.kind() != BackendKind::Lsm {
        return Err(RegraphRefusal::NotLsm);
    }
    storage
        .ensure_not_degraded()
        .map_err(RegraphRefusal::Storage)?;
    let mut requests = REGRAPH_REQUESTS.lock().unwrap_or_else(|e| e.into_inner());
    match requests.get(&resolved).map(|(status, ..)| status.state) {
        Some("done") => return Ok(requests[&resolved].0.clone()),
        Some(state @ ("queued" | "running")) => return Err(RegraphRefusal::InProgress(state)),
        _ => {}
    }
    if storage.optimizer_running.load(Ordering::Acquire) {
        return Err(RegraphRefusal::InProgress("optimizing"));
    }
    let spaces_to_check: Vec<String> = storage
        .named_vectors
        .list_dense_vector_spaces()
        .into_keys()
        .collect();
    let job = IndexJob {
        collection: resolved.clone(),
        storage: Arc::downgrade(&storage),
        spaces_to_check: spaces_to_check.clone(),
        hnsw_config: collection_hnsw_config(&storage, config).map_err(RegraphRefusal::Storage)?,
        flat_scan_threshold: storage
            .named_vectors
            .effective_indexing_threshold(config.vector_flat_scan_threshold()),
    };
    let status = RegraphStatus {
        state: "queued",
        ..RegraphStatus::default()
    };
    requests.insert(
        resolved.clone(),
        (
            status.clone(),
            collections.point_mutation_gate(&resolved),
            spaces_to_check,
        ),
    );
    drop(requests);
    if let Err(e) = INDEX_EXECUTOR.submit(job) {
        REGRAPH_REQUESTS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&resolved);
        return Err(RegraphRefusal::Storage(GraphError::New(format!(
            "failed to submit regraph job: {e}"
        ))));
    }
    Ok(status)
}

/// Runs a queued regraph for `collection` (index job tail) over the spaces
/// it was requested for, recording the outcome. No-op when none is queued.
fn run_queued_regraph(
    storage: &Arc<HelixGraphStorage>,
    collection: &str,
    hnsw_config: &HNSWConfig,
) -> Result<(), GraphError> {
    let (gate, spaces) = {
        let mut requests = REGRAPH_REQUESTS.lock().unwrap_or_else(|e| e.into_inner());
        match requests.get_mut(collection) {
            Some((status, gate, spaces)) if status.state == "queued" => {
                status.state = "running";
                (Arc::clone(gate), spaces.clone())
            }
            _ => return Ok(()),
        }
    };
    let result = regraph_segments(
        storage,
        collection,
        &spaces,
        hnsw_config,
        &gate,
        regraph_skips_current(),
    );
    set_regraph_status(
        collection,
        match &result {
            Ok(status) => status.clone(),
            Err(e) => RegraphStatus {
                state: "failed",
                error: Some(e.to_string()),
                ..RegraphStatus::default()
            },
        },
    );
    result.map(|_| ())
}

/// How many unreached rows one call of the link pass looks at, a walk and at
/// most a link each: what bounds the walks it makes, the lists it writes and
/// the time it holds the write gate.
pub const LINK_PASS_ROWS_A_CALL: usize = 256;

/// What one call of the link pass over a collection did: `done`, or `busy`
/// when the collection was written to or an index job started while it
/// planned (nothing is written then; ask again).
#[derive(Debug, Clone, Default, Serialize)]
pub struct LinkPassStatus {
    pub state: &'static str,
    /// The segment this call read, none when the pass has none left to do;
    /// the served rows of it no stored link led to; of the rows this call
    /// looked at, those it wrote a link for and those no row had room for,
    /// near them or among the rows the pass has linked below them (a list
    /// has room up to its bound, a quarter past what a build fills it to),
    /// which stay unreached: the pass comes back to them only when it comes
    /// back to the segment, after a hard delete there. `without_room` also
    /// counts the rows left because eight rows with their own vector are
    /// arrived at already.
    pub segment: Option<String>,
    pub unreached: usize,
    pub linked: usize,
    pub without_room: usize,
    /// Segments the pass has still to do after this call, the one it read
    /// counted where rows are left to look at in it. Call again until it is
    /// 0.
    pub remaining: usize,
    /// When a reader replica walks the new links: it keeps the lists it has
    /// cached until then. Its count of unreached rows is read from the store
    /// and falls with the writer's all the same, so that count does not say
    /// that its searches arrive at them.
    pub readers_see_after: &'static str,
    /// Milliseconds reading and planning (no write gate), and writing (the
    /// write gate is held).
    pub plan_ms: u64,
    pub write_ms: u64,
}

/// One call of the link pass over `collection`'s graphs as they are stored,
/// on the writer. An Indexed segment a search walks, and the pass has not
/// marked, gets a link to each served row no stored link leads to, at the
/// end of the list of a near row with room (`plan_stored_links_be`). It
/// adds links and removes none, writes one list a linked row, and marks the
/// segment so that later calls read one key for it.
///
/// A list has room up to a quarter past the `m_max_0` ids a build fills it
/// to (`link_pass_list_bound`): 40 ids at an `m` of 16. The graphs built
/// before the builds linked their own unreached rows have every near list
/// full, and the pass that linked only under `m_max_0` left their rows as
/// they were: a segment it marked is due again.
///
/// Where a row's eight nearest rows are all at the bound, its link comes
/// from a row the pass has linked below one of them, the nearest first, one
/// of this call's links among them. The pass before this one left such a
/// row, as it left most of a group of rows with one vector, which share
/// their eight nearest. Its mark does not say whether it left any, so a
/// segment it marked is due again too, and is read once more.
///
/// A call takes one segment: the first that is due. It reads that
/// segment's ids and lists and no other segment's, and looks at no more
/// than `max_rows` of its unreached rows: a walk each, a link where a near
/// row has room, so it holds the gate for at most that many lists. With the
/// links it writes the row it has come to, and the next call goes on after
/// that row; the segment is marked when a call comes to the end. A row is
/// looked at once: one that found no room is counted (`without_room`) and
/// left. `remaining` is the number of segments that are due, which the
/// marks give with no list read. `HELIX_REGRAPH_ENABLED=0` refuses it, as it
/// refuses a regraph.
///
/// A marked segment is due again once a hard delete has taken a row of its
/// graph: a walk crosses no list a delete leaves, so what only that row led
/// to is unreached again, and a row the pass linked from can be such a row.
/// The mark keeps the count of deleted rows the pass began at, and a
/// segment whose count has grown is read again, all of it, as one without a
/// mark is. A segment that is not due costs one more point read a call, of
/// that count.
///
/// A collection with a quantized named vector is refused whole, its other
/// spaces with it: such a space scores its rows from the codes in the files
/// its readers build (`.hspn`, `.hvtq`), and the pass has been run over
/// none.
///
/// The plan holds the collection's cores for reading while it reads the
/// segment and walks, as the count route and a regraph's export do for a
/// segment: a write that needs them for writing (a seal, a merge's swap)
/// waits that long, and searches behind it. `plan_ms` says how long.
///
/// The reading and the walks run without the point-mutation gate. The gate
/// is held to check that nothing was written meanwhile and to write: a
/// collection that was written to answers `busy` with nothing changed. A
/// merge that retires a segment meanwhile is seen in the layout, and its
/// segment is left out.
///
/// This is the one writer of a list in a segment a search already walks:
/// every other list is written once, before its segment's entry point is.
/// The writer drops the lists it changed from its cache. A reader replica
/// has no such step: one that has a changed list cached walks the older,
/// shorter one until it evicts it or restarts. Its walks do not follow the
/// new link until then, though its count of unreached rows, which is read
/// from the store, falls with the writer's; and nothing it returns is
/// wrong, since the older list is a part of the newer. So roll the readers
/// after a run over the stock of old segments. A segment built since links
/// its rows at the build, so the pass finds little to do there: a row the
/// build found no room for under `m_max_0`, or one a delete has cut off
/// since.
// shortcut: the optimizer flag is read, not held. An index job that starts
// after the check and retires a segment before the commit leaves this
// pass's few keys in a namespace the reaper may already have cleared; they
// are never read. Run it as an index job's tail if that is ever seen.
pub fn link_stored_unreached(
    collections: &Arc<CollectionManager>,
    config: &Config,
    collection: &str,
    max_rows: usize,
) -> Result<LinkPassStatus, RegraphRefusal> {
    if !regraph_enabled() {
        return Err(RegraphRefusal::Disabled);
    }
    if lsm_config_is_reader(config) {
        return Err(RegraphRefusal::ReaderReplica);
    }
    let resolved = collections.resolve_alias(collection);
    let storage = collections
        .get_collection(&resolved)
        .map_err(RegraphRefusal::Storage)?;
    if storage.backend.is_reader_replica() {
        return Err(RegraphRefusal::ReaderReplica);
    }
    if storage.backend.kind() != BackendKind::Lsm {
        return Err(RegraphRefusal::NotLsm);
    }
    // A poisoned lock on the configs is an error here and not "none".
    if storage
        .named_vectors
        .quantizes_a_dense_space()
        .map_err(|e| RegraphRefusal::Storage(GraphError::from(e)))?
    {
        return Err(RegraphRefusal::Quantized);
    }
    storage
        .ensure_not_degraded()
        .map_err(RegraphRefusal::Storage)?;
    if storage.optimizer_running.load(Ordering::Acquire) {
        return Err(RegraphRefusal::InProgress("optimizing"));
    }
    let gate = collections.point_mutation_gate(&resolved);
    let started = Instant::now();
    // Snapshot and write count are taken together, between batches.
    let (mutations, r) = {
        let _held = lock_point_mutation_gate(&gate, &resolved, "link_pass");
        let r = storage
            .backend
            .begin_read()
            .map_err(|e| RegraphRefusal::Storage(GraphError::New(e.to_string())))?;
        (storage.point_mutations.load(Ordering::Acquire), r)
    };
    let (plan, due) = storage
        .named_vectors
        .plan_stored_links_be(&r, max_rows)
        .map_err(|e| RegraphRefusal::Storage(GraphError::from(e)))?;
    drop(r);
    let written = Instant::now();
    let mut status = LinkPassStatus {
        state: "done",
        remaining: due,
        readers_see_after: "restart",
        plan_ms: (written - started).as_millis() as u64,
        ..LinkPassStatus::default()
    };
    let Some(plan) = plan else {
        return Ok(status);
    };
    status.segment = Some(plan.segment.clone());
    status.unreached = plan.unreached;
    let changed = {
        let _held = lock_point_mutation_gate(&gate, &resolved, "link_pass");
        if storage.point_mutations.load(Ordering::Acquire) != mutations
            || storage.optimizer_running.load(Ordering::Acquire)
        {
            // Nothing is written: the segment is still to do.
            status.state = "busy";
            return Ok(status);
        }
        storage
            .with_write_backend(|w| {
                storage
                    .named_vectors
                    .apply_stored_links_be(w, std::slice::from_ref(&plan))
                    .map_err(GraphError::from)
            })
            .map_err(RegraphRefusal::Storage)?
    };
    // Counted once the plan is written: a busy call's rows are looked at
    // again by the next.
    status.without_room = plan.without_room;
    // The segment stays counted where a delete took a row of it between two
    // calls of its pass: the mark just written reads as due.
    status.remaining -= usize::from(plan.marks_done());
    storage.named_vectors.forget_stored_links(&changed);
    status.linked = changed.iter().map(|(_, rows)| rows.len()).sum();
    status.write_ms = written.elapsed().as_millis() as u64;
    tracing::info!(collection = %resolved, ?status, "link pass finished");
    Ok(status)
}

/// One exported dense row: id, vector and what the row keeps of the payload
/// it was written with, which is that payload or its digest.
type ExportedRow = (u128, Vec<f32>, HashMap<String, Value>);

/// What every key a digest is kept under begins with: those of the lists this
/// image has ([`KeyLists::row_digest_key`]) and those of rules it does not
/// have. Such a key is the one key of a dense row's `fields` when the row
/// keeps the digest of the payload it was written with in place of the
/// payload, and its value is a `Value::U64`. An upsert's payload reaches a
/// row through `json_to_value`, which makes every JSON number an `I64` or an
/// `F64`, so no payload copy reads as a digest, whatever keys a client sends.
///
/// A stored digest is frozen, and [`row_payload_digest`] is the content hash:
/// it moves with the hash and with the lists of keys the hash leaves out. A
/// change to any of them needs a new key of this family (a test pins them),
/// or a row digested before it would read as written with another payload
/// and lose its point to the other row of a pair.
const ROW_PAYLOAD_DIGEST_FAMILY: &str = "__helix_row_payload_digest";

/// What a dense row keeps of the payload it was written with.
#[derive(Debug, PartialEq)]
enum RowKeeps<'a> {
    /// No `VectorData` row, or an empty one.
    Nothing,
    /// The payload.
    Payload(&'a HashMap<String, Value>),
    /// Its digest, and the lists of keys the digest was made without.
    Digest(KeyLists, u64),
    /// A digest under another key of the family: a later image's rule, which
    /// this one cannot check and must not replace.
    DigestOfAnotherRule,
}

fn row_keeps(fields: &HashMap<String, Value>) -> RowKeeps<'_> {
    if fields.is_empty() {
        return RowKeeps::Nothing;
    }
    if let (1, Some((key, Value::U64(digest)))) = (fields.len(), fields.iter().next()) {
        for lists in [KeyLists::First, KeyLists::Second] {
            if key == lists.row_digest_key() {
                return RowKeeps::Digest(lists, *digest);
            }
        }
        if key.starts_with(ROW_PAYLOAD_DIGEST_FAMILY) {
            return RowKeeps::DigestOfAnotherRule;
        }
    }
    RowKeeps::Payload(fields)
}

/// `HELIX_DENSE_ROW_PAYLOAD=0` makes an LSM dense row keep the digest of its
/// point's payload where it kept the payload: an upsert writes the digest,
/// and a merge or regraph turns the copy of a row it rewrites into one. With
/// six dense spaces the payload was stored seven times, and every merge wrote
/// the six copies again. Only [`keep_the_copies_of_their_points`] reads what
/// a row keeps, and it reads both, so the switch decides what is written and
/// a row has its shape until its segment is next rewritten.
///
/// The copy is still the default: an image without this rule reads a digest
/// as a payload that is not the point's and would keep the other row of a
/// pair. Turn the switch on once no such image is rolled back to.
fn dense_row_payload_enabled() -> bool {
    !matches!(
        std::env::var("HELIX_DENSE_ROW_PAYLOAD")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref(),
        Some("0") | Some("false") | Some("off") | Some("no")
    )
}

/// The content hash of `payload` alone: the keys the content hash covers
/// under `lists` and no vector.
fn row_payload_digest(lists: KeyLists, payload: &HashMap<String, Value>) -> u64 {
    content_hash_of(lists, payload, &HashMap::new(), &HashMap::new())
}

/// The digest a row is written with: today's lists, under their key.
fn row_digest_fields(payload: &HashMap<String, Value>) -> HashMap<String, Value> {
    HashMap::from([(
        KeyLists::CURRENT.row_digest_key().to_string(),
        Value::U64(row_payload_digest(KeyLists::CURRENT, payload)),
    )])
}

/// What an upsert writes beside each dense vector of a point with `payload`.
fn dense_row_fields(payload: &HashMap<String, Value>) -> HashMap<String, Value> {
    if dense_row_payload_enabled() {
        payload.clone()
    } else {
        row_digest_fields(payload)
    }
}

/// Under `HELIX_DENSE_ROW_PAYLOAD=0` the rows a merge or regraph is about to
/// write keep the digest of the payload copy they carry, so a row rewritten
/// away from its pair can still be told from it. A row that has a digest, of
/// this rule or another, or that keeps nothing, stays as it is.
fn digest_in_place_of_a_payload_copy(rows: &mut [ExportedRow]) {
    if dense_row_payload_enabled() {
        return;
    }
    for (_, _, fields) in rows {
        let digest = match row_keeps(fields) {
            RowKeeps::Payload(payload) => row_digest_fields(payload),
            _ => continue,
        };
        *fields = digest;
    }
}

/// What a dense row says about the payload it was written with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WrittenWith {
    /// The point's payload: the row keeps that payload or its digest.
    ThePointsPayload,
    /// Another payload, or the digest of another.
    AnotherPayload,
    /// The row keeps neither, or a digest this image cannot check: it says
    /// nothing.
    Unknown,
}

/// `properties` is the point's payload as its node has it and `payload` the
/// same on the keys the content hash covers today; `digest` is worked out
/// for the first row that keeps one of today's lists.
///
/// A payload is compared with numbers of different kinds equal (`Value`'s
/// equality) and a digest is not. An upsert and the node's codec keep a
/// number's kind, so the two agree for a point as its upsert left it. A
/// `set_payload` puts a non-negative integer on the node as a `U64` where the
/// upsert made an `I64`: if it rewrites a key the hash covers with the number
/// it had, a row with the payload still reads as the point's and a row with
/// the digest as another's.
fn row_written_with(
    fields: &HashMap<String, Value>,
    properties: &HashMap<String, Value>,
    payload: &HashMap<String, Value>,
    digest: &mut Option<u64>,
) -> WrittenWith {
    let the_points = match row_keeps(fields) {
        RowKeeps::Nothing | RowKeeps::DigestOfAnotherRule => return WrittenWith::Unknown,
        RowKeeps::Digest(KeyLists::CURRENT, kept) => {
            kept == *digest.get_or_insert_with(|| row_payload_digest(KeyLists::CURRENT, payload))
        }
        // Older lists cover keys a node write may have changed since with no
        // row written: equal, the row was written with the point's payload;
        // unequal, it may still have been, and says nothing.
        RowKeeps::Digest(lists, kept) => {
            if kept != row_payload_digest(lists, properties) {
                return WrittenWith::Unknown;
            }
            true
        }
        RowKeeps::Payload(kept) => payload_without_content_hash_payload_only_keys(kept) == *payload,
    };
    if the_points {
        WrittenWith::ThePointsPayload
    } else {
        WrittenWith::AnotherPayload
    }
}

/// Which copy of a point a merge keeps: the one search serves, the other
/// copy at this index, or the one search serves because the rows do not say.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeptCopy {
    Search,
    Other(usize),
    Undecided,
}

/// The copy written with the point's payload where exactly one was. A row
/// that says nothing leaves its pair undecided: read as "another payload" it
/// would hand the point to whichever row was rewritten last.
fn copy_to_keep(served: WrittenWith, others: &[WrittenWith]) -> KeptCopy {
    if served == WrittenWith::Unknown || others.contains(&WrittenWith::Unknown) {
        return KeptCopy::Undecided;
    }
    let mut the_points = others
        .iter()
        .enumerate()
        .filter(|(_, other)| **other == WrittenWith::ThePointsPayload)
        .map(|(at, _)| at);
    match (served, the_points.next(), the_points.next()) {
        (WrittenWith::ThePointsPayload, None, _) => KeptCopy::Search,
        (WrittenWith::AnotherPayload, Some(other), None) => KeptCopy::Other(other),
        _ => KeptCopy::Undecided,
    }
}

/// Two sources of a merge can each hold a row of one id: until an upsert
/// removed the rows it replaces, a changed re-upsert left the old one where
/// it was. `rows` has the copy search serves, which is the old one when its
/// segment is numbered above the new one's (a merge target made while the
/// new row's tail was mutable). A row keeps the payload of the write that
/// made it, or that payload's digest, so the copy to keep is the one written
/// with the point's payload, compared on the keys the content hash covers:
/// `rows` gets that copy where exactly one has it. Where none or several have
/// it (a rewrite that changed vectors only), or a row keeps neither, the copy
/// search serves stays. Counted by which copy stayed: `search_copy` (it was
/// the point's own), `other_copy` (search was serving an older one) or
/// `undecided`. The points are read `NODE_CHECK_CHUNK` to a batched read: a
/// collection indexed again before upserts removed old rows has a second copy
/// of every rewritten point.
fn keep_the_copies_of_their_points(
    storage: &HelixGraphStorage,
    r: &super::backend_any::AnyRead<'_>,
    rows: &mut [ExportedRow],
    row_of: &HashMap<u128, usize>,
    mut other_copies: HashMap<u128, Vec<(Vec<f32>, HashMap<String, Value>)>>,
) -> Result<(), GraphError> {
    if other_copies.is_empty() {
        return Ok(());
    }
    let started = Instant::now();
    let (mut search_copy, mut other_copy, mut undecided) = (0u64, 0u64, 0u64);
    let ids: Vec<u128> = other_copies.keys().copied().collect();
    for chunk in ids.chunks(NODE_CHECK_CHUNK) {
        let keys: Vec<Vec<u8>> = chunk.iter().map(|id| id.to_be_bytes().to_vec()).collect();
        let nodes = storage
            .backend
            .get_many(r, Namespace::Nodes, &keys)
            .map_err(|e| GraphError::New(e.to_string()))?;
        for (id, node) in chunk.iter().zip(nodes) {
            // No point: the node check that follows leaves the row out.
            let Some(node) = node.and_then(|bytes| SerializedNode::decode_node(&bytes, *id).ok())
            else {
                continue;
            };
            let Some(&at) = row_of.get(id) else {
                continue;
            };
            let Some(row) = rows.get_mut(at) else {
                continue;
            };
            let payload = payload_without_content_hash_payload_only_keys(&node.properties);
            let mut digest = None;
            let served = row_written_with(&row.2, &node.properties, &payload, &mut digest);
            let mut others = other_copies.remove(id).unwrap_or_default();
            let said: Vec<WrittenWith> = others
                .iter()
                .map(|(_, fields)| {
                    row_written_with(fields, &node.properties, &payload, &mut digest)
                })
                .collect();
            match copy_to_keep(served, &said) {
                KeptCopy::Search => search_copy += 1,
                KeptCopy::Other(at) => {
                    let (data, fields) = others.swap_remove(at);
                    row.1 = data;
                    row.2 = fields;
                    other_copy += 1;
                }
                KeptCopy::Undecided => undecided += 1,
            }
        }
    }
    for (kept, ids) in [
        ("search_copy", search_copy),
        ("other_copy", other_copy),
        ("undecided", undecided),
    ] {
        if ids > 0 {
            metrics::counter!("helix_dense_merge_duplicate_ids_total", "kept" => kept)
                .increment(ids);
        }
    }
    record_dense_merge_publish_phase("copy_choice", started);
    Ok(())
}

/// The rows a regraph read of ids a more recent segment holds too (`doubled`:
/// each with the fields of those copies, the one search serves first).
/// Carried into the rebuilt segment a row outranks them all, so it goes to
/// `rows` only where `keep_the_copies_of_their_points` would keep it in a
/// merge of those segments: where it alone was written with the point's
/// payload. Returns how many stay out.
fn keep_the_doubled_rows_of_their_points(
    storage: &HelixGraphStorage,
    r: &super::backend_any::AnyRead<'_>,
    rows: &mut Vec<ExportedRow>,
    doubled: Vec<(ExportedRow, Vec<HashMap<String, Value>>)>,
) -> Result<u64, GraphError> {
    let exported = rows.len() + doubled.len();
    // The copy search serves stands in with no vector: a row that has one
    // after the choice is the rebuilt segment's.
    let mut served: Vec<ExportedRow> = Vec::with_capacity(doubled.len());
    let mut row_of = HashMap::new();
    let mut other_copies = HashMap::new();
    for ((id, data, fields), copies) in doubled {
        let mut copies = copies.into_iter();
        row_of.insert(id, served.len());
        served.push((id, Vec::new(), copies.next().unwrap_or_default()));
        let mut others = vec![(data, fields)];
        others.extend(copies.map(|fields| (Vec::new(), fields)));
        other_copies.insert(id, others);
    }
    keep_the_copies_of_their_points(storage, r, &mut served, &row_of, other_copies)?;
    rows.extend(served.into_iter().filter(|(_, data, _)| !data.is_empty()));
    Ok((exported - rows.len()) as u64)
}

/// `HELIX_LSM_MERGE_NODE_CHECK=0` lets merges and regraphs carry rows without
/// a point along again. The check costs a node read per exported row and
/// finds nothing once every segment written before it has merged.
fn lsm_merge_node_check_enabled() -> bool {
    !matches!(
        std::env::var("HELIX_LSM_MERGE_NODE_CHECK")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref(),
        Some("0") | Some("false") | Some("off") | Some("no")
    )
}

/// Node keys per batched read of [`retain_rows_with_a_node`]: the node rows
/// of one chunk are held while it is checked, beside an export that holds
/// every row anyway.
const NODE_CHECK_CHUNK: usize = 512;

/// Leaves out the exported `rows` whose point is gone and returns how many.
/// A merge used to republish the row of a point deleted while it was building
/// (see `merge_windows`), and no delete reaches that row again, so it would
/// be merged forward for good. `r` is the export's own snapshot: in it a row
/// has its node unless the point is gone, because an upsert writes both in
/// one batch and a delete removes both in one. The rows stay where there is
/// no batched read (the kill switch for those, or another backend).
fn retain_rows_with_a_node(
    storage: &HelixGraphStorage,
    r: &super::backend_any::AnyRead<'_>,
    rows: &mut Vec<ExportedRow>,
) -> Result<u64, GraphError> {
    if !lsm_merge_node_check_enabled() {
        return Ok(0);
    }
    let started = Instant::now();
    let mut present = Vec::with_capacity(rows.len());
    for chunk in rows.chunks(NODE_CHECK_CHUNK) {
        let keys: Vec<(Namespace<'_>, Vec<u8>)> = chunk
            .iter()
            .map(|(id, _, _)| (Namespace::Nodes, id.to_be_bytes().to_vec()))
            .collect();
        match storage
            .backend
            .contains_many(r, &keys)
            .map_err(|e| GraphError::New(e.to_string()))?
        {
            Some(found) => present.extend(found),
            None => return Ok(0),
        }
    }
    let exported = rows.len();
    let mut present = present.into_iter();
    rows.retain(|_| present.next().unwrap_or(true));
    let gone = (exported - rows.len()) as u64;
    if gone > 0 {
        metrics::counter!("helix_dense_merge_nodeless_rows_total").increment(gone);
    }
    metrics::counter!("helix_dense_merge_node_checked_rows_total").increment(exported as u64);
    record_dense_merge_publish_phase("node_check", started);
    Ok(gone)
}

/// `regraph_segments` over every eligible segment, also those the current
/// build made.
#[cfg(test)]
fn regraph_collection(
    storage: &Arc<HelixGraphStorage>,
    collection: &str,
    spaces: &[String],
    hnsw_config: &HNSWConfig,
    gate: &Arc<Mutex<()>>,
) -> Result<RegraphStatus, GraphError> {
    regraph_segments(storage, collection, spaces, hnsw_config, gate, false)
}

/// Rebuild every eligible segment of `spaces` one at a time; with
/// `skip_current`, not those whose graph the current build made. Export,
/// graph build and target write run without the point-mutation gate; it is
/// held only to re-check the write count and swap, so an upsert or delete
/// either lands before the export (and is in it), stands this segment down
/// as `busy` (its target dropped), or waits and lands on the swapped layout.
fn regraph_segments(
    storage: &Arc<HelixGraphStorage>,
    collection: &str,
    spaces: &[String],
    hnsw_config: &HNSWConfig,
    gate: &Arc<Mutex<()>>,
    skip_current: bool,
) -> Result<RegraphStatus, GraphError> {
    use crate::helix_engine::vector_core::named_vectors::DenseVectorSegmentRole;
    use crate::helix_engine::vector_core::vector_core::HNSW_DIVERSE_MIN_POINTS;
    let mut status = RegraphStatus::default();
    for vec_name in spaces {
        let Some(distance) = storage
            .named_vectors
            .get_config(vec_name)
            .map(|config| config.distance)
        else {
            continue;
        };
        let indexed: Vec<String> = storage
            .named_vectors
            .list_dense_vector_spaces()
            .remove(vec_name)
            .map(|space| space.segments)
            .unwrap_or_default()
            .into_iter()
            .filter(|segment| segment.role == DenseVectorSegmentRole::Indexed)
            .map(|segment| segment.physical_name)
            .collect();
        for phys in indexed {
            if skip_current && has_current_graph(storage, &phys)? {
                status.segments_current += 1;
                continue;
            }
            crate::helix_engine::vector_core::vector_core::defer_build_start_under_pressure();
            let build_permit = acquire_build_permit().map_err(GraphError::from)?;
            let started = Instant::now();
            // Snapshot and write count are taken together, between batches.
            let (mutations, r) = {
                let _held = lock_point_mutation_gate(gate, collection, "regraph");
                let r = storage
                    .backend
                    .begin_read()
                    .map_err(|e| GraphError::New(e.to_string()))?;
                (storage.point_mutations.load(Ordering::Acquire), r)
            };
            let exported = storage
                .named_vectors
                .export_for_regraph_be(&r, vec_name, &phys, HNSW_DIVERSE_MIN_POINTS)
                .map_err(GraphError::from)?;
            let Some(exported) = exported else {
                status.segments_small += 1;
                continue;
            };
            let mut rows = exported.rows;
            let dropped = exported.dropped
                + keep_the_doubled_rows_of_their_points(storage, &r, &mut rows, exported.doubled)?;
            status.nodeless_dropped += retain_rows_with_a_node(storage, &r, &mut rows)?;
            digest_in_place_of_a_payload_copy(&mut rows);
            drop(r);
            #[cfg(test)]
            test_support::reached("regraph_exported");
            // No row stays: every search walks the segment and it serves
            // nothing, so it goes with no target in its place.
            if rows.is_empty() {
                let held = lock_point_mutation_gate(gate, collection, "regraph");
                if storage.point_mutations.load(Ordering::Acquire) != mutations {
                    status.segments_busy += 1;
                    continue;
                }
                retire_dense_segment(storage, vec_name, &phys)?;
                storage.refresh_metadata_snapshot_best_effort();
                drop(held);
                status.segments_retired += 1;
                status.superseded_dropped += dropped;
                continue;
            }
            let points = rows.len();
            let prepared_index = VectorCore::build_hnsw_in_memory_owned_with_permit(
                rows,
                hnsw_config,
                distance.clone(),
                &build_permit,
            )
            .map_err(GraphError::from)?;
            let written = Instant::now();
            status.build_ms += (written - started).as_millis() as u64;
            // Already stale: skip the target write rather than write and drop it.
            if storage.point_mutations.load(Ordering::Acquire) != mutations {
                status.segments_busy += 1;
                continue;
            }
            let target = write_merge_target(storage, vec_name, hnsw_config, prepared_index)?;
            let swap_started = Instant::now();
            status.write_ms += (swap_started - written).as_millis() as u64;
            let held = lock_point_mutation_gate(gate, collection, "regraph");
            if storage.point_mutations.load(Ordering::Acquire) != mutations {
                drop(held);
                retire_dense_segment(storage, vec_name, &target)?;
                status.segments_busy += 1;
                continue;
            }
            swap_in_merge_target(storage, vec_name, &target, vec![phys])?;
            drop(held);
            status.swap_ms += swap_started.elapsed().as_millis() as u64;
            status.segments_rebuilt += 1;
            status.points_rebuilt += points;
            status.superseded_dropped += dropped;
        }
    }
    status.state = if status.segments_busy > 0 {
        "busy"
    } else {
        "done"
    };
    tracing::info!(collection, ?status, "regraph finished");
    Ok(status)
}

/// True when dense segment `phys` carries the current build's number.
fn has_current_graph(storage: &HelixGraphStorage, phys: &str) -> Result<bool, GraphError> {
    let r = storage
        .backend
        .begin_read()
        .map_err(|e| GraphError::New(e.to_string()))?;
    let cores = storage
        .named_vectors
        .cores_read()
        .map_err(GraphError::from)?;
    match cores.get(phys) {
        Some(core) => core.has_current_graph(&r).map_err(GraphError::from),
        None => Ok(false),
    }
}

/// Stores the dense layout of an LSM collection: the one its memory has when
/// the transaction runs, with `seal` in it where the caller is a seal whose
/// next tail memory does not have yet. Every other caller has changed memory
/// under `held` and passes none.
///
/// This is every store of the layout on LSM, and it is made under the
/// collection's [`DenseLayoutChange`]: no two cross, so the store that lands
/// last read memory last. Memory is read inside the transaction because a
/// conflict runs the transaction again (another writer of the metadata row:
/// a payload index, a sparse space), and the second run stores what memory
/// has then.
///
/// It is also the one place that records a dead segment name
/// ([`DeadNamesOfAStore`]): the names this store takes off the stored layout
/// get their keys in its transaction, and memory hears of them once it has
/// committed.
fn store_dense_layout(
    storage: &HelixGraphStorage,
    _held: &DenseLayoutChange<'_>,
    w: &mut super::backend_any::AnyWrite<'_>,
    seal: Option<&PreparedTailSeal>,
) -> Result<(), GraphError> {
    let mut dead = None;
    storage.update_metadata_be_with_rows(w, |metadata| {
        let mut layout = storage
            .named_vectors
            .dense_vector_spaces_to_store()
            .map_err(GraphError::from)?;
        #[cfg(test)]
        test_support::reached("layout_read");
        if let Some(seal) = seal {
            seal.flip_in(&mut layout);
        }
        report_dense_layout_faults(storage, &metadata.dense_vector_spaces, &layout);
        // Of this run of the transaction: a run a conflict makes reads the
        // stored layout again and decides again.
        dead = DeadNamesOfAStore::of(storage, &metadata.dense_vector_spaces, &layout);
        let rows = dead
            .as_ref()
            .map(DeadNamesOfAStore::rows)
            .unwrap_or_default();
        metadata.set_dense_vector_spaces(layout);
        Ok(rows)
    })?;
    if let Some(dead) = dead {
        dead.committed(storage);
    }
    Ok(())
}

/// What a store of the dense layout does to the dead segment names
/// (`DEAD_SEGMENT_KEY_PREFIX` in named_vectors.rs), decided from the layout
/// the store replaces as its transaction read it.
///
/// A segment the stored layout lists and the next one does not gets a key.
/// That is every way a name leaves the layout (a merge's sources, a retire,
/// a merge target nothing published, an emptied segment) with no site
/// counted, and it holds where a site took the name out of memory and its
/// own store then failed: the name is still in the stored layout, and the
/// next store that lands records it, whichever site makes it.
///
/// A segment the next layout lists that has a key loses the key: a listed
/// segment is served, whatever a key says.
///
/// Where a store takes every stored segment of a space off at once, none of
/// them gets a key. No change of a healthy space does that (its tail stays
/// through a seal, a build, a merge and a removal of empties): it is a fault,
/// or the removal of the emptied segments of a space that had no tail, and
/// keys for them would call every row the space ever had dead.
struct DeadNamesOfAStore {
    recorded: Vec<(String, u64)>,
    listed_again: Vec<String>,
    withheld: usize,
}

impl DeadNamesOfAStore {
    /// None where the collection keeps no record of dead names.
    fn of(
        storage: &HelixGraphStorage,
        stored: &HashMap<String, DenseVectorSpaceMetadata>,
        next: &HashMap<String, DenseVectorSpaceMetadata>,
    ) -> Option<Self> {
        fn names(space: &DenseVectorSpaceMetadata) -> impl Iterator<Item = &str> {
            space
                .segments
                .iter()
                .map(|segment| segment.physical_name.as_str())
        }
        let listed_again = storage
            .named_vectors
            .dead_among(next.values().flat_map(names))?;
        let at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_millis() as u64);
        let mut dead = Self {
            recorded: Vec::new(),
            listed_again,
            withheld: 0,
        };
        for (space, had) in stored {
            let now: HashSet<&str> = next.get(space).into_iter().flat_map(names).collect();
            let lost: Vec<&str> = names(had).filter(|name| !now.contains(name)).collect();
            if lost.len() == had.segments.len() {
                dead.withheld += lost.len();
                continue;
            }
            dead.recorded
                .extend(lost.into_iter().map(|name| (name.to_string(), at)));
        }
        Some(dead)
    }

    /// The rows of the metadata table that go into the store's transaction.
    fn rows(&self) -> Vec<super::backend::TransactionRow> {
        use crate::helix_engine::vector_core::named_vectors::dead_segment_key;
        let puts = self
            .recorded
            .iter()
            .map(|(name, at)| (dead_segment_key(name), Some(at.to_be_bytes().to_vec())));
        let deletes = self
            .listed_again
            .iter()
            .map(|name| (dead_segment_key(name), None));
        puts.chain(deletes).collect()
    }

    /// Once the store has committed: memory takes the names, and they are
    /// counted and said.
    fn committed(self, storage: &HelixGraphStorage) {
        storage
            .named_vectors
            .dead_segments_stored(&self.recorded, &self.listed_again);
        if !self.recorded.is_empty() {
            metrics::counter!("helix_dense_dead_names_recorded_total")
                .increment(self.recorded.len() as u64);
        }
        if self.withheld > 0 {
            metrics::counter!("helix_dense_dead_names_withheld_total")
                .increment(self.withheld as u64);
            tracing::error!(
                collection = storage.collection_display_name(),
                segments = self.withheld,
                "A store of the dense layout took every stored segment of a space off at once; none of them was recorded as dead"
            );
        }
        for name in &self.listed_again {
            metrics::counter!("helix_dense_dead_names_cleared_total", "reason" => "listed_again")
                .increment(1);
            tracing::error!(
                collection = storage.collection_display_name(),
                segment = %name,
                "A dense segment with a dead key is in the layout that was stored; its key was deleted with that store"
            );
        }
    }
}

/// Says at ERROR what a layout about to be stored should never show. A
/// space that has segments stored and none now: stored so, it opens as its
/// bare name alone and every segment it had is unlisted. A space with
/// segments and no mutable tail: no append can reach it. Neither stops the
/// store: a space whose segments were all emptied and removed comes to the
/// first with nothing lost, and what passes through the second is not known.
fn report_dense_layout_faults(
    storage: &HelixGraphStorage,
    stored: &HashMap<String, DenseVectorSpaceMetadata>,
    next: &HashMap<String, DenseVectorSpaceMetadata>,
) {
    use crate::helix_engine::vector_core::named_vectors::DenseVectorSegmentRole;
    for (space, had) in stored {
        if !had.segments.is_empty() && next.get(space).is_none_or(|now| now.segments.is_empty()) {
            metrics::counter!("helix_dense_layout_store_fault_total", "fault" => "space_without_segments")
                .increment(1);
            tracing::error!(
                collection = storage.collection_display_name(),
                vector = %space,
                segments_stored = had.segments.len(),
                "Dense layout store: a space that has segments stored is stored with none"
            );
        }
    }
    for (space, now) in next {
        let has_tail = now
            .segments
            .iter()
            .any(|segment| segment.role == DenseVectorSegmentRole::Mutable);
        if !now.segments.is_empty() && !has_tail {
            metrics::counter!("helix_dense_layout_store_fault_total", "fault" => "space_without_tail")
                .increment(1);
            tracing::error!(
                collection = storage.collection_display_name(),
                vector = %space,
                segments = now.segments.len(),
                "Dense layout store: a space has segments and no mutable tail"
            );
        }
    }
}

/// Retires an obsolete segment: persist its removal before unlinking sidecars
/// and handing its pages to the reaper.
fn retire_dense_segment(
    storage: &Arc<HelixGraphStorage>,
    vec_name: &str,
    target: &str,
) -> Result<(), GraphError> {
    let names = [target.to_string()];
    {
        let held = storage.named_vectors.begin_dense_layout_change();
        storage
            .named_vectors
            .apply_dense_segment_removal(vec_name, &names)
            .map_err(GraphError::from)?;
        storage.with_write_backend(|w| store_dense_layout(storage, &held, w, None))?;
    }
    let sidecars = storage
        .named_vectors
        .unlink_sidecar_files_for_segments(storage.path(), &names);
    if sidecars.error_count > 0 {
        tracing::warn!(
            errors = sidecars.error_count,
            "sidecar unlink for a retired dense segment had errors"
        );
    }
    SEGMENT_REAPER.submit(Arc::clone(storage), target.to_string());
    Ok(())
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
    if storage.backend.is_reader_replica() {
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
    let _ = storage.submit_payload_marker_backfill(collection);
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
    if !optimizer_has_pending_work(
        storage,
        &spaces_to_check,
        flat_scan_threshold,
        index_subthreshold_quiesced_tails(),
    ) {
        metrics::counter!(
            "helix_post_open_vector_maintenance_total",
            "outcome" => "no_pending_work"
        )
        .increment(1);
        return Ok(0);
    }
    let hnsw_config = collection_hnsw_config(storage, config)?;

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
        Self::with_workers(segment_reaper_threads())
    }

    fn with_workers(worker_count: usize) -> Self {
        let state = Arc::new((
            Mutex::new(ReaperQueueState::default()),
            std::sync::Condvar::new(),
        ));
        let queue_depth = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let queued_or_running = Arc::new(Mutex::new(HashSet::new()));
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
            // Uncaught, a panic ended this thread and dropped the job with
            // its key still in `queued_or_running`, so its segment was never
            // submitted again. Caught, the job takes the failed path.
            let turn = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                Self::run_job_turn(&mut job, yield_ms, scale)
            }));
            let outcome = turn.unwrap_or_else(|panic| {
                metrics::counter!("helix_segment_reaper_panics_total").increment(1);
                tracing::error!(
                    collection = %job.storage.path().display(),
                    segment = %job.physical_name,
                    panic = panic_text(&*panic),
                    "segment reaper turn panicked; its worker goes on and the job is tried again as one that failed"
                );
                ReaperJobOutcome::Failed
            });

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
        #[cfg(test)]
        panic_once_for_test(&job.physical_name);
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
        use crate::helix_engine::vector_core::named_vectors::dead_segment_key;
        // The segment's dead key, where this job has shown its rows gone. On
        // LSM a table counts as drained for a segment with no core open
        // without being read; with the core open, each of the eight was read
        // empty by a step of this job, and only that takes the key away.
        let named = &job.storage.named_vectors;
        let dead_key = (job.storage.backend.kind() == BackendKind::Lsm
            && named.is_dead_segment(&job.physical_name)
            && named
                .cores_read()
                .is_ok_and(|cores| cores.contains_key(&job.physical_name)))
        .then(|| dead_segment_key(&job.physical_name));
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
                // leftover keys are harmless and reconstructed/ignored. The
                // dead key goes in the same write: no commit of its own.
                let cleared = if job.storage.backend.kind() == BackendKind::Lsm {
                    job.storage
                        .backend
                        .begin_read()
                        .map_err(|e| GraphError::New(e.to_string()))
                        .and_then(|r| {
                            job.storage.with_write_backend(|w| {
                                if let Some(key) = &dead_key {
                                    job.storage
                                        .backend
                                        .delete(w, Namespace::Metadata, key)
                                        .map_err(|e| GraphError::New(e.to_string()))?;
                                }
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
                    Ok(n) => {
                        if n > 0 {
                            metrics::counter!("helix_reupsert_tombstones_cleared_on_retire_total")
                                .increment(n as u64);
                        }
                        if dead_key.is_some() {
                            named.dead_segments_stored(
                                &[],
                                std::slice::from_ref(&job.physical_name),
                            );
                            metrics::counter!(
                                "helix_dense_dead_names_cleared_total",
                                "reason" => "read_empty"
                            )
                            .increment(1);
                        }
                    }
                    Err(e) => {
                        // Its dead key, if it has one, stays with it: the
                        // name is passed over by the allocator and no more.
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

    /// Test-only: nothing queued or running, so no reaper reads are in flight.
    #[cfg(test)]
    fn idle(&self) -> bool {
        self.queued_or_running
            .lock()
            .map(|names| names.is_empty())
            .unwrap_or(true)
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
///
/// `window` was opened before the merge read its sources: the points removed
/// since then are deleted from the target between phases C and D, so the swap
/// never publishes their rows.
fn run_chunked_merge_publish(
    storage: &Arc<HelixGraphStorage>,
    vec_name: &str,
    hnsw_config: &HNSWConfig,
    prepared_merge: crate::helix_engine::vector_core::named_vectors::PreparedMerge,
    window: Option<&crate::helix_engine::vector_core::merge_windows::MergeWindow<'_>>,
) -> Result<(), GraphError> {
    let publish_started = Instant::now();
    // The ids the target gets a row for: a removal of any other id left
    // nothing in it.
    let exported: HashSet<u128> = match window {
        Some(_) => prepared_merge
            .prepared_index
            .point_ids
            .iter()
            .copied()
            .collect(),
        None => HashSet::new(),
    };
    let merged_name = write_merge_target(
        storage,
        vec_name,
        hnsw_config,
        prepared_merge.prepared_index,
    )?;
    #[cfg(test)]
    test_support::reached("written");
    if let Some(window) = window {
        let phase_started = Instant::now();
        loop {
            let mut removed = window.removed();
            if removed.is_empty() {
                break;
            }
            removed.retain(|id| exported.contains(id));
            if removed.is_empty() {
                continue;
            }
            let staged = storage.with_write_backend(|w| {
                storage
                    .named_vectors
                    .delete_from_segment_be(w, &merged_name, &removed)
                    .map_err(GraphError::from)
            })?;
            storage.named_vectors.apply_delete_tombstones(&staged);
            metrics::counter!("helix_dense_merge_removed_ids_total")
                .increment(removed.len() as u64);
        }
        record_dense_merge_publish_phase("removed_rows", phase_started);
    }
    #[cfg(test)]
    test_support::reached("drained");
    swap_in_merge_target(
        storage,
        vec_name,
        &merged_name,
        prepared_merge.merge_targets,
    )?;
    metrics::histogram!("helix_dense_merge_publish_duration_ms")
        .record(publish_started.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}

/// Phases A-C of `run_chunked_merge_publish`: writes the merge target, an
/// unmarked `Building` segment search does not see, and returns its name.
fn write_merge_target(
    storage: &Arc<HelixGraphStorage>,
    vec_name: &str,
    hnsw_config: &HNSWConfig,
    mut prepared_index: crate::helix_engine::vector_core::vector_core::PreparedIndex,
) -> Result<String, GraphError> {
    let max_chunk_size = merge_chunk_size();
    let segment_format = dense_segment_format();
    // Group commit: the commits after phase A go out buffered and one durable
    // flush ends the write, so a swap only ever sees a durable target. A crash
    // before the flush leaves phase A's marked target, which the index job reaps.
    let buffered = storage.backend.kind() == BackendKind::Lsm && lsm_group_commit_enabled();

    // Phase A: allocate + create LMDB databases. `create_merge_target` now
    // registers the segment in `space.segments` as `Building` so the ID is
    // reserved in persisted metadata. Persisting the metadata blob here means
    // a crash between A and D leaves an orphan that
    // `gc_orphan_building_segments_deferred` can detect and queue to
    // `SEGMENT_REAPER` on the next index cycle.
    let phase_started = Instant::now();
    let merged_name = if storage.backend.kind() == BackendKind::Lsm {
        let held = storage.named_vectors.begin_dense_layout_change();
        let name = storage
            .named_vectors
            .create_merge_target_lsm(
                &held,
                storage.collection_path(),
                vec_name,
                hnsw_config.clone(),
            )
            .map_err(GraphError::from)?;
        storage.with_write_backend(|w| {
            storage
                .named_vectors
                .mark_merge_target_be(w, &name)
                .map_err(GraphError::from)?;
            store_dense_layout(storage, &held, w, None)
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
            with_merge_target_write(storage, buffered, |w| {
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
            with_merge_target_write(storage, buffered, |w| {
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
    #[cfg(test)]
    test_support::reached("flat_written");

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
            with_merge_target_write(storage, buffered, |w| {
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
        with_merge_target_write(storage, buffered, |w| {
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
                with_merge_target_write(storage, buffered, |w| {
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
    if buffered {
        let phase_started = Instant::now();
        storage
            .backend
            .flush_durable()
            .map_err(|e| GraphError::New(e.to_string()))?;
        record_dense_merge_publish_phase("group_flush", phase_started);
    }
    Ok(merged_name)
}

/// `with_write_backend`, or its buffered form when `buffered` (group commit).
fn with_merge_target_write<T>(
    storage: &HelixGraphStorage,
    buffered: bool,
    f: impl FnOnce(&mut super::backend_any::AnyWrite<'_>) -> Result<T, GraphError>,
) -> Result<T, GraphError> {
    if buffered {
        storage.with_write_backend_buffered(f)
    } else {
        storage.with_write_backend(f)
    }
}

/// Phases D-E of `run_chunked_merge_publish`: swaps `merged_name` in for
/// `merge_targets` and hands the retired segments to the reaper.
fn swap_in_merge_target(
    storage: &Arc<HelixGraphStorage>,
    vec_name: &str,
    merged_name: &str,
    merge_targets: Vec<String>,
) -> Result<(), GraphError> {
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
        let held = storage.named_vectors.begin_dense_layout_change();
        storage
            .named_vectors
            .publish_merge_target_be(&r, vec_name, merged_name, &merge_targets)
            .map_err(GraphError::from)?;
        storage
            .named_vectors
            .apply_dense_segment_removal(vec_name, &deferred_empties)
            .map_err(GraphError::from)?;
        storage.with_write_backend(|w| store_dense_layout(storage, &held, w, None))?;
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
                .publish_merge_target(txn, vec_name, merged_name, &merge_targets)
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
    crate::diag::log("phase_d_done", "chunked_merge", merged_name);

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
    crate::diag::log("phase_e_done", "chunked_merge", merged_name);

    Ok(())
}

/// Seal an LSM space's mutable tail to `Building` and persist the layout,
/// keeping the sealed points searchable (flat) until the build promotes them.
///
/// In three steps under one layout change. The next tail is opened and
/// memory is left as it is; the layout with the seal in it is stored; only
/// then does memory get the seal, and with it the next append its tail. An
/// upsert applied beside a seal takes its tail from memory, so a row is
/// never under a tail that the store does not name yet: with memory changed
/// first, a writer that died before the store, or whose store failed, had
/// acknowledged rows the next writer could not find. A step that fails
/// leaves memory unchanged. Rows appended between the store and the change
/// of memory go to the sealed tail, as the rows of an upsert that was being
/// applied when it was sealed always did: the build waits for those.
///
/// The sealed-tail marker commits before the layout that shows the tail
/// `Building`, so neither a writer-side search nor a reader that sees the new
/// layout can observe an unmarked sealed tail (which would hide its points
/// and let a stale older copy of a re-upserted id win).
pub(super) fn seal_lsm_dense_tail(
    storage: &Arc<HelixGraphStorage>,
    vec_name: &str,
    hnsw_config: &HNSWConfig,
) -> Result<bool, GraphError> {
    let begin_read = || {
        storage
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))
    };
    let held = storage.named_vectors.begin_dense_layout_change();
    let Some(seal) = storage
        .named_vectors
        .prepare_tail_seal_lsm(
            &held,
            &begin_read()?,
            storage.collection_path(),
            vec_name,
            hnsw_config.clone(),
        )
        .map_err(GraphError::from)?
    else {
        return Ok(false);
    };
    if !storage
        .named_vectors
        .sealed_tail_marked_be(&begin_read()?, seal.sealed())
        .map_err(GraphError::from)?
    {
        storage.with_write_backend(|w| {
            storage
                .named_vectors
                .mark_sealed_tail_be(w, seal.sealed())
                .map_err(GraphError::from)
        })?;
    }
    #[cfg(test)]
    test_support::reached("seal_prepared");
    // The batch stays empty, and the commit of an empty batch writes nothing
    // and cannot fail: once the layout's own transaction is in, nothing
    // before the change of memory fails or waits.
    storage.with_write_backend(|w| store_dense_layout(storage, &held, w, Some(&seal)))?;
    #[cfg(test)]
    test_support::reached("seal_midway");
    storage.named_vectors.apply_tail_seal(&held, seal);
    Ok(true)
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
/// The segment is in `Building` role throughout; dense search either skips
/// it or, for a marked sealed tail, scans its level-0 rows flat (it has no
/// entry point until the final chunk). Worker death between chunks leaves a
/// Building segment with partial HNSW; gc_orphan_building_segments / next
/// pass re-prepares from level zero (idempotent).
///
/// Group commit on LSM, as in `write_merge_target`: a durable commit does not
/// flush the WAL, it waits for the next flush tick, so a durable commit per
/// chunk published two chunks a second. The chunks go out buffered and one
/// flush makes them durable before the entry point is written. The entry
/// point is the only thing a restart promotes a Building segment on, and it
/// is still committed durably, after every chunk, so a crash at any point
/// leaves either no entry point (re-prepared) or the whole graph.
fn run_chunked_indices_publish(
    storage: &Arc<HelixGraphStorage>,
    vec_name: &str,
    prepared: Vec<(
        String,
        crate::helix_engine::vector_core::vector_core::PreparedIndex,
    )>,
) -> Result<bool, GraphError> {
    let max_chunk_size = merge_chunk_size();
    let buffered = storage.backend.kind() == BackendKind::Lsm && lsm_group_commit_enabled();
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
                with_merge_target_write(storage, buffered, |w| {
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
        if buffered {
            let phase_started = Instant::now();
            storage
                .backend
                .flush_durable()
                .map_err(|e| storage.backend_error(e, "indices_group_flush"))?;
            record_dense_merge_publish_phase("indices_group_flush", phase_started);
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
            let held = storage.named_vectors.begin_dense_layout_change();
            storage
                .named_vectors
                .promote_building_segments_to_indexed_be(&r, vec_name, &flushed_names)
                .map_err(GraphError::from)?;
            storage.with_write_backend(|w| store_dense_layout(storage, &held, w, None))?;
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

fn recover_indexed_building_segments(
    storage: &Arc<HelixGraphStorage>,
    collection: &str,
    vec_name: &str,
) -> Result<usize, GraphError> {
    let indexed = if storage.backend.kind() == BackendKind::Lsm {
        let read = storage
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))?;
        storage
            .named_vectors
            .indexed_building_segment_names_be(&read, vec_name)
            .map_err(GraphError::from)?
    } else {
        storage.with_read_txn(|txn| {
            storage
                .named_vectors
                .indexed_building_segment_names(txn, vec_name)
                .map_err(GraphError::from)
        })?
    };
    if indexed.is_empty() {
        return Ok(0);
    }

    let held = storage.named_vectors.begin_dense_layout_change();
    let changed = if storage.backend.kind() == BackendKind::Lsm {
        let read = storage
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))?;
        storage
            .named_vectors
            .promote_building_segments_to_indexed_be(&read, vec_name, &indexed)
            .map_err(GraphError::from)?
    } else {
        storage.with_read_txn(|txn| {
            storage
                .named_vectors
                .promote_building_segments_to_indexed(txn, vec_name, &indexed)
                .map_err(GraphError::from)
        })?
    };
    if !changed {
        return Ok(0);
    }

    if storage.backend.kind() == BackendKind::Lsm {
        storage.with_write_backend(|write| store_dense_layout(storage, &held, write, None))?;
    } else {
        storage.with_write_txn(|txn| {
            storage.set_dense_vector_spaces_metadata(
                txn,
                storage.named_vectors.list_dense_vector_spaces(),
            )?;
            Ok(())
        })?;
    }
    drop(held);

    metrics::counter!(
        "helix_index_job_recovered_building_segments_total",
        "collection" => collection.to_string(),
        "vector" => vec_name.to_string()
    )
    .increment(indexed.len() as u64);
    tracing::info!(
        collection = %collection,
        vector = %vec_name,
        segments = indexed.len(),
        "optimizer promoted completed Building segments without rebuilding"
    );
    Ok(indexed.len())
}

fn run_index_job(job: IndexJob) {
    let IndexJob {
        collection,
        storage,
        spaces_to_check,
        hnsw_config,
        flat_scan_threshold,
    } = job;
    let _regraph = RegraphJobGuard(collection.clone());
    #[cfg(test)]
    panic_once_for_test(&collection);
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
            let held = storage.named_vectors.begin_dense_layout_change();
            storage.with_write_backend(|w| store_dense_layout(&storage, &held, w, None))
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
        const MAX_PASSES: usize = 20;
        let index_quiesced_tails = index_subthreshold_quiesced_tails();
        let quiet_ms = collection_tail_quiet_ms(&storage);
        let seal_min = collection_tail_seal_min(&storage, flat_scan_threshold);
        let waits_for_quiet = quiet_ms <= INDEX_TAIL_QUIET_WAIT_IN_JOB_MAX_MS;
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
                    // Until the store of what the cleanup drops from memory.
                    let held = storage.named_vectors.begin_dense_layout_change();
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
                                        store_dense_layout(&storage, &held, w, None)
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
                            drop(held);
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

                let had_building_debt = spaces_to_check.iter().any(|name| {
                    !storage
                        .named_vectors
                        .building_segment_names(name)
                        .is_empty()
                });
                if had_building_debt {
                    for vec_name in &spaces_to_check {
                        recover_indexed_building_segments(&storage, &collection, vec_name)?;
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
                    if had_building_debt {
                        break false;
                    }
                    if point_mutations_quiet_for(&storage, quiet_ms) {
                        break true;
                    }
                    if !waits_for_quiet {
                        break false;
                    }
                    let above_threshold = spaces_to_check.iter().any(|name| {
                        // LSM counts the tail's rows only, and only up to the
                        // threshold: this runs every 100 ms, and the full
                        // stats scan every row of every segment.
                        let unindexed = if storage.backend.kind() == BackendKind::Lsm {
                            storage.backend.begin_read().ok().and_then(|r| {
                                storage
                                    .named_vectors
                                    .mutable_tail_count_upto_be(
                                        &r,
                                        name,
                                        u64::try_from(flat_scan_threshold.max(1))
                                            .unwrap_or(u64::MAX),
                                    )
                                    .ok()
                            })
                        } else {
                            storage.begin_resize_safe_read_txn().ok().and_then(|rtxn| {
                                storage
                                    .named_vectors
                                    .get_dense_space_stats(&rtxn, name)
                                    .ok()
                                    .map(|stats| {
                                        stats
                                            .vectors_count
                                            .saturating_sub(stats.indexed_vectors_count)
                                    })
                            })
                        };
                        if let Some(unindexed) = unindexed {
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
                let spaces_to_seal: Vec<String> = if had_building_debt {
                    Vec::new()
                } else if storage.backend.kind() == BackendKind::Lsm {
                    let r = storage
                        .backend
                        .begin_read()
                        .map_err(|e| GraphError::New(e.to_string()))?;
                    let mut to_seal = Vec::new();
                    for vec_name in &spaces_to_check {
                        // The tail's own rows, counted up to max(threshold, 1):
                        // enough for both tests below.
                        if let Ok(tail_rows) = storage.named_vectors.mutable_tail_count_upto_be(
                            &r,
                            vec_name,
                            u64::try_from(flat_scan_threshold.max(1)).unwrap_or(u64::MAX),
                        ) {
                            let above_threshold = tail_rows as usize >= flat_scan_threshold;
                            let quiesced_tail = index_quiesced_tails
                                && subthreshold_tail_is_sealable(tail_rows, quiesced, seal_min);
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
                                let quiesced_tail = index_quiesced_tails
                                    && subthreshold_tail_is_sealable(unindexed, quiesced, seal_min);
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
                        seal_lsm_dense_tail(&storage, vec_name, &hnsw_config)?
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
                                // The export is every row the graph will
                                // reach. A batch being applied when the tail
                                // was sealed commits its rows afterwards.
                                if !point_mutation_in_flight_has_ended(&storage, || {
                                    budget.elapsed_exceeded()
                                }) {
                                    needs_reschedule = true;
                                    break 'build_spaces;
                                }
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
                            let held = storage.named_vectors.begin_dense_layout_change();
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
                                    store_dense_layout(&storage, &held, w, None)
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
                    // Open from before the sources are read until the target
                    // is published: a point deleted or rewritten meanwhile
                    // must not come back in the target.
                    let merge_window = (storage.backend.kind() == BackendKind::Lsm)
                        .then(|| storage.named_vectors.merge_windows.open());
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
                        // Where each id's row is in `exported`, and the other
                        // copies of an id two sources hold.
                        let mut seen: HashMap<u128, usize> = HashMap::new();
                        let mut other_copies: HashMap<u128, Vec<_>> = HashMap::new();
                        let mut dropped_tombstoned = 0u64;
                        // Newest first, so `exported` gets the copy search
                        // serves when two sources hold an id.
                        for phys in &NamedVectorManager::newest_first(vec_name, &targets) {
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
                                    if seen.contains_key(&id) {
                                        other_copies.entry(id).or_default().push((data, fields));
                                    } else {
                                        seen.insert(id, exported.len());
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
                        keep_the_copies_of_their_points(
                            &storage,
                            &r,
                            &mut exported,
                            &seen,
                            other_copies,
                        )?;
                        retain_rows_with_a_node(&storage, &r, &mut exported)?;
                        // After the choice, which reads the copies as stored.
                        digest_in_place_of_a_payload_copy(&mut exported);
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
                    #[cfg(test)]
                    test_support::reached("exported");
                    // Tombstones can leave no live row to publish; retire the sources directly.
                    if exported.is_empty() {
                        for target in &targets {
                            retire_dense_segment(&storage, vec_name, target)?;
                        }
                        budget.record_merge();
                        continue;
                    }
                    let prepared_index = VectorCore::build_hnsw_in_memory_owned_with_permit(
                        exported,
                        &hnsw_config,
                        storage
                            .named_vectors
                            .get_config(vec_name)
                            .map(|config| config.distance)
                            .ok_or_else(|| {
                                GraphError::New(format!("Named vector '{}' not found", vec_name))
                            })?,
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
                    let flush_outcome = run_chunked_merge_publish(
                        &storage,
                        vec_name,
                        &hnsw_config,
                        prepared_merge,
                        merge_window.as_ref(),
                    );
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
                        if let Ok(tail_rows) = storage
                            .named_vectors
                            .mutable_tail_count_upto_be(&r, name, 1)
                        {
                            return tail_rows > 0;
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

            if point_mutations_quiet_for(&storage, quiet_ms) {
                continue;
            }
            if !waits_for_quiet {
                // The next pass would only wait. A turn that spent its budget
                // hands over as it does at the top of a pass; a tail that
                // outgrew the seal minimum meanwhile is then pending work.
                needs_reschedule |= budget.should_yield();
                break;
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
                let held = storage.named_vectors.begin_dense_layout_change();
                let mut layout_changed = false;
                for (vec_name, empties) in &empties_per_space {
                    let removed = storage
                        .named_vectors
                        .apply_dense_segment_removal(vec_name, empties)
                        .map_err(GraphError::from)?;
                    layout_changed |= removed;
                }
                if layout_changed {
                    storage.with_write_backend(|w| store_dense_layout(&storage, &held, w, None))?;
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
    let result = result.and_then(|()| run_queued_regraph(&storage, &collection, &hnsw_config));

    let status = if result.is_ok() { "ok" } else { "error" };
    if let Err(e) = result {
        storage.mark_collection_degraded(&e, "index_job");
        if matches!(e, GraphError::MapFull) {
            tracing::warn!(
                error = %e,
                collection = %collection,
                "background optimizer deferred cycle (MapFull); next cycle retries on larger map"
            );
        } else if storage.closing_for_shutdown.load(Ordering::Acquire) {
            // The shutdown step closed the database under this job. The next
            // start runs it again; it is not an optimizer failure to alert on.
            tracing::debug!(
                error = %e,
                collection = %collection,
                "background optimizer stopped by the shutdown close"
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
        let building = storage.named_vectors.building_segment_names(name);
        storage
            .named_vectors
            .cores_read()
            .is_ok_and(|cores| building.iter().any(|segment| cores.contains_key(segment)))
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

    if storage.backend.kind() != BackendKind::Lsm
        && externalized_marker_repair_limit_per_segment() > 0
        && spaces_to_check.iter().any(|name| {
            storage
                .named_vectors
                .has_externalized_marker_repair_debt(name)
                .unwrap_or(false)
        })
    {
        return true;
    }

    // The index job's sealing rule: a tail it would leave flat is not work.
    // A job that waits a short quiet period out seals any tail it is given.
    let quiet_ms = collection_tail_quiet_ms(storage);
    let quiesced = quiet_ms <= INDEX_TAIL_QUIET_WAIT_IN_JOB_MAX_MS
        || point_mutations_quiet_for(storage, quiet_ms);
    let seal_min = collection_tail_seal_min(storage, flat_scan_threshold);
    let merge_ceiling = NamedVectorManager::max_indexed_segments_target();
    if storage.backend.kind() == BackendKind::Lsm {
        (|| -> Result<bool, GraphError> {
            let r = storage
                .backend
                .begin_read()
                .map_err(|e| GraphError::New(e.to_string()))?;
            for vec_name in spaces_to_check {
                // One row answers a quiet collection; otherwise count up to
                // the threshold, which the seal minimum never exceeds.
                let limit = if index_quiesced_tails && quiesced {
                    1
                } else {
                    flat_scan_threshold.max(1)
                };
                if let Ok(tail_rows) = storage.named_vectors.mutable_tail_count_upto_be(
                    &r,
                    vec_name,
                    u64::try_from(limit).unwrap_or(u64::MAX),
                ) {
                    if (tail_rows as usize) >= flat_scan_threshold
                        || (index_quiesced_tails
                            && subthreshold_tail_is_sealable(tail_rows, quiesced, seal_min))
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
                            || (index_quiesced_tails
                                && subthreshold_tail_is_sealable(unindexed, quiesced, seal_min))
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
    /// Route collection-scoped reads to a rendezvous-chosen reader so each
    /// collection is cold-opened and cached on one replica, not all of them.
    reader_affinity: bool,
    /// Replica-shed 503s from one reader within [`GATEWAY_READER_SHED_WINDOW`]
    /// that eject it (`HELIX_GATEWAY_READER_SHED_EJECT_COUNT`; 1 = eject on
    /// the first shed). Below it, a shed collection-scoped read spills to its
    /// next-ranked reader and the shedding reader keeps its other collections.
    reader_shed_eject_count: u32,
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
    /// Per reader: start of the current shed window and sheds counted in it.
    reader_sheds: Mutex<HashMap<String, (Instant, u32)>>,
    buffer: Option<Arc<GatewayDurableBuffer>>,
}

impl CloudGatewayConfig {
    fn from_env(storage_backend: StorageBackendConfig) -> Option<Self> {
        // Reader replicas must execute requests against their local DbReader.
        // Enabling the shared gateway config on a reader sends the request back
        // through the reader Service and recursively proxies it until timeout.
        if storage_backend.is_reader() {
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
        let reader_affinity = gateway_env_bool("HELIX_GATEWAY_READER_AFFINITY", true);
        let reader_shed_eject_count = gateway_env_u64(
            "HELIX_GATEWAY_READER_SHED_EJECT_COUNT",
            u64::from(GATEWAY_READER_SHED_EJECT_COUNT),
        )
        .clamp(1, u64::from(u32::MAX)) as u32;
        let buffer = gateway_buffer_config_from_env();

        Some(Self {
            writer_url,
            reader_urls,
            max_reader_attempts,
            reader_eviction,
            reader_proxy_timeout,
            writer_proxy_timeout,
            writer_is_local: storage_backend.is_writer(),
            include_writer_for_reads,
            reader_affinity,
            reader_shed_eject_count,
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
fn gateway_ejectable_reader_response(response: &Response) -> bool {
    match response.status {
        429 | 502 | 504 => true,
        // A reader's 503 is replica-level only when its blocking pool shed the
        // request (`REPLICA_SHED_HEADER`). Handlers also answer 503 for one
        // collection (lost metadata sidecar, purge pending); every replica
        // gives the same answer there, so ejecting would just move all reads
        // to the writer for the eviction window.
        503 => gateway_reader_shed_response(response),
        _ => false,
    }
}

/// A reader's blocking pool shed this request (503 + `REPLICA_SHED_HEADER`).
fn gateway_reader_shed_response(response: &Response) -> bool {
    response.status == 503
        && response.headers.iter().any(|(name, value)| {
            name.eq_ignore_ascii_case(crate::helix_gateway::async_gateway::REPLICA_SHED_HEADER)
                && value.trim() == "1"
        })
}

fn gateway_retryable_writer_status(status: u16) -> bool {
    matches!(status, 429 | 502 | 503 | 504) || status >= 500
}

#[derive(Deserialize)]
struct GatewayAffinityBody {
    collection: Option<String>,
}

/// The collection a read is scoped to, with any `_graph` suffix stripped so a
/// collection and its graph companion share one reader. `None` for
/// cluster-wide reads (`/collections`, `/v1/collections/list`, ...).
fn reader_affinity_key(request: &Request) -> Option<String> {
    let path = request.path.split('?').next().unwrap_or(&request.path);
    let name = if path.starts_with("/v1/graph/") {
        serde_json::from_slice::<GatewayAffinityBody>(&request.body)
            .ok()?
            .collection?
    } else if let Some(rest) = path.strip_prefix("/v1/collections/") {
        // Only `/v1/collections/{name}/...` is scoped; `list`, `stats`, ... are not.
        rest.split_once('/')?.0.to_string()
    } else {
        crate::helix_gateway::inflight::collection_from_path(path)?.to_string()
    };
    let base = name.strip_suffix("_graph").unwrap_or(&name);
    (!base.is_empty()).then(|| base.to_string())
}

/// Rendezvous (highest-random-weight) score of `reader` for `key`. Each key
/// ranks readers independently, so adding or ejecting one reader only moves
/// the keys that rank it first.
fn reader_rendezvous_weight(key: &str, reader: &str) -> u64 {
    use std::hash::Hasher;
    let mut hasher = twox_hash::XxHash64::with_seed(0);
    hasher.write(key.as_bytes());
    hasher.write_u8(0xFF);
    hasher.write(reader.as_bytes());
    hasher.finish()
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
            reader_sheds: Mutex::new(HashMap::new()),
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

    fn from_env(storage_backend: StorageBackendConfig) -> Option<Self> {
        CloudGatewayConfig::from_env(storage_backend).map(Self::new)
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

        let mut readers = self.healthy_reader_urls(now);
        metrics::gauge!("helix_gateway_healthy_readers").set(readers.len() as f64);
        let mut targets = Vec::new();
        if !readers.is_empty() {
            let affinity_key = if self.config.reader_affinity {
                reader_affinity_key(request)
            } else {
                None
            };
            match affinity_key {
                // Ranked over the CURRENT healthy readers: an ejected preferred
                // reader's keys fall over to their next-ranked reader.
                Some(key) => readers.sort_by_cached_key(|url| {
                    std::cmp::Reverse(reader_rendezvous_weight(&key, url))
                }),
                None => {
                    let start =
                        self.next_reader.fetch_add(1, Ordering::Relaxed) as usize % readers.len();
                    readers.rotate_left(start);
                }
            }
            targets.extend(
                readers
                    .into_iter()
                    .take(self.config.max_reader_attempts)
                    .map(|base_url| GatewayTarget {
                        kind: GatewayTargetKind::Reader,
                        base_url,
                    }),
            );
            if self.config.include_writer_for_reads {
                targets.push(self.writer_target());
            }
        }
        if targets.is_empty() {
            targets.push(self.writer_target());
        }
        Some(targets)
    }

    /// Count one replica shed from `base_url`; true once the reader shed
    /// `reader_shed_eject_count` requests inside one shed window, i.e. it is
    /// overloaded as a whole and should be ejected.
    fn record_reader_shed(&self, base_url: &str, now: Instant) -> bool {
        let mut guard = self
            .reader_sheds
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let entry = guard.entry(base_url.to_string()).or_insert((now, 0));
        if now.saturating_duration_since(entry.0) >= GATEWAY_READER_SHED_WINDOW {
            *entry = (now, 0);
        }
        entry.1 += 1;
        if entry.1 >= self.config.reader_shed_eject_count {
            guard.remove(base_url);
            return true;
        }
        false
    }

    /// The next-ranked healthy reader for a collection-scoped read that is not
    /// already in `planned`: where a shed request spills instead of ejecting
    /// the shedding reader's whole slice.
    fn spill_target(
        &self,
        request: &Request,
        planned: &[GatewayTarget],
        now: Instant,
    ) -> Option<GatewayTarget> {
        if !self.config.reader_affinity {
            return None;
        }
        let key = reader_affinity_key(request)?;
        self.healthy_reader_urls(now)
            .into_iter()
            .filter(|url| !planned.iter().any(|target| &target.base_url == url))
            .max_by_key(|url| reader_rendezvous_weight(&key, url))
            .map(|base_url| GatewayTarget {
                kind: GatewayTargetKind::Reader,
                base_url,
            })
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
        spawn_idle_gc_sweeper(&collections);
        let runtime = if config.graph_config.raft.enabled {
            Some(Arc::new(RuntimeHandle::new(
                Arc::clone(&collections),
                config.clone(),
                Arc::clone(&transport),
            )?))
        } else {
            None
        };
        let cloud_gateway = CloudGatewayRouter::from_env(config.storage_backend).map(Arc::new);
        Ok(Self {
            collections,
            config,
            runtime,
            transport,
            cloud_gateway,
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
        let Some(mut targets) = gateway.target_plan(request, Instant::now()) else {
            return Ok(None);
        };
        let class = route_class(&request.method, &request.path);

        // Indexed (not `iter`) because a shed may insert a spill reader.
        for idx in 0.. {
            let Some(target) = targets.get(idx).cloned() else {
                break;
            };
            let target = &target;
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
                    let shed = retryable_reader && gateway_reader_shed_response(&response);
                    // Below the reader-wide shed threshold the reader keeps its
                    // other collections: this request spills to the next-ranked
                    // reader when one is free, else continues down the plan.
                    let spill_only =
                        shed && !gateway.record_reader_shed(&target.base_url, Instant::now());
                    let spill = spill_only
                        .then(|| gateway.spill_target(request, &targets, Instant::now()))
                        .flatten();
                    let spilled = spill.is_some();
                    if let Some(spill) = spill {
                        targets.insert(idx + 1, spill);
                    }
                    let last_index = targets.len().saturating_sub(1);
                    if retryable_reader && idx < last_index {
                        if spill_only {
                            metrics::counter!(
                                "helix_gateway_reader_sheds_total",
                                "outcome" => if spilled { "spilled" } else { "next_target" }
                            )
                            .increment(1);
                            tracing::debug!(
                                target = %target.base_url,
                                spilled,
                                "cloud gateway reader shed request below the eject threshold; not ejecting"
                            );
                        } else if gateway_ejectable_reader_response(&response) {
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
                    if target.kind == GatewayTargetKind::Reader
                        && idx < targets.len().saturating_sub(1)
                    {
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

pub(super) fn apply_mutation(
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

/// How the error for a dense vector space's name that cannot be a sidecar
/// file's stem ends ([`refuse_unusable_vector_names`]). The gateway answers
/// it with 400, as it does a collection's name that is not one path segment.
pub(crate) const VECTOR_NAME_REFUSAL: &str =
    "is not one path segment without a dot and cannot name a vector space";

/// A dense vector space's name is the stem of its segments' sidecar files
/// under the collection's directory, joined as it comes
/// (`VectorCore::new_named_lsm_with_dir`): `../x` names files beside the
/// directory and an absolute name files anywhere. A sidecar's file is that
/// stem `with_extension`, which cuts it at its last dot, so every segment of
/// a space with a dot in its name gets one file name. Refused before
/// anything is opened or created: a refused create leaves no collection.
/// A sparse space's name is no file's.
fn refuse_unusable_vector_names(
    vectors: &HashMap<String, NamedVectorConfig>,
) -> Result<(), GraphError> {
    let unusable = |name: &&String| !is_one_path_segment(name) || name.contains('.');
    match vectors.keys().filter(unusable).min() {
        Some(name) => Err(GraphError::New(format!(
            "Vector name {name:?} {VECTOR_NAME_REFUSAL}"
        ))),
        None => Ok(()),
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
    refuse_unusable_vector_names(vectors)?;
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

        // A create that makes a dense space holds the layout change from
        // before it makes one to the end of its store. One that makes none
        // takes nothing the index job takes, unless it finds a space that is
        // not stored: then it takes the change for that store.
        let mut held = vectors
            .keys()
            .any(|vec_name| storage.named_vectors.get_config(vec_name).is_none())
            .then(|| storage.named_vectors.begin_dense_layout_change());
        let mut created_dense = false;
        for (vec_name, vector_config) in vectors {
            if let Some(existing) = storage.named_vectors.get_config(vec_name) {
                if existing != *vector_config {
                    return Err(GraphError::New(format!(
                        "Named vector '{}' already exists with a different config",
                        vec_name
                    )));
                }
                // In memory and not stored: the store of the create that
                // made it failed. This create stores it; no seal stores a
                // named vector's config.
                created_dense |= !storage
                    .metadata_snapshot()
                    .is_ok_and(|stored| stored.named_vectors.contains_key(vec_name));
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
            created_dense = true;
        }

        let mut created_sparse: Vec<&str> = Vec::new();
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
            created_sparse.push(sp_name.as_str());
        }

        storage.with_write_backend(|w| {
            // A fresh token per new space: a reader replica that cached the
            // dropped incarnation of a same-named collection must not keep
            // serving it under the initial (absent-key) epoch.
            for sp_name in &created_sparse {
                storage
                    .named_vectors
                    .with_sparse_core(sp_name, |core| core.init_cache_epoch_be(w))
                    .map_err(GraphError::from)?;
            }
            if let Some(merged) = merged_overrides.as_ref() {
                storage.set_hnsw_overrides_be(w, merged)?;
            }
            // Only when this create made a space, or found one that is not
            // stored. A create of a collection that has its spaces changes
            // no layout and stores none.
            if created_dense {
                storage.set_named_vectors_metadata_be(w, storage.named_vectors.list_vectors())?;
                let held =
                    held.get_or_insert_with(|| storage.named_vectors.begin_dense_layout_change());
                store_dense_layout(&storage, held, w, None)?;
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
                let held = storage.named_vectors.begin_dense_layout_change();
                store_dense_layout(&storage, &held, w, None)?;
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
    refuse_unusable_vector_names(vectors)?;
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

        // As in `apply_create_collection`: from before a dense space is made
        // to the end of its store.
        let mut held = vectors
            .keys()
            .any(|vec_name| storage.named_vectors.get_config(vec_name).is_none())
            .then(|| storage.named_vectors.begin_dense_layout_change());
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

        let mut created_sparse: Vec<&str> = Vec::new();
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
            created_sparse.push(sp_name.as_str());
        }

        if dense_changed || sparse_changed {
            storage.with_write_backend(|w| {
                // See apply_create_collection: new spaces get a fresh epoch.
                for sp_name in &created_sparse {
                    storage
                        .named_vectors
                        .with_sparse_core(sp_name, |core| core.init_cache_epoch_be(w))
                        .map_err(GraphError::from)?;
                }
                if dense_changed {
                    storage
                        .set_named_vectors_metadata_be(w, storage.named_vectors.list_vectors())?;
                    let held = held
                        .get_or_insert_with(|| storage.named_vectors.begin_dense_layout_change());
                    store_dense_layout(&storage, held, w, None)?;
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
    // Display / graph-coordinate fields. The client can refresh these without
    // changing the dense/sparse index content for a point.
    "start_line",
    "end_line",
    "start_col",
    "end_col",
];
const CONTENT_HASH_METADATA_KEY: &str = "metadata";
/// Subkeys of the top-level `metadata` payload value that are written by
/// the client ingest/backfill jobs but are not used as part of the index-relevant
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
    // (e.g. /Users/example/Downloads/...) and container_path embeds a
    // content-addressed slug that flips every ingest run. Both are
    // unindexed by the client's contract — display
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
/// Left out as well under [`KeyLists::Second`]: what the client writes new on a chunk
/// whose text, and so whose vectors, did not change when the file around it
/// did. `document` and `information` are one display line that names the
/// chunk's lines. A change to any of these is a node write, which rewrites
/// the node row and its payload-index rows; no dense row, sparse posting or
/// index job is derived from a payload.
const CONTENT_HASH_MOVED_CHUNK_KEYS: &[&str] = &["document", "information"];
/// The same under `metadata`: where the chunk's symbol sits, the import and
/// inheritance maps of its file, which the client repeats on every chunk of the file,
/// and the markers a client graph repair sets through `set_payload`, which does
/// not stamp the hash again.
const CONTENT_HASH_METADATA_MOVED_CHUNK_SUBKEYS: &[&str] = &[
    "symbol_start_line",
    "symbol_end_line",
    "import_map",
    "inheritance_map",
    "_graph_repair_backfilled",
    "_graph_repair_backfilled_version",
];

/// The lists of keys a content hash, or the digest a dense row keeps of its
/// payload, was made without. A stored hash and a stored digest are frozen,
/// so a list that grows is a new member here and the older members stay as
/// they are: a hash of older lists reads as a mismatch (one full write, as on
/// an image that does not have the newer lists), and a digest is kept under
/// the key of its lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeyLists {
    /// `CONTENT_HASH_PAYLOAD_ONLY_KEYS` and
    /// `CONTENT_HASH_METADATA_PAYLOAD_ONLY_SUBKEYS`.
    First,
    /// Those and the two `MOVED_CHUNK` lists.
    Second,
}

impl KeyLists {
    /// What an upsert stamps and a row is digested with.
    const CURRENT: Self = Self::Second;

    fn leaves_out(self, key: &str) -> bool {
        key == CONTENT_HASH_KEY
            || CONTENT_HASH_PAYLOAD_ONLY_KEYS.contains(&key)
            || (self == Self::Second && CONTENT_HASH_MOVED_CHUNK_KEYS.contains(&key))
    }

    fn leaves_out_of_metadata(self, key: &str) -> bool {
        CONTENT_HASH_METADATA_PAYLOAD_ONLY_SUBKEYS.contains(&key)
            || (self == Self::Second && CONTENT_HASH_METADATA_MOVED_CHUNK_SUBKEYS.contains(&key))
    }

    /// The one key of a dense row that keeps a digest made without these
    /// lists. An image that does not have the lists reads the key as another
    /// rule's (`ROW_PAYLOAD_DIGEST_FAMILY`) and leaves the row's pair as it is.
    fn row_digest_key(self) -> &'static str {
        match self {
            Self::First => "__helix_row_payload_digest__",
            Self::Second => "__helix_row_payload_digest_2__",
        }
    }
}

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
/// the client's purposes.
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
    content_hash_of(
        KeyLists::CURRENT,
        &point.payload,
        &point.vectors,
        &point.sparse_vectors,
    )
}

/// The content hash of a point with this payload and these vectors, made
/// without the keys `lists` leave out.
fn content_hash_of(
    lists: KeyLists,
    payload: &HashMap<String, Value>,
    vectors: &HashMap<String, Vec<f32>>,
    sparse_vectors: &HashMap<String, SparseVector>,
) -> u64 {
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
    let payload_without_marker = payload_covered_under(lists, payload);
    hash_payload_map(&mut hasher, &payload_without_marker);

    // Dense vectors: sort by vector name for determinism. Floats go
    // through `f32_to_quantized_bytes(q)` so embedder ULP noise
    // (common with CPU-fp32 ONNX/fastembed re-runs) doesn't flip the
    // hash when the semantic vector is effectively identical.
    let mut dense_names: Vec<&String> = vectors.keys().collect();
    dense_names.sort();
    hasher.write_u8(0xD1);
    hasher.write_u64(dense_names.len() as u64);
    for name in dense_names {
        hasher.write_u8(0xD2);
        hasher.write_u64(name.len() as u64);
        hasher.write(name.as_bytes());
        let v = &vectors[name];
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
    let mut sparse_names: Vec<&String> = sparse_vectors.keys().collect();
    sparse_names.sort();
    hasher.write_u8(0xE1);
    hasher.write_u64(sparse_names.len() as u64);
    for name in sparse_names {
        hasher.write_u8(0xE2);
        hasher.write_u64(name.len() as u64);
        hasher.write(name.as_bytes());
        let sv = &sparse_vectors[name];
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

/// Strip operational subkeys from a `metadata` Value::Object so the content
/// hash ignores fields that are refreshed on every upsert (timestamps,
/// backfill markers, derived stats). Leaves non-Object metadata values
/// untouched so we don't change hashing for collections that happen to use
/// `metadata` as a non-object field.
fn metadata_without_payload_only_subkeys(lists: KeyLists, value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let filtered: HashMap<String, Value> = map
                .iter()
                .filter(|(k, _)| !lists.leaves_out_of_metadata(k.as_str()))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            Value::Object(filtered)
        }
        other => other.clone(),
    }
}

/// `payload` on the keys the content hash covers today.
fn payload_without_content_hash_payload_only_keys(
    payload: &HashMap<String, Value>,
) -> HashMap<String, Value> {
    payload_covered_under(KeyLists::CURRENT, payload)
}

fn payload_covered_under(
    lists: KeyLists,
    payload: &HashMap<String, Value>,
) -> HashMap<String, Value> {
    payload
        .iter()
        .filter(|(k, _)| !lists.leaves_out(k.as_str()))
        .map(|(k, v)| {
            if k.as_str() == CONTENT_HASH_METADATA_KEY {
                (k.clone(), metadata_without_payload_only_subkeys(lists, v))
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

/// Whether an upsert of `point` leaves the dense rows and sparse postings of
/// its stored point as they are, and writes the node at most. `stored` is
/// the node's payload as the upsert read it and `incoming_hash` the content
/// hash of `point`: the stamp says so when it equals that hash.
///
/// A stamp of the first lists, as every point has that was last written by
/// an image without the second, never equals it for a payload with one of
/// the keys only the second leave out. Such a stamp is told by hashing the
/// stored payload with the incoming vectors as that image did. The same
/// value says three things: the stamp is of those lists; the node still has
/// the payload it was stamped with on the keys they cover (a `set_payload`
/// on one of them since leaves the stamp another payload's); and the
/// incoming vectors are the stamped ones. The two payloads are then compared
/// on the keys today's lists cover.
///
/// This reads nothing: the node is the one the upsert has. It costs a point
/// whose stamp is not the incoming hash one more content hash.
fn keeps_its_vectors(
    stored: &HashMap<String, Value>,
    point: &ReplicatedPoint,
    incoming_hash: u64,
) -> bool {
    let stamp = content_hash_from_properties(stored);
    if stamp == Some(incoming_hash) {
        return true;
    }
    let of_the_first_lists = content_hash_of(
        KeyLists::First,
        stored,
        &point.vectors,
        &point.sparse_vectors,
    );
    stamp == Some(of_the_first_lists)
        && payload_without_content_hash_payload_only_keys(stored)
            == payload_without_content_hash_payload_only_keys(&point.payload)
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

/// `HELIX_LSM_REUPSERT_DELETE=0` puts an LSM upsert back to appending only:
/// the dense rows a changed re-upsert replaces stay where they are.
fn lsm_reupsert_delete_enabled() -> bool {
    !matches!(
        std::env::var("HELIX_LSM_REUPSERT_DELETE")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref(),
        Some("0") | Some("false") | Some("off") | Some("no")
    )
}

/// Kill switch for flushing a delete's and a set_payload's commits at once,
/// and the one write of an index create on an empty collection.
/// Enabled by default: they commit BUFFERED and issue one durability barrier
/// before they return, as an upsert does. A durable commit waits for the next
/// WAL flush tick instead (`HELIX_LSM_FLUSH_INTERVAL_MS`, 500 ms in
/// production), which was 400 of the 579 ms a 64-id delete chunk took there,
/// and the whole second of an index create, which made two such commits.
/// `HELIX_LSM_FLUSH_POINT_WRITES=0` (or `false`/`off`/`no`) waits again.
pub(crate) fn lsm_point_write_flush_enabled() -> bool {
    !matches!(
        std::env::var("HELIX_LSM_FLUSH_POINT_WRITES")
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
    let reupsert_delete_on = lsm_reupsert_delete_enabled();
    let mut replaced_rows = Vec::new();
    // The rewritten ids, for the merges building from their old rows.
    let mut removal = None;

    // Group-commit (#1): when `buffered`, commit this chunk visible-but-not-yet-
    // durable; `apply_upsert_points` issues ONE `flush_durable` barrier after the
    // whole batch, so N chunks collapse to ONE durable flush and we never ack a
    // non-durable write. When disabled, commit fully durable per chunk.
    let run = |w: &mut super::backend_any::AnyWrite<'_>| -> Result<(), GraphError> {
        let mut sparse_buckets: HashMap<String, Vec<(u128, &SparseVector)>> = HashMap::new();
        let read_node = |w: &mut super::backend_any::AnyWrite<'_>,
                         id: u128|
         -> Result<Option<crate::protocol::items::Node>, GraphError> {
            storage
                .backend
                .get_for_update(w, Namespace::Nodes, &id.to_be_bytes(), |v| {
                    v.map(|bytes| SerializedNode::decode_node(bytes, id))
                })
                .map_err(|e| GraphError::New(e.to_string()))?
                .transpose()
        };
        // One multi-get reads every point's node when the chunk's ids are
        // unique; a repeated id must see its own earlier write, so per point.
        let mut seen_ids = HashSet::with_capacity(points.len());
        let mut prefetched_nodes = if points.iter().all(|point| seen_ids.insert(point.id)) {
            let keys: Vec<Vec<u8>> = points
                .iter()
                .map(|point| point.id.to_be_bytes().to_vec())
                .collect();
            let values = storage
                .backend
                .get_for_update_many(w, Namespace::Nodes, &keys)
                .map_err(|e| GraphError::New(e.to_string()))?;
            Some(
                values
                    .into_iter()
                    .zip(points)
                    .map(|(row, point)| {
                        row.map(|bytes| SerializedNode::decode_node(&bytes, point.id))
                            .transpose()
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            None
        };
        let incoming_hashes: Vec<Option<u64>> = points
            .iter()
            .map(|point| hash_skip_on.then(|| compute_content_hash(point)))
            .collect();

        // The stored points this chunk rewrites.
        let mut replaced = Vec::new();
        if reupsert_delete_on {
            for (index, point) in points.iter().enumerate() {
                let read;
                let stored = match prefetched_nodes.as_ref() {
                    Some(nodes) => nodes[index].as_ref(),
                    None => {
                        read = read_node(w, point.id)?;
                        read.as_ref()
                    }
                };
                let Some(stored) = stored.filter(|node| node.label == "point") else {
                    continue;
                };
                if let Some(incoming_hash) = incoming_hashes[index] {
                    if keeps_its_vectors(&stored.properties, point, incoming_hash) {
                        continue;
                    }
                    observe_content_hash_mismatch(
                        point,
                        Some(&stored.properties),
                        content_hash_from_properties(&stored.properties),
                        incoming_hash,
                    );
                }
                if !replaced.contains(&point.id) {
                    replaced.push(point.id);
                }
            }
        }
        // They lose their older dense rows in this batch, as on LMDB
        // (`plan_dense_delete_for_upsert`): the appends below write to the
        // mutable tail only, and a row left in another segment is what
        // search, the by-id read or a merge may pick instead. Staged ahead
        // of the appends, so its reads do not hold an appended row back from
        // its commit. `kept`: the rows left for the appends to overwrite.
        let mut kept = Vec::new();
        if !replaced.is_empty() {
            // What an append or the sparse upsert below would refuse is
            // refused first. A removal takes its row out of the writer's
            // cached ordinals when it is staged and nothing puts it back, so
            // a chunk failing later would leave a stored row out of the
            // writer's flat scan until the collection is opened again.
            for point in points {
                for (name, data) in &point.vectors {
                    storage
                        .named_vectors
                        .check_dense_append(name, data)
                        .map_err(GraphError::from)?;
                }
                for (name, sparse) in &point.sparse_vectors {
                    storage
                        .named_vectors
                        .with_sparse_core(name, |_| sparse.validate())
                        .map_err(GraphError::from)?;
                }
            }
            // Before the plan below: a merge that has written its target
            // leaves the rows there to this write.
            let begun = storage.named_vectors.merge_windows.begin_removal();
            let plan = {
                let r = storage
                    .backend
                    .begin_read()
                    .map_err(|e| GraphError::New(e.to_string()))?;
                storage
                    .named_vectors
                    .plan_delete_vectors_batch_be(&r, &replaced)
                    .map_err(GraphError::from)?
            };
            let (staged, in_tail) = storage
                .named_vectors
                .delete_replaced_vectors_be(w, &plan, &|space, id| {
                    points
                        .iter()
                        .any(|point| point.id == id && point.vectors.contains_key(space))
                })
                .map_err(GraphError::from)?;
            replaced_rows.push(staged);
            kept = in_tail;
            removal = Some((begun, replaced));
        }
        #[cfg(test)]
        test_support::reached("upsert_prepared");
        // Counter bumps fold into one merge each after the loop: every per-point
        // bump re-read the counter key from SlateDB (merges are not pending reads).
        let mut created_nodes = 0i64;
        let mut created_vectors = 0i64;
        for (index, point) in points.iter().enumerate() {
            let existing_node = match prefetched_nodes.as_mut() {
                Some(nodes) => nodes[index].take(),
                None => read_node(w, point.id)?,
            };
            let existing_is_point = existing_node
                .as_ref()
                .map(|node| node.label == "point")
                .unwrap_or(false);

            let incoming_hash = incoming_hashes[index];
            let mut payload_only_update = false;
            // The same call that planned `replaced` above, so the two agree
            // on which points keep their rows.
            if let (Some(stored), Some(incoming_hash)) = (
                existing_node.as_ref().filter(|_| existing_is_point),
                incoming_hash,
            ) {
                let stored = &stored.properties;
                if keeps_its_vectors(stored, point, incoming_hash) {
                    // A node under an older stamp is written for today's.
                    if content_hash_from_properties(stored) == Some(incoming_hash)
                        && payloads_equal_without_content_hash_marker(stored, &point.payload)
                    {
                        metrics::counter!("helix_upsert_content_hash_skipped_total").increment(1);
                        continue;
                    }
                    payload_only_update = true;
                    metrics::counter!("helix_upsert_content_hash_payload_only_update_total")
                        .increment(1);
                }
            }

            let mut props = point.payload.clone();
            if let Some(h) = incoming_hash {
                props.insert(CONTENT_HASH_KEY.to_string(), Value::U64(h));
            }
            let node_upsert = NodeUpsert {
                id: point.id,
                label: "point".into(),
                properties: props,
            };
            if existing_node.is_none() {
                created_nodes += 1;
            }
            storage.upsert_node_be_with_existing_uncounted(
                w,
                &node_upsert,
                existing_node.clone(),
            )?;

            if payload_only_update {
                continue;
            }

            // The payload or its digest; the hash stamped on the node is in neither.
            let fields = dense_row_fields(&point.payload);
            for (vec_name, vec_data) in &point.vectors {
                let tail = storage
                    .named_vectors
                    .dense_append_to_mutable_be(w, vec_name, vec_data, point.id, fields.clone())
                    .map_err(GraphError::from)?;
                // Overwritten, unless the tail was sealed since the row was kept.
                kept.retain(|(space, kept_in, id)| {
                    *id != point.id || space != vec_name || *kept_in != tail
                });
                *dense_batch_counts.entry(vec_name.clone()).or_default() += 1;
            }

            for (sp_name, sp_vec) in &point.sparse_vectors {
                sparse_buckets
                    .entry(sp_name.clone())
                    .or_default()
                    .push((point.id, sp_vec));
            }

            if !existing_is_point {
                created_vectors += 1;
            }
        }
        if created_nodes > 0 {
            storage.adjust_metadata_counter_be(w, MetadataCounter::Nodes, created_nodes)?;
        }
        if created_vectors > 0 {
            storage.adjust_metadata_counter_be(w, MetadataCounter::Vectors, created_vectors)?;
        }

        // A tail sealed in between: the append went to the next one, and the
        // row kept for it is an older copy like the others.
        for (_, sealed_tail, id) in kept {
            replaced_rows.push(
                storage
                    .named_vectors
                    .delete_from_segment_be(w, &sealed_tail, &[id])
                    .map_err(GraphError::from)?,
            );
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
    for staged in &replaced_rows {
        storage.named_vectors.apply_delete_tombstones(staged);
    }
    if let Some((removal, replaced)) = removal {
        removal.committed(&replaced);
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
/// `apply_upsert_points_in_txn` under ~250 ms for typical client workloads
/// (1536-dim vectors + payload index puts).
fn upsert_apply_chunk_size() -> usize {
    env_usize_at_least("HELIX_UPSERT_APPLY_CHUNK", 1).unwrap_or(64)
}

/// Cap on total sparse posting count per upsert chunk. Client workloads write
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
/// `HELIX_UPSERT_WRITE_BUDGET`. Default 32K is intentionally above normal client
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

pub(crate) fn lock_point_mutation_gate<'a>(
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
    storage.point_mutations.fetch_add(1, Ordering::AcqRel);
    let _in_flight = PointMutationInFlight::begin(&storage);
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
    storage.point_mutations.fetch_add(1, Ordering::AcqRel);
    // A delete is a write too: the tail of a collection still being reindexed
    // is not sealed between a file's delete and its upsert.
    let _in_flight = PointMutationInFlight::begin(&storage);

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
    // Chunks commit visible but not yet durable; the barrier after the loop
    // makes them durable before the delete is acknowledged.
    let flush_at_once =
        storage.backend.kind() == BackendKind::Lsm && lsm_point_write_flush_enabled();
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
            // Before the plan below: a merge that has written its target
            // leaves the rows there to this delete.
            let removal = storage.named_vectors.merge_windows.begin_removal();
            let r = storage
                .backend
                .begin_read()
                .map_err(|e| GraphError::New(e.to_string()))?;
            // One multi-get reads the chunk's node rows, not a round trip per id.
            let node_keys: Vec<Vec<u8>> = delete_ids
                .iter()
                .map(|id| id.to_be_bytes().to_vec())
                .collect();
            let node_rows = storage
                .backend
                .get_many(&r, Namespace::Nodes, &node_keys)
                .map_err(|e| GraphError::New(e.to_string()))?;
            let mut existing_point_ids = HashSet::with_capacity(delete_ids.len());
            for (id, row) in delete_ids.iter().zip(node_rows) {
                let existing_is_point = row
                    .map(|bytes| SerializedNode::decode_node(&bytes, *id))
                    .transpose()?
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
            let delete = |w: &mut super::backend_any::AnyWrite<'_>| -> Result<(), GraphError> {
                let t = metrics_enabled.then(Instant::now);
                staged_tombstones = Some(
                    storage
                        .named_vectors
                        .delete_vectors_batch_with_plan_be(w, &dense_delete_plan)
                        .map_err(GraphError::from)?,
                );
                record_delete_step("lsm_dense_delete", t);
                // One batched sparse delete per chunk: per-id deletes cost one
                // sequential SlateDB read per point plus one per distinct term
                // in the chunk.
                let t = metrics_enabled.then(Instant::now);
                storage
                    .named_vectors
                    .delete_sparse_vectors_be(w, &delete_ids)
                    .map_err(GraphError::from)?;
                record_delete_step("lsm_sparse_delete", t);
                let t = metrics_enabled.then(Instant::now);
                storage.drop_nodes_be(w, &delete_ids)?;
                record_delete_step("lsm_drop_nodes", t);
                if !existing_point_ids.is_empty() {
                    storage.adjust_metadata_counter_be(
                        w,
                        MetadataCounter::Vectors,
                        -(existing_point_ids.len() as i64),
                    )?;
                }
                Ok(())
            };
            if flush_at_once {
                storage.with_write_backend_buffered(delete)?;
            } else {
                storage.with_write_backend(delete)?;
            }
            if let Some(staged) = staged_tombstones {
                storage.named_vectors.apply_delete_tombstones(&staged);
            }
            // After the tombstones are in memory: a merge waiting for this
            // delete reads its sources next.
            removal.committed(&delete_ids);
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

    if flush_at_once && !ids.is_empty() {
        let t = metrics_enabled.then(Instant::now);
        storage
            .backend
            .flush_durable()
            .map_err(|e| storage.backend_error(e, "delete_points_flush_durable"))?;
        record_delete_step("lsm_flush_durable", t);
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

/// Optimizer steps for tests outside this module, run on the calling thread.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    thread_local! {
        /// `(stage, action)` for the next merge or upsert on this thread.
        static HOOK: std::cell::RefCell<Option<(&'static str, Box<dyn FnOnce()>)>> =
            const { std::cell::RefCell::new(None) };
        /// `(stage, action)` run every time this thread reaches `stage`.
        static HOOK_EACH_TIME: std::cell::RefCell<Option<(&'static str, Box<dyn FnMut()>)>> =
            const { std::cell::RefCell::new(None) };
    }

    /// Called at `stage`. A merge, none of them published: "exported" once it
    /// has read its sources, "flat_written" once its target has its flat rows
    /// and no graph yet, "written" once the target is whole, "drained" once
    /// it has taken the removed ids out of the target and closed its window.
    /// An LSM upsert chunk: "upsert_prepared" once it has read its points'
    /// nodes and before it appends. A regraph: "regraph_exported" once it has
    /// read a segment's rows and before it takes the gate again. A store of
    /// an LSM collection's dense layout: "layout_read" inside its
    /// transaction, once it has read the layout from memory. A seal of an
    /// LSM tail: "seal_prepared" once its next tail is open and before the
    /// layout with the seal in it is stored, "seal_midway" once that is
    /// stored and before memory has the seal.
    pub(super) fn reached(stage: &str) {
        let hook = HOOK.with(|hook| {
            let mut hook = hook.borrow_mut();
            if hook.as_ref().map(|(at, _)| *at) == Some(stage) {
                hook.take()
            } else {
                None
            }
        });
        if let Some((_, action)) = hook {
            action();
        }
        // Out of its place while it runs: a stage reached inside it finds none.
        let each_time = HOOK_EACH_TIME.with(|hook| {
            let mut hook = hook.borrow_mut();
            if hook.as_ref().map(|(at, _)| *at) == Some(stage) {
                hook.take()
            } else {
                None
            }
        });
        if let Some((at, mut action)) = each_time {
            action();
            HOOK_EACH_TIME.with(|hook| *hook.borrow_mut() = Some((at, action)));
        }
    }

    /// Seals `space`'s mutable tail: flat rows only, until its index is built.
    pub(crate) fn seal_tail(storage: &Arc<HelixGraphStorage>, space: &str) {
        let hnsw = HNSWConfig::new(Some(16), Some(64), Some(128));
        assert!(seal_lsm_dense_tail(storage, space, &hnsw).unwrap());
    }

    /// Seals `space`'s mutable tail and builds its index.
    pub(crate) fn seal_and_index(storage: &Arc<HelixGraphStorage>, space: &str) {
        seal_tail(storage, space);
        let prepared = {
            let read = storage.backend.begin_read().unwrap();
            storage
                .named_vectors
                .prepare_building_indices_limited_be(&read, space, 1)
                .unwrap()
        };
        assert_eq!(prepared.len(), 1);
        run_chunked_indices_publish(storage, space, prepared).unwrap();
    }

    /// One optimizer job over `space`.
    pub(crate) fn optimize(storage: &Arc<HelixGraphStorage>, collection: &str, space: &str) {
        run_index_job(IndexJob {
            collection: collection.into(),
            storage: Arc::downgrade(storage),
            spaces_to_check: vec![space.into()],
            hnsw_config: HNSWConfig::new(Some(16), Some(64), Some(128)),
            flat_scan_threshold: 1_000,
        });
    }

    /// One regraph of `space`, with no other writer; what it reports.
    pub(crate) fn regraph(
        storage: &Arc<HelixGraphStorage>,
        collection: &str,
        space: &str,
    ) -> RegraphStatus {
        let hnsw = HNSWConfig::new(Some(16), Some(64), Some(128));
        let gate = Arc::new(Mutex::new(()));
        regraph_collection(storage, collection, &[space.to_string()], &hnsw, &gate).unwrap()
    }

    /// `run`, with `action` run inside it at `stage` (see `reached`).
    pub(crate) fn with_hook(
        stage: &'static str,
        action: impl FnOnce() + 'static,
        run: impl FnOnce(),
    ) {
        let action: Box<dyn FnOnce()> = Box::new(action);
        HOOK.with(|hook| *hook.borrow_mut() = Some((stage, action)));
        run();
        assert!(
            HOOK.with(|hook| hook.borrow_mut().take()).is_none(),
            "nothing reached {stage}"
        );
    }

    /// `run`, with `action` run inside it every time `stage` is reached.
    pub(crate) fn with_hook_each_time(
        stage: &'static str,
        action: impl FnMut() + 'static,
        run: impl FnOnce(),
    ) {
        let action: Box<dyn FnMut()> = Box::new(action);
        HOOK_EACH_TIME.with(|hook| *hook.borrow_mut() = Some((stage, action)));
        run();
        assert!(
            HOOK_EACH_TIME
                .with(|hook| hook.borrow_mut().take())
                .is_some(),
            "the hook at {stage} was taken and not put back"
        );
    }

    /// Whether a dense row's `fields` are the digest of a payload, not a payload.
    pub(crate) fn row_keeps_a_digest(fields: &HashMap<String, Value>) -> bool {
        matches!(row_keeps(fields), RowKeeps::Digest(..))
    }

    /// The content hash an image that had only the first lists stamped on
    /// `point`.
    pub(crate) fn stamp_of_the_first_lists(point: &ReplicatedPoint) -> u64 {
        content_hash_of(
            KeyLists::First,
            &point.payload,
            &point.vectors,
            &point.sparse_vectors,
        )
    }

    /// `optimize`, with `action` run inside its merge at `stage`.
    pub(crate) fn optimize_with(
        storage: &Arc<HelixGraphStorage>,
        collection: &str,
        space: &str,
        stage: &'static str,
        action: impl FnOnce() + 'static,
    ) {
        with_hook(stage, action, || optimize(storage, collection, space));
    }
}

#[cfg(test)]
mod tests {
    mod backlog_sweeper;
    mod bounded_optimizer_reads;
    mod dead_segment_names;
    mod empty_index_create;
    mod empty_merge;
    mod fenced_index_store;
    mod optimizer_followups;
    mod publish_crash;
    mod regraph_retire;
    mod reupsert_copy_choice;
    mod reupsert_stamps;
    mod seal_store_order;
    mod stored_row_wins;
    mod tail_seal_debounce;
    mod vector_space_names;
    mod worker_panics;

    use super::*;
    use crate::helix_engine::graph_core::config::{
        Config, GraphConfig, RaftConfig, RaftPeerConfig, VectorConfig,
    };
    use crate::helix_engine::storage_core::backend::{LsmRole, LsmStorage};
    use crate::helix_engine::storage_core::backend_any::AnyBackend;
    use crate::helix_engine::storage_core::storage_methods::StorageMethods;
    use std::sync::Weak;
    use tempfile::TempDir;

    /// The lock shared by the tests that set the process environment. `lock`
    /// hands the guard out of a poisoned lock too: a test that panics while
    /// it holds the lock puts the environment back as its guards drop, and
    /// is one failure. On a plain mutex every test after it failed as well,
    /// on the lock (as `ENV_TEST_LOCK` of the collection manager's tests).
    struct EnvLock(Mutex<()>);

    impl EnvLock {
        fn lock(&self) -> std::sync::LockResult<std::sync::MutexGuard<'_, ()>> {
            Ok(self
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner))
        }
    }

    static ENV_LOCK: EnvLock = EnvLock(Mutex::new(()));

    #[test]
    fn env_lock_is_handed_out_after_a_test_panicked_holding_it() {
        let lock = EnvLock(Mutex::new(()));
        let panicked = thread::scope(|scope| {
            scope
                .spawn(|| {
                    let _held = lock.lock().unwrap();
                    panic!("a test that fails while it holds the lock");
                })
                .join()
        });
        assert!(panicked.is_err() && lock.0.is_poisoned());
        drop(lock.lock().unwrap());
    }

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
            reader_affinity: false,
            reader_shed_eject_count: GATEWAY_READER_SHED_EJECT_COUNT,
            buffer: None,
        })
    }

    fn lsm_reader_backend_config_for_tests() -> StorageBackendConfig {
        StorageBackendConfig::Lsm {
            storage: LsmStorage::InMemory,
            role: LsmRole::Reader,
        }
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
        // What a reader answers for a collection it could not open now.
        assert!(gateway_retryable_reader_status(503));
        assert!(gateway_retryable_reader_status(504));
        assert!(!gateway_retryable_reader_status(400));
    }

    #[test]
    fn cloud_gateway_ejects_readers_only_for_replica_scoped_statuses() {
        let status = |code: u16| Response {
            status: code,
            ..Response::new()
        };
        assert!(gateway_ejectable_reader_response(&status(429)));
        assert!(gateway_ejectable_reader_response(&status(502)));
        assert!(gateway_ejectable_reader_response(&status(504)));
        // Collection-scoped: identical on every replica, must not drain the
        // healthy-reader pool.
        assert!(!gateway_ejectable_reader_response(&status(404)));
        assert!(!gateway_ejectable_reader_response(&status(500)));
        // A bare 503 is a handler's collection-scoped answer (lost metadata
        // sidecar, purge pending): on 2026-10-20 one such collection per
        // storage-metering sweep ejected both prod readers every ~2 minutes.
        assert!(!gateway_ejectable_reader_response(&status(503)));
        // Only a blocking-pool shed marks the 503 as replica-level.
        let mut shed = status(503);
        shed.headers
            .insert("x-helix-replica-shed".to_string(), "1".to_string());
        assert!(gateway_ejectable_reader_response(&shed));
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

    /// The sparse statistics route is a POST, and the gateway still plans it
    /// as the read it is: readers, never the writer while one is healthy.
    #[test]
    fn cloud_gateway_routes_sparse_stats_to_readers() {
        let router = gateway_router_for_tests(false);
        let request = gateway_request("POST", "/collections/repo/sparse/stats");

        let plan = router.target_plan(&request, Instant::now()).unwrap();

        assert_eq!(
            plan,
            vec![reader_target("reader-a"), reader_target("reader-b")]
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

    fn affinity_router_for_tests(readers: usize) -> CloudGatewayRouter {
        let mut router = gateway_router_for_tests(false);
        router.config.reader_affinity = true;
        router.config.reader_urls = (0..readers)
            .map(|i| format!("http://reader-{i}:8080"))
            .collect();
        router
    }

    fn search_request(collection: &str) -> Request {
        gateway_request("POST", &format!("/collections/{collection}/points/search"))
    }

    fn first_reader(router: &CloudGatewayRouter, request: &Request, now: Instant) -> String {
        router.target_plan(request, now).unwrap()[0]
            .base_url
            .clone()
    }

    fn synthetic_collection(i: usize) -> String {
        format!(
            "tenant{}_repo-{:08x}",
            i % 97,
            i.wrapping_mul(2_654_435_761)
        )
    }

    #[test]
    fn cloud_gateway_affinity_maps_each_collection_to_one_reader_deterministically() {
        let router = affinity_router_for_tests(3);
        // A restarted writer listing the same readers in another order.
        let mut restarted = affinity_router_for_tests(3);
        restarted.config.reader_urls.reverse();
        let now = Instant::now();
        for i in 0..50 {
            let request = search_request(&synthetic_collection(i));
            let plan = router.target_plan(&request, now).unwrap();
            assert_eq!(plan.len(), 3);
            assert_eq!(router.target_plan(&request, now).unwrap(), plan);
            assert_eq!(restarted.target_plan(&request, now).unwrap(), plan);
        }

        let mut bypass = search_request("repo");
        bypass.headers.insert(
            GATEWAY_BYPASS_HEADER.to_string(),
            GATEWAY_BYPASS_VALUE.to_string(),
        );
        assert!(router.target_plan(&bypass, now).is_none());
    }

    #[test]
    fn cloud_gateway_affinity_colocates_graph_companion_with_its_collection() {
        let router = affinity_router_for_tests(3);
        let now = Instant::now();
        for i in 0..50 {
            let base = synthetic_collection(i);
            let expected = first_reader(&router, &search_request(&base), now);
            let mut graph_api = gateway_request("POST", "/v1/graph/callers");
            graph_api.body =
                format!(r#"{{"collection":"{base}_graph","symbol":"main"}}"#).into_bytes();
            for request in [
                search_request(&format!("{base}_graph")),
                gateway_request(
                    "POST",
                    &format!("/collections/{base}_graph/points/scroll?limit=1"),
                ),
                gateway_request("GET", &format!("/collections/{base}")),
                gateway_request("POST", &format!("/v1/collections/{base}/points/scan")),
                graph_api,
            ] {
                assert_eq!(
                    first_reader(&router, &request, now),
                    expected,
                    "{}",
                    request.path
                );
            }
        }
    }

    #[test]
    fn cloud_gateway_affinity_fails_over_in_rendezvous_order_when_preferred_reader_is_ejected() {
        let router = affinity_router_for_tests(3);
        let now = Instant::now();
        let request = search_request("example-code-collection");
        let plan = router.target_plan(&request, now).unwrap();

        router.mark_reader_unhealthy(&plan[0].base_url, now);
        assert_eq!(
            router.target_plan(&request, now).unwrap(),
            plan[1..].to_vec()
        );

        // The preferred reader takes the collection back once its ejection lapses.
        let later = now + router.config.reader_eviction + Duration::from_millis(1);
        assert_eq!(router.target_plan(&request, later).unwrap(), plan);

        // Prod shape: one reader attempt, then the writer.
        let mut prod = affinity_router_for_tests(2);
        prod.config.max_reader_attempts = 1;
        prod.config.include_writer_for_reads = true;
        let plan = prod.target_plan(&request, now).unwrap();
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[1], writer_target());
        prod.mark_reader_unhealthy(&plan[0].base_url, now);
        let failover = prod.target_plan(&request, now).unwrap();
        assert_eq!(failover[0].kind, GatewayTargetKind::Reader);
        assert_ne!(failover[0], plan[0]);
        assert_eq!(failover[1], writer_target());
    }

    #[test]
    fn cloud_gateway_ejects_a_shedding_reader_only_past_the_reader_wide_threshold() {
        let mut router = affinity_router_for_tests(2);
        router.config.reader_shed_eject_count = 3;
        let now = Instant::now();
        let reader = "http://reader-0:8080";
        assert!(!router.record_reader_shed(reader, now));
        assert!(!router.record_reader_shed(reader, now));
        assert!(router.record_reader_shed(reader, now));
        // The count restarts after an ejection decision and after the window.
        assert!(!router.record_reader_shed(reader, now));
        assert!(!router.record_reader_shed(reader, now));
        let later = now + GATEWAY_READER_SHED_WINDOW;
        assert!(!router.record_reader_shed(reader, later));
        assert!(!router.record_reader_shed(reader, later));
        // Other readers keep their own counts.
        assert!(!router.record_reader_shed("http://reader-1:8080", later));

        router.config.reader_shed_eject_count = 1;
        assert!(
            router.record_reader_shed(reader, later),
            "1 restores eject-on-first-shed"
        );
    }

    #[test]
    fn cloud_gateway_spill_target_is_the_next_ranked_unplanned_reader() {
        let mut router = affinity_router_for_tests(3);
        let now = Instant::now();
        let request = search_request("example-code-collection");
        let ranked = router.target_plan(&request, now).unwrap();

        router.config.max_reader_attempts = 1;
        router.config.include_writer_for_reads = true;
        let plan = router.target_plan(&request, now).unwrap();
        assert_eq!(plan, vec![ranked[0].clone(), writer_target()]);
        assert_eq!(
            router.spill_target(&request, &plan, now),
            Some(ranked[1].clone())
        );
        let spilled = vec![ranked[0].clone(), ranked[1].clone(), writer_target()];
        assert_eq!(
            router.spill_target(&request, &spilled, now),
            Some(ranked[2].clone())
        );

        router.mark_reader_unhealthy(&ranked[2].base_url, now);
        assert_eq!(router.spill_target(&request, &spilled, now), None);
        // Unscoped reads and round-robin routing never spill.
        assert_eq!(
            router.spill_target(&gateway_request("GET", "/collections"), &plan, now),
            None
        );
        router.config.reader_affinity = false;
        assert_eq!(router.spill_target(&request, &plan, now), None);
    }

    /// Minimal HTTP/1.1 upstream answering every request with `status`, an
    /// optional replica-shed header, and `body`. Returns its base URL and hit count.
    fn fake_upstream(status: u16, shed: bool, body: &'static str) -> (String, Arc<AtomicU64>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let hits = Arc::new(AtomicU64::new(0));
        let counter = Arc::clone(&hits);
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
                    match stream.read(&mut chunk) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => buf.extend_from_slice(&chunk[..n]),
                    }
                }
                counter.fetch_add(1, Ordering::SeqCst);
                let shed_header = if shed {
                    "X-Helix-Replica-Shed: 1\r\n"
                } else {
                    ""
                };
                let reply = format!(
                    "HTTP/1.1 {status} X\r\n{shed_header}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(reply.as_bytes());
            }
        });
        (url, hits)
    }

    /// A shed collection-scoped read is retried on the next-ranked reader
    /// without ejecting the shedding reader; eject count 1 keeps the old
    /// eject-and-fall-back-to-writer behavior.
    #[test]
    fn cloud_gateway_spills_a_shed_read_to_the_next_reader_without_ejecting() {
        let (shedding, shed_hits) = fake_upstream(503, true, "shed");
        let (healthy, healthy_hits) = fake_upstream(200, false, "from-reader");
        let (writer, writer_hits) = fake_upstream(200, false, "from-writer");
        let temp_dir = TempDir::new().unwrap();
        let config = Config::new(8, 32, 64, 1).with_lsm_in_memory();
        let collections =
            Arc::new(CollectionManager::new(temp_dir.path().join("data"), config.clone()).unwrap());
        let mut router = affinity_router_for_tests(0);
        router.config.reader_urls = vec![shedding.clone(), healthy.clone()];
        router.config.writer_url = writer.clone();
        router.config.max_reader_attempts = 1;
        router.config.include_writer_for_reads = true;
        let now = Instant::now();
        let request = (0..200)
            .map(|i| search_request(&synthetic_collection(i)))
            .find(|request| first_reader(&router, request, now) == shedding)
            .expect("some collection prefers the shedding reader");
        let manager = ReplicationManager {
            collections,
            config,
            runtime: None,
            transport: Arc::new(NoopRaftTransport),
            cloud_gateway: Some(Arc::new(router)),
        };
        let gateway = manager.cloud_gateway.as_ref().unwrap();

        let response = manager
            .prepare_cloud_gateway_request(&request)
            .unwrap()
            .unwrap();
        assert_eq!(response.status, 200);
        assert_eq!(response.body, b"from-reader".to_vec());
        assert_eq!(shed_hits.load(Ordering::SeqCst), 1);
        assert_eq!(healthy_hits.load(Ordering::SeqCst), 1);
        assert_eq!(writer_hits.load(Ordering::SeqCst), 0);
        assert!(
            gateway
                .healthy_reader_urls(Instant::now())
                .contains(&shedding),
            "one shed must not eject the reader's whole slice"
        );

        let mut eager = gateway.config.clone();
        eager.reader_shed_eject_count = 1;
        let manager = ReplicationManager {
            cloud_gateway: Some(Arc::new(CloudGatewayRouter::new(eager))),
            ..manager
        };
        let gateway = manager.cloud_gateway.as_ref().unwrap();
        let response = manager
            .prepare_cloud_gateway_request(&request)
            .unwrap()
            .unwrap();
        assert_eq!(response.body, b"from-writer".to_vec());
        assert!(!gateway
            .healthy_reader_urls(Instant::now())
            .contains(&shedding));
        assert_eq!(healthy_hits.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cloud_gateway_affinity_spreads_collections_near_evenly() {
        let mut router = affinity_router_for_tests(2);
        router.config.reader_urls = (0..2)
            .map(|i| format!("http://reader-{i}.readers.example.test:6969"))
            .collect();
        let now = Instant::now();
        let on_first = (0..1000)
            .filter(|i| {
                first_reader(&router, &search_request(&synthetic_collection(*i)), now)
                    == router.config.reader_urls[0]
            })
            .count();
        // Binomial(1000, 0.5) has sigma ~16; this guards against a degenerate hash.
        assert!(
            (450..=550).contains(&on_first),
            "reader-0 owns {on_first}/1000"
        );
    }

    #[test]
    fn cloud_gateway_affinity_adding_a_reader_moves_only_its_share() {
        let now = Instant::now();
        for n in 2..=4 {
            let before = affinity_router_for_tests(n);
            let after = affinity_router_for_tests(n + 1);
            let new_reader = format!("http://reader-{n}:8080");
            let mut moved = 0usize;
            for i in 0..1000 {
                let request = search_request(&synthetic_collection(i));
                let old = first_reader(&before, &request, now);
                let new = first_reader(&after, &request, now);
                if old != new {
                    // Existing readers never trade collections among themselves.
                    assert_eq!(new, new_reader);
                    moved += 1;
                }
            }
            let expected = 1000 / (n + 1);
            assert!(
                moved.abs_diff(expected) <= 60,
                "{n}->{} readers moved {moved}/1000, expected ~{expected}",
                n + 1
            );
        }
    }

    #[test]
    fn cloud_gateway_affinity_keeps_unscoped_reads_round_robin() {
        let router = affinity_router_for_tests(2);
        let now = Instant::now();
        let mut graph_without_collection = gateway_request("POST", "/v1/graph/callers");
        graph_without_collection.body = br#"{"symbol":"main"}"#.to_vec();
        for request in [
            gateway_request("GET", "/collections"),
            gateway_request("POST", "/v1/collections/list"),
            gateway_request("POST", "/v1/collections/stats"),
            graph_without_collection,
        ] {
            assert_eq!(reader_affinity_key(&request), None, "{}", request.path);
            let first = first_reader(&router, &request, now);
            assert_ne!(
                first_reader(&router, &request, now),
                first,
                "{}",
                request.path
            );
        }
    }

    #[test]
    fn cloud_gateway_reader_affinity_defaults_on_and_zero_disables_it() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _writer_url = EnvVarGuard::set("HELIX_GATEWAY_WRITER_URL", "http://writer:8080");
        let writer = StorageBackendConfig::Lsm {
            storage: LsmStorage::InMemory,
            role: LsmRole::Writer,
        };
        {
            let _unset = EnvVarGuard::unset("HELIX_GATEWAY_READER_AFFINITY");
            assert!(
                CloudGatewayConfig::from_env(writer)
                    .unwrap()
                    .reader_affinity
            );
        }
        let _off = EnvVarGuard::set("HELIX_GATEWAY_READER_AFFINITY", "0");
        assert!(
            !CloudGatewayConfig::from_env(writer)
                .unwrap()
                .reader_affinity
        );
    }

    #[test]
    fn cloud_gateway_is_disabled_inside_reader_replicas() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _writer_url = EnvVarGuard::set("HELIX_GATEWAY_WRITER_URL", "http://writer:8080");

        assert!(CloudGatewayConfig::from_env(lsm_reader_backend_config_for_tests()).is_none());
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
            reader_affinity: false,
            reader_shed_eject_count: GATEWAY_READER_SHED_EJECT_COUNT,
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

        run_chunked_merge_publish(&storage, TEST_DENSE_VECTOR, &hnsw_config, merge, None).unwrap();

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
    fn lsm_create_collection_writes_fresh_sparse_cache_epoch() {
        let temp_dir = TempDir::new().unwrap();
        let config = Config::new(8, 32, 64, 1).with_lsm_in_memory();
        let collections =
            Arc::new(CollectionManager::new(temp_dir.path().join("data"), config.clone()).unwrap());
        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::CreateCollection {
                name: "repo".into(),
                vectors: HashMap::new(),
                sparse_vectors: HashMap::from([(
                    "lex_sparse".into(),
                    SparseVectorConfig::default(),
                )]),
                hnsw_overrides: None,
            },
        )
        .unwrap();
        let storage = collections.get_collection("repo").unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);
        let token = storage
            .named_vectors
            .with_sparse_core("lex_sparse", |core| Ok(core.cache_epoch_token_for_test()))
            .unwrap();
        assert!(
            matches!(token, Some(t) if t != 0),
            "a new sparse space must not read as the initial epoch, got {token:?}"
        );
    }

    #[test]
    fn lsm_update_collection_writes_fresh_epoch_for_added_sparse_space() {
        let temp_dir = TempDir::new().unwrap();
        let config = Config::new(8, 32, 64, 1).with_lsm_in_memory();
        let collections =
            Arc::new(CollectionManager::new(temp_dir.path().join("data"), config.clone()).unwrap());
        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::CreateCollection {
                name: "repo".into(),
                vectors: HashMap::new(),
                sparse_vectors: HashMap::new(),
                hnsw_overrides: None,
            },
        )
        .unwrap();
        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::UpdateCollection {
                name: "repo".into(),
                vectors: HashMap::new(),
                sparse_vectors: HashMap::from([(
                    "lex_sparse".into(),
                    SparseVectorConfig::default(),
                )]),
            },
        )
        .unwrap();
        let storage = collections.get_collection("repo").unwrap();
        let token = storage
            .named_vectors
            .with_sparse_core("lex_sparse", |core| Ok(core.cache_epoch_token_for_test()))
            .unwrap();
        assert!(
            matches!(token, Some(t) if t != 0),
            "an added sparse space must not read as the initial epoch, got {token:?}"
        );
    }

    #[test]
    fn chunked_merge_publish_tq_attaches_ordinals_on_lsm_backend() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _format = EnvVarGuard::set("HELIX_SEGMENT_FORMAT", "tq");
        let spindle =
            crate::helix_engine::vector_core::spindle::SpindleConfig::turbo_prod_compact(4);
        let temp_dir = TempDir::new().unwrap();
        let config = Config::new(8, 32, 64, 1).with_lsm_in_memory();
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

        run_chunked_merge_publish(&storage, TEST_DENSE_VECTOR, &hnsw_config, merge, None).unwrap();

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
        let vector_row = storage
            .backend
            .get_with(
                &r,
                Namespace::Segment {
                    physical_name: TEST_MERGE_TARGET,
                    db: crate::helix_engine::storage_core::backend::SegmentDb::Vectors,
                },
                vector_key_for_test(401, 0).as_slice(),
                |opt| opt.map(|b| b.to_vec()),
            )
            .unwrap()
            .unwrap();
        assert!(
            vector_row.is_empty(),
            "HVTQ LSM publish must store a marker instead of duplicate vector bytes"
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

        run_chunked_merge_publish(&storage, TEST_DENSE_VECTOR, &hnsw_config, merge, None).unwrap();

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
                unreached: 0,
            },
        };

        let err = run_chunked_merge_publish(&storage, TEST_DENSE_VECTOR, &hnsw_config, merge, None)
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
        // Every active segment Indexed: a Building one is submitted by itself.
        let below_cap_gate = DenseSegmentDebt {
            indexed_segments: cap.saturating_sub(1),
            active_segments: cap.saturating_sub(1),
            ..gate_only
        };
        let cap_hit_gate = DenseSegmentDebt {
            indexed_segments: cap,
            active_segments: cap,
            ..gate_only
        };
        let merge_backlog = DenseSegmentDebt {
            indexed_segments: 3,
            active_segments: 3,
            gate_debt: 0,
            ..gate_only
        };
        let building_only = DenseSegmentDebt {
            active_segments: 1,
            gate_debt: 0,
            ..gate_only
        };

        assert!(!optimizer_debt_has_submit_work(gate_only, 1, true, false));
        assert!(!optimizer_debt_has_submit_work(
            below_cap_gate,
            1,
            false,
            false
        ));
        // The gate term does not depend on a space being over the merge target.
        assert!(optimizer_debt_has_submit_work(
            cap_hit_gate,
            1,
            false,
            false
        ));
        assert!(optimizer_debt_has_submit_work(
            building_only,
            1,
            false,
            true
        ));
        assert!(!optimizer_debt_has_submit_work(
            building_only,
            1,
            false,
            false
        ));
        assert!(optimizer_debt_has_submit_work(
            merge_backlog,
            1,
            true,
            false
        ));
        // Segments summed over spaces exceed the ratio, but no space has a merge.
        assert!(!optimizer_debt_has_submit_work(
            merge_backlog,
            1,
            false,
            false
        ));
        // A space over the target, still under the collection-wide ratio.
        assert!(!optimizer_debt_has_submit_work(
            merge_backlog,
            3,
            true,
            false
        ));
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

    /// With memory to spare a sweep looks down its whole list for candidates
    /// past their cooldown. Under pressure it has one place and looks at one
    /// candidate: while that one is in its cooldown the sweep submits
    /// nothing, where looking further would submit another collection's job
    /// at every sweep.
    #[test]
    fn optimizer_backlog_sweep_under_pressure_looks_no_further_than_its_places() {
        let looked_at = optimizer_backlog_candidates_looked_at;
        assert_eq!(looked_at(32, MaintenanceAdmission::Admitted), usize::MAX);
        assert_eq!(looked_at(1, MaintenanceAdmission::Rejected("memory")), 1);
        assert_eq!(looked_at(0, MaintenanceAdmission::Rejected("fds")), 0);
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
                unreached: 0,
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
            storage_backend: StorageBackendConfig::Lmdb,
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

        let temp_dir = TempDir::new().unwrap();
        let mut config = Config::new(8, 32, 64, 1).with_lsm_in_memory();
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

    /// Sparse terms for doc `id` of one "file": `terms` distinct ids drawn from
    /// a shared vocabulary so chunks of one file overlap like real code.
    fn ce_shaped_sparse(id: u128, terms: usize) -> SparseVector {
        let vocab = 4 * terms as u32;
        let mut indices: Vec<u32> = (0..terms as u32)
            .map(|k| (id as u32 * 37 + k * 3) % vocab) // 3 is coprime with every 4*terms here
            .collect();
        indices.sort_unstable();
        let values = indices.iter().map(|i| 1.0 + (*i % 5) as f32).collect();
        SparseVector { indices, values }
    }

    /// LSM collection shaped like a client code collection: `dense` + `mini` dense
    /// vectors, the given Idf sparse vectors, and a keyword index on `path`.
    fn ce_shaped_lsm_collection(
        temp_dir: &TempDir,
        ids: std::ops::RangeInclusive<u128>,
        terms: usize,
        sparse_names: &[&str],
    ) -> (Arc<CollectionManager>, Config) {
        let config = Config::new(8, 32, 64, 1).with_lsm_in_memory();
        let collections =
            Arc::new(CollectionManager::new(temp_dir.path().join("data"), config.clone()).unwrap());
        let dense = |size| NamedVectorConfig {
            size,
            distance: crate::helix_engine::vector_core::named_vectors::DistanceMetric::Cosine,
            spindle: crate::helix_engine::vector_core::spindle::SpindleConfig::default(),
        };
        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::CreateCollection {
                name: "repo".into(),
                vectors: HashMap::from([("dense".into(), dense(4)), ("mini".into(), dense(2))]),
                sparse_vectors: sparse_names
                    .iter()
                    .map(|name| {
                        let config = SparseVectorConfig {
                            modifier: crate::helix_engine::vector_core::sparse::SparseModifier::Idf,
                            ..Default::default()
                        };
                        (name.to_string(), config)
                    })
                    .collect(),
                hnsw_overrides: None,
            },
        )
        .unwrap();
        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::CreatePayloadIndex {
                collection: "repo".into(),
                field_name: "path".into(),
                schema: PayloadIndexSchema::Keyword,
            },
        )
        .unwrap();
        let points = ids
            .map(|id| {
                let mut v = [0.1f32; 4];
                v[(id as usize) % 4] = 1.0 + id as f32 * 0.001;
                ReplicatedPoint {
                    id,
                    vectors: HashMap::from([
                        ("dense".into(), v.to_vec()),
                        ("mini".into(), vec![1.0, id as f32 * 0.01]),
                    ]),
                    sparse_vectors: sparse_names
                        .iter()
                        .enumerate()
                        .map(|(i, name)| {
                            (
                                name.to_string(),
                                ce_shaped_sparse(id + 1000 * i as u128, terms),
                            )
                        })
                        .collect(),
                    payload: HashMap::from([(
                        "path".into(),
                        Value::String(format!("src/file_{}.rs", id % 2)),
                    )]),
                }
            })
            .collect();
        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::UpsertPoints {
                collection: "repo".into(),
                points,
            },
        )
        .unwrap();
        (collections, config)
    }

    /// LSM node/vector counters (served as points_count) must go down when
    /// points are deleted, not only up.
    #[test]
    fn apply_delete_points_lsm_decrements_node_and_vector_counters() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let (collections, config) =
            ce_shaped_lsm_collection(&temp_dir, 1..=12, 20, &["lex_sparse"]);
        let storage = collections.get_collection("repo").unwrap();
        let counts = || {
            let stats = storage
                .with_read_backend(|r| storage.get_metadata_be(r))
                .unwrap()
                .stats;
            (stats.node_count, stats.vector_count)
        };
        let delete = |ids: Vec<u128>| {
            apply_mutation(
                &collections,
                &config,
                &ReplicatedMutation::DeletePoints {
                    collection: "repo".into(),
                    ids,
                },
            )
            .unwrap()
        };
        assert_eq!(counts(), (12, 12));
        delete(vec![1, 2, 3, 4, 5, 6, 3, 999]);
        assert_eq!(counts(), (6, 6), "after deleting 6 of 12 points");
        delete((7..=12).collect());
        assert_eq!(counts(), (0, 0), "after deleting every point");
    }

    /// Every sparse vector of a point is deleted, leaving each sparse index
    /// scoring exactly like an index that only ever held the survivors.
    #[test]
    fn apply_delete_points_lsm_matches_fresh_index_for_every_sparse_vector() {
        let _guard = ENV_LOCK.lock().unwrap();
        let names = ["lex_sparse", "lex_title"];
        let (deleted_dir, fresh_dir) = (TempDir::new().unwrap(), TempDir::new().unwrap());
        let (deleted, config) = ce_shaped_lsm_collection(&deleted_dir, 1..=12, 40, &names);
        let (fresh, _) = ce_shaped_lsm_collection(&fresh_dir, 7..=12, 40, &names);
        apply_mutation(
            &deleted,
            &config,
            &ReplicatedMutation::DeletePoints {
                collection: "repo".into(),
                ids: vec![1, 2, 3, 4, 5, 6, 3, 999],
            },
        )
        .unwrap();

        let every_term = SparseVector {
            indices: (0..160).collect(),
            values: vec![1.0; 160],
        };
        let hits = |collections: &CollectionManager, name: &str| {
            let storage = collections.get_collection("repo").unwrap();
            let r = storage.backend.begin_read().unwrap();
            storage
                .named_vectors
                .with_sparse_core(name, |core| {
                    core.search_be::<fn(u128) -> bool>(&r, &every_term, 100, None)
                })
                .unwrap()
        };
        for name in names {
            let after_delete = hits(&deleted, name);
            assert!(
                !after_delete.is_empty(),
                "{name}: survivors must still match"
            );
            assert!(
                after_delete.iter().all(|(id, _)| *id >= 7),
                "{name}: deleted id returned: {after_delete:?}"
            );
            assert_eq!(
                after_delete,
                hits(&fresh, name),
                "{name}: Idf scores differ from an index of only the survivors"
            );
        }
    }

    /// A filter delete of one file's points must not pay one sequential SlateDB
    /// read per distinct sparse term: the sparse df/forward reads are batched
    /// per chunk, so the round-trip count is independent of terms per point.
    #[test]
    fn apply_delete_points_lsm_round_trips_do_not_scale_with_sparse_terms() {
        use crate::helix_engine::storage_core::backend_lsm::lsm_round_trips_for_test as round_trips;
        let _guard = ENV_LOCK.lock().unwrap();
        let delete_round_trips = |terms: usize| {
            let temp_dir = TempDir::new().unwrap();
            let (collections, config) =
                ce_shaped_lsm_collection(&temp_dir, 1..=64, terms, &["lex_sparse"]);
            let storage = collections.get_collection("repo").unwrap();
            // A duplicate and a never-inserted id are no-ops.
            let mut ids: Vec<u128> = (1..=64).collect();
            ids.extend([5, 9_999]);
            let before = round_trips();
            apply_mutation(
                &collections,
                &config,
                &ReplicatedMutation::DeletePoints {
                    collection: "repo".into(),
                    ids,
                },
            )
            .unwrap();
            let spent = round_trips() - before;

            let r = storage.backend.begin_read().unwrap();
            let every_term = SparseVector {
                indices: (0..4 * terms as u32).collect(),
                values: vec![1.0; 4 * terms],
            };
            let hits = storage
                .named_vectors
                .with_sparse_core("lex_sparse", |core| {
                    core.search_be::<fn(u128) -> bool>(&r, &every_term, 100, None)
                })
                .unwrap();
            assert!(
                hits.is_empty(),
                "deleted points still in sparse search: {hits:?}"
            );
            spent
        };
        let few_terms = delete_round_trips(20);
        let many_terms = delete_round_trips(280);
        assert!(
            few_terms > 0,
            "the delete must run on this thread: {few_terms}"
        );
        assert!(
            many_terms <= few_terms + 8,
            "delete round trips grew with sparse terms per point: {few_terms} (20 terms) vs {many_terms} (280 terms)"
        );
    }

    /// A file delete used to pay one SlateDB round trip per id for its node
    /// row, one per (dense segment, id) for the ownership probe and three per
    /// owned (segment, id) for the rows to remove. Batched, those reads cost a
    /// fixed number of round trips, and leave exactly what the one-at-a-time
    /// reads leave.
    #[test]
    fn apply_delete_points_lsm_batches_its_reads_without_changing_the_result() {
        use crate::helix_engine::storage_core::backend_lsm::lsm_round_trips_for_test as round_trips;
        let _guard = ENV_LOCK.lock().unwrap();
        let _hard_deletes = EnvVarGuard::unset("HELIX_LSM_DELETE_TOMBSTONES");
        let delete = |batch_reads: &'static str| {
            let _knob = EnvVarGuard::set("HELIX_LSM_DELETE_BATCH_READS", batch_reads);
            let temp_dir = TempDir::new().unwrap();
            let (collections, config) =
                ce_shaped_lsm_collection(&temp_dir, 1..=64, 20, &["lex_sparse"]);
            let storage = collections.get_collection("repo").unwrap();
            let before = round_trips();
            apply_mutation(
                &collections,
                &config,
                &ReplicatedMutation::DeletePoints {
                    collection: "repo".into(),
                    ids: (1..=48).collect(),
                },
            )
            .unwrap();
            let spent = round_trips() - before;

            let r = storage.backend.begin_read().unwrap();
            let cores = storage.named_vectors.cores_read().unwrap();
            let state: Vec<(u128, bool, bool)> = (1..=64u128)
                .map(|id| {
                    let node = storage
                        .backend
                        .get_with(&r, Namespace::Nodes, &id.to_be_bytes(), |v| v.is_some())
                        .unwrap();
                    let dense = cores
                        .values()
                        .any(|core| core.contains_id(&r, id).unwrap_or(false));
                    (id, node, dense)
                })
                .collect();
            let stats = storage
                .with_read_backend(|r| storage.get_metadata_be(r))
                .unwrap()
                .stats;
            (spent, state, (stats.node_count, stats.vector_count))
        };
        let (batched, batched_state, batched_counts) = delete("1");
        let (serial, serial_state, serial_counts) = delete("0");

        assert_eq!(batched_state, serial_state);
        assert_eq!(batched_counts, (16, 16), "48 of 64 points deleted");
        assert_eq!(batched_counts, serial_counts);
        for (id, node, dense) in &batched_state {
            assert_eq!(*node, *id > 48, "node row of {id}");
            assert_eq!(*dense, *id > 48, "dense vector of {id}");
        }
        // Serial: 48 node reads + 2 segments x 48 probes + 2 x 48 x 3 dense
        // scans, then 48 node reads + 2 x 48 edge scans to drop the nodes.
        assert!(
            batched + 300 <= serial,
            "batched delete reads saved too few round trips: {batched} batched vs {serial} serial"
        );
    }

    /// A chunk's node reads, sparse forward reads and counter bumps are batched,
    /// so upsert round trips do not grow with the number of points, while the
    /// node and vector counters still count each new point once (a repeated id
    /// in one chunk included).
    #[test]
    fn apply_upsert_points_lsm_round_trips_do_not_scale_with_points() {
        use crate::helix_engine::storage_core::backend_lsm::lsm_round_trips_for_test as round_trips;
        let _guard = ENV_LOCK.lock().unwrap();
        let point = |id: u128, path: &str| {
            let mut v = [0.1f32; 4];
            v[(id as usize) % 4] = 1.0 + id as f32 * 0.001;
            ReplicatedPoint {
                id,
                vectors: HashMap::from([
                    ("dense".into(), v.to_vec()),
                    ("mini".into(), vec![1.0, id as f32 * 0.01]),
                ]),
                sparse_vectors: HashMap::from([(
                    "lex_sparse".to_string(),
                    ce_shaped_sparse(id, 20),
                )]),
                payload: HashMap::from([("path".into(), Value::String(path.to_string()))]),
            }
        };
        // Upsert `ids` into a 64-point collection: (round trips, node/vector counts).
        let upsert = |ids: Vec<u128>, path: &str| {
            let temp_dir = TempDir::new().unwrap();
            let (collections, config) =
                ce_shaped_lsm_collection(&temp_dir, 1..=64, 20, &["lex_sparse"]);
            let storage = collections.get_collection("repo").unwrap();
            let points = ids.iter().map(|id| point(*id, path)).collect();
            let before = round_trips();
            apply_mutation(
                &collections,
                &config,
                &ReplicatedMutation::UpsertPoints {
                    collection: "repo".into(),
                    points,
                },
            )
            .unwrap();
            let spent = round_trips() - before;
            let stats = storage
                .with_read_backend(|r| storage.get_metadata_be(r))
                .unwrap()
                .stats;
            (spent, (stats.node_count, stats.vector_count))
        };

        let (fresh8, counts) = upsert((101..=108).collect(), "a.rs");
        assert_eq!(counts, (72, 72), "8 new points");
        let (fresh64, counts) = upsert((201..=264).collect(), "a.rs");
        assert_eq!(counts, (128, 128), "64 new points");
        let (changed8, counts) = upsert((1..=8).collect(), "b.rs");
        assert_eq!(counts, (64, 64), "8 changed points");
        let (changed64, counts) = upsert((1..=64).collect(), "b.rs");
        assert_eq!(counts, (64, 64), "64 changed points");
        let mut repeated: Vec<u128> = (301..=310).collect();
        repeated.extend([305, 7]);
        let (_, counts) = upsert(repeated, "a.rs");
        assert_eq!(
            counts,
            (74, 74),
            "10 new points, one repeated, one existing"
        );

        assert!(fresh8 > 0, "the upsert must run on this thread");
        // The per-point reads paid 4 round trips per new point and 2 per changed one.
        assert!(
            fresh64 <= fresh8 + 8,
            "upsert round trips grew with new points: {fresh8} (8) vs {fresh64} (64)"
        );
        assert!(
            changed64 <= changed8 + 8,
            "upsert round trips grew with changed points: {changed8} (8) vs {changed64} (64)"
        );
    }

    /// Deterministic unit-ish vectors: xorshift64 so the fixture needs no seed
    /// plumbing and brute-force ground truth can be recomputed from the ids.
    fn prod_shaped_vector(id: u128, dim: usize) -> Vec<f32> {
        let mut state = (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..dim)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    /// LSM collection shaped like a client code collection on the prod writer at
    /// reduced scale: `points` dense points sealed into `rounds` Indexed HNSW
    /// segments by the real optimizer, plus `edges` graph edges among the
    /// points above `protected_below` (so a delete of `1..=protected_below`
    /// touches none of them). Returns the ids as point ids.
    fn prod_shaped_lsm_collection(
        temp_dir: &TempDir,
        points: u128,
        rounds: u128,
        protected_below: u128,
        edges: usize,
    ) -> (Arc<CollectionManager>, Config) {
        let mut config = Config::new(8, 32, 64, 1).with_lsm_in_memory();
        config.vector_config.flat_scan_threshold = Some(16);
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
                        size: 8,
                        distance: DistanceMetric::Cosine,
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
        let flat_scan_threshold = storage
            .named_vectors
            .effective_indexing_threshold(config.vector_flat_scan_threshold());
        let per_round = points / rounds;
        for round in 0..rounds {
            let first = round * per_round + 1;
            let last = if round + 1 == rounds {
                points
            } else {
                first + per_round - 1
            };
            apply_mutation(
                &collections,
                &config,
                &ReplicatedMutation::UpsertPoints {
                    collection: "repo".into(),
                    points: (first..=last)
                        .map(|id| ReplicatedPoint {
                            id,
                            vectors: HashMap::from([("dense".into(), prod_shaped_vector(id, 8))]),
                            sparse_vectors: HashMap::new(),
                            payload: HashMap::from([(
                                "path".into(),
                                Value::String(format!("src/file_{}.rs", id % 50)),
                            )]),
                        })
                        .collect(),
                },
            )
            .unwrap();
            run_index_job(IndexJob {
                collection: "repo".to_string(),
                storage: Arc::downgrade(&storage),
                spaces_to_check: vec!["dense".to_string()],
                hnsw_config: HNSWConfig::new(Some(8), Some(32), Some(64)),
                flat_scan_threshold,
            });
        }
        // Settle before measuring: every point sealed into an Indexed segment
        // and the segment reaper drained, so neither search visibility nor the
        // backend read counter is perturbed by background optimizer work.
        let settle_deadline = Instant::now() + Duration::from_secs(20);
        loop {
            let r = storage.backend.begin_read().unwrap();
            let stats = storage
                .named_vectors
                .get_dense_space_stats_be(&r, "dense")
                .unwrap();
            drop(r);
            let all_indexed = stats.indexed_vectors_count == stats.vectors_count;
            if all_indexed && SEGMENT_REAPER.idle() {
                break;
            }
            assert!(
                Instant::now() < settle_deadline,
                "fixture did not settle: indexed {}/{} vectors, reaper idle {}",
                stats.indexed_vectors_count,
                stats.vectors_count,
                SEGMENT_REAPER.idle()
            );
            if !all_indexed {
                run_index_job(IndexJob {
                    collection: "repo".to_string(),
                    storage: Arc::downgrade(&storage),
                    spaces_to_check: vec!["dense".to_string()],
                    hnsw_config: HNSWConfig::new(Some(8), Some(32), Some(64)),
                    flat_scan_threshold,
                });
            }
            thread::sleep(Duration::from_millis(100));
        }
        if edges > 0 {
            let survivors = points - protected_below;
            let endpoint = |k: usize| protected_below + 1 + (k as u128 % survivors);
            apply_mutation(
                &collections,
                &config,
                &ReplicatedMutation::IngestBatch {
                    collection: "repo".into(),
                    ops: (0..edges)
                        .map(|i| {
                            ReplicatedIngestOp::Edge(EdgeUpsert {
                                id: (1u128 << 64) | i as u128,
                                label: "calls".into(),
                                from_node: endpoint(i * 7),
                                to_node: endpoint(i * 13 + 5),
                                properties: HashMap::new(),
                            })
                        })
                        .collect(),
                },
            )
            .unwrap();
        }
        (collections, config)
    }

    /// Prod shape: a batch delete on a collection whose graph holds many edges.
    /// Every deleted point used to pay two scans over the ENTIRE OutEdges and
    /// InEdges namespaces (`drop_node_be` -> `collect_incident_edges` ->
    /// LSM dup-namespace prefix scan that ignored the prefix), so delete work
    /// was O(deleted ids x edges) instead of O(deleted ids' own edges).
    #[test]
    fn apply_delete_points_lsm_reads_do_not_scale_with_graph_edges() {
        use crate::helix_engine::storage_core::backend_lsm::dup_scan_rows_for_test;
        let _guard = ENV_LOCK.lock().unwrap();
        const POINTS: u128 = 1200;
        const DELETED: u128 = 300;
        const EDGES: usize = 6000;
        let measure = |edges: usize| {
            let temp_dir = TempDir::new().unwrap();
            let (collections, config) =
                prod_shaped_lsm_collection(&temp_dir, POINTS, 3, DELETED, edges);
            let storage = collections.get_collection("repo").unwrap();
            let AnyBackend::Lsm(backend) = &*storage.backend else {
                panic!("fixture must run on the LSM writer backend");
            };
            {
                let r = storage.backend.begin_read().unwrap();
                let stats = storage
                    .named_vectors
                    .get_dense_space_stats_be(&r, "dense")
                    .unwrap();
                assert!(
                    stats.segments_count >= 2,
                    "fixture must span several segments, got {}",
                    stats.segments_count
                );
            }
            backend.reset_read_count();
            let edge_rows_before = dup_scan_rows_for_test();
            let started = Instant::now();
            apply_mutation(
                &collections,
                &config,
                &ReplicatedMutation::DeletePoints {
                    collection: "repo".into(),
                    ids: (1..=DELETED).collect(),
                },
            )
            .unwrap();
            let elapsed = started.elapsed();
            let reads = backend.read_count();
            let edge_rows = dup_scan_rows_for_test() - edge_rows_before;

            let r = storage.backend.begin_read().unwrap();
            let stats = storage
                .named_vectors
                .get_dense_space_stats_be(&r, "dense")
                .unwrap();
            assert_eq!(stats.vectors_count, (POINTS - DELETED) as u64);
            for id in [1u128, DELETED] {
                let node = storage
                    .backend
                    .get_with(&r, Namespace::Nodes, &id.to_be_bytes(), |v| v.is_some())
                    .unwrap();
                assert!(!node, "deleted point {id} must be gone");
            }
            if edges > 0 {
                let edge_id = (1u128 << 64) | (edges as u128 - 1);
                let kept = storage
                    .backend
                    .get_with(&r, Namespace::Edges, &edge_id.to_be_bytes(), |v| {
                        v.is_some()
                    })
                    .unwrap();
                assert!(kept, "edges among survivors must be untouched");
            }
            eprintln!(
                "delete {DELETED}/{POINTS} points with {edges} survivor edges: {edge_rows} edge rows scanned, {reads} backend reads, {elapsed:?}"
            );
            (edge_rows, reads, elapsed)
        };
        let (edge_rows_without_edges, ..) = measure(0);
        let (edge_rows_with_edges, ..) = measure(EDGES);
        // Each deleted id may probe one row past its own (empty) adjacency per
        // namespace; anything proportional to EDGES means the scan is unbounded.
        assert!(
            edge_rows_with_edges <= edge_rows_without_edges + DELETED as usize * 4,
            "delete scanned edge rows grew with the graph size: {edge_rows_without_edges} rows with no edges vs {edge_rows_with_edges} with {EDGES} edges"
        );
    }

    /// Search after a batch delete: deleted ids never surface and recall@10
    /// against brute force over the survivors stays at the level the HNSW
    /// index delivers today (the delete fix does not touch graph repair).
    #[test]
    fn apply_delete_points_lsm_keeps_dense_recall_after_delete() {
        let _guard = ENV_LOCK.lock().unwrap();
        const POINTS: u128 = 1200;
        const DELETED: u128 = 300;
        const QUERIES: u128 = 40;
        const K: usize = 10;
        let temp_dir = TempDir::new().unwrap();
        let (collections, config) = prod_shaped_lsm_collection(&temp_dir, POINTS, 3, DELETED, 0);
        let storage = collections.get_collection("repo").unwrap();

        let cosine = |a: &[f32], b: &[f32]| {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (na * nb)
        };
        let recall = |live: std::ops::RangeInclusive<u128>| -> f32 {
            let r = storage.backend.begin_read().unwrap();
            let mut hits = 0usize;
            for q in 0..QUERIES {
                let query = prod_shaped_vector(1_000_000 + q, 8);
                let mut exact: Vec<(u128, f32)> = live
                    .clone()
                    .map(|id| (id, cosine(&query, &prod_shaped_vector(id, 8))))
                    .collect();
                exact.sort_by(|a, b| b.1.total_cmp(&a.1));
                let truth: HashSet<u128> = exact.iter().take(K).map(|(id, _)| *id).collect();
                let found = storage
                    .named_vectors
                    .dense_search_with_id_filter_ef_be::<fn(u128) -> bool>(
                        &r, "dense", &query, K, None, false, None, None,
                    )
                    .unwrap();
                for hit in &found {
                    assert!(
                        live.contains(&hit.get_id()),
                        "deleted point {} surfaced in search",
                        hit.get_id()
                    );
                }
                hits += found.iter().filter(|h| truth.contains(&h.get_id())).count();
            }
            hits as f32 / (QUERIES as usize * K) as f32
        };

        let before = recall(1..=POINTS);
        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::DeletePoints {
                collection: "repo".into(),
                ids: (1..=DELETED).collect(),
            },
        )
        .unwrap();
        let after = recall(DELETED + 1..=POINTS);
        eprintln!("dense recall@{K}: before delete {before:.3}, after delete {after:.3}");
        // Floor measured on the pre-fix code with this exact seed and shape.
        assert!(
            after >= 0.9 && after + 0.05 >= before,
            "recall@{K} regressed after delete: {before:.3} -> {after:.3}"
        );
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
        const COLLECTION: &str = "bulk-upsert-finalizes-dense-tail";
        let temp_dir = TempDir::new().unwrap();
        let mut config = Config::new(8, 32, 64, 1);
        config.vector_config.flat_scan_threshold = Some(2);
        let collections =
            Arc::new(CollectionManager::new(temp_dir.path().join("data"), config.clone()).unwrap());

        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::CreateCollection {
                name: COLLECTION.into(),
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
                collection: COLLECTION.into(),
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
        let storage = collections.get_collection(COLLECTION).unwrap();
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

        let temp_dir = TempDir::new().unwrap();
        let mut config = Config::new(8, 32, 64, 1).with_lsm_in_memory();
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
    fn optimizer_builds_existing_lsm_segment_before_waiting_for_tail_quiescence() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _runtime = EnvVarGuard::set("HELIX_INDEX_JOB_MAX_RUNTIME_MS", "50");
        // The upsert below queues an index job of its own. It must leave the
        // tail flat and be gone before this test seals the tail and runs a job
        // by hand: alongside that job it can build the sealed segment first
        // and still hold it `Building` when the test looks.
        let _flat_tail = EnvVarGuard::set("HELIX_INDEX_SUBTHRESHOLD_QUIESCED_TAILS", "0");

        let temp_dir = TempDir::new().unwrap();
        let mut config = Config::new(8, 32, 64, 1).with_lsm_in_memory();
        config.vector_config.flat_scan_threshold = Some(1_000);
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
                points: (1..=6u128)
                    .map(|id| ReplicatedPoint {
                        id,
                        vectors: HashMap::from([("dense".into(), vec![id as f32, 1.0, 0.0, 0.0])]),
                        sparse_vectors: HashMap::new(),
                        payload: HashMap::new(),
                    })
                    .collect(),
            },
        )
        .unwrap();
        wait_for_index_executor("repo");
        let _quiesced_tails = EnvVarGuard::set("HELIX_INDEX_SUBTHRESHOLD_QUIESCED_TAILS", "1");

        let storage = collections.get_collection("repo").unwrap();
        let read = storage.backend.begin_read().unwrap();
        assert!(storage
            .named_vectors
            .seal_mutable_tail_lsm(
                &read,
                storage.collection_path(),
                "dense",
                HNSWConfig::new(Some(8), Some(32), Some(64)),
            )
            .unwrap()
            .is_some());
        drop(read);
        storage
            .with_write_backend(|write| {
                storage.set_dense_vector_spaces_metadata_be(
                    write,
                    storage.named_vectors.list_dense_vector_spaces(),
                )?;
                Ok(())
            })
            .unwrap();

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        storage
            .last_upsert_at
            .store(now_ms, std::sync::atomic::Ordering::Release);

        run_index_job(IndexJob {
            collection: "repo".to_string(),
            storage: Arc::downgrade(&storage),
            spaces_to_check: vec!["dense".to_string()],
            hnsw_config: HNSWConfig::new(Some(8), Some(32), Some(64)),
            flat_scan_threshold: 1_000,
        });

        let read = storage.backend.begin_read().unwrap();
        let stats = storage
            .named_vectors
            .get_dense_space_stats_be(&read, "dense")
            .unwrap();
        assert_eq!(stats.indexed_vectors_count, 6);
        assert!(storage
            .named_vectors
            .building_segment_names("dense")
            .is_empty());
    }

    /// A sealed LSM tail is `Building` until its build publishes; its points
    /// live in no other segment, so every dense path must keep returning them
    /// (newest copy wins on re-upsert, deletes stay deleted), while a merge
    /// target's `Building` copy stays hidden.
    #[test]
    fn sealed_lsm_tail_stays_searchable_until_its_build_publishes() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let mut config = Config::new(8, 32, 64, 1).with_lsm_in_memory();
        // 0 keeps upserts from queueing background index jobs; the explicit
        // job run below may still reschedule one, so layout checks name
        // segments instead of counting them.
        config.vector_config.flat_scan_threshold = Some(0);
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
        let upsert = |points: Vec<(u128, [f32; 4])>| {
            apply_mutation(
                &collections,
                &config,
                &ReplicatedMutation::UpsertPoints {
                    collection: "repo".into(),
                    points: points
                        .into_iter()
                        .map(|(id, vector)| ReplicatedPoint {
                            id,
                            vectors: HashMap::from([("dense".into(), vector.to_vec())]),
                            sparse_vectors: HashMap::new(),
                            payload: HashMap::new(),
                        })
                        .collect(),
                },
            )
            .unwrap();
        };
        upsert(
            (1..=6u128)
                .map(|id| (id, [id as f32, 1.0, 0.0, 0.0]))
                .collect(),
        );
        let storage = collections.get_collection("repo").unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let query = [0.0, 1.0, 0.0, 0.0];
        // (id, distance bits) from every LSM dense path, sorted by id.
        let search_all = || {
            let read = storage.backend.begin_read().unwrap();
            let nv = &storage.named_vectors;
            let ids = (1..=99u128).collect::<HashSet<_>>();
            let paths = [
                nv.dense_search_with_id_filter_ef_be::<fn(u128) -> bool>(
                    &read, "dense", &query, 10, None, true, None, None,
                ),
                nv.dense_search_candidate_set_filter_ef_be(
                    &read, "dense", &query, 10, &ids, true, None, None,
                ),
                nv.dense_search_candidate_ids_exact_be(
                    &read,
                    "dense",
                    &query,
                    10,
                    ids.iter().copied(),
                    None,
                ),
            ];
            paths
                .into_iter()
                .map(|hits| {
                    let mut hits = hits
                        .unwrap()
                        .into_iter()
                        .map(|v| (v.get_id(), v.get_distance().to_bits()))
                        .collect::<Vec<_>>();
                    hits.sort_unstable();
                    hits
                })
                .collect::<Vec<_>>()
        };
        let ids_of = |hits: &[(u128, u32)]| hits.iter().map(|(id, _)| *id).collect::<Vec<_>>();

        let tail = storage
            .named_vectors
            .mutable_dense_segment_name("dense")
            .unwrap();
        assert!(seal_lsm_dense_tail(&storage, "dense", &hnsw).unwrap());
        assert_eq!(
            storage.named_vectors.building_segment_names("dense"),
            vec![tail.clone()]
        );
        for hits in search_all() {
            assert_eq!(
                ids_of(&hits),
                (1..=6).collect::<Vec<_>>(),
                "sealed points vanished"
            );
        }

        // Re-upsert id 1 onto the query into the fresh tail; delete id 2.
        upsert(vec![(1, [0.0, 1.0, 0.0, 0.0])]);
        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::DeletePoints {
                collection: "repo".into(),
                ids: vec![2],
            },
        )
        .unwrap();
        let building = search_all();
        for hits in &building {
            assert_eq!(ids_of(hits), vec![1, 3, 4, 5, 6]);
            assert!(
                f32::from_bits(hits[0].1) < 1e-4,
                "stale sealed copy of id 1 won"
            );
        }
        assert_eq!(building[0], building[1]);
        assert_eq!(building[0], building[2]);

        // Mid-publish states stay exact: half the adjacency flushed (no entry
        // point, still flat), then a complete index not yet promoted.
        let (name, mut prepared) = {
            let read = storage.backend.begin_read().unwrap();
            storage
                .named_vectors
                .prepare_building_indices_limited_be(&read, "dense", 1)
                .unwrap()
                .pop()
                .unwrap()
        };
        assert_eq!(name, tail);
        let total = prepared.point_ids.len();
        for range in [0..total / 2, total / 2..total] {
            let finalize = range.end == total;
            storage
                .with_write_backend(|w| {
                    let cores = storage.named_vectors.cores_read().unwrap();
                    let core = cores.get(&tail).unwrap();
                    core.flush_prepared_index_chunk_be(w, &mut prepared, range.start, range.end)?;
                    if finalize {
                        core.finalize_prepared_index_entry_be(w, &prepared)?;
                    }
                    Ok(())
                })
                .unwrap();
            let read = storage.backend.begin_read().unwrap();
            let indexed = storage
                .named_vectors
                .cores_read()
                .unwrap()
                .get(&tail)
                .unwrap()
                .has_index(&read)
                .unwrap();
            assert_eq!(indexed, finalize);
            assert_eq!(search_all(), building, "mid-publish state changed results");
        }

        run_index_job(IndexJob {
            collection: "repo".to_string(),
            storage: Arc::downgrade(&storage),
            spaces_to_check: vec!["dense".to_string()],
            hnsw_config: hnsw.clone(),
            flat_scan_threshold: 1_000,
        });
        assert!(!storage
            .named_vectors
            .building_segment_names("dense")
            .contains(&tail));
        assert_eq!(search_all(), building, "promotion changed results");

        // An in-flight merge target is Building too, but unmarked: hidden.
        let target = storage
            .named_vectors
            .create_merge_target_lsm(
                &storage.named_vectors.begin_dense_layout_change(),
                storage.collection_path(),
                "dense",
                hnsw,
            )
            .unwrap();
        storage
            .with_write_backend(|w| {
                let cores = storage.named_vectors.cores_read().unwrap();
                cores
                    .get(&target)
                    .unwrap()
                    .insert_flat_be(w, &query, Some(99), None)
                    .map_err(GraphError::from)?;
                drop(cores);
                storage.set_dense_vector_spaces_metadata_be(
                    w,
                    storage.named_vectors.list_dense_vector_spaces(),
                )
            })
            .unwrap();
        assert!(storage
            .named_vectors
            .building_segment_names("dense")
            .contains(&target));
        assert_eq!(search_all(), building, "merge target leaked into search");
    }

    /// A search whose read began before the index job sealed the tail still
    /// finds the tail's points on every dense path. The seal commits the
    /// sealed-tail marker and then flips the tail to `Building` in the
    /// writer's layout; that read lists the flipped layout and has no marker
    /// at its own snapshot.
    #[test]
    fn lsm_writer_read_begun_before_a_seal_finds_the_tail_on_every_dense_path() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let mut config = Config::new(8, 32, 64, 1).with_lsm_in_memory();
        // 0 keeps upserts from queueing background index jobs.
        config.vector_config.flat_scan_threshold = Some(0);
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
                points: (1..=6u128)
                    .map(|id| ReplicatedPoint {
                        id,
                        vectors: HashMap::from([("dense".into(), vec![id as f32, 1.0, 0.0, 0.0])]),
                        sparse_vectors: HashMap::new(),
                        payload: HashMap::new(),
                    })
                    .collect(),
            },
        )
        .unwrap();
        let storage = collections.get_collection("repo").unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let query = [0.0, 1.0, 0.0, 0.0];
        // Ids from every LSM dense path for one read, sorted.
        let found = |read: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>| {
            let nv = &storage.named_vectors;
            let ids = (1..=99u128).collect::<HashSet<_>>();
            [
                nv.dense_search_with_id_filter_ef_be::<fn(u128) -> bool>(
                    read, "dense", &query, 10, None, true, None, None,
                ),
                nv.dense_search_candidate_set_filter_ef_be(
                    read, "dense", &query, 10, &ids, true, None, None,
                ),
                nv.dense_search_candidate_ids_exact_be(
                    read,
                    "dense",
                    &query,
                    10,
                    ids.iter().copied(),
                    None,
                ),
            ]
            .into_iter()
            .map(|hits| {
                let mut ids = hits
                    .unwrap()
                    .into_iter()
                    .map(|v| v.get_id())
                    .collect::<Vec<_>>();
                ids.sort_unstable();
                ids
            })
            .collect::<Vec<_>>()
        };
        let all = vec![(1..=6u128).collect::<Vec<_>>(); 3];

        // A request that has its snapshot and has not listed the segments yet.
        let before = storage.backend.begin_read().unwrap();
        assert_eq!(found(&before), all, "before the seal");
        assert!(seal_lsm_dense_tail(&storage, "dense", &hnsw).unwrap());
        let after = storage.backend.begin_read().unwrap();
        assert_eq!(found(&after), all, "a read begun after the seal");
        assert_eq!(
            found(&before),
            all,
            "a read begun before the seal lost the sealed tail's points"
        );
    }

    /// The LSM open builds its snapshot from the metadata blob, and the
    /// counts live in counter keys beside it. A collection that was just
    /// opened must answer its stats with those counts, on a reader and on a
    /// writer, before any commit or later refresh.
    #[test]
    fn lsm_metadata_snapshot_has_the_counts_right_after_an_open() {
        use crate::helix_engine::storage_core::backend_any;

        let _guard = ENV_LOCK.lock().unwrap();
        let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "stats_after_open_lsm");
        let store: Arc<dyn slatedb::object_store::ObjectStore> =
            Arc::new(slatedb::object_store::memory::InMemory::new());
        let _store = backend_any::set_lsm_test_object_store(store);
        let lsm = |role| StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role,
        };
        let config = Config::new(16, 64, 128, 1).with_storage_backend(lsm(LsmRole::Writer));
        let reader_config = Config::new(16, 64, 128, 1).with_storage_backend(lsm(LsmRole::Reader));
        let name = "stats_repo";
        let counts = |manager: &CollectionManager| {
            let stats = manager.collection_stats(name).unwrap();
            (stats.node_count, stats.edge_count, stats.vector_count)
        };

        let written = {
            let dir = TempDir::new().unwrap();
            let writer =
                Arc::new(CollectionManager::new(dir.path().to_path_buf(), config.clone()).unwrap());
            apply_mutation(
                &writer,
                &config,
                &ReplicatedMutation::CreateCollection {
                    name: name.into(),
                    vectors: HashMap::from([(
                        "dense".into(),
                        NamedVectorConfig {
                            size: 4,
                            distance:
                                crate::helix_engine::vector_core::named_vectors::DistanceMetric::Cosine,
                            spindle:
                                crate::helix_engine::vector_core::spindle::SpindleConfig::default(),
                        },
                    )]),
                    sparse_vectors: HashMap::new(),
                    hnsw_overrides: None,
                },
            )
            .unwrap();
            apply_mutation(
                &writer,
                &config,
                &ReplicatedMutation::UpsertPoints {
                    collection: name.into(),
                    points: (1..=3u128)
                        .map(|id| ReplicatedPoint {
                            id,
                            vectors: HashMap::from([(
                                "dense".into(),
                                vec![1.0, 0.0, 0.0, id as f32],
                            )]),
                            sparse_vectors: HashMap::new(),
                            payload: HashMap::new(),
                        })
                        .collect(),
                },
            )
            .unwrap();
            counts(&writer)
        };
        assert_ne!(written, (0, 0, 0), "the upsert is counted");

        for config in [reader_config, config] {
            let dir = TempDir::new().unwrap();
            let manager = CollectionManager::new(dir.path().to_path_buf(), config).unwrap();
            drop(manager.get_collection(name).unwrap());
            assert_eq!(counts(&manager), written);
        }
    }

    /// A nearest-M segment rebuilt by regraph from its stored vectors: recall
    /// rises on clustered data, live rows keep their vectors and payloads,
    /// deleted and superseded copies stay gone, a reader replica serves the
    /// new segment after refresh, upserts and a delete racing a rebuild hold,
    /// and the route path refuses, queues once, then does no work.
    #[test]
    fn regraph_rebuilds_nearest_m_segments_from_stored_vectors() {
        use crate::helix_engine::storage_core::backend_any;
        use crate::helix_engine::vector_core::named_vectors::DenseVectorSegmentRole;
        use crate::helix_engine::vector_core::vector_core::with_diverse_neighbors_for_test;
        use crate::protocol::value::Value;
        use rand::{Rng, SeedableRng};

        let _guard = ENV_LOCK.lock().unwrap();
        let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "regraph_lsm");
        let _enabled = EnvVarGuard::unset("HELIX_REGRAPH_ENABLED");
        let _diverse = EnvVarGuard::unset("HELIX_HNSW_DIVERSE_NEIGHBORS");
        let store: Arc<dyn slatedb::object_store::ObjectStore> =
            Arc::new(slatedb::object_store::memory::InMemory::new());
        let _store = backend_any::set_lsm_test_object_store(store);
        let lsm = |role| StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role,
        };
        let mut config = Config::new(16, 64, 128, 1).with_storage_backend(lsm(LsmRole::Writer));
        config.vector_config.flat_scan_threshold = Some(0);
        let reader_config = Config::new(16, 64, 128, 1).with_storage_backend(lsm(LsmRole::Reader));
        let (writer_dir, reader_dir) = (TempDir::new().unwrap(), TempDir::new().unwrap());
        let writer = Arc::new(
            CollectionManager::new(writer_dir.path().to_path_buf(), config.clone()).unwrap(),
        );
        let reader = Arc::new(
            CollectionManager::new(reader_dir.path().to_path_buf(), reader_config.clone()).unwrap(),
        );
        let name = "regraph_repo";
        let dim = 32;
        apply_mutation(
            &writer,
            &config,
            &ReplicatedMutation::CreateCollection {
                name: name.into(),
                vectors: HashMap::from([(
                    "dense".into(),
                    NamedVectorConfig {
                        size: dim,
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
        let point = |id: u128, vector: &[f32]| ReplicatedPoint {
            id,
            vectors: HashMap::from([("dense".into(), vector.to_vec())]),
            sparse_vectors: HashMap::new(),
            payload: HashMap::from([("path".into(), Value::String(format!("src/f{id}.rs")))]),
        };
        let upsert = |points: Vec<ReplicatedPoint>| {
            apply_mutation(
                &writer,
                &config,
                &ReplicatedMutation::UpsertPoints {
                    collection: name.into(),
                    points,
                },
            )
            .unwrap();
        };

        // Tight topic clusters, as in `diverse_neighbors_lift_recall_on_clustered_data`.
        let unit = |v: Vec<f32>| {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / norm).collect::<Vec<f32>>()
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(41);
        let centers: Vec<Vec<f32>> = (0..60)
            .map(|_| unit((0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect()))
            .collect();
        let clustered = |rng: &mut rand::rngs::StdRng| {
            let center = &centers[rng.random_range(0..centers.len())];
            unit(
                center
                    .iter()
                    .map(|c| c + 0.05 * rng.random_range(-1.0f32..1.0))
                    .collect(),
            )
        };
        let rows: Vec<Vec<f32>> = (0..3000).map(|_| clustered(&mut rng)).collect();
        for (chunk_index, chunk) in rows.chunks(500).enumerate() {
            upsert(
                chunk
                    .iter()
                    .enumerate()
                    .map(|(i, v)| point((chunk_index * 500 + i + 1) as u128, v))
                    .collect(),
            );
        }
        let mut live: HashMap<u128, Vec<f32>> = rows
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u128 + 1, v.clone()))
            .collect();
        let storage = writer.get_collection(name).unwrap();
        let hnsw = HNSWConfig::new(Some(16), Some(64), Some(128));
        // The graph production built before diverse neighbors.
        with_diverse_neighbors_for_test(None, || {
            seal_lsm_dense_tail(&storage, "dense", &hnsw).unwrap();
            run_index_job(IndexJob {
                collection: name.into(),
                storage: Arc::downgrade(&storage),
                spaces_to_check: vec!["dense".into()],
                hnsw_config: hnsw.clone(),
                flat_scan_threshold: 1_000,
            });
        });
        let indexed = |s: &HelixGraphStorage| {
            s.named_vectors.list_dense_vector_spaces()["dense"]
                .segments
                .iter()
                .filter(|segment| segment.role == DenseVectorSegmentRole::Indexed)
                .map(|segment| segment.physical_name.clone())
                .collect::<Vec<_>>()
        };
        let old = indexed(&storage);
        assert_eq!(old.len(), 1, "{old:?}");
        // id -> (vector bits, payload fields) of one segment's level 0.
        let export = |segment: &str| {
            let r = storage.backend.begin_read().unwrap();
            let cores = storage.named_vectors.cores_read().unwrap();
            cores[segment]
                .export_level_zero(&r)
                .unwrap()
                .into_iter()
                .map(|(id, v, fields)| (id, (v.iter().map(|x| x.to_bits()).collect(), fields)))
                .collect::<HashMap<u128, (Vec<u32>, HashMap<String, Value>)>>()
        };
        let before_rows = export(&old[0]);
        assert_eq!(before_rows.len(), 3000);
        let search = |s: &HelixGraphStorage, query: &[f32], k: usize| {
            let r = s.backend.begin_read().unwrap();
            s.named_vectors
                .dense_search_with_id_filter_ef_be::<fn(u128) -> bool>(
                    &r,
                    "dense",
                    query,
                    k,
                    None,
                    false,
                    None,
                    Some(128),
                )
                .unwrap()
                .into_iter()
                .map(|v| (v.get_id(), v.get_distance().to_bits()))
                .collect::<Vec<_>>()
        };
        let mut qrng = rand::rngs::StdRng::seed_from_u64(43);
        let queries: Vec<Vec<f32>> = (0..60)
            .map(|_| {
                let a = &rows[qrng.random_range(0..rows.len())];
                let b = &rows[qrng.random_range(0..rows.len())];
                unit(a.iter().zip(b).map(|(x, y)| x + y).collect())
            })
            .collect();
        let recall = |live: &HashMap<u128, Vec<f32>>| {
            queries
                .iter()
                .map(|query| {
                    let mut scored: Vec<(u128, f32)> = live
                        .iter()
                        .map(|(id, v)| (*id, v.iter().zip(query).map(|(a, b)| a * b).sum()))
                        .collect();
                    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
                    let truth: HashSet<u128> = scored[..10].iter().map(|(id, _)| *id).collect();
                    search(&storage, query, 10)
                        .iter()
                        .filter(|(id, _)| truth.contains(id))
                        .count() as f64
                        / 10.0
                })
                .sum::<f64>()
                / queries.len() as f64
        };
        // Measured before deletes: one of them leaves the graph flat-searched.
        let before = recall(&live);
        // The graph's entry point is deleted too: LSM deletes leave its key
        // behind (`has_index` false), and the segment must still be rebuilt.
        let entry = {
            let r = storage.backend.begin_read().unwrap();
            storage.named_vectors.cores_read().unwrap()[&old[0]]
                .get_entry_point(&r)
                .unwrap()
                .get_id()
        };
        let reupserted = |id: u128| id == 11 || (20..=40).contains(&id);
        let deleted: Vec<u128> = (1..=10)
            .chain(41..=50)
            .chain((!reupserted(entry)).then_some(entry))
            .collect();

        apply_mutation(
            &writer,
            &config,
            &ReplicatedMutation::DeletePoints {
                collection: name.into(),
                ids: (1..=10).chain([entry]).collect(),
            },
        )
        .unwrap();
        let fresh_11 = clustered(&mut rng);
        upsert(vec![point(11, &fresh_11)]);
        live.retain(|id, _| *id > 10 && *id != entry);
        live.insert(11, fresh_11.clone());

        let replica = reader.get_collection(name).unwrap();
        assert_eq!(indexed(&replica), old);
        assert_eq!(
            search(&replica, &queries[0], 10),
            search(&storage, &queries[0], 10)
        );

        // Upserts (new ids and re-upserts) and a delete race a rebuild.
        let racing: Vec<(u128, Vec<f32>)> = (5001..=5200u128)
            .chain(20..=40)
            .map(|id| (id, clustered(&mut rng)))
            .collect();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let racer = {
            let (writer, config, racing) = (Arc::clone(&writer), config.clone(), racing.clone());
            std::thread::spawn(move || {
                for (i, batch) in racing.chunks(11).enumerate() {
                    apply_mutation(
                        &writer,
                        &config,
                        &ReplicatedMutation::UpsertPoints {
                            collection: name.into(),
                            points: batch.iter().map(|(id, v)| point(*id, v)).collect(),
                        },
                    )
                    .unwrap();
                    if i == 0 {
                        started_tx.send(()).unwrap();
                    }
                    if i == 1 {
                        apply_mutation(
                            &writer,
                            &config,
                            &ReplicatedMutation::DeletePoints {
                                collection: name.into(),
                                ids: (41..=50).collect(),
                            },
                        )
                        .unwrap();
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            })
        };
        started_rx.recv().unwrap();
        let gate = writer.point_mutation_gate(name);
        let spaces = ["dense".to_string()];
        let raced = regraph_collection(&storage, name, &spaces, &hnsw, &gate).unwrap();
        racer.join().unwrap();
        assert!(matches!(raced.state, "busy" | "done"), "{raced:?}");
        live.extend(racing.iter().cloned());
        live.retain(|id, _| !(41..=50).contains(id));
        for (id, v) in racing.iter().chain([(11, fresh_11.clone())].iter()) {
            let hit = search(&storage, v, 1);
            assert_eq!(hit[0].0, *id, "upsert of {id} lost");
            assert!(f32::from_bits(hit[0].1) < 1e-4, "stale copy of {id} served");
        }

        let quiet = regraph_collection(&storage, name, &spaces, &hnsw, &gate).unwrap();
        assert_eq!(quiet.state, "done", "{quiet:?}");
        let rebuilt = indexed(&storage);
        assert_eq!(rebuilt.len(), 1, "{rebuilt:?}");
        assert_ne!(
            rebuilt,
            old,
            "raced {raced:?} quiet {quiet:?} layout {:?}",
            storage.named_vectors.list_dense_vector_spaces()["dense"].segments
        );
        let superseded: HashSet<u128> = (1..=11).chain(20..=50).chain([entry]).collect();
        let expected: HashMap<_, _> = before_rows
            .into_iter()
            .filter(|(id, _)| !superseded.contains(id))
            .collect();
        assert_eq!(quiet.points_rebuilt, expected.len(), "{quiet:?}");
        assert!(
            export(&rebuilt[0]) == expected,
            "rebuilt rows differ from the live stored rows"
        );
        let after = recall(&live);
        println!("regraph recall@10: before {before:.3} after {after:.3}; raced {raced:?}");
        assert!(
            after >= 0.65 && after >= before + 0.15,
            "{before} -> {after}"
        );

        let served = |s: &HelixGraphStorage| {
            for &id in &deleted {
                assert!(
                    search(s, &rows[id as usize - 1], 10)
                        .iter()
                        .all(|(hit, _)| *hit != id),
                    "deleted {id} came back"
                );
            }
            for id in (11..=11).chain(20..=40u128) {
                assert!(
                    !search(s, &rows[id as usize - 1], 10)
                        .iter()
                        .any(|(hit, d)| *hit == id && f32::from_bits(*d) < 1e-4),
                    "superseded copy of {id} came back"
                );
            }
        };
        served(&storage);

        // The replica opened on the old segment serves the new one after a refresh.
        replica
            .backend
            .refresh_lsm_reader_with_poll_interval(Duration::from_millis(100))
            .unwrap();
        replica.refresh_reader_view_best_effort();
        assert_eq!(indexed(&replica), rebuilt);
        for query in &queries {
            assert_eq!(search(&replica, query, 10), search(&storage, query, 10));
        }
        served(&replica);

        // Route path: refusals, one queued job, then no work.
        {
            let _off = EnvVarGuard::set("HELIX_REGRAPH_ENABLED", "0");
            assert!(matches!(
                submit_collection_regraph(&writer, &config, name),
                Err(RegraphRefusal::Disabled)
            ));
        }
        assert!(matches!(
            submit_collection_regraph(&reader, &reader_config, name),
            Err(RegraphRefusal::ReaderReplica)
        ));
        storage.optimizer_running.store(true, Ordering::Release);
        assert!(matches!(
            submit_collection_regraph(&writer, &config, name),
            Err(RegraphRefusal::InProgress("optimizing"))
        ));
        storage.optimizer_running.store(false, Ordering::Release);
        assert_eq!(
            submit_collection_regraph(&writer, &config, name)
                .unwrap()
                .state,
            "queued"
        );
        let deadline = std::time::Instant::now() + Duration::from_secs(120);
        let done = loop {
            let status = regraph_status(&writer, name).unwrap();
            if !matches!(status.state, "queued" | "running") {
                break status;
            }
            assert!(std::time::Instant::now() < deadline, "regraph never ran");
            std::thread::sleep(Duration::from_millis(20));
        };
        assert_eq!(done.state, "done", "{done:?}");
        // The segment the quiet regraph built is the current build's.
        assert_eq!(
            (done.segments_rebuilt, done.segments_current),
            (0, 1),
            "{done:?}"
        );
        let again = submit_collection_regraph(&writer, &config, name).unwrap();
        assert_eq!(format!("{again:?}"), format!("{done:?}"));
        assert_eq!(
            format!("{:?}", regraph_status(&writer, name).unwrap()),
            format!("{done:?}"),
            "a done collection was queued again"
        );
        while storage.optimizer_running.load(Ordering::Acquire) {
            assert!(
                std::time::Instant::now() < deadline,
                "index job never ended"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// A merge that died after writing part of its target leaves an
    /// unmarked `Building` segment (hidden from search) holding copies of the
    /// source's ids. Regraph must not count those copies as newer live ones:
    /// every live id must still be served after the rebuild.
    #[test]
    fn regraph_keeps_ids_copied_into_a_hidden_building_segment() {
        use crate::helix_engine::storage_core::backend_any;
        use crate::helix_engine::vector_core::named_vectors::DenseVectorSegmentRole;
        use crate::helix_engine::vector_core::vector_core::with_diverse_neighbors_for_test;
        use crate::protocol::value::Value;
        use rand::{Rng, SeedableRng};

        let _guard = ENV_LOCK.lock().unwrap();
        let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "regraph_orphan_lsm");
        let _enabled = EnvVarGuard::unset("HELIX_REGRAPH_ENABLED");
        let store: Arc<dyn slatedb::object_store::ObjectStore> =
            Arc::new(slatedb::object_store::memory::InMemory::new());
        let _store = backend_any::set_lsm_test_object_store(store);
        let mut config =
            Config::new(16, 64, 128, 1).with_storage_backend(StorageBackendConfig::Lsm {
                storage: LsmStorage::ObjectStore,
                role: LsmRole::Writer,
            });
        config.vector_config.flat_scan_threshold = Some(0);
        let writer_dir = TempDir::new().unwrap();
        let writer = Arc::new(
            CollectionManager::new(writer_dir.path().to_path_buf(), config.clone()).unwrap(),
        );
        let name = "regraph_orphan";
        let dim = 16;
        apply_mutation(
            &writer,
            &config,
            &ReplicatedMutation::CreateCollection {
                name: name.into(),
                vectors: HashMap::from([(
                    "dense".into(),
                    NamedVectorConfig {
                        size: dim,
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
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let rows: Vec<Vec<f32>> = (0..1200)
            .map(|_| (0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect())
            .collect();
        for (chunk_index, chunk) in rows.chunks(400).enumerate() {
            apply_mutation(
                &writer,
                &config,
                &ReplicatedMutation::UpsertPoints {
                    collection: name.into(),
                    points: chunk
                        .iter()
                        .enumerate()
                        .map(|(i, v)| ReplicatedPoint {
                            id: (chunk_index * 400 + i + 1) as u128,
                            vectors: HashMap::from([("dense".into(), v.clone())]),
                            sparse_vectors: HashMap::new(),
                            payload: HashMap::from([(
                                "path".into(),
                                Value::String(format!("src/f{chunk_index}_{i}.rs")),
                            )]),
                        })
                        .collect(),
                },
            )
            .unwrap();
        }
        let storage = writer.get_collection(name).unwrap();
        let hnsw = HNSWConfig::new(Some(16), Some(64), Some(128));
        with_diverse_neighbors_for_test(None, || {
            seal_lsm_dense_tail(&storage, "dense", &hnsw).unwrap();
            run_index_job(IndexJob {
                collection: name.into(),
                storage: Arc::downgrade(&storage),
                spaces_to_check: vec!["dense".into()],
                hnsw_config: hnsw.clone(),
                flat_scan_threshold: 1_000,
            });
        });
        let roles = |s: &HelixGraphStorage| {
            s.named_vectors.list_dense_vector_spaces()["dense"]
                .segments
                .iter()
                .map(|segment| (segment.physical_name.clone(), segment.role))
                .collect::<Vec<_>>()
        };
        let indexed: Vec<String> = roles(&storage)
            .into_iter()
            .filter(|(_, role)| *role == DenseVectorSegmentRole::Indexed)
            .map(|(name, _)| name)
            .collect();
        assert_eq!(indexed.len(), 1, "{:?}", roles(&storage));

        // A merge of that segment that died in phase B: target allocated and
        // persisted as `Building`, half its rows flushed, never published.
        let exported = {
            let r = storage.backend.begin_read().unwrap();
            let cores = storage.named_vectors.cores_read().unwrap();
            cores[&indexed[0]].export_level_zero(&r).unwrap()
        };
        let orphan = storage
            .named_vectors
            .create_merge_target_lsm(
                &storage.named_vectors.begin_dense_layout_change(),
                storage.collection_path(),
                "dense",
                hnsw.clone(),
            )
            .unwrap();
        storage
            .with_write_backend(|w| {
                storage.set_dense_vector_spaces_metadata_be(
                    w,
                    storage.named_vectors.list_dense_vector_spaces(),
                )
            })
            .unwrap();
        let mut prepared = {
            let permit =
                crate::helix_engine::vector_core::vector_core::acquire_build_permit().unwrap();
            VectorCore::build_hnsw_in_memory_owned_with_permit(
                exported,
                &hnsw,
                crate::helix_engine::vector_core::named_vectors::DistanceMetric::Cosine,
                &permit,
            )
            .unwrap()
        };
        let half = prepared.point_ids.len() / 2;
        storage
            .with_write_backend(|w| {
                storage
                    .named_vectors
                    .flush_merge_flat_chunk_be(w, &orphan, &mut prepared, 0, half)
                    .map_err(GraphError::from)
            })
            .unwrap();
        assert!(roles(&storage)
            .iter()
            .any(|(n, role)| *n == orphan && *role == DenseVectorSegmentRole::Building));

        let search_top = |query: &[f32]| {
            let r = storage.backend.begin_read().unwrap();
            storage
                .named_vectors
                .dense_search_with_id_filter_ef_be::<fn(u128) -> bool>(
                    &r,
                    "dense",
                    query,
                    1,
                    None,
                    false,
                    None,
                    Some(128),
                )
                .unwrap()
                .first()
                .map(|v| v.get_id())
        };
        let served = || {
            (1..=rows.len() as u128)
                .filter(|id| search_top(&rows[*id as usize - 1]) == Some(*id))
                .count()
        };
        let before = served();
        let gate = writer.point_mutation_gate(name);
        let status =
            regraph_collection(&storage, name, &["dense".to_string()], &hnsw, &gate).unwrap();
        let after = served();
        println!(
            "regraph orphan: served before {before}/{} after {after}/{}; {status:?}; layout {:?}",
            rows.len(),
            rows.len(),
            roles(&storage)
        );
        let rebuilt = roles(&storage)
            .into_iter()
            .find(|(_, role)| *role == DenseVectorSegmentRole::Indexed)
            .map(|(name, _)| name)
            .unwrap();
        let kept: HashSet<u128> = {
            let r = storage.backend.begin_read().unwrap();
            let cores = storage.named_vectors.cores_read().unwrap();
            cores[&rebuilt]
                .export_level_zero(&r)
                .unwrap()
                .into_iter()
                .map(|(id, _, _)| id)
                .collect()
        };
        assert_eq!(kept.len(), rows.len(), "{status:?}");
        // Top-1 self-hits vary by up to ~1.5% between graph builds; a lost
        // half does not.
        assert!(
            after + rows.len() / 20 >= before,
            "regraph lost live points: {before} -> {after}"
        );
    }

    /// A requested regraph leaves a segment the current build made alone: one
    /// built without diverse neighbours is rebuilt once, one built as builds
    /// are now is not, and without the skip both are rebuilt.
    #[test]
    fn regraph_leaves_the_segments_the_current_build_made() {
        use crate::helix_engine::storage_core::backend_any;
        use crate::helix_engine::vector_core::vector_core::with_diverse_neighbors_for_test;

        let _guard = ENV_LOCK.lock().unwrap();
        let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "regraph_current_lsm");
        let hnsw = HNSWConfig::new(Some(16), Some(64), Some(128));
        let spaces = ["dense".to_string()];
        let mut seen = Vec::new();
        for (name, as_now) in [
            ("regraph_old_build", false),
            ("regraph_current_build", true),
        ] {
            let store: Arc<dyn slatedb::object_store::ObjectStore> =
                Arc::new(slatedb::object_store::memory::InMemory::new());
            let _store = backend_any::set_lsm_test_object_store(store);
            let dir = TempDir::new().unwrap();
            let build = || indexed_lsm_collection(dir.path(), name, 16, 600);
            let (writer, _config, storage, _rows) = if as_now {
                build()
            } else {
                with_diverse_neighbors_for_test(None, build)
            };
            let gate = writer.point_mutation_gate(name);
            let regraph = |skip_current| {
                let status =
                    regraph_segments(&storage, name, &spaces, &hnsw, &gate, skip_current).unwrap();
                (status.segments_rebuilt, status.segments_current)
            };
            seen.push((name, [regraph(true), regraph(true), regraph(false)]));
        }
        assert_eq!(
            seen,
            [
                ("regraph_old_build", [(1, 0), (0, 1), (1, 0)]),
                ("regraph_current_build", [(0, 1), (0, 1), (1, 0)]),
            ]
        );
    }

    /// An LSM writer collection `name` with `n` random `dim`-wide rows (ids
    /// 1..=n, payload `path`), sealed and indexed into one HNSW segment.
    fn indexed_lsm_collection(
        dir: &std::path::Path,
        name: &str,
        dim: usize,
        n: usize,
    ) -> (
        Arc<CollectionManager>,
        Config,
        Arc<HelixGraphStorage>,
        Vec<Vec<f32>>,
    ) {
        let (writer, config, storage, rows) = sealed_lsm_collection(dir, name, &["dense"], dim, n);
        run_index_job(IndexJob {
            collection: name.into(),
            storage: Arc::downgrade(&storage),
            spaces_to_check: vec!["dense".into()],
            hnsw_config: HNSWConfig::new(Some(16), Some(64), Some(128)),
            flat_scan_threshold: 1_000,
        });
        (writer, config, storage, rows)
    }

    /// `indexed_lsm_collection` before its index job: the same rows in each of
    /// `spaces`, every space sealed into one `Building` segment.
    fn sealed_lsm_collection(
        dir: &std::path::Path,
        name: &str,
        spaces: &[&str],
        dim: usize,
        n: usize,
    ) -> (
        Arc<CollectionManager>,
        Config,
        Arc<HelixGraphStorage>,
        Vec<Vec<f32>>,
    ) {
        use rand::{Rng, SeedableRng};
        let mut config =
            Config::new(16, 64, 128, 1).with_storage_backend(StorageBackendConfig::Lsm {
                storage: LsmStorage::ObjectStore,
                role: LsmRole::Writer,
            });
        config.vector_config.flat_scan_threshold = Some(0);
        let writer = Arc::new(CollectionManager::new(dir.to_path_buf(), config.clone()).unwrap());
        apply_mutation(
            &writer,
            &config,
            &ReplicatedMutation::CreateCollection {
                name: name.into(),
                vectors: spaces
                    .iter()
                    .map(|space| {
                        (
                            space.to_string(),
                            NamedVectorConfig {
                                size: dim,
                                distance: DistanceMetric::Cosine,
                                spindle: Default::default(),
                            },
                        )
                    })
                    .collect(),
                sparse_vectors: HashMap::new(),
                hnsw_overrides: None,
            },
        )
        .unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let rows: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect())
            .collect();
        apply_mutation(
            &writer,
            &config,
            &ReplicatedMutation::UpsertPoints {
                collection: name.into(),
                points: rows
                    .iter()
                    .enumerate()
                    .map(|(i, v)| ReplicatedPoint {
                        id: i as u128 + 1,
                        vectors: spaces
                            .iter()
                            .map(|space| (space.to_string(), v.clone()))
                            .collect(),
                        sparse_vectors: HashMap::new(),
                        payload: HashMap::from([(
                            "path".into(),
                            crate::protocol::value::Value::String(format!("src/f{i}.rs")),
                        )]),
                    })
                    .collect(),
            },
        )
        .unwrap();
        let storage = writer.get_collection(name).unwrap();
        let hnsw = HNSWConfig::new(Some(16), Some(64), Some(128));
        // An index job queued by the create would build the sealed segments.
        wait_for_index_executor(name);
        for space in spaces {
            seal_lsm_dense_tail(&storage, space, &hnsw).unwrap();
        }
        (writer, config, storage, rows)
    }

    /// Waits until the index executor holds no job for collection `name`.
    fn wait_for_index_executor(name: &str) {
        let busy = || {
            let state = INDEX_EXECUTOR.state.0.lock().unwrap();
            state.queued_jobs.contains_key(name) || state.active.contains(name)
        };
        let deadline = Instant::now() + Duration::from_secs(30);
        while busy() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert!(!busy(), "index executor still holds a job for {name}");
    }

    fn dense_layout(
        storage: &HelixGraphStorage,
    ) -> Vec<(
        String,
        crate::helix_engine::vector_core::named_vectors::DenseVectorSegmentRole,
    )> {
        storage.named_vectors.list_dense_vector_spaces()["dense"]
            .segments
            .iter()
            .map(|segment| (segment.physical_name.clone(), segment.role))
            .collect()
    }

    fn sole_indexed_segment(storage: &HelixGraphStorage) -> String {
        use crate::helix_engine::vector_core::named_vectors::DenseVectorSegmentRole;
        let indexed: Vec<String> = dense_layout(storage)
            .into_iter()
            .filter(|(_, role)| *role == DenseVectorSegmentRole::Indexed)
            .map(|(name, _)| name)
            .collect();
        assert_eq!(indexed.len(), 1, "{:?}", dense_layout(storage));
        indexed[0].clone()
    }

    /// A merge's prepared graph over every row of `segment`.
    fn prepared_merge_of(
        storage: &HelixGraphStorage,
        segment: &str,
    ) -> crate::helix_engine::vector_core::vector_core::PreparedIndex {
        let exported = {
            let r = storage.backend.begin_read().unwrap();
            storage.named_vectors.cores_read().unwrap()[segment]
                .export_level_zero(&r)
                .unwrap()
        };
        let permit = acquire_build_permit().unwrap();
        VectorCore::build_hnsw_in_memory_owned_with_permit(
            exported,
            &HNSWConfig::new(Some(16), Some(64), Some(128)),
            crate::helix_engine::vector_core::named_vectors::DistanceMetric::Cosine,
            &permit,
        )
        .unwrap()
    }

    /// Every key and value under a dense segment's keyspace.
    fn segment_dump(
        storage: &HelixGraphStorage,
        physical_name: &str,
    ) -> Vec<(usize, Vec<u8>, Vec<u8>)> {
        use crate::helix_engine::storage_core::backend::{KeyRange, SegmentDb};
        let r = storage.backend.begin_read().unwrap();
        let mut rows = Vec::new();
        for (i, db) in [
            SegmentDb::Vectors,
            SegmentDb::VectorData,
            SegmentDb::HnswOut,
            SegmentDb::HnswNeighbors,
            SegmentDb::Ordinals,
            SegmentDb::IvfCentroids,
            SegmentDb::IvfPostings,
            SegmentDb::SimHash,
        ]
        .into_iter()
        .enumerate()
        {
            storage
                .backend
                .scan(
                    &r,
                    Namespace::Segment { physical_name, db },
                    KeyRange::all(),
                    |k, v| {
                        rows.push((i, k.to_vec(), v.to_vec()));
                        true
                    },
                )
                .unwrap();
        }
        rows
    }

    /// Merge targets group-commit their chunks: the same prepared graph
    /// written durably per chunk and group-committed gives byte-identical
    /// segments, and each chunk stops waiting for a WAL flush (prod cadence
    /// and chunk budgets). Merges and regraph share `write_merge_target`.
    #[test]
    fn merge_target_group_commit_writes_the_same_segment_without_per_chunk_flushes() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "merge_group_commit_lsm");
        let _flush = EnvVarGuard::set("HELIX_LSM_FLUSH_INTERVAL_MS", "500");
        let _flat = EnvVarGuard::set("HELIX_MERGE_FLAT_BYTE_BUDGET", "524288");
        let _index = EnvVarGuard::set("HELIX_MERGE_INDEX_BYTE_BUDGET", "524288");
        let _edges = EnvVarGuard::set("HELIX_MERGE_INDEX_HNSW_EDGE_BUDGET", "1024");
        let _group = EnvVarGuard::unset("HELIX_LSM_GROUP_COMMIT");
        let store: Arc<dyn slatedb::object_store::ObjectStore> =
            Arc::new(slatedb::object_store::memory::InMemory::new());
        let _store =
            crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store(store);
        let dir = TempDir::new().unwrap();
        let (_writer, _config, storage, _rows) =
            indexed_lsm_collection(dir.path(), "merge_group_commit", 64, 600);
        let prepared = prepared_merge_of(&storage, &sole_indexed_segment(&storage));
        let hnsw = HNSWConfig::new(Some(16), Some(64), Some(128));
        // (target, write ms, mean chunk ms, chunks, group_flush ms)
        let write = |prepared| {
            let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            let started = std::time::Instant::now();
            let target = metrics::with_local_recorder(&recorder, || {
                write_merge_target(&storage, "dense", &hnsw, prepared)
            })
            .unwrap();
            let write_ms = started.elapsed().as_secs_f64() * 1000.0;
            let (mut sum, mut count, mut flush_ms) = (0.0, 0.0, 0.0);
            for line in handle.render().lines() {
                let value = || line.rsplit(' ').next().unwrap().parse::<f64>().unwrap();
                if line.starts_with("helix_dense_merge_publish_chunk_ms_sum") {
                    sum += value();
                } else if line.starts_with("helix_dense_merge_publish_chunk_ms_count") {
                    count += value();
                } else if line.starts_with("helix_dense_merge_publish_phase_ms_sum")
                    && line.contains("group_flush")
                {
                    flush_ms += value();
                }
            }
            (target, write_ms, sum / count, count, flush_ms)
        };
        let durable = {
            let _off = EnvVarGuard::set("HELIX_LSM_GROUP_COMMIT", "0");
            write(prepared.clone())
        };
        let grouped = write(prepared);
        println!(
            "merge target write, {} chunks: durable chunks {:.1} ms each, {:.0} ms total; \
             group commit {:.2} ms each, {:.0} ms total ({:.0} ms group_flush)",
            durable.3, durable.2, durable.1, grouped.2, grouped.1, grouped.4
        );
        assert_eq!(durable.3, grouped.3);
        assert!(durable.3 >= 10.0, "{durable:?}");
        let dump = segment_dump(&storage, &durable.0);
        assert!(dump.len() > 600);
        assert!(
            segment_dump(&storage, &grouped.0) == dump,
            "the group-committed target differs"
        );
        assert!(
            grouped.2 * 10.0 < durable.2 && grouped.1 * 5.0 < durable.1,
            "durable {durable:?} grouped {grouped:?}"
        );
    }

    /// A delete commits its chunks buffered and one flush makes them durable
    /// before it returns: it stops waiting out the WAL flush interval once per
    /// chunk, and writes no more WAL objects than a durable commit per chunk.
    #[test]
    fn point_write_flush_makes_delete_chunks_durable_without_a_wal_tick_each() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "delete_flush_lsm");
        let _interval = EnvVarGuard::set("HELIX_LSM_FLUSH_INTERVAL_MS", "500");
        let _chunk = EnvVarGuard::unset("HELIX_DELETE_APPLY_CHUNK");
        let _min_chunk = EnvVarGuard::unset("HELIX_DELETE_APPLY_MIN_CHUNK");
        let _upsert_chunk = EnvVarGuard::unset("HELIX_UPSERT_APPLY_CHUNK");
        let _switch = EnvVarGuard::unset("HELIX_LSM_FLUSH_POINT_WRITES");
        let store: Arc<dyn slatedb::object_store::ObjectStore> =
            Arc::new(slatedb::object_store::memory::InMemory::new());
        let _store = crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store(
            Arc::clone(&store),
        );
        let dir = TempDir::new().unwrap();
        let name = "delete_flush";
        let (writer, config, storage, _rows) = indexed_lsm_collection(dir.path(), name, 16, 1_100);
        let wal_objects = || {
            use futures::TryStreamExt;
            futures::executor::block_on(store.list(None).try_collect::<Vec<_>>())
                .unwrap()
                .iter()
                .filter(|object| object.location.as_ref().contains("/wal/"))
                .count()
        };
        // (elapsed, WAL objects written) for a delete of 512 ids: eight chunks.
        let delete = |first: u128| {
            let before = wal_objects();
            let started = std::time::Instant::now();
            apply_mutation(
                &writer,
                &config,
                &ReplicatedMutation::DeletePoints {
                    collection: name.into(),
                    ids: (first..first + 512).collect(),
                },
            )
            .unwrap();
            (started.elapsed(), wal_objects() - before)
        };
        let waited = {
            let _off = EnvVarGuard::set("HELIX_LSM_FLUSH_POINT_WRITES", "0");
            delete(1)
        };
        // The delete above returned on a WAL tick, so the next tick is a whole
        // interval away: a WAL object written by this one is its own flush.
        let flushed = delete(513);
        println!(
            "delete of 512 ids: a durable commit per chunk {:?} and {} WAL objects; \
             buffered chunks and one flush {:?} and {} WAL objects",
            waited.0, waited.1, flushed.0, flushed.1
        );
        let read = storage.backend.begin_read().unwrap();
        for id in [1u128, 512, 513, 1_024] {
            assert!(storage.get_node_be(&read, &id).is_err(), "{id} is deleted");
        }
        assert!(storage.get_node_be(&read, &1_025).is_ok());
        assert!(
            waited.0 >= std::time::Duration::from_millis(2_500),
            "{waited:?}"
        );
        assert!(
            flushed.0 * 5 < waited.0,
            "waited {waited:?}, flushed {flushed:?}"
        );
        assert!(waited.1 >= 8, "a WAL object per durable chunk: {waited:?}");
        assert!(
            flushed.1 >= 1 && flushed.1 <= waited.1,
            "WAL objects: waited {waited:?}, flushed {flushed:?}"
        );
    }

    /// A merge or regraph that dies after its buffered target chunks, before
    /// the group-commit flush, leaves phase A's marked target; after a
    /// restart the index job reaps it. Nothing deleted comes back and the
    /// source keeps every live row.
    #[test]
    fn unflushed_merge_target_is_reaped_after_a_crash() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "merge_crash_lsm");
        let _group = EnvVarGuard::unset("HELIX_LSM_GROUP_COMMIT");
        let store: Arc<dyn slatedb::object_store::ObjectStore> =
            Arc::new(slatedb::object_store::memory::InMemory::new());
        let _store =
            crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store(store);
        let name = "merge_crash";
        let hnsw = HNSWConfig::new(Some(16), Some(64), Some(128));
        let dir = TempDir::new().unwrap();
        let (writer, config, storage, rows) = indexed_lsm_collection(dir.path(), name, 16, 1200);
        let source = sole_indexed_segment(&storage);
        let mut prepared = prepared_merge_of(&storage, &source);
        let deleted: Vec<u128> = (1..=20).collect();
        apply_mutation(
            &writer,
            &config,
            &ReplicatedMutation::DeletePoints {
                collection: name.into(),
                ids: deleted.clone(),
            },
        )
        .unwrap();
        storage.backend.flush_durable().unwrap();

        // `write_merge_target` up to its final flush: phase A durable and
        // marked, then every row and edge (deleted ones included) buffered.
        let target = storage
            .named_vectors
            .create_merge_target_lsm(
                &storage.named_vectors.begin_dense_layout_change(),
                storage.collection_path(),
                "dense",
                hnsw.clone(),
            )
            .unwrap();
        storage
            .with_write_backend(|w| {
                storage
                    .named_vectors
                    .mark_merge_target_be(w, &target)
                    .map_err(GraphError::from)?;
                storage.set_dense_vector_spaces_metadata_be(
                    w,
                    storage.named_vectors.list_dense_vector_spaces(),
                )
            })
            .unwrap();
        let total = prepared.point_ids.len();
        storage
            .with_write_backend_buffered(|w| {
                let nv = &storage.named_vectors;
                nv.flush_merge_flat_chunk_be(w, &target, &mut prepared, 0, total)?;
                nv.flush_merge_index_chunk_be(w, &target, &mut prepared, 0, total)?;
                nv.finalize_merge_index_be(w, &target, &prepared)
                    .map_err(GraphError::from)
            })
            .unwrap();
        // Rows of `segment`, or None once it is gone.
        let rows_in = |storage: &HelixGraphStorage, segment: &str| {
            let r = storage.backend.begin_read().unwrap();
            storage
                .named_vectors
                .cores_read()
                .unwrap()
                .get(segment)
                .map(|core| core.level_zero_count(&r).unwrap())
        };
        assert_eq!(rows_in(&storage, &target), Some(1200));

        // Crash and restart on a fresh pod.
        let restart = |writer: Arc<CollectionManager>, storage: Arc<HelixGraphStorage>| {
            drop(storage);
            drop(writer);
            let dir = TempDir::new().unwrap();
            let writer =
                Arc::new(CollectionManager::new(dir.path().to_path_buf(), config.clone()).unwrap());
            let storage = writer.get_collection(name).unwrap();
            (writer, storage, dir)
        };
        let (_writer, storage, _dir2) = restart(writer, storage);
        // None if the reopen's own maintenance job reaped it first.
        let survived = rows_in(&storage, &target);
        let index_job = |storage: &Arc<HelixGraphStorage>| {
            run_index_job(IndexJob {
                collection: name.into(),
                storage: Arc::downgrade(storage),
                spaces_to_check: vec!["dense".into()],
                hnsw_config: hnsw.clone(),
                flat_scan_threshold: 1_000,
            })
        };
        index_job(&storage);
        let after = dense_layout(&storage);
        assert!(
            !after.iter().any(|(segment, _)| *segment == target),
            "{after:?}"
        );
        assert_eq!(sole_indexed_segment(&storage), source);
        let live: HashSet<u128> = {
            let r = storage.backend.begin_read().unwrap();
            storage.named_vectors.cores_read().unwrap()[&source]
                .export_level_zero(&r)
                .unwrap()
                .into_iter()
                .map(|(id, _, _)| id)
                .collect()
        };
        assert_eq!(live, (21..=1200).collect::<HashSet<u128>>());
        let resurrected: Vec<u128> = deleted
            .iter()
            .copied()
            .filter(|id| {
                let r = storage.backend.begin_read().unwrap();
                storage
                    .named_vectors
                    .dense_search_with_id_filter_ef_be::<fn(u128) -> bool>(
                        &r,
                        "dense",
                        &rows[*id as usize - 1],
                        3,
                        None,
                        false,
                        None,
                        Some(128),
                    )
                    .unwrap()
                    .iter()
                    .any(|hit| hit.get_id() == *id)
            })
            .collect();
        assert!(resurrected.is_empty(), "{resurrected:?}");

        // A full write is durable when it returns: a reader replica, which
        // sees only flushed data, has every row of the unswapped target.
        let written = write_merge_target(
            &storage,
            "dense",
            &hnsw,
            prepared_merge_of(&storage, &source),
        )
        .unwrap();
        let reader_dir = TempDir::new().unwrap();
        let reader = CollectionManager::new(
            reader_dir.path().to_path_buf(),
            Config::new(16, 64, 128, 1).with_storage_backend(StorageBackendConfig::Lsm {
                storage: LsmStorage::ObjectStore,
                role: LsmRole::Reader,
            }),
        )
        .unwrap();
        let replica = reader.get_collection(name).unwrap();
        println!(
            "unflushed target kept {survived:?}/1200 rows across the crash; the written one {:?} on a replica",
            rows_in(&replica, &written)
        );
        assert_eq!(rows_in(&replica, &written), Some(1180));
    }

    /// A replica query pinned before a merge swap keeps searching the source
    /// after another read's reconcile has moved the shared segment list to
    /// the target, which has no rows at its snapshot, and after the
    /// reconciles that follow, for as long as the replica keeps the source's
    /// core. Past that it loses the source and counts it as its own loss,
    /// not as a segment the metadata names and nothing can serve. One source
    /// is swapped for its rebuilt copy, as regraph does; a merge of several
    /// publishes through the same swap.
    #[test]
    fn lsm_reader_read_pinned_before_a_merge_swap_keeps_its_points() {
        use crate::helix_engine::vector_core::named_vectors::PreparedMerge;

        let _guard = ENV_LOCK.lock().unwrap();
        let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "reader_pinned_before_merge");
        let _poll = EnvVarGuard::set("HELIX_LSM_READER_MANIFEST_POLL_MS", "60000");
        let _pinned = EnvVarGuard::set("HELIX_LSM_READER_SNAPSHOT_READS", "1");
        let store: Arc<dyn slatedb::object_store::ObjectStore> =
            Arc::new(slatedb::object_store::memory::InMemory::new());
        let _store =
            crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store(store);
        let name = "pinned_before_merge";
        let hnsw = HNSWConfig::new(Some(16), Some(64), Some(128));
        let dir = TempDir::new().unwrap();
        let (_writer, _config, storage, rows) = indexed_lsm_collection(dir.path(), name, 16, 600);
        let source = sole_indexed_segment(&storage);

        let reader_dir = TempDir::new().unwrap();
        let reader = CollectionManager::new(
            reader_dir.path().to_path_buf(),
            Config::new(16, 64, 128, 1).with_storage_backend(StorageBackendConfig::Lsm {
                storage: LsmStorage::ObjectStore,
                role: LsmRole::Reader,
            }),
        )
        .unwrap();
        let replica = reader.get_collection(name).unwrap();
        replica
            .named_vectors
            .set_retired_core_grace(Duration::from_secs(3600));
        let search = |read: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>| {
            let mut ids = replica
                .named_vectors
                .dense_search_with_id_filter_ef_be::<fn(u128) -> bool>(
                    read,
                    "dense",
                    &rows[0],
                    10,
                    None,
                    false,
                    None,
                    Some(128),
                )
                .unwrap()
                .into_iter()
                .map(|hit| hit.get_id())
                .collect::<Vec<_>>();
            ids.sort_unstable();
            ids
        };
        // Pinned on the source, as a query in its filter stage is.
        let before_merge = replica.backend.begin_read().unwrap();
        let expected = search(&before_merge);
        assert_eq!(expected.len(), 10, "{expected:?}");

        let merge = PreparedMerge {
            merge_targets: vec![source.clone()],
            prepared_index: prepared_merge_of(&storage, &source),
        };
        run_chunked_merge_publish(&storage, "dense", &hnsw, merge, None).unwrap();

        // The view moves (a reopen stands in for the poll), then another
        // read arrives and reconciles.
        replica
            .backend
            .refresh_lsm_reader_with_poll_interval(Duration::from_secs(60))
            .unwrap();
        drop(reader.get_collection(name).unwrap());
        let layout = dense_layout(&replica);
        assert!(
            !layout.iter().any(|(segment, _)| *segment == source),
            "the shared segment list did not follow the view: {layout:?}"
        );
        let after_merge = replica.backend.begin_read().unwrap();
        assert_eq!(search(&after_merge).len(), 10, "the target is not searched");
        assert_eq!(
            search(&before_merge),
            expected,
            "a read pinned before the merge searched the segment list of a newer view"
        );

        // Two more reconciles: the source is off both the list and the one
        // before it, and its core stays for the grace.
        replica.refresh_reader_view_best_effort();
        replica.refresh_reader_view_best_effort();
        assert_eq!(
            search(&before_merge),
            expected,
            "a later reconcile took the source's core from a read inside the grace"
        );

        // Past the grace the core goes with the next reconcile. The search
        // is a filtered one: those run their segments on the calling thread,
        // where the recorder is.
        replica.named_vectors.set_retired_core_grace(Duration::ZERO);
        replica.refresh_reader_view_best_effort();
        use crate::helix_engine::vector_core::named_vectors::{
            take_missing_segment_note, take_retired_segment_sightings,
        };
        // Whatever an earlier test of this process left under the same names.
        let _ = take_retired_segment_sightings(name, &source);
        let _ = take_missing_segment_note(name, "dense", &source);
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let any = |_: u128| true;
        let late = metrics::with_local_recorder(&recorder, || {
            replica
                .named_vectors
                .dense_search_with_id_filter_ef_be(
                    &before_merge,
                    "dense",
                    &rows[0],
                    10,
                    Some(&[any][..]),
                    false,
                    None,
                    Some(128),
                )
                .unwrap()
        });
        assert_eq!(late.len(), 0, "the source was searched past its grace");
        let counted = handle.render();
        assert!(
            counted.contains("helix_dense_read_segment_list_total{outcome=\"own_core_missing\"} 1"),
            "{counted}"
        );
        assert!(
            !counted.contains("helix_dense_segment_missing_total"),
            "{counted}"
        );
        // The read searched its own list and lost one segment of it. A list
        // that had dropped the segment beforehand would count one or the other.
        assert!(
            counted.contains("helix_dense_read_segment_list_total{outcome=\"own\"} 1"),
            "{counted}"
        );

        // Every other dense path loses the source the same way, each at its
        // own place, and none warns of a segment the metadata names and
        // nothing can serve. The unfiltered and the candidate-set search run
        // their segments on worker threads, so both registries are asked.
        let vectors = &replica.named_vectors;
        let ids: std::collections::HashSet<u128> = expected.iter().copied().collect();
        let mut seen = Vec::new();
        let mut ask = |path: &'static str| {
            seen.push((
                path,
                take_retired_segment_sightings(name, &source),
                take_missing_segment_note(name, "dense", &source),
            ));
        };
        ask("filtered search");
        assert_eq!(search(&before_merge).len(), 0, "unfiltered search");
        ask("unfiltered search");
        let hits = vectors
            .dense_search_candidate_set_filter_ef_be(
                &before_merge,
                "dense",
                &rows[0],
                10,
                &ids,
                false,
                None,
                Some(128),
            )
            .unwrap();
        assert_eq!(hits.len(), 0, "candidate-set search");
        ask("candidate-set search");
        let hits = vectors
            .dense_search_candidate_ids_exact_be(
                &before_merge,
                "dense",
                &rows[0],
                10,
                expected.iter().copied(),
                None,
            )
            .unwrap();
        assert_eq!(hits.len(), 0, "exact-id search");
        ask("exact-id search");
        // The scan declines at a segment without a sidecar; the source comes
        // before that one.
        let _ = vectors.dense_search_broad_prefilter_scan_be::<fn(u128) -> bool>(
            &before_merge,
            "dense",
            &rows[0],
            10,
            &ids,
            None,
            None,
        );
        ask("broad prefilter scan");
        assert_eq!(
            seen,
            [
                ("filtered search", vec![("search", true)], false),
                ("unfiltered search", vec![("search", false)], false),
                (
                    "candidate-set search",
                    vec![("candidate_filter_search", false)],
                    false
                ),
                ("exact-id search", vec![("exact_id_search", true)], false),
                (
                    "broad prefilter scan",
                    vec![("broad_prefilter_scan", true)],
                    false
                ),
            ],
            "per path: the sites that found the source retired and without a core (true: on \
             the calling thread), then whether the missing-segment warning fired"
        );
    }

    /// A writer search whose read began before a merge target was written
    /// still finds the merged points after the swap. The publish swaps the
    /// target in for its sources in the writer's layout; that read lists the
    /// target, which has no rows at its snapshot, and no longer lists the
    /// sources, which still hold the points at its snapshot.
    #[test]
    fn lsm_writer_read_begun_before_a_merge_swap_finds_the_merged_points() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let mut config = Config::new(8, 32, 64, 1).with_lsm_in_memory();
        // 0 keeps upserts from queueing background index jobs.
        config.vector_config.flat_scan_threshold = Some(0);
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
                points: (1..=6u128)
                    .map(|id| ReplicatedPoint {
                        id,
                        vectors: HashMap::from([("dense".into(), vec![id as f32, 1.0, 0.0, 0.0])]),
                        sparse_vectors: HashMap::new(),
                        payload: HashMap::new(),
                    })
                    .collect(),
            },
        )
        .unwrap();
        let storage = collections.get_collection("repo").unwrap();
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let query = [0.0, 1.0, 0.0, 0.0];
        // Ids from every LSM dense path for one read, sorted.
        let found = |read: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>| {
            let nv = &storage.named_vectors;
            let ids = (1..=99u128).collect::<HashSet<_>>();
            [
                nv.dense_search_with_id_filter_ef_be::<fn(u128) -> bool>(
                    read, "dense", &query, 10, None, true, None, None,
                ),
                nv.dense_search_candidate_set_filter_ef_be(
                    read, "dense", &query, 10, &ids, true, None, None,
                ),
                nv.dense_search_candidate_ids_exact_be(
                    read,
                    "dense",
                    &query,
                    10,
                    ids.iter().copied(),
                    None,
                ),
            ]
            .into_iter()
            .map(|hits| {
                let mut ids = hits
                    .unwrap()
                    .into_iter()
                    .map(|v| v.get_id())
                    .collect::<Vec<_>>();
                ids.sort_unstable();
                ids
            })
            .collect::<Vec<_>>()
        };
        let all = vec![(1..=6u128).collect::<Vec<_>>(); 3];

        // The points in one published segment: seal the tail and promote it.
        let source = storage
            .named_vectors
            .mutable_dense_segment_name("dense")
            .unwrap();
        assert!(seal_lsm_dense_tail(&storage, "dense", &hnsw).unwrap());
        run_index_job(IndexJob {
            collection: "repo".to_string(),
            storage: Arc::downgrade(&storage),
            spaces_to_check: vec!["dense".to_string()],
            hnsw_config: hnsw.clone(),
            flat_scan_threshold: 1_000,
        });
        assert!(!storage
            .named_vectors
            .building_segment_names("dense")
            .contains(&source));

        // A request that has its snapshot and has not listed the segments yet.
        let before = storage.backend.begin_read().unwrap();
        assert_eq!(found(&before), all, "before the merge");

        // The merge: a target that holds the source's rows, then the swap.
        let target = storage
            .named_vectors
            .create_merge_target_lsm(
                &storage.named_vectors.begin_dense_layout_change(),
                storage.collection_path(),
                "dense",
                hnsw,
            )
            .unwrap();
        storage
            .with_write_backend(|w| {
                let cores = storage.named_vectors.cores_read().unwrap();
                let core = cores.get(&target).unwrap();
                for id in 1..=6u128 {
                    core.insert_flat_be(w, &[id as f32, 1.0, 0.0, 0.0], Some(id), None)
                        .map_err(GraphError::from)?;
                }
                drop(cores);
                storage.set_dense_vector_spaces_metadata_be(
                    w,
                    storage.named_vectors.list_dense_vector_spaces(),
                )
            })
            .unwrap();
        let swap_read = storage.backend.begin_read().unwrap();
        storage
            .named_vectors
            .publish_merge_target_be(&swap_read, "dense", &target, &[source.clone()])
            .unwrap();
        drop(swap_read);
        storage
            .with_write_backend(|w| {
                storage.set_dense_vector_spaces_metadata_be(
                    w,
                    storage.named_vectors.list_dense_vector_spaces(),
                )
            })
            .unwrap();

        let after = storage.backend.begin_read().unwrap();
        assert_eq!(found(&after), all, "a read begun after the swap");
        assert_eq!(
            found(&before),
            all,
            "a read begun before the merge target was written lost the merged points"
        );
    }

    /// Points `1..=6` with vectors `[id, second, 0, 0]` in collection `name`.
    fn upsert_six_points(
        collections: &Arc<CollectionManager>,
        config: &Config,
        name: &str,
        second: f32,
    ) {
        apply_mutation(
            collections,
            config,
            &ReplicatedMutation::UpsertPoints {
                collection: name.into(),
                points: (1..=6u128)
                    .map(|id| ReplicatedPoint {
                        id,
                        vectors: HashMap::from([(
                            "dense".into(),
                            vec![id as f32, second, 0.0, 0.0],
                        )]),
                        sparse_vectors: HashMap::new(),
                        payload: HashMap::new(),
                    })
                    .collect(),
            },
        )
        .unwrap();
    }

    /// A writer over an in-memory LSM store whose collection `name` holds the
    /// six points in its mutable tail. Its upserts queue no index job.
    fn six_point_lsm_writer(
        name: &str,
    ) -> (
        TempDir,
        Config,
        Arc<CollectionManager>,
        Arc<HelixGraphStorage>,
    ) {
        let temp_dir = TempDir::new().unwrap();
        let mut config = Config::new(8, 32, 64, 1).with_lsm_in_memory();
        config.vector_config.flat_scan_threshold = Some(0);
        let collections =
            Arc::new(CollectionManager::new(temp_dir.path().join("data"), config.clone()).unwrap());
        apply_mutation(
            &collections,
            &config,
            &ReplicatedMutation::CreateCollection {
                name: name.into(),
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
        upsert_six_points(&collections, &config, name, 1.0);
        let storage = collections.get_collection(name).unwrap();
        (temp_dir, config, collections, storage)
    }

    /// Seals the tail of `dense` and builds it: the points sit in one
    /// published segment, whose name this returns.
    fn publish_dense_tail(name: &str, storage: &Arc<HelixGraphStorage>) -> String {
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let sealed = storage
            .named_vectors
            .mutable_dense_segment_name("dense")
            .unwrap();
        assert!(seal_lsm_dense_tail(storage, "dense", &hnsw).unwrap());
        run_index_job(IndexJob {
            collection: name.to_string(),
            storage: Arc::downgrade(storage),
            spaces_to_check: vec!["dense".to_string()],
            hnsw_config: hnsw,
            flat_scan_threshold: 1_000,
        });
        assert!(!storage
            .named_vectors
            .building_segment_names("dense")
            .contains(&sealed));
        sealed
    }

    /// The ids each of the three LSM dense paths returns for `read`, sorted.
    fn dense_path_ids(
        storage: &HelixGraphStorage,
        read: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    ) -> Vec<Vec<u128>> {
        let nv = &storage.named_vectors;
        let query = [0.0, 1.0, 0.0, 0.0];
        let ids = (1..=99u128).collect::<HashSet<_>>();
        [
            nv.dense_search_with_id_filter_ef_be::<fn(u128) -> bool>(
                read, "dense", &query, 10, None, true, None, None,
            ),
            nv.dense_search_candidate_set_filter_ef_be(
                read, "dense", &query, 10, &ids, true, None, None,
            ),
            nv.dense_search_candidate_ids_exact_be(
                read,
                "dense",
                &query,
                10,
                ids.iter().copied(),
                None,
            ),
        ]
        .into_iter()
        .map(|hits| {
            let mut ids = hits
                .unwrap()
                .into_iter()
                .map(|v| v.get_id())
                .collect::<Vec<_>>();
            ids.sort_unstable();
            ids
        })
        .collect()
    }

    /// Runs `search` and returns its answer with the lines the counter of
    /// searches answered on a newer read left.
    fn with_newer_read_lines<T>(search: impl FnOnce() -> T) -> (T, Vec<String>) {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let answer = metrics::with_local_recorder(&recorder, search);
        let lines = handle
            .render()
            .lines()
            .filter(|line| line.starts_with("helix_dense_search_newer_read_total"))
            .map(str::to_string)
            .collect();
        (answer, lines)
    }

    /// A seal and the index build after it move no writer search to a newer
    /// read: the list they leave is still right for a read begun before them.
    #[test]
    fn lsm_seal_and_index_build_leave_an_older_writer_read_on_its_own_read() {
        let _guard = ENV_LOCK.lock().unwrap();
        let (_dir, _config, _collections, storage) = six_point_lsm_writer("repo");
        let before = storage.backend.begin_read().unwrap();
        publish_dense_tail("repo", &storage);

        let (found, newer_read) = with_newer_read_lines(|| dense_path_ids(&storage, &before));
        assert_eq!(found, vec![(1..=6u128).collect::<Vec<_>>(); 3]);
        assert!(newer_read.is_empty(), "{newer_read:?}");
    }

    /// A segment whose points were all rewritten is dropped from the writer's
    /// list as empty. A read begun before the rewrites still has its rows
    /// there, and its search finds the points all the same.
    #[test]
    fn lsm_writer_read_begun_before_an_emptied_segment_is_dropped_finds_its_points() {
        let _guard = ENV_LOCK.lock().unwrap();
        let (_dir, config, collections, storage) = six_point_lsm_writer("repo");
        let source = publish_dense_tail("repo", &storage);
        let all = vec![(1..=6u128).collect::<Vec<_>>(); 3];
        let before = storage.backend.begin_read().unwrap();
        assert_eq!(
            dense_path_ids(&storage, &before),
            all,
            "before the rewrites"
        );

        // Every point rewritten: the segment is empty and leaves the list.
        upsert_six_points(&collections, &config, "repo", 2.0);
        let located = storage.backend.begin_read().unwrap();
        let empties = storage
            .named_vectors
            .cleanup_empty_dense_segments_locate_be(&located, "dense")
            .unwrap();
        drop(located);
        assert!(empties.contains(&source), "{empties:?}");
        assert!(storage
            .named_vectors
            .apply_dense_segment_removal("dense", &empties)
            .unwrap());

        let after = storage.backend.begin_read().unwrap();
        assert_eq!(
            dense_path_ids(&storage, &after),
            all,
            "a read begun after the drop"
        );
        let (found, newer_read) = with_newer_read_lines(|| dense_path_ids(&storage, &before));
        assert_eq!(
            found, all,
            "a read begun before the rewrites lost the points"
        );
        assert!(
            !newer_read.is_empty(),
            "the older read was not moved to a newer one"
        );
    }

    /// The real merge, then the reaper. A writer search on a read begun
    /// before the merge answers like one begun after it, right after the swap
    /// and again once the reaper has taken the source's core.
    #[test]
    fn lsm_writer_read_begun_before_a_real_merge_answers_like_one_begun_after() {
        use crate::helix_engine::vector_core::named_vectors::PreparedMerge;

        let _guard = ENV_LOCK.lock().unwrap();
        let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "writer_read_before_real_merge");
        let store: Arc<dyn slatedb::object_store::ObjectStore> =
            Arc::new(slatedb::object_store::memory::InMemory::new());
        let _store =
            crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store(store);
        let hnsw = HNSWConfig::new(Some(16), Some(64), Some(128));
        let dir = TempDir::new().unwrap();
        let (_writer, _config, storage, rows) =
            indexed_lsm_collection(dir.path(), "before_real_merge", 16, 600);
        let source = sole_indexed_segment(&storage);
        let search = |read: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>| {
            let mut ids = storage
                .named_vectors
                .dense_search_with_id_filter_ef_be::<fn(u128) -> bool>(
                    read,
                    "dense",
                    &rows[0],
                    10,
                    None,
                    false,
                    None,
                    Some(128),
                )
                .unwrap()
                .into_iter()
                .map(|hit| hit.get_id())
                .collect::<Vec<_>>();
            ids.sort_unstable();
            ids
        };
        let before = storage.backend.begin_read().unwrap();
        assert_eq!(search(&before).len(), 10, "before the merge");

        let merge = PreparedMerge {
            merge_targets: vec![source.clone()],
            prepared_index: prepared_merge_of(&storage, &source),
        };
        run_chunked_merge_publish(&storage, "dense", &hnsw, merge, None).unwrap();

        let after = storage.backend.begin_read().unwrap();
        let expected = search(&after);
        assert_eq!(expected.len(), 10, "the merge target is not searched");
        assert_eq!(search(&before), expected, "right after the swap");

        let reaped = || {
            !storage
                .named_vectors
                .cores_read()
                .unwrap()
                .contains_key(&source)
        };
        let deadline = Instant::now() + Duration::from_secs(30);
        while !reaped() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(20));
        }
        assert!(reaped(), "the reaper did not take the merge source");
        assert_eq!(search(&before), expected, "once the reaper took the source");
    }

    /// A sealed tail whose points are all rewritten before its build leaves
    /// the list through the orphan clean-up at the top of an index pass. A
    /// read begun before the rewrites still has its rows there, and its
    /// search finds the points all the same.
    #[test]
    fn lsm_writer_read_begun_before_an_emptied_sealed_tail_is_dropped_finds_its_points() {
        let _guard = ENV_LOCK.lock().unwrap();
        let (_dir, config, collections, storage) = six_point_lsm_writer("repo");
        let hnsw = HNSWConfig::new(Some(8), Some(32), Some(64));
        let all = vec![(1..=6u128).collect::<Vec<_>>(); 3];
        let sealed = storage
            .named_vectors
            .mutable_dense_segment_name("dense")
            .unwrap();
        assert!(seal_lsm_dense_tail(&storage, "dense", &hnsw).unwrap());
        let before = storage.backend.begin_read().unwrap();
        assert_eq!(
            dense_path_ids(&storage, &before),
            all,
            "before the rewrites"
        );

        // Every point rewritten while the sealed tail waits for its build.
        upsert_six_points(&collections, &config, "repo", 2.0);
        let located = storage.backend.begin_read().unwrap();
        let orphans = storage
            .named_vectors
            .gc_orphan_building_segments_deferred_be(&located, "dense")
            .unwrap();
        drop(located);
        assert_eq!(
            orphans,
            vec![sealed],
            "the emptied sealed tail did not leave as an orphan"
        );

        let after = storage.backend.begin_read().unwrap();
        assert_eq!(
            dense_path_ids(&storage, &after),
            all,
            "a read begun after the drop"
        );
        let (found, newer_read) = with_newer_read_lines(|| dense_path_ids(&storage, &before));
        assert_eq!(
            found, all,
            "a read begun before the rewrites lost the points"
        );
        assert!(
            !newer_read.is_empty(),
            "the older read was not moved to a newer one"
        );
    }

    /// Index publishing group-commits its chunks: the same prepared graph
    /// published with a durable commit per chunk and group-committed gives
    /// byte-identical segments, with one durable flush before the entry point
    /// instead of a WAL-tick wait per chunk.
    #[test]
    fn indices_publish_group_commit_writes_the_same_segment_without_per_chunk_flushes() {
        use crate::helix_engine::vector_core::named_vectors::DenseVectorSegmentRole;

        let _guard = ENV_LOCK.lock().unwrap();
        let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "indices_group_commit_lsm");
        let _flush = EnvVarGuard::set("HELIX_LSM_FLUSH_INTERVAL_MS", "200");
        let _index = EnvVarGuard::set("HELIX_MERGE_INDEX_BYTE_BUDGET", "524288");
        let _edges = EnvVarGuard::set("HELIX_MERGE_INDEX_HNSW_EDGE_BUDGET", "1024");
        let _group = EnvVarGuard::unset("HELIX_LSM_GROUP_COMMIT");
        let store: Arc<dyn slatedb::object_store::ObjectStore> =
            Arc::new(slatedb::object_store::memory::InMemory::new());
        let _store =
            crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store(store);
        let dir = TempDir::new().unwrap();
        // Two spaces with the same rows: two Building segments for one graph.
        let (_writer, _config, storage, _rows) = sealed_lsm_collection(
            dir.path(),
            "indices_group_commit",
            &["durable", "grouped"],
            64,
            300,
        );
        let building = |space: &str| {
            let names = storage.named_vectors.building_segment_names(space);
            assert_eq!(names.len(), 1, "{space}: {names:?}");
            names[0].clone()
        };
        let (durable_segment, grouped_segment) = (building("durable"), building("grouped"));
        let graph = {
            let r = storage.backend.begin_read().unwrap();
            let mut prepared = storage
                .named_vectors
                .prepare_building_indices_limited_be(&r, "durable", 1)
                .unwrap();
            assert_eq!(prepared.len(), 1);
            prepared.remove(0).1
        };
        // (publish ms, mean chunk ms, chunks, group flushes)
        let publish = |space: &str, segment: &str, graph| {
            let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            let started = std::time::Instant::now();
            let published = metrics::with_local_recorder(&recorder, || {
                run_chunked_indices_publish(&storage, space, vec![(segment.to_string(), graph)])
            })
            .unwrap();
            assert!(published);
            let publish_ms = started.elapsed().as_secs_f64() * 1000.0;
            let (mut sum, mut count, mut flushes) = (0.0, 0.0, 0.0);
            for line in handle.render().lines() {
                let value = || line.rsplit(' ').next().unwrap().parse::<f64>().unwrap();
                if line.starts_with("helix_dense_merge_publish_chunk_ms_sum") {
                    sum += value();
                } else if line.starts_with("helix_dense_merge_publish_chunk_ms_count") {
                    count += value();
                } else if line.starts_with("helix_dense_merge_publish_phase_ms_count")
                    && line.contains("indices_group_flush")
                {
                    flushes += value();
                }
            }
            (publish_ms, sum / count, count, flushes)
        };
        let durable = {
            let _off = EnvVarGuard::set("HELIX_LSM_GROUP_COMMIT", "0");
            publish("durable", &durable_segment, graph.clone())
        };
        let grouped = publish("grouped", &grouped_segment, graph);
        println!(
            "index publish, {} chunks: durable chunks {:.1} ms each, {:.0} ms total; \
             group commit {:.2} ms each, {:.0} ms total",
            durable.2, durable.1, durable.0, grouped.1, grouped.0
        );
        assert_eq!(durable.2, grouped.2);
        assert!(durable.2 >= 6.0, "{durable:?}");
        assert_eq!((durable.3, grouped.3), (0.0, 1.0));
        for (space, segment) in [("durable", &durable_segment), ("grouped", &grouped_segment)] {
            let layout = storage.named_vectors.list_dense_vector_spaces();
            let role = layout[space]
                .segments
                .iter()
                .find(|entry| entry.physical_name == *segment)
                .map(|entry| entry.role);
            assert_eq!(role, Some(DenseVectorSegmentRole::Indexed), "{space}");
        }
        let dump = segment_dump(&storage, &durable_segment);
        assert!(dump.len() > 600);
        assert!(
            segment_dump(&storage, &grouped_segment) == dump,
            "the group-committed segment differs"
        );
        assert!(
            grouped.1 * 10.0 < durable.1 && grouped.0 * 3.0 < durable.0,
            "durable {durable:?} grouped {grouped:?}"
        );
    }

    /// An index publish that dies part-way leaves a `Building` segment with no
    /// entry point, whatever part of its graph reached the object store: a
    /// replica keeps scanning it flat and a restarted writer's index job
    /// rebuilds it. Only a segment whose entry point is durable is promoted as
    /// it is, and that one has its whole graph.
    #[test]
    fn interrupted_indices_publish_never_leaves_a_partly_indexed_segment() {
        use crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store;
        use crate::helix_engine::vector_core::named_vectors::DenseVectorSegmentRole;
        use slatedb::object_store::{memory::InMemory, ObjectStore};

        let _guard = ENV_LOCK.lock().unwrap();
        let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "indices_crash_lsm");
        let _group = EnvVarGuard::unset("HELIX_LSM_GROUP_COMMIT");
        let store = Arc::new(InMemory::new());
        let _store = set_lsm_test_object_store(Arc::clone(&store) as Arc<dyn ObjectStore>);
        let name = "indices_crash";
        let points = 600usize;
        let hnsw = HNSWConfig::new(Some(16), Some(64), Some(128));
        let dir = TempDir::new().unwrap();
        let (writer, config, storage, _rows) =
            sealed_lsm_collection(dir.path(), name, &["dense"], 16, points);
        let segment = storage
            .named_vectors
            .building_segment_names("dense")
            .remove(0);
        let mut graph = {
            let r = storage.backend.begin_read().unwrap();
            let mut prepared = storage
                .named_vectors
                .prepare_building_indices_limited_be(&r, "dense", 1)
                .unwrap();
            assert_eq!(prepared.len(), 1);
            prepared.remove(0).1
        };
        let total = graph.point_ids.len();
        assert_eq!(total, points);
        // The segment's neighbor lists, keyed `id ++ level`.
        let edges = |storage: &HelixGraphStorage| -> Vec<(Vec<u8>, Vec<u8>)> {
            segment_dump(storage, &segment)
                .into_iter()
                .filter(|(db, _, _)| *db == 3)
                .map(|(_, key, value)| (key, value))
                .collect()
        };
        let role = |storage: &HelixGraphStorage| {
            dense_layout(storage)
                .into_iter()
                .find(|(physical_name, _)| *physical_name == segment)
                .map(|(_, role)| role)
        };
        let has_entry_point = |storage: &HelixGraphStorage| {
            let r = storage.backend.begin_read().unwrap();
            storage.named_vectors.cores_read().unwrap()[&segment]
                .has_index(&r)
                .unwrap()
        };

        // The steps of `run_chunked_indices_publish`, with a copy of the object
        // store at each point the process could die: what a restart finds.
        let mut publish_chunks = |start: usize, end: usize| {
            storage
                .with_write_backend_buffered(|w| {
                    storage
                        .named_vectors
                        .flush_merge_index_chunk_be(w, &segment, &mut graph, start, end)
                        .map_err(GraphError::from)
                })
                .unwrap();
        };
        // Half of the chunks reached the WAL on a flush tick, the rest did not.
        publish_chunks(0, total / 2);
        storage.backend.flush_durable().unwrap();
        let half_edges = edges(&storage);
        let died_between_chunks = store.fork();
        // Every chunk is durable (the group-commit flush), no entry point yet.
        publish_chunks(total / 2, total);
        storage.backend.flush_durable().unwrap();
        let all_edges = edges(&storage);
        let died_before_entry_point = store.fork();
        // The entry point is durable; the layout still says Building.
        storage
            .with_write_backend(|w| {
                storage
                    .named_vectors
                    .finalize_merge_index_be(w, &segment, &graph)
                    .map_err(GraphError::from)
            })
            .unwrap();
        let died_before_promotion = store.fork();
        assert!(!half_edges.is_empty() && half_edges.len() < all_edges.len());
        drop(storage);
        drop(writer);

        for (crash, store, durable_edges, entry_point_durable) in [
            ("between chunks", died_between_chunks, &half_edges, false),
            (
                "before the entry point",
                died_before_entry_point,
                &all_edges,
                false,
            ),
            ("before promotion", died_before_promotion, &all_edges, true),
        ] {
            let _crashed = set_lsm_test_object_store(Arc::new(store));
            // A replica sees exactly what was durable: a Building segment, with
            // an entry point only once its whole graph is there.
            {
                let reader_dir = TempDir::new().unwrap();
                let reader = CollectionManager::new(
                    reader_dir.path().to_path_buf(),
                    Config::new(16, 64, 128, 1).with_storage_backend(StorageBackendConfig::Lsm {
                        storage: LsmStorage::ObjectStore,
                        role: LsmRole::Reader,
                    }),
                )
                .unwrap();
                let replica = reader.get_collection(name).unwrap();
                assert_eq!(
                    role(&replica),
                    Some(DenseVectorSegmentRole::Building),
                    "{crash}"
                );
                assert_eq!(has_entry_point(&replica), entry_point_durable, "{crash}");
                assert!(edges(&replica) == *durable_edges, "{crash}");
            }

            // A restarted writer's index job leaves it Indexed with every
            // point's level-0 neighbor list.
            let dir = TempDir::new().unwrap();
            let writer =
                Arc::new(CollectionManager::new(dir.path().to_path_buf(), config.clone()).unwrap());
            let storage = writer.get_collection(name).unwrap();
            wait_for_index_executor(name);
            run_index_job(IndexJob {
                collection: name.into(),
                storage: Arc::downgrade(&storage),
                spaces_to_check: vec!["dense".into()],
                hnsw_config: hnsw.clone(),
                flat_scan_threshold: 1_000,
            });
            wait_for_index_executor(name);
            assert_eq!(
                role(&storage),
                Some(DenseVectorSegmentRole::Indexed),
                "{crash}"
            );
            assert!(has_entry_point(&storage), "{crash}");
            let recovered = edges(&storage);
            let with_level_zero: HashSet<u128> = recovered
                .iter()
                .filter(|(key, _)| key[16..] == 0usize.to_be_bytes()[..])
                .map(|(key, _)| u128::from_be_bytes(key[..16].try_into().unwrap()))
                .collect();
            assert_eq!(
                with_level_zero,
                (1..=points as u128).collect::<HashSet<u128>>(),
                "{crash}"
            );
            // Promoted without a rebuild: the graph is the one published.
            if entry_point_durable {
                assert!(recovered == *durable_edges, "{crash}");
            }
        }
    }

    /// A regraph target is written without the gate, so a delete can
    /// land before its row's chunk is written. If the writer dies before the
    /// swap (or the busy drop's persist fails), the target stays `Building` in
    /// the persisted layout; the next index job must not make it searchable.
    #[test]
    fn dead_regraph_target_is_not_promoted_by_the_index_job() {
        use crate::helix_engine::storage_core::backend_any;
        use crate::helix_engine::vector_core::named_vectors::DenseVectorSegmentRole;
        use crate::helix_engine::vector_core::vector_core::with_diverse_neighbors_for_test;
        use rand::{Rng, SeedableRng};

        let _guard = ENV_LOCK.lock().unwrap();
        let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "regraph_dead_target_lsm");
        let store: Arc<dyn slatedb::object_store::ObjectStore> =
            Arc::new(slatedb::object_store::memory::InMemory::new());
        let _store = backend_any::set_lsm_test_object_store(store);
        let mut config =
            Config::new(16, 64, 128, 1).with_storage_backend(StorageBackendConfig::Lsm {
                storage: LsmStorage::ObjectStore,
                role: LsmRole::Writer,
            });
        config.vector_config.flat_scan_threshold = Some(0);
        let writer_dir = TempDir::new().unwrap();
        let writer = Arc::new(
            CollectionManager::new(writer_dir.path().to_path_buf(), config.clone()).unwrap(),
        );
        let name = "regraph_dead_target";
        let dim = 16;
        apply_mutation(
            &writer,
            &config,
            &ReplicatedMutation::CreateCollection {
                name: name.into(),
                vectors: HashMap::from([(
                    "dense".into(),
                    NamedVectorConfig {
                        size: dim,
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
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let rows: Vec<Vec<f32>> = (0..1200)
            .map(|_| (0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect())
            .collect();
        apply_mutation(
            &writer,
            &config,
            &ReplicatedMutation::UpsertPoints {
                collection: name.into(),
                points: rows
                    .iter()
                    .enumerate()
                    .map(|(i, v)| ReplicatedPoint {
                        id: i as u128 + 1,
                        vectors: HashMap::from([("dense".into(), v.clone())]),
                        sparse_vectors: HashMap::new(),
                        payload: HashMap::new(),
                    })
                    .collect(),
            },
        )
        .unwrap();
        let storage = writer.get_collection(name).unwrap();
        let hnsw = HNSWConfig::new(Some(16), Some(64), Some(128));
        let index_job = || {
            run_index_job(IndexJob {
                collection: name.into(),
                storage: Arc::downgrade(&storage),
                spaces_to_check: vec!["dense".into()],
                hnsw_config: hnsw.clone(),
                flat_scan_threshold: 1_000,
            })
        };
        with_diverse_neighbors_for_test(None, || {
            seal_lsm_dense_tail(&storage, "dense", &hnsw).unwrap();
            index_job();
        });
        let layout = || {
            storage.named_vectors.list_dense_vector_spaces()["dense"]
                .segments
                .iter()
                .map(|segment| (segment.physical_name.clone(), segment.role))
                .collect::<Vec<_>>()
        };
        let source: Vec<String> = layout()
            .into_iter()
            .filter(|(_, role)| *role == DenseVectorSegmentRole::Indexed)
            .map(|(name, _)| name)
            .collect();
        assert_eq!(source.len(), 1, "{:?}", layout());

        // Regraph's export snapshot and build.
        let r = storage.backend.begin_read().unwrap();
        let exported = storage
            .named_vectors
            .export_for_regraph_be(&r, "dense", &source[0], 512)
            .unwrap()
            .unwrap()
            .rows;
        drop(r);
        let prepared = {
            let permit =
                crate::helix_engine::vector_core::vector_core::acquire_build_permit().unwrap();
            VectorCore::build_hnsw_in_memory_owned_with_permit(
                exported,
                &hnsw,
                crate::helix_engine::vector_core::named_vectors::DistanceMetric::Cosine,
                &permit,
            )
            .unwrap()
        };
        // Deletes landing during the ungated target write, ahead of their rows' chunks.
        let deleted: Vec<u128> = (1..=20).collect();
        apply_mutation(
            &writer,
            &config,
            &ReplicatedMutation::DeletePoints {
                collection: name.into(),
                ids: deleted.clone(),
            },
        )
        .unwrap();
        let target = write_merge_target(&storage, "dense", &hnsw, prepared).unwrap();
        // The writer dies here: no swap, no busy drop. Its next index job runs.
        index_job();
        let search_top = |query: &[f32]| {
            let r = storage.backend.begin_read().unwrap();
            storage
                .named_vectors
                .dense_search_with_id_filter_ef_be::<fn(u128) -> bool>(
                    &r,
                    "dense",
                    query,
                    3,
                    None,
                    false,
                    None,
                    Some(128),
                )
                .unwrap()
                .into_iter()
                .map(|v| v.get_id())
                .collect::<Vec<_>>()
        };
        let resurrected: Vec<u128> = deleted
            .iter()
            .copied()
            .filter(|id| search_top(&rows[*id as usize - 1]).contains(id))
            .collect();
        println!(
            "dead regraph target {target}: layout {:?}; resurrected {}/{}",
            layout(),
            resurrected.len(),
            deleted.len()
        );
        assert!(
            resurrected.is_empty(),
            "deleted ids came back: {resurrected:?}"
        );
        assert!(
            !layout().iter().any(|(n, _)| *n == target),
            "dead regraph target still in the layout: {:?}",
            layout()
        );
    }

    /// A regraph writes its new segment without the point-mutation gate: an
    /// upsert issued while the target is being written finishes before the
    /// regraph returns, which then stands down as busy, drops its target and
    /// loses nothing.
    #[test]
    fn regraph_write_phase_does_not_block_upserts() {
        use crate::helix_engine::storage_core::backend_any;
        use crate::helix_engine::vector_core::named_vectors::DenseVectorSegmentRole;
        use rand::{Rng, SeedableRng};

        let _guard = ENV_LOCK.lock().unwrap();
        let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "regraph_unblocked_lsm");
        // The smallest chunks: a long run of write-phase commits to land in.
        let _chunks = EnvVarGuard::set("HELIX_MERGE_CHUNK_SIZE", "64");
        let store: Arc<dyn slatedb::object_store::ObjectStore> =
            Arc::new(slatedb::object_store::memory::InMemory::new());
        let _store = backend_any::set_lsm_test_object_store(store);
        let mut config =
            Config::new(16, 64, 128, 1).with_storage_backend(StorageBackendConfig::Lsm {
                storage: LsmStorage::ObjectStore,
                role: LsmRole::Writer,
            });
        config.vector_config.flat_scan_threshold = Some(0);
        let writer_dir = TempDir::new().unwrap();
        let writer = Arc::new(
            CollectionManager::new(writer_dir.path().to_path_buf(), config.clone()).unwrap(),
        );
        let name = "regraph_unblocked";
        let dim = 16;
        apply_mutation(
            &writer,
            &config,
            &ReplicatedMutation::CreateCollection {
                name: name.into(),
                vectors: HashMap::from([(
                    "dense".into(),
                    NamedVectorConfig {
                        size: dim,
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
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let mut random =
            || -> Vec<f32> { (0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect() };
        let upsert = |points: Vec<(u128, Vec<f32>)>| {
            apply_mutation(
                &writer,
                &config,
                &ReplicatedMutation::UpsertPoints {
                    collection: name.into(),
                    points: points
                        .into_iter()
                        .map(|(id, vector)| ReplicatedPoint {
                            id,
                            vectors: HashMap::from([("dense".into(), vector)]),
                            sparse_vectors: HashMap::new(),
                            payload: HashMap::new(),
                        })
                        .collect(),
                },
            )
            .unwrap();
        };
        for chunk in 0..6u128 {
            upsert((1..=500u128).map(|i| (chunk * 500 + i, random())).collect());
        }
        let storage = writer.get_collection(name).unwrap();
        let hnsw = HNSWConfig::new(Some(16), Some(64), Some(128));
        seal_lsm_dense_tail(&storage, "dense", &hnsw).unwrap();
        run_index_job(IndexJob {
            collection: name.into(),
            storage: Arc::downgrade(&storage),
            spaces_to_check: vec!["dense".into()],
            hnsw_config: hnsw.clone(),
            flat_scan_threshold: 1_000,
        });
        let layout = || {
            storage.named_vectors.list_dense_vector_spaces()["dense"]
                .segments
                .iter()
                .map(|segment| (segment.physical_name.clone(), segment.role))
                .collect::<Vec<_>>()
        };
        let indexed = |layout: &[(String, DenseVectorSegmentRole)]| {
            layout
                .iter()
                .filter(|(_, role)| *role == DenseVectorSegmentRole::Indexed)
                .map(|(segment, _)| segment.clone())
                .collect::<Vec<_>>()
        };
        let before = layout();
        assert_eq!(indexed(&before).len(), 1, "{before:?}");
        assert!(before
            .iter()
            .all(|(_, role)| *role != DenseVectorSegmentRole::Building));

        let regraph = {
            let (storage, gate, hnsw) = (
                Arc::clone(&storage),
                writer.point_mutation_gate(name),
                hnsw.clone(),
            );
            std::thread::spawn(move || {
                let status =
                    regraph_collection(&storage, name, &["dense".to_string()], &hnsw, &gate)
                        .unwrap();
                (status, std::time::Instant::now())
            })
        };
        // The target appears as a Building segment once the write phase starts.
        let deadline = std::time::Instant::now() + Duration::from_secs(60);
        let target = loop {
            if let Some((segment, _)) = layout().into_iter().find(|(segment, role)| {
                *role == DenseVectorSegmentRole::Building
                    && !before.iter().any(|(old, _)| old == segment)
            }) {
                break segment;
            }
            assert!(
                std::time::Instant::now() < deadline && !regraph.is_finished(),
                "the regraph target never appeared"
            );
            std::thread::sleep(Duration::from_micros(200));
        };
        let fresh = random();
        let upsert_started = std::time::Instant::now();
        upsert(vec![(9001, fresh.clone())]);
        let upserted = std::time::Instant::now();
        let (status, regraph_returned) = regraph.join().unwrap();
        println!(
            "regraph write phase: upsert took {:?}, regraph returned {:?} after it started; {status:?}",
            upserted - upsert_started,
            regraph_returned.saturating_duration_since(upsert_started)
        );
        assert!(
            upserted < regraph_returned,
            "the upsert waited for the regraph publish"
        );
        assert_eq!(status.state, "busy", "{status:?}");

        // Nothing lost or orphaned: the source still serves, the target is gone.
        let after = layout();
        assert_eq!(indexed(&after), indexed(&before), "{after:?}");
        assert!(
            after
                .iter()
                .all(|(_, role)| *role != DenseVectorSegmentRole::Building),
            "{after:?}"
        );
        let r = storage.backend.begin_read().unwrap();
        assert_eq!(
            storage.named_vectors.cores_read().unwrap()[&indexed(&before)[0]]
                .export_level_zero(&r)
                .unwrap()
                .len(),
            3000
        );
        let hit = storage
            .named_vectors
            .dense_search_with_id_filter_ef_be::<fn(u128) -> bool>(
                &r,
                "dense",
                &fresh,
                1,
                None,
                false,
                None,
                Some(128),
            )
            .unwrap();
        assert_eq!(hit[0].get_id(), 9001);
        drop(r);
        while storage
            .named_vectors
            .cores_read()
            .unwrap()
            .contains_key(&target)
        {
            assert!(
                std::time::Instant::now() < deadline,
                "the dropped target was never reaped"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
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
                unreached: 0,
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
            unreached: 0,
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
            unreached: 0,
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
            "host_path and container_path are unindexed per client contract; \
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
            DistanceMetric::Cosine,
            &build_permit,
        )
        .unwrap();
        drop(build_permit);
        let merge = crate::helix_engine::vector_core::named_vectors::PreparedMerge {
            merge_targets: targets.clone(),
            prepared_index,
        };
        run_chunked_merge_publish(&storage, TEST_DENSE_VECTOR, &hnsw_config, merge, None).unwrap();

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
            DistanceMetric::Cosine,
            &build_permit,
        )
        .unwrap();
        drop(build_permit);
        let merge = crate::helix_engine::vector_core::named_vectors::PreparedMerge {
            merge_targets: targets.clone(),
            prepared_index,
        };
        run_chunked_merge_publish(&storage, TEST_DENSE_VECTOR, &hnsw_config, merge, None).unwrap();

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
    ///   * `get_dense_space_stats_be`            (full per-segment stats; the
    ///     index job now probes with the bounded count behind the gate below)
    ///   * `cleanup_empty_dense_segments_locate_be`
    ///   * `gc_orphan_building_segments_deferred_be`   (NEW in the fix)
    ///   * `optimizer_has_pending_work`          (module-private optimizer gate)
    /// and that the collection is not left degraded afterward.
    #[test]
    fn optimizer_read_helpers_succeed_on_lsm_backend() {
        let _guard = ENV_LOCK.lock().unwrap();

        let temp_dir = TempDir::new().unwrap();
        let mut config = Config::new(8, 32, 64, 1).with_lsm_in_memory();
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
            "all flat vectors must be visible to the dense stats on LSM"
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
