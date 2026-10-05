use crate::helix_engine::{
    graph_core::graph_core::HelixGraphEngine,
    storage_core::{
        backend_lsm::{allow_lsm_blocking, allow_lsm_blocking_cancellable, LsmReadCancellation},
        collection_manager::CollectionManager,
        reader_warm,
        replication::{self, ReplicationManager},
    },
    types::GraphError,
};
use crate::helix_gateway::point_mutation_admission::{self, PointMutationGuard};
use crate::helix_gateway::router::router::HelixRouter;
use crate::helix_gateway::thread_pool::thread_pool::{
    collection_inflight_cap, lsm_reader_write_rejection, observe_collection_inflight_admission,
    route_class, route_label_for, status_class, try_acquire_route_admission, AdmissionGuard,
    RouteClass,
};
use crate::protocol::{
    request::{max_body_bytes, Request as HelixRequest},
    response::Response as HelixResponse,
};
use axum::{
    body::{to_bytes, Body},
    error_handling::HandleErrorLayer,
    extract::State,
    http::{Request as HttpRequest, Response as HttpResponse, StatusCode},
    response::IntoResponse,
    routing::any,
    BoxError, Router,
};
use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex},
    time::{Duration, Instant},
};
use tokio::{
    net::TcpListener,
    sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError},
};
use tower::ServiceBuilder;
use tracing::{debug, error, info, warn};

#[derive(Clone)]
pub struct AsyncGatewayState {
    pub graph: Arc<HelixGraphEngine>,
    pub collections: Arc<CollectionManager>,
    pub replication: Arc<ReplicationManager>,
    pub router: Arc<HelixRouter>,
    pub write_submitter: Arc<WriteSubmitter>,
}

type BlockingSemaphore = (Arc<Semaphore>, usize);

/// Cancels side-effect-free SlateDB work associated with an HTTP request when
/// Axum drops the handler future. This covers reader replicas and writer-local
/// read fallbacks; a normally completed request disarms the guard.
struct CancelLsmReadOnDrop {
    cancellation: Option<LsmReadCancellation>,
}

impl CancelLsmReadOnDrop {
    fn new(cancellation: LsmReadCancellation) -> Self {
        Self {
            cancellation: Some(cancellation),
        }
    }

    fn disarm(&mut self) {
        self.cancellation.take();
    }
}

impl Drop for CancelLsmReadOnDrop {
    fn drop(&mut self) {
        if let Some(cancellation) = self.cancellation.take() {
            cancellation.cancel();
        }
    }
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn blocking_semaphore_from_env(name: &str) -> Option<BlockingSemaphore> {
    env_usize(name).map(|cap| (Arc::new(Semaphore::new(cap)), cap))
}

static READ_BLOCKING: LazyLock<Option<BlockingSemaphore>> =
    LazyLock::new(|| blocking_semaphore_from_env("HELIX_READ_BLOCKING_CAP"));
static SEARCH_BLOCKING: LazyLock<Option<BlockingSemaphore>> =
    LazyLock::new(|| blocking_semaphore_from_env("HELIX_SEARCH_BLOCKING_CAP"));
static SCAN_BLOCKING: LazyLock<Option<BlockingSemaphore>> =
    LazyLock::new(|| blocking_semaphore_from_env("HELIX_SCAN_BLOCKING_CAP"));
static WRITE_BLOCKING: LazyLock<Option<BlockingSemaphore>> =
    LazyLock::new(|| blocking_semaphore_from_env("HELIX_WRITE_BLOCKING_CAP"));
static INDEX_BLOCKING: LazyLock<Option<BlockingSemaphore>> =
    LazyLock::new(|| blocking_semaphore_from_env("HELIX_INDEX_BLOCKING_CAP"));
static MAINTENANCE_BLOCKING: LazyLock<Option<BlockingSemaphore>> =
    LazyLock::new(|| blocking_semaphore_from_env("HELIX_MAINTENANCE_BLOCKING_CAP"));

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BlockingAdmissionPool {
    Read,
    Search,
    Scan,
    Write,
    Index,
    Maintenance,
}

/// Metrics and readiness perform the same S3-backed catalog reads as ordinary
/// read routes, so they share the read pool. Fast health/liveness probes return
/// before blocking admission is consulted.
fn blocking_admission_pool(class: RouteClass) -> BlockingAdmissionPool {
    match class {
        RouteClass::Probe | RouteClass::Read => BlockingAdmissionPool::Read,
        RouteClass::Search => BlockingAdmissionPool::Search,
        RouteClass::Scan => BlockingAdmissionPool::Scan,
        RouteClass::Write => BlockingAdmissionPool::Write,
        RouteClass::Index => BlockingAdmissionPool::Index,
        RouteClass::Maintenance | RouteClass::Admin => BlockingAdmissionPool::Maintenance,
    }
}

fn route_has_cancellable_lsm_read(method: &str, path: &str, class: RouteClass) -> bool {
    if matches!(
        path,
        "/" | "/health" | "/healthz" | "/livez" | "/internal/changes"
    ) {
        return false;
    }
    matches!(
        class,
        RouteClass::Read | RouteClass::Search | RouteClass::Scan
    ) || (method == "GET"
        && class == RouteClass::Probe
        && matches!(path, "/metrics" | "/ready" | "/readyz"))
}

/// Per-collection segment breaker state: tracks last optimizer submit time and
/// whether the breaker is currently tripped (for hysteresis).
struct SegmentBreakerState {
    /// `None` until the first optimizer submit, which is always allowed.
    /// (Computing a synthetic past instant via `now - throttle` can underflow
    /// and panic when the monotonic clock is younger than the throttle,
    /// poisoning the breaker mutex pod-wide.)
    last_submit: Option<Instant>,
    tripped: bool,
}

static SEGMENT_BREAKER_OPTIMIZER_SUBMIT_TIMES: LazyLock<
    Mutex<HashMap<String, SegmentBreakerState>>,
> = LazyLock::new(|| Mutex::new(HashMap::new()));

struct SegmentCircuitBreakerRejection {
    response: HttpResponse<Body>,
    log_rejection: bool,
}

fn segment_breaker_optimizer_throttle() -> Duration {
    Duration::from_secs(
        env_usize("HELIX_SEGMENT_BREAKER_OPTIMIZER_THROTTLE_SECS").unwrap_or(30) as u64,
    )
}

fn segment_breaker_recovery_ratio() -> f64 {
    std::env::var("HELIX_SEGMENT_BREAKER_RECOVERY_RATIO")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|r| *r > 0.0 && *r < 1.0)
        .unwrap_or(0.75)
}

fn segment_breaker_reeval_secs() -> u64 {
    std::env::var("HELIX_SEGMENT_BREAKER_REEVAL_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(15)
}

/// Pure decision for whether the throttle window has elapsed for `last_submit`.
/// Returns `true` when an optimizer submit is allowed (window elapsed or never
/// submitted), `false` when the call should be throttled. Kept pure so the
/// throttle semantics are unit-testable without touching the global state map
/// or spawning tasks.
fn segment_breaker_submit_allowed(
    last_submit: Option<Instant>,
    throttle: Duration,
    now: Instant,
) -> bool {
    match last_submit {
        None => true,
        Some(last) => now.duration_since(last) >= throttle,
    }
}

/// Emits the tripped gauge for `collection` (1.0 = open, 0.0 = closed).
fn set_segment_breaker_tripped_gauge(collection: &str, tripped: bool) {
    metrics::gauge!(
        "helix_segment_circuit_breaker_tripped",
        "collection" => collection.to_string()
    )
    .set(if tripped { 1.0 } else { 0.0 });
}

/// Spawns the merge optimizer for `collection` on the blocking pool and records
/// the result-labeled submit counter. The caller owns the throttle decision; by
/// the time we get here a submit has already been authorized.
fn spawn_segment_breaker_optimizer(state: &AsyncGatewayState, collection: &str) {
    let collections = Arc::clone(&state.collections);
    let config = state.collections.config().clone();
    let collection = collection.to_string();
    tokio::task::spawn_blocking(move || {
        let result = allow_lsm_blocking(|| {
            match replication::submit_collection_optimizer(&collections, &config, &collection) {
                Ok(0) => "empty",
                Ok(_) => "submitted",
                Err(err) => {
                    warn!(collection = %collection, error = %err, "segment breaker optimizer submit failed");
                    "error"
                }
            }
        });
        metrics::counter!(
            "helix_segment_circuit_breaker_optimizer_submit_total",
            "result" => result
        )
        .increment(1);
    });
}

/// Records the throttled submit counters (the optimizer was *not* spawned).
fn record_segment_breaker_submit_throttled() {
    metrics::counter!(
        "helix_segment_circuit_breaker_optimizer_submit_total",
        "result" => "throttled"
    )
    .increment(1);
    metrics::counter!("helix_segment_circuit_breaker_optimizer_submit_throttled_total")
        .increment(1);
}

fn observe_http_request(method: &str, path: &str, status: u16, started: Instant) {
    let route = route_label_for(path);
    metrics::counter!(
        "helix_http_requests_total",
        "method" => method.to_string(),
        "route" => route.to_string(),
        "status" => status_class(status),
    )
    .increment(1);
    metrics::histogram!(
        "helix_http_request_duration_seconds",
        "method" => method.to_string(),
        "route" => route.to_string(),
    )
    .record(started.elapsed().as_secs_f64());
}

struct BlockingAdmissionGuard {
    permit: Option<OwnedSemaphorePermit>,
    semaphore: Arc<Semaphore>,
    cap: usize,
    class: RouteClass,
}

impl Drop for BlockingAdmissionGuard {
    fn drop(&mut self) {
        drop(self.permit.take());
        let avail = self.semaphore.available_permits().min(self.cap);
        metrics::gauge!(
            "helix_blocking_admission_available",
            "class" => self.class.as_label().to_string()
        )
        .set(avail as f64);
        crate::diag::log(
            "release",
            "blocking_admission",
            &format!(
                "class={} avail={}/{}",
                self.class.as_label(),
                avail,
                self.cap
            ),
        );
    }
}

fn blocking_semaphore(class: RouteClass) -> Option<&'static BlockingSemaphore> {
    match blocking_admission_pool(class) {
        BlockingAdmissionPool::Read => READ_BLOCKING.as_ref(),
        BlockingAdmissionPool::Search => SEARCH_BLOCKING.as_ref(),
        BlockingAdmissionPool::Scan => SCAN_BLOCKING.as_ref(),
        BlockingAdmissionPool::Write => WRITE_BLOCKING.as_ref(),
        BlockingAdmissionPool::Index => INDEX_BLOCKING.as_ref(),
        BlockingAdmissionPool::Maintenance => MAINTENANCE_BLOCKING.as_ref(),
    }
}

fn blocking_queue_wait_ms(class: RouteClass) -> u64 {
    let class_env = match class {
        RouteClass::Search => Some("HELIX_SEARCH_BLOCKING_QUEUE_WAIT_MS"),
        RouteClass::Scan => Some("HELIX_SCAN_BLOCKING_QUEUE_WAIT_MS"),
        RouteClass::Read => Some("HELIX_READ_BLOCKING_QUEUE_WAIT_MS"),
        RouteClass::Write => Some("HELIX_WRITE_BLOCKING_QUEUE_WAIT_MS"),
        RouteClass::Index => Some("HELIX_INDEX_BLOCKING_QUEUE_WAIT_MS"),
        RouteClass::Maintenance | RouteClass::Admin => {
            Some("HELIX_MAINTENANCE_BLOCKING_QUEUE_WAIT_MS")
        }
        RouteClass::Probe => None,
    };
    class_env
        .and_then(|name| std::env::var(name).ok())
        .or_else(|| std::env::var("HELIX_BLOCKING_QUEUE_WAIT_MS").ok())
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or_else(|| match class {
            RouteClass::Search => 2_000,
            RouteClass::Scan => 2_000,
            _ => 0,
        })
}

/// Circuit breaker decision with hysteresis.
///
/// Returns `(should_reject, should_log)`:
/// - `should_reject` is true when the breaker is tripped
/// - `should_log` is true when the breaker just transitioned from open->tripped
///
/// Hysteresis prevents flip-flopping: once tripped at `limit`, the breaker
/// only closes when segments drop to `limit * recovery_ratio`.
fn segment_circuit_breaker_should_reject(
    segments: usize,
    limit: usize,
    recovery_ratio: f64,
    currently_tripped: bool,
) -> (bool, bool) {
    let recovery_threshold = (limit as f64 * recovery_ratio) as usize;
    if currently_tripped {
        // Close when segments drop below recovery threshold
        if segments <= recovery_threshold {
            return (false, false);
        }
        (true, false) // still tripped, don't re-log
    } else {
        // Open when segments exceed limit
        let tripped = segments > limit;
        (tripped, tripped) // log only on the transition
    }
}

/// Per-collection segment-count circuit breaker for point upserts.
/// The call site only invokes this for `PUT /collections/{name}/points`;
/// deletes intentionally bypass the breaker so a tenant can shed data as a
/// remediation path when its segment count is over the limit.
///
/// Returns `Some(rejection)` only when the loaded collection has more indexed
/// dense segments than `HELIX_PER_COLLECTION_SEGMENT_LIMIT`. Returns `None` and
/// allows the write through when:
///   * the env var is unset (feature disabled),
///   * the collection is not currently in the loaded LRU (cold collections
///     cannot have a live merge backlog by definition; reopening will not
///     change segment count),
///   * the collection has zero indexed dense segments (e.g. `_graph` /
///     `_history` flat metadata indexes).
///
/// The intent is to convert a runaway-segment failure mode (which previously
/// surfaced as RSS pressure → OOMKill) into an explicit, retryable 503 on the
/// offending tenant only, while leaving all other collections unaffected.
fn segment_circuit_breaker_response(
    state: &AsyncGatewayState,
    collection: &str,
) -> Option<SegmentCircuitBreakerRejection> {
    let limit = env_usize("HELIX_PER_COLLECTION_SEGMENT_LIMIT")?;
    let recovery_ratio = segment_breaker_recovery_ratio();
    let storage = match state.collections.get_loaded_collection(collection) {
        Ok(Some(s)) => s,
        _ => return None,
    };
    let segments = storage.named_vectors.indexed_dense_segment_count();
    if segments == 0 {
        return None;
    }
    metrics::gauge!(
        "helix_segment_circuit_breaker_segments",
        "collection" => collection.to_string()
    )
    .set(segments as f64);

    let throttle = segment_breaker_optimizer_throttle();
    let now = Instant::now();
    // Single lock scope: read tripped, decide reject via the pure breaker,
    // clear tripped on close, and on rejection own the trip/throttle state
    // transitions. Side effects (gauge, optimizer spawn, metrics) happen AFTER
    // the lock is released based on the captured decision, so we never hold the
    // mutex across a spawn_blocking submit.
    let mut newly_tripped = false;
    let mut newly_closed = false;
    let mut should_submit = false;
    let mut should_log = false;
    let mut should_reject = false;
    if let Ok(mut states) = SEGMENT_BREAKER_OPTIMIZER_SUBMIT_TIMES.lock() {
        let currently_tripped = states.get(collection).map(|s| s.tripped).unwrap_or(false);
        let (reject, log) = segment_circuit_breaker_should_reject(
            segments,
            limit,
            recovery_ratio,
            currently_tripped,
        );
        should_reject = reject;
        should_log = log;
        if !reject {
            // Breaker is closing — clear tripped so future trips are detected.
            if currently_tripped {
                if let Some(entry) = states.get_mut(collection) {
                    entry.tripped = false;
                }
                newly_closed = true;
            }
        } else {
            let entry = states.entry(collection.to_string()).or_insert_with(|| {
                SegmentBreakerState {
                    last_submit: None, // allow immediate first submit
                    tripped: false,
                }
            });
            newly_tripped = !entry.tripped;
            entry.tripped = true;
            // Only submit when the throttle window has elapsed; otherwise the
            // call is throttled and no optimizer job is spawned.
            if segment_breaker_submit_allowed(entry.last_submit, throttle, now) {
                entry.last_submit = Some(now);
                should_submit = true;
            }
        }
    }

    if newly_closed {
        set_segment_breaker_tripped_gauge(collection, false);
    }

    if !should_reject {
        return None;
    }
    if newly_tripped {
        set_segment_breaker_tripped_gauge(collection, true);
    }
    metrics::counter!(
        "helix_segment_circuit_breaker_rejected_total",
        "collection" => collection.to_string()
    )
    .increment(1);
    if should_submit {
        spawn_segment_breaker_optimizer(state, collection);
    } else {
        record_segment_breaker_submit_throttled();
    }
    // Log only on a state transition (new trip) — not on every rejection.
    let log_rejection = should_log || newly_tripped;
    let retry_after = env_usize("HELIX_PER_COLLECTION_SEGMENT_RETRY_SECS").unwrap_or(5);
    let body = format!(
        "Collection '{}' has {} indexed dense segments (limit {}); merge optimizer is behind. Retry after {}s.\n",
        collection, segments, limit, retry_after
    );
    let response = HttpResponse::builder()
        .status(StatusCode::SERVICE_UNAVAILABLE)
        .header("Retry-After", retry_after.to_string())
        .header("Connection", "close")
        .header("X-Helix-Collection", collection)
        .header("X-Helix-Indexed-Segments", segments.to_string())
        .header("X-Helix-Segment-Limit", limit.to_string())
        .body(Body::from(body))
        .unwrap();
    Some(SegmentCircuitBreakerRejection {
        response,
        log_rejection,
    })
}

fn segment_circuit_breaker_collection<'a>(method: &str, path: &'a str) -> Option<&'a str> {
    if method != "PUT" {
        return None;
    }
    point_mutation_admission::point_mutation_collection(method, path)
}

/// Decision for a background re-evaluation of a tripped collection.
#[derive(Debug, PartialEq, Eq)]
enum SegmentBreakerReevalDecision {
    /// Segment count dropped to/below the recovery threshold (or the collection
    /// is no longer loaded) — close the breaker.
    Close,
    /// Still above the recovery threshold — keep the breaker tripped and let the
    /// caller re-submit the optimizer (subject to its own throttle) so draining
    /// continues without client traffic.
    KeepTripped,
}

/// Pure per-collection re-eval decision used by the background liveness task.
///
/// `segments` is `None` when the collection is no longer loaded (cold), which
/// means it cannot have a live merge backlog and the breaker should close.
/// Otherwise we close once the count is at or below `limit * recovery_ratio`,
/// mirroring the hysteresis used on the upsert path.
fn segment_breaker_reeval_decision(
    segments: Option<usize>,
    limit: usize,
    recovery_ratio: f64,
) -> SegmentBreakerReevalDecision {
    match segments {
        None => SegmentBreakerReevalDecision::Close,
        Some(segments) => {
            let recovery_threshold = (limit as f64 * recovery_ratio) as usize;
            if segments <= recovery_threshold {
                SegmentBreakerReevalDecision::Close
            } else {
                SegmentBreakerReevalDecision::KeepTripped
            }
        }
    }
}

/// Background liveness re-evaluation. Runs every `HELIX_SEGMENT_BREAKER_REEVAL_SECS`
/// and, for each collection currently marked tripped, recomputes the indexed
/// dense segment count. If the count has drained to/below the recovery threshold
/// (or the collection is no longer loaded) the breaker is closed; otherwise the
/// optimizer is re-submitted through the same throttled path so draining
/// continues even when clients correctly back off on 503s.
fn run_segment_breaker_reeval(state: &AsyncGatewayState) {
    let Some(limit) = env_usize("HELIX_PER_COLLECTION_SEGMENT_LIMIT") else {
        return;
    };
    let recovery_ratio = segment_breaker_recovery_ratio();
    let throttle = segment_breaker_optimizer_throttle();

    // Snapshot the currently-tripped collections under the lock, then release it
    // before touching collection storage or spawning optimizer jobs.
    let tripped: Vec<String> = match SEGMENT_BREAKER_OPTIMIZER_SUBMIT_TIMES.lock() {
        Ok(states) => states
            .iter()
            .filter(|(_, s)| s.tripped)
            .map(|(name, _)| name.clone())
            .collect(),
        Err(_) => return,
    };

    for collection in tripped {
        let segments = match state.collections.get_loaded_collection(&collection) {
            Ok(Some(storage)) => Some(storage.named_vectors.indexed_dense_segment_count()),
            Ok(None) => None,
            Err(err) => {
                warn!(collection = %collection, error = %err, "segment breaker re-eval failed to load collection");
                continue;
            }
        };
        if let Some(segments) = segments {
            metrics::gauge!(
                "helix_segment_circuit_breaker_segments",
                "collection" => collection.clone()
            )
            .set(segments as f64);
        }

        match segment_breaker_reeval_decision(segments, limit, recovery_ratio) {
            SegmentBreakerReevalDecision::Close => {
                let closed = if let Ok(mut states) = SEGMENT_BREAKER_OPTIMIZER_SUBMIT_TIMES.lock() {
                    match states.get_mut(&collection) {
                        Some(entry) if entry.tripped => {
                            entry.tripped = false;
                            true
                        }
                        _ => false,
                    }
                } else {
                    false
                };
                if closed {
                    set_segment_breaker_tripped_gauge(&collection, false);
                    info!(
                        collection = %collection,
                        segments = ?segments,
                        "Segment circuit breaker closed by background re-eval"
                    );
                }
            }
            SegmentBreakerReevalDecision::KeepTripped => {
                // Re-submit through the throttle so draining continues. Take the
                // lock only to check/advance the throttle window, then spawn the
                // optimizer outside the lock.
                let now = Instant::now();
                let should_submit = if let Ok(mut states) =
                    SEGMENT_BREAKER_OPTIMIZER_SUBMIT_TIMES.lock()
                {
                    match states.get_mut(&collection) {
                        Some(entry)
                            if segment_breaker_submit_allowed(entry.last_submit, throttle, now) =>
                        {
                            entry.last_submit = Some(now);
                            true
                        }
                        _ => false,
                    }
                } else {
                    false
                };
                if should_submit {
                    spawn_segment_breaker_optimizer(state, &collection);
                } else {
                    record_segment_breaker_submit_throttled();
                }
            }
        }
    }
}

/// Spawns the background segment-breaker liveness loop. Returns immediately; the
/// loop ticks every `HELIX_SEGMENT_BREAKER_REEVAL_SECS` for the lifetime of the
/// process. Only active when `HELIX_PER_COLLECTION_SEGMENT_LIMIT` is set.
pub fn spawn_segment_breaker_reeval_task(state: AsyncGatewayState) {
    if env_usize("HELIX_PER_COLLECTION_SEGMENT_LIMIT").is_none() {
        return;
    }
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(segment_breaker_reeval_secs()));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        ticker.tick().await; // consume the immediate first tick
        loop {
            ticker.tick().await;
            let state = state.clone();
            // Snapshot + storage reads can block briefly; keep them off the
            // async reactor by running the synchronous re-eval on a blocking
            // worker. (Optimizer submits inside spawn their own blocking task.)
            let _ = tokio::task::spawn_blocking(move || {
                allow_lsm_blocking(|| run_segment_breaker_reeval(&state))
            })
            .await;
        }
    });
}

/// Maps a handler `GraphError` to an HTTP status code: not-found variants → 404,
/// resource-exhaustion (map full / resize backpressure) → 503, everything else
/// → 500. The body remains the error string at the call site.
/// Purge retries back off from seconds up to minutes; 10s keeps clients from
/// hammering a create that cannot succeed until the purge completes.
const PURGE_PENDING_RETRY_AFTER_SECS: u64 = 10;

fn graph_error_status(error: &GraphError) -> u16 {
    match error {
        GraphError::EdgeNotFound | GraphError::NodeNotFound | GraphError::LabelNotFound => 404,
        GraphError::FatalCollectionStorage { .. } => 500,
        GraphError::MapFull | GraphError::ResizeBackpressure(_) => 503,
        // Transient: the dropped incarnation's object-store purge is retrying.
        GraphError::PurgePending(_) => 503,
        // `VectorError(_)` can wrap a "Vector not found" message that has no
        // dedicated variant; fall back to the string for that not-found case.
        GraphError::VectorError(_) if error.to_string().contains("not found") => 404,
        // `collection_manager::get_loaded_collection`/`get_collection` raise a
        // missing collection as `GraphError::New(format!("Collection '{}' not
        // found", name))` — no dedicated variant either. Same string fallback.
        GraphError::New(msg) if msg.contains("Collection") && msg.contains("not found") => 404,
        _ => 500,
    }
}

/// Operational classification for router failures. Missing resources and
/// request-scoped read cancellation are expected outcomes, not server faults;
/// keeping them below `ERROR` prevents alert streams from masking genuine
/// storage and 5xx failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HandlerErrorLogClass {
    CooperativeCancellation,
    ClientOutcome,
    ServerFailure,
}

fn handler_error_log_class(error: &GraphError, status: u16) -> HandlerErrorLogClass {
    if error.to_string().contains("LSM read request cancelled") {
        HandlerErrorLogClass::CooperativeCancellation
    } else if (400..500).contains(&status) {
        HandlerErrorLogClass::ClientOutcome
    } else {
        HandlerErrorLogClass::ServerFailure
    }
}

fn log_handler_error(method: &str, path: &str, status: u16, error: &GraphError) {
    match handler_error_log_class(error, status) {
        HandlerErrorLogClass::CooperativeCancellation => {
            // The backend cancellation counter is incremented where the
            // cancellation originates. This log is diagnostic only: it means
            // the HTTP caller/deadline already abandoned safe read work.
            debug!(
                method,
                path,
                status,
                error = ?error,
                "Handler read cancelled after request was abandoned"
            );
        }
        HandlerErrorLogClass::ClientOutcome => {
            debug!(
                method,
                path,
                status,
                error = ?error,
                "Handler returned client outcome"
            );
        }
        HandlerErrorLogClass::ServerFailure => {
            error!(method, path, status, error = ?error, "Handler error");
        }
    }
}

fn quarantine_collection_from_handler_error(
    collections: &CollectionManager,
    path: &str,
    error: &GraphError,
) {
    if !error.should_quarantine_collection() {
        return;
    }
    let Some(name) = crate::helix_gateway::inflight::collection_from_path(path) else {
        return;
    };
    collections.quarantine_collection_after_storage_error(name, error);
}

fn resize_admission_poll_ms() -> u64 {
    std::env::var("HELIX_RESIZE_ADMISSION_POLL_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(10)
}

fn resize_admission_warn_ms() -> u128 {
    std::env::var("HELIX_RESIZE_ADMISSION_WARN_MS")
        .ok()
        .and_then(|value| value.parse::<u128>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(5_000)
}

async fn wait_for_collection_resize_admission(state: &AsyncGatewayState, method: &str, path: &str) {
    let Some(collection) =
        crate::helix_gateway::inflight::collection_from_path(path).map(str::to_string)
    else {
        return;
    };
    let storage = match state.collections.get_loaded_collection(&collection) {
        Ok(Some(storage)) => storage,
        Ok(None) => return,
        Err(err) => {
            warn!(
                method = %method,
                path = %path,
                collection = %collection,
                error = ?err,
                "Failed to inspect collection resize state; continuing without async resize admission"
            );
            return;
        }
    };
    if !storage.resize_admission_blocked() {
        return;
    }

    let class = route_class(method, path);
    let started = Instant::now();
    let poll = Duration::from_millis(resize_admission_poll_ms());
    metrics::counter!(
        "helix_resize_admission_wait_total",
        "collection" => collection.clone(),
        "class" => class.as_label().to_string()
    )
    .increment(1);
    crate::diag::log(
        "wait",
        "resize_admission",
        &format!("collection={} class={}", collection, class.as_label()),
    );

    while storage.resize_admission_blocked() {
        tokio::time::sleep(poll).await;
    }

    let waited = started.elapsed();
    metrics::histogram!(
        "helix_resize_admission_wait_ms",
        "collection" => collection.clone(),
        "class" => class.as_label().to_string()
    )
    .record(waited.as_secs_f64() * 1000.0);
    if waited.as_millis() >= resize_admission_warn_ms() {
        warn!(
            method = %method,
            path = %path,
            collection = %collection,
            waited_ms = waited.as_millis(),
            "Async gateway waited for collection resize admission before dispatching to blocking worker"
        );
    }
}

fn resize_admission_backpressure_response(
    state: &AsyncGatewayState,
    method: &str,
    path: &str,
    class: RouteClass,
) -> Option<HttpResponse<Body>> {
    if !class.is_write_like() || class == RouteClass::Admin {
        return None;
    }
    let collection = crate::helix_gateway::inflight::collection_from_path(path)?.to_string();
    let storage = match state.collections.get_loaded_collection(&collection) {
        Ok(Some(storage)) => storage,
        Ok(None) => return None,
        Err(err) => {
            warn!(
                method = %method,
                path = %path,
                collection = %collection,
                error = ?err,
                "Failed to inspect collection resize state; continuing without resize backpressure"
            );
            return None;
        }
    };
    if !storage.resize_admission_blocked() {
        return None;
    }

    metrics::counter!(
        "helix_resize_admission_backpressure_total",
        "collection" => collection.clone(),
        "class" => class.as_label().to_string()
    )
    .increment(1);
    warn!(
        method = %method,
        path = %path,
        collection = %collection,
        class = class.as_label(),
        "Rejecting write-like request while LMDB resize is pending"
    );
    Some(
        HttpResponse::builder()
            .status(StatusCode::SERVICE_UNAVAILABLE)
            .header("Retry-After", "1")
            .header("Connection", "close")
            .header("X-Helix-Collection", collection)
            .header("X-Helix-Route-Class", class.as_label())
            .body(Body::from("Helix collection is resizing; retry shortly\n"))
            .unwrap(),
    )
}

fn blocking_overload_response(class: RouteClass, cap: usize) -> HelixResponse {
    metrics::counter!(
        "helix_blocking_admission_rejected_total",
        "class" => class.as_label().to_string()
    )
    .increment(1);
    let mut response = HelixResponse::new();
    response.status = 503;
    response.body = b"Helix blocking capacity exhausted".to_vec();
    response
        .headers
        .insert("Retry-After".to_string(), "1".to_string());
    response.headers.insert(
        "X-Helix-Route-Class".to_string(),
        class.as_label().to_string(),
    );
    response
        .headers
        .insert("X-Helix-Blocking-Cap".to_string(), cap.to_string());
    response
}

fn unhealthy_lsm_writer_probe_response(
    state: &AsyncGatewayState,
    status: StatusCode,
) -> Option<HttpResponse<Body>> {
    let unhealthy = match state.collections.unhealthy_lsm_writer_close_reasons() {
        Ok(unhealthy) => unhealthy,
        Err(error) => {
            warn!(error = %error, "failed to inspect LSM writer health for probe");
            return Some(
                HttpResponse::builder()
                    .status(StatusCode::SERVICE_UNAVAILABLE)
                    .header("Content-Type", "application/json")
                    .header("X-Helix-Route-Class", RouteClass::Probe.as_label())
                    .body(Body::from(
                        b"{\"status\":\"unhealthy\",\"reason\":\"lsm_writer_health_check_failed\"}"
                            .to_vec(),
                    ))
                    .unwrap(),
            );
        }
    };
    let (collection, reason) = unhealthy.first()?;
    let body = sonic_rs::json!({
        "status": "unhealthy",
        "reason": "lsm_writer_closed",
        "collection": collection,
        "close_reason": format!("{reason:?}"),
        "unhealthy_loaded_collections": unhealthy.len(),
    });
    Some(
        HttpResponse::builder()
            .status(status)
            .header("Content-Type", "application/json")
            .header("X-Helix-Route-Class", RouteClass::Probe.as_label())
            .body(Body::from(sonic_rs::to_vec(&body).unwrap_or_else(|_| {
                b"{\"status\":\"unhealthy\",\"reason\":\"lsm_writer_closed\"}".to_vec()
            })))
            .unwrap(),
    )
}

async fn acquire_blocking_admission(
    class: RouteClass,
) -> Result<Option<BlockingAdmissionGuard>, HelixResponse> {
    let Some((semaphore, cap)) = blocking_semaphore(class) else {
        return Ok(None);
    };

    let started = Instant::now();
    let wait_ms = blocking_queue_wait_ms(class);
    crate::diag::log(
        "wait",
        "blocking_admission",
        &format!("class={} wait_ms={}", class.as_label(), wait_ms),
    );
    let permit_result = if wait_ms == 0 {
        Arc::clone(semaphore).try_acquire_owned()
    } else {
        match tokio::time::timeout(
            Duration::from_millis(wait_ms),
            Arc::clone(semaphore).acquire_owned(),
        )
        .await
        {
            Ok(Ok(permit)) => Ok(permit),
            Ok(Err(_closed)) => Err(TryAcquireError::Closed),
            Err(_elapsed) => {
                metrics::counter!(
                    "helix_blocking_admission_wait_timeout_total",
                    "class" => class.as_label().to_string()
                )
                .increment(1);
                crate::diag::log(
                    "timeout",
                    "blocking_admission",
                    &format!(
                        "class={} waited_ms={}",
                        class.as_label(),
                        started.elapsed().as_millis()
                    ),
                );
                Err(TryAcquireError::NoPermits)
            }
        }
    };

    match permit_result {
        Ok(permit) => {
            let waited = started.elapsed().as_millis();
            metrics::histogram!(
                "helix_blocking_admission_wait_ms",
                "class" => class.as_label().to_string()
            )
            .record(started.elapsed().as_secs_f64() * 1000.0);
            let avail = semaphore.available_permits();
            metrics::gauge!(
                "helix_blocking_admission_available",
                "class" => class.as_label().to_string()
            )
            .set(avail as f64);
            crate::diag::log(
                "got",
                "blocking_admission",
                &format!(
                    "class={} waited_ms={} avail={}/{}",
                    class.as_label(),
                    waited,
                    avail,
                    *cap
                ),
            );
            Ok(Some(BlockingAdmissionGuard {
                permit: Some(permit),
                semaphore: Arc::clone(semaphore),
                cap: *cap,
                class,
            }))
        }
        Err(TryAcquireError::NoPermits) => {
            crate::diag::log(
                "reject",
                "blocking_admission",
                &format!("class={} cap={}", class.as_label(), *cap),
            );
            Err(blocking_overload_response(class, *cap))
        }
        Err(TryAcquireError::Closed) => {
            let mut response = HelixResponse::new();
            response.status = 503;
            response.body = b"Helix blocking capacity closed".to_vec();
            Err(response)
        }
    }
}

pub async fn serve(
    listener: TcpListener,
    state: AsyncGatewayState,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<(), std::io::Error> {
    let app = Router::new().fallback(any(handle)).with_state(state).layer(
        ServiceBuilder::new()
            .layer(HandleErrorLayer::new(|_: BoxError| async {
                StatusCode::SERVICE_UNAVAILABLE
            }))
            .load_shed()
            .timeout(Duration::from_secs(300)),
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
}

/// RAII guard for `helix_submit_inflight` and `helix_submit_active_writers`
/// gauges. The submit handler awaits inside the router loop, so a request
/// cancellation between gauge increment and decrement would leak the gauge
/// even though the semaphore permit drops cleanly via RAII. Wrapping the
/// pair in a Drop impl keeps the gauge consistent with the semaphore on
/// every exit path (success, panic, cancellation).
///
/// Also refreshes `helix_submit_available_permits` on drop so the gauge
/// reflects post-permit-release state.
struct InflightGuard {
    class_label: String,
    semaphore: Arc<Semaphore>,
}

impl InflightGuard {
    fn new(class_label: String, semaphore: Arc<Semaphore>) -> Self {
        metrics::gauge!("helix_submit_active_writers").increment(1.0);
        metrics::gauge!("helix_submit_inflight", "class" => class_label.clone()).increment(1.0);
        Self {
            class_label,
            semaphore,
        }
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        metrics::gauge!("helix_submit_inflight", "class" => self.class_label.clone())
            .decrement(1.0);
        metrics::gauge!("helix_submit_active_writers").decrement(1.0);
        metrics::gauge!(
            "helix_submit_available_permits",
            "class" => self.class_label.clone()
        )
        .set(self.semaphore.available_permits() as f64);
    }
}

pub struct WriteSubmitter {
    writer_semaphore: Arc<Semaphore>,
}

impl WriteSubmitter {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            writer_semaphore: Arc::new(Semaphore::new(
                std::env::var("HELIX_SUBMIT_CONCURRENCY")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .filter(|value| *value > 0)
                    .unwrap_or(32),
            )),
        })
    }

    async fn submit(
        self: &Arc<Self>,
        state: AsyncGatewayState,
        class: RouteClass,
        request: HelixRequest,
        point_mutation_guard: Option<PointMutationGuard>,
    ) -> HelixResponse {
        let timeout = submit_timeout();
        let wait_started = Instant::now();
        let permit = match tokio::time::timeout(
            timeout,
            self.writer_semaphore.clone().acquire_owned(),
        )
        .await
        {
            Ok(Ok(permit)) => {
                metrics::histogram!(
                    "helix_submit_wait_ms",
                    "class" => class.as_label().to_string()
                )
                .record(wait_started.elapsed().as_secs_f64() * 1000.0);
                metrics::gauge!(
                    "helix_submit_available_permits",
                    "class" => class.as_label().to_string()
                )
                .set(self.writer_semaphore.available_permits() as f64);
                permit
            }
            Ok(Err(_)) => {
                metrics::histogram!(
                    "helix_submit_wait_ms",
                    "class" => class.as_label().to_string()
                )
                .record(wait_started.elapsed().as_secs_f64() * 1000.0);
                let mut response = HelixResponse::new();
                response.status = 503;
                response.body = b"Helix write capacity unavailable".to_vec();
                response
                    .headers
                    .insert("Retry-After".to_string(), "1".to_string());
                return response;
            }
            Err(_) => {
                metrics::histogram!(
                    "helix_submit_wait_ms",
                    "class" => class.as_label().to_string()
                )
                .record(wait_started.elapsed().as_secs_f64() * 1000.0);
                metrics::counter!(
                    "helix_submit_rejected_total",
                    "class" => class.as_label().to_string()
                )
                .increment(1);
                let mut response = HelixResponse::new();
                response.status = 503;
                response.body = b"Helix write capacity exhausted".to_vec();
                response
                    .headers
                    .insert("Retry-After".to_string(), "1".to_string());
                return response;
            }
        };

        let class_label = class.as_label().to_string();
        metrics::counter!(
            "helix_write_queue_pushed_total",
            "class" => class_label.clone(),
            "path" => "bypass".to_string()
        )
        .increment(1);
        // RAII so cancellation between here and end-of-fn doesn't leak the
        // inflight/active_writers gauges. See InflightGuard above.
        let _inflight_guard =
            InflightGuard::new(class_label.clone(), Arc::clone(&self.writer_semaphore));

        let max_attempts = submit_max_attempts();
        let retry_snapshot =
            if max_attempts > 1 && request.body.len() <= submit_retry_max_body_bytes() {
                Some((
                    request.method.clone(),
                    request.headers.clone(),
                    request.path.clone(),
                    request.body.clone(),
                ))
            } else {
                None
            };
        let mut next_request = Some(request);
        let mut attempt = 1u32;
        // Hold the per-collection point-mutation gate across all retry
        // attempts so concurrent same-collection mutations stay serialized
        // until this request completes (or finally fails). Each blocking
        // attempt receives a clone backed by the same owned Tokio mutex guard.
        // If the HTTP future is canceled, the blocking worker's clone keeps
        // admission closed until the abandoned storage mutation finishes.
        let _point_mutation_guard = point_mutation_guard;
        // Same cancellation contract for the writer permit: each blocking
        // attempt holds a clone, so an abandoned HTTP future cannot release
        // capacity while its storage mutation is still running.
        let permit = Arc::new(permit);
        let response = loop {
            let request = next_request.take().unwrap_or_else(|| {
                let (method, headers, path, body) = retry_snapshot
                    .as_ref()
                    .expect("retry snapshot required after first attempt");
                HelixRequest {
                    method: method.clone(),
                    headers: headers.clone(),
                    path: path.clone(),
                    body: body.clone(),
                }
            });
            let response = run_router_with_guard(
                state.clone(),
                request,
                None,
                _point_mutation_guard.clone(),
                Some(Arc::clone(&permit)),
            )
            .await;
            if retry_snapshot.is_none()
                && retryable_background_status(response.status)
                && attempt < max_attempts
            {
                metrics::counter!(
                    "helix_submit_retry_disabled_total",
                    "class" => class_label.clone(),
                    "reason" => "body_too_large".to_string()
                )
                .increment(1);
            }
            if retry_snapshot.is_none()
                || !retryable_background_status(response.status)
                || attempt >= max_attempts
            {
                break response;
            }
            metrics::counter!(
                "helix_submit_retry_total",
                "class" => class_label.clone(),
                "status" => response.status.to_string()
            )
            .increment(1);
            tokio::time::sleep(submit_retry_delay(attempt)).await;
            attempt = attempt.saturating_add(1);
        };
        drop(permit);
        // _inflight_guard drops at end of scope; its Drop decrements the
        // inflight/active_writers gauges and refreshes available_permits.
        let _ = class_label;
        response
    }
}

fn submit_retry_delay(attempt: u32) -> Duration {
    let base_ms = std::env::var("HELIX_WRITE_QUEUE_RETRY_BASE_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(250);
    let max_ms = std::env::var("HELIX_WRITE_QUEUE_RETRY_MAX_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(5_000);
    let shift = attempt.saturating_sub(1).min(6);
    Duration::from_millis(base_ms.saturating_mul(1u64 << shift).min(max_ms))
}

fn submit_timeout() -> Duration {
    let millis = std::env::var("HELIX_SUBMIT_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(5_000);
    Duration::from_millis(millis)
}

fn submit_max_attempts() -> u32 {
    std::env::var("HELIX_SUBMIT_MAX_ATTEMPTS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(3)
}

fn submit_retry_max_body_bytes() -> usize {
    std::env::var("HELIX_SUBMIT_RETRY_MAX_BODY_MB")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1)
        .saturating_mul(1024 * 1024)
}

fn retryable_background_status(status: u16) -> bool {
    status == 429 || status == 503 || status >= 500
}

async fn handle(
    State(state): State<AsyncGatewayState>,
    request: HttpRequest<Body>,
) -> impl IntoResponse {
    let request_started = Instant::now();
    let (parts, body) = request.into_parts();
    let method = parts.method.as_str().to_ascii_uppercase();
    let path = parts.uri.path().to_string();
    let class = route_class(&method, &path);

    // Fast health/liveness endpoints must respond even when the Tokio blocking
    // pool is saturated, so short-circuit them before admission or blocking
    // dispatch. Readiness is also handled inline below, but its S3 catalog read
    // uses bounded admission and cooperative cancellation.
    if method == "GET" && matches!(path.as_str(), "/" | "/health" | "/healthz" | "/livez") {
        // Drop the request body without reading it — probes carry no payload.
        drop(body);
        if let Some(response) =
            unhealthy_lsm_writer_probe_response(&state, StatusCode::INTERNAL_SERVER_ERROR)
        {
            observe_http_request(&method, &path, response.status().as_u16(), request_started);
            return response;
        }
        observe_http_request(&method, &path, 200, request_started);
        return HttpResponse::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/json")
            .header("X-Helix-Route-Class", RouteClass::Probe.as_label())
            .body(Body::from(b"{\"status\":\"ok\"}".to_vec()))
            .unwrap();
    }
    // Peer hot-list for reader warm (storage_core::reader_warm): an in-memory
    // snapshot, answered inline before admission. 404 unless this is a reader
    // whose warm task published a list.
    if method == "GET" && path == reader_warm::HOT_COLLECTIONS_PATH {
        drop(body);
        let (status, payload) = match reader_warm::hot_collections_response_body() {
            Some(payload) => (StatusCode::OK, payload),
            None => (
                StatusCode::NOT_FOUND,
                b"{\"status\":\"not_found\"}".to_vec(),
            ),
        };
        observe_http_request(&method, &path, status.as_u16(), request_started);
        return HttpResponse::builder()
            .status(status)
            .header("Content-Type", "application/json")
            .header("X-Helix-Route-Class", RouteClass::Probe.as_label())
            .body(Body::from(payload))
            .unwrap();
    }
    // Reader warm-before-ready gate (storage_core::reader_warm). `/readyz/warm`
    // is an atomic load only — no admission, no storage access — so the reader
    // readiness probe never queues behind or adds to S3-backed work.
    if method == "GET" && matches!(path.as_str(), "/ready" | "/readyz" | "/readyz/warm") {
        let warm_ready = reader_warm::startup_warm_ready();
        if !warm_ready || path == "/readyz/warm" {
            drop(body);
            let (status, payload): (StatusCode, &[u8]) = if warm_ready {
                (StatusCode::OK, b"{\"status\":\"ready\"}")
            } else {
                (StatusCode::SERVICE_UNAVAILABLE, b"{\"status\":\"warming\"}")
            };
            observe_http_request(&method, &path, status.as_u16(), request_started);
            return HttpResponse::builder()
                .status(status)
                .header("Content-Type", "application/json")
                .header("X-Helix-Route-Class", RouteClass::Probe.as_label())
                .body(Body::from(payload.to_vec()))
                .unwrap();
        }
    }
    if method == "GET" && matches!(path.as_str(), "/ready" | "/readyz") {
        if let Some(response) =
            unhealthy_lsm_writer_probe_response(&state, StatusCode::SERVICE_UNAVAILABLE)
        {
            drop(body);
            observe_http_request(&method, &path, response.status().as_u16(), request_started);
            return response;
        }
        // Match `handle_ready` without dispatching through the blocking router.
        // This still performs an S3-backed catalog read, so share the bounded
        // read pool and cancel the side-effect-free list if kubelet disconnects.
        let blocking_guard = match acquire_blocking_admission(RouteClass::Probe).await {
            Ok(guard) => guard,
            Err(response) => {
                drop(body);
                observe_http_request(&method, &path, response.status, request_started);
                return http_response(response);
            }
        };
        let cancellation = route_has_cancellable_lsm_read(&method, &path, class)
            .then(LsmReadCancellation::new)
            .expect("readiness catalog listing must be cancellable");
        let worker_cancellation = cancellation.clone();
        let mut cancel_on_drop = CancelLsmReadOnDrop::new(cancellation);
        let collections_for_list = Arc::clone(&state.collections);
        let collections_on_disk = tokio::task::spawn_blocking(move || {
            let _blocking_guard = blocking_guard;
            allow_lsm_blocking_cancellable(worker_cancellation, || {
                collections_for_list.list_collections()
            })
        })
        .await
        .ok()
        .and_then(Result::ok)
        .map(|names| names.len())
        .unwrap_or(0);
        cancel_on_drop.disarm();
        let collections_loaded = state.collections.loaded_count();
        drop(body);
        observe_http_request(&method, &path, 200, request_started);
        return HttpResponse::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/json")
            .header("X-Helix-Route-Class", RouteClass::Probe.as_label())
            .body(Body::from(
                format!(
                    "{{\"status\":\"ready\",\"collections_on_disk\":{},\"collections_loaded\":{}}}",
                    collections_on_disk, collections_loaded
                )
                .into_bytes(),
            ))
            .unwrap();
    }

    // Durable-commit change feed for reader replicas (see
    // storage_core::change_feed). Control-plane poll at ~1 req/s per reader:
    // served inline like the probes so it never queues behind admission,
    // breakers, or blocking workers.
    if method == "GET" && path == "/internal/changes" {
        drop(body);
        let since = parts
            .uri
            .query()
            .and_then(|query| {
                query
                    .split('&')
                    .find_map(|pair| pair.strip_prefix("since="))
                    .and_then(|value| value.parse::<u64>().ok())
            })
            .unwrap_or(0);
        let snapshot =
            crate::helix_engine::storage_core::change_feed::change_feed().changes_since(since);
        let body_json = serde_json::json!({
            "epoch": snapshot.epoch,
            "cursor": snapshot.cursor,
            "changes": snapshot.changes,
        });
        observe_http_request(&method, &path, 200, request_started);
        return HttpResponse::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/json")
            .header("X-Helix-Route-Class", RouteClass::Probe.as_label())
            .body(Body::from(body_json.to_string().into_bytes()))
            .unwrap();
    }

    if let Some(response) =
        lsm_reader_write_rejection(class, state.collections.config().storage_backend)
    {
        drop(body);
        observe_http_request(&method, &path, response.status, request_started);
        return http_response(response);
    }

    let direct_write = class.is_write_like() && class != RouteClass::Admin;
    // If this collection is waiting on or running an LMDB resize, bounce new
    // writes before body buffering, write-submit permits, or blocking workers.
    // Reads retain the previous wait-and-complete behavior.
    if let Some(response) = resize_admission_backpressure_response(&state, &method, &path, class) {
        drop(body);
        observe_http_request(&method, &path, 503, request_started);
        return response;
    }
    if !direct_write {
        wait_for_collection_resize_admission(&state, &method, &path).await;
    }

    let mut route_guard = None;
    if !direct_write {
        route_guard = match try_acquire_route_admission(class, &path) {
            Ok(guard) => guard,
            Err(reject) => {
                warn!(
                    method = %method,
                    path = %path,
                    class = reject.class.as_label(),
                    status = reject.status,
                    "Route admission saturated; returning retryable overload"
                );
                let status =
                    StatusCode::from_u16(reject.status).unwrap_or(StatusCode::SERVICE_UNAVAILABLE);
                let mut response = HttpResponse::builder()
                    .status(status)
                    .header("Retry-After", "1")
                    .header("Connection", "close")
                    .body(Body::from("Helix write/index capacity exhausted\n"))
                    .unwrap();
                response.headers_mut().insert(
                    "X-Helix-Route-Class",
                    reject.class.as_label().parse().unwrap(),
                );
                observe_http_request(&method, &path, reject.status, request_started);
                return response;
            }
        };
    }

    let _inflight_guard = match crate::helix_gateway::inflight::collection_from_path(&path) {
        Some(name) => {
            let cap = collection_inflight_cap(&method, &path);
            match crate::helix_gateway::inflight::try_acquire_with_cap(name, cap) {
                crate::helix_gateway::inflight::AcquireOutcome::Admitted(guard) => {
                    observe_collection_inflight_admission(class, &path, cap, "admitted", None);
                    Some(guard)
                }
                crate::helix_gateway::inflight::AcquireOutcome::Rejected { current, cap } => {
                    observe_collection_inflight_admission(
                        class,
                        &path,
                        cap,
                        "rejected",
                        Some(current),
                    );
                    warn!(
                        method = %method,
                        path = %path,
                        collection = name,
                        current,
                        cap,
                        "Per-collection in-flight cap exceeded; returning retryable overload"
                    );
                    observe_http_request(&method, &path, 429, request_started);
                    return HttpResponse::builder()
                        .status(StatusCode::TOO_MANY_REQUESTS)
                        .header("Retry-After", "1")
                        .header("Connection", "close")
                        .header("X-Helix-Collection", name)
                        .header("X-Helix-Collection-Cap", cap.to_string())
                        .body(Body::from("Too many in-flight requests for collection\n"))
                        .unwrap();
                }
            }
        }
        None => None,
    };

    let point_mutation_collection =
        point_mutation_admission::point_mutation_collection(&method, &path).map(str::to_owned);
    let breaker_collection = segment_circuit_breaker_collection(&method, &path).map(str::to_owned);
    if let Some(coll) = breaker_collection.as_deref() {
        // Circuit-break upserts only. Deletes must always pass so a tenant
        // can shed data as a remediation path when its segment count is over
        // the limit; blocking deletes would make recovery impossible.
        // Run before body buffering so a saturated collection cannot force us
        // to allocate a large payload only to reject it afterwards.
        if let Some(rejection) = segment_circuit_breaker_response(&state, coll) {
            drop(body);
            if rejection.log_rejection {
                warn!(
                    method = %method,
                    path = %path,
                    collection = coll,
                    "Segment circuit breaker rejected upsert; collection has too many indexed segments"
                );
            }
            observe_http_request(&method, &path, 503, request_started);
            return rejection.response;
        }
    }

    let mut headers = HashMap::new();
    for (name, value) in parts.headers.iter() {
        if let Ok(value) = value.to_str() {
            headers.insert(name.as_str().to_ascii_lowercase(), value.to_string());
        }
    }

    let body = match to_bytes(body, max_body_bytes()).await {
        Ok(body) => body.to_vec(),
        Err(e) => {
            warn!(path = %path, error = ?e, "Bad request body");
            observe_http_request(&method, &path, 413, request_started);
            return HttpResponse::builder()
                .status(StatusCode::PAYLOAD_TOO_LARGE)
                .body(Body::from("Payload too large"))
                .unwrap();
        }
    };

    let request = HelixRequest {
        method: method.clone(),
        headers,
        path: path.clone(),
        body,
    };

    // Acquire before `WriteSubmitter` / blocking-pool admission for both
    // backends so same-collection waiters park cheaply in async context. The
    // per-collection key preserves full parallelism across different
    // collections.
    let point_mutation_guard = match point_mutation_collection.as_deref() {
        Some(coll) => Some(
            point_mutation_admission::acquire_for_backend(state.collections.backend_kind(), coll)
                .await,
        ),
        None => None,
    };

    let response = if direct_write {
        state
            .write_submitter
            .submit(state.clone(), class, request, point_mutation_guard)
            .await
    } else {
        run_router_with_guard(
            state,
            request,
            route_guard.take(),
            point_mutation_guard,
            None,
        )
        .await
    };

    observe_http_request(&method, &path, response.status, request_started);
    http_response(response)
}

async fn run_router_with_guard(
    state: AsyncGatewayState,
    request: HelixRequest,
    route_guard: Option<AdmissionGuard>,
    point_mutation_guard: Option<PointMutationGuard>,
    submit_permit: Option<Arc<OwnedSemaphorePermit>>,
) -> HelixResponse {
    let class = route_class(&request.method, &request.path);
    crate::diag::log(
        "enter",
        "run_router_with_guard",
        &format!("class={} path={}", class.as_label(), request.path),
    );
    let blocking_guard = match acquire_blocking_admission(class).await {
        Ok(guard) => guard,
        Err(response) => {
            crate::diag::log(
                "exit_blocking_reject",
                "run_router_with_guard",
                &format!("class={} path={}", class.as_label(), request.path),
            );
            return response;
        }
    };
    let _route_guard = route_guard;
    let resp_path = request.path.clone();
    let response = run_router(
        state,
        request,
        blocking_guard,
        point_mutation_guard,
        submit_permit,
    )
    .await;
    crate::diag::log(
        "exit",
        "run_router_with_guard",
        &format!(
            "class={} path={} status={}",
            class.as_label(),
            resp_path,
            response.status
        ),
    );
    response
}

async fn run_router(
    state: AsyncGatewayState,
    request: HelixRequest,
    blocking_guard: Option<BlockingAdmissionGuard>,
    point_mutation_guard: Option<PointMutationGuard>,
    submit_permit: Option<Arc<OwnedSemaphorePermit>>,
) -> HelixResponse {
    let method = request.method.clone();
    let path = request.path.clone();
    let class = route_class(&method, &path);
    // Durable writes are intentionally excluded: dropping a commit future can
    // leave its outcome ambiguous. Reader/search/scan and metrics catalog work
    // is side-effect-free and may be abandoned when its HTTP caller goes away.
    let cancellation =
        route_has_cancellable_lsm_read(&method, &path, class).then(LsmReadCancellation::new);
    let worker_cancellation = cancellation.clone();
    let mut cancel_on_drop = cancellation.map(CancelLsmReadOnDrop::new);
    let panic_method = method.clone();
    let panic_path = path.clone();
    let result = tokio::task::spawn_blocking(move || {
        let handler = || {
            let _blocking_guard = blocking_guard;
            // Hold the async point-mutation guard for the full duration of the
            // blocking handler. A direct-write submitter retains a sibling
            // clone across retries; this worker clone also survives HTTP
            // cancellation until the abandoned blocking mutation completes.
            let _point_mutation_guard = point_mutation_guard;
            // WriteSubmitter capacity is released only when the blocking
            // mutation finishes, not when the HTTP future is dropped.
            let _submit_permit = submit_permit;
            let mut response = HelixResponse::new();
            if let Err(e) = state.router.handle(
                Arc::clone(&state.graph),
                Arc::clone(&state.collections),
                Arc::clone(&state.replication),
                request,
                &mut response,
            ) {
                let status = graph_error_status(&e);
                log_handler_error(&method, &path, status, &e);
                quarantine_collection_from_handler_error(&state.collections, &path, &e);
                response.status = status;
                if matches!(e, GraphError::PurgePending(_)) {
                    response.headers.insert(
                        "Retry-After".to_string(),
                        PURGE_PENDING_RETRY_AFTER_SECS.to_string(),
                    );
                }
                if let Some(body) = e.fatal_collection_response_body() {
                    response.body = body;
                    response
                        .headers
                        .insert("Content-Type".to_string(), "application/json".to_string());
                } else {
                    response.body = e.to_string().into_bytes();
                }
            }
            response
        };
        match worker_cancellation {
            Some(cancellation) => allow_lsm_blocking_cancellable(cancellation, handler),
            None => allow_lsm_blocking(handler),
        }
    })
    .await
    .unwrap_or_else(|e| {
        error!(method = %panic_method, path = %panic_path, error = ?e, "Handler panicked");
        let mut response = HelixResponse::new();
        response.status = 500;
        response.body = b"Internal server error".to_vec();
        response
    });
    if let Some(guard) = cancel_on_drop.as_mut() {
        guard.disarm();
    }
    result
}

fn http_response(response: HelixResponse) -> HttpResponse<Body> {
    let status = StatusCode::from_u16(response.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let mut builder = HttpResponse::builder().status(status);
    for (name, value) in response.headers {
        builder = builder.header(name, value);
    }
    builder.body(Body::from(response.body)).unwrap()
}

#[cfg(test)]
mod reader_request_cancellation_tests {
    use super::{
        blocking_admission_pool, route_has_cancellable_lsm_read, BlockingAdmissionPool,
        CancelLsmReadOnDrop,
    };
    use crate::helix_engine::storage_core::backend_lsm::LsmReadCancellation;
    use crate::helix_gateway::thread_pool::thread_pool::{route_class, RouteClass};
    use axum::{extract::State, routing::get, Router};
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    use tokio::{io::AsyncWriteExt, sync::Notify};

    struct PendingHandlerState {
        started: Notify,
        dropped: Notify,
        did_drop: AtomicBool,
    }

    struct MarkHandlerDropped(Arc<PendingHandlerState>);

    impl Drop for MarkHandlerDropped {
        fn drop(&mut self) {
            self.0.did_drop.store(true, Ordering::Release);
            self.0.dropped.notify_one();
        }
    }

    async fn pending_handler(State(state): State<Arc<PendingHandlerState>>) -> &'static str {
        let _drop_marker = MarkHandlerDropped(Arc::clone(&state));
        state.started.notify_one();
        futures::future::pending::<&'static str>().await
    }

    #[test]
    fn drop_guard_cancels_only_while_armed() {
        let cancelled = LsmReadCancellation::new();
        {
            let _guard = CancelLsmReadOnDrop::new(cancelled.clone());
        }
        assert!(cancelled.is_cancelled());

        let completed = LsmReadCancellation::new();
        {
            let mut guard = CancelLsmReadOnDrop::new(completed.clone());
            guard.disarm();
        }
        assert!(!completed.is_cancelled());
    }

    #[test]
    fn metrics_and_readiness_share_read_admission_and_cancellation() {
        assert_eq!(
            blocking_admission_pool(RouteClass::Probe),
            BlockingAdmissionPool::Read
        );
        for path in ["/metrics", "/ready", "/readyz"] {
            let class = route_class("GET", path);
            assert_eq!(class, RouteClass::Probe);
            assert!(
                route_has_cancellable_lsm_read("GET", path, class),
                "{path} performs a cancellable catalog read"
            );
        }
        for path in ["/", "/health", "/healthz", "/livez"] {
            assert!(!route_has_cancellable_lsm_read(
                "GET",
                path,
                route_class("GET", path)
            ));
        }
        assert!(!route_has_cancellable_lsm_read(
            "GET",
            "/internal/changes",
            route_class("GET", "/internal/changes")
        ));
    }

    #[test]
    fn public_kubernetes_config_bounds_read_blocking_work_with_default_queue_waits() {
        let config = include_str!("../../../deploy/lsm-cloud/kubernetes/10-configmap.yaml");
        for (key, value) in [
            ("HELIX_READ_BLOCKING_CAP", "16"),
            ("HELIX_SEARCH_BLOCKING_CAP", "16"),
            ("HELIX_SCAN_BLOCKING_CAP", "8"),
        ] {
            let expected = format!("{key}: \"{value}\"");
            assert!(
                config.lines().any(|line| line.trim() == expected),
                "public Kubernetes config must set {key}={value}"
            );
        }
        for key in [
            "HELIX_BLOCKING_QUEUE_WAIT_MS",
            "HELIX_READ_BLOCKING_QUEUE_WAIT_MS",
            "HELIX_SEARCH_BLOCKING_QUEUE_WAIT_MS",
            "HELIX_SCAN_BLOCKING_QUEUE_WAIT_MS",
        ] {
            let prefix = format!("{key}:");
            assert!(
                !config.lines().any(|line| line.trim().starts_with(&prefix)),
                "public Kubernetes config should retain the built-in {key} default"
            );
        }
    }

    #[tokio::test]
    async fn axum_drops_pending_handler_when_http_peer_disconnects() {
        let state = Arc::new(PendingHandlerState {
            started: Notify::new(),
            dropped: Notify::new(),
            did_drop: AtomicBool::new(false),
        });
        let app = Router::new()
            .route("/", get(pending_handler))
            .with_state(Arc::clone(&state));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await
        });

        let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
        client
            .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), state.started.notified())
            .await
            .expect("pending handler should start");
        drop(client);

        tokio::time::timeout(std::time::Duration::from_secs(1), state.dropped.notified())
            .await
            .expect("disconnect should drop the pending handler future");
        assert!(state.did_drop.load(Ordering::Acquire));

        let _ = shutdown_tx.send(());
        server.await.unwrap().unwrap();
    }
}

#[cfg(test)]
mod segment_circuit_breaker_tests {
    use super::{
        graph_error_status, handler_error_log_class, segment_breaker_reeval_decision,
        segment_breaker_submit_allowed, segment_circuit_breaker_collection,
        segment_circuit_breaker_should_reject, HandlerErrorLogClass, SegmentBreakerReevalDecision,
        SegmentBreakerState, SEGMENT_BREAKER_OPTIMIZER_SUBMIT_TIMES,
    };
    use crate::helix_engine::types::GraphError;
    use std::time::{Duration, Instant};

    #[test]
    fn at_or_below_limit_does_not_reject() {
        let (reject, _) = segment_circuit_breaker_should_reject(0, 64, 0.75, false);
        assert!(!reject);
        let (reject, _) = segment_circuit_breaker_should_reject(8, 64, 0.75, false);
        assert!(!reject);
        let (reject, _) = segment_circuit_breaker_should_reject(63, 64, 0.75, false);
        assert!(!reject);
        let (reject, _) = segment_circuit_breaker_should_reject(64, 64, 0.75, false);
        assert!(!reject);
    }

    #[test]
    fn above_limit_rejects() {
        let (reject, _) = segment_circuit_breaker_should_reject(65, 64, 0.75, false);
        assert!(reject);
        let (reject, _) = segment_circuit_breaker_should_reject(325, 64, 0.75, false);
        assert!(reject);
        let (reject, _) = segment_circuit_breaker_should_reject(577, 64, 0.75, false);
        assert!(reject);
    }

    #[test]
    fn hysteresis_keeps_tripped_until_recovery_ratio() {
        // Limit=100, recovery_ratio=0.75 → close at <=75
        // Go from tripped at 120 segments
        let (reject, _) = segment_circuit_breaker_should_reject(120, 100, 0.75, true);
        assert!(reject); // still tripped

        // Dropped to 90 — still above recovery threshold (75), still tripped
        let (reject, _) = segment_circuit_breaker_should_reject(90, 100, 0.75, true);
        assert!(reject);

        // Dropped to 75 — at recovery threshold, breaker closes
        let (reject, should_log) = segment_circuit_breaker_should_reject(75, 100, 0.75, true);
        assert!(!reject); // not rejected anymore, breaker closed
        assert!(!should_log); // no log on close

        // Dropped to 74 — well below recovery, not tripped
        let (reject, _) = segment_circuit_breaker_should_reject(74, 100, 0.75, false);
        assert!(!reject);

        // New trip: goes back above limit
        let (reject, should_log) = segment_circuit_breaker_should_reject(105, 100, 0.75, false);
        assert!(reject);
        assert!(should_log); // log on new trip
    }

    #[test]
    fn breaker_only_targets_point_upserts() {
        assert_eq!(
            segment_circuit_breaker_collection("PUT", "/collections/foo/points"),
            Some("foo")
        );
        assert_eq!(
            segment_circuit_breaker_collection("POST", "/collections/foo/points/delete"),
            None
        );
        assert_eq!(
            segment_circuit_breaker_collection("POST", "/collections/foo/points/search"),
            None
        );
    }

    #[test]
    fn reeval_closes_when_drained_keeps_tripped_when_over_threshold() {
        // Limit=100, recovery_ratio=0.75 → recovery threshold = 75.
        // Still over threshold → keep tripped.
        assert_eq!(
            segment_breaker_reeval_decision(Some(120), 100, 0.75),
            SegmentBreakerReevalDecision::KeepTripped
        );
        assert_eq!(
            segment_breaker_reeval_decision(Some(76), 100, 0.75),
            SegmentBreakerReevalDecision::KeepTripped
        );
        // Drained to/below threshold → close.
        assert_eq!(
            segment_breaker_reeval_decision(Some(75), 100, 0.75),
            SegmentBreakerReevalDecision::Close
        );
        assert_eq!(
            segment_breaker_reeval_decision(Some(0), 100, 0.75),
            SegmentBreakerReevalDecision::Close
        );
        // Collection no longer loaded → close.
        assert_eq!(
            segment_breaker_reeval_decision(None, 100, 0.75),
            SegmentBreakerReevalDecision::Close
        );
    }

    /// Regression for the production bug: once tripped, with NO further
    /// upsert-path call, the background re-eval observes the drained count and
    /// closes the breaker; while still over threshold it stays tripped.
    #[test]
    fn background_reeval_closes_tripped_breaker_without_upsert() {
        let collection = "breaker_reeval_regression";
        let limit = 100usize;
        let ratio = 0.75; // recovery threshold = 75

        // Trip the breaker (as the upsert path would on segments > limit),
        // recorded directly in the shared state map.
        {
            let mut states = SEGMENT_BREAKER_OPTIMIZER_SUBMIT_TIMES.lock().unwrap();
            states.insert(
                collection.to_string(),
                SegmentBreakerState {
                    last_submit: Some(Instant::now()),
                    tripped: true,
                },
            );
        }

        // First background re-eval while still over threshold (segments=90):
        // decision keeps it tripped, so the background loop leaves tripped set.
        assert_eq!(
            segment_breaker_reeval_decision(Some(90), limit, ratio),
            SegmentBreakerReevalDecision::KeepTripped
        );
        assert!(
            SEGMENT_BREAKER_OPTIMIZER_SUBMIT_TIMES
                .lock()
                .unwrap()
                .get(collection)
                .map(|s| s.tripped)
                .unwrap_or(false),
            "breaker must stay tripped while still over recovery threshold"
        );

        // Drained to 70 (<=75): the background re-eval decides Close and clears
        // tripped exactly as run_segment_breaker_reeval does — no upsert needed.
        assert_eq!(
            segment_breaker_reeval_decision(Some(70), limit, ratio),
            SegmentBreakerReevalDecision::Close
        );
        if let Some(entry) = SEGMENT_BREAKER_OPTIMIZER_SUBMIT_TIMES
            .lock()
            .unwrap()
            .get_mut(collection)
        {
            if entry.tripped {
                entry.tripped = false;
            }
        }
        assert!(
            !SEGMENT_BREAKER_OPTIMIZER_SUBMIT_TIMES
                .lock()
                .unwrap()
                .get(collection)
                .map(|s| s.tripped)
                .unwrap_or(false),
            "background re-eval must clear tripped once drained, with no upsert"
        );

        // Cleanup so other tests sharing the static map are unaffected.
        SEGMENT_BREAKER_OPTIMIZER_SUBMIT_TIMES
            .lock()
            .unwrap()
            .remove(collection);
    }

    #[test]
    fn optimizer_submit_respects_throttle_window() {
        let throttle = Duration::from_secs(30);
        let now = Instant::now();
        // A submit just happened: within the window, no new submit allowed.
        assert!(!segment_breaker_submit_allowed(Some(now), throttle, now));
        assert!(!segment_breaker_submit_allowed(
            Some(now - Duration::from_secs(5)),
            throttle,
            now
        ));
        // Once the throttle window has fully elapsed, a submit is allowed.
        assert!(segment_breaker_submit_allowed(
            Some(now - Duration::from_secs(30)),
            throttle,
            now
        ));
        assert!(segment_breaker_submit_allowed(
            Some(now - Duration::from_secs(45)),
            throttle,
            now
        ));
        // Never submitted: always allowed, regardless of clock age.
        assert!(segment_breaker_submit_allowed(None, throttle, now));
    }

    #[test]
    fn graph_error_status_maps_variants() {
        assert_eq!(graph_error_status(&GraphError::NodeNotFound), 404);
        assert_eq!(graph_error_status(&GraphError::EdgeNotFound), 404);
        assert_eq!(graph_error_status(&GraphError::LabelNotFound), 404);
        assert_eq!(graph_error_status(&GraphError::MapFull), 503);
        assert_eq!(
            graph_error_status(&GraphError::ResizeBackpressure("busy".into())),
            503
        );
        assert_eq!(
            graph_error_status(&GraphError::PurgePending("purge pending".into())),
            503
        );
        assert_eq!(
            graph_error_status(&GraphError::VectorError("Vector not found: x".into())),
            404
        );
        assert_eq!(
            graph_error_status(&GraphError::New("Collection 'x' not found".into())),
            404
        );
        // LSM reader replica: a collection with no manifest surfaces from
        // `get_collection` as the same not-found the writer raises → 404, while
        // other storage failures on the same path stay 500.
        assert_eq!(
            graph_error_status(&GraphError::New(
                "Collection 'codebase' not found (no manifest in object store)".into()
            )),
            404
        );
        assert_eq!(
            graph_error_status(&GraphError::StorageError(
                "io error: Generic S3 error: request failed: 503 Slow Down".into()
            )),
            500
        );
        assert_eq!(
            graph_error_status(&GraphError::New("Lock poisoned: x".into())),
            500
        );
        assert_eq!(
            graph_error_status(&GraphError::StorageError("boom".into())),
            500
        );
        assert_eq!(
            graph_error_status(&GraphError::FatalCollectionStorage {
                code: "mdb_problem",
                message: "txn should abort".into(),
            }),
            500
        );
        assert_eq!(graph_error_status(&GraphError::Default), 500);
    }

    #[test]
    fn handler_error_logging_keeps_expected_outcomes_below_error() {
        for error in [
            GraphError::NodeNotFound,
            GraphError::EdgeNotFound,
            GraphError::LabelNotFound,
            GraphError::New("Collection 'gone' not found".into()),
            GraphError::VectorError("Vector not found: gone".into()),
        ] {
            let status = graph_error_status(&error);
            assert_eq!(status, 404);
            assert_eq!(
                handler_error_log_class(&error, status),
                HandlerErrorLogClass::ClientOutcome,
                "expected 404 must not be logged as a server failure: {error}"
            );
        }

        // Cancellation is returned through the generic backend-error wrapper,
        // so it currently maps to 500. It is still an expected outcome because
        // the request drop/deadline initiated it, and must remain diagnostic.
        let cancelled = GraphError::New("LSM I/O: LSM read request cancelled".into());
        assert_eq!(graph_error_status(&cancelled), 500);
        assert_eq!(
            handler_error_log_class(&cancelled, graph_error_status(&cancelled)),
            HandlerErrorLogClass::CooperativeCancellation
        );
    }

    #[test]
    fn handler_error_logging_retains_server_failures_at_error() {
        for error in [
            GraphError::MapFull,
            GraphError::ResizeBackpressure("busy".into()),
            GraphError::StorageError("object store unavailable".into()),
            GraphError::FatalCollectionStorage {
                code: "lsm_corrupted",
                message: "checksum mismatch".into(),
            },
        ] {
            let status = graph_error_status(&error);
            assert!(status >= 500);
            assert_eq!(
                handler_error_log_class(&error, status),
                HandlerErrorLogClass::ServerFailure,
                "server/storage failures must remain ERROR: {error}"
            );
        }
    }
}

#[cfg(test)]
mod write_submitter_permit_tests {
    use super::{run_router_with_guard, AsyncGatewayState, WriteSubmitter};
    use crate::helix_engine::graph_core::config::Config;
    use crate::helix_engine::graph_core::graph_core::{HelixGraphEngine, HelixGraphEngineOpts};
    use crate::helix_engine::storage_core::{
        collection_manager::CollectionManager, replication::ReplicationManager,
    };
    use crate::helix_engine::types::GraphError;
    use crate::helix_gateway::router::router::{HandlerInput, HelixRouter};
    use crate::helix_gateway::thread_pool::thread_pool::RouteClass;
    use crate::protocol::{request::Request as HelixRequest, response::Response as HelixResponse};
    use std::collections::HashMap;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    use std::time::Duration;
    use tempfile::TempDir;
    use tokio::sync::Semaphore;

    static HANDLER_STARTED: AtomicBool = AtomicBool::new(false);
    static HANDLER_RELEASE: AtomicBool = AtomicBool::new(false);
    static HANDLER_DONE: AtomicBool = AtomicBool::new(false);

    fn slow_write(_input: &HandlerInput, response: &mut HelixResponse) -> Result<(), GraphError> {
        HANDLER_STARTED.store(true, Ordering::Release);
        while !HANDLER_RELEASE.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(5));
        }
        HANDLER_DONE.store(true, Ordering::Release);
        response.status = 200;
        Ok(())
    }

    /// Unblocks the handler even when an assertion panics; otherwise runtime
    /// shutdown would wait forever on the parked blocking task.
    struct ReleaseHandlerOnDrop;

    impl Drop for ReleaseHandlerOnDrop {
        fn drop(&mut self) {
            HANDLER_RELEASE.store(true, Ordering::Release);
        }
    }

    fn purge_pending_create(
        _input: &HandlerInput,
        _response: &mut HelixResponse,
    ) -> Result<(), GraphError> {
        Err(GraphError::PurgePending(
            "Collection 'c' cannot be created yet: object-store purge of the previously \
             dropped collection is still pending (s3 down); retry later"
                .into(),
        ))
    }

    fn test_state(
        tmp: &TempDir,
        router: HelixRouter,
        write_submitter: Arc<WriteSubmitter>,
    ) -> AsyncGatewayState {
        let config = Config::default();
        let graph = Arc::new(
            HelixGraphEngine::new(HelixGraphEngineOpts {
                path: tmp.path().join("graph").display().to_string(),
                config: config.clone(),
            })
            .unwrap(),
        );
        let collections =
            Arc::new(CollectionManager::new(tmp.path().join("data"), config.clone()).unwrap());
        let replication =
            Arc::new(ReplicationManager::new(Arc::clone(&collections), config).unwrap());
        AsyncGatewayState {
            graph,
            collections,
            replication,
            router: Arc::new(router),
            write_submitter,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn hot_collections_route_is_inline_and_404_unless_published() {
        use crate::helix_engine::storage_core::reader_warm;
        use axum::{body::to_bytes, extract::State, http::Request, response::IntoResponse};
        let _gate_lock = reader_warm::warm_gate_test_lock();
        let tmp = TempDir::new().unwrap();
        let state = test_state(&tmp, HelixRouter::new(None), WriteSubmitter::new());
        let get = || {
            Request::builder()
                .method("GET")
                .uri(reader_warm::HOT_COLLECTIONS_PATH)
                .body(axum::body::Body::empty())
                .unwrap()
        };

        reader_warm::publish_hot_list_for_test(None);
        let response = super::handle(State(state.clone()), get())
            .await
            .into_response();
        assert_eq!(response.status().as_u16(), 404);

        reader_warm::publish_hot_list_for_test(Some(vec!["a".into(), "b".into()]));
        let response = super::handle(State(state), get()).await.into_response();
        reader_warm::publish_hot_list_for_test(None);
        assert_eq!(response.status().as_u16(), 200);
        let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["collections"], serde_json::json!(["a", "b"]));
        assert_eq!(json["version"], 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn purge_pending_create_maps_to_retryable_503() {
        let tmp = TempDir::new().unwrap();
        let mut router = HelixRouter::new(None);
        router.add_route("PUT", "/collections/c", purge_pending_create);
        let state = test_state(&tmp, router, WriteSubmitter::new());
        let request = HelixRequest {
            method: "PUT".into(),
            headers: HashMap::new(),
            path: "/collections/c".into(),
            body: Vec::new(),
        };
        let response = run_router_with_guard(state, request, None, None, None).await;
        assert_eq!(response.status, 503);
        assert_eq!(
            response.headers.get("Retry-After").map(String::as_str),
            Some("10")
        );
        assert!(String::from_utf8_lossy(&response.body).contains("retry later"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_submit_keeps_writer_permit_until_blocking_work_finishes() {
        let _release = ReleaseHandlerOnDrop;
        let tmp = TempDir::new().unwrap();
        let mut router = HelixRouter::new(None);
        router.add_route("POST", "/test/slow-write", slow_write);
        let submitter = Arc::new(WriteSubmitter {
            writer_semaphore: Arc::new(Semaphore::new(1)),
        });
        let state = test_state(&tmp, router, Arc::clone(&submitter));
        let request = HelixRequest {
            method: "POST".into(),
            headers: HashMap::new(),
            path: "/test/slow-write".into(),
            body: Vec::new(),
        };

        // Client gives up while the blocking mutation is still running.
        let submit = submitter.submit(state, RouteClass::Write, request, None);
        let outcome = tokio::time::timeout(Duration::from_millis(200), submit).await;
        assert!(outcome.is_err(), "submit should still be blocked");
        assert!(HANDLER_STARTED.load(Ordering::Acquire));
        assert!(!HANDLER_DONE.load(Ordering::Acquire));
        assert_eq!(
            submitter.writer_semaphore.available_permits(),
            0,
            "abandoned blocking write must keep its submit permit"
        );

        HANDLER_RELEASE.store(true, Ordering::Release);
        for _ in 0..400 {
            if submitter.writer_semaphore.available_permits() == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(HANDLER_DONE.load(Ordering::Acquire));
        assert_eq!(submitter.writer_semaphore.available_permits(), 1);
    }
}
