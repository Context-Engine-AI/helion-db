use crate::helix_engine::graph_core::graph_core::HelixGraphEngine;
use crate::helix_engine::storage_core::{
    backend::StorageBackendConfig, backend_lsm::allow_lsm_blocking,
    collection_manager::CollectionManager, reader_warm, replication::ReplicationManager,
};
use crate::helix_engine::types::GraphError;
use crate::helix_gateway::api::ingest;
use flume::{Receiver, Sender};
use governor::{DefaultKeyedRateLimiter, Quota, RateLimiter};
use std::{
    num::NonZeroU32,
    sync::{Arc, LazyLock, Mutex},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

use crate::helix_gateway::router::router::{HelixRouter, RouterError};
use crate::protocol::request::{Request, RequestHead};
use crate::protocol::response::Response;

extern crate tokio;

use tokio::io::BufReader;
use tokio::net::TcpStream;

const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 300;

fn request_timeout_secs(input: Option<&str>) -> u64 {
    input
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_REQUEST_TIMEOUT_SECS)
        .max(1)
}

fn request_timeout() -> std::time::Duration {
    std::time::Duration::from_secs(request_timeout_secs(
        std::env::var("HELIX_REQUEST_TIMEOUT_SECS").ok().as_deref(),
    ))
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn queue_capacity(worker_count: usize) -> usize {
    if let Some(cap) = env_usize("HELIX_THREAD_POOL_QUEUE_CAP") {
        return cap;
    }
    let multiplier = env_usize("HELIX_THREAD_POOL_QUEUE_MULTIPLIER").unwrap_or(2);
    worker_count.saturating_mul(multiplier).max(1)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RouteClass {
    Probe,
    Read,
    Search,
    Scan,
    Write,
    Index,
    Maintenance,
    Admin,
}

impl RouteClass {
    pub(crate) fn as_label(self) -> &'static str {
        match self {
            RouteClass::Probe => "probe",
            RouteClass::Read => "read",
            RouteClass::Search => "search",
            RouteClass::Scan => "scan",
            RouteClass::Write => "write",
            RouteClass::Index => "index",
            RouteClass::Maintenance => "maintenance",
            RouteClass::Admin => "admin",
        }
    }

    pub(crate) fn is_write_like(self) -> bool {
        matches!(
            self,
            RouteClass::Write | RouteClass::Index | RouteClass::Maintenance | RouteClass::Admin
        )
    }
}

pub(crate) fn route_class(method: &str, path: &str) -> RouteClass {
    let route = route_label_for(path);
    if matches!(route, "/health" | "/ready" | "/metrics") {
        return RouteClass::Probe;
    }
    if route == "/_raft/*" {
        return RouteClass::Admin;
    }
    if path.starts_with("/v1/collections/") && path.ends_with("/ingest/stream") {
        return RouteClass::Write;
    }
    if path == "/v1/collections/create" || path == "/v1/collections/drop" {
        return RouteClass::Write;
    }
    if path == "/v1/collections/maintenance" {
        return RouteClass::Maintenance;
    }
    if matches!(
        path,
        "/v1/collections/list" | "/v1/collections/stats" | "/v1/collections/storage_bytes"
    ) {
        return RouteClass::Read;
    }
    if path.starts_with("/v1/collections/") && path.ends_with("/points/scan") {
        return RouteClass::Scan;
    }
    if path == "/v1/graph/delete_by_path"
        || path == "/v1/graph/delete_by_paths"
        || path == "/v1/graph/backfill_edge_path_index"
        || path == "/v1/graph/backfill_adjacency_from_points"
        || path == "/v1/graph/rebuild_adjacency_for_paths"
    {
        return RouteClass::Maintenance;
    }
    if path.starts_with("/v1/ingest/") {
        return RouteClass::Write;
    }
    if path.starts_with("/v1/graph/") {
        return RouteClass::Search;
    }

    match route {
        "/collections/{n}" => {
            if method == "GET" {
                RouteClass::Read
            } else {
                RouteClass::Write
            }
        }
        "/collections/{n}/points/search"
        | "/collections/{n}/points/query"
        | "/collections/{n}/points/hybrid_query" => RouteClass::Search,
        "/collections/{n}/points/scroll" => RouteClass::Scan,
        "/collections/{n}/points/count" | "/collections/{n}/facet" => RouteClass::Read,
        "/collections/{n}/points" => {
            if method == "PUT" {
                RouteClass::Write
            } else {
                RouteClass::Read
            }
        }
        "/collections/{n}/points/payload" | "/collections/{n}/points/delete" => RouteClass::Write,
        "/collections/{n}/index" => RouteClass::Index,
        "/collections/{n}/snapshots" => RouteClass::Maintenance,
        _ => {
            if matches!(method, "GET" | "HEAD" | "OPTIONS") {
                RouteClass::Read
            } else {
                RouteClass::Write
            }
        }
    }
}

pub(crate) struct AdmissionGuard {
    permit: Option<OwnedSemaphorePermit>,
    semaphore: Arc<Semaphore>,
    cap: usize,
    class: RouteClass,
}

impl Drop for AdmissionGuard {
    fn drop(&mut self) {
        drop(self.permit.take());
        let avail = self.semaphore.available_permits().min(self.cap);
        metrics::gauge!(
            "helix_route_admission_available",
            "class" => self.class.as_label().to_string()
        )
        .set(avail as f64);
        crate::diag::log(
            "release",
            "route_admission",
            &format!(
                "class={} avail={}/{}",
                self.class.as_label(),
                avail,
                self.cap
            ),
        );
    }
}

pub(crate) struct AdmissionReject {
    pub(crate) class: RouteClass,
    pub(crate) status: u16,
    pub(crate) current: usize,
    pub(crate) cap: usize,
}

type AdmissionSemaphore = (Arc<Semaphore>, usize);

fn semaphore_from_env(name: &str) -> Option<AdmissionSemaphore> {
    env_usize(name).map(|cap| (Arc::new(Semaphore::new(cap)), cap))
}

static WRITE_ADMISSION: LazyLock<Option<AdmissionSemaphore>> =
    LazyLock::new(|| semaphore_from_env("HELIX_WRITE_INFLIGHT_CAP"));
static INDEX_ADMISSION: LazyLock<Option<AdmissionSemaphore>> =
    LazyLock::new(|| semaphore_from_env("HELIX_INDEX_INFLIGHT_CAP"));
static MAINTENANCE_ADMISSION: LazyLock<Option<AdmissionSemaphore>> =
    LazyLock::new(|| semaphore_from_env("HELIX_MAINTENANCE_INFLIGHT_CAP"));
static WRITE_RATE_LIMITER: LazyLock<Option<DefaultKeyedRateLimiter<String>>> =
    LazyLock::new(|| {
        let per_second = env_usize("HELIX_WRITE_RATE_PER_COLLECTION_PER_SEC")?;
        let burst = env_usize("HELIX_WRITE_RATE_BURST")
            .unwrap_or(per_second)
            .max(per_second);
        let per_second = NonZeroU32::new(per_second.min(u32::MAX as usize) as u32)?;
        let burst = NonZeroU32::new(burst.min(u32::MAX as usize) as u32)?;
        let quota = Quota::per_second(per_second).allow_burst(burst);
        Some(RateLimiter::keyed(quota))
    });

fn route_admission_semaphore(class: RouteClass) -> Option<&'static AdmissionSemaphore> {
    match class {
        RouteClass::Write => WRITE_ADMISSION.as_ref(),
        RouteClass::Index => INDEX_ADMISSION.as_ref(),
        RouteClass::Maintenance | RouteClass::Admin => MAINTENANCE_ADMISSION.as_ref(),
        RouteClass::Probe | RouteClass::Read | RouteClass::Search | RouteClass::Scan => None,
    }
}

fn write_rate_limit_key(path: &str) -> String {
    if let Some(collection) = crate::helix_gateway::inflight::collection_from_path(path) {
        return collection.to_string();
    }
    let path = path.split('?').next().unwrap_or(path);
    if let Some(rest) = path.strip_prefix("/v1/collections/") {
        if let Some((collection, _)) = rest.split_once('/') {
            if !collection.is_empty() {
                return collection.to_string();
            }
        }
    }
    "__global__".to_string()
}

pub(crate) fn try_acquire_route_admission(
    class: RouteClass,
    path: &str,
) -> Result<Option<AdmissionGuard>, AdmissionReject> {
    if class.is_write_like() {
        if let Some(limiter) = WRITE_RATE_LIMITER.as_ref() {
            let key = write_rate_limit_key(path);
            if limiter.check_key(&key).is_err() {
                metrics::counter!(
                    "helix_write_rate_rejected_total",
                    "class" => class.as_label().to_string()
                )
                .increment(1);
                return Err(AdmissionReject {
                    class,
                    status: 429,
                    current: 0,
                    cap: 0,
                });
            }
        }
    }

    let Some(semaphore) = route_admission_semaphore(class) else {
        return Ok(None);
    };
    let (semaphore, cap) = semaphore;
    match Arc::clone(semaphore).try_acquire_owned() {
        Ok(permit) => {
            let avail = semaphore.available_permits();
            metrics::gauge!(
                "helix_route_admission_available",
                "class" => class.as_label().to_string()
            )
            .set(avail as f64);
            crate::diag::log(
                "got",
                "route_admission",
                &format!(
                    "class={} path={} avail={}/{}",
                    class.as_label(),
                    path,
                    avail,
                    *cap
                ),
            );
            Ok(Some(AdmissionGuard {
                permit: Some(permit),
                semaphore: Arc::clone(semaphore),
                cap: *cap,
                class,
            }))
        }
        Err(TryAcquireError::NoPermits) => {
            metrics::counter!(
                "helix_route_admission_rejected_total",
                "class" => class.as_label().to_string()
            )
            .increment(1);
            crate::diag::log(
                "reject",
                "route_admission",
                &format!("class={} path={} cap={}", class.as_label(), path, *cap),
            );
            Err(AdmissionReject {
                class,
                status: 503,
                current: *cap,
                cap: *cap,
            })
        }
        Err(TryAcquireError::Closed) => Err(AdmissionReject {
            class,
            status: 503,
            current: 0,
            cap: *cap,
        }),
    }
}

fn route_inflight_cap(method: &str, path: &str) -> Option<usize> {
    let route = route_label_for(path);
    match route {
        "/collections/{n}" if method != "GET" => env_usize("HELIX_INFLIGHT_CAP_WRITES"),
        "/collections/{n}" if method == "GET" => env_usize("HELIX_INFLIGHT_CAP_COLLECTION_INFO"),
        "/collections/{n}/points" if method == "PUT" => env_usize("HELIX_INFLIGHT_CAP_WRITES"),
        "/collections/{n}/points/payload"
        | "/collections/{n}/points/delete"
        | "/collections/{n}/index"
        | "/collections/{n}/snapshots" => env_usize("HELIX_INFLIGHT_CAP_WRITES"),
        "/collections/{n}/points/search"
        | "/collections/{n}/points/query"
        | "/collections/{n}/points/hybrid_query" => env_usize("HELIX_INFLIGHT_CAP_SEARCH"),
        "/collections/{n}/points/scroll" | "/v1/collections/{n}/points/scan" => {
            env_usize("HELIX_INFLIGHT_CAP_SCROLL")
        }
        "/collections/{n}/points/count" => env_usize("HELIX_INFLIGHT_CAP_COUNT"),
        "/collections/{n}/facet" => env_usize("HELIX_INFLIGHT_CAP_FACET"),
        "/v1/collections/maintenance" => env_usize("HELIX_MAINTENANCE_INFLIGHT_CAP"),
        "/v1/graph/*" | "/v1/ingest/*" => env_usize("HELIX_INFLIGHT_CAP_GRAPH_WRITES"),
        _ => None,
    }
}

fn collection_route_is_write(method: &str, path: &str) -> bool {
    route_class(method, path).is_write_like()
        && crate::helix_gateway::inflight::collection_from_path(path).is_some()
}

pub(crate) fn collection_inflight_cap(method: &str, path: &str) -> usize {
    if let Some(cap) = route_inflight_cap(method, path) {
        return cap;
    }
    if collection_route_is_write(method, path) {
        return std::env::var("HELIX_PER_COLLECTION_INFLIGHT_CAP")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
    }
    std::env::var("HELIX_PER_COLLECTION_READ_INFLIGHT_CAP")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
}

pub(crate) fn observe_collection_inflight_admission(
    class: RouteClass,
    path: &str,
    cap: usize,
    outcome: &'static str,
    current: Option<usize>,
) {
    if cap == 0 {
        return;
    }
    let route = route_label_for(path);
    let class_label = class.as_label();
    metrics::counter!(
        "helix_collection_inflight_admission_total",
        "class" => class_label,
        "route" => route,
        "outcome" => outcome,
        "cap" => cap.to_string(),
    )
    .increment(1);
    metrics::gauge!(
        "helix_collection_inflight_cap",
        "class" => class_label,
        "route" => route,
    )
    .set(cap as f64);
    if let Some(current) = current {
        metrics::histogram!(
            "helix_collection_inflight_rejected_current",
            "class" => class_label,
            "route" => route,
            "cap" => cap.to_string(),
        )
        .record(current as f64);
    }
}

fn route_timeout(
    method: &str,
    path: &str,
    default_timeout: std::time::Duration,
) -> std::time::Duration {
    let route = route_label_for(path);
    let env_name = match route {
        "/collections/{n}/points/scroll"
        | "/v1/collections/{n}/points/scan"
        | "/collections/{n}/points/count"
        | "/collections/{n}/facet" => Some("HELIX_READ_REQUEST_TIMEOUT_SECS"),
        "/collections/{n}" if method == "GET" => Some("HELIX_COLLECTION_INFO_TIMEOUT_SECS"),
        "/collections/{n}" => Some("HELIX_WRITE_REQUEST_TIMEOUT_SECS"),
        "/collections/{n}/points" if method == "PUT" => Some("HELIX_WRITE_REQUEST_TIMEOUT_SECS"),
        "/collections/{n}/points/payload"
        | "/collections/{n}/points/delete"
        | "/collections/{n}/index"
        | "/collections/{n}/snapshots" => Some("HELIX_WRITE_REQUEST_TIMEOUT_SECS"),
        "/v1/graph/*" | "/v1/ingest/*" => Some("HELIX_GRAPH_REQUEST_TIMEOUT_SECS"),
        _ => None,
    };
    env_name
        .and_then(|name| {
            std::env::var(name)
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .filter(|value| *value > 0)
        })
        .map(std::time::Duration::from_secs)
        .unwrap_or(default_timeout)
}

/// Map an HTTP status to a Prometheus-friendly class label (2xx/4xx/5xx…).
/// Keeps cardinality bounded in `helix_http_requests_total`.
pub(crate) fn status_class(status: u16) -> String {
    match status {
        100..=199 => "1xx",
        200..=299 => "2xx",
        300..=399 => "3xx",
        400..=499 => "4xx",
        500..=599 => "5xx",
        _ => "other",
    }
    .to_string()
}

/// Collapse a concrete request path into a bounded route label. The router
/// already knows pattern templates (`/collections/{name}/points/search`), but
/// reaching them from here would require refactoring the dispatch path; this
/// simpler collapse covers the hot endpoints without unbounded label growth.
pub(crate) fn route_label_for(path: &str) -> &'static str {
    // Strip query string before matching.
    let path = path.split('?').next().unwrap_or(path);
    if path == "/metrics" {
        return "/metrics";
    }
    if path == "/health" || path == "/healthz" || path == "/livez" || path == "/" {
        return "/health";
    }
    if path == "/ready"
        || path == "/readyz"
        || path == "/readyz/warm"
        || path == reader_warm::HOT_COLLECTIONS_PATH
    {
        return "/ready";
    }
    if path.starts_with("/_raft/") {
        return "/_raft/*";
    }
    if path.starts_with("/v1/graph/") {
        return "/v1/graph/*";
    }
    if path.starts_with("/v1/ingest/") {
        return "/v1/ingest/*";
    }
    if path.starts_with("/v1/collections/") && path.ends_with("/points/scan") {
        return "/v1/collections/{n}/points/scan";
    }
    if path == "/v1/collections/maintenance" {
        return "/v1/collections/maintenance";
    }
    if path.starts_with("/v1/collections/") {
        return "/v1/collections/*";
    }
    if path.starts_with("/collections/") {
        // Further bucketize the Qdrant-compatible hot paths so latency
        // histograms for search vs upsert vs scroll are separable.
        if path.ends_with("/points/search") {
            return "/collections/{n}/points/search";
        }
        if path.ends_with("/points/query") {
            return "/collections/{n}/points/query";
        }
        if path.ends_with("/points/hybrid_query") {
            return "/collections/{n}/points/hybrid_query";
        }
        if path.ends_with("/points/scroll") {
            return "/collections/{n}/points/scroll";
        }
        if path.ends_with("/points/count") {
            return "/collections/{n}/points/count";
        }
        if path.ends_with("/points/delete") {
            return "/collections/{n}/points/delete";
        }
        if path.ends_with("/points/payload") {
            return "/collections/{n}/points/payload";
        }
        if path.ends_with("/points") {
            return "/collections/{n}/points";
        }
        if path.contains("/snapshots") {
            return "/collections/{n}/snapshots";
        }
        if path.contains("/index") {
            return "/collections/{n}/index";
        }
        if path.contains("/facet") {
            return "/collections/{n}/facet";
        }
        return "/collections/{n}";
    }
    "other"
}

fn error_response(status: u16, body: &[u8]) -> Response {
    let mut response = Response::new();
    response.status = status;
    response.body = body.to_vec();
    response
}

pub(crate) fn lsm_reader_write_rejection(
    class: RouteClass,
    storage_backend: StorageBackendConfig,
) -> Option<Response> {
    if !class.is_write_like() || !storage_backend.is_reader() {
        return None;
    }
    metrics::counter!(
        "helix_lsm_reader_write_rejected_total",
        "class" => class.as_label().to_string()
    )
    .increment(1);
    let mut response = error_response(
        503,
        b"Helix LSM reader node is read-only; send mutations to the writer node",
    );
    response
        .headers
        .insert("Retry-After".to_string(), "1".to_string());
    response
        .headers
        .insert("Connection".to_string(), "close".to_string());
    response
        .headers
        .insert("X-Helix-Node-Role".to_string(), "reader".to_string());
    response.headers.insert(
        "X-Helix-Route-Class".to_string(),
        class.as_label().to_string(),
    );
    Some(response)
}

fn probe_response(path: &str) -> Option<Response> {
    let path = path.split('?').next().unwrap_or(path);
    let mut response = Response::new();
    response
        .headers
        .insert("Content-Type".to_string(), "application/json".to_string());

    match path {
        "/" | "/health" | "/healthz" | "/livez" => {
            response.status = 200;
            response.body = b"{\"status\":\"ok\"}".to_vec();
            Some(response)
        }
        reader_warm::HOT_COLLECTIONS_PATH => {
            match reader_warm::hot_collections_response_body() {
                Some(body) => {
                    response.status = 200;
                    response.body = body;
                }
                None => {
                    response.status = 404;
                    response.body = b"{\"status\":\"not_found\"}".to_vec();
                }
            }
            Some(response)
        }
        "/ready" | "/readyz" | "/readyz/warm" => {
            if reader_warm::startup_warm_ready() {
                response.status = 200;
                response.body = b"{\"status\":\"ready\"}".to_vec();
            } else {
                response.status = 503;
                response.body = b"{\"status\":\"warming\"}".to_vec();
            }
            Some(response)
        }
        _ => None,
    }
}

fn graph_error_response(error: GraphError) -> Response {
    let mut response = Response::new();
    if let Some(body) = error.fatal_collection_response_body() {
        response.status = 500;
        response.body = body;
        response
            .headers
            .insert("Content-Type".to_string(), "application/json".to_string());
        return response;
    }
    let msg = error.to_string();
    // Map common error shapes to more accurate HTTP status codes so clients
    // can distinguish "the collection is gone" (404 — safe to recreate or
    // skip) from actual server failures (500 — retry, alert). Without this,
    // CE's `_is_collection_not_found_error` detector sees a 500 with
    // "Collection 'X' not found" body only half the time (client library
    // may swallow the body), silently escalating to failure path.
    response.status = if matches!(error, GraphError::ResizeBackpressure(_))
        || msg.contains("Resize backpressure")
    {
        response
            .headers
            .insert("Retry-After".to_string(), "1".to_string());
        503
    } else if matches!(error, GraphError::MapFull)
        || msg.contains("LMDB map full")
        || msg.contains("maximum resize retries")
        || msg.contains("maximum size")
        || msg.contains("cannot grow further")
    {
        507
    } else if msg.contains("not found")
        || msg.contains("does not exist")
        || msg.contains("NotFound")
    {
        404
    } else {
        500
    };
    response.body = if response.status == 500 {
        b"Internal server error".to_vec()
    } else {
        msg.into_bytes()
    };
    response
}

/// Map a body-parse `io::Error` to an HTTP response. Oversized bodies
/// surface as `ErrorKind::FileTooLarge` from `Request::from_reader` and
/// are returned as 413. Everything else maps to 400.
fn body_parse_error_response(err: &std::io::Error) -> Response {
    if err.kind() == std::io::ErrorKind::FileTooLarge {
        error_response(413, b"Payload too large")
    } else {
        error_response(400, b"Bad request")
    }
}

pub struct Worker {
    pub id: usize,
    pub handle: JoinHandle<()>,
}

impl Worker {
    fn new(
        id: usize,
        graph_access: Arc<HelixGraphEngine>,
        collections: Arc<CollectionManager>,
        replication: Arc<ReplicationManager>,
        router: Arc<HelixRouter>,
        rx: Receiver<TcpStream>,
    ) -> Worker {
        let handle = tokio::spawn(async move {
            let request_timeout = request_timeout();
            loop {
                let conn = match rx.recv_async().await {
                    Ok(stream) => stream,
                    Err(e) => {
                        debug!(error = ?e, "Worker channel closed");
                        break; // channel closed = shutting down
                    }
                };

                let mut reader = BufReader::new(conn);

                // ── Keep-alive loop: serve multiple requests on the same TCP socket ──
                // Idle timeout: if no new request arrives within 30s, release the
                // worker back to the pool. Prevents worker starvation from idle
                // connections.
                let keep_alive_secs: u64 = std::env::var("HELIX_KEEP_ALIVE_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(30);
                let keep_alive_timeout = std::time::Duration::from_secs(keep_alive_secs);
                loop {
                    let request_head = match tokio::time::timeout(
                        keep_alive_timeout,
                        RequestHead::from_reader(&mut reader),
                    )
                    .await
                    {
                        Ok(Ok(request)) => request,
                        Ok(Err(_)) | Err(_) => {
                            // Connection closed, malformed, or idle timeout.
                            break;
                        }
                    };
                    let want_close = request_head
                        .headers
                        .get("connection") // headers are lowercased during parsing
                        .map(|v| v.eq_ignore_ascii_case("close"))
                        .unwrap_or(false);
                    let wants_msgpack = request_head
                        .headers
                        .get("accept")
                        .map(|v| {
                            v.contains("application/msgpack") || v.contains("application/x-msgpack")
                        })
                        .unwrap_or(false);

                    let req_method = request_head.method.clone();
                    let req_path = request_head.path.clone();
                    let req_start = std::time::Instant::now();
                    let route_timeout = route_timeout(&req_method, &req_path, request_timeout);
                    let req_class = route_class(&req_method, &req_path);

                    // Health/readiness probes must never wait behind blocking
                    // collection work. Serve them directly in the async worker
                    // before body parsing, inflight limiting, or spawn_blocking.
                    if request_head.method == "GET" {
                        if let Some(mut response) = probe_response(&req_path) {
                            let status = response.status;
                            let _ = response.send(reader.get_mut()).await;
                            let route_label = route_label_for(&req_path);
                            metrics::counter!(
                                "helix_http_requests_total",
                                "method" => req_method.clone(),
                                "route" => route_label.to_string(),
                                "status_class" => status_class(status),
                            )
                            .increment(1);
                            metrics::histogram!(
                                "helix_http_request_duration_ms",
                                "method" => req_method.clone(),
                                "route" => route_label.to_string(),
                            )
                            .record(req_start.elapsed().as_secs_f64() * 1000.0);
                            if want_close {
                                break;
                            }
                            continue;
                        }
                    }

                    if let Some(mut response) =
                        lsm_reader_write_rejection(req_class, collections.config().storage_backend)
                    {
                        let status = response.status;
                        let _ = response.send(reader.get_mut()).await;
                        let route_label = route_label_for(&req_path);
                        metrics::counter!(
                            "helix_http_requests_total",
                            "method" => req_method.clone(),
                            "route" => route_label.to_string(),
                            "status_class" => status_class(status),
                        )
                        .increment(1);
                        metrics::histogram!(
                            "helix_http_request_duration_ms",
                            "method" => req_method.clone(),
                            "route" => route_label.to_string(),
                        )
                        .record(req_start.elapsed().as_secs_f64() * 1000.0);
                        break;
                    }

                    let mut route_admission_guard =
                        match try_acquire_route_admission(req_class, &req_path) {
                            Ok(guard) => guard,
                            Err(reject) => {
                                warn!(
                                    path = %req_path,
                                    class = reject.class.as_label(),
                                    current = reject.current,
                                    cap = reject.cap,
                                    "Route admission saturated; returning retryable overload"
                                );
                                let mut resp = error_response(
                                    reject.status,
                                    b"Helix write/index capacity exhausted",
                                );
                                resp.headers
                                    .insert("Retry-After".to_string(), "1".to_string());
                                resp.headers
                                    .insert("Connection".to_string(), "close".to_string());
                                let _ = resp.send(reader.get_mut()).await;
                                break;
                            }
                        };

                    // Per-collection in-flight cap (opt-in via env var). The
                    // guard is held for the entire request lifetime, including
                    // the handler's spawn_blocking; dropping at end-of-request
                    // releases the slot. Disabled by default — when the cap is
                    // unset, `_inflight_guard` is a cheap no-op.
                    //
                    // Rejection closes the connection unconditionally: we stop
                    // before reading the request body, so any body bytes left on
                    // the socket would desynchronise a subsequent keep-alive
                    // request. `Connection: close` tells the client we're done,
                    // and we `break` out of the loop to drop the socket.
                    let mut inflight_guard =
                        match crate::helix_gateway::inflight::collection_from_path(&req_path) {
                            Some(name) => {
                                let cap = collection_inflight_cap(&req_method, &req_path);
                                match crate::helix_gateway::inflight::try_acquire_with_cap(
                                    name, cap,
                                ) {
                                    crate::helix_gateway::inflight::AcquireOutcome::Admitted(g) => {
                                        observe_collection_inflight_admission(
                                            req_class, &req_path, cap, "admitted", None,
                                        );
                                        Some(g)
                                    }
                                    crate::helix_gateway::inflight::AcquireOutcome::Rejected {
                                        current,
                                        cap,
                                    } => {
                                        observe_collection_inflight_admission(
                                            req_class,
                                            &req_path,
                                            cap,
                                            "rejected",
                                            Some(current),
                                        );
                                        warn!(
                                            path = %req_path,
                                            collection = name,
                                            current,
                                            cap,
                                            "Per-collection in-flight cap exceeded; returning 429"
                                        );
                                        let mut resp = error_response(
                                            429,
                                            b"Too many in-flight requests for collection",
                                        );
                                        resp.headers
                                            .insert("Retry-After".to_string(), "1".to_string());
                                        resp.headers
                                            .insert("Connection".to_string(), "close".to_string());
                                        let _ = resp.send(reader.get_mut()).await;
                                        break;
                                    }
                                }
                            }
                            None => None,
                        };

                    let mut response = if request_head.method == "POST" {
                        if let Some(collection) = ingest::stream_collection_name(&request_head.path)
                        {
                            match ingest::handle_ingest_stream_socket(
                                request_head.clone(),
                                &mut reader,
                                collection,
                                Arc::clone(&collections),
                                Arc::clone(&replication),
                            )
                            .await
                            {
                                Ok(resp) => resp,
                                Err(e) => {
                                    error!(path = %req_path, error = ?e, "Stream ingest failed");
                                    graph_error_response(e)
                                }
                            }
                        } else {
                            let request = match Request::from_reader(&mut reader, request_head)
                                .await
                            {
                                Ok(request) => request,
                                Err(e) => {
                                    warn!(path = %req_path, error = ?e, "Bad request body");
                                    let _ =
                                        body_parse_error_response(&e).send(reader.get_mut()).await;
                                    continue;
                                }
                            };

                            let router_ref = Arc::clone(&router);
                            let graph_ref = Arc::clone(&graph_access);
                            let collections_ref = Arc::clone(&collections);
                            let replication_ref = Arc::clone(&replication);
                            // `spawn_blocking` cannot be cancelled by dropping
                            // its JoinHandle. Keep the route/collection slot
                            // inside the blocking handler so a timed-out HTTP
                            // request does not release capacity while LMDB
                            // work continues in the background.
                            let handler_inflight_guard = inflight_guard.take();
                            let route_admission_guard = route_admission_guard.take();
                            let handler_fut = tokio::task::spawn_blocking(move || {
                                allow_lsm_blocking(|| {
                                    let _handler_inflight_guard = handler_inflight_guard;
                                    let _route_admission_guard = route_admission_guard;
                                    let mut resp = Response::new();
                                    if let Err(e) = router_ref.handle(
                                        graph_ref,
                                        collections_ref,
                                        replication_ref,
                                        request,
                                        &mut resp,
                                    ) {
                                        error!(error = ?e, "Handler error");
                                        // Route through graph_error_response so "not found"
                                        // errors surface as 404 and let CE's UPSERT_RECOVERY
                                        // detector trigger ensure+retry instead of treating
                                        // every missing-collection upsert as a server bug.
                                        resp = graph_error_response(e);
                                    }
                                    resp
                                })
                            });

                            match tokio::time::timeout(route_timeout, handler_fut).await {
                                Ok(Ok(resp)) => resp,
                                Ok(Err(e)) => {
                                    error!(path = %req_path, error = ?e, "Handler panicked");
                                    error_response(500, b"Internal server error")
                                }
                                Err(_) => {
                                    error!(path = %req_path, timeout = ?route_timeout, "Request timeout");
                                    error_response(504, b"Gateway timeout")
                                }
                            }
                        }
                    } else {
                        let request = match Request::from_reader(&mut reader, request_head).await {
                            Ok(request) => request,
                            Err(e) => {
                                warn!(path = %req_path, error = ?e, "Bad request body");
                                let _ = body_parse_error_response(&e).send(reader.get_mut()).await;
                                continue;
                            }
                        };

                        let router_ref = Arc::clone(&router);
                        let graph_ref = Arc::clone(&graph_access);
                        let collections_ref = Arc::clone(&collections);
                        let replication_ref = Arc::clone(&replication);
                        // `spawn_blocking` keeps running after a timeout. Move
                        // the limiter guard into the blocking closure so timed
                        // out reads/searches still count until they truly exit.
                        let handler_inflight_guard = inflight_guard.take();
                        let route_admission_guard = route_admission_guard.take();
                        let handler_fut = tokio::task::spawn_blocking(move || {
                            allow_lsm_blocking(|| {
                                let _handler_inflight_guard = handler_inflight_guard;
                                let _route_admission_guard = route_admission_guard;
                                let mut resp = Response::new();
                                if let Err(e) = router_ref.handle(
                                    graph_ref,
                                    collections_ref,
                                    replication_ref,
                                    request,
                                    &mut resp,
                                ) {
                                    error!(error = ?e, "Handler error");
                                    resp = graph_error_response(e);
                                }
                                resp
                            })
                        });

                        match tokio::time::timeout(route_timeout, handler_fut).await {
                            Ok(Ok(resp)) => resp,
                            Ok(Err(e)) => {
                                error!(path = %req_path, error = ?e, "Handler panicked");
                                error_response(500, b"Internal server error")
                            }
                            Err(_) => {
                                error!(path = %req_path, timeout = ?route_timeout, "Request timeout");
                                error_response(504, b"Gateway timeout")
                            }
                        }
                    };

                    // Content-type negotiation: if client accepts msgpack, re-encode.
                    if wants_msgpack && response.status == 200 && !response.body.is_empty() {
                        if let Ok(json_val) =
                            sonic_rs::from_slice::<sonic_rs::Value>(&response.body)
                        {
                            if let Ok(msgpack_bytes) = rmp_serde::to_vec(&json_val) {
                                response.body = msgpack_bytes;
                                response.headers.insert(
                                    "Content-Type".to_string(),
                                    "application/msgpack".to_string(),
                                );
                            }
                        }
                    }

                    // Set keep-alive header unless the client asked to close.
                    if !want_close {
                        response
                            .headers
                            .insert("Connection".to_string(), "keep-alive".to_string());
                    }

                    // Log completed request at appropriate level
                    let elapsed = req_start.elapsed();

                    // Emit request metrics. `route_label` bucketizes paths to
                    // avoid unbounded label cardinality (per-collection paths
                    // would explode the series count at 250+ collections).
                    let route_label = route_label_for(&req_path);
                    let status_label = status_class(response.status);
                    metrics::counter!(
                        "helix_http_requests_total",
                        "method" => req_method.clone(),
                        "route" => route_label.to_string(),
                        "status" => status_label,
                    )
                    .increment(1);
                    metrics::histogram!(
                        "helix_http_request_duration_seconds",
                        "method" => req_method.clone(),
                        "route" => route_label.to_string(),
                    )
                    .record(elapsed.as_secs_f64());

                    if response.status >= 500 {
                        error!(
                            method = %req_method, path = %req_path,
                            status = response.status, elapsed_ms = elapsed.as_millis(),
                            "Request failed"
                        );
                    } else if response.status >= 400 {
                        warn!(
                            method = %req_method, path = %req_path,
                            status = response.status, elapsed_ms = elapsed.as_millis(),
                            "Client error"
                        );
                    } else if elapsed.as_millis() > 1000 {
                        warn!(
                            method = %req_method, path = %req_path,
                            status = response.status, elapsed_ms = elapsed.as_millis(),
                            "Slow request"
                        );
                    } else {
                        debug!(
                            method = %req_method, path = %req_path,
                            status = response.status, elapsed_ms = elapsed.as_millis(),
                            "Request complete"
                        );
                    }

                    if let Err(e) = response.send(reader.get_mut()).await {
                        warn!(error = ?e, "Response send failed (socket gone)");
                        break;
                    }

                    if want_close {
                        break;
                    }
                } // end keep-alive loop
            }
        });

        Worker { id, handle }
    }
}

pub struct ThreadPool {
    pub sender: Sender<TcpStream>,
    pub num_unused_workers: Mutex<usize>,
    pub num_used_workers: Mutex<usize>,
    pub workers: Vec<Worker>,
}

impl ThreadPool {
    pub fn new(
        size: usize,
        graph: Arc<HelixGraphEngine>,
        collections: Arc<CollectionManager>,
        replication: Arc<ReplicationManager>,
        router: Arc<HelixRouter>,
    ) -> Result<ThreadPool, RouterError> {
        assert!(
            size > 0,
            "Expected number of threads in thread pool to be more than 0, got {}",
            size
        );

        // Bounded channel: backpressure when all workers are busy. Defaults
        // to 2× pool size and can be tuned without rebuilds for prod traffic.
        let queue_cap = queue_capacity(size);
        let (tx, rx) = flume::bounded::<TcpStream>(queue_cap);
        let mut workers = Vec::with_capacity(size);
        for id in 0..size {
            workers.push(Worker::new(
                id,
                Arc::clone(&graph),
                Arc::clone(&collections),
                Arc::clone(&replication),
                Arc::clone(&router),
                rx.clone(),
            ));
        }
        info!(
            workers = workers.len(),
            queue_cap, "Thread pool initialized"
        );

        Ok(ThreadPool {
            sender: tx,
            num_unused_workers: Mutex::new(size),
            num_used_workers: Mutex::new(0),
            workers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        collection_inflight_cap, graph_error_response, lsm_reader_write_rejection, probe_response,
        queue_capacity, request_timeout_secs, route_class, route_inflight_cap, route_label_for,
        route_timeout, RouteClass,
    };
    use crate::helix_engine::storage_core::backend::{LsmRole, LsmStorage, StorageBackendConfig};
    use crate::helix_engine::storage_core::reader_warm;
    use crate::helix_engine::types::GraphError;
    use std::sync::{LazyLock, Mutex};

    static ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    #[test]
    fn request_timeout_defaults_and_clamps() {
        assert_eq!(request_timeout_secs(None), 300);
        assert_eq!(request_timeout_secs(Some("0")), 1);
        assert_eq!(request_timeout_secs(Some("45")), 45);
        assert_eq!(request_timeout_secs(Some("not-a-number")), 300);
    }

    #[test]
    fn route_labels_bucketize_hot_paths() {
        assert_eq!(
            route_label_for("/collections/my-coll/points/scroll"),
            "/collections/{n}/points/scroll"
        );
        assert_eq!(
            route_label_for("/v1/collections/my-coll/points/scan"),
            "/v1/collections/{n}/points/scan"
        );
        assert_eq!(
            route_label_for("/v1/collections/maintenance"),
            "/v1/collections/maintenance"
        );
        assert_eq!(
            route_label_for("/collections/my-coll/points"),
            "/collections/{n}/points"
        );
        assert_eq!(
            route_label_for("/collections/my-coll/points/hybrid_query"),
            "/collections/{n}/points/hybrid_query"
        );
        assert_eq!(route_label_for("/collections/my-coll"), "/collections/{n}");
        assert_eq!(route_label_for("/v1/graph/delete_by_paths"), "/v1/graph/*");
        assert_eq!(route_label_for("/health"), "/health");
    }

    #[test]
    fn warm_readiness_probe_gates_ready_but_never_health() {
        let _gate_lock = reader_warm::warm_gate_test_lock();
        assert_eq!(route_label_for("/readyz/warm"), "/ready");
        assert_eq!(route_class("GET", "/readyz/warm"), RouteClass::Probe);
        let status = |path: &str| probe_response(path).map(|response| response.status);

        reader_warm::set_startup_warm_pending(true);
        let warming = ["/ready", "/readyz", "/readyz/warm", "/health"].map(status);
        reader_warm::set_startup_warm_pending(false);
        assert_eq!(warming, [Some(503), Some(503), Some(503), Some(200)]);

        for path in ["/ready", "/readyz", "/readyz/warm", "/health"] {
            assert_eq!(status(path), Some(200), "{path} after warm");
        }
    }

    #[test]
    fn hot_collections_probe_is_404_unless_reader_published() {
        let _gate_lock = reader_warm::warm_gate_test_lock();
        let path = reader_warm::HOT_COLLECTIONS_PATH;
        assert_eq!(route_label_for(path), "/ready");
        assert_eq!(route_class("GET", path), RouteClass::Probe);
        // No reader warm task has published in this test → writer behavior.
        assert_eq!(probe_response(path).map(|r| r.status), Some(404));
    }

    #[test]
    fn route_classes_keep_reads_out_of_write_admission() {
        assert_eq!(route_class("GET", "/health"), RouteClass::Probe);
        assert_eq!(route_class("GET", "/collections/c"), RouteClass::Read);
        assert_eq!(
            route_class("POST", "/collections/c/points/search"),
            RouteClass::Search
        );
        assert_eq!(
            route_class("POST", "/collections/c/points/hybrid_query"),
            RouteClass::Search
        );
        assert_eq!(
            route_class("POST", "/collections/c/points/count"),
            RouteClass::Read
        );
        assert_eq!(
            route_class("POST", "/collections/c/points/scroll"),
            RouteClass::Scan
        );
        assert_eq!(
            route_class("POST", "/v1/collections/c/points/scan"),
            RouteClass::Scan
        );
        assert_eq!(
            route_class("POST", "/collections/c/points"),
            RouteClass::Read
        );
        assert_eq!(
            route_class("PUT", "/collections/c/points"),
            RouteClass::Write
        );
        assert_eq!(
            route_class("PUT", "/collections/c/index"),
            RouteClass::Index
        );
        assert_eq!(
            route_class("POST", "/v1/graph/delete_by_paths"),
            RouteClass::Maintenance
        );
        assert_eq!(
            route_class("POST", "/v1/graph/backfill_adjacency_from_points"),
            RouteClass::Maintenance
        );
        assert_eq!(
            route_class("POST", "/v1/collections/maintenance"),
            RouteClass::Maintenance
        );
        assert_eq!(
            route_class("POST", "/v1/collections/repo_graph/ingest/stream"),
            RouteClass::Write
        );
        assert_eq!(route_class("POST", "/v1/graph/callers"), RouteClass::Search);
    }

    #[test]
    fn lsm_reader_role_rejects_write_like_routes() {
        let reader = StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role: LsmRole::Reader,
        };
        let writer = StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role: LsmRole::Writer,
        };

        let response = lsm_reader_write_rejection(RouteClass::Write, reader).unwrap();
        assert_eq!(response.status, 503);
        assert_eq!(
            response.headers.get("Retry-After").map(String::as_str),
            Some("1")
        );
        assert_eq!(
            response
                .headers
                .get("X-Helix-Node-Role")
                .map(String::as_str),
            Some("reader")
        );
        assert_eq!(
            response
                .headers
                .get("X-Helix-Route-Class")
                .map(String::as_str),
            Some("write")
        );
        assert!(String::from_utf8_lossy(&response.body).contains("read-only"));
        assert!(lsm_reader_write_rejection(RouteClass::Index, reader).is_some());
        assert!(lsm_reader_write_rejection(RouteClass::Maintenance, reader).is_some());
        assert!(lsm_reader_write_rejection(RouteClass::Read, reader).is_none());
        assert!(lsm_reader_write_rejection(RouteClass::Search, reader).is_none());
        assert!(lsm_reader_write_rejection(RouteClass::Write, writer).is_none());
    }

    #[test]
    fn graph_error_response_maps_map_full_to_507() {
        let response = graph_error_response(GraphError::MapFull);

        assert_eq!(response.status, 507);
        assert!(
            String::from_utf8_lossy(&response.body).contains("LMDB map full"),
            "body: {}",
            String::from_utf8_lossy(&response.body)
        );
    }

    #[test]
    fn graph_error_response_maps_resize_backpressure_to_retryable_503() {
        let response =
            graph_error_response(GraphError::ResizeBackpressure("gate busy".to_string()));

        assert_eq!(response.status, 503);
        assert_eq!(
            response.headers.get("Retry-After").map(String::as_str),
            Some("1")
        );
        assert!(
            String::from_utf8_lossy(&response.body).contains("Resize backpressure"),
            "body: {}",
            String::from_utf8_lossy(&response.body)
        );
    }

    #[test]
    fn graph_error_response_maps_fatal_collection_storage_to_json_500() {
        let response = graph_error_response(GraphError::FatalCollectionStorage {
            code: "mdb_problem",
            message: "txn should abort".into(),
        });

        assert_eq!(response.status, 500);
        assert_eq!(
            response.headers.get("Content-Type").map(String::as_str),
            Some("application/json")
        );
        let body = String::from_utf8_lossy(&response.body);
        assert!(body.contains("\"code\":\"mdb_problem\""), "body: {body}");
        assert!(body.contains("\"fatal\":true"), "body: {body}");
        assert!(
            body.contains("\"action\":\"quarantine_rebuild\""),
            "body: {body}"
        );
    }

    #[test]
    fn graph_error_response_hides_unclassified_500_details() {
        let response = graph_error_response(GraphError::StorageError("secret path leaked".into()));

        assert_eq!(response.status, 500);
        assert_eq!(response.body, b"Internal server error".to_vec());
    }

    #[test]
    fn route_specific_timeouts_override_default() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("HELIX_READ_REQUEST_TIMEOUT_SECS");
        std::env::remove_var("HELIX_WRITE_REQUEST_TIMEOUT_SECS");
        std::env::remove_var("HELIX_GRAPH_REQUEST_TIMEOUT_SECS");
        std::env::remove_var("HELIX_COLLECTION_INFO_TIMEOUT_SECS");

        let default_timeout = std::time::Duration::from_secs(60);
        assert_eq!(
            route_timeout("POST", "/collections/c/points/scroll", default_timeout),
            default_timeout
        );

        std::env::set_var("HELIX_READ_REQUEST_TIMEOUT_SECS", "7");
        std::env::set_var("HELIX_WRITE_REQUEST_TIMEOUT_SECS", "90");
        std::env::set_var("HELIX_GRAPH_REQUEST_TIMEOUT_SECS", "120");
        std::env::set_var("HELIX_COLLECTION_INFO_TIMEOUT_SECS", "5");

        assert_eq!(
            route_timeout("POST", "/collections/c/points/scroll", default_timeout),
            std::time::Duration::from_secs(7)
        );
        assert_eq!(
            route_timeout("PUT", "/collections/c/points", default_timeout),
            std::time::Duration::from_secs(90)
        );
        assert_eq!(
            route_timeout("POST", "/v1/graph/delete_by_paths", default_timeout),
            std::time::Duration::from_secs(120)
        );
        assert_eq!(
            route_timeout("GET", "/collections/c", default_timeout),
            std::time::Duration::from_secs(5)
        );
        assert_eq!(
            route_timeout("PUT", "/collections/c", default_timeout),
            std::time::Duration::from_secs(90)
        );

        std::env::remove_var("HELIX_READ_REQUEST_TIMEOUT_SECS");
        std::env::remove_var("HELIX_WRITE_REQUEST_TIMEOUT_SECS");
        std::env::remove_var("HELIX_GRAPH_REQUEST_TIMEOUT_SECS");
        std::env::remove_var("HELIX_COLLECTION_INFO_TIMEOUT_SECS");
    }

    #[test]
    fn route_specific_caps_override_global_cap() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("HELIX_PER_COLLECTION_INFLIGHT_CAP");
        assert_eq!(
            route_inflight_cap("POST", "/collections/c/points/scroll"),
            None
        );
        assert_eq!(route_inflight_cap("GET", "/collections/c"), None);

        std::env::set_var("HELIX_PER_COLLECTION_INFLIGHT_CAP", "99");
        std::env::set_var("HELIX_INFLIGHT_CAP_WRITES", "17");
        std::env::set_var("HELIX_MAINTENANCE_INFLIGHT_CAP", "2");
        assert_eq!(
            collection_inflight_cap("POST", "/collections/c/points/scroll"),
            0
        );
        assert_eq!(collection_inflight_cap("GET", "/collections/c"), 0);
        assert_eq!(collection_inflight_cap("POST", "/collections/c/points"), 0);
        assert_eq!(route_inflight_cap("PUT", "/collections/c"), Some(17));
        assert_eq!(
            route_inflight_cap("POST", "/v1/collections/maintenance"),
            Some(2)
        );
        assert_eq!(collection_inflight_cap("PUT", "/collections/c"), 17);
        assert_eq!(collection_inflight_cap("PUT", "/collections/c/points"), 17);

        std::env::remove_var("HELIX_PER_COLLECTION_INFLIGHT_CAP");
        std::env::remove_var("HELIX_INFLIGHT_CAP_WRITES");
        std::env::remove_var("HELIX_MAINTENANCE_INFLIGHT_CAP");
    }

    #[test]
    fn route_specific_read_caps_apply_per_collection() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("HELIX_INFLIGHT_CAP_SEARCH", "16");
        std::env::set_var("HELIX_INFLIGHT_CAP_SCROLL", "8");
        std::env::set_var("HELIX_INFLIGHT_CAP_COUNT", "4");
        std::env::set_var("HELIX_INFLIGHT_CAP_FACET", "3");
        std::env::set_var("HELIX_INFLIGHT_CAP_COLLECTION_INFO", "2");

        assert_eq!(
            collection_inflight_cap("POST", "/collections/c/points/search"),
            16
        );
        assert_eq!(
            collection_inflight_cap("POST", "/collections/c/points/hybrid_query"),
            16
        );
        assert_eq!(
            collection_inflight_cap("POST", "/collections/c/points/scroll"),
            8
        );
        assert_eq!(
            collection_inflight_cap("POST", "/v1/collections/c/points/scan"),
            8
        );
        assert_eq!(
            collection_inflight_cap("POST", "/collections/c/points/count"),
            4
        );
        assert_eq!(collection_inflight_cap("POST", "/collections/c/facet"), 3);
        assert_eq!(collection_inflight_cap("GET", "/collections/c"), 2);

        std::env::remove_var("HELIX_INFLIGHT_CAP_SEARCH");
        std::env::remove_var("HELIX_INFLIGHT_CAP_SCROLL");
        std::env::remove_var("HELIX_INFLIGHT_CAP_COUNT");
        std::env::remove_var("HELIX_INFLIGHT_CAP_FACET");
        std::env::remove_var("HELIX_INFLIGHT_CAP_COLLECTION_INFO");
    }

    #[test]
    fn fallback_collection_cap_applies_only_to_writes() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("HELIX_INFLIGHT_CAP_WRITES");
        std::env::set_var("HELIX_PER_COLLECTION_INFLIGHT_CAP", "23");

        assert_eq!(collection_inflight_cap("GET", "/collections/c"), 0);
        assert_eq!(
            collection_inflight_cap("POST", "/collections/c/points/search"),
            0
        );
        assert_eq!(
            collection_inflight_cap("POST", "/collections/c/points/count"),
            0
        );
        assert_eq!(collection_inflight_cap("POST", "/collections/c/points"), 0);
        assert_eq!(collection_inflight_cap("PUT", "/collections/c/points"), 23);
        assert_eq!(
            collection_inflight_cap("POST", "/collections/c/points/delete"),
            23
        );

        std::env::remove_var("HELIX_PER_COLLECTION_INFLIGHT_CAP");
    }

    #[test]
    fn queue_capacity_defaults_to_double_worker_count() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("HELIX_THREAD_POOL_QUEUE_CAP");
        std::env::remove_var("HELIX_THREAD_POOL_QUEUE_MULTIPLIER");
        assert_eq!(queue_capacity(512), 1024);
        assert_eq!(queue_capacity(0), 1);
    }

    #[test]
    fn queue_capacity_allows_env_override() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("HELIX_THREAD_POOL_QUEUE_CAP", "64");
        std::env::set_var("HELIX_THREAD_POOL_QUEUE_MULTIPLIER", "9");
        assert_eq!(queue_capacity(512), 64);
        std::env::remove_var("HELIX_THREAD_POOL_QUEUE_CAP");

        std::env::set_var("HELIX_THREAD_POOL_QUEUE_MULTIPLIER", "3");
        assert_eq!(queue_capacity(10), 30);
        std::env::remove_var("HELIX_THREAD_POOL_QUEUE_MULTIPLIER");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn timed_out_blocking_handler_keeps_inflight_slot_until_done() {
        use crate::helix_gateway::inflight::{forget, try_acquire_with_cap, AcquireOutcome};
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };
        use std::time::Duration;

        let collection = "unit-test-timeout-keeps-slot";
        forget(collection);

        let guard = match try_acquire_with_cap(collection, 1) {
            AcquireOutcome::Admitted(guard) => Some(guard),
            AcquireOutcome::Rejected { .. } => panic!("first acquire should pass"),
        };

        let done = Arc::new(AtomicBool::new(false));
        let done_for_task = Arc::clone(&done);
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let handler = tokio::task::spawn_blocking(move || {
            let _handler_guard = guard;
            let _ = started_tx.send(());
            while !done_for_task.load(Ordering::Acquire) {
                std::thread::sleep(Duration::from_millis(5));
            }
        });

        started_rx.await.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(10), handler)
                .await
                .is_err(),
            "spawn_blocking handle should time out while task keeps running",
        );

        match try_acquire_with_cap(collection, 1) {
            AcquireOutcome::Rejected { current, cap } => {
                assert_eq!(current, 1);
                assert_eq!(cap, 1);
            }
            AcquireOutcome::Admitted(_) => panic!("slot released before blocking task completed"),
        }

        done.store(true, Ordering::Release);
        tokio::time::sleep(Duration::from_millis(25)).await;
        match try_acquire_with_cap(collection, 1) {
            AcquireOutcome::Admitted(_guard) => {}
            AcquireOutcome::Rejected { current, cap } => {
                panic!("slot did not release after task completion: current={current}, cap={cap}")
            }
        }
        forget(collection);
    }
}
