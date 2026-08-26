//! Prometheus-style metrics endpoint.
//!
//! Installs a global `metrics` recorder the first time `/metrics` is hit (or
//! `init()` is called explicitly from `main`). Hot paths call `metrics::counter!`,
//! `metrics::histogram!`, and `metrics::gauge!` directly; this module only owns
//! the recorder lifecycle and the HTTP endpoint.
//!
//! Keeping the recorder lazy means the metrics subsystem has zero cost if the
//! `/metrics` endpoint is never scraped — important for embedded / test usage
//! where we don't want a background Prometheus exporter task.

use std::sync::OnceLock;

use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};

use crate::helix_engine::types::GraphError;
use crate::helix_gateway::router::router::HandlerInput;
use crate::protocol::response::Response;

static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();

const MS_BUCKETS: &[f64] = &[
    0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0,
    4096.0, 8192.0, 16384.0, 32768.0, 65536.0, 131072.0,
];
const SECONDS_BUCKETS: &[f64] = &[
    0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.0, 2.0, 4.0, 8.0, 16.0,
    32.0, 64.0, 128.0, 300.0,
];
const COUNT_BUCKETS: &[f64] = &[
    1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0,
    16384.0, 32768.0, 65536.0, 131072.0, 262144.0, 524288.0, 1048576.0,
];
const BYTES_BUCKETS: &[f64] = &[
    128.0,
    256.0,
    512.0,
    1024.0,
    2048.0,
    4096.0,
    8192.0,
    16384.0,
    32768.0,
    65536.0,
    131072.0,
    262144.0,
    524288.0,
    1048576.0,
    2097152.0,
    4194304.0,
    8388608.0,
    16777216.0,
    33554432.0,
    67108864.0,
    134217728.0,
];
const RATIO_BUCKETS: &[f64] = &[
    0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0,
];

const SECONDS_HISTOGRAMS: &[&str] = &[
    "helix_http_request_duration_seconds",
    "helix_snapshot_duration_seconds",
];
const COUNT_HISTOGRAMS: &[&str] = &[
    "helix_collection_inflight_rejected_current",
    "helix_delete_chunk_ids",
    "helix_delete_chunk_size_effective",
    "helix_delete_existing_point_ids",
    "helix_delete_indexed_segments",
    "helix_dense_segment_search_results",
    "helix_facet_candidates",
    "helix_facet_distinct_values",
    "helix_facet_entries_scanned",
    "helix_facet_matched_items",
    "helix_facet_returned_hits",
    "helix_index_job_spaces",
    "helix_graph_delete_items",
    "helix_point_scan_returned_points",
    "helix_point_scan_scanned_points",
    "helix_qdrant_delete_matched_ids",
    "helix_sparse_upsert_batch_dedup_count",
];
const BYTES_HISTOGRAMS: &[&str] = &[
    "helix_facet_response_bytes",
    "helix_point_scan_response_bytes",
];
const RATIO_HISTOGRAMS: &[&str] = &["helix_sparse_upsert_batch_dedup_density"];

fn prometheus_builder() -> PrometheusBuilder {
    let mut builder = PrometheusBuilder::new()
        .set_buckets(MS_BUCKETS)
        .expect("ms histogram buckets are non-empty");
    builder = set_bucket_overrides(builder, SECONDS_HISTOGRAMS, SECONDS_BUCKETS);
    builder = set_bucket_overrides(builder, COUNT_HISTOGRAMS, COUNT_BUCKETS);
    builder = set_bucket_overrides(builder, BYTES_HISTOGRAMS, BYTES_BUCKETS);
    set_bucket_overrides(builder, RATIO_HISTOGRAMS, RATIO_BUCKETS)
}

fn set_bucket_overrides(
    mut builder: PrometheusBuilder,
    metrics: &[&str],
    buckets: &[f64],
) -> PrometheusBuilder {
    for metric in metrics {
        builder = builder
            .set_buckets_for_metric(Matcher::Full((*metric).to_string()), buckets)
            .expect("histogram override buckets are non-empty");
    }
    builder
}

/// Install the global Prometheus recorder. Safe to call multiple times; only
/// the first call actually installs. Returns the handle used to render metrics.
fn get_or_init_handle() -> &'static PrometheusHandle {
    HANDLE.get_or_init(|| {
        // `install_recorder()` returns the handle and registers the recorder
        // as the global `metrics` sink. If another recorder is already set
        // (e.g. a test harness installed one), we fall back to a detached
        // recorder whose handle still works for rendering — it just won't
        // receive any data. In production this never happens because we own
        // the process.
        prometheus_builder()
            .install_recorder()
            .unwrap_or_else(|_| prometheus_builder().build_recorder().handle())
    })
}

/// Optional explicit initialization from `main`. Calling this early lets
/// metrics from startup paths (collection scan, WAL recovery) be captured.
pub fn init() {
    let _ = get_or_init_handle();
}

/// GET /metrics — renders Prometheus text format.
///
/// When `HELIX_METRICS_TOKEN` is set in the environment, requests must
/// include a matching `Authorization: Bearer <token>` header. Unset means
/// unauthenticated access (safe for clusters that expose `/metrics` only on
/// an internal network). Comparison is length-checked and constant-time to
/// avoid leaking the token via timing.
pub fn handle_metrics(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    if let Some(expected) = metrics_token() {
        let provided = input
            .request
            .headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("authorization"))
            .and_then(|(_, v)| {
                v.strip_prefix("Bearer ")
                    .or_else(|| v.strip_prefix("bearer "))
            });
        let ok = match provided {
            Some(token) => constant_time_eq(token.as_bytes(), expected.as_bytes()),
            None => false,
        };
        if !ok {
            response.status = 401;
            response.body = b"unauthorized".to_vec();
            response
                .headers
                .insert("WWW-Authenticate".to_string(), "Bearer".to_string());
            return Ok(());
        }
    }

    let handle = get_or_init_handle();

    // Refresh dynamic gauges that aren't updated on every request. These
    // reads are cheap (single atomic loads or a read lock).
    refresh_runtime_gauges(input);

    let body = handle.render();
    response.status = 200;
    response.body = body.into_bytes();
    response.headers.insert(
        "Content-Type".to_string(),
        "text/plain; version=0.0.4; charset=utf-8".to_string(),
    );
    Ok(())
}

/// Read `HELIX_METRICS_TOKEN` lazily. Cached after first call so the scrape
/// path is a single atomic load. An empty env var is treated the same as
/// "not set" (allow anonymous) — this matches how ops commonly default env
/// values in manifests.
fn metrics_token() -> Option<&'static str> {
    static TOKEN: OnceLock<Option<String>> = OnceLock::new();
    TOKEN
        .get_or_init(|| {
            std::env::var("HELIX_METRICS_TOKEN")
                .ok()
                .filter(|s| !s.is_empty())
        })
        .as_deref()
}

/// Constant-time byte comparison so a timing side-channel can't reveal the
/// configured token. `subtle` would be idiomatic but we avoid a new dep for
/// a two-line routine.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Sample gauges that aren't updated in the hot path. Called on every scrape
/// so Prometheus sees fresh values without background threads.
fn refresh_runtime_gauges(input: &HandlerInput) {
    let collections = &input.collections;
    // Total collections visible on disk (manifest) vs currently resident in
    // the LRU cache.
    if let Ok(list) = collections.list_collections() {
        metrics::gauge!("helix_collections_on_disk").set(list.len() as f64);
    }
    let loaded = collections.loaded_count();
    let max_open = collections.max_open_collections_limit();
    metrics::gauge!("helix_collections_loaded").set(loaded as f64);
    metrics::gauge!("helix_collections_max_open").set(max_open as f64);
    metrics::gauge!("helix_collection_cache_headroom").set(max_open.saturating_sub(loaded) as f64);
    if max_open > 0 {
        metrics::gauge!("helix_collection_cache_utilization").set(loaded as f64 / max_open as f64);
    }

    // LMDB VA (map_size) is exposed by storage_core; summing is cheap because
    // we only iterate the already-open envs in the LRU, not the full manifest.
    if let Some(va_bytes) = collections.total_lmdb_va_bytes() {
        metrics::gauge!("helix_lmdb_va_bytes").set(va_bytes as f64);
    }

    let memory = collections.memory_snapshot();
    if let Some(value) = memory.current_bytes {
        metrics::gauge!("helix_collection_open_memory_current_bytes").set(value as f64);
    }
    if let Some(value) = memory.inactive_file_bytes {
        metrics::gauge!("helix_collection_open_memory_inactive_file_bytes").set(value as f64);
    }
    if let Some(value) = memory.working_set_bytes {
        metrics::gauge!("helix_collection_open_memory_working_set_bytes").set(value as f64);
    }
    if let Some(value) = memory.ceiling_bytes {
        metrics::gauge!("helix_collection_open_memory_ceiling_bytes").set(value as f64);
    }
    if let Some(value) = memory.low_bytes {
        metrics::gauge!("helix_collection_open_memory_low_bytes").set(value as f64);
    }
    if let Some(value) = memory.high_bytes {
        metrics::gauge!("helix_collection_open_memory_high_bytes").set(value as f64);
    }
}
