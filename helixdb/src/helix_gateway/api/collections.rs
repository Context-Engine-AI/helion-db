use crate::helix_engine::{
    storage_core::replication::{submit_collection_optimizer, ReplicatedMutation},
    types::GraphError,
    vector_core::named_vectors::{SidecarCleanupStats, SidecarQuantizeStats},
};
use crate::helix_gateway::inflight;
use crate::helix_gateway::router::router::HandlerInput;
use crate::protocol::response::Response;

use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Deserialize)]
struct CollectionRequest {
    name: String,
}

#[derive(Deserialize)]
struct MaintenanceRequest {
    name: Option<String>,
    names: Option<Vec<String>>,
    #[serde(default)]
    all: bool,
    #[serde(default)]
    dry_run: bool,
    #[serde(default = "default_true")]
    cleanup_sidecars: bool,
    #[serde(default)]
    backfill_metadata_sidecars: bool,
    #[serde(default)]
    compact: bool,
    #[serde(default)]
    requantize_sidecars: bool,
    max_requantize_segments: Option<usize>,
    max_collections: Option<usize>,
}

#[derive(Serialize)]
struct MaintenanceCollectionResult {
    name: String,
    resolved_name: String,
    cleanup: Option<SidecarCleanupStats>,
    requantize: Option<SidecarQuantizeStats>,
    metadata_sidecar_backfilled: bool,
    metadata_sidecar_duration_ms: u64,
    compact_submitted_spaces: usize,
    cleanup_duration_ms: u64,
    requantize_duration_ms: u64,
    compact_submit_duration_ms: u64,
    duration_ms: u64,
    error: Option<String>,
}

#[derive(Serialize)]
struct MaintenanceResponse {
    dry_run: bool,
    cleanup_sidecars: bool,
    requantize_sidecars: bool,
    backfill_metadata_sidecars: bool,
    compact: bool,
    requested_collections: usize,
    processed_collections: usize,
    metadata_sidecars_backfilled: usize,
    totals: SidecarCleanupStats,
    requantize_totals: SidecarQuantizeStats,
    collections: Vec<MaintenanceCollectionResult>,
    duration_ms: u64,
}

fn default_true() -> bool {
    true
}

fn json_response(
    response: &mut Response,
    status: u16,
    body: &impl serde::Serialize,
) -> Result<(), GraphError> {
    response.status = status;
    response.body = sonic_rs::to_vec(body)?;
    response
        .headers
        .insert("Content-Type".to_string(), "application/json".to_string());
    Ok(())
}

fn json_error(response: &mut Response, status: u16, msg: &str) -> Result<(), GraphError> {
    json_response(response, status, &sonic_rs::json!({"error": msg}))
}

pub fn handle_create(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    let req: CollectionRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_error(response, 400, &format!("Invalid JSON: {}", e)),
    };

    match input
        .replication
        .apply(ReplicatedMutation::CreateCollection {
            name: req.name.clone(),
            vectors: std::collections::HashMap::new(),
            sparse_vectors: std::collections::HashMap::new(),
            hnsw_overrides: None,
        }) {
        Ok(_) => json_response(response, 201, &sonic_rs::json!({"created": req.name})),
        Err(e) => json_error(response, 409, &e.to_string()),
    }
}

pub fn handle_drop(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    let req: CollectionRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_error(response, 400, &format!("Invalid JSON: {}", e)),
    };

    match input
        .replication
        .apply(ReplicatedMutation::DeleteCollection {
            name: req.name.clone(),
        }) {
        Ok(_) => {
            // Purge per-collection gateway state so the limiter map doesn't
            // grow forever under tenant churn. In-flight requests holding
            // guards remain correct because the `Arc<AtomicUsize>` survives
            // until the last guard drops.
            inflight::forget(&req.name);
            json_response(response, 200, &sonic_rs::json!({"dropped": req.name}))
        }
        Err(e) => json_error(response, 404, &e.to_string()),
    }
}

pub fn handle_list(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    let names = input.collections.list_collections()?;
    json_response(response, 200, &sonic_rs::json!({"collections": names}))
}

pub fn handle_stats(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    let req: CollectionRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_error(response, 400, &format!("Invalid JSON: {}", e)),
    };

    match input.collections.collection_stats(&req.name) {
        Ok(stats) => json_response(response, 200, &stats),
        Err(e) if e.to_string().contains("metadata sidecar") => {
            json_error(response, 503, &e.to_string())
        }
        Err(e) => json_error(response, 404, &e.to_string()),
    }
}

pub fn handle_storage_bytes(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let req: CollectionRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_error(response, 400, &format!("Invalid JSON: {}", e)),
    };

    match input.collections.collection_storage_bytes(&req.name) {
        Ok(storage) => json_response(response, 200, &storage),
        Err(e) if e.to_string().contains("does not exist") => {
            json_error(response, 404, &e.to_string())
        }
        Err(e) => json_error(response, 503, &e.to_string()),
    }
}

/// POST /v1/collections/recount — fleet-repair tool for the LSM
/// counter-corruption class: recompute node/edge/vector counters from scan
/// truth and unconditionally overwrite the merge-key, regardless of whether it
/// was absent, present-but-stale, or already correct. Only runs on the LSM
/// writer; `recount_lsm_counters` rejects reader replicas and non-LSM backends.
pub fn handle_recount(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    let req: CollectionRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_error(response, 400, &format!("Invalid JSON: {}", e)),
    };

    match input.collections.recount_collection(&req.name) {
        Ok(result) => json_response(response, 200, &result),
        Err(e) if e.to_string().contains("not found") => json_error(response, 404, &e.to_string()),
        Err(e) if e.to_string().contains("already in progress") => {
            json_error(response, 409, &e.to_string())
        }
        Err(e) => json_error(response, 400, &e.to_string()),
    }
}

#[derive(Deserialize)]
struct GcPayloadIndexRequest {
    name: String,
    #[serde(default)]
    field: Option<String>,
}

/// POST /v1/collections/gc_payload_index — fleet-repair tool for the ghost
/// payload-index class: `drop_node`/`drop_node_be`'s de-index step used a
/// flat property lookup instead of the nested-aware `payload_value_for_key`,
/// so a node indexed on a dotted field name (e.g. `metadata.repo`) never
/// de-indexed on delete and left a dup entry pointing at a gone node id
/// forever. This scans the index (or just `field`, if given) and removes
/// entries whose node no longer exists. Run this AFTER the resolver fix
/// ships, or newly-deleted nodes will re-poison the index just cleaned. Only
/// runs on the LSM writer, same as recount; a concurrent run on the same
/// collection is rejected with 409.
pub fn handle_gc_payload_index(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let req: GcPayloadIndexRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_error(response, 400, &format!("Invalid JSON: {}", e)),
    };

    match input
        .collections
        .gc_payload_index_for_collection(&req.name, req.field.as_deref())
    {
        Ok(result) => json_response(response, 200, &result),
        Err(e) if e.to_string().contains("not found") => json_error(response, 404, &e.to_string()),
        Err(e) if e.to_string().contains("already in progress") => {
            json_error(response, 409, &e.to_string())
        }
        Err(e) => json_error(response, 400, &e.to_string()),
    }
}

pub fn handle_maintenance(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    let started = Instant::now();
    let req: MaintenanceRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_error(response, 400, &format!("Invalid JSON: {}", e)),
    };

    if req.dry_run && (req.compact || req.requantize_sidecars) {
        return json_error(
            response,
            400,
            "compact/requantize_sidecars cannot run in dry_run mode",
        );
    }

    let mut names = if req.all {
        input.collections.list_collections()?
    } else {
        let mut names = Vec::new();
        if let Some(name) = req.name {
            names.push(name);
        }
        if let Some(mut requested) = req.names {
            names.append(&mut requested);
        }
        if names.is_empty() {
            return json_error(response, 400, "set all=true or provide name/names");
        }
        names
    };
    names.sort();
    names.dedup();
    if let Some(max) = req.max_collections {
        names.truncate(max);
    }

    let requested_collections = names.len();
    let mut totals = SidecarCleanupStats {
        dry_run: req.dry_run,
        ..SidecarCleanupStats::default()
    };
    let mut requantize_totals = SidecarQuantizeStats::default();
    let mut results = Vec::with_capacity(names.len());

    for name in names {
        let collection_started = Instant::now();
        let resolved_name = input.collections.resolve_alias(&name);
        let mut cleanup = None;
        let mut requantize = None;
        let mut cleanup_duration_ms = 0;
        let mut requantize_duration_ms = 0;
        let mut metadata_sidecar_backfilled = false;
        let mut metadata_sidecar_duration_ms = 0;
        let mut compact_submitted_spaces = 0;
        let mut compact_submit_duration_ms = 0;
        let mut error = None;

        if req.cleanup_sidecars {
            let cleanup_started = Instant::now();
            match input.collections.get_collection(&resolved_name) {
                Ok(storage) => {
                    let stats = storage
                        .named_vectors
                        .cleanup_inactive_sidecar_files(storage.path(), req.dry_run);
                    cleanup_duration_ms = stats.duration_ms;
                    totals.merge(stats.clone());
                    cleanup = Some(stats);
                }
                Err(e) => {
                    error = Some(e.to_string());
                    metrics::counter!(
                        "helix_maintenance_collection_errors_total",
                        "phase" => "cleanup"
                    )
                    .increment(1);
                }
            }
            metrics::histogram!(
                "helix_maintenance_collection_phase_ms",
                "phase" => "cleanup",
                "status" => if error.is_some() { "error" } else { "ok" }
            )
            .record(cleanup_started.elapsed().as_secs_f64() * 1000.0);
        }

        if req.requantize_sidecars && error.is_none() {
            let requantize_started = Instant::now();
            match input.collections.get_collection(&resolved_name) {
                Ok(storage) => {
                    let stats = storage
                        .named_vectors
                        .quantize_indexed_mmap_sidecars(req.max_requantize_segments);
                    requantize_duration_ms = stats.duration_ms;
                    requantize_totals.merge(stats.clone());
                    if stats.error_count > 0 {
                        metrics::counter!(
                            "helix_maintenance_collection_errors_total",
                            "phase" => "requantize"
                        )
                        .increment(stats.error_count as u64);
                    }
                    requantize = Some(stats);
                }
                Err(e) => {
                    error = Some(e.to_string());
                    metrics::counter!(
                        "helix_maintenance_collection_errors_total",
                        "phase" => "requantize"
                    )
                    .increment(1);
                }
            }
            metrics::histogram!(
                "helix_maintenance_collection_phase_ms",
                "phase" => "requantize",
                "status" => if error.is_some() { "error" } else { "ok" }
            )
            .record(requantize_started.elapsed().as_secs_f64() * 1000.0);
        }

        if req.backfill_metadata_sidecars && error.is_none() {
            let sidecar_started = Instant::now();
            let sidecar_present = match input
                .collections
                .collection_metadata_sidecar(&resolved_name)
            {
                Ok(Some(_)) => true,
                Ok(None) => false,
                Err(e) => {
                    tracing::warn!(
                        collection = %resolved_name,
                        error = ?e,
                        "Metadata sidecar is unreadable; maintenance will rewrite it"
                    );
                    false
                }
            };

            if !sidecar_present && !req.dry_run {
                let was_loaded = input
                    .collections
                    .get_loaded_collection(&resolved_name)?
                    .is_some();
                match input.collections.get_collection(&resolved_name) {
                    Ok(storage) => {
                        if let Err(e) = storage.refresh_metadata_snapshot() {
                            error = Some(e.to_string());
                            metrics::counter!(
                                "helix_maintenance_collection_errors_total",
                                "phase" => "metadata_sidecar"
                            )
                            .increment(1);
                        } else {
                            metadata_sidecar_backfilled = true;
                            metrics::counter!("helix_metadata_sidecars_backfilled_total")
                                .increment(1);
                        }
                        drop(storage);
                        if !was_loaded {
                            input.collections.evict_collection(&resolved_name);
                        }
                    }
                    Err(e) => {
                        error = Some(e.to_string());
                        metrics::counter!(
                            "helix_maintenance_collection_errors_total",
                            "phase" => "metadata_sidecar"
                        )
                        .increment(1);
                    }
                }
            }
            metadata_sidecar_duration_ms = sidecar_started.elapsed().as_millis() as u64;
            metrics::histogram!(
                "helix_maintenance_collection_phase_ms",
                "phase" => "metadata_sidecar",
                "status" => if error.is_some() { "error" } else { "ok" }
            )
            .record(sidecar_started.elapsed().as_secs_f64() * 1000.0);
        }

        if req.compact && error.is_none() {
            let compact_started = Instant::now();
            match submit_collection_optimizer(
                &input.collections,
                input.collections.config(),
                &resolved_name,
            ) {
                Ok(count) => {
                    compact_submitted_spaces = count;
                    metrics::counter!("helix_maintenance_compact_submitted_spaces_total")
                        .increment(count as u64);
                }
                Err(e) => {
                    error = Some(e.to_string());
                    metrics::counter!(
                        "helix_maintenance_collection_errors_total",
                        "phase" => "compact"
                    )
                    .increment(1);
                }
            }
            compact_submit_duration_ms = compact_started.elapsed().as_millis() as u64;
            metrics::histogram!(
                "helix_maintenance_collection_phase_ms",
                "phase" => "compact_submit",
                "status" => if error.is_some() { "error" } else { "ok" }
            )
            .record(compact_started.elapsed().as_secs_f64() * 1000.0);
        }

        let duration_ms = collection_started.elapsed().as_millis() as u64;
        metrics::histogram!(
            "helix_maintenance_collection_duration_ms",
            "status" => if error.is_some() { "error" } else { "ok" }
        )
        .record(duration_ms as f64);

        results.push(MaintenanceCollectionResult {
            name,
            resolved_name,
            cleanup,
            requantize,
            metadata_sidecar_backfilled,
            metadata_sidecar_duration_ms,
            compact_submitted_spaces,
            cleanup_duration_ms,
            requantize_duration_ms,
            compact_submit_duration_ms,
            duration_ms,
            error,
        });
    }

    let duration_ms = started.elapsed().as_millis() as u64;
    metrics::histogram!("helix_maintenance_request_duration_ms")
        .record(started.elapsed().as_secs_f64() * 1000.0);
    json_response(
        response,
        200,
        &MaintenanceResponse {
            dry_run: req.dry_run,
            cleanup_sidecars: req.cleanup_sidecars,
            requantize_sidecars: req.requantize_sidecars,
            backfill_metadata_sidecars: req.backfill_metadata_sidecars,
            compact: req.compact,
            requested_collections,
            processed_collections: results.len(),
            metadata_sidecars_backfilled: results
                .iter()
                .filter(|result| result.metadata_sidecar_backfilled)
                .count(),
            totals,
            requantize_totals,
            collections: results,
            duration_ms,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helix_engine::graph_core::config::Config;
    use crate::helix_engine::graph_core::graph_core::{HelixGraphEngine, HelixGraphEngineOpts};
    use crate::helix_engine::storage_core::backend::{BackendKind, StorageBackend};
    use crate::helix_engine::storage_core::metadata::{
        encode_lsm_counter_value, lsm_counter_key, MetadataCounter,
    };
    use crate::helix_engine::storage_core::{
        backend::Namespace, collection_manager::CollectionManager, replication::ReplicationManager,
    };
    use crate::protocol::request::Request;
    use crate::protocol::value::Value;
    use sonic_rs::JsonValueTrait;
    use std::collections::HashMap;
    use tempfile::TempDir;

    struct TestContext {
        _tmp: TempDir,
        graph: std::sync::Arc<HelixGraphEngine>,
        collections: std::sync::Arc<CollectionManager>,
        replication: std::sync::Arc<ReplicationManager>,
    }

    fn setup() -> TestContext {
        setup_with_config(Config::default())
    }

    fn setup_with_config(config: Config) -> TestContext {
        let tmp = TempDir::new().unwrap();
        let graph_path = tmp.path().join("graph");
        let collections_path = tmp.path().join("data");
        let graph = std::sync::Arc::new(
            HelixGraphEngine::new(HelixGraphEngineOpts {
                path: graph_path.display().to_string(),
                config: config.clone(),
            })
            .unwrap(),
        );
        let collections =
            std::sync::Arc::new(CollectionManager::new(collections_path, config.clone()).unwrap());
        let replication = std::sync::Arc::new(
            ReplicationManager::new(std::sync::Arc::clone(&collections), config).unwrap(),
        );
        TestContext {
            _tmp: tmp,
            graph,
            collections,
            replication,
        }
    }

    fn make_input(ctx: &TestContext, path: &str, body: Vec<u8>) -> HandlerInput {
        HandlerInput {
            request: Request {
                method: "POST".into(),
                headers: HashMap::new(),
                path: path.into(),
                body,
            },
            graph: std::sync::Arc::clone(&ctx.graph),
            collections: std::sync::Arc::clone(&ctx.collections),
            replication: std::sync::Arc::clone(&ctx.replication),
            path_params: HashMap::new(),
        }
    }

    #[test]
    fn handle_recount_unknown_collection_returns_404() {
        let ctx = setup();
        let body = sonic_rs::to_vec(&sonic_rs::json!({"name": "nope"})).unwrap();
        let input = make_input(&ctx, "/v1/collections/recount", body);
        let mut response = Response::new();

        handle_recount(&input, &mut response).unwrap();
        assert_eq!(response.status, 404);
    }

    #[test]
    fn handle_recount_repairs_corrupted_counter() {
        let ctx = setup_with_config(Config::default().with_lsm_in_memory());

        ctx.collections.create_collection("repo").unwrap();
        let storage = ctx.collections.get_collection("repo").unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);
        for _ in 0..5 {
            let mut w = storage.backend.begin_write().unwrap();
            storage
                .create_node_be(&mut w, "L", Vec::<(String, Value)>::new(), None, None)
                .unwrap();
            storage.backend.commit(w).unwrap();
        }

        // Corrupt: overwrite the true counter (5) with a wrong low value —
        // the "present but wrong" case recount must repair regardless of
        // whether the key was ever absent.
        let mut w = storage.backend.begin_write().unwrap();
        storage
            .backend
            .put(
                &mut w,
                Namespace::Metadata,
                lsm_counter_key(MetadataCounter::Nodes),
                &encode_lsm_counter_value(1),
            )
            .unwrap();
        storage.backend.commit(w).unwrap();

        let body = sonic_rs::to_vec(&sonic_rs::json!({"name": "repo"})).unwrap();
        let input = make_input(&ctx, "/v1/collections/recount", body);
        let mut response = Response::new();

        handle_recount(&input, &mut response).unwrap();
        assert_eq!(response.status, 200);

        let parsed: sonic_rs::Value = sonic_rs::from_slice(&response.body).unwrap();
        assert_eq!(parsed["counters"]["nodes"]["before"].as_u64(), Some(1));
        assert_eq!(parsed["counters"]["nodes"]["after"].as_u64(), Some(5));

        let stats = ctx.collections.collection_stats("repo").unwrap();
        assert_eq!(
            stats.node_count, 5,
            "unfiltered count must reflect the repaired counter"
        );
    }

    /// MEDIUM finding from the poll-tiering review's final round: the scan
    /// (read) + overwrite (write) pair in `recount_lsm_counters` is not
    /// atomic against a concurrent recount of the same collection, so two
    /// racing calls could each read a mid-flight scan value and stomp each
    /// other's result. A per-collection CAS guard now rejects a concurrent
    /// call with 409 instead of racing.
    #[test]
    fn handle_recount_returns_409_when_already_in_progress() {
        let ctx = setup_with_config(Config::default().with_lsm_in_memory());

        ctx.collections.create_collection("repo").unwrap();
        let storage = ctx.collections.get_collection("repo").unwrap();

        // Simulate an in-flight recount without releasing it — the same
        // guard `recount_lsm_counters` acquires at entry.
        assert!(
            storage.try_begin_recount(),
            "guard must be free before any recount has started"
        );

        let body = sonic_rs::to_vec(&sonic_rs::json!({"name": "repo"})).unwrap();
        let input = make_input(&ctx, "/v1/collections/recount", body);
        let mut response = Response::new();
        handle_recount(&input, &mut response).unwrap();
        assert_eq!(
            response.status, 409,
            "a recount already in flight must be rejected, not raced"
        );

        // Release the guard: a subsequent recount must succeed normally.
        storage.end_recount();
        let mut response = Response::new();
        handle_recount(&input, &mut response).unwrap();
        assert_eq!(
            response.status, 200,
            "recount must succeed once the prior guard is released"
        );
    }

    #[test]
    fn handle_gc_payload_index_unknown_collection_returns_404() {
        let ctx = setup();
        let body = sonic_rs::to_vec(&sonic_rs::json!({"name": "nope"})).unwrap();
        let input = make_input(&ctx, "/v1/collections/gc_payload_index", body);
        let mut response = Response::new();

        handle_gc_payload_index(&input, &mut response).unwrap();
        assert_eq!(response.status, 404);
    }

    /// Reproduces the ghost payload-index class end to end: index a dotted
    /// field (`metadata.repo`), insert 3 nodes, delete one, then directly
    /// re-add a dup entry for the now-gone node id — exactly what the OLD
    /// (pre-fix) `drop_node_be` left behind on every delete of a
    /// dotted-field-indexed node, since its flat property lookup silently
    /// no-op'd. The GC endpoint must find and remove exactly that one ghost,
    /// and a subsequent filtered lookup must reflect the corrected count.
    #[test]
    fn handle_gc_payload_index_repairs_ghost_entry() {
        use crate::helix_engine::storage_core::metadata::PayloadIndexSchema;

        let ctx = setup_with_config(Config::default().with_lsm_in_memory());

        ctx.collections.create_collection("repo").unwrap();
        let storage = ctx.collections.get_collection("repo").unwrap();
        storage
            .create_payload_index("metadata.repo", PayloadIndexSchema::Keyword)
            .unwrap();

        // `create_node_be` never touches `payload_indices` (only
        // `upsert_node_be`/`update_node_be`/`drop_node_be` do), so use
        // `upsert_node_be` here to actually get the nodes indexed.
        use crate::helix_engine::storage_core::upsert::NodeUpsert;
        let ids: Vec<u128> = (100u128..103u128).collect();
        for &id in &ids {
            let mut w = storage.backend.begin_write().unwrap();
            let upsert = NodeUpsert {
                id,
                label: "L".to_string(),
                properties: HashMap::from([(
                    "metadata".to_string(),
                    Value::Object(HashMap::from([(
                        "repo".to_string(),
                        Value::String("acme".to_string()),
                    )])),
                )]),
            };
            storage.upsert_node_be(&mut w, &upsert).unwrap();
            storage.backend.commit(w).unwrap();
        }

        // Delete the middle node via the (now-fixed) real path — this
        // correctly de-indexes it.
        let mut w = storage.backend.begin_write().unwrap();
        storage.drop_node_be(&mut w, &ids[1]).unwrap();
        storage.backend.commit(w).unwrap();

        // Simulate the pre-fix poisoned state: directly re-add a dup entry
        // for that now-gone node id, bypassing the (working) deindex step.
        let mut w = storage.backend.begin_write().unwrap();
        storage
            .index_node_payload_field_be(
                &mut w,
                "metadata.repo",
                &PayloadIndexSchema::Keyword,
                ids[1],
                Some(&Value::String("acme".to_string())),
            )
            .unwrap();
        storage.backend.commit(w).unwrap();

        let body = sonic_rs::to_vec(&sonic_rs::json!({"name": "repo"})).unwrap();
        let input = make_input(&ctx, "/v1/collections/gc_payload_index", body);
        let mut response = Response::new();

        handle_gc_payload_index(&input, &mut response).unwrap();
        assert_eq!(response.status, 200);

        let parsed: sonic_rs::Value = sonic_rs::from_slice(&response.body).unwrap();
        assert_eq!(
            parsed["fields"]["metadata.repo"]["scanned"].as_u64(),
            Some(3)
        );
        assert_eq!(
            parsed["fields"]["metadata.repo"]["removed"].as_u64(),
            Some(1)
        );

        let r = storage.backend.begin_read().unwrap();
        let hits = storage
            .get_nodes_by_payload_value_be(&r, "metadata.repo", &Value::String("acme".to_string()))
            .unwrap();
        assert_eq!(
            hits.len(),
            2,
            "filtered count must reflect the 2 surviving nodes, not the ghost"
        );
    }

    /// Same CAS-guard contract as `handle_recount_returns_409_when_already_in_progress`.
    #[test]
    fn handle_gc_payload_index_returns_409_when_already_in_progress() {
        let ctx = setup_with_config(Config::default().with_lsm_in_memory());

        ctx.collections.create_collection("repo").unwrap();
        let storage = ctx.collections.get_collection("repo").unwrap();

        assert!(
            storage.try_begin_payload_index_gc(),
            "guard must be free before any GC run has started"
        );

        let body = sonic_rs::to_vec(&sonic_rs::json!({"name": "repo"})).unwrap();
        let input = make_input(&ctx, "/v1/collections/gc_payload_index", body);
        let mut response = Response::new();
        handle_gc_payload_index(&input, &mut response).unwrap();
        assert_eq!(
            response.status, 409,
            "a GC run already in flight must be rejected, not raced"
        );

        storage.end_payload_index_gc();
        let mut response = Response::new();
        handle_gc_payload_index(&input, &mut response).unwrap();
        assert_eq!(
            response.status, 200,
            "GC must succeed once the prior guard is released"
        );
    }
}
