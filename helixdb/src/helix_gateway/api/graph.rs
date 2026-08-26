use std::collections::{HashMap, HashSet};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::helix_engine::graph_core::traversals::algorithms;
#[cfg(test)]
use crate::helix_engine::graph_core::traversals::bfs::bfs_reverse;
use crate::helix_engine::graph_core::traversals::bfs::{
    bfs_forward_be, bfs_impact_be, bfs_reverse_be, BfsResult,
};
use crate::helix_engine::graph_core::traversals::cycles::detect_cycles_be;
use crate::helix_engine::storage_core::backend::{
    BackendKind, KeyRange, Namespace, StorageBackend,
};
use crate::helix_engine::storage_core::backend_any::AnyRead;
use crate::helix_engine::storage_core::metadata::PayloadIndexSchema;
use crate::helix_engine::storage_core::storage_core::{ChunkEdgeCandidate, HelixGraphStorage};
use crate::helix_engine::storage_core::storage_methods::StorageMethods;
use crate::helix_engine::storage_core::upsert::EdgeUpsert;
use crate::helix_engine::types::GraphError;
use crate::helix_gateway::api::qdrant::string_id_to_u128;
use crate::helix_gateway::router::router::HandlerInput;
use crate::protocol::deterministic_id;
use crate::protocol::items::{Edge, SerializedEdge};
use crate::protocol::label_hash::hash_label;
use crate::protocol::response::Response;
use crate::protocol::value::Value;

const DEFAULT_GRAPH_DEPTH_MAX: usize = 10;
const DEFAULT_GRAPH_LIMIT_MAX: usize = 500;
const DEFAULT_GRAPH_ALGORITHM_ITERATIONS_MAX: usize = 50;
const DEFAULT_GRAPH_SHORTEST_PATH_VISITED_MAX: usize = 100_000;
const DEFAULT_GRAPH_SYMBOL_RESOLVE_MAX: usize = 32;
const DEFAULT_GRAPH_DELETE_TXN_CHUNK_SIZE: usize = 5_000;

#[derive(Deserialize)]
struct GraphQuery {
    collection: String,
    symbol: String,
    /// Path filter (optional — if omitted, matches any path)
    #[serde(default)]
    path: Option<String>,
    /// Repo filter (optional — if omitted, matches any repo)
    #[serde(default)]
    repo: Option<String>,
    #[serde(default = "default_depth")]
    depth: usize,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_depth() -> usize {
    1
}
fn default_limit() -> usize {
    100
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn graph_depth_max() -> usize {
    env_usize("HELIX_GRAPH_DEPTH_MAX", DEFAULT_GRAPH_DEPTH_MAX)
}

fn graph_limit_max() -> usize {
    env_usize("HELIX_GRAPH_LIMIT_MAX", DEFAULT_GRAPH_LIMIT_MAX)
}

fn graph_algorithm_iterations_max() -> usize {
    env_usize(
        "HELIX_GRAPH_ALGORITHM_ITERATIONS_MAX",
        DEFAULT_GRAPH_ALGORITHM_ITERATIONS_MAX,
    )
}

fn graph_shortest_path_visited_max() -> usize {
    env_usize(
        "HELIX_GRAPH_SHORTEST_PATH_VISITED_MAX",
        DEFAULT_GRAPH_SHORTEST_PATH_VISITED_MAX,
    )
}

fn graph_symbol_resolve_max() -> usize {
    env_usize(
        "HELIX_GRAPH_SYMBOL_RESOLVE_MAX",
        DEFAULT_GRAPH_SYMBOL_RESOLVE_MAX,
    )
}

fn graph_delete_txn_chunk_size() -> usize {
    env_usize(
        "HELIX_GRAPH_DELETE_TXN_CHUNK_SIZE",
        DEFAULT_GRAPH_DELETE_TXN_CHUNK_SIZE,
    )
    .clamp(1, 50_000)
}

fn observe_graph_delete_phase(
    route: &'static str,
    phase: &'static str,
    elapsed: std::time::Duration,
) {
    if !crate::telemetry::hot_path_metrics_enabled() {
        return;
    }
    metrics::histogram!(
        "helix_graph_delete_phase_ms",
        "route" => route,
        "phase" => phase
    )
    .record(elapsed.as_secs_f64() * 1000.0);
}

fn observe_graph_delete_items(route: &'static str, kind: &'static str, count: usize) {
    if !crate::telemetry::hot_path_metrics_enabled() {
        return;
    }
    metrics::histogram!(
        "helix_graph_delete_items",
        "route" => route,
        "kind" => kind
    )
    .record(count as f64);
}

fn graph_edge_repo_matches(edge: &Edge, repo: Option<&str>) -> bool {
    match (repo, edge.properties.get("repo")) {
        (Some(expected), Some(Value::String(actual))) => actual == expected,
        (Some(_), _) => false,
        (None, _) => true,
    }
}

fn graph_edge_matches_path(edge: &Edge, path: &str) -> bool {
    matches!(edge.properties.get("from_path"), Some(Value::String(p)) if p == path)
        || matches!(edge.properties.get("to_path"), Some(Value::String(p)) if p == path)
}

fn graph_edge_matches_any_path(edge: &Edge, paths: &HashSet<String>) -> bool {
    matches!(edge.properties.get("from_path"), Some(Value::String(p)) if paths.contains(p))
        || matches!(edge.properties.get("to_path"), Some(Value::String(p)) if paths.contains(p))
}

fn clamp_graph_depth(depth: usize) -> usize {
    depth.clamp(1, graph_depth_max())
}

fn clamp_graph_limit(limit: usize) -> usize {
    limit.clamp(1, graph_limit_max())
}

fn clamp_graph_iterations(iterations: usize) -> usize {
    iterations.clamp(1, graph_algorithm_iterations_max())
}

fn default_shortest_path_depth() -> usize {
    graph_depth_max()
}

fn delete_edges_in_write_chunks(
    storage: &HelixGraphStorage,
    edge_ids: &[u128],
) -> Result<usize, GraphError> {
    let chunk_size = graph_delete_txn_chunk_size();
    let mut edges_deleted = 0usize;
    for chunk in edge_ids.chunks(chunk_size) {
        if storage.backend.kind() == BackendKind::Lsm {
            edges_deleted += storage.with_write_backend(|w| {
                let mut chunk_deleted = 0usize;
                for eid in chunk {
                    match storage.drop_edge_be(w, eid) {
                        Ok(()) => chunk_deleted += 1,
                        Err(_) => {}
                    }
                }
                Ok(chunk_deleted)
            })?;
        } else {
            edges_deleted += storage.with_write_txn(|wtxn| {
                let mut chunk_deleted = 0usize;
                for eid in chunk {
                    // Ignore EdgeNotFound — edge may have been deleted by cascade.
                    // MapFull, however, must bubble out so the wrapper can retry.
                    match storage.drop_edge(wtxn, eid) {
                        Ok(()) => chunk_deleted += 1,
                        Err(GraphError::MapFull) => return Err(GraphError::MapFull),
                        Err(_) => {}
                    }
                }
                Ok(chunk_deleted)
            })?;
        }
    }
    Ok(edges_deleted)
}

fn delete_orphaned_nodes_in_write_chunks(
    storage: &HelixGraphStorage,
    affected_nodes: &HashSet<u128>,
) -> Result<usize, GraphError> {
    if affected_nodes.is_empty() {
        return Ok(0);
    }

    let mut node_ids: Vec<u128> = affected_nodes.iter().copied().collect();
    node_ids.sort_unstable();

    let chunk_size = graph_delete_txn_chunk_size();
    let mut nodes_deleted = 0usize;
    for chunk in node_ids.chunks(chunk_size) {
        if storage.backend.kind() == BackendKind::Lsm {
            let r = storage
                .backend
                .begin_read()
                .map_err(|e| GraphError::New(e.to_string()))?;
            let mut removable = Vec::new();
            for node_id in chunk {
                let has_out = storage.node_has_adjacency_be(&r, *node_id, true)?;
                let has_in = storage.node_has_adjacency_be(&r, *node_id, false)?;
                if !has_out && !has_in {
                    removable.push(*node_id);
                }
            }
            nodes_deleted += storage.with_write_backend(|w| {
                let mut chunk_deleted = 0usize;
                for node_id in &removable {
                    match storage.drop_node_be(w, node_id) {
                        Ok(()) => chunk_deleted += 1,
                        Err(_) => {}
                    }
                }
                Ok(chunk_deleted)
            })?;
        } else {
            nodes_deleted += storage.with_write_txn(|wtxn| {
                let mut chunk_deleted = 0usize;
                for node_id in chunk {
                    // Backend-routed prefix existence check: on LSM this reads SlateDB
                    // adjacency (where edges now live), so a node with edges in S3 is
                    // not wrongly orphan-deleted.
                    let has_out = storage.node_has_adjacency(wtxn, *node_id, true)?;
                    let has_in = storage.node_has_adjacency(wtxn, *node_id, false)?;
                    if !has_out && !has_in {
                        match storage.drop_node(wtxn, node_id) {
                            Ok(()) => chunk_deleted += 1,
                            Err(GraphError::MapFull) => return Err(GraphError::MapFull),
                            Err(_) => {}
                        }
                    }
                }
                Ok(chunk_deleted)
            })?;
        }
    }

    Ok(nodes_deleted)
}

#[derive(Serialize)]
struct NodeResult {
    id: String,
    label: String,
    properties: std::collections::HashMap<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    depth: Option<usize>,
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

fn parse_query(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<Option<GraphQuery>, GraphError> {
    match sonic_rs::from_slice(&input.request.body) {
        Ok(q) => Ok(Some(q)),
        Err(e) => {
            json_error(response, 400, &format!("Invalid JSON: {}", e))?;
            Ok(None)
        }
    }
}

/// Resolve symbol to node IDs.
///
/// If `path` is provided, constructs the deterministic ID directly.
/// If `path` is absent, uses the `name` multi-index for O(1) lookup
/// by symbol name and returns all bounded matches. CE can have multiple
/// Symbol nodes with the same name: exact definitions carry a real path,
/// while call-edge callees may carry an unresolved/empty path. Pathless
/// graph queries should behave like the legacy flat edge index and match
/// all of those nodes, not whichever node the name index returns first.
#[cfg(test)]
fn resolve_symbol_ids(
    q: &GraphQuery,
    storage: &HelixGraphStorage,
    txn: &heed3::RoTxn,
) -> Vec<u128> {
    let r = storage.read_view(txn);
    resolve_symbol_ids_be(q, storage, &r)
}

fn resolve_symbol_ids_be(
    q: &GraphQuery,
    storage: &HelixGraphStorage,
    r: &AnyRead<'_>,
) -> Vec<u128> {
    if q.path.is_some() {
        let path = q.path.as_deref().unwrap_or("");
        return vec![deterministic_id::node_id(
            &q.collection,
            "Symbol",
            &q.symbol,
            path,
        )];
    }
    // No path — use `name` multi-index for efficient lookup. Return all
    // bounded matches so same-name nodes across unresolved/resolved paths are
    // represented in traversal.
    if let Ok(ids) =
        storage.get_nodes_by_multi_index_be(r, "name", &Value::String(q.symbol.clone()))
    {
        if !ids.is_empty() {
            return ids.into_iter().take(graph_symbol_resolve_max()).collect();
        }
    }
    // Fallback: deterministic ID with empty path
    vec![deterministic_id::node_id(
        &q.collection,
        "Symbol",
        &q.symbol,
        "",
    )]
}

fn resolve_symbol_id_be(q: &GraphQuery, storage: &HelixGraphStorage, r: &AnyRead<'_>) -> u128 {
    resolve_symbol_ids_be(q, storage, r)
        .into_iter()
        .next()
        .unwrap_or_else(|| deterministic_id::node_id(&q.collection, "Symbol", &q.symbol, ""))
}

fn bfs_to_results(bfs: Vec<BfsResult>) -> Vec<NodeResult> {
    bfs.into_iter()
        .map(|r| {
            let mut props = r.node.properties;
            // Merge edge properties (repo, git_branches, etc.) into the node result
            // so the CE backend can extract them for filtering.
            for (k, v) in r.edge_properties {
                props.entry(k).or_insert(v);
            }
            NodeResult {
                id: format!("{:032x}", r.node.id),
                label: r.node.label,
                properties: props,
                depth: Some(r.depth),
            }
        })
        .collect()
}

fn properties_match_repo(
    properties: &std::collections::HashMap<String, Value>,
    repo: Option<&str>,
) -> bool {
    match (repo, properties.get("repo")) {
        (Some(expected), Some(Value::String(actual))) => actual == expected,
        (Some(_), _) => false,
        (None, _) => true,
    }
}

// --- Single-hop queries ---

pub fn handle_callers(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    let start = Instant::now();
    let q = match parse_query(input, response)? {
        Some(q) => q,
        None => return Ok(()),
    };
    let collection = q.collection.clone();
    let storage = match input.collections.get_collection(&q.collection) {
        Ok(s) => s,
        Err(e) => return json_error(response, 404, &e.to_string()),
    };
    let repo = q.repo.as_deref();
    let limit = clamp_graph_limit(q.limit);
    storage.with_read_backend(|r| {
        let starts = resolve_symbol_ids_be(&q, &storage, r);
        let results = bfs_reverse_be(&storage, r, &starts, &["CALLS"], 1, limit, repo)?;
        json_response(
            response,
            200,
            &sonic_rs::json!({"results": bfs_to_results(results)}),
        )
    })?;
    tracing::debug!(
        collection = %collection,
        elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
        "handle_callers complete"
    );
    Ok(())
}

pub fn handle_callees(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    let start = Instant::now();
    let q = match parse_query(input, response)? {
        Some(q) => q,
        None => return Ok(()),
    };
    let collection = q.collection.clone();
    let storage = match input.collections.get_collection(&q.collection) {
        Ok(s) => s,
        Err(e) => return json_error(response, 404, &e.to_string()),
    };
    let repo = q.repo.as_deref();
    let limit = clamp_graph_limit(q.limit);
    storage.with_read_backend(|r| {
        let starts = resolve_symbol_ids_be(&q, &storage, r);
        let results = bfs_forward_be(&storage, r, &starts, &["CALLS"], 1, limit, repo)?;
        json_response(
            response,
            200,
            &sonic_rs::json!({"results": bfs_to_results(results)}),
        )
    })?;
    tracing::debug!(
        collection = %collection,
        elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
        "handle_callees complete"
    );
    Ok(())
}

pub fn handle_importers(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    let start = Instant::now();
    let q = match parse_query(input, response)? {
        Some(q) => q,
        None => return Ok(()),
    };
    let storage = match input.collections.get_collection(&q.collection) {
        Ok(s) => s,
        Err(e) => return json_error(response, 404, &e.to_string()),
    };
    let repo = q.repo.as_deref();
    let limit = clamp_graph_limit(q.limit);
    storage.with_read_backend(|r| {
        let starts = resolve_symbol_ids_be(&q, &storage, r);
        let results = bfs_reverse_be(&storage, r, &starts, &["IMPORTS"], 1, limit, repo)?;
        json_response(
            response,
            200,
            &sonic_rs::json!({"results": bfs_to_results(results)}),
        )
    })?;
    tracing::debug!(
        collection = %q.collection,
        elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
        "handle_importers complete"
    );
    Ok(())
}

pub fn handle_definition(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    let start = Instant::now();
    let q = match parse_query(input, response)? {
        Some(q) => q,
        None => return Ok(()),
    };
    let storage = match input.collections.get_collection(&q.collection) {
        Ok(s) => s,
        Err(e) => return json_error(response, 404, &e.to_string()),
    };
    let result = storage.with_read_backend(|r| {
        let id = resolve_symbol_id_be(&q, &storage, r);
        match storage.get_node_be(r, &id) {
            Ok(node) => {
                let result = NodeResult {
                    id: format!("{:032x}", node.id),
                    label: node.label,
                    properties: node.properties,
                    depth: None,
                };
                json_response(response, 200, &sonic_rs::json!({"result": result}))?;
                tracing::debug!(
                    collection = %q.collection,
                    elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
                    "handle_importers complete"
                );
                Ok(())
            }
            Err(GraphError::NodeNotFound) => json_error(response, 404, "Symbol not found"),
            Err(e) => Err(e),
        }
    });
    result
}

// --- Multi-hop queries ---

pub fn handle_transitive_callers(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let start = Instant::now();
    let q = match parse_query(input, response)? {
        Some(q) => q,
        None => return Ok(()),
    };
    let storage = match input.collections.get_collection(&q.collection) {
        Ok(s) => s,
        Err(e) => return json_error(response, 404, &e.to_string()),
    };
    let repo = q.repo.as_deref();
    let depth = clamp_graph_depth(q.depth);
    let limit = clamp_graph_limit(q.limit);
    storage.with_read_backend(|r| {
        let starts = resolve_symbol_ids_be(&q, &storage, r);
        let results = bfs_reverse_be(&storage, r, &starts, &["CALLS"], depth, limit, repo)?;
        json_response(
            response,
            200,
            &sonic_rs::json!({"results": bfs_to_results(results)}),
        )
    })?;
    tracing::debug!(
        collection = %q.collection,
        elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
        "handle_transitive_callers complete"
    );
    Ok(())
}

pub fn handle_transitive_callees(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let start = Instant::now();
    let q = match parse_query(input, response)? {
        Some(q) => q,
        None => return Ok(()),
    };
    let collection = q.collection.clone();
    let storage = match input.collections.get_collection(&q.collection) {
        Ok(s) => s,
        Err(e) => return json_error(response, 404, &e.to_string()),
    };
    let repo = q.repo.as_deref();
    let depth = clamp_graph_depth(q.depth);
    let limit = clamp_graph_limit(q.limit);
    storage.with_read_backend(|r| {
        let starts = resolve_symbol_ids_be(&q, &storage, r);
        let results = bfs_forward_be(&storage, r, &starts, &["CALLS"], depth, limit, repo)?;
        json_response(
            response,
            200,
            &sonic_rs::json!({"results": bfs_to_results(results)}),
        )
    })?;
    tracing::debug!(
        collection = %collection,
        elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
        "handle_transitive_callees complete"
    );
    Ok(())
}

pub fn handle_impact(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    let start = Instant::now();
    let q = match parse_query(input, response)? {
        Some(q) => q,
        None => return Ok(()),
    };
    let collection = q.collection.clone();
    let storage = match input.collections.get_collection(&q.collection) {
        Ok(s) => s,
        Err(e) => return json_error(response, 404, &e.to_string()),
    };
    let repo = q.repo.as_deref();
    let depth = clamp_graph_depth(q.depth);
    let limit = clamp_graph_limit(q.limit);
    storage.with_read_backend(|r| {
        let starts = resolve_symbol_ids_be(&q, &storage, r);
        let results = bfs_impact_be(&storage, r, &starts, depth, limit, repo)?;
        json_response(
            response,
            200,
            &sonic_rs::json!({"results": bfs_to_results(results)}),
        )
    })?;
    tracing::debug!(
        collection = %collection,
        elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
        "handle_impact complete"
    );
    Ok(())
}

pub fn handle_cycles(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    let t0 = Instant::now();
    let q = match parse_query(input, response)? {
        Some(q) => q,
        None => return Ok(()),
    };
    let collection = q.collection.clone();
    let storage = match input.collections.get_collection(&q.collection) {
        Ok(s) => s,
        Err(e) => return json_error(response, 404, &e.to_string()),
    };
    let limit = clamp_graph_limit(q.limit);
    storage.with_read_backend(|r| {
        let start = resolve_symbol_id_be(&q, &storage, r);
        let cycles = detect_cycles_be(&storage, r, start, &["CALLS"], limit)?;
        let cycle_strs: Vec<Vec<String>> = cycles
            .into_iter()
            .map(|c| c.into_iter().map(|id| format!("{:032x}", id)).collect())
            .collect();
        json_response(response, 200, &sonic_rs::json!({"cycles": cycle_strs}))
    })?;
    tracing::debug!(
        collection = %collection,
        elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0,
        "handle_cycles complete"
    );
    Ok(())
}

pub fn handle_dependencies(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let start = Instant::now();
    let q = match parse_query(input, response)? {
        Some(q) => q,
        None => return Ok(()),
    };
    let collection = q.collection.clone();
    let storage = match input.collections.get_collection(&q.collection) {
        Ok(s) => s,
        Err(e) => return json_error(response, 404, &e.to_string()),
    };
    let repo = q.repo.as_deref();
    let depth = clamp_graph_depth(q.depth);
    let limit = clamp_graph_limit(q.limit);
    storage.with_read_backend(|r| {
        let starts = resolve_symbol_ids_be(&q, &storage, r);
        let results = bfs_forward_be(
            &storage,
            r,
            &starts,
            &["CALLS", "IMPORTS"],
            depth,
            limit,
            repo,
        )?;
        json_response(
            response,
            200,
            &sonic_rs::json!({"results": bfs_to_results(results)}),
        )
    })?;
    tracing::debug!(
        collection = %collection,
        elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
        "handle_dependencies complete"
    );
    Ok(())
}

pub fn handle_subclasses(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    let start = Instant::now();
    let q = match parse_query(input, response)? {
        Some(q) => q,
        None => return Ok(()),
    };
    let collection = q.collection.clone();
    let storage = match input.collections.get_collection(&q.collection) {
        Ok(s) => s,
        Err(e) => return json_error(response, 404, &e.to_string()),
    };
    let repo = q.repo.as_deref();
    let depth = clamp_graph_depth(q.depth);
    let limit = clamp_graph_limit(q.limit);
    storage.with_read_backend(|r| {
        let starts = resolve_symbol_ids_be(&q, &storage, r);
        let results = bfs_reverse_be(&storage, r, &starts, &["INHERITS_FROM"], depth, limit, repo)?;
        json_response(
            response,
            200,
            &sonic_rs::json!({"results": bfs_to_results(results)}),
        )
    })?;
    tracing::debug!(
        collection = %collection,
        elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
        "handle_subclasses complete"
    );
    Ok(())
}

pub fn handle_base_classes(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let start = Instant::now();
    let q = match parse_query(input, response)? {
        Some(q) => q,
        None => return Ok(()),
    };
    let collection = q.collection.clone();
    let storage = match input.collections.get_collection(&q.collection) {
        Ok(s) => s,
        Err(e) => return json_error(response, 404, &e.to_string()),
    };
    let repo = q.repo.as_deref();
    let depth = clamp_graph_depth(q.depth);
    let limit = clamp_graph_limit(q.limit);
    storage.with_read_backend(|r| {
        let starts = resolve_symbol_ids_be(&q, &storage, r);
        let results = bfs_forward_be(&storage, r, &starts, &["INHERITS_FROM"], depth, limit, repo)?;
        json_response(
            response,
            200,
            &sonic_rs::json!({"results": bfs_to_results(results)}),
        )
    })?;
    tracing::debug!(
        collection = %collection,
        elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
        "handle_base_classes complete"
    );
    Ok(())
}

// ── Algorithm request types ─────────────────────────────────────────────

#[derive(Deserialize)]
struct PageRankQuery {
    collection: String,
    #[serde(default = "default_edge_labels")]
    edge_labels: Vec<String>,
    #[serde(default = "default_pr_iterations")]
    iterations: usize,
    #[serde(default = "default_pr_damping")]
    damping: f64,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    repo: Option<String>,
    /// Bypass the materialized algorithm cache for debugging/canaries.
    #[serde(default, alias = "fresh")]
    live: bool,
}

fn default_edge_labels() -> Vec<String> {
    vec!["CALLS".to_string(), "IMPORTS".to_string()]
}
fn default_pr_iterations() -> usize {
    20
}
fn default_pr_damping() -> f64 {
    0.85
}

#[derive(Deserialize)]
struct CommunityQuery {
    collection: String,
    #[serde(default = "default_edge_labels")]
    edge_labels: Vec<String>,
    #[serde(default = "default_community_iterations")]
    max_iterations: usize,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    repo: Option<String>,
    /// Bypass the materialized algorithm cache for debugging/canaries.
    #[serde(default, alias = "fresh")]
    live: bool,
}

fn default_community_iterations() -> usize {
    50
}

#[derive(Deserialize)]
struct ShortestPathQuery {
    collection: String,
    from_symbol: String,
    to_symbol: String,
    #[serde(default)]
    from_path: Option<String>,
    #[serde(default)]
    to_path: Option<String>,
    #[serde(default = "default_sp_edge_labels")]
    edge_labels: Vec<String>,
    #[serde(default = "default_shortest_path_depth", alias = "max_depth")]
    depth: usize,
}

fn default_sp_edge_labels() -> Vec<String> {
    vec!["CALLS".to_string()]
}

#[derive(Deserialize)]
struct JaccardQuery {
    collection: String,
    symbol_a: String,
    symbol_b: String,
    #[serde(default)]
    path_a: Option<String>,
    #[serde(default)]
    path_b: Option<String>,
    #[serde(default = "default_edge_labels")]
    edge_labels: Vec<String>,
}

// ── Algorithm handlers ──────────────────────────────────────────────────

pub fn handle_pagerank(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    let start = Instant::now();
    let q: PageRankQuery = match sonic_rs::from_slice(&input.request.body) {
        Ok(q) => q,
        Err(e) => return json_error(response, 400, &format!("Invalid JSON: {}", e)),
    };
    let collection = q.collection.clone();
    let storage = match input.collections.get_collection(&q.collection) {
        Ok(s) => s,
        Err(e) => return json_error(response, 404, &e.to_string()),
    };
    let labels: Vec<&str> = q.edge_labels.iter().map(|s| s.as_str()).collect();
    let iterations = clamp_graph_iterations(q.iterations);
    let limit = clamp_graph_limit(q.limit);
    let repo = q.repo.as_deref();
    storage.with_read_backend(|r| {
        let ranks = if q.live {
            algorithms::pagerank_scoped(&storage, r, &labels, iterations, q.damping, repo)?
        } else {
            algorithms::cached_pagerank(
                &q.collection,
                &storage,
                r,
                &labels,
                iterations,
                q.damping,
                repo,
            )?
            .0
        };

        // Return top-N with node metadata
        let results: Vec<sonic_rs::Value> = ranks
            .into_iter()
            .take(limit)
            .filter_map(|(id, rank)| {
                storage.get_node_be(r, &id).ok().map(|node| {
                    sonic_rs::json!({
                        "id": format!("{:032x}", id),
                        "label": node.label,
                        "rank": rank,
                        "properties": node.properties
                    })
                })
            })
            .collect();

        json_response(response, 200, &sonic_rs::json!({"results": results}))
    })?;
    tracing::debug!(
        collection = %collection,
        elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
        "handle_pagerank complete"
    );
    Ok(())
}

pub fn handle_communities(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    let start = Instant::now();
    let q: CommunityQuery = match sonic_rs::from_slice(&input.request.body) {
        Ok(q) => q,
        Err(e) => return json_error(response, 400, &format!("Invalid JSON: {}", e)),
    };
    let collection = q.collection.clone();
    let storage = match input.collections.get_collection(&q.collection) {
        Ok(s) => s,
        Err(e) => return json_error(response, 404, &e.to_string()),
    };
    let labels: Vec<&str> = q.edge_labels.iter().map(|s| s.as_str()).collect();
    let iterations = clamp_graph_iterations(q.max_iterations);
    let limit = clamp_graph_limit(q.limit);
    let repo = q.repo.as_deref();
    storage.with_read_backend(|r| {
        let communities = if q.live {
            algorithms::label_propagation_scoped(&storage, r, &labels, iterations, repo)?
        } else {
            algorithms::cached_label_propagation(
                &q.collection,
                &storage,
                r,
                &labels,
                iterations,
                repo,
            )?
            .0
        };

        let results: Vec<sonic_rs::Value> = communities
            .into_iter()
            .take(limit)
            .map(|(community_id, members)| {
                sonic_rs::json!({
                    "community_id": format!("{:032x}", community_id),
                    "size": members.len(),
                    "members": members.iter().map(|id| format!("{:032x}", id)).collect::<Vec<_>>()
                })
            })
            .collect();

        json_response(
            response,
            200,
            &sonic_rs::json!({"communities": results, "total": results.len()}),
        )
    })?;
    tracing::debug!(
        collection = %collection,
        elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
        "handle_communities complete"
    );
    Ok(())
}

pub fn handle_shortest_path(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let start = Instant::now();
    let q: ShortestPathQuery = match sonic_rs::from_slice(&input.request.body) {
        Ok(q) => q,
        Err(e) => return json_error(response, 400, &format!("Invalid JSON: {}", e)),
    };
    let collection = q.collection.clone();
    let storage = match input.collections.get_collection(&q.collection) {
        Ok(s) => s,
        Err(e) => return json_error(response, 404, &e.to_string()),
    };
    let from_path = q.from_path.as_deref().unwrap_or("");
    let to_path = q.to_path.as_deref().unwrap_or("");
    let from_id = deterministic_id::node_id(&q.collection, "Symbol", &q.from_symbol, from_path);
    let to_id = deterministic_id::node_id(&q.collection, "Symbol", &q.to_symbol, to_path);

    let labels: Vec<&str> = q.edge_labels.iter().map(|s| s.as_str()).collect();
    let depth = clamp_graph_depth(q.depth);
    let max_visited = graph_shortest_path_visited_max();
    storage.with_read_backend(|r| {
        let (path_nodes, distance) =
            algorithms::shortest_path(&storage, r, from_id, to_id, &labels, depth, max_visited)?;

        let nodes: Vec<NodeResult> = path_nodes
            .into_iter()
            .map(|node| NodeResult {
                id: format!("{:032x}", node.id),
                label: node.label,
                properties: node.properties,
                depth: None,
            })
            .collect();

        json_response(
            response,
            200,
            &sonic_rs::json!({"path": nodes, "distance": distance}),
        )
    })?;
    tracing::debug!(
        collection = %collection,
        elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
        "handle_shortest_path complete"
    );
    Ok(())
}

pub fn handle_jaccard(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    let start = Instant::now();
    let q: JaccardQuery = match sonic_rs::from_slice(&input.request.body) {
        Ok(q) => q,
        Err(e) => return json_error(response, 400, &format!("Invalid JSON: {}", e)),
    };
    let collection = q.collection.clone();
    let storage = match input.collections.get_collection(&q.collection) {
        Ok(s) => s,
        Err(e) => return json_error(response, 404, &e.to_string()),
    };
    let path_a = q.path_a.as_deref().unwrap_or("");
    let path_b = q.path_b.as_deref().unwrap_or("");
    let id_a = deterministic_id::node_id(&q.collection, "Symbol", &q.symbol_a, path_a);
    let id_b = deterministic_id::node_id(&q.collection, "Symbol", &q.symbol_b, path_b);

    let labels: Vec<&str> = q.edge_labels.iter().map(|s| s.as_str()).collect();
    storage.with_read_backend(|r| {
        let similarity = algorithms::jaccard_similarity(&storage, r, id_a, id_b, &labels)?;
        json_response(response, 200, &sonic_rs::json!({"similarity": similarity}))
    })?;
    tracing::debug!(
        collection = %collection,
        elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
        "handle_jaccard complete"
    );
    Ok(())
}

// --- Delete by path ---

#[derive(Deserialize)]
struct DeleteByPathRequest {
    collection: String,
    path: String,
    #[serde(default)]
    repo: Option<String>,
    /// If true, also delete nodes that become orphaned (no remaining edges).
    #[serde(default)]
    delete_orphaned_nodes: bool,
}

#[derive(Deserialize)]
struct DeleteByPathsRequest {
    collection: String,
    paths: Vec<String>,
    #[serde(default)]
    repo: Option<String>,
    /// If true, also delete nodes that become orphaned (no remaining edges).
    #[serde(default)]
    delete_orphaned_nodes: bool,
}

#[derive(Deserialize)]
struct BackfillEdgePathIndexRequest {
    collection: String,
    #[serde(default = "default_edge_path_backfill_batch")]
    batch_size: usize,
}

fn default_edge_path_backfill_batch() -> usize {
    1000
}

pub fn handle_backfill_edge_path_index(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let start = Instant::now();
    let req: BackfillEdgePathIndexRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_error(response, 400, &format!("Invalid JSON: {}", e)),
    };
    let collection = req.collection.clone();
    let storage = match input.collections.get_collection(&req.collection) {
        Ok(s) => s,
        Err(e) => return json_error(response, 404, &e.to_string()),
    };

    let batch_size = req.batch_size.clamp(1, 10_000);
    let (processed, complete, cursor) = storage.backfill_edge_path_index_batch(batch_size)?;

    json_response(
        response,
        200,
        &sonic_rs::json!({
            "processed": processed,
            "complete": complete,
            "cursor": cursor.map(|c| c.to_string()),
        }),
    )?;
    tracing::debug!(
        collection = %collection,
        elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
        "handle_backfill_edge_path_index complete"
    );
    Ok(())
}

#[derive(Deserialize)]
struct BackfillAdjacencyFromPointsRequest {
    collection: String,
    #[serde(default = "default_adjacency_backfill_batch")]
    batch_size: usize,
    #[serde(default)]
    force: bool,
}

fn default_adjacency_backfill_batch() -> usize {
    1000
}

/// Rebuild native graph adjacency from relationship points already stored in a
/// `_graph` collection. Relationship points carry `caller_symbol`,
/// `callee_symbol`, `caller_path`, `callee_path`, and `edge_type`; this
/// endpoint turns each one into the caller/callee Symbol nodes and the
/// deterministic edge that the native graph traversal endpoints read.
pub fn handle_backfill_adjacency_from_points(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let start = Instant::now();
    let req: BackfillAdjacencyFromPointsRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_error(response, 400, &format!("Invalid JSON: {}", e)),
    };
    let collection = req.collection.clone();
    let storage = match input.collections.get_collection(&req.collection) {
        Ok(s) => s,
        Err(e) => return json_error(response, 404, &e.to_string()),
    };

    let batch_size = req.batch_size.clamp(1, 10_000);
    let (nodes_upserted, edges_upserted, complete, cursor) =
        storage.backfill_adjacency_from_points_batch(&req.collection, batch_size, req.force)?;

    json_response(
        response,
        200,
        &sonic_rs::json!({
            "nodes_upserted": nodes_upserted,
            "edges_upserted": edges_upserted,
            "complete": complete,
            "cursor": cursor.map(|c| c.to_string()),
        }),
    )?;
    tracing::debug!(
        collection = %collection,
        elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
        "handle_backfill_adjacency_from_points complete"
    );
    Ok(())
}

#[derive(Deserialize)]
struct RebuildAdjacencyForPathsRequest {
    collection: String,
    paths: Vec<String>,
    #[serde(default)]
    base_collection: Option<String>,
}

/// Incrementally rebuild native graph adjacency for a delta of changed source
/// paths, deriving edges from the `_graph` relationship points already stored on
/// S3. CE calls this after re-indexing changed files — paired with
/// `delete_by_path(s)`, which wipes the stale edges first — so native graph stays
/// fresh without a full `backfill_adjacency_from_points` rebuild or a per-edge
/// dual-write. The adjacency persists in the same SlateDB namespaces readers
/// already traverse and cache.
///
/// When the request also carries `base_collection`, this additionally derives
/// chunk-level edges (CALLS/IMPORTS/INHERITS_FROM between Qdrant point ids)
/// into that collection, so the hybrid-search graph channel — which walks
/// adjacency keyed by point id in the base collection — has edges to
/// traverse. A missing/unavailable `base_collection` only skips that half of
/// the work; the `_graph` rebuild above always completes.
pub fn handle_rebuild_adjacency_for_paths(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let start = Instant::now();
    let req: RebuildAdjacencyForPathsRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_error(response, 400, &format!("Invalid JSON: {}", e)),
    };
    let collection = req.collection.clone();
    let storage = match input.collections.get_collection(&req.collection) {
        Ok(s) => s,
        Err(e) => return json_error(response, 404, &e.to_string()),
    };

    let (nodes_upserted, edges_upserted, chunk_candidates) =
        storage.rebuild_adjacency_for_paths(&req.collection, &req.paths)?;

    let mut chunk_edges_upserted = 0usize;
    let mut warning: Option<String> = None;
    if let Some(base_collection) = &req.base_collection {
        match input.collections.get_collection(base_collection) {
            Ok(base_storage) => {
                chunk_edges_upserted =
                    apply_chunk_edges(&base_storage, base_collection, &chunk_candidates)?;
            }
            Err(e) => {
                warning = Some(format!(
                    "base_collection '{}' unavailable: {}",
                    base_collection, e
                ));
            }
        }
    }

    json_response(
        response,
        200,
        &sonic_rs::json!({
            "nodes_upserted": nodes_upserted,
            "edges_upserted": edges_upserted,
            "paths": req.paths.len(),
            "chunk_edges_upserted": chunk_edges_upserted,
            "warning": warning,
        }),
    )?;
    tracing::debug!(
        collection = %collection,
        paths = req.paths.len(),
        nodes_upserted,
        edges_upserted,
        chunk_edges_upserted,
        elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
        "handle_rebuild_adjacency_for_paths complete"
    );
    Ok(())
}

/// Resolve a chunk-edge endpoint string to the point's node id. CE renders
/// numeric Qdrant point ids as decimal strings, so decimal parses first; a
/// hex/hash fallback covers UUID-style string point ids, matching how the
/// point's "point" node id was derived on ingest (`point_id_to_u128`).
fn chunk_edge_point_id(raw: &str) -> u128 {
    u128::from_str_radix(raw, 10).unwrap_or_else(|_| string_id_to_u128(raw))
}

/// Apply chunk-edge candidates derived from `_graph` relationship points to
/// the base collection, where every vector point already has a "point" node
/// whose id is the point id. Both endpoints must already exist — this never
/// creates nodes, so a caller/callee id CE couldn't resolve (or one
/// belonging to a different, contaminated collection) is silently skipped
/// instead of creating a dangling edge. All upserts run in one write batch.
fn apply_chunk_edges(
    storage: &HelixGraphStorage,
    base_collection: &str,
    candidates: &[ChunkEdgeCandidate],
) -> Result<usize, GraphError> {
    if candidates.is_empty() {
        return Ok(0);
    }
    storage.with_write_backend(|w| {
        let mut upserted = 0usize;
        for candidate in candidates {
            let from = chunk_edge_point_id(&candidate.caller_point_id);
            let to = chunk_edge_point_id(&candidate.callee_point_id);

            let from_exists = storage
                .backend
                .get_for_update(w, Namespace::Nodes, &from.to_be_bytes(), |v| v.is_some())
                .map_err(|e| GraphError::New(e.to_string()))?;
            if !from_exists {
                continue;
            }
            let to_exists = storage
                .backend
                .get_for_update(w, Namespace::Nodes, &to.to_be_bytes(), |v| v.is_some())
                .map_err(|e| GraphError::New(e.to_string()))?;
            if !to_exists {
                continue;
            }

            let id = deterministic_id::edge_id(
                base_collection,
                &candidate.edge_type,
                &candidate.caller_point_id,
                &candidate.callee_point_id,
                &candidate.caller_path,
                &candidate.callee_path,
            );
            storage.upsert_edge_be(
                w,
                &EdgeUpsert {
                    id,
                    label: candidate.edge_type.clone(),
                    from_node: from,
                    to_node: to,
                    properties: HashMap::from([
                        (
                            "edge_type".to_string(),
                            Value::String(candidate.edge_type.clone()),
                        ),
                        (
                            "caller_path".to_string(),
                            Value::String(candidate.caller_path.clone()),
                        ),
                    ]),
                },
            )?;
            upserted += 1;
        }
        Ok(upserted)
    })
}

/// Delete all edges where `from_path` or `to_path` matches the given path.
/// Used by CE's incremental re-indexing to wipe stale edges before re-ingesting.
///
/// The request remains a single batched cleanup call, but the delete phase is
/// split into bounded write transactions so one stale-file cleanup cannot hold
/// the LMDB writer for the full `MAX_EDGES_PER_CALL` pass. Each chunk still runs
/// inside [`HelixGraphStorage::with_write_txn`] so `MapFull` gets the normal
/// auto-resize + retry behavior.
pub fn handle_delete_by_path(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    const ROUTE: &str = "/v1/graph/delete_by_path";
    let total_start = Instant::now();
    let req: DeleteByPathRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_error(response, 400, &format!("Invalid JSON: {}", e)),
    };
    let storage = match input.collections.get_collection(&req.collection) {
        Ok(s) => s,
        Err(e) => return json_error(response, 404, &e.to_string()),
    };

    const MAX_EDGES_PER_CALL: usize = 50_000;

    let read_start = Instant::now();
    let (edge_ids_to_delete, affected_nodes): (Vec<u128>, HashSet<u128>) = if storage.backend.kind()
        == BackendKind::Lsm
    {
        let r = storage
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))?;
        let mut edge_ids = Vec::new();
        let mut nodes = HashSet::new();

        for edge_id in storage.edge_ids_for_path_be(&r, &req.path, MAX_EDGES_PER_CALL)? {
            let edge = match storage.get_edge_be(&r, &edge_id) {
                Ok(edge) => edge,
                Err(_) => continue,
            };
            if !graph_edge_repo_matches(&edge, req.repo.as_deref()) {
                continue;
            }
            edge_ids.push(edge_id);
            if req.delete_orphaned_nodes {
                nodes.insert(edge.from_node);
                nodes.insert(edge.to_node);
            }
        }

        if !storage.edge_path_index_backfill_complete_be(&r)? && edge_ids.len() < MAX_EDGES_PER_CALL
        {
            let mut scan_err: Option<GraphError> = None;
            storage
                .backend
                .scan(&r, Namespace::Edges, KeyRange::all(), |k, data| {
                    if edge_ids.len() >= MAX_EDGES_PER_CALL {
                        return false;
                    }
                    let Ok(raw) = <[u8; 16]>::try_from(k) else {
                        scan_err = Some(GraphError::New(
                            "invalid edge key length during LSM delete_by_path scan".into(),
                        ));
                        return false;
                    };
                    let edge_id = u128::from_be_bytes(raw);
                    if edge_ids.contains(&edge_id) {
                        return true;
                    }
                    match SerializedEdge::decode_edge(data, edge_id) {
                        Ok(edge)
                            if graph_edge_repo_matches(&edge, req.repo.as_deref())
                                && graph_edge_matches_path(&edge, &req.path) =>
                        {
                            edge_ids.push(edge_id);
                            if req.delete_orphaned_nodes {
                                nodes.insert(edge.from_node);
                                nodes.insert(edge.to_node);
                            }
                        }
                        Ok(_) => {}
                        Err(e) => {
                            scan_err = Some(e);
                            return false;
                        }
                    }
                    true
                })
                .map_err(|e| GraphError::New(e.to_string()))?;
            if let Some(e) = scan_err {
                return Err(e);
            }
        }

        Ok((edge_ids, nodes))
    } else {
        storage.with_read_txn(|rtxn| {
            let mut edge_ids = Vec::new();
            let mut nodes = HashSet::new();
            let edges_db = storage.lmdb_edges_db()?;

            for edge_id in storage.edge_ids_for_path(rtxn, &req.path, MAX_EDGES_PER_CALL)? {
                let Some(val_bytes) = edges_db.get(rtxn, &edge_id)? else {
                    continue;
                };
                let edge = match SerializedEdge::decode_edge(val_bytes, edge_id) {
                    Ok(edge) => edge,
                    Err(_) => continue,
                };
                if !graph_edge_repo_matches(&edge, req.repo.as_deref()) {
                    continue;
                }
                edge_ids.push(edge_id);
                if req.delete_orphaned_nodes {
                    nodes.insert(edge.from_node);
                    nodes.insert(edge.to_node);
                }
            }

            if !storage.edge_path_index_backfill_complete(rtxn)?
                && edge_ids.len() < MAX_EDGES_PER_CALL
            {
                for result in edges_db.iter(rtxn)? {
                    if edge_ids.len() >= MAX_EDGES_PER_CALL {
                        break;
                    }
                    let (edge_id, val_bytes) = result?;
                    if edge_ids.contains(&edge_id) {
                        continue;
                    }
                    let edge = match SerializedEdge::decode_edge(val_bytes, edge_id) {
                        Ok(edge) => edge,
                        Err(_) => continue,
                    };
                    if !graph_edge_repo_matches(&edge, req.repo.as_deref()) {
                        continue;
                    }
                    if graph_edge_matches_path(&edge, &req.path) {
                        edge_ids.push(edge_id);
                        if req.delete_orphaned_nodes {
                            nodes.insert(edge.from_node);
                            nodes.insert(edge.to_node);
                        }
                    }
                }
            }

            Ok((edge_ids, nodes))
        })
    }?;
    observe_graph_delete_phase(ROUTE, "read_resolve", read_start.elapsed());
    observe_graph_delete_items(ROUTE, "candidate_edges", edge_ids_to_delete.len());
    observe_graph_delete_items(ROUTE, "candidate_nodes", affected_nodes.len());

    if edge_ids_to_delete.is_empty() && affected_nodes.is_empty() {
        observe_graph_delete_phase(ROUTE, "total", total_start.elapsed());
        return json_response(
            response,
            200,
            &sonic_rs::json!({
                "edges_deleted": 0,
                "nodes_deleted": 0,
                "truncated": false,
            }),
        );
    }

    let edge_write_start = Instant::now();
    let edges_deleted = delete_edges_in_write_chunks(&storage, &edge_ids_to_delete)?;
    observe_graph_delete_phase(ROUTE, "edge_write", edge_write_start.elapsed());
    let nodes_deleted = if req.delete_orphaned_nodes {
        let node_write_start = Instant::now();
        let deleted = delete_orphaned_nodes_in_write_chunks(&storage, &affected_nodes)?;
        observe_graph_delete_phase(ROUTE, "node_write", node_write_start.elapsed());
        deleted
    } else {
        observe_graph_delete_phase(ROUTE, "node_write", std::time::Duration::ZERO);
        0
    };
    observe_graph_delete_items(ROUTE, "deleted_edges", edges_deleted);
    observe_graph_delete_items(ROUTE, "deleted_nodes", nodes_deleted);
    observe_graph_delete_phase(ROUTE, "total", total_start.elapsed());
    if edges_deleted > 0 || nodes_deleted > 0 {
        algorithms::invalidate_algorithm_cache(&req.collection);
    }

    json_response(
        response,
        200,
        &sonic_rs::json!({
            "edges_deleted": edges_deleted,
            "nodes_deleted": nodes_deleted,
            "truncated": edge_ids_to_delete.len() >= MAX_EDGES_PER_CALL,
        }),
    )
}

/// Delete all edges where `from_path` or `to_path` matches any path in the
/// provided batch. This is the plural form used by CE to collapse a burst of
/// per-file graph cleanup requests into one indexed read phase. The write phase
/// is chunked internally to preserve batching speed without monopolizing the
/// LMDB writer for the full batch.
pub fn handle_delete_by_paths(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    const ROUTE: &str = "/v1/graph/delete_by_paths";
    let total_start = Instant::now();
    let req: DeleteByPathsRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_error(response, 400, &format!("Invalid JSON: {}", e)),
    };
    let storage = match input.collections.get_collection(&req.collection) {
        Ok(s) => s,
        Err(e) => return json_error(response, 404, &e.to_string()),
    };

    const MAX_PATHS_PER_CALL: usize = 1_000;
    const MAX_EDGES_PER_CALL: usize = 50_000;

    let mut path_set: HashSet<String> = HashSet::new();
    for path in req.paths.into_iter().take(MAX_PATHS_PER_CALL) {
        let p = path.trim();
        if !p.is_empty() {
            path_set.insert(p.to_string());
        }
    }
    if path_set.is_empty() {
        observe_graph_delete_phase(ROUTE, "total", total_start.elapsed());
        return json_response(
            response,
            200,
            &sonic_rs::json!({
                "edges_deleted": 0,
                "nodes_deleted": 0,
                "paths": 0,
                "truncated": false,
            }),
        );
    }

    let read_start = Instant::now();
    let (edge_ids_to_delete, affected_nodes): (Vec<u128>, HashSet<u128>) = if storage.backend.kind()
        == BackendKind::Lsm
    {
        let r = storage
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))?;
        let mut edge_ids = Vec::new();
        let mut seen_edge_ids: HashSet<u128> = HashSet::new();
        let mut nodes = HashSet::new();

        for path in &path_set {
            if edge_ids.len() >= MAX_EDGES_PER_CALL {
                break;
            }
            let remaining = MAX_EDGES_PER_CALL - edge_ids.len();
            for edge_id in storage.edge_ids_for_path_be(&r, path, remaining)? {
                if !seen_edge_ids.insert(edge_id) {
                    continue;
                }
                let edge = match storage.get_edge_be(&r, &edge_id) {
                    Ok(edge) => edge,
                    Err(_) => continue,
                };
                if !graph_edge_repo_matches(&edge, req.repo.as_deref()) {
                    continue;
                }
                edge_ids.push(edge_id);
                if req.delete_orphaned_nodes {
                    nodes.insert(edge.from_node);
                    nodes.insert(edge.to_node);
                }
            }
        }

        if !storage.edge_path_index_backfill_complete_be(&r)? && edge_ids.len() < MAX_EDGES_PER_CALL
        {
            let mut scan_err: Option<GraphError> = None;
            storage
                .backend
                .scan(&r, Namespace::Edges, KeyRange::all(), |k, data| {
                    if edge_ids.len() >= MAX_EDGES_PER_CALL {
                        return false;
                    }
                    let Ok(raw) = <[u8; 16]>::try_from(k) else {
                        scan_err = Some(GraphError::New(
                            "invalid edge key length during LSM delete_by_paths scan".into(),
                        ));
                        return false;
                    };
                    let edge_id = u128::from_be_bytes(raw);
                    if seen_edge_ids.contains(&edge_id) {
                        return true;
                    }
                    match SerializedEdge::decode_edge(data, edge_id) {
                        Ok(edge)
                            if graph_edge_repo_matches(&edge, req.repo.as_deref())
                                && graph_edge_matches_any_path(&edge, &path_set) =>
                        {
                            seen_edge_ids.insert(edge_id);
                            edge_ids.push(edge_id);
                            if req.delete_orphaned_nodes {
                                nodes.insert(edge.from_node);
                                nodes.insert(edge.to_node);
                            }
                        }
                        Ok(_) => {}
                        Err(e) => {
                            scan_err = Some(e);
                            return false;
                        }
                    }
                    true
                })
                .map_err(|e| GraphError::New(e.to_string()))?;
            if let Some(e) = scan_err {
                return Err(e);
            }
        }

        Ok((edge_ids, nodes))
    } else {
        storage.with_read_txn(|rtxn| {
            let mut edge_ids = Vec::new();
            let mut seen_edge_ids: HashSet<u128> = HashSet::new();
            let mut nodes = HashSet::new();
            let edges_db = storage.lmdb_edges_db()?;

            for path in &path_set {
                if edge_ids.len() >= MAX_EDGES_PER_CALL {
                    break;
                }
                let remaining = MAX_EDGES_PER_CALL - edge_ids.len();
                for edge_id in storage.edge_ids_for_path(rtxn, path, remaining)? {
                    if !seen_edge_ids.insert(edge_id) {
                        continue;
                    }
                    let Some(val_bytes) = edges_db.get(rtxn, &edge_id)? else {
                        continue;
                    };
                    let edge = match SerializedEdge::decode_edge(val_bytes, edge_id) {
                        Ok(edge) => edge,
                        Err(_) => continue,
                    };
                    if !graph_edge_repo_matches(&edge, req.repo.as_deref()) {
                        continue;
                    }
                    edge_ids.push(edge_id);
                    if req.delete_orphaned_nodes {
                        nodes.insert(edge.from_node);
                        nodes.insert(edge.to_node);
                    }
                }
            }

            if !storage.edge_path_index_backfill_complete(rtxn)?
                && edge_ids.len() < MAX_EDGES_PER_CALL
            {
                for result in edges_db.iter(rtxn)? {
                    if edge_ids.len() >= MAX_EDGES_PER_CALL {
                        break;
                    }
                    let (edge_id, val_bytes) = result?;
                    if seen_edge_ids.contains(&edge_id) {
                        continue;
                    }
                    let edge = match SerializedEdge::decode_edge(val_bytes, edge_id) {
                        Ok(edge) => edge,
                        Err(_) => continue,
                    };
                    if !graph_edge_repo_matches(&edge, req.repo.as_deref()) {
                        continue;
                    }
                    if graph_edge_matches_any_path(&edge, &path_set) {
                        seen_edge_ids.insert(edge_id);
                        edge_ids.push(edge_id);
                        if req.delete_orphaned_nodes {
                            nodes.insert(edge.from_node);
                            nodes.insert(edge.to_node);
                        }
                    }
                }
            }

            Ok((edge_ids, nodes))
        })
    }?;
    observe_graph_delete_phase(ROUTE, "read_resolve", read_start.elapsed());
    observe_graph_delete_items(ROUTE, "candidate_edges", edge_ids_to_delete.len());
    observe_graph_delete_items(ROUTE, "candidate_nodes", affected_nodes.len());

    if edge_ids_to_delete.is_empty() && affected_nodes.is_empty() {
        observe_graph_delete_phase(ROUTE, "total", total_start.elapsed());
        return json_response(
            response,
            200,
            &sonic_rs::json!({
                "edges_deleted": 0,
                "nodes_deleted": 0,
                "paths": path_set.len(),
                "truncated": false,
            }),
        );
    }

    let edge_write_start = Instant::now();
    let edges_deleted = delete_edges_in_write_chunks(&storage, &edge_ids_to_delete)?;
    observe_graph_delete_phase(ROUTE, "edge_write", edge_write_start.elapsed());
    let nodes_deleted = if req.delete_orphaned_nodes {
        let node_write_start = Instant::now();
        let deleted = delete_orphaned_nodes_in_write_chunks(&storage, &affected_nodes)?;
        observe_graph_delete_phase(ROUTE, "node_write", node_write_start.elapsed());
        deleted
    } else {
        observe_graph_delete_phase(ROUTE, "node_write", std::time::Duration::ZERO);
        0
    };
    observe_graph_delete_items(ROUTE, "deleted_edges", edges_deleted);
    observe_graph_delete_items(ROUTE, "deleted_nodes", nodes_deleted);
    observe_graph_delete_phase(ROUTE, "total", total_start.elapsed());
    if edges_deleted > 0 || nodes_deleted > 0 {
        algorithms::invalidate_algorithm_cache(&req.collection);
    }

    json_response(
        response,
        200,
        &sonic_rs::json!({
            "edges_deleted": edges_deleted,
            "nodes_deleted": nodes_deleted,
            "paths": path_set.len(),
            "truncated": edge_ids_to_delete.len() >= MAX_EDGES_PER_CALL,
        }),
    )
}

// ─── Subgraph (for visualization) ────────────────────────────────────────

#[derive(Deserialize)]
struct SubgraphQuery {
    collection: String,
    /// Node IDs to include. If empty, uses top-N by PageRank.
    #[serde(default)]
    node_ids: Vec<String>,
    #[serde(default = "default_subgraph_edge_labels")]
    edge_labels: Vec<String>,
    #[serde(default = "default_subgraph_limit")]
    limit: usize,
    /// Optional repo filter. If node_ids is empty, sample top N by repo-scoped PageRank.
    #[serde(default)]
    repo: Option<String>,
}

fn default_subgraph_edge_labels() -> Vec<String> {
    vec!["CALLS".into(), "IMPORTS".into()]
}
fn default_subgraph_limit() -> usize {
    50
}

fn value_as_rank(v: Option<&Value>) -> f64 {
    match v {
        Some(Value::F64(f)) => *f,
        Some(Value::F32(f)) => *f as f64,
        Some(Value::I64(i)) => *i as f64,
        Some(Value::U64(u)) => *u as f64,
        _ => 0.0,
    }
}

/// POST /v1/graph/subgraph — return nodes + edges between a set of node IDs.
/// Used by the graph visualization page.
///
/// If `node_ids` is empty, returns top-N nodes by PageRank + their interconnecting edges.
pub fn handle_subgraph(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    let start = Instant::now();
    let q: SubgraphQuery = match sonic_rs::from_slice(&input.request.body) {
        Ok(q) => q,
        Err(e) => return json_error(response, 400, &format!("Invalid JSON: {}", e)),
    };
    let collection = q.collection.clone();
    let storage = match input.collections.get_collection(&q.collection) {
        Ok(s) => s,
        Err(e) => return json_error(response, 404, &e.to_string()),
    };
    let labels: Vec<&str> = q.edge_labels.iter().map(|s| s.as_str()).collect();
    let limit = clamp_graph_limit(q.limit);
    let repo = q.repo.as_deref();

    storage.with_read_backend(|r| {
        // Resolve node IDs: either from request or from PageRank top-N. PageRank
        // now runs identically on both backends through the shared AnyRead
        // snapshot, so the visualization gets real ranks on LSM too.
        let node_ids: Vec<u128> = if q.node_ids.is_empty() {
            let iterations = clamp_graph_iterations(default_pr_iterations());
            algorithms::pagerank_scoped(&storage, r, &labels, iterations, 0.85, repo)?
                .into_iter()
                .take(limit)
                .map(|(id, _rank)| id)
                .collect()
        } else {
            q.node_ids
                .iter()
                .take(limit)
                .filter_map(|s| u128::from_str_radix(s.trim_start_matches("0x"), 16).ok())
                .collect()
        };

        let node_set: HashSet<u128> = node_ids.iter().copied().collect();

        // Build node list with metadata
        let mut nodes: Vec<sonic_rs::Value> = Vec::new();
        for &id in &node_ids {
            if let Ok(node) = storage.get_node_be(r, &id) {
                let importance = value_as_rank(
                    node.properties
                        .get("pagerank_scoped")
                        .or_else(|| node.properties.get("pagerank")),
                );
                let symbol_type = node
                    .properties
                    .get("symbol_type")
                    .and_then(|v| match v {
                        Value::String(s) => Some(s.as_str()),
                        _ => None,
                    })
                    .unwrap_or("function");
                nodes.push(sonic_rs::json!({
                    "id": format!("{:032x}", id),
                    "name": node.properties.get("name").and_then(|v| match v {
                        Value::String(s) => Some(s.as_str()),
                        _ => None,
                    }).unwrap_or("unknown"),
                    "type": symbol_type,
                    "importance": importance,
                    "path": node.properties.get("path").and_then(|v| match v {
                        Value::String(s) => Some(s.as_str()),
                        _ => None,
                    }).unwrap_or(""),
                }));
            }
        }

        // Find edges between nodes in the set.
        let label_hashes: Vec<[u8; 4]> = labels.iter().map(|l| hash_label(l, None)).collect();
        let mut edges: Vec<sonic_rs::Value> = Vec::new();
        let mut edge_count = 0usize;

        for &source_id in &node_ids {
            if edge_count >= 200 {
                break;
            }
            for label_hash in &label_hashes {
                let pairs = storage
                    .adjacency_pairs_be(r, source_id, label_hash, true)
                    .unwrap_or_default();
                for (target_id, edge_id) in pairs {
                    if node_set.contains(&target_id) {
                        if repo.is_some() {
                            match storage.get_edge_be(r, &edge_id) {
                                Ok(edge) if properties_match_repo(&edge.properties, repo) => {}
                                _ => continue,
                            }
                        }
                        let label_idx = label_hashes
                            .iter()
                            .position(|h| h == label_hash)
                            .unwrap_or(0);
                        edges.push(sonic_rs::json!({
                            "source": format!("{:032x}", source_id),
                            "target": format!("{:032x}", target_id),
                            "type": labels.get(label_idx).unwrap_or(&"CALLS"),
                        }));
                        edge_count += 1;
                        if edge_count >= 200 {
                            break;
                        }
                    }
                }
            }
        }

        json_response(
            response,
            200,
            &sonic_rs::json!({"ok": true, "nodes": nodes, "edges": edges}),
        )
    })?;
    tracing::debug!(
        collection = %collection,
        elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
        "handle_subgraph complete"
    );
    Ok(())
}

// ─── Distinct Values (cheap via keyword payload index) ───────────────────

#[derive(Deserialize)]
struct DistinctQuery {
    collection: String,
    field: String,
    #[serde(default = "default_distinct_limit")]
    limit: usize,
    /// When set, returns a parallel `counts` array with the number of
    /// nodes carrying each distinct value. Still O(distinct_values)
    /// because duplicates only bump the last count.
    #[serde(default)]
    with_counts: bool,
}

fn default_distinct_limit() -> usize {
    1000
}

/// POST /v1/graph/distinct — distinct values of a keyword-indexed node property.
///
/// This endpoint deliberately requires a keyword payload index on `field`:
/// it walks the index DB and dedupes keys as they pass, so the work scales
/// with `O(distinct_values)` rather than a full node scan. If no index
/// exists the request is rejected with 400 so a missing index can never
/// trigger a surprise scan over millions of nodes.
///
/// Create the required index first via:
///   PUT /collections/{name}/index   {"field_name": "repo", "field_schema": "keyword"}
pub fn handle_distinct_values(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let start = Instant::now();
    let q: DistinctQuery = match sonic_rs::from_slice(&input.request.body) {
        Ok(q) => q,
        Err(e) => return json_error(response, 400, &format!("Invalid JSON: {}", e)),
    };
    if q.field.is_empty() {
        return json_error(response, 400, "Missing 'field'");
    }
    let collection = q.collection.clone();
    let storage = match input.collections.get_collection(&q.collection) {
        Ok(s) => s,
        Err(e) => return json_error(response, 404, &e.to_string()),
    };
    let Some(schema) = storage.has_payload_index(&q.field) else {
        return json_error(
            response,
            400,
            &format!(
                "Field '{}' has no payload index — create one via PUT /collections/{{name}}/index to enable cheap distinct scans",
                q.field
            ),
        );
    };
    if schema != PayloadIndexSchema::Keyword {
        return json_error(
            response,
            400,
            &format!(
                "Field '{}' index is not keyword-typed (distinct scan only supports keyword indices)",
                q.field
            ),
        );
    }

    // Walk the DUP_SORT index DB; keys are sorted, so we only decode a new
    // key when it differs from the previous one. This keeps per-duplicate
    // cost to a single byte-slice comparison. When `with_counts` is set,
    // duplicates bump the last value's count instead of being skipped.
    let mut values: Vec<Value> = Vec::new();
    let mut counts: Vec<u64> = Vec::new();
    let mut last_key: Vec<u8> = Vec::new();
    let mut last_key_emitted = false;
    let mut first = true;
    let db_name = HelixGraphStorage::payload_index_db_name(&q.field, &schema);
    storage.with_read_backend(|r| {
        storage
            .backend
            .scan(
                r,
                Namespace::PayloadIndex(&db_name),
                KeyRange::all(),
                |key, _val| {
                    if !first && last_key == key {
                        if q.with_counts && last_key_emitted {
                            if let Some(c) = counts.last_mut() {
                                *c += 1;
                            }
                        }
                        return true;
                    }
                    first = false;
                    last_key_emitted = false;
                    if let Ok(v) = bincode::deserialize::<Value>(key) {
                        values.push(v);
                        if q.with_counts {
                            counts.push(1);
                        }
                        last_key_emitted = true;
                        if values.len() >= q.limit {
                            return false;
                        }
                    }
                    last_key.clear();
                    last_key.extend_from_slice(key);
                    true
                },
            )
            .map_err(|e| GraphError::New(e.to_string()))
    })?;

    if q.with_counts {
        json_response(
            response,
            200,
            &sonic_rs::json!({"values": values, "counts": counts}),
        )?;
    } else {
        json_response(response, 200, &sonic_rs::json!({"values": values}))?;
    }
    tracing::debug!(
        collection = %collection,
        elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
        "handle_distinct_values complete"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helix_engine::graph_core::config::Config;
    use crate::helix_engine::graph_core::graph_core::{HelixGraphEngine, HelixGraphEngineOpts};
    use crate::helix_engine::storage_core::backend::Namespace;
    use crate::helix_engine::storage_core::upsert::{EdgeUpsert, NodeUpsert};
    use crate::helix_engine::storage_core::{
        collection_manager::CollectionManager, replication::ReplicationManager,
    };
    use crate::protocol::request::Request;
    use std::collections::{HashMap, HashSet};
    use std::sync::{Arc, Mutex};
    use tempfile::TempDir;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn clear_budget_env() {
        std::env::remove_var("HELIX_GRAPH_DEPTH_MAX");
        std::env::remove_var("HELIX_GRAPH_LIMIT_MAX");
        std::env::remove_var("HELIX_GRAPH_ALGORITHM_ITERATIONS_MAX");
        std::env::remove_var("HELIX_GRAPH_SHORTEST_PATH_VISITED_MAX");
        std::env::remove_var("HELIX_GRAPH_SYMBOL_RESOLVE_MAX");
        std::env::remove_var("HELIX_GRAPH_DELETE_TXN_CHUNK_SIZE");
    }

    struct EnvGuard(&'static str, Option<String>);
    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self(key, prev)
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.1 {
                Some(v) => std::env::set_var(self.0, v),
                None => std::env::remove_var(self.0),
            }
        }
    }

    fn setup_storage() -> (HelixGraphStorage, TempDir) {
        let tmp = TempDir::new().unwrap();
        let config = Config::new(16, 128, 768, 1);
        let storage = HelixGraphStorage::new(tmp.path().to_str().unwrap(), config).unwrap();
        (storage, tmp)
    }

    struct TestContext {
        _tmp: TempDir,
        graph: Arc<HelixGraphEngine>,
        collections: Arc<CollectionManager>,
        replication: Arc<ReplicationManager>,
    }

    fn setup_context() -> TestContext {
        let tmp = TempDir::new().unwrap();
        let graph_path = tmp.path().join("graph");
        let collections_path = tmp.path().join("data");
        let graph = Arc::new(
            HelixGraphEngine::new(HelixGraphEngineOpts {
                path: graph_path.display().to_string(),
                config: Config::default(),
            })
            .unwrap(),
        );
        let collections =
            Arc::new(CollectionManager::new(collections_path, Config::default()).unwrap());
        let replication =
            Arc::new(ReplicationManager::new(Arc::clone(&collections), Config::default()).unwrap());
        TestContext {
            _tmp: tmp,
            graph,
            collections,
            replication,
        }
    }

    fn make_input(ctx: &TestContext, body: Vec<u8>) -> HandlerInput {
        HandlerInput {
            request: Request {
                method: "POST".into(),
                headers: HashMap::new(),
                path: "/v1/graph/subgraph".into(),
                body,
            },
            graph: Arc::clone(&ctx.graph),
            collections: Arc::clone(&ctx.collections),
            replication: Arc::clone(&ctx.replication),
            path_params: HashMap::new(),
        }
    }

    fn make_node(collection: &str, name: &str, path: &str) -> NodeUpsert {
        NodeUpsert {
            id: deterministic_id::node_id(collection, "Symbol", name, path),
            label: "Symbol".into(),
            properties: std::collections::HashMap::from([
                ("name".into(), Value::String(name.into())),
                ("path".into(), Value::String(path.into())),
            ]),
        }
    }

    fn make_edge(
        collection: &str,
        from_name: &str,
        to_name: &str,
        from_path: &str,
        to_path: &str,
    ) -> EdgeUpsert {
        make_labeled_edge(collection, "CALLS", from_name, to_name, from_path, to_path)
    }

    fn make_labeled_edge(
        collection: &str,
        label: &str,
        from_name: &str,
        to_name: &str,
        from_path: &str,
        to_path: &str,
    ) -> EdgeUpsert {
        EdgeUpsert {
            id: deterministic_id::edge_id(
                collection, label, from_name, to_name, from_path, to_path,
            ),
            label: label.into(),
            from_node: deterministic_id::node_id(collection, "Symbol", from_name, from_path),
            to_node: deterministic_id::node_id(collection, "Symbol", to_name, to_path),
            properties: std::collections::HashMap::new(),
        }
    }

    fn make_edge_with_repo(
        collection: &str,
        from_name: &str,
        to_name: &str,
        from_path: &str,
        to_path: &str,
        repo: &str,
    ) -> EdgeUpsert {
        let mut edge = make_edge(collection, from_name, to_name, from_path, to_path);
        edge.properties
            .insert("repo".into(), Value::String(repo.into()));
        edge
    }

    fn make_edge_with_repo_and_paths(
        collection: &str,
        from_name: &str,
        to_name: &str,
        from_path: &str,
        to_path: &str,
        repo: &str,
    ) -> EdgeUpsert {
        let mut edge =
            make_edge_with_repo(collection, from_name, to_name, from_path, to_path, repo);
        edge.properties
            .insert("from_path".into(), Value::String(from_path.into()));
        edge.properties
            .insert("to_path".into(), Value::String(to_path.into()));
        edge
    }

    fn make_relationship_point(
        id: u128,
        caller_symbol: &str,
        callee_symbol: &str,
        caller_path: &str,
        callee_path: &str,
        edge_type: &str,
        repo: &str,
    ) -> NodeUpsert {
        NodeUpsert {
            id,
            label: "point".into(),
            properties: HashMap::from([
                ("caller_symbol".into(), Value::String(caller_symbol.into())),
                ("callee_symbol".into(), Value::String(callee_symbol.into())),
                ("caller_path".into(), Value::String(caller_path.into())),
                ("callee_path".into(), Value::String(callee_path.into())),
                ("edge_type".into(), Value::String(edge_type.into())),
                ("repo".into(), Value::String(repo.into())),
            ]),
        }
    }

    #[test]
    fn backfill_adjacency_from_points_rebuilds_lsm_traversal_from_graph_point_payload() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _backend = EnvGuard::set("HELIX_STORAGE_BACKEND", "lsm");
        let _in_mem = EnvGuard::set("HELIX_LSM_IN_MEMORY", "1");

        let ctx = setup_context();
        let collection = "repo_graph";
        let storage = ctx.collections.create_collection(collection).unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);

        let point = make_relationship_point(
            0xA11CE,
            "caller_fn",
            "target_fn",
            "src/caller.rs",
            "src/target.rs",
            "calls",
            "demo-repo",
        );
        storage
            .with_write_backend(|w| {
                storage.upsert_node_be(w, &point)?;
                Ok(())
            })
            .unwrap();

        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "collection": collection,
            "batch_size": 10,
        }))
        .unwrap();
        let input = make_input(&ctx, body);
        let mut response = Response::new();

        handle_backfill_adjacency_from_points(&input, &mut response).unwrap();
        assert_eq!(response.status, 200);
        let payload: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(payload["nodes_upserted"], 2);
        assert_eq!(payload["edges_upserted"], 1);
        assert_eq!(payload["complete"], true);

        let edge_id = deterministic_id::edge_id(
            collection,
            "CALLS",
            "caller_fn",
            "target_fn",
            "src/caller.rs",
            "src/target.rs",
        );
        storage
            .with_read_backend(|r| storage.get_edge_be(r, &edge_id).map(|_| ()))
            .unwrap();

        let callees_body = sonic_rs::to_vec(&sonic_rs::json!({
            "collection": collection,
            "symbol": "caller_fn",
            "path": "src/caller.rs",
        }))
        .unwrap();
        let callees_input = make_input(&ctx, callees_body);
        let mut callees_response = Response::new();

        handle_callees(&callees_input, &mut callees_response).unwrap();
        assert_eq!(callees_response.status, 200);
        let callees_payload: serde_json::Value =
            serde_json::from_slice(&callees_response.body).unwrap();
        assert_eq!(callees_payload["results"].as_array().unwrap().len(), 1);
        assert_eq!(
            callees_payload["results"][0]["properties"]["name"],
            "target_fn"
        );
        assert_eq!(
            callees_payload["results"][0]["properties"]["edge_type"],
            "CALLS"
        );

        let callers_body = sonic_rs::to_vec(&sonic_rs::json!({
            "collection": collection,
            "symbol": "target_fn",
            "path": "src/target.rs",
        }))
        .unwrap();
        let callers_input = make_input(&ctx, callers_body);
        let mut callers_response = Response::new();

        handle_callers(&callers_input, &mut callers_response).unwrap();
        assert_eq!(callers_response.status, 200);
        let callers_payload: serde_json::Value =
            serde_json::from_slice(&callers_response.body).unwrap();
        assert_eq!(callers_payload["results"].as_array().unwrap().len(), 1);
        assert_eq!(
            callers_payload["results"][0]["properties"]["name"],
            "caller_fn"
        );
    }

    #[test]
    fn backfill_adjacency_force_ignores_stale_complete_marker() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _backend = EnvGuard::set("HELIX_STORAGE_BACKEND", "lsm");
        let _in_mem = EnvGuard::set("HELIX_LSM_IN_MEMORY", "1");

        let ctx = setup_context();
        let collection = "repo_graph_stale_marker";
        let storage = ctx.collections.create_collection(collection).unwrap();
        let point = make_relationship_point(
            0xBEEF,
            "caller_fn",
            "target_fn",
            "src/caller.rs",
            "src/target.rs",
            "calls",
            "demo-repo",
        );
        storage
            .with_write_backend(|w| {
                storage.upsert_node_be(w, &point)?;
                storage
                    .backend
                    .put(
                        w,
                        Namespace::Metadata,
                        b"adj_backfill_from_points:complete",
                        b"1",
                    )
                    .unwrap();
                Ok(())
            })
            .unwrap();

        let normal_body = sonic_rs::to_vec(&sonic_rs::json!({
            "collection": collection,
            "batch_size": 10,
        }))
        .unwrap();
        let normal_input = make_input(&ctx, normal_body);
        let mut normal_response = Response::new();
        handle_backfill_adjacency_from_points(&normal_input, &mut normal_response).unwrap();
        let normal_payload: serde_json::Value =
            serde_json::from_slice(&normal_response.body).unwrap();
        assert_eq!(normal_payload["nodes_upserted"], 0);
        assert_eq!(normal_payload["edges_upserted"], 0);
        assert_eq!(normal_payload["complete"], true);

        let force_body = sonic_rs::to_vec(&sonic_rs::json!({
            "collection": collection,
            "batch_size": 10,
            "force": true,
        }))
        .unwrap();
        let force_input = make_input(&ctx, force_body);
        let mut force_response = Response::new();
        handle_backfill_adjacency_from_points(&force_input, &mut force_response).unwrap();
        assert_eq!(force_response.status, 200);
        let force_payload: serde_json::Value =
            serde_json::from_slice(&force_response.body).unwrap();
        assert_eq!(force_payload["nodes_upserted"], 2);
        assert_eq!(force_payload["edges_upserted"], 1);
        assert_eq!(force_payload["complete"], true);

        let edge_id = deterministic_id::edge_id(
            collection,
            "CALLS",
            "caller_fn",
            "target_fn",
            "src/caller.rs",
            "src/target.rs",
        );
        storage
            .with_read_backend(|r| storage.get_edge_be(r, &edge_id).map(|_| ()))
            .unwrap();
    }

    #[test]
    fn delete_by_paths_keeps_batch_semantics_with_chunked_write_phase() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_budget_env();
        std::env::set_var("HELIX_GRAPH_DELETE_TXN_CHUNK_SIZE", "1");

        let ctx = setup_context();
        let storage = ctx.collections.create_collection("repo").unwrap();
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        for node in [
            make_node("repo", "caller_a", "src/a.rs"),
            make_node("repo", "caller_b", "src/b.rs"),
            make_node("repo", "caller_other", "src/a.rs"),
            make_node("repo", "target", "src/common.rs"),
            make_node("repo", "target_other", "src/other.rs"),
        ] {
            storage.upsert_node(&mut txn, &node).unwrap();
        }
        let edge_a = make_edge_with_repo_and_paths(
            "repo",
            "caller_a",
            "target",
            "src/a.rs",
            "src/common.rs",
            "repo-a",
        );
        let edge_b = make_edge_with_repo_and_paths(
            "repo",
            "caller_b",
            "target",
            "src/b.rs",
            "src/common.rs",
            "repo-a",
        );
        let edge_other = make_edge_with_repo_and_paths(
            "repo",
            "caller_other",
            "target_other",
            "src/a.rs",
            "src/other.rs",
            "repo-b",
        );
        let edge_a_id = edge_a.id;
        let edge_b_id = edge_b.id;
        let edge_other_id = edge_other.id;
        for edge in [&edge_a, &edge_b, &edge_other] {
            storage.upsert_edge(&mut txn, edge).unwrap();
        }
        txn.commit().unwrap();

        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "collection": "repo",
            "paths": ["src/a.rs", "src/b.rs", "src/a.rs"],
            "repo": "repo-a",
            "delete_orphaned_nodes": false,
        }))
        .unwrap();
        let input = make_input(&ctx, body);
        let mut response = Response::new();

        handle_delete_by_paths(&input, &mut response).unwrap();
        assert_eq!(response.status, 200);

        let payload: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(payload["edges_deleted"], 2);
        assert_eq!(payload["nodes_deleted"], 0);
        assert_eq!(payload["paths"], 2);
        assert_eq!(payload["truncated"], false);

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        assert!(matches!(
            storage.get_edge(&txn, &edge_a_id),
            Err(GraphError::EdgeNotFound)
        ));
        assert!(matches!(
            storage.get_edge(&txn, &edge_b_id),
            Err(GraphError::EdgeNotFound)
        ));
        assert!(storage.get_edge(&txn, &edge_other_id).is_ok());

        clear_budget_env();
    }

    #[test]
    fn delete_by_paths_uses_lsm_backend_without_lmdb_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _backend = EnvGuard::set("HELIX_STORAGE_BACKEND", "lsm");
        let _in_mem = EnvGuard::set("HELIX_LSM_IN_MEMORY", "1");

        let ctx = setup_context();
        let storage = ctx.collections.create_collection("repo_lsm").unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);

        let nodes = [
            make_node("repo_lsm", "caller_a", "src/a.rs"),
            make_node("repo_lsm", "caller_b", "src/b.rs"),
            make_node("repo_lsm", "caller_other", "src/a.rs"),
            make_node("repo_lsm", "target", "src/common.rs"),
            make_node("repo_lsm", "target_other", "src/other.rs"),
        ];
        let edge_a = make_edge_with_repo_and_paths(
            "repo_lsm",
            "caller_a",
            "target",
            "src/a.rs",
            "src/common.rs",
            "repo-a",
        );
        let edge_b = make_edge_with_repo_and_paths(
            "repo_lsm",
            "caller_b",
            "target",
            "src/b.rs",
            "src/common.rs",
            "repo-a",
        );
        let edge_other = make_edge_with_repo_and_paths(
            "repo_lsm",
            "caller_other",
            "target_other",
            "src/a.rs",
            "src/other.rs",
            "repo-b",
        );
        let edge_a_id = edge_a.id;
        let edge_b_id = edge_b.id;
        let edge_other_id = edge_other.id;

        storage
            .with_write_backend(|w| {
                for node in &nodes {
                    storage.upsert_node_be(w, node)?;
                }
                for edge in [&edge_a, &edge_b, &edge_other] {
                    storage.upsert_edge_be(w, edge)?;
                }
                Ok(())
            })
            .unwrap();

        let single_body = sonic_rs::to_vec(&sonic_rs::json!({
            "collection": "repo_lsm",
            "path": "src/a.rs",
            "repo": "repo-a",
            "delete_orphaned_nodes": false,
        }))
        .unwrap();
        let single_input = make_input(&ctx, single_body);
        let mut single_response = Response::new();

        handle_delete_by_path(&single_input, &mut single_response).unwrap();
        assert_eq!(single_response.status, 200);
        let single_payload: serde_json::Value =
            serde_json::from_slice(&single_response.body).unwrap();
        assert_eq!(single_payload["edges_deleted"], 1);
        assert_eq!(single_payload["nodes_deleted"], 0);

        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "collection": "repo_lsm",
            "paths": ["src/b.rs"],
            "repo": "repo-a",
            "delete_orphaned_nodes": false,
        }))
        .unwrap();
        let input = make_input(&ctx, body);
        let mut response = Response::new();

        handle_delete_by_paths(&input, &mut response).unwrap();
        assert_eq!(response.status, 200);

        let payload: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(payload["edges_deleted"], 1);
        assert_eq!(payload["nodes_deleted"], 0);

        let r = storage.backend.begin_read().unwrap();
        assert!(matches!(
            storage.get_edge_be(&r, &edge_a_id),
            Err(GraphError::EdgeNotFound)
        ));
        assert!(matches!(
            storage.get_edge_be(&r, &edge_b_id),
            Err(GraphError::EdgeNotFound)
        ));
        assert!(storage.get_edge_be(&r, &edge_other_id).is_ok());
    }

    #[test]
    fn native_graph_query_routes_use_lsm_backend_without_lmdb_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _backend = EnvGuard::set("HELIX_STORAGE_BACKEND", "lsm");
        let _in_mem = EnvGuard::set("HELIX_LSM_IN_MEMORY", "1");

        let ctx = setup_context();
        let collection = "repo_graph_lsm";
        let storage = ctx.collections.create_collection(collection).unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);

        let mut nodes = vec![
            make_node(collection, "main", "src/main.rs"),
            make_node(collection, "helper", "src/helper.rs"),
            make_node(collection, "leaf", "src/leaf.rs"),
            make_node(collection, "imported", "src/imported.rs"),
        ];
        for node in &mut nodes {
            node.properties
                .insert("repo".into(), Value::String("repo-a".into()));
        }
        let mut edges = vec![
            make_labeled_edge(
                collection,
                "CALLS",
                "main",
                "helper",
                "src/main.rs",
                "src/helper.rs",
            ),
            make_labeled_edge(
                collection,
                "CALLS",
                "helper",
                "leaf",
                "src/helper.rs",
                "src/leaf.rs",
            ),
            make_labeled_edge(
                collection,
                "IMPORTS",
                "main",
                "imported",
                "src/main.rs",
                "src/imported.rs",
            ),
            make_labeled_edge(
                collection,
                "CALLS",
                "leaf",
                "main",
                "src/leaf.rs",
                "src/main.rs",
            ),
        ];
        for edge in &mut edges {
            edge.properties
                .insert("repo".into(), Value::String("repo-a".into()));
        }

        storage
            .with_write_backend(|w| {
                for node in &nodes {
                    storage.upsert_node_be(w, node)?;
                }
                for edge in &edges {
                    storage.upsert_edge_be(w, edge)?;
                }
                Ok(())
            })
            .unwrap();
        storage
            .create_payload_index("repo", PayloadIndexSchema::Keyword)
            .unwrap();

        let run = |handler: fn(&HandlerInput, &mut Response) -> Result<(), GraphError>,
                   symbol: &str,
                   depth: usize| {
            let body = sonic_rs::to_vec(&sonic_rs::json!({
                "collection": collection,
                "symbol": symbol,
                "depth": depth,
                "limit": 20,
            }))
            .unwrap();
            let input = make_input(&ctx, body);
            let mut response = Response::new();
            handler(&input, &mut response).unwrap();
            assert_eq!(response.status, 200);
            serde_json::from_slice::<serde_json::Value>(&response.body).unwrap()
        };

        let dependencies = run(handle_dependencies, "main", 2);
        let dependency_names: HashSet<String> = dependencies["results"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|node| node["properties"]["name"].as_str().map(str::to_string))
            .collect();
        assert!(dependency_names.contains("helper"));
        assert!(dependency_names.contains("leaf"));
        assert!(dependency_names.contains("imported"));

        let impact = run(handle_impact, "helper", 2);
        let impact_names: HashSet<String> = impact["results"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|node| node["properties"]["name"].as_str().map(str::to_string))
            .collect();
        assert!(impact_names.contains("main"));

        let callers = run(handle_transitive_callers, "helper", 1);
        assert!(callers["results"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| { node["properties"]["name"].as_str() == Some("main") }));

        let cycles = run(handle_cycles, "main", 10);
        assert!(!cycles["cycles"].as_array().unwrap().is_empty());

        let definition = run(handle_definition, "main", 1);
        assert_eq!(definition["result"]["properties"]["name"], "main");

        let subgraph_body = sonic_rs::to_vec(&sonic_rs::json!({
            "collection": collection,
            "repo": "repo-a",
            "limit": 20,
        }))
        .unwrap();
        let subgraph_input = make_input(&ctx, subgraph_body);
        let mut subgraph_response = Response::new();
        handle_subgraph(&subgraph_input, &mut subgraph_response).unwrap();
        assert_eq!(subgraph_response.status, 200);
        let subgraph: serde_json::Value = serde_json::from_slice(&subgraph_response.body).unwrap();
        assert!(!subgraph["nodes"].as_array().unwrap().is_empty());
        assert!(!subgraph["edges"].as_array().unwrap().is_empty());

        let distinct_body = sonic_rs::to_vec(&sonic_rs::json!({
            "collection": collection,
            "field": "repo",
            "with_counts": true,
        }))
        .unwrap();
        let distinct_input = make_input(&ctx, distinct_body);
        let mut distinct_response = Response::new();
        handle_distinct_values(&distinct_input, &mut distinct_response).unwrap();
        assert_eq!(distinct_response.status, 200);
        let distinct: serde_json::Value = serde_json::from_slice(&distinct_response.body).unwrap();
        assert_eq!(distinct["values"].as_array().unwrap().len(), 1);
        assert_eq!(distinct["values"][0], "repo-a");
        assert_eq!(distinct["counts"][0], 4);
    }

    #[test]
    fn subgraph_repo_filter_scopes_sampled_nodes_and_edges() {
        let ctx = setup_context();
        let storage = ctx.collections.create_collection("repo").unwrap();
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        for node in [
            make_node("repo", "caller_a", "src/a.py"),
            make_node("repo", "target_a", "src/target.py"),
            make_node("repo", "caller_b", "src/b.py"),
        ] {
            storage.upsert_node(&mut txn, &node).unwrap();
        }
        storage
            .upsert_edge(
                &mut txn,
                &make_edge_with_repo(
                    "repo",
                    "caller_a",
                    "target_a",
                    "src/a.py",
                    "src/target.py",
                    "repo-a",
                ),
            )
            .unwrap();
        storage
            .upsert_edge(
                &mut txn,
                &make_edge_with_repo(
                    "repo",
                    "caller_b",
                    "target_a",
                    "src/b.py",
                    "src/target.py",
                    "repo-b",
                ),
            )
            .unwrap();
        txn.commit().unwrap();

        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "collection": "repo",
            "repo": "repo-a",
            "limit": 10
        }))
        .unwrap();
        let input = make_input(&ctx, body);
        let mut response = Response::new();

        handle_subgraph(&input, &mut response).unwrap();
        assert_eq!(response.status, 200);

        let payload: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        let nodes = payload["nodes"].as_array().unwrap();
        let names: HashSet<String> = nodes
            .iter()
            .filter_map(|node| node["name"].as_str().map(str::to_string))
            .collect();
        assert!(names.contains("caller_a"));
        assert!(names.contains("target_a"));
        assert!(!names.contains("caller_b"));
        assert_eq!(payload["edges"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn graph_budget_helpers_clamp_to_env_maxima() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_budget_env();
        std::env::set_var("HELIX_GRAPH_DEPTH_MAX", "3");
        std::env::set_var("HELIX_GRAPH_LIMIT_MAX", "5");
        std::env::set_var("HELIX_GRAPH_ALGORITHM_ITERATIONS_MAX", "7");
        std::env::set_var("HELIX_GRAPH_SHORTEST_PATH_VISITED_MAX", "11");

        assert_eq!(clamp_graph_depth(0), 1);
        assert_eq!(clamp_graph_depth(99), 3);
        assert_eq!(clamp_graph_limit(0), 1);
        assert_eq!(clamp_graph_limit(99), 5);
        assert_eq!(clamp_graph_iterations(0), 1);
        assert_eq!(clamp_graph_iterations(99), 7);
        assert_eq!(graph_shortest_path_visited_max(), 11);

        clear_budget_env();
    }

    #[test]
    fn pathless_symbol_resolution_returns_all_matching_symbol_nodes() {
        let (storage, _tmp) = setup_storage();
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        for node in [
            make_node("repo", "target", ""),
            make_node("repo", "target", "src/target.py"),
            make_node("repo", "caller_empty", "src/a.py"),
            make_node("repo", "caller_real", "src/b.py"),
        ] {
            storage.upsert_node(&mut txn, &node).unwrap();
        }
        storage
            .upsert_edge(
                &mut txn,
                &make_edge("repo", "caller_empty", "target", "src/a.py", ""),
            )
            .unwrap();
        storage
            .upsert_edge(
                &mut txn,
                &make_edge("repo", "caller_real", "target", "src/b.py", "src/target.py"),
            )
            .unwrap();
        txn.commit().unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let query = GraphQuery {
            collection: "repo".into(),
            symbol: "target".into(),
            path: None,
            repo: None,
            depth: 1,
            limit: 100,
        };

        let starts = resolve_symbol_ids(&query, &storage, &txn);
        assert!(starts.contains(&deterministic_id::node_id("repo", "Symbol", "target", "")));
        assert!(starts.contains(&deterministic_id::node_id(
            "repo",
            "Symbol",
            "target",
            "src/target.py"
        )));

        let results = bfs_reverse(&storage, &txn, &starts, &["CALLS"], 1, 100, None).unwrap();
        let names: HashSet<String> = results
            .into_iter()
            .filter_map(|r| match r.node.properties.get("name") {
                Some(Value::String(name)) => Some(name.clone()),
                _ => None,
            })
            .collect();
        assert!(names.contains("caller_empty"));
        assert!(names.contains("caller_real"));
    }

    #[test]
    fn path_specific_symbol_resolution_stays_exact() {
        let (storage, _tmp) = setup_storage();
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        for node in [
            make_node("repo", "target", ""),
            make_node("repo", "target", "src/target.py"),
        ] {
            storage.upsert_node(&mut txn, &node).unwrap();
        }
        txn.commit().unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let query = GraphQuery {
            collection: "repo".into(),
            symbol: "target".into(),
            path: Some("src/target.py".into()),
            repo: None,
            depth: 1,
            limit: 100,
        };

        let starts = resolve_symbol_ids(&query, &storage, &txn);
        assert_eq!(
            starts,
            vec![deterministic_id::node_id(
                "repo",
                "Symbol",
                "target",
                "src/target.py"
            )]
        );
    }

    #[test]
    fn graph_budget_helpers_ignore_invalid_env_values() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_budget_env();
        std::env::set_var("HELIX_GRAPH_DEPTH_MAX", "0");
        std::env::set_var("HELIX_GRAPH_LIMIT_MAX", "invalid");
        std::env::set_var("HELIX_GRAPH_ALGORITHM_ITERATIONS_MAX", "0");
        std::env::set_var("HELIX_GRAPH_SHORTEST_PATH_VISITED_MAX", "invalid");

        assert_eq!(clamp_graph_depth(99), DEFAULT_GRAPH_DEPTH_MAX);
        assert_eq!(clamp_graph_limit(999), DEFAULT_GRAPH_LIMIT_MAX);
        assert_eq!(
            clamp_graph_iterations(999),
            DEFAULT_GRAPH_ALGORITHM_ITERATIONS_MAX
        );
        assert_eq!(
            graph_shortest_path_visited_max(),
            DEFAULT_GRAPH_SHORTEST_PATH_VISITED_MAX
        );

        clear_budget_env();
    }
}
