use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use std::time::{Duration, Instant};

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
use crate::helix_engine::storage_core::backend_any::{AnyRead, AnyWrite};
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
const DEFAULT_GRAPH_CYCLES_VISITED_MAX: usize = 100_000;
const DEFAULT_GRAPH_CYCLES_TIMEOUT_MS: usize = 10_000;
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

fn graph_cycles_visited_max() -> usize {
    env_usize(
        "HELIX_GRAPH_CYCLES_VISITED_MAX",
        DEFAULT_GRAPH_CYCLES_VISITED_MAX,
    )
}

fn graph_cycles_timeout() -> std::time::Duration {
    std::time::Duration::from_millis(env_usize(
        "HELIX_GRAPH_CYCLES_TIMEOUT_MS",
        DEFAULT_GRAPH_CYCLES_TIMEOUT_MS,
    ) as u64)
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

/// An end of the edge is in `paths`; with `from_only`, its start is.
fn graph_edge_matches_any_path(edge: &Edge, paths: &HashSet<String>, from_only: bool) -> bool {
    matches!(edge.properties.get("from_path"), Some(Value::String(p)) if paths.contains(p))
        || (!from_only
            && matches!(edge.properties.get("to_path"), Some(Value::String(p)) if paths.contains(p)))
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

/// Drops the edges of `chunk` on `w` and returns how many went. An edge that
/// does not drop is skipped, as it always was on LSM, and listed in `failed`.
fn drop_edge_chunk_be(
    storage: &HelixGraphStorage,
    w: &mut AnyWrite<'_>,
    chunk: &[u128],
    failed: &mut Vec<u128>,
) -> usize {
    let mut dropped = 0usize;
    for edge_id in chunk {
        match storage.drop_edge_be(w, edge_id) {
            Ok(()) => dropped += 1,
            Err(_) => failed.push(*edge_id),
        }
    }
    dropped
}

/// One LSM write batch for a chunk of a delete: its edges, and the nodes they
/// leave without an edge. A request cut short between two such batches leaves
/// no node that only an edge it has already deleted could name, so a repeat
/// finds everything that is left. The nodes are worked out in the batch, once
/// its edge drops are staged, on what other writers have committed by then
/// ([`HelixGraphStorage::nodes_orphaned_by_dropping_be`]). Returns the edges
/// and the nodes dropped.
///
/// If the nodes cannot be worked out, the batch is dropped unwritten and the
/// request fails: nothing of the chunk is stored and a repeat finds all of
/// it. If their drop fails, the batch is dropped unwritten and the chunk goes
/// the way it did before: its edges in one batch, then its ends through
/// [`delete_orphaned_nodes_in_write_chunks`], which skips a failing node.
fn delete_edges_with_their_orphans(
    storage: &HelixGraphStorage,
    chunk: &[u128],
) -> Result<(usize, usize), GraphError> {
    let r = storage
        .backend
        .begin_read()
        .map_err(|e| GraphError::New(e.to_string()))?;
    let keys: Vec<Vec<u8>> = chunk.iter().map(|id| id.to_be_bytes().to_vec()).collect();
    let rows = storage
        .backend
        .get_many(&r, Namespace::Edges, &keys)
        .map_err(|e| GraphError::New(e.to_string()))?;
    drop(r);
    // Each edge of the chunk that is there, with its two ends.
    let mut dropped: BTreeMap<u128, (u128, u128)> = chunk
        .iter()
        .zip(&rows)
        .filter_map(|(id, row)| {
            let edge = SerializedEdge::decode_edge(row.as_deref()?, *id).ok()?;
            Some((*id, (edge.from_node, edge.to_node)))
        })
        .collect();
    let mut ends: Vec<u128> = dropped
        .values()
        .flat_map(|(from, to)| [*from, *to])
        .collect();
    ends.sort_unstable();
    ends.dedup();

    let mut nodes_failed = false;
    let batched = storage.with_write_backend(|w| {
        let mut failed = Vec::new();
        let edges = drop_edge_chunk_be(storage, w, chunk, &mut failed);
        // An edge that did not drop is not one of the dropped. Where it is
        // still there its rows keep its ends; where another writer has
        // dropped it since the read above, they no longer do.
        for edge_id in &failed {
            dropped.remove(edge_id);
        }
        // A check that cannot be made fails the batch and the request: the
        // fallback below needs the same reads after it has dropped the edges.
        let orphans = storage.nodes_orphaned_by_dropping_be(w, &ends, &dropped)?;
        let nodes = storage.drop_node_rows_be(w, &orphans).map_err(|error| {
            nodes_failed = true;
            // Kept out of the wrapper's fatal-error handling: the chunk is
            // redone below, and the per-node drops decide.
            GraphError::New(error.to_string())
        })?;
        Ok((edges, nodes))
    });
    match batched {
        Err(error) if nodes_failed => tracing::warn!(
            collection = storage.collection_display_name(),
            edges = chunk.len(),
            ends = ends.len(),
            error = %error,
            "a chunk's edges and orphaned nodes did not drop in one batch; dropping the edges, then the nodes"
        ),
        batched => return batched,
    }
    let edges = storage
        .with_write_backend(|w| Ok(drop_edge_chunk_be(storage, w, chunk, &mut Vec::new())))?;
    let nodes = delete_orphaned_nodes_in_write_chunks(storage, &ends.into_iter().collect())?;
    Ok((edges, nodes))
}

/// Kill switch for dropping a delete chunk's orphaned nodes in the batch that
/// drops its edges ([`delete_edges_with_their_orphans`]). Enabled by default.
/// `HELIX_GRAPH_DELETE_NODES_WITH_EDGES=0` (or `false`/`off`/`no`) drops every
/// edge chunk first again and then the nodes, in batches of their own.
fn graph_delete_nodes_with_edges() -> bool {
    !matches!(
        std::env::var("HELIX_GRAPH_DELETE_NODES_WITH_EDGES")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref(),
        Some("0") | Some("false") | Some("off") | Some("no")
    )
}

/// Drops `edge_ids` in write chunks and returns how many edges went and how
/// many nodes with them. With `orphans`, the ends the request resolved, the
/// nodes its edges leave without one go too: on LSM in each chunk's own write
/// batch ([`delete_edges_with_their_orphans`]); on LMDB, and on LSM with
/// [`graph_delete_nodes_with_edges`] off, after the edges.
fn delete_edges_in_write_chunks(
    storage: &HelixGraphStorage,
    edge_ids: &[u128],
    orphans: Option<&HashSet<u128>>,
) -> Result<(usize, usize), GraphError> {
    let lsm = storage.backend.kind() == BackendKind::Lsm;
    // Read once: a request goes one way for all its chunks.
    let nodes_with_edges = lsm && orphans.is_some() && graph_delete_nodes_with_edges();
    let (mut edges_deleted, mut nodes_deleted) = (0usize, 0usize);
    for chunk in edge_ids.chunks(graph_delete_txn_chunk_size()) {
        if nodes_with_edges {
            let (edges, nodes) = delete_edges_with_their_orphans(storage, chunk)?;
            edges_deleted += edges;
            nodes_deleted += nodes;
        } else if lsm {
            edges_deleted += storage.with_write_backend(|w| {
                Ok(drop_edge_chunk_be(storage, w, chunk, &mut Vec::new()))
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
    if let (false, Some(orphans)) = (nodes_with_edges, orphans) {
        nodes_deleted += delete_orphaned_nodes_in_write_chunks(storage, orphans)?;
    }
    Ok((edges_deleted, nodes_deleted))
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
            let removable = storage.nodes_without_adjacency_be(&r, chunk)?;
            // One batched drop for the chunk. It reads the adjacency again and
            // leaves a node that got an edge since the check, with that edge:
            // nothing here deletes an edge. If it fails, its write batch is
            // dropped unwritten (the wrapper never commits after an error) and
            // the nodes are dropped one at a time in a fresh batch, a failing
            // node skipped, as they were before the drop was batched.
            let mut batch_failed = false;
            let batched = storage.with_write_backend(|w| {
                storage
                    .drop_orphan_nodes_be(w, &removable)
                    .map_err(|error| {
                        batch_failed = true;
                        // Kept out of the wrapper's fatal-error handling: the
                        // per-node drops decide, and they never reached it.
                        GraphError::New(error.to_string())
                    })
            });
            nodes_deleted += match batched {
                Err(error) if batch_failed => {
                    tracing::warn!(
                        collection = storage.collection_display_name(),
                        nodes = removable.len(),
                        error = %error,
                        "batched orphan node drop failed; dropping one node at a time"
                    );
                    metrics::counter!("helix_graph_delete_batch_fallback_total", "step" => "drop")
                        .increment(1);
                    storage.with_write_backend(|w| {
                        let mut chunk_deleted = 0usize;
                        for node_id in &removable {
                            match storage.drop_orphan_nodes_be(w, std::slice::from_ref(node_id)) {
                                Ok(dropped) => chunk_deleted += dropped,
                                Err(_) => {}
                            }
                        }
                        Ok(chunk_deleted)
                    })?
                }
                batched => batched?,
            };
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

/// Answers a failed `get_collection` of a graph route. An absent collection
/// keeps the JSON 404 these routes always gave: the client reads its `error` as "no
/// native graph to keep in step", and a 404 without one as "this Helion has
/// no such route". Any other failure answers what the router answers for it
/// (`collection_open_err` of the point routes). A 404 for a collection that
/// was only not opened told the client that the clear or the rebuild it asked for
/// was done, and made it note in a read's reply that the graph collection
/// does not exist.
fn collection_open_err(
    input: &HandlerInput,
    response: &mut Response,
    error: GraphError,
) -> Result<(), GraphError> {
    super::qdrant::collection_open_err_as(input, response, error, json_error)
}

/// The gate point upserts, deletes and payload writes hold on a collection's
/// resolved name. A graph write holds it from its first read to its last
/// commit: an LSM write batch is ordered against no other, so without it a
/// row write can commit between a rebuild's read of the rows and its write,
/// and two graph writes can each work out their orphans on edges the other
/// is changing. It is taken before the collection is opened, as an upsert
/// does: a handle held through the wait would keep the collection from
/// closing for that long.
fn graph_gate(input: &HandlerInput, collection: &str) -> Arc<Mutex<()>> {
    let resolved = input.collections.resolve_alias(collection);
    input.collections.point_mutation_gate(&resolved)
}

/// How long a graph write waits for the gate before it answers busy. The
/// wait is paid once per submit attempt: the gateway repeats a busy answer
/// (`HELIX_SUBMIT_MAX_ATTEMPTS`, 3, with 250 and 500 ms between), so the
/// caller hears busy after three waits and two pauses, 24.75 s. That is
/// under the 30 s the client gives its path-index and repair steps (its other graph
/// writes get 120 s), so the busy answer reaches the caller that sent the
/// write.
fn graph_gate_wait() -> Duration {
    let millis = std::env::var("HELIX_GRAPH_GATE_WAIT_MS")
        .ok()
        .and_then(|value| value.parse().ok());
    Duration::from_millis(millis.unwrap_or(8_000))
}

/// `None` when the gate stayed held for the whole wait.
// shortcut: polls, as a std Mutex has no timed lock, so writers queued on
// `lock()` can keep taking the gate ahead of this until it answers busy;
// give the gate a timed lock if `helix_graph_gate_busy_total` shows that.
fn lock_graph_gate(gate: &Mutex<()>) -> Option<MutexGuard<'_, ()>> {
    let deadline = Instant::now() + graph_gate_wait();
    loop {
        match gate.try_lock() {
            Ok(held) => return Some(held),
            // The point writes that take this gate report its poison.
            Err(TryLockError::Poisoned(poisoned)) => return Some(poisoned.into_inner()),
            Err(TryLockError::WouldBlock) if Instant::now() >= deadline => return None,
            Err(TryLockError::WouldBlock) => std::thread::sleep(Duration::from_millis(5)),
        }
    }
}

/// The answer of a graph write that never got the gate: it wrote nothing, and
/// the same request can be sent again.
fn graph_gate_busy(
    response: &mut Response,
    collection: &str,
    route: &'static str,
) -> Result<(), GraphError> {
    metrics::counter!("helix_graph_gate_busy_total", "route" => route).increment(1);
    tracing::warn!(
        collection = %collection,
        route,
        "graph write gave up waiting for the collection's point-mutation gate"
    );
    response
        .headers
        .insert("Retry-After".to_string(), "1".to_string());
    json_error(
        response,
        503,
        "collection is busy with another write; retry",
    )
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
/// by symbol name and returns all bounded matches. The client can have multiple
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
            // so the client's backend can extract them for filtering.
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
        Err(e) => return collection_open_err(input, response, e),
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
        Err(e) => return collection_open_err(input, response, e),
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
        Err(e) => return collection_open_err(input, response, e),
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
        Err(e) => return collection_open_err(input, response, e),
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
        Err(e) => return collection_open_err(input, response, e),
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
        Err(e) => return collection_open_err(input, response, e),
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
        Err(e) => return collection_open_err(input, response, e),
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
        Err(e) => return collection_open_err(input, response, e),
    };
    let repo = q.repo.as_deref();
    let limit = clamp_graph_limit(q.limit);
    let max_visited = graph_cycles_visited_max();
    let deadline = t0 + graph_cycles_timeout();
    storage.with_read_backend(|r| {
        let start = resolve_symbol_id_be(&q, &storage, r);
        let search = detect_cycles_be(
            &storage,
            r,
            start,
            &["CALLS"],
            limit,
            repo,
            max_visited,
            Some(deadline),
        )?;
        let cycle_strs: Vec<Vec<String>> = search
            .cycles
            .into_iter()
            .map(|c| c.into_iter().map(|id| format!("{:032x}", id)).collect())
            .collect();
        json_response(
            response,
            200,
            &sonic_rs::json!({"cycles": cycle_strs, "truncated": search.truncated}),
        )
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
        Err(e) => return collection_open_err(input, response, e),
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
        Err(e) => return collection_open_err(input, response, e),
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
        Err(e) => return collection_open_err(input, response, e),
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
        Err(e) => return collection_open_err(input, response, e),
    };
    if !(q.damping > 0.0 && q.damping < 1.0) {
        return json_error(response, 400, "damping must be in the open interval (0, 1)");
    }
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
            .take(limit)
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
        Err(e) => return collection_open_err(input, response, e),
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

        let total = communities.len();
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
            &sonic_rs::json!({"communities": results, "total": total}),
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
        Err(e) => return collection_open_err(input, response, e),
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
        Err(e) => return collection_open_err(input, response, e),
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
    /// If true, delete only the edges that start in `paths` and keep the
    /// edges other files have into them. Those are derived from the other
    /// files' relationship points, which a rewrite of these paths leaves in
    /// place, and only re-deriving those files would put them back. The
    /// response echoes the flag, so a caller can tell a Helion that knows it.
    #[serde(default)]
    from_only: bool,
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
    let gate = graph_gate(input, &req.collection);
    let Some(_held) = lock_graph_gate(&gate) else {
        return graph_gate_busy(response, &collection, "/v1/graph/backfill_edge_path_index");
    };
    let storage = match input.collections.get_collection(&req.collection) {
        Ok(s) => s,
        Err(e) => return collection_open_err(input, response, e),
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
    /// For an operator: write every edge this call derives, also one that
    /// is stored as derived with its keys, as every call did before.
    #[serde(default)]
    rewrite_unchanged_edges: bool,
}

fn default_adjacency_backfill_batch() -> usize {
    1000
}

/// Rebuild native graph adjacency from relationship points already stored in a
/// `_graph` collection. Relationship points carry `caller_symbol`,
/// `callee_symbol`, `caller_path`, `callee_path`, and `edge_type`; this
/// endpoint turns each one into the caller/callee Symbol nodes and the
/// deterministic edge that the native graph traversal endpoints read.
///
/// `edges_upserted` counts the edges derived. On LSM one that is stored
/// exactly as derived, with its adjacency and path keys, is left, and
/// `edges_unchanged` says how many were: the rest were written. `force`
/// starts the pass over and nothing more; a pass over a graph that needs no
/// repair still walks to its end, a cursor each call, and sets its mark.
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
    let gate = graph_gate(input, &req.collection);
    let Some(_held) = lock_graph_gate(&gate) else {
        return graph_gate_busy(
            response,
            &collection,
            "/v1/graph/backfill_adjacency_from_points",
        );
    };
    let storage = match input.collections.get_collection(&req.collection) {
        Ok(s) => s,
        Err(e) => return collection_open_err(input, response, e),
    };

    let batch_size = req.batch_size.clamp(1, 10_000);
    let (nodes_upserted, edges_upserted, complete, cursor, edges_unchanged) = storage
        .backfill_adjacency_from_points_batch(
            &req.collection,
            batch_size,
            req.force,
            req.rewrite_unchanged_edges,
        )?;
    if nodes_upserted > 0 || edges_upserted > 0 {
        algorithms::invalidate_algorithm_cache(&req.collection);
    }

    json_response(
        response,
        200,
        &sonic_rs::json!({
            "nodes_upserted": nodes_upserted,
            "edges_upserted": edges_upserted,
            "edges_unchanged": edges_unchanged,
            "complete": complete,
            "cursor": cursor.map(|c| c.to_string()),
        }),
    )?;
    tracing::debug!(
        collection = %collection,
        edges_upserted,
        edges_unchanged,
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
    /// If true, the edges that start in `paths` and that no relationship
    /// point derives any more are dropped in the write that upserts the
    /// derived ones, with the nodes they leave without an edge: no
    /// `delete_by_paths` first, and no moment at which the paths have no
    /// outgoing edge. The response says `replaced_outgoing: true` when that
    /// was done; a Helion that does not know the field rebuilds as before
    /// and answers without it. A request with no path writes nothing and
    /// answers the same way, so a caller can ask first: `true` only where
    /// the next request's edges would be replaced.
    #[serde(default)]
    replace_outgoing: bool,
}

/// Incrementally rebuild native graph adjacency for a delta of changed source
/// paths, deriving edges from the `_graph` relationship points already stored on
/// S3. The client calls this after re-indexing changed files — paired with
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
///
/// `edges_upserted` and `chunk_edges_upserted` count what was derived. On
/// LSM an edge stored exactly as derived, with its adjacency and path keys,
/// is left as it is, and `edges_unchanged` and `chunk_edges_unchanged` say
/// how many of them were: the rest were written, an edge that two rows of
/// the request derive once, as the later derives it.
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
    // The graph collection's gate only: the chunk edges written into
    // `base_collection` below stay unordered against its point writes.
    let gate = graph_gate(input, &req.collection);
    let Some(_held) = lock_graph_gate(&gate) else {
        return graph_gate_busy(
            response,
            &collection,
            "/v1/graph/rebuild_adjacency_for_paths",
        );
    };
    let storage = match input.collections.get_collection(&req.collection) {
        Ok(s) => s,
        Err(e) => return collection_open_err(input, response, e),
    };

    let rebuilt =
        storage.rebuild_adjacency_for_paths(&req.collection, &req.paths, req.replace_outgoing)?;
    let (nodes_upserted, edges_upserted) = (rebuilt.nodes_upserted, rebuilt.edges_upserted);
    let (edges_deleted, nodes_deleted) = rebuilt.replaced.unwrap_or((0, 0));
    if nodes_upserted > 0 || edges_upserted > 0 || edges_deleted > 0 || nodes_deleted > 0 {
        algorithms::invalidate_algorithm_cache(&req.collection);
    }

    let (mut chunk_edges_upserted, mut chunk_edges_unchanged) = (0usize, 0usize);
    let mut warning: Option<String> = None;
    if let Some(base_collection) = &req.base_collection {
        match input.collections.get_collection(base_collection) {
            Ok(base_storage) => {
                (chunk_edges_upserted, chunk_edges_unchanged) =
                    apply_chunk_edges(&base_storage, base_collection, &rebuilt.chunk_candidates)?;
                if chunk_edges_upserted > 0 {
                    algorithms::invalidate_algorithm_cache(base_collection);
                }
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
            "edges_unchanged": rebuilt.edges_unchanged,
            "paths": req.paths.len(),
            "chunk_edges_upserted": chunk_edges_upserted,
            "chunk_edges_unchanged": chunk_edges_unchanged,
            "warning": warning,
            "replaced_outgoing": rebuilt.replaced.is_some(),
            "edges_deleted": edges_deleted,
            "nodes_deleted": nodes_deleted,
        }),
    )?;
    tracing::debug!(
        collection = %collection,
        paths = req.paths.len(),
        nodes_upserted,
        edges_upserted,
        edges_unchanged = rebuilt.edges_unchanged,
        edges_deleted,
        nodes_deleted,
        chunk_edges_upserted,
        chunk_edges_unchanged,
        elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
        "handle_rebuild_adjacency_for_paths complete"
    );
    Ok(())
}

/// Resolve a chunk-edge endpoint string to the point's node id. The client renders
/// numeric Qdrant point ids as decimal strings, so decimal parses first; a
/// hex/hash fallback covers UUID-style string point ids, matching how the
/// point's "point" node id was derived on ingest (`point_id_to_u128`).
///
/// The client stores the callee's id as it read it back from a scroll: the 32 hex
/// digits every point id is rendered in. That form is read as hex first,
/// since one made only of the digits 0-9 also parses as a decimal number,
/// which names another point or none.
fn chunk_edge_point_id(raw: &str) -> u128 {
    if raw.len() == 32 {
        if let Ok(id) = u128::from_str_radix(raw, 16) {
            return id;
        }
    }
    u128::from_str_radix(raw, 10).unwrap_or_else(|_| string_id_to_u128(raw))
}

/// Apply chunk-edge candidates derived from `_graph` relationship points to
/// the base collection, where every vector point already has a "point" node
/// whose id is the point id. Both endpoints must already exist — this never
/// creates nodes, so a caller/callee id the client couldn't resolve (or one
/// belonging to a different, contaminated collection) is silently skipped
/// instead of creating a dangling edge. All upserts run in one write batch.
///
/// Answers the candidates whose two points exist and, of them, the edges
/// that were stored already, with their adjacency keys, and were left: a
/// chunk edge's properties are made of what its id is made of, so a stored
/// one is stored as derived. The edges whose points exist are read and
/// staged together ([`HelixGraphStorage::upsert_derived_edges_be`]), after
/// the checks of their points. A request whose chunk edges are all stored
/// writes no batch here.
fn apply_chunk_edges(
    storage: &HelixGraphStorage,
    base_collection: &str,
    candidates: &[ChunkEdgeCandidate],
) -> Result<(usize, usize), GraphError> {
    if candidates.is_empty() {
        return Ok((0, 0));
    }
    let leave_as_derived = storage.leaves_edges_stored_as_derived();
    storage.with_write_backend(|w| {
        let mut upserts: Vec<EdgeUpsert> = Vec::new();
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
            let upsert = EdgeUpsert {
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
            };
            upserted += 1;
            if leave_as_derived {
                upserts.push(upsert);
            } else {
                storage.upsert_edge_be(w, &upsert)?;
            }
        }
        let unchanged = if leave_as_derived {
            storage.upsert_derived_edges_be(w, &upserts)?
        } else {
            0
        };
        Ok((upserted, unchanged))
    })
}

/// Delete all edges where `from_path` or `to_path` matches the given path.
/// Used by the client's incremental re-indexing to wipe stale edges before re-ingesting.
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
    let gate = graph_gate(input, &req.collection);
    let Some(_held) = lock_graph_gate(&gate) else {
        return graph_gate_busy(response, &req.collection, ROUTE);
    };
    let storage = match input.collections.get_collection(&req.collection) {
        Ok(s) => s,
        Err(e) => return collection_open_err(input, response, e),
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

    // One phase: the nodes go with the edges that orphan them.
    let edge_write_start = Instant::now();
    let (edges_deleted, nodes_deleted) = delete_edges_in_write_chunks(
        &storage,
        &edge_ids_to_delete,
        req.delete_orphaned_nodes.then_some(&affected_nodes),
    )?;
    observe_graph_delete_phase(ROUTE, "edge_write", edge_write_start.elapsed());
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
/// provided batch. This is the plural form used by the client to collapse a burst of
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
    let gate = graph_gate(input, &req.collection);
    let Some(_held) = lock_graph_gate(&gate) else {
        return graph_gate_busy(response, &req.collection, ROUTE);
    };
    let storage = match input.collections.get_collection(&req.collection) {
        Ok(s) => s,
        Err(e) => return collection_open_err(input, response, e),
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
                "from_only": req.from_only,
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
            for edge_id in storage.edge_ids_for_path_ends_be(&r, path, remaining, req.from_only)? {
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
                                && graph_edge_matches_any_path(&edge, &path_set, req.from_only) =>
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
                for edge_id in
                    storage.edge_ids_for_path_ends(rtxn, path, remaining, req.from_only)?
                {
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
                    if graph_edge_matches_any_path(&edge, &path_set, req.from_only) {
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
                "from_only": req.from_only,
            }),
        );
    }

    // One phase: the nodes go with the edges that orphan them.
    let edge_write_start = Instant::now();
    let (edges_deleted, nodes_deleted) = delete_edges_in_write_chunks(
        &storage,
        &edge_ids_to_delete,
        req.delete_orphaned_nodes.then_some(&affected_nodes),
    )?;
    observe_graph_delete_phase(ROUTE, "edge_write", edge_write_start.elapsed());
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
            "from_only": req.from_only,
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
        Err(e) => return collection_open_err(input, response, e),
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
            algorithms::cached_pagerank(
                &q.collection,
                &storage,
                r,
                &labels,
                iterations,
                0.85,
                repo,
            )?
            .0
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
        Err(e) => return collection_open_err(input, response, e),
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
#[path = "graph_open_error_tests.rs"]
mod open_error_tests;

#[cfg(test)]
#[path = "graph_equal_edge_tests.rs"]
mod equal_edge_tests;

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
    use crate::protocol::items::SerializedNode;
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
        std::env::remove_var("HELIX_GRAPH_DELETE_NODES_WITH_EDGES");
    }

    /// Sets an env knob for a scope; the previous value comes back on drop,
    /// also when the test panics.
    pub(super) struct EnvVarGuard {
        name: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        pub(super) fn set(name: &'static str, value: &str) -> Self {
            let previous = std::env::var_os(name);
            std::env::set_var(name, value);
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
        let tmp = TempDir::new().unwrap();
        let config = Config::new(16, 128, 768, 1);
        let storage = HelixGraphStorage::new(tmp.path().to_str().unwrap(), config).unwrap();
        (storage, tmp)
    }

    pub(super) struct TestContext {
        _tmp: TempDir,
        graph: Arc<HelixGraphEngine>,
        pub(super) collections: Arc<CollectionManager>,
        replication: Arc<ReplicationManager>,
    }

    fn setup_context() -> TestContext {
        setup_context_with_config(Config::default())
    }

    pub(super) fn setup_context_with_config(config: Config) -> TestContext {
        let tmp = TempDir::new().unwrap();
        let graph_path = tmp.path().join("graph");
        let collections_path = tmp.path().join("data");
        let graph = Arc::new(
            HelixGraphEngine::new(HelixGraphEngineOpts {
                path: graph_path.display().to_string(),
                config: config.clone(),
            })
            .unwrap(),
        );
        let collections =
            Arc::new(CollectionManager::new(collections_path, config.clone()).unwrap());
        let replication =
            Arc::new(ReplicationManager::new(Arc::clone(&collections), config).unwrap());
        TestContext {
            _tmp: tmp,
            graph,
            collections,
            replication,
        }
    }

    pub(super) fn make_input(ctx: &TestContext, body: Vec<u8>) -> HandlerInput {
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

    pub(super) fn make_relationship_point(
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
        let ctx = setup_context_with_config(Config::default().with_lsm_in_memory());
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

    pub(super) fn post_json(
        ctx: &TestContext,
        handler: fn(&HandlerInput, &mut Response) -> Result<(), GraphError>,
        body: serde_json::Value,
    ) -> (u16, serde_json::Value) {
        let input = make_input(ctx, serde_json::to_vec(&body).unwrap());
        let mut response = Response::new();
        handler(&input, &mut response).unwrap();
        let payload = serde_json::from_slice(&response.body).unwrap();
        (response.status, payload)
    }

    #[test]
    fn backfill_adjacency_invalidates_cached_pagerank() {
        let ctx = setup_context_with_config(Config::default().with_lsm_in_memory());
        let collection = "repo_graph_backfill_cache";
        let storage = ctx.collections.create_collection(collection).unwrap();
        let point = make_relationship_point(
            0xCAC4E,
            "caller_fn",
            "target_fn",
            "src/caller.rs",
            "src/target.rs",
            "calls",
            "demo-repo",
        );
        storage
            .with_write_backend(|w| storage.upsert_node_be(w, &point))
            .unwrap();

        let pagerank_body = serde_json::json!({"collection": collection, "edge_labels": ["CALLS"]});
        let (status, before) = post_json(&ctx, handle_pagerank, pagerank_body.clone());
        assert_eq!(status, 200);
        assert_eq!(before["results"].as_array().unwrap().len(), 1);

        let (status, _) = post_json(
            &ctx,
            handle_backfill_adjacency_from_points,
            serde_json::json!({"collection": collection, "batch_size": 10}),
        );
        assert_eq!(status, 200);

        let (status, after) = post_json(&ctx, handle_pagerank, pagerank_body);
        assert_eq!(status, 200);
        assert_eq!(
            after["results"].as_array().unwrap().len(),
            3,
            "cached pre-backfill ranks must not be served: {}",
            after
        );
    }

    #[test]
    fn pagerank_rejects_out_of_range_damping() {
        let ctx = setup_context();
        ctx.collections.create_collection("repo_damping").unwrap();
        for damping in [0.0, 1.0, -0.5, 1.5] {
            let (status, body) = post_json(
                &ctx,
                handle_pagerank,
                serde_json::json!({"collection": "repo_damping", "damping": damping}),
            );
            assert_eq!(status, 400, "damping {damping}: {body}");
        }
        let (status, _) = post_json(
            &ctx,
            handle_pagerank,
            serde_json::json!({"collection": "repo_damping", "damping": 0.85}),
        );
        assert_eq!(status, 200);
    }

    #[test]
    fn cycles_query_follows_split_variants_and_honors_repo() {
        let ctx = setup_context_with_config(Config::default().with_lsm_in_memory());
        let collection = "repo_cycles_split";
        let storage = ctx.collections.create_collection(collection).unwrap();
        // The client model: caller(name, path) -> callee(name, "").
        let nodes = [
            make_node(collection, "alpha", "src/a.rs"),
            make_node(collection, "alpha", ""),
            make_node(collection, "beta", "src/b.rs"),
            make_node(collection, "beta", ""),
        ];
        let forward = make_edge_with_repo(collection, "alpha", "beta", "src/a.rs", "", "repo-a");
        let back = make_edge_with_repo(collection, "beta", "alpha", "src/b.rs", "", "repo-b");
        storage
            .with_write_backend(|w| {
                for node in &nodes {
                    storage.upsert_node_be(w, node)?;
                }
                storage.upsert_edge_be(w, &forward)?;
                storage.upsert_edge_be(w, &back)?;
                Ok(())
            })
            .unwrap();

        let query = |repo: Option<&str>| {
            let mut body = serde_json::json!({
                "collection": collection,
                "symbol": "alpha",
                "path": "src/a.rs",
            });
            if let Some(repo) = repo {
                body["repo"] = serde_json::json!(repo);
            }
            let (status, payload) = post_json(&ctx, handle_cycles, body);
            assert_eq!(status, 200);
            assert_eq!(payload["truncated"], serde_json::json!(false));
            payload["cycles"].as_array().unwrap().len()
        };
        assert_eq!(query(None), 1, "logical alpha <-> beta cycle");
        assert_eq!(query(Some("repo-a")), 0, "back edge belongs to repo-b");
    }

    #[test]
    fn backfill_adjacency_force_ignores_stale_complete_marker() {
        let ctx = setup_context_with_config(Config::default().with_lsm_in_memory());
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
        let ctx = setup_context_with_config(Config::default().with_lsm_in_memory());
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
    fn chunk_edge_point_id_reads_both_forms_ce_stores() {
        // The caller's id, as the client computed it: a decimal number.
        assert_eq!(
            chunk_edge_point_id("13987204703970261753"),
            13987204703970261753
        );
        // The callee's id, as a scroll rendered it: 32 hex digits.
        assert_eq!(
            chunk_edge_point_id("0000000000000000522f42c745699236"),
            0x522f_42c7_4569_9236
        );
        // The same form with no letter in it is still hex, not a decimal number.
        assert_eq!(
            chunk_edge_point_id("00000000000000001234567890123456"),
            0x1234_5678_9012_3456
        );
        assert_eq!(chunk_edge_point_id("chunk_a"), string_id_to_u128("chunk_a"));
    }

    /// The rebuild derives the same nodes, edges and chunk-edge candidates,
    /// in the same order, whether it finds the paths' relationship points
    /// through the `caller_path` keyword index or by reading every node. The
    /// index is built over points written before it and kept up for points
    /// written after, and one path is longer than an index key may be.
    #[test]
    fn rebuild_adjacency_for_paths_finds_the_same_points_through_the_caller_path_index() {
        let ctx = setup_context_with_config(Config::default().with_lsm_in_memory());
        let long_path = format!("src/{}.rs", "deep/".repeat(200));
        let point = |id: u128, caller: &str, callee: &str, caller_path: &str, callee_path: &str| {
            make_relationship_point(id, caller, callee, caller_path, callee_path, "calls", "r")
        };
        let mut with_chunks = point(0x16, "a_fn", "d_fn", "src/a.rs", "src/d.rs");
        with_chunks
            .properties
            .insert("caller_point_id".into(), Value::String("11".into()));
        with_chunks
            .properties
            .insert("callee_point_id".into(), Value::String("12".into()));
        let before_index = [
            point(0x11, "a_fn", "b_fn", "src/a.rs", "src/b.rs"),
            point(0x12, "a_fn", "c_fn", "src/a.rs", "src/c.rs"),
            make_relationship_point(
                0x13,
                "a_mod",
                "dep",
                "src/a.rs",
                "<external>/dep",
                "imports",
                "r",
            ),
            // Another file's edge into a requested path: not this rebuild's to derive.
            point(0x14, "b_fn", "a_fn", "src/b.rs", "src/a.rs"),
        ];
        let after_index = [
            point(0x15, "deep_fn", "c_fn", &long_path, "src/c.rs"),
            with_chunks,
            point(0x17, "e_fn", "a_fn", "src/e.rs", "src/a.rs"),
        ];
        let paths = [
            "src/a.rs".to_string(),
            long_path.clone(),
            "src/absent.rs".to_string(),
        ];

        let rebuild = |name: &str, indexed: bool| {
            let storage = ctx.collections.create_collection(name).unwrap();
            let write = |points: &[NodeUpsert]| {
                storage
                    .with_write_backend(|w| {
                        for point in points {
                            storage.upsert_node_be(w, point)?;
                        }
                        Ok(())
                    })
                    .unwrap();
            };
            write(&before_index);
            if indexed {
                storage
                    .create_payload_index("caller_path", PayloadIndexSchema::Keyword)
                    .unwrap();
            }
            write(&after_index);
            let r = storage.backend.begin_read().unwrap();
            assert_eq!(
                storage.has_payload_index_at("caller_path", &r).is_some(),
                indexed,
                "{name}"
            );
            drop(r);

            // Ids derive from the collection name: one name for both stores.
            let rebuilt = storage
                .rebuild_adjacency_for_paths("parity", &paths, false)
                .unwrap();
            let (nodes, edges) = (rebuilt.nodes_upserted, rebuilt.edges_upserted);
            let candidates = rebuilt
                .chunk_candidates
                .into_iter()
                .map(|c| {
                    (
                        c.edge_type,
                        c.caller_point_id,
                        c.callee_point_id,
                        c.caller_path,
                    )
                })
                .collect::<Vec<_>>();
            // Node and edge rows are compared decoded: their property maps
            // are written in no fixed order.
            let r = storage.backend.begin_read().unwrap();
            let (mut node_rows, mut edge_rows, mut index_rows) =
                (Vec::new(), Vec::new(), Vec::new());
            let id = |key: &[u8]| u128::from_be_bytes(key.try_into().unwrap());
            storage
                .backend
                .scan(&r, Namespace::Nodes, KeyRange::all(), |key, value| {
                    let node = SerializedNode::decode_node(value, id(key)).unwrap();
                    node_rows.push((node.id, node.label, node.properties));
                    true
                })
                .unwrap();
            storage
                .backend
                .scan(&r, Namespace::Edges, KeyRange::all(), |key, value| {
                    let edge = SerializedEdge::decode_edge(value, id(key)).unwrap();
                    edge_rows.push((
                        edge.id,
                        edge.label,
                        edge.from_node,
                        edge.to_node,
                        edge.properties,
                    ));
                    true
                })
                .unwrap();
            for namespace in [
                Namespace::OutEdges,
                Namespace::InEdges,
                Namespace::EdgePathIdx,
            ] {
                storage
                    .backend
                    .scan(&r, namespace, KeyRange::all(), |key, value| {
                        index_rows.push((index_rows.len(), key.to_vec(), value.to_vec()));
                        true
                    })
                    .unwrap();
            }
            (nodes, edges, candidates, node_rows, edge_rows, index_rows)
        };

        let scanned = rebuild("parity_scan", false);
        let indexed = rebuild("parity_index", true);
        assert_eq!(
            scanned.1, 5,
            "the five points whose caller path was asked for"
        );
        assert_eq!(scanned.2.len(), 1, "the one point that names its chunks");
        assert_eq!(scanned.4.len(), 5, "and no edge of another file's point");
        assert_eq!((scanned.0, scanned.1), (indexed.0, indexed.1));
        assert_eq!(scanned.2, indexed.2);
        assert!(scanned.3 == indexed.3, "the stores hold the same nodes");
        assert!(scanned.4 == indexed.4, "the stores hold the same edges");
        assert!(
            scanned.5 == indexed.5,
            "the stores hold the same adjacency and path rows"
        );
    }

    /// `from_only` removes the edges that start in the paths and keeps the
    /// edges other files have into them, whichever way the edges are found:
    /// through the path index once its backfill is complete, or by the edge
    /// scan before that (here for edges with no index rows at all). Without
    /// the option both ends match, as they always did.
    #[test]
    fn delete_by_paths_from_only_keeps_the_edges_into_the_paths() {
        for (indexed, from_only) in [(true, true), (false, true), (true, false), (false, false)] {
            let ctx = setup_context_with_config(Config::default().with_lsm_in_memory());
            let collection = format!("repo_from_{indexed}_{from_only}");
            let collection = collection.as_str();
            let storage = ctx.collections.create_collection(collection).unwrap();
            let nodes = [
                make_node(collection, "a_fn", "src/a.rs"),
                make_node(collection, "b_fn", "src/b.rs"),
                make_node(collection, "c_fn", "src/c.rs"),
            ];
            let edge = |from: &str, to: &str, from_path: &str, to_path: &str| {
                make_edge_with_repo_and_paths(collection, from, to, from_path, to_path, "repo-a")
            };
            let outgoing = edge("a_fn", "b_fn", "src/a.rs", "src/b.rs");
            let incoming = edge("c_fn", "a_fn", "src/c.rs", "src/a.rs");
            let elsewhere = edge("c_fn", "b_fn", "src/c.rs", "src/b.rs");
            let written = storage
                .with_write_backend(|w| {
                    for node in &nodes {
                        storage.upsert_node_be(w, node)?;
                    }
                    [&outgoing, &incoming, &elsewhere]
                        .into_iter()
                        .map(|edge| storage.upsert_edge_be(w, edge))
                        .collect::<Result<Vec<_>, _>>()
                })
                .unwrap();
            if indexed {
                while !storage.backfill_edge_path_index_batch(2).unwrap().1 {}
            } else {
                storage
                    .with_write_backend(|w| {
                        for edge in &written {
                            storage.delete_edge_paths_be(w, edge)?;
                        }
                        Ok(())
                    })
                    .unwrap();
            }

            let body = sonic_rs::to_vec(&sonic_rs::json!({
                "collection": collection,
                "paths": ["src/a.rs"],
                "repo": "repo-a",
                "delete_orphaned_nodes": true,
                "from_only": from_only,
            }))
            .unwrap();
            let mut response = Response::new();
            handle_delete_by_paths(&make_input(&ctx, body), &mut response).unwrap();
            let payload: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
            let case = format!("indexed={indexed} from_only={from_only}");
            assert_eq!(response.status, 200, "{case}");
            assert_eq!(payload["from_only"], from_only, "{case}");
            assert_eq!(
                payload["edges_deleted"],
                if from_only { 1 } else { 2 },
                "{case}"
            );

            let r = storage.backend.begin_read().unwrap();
            let present = |edge: &EdgeUpsert| storage.get_edge_be(&r, &edge.id).is_ok();
            assert!(!present(&outgoing), "{case}: the path's own edge goes");
            assert_eq!(
                present(&incoming),
                from_only,
                "{case}: the edge into the path"
            );
            assert!(
                present(&elsewhere),
                "{case}: an edge that touches no path stays"
            );
            // The kept edge keeps its end in the path: the node is not an orphan.
            assert_eq!(
                storage.get_node_be(&r, &nodes[0].id).is_ok(),
                from_only,
                "{case}: the path's node"
            );
            let callers = bfs_reverse_be(&storage, &r, &[nodes[0].id], &["CALLS"], 1, 10, None)
                .unwrap()
                .len();
            assert_eq!(
                callers,
                usize::from(from_only),
                "{case}: callers of the path's symbol"
            );
        }
    }

    /// The stored node and edge counters against the rows a recount finds.
    fn assert_counters_match_the_rows(storage: &HelixGraphStorage, case: &str) {
        let recount = storage.recount_lsm_counters().unwrap();
        assert_eq!(
            (recount.nodes.before, recount.edges.before),
            (recount.nodes.after, recount.edges.after),
            "{case}: stored node and edge counters against a recount"
        );
    }

    /// `replace_outgoing` makes a rebuild also the delete of what it replaces:
    /// the edges that start in the paths and that no relationship point
    /// derives any more go, in the same write, with the nodes they leave
    /// without an edge. Edges other files have into the paths stay, with or
    /// without a point of their own, and so does a node another edge still
    /// ends at. A path with no point left loses its outgoing edges and
    /// nothing else does. The edges are found through the path index once its
    /// backfill is complete and by reading every edge before that (here for
    /// edges with no index row at all). Without the option nothing is
    /// dropped, as before.
    #[test]
    fn rebuild_with_replace_outgoing_drops_what_the_points_no_longer_derive() {
        for (indexed, replace) in [(true, true), (false, true), (true, false), (false, false)] {
            let ctx = setup_context_with_config(Config::default().with_lsm_in_memory());
            let collection = format!("repo_replace_{indexed}_{replace}");
            let c = collection.as_str();
            let storage = ctx.collections.create_collection(c).unwrap();
            let case = format!("indexed={indexed} replace={replace}");

            // The points as they are now: `a_fn` calls `c_fn` (it called
            // `b_fn` and `shared` when its edges were written), `a2_fn` is
            // new, and `x_fn` of another file calls `a_fn`. `src/gone.rs` has
            // no point left.
            let point = |id: u128,
                         caller: &str,
                         callee: &str,
                         caller_path: &str,
                         callee_path: &str| {
                make_relationship_point(id, caller, callee, caller_path, callee_path, "calls", "r")
            };
            let points = [
                point(0x11, "a_fn", "c_fn", "src/a.rs", "src/c.rs"),
                point(0x12, "a2_fn", "d_fn", "src/a.rs", "src/d.rs"),
                point(0x13, "x_fn", "a_fn", "src/x.rs", "src/a.rs"),
            ];
            let edge = |from: &str, to: &str, from_path: &str, to_path: &str| {
                make_edge_with_repo_and_paths(c, from, to, from_path, to_path, "r")
            };
            let derived_before = edge("a_fn", "c_fn", "src/a.rs", "src/c.rs");
            let stale = [
                edge("a_fn", "b_fn", "src/a.rs", "src/b.rs"),
                edge("a_fn", "shared", "src/a.rs", "src/s.rs"),
                edge("dead_fn", "b_fn", "src/a.rs", "src/b.rs"),
                edge("g_fn", "h_fn", "src/gone.rs", "src/h.rs"),
            ];
            let others = [
                // Into the paths: one with a point, two without.
                edge("x_fn", "a_fn", "src/x.rs", "src/a.rs"),
                edge("y_fn", "a_fn", "src/y.rs", "src/a.rs"),
                edge("z_fn", "g_fn", "src/z.rs", "src/gone.rs"),
                // Away from them.
                edge("x_fn", "shared", "src/x.rs", "src/s.rs"),
                edge("u_fn", "v_fn", "src/u.rs", "src/v.rs"),
            ];
            let node = |name: &str, path: &str| make_node(c, name, path);
            let nodes = [
                node("a_fn", "src/a.rs"),
                node("dead_fn", "src/a.rs"),
                node("b_fn", "src/b.rs"),
                node("c_fn", "src/c.rs"),
                node("shared", "src/s.rs"),
                node("g_fn", "src/gone.rs"),
                node("h_fn", "src/h.rs"),
                node("x_fn", "src/x.rs"),
                node("y_fn", "src/y.rs"),
                node("z_fn", "src/z.rs"),
                node("u_fn", "src/u.rs"),
                node("v_fn", "src/v.rs"),
            ];
            let written = storage
                .with_write_backend(|w| {
                    for point in &points {
                        storage.upsert_node_be(w, point)?;
                    }
                    for node in &nodes {
                        storage.upsert_node_be(w, node)?;
                    }
                    std::iter::once(&derived_before)
                        .chain(&stale)
                        .chain(&others)
                        .map(|edge| storage.upsert_edge_be(w, edge))
                        .collect::<Result<Vec<_>, _>>()
                })
                .unwrap();
            if indexed {
                while !storage.backfill_edge_path_index_batch(4).unwrap().1 {}
            } else {
                storage
                    .with_write_backend(|w| {
                        for edge in &written {
                            storage.delete_edge_paths_be(w, edge)?;
                        }
                        Ok(())
                    })
                    .unwrap();
            }

            let (status, payload) = post_json(
                &ctx,
                handle_rebuild_adjacency_for_paths,
                serde_json::json!({
                    "collection": c,
                    "paths": ["src/a.rs", "src/gone.rs"],
                    "replace_outgoing": replace,
                }),
            );
            assert_eq!(status, 200, "{case}");
            assert_eq!(payload["replaced_outgoing"], replace, "{case}");
            assert_eq!(
                (
                    payload["nodes_upserted"].as_u64(),
                    payload["edges_upserted"].as_u64()
                ),
                (Some(4), Some(2)),
                "{case}: the two points of `src/a.rs`"
            );
            assert_eq!(
                (
                    payload["edges_deleted"].as_u64(),
                    payload["nodes_deleted"].as_u64()
                ),
                if replace {
                    (Some(4), Some(3))
                } else {
                    (Some(0), Some(0))
                },
                "{case}: the four edges no point derives, and `b_fn`, `dead_fn`, `h_fn`"
            );

            let r = storage.backend.begin_read().unwrap();
            let has_edge = |edge: &EdgeUpsert| storage.get_edge_be(&r, &edge.id).is_ok();
            let has_node =
                |name: &str, path: &str| storage.get_node_be(&r, &node(name, path).id).is_ok();
            assert!(
                has_edge(&derived_before),
                "{case}: a derived edge that was there"
            );
            assert!(
                has_edge(&edge("a2_fn", "d_fn", "src/a.rs", "src/d.rs")),
                "{case}: a derived edge that was not"
            );
            for edge in &stale {
                assert_eq!(has_edge(edge), !replace, "{case}: an edge no point derives");
            }
            for edge in &others {
                assert!(
                    has_edge(edge),
                    "{case}: an edge that starts in another file"
                );
            }
            for (name, path, has_another_edge) in [
                ("b_fn", "src/b.rs", false),
                ("dead_fn", "src/a.rs", false),
                ("h_fn", "src/h.rs", false),
                ("shared", "src/s.rs", true),
                ("g_fn", "src/gone.rs", true),
                ("a_fn", "src/a.rs", true),
            ] {
                assert_eq!(
                    has_node(name, path),
                    has_another_edge || !replace,
                    "{case}: the node of {name}"
                );
            }
            let callers = bfs_reverse_be(
                &storage,
                &r,
                &[node("a_fn", "src/a.rs").id],
                &["CALLS"],
                1,
                10,
                None,
            )
            .unwrap()
            .len();
            assert_eq!(callers, 2, "{case}: `x_fn` and `y_fn` call `a_fn`");
            drop(r);
            assert_counters_match_the_rows(&storage, &case);
        }
    }

    /// A replace is one write batch: when it fails part-way, here at a
    /// derived node whose stored row does not decode, after the batch has
    /// buffered the drops of the edges the points no longer derive, nothing
    /// of it is stored. The same call replaces all of it once the row is out
    /// of the way. A node left without an edge whose row does not decode is
    /// no reason to fail: its row stays and it is not counted.
    #[test]
    fn a_replace_that_fails_leaves_every_row_as_it_was() {
        const C: &str = "repo_replace_whole";
        let ctx = setup_context_with_config(Config::default().with_lsm_in_memory());
        let storage = ctx.collections.create_collection(C).unwrap();
        let stale = make_edge_with_repo_and_paths(C, "a_fn", "b_fn", "src/a.rs", "src/b.rs", "r");
        let to_bad = make_edge_with_repo_and_paths(C, "a_fn", "bad", "src/a.rs", "src/b.rs", "r");
        let bad_key = make_node(C, "bad", "src/b.rs").id.to_be_bytes();
        let derived = make_edge_with_repo_and_paths(C, "a_fn", "c_fn", "src/a.rs", "src/c.rs", "r");
        let point =
            make_relationship_point(0x11, "a_fn", "c_fn", "src/a.rs", "src/c.rs", "calls", "r");
        storage
            .with_write_backend(|w| {
                storage.upsert_node_be(w, &point)?;
                storage.upsert_node_be(w, &make_node(C, "a_fn", "src/a.rs"))?;
                storage.upsert_node_be(w, &make_node(C, "b_fn", "src/b.rs"))?;
                storage.upsert_node_be(w, &make_node(C, "bad", "src/b.rs"))?;
                storage.upsert_edge_be(w, &stale)?;
                storage.upsert_edge_be(w, &to_bad)?;
                // Counted as a node, and no longer one that decodes.
                storage
                    .backend
                    .put(w, Namespace::Nodes, &bad_key, &[0xFF])
                    .map_err(|e| GraphError::New(e.to_string()))
            })
            .unwrap();
        assert_counters_match_the_rows(&storage, "before");

        // The row of the derived callee `c_fn`: undecodable, or not there.
        let callee_key = make_node(C, "c_fn", "src/c.rs").id.to_be_bytes();
        let set_undecodable_callee = |undecodable: bool| {
            storage
                .with_write_backend(|w| {
                    let written = if undecodable {
                        storage
                            .backend
                            .put(w, Namespace::Nodes, &callee_key, &[0xFF])
                    } else {
                        storage.backend.delete(w, Namespace::Nodes, &callee_key)
                    };
                    written.map_err(|e| GraphError::New(e.to_string()))
                })
                .unwrap();
        };
        let stored = || {
            let r = storage.backend.begin_read().unwrap();
            let mut rows = Vec::new();
            for namespace in [
                Namespace::Nodes,
                Namespace::Edges,
                Namespace::OutEdges,
                Namespace::InEdges,
                Namespace::EdgePathIdx,
            ] {
                storage
                    .backend
                    .scan(&r, namespace, KeyRange::all(), |key, value| {
                        rows.push((namespace, key.to_vec(), value.to_vec()));
                        true
                    })
                    .unwrap();
            }
            let stats = storage.get_metadata_be(&r).unwrap().stats;
            (rows, stats.node_count, stats.edge_count)
        };

        set_undecodable_callee(true);
        let before = stored();
        let paths = ["src/a.rs".to_string()];
        assert!(
            storage
                .rebuild_adjacency_for_paths(C, &paths, true)
                .is_err(),
            "the row of the derived callee does not decode"
        );
        assert!(stored() == before, "a replace that failed stored a row");

        set_undecodable_callee(false);
        let rebuilt = storage
            .rebuild_adjacency_for_paths(C, &paths, true)
            .unwrap();
        assert_eq!(
            rebuilt.replaced,
            Some((2, 1)),
            "the edges to `b_fn` and `bad`, and `b_fn`"
        );
        let r = storage.backend.begin_read().unwrap();
        assert!(storage.get_edge_be(&r, &stale.id).is_err());
        assert!(storage.get_edge_be(&r, &to_bad.id).is_err());
        assert!(storage.get_edge_be(&r, &derived.id).is_ok());
        let has_row = |key: &[u8]| {
            storage
                .backend
                .get_with(&r, Namespace::Nodes, key, |row| row.is_some())
                .unwrap()
        };
        assert!(has_row(&bad_key), "the row that does not decode stays");
        assert!(!has_row(&make_node(C, "b_fn", "src/b.rs").id.to_be_bytes()));
        drop(r);
        assert_counters_match_the_rows(&storage, "after");
    }

    /// While a replace moves a caller's edge from one callee to the next, a
    /// reader finds the caller with one callee at every moment: the one
    /// before or the new one. A delete followed by a rebuild left it with
    /// none in between.
    #[test]
    fn a_reader_finds_the_old_outgoing_edge_or_the_new_one_during_a_replace() {
        use std::sync::atomic::{AtomicBool, Ordering};

        const C: &str = "repo_replace_reader";
        const ROUNDS: usize = 5;
        let ctx = setup_context_with_config(Config::default().with_lsm_in_memory());
        let storage = ctx.collections.create_collection(C).unwrap();
        let caller = make_node(C, "a_fn", "src/a.rs").id;
        let callee = |round: usize| make_node(C, &format!("t{round}"), "src/t.rs").id;
        let paths = ["src/a.rs".to_string()];
        // The caller's one point now names the callee of this round.
        let point_to = |round: usize| {
            let point = make_relationship_point(
                0x11,
                "a_fn",
                &format!("t{round}"),
                "src/a.rs",
                "src/t.rs",
                "calls",
                "r",
            );
            storage
                .with_write_backend(|w| storage.upsert_node_be(w, &point).map(|_| ()))
                .unwrap();
            storage
                .rebuild_adjacency_for_paths(C, &paths, true)
                .unwrap()
        };
        point_to(0);

        let stop = AtomicBool::new(false);
        let found = std::thread::scope(|scope| {
            let (storage, stop) = (&storage, &stop);
            let reader = scope.spawn(move || {
                let mut found = HashSet::new();
                // Reads once more after the last round.
                loop {
                    let stopped = stop.load(Ordering::SeqCst);
                    let r = storage.backend.begin_read().unwrap();
                    let callees = storage.out_adj_be(&r, &caller, "CALLS").unwrap();
                    assert_eq!(callees.len(), 1, "the caller's callees: {callees:?}");
                    found.insert(callees[0].0);
                    if stopped {
                        return found;
                    }
                }
            });
            // Stops the reader on a failed assert too, so the scope can join it.
            struct StopOnDrop<'a>(&'a AtomicBool);
            impl Drop for StopOnDrop<'_> {
                fn drop(&mut self) {
                    self.0.store(true, Ordering::SeqCst);
                }
            }
            let stopper = StopOnDrop(stop);
            for round in 1..=ROUNDS {
                assert_eq!(
                    point_to(round).replaced,
                    Some((1, 1)),
                    "round {round}: the edge to the callee before, and that callee"
                );
            }
            drop(stopper);
            reader.join().unwrap()
        });
        let every: HashSet<u128> = (0..=ROUNDS).map(&callee).collect();
        assert!(
            found.is_subset(&every) && found.contains(&callee(ROUNDS)),
            "callees a reader found: {found:?}"
        );
        let r = storage.backend.begin_read().unwrap();
        assert_eq!(
            storage.out_adj_be(&r, &caller, "CALLS").unwrap(),
            vec![(
                callee(ROUNDS),
                deterministic_id::edge_id(
                    C,
                    "CALLS",
                    "a_fn",
                    &format!("t{ROUNDS}"),
                    "src/a.rs",
                    "src/t.rs"
                )
            )]
        );
        drop(r);
        assert_counters_match_the_rows(&storage, "after the rounds");
    }

    /// A request with no path writes nothing and says whether the edges of
    /// the next one would be replaced: on an LSM collection that is asked
    /// to, and never on LMDB, which rebuilds as it does without the option.
    #[test]
    fn rebuild_with_no_path_says_whether_replace_outgoing_is_honoured() {
        for (lsm, replace) in [(true, true), (true, false), (false, true)] {
            let config = if lsm {
                Config::default().with_lsm_in_memory()
            } else {
                Config::default()
            };
            let ctx = setup_context_with_config(config);
            let collection = format!("repo_ask_{lsm}_{replace}");
            let storage = ctx.collections.create_collection(&collection).unwrap();
            assert_eq!(storage.backend.kind() == BackendKind::Lsm, lsm);
            let (status, payload) = post_json(
                &ctx,
                handle_rebuild_adjacency_for_paths,
                serde_json::json!({
                    "collection": collection,
                    "paths": [],
                    "replace_outgoing": replace,
                }),
            );
            assert_eq!(status, 200);
            assert_eq!(
                payload["replaced_outgoing"],
                lsm && replace,
                "lsm={lsm} replace={replace}"
            );
        }
    }

    /// Every graph write waits for the gate point writes hold on its
    /// collection, found by the collection's resolved name. One that never
    /// gets the gate answers busy and has written nothing.
    #[test]
    #[serial_test::serial]
    fn graph_writes_wait_for_the_point_mutation_gate_and_answer_busy_when_it_stays_held() {
        type Handler = fn(&HandlerInput, &mut Response) -> Result<(), GraphError>;
        const C: &str = "repo_gated_graph";
        const ALIAS: &str = "repo_gated_graph_alias";
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let ctx = setup_context_with_config(Config::default().with_lsm_in_memory());
        let storage = ctx.collections.create_collection(C).unwrap();
        ctx.collections.create_alias(ALIAS, C).unwrap();
        let edge = make_edge(C, "a_fn", "b_fn", "src/a.rs", "src/b.rs");
        storage
            .with_write_backend(|w| {
                storage.upsert_node_be(w, &make_node(C, "a_fn", "src/a.rs"))?;
                storage.upsert_node_be(w, &make_node(C, "b_fn", "src/b.rs"))?;
                storage.upsert_edge_be(w, &edge)
            })
            .unwrap();
        let writes = |collection: &str| -> Vec<(&'static str, Handler, serde_json::Value)> {
            let only = serde_json::json!({"collection": collection});
            let path = serde_json::json!({"collection": collection, "path": "src/a.rs"});
            let paths = serde_json::json!({"collection": collection, "paths": ["src/a.rs"]});
            vec![
                (
                    "backfill_edge_path_index",
                    handle_backfill_edge_path_index as Handler,
                    only.clone(),
                ),
                (
                    "backfill_adjacency_from_points",
                    handle_backfill_adjacency_from_points as Handler,
                    only,
                ),
                (
                    "rebuild_adjacency_for_paths",
                    handle_rebuild_adjacency_for_paths as Handler,
                    paths.clone(),
                ),
                ("delete_by_path", handle_delete_by_path as Handler, path),
                ("delete_by_paths", handle_delete_by_paths as Handler, paths),
            ]
        };

        let gate = ctx.collections.point_mutation_gate(C);
        let held = gate.lock().unwrap();
        {
            let _wait = EnvVarGuard::set("HELIX_GRAPH_GATE_WAIT_MS", "50");
            for (route, handler, body) in writes(ALIAS) {
                let input = make_input(&ctx, serde_json::to_vec(&body).unwrap());
                let mut response = Response::new();
                handler(&input, &mut response).unwrap();
                assert_eq!(response.status, 503, "{route}");
                assert_eq!(
                    response.headers.get("Retry-After").map(String::as_str),
                    Some("1"),
                    "{route}"
                );
            }
        }
        let r = storage.backend.begin_read().unwrap();
        assert!(
            storage.get_edge_be(&r, &edge.id).is_ok(),
            "a delete that answered busy deleted the edge"
        );
        drop(r);

        let _wait = EnvVarGuard::set("HELIX_GRAPH_GATE_WAIT_MS", "30000");
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            for (route, handler, body) in writes(C) {
                let (ctx, done_tx) = (&ctx, done_tx.clone());
                scope.spawn(move || {
                    let (status, _) = post_json(ctx, handler, body);
                    done_tx.send((route, status)).unwrap();
                });
            }
            let ran_while_gated = done_rx.recv_timeout(Duration::from_millis(500)).ok();
            drop(held);
            let answers: Vec<(&str, u16)> = (0..5)
                .filter_map(|_| done_rx.recv_timeout(Duration::from_secs(30)).ok())
                .collect();
            assert_eq!(
                ran_while_gated, None,
                "a graph write ran while the gate was held"
            );
            assert_eq!(answers.len(), 5, "after the gate was released: {answers:?}");
            assert!(
                answers.iter().all(|(_, status)| *status == 200),
                "{answers:?}"
            );
        });
    }

    /// The wait is paid once per submit attempt: the gateway repeats a busy
    /// answer (`HELIX_SUBMIT_MAX_ATTEMPTS`, 3, with 250 and 500 ms between),
    /// so the caller hears busy after three waits and two pauses. That has
    /// to stay under the 30 s the client gives its path-index and repair steps.
    #[test]
    #[serial_test::serial]
    fn a_graph_write_that_never_gets_the_gate_answers_busy_inside_30_s() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        std::env::remove_var("HELIX_GRAPH_GATE_WAIT_MS");
        let wait = graph_gate_wait();
        assert_eq!(wait, Duration::from_millis(8_000));
        let busy_after = wait * 3 + Duration::from_millis(250 + 500);
        assert!(busy_after < Duration::from_secs(30), "{busy_after:?}");
    }

    /// Another writer on the collection while a delete or a replace is under
    /// way: the client sends its next request once it stops waiting for the first.
    /// The batch works out the nodes it orphans once its edge drops are
    /// staged, on what is committed by then, and here the other writer
    /// commits at that moment. When it is a rebuild of another file that
    /// brings an edge to the callee, the callee keeps its row. When it is a
    /// second request of the same kind that drops the callee's other edge,
    /// and keeps the callee for the edge this one is dropping, this one
    /// drops the callee: neither did while each decided on a read from
    /// before the other was stored.
    #[test]
    fn a_batch_works_out_its_orphans_on_what_another_writer_has_committed_by_then() {
        use crate::helix_engine::storage_core::backend_adj_ops::before_next_orphan_check_for_test;
        use std::cell::Cell;
        use std::rc::Rc;

        /// A delete of the edge, or a replace of the file it starts in: the
        /// edges and the nodes that went.
        fn drop_outgoing(
            storage: &HelixGraphStorage,
            collection: &str,
            replace: bool,
            path: &str,
            edge_id: u128,
        ) -> (usize, usize) {
            if replace {
                storage
                    .rebuild_adjacency_for_paths(collection, &[path.to_string()], true)
                    .unwrap()
                    .replaced
                    .unwrap()
            } else {
                delete_edges_with_their_orphans(storage, &[edge_id]).unwrap()
            }
        }

        for (replace, other_drops) in [(false, false), (false, true), (true, false), (true, true)] {
            let ctx = setup_context_with_config(Config::default().with_lsm_in_memory());
            let collection = format!("repo_two_writers_{replace}_{other_drops}");
            let c = collection.as_str();
            let storage = ctx.collections.create_collection(c).unwrap();
            let case = format!("replace={replace} other_drops={other_drops}");

            let node = |name: &str| make_node(c, &format!("{name}_fn"), &format!("src/{name}.rs"));
            let edge_to_n = |from: &str| {
                make_edge_with_repo_and_paths(
                    c,
                    &format!("{from}_fn"),
                    "n_fn",
                    &format!("src/{from}.rs"),
                    "src/n.rs",
                    "r",
                )
            };
            let (a_n, b_n, x_n) = (edge_to_n("a"), edge_to_n("b"), edge_to_n("x"));
            // No point derives the edges of `a_fn` and `b_fn` any more. The
            // call of `x_fn` has its point and no edge yet.
            let x_point =
                make_relationship_point(0x11, "x_fn", "n_fn", "src/x.rs", "src/n.rs", "calls", "r");
            storage
                .with_write_backend(|w| {
                    storage.upsert_node_be(w, &x_point)?;
                    storage.upsert_node_be(w, &node("a"))?;
                    storage.upsert_node_be(w, &node("n"))?;
                    storage.upsert_edge_be(w, &a_n)?;
                    if other_drops {
                        storage.upsert_node_be(w, &node("b"))?;
                        storage.upsert_edge_be(w, &b_n)?;
                    }
                    Ok(())
                })
                .unwrap();

            let ran = Rc::new(Cell::new(false));
            before_next_orphan_check_for_test({
                let (other, collection, ran) =
                    (Arc::clone(&storage), collection.clone(), Rc::clone(&ran));
                let b_n = b_n.id;
                move || {
                    ran.set(true);
                    if other_drops {
                        assert_eq!(
                            drop_outgoing(&other, &collection, replace, "src/b.rs", b_n),
                            (1, 1),
                            "the other writer: the edge of `b_fn`, and `b_fn`"
                        );
                    } else {
                        let rebuilt = other
                            .rebuild_adjacency_for_paths(
                                &collection,
                                &["src/x.rs".to_string()],
                                false,
                            )
                            .unwrap();
                        assert_eq!((rebuilt.nodes_upserted, rebuilt.edges_upserted), (2, 1));
                    }
                }
            });
            let dropped = drop_outgoing(&storage, c, replace, "src/a.rs", a_n.id);
            assert!(ran.get(), "{case}: the other writer ran inside the batch");
            assert_eq!(
                dropped,
                (1, if other_drops { 2 } else { 1 }),
                "{case}: the edge of `a_fn`, `a_fn`, and `n_fn` once its other edge has gone"
            );

            let r = storage.backend.begin_read().unwrap();
            let has_node = |name: &str| storage.get_node_be(&r, &node(name).id).is_ok();
            let has_edge = |edge: &EdgeUpsert| storage.get_edge_be(&r, &edge.id).is_ok();
            assert_eq!(
                ["a", "b", "n", "x"].map(has_node),
                [false, false, !other_drops, !other_drops],
                "{case}: the node rows of `a_fn`, `b_fn`, `n_fn` and `x_fn`"
            );
            assert_eq!(
                [&a_n, &b_n, &x_n].map(has_edge),
                [false, false, !other_drops],
                "{case}: the edges to `n_fn` from `a_fn`, `b_fn` and `x_fn`"
            );
            drop(r);
            assert_counters_match_the_rows(&storage, &case);
        }
    }

    /// The orphan cleanup of a delete leaves the same nodes, edges, index rows
    /// and counters whether it reads in batches or one row at a time
    /// (`HELIX_LSM_DELETE_BATCH_READS=0`), across chunk boundaries. Nodes left
    /// without edges go, a node that keeps an edge stays (one of them a hub),
    /// and an endpoint that never had a node row is counted as before without
    /// moving the node counter.
    #[test]
    #[serial_test::serial]
    fn lsm_delete_by_paths_removes_the_same_orphans_batched_and_one_read_at_a_time() {
        use crate::helix_engine::storage_core::backend_lsm::lsm_round_trips_for_test as round_trips;

        const C: &str = "repo_orphans";
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_budget_env();
        let node_id = |name: &str, path: &str| deterministic_id::node_id(C, "Symbol", name, path);
        let common = node_id("common", "src/common.rs");
        let mut kept = vec![common, node_id("keep1", "src/keep.rs"), node_id("ext", "")];
        kept.sort_unstable();

        // What the delete of `src/a.rs` and `src/b.rs` leaves, and the storage
        // round trips the request took.
        let run = |chunk_size: &str, batch_reads: &str| {
            let _chunk = EnvVarGuard::set("HELIX_GRAPH_DELETE_TXN_CHUNK_SIZE", chunk_size);
            let _reads = EnvVarGuard::set("HELIX_LSM_DELETE_BATCH_READS", batch_reads);
            let ctx = setup_context_with_config(Config::default().with_lsm_in_memory());
            let storage = ctx.collections.create_collection(C).unwrap();
            assert_eq!(storage.backend.kind(), BackendKind::Lsm);
            storage
                .create_payload_index("path", PayloadIndexSchema::Keyword)
                .unwrap();

            let nodes = [
                make_node(C, "a1", "src/a.rs"),
                make_node(C, "a2", "src/a.rs"),
                make_node(C, "b1", "src/b.rs"),
                make_node(C, "solo1", "src/b.rs"),
                make_node(C, "solo2", "src/b.rs"),
                make_node(C, "common", "src/common.rs"),
                make_node(C, "keep1", "src/keep.rs"),
                make_node(C, "ext", ""),
            ];
            let edge = |from: &str, to: &str, from_path: &str, to_path: &str| {
                make_edge_with_repo_and_paths(C, from, to, from_path, to_path, "r")
            };
            let mut edges = vec![
                edge("a1", "common", "src/a.rs", "src/common.rs"),
                edge("a1", "a2", "src/a.rs", "src/a.rs"),
                edge("a1", "ext", "src/a.rs", ""),
                // `ghost` has no node row.
                edge("a2", "ghost", "src/a.rs", "src/ghost.rs"),
                edge("b1", "common", "src/b.rs", "src/common.rs"),
                edge("keep1", "b1", "src/keep.rs", "src/b.rs"),
                edge("solo1", "solo2", "src/b.rs", "src/b.rs"),
                edge("keep1", "common", "src/keep.rs", "src/common.rs"),
            ];
            edges.extend((0..40).map(|i| edge(&format!("k{i}"), "ext", "src/keep.rs", "")));
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

            let before = round_trips();
            let (status, payload) = post_json(
                &ctx,
                handle_delete_by_paths,
                serde_json::json!({
                    "collection": C,
                    "paths": ["src/a.rs", "src/b.rs"],
                    "delete_orphaned_nodes": true,
                }),
            );
            let spent = round_trips() - before;
            assert_eq!(status, 200);
            // Seven edges touch the two files. Five nodes with a row and
            // `ghost` are left without edges.
            assert_eq!(
                (
                    payload["edges_deleted"].as_u64(),
                    payload["nodes_deleted"].as_u64()
                ),
                (Some(7), Some(6)),
                "chunk {chunk_size}, batch {batch_reads}"
            );

            let r = storage.backend.begin_read().unwrap();
            let path_index =
                HelixGraphStorage::payload_index_db_name("path", &PayloadIndexSchema::Keyword);
            // Node and edge rows hold a property map serialized in hash order,
            // so those two are compared by id only.
            let rows: Vec<(&str, Vec<(Vec<u8>, Vec<u8>)>)> = [
                ("nodes", Namespace::Nodes, false),
                ("edges", Namespace::Edges, false),
                ("out adjacency", Namespace::OutEdges, true),
                ("in adjacency", Namespace::InEdges, true),
                ("edge path index", Namespace::EdgePathIdx, true),
                ("name index", Namespace::MultiIndex("name"), true),
                ("path index", Namespace::PayloadIndex(&path_index), true),
            ]
            .into_iter()
            .map(|(name, ns, with_values)| {
                let mut rows = Vec::new();
                storage
                    .backend
                    .scan(&r, ns, KeyRange::all(), |key, value| {
                        let value: &[u8] = if with_values { value } else { &[] };
                        rows.push((key.to_vec(), value.to_vec()));
                        true
                    })
                    .unwrap();
                (name, rows)
            })
            .collect();
            let node_rows: Vec<u128> = rows[0]
                .1
                .iter()
                .map(|(key, _)| u128::from_be_bytes(key.as_slice().try_into().unwrap()))
                .collect();
            assert_eq!(node_rows, kept, "chunk {chunk_size}, batch {batch_reads}");
            assert_eq!(rows[1].1.len(), 41, "edges left");
            let named = |name: &str| {
                storage
                    .get_nodes_by_multi_index_be(&r, "name", &Value::String(name.into()))
                    .unwrap()
            };
            let at_path = |path: &str| {
                storage
                    .get_nodes_by_payload_value_be(&r, "path", &Value::String(path.into()))
                    .unwrap()
            };
            assert!(named("a1").is_empty() && named("solo2").is_empty());
            assert_eq!(named("common"), vec![common]);
            assert!(at_path("src/a.rs").is_empty() && at_path("src/b.rs").is_empty());
            assert_eq!(at_path("src/common.rs"), vec![common]);
            let stats = storage
                .with_read_backend(|r| storage.get_metadata_be(r))
                .unwrap()
                .stats;
            assert_eq!(
                (stats.node_count, stats.edge_count),
                (3, 41),
                "chunk {chunk_size}, batch {batch_reads}"
            );
            (rows, spent)
        };

        let (batched, batched_trips) = run("3", "1");
        let same_rows = |other: Vec<(&str, Vec<(Vec<u8>, Vec<u8>)>)>, arm: &str| {
            for ((name, rows), (_, expected)) in other.iter().zip(&batched) {
                assert!(
                    rows == expected,
                    "{arm}: {name} holds {} rows that differ from the {} of the batched run",
                    rows.len(),
                    expected.len()
                );
            }
        };
        let (one_at_a_time, one_at_a_time_trips) = run("3", "0");
        same_rows(one_at_a_time, "batching off");
        // Each chunk of three edges checks its ends, nine nodes between them:
        // two adjacency reads per end one at a time, two per chunk batched
        // (and the edge and node reads save more).
        assert!(
            batched_trips + 12 <= one_at_a_time_trips,
            "batched orphan cleanup took {batched_trips} round trips, one read at a time {one_at_a_time_trips}"
        );
        // One node per chunk: every boundary, and chunks with nothing to remove.
        same_rows(run("1", "1").0, "chunk size 1");
        same_rows(run("1", "0").0, "chunk size 1, batching off");
        // The nodes after every edge chunk, as before they went with the edges.
        let _two_steps = EnvVarGuard::set("HELIX_GRAPH_DELETE_NODES_WITH_EDGES", "0");
        same_rows(run("3", "1").0, "nodes after the edges");
    }

    /// Edges deleted by a request leave their endpoints removable in that same
    /// request: the orphan check must not count the rows of the edges the
    /// request is deleting, which the snapshot it reads still holds.
    #[test]
    #[serial_test::serial]
    fn lsm_delete_by_path_removes_the_nodes_its_own_edge_deletes_orphan() {
        const C: &str = "repo_same_request";
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_budget_env();
        let ctx = setup_context_with_config(Config::default().with_lsm_in_memory());
        let storage = ctx.collections.create_collection(C).unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);

        let nodes = [
            make_node(C, "caller", "src/a.rs"),
            make_node(C, "callee", "src/b.rs"),
            make_node(C, "other_caller", "src/c.rs"),
            make_node(C, "other_callee", "src/d.rs"),
        ];
        let edges = [
            make_edge_with_repo_and_paths(C, "caller", "callee", "src/a.rs", "src/b.rs", "r"),
            make_edge_with_repo_and_paths(
                C,
                "other_caller",
                "other_callee",
                "src/c.rs",
                "src/d.rs",
                "r",
            ),
        ];
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

        let (status, payload) = post_json(
            &ctx,
            handle_delete_by_path,
            serde_json::json!({
                "collection": C,
                "path": "src/a.rs",
                "delete_orphaned_nodes": true,
            }),
        );
        assert_eq!(status, 200);
        assert_eq!(payload["edges_deleted"], 1);
        assert_eq!(payload["nodes_deleted"], 2);

        let r = storage.backend.begin_read().unwrap();
        for (node, kept) in nodes.iter().zip([false, false, true, true]) {
            assert_eq!(
                storage.get_node_be(&r, &node.id).is_ok(),
                kept,
                "node row of {:?}",
                node.properties.get("name")
            );
        }
        let stats = storage
            .with_read_backend(|r| storage.get_metadata_be(r))
            .unwrap()
            .stats;
        assert_eq!((stats.node_count, stats.edge_count), (2, 1));
    }

    /// A delete that stops between two write batches (a kill; here only its
    /// first chunk is run) leaves no node without an edge: each batch drops
    /// its edges together with the nodes they orphan. The nodes used to go in
    /// batches of their own after every edge, and a node left behind had no
    /// edge by which a repeat of the request could find it. The repeat removes
    /// what is left.
    #[test]
    fn lsm_delete_cut_short_after_a_chunk_leaves_no_node_without_an_edge() {
        const C: &str = "repo_cut_short";
        let ctx = setup_context_with_config(Config::default().with_lsm_in_memory());
        let storage = ctx.collections.create_collection(C).unwrap();
        let nodes =
            ["a", "b", "c", "d", "keep"].map(|name| make_node(C, name, &format!("src/{name}.rs")));
        let node_ids: Vec<u128> = nodes.iter().map(|node| node.id).collect();
        let edge = |from: &str, to: &str| {
            let (from_path, to_path) = (format!("src/{from}.rs"), format!("src/{to}.rs"));
            make_edge_with_repo_and_paths(C, from, to, &from_path, &to_path, "r")
        };
        let edges = [
            edge("a", "b"),
            edge("a", "c"),
            edge("a", "d"),
            edge("keep", "d"),
        ];
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
        // The nodes that have a row, and those of them without an edge.
        let node_rows = || {
            let r = storage.backend.begin_read().unwrap();
            let with_row: Vec<u128> = node_ids
                .iter()
                .copied()
                .filter(|id| storage.get_node_be(&r, id).is_ok())
                .collect();
            let without_an_edge = storage.nodes_without_adjacency_be(&r, &with_row).unwrap();
            (with_row, without_an_edge)
        };

        // The first chunk of a delete of `src/a.rs`, and nothing after it.
        assert_eq!(
            delete_edges_with_their_orphans(&storage, &[edges[0].id, edges[1].id]).unwrap(),
            (2, 2),
            "the edges to `b` and `c`, and those two nodes"
        );
        assert_eq!(
            node_rows(),
            (vec![node_ids[0], node_ids[3], node_ids[4]], vec![]),
            "`a` keeps its edge to `d`; no node is left without an edge"
        );
        assert_counters_match_the_rows(&storage, "after the first chunk");

        let (status, payload) = post_json(
            &ctx,
            handle_delete_by_paths,
            serde_json::json!({
                "collection": C,
                "paths": ["src/a.rs"],
                "delete_orphaned_nodes": true,
            }),
        );
        assert_eq!(status, 200);
        assert_eq!(
            (
                payload["edges_deleted"].as_u64(),
                payload["nodes_deleted"].as_u64()
            ),
            (Some(1), Some(1)),
            "the repeat: the edge to `d`, and `a`"
        );
        assert_eq!(
            node_rows(),
            (vec![node_ids[3], node_ids[4]], vec![]),
            "`d` and `keep` keep their edge"
        );
        assert_counters_match_the_rows(&storage, "after the repeat");
    }

    /// `HELIX_GRAPH_DELETE_NODES_WITH_EDGES=0` is the delete as it was: the
    /// edges, then the nodes the request resolved, here none. With the
    /// switch on, the batch that drops an edge finds its orphaned ends itself.
    #[test]
    #[serial_test::serial]
    fn lsm_delete_nodes_with_edges_switch_returns_to_the_two_steps() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_budget_env();
        for (setting, nodes_dropped) in [("0", 0), ("off", 0), ("1", 2)] {
            let _switch = EnvVarGuard::set("HELIX_GRAPH_DELETE_NODES_WITH_EDGES", setting);
            let ctx = setup_context_with_config(Config::default().with_lsm_in_memory());
            let collection = format!("repo_switch_{setting}");
            let c = collection.as_str();
            let storage = ctx.collections.create_collection(c).unwrap();
            let edge = make_edge_with_repo_and_paths(c, "a", "b", "src/a.rs", "src/b.rs", "r");
            storage
                .with_write_backend(|w| {
                    storage.upsert_node_be(w, &make_node(c, "a", "src/a.rs"))?;
                    storage.upsert_node_be(w, &make_node(c, "b", "src/b.rs"))?;
                    storage.upsert_edge_be(w, &edge)
                })
                .unwrap();
            assert_eq!(
                delete_edges_in_write_chunks(&storage, &[edge.id], Some(&HashSet::new())).unwrap(),
                (1, nodes_dropped),
                "switch {setting}"
            );
        }
    }

    /// A node whose drop fails (its row does not decode) fails the batch that
    /// drops its chunk's edges and nodes together, which stores nothing, and
    /// then, the edges gone in a batch of their own, the batched drop of the
    /// nodes. The chunk is then dropped one node at a time in a
    /// fresh write batch, as it was before the drop was batched: the failing
    /// node is skipped and not counted, the others are removed and counted
    /// once. A surviving node and edge keep both counters off zero, where a
    /// second decrement would not show. The `path` index is held in its
    /// building state, so every buffered node drop leaves a build-delta row:
    /// a retry on the failed batch would leave two for the node that batch had
    /// already buffered.
    #[test]
    #[serial_test::serial]
    fn lsm_orphan_delete_drops_a_chunk_per_node_when_its_batched_drop_fails() {
        use crate::helix_engine::storage_core::storage_core::PayloadIndexState;

        const C: &str = "repo_bad_node";
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_budget_env();
        let ctx = setup_context_with_config(Config::default().with_lsm_in_memory());
        let storage = ctx.collections.create_collection(C).unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);
        storage
            .create_payload_index("path", PayloadIndexSchema::Keyword)
            .unwrap();

        // The undecodable row sorts between the two good nodes of the chunk,
        // so the batched drop has buffered the first node's deletes when it
        // fails.
        let (first, bad, last, kept_from, kept_to) = (1u128, 2u128, 3u128, 4u128, 5u128);
        let node = |id: u128, name: &str| NodeUpsert {
            id,
            label: "Symbol".into(),
            properties: HashMap::from([
                ("name".into(), Value::String(name.into())),
                ("path".into(), Value::String("src/nodes.rs".into())),
            ]),
        };
        let edge = |id: u128, from_node: u128, to_node: u128, path: &str| EdgeUpsert {
            id,
            label: "CALLS".into(),
            from_node,
            to_node,
            properties: HashMap::from([
                ("from_path".into(), Value::String(path.into())),
                ("to_path".into(), Value::String(path.into())),
            ]),
        };
        storage
            .with_write_backend(|w| {
                for (id, name) in [
                    (first, "first"),
                    (last, "last"),
                    (kept_from, "kept_from"),
                    (kept_to, "kept_to"),
                ] {
                    storage.upsert_node_be(w, &node(id, name))?;
                }
                storage
                    .backend
                    .put(w, Namespace::Nodes, &bad.to_be_bytes(), &[0xFF])
                    .map_err(|e| GraphError::New(e.to_string()))?;
                storage.upsert_edge_be(w, &edge(0xE1, first, bad, "src/a.rs"))?;
                storage.upsert_edge_be(w, &edge(0xE2, last, bad, "src/a.rs"))?;
                storage.upsert_edge_be(w, &edge(0xE3, kept_from, kept_to, "src/keep.rs"))?;
                Ok(())
            })
            .unwrap();
        // From here a node drop also writes a build-delta row for `path`.
        *storage.payload_indices.read().unwrap()["path"]
            .state
            .write()
            .unwrap() = PayloadIndexState::Building {
            job_id: "held-open-by-the-test".into(),
            started_at_millis: 0,
            updated_at_millis: 0,
            cursor: None,
            indexed_nodes: 0,
        };

        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let (status, payload) = metrics::with_local_recorder(&recorder, || {
            post_json(
                &ctx,
                handle_delete_by_paths,
                serde_json::json!({
                    "collection": C,
                    "paths": ["src/a.rs"],
                    "delete_orphaned_nodes": true,
                }),
            )
        });
        assert_eq!(status, 200);
        assert_eq!(payload["edges_deleted"], 2);
        assert_eq!(payload["nodes_deleted"], 2);
        let rendered = handle.render();
        assert!(
            rendered
                .lines()
                .any(|line| line == "helix_graph_delete_batch_fallback_total{step=\"drop\"} 1"),
            "the batched drop of the chunk must have failed, once: {rendered}"
        );

        let r = storage.backend.begin_read().unwrap();
        let has_row = |id: u128| {
            storage
                .backend
                .get_with(&r, Namespace::Nodes, &id.to_be_bytes(), |v| v.is_some())
                .unwrap()
        };
        assert_eq!(
            [first, bad, last, kept_from, kept_to].map(has_row),
            [false, true, false, true, true]
        );
        for name in ["first", "last"] {
            assert!(storage
                .get_nodes_by_multi_index_be(&r, "name", &Value::String(name.into()))
                .unwrap()
                .is_empty());
        }
        let path_index =
            HelixGraphStorage::payload_index_db_name("path", &PayloadIndexSchema::Keyword);
        let build_deltas = |id: u128| {
            let mut rows = 0;
            storage
                .backend
                .scan(
                    &r,
                    Namespace::PayloadIndexBuild(&path_index),
                    KeyRange::prefix(&id.to_be_bytes()),
                    |_, _| {
                        rows += 1;
                        true
                    },
                )
                .unwrap();
            rows
        };
        assert_eq!(
            [first, bad, last].map(build_deltas),
            [1, 0, 1],
            "one build-delta row per dropped node"
        );
        let stats = storage
            .with_read_backend(|r| storage.get_metadata_be(r))
            .unwrap()
            .stats;
        assert_eq!((stats.node_count, stats.edge_count), (2, 1));
    }

    #[test]
    fn native_graph_query_routes_use_lsm_backend_without_lmdb_env() {
        let ctx = setup_context_with_config(Config::default().with_lsm_in_memory());
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
