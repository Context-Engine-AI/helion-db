use std::collections::{HashMap, HashSet, VecDeque};

use heed3::RoTxn;

use crate::helix_engine::storage_core::backend_any::AnyRead;
use crate::helix_engine::storage_core::storage_core::HelixGraphStorage;
use crate::helix_engine::types::GraphError;
use crate::protocol::items::Node;
use crate::protocol::label_hash::hash_label;
use crate::protocol::value::Value;

/// Result of a BFS traversal: node + the depth at which it was found,
/// plus edge properties from the traversing edge (repo, git_branches, etc.).
#[derive(Debug, Clone)]
pub struct BfsResult {
    pub node: Node,
    pub depth: usize,
    /// Properties from the edge that connected this node to the traversal.
    /// Empty for start nodes.
    pub edge_properties: HashMap<String, Value>,
}

/// Depth-limited BFS following outgoing edges.
/// Used for: callees, transitive_callees, subclasses.
///
/// Traverses `out_edges_db` for each edge label, collecting unique target nodes
/// up to `max_depth` hops, returning at most `limit` results.
/// If `repo_filter` is set, only edges whose `repo` property matches are traversed.
pub fn bfs_forward(
    storage: &HelixGraphStorage,
    txn: &RoTxn<'_>,
    start_ids: &[u128],
    edge_labels: &[&str],
    max_depth: usize,
    limit: usize,
    repo_filter: Option<&str>,
) -> Result<Vec<BfsResult>, GraphError> {
    let r = storage.read_view(txn);
    bfs_forward_be(
        storage,
        &r,
        start_ids,
        edge_labels,
        max_depth,
        limit,
        repo_filter,
    )
}

pub fn bfs_forward_be(
    storage: &HelixGraphStorage,
    r: &AnyRead<'_>,
    start_ids: &[u128],
    edge_labels: &[&str],
    max_depth: usize,
    limit: usize,
    repo_filter: Option<&str>,
) -> Result<Vec<BfsResult>, GraphError> {
    let label_hashes: Vec<[u8; 4]> = edge_labels.iter().map(|l| hash_label(l, None)).collect();
    bfs_inner_be(
        storage,
        r,
        start_ids,
        &label_hashes,
        max_depth,
        limit,
        Direction::Forward,
        repo_filter,
    )
}

/// Depth-limited BFS following incoming edges.
/// Used for: callers, transitive_callers, importers, base_classes.
///
/// Traverses `in_edges_db` for each edge label, collecting unique source nodes.
/// If `repo_filter` is set, only edges whose `repo` property matches are traversed.
pub fn bfs_reverse(
    storage: &HelixGraphStorage,
    txn: &RoTxn<'_>,
    start_ids: &[u128],
    edge_labels: &[&str],
    max_depth: usize,
    limit: usize,
    repo_filter: Option<&str>,
) -> Result<Vec<BfsResult>, GraphError> {
    let r = storage.read_view(txn);
    bfs_reverse_be(
        storage,
        &r,
        start_ids,
        edge_labels,
        max_depth,
        limit,
        repo_filter,
    )
}

pub fn bfs_reverse_be(
    storage: &HelixGraphStorage,
    r: &AnyRead<'_>,
    start_ids: &[u128],
    edge_labels: &[&str],
    max_depth: usize,
    limit: usize,
    repo_filter: Option<&str>,
) -> Result<Vec<BfsResult>, GraphError> {
    let label_hashes: Vec<[u8; 4]> = edge_labels.iter().map(|l| hash_label(l, None)).collect();
    bfs_inner_be(
        storage,
        r,
        start_ids,
        &label_hashes,
        max_depth,
        limit,
        Direction::Reverse,
        repo_filter,
    )
}

/// Impact analysis: BFS reverse across multiple edge types (CALLS + IMPORTS).
/// Returns all nodes that transitively depend on the start nodes.
pub fn bfs_impact(
    storage: &HelixGraphStorage,
    txn: &RoTxn<'_>,
    start_ids: &[u128],
    max_depth: usize,
    limit: usize,
    repo_filter: Option<&str>,
) -> Result<Vec<BfsResult>, GraphError> {
    let r = storage.read_view(txn);
    bfs_impact_be(storage, &r, start_ids, max_depth, limit, repo_filter)
}

pub fn bfs_impact_be(
    storage: &HelixGraphStorage,
    r: &AnyRead<'_>,
    start_ids: &[u128],
    max_depth: usize,
    limit: usize,
    repo_filter: Option<&str>,
) -> Result<Vec<BfsResult>, GraphError> {
    bfs_reverse_be(
        storage,
        r,
        start_ids,
        &["CALLS", "IMPORTS"],
        max_depth,
        limit,
        repo_filter,
    )
}

#[derive(Clone, Copy)]
enum Direction {
    Forward,
    Reverse,
}

fn bfs_inner_be(
    storage: &HelixGraphStorage,
    r: &AnyRead<'_>,
    start_ids: &[u128],
    label_hashes: &[[u8; 4]],
    max_depth: usize,
    limit: usize,
    direction: Direction,
    repo_filter: Option<&str>,
) -> Result<Vec<BfsResult>, GraphError> {
    let mut visited = HashSet::with_capacity(256);
    let mut queue: VecDeque<(u128, usize)> = VecDeque::with_capacity(256);
    let mut results = Vec::with_capacity(limit.min(1024));

    // Per-BFS cache of symbol name -> all node-id variants. CE writes
    // pathless callee Symbols separately from path-resolved definition
    // Symbols (see helix_gateway/api/graph.rs:260-264), so a single
    // logical symbol can have multiple node IDs. Walking incoming edges
    // from only one variant per hop is what made transitive_callers,
    // impact, and cycles return empty past depth 1. Unifying variants
    // here treats them as one node for traversal. Skipped at depth 0
    // (seed is already variant-resolved upstream) and when max_depth==1
    // (no extra hops to compose), so single-hop queries pay zero cost.
    let mut name_cache: HashMap<String, Vec<u128>> = HashMap::new();
    let unify_variants = max_depth > 1;

    // Seed the queue with start nodes
    for &id in start_ids {
        visited.insert(id);
        queue.push_back((id, 0));
    }

    while let Some((current_id, depth)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }

        // Resolve same-name variants (cached). At depth 0 the seed list
        // already contains every variant via resolve_symbol_ids, so skip
        // the redundant lookup.
        let walk_ids: Vec<u128> = if unify_variants && depth > 0 {
            expand_name_variants_be(storage, r, current_id, &mut name_cache, &mut visited)
        } else {
            vec![current_id]
        };

        // For each edge label, stream adjacent nodes (with edge IDs).
        for label_hash in label_hashes {
            let mut hit_limit = false;
            for &walk_id in &walk_ids {
                stream_adjacency_pairs(
                    storage,
                    r,
                    walk_id,
                    label_hash,
                    direction,
                    |neighbor_id, edge_id| {
                        if visited.contains(&neighbor_id) {
                            return Ok(true);
                        }

                        // Look up the edge to get properties (repo, git_branches, etc.)
                        let edge_props = match storage.get_edge_be(r, &edge_id) {
                            Ok(edge) => {
                                // If repo filter is set, skip edges that don't match.
                                if let Some(repo) = repo_filter {
                                    match edge.properties.get("repo") {
                                        Some(Value::String(r)) if r == repo => {}
                                        _ => return Ok(true),
                                    }
                                }
                                edge.properties
                            }
                            Err(_) => {
                                // Edge missing — skip repo filter, return empty props.
                                if repo_filter.is_some() {
                                    return Ok(true);
                                }
                                HashMap::new()
                            }
                        };

                        // Fetch the node
                        match storage.get_node_be(r, &neighbor_id) {
                            Ok(node) => {
                                if !visited.insert(neighbor_id) {
                                    return Ok(true);
                                }
                                results.push(BfsResult {
                                    node,
                                    depth: depth + 1,
                                    edge_properties: edge_props,
                                });
                                if results.len() >= limit {
                                    hit_limit = true;
                                    return Ok(false);
                                }
                                queue.push_back((neighbor_id, depth + 1));
                            }
                            Err(GraphError::NodeNotFound) => {
                                // Dangling edge reference — skip silently.
                                return Ok(true);
                            }
                            Err(e) => return Err(e),
                        }
                        Ok(true)
                    },
                )?;
                if hit_limit {
                    break;
                }
            }
            if hit_limit {
                return Ok(results);
            }
        }
    }

    Ok(results)
}

/// Stream adjacency entries from DUP_SORT database as (node_id, edge_id) pairs.
/// Pack format: [edge_id(16) | node_id(16)] = 32 bytes.
fn stream_adjacency_pairs<F>(
    storage: &HelixGraphStorage,
    r: &AnyRead<'_>,
    node_id: u128,
    label_hash: &[u8; 4],
    direction: Direction,
    mut on_pair: F,
) -> Result<(), GraphError>
where
    F: FnMut(u128, u128) -> Result<bool, GraphError>,
{
    // Backend-routed: byte-identical heed cursor on LMDB, fresh SlateDB snapshot
    // on LSM (so adjacency in S3 is visible and readers can traverse).
    let out = matches!(direction, Direction::Forward);
    for (neighbor_id, edge_id) in storage.adjacency_pairs_be(r, node_id, label_hash, out)? {
        if !on_pair(neighbor_id, edge_id)? {
            break;
        }
    }

    Ok(())
}

const DEFAULT_BFS_VARIANT_FAN_OUT: usize = 32;

fn bfs_variant_fan_out() -> usize {
    std::env::var("HELIX_BFS_VARIANT_FAN_OUT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(DEFAULT_BFS_VARIANT_FAN_OUT)
}

/// Resolve `node_id` to its full set of same-name variants, caching the
/// result by name for the lifetime of one BFS. All variants are inserted
/// into `visited` so subsequent dequeues skip them.
///
/// On any lookup failure we degrade to walking just `node_id`, which
/// preserves the pre-patch behaviour rather than dropping a hop.
fn expand_name_variants_be(
    storage: &HelixGraphStorage,
    r: &AnyRead<'_>,
    node_id: u128,
    name_cache: &mut HashMap<String, Vec<u128>>,
    visited: &mut HashSet<u128>,
) -> Vec<u128> {
    let name = match storage.get_node_be(r, &node_id) {
        Ok(node) => match node.properties.get("name") {
            Some(Value::String(s)) if !s.is_empty() => s.clone(),
            _ => return vec![node_id],
        },
        Err(_) => return vec![node_id],
    };

    let variants = name_cache.entry(name.clone()).or_insert_with(|| {
        match storage.get_nodes_by_multi_index_be(r, "name", &Value::String(name)) {
            Ok(ids) if !ids.is_empty() => ids.into_iter().take(bfs_variant_fan_out()).collect(),
            _ => vec![node_id],
        }
    });

    for &vid in variants.iter() {
        visited.insert(vid);
    }
    variants.clone()
}
