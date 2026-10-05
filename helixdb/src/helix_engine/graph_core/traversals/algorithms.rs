//! Graph algorithms: PageRank, Label Propagation, Shortest Path, Jaccard Similarity.
//!
//! All operate on read-only LMDB transactions and return computed results
//! without mutating storage.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{LazyLock, RwLock};
use std::time::{Duration, Instant};

use crate::helix_engine::storage_core::backend_any::AnyRead;
use crate::helix_engine::storage_core::storage_core::HelixGraphStorage;
use crate::helix_engine::types::GraphError;
use crate::protocol::items::Node;
use crate::protocol::label_hash::hash_label;
use crate::protocol::value::Value;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GraphAlgorithmCacheKey {
    collection: String,
    repo: Option<String>,
    algorithm: &'static str,
    edge_labels: Vec<String>,
    iterations: usize,
    damping_bits: u64,
}

#[derive(Clone)]
enum GraphAlgorithmCacheValue {
    Pagerank(Vec<(u128, f64)>),
    Communities(Vec<(u128, Vec<u128>)>),
}

#[derive(Clone)]
struct GraphAlgorithmCacheEntry {
    generation: u64,
    value: GraphAlgorithmCacheValue,
}

static GRAPH_ALGORITHM_CACHE: LazyLock<
    RwLock<HashMap<GraphAlgorithmCacheKey, GraphAlgorithmCacheEntry>>,
> = LazyLock::new(|| RwLock::new(HashMap::new()));
static GRAPH_ALGORITHM_GENERATIONS: LazyLock<RwLock<HashMap<String, u64>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

fn graph_algorithm_cache_max_entries() -> usize {
    static MAX: LazyLock<usize> = LazyLock::new(|| {
        std::env::var("HELIX_GRAPH_ALGORITHM_CACHE_MAX_ENTRIES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1024)
    });
    *MAX
}

fn normalized_repo(repo: Option<&str>) -> Option<String> {
    repo.map(str::trim)
        .filter(|repo| !repo.is_empty() && *repo != "*")
        .map(ToOwned::to_owned)
}

fn normalized_edge_labels(edge_labels: &[&str]) -> Vec<String> {
    let mut labels: Vec<String> = edge_labels
        .iter()
        .map(|label| (*label).to_string())
        .collect();
    labels.sort();
    labels.dedup();
    labels
}

fn graph_algo_timeout() -> Duration {
    static SECS: LazyLock<u64> = LazyLock::new(|| {
        std::env::var("HELIX_GRAPH_ALGO_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(30)
    });
    Duration::from_secs(*SECS)
}

fn cache_generation(collection: &str) -> u64 {
    GRAPH_ALGORITHM_GENERATIONS
        .read()
        .ok()
        .and_then(|generations| generations.get(collection).copied())
        .unwrap_or(0)
}

/// Invalidate cached whole-graph algorithm results for one collection.
///
/// This is intentionally generation-based instead of eagerly deleting all
/// entries, so write paths stay O(1) even when multiple repo/label variants
/// have been materialized.
pub fn invalidate_algorithm_cache(collection: &str) {
    if collection.trim().is_empty() {
        return;
    }
    if let Ok(mut generations) = GRAPH_ALGORITHM_GENERATIONS.write() {
        let generation = generations.entry(collection.to_string()).or_insert(0);
        *generation = generation.saturating_add(1);
    }
    metrics::counter!(
        "helix_graph_algorithm_cache_invalidated_total",
        "collection" => collection.to_string(),
    )
    .increment(1);
}

fn put_cache_entry(key: GraphAlgorithmCacheKey, entry: GraphAlgorithmCacheEntry) {
    let max_entries = graph_algorithm_cache_max_entries();
    if max_entries == 0 {
        return;
    }
    let Ok(mut cache) = GRAPH_ALGORITHM_CACHE.write() else {
        return;
    };
    if cache.len() >= max_entries {
        cache.clear();
        metrics::counter!("helix_graph_algorithm_cache_cleared_total").increment(1);
    }
    cache.insert(key, entry);
}

pub fn cached_pagerank(
    collection: &str,
    storage: &HelixGraphStorage,
    r: &AnyRead<'_>,
    edge_labels: &[&str],
    iterations: usize,
    damping: f64,
    repo_filter: Option<&str>,
) -> Result<(Vec<(u128, f64)>, bool), GraphError> {
    let repo = normalized_repo(repo_filter);
    let key = GraphAlgorithmCacheKey {
        collection: collection.to_string(),
        repo,
        algorithm: "pagerank",
        edge_labels: normalized_edge_labels(edge_labels),
        iterations,
        damping_bits: damping.to_bits(),
    };
    let generation = cache_generation(collection);
    if graph_algorithm_cache_max_entries() > 0 {
        if let Ok(cache) = GRAPH_ALGORITHM_CACHE.read() {
            if let Some(entry) = cache.get(&key) {
                if entry.generation == generation {
                    if let GraphAlgorithmCacheValue::Pagerank(results) = &entry.value {
                        metrics::counter!(
                            "helix_graph_algorithm_cache_hit_total",
                            "collection" => collection.to_string(),
                            "algorithm" => "pagerank".to_string(),
                        )
                        .increment(1);
                        return Ok((results.clone(), true));
                    }
                }
            }
        }
    }

    metrics::counter!(
        "helix_graph_algorithm_cache_miss_total",
        "collection" => collection.to_string(),
        "algorithm" => "pagerank".to_string(),
    )
    .increment(1);
    let results = pagerank_scoped(storage, r, edge_labels, iterations, damping, repo_filter)?;
    put_cache_entry(
        key,
        GraphAlgorithmCacheEntry {
            generation,
            value: GraphAlgorithmCacheValue::Pagerank(results.clone()),
        },
    );
    Ok((results, false))
}

pub fn cached_label_propagation(
    collection: &str,
    storage: &HelixGraphStorage,
    r: &AnyRead<'_>,
    edge_labels: &[&str],
    max_iterations: usize,
    repo_filter: Option<&str>,
) -> Result<(Vec<(u128, Vec<u128>)>, bool), GraphError> {
    let repo = normalized_repo(repo_filter);
    let key = GraphAlgorithmCacheKey {
        collection: collection.to_string(),
        repo,
        algorithm: "communities",
        edge_labels: normalized_edge_labels(edge_labels),
        iterations: max_iterations,
        damping_bits: 0,
    };
    let generation = cache_generation(collection);
    if graph_algorithm_cache_max_entries() > 0 {
        if let Ok(cache) = GRAPH_ALGORITHM_CACHE.read() {
            if let Some(entry) = cache.get(&key) {
                if entry.generation == generation {
                    if let GraphAlgorithmCacheValue::Communities(results) = &entry.value {
                        metrics::counter!(
                            "helix_graph_algorithm_cache_hit_total",
                            "collection" => collection.to_string(),
                            "algorithm" => "communities".to_string(),
                        )
                        .increment(1);
                        return Ok((results.clone(), true));
                    }
                }
            }
        }
    }

    metrics::counter!(
        "helix_graph_algorithm_cache_miss_total",
        "collection" => collection.to_string(),
        "algorithm" => "communities".to_string(),
    )
    .increment(1);
    let results = label_propagation_scoped(storage, r, edge_labels, max_iterations, repo_filter)?;
    put_cache_entry(
        key,
        GraphAlgorithmCacheEntry {
            generation,
            value: GraphAlgorithmCacheValue::Communities(results.clone()),
        },
    );
    Ok((results, false))
}

// ── helpers ──────────────────────────────────────────────────────────────

/// Wall-clock budget for one whole-graph algorithm run. Checked while
/// processing the node/edge scans and during adjacency precompute as well as
/// the iteration loop, since on large collections the pre-iteration work alone
/// can exceed the timeout. (The `scan_all_*_be` calls materialize their
/// results, so a single scan call itself cannot be interrupted.)
#[derive(Clone, Copy)]
struct AlgoDeadline {
    algorithm: &'static str,
    started: Instant,
    timeout: Duration,
}

impl AlgoDeadline {
    fn start(algorithm: &'static str) -> Self {
        Self {
            algorithm,
            started: Instant::now(),
            timeout: graph_algo_timeout(),
        }
    }

    fn check(&self, phase: &str) -> Result<(), GraphError> {
        if self.started.elapsed() > self.timeout {
            return Err(GraphError::New(format!(
                "{} timed out after {}s during {}",
                self.algorithm,
                self.timeout.as_secs(),
                phase,
            )));
        }
        Ok(())
    }
}

/// Scan loops check the deadline every this many items.
const DEADLINE_CHECK_STRIDE: usize = 1024;

fn value_matches_repo(value: Option<&Value>, repo: &str) -> bool {
    match value {
        Some(Value::String(value)) => value == repo,
        Some(Value::Array(values)) => values
            .iter()
            .any(|value| value_matches_repo(Some(value), repo)),
        _ => false,
    }
}

fn properties_match_repo(properties: &HashMap<String, Value>, repo: &str) -> bool {
    value_matches_repo(properties.get("repo"), repo)
}

/// Node ids in scope, sorted ascending so downstream iteration order (and
/// therefore results) is deterministic.
fn all_node_ids_scoped(
    storage: &HelixGraphStorage,
    r: &AnyRead<'_>,
    repo_filter: Option<&str>,
    deadline: &AlgoDeadline,
) -> Result<Vec<u128>, GraphError> {
    let Some(repo) = repo_filter else {
        let mut ids = Vec::new();
        for (i, node) in storage.scan_all_nodes_be(r)?.into_iter().enumerate() {
            if i % DEADLINE_CHECK_STRIDE == 0 {
                deadline.check("node scan")?;
            }
            ids.push(node?.id);
        }
        ids.sort_unstable();
        return Ok(ids);
    };

    let mut ids = HashSet::new();
    for (i, node) in storage.scan_all_nodes_be(r)?.into_iter().enumerate() {
        if i % DEADLINE_CHECK_STRIDE == 0 {
            deadline.check("node scan")?;
        }
        let node = node?;
        if properties_match_repo(&node.properties, repo) {
            ids.insert(node.id);
        }
    }
    for (i, edge) in storage.scan_all_edges_be(r)?.into_iter().enumerate() {
        if i % DEADLINE_CHECK_STRIDE == 0 {
            deadline.check("edge scan")?;
        }
        let edge = edge?;
        if properties_match_repo(&edge.properties, repo) {
            ids.insert(edge.from_node);
            ids.insert(edge.to_node);
        }
    }

    let mut ids: Vec<u128> = ids.into_iter().collect();
    ids.sort_unstable();
    Ok(ids)
}

/// Outgoing neighbor IDs for a node across multiple edge labels.
///
/// `unpack_adj_edge_data` returns `(edge_id, node_id)` — we want `node_id`.
fn out_neighbors(
    storage: &HelixGraphStorage,
    r: &AnyRead<'_>,
    node_id: u128,
    label_hashes: &[[u8; 4]],
) -> Result<Vec<u128>, GraphError> {
    out_neighbors_scoped(storage, r, node_id, label_hashes, None)
}

fn out_neighbors_scoped(
    storage: &HelixGraphStorage,
    r: &AnyRead<'_>,
    node_id: u128,
    label_hashes: &[[u8; 4]],
    repo_filter: Option<&str>,
) -> Result<Vec<u128>, GraphError> {
    let mut neighbors = Vec::new();
    for lh in label_hashes {
        // Backend-routed adjacency (heed cursor on LMDB, SlateDB snapshot on LSM).
        for (neighbor_id, edge_id) in storage.adjacency_pairs_be(r, node_id, lh, true)? {
            if let Some(repo) = repo_filter {
                // A dangling adjacency entry (edge already dropped) is skipped
                // rather than failing the whole algorithm run.
                let edge = match storage.get_edge_be(r, &edge_id) {
                    Ok(edge) => edge,
                    Err(GraphError::EdgeNotFound) => continue,
                    Err(e) => return Err(e),
                };
                if !properties_match_repo(&edge.properties, repo) {
                    continue;
                }
            }
            neighbors.push(neighbor_id);
        }
    }
    Ok(neighbors)
}

/// Incoming neighbor IDs for a node across multiple edge labels.
fn in_neighbors_scoped(
    storage: &HelixGraphStorage,
    r: &AnyRead<'_>,
    node_id: u128,
    label_hashes: &[[u8; 4]],
    repo_filter: Option<&str>,
) -> Result<Vec<u128>, GraphError> {
    let mut neighbors = Vec::new();
    for lh in label_hashes {
        // Backend-routed adjacency (heed cursor on LMDB, SlateDB snapshot on LSM).
        for (neighbor_id, edge_id) in storage.adjacency_pairs_be(r, node_id, lh, false)? {
            if let Some(repo) = repo_filter {
                // A dangling adjacency entry (edge already dropped) is skipped
                // rather than failing the whole algorithm run.
                let edge = match storage.get_edge_be(r, &edge_id) {
                    Ok(edge) => edge,
                    Err(GraphError::EdgeNotFound) => continue,
                    Err(e) => return Err(e),
                };
                if !properties_match_repo(&edge.properties, repo) {
                    continue;
                }
            }
            neighbors.push(neighbor_id);
        }
    }
    Ok(neighbors)
}

// ── PageRank ─────────────────────────────────────────────────────────────

/// Iterative PageRank over nodes connected by `edge_labels`.
///
/// Returns a map of node_id → rank. The `damping` factor is typically 0.85.
/// Runs for `iterations` rounds (default 20 is fine for most graphs).
pub fn pagerank(
    storage: &HelixGraphStorage,
    r: &AnyRead<'_>,
    edge_labels: &[&str],
    iterations: usize,
    damping: f64,
) -> Result<Vec<(u128, f64)>, GraphError> {
    pagerank_scoped(storage, r, edge_labels, iterations, damping, None)
}

pub fn pagerank_scoped(
    storage: &HelixGraphStorage,
    r: &AnyRead<'_>,
    edge_labels: &[&str],
    iterations: usize,
    damping: f64,
    repo_filter: Option<&str>,
) -> Result<Vec<(u128, f64)>, GraphError> {
    let deadline = AlgoDeadline::start("pagerank");

    let label_hashes: Vec<[u8; 4]> = edge_labels.iter().map(|l| hash_label(l, None)).collect();
    let nodes = all_node_ids_scoped(storage, r, repo_filter, &deadline)?;
    let n = nodes.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let node_idx: HashMap<u128, usize> = nodes.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let init = 1.0 / n as f64;
    let mut rank = vec![init; n];
    let mut new_rank = vec![0.0f64; n];

    // Pre-compute out-degree for each node
    let mut out_degree = vec![0usize; n];
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n]; // incoming edges by index

    for (i, &nid) in nodes.iter().enumerate() {
        deadline.check("adjacency precompute")?;
        let outs = out_neighbors_scoped(storage, r, nid, &label_hashes, repo_filter)?;
        out_degree[i] = outs.len();
        for &target in &outs {
            if let Some(&j) = node_idx.get(&target) {
                adjacency[j].push(i); // j receives from i
            }
        }
    }

    let teleport = (1.0 - damping) / n as f64;

    for iter in 0..iterations {
        deadline.check(&format!("iteration {} of {}", iter, iterations))?;

        // Dangling mass: nodes with no out-edges redistribute rank uniformly
        let dangling: f64 = nodes
            .iter()
            .enumerate()
            .filter(|(i, _)| out_degree[*i] == 0)
            .map(|(i, _)| rank[i])
            .sum();
        let dangling_share = damping * dangling / n as f64;

        for j in 0..n {
            let mut incoming_rank = 0.0;
            for &i in &adjacency[j] {
                incoming_rank += rank[i] / out_degree[i] as f64;
            }
            new_rank[j] = teleport + dangling_share + damping * incoming_rank;
        }
        std::mem::swap(&mut rank, &mut new_rank);
    }

    let mut results: Vec<(u128, f64)> = nodes.into_iter().zip(rank).collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(results)
}

/// Label Propagation Algorithm for community detection.
///
/// Each node starts with its own label (community). In each iteration,
/// every node adopts the most frequent label among its neighbors (both
/// directions). Converges when no labels change.
///
/// Returns communities as `Vec<(community_id, Vec<node_id>)>`.
pub fn label_propagation(
    storage: &HelixGraphStorage,
    r: &AnyRead<'_>,
    edge_labels: &[&str],
    max_iterations: usize,
) -> Result<Vec<(u128, Vec<u128>)>, GraphError> {
    label_propagation_scoped(storage, r, edge_labels, max_iterations, None)
}

pub fn label_propagation_scoped(
    storage: &HelixGraphStorage,
    r: &AnyRead<'_>,
    edge_labels: &[&str],
    max_iterations: usize,
    repo_filter: Option<&str>,
) -> Result<Vec<(u128, Vec<u128>)>, GraphError> {
    let deadline = AlgoDeadline::start("label propagation");

    let label_hashes: Vec<[u8; 4]> = edge_labels.iter().map(|l| hash_label(l, None)).collect();
    // Sorted, so the in-place (asynchronous) label updates below run in a
    // fixed order.
    let nodes = all_node_ids_scoped(storage, r, repo_filter, &deadline)?;
    if nodes.is_empty() {
        return Ok(Vec::new());
    }

    // Pre-compute adjacency (both directions) for all nodes once.
    // This avoids re-reading LMDB on every node every iteration.
    let mut neighbors: HashMap<u128, Vec<u128>> = HashMap::with_capacity(nodes.len());
    for &nid in &nodes {
        deadline.check("adjacency precompute")?;
        let outs = out_neighbors_scoped(storage, r, nid, &label_hashes, repo_filter)?;
        let ins = in_neighbors_scoped(storage, r, nid, &label_hashes, repo_filter)?;
        let mut combined = outs;
        combined.extend_from_slice(&ins);
        neighbors.insert(nid, combined);
    }

    // Each node starts as its own community
    let mut labels: HashMap<u128, u128> = nodes.iter().map(|&id| (id, id)).collect();

    for iter in 0..max_iterations {
        deadline.check(&format!("iteration {} of {}", iter, max_iterations))?;

        let mut changed = false;

        for &nid in &nodes {
            // Collect neighbor labels (both directions for undirected community)
            let mut label_counts: HashMap<u128, usize> = HashMap::new();

            if let Some(nbrs) = neighbors.get(&nid) {
                for &neighbor in nbrs {
                    if let Some(&lbl) = labels.get(&neighbor) {
                        *label_counts.entry(lbl).or_insert(0) += 1;
                    }
                }
            }

            // Deterministic tie-break: keep the current label if it is among
            // the most frequent, otherwise take the smallest label id. Keeping
            // the current label on ties is what lets the loop converge.
            let Some(&max_count) = label_counts.values().max() else {
                continue;
            };
            let current = labels[&nid];
            if label_counts.get(&current) == Some(&max_count) {
                continue;
            }
            let best_label = label_counts
                .iter()
                .filter(|(_, &count)| count == max_count)
                .map(|(&label, _)| label)
                .min()
                .unwrap_or(current);
            if best_label != current {
                labels.insert(nid, best_label);
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    // Group nodes by community label
    let mut communities: HashMap<u128, Vec<u128>> = HashMap::new();
    for (&nid, &lbl) in &labels {
        communities.entry(lbl).or_default().push(nid);
    }

    let mut result: Vec<(u128, Vec<u128>)> = communities
        .into_iter()
        .map(|(label, mut members)| {
            members.sort_unstable();
            (label, members)
        })
        .collect();
    // Largest communities first; equal sizes by ascending community id.
    result.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(&b.0)));
    Ok(result)
}

// ── Shortest Path (BFS) ─────────────────────────────────────────────────

/// BFS shortest path between two nodes following `edge_labels`.
///
/// Returns `(path_nodes, distance)`. If no path exists, returns empty vec and 0.
/// Search is bounded by `max_depth` hops and `max_visited` visited nodes.
pub fn shortest_path(
    storage: &HelixGraphStorage,
    r: &AnyRead<'_>,
    from_id: u128,
    to_id: u128,
    edge_labels: &[&str],
    max_depth: usize,
    max_visited: usize,
) -> Result<(Vec<Node>, usize), GraphError> {
    if from_id == to_id {
        let node = storage.get_node_be(r, &from_id)?;
        return Ok((vec![node], 0));
    }

    let max_depth = max_depth.max(1);
    let max_visited = max_visited.max(1);
    let label_hashes: Vec<[u8; 4]> = edge_labels.iter().map(|l| hash_label(l, None)).collect();
    let mut visited: HashSet<u128> = HashSet::new();
    let mut parent: HashMap<u128, u128> = HashMap::new();
    let mut queue: VecDeque<(u128, usize)> = VecDeque::new();

    visited.insert(from_id);
    queue.push_back((from_id, 0));

    let mut found = false;
    while let Some((current, depth)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }
        let neighbors = out_neighbors(storage, r, current, &label_hashes)?;
        for neighbor in neighbors {
            if visited.contains(&neighbor) {
                continue;
            }
            // Reaching the target never needs budget for further expansion,
            // so test it before the visited cap.
            if neighbor == to_id {
                visited.insert(neighbor);
                parent.insert(neighbor, current);
                found = true;
                break;
            }
            if visited.len() >= max_visited {
                return Ok((Vec::new(), 0));
            }
            visited.insert(neighbor);
            parent.insert(neighbor, current);
            queue.push_back((neighbor, depth + 1));
        }
        if found {
            break;
        }
    }

    if !found {
        return Ok((Vec::new(), 0));
    }

    // Reconstruct path
    let mut path_ids = vec![to_id];
    let mut current = to_id;
    while let Some(&prev) = parent.get(&current) {
        path_ids.push(prev);
        current = prev;
    }
    path_ids.reverse();

    let distance = path_ids.len() - 1;
    let mut path_nodes = Vec::with_capacity(path_ids.len());
    for id in &path_ids {
        match storage.get_node_be(r, id) {
            Ok(node) => path_nodes.push(node),
            Err(GraphError::NodeNotFound) => continue, // dangling
            Err(e) => return Err(e),
        }
    }

    Ok((path_nodes, distance))
}

// ── Jaccard Similarity ──────────────────────────────────────────────────

/// Jaccard similarity between two nodes based on shared neighbors.
///
/// J(A, B) = |neighbors(A) ∩ neighbors(B)| / |neighbors(A) ∪ neighbors(B)|
/// Uses outgoing edges only. Returns 0.0 if both neighbor sets are empty.
pub fn jaccard_similarity(
    storage: &HelixGraphStorage,
    r: &AnyRead<'_>,
    node_a: u128,
    node_b: u128,
    edge_labels: &[&str],
) -> Result<f64, GraphError> {
    let label_hashes: Vec<[u8; 4]> = edge_labels.iter().map(|l| hash_label(l, None)).collect();

    let neighbors_a: HashSet<u128> = out_neighbors(storage, r, node_a, &label_hashes)?
        .into_iter()
        .collect();
    let neighbors_b: HashSet<u128> = out_neighbors(storage, r, node_b, &label_hashes)?
        .into_iter()
        .collect();

    let intersection = neighbors_a.intersection(&neighbors_b).count();
    let union = neighbors_a.union(&neighbors_b).count();

    if union == 0 {
        return Ok(0.0);
    }

    Ok(intersection as f64 / union as f64)
}
