use std::collections::HashMap;
use std::time::Instant;

use heed3::RoTxn;

use super::bfs::name_variants_be;
use crate::helix_engine::storage_core::backend_any::AnyRead;
use crate::helix_engine::storage_core::storage_core::HelixGraphStorage;
use crate::helix_engine::types::GraphError;
use crate::protocol::label_hash::hash_label;
use crate::protocol::value::Value;

/// Three-color DFS cycle detection.
/// WHITE = unvisited, GRAY = in current path, BLACK = fully explored.
#[derive(Clone, Copy, PartialEq)]
enum Color {
    White,
    Gray,
    Black,
}

/// A frame on the explicit DFS stack.
enum Frame {
    /// Enter this node: mark Gray, push Exit frame, then push child Enter
    /// frames. Carries the logical parent so the DFS tree is recorded when
    /// the node is actually entered, not when it is first pushed.
    Enter { id: u128, parent: Option<u128> },
    /// Exit this logical node: mark Black.
    Exit(u128),
}

/// Result of a budgeted cycle search.
#[derive(Debug, Default)]
pub struct CycleSearch {
    pub cycles: Vec<Vec<u128>>,
    /// The visit or time budget ran out before the walk finished, so
    /// `cycles` may be incomplete. Reaching `limit` is not truncation.
    pub truncated: bool,
}

/// Detect cycles reachable from `start_id` following edges with given labels.
/// Returns a list of cycles, each represented as a Vec of node IDs forming the cycle.
/// Limits total cycles found to `limit`. Unscoped and unbounded; see
/// [`detect_cycles_be`] for the repo filter and visit/time budgets.
pub fn detect_cycles(
    storage: &HelixGraphStorage,
    txn: &RoTxn<'_>,
    start_id: u128,
    edge_labels: &[&str],
    limit: usize,
) -> Result<Vec<Vec<u128>>, GraphError> {
    let r = storage.read_view(txn);
    detect_cycles_be(
        storage,
        &r,
        start_id,
        edge_labels,
        limit,
        None,
        usize::MAX,
        None,
    )
    .map(|search| search.cycles)
}

/// Cycle detection over logical symbols.
///
/// CE writes `caller(name, caller_path) -> callee(name, "")`, so one function
/// is split into a pathless stub (incoming calls) and a path-resolved node
/// (outgoing calls). Like BFS, every same-name variant is treated as one
/// logical node: entering a node walks the out-edges of all its variants, and
/// coloring/parent tracking is keyed by the logical representative (the first
/// variant reached). Reported cycles use those representative ids.
///
/// An edge between two variants of one logical node is a self-cycle when it
/// can be a call of the symbol to itself: a literal `x -> x` edge, or CE's
/// recursion shape `A(path) -> A("")` (or two variants with the same path).
/// An edge between two path-resolved variants with different paths links two
/// distinct same-name definitions; merging made it look like a self-loop, so
/// it is ignored. Name merging can still over-merge unrelated same-name
/// symbols (shared with BFS; needs an ingest-side fix).
///
/// `repo_filter` skips edges whose `repo` property does not match (as BFS);
/// a missing edge is skipped, any other edge read error is returned. The walk
/// stops early once `max_visited` logical nodes were entered or `deadline`
/// passed, returning the cycles found so far with `truncated` set.
#[allow(clippy::too_many_arguments)]
pub fn detect_cycles_be(
    storage: &HelixGraphStorage,
    r: &AnyRead<'_>,
    start_id: u128,
    edge_labels: &[&str],
    limit: usize,
    repo_filter: Option<&str>,
    max_visited: usize,
    deadline: Option<Instant>,
) -> Result<CycleSearch, GraphError> {
    let label_hashes: Vec<[u8; 4]> = edge_labels.iter().map(|l| hash_label(l, None)).collect();

    // Keyed by logical representative.
    let mut color: HashMap<u128, Color> = HashMap::with_capacity(256);
    let mut parent: HashMap<u128, u128> = HashMap::with_capacity(256);
    // Variant node id -> logical representative.
    let mut canonical: HashMap<u128, u128> = HashMap::with_capacity(256);
    let mut name_cache: HashMap<String, Vec<u128>> = HashMap::new();
    let mut cycles: Vec<Vec<u128>> = Vec::new();
    let mut entered = 0usize;

    // Iterative three-color DFS using explicit Enter/Exit frames.
    // This avoids stack overflow on deep linear chains.
    let mut stack: Vec<Frame> = Vec::with_capacity(256);
    stack.push(Frame::Enter {
        id: start_id,
        parent: None,
    });

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Exit(rep) => {
                color.insert(rep, Color::Black);
            }
            Frame::Enter { id, parent: from } => {
                let rep = canonical.get(&id).copied().unwrap_or(id);
                if color.get(&rep).copied().unwrap_or(Color::White) != Color::White {
                    // Already visited (Gray or Black): skip re-entry.
                    continue;
                }

                if cycles.len() >= limit {
                    return Ok(CycleSearch {
                        cycles,
                        truncated: false,
                    });
                }
                if entered >= max_visited || deadline.is_some_and(|d| Instant::now() >= d) {
                    tracing::debug!(
                        entered,
                        cycles = cycles.len(),
                        "detect_cycles budget exhausted; returning partial result"
                    );
                    return Ok(CycleSearch {
                        cycles,
                        truncated: true,
                    });
                }
                entered += 1;

                // Claim every unclaimed same-name variant for this logical node.
                // A variant already owned by another representative was (or
                // will be) walked under that one.
                let mut walk_ids = vec![rep];
                canonical.insert(rep, rep);
                for vid in name_variants_be(storage, r, rep, &mut name_cache) {
                    if *canonical.entry(vid).or_insert(rep) == rep && vid != rep {
                        walk_ids.push(vid);
                    }
                }

                color.insert(rep, Color::Gray);
                if let Some(p) = from {
                    parent.insert(rep, p);
                }

                // Push Exit frame so Black is set after all children are processed.
                stack.push(Frame::Exit(rep));

                // Enumerate neighbors and push child Enter frames.
                // Neighbors are collected first to maintain deterministic ordering
                // (stack reverses push order, so we push in reverse).
                let mut neighbors: Vec<u128> = Vec::new();
                for label_hash in &label_hashes {
                    for &walk_id in &walk_ids {
                        // Backend-routed adjacency (heed cursor on LMDB, SlateDB
                        // snapshot on LSM). Dup-sorted order preserved.
                        for (neighbor_id, edge_id) in
                            storage.adjacency_pairs_be(r, walk_id, label_hash, true)?
                        {
                            if let Some(repo) = repo_filter {
                                // Missing edge or non-matching repo: skip, as BFS
                                // does; a real read error must not look like "no
                                // cycle".
                                match storage.get_edge_be(r, &edge_id) {
                                    Ok(edge) => match edge.properties.get("repo") {
                                        Some(Value::String(edge_repo)) if edge_repo == repo => {}
                                        _ => continue,
                                    },
                                    Err(GraphError::EdgeNotFound) => continue,
                                    Err(e) => return Err(e),
                                }
                            }
                            let neighbor_rep =
                                canonical.get(&neighbor_id).copied().unwrap_or(neighbor_id);
                            if neighbor_rep == rep
                                && neighbor_id != walk_id
                                && !is_same_symbol_call(storage, r, walk_id, neighbor_id)?
                            {
                                // Two distinct same-name definitions, merged.
                                continue;
                            }
                            match color.get(&neighbor_rep).copied().unwrap_or(Color::White) {
                                Color::White => {
                                    neighbors.push(neighbor_id);
                                }
                                Color::Gray => {
                                    // Back edge — found a cycle.
                                    let cycle = reconstruct_cycle(&parent, rep, neighbor_rep);
                                    cycles.push(cycle);
                                    if cycles.len() >= limit {
                                        return Ok(CycleSearch {
                                            cycles,
                                            truncated: false,
                                        });
                                    }
                                }
                                Color::Black => {
                                    // Already fully explored, skip.
                                }
                            }
                        }
                    }
                }

                // Push unvisited neighbors in reverse so the first neighbor is
                // processed first (LIFO).
                for &neighbor_id in neighbors.iter().rev() {
                    stack.push(Frame::Enter {
                        id: neighbor_id,
                        parent: Some(rep),
                    });
                }
            }
        }
    }

    Ok(CycleSearch {
        cycles,
        truncated: false,
    })
}

/// `path` property of a symbol node; empty for CE's pathless callee stubs.
fn node_path(storage: &HelixGraphStorage, r: &AnyRead<'_>, id: u128) -> Result<String, GraphError> {
    match storage.get_node_be(r, &id) {
        Ok(node) => Ok(match node.properties.get("path") {
            Some(Value::String(path)) => path.clone(),
            _ => String::new(),
        }),
        Err(GraphError::NodeNotFound) => Ok(String::new()),
        Err(e) => Err(e),
    }
}

/// Whether an edge between two same-name variants can be the symbol calling
/// itself: the callee is a pathless stub (CE's unresolved-callee shape) or
/// both variants carry the same path.
fn is_same_symbol_call(
    storage: &HelixGraphStorage,
    r: &AnyRead<'_>,
    from: u128,
    to: u128,
) -> Result<bool, GraphError> {
    let to_path = node_path(storage, r, to)?;
    Ok(to_path.is_empty() || node_path(storage, r, from)? == to_path)
}

/// Reconstruct cycle path from `back_edge_target` through parent chain to `back_edge_source`.
fn reconstruct_cycle(
    parent: &HashMap<u128, u128>,
    back_edge_source: u128,
    back_edge_target: u128,
) -> Vec<u128> {
    let mut path = vec![back_edge_target];
    let mut current = back_edge_source;

    // Walk parent chain from source back to target
    let mut safety = 0;
    while current != back_edge_target && safety < 10_000 {
        path.push(current);
        match parent.get(&current) {
            Some(&p) => current = p,
            None => break,
        }
        safety += 1;
    }

    path.reverse();
    path
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use tempfile::TempDir;

    use crate::helix_engine::graph_core::config::Config;
    use crate::helix_engine::storage_core::storage_core::HelixGraphStorage;
    use crate::helix_engine::storage_core::upsert::{EdgeUpsert, NodeUpsert};
    use crate::protocol::deterministic_id;
    use crate::protocol::value::Value;

    use std::time::Instant;

    use super::{detect_cycles, detect_cycles_be};

    fn setup() -> (HelixGraphStorage, TempDir) {
        let tmp = TempDir::new().unwrap();
        let config = Config::new(16, 128, 768, 1);
        let storage = HelixGraphStorage::new(tmp.path().to_str().unwrap(), config).unwrap();
        (storage, tmp)
    }

    fn make_node(coll: &str, name: &str) -> NodeUpsert {
        NodeUpsert {
            id: deterministic_id::node_id(coll, "Symbol", name, "test.rs"),
            label: "Symbol".into(),
            properties: HashMap::from([("name".into(), Value::String(name.into()))]),
        }
    }

    fn make_edge(coll: &str, label: &str, from_name: &str, to_name: &str) -> EdgeUpsert {
        EdgeUpsert {
            id: deterministic_id::edge_id(coll, label, from_name, to_name, "test.rs", "test.rs"),
            label: label.into(),
            from_node: deterministic_id::node_id(coll, "Symbol", from_name, "test.rs"),
            to_node: deterministic_id::node_id(coll, "Symbol", to_name, "test.rs"),
            properties: HashMap::new(),
        }
    }

    /// Diamond (A->B, A->C, B->D, C->D) plus a back-edge D->A to form a cycle.
    /// Expected: detect_cycles finds the cycle; diamond itself is not a cycle.
    #[test]
    fn test_diamond_with_cycle() {
        let (storage, _tmp) = setup();
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        for name in &["A", "B", "C", "D"] {
            storage
                .upsert_node(&mut txn, &make_node("dc", name))
                .unwrap();
        }
        // Diamond edges
        storage
            .upsert_edge(&mut txn, &make_edge("dc", "CALLS", "A", "B"))
            .unwrap();
        storage
            .upsert_edge(&mut txn, &make_edge("dc", "CALLS", "A", "C"))
            .unwrap();
        storage
            .upsert_edge(&mut txn, &make_edge("dc", "CALLS", "B", "D"))
            .unwrap();
        storage
            .upsert_edge(&mut txn, &make_edge("dc", "CALLS", "C", "D"))
            .unwrap();
        // Back-edge that closes a cycle
        storage
            .upsert_edge(&mut txn, &make_edge("dc", "CALLS", "D", "A"))
            .unwrap();
        txn.commit().unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let start = deterministic_id::node_id("dc", "Symbol", "A", "test.rs");
        let cycles = detect_cycles(&storage, &txn, start, &["CALLS"], 10).unwrap();
        assert!(
            !cycles.is_empty(),
            "Should detect at least one cycle via D->A back edge"
        );
        // Every cycle must actually contain the back-edge target (A)
        let a_id = deterministic_id::node_id("dc", "Symbol", "A", "test.rs");
        assert!(
            cycles.iter().any(|c| c.contains(&a_id)),
            "At least one cycle should include node A"
        );
    }

    /// Pure diamond (no back edges) should produce zero cycles.
    #[test]
    fn test_diamond_no_cycle() {
        let (storage, _tmp) = setup();
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        for name in &["A", "B", "C", "D"] {
            storage
                .upsert_node(&mut txn, &make_node("dn", name))
                .unwrap();
        }
        storage
            .upsert_edge(&mut txn, &make_edge("dn", "CALLS", "A", "B"))
            .unwrap();
        storage
            .upsert_edge(&mut txn, &make_edge("dn", "CALLS", "A", "C"))
            .unwrap();
        storage
            .upsert_edge(&mut txn, &make_edge("dn", "CALLS", "B", "D"))
            .unwrap();
        storage
            .upsert_edge(&mut txn, &make_edge("dn", "CALLS", "C", "D"))
            .unwrap();
        txn.commit().unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let start = deterministic_id::node_id("dn", "Symbol", "A", "test.rs");
        let cycles = detect_cycles(&storage, &txn, start, &["CALLS"], 10).unwrap();
        assert!(cycles.is_empty(), "Pure diamond DAG should have no cycles");
    }

    fn split_node(coll: &str, name: &str, path: &str) -> NodeUpsert {
        NodeUpsert {
            id: deterministic_id::node_id(coll, "Symbol", name, path),
            label: "Symbol".into(),
            properties: HashMap::from([
                ("name".into(), Value::String(name.into())),
                ("path".into(), Value::String(path.into())),
            ]),
        }
    }

    /// CE edge model: caller(name, caller_path) -> callee(name, "") because
    /// callee_path is usually unresolved.
    fn split_edge(coll: &str, from: (&str, &str), to: &str, repo: &str) -> EdgeUpsert {
        EdgeUpsert {
            id: deterministic_id::edge_id(coll, "CALLS", from.0, to, from.1, ""),
            label: "CALLS".into(),
            from_node: deterministic_id::node_id(coll, "Symbol", from.0, from.1),
            to_node: deterministic_id::node_id(coll, "Symbol", to, ""),
            properties: HashMap::from([("repo".into(), Value::String(repo.into()))]),
        }
    }

    /// A(a.rs) -> B("") and B(b.rs) -> A(""): one logical A <-> B cycle split
    /// across pathless stubs and path-resolved definitions.
    fn build_split_cycle(storage: &HelixGraphStorage, coll: &str, back_repo: &str) {
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        for (name, path) in [("A", "a.rs"), ("A", ""), ("B", "b.rs"), ("B", "")] {
            storage
                .upsert_node(&mut txn, &split_node(coll, name, path))
                .unwrap();
        }
        storage
            .upsert_edge(&mut txn, &split_edge(coll, ("A", "a.rs"), "B", "repo-a"))
            .unwrap();
        storage
            .upsert_edge(&mut txn, &split_edge(coll, ("B", "b.rs"), "A", back_repo))
            .unwrap();
        txn.commit().unwrap();
    }

    #[test]
    fn test_cycle_across_split_name_variants() {
        let (storage, _tmp) = setup();
        build_split_cycle(&storage, "split", "repo-a");

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let start = deterministic_id::node_id("split", "Symbol", "A", "a.rs");
        let cycles = detect_cycles(&storage, &txn, start, &["CALLS"], 10).unwrap();
        assert_eq!(cycles.len(), 1, "logical A -> B -> A cycle: {cycles:?}");
        assert_eq!(cycles[0].len(), 2);
        assert!(cycles[0].contains(&start));
        let b_stub = deterministic_id::node_id("split", "Symbol", "B", "");
        assert!(cycles[0].contains(&b_stub));
    }

    #[test]
    fn test_split_cycle_respects_repo_filter() {
        let (storage, _tmp) = setup();
        build_split_cycle(&storage, "split_repo", "repo-b");

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let r = storage.read_view(&txn);
        let start = deterministic_id::node_id("split_repo", "Symbol", "A", "a.rs");
        let scoped = detect_cycles_be(
            &storage,
            &r,
            start,
            &["CALLS"],
            10,
            Some("repo-a"),
            100,
            None,
        )
        .unwrap();
        assert!(scoped.cycles.is_empty(), "back edge is repo-b: {scoped:?}");
        assert!(!scoped.truncated);
        let unscoped =
            detect_cycles_be(&storage, &r, start, &["CALLS"], 10, None, 100, None).unwrap();
        assert_eq!(unscoped.cycles.len(), 1);
        assert!(!unscoped.truncated);
    }

    #[test]
    fn test_cycle_search_stops_at_visit_and_time_budget() {
        let (storage, _tmp) = setup();
        build_split_cycle(&storage, "split_budget", "repo-a");

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let r = storage.read_view(&txn);
        let start = deterministic_id::node_id("split_budget", "Symbol", "A", "a.rs");
        let capped = detect_cycles_be(&storage, &r, start, &["CALLS"], 10, None, 1, None).unwrap();
        assert!(
            capped.cycles.is_empty(),
            "one visited node cannot close the cycle"
        );
        assert!(capped.truncated, "visit budget exhaustion must be reported");
        let expired = detect_cycles_be(
            &storage,
            &r,
            start,
            &["CALLS"],
            10,
            None,
            100,
            Some(Instant::now()),
        )
        .unwrap();
        assert!(
            expired.cycles.is_empty(),
            "expired deadline stops before the walk"
        );
        assert!(expired.truncated, "deadline exhaustion must be reported");
    }

    #[test]
    fn test_recursion_through_pathless_stub_is_a_self_cycle() {
        // CE writes recursion as A(a.rs) -> A(""): different node ids, one
        // logical symbol. It must be reported, not dropped as a merge artifact.
        let (storage, _tmp) = setup();
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        for (name, path) in [("A", "a.rs"), ("A", "")] {
            storage
                .upsert_node(&mut txn, &split_node("rec", name, path))
                .unwrap();
        }
        storage
            .upsert_edge(&mut txn, &split_edge("rec", ("A", "a.rs"), "A", "repo-a"))
            .unwrap();
        txn.commit().unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let start = deterministic_id::node_id("rec", "Symbol", "A", "a.rs");
        let cycles = detect_cycles(&storage, &txn, start, &["CALLS"], 10).unwrap();
        assert_eq!(cycles, vec![vec![start]], "recursive A must be reported");
    }

    #[test]
    fn test_call_between_distinct_same_name_definitions_is_not_a_self_cycle() {
        // A(a.rs) -> A(b.rs): two path-resolved definitions that only share a
        // name; merging them must not invent a recursion.
        let (storage, _tmp) = setup();
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        for (name, path) in [("A", "a.rs"), ("A", "b.rs")] {
            storage
                .upsert_node(&mut txn, &split_node("dup", name, path))
                .unwrap();
        }
        storage
            .upsert_edge(
                &mut txn,
                &EdgeUpsert {
                    id: deterministic_id::edge_id("dup", "CALLS", "A", "A", "a.rs", "b.rs"),
                    label: "CALLS".into(),
                    from_node: deterministic_id::node_id("dup", "Symbol", "A", "a.rs"),
                    to_node: deterministic_id::node_id("dup", "Symbol", "A", "b.rs"),
                    properties: HashMap::new(),
                },
            )
            .unwrap();
        txn.commit().unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let start = deterministic_id::node_id("dup", "Symbol", "A", "a.rs");
        let cycles = detect_cycles(&storage, &txn, start, &["CALLS"], 10).unwrap();
        assert!(cycles.is_empty(), "{cycles:?}");
    }

    /// A deep linear chain (1000 nodes) must not overflow the stack.
    /// With recursive DFS this would SIGSEGV; with iterative it must return Ok([]).
    #[test]
    fn test_deep_linear_chain_no_crash() {
        let (storage, _tmp) = setup();
        let depth: usize = 1000;

        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let names: Vec<String> = (0..depth).map(|i| format!("N{}", i)).collect();
        for name in &names {
            storage
                .upsert_node(&mut txn, &make_node("deep", name))
                .unwrap();
        }
        for i in 0..depth - 1 {
            storage
                .upsert_edge(
                    &mut txn,
                    &make_edge("deep", "CALLS", &names[i], &names[i + 1]),
                )
                .unwrap();
        }
        txn.commit().unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let start = deterministic_id::node_id("deep", "Symbol", "N0", "test.rs");
        let cycles = detect_cycles(&storage, &txn, start, &["CALLS"], 10).unwrap();
        assert!(
            cycles.is_empty(),
            "Deep linear chain has no cycles and must not crash"
        );
    }
}
