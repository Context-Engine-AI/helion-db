use std::collections::HashMap;

use heed3::RoTxn;

use crate::helix_engine::storage_core::backend_any::AnyRead;
use crate::helix_engine::storage_core::storage_core::HelixGraphStorage;
use crate::helix_engine::types::GraphError;
use crate::protocol::label_hash::hash_label;

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
    /// Enter this node: mark Gray, push Exit frame, then push child Enter frames.
    Enter(u128),
    /// Exit this node: mark Black.
    Exit(u128),
}

/// Detect cycles reachable from `start_id` following edges with given labels.
/// Returns a list of cycles, each represented as a Vec of node IDs forming the cycle.
/// Limits total cycles found to `limit`.
pub fn detect_cycles(
    storage: &HelixGraphStorage,
    txn: &RoTxn<'_>,
    start_id: u128,
    edge_labels: &[&str],
    limit: usize,
) -> Result<Vec<Vec<u128>>, GraphError> {
    let r = storage.read_view(txn);
    detect_cycles_be(storage, &r, start_id, edge_labels, limit)
}

pub fn detect_cycles_be(
    storage: &HelixGraphStorage,
    r: &AnyRead<'_>,
    start_id: u128,
    edge_labels: &[&str],
    limit: usize,
) -> Result<Vec<Vec<u128>>, GraphError> {
    let label_hashes: Vec<[u8; 4]> = edge_labels.iter().map(|l| hash_label(l, None)).collect();

    let mut color: HashMap<u128, Color> = HashMap::with_capacity(256);
    let mut parent: HashMap<u128, u128> = HashMap::with_capacity(256);
    let mut cycles: Vec<Vec<u128>> = Vec::new();

    // Iterative three-color DFS using explicit Enter/Exit frames.
    // This avoids stack overflow on deep linear chains.
    let mut stack: Vec<Frame> = Vec::with_capacity(256);
    stack.push(Frame::Enter(start_id));

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Exit(node_id) => {
                color.insert(node_id, Color::Black);
            }
            Frame::Enter(node_id) => {
                let current_color = color.get(&node_id).copied().unwrap_or(Color::White);
                if current_color != Color::White {
                    // Already visited (Gray or Black): skip re-entry.
                    continue;
                }

                if cycles.len() >= limit {
                    return Ok(cycles);
                }

                color.insert(node_id, Color::Gray);

                // Push Exit frame so Black is set after all children are processed.
                stack.push(Frame::Exit(node_id));

                // Enumerate neighbors and push child Enter frames.
                // Neighbors are collected first to maintain deterministic ordering
                // (stack reverses push order, so we push in reverse).
                let mut neighbors: Vec<u128> = Vec::new();
                for label_hash in &label_hashes {
                    // Backend-routed adjacency (heed cursor on LMDB, SlateDB
                    // snapshot on LSM). Dup-sorted order preserved.
                    for (neighbor_id, _edge_id) in
                        storage.adjacency_pairs_be(r, node_id, label_hash, true)?
                    {
                        match color.get(&neighbor_id).copied().unwrap_or(Color::White) {
                            Color::White => {
                                neighbors.push(neighbor_id);
                            }
                            Color::Gray => {
                                // Back edge — found a cycle.
                                let cycle = reconstruct_cycle(&parent, node_id, neighbor_id);
                                cycles.push(cycle);
                                if cycles.len() >= limit {
                                    return Ok(cycles);
                                }
                            }
                            Color::Black => {
                                // Already fully explored, skip.
                            }
                        }
                    }
                }

                // Push unvisited neighbors in reverse so the first neighbor is
                // processed first (LIFO).
                for &neighbor_id in neighbors.iter().rev() {
                    // Set parent before pushing so it's recorded when Enter fires.
                    parent.insert(neighbor_id, node_id);
                    stack.push(Frame::Enter(neighbor_id));
                }
            }
        }
    }

    Ok(cycles)
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

    use super::detect_cycles;

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
