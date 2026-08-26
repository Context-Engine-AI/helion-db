use std::collections::HashMap;
use tempfile::TempDir;

use crate::helix_engine::graph_core::config::Config;
use crate::helix_engine::storage_core::storage_core::HelixGraphStorage;
use crate::helix_engine::storage_core::upsert::{EdgeUpsert, NodeUpsert};
use crate::protocol::deterministic_id;
use crate::protocol::value::Value;

use super::bfs::{bfs_forward, bfs_impact, bfs_reverse};
use super::cycles::detect_cycles;

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

fn make_edge_with_repo(
    coll: &str,
    label: &str,
    from_name: &str,
    to_name: &str,
    repo: &str,
) -> EdgeUpsert {
    let mut edge = make_edge(coll, label, from_name, to_name);
    edge.properties
        .insert("repo".into(), Value::String(repo.into()));
    edge
}

/// Build a simple call chain: A -> B -> C -> D
fn build_chain(storage: &HelixGraphStorage) {
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
    for name in &["A", "B", "C", "D"] {
        storage
            .upsert_node(&mut txn, &make_node("test", name))
            .unwrap();
    }
    storage
        .upsert_edge(&mut txn, &make_edge("test", "CALLS", "A", "B"))
        .unwrap();
    storage
        .upsert_edge(&mut txn, &make_edge("test", "CALLS", "B", "C"))
        .unwrap();
    storage
        .upsert_edge(&mut txn, &make_edge("test", "CALLS", "C", "D"))
        .unwrap();
    txn.commit().unwrap();
}

/// Build a diamond: A -> B, A -> C, B -> D, C -> D
fn build_diamond(storage: &HelixGraphStorage) {
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
    for name in &["A", "B", "C", "D"] {
        storage
            .upsert_node(&mut txn, &make_node("test", name))
            .unwrap();
    }
    storage
        .upsert_edge(&mut txn, &make_edge("test", "CALLS", "A", "B"))
        .unwrap();
    storage
        .upsert_edge(&mut txn, &make_edge("test", "CALLS", "A", "C"))
        .unwrap();
    storage
        .upsert_edge(&mut txn, &make_edge("test", "CALLS", "B", "D"))
        .unwrap();
    storage
        .upsert_edge(&mut txn, &make_edge("test", "CALLS", "C", "D"))
        .unwrap();
    txn.commit().unwrap();
}

#[test]
fn test_bfs_forward_single_hop() {
    let (storage, _tmp) = setup();
    build_chain(&storage);

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let start = deterministic_id::node_id("test", "Symbol", "A", "test.rs");
    let results = bfs_forward(&storage, &txn, &[start], &["CALLS"], 1, 100, None).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].node.properties.get("name"),
        Some(&Value::String("B".into()))
    );
    assert_eq!(results[0].depth, 1);
}

#[test]
fn test_bfs_forward_multi_hop() {
    let (storage, _tmp) = setup();
    build_chain(&storage);

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let start = deterministic_id::node_id("test", "Symbol", "A", "test.rs");
    let results = bfs_forward(&storage, &txn, &[start], &["CALLS"], 3, 100, None).unwrap();

    assert_eq!(results.len(), 3); // B, C, D
    let names: Vec<_> = results
        .iter()
        .map(|r| r.node.properties.get("name").unwrap().clone())
        .collect();
    assert!(names.contains(&Value::String("B".into())));
    assert!(names.contains(&Value::String("C".into())));
    assert!(names.contains(&Value::String("D".into())));
}

#[test]
fn test_bfs_reverse_single_hop() {
    let (storage, _tmp) = setup();
    build_chain(&storage);

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let start = deterministic_id::node_id("test", "Symbol", "D", "test.rs");
    let results = bfs_reverse(&storage, &txn, &[start], &["CALLS"], 1, 100, None).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].node.properties.get("name"),
        Some(&Value::String("C".into()))
    );
}

#[test]
fn test_bfs_reverse_transitive() {
    let (storage, _tmp) = setup();
    build_chain(&storage);

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let start = deterministic_id::node_id("test", "Symbol", "D", "test.rs");
    let results = bfs_reverse(&storage, &txn, &[start], &["CALLS"], 10, 100, None).unwrap();

    assert_eq!(results.len(), 3); // C, B, A
}

#[test]
fn test_bfs_diamond_no_duplicates() {
    let (storage, _tmp) = setup();
    build_diamond(&storage);

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let start = deterministic_id::node_id("test", "Symbol", "A", "test.rs");
    let results = bfs_forward(&storage, &txn, &[start], &["CALLS"], 3, 100, None).unwrap();

    // B, C at depth 1, D at depth 2 — D should only appear once
    assert_eq!(results.len(), 3);
    let d_count = results
        .iter()
        .filter(|r| r.node.properties.get("name") == Some(&Value::String("D".into())))
        .count();
    assert_eq!(d_count, 1, "Diamond should not produce duplicate D");
}

#[test]
fn test_bfs_limit() {
    let (storage, _tmp) = setup();
    build_chain(&storage);

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let start = deterministic_id::node_id("test", "Symbol", "A", "test.rs");
    let results = bfs_forward(&storage, &txn, &[start], &["CALLS"], 10, 2, None).unwrap();

    assert_eq!(results.len(), 2); // Only first 2 results
}

#[test]
fn test_bfs_repo_filter_does_not_mark_wrong_repo_edge_visited() {
    let (storage, _tmp) = setup();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    for name in &["A", "B"] {
        storage
            .upsert_node(&mut txn, &make_node("test", name))
            .unwrap();
    }
    storage
        .upsert_edge(
            &mut txn,
            &make_edge_with_repo("test", "IMPORTS", "A", "B", "repo-b"),
        )
        .unwrap();
    storage
        .upsert_edge(
            &mut txn,
            &make_edge_with_repo("test", "CALLS", "A", "B", "repo-a"),
        )
        .unwrap();
    txn.commit().unwrap();

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let start = deterministic_id::node_id("test", "Symbol", "A", "test.rs");
    let results = bfs_forward(
        &storage,
        &txn,
        &[start],
        &["IMPORTS", "CALLS"],
        1,
        100,
        Some("repo-a"),
    )
    .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].node.properties.get("name"),
        Some(&Value::String("B".into()))
    );
    assert_eq!(
        results[0].edge_properties.get("repo"),
        Some(&Value::String("repo-a".into()))
    );
}

#[test]
fn test_bfs_impact() {
    let (storage, _tmp) = setup();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    // A -CALLS-> B, C -IMPORTS-> B
    for name in &["A", "B", "C"] {
        storage
            .upsert_node(&mut txn, &make_node("test", name))
            .unwrap();
    }
    storage
        .upsert_edge(&mut txn, &make_edge("test", "CALLS", "A", "B"))
        .unwrap();
    storage
        .upsert_edge(&mut txn, &make_edge("test", "IMPORTS", "C", "B"))
        .unwrap();
    txn.commit().unwrap();

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let start = deterministic_id::node_id("test", "Symbol", "B", "test.rs");
    let results = bfs_impact(&storage, &txn, &[start], 3, 100, None).unwrap();

    // Both A (caller) and C (importer) should be found
    assert_eq!(results.len(), 2);
    let names: Vec<_> = results
        .iter()
        .map(|r| r.node.properties.get("name").unwrap().clone())
        .collect();
    assert!(names.contains(&Value::String("A".into())));
    assert!(names.contains(&Value::String("C".into())));
}

#[test]
fn test_cycle_detection() {
    let (storage, _tmp) = setup();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    // A -> B -> C -> A (cycle)
    for name in &["A", "B", "C"] {
        storage
            .upsert_node(&mut txn, &make_node("test", name))
            .unwrap();
    }
    storage
        .upsert_edge(&mut txn, &make_edge("test", "CALLS", "A", "B"))
        .unwrap();
    storage
        .upsert_edge(&mut txn, &make_edge("test", "CALLS", "B", "C"))
        .unwrap();
    storage
        .upsert_edge(&mut txn, &make_edge("test", "CALLS", "C", "A"))
        .unwrap();
    txn.commit().unwrap();

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let start = deterministic_id::node_id("test", "Symbol", "A", "test.rs");
    let cycles = detect_cycles(&storage, &txn, start, &["CALLS"], 10).unwrap();

    assert!(!cycles.is_empty(), "Should detect the A->B->C->A cycle");
    assert_eq!(cycles[0].len(), 3); // A, B, C
}

#[test]
fn test_no_cycles_in_dag() {
    let (storage, _tmp) = setup();
    build_chain(&storage); // A -> B -> C -> D (no cycle)

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let start = deterministic_id::node_id("test", "Symbol", "A", "test.rs");
    let cycles = detect_cycles(&storage, &txn, start, &["CALLS"], 10).unwrap();

    assert!(cycles.is_empty(), "DAG should have no cycles");
}
