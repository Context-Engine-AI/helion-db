use std::collections::{HashMap, HashSet};

use tempfile::TempDir;

use crate::helix_engine::graph_core::config::Config;
use crate::helix_engine::storage_core::storage_core::HelixGraphStorage;
use crate::helix_engine::storage_core::upsert::{EdgeUpsert, NodeUpsert};
use crate::protocol::deterministic_id;
use crate::protocol::value::Value;

use super::algorithms::{
    cached_pagerank, invalidate_algorithm_cache, jaccard_similarity, label_propagation,
    label_propagation_scoped, pagerank, pagerank_scoped, shortest_path,
};

const COLLECTION: &str = "test";
const PATH: &str = "test.rs";

fn setup() -> (HelixGraphStorage, TempDir) {
    let tmp = TempDir::new().unwrap();
    let config = Config::new(16, 128, 768, 1);
    let storage = HelixGraphStorage::new(tmp.path().to_str().unwrap(), config).unwrap();
    (storage, tmp)
}

fn node_id(name: &str) -> u128 {
    deterministic_id::node_id(COLLECTION, "Symbol", name, PATH)
}

fn make_node(name: &str) -> NodeUpsert {
    NodeUpsert {
        id: node_id(name),
        label: "Symbol".into(),
        properties: HashMap::from([("name".into(), Value::String(name.into()))]),
    }
}

fn make_edge(label: &str, from_name: &str, to_name: &str) -> EdgeUpsert {
    EdgeUpsert {
        id: deterministic_id::edge_id(COLLECTION, label, from_name, to_name, PATH, PATH),
        label: label.into(),
        from_node: node_id(from_name),
        to_node: node_id(to_name),
        properties: HashMap::new(),
    }
}

fn make_edge_with_repo(label: &str, from_name: &str, to_name: &str, repo: &str) -> EdgeUpsert {
    EdgeUpsert {
        id: deterministic_id::edge_id(COLLECTION, label, from_name, to_name, PATH, PATH),
        label: label.into(),
        from_node: node_id(from_name),
        to_node: node_id(to_name),
        properties: HashMap::from([("repo".into(), Value::String(repo.into()))]),
    }
}

fn build_graph(storage: &HelixGraphStorage, nodes: &[&str], edges: &[(&str, &str, &str)]) {
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
    for name in nodes {
        storage.upsert_node(&mut txn, &make_node(name)).unwrap();
    }
    for (label, from, to) in edges {
        storage
            .upsert_edge(&mut txn, &make_edge(label, from, to))
            .unwrap();
    }
    txn.commit().unwrap();
}

#[test]
fn pagerank_orders_heavily_referenced_node_first() {
    let (storage, _tmp) = setup();
    build_graph(
        &storage,
        &["A", "B", "C", "D"],
        &[
            ("CALLS", "A", "B"),
            ("CALLS", "C", "B"),
            ("CALLS", "D", "B"),
        ],
    );

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let r = storage.read_view(&txn);
    let ranks = pagerank(&storage, &r, &["CALLS"], 20, 0.85).unwrap();

    assert_eq!(ranks.first().map(|(id, _)| *id), Some(node_id("B")));
}

#[test]
fn cached_pagerank_hits_until_collection_invalidates() {
    let (storage, _tmp) = setup();
    build_graph(
        &storage,
        &["A", "B", "C"],
        &[("CALLS", "A", "B"), ("CALLS", "C", "B")],
    );

    let collection = "cache-test-pagerank";
    invalidate_algorithm_cache(collection);
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let r = storage.read_view(&txn);
    let (_, was_cached) =
        cached_pagerank(collection, &storage, &r, &["CALLS"], 20, 0.85, None).unwrap();
    assert!(!was_cached);

    let (_, was_cached) =
        cached_pagerank(collection, &storage, &r, &["CALLS"], 20, 0.85, None).unwrap();
    assert!(was_cached);

    invalidate_algorithm_cache(collection);
    let (_, was_cached) =
        cached_pagerank(collection, &storage, &r, &["CALLS"], 20, 0.85, None).unwrap();
    assert!(!was_cached);
}

#[test]
fn pagerank_scoped_to_repo_only_materializes_matching_edges() {
    let (storage, _tmp) = setup();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
    for name in ["A", "B", "C", "D"] {
        storage.upsert_node(&mut txn, &make_node(name)).unwrap();
    }
    storage
        .upsert_edge(&mut txn, &make_edge_with_repo("CALLS", "A", "B", "repo-a"))
        .unwrap();
    storage
        .upsert_edge(&mut txn, &make_edge_with_repo("CALLS", "C", "D", "repo-b"))
        .unwrap();
    txn.commit().unwrap();

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let r = storage.read_view(&txn);
    let ranks = pagerank_scoped(&storage, &r, &["CALLS"], 20, 0.85, Some("repo-a")).unwrap();
    let ids: HashSet<u128> = ranks.into_iter().map(|(id, _)| id).collect();

    assert_eq!(ids, HashSet::from([node_id("A"), node_id("B")]));
}

#[test]
fn label_propagation_keeps_disconnected_components_separate() {
    let (storage, _tmp) = setup();
    build_graph(
        &storage,
        &["A", "B", "C", "D"],
        &[
            ("CALLS", "A", "B"),
            ("CALLS", "B", "A"),
            ("CALLS", "C", "D"),
            ("CALLS", "D", "C"),
        ],
    );

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let r = storage.read_view(&txn);
    let communities = label_propagation(&storage, &r, &["CALLS"], 10).unwrap();
    let member_sets: Vec<HashSet<u128>> = communities
        .into_iter()
        .map(|(_, members)| members.into_iter().collect())
        .collect();

    assert!(member_sets
        .iter()
        .any(|members| members == &HashSet::from([node_id("A"), node_id("B")])));
    assert!(member_sets
        .iter()
        .any(|members| members == &HashSet::from([node_id("C"), node_id("D")])));
}

#[test]
fn label_propagation_is_deterministic_and_orders_ties_by_id() {
    let (storage, _tmp) = setup();
    // A 6-ring (every node ties between two neighbor labels) plus two
    // equal-size disconnected pairs.
    build_graph(
        &storage,
        &["R0", "R1", "R2", "R3", "R4", "R5", "P", "Q", "X", "Y"],
        &[
            ("CALLS", "R0", "R1"),
            ("CALLS", "R1", "R2"),
            ("CALLS", "R2", "R3"),
            ("CALLS", "R3", "R4"),
            ("CALLS", "R4", "R5"),
            ("CALLS", "R5", "R0"),
            ("CALLS", "P", "Q"),
            ("CALLS", "X", "Y"),
        ],
    );

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let r = storage.read_view(&txn);
    let first = label_propagation(&storage, &r, &["CALLS"], 50).unwrap();
    for _ in 0..20 {
        assert_eq!(
            label_propagation(&storage, &r, &["CALLS"], 50).unwrap(),
            first
        );
    }
    for pair in first.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        assert!(
            a.1.len() > b.1.len() || (a.1.len() == b.1.len() && a.0 < b.0),
            "communities must be ordered by (size desc, id asc): {:?}",
            first
        );
    }
    for (_, members) in &first {
        assert!(members.windows(2).all(|m| m[0] < m[1]));
    }
}

#[test]
fn scoped_algorithms_skip_dangling_adjacency_edges() {
    let (storage, _tmp) = setup();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
    for name in ["A", "B", "C"] {
        storage.upsert_node(&mut txn, &make_node(name)).unwrap();
    }
    let kept = make_edge_with_repo("CALLS", "A", "B", "repo-a");
    let dangling = make_edge_with_repo("CALLS", "A", "C", "repo-a");
    storage.upsert_edge(&mut txn, &kept).unwrap();
    storage.upsert_edge(&mut txn, &dangling).unwrap();
    // Drop only the edge record, leaving its adjacency entries behind.
    storage
        .edges_db
        .unwrap()
        .delete(&mut txn, &dangling.id)
        .unwrap();
    txn.commit().unwrap();

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let r = storage.read_view(&txn);
    let ranks = pagerank_scoped(&storage, &r, &["CALLS"], 20, 0.85, Some("repo-a")).unwrap();
    let ids: HashSet<u128> = ranks.into_iter().map(|(id, _)| id).collect();
    assert_eq!(ids, HashSet::from([node_id("A"), node_id("B")]));
    label_propagation_scoped(&storage, &r, &["CALLS"], 10, Some("repo-a")).unwrap();
}

#[test]
fn shortest_path_reaches_adjacent_target_at_visited_budget() {
    let (storage, _tmp) = setup();
    build_graph(&storage, &["A", "B"], &[("CALLS", "A", "B")]);

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let r = storage.read_view(&txn);
    let (path, distance) =
        shortest_path(&storage, &r, node_id("A"), node_id("B"), &["CALLS"], 10, 1).unwrap();
    assert_eq!(distance, 1);
    assert_eq!(path.len(), 2);
}

#[test]
fn shortest_path_returns_ordered_path_and_respects_depth_budget() {
    let (storage, _tmp) = setup();
    build_graph(
        &storage,
        &["A", "B", "C", "D"],
        &[
            ("CALLS", "A", "B"),
            ("CALLS", "B", "C"),
            ("CALLS", "C", "D"),
        ],
    );

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let r = storage.read_view(&txn);
    let (path, distance) = shortest_path(
        &storage,
        &r,
        node_id("A"),
        node_id("D"),
        &["CALLS"],
        10,
        100,
    )
    .unwrap();
    let names: Vec<String> = path
        .iter()
        .map(|node| match node.properties.get("name").unwrap() {
            Value::String(name) => name.clone(),
            _ => String::new(),
        })
        .collect();

    assert_eq!(distance, 3);
    assert_eq!(names, vec!["A", "B", "C", "D"]);

    let (path, distance) =
        shortest_path(&storage, &r, node_id("A"), node_id("D"), &["CALLS"], 2, 100).unwrap();
    assert!(path.is_empty());
    assert_eq!(distance, 0);
}

#[test]
fn shortest_path_respects_visited_budget_on_large_chain() {
    let (storage, _tmp) = setup();
    let names: Vec<String> = (0..120).map(|i| format!("N{}", i)).collect();

    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
    for name in &names {
        storage.upsert_node(&mut txn, &make_node(name)).unwrap();
    }
    for i in 0..names.len() - 1 {
        storage
            .upsert_edge(&mut txn, &make_edge("CALLS", &names[i], &names[i + 1]))
            .unwrap();
    }
    txn.commit().unwrap();

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let r = storage.read_view(&txn);
    let (path, distance) = shortest_path(
        &storage,
        &r,
        node_id("N0"),
        node_id("N119"),
        &["CALLS"],
        200,
        50,
    )
    .unwrap();

    assert!(path.is_empty());
    assert_eq!(distance, 0);

    let (path, distance) = shortest_path(
        &storage,
        &r,
        node_id("N0"),
        node_id("N119"),
        &["CALLS"],
        200,
        200,
    )
    .unwrap();

    assert_eq!(path.len(), 120);
    assert_eq!(distance, 119);
}

#[test]
fn jaccard_similarity_scores_shared_out_neighbors() {
    let (storage, _tmp) = setup();
    build_graph(
        &storage,
        &["A", "B", "C", "D", "E"],
        &[
            ("CALLS", "A", "C"),
            ("CALLS", "A", "D"),
            ("CALLS", "B", "C"),
            ("CALLS", "B", "E"),
        ],
    );

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let r = storage.read_view(&txn);
    let score = jaccard_similarity(&storage, &r, node_id("A"), node_id("B"), &["CALLS"]).unwrap();

    assert!((score - (1.0 / 3.0)).abs() < 1e-9);
}
