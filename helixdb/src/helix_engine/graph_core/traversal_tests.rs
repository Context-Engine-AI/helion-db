use std::{sync::Arc, time::Instant};

use crate::protocol::{filterable::Filterable, items::Node, value::Value};
use crate::{helix_engine::graph_core::ops::source::bulk_add_e::BulkAddEAdapter, props};
use crate::{
    helix_engine::{
        graph_core::ops::{
            g::G,
            in_::{in_e::InEdgesAdapter, to_n::ToNAdapter},
            out::{from_n::FromNAdapter, out::OutAdapter},
            source::{
                add_n::AddNAdapter, bulk_add_n::BulkAddNAdapter, e::EAdapter,
                e_from_id::EFromIdAdapter, n::NAdapter, n_from_id::NFromIdAdapter,
            },
            tr_val::{Traversable, TraversalVal},
            util::{dedup::DedupAdapter, range::RangeAdapter, update::UpdateAdapter},
        },
        storage_core::{
            backend::{BackendKind, StorageBackend},
            storage_core::HelixGraphStorage,
            storage_methods::StorageMethods,
        },
        types::GraphError,
    },
    protocol::items::v6_uuid,
};
use rand::Rng;
use tempfile::TempDir;

use super::ops::{
    in_::in_::InAdapter,
    out::out_e::OutEdgesAdapter,
    source::add_e::{AddEAdapter, EdgeType},
    util::filter_ref::FilterRefAdapter,
};

fn setup_test_db() -> (Arc<HelixGraphStorage>, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().to_str().unwrap();
    let storage = HelixGraphStorage::new(db_path, super::config::Config::default()).unwrap();
    (Arc::new(storage), temp_dir)
}

fn setup_test_db_with_secondary_indices(indices: &[&str]) -> (Arc<HelixGraphStorage>, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().to_str().unwrap();
    let mut config = super::config::Config::default();
    config.graph_config.secondary_indices =
        Some(indices.iter().map(|index| index.to_string()).collect());
    let storage = HelixGraphStorage::new(db_path, config).unwrap();
    (Arc::new(storage), temp_dir)
}

struct EnvGuard(&'static str, Option<String>);

impl Drop for EnvGuard {
    fn drop(&mut self) {
        unsafe {
            match &self.1 {
                Some(v) => std::env::set_var(self.0, v),
                None => std::env::remove_var(self.0),
            }
        }
    }
}

fn set_env(key: &'static str, value: &str) -> EnvGuard {
    let prev = std::env::var(key).ok();
    unsafe { std::env::set_var(key, value) };
    EnvGuard(key, prev)
}

fn force_lsm_in_memory() -> (EnvGuard, EnvGuard) {
    (
        set_env("HELIX_STORAGE_BACKEND", "lsm"),
        set_env("HELIX_LSM_IN_MEMORY", "1"),
    )
}

/// End-to-end acceptance gate for the LSM cutover: a fully in-memory
/// SlateDB-backed storage (graph *and* vectors) that inserts nodes, an edge and
/// dense vectors through the flipped write traversal, then reads the graph back
/// and runs a vector search through the flipped single-snapshot read traversal —
/// i.e. the whole stack runs on the LSM backend, not just LMDB.
///
/// `HELIX_STORAGE_BACKEND=lsm` + `HELIX_LSM_IN_MEMORY=1` are process-global, so
/// this test is `#[serial]` and must run single-threaded (the repo's test
/// convention, `--test-threads=1`), exactly like the env-mutating `#[serial]`
/// tests in `storage_core.rs`.
#[test]
#[serial_test::serial]
fn lsm_end_to_end_graph_and_vector_in_memory() {
    use crate::helix_engine::graph_core::ops::vectors::{
        insert::InsertVAdapter, search::SearchVAdapter,
    };
    use crate::helix_engine::vector_core::vector::HVector;

    let (_backend, _in_mem) = force_lsm_in_memory();

    let (storage, _temp_dir) = setup_test_db();
    assert_eq!(
        storage.backend.kind(),
        BackendKind::Lsm,
        "construction selected the in-memory LSM backend"
    );

    let main_id: u128 = 0xE2E_0001;
    let helper_id: u128 = 0xE2E_0002;

    // 1. Two nodes through the flipped write traversal (writes via AnyWrite).
    {
        let mut wv = storage.begin_write_view().unwrap();
        let main_node = G::new_mut(Arc::clone(&storage), wv.write_mut())
            .add_n(
                "Sym",
                vec![("name".to_string(), Value::String("main".into()))],
                None,
                Some(main_id),
            )
            .collect::<Vec<_>>();
        assert!(
            main_node.iter().all(Result::is_ok),
            "main node insert failed on LSM: {main_node:?}"
        );
        let helper_node = G::new_mut(Arc::clone(&storage), wv.write_mut())
            .add_n(
                "Sym",
                vec![("name".to_string(), Value::String("helper".into()))],
                None,
                Some(helper_id),
            )
            .collect::<Vec<_>>();
        assert!(
            helper_node.iter().all(Result::is_ok),
            "helper node insert failed on LSM: {helper_node:?}"
        );
        wv.commit().unwrap();
    }

    // 2. An edge main --CALLS--> helper (endpoints already committed above).
    {
        let mut wv = storage.begin_write_view().unwrap();
        let edge = G::new_mut(Arc::clone(&storage), wv.write_mut())
            .add_e(
                "CALLS",
                Vec::<(String, Value)>::new(),
                None,
                main_id,
                helper_id,
                false,
                EdgeType::Std,
            )
            .collect::<Vec<_>>();
        assert!(
            edge.iter().all(Result::is_ok),
            "edge insert failed on LSM: {edge:?}"
        );
        wv.commit().unwrap();
    }

    // 3. Two dense vectors through the flipped write traversal. On LSM
    //    `insert_v` appends to the mutable flat segment (`insert_flat_be`).
    {
        let mut wv = storage.begin_write_view().unwrap();
        let r1 = G::new_mut(Arc::clone(&storage), wv.write_mut())
            .insert_v::<fn(&HVector) -> bool>(&vec![1.0, 0.0, 0.0, 0.0], None)
            .collect::<Vec<_>>();
        assert!(
            r1.iter().all(|x| x.is_ok()),
            "insert_v #1 failed on LSM: {:?}",
            r1
        );
        let r2 = G::new_mut(Arc::clone(&storage), wv.write_mut())
            .insert_v::<fn(&HVector) -> bool>(&vec![0.0, 1.0, 0.0, 0.0], None)
            .collect::<Vec<_>>();
        assert!(
            r2.iter().all(|x| x.is_ok()),
            "insert_v #2 failed on LSM: {:?}",
            r2
        );
        wv.commit().unwrap();
    }

    // 4. Read the graph back through the flipped single-snapshot read traversal.
    let read = storage.backend.begin_read().unwrap();
    let out_nodes = G::new(Arc::clone(&storage), &read)
        .n_from_id(&main_id)
        .out("CALLS")
        .filter_map(|n| n.ok())
        .collect::<Vec<_>>();
    assert_eq!(
        out_nodes.len(),
        1,
        "out('CALLS') from main yields one node on LSM"
    );
    assert_eq!(
        out_nodes[0].id(),
        helper_id,
        "the traversal resolves the helper node on LSM"
    );

    // 5. Vector search through the same read traversal. A filtered flat search
    //    enumerates the flat segment through the backend seam (`get_all_vectors`
    //    -> `scan`), i.e. the SlateDB-stored vectors that `insert_v` wrote —
    //    proving dense vectors round-trip insert+search on LSM. (The *unfiltered*
    //    hot path reads the mmap sidecar, which the pure-backend flat insert does
    //    not populate; populating it is an mmap follow-up orthogonal to the seam.)
    let accept_all: [fn(&HVector) -> bool; 1] = [|_| true];
    let hits = G::new(Arc::clone(&storage), &read)
        .search_v::<fn(&HVector) -> bool>(&vec![1.0, 0.0, 0.0, 0.0], 2, Some(&accept_all[..]))
        .filter_map(|v| v.ok())
        .collect::<Vec<_>>();
    assert_eq!(
        hits.len(),
        2,
        "filtered vector search returns both inserted vectors from the LSM flat segment"
    );
    assert!(
        matches!(hits[0], TraversalVal::Vector(_)),
        "search yields vector results"
    );
}

#[test]
#[serial_test::serial]
fn lsm_insert_vs_round_trips_from_backend_flat_segment() {
    use crate::helix_engine::graph_core::ops::vectors::{
        insert::InsertVAdapter, search::SearchVAdapter,
    };
    use crate::helix_engine::vector_core::vector::HVector;

    let (_backend, _in_mem) = force_lsm_in_memory();
    let (storage, _temp_dir) = setup_test_db();
    assert_eq!(storage.backend.kind(), BackendKind::Lsm);

    {
        let mut wv = storage.begin_write_view().unwrap();
        let vectors = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let inserted = G::new_mut(Arc::clone(&storage), wv.write_mut())
            .insert_vs::<fn(&HVector) -> bool>(&vectors, None)
            .collect::<Vec<_>>();
        assert_eq!(inserted.len(), 2);
        assert!(
            inserted.iter().all(Result::is_ok),
            "insert_vs failed on LSM: {inserted:?}"
        );
        wv.commit().unwrap();
    }

    let read = storage.backend.begin_read().unwrap();
    let accept_all: [fn(&HVector) -> bool; 1] = [|_| true];
    let hits = G::new(Arc::clone(&storage), &read)
        .search_v::<fn(&HVector) -> bool>(&vec![1.0, 0.0, 0.0, 0.0], 2, Some(&accept_all[..]))
        .filter_map(|v| v.ok())
        .collect::<Vec<_>>();
    assert_eq!(
        hits.len(),
        2,
        "insert_vs vectors are visible through backend-routed flat search on LSM"
    );
}

#[test]
fn test_n() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    let person1 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let person2 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let thing = storage
        .create_node(&mut txn, "thing", props!(), None, None)
        .unwrap();
    txn.commit().unwrap();

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    // let mut traversal = TraversalBuilder::new(Arc::clone(&storage), TraversalValue::Empty);
    let nodes = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n()
        .collect_to::<Vec<_>>();
    // Check that the node array contains all nodes
    assert_eq!(nodes.len(), 3);

    let node_ids: Vec<u128> = nodes.iter().map(|n| n.id()).collect();
    let node_labels: Vec<String> = nodes.iter().map(|n| n.label()).collect();

    assert!(node_ids.contains(&person1.id));
    assert!(node_ids.contains(&person2.id));
    assert!(node_ids.contains(&thing.id));

    assert_eq!(node_labels.iter().filter(|&l| l == "person").count(), 2);
    assert_eq!(node_labels.iter().filter(|&l| l == "thing").count(), 1);
}

#[test]
fn test_e() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    // Graph Structure:
    // (person1)-[knows]->(person2)
    //         \-[likes]->(person3)
    // (person2)-[follows]->(person3)

    let person1 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let person2 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let person3 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();

    let knows_edge = storage
        .create_edge(&mut txn, "knows", &person1.id, &person2.id, props!())
        .unwrap();
    let likes_edge = storage
        .create_edge(&mut txn, "likes", &person1.id, &person3.id, props!())
        .unwrap();
    let follows_edge = storage
        .create_edge(&mut txn, "follows", &person2.id, &person3.id, props!())
        .unwrap();

    txn.commit().unwrap();

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let edges = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .e()
        .collect_to::<Vec<_>>();

    // Check that the edge array contains the three edges
    assert_eq!(edges.len(), 3);

    let edge_ids: Vec<u128> = edges.iter().map(|e| e.id()).collect();
    let edge_labels: Vec<String> = edges.iter().map(|e| e.label().to_string()).collect();

    assert!(edge_ids.contains(&knows_edge.id));
    assert!(edge_ids.contains(&likes_edge.id));
    assert!(edge_ids.contains(&follows_edge.id));

    assert!(edge_labels.contains(&"knows".to_string()));
    assert!(edge_labels.contains(&"likes".to_string()));
    assert!(edge_labels.contains(&"follows".to_string()));

    for edge in edges {
        match edge {
            TraversalVal::Edge(edge) => match edge.label() {
                "knows" => {
                    assert_eq!(edge.from_node(), person1.id);
                    assert_eq!(edge.to_node(), person2.id);
                }
                "likes" => {
                    assert_eq!(edge.from_node(), person1.id);
                    assert_eq!(edge.to_node(), person3.id);
                }
                "follows" => {
                    assert_eq!(edge.from_node(), person2.id);
                    assert_eq!(edge.to_node(), person3.id);
                }
                _ => panic!("Unexpected edge label"),
            },
            _ => panic!("Expected Edge value"),
        }
    }
}

#[test]
fn test_n_empty_graph() {
    let (storage, _temp_dir) = setup_test_db();

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();

    let nodes = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n()
        .collect_to::<Vec<_>>();

    // Check that the node array is empty
    assert_eq!(nodes.len(), 0);
}

#[test]
fn test_e_empty_graph() {
    let (storage, _temp_dir) = setup_test_db();

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let edges = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .e()
        .collect_to::<Vec<_>>();

    // Check that the edge array is empty
    assert_eq!(edges.len(), 0);
}

#[test]
fn test_n_nodes_without_edges() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    let person1 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let person2 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();

    txn.commit().unwrap();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();

    let nodes = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n()
        .collect_to::<Vec<_>>();

    // Check that the node array contains the two nodes
    assert_eq!(nodes.len(), 2);
    let node_ids: Vec<u128> = nodes.iter().map(|n| n.id()).collect();
    assert!(node_ids.contains(&person1.id));
    assert!(node_ids.contains(&person2.id));
}

#[test]
fn test_add_n() {
    let (storage, _temp_dir) = setup_test_db();

    let mut wv = storage.begin_write_view().unwrap();

    let nodes = G::new_mut(Arc::clone(&storage), wv.write_mut())
        .add_n("person", props! {}, None, None)
        .filter_map(|node| node.ok())
        .collect::<Vec<_>>();

    assert_eq!(nodes.first().unwrap().label(), "person");

    wv.commit().unwrap();
}

#[test]
fn test_add_n_missing_secondary_property_does_not_commit_orphan_node() {
    let (storage, _temp_dir) = setup_test_db_with_secondary_indices(&["email"]);
    let secondary_indices = vec!["email".to_string()];
    let node_id = 42;

    let mut wv = storage.begin_write_view().unwrap();
    let nodes = G::new_mut(Arc::clone(&storage), wv.write_mut())
        .add_n(
            "person",
            props! { "name" => "Ada" },
            Some(&secondary_indices),
            Some(node_id),
        )
        .filter_map(|node| node.ok())
        .collect::<Vec<_>>();
    assert!(nodes.is_empty());
    wv.commit().unwrap();

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    assert!(matches!(
        storage.get_node(&txn, &node_id),
        Err(GraphError::NodeNotFound)
    ));
}

#[test]
fn test_add_n_secondary_index_uses_storage_key_format() {
    let (storage, _temp_dir) = setup_test_db_with_secondary_indices(&["email"]);
    let secondary_indices = vec!["email".to_string()];
    let node_id = 43;

    let mut wv = storage.begin_write_view().unwrap();
    let nodes = G::new_mut(Arc::clone(&storage), wv.write_mut())
        .add_n(
            "person",
            props! { "email" => "ada@example.com" },
            Some(&secondary_indices),
            Some(node_id),
        )
        .filter_map(|node| node.ok())
        .collect::<Vec<_>>();
    assert_eq!(nodes.len(), 1);
    wv.commit().unwrap();

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let node = storage
        .get_node_by_secondary_index(&txn, "email", &Value::String("ada@example.com".to_string()))
        .unwrap();
    assert_eq!(node.id, node_id);
}

#[test]
fn test_update_moves_secondary_index_and_persists_node() {
    let (storage, _temp_dir) = setup_test_db_with_secondary_indices(&["email"]);
    let secondary_indices = vec!["email".to_string()];
    let node_id = 44;

    let mut wv = storage.begin_write_view().unwrap();
    let updated = G::new_mut(Arc::clone(&storage), wv.write_mut())
        .add_n(
            "person",
            props! { "email" => "old@example.com", "name" => "Ada" },
            Some(&secondary_indices),
            Some(node_id),
        )
        .update(props! { "email" => "new@example.com", "name" => "Grace" })
        .collect::<Vec<_>>();
    assert_eq!(updated.len(), 1);
    assert!(updated[0].is_ok());
    wv.commit().unwrap();

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let node = storage.get_node(&txn, &node_id).unwrap();
    assert_eq!(
        node.properties.get("email"),
        Some(&Value::String("new@example.com".to_string()))
    );
    assert_eq!(
        node.properties.get("name"),
        Some(&Value::String("Grace".to_string()))
    );

    let by_new_email = storage
        .get_node_by_secondary_index(&txn, "email", &Value::String("new@example.com".to_string()))
        .unwrap();
    assert_eq!(by_new_email.id, node_id);
    assert!(matches!(
        storage.get_node_by_secondary_index(
            &txn,
            "email",
            &Value::String("old@example.com".to_string()),
        ),
        Err(GraphError::NodeNotFound)
    ));
}

#[test]
#[serial_test::serial]
fn lsm_update_moves_secondary_index_and_persists_node() {
    let (_backend, _in_mem) = force_lsm_in_memory();
    let (storage, _temp_dir) = setup_test_db_with_secondary_indices(&["email"]);
    assert_eq!(storage.backend.kind(), BackendKind::Lsm);

    let secondary_indices = vec!["email".to_string()];
    let node_id = 0x1A_0001;

    {
        let mut wv = storage.begin_write_view().unwrap();
        let updated = G::new_mut(Arc::clone(&storage), wv.write_mut())
            .add_n(
                "person",
                props! { "email" => "old@example.com", "name" => "Ada" },
                Some(&secondary_indices),
                Some(node_id),
            )
            .update(props! { "email" => "new@example.com", "name" => "Grace" })
            .collect::<Vec<_>>();
        assert_eq!(updated.len(), 1);
        assert!(updated[0].is_ok(), "LSM node update failed: {updated:?}");
        wv.commit().unwrap();
    }

    let r = storage.backend.begin_read().unwrap();
    let node = storage.get_node_be(&r, &node_id).unwrap();
    assert_eq!(
        node.properties.get("email"),
        Some(&Value::String("new@example.com".to_string()))
    );
    assert_eq!(
        node.properties.get("name"),
        Some(&Value::String("Grace".to_string()))
    );

    let by_new_email = storage
        .get_node_by_secondary_index_be(&r, "email", &Value::String("new@example.com".to_string()))
        .unwrap();
    assert_eq!(by_new_email.id, node_id);
    assert!(matches!(
        storage.get_node_by_secondary_index_be(
            &r,
            "email",
            &Value::String("old@example.com".to_string()),
        ),
        Err(GraphError::NodeNotFound)
    ));
}

#[test]
fn test_update_persists_edge_properties() {
    let (storage, _temp_dir) = setup_test_db();
    let mut wv = storage.begin_write_view().unwrap();
    let node1 = storage
        .create_node(
            wv.write_mut().lmdb_rw_mut().unwrap(),
            "person",
            props!(),
            None,
            Some(45),
        )
        .unwrap();
    let node2 = storage
        .create_node(
            wv.write_mut().lmdb_rw_mut().unwrap(),
            "person",
            props!(),
            None,
            Some(46),
        )
        .unwrap();

    let updated = G::new_mut(Arc::clone(&storage), wv.write_mut())
        .add_e(
            "knows",
            props! { "since" => 2020 },
            None,
            node1.id,
            node2.id,
            false,
            EdgeType::Std,
        )
        .update(props! { "since" => 2024 })
        .collect::<Vec<_>>();
    assert_eq!(updated.len(), 1);
    let edge_id = updated[0].as_ref().unwrap().id();
    wv.commit().unwrap();

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let edge = storage.get_edge(&txn, &edge_id).unwrap();
    assert_eq!(edge.properties.get("since"), Some(&Value::I32(2024)));
}

#[test]
#[serial_test::serial]
fn lsm_update_persists_edge_properties() {
    let (_backend, _in_mem) = force_lsm_in_memory();
    let (storage, _temp_dir) = setup_test_db();
    assert_eq!(storage.backend.kind(), BackendKind::Lsm);

    let node1_id = 0x1A_0010;
    let node2_id = 0x1A_0011;
    {
        let mut wv = storage.begin_write_view().unwrap();
        let first = G::new_mut(Arc::clone(&storage), wv.write_mut())
            .add_n("person", props!(), None, Some(node1_id))
            .collect::<Vec<_>>();
        let second = G::new_mut(Arc::clone(&storage), wv.write_mut())
            .add_n("person", props!(), None, Some(node2_id))
            .collect::<Vec<_>>();
        let nodes = first.into_iter().chain(second).collect::<Vec<_>>();
        assert_eq!(nodes.len(), 2);
        assert!(
            nodes.iter().all(Result::is_ok),
            "LSM node seed failed: {nodes:?}"
        );
        wv.commit().unwrap();
    }

    let edge_id = {
        let mut wv = storage.begin_write_view().unwrap();
        let updated = G::new_mut(Arc::clone(&storage), wv.write_mut())
            .add_e(
                "knows",
                props! { "since" => 2020 },
                None,
                node1_id,
                node2_id,
                false,
                EdgeType::Std,
            )
            .update(props! { "since" => 2024 })
            .collect::<Vec<_>>();
        assert_eq!(updated.len(), 1);
        assert!(updated[0].is_ok(), "LSM edge update failed: {updated:?}");
        let edge_id = updated[0].as_ref().unwrap().id();
        wv.commit().unwrap();
        edge_id
    };

    let r = storage.backend.begin_read().unwrap();
    let edge = storage.get_edge_be(&r, &edge_id).unwrap();
    assert_eq!(edge.properties.get("since"), Some(&Value::I32(2024)));
}

/// End-to-end proof that write-context `e()` is read-your-writes on LSM: a second
/// edge created in the SAME uncommitted write batch is visible to `.e()` alongside
/// the already-committed edge. Exercises the full traversal path (`G::new_mut().e()`
/// → backend `scan` pending-overlay), not just the backend unit tests.
#[test]
#[serial_test::serial]
fn lsm_write_context_e_reads_your_writes() {
    use crate::helix_engine::graph_core::ops::source::e::RwEAdapter;
    let (_backend, _in_mem) = force_lsm_in_memory();
    let (storage, _temp_dir) = setup_test_db();
    assert_eq!(storage.backend.kind(), BackendKind::Lsm);

    let n1: u128 = 0x1A_0020;
    let n2: u128 = 0x1A_0021;
    let n3: u128 = 0x1A_0022;

    // Seed three nodes + one COMMITTED edge (knows: n1->n2).
    {
        let mut wv = storage.begin_write_view().unwrap();
        for id in [n1, n2, n3] {
            let seeded = G::new_mut(Arc::clone(&storage), wv.write_mut())
                .add_n("person", props!(), None, Some(id))
                .collect::<Vec<_>>();
            assert!(
                seeded.iter().all(Result::is_ok),
                "seed node failed: {seeded:?}"
            );
        }
        let knows = G::new_mut(Arc::clone(&storage), wv.write_mut())
            .add_e("knows", props!(), None, n1, n2, false, EdgeType::Std)
            .collect::<Vec<_>>();
        assert!(knows[0].is_ok(), "seed edge failed: {knows:?}");
        wv.commit().unwrap();
    }

    // In ONE write batch: create a second edge (likes: n1->n3), then scan `.e()`
    // BEFORE commit. The write-context read-your-writes view must return BOTH the
    // committed `knows` (from the snapshot) and the uncommitted `likes` (from the
    // batch's pending overlay).
    {
        let mut wv = storage.begin_write_view().unwrap();
        let likes = G::new_mut(Arc::clone(&storage), wv.write_mut())
            .add_e("likes", props!(), None, n1, n3, false, EdgeType::Std)
            .collect::<Vec<_>>();
        assert!(likes[0].is_ok(), "add likes edge failed: {likes:?}");

        let edges = G::new_mut(Arc::clone(&storage), wv.write_mut())
            .e()
            .collect::<Vec<_>>();
        let labels: Vec<String> = edges
            .iter()
            .filter_map(|r| r.as_ref().ok())
            .filter_map(|tv| match tv {
                TraversalVal::Edge(e) => Some(e.label().to_string()),
                _ => None,
            })
            .collect();
        assert_eq!(
            edges.len(),
            2,
            "write-context e() must see committed + uncommitted batch edges: {labels:?}"
        );
        assert!(labels.contains(&"knows".to_string()));
        assert!(labels.contains(&"likes".to_string()));
        wv.commit().unwrap();
    }

    // After commit, both edges are durably visible to a fresh read.
    let r = storage.backend.begin_read().unwrap();
    let all = storage.scan_all_edges_be(&r).unwrap();
    assert_eq!(all.len(), 2, "both edges must persist after commit");
}

#[test]
fn test_add_e() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    let node1 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let node2 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();

    txn.commit().unwrap();
    let mut wv = storage.begin_write_view().unwrap();
    let edges = G::new_mut(Arc::clone(&storage), wv.write_mut())
        .add_e(
            "knows",
            props! {},
            None,
            node1.id.clone(),
            node2.id.clone(),
            false,
            EdgeType::Std,
        )
        .filter_map(|edge| edge.ok())
        .collect::<Vec<_>>();
    wv.commit().unwrap();
    // Check that the current step contains a single edge
    match edges.first() {
        Some(edge) => {
            assert_eq!(edge.label(), "knows");
            match edge {
                TraversalVal::Edge(edge) => {
                    assert_eq!(edge.from_node(), node1.id);
                    assert_eq!(edge.to_node(), node2.id);
                }
                _ => panic!("Expected Edge value"),
            }
        }
        None => panic!("Expected SingleEdge value"),
    }
}

#[test]
fn test_out() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    // Create graph: (person1)-[knows]->(person2)-[knows]->(person3)
    let person1 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let person2 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let person3 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();

    storage
        .create_edge(&mut txn, "knows", &person1.id, &person2.id, props!())
        .unwrap();
    storage
        .create_edge(&mut txn, "knows", &person2.id, &person3.id, props!())
        .unwrap();

    txn.commit().unwrap();
    let txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    // let nodes = VFromId::new(&storage, &txn, person1.id.as_str())
    //     .out("knows")
    //     .filter_map(|node| node.ok())
    //     .collect::<Vec<_>>();
    let nodes = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n_from_id(&person1.id)
        .out("knows")
        .filter_map(|node| node.ok())
        .collect::<Vec<_>>();

    // txn.commit().unwrap();
    // Check that current step is at person2
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].id(), person2.id);
}

#[test]
fn test_out_e() {
    let (storage, _temp_dir) = setup_test_db();

    // Create graph: (person1)-[knows]->(person2)

    let mut wv = storage.begin_write_view().unwrap();
    let person1 = G::new_mut(Arc::clone(&storage), wv.write_mut())
        .add_n("person", props! {}, None, Some(1))
        .filter_map(|node| node.ok())
        .collect::<Vec<_>>();
    let person1 = person1.first().unwrap();
    wv.commit().unwrap();
    let mut wv = storage.begin_write_view().unwrap();
    let person2 = G::new_mut(Arc::clone(&storage), wv.write_mut())
        .add_n("person", props! {}, None, Some(2))
        .filter_map(|node| node.ok())
        .collect::<Vec<_>>();
    let person2 = person2.first().unwrap();
    wv.commit().unwrap();
    let mut wv = storage.begin_write_view().unwrap();
    let edge = G::new_mut(Arc::clone(&storage), wv.write_mut())
        .add_e(
            "knows",
            props! {},
            None,
            person1.id().clone(),
            person2.id().clone(),
            false,
            EdgeType::Std,
        )
        .filter_map(|edge| edge.ok())
        .collect::<Vec<_>>();
    let edge = edge.first().unwrap();
    // println!("traversal edge: {:?}", edge);

    wv.commit().unwrap();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    println!("processing");
    let edges = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n_from_id(&person1.id())
        .out_e("knows")
        .collect_to::<Vec<_>>();
    println!("edges: {}", edges.len());

    // Check that current step is at the edge between person1 and person2
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].id(), edge.id());
    assert_eq!(edges[0].label(), "knows");
}

#[test]
fn test_in() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    // Create graph: (person1)-[knows]->(person2)
    let person1 = storage
        .create_node(&mut txn, "person", props!(), None, Some(1))
        .unwrap();
    let person2 = storage
        .create_node(&mut txn, "person", props!(), None, Some(2))
        .unwrap();

    storage
        .create_edge(&mut txn, "knows", &person1.id, &person2.id, props!())
        .unwrap();
    txn.commit().unwrap();

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let nodes = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n_from_id(&person2.id)
        .in_("knows")
        .collect_to::<Vec<_>>();

    // Check that current step is at person1
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].id(), person1.id);
}

#[test]
fn test_in_e() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    // Create test graph: (person1)-[knows]->(person2)
    let person1 = storage
        .create_node(&mut txn, "person", props!(), None, Some(1))
        .unwrap();
    let person2 = storage
        .create_node(&mut txn, "person", props!(), None, Some(2))
        .unwrap();
    println!("person1: {:?}", person1);
    println!("person2: {:?}", person2);

    let edge = storage
        .create_edge(&mut txn, "knows", &person1.id, &person2.id, props!())
        .unwrap();
    println!("edge: {:?}", edge);

    txn.commit().unwrap();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();

    let edges = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n_from_id(&person2.id)
        .in_e("knows")
        .collect_to::<Vec<_>>();

    // Check that current step is at the edge between person1 and person2
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].id(), edge.id);
    assert_eq!(edges[0].label(), "knows");
}

#[test]
fn test_complex_traversal() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    // Graph structure:
    // (person1)-[knows]->(person2)-[likes]->(person3)
    //     ^                                     |
    //     |                                     |
    //     +-------<------[follows]------<-------+

    let person1 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let person2 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let person3 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();

    storage
        .create_edge(&mut txn, "knows", &person1.id, &person2.id, props!())
        .unwrap();
    storage
        .create_edge(&mut txn, "likes", &person2.id, &person3.id, props!())
        .unwrap();
    storage
        .create_edge(&mut txn, "follows", &person3.id, &person1.id, props!())
        .unwrap();

    txn.commit().unwrap();

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();

    let nodes = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n_from_id(&person1.id)
        .out("knows")
        .collect_to::<Vec<_>>();

    // Check that current step is at person2
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].id(), person2.id);

    // Traverse from person2 to person3
    let nodes = G::new_from(
        Arc::clone(&storage),
        &storage.read_view(&txn),
        vec![nodes[0].clone()],
    )
    .out("likes")
    .collect_to::<Vec<_>>();

    // Check that current step is at person3
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].id(), person3.id);

    // Traverse from person3 to person1
    let nodes = G::new_from(
        Arc::clone(&storage),
        &storage.read_view(&txn),
        vec![nodes[0].clone()],
    )
    .out("follows")
    .collect_to::<Vec<_>>();

    // Check that current step is at person1
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].id(), person1.id);
}

#[test]
fn test_count_single_node() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
    let person = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    txn.commit().unwrap();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let count = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n_from_id(&person.id)
        .count();

    assert_eq!(count, 1);
}

#[test]
fn test_count_node_array() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
    let _ = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let _ = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let _ = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();

    txn.commit().unwrap();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let count = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n() // Get all nodes
        .count();
    assert_eq!(count, 3);
}

#[test]
fn test_count_mixed_steps() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    // Create a graph with multiple paths
    let person1 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let person2 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let person3 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();

    storage
        .create_edge(&mut txn, "knows", &person1.id, &person2.id, props!())
        .unwrap();
    storage
        .create_edge(&mut txn, "knows", &person1.id, &person3.id, props!())
        .unwrap();
    txn.commit().unwrap();
    println!(
        "person1: {:?},\nperson2: {:?},\nperson3: {:?}",
        person1, person2, person3
    );

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let count = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n_from_id(&person1.id)
        .out("knows")
        .count();

    assert_eq!(count, 2);
}

#[test]
fn test_range_subset() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    // Create multiple nodes
    let _: Vec<Node> = (0..5)
        .map(|_| {
            storage
                .create_node(&mut txn, "person", props!(), None, None)
                .unwrap()
        })
        .collect();

    txn.commit().unwrap();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let count = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n() // Get all nodes
        .range(1, 3) // Take nodes at index 1 and 2
        .count();

    assert_eq!(count, 2);
}

#[test]
fn test_range_chaining() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    // Create graph: (p1)-[knows]->(p2)-[knows]->(p3)-[knows]->(p4)-[knows]->(p5)
    let nodes: Vec<Node> = (0..5)
        .map(|i| {
            storage
                .create_node(&mut txn, "person", props! { "name" => i }, None, None)
                .unwrap()
        })
        .collect();

    // Create edges connecting nodes sequentially
    for i in 0..4 {
        storage
            .create_edge(&mut txn, "knows", &nodes[i].id, &nodes[i + 1].id, props!())
            .unwrap();
    }

    storage
        .create_edge(&mut txn, "knows", &nodes[4].id, &nodes[0].id, props!())
        .unwrap();

    txn.commit().unwrap();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let count = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n() // Get all nodes
        .range(0, 3) // Take first 3 nodes
        .out("knows") // Get their outgoing nodes
        .collect_to::<Vec<_>>();

    assert_eq!(count.len(), 3);
}

#[test]
fn test_range_empty() {
    let (storage, _temp_dir) = setup_test_db();

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let count = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n() // Get all nodes
        .range(0, 0) // Take first 3 nodes
        .collect_to::<Vec<_>>();

    assert_eq!(count.len(), 0);
}

#[test]
fn test_count_empty() {
    let (storage, _temp_dir) = setup_test_db();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let count = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n() // Get all nodes
        .range(0, 0) // Take first 3 nodes
        .count();

    assert_eq!(count, 0);
}

#[test]
fn test_n_from_id() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    // Create a test node
    let person = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let node_id = person.id.clone();

    txn.commit().unwrap();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let count = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n_from_id(&node_id)
        .collect_to::<Vec<_>>();

    assert_eq!(count.len(), 1);
}

#[test]
fn test_n_from_id_with_traversal() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    // Create test graph: (person1)-[knows]->(person2)
    let person1 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let person2 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    storage
        .create_edge(&mut txn, "knows", &person1.id, &person2.id, props!())
        .unwrap();

    txn.commit().unwrap();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let count = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n_from_id(&person1.id)
        .out("knows")
        .collect_to::<Vec<_>>();

    // Check that traversal reaches person2
    assert_eq!(count.len(), 1);
    assert_eq!(count[0].id(), person2.id);
}

#[test]
fn test_e_from_id() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    // Create test graph and edge
    let person1 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let person2 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let edge = storage
        .create_edge(&mut txn, "knows", &person1.id, &person2.id, props!())
        .unwrap();
    let edge_id = edge.id.clone();

    txn.commit().unwrap();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let edges = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .e_from_id(&edge_id)
        .collect_to::<Vec<_>>();

    // Check that the current step contains the correct single edge
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].id(), edge_id);
    assert_eq!(edges[0].label(), "knows");
    if let Some(TraversalVal::Edge(edge)) = edges.first() {
        assert_eq!(edge.from_node(), person1.id);
        assert_eq!(edge.to_node(), person2.id);
    } else {
        assert!(false, "Expected Edge value");
    }
}

#[test]
fn test_n_from_id_nonexistent() {
    let (storage, _temp_dir) = setup_test_db();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let nodes = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n_from_id(&100)
        .collect_to::<Vec<_>>();
    assert!(nodes.is_empty());
}

#[test]
fn test_e_from_id_nonexistent() {
    let (storage, _temp_dir) = setup_test_db();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let edges = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .e_from_id(&100)
        .collect_to::<Vec<_>>();
    assert!(edges.is_empty());
}

#[test]
fn test_n_from_id_chain_operations() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    // Create test graph: (person1)-[knows]->(person2)-[likes]->(person3)
    let person1 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let person2 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let person3 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();

    storage
        .create_edge(&mut txn, "knows", &person1.id, &person2.id, props!())
        .unwrap();
    storage
        .create_edge(&mut txn, "likes", &person2.id, &person3.id, props!())
        .unwrap();

    txn.commit().unwrap();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let nodes = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n_from_id(&person1.id)
        .out("knows")
        .out("likes")
        .collect_to::<Vec<_>>();

    // Check that the chain of traversals reaches person3
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].id(), person3.id);
}

#[test]
fn test_e_from_id_chain_operations() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    // Create test graph and edges
    let person1 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let person2 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let person3 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();

    let edge1 = storage
        .create_edge(&mut txn, "knows", &person1.id, &person2.id, props!())
        .unwrap();
    storage
        .create_edge(&mut txn, "likes", &person2.id, &person3.id, props!())
        .unwrap();

    txn.commit().unwrap();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let edges = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .e_from_id(&edge1.id)
        .from_n()
        .collect_to::<Vec<_>>();

    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].id(), person1.id);
    assert_eq!(edges[0].label(), "person");
}

#[test]
fn test_filter_nodes() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    // Create nodes with different properties
    let _ = storage
        .create_node(&mut txn, "person", props! { "age" => 25 }, None, None)
        .unwrap();
    let _ = storage
        .create_node(&mut txn, "person", props! { "age" => 30 }, None, None)
        .unwrap();
    let person3 = storage
        .create_node(&mut txn, "person", props! { "age" => 35 }, None, None)
        .unwrap();

    txn.commit().unwrap();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();

    let traversal = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n()
        .filter_ref(|val, _| {
            if let Ok(TraversalVal::Node(node)) = val {
                if let Some(value) = node.check_property("age") {
                    match value {
                        Value::F64(age) => *age > 30.0,
                        Value::I32(age) => *age > 30,
                        _ => false,
                    }
                } else {
                    false
                }
            } else {
                false
            }
        })
        .collect_to::<Vec<_>>();
    assert_eq!(traversal.len(), 1);
    assert_eq!(traversal[0].id(), person3.id);
}

#[test]
fn test_filter_macro_single_argument() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    let _ = storage
        .create_node(&mut txn, "person", props! { "name" => "Alice" }, None, None)
        .unwrap();
    let _ = storage
        .create_node(&mut txn, "person", props! { "name" => "Bob" }, None, None)
        .unwrap();

    fn has_name(val: &Result<TraversalVal, GraphError>) -> bool {
        if let Ok(TraversalVal::Node(node)) = val {
            return node.check_property("name").is_some();
        } else {
            return false;
        }
    }

    txn.commit().unwrap();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let traversal = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n()
        .filter_ref(|val, _| has_name(val))
        .collect_to::<Vec<_>>();
    assert_eq!(traversal.len(), 2);
    assert!(traversal
        .iter()
        .any(|val| if let TraversalVal::Node(node) = val {
            let name = node.check_property("name").unwrap();
            name == &Value::String("Alice".to_string()) || name == &Value::String("Bob".to_string())
        } else {
            false
        }));
}

#[test]
fn test_filter_macro_multiple_arguments() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    let _ = storage
        .create_node(&mut txn, "person", props! { "age" => 25 }, None, None)
        .unwrap();
    let person2 = storage
        .create_node(&mut txn, "person", props! { "age" => 30 }, None, None)
        .unwrap();
    txn.commit().unwrap();

    fn age_greater_than(val: &Result<TraversalVal, GraphError>, min_age: i32) -> bool {
        if let Ok(TraversalVal::Node(node)) = val {
            if let Some(value) = node.check_property("age") {
                match value {
                    Value::F64(age) => *age > min_age as f64,
                    Value::I32(age) => *age > min_age,
                    _ => false,
                }
            } else {
                false
            }
        } else {
            false
        }
    }

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let traversal = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n()
        .filter_ref(|val, _| age_greater_than(val, 27))
        .collect_to::<Vec<_>>();

    assert_eq!(traversal.len(), 1);
    assert_eq!(traversal[0].id(), person2.id);
}

#[test]
fn test_filter_edges() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    let person1 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let person2 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();

    let _ = storage
        .create_edge(
            &mut txn,
            "knows",
            &person1.id,
            &person2.id,
            props! { "since" => 2020 },
        )
        .unwrap();
    let edge2 = storage
        .create_edge(
            &mut txn,
            "knows",
            &person2.id,
            &person1.id,
            props! { "since" => 2022 },
        )
        .unwrap();

    txn.commit().unwrap();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();

    fn recent_edge(val: &Result<TraversalVal, GraphError>, year: i32) -> bool {
        if let Ok(TraversalVal::Edge(edge)) = val {
            if let Some(value) = edge.check_property("since") {
                match value {
                    Value::I32(since) => return *since > year,
                    Value::F64(since) => return *since > year as f64,
                    _ => return false,
                }
            } else {
                false
            }
        } else {
            false
        }
    }

    let traversal = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .e()
        .filter_ref(|val, _| recent_edge(val, 2021))
        .collect_to::<Vec<_>>();

    assert_eq!(traversal.len(), 1);
    assert_eq!(traversal[0].id(), edge2.id);
}

#[test]
fn test_filter_empty_result() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    let _ = storage
        .create_node(&mut txn, "person", props! { "age" => 25 }, None, None)
        .unwrap();

    txn.commit().unwrap();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let traversal = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n()
        .filter_ref(|val, _| {
            if let Ok(TraversalVal::Node(node)) = val {
                if let Some(value) = node.check_property("age") {
                    match value {
                        Value::I32(age) => return *age > 100,
                        Value::F64(age) => return *age > 100.0,
                        _ => return false,
                    }
                } else {
                    false
                }
            } else {
                false
            }
        })
        .collect_to::<Vec<_>>();
    assert!(traversal.is_empty());
}

#[test]
fn test_filter_chain() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    let _ = storage
        .create_node(
            &mut txn,
            "person",
            props! { "age" => 25, "name" => "Alice" },
            None,
            None,
        )
        .unwrap();
    let person2 = storage
        .create_node(
            &mut txn,
            "person",
            props! { "age" => 30, "name" => "Bob" },
            None,
            None,
        )
        .unwrap();
    let _ = storage
        .create_node(&mut txn, "person", props! { "age" => 35 }, None, None)
        .unwrap();

    txn.commit().unwrap();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();

    fn has_name(val: &Result<TraversalVal, GraphError>) -> bool {
        if let Ok(TraversalVal::Node(node)) = val {
            return node.check_property("name").is_some();
        } else {
            return false;
        }
    }

    fn age_greater_than(val: &Result<TraversalVal, GraphError>, min_age: i32) -> bool {
        if let Ok(TraversalVal::Node(node)) = val {
            if let Some(value) = node.check_property("age") {
                match value {
                    Value::F64(age) => return *age > min_age as f64,
                    Value::I32(age) => return *age > min_age,
                    _ => return false,
                }
            } else {
                return false;
            }
        } else {
            return false;
        }
    }

    let traversal = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n()
        .filter_ref(|val, _| has_name(val))
        .filter_ref(|val, _| age_greater_than(val, 27))
        .collect_to::<Vec<_>>();

    assert_eq!(traversal.len(), 1);
    assert_eq!(traversal[0].id(), person2.id);
}

#[test]
fn test_in_n() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    let person1 = storage
        .create_node(&mut txn, "person", props!(), None, Some(1))
        .unwrap();
    let person2 = storage
        .create_node(&mut txn, "person", props!(), None, Some(2))
        .unwrap();

    let edge = storage
        .create_edge(&mut txn, "knows", &person1.id, &person2.id, props!())
        .unwrap();

    txn.commit().unwrap();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let traversal = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .e_from_id(&edge.id)
        .to_n()
        .collect_to::<Vec<_>>();

    assert_eq!(traversal.len(), 1);
    assert_eq!(traversal[0].id(), person2.id);
}

#[test]
fn test_out_n() {
    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

    let person1 = storage
        .create_node(&mut txn, "person", props!(), None, Some(1))
        .unwrap();
    let person2 = storage
        .create_node(&mut txn, "person", props!(), None, Some(2))
        .unwrap();

    let edge = storage
        .create_edge(&mut txn, "knows", &person1.id, &person2.id, props!())
        .unwrap();

    txn.commit().unwrap();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let traversal = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .e_from_id(&edge.id)
        .from_n()
        .collect_to::<Vec<_>>();
    assert_eq!(traversal.len(), 1);
    assert_eq!(traversal[0].id(), person1.id);
}

#[test]
fn test_edge_properties() {
    let (storage, _temp_dir) = setup_test_db();
    let mut wv = storage.begin_write_view().unwrap();

    let node1 = G::new_mut(Arc::clone(&storage), wv.write_mut())
        .add_n("person", props!(), None, None)
        .collect_to::<Vec<_>>();
    let node1 = node1.first().unwrap().clone();
    let node2 = storage
        .create_node(
            wv.write_mut().lmdb_rw_mut().unwrap(),
            "person",
            props!(),
            None,
            None,
        )
        .unwrap();
    let props = props! { "since" => 2020, "date" => 1744965900, "name" => "hello"};
    let _edge = G::new_mut(Arc::clone(&storage), wv.write_mut())
        .add_e(
            "knows",
            props.clone(),
            Some(v6_uuid()),
            node1.id(),
            node2.id,
            false,
            EdgeType::Std,
        )
        .collect_to::<Vec<_>>();

    wv.commit().unwrap();
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let edge = G::new_from(Arc::clone(&storage), &storage.read_view(&txn), vec![node1])
        .out_e("knows")
        .filter_ref(|val, _| {
            if let Ok(val) = val {
                println!("val: {:?}", val.check_property("date"));
                val.check_property("date").map_or(false, |v| {
                    println!("v: {:?}", v);
                    println!("v: {:?}", *v == 1743290007);
                    *v >= 1743290007
                })
            } else {
                false
            }
        })
        .collect_to::<Vec<_>>();
    let edge = edge.first().unwrap();
    match edge {
        TraversalVal::Edge(edge) => {
            assert_eq!(edge.properties, props.into_iter().collect());
        }
        _ => {
            panic!("Expected Edge value");
        }
    }
}

// #[test]
// fn test_shortest_mutual_path() {
//     let (storage, _temp_dir) = setup_test_db();
//     let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

//     // Create a complex network of mutual and one-way connections
//     // Mutual: Alice <-> Bob <-> Charlie <-> David
//     // One-way: Alice -> Eve -> David
//     let users: Vec<Node> = vec!["alice", "bob", "charlie", "dave", "eve"]
//         .iter()
//         .map(|name| {
//             storage
//                 .create_node(&mut txn, "person", props! { "name" => *name }, None, None)
//                 .unwrap()
//         })
//         .collect();

//     for (i, j) in [(0, 1), (1, 2), (2, 3)].iter() {
//         storage
//             .create_edge(&mut txn, "knows", &users[*i].id, &users[*j].id, props!())
//             .unwrap();
//         storage
//             .create_edge(&mut txn, "knows", &users[*j].id, &users[*i].id, props!())
//             .unwrap();
//     }

//     storage
//         .create_edge(&mut txn, "knows", &users[0].id, &users[4].id, props!())
//         .unwrap();
//     storage
//         .create_edge(&mut txn, "knows", &users[4].id, &users[3].id, props!())
//         .unwrap();

//     txn.commit().unwrap();

//     let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
//     let mut tr =
//         TraversalBuilder::new(Arc::clone(&storage), TraversalValue::from(users[0].clone()));
//     tr.shortest_mutual_path_to(&txn, &users[3].id);

//     let result = tr.result(txn);
//     let paths = match result.unwrap() {
//         TraversalValue::Paths(paths) => paths,
//         _ => {
//             panic!("Expected PathArray value")
//         }
//     };

//     assert_eq!(paths.len(), 1);
//     let (nodes, edges) = &paths[0];

//     assert_eq!(nodes.len(), 4);
//     assert_eq!(edges.len(), 3);
//     assert_eq!(nodes[0].id, users[3].id); // David
//     assert_eq!(nodes[1].id, users[2].id); // Charlie
//     assert_eq!(nodes[2].id, users[1].id); // Bob
//     assert_eq!(nodes[3].id, users[0].id); // Alice
// }

#[test]
#[ignore = "stress benchmark, not suitable for routine unit test runs"]
fn huge_traversal() {
    let (storage, _temp_dir) = setup_test_db();
    let mut wv = storage.begin_write_view().unwrap();

    let mut nodes = Vec::with_capacity(65_000_000);
    let mut start = Instant::now();

    for _ in 0..100_000 {
        // nodes.push(Node::new("person", props! { "name" => i}));
        nodes.push(v6_uuid());
    }
    println!("time taken to initialise nodes: {:?}", start.elapsed());
    start = Instant::now();
    nodes.sort();
    println!("time taken to sort nodes: {:?}", start.elapsed());
    let now = Instant::now();
    let _res = G::new_mut(Arc::clone(&storage), wv.write_mut())
        .bulk_add_n(&mut nodes, None, 1000000)
        .map(|res| res.unwrap())
        .collect::<Vec<_>>();
    wv.commit().unwrap();
    println!("time taken to add nodes: {:?}", now.elapsed());
    let start = Instant::now();
    let mut edges = Vec::with_capacity(6000 * 2000);
    for _ in 0..100_000_000 {
        let random_node1 = &nodes[rand::rng().random_range(0..nodes.len())];
        let random_node2 = &nodes[rand::rng().random_range(0..nodes.len())];
        // edges.push(Edge {
        //     id: v6_uuid(),
        //     label: "knows".to_string(),
        //     properties: HashMap::new(),
        //     from_node: random_node1.id,
        //     to_node: random_node2.id,
        // });
        edges.push((*random_node1, *random_node2, v6_uuid()));
    }
    println!(
        "time taken to create {} edges: {:?}",
        edges.len(),
        start.elapsed()
    );
    let start = Instant::now();
    let mut wv = storage.begin_write_view().unwrap();
    let _res = G::new_mut(Arc::clone(&storage), wv.write_mut())
        .bulk_add_e(edges, false, 1000000)
        .map(|res| res.unwrap())
        .collect::<Vec<_>>();
    wv.commit().unwrap();
    println!("time taken to add edges: {:?}", start.elapsed());

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let now = Instant::now();
    let traversal = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n()
        .out_e("knows")
        .to_n()
        .out("knows")
        // .filter_ref(|val, _| {
        //     if let Ok(TraversalVal::Node(node)) = val {
        //         if let Some(value) = node.check_property("name") {
        //             match value {
        //                 Value::I32(name) => return *name < 700000,
        //                 _ => return false,
        //             }
        //         } else {
        //             return false;
        //         }
        //     } else {
        //         return false;
        //     }
        // })
        .out("knows")
        .out("knows")
        .out("knows")
        .out("knows")
        .dedup()
        .range(0, 10000)
        .count();
    println!("optimized version time: {:?}", now.elapsed());
    println!("traversal: {:?}", traversal);
    println!(
        "size of mdb file on disk: {:?}",
        storage.lmdb_env().unwrap().real_disk_size()
    );
    txn.commit().unwrap();

    // let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    // let now = Instant::now();
    // let mut tr = TraversalBuilder::new(Arc::clone(&storage), TraversalValue::Empty);
    // tr.v(&txn)
    //     .out_e(&txn, "knows")
    //     .in_v(&txn)
    //     .out(&txn, "knows")
    //     .filter_nodes(&txn, |val| {
    //         if let Some(value) = val.check_property("name") {
    //             match value {
    //                 Value::I32(name) => return Ok(*name < 1000),
    //                 _ => return Err(GraphError::Default),
    //             }
    //         } else {
    //             return Err(GraphError::Default);
    //         }
    //     })
    //     .out(&txn, "knows")
    //     .out(&txn, "knows")
    //     .out(&txn, "knows")
    //     .out(&txn, "knows")
    //     .range(0, 100);

    // let result = tr.finish();
    // println!("original version time: {:?}", now.elapsed());
    // println!(
    //     "traversal: {:?}",
    //     match result {
    //         Ok(TraversalValue::NodeArray(nodes)) => nodes.len(),
    //         Err(e) => {
    //             println!("error: {:?}", e);
    //             0
    //         }
    //         _ => {
    //             println!("error: {:?}", result);
    //             0
    //         }
    //     }
    // );
    // // print size of mdb file on disk
    // println!(
    //     "size of mdb file on disk: {:?}",
    //     storage.lmdb_env().unwrap().real_disk_size()
    // );
    assert!(false);
}

#[test]
fn test_n_from_types_filters_by_label() {
    use crate::helix_engine::graph_core::ops::source::n_from_types::NFromTypesAdapter;

    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
    let p1 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let _c1 = storage
        .create_node(&mut txn, "company", props!(), None, None)
        .unwrap();
    let p2 = storage
        .create_node(&mut txn, "person", props!(), None, None)
        .unwrap();
    let _c2 = storage
        .create_node(&mut txn, "company", props!(), None, None)
        .unwrap();
    txn.commit().unwrap();

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let mut ids: Vec<u128> = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n_from_types(&["person"])
        .filter_map(|r| r.ok())
        .map(|tv| tv.id())
        .collect();
    ids.sort_unstable();
    let mut expected = vec![p1.id, p2.id];
    expected.sort_unstable();
    assert_eq!(ids, expected, "n_from_types returns only 'person' nodes");

    // count via the adapter
    let count = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n_from_types(&["person"])
        .count();
    assert_eq!(count, 2);

    // a label that matches nothing yields an empty traversal
    let none = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n_from_types(&["nonexistent"])
        .count();
    assert_eq!(none, 0);
}

#[test]
fn test_e_from_types_new_filters_by_label() {
    use crate::helix_engine::graph_core::ops::source::e_from_types::EFromTypes;

    let (storage, _temp_dir) = setup_test_db();
    let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
    let a = storage
        .create_node(&mut txn, "n", props!(), None, None)
        .unwrap();
    let b = storage
        .create_node(&mut txn, "n", props!(), None, None)
        .unwrap();
    let e_knows = storage
        .create_edge(&mut txn, "knows", &a.id, &b.id, props!())
        .unwrap();
    let _e_likes = storage
        .create_edge(&mut txn, "likes", &a.id, &b.id, props!())
        .unwrap();
    txn.commit().unwrap();

    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let ids: Vec<u128> = EFromTypes::new(&storage, &txn, "knows")
        .filter_map(|r| r.ok())
        .map(|tv| tv.id())
        .collect();
    assert_eq!(
        ids,
        vec![e_knows.id],
        "e_from_types::new returns only 'knows' edges"
    );
}

#[test]
fn drop_edge_removes_only_its_own_adjacency_dup() {
    // Regression test for the user-approved drop_edge fix: dropping ONE of two
    // same-label out edges from a node must leave the OTHER edge's adjacency
    // intact. The prior heed drop_edge did out_edges_db.delete(key), which wipes
    // EVERY dup under the (from|label) key — so the survivor count would be 0.
    let (storage, _temp_dir) = setup_test_db();
    let (from, e_a, e_b) = {
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let from = storage
            .create_node(&mut txn, "n", props!(), None, None)
            .unwrap();
        let a = storage
            .create_node(&mut txn, "n", props!(), None, None)
            .unwrap();
        let b = storage
            .create_node(&mut txn, "n", props!(), None, None)
            .unwrap();
        let e_a = storage
            .create_edge(&mut txn, "knows", &from.id, &a.id, props!())
            .unwrap();
        let e_b = storage
            .create_edge(&mut txn, "knows", &from.id, &b.id, props!())
            .unwrap();
        txn.commit().unwrap();
        (from.id, e_a.id, e_b.id)
    };

    // Both same-label out edges present before the drop.
    {
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let before = G::new(Arc::clone(&storage), &storage.read_view(&txn))
            .n_from_id(&from)
            .out_e("knows")
            .count();
        assert_eq!(before, 2, "both same-label out edges present before drop");
    }

    // Drop ONLY the from->a edge.
    storage
        .with_write_txn(|txn| storage.drop_edge(txn, &e_a))
        .unwrap();

    // The from->b adjacency MUST survive (was 0 with the over-delete bug).
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    let surviving: Vec<u128> = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n_from_id(&from)
        .out_e("knows")
        .filter_map(|r| r.ok())
        .map(|tv| tv.id())
        .collect();
    assert_eq!(
        surviving,
        vec![e_b],
        "dropping from->a must leave exactly the from->b adjacency (e_a={e_a:#x})"
    );
}

#[test]
fn drop_node_keeps_peers_same_label_adjacency() {
    // Regression test for the drop_node seam port: dropping a node tears down its
    // own edges with single-dup adjacency deletes, so a peer edge that merely
    // SHARES an adjacency key with one of the dropped node's edges must survive.
    //
    // Graph (all "knows"):   victim -> p ,  o -> victim ,  o -> p
    // The single edge o->p shares BOTH:
    //   * P's in_edge_key (p|knows) with victim->p  (in-side teardown of victim), and
    //   * O's out_edge_key (o|knows) with o->victim (out-side teardown of victim).
    // The prior heed drop_node did out/in_edges_db.delete(key), wiping every dup
    // under those keys — so o->p would vanish (survivor counts would be 0).
    let (storage, _temp_dir) = setup_test_db();
    let (victim, o, p, e_op) = {
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let victim = storage
            .create_node(&mut txn, "n", props!(), None, None)
            .unwrap();
        let o = storage
            .create_node(&mut txn, "n", props!(), None, None)
            .unwrap();
        let p = storage
            .create_node(&mut txn, "n", props!(), None, None)
            .unwrap();
        // victim's own edges (must be removed by drop_node).
        storage
            .create_edge(&mut txn, "knows", &victim.id, &p.id, props!())
            .unwrap();
        storage
            .create_edge(&mut txn, "knows", &o.id, &victim.id, props!())
            .unwrap();
        // The peer edge that shares adjacency keys with victim's edges.
        let e_op = storage
            .create_edge(&mut txn, "knows", &o.id, &p.id, props!())
            .unwrap();
        txn.commit().unwrap();
        (victim.id, o.id, p.id, e_op.id)
    };

    // Before the drop: O has two out edges, P has two in edges.
    {
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        assert_eq!(
            G::new(Arc::clone(&storage), &storage.read_view(&txn))
                .n_from_id(&o)
                .out_e("knows")
                .count(),
            2,
            "O has two same-label out edges before drop"
        );
        assert_eq!(
            G::new(Arc::clone(&storage), &storage.read_view(&txn))
                .n_from_id(&p)
                .in_e("knows")
                .count(),
            2,
            "P has two same-label in edges before drop"
        );
    }

    // Drop the whole victim node (tears down victim->p and o->victim).
    storage
        .with_write_txn(|txn| storage.drop_node(txn, &victim))
        .unwrap();

    // The victim is gone, but the unrelated o->p edge and BOTH of its adjacency
    // sides survive (each would be 0 under the over-delete bug).
    let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
    assert!(
        storage.get_node(&txn, &victim).is_err(),
        "victim node must be removed"
    );
    let o_out: Vec<u128> = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n_from_id(&o)
        .out_e("knows")
        .filter_map(|r| r.ok())
        .map(|tv| tv.id())
        .collect();
    assert_eq!(
        o_out,
        vec![e_op],
        "O's out adjacency must keep exactly o->p (e_op={e_op:#x})"
    );
    let p_in: Vec<u128> = G::new(Arc::clone(&storage), &storage.read_view(&txn))
        .n_from_id(&p)
        .in_e("knows")
        .filter_map(|r| r.ok())
        .map(|tv| tv.id())
        .collect();
    assert_eq!(
        p_in,
        vec![e_op],
        "P's in adjacency must keep exactly o->p (e_op={e_op:#x})"
    );
}
