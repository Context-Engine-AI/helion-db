//! The adjacency backfill's mark in a collection's info,
//! `config.metadata.helix_graph.adjacency_backfill_complete`: `true` once
//! `/v1/graph/backfill_adjacency_from_points` has run to its end, `false`
//! while it has not, and `null` where the answering process does not know.
//!
//! A caller reads it at a reader instead of opening the graph on the writer
//! to ask, so the answer reads no row and opens nothing: a loaded collection
//! answers from memory, any other from its metadata sidecar. What the object
//! store's sidecar says, and when, is held in the collection manager's
//! sidecar tests.

use super::hybrid_parallel_tests::EnvVar;
use super::tests::{make_input, setup_with_config, TestContext};
use super::*;
use crate::helix_engine::graph_core::config::Config;
use crate::helix_engine::storage_core::backend::{LsmRole, LsmStorage, StorageBackendConfig};
use crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store;
use crate::helix_engine::storage_core::backend_lsm::{
    allow_lsm_blocking, lsm_round_trips_for_test as round_trips,
};
use crate::helix_engine::storage_core::upsert::NodeUpsert;
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};

const GRAPH: &str = "calls_graph";

/// A relationship row, which the adjacency backfill derives an edge from.
fn relationship_row(id: u128, callee: &str) -> NodeUpsert {
    let text = |value: &str| Value::String(value.to_string());
    NodeUpsert {
        id,
        label: "point".into(),
        properties: HashMap::from([
            ("caller_symbol".to_string(), text("a_fn")),
            ("callee_symbol".to_string(), text(callee)),
            ("caller_path".to_string(), text("src/a.rs")),
            ("callee_path".to_string(), text("src/b.rs")),
            ("edge_type".to_string(), text("calls")),
        ]),
    }
}

fn graph_of_two_rows(ctx: &TestContext) -> Arc<HelixGraphStorage> {
    let storage = ctx.collections.create_collection(GRAPH).unwrap();
    storage
        .with_write_backend(|w| {
            storage.upsert_node_be(w, &relationship_row(1, "b_fn"))?;
            storage.upsert_node_be(w, &relationship_row(2, "c_fn"))?;
            Ok(())
        })
        .unwrap();
    storage
}

/// One call of the backfill, a row a call: whether the pass is over.
fn call(storage: &HelixGraphStorage, force: bool) -> bool {
    storage
        .backfill_adjacency_from_points_batch(GRAPH, 1, force, false)
        .unwrap()
        .2
}

/// The mark in the collection's info, and the storage round trips that the
/// info took.
fn info_mark(ctx: &TestContext) -> (serde_json::Value, usize) {
    let input = make_input(
        ctx,
        "GET",
        &format!("/collections/{GRAPH}"),
        Vec::new(),
        HashMap::from([("name".into(), GRAPH.into())]),
    );
    let mut response = Response::new();
    let before = round_trips();
    allow_lsm_blocking(|| handle_get_collection(&input, &mut response)).unwrap();
    let took = round_trips() - before;
    assert_eq!(response.status, 200);
    let body: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
    let mark = &body["result"]["config"]["metadata"]["helix_graph"]["adjacency_backfill_complete"];
    (mark.clone(), took)
}

/// On the writer a loaded collection answers the mark from memory, with no
/// storage round trip: "not complete" until the pass's last call is stored,
/// "complete" from then on, and "not complete" again once a forced call has
/// started the pass over.
#[test]
fn a_loaded_collection_answers_the_mark_from_memory() {
    let ctx = setup_with_config(Config::default().with_lsm_in_memory());
    let storage = graph_of_two_rows(&ctx);
    assert_eq!(info_mark(&ctx), (json!(false), 0), "no pass has run");
    while !call(&storage, false) {
        assert_eq!(info_mark(&ctx), (json!(false), 0), "the pass is not over");
    }
    assert_eq!(info_mark(&ctx), (json!(true), 0), "the pass is over");

    assert!(!call(&storage, true), "a forced call starts the pass over");
    assert_eq!(info_mark(&ctx), (json!(false), 0));
    while !call(&storage, false) {}
    assert_eq!(info_mark(&ctx), (json!(true), 0));
}

/// A collection that is not loaded answers the mark its metadata sidecar
/// carries, and the info opens nothing. A sidecar written before the mark
/// was kept in it answers `null`.
#[test]
fn a_collection_that_is_not_loaded_answers_the_mark_of_its_sidecar_and_opens_nothing() {
    let ctx = setup_with_config(Config::default().with_lsm_in_memory());
    let storage = graph_of_two_rows(&ctx);
    while !call(&storage, false) {}
    let sidecar = HelixGraphStorage::metadata_sidecar_path(storage.path());
    drop(storage);
    ctx.collections.evict_collection(GRAPH);
    assert_eq!(ctx.collections.loaded_count(), 0);

    assert_eq!(info_mark(&ctx).0, json!(true));
    assert_eq!(ctx.collections.loaded_count(), 0, "the info opened it");

    let mut older: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&sidecar).unwrap()).unwrap();
    older
        .as_object_mut()
        .unwrap()
        .remove("adjacency_backfill_complete")
        .expect("the sidecar carried the mark");
    std::fs::write(&sidecar, serde_json::to_vec(&older).unwrap()).unwrap();
    assert!(info_mark(&ctx).0.is_null(), "an older sidecar");
    assert_eq!(ctx.collections.loaded_count(), 0, "the info opened it");
}

/// A reader that has the graph open answers the mark that its view of the
/// store holds: read at its open, and again when its view is refreshed.
#[test]
#[serial_test::serial]
fn a_reader_answers_the_mark_that_its_view_holds() {
    let _store = set_lsm_test_object_store(Arc::new(object_store::memory::InMemory::new()));
    let _prefix = EnvVar::set("HELIX_LSM_PREFIX", "backfill_mark_reader");
    let _poll = EnvVar::set("HELIX_LSM_READER_MANIFEST_POLL_MS", "500");
    let config = |role| {
        Config::default().with_storage_backend(StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role,
        })
    };
    let writer = setup_with_config(config(LsmRole::Writer));
    let storage = graph_of_two_rows(&writer);
    while !call(&storage, false) {}

    let reader = setup_with_config(config(LsmRole::Reader));
    let replica = reader.collections.get_collection(GRAPH).unwrap();
    assert!(replica.backend.is_reader_replica());
    assert_eq!(info_mark(&reader).0, json!(true), "read at its open");

    assert!(!call(&storage, true), "the writer starts the pass over");
    // The slack is for a stalled runner: the reader polls every 500 ms.
    let started = Instant::now();
    loop {
        allow_lsm_blocking(|| replica.refresh_reader_view_best_effort());
        if info_mark(&reader).0 == json!(false) {
            break;
        }
        assert!(
            started.elapsed() < Duration::from_secs(15),
            "the reader still says complete"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}
