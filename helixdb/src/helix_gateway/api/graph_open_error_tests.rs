//! What a graph route answers when its collection is not opened. An absent
//! collection is the JSON 404 it always was: the client reads its `error` as "there
//! is no native graph to keep in step", and a 404 without one as "this Helion
//! has no such route". A collection that is there and was not opened is no
//! 404: of a write, the caller took the clear or the rebuild it had asked for
//! as done; of a read, it noted in its reply that the graph collection does
//! not exist.

use super::tests::{make_input, setup_context_with_config, EnvVarGuard, TestContext};
use super::*;
use crate::helix_engine::graph_core::config::Config;
use crate::helix_engine::storage_core::collection_manager::{
    clear_open_failure_for_test, record_open_failure_for_test,
};
use serde_json::json;

type Handler = fn(&HandlerInput, &mut Response) -> Result<(), GraphError>;

const INJECTED: &str = "injected open failure";

/// The five graph writes, then the seventeen reads.
const ROUTES: [&str; 22] = [
    "backfill_edge_path_index",
    "backfill_adjacency_from_points",
    "rebuild_adjacency_for_paths",
    "delete_by_path",
    "delete_by_paths",
    "callers",
    "callees",
    "importers",
    "definition",
    "transitive_callers",
    "transitive_callees",
    "impact",
    "cycles",
    "dependencies",
    "subclasses",
    "base_classes",
    "pagerank",
    "communities",
    "shortest_path",
    "jaccard",
    "subgraph",
    "distinct_values",
];

/// The handler of `route`, and a body it takes for `collection`.
fn handler_and_body(route: &str, collection: &str) -> (Handler, serde_json::Value) {
    let only = json!({"collection": collection});
    let paths = json!({"collection": collection, "paths": ["src/a.rs"]});
    let symbol = json!({"collection": collection, "symbol": "a_fn"});
    match route {
        "backfill_edge_path_index" => (handle_backfill_edge_path_index, only),
        "backfill_adjacency_from_points" => (handle_backfill_adjacency_from_points, only),
        "rebuild_adjacency_for_paths" => (handle_rebuild_adjacency_for_paths, paths),
        "delete_by_path" => (
            handle_delete_by_path,
            json!({"collection": collection, "path": "src/a.rs"}),
        ),
        "delete_by_paths" => (handle_delete_by_paths, paths),
        "callers" => (handle_callers, symbol),
        "callees" => (handle_callees, symbol),
        "importers" => (handle_importers, symbol),
        "definition" => (handle_definition, symbol),
        "transitive_callers" => (handle_transitive_callers, symbol),
        "transitive_callees" => (handle_transitive_callees, symbol),
        "impact" => (handle_impact, symbol),
        "cycles" => (handle_cycles, symbol),
        "dependencies" => (handle_dependencies, symbol),
        "subclasses" => (handle_subclasses, symbol),
        "base_classes" => (handle_base_classes, symbol),
        "pagerank" => (handle_pagerank, only),
        "communities" => (handle_communities, only),
        "shortest_path" => (
            handle_shortest_path,
            json!({"collection": collection, "from_symbol": "a_fn", "to_symbol": "b_fn"}),
        ),
        "jaccard" => (
            handle_jaccard,
            json!({"collection": collection, "symbol_a": "a_fn", "symbol_b": "b_fn"}),
        ),
        "subgraph" => (handle_subgraph, only),
        "distinct_values" => (
            handle_distinct_values,
            json!({"collection": collection, "field": "repo"}),
        ),
        other => unreachable!("{other} is not one of ROUTES"),
    }
}

fn call(ctx: &TestContext, handler: Handler, body: &serde_json::Value) -> Response {
    let input = make_input(ctx, serde_json::to_vec(body).unwrap());
    let mut response = Response::new();
    handler(&input, &mut response).expect("the route answers a failed open itself");
    response
}

/// The status each of [`ROUTES`] answered, in their order.
fn statuses(answers: &[(&'static str, Response)]) -> Vec<(&'static str, u16)> {
    answers
        .iter()
        .map(|(route, response)| (*route, response.status))
        .collect()
}

fn all(status: u16) -> Vec<(&'static str, u16)> {
    ROUTES.iter().map(|route| (*route, status)).collect()
}

fn text(response: &Response) -> String {
    String::from_utf8_lossy(&response.body).into_owned()
}

/// Each route on a collection whose last open failed and whose cooldown runs.
#[test]
fn a_graph_route_answers_a_failed_open_500() {
    let ctx = setup_context_with_config(Config::default());
    let answers: Vec<_> = ROUTES
        .iter()
        .map(|route| {
            // The open-failure state is the process's: one name a route.
            let failed = format!("graph_open_failed_{route}");
            // Several failures, so the cooldown outlasts a slow machine.
            for _ in 0..5 {
                record_open_failure_for_test(&failed, &GraphError::New(INJECTED.into()));
            }
            let (handler, body) = handler_and_body(route, &failed);
            let response = call(&ctx, handler, &body);
            clear_open_failure_for_test(&failed);
            (*route, response)
        })
        .collect();

    assert_eq!(statuses(&answers), all(500));
    for (route, response) in &answers {
        let text = text(response);
        assert!(text.contains(INJECTED), "{route}: {text}");
        assert!(!text.contains("not found"), "{route}: {text}");
        // No "later" for an open that failed: it would fail again.
        assert!(!response.headers.contains_key("Retry-After"), "{route}");
    }
}

/// No room among the open collections passes by itself: 503 with
/// `Retry-After`, as the point routes answer it.
#[test]
#[serial_test::serial]
fn a_graph_route_answers_an_open_with_no_room_503_with_retry_after() {
    let ctx = setup_context_with_config(Config::default());
    let name = "graph_open_no_room";
    drop(ctx.collections.create_collection(name).unwrap());
    ctx.collections.evict_collection(name);
    // In use, so it is not evicted: with a limit of one it is the room.
    let held = ctx
        .collections
        .create_collection("graph_open_no_room_holder")
        .unwrap();

    let answers: Vec<_> = ROUTES
        .iter()
        .map(|route| {
            let (handler, body) = handler_and_body(route, name);
            // The limit is the process's, so it holds for this one request only.
            let _limit = EnvVarGuard::set("HELIX_MAX_OPEN_COLLECTIONS", "1");
            let _wait = EnvVarGuard::set("HELIX_COLLECTION_OPEN_WAIT_MS", "0");
            (*route, call(&ctx, handler, &body))
        })
        .collect();
    drop(held);

    assert_eq!(statuses(&answers), all(503));
    for (route, response) in &answers {
        let text = text(response);
        assert_eq!(
            response.headers.get("Retry-After").map(String::as_str),
            Some("1"),
            "{route}"
        );
        assert!(text.contains("backpressure"), "{route}: {text}");
        assert!(!text.contains("not found"), "{route}: {text}");
    }
}

/// The answer the client reads by its body. With the `error` it is a collection that
/// is not there; without one the client takes the route for missing and, for the
/// path index, never asks this Helion again.
#[test]
fn a_graph_route_answers_an_absent_collection_the_json_404_it_always_gave() {
    let ctx = setup_context_with_config(Config::default());
    for route in ROUTES {
        let absent = format!("graph_open_absent_{route}");
        let (handler, body) = handler_and_body(route, &absent);
        let response = call(&ctx, handler, &body);

        assert_eq!(response.status, 404, "{route}");
        assert_eq!(
            response.headers.get("Content-Type").map(String::as_str),
            Some("application/json"),
            "{route}"
        );
        let answer: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(
            answer,
            json!({"error": format!("Graph error: Collection '{absent}' not found")}),
            "{route}"
        );
    }
}

/// A name that is not one path segment is the caller's mistake here as on
/// the point routes, not a collection that is absent.
#[test]
fn a_graph_route_answers_a_name_that_cannot_be_a_collection_400() {
    let ctx = setup_context_with_config(Config::default());
    let answers: Vec<_> = ROUTES
        .iter()
        .map(|route| {
            let (handler, body) = handler_and_body(route, "..");
            (*route, call(&ctx, handler, &body))
        })
        .collect();
    assert_eq!(statuses(&answers), all(400));
}

/// A failure that is a 404 for the router and is not the manager's "absent"
/// is rewritten so that it does not read as "the collection is gone". Where
/// the rewritten error is a 404 again (here a vector error over a database
/// that is missing), the answer is still the route's own: left to the router
/// it had a plain-text body, and no `error` for the client to read.
#[test]
fn a_404_that_is_left_after_the_rewrite_is_the_routes_json_404() {
    let ctx = setup_context_with_config(Config::default());
    let input = make_input(&ctx, Vec::new());
    let mut response = Response::new();
    let error = GraphError::VectorError("segment not found: database does not exist".into());
    collection_open_err(&input, &mut response, error).unwrap();

    assert_eq!(response.status, 404);
    let answer: serde_json::Value =
        serde_json::from_slice(&response.body).expect("the route's JSON body");
    let text = answer["error"].as_str().expect("an `error` in it");
    assert!(text.contains("database does not exist"), "{text}");
}
