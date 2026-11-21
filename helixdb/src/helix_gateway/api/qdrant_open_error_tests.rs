//! What a route answers when its collection is not opened. An absent
//! collection is the JSON 404 it always was. A collection that is there and
//! was not opened is no 404: a client reads that as "gone", caches it, and
//! takes a filter delete for done. An open that failed is a 500, also while
//! its cooldown runs. An open there was no room for is a 503 that says so.

use super::tests::{make_input, setup_with_config, TestContext};
use super::*;
use crate::helix_engine::graph_core::config::Config;
use crate::helix_engine::storage_core::backend::{LsmRole, LsmStorage, StorageBackendConfig};
use crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store;
use crate::helix_engine::storage_core::collection_manager::{
    clear_open_failure_for_test, record_open_failure_for_test,
};
use serde_json::json;
use std::sync::Arc;

type Handler = fn(&HandlerInput, &mut Response) -> Result<(), GraphError>;

const INJECTED: &str = "injected open failure";

/// Sets an environment variable for a test and restores it after.
struct EnvVar(&'static str, Option<String>);

impl EnvVar {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        std::env::set_var(key, value);
        Self(key, previous)
    }
}

impl Drop for EnvVar {
    fn drop(&mut self) {
        match &self.1 {
            Some(value) => std::env::set_var(self.0, value),
            None => std::env::remove_var(self.0),
        }
    }
}

fn call(
    ctx: &TestContext,
    handler: Handler,
    method: &str,
    route: &str,
    name: &str,
    body: &serde_json::Value,
) -> Response {
    let input = make_input(
        ctx,
        method,
        &format!("/collections/{name}/{route}"),
        serde_json::to_vec(body).unwrap(),
        HashMap::from([("name".to_string(), name.to_string())]),
    );
    let mut response = Response::new();
    handler(&input, &mut response).expect("the route answers a failed open itself");
    response
}

/// `handler` on a collection whose last open failed and whose cooldown runs,
/// then on one that does not exist.
fn failed_then_absent(handler: Handler, method: &str, route: &str, body: serde_json::Value) {
    let ctx = setup_with_config(Config::default());
    // The open-failure state is the process's: one name a route.
    let failed = format!("open_failed_{}", route.replace('/', "_"));
    let absent = format!("open_absent_{}", route.replace('/', "_"));
    // Several failures, so the cooldown outlasts a slow machine.
    for _ in 0..5 {
        record_open_failure_for_test(&failed, &GraphError::New(INJECTED.into()));
    }

    let response = call(&ctx, handler, method, route, &failed, &body);
    clear_open_failure_for_test(&failed);
    let text = String::from_utf8_lossy(&response.body).into_owned();
    assert_eq!(response.status, 500, "{text}");
    assert!(text.contains(INJECTED), "{text}");
    assert!(!text.contains("not found"), "{text}");
    // No "later" for an open that failed: the client requeues on that without a bound.
    assert!(!response.headers.contains_key("Retry-After"));

    let response = call(&ctx, handler, method, route, &absent, &body);
    assert_eq!(response.status, 404);
    let answer: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
    let error = answer["status"]["error"].as_str().unwrap();
    assert!(error.contains("not found"), "{error}");
    assert!(answer["result"].is_null());
}

#[test]
fn search_answers_a_failed_open_500_and_an_absent_collection_404() {
    failed_then_absent(
        handle_search_points,
        "POST",
        "points/search",
        json!({"vector": [1.0, 0.0, 0.0]}),
    );
}

#[test]
fn scroll_answers_a_failed_open_500_and_an_absent_collection_404() {
    failed_then_absent(handle_scroll_points, "POST", "points/scroll", json!({}));
}

#[test]
fn scan_answers_a_failed_open_500_and_an_absent_collection_404() {
    failed_then_absent(handle_scan_points, "POST", "points/scan", json!({}));
}

#[test]
fn facet_answers_a_failed_open_500_and_an_absent_collection_404() {
    failed_then_absent(handle_facet, "POST", "facet", json!({"key": "path"}));
}

/// The one write among them: a 404 here told the caller its delete was done.
#[test]
fn filter_delete_answers_a_failed_open_500_and_an_absent_collection_404() {
    failed_then_absent(
        handle_delete_points,
        "POST",
        "points/delete",
        json!({"filter": {}}),
    );
}

#[test]
fn query_answers_a_failed_open_500_and_an_absent_collection_404() {
    failed_then_absent(
        handle_query_points,
        "POST",
        "points/query",
        json!({"query": [1.0, 0.0, 0.0]}),
    );
}

#[test]
fn hybrid_query_answers_a_failed_open_500_and_an_absent_collection_404() {
    failed_then_absent(
        handle_hybrid_query_points,
        "POST",
        "points/hybrid_query",
        json!({"dense": [{"query": [1.0, 0.0, 0.0]}]}),
    );
}

/// No room among the open collections passes by itself, so it alone is a
/// "later": 503 with `Retry-After`, in the words the client takes for backpressure.
#[test]
#[serial_test::serial]
fn query_answers_an_open_with_no_room_503_as_backpressure() {
    let ctx = setup_with_config(Config::default());
    let name = "open_no_room";
    drop(ctx.collections.create_collection(name).unwrap());
    ctx.collections.evict_collection(name);
    // In use, so it is not evicted: with a limit of one it is the room.
    let held = ctx
        .collections
        .create_collection("open_no_room_holder")
        .unwrap();

    // The limit is the process's, so it holds for this one request only. A
    // test beside this one that fails with "capacity saturated" met it.
    let response = {
        let _limit = EnvVar::set("HELIX_MAX_OPEN_COLLECTIONS", "1");
        let _wait = EnvVar::set("HELIX_COLLECTION_OPEN_WAIT_MS", "0");
        call(
            &ctx,
            handle_query_points,
            "POST",
            "points/query",
            name,
            &json!({"query": [1.0, 0.0, 0.0]}),
        )
    };
    drop(held);

    let text = String::from_utf8_lossy(&response.body).into_owned();
    assert_eq!(response.status, 503, "{text}");
    assert_eq!(
        response.headers.get("Retry-After").map(String::as_str),
        Some("1")
    );
    assert!(text.contains("backpressure"), "{text}");
    assert!(!text.contains("not found"), "{text}");
}

/// The eighth route opens nothing: it reads the snapshot directory. A read
/// that fails is a 500, and only a collection without a directory is absent.
#[test]
fn snapshot_listing_answers_an_unreadable_directory_500_and_an_absent_collection_404() {
    let ctx = setup_with_config(Config::default());
    let name = "open_error_snapshots";
    let directory = {
        let storage = ctx.collections.create_collection(name).unwrap();
        storage.lmdb_env().unwrap().path().to_path_buf()
    };
    let snapshots = directory.join("snapshots");
    let _ = std::fs::remove_dir_all(&snapshots);
    std::fs::write(&snapshots, b"not a directory").unwrap();

    let response = call(
        &ctx,
        handle_list_snapshots,
        "GET",
        "snapshots",
        name,
        &json!({}),
    );
    assert_eq!(
        response.status,
        500,
        "{}",
        String::from_utf8_lossy(&response.body)
    );

    let response = call(
        &ctx,
        handle_list_snapshots,
        "GET",
        "snapshots",
        "open_error_snapshots_absent",
        &json!({}),
    );
    assert_eq!(response.status, 404);
    let answer: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
    assert!(answer["status"]["error"]
        .as_str()
        .unwrap()
        .contains("not found"));
}

/// A directory with nothing loaded from it and no metadata sidecar: what a
/// collection whose open never got through leaves on a pod.
fn never_opened(ctx: &TestContext, name: &str) {
    let directory = ctx.collections.data_dir().join("collections").join(name);
    std::fs::create_dir_all(directory).unwrap();
}

/// The info route's answer for `name` after its last open ended with `open`.
fn info_after(ctx: &TestContext, name: &str, open: Option<GraphError>) -> (u16, String, bool) {
    // Several failures, so the cooldown outlasts a slow machine.
    for _ in 0..5 {
        if let Some(error) = &open {
            record_open_failure_for_test(name, error);
        }
    }
    let response = call(ctx, handle_get_collection, "GET", "", name, &json!({}));
    clear_open_failure_for_test(name);
    (
        response.status,
        String::from_utf8_lossy(&response.body).into_owned(),
        response.headers.contains_key("Retry-After"),
    )
}

const NO_SIDECAR: &str = "metadata sidecar is not available";

/// The info route opens nothing: it answers from the loaded collection or
/// from its metadata sidecar. A collection whose open fails leaves neither,
/// and "not available" sent the client away to come back later, for ever. Once an
/// open of it has failed on this writer, the info route answers that
/// failure, as every route that opens does.
#[test]
fn collection_info_answers_the_open_that_failed_where_nothing_was_ever_opened() {
    let ctx = setup_with_config(Config::default());
    let name = "open_failed_info";
    never_opened(&ctx, name);

    // No open was tried: all that is known is that there is no sidecar.
    let (status, text, _) = info_after(&ctx, name, None);
    assert_eq!(status, 503, "{text}");
    assert!(text.contains(NO_SIDECAR), "{text}");

    let failed = GraphError::New(INJECTED.into());
    let (status, text, later) = info_after(&ctx, name, Some(failed));
    assert_eq!(status, 500, "{text}");
    assert!(
        text.contains(INJECTED) && !text.contains(NO_SIDECAR),
        "{text}"
    );
    // No "later" for an open that failed: the client requeues on that without a bound.
    assert!(!later);

    // An open that found no manifest under the name: the collection is absent.
    let absent = GraphError::New(format!(
        "Collection '{name}' not found (no manifest in object store)"
    ));
    let (status, text, _) = info_after(&ctx, name, Some(absent));
    assert_eq!(status, 404, "{text}");
    let answer: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert!(answer["status"]["error"]
        .as_str()
        .unwrap()
        .contains("not found"));

    // An open that only gave up waiting for the last handle to close did not
    // fail, and says nothing of the collection.
    let waited = GraphError::OpenDeferred {
        retry_after_secs: 1,
        message: "backpressure, still closing".into(),
    };
    let (status, text, _) = info_after(&ctx, name, Some(waited));
    assert_eq!(status, 503, "{text}");
    assert!(text.contains(NO_SIDECAR), "{text}");
}

/// A reader has no sidecar for a collection it has not loaded, and falls
/// back on the writer's answer: its own last open says nothing here.
#[test]
#[serial_test::serial]
fn collection_info_of_a_reader_is_not_available_whatever_its_last_open() {
    let _store = set_lsm_test_object_store(Arc::new(object_store::memory::InMemory::new()));
    let config = |role| {
        Config::default().with_storage_backend(StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role,
        })
    };
    // A reader opens the root database its writer made.
    let _writer = setup_with_config(config(LsmRole::Writer));
    let ctx = setup_with_config(config(LsmRole::Reader));
    let name = "open_failed_info_reader";
    never_opened(&ctx, name);

    let failed = GraphError::New(INJECTED.into());
    let (status, text, _) = info_after(&ctx, name, Some(failed));
    assert_eq!(status, 503, "{text}");
    assert!(text.contains(NO_SIDECAR), "{text}");
}

/// The root engine's name and a name that is not one path segment are the
/// caller's mistake on these routes too, not an absent collection.
#[test]
fn a_name_that_cannot_be_a_collection_answers_400_on_a_read() {
    let ctx = setup_with_config(Config::default());
    let response = call(
        &ctx,
        handle_scroll_points,
        "POST",
        "points/scroll",
        "..",
        &json!({}),
    );
    assert_eq!(
        response.status,
        400,
        "{}",
        String::from_utf8_lossy(&response.body)
    );
}
