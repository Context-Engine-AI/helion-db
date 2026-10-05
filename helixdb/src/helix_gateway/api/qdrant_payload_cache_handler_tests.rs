use super::tests::{
    create_dense_collection_lsm, hybrid_query_points_raw, make_input, setup_with_config,
    upsert_mixed_points,
};
use super::*;
use crate::helix_engine::graph_core::config::Config;
use crate::helix_engine::storage_core::backend_any::AnyBackend;
use std::collections::HashMap;

fn call_qdrant(
    ctx: &super::tests::TestContext,
    method: &str,
    path: &str,
    name: &str,
    body: serde_json::Value,
    handler: fn(&HandlerInput, &mut Response) -> Result<(), GraphError>,
) {
    let input = make_input(
        ctx,
        method,
        path,
        sonic_rs::to_vec(&body).unwrap(),
        HashMap::from([("name".into(), name.into())]),
    );
    let mut response = Response::new();
    handler(&input, &mut response).unwrap();
}

fn add_sparse_vector(ctx: &super::tests::TestContext, name: &str) {
    call_qdrant(
        ctx,
        "PATCH",
        &format!("/collections/{name}"),
        name,
        serde_json::json!({"sparse_vectors": {"lex_sparse": {"modifier": "idf"}}}),
        handle_update_collection,
    );
}

fn request(with_payload: bool) -> serde_json::Value {
    serde_json::json!({
        "dense": [{"using": "dense", "query": [1.0, 0.0, 0.0], "limit": 2}],
        "sparse": [{"using": "lex_sparse", "query": {"indices": [13], "values": [1.0]}, "limit": 2}],
        "limit": 2,
        "with_payload": with_payload,
        "with_channel_points": true
    })
}

fn point_sequence(points: &serde_json::Value) -> Vec<(String, serde_json::Value)> {
    points
        .as_array()
        .unwrap()
        .iter()
        .map(|point| {
            (
                point["id"].as_str().unwrap().to_string(),
                point["score"].clone(),
            )
        })
        .collect()
}

fn hybrid_point(id: &str, dense: [f64; 3], sparse: f64) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "vector": {
            "dense": dense,
            "lex_sparse": {"indices": [13], "values": [sparse]}
        },
        "payload": {"label": id}
    })
}

#[test]
fn hybrid_query_lsm_handler_batches_payload_reads_once_per_new_response_stage() {
    let ctx = setup_with_config(Config::default().with_lsm_in_memory());
    create_dense_collection_lsm(&ctx, "lsm_hybrid_payload_handler_cache");
    add_sparse_vector(&ctx, "lsm_hybrid_payload_handler_cache");
    upsert_mixed_points(
        &ctx,
        "lsm_hybrid_payload_handler_cache",
        serde_json::json!([
            hybrid_point("overlap-a", [1.0, 0.0, 0.0], 10.0),
            hybrid_point("overlap-b", [0.99, 0.01, 0.0], 9.0),
        ]),
    );

    let storage = ctx
        .collections
        .get_collection("lsm_hybrid_payload_handler_cache")
        .unwrap();
    let AnyBackend::Lsm(lsm) = storage.backend.as_ref() else {
        panic!("test collection must use LSM backend");
    };

    let (_status, without_payload) =
        hybrid_query_points_raw(&ctx, "lsm_hybrid_payload_handler_cache", request(false));
    let (_status, with_payload) =
        hybrid_query_points_raw(&ctx, "lsm_hybrid_payload_handler_cache", request(true));
    assert_eq!(
        point_sequence(&with_payload["result"]["points"]),
        point_sequence(&without_payload["result"]["points"])
    );

    lsm.reset_read_count();
    let (_status, without_payload) =
        hybrid_query_points_raw(&ctx, "lsm_hybrid_payload_handler_cache", request(false));
    let reads_without_payload = lsm.read_count();

    lsm.reset_read_count();
    let (_status, with_payload) =
        hybrid_query_points_raw(&ctx, "lsm_hybrid_payload_handler_cache", request(true));
    let reads_with_payload = lsm.read_count();
    let payload_reads = reads_with_payload.saturating_sub(reads_without_payload);

    assert_eq!(
        point_sequence(&with_payload["result"]["channels"]["dense"][0]["points"]),
        point_sequence(&without_payload["result"]["channels"]["dense"][0]["points"])
    );
    assert_eq!(
        point_sequence(&with_payload["result"]["channels"]["sparse"][0]["points"]),
        point_sequence(&without_payload["result"]["channels"]["sparse"][0]["points"])
    );
    assert!(
        payload_reads <= 2,
        "handler must share request payload cache across dense, sparse, and fused hydration; \
         reads_without_payload={reads_without_payload}, reads_with_payload={reads_with_payload}, \
         payload_reads={payload_reads}"
    );
}
