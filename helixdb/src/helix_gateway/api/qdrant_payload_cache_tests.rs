use super::tests::{
    create_dense_collection_lsm, hybrid_query_points_raw, make_input, setup_with_config,
    upsert_mixed_points,
};
use super::*;
use crate::helix_engine::graph_core::config::Config;
use crate::helix_engine::storage_core::backend_any::AnyBackend;
use std::collections::{BTreeMap, HashMap};

fn call_qdrant(
    ctx: &super::tests::TestContext,
    method: &str,
    path: &str,
    name: &str,
    body: serde_json::Value,
    handler: fn(&HandlerInput, &mut Response) -> Result<(), GraphError>,
) -> serde_json::Value {
    let input = make_input(
        ctx,
        method,
        path,
        sonic_rs::to_vec(&body).unwrap(),
        HashMap::from([("name".into(), name.into())]),
    );
    let mut response = Response::new();
    handler(&input, &mut response).unwrap();
    serde_json::from_slice(&response.body).unwrap_or(serde_json::Value::Null)
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

fn set_payload(ctx: &super::tests::TestContext, collection: &str, id: &str, label: &str) {
    call_qdrant(
        ctx,
        "POST",
        &format!("/collections/{collection}/points/payload"),
        collection,
        serde_json::json!({"points": [id], "payload": {"label": label}}),
        handle_set_payload,
    );
}

fn request(with_payload: bool) -> serde_json::Value {
    serde_json::json!({
        "dense": [{"using": "dense", "query": [1.0, 0.0, 0.0], "limit": 2}],
        "sparse": [{"using": "lex_sparse", "query": {"indices": [13], "values": [1.0]}, "limit": 3}],
        "limit": 3,
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

fn payload_labels(points: &serde_json::Value) -> BTreeMap<String, String> {
    points
        .as_array()
        .unwrap()
        .iter()
        .map(|point| {
            (
                point["id"].as_str().unwrap().to_string(),
                point["payload"]["label"].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

fn assert_payload_labels(points: &serde_json::Value, expected: &[(&str, &str)]) {
    let expected: BTreeMap<String, String> = expected
        .iter()
        .map(|(id, label)| ((*id).to_string(), (*label).to_string()))
        .collect();
    assert_eq!(payload_labels(points), expected);
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
fn hybrid_query_lsm_reuses_payload_cache_across_overlapping_response_stages() {
    let ctx = setup_with_config(Config::default().with_lsm_in_memory());
    create_dense_collection_lsm(&ctx, "lsm_hybrid_payload_cache");
    add_sparse_vector(&ctx, "lsm_hybrid_payload_cache");
    upsert_mixed_points(
        &ctx,
        "lsm_hybrid_payload_cache",
        serde_json::json!([
            hybrid_point("overlap-a", [1.0, 0.0, 0.0], 10.0),
            hybrid_point("overlap-b", [0.99, 0.01, 0.0], 9.0),
            hybrid_point("sparse-only", [0.0, 1.0, 0.0], 8.0),
        ]),
    );

    let storage = ctx
        .collections
        .get_collection("lsm_hybrid_payload_cache")
        .unwrap();
    let AnyBackend::Lsm(lsm) = storage.backend.as_ref() else {
        panic!("test collection must use LSM backend");
    };

    let (_status, without_payload) =
        hybrid_query_points_raw(&ctx, "lsm_hybrid_payload_cache", request(false));
    for points in [
        &without_payload["result"]["points"],
        &without_payload["result"]["channels"]["dense"][0]["points"],
        &without_payload["result"]["channels"]["sparse"][0]["points"],
    ] {
        assert!(points
            .as_array()
            .unwrap()
            .iter()
            .all(|point| point.get("payload").is_none()));
    }

    let (_status, with_payload) =
        hybrid_query_points_raw(&ctx, "lsm_hybrid_payload_cache", request(true));
    for (with_points, without_points) in [
        (
            &with_payload["result"]["points"],
            &without_payload["result"]["points"],
        ),
        (
            &with_payload["result"]["channels"]["dense"][0]["points"],
            &without_payload["result"]["channels"]["dense"][0]["points"],
        ),
        (
            &with_payload["result"]["channels"]["sparse"][0]["points"],
            &without_payload["result"]["channels"]["sparse"][0]["points"],
        ),
    ] {
        assert_eq!(point_sequence(with_points), point_sequence(without_points));
    }

    let overlap_a_id = point_id_to_u128(&serde_json::json!("overlap-a"));
    let overlap_b_id = point_id_to_u128(&serde_json::json!("overlap-b"));
    let sparse_only_id = point_id_to_u128(&serde_json::json!("sparse-only"));
    let overlap_a = format_point_id(overlap_a_id);
    let overlap_b = format_point_id(overlap_b_id);
    let sparse_only = format_point_id(sparse_only_id);
    assert_payload_labels(
        &with_payload["result"]["points"],
        &[
            (&overlap_a, "overlap-a"),
            (&overlap_b, "overlap-b"),
            (&sparse_only, "sparse-only"),
        ],
    );
    assert_payload_labels(
        &with_payload["result"]["channels"]["dense"][0]["points"],
        &[(&overlap_a, "overlap-a"), (&overlap_b, "overlap-b")],
    );
    assert_payload_labels(
        &with_payload["result"]["channels"]["sparse"][0]["points"],
        &[
            (&overlap_a, "overlap-a"),
            (&overlap_b, "overlap-b"),
            (&sparse_only, "sparse-only"),
        ],
    );

    let read = storage.backend.begin_read().unwrap();
    let mut cache = RequestPayloadCache::default();
    lsm.reset_read_count();
    cache
        .get_or_load(
            "lsm_hybrid_payload_cache",
            &storage,
            &read,
            [overlap_a_id, overlap_b_id],
        )
        .unwrap()
        .unwrap();
    let reads_after_dense = lsm.read_count();
    {
        let payloads = cache
            .get_or_load(
                "lsm_hybrid_payload_cache",
                &storage,
                &read,
                [overlap_a_id, overlap_b_id, sparse_only_id],
            )
            .unwrap()
            .unwrap();
        assert_eq!(payloads[&sparse_only_id]["label"], "sparse-only");
    }
    let reads_after_sparse = lsm.read_count();
    assert!(reads_after_sparse > reads_after_dense);
    cache
        .get_or_load(
            "lsm_hybrid_payload_cache",
            &storage,
            &read,
            [overlap_a_id, overlap_b_id, sparse_only_id],
        )
        .unwrap()
        .unwrap();
    assert_eq!(lsm.read_count(), reads_after_sparse);

    set_payload(&ctx, "lsm_hybrid_payload_cache", "overlap-a", "updated-a");
    let (_status, updated) =
        hybrid_query_points_raw(&ctx, "lsm_hybrid_payload_cache", request(true));
    assert!(updated["result"]["points"]
        .as_array()
        .unwrap()
        .iter()
        .any(|point| point["payload"]["label"] == "updated-a"));

    create_dense_collection_lsm(&ctx, "lsm_hybrid_payload_cache_other");
    upsert_mixed_points(
        &ctx,
        "lsm_hybrid_payload_cache_other",
        serde_json::json!([{
            "id": "overlap-a",
            "vector": {"dense": [1.0, 0.0, 0.0]},
            "payload": {"label": "other-collection"}
        }]),
    );
    let other = call_qdrant(
        &ctx,
        "POST",
        "/collections/lsm_hybrid_payload_cache_other/points/hybrid_query",
        "lsm_hybrid_payload_cache_other",
        serde_json::json!({
            "dense": [{"using": "dense", "query": [1.0, 0.0, 0.0], "limit": 1}],
            "limit": 1,
            "with_payload": true
        }),
        handle_hybrid_query_points,
    );
    assert_eq!(
        other["result"]["points"][0]["payload"]["label"],
        "other-collection"
    );
}
