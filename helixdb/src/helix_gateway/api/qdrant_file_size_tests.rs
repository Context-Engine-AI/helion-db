use super::tests::*;
use super::*;
use crate::helix_engine::graph_core::config::Config;

fn retrieve(ctx: &TestContext) -> serde_json::Value {
    let input = make_input(
        ctx,
        "POST",
        "/collections/repo/points",
        serde_json::to_vec(
            &serde_json::json!({"ids": [11, 12], "with_payload": true, "with_vector": true}),
        )
        .unwrap(),
        HashMap::from([("name".into(), "repo".into())]),
    );
    let mut response = Response::new();
    handle_get_points(&input, &mut response).unwrap();
    assert_eq!(response.status, 200);
    serde_json::from_slice::<serde_json::Value>(&response.body).unwrap()["result"].clone()
}

fn repair(ctx: &TestContext, points: serde_json::Value) -> (u16, serde_json::Value) {
    let input = make_input(ctx, "POST", "/collections/repo/points/repair-file-size", serde_json::to_vec(&serde_json::json!({
        "points": points, "expected_file_hash": "H1", "file_size_bytes": 42,
        "filter": {"must": [{"key": "metadata.repo_id", "match": {"value": "git:repo"}}, {"key": "metadata.path", "match": {"value": "src/lib.rs"}}]}
    })).unwrap(), HashMap::from([("name".into(), "repo".into())]));
    let mut response = Response::new();
    handle_repair_file_size(&input, &mut response).unwrap();
    (
        response.status,
        serde_json::from_slice(&response.body).unwrap(),
    )
}

#[test]
fn file_size_repair_is_all_or_none_and_preserves_vectors_and_metadata() {
    for config in [Config::default(), Config::default().with_lsm_in_memory()] {
        let ctx = setup_with_config(config);
        if ctx.collections.backend_kind() == BackendKind::Lsm {
            create_dense_collection_lsm(&ctx, "repo");
        } else {
            create_dense_collection(&ctx, "repo");
        }
        let points = serde_json::json!([
            {"id": 11, "vector": {"dense": [0.1, 0.2, 0.3]}, "payload": {"document": "keep canonical", "metadata": {"file_hash": "H1", "file_size_bytes": 1, "path": "src/lib.rs", "repo_id": "git:repo", "legacy_marker": "keep"}}},
            {"id": 12, "vector": {"dense": [0.3, 0.2, 0.1]}, "payload": {"document": "keep alias", "metadata": {"file_hash": "H1", "file_size_bytes": 1, "path": "src/lib.rs", "repo_id": "git:repo", "legacy_marker": "keep"}}}
        ]);
        upsert_points(&ctx, "repo", points.clone());
        let before = retrieve(&ctx);
        assert_eq!(repair(&ctx, serde_json::json!([11, 99])).0, 409);
        assert_eq!(
            retrieve(&ctx),
            before,
            "missing last target rolls back the first patch"
        );
        assert_eq!(repair(&ctx, serde_json::json!([11, 11])).0, 400);
        assert_eq!(retrieve(&ctx), before);
        for size in [serde_json::Value::Null, serde_json::json!("unknown")] {
            let mut legacy = points.clone();
            legacy[1]["payload"]["metadata"]["file_size_bytes"] = size;
            upsert_points(&ctx, "repo", legacy);
            let before = retrieve(&ctx);
            assert_eq!(repair(&ctx, serde_json::json!([11, 12])).0, 409);
            assert_eq!(
                retrieve(&ctx),
                before,
                "unknown legacy sizes do not receive a backfill"
            );
        }
        for (field, value) in [
            ("file_hash", serde_json::json!("H2")),
            ("repo_id", serde_json::json!("git:other")),
            ("path", serde_json::json!("other.rs")),
        ] {
            let mut updated = points.clone();
            updated[1]["payload"]["metadata"][field] = value;
            updated[1]["payload"]["metadata"]["file_size_bytes"] = serde_json::json!(500);
            upsert_points(&ctx, "repo", updated);
            let before = retrieve(&ctx);
            assert_eq!(repair(&ctx, serde_json::json!([11, 12])).0, 409);
            assert_eq!(
                retrieve(&ctx),
                before,
                "stable-ID update must survive stale size repair"
            );
        }
        let mut pending = points.clone();
        pending[1]["payload"]["_pending_prune"] = serde_json::json!(true);
        upsert_points(&ctx, "repo", pending);
        let before = retrieve(&ctx);
        assert_eq!(repair(&ctx, serde_json::json!([11, 12])).0, 409);
        assert_eq!(retrieve(&ctx), before);
        upsert_points(&ctx, "repo", points);
        let mut expected = retrieve(&ctx);
        for point in expected.as_array_mut().unwrap() {
            point["payload"]["metadata"]["file_size_bytes"] = serde_json::json!(42);
        }
        let (status, result) = repair(&ctx, serde_json::json!([11, 12]));
        assert_eq!(status, 200);
        assert_eq!(
            result["result"],
            serde_json::json!({"status": "completed", "affected": 2})
        );
        assert_eq!(retrieve(&ctx), expected);
    }
}

#[test]
fn file_size_repair_fails_closed_when_lsm_size_is_indexed() {
    let ctx = setup_with_config(Config::default().with_lsm_in_memory());
    create_dense_collection_lsm(&ctx, "repo");
    upsert_points(
        &ctx,
        "repo",
        serde_json::json!([{"id": 11, "vector": {"dense": [0.1, 0.2, 0.3]}, "payload": {"metadata": {"file_hash": "H1", "file_size_bytes": 1, "repo_id": "git:repo", "path": "src/lib.rs"}}}]),
    );
    let input = make_input(&ctx, "PUT", "/collections/repo/index", serde_json::to_vec(&serde_json::json!({"field_name": "metadata.file_size_bytes", "field_schema": "integer"})).unwrap(), HashMap::from([("name".into(), "repo".into())]));
    let mut response = Response::new();
    handle_create_index(&input, &mut response).unwrap();
    assert_eq!(response.status, 200);
    let before = retrieve(&ctx);
    assert_eq!(repair(&ctx, serde_json::json!([11])).0, 400);
    assert_eq!(
        retrieve(&ctx),
        before,
        "unsupported indexed-size repairs must not stale the index"
    );
}
