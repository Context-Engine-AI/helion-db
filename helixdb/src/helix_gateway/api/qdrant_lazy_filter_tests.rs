use super::tests::{create_dense_collection_lsm, make_input, setup_with_config, upsert_points};
use super::*;
use crate::helix_engine::graph_core::config::Config;
use crate::helix_engine::storage_core::backend_lsm::lsm_round_trips_for_test;
use crate::helix_engine::vector_core::vector_core::with_lazy_filter_for_test;

const REPO: &str = "example-go-gitea__gitea-4875bb3b-7dc453f25a60e8e1";

/// The client's single-repo shape: every point carries the same `metadata.repo`, so
/// the `match.text` repo clause is broad and filtered HNSW rechecks it with a
/// node read per evaluated neighbor.
fn single_repo_hnsw(count: u32) -> super::tests::TestContext {
    let mut config = Config::default().with_lsm_in_memory();
    config.vector_config.flat_scan_threshold = Some(0);
    let ctx = setup_with_config(config);
    create_dense_collection_lsm(&ctx, "single_repo");
    let storage = ctx.collections.get_collection("single_repo").unwrap();
    storage
        .create_payload_index("metadata.repo", PayloadIndexSchema::Keyword)
        .unwrap();
    let points = (1..=count)
        .map(|id| {
            let angle = f32::from(u16::try_from(id).unwrap()) * 0.37;
            let lift = f32::from(u16::try_from(id % 17).unwrap()) / 17.0;
            serde_json::json!({
                "id": id,
                "vector": {"dense": [angle.cos(), angle.sin(), lift]},
                "payload": {"metadata": {"repo": REPO, "path": format!("/work/{REPO}/f{id}.go")}}
            })
        })
        .collect::<Vec<_>>();
    upsert_points(&ctx, "single_repo", serde_json::Value::Array(points));
    storage
        .named_vectors
        .with_core("dense", |core| {
            let read = storage.backend.begin_read().unwrap();
            let mut prepared = core.prepare_index_from_flat(&read)?.unwrap();
            drop(read);
            let mut write = storage.backend.begin_write().unwrap();
            let total = prepared.point_ids.len();
            core.flush_prepared_index_chunk_be(&mut write, &mut prepared, 0, total)?;
            core.finalize_prepared_index_entry_be(&mut write, &prepared)?;
            storage.backend.commit(write).unwrap();
            Ok(())
        })
        .unwrap();
    ctx
}

/// `(id, score)` hits and the SlateDB round trips the request issued.
fn query(
    ctx: &super::tests::TestContext,
    vector: [f32; 3],
    filtered: bool,
) -> (Vec<(String, String)>, usize) {
    let mut request = serde_json::json!({"query": vector, "using": "dense", "limit": 24});
    if filtered {
        request["filter"] = serde_json::json!({"must": [
            {"key": "metadata.repo", "match": {"text": REPO}}
        ]});
    }
    let input = make_input(
        ctx,
        "POST",
        "/collections/single_repo/points/query",
        serde_json::to_vec(&request).unwrap(),
        HashMap::from([("name".into(), "single_repo".into())]),
    );
    let mut response = Response::new();
    let before = lsm_round_trips_for_test();
    handle_query_points(&input, &mut response).unwrap();
    let trips = lsm_round_trips_for_test() - before;
    assert_eq!(response.status, 200);
    let body: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
    let hits = body["result"]["points"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| (p["id"].to_string(), p["score"].to_string()))
        .collect();
    (hits, trips)
}

#[test]
fn broad_repo_text_filter_keeps_results_and_skips_node_reads() {
    let ctx = single_repo_hnsw(2_000);
    let storage = ctx.collections.get_collection("single_repo").unwrap();
    let read = storage.backend.begin_read().unwrap();
    assert!(storage
        .named_vectors
        .with_core("dense", |core| core.has_index(&read))
        .unwrap());
    drop(read);

    let (mut eager_trips, mut lazy_trips, mut unfiltered_trips) = (0, 0, 0);
    for vector in [
        [1.0, 0.0, 0.2],
        [0.0, 1.0, 0.9],
        [-0.7, 0.7, 0.5],
        [0.3, -0.9, 0.1],
    ] {
        let (eager, eager_n) = with_lazy_filter_for_test(false, || query(&ctx, vector, true));
        let (lazy, lazy_n) = with_lazy_filter_for_test(true, || query(&ctx, vector, true));
        assert_eq!(eager.len(), 24);
        assert_eq!(lazy, eager, "lazy filter changed /points/query results");
        eager_trips += eager_n;
        lazy_trips += lazy_n;
        unfiltered_trips += query(&ctx, vector, false).1;
    }
    println!(
        "single-repo match.text: LSM round trips eager={eager_trips} lazy={lazy_trips} \
         unfiltered={unfiltered_trips}"
    );
    assert!(
        lazy_trips < eager_trips,
        "lazy {lazy_trips} vs eager {eager_trips}"
    );
}
