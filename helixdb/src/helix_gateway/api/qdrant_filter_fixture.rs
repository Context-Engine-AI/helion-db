use super::tests::{create_dense_collection_lsm, setup_with_config, upsert_points, TestContext};
use super::*;
use crate::helix_engine::graph_core::config::Config;
use std::sync::Arc;

pub(super) fn fixture(count: u32) -> (TestContext, Arc<HelixGraphStorage>) {
    let mut config = Config::default().with_lsm_in_memory();
    config.vector_config.flat_scan_threshold = Some(0);
    let ctx = setup_with_config(config);
    create_dense_collection_lsm(&ctx, "filter_closeout");
    let storage = ctx.collections.get_collection("filter_closeout").unwrap();
    for field in ["scope", "branch", "language"] {
        storage
            .create_payload_index(field, PayloadIndexSchema::Keyword)
            .unwrap();
    }
    let points = (1..=count)
        .map(|id| {
            let angle = 0.2
                + f32::from(u16::try_from(id).unwrap()) / f32::from(u16::try_from(count).unwrap());
            serde_json::json!({
                "id": id,
                "vector": {"dense": [angle.cos(), angle.sin(), 0.0]},
                "payload": {
                    "scope": "all",
                    "branch": if id == 1 { "private" } else { "public" },
                    "language": if id % 2 == 0 { "rust" } else { "python" }
                }
            })
        })
        .collect::<Vec<_>>();
    upsert_points(&ctx, "filter_closeout", serde_json::Value::Array(points));
    (ctx, storage)
}

pub(super) fn nested_exclusion() -> Filter {
    serde_json::from_value(serde_json::json!({"must": [
        {"key": "scope", "match": {"value": "all"}},
        {"must_not": [{"key": "branch", "match": {"value": "private"}}]}
    ]}))
    .unwrap()
}
