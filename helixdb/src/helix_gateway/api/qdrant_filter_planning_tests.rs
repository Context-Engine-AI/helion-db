use super::filter_fixture::fixture;
use super::*;
use crate::helix_engine::storage_core::backend_lsm::dup_scan_rows_for_test;
use crate::helix_engine::storage_core::metadata::MetadataCounter;

#[test]
fn posting_visit_counter_observes_real_rows() {
    let (_ctx, storage) = fixture(12);
    let read = storage.backend.begin_read().unwrap();
    let before = dup_scan_rows_for_test();

    let ids = storage
        .get_nodes_by_payload_value_be(&read, "scope", &Value::String("all".into()))
        .unwrap();

    assert_eq!(ids.len(), 12);
    assert_eq!(dup_scan_rows_for_test() - before, 12);
}

#[test]
fn broad_scope_skips_uncounted_nested_postings() {
    let (_ctx, storage) = fixture(400);
    let filter: Filter = serde_json::from_value(serde_json::json!({"must": [
        {"key": "scope", "match": {"value": "all"}},
        {"should": [{"key": "language", "match": {"value": "python"}},
                    {"key": "language", "match": {"value": "rust"}}]}
    ]}))
    .unwrap();
    let read = storage.backend.begin_read().unwrap();
    let before = dup_scan_rows_for_test();

    let result = dense_filter_candidates_be(&storage, &read, &filter, true, 0, true).unwrap();

    let visited = dup_scan_rows_for_test() - before;
    assert!(visited > 0, "count probe must visit postings");
    assert!(
        visited <= vector_query_broad_candidate_cap(400) + 1,
        "uncounted nested branches read {visited} posting rows"
    );
    assert!(result.probe_skipped);
    assert!(
        result.candidates.is_none(),
        "over-budget is Unknown, not an exact truncated set"
    );
}

#[test]
fn large_has_id_gate_uses_more_selective_indexed_sibling() {
    let (_ctx, storage) = fixture(400);
    // Planner cardinality fixture: absent gate IDs are legal, no need to score 100k vectors.
    storage
        .with_write_backend(|write| {
            storage.adjust_metadata_counter_be(write, MetadataCounter::Vectors, 99_600)
        })
        .unwrap();
    assert_eq!(collection_vector_count_be(&storage), 100_000);
    let filter: Filter = serde_json::from_value(serde_json::json!({"must": [
        {"has_id": (1..=10_000).collect::<Vec<_>>()},
        {"key": "branch", "match": {"value": "private"}}
    ]}))
    .unwrap();
    let read = storage.backend.begin_read().unwrap();

    let result =
        vector_query_indexed_filter_candidates_be_probed(&storage, &read, &filter, true).unwrap();

    let candidates = result.candidates.unwrap();
    assert_eq!(
        candidates.len(),
        1,
        "selective sibling should choose one exact candidate"
    );
    assert_eq!(candidates, HashSet::from([1]));
    assert!(should_exact_score_index_candidates(
        &candidates,
        collection_vector_count_be(&storage)
    ));
}

#[test]
fn selective_counted_nested_filter_keeps_exact_candidates() {
    let (_ctx, storage) = fixture(400);
    let filter: Filter = serde_json::from_value(serde_json::json!({"must": [
        {"key": "scope", "match": {"value": "all"}},
        {"must": [{"key": "branch", "match": {"value": "private"}}]}
    ]}))
    .unwrap();
    let read = storage.backend.begin_read().unwrap();

    let result = dense_filter_candidates_be(&storage, &read, &filter, true, 0, true).unwrap();

    assert_eq!(result.candidates, Some(HashSet::from([1])));
    assert!(!result.probe_skipped);
    assert!(indexed_filter_candidates_are_exact(&storage, &filter));
}
