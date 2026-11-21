use super::filter_fixture::{fixture, nested_exclusion};
use super::tests::make_input;
use super::*;

#[test]
fn nested_exclusion_is_rechecked_before_exact_scoring() {
    let (_ctx, storage) = fixture(400);
    let filter = nested_exclusion();
    let read = storage.backend.begin_read().unwrap();
    let candidates = dense_filter_candidates_be(&storage, &read, &filter, false, 100_000, false)
        .unwrap()
        .candidates
        .unwrap();

    let ids = exact_dense_candidate_ids_be(
        &storage,
        &read,
        Some(&candidates),
        &filter,
        !indexed_filter_candidates_are_exact(&storage, &filter),
        400,
    )
    .unwrap();

    assert_eq!(ids.into_iter().collect::<HashSet<_>>(), (2..=400).collect());
}

#[test]
fn nested_exclusion_is_rechecked_on_large_candidate_hnsw() {
    let (_ctx, storage) = fixture(2100);
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
    let filter = nested_exclusion();
    let read = storage.backend.begin_read().unwrap();
    assert!(storage
        .named_vectors
        .with_core("dense", |core| core.has_index(&read))
        .unwrap());
    let control = storage
        .named_vectors
        .dense_search_with_id_filter_ef_be::<fn(u128) -> bool>(
            &read,
            "dense",
            &[1.0, 0.0, 0.0],
            5,
            None,
            true,
            None,
            None,
        )
        .unwrap();
    assert_eq!(
        control[0].id, 1,
        "built HNSW must find the nearest private point"
    );
    let candidates = dense_filter_candidates_be(&storage, &read, &filter, false, 100_000, false)
        .unwrap()
        .candidates
        .unwrap();
    assert!(exact_dense_candidate_ids_be(
        &storage,
        &read,
        Some(&candidates),
        &filter,
        !indexed_filter_candidates_are_exact(&storage, &filter),
        2100
    )
    .is_none());
    let index_exact = exact_index_filter_candidates(
        true,
        !indexed_filter_candidates_are_exact(&storage, &filter),
        Some(&candidates),
    );
    let predicate = |id| {
        index_exact.map_or_else(
            || {
                storage
                    .get_node_be(&read, &id)
                    .is_ok_and(|node| filter.matches_point(Some(id), &node.properties))
            },
            |ids| ids.contains(&id),
        )
    };

    let hits = match index_exact {
        Some(ids) => storage
            .named_vectors
            .dense_search_candidate_set_filter_ef_be(
                &read,
                "dense",
                &[1.0, 0.0, 0.0],
                5,
                ids,
                true,
                None,
                None,
            ),
        None => storage.named_vectors.dense_search_with_id_filter_ef_be(
            &read,
            "dense",
            &[1.0, 0.0, 0.0],
            5,
            Some(&[predicate][..]),
            true,
            None,
            None,
        ),
    }
    .unwrap();

    assert_eq!(hits.len(), 5);
    assert!(
        hits.iter().all(|hit| hit.id != 1),
        "nearest private point leaked"
    );
}

fn assert_endpoint_excludes_private(endpoint: &str) {
    let (ctx, _storage) = fixture(400);
    let filter = serde_json::to_value(nested_exclusion()).unwrap();
    let request = match endpoint {
        "search" => serde_json::json!({"vector": {"name": "dense", "vector": [1.0, 0.0, 0.0]},
            "limit": 5, "with_payload": true, "filter": filter}),
        "query" => serde_json::json!({"query": [1.0, 0.0, 0.0], "using": "dense",
            "limit": 5, "with_payload": true, "filter": filter}),
        "query/hybrid" => serde_json::json!({"dense": [{"query": [1.0, 0.0, 0.0],
            "using": "dense", "limit": 5, "filter": filter}], "sparse": [],
            "limit": 5, "with_payload": true}),
        _ => unreachable!(),
    };
    let input = make_input(
        &ctx,
        "POST",
        &format!("/collections/filter_closeout/points/{endpoint}"),
        serde_json::to_vec(&request).unwrap(),
        HashMap::from([("name".into(), "filter_closeout".into())]),
    );
    let mut response = Response::new();

    match endpoint {
        "search" => handle_search_points(&input, &mut response),
        "query" => handle_query_points(&input, &mut response),
        "query/hybrid" => handle_hybrid_query_points(&input, &mut response),
        _ => unreachable!(),
    }
    .unwrap();

    assert_eq!(
        response.status,
        200,
        "{}",
        String::from_utf8_lossy(&response.body)
    );
    let body: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
    let points = if endpoint == "search" {
        &body["result"]
    } else {
        &body["result"]["points"]
    };
    assert_eq!(points.as_array().unwrap().len(), 5);
    assert!(
        points
            .as_array()
            .unwrap()
            .iter()
            .all(|point| point["payload"]["branch"] == "public"),
        "{body}"
    );
}

#[test]
fn search_honors_nested_exclusion() {
    assert_endpoint_excludes_private("search");
}

#[test]
fn query_honors_nested_exclusion() {
    assert_endpoint_excludes_private("query");
}

#[test]
fn hybrid_honors_nested_exclusion() {
    assert_endpoint_excludes_private("query/hybrid");
}

#[test]
fn mixed_must_should_matches_full_predicate() {
    let (_ctx, storage) = fixture(12);
    let filter: Filter = serde_json::from_value(serde_json::json!({
        "must": [{"key": "scope", "match": {"value": "all"}}],
        "should": [{"key": "branch", "match": {"value": "public"}}]
    }))
    .unwrap();
    let read = storage.backend.begin_read().unwrap();
    let expected = (1..=12)
        .filter(|id| {
            filter.matches_point(
                Some(*id),
                &storage.get_node_be(&read, id).unwrap().properties,
            )
        })
        .collect::<HashSet<_>>();

    let candidates = indexed_filter_candidates_be(&storage, &read, &filter)
        .unwrap()
        .unwrap();

    assert_eq!(expected, (2..=12).collect());
    assert_eq!(candidates, expected);
    assert!(indexed_filter_candidates_are_exact(&storage, &filter));
}

#[test]
fn query_empty_gate_rejects_unknown_named_vector() {
    assert_query_rejects_unknown_vector(serde_json::json!([]));
}

#[test]
fn query_absent_gate_rejects_unknown_named_vector() {
    assert_query_rejects_unknown_vector(serde_json::json!([999_999]));
}

fn assert_query_rejects_unknown_vector(gate: serde_json::Value) {
    let (ctx, _storage) = fixture(1);
    let input = make_input(
        &ctx,
        "POST",
        "/collections/filter_closeout/points/query",
        serde_json::to_vec(
            &serde_json::json!({"query": [1.0, 0.0, 0.0], "using": "missing_vector",
            "limit": 1, "filter": {"must": [{"has_id": gate}]}}),
        )
        .unwrap(),
        HashMap::from([("name".into(), "filter_closeout".into())]),
    );
    let mut response = Response::new();

    let result = handle_query_points(&input, &mut response);

    assert!(
        result.is_err(),
        "unknown vector accepted: {}",
        String::from_utf8_lossy(&response.body)
    );
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Named vector 'missing_vector' not found"));
}

/// JSON `null` payloads are stored as the string `"null"`, which
/// `matches_point` never equals to a `null` operand. The keyword lookup must
/// follow that: a `must_not` on `null` keeps every point and a `must` on
/// `null` matches none, on the indexed path exactly as on a scan.
#[test]
fn null_match_value_follows_matches_point_on_the_index_path() {
    let (ctx, storage) = fixture(1);
    let branches = [
        serde_json::json!("public"),
        serde_json::Value::Null,
        serde_json::json!("null"),
        serde_json::json!(["null"]),
        serde_json::json!("private"),
    ];
    let points = (1..=40u32)
        .map(|id| {
            let mut payload = serde_json::json!({"scope": "all"});
            if id % 6 != 5 {
                payload["branch"] = branches[(id % 6) as usize % branches.len()].clone();
            }
            let angle = f32::from(u16::try_from(id).unwrap()) * 0.1;
            serde_json::json!({"id": id, "vector": {"dense": [angle.cos(), angle.sin(), 0.0]},
                "payload": payload})
        })
        .collect::<Vec<_>>();
    super::tests::upsert_points(&ctx, "filter_closeout", serde_json::Value::Array(points));
    let scope = serde_json::json!({"key": "scope", "match": {"value": "all"}});
    let null_branch = serde_json::json!({"key": "branch", "match": {"value": null}});
    let any_null = serde_json::json!({"key": "branch", "match": {"any": [null, "private"]}});
    let read = storage.backend.begin_read().unwrap();
    let nodes: Vec<_> = (1..=40u128)
        .map(|id| (id, storage.get_node_be(&read, &id).unwrap().properties))
        .collect();
    for value in [
        serde_json::json!({"must": [scope], "must_not": [null_branch]}),
        serde_json::json!({"must": [scope], "must_not": [any_null]}),
        serde_json::json!({"must": [scope, null_branch]}),
        serde_json::json!({"must": [null_branch]}),
        serde_json::json!({"must": [any_null]}),
    ] {
        let filter: Filter = serde_json::from_value(value.clone()).unwrap();
        let expected: HashSet<u128> = nodes
            .iter()
            .filter(|(id, properties)| filter.matches_point(Some(*id), properties))
            .map(|(id, _)| *id)
            .collect();
        let input = make_input(
            &ctx,
            "POST",
            "/collections/filter_closeout/points/query",
            serde_json::to_vec(&serde_json::json!({"query": [1.0, 0.0, 0.0],
                "using": "dense", "limit": 100, "filter": value}))
            .unwrap(),
            HashMap::from([("name".into(), "filter_closeout".into())]),
        );
        let mut response = Response::new();
        handle_query_points(&input, &mut response).unwrap();
        assert_eq!(response.status, 200);
        let body: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        let returned: HashSet<u128> = body["result"]["points"]
            .as_array()
            .unwrap()
            .iter()
            .map(|point| u128::from_str_radix(point["id"].as_str().unwrap(), 16).unwrap())
            .collect();
        assert_eq!(returned, expected, "{value}");
    }
}

fn late_index_collection() -> (super::tests::TestContext, std::sync::Arc<HelixGraphStorage>) {
    let mut config =
        crate::helix_engine::graph_core::config::Config::default().with_lsm_in_memory();
    config.vector_config.flat_scan_threshold = Some(0);
    let ctx = super::tests::setup_with_config(config);
    super::tests::create_dense_collection_lsm(&ctx, "late_index");
    let storage = ctx.collections.get_collection("late_index").unwrap();
    let points = (1..=30u32)
        .map(|id| {
            let angle = f32::from(u16::try_from(id).unwrap()) * 0.1;
            serde_json::json!({"id": id, "vector": {"dense": [angle.cos(), angle.sin(), 0.0]},
                "payload": {"scope": "all", "branch": if id <= 3 { "private" } else { "public" }}})
        })
        .collect::<Vec<_>>();
    super::tests::upsert_points(&ctx, "late_index", serde_json::Value::Array(points));
    (ctx, storage)
}

/// A request reads one snapshot. An index whose build published Ready after
/// that snapshot was opened lacks rows there, so the planner must treat the
/// field as unindexed: neither an empty "exact" set nor a must_not
/// subtraction of rows the snapshot does not hold.
#[test]
fn planner_ignores_an_index_published_after_its_snapshot() {
    let (_ctx, storage) = late_index_collection();
    let scope_only: Filter = serde_json::from_value(
        serde_json::json!({"must": [{"key": "scope", "match": {"value": "all"}}]}),
    )
    .unwrap();
    let exclusion: Filter = serde_json::from_value(serde_json::json!({
        "must": [{"key": "scope", "match": {"value": "all"}}],
        "must_not": [{"key": "branch", "match": {"value": "private"}}]}))
    .unwrap();

    let before_scope = storage.backend.begin_read().unwrap();
    storage
        .create_payload_index("scope", PayloadIndexSchema::Keyword)
        .unwrap();
    let before_branch = storage.backend.begin_read().unwrap();
    storage
        .create_payload_index("branch", PayloadIndexSchema::Keyword)
        .unwrap();
    let fresh = storage.backend.begin_read().unwrap();

    let plan = |read, filter| {
        dense_filter_candidates_be(&storage, read, filter, true, 100_000, true).unwrap()
    };
    let stale = plan(&before_scope, &scope_only);
    assert!(
        stale.candidates.is_none() && !stale.exact,
        "used an index newer than the snapshot"
    );

    // scope is complete in this snapshot, branch is not: no subtraction, and
    // the set is a superset that callers must recheck.
    let partial = plan(&before_branch, &exclusion);
    assert_eq!(partial.candidates, Some((1..=30).collect()));
    assert!(!partial.exact);

    let current = plan(&fresh, &exclusion);
    assert_eq!(current.candidates, Some((4..=30).collect()));
    assert!(current.exact);
}

/// Dropping an index leaves its rows, and nothing maintains them while no
/// handle exists; a rebuild (sync or the async job) must not publish them.
#[test]
fn rebuilt_index_drops_rows_left_by_the_dropped_index() {
    for async_build in [false, true] {
        let (ctx, storage) = late_index_collection();
        storage
            .create_payload_index("branch", PayloadIndexSchema::Keyword)
            .unwrap();
        storage.delete_payload_index("branch").unwrap();
        let moved = (1..=30u32)
            .map(|id| {
                serde_json::json!({"id": id, "vector": {"dense": [1.0, 0.0, 0.0]},
                    "payload": {"scope": "all", "branch": "moved"}})
            })
            .collect::<Vec<_>>();
        super::tests::upsert_points(&ctx, "late_index", serde_json::Value::Array(moved));
        if async_build {
            storage
                .enqueue_payload_index_build(
                    "late_index",
                    "branch",
                    PayloadIndexSchema::Keyword,
                    false,
                )
                .unwrap();
            let deadline = Instant::now() + Duration::from_secs(10);
            while storage.has_payload_index("branch").is_none() {
                assert!(Instant::now() < deadline, "async build did not finish");
                std::thread::sleep(Duration::from_millis(5));
            }
        } else {
            storage
                .create_payload_index("branch", PayloadIndexSchema::Keyword)
                .unwrap();
        }
        let read = storage.backend.begin_read().unwrap();
        let stale = storage
            .get_nodes_by_payload_value_be(&read, "branch", &Value::String("public".into()))
            .unwrap();
        assert!(
            stale.is_empty(),
            "async={async_build}: stale rows {stale:?}"
        );
        let current = storage
            .get_nodes_by_payload_value_be(&read, "branch", &Value::String("moved".into()))
            .unwrap();
        assert_eq!(current.len(), 30, "async={async_build}");
    }
}

/// An integer above i64::MAX is stored as its f64; the lookup must use that
/// key, since `matches_point` compares the two as equal.
#[test]
fn match_value_above_i64_max_finds_the_stored_float() {
    let (ctx, storage) = fixture(1);
    storage
        .create_payload_index("n", PayloadIndexSchema::Keyword)
        .unwrap();
    let big: serde_json::Value = serde_json::from_str("9223372036854775809").unwrap();
    super::tests::upsert_points(
        &ctx,
        "filter_closeout",
        serde_json::json!([
            {"id": 2, "vector": {"dense": [1.0, 0.0, 0.0]}, "payload": {"scope": "all", "n": big}},
            {"id": 3, "vector": {"dense": [0.0, 1.0, 0.0]}, "payload": {"scope": "all", "n": 5}}
        ]),
    );
    let read = storage.backend.begin_read().unwrap();
    for (value, expected) in [
        (
            serde_json::json!({"must": [{"key": "n", "match": {"value": big}}]}),
            vec![2u128],
        ),
        (
            serde_json::json!({"must": [{"key": "scope", "match": {"value": "all"}}],
                "must_not": [{"key": "n", "match": {"value": big}}]}),
            vec![1, 3],
        ),
    ] {
        let filter: Filter = serde_json::from_value(value.clone()).unwrap();
        let oracle: Vec<u128> = (1..=3u128)
            .filter(|id| {
                filter.matches_point(
                    Some(*id),
                    &storage.get_node_be(&read, id).unwrap().properties,
                )
            })
            .collect();
        assert_eq!(oracle, expected, "{value}");
        let (ids, exact) = indexed_filter_ids_be(&storage, &read, &filter, 0)
            .unwrap()
            .unwrap();
        assert!(exact, "{value}");
        let mut ids: Vec<u128> = ids.into_iter().collect();
        ids.sort_unstable();
        assert_eq!(ids, expected, "{value}");
    }
}
