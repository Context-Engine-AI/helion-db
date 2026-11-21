use super::tests::{create_dense_collection_lsm, make_input, setup_with_config, upsert_points};
use super::text_prefilter_tests::{dir_fixture_with, filter_shapes, path_text, requests, run, N};
use super::*;
use crate::helix_engine::graph_core::config::Config;
use rand::{Rng, SeedableRng};

/// `match.text`'s test as the index walk applies it to a stored value:
/// strings by lowercase substring, arrays element by element.
fn walk_matches(value: &Value, needle: &str) -> bool {
    match value {
        Value::String(s) => s.to_lowercase().contains(&needle.to_lowercase()),
        Value::Array(values) => values.iter().any(|value| walk_matches(value, needle)),
        _ => false,
    }
}

/// Random payload shapes, case and unicode, plus rows written the way older
/// binaries wrote them (no plain-row marker): a walk reports exact exactly
/// when no matching row came from an array or an unmarked row, and an exact
/// walk equals the `matches_point` set. Again after the online backfill
/// marked the legacy rows.
#[test]
fn exact_text_walk_equals_match_text() {
    let mut config = Config::default().with_lsm_in_memory();
    config.vector_config.flat_scan_threshold = Some(0);
    let ctx = setup_with_config(config);
    create_dense_collection_lsm(&ctx, "shapes");
    let storage = ctx.collections.get_collection("shapes").unwrap();
    storage
        .create_payload_index("metadata.path", PayloadIndexSchema::Keyword)
        .unwrap();
    let alphabet: Vec<char> = "abAB/_.-xyzßİÄΣσ7Kk".chars().collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(29);
    let word = |rng: &mut rand::rngs::StdRng| -> String {
        let len = rng.random_range(0..9);
        (0..len)
            .map(|_| alphabet[rng.random_range(0..alphabet.len())])
            .collect()
    };
    let mut strings = Vec::new();
    let points = (1..=500u32)
        .map(|id| {
            let s = word(&mut rng);
            strings.push(s.clone());
            // Mostly plain strings, so many needles see no array hit.
            let metadata = match id % 23 {
                0 => serde_json::json!({"path": [s, word(&mut rng)]}),
                1 => serde_json::json!({"path": [s]}),
                2 => serde_json::json!({"path": [[s]]}),
                3 => serde_json::json!({"path": id}),
                4 => serde_json::json!({"path": id % 3 == 0}),
                5 => serde_json::json!({"path": null}),
                6 => serde_json::json!({"path": {"inner": s}}),
                7 => serde_json::json!({"other": s}),
                8 => serde_json::json!({"path": "null"}),
                9 => serde_json::json!(s),
                _ => serde_json::json!({"path": s}),
            };
            serde_json::json!({"id": id, "vector": {"dense": [1.0, 0.0, 0.0]},
                               "payload": {"metadata": metadata}})
        })
        .collect::<Vec<_>>();
    upsert_points(&ctx, "shapes", serde_json::Value::Array(points));

    let read = storage.backend.begin_read().unwrap();
    let nodes: Vec<_> = (1..=500u128)
        .map(|id| (id, storage.get_node_be(&read, &id).unwrap().properties))
        .collect();
    drop(read);
    // Rewrite some plain-string rows as an older binary wrote them.
    let db_name =
        HelixGraphStorage::payload_index_db_name("metadata.path", &PayloadIndexSchema::Keyword);
    let mut legacy = HashSet::new();
    let mut write = storage.backend.begin_write().unwrap();
    for (id, properties) in &nodes {
        let value = HelixGraphStorage::payload_value_for_key(properties, "metadata.path");
        if let (Some(value @ Value::String(_)), true) = (value, id % 41 == 0) {
            let key = HelixGraphStorage::stable_index_key_for_value(value).unwrap();
            storage
                .backend
                .put_dup(
                    &mut write,
                    Namespace::PayloadIndex(&db_name),
                    &key,
                    &id.to_be_bytes(),
                )
                .unwrap();
            legacy.insert(*id);
        }
    }
    storage.backend.commit(write).unwrap();
    storage.forget_payload_marker_backfill_for_test("metadata.path");
    assert!(!legacy.is_empty());

    let mut needles: Vec<String> = vec![
        "".into(),
        "zz-none".into(),
        "İ".into(),
        "i̇".into(),
        "SS".into(),
        "nul".into(),
        "K".into(),
    ];
    for s in strings.iter().filter(|s| !s.is_empty()).take(200) {
        let chars: Vec<char> = s.chars().collect();
        let start = rng.random_range(0..chars.len());
        let end = rng.random_range(start + 1..=chars.len());
        let piece: String = chars[start..end].iter().collect();
        needles.push(if rng.random_bool(0.5) {
            piece.to_uppercase()
        } else {
            piece
        });
    }
    let check =
        |legacy: &HashSet<u128>| {
            let read = storage.backend.begin_read().unwrap();
            let mut exact_walks = 0;
            for needle in &needles {
                let filter: Filter =
                    serde_json::from_value(serde_json::json!({"must": [path_text(needle)]}))
                        .unwrap();
                let (walked, exact) = storage
                    .get_nodes_by_payload_text_exact_be(&read, "metadata.path", needle, 0)
                    .unwrap()
                    .expect("short terms decode");
                let walked: HashSet<u128> = walked.into_iter().collect();
                let matches: HashSet<u128> = nodes
                    .iter()
                    .filter(|(id, properties)| filter.matches_point(Some(*id), properties))
                    .map(|(id, _)| *id)
                    .collect();
                let unproven = nodes.iter().any(|(id, properties)| {
                    match HelixGraphStorage::payload_value_for_key(properties, "metadata.path") {
                        Some(value @ Value::Array(_)) => walk_matches(value, needle),
                        Some(value) => legacy.contains(id) && walk_matches(value, needle),
                        None => false,
                    }
                });
                assert!(
                    walked.is_superset(&matches),
                    "{needle:?}: walk lost a match"
                );
                assert_eq!(exact, !unproven, "{needle:?}: exactness");
                if exact {
                    exact_walks += 1;
                    assert_eq!(walked, matches, "{needle:?}: exact walk differs");
                }
                let planned = |enabled| {
                    with_text_walk_exact_for_test(enabled, || {
                        indexed_filter_ids_be(&storage, &read, &filter, 0)
                            .unwrap()
                            .unwrap()
                    })
                };
                assert_eq!(planned(true), (walked.clone(), exact), "{needle:?}");
                assert_eq!(planned(false), (walked, false), "{needle:?}: kill switch");
            }
            exact_walks
        };
    let exact_walks = check(&legacy);
    assert!(
        exact_walks > 20 && exact_walks < needles.len(),
        "{exact_walks} of {} walks exact",
        needles.len()
    );
    // The online backfill marks the legacy rows: only array hits remain.
    let backfill =
        HelixGraphStorage::run_payload_marker_backfill(&std::sync::Arc::downgrade(&storage))
            .unwrap();
    assert!(
        backfill.completed && backfill.marked == legacy.len(),
        "{backfill:?}"
    );
    let backfilled = check(&HashSet::new());
    assert!(
        backfilled > exact_walks && backfilled < needles.len(),
        "{backfilled} of {} walks exact after the backfill",
        needles.len()
    );
}

fn count(ctx: &super::tests::TestContext, filter: &serde_json::Value) -> serde_json::Value {
    let input = make_input(
        ctx,
        "POST",
        "/collections/dirs/points/count",
        serde_json::to_vec(&serde_json::json!({"filter": filter, "exact": true})).unwrap(),
        HashMap::from([("name".into(), "dirs".into())]),
    );
    let mut response = Response::new();
    handle_count_points(&input, &mut response).unwrap();
    assert_eq!(response.status, 200);
    serde_json::from_slice::<serde_json::Value>(&response.body).unwrap()["result"].clone()
}

/// Every endpoint returns the same ids and score bits with exact text walks
/// on and off; only the per-point rechecks of exact members disappear. Plain
/// paths, plus a second (mutable) segment so candidate-set HNSW runs its
/// parallel arm against the serial filtered arm the recheck path uses. The
/// first points' markers come from the online backfill. The broad-filter scan
/// stays off: it serves only exact candidate sets, so with it on the walk-on
/// arm is brute-force scored while the walk-off arm stays on approximate HNSW
/// (the scan's own recall is checked against brute force elsewhere).
#[test]
fn exact_text_matches_kill_switch_and_skips_member_rechecks() {
    let (ctx, storage) = dir_fixture_with(false);
    // Markers from the online backfill rather than the write path.
    for field in ["metadata.path", "metadata.repo"] {
        storage.strip_payload_markers_for_test(field);
    }
    let backfill =
        HelixGraphStorage::run_payload_marker_backfill(&std::sync::Arc::downgrade(&storage))
            .unwrap();
    assert!(
        backfill.completed && backfill.marked == 2 * usize::try_from(N).unwrap(),
        "{backfill:?}"
    );
    // Seal the indexed tail (kept searchable while it is `Building`) so the
    // extra points land in a second, mutable segment.
    let tail = storage
        .named_vectors
        .mutable_dense_segment_name("dense")
        .unwrap();
    storage
        .with_write_backend(|w| {
            storage
                .named_vectors
                .mark_sealed_tail_be(w, &tail)
                .map_err(GraphError::from)
        })
        .unwrap();
    let read = storage.backend.begin_read().unwrap();
    let sealed = storage
        .named_vectors
        .seal_mutable_tail_lsm(
            &read,
            storage.collection_path(),
            "dense",
            crate::helix_engine::vector_core::vector_core::HNSWConfig::new(
                Some(8),
                Some(32),
                Some(64),
            ),
        )
        .unwrap();
    drop(read);
    assert_eq!(sealed.as_deref(), Some(tail.as_str()));
    storage
        .with_write_backend(|w| {
            storage.set_dense_vector_spaces_metadata_be(
                w,
                storage.named_vectors.list_dense_vector_spaces(),
            )
        })
        .unwrap();
    let extra = (N + 1..=N + 300)
        .map(|id| {
            let angle = f32::from(u16::try_from(id).unwrap()) * 0.29;
            serde_json::json!({"id": id, "vector": {
                    "dense": [angle.sin(), angle.cos(), 0.5],
                    "lex_sparse": {"indices": [id % 13, 100 + id % 7], "values": [0.5, 1.0]}},
                "payload": {"metadata": {
                    "repo": super::text_prefilter_tests::REPO,
                    "path": format!("/work/{}/c40_routers/x60/G{id}.go",
                        super::text_prefilter_tests::REPO),
                    "language": "go"},
                    "branch": "public"}})
        })
        .collect::<Vec<_>>();
    upsert_points(&ctx, "dirs", serde_json::Value::Array(extra));
    let segments = storage.named_vectors.list_dense_vector_spaces()["dense"]
        .segments
        .len();
    assert_eq!(segments, 2, "dense segments");
    let mut rng = rand::rngs::StdRng::seed_from_u64(17);
    let mut vector = || -> [f32; 3] { std::array::from_fn(|_| rng.random_range(-1.0f32..1.0)) };
    let queries: Vec<([f32; 3], [f32; 3])> = (0..3).map(|_| (vector(), vector())).collect();
    let mut saved = 0usize;
    for (label, filter) in filter_shapes() {
        // [node reads, microseconds] for /points/query, then all endpoints.
        let mut old = [[0usize; 2]; 2];
        let mut new = [[0usize; 2]; 2];
        for (v, w) in &queries {
            for (index, (endpoint, body)) in requests(&filter, *v, *w).into_iter().enumerate() {
                let measure = |enabled: bool, acc: &mut [[usize; 2]; 2]| {
                    let started = Instant::now();
                    let (result, cost) = with_broad_scan_for_test(false, || {
                        with_text_walk_exact_for_test(enabled, || run(&ctx, endpoint, &body))
                    });
                    let micros = usize::try_from(started.elapsed().as_micros()).unwrap();
                    for (slot, add) in [(0, cost[0]), (1, micros)] {
                        acc[1][slot] += add;
                        if index == 0 {
                            acc[0][slot] += add;
                        }
                    }
                    (result, cost)
                };
                let (old_result, old_cost) = measure(false, &mut old);
                let (new_result, new_cost) = measure(true, &mut new);
                assert_eq!(
                    new_result, old_result,
                    "{label} {endpoint}: results changed"
                );
                assert!(
                    new_cost[0] <= old_cost[0],
                    "{label} {endpoint}: {new_cost:?} > {old_cost:?}"
                );
            }
        }
        let counted = |enabled| with_text_walk_exact_for_test(enabled, || count(&ctx, &filter));
        assert_eq!(counted(true), counted(false), "{label}: count changed");
        saved += old[1][0] - new[1][0];
        println!(
            "{label:>26} | /query node reads {:>6} -> {:>6}, ms {:>5} -> {:>5} \
             | all endpoints node reads {:>6} -> {:>6}, ms {:>5} -> {:>5}",
            old[0][0],
            new[0][0],
            old[0][1] / 1000,
            new[0][1] / 1000,
            old[1][0],
            new[1][0],
            old[1][1] / 1000,
            new[1][1] / 1000,
        );
        // A path scope alone is exact: its members cost no node read.
        if ["path 5%", "path 20%", "path 40%", "path 60%"].contains(&label.as_str()) {
            assert!(
                new[1][0] * 10 <= old[1][0].max(1),
                "{label}: {new:?} vs {old:?}"
            );
        }
    }
    assert!(saved > 0);
}

/// The sparse paths check the filter lazily (best-first, until the cut is
/// decided), including the exact-scored candidate path: endpoints return the
/// same ids and score bits as eager evaluation with fewer node reads. Every
/// 50th path is an array here, so text walks stay supersets and rechecks run.
#[test]
fn sparse_lazy_recheck_matches_eager_on_endpoints() {
    use crate::helix_engine::vector_core::sparse::with_sparse_lazy_filter_for_test;
    let (ctx, _storage) = dir_fixture_with(true);
    let mut rng = rand::rngs::StdRng::seed_from_u64(23);
    let mut vector = || -> [f32; 3] { std::array::from_fn(|_| rng.random_range(-1.0f32..1.0)) };
    let queries: Vec<([f32; 3], [f32; 3])> = (0..2).map(|_| (vector(), vector())).collect();
    let (mut eager_reads, mut lazy_reads) = (0usize, 0usize);
    for (label, filter) in filter_shapes() {
        for (v, w) in &queries {
            for (endpoint, body) in requests(&filter, *v, *w) {
                let sparse = endpoint == "query/hybrid" || body["using"] == "lex_sparse";
                if !sparse {
                    continue;
                }
                let measure = |lazy: bool| {
                    with_sparse_lazy_filter_for_test(lazy, || run(&ctx, endpoint, &body))
                };
                let (eager, eager_cost) = measure(false);
                let (lazy, lazy_cost) = measure(true);
                assert_eq!(lazy, eager, "{label} {endpoint}: results changed");
                assert!(lazy_cost[0] <= eager_cost[0], "{label} {endpoint}");
                eager_reads += eager_cost[0];
                lazy_reads += lazy_cost[0];
            }
        }
    }
    println!("sparse endpoint node reads: eager {eager_reads}, lazy {lazy_reads}");
    assert!(
        lazy_reads * 2 < eager_reads,
        "{lazy_reads} vs {eager_reads}"
    );
}
