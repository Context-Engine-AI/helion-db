use super::dense_candidate_plan::with_prefilter_max_for_test;
use super::tests::{create_dense_collection_lsm, make_input, setup_with_config, upsert_points};
use super::*;
use crate::helix_engine::graph_core::config::Config;
use crate::helix_engine::storage_core::backend_lsm::lsm_round_trips_for_test;
use crate::helix_engine::storage_core::storage_core::node_reads_for_test;
use rand::{Rng, SeedableRng};
use std::sync::Arc;

pub(super) const REPO: &str = "example-go-gitea__gitea-4875bb3b-7dc453f25a60e8e1";
pub(super) const N: u32 = 6_000;
const ON: usize = 100_000;

/// Directory needles and the share of points under them; `/x60/` spans the
/// models and routers trees, `/work/` every point.
const NEEDLES: [(&str, u32); 5] = [
    ("a5_cmd/", 5),
    ("b20_models/", 20),
    ("C40_ROUTERS/", 40),
    ("/x60/", 60),
    ("/work/", 100),
];

fn dir_path(id: u32) -> String {
    let (top, mid) = match id % 20 {
        0 => ("a5_cmd", "y40"),
        1..=4 => ("b20_models", "x60"),
        5..=12 => ("c40_routers", "x60"),
        _ => ("d35_services", "y40"),
    };
    format!("/work/{REPO}/{top}/{mid}/F{id}.go")
}

/// One indexed HNSW segment (the serial search path, so the thread-local
/// counters see every node read) plus a sparse space. Every 50th path is a
/// one-element array: the text walk keeps it, `match.text` rejects it.
fn dir_fixture() -> (super::tests::TestContext, Arc<HelixGraphStorage>) {
    dir_fixture_with(true)
}

/// [`dir_fixture`]; `arrays = false` stores every path as a plain string.
pub(super) fn dir_fixture_with(
    arrays: bool,
) -> (super::tests::TestContext, Arc<HelixGraphStorage>) {
    let mut config = Config::default().with_lsm_in_memory();
    config.vector_config.flat_scan_threshold = Some(0);
    let ctx = setup_with_config(config);
    let input = make_input(
        &ctx,
        "PUT",
        "/collections/dirs",
        serde_json::to_vec(&serde_json::json!({
            "vectors": {"dense": {"size": 3, "distance": "Cosine"}},
            "sparse_vectors": {"lex_sparse": {"modifier": "idf"}}
        }))
        .unwrap(),
        HashMap::from([("name".into(), "dirs".into())]),
    );
    let mut response = Response::new();
    handle_create_collection(&input, &mut response).unwrap();
    assert!(response.status == 200 || response.status == 201);
    let storage = ctx.collections.get_collection("dirs").unwrap();
    assert_eq!(storage.backend.kind(), BackendKind::Lsm);
    for field in [
        "metadata.repo",
        "metadata.path",
        "metadata.language",
        "branch",
    ] {
        storage
            .create_payload_index(field, PayloadIndexSchema::Keyword)
            .unwrap();
    }
    let points = (1..=N)
        .map(|id| {
            let angle = f32::from(u16::try_from(id).unwrap()) * 0.37;
            let lift = f32::from(u16::try_from(id % 17).unwrap()) / 17.0;
            let path = if arrays && id % 50 == 7 {
                serde_json::json!([dir_path(id)])
            } else {
                serde_json::json!(dir_path(id))
            };
            let branch = match id % 97 {
                3 => serde_json::json!("private"),
                5 => serde_json::Value::Null,
                _ => serde_json::json!("public"),
            };
            serde_json::json!({
                "id": id,
                "vector": {
                    "dense": [angle.cos(), angle.sin(), lift],
                    "lex_sparse": {"indices": [id % 13, 100 + id % 7, 200 + id % 5, 300 + id % 3],
                                   "values": [1.0, 0.5, 0.25, 0.125]}
                },
                "payload": {
                    "metadata": {"repo": REPO, "path": path,
                                 "language": if id % 2 == 0 { "rust" } else { "go" }},
                    "branch": branch
                }
            })
        })
        .collect::<Vec<_>>();
    upsert_points(&ctx, "dirs", serde_json::Value::Array(points));
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
    (ctx, storage)
}

pub(super) fn path_text(needle: &str) -> serde_json::Value {
    serde_json::json!({"key": "metadata.path", "match": {"text": needle}})
}

/// `(label, filter)`: the client's directory scope alone and next to its single-repo
/// clause, plus exclusion, `should`, nested and `has_id` combinations.
pub(super) fn filter_shapes() -> Vec<(String, serde_json::Value)> {
    let repo_text = serde_json::json!({"key": "metadata.repo", "match": {"text": REPO}});
    let mut shapes = Vec::new();
    for (needle, share) in NEEDLES {
        shapes.push((
            format!("path {share}%"),
            serde_json::json!({"must": [path_text(needle)]}),
        ));
        shapes.push((
            format!("repo+path {share}%"),
            serde_json::json!({"must": [repo_text.clone(), path_text(needle)]}),
        ));
    }
    for needle in ["C40_ROUTERS/", "/x60/"] {
        shapes.extend([
            (
                format!("must_not keyword {needle}"),
                serde_json::json!({
                "must": [path_text(needle)],
                "must_not": [{"key": "branch", "match": {"value": "private"}}]}),
            ),
            (
                format!("must_not text {needle}"),
                serde_json::json!({
                "must": [repo_text.clone(), path_text(needle)],
                "must_not": [path_text("f1")]}),
            ),
            (
                format!("should {needle}"),
                serde_json::json!({
                "must": [path_text(needle)],
                "should": [{"key": "metadata.language", "match": {"value": "rust"}},
                           path_text("7.go")]}),
            ),
            (
                format!("has_id {needle}"),
                serde_json::json!({
                "must": [{"has_id": (1..=3_000).collect::<Vec<_>>()}, path_text(needle)]}),
            ),
            (
                format!("nested exclusion {needle}"),
                serde_json::json!({
                "must": [{"key": "metadata.repo", "match": {"value": REPO}},
                         {"must_not": [{"key": "branch", "match": {"value": "private"}}]},
                         path_text(needle)]}),
            ),
        ]);
    }
    let null_branch = serde_json::json!({"key": "branch", "match": {"value": null}});
    shapes.extend([
        (
            "must_not null path".into(),
            serde_json::json!({"must": [path_text("/x60/")], "must_not": [null_branch.clone()]}),
        ),
        (
            "must_not null repo+path".into(),
            serde_json::json!({"must": [repo_text.clone(), path_text("/x60/")],
                "must_not": [null_branch]}),
        ),
    ]);
    shapes.push((
        "nested should scopes".into(),
        serde_json::json!({"must": [repo_text,
            {"should": [path_text("c40_routers/"), path_text("b20_models/")]}]}),
    ));
    shapes
}

/// `(result json, [payload node reads, SlateDB round trips])` of one request.
pub(super) fn run(
    ctx: &super::tests::TestContext,
    endpoint: &str,
    body: &serde_json::Value,
) -> (serde_json::Value, [usize; 2]) {
    let input = make_input(
        ctx,
        "POST",
        &format!("/collections/dirs/points/{endpoint}"),
        serde_json::to_vec(body).unwrap(),
        HashMap::from([("name".into(), "dirs".into())]),
    );
    let mut response = Response::new();
    let before = [node_reads_for_test(), lsm_round_trips_for_test()];
    match endpoint {
        "search" => handle_search_points(&input, &mut response),
        "query" => handle_query_points(&input, &mut response),
        // The costs below and the callers' overrides are thread-local: keep the
        // sparse channels on this thread.
        "query/hybrid" => with_hybrid_parallel_channels_for_test(false, || {
            handle_hybrid_query_points(&input, &mut response)
        }),
        _ => unreachable!(),
    }
    .unwrap();
    let cost = [
        node_reads_for_test() - before[0],
        lsm_round_trips_for_test() - before[1],
    ];
    assert_eq!(
        response.status,
        200,
        "{}",
        String::from_utf8_lossy(&response.body)
    );
    let body: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
    (body["result"].clone(), cost)
}

pub(super) fn requests(
    filter: &serde_json::Value,
    v: [f32; 3],
    w: [f32; 3],
) -> Vec<(&'static str, serde_json::Value)> {
    // A channel filter of its own (a different broad scope) must pair with
    // its own prefilter, not the request's.
    let other = serde_json::json!({"must": [path_text("d35_services/")]});
    let sparse = serde_json::json!({"indices": [3, 102, 201, 300], "values": [1.0, 1.0, 1.0, 1.0]});
    vec![
        (
            "query",
            serde_json::json!({"query": v, "using": "dense", "limit": 24, "filter": filter}),
        ),
        (
            "search",
            serde_json::json!({"vector": {"name": "dense", "vector": v}, "limit": 24,
                "filter": filter}),
        ),
        (
            "query",
            serde_json::json!({"prefetch": [
                {"query": v, "using": "dense", "limit": 30},
                {"query": w, "using": "dense", "limit": 30}],
                "query": {"fusion": "rrf"}, "limit": 24, "filter": filter}),
        ),
        (
            "query/hybrid",
            serde_json::json!({"dense": [
                {"query": v, "using": "dense", "limit": 24},
                {"query": w, "using": "dense", "limit": 24, "filter": other}],
                "sparse": [
                {"query": sparse, "using": "lex_sparse", "limit": 24},
                {"query": sparse, "using": "lex_sparse", "limit": 24, "filter": other}],
                "limit": 24, "filter": filter, "with_channel_points": true}),
        ),
        (
            "query",
            serde_json::json!({"query": sparse, "using": "lex_sparse", "limit": 24,
                "filter": filter}),
        ),
    ]
}

/// The prefilter only skips node reads for points the superset rules out, so
/// every endpoint returns the same ids and score bits with it on or off.
#[test]
fn prefilter_matches_kill_switch_and_saves_node_reads() {
    let (ctx, _storage) = dir_fixture();
    let mut rng = rand::rngs::StdRng::seed_from_u64(11);
    let mut vector = || -> [f32; 3] { std::array::from_fn(|_| rng.random_range(-1.0f32..1.0)) };
    let queries: Vec<([f32; 3], [f32; 3])> = (0..3).map(|_| (vector(), vector())).collect();
    for (label, filter) in filter_shapes() {
        // [node reads, round trips, microseconds] for /points/query, then all.
        let mut old = [[0usize; 3]; 2];
        let mut new = [[0usize; 3]; 2];
        for (v, w) in &queries {
            for (index, (endpoint, body)) in requests(&filter, *v, *w).into_iter().enumerate() {
                let measure = |max: usize, acc: &mut [[usize; 3]; 2]| {
                    let started = Instant::now();
                    let (result, cost) =
                        with_prefilter_max_for_test(max, || run(&ctx, endpoint, &body));
                    let micros = usize::try_from(started.elapsed().as_micros()).unwrap();
                    for (slot, add) in [(0, cost[0]), (1, cost[1]), (2, micros)] {
                        acc[1][slot] += add;
                        if index == 0 {
                            acc[0][slot] += add;
                        }
                    }
                    (result, cost)
                };
                let (old_result, old_cost) = measure(0, &mut old);
                // Disable the broad scan so only the prefilter
                // membership optimization is tested, not the flat-scan
                // search path (which legitimately reorders ties).
                let (new_result, new_cost) =
                    with_broad_scan_for_test(false, || measure(ON, &mut new));
                assert_eq!(
                    new_result, old_result,
                    "{label} {endpoint}: results changed"
                );
                assert!(
                    new_cost[0] <= old_cost[0] && new_cost[1] <= old_cost[1],
                    "{label} {endpoint}: {new_cost:?} > {old_cost:?}"
                );
            }
        }
        println!(
            "{label:>26} | /query node reads {:>6} -> {:>6}, trips {:>6} -> {:>6}, ms {:>4} -> {:>4} \
             | all endpoints node reads {:>6} -> {:>6}",
            old[0][0], new[0][0], old[0][1], new[0][1], old[0][2] / 1000, new[0][2] / 1000,
            old[1][0], new[1][0],
        );
        // Past the keep caps, only superset members (the share plus a few
        // array paths) still cost a node read.
        let prefiltered = [
            ("path 40", 40),
            ("path 60", 60),
            ("repo+path 40", 40),
            ("repo+path 60", 60),
        ]
        .into_iter()
        .find(|(prefix, _)| label.starts_with(prefix));
        if let Some((_, share)) = prefiltered {
            assert!(
                new[1][0] * 100 < old[1][0] * (share + 15),
                "{label}: {new:?} vs {old:?}"
            );
        }
    }
}

/// Shapes that must get a prefilter on this fixture (6,000 points: keep cap
/// 1,500, exact-scoring cap 2,048); every other shape must not.
fn expects_prefilter(label: &str) -> bool {
    let broad = label.contains(" 40%") || label.contains(" 60%");
    let by_walk = [
        "must_not text",
        "has_id",
        "nested exclusion",
        "must_not null repo+path",
    ];
    broad || by_walk.iter().any(|prefix| label.starts_with(prefix))
}

#[test]
fn planner_keeps_dropped_supersets_as_prefilters_only() {
    let (_ctx, storage) = dir_fixture();
    let read = storage.backend.begin_read().unwrap();
    let nodes: Vec<_> = (1..=u128::from(N))
        .map(|id| (id, storage.get_node_be(&read, &id).unwrap().properties))
        .collect();
    for (label, value) in filter_shapes() {
        let filter: Filter = serde_json::from_value(value).unwrap();
        let plan = |max| {
            with_prefilter_max_for_test(max, || {
                vector_query_indexed_filter_candidates_be_probed(&storage, &read, &filter, true)
                    .unwrap()
            })
        };
        let (off, on) = (plan(0), plan(ON));
        assert_eq!(on.candidates, off.candidates, "{label}: planning changed");
        assert_eq!(on.probe_skipped, off.probe_skipped, "{label}");
        assert!(
            off.prefilter.is_none(),
            "{label}: kill switch kept a prefilter"
        );
        assert_eq!(on.prefilter.is_some(), expects_prefilter(&label), "{label}");
        if let Some(prefilter) = &on.prefilter {
            assert!(on.candidates.is_none(), "{label}");
            let matches: HashSet<u128> = nodes
                .iter()
                .filter(|(id, properties)| filter.matches_point(Some(*id), properties))
                .map(|(id, _)| *id)
                .collect();
            assert!(
                prefilter.is_superset(&matches),
                "{label}: prefilter lost a match"
            );
            assert!(prefilter.len() < nodes.len(), "{label}");
        }
    }
}

/// Every payload shape `match.text` can meet: the index walk is a superset of
/// the matches, and only array values (indexed per element, never matched)
/// are extra. A compacted (>480-byte) term cannot be tested by the walk and
/// stays a candidate.
#[test]
fn text_walk_is_a_superset_of_match_text() {
    let mut config = Config::default().with_lsm_in_memory();
    config.vector_config.flat_scan_threshold = Some(0);
    let ctx = setup_with_config(config);
    create_dense_collection_lsm(&ctx, "shapes");
    let storage = ctx.collections.get_collection("shapes").unwrap();
    storage
        .create_payload_index("metadata.path", PayloadIndexSchema::Keyword)
        .unwrap();
    let alphabet: Vec<char> = "abAB/_.-xyzßİÄΣσ7".chars().collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(5);
    let word = |rng: &mut rand::rngs::StdRng| -> String {
        let len = rng.random_range(0..10);
        (0..len)
            .map(|_| alphabet[rng.random_range(0..alphabet.len())])
            .collect()
    };
    let mut strings = Vec::new();
    let points = (1..=400u32)
        .map(|id| {
            let s = word(&mut rng);
            strings.push(s.clone());
            let metadata = match id % 10 {
                0 => serde_json::json!({"path": [s, word(&mut rng)]}),
                1 => serde_json::json!({"path": [s]}),
                2 => serde_json::json!({"path": [[s]]}),
                3 => serde_json::json!({"path": id}),
                4 => serde_json::json!({"path": id % 3 == 0}),
                5 => serde_json::json!({"path": null}),
                6 => serde_json::json!({"path": {"inner": s}}),
                7 => serde_json::json!({"other": s}),
                _ => serde_json::json!({"path": s}),
            };
            let metadata = if id == 399 {
                serde_json::json!(s)
            } else {
                metadata
            };
            serde_json::json!({"id": id, "vector": {"dense": [1.0, 0.0, 0.0]},
                               "payload": {"metadata": metadata}})
        })
        .collect::<Vec<_>>();
    upsert_points(&ctx, "shapes", serde_json::Value::Array(points));
    let read = storage.backend.begin_read().unwrap();
    let nodes: Vec<_> = (1..=400u128)
        .map(|id| (id, storage.get_node_be(&read, &id).unwrap()))
        .collect();

    let mut needles: Vec<String> = vec![
        "".into(),
        "zz-none".into(),
        "İ".into(),
        "i̇".into(),
        "SS".into(),
    ];
    for s in strings.iter().filter(|s| !s.is_empty()).take(120) {
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
    for needle in &needles {
        let filter: Filter =
            serde_json::from_value(serde_json::json!({"must": [path_text(needle)]})).unwrap();
        let walked: HashSet<u128> = storage
            .get_nodes_by_payload_text_be(&read, "metadata.path", needle, 0)
            .unwrap()
            .expect("short terms decode")
            .into_iter()
            .collect();
        for (id, node) in &nodes {
            let matched = filter.matches_point(Some(*id), &node.properties);
            let is_array = matches!(
                HelixGraphStorage::payload_value_for_key(&node.properties, "metadata.path"),
                Some(Value::Array(_))
            );
            assert!(
                !matched || walked.contains(id),
                "{needle:?}: walk missed {id}"
            );
            assert!(
                matched || !walked.contains(id) || is_array,
                "{needle:?}: non-array false positive {id}"
            );
        }
    }

    upsert_points(
        &ctx,
        "shapes",
        serde_json::json!([{"id": 401, "vector": {"dense": [1.0, 0.0, 0.0]},
        "payload": {"metadata": {"path": "p".repeat(600)}}}]),
    );
    // 401's path is indexed under a hash, so the walk cannot test it and
    // must keep it for the recheck.
    let read = storage.backend.begin_read().unwrap();
    assert!(storage
        .get_nodes_by_payload_text_be(&read, "metadata.path", "p", 0)
        .unwrap()
        .expect("a hashed key stays a candidate")
        .contains(&401));
}
