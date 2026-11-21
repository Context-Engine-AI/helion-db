//! Adversarial review of exact `match.text` walks: once a walk is exact the
//! planner stops rechecking nodes, so every write path must leave no marked
//! row that disagrees with its node and no indexed node without its row.
//! Indexed answers are compared with an unindexed collection (ground truth)
//! and with the exact-walk kill switch.

use super::tests::{create_dense_collection_lsm, make_input, setup_with_config, TestContext};
use super::*;
use crate::helix_engine::graph_core::config::Config;
use crate::helix_gateway::point_mutation_admission::point_mutation_collection;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

const FIELDS: [&str; 2] = ["metadata.path", "metadata.repo"];

/// Brute-force dense search, so indexed and unindexed answers are comparable.
fn lsm_ctx() -> TestContext {
    let mut config = Config::default().with_lsm_in_memory();
    config.vector_config.flat_scan_threshold = Some(1_000_000);
    setup_with_config(config)
}

/// One handler call; `result` of a 200 response.
fn call(ctx: &TestContext, name: &str, route: &str, body: serde_json::Value) -> serde_json::Value {
    let (method, path, mut params) = match route {
        "upsert" => ("PUT", format!("/collections/{name}/points"), HashMap::new()),
        "create_index" => ("PUT", format!("/collections/{name}/index"), HashMap::new()),
        "delete_index" => (
            "DELETE",
            format!("/collections/{name}/index/metadata.path"),
            HashMap::from([("field_name".to_string(), "metadata.path".to_string())]),
        ),
        other => (
            "POST",
            format!("/collections/{name}/points/{other}"),
            HashMap::new(),
        ),
    };
    params.insert("name".to_string(), name.to_string());
    let input = make_input(
        ctx,
        method,
        &path,
        serde_json::to_vec(&body).unwrap(),
        params,
    );
    let mut response = Response::new();
    match route {
        "upsert" => handle_upsert_points(&input, &mut response),
        "payload" => handle_set_payload(&input, &mut response),
        "delete" => handle_delete_points(&input, &mut response),
        "count" => handle_count_points(&input, &mut response),
        "query" => handle_query_points(&input, &mut response),
        "search" => handle_search_points(&input, &mut response),
        // Callers set thread-local overrides: keep the sparse channels on this
        // thread.
        "hybrid_query" => with_hybrid_parallel_channels_for_test(false, || {
            handle_hybrid_query_points(&input, &mut response)
        }),
        "scroll" => handle_scroll_points(&input, &mut response),
        "create_index" => handle_create_index(&input, &mut response),
        "delete_index" => handle_delete_index(&input, &mut response),
        _ => unreachable!(),
    }
    .unwrap();
    assert_eq!(
        response.status,
        200,
        "{route}: {}",
        String::from_utf8_lossy(&response.body)
    );
    let body: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
    body["result"].clone()
}

/// Values the index must hold rows for: the top-level value (a marked row)
/// or each array element (unmarked rows).
fn expected_values(props: &HashMap<String, Value>, field: &str) -> Vec<(Value, bool)> {
    match HelixGraphStorage::payload_value_for_key(props, field) {
        None => Vec::new(),
        Some(Value::Array(values)) => values.iter().map(|v| (v.clone(), false)).collect(),
        Some(v) => vec![(v.clone(), true)],
    }
}

/// A row key stands for `value` (`Value`'s numeric equality lets an update
/// from `I64(n)` to `U64(n)` keep the old key; both hold the same number).
fn row_holds(key: &[u8], value: &Value) -> bool {
    key == HelixGraphStorage::stable_index_key_for_value(value)
        .unwrap()
        .as_slice()
        || bincode::deserialize::<Value>(key).is_ok_and(|decoded| &decoded == value)
}

/// Every way an index row can disagree with its node: a row for a value the
/// node no longer holds (or a missing node) — a false positive once trusted;
/// a marker on an array row, which an exact walk would trust; and a value
/// with no row — a false negative through a kept set or the prefilter.
fn index_violations(storage: &HelixGraphStorage) -> Vec<String> {
    let read = storage.backend.begin_read().unwrap();
    let mut nodes = HashMap::new();
    storage
        .backend
        .scan(&read, Namespace::Nodes, KeyRange::all(), |key, bytes| {
            let id = u128::from_be_bytes(<[u8; 16]>::try_from(key).unwrap());
            nodes.insert(
                id,
                SerializedNode::decode_node(bytes, id).unwrap().properties,
            );
            true
        })
        .unwrap();
    let mut bad = Vec::new();
    for field in FIELDS {
        if storage.has_payload_index(field).is_none() {
            continue;
        }
        let db = HelixGraphStorage::payload_index_db_name(field, &PayloadIndexSchema::Keyword);
        let mut rows: HashMap<u128, Vec<(Vec<u8>, bool)>> = HashMap::new();
        storage
            .backend
            .scan_dup_stored(
                &read,
                Namespace::PayloadIndex(&db),
                KeyRange::all(),
                |key, id, stored| {
                    let id = u128::from_be_bytes(<[u8; 16]>::try_from(id).unwrap());
                    rows.entry(id)
                        .or_default()
                        .push((key.to_vec(), stored == [1u8]));
                    true
                },
            )
            .unwrap();
        let decode = |key: &[u8]| bincode::deserialize::<Value>(key).ok();
        for (id, id_rows) in &rows {
            let expected = nodes
                .get(id)
                .map(|props| expected_values(props, field))
                .unwrap_or_default();
            for (key, marked) in id_rows {
                match expected.iter().find(|(value, _)| row_holds(key, value)) {
                    None => bad.push(format!(
                        "{field}: stale {} row {:?} for node {id} ({:?})",
                        if *marked { "marked" } else { "unmarked" },
                        decode(key),
                        nodes
                            .get(id)
                            .and_then(|p| HelixGraphStorage::payload_value_for_key(p, field)),
                    )),
                    Some((_, false)) if *marked => bad.push(format!(
                        "{field}: marked array row {:?} for node {id}",
                        decode(key)
                    )),
                    _ => {}
                }
            }
        }
        for (id, props) in &nodes {
            for (value, _) in expected_values(props, field) {
                let held = rows
                    .get(id)
                    .is_some_and(|id_rows| id_rows.iter().any(|(key, _)| row_holds(key, &value)));
                if !held {
                    bad.push(format!("{field}: node {id} has no row for {value:?}"));
                }
            }
        }
    }
    bad
}

fn text(key: &str, needle: &str) -> serde_json::Value {
    json!({"key": key, "match": {"text": needle}})
}

fn filter_shapes() -> Vec<serde_json::Value> {
    let mut shapes = Vec::new();
    for needle in [
        "/a/", "/B/", "sub1", "sub2/f1", "f3", "/work/", "ß", "SS", "zz-none",
    ] {
        shapes.extend([
            json!({"must": [text("metadata.path", needle)]}),
            json!({"must": [text("metadata.repo", "r1"), text("metadata.path", needle)]}),
            json!({"must": [text("metadata.path", needle)],
                   "must_not": [{"key": "metadata.repo", "match": {"value": "r2"}}]}),
            json!({"should": [text("metadata.path", needle), text("metadata.repo", "R3")]}),
            json!({"must": [{"should": [text("metadata.path", needle), text("metadata.path", "sub0")]}]}),
            json!({"must": [text("metadata.path", needle)], "must_not": [text("metadata.path", "f1")]}),
            json!({"must": [text("metadata.path", needle),
                            {"key": "metadata.repo", "match": {"any": ["r1", "r3"]}}]}),
        ]);
    }
    shapes
}

const QUERY: [f32; 3] = [0.3, -0.8, 0.5];
const LIMIT: usize = 15;

/// Count, query and search answers for `filter`.
fn answers(ctx: &TestContext, name: &str, filter: &serde_json::Value) -> Vec<serde_json::Value> {
    vec![
        call(ctx, name, "count", json!({"filter": filter, "exact": true})),
        call(
            ctx,
            name,
            "query",
            json!({"query": QUERY, "using": "dense", "limit": LIMIT, "filter": filter}),
        )["points"]
            .clone(),
        call(
            ctx,
            name,
            "search",
            json!({"vector": {"name": "dense", "vector": QUERY}, "limit": LIMIT, "filter": filter}),
        ),
    ]
}

fn cosine(a: [f32; 3], b: [f32; 3]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let norm = |v: [f32; 3]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

/// Against `matches_point` over every stored node (the truth): exact counts
/// are equal, results hold no non-match (false positive) and miss no match
/// scoring above the last result (false negative); with the exact-walk
/// switch on and off, which must also agree with each other.
fn assert_answers_match(
    ctx: &TestContext,
    name: &str,
    vectors: &HashMap<u128, [f32; 3]>,
    stage: &str,
) {
    let storage = ctx.collections.get_collection(name).unwrap();
    let read = storage.backend.begin_read().unwrap();
    let mut nodes = Vec::new();
    storage
        .backend
        .scan(&read, Namespace::Nodes, KeyRange::all(), |key, bytes| {
            let id = u128::from_be_bytes(<[u8; 16]>::try_from(key).unwrap());
            nodes.push((
                id,
                SerializedNode::decode_node(bytes, id).unwrap().properties,
            ));
            true
        })
        .unwrap();
    drop(read);
    for filter_json in filter_shapes() {
        let filter: Filter = serde_json::from_value(filter_json.clone()).unwrap();
        let matching: HashSet<u128> = nodes
            .iter()
            .filter(|(id, props)| filter.matches_point(Some(*id), props))
            .map(|(id, _)| *id)
            .collect();
        let on = with_text_walk_exact_for_test(true, || answers(ctx, name, &filter_json));
        let off = with_text_walk_exact_for_test(false, || answers(ctx, name, &filter_json));
        assert_eq!(on, off, "{stage}: switch on vs off for {filter_json}");
        assert_eq!(
            on[0]["count"].as_u64(),
            Some(matching.len() as u64),
            "{stage}: count for {filter_json}"
        );
        for list in &on[1..] {
            let hits: Vec<(u128, f64)> = list
                .as_array()
                .unwrap()
                .iter()
                .map(|hit| {
                    let id = u128::from_str_radix(hit["id"].as_str().unwrap(), 16).unwrap();
                    (id, hit["score"].as_f64().unwrap())
                })
                .collect();
            for (id, _) in &hits {
                assert!(
                    matching.contains(id),
                    "{stage}: false positive {id} for {filter_json}"
                );
            }
            let floor = if hits.len() < LIMIT.min(matching.len()) {
                panic!(
                    "{stage}: {} hits of {} matches for {filter_json}",
                    hits.len(),
                    matching.len()
                );
            } else {
                hits.iter()
                    .map(|(_, score)| *score)
                    .fold(f64::INFINITY, f64::min)
            };
            for id in &matching {
                let missed = !hits.iter().any(|(hit, _)| hit == id)
                    && cosine(QUERY, vectors[id]) > floor + 1e-4;
                assert!(!missed, "{stage}: false negative {id} for {filter_json}");
            }
        }
    }
}

fn path_value(rng: &mut StdRng, id: u128) -> serde_json::Value {
    let dir = ["a", "B", "c", "straße"][rng.random_range(0..4)];
    let path = format!("/work/r/{dir}/sub{}/f{id}.rs", rng.random_range(0..4));
    match rng.random_range(0..12) {
        0 => json!([path]),
        1 => json!([[path]]),
        2 => json!({"inner": path}),
        3 => serde_json::Value::Null,
        4 => json!(id as u64),
        _ => json!(path),
    }
}

fn repo_value(rng: &mut StdRng) -> serde_json::Value {
    match rng.random_range(0..8) {
        0 => json!(["r1"]),
        1 => json!("R2"),
        2 => json!("r2"),
        3 => json!("r3"),
        _ => json!("r1"),
    }
}

/// A point; most re-upserts keep the id's vector, so a payload change takes
/// the content-hash payload-only path.
fn random_point(
    rng: &mut StdRng,
    vectors: &mut HashMap<u128, [f32; 3]>,
    id: u128,
    round: u32,
) -> serde_json::Value {
    let metadata = match rng.random_range(0..14) {
        0 => json!("not-an-object"),
        1 => json!({"repo": repo_value(rng)}),
        _ => json!({"path": path_value(rng, id), "repo": repo_value(rng)}),
    };
    let shift = if rng.random_bool(0.3) {
        round as f32
    } else {
        0.0
    };
    let angle = id as f32 * 0.37 + shift;
    let vector = [angle.cos(), angle.sin(), (id % 17) as f32 / 17.0];
    vectors.insert(id, vector);
    json!({"id": id as u64, "vector": {"dense": vector}, "payload": {"metadata": metadata}})
}

fn live_ids(storage: &HelixGraphStorage) -> Vec<u64> {
    let read = storage.backend.begin_read().unwrap();
    let mut ids = Vec::new();
    storage
        .backend
        .scan(&read, Namespace::Nodes, KeyRange::all(), |key, _| {
            ids.push(u128::from_be_bytes(<[u8; 16]>::try_from(key).unwrap()) as u64);
            true
        })
        .unwrap();
    ids
}

/// Random shapes (plain, arrays, nested arrays, objects, null, numbers,
/// missing and non-object parents) go through every point write route the
/// gateway serves: upserts (full and content-hash payload-only), repeated
/// ids in one batch, nested set_payload, deletes by id and by
/// a text filter, and an index drop + rebuild. After each round the index
/// holds no stale marked row and no missing row, and count, query and
/// search equal an unindexed collection's answers with the switch on or off.
#[test]
fn every_point_write_route_keeps_exact_walks_equal_to_truth() {
    let ctx = lsm_ctx();
    create_dense_collection_lsm(&ctx, "idx");
    let idx = ctx.collections.get_collection("idx").unwrap();
    let mut rng = StdRng::seed_from_u64(97);
    let mut vectors = HashMap::new();
    let upsert =
        |points: Vec<serde_json::Value>| call(&ctx, "idx", "upsert", json!({"points": points}));
    let initial: Vec<_> = (1..=300u128)
        .map(|id| random_point(&mut rng, &mut vectors, id, 0))
        .collect();
    upsert(initial);
    for field in FIELDS {
        idx.create_payload_index(field, PayloadIndexSchema::Keyword)
            .unwrap();
    }
    let bad = index_violations(&idx);
    assert!(bad.is_empty(), "after build: {bad:#?}");
    assert_answers_match(&ctx, "idx", &vectors, "after build");

    for round in 1..=6u32 {
        let batch: Vec<_> = (0..80)
            .map(|_| {
                let id = rng.random_range(1..=330u128);
                random_point(&mut rng, &mut vectors, id, round)
            })
            .collect();
        upsert(batch);

        // The later copy of a repeated id wins, as in the vector map.
        let repeated: Vec<_> = (0..10)
            .flat_map(|_| {
                let id = rng.random_range(1..=300u128);
                [
                    random_point(&mut rng, &mut vectors, id, round),
                    random_point(&mut rng, &mut vectors, id, round + 7),
                ]
            })
            .collect();
        upsert(repeated);

        // set_payload values cannot be JSON objects or null (400).
        let live = live_ids(&idx);
        for _ in 0..20 {
            let id = live[rng.random_range(0..live.len())];
            let mut path = path_value(&mut rng, u128::from(id));
            if path.is_object() || path.is_null() {
                path = json!(format!("/work/r/a/sub{}/f{id}.rs", round % 4));
            }
            call(
                &ctx,
                "idx",
                "payload",
                json!({"points": [id], "key": "metadata",
                       "payload": {"path": path, "repo": repo_value(&mut rng)}}),
            );
        }

        let doomed: Vec<u64> = (0..10)
            .map(|_| live[rng.random_range(0..live.len())])
            .collect();
        call(&ctx, "idx", "delete", json!({"points": doomed}));
        let needle = format!("sub{}/f{}", round % 4, round);
        call(
            &ctx,
            "idx",
            "delete",
            json!({"filter": {"must": [text("metadata.path", &needle)]}}),
        );

        if round == 3 {
            // Points move while no index maintains them, then a rebuild.
            idx.delete_payload_index("metadata.path").unwrap();
            let moved: Vec<_> = (1..=40u128)
                .map(|id| random_point(&mut rng, &mut vectors, id, round))
                .collect();
            upsert(moved);
            idx.create_payload_index("metadata.path", PayloadIndexSchema::Keyword)
                .unwrap();
        }

        let bad = index_violations(&idx);
        assert!(bad.is_empty(), "round {round}: {bad:#?}");
        assert_answers_match(&ctx, "idx", &vectors, &format!("round {round}"));
    }
}

/// The gateway's point-mutation gate is keyed by the URL name, so a write
/// under an alias (the client's collection migrations create them) and one under the
/// real name are admitted together. Only the storage-level gate, which
/// upserts, deletes, payload-index creation and GC take on the resolved
/// name, can order them: set_payload must wait for it too.
#[test]
fn set_payload_waits_for_the_resolved_point_mutation_gate() {
    let ctx = lsm_ctx();
    create_dense_collection_lsm(&ctx, "gated");
    let mut rng = StdRng::seed_from_u64(5);
    let point = random_point(&mut rng, &mut HashMap::new(), 1, 0);
    call(&ctx, "gated", "upsert", json!({"points": [point]}));
    ctx.collections
        .create_alias("gated_alias", "gated")
        .unwrap();
    assert_ne!(
        point_mutation_collection("POST", "/collections/gated_alias/points/payload"),
        point_mutation_collection("PUT", "/collections/gated/points"),
    );

    let gate = ctx.collections.point_mutation_gate("gated");
    let held = gate.lock().unwrap();
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let ctx = &ctx;
        scope.spawn(move || {
            call(
                ctx,
                "gated_alias",
                "payload",
                json!({"points": [1], "key": "metadata", "payload": {"note": 1}}),
            );
            done_tx.send(()).unwrap();
        });
        let ran_while_gated = done_rx
            .recv_timeout(std::time::Duration::from_millis(500))
            .is_ok();
        drop(held);
        done_rx
            .recv_timeout(std::time::Duration::from_secs(30))
            .ok();
        assert!(
            !ran_while_gated,
            "set_payload committed while the collection's point-mutation gate was held"
        );
    });
}

fn dir_point(id: u128, dir: &str) -> serde_json::Value {
    let angle = id as f32 * 0.37;
    json!({"id": id as u64, "vector": {"dense": [angle.cos(), angle.sin(), 0.5]},
           "payload": {"metadata": {"path": format!("/work/r/{dir}/f{id}.rs"), "repo": "r1"}}})
}

/// set_payload under an alias racing upserts under the real name: the
/// upsert moves point 1 between directories while set_payload rewrites
/// 150 points (point 1 first). An upsert landing between set_payload's read
/// of point 1 and its commit leaves the upsert's marked row on a node that
/// set_payload rewrote with the old path, and no row for that path.
#[test]
fn alias_set_payload_racing_upserts_leaves_no_stale_marked_row() {
    let ctx = lsm_ctx();
    create_dense_collection_lsm(&ctx, "race");
    let storage = ctx.collections.get_collection("race").unwrap();
    storage
        .create_payload_index("metadata.path", PayloadIndexSchema::Keyword)
        .unwrap();
    let points: Vec<_> = (1..=150u128).map(|id| dir_point(id, "d1")).collect();
    call(&ctx, "race", "upsert", json!({"points": points}));
    ctx.collections.create_alias("race_alias", "race").unwrap();

    let stop = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let (ctx, stop) = (&ctx, &stop);
        scope.spawn(move || {
            // Jitter: both writers otherwise wake on the same WAL flush and
            // the upsert always commits before set_payload's window opens.
            let mut rng = StdRng::seed_from_u64(3);
            for _ in 0..400 {
                let dir = ["d1", "d2", "d3", "d4", "d5"][rng.random_range(0..5)];
                std::thread::sleep(std::time::Duration::from_micros(rng.random_range(0..4_000)));
                call(
                    ctx,
                    "race",
                    "upsert",
                    json!({"points": [dir_point(1, dir)]}),
                );
            }
            stop.store(true, AtomicOrdering::SeqCst);
        });
        scope.spawn(move || {
            let ids: Vec<u64> = (1..=150).collect();
            let mut note = 0u64;
            while !stop.load(AtomicOrdering::SeqCst) {
                call(
                    ctx,
                    "race_alias",
                    "payload",
                    json!({"points": ids, "key": "metadata", "payload": {"note": note}}),
                );
                note += 1;
                // std mutexes are unfair: let the upserts in once gated.
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            println!("set_payload calls: {note}");
        });
    });
    let bad = index_violations(&storage);
    assert!(bad.is_empty(), "{bad:#?}");
    for dir in ["/d1/", "/d2/", "/d3/", "/d4/", "/d5/"] {
        let filter = json!({"must": [text("metadata.path", dir)]});
        let exact = with_text_walk_exact_for_test(true, || {
            call(
                &ctx,
                "race",
                "count",
                json!({"filter": filter, "exact": true}),
            )
        });
        let rechecked = with_text_walk_exact_for_test(false, || {
            call(
                &ctx,
                "race",
                "count",
                json!({"filter": filter, "exact": true}),
            )
        });
        assert_eq!(exact, rechecked, "{dir}");
    }
}

/// Index creation (`ReplicatedMutation::CreatePayloadIndex`) publishes its
/// Building handle under the same gate, so a gated set_payload either sees
/// the handle (and records build deltas) or commits before the build reads.
#[test]
fn index_creation_waits_for_the_point_mutation_gate() {
    let ctx = lsm_ctx();
    create_dense_collection_lsm(&ctx, "gated_build");
    call(
        &ctx,
        "gated_build",
        "upsert",
        json!({"points": [dir_point(1, "d1")]}),
    );
    let gate = ctx.collections.point_mutation_gate("gated_build");
    let held = gate.lock().unwrap();
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let ctx = &ctx;
        scope.spawn(move || {
            call(
                ctx,
                "gated_build",
                "create_index",
                json!({"field_name": "metadata.path", "field_schema": "keyword"}),
            );
            done_tx.send(()).unwrap();
        });
        let ran_while_gated = done_rx
            .recv_timeout(std::time::Duration::from_millis(500))
            .is_ok();
        drop(held);
        done_rx
            .recv_timeout(std::time::Duration::from_secs(30))
            .ok();
        assert!(
            !ran_while_gated,
            "index creation ran while the gate was held"
        );
    });
}

/// Values longer than the raw key limit are stored under a hashed key the
/// walk cannot decode; it must leave them to the recheck instead of missing
/// them.
#[test]
fn oversized_paths_keep_exact_answers_equal_to_truth() {
    let ctx = lsm_ctx();
    create_dense_collection_lsm(&ctx, "big");
    let storage = ctx.collections.get_collection("big").unwrap();
    storage
        .create_payload_index("metadata.path", PayloadIndexSchema::Keyword)
        .unwrap();
    let long_tail = "x".repeat(600);
    let mut vectors = HashMap::new();
    let points: Vec<_> = (1..=120u128)
        .map(|id| {
            let path = if id % 10 == 0 {
                format!("/work/r/big/{long_tail}/needle_tail_{id}.rs")
            } else {
                format!("/work/r/small/f{id}.rs")
            };
            let angle = id as f32 * 0.37;
            let vector = [angle.cos(), angle.sin(), 0.5];
            vectors.insert(id, vector);
            json!({"id": id as u64, "vector": {"dense": vector},
                   "payload": {"metadata": {"path": path, "repo": "r1"}}})
        })
        .collect();
    call(&ctx, "big", "upsert", json!({"points": points}));
    assert!(index_violations(&storage).is_empty());
    assert_answers_match(&ctx, "big", &vectors, "oversized");
}

/// Rollback hazard (documents the lever): rows marked by this binary that an
/// older binary leaves behind — it neither maintains a dropped index nor
/// clears rows before rebuilding one — survive roll-forward with the
/// backfill already `done`, and exact walks then trust them. The
/// documented lever (`HELIX_TEXT_WALK_EXACT=0`) answers correctly.
#[test]
fn rollback_left_marked_rows_are_answered_correctly_with_the_kill_switch() {
    let ctx = lsm_ctx();
    create_dense_collection_lsm(&ctx, "rb");
    create_dense_collection_lsm(&ctx, "raw");
    let storage = ctx.collections.get_collection("rb").unwrap();
    storage
        .create_payload_index("metadata.path", PayloadIndexSchema::Keyword)
        .unwrap();
    let before: Vec<_> = (1..=40u128)
        .map(|id| dir_point(id, if id <= 20 { "old" } else { "keep" }))
        .collect();
    let after: Vec<_> = (1..=40u128).map(|id| dir_point(id, "keep")).collect();
    for name in ["rb", "raw"] {
        call(&ctx, name, "upsert", json!({"points": before}));
    }
    let db =
        HelixGraphStorage::payload_index_db_name("metadata.path", &PayloadIndexSchema::Keyword);
    let old_rows: Vec<(Vec<u8>, u128)> = (1..=20u128)
        .map(|id| {
            let key = HelixGraphStorage::stable_index_key_for_value(&Value::String(format!(
                "/work/r/old/f{id}.rs"
            )))
            .unwrap();
            (key, id)
        })
        .collect();
    // Older binary: drop (rows stay), points move while nothing maintains
    // the index, rebuild without clearing (this binary's rebuild plus the
    // dropped generation's rows it would have left).
    storage.delete_payload_index("metadata.path").unwrap();
    for name in ["rb", "raw"] {
        call(&ctx, name, "upsert", json!({"points": after}));
    }
    storage
        .create_payload_index("metadata.path", PayloadIndexSchema::Keyword)
        .unwrap();
    let mut write = storage.backend.begin_write().unwrap();
    for (key, id) in &old_rows {
        storage
            .backend
            .put_dup_stored(
                &mut write,
                Namespace::PayloadIndex(&db),
                key,
                &id.to_be_bytes(),
                &[1],
            )
            .unwrap();
    }
    storage.backend.commit(write).unwrap();

    let filter = json!({"must": [text("metadata.path", "/old/")]});
    let truth = answers(&ctx, "raw", &filter);
    let trusted = with_text_walk_exact_for_test(true, || answers(&ctx, "rb", &filter));
    let lever = with_text_walk_exact_for_test(false, || answers(&ctx, "rb", &filter));
    println!(
        "rollback hazard: truth {} / trusted {}",
        truth[0], trusted[0]
    );
    assert_eq!(lever, truth);
}

/// Writers that keep changing the indexed field (each set_payload rewrites
/// all 300 points, back to back) must not keep an async build from
/// publishing: the delta drain is one bounded pass, so the index turns Ready
/// within seconds, agrees with every node, and a delete right after returns
/// promptly (the build lock is held per commit only).
#[test]
fn async_build_publishes_under_hot_set_payload() {
    let ctx = lsm_ctx();
    create_dense_collection_lsm(&ctx, "hot");
    let storage = ctx.collections.get_collection("hot").unwrap();
    let points: Vec<_> = (1..=300u128).map(|id| dir_point(id, "d1")).collect();
    call(&ctx, "hot", "upsert", json!({"points": points}));
    let stop = AtomicBool::new(false);
    let writes = std::thread::scope(|scope| {
        let (ctx, stop) = (&ctx, &stop);
        let writer = scope.spawn(move || {
            let ids: Vec<u64> = (1..=300).collect();
            let mut writes = 0u64;
            while !stop.load(AtomicOrdering::SeqCst) {
                let dir = if writes % 2 == 0 { "d2" } else { "d1" };
                call(
                    ctx,
                    "hot",
                    "payload",
                    json!({"points": ids, "key": "metadata",
                    "payload": {"path": format!("/work/r/{dir}/f.rs")}}),
                );
                writes += 1;
            }
            writes
        });
        // Stops the writer on a failed assert too, so the scope can join it.
        struct StopOnDrop<'a>(&'a AtomicBool);
        impl Drop for StopOnDrop<'_> {
            fn drop(&mut self) {
                self.0.store(true, AtomicOrdering::SeqCst);
            }
        }
        let stopper = StopOnDrop(stop);
        for cycle in 0..2 {
            let started = Instant::now();
            call(
                ctx,
                "hot",
                "create_index",
                json!({"field_name": "metadata.path", "field_schema": "keyword"}),
            );
            loop {
                let status = storage.payload_index_status("metadata.path").unwrap();
                let status = status.map(|s| (s.status, s.error));
                match status.as_ref().map(|(status, _)| *status) {
                    Some("ready") => break,
                    Some("building") => assert!(
                        started.elapsed() < std::time::Duration::from_secs(20),
                        "cycle {cycle}: still building after {:?}",
                        started.elapsed()
                    ),
                    _ => panic!("cycle {cycle}: {status:?}"),
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            assert_eq!(
                index_violations(&storage),
                Vec::<String>::new(),
                "cycle {cycle}"
            );
            let deleting = Instant::now();
            call(ctx, "hot", "delete_index", json!({}));
            assert!(
                deleting.elapsed() < std::time::Duration::from_secs(5),
                "cycle {cycle}: delete took {:?}",
                deleting.elapsed()
            );
        }
        drop(stopper);
        writer.join().unwrap()
    });
    // The writer must have stayed busy through both builds. The count depends
    // on the machine (11 or more locally, 9 on a CI runner), so this only
    // rules out a writer that barely ran.
    assert!(writes > 3, "{writes} writes");
}

/// handle_set_payload's LSM body (props from a read, then one write batch
/// that re-reads each node) with `between` running inside its window.
fn ungated_set_payload(storage: &HelixGraphStorage, id: u128, between: impl FnOnce()) {
    let read = storage.backend.begin_read().unwrap();
    let mut props = storage.get_node_be(&read, &id).unwrap().properties;
    drop(read);
    if let Some(Value::Object(metadata)) = props.get_mut("metadata") {
        metadata.insert("note".into(), Value::U64(1));
    }
    let upsert = crate::helix_engine::storage_core::upsert::NodeUpsert {
        id,
        label: "point".into(),
        properties: props,
    };
    storage
        .with_write_backend(|w| {
            storage.upsert_node_be(w, &upsert)?;
            between();
            Ok(())
        })
        .unwrap();
}

/// What the gate prevents, and proof the audit sees it: an upsert landing
/// inside set_payload's window leaves its marked row on a node set_payload
/// rewrote with the old path, and an exact walk counts it.
#[test]
fn audit_flags_an_upsert_inside_an_ungated_set_payload() {
    let ctx = lsm_ctx();
    create_dense_collection_lsm(&ctx, "det");
    let storage = ctx.collections.get_collection("det").unwrap();
    storage
        .create_payload_index("metadata.path", PayloadIndexSchema::Keyword)
        .unwrap();
    call(
        &ctx,
        "det",
        "upsert",
        json!({"points": [dir_point(1, "d1")]}),
    );
    ungated_set_payload(&storage, 1, || {
        call(
            &ctx,
            "det",
            "upsert",
            json!({"points": [dir_point(1, "d2")]}),
        );
    });
    let bad = index_violations(&storage);
    assert_eq!(bad.len(), 2, "{bad:#?}");
    let filter = json!({"must": [text("metadata.path", "/d2/")]});
    let count = |exact| {
        with_text_walk_exact_for_test(exact, || {
            call(
                &ctx,
                "det",
                "count",
                json!({"filter": filter, "exact": true}),
            )["count"]
                .as_u64()
        })
    };
    assert_eq!((count(true), count(false)), (Some(1), Some(0)));
}

/// The same for index creation: the build reads and marks the old value
/// while set_payload, which saw no index, commits the new one unindexed.
#[test]
fn audit_flags_index_creation_inside_an_ungated_set_payload() {
    let ctx = lsm_ctx();
    create_dense_collection_lsm(&ctx, "det_build");
    let storage = ctx.collections.get_collection("det_build").unwrap();
    call(
        &ctx,
        "det_build",
        "upsert",
        json!({"points": [dir_point(1, "d1")]}),
    );
    let read = storage.backend.begin_read().unwrap();
    let mut moved = storage.get_node_be(&read, &1).unwrap().properties;
    drop(read);
    moved.insert(
        "metadata".into(),
        Value::from(json!({"path": "/work/r/d2/f1.rs"})),
    );
    let upsert = crate::helix_engine::storage_core::upsert::NodeUpsert {
        id: 1,
        label: "point".into(),
        properties: moved,
    };
    storage
        .with_write_backend(|w| {
            storage.upsert_node_be(w, &upsert)?;
            storage
                .create_payload_index("metadata.path", PayloadIndexSchema::Keyword)
                .unwrap();
            Ok(())
        })
        .unwrap();
    let bad = index_violations(&storage);
    assert_eq!(bad.len(), 2, "{bad:#?}");
}

/// Lazy sparse filtering under massive score ties (the near-tie rescore cap
/// and id tie-breaks decide the cut) and negative weights: WAND, full scan
/// and exact candidate scoring return the same ids and score bits lazily
/// and eagerly.
#[test]
fn lazy_sparse_filter_matches_eager_under_massive_ties() {
    use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
    use crate::helix_engine::vector_core::sparse::{
        with_sparse_lazy_filter_for_test, SparseVectorCore,
    };
    let backend = std::sync::Arc::new(AnyBackend::Lsm(
        LsmBackend::open_in_memory("review-sparse-ties").unwrap(),
    ));
    let config = |wand_enabled| SparseVectorConfig {
        full_scan_threshold: 1,
        modifier: SparseModifier::Idf,
        wand_enabled,
    };
    let wand =
        SparseVectorCore::new_lsm("ties_wand", config(true), std::sync::Arc::clone(&backend))
            .unwrap();
    let full =
        SparseVectorCore::new_lsm("ties_full", config(false), std::sync::Arc::clone(&backend))
            .unwrap();
    let docs: Vec<(u128, SparseVector)> = (1..=2_000u128)
        .map(|id| {
            let (indices, values) = match id % 10 {
                0 => (vec![1, 2, 3, 9], vec![1.0, 1.0, 0.5, 2.0]),
                1 => (vec![2, 3], vec![1.0, 1.0]),
                _ => (vec![1, 2, 3], vec![1.0, 1.0, 1.0]),
            };
            (id, SparseVector { indices, values })
        })
        .collect();
    let items: Vec<(u128, &SparseVector)> = docs.iter().map(|(id, sv)| (*id, sv)).collect();
    let mut w = backend.begin_write().unwrap();
    wand.upsert_batch_be(&mut w, &items).unwrap();
    full.upsert_batch_be(&mut w, &items).unwrap();
    backend.commit(w).unwrap();
    let r = backend.begin_read().unwrap();
    let bits = |hits: Vec<(u128, f64)>| -> Vec<(u128, u64)> {
        hits.into_iter()
            .map(|(id, score)| (id, score.to_bits()))
            .collect()
    };
    let queries = [
        SparseVector {
            indices: vec![1, 2, 3],
            values: vec![1.0, 1.0, 1.0],
        },
        SparseVector {
            indices: vec![1, 2, 3, 9],
            values: vec![1.0, 1.0, 1.0, 0.25],
        },
        SparseVector {
            indices: vec![1, 2, 3, 9],
            values: vec![1.0, -0.5, 1.0, 1.0],
        },
    ];
    let candidates: Vec<u128> = (1..=2_000u128).filter(|id| id % 4 != 0).collect();
    for (q, query) in queries.iter().enumerate() {
        for kind in 0..4 {
            let passes = |id: u128| match kind {
                0 => id > 500,
                1 => id % 7 == 0,
                2 => id > 1_990 || id % 10 == 1,
                _ => id % 10 != 0,
            };
            for limit in [1usize, 7, 33, 100, 3_000] {
                let run = |lazy: bool| {
                    with_sparse_lazy_filter_for_test(lazy, || {
                        (
                            bits(wand.search_be(&r, query, limit, Some(&passes)).unwrap()),
                            bits(full.search_be(&r, query, limit, Some(&passes)).unwrap()),
                            bits(
                                full.search_candidate_ids_exact_filtered_be(
                                    &r,
                                    query,
                                    limit,
                                    candidates.iter().copied(),
                                    Some(&passes),
                                )
                                .unwrap(),
                            ),
                        )
                    })
                };
                assert_eq!(
                    run(true),
                    run(false),
                    "query {q} filter {kind} limit {limit}"
                );
            }
        }
    }
}

/// `match.value` / `match.any` over stored numbers (above i64::MAX, floats
/// equal to integers, negatives), JSON null and the string "null", bools and
/// flat arrays: the indexed answers (trusted as exact) equal `matches_point`.
#[test]
fn match_value_lookups_equal_truth_across_number_and_null_shapes() {
    let ctx = lsm_ctx();
    create_dense_collection_lsm(&ctx, "vals");
    let storage = ctx.collections.get_collection("vals").unwrap();
    storage
        .create_payload_index("metadata.repo", PayloadIndexSchema::Keyword)
        .unwrap();
    let shapes = [
        json!(7),
        json!(7.0),
        json!(-7),
        json!(u64::MAX),
        json!(9_223_372_036_854_775_808u64),
        json!(1.5),
        serde_json::Value::Null,
        json!("null"),
        json!(true),
        json!("7"),
        json!([7, "x"]),
        json!([null]),
        // Not `[[7]]`: keyword rows hold a nested array's inner array, so a
        // match.value lookup misses what matches_point matches (also on main).
    ];
    let mut vectors = HashMap::new();
    let points: Vec<_> = (1..=130u128)
        .map(|id| {
            let angle = id as f32 * 0.37;
            let vector = [angle.cos(), angle.sin(), 0.5];
            vectors.insert(id, vector);
            json!({"id": id as u64, "vector": {"dense": vector},
                   "payload": {"metadata": {"repo": shapes[(id as usize) % shapes.len()]}}})
        })
        .collect();
    call(&ctx, "vals", "upsert", json!({"points": points}));
    assert!(index_violations(&storage).is_empty());
    let read = storage.backend.begin_read().unwrap();
    let nodes: Vec<_> = (1..=130u128)
        .map(|id| (id, storage.get_node_be(&read, &id).unwrap().properties))
        .collect();
    drop(read);
    let mut filters = Vec::new();
    for operand in shapes.iter().filter(|v| !v.is_array()) {
        let cond = json!({"key": "metadata.repo", "match": {"value": operand}});
        filters.push(json!({"must": [cond]}));
        filters.push(
            json!({"must": [{"key": "metadata.repo", "match": {"text": "x"}}], "must_not": [cond]}),
        );
        filters.push(json!({"must": [{"key": "metadata.repo", "match": {"any": [operand, "7"]}}]}));
    }
    for filter_json in filters {
        let filter: Filter = serde_json::from_value(filter_json.clone()).unwrap();
        let truth = nodes
            .iter()
            .filter(|(id, props)| filter.matches_point(Some(*id), props))
            .count() as u64;
        let count = call(
            &ctx,
            "vals",
            "count",
            json!({"filter": filter_json, "exact": true}),
        );
        assert_eq!(count["count"].as_u64(), Some(truth), "{filter_json}");
        let hits = call(
            &ctx,
            "vals",
            "query",
            json!({"query": QUERY, "using": "dense", "limit": 200, "filter": filter_json}),
        );
        assert_eq!(
            hits["points"].as_array().unwrap().len() as u64,
            truth,
            "query {filter_json}"
        );
    }
}

/// [`call`] that tolerates rejected requests (a set_payload or delete of a
/// point another writer just deleted).
fn try_call(ctx: &TestContext, name: &str, route: &str, body: serde_json::Value) -> bool {
    let path = match route {
        "upsert" => format!("/collections/{name}/points"),
        other => format!("/collections/{name}/points/{other}"),
    };
    let method = if route == "upsert" { "PUT" } else { "POST" };
    let input = make_input(
        ctx,
        method,
        &path,
        serde_json::to_vec(&body).unwrap(),
        HashMap::from([("name".to_string(), name.to_string())]),
    );
    let mut response = Response::new();
    let result = match route {
        "upsert" => handle_upsert_points(&input, &mut response),
        "payload" => handle_set_payload(&input, &mut response),
        "delete" => handle_delete_points(&input, &mut response),
        _ => unreachable!(),
    };
    result.is_ok() && response.status == 200
}

/// Stops writer threads when the test thread leaves the scope, even on a
/// failed assert, so the scope can join them.
struct StopWriters<'a>(&'a AtomicBool);

impl Drop for StopWriters<'_> {
    fn drop(&mut self) {
        self.0.store(true, AtomicOrdering::SeqCst);
    }
}

const DIRS: [&str; 5] = ["d1", "d2", "d3", "d4", "d5"];

/// Upserts (creating, moving, array-valued), set_payload path moves and
/// deletes, under the real name and an alias, with jitter.
fn mixed_writer(ctx: &TestContext, name: &str, alias: &str, stop: &AtomicBool, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    while !stop.load(AtomicOrdering::SeqCst) {
        let target = if rng.random_bool(0.5) { name } else { alias };
        match rng.random_range(0..10) {
            0..=4 => {
                let points: Vec<_> = (0..rng.random_range(1..=20))
                    .map(|_| {
                        let id = rng.random_range(1..=600u128);
                        let mut point = dir_point(id, DIRS[rng.random_range(0..5)]);
                        if rng.random_bool(0.1) {
                            let path = point["payload"]["metadata"]["path"].clone();
                            point["payload"]["metadata"]["path"] = json!([path]);
                        }
                        point
                    })
                    .collect();
                try_call(ctx, target, "upsert", json!({"points": points}));
            }
            5..=7 => {
                let ids: Vec<u64> = (0..rng.random_range(1..=30))
                    .map(|_| rng.random_range(1..=400))
                    .collect();
                let dir = DIRS[rng.random_range(0..5)];
                try_call(
                    ctx,
                    target,
                    "payload",
                    json!({"points": ids, "key": "metadata",
                           "payload": {"path": format!("/work/r/{dir}/sp.rs")}}),
                );
            }
            _ => {
                let ids: Vec<u64> = (0..rng.random_range(1..=3))
                    .map(|_| rng.random_range(1..=600))
                    .collect();
                try_call(ctx, target, "delete", json!({"points": ids}));
            }
        }
        std::thread::sleep(std::time::Duration::from_micros(rng.random_range(0..2_000)));
    }
}

fn wait_ready(storage: &HelixGraphStorage, field: &str, limit: std::time::Duration) -> f64 {
    let started = Instant::now();
    loop {
        let status = storage.payload_index_status(field).unwrap();
        match status.as_ref().map(|s| s.status) {
            Some("ready") => return started.elapsed().as_secs_f64(),
            Some("building") => assert!(
                started.elapsed() < limit,
                "{field} still building after {:?}",
                started.elapsed()
            ),
            other => panic!("{field}: {other:?} {:?}", status.and_then(|s| s.error)),
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

/// Three writers (upserts that create, move and array-wrap points;
/// set_payload path moves; deletes; half under an alias) run through
/// alternating async (handler) and sync builds of metadata.path, each
/// followed by a delete of the index. At every Ready the index holds no
/// stale, marker-mismatched or missing row, while the writers keep going.
/// The sync build (a test-only API that publishes its Building handle
/// without the point-mutation gate) is started under the gate, as
/// CreatePayloadIndex starts the async one.
#[test]
fn concurrent_writers_leave_no_violation_at_ready_on_either_build_path() {
    let ctx = lsm_ctx();
    create_dense_collection_lsm(&ctx, "mix");
    let storage = ctx.collections.get_collection("mix").unwrap();
    let points: Vec<_> = (1..=400u128)
        .map(|id| dir_point(id, DIRS[(id % 5) as usize]))
        .collect();
    call(&ctx, "mix", "upsert", json!({"points": points}));
    ctx.collections.create_alias("mix_alias", "mix").unwrap();
    let stop = AtomicBool::new(false);
    let mut failures = Vec::new();
    let mut ready_secs = Vec::new();
    std::thread::scope(|scope| {
        let (ctx, stop, storage) = (&ctx, &stop, &storage);
        let _stop = StopWriters(stop);
        for seed in 0..3u64 {
            scope.spawn(move || mixed_writer(ctx, "mix", "mix_alias", stop, seed));
        }
        for cycle in 0..12 {
            if cycle % 2 == 1 {
                let gate = ctx.collections.point_mutation_gate("mix");
                let held = gate.lock().unwrap();
                let builder = scope.spawn(move || {
                    storage.create_payload_index("metadata.path", PayloadIndexSchema::Keyword)
                });
                while storage
                    .payload_index_status("metadata.path")
                    .unwrap()
                    .is_none()
                {
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
                drop(held);
                builder.join().unwrap().unwrap();
            } else {
                call(
                    ctx,
                    "mix",
                    "create_index",
                    json!({"field_name": "metadata.path", "field_schema": "keyword"}),
                );
            }
            ready_secs.push(wait_ready(
                storage,
                "metadata.path",
                std::time::Duration::from_secs(60),
            ));
            let bad = index_violations(storage);
            if !bad.is_empty() {
                failures.push(format!("cycle {cycle}: {} e.g. {}", bad.len(), bad[0]));
            }
            call(ctx, "mix", "delete_index", json!({}));
        }
    });
    println!("ready after (s): {ready_secs:.2?}");
    assert!(failures.is_empty(), "{failures:#?}");
}

/// One point rewritten back to back (no pause) while its build-delta row
/// waits for the drain: the single-node retry must still commit between two
/// of its writes rather than wedge the build.
#[test]
fn one_hot_point_does_not_wedge_the_build_drain() {
    let ctx = lsm_ctx();
    create_dense_collection_lsm(&ctx, "hot1");
    let storage = ctx.collections.get_collection("hot1").unwrap();
    let points: Vec<_> = (1..=300u128).map(|id| dir_point(id, "d1")).collect();
    call(&ctx, "hot1", "upsert", json!({"points": points}));
    let stop = AtomicBool::new(false);
    let mut ready_secs = Vec::new();
    std::thread::scope(|scope| {
        let (ctx, stop, storage) = (&ctx, &stop, &storage);
        let _stop = StopWriters(stop);
        for writer in 0..2u64 {
            scope.spawn(move || {
                let mut i = writer;
                while !stop.load(AtomicOrdering::SeqCst) {
                    let dir = DIRS[(i % 5) as usize];
                    let body = if writer == 0 {
                        json!({"points": [dir_point(1, dir)]})
                    } else {
                        json!({"points": [1], "key": "metadata",
                               "payload": {"path": format!("/work/r/{dir}/f1.rs")}})
                    };
                    try_call(
                        ctx,
                        "hot1",
                        if writer == 0 { "upsert" } else { "payload" },
                        body,
                    );
                    i += 2;
                }
            });
        }
        for _ in 0..3 {
            call(
                ctx,
                "hot1",
                "create_index",
                json!({"field_name": "metadata.path", "field_schema": "keyword"}),
            );
            ready_secs.push(wait_ready(
                storage,
                "metadata.path",
                std::time::Duration::from_secs(30),
            ));
            assert_eq!(index_violations(storage), Vec::<String>::new());
            call(ctx, "hot1", "delete_index", json!({}));
        }
    });
    println!("ready after (s): {ready_secs:.2?}");
}

/// `commit_write` writes exactly what a plain commit of the same batch
/// writes (puts, deletes, dup rows, overwrites in either order), refuses a
/// batch carrying a merge, and is refused when a key it marked read was
/// written after it began.
#[test]
fn checked_commit_write_matches_a_plain_commit() {
    use crate::helix_engine::storage_core::backend::StorageBackend;
    use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
    let fill = |lsm: &LsmBackend, w: &mut <LsmBackend as StorageBackend>::Write<'_>| {
        let ns = Namespace::Metadata;
        let dup = Namespace::PayloadIndex("review_dup");
        lsm.put(w, ns, b"a", b"1").unwrap();
        lsm.put(w, ns, b"b", b"1").unwrap();
        lsm.delete(w, ns, b"b").unwrap();
        lsm.delete(w, ns, b"c").unwrap();
        lsm.put(w, ns, b"c", b"2").unwrap();
        lsm.delete(w, ns, b"seed_gone").unwrap();
        lsm.put(w, ns, b"seed_kept", b"new").unwrap();
        lsm.put_dup(w, dup, b"k", &1u128.to_be_bytes()).unwrap();
        lsm.put_dup_stored(w, dup, b"k", &2u128.to_be_bytes(), &[1])
            .unwrap();
        lsm.delete_dup(w, dup, b"k", &3u128.to_be_bytes()).unwrap();
        lsm.put_dup_stored(w, dup, b"j", &4u128.to_be_bytes(), &[1])
            .unwrap();
        lsm.delete_dup(w, dup, b"j", &4u128.to_be_bytes()).unwrap();
    };
    let dump = |lsm: &LsmBackend| {
        let r = lsm.begin_read().unwrap();
        let mut rows = Vec::new();
        lsm.scan(&r, Namespace::Metadata, KeyRange::all(), |k, v| {
            rows.push((k.to_vec(), v.to_vec(), Vec::new()));
            true
        })
        .unwrap();
        lsm.scan_dup_stored(
            &r,
            Namespace::PayloadIndex("review_dup"),
            KeyRange::all(),
            |k, v, s| {
                rows.push((k.to_vec(), v.to_vec(), s.to_vec()));
                true
            },
        )
        .unwrap();
        rows
    };
    let seed = |lsm: &LsmBackend| {
        let mut w = lsm.begin_write().unwrap();
        for key in [&b"seed_gone"[..], b"seed_kept", b"b"] {
            lsm.put(&mut w, Namespace::Metadata, key, b"old").unwrap();
        }
        lsm.put_dup(
            &mut w,
            Namespace::PayloadIndex("review_dup"),
            b"k",
            &3u128.to_be_bytes(),
        )
        .unwrap();
        lsm.commit(w).unwrap();
    };
    let plain = LsmBackend::open_in_memory("review-plain-commit").unwrap();
    let checked = LsmBackend::open_in_memory("review-checked-commit").unwrap();
    seed(&plain);
    seed(&checked);
    let mut w = plain.begin_write().unwrap();
    fill(&plain, &mut w);
    plain.commit(w).unwrap();
    let txn = checked.begin_checked().unwrap();
    let mut w = checked.begin_write().unwrap();
    fill(&checked, &mut w);
    assert!(txn.commit_write(w).unwrap());
    assert_eq!(dump(&checked), dump(&plain));

    let txn = checked.begin_checked().unwrap();
    let mut w = checked.begin_write().unwrap();
    checked
        .put(&mut w, Namespace::Metadata, b"m", b"1")
        .unwrap();
    checked
        .merge(&mut w, Namespace::Metadata, b"ctr", &[0; 8])
        .unwrap();
    assert!(txn.commit_write(w).is_err(), "a merge must be refused");

    let before = dump(&checked);
    let txn = checked.begin_checked().unwrap();
    txn.mark_read(Namespace::Metadata, b"a").unwrap();
    let mut other = checked.begin_write().unwrap();
    checked
        .put(&mut other, Namespace::Metadata, b"a", b"raced")
        .unwrap();
    checked.commit(other).unwrap();
    let mut w = checked.begin_write().unwrap();
    checked
        .put(&mut w, Namespace::Metadata, b"z", b"1")
        .unwrap();
    assert!(
        !txn.commit_write(w).unwrap(),
        "a write after begin must refuse it"
    );
    let after: Vec<_> = dump(&checked)
        .into_iter()
        .filter(|(k, _, _)| k.as_slice() != b"a")
        .collect();
    let before: Vec<_> = before
        .into_iter()
        .filter(|(k, _, _)| k.as_slice() != b"a")
        .collect();
    assert_eq!(after, before, "a refused commit wrote nothing");
}

/// The client re-sends creates for indexes that exist: each is a no-op (same job,
/// same rows), and none waits for the admin gate an async rebuild holds
/// while it clears a large dropped index's leftover rows (the create holds
/// the point gate meanwhile, so an upsert would wait behind it).
#[test]
fn existing_index_creates_are_no_ops_that_skip_the_leftover_clear() {
    let ctx = lsm_ctx();
    create_dense_collection_lsm(&ctx, "big_rebuild");
    let storage = ctx.collections.get_collection("big_rebuild").unwrap();
    let points: Vec<_> = (1..=200u128).map(|id| dir_point(id, "d1")).collect();
    call(&ctx, "big_rebuild", "upsert", json!({"points": points}));
    let create = |field: &str| {
        call(
            &ctx,
            "big_rebuild",
            "create_index",
            json!({"field_name": field, "field_schema": "keyword"}),
        )
    };
    for field in ["metadata.path", "metadata.repo"] {
        create(field);
        wait_ready(&storage, field, std::time::Duration::from_secs(30));
    }
    let job = |field: &str| storage.payload_index_status(field).unwrap().unwrap().job_id;
    let db =
        HelixGraphStorage::payload_index_db_name("metadata.path", &PayloadIndexSchema::Keyword);
    let rows = || {
        let read = storage.backend.begin_read().unwrap();
        let mut rows = Vec::new();
        storage
            .backend
            .scan_dup_stored(
                &read,
                Namespace::PayloadIndex(&db),
                KeyRange::all(),
                |k, v, s| {
                    rows.push((k.to_vec(), v.to_vec(), s.to_vec()));
                    true
                },
            )
            .unwrap();
        rows
    };
    let (job_before, rows_before) = (job("metadata.path"), rows());
    for _ in 0..25 {
        create("metadata.path");
        create("metadata.repo");
    }
    assert_eq!(job("metadata.path"), job_before);
    assert_eq!(rows(), rows_before);

    // A dropped big index's leftover rows (written directly).
    call(&ctx, "big_rebuild", "delete_index", json!({}));
    let mut w = storage.backend.begin_write().unwrap();
    for i in 0..120_000u128 {
        let key = HelixGraphStorage::stable_index_key_for_value(&Value::String(format!(
            "/work/r/old/f{}.rs",
            i % 5_000
        )))
        .unwrap();
        storage
            .backend
            .put_dup_stored(
                &mut w,
                Namespace::PayloadIndex(&db),
                &key,
                &(1_000 + i).to_be_bytes(),
                &[1],
            )
            .unwrap();
    }
    storage.backend.commit(w).unwrap();
    let started = Instant::now();
    create("metadata.path");
    std::thread::sleep(std::time::Duration::from_millis(30));
    let waits = std::thread::scope(|scope| {
        let noop = scope.spawn(|| {
            let t = Instant::now();
            create("metadata.repo");
            t.elapsed()
        });
        std::thread::sleep(std::time::Duration::from_millis(5));
        let t = Instant::now();
        call(
            &ctx,
            "big_rebuild",
            "upsert",
            json!({"points": [dir_point(1, "d2")]}),
        );
        (noop.join().unwrap(), t.elapsed())
    });
    let ready = wait_ready(
        &storage,
        "metadata.path",
        std::time::Duration::from_secs(300),
    );
    let rebuild = started.elapsed().as_secs_f64().max(ready);
    println!(
        "leftover clear: rebuild ready after {rebuild:.2}s; no-op create waited {:?}, upsert waited {:?}",
        waits.0, waits.1
    );
    assert!(
        rebuild > 2.0,
        "the clear must outlast the waits it is compared with"
    );
    let limit = std::time::Duration::from_secs(1);
    assert!(
        waits.0 < limit && waits.1 < limit,
        "{waits:?} behind a {rebuild:.1}s clear"
    );
    assert_eq!(index_violations(&storage), Vec::<String>::new());
}

/// The walk `get_nodes_by_payload_text_exact_be` must return for `r`,
/// recomputed from the rows `r` sees: lowercase substring over decoded keys
/// (array elements included), exact when every hit is a marked string row.
fn uncached_walk(
    storage: &HelixGraphStorage,
    r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    field: &str,
    needle: &str,
) -> Option<(Vec<u128>, bool)> {
    fn matches(value: &Value, needle: &str) -> bool {
        match value {
            Value::String(s) => s.to_lowercase().contains(needle),
            Value::Array(values) => values.iter().any(|v| matches(v, needle)),
            _ => false,
        }
    }
    let needle = needle.to_lowercase();
    let db = HelixGraphStorage::payload_index_db_name(field, &PayloadIndexSchema::Keyword);
    let (mut ids, mut exact, mut undecodable) = (Vec::new(), true, false);
    storage
        .backend
        .scan_dup_stored(
            r,
            Namespace::PayloadIndex(&db),
            KeyRange::all(),
            |key, id, stored| {
                match bincode::deserialize::<Value>(key) {
                    Ok(value) if matches(&value, &needle) => {
                        exact &= matches!(value, Value::String(_)) && stored == [1u8];
                        ids.push(u128::from_be_bytes(<[u8; 16]>::try_from(id).unwrap()));
                    }
                    Ok(_) => {}
                    Err(_) => {
                        undecodable = true;
                        return false;
                    }
                }
                true
            },
        )
        .unwrap();
    ids.sort_unstable();
    ids.dedup();
    (!undecodable).then_some((ids, exact))
}

const NEEDLES: [&str; 7] = ["/d1/", "D2", "/work/", "sp.rs", "f1", "/r/d3/f3", "zz-none"];

/// Writers (creating, moving, array-wrapping upserts; set_payload moves;
/// deletes; half via an alias) and the marker backfill (flipping rows from
/// unmarked to marked, so exactness changes under cached walks) run while
/// every walk of each fresh snapshot, first (miss or cross-snapshot hit) and
/// repeated (hit), equals the walk recomputed from that snapshot's rows.
#[test]
fn cached_text_walks_equal_a_recomputed_walk_of_the_same_snapshot_under_writes() {
    let ctx = lsm_ctx();
    create_dense_collection_lsm(&ctx, "twc");
    let storage = ctx.collections.get_collection("twc").unwrap();
    storage
        .create_payload_index("metadata.path", PayloadIndexSchema::Keyword)
        .unwrap();
    let points: Vec<_> = (1..=400u128)
        .map(|id| dir_point(id, DIRS[(id % 5) as usize]))
        .collect();
    call(&ctx, "twc", "upsert", json!({"points": points}));
    ctx.collections.create_alias("twc_alias", "twc").unwrap();
    storage.strip_payload_markers_for_test("metadata.path");
    let stop = AtomicBool::new(false);
    let mut mismatches = Vec::new();
    let mut checked = 0usize;
    let (_, hits_before) = storage.text_walk_cache_stats();
    std::thread::scope(|scope| {
        let (ctx, stop, storage) = (&ctx, &stop, &storage);
        let _stop = StopWriters(stop);
        for seed in 0..3u64 {
            scope.spawn(move || mixed_writer(ctx, "twc", "twc_alias", stop, 40 + seed));
        }
        scope.spawn(move || {
            while !stop.load(AtomicOrdering::SeqCst) {
                HelixGraphStorage::run_payload_marker_backfill(&std::sync::Arc::downgrade(storage))
                    .unwrap();
                std::thread::sleep(std::time::Duration::from_millis(20));
                storage.strip_payload_markers_for_test("metadata.path");
            }
        });
        let started = Instant::now();
        while started.elapsed() < std::time::Duration::from_secs(8) {
            let r = storage.backend.begin_read().unwrap();
            for needle in NEEDLES {
                let first = storage
                    .get_nodes_by_payload_text_exact_be(&r, "metadata.path", needle, 0)
                    .unwrap();
                let again = storage
                    .get_nodes_by_payload_text_exact_be(&r, "metadata.path", needle, 0)
                    .unwrap();
                let truth = uncached_walk(storage, &r, "metadata.path", needle);
                checked += 1;
                if first != truth || again != truth {
                    mismatches.push(format!(
                        "seq {:?} {needle}: first {:?} again {:?} truth {:?}",
                        r.snapshot_seq(),
                        first.map(|(ids, exact)| (ids.len(), exact)),
                        again.map(|(ids, exact)| (ids.len(), exact)),
                        truth.map(|(ids, exact)| (ids.len(), exact)),
                    ));
                }
            }
        }
    });
    let (_, hits) = storage.text_walk_cache_stats();
    println!("checked {checked} walks, {} cache hits", hits - hits_before);
    assert!(
        hits - hits_before as u64 > checked as u64 / 2,
        "the cache was exercised"
    );
    assert!(
        mismatches.is_empty(),
        "{:#?}",
        &mismatches[..mismatches.len().min(10)]
    );
}

fn create_hybrid_collection(ctx: &TestContext, name: &str) {
    let input = make_input(
        ctx,
        "PUT",
        &format!("/collections/{name}"),
        serde_json::to_vec(&json!({
            "vectors": {"dense": {"size": 3, "distance": "Cosine"}},
            "sparse_vectors": {"lex_sparse": {"modifier": "idf"}}
        }))
        .unwrap(),
        HashMap::from([("name".into(), name.into())]),
    );
    let mut response = Response::new();
    handle_create_collection(&input, &mut response).unwrap();
    assert!(response.status == 200 || response.status == 201);
}

/// Every endpoint (dense query and search, sparse query, RRF fusion,
/// hybrid with channel filters, count, scroll) answers each filter shape
/// identically from a fresh walk (a commit just moved the sequence: miss)
/// and from the cache (no commit since: hit). Two identically loaded
/// collections are not compared: their dense answers to broad filters
/// already differ with the cache off (approximate search per collection).
#[test]
#[serial_test::serial]
fn endpoints_answer_identically_from_a_fresh_and_a_cached_walk() {
    let ctx = lsm_ctx();
    create_hybrid_collection(&ctx, "on");
    let points: Vec<_> = (1..=600u128)
        .map(|id| {
            let mut point = dir_point(id, DIRS[(id % 5) as usize]);
            if id % 37 == 0 {
                let path = point["payload"]["metadata"]["path"].clone();
                point["payload"]["metadata"]["path"] = json!([path]);
            }
            point["vector"]["lex_sparse"] = json!({
                "indices": [(id % 13) as u32, 100 + (id % 7) as u32, 200 + (id % 5) as u32],
                "values": [1.0, 0.5, 0.25]});
            point["payload"]["branch"] = json!(if id % 9 == 0 { "private" } else { "public" });
            point
        })
        .collect();
    for name in ["on"] {
        call(&ctx, name, "upsert", json!({"points": points}));
        let storage = ctx.collections.get_collection(name).unwrap();
        for field in ["metadata.path", "metadata.repo", "branch"] {
            storage
                .create_payload_index(field, PayloadIndexSchema::Keyword)
                .unwrap();
        }
    }
    let mut shapes = Vec::new();
    for needle in ["/d1/", "D2/", "/work/", "f1", "zz-none"] {
        shapes.extend([
            json!({"must": [text("metadata.path", needle)]}),
            json!({"must": [text("metadata.repo", "r1"), text("metadata.path", needle)]}),
            json!({"must": [text("metadata.path", "/work/")], "must_not": [text("metadata.path", needle)]}),
            json!({"should": [text("metadata.path", needle), text("metadata.path", "/d3/")]}),
            json!({"must": [{"should": [text("metadata.path", needle)]}],
                   "must_not": [{"key": "branch", "match": {"value": "private"}}]}),
        ]);
    }
    let sparse = json!({"indices": [3, 102, 201], "values": [1.0, 1.0, 1.0]});
    let v = [0.3f32, -0.8, 0.5];
    let w = [-0.6f32, 0.2, 0.9];
    let storage = ctx.collections.get_collection("on").unwrap();
    let (mut compared, mut fresh_without_hits) = (0usize, 0usize);
    for filter in &shapes {
        let requests = [
            (
                "query",
                json!({"query": v, "using": "dense", "limit": 20, "filter": filter}),
            ),
            (
                "search",
                json!({"vector": {"name": "dense", "vector": v}, "limit": 20, "filter": filter}),
            ),
            (
                "query",
                json!({"query": sparse, "using": "lex_sparse", "limit": 20, "filter": filter}),
            ),
            (
                "query",
                json!({"prefetch": [
                {"query": v, "using": "dense", "limit": 30},
                {"query": w, "using": "dense", "limit": 30}],
                "query": {"fusion": "rrf"}, "limit": 20, "filter": filter}),
            ),
            (
                "hybrid_query",
                json!({"dense": [
                {"query": v, "using": "dense", "limit": 20},
                {"query": w, "using": "dense", "limit": 20, "filter": {"must": [text("metadata.path", "/d4/")]}}],
                "sparse": [{"query": sparse, "using": "lex_sparse", "limit": 20}],
                "limit": 20, "filter": filter, "with_channel_points": true}),
            ),
            ("count", json!({"filter": filter, "exact": true})),
            (
                "scroll",
                json!({"filter": filter, "limit": 50, "with_payload": false}),
            ),
        ];
        for (route, body) in requests {
            // A commit that touches no index row moves the snapshot sequence.
            let mut w = storage.backend.begin_write().unwrap();
            storage
                .backend
                .put(
                    &mut w,
                    Namespace::Metadata,
                    b"review_seq_bump",
                    &compared.to_be_bytes(),
                )
                .unwrap();
            storage.backend.commit(w).unwrap();
            let (_, hits) = storage.text_walk_cache_stats();
            let fresh = call(&ctx, "on", route, body.clone());
            let walked_fresh = storage.text_walk_cache_stats().1 == hits;
            let cached = call(&ctx, "on", route, body.clone());
            assert_eq!(cached, fresh, "{route} for {filter}");
            // `must_not` text is walked too: excluding the scope it walked,
            // a request reuses its own walk of the same snapshot.
            let own_walk = !filter["must_not"][0]["match"]["text"].is_null()
                && filter["must_not"][0]["match"]["text"] == filter["must"][0]["match"]["text"];
            fresh_without_hits += usize::from(walked_fresh || own_walk);
            compared += 1;
        }
    }
    println!(
        "compared {compared}; {fresh_without_hits} first calls walked without a hit; cache {:?}",
        storage.text_walk_cache_stats()
    );
    assert_eq!(
        fresh_without_hits, compared,
        "a sequence bump must force a fresh walk"
    );
    assert!(storage.text_walk_cache_stats().1 > 0);
}

/// A filter delete that matches nothing is not a mutation: the collection's
/// write count stays put (regraph stands down when it moves) and nothing
/// queues on the mutation gate. A matching delete still removes exactly its
/// points, resolved by scan and then through the indexed recheck.
#[test]
fn filter_delete_that_matches_nothing_is_not_a_mutation() {
    let ctx = lsm_ctx();
    let name = "noop_delete";
    create_dense_collection_lsm(&ctx, name);
    let storage = ctx.collections.get_collection(name).unwrap();
    let delete = |path: &str| {
        let filter = json!({"must": [{"key": "metadata.path", "match": {"value": path}}]});
        call(&ctx, name, "delete", json!({"filter": filter}));
    };
    let count = || call(&ctx, name, "count", json!({"exact": true}))["count"].as_u64();
    let writes = || storage.point_mutations.load(AtomicOrdering::Acquire);

    for indexed in [false, true] {
        if indexed {
            call(
                &ctx,
                name,
                "create_index",
                json!({"field_name": "metadata.path", "field_schema": "keyword"}),
            );
            wait_ready(
                &storage,
                "metadata.path",
                std::time::Duration::from_secs(60),
            );
        }
        call(
            &ctx,
            name,
            "upsert",
            json!({"points": [
                {"id": 1, "vector": {"dense": [1.0, 0.0, 0.0]}, "payload": {"metadata": {"path": "a.rs"}}},
                {"id": 2, "vector": {"dense": [0.0, 1.0, 0.0]}, "payload": {"metadata": {"path": "b.rs"}}},
            ]}),
        );
        let before = writes();

        delete("missing.rs");
        assert_eq!(
            writes(),
            before,
            "indexed={indexed}: an empty delete counted"
        );
        assert_eq!(count(), Some(2), "indexed={indexed}");

        delete("a.rs");
        assert!(
            writes() > before,
            "indexed={indexed}: a real delete must count"
        );
        assert_eq!(count(), Some(1), "indexed={indexed}: only a.rs is deleted");
    }
}
