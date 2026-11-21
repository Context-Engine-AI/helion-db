//! What the client writes new on a chunk it moved (`document`, `information`, where
//! the chunk's symbol sits, the import and inheritance maps of its file) is
//! left out of the content hash: an upsert that changes nothing else writes
//! the node and its payload-index rows, and no vector. Each test makes the
//! same writes to a second collection that writes every point in full
//! (`HELIX_CONTENT_HASH_SKIP=0`), and the two must read alike.

use super::hybrid_parallel_tests::EnvVar;
use super::tests::{create_collection_with_sparse, make_input, setup_with_config, TestContext};
use super::*;
use crate::helix_engine::graph_core::config::Config;
use crate::helix_engine::storage_core::replication::test_support::stamp_of_the_first_lists;
use serde_json::json;
use std::collections::BTreeMap;

/// The collection under test.
const MOVED: &str = "moved";
/// The one that writes every point in full.
const FULL: &str = "full";

/// Both collections on the LSM backend, searched by exact scan, with a
/// payload index on two of the keys a moved chunk changes.
fn lsm_ctx() -> TestContext {
    let mut config = Config::default().with_lsm_in_memory();
    config.vector_config.flat_scan_threshold = Some(1_000_000);
    let ctx = setup_with_config(config);
    for name in [MOVED, FULL] {
        create_collection_with_sparse(&ctx, name);
        let storage = ctx.collections.get_collection(name).unwrap();
        assert!(storage.backend.kind() == BackendKind::Lsm);
        for (field, schema) in [
            ("information", "keyword"),
            ("metadata.symbol_start_line", "integer"),
        ] {
            let index = json!({"field_name": field, "field_schema": schema});
            call(&ctx, name, "create_index", index);
            // The build is another thread's: every write of a test meets the index.
            let deadline = Instant::now() + std::time::Duration::from_secs(5);
            while storage.has_payload_index(field).is_none() {
                assert!(Instant::now() < deadline, "no index on {field} of {name}");
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
    }
    ctx
}

/// One handler call; `result` of its 200 response.
fn call(ctx: &TestContext, name: &str, route: &str, body: serde_json::Value) -> serde_json::Value {
    let (method, path) = match route {
        "upsert" => ("PUT", format!("/collections/{name}/points")),
        "get" => ("POST", format!("/collections/{name}/points")),
        "create_index" => ("PUT", format!("/collections/{name}/index")),
        other => ("POST", format!("/collections/{name}/points/{other}")),
    };
    let input = make_input(
        ctx,
        method,
        &path,
        serde_json::to_vec(&body).unwrap(),
        HashMap::from([("name".to_string(), name.to_string())]),
    );
    let mut response = Response::new();
    match route {
        "upsert" => handle_upsert_points(&input, &mut response),
        "payload" => handle_set_payload(&input, &mut response),
        "get" => handle_get_points(&input, &mut response),
        "scroll" => handle_scroll_points(&input, &mut response),
        "query" => handle_query_points(&input, &mut response),
        "create_index" => handle_create_index(&input, &mut response),
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

/// The display line of chunk `id` when it sits at `line`.
fn information(id: u64, line: u64) -> String {
    format!("fn f{id} in src/lib.rs, lines {line} to {}", line + 2)
}

/// A chunk as the client writes it. `line` is where it sits in its file: it moves
/// every key the second lists leave out, and `start_line`, which the first
/// lists left out already. `calls` is a key the hash covers, and the rest of
/// what it covers has the kinds of value a client payload holds: a null, a
/// float, a bool, an empty array and nested objects.
fn chunk(id: u64, line: u64, calls: &str, dense: [f32; 3]) -> serde_json::Value {
    json!({
        "id": id,
        "vector": {
            "dense": dense,
            "lex_sparse": {"indices": [id, 100], "values": [1.0, 1.0 / id as f32]},
        },
        "payload": {
            "document": information(id, line),
            "information": information(id, line),
            "start_line": line,
            "pid_str": null,
            "metadata": {
                "path": "src/lib.rs",
                "calls": [calls],
                "imports": [],
                "parent_symbol": null,
                "is_test": false,
                "weight": 0.5,
                "symbol": {"name": format!("f{id}"), "scope": {"depth": 2, "generic": null}},
                "symbol_start_line": line,
                "symbol_end_line": line + 2,
                "import_map": {"a": format!("crate::a{line}")},
                "inheritance_map": {"T": [format!("U{line}")]},
            },
        },
    })
}

/// `points` written to both collections. How the upsert counters of the one
/// under test moved: `[skipped, node only, in full for its payload, in full
/// with the payload the node had]`.
fn write(ctx: &TestContext, points: serde_json::Value) -> [u64; 4] {
    {
        let _full = EnvVar::set("HELIX_CONTENT_HASH_SKIP", "0");
        call(ctx, FULL, "upsert", json!({ "points": points }));
    }
    let _skip = EnvVar::set("HELIX_CONTENT_HASH_SKIP", "1");
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        call(ctx, MOVED, "upsert", json!({ "points": points }));
    });
    let rendered = handle.render();
    [
        "skipped_total",
        "payload_only_update_total",
        r#"mismatch_reason_total{reason="payload_changed"}"#,
        r#"mismatch_reason_total{reason="payload_same"}"#,
    ]
    .map(|counter| {
        let counter = format!("helix_upsert_content_hash_{counter} ");
        rendered
            .lines()
            .find_map(|line| line.strip_prefix(&counter)?.parse().ok())
            .unwrap_or(0)
    })
}

/// `key` of the payload each dense row of `id` keeps, which is the payload
/// of the write that made the row.
fn rows_keep(ctx: &TestContext, name: &str, id: u64, key: &str) -> Vec<Option<Value>> {
    let storage = ctx.collections.get_collection(name).unwrap();
    let spaces = storage.named_vectors.list_dense_vector_spaces();
    let read = storage.backend.begin_read().unwrap();
    let cores = storage.named_vectors.cores_read().unwrap();
    let mut kept = Vec::new();
    for segment in &spaces["dense"].segments {
        let Some(core) = cores.get(&segment.physical_name) else {
            continue;
        };
        for (row, _, fields) in core.export_level_zero(&read).unwrap() {
            if row == u128::from(id) {
                kept.push(HelixGraphStorage::payload_value_for_key(&fields, key).cloned());
            }
        }
    }
    kept
}

/// Filters on keys a moved chunk changes, by what they ask. `information`
/// has a keyword index and `document` has none.
fn filters() -> [(&'static str, serde_json::Value); 6] {
    let symbol =
        |range: serde_json::Value| json!({"key": "metadata.symbol_start_line", "range": range});
    let text = |key: &str, needle: &str| json!({"key": key, "match": {"text": needle}});
    [
        ("symbol from line 100", symbol(json!({"gte": 100}))),
        ("symbol before line 100", symbol(json!({"lt": 100}))),
        ("information at 110", text("information", "lines 110 to")),
        ("information at 10", text("information", "lines 10 to")),
        ("document at 110", text("document", "lines 110 to")),
        ("document at 10", text("document", "lines 10 to")),
    ]
}

/// What a reader gets from `name`, by what it asked: the by-id read and the
/// scroll with every vector, a dense and a sparse search, and each of
/// `filters` through a scroll and through a dense search. The stamped hash
/// is not a client's key and is left out.
fn answers(ctx: &TestContext, name: &str) -> BTreeMap<String, Vec<serde_json::Value>> {
    let whole = |mut body: serde_json::Value| {
        body["with_payload"] = json!(true);
        body["with_vector"] = json!(true);
        body
    };
    let dense = json!({"query": [1.0, 0.0, 0.0], "using": "dense", "limit": 10,
        "with_payload": true});
    let sparse = json!({"query": {"indices": [100], "values": [1.0]}, "using": "lex_sparse",
        "limit": 10, "with_payload": true});
    let mut asked = vec![
        (
            "by id".to_string(),
            "get",
            whole(json!({"ids": [1, 2, 3, 4]})),
        ),
        ("scroll".to_string(), "scroll", whole(json!({"limit": 10}))),
        ("dense".to_string(), "query", dense.clone()),
        ("sparse".to_string(), "query", sparse),
    ];
    for (what, condition) in filters() {
        let filter = json!({"must": [condition]});
        let scroll = json!({"limit": 10, "with_payload": true, "filter": filter});
        asked.push((format!("scroll, {what}"), "scroll", scroll));
        let mut search = dense.clone();
        search["filter"] = filter;
        asked.push((format!("dense, {what}"), "query", search));
    }
    asked
        .into_iter()
        .map(|(what, route, body)| {
            let result = call(ctx, name, route, body);
            let mut points = result
                .get("points")
                .unwrap_or(&result)
                .as_array()
                .unwrap_or_else(|| panic!("{what}: no points in {result}"))
                .clone();
            for point in &mut points {
                if let Some(payload) = point.get_mut("payload").and_then(|p| p.as_object_mut()) {
                    payload.remove("__helix_content_hash__");
                }
            }
            // A search is ranked; the other two are in no order a client is owed.
            if route != "query" {
                points.sort_by_key(|point| point_id_to_u128(&point["id"]));
            }
            (what, points)
        })
        .collect()
}

/// The ids of an answer, lowest first.
fn ids(points: &[serde_json::Value]) -> Vec<u128> {
    let mut ids: Vec<u128> = points.iter().map(|p| point_id_to_u128(&p["id"])).collect();
    ids.sort_unstable();
    ids
}

#[test]
#[serial_test::serial]
fn a_moved_chunk_is_a_node_write_and_reads_as_a_full_write() {
    let ctx = lsm_ctx();
    let first = json!([
        chunk(1, 10, "g", [1.0, 0.0, 0.0]),
        chunk(2, 20, "g", [0.9, 0.1, 0.0]),
        chunk(3, 30, "g", [0.7, 0.3, 0.0]),
        chunk(4, 40, "g", [0.5, 0.5, 0.0]),
    ]);
    assert_eq!(write(&ctx, first), [0, 0, 0, 0]);
    let before = answers(&ctx, MOVED);
    assert_eq!(before, answers(&ctx, FULL));
    assert_eq!(ids(&before["scroll, information at 10"]), [1]);

    // 1 moved; 2 is as it was; 3 moved and calls something else; 4 moved and
    // has another vector.
    let second = json!([
        chunk(1, 110, "g", [1.0, 0.0, 0.0]),
        chunk(2, 20, "g", [0.9, 0.1, 0.0]),
        chunk(3, 130, "h", [0.7, 0.3, 0.0]),
        chunk(4, 140, "g", [0.4, 0.6, 0.0]),
    ]);
    assert_eq!(write(&ctx, second), [1, 1, 1, 1]);
    // The node write made no dense row: the row of 1 is the one its first
    // write made, and it keeps that write's payload.
    let line = |id, line| vec![Some(Value::String(information(id, line)))];
    assert_eq!(rows_keep(&ctx, MOVED, 1, "information"), line(1, 10));
    assert_eq!(rows_keep(&ctx, MOVED, 3, "information"), line(3, 130));
    assert_eq!(rows_keep(&ctx, MOVED, 4, "information"), line(4, 140));
    assert_eq!(rows_keep(&ctx, FULL, 1, "information"), line(1, 110));

    let after = answers(&ctx, MOVED);
    assert_eq!(after, answers(&ctx, FULL));
    // A filter and a text match on a key the node write changed answer from
    // the new value, through the payload index and without one.
    let mut wrong = Vec::new();
    for (what, _) in filters() {
        let expected: &[u128] = match what {
            "symbol from line 100" => &[1, 3, 4],
            "symbol before line 100" => &[2],
            "information at 110" | "document at 110" => &[1],
            _ => &[],
        };
        for route in ["scroll", "dense"] {
            let asked = format!("{route}, {what}");
            let got = ids(&after[&asked]);
            if got != expected {
                wrong.push(format!("{asked}: {got:?}, not {expected:?}"));
            }
        }
    }
    assert!(wrong.is_empty(), "assertion on the filters: {wrong:?}");
    assert_eq!(ids(&after["by id"]), [1, 2, 3, 4]);
    assert_eq!(ids(&after["sparse"]), [1, 2, 3, 4]);
}

/// `set_payload` writes the node and leaves the stamped hash as it is. On a
/// key the hash covers, the stamp is then that of another payload: the next
/// upsert of the point is written in full even when it brings the payload
/// the node has, because the stamp no longer says what the rows were written
/// with. On a key the hash leaves out, as the markers the client's graph repair
/// sets, the stamp holds and the next upsert writes no vector.
#[test]
#[serial_test::serial]
fn a_set_payload_on_a_covered_key_makes_the_next_upsert_a_full_write() {
    let ctx = lsm_ctx();
    let vector = [1.0, 0.0, 0.0];
    assert_eq!(
        write(&ctx, json!([chunk(1, 10, "g", vector)])),
        [0, 0, 0, 0]
    );
    let patch = |payload: serde_json::Value| {
        for name in [MOVED, FULL] {
            let body = json!({"payload": payload, "key": "metadata", "points": [1]});
            call(&ctx, name, "payload", body);
        }
    };
    let calls = |callee: &str| vec![Some(Value::Array(vec![Value::String(callee.into())]))];

    patch(json!({"calls": ["h"]}));
    assert_eq!(rows_keep(&ctx, MOVED, 1, "metadata.calls"), calls("g"));
    assert_eq!(
        write(&ctx, json!([chunk(1, 10, "h", vector)])),
        [0, 0, 0, 1]
    );
    assert_eq!(rows_keep(&ctx, MOVED, 1, "metadata.calls"), calls("h"));
    assert_eq!(
        write(&ctx, json!([chunk(1, 10, "h", vector)])),
        [1, 0, 0, 0]
    );

    patch(json!({"_graph_repair_backfilled": true, "_graph_repair_backfilled_version": 3}));
    assert_eq!(
        write(&ctx, json!([chunk(1, 110, "h", vector)])),
        [0, 1, 0, 0]
    );
    assert_eq!(
        rows_keep(&ctx, MOVED, 1, "information"),
        [Some(Value::String(information(1, 10)))]
    );
    assert_eq!(answers(&ctx, MOVED), answers(&ctx, FULL));
}

/// Stamps the stored `point` as an image that had only the first lists left
/// it: `set_payload` writes the node and no row, and stores a non-negative
/// integer as the `U64` a stamp is.
fn stamp_as_the_first_lists_did(ctx: &TestContext, point: &serde_json::Value) {
    let input: PointInput = serde_json::from_value(point.clone()).unwrap();
    let (vectors, sparse_vectors) = split_vectors(&input.vector).unwrap();
    let stamp = stamp_of_the_first_lists(&ReplicatedPoint {
        id: point_id_to_u128(&input.id),
        vectors,
        sparse_vectors,
        payload: json_payload_to_value_map(&input.payload),
    });
    let body = json!({"payload": {"__helix_content_hash__": stamp}, "points": [input.id]});
    call(ctx, MOVED, "payload", body);
}

/// Every point an image without the second lists wrote has a stamp of the
/// first. Such a stamp is told, so the first upsert on this image writes a
/// point's vectors only where they or a covered key changed: a chunk that
/// moved and one that did not change get their node written, with today's
/// stamp, and are skipped from then on. A first stamp says nothing where the
/// vectors differ, or where a `set_payload` changed a covered key after it.
#[test]
#[serial_test::serial]
fn a_stamp_of_the_first_lists_costs_a_moved_chunk_a_node_write() {
    let ctx = lsm_ctx();
    let vectors = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.7, 0.3, 0.0],
        [0.5, 0.5, 0.0],
        [0.3, 0.7, 0.0],
    ];
    let first: Vec<serde_json::Value> = (1..=5)
        .map(|id| chunk(id, id * 10, "g", vectors[id as usize - 1]))
        .collect();
    assert_eq!(write(&ctx, json!(first)), [0, 0, 0, 0]);
    for point in &first {
        stamp_as_the_first_lists_did(&ctx, point);
    }
    // 5 got another callee from a `set_payload` after its stamp.
    for name in [MOVED, FULL] {
        let body = json!({"payload": {"calls": ["h"]}, "key": "metadata", "points": [5]});
        call(&ctx, name, "payload", body);
    }

    // 1 moved; 2 is as it was; 3 moved and calls something else; 4 moved and
    // has another vector; 5 comes with the payload its node has.
    let second = json!([
        chunk(1, 110, "g", vectors[0]),
        chunk(2, 20, "g", vectors[1]),
        chunk(3, 130, "h", vectors[2]),
        chunk(4, 140, "g", [0.4, 0.6, 0.0]),
        chunk(5, 50, "h", vectors[4]),
    ]);
    assert_eq!(write(&ctx, second.clone()), [0, 2, 1, 2]);
    let line = |id, line| vec![Some(Value::String(information(id, line)))];
    assert_eq!(rows_keep(&ctx, MOVED, 1, "information"), line(1, 10));
    assert_eq!(rows_keep(&ctx, MOVED, 2, "information"), line(2, 20));
    assert_eq!(rows_keep(&ctx, MOVED, 3, "information"), line(3, 130));
    assert_eq!(rows_keep(&ctx, MOVED, 4, "information"), line(4, 140));
    let after = answers(&ctx, MOVED);
    assert_eq!(after, answers(&ctx, FULL));
    assert_eq!(ids(&after["scroll, information at 110"]), [1]);
    assert_eq!(ids(&after["scroll, symbol from line 100"]), [1, 3, 4]);

    // Each has today's stamp now.
    assert_eq!(write(&ctx, second), [5, 0, 0, 0]);
    assert_eq!(answers(&ctx, MOVED), answers(&ctx, FULL));
}

/// One id twice in a chunk. The plan of which stored rows the chunk removes
/// judges both occurrences against the node as it was stored; the write of
/// the second judges it against the node the first one left, so the two
/// places need not say the same of an occurrence. They agree on the point:
/// its older row is removed exactly when an occurrence appends one, there is
/// one row after, and what is stored is the last occurrence, under a stamp
/// of today's lists and under one of the first.
#[test]
#[serial_test::serial]
fn one_id_twice_in_a_chunk_is_stored_as_its_last_occurrence() {
    // The vector a point is stored with and another one for it. No two of
    // them are equally near the query `answers` ranks by.
    let stored = |id: u64| [1.0, 0.1 * id as f32, 0.0];
    let other = |id: u64| [0.1 * id as f32, 1.0, 0.0];
    for stamped_by_the_first_lists in [false, true] {
        let ctx = lsm_ctx();
        let first: Vec<serde_json::Value> = (1..=4)
            .map(|id| chunk(id, id * 10, "g", stored(id)))
            .collect();
        assert_eq!(write(&ctx, json!(first)), [0, 0, 0, 0]);
        if stamped_by_the_first_lists {
            for point in &first {
                stamp_as_the_first_lists_did(&ctx, point);
            }
        }
        // (id, whether each occurrence comes with another vector, the
        // upsert's counts, the line of the write that made the row there is
        // after).
        for (id, another, counts, row) in [
            // Moved, and moved again: two node writes, the first row stays.
            (1, [false, false], [0, 2, 0, 0], 10),
            // Another vector, then the stored one: the plan removes the row
            // for the first, and both append.
            (2, [true, false], [0, 0, 0, 1], 220),
            // The stored vector, then another: a node write and a full one.
            (3, [false, true], [0, 1, 0, 1], 230),
            // Another vector twice: the second is a node write over the row
            // the first appended in this chunk.
            (4, [true, true], [0, 1, 0, 2], 140),
        ] {
            let what = format!("id {id}, first-list stamp {stamped_by_the_first_lists}");
            let vectors = another.map(|another| if another { other(id) } else { stored(id) });
            let twice = json!([
                chunk(id, id * 10 + 100, "g", vectors[0]),
                chunk(id, id * 10 + 200, "g", vectors[1]),
            ]);
            assert_eq!(write(&ctx, twice), counts, "{what}");
            assert_eq!(
                rows_keep(&ctx, MOVED, id, "information"),
                [Some(Value::String(information(id, row)))],
                "{what}"
            );
            let read = call(
                &ctx,
                MOVED,
                "get",
                json!({"ids": [id], "with_payload": true, "with_vector": true}),
            );
            assert_eq!(
                read[0]["payload"]["information"],
                json!(information(id, id * 10 + 200)),
                "{what}"
            );
            let vector: Vec<f32> = serde_json::from_value(read[0]["vector"]["dense"].clone())
                .unwrap_or_else(|error| panic!("{what}: {error} in {}", read[0]));
            // The same direction: the space is cosine.
            let last = vectors[1];
            let dot: f32 = vector.iter().zip(last).map(|(a, b)| a * b).sum();
            let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                vector.len() == 3 && dot / (norm(&vector) * norm(&last)) > 0.99999,
                "assertion: {what}: the vector read is {vector:?}, not {last:?}"
            );
        }
        assert_eq!(
            answers(&ctx, MOVED),
            answers(&ctx, FULL),
            "first-list stamp {stamped_by_the_first_lists}"
        );
    }
}
