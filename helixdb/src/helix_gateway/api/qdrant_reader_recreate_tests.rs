//! A collection dropped and created again under a reader replica, asked
//! through every read route. Once the reader has met the new collection its
//! `DbReader` is closed, and the first request after that has to be answered
//! from the new collection: no route refuses, and none answers from what the
//! old storage object holds in memory.
//!
//! Before, and until the idle sweep evicted the storage (30 s in
//! production), searches answered 500, scroll and filtered counts 404, a
//! plain count and the collection's info 200 with the dropped collection's
//! numbers; and a reopen of the closed reader brought it back for good on
//! the old storage object, where facet answered from the dropped
//! collection's index handle.

use super::hybrid_parallel_tests::EnvVar;
use super::sparse_filter_tests::fixture;
use super::tests::{make_input, upsert_mixed_points, TestContext};
use super::*;
use crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store;
use crate::helix_engine::storage_core::backend_lsm::allow_lsm_blocking;
use crate::helix_gateway::async_gateway::graph_error_status;
use slatedb::CloseReason;
use std::sync::Arc;
use std::time::{Duration, Instant};

type Route = fn(&HandlerInput, &mut Response) -> Result<(), GraphError>;

/// The ids the new collection holds; the dropped one held 1 to 1,200.
const NEW_IDS: std::ops::RangeInclusive<u128> = 5_001..=5_040;

/// Every `id` in a reply, whether a number or the hex string of a point id.
fn reply_ids(value: &serde_json::Value, out: &mut Vec<u128>) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, value) in map {
                if key == "id" {
                    let id = match value {
                        serde_json::Value::String(hex) => u128::from_str_radix(hex, 16).ok(),
                        other => other.as_u64().map(u128::from),
                    };
                    out.extend(id);
                }
                reply_ids(value, out);
            }
        }
        serde_json::Value::Array(items) => items.iter().for_each(|item| reply_ids(item, out)),
        _ => {}
    }
}

/// The status a request to `symbols` gets, its body, and the ids in it.
fn ask(
    ctx: &TestContext,
    route: Route,
    method: &str,
    path: &str,
    body: serde_json::Value,
) -> (u16, serde_json::Value, Vec<u128>) {
    let input = make_input(
        ctx,
        method,
        &format!("/collections/symbols/{path}"),
        serde_json::to_vec(&body).unwrap(),
        HashMap::from([("name".into(), "symbols".into())]),
    );
    let mut response = Response::new();
    match allow_lsm_blocking(|| route(&input, &mut response)) {
        Err(error) => (
            graph_error_status(&error),
            serde_json::json!(format!("{error:?}")),
            Vec::new(),
        ),
        Ok(()) => {
            let body: serde_json::Value =
                serde_json::from_slice(&response.body).unwrap_or(serde_json::Value::Null);
            let mut ids = Vec::new();
            reply_ids(&body, &mut ids);
            (response.status, body, ids)
        }
    }
}

/// Drops `symbols` on the writer and creates it again at once with 40 points
/// of its own and no payload index.
fn create_again(writer: &TestContext) {
    let name = || HashMap::from([("name".to_string(), "symbols".to_string())]);
    let delete = make_input(writer, "DELETE", "/collections/symbols", Vec::new(), name());
    handle_delete_collection(&delete, &mut Response::new()).unwrap();
    let create = make_input(
        writer,
        "PUT",
        "/collections/symbols",
        serde_json::to_vec(&serde_json::json!({
            "vectors": {"dense": {"size": 3, "distance": "Cosine"}},
            "sparse_vectors": {"lex_sparse": {
                "modifier": "idf", "index": {"full_scan_threshold": 1}}}
        }))
        .unwrap(),
        name(),
    );
    handle_create_collection(&create, &mut Response::new()).unwrap();
    let points = NEW_IDS
        .map(|id| {
            let angle = id as f32 * 0.37;
            serde_json::json!({
                "id": id as u64,
                "vector": {
                    "dense": [angle.cos(), angle.sin(), (id % 17) as f32 / 17.0],
                    "lex_sparse": {"indices": [1, 2, 3], "values": [1.0, 1.0, 1.0]}
                },
                "payload": {
                    "metadata": {"symbol": "handle_request",
                                 "lang": if id % 2 == 1 { "rust" } else { "go" }},
                    "code": format!("fn handle_request() {{ step_{id}(); }}")
                }
            })
        })
        .collect::<Vec<_>>();
    upsert_mixed_points(writer, "symbols", serde_json::Value::Array(points));
}

#[test]
#[serial_test::serial]
fn reads_after_a_recreate_are_answered_from_the_new_collection_at_once() {
    // Pinned reads and not; and which lookup meets the closed reader first:
    // the one that opens (a search), or the one that only hands out what is
    // loaded (the collection's info).
    for (pinned, info_first, prefix) in [("1", false, "rc_reads_a"), ("0", true, "rc_reads_b")] {
        let _store = set_lsm_test_object_store(Arc::new(object_store::memory::InMemory::new()));
        let _prefix = EnvVar::set("HELIX_LSM_PREFIX", prefix);
        let _poll = EnvVar::set("HELIX_LSM_READER_MANIFEST_POLL_MS", "500");
        let _pinned = EnvVar::set("HELIX_LSM_READER_SNAPSHOT_READS", pinned);
        let (writer, reader) = fixture();
        let replica = reader.collections.get_collection("symbols").unwrap();
        create_again(&writer);
        // The slack is for a stalled runner: the reader polls every 500 ms.
        let created = Instant::now();
        while replica.unhealthy_lsm_writer_close_reason() != Some(CloseReason::DatabaseMissing) {
            assert!(
                created.elapsed() < Duration::from_secs(15),
                "the reader went on in the dropped collection"
            );
            std::thread::sleep(Duration::from_millis(50));
        }

        // A tier change in that state does not bring the reader back.
        let AnyBackend::LsmReader(handle) = replica.backend.as_ref() else {
            panic!("not a reader replica");
        };
        let reopened = handle.refresh_with_poll_interval(Duration::from_millis(500));
        assert!(
            reopened.is_err()
                && replica.unhealthy_lsm_writer_close_reason()
                    == Some(CloseReason::DatabaseMissing),
            "a reopen brought the closed reader back: {reopened:?}"
        );
        drop(replica);

        let dense = serde_json::json!({"name": "dense", "vector": [1.0, 0.0, 0.0]});
        let sparse = serde_json::json!({"indices": [1, 2, 3], "values": [1.0, 1.0, 1.0]});
        let lang =
            serde_json::json!({"must": [{"key": "metadata.lang", "match": {"value": "rust"}}]});
        let text =
            serde_json::json!({"must": [{"key": "code", "match": {"text": "handle_request"}}]});
        let hybrid = |with_payload: bool| {
            serde_json::json!({
                "dense": [{"query": [1.0, 0.0, 0.0], "using": "dense", "limit": 5}],
                "sparse": [{"query": sparse, "using": "lex_sparse", "limit": 5}],
                "limit": 5, "with_payload": with_payload})
        };
        // (label, route, method, path, body, status, hits wanted)
        let mut routes: Vec<(&str, Route, &str, &str, serde_json::Value, u16, bool)> = vec![
            (
                "search dense with payload",
                handle_search_points,
                "POST",
                "points/search",
                serde_json::json!({"vector": dense, "limit": 5, "with_payload": true}),
                200,
                true,
            ),
            (
                "collection info",
                handle_get_collection,
                "GET",
                "",
                serde_json::json!({}),
                200,
                false,
            ),
            (
                "search dense no payload",
                handle_search_points,
                "POST",
                "points/search",
                serde_json::json!({"vector": dense, "limit": 5, "with_payload": false}),
                200,
                true,
            ),
            (
                "search dense keyword filter",
                handle_search_points,
                "POST",
                "points/search",
                serde_json::json!({"vector": dense, "limit": 5, "with_payload": false, "filter": lang}),
                200,
                true,
            ),
            (
                "search dense text filter",
                handle_search_points,
                "POST",
                "points/search",
                serde_json::json!({"vector": dense, "limit": 5, "with_payload": false, "filter": text}),
                200,
                true,
            ),
            (
                "query sparse",
                handle_query_points,
                "POST",
                "points/query",
                serde_json::json!({"query": sparse, "using": "lex_sparse", "limit": 5, "with_payload": true}),
                200,
                true,
            ),
            (
                "query sparse no payload",
                handle_query_points,
                "POST",
                "points/query",
                serde_json::json!({"query": sparse, "using": "lex_sparse", "limit": 5, "with_payload": false}),
                200,
                true,
            ),
            (
                "query dense",
                handle_query_points,
                "POST",
                "points/query",
                serde_json::json!({"query": [1.0, 0.0, 0.0], "using": "dense", "limit": 5, "with_payload": false}),
                200,
                true,
            ),
            (
                "hybrid",
                handle_hybrid_query_points,
                "POST",
                "points/hybrid_query",
                hybrid(true),
                200,
                true,
            ),
            (
                "hybrid no payload",
                handle_hybrid_query_points,
                "POST",
                "points/hybrid_query",
                hybrid(false),
                200,
                true,
            ),
            (
                "scroll",
                handle_scroll_points,
                "POST",
                "points/scroll",
                serde_json::json!({"limit": 5, "with_payload": true}),
                200,
                true,
            ),
            (
                "scroll no payload",
                handle_scroll_points,
                "POST",
                "points/scroll",
                serde_json::json!({"limit": 5, "with_payload": false}),
                200,
                true,
            ),
            (
                "scan",
                handle_scan_points,
                "POST",
                "points/scan",
                serde_json::json!({}),
                200,
                false,
            ),
            (
                "count exact",
                handle_count_points,
                "POST",
                "points/count",
                serde_json::json!({"exact": true}),
                200,
                false,
            ),
            (
                "count default",
                handle_count_points,
                "POST",
                "points/count",
                serde_json::json!({}),
                200,
                false,
            ),
            (
                "count keyword filter",
                handle_count_points,
                "POST",
                "points/count",
                serde_json::json!({"exact": true, "filter": lang}),
                200,
                false,
            ),
            (
                "get points",
                handle_get_points,
                "POST",
                "points",
                serde_json::json!({"ids": [3, 4, 5001], "with_payload": true}),
                200,
                true,
            ),
            // The new collection has no payload index; the dropped one had.
            (
                "facet",
                handle_facet,
                "POST",
                "facet",
                serde_json::json!({"key": "metadata.lang"}),
                422,
                false,
            ),
            (
                "collection exists",
                handle_collection_exists,
                "GET",
                "exists",
                serde_json::json!({}),
                200,
                false,
            ),
        ];
        if info_first {
            routes.swap(0, 1);
        }
        for (label, route, method, path, body, status, hits) in routes {
            let (got, reply, ids) = ask(&reader, route, method, path, body);
            let said = format!("pinned={pinned} {label}: status {got}, ids {ids:?}, {reply}");
            // The info of a collection that is not loaded comes from its
            // metadata file: with none it is 503 until something opens the
            // collection. It is never the dropped collection's.
            let unopened_info = label == "collection info" && info_first && got == 503;
            assert!(got == status || unopened_info, "{said}");
            assert!(ids.iter().all(|id| NEW_IDS.contains(id)), "{said}");
            assert!(!hits || !ids.is_empty(), "{said}");
            let count = &reply["result"]["count"];
            match label {
                "count exact" | "count default" => assert_eq!(*count, 40, "{said}"),
                "count keyword filter" => assert_eq!(*count, 20, "{said}"),
                "search dense keyword filter" => {
                    assert!(ids.iter().all(|id| id % 2 == 1), "{said}")
                }
                "get points" => assert_eq!(ids, [5_001], "{said}"),
                "collection info" if !unopened_info => {
                    assert_eq!(reply["result"]["points_count"], 40, "{said}")
                }
                "facet" => assert!(
                    reply.to_string().contains("Payload index required"),
                    "{said}"
                ),
                _ => {}
            }
        }
        // And once a request has opened it.
        let (got, reply, _) = ask(
            &reader,
            handle_get_collection,
            "GET",
            "",
            serde_json::json!({}),
        );
        assert_eq!(
            (got, &reply["result"]["points_count"]),
            (200, &serde_json::json!(40)),
            "pinned={pinned} collection info at the end: {reply}"
        );
    }
}

/// The files of dense vectors in a collection's local directory, each with
/// its inode: a file that was unlinked and built again has another.
fn dense_files(dir: &std::path::Path) -> Vec<(std::path::PathBuf, u64)> {
    use std::os::unix::fs::MetadataExt;
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter(|entry| {
            matches!(
                entry.path().extension().and_then(|ext| ext.to_str()),
                Some("hvec" | "hvs8" | "hvtq" | "hspn")
            )
        })
        .filter_map(|entry| Some((entry.path(), entry.metadata().ok()?.ino())))
        .collect();
    files.sort();
    files
}

/// The dense vector the fixture and `create_again` give point `id`.
fn dense_of(id: u128) -> [f32; 3] {
    let angle = id as f32 * 0.37;
    [angle.cos(), angle.sin(), (id % 17) as f32 / 17.0]
}

/// A reader scores a flat dense search from a local file of the segment's
/// vectors, which it opens by the segment's name. The names start again in
/// a collection that is created again, so the dropped collection's file was
/// opened for the new one and the new points were scored by the dropped
/// collection's vectors: the right ids in the wrong order with the wrong
/// scores, and no error.
///
/// Two ways there: the reader had the collection open and its `DbReader`
/// ended on the new one, or the collection was not loaded when it was
/// dropped and created again, and no reader of the old one is involved. A
/// collection that is opened again as it was keeps its files.
///
/// The marker of a fatal read error lies in the same directory and is read
/// at every open: the dropped collection's would refuse the new one's graph
/// reads and the refresh of its metadata. It goes with the files, and stays
/// where the collection is the one it was written for.
///
/// And a file that cannot be removed fails the open, which leaves the local
/// copy of the metadata as it was: the next open tries again. An open that
/// went through would open that file by its name.
#[test]
#[serial_test::serial]
fn dense_search_after_a_recreate_is_scored_from_the_new_vectors() {
    let mut failed = Vec::new();
    for (case, prefix) in [
        ("open at the recreate", "rc_dense_a"),
        ("not loaded at the recreate", "rc_dense_b"),
        ("a file stays at the recreate", "rc_dense_d"),
        ("not created again", "rc_dense_c"),
    ] {
        let _store = set_lsm_test_object_store(Arc::new(object_store::memory::InMemory::new()));
        let _prefix = EnvVar::set("HELIX_LSM_PREFIX", prefix);
        let _poll = EnvVar::set("HELIX_LSM_READER_MANIFEST_POLL_MS", "500");
        // The open that has to fail is followed at once by one that must not.
        let _backoff = EnvVar::set("HELIX_COLLECTION_OPEN_FAILURE_BACKOFF_MS", "0");
        let (writer, reader) = fixture();
        // The nearest point to `id`'s own vector and its score: `id` and 1,
        // unless the points are scored by other vectors than their own.
        // A search that got no answer is the case's failure, the first kept.
        let refused = std::cell::RefCell::new(None::<String>);
        let own_score = |id: u128| {
            let (status, reply, ids) = ask(
                &reader,
                handle_search_points,
                "POST",
                "points/search",
                serde_json::json!({"vector": {"name": "dense", "vector": dense_of(id)},
                                   "limit": 1, "with_payload": false}),
            );
            if status != 200 {
                refused
                    .borrow_mut()
                    .get_or_insert(format!("{status} {reply}"));
                return (None, None);
            }
            (ids.first().copied(), reply["result"][0]["score"].as_f64())
        };
        // The reader's refresh of the collection's metadata: a read through
        // `with_read_backend`, which a collection marked degraded refuses.
        // The graph routes read through it as well; a points search does
        // not, and is answered with the marker there.
        let metadata_read = || {
            reader
                .collections
                .get_collection("symbols")
                .and_then(|storage| storage.refresh_metadata_snapshot())
                .map_err(|error| format!("{error:?}"))
        };
        let not_found = |ids: std::ops::RangeInclusive<u128>| {
            ids.map(|id| (id, own_score(id)))
                .filter(|(id, (top, score))| {
                    *top != Some(*id) || score.is_none_or(|score| score < 0.999)
                })
                .collect::<Vec<_>>()
        };

        // The reader's first flat search leaves its file of the segment.
        assert!(
            not_found(1..=40).is_empty(),
            "{case}: before anything {:?}",
            refused.borrow()
        );
        let replica = reader.collections.get_collection("symbols").unwrap();
        let dir = replica.collection_path().to_path_buf();
        let files = dense_files(&dir);
        assert!(
            !files.is_empty(),
            "{case}: the reader has no local file of vectors to go stale"
        );
        // The marker a fatal read error of this collection leaves here.
        let marker = dir.join(".helix_collection_degraded");
        let plant_marker = || {
            std::fs::write(&marker, "lsm_io\nplanted by the test\n").unwrap();
            assert!(
                HelixGraphStorage::read_degraded_marker(&dir.to_string_lossy()).is_some(),
                "{case}: the planted file is not read as the marker"
            );
        };
        let asked = match case {
            "open at the recreate" => {
                plant_marker();
                create_again(&writer);
                // The slack is for a stalled runner: the reader polls every 500 ms.
                let created = Instant::now();
                while replica.unhealthy_lsm_writer_close_reason()
                    != Some(CloseReason::DatabaseMissing)
                {
                    assert!(created.elapsed() < Duration::from_secs(15));
                    std::thread::sleep(Duration::from_millis(50));
                }
                drop(replica);
                NEW_IDS
            }
            "not loaded at the recreate" => {
                drop(replica);
                reader.collections.evict_collection("symbols");
                plant_marker();
                create_again(&writer);
                NEW_IDS
            }
            "a file stays at the recreate" => {
                drop(replica);
                reader.collections.evict_collection("symbols");
                let created = || {
                    HelixGraphStorage::read_metadata_sidecar_from_path(&dir)
                        .ok()
                        .flatten()
                        .map(|local| local.metadata.created_at_millis)
                };
                let before = created();
                // What `remove_file` cannot remove, whoever runs the test.
                let stuck = dir.join("stuck.hvec");
                std::fs::create_dir(&stuck).unwrap();
                create_again(&writer);
                let (_, score) = own_score(*NEW_IDS.start());
                let reply = refused.take();
                if score.is_some() || reply.is_none() || created() != before {
                    failed.push(format!(
                        "{case}: the open with a file left answered {score:?} {reply:?}, \
                         and the local copy's creation time went from {before:?} to {:?}",
                        created()
                    ));
                }
                std::fs::remove_dir(&stuck).unwrap();
                NEW_IDS
            }
            _ => {
                drop(replica);
                reader.collections.evict_collection("symbols");
                1..=40
            }
        };

        let wrong = not_found(asked);
        if let Some(reply) = refused.take() {
            failed.push(format!("{case}: a search was refused, {reply}"));
            continue;
        }
        if case != "not created again" {
            if let Err(reply) = metadata_read() {
                failed.push(format!(
                    "{case}: the dropped collection's marker refused a read of the new one, {reply}"
                ));
            }
            if marker.exists() {
                failed.push(format!(
                    "{case}: the dropped collection's marker is still there"
                ));
            }
        }
        if let Some(first) = wrong.first() {
            failed.push(format!(
                "{case}: {} of 40, the first {first:?}",
                wrong.len()
            ));
        }
        // And the reader ranks as the writer does, with the writer's scores.
        let (on_reader, on_writer) = (top_three(&reader), top_three(&writer));
        if on_reader != on_writer {
            failed.push(format!(
                "{case}: the reader's top three {on_reader:?}, the writer's {on_writer:?}"
            ));
        }
        if case == "not created again" {
            if dense_files(&dir) != files {
                failed.push(format!(
                    "{case}: the reader built its files again, {files:?} then {:?}",
                    dense_files(&dir)
                ));
            }
            // And a marker of its own is read at its next open, as before.
            plant_marker();
            reader.collections.evict_collection("symbols");
            if metadata_read().is_ok() || !marker.exists() {
                failed.push(format!(
                    "{case}: the collection's own marker did not refuse the read"
                ));
            }
        }
    }
    assert!(
        failed.is_empty(),
        "points not found by their own vector (point, (top id, score)), \
         or ranked otherwise than on the writer (id, score to five places): {failed:#?}"
    );
}

/// The three nearest points to one fixed vector on `ctx`, each with its
/// score to five places.
fn top_three(ctx: &TestContext) -> Vec<(u128, i64)> {
    let (status, reply, ids) = ask(
        ctx,
        handle_search_points,
        "POST",
        "points/search",
        serde_json::json!({"vector": {"name": "dense", "vector": [1.0, 0.0, 0.0]},
                           "limit": 3, "with_payload": false}),
    );
    assert_eq!(status, 200, "{reply}");
    let scores = reply["result"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|hit| (hit["score"].as_f64().unwrap_or(f64::NAN) * 100_000.0).round() as i64);
    ids.into_iter().zip(scores).collect()
}
