//! `POST /collections/{name}/sparse/stats`: the document count and document
//! frequencies a lexical (IDF) search on this node weights a query's terms
//! with. A client that searches two collections as one view reads both and
//! rescales its query weights, since each collection scores with its own.

use super::*;

/// Most term ids one request may ask for.
const SPARSE_STATS_MAX_INDICES: usize = 1024;

#[cfg(test)]
thread_local! {
    static ROUTE_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Test-only: run `f` with `sparse_stats_route_enabled` forced on this thread.
#[cfg(test)]
fn with_sparse_stats_route_for_test<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    let previous = ROUTE_OVERRIDE.with(|cell| cell.replace(Some(enabled)));
    let out = f();
    ROUTE_OVERRIDE.with(|cell| cell.set(previous));
    out
}

/// Kill switch for the route. `HELIX_SPARSE_STATS_ROUTE=0` makes it answer
/// 404, as a writer built without the route does (a reader built without it
/// answers 503: there the POST still counts as a write). Read once, at the
/// first request.
fn sparse_stats_route_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = ROUTE_OVERRIDE.with(std::cell::Cell::get) {
        return enabled;
    }
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        route_enabled_for(std::env::var("HELIX_SPARSE_STATS_ROUTE").ok().as_deref())
    })
}

fn route_enabled_for(value: Option<&str>) -> bool {
    !matches!(
        value.map(str::trim),
        Some("0" | "false" | "FALSE" | "no" | "off")
    )
}

#[derive(Deserialize)]
struct SparseStatsRequest {
    /// The sparse vector space.
    using: String,
    /// Term ids, as in a sparse query's `indices`.
    #[serde(default)]
    indices: Vec<u32>,
}

#[derive(Serialize)]
struct SparseStats<'a> {
    using: &'a str,
    modifier: SparseModifier,
    doc_count: u64,
    /// One per requested index, in the request's order; 0 without postings.
    df: Vec<u64>,
    /// `writer` or `reader`.
    role: &'static str,
    /// The collection's commit sequence the numbers were read at. A reader's
    /// trails the writer's by what it has not polled yet, and two replies
    /// with one value carry the same numbers. (The writer also answers from
    /// commits that are not durable yet, so after a writer crash a value it
    /// gave before can come back with other numbers.) Null for a read that
    /// is not pinned to one sequence.
    seq: Option<u64>,
}

/// A read: it writes nothing, readers serve it, and a reader that is behind
/// answers from the view its own search would score with, without waiting.
/// It costs what a search's first step costs: the count is one point read and
/// the frequencies come through the scorer's term cache.
///
/// It answers only for a collection this node already holds open, and 409
/// otherwise. The call comes with a search, and the search opens the
/// collection in the search pool. A stats call that waited for that open
/// would hold a slot of the much smaller read pool until the open ended,
/// also after its caller had given up.
pub fn handle_sparse_stats(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    if !sparse_stats_route_enabled() {
        return json_err(response, 404, "Sparse stats route is switched off");
    }
    let name = match get_collection_name(input) {
        Some(n) => n,
        None => return json_err(response, 400, "Missing collection name"),
    };
    let req: SparseStatsRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_err(response, 400, &format!("Invalid JSON: {}", e)),
    };
    if req.indices.len() > SPARSE_STATS_MAX_INDICES {
        return json_err(
            response,
            400,
            &format!(
                "At most {} indices per request, got {}",
                SPARSE_STATS_MAX_INDICES,
                req.indices.len()
            ),
        );
    }

    let Some(storage) = input.collections.get_loaded_collection(&name)? else {
        return json_err(
            response,
            409,
            &format!("Collection '{}' is not open on this node", name),
        );
    };
    // shortcut: LSM only, the backend production runs. LMDB needs its
    // resize-safe read txn and the heed readers; add them if it ever serves this.
    if storage.backend.kind() != BackendKind::Lsm {
        return json_err(
            response,
            404,
            "Sparse stats are served on the LSM backend only",
        );
    }
    let Some(config) = storage.named_vectors.get_sparse_config(&req.using) else {
        return json_err(
            response,
            404,
            &format!(
                "Sparse vector '{}' not found in collection '{}'",
                req.using, name
            ),
        );
    };

    let r = storage
        .backend
        .begin_read()
        .map_err(|e| GraphError::New(e.to_string()))?;
    let (doc_count, df) = storage
        .named_vectors
        .with_sparse_core(&req.using, |core| core.term_stats_be(&r, &req.indices))
        .map_err(GraphError::from)?;
    json_ok(
        response,
        &SparseStats {
            using: &req.using,
            modifier: config.modifier,
            doc_count,
            df,
            role: if storage.backend.is_reader_replica() {
                "reader"
            } else {
                "writer"
            },
            seq: r.snapshot_seq(),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::super::tests::{make_input, setup_with_config, upsert_mixed_points, TestContext};
    use super::*;
    use crate::helix_engine::graph_core::config::Config;
    use crate::helix_engine::storage_core::backend::{LsmRole, LsmStorage, StorageBackendConfig};
    use crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store;
    use crate::helix_engine::storage_core::backend_lsm::allow_lsm_blocking;
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    /// Lexical vectors by point id: what the collection has been told to hold.
    type Model = BTreeMap<u32, Vec<(u32, f32)>>;

    /// Point `id` of the fixture: one term of thirteen and one of seven, so
    /// every term has many documents.
    fn fixture_point(id: u32) -> (u32, Vec<(u32, f32)>) {
        (id, vec![(id % 13, 1.0), (100 + id % 7, 0.5)])
    }

    fn create(ctx: &TestContext, name: &str) {
        let input = make_input(
            ctx,
            "PUT",
            &format!("/collections/{name}"),
            serde_json::to_vec(&json!({
                "vectors": {"dense": {"size": 3, "distance": "Cosine"}},
                // Threshold 1: a query of three terms takes the WAND scorer,
                // a shorter one the full scan.
                "sparse_vectors": {"lex_sparse": {
                    "modifier": "idf", "index": {"full_scan_threshold": 1}}}
            }))
            .unwrap(),
            HashMap::from([("name".into(), name.into())]),
        );
        let mut response = Response::new();
        handle_create_collection(&input, &mut response).unwrap();
        assert!(response.status == 200 || response.status == 201);
    }

    /// Writes the points and records them in the model.
    fn put(ctx: &TestContext, name: &str, model: &mut Model, points: Vec<(u32, Vec<(u32, f32)>)>) {
        let body = points
            .iter()
            .map(|(id, terms)| {
                let angle = *id as f32 * 0.37;
                json!({
                    "id": id,
                    "vector": {
                        "dense": [angle.cos(), angle.sin(), 0.5],
                        "lex_sparse": {
                            "indices": terms.iter().map(|(term, _)| *term).collect::<Vec<_>>(),
                            "values": terms.iter().map(|(_, value)| *value).collect::<Vec<_>>(),
                        }
                    }
                })
            })
            .collect::<Vec<_>>();
        upsert_mixed_points(ctx, name, serde_json::Value::Array(body));
        model.extend(points);
    }

    fn delete(ctx: &TestContext, name: &str, model: &mut Model, ids: &[u32]) {
        let input = make_input(
            ctx,
            "POST",
            &format!("/collections/{name}/points/delete"),
            serde_json::to_vec(&json!({"points": ids})).unwrap(),
            HashMap::from([("name".into(), name.into())]),
        );
        let mut response = Response::new();
        handle_delete_points(&input, &mut response).unwrap();
        assert_eq!(
            response.status,
            200,
            "{}",
            String::from_utf8_lossy(&response.body)
        );
        for id in ids {
            model.remove(id);
        }
    }

    /// One request to the route: the handler's error, or the status and body.
    fn stats(
        ctx: &TestContext,
        name: &str,
        body: &serde_json::Value,
    ) -> Result<(u16, serde_json::Value), String> {
        let input = make_input(
            ctx,
            "POST",
            &format!("/collections/{name}/sparse/stats"),
            serde_json::to_vec(body).unwrap(),
            HashMap::from([("name".into(), name.into())]),
        );
        let mut response = Response::new();
        // As the gateway runs a handler.
        allow_lsm_blocking(|| handle_sparse_stats(&input, &mut response))
            .map_err(|e| e.to_string())?;
        Ok((
            response.status,
            serde_json::from_slice(&response.body).unwrap(),
        ))
    }

    /// Document count and each term's document frequency, counted from the
    /// model's vectors.
    fn counted(model: &Model, terms: &[u32]) -> (u64, Vec<u64>) {
        let df = terms
            .iter()
            .map(|term| {
                model
                    .values()
                    .filter(|doc| doc.iter().any(|(held, _)| held == term))
                    .count() as u64
            })
            .collect();
        (model.len() as u64, df)
    }

    /// The route's reply for `terms` is what the model counts; returns its result.
    fn assert_counts(
        ctx: &TestContext,
        name: &str,
        model: &Model,
        terms: &[u32],
        role: &str,
    ) -> serde_json::Value {
        let (status, body) =
            stats(ctx, name, &json!({"using": "lex_sparse", "indices": terms})).unwrap();
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["status"], "ok", "{body}");
        let (doc_count, df) = counted(model, terms);
        assert_eq!(
            body["result"],
            json!({
                "using": "lex_sparse", "modifier": "idf", "doc_count": doc_count, "df": df,
                "role": role, "seq": body["result"]["seq"],
            })
        );
        body["result"].clone()
    }

    /// Every hit of a lexical query scores what the route's numbers give: the
    /// sum over the terms it shares with the query of
    /// query x stored x ln(1 + doc_count / df).
    fn assert_scorer_uses(ctx: &TestContext, name: &str, model: &Model, query: &[(u32, f32)]) {
        let terms: Vec<u32> = query.iter().map(|(term, _)| *term).collect();
        let (_, body) =
            stats(ctx, name, &json!({"using": "lex_sparse", "indices": terms})).unwrap();
        let doc_count = body["result"]["doc_count"].as_u64().unwrap() as f64;
        let df: Vec<f64> = body["result"]["df"]
            .as_array()
            .unwrap()
            .iter()
            .map(|df| df.as_u64().unwrap() as f64)
            .collect();

        let input = make_input(
            ctx,
            "POST",
            &format!("/collections/{name}/points/query"),
            serde_json::to_vec(&json!({
                "query": {
                    "indices": terms,
                    "values": query.iter().map(|(_, weight)| *weight).collect::<Vec<_>>(),
                },
                "using": "lex_sparse", "limit": 5, "with_payload": false
            }))
            .unwrap(),
            HashMap::from([("name".into(), name.into())]),
        );
        let mut response = Response::new();
        allow_lsm_blocking(|| handle_query_points(&input, &mut response)).unwrap();
        let reply: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(response.status, 200, "{reply}");
        let hits = reply["result"]["points"].as_array().unwrap();
        assert!(!hits.is_empty(), "no hit for {query:?}");
        for hit in hits {
            let id = u32::from_str_radix(hit["id"].as_str().unwrap(), 16).unwrap();
            let expected: f64 = query
                .iter()
                .zip(&df)
                .filter_map(|((term, weight), df)| {
                    let stored = model[&id].iter().find(|(held, _)| held == term)?.1;
                    Some(f64::from(*weight) * f64::from(stored) * (1.0 + doc_count / df).ln())
                })
                .sum();
            let score = hit["score"].as_f64().unwrap();
            assert!(
                expected > 0.0 && (score - expected).abs() <= 1e-9 * expected,
                "point {id} scored {score}, the route's numbers give {expected}"
            );
        }
    }

    #[test]
    fn stats_are_the_counts_of_the_stored_vectors() {
        let ctx = setup_with_config(Config::default().with_lsm_in_memory());
        create(&ctx, "stats");
        let mut model = Model::new();
        put(
            &ctx,
            "stats",
            &mut model,
            (1..=300).map(fixture_point).collect(),
        );
        // Two terms many points hold, one that none holds, one asked twice.
        let first = assert_counts(&ctx, "stats", &model, &[3, 102, 999, 3], "writer");
        assert!(first["seq"].is_u64(), "{first}");
        assert_scorer_uses(&ctx, "stats", &model, &[(3, 1.0)]);
        assert_scorer_uses(&ctx, "stats", &model, &[(3, 1.0), (102, 2.0), (5, 0.5)]);

        // Deleted points leave the count and their terms' frequencies; a
        // rewritten point keeps the count and moves between terms.
        delete(&ctx, "stats", &mut model, &(1..=20).collect::<Vec<_>>());
        put(
            &ctx,
            "stats",
            &mut model,
            (21..=30)
                .map(|id| (id, vec![(500, 0.5), (100 + id % 7, 0.5)]))
                .collect(),
        );
        let later = assert_counts(&ctx, "stats", &model, &[3, 102, 500, 8], "writer");
        assert_eq!(later["doc_count"], 280);
        assert_eq!(later["df"][2], 10);
        assert!(later["seq"].as_u64() > first["seq"].as_u64());
        assert_scorer_uses(&ctx, "stats", &model, &[(500, 2.0)]);
        assert_scorer_uses(&ctx, "stats", &model, &[(500, 2.0), (3, 1.0), (102, 1.0)]);

        // No term asked: the count alone.
        assert_counts(&ctx, "stats", &model, &[], "writer");
    }

    /// Sets an env var for the test and restores it after.
    struct EnvVar(&'static str, Option<String>);

    impl EnvVar {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self(key, previous)
        }
    }

    impl Drop for EnvVar {
        fn drop(&mut self) {
            match &self.1 {
                Some(value) => std::env::set_var(self.0, value),
                None => std::env::remove_var(self.0),
            }
        }
    }

    /// A reader replica answers from its own view and names it: the numbers
    /// at the sequence it read, the old ones while it has not polled, the
    /// writer's once it has.
    #[test]
    #[serial_test::serial]
    fn a_reader_replica_answers_from_its_view_and_names_it() {
        let _store = set_lsm_test_object_store(Arc::new(object_store::memory::InMemory::new()));
        let _prefix = EnvVar::set("HELIX_LSM_PREFIX", "sparse_stats_reader");
        let _pinned = EnvVar::set("HELIX_LSM_READER_SNAPSHOT_READS", "1");
        let config = |role| {
            Config::default().with_storage_backend(StorageBackendConfig::Lsm {
                storage: LsmStorage::ObjectStore,
                role,
            })
        };
        let writer = setup_with_config(config(LsmRole::Writer));
        create(&writer, "replica");
        let mut model = Model::new();
        put(
            &writer,
            "replica",
            &mut model,
            (1..=300).map(fixture_point).collect(),
        );
        let terms = [3, 102, 777, 999];

        let reader = setup_with_config(config(LsmRole::Reader));
        // A node that does not hold the collection open says so at once and
        // does not open it; the search that comes with the call does.
        let (status, body) = stats(
            &reader,
            "replica",
            &json!({"using": "lex_sparse", "indices": terms}),
        )
        .unwrap();
        assert_eq!(status, 409, "{body}");
        assert!(reader
            .collections
            .get_loaded_collection("replica")
            .unwrap()
            .is_none());
        let replica = reader.collections.get_collection("replica").unwrap();
        assert!(replica.backend.is_reader_replica());
        let opened = assert_counts(&reader, "replica", &model, &terms, "reader");
        let opened_seq = opened["seq"].as_u64().unwrap();
        assert_scorer_uses(&reader, "replica", &model, &[(3, 1.0)]);
        assert_scorer_uses(
            &reader,
            "replica",
            &model,
            &[(3, 1.0), (102, 2.0), (5, 0.5)],
        );

        // The writer moves on: points under a new term, and three deleted.
        let before = model.clone();
        put(
            &writer,
            "replica",
            &mut model,
            (301..=320).map(|id| (id, vec![(777, 1.0)])).collect(),
        );
        delete(&writer, "replica", &mut model, &[1, 2, 3]);
        assert_counts(&writer, "replica", &model, &terms, "writer");

        // The reader does not wait for that. While its view is the one it
        // opened with, so are its numbers.
        let (status, body) = stats(
            &reader,
            "replica",
            &json!({"using": "lex_sparse", "indices": terms}),
        )
        .unwrap();
        assert_eq!(status, 200, "{body}");
        if body["result"]["seq"].as_u64() == Some(opened_seq) {
            let (doc_count, df) = counted(&before, &terms);
            assert_eq!(body["result"]["doc_count"], doc_count, "{body}");
            assert_eq!(body["result"]["df"], json!(df), "{body}");
        }

        // After its poll (forced here) it answers the writer's numbers, at a
        // later sequence that is not past the writer's.
        assert!(replica
            .backend
            .refresh_lsm_reader_with_poll_interval(Duration::from_millis(100))
            .unwrap());
        let caught_up = assert_counts(&reader, "replica", &model, &terms, "reader");
        let seq = caught_up["seq"].as_u64().unwrap();
        assert!(
            seq > opened_seq,
            "sequence {seq} after the poll, {opened_seq} before"
        );
        let writer_seq = assert_counts(&writer, "replica", &model, &terms, "writer")["seq"]
            .as_u64()
            .unwrap();
        assert!(seq <= writer_seq, "reader at {seq}, writer at {writer_seq}");
        assert_scorer_uses(&reader, "replica", &model, &[(777, 2.0)]);

        // A read that is not pinned to a snapshot has no sequence to name.
        let _unpinned = EnvVar::set("HELIX_LSM_READER_SNAPSHOT_READS", "0");
        let unpinned = assert_counts(&reader, "replica", &model, &terms, "reader");
        assert!(unpinned["seq"].is_null(), "{unpinned}");
    }

    #[test]
    fn requests_the_route_cannot_answer_are_refused_by_status() {
        let ctx = setup_with_config(Config::default().with_lsm_in_memory());
        create(&ctx, "refusals");
        let status = |ctx: &TestContext, collection: &str, body: serde_json::Value| {
            stats(ctx, collection, &body).unwrap().0
        };
        let many = |n: usize| json!({"using": "lex_sparse", "indices": vec![7; n]});
        assert_eq!(status(&ctx, "refusals", json!({"indices": [1]})), 400);
        assert_eq!(
            status(
                &ctx,
                "refusals",
                json!({"using": "lex_sparse", "indices": [-1]})
            ),
            400
        );
        assert_eq!(
            status(&ctx, "refusals", many(SPARSE_STATS_MAX_INDICES + 1)),
            400
        );
        assert_eq!(
            status(&ctx, "refusals", many(SPARSE_STATS_MAX_INDICES)),
            200
        );
        assert_eq!(
            status(&ctx, "refusals", json!({"using": "no_such_space"})),
            404
        );
        assert_eq!(status(&ctx, "refusals", json!({"using": "dense"})), 404);

        // A name this node holds no open collection for, existing or not.
        assert_eq!(status(&ctx, "no_such_collection", many(1)), 409);
        // A collection without points: zero documents, zero frequencies.
        assert_counts(&ctx, "refusals", &Model::new(), &[7, 8], "writer");

        // Switched off, the route answers as a node without it does.
        assert_eq!(
            with_sparse_stats_route_for_test(false, || status(&ctx, "refusals", many(1))),
            404
        );
        assert_eq!(status(&ctx, "refusals", many(1)), 200);

        // The legacy backend does not serve it.
        let lmdb = setup_with_config(Config::default());
        create(&lmdb, "legacy");
        assert_eq!(status(&lmdb, "legacy", many(1)), 404);
    }

    #[test]
    fn the_switch_is_on_unless_set_to_an_off_value() {
        for on in [None, Some(""), Some("1"), Some("true"), Some("on")] {
            assert!(route_enabled_for(on), "{on:?}");
        }
        for off in ["0", "false", "FALSE", "no", "off", " 0 "] {
            assert!(!route_enabled_for(Some(off)), "{off:?}");
        }
    }
}
