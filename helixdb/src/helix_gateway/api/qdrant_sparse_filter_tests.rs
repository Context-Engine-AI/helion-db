//! A filtered sparse search is asked for its result window, no longer for
//! four times as many docs, and its filter reads payloads in batches. Neither
//! changes a reply: whole pages are compared with both switched back, on a
//! writer and on a reader replica.

use super::hybrid_parallel_tests::EnvVar;
use super::sparse_filter::with_sparse_filter_reads_for_test;
use super::tests::{make_input, setup_with_config, upsert_mixed_points, TestContext};
use super::*;
use crate::helix_engine::graph_core::config::Config;
use crate::helix_engine::storage_core::backend::{LsmRole, LsmStorage, StorageBackendConfig};
use crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store;
use crate::helix_engine::storage_core::backend_lsm::{
    allow_lsm_blocking, allow_lsm_blocking_cancellable, lsm_round_trips_for_test,
    LsmReadCancellation,
};
use crate::helix_engine::storage_core::backend_lsm_reader::LsmReader;
use crate::helix_engine::storage_core::storage_core::node_reads_for_test;
use crate::helix_engine::vector_core::sparse::DocFilter;
use std::cell::Cell;
use std::sync::Arc;

const POINTS: u32 = 1_200;

/// The identifiers looked up, the terms of each, and how many points carry it
/// as their symbol: far more than a page, more than a page but fewer than
/// four, fewer than a page, none.
const IDENTIFIERS: [(&str, [u32; 3], usize); 4] = [
    ("handle_request", [1, 2, 3], 400),
    ("open_buffer", [4, 5, 6], 40),
    ("serialize_anchor", [7, 8, 9], 5),
    // In no point. Its first term is one of `open_buffer`'s, so docs score.
    ("refresh_buffer", [5, 10, 11], 0),
];

fn symbol(id: u32) -> String {
    if id % 3 == 0 {
        "handle_request".into()
    } else if id % 30 == 1 {
        "open_buffer".into()
    } else if id % 240 == 7 {
        "serialize_anchor".into()
    } else {
        format!("helper_{}", id % 50)
    }
}

/// A point's terms: its identifier's three. A helper has one term of some
/// identifier, weighted so that helpers score between the real matches and
/// the filter finds docs to reject among the best, and one term of its own.
fn lex_sparse(id: u32) -> serde_json::Value {
    let weight = 0.5 + (id % 4) as f32 * 0.5;
    match IDENTIFIERS.iter().find(|(name, _, _)| *name == symbol(id)) {
        Some((_, terms, _)) => {
            serde_json::json!({"indices": terms, "values": [weight, weight, weight]})
        }
        None => {
            let borrowed = [1u32, 2, 4, 5, 7, 8][(id / 3 % 6) as usize];
            serde_json::json!({
                "indices": [borrowed, 20 + id % 30],
                "values": [2.0 + (id % 5) as f32, weight]
            })
        }
    }
}

/// A writer and a reader replica of one collection.
pub(super) fn fixture() -> (TestContext, TestContext) {
    let config = |role| {
        Config::default().with_storage_backend(StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role,
        })
    };
    let writer = setup_with_config(config(LsmRole::Writer));
    let input = make_input(
        &writer,
        "PUT",
        "/collections/symbols",
        serde_json::to_vec(&serde_json::json!({
            "vectors": {"dense": {"size": 3, "distance": "Cosine"}},
            // WAND from the first doc, as on a collection of production size.
            "sparse_vectors": {"lex_sparse": {
                "modifier": "idf", "index": {"full_scan_threshold": 1}}}
        }))
        .unwrap(),
        HashMap::from([("name".into(), "symbols".into())]),
    );
    let mut response = Response::new();
    handle_create_collection(&input, &mut response).unwrap();
    assert!(response.status == 200 || response.status == 201);
    let storage = writer.collections.get_collection("symbols").unwrap();
    for field in ["metadata.symbol", "metadata.lang"] {
        storage
            .create_payload_index(field, PayloadIndexSchema::Keyword)
            .unwrap();
    }
    let points = (1..=POINTS)
        .map(|id| {
            let angle = id as f32 * 0.37;
            serde_json::json!({
                "id": id,
                "vector": {
                    "dense": [angle.cos(), angle.sin(), (id % 17) as f32 / 17.0],
                    "lex_sparse": lex_sparse(id)
                },
                "payload": {
                    "metadata": {"symbol": symbol(id),
                                 "lang": if id % 2 == 1 { "rust" } else { "go" }},
                    "code": format!("fn {}() {{ step_{id}(); }}", symbol(id))
                }
            })
        })
        .collect::<Vec<_>>();
    upsert_mixed_points(&writer, "symbols", serde_json::Value::Array(points));

    let reader = setup_with_config(config(LsmRole::Reader));
    let replica = reader.collections.get_collection("symbols").unwrap();
    assert!(replica.backend.is_reader_replica());
    (writer, reader)
}

/// `(label, filter)` for one identifier.
fn filters(identifier: &str) -> Vec<(&'static str, Option<serde_json::Value>)> {
    // As the client looks an identifier up: one `should` over several text fields,
    // not all of them indexed, so no index answers it and each doc's payload
    // is checked.
    let text = serde_json::json!({"should": [
        {"key": "metadata.symbol", "match": {"text": identifier}},
        {"key": "code", "match": {"text": identifier}}]});
    vec![
        ("text", Some(serde_json::json!({"must": [text.clone()]}))),
        // An index names the docs to score; each still needs its payload.
        (
            "lang and text",
            Some(serde_json::json!({"must": [
                {"key": "metadata.lang", "match": {"value": "rust"}}, text]})),
        ),
        // Answered by an index alone.
        (
            "symbol",
            Some(serde_json::json!({"must": [
                {"key": "metadata.symbol", "match": {"value": identifier}}]})),
        ),
        ("none", None),
    ]
}

/// One page through each of the three routes a sparse search arrives by.
fn requests(
    terms: &[u32],
    filter: Option<&serde_json::Value>,
    limit: usize,
    offset: usize,
) -> Vec<(&'static str, serde_json::Value)> {
    let sparse = serde_json::json!({"indices": terms, "values": vec![1.0; terms.len()]});
    let mut bodies = vec![
        (
            "query",
            serde_json::json!({"query": sparse, "using": "lex_sparse", "limit": limit,
                "offset": offset, "with_payload": true}),
        ),
        (
            "search",
            serde_json::json!({"vector": {"name": "lex_sparse", "vector": sparse},
                "limit": limit, "offset": offset, "with_payload": true}),
        ),
        // No offset on this route: the channel is asked for the window.
        (
            "hybrid_query",
            serde_json::json!({"sparse": [
                {"query": sparse, "using": "lex_sparse", "limit": limit + offset}],
                "limit": limit + offset, "with_payload": true, "with_channel_points": true}),
        ),
    ];
    if let Some(filter) = filter {
        for (_, body) in bodies.iter_mut() {
            body["filter"] = filter.clone();
        }
    }
    bodies
}

/// One request with the search asked for `overfetch` windows and the filter
/// reading `read_batch` payloads at a time: the status and the result, the
/// payload node reads, and the LSM round trips made one after another.
fn send(
    ctx: &TestContext,
    endpoint: &str,
    body: &serde_json::Value,
    overfetch: usize,
    read_batch: usize,
) -> ((u16, serde_json::Value), usize, usize) {
    let input = make_input(
        ctx,
        "POST",
        &format!("/collections/symbols/points/{endpoint}"),
        serde_json::to_vec(body).unwrap(),
        HashMap::from([("name".into(), "symbols".into())]),
    );
    let mut response = Response::new();
    let (reads, trips) = (node_reads_for_test(), lsm_round_trips_for_test());
    // As the gateway runs a handler. A hybrid sparse channel stays on this
    // thread, where the switches and the counters are.
    allow_lsm_blocking(|| {
        with_sparse_filter_reads_for_test(overfetch, read_batch, || {
            with_hybrid_parallel_channels_for_test(false, || match endpoint {
                "query" => handle_query_points(&input, &mut response),
                "search" => handle_search_points(&input, &mut response),
                _ => handle_hybrid_query_points(&input, &mut response),
            })
        })
    })
    .unwrap();
    let reply: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
    (
        (response.status, reply["result"].clone()),
        node_reads_for_test() - reads,
        lsm_round_trips_for_test() - trips,
    )
}

#[test]
#[serial_test::serial]
fn sparse_pages_do_not_depend_on_the_fourfold_limit_or_on_batched_filter_reads() {
    let _store = set_lsm_test_object_store(Arc::new(object_store::memory::InMemory::new()));
    let _prefix = EnvVar::set("HELIX_LSM_PREFIX", "sparse_filter_reads");
    let (writer, reader) = fixture();
    // Where the filter checks payloads: the fourfold limit, the window, and
    // the window with payloads read in batches.
    let (mut reads, mut trips) = ([0usize; 3], [0usize; 3]);
    let mut pages = 0usize;
    for (ctx, role) in [(&writer, "writer"), (&reader, "reader")] {
        // Production readers run with multi-get off.
        for multi_get in ["1", "0"] {
            let _multi_get = EnvVar::set("HELIX_LSM_MULTI_GET", multi_get);
            for (identifier, terms, count) in IDENTIFIERS {
                // Three terms take WAND, two the full scan.
                for terms in [&terms[..], &terms[..2]] {
                    for (label, filter) in filters(identifier) {
                        let rechecked = label == "text" || label == "lang and text";
                        for (limit, offset) in [(24usize, 0usize), (10, 0), (10, 10), (3, 0)] {
                            for (endpoint, body) in requests(terms, filter.as_ref(), limit, offset)
                            {
                                let case = format!(
                                    "{role}, multi-get {multi_get}, {identifier} by {} terms, \
                                     filter {label}, limit {limit} offset {offset}, {endpoint}",
                                    terms.len()
                                );
                                let (fourfold, fourfold_reads, fourfold_trips) =
                                    send(ctx, endpoint, &body, 4, 1);
                                assert_eq!(fourfold.0, 200, "{case}: {}", fourfold.1);
                                if endpoint == "query" && label == "text" {
                                    assert_eq!(
                                        fourfold.1["points"].as_array().map(Vec::len),
                                        Some(count.saturating_sub(offset).min(limit)),
                                        "{case}"
                                    );
                                }
                                let (window, window_reads, window_trips) =
                                    send(ctx, endpoint, &body, 1, 1);
                                assert_eq!(window, fourfold, "{case}: without the fourfold limit");
                                let (batched, batched_reads, batched_trips) =
                                    send(ctx, endpoint, &body, 1, 96);
                                assert_eq!(batched, fourfold, "{case}: with batched reads");
                                pages += 1;
                                if rechecked {
                                    for (arm, (arm_reads, arm_trips)) in [
                                        (fourfold_reads, fourfold_trips),
                                        (window_reads, window_trips),
                                        (batched_reads, batched_trips),
                                    ]
                                    .into_iter()
                                    .enumerate()
                                    {
                                        reads[arm] += arm_reads;
                                        trips[arm] += arm_trips;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    println!(
        "{pages} sparse pages; where payloads are checked (fourfold limit, window, window with \
         batched reads): node reads {reads:?}, LSM round trips {trips:?}"
    );
    assert!(
        reads[1] < reads[0],
        "the window alone read {} payloads, the fourfold limit {}",
        reads[1],
        reads[0]
    );
    assert!(
        trips[2] < trips[1],
        "batched reads took {} round trips, one read per doc {}",
        trips[2],
        trips[1]
    );
}

/// What the filter answers for points 3, 4 and 6 read together, and for point
/// 3 read alone. Points 3 and 6 carry `handle_request`, point 4 does not.
fn verdicts(doc_filter: &SparseDocFilter<'_, '_>) -> (Vec<bool>, bool) {
    (doc_filter.passes_many(&[3, 4, 6]), doc_filter.passes(3))
}

/// A payload read that fails says nothing about its docs. The filter answers
/// no and keeps the failure, on which the search fails. A read cut because
/// the caller left is not kept: that search fails as cancelled.
#[test]
#[serial_test::serial]
fn a_failed_payload_read_is_kept_and_a_cancelled_one_is_not() {
    let _store = set_lsm_test_object_store(Arc::new(object_store::memory::InMemory::new()));
    let _prefix = EnvVar::set("HELIX_LSM_PREFIX", "sparse_filter_failed_read");
    let (writer, _reader) = fixture();
    let storage = writer.collections.get_collection("symbols").unwrap();
    // The lookup's text filter: no index answers it.
    let text = filters("handle_request").swap_remove(0).1.unwrap();
    let filter: Filter = serde_json::from_value(text).unwrap();
    let r = storage.backend.begin_read().unwrap();
    let doc_filter = || SparseDocFilter {
        storage: &storage,
        r: &r,
        filter: &filter,
        candidates: None,
        recheck: true,
        failure: OnceLock::new(),
    };

    let read = doc_filter();
    assert_eq!(
        allow_lsm_blocking(|| verdicts(&read)),
        (vec![true, false, true], true)
    );
    assert_eq!(read.failure(), None);

    let gone = LsmReadCancellation::new();
    gone.cancel();
    let cancelled = doc_filter();
    assert_eq!(
        allow_lsm_blocking_cancellable(gone, || verdicts(&cancelled)),
        (vec![false; 3], false)
    );
    assert_eq!(cancelled.failure(), None);

    // The store closes under the read, as a replaced reader does 30 s after
    // a refresh.
    storage.backend.close_lsm_for_cache_eviction().unwrap();
    let closed = doc_filter();
    assert_eq!(
        allow_lsm_blocking(|| verdicts(&closed)),
        (vec![false; 3], false)
    );
    let failure = closed.failure().expect("the failed read was not kept");
    assert!(failure.contains("closed"), "{failure}");
}

/// The lookup's filter, with the reader's handle closed while the walk checks
/// its third doc.
struct ClosedUnderTheWalk<'a, 'r> {
    inner: SparseDocFilter<'a, 'r>,
    /// The handle to close; `None` leaves it open.
    close: Option<&'a LsmReader>,
    checked: Cell<usize>,
}

impl ClosedUnderTheWalk<'_, '_> {
    /// Before `docs` more are checked.
    fn before(&self, docs: usize) {
        let checked = self.checked.get();
        self.checked.set(checked + docs);
        if let Some(reader) = self.close.filter(|_| checked < 3 && checked + docs >= 3) {
            reader.close_current_for_test().unwrap();
        }
    }
}

impl DocFilter for ClosedUnderTheWalk<'_, '_> {
    fn passes(&self, id: u128) -> bool {
        self.before(1);
        self.inner.passes(id)
    }

    fn batch(&self) -> usize {
        self.inner.batch()
    }

    fn passes_many(&self, ids: &[u128]) -> Vec<bool> {
        self.before(ids.len());
        self.inner.passes_many(ids)
    }

    fn failure(&self) -> Option<String> {
        self.inner.failure()
    }
}

/// A reader refresh closes the replaced handle 30 s later, under any request
/// that began before it. A lookup then cannot read the payloads its filter
/// checks: it fails, where it answered 200 with the rows read so far, or
/// none.
#[test]
#[serial_test::serial]
fn a_lookup_whose_reader_closes_under_it_fails_instead_of_answering_short() {
    let text = filters("handle_request").swap_remove(0).1.unwrap();
    let filter: Filter = serde_json::from_value(text).unwrap();
    // Two terms: a full scan, which reads nothing after its filter's walk.
    let query = SparseVector {
        indices: vec![1, 2],
        values: vec![1.0, 1.0],
    };
    // One payload read per doc, and the docs the walk is about to reach.
    for (read_batch, prefix) in [
        (1usize, "sparse_filter_closed_reader_1"),
        (96, "sparse_filter_closed_reader_96"),
    ] {
        let _store = set_lsm_test_object_store(Arc::new(object_store::memory::InMemory::new()));
        let _prefix = EnvVar::set("HELIX_LSM_PREFIX", prefix);
        // As production's readers: a request reads one pinned snapshot.
        let _pinned = EnvVar::set("HELIX_LSM_READER_SNAPSHOT_READS", "1");
        let (_writer, reader) = fixture();
        let replica = reader.collections.get_collection("symbols").unwrap();
        let AnyBackend::LsmReader(handle) = replica.backend.as_ref() else {
            panic!("not a reader replica");
        };
        // The request's read, begun before the close.
        let r = replica.backend.begin_read().unwrap();
        assert!(
            matches!(&r, AnyRead::LsmReader(Some(_))),
            "the lookup's read is not pinned"
        );
        let lookup = |close: Option<&LsmReader>| {
            let doc_filter = ClosedUnderTheWalk {
                inner: SparseDocFilter {
                    storage: &replica,
                    r: &r,
                    filter: &filter,
                    candidates: None,
                    recheck: true,
                    failure: OnceLock::new(),
                },
                close,
                checked: Cell::new(0),
            };
            allow_lsm_blocking(|| {
                with_sparse_filter_reads_for_test(1, read_batch, || {
                    replica
                        .named_vectors
                        .with_sparse_core("lex_sparse", |core| {
                            core.search_be_with(&r, &query, 24, Some(&doc_filter), true)
                        })
                        .map(|rows| rows.len())
                        .map_err(|error| error.to_string())
                })
            })
        };
        assert_eq!(lookup(None), Ok(24), "read batch {read_batch}");
        let failed = lookup(Some(handle));
        assert!(
            matches!(&failed, Err(error) if error.contains("closed")),
            "read batch {read_batch}: {failed:?}"
        );
    }
}
