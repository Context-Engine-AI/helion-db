//! A scan page reads its candidate rows in batches (`scan_read_batch`): the
//! rows it read one at a time, in their order, and no others. Every page here
//! is compared, rows and offset handed on, with the same request read one row
//! at a time.

use super::tests::{make_input, setup_with_config, upsert_points, TestContext};
use super::*;
use crate::helix_engine::graph_core::config::Config;
use crate::helix_engine::storage_core::backend_lsm::lsm_round_trips_for_test;
use crate::helix_engine::storage_core::storage_core::node_reads_for_test;
use serde_json::json;

const NAME: &str = "scan_batch";
/// Rows of the small store, and of the large one: more than a scroll's
/// largest scan budget.
const ROWS: u64 = 120;
const LARGE_ROWS: u64 = 4_500;
/// Rows deleted from the small store, two pairs of neighbours.
const DELETED: [u64; 4] = [7, 8, 40, 41];
/// A row no store ever had.
const NEVER: u64 = 9_999;

fn near(id: u64) -> bool {
    id <= 80
}

fn noted(id: u64) -> bool {
    id % 3 != 0
}

/// Rows `1..=rows`, `size` being the row's number. `group` is "near" on rows
/// 1-80 and has an index; `note` is "x" on two rows in three and has none, so
/// a row is read to check it.
fn store(rows: u64) -> TestContext {
    let ctx = setup_with_config(Config::default().with_lsm_in_memory());
    ctx.collections.create_collection(NAME).unwrap();
    let storage = ctx.collections.get_collection(NAME).unwrap();
    storage
        .create_payload_index("group", PayloadIndexSchema::Keyword)
        .unwrap();
    let ids: Vec<u64> = (1..=rows).collect();
    for chunk in ids.chunks(1_500) {
        let points = chunk
            .iter()
            .map(|&id| {
                json!({"id": id, "payload": {
                    "group": if near(id) { "near" } else { "far" },
                    "note": if noted(id) { "x" } else { "y" },
                    "size": id
                }})
            })
            .collect::<Vec<_>>();
        upsert_points(&ctx, NAME, serde_json::Value::Array(points));
    }
    ctx
}

/// The small store: 120 rows less the deleted ones.
fn fixture() -> TestContext {
    let ctx = store(ROWS);
    post(&ctx, "delete", json!({"points": DELETED}));
    ctx
}

fn post(ctx: &TestContext, endpoint: &str, body: serde_json::Value) -> serde_json::Value {
    let input = make_input(
        ctx,
        "POST",
        &format!("/collections/{NAME}/points/{endpoint}"),
        serde_json::to_vec(&body).unwrap(),
        HashMap::from([("name".to_string(), NAME.to_string())]),
    );
    let mut response = Response::new();
    match endpoint {
        "scroll" => handle_scroll_points(&input, &mut response).unwrap(),
        "scan" => handle_scan_points(&input, &mut response).unwrap(),
        "delete" => handle_delete_points(&input, &mut response).unwrap(),
        other => panic!("no handler for {other}"),
    }
    assert_eq!(response.status, 200, "{endpoint} {body}");
    serde_json::from_slice(&response.body).unwrap()
}

/// A page: its row numbers and the offset it hands on, null after the last.
type Page = (Vec<u64>, serde_json::Value);

/// What a request's pages cost: the pages, the rows read and the round trips
/// to the store.
type Cost = (Vec<Page>, usize, usize);

/// Every page of `request` (its body without an offset) from row `first` on,
/// following the page offsets, with rows read `batch` at a time.
fn pages(
    ctx: &TestContext,
    endpoint: &str,
    request: &serde_json::Value,
    first: u64,
    batch: usize,
) -> Cost {
    let (reads, trips) = (node_reads_for_test(), lsm_round_trips_for_test());
    let mut pages = Vec::new();
    let mut offset = json!(first);
    with_scan_read_batch_for_test(batch, || {
        for _ in 0..1_000 {
            let mut body = request.clone();
            body["offset"] = offset.clone();
            let reply = post(ctx, endpoint, body);
            let rows = reply["result"]["points"]
                .as_array()
                .unwrap()
                .iter()
                .map(|point| point["payload"]["size"].as_u64().unwrap())
                .collect();
            offset = reply["result"]["next_page_offset"].clone();
            pages.push((rows, offset.clone()));
            if offset.is_null() {
                return;
            }
        }
        panic!("{endpoint} did not end");
    });
    (
        pages,
        node_reads_for_test() - reads,
        lsm_round_trips_for_test() - trips,
    )
}

fn rows_of(pages: &[Page]) -> Vec<u64> {
    pages.iter().flat_map(|(rows, _)| rows.clone()).collect()
}

fn eq(key: &str, value: &str) -> serde_json::Value {
    json!({"key": key, "match": {"value": value}})
}

fn scroll(limit: usize, filter: &serde_json::Value) -> (&'static str, serde_json::Value) {
    let request = json!({"limit": limit, "with_payload": true, "filter": filter});
    ("scroll", request)
}

fn scan(
    limit: usize,
    max_scan: usize,
    filter: &serde_json::Value,
) -> (&'static str, serde_json::Value) {
    let mut request = scroll(limit, filter).1;
    request["max_scan"] = json!(max_scan);
    ("scan", request)
}

/// A filter of the small store and the rows that match it.
type Shape = (&'static str, serde_json::Value, fn(u64) -> bool);

fn shape(name: &'static str, filter: serde_json::Value, matches: fn(u64) -> bool) -> Shape {
    (name, filter, matches)
}

fn shapes() -> Vec<Shape> {
    // Rows 1-60 and the row no store had: the list names the deleted rows.
    let listed: Vec<u64> = (1..=60).chain([NEVER]).collect();
    vec![
        // The candidates are the matches.
        shape(
            "an index's rows",
            json!({"must": [eq("group", "near")]}),
            near,
        ),
        // One candidate in three is read and rejected.
        shape(
            "rows the recheck rejects",
            json!({"must": [eq("group", "near"), eq("note", "x")]}),
            |id| near(id) && noted(id),
        ),
        // The list is the candidate set: five of its rows are not there.
        shape(
            "a list that names deleted rows",
            json!({"must": [{"has_id": listed}, eq("note", "x")]}),
            |id| id <= 60 && noted(id),
        ),
    ]
}

/// Every page, its rows and the offset it hands on, is the same read in
/// batches of 2, 4, 5 and 96 rows as read one row at a time: at page limits
/// under, at and over a batch, where a batch holds deleted rows, and where a
/// `/points/scan` budget ends the page inside a batch. The rows are the ones
/// the store has that match. A batch reads no row the page would not have
/// read, so none past the page's end, and a row the store does not have is
/// read once, as it was: the rows read are the same.
#[test]
fn pages_and_rows_read_are_the_same_in_batches_as_one_row_at_a_time() {
    let ctx = fixture();
    for (name, filter, matches) in shapes() {
        let expected: Vec<u64> = (1..=ROWS)
            .filter(|id| !DELETED.contains(id) && matches(*id))
            .collect();
        let requests = [
            scroll(1, &filter),
            scroll(3, &filter),
            scroll(5, &filter),
            scroll(100, &filter),
            // A budget the list fits under: its rows are the candidates.
            scan(3, 100, &filter),
            // The budget ends a page before its limit, inside a batch. (A
            // list that does not fit under the budget is a walk, with no
            // candidate rows to read.)
            scan(100, 7, &filter),
            scan(100, 5, &filter),
            scan(3, 4, &filter),
        ];
        for (endpoint, request) in requests {
            let case = format!(
                "{name}: {endpoint} of {}, max_scan {}",
                request["limit"], request["max_scan"]
            );
            let (one_by_one, reads, _) = pages(&ctx, endpoint, &request, 1, 1);
            assert_eq!(rows_of(&one_by_one), expected, "{case}: rows");
            for batch in [2, 4, 5, 96] {
                let (batched, batched_reads, _) = pages(&ctx, endpoint, &request, 1, batch);
                assert_eq!(batched, one_by_one, "{case}: pages in batches of {batch}");
                assert_eq!(
                    batched_reads, reads,
                    "{case}: rows read in batches of {batch}"
                );
            }
        }
    }
}

/// Pages of 3, 100, 800 and 6,400 rows of a store of 4,500, with a filter
/// whose candidates all match and with one that rejects a row in three: read
/// in batches of 96, every page and the rows read are those of one row at a
/// time. A scroll page ends at 2 s, and a page of 4,096 rows read one at a
/// time in a test build is too near that for a test, so the two large limits
/// go through `/points/scan` with a scroll's largest budget: the same loop
/// without the clock.
#[test]
fn pages_of_3_100_800_and_6400_rows_are_the_same_in_batches() {
    let ctx = store(LARGE_ROWS);
    let no_rejects = json!({"must": [eq("group", "far")]});
    let rejects = json!({"must": [eq("group", "far"), eq("note", "x")]});
    for (name, filter, rejecting) in [
        ("far", &no_rejects, false),
        ("far and noted", &rejects, true),
    ] {
        // The request and the row it starts at: the small pages over the
        // store's last rows, to keep this short.
        let requests = [
            (scroll(3, filter), LARGE_ROWS - 59),
            (scroll(100, filter), LARGE_ROWS - 999),
            (scan(800, 4_096, filter), 1),
            (scan(6_400, 4_096, filter), 1),
        ];
        for ((endpoint, request), first) in requests {
            let case = format!(
                "{name}: {endpoint} of {} from row {first}",
                request["limit"]
            );
            let expected: Vec<u64> = (first..=LARGE_ROWS)
                .filter(|id| !near(*id) && (!rejecting || noted(*id)))
                .collect();
            let (one_by_one, reads, trips) = pages(&ctx, endpoint, &request, first, 1);
            assert_eq!(rows_of(&one_by_one), expected, "{case}: rows");
            let (batched, batched_reads, batched_trips) =
                pages(&ctx, endpoint, &request, first, 96);
            assert_eq!(batched, one_by_one, "{case}: pages");
            assert_eq!(batched_reads, reads, "{case}: rows read");
            assert!(
                batched_trips < trips,
                "{case}: {batched_trips} round trips in batches, {trips} one row at a time"
            );
        }
    }
}

/// The 76 rows of a page take one read in a batch of 96 and four in batches
/// of 20, where they took 76.
#[test]
fn a_page_reads_its_rows_in_batches() {
    let ctx = fixture();
    let (endpoint, request) = scroll(100, &json!({"must": [eq("group", "near")]}));
    let (one_by_one, reads, trips) = pages(&ctx, endpoint, &request, 1, 1);
    assert_eq!((one_by_one.len(), reads), (1, 76));
    for (batch, batches) in [(96, 1), (20, 4)] {
        let (batched, batched_reads, batched_trips) = pages(&ctx, endpoint, &request, 1, batch);
        assert_eq!(batched, one_by_one);
        assert_eq!(batched_reads, 76);
        assert_eq!(
            trips - batched_trips,
            76 - batches,
            "round trips in batches of {batch}: {batched_trips}, one at a time {trips}"
        );
    }
}

/// A candidate row that does not decode fails the page read in a batch with
/// the error it failed with read on its own; a row that is not there is
/// skipped either way (the pages tests above).
#[test]
fn a_row_that_does_not_decode_fails_the_page_as_it_did_read_alone() {
    let ctx = fixture();
    let storage = ctx.collections.get_collection(NAME).unwrap();
    // Row 20 stays a candidate of the group's index and holds no node now.
    let mut write = storage.backend.begin_write().unwrap();
    storage
        .backend
        .put(
            &mut write,
            Namespace::Nodes,
            &20u128.to_be_bytes(),
            b"not a node",
        )
        .unwrap();
    storage.backend.commit(write).unwrap();

    let failure = |batch: usize| {
        let body = scroll(100, &json!({"must": [eq("group", "near")]})).1;
        let input = make_input(
            &ctx,
            "POST",
            &format!("/collections/{NAME}/points/scroll"),
            serde_json::to_vec(&body).unwrap(),
            HashMap::from([("name".to_string(), NAME.to_string())]),
        );
        let mut response = Response::new();
        with_scan_read_batch_for_test(batch, || handle_scroll_points(&input, &mut response))
            .expect_err("a page over a row that does not decode")
            .to_string()
    };
    let alone = failure(1);
    for batch in [2, 96] {
        assert_eq!(failure(batch), alone, "in batches of {batch}");
    }
}
