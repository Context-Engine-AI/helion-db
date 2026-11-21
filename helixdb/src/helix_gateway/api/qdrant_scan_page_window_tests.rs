//! A scan page whose candidates are one keyword's rows reads them from its
//! offset as it goes (`scan_page_window`). Before, every page counted the
//! keyword's whole list, read it into a set and sorted it, to scan one page of
//! it. Each test here compares a page with the same page planned as before,
//! which is the page with the switch off.

use super::hybrid_parallel_tests::EnvVar;
use super::tests::{make_input, setup_with_config, upsert_points, TestContext};
use super::*;
use crate::helix_engine::graph_core::config::Config;
use crate::helix_engine::storage_core::backend::{LsmRole, LsmStorage, StorageBackendConfig};
use crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store;
use crate::helix_engine::storage_core::backend_lsm::{allow_lsm_blocking, dup_scan_rows_for_test};
use crate::helix_engine::storage_core::storage_core::node_reads_for_test;
use serde_json::json;
use std::sync::Arc;

const NAME: &str = "page_window";
const ROWS: u64 = 200;
/// A page's row budget, where a test gives no other.
const BUDGET: usize = 1_024;
/// Fewer than the `name` index has rows, so that a text walk of it gives up.
const WALK_CAP: usize = 20;

/// Row `id` of `rows`. `all` is one value on every row, `kind` is "calls" on
/// nine rows in ten, and `late` is "yes" past the first fifth of the rows.
/// `name` is the row's own. `num` is one number, written as an integer on the
/// even rows and as a float on the odd ones. `note` has no index.
fn point(id: u64, rows: u64, kind: &str) -> serde_json::Value {
    json!({"id": id, "payload": {
        "all": "yes",
        "kind": kind,
        "late": if id > rows / 5 { "yes" } else { "no" },
        "name": format!("sym_{id:04}"),
        "num": if id % 2 == 0 { json!(7) } else { json!(7.0) },
        "size": id,
        "note": if id % 3 == 0 { "x" } else { "y" }
    }})
}

fn kind(id: u64) -> &'static str {
    if id % 10 == 0 {
        "imports"
    } else {
        "calls"
    }
}

/// The `calls` rows of a collection of `rows`.
fn calls(rows: u64) -> Vec<u128> {
    (1..=rows)
        .filter(|id| kind(*id) == "calls")
        .map(u128::from)
        .collect()
}

fn fixture_with(config: Config, rows: u64) -> (TestContext, Arc<HelixGraphStorage>) {
    let ctx = setup_with_config(config);
    ctx.collections.create_collection(NAME).unwrap();
    let storage = ctx.collections.get_collection(NAME).unwrap();
    for field in ["all", "kind", "late", "name", "num"] {
        storage
            .create_payload_index(field, PayloadIndexSchema::Keyword)
            .unwrap();
    }
    storage
        .create_payload_index("size", PayloadIndexSchema::Integer)
        .unwrap();
    let points = (1..=rows)
        .map(|id| point(id, rows, kind(id)))
        .collect::<Vec<_>>();
    upsert_points(&ctx, NAME, serde_json::Value::Array(points));
    (ctx, storage)
}

fn fixture(rows: u64) -> (TestContext, Arc<HelixGraphStorage>) {
    let mut config = Config::default().with_lsm_in_memory();
    config.vector_config.flat_scan_threshold = Some(0);
    fixture_with(config, rows)
}

fn eq(key: &str, value: &str) -> serde_json::Value {
    json!({"key": key, "match": {"value": value}})
}

fn filter(body: serde_json::Value) -> Filter {
    serde_json::from_value(body).unwrap()
}

struct Page {
    ids: Vec<u128>,
    next: Option<String>,
    plan: &'static str,
    candidates: Option<usize>,
    scanned: usize,
    /// Index rows read to plan and serve it (the LSM writer counts them).
    index_rows: usize,
    /// Point rows read to serve it.
    node_reads: usize,
}

/// One scan page on a read of its own, as each page of a scroll is, with the
/// window switched on or off.
fn page_with(
    storage: &HelixGraphStorage,
    filter: &Filter,
    limit: usize,
    offset: u128,
    max_scan: Option<usize>,
    partition: Option<ScanPartition>,
    window: bool,
) -> Page {
    let read = storage.backend.begin_read().unwrap();
    page_on(
        storage, &read, filter, limit, offset, max_scan, partition, window,
    )
}

/// [`page_with`] on a read the caller began.
#[allow(clippy::too_many_arguments)]
fn page_on(
    storage: &HelixGraphStorage,
    read: &AnyRead<'_>,
    filter: &Filter,
    limit: usize,
    offset: u128,
    max_scan: Option<usize>,
    partition: Option<ScanPartition>,
    window: bool,
) -> Page {
    let opts = PointScanOptions {
        collection: NAME,
        limit,
        offset_id: offset,
        filter,
        with_payload: &WithPayload::None,
        with_vectors: &WithVectors::None,
        partition,
        max_scan,
        deadline: None,
        indexed_deadline: None,
        filter_hash: scan_filter_hash(filter),
        cursor_generation: None,
        fuse_pending: false,
    };
    let before = (dup_scan_rows_for_test(), node_reads_for_test());
    let scan =
        with_scan_page_window_for_test(window, || execute_point_scan_be(storage, read, &opts))
            .unwrap();
    Page {
        ids: scan.points.iter().filter_map(point_json_id).collect(),
        next: scan.next_offset,
        plan: scan.plan.as_str(),
        candidates: scan.index_candidates,
        scanned: scan.scanned,
        index_rows: dup_scan_rows_for_test() - before.0,
        node_reads: node_reads_for_test() - before.1,
    }
}

fn page(
    storage: &HelixGraphStorage,
    filter: &Filter,
    limit: usize,
    offset: u128,
    window: bool,
) -> Page {
    page_with(storage, filter, limit, offset, Some(BUDGET), None, window)
}

/// A page's rows and the offset it hands on.
type Rows = (Vec<u128>, Option<String>);

/// Every page of a scan from the first row on, following the offsets.
fn pages(
    storage: &HelixGraphStorage,
    filter: &Filter,
    limit: usize,
    max_scan: usize,
    window: bool,
) -> Vec<Rows> {
    let mut pages = Vec::new();
    let mut offset = 0u128;
    for _ in 0..10_000 {
        let page = page_with(storage, filter, limit, offset, Some(max_scan), None, window);
        pages.push((page.ids, page.next.clone()));
        match page.next {
            Some(next) => offset = parse_scroll_offset(Some(&serde_json::Value::String(next))),
            None => return pages,
        }
    }
    panic!("the scan did not end");
}

fn rows_of(pages: &[Rows]) -> Vec<u128> {
    pages.iter().flat_map(|(ids, _)| ids.clone()).collect()
}

fn offset_of(id: u128) -> Option<String> {
    Some(format_point_id(id))
}

#[test]
fn a_page_reads_its_window_of_the_keywords_rows_whatever_the_lists_length() {
    let all = filter(json!({"must": [eq("all", "yes")]}));
    let mut read = Vec::new();
    for rows in [ROWS, 4 * ROWS] {
        let (_ctx, storage) = fixture(rows);
        // Ten rows from the middle of the list.
        let from = u128::from(rows / 2);
        let window = page(&storage, &all, 10, from, true);
        assert_eq!(
            window.ids,
            (from..from + 10).collect::<Vec<_>>(),
            "{rows} rows"
        );
        assert_eq!(window.plan, "indexed_window", "{rows} rows");

        // The plan before counts the list, then reads it into a set: every
        // row of it twice, for the same ten rows and the same offset.
        let before = page(&storage, &all, 10, from, false);
        assert_eq!(before.plan, "indexed_candidates", "{rows} rows");
        assert_eq!(before.index_rows, 2 * rows as usize, "{rows} rows");
        assert_eq!(
            (&window.ids, &window.next),
            (&before.ids, &before.next),
            "{rows} rows"
        );
        assert_eq!(window.next, offset_of(from + 10), "{rows} rows");

        assert!(window.index_rows <= BUDGET, "{}", window.index_rows);
        read.push(window.index_rows);
    }
    // The first read of a window is a batch of rows, whatever the list holds.
    assert_eq!(read, [scan_read_batch(), scan_read_batch()]);
}

/// What the plan before reads, with the switch off. This test holds with and
/// without the window in the tree: the switch's off state is the plan before.
#[test]
fn with_the_switch_off_a_page_is_planned_from_the_whole_list_as_before() {
    let (_ctx, storage) = fixture(ROWS);
    let page = page(
        &storage,
        &filter(json!({"must": [eq("kind", "calls")]})),
        10,
        0,
        false,
    );
    assert_eq!(page.ids, calls(ROWS)[..10]);
    assert_eq!(page.next, offset_of(12));
    assert_eq!(page.plan, "indexed_candidates");
    assert_eq!(page.candidates, Some(calls(ROWS).len()));
    assert_eq!(page.scanned, 10);
    assert_eq!(page.index_rows, 2 * calls(ROWS).len(), "counted, then read");
}

#[test]
fn a_window_ends_with_its_list_and_hands_on_no_offset() {
    let (_ctx, storage) = fixture(ROWS);
    let filter = filter(json!({"must": [eq("kind", "calls")]}));
    for limit in [1, 7, 10, 96, 97, 500] {
        let window = pages(&storage, &filter, limit, BUDGET, true);
        assert_eq!(rows_of(&window), calls(ROWS), "pages of {limit}");
        assert_eq!(
            window,
            pages(&storage, &filter, limit, BUDGET, false),
            "pages of {limit}"
        );
    }
    // A budget under the page: each page ends at its budget, as before.
    for budget in [1, 5, 96, 97] {
        let window = pages(&storage, &filter, 50, budget, true);
        assert_eq!(rows_of(&window), calls(ROWS), "a budget of {budget}");
        assert_eq!(
            window,
            pages(&storage, &filter, 50, budget, false),
            "a budget of {budget}"
        );
    }
    // No more index rows than the page may scan.
    let page = page_with(&storage, &filter, 50, 0, Some(5), None, true);
    assert_eq!(
        (page.ids.len(), page.scanned, page.index_rows),
        (5, 5, 5),
        "a budget of 5"
    );
}

/// The window is a list of candidates and no more: each row it names is read
/// and judged against the whole filter before it is returned, as a row of the
/// whole list is.
#[test]
fn an_index_row_whose_point_no_longer_has_the_value_puts_no_row_in_a_page() {
    let (_ctx, storage) = fixture(ROWS);
    let filter = filter(json!({"must": [eq("kind", "calls")]}));
    // Row 11 is given row 10's payload, an import, with no index row
    // written: the index goes on listing it under "calls".
    let read = storage.backend.begin_read().unwrap();
    let an_import = storage
        .backend
        .get_with(&read, Namespace::Nodes, &10u128.to_be_bytes(), |row| {
            row.map(<[u8]>::to_vec)
        })
        .unwrap()
        .unwrap();
    drop(read);
    let mut write = storage.backend.begin_write().unwrap();
    storage
        .backend
        .put(
            &mut write,
            Namespace::Nodes,
            &11u128.to_be_bytes(),
            &an_import,
        )
        .unwrap();
    storage.backend.commit(write).unwrap();
    let read = storage.backend.begin_read().unwrap();
    let listed = storage
        .get_nodes_by_payload_value_be(&read, "kind", &match_lookup_value(&json!("calls")))
        .unwrap();
    assert!(listed.contains(&11), "the index row is still there");
    drop(read);

    let window = page(&storage, &filter, 10, 0, true);
    assert_eq!(window.plan, "indexed_window");
    assert_eq!(window.ids, [1, 2, 3, 4, 5, 6, 7, 8, 9, 12]);
    assert_eq!(window.scanned, 11, "row 11 was read and turned down");
    let before = page(&storage, &filter, 10, 0, false);
    assert_eq!((&window.ids, &window.next), (&before.ids, &before.next));
}

/// An index made over points that are already stored lists every one of
/// them once it is ready, and until then the planner does not use it: a read
/// begun before the build ended walks the rows. Made by the storage call,
/// which builds it before it returns, and through the route, which hands the
/// build to the index executor.
#[test]
fn an_index_made_over_stored_points_gives_the_window_the_walks_rows() {
    for through_the_route in [false, true] {
        an_index_made_over_stored_points(through_the_route);
    }
}

fn an_index_made_over_stored_points(through_the_route: bool) {
    let mut config = Config::default().with_lsm_in_memory();
    config.vector_config.flat_scan_threshold = Some(0);
    let ctx = setup_with_config(config);
    ctx.collections.create_collection(NAME).unwrap();
    let storage = ctx.collections.get_collection(NAME).unwrap();
    let points = (1..=ROWS)
        .map(|id| point(id, ROWS, kind(id)))
        .collect::<Vec<_>>();
    upsert_points(&ctx, NAME, serde_json::Value::Array(points));
    let filter = filter(json!({"must": [eq("kind", "calls")]}));

    // No index: the walk, which reads every row's payload.
    let walk = pages(&storage, &filter, 10, BUDGET, true);
    assert_eq!(rows_of(&walk), calls(ROWS));
    assert_eq!(page(&storage, &filter, 10, 0, true).plan, "primary");

    let begun_before = storage.backend.begin_read().unwrap();
    if through_the_route {
        let body = json!({"field_name": "kind", "field_schema": "keyword"});
        let input = make_input(
            &ctx,
            "PUT",
            &format!("/collections/{NAME}/index"),
            serde_json::to_vec(&body).unwrap(),
            HashMap::from([("name".to_string(), NAME.to_string())]),
        );
        let mut response = Response::new();
        handle_create_index(&input, &mut response).unwrap();
        assert_eq!(response.status, 200);
        let asked = Instant::now();
        while storage.has_payload_index("kind").is_none() {
            assert!(
                asked.elapsed() < Duration::from_secs(10),
                "the index was not built"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    } else {
        storage
            .create_payload_index("kind", PayloadIndexSchema::Keyword)
            .unwrap();
    }
    // A read begun before the index was ready is not given it.
    let old = page_on(
        &storage,
        &begun_before,
        &filter,
        10,
        0,
        Some(BUDGET),
        None,
        true,
    );
    assert_eq!(old.plan, "primary");
    drop(begun_before);

    let window = pages(&storage, &filter, 10, BUDGET, true);
    assert_eq!(page(&storage, &filter, 10, 0, true).plan, "indexed_window");
    assert_eq!(window, walk);
}

/// The most a page costs where the conditions beside the keyword turn every
/// row down: its budget in index rows and its budget in point rows, whatever
/// the list holds. The page ends there and hands on an offset.
#[test]
fn a_page_that_turns_every_row_down_reads_no_more_than_its_budget() {
    let (_ctx, storage) = fixture(ROWS);
    let none = filter(json!({"must": [eq("kind", "calls"), eq("note", "never")]}));
    for budget in [1, 40, 96, 100] {
        let page = page_with(&storage, &none, 10, 0, Some(budget), None, true);
        assert_eq!(page.plan, "indexed_window", "a budget of {budget}");
        assert_eq!(
            (
                page.ids.len(),
                page.scanned,
                page.index_rows,
                page.node_reads
            ),
            (0, budget, budget, budget),
            "a budget of {budget}"
        );
        // The offset is the row after the last one read.
        assert_eq!(
            page.next,
            offset_of(calls(ROWS)[budget - 1] + 1),
            "a budget of {budget}"
        );
    }
    // A budget over the list: the list's rows and no more, and no offset.
    let page = page(&storage, &none, 10, 0, true);
    let listed = calls(ROWS).len();
    assert_eq!(
        (page.scanned, page.index_rows, page.node_reads, page.next),
        (listed, listed, listed, None)
    );
}

/// A page whose clock ran out before its first row reads that row alone,
/// from the index as from the points, and hands on its offset.
#[test]
fn a_page_out_of_time_reads_one_index_row_and_one_point() {
    let (_ctx, storage) = fixture(ROWS);
    let filter = filter(json!({"must": [eq("kind", "calls")]}));
    let read = storage.backend.begin_read().unwrap();
    let opts = PointScanOptions {
        collection: NAME,
        limit: 10,
        offset_id: 0,
        filter: &filter,
        with_payload: &WithPayload::None,
        with_vectors: &WithVectors::None,
        partition: None,
        max_scan: Some(BUDGET),
        deadline: None,
        indexed_deadline: Some(Instant::now()),
        filter_hash: scan_filter_hash(&filter),
        cursor_generation: None,
        fuse_pending: false,
    };
    let before = (dup_scan_rows_for_test(), node_reads_for_test());
    let scan =
        with_scan_page_window_for_test(true, || execute_point_scan_be(&storage, &read, &opts))
            .unwrap();
    assert_eq!(scan.plan.as_str(), "indexed_window");
    assert_eq!(
        (scan.returned, scan.scanned, scan.budget_exhausted),
        (1, 1, true)
    );
    assert_eq!(scan.next_offset, offset_of(2));
    assert_eq!(
        (
            dup_scan_rows_for_test() - before.0,
            node_reads_for_test() - before.1
        ),
        (1, 1),
        "index rows and point rows read"
    );
}

fn post(
    ctx: &TestContext,
    endpoint: &str,
    body: serde_json::Value,
    window: bool,
) -> serde_json::Value {
    let input = make_input(
        ctx,
        "POST",
        &format!("/collections/{NAME}/points/{endpoint}"),
        serde_json::to_vec(&body).unwrap(),
        HashMap::from([("name".to_string(), NAME.to_string())]),
    );
    let mut response = Response::new();
    with_scan_page_window_for_test(window, || match endpoint {
        "scroll" => handle_scroll_points(&input, &mut response),
        "delete" => handle_delete_points(&input, &mut response),
        other => panic!("no handler for {other}"),
    })
    .unwrap();
    assert_eq!(response.status, 200, "{endpoint} {body}");
    serde_json::from_slice(&response.body).unwrap()
}

/// A scroll of the `calls` rows ten at a time through the handler, with rows
/// written between its first pages: each page's rows and the offset it hands
/// on.
fn scroll_written_to(window: bool) -> Vec<(Vec<u128>, serde_json::Value)> {
    let (ctx, _storage) = fixture(ROWS);
    let mut pages = Vec::new();
    let mut offset = serde_json::Value::Null;
    for turn in 0..1_000 {
        let mut body = json!({
            "filter": {"must": [eq("kind", "calls")]}, "limit": 10, "with_payload": false});
        if !offset.is_null() {
            body["offset"] = offset.clone();
        }
        let reply = post(&ctx, "scroll", body, window);
        let rows = reply["result"]["points"]
            .as_array()
            .unwrap()
            .iter()
            .map(|point| point_json_id(point).unwrap())
            .collect::<Vec<_>>();
        offset = reply["result"]["next_page_offset"].clone();
        pages.push((rows, offset.clone()));
        match turn {
            // Ahead of the second page: two rows go, one stops being a call,
            // one that was none becomes one, and a row is added at the end.
            0 => {
                post(&ctx, "delete", json!({"points": [13, 14]}), window);
                let written = json!([
                    point(15, ROWS, "imports"),
                    point(20, ROWS, "calls"),
                    point(ROWS + 7, ROWS, "calls")
                ]);
                upsert_points(&ctx, NAME, written);
            }
            // Behind the third page: a row already returned goes, and a row
            // the scroll has passed becomes a call.
            1 => {
                post(&ctx, "delete", json!({"points": [3]}), window);
                upsert_points(&ctx, NAME, json!([point(10, ROWS, "calls")]));
            }
            _ => {}
        }
        if offset.is_null() {
            return pages;
        }
    }
    panic!("the scroll did not end");
}

#[test]
fn a_scroll_written_to_between_its_pages_gives_the_pages_of_the_plan_before() {
    let window = scroll_written_to(true);
    assert_eq!(window, scroll_written_to(false));

    // A page is a read of its own: what was written ahead of it is in it,
    // and what was written behind it is not.
    assert_eq!(window[0].0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]);
    assert_eq!(window[1].0, [12, 16, 17, 18, 19, 20, 21, 22, 23, 24]);
    let mut expected: Vec<u128> = (1..=ROWS)
        .filter(|id| kind(*id) == "calls" && ![13, 14, 15].contains(id))
        .chain([20, ROWS + 7])
        .map(u128::from)
        .collect();
    expected.sort_unstable();
    let rows: Vec<u128> = window.iter().flat_map(|(rows, _)| rows.clone()).collect();
    assert_eq!(rows, expected);
}

#[test]
fn a_text_condition_keeps_the_window_once_its_fields_walk_is_known_to_give_up() {
    let (ctx, storage) = fixture(ROWS);
    let capped = |filter: &Filter, window: bool| {
        with_text_walk_max_for_test(WALK_CAP, || page(&storage, filter, 10, 0, window))
    };
    // "sym_01" is in the names of rows 100 to 199.
    let text = json!({"key": "name", "match": {"text": "sym_01"}});
    let both = filter(json!({"must": [text.clone(), eq("all", "yes")]}));
    let matched: Vec<u128> = (100..110).collect();

    // No walk of `name` has given up in this snapshot, so the planner may
    // yet take the text's rows: the page is planned as before.
    let unmarked = capped(&both, true);
    assert_eq!(unmarked.plan, "indexed_candidates");
    assert_eq!(unmarked.ids, matched);

    // A page of the text alone walks the field and gives up.
    let alone = capped(&filter(json!({"must": [text]})), true);
    assert_eq!((alone.plan, &alone.ids), ("primary", &matched));

    // Now the planner takes nothing from the text: the keyword's window,
    // with no walk and no count.
    let marked = capped(&both, true);
    assert_eq!(marked.plan, "indexed_window");
    let before = capped(&both, false);
    assert_eq!(before.plan, "indexed_candidates");
    assert_eq!((&marked.ids, &marked.next), (&before.ids, &before.next));
    assert_eq!((&marked.ids, &marked.next), (&unmarked.ids, &unmarked.next));
    assert_eq!(marked.scanned, before.scanned);
    // The plan before counts the list to look for a selective condition,
    // counts it again to see that it is not too broad, and reads it.
    assert_eq!(before.index_rows, 3 * ROWS as usize);
    assert!(
        marked.index_rows <= marked.scanned.next_multiple_of(scan_read_batch()) * 2,
        "{} index rows for {} rows scanned",
        marked.index_rows,
        marked.scanned
    );

    // The mark is of a walk under that cap. Under one the field's rows fit,
    // a walk would not give up: the page is planned as before.
    let roomy = page(&storage, &both, 10, 0, true);
    assert_eq!(roomy.plan, "indexed_candidates");
    assert_eq!(roomy.ids, matched);

    // A write is a new snapshot, where nothing is known of the field's walk.
    upsert_points(&ctx, NAME, json!([point(ROWS + 1, ROWS, "calls")]));
    assert_eq!(capped(&both, true).plan, "indexed_candidates");
}

#[test]
fn only_a_filter_whose_candidates_are_one_keywords_rows_takes_the_window() {
    let (_ctx, storage) = fixture(ROWS);
    let calls = eq("kind", "calls");
    // The filter, and whether its pages take the window.
    let cases = [
        (json!({"must": [calls]}), true),
        // Nothing is taken from a field with no index, an `is_empty`, an
        // `is_null` or a `min_should`.
        (json!({"must": [calls, eq("note", "x")]}), true),
        (
            json!({"must": [{"is_empty": {"key": "gone"}}, calls]}),
            true,
        ),
        (json!({"must": [calls, {"is_null": {"key": "gone"}}]}), true),
        (
            json!({"must": [calls], "min_should": {
                "conditions": [eq("note", "x"), eq("late", "no")], "min_count": 1}}),
            true,
        ),
        // A second list, several values, a range, a list of ids, a nested
        // filter, an exclusion and a `should` give candidates of their own.
        (json!({"must": [calls, eq("late", "yes")]}), false),
        (
            json!({"must": [{"key": "kind", "match": {"any": ["calls", "imports"]}}]}),
            false,
        ),
        (
            json!({"must": [calls, {"key": "size", "range": {"gte": 50}}]}),
            false,
        ),
        (json!({"must": [calls, {"has_id": [5, 6, 7, 300]}]}), false),
        (
            json!({"must": [calls, {"must": [eq("late", "yes")]}]}),
            false,
        ),
        (
            json!({"must": [calls], "must_not": [eq("late", "no")]}),
            false,
        ),
        (
            json!({"must": [calls], "should": [eq("late", "yes")]}),
            false,
        ),
        // A text condition on a field no walk of which has given up here.
        (
            json!({"must": [calls, {"key": "name", "match": {"text": "sym_01"}}]}),
            false,
        ),
        // No keyword condition on an indexed field.
        (json!({"must": [eq("note", "x")]}), false),
    ];
    for (body, takes) in cases {
        let filter = filter(body.clone());
        let window = pages(&storage, &filter, 7, 40, true);
        assert_eq!(window, pages(&storage, &filter, 7, 40, false), "{body}");
        let plan = page_with(&storage, &filter, 7, 0, Some(40), None, true).plan;
        assert_eq!(plan == "indexed_window", takes, "{body}: {plan}");
    }
}

/// A list over half the collection (and over the cap, 100,000 rows where it
/// is not lowered) was too broad for the plan before: it counted the list as
/// far as the cap and then walked every row from the offset.
#[test]
fn a_list_too_broad_for_the_plan_before_is_read_by_its_window() {
    let (_ctx, storage) = fixture(ROWS);
    // `late` is "yes" on rows 41 to 200.
    let late = filter(json!({"must": [eq("late", "yes")]}));
    let scroll = |window: bool| {
        with_scan_index_candidate_max_for_test(50, || pages(&storage, &late, 10, 20, window))
    };
    let (window, before) = (scroll(true), scroll(false));

    // The plan before spends two pages' budgets on the forty rows that are
    // not in the list, and returns none of them.
    assert_eq!(before[0], (vec![], offset_of(21)));
    assert_eq!(before[1], (vec![], offset_of(41)));
    assert_eq!(before[2], ((41..=50).collect(), offset_of(51)));
    let plan = with_scan_index_candidate_max_for_test(50, || {
        page_with(&storage, &late, 10, 0, Some(20), None, false).plan
    });
    assert_eq!(plan, "primary");

    // The window scans rows of the list alone: the same rows in the same
    // order, in fewer pages.
    assert_eq!(window[0], ((41..=50).collect(), offset_of(51)));
    assert_eq!(rows_of(&window), (41..=200).collect::<Vec<u128>>());
    assert_eq!(rows_of(&window), rows_of(&before));
    assert_eq!(before.len(), window.len() + 2);
}

#[test]
fn a_window_ends_where_its_partition_does() {
    let (ctx, storage) = fixture(30);
    // Three rows in the upper half of the id space, the second of two
    // partitions.
    let upper: Vec<u128> = (1..=3u128).map(|n| (8u128 << 124) | n).collect();
    let points = upper
        .iter()
        .map(|id| json!({"id": format_point_id(*id), "payload": {"all": "yes"}}))
        .collect::<Vec<_>>();
    upsert_points(&ctx, NAME, serde_json::Value::Array(points));
    let all = filter(json!({"must": [eq("all", "yes")]}));
    for (index, expected) in [(0, (1..=30).collect::<Vec<u128>>()), (1, upper)] {
        let partition = Some(ScanPartition { index, total: 2 });
        let window = page_with(&storage, &all, 100, 0, None, partition, true);
        assert_eq!(window.plan, "indexed_window", "partition {index}");
        assert_eq!(window.ids, expected, "partition {index}");
        let before = page_with(&storage, &all, 100, 0, None, partition, false);
        assert_eq!(
            (&window.ids, &window.next),
            (&before.ids, &before.next),
            "partition {index}"
        );
    }
}

/// The window read against the whole list it is a window of.
fn windows_are_slices(storage: &HelixGraphStorage, rows: u64) {
    let read = storage.backend.begin_read().unwrap();
    for (field, value, listed) in [
        ("kind", json!("calls"), calls(rows).len()),
        // An integer on some rows and a float on the others: two lists, read
        // as one.
        ("num", json!(7), rows as usize),
    ] {
        let value = match_lookup_value(&value);
        let list = storage
            .get_nodes_by_payload_value_be(&read, field, &value)
            .unwrap();
        assert_eq!(list.len(), listed, "{field}");
        let listed_rows = storage.payload_value_rows(field, &value).unwrap();
        for from in [
            0,
            1,
            7,
            u128::from(rows / 2),
            u128::from(rows),
            u128::from(rows) + 1,
        ] {
            for max in [1, 5, 96, 10_000] {
                let window = storage
                    .payload_value_ids_from_be(&read, &listed_rows, from, max)
                    .unwrap();
                let slice: Vec<u128> = list
                    .iter()
                    .copied()
                    .filter(|id| *id >= from)
                    .take(max)
                    .collect();
                assert_eq!(window, slice, "{field} from {from}, {max} rows");
            }
        }
    }
}

#[test]
fn a_window_is_a_slice_of_the_list_on_the_writer_and_on_lmdb() {
    let (_ctx, lsm) = fixture(ROWS);
    windows_are_slices(&lsm, ROWS);
    // The writer reads the window's rows and none before them.
    let read = lsm.backend.begin_read().unwrap();
    let before = dup_scan_rows_for_test();
    let all = lsm
        .payload_value_rows("all", &match_lookup_value(&json!("yes")))
        .unwrap();
    let window = lsm.payload_value_ids_from_be(&read, &all, 150, 5).unwrap();
    assert_eq!(window, [150, 151, 152, 153, 154]);
    assert_eq!(dup_scan_rows_for_test() - before, 5);

    // LMDB has no read from a value: it reads the list and leaves out what
    // is before the window.
    let (_ctx, lmdb) = fixture_with(Config::default(), ROWS);
    assert!(lmdb.backend.kind() != BackendKind::Lsm);
    windows_are_slices(&lmdb, ROWS);
}

#[test]
#[serial_test::serial]
fn a_reader_replica_reads_the_same_windows() {
    let _store = set_lsm_test_object_store(Arc::new(object_store::memory::InMemory::new()));
    let _prefix = EnvVar::set("HELIX_LSM_PREFIX", "scan_page_window");
    let config = |role| {
        Config::default().with_storage_backend(StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role,
        })
    };
    let (_writer, written) = fixture_with(config(LsmRole::Writer), ROWS);
    assert!(!written.backend.is_reader_replica());
    let reader = setup_with_config(config(LsmRole::Reader));
    let replica = reader.collections.get_collection(NAME).unwrap();
    assert!(replica.backend.is_reader_replica());
    let filter = filter(json!({"must": [eq("kind", "calls")]}));

    // As the gateway runs a handler, and as production's readers read: each
    // request on a snapshot of its own.
    let _pinned = EnvVar::set("HELIX_LSM_READER_SNAPSHOT_READS", "1");
    allow_lsm_blocking(|| {
        assert!(replica.backend.begin_read().unwrap().is_snapshot_pinned());
        windows_are_slices(&replica, ROWS);
        assert_eq!(page(&replica, &filter, 10, 0, true).plan, "indexed_window");
        let window = pages(&replica, &filter, 10, BUDGET, true);
        assert_eq!(rows_of(&window), calls(ROWS));
        assert_eq!(window, pages(&replica, &filter, 10, BUDGET, false));

        // A read of the latest state may see a refresh between two reads of
        // the list: it is planned as before, from one read of it.
        let latest = AnyRead::LsmReader(None);
        assert!(!latest.is_snapshot_pinned());
        let page = page_on(&replica, &latest, &filter, 10, 0, Some(BUDGET), None, true);
        assert_eq!(page.plan, "indexed_candidates");
        assert_eq!(page.ids, calls(ROWS)[..10]);
    });
}
