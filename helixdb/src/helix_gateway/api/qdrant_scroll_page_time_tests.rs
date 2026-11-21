//! The clock of a scroll page served from indexed candidates: its scan has a
//! budget of its own, counted from the end of the page's planning, and the
//! page a wall counted from its start (`indexed_scan_deadline`). And the
//! clock of a page that walks the rows: its deadline is put off by the time
//! its planning took, under the same wall (`primary_scan_deadline`).

use super::tests::{make_input, setup_with_config, upsert_points, TestContext};
use super::*;
use crate::helix_engine::graph_core::config::Config;
use crate::helix_engine::storage_core::storage_core::node_reads_for_test;
use serde_json::json;

const NAME: &str = "scroll_page_time";
const ROWS: u64 = 30;

fn group_filter() -> serde_json::Value {
    json!({"must": [{"key": "group", "match": {"value": "all"}}]})
}

/// Thirty rows of one indexed group: a filtered page of them is served from
/// indexed candidates.
fn fixture() -> TestContext {
    let ctx = setup_with_config(Config::default().with_lsm_in_memory());
    ctx.collections.create_collection(NAME).unwrap();
    let storage = ctx.collections.get_collection(NAME).unwrap();
    storage
        .create_payload_index("group", PayloadIndexSchema::Keyword)
        .unwrap();
    let points = (1..=ROWS)
        .map(|id| json!({"id": id, "payload": {"group": "all", "size": id}}))
        .collect::<Vec<_>>();
    upsert_points(&ctx, NAME, serde_json::Value::Array(points));
    ctx
}

/// The rows of the first page of the group and whether it hands on an offset,
/// for a page whose scan budget and wall are `budget_ms` and `wall_ms` and
/// whose planning takes `planning_ms` longer than it does.
fn first_page(
    ctx: &TestContext,
    endpoint: &str,
    budget_ms: usize,
    wall_ms: usize,
    planning_ms: u64,
) -> (Vec<u64>, bool) {
    first_page_of(
        ctx,
        endpoint,
        group_filter(),
        budget_ms,
        wall_ms,
        planning_ms,
    )
}

/// [`first_page`] of the rows `filter` takes.
fn first_page_of(
    ctx: &TestContext,
    endpoint: &str,
    filter: serde_json::Value,
    budget_ms: usize,
    wall_ms: usize,
    planning_ms: u64,
) -> (Vec<u64>, bool) {
    let body = json!({"limit": 100, "with_payload": true, "filter": filter});
    let input = make_input(
        ctx,
        "POST",
        &format!("/collections/{NAME}/points/{endpoint}"),
        serde_json::to_vec(&body).unwrap(),
        HashMap::from([("name".to_string(), NAME.to_string())]),
    );
    let mut response = Response::new();
    with_indexed_scroll_times_for_test(
        budget_ms,
        wall_ms,
        Duration::from_millis(planning_ms),
        || match endpoint {
            "scroll" => handle_scroll_points(&input, &mut response).unwrap(),
            _ => handle_scan_points(&input, &mut response).unwrap(),
        },
    );
    assert_eq!(response.status, 200);
    let reply: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
    let rows = reply["result"]["points"]
        .as_array()
        .unwrap()
        .iter()
        .map(|point| point["payload"]["size"].as_u64().unwrap())
        .collect();
    (rows, !reply["result"]["next_page_offset"].is_null())
}

fn all_rows() -> Vec<u64> {
    (1..=ROWS).collect()
}

/// The scan ends its budget after the planning did, and no later than the
/// wall: after 0.2 s of planning it has its 2 s, after 2 s of planning the
/// page ends at 2.7 s, and a plan that passes the wall leaves the scan none.
#[test]
fn an_indexed_scan_ends_its_budget_after_planning_and_no_later_than_the_wall() {
    let start = Instant::now();
    let after = |ms: u64| start + Duration::from_millis(ms);
    let (budget, wall) = (Duration::from_millis(2_000), after(2_700));
    let ends = |planned_ms: u64| indexed_scan_deadline(wall, after(planned_ms), budget);
    assert_eq!(ends(200), after(2_200));
    assert_eq!(ends(700), after(2_700));
    assert_eq!(ends(2_000), after(2_700));
    assert_eq!(ends(3_000), after(2_700));
}

/// The two settings as shipped; a wall never under the budget; and no wall
/// for a page with no filter or with the budget switched off.
#[test]
fn the_wall_is_2700_ms_over_a_budget_of_2000_and_never_under_it() {
    assert_eq!(qdrant_indexed_scroll_times_ms(), (2_000, 2_700));
    let times = |budget_ms: usize, wall_ms: usize| {
        with_indexed_scroll_times_for_test(budget_ms, wall_ms, Duration::ZERO, || {
            qdrant_indexed_scroll_times_ms()
        })
    };
    assert_eq!(times(2_000, 0), (2_000, 2_000));
    assert_eq!(times(2_000, 1_500), (2_000, 2_000));
    assert_eq!(times(2_000, 14_000), (2_000, 14_000));

    let filter: Filter = serde_json::from_value(group_filter()).unwrap();
    let wall = |budget_ms: usize, filter: &Filter| {
        with_indexed_scroll_times_for_test(budget_ms, 2_700, Duration::ZERO, || {
            qdrant_indexed_scroll_page_wall(filter)
        })
    };
    assert!(wall(2_000, &filter).is_some());
    assert!(wall(0, &filter).is_none());
    assert!(wall(2_000, &Filter::default()).is_none());
}

/// A page planned for longer than its scan budget still scans for the
/// budget: the thirty rows, where the page had one row and an offset because
/// its time had run out before the first row was read.
#[test]
fn a_page_planned_for_longer_than_its_scan_budget_returns_its_rows() {
    let ctx = fixture();
    assert_eq!(
        first_page(&ctx, "scroll", 1_000, 10_000, 1_200),
        (all_rows(), false)
    );
}

/// A page whose planning passes its wall returns the row it has and hands on
/// its offset, as a page did whose planning passed the 2 s. Its clock out
/// before its first row, it reads that row alone: a batch would be read for
/// a page that ends at its next row whatever the row is.
#[test]
fn a_page_planned_past_its_wall_returns_one_row_and_an_offset() {
    let ctx = fixture();
    let before = node_reads_for_test();
    assert_eq!(first_page(&ctx, "scroll", 300, 600, 700), (vec![1], true));
    assert_eq!(node_reads_for_test() - before, 1, "rows read");
}

/// With the wall at the budget's value or under it, the page's time is
/// counted from its start, as before: planning for longer than the budget
/// leaves one row, and one row read.
#[test]
fn a_wall_no_longer_than_the_budget_counts_the_page_from_its_start() {
    let ctx = fixture();
    let before = node_reads_for_test();
    assert_eq!(first_page(&ctx, "scroll", 300, 0, 400), (vec![1], true));
    assert_eq!(first_page(&ctx, "scroll", 300, 300, 400), (vec![1], true));
    assert_eq!(node_reads_for_test() - before, 2, "rows read by two pages");
}

/// A scroll page with the budget switched off and a `/points/scan` page have
/// no clock, before or after.
#[test]
fn a_page_with_no_budget_and_a_native_scan_page_have_no_clock() {
    let ctx = fixture();
    assert_eq!(first_page(&ctx, "scroll", 0, 100, 150), (all_rows(), false));
    assert_eq!(first_page(&ctx, "scan", 50, 100, 150), (all_rows(), false));
}

/// `size` has no index: a page of it walks the rows.
fn size_from(size: u64) -> serde_json::Value {
    json!({"must": [{"key": "size", "range": {"gte": size}}]})
}

/// The walk ends at its deadline put off by the planning, and no further
/// than the wall: after 0.4 s of planning it ends at 0.65 s, after 2.45 s
/// or more at the wall, and with no wall it is put off whatever the planning
/// took. A deadline set later than the wall is not brought forward.
#[test]
fn a_walk_ends_its_deadline_put_off_by_its_planning_and_not_past_the_wall() {
    let start = Instant::now();
    let after = |ms: u64| start + Duration::from_millis(ms);
    let ends = |planning_ms: u64, wall: Option<Instant>| {
        primary_scan_deadline(after(250), Duration::from_millis(planning_ms), wall)
    };
    let wall = Some(after(2_700));
    assert_eq!(ends(0, wall), after(250));
    assert_eq!(ends(400, wall), after(650));
    assert_eq!(ends(2_450, wall), after(2_700));
    assert_eq!(ends(5_000, wall), after(2_700));
    assert_eq!(ends(5_000, None), after(5_250));
    let set_late = primary_scan_deadline(after(5_000), Duration::from_millis(100), wall);
    assert_eq!(set_late, after(5_000));
}

/// A walked page planned for longer than its 250 ms still walks for them:
/// the thirty rows, where the page had one row and an offset because its
/// time had run out before the first row was read.
#[test]
fn a_walked_page_planned_for_longer_than_its_scan_time_returns_its_rows() {
    let ctx = fixture();
    assert_eq!(
        first_page_of(&ctx, "scroll", size_from(1), 2_000, 10_000, 400),
        (all_rows(), false)
    );
}

/// The same where the page walks the rows because its text condition's walk
/// of the index gave up, which is what takes a page's time before its scan.
#[test]
fn a_page_whose_text_walk_gave_up_still_walks_its_rows() {
    let ctx = fixture();
    let text = json!({"must": [{"key": "group", "match": {"text": "al"}}]});
    let page = with_text_walk_max_for_test(5, || {
        first_page_of(&ctx, "scroll", text, 2_000, 10_000, 400)
    });
    assert_eq!(page, (all_rows(), false));
}

/// By the scan's own count: a walk that turns every row down after a slow
/// plan reads its budget of rows, where it read one. And the wall ends a
/// walk as it ends an indexed scan: planned past it, the page reads one row.
#[test]
fn a_walk_after_a_slow_plan_scans_its_budget_and_the_wall_still_ends_it() {
    let ctx = fixture();
    let storage = ctx.collections.get_collection(NAME).unwrap();
    // No row is this large.
    let filter: Filter = serde_json::from_value(size_from(1_000)).unwrap();
    // The rows scanned and the offset handed on by a page of a budget of
    // eight rows and 250 ms, under a wall `wall_ms` from its start.
    let scan = |wall_ms: Option<u64>, planning_ms: u64| {
        let read = storage.backend.begin_read().unwrap();
        let opts = PointScanOptions {
            collection: NAME,
            limit: 10,
            offset_id: 0,
            filter: &filter,
            with_payload: &WithPayload::None,
            with_vectors: &WithVectors::None,
            partition: None,
            max_scan: Some(8),
            deadline: Some(Instant::now() + Duration::from_millis(250)),
            indexed_deadline: wall_ms.map(|ms| Instant::now() + Duration::from_millis(ms)),
            filter_hash: scan_filter_hash(&filter),
            cursor_generation: None,
            fuse_pending: false,
        };
        let planning = Duration::from_millis(planning_ms);
        let scan = with_indexed_scroll_times_for_test(2_000, 10_000, planning, || {
            execute_point_scan_be(&storage, &read, &opts)
        })
        .unwrap();
        assert!(matches!(scan.plan, PointScanPlan::Primary));
        (scan.scanned, scan.next_offset)
    };
    // Planned for 400 ms: the walk's 250 ms are still its own.
    assert_eq!(scan(Some(10_000), 400), (8, Some(format_point_id(9))));
    assert_eq!(scan(None, 400), (8, Some(format_point_id(9))));
    // Planned past a wall of 600 ms: one row.
    assert_eq!(scan(Some(600), 700), (1, Some(format_point_id(2))));
}
