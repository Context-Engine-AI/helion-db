//! A `must` condition that is broad next to a selective sibling is left out
//! of the candidate set and applied by the per-row recheck
//! (`without_broad_musts`), and a `has_id` list in `must` is the candidate
//! set (`must_has_id_ids`). Every result here is compared with the rows the
//! fixture says match and with the same request under the kill switch.

use super::tests::{make_input, setup_with_config, upsert_points, TestContext};
use super::*;
use crate::helix_engine::graph_core::config::Config;
use crate::helix_engine::storage_core::backend_lsm::dup_scan_rows_for_test;
use crate::helix_engine::storage_core::storage_core::node_reads_for_test;
use serde_json::json;

const NAME: &str = "selective_must";

fn repo(id: u64) -> &'static str {
    if (6..=10).contains(&id) {
        "b"
    } else {
        "a"
    }
}

fn path(id: u64) -> &'static str {
    match id {
        1..=10 => "hot",
        11..=300 => "warm",
        _ => "cold",
    }
}

fn lang(id: u64) -> &'static str {
    if id % 2 == 0 {
        "rust"
    } else {
        "python"
    }
}

fn noted(id: u64) -> bool {
    id % 3 == 0
}

/// Rows `1..=rows`, `size` being the row's number. `repo` is broad ("a" on
/// all but rows 6-10), `path` selective ("hot" on rows 1-10, five of them in
/// repo "b"). `note` and `at` (the row's number again) have no index.
fn fixture(rows: u64) -> TestContext {
    let mut config = Config::default().with_lsm_in_memory();
    config.vector_config.flat_scan_threshold = Some(0);
    let ctx = setup_with_config(config);
    ctx.collections.create_collection(NAME).unwrap();
    let storage = ctx.collections.get_collection(NAME).unwrap();
    for field in ["repo", "path", "lang"] {
        storage
            .create_payload_index(field, PayloadIndexSchema::Keyword)
            .unwrap();
    }
    storage
        .create_payload_index("size", PayloadIndexSchema::Integer)
        .unwrap();
    let points = (1..=rows)
        .map(|id| {
            json!({"id": id, "payload": {
                "repo": repo(id),
                "path": path(id),
                "lang": lang(id),
                "size": id,
                "at": id,
                "note": if noted(id) { "x" } else { "y" }
            }})
        })
        .collect::<Vec<_>>();
    upsert_points(&ctx, NAME, serde_json::Value::Array(points));
    ctx
}

fn eq(key: &str, value: &str) -> serde_json::Value {
    json!({"key": key, "match": {"value": value}})
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
        "count" => handle_count_points(&input, &mut response).unwrap(),
        "delete" => handle_delete_points(&input, &mut response).unwrap(),
        other => panic!("no handler for {other}"),
    }
    assert_eq!(response.status, 200, "{endpoint} {body}");
    serde_json::from_slice(&response.body).unwrap()
}

/// A page: its row numbers and the offset it hands on, null after the last.
type Page = (Vec<u64>, serde_json::Value);

/// Every page of a scroll or a `/points/scan` (`endpoint`) from `offset` on,
/// following the page offsets. `request` is the body without the offset.
fn pages_from(
    ctx: &TestContext,
    endpoint: &str,
    request: &serde_json::Value,
    mut offset: serde_json::Value,
) -> Vec<Page> {
    let mut pages = Vec::new();
    for _ in 0..1_000 {
        let mut body = request.clone();
        if !offset.is_null() {
            body["offset"] = offset.clone();
        }
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
            return pages;
        }
    }
    panic!("{endpoint} did not end");
}

/// The row numbers of every page of a scroll, following the page offsets.
fn scroll_pages(ctx: &TestContext, filter: &serde_json::Value, limit: usize) -> Vec<Vec<u64>> {
    let request = json!({"limit": limit, "with_payload": true, "filter": filter});
    pages_from(ctx, "scroll", &request, serde_json::Value::Null)
        .into_iter()
        .map(|(rows, _)| rows)
        .collect()
}

fn count(ctx: &TestContext, filter: &serde_json::Value, exact: bool) -> u64 {
    post(ctx, "count", json!({"filter": filter, "exact": exact}))["result"]["count"]
        .as_u64()
        .unwrap()
}

/// A filter, the rows that match it, and whether a `must` is left out of its
/// candidate set in the 400-row fixture (repo "a" 395 rows, "b" 5, path "hot"
/// 10, "warm" 290, each language 200).
type Case = (&'static str, serde_json::Value, fn(u64) -> bool, bool);

fn case(
    name: &'static str,
    filter: serde_json::Value,
    matches: fn(u64) -> bool,
    narrows: bool,
) -> Case {
    (name, filter, matches, narrows)
}

fn cases() -> Vec<Case> {
    let hot_in_a = || json!([eq("repo", "a"), eq("path", "hot")]);
    vec![
        case(
            "selective next to broad",
            json!({"must": hot_in_a()}),
            |id| repo(id) == "a" && path(id) == "hot",
            true,
        ),
        case(
            "with must_not",
            json!({"must": hot_in_a(), "must_not": [eq("lang", "rust")]}),
            |id| repo(id) == "a" && path(id) == "hot" && lang(id) != "rust",
            true,
        ),
        case(
            "with should",
            json!({"must": hot_in_a(), "should": [
                eq("lang", "rust"),
                {"key": "size", "range": {"gte": 4}}
            ]}),
            |id| repo(id) == "a" && path(id) == "hot" && (lang(id) == "rust" || id >= 4),
            true,
        ),
        case(
            "with a nested must_not",
            json!({"must": hot_in_a(), "must_not": [
                {"must": [eq("lang", "rust"), eq("repo", "a")]}
            ]}),
            |id| repo(id) == "a" && path(id) == "hot" && lang(id) != "rust",
            true,
        ),
        case(
            "two broad siblings, one under the bound",
            json!({"must": [eq("repo", "a"), eq("path", "hot"), eq("lang", "python")]}),
            |id| repo(id) == "a" && path(id) == "hot" && lang(id) == "python",
            true,
        ),
        case(
            "broad any",
            json!({"must": [{"key": "repo", "match": {"any": ["a", "b"]}}, eq("path", "hot")]}),
            |id| path(id) == "hot",
            true,
        ),
        case(
            "selective range",
            json!({"must": [{"key": "size", "range": {"lte": 8}}, eq("repo", "a")]}),
            |id| id <= 8 && repo(id) == "a",
            true,
        ),
        case(
            "unindexed sibling",
            json!({"must": [eq("note", "x"), eq("repo", "a"), eq("path", "hot")]}),
            |id| noted(id) && repo(id) == "a" && path(id) == "hot",
            true,
        ),
        case(
            "selective matches nothing",
            json!({"must": [eq("repo", "a"), eq("path", "missing")]}),
            |_| false,
            true,
        ),
        case(
            "selective inside a nested must",
            json!({"must": [eq("repo", "a"), {"must": [eq("path", "hot"), eq("repo", "a")]}]}),
            |id| repo(id) == "a" && path(id) == "hot",
            false,
        ),
        case(
            "nested exclusion",
            json!({"must": [eq("path", "hot"), {"must_not": [eq("repo", "b")]}]}),
            |id| path(id) == "hot" && repo(id) != "b",
            false,
        ),
        case(
            "nested should",
            json!({"must": [
                {"should": [eq("path", "hot"), eq("path", "warm")]},
                eq("repo", "b")
            ]}),
            |id| repo(id) == "b" && path(id) != "cold",
            false,
        ),
        case(
            "sibling only a few times larger",
            json!({"must": [eq("repo", "a"), eq("path", "warm")]}),
            |id| repo(id) == "a" && path(id) == "warm",
            false,
        ),
        case(
            "nothing indexed is broad",
            json!({"must": [eq("note", "x"), eq("path", "hot")]}),
            |id| noted(id) && path(id) == "hot",
            false,
        ),
        case(
            "siblings under the bound",
            json!({"must": [eq("repo", "b"), eq("lang", "python"), eq("path", "hot")]}),
            |id| repo(id) == "b" && lang(id) == "python" && path(id) == "hot",
            false,
        ),
    ]
}

/// Scroll pages and counts are the rows the fixture says match, with the
/// broad condition left to the recheck and with every condition materialized.
/// A recheck that did not apply the condition left out would return rows 6-10.
#[test]
fn scrolls_and_counts_match_the_rows_with_and_without_the_skip() {
    const ROWS: u64 = 400;
    let ctx = fixture(ROWS);
    let storage = ctx.collections.get_collection(NAME).unwrap();
    for (name, filter, matches, narrows) in cases() {
        let expected: Vec<u64> = (1..=ROWS).filter(|id| matches(*id)).collect();
        let typed: Filter = serde_json::from_value(filter.clone()).unwrap();
        let read = storage.backend.begin_read().unwrap();
        let narrowed = with_selective_must_for_test(true, || {
            without_broad_musts(&storage, &read, &typed, usize::MAX).unwrap()
        });
        assert_eq!(narrowed.is_some(), narrows, "{name}");
        assert!(
            with_selective_must_for_test(false, || {
                without_broad_musts(&storage, &read, &typed, usize::MAX).unwrap()
            })
            .is_none(),
            "{name}: kill switch"
        );
        drop(read);

        let observe = |enabled: bool| {
            with_selective_must_for_test(enabled, || {
                (
                    scroll_pages(&ctx, &filter, 3),
                    count(&ctx, &filter, true),
                    count(&ctx, &filter, false),
                )
            })
        };
        let (pages, exact_count, loose_count) = observe(true);
        assert_eq!(pages.concat(), expected, "{name}: scrolled rows");
        assert_eq!(exact_count, expected.len() as u64, "{name}: exact count");
        assert_eq!(loose_count, expected.len() as u64, "{name}: count");
        assert_eq!(
            observe(false),
            (pages, exact_count, loose_count),
            "{name}: kill switch"
        );
    }
}

/// A filtered delete removes exactly the matching rows either way.
#[test]
fn filtered_delete_resolves_the_same_rows_with_and_without_the_skip() {
    const ROWS: u64 = 400;
    // Every other narrowed shape: a fixture each, for both settings.
    let narrowed = cases().into_iter().filter(|case| case.3).step_by(2);
    for (name, filter, matches, _) in narrowed {
        let kept: Vec<u64> = (1..=ROWS).filter(|id| !matches(*id)).collect();
        for enabled in [true, false] {
            let ctx = fixture(ROWS);
            with_selective_must_for_test(enabled, || {
                post(&ctx, "delete", json!({"filter": filter}));
            });
            assert_eq!(
                scroll_pages(&ctx, &json!({}), 128).concat(),
                kept,
                "{name}: rows left, skip {enabled}"
            );
        }
    }
}

/// The broad posting list is counted only as far as the bound and never
/// materialized; a page whose scan budget the selective set would not fit
/// under, and a filter whose conditions are all broad, are planned as before.
#[test]
fn broad_posting_is_walked_no_further_than_the_bound() {
    const ROWS: u64 = 1_200;
    let ctx = fixture(ROWS);
    let storage = ctx.collections.get_collection(NAME).unwrap();
    let read = storage.backend.begin_read().unwrap();
    let filter: Filter =
        serde_json::from_value(json!({"must": [eq("repo", "a"), eq("path", "hot")]})).unwrap();
    let plan = |enabled: bool, max_scan: Option<usize>| {
        let before = dup_scan_rows_for_test();
        let ids = with_selective_must_for_test(enabled, || {
            scan_indexed_filter_candidates_be(&storage, &read, &filter, max_scan).unwrap()
        })
        .unwrap();
        let mut ids: Vec<u128> = ids.into_iter().collect();
        ids.sort_unstable();
        (ids, dup_scan_rows_for_test() - before)
    };
    let hot: Vec<u128> = (1..=10).collect();
    let hot_in_a: Vec<u128> = (1..=5).collect();

    let (ids, visited) = plan(true, None);
    assert_eq!(ids, hot, "the selective condition's rows, to be rechecked");
    assert!(
        visited <= SELECTIVE_MUST_MAX_ROWS + 64,
        "visited {visited} posting rows"
    );
    let (ids, visited) = plan(false, None);
    assert_eq!(ids, hot_in_a);
    assert!(
        visited >= 2 * 1_195,
        "the kill switch counts and materializes repo: {visited}"
    );

    // Ten rows do not fit under a budget of ten: the page keeps the plan
    // whose rows scanned cannot reach it.
    assert_eq!(plan(true, Some(10)).0, hot_in_a);
    assert_eq!(plan(true, Some(11)).0, hot);
    // A page whose budget is over the 512 rows counts no further where a
    // condition is already under them.
    let (ids, visited) = plan(true, Some(4_096));
    assert_eq!(ids, hot);
    assert!(
        visited <= SELECTIVE_MUST_MAX_ROWS + 64,
        "visited {visited} posting rows"
    );

    let all_broad: Filter =
        serde_json::from_value(json!({"must": [eq("repo", "a"), eq("lang", "rust")]})).unwrap();
    assert!(with_selective_must_for_test(true, || {
        without_broad_musts(&storage, &read, &all_broad, usize::MAX).unwrap()
    })
    .is_none());
}

/// The rows a `has_id` list names here: 6 and 9 are in repo "b", 250 has no
/// note, 303 is "cold", and there is no row 9,999.
const LISTED: [u64; 7] = [3, 6, 9, 12, 250, 303, 9_999];

fn listed(id: u64) -> bool {
    LISTED.contains(&id)
}

fn has_id(ids: &[u64]) -> serde_json::Value {
    json!({"has_id": ids})
}

/// As [`cases`], each with a `has_id` list; the flag says whether the list is
/// in the top-level `must`, where it is the candidate set.
fn listed_cases() -> Vec<Case> {
    vec![
        case(
            "the list alone",
            json!({"must": [has_id(&LISTED)]}),
            listed,
            true,
        ),
        case(
            "rows fail a broad sibling",
            json!({"must": [has_id(&LISTED), eq("repo", "a")]}),
            |id| listed(id) && repo(id) == "a",
            true,
        ),
        case(
            "next to several paths",
            json!({"must": [{"key": "path", "match": {"any": ["hot", "warm"]}}, has_id(&LISTED)]}),
            |id| listed(id) && path(id) != "cold",
            true,
        ),
        case(
            "next to a selective sibling",
            json!({"must": [eq("path", "hot"), has_id(&LISTED)]}),
            |id| listed(id) && path(id) == "hot",
            true,
        ),
        case(
            "next to an unindexed sibling",
            json!({"must": [has_id(&LISTED), eq("note", "x")]}),
            |id| listed(id) && noted(id),
            true,
        ),
        case(
            "the prune page",
            prune_page(),
            |id| listed(id) && noted(id) && id < 1_000 && path(id) != "hot",
            true,
        ),
        case(
            "with must_not",
            json!({"must": [has_id(&LISTED), eq("repo", "a")], "must_not": [eq("lang", "rust")]}),
            |id| listed(id) && repo(id) == "a" && lang(id) != "rust",
            true,
        ),
        case(
            "with should",
            json!({"must": [has_id(&LISTED)], "should": [
                eq("path", "hot"),
                {"key": "size", "range": {"gte": 300}}
            ]}),
            |id| listed(id) && (path(id) == "hot" || id >= 300),
            true,
        ),
        case(
            "with a nested filter",
            json!({"must": [has_id(&LISTED), {
                "must": [eq("repo", "a")],
                "must_not": [eq("path", "warm")]
            }]}),
            |id| listed(id) && repo(id) == "a" && path(id) != "warm",
            true,
        ),
        case(
            "two lists",
            json!({"must": [has_id(&[3, 6, 9, 12]), has_id(&[9, 12, 250])]}),
            |id| id == 9 || id == 12,
            true,
        ),
        case(
            "an empty list",
            json!({"must": [has_id(&[])]}),
            |_| false,
            true,
        ),
        case(
            "an empty list next to a condition",
            json!({"must": [has_id(&[]), eq("repo", "a")]}),
            |_| false,
            true,
        ),
        case(
            "only a row that does not exist",
            json!({"must": [has_id(&[9_999])]}),
            |_| false,
            true,
        ),
        case(
            "list in must_not",
            json!({"must": [eq("path", "hot")], "must_not": [has_id(&LISTED)]}),
            |id| path(id) == "hot" && !listed(id),
            false,
        ),
        case(
            "list in should",
            json!({"must": [eq("repo", "b")], "should": [has_id(&LISTED), eq("path", "warm")]}),
            |id| repo(id) == "b" && (listed(id) || path(id) == "warm"),
            false,
        ),
        case(
            "list in a nested filter",
            json!({"must": [{"must": [has_id(&LISTED)]}, eq("repo", "a")]}),
            |id| listed(id) && repo(id) == "a",
            false,
        ),
    ]
}

/// The filter of one page of the client's prune of replaced chunks, in its order:
/// two conditions on the rows' markers, which have no index, the page's files
/// and the page's ids as the hex strings a scroll answers.
fn prune_page() -> serde_json::Value {
    let ids: Vec<String> = LISTED.iter().map(|id| format!("{id:032x}")).collect();
    json!({"must": [
        eq("note", "x"),
        {"key": "at", "range": {"lt": 1_000.0}},
        {"key": "path", "match": {"any": ["warm", "cold"]}},
        {"has_id": ids}
    ]})
}

/// With a `has_id` list in `must` as the candidate set, scroll pages and
/// counts are the rows the fixture says match, and the rows the kill switch
/// returns. A recheck that did not apply the rest of the filter would return
/// rows 6 and 9 next to repo "a", and an exact count from the list would
/// count row 9,999.
#[test]
fn listed_ids_scroll_and_count_the_same_rows_as_candidates_or_not() {
    const ROWS: u64 = 400;
    let ctx = fixture(ROWS);
    let storage = ctx.collections.get_collection(NAME).unwrap();
    for (name, filter, matches, in_must) in listed_cases() {
        let expected: Vec<u64> = (1..=ROWS).filter(|id| matches(*id)).collect();
        if in_must {
            let typed: Filter = serde_json::from_value(filter.clone()).unwrap();
            let read = storage.backend.begin_read().unwrap();
            let planned = with_selective_must_for_test(true, || {
                [
                    scan_indexed_filter_candidates_be(&storage, &read, &typed, None).unwrap(),
                    filter_delete_candidates_be(&storage, &read, &typed).unwrap(),
                ]
            });
            for candidates in planned {
                let candidates = candidates.expect(name);
                assert!(
                    candidates.iter().all(|id| listed(*id as u64)),
                    "{name}: candidates {candidates:?}"
                );
            }
        }

        // Pages are compared joined: under the kill switch a list with no
        // indexed sibling is a walk, whose page may end at the walk's deadline.
        let observe = |enabled: bool| {
            with_selective_must_for_test(enabled, || {
                (
                    scroll_pages(&ctx, &filter, 3).concat(),
                    count(&ctx, &filter, true),
                    count(&ctx, &filter, false),
                )
            })
        };
        let rows = (
            expected.clone(),
            expected.len() as u64,
            expected.len() as u64,
        );
        assert_eq!(observe(true), rows, "{name}");
        assert_eq!(observe(false), rows, "{name}: kill switch");
    }
}

/// A filtered delete removes exactly the matching rows either way: a listed
/// row that fails the rest of the filter stays.
#[test]
fn filtered_delete_removes_the_same_rows_with_the_list_as_candidates_or_not() {
    const ROWS: u64 = 400;
    for (name, filter, matches, _) in listed_cases() {
        let kept: Vec<u64> = (1..=ROWS).filter(|id| !matches(*id)).collect();
        for enabled in [true, false] {
            let ctx = fixture(ROWS);
            with_selective_must_for_test(enabled, || {
                post(&ctx, "delete", json!({"filter": filter}));
            });
            assert_eq!(
                scroll_pages(&ctx, &json!({}), 128).concat(),
                kept,
                "{name}: rows left, list as candidates {enabled}"
            );
        }
    }
}

/// A prune page reads the rows it lists: seven here, against every row of
/// the files it names (1,190, rows 11 and up), which took the scroll a second
/// page. A list with no indexed sibling was a walk of the collection, a scan
/// budget of rows a page.
#[test]
fn a_page_of_listed_ids_reads_the_listed_rows() {
    const ROWS: u64 = 1_200;
    const FILE_ROWS: usize = 1_190;
    let ctx = fixture(ROWS);
    let storage = ctx.collections.get_collection(NAME).unwrap();
    let page = prune_page();

    let scroll = |enabled: bool| {
        let before = node_reads_for_test();
        let pages = with_selective_must_for_test(enabled, || scroll_pages(&ctx, &page, 10));
        (pages, node_reads_for_test() - before)
    };
    let (pages, reads) = scroll(true);
    assert_eq!(pages, [[12, 303]]);
    assert!(reads <= LISTED.len(), "read {reads} rows");
    let (pages, reads) = scroll(false);
    assert_eq!(pages.concat(), [12, 303]);
    assert!(reads >= FILE_ROWS, "the kill switch read {reads} rows");

    let read = storage.backend.begin_read().unwrap();
    let typed: Filter = serde_json::from_value(page).unwrap();
    let delete_reads = |enabled: bool| {
        with_selective_must_for_test(enabled, || {
            filter_delete_candidates_be(&storage, &read, &typed).unwrap()
        })
        .map(|rows| rows.len())
    };
    assert_eq!(delete_reads(true), Some(LISTED.len()));
    assert_eq!(delete_reads(false), Some(FILE_ROWS));

    // A list that does not fit under the page's scan budget is planned as
    // before: its rows scanned could reach the budget.
    let alone = json!({"must": [has_id(&LISTED)]});
    let typed: Filter = serde_json::from_value(alone.clone()).unwrap();
    let plan = |max_scan: usize| {
        with_selective_must_for_test(true, || {
            scan_indexed_filter_candidates_be(&storage, &read, &typed, Some(max_scan)).unwrap()
        })
        .map(|rows| rows.len())
    };
    assert_eq!(plan(LISTED.len()), None);
    assert_eq!(plan(LISTED.len() + 1), Some(LISTED.len()));
    drop(read);

    // The walk ends its first page at the budget, short of the last rows;
    // the list's page is whole. The rows over all pages are the same.
    let walked = with_selective_must_for_test(false, || scroll_pages(&ctx, &alone, 10));
    let listed_page = with_selective_must_for_test(true, || scroll_pages(&ctx, &alone, 10));
    assert_eq!(listed_page, [[3, 6, 9, 12, 250, 303]]);
    assert!(walked.len() > 1, "the walk's pages: {walked:?}");
    assert_eq!(walked.concat(), listed_page.concat());
}

/// The count probe as it was before a later condition was counted only as
/// far as the smallest count so far: every `must` condition counted to the
/// cap on its own. What the planners read of it, to compare.
fn count_probe_counting_each_to_the_cap(
    storage: &HelixGraphStorage,
    read: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    filter: &Filter,
    cap: usize,
) -> (Option<usize>, Vec<usize>) {
    let mut estimate: Option<usize> = None;
    let mut unprobed_indexable = Vec::new();
    for (index, condition) in filter.must.iter().enumerate() {
        let mut unprobed = false;
        let count = match condition {
            Condition::Field(field) => {
                indexed_condition_count_be(storage, read, field, cap).unwrap()
            }
            Condition::Nested(nested) => {
                let (inner, inner_unprobed) =
                    count_probe_counting_each_to_the_cap(storage, read, nested, cap);
                unprobed = !inner_unprobed.is_empty();
                inner
            }
            Condition::HasId(_) | Condition::IsEmpty(_) | Condition::IsNull(_) => None,
        };
        match count {
            Some(count) => estimate = Some(estimate.map_or(count, |least| least.min(count))),
            None => unprobed |= condition_may_yield_candidates(storage, condition),
        }
        if unprobed {
            unprobed_indexable.push(index);
        }
    }
    (estimate, unprobed_indexable)
}

/// Text conditions for the count probe: one with an exact keyword hit, which
/// it counts, and one it cannot count and leaves to the walk.
fn text_cases() -> Vec<serde_json::Value> {
    let text = |key: &str, text: &str| json!({"key": key, "match": {"text": text}});
    vec![
        json!({"must": [text("path", "hot"), eq("repo", "a")]}),
        json!({"must": [eq("repo", "a"), text("path", "ho")]}),
        json!({"must": [text("path", "ho"), eq("path", "hot"), eq("repo", "a")]}),
        // After a condition that matches nothing the rest are counted to no
        // rows: a text with an exact hit must still count as one it counted.
        json!({"must": [eq("path", "missing"), text("path", "hot"), eq("repo", "a")]}),
        json!({"must": [{"must": [text("path", "war"), eq("repo", "a")]}, eq("lang", "rust")]}),
    ]
}

/// The count probe gives every planner what it gave: the same estimate and
/// the same conditions left uncounted, for every filter of this file in the
/// order written and in reverse, at caps under, at and over the fixture's
/// counts (repo "a" 395 rows, "b" 5, path "hot" 10, "warm" 290, each language
/// 200).
#[test]
fn the_count_probe_estimates_what_it_did_for_every_filter_order_and_cap() {
    const ROWS: u64 = 400;
    let ctx = fixture(ROWS);
    let storage = ctx.collections.get_collection(NAME).unwrap();
    let read = storage.backend.begin_read().unwrap();
    let filters = cases()
        .into_iter()
        .chain(listed_cases())
        .map(|(_, filter, _, _)| filter)
        .chain(text_cases());
    let mut compared = 0;
    for filter in filters {
        let mut typed: Filter = serde_json::from_value(filter.clone()).unwrap();
        for order in ["as written", "reversed"] {
            for cap in [
                0, 1, 4, 5, 6, 9, 10, 11, 199, 200, 201, 289, 290, 394, 395, 100_000,
            ] {
                let probe = must_count_probe_be(&storage, &read, &typed, cap).unwrap();
                assert_eq!(
                    (probe.estimate, probe.unprobed_indexable),
                    count_probe_counting_each_to_the_cap(&storage, &read, &typed, cap),
                    "{filter} {order}, cap {cap}"
                );
                compared += 1;
            }
            typed.must.reverse();
        }
    }
    assert!(compared >= 1_000, "{compared} probes compared");
}

/// A condition after the smallest so far is counted one row past it and no
/// further: its count could not lower the estimate, which is the minimum.
#[test]
fn the_count_probe_counts_a_later_condition_as_far_as_the_smallest_so_far() {
    const ROWS: u64 = 1_200;
    let ctx = fixture(ROWS);
    let storage = ctx.collections.get_collection(NAME).unwrap();
    let read = storage.backend.begin_read().unwrap();
    let probed = |must: serde_json::Value| {
        let filter: Filter = serde_json::from_value(json!({"must": must})).unwrap();
        let before = dup_scan_rows_for_test();
        let probe = must_count_probe_be(&storage, &read, &filter, 100_000).unwrap();
        (probe.estimate, dup_scan_rows_for_test() - before)
    };

    // Path "hot" is 10 rows and repo "a" 1,195: the ten, then eleven of "a".
    assert_eq!(
        probed(json!([eq("path", "hot"), eq("repo", "a")])),
        (Some(10), 10 + 11)
    );
    // The broad one first is counted to its end, as it was.
    assert_eq!(
        probed(json!([eq("repo", "a"), eq("path", "hot")])),
        (Some(10), 1_195 + 10)
    );
    // Each later one stops at the smallest before it: 290 rows are "warm",
    // then 291 of "a", then the ten "hot" (none is in both paths), then one
    // row of each language.
    assert_eq!(
        probed(json!([
            eq("path", "warm"),
            eq("repo", "a"),
            eq("path", "hot"),
            eq("lang", "rust"),
            eq("lang", "python")
        ])),
        (Some(10), 290 + 291 + 10 + 11 + 11)
    );
    // A nested filter's conditions are counted as far as the smallest
    // before it too: eleven rows of "a", eleven of the language.
    assert_eq!(
        probed(json!([
            eq("path", "hot"),
            {"must": [eq("repo", "a"), eq("lang", "rust")]}
        ])),
        (Some(10), 10 + 11 + 11)
    );

    // A scan page the ten rows do not fit under is planned through the
    // probe: the same candidates, and "a" walked once, for its ids.
    let filter: Filter =
        serde_json::from_value(json!({"must": [eq("path", "hot"), eq("repo", "a")]})).unwrap();
    let before = dup_scan_rows_for_test();
    let mut ids: Vec<u128> = scan_indexed_filter_candidates_be(&storage, &read, &filter, Some(10))
        .unwrap()
        .unwrap()
        .into_iter()
        .collect();
    ids.sort_unstable();
    assert_eq!(ids, (1..=5).collect::<Vec<u128>>());
    // Both counted to the nine rows a budget of ten leaves, the probe's 21,
    // and both lists for their ids.
    assert_eq!(dup_scan_rows_for_test() - before, 2 * 10 + 21 + 10 + 1_195);
}

/// Rows of [`page_fixture`], and of one of its files.
const PAGE_ROWS: u64 = 6_000;
const FILE_ROWS: u64 = 100;
/// The files a page asks for: rows 2,001 to 2,600.
const ASKED_FILES: std::ops::Range<u64> = 20..26;

fn file(id: u64) -> u64 {
    (id - 1) / FILE_ROWS
}

fn kind(id: u64) -> &'static str {
    if id % 10 == 0 {
        "imports"
    } else {
        "calls"
    }
}

fn asked(id: u64) -> bool {
    ASKED_FILES.contains(&file(id))
}

fn asked_call(id: u64) -> bool {
    asked(id) && kind(id) == "calls"
}

/// The first of the files asked for, and the fourth: rows that are "out" in
/// `early`, and in `late`.
const EARLY_ROWS: std::ops::RangeInclusive<u64> = 2_001..=2_100;
const LATE_ROWS: std::ops::RangeInclusive<u64> = 2_301..=2_400;

fn side(out: bool) -> &'static str {
    if out {
        "out"
    } else {
        "in"
    }
}

/// 6,000 rows, `size` again the row's number, with a number index. `file` is
/// the same on 100 rows running and `kind` is "calls" on nine rows in ten.
/// The six files asked for are 600 rows, over the 512 a condition may match
/// to be the selective one where there is no page, and 60 of them are no
/// calls. "calls" is 5,400 rows, over eight times the files'. `early` and
/// `late` are "in" on 5,900 rows each: all but the files' first hundred, and
/// all but their fourth.
fn page_fixture() -> TestContext {
    let mut config = Config::default().with_lsm_in_memory();
    config.vector_config.flat_scan_threshold = Some(0);
    let ctx = setup_with_config(config);
    ctx.collections.create_collection(NAME).unwrap();
    let storage = ctx.collections.get_collection(NAME).unwrap();
    for field in ["file", "kind", "early", "late"] {
        storage
            .create_payload_index(field, PayloadIndexSchema::Keyword)
            .unwrap();
    }
    storage
        .create_payload_index("size", PayloadIndexSchema::Integer)
        .unwrap();
    let rows: Vec<u64> = (1..=PAGE_ROWS).collect();
    for batch in rows.chunks(1_500) {
        let points = batch
            .iter()
            .map(|&id| {
                json!({"id": id, "payload": {
                    "file": format!("f{}", file(id)),
                    "kind": kind(id),
                    "early": side(EARLY_ROWS.contains(&id)),
                    "late": side(LATE_ROWS.contains(&id)),
                    "size": id
                }})
            })
            .collect::<Vec<_>>();
        upsert_points(&ctx, NAME, serde_json::Value::Array(points));
    }
    ctx
}

/// The filter of the client's callers read: the files, each in a second spelling no
/// row has, and the kind.
fn asked_calls() -> serde_json::Value {
    let files: Vec<String> = ASKED_FILES
        .flat_map(|file| [format!("f{file}"), format!("/f{file}")])
        .collect();
    json!({"must": [{"key": "file", "match": {"any": files}}, eq("kind", "calls")]})
}

fn rows_where(matches: fn(u64) -> bool) -> Vec<u128> {
    (1..=PAGE_ROWS)
        .filter(|id| matches(*id))
        .map(u128::from)
        .collect()
}

/// Where no condition is under 512 rows, a page with a scan budget takes one
/// that fits under the budget as the selective one: its rows are the
/// candidates, and the broad sibling's posting list is counted as far as
/// eight times those rows and never materialized.
#[test]
fn a_page_takes_a_condition_of_over_512_rows_that_fits_under_its_scan_budget() {
    let ctx = page_fixture();
    let storage = ctx.collections.get_collection(NAME).unwrap();
    let read = storage.backend.begin_read().unwrap();
    let filter: Filter = serde_json::from_value(asked_calls()).unwrap();
    let plan = |enabled: bool, max_scan: Option<usize>| {
        let before = dup_scan_rows_for_test();
        let ids = with_selective_must_for_test(enabled, || {
            scan_indexed_filter_candidates_be(&storage, &read, &filter, max_scan).unwrap()
        })
        .unwrap();
        let mut ids: Vec<u128> = ids.into_iter().collect();
        ids.sort_unstable();
        (ids, dup_scan_rows_for_test() - before)
    };
    let (files, calls) = (rows_where(asked), rows_where(asked_call));
    let all_calls = rows_where(|id| kind(id) == "calls").len();
    assert_eq!((files.len(), calls.len(), all_calls), (600, 540, 5_400));

    // A scroll page of 64 rows or more has a budget of 4,096 rows.
    let (ids, page_visited) = plan(true, Some(4_096));
    assert_eq!(ids, files, "the files' rows, to be rechecked");
    // The files' rows three times (counted to the 512, which the sixth file
    // passes, counted to the page's bound, and read for their ids), and of
    // the kind's 513, then 4,096, then one more than eight times the files'.
    let decided = 3 * files.len() + (SELECTIVE_MUST_MAX_ROWS + 1) + 4_096 + (8 * files.len() + 1);
    assert!(
        page_visited <= decided,
        "visited {page_visited} posting rows"
    );
    let (ids, visited) = plan(false, Some(4_096));
    assert_eq!(ids, calls);
    assert!(
        visited >= all_calls,
        "the kill switch reads the kind's whole list for its ids: {visited}"
    );

    // The set stays under the page's budget, so that its rows scanned cannot
    // reach it: 600 rows do not fit under a budget of 600.
    assert_eq!(plan(true, Some(600)).0, calls);
    assert_eq!(plan(true, Some(601)).0, files);
    // A scan with no budget has no page, and the 512 rows stand.
    assert_eq!(plan(true, None).0, calls);
    // A budget of a million rows is planned as a scroll's largest: the same
    // rows, and no condition counted further than for it.
    assert_eq!(plan(true, Some(1_000_000)), (files, page_visited));

    // A sibling is broad over eight times the files' rows, as under the 512:
    // 48 files are 4,800 rows, past the page's bound, and 49 are 4,900.
    let left_beside = |many: u64| {
        let named: Vec<String> = (0..many).map(|file| format!("f{file}")).collect();
        let beside: Filter = serde_json::from_value(json!({"must": [
            asked_calls()["must"][0],
            {"key": "file", "match": {"any": named}}
        ]}))
        .unwrap();
        with_selective_must_for_test(true, || {
            without_broad_musts(&storage, &read, &beside, 4_095).unwrap()
        })
        .map(|narrowed| narrowed.must.len())
    };
    assert_eq!(left_beside(48), None, "nothing left out");
    assert_eq!(left_beside(49), Some(1), "the 49 files left out");
}

#[test]
fn a_pages_bound_is_under_its_budget_and_no_more_than_a_scrolls_largest() {
    // No page: a count, a filtered delete, a scan with no budget.
    assert_eq!(page_selective_rows(usize::MAX), None);
    // A budget the 512 rows already cover.
    assert_eq!(page_selective_rows(SELECTIVE_MUST_MAX_ROWS), None);
    assert_eq!(page_selective_rows(513), Some(513));
    // A scroll of 16 rows or fewer, and of 64 rows or more.
    assert_eq!(page_selective_rows(1_023), Some(1_023));
    assert_eq!(page_selective_rows(4_095), Some(4_095));
    // A `/points/scan` page may ask for a budget of a million rows; the
    // counts that decide would walk every condition that far.
    assert_eq!(page_selective_rows(999_999), Some(4_095));
}

/// A count and a filtered delete have no page: over 512 rows they
/// materialize every condition, as before.
#[test]
fn count_and_filtered_delete_keep_the_bound_of_512_rows() {
    let ctx = page_fixture();
    let storage = ctx.collections.get_collection(NAME).unwrap();
    let read = storage.backend.begin_read().unwrap();
    let filter = asked_calls();
    let typed: Filter = serde_json::from_value(filter.clone()).unwrap();
    let (files, calls) = (rows_where(asked).len(), rows_where(asked_call));
    let all_calls = rows_where(|id| kind(id) == "calls").len();

    with_selective_must_for_test(true, || {
        assert!(without_broad_musts(&storage, &read, &typed, usize::MAX)
            .unwrap()
            .is_none());
        // The rows a delete reads are the matches, not the files' 600.
        assert_eq!(
            filter_delete_candidates_be(&storage, &read, &typed).unwrap(),
            Some(calls.iter().copied().collect())
        );
    });
    drop(read);

    // A count walks what it walked: both conditions as far as the 512 (the
    // sixth file passes them), then both to their end for their ids.
    let before = dup_scan_rows_for_test();
    let counted = with_selective_must_for_test(true, || {
        (count(&ctx, &filter, true), count(&ctx, &filter, false))
    });
    let visited = dup_scan_rows_for_test() - before;
    assert_eq!(counted, (calls.len() as u64, calls.len() as u64));
    assert_eq!(
        visited,
        2 * (files + (SELECTIVE_MUST_MAX_ROWS + 1) + files + all_calls)
    );
}

/// Every page of a scroll and of a `/points/scan`, its rows and the offset it
/// hands on, is the same with the page's bound as under the kill switch, and
/// the rows are the ones the fixture says match. 60 of the 600 rows the bound
/// takes as candidates fail the kind and are rejected by the recheck.
#[test]
fn pages_are_the_same_with_the_page_bound_and_under_the_kill_switch() {
    let ctx = page_fixture();
    let filter = asked_calls();
    let scroll = |limit: usize| {
        let request = json!({"limit": limit, "with_payload": true, "filter": filter});
        ("scroll", request)
    };
    let scan = |limit: usize, max_scan: Option<usize>| {
        let mut request = scroll(limit).1;
        request["max_scan"] = json!(max_scan);
        ("scan", request)
    };
    // The request, the row it starts at, and whether the page's bound applies.
    let cases = [
        // A budget of 1,024 rows; the last 60 rows, to keep this short.
        (scroll(3), 2_541, true),
        // A budget of 4,096 rows.
        (scroll(100), 1, true),
        (scroll(800), 1, true),
        (scan(100, Some(2_000)), 1, true),
        // The 600 rows fit under 601 and are all scanned: one row short of
        // the budget, as the 540 matches are.
        (scan(800, Some(601)), 1, true),
        (scan(800, Some(600)), 1, false),
        // Both plans end the first page at the budget, on the same row.
        (scan(800, Some(300)), 1, false),
        // No budget is no page.
        (scan(800, None), 1, false),
    ];
    for ((endpoint, request), first, bound) in cases {
        let name = format!(
            "{endpoint} of {} from row {first}, max_scan {}",
            request["limit"], request["max_scan"]
        );
        let expected: Vec<u64> = (first..=PAGE_ROWS).filter(|id| asked_call(*id)).collect();
        let observe = |enabled: bool| {
            let before = node_reads_for_test();
            let pages = with_selective_must_for_test(enabled, || {
                pages_from(&ctx, endpoint, &request, json!(first))
            });
            (pages, node_reads_for_test() - before)
        };
        let (pages, reads) = observe(true);
        let (switched_off, reads_off) = observe(false);
        assert_eq!(pages, switched_off, "{name}: pages");
        let rows: Vec<u64> = pages.iter().flat_map(|(rows, _)| rows.clone()).collect();
        assert_eq!(rows, expected, "{name}: rows");
        // With the bound the rows that fail the kind are read and rejected.
        assert_eq!(
            reads > reads_off,
            bound,
            "{name}: read {reads} and {reads_off} rows"
        );
    }
}

fn rows_of(pages: &[Page]) -> Vec<u64> {
    pages.iter().flat_map(|(rows, _)| rows.clone()).collect()
}

/// The page's bound is no more than its setting, and at 0 there is none.
#[test]
fn a_pages_bound_is_no_more_than_its_setting() {
    let bound = |page_max: usize, max_rows: usize| {
        with_selective_must_page_for_test(page_max, 32, || page_selective_rows(max_rows))
    };
    assert_eq!(bound(600, 4_095), Some(600));
    // Under the page's budget still.
    assert_eq!(bound(600, 513), Some(513));
    // A setting over a scroll's largest budget serves a `/points/scan` page.
    assert_eq!(bound(20_000, 999_999), Some(20_000));
    assert_eq!(bound(20_000, 1_023), Some(1_023));
    // The 512 rows already cover a setting of no more, and 0 is off.
    assert_eq!(bound(SELECTIVE_MUST_MAX_ROWS, 4_095), None);
    assert_eq!(bound(0, 4_095), None);
    assert_eq!(bound(0, 999_999), None);
    // As shipped: 4,095 rows, and 32 rows read for nothing before the
    // conditions left out come back.
    assert_eq!(selective_must_page_settings(), (4_095, 32));
}

/// `HELIX_SELECTIVE_MUST_PAGE_MAX_ROWS` is the most rows a page's selective
/// condition may match where none is under 512: the files' 600 rows fit
/// under 600 and not under 599, and at 0 a page counts nothing a second time
/// while the 512 rows stand.
#[test]
fn the_page_bound_follows_its_setting_and_is_off_at_0() {
    let ctx = page_fixture();
    let storage = ctx.collections.get_collection(NAME).unwrap();
    let read = storage.backend.begin_read().unwrap();
    let plan = |filter: &serde_json::Value, page_max: usize| {
        let typed: Filter = serde_json::from_value(filter.clone()).unwrap();
        with_selective_must_page_for_test(page_max, 32, || {
            let before = dup_scan_rows_for_test();
            let ids = scan_indexed_filter_candidates_be(&storage, &read, &typed, Some(4_096))
                .unwrap()
                .unwrap();
            let mut ids: Vec<u128> = ids.into_iter().collect();
            ids.sort_unstable();
            (ids, dup_scan_rows_for_test() - before)
        })
    };
    let (files, calls) = (rows_where(asked), rows_where(asked_call));
    let filter = asked_calls();

    assert_eq!(plan(&filter, 4_095).0, files, "the default");
    assert_eq!(plan(&filter, 600).0, files);
    let (ids, counted_twice) = plan(&filter, 599);
    assert_eq!(ids, calls, "one row short: the plan from before the bound");
    let (ids, counted_once) = plan(&filter, 0);
    assert_eq!(ids, calls, "off");
    // At 599 both conditions were counted to 600 rows for nothing.
    assert_eq!(counted_twice - counted_once, 2 * 600);

    // One file is 100 rows, under the 512: its rows are the candidates with
    // the page's bound off as with it on.
    let one_file = json!({"must": [eq("file", "f20"), eq("kind", "calls")]});
    let file_rows = rows_where(|id| file(id) == 20);
    assert_eq!(plan(&one_file, 0).0, file_rows);
    assert_eq!(plan(&one_file, 4_095).0, file_rows);
}

/// The files beside a broad condition a page's bound leaves out: the filter,
/// the rows that pass the condition, and the rows of its posting list.
type Rejecting = (&'static str, serde_json::Value, fn(u64) -> bool, usize);

fn rejecting(
    name: &'static str,
    broad: serde_json::Value,
    passes: fn(u64) -> bool,
    broad_rows: usize,
) -> Rejecting {
    let files = asked_calls()["must"][0].clone();
    (name, json!({"must": [files, broad]}), passes, broad_rows)
}

fn rejecting_cases() -> Vec<Rejecting> {
    vec![
        // The page's first 100 rows fail the condition.
        rejecting(
            "rejected rows first",
            eq("early", "in"),
            |id| !EARLY_ROWS.contains(&id),
            5_900,
        ),
        // 300 rows pass, then 100 fail in a run.
        rejecting(
            "a run after rows that pass",
            eq("late", "in"),
            |id| !LATE_ROWS.contains(&id),
            5_900,
        ),
        // One row in ten fails, all through the page.
        rejecting(
            "spread through the page",
            eq("kind", "calls"),
            |id| kind(id) == "calls",
            5_400,
        ),
    ]
}

/// A page that has read 32 rows it did not return takes the conditions its
/// bound left out back for the rows it has not scanned yet: each condition's
/// posting list is read once, and of the rest only the rows in it are read.
/// It counts as each batch ends, so a page reads whole batches until 32 of
/// its rows have failed and its matches alone after that, where it read
/// every row of the selective condition, wherever in the page the rejected
/// rows are; and every page is the kill switch's. At 0 rows the conditions
/// are back before the first row is read; a setting no page reaches leaves
/// the page as the bound alone had it.
#[test]
fn a_page_takes_the_left_out_conditions_back_after_32_rows_read_for_nothing() {
    let ctx = page_fixture();
    let files: Vec<u64> = (1..=PAGE_ROWS).filter(|id| asked(*id)).collect();
    let candidates = files.len();
    for (name, filter, passes, broad_rows) in rejecting_cases() {
        let expected: Vec<u64> = files.iter().copied().filter(|id| passes(*id)).collect();
        assert!(candidates - expected.len() > 32, "{name}: rows that fail");
        // What a page of every match reads: whole batches until 32 of its
        // rows have failed, then its matches alone. That is its matches, the
        // 32 rows, and less than a batch more.
        let (mut read, mut failed) = (0, 0);
        for rows in files.chunks(scan_read_batch()) {
            let pass = rows.iter().filter(|id| passes(**id)).count();
            read += if failed < 32 { rows.len() } else { pass };
            failed += rows.len() - pass;
        }
        let for_nothing = read - expected.len();
        assert!(
            (32..32 + scan_read_batch()).contains(&for_nothing),
            "{name}: {for_nothing} rows for nothing"
        );
        // The pages of a scroll of `limit`, the rows read and the posting
        // rows walked, with the conditions back after `reject_rows` rows or,
        // for `None`, under the kill switch.
        let observe = |limit: usize, reject_rows: Option<usize>| {
            let request = json!({"limit": limit, "with_payload": true, "filter": filter});
            let (reads, walked) = (node_reads_for_test(), dup_scan_rows_for_test());
            let run = || pages_from(&ctx, "scroll", &request, json!(1));
            let pages = match reject_rows {
                Some(rows) => with_selective_must_page_for_test(4_095, rows, run),
                None => with_selective_must_for_test(false, run),
            };
            (
                pages,
                node_reads_for_test() - reads,
                dup_scan_rows_for_test() - walked,
            )
        };

        // A page of 800 holds every match.
        let (switched_off, reads_off, _) = observe(800, None);
        assert_eq!(rows_of(&switched_off), expected, "{name}: rows");
        assert_eq!(reads_off, expected.len(), "{name}: the kill switch");
        let (never, reads_never, walked_never) = observe(800, Some(usize::MAX));
        assert_eq!(never, switched_off, "{name}: pages, never back");
        assert_eq!(reads_never, candidates, "{name}: never back");
        let (pages, reads, walked) = observe(800, Some(32));
        assert_eq!(pages, switched_off, "{name}: pages");
        assert_eq!(reads, read, "{name}: rows read");
        assert_eq!(walked - walked_never, broad_rows, "{name}: its list");
        let (at_once, reads_at_once, _) = observe(800, Some(0));
        assert_eq!(at_once, switched_off, "{name}: pages, back at once");
        assert_eq!(reads_at_once, expected.len(), "{name}: back at once");

        // Pages of 100: each counts its own rows read for nothing.
        let (switched_off, _, _) = observe(100, None);
        let (pages, reads, _) = observe(100, Some(32));
        assert_eq!(pages, switched_off, "{name}: pages of 100");
        assert!(
            reads < expected.len() + (32 + scan_read_batch()) * pages.len(),
            "{name}: {reads} rows read over {} pages",
            pages.len()
        );
    }
}

/// Under the 512 rows nothing comes back: a file of 100 rows that all fail
/// the broad condition is read through, as before the page's bound, and the
/// condition's posting list is not read for its ids.
#[test]
fn a_condition_left_out_under_512_rows_stays_out() {
    let ctx = page_fixture();
    let filter = json!({"must": [eq("file", "f20"), eq("early", "in")]});
    let request = json!({"limit": 800, "with_payload": true, "filter": filter});
    let (reads, walked) = (node_reads_for_test(), dup_scan_rows_for_test());
    let pages = with_selective_must_page_for_test(4_095, 32, || {
        pages_from(&ctx, "scroll", &request, json!(1))
    });
    assert_eq!(pages, [(vec![], serde_json::Value::Null)]);
    assert_eq!(node_reads_for_test() - reads, 100);
    let walked = dup_scan_rows_for_test() - walked;
    assert!(walked < 5_900, "walked {walked} posting rows");
}

/// A page under the bound keeps the page's clock: planned past its wall, it
/// reads its next row alone and hands on its offset, and takes no condition
/// back for that row, though at 0 rows they were due before the first.
#[test]
fn a_page_under_the_bound_planned_past_its_wall_reads_one_row() {
    let ctx = page_fixture();
    let filter = json!({"must": [asked_calls()["must"][0], eq("late", "in")]});
    let request = json!({"limit": 800, "with_payload": true, "filter": filter});
    // The rows read and the posting rows walked, with the conditions due
    // back after `reject_rows` rows.
    let observed = [32, 0].map(|reject_rows| {
        let before = (node_reads_for_test(), dup_scan_rows_for_test());
        let reply = with_selective_must_page_for_test(4_095, reject_rows, || {
            with_indexed_scroll_times_for_test(300, 600, Duration::from_millis(700), || {
                post(&ctx, "scroll", request.clone())
            })
        });
        let rows = reply["result"]["points"].as_array().unwrap();
        assert_eq!(rows.len(), 1, "back after {reject_rows}");
        assert_eq!(rows[0]["payload"]["size"], 2_001);
        assert!(!reply["result"]["next_page_offset"].is_null());
        (
            node_reads_for_test() - before.0,
            dup_scan_rows_for_test() - before.1,
        )
    });
    assert_eq!(observed[0].0, 1, "rows read");
    assert_eq!(observed[1], observed[0], "due back at once");
}

/// A filter of the page fixture, the rows that match it, and whether a page
/// with a budget of 4,096 rows leaves a `must` of it to the recheck beside a
/// selective one of over 512 rows.
type PageShape = (&'static str, serde_json::Value, fn(u64) -> bool, bool);

fn page_shape(
    name: &'static str,
    filter: serde_json::Value,
    matches: fn(u64) -> bool,
    bound: bool,
) -> PageShape {
    (name, filter, matches, bound)
}

fn page_shapes() -> Vec<PageShape> {
    let files = || asked_calls()["must"][0].clone();
    let calls = || eq("kind", "calls");
    let sizes = |from: u64, to: u64| json!({"key": "size", "range": {"gte": from, "lte": to}});
    vec![
        // A range of 600 rows is the selective one as the files are.
        page_shape(
            "a range",
            json!({"must": [sizes(2_001, 2_600), calls()]}),
            asked_call,
            true,
        ),
        // A range of 5,400 rows is the broad one beside the files.
        page_shape(
            "a broad range",
            json!({"must": [files(), sizes(1, 5_400)]}),
            asked,
            true,
        ),
        // What is not a top-level `must` stays in the filter the candidates
        // are taken from.
        page_shape(
            "with must_not",
            json!({"must": [files(), calls()], "must_not": [sizes(2_301, 2_400)]}),
            |id| asked_call(id) && !LATE_ROWS.contains(&id),
            true,
        ),
        page_shape(
            "with should",
            json!({"must": [files(), calls()], "should": [
                eq("file", "f20"),
                sizes(2_500, 6_000)
            ]}),
            |id| asked_call(id) && (file(id) == 20 || id >= 2_500),
            true,
        ),
        page_shape(
            "with a nested sibling",
            json!({"must": [files(), calls(), {"must": [sizes(2_101, 6_000)]}]}),
            |id| asked_call(id) && id >= 2_101,
            true,
        ),
        // A nested filter is not counted, so it is never the selective one:
        // beside a broad sibling alone the page is planned as before. The
        // page's bound cannot reach this shape.
        page_shape(
            "selective inside a nested must",
            json!({"must": [calls(), {"must": [files()]}]}),
            asked_call,
            false,
        ),
    ]
}

/// The shapes a page's bound can take over 512 rows: for each, every page of
/// a scroll of 3 (over the last 60 rows) and of 800, its rows and the offset
/// it hands on, is the page under the kill switch, and the rows are the ones
/// the fixture says match. Only the page's bound takes them: with it off
/// nothing is left out of their candidates.
#[test]
fn pages_of_every_shape_on_the_page_bound_are_the_kill_switchs() {
    let ctx = page_fixture();
    let storage = ctx.collections.get_collection(NAME).unwrap();
    for (name, filter, matches, bound) in page_shapes() {
        let typed: Filter = serde_json::from_value(filter.clone()).unwrap();
        let read = storage.backend.begin_read().unwrap();
        let left_out = |page_max: usize| {
            with_selective_must_page_for_test(page_max, 32, || {
                with_selective_must_for_test(true, || {
                    without_broad_musts(&storage, &read, &typed, 4_095).unwrap()
                })
            })
            .is_some()
        };
        assert_eq!(left_out(4_095), bound, "{name}: on the page's bound");
        assert!(!left_out(0), "{name}: with the page's bound off");
        drop(read);

        for (limit, first) in [(3, 2_541), (800, 1)] {
            let request = json!({"limit": limit, "with_payload": true, "filter": filter});
            let expected: Vec<u64> = (first..=PAGE_ROWS).filter(|id| matches(*id)).collect();
            let observe = |enabled: bool| {
                with_selective_must_for_test(enabled, || {
                    pages_from(&ctx, "scroll", &request, json!(first))
                })
            };
            let pages = observe(true);
            assert_eq!(pages, observe(false), "{name}: pages of {limit}");
            assert_eq!(rows_of(&pages), expected, "{name}: rows, pages of {limit}");
        }
    }
}

/// A `has_id` list of over 512 ids is a page's selective condition as a
/// field condition is: its ids are the candidates and the broad sibling's
/// list is counted, not read. Every page is the page with the page's bound
/// off, where the list cuts down the sibling's rows once they are read.
/// Under the kill switch a list is no candidate set at all and a page may end
/// at its budget before the first listed row, as it did for a list of 512 ids
/// or fewer: those pages are compared by their rows.
#[test]
fn a_list_of_over_512_ids_is_a_pages_selective_condition() {
    let ctx = page_fixture();
    let storage = ctx.collections.get_collection(NAME).unwrap();
    let listed: Vec<u64> = (2_001..=2_600).collect();
    let filter = json!({"must": [{"has_id": listed}, eq("kind", "calls")]});
    let typed: Filter = serde_json::from_value(filter.clone()).unwrap();
    let read = storage.backend.begin_read().unwrap();
    let plan = |page_max: usize| {
        with_selective_must_page_for_test(page_max, 32, || {
            let before = dup_scan_rows_for_test();
            let ids = scan_indexed_filter_candidates_be(&storage, &read, &typed, Some(4_096))
                .unwrap()
                .unwrap();
            let mut ids: Vec<u128> = ids.into_iter().collect();
            ids.sort_unstable();
            (ids, dup_scan_rows_for_test() - before)
        })
    };
    let (ids, visited) = plan(4_095);
    assert_eq!(ids, rows_where(asked), "the listed rows, to be rechecked");
    // The kind counted to 513 rows, to 4,096 and to 4,801; the list is not read.
    assert!(
        visited <= (SELECTIVE_MUST_MAX_ROWS + 1) + 4_096 + 4_801,
        "visited {visited} posting rows"
    );
    let (ids, visited) = plan(0);
    assert_eq!(
        ids,
        rows_where(asked_call),
        "the kind's rows that are listed"
    );
    assert!(visited >= 5_400, "the kind is read for its ids: {visited}");
    drop(read);

    for (limit, first) in [(3, 2_541), (800, 1)] {
        let request = json!({"limit": limit, "with_payload": true, "filter": filter});
        let expected: Vec<u64> = (first..=PAGE_ROWS).filter(|id| asked_call(*id)).collect();
        let with_bound = |page_max: usize| {
            with_selective_must_page_for_test(page_max, 32, || {
                pages_from(&ctx, "scroll", &request, json!(first))
            })
        };
        let pages = with_bound(4_095);
        assert_eq!(pages, with_bound(0), "pages of {limit}");
        assert_eq!(rows_of(&pages), expected, "rows, pages of {limit}");
        let switched_off = with_selective_must_for_test(false, || {
            pages_from(&ctx, "scroll", &request, json!(first))
        });
        assert_eq!(rows_of(&switched_off), expected, "kill switch, {limit}");
    }
}
