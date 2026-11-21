//! A `match.text` walk reads every row of its field's index, whatever the
//! text asked for, and gives up past its cap. One that gave up was not
//! remembered: each page of a scroll, and each search with a text clause on
//! that field, read the cap's rows again to learn the same.

use super::filter_fixture::fixture;
use super::tests::make_input;
use super::*;
use crate::helix_engine::storage_core::backend_lsm::dup_scan_rows_for_test;

/// The fixture's `language` index has a row a point: three times this.
const CAP: usize = 20;
const POINTS: u32 = 60;

struct Page {
    ids: Vec<u128>,
    indexed: bool,
    /// Index rows read to plan and serve it.
    index_rows: usize,
}

/// A scroll page of five rows whose `language` holds `text`, from `offset`,
/// on a read of its own, as each page of a scroll is.
fn page(storage: &HelixGraphStorage, text: &str, offset: u128) -> Page {
    page_under(storage, text, offset, CAP)
}

/// [`page`] with the walk's cap at `cap`; none where it is 0.
fn page_under(storage: &HelixGraphStorage, text: &str, offset: u128, cap: usize) -> Page {
    let filter: Filter = serde_json::from_value(serde_json::json!({"must": [
        {"key": "language", "match": {"text": text}}
    ]}))
    .unwrap();
    let opts = PointScanOptions {
        collection: "filter_closeout",
        limit: 5,
        offset_id: offset,
        filter: &filter,
        with_payload: &WithPayload::All,
        with_vectors: &WithVectors::None,
        partition: None,
        max_scan: Some(1_024),
        deadline: None,
        indexed_deadline: None,
        filter_hash: scan_filter_hash(&filter),
        cursor_generation: None,
        fuse_pending: false,
    };
    let read = storage.backend.begin_read().unwrap();
    let before = dup_scan_rows_for_test();
    let scan =
        with_text_walk_max_for_test(cap, || execute_point_scan_be(storage, &read, &opts)).unwrap();
    Page {
        ids: scan.points.iter().filter_map(point_json_id).collect(),
        indexed: matches!(scan.plan, PointScanPlan::IndexedCandidates),
        index_rows: dup_scan_rows_for_test() - before,
    }
}

#[test]
fn a_text_walk_that_gave_up_is_not_walked_again_at_the_same_snapshot() {
    let (ctx, storage) = fixture(POINTS);

    // "ust" is in "rust", the even ids, and is no whole value of the field:
    // nothing is counted for it, and the walk is all the index rows read.
    let first = page(&storage, "ust", 0);
    assert_eq!(first.ids, vec![2, 4, 6, 8, 10]);
    assert!(!first.indexed, "a walk that gave up names no candidates");
    assert_eq!(
        first.index_rows,
        CAP + 1,
        "the walk stops a row past its cap"
    );

    // The next page, and another text on the same field: the same rows of
    // the same snapshot would be read to the same end.
    let second = page(&storage, "ust", 11);
    assert_eq!(second.ids, vec![12, 14, 16, 18, 20]);
    assert_eq!(second.index_rows, 0, "the page walked the field again");
    let other = page(&storage, "ytho", 0);
    assert_eq!(other.ids, vec![1, 3, 5, 7, 9]);
    assert_eq!(other.index_rows, 0, "another text walked the field again");

    // The mark is of a walk under that cap, and says nothing of a walk
    // under none: that one is made, and it resolves the text.
    let uncapped = page_under(&storage, "ust", 0, 0);
    assert_eq!(uncapped.ids, vec![2, 4, 6, 8, 10]);
    assert!(
        uncapped.indexed,
        "with no cap the walk names the candidates"
    );
    assert_eq!(
        uncapped.index_rows, POINTS as usize,
        "every row of the field, once"
    );
    // What that walk found is kept, and a page under the cap is still told
    // that its walk would give up.
    let capped_again = page(&storage, "ust", 0);
    assert!(!capped_again.indexed);
    assert_eq!(
        capped_again.index_rows, 0,
        "the page walked the field again"
    );

    // Rows leave the field until it is under the cap. That is a write, so a
    // new snapshot: the walk is made again, and now it resolves the text.
    let gone: Vec<u32> = (21..=POINTS).collect();
    let input = make_input(
        &ctx,
        "POST",
        "/collections/filter_closeout/points/delete",
        serde_json::to_vec(&serde_json::json!({"points": gone})).unwrap(),
        HashMap::from([("name".into(), "filter_closeout".into())]),
    );
    let mut response = Response::new();
    handle_delete_points(&input, &mut response).unwrap();
    assert_eq!(response.status, 200);

    let after = page(&storage, "ust", 0);
    assert_eq!(after.ids, vec![2, 4, 6, 8, 10]);
    assert!(after.indexed, "under the cap the walk names the candidates");
    assert_eq!(after.index_rows, CAP, "every row left in the field, once");
}
