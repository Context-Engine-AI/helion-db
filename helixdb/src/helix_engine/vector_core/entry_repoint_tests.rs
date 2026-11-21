//! A segment keeps a usable entry point when a hard delete takes it
//! (`delete_vectors_batch_be`) and when a build publishes after a delete took
//! its entry point's row (`finalize_prepared_index_entry_be`). Built on the
//! parent module's segment and recall harness.

use super::*;
use crate::helix_engine::storage_core::backend_lsm::lsm_round_trips_for_test;

const SWITCH: &str = "HELIX_DENSE_REPOINT_ENTRY";

/// Sets the kill switch until dropped. Every test here is `serial`: the
/// switch is process-wide.
struct RepointSwitch(Option<String>);

impl RepointSwitch {
    fn set(value: &str) -> Self {
        let previous = std::env::var(SWITCH).ok();
        std::env::set_var(SWITCH, value);
        Self(previous)
    }
}

impl Drop for RepointSwitch {
    fn drop(&mut self) {
        match &self.0 {
            Some(value) => std::env::set_var(SWITCH, value),
            None => std::env::remove_var(SWITCH),
        }
    }
}

fn graph_config() -> HNSWConfig {
    HNSWConfig::new(Some(16), Some(64), Some(128))
}

/// Tight topic clusters, the shape of code embeddings (see
/// `diverse_neighbors_lift_recall_on_clustered_data`).
fn clustered_rows(count: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(17);
    let centers: Vec<Vec<f32>> = (0..40)
        .map(|_| normalized((0..32).map(|_| rng.random_range(-1.0f32..1.0)).collect()))
        .collect();
    (0..count)
        .map(|_| {
            let center = &centers[rng.random_range(0..centers.len())];
            normalized(
                center
                    .iter()
                    .map(|c| c + 0.05 * rng.random_range(-1.0f32..1.0))
                    .collect(),
            )
        })
        .collect()
}

fn entry_point(segment: &BuiltSegment) -> HVector {
    let read = segment.backend.begin_read().unwrap();
    segment.core.get_entry_point(&read).unwrap()
}

/// A hard delete as the writer applies it: the batch, its commit, then the
/// post-commit cache step.
fn hard_delete(segment: &BuiltSegment, ids: &[u128]) {
    let mut write = segment.backend.begin_write().unwrap();
    let levels = segment
        .core
        .delete_vectors_batch_be(&mut write, ids)
        .unwrap();
    segment.backend.commit(write).unwrap();
    segment.core.apply_hard_deletes_post_commit(ids, &levels);
}

/// Builds a segment over `rows` and hard-deletes its entry point. Returns the
/// segment, the deleted id, and mean recall@K at ef 128 before and after, over
/// queries whose exact answers leave that id out.
fn delete_entry_point(
    name: &str,
    hnsw: HNSWConfig,
    rows: &[Vec<f32>],
    queries: usize,
) -> (BuiltSegment, u128, f64, f64) {
    let segment = build_segment(name, hnsw, rows, Some(HNSW_KEEP_PRUNED_CONNECTIONS));
    let entry = entry_point(&segment).get_id();
    let survivors: Vec<usize> = (0..rows.len())
        .filter(|&ord| ord as u128 + 1 != entry)
        .collect();
    let queries = midpoint_queries(rows, &survivors, queries, 20261117);
    let before = mean(&recall_at(&segment.core, &queries, 128, None).recalls);
    hard_delete(&segment, &[entry]);
    let after = mean(&recall_at(&segment.core, &queries, 128, None).recalls);
    (segment, entry, before, after)
}

/// What the merge selection asks before it forces a segment's rewrite.
#[test]
#[serial_test::serial]
fn entry_point_row_is_gone_only_where_a_delete_left_the_key_on_a_deleted_row() {
    let rows = clustered_rows(1500);
    // A built segment has its entry point's row.
    let built = build_segment(
        "entry_gone_built",
        graph_config(),
        &rows,
        Some(HNSW_KEEP_PRUNED_CONNECTIONS),
    );
    let read = built.backend.begin_read().unwrap();
    assert!(!built.core.entry_point_row_is_gone(&read).unwrap());
    drop(read);

    // With the repoint off, a delete of the entry point leaves the key on a
    // row that is gone, as segments stored before the repoint still have it.
    let _off = RepointSwitch::set("0");
    let (segment, _, _, _) = delete_entry_point("entry_gone_deleted", graph_config(), &rows, 10);
    let read = segment.backend.begin_read().unwrap();
    assert!(segment.core.entry_point_row_is_gone(&read).unwrap());
}

#[test]
#[serial_test::serial]
fn delete_that_takes_the_entry_point_moves_it_to_the_highest_level_survivor() {
    let rows = clustered_rows(1500);
    // With the switch off the key stays on the deleted id, as before this
    // change: no usable entry point, so every search scans the segment.
    {
        let _off = RepointSwitch::set("0");
        let (segment, _, _, _) = delete_entry_point("entry_delete_off", graph_config(), &rows, 10);
        let read = segment.backend.begin_read().unwrap();
        assert!(!segment.core.has_index(&read).unwrap());
    }

    // Given a built segment, when a delete batch takes its entry point.
    let _on = RepointSwitch::set("1");
    let (segment, deleted, before, after) =
        delete_entry_point("entry_delete_on", graph_config(), &rows, 100);

    // Then the segment still has one, on the highest layer a survivor reaches.
    let read = segment.backend.begin_read().unwrap();
    assert!(
        segment.core.has_index(&read).unwrap(),
        "the delete left the segment without a usable entry point"
    );
    let entry = segment.core.get_entry_point(&read).unwrap();
    assert_ne!(entry.get_id(), deleted);
    let top_survivor_level = (1..=rows.len() as u128)
        .filter(|&id| id != deleted)
        .map(|id| segment.core.get_highest_level(&read, id).unwrap())
        .max()
        .unwrap();
    assert_eq!(
        entry.get_level(),
        top_survivor_level,
        "the new entry point must keep the upper layers reachable"
    );
    // And the walk from it finds what the walk from the old one found.
    assert!(
        after >= before - 0.05,
        "recall@{K} fell from {before:.3} to {after:.3}"
    );
}

/// The entry point and every neighbor it has, at every level, go in one
/// batch: no list leads to a survivor, so the bounded row scan picks one.
#[test]
#[serial_test::serial]
fn delete_that_takes_the_entry_point_and_all_its_neighbors_finds_a_survivor() {
    let rows = clustered_rows(1500);
    let segment = build_segment(
        "entry_delete_neighbors",
        graph_config(),
        &rows,
        Some(HNSW_KEEP_PRUNED_CONNECTIONS),
    );
    let entry = entry_point(&segment);
    let batch: HashSet<u128> = {
        let read = segment.backend.begin_read().unwrap();
        (0..=entry.get_level())
            .flat_map(|level| {
                segment
                    .core
                    .get_neighbor_ids(&read, entry.get_id(), level)
                    .unwrap_or_default()
            })
            .chain([entry.get_id()])
            .collect()
    };
    assert!(batch.len() > 1, "the entry point has neighbors");

    hard_delete(&segment, &batch.iter().copied().collect::<Vec<_>>());

    let read = segment.backend.begin_read().unwrap();
    assert!(segment.core.has_index(&read).unwrap());
    let entry = segment.core.get_entry_point(&read).unwrap();
    assert!(!batch.contains(&entry.get_id()));
}

/// The fallback scan of `replacement_entry_point` stops at its bound, and
/// passes over an id that has an upper-level row but no level-0 row (what a
/// build leaves for an id deleted while it ran).
#[test]
fn entry_replacement_scan_is_bounded_and_needs_a_level_zero_row() {
    let backend = Arc::new(AnyBackend::Lsm(
        LsmBackend::open_in_memory(&format!("entry-scan-{}", uuid::Uuid::new_v4())).unwrap(),
    ));
    let dir = TempDir::new().unwrap();
    let core = VectorCore::new_named_lsm_with_dir(
        "entry_scan",
        graph_config(),
        DistanceMetric::Cosine,
        SpindleConfig {
            mode: SpindleMode::None,
            ..SpindleConfig::default()
        },
        Some(dir.path()),
        4,
        Arc::clone(&backend),
    )
    .unwrap();
    let mut write = backend.begin_write().unwrap();
    for id in 1..=6u128 {
        core.insert_flat_be(&mut write, &[id as f32, 1.0, 0.0, 0.0], Some(id), None)
            .unwrap();
    }
    // Id 9: a level-3 row and nothing at level 0.
    backend
        .put(
            &mut write,
            core.seg_ns(SegmentDb::Vectors),
            &VectorCore::vector_key(9, 3),
            &encode_vector(&[9.0, 1.0, 0.0, 0.0], &core.spindle).unwrap(),
        )
        .unwrap();
    backend.commit(write).unwrap();

    let read = backend.begin_read().unwrap();
    let old_entry = HVector::from_slice(1, 0, vec![]);
    let replacement = |max_scan_rows| {
        core.replacement_entry_point(&read, &old_entry, |id| id == 2, max_scan_rows)
            .unwrap()
            .map(|entry| entry.get_id())
    };
    // Rows 1 (the old entry) and 2 (deleted) come first; 3 is the first survivor.
    assert_eq!(replacement(usize::MAX), Some(3));
    assert_eq!(replacement(3), Some(3));
    assert_eq!(replacement(2), None, "the scan must stop at its bound");
}

/// A delete batch that does not hold the entry point pays one read for the
/// entry key, per segment that owns one of its ids.
#[test]
#[serial_test::serial]
fn delete_batch_reads_the_entry_key_once() {
    let rows = clustered_rows(600);
    let round_trips = |switch: &str| {
        let _switch = RepointSwitch::set(switch);
        let segment = build_segment(
            &format!("entry_delete_cost_{switch}"),
            graph_config(),
            &rows,
            Some(HNSW_KEEP_PRUNED_CONNECTIONS),
        );
        let entry = entry_point(&segment).get_id();
        let ids: Vec<u128> = (1..=rows.len() as u128)
            .filter(|&id| id != entry)
            .take(64)
            .collect();
        let mut write = segment.backend.begin_write().unwrap();
        let before = lsm_round_trips_for_test();
        segment
            .core
            .delete_vectors_batch_be(&mut write, &ids)
            .unwrap();
        let spent = lsm_round_trips_for_test() - before;
        segment.backend.commit(write).unwrap();
        spent
    };
    assert_eq!(round_trips("1"), round_trips("0") + 1);
}

/// A delete lands between a build's export and its publish and takes the id
/// the build chose as entry point: the publish must not write the key over a
/// row that is gone.
#[test]
#[serial_test::serial]
fn build_published_after_its_entry_point_was_deleted_gets_a_usable_one() {
    let rows = clustered_rows(1500);
    let backend = Arc::new(AnyBackend::Lsm(
        LsmBackend::open_in_memory(&format!("entry-publish-{}", uuid::Uuid::new_v4())).unwrap(),
    ));
    let dir = TempDir::new().unwrap();
    let core = VectorCore::new_named_lsm_with_dir(
        "entry_publish",
        graph_config(),
        DistanceMetric::Cosine,
        SpindleConfig {
            mode: SpindleMode::None,
            ..SpindleConfig::default()
        },
        Some(dir.path()),
        rows[0].len(),
        Arc::clone(&backend),
    )
    .unwrap();
    let mut write = backend.begin_write().unwrap();
    for (ord, row) in rows.iter().enumerate() {
        core.insert_flat_be(&mut write, row, Some(ord as u128 + 1), None)
            .unwrap();
    }
    backend.commit(write).unwrap();

    // Given a build over the flat rows, and a delete of its entry point
    // before it publishes.
    let read = backend.begin_read().unwrap();
    let mut prepared = core.prepare_index_from_flat(&read).unwrap().unwrap();
    drop(read);
    let built_entry = prepared.point_ids[prepared.entry_ord];
    let top_survivor_level = (0..prepared.point_ids.len())
        .filter(|&ord| ord != prepared.entry_ord)
        .map(|ord| prepared.levels[ord])
        .max()
        .unwrap();
    let mut write = backend.begin_write().unwrap();
    let levels = core
        .delete_vectors_batch_be(&mut write, &[built_entry])
        .unwrap();
    backend.commit(write).unwrap();
    core.apply_hard_deletes_post_commit(&[built_entry], &levels);

    // When the build publishes.
    let mut write = backend.begin_write().unwrap();
    let total = prepared.point_ids.len();
    core.flush_prepared_index_chunk_be(&mut write, &mut prepared, 0, total)
        .unwrap();
    core.finalize_prepared_index_entry_be(&mut write, &prepared)
        .unwrap();
    backend.commit(write).unwrap();

    // Then the entry point is the highest-level id whose row is still there,
    // and the walk from it answers.
    let read = backend.begin_read().unwrap();
    assert!(
        core.has_index(&read).unwrap(),
        "the publish wrote the entry key over a deleted row"
    );
    let entry = core.get_entry_point(&read).unwrap();
    assert_ne!(entry.get_id(), built_entry);
    assert_eq!(entry.get_level(), top_survivor_level);
    drop(read);
    let survivors: Vec<usize> = (0..rows.len())
        .filter(|&ord| ord as u128 + 1 != built_entry)
        .collect();
    let queries = midpoint_queries(&rows, &survivors, 100, 20261117);
    let recall = mean(&recall_at(&core, &queries, 128, None).recalls);
    assert!(
        recall >= 0.5,
        "recall@{K} from the new entry point: {recall:.3}"
    );
}

/// Recall before and after a hard delete of the entry point on `eval_inputs`.
/// Run with
/// `cargo test -p helixdb --lib entry_repoint_tests::eval -- --ignored --nocapture`.
#[test]
#[ignore = "offline eval; needs HNSW_EVAL_DIR"]
fn eval_entry_repoint_on_real_vectors() {
    for (name, rows) in eval_inputs() {
        for build in 0..eval_builds() {
            let (_, _, before, after) = delete_entry_point("eval", eval_hnsw(), &rows, 200);
            println!(
                "ROW {name} n={} build={build}: recall@{K} before {before:.4} after {after:.4}",
                rows.len()
            );
        }
    }
}
