use super::tests::setup_lsm_manager_with_flat_segment;
use super::*;

#[test]
fn empty_candidates_reject_unknown_named_vector() {
    let (backend, manager, _dir) = setup_lsm_manager_with_flat_segment(
        "empty_candidate_validation",
        "dense__seg_000000",
        &[(1, [1.0, 0.0, 0.0])],
    );
    let read = backend.begin_read().unwrap();

    let result = manager.dense_search_candidate_ids_exact_be(
        &read,
        "missing_vector",
        &[1.0, 0.0, 0.0],
        1,
        [],
        None,
    );

    assert!(
        matches!(result, Err(VectorError::VectorCoreError(ref error))
        if error == "Named vector 'missing_vector' not found"),
        "{result:?}"
    );
}

#[test]
fn absent_candidates_reject_unknown_named_vector() {
    let (backend, manager, _dir) = setup_lsm_manager_with_flat_segment(
        "absent_candidate_validation",
        "dense__seg_000000",
        &[(1, [1.0, 0.0, 0.0])],
    );
    let read = backend.begin_read().unwrap();

    let result = manager.dense_search_candidate_ids_exact_be(
        &read,
        "missing_vector",
        &[1.0, 0.0, 0.0],
        1,
        [999],
        None,
    );

    assert!(
        matches!(result, Err(VectorError::VectorCoreError(ref error))
        if error == "Named vector 'missing_vector' not found"),
        "{result:?}"
    );
}

#[test]
fn zero_limit_preserves_early_success_for_unknown_named_vector() {
    let (backend, manager, _dir) = setup_lsm_manager_with_flat_segment(
        "zero_limit_validation",
        "dense__seg_000000",
        &[(1, [1.0, 0.0, 0.0])],
    );
    let read = backend.begin_read().unwrap();

    let result = manager
        .dense_search_candidate_ids_exact_be(&read, "missing_vector", &[], 0, [], None)
        .unwrap();

    assert!(result.is_empty());
}
