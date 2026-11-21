use super::*;

pub(super) fn indexed_condition_candidates_are_exact(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    condition: &Condition,
) -> bool {
    match condition {
        // Text candidates are resolved by a substring walk over index terms
        // (historically they were only the exact keyword hits — a subset).
        // Only the walk itself can prove its set exact
        // (`indexed_filter_ids_be`), so text is never exact statically.
        Condition::Field(field) => {
            indexed_field_condition_supported(storage, field)
                && !matches!(field.match_cond, Some(MatchCondition::Text(_)))
        }
        Condition::Nested(filter) => indexed_filter_candidates_are_exact(storage, filter),
        Condition::HasId(_) | Condition::IsEmpty(_) | Condition::IsNull(_) => false,
    }
}

pub(super) fn indexed_filter_candidates_are_exact(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    filter: &Filter,
) -> bool {
    (!filter.must.is_empty() || !filter.should.is_empty())
        && filter
        .must
        .iter()
        .all(|condition| indexed_condition_candidates_are_exact(storage, condition))
        && filter
            .must_not
            .iter()
            .all(|condition| indexed_condition_candidates_are_exact(storage, condition))
        && filter
            .should
            .iter()
            .all(|condition| indexed_condition_candidates_are_exact(storage, condition))
        // Candidate derivation ignores `min_should`, so the set is a superset.
        && filter.min_should.is_none()
}

/// Read-free check that `indexed_filter_ids_be` could report `filter`'s set
/// as exact: like [`indexed_filter_candidates_are_exact`], except a text
/// clause on a keyword index qualifies (its walk may be exact), in `must_not`
/// too since the planner subtracts an exact text walk.
pub(super) fn indexed_filter_candidates_may_be_exact(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    filter: &Filter,
) -> bool {
    let positive = |condition: &Condition| match condition {
        Condition::Field(field) => indexed_field_condition_supported(storage, field),
        Condition::Nested(nested) => indexed_filter_candidates_may_be_exact(storage, nested),
        Condition::HasId(_) | Condition::IsEmpty(_) | Condition::IsNull(_) => false,
    };
    (!filter.must.is_empty() || !filter.should.is_empty())
        && filter.min_should.is_none()
        && filter.must.iter().all(positive)
        && filter.should.iter().all(positive)
        && filter.must_not.iter().all(|condition| {
            indexed_condition_candidates_are_exact(storage, condition)
                || (is_text_field_condition(condition) && positive(condition))
        })
}
