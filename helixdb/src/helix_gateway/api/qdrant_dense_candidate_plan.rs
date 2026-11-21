use super::*;

#[cfg(test)]
thread_local! {
    static PREFILTER_MAX_OVERRIDE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Test-only: run `f` with `dense_text_prefilter_max` forced on this thread.
#[cfg(test)]
pub(super) fn with_prefilter_max_for_test<R>(max: usize, f: impl FnOnce() -> R) -> R {
    let previous = PREFILTER_MAX_OVERRIDE.with(|cell| cell.replace(Some(max)));
    let out = f();
    PREFILTER_MAX_OVERRIDE.with(|cell| cell.set(previous));
    out
}

/// Largest `match.text` walk (a directory scope) the planner drops as too
/// broad but still hands to the filter closures as a membership prefilter
/// (`ProbedCandidates::prefilter`). `0` disables.
fn dense_text_prefilter_max() -> usize {
    #[cfg(test)]
    if let Some(max) = PREFILTER_MAX_OVERRIDE.with(std::cell::Cell::get) {
        return max;
    }
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| env_usize_or("HELIX_DENSE_TEXT_PREFILTER_MAX", 100_000))
}

/// A dropped set is worth keeping as a prefilter only if it rejects some
/// points; the walk that built it already ran, so keeping it costs a set
/// lookup per filter call, far below the node read it can save.
fn keep_as_prefilter(ids: HashSet<u128>, total: usize) -> Option<HashSet<u128>> {
    let max = dense_text_prefilter_max();
    (max > 0 && ids.len() <= max && ids.len() < total).then_some(ids)
}

/// Index rows a remaining-clause walk may scan per node read it saves: a
/// row is a sequential scan step, a node read a point lookup plus decode.
const REST_ROWS_PER_NODE_READ: usize = 64;

/// Exact matches of `filter`'s clauses other than the walked uncounted text
/// ones (`walked`, indices into `must`), when every remaining clause can be
/// and is resolved exactly from the indexes. With an exact walk, walk ∩ rest
/// is exactly the filter's matches. Bounded like the prefilter: no set larger
/// than `dense_text_prefilter_max` is materialized, nothing is read when a
/// remaining clause cannot resolve exactly, and the remaining clauses (up to
/// `total` rows) are resolved only when that is cheaper than one node read
/// per walked id.
fn exact_rest_ids(
    storage: &HelixGraphStorage,
    r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    filter: &Filter,
    walked: &[usize],
    walk_len: usize,
    total: usize,
) -> Result<Option<HashSet<u128>>, GraphError> {
    if !text_walk_exact_enabled() || total > dense_text_prefilter_max() {
        return Ok(None);
    }
    if walk_len.saturating_mul(REST_ROWS_PER_NODE_READ) < total {
        // The search then checks its `walk_len` rows one node read each, at
        // every ask. Counted, with the rows, to weigh a rule that pays the
        // remaining clauses' walk once a snapshot instead.
        metrics::counter!("helix_dense_rest_walk_declined_total").increment(1);
        metrics::counter!("helix_dense_rest_walk_declined_rows_total").increment(walk_len as u64);
        return Ok(None);
    }
    let rest = Filter {
        must: filter
            .must
            .iter()
            .enumerate()
            .filter(|(index, _)| !walked.contains(index))
            .map(|(_, condition)| condition.clone())
            .collect(),
        must_not: filter.must_not.clone(),
        should: filter.should.clone(),
        min_should: filter.min_should.clone(),
    };
    if !indexed_filter_candidates_may_be_exact(storage, &rest) {
        return Ok(None);
    }
    Ok(match indexed_filter_ids_be(storage, r, &rest, 0)? {
        Some((rest_ids, true)) => Some(rest_ids),
        _ => None,
    })
}

/// True when `filter`'s indexed set is an intersection of `must` text walks,
/// a superset of the matches for every stored value (the walk applies the
/// same lowercase substring test, extra hits are only array values). Keyword
/// lookups miss nested-array and object values, and a keyword `must_not`
/// subtraction can drop matches (a `null` exclusion is indexed as `"null"` but
/// never matched), so such sets are no prefilter. A text exclusion is only
/// subtracted when its walk is exact.
fn set_is_text_walk_superset(storage: &HelixGraphStorage, filter: &Filter) -> bool {
    let subtracts = filter
        .must_not
        .iter()
        .any(|condition| !is_text_field_condition(condition))
        && filter.must_not.iter().all(|condition| {
            is_text_field_condition(condition)
                || indexed_condition_candidates_are_exact(storage, condition)
        });
    filter.should.is_empty()
        && !subtracts
        && filter.must.iter().all(|condition| {
            is_text_field_condition(condition)
                || matches!(
                    condition,
                    Condition::HasId(_) | Condition::IsEmpty(_) | Condition::IsNull(_)
                )
        })
}

/// Explicit `must` `has_id` lists (the client's ReFRAG point-id gate): the planner
/// cannot resolve them through a payload index, so filtered HNSW had to hunt
/// a few hundred ids through the whole graph. The id set is a superset of the
/// matches (ids may be absent, other clauses still apply); `HasId` is never
/// index-exact, so callers recheck the full filter per candidate.
pub(super) fn must_has_id_candidates(filter: &Filter) -> Option<HashSet<u128>> {
    filter
        .must
        .iter()
        .filter_map(|condition| match condition {
            Condition::HasId(has_id) => Some(has_id.point_ids().collect::<HashSet<u128>>()),
            _ => None,
        })
        .reduce(|acc, ids| acc.intersection(&ids).copied().collect())
}

/// Dense-search candidate resolution on the backend seam, shared by
/// `/points/search` (`drop_broad = false`: it historically keeps whatever it
/// materialized) and `/points/query` + hybrid channels (`drop_broad = true`).
/// Uncounted text walks can narrow a broad scope; uncounted nested clauses
/// have no posting-work bound and fall back to full predicate evaluation.
pub(super) fn dense_filter_candidates_be(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    filter: &Filter,
    probe_enabled: bool,
    exact_keep_max: usize,
    drop_broad: bool,
) -> Result<ProbedCandidates, GraphError> {
    let total_vectors = collection_vector_count_be(storage);
    if let Some(ids) = must_has_id_candidates(filter)
        .filter(|ids| should_exact_score_index_candidates(ids, total_vectors))
    {
        return Ok(ProbedCandidates {
            candidates: Some(ids),
            probe_estimate: None,
            probe_skipped: false,
            prefilter: None,
            exact: false,
        });
    }
    let total = total_vectors as usize;
    let filter_exact = indexed_filter_candidates_are_exact(storage, filter);
    let mut keep_cap = vector_query_broad_candidate_cap(total);
    if filter_exact {
        keep_cap = keep_cap.max(exact_keep_max);
    }
    let too_broad = |count: usize| total > 0 && count > keep_cap;

    let text_skip_max = if total > 0 {
        text_candidate_skip_max().min(keep_cap)
    } else {
        text_candidate_skip_max()
    };
    let mut probe_estimate = None;
    if probe_enabled && total > 0 && !filter.must.is_empty() {
        let probe = must_count_probe_be(storage, r, filter, keep_cap)?;
        probe_estimate = probe.estimate;
        if probe.estimate.is_some_and(too_broad) {
            // The counted clauses are broad. Resolve only uncounted text
            // (never the broad sibling) and keep them when small enough to
            // recheck. Callers already recheck a non-exact filter per
            // candidate; an exact one is rechecked here so the set is exact.
            let mut candidates = None;
            let mut prefilter = None;
            let mut exact = false;
            if !probe.unprobed_indexable.is_empty() {
                let walked: Vec<usize> = probe
                    .unprobed_indexable
                    .iter()
                    .copied()
                    .filter(|&index| is_text_field_condition(&filter.must[index]))
                    .collect();
                let uncounted = Filter {
                    must: walked
                        .iter()
                        .map(|&index| filter.must[index].clone())
                        .collect(),
                    ..Filter::default()
                };
                // Every match satisfies each `must` clause, so the walk of
                // the uncounted text ones is a superset of the matches.
                match indexed_filter_ids_be(storage, r, &uncounted, text_skip_max)? {
                    Some((ids, walk_exact)) if ids.len() <= dense_exact_index_candidate_max() => {
                        if filter_exact {
                            // Rechecked against the full filter: exact.
                            candidates =
                                filter_candidate_ids_be(storage, r, Some(&ids), filter, true)
                                    .map(|ids| ids.into_iter().collect());
                            exact = candidates.is_some();
                        } else {
                            // The set itself must not change (planning reads
                            // it); it is exact when every walked id also
                            // satisfies the remaining clauses.
                            exact = walk_exact
                                && exact_rest_ids(storage, r, filter, &walked, ids.len(), total)?
                                    .is_some_and(|rest| ids.is_subset(&rest));
                            candidates = Some(ids);
                        }
                    }
                    Some((ids, walk_exact)) => {
                        prefilter = keep_as_prefilter(ids, total);
                        if let Some(ids) = prefilter.as_ref().filter(|_| walk_exact) {
                            if let Some(rest) =
                                exact_rest_ids(storage, r, filter, &walked, ids.len(), total)?
                            {
                                // Planning ignores the prefilter, so it may narrow.
                                prefilter = Some(ids.intersection(&rest).copied().collect());
                                exact = true;
                            }
                        }
                    }
                    None => {}
                }
            }
            return Ok(ProbedCandidates {
                probe_skipped: candidates.is_none(),
                exact,
                candidates,
                probe_estimate,
                prefilter,
            });
        }
    }

    let (candidates, prefilter, exact) =
        match indexed_filter_ids_be(storage, r, filter, text_skip_max)? {
            Some((ids, exact)) if drop_broad && too_broad(ids.len()) => {
                let prefilter = keep_as_prefilter(ids, total)
                    .filter(|_| set_is_text_walk_superset(storage, filter));
                let exact = exact && prefilter.is_some();
                (None, prefilter, exact)
            }
            Some((ids, exact)) => (Some(ids), None, exact),
            None => (None, None, false),
        };
    Ok(ProbedCandidates {
        candidates,
        probe_estimate,
        probe_skipped: false,
        prefilter,
        exact,
    })
}
