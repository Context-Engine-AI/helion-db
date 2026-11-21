//! Lazy filter evaluation for sparse top-k selection.
//!
//! A sparse search filter is a node read per call on the Qdrant path, yet only
//! the docs a top-k cut keeps can affect the result. Scores are accumulated
//! unfiltered (a passing doc's score is the same sum either way), then docs are
//! walked best-first and the filter runs only until the cut is decided. Every
//! caller then applies its usual selection (heap, maxscore stop, rescore pool)
//! to the passing docs the walk returned, which always include every passing
//! doc that selection could pick, so results are identical by construction.
//!
//! A filter that reads a row per doc can check several docs in one read
//! ([`DocFilter::passes_many`]). The walk then has the docs it is about to
//! reach checked together, so the waits of cold rows overlap instead of
//! adding up. Verdicts are the same either way, and so are results.

use super::ScoredDoc;
use crate::helix_engine::storage_core::backend_lsm::lsm_read_cancelled;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};

/// The filter of a sparse search. Any `Fn(u128) -> bool` is one.
pub trait DocFilter {
    /// Whether doc `id` passes.
    fn passes(&self, id: u128) -> bool;

    /// Most docs worth one [`Self::passes_many`] call; 1 (the default) when
    /// checking docs together saves nothing.
    fn batch(&self) -> usize {
        1
    }

    /// [`Self::passes`] for each of `ids`, in order.
    fn passes_many(&self, ids: &[u128]) -> Vec<bool> {
        ids.iter().map(|&id| self.passes(id)).collect()
    }

    /// Why a check could not be made, if one could not: the filter said no to
    /// a doc it knows nothing about, and the search fails with this rather
    /// than return docs chosen on such an answer.
    fn failure(&self) -> Option<String> {
        None
    }
}

impl<F: Fn(u128) -> bool> DocFilter for F {
    fn passes(&self, id: u128) -> bool {
        self(id)
    }
}

#[cfg(test)]
thread_local! {
    static LAZY_FILTER_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Test-only: run `f` with `sparse_lazy_filter_enabled` forced on this thread.
#[cfg(test)]
pub(crate) fn with_sparse_lazy_filter_for_test<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    let previous = LAZY_FILTER_OVERRIDE.with(|cell| cell.replace(Some(enabled)));
    let out = f();
    LAZY_FILTER_OVERRIDE.with(|cell| cell.set(previous));
    out
}

/// `HELIX_SPARSE_LAZY_FILTER=0` restores evaluating the filter for every
/// scored doc (at first touch in WAND, before the heap in a full scan).
pub(crate) fn sparse_lazy_filter_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = LAZY_FILTER_OVERRIDE.with(std::cell::Cell::get) {
        return enabled;
    }
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HELIX_SPARSE_LAZY_FILTER")
            .ok()
            .map(|v| !matches!(v.trim(), "0" | "false" | "FALSE" | "no"))
            .unwrap_or(true)
    })
}

/// A filter evaluated at most once per doc, and only for docs a best-first
/// walk reaches (or, with a filter that checks docs together, is about to).
pub(super) struct LazyFilter<'a, F> {
    filter: &'a F,
    verdicts: HashMap<u128, bool>,
    /// How many of `verdicts` passed.
    passed: usize,
}

impl<'a, F: DocFilter> LazyFilter<'a, F> {
    pub(super) fn new(filter: &'a F) -> Self {
        Self {
            filter,
            verdicts: HashMap::new(),
            passed: 0,
        }
    }

    fn passes(&mut self, id: u128) -> bool {
        if let Some(&verdict) = self.verdicts.get(&id) {
            return verdict;
        }
        let verdict = self.filter.passes(id);
        self.record(id, verdict);
        verdict
    }

    fn record(&mut self, id: u128, verdict: bool) {
        self.verdicts.insert(id, verdict);
        self.passed += usize::from(verdict);
    }

    /// Docs the filter rejected so far.
    pub(super) fn rejected(&self) -> usize {
        self.verdicts.len() - self.passed
    }

    /// Docs the filter was asked about so far.
    pub(super) fn checked(&self) -> usize {
        self.verdicts.len()
    }

    /// Checks `id` together with the unchecked docs that follow it in
    /// `best_first`; every doc it passes over moves to `ahead`, in order. It
    /// looks as far as the walk is expected to go for its `wanted` docs: a doc
    /// known to pass counts as one of them, an unchecked doc as the share of
    /// checked docs that passed. While every doc has passed, those are the
    /// docs a one-at-a-time walk checks; the more were rejected the further it
    /// looks, `batch` docs a call at most.
    fn check_ahead(
        &mut self,
        id: u128,
        wanted: usize,
        batch: usize,
        best_first: &mut BinaryHeap<Reverse<ScoredDoc>>,
        ahead: &mut VecDeque<ScoredDoc>,
    ) {
        let (per_pass, per_unchecked) = (self.verdicts.len() + 1, self.passed + 1);
        let goal = wanted.saturating_mul(per_pass);
        let mut expected = per_unchecked;
        let mut ids = vec![id];
        while expected < goal && ids.len() < batch {
            let Some(Reverse(next)) = best_first.pop() else {
                break;
            };
            match self.verdicts.get(&next.id) {
                Some(true) => expected += per_pass,
                Some(false) => {}
                None => {
                    expected += per_unchecked;
                    ids.push(next.id);
                }
            }
            ahead.push_back(next);
        }
        for (id, verdict) in ids.iter().zip(self.filter.passes_many(&ids)) {
            self.record(*id, verdict);
        }
    }

    /// Passing docs of `scores` walked best-first (score descending, id
    /// ascending: the order every top-k cut here keeps), until `want(taken,
    /// next)` is 0. Before that it is how many more passing docs the walk
    /// takes at least, if every doc from `next` on passes. A NaN or -0.0 score
    /// has no such order that agrees with the cuts' `>`/`==` comparisons, so
    /// then every doc is evaluated and all passing docs are returned.
    fn take(
        &mut self,
        scores: &HashMap<u128, f64>,
        mut want: impl FnMut(&[ScoredDoc], &ScoredDoc) -> usize,
    ) -> HashMap<u128, f64> {
        let ordered = scores
            .iter()
            .all(|(_, score)| !score.is_nan() && !(*score == 0.0 && score.is_sign_negative()));
        if !ordered {
            return scores
                .iter()
                .filter(|(id, _)| self.passes(**id))
                .map(|(&id, &score)| (id, score))
                .collect();
        }
        let batch = self.filter.batch();
        let mut best_first: BinaryHeap<Reverse<ScoredDoc>> = scores
            .iter()
            .map(|(&id, &score)| Reverse(ScoredDoc { id, score }))
            .collect();
        // Docs taken off `best_first` early, for `check_ahead`.
        let mut ahead: VecDeque<ScoredDoc> = VecDeque::new();
        let mut taken: Vec<ScoredDoc> = Vec::new();
        while let Some(doc) = ahead
            .pop_front()
            .or_else(|| best_first.pop().map(|Reverse(doc)| doc))
        {
            let wanted = want(&taken, &doc);
            if wanted == 0 {
                break;
            }
            if batch > 1 && !self.verdicts.contains_key(&doc.id) {
                self.check_ahead(doc.id, wanted, batch, &mut best_first, &mut ahead);
            }
            if self.passes(doc.id) {
                taken.push(doc);
            } else if lsm_read_cancelled() || self.filter.failure().is_some() {
                // The caller has gone, or a check could not be made: the
                // search fails whatever the walk finds from here on.
                break;
            }
        }
        taken.into_iter().map(|doc| (doc.id, doc.score)).collect()
    }

    /// The best `n` passing docs (every passing doc when fewer pass).
    pub(super) fn top(&mut self, scores: &HashMap<u128, f64>, n: usize) -> HashMap<u128, f64> {
        self.take(scores, |taken, _| n.saturating_sub(taken.len()))
    }

    /// The passing docs `wand_rescore_candidates(passing, limit, slack)` can
    /// return: the best `margin`, then near ties down to the `limit`-th score
    /// minus `slack`, `cap` docs at most.
    pub(super) fn rescore_pool(
        &mut self,
        scores: &HashMap<u128, f64>,
        limit: usize,
        margin: usize,
        cap: usize,
        slack: f64,
    ) -> HashMap<u128, f64> {
        self.take(scores, |taken, next| {
            if taken.len() < margin {
                margin - taken.len()
            } else {
                usize::from(taken.len() < cap && next.score >= taken[limit - 1].score - slack)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::{SparseModifier, SparseVector, SparseVectorConfig, SparseVectorCore};
    use super::{with_sparse_lazy_filter_for_test, DocFilter};
    use crate::helix_engine::storage_core::backend::StorageBackend;
    use crate::helix_engine::storage_core::backend_any::AnyBackend;
    use crate::helix_engine::storage_core::backend_lsm::{
        allow_lsm_blocking_cancellable, LsmBackend, LsmReadCancellation,
    };
    use rand::{Rng, SeedableRng};
    use std::cell::Cell;
    use std::sync::Arc;

    fn bits(results: &[(u128, f64)]) -> Vec<(u128, u64)> {
        results
            .iter()
            .map(|(id, score)| (*id, score.to_bits()))
            .collect()
    }

    /// Skewed term draws give long and short posting lists; quantized values
    /// give score ties, so tie-breaking by id is exercised too.
    fn docs(rng: &mut rand::rngs::StdRng) -> Vec<(u128, SparseVector)> {
        (1..=3_000u128)
            .map(|id| {
                let mut indices: Vec<u32> = (0..rng.random_range(2..12))
                    .map(|_| {
                        let draw: f64 = rng.random();
                        (draw * draw * 400.0) as u32
                    })
                    .collect();
                indices.sort_unstable();
                indices.dedup();
                let values = indices
                    .iter()
                    .map(|_| f32::from(rng.random_range(1u8..=8)) / 4.0)
                    .collect();
                (id, SparseVector { indices, values })
            })
            .collect()
    }

    /// Filters by id hash at several pass rates, plus one that rejects the
    /// best-scoring docs first so the lazy walk must go deep.
    fn passes(kind: usize, id: u128, top: &[u128]) -> bool {
        let hash = (id.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 7) % 100;
        match kind {
            0 => false,
            1 => true,
            2 => hash < 5,
            3 => hash < 30,
            4 => hash < 80,
            _ => !top.contains(&id),
        }
    }

    /// WAND, full scan (also the WAND fallback for negative weights) and
    /// exact candidate scoring return the same ids and score bits with lazy
    /// and eager filtering, and the lazy path never calls the filter more.
    #[test]
    fn lazy_filter_matches_eager_results_with_fewer_filter_calls() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(41);
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_in_memory("sparse-lazy-filter").unwrap(),
        ));
        let config = |wand_enabled| SparseVectorConfig {
            full_scan_threshold: 1,
            modifier: SparseModifier::Idf,
            wand_enabled,
        };
        let wand =
            SparseVectorCore::new_lsm("lazy_wand", config(true), Arc::clone(&backend)).unwrap();
        let full =
            SparseVectorCore::new_lsm("lazy_full", config(false), Arc::clone(&backend)).unwrap();
        let docs = docs(&mut rng);
        let items: Vec<(u128, &SparseVector)> = docs.iter().map(|(id, sv)| (*id, sv)).collect();
        let mut w = backend.begin_write().unwrap();
        wand.upsert_batch_be(&mut w, &items).unwrap();
        full.upsert_batch_be(&mut w, &items).unwrap();
        backend.commit(w).unwrap();
        let r = backend.begin_read().unwrap();

        let (mut eager_calls, mut lazy_calls) = (0usize, 0usize);
        for query_index in 0..24 {
            let mut indices: Vec<u32> = (0..rng.random_range(3..9))
                .map(|_| rng.random_range(0..120))
                .collect();
            indices.sort_unstable();
            indices.dedup();
            let values = indices
                .iter()
                .map(|_| {
                    if query_index % 8 == 7 && rng.random_bool(0.3) {
                        -0.5
                    } else {
                        rng.random_range(0.1f32..2.0)
                    }
                })
                .collect();
            let query = SparseVector { indices, values };
            let top: Vec<u128> = full
                .search_be::<fn(u128) -> bool>(&r, &query, 25, None)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let candidates: Vec<u128> = (1..=3_000u128).filter(|id| id % 3 != 0).collect();
            for kind in 0..6 {
                for limit in [1usize, 10, 40, 500] {
                    let run = |lazy: bool, calls: &mut usize| {
                        let counter = Cell::new(0usize);
                        let filter = |id: u128| {
                            counter.set(counter.get() + 1);
                            passes(kind, id, &top)
                        };
                        let out = with_sparse_lazy_filter_for_test(lazy, || {
                            let by_wand = wand.search_be(&r, &query, limit, Some(&filter)).unwrap();
                            let by_scan = full.search_be(&r, &query, limit, Some(&filter)).unwrap();
                            let by_ids = full
                                .search_candidate_ids_exact_filtered_be(
                                    &r,
                                    &query,
                                    limit,
                                    candidates.iter().copied(),
                                    Some(&filter),
                                )
                                .unwrap();
                            (bits(&by_wand), bits(&by_scan), bits(&by_ids))
                        });
                        *calls += counter.get();
                        out
                    };
                    let eager = run(false, &mut eager_calls);
                    let lazy = run(true, &mut lazy_calls);
                    assert_eq!(
                        lazy, eager,
                        "query {query_index} filter {kind} limit {limit}"
                    );
                    // Eager exact scoring of pre-filtered candidates agrees too.
                    let prefiltered: Vec<u128> = candidates
                        .iter()
                        .copied()
                        .filter(|id| passes(kind, *id, &top))
                        .collect();
                    let exact = full
                        .search_candidate_ids_exact_be(&r, &query, limit, prefiltered)
                        .unwrap();
                    assert_eq!(bits(&exact), eager.2, "query {query_index} filter {kind}");
                }
            }
        }
        println!("sparse filter calls: eager {eager_calls}, lazy {lazy_calls}");
        assert!(
            lazy_calls * 3 < eager_calls * 2,
            "{lazy_calls} vs {eager_calls}"
        );
    }

    /// The 3,000 docs indexed twice, for WAND and for a full scan.
    fn cores(
        name: &str,
        rng: &mut rand::rngs::StdRng,
    ) -> (Arc<AnyBackend>, SparseVectorCore, SparseVectorCore) {
        let backend = Arc::new(AnyBackend::Lsm(LsmBackend::open_in_memory(name).unwrap()));
        let config = |wand_enabled| SparseVectorConfig {
            full_scan_threshold: 1,
            modifier: SparseModifier::Idf,
            wand_enabled,
        };
        let wand = SparseVectorCore::new_lsm("wand", config(true), Arc::clone(&backend)).unwrap();
        let full = SparseVectorCore::new_lsm("full", config(false), Arc::clone(&backend)).unwrap();
        let docs = docs(rng);
        let items: Vec<(u128, &SparseVector)> = docs.iter().map(|(id, sv)| (*id, sv)).collect();
        let mut w = backend.begin_write().unwrap();
        wand.upsert_batch_be(&mut w, &items).unwrap();
        full.upsert_batch_be(&mut w, &items).unwrap();
        backend.commit(w).unwrap();
        (backend, wand, full)
    }

    fn query(rng: &mut rand::rngs::StdRng) -> SparseVector {
        let mut indices: Vec<u32> = (0..rng.random_range(3..9))
            .map(|_| rng.random_range(0..120))
            .collect();
        indices.sort_unstable();
        indices.dedup();
        let values = indices
            .iter()
            .map(|_| rng.random_range(0.1f32..2.0))
            .collect();
        SparseVector { indices, values }
    }

    /// `passes(kind, ..)` as a filter that checks up to `batch` docs in one
    /// call, counting the docs it is asked about.
    struct Together<'a> {
        kind: usize,
        top: &'a [u128],
        batch: usize,
        checked: Cell<usize>,
    }

    impl DocFilter for Together<'_> {
        fn passes(&self, id: u128) -> bool {
            self.checked.set(self.checked.get() + 1);
            passes(self.kind, id, self.top)
        }

        fn batch(&self) -> usize {
            self.batch
        }

        fn passes_many(&self, ids: &[u128]) -> Vec<bool> {
            assert!(
                !ids.is_empty() && ids.len() <= self.batch,
                "{} docs in a call, batch {}",
                ids.len(),
                self.batch
            );
            self.checked.set(self.checked.get() + ids.len());
            ids.iter()
                .map(|&id| passes(self.kind, id, self.top))
                .collect()
        }
    }

    /// A filter that checks docs together gives WAND, a full scan and exact
    /// candidate scoring the results of the same filter asked doc by doc. It
    /// is asked about every doc the doc-by-doc walk reaches, and while every
    /// doc passes about no other.
    #[test]
    fn checking_docs_together_matches_one_at_a_time() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(43);
        let (backend, wand, full) = cores("sparse-filter-together", &mut rng);
        let r = backend.begin_read().unwrap();
        let candidates: Vec<u128> = (1..=3_000u128).filter(|id| id % 3 != 0).collect();
        // Docs checked one at a time, and with 7 and 96 docs a call.
        let mut checked = [0usize; 3];
        for query_index in 0..12 {
            let query = query(&mut rng);
            let top: Vec<u128> = full
                .search_be::<fn(u128) -> bool>(&r, &query, 25, None)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            for kind in 0..6 {
                for limit in [1usize, 10, 40, 500] {
                    let run = |batch: usize| {
                        let filter = Together {
                            kind,
                            top: &top,
                            batch,
                            checked: Cell::new(0),
                        };
                        let out = with_sparse_lazy_filter_for_test(true, || {
                            let by_wand = wand
                                .search_be_with(&r, &query, limit, Some(&filter), true)
                                .unwrap();
                            let by_scan = full
                                .search_be_with(&r, &query, limit, Some(&filter), true)
                                .unwrap();
                            let by_ids = full
                                .search_candidate_ids_exact_filtered_be(
                                    &r,
                                    &query,
                                    limit,
                                    candidates.iter().copied(),
                                    Some(&filter),
                                )
                                .unwrap();
                            (bits(&by_wand), bits(&by_scan), bits(&by_ids))
                        });
                        (out, filter.checked.get())
                    };
                    let (alone, alone_checked) = run(1);
                    checked[0] += alone_checked;
                    for (slot, batch) in [(1usize, 7usize), (2, 96)] {
                        let case = format!(
                            "query {query_index} filter {kind} limit {limit} batch {batch}"
                        );
                        let (together, together_checked) = run(batch);
                        assert_eq!(together, alone, "{case}");
                        assert!(
                            together_checked >= alone_checked,
                            "{case}: {together_checked} docs checked, {alone_checked} one at a time"
                        );
                        if kind == 1 {
                            assert_eq!(together_checked, alone_checked, "{case}");
                        }
                        checked[slot] += together_checked;
                    }
                }
            }
        }
        println!(
            "sparse filter docs checked: one at a time {}, 7 a call {}, 96 a call {}",
            checked[0], checked[1], checked[2]
        );
    }

    /// The first `limit` rows of a search asked for four times as many are
    /// the rows of a search asked for `limit`, so a caller that wants a
    /// window need not ask for more and cut.
    #[test]
    fn the_best_rows_do_not_depend_on_how_many_more_are_asked_for() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(53);
        let (backend, wand, full) = cores("sparse-filter-window", &mut rng);
        let r = backend.begin_read().unwrap();
        let candidates: Vec<u128> = (1..=3_000u128).filter(|id| id % 3 != 0).collect();
        for query_index in 0..24 {
            let query = query(&mut rng);
            let top: Vec<u128> = full
                .search_be::<fn(u128) -> bool>(&r, &query, 25, None)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            for kind in 0..6 {
                let filter = |id: u128| passes(kind, id, &top);
                for limit in [1usize, 6, 24, 96] {
                    for lazy in [true, false] {
                        let rows = |asked: usize| {
                            with_sparse_lazy_filter_for_test(lazy, || {
                                let by_wand = wand
                                    .search_be_with(&r, &query, asked, Some(&filter), lazy)
                                    .unwrap();
                                let by_scan = full
                                    .search_be_with(&r, &query, asked, Some(&filter), lazy)
                                    .unwrap();
                                let by_ids = full
                                    .search_candidate_ids_exact_filtered_be(
                                        &r,
                                        &query,
                                        asked,
                                        candidates.iter().copied(),
                                        Some(&filter),
                                    )
                                    .unwrap();
                                [bits(&by_wand), bits(&by_scan), bits(&by_ids)]
                            })
                        };
                        let mut cut = rows(limit * 4);
                        for page in cut.iter_mut() {
                            page.truncate(limit);
                        }
                        assert_eq!(
                            cut,
                            rows(limit),
                            "query {query_index} filter {kind} limit {limit} lazy {lazy}"
                        );
                    }
                }
            }
        }
    }

    /// Once the caller has gone a filter's reads fail, and it says no to
    /// every doc. The walk stops asking, and the search fails instead of
    /// returning what such answers leave.
    #[test]
    fn a_cancelled_request_ends_the_filter_walk_and_fails_the_search() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(47);
        let (backend, wand, full) = cores("sparse-filter-cancelled", &mut rng);
        let r = backend.begin_read().unwrap();
        let query = query(&mut rng);
        for (core, name) in [(&wand, "wand"), (&full, "full scan")] {
            let cancellation = LsmReadCancellation::new();
            let calls = Cell::new(0usize);
            // The caller leaves while the fifth doc is checked.
            let filter = |_id: u128| {
                calls.set(calls.get() + 1);
                if calls.get() == 5 {
                    cancellation.cancel();
                }
                false
            };
            let outcome = allow_lsm_blocking_cancellable(cancellation.clone(), || {
                core.search_be_with(&r, &query, 10, Some(&filter), true)
            });
            let error = match outcome {
                Ok(rows) => panic!("{name}: a cancelled search returned {} rows", rows.len()),
                Err(error) => error.to_string(),
            };
            assert!(
                error.contains("LSM read request cancelled"),
                "{name}: {error}"
            );
            assert!(
                (5..40).contains(&calls.get()),
                "{name}: the filter was asked about {} docs",
                calls.get()
            );
        }
    }

    /// Every doc passes, but the check of `lost` cannot be made.
    struct Unread {
        lost: u128,
        failed: Cell<bool>,
    }

    impl DocFilter for Unread {
        fn passes(&self, id: u128) -> bool {
            if id == self.lost {
                self.failed.set(true);
            }
            id != self.lost
        }

        fn failure(&self) -> Option<String> {
            self.failed
                .get()
                .then(|| "io error: db is closed".to_string())
        }
    }

    /// A filter that could not check a doc says no and reports why. On every
    /// path the search then fails with that reason instead of returning the
    /// docs such an answer leaves.
    #[test]
    fn a_filter_that_could_not_check_a_doc_fails_the_search() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(59);
        let (backend, wand, full) = cores("sparse-filter-unread", &mut rng);
        let r = backend.begin_read().unwrap();
        let query = query(&mut rng);
        let best = full
            .search_be::<fn(u128) -> bool>(&r, &query, 1, None)
            .unwrap()[0]
            .0;
        for lazy in [true, false] {
            // Every search checks its best doc; 0 is no doc's id.
            for lost in [best, 0] {
                let unread = || Unread {
                    lost,
                    failed: Cell::new(false),
                };
                let (by_wand, by_scan, by_ids) = (unread(), unread(), unread());
                let outcomes = with_sparse_lazy_filter_for_test(lazy, || {
                    [
                        wand.search_be_with(&r, &query, 10, Some(&by_wand), lazy),
                        full.search_be_with(&r, &query, 10, Some(&by_scan), lazy),
                        full.search_candidate_ids_exact_filtered_be(
                            &r,
                            &query,
                            10,
                            1..=3_000u128,
                            Some(&by_ids),
                        ),
                    ]
                });
                for (outcome, path) in outcomes.into_iter().zip(["wand", "full scan", "exact"]) {
                    let outcome = outcome
                        .map(|rows| rows.len())
                        .map_err(|error| error.to_string());
                    let case = format!("{path}, lazy {lazy}, lost {lost}: {outcome:?}");
                    match outcome {
                        Ok(rows) => assert!(lost == 0 && rows == 10, "{case}"),
                        Err(error) => {
                            assert!(lost != 0 && error.contains("db is closed"), "{case}")
                        }
                    }
                }
            }
        }
    }
}
