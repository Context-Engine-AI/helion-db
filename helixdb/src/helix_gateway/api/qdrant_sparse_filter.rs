//! The filter of a sparse search (`/points/search`, `/points/query` and a
//! hybrid sparse channel on an LSM store): is a doc in the planner's id set
//! and, where the set does not settle it, does its payload match.
//!
//! A payload check is a node read. The search asks for the docs it is about
//! to reach together ([`DocFilter::passes_many`]) and their nodes are read in
//! one batch, so on a cold reader the waits overlap instead of adding up.

use super::*;
use crate::helix_engine::vector_core::sparse::DocFilter;

#[cfg(test)]
thread_local! {
    static SPARSE_FILTER_READS_OVERRIDE: std::cell::Cell<Option<(usize, usize)>> =
        const { std::cell::Cell::new(None) };
}

/// Test-only: run `f` with `sparse_filter_overfetch` and
/// `sparse_filter_read_batch` forced on this thread.
#[cfg(test)]
pub(super) fn with_sparse_filter_reads_for_test<R>(
    overfetch: usize,
    read_batch: usize,
    f: impl FnOnce() -> R,
) -> R {
    let previous =
        SPARSE_FILTER_READS_OVERRIDE.with(|cell| cell.replace(Some((overfetch, read_batch))));
    let out = f();
    SPARSE_FILTER_READS_OVERRIDE.with(|cell| cell.set(previous));
    out
}

/// How many result windows a filtered sparse search is asked for. The filter
/// runs inside the search, which returns the best passing docs, so one window
/// is enough. `HELIX_SPARSE_FILTER_OVERFETCH=4` asks for four and drops
/// three, as before: the search then checks payloads until four times as
/// many docs have passed.
///
/// The first window of a search asked for four is the search asked for one
/// in the exact-candidate and full-scan paths. WAND rescores a pool sized by
/// its limit, so there it can differ in two cases: more than 256 passing docs
/// within rounding of the last score of the window (the near-tie cap is
/// limit + 256), and postings without a forward row among the pool's best
/// (the pool keeps min(2 * limit, limit + 32) docs before near ties, so it
/// has fewer spares to lose).
fn sparse_filter_overfetch() -> usize {
    #[cfg(test)]
    if let Some((overfetch, _)) = SPARSE_FILTER_READS_OVERRIDE.with(std::cell::Cell::get) {
        return overfetch;
    }
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| env_usize_or("HELIX_SPARSE_FILTER_OVERFETCH", 1).max(1))
}

/// Docs to ask a sparse search for when the reply needs `window` of them.
pub(super) fn sparse_search_limit(window: usize, filtered: bool) -> usize {
    if filtered {
        window.saturating_mul(sparse_filter_overfetch())
    } else {
        window
    }
}

/// Most nodes a sparse filter reads in one batch.
/// `HELIX_SPARSE_FILTER_READ_BATCH=1` reads one node per doc again.
fn sparse_filter_read_batch() -> usize {
    #[cfg(test)]
    if let Some((_, read_batch)) = SPARSE_FILTER_READS_OVERRIDE.with(std::cell::Cell::get) {
        return read_batch;
    }
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| env_usize_or("HELIX_SPARSE_FILTER_READ_BATCH", 96))
}

pub(super) struct SparseDocFilter<'a, 'r> {
    pub(super) storage: &'a HelixGraphStorage,
    pub(super) r: &'a AnyRead<'r>,
    pub(super) filter: &'a Filter,
    /// The planner's id set (its candidates, else its prefilter): a doc
    /// outside it does not match.
    pub(super) candidates: Option<&'a HashSet<u128>>,
    /// Whether a doc of the set still needs its payload checked.
    pub(super) recheck: bool,
    /// The first payload read that failed for a reason other than its caller
    /// having left; see [`DocFilter::failure`].
    pub(super) failure: OnceLock<String>,
}

impl SparseDocFilter<'_, '_> {
    fn in_candidates(&self, id: u128) -> bool {
        self.candidates
            .is_none_or(|candidates| candidates.contains(&id))
    }

    /// Whether each of `ids` has a node whose payload matches. No node, or one
    /// that does not decode, is no match. A read that fails says nothing
    /// about its docs: it is "no" for each, and it is kept unless the caller
    /// has gone (then the search fails as cancelled).
    fn payloads_match(&self, ids: &[u128]) -> Vec<bool> {
        let matches =
            |id: &u128, node: &Node| self.filter.matches_point(Some(*id), &node.properties);
        match self.storage.get_nodes_many_be(self.r, ids) {
            Ok(Some(nodes)) => ids
                .iter()
                .zip(nodes)
                .map(|(id, node)| node.as_ref().is_some_and(|node| matches(id, node)))
                .collect(),
            // No batched read on this backend: one read per doc.
            Ok(None) => ids
                .iter()
                .map(|id| {
                    self.storage
                        .get_node_be(self.r, id)
                        .is_ok_and(|node| matches(id, &node))
                })
                .collect(),
            Err(error) => {
                if !error.is_lsm_read_cancelled() {
                    let _ = self.failure.set(error.to_string());
                }
                vec![false; ids.len()]
            }
        }
    }
}

impl DocFilter for SparseDocFilter<'_, '_> {
    fn passes(&self, id: u128) -> bool {
        self.in_candidates(id)
            && (!self.recheck || self.payloads_match(&[id]).first() == Some(&true))
    }

    fn batch(&self) -> usize {
        if self.recheck {
            sparse_filter_read_batch()
        } else {
            1
        }
    }

    fn passes_many(&self, ids: &[u128]) -> Vec<bool> {
        if !self.recheck {
            return ids.iter().map(|&id| self.in_candidates(id)).collect();
        }
        // As in `passes`, only a doc of the set is read.
        let read: Vec<u128> = ids
            .iter()
            .copied()
            .filter(|&id| self.in_candidates(id))
            .collect();
        if read.is_empty() {
            return vec![false; ids.len()];
        }
        let mut verdicts = self.payloads_match(&read).into_iter();
        ids.iter()
            .map(|&id| self.in_candidates(id) && verdicts.next().unwrap_or(false))
            .collect()
    }

    fn failure(&self) -> Option<String> {
        self.failure.get().cloned()
    }
}
