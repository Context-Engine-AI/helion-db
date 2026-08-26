//! Lightweight HNSW search-frontier types and heap helpers.
//!
//! HNSW navigation walks the graph one node at a time. The classic mistake
//! is to materialize a full vector (id + raw f32 payload + properties +
//! distance) for every node visited. On 768-dim vectors that's ~3KB of
//! payload pulled through L2 per neighbor visit — far more than the heap
//! actually needs to compare.
//!
//! `Candidate` is the *minimum* shape a search heap needs: 16-byte id plus
//! 4-byte distance. Loading payloads is deferred until top-k materialization,
//! at which point `VectorFilter::to_vec_with_filter` decodes only the
//! candidates that survive the filter.
//!
//! Phase A: types + traits land here so callers and tests can start using
//! them. The `to_vec_with_filter` body is wired in Phase B alongside the
//! `VectorWithoutData` split.

use bumpalo::Bump;
use core::cmp::Ordering;

use super::arena_heap::ArenaHeap;

/// Minimal heap entry for HNSW navigation.
///
/// Reverse `Ord` so the *smallest* distance compares as *greatest*: a
/// max-heap of `Candidate` thus surfaces the closest unvisited node first,
/// matching the priority needed by `search_level` on the candidate frontier.
/// Use the natural (non-reversed) order via a wrapper if you need a
/// max-by-distance heap (e.g. the ef-bounded result heap).
#[derive(Copy, Clone, Debug)]
pub struct Candidate {
    pub id: u128,
    pub distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.distance.to_bits() == other.distance.to_bits()
    }
}
impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse: smaller distance => "greater" so a max-heap pops the
        // closest neighbor first. NaN sorts as Equal (defensive — distance
        // kernels are expected to never produce NaN).
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Wrapper that flips ordering: a heap of `MaxByDistance(Candidate)` pops
/// the *farthest* element first. Used for the ef-bounded result frontier
/// in HNSW where we evict the worst on overflow.
#[derive(Copy, Clone, Debug)]
pub struct MaxByDistance(pub Candidate);

impl PartialEq for MaxByDistance {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for MaxByDistance {}

impl PartialOrd for MaxByDistance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MaxByDistance {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .distance
            .partial_cmp(&other.0.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Heap helpers used by HNSW search bookkeeping.
///
/// `take_inord` is for HNSW's "shrink to M neighbors" step where we pull
/// the top-k from a candidate set in order. `get_max` is a non-destructive
/// peek at the worst element of the result heap (used to short-circuit
/// when the next candidate cannot improve top-k).
pub trait HeapOps<'arena, T>
where
    T: Ord,
{
    /// Pop up to `k` greatest elements into a fresh heap, in pop-order.
    ///
    /// The returned heap, when iterated by `pop`, yields the same ordering
    /// as repeated `pop` on `self`.
    fn take_inord(&mut self, k: usize) -> ArenaHeap<'arena, T>;

    /// Peek the maximum element by reference.
    ///
    /// O(n) because the heap stores T by value and items in the slice past
    /// position 0 may be larger by the heap's reverse-ordering convention
    /// (e.g. `Candidate`). Cheap in practice for the small ef-bounded
    /// frontier sizes HNSW uses (typically <= 512).
    fn get_max(&self) -> Option<&T>;
}

impl<'arena, T> HeapOps<'arena, T> for ArenaHeap<'arena, T>
where
    T: Ord,
{
    #[inline]
    fn take_inord(&mut self, k: usize) -> ArenaHeap<'arena, T> {
        let cap = k.min(self.len());
        let mut out = ArenaHeap::with_capacity(self.arena(), cap);
        for _ in 0..cap {
            match self.pop() {
                Some(v) => out.push(v),
                None => break,
            }
        }
        out
    }

    #[inline]
    fn get_max(&self) -> Option<&T> {
        self.iter().max()
    }
}

/// Filter and materialize top-k from a HNSW candidate heap.
///
/// Phase A defines the trait surface so call sites in `vector_core.rs` can
/// be written ahead of the `VectorWithoutData` split. The default
/// implementation simply pops without filtering — it returns the bare
/// candidates and lets callers do their own decode. Phase B will provide a
/// specialized impl that:
///   1. Pops the next candidate.
///   2. Reads its property bytes from the properties DB and decodes into
///      `VectorWithoutData` (no payload load).
///   3. Skips if `deleted` or label mismatch or any closure filter rejects.
///   4. Otherwise expands into a full `HVector` and pushes onto the result.
///
/// Const generic `SHOULD_CHECK_DELETED` lets callers skip the deleted check
/// on paths where deletion is impossible (e.g. fresh build), saving one
/// branch per candidate.
pub trait VectorFilter<'arena, T> {
    fn to_vec_with_filter<F, const SHOULD_CHECK_DELETED: bool>(
        self,
        k: usize,
        filter: Option<&[F]>,
        arena: &'arena Bump,
    ) -> bumpalo::collections::Vec<'arena, T>
    where
        F: Fn(&T) -> bool;
}

impl<'arena, T> VectorFilter<'arena, T> for ArenaHeap<'arena, T>
where
    T: Ord,
{
    #[inline]
    fn to_vec_with_filter<F, const SHOULD_CHECK_DELETED: bool>(
        mut self,
        k: usize,
        filter: Option<&[F]>,
        arena: &'arena Bump,
    ) -> bumpalo::collections::Vec<'arena, T>
    where
        F: Fn(&T) -> bool,
    {
        let mut out = bumpalo::collections::Vec::with_capacity_in(k, arena);
        while out.len() < k {
            match self.pop() {
                Some(item) => match filter {
                    Some(fs) if !fs.iter().all(|f| f(&item)) => continue,
                    _ => out.push(item),
                },
                None => break,
            }
        }
        out
    }
}

#[cfg(test)]
mod heap_utils_tests {
    use super::*;
    use bumpalo::Bump;

    #[test]
    fn candidate_orders_smallest_distance_as_greatest() {
        let a = Candidate {
            id: 1,
            distance: 0.5,
        };
        let b = Candidate {
            id: 2,
            distance: 1.0,
        };
        let c = Candidate {
            id: 3,
            distance: 0.2,
        };
        assert!(c > a);
        assert!(a > b);
        assert!(c > b);
    }

    #[test]
    fn candidate_min_heap_via_max_heap_pops_closest_first() {
        let arena = Bump::new();
        let mut heap = ArenaHeap::<Candidate>::new(&arena);
        heap.push(Candidate {
            id: 1,
            distance: 0.5,
        });
        heap.push(Candidate {
            id: 2,
            distance: 1.0,
        });
        heap.push(Candidate {
            id: 3,
            distance: 0.2,
        });
        assert_eq!(heap.pop().unwrap().id, 3);
        assert_eq!(heap.pop().unwrap().id, 1);
        assert_eq!(heap.pop().unwrap().id, 2);
    }

    #[test]
    fn max_by_distance_pops_farthest_first() {
        let arena = Bump::new();
        let mut heap = ArenaHeap::<MaxByDistance>::new(&arena);
        heap.push(MaxByDistance(Candidate {
            id: 1,
            distance: 0.5,
        }));
        heap.push(MaxByDistance(Candidate {
            id: 2,
            distance: 1.0,
        }));
        heap.push(MaxByDistance(Candidate {
            id: 3,
            distance: 0.2,
        }));
        assert_eq!(heap.pop().unwrap().0.id, 2);
        assert_eq!(heap.pop().unwrap().0.id, 1);
        assert_eq!(heap.pop().unwrap().0.id, 3);
    }

    #[test]
    fn take_inord_yields_descending() {
        let arena = Bump::new();
        let mut heap = ArenaHeap::<i32>::new(&arena);
        for v in [5, 1, 8, 3, 9, 2, 7] {
            heap.push(v);
        }
        let mut top3 = HeapOps::take_inord(&mut heap, 3);
        assert_eq!(top3.pop(), Some(9));
        assert_eq!(top3.pop(), Some(8));
        assert_eq!(top3.pop(), Some(7));
        assert_eq!(top3.pop(), None);
    }

    #[test]
    fn take_inord_caps_at_size() {
        let arena = Bump::new();
        let mut heap = ArenaHeap::<i32>::new(&arena);
        heap.push(1);
        heap.push(2);
        let mut top = HeapOps::take_inord(&mut heap, 10);
        assert_eq!(top.len(), 2);
        assert_eq!(top.pop(), Some(2));
        assert_eq!(top.pop(), Some(1));
    }

    #[test]
    fn get_max_returns_global_max_for_candidate() {
        // Candidate uses reverse Ord (closest=greatest). get_max here
        // therefore returns the *closest* candidate (the global max under
        // reverse ordering). HNSW will typically wrap candidates in
        // MaxByDistance for the result frontier when it wants the farthest.
        let arena = Bump::new();
        let mut heap = ArenaHeap::<Candidate>::new(&arena);
        heap.push(Candidate {
            id: 1,
            distance: 0.5,
        });
        heap.push(Candidate {
            id: 2,
            distance: 1.0,
        });
        heap.push(Candidate {
            id: 3,
            distance: 0.2,
        });
        let max = HeapOps::get_max(&heap).copied().unwrap();
        assert_eq!(max.id, 3); // closest is "greatest" under Candidate's reverse Ord
    }

    #[test]
    fn get_max_for_max_by_distance_returns_farthest() {
        let arena = Bump::new();
        let mut heap = ArenaHeap::<MaxByDistance>::new(&arena);
        heap.push(MaxByDistance(Candidate {
            id: 1,
            distance: 0.5,
        }));
        heap.push(MaxByDistance(Candidate {
            id: 2,
            distance: 1.0,
        }));
        heap.push(MaxByDistance(Candidate {
            id: 3,
            distance: 0.2,
        }));
        let max = HeapOps::get_max(&heap).copied().unwrap();
        assert_eq!(max.0.id, 2);
    }

    #[test]
    fn vector_filter_no_filter_passes_all_in_pop_order() {
        let arena = Bump::new();
        let mut heap = ArenaHeap::<i32>::new(&arena);
        for v in [5, 1, 8, 3, 9] {
            heap.push(v);
        }
        let out = VectorFilter::to_vec_with_filter::<fn(&i32) -> bool, true>(heap, 3, None, &arena);
        assert_eq!(out.as_slice(), &[9, 8, 5]);
    }

    #[test]
    fn vector_filter_skips_rejected_until_k_satisfied() {
        let arena = Bump::new();
        let mut heap = ArenaHeap::<i32>::new(&arena);
        for v in [5, 1, 8, 3, 9, 2, 7] {
            heap.push(v);
        }
        // Filter out odd numbers; result should be [8, 2] (next two evens
        // in descending order).
        let evens: [fn(&i32) -> bool; 1] = [|v: &i32| *v % 2 == 0];
        let out = VectorFilter::to_vec_with_filter::<_, true>(heap, 2, Some(&evens), &arena);
        assert_eq!(out.as_slice(), &[8, 2]);
    }

    #[test]
    fn vector_filter_returns_fewer_than_k_when_pool_exhausted() {
        let arena = Bump::new();
        let mut heap = ArenaHeap::<i32>::new(&arena);
        heap.push(2);
        heap.push(4);
        let evens: [fn(&i32) -> bool; 1] = [|v: &i32| *v % 2 == 0];
        let out = VectorFilter::to_vec_with_filter::<_, true>(heap, 10, Some(&evens), &arena);
        assert_eq!(out.len(), 2);
    }
}
