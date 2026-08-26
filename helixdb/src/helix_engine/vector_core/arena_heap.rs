//! Bumpalo-arena-backed max-heap for HNSW search/insert hot paths.
//!
//! Why a custom heap when `std::collections::BinaryHeap` exists?
//! - HNSW search builds many short-lived heaps per query (one per level,
//!   plus a candidate frontier). On the global allocator that's a steady
//!   stream of `malloc`/`free` calls during graph descent. With an arena,
//!   the whole heap lives in a bump-allocated region freed in one shot
//!   when the arena drops at request end.
//! - The `Hole` sift technique below moves N elements with N+1 memcpys
//!   instead of swaps (which cost ~2N memcpys). Same trick `std` uses
//!   internally.
//!
//! Behavior matches `std::collections::BinaryHeap`:
//! - Max-heap on `T: Ord`. Use a wrapper type with reverse `Ord` for
//!   min-heap semantics (see `heap_utils::Candidate`).
//! - `pop` returns greatest element. `peek` is non-destructive.
//!
//! Adapted from the std::collections::BinaryHeap design (MIT/Apache-2.0).
//! Sift loops use `unsafe` for `ptr::copy_nonoverlapping`; safety invariants
//! are documented at each callsite and exercised by `arena_heap_tests`.

use bumpalo::collections::Vec as BumpVec;
use bumpalo::Bump;
use core::mem::ManuallyDrop;
use core::ptr;

pub struct ArenaHeap<'arena, T> {
    pub(crate) arena: &'arena Bump,
    data: BumpVec<'arena, T>,
}

impl<'arena, T: Ord> ArenaHeap<'arena, T> {
    #[inline]
    pub fn new(arena: &'arena Bump) -> Self {
        Self {
            arena,
            data: BumpVec::new_in(arena),
        }
    }

    #[inline]
    pub fn with_capacity(arena: &'arena Bump, capacity: usize) -> Self {
        Self {
            arena,
            data: BumpVec::with_capacity_in(capacity, arena),
        }
    }

    /// Wrap an already-populated `BumpVec` and heapify it in O(n).
    #[inline]
    pub fn from_vec(arena: &'arena Bump, data: BumpVec<'arena, T>) -> Self {
        let mut heap = Self { arena, data };
        heap.heapify();
        heap
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    #[inline]
    pub fn arena(&self) -> &'arena Bump {
        self.arena
    }

    /// Returns a reference to the greatest element without removing it.
    #[inline]
    pub fn peek(&self) -> Option<&T> {
        self.data.first()
    }

    /// Inserts `item` and restores the heap invariant.
    pub fn push(&mut self, item: T) {
        let old_len = self.data.len();
        self.data.push(item);
        // SAFETY: we just pushed, so old_len < self.len().
        unsafe { self.sift_up(0, old_len) };
    }

    /// Removes and returns the greatest element.
    pub fn pop(&mut self) -> Option<T> {
        self.data.pop().map(|mut item| {
            if !self.data.is_empty() {
                core::mem::swap(&mut item, &mut self.data[0]);
                // SAFETY: data is non-empty, so 0 < len().
                unsafe { self.sift_down(0) };
            }
            item
        })
    }

    /// Iterate in arbitrary order (heap order is not sorted order).
    #[inline]
    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.data.iter()
    }

    /// Reserve space for at least `additional` more elements.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
    }

    /// Extend with another iterator. Each insert restores the invariant.
    /// O(n log n) — for bulk loads, prefer `from_vec`.
    pub fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for x in iter {
            self.push(x);
        }
    }

    /// Drain the heap into a new heap containing the top-`k` elements
    /// (popped in descending order, then re-heapified). Used by
    /// `HeapOps::take_inord` to keep result ordering intact across cuts.
    pub fn drain_top(mut self, k: usize) -> Self {
        let cap = k.min(self.len());
        let mut out = BumpVec::with_capacity_in(cap, self.arena);
        for _ in 0..cap {
            match self.pop() {
                Some(v) => out.push(v),
                None => break,
            }
        }
        // out is already in descending order from successive pops; that's
        // a valid max-heap layout, so we skip a heapify pass.
        Self {
            arena: self.arena,
            data: out,
        }
    }

    fn heapify(&mut self) {
        let n = self.data.len();
        if n < 2 {
            return;
        }
        let mut i = n / 2;
        while i > 0 {
            i -= 1;
            // SAFETY: i < n / 2 < n.
            unsafe { self.sift_down(i) };
        }
    }

    /// Move element at `pos` up toward the root until the heap property holds.
    /// `start` lets callers limit how far up the element may travel.
    ///
    /// # Safety
    /// `pos < self.data.len()`.
    unsafe fn sift_up(&mut self, start: usize, pos: usize) {
        // SAFETY: caller guarantees pos < len.
        let mut hole = unsafe { Hole::new(self.data.as_mut_slice(), pos) };
        while hole.pos() > start {
            let parent = (hole.pos() - 1) / 2;
            // SAFETY: parent < hole.pos() < len, and parent != hole.pos().
            if unsafe { hole.element() <= hole.get(parent) } {
                break;
            }
            // SAFETY: same as above.
            unsafe { hole.move_to(parent) };
        }
        // Hole's Drop fills the slot with the saved element.
    }

    /// Move element at `pos` down toward the leaves until the heap property holds.
    ///
    /// # Safety
    /// `pos < self.data.len()`.
    unsafe fn sift_down(&mut self, pos: usize) {
        let end = self.data.len();
        // SAFETY: caller guarantees pos < end.
        let mut hole = unsafe { Hole::new(self.data.as_mut_slice(), pos) };
        let mut child = 2 * hole.pos() + 1;
        while child + 1 < end {
            // Pick greater of the two children.
            // SAFETY: child < end-1, child+1 < end, and child(+1) != hole.pos().
            let pick_right = unsafe { hole.get(child) <= hole.get(child + 1) };
            if pick_right {
                child += 1;
            }
            // SAFETY: child < end, child != hole.pos() by construction.
            if unsafe { hole.element() >= hole.get(child) } {
                return;
            }
            // SAFETY: same as above.
            unsafe { hole.move_to(child) };
            child = 2 * hole.pos() + 1;
        }
        // Trailing single child case.
        if child < end {
            // SAFETY: child < end == len, child != hole.pos().
            if unsafe { hole.element() < hole.get(child) } {
                // SAFETY: same as above.
                unsafe { hole.move_to(child) };
            }
        }
    }
}

impl<'arena, T> IntoIterator for ArenaHeap<'arena, T> {
    type Item = T;
    type IntoIter = bumpalo::collections::vec::IntoIter<'arena, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

/// `Hole` represents a temporarily-vacant slot in the backing slice.
///
/// During sift, instead of swapping pairs of elements (two memcpys per move),
/// we lift the moving element out into `elt`, slide neighbors into the hole
/// position with `ptr::copy_nonoverlapping` (one memcpy per move), and on
/// `Drop` deposit `elt` back into its final slot. Result: ~half the moves.
///
/// The slot at `pos` is logically uninitialized while the hole is alive.
/// All `unsafe` invariants below assume the caller never reads `data[pos]`.
struct Hole<'a, T: 'a> {
    data: &'a mut [T],
    /// The element that was pulled out at construction time.
    elt: ManuallyDrop<T>,
    /// Current index of the hole.
    pos: usize,
}

impl<'a, T> Hole<'a, T> {
    /// # Safety
    /// `pos < data.len()`.
    #[inline]
    unsafe fn new(data: &'a mut [T], pos: usize) -> Self {
        debug_assert!(pos < data.len());
        // SAFETY: pos is in-bounds; we logically remove data[pos].
        let elt = unsafe { ptr::read(data.get_unchecked(pos)) };
        Self {
            data,
            elt: ManuallyDrop::new(elt),
            pos,
        }
    }

    #[inline]
    fn pos(&self) -> usize {
        self.pos
    }

    #[inline]
    fn element(&self) -> &T {
        &self.elt
    }

    /// # Safety
    /// `index < data.len()` and `index != self.pos` (data[pos] is the hole).
    #[inline]
    unsafe fn get(&self, index: usize) -> &T {
        debug_assert!(index < self.data.len());
        debug_assert!(index != self.pos);
        // SAFETY: caller upholds bounds + non-aliasing.
        unsafe { self.data.get_unchecked(index) }
    }

    /// Move the value at `index` into the hole, then track the hole's new pos.
    ///
    /// # Safety
    /// `index < data.len()` and `index != self.pos`.
    #[inline]
    unsafe fn move_to(&mut self, index: usize) {
        debug_assert!(index < self.data.len());
        debug_assert!(index != self.pos);
        // SAFETY: caller upholds bounds + non-aliasing; src and dst do not overlap.
        unsafe {
            let base = self.data.as_mut_ptr();
            let src = base.add(index);
            let dst = base.add(self.pos);
            ptr::copy_nonoverlapping(src, dst, 1);
        }
        self.pos = index;
    }
}

impl<T> Drop for Hole<'_, T> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: pos is always a valid in-bounds index; we restore the saved
        // element (originally read out in `new`) to fill the hole.
        unsafe {
            let dst = self.data.get_unchecked_mut(self.pos);
            ptr::copy_nonoverlapping(&*self.elt, dst, 1);
        }
    }
}

#[cfg(test)]
mod arena_heap_tests {
    use super::*;
    use bumpalo::Bump;

    #[test]
    fn push_pop_returns_max_first() {
        let arena = Bump::new();
        let mut heap = ArenaHeap::<i32>::new(&arena);
        for v in [5, 1, 8, 3, 9, 2, 7] {
            heap.push(v);
        }
        assert_eq!(heap.pop(), Some(9));
        assert_eq!(heap.pop(), Some(8));
        assert_eq!(heap.pop(), Some(7));
        assert_eq!(heap.pop(), Some(5));
        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(2));
        assert_eq!(heap.pop(), Some(1));
        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn peek_does_not_consume() {
        let arena = Bump::new();
        let mut heap = ArenaHeap::<i32>::new(&arena);
        heap.push(3);
        heap.push(7);
        heap.push(1);
        assert_eq!(heap.peek(), Some(&7));
        assert_eq!(heap.len(), 3);
    }

    #[test]
    fn empty_heap_pops_none() {
        let arena = Bump::new();
        let mut heap = ArenaHeap::<i32>::new(&arena);
        assert!(heap.is_empty());
        assert_eq!(heap.pop(), None);
        assert_eq!(heap.peek(), None);
    }

    #[test]
    fn from_vec_heapifies() {
        let arena = Bump::new();
        let mut data = BumpVec::with_capacity_in(8, &arena);
        for v in [3, 1, 4, 1, 5, 9, 2, 6] {
            data.push(v);
        }
        let mut heap = ArenaHeap::from_vec(&arena, data);
        let mut popped = Vec::new();
        while let Some(v) = heap.pop() {
            popped.push(v);
        }
        let mut expected = vec![3, 1, 4, 1, 5, 9, 2, 6];
        expected.sort_by(|a, b| b.cmp(a));
        assert_eq!(popped, expected);
    }

    #[test]
    fn drain_top_returns_topk_descending_when_popped() {
        let arena = Bump::new();
        let mut heap = ArenaHeap::<i32>::new(&arena);
        for v in [5, 1, 8, 3, 9, 2, 7] {
            heap.push(v);
        }
        let mut top3 = heap.drain_top(3);
        assert_eq!(top3.pop(), Some(9));
        assert_eq!(top3.pop(), Some(8));
        assert_eq!(top3.pop(), Some(7));
    }

    #[test]
    fn drain_top_caps_at_size() {
        let arena = Bump::new();
        let mut heap = ArenaHeap::<i32>::new(&arena);
        heap.push(1);
        heap.push(2);
        let top = heap.drain_top(10);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn extend_preserves_heap_invariant() {
        let arena = Bump::new();
        let mut heap = ArenaHeap::<i32>::new(&arena);
        heap.extend([4, 2, 6, 1, 5, 3]);
        let mut last = i32::MAX;
        while let Some(v) = heap.pop() {
            assert!(v <= last);
            last = v;
        }
    }

    #[test]
    fn stress_random_pushes_match_sorted() {
        let arena = Bump::new();
        let mut heap = ArenaHeap::<u32>::new(&arena);
        let mut state = 0xdead_beefu32;
        let mut input = Vec::with_capacity(1024);
        for _ in 0..1024 {
            // xorshift32
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            input.push(state);
            heap.push(state);
        }
        input.sort_by(|a, b| b.cmp(a));
        let mut popped = Vec::with_capacity(1024);
        while let Some(v) = heap.pop() {
            popped.push(v);
        }
        assert_eq!(popped, input);
    }
}
