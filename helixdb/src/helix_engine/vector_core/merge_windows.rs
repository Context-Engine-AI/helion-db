//! Point removals a merge must not republish.
//!
//! A merge reads its sources, builds for seconds to minutes and then
//! publishes its target. A point deleted or rewritten in between loses its
//! rows in the sources, but the row the merge already read would come back
//! in the target, whose segment id is above the tail holding the rewritten
//! copy. So a merge keeps a window open from before it reads its sources
//! until it publishes: a removal that commits meanwhile is recorded in it,
//! and the merge deletes the recorded ids from its target before publishing.
//!
//! The lock guards the bookkeeping below and nothing else: no read, write or
//! commit runs under it. A point mutation takes it twice, for a few map
//! operations each; it never waits for a merge. A merge waits only for the
//! removals already under way at two moments, so it always finishes.

use std::collections::{HashMap, HashSet};
use std::sync::{Condvar, Mutex, MutexGuard};

#[derive(Default)]
struct State {
    /// Removals and windows draw their tickets from one counter, so a ticket
    /// also says which of two started first.
    next: u64,
    /// Removals between their plan and their commit.
    removing: HashSet<u64>,
    open: HashMap<u64, Window>,
}

#[derive(Default)]
struct Window {
    /// The next ticket when the target held every row the merge read. A
    /// removal that starts later finds its row in the target and deletes it
    /// there itself.
    written_at: Option<u64>,
    removed: Vec<u128>,
}

#[derive(Default)]
pub(crate) struct MergeWindows {
    state: Mutex<State>,
    changed: Condvar,
}

impl MergeWindows {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Call before a point mutation plans which dense rows it removes.
    pub(crate) fn begin_removal(&self) -> Removal<'_> {
        let mut state = self.lock();
        let ticket = state.next;
        state.next += 1;
        state.removing.insert(ticket);
        Removal {
            windows: self,
            ticket,
        }
    }

    fn end_removal(&self, ticket: u64, removed: &[u128]) {
        let mut state = self.lock();
        if !state.removing.remove(&ticket) {
            return;
        }
        for (opened, window) in &mut state.open {
            // Begun before the window: the merge's read sees its commit
            // (`open` waited for it). Begun after the target was written: it
            // deleted its rows from the target itself.
            if ticket > *opened && window.written_at.map_or(true, |at| ticket < at) {
                window.removed.extend_from_slice(removed);
            }
        }
        drop(state);
        self.changed.notify_all();
    }

    /// Call after a merge chose its sources and before it reads them. Waits
    /// for the removals already under way: they are not recorded in this
    /// window, so the read has to see their commit.
    pub(crate) fn open(&self) -> MergeWindow<'_> {
        let mut state = self.lock();
        let id = state.next;
        state.next += 1;
        state.open.insert(id, Window::default());
        while state.removing.iter().any(|ticket| *ticket < id) {
            state = self.changed.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        MergeWindow { windows: self, id }
    }
}

/// A point mutation that removes dense rows, from before its plan until
/// after its commit. Dropped without `committed`, it removed nothing.
pub(crate) struct Removal<'a> {
    windows: &'a MergeWindows,
    ticket: u64,
}

impl Removal<'_> {
    /// The mutation's batch committed and removed the dense rows of `ids`.
    pub(crate) fn committed(self, ids: &[u128]) {
        self.windows.end_removal(self.ticket, ids);
    }
}

impl Drop for Removal<'_> {
    fn drop(&mut self) {
        self.windows.end_removal(self.ticket, &[]);
    }
}

/// A merge, from before it reads its sources until it publishes or fails.
pub(crate) struct MergeWindow<'a> {
    windows: &'a MergeWindows,
    id: u64,
}

impl MergeWindow<'_> {
    /// Call once the target holds every row the merge read, and again until
    /// it returns nothing: the ids removed since the merge read its sources,
    /// which the caller deletes from the target before it publishes. Waits
    /// for a removal begun before the first call to commit; later ones see
    /// the whole target. The window is closed when this returns nothing.
    pub(crate) fn removed(&self) -> Vec<u128> {
        let mut state = self.windows.lock();
        let next = state.next;
        loop {
            let Some(window) = state.open.get_mut(&self.id) else {
                return Vec::new();
            };
            let written_at = *window.written_at.get_or_insert(next);
            if !window.removed.is_empty() {
                return std::mem::take(&mut window.removed);
            }
            let pending = |ticket: &u64| *ticket > self.id && *ticket < written_at;
            if !state.removing.iter().any(pending) {
                state.open.remove(&self.id);
                return Vec::new();
            }
            state = self
                .windows
                .changed
                .wait(state)
                .unwrap_or_else(|e| e.into_inner());
        }
    }
}

impl Drop for MergeWindow<'_> {
    fn drop(&mut self) {
        self.windows.lock().open.remove(&self.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;
    use std::time::Duration;

    #[test]
    fn a_window_collects_the_removals_committed_while_its_merge_builds() {
        let windows = MergeWindows::default();
        let merge = windows.open();
        let slower = windows.open();
        windows.begin_removal().committed(&[1, 2]);
        // A mutation that failed removed nothing.
        drop(windows.begin_removal());
        assert_eq!(merge.removed(), vec![1, 2]);

        // `merge` has written its target: a later removal deletes from the
        // target itself, and only the merge still building records it.
        windows.begin_removal().committed(&[3]);
        assert!(merge.removed().is_empty());
        assert_eq!(slower.removed(), vec![1, 2, 3]);
        assert!(slower.removed().is_empty());

        // Both windows are closed.
        windows.begin_removal().committed(&[4]);
        assert!(merge.removed().is_empty());
        assert!(slower.removed().is_empty());
        assert!(windows.lock().open.is_empty());
    }

    #[test]
    fn a_merge_waits_for_the_removals_it_would_not_see() {
        let windows = &MergeWindows::default();
        let (opened_tx, opened_rx) = channel();
        let (go_tx, go_rx) = channel();
        let (removed_tx, removed_rx) = channel();
        let before = windows.begin_removal();
        std::thread::scope(|scope| {
            scope.spawn(move || {
                let window = windows.open();
                opened_tx.send(()).unwrap();
                go_rx.recv().unwrap();
                removed_tx.send(window.removed()).unwrap();
                removed_tx.send(window.removed()).unwrap();
            });
            // Begun before the window: the merge may not read before it commits.
            assert!(opened_rx.recv_timeout(Duration::from_millis(100)).is_err());
            before.committed(&[1]);
            opened_rx.recv().unwrap();

            // Begun while the merge builds: the merge may not publish before
            // it knows what this one removed.
            let during = windows.begin_removal();
            go_tx.send(()).unwrap();
            assert!(removed_rx.recv_timeout(Duration::from_millis(100)).is_err());
            during.committed(&[2]);
            assert_eq!(removed_rx.recv().unwrap(), vec![2]);
            assert!(removed_rx.recv().unwrap().is_empty());
        });
    }
}
