//! A durable write that reaches the WAL after the WAL's flush task has ended.
//!
//! The flush task ends on a fatal error: a newer writer of the same path has
//! fenced this one and its WAL put finds the newer writer's object. It marks
//! the database closed, tells its error to every WAL that waits, and only once
//! it has ended is the batch writer stopped. A write that had passed the
//! closed check before that, and that the batch writer appended after it, sat
//! in a WAL nobody would flush or tell: its commit never returned.
//!
//! The test leaves the object itself (`HeldSsts::fence_the_next_wal`).

use super::super::flush_gate::test_support::{joined, HeldSsts};
use super::*;
use std::sync::{Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// A merge operator whose first fold waits until the test lets it go. SlateDB
/// folds the operands of a batch inside the batch writer's write of it, before
/// the WAL append: the write whose fold waits is past the closed check and
/// not yet in the WAL, and the batch writer takes no other.
#[derive(Default)]
struct HeldFold {
    /// 0: nothing folded yet. 1: a fold waits. 2: let go.
    state: Mutex<u8>,
    changed: Condvar,
}

impl HeldFold {
    fn waits(&self) -> bool {
        *self.state.lock().unwrap() == 1
    }

    fn let_go(&self) {
        *self.state.lock().unwrap() = 2;
        self.changed.notify_all();
    }
}

impl MergeOperator for HeldFold {
    fn merge(
        &self,
        _key: &Bytes,
        _existing_value: Option<Bytes>,
        operand: Bytes,
    ) -> Result<Bytes, MergeOperatorError> {
        let mut state = self.state.lock().unwrap();
        if *state == 0 {
            *state = 1;
        }
        // Bounded: a test that fails before `let_go` leaves no runtime thread
        // waiting for good.
        let _state = self
            .changed
            .wait_timeout_while(state, Duration::from_secs(120), |state| *state != 2)
            .unwrap();
        Ok(operand)
    }
}

/// A writer of `path` that flushes its WAL every 5 ms and folds with `fold`.
fn open_folding(store: &Arc<HeldSsts>, path: &str, fold: &Arc<HeldFold>) -> LsmBackend {
    let mut settings = Settings::default();
    settings.flush_interval = Some(Duration::from_millis(5));
    let fold = Arc::clone(fold);
    with_flush_gate_for_test(None, || {
        with_db_builder_for_test(
            move |builder| {
                builder
                    .with_merge_operator(Arc::clone(&fold) as Arc<dyn MergeOperator + Send + Sync>)
            },
            || {
                LsmBackend::open_with_store_settings(
                    path,
                    Arc::clone(store) as Arc<dyn ObjectStore>,
                    Some(settings),
                )
            },
        )
    })
    .unwrap()
}

fn put_durably(be: &LsmBackend, key: &[u8]) -> Result<(), BackendError> {
    let mut w = be.begin_write()?;
    be.put(&mut w, Namespace::Nodes, key, b"v")?;
    be.commit(w)
}

/// A durable write whose batch holds a merge operand: the batch writer folds
/// it before the append.
fn fold_durably(be: &LsmBackend) -> Result<(), BackendError> {
    let mut w = be.begin_write()?;
    be.merge(&mut w, Namespace::Nodes, b"folded", b"1")?;
    be.commit(w)
}

/// Waits until `reached`, for 30 s at most, with `write` still on its way:
/// one that has returned by then is the failure, with what it returned.
fn wait_with<T: std::fmt::Debug>(
    what: &str,
    write: JoinHandle<T>,
    reached: impl Fn() -> bool,
) -> JoinHandle<T> {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !reached() {
        if write.is_finished() {
            panic!("the write returned before {what}: {:?}", write.join());
        }
        assert!(Instant::now() < deadline, "never: {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
    write
}

fn assert_fenced(what: &str, result: Result<(), BackendError>) {
    let error = result.expect_err(what);
    assert!(LsmBackend::is_fenced_conflict(&error), "{what}: {error}");
}

#[test]
fn a_durable_write_appended_after_the_wal_flush_task_ended_is_told_its_error() {
    let store = HeldSsts::holding_wal("no WAL is held yet");
    let path = "fenced/appended-after-the-flush-task-ended";
    let fold = Arc::new(HeldFold::default());
    let stale = Arc::new(open_folding(&store, path, &fold));
    store.fence_the_next_wal();

    // A durable write whose WAL put waits at the store: the flush task is
    // alive and the database open.
    store.hold("/wal/");
    let flushing = {
        let stale = Arc::clone(&stale);
        std::thread::spawn(move || put_durably(&stale, b"k1"))
    };
    let flushing = wait_with("its WAL put is at the store", flushing, || {
        store.held() == 1
    });
    // A second one, past the closed check and held in the batch writer
    // before its append.
    let appended_after = {
        let stale = Arc::clone(&stale);
        std::thread::spawn(move || fold_durably(&stale))
    };
    let appended_after = wait_with("it is in the batch writer", appended_after, || fold.waits());

    // The put finds the object. The flush task ends: it closes the database,
    // freezes what the WAL buffer holds and tells its waiters.
    store.release_all();
    assert_fenced(
        "the write whose WAL the flush task was putting",
        joined("the write whose WAL the flush task was putting", flushing),
    );
    assert!(matches!(stale.close_reason(), Some(CloseReason::Fenced)));
    // That one was told after the freeze, so what the batch writer appends
    // now is in no WAL the flush task knew of.
    fold.let_go();
    assert_fenced(
        "the write appended after the flush task ended",
        joined(
            "the write appended after the flush task ended",
            appended_after,
        ),
    );
}
