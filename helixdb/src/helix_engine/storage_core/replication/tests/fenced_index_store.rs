//! An index job whose store reaches the WAL after the WAL's flush task has
//! ended (the write itself: `backend_lsm/tests/fenced_write.rs`).
//!
//! What the index worker stores, it stores durably: transactions and commits
//! that wait for their WAL, where an upsert commits buffered and waits in one
//! flush. A store the batch writer appended after the flush task had ended
//! never returned. Its job kept its worker, of two for the whole writer, and
//! its collection stayed in the executor's active set, where every later job
//! of it waits behind the one that runs.

use super::*;
use crate::helix_engine::storage_core::backend::{BackendError, Namespace, StorageBackend};
use crate::helix_engine::storage_core::backend_any::AnyWrite;
use crate::helix_engine::storage_core::backend_lsm::flush_gate::test_support::{
    joined, wait_until, HeldSsts,
};
use crate::helix_engine::storage_core::backend_lsm::{
    shared_merge_operator, with_db_builder_for_test, with_flush_gate_for_test,
};
use slatedb::bytes::Bytes;
use slatedb::object_store::ObjectStore;
use slatedb::slatedb_common::clock::{DefaultSystemClock, SystemClock, SystemClockTicker};
use slatedb::{MergeOperator, MergeOperatorError};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::Condvar;

/// The key whose fold tells [`HeldClock`] the batch writer's task.
const TELLS_THE_TASK: &[u8] = b"fenced_index_store_tells_the_task";

/// A clock that keeps one read of it waiting: the next one by the batch
/// writer's task, once the test says so. The batch writer reads the clock in
/// its write of a batch, before the WAL append, and at no other time; every
/// other reader of the clock is another task.
#[derive(Debug)]
struct HeldClock {
    clock: DefaultSystemClock,
    held: Mutex<HeldRead>,
    changed: Condvar,
}

#[derive(Debug, Default)]
struct HeldRead {
    /// The batch writer's task, once a fold has told it.
    task: Option<tokio::task::Id>,
    /// 0: no read waits. 1: the task's next one is to. 2: it waits. 3: let go.
    state: u8,
}

impl HeldClock {
    fn new() -> Self {
        Self {
            clock: DefaultSystemClock::new(),
            held: Mutex::default(),
            changed: Condvar::new(),
        }
    }

    fn told(&self, task: Option<tokio::task::Id>) {
        let mut held = self.held.lock().unwrap();
        held.task = held.task.or(task);
    }

    fn knows_the_task(&self) -> bool {
        self.held.lock().unwrap().task.is_some()
    }

    fn hold_the_next_read(&self) {
        self.held.lock().unwrap().state = 1;
    }

    fn a_read_waits(&self) -> bool {
        self.held.lock().unwrap().state == 2
    }

    fn let_go(&self) {
        self.held.lock().unwrap().state = 3;
        self.changed.notify_all();
    }
}

impl SystemClock for HeldClock {
    fn now(&self) -> chrono::DateTime<chrono::Utc> {
        let mut held = self.held.lock().unwrap();
        if held.state == 1 && held.task.is_some() && tokio::task::try_id() == held.task {
            held.state = 2;
            // Bounded: a test that fails before `let_go` leaves no runtime
            // thread waiting for good.
            let _held = self
                .changed
                .wait_timeout_while(held, Duration::from_secs(120), |held| held.state != 3)
                .unwrap();
        }
        self.clock.now()
    }

    fn sleep<'a>(&'a self, duration: Duration) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        self.clock.sleep(duration)
    }

    fn ticker<'a>(&'a self, duration: Duration) -> SystemClockTicker<'a> {
        SystemClockTicker::new(self, duration)
    }
}

/// The collection's merge operator, which tells the clock the task that folds
/// [`TELLS_THE_TASK`]: SlateDB folds a batch's operands in the batch writer.
struct TellsTheTask {
    clock: Arc<HeldClock>,
    folds: Arc<dyn MergeOperator + Send + Sync>,
}

impl MergeOperator for TellsTheTask {
    fn merge(
        &self,
        key: &Bytes,
        existing_value: Option<Bytes>,
        operand: Bytes,
    ) -> Result<Bytes, MergeOperatorError> {
        self.merge_batch(key, existing_value, std::slice::from_ref(&operand))
    }

    fn merge_batch(
        &self,
        key: &Bytes,
        existing_value: Option<Bytes>,
        operands: &[Bytes],
    ) -> Result<Bytes, MergeOperatorError> {
        if !key.ends_with(TELLS_THE_TASK) {
            return self.folds.merge_batch(key, existing_value, operands);
        }
        self.clock.told(tokio::task::try_id());
        Ok(operands.last().cloned().unwrap_or_default())
    }
}

/// A durable commit of one write through the collection's backend.
fn commit_durably(
    storage: &HelixGraphStorage,
    write: impl FnOnce(&mut AnyWrite<'_>) -> Result<(), BackendError>,
) -> Result<(), BackendError> {
    let mut w = storage.backend.begin_write()?;
    write(&mut w)?;
    storage.backend.commit(w)
}

fn job(collection: &str, storage: &Arc<HelixGraphStorage>) -> IndexJob {
    IndexJob {
        collection: collection.to_string(),
        storage: Arc::downgrade(storage),
        spaces_to_check: Vec::new(),
        hnsw_config: HNSWConfig::new(Some(8), Some(32), Some(64)),
        flat_scan_threshold: 1,
    }
}

fn nothing_queued_or_running(executor: &IndexExecutor) -> bool {
    let state = executor.state.0.lock().unwrap();
    state.queued_jobs.is_empty() && state.active.is_empty()
}

#[test]
fn an_index_job_whose_store_is_appended_after_the_wal_flush_task_ended_leaves_its_worker() {
    let store = HeldSsts::holding_wal("no WAL is held yet");
    let clock = Arc::new(HeldClock::new());
    let dir = TempDir::new().unwrap();
    // An in-memory writer whose WAL alone is in `store`.
    let storage = {
        let (clock, wal) = (Arc::clone(&clock), Arc::clone(&store));
        with_flush_gate_for_test(None, || {
            with_db_builder_for_test(
                move |builder| {
                    builder
                        .with_merge_operator(Arc::new(TellsTheTask {
                            clock: Arc::clone(&clock),
                            folds: shared_merge_operator(),
                        }))
                        .with_system_clock(Arc::clone(&clock) as Arc<dyn SystemClock>)
                        .with_wal_object_store(Arc::clone(&wal) as Arc<dyn ObjectStore>)
                },
                || {
                    HelixGraphStorage::new(
                        dir.path().to_str().unwrap(),
                        Config::new(8, 32, 64, 1).with_lsm_in_memory(),
                    )
                },
            )
        })
        .unwrap()
    };
    let storage = Arc::new(storage);
    commit_durably(&storage, |w| {
        storage
            .backend
            .merge(w, Namespace::Metadata, TELLS_THE_TASK, b"1")
    })
    .unwrap();
    assert!(clock.knows_the_task(), "no fold in the batch writer");
    store.fence_the_next_wal();

    // A durable write whose WAL put waits at the store: the flush task is
    // alive and the database open.
    store.hold("/wal/");
    let flushing = {
        let storage = Arc::clone(&storage);
        thread::spawn(move || {
            commit_durably(&storage, |w| {
                storage
                    .backend
                    .put(w, Namespace::Metadata, b"fenced_index_store_row", b"v")
            })
        })
    };
    wait_until("a WAL put at the store", || store.held() == 1);

    // A job whose first store, of the dense metadata the upsert path left
    // dirty, is past the closed check and held in the batch writer before
    // its append.
    let collection = "fenced_index_store";
    let executor = IndexExecutor::new();
    storage.dense_metadata_dirty.store(true, Ordering::Release);
    clock.hold_the_next_read();
    executor.submit(job(collection, &storage)).unwrap();
    wait_until("the job's store in the batch writer", || {
        clock.a_read_waits()
    });
    assert!(
        !storage.dense_metadata_dirty.load(Ordering::Acquire),
        "the write in the batch writer is not the job's store"
    );

    // The put finds the object. The flush task ends: it closes the database,
    // freezes what the WAL buffer holds and tells its waiters.
    store.release_all();
    let told = joined("the write whose WAL the flush task was putting", flushing)
        .expect_err("the write whose WAL the flush task was putting");
    assert!(told.to_string().contains("fenced"), "{told}");
    // That one was told after the freeze, so the job's store is appended to
    // no WAL the flush task knew of.
    clock.let_go();

    // The job ends, and its worker takes the collection's next.
    executor.submit(job(collection, &storage)).unwrap();
    wait_until(
        "the job has ended and the collection's next has run",
        || nothing_queued_or_running(&executor),
    );
}
