//! Parallel L0 memtable flusher pipeline.
//!
//! Coordinates the flush of immutable memtables to L0 SSTs in object storage.
//! The pipeline consists of three components:
//! - **tracker**: event loop for flush requests, dispatch, and waiter tracking
//! - **uploader**: parallel worker pool for SST build and upload
//! - **manifest_writer**: ordered manifest retirement and checkpoint creation

mod manifest_writer;
mod tracker;
mod uploader;

pub(crate) use manifest_writer::FlushResult;
#[cfg(test)]
pub(crate) use tracker::MANIFEST_REFRESH_COUNT;

use crate::checkpoint::CheckpointCreateResult;
use crate::config::CheckpointOptions;
use crate::db::DbInner;
use crate::db_status::ClosedResultWriter;
use crate::dispatcher::MessageHandlerExecutor;
use crate::error::SlateDBError;
use crate::manifest::store::FenceableManifest;
use crate::mem_table::ImmutableMemtable;
use crate::memtable_flusher::manifest_writer::ManifestWriter;
use crate::memtable_flusher::tracker::{FlushTracker, TrackerMessage};
use crate::memtable_flusher::uploader::Uploader;
use crate::utils::SafeSender;
use futures::future::BoxFuture;
use log::warn;
use std::sync::Arc;
use std::task::Poll;
use tokio::runtime::Handle;
use tokio::sync::{oneshot, watch};

const TRACKER_TASK_NAME: &str = "l0_flush_tracker";

/// HELION LOCAL PATCH (see vendor/slatedb/PROVENANCE.md): asked before an L0
/// flush encodes its memtable. A flush holds its whole SST encoded until the
/// upload ends, and each database has its own uploader workers, so without it
/// nothing bounds how many of a process's databases encode at the same moment.
/// One gate is shared by the databases it should bound together
/// ([`crate::DbBuilder::with_l0_flush_gate`]).
pub trait L0FlushGate: Send + Sync {
    /// Resolves when a flush of the database at path `db`, whose SST is
    /// estimated at `sst_bytes`, may start. What it returns is dropped when
    /// that flush's upload has ended, however it ended, and while a failed
    /// upload waits to be tried again. Dropped unresolved, it leaves nothing
    /// counted.
    fn admit(&self, db: &str, sst_bytes: usize) -> BoxFuture<'_, Box<dyn Send>>;

    /// The same for a flush that does not wait, because its database is
    /// closing: it is counted, and returns at once.
    fn start(&self, db: &str, sst_bytes: usize) -> Box<dyn Send>;

    /// Told each time the upload of an L0 SST of the database at `db` has
    /// failed with `error` and waits `retry_in` to be tried again: the
    /// uploader writes no line for it. An SST sent in one put fails only
    /// with an error the object store wrapper does not retry inside the
    /// call; one sent in parts, with any error of a part or of the call
    /// that completes it.
    fn upload_failed(&self, _db: &str, _error: &str, _retry_in: std::time::Duration) {}
}

/// One flush of this database waiting at its [`L0FlushGate`], for as long as
/// it lives.
struct WaitingAtGate<'a>(&'a watch::Sender<usize>);

impl Drop for WaitingAtGate<'_> {
    fn drop(&mut self) {
        self.0.send_modify(|waiting| *waiting -= 1);
    }
}

/// Flush request target exposed by the memtable flusher.
#[derive(Clone, Copy, Debug)]
pub(crate) enum FlushTarget {
    /// Return the current durability frontier without initiating new flush work.
    /// Used for `CheckpointScope::Durable` when the caller just needs a consistent
    /// snapshot of what is already durable.
    CurrentDurable,
    /// Wait until all currently known immutable memtables are durably flushed. Used
    /// for explicit `flush()` calls and checkpoint creation, where full durability
    /// is required before proceeding.
    All,
}

/// Parallel L0 memtable flusher subsystem.
pub(crate) struct MemtableFlusher {
    messages_tx: SafeSender<tracker::TrackerMessage>,
    messages_rx: async_channel::Receiver<tracker::TrackerMessage>,
    /// HELION LOCAL PATCH: see [`L0FlushGate`]. `None`: no flush is held back.
    gate: Option<Arc<dyn L0FlushGate>>,
    /// This database's path, which the gate is told: it names it in its logs.
    path: String,
    /// Flushes of this database that `gate` now keeps waiting. A watch, so
    /// that a write held back for one is told when none waits any more.
    waiting_at_gate: watch::Sender<usize>,
}

impl MemtableFlusher {
    /// Creates a new memtable flusher.
    /// Call [`start`](Self::start) to register background tasks with the executor.
    pub(crate) fn new(closed_result: &dyn ClosedResultWriter) -> Self {
        let (messages_tx, messages_rx) =
            SafeSender::unbounded_channel(closed_result.result_reader());
        Self {
            messages_tx,
            messages_rx,
            gate: None,
            path: String::new(),
            waiting_at_gate: watch::Sender::new(0),
        }
    }

    /// HELION LOCAL PATCH: see [`L0FlushGate`]. `path` is this database's.
    pub(crate) fn with_gate(mut self, gate: Option<Arc<dyn L0FlushGate>>, path: String) -> Self {
        self.gate = gate;
        self.path = path;
        self
    }

    /// HELION LOCAL PATCH: a flush of this database waits at its gate.
    /// [`DbInner::maybe_apply_backpressure`] holds writes back while it does:
    /// the memory a waiting flush was kept from taking must not be taken by
    /// the next memtable instead.
    pub(crate) fn flush_waits_at_gate(&self) -> bool {
        *self.waiting_at_gate.borrow() > 0
    }

    /// HELION LOCAL PATCH: resolves once no flush of this database waits at
    /// its gate, at once if none does. A write that
    /// [`DbInner::maybe_apply_backpressure`] held back for a waiting flush
    /// goes on when that flush is let in, as a write does that arrives after
    /// it, and not only when its upload has ended. A watch and not a notify:
    /// the count is read again as the wait begins, so a flush let in between
    /// the write's look and its wait is not missed.
    pub(crate) async fn no_flush_waits_at_gate(&self) {
        let mut waiting = self.waiting_at_gate.subscribe();
        // The sender is `self`'s: the wait cannot end on a closed channel.
        let _ = waiting.wait_for(|waiting| *waiting == 0).await;
    }

    /// HELION LOCAL PATCH: asks the gate, if there is one, before
    /// `imm_memtable` is encoded, and again before a failed upload of it is
    /// tried again. The caller keeps what this returns until the upload has
    /// ended or failed.
    ///
    /// Never waits where the flush is what makes a write durable (no WAL), nor
    /// once the database is closing: a close holds what its caller holds, and
    /// has a deadline of its own at shutdown. A flush that is waiting when its
    /// database starts to close stops waiting. Either is still counted.
    pub(crate) async fn admit_at_gate(
        &self,
        db: &DbInner,
        imm_memtable: &ImmutableMemtable,
    ) -> Option<Box<dyn Send>> {
        let gate = self.gate.as_ref()?;
        if !db.wal_enabled {
            return None;
        }
        let metadata = imm_memtable.table().metadata();
        let sst_bytes = db
            .table_store
            .estimate_encoded_size_compacted(metadata.entry_num, metadata.entries_size_in_bytes);
        if db.check_closed().is_err() {
            return Some(gate.start(&self.path, sst_bytes));
        }
        let mut admitted = gate.admit(&self.path, sst_bytes);
        // A flush that starts as it asks was never waiting. Counted as waiting
        // while it asked, it would send a write of its database that looked in
        // that moment into the wait for its whole upload, with memory ample.
        let asked = std::future::poll_fn(|cx| Poll::Ready(admitted.as_mut().poll(cx))).await;
        if let Poll::Ready(admitted) = asked {
            return Some(admitted);
        }
        self.waiting_at_gate.send_modify(|waiting| *waiting += 1);
        let _waiting = WaitingAtGate(&self.waiting_at_gate);
        let mut closed = db.status_manager.result_reader();
        Some(tokio::select! {
            biased;
            _ = closed.await_value() => gate.start(&self.path, sst_bytes),
            admitted = admitted => admitted,
        })
    }

    pub(crate) fn start(
        &self,
        inner: Arc<DbInner>,
        manifest: FenceableManifest,
        tokio_handle: &Handle,
        executor: &MessageHandlerExecutor,
        closed_result: &dyn ClosedResultWriter,
    ) -> Result<(), SlateDBError> {
        let uploader = Uploader::start(
            Arc::clone(&inner),
            closed_result,
            self.messages_tx.clone(),
            executor,
            tokio_handle,
        )?;

        let manifest_writer = ManifestWriter::start(
            Arc::clone(&inner),
            manifest,
            inner.settings.manifest_poll_interval,
            closed_result,
            executor,
            tokio_handle,
            self.messages_tx.clone(),
        )?;

        let tracker = FlushTracker::new(inner, uploader, manifest_writer);
        executor.add_handler(
            TRACKER_TASK_NAME.to_string(),
            Box::new(tracker),
            self.messages_rx.clone(),
            tokio_handle,
        )?;

        Ok(())
    }

    /// Enqueues a manifest refresh and waits until it completes.
    pub(crate) async fn refresh_manifest(&self) -> Result<(), SlateDBError> {
        let (tx, rx) = oneshot::channel();
        self.messages_tx
            .send(TrackerMessage::PollManifest { sender: tx })?;
        rx.await.map_err(SlateDBError::ReadChannelError)?
    }

    /// Processes one flush request using the requested target.
    pub(crate) async fn flush(&self, target: FlushTarget) -> Result<FlushResult, SlateDBError> {
        let (tx, rx) = oneshot::channel();
        self.messages_tx
            .send(tracker::TrackerMessage::FlushRequest { target, sender: tx })?;
        rx.await.map_err(SlateDBError::ReadChannelError)?
    }

    /// Notifies the flusher that a memtable may have been frozen.
    /// Triggers reconcile and dispatch without waiting for a result.
    pub(crate) fn notify_memtable_frozen(&self) -> Result<(), SlateDBError> {
        self.messages_tx
            .send(tracker::TrackerMessage::MemtableFrozen)
    }

    /// Creates a checkpoint using the memtable flusher's flush semantics.
    pub(crate) async fn create_checkpoint(
        &self,
        target: FlushTarget,
        options: CheckpointOptions,
    ) -> Result<CheckpointCreateResult, SlateDBError> {
        let (tx, rx) = oneshot::channel();
        self.messages_tx
            .send(tracker::TrackerMessage::CheckpointRequest {
                target,
                options,
                sender: tx,
            })?;
        rx.await.map_err(SlateDBError::ReadChannelError)?
    }

    /// Closes the flusher and its subsystems via the executor.
    /// On clean shutdown, the caller is expected to have already flushed
    /// (via `Db::flush`), so the pipeline should be idle. The ordering
    /// here matters mainly for error shutdown: shutting down the uploader
    /// and manifest writer first ensures their final completion messages
    /// reach the tracker before it drains.
    pub(crate) async fn shutdown(executor: &MessageHandlerExecutor) {
        Uploader::shutdown(executor).await;
        ManifestWriter::shutdown(executor).await;
        if let Err(e) = executor.shutdown_task(TRACKER_TASK_NAME).await {
            warn!("failed to shutdown l0 flush tracker [error={:?}]", e);
        }
    }
}
