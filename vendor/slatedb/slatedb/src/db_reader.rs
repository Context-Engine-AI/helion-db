use crate::bytes_range::{ByteRangeBounds, BytesRange};
use crate::cached_object_store::CachedObjectStore;
use crate::clock::MonotonicClock;
use crate::config::{CheckpointOptions, DbReaderOptions, ReadOptions, ScanOptions};
use crate::db_cache_manager::{self, CacheTarget};
use crate::db_common::extract_segment_prefix;
use crate::db_iter::DbIteratorGuard;
use crate::db_state::{collect_touched_segments, SsTableId};
use crate::db_stats::DbStats;
use crate::db_status::{ClosedResultWriter, DbStatus, DbStatusManager};
use crate::dispatcher::{MessageHandler, MessageHandlerExecutor, MessageTickerDef};
use crate::error::SlateDBError;
use crate::iter::IterationOrder;
use crate::manifest::store::{ManifestStore, StoredManifest};
use crate::manifest::{Manifest, ManifestCore, VersionedManifest};
use crate::mem_table::{ImmutableMemtable, KVTable, WritableKVTable};
use crate::merge_operator::MergeOperatorType;
use crate::oracle::DbReaderOracle;
use crate::paths::PathResolver;
use crate::prefix_extractor::PrefixExtractor;
use crate::reader::{DbStateReader, Reader, ScanContext};
use crate::sst_iter::SstIteratorOptions;
use crate::tablestore::TableStore;
use crate::types::KeyValue;
use crate::utils::IdGenerator;
use crate::wal_replay::{WalReplayIterator, WalReplayOptions};
use crate::{Checkpoint, DbIterator, DbSnapshot};
use crate::{DbCacheManagerOps, DbMetadataOps, DbReadOps};
use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use log::{info, warn};
use object_store::path::Path;
use object_store::ObjectStore;
use parking_lot::RwLock;
use slatedb_common::clock::SystemClock;
use slatedb_common::DbRand;
use std::collections::{BTreeSet, HashMap};
use std::ops::Sub;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;
use std::sync::{Arc, Weak};
use tokio::runtime::Handle;
use tokio::sync::Notify;
use uuid::Uuid;

pub(crate) const DB_READER_TASK_NAME: &str = "manifest_poller";

/// Read-only interface for accessing a database from either
/// the latest persistent state or from an arbitrary checkpoint.
///
/// A reader that follows the latest state is read-only with respect to user
/// data, but it appends manifest versions to create, refresh, and release GC
/// checkpoints. Opening from an explicit checkpoint does not start that
/// manifest-writing poller.
pub struct DbReader {
    inner: Arc<DbReaderInner>,
    task_executor: MessageHandlerExecutor,
    pub(crate) object_store_cache: Option<Arc<CachedObjectStore>>,
}

pub(crate) struct DbReaderInner {
    manifest_store: Arc<ManifestStore>,
    table_store: Arc<TableStore>,
    options: DbReaderOptions,
    state: RwLock<Arc<CheckpointState>>,
    system_clock: Arc<dyn SystemClock>,
    #[cfg(test)]
    user_checkpoint_id: Option<Uuid>,
    oracle: Arc<DbReaderOracle>,
    reader: Reader,
    db_stats: DbStats,
    status_manager: DbStatusManager,
    segment_extractor: Option<Arc<dyn PrefixExtractor>>,
    rand: Arc<DbRand>,
    /// Kept alive so the underlying `MetricsRecorder` is not dropped while
    /// metric handles in `DbStats` (and other stats structs) are still in use.
    /// See: https://github.com/slatedb/slatedb/issues/1469
    #[allow(dead_code)]
    recorder: slatedb_common::metrics::MetricsRecorderHelper,
}

#[derive(Debug)]
enum DbReaderMessage {
    PollManifest,
}

#[derive(Clone)]
pub(crate) struct CheckpointState {
    generation: Arc<ReaderGeneration>,
    imm_memtable: ReplayMemtables,
    last_wal_id: u64,
    last_remote_persisted_seq: u64,
}

struct ReaderGeneration {
    checkpoint: RwLock<Checkpoint>,
    manifest: Manifest,
    operation_state: AtomicUsize,
    operation_drained: Notify,
}

const GENERATION_INVALID: usize = 1 << (usize::BITS - 1);
const GENERATION_OPERATION_COUNT: usize = !GENERATION_INVALID;

struct ReaderGenerationPermit {
    generation: Arc<ReaderGeneration>,
}

impl Drop for ReaderGenerationPermit {
    fn drop(&mut self) {
        self.generation.exit_operation();
    }
}

impl ReaderGeneration {
    fn new(checkpoint: Checkpoint, manifest: Manifest) -> Arc<Self> {
        Arc::new(Self {
            checkpoint: RwLock::new(checkpoint),
            manifest,
            operation_state: AtomicUsize::new(0),
            operation_drained: Notify::new(),
        })
    }

    fn checkpoint(&self) -> Checkpoint {
        self.checkpoint.read().clone()
    }

    fn invalidate(&self) {
        self.operation_state
            .fetch_or(GENERATION_INVALID, Ordering::AcqRel);
    }

    fn acquire(self: &Arc<Self>) -> Result<ReaderGenerationPermit, SlateDBError> {
        self.enter_operation()?;
        Ok(ReaderGenerationPermit {
            generation: Arc::clone(self),
        })
    }

    fn enter_operation(&self) -> Result<(), SlateDBError> {
        let mut state = self.operation_state.load(Ordering::Acquire);
        loop {
            if state & GENERATION_INVALID != 0 {
                return Err(SlateDBError::CheckpointLeaseLost(self.checkpoint().id));
            }
            assert_ne!(
                state & GENERATION_OPERATION_COUNT,
                GENERATION_OPERATION_COUNT,
                "reader generation in-flight operation count overflow"
            );
            match self.operation_state.compare_exchange_weak(
                state,
                state + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(current) => state = current,
            }
        }
    }

    fn exit_operation(&self) {
        let previous = self.operation_state.fetch_sub(1, Ordering::AcqRel);
        assert!(
            previous & GENERATION_OPERATION_COUNT > 0,
            "reader generation in-flight operation count underflow"
        );
        if previous & GENERATION_OPERATION_COUNT == 1 {
            self.operation_drained.notify_waiters();
        }
    }

    async fn drain(&self) {
        loop {
            let notified = self.operation_drained.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.operation_state.load(Ordering::Acquire) & GENERATION_OPERATION_COUNT == 0 {
                return;
            }
            notified.await;
        }
    }
}

impl DbIteratorGuard for ReaderGeneration {
    fn enter(&self) -> Result<(), SlateDBError> {
        self.enter_operation()
    }

    fn exit(&self) {
        self.exit_operation();
    }
}

/// Persistent newest-first collection of WAL-replayed immutable memtables.
///
/// A new replay poll only prepends newly decoded tables and shares the
/// existing tail with snapshots and in-flight reads. This makes applying a WAL
/// delta proportional to the delta rather than to the complete replay history.
#[derive(Clone, Default)]
struct ReplayMemtables {
    head: Option<Arc<ReplayMemtableNode>>,
    len: usize,
}

struct ReplayMemtableNode {
    table: Arc<ImmutableMemtable>,
    older: Option<Arc<ReplayMemtableNode>>,
}

type ReplayPublisher<'a> = &'a mut (dyn FnMut(&ReplayMemtables, u64, u64) + Send);

impl CheckpointState {
    pub(crate) fn applied_seq(&self) -> u64 {
        self.last_remote_persisted_seq
    }
}

impl ReplayMemtables {
    fn prepend(&mut self, table: Arc<ImmutableMemtable>) {
        self.head = Some(Arc::new(ReplayMemtableNode {
            table,
            older: self.head.take(),
        }));
        self.len += 1;
    }

    fn front(&self) -> Option<&Arc<ImmutableMemtable>> {
        self.head.as_ref().map(|node| &node.table)
    }

    fn iter(&self) -> ReplayMemtablesIter<'_> {
        ReplayMemtablesIter {
            next: self.head.as_deref(),
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl FromIterator<Arc<ImmutableMemtable>> for ReplayMemtables {
    fn from_iter<T: IntoIterator<Item = Arc<ImmutableMemtable>>>(iter: T) -> Self {
        let tables = iter.into_iter().collect::<Vec<_>>();
        let mut result = Self::default();
        for table in tables.into_iter().rev() {
            result.prepend(table);
        }
        result
    }
}

struct ReplayMemtablesIter<'a> {
    next: Option<&'a ReplayMemtableNode>,
}

impl Iterator for ReplayMemtablesIter<'_> {
    type Item = Arc<ImmutableMemtable>;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.next?;
        self.next = node.older.as_deref();
        Some(Arc::clone(&node.table))
    }
}

static EMPTY_TABLE: LazyLock<Arc<KVTable>> = LazyLock::new(|| Arc::new(KVTable::new()));

impl DbStateReader for CheckpointState {
    fn memtable(&self) -> Arc<KVTable> {
        Arc::clone(&EMPTY_TABLE)
    }

    fn imm_memtables(&self) -> Box<dyn Iterator<Item = Arc<ImmutableMemtable>> + Send + '_> {
        Box::new(self.imm_memtable.iter())
    }

    fn core(&self) -> &ManifestCore {
        &self.generation.manifest.core
    }
}

impl From<&CheckpointState> for VersionedManifest {
    fn from(state: &CheckpointState) -> Self {
        let checkpoint = state.generation.checkpoint();
        Self::from_manifest(checkpoint.manifest_id, state.generation.manifest.clone())
    }
}

impl DbReaderInner {
    async fn new(
        manifest_store: Arc<ManifestStore>,
        table_store: Arc<TableStore>,
        options: DbReaderOptions,
        checkpoint_id: Option<Uuid>,
        merge_operator: Option<MergeOperatorType>,
        segment_extractor: Option<Arc<dyn PrefixExtractor>>,
        system_clock: Arc<dyn SystemClock>,
        rand: Arc<DbRand>,
        recorder: slatedb_common::metrics::MetricsRecorderHelper,
        mut manifest: StoredManifest,
    ) -> Result<Self, SlateDBError> {
        let checkpoint =
            Self::get_or_create_checkpoint(&mut manifest, checkpoint_id, &options, rand.clone())
                .await?;

        let db_stats = DbStats::new(&recorder);
        let replay_new_wals = checkpoint_id.is_none() && !options.skip_wal_replay;
        let initial_state = Arc::new(
            Self::build_initial_checkpoint_state(
                Arc::clone(&manifest_store),
                Arc::clone(&table_store),
                &options,
                segment_extractor.as_ref(),
                checkpoint,
                replay_new_wals,
                &db_stats,
            )
            .await?,
        );

        let mono_clock = Arc::new(MonotonicClock::new(
            system_clock.clone(),
            initial_state.core().last_l0_clock_tick,
        ));

        // initial_state contains the last_committed_seq after WAL replay. in no-wal mode, we can simply fallback
        // to last_l0_seq.
        let initial_durable_seq = initial_state
            .last_remote_persisted_seq
            .max(initial_state.core().last_l0_seq);
        let status_manager = DbStatusManager::new_with_initial_values(
            initial_durable_seq,
            VersionedManifest::from(initial_state.as_ref()),
            collect_touched_segments(initial_state.as_ref()),
        );
        let oracle = Arc::new(DbReaderOracle::new(
            initial_durable_seq,
            status_manager.clone(),
        ));

        let state = RwLock::new(initial_state);
        let reader = Reader::new(
            Arc::clone(&table_store),
            db_stats.clone(),
            Arc::clone(&mono_clock),
            oracle.clone(),
            merge_operator,
        );

        let inner = Self {
            manifest_store,
            table_store,
            options,
            state,
            system_clock,
            #[cfg(test)]
            user_checkpoint_id: checkpoint_id,
            oracle,
            reader,
            db_stats,
            status_manager,
            segment_extractor,
            rand,
            recorder,
        };
        Ok(inner)
    }

    async fn get_or_create_checkpoint(
        manifest: &mut StoredManifest,
        checkpoint_id: Option<Uuid>,
        options: &DbReaderOptions,
        rand: Arc<DbRand>,
    ) -> Result<Checkpoint, SlateDBError> {
        let checkpoint = if let Some(checkpoint_id) = checkpoint_id {
            manifest
                .db_state()
                .find_checkpoint(checkpoint_id)
                .ok_or(SlateDBError::CheckpointMissing(checkpoint_id))?
                .clone()
        } else {
            let options = CheckpointOptions {
                lifetime: Some(options.checkpoint_lifetime),
                ..CheckpointOptions::default()
            };
            let checkpoint_id = rand.rng().gen_uuid();
            manifest.write_checkpoint(checkpoint_id, &options).await?
        };
        Ok(checkpoint)
    }

    async fn get_with_options<K: AsRef<[u8]> + Send>(
        &self,
        key: K,
        options: &ReadOptions,
    ) -> Result<Option<Bytes>, SlateDBError> {
        self.get_key_value_with_options(key, options)
            .await
            .map(|kv_opt| kv_opt.map(|kv| kv.value))
    }

    async fn get_key_value_with_options<K: AsRef<[u8]> + Send>(
        &self,
        key: K,
        options: &ReadOptions,
    ) -> Result<Option<KeyValue>, SlateDBError> {
        self.check_closed()?;
        let db_state = Arc::clone(&self.state.read());
        let _permit = db_state.generation.acquire()?;
        self.reader
            .get_key_value_with_options(key, options, db_state.as_ref(), None, None)
            .await
    }

    async fn multi_get_with_options<K: AsRef<[u8]> + Send + Sync>(
        &self,
        keys: &[K],
        options: &ReadOptions,
    ) -> Result<Vec<Option<Bytes>>, SlateDBError> {
        self.multi_get_key_value_with_options(keys, options)
            .await
            .map(|values| values.into_iter().map(|kv| kv.map(|kv| kv.value)).collect())
    }

    async fn multi_get_key_value_with_options<K: AsRef<[u8]> + Send + Sync>(
        &self,
        keys: &[K],
        options: &ReadOptions,
    ) -> Result<Vec<Option<KeyValue>>, SlateDBError> {
        self.check_closed()?;
        let db_state = Arc::clone(&self.state.read());
        let _permit = db_state.generation.acquire()?;
        let keys = keys
            .iter()
            .map(|key| Bytes::copy_from_slice(key.as_ref()))
            .collect::<Vec<_>>();
        self.reader
            .multi_get_key_value_with_options(&keys, options, db_state.as_ref(), None)
            .await
    }

    pub(crate) async fn snapshot_get_key_value_with_options<K: AsRef<[u8]> + Send>(
        &self,
        state: Arc<CheckpointState>,
        max_seq: u64,
        key: K,
        options: &ReadOptions,
    ) -> Result<Option<KeyValue>, SlateDBError> {
        self.check_closed()?;
        let _permit = state.generation.acquire()?;
        self.reader
            .get_key_value_with_options(key, options, state.as_ref(), None, Some(max_seq))
            .await
    }

    pub(crate) async fn snapshot_multi_get_key_value_with_options<K: AsRef<[u8]> + Send + Sync>(
        &self,
        state: Arc<CheckpointState>,
        max_seq: u64,
        keys: &[K],
        options: &ReadOptions,
    ) -> Result<Vec<Option<KeyValue>>, SlateDBError> {
        self.check_closed()?;
        let _permit = state.generation.acquire()?;
        let keys = keys
            .iter()
            .map(|key| Bytes::copy_from_slice(key.as_ref()))
            .collect::<Vec<_>>();
        self.reader
            .multi_get_key_value_with_options(&keys, options, state.as_ref(), Some(max_seq))
            .await
    }

    async fn scan_with_options(
        &self,
        range: BytesRange,
        options: &ScanOptions,
        prefix: Option<Bytes>,
    ) -> Result<DbIterator, SlateDBError> {
        self.check_closed()?;
        let db_state = Arc::clone(&self.state.read());
        let _permit = db_state.generation.acquire()?;
        let iter = self
            .reader
            .scan_with_options(
                range,
                options,
                ScanContext {
                    db_state: db_state.as_ref(),
                    write_batch_iter: None,
                    max_seq: None,
                    prefix,
                },
            )
            .await?;
        Ok(iter.with_iteration_guard(Arc::clone(&db_state.generation)))
    }

    pub(crate) async fn snapshot_scan_with_options(
        &self,
        state: Arc<CheckpointState>,
        max_seq: u64,
        range: BytesRange,
        options: &ScanOptions,
        prefix: Option<Bytes>,
    ) -> Result<DbIterator, SlateDBError> {
        self.check_closed()?;
        let _permit = state.generation.acquire()?;
        let iter = self
            .reader
            .scan_with_options(
                range,
                options,
                ScanContext {
                    db_state: state.as_ref(),
                    write_batch_iter: None,
                    max_seq: Some(max_seq),
                    prefix,
                },
            )
            .await?;
        Ok(iter.with_iteration_guard(Arc::clone(&state.generation)))
    }

    fn should_reestablish_checkpoint(&self, latest: &ManifestCore) -> bool {
        let read_guard = self.state.read();
        let current_state = read_guard.core();
        latest.tree.last_compacted_l0_sst_view_id
            != current_state.tree.last_compacted_l0_sst_view_id
            || latest.last_l0_seq > current_state.last_l0_seq
            || latest.tree.compacted != current_state.tree.compacted
            // RFC-0024: segment-only progress (per-segment compactions,
            // drains, segment-set changes) is invisible to the root-tree
            // diff above. Structural equality on `Segment` covers each
            // segment's tree state, so any change retires the snapshot.
            || latest.segments != current_state.segments
    }

    async fn create_checkpoint(
        &self,
        stored_manifest: &mut StoredManifest,
    ) -> Result<Checkpoint, SlateDBError> {
        let options = CheckpointOptions {
            lifetime: Some(self.options.checkpoint_lifetime),
            ..CheckpointOptions::default()
        };
        let new_checkpoint_id = self.rand.rng().gen_uuid();
        stored_manifest
            .write_checkpoint(new_checkpoint_id, &options)
            .await
    }

    async fn reestablish_checkpoint(&self, checkpoint: Checkpoint) -> Result<(), SlateDBError> {
        let new_checkpoint_state = self.rebuild_checkpoint_state(checkpoint).await?;
        let durable_seq = new_checkpoint_state.last_remote_persisted_seq;
        let versioned_manifest = VersionedManifest::from(&new_checkpoint_state);
        self.oracle.advance_durable_seq(durable_seq);
        let mut write_guard = self.state.write();
        *write_guard = Arc::new(new_checkpoint_state);
        drop(write_guard);
        self.status_manager.report_manifest_and_memtable_segments(
            versioned_manifest,
            collect_touched_segments(self.state.read().as_ref()),
        );
        Ok(())
    }

    async fn maybe_replay_new_wals(&self) -> Result<(), SlateDBError> {
        if self.options.skip_wal_replay {
            return Ok(());
        }
        let current_checkpoint = Arc::clone(&self.state.read());
        let mut imm_memtable = current_checkpoint.imm_memtable.clone();
        let generation = Arc::clone(&current_checkpoint.generation);
        let mut publish =
            |imm_memtable: &ReplayMemtables, last_wal_id: u64, last_committed_seq: u64| {
                self.oracle.advance_durable_seq(last_committed_seq);
                self.db_stats
                    .reader_replay_memtables
                    .set(imm_memtable.len() as i64);
                let mut write_guard = self.state.write();
                *write_guard = Arc::new(CheckpointState {
                    generation: Arc::clone(&generation),
                    imm_memtable: imm_memtable.clone(),
                    last_wal_id,
                    last_remote_persisted_seq: last_committed_seq,
                });
                drop(write_guard);
                self.status_manager
                    .report_memtable_segments(collect_touched_segments(self.state.read().as_ref()));
            };

        Self::replay_wal_into(
            Arc::clone(&self.table_store),
            &self.options,
            current_checkpoint.core(),
            &mut imm_memtable,
            Some((
                current_checkpoint.last_wal_id,
                current_checkpoint.last_remote_persisted_seq,
            )),
            true,
            self.segment_extractor.as_ref(),
            Some(&mut publish),
            Some(&self.db_stats),
        )
        .await?;
        Ok(())
    }

    async fn build_initial_checkpoint_state(
        manifest_store: Arc<ManifestStore>,
        table_store: Arc<TableStore>,
        options: &DbReaderOptions,
        segment_extractor: Option<&Arc<dyn PrefixExtractor>>,
        checkpoint: Checkpoint,
        replay_new_wals: bool,
        db_stats: &DbStats,
    ) -> Result<CheckpointState, SlateDBError> {
        let manifest = manifest_store.read_manifest(checkpoint.manifest_id).await?;
        let imm_memtable = ReplayMemtables::default();
        Self::build_checkpoint_state(
            checkpoint,
            manifest,
            imm_memtable,
            replay_new_wals,
            Arc::clone(&table_store),
            options,
            segment_extractor,
            None,
            db_stats,
        )
        .await
    }

    async fn rebuild_checkpoint_state(
        &self,
        new_checkpoint: Checkpoint,
    ) -> Result<CheckpointState, SlateDBError> {
        let prior = self.state.read().clone();
        let manifest = self
            .manifest_store
            .read_manifest(new_checkpoint.manifest_id)
            .await?;
        let replay_cursor = Some((
            prior.last_wal_id.max(manifest.core.replay_after_wal_id),
            prior
                .last_remote_persisted_seq
                .max(manifest.core.last_l0_seq),
        ));
        let mut retained_memtables = Vec::new();

        for table in prior.imm_memtable.iter() {
            let table_meta = table.table().metadata();
            if table_meta.last_seq <= manifest.core.last_l0_seq {
                // Skip since the entire table is older than L0+.
                continue;
            } else if table_meta.first_seq > manifest.core.last_l0_seq {
                // Keep the entire table since all rows are newer than L0+.
                retained_memtables.push(table);
            } else {
                // The table has some rows that are newer than L0+ and some that are older. This
                // happens when the table spans multiple WAL files. Some of those WAL files can
                // have sequence numbers < manifest.core.last_l0_seq, while others have sequence
                // numbers > manifest.core.last_l0_seq. Retain only those that are more recent
                // than the manifest's last L0 sequence number.
                let filtered_table = table.filter_after_seq(
                    manifest.core.last_l0_seq,
                    self.segment_extractor.as_deref(),
                )?;
                // Push to the back because we are iterating prior from newest to oldest, and we
                // want the imm memtables in checkpoint state to be ordered the same way.
                retained_memtables.push(Arc::new(filtered_table));
            }
        }

        let imm_memtable = retained_memtables.into_iter().collect();
        Self::build_checkpoint_state(
            new_checkpoint,
            manifest,
            imm_memtable,
            !self.options.skip_wal_replay,
            Arc::clone(&self.table_store),
            &self.options,
            self.segment_extractor.as_ref(),
            replay_cursor,
            &self.db_stats,
        )
        .await
    }

    async fn build_checkpoint_state(
        checkpoint: Checkpoint,
        manifest: Manifest,
        mut imm_memtable: ReplayMemtables,
        replay_new_wals: bool,
        table_store: Arc<TableStore>,
        options: &DbReaderOptions,
        segment_extractor: Option<&Arc<dyn PrefixExtractor>>,
        replay_cursor: Option<(u64, u64)>,
        db_stats: &DbStats,
    ) -> Result<CheckpointState, SlateDBError> {
        let (last_wal_id, last_committed_seq) = Self::replay_wal_into(
            Arc::clone(&table_store),
            options,
            &manifest.core,
            &mut imm_memtable,
            replay_cursor,
            replay_new_wals,
            segment_extractor,
            None,
            Some(db_stats),
        )
        .await?;

        db_stats
            .reader_replay_memtables
            .set(imm_memtable.len() as i64);

        Ok(CheckpointState {
            generation: ReaderGeneration::new(checkpoint, manifest),
            imm_memtable,
            last_wal_id,
            last_remote_persisted_seq: last_committed_seq,
        })
    }

    #[cfg(test)]
    async fn maybe_refresh_checkpoint(
        &self,
        stored_manifest: &mut StoredManifest,
    ) -> Result<(), SlateDBError> {
        let generation = Arc::clone(&self.state.read().generation);
        let checkpoint = generation.checkpoint();
        let half_lifetime = self
            .options
            .checkpoint_lifetime
            .checked_div(2)
            .expect("Failed to divide checkpoint lifetime");
        let refresh_deadline = checkpoint
            .expire_time
            .expect("Expected checkpoint expiration time to be set")
            .sub(half_lifetime);
        if self.system_clock.now() > refresh_deadline {
            let refreshed_checkpoint = match stored_manifest
                .refresh_checkpoint(checkpoint.id, self.options.checkpoint_lifetime)
                .await
            {
                Ok(refreshed_checkpoint) => refreshed_checkpoint,
                Err(SlateDBError::CheckpointMissing(id)) if self.user_checkpoint_id.is_none() => {
                    // Our self-established checkpoint lapsed (e.g. a stalled poll tick
                    // outlived the lease during an object-store outage) and the writer's
                    // GC reaped it. Re-establish a fresh checkpoint against the latest
                    // manifest instead of failing the reader permanently. A user-supplied
                    // checkpoint must still fail loud: the caller's pinned view is gone.
                    warn!("reader checkpoint missing, re-establishing [checkpoint_id={id}]");
                    let checkpoint = self.create_checkpoint(stored_manifest).await?;
                    self.reestablish_checkpoint(checkpoint).await?;
                    return Ok(());
                }
                Err(e) => return Err(e),
            };

            // Update our local checkpoint copy so we know the latest expiration time
            // and can calculate future refresh deadlines correctly.
            {
                let mut current_checkpoint = generation.checkpoint.write();
                if current_checkpoint.id == checkpoint.id
                    && current_checkpoint.expire_time == checkpoint.expire_time
                {
                    *current_checkpoint = refreshed_checkpoint.clone();
                }
            }

            info!(
                "refreshed checkpoint [checkpoint_id={}, expire_time={:?}]",
                checkpoint.id, refreshed_checkpoint.expire_time
            )
        }
        Ok(())
    }

    fn spawn_manifest_poller(
        self: &Arc<Self>,
        task_executor: &MessageHandlerExecutor,
    ) -> Result<(), SlateDBError> {
        let poller = ManifestPoller::new(Arc::clone(self));
        let (_tx, rx) = async_channel::unbounded();
        let result = task_executor.add_handler(
            DB_READER_TASK_NAME.to_string(),
            Box::new(poller),
            rx,
            &Handle::current(),
        );
        task_executor.monitor_on(&Handle::current())?;
        result
    }

    async fn replay_wal_into(
        table_store: Arc<TableStore>,
        reader_options: &DbReaderOptions,
        core: &ManifestCore,
        into_tables: &mut ReplayMemtables,
        replay_cursor: Option<(u64, u64)>,
        replay_new_wals: bool,
        segment_extractor: Option<&Arc<dyn PrefixExtractor>>,
        mut publish: Option<ReplayPublisher<'_>>,
        db_stats: Option<&DbStats>,
    ) -> Result<(u64, u64), SlateDBError> {
        let sst_iter_options = SstIteratorOptions {
            max_fetch_tasks: 1,
            blocks_to_fetch: 256,
            cache_blocks: true,
            cache_metadata: false,
            eager_spawn: true,
            order: IterationOrder::Ascending,
            prefix: None,
            filter_context: None,
        };

        let (mut replay_after_wal_id, mut last_committed_seq) =
            replay_cursor.unwrap_or_else(|| {
                if let Some(latest_replayed_table) = into_tables.front() {
                    (
                        latest_replayed_table.recent_flushed_wal_id(),
                        latest_replayed_table.table().last_seq().unwrap_or(0),
                    )
                } else {
                    (core.replay_after_wal_id, core.last_l0_seq)
                }
            });
        let wal_id_end = if replay_new_wals {
            table_store.last_seen_wal_id(replay_after_wal_id).await? + 1
        } else {
            core.next_wal_sst_id
        };

        let replay_options = WalReplayOptions {
            sst_batch_size: 4,
            max_memtable_bytes: reader_options.max_memtable_bytes as usize,
            sst_iter_options,
            // Skip entries that we already have in `imm_memtable` (that might be above last_l0_seq).
            min_seq: Some(last_committed_seq),
        };

        let mut replay_iter = WalReplayIterator::range(
            (replay_after_wal_id + 1)..wal_id_end,
            core,
            replay_options,
            Arc::clone(&table_store),
        )
        .await?;

        while let Some(replayed_table) = match replay_iter.next().await {
            Ok(Some(replayed_table)) => Some(replayed_table),
            Ok(None) => None,
            Err(err) if has_not_found_object_store_error(&err) => None,
            Err(err) => return Err(err),
        } {
            assert!(replayed_table.last_wal_id > replay_after_wal_id);
            let replayed_ssts = replayed_table.last_wal_id - replay_after_wal_id;
            replay_after_wal_id = replayed_table.last_wal_id;
            if let Some(db_stats) = db_stats {
                let metadata = replayed_table.table.metadata();
                db_stats.reader_wal_replay_ssts.increment(replayed_ssts);
                db_stats
                    .reader_wal_replay_bytes
                    .increment(metadata.entries_size_in_bytes as u64);
                db_stats.reader_wal_replay_batches.increment(1);
            }
            if !replayed_table.table.is_empty() && replayed_table.last_seq > last_committed_seq {
                let first_seq = replayed_table
                    .table
                    .table()
                    .first_seq()
                    .expect("expected first_seq on non-empty table");
                // The entire table should be newer than the last committed seq, since we filtered
                // out entries <= last_committed_seq when creating the replay iterator.
                assert!(first_seq > last_committed_seq);
                last_committed_seq = replayed_table.last_seq;
                if let Some(extractor) = segment_extractor {
                    Self::record_replayed_touched_segments(
                        extractor.as_ref(),
                        &replayed_table.table,
                    )?;
                }
                let imm_memtable =
                    ImmutableMemtable::new(replayed_table.table, replayed_table.last_wal_id);
                into_tables.prepend(Arc::new(imm_memtable));
            }
            if let Some(publish) = publish.as_mut() {
                publish(into_tables, replay_after_wal_id, last_committed_seq);
            }
        }

        Ok((replay_after_wal_id, last_committed_seq))
    }

    /// Re-derive each replayed entry's segment prefix (RFC-0024) and record the
    /// table's touched-segment set, mirroring the writer's replay path. Durable
    /// WAL entries were validated when accepted, so the antichain check is not
    /// re-run; an empty/absent prefix under the configured extractor remains a
    /// hard error.
    fn record_replayed_touched_segments(
        extractor: &dyn PrefixExtractor,
        table: &WritableKVTable,
    ) -> Result<(), SlateDBError> {
        let mut touched_segments: BTreeSet<Bytes> = BTreeSet::new();
        let mut iter = table.table().iter();
        while let Some(entry) = iter.next_sync() {
            touched_segments.insert(extract_segment_prefix(extractor, &entry.key)?);
        }
        table.record_touched_segments(touched_segments);
        Ok(())
    }

    /// Returns the latest database status.
    ///
    /// This is a snapshot of the current state and will not update automatically.
    /// Use [`subscribe`](DbReader::subscribe) to receive real-time updates.
    pub(crate) fn status(&self) -> DbStatus {
        self.status_manager.status()
    }

    /// Returns an error if the reader has been closed.
    ///
    /// ## Returns
    /// - `Ok(())` if the reader is still open.
    /// - `Err(SlateDBError::Closed)` if the reader was closed successfully
    ///   (state.result_reader() returns Ok(())).
    /// - `Err(e)` if the reader was closed with an error, where `e` is the error
    ///   (state.result_reader() returns Err(e)).
    pub(crate) fn check_closed(&self) -> Result<(), SlateDBError> {
        let closed_result_reader = self.status_manager.result_reader();
        if let Some(result) = closed_result_reader.read() {
            return match result {
                Ok(()) => Err(SlateDBError::Closed),
                Err(e) => Err(e),
            };
        }
        Ok(())
    }
}

struct ManifestPoller {
    inner: Arc<DbReaderInner>,
    generations: HashMap<Uuid, Weak<ReaderGeneration>>,
}

impl ManifestPoller {
    fn new(inner: Arc<DbReaderInner>) -> Self {
        let generation = Arc::clone(&inner.state.read().generation);
        let checkpoint_id = generation.checkpoint().id;
        let poller = Self {
            inner,
            generations: HashMap::from([(checkpoint_id, Arc::downgrade(&generation))]),
        };
        poller.report_active_checkpoints();
        poller
    }

    fn report_active_checkpoints(&self) {
        self.inner
            .db_stats
            .reader_active_checkpoints
            .set(self.generations.len() as i64);
    }

    fn register_current_generation(&mut self) {
        let generation = Arc::clone(&self.inner.state.read().generation);
        self.generations
            .insert(generation.checkpoint().id, Arc::downgrade(&generation));
        self.report_active_checkpoints();
    }

    async fn delete_released_checkpoints(
        &mut self,
        manifest: &mut StoredManifest,
    ) -> Result<(), SlateDBError> {
        let released = self
            .generations
            .iter()
            .filter_map(|(id, generation)| generation.upgrade().is_none().then_some(*id))
            .collect::<Vec<_>>();
        if released.is_empty() {
            return Ok(());
        }
        manifest.delete_checkpoints(&released).await?;
        for id in released {
            self.generations.remove(&id);
        }
        self.report_active_checkpoints();
        Ok(())
    }

    async fn refresh_live_checkpoints(
        &mut self,
        manifest: &mut StoredManifest,
    ) -> Result<(), SlateDBError> {
        let half_lifetime = self
            .inner
            .options
            .checkpoint_lifetime
            .checked_div(2)
            .expect("checkpoint lifetime division failed");
        loop {
            let live = self
                .generations
                .iter()
                .filter_map(|(id, generation)| {
                    generation.upgrade().map(|generation| (*id, generation))
                })
                .collect::<Vec<_>>();
            let now = self.inner.system_clock.now();
            let refresh_due = live.iter().any(|(_, generation)| {
                generation
                    .checkpoint()
                    .expire_time
                    .is_some_and(|expiry| now > expiry.sub(half_lifetime))
            });
            if !refresh_due {
                return Ok(());
            }

            let ids = live.iter().map(|(id, _)| *id).collect::<Vec<_>>();
            match manifest
                .refresh_checkpoints(&ids, self.inner.options.checkpoint_lifetime)
                .await
            {
                Ok(refreshed) => {
                    for checkpoint in refreshed {
                        if let Some(generation) =
                            self.generations.get(&checkpoint.id).and_then(Weak::upgrade)
                        {
                            *generation.checkpoint.write() = checkpoint;
                        }
                    }
                    return Ok(());
                }
                Err(SlateDBError::CheckpointMissing(id)) => {
                    warn!("reader checkpoint lease lost [checkpoint_id={id}]");
                    if let Some(generation) = self.generations.remove(&id).and_then(|g| g.upgrade())
                    {
                        generation.invalidate();
                        generation.drain().await;
                    }
                    self.report_active_checkpoints();

                    let current_id = self.inner.state.read().generation.checkpoint().id;
                    if current_id == id {
                        let checkpoint = self.inner.create_checkpoint(manifest).await?;
                        self.inner.reestablish_checkpoint(checkpoint).await?;
                        self.register_current_generation();
                    }
                    // Other live generations may be due at the same time. Retry
                    // the batch immediately after removing the missing lease so
                    // one lost checkpoint cannot make their refresh a poll late.
                }
                Err(err) => return Err(err),
            }
        }
    }
}

impl Drop for ManifestPoller {
    fn drop(&mut self) {
        // The gauge tracks only GC checkpoints actively managed by this
        // poller. Reset it even if startup, cleanup, or the poller task fails.
        self.inner.db_stats.reader_active_checkpoints.set(0);
    }
}

#[async_trait]
impl MessageHandler<DbReaderMessage> for ManifestPoller {
    fn tickers(&mut self) -> Vec<MessageTickerDef<DbReaderMessage>> {
        vec![MessageTickerDef::new(
            self.inner.options.manifest_poll_interval,
            Box::new(|| DbReaderMessage::PollManifest),
        )]
    }

    async fn handle(&mut self, message: DbReaderMessage) -> Result<(), SlateDBError> {
        assert!(matches!(message, DbReaderMessage::PollManifest));
        let mut manifest = StoredManifest::load(
            Arc::clone(&self.inner.manifest_store),
            self.inner.system_clock.clone(),
        )
        .await?;

        self.delete_released_checkpoints(&mut manifest).await?;

        let latest_manifest = manifest.manifest();
        if self
            .inner
            .should_reestablish_checkpoint(&latest_manifest.core)
        {
            let checkpoint = self.inner.create_checkpoint(&mut manifest).await?;
            self.inner.reestablish_checkpoint(checkpoint).await?;
            self.register_current_generation();
        } else {
            self.inner.maybe_replay_new_wals().await?;
        }

        self.refresh_live_checkpoints(&mut manifest).await?;
        self.inner.db_stats.reader_manifest_polls.increment(1);
        Ok(())
    }

    async fn cleanup(
        &mut self,
        _messages: BoxStream<'async_trait, DbReaderMessage>,
        _result: Result<(), SlateDBError>,
    ) -> Result<(), SlateDBError> {
        let mut manifest = StoredManifest::load(
            Arc::clone(&self.inner.manifest_store),
            self.inner.system_clock.clone(),
        )
        .await?;
        let checkpoint_ids = self.generations.keys().copied().collect::<Vec<_>>();
        if !checkpoint_ids.is_empty() {
            let live_generations = self
                .generations
                .values()
                .filter_map(Weak::upgrade)
                .collect::<Vec<_>>();
            // Invalidate every generation before waiting on any one of them,
            // otherwise operations could continue entering a later gate while
            // shutdown is draining an earlier one.
            for generation in &live_generations {
                generation.invalidate();
            }
            for generation in live_generations {
                generation.drain().await;
            }
            info!(
                "deleting reader established checkpoints for shutdown [checkpoint_ids={:?}]",
                checkpoint_ids
            );
            manifest.delete_checkpoints(&checkpoint_ids).await?;
        }
        self.inner.db_stats.reader_active_checkpoints.set(0);
        Ok(())
    }
}

impl DbReader {
    fn validate_options(options: &DbReaderOptions) -> Result<(), SlateDBError> {
        if options.checkpoint_lifetime.as_millis() < 1000 {
            return Err(SlateDBError::InvalidCheckpointLifetime(
                options.checkpoint_lifetime,
            ));
        }

        let double_poll_interval = options.manifest_poll_interval.checked_mul(2).ok_or(
            SlateDBError::InvalidManifestPollInterval(options.manifest_poll_interval),
        )?;
        if options.checkpoint_lifetime < double_poll_interval {
            return Err(SlateDBError::CheckpointLifetimeTooShort {
                lifetime: options.checkpoint_lifetime,
                interval: double_poll_interval,
            });
        }
        Ok(())
    }

    /// Preload the disk cache from the current manifest state.
    pub(crate) async fn preload_cache(
        &self,
        cached_obj_store: &CachedObjectStore,
        path: object_store::path::Path,
    ) -> Result<(), SlateDBError> {
        let state = Arc::clone(&self.inner.state.read());
        let external_ssts = state.generation.manifest.external_ssts();
        let path_resolver = PathResolver::new_with_external_ssts(path, external_ssts);
        let cache_opts = &self.inner.options.object_store_cache_options;
        crate::utils::preload_cache_from_manifest(
            &state.generation.manifest.core,
            cached_obj_store,
            &path_resolver,
            cache_opts.preload_disk_cache_on_startup,
            cache_opts.max_cache_size_bytes.unwrap_or(usize::MAX),
        )
        .await
    }

    /// Creates a database reader that can read the contents of a database (but cannot write any
    /// data). The caller can provide an optional checkpoint. If the checkpoint is provided, the
    /// reader will read using the specified checkpoint and will not periodically refresh the
    /// checkpoint. Otherwise, the reader creates a new checkpoint pointing to the current manifest
    /// and refreshes it periodically as specified in the options. Each manifest generation keeps
    /// its checkpoint until all snapshots, iterators, and in-flight reads using that generation are
    /// gone. This mode therefore requires permission to append manifest objects even though it
    /// cannot write user data.
    pub async fn open<P: Into<Path>>(
        path: P,
        object_store: Arc<dyn ObjectStore>,
        checkpoint_id: Option<Uuid>,
        options: DbReaderOptions,
    ) -> Result<Self, crate::Error> {
        // Use the builder API internally
        let mut builder = Self::builder(path, object_store).with_options(options);
        if let Some(id) = checkpoint_id {
            builder = builder.with_checkpoint_id(id);
        }
        builder.build().await
    }

    /// Captures the reader's latest fully applied state as a read-only
    /// snapshot-isolation transaction.
    ///
    /// This is an O(1) local operation: WAL discovery and replay remain the
    /// responsibility of the long-lived reader poller. Creating a snapshot
    /// never performs object-store I/O or forces a flush. Snapshots created
    /// from the same manifest generation share one GC checkpoint.
    pub async fn snapshot(&self) -> Result<Arc<DbSnapshot>, crate::Error> {
        loop {
            self.inner.check_closed()?;
            let state = Arc::clone(&self.inner.state.read());
            match state.generation.acquire() {
                Ok(_permit) => {
                    // Keep the generation gate held until the snapshot has
                    // captured the state. Lease-loss recovery cannot
                    // invalidate this generation between validation and
                    // construction.
                    return Ok(DbSnapshot::new_reader(Arc::clone(&self.inner), state));
                }
                Err(err @ SlateDBError::CheckpointLeaseLost(_)) => {
                    // Recovery may have replaced the invalid generation while
                    // we were waiting for its gate. Retry only in that case;
                    // otherwise report the lease loss instead of returning a
                    // snapshot whose every read is guaranteed to fail.
                    let current = Arc::clone(&self.inner.state.read());
                    if !Arc::ptr_eq(&state, &current) {
                        continue;
                    }
                    return Err(err.into());
                }
                Err(err) => return Err(err.into()),
            }
        }
    }

    /// Creates a new builder for a database reader at the given path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the database.
    /// * `object_store` - The object store to use.
    ///
    /// # Returns
    ///
    /// A `DbReaderBuilder` that can be used to configure and build a `DbReader`.
    ///
    /// # Examples
    ///
    /// ```
    /// use slatedb::{Db, DbReader, Error};
    /// use slatedb::object_store::{ObjectStore, memory::InMemory};
    /// use std::sync::Arc;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Error> {
    ///     let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    ///     // First create a database
    ///     let db = Db::open("test_db", Arc::clone(&object_store)).await?;
    ///     db.close().await?;
    ///     // Then open a reader
    ///     let reader = DbReader::builder("test_db", object_store)
    ///         .build()
    ///         .await?;
    ///     Ok(())
    /// }
    /// ```
    pub fn builder<P: Into<Path>>(
        path: P,
        object_store: Arc<dyn ObjectStore>,
    ) -> crate::db::builder::DbReaderBuilder<P> {
        crate::db::builder::DbReaderBuilder::new(path, object_store)
    }

    pub(crate) async fn open_internal(
        manifest_store: Arc<ManifestStore>,
        table_store: Arc<TableStore>,
        checkpoint_id: Option<Uuid>,
        merge_operator: Option<MergeOperatorType>,
        segment_extractor: Option<Arc<dyn PrefixExtractor>>,
        options: DbReaderOptions,
        system_clock: Arc<dyn SystemClock>,
        rand: Arc<DbRand>,
        recorder: slatedb_common::metrics::MetricsRecorderHelper,
    ) -> Result<Self, SlateDBError> {
        Self::validate_options(&options)?;

        let manifest =
            match StoredManifest::load(Arc::clone(&manifest_store), system_clock.clone()).await {
                Ok(manifest) => manifest,
                Err(SlateDBError::LatestTransactionalObjectVersionMissing) => {
                    return Err(SlateDBError::DatabaseMissing);
                }
                Err(error) => return Err(error),
            };
        if !manifest.db_state().initialized {
            return Err(SlateDBError::InvalidDBState);
        }

        manifest
            .db_state()
            .validate_extractor_configuration(segment_extractor.as_deref())?;

        let inner = Arc::new(
            DbReaderInner::new(
                manifest_store,
                table_store,
                options,
                checkpoint_id,
                merge_operator,
                segment_extractor,
                system_clock.clone(),
                rand,
                recorder,
                manifest,
            )
            .await?,
        );
        let task_executor = MessageHandlerExecutor::new(
            Arc::new(inner.status_manager.clone()),
            system_clock.clone(),
        );

        // If no checkpoint was provided, then we have established a new checkpoint
        // from the latest state, and we need to refresh it according to the params
        // of `DbReaderOptions`.
        if checkpoint_id.is_none() {
            inner.spawn_manifest_poller(&task_executor)?;
        }

        Ok(Self {
            inner,
            task_executor,
            object_store_cache: None,
        })
    }

    /// Returns a synchronous point-in-time cache accounting snapshot.
    ///
    /// This method reads only in-memory counters and never performs object-store
    /// or filesystem I/O.
    pub fn cache_usage_snapshot(&self) -> crate::db_cache::SlateDbCacheUsageSnapshot {
        let db_cache = self.inner.table_store.cache().map_or_else(
            || crate::db_cache::DbCacheUsageSnapshot {
                memory: crate::db_cache::CacheUsageSnapshot::Disabled,
                disk: crate::db_cache::CacheUsageSnapshot::Disabled,
            },
            |cache| cache.usage_snapshot(),
        );
        let object_store = self
            .object_store_cache
            .as_ref()
            .map_or(crate::db_cache::CacheUsageSnapshot::Disabled, |cache| {
                cache.usage_snapshot()
            });
        crate::db_cache::SlateDbCacheUsageSnapshot {
            db_cache,
            object_store,
        }
    }

    /// Get a value from the database with default read options.
    ///
    /// The `Bytes` object returned contains a slice of an entire
    /// 4 KiB block. The block will be held in memory as long as the
    /// caller holds a reference to the `Bytes` object. Consider
    /// copying the data if you need to hold it for a long time.
    ///
    /// ## Arguments
    /// - `key`: the key to get
    ///
    /// ## Returns
    /// - `Result<Option<Bytes>, Error>`:
    ///     - `Some(Bytes)`: the value if it exists
    ///     - `None`: if the value does not exist
    ///
    /// ## Errors
    /// - `Error`: if there was an error getting the value
    ///
    /// ## Examples
    ///
    /// ```
    /// use slatedb::{Db, DbReader, config::DbReaderOptions, Error};
    /// use slatedb::object_store::{ObjectStore, memory::InMemory};
    /// use std::sync::Arc;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Error> {
    ///     let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    ///     let db = Db::open("test_db", Arc::clone(&object_store)).await?;
    ///     db.put(b"key", b"value").await?;
    ///     db.flush().await?;
    ///
    ///     let reader = DbReader::open(
    ///       "test_db",
    ///       Arc::clone(&object_store),
    ///       None,
    ///       DbReaderOptions::default(),
    ///     ).await?;
    ///     assert_eq!(reader.get(b"key").await?, Some("value".into()));
    ///     Ok(())
    /// }
    /// ```
    pub async fn get<K: AsRef<[u8]> + Send>(&self, key: K) -> Result<Option<Bytes>, crate::Error> {
        self.get_with_options(key, &ReadOptions::default()).await
    }

    /// Get a value from the database with custom read options.
    ///
    /// The `Bytes` object returned contains a slice of an entire
    /// 4 KiB block. The block will be held in memory as long as the
    /// caller holds a reference to the `Bytes` object. Consider
    /// copying the data if you need to hold it for a long time.
    ///
    /// ## Arguments
    /// - `key`: the key to get
    /// - `options`: the read options to use (Note that [`ReadOptions::read_level`] has no effect
    ///   for readers, which can only observe committed state).
    ///
    /// ## Returns
    /// - `Result<Option<Bytes>, Error>`:
    ///     - `Some(Bytes)`: the value if it exists
    ///     - `None`: if the value does not exist
    ///
    /// ## Errors
    /// - `Error`: if there was an error getting the value
    ///
    /// ## Examples
    ///
    /// ```
    /// use slatedb::{Db, DbReader, config::DbReaderOptions, config::ReadOptions, Error};
    /// use slatedb::object_store::{ObjectStore, memory::InMemory};
    /// use std::sync::Arc;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Error> {
    ///     let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    ///     let db = Db::open("test_db", Arc::clone(&object_store)).await?;
    ///     db.put(b"key", b"value").await?;
    ///     db.flush().await?;
    ///
    ///     let reader = DbReader::open(
    ///       "test_db",
    ///       Arc::clone(&object_store),
    ///       None,
    ///       DbReaderOptions::default(),
    ///     ).await?;
    ///     assert_eq!(db.get_with_options(b"key", &ReadOptions::default()).await?, Some("value".into()));
    ///     Ok(())
    /// }
    /// ```
    pub async fn get_with_options<K: AsRef<[u8]> + Send>(
        &self,
        key: K,
        options: &ReadOptions,
    ) -> Result<Option<Bytes>, crate::Error> {
        self.inner
            .get_with_options(key, options)
            .await
            .map_err(Into::into)
    }

    /// Get a key-value pair from the reader with default read options.
    pub async fn get_key_value<K: AsRef<[u8]> + Send>(
        &self,
        key: K,
    ) -> Result<Option<KeyValue>, crate::Error> {
        self.get_key_value_with_options(key, &ReadOptions::default())
            .await
    }

    /// Get a key-value pair from the reader with custom read options.
    pub async fn get_key_value_with_options<K: AsRef<[u8]> + Send>(
        &self,
        key: K,
        options: &ReadOptions,
    ) -> Result<Option<KeyValue>, crate::Error> {
        let kv = self
            .inner
            .get_key_value_with_options(key, options)
            .await
            .map_err(crate::Error::from)?;
        Ok(kv)
    }

    /// Get multiple values from the reader with default read options.
    ///
    /// The returned vector preserves input order and duplicates.
    pub async fn multi_get<K: AsRef<[u8]> + Send + Sync>(
        &self,
        keys: &[K],
    ) -> Result<Vec<Option<Bytes>>, crate::Error> {
        self.multi_get_with_options(keys, &ReadOptions::default())
            .await
    }

    /// Get multiple values from the reader with custom read options.
    ///
    /// The returned vector preserves input order and duplicates.
    pub async fn multi_get_with_options<K: AsRef<[u8]> + Send + Sync>(
        &self,
        keys: &[K],
        options: &ReadOptions,
    ) -> Result<Vec<Option<Bytes>>, crate::Error> {
        self.inner
            .multi_get_with_options(keys, options)
            .await
            .map_err(Into::into)
    }

    /// Scan a range of keys using the default scan options.
    ///
    /// returns a `DbIterator`
    ///
    /// ## Arguments
    /// - `range`: the range of keys to scan
    ///
    /// ## Errors
    /// - `Error`: if there was an error scanning the range of keys
    ///
    /// ## Returns
    /// - `Result<DbIterator, Error>`: An iterator with the results of the scan
    ///
    /// ## Examples
    ///
    /// ```
    /// use slatedb::{Db, DbReader, config::DbReaderOptions, Error};
    /// use slatedb::object_store::{ObjectStore, memory::InMemory};
    /// use std::sync::Arc;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Error> {
    ///     let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    ///     let db = Db::open("test_db", Arc::clone(&object_store)).await?;
    ///     db.put(b"a", b"a_value").await?;
    ///     db.put(b"b", b"b_value").await?;
    ///     db.flush().await?;
    ///
    ///     let reader = DbReader::open(
    ///       "test_db",
    ///       Arc::clone(&object_store),
    ///       None,
    ///       DbReaderOptions::default(),
    ///     ).await?;
    ///     let mut iter = reader.scan("a".."b").await?;
    ///     let kv = iter.next().await?.unwrap();
    ///     assert_eq!(kv.key.as_ref(), b"a");
    ///     assert_eq!(kv.value.as_ref(), b"a_value");
    ///     assert_eq!(None, iter.next().await?);
    ///     Ok(())
    /// }
    /// ```
    pub async fn scan<T>(&self, range: T) -> Result<DbIterator, crate::Error>
    where
        T: ByteRangeBounds + Send,
    {
        self.scan_with_options(range, &ScanOptions::default()).await
    }

    /// Scan a range of keys with the provided options.
    ///
    /// returns a `DbIterator`
    ///
    /// ## Arguments
    /// - `range`: the range of keys to scan
    /// - `options`: the read options to use (Note that [`ReadOptions::read_level`] has no effect
    ///   for readers, which can only observe committed state).
    ///
    /// ## Errors
    /// - `Error`: if there was an error scanning the range of keys
    ///
    /// ## Returns
    /// - `Result<DbIterator, Error>`: An iterator with the results of the scan
    ///
    /// ## Examples
    ///
    /// ```
    /// use slatedb::{Db, DbReader, config::DbReaderOptions, config::ScanOptions, config::DurabilityLevel, Error};
    /// use slatedb::object_store::{ObjectStore, memory::InMemory};
    /// use std::sync::Arc;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Error> {
    ///     let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    ///     let db = Db::open("test_db", Arc::clone(&object_store)).await?;
    ///     db.put(b"a", b"a_value").await?;
    ///     db.put(b"b", b"b_value").await?;
    ///     db.flush().await?;
    ///
    ///     let reader = DbReader::open(
    ///       "test_db",
    ///       Arc::clone(&object_store),
    ///       None,
    ///       DbReaderOptions::default(),
    ///     ).await?;
    ///     let mut iter = reader.scan_with_options("a".."b", &ScanOptions {
    ///         read_ahead_bytes: 1024 * 1024,
    ///         ..ScanOptions::default()
    ///     }).await?;
    ///     let kv = iter.next().await?.unwrap();
    ///     assert_eq!(kv.key.as_ref(), b"a");
    ///     assert_eq!(kv.value.as_ref(), b"a_value");
    ///     assert_eq!(None, iter.next().await?);
    ///     Ok(())
    /// }
    pub async fn scan_with_options<T>(
        &self,
        range: T,
        options: &ScanOptions,
    ) -> Result<DbIterator, crate::Error>
    where
        T: ByteRangeBounds + Send,
    {
        let start = range.start_bound().map(Bytes::copy_from_slice);
        let end = range.end_bound().map(Bytes::copy_from_slice);
        let range = BytesRange::from((start, end));
        self.inner
            .scan_with_options(range, options, None)
            .await
            .map_err(Into::into)
    }

    /// Scan keys that share the provided prefix, restricted to `subrange`,
    /// using the default scan options.
    ///
    /// The subrange bounds are key *suffixes* interpreted relative to the
    /// prefix: a bound `s` selects the full key `prefix ++ s`. Pass `..` to
    /// scan the prefix's entire keyspace.
    ///
    /// ## Arguments
    /// - `prefix`: the key prefix to scan
    /// - `subrange`: the range of key suffixes (relative to `prefix`) to
    ///   scan; `..` scans all keys with the prefix
    ///
    /// ## Returns
    /// - `Result<DbIterator, Error>`: An iterator with the results of the scan
    pub async fn scan_prefix<P, T>(
        &self,
        prefix: P,
        subrange: T,
    ) -> Result<DbIterator, crate::Error>
    where
        P: AsRef<[u8]> + Send,
        T: ByteRangeBounds + Send,
    {
        self.scan_prefix_with_options(prefix, subrange, &ScanOptions::default())
            .await
    }

    /// Scan keys that share the provided prefix, restricted to `subrange`,
    /// with custom options. See [`Self::scan_prefix`] for the subrange
    /// semantics.
    ///
    /// ## Arguments
    /// - `prefix`: the key prefix to scan
    /// - `subrange`: the range of key suffixes (relative to `prefix`) to
    ///   scan; `..` scans all keys with the prefix
    /// - `options`: the scan options to use
    ///
    /// ## Returns
    /// - `Result<DbIterator, Error>`: An iterator with the results of the scan
    pub async fn scan_prefix_with_options<P, T>(
        &self,
        prefix: P,
        subrange: T,
        options: &ScanOptions,
    ) -> Result<DbIterator, crate::Error>
    where
        P: AsRef<[u8]> + Send,
        T: ByteRangeBounds + Send,
    {
        let prefix = Bytes::copy_from_slice(prefix.as_ref());
        let range = BytesRange::from_prefix_and_subrange(prefix.as_ref(), subrange);
        self.inner
            .scan_with_options(range, options, Some(prefix))
            .await
            .map_err(Into::into)
    }

    /// Close the database reader.
    ///
    /// ## Returns
    /// - `Result<(), Error>`: if there was an error closing the reader
    ///
    /// ## Examples
    ///
    /// ```
    /// use slatedb::{Db, DbReader, config::DbReaderOptions, Error};
    /// use slatedb::object_store::{ObjectStore, memory::InMemory};
    /// use std::sync::Arc;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Error> {
    ///     let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    ///     let db = Db::open("test_db", object_store.clone()).await?;
    ///     let options = DbReaderOptions::default();
    ///     let reader = DbReader::open("test_db", object_store.clone(), None, options).await?;
    ///     reader.close().await?;
    ///     Ok(())
    /// }
    /// ```
    ///
    pub async fn close(&self) -> Result<(), crate::Error> {
        // Fixed-checkpoint readers do not have a manifest poller to publish
        // their clean shutdown, so close the shared status explicitly for
        // both reader modes before shutting down any managed task.
        self.inner.status_manager.write_result(Ok(()));

        self.task_executor
            .shutdown_task(DB_READER_TASK_NAME)
            .await
            .map_err(Into::<crate::Error>::into)?;

        if let Err(e) = self.inner.table_store.close_cache().await {
            warn!("failed to close block cache [error={:?}]", e);
        }

        Ok(())
    }
}

#[async_trait::async_trait]
impl DbReadOps for DbReader {
    async fn get_with_options<K: AsRef<[u8]> + Send>(
        &self,
        key: K,
        options: &ReadOptions,
    ) -> Result<Option<Bytes>, crate::Error> {
        DbReader::get_with_options(self, key, options).await
    }

    async fn get_key_value_with_options<K: AsRef<[u8]> + Send>(
        &self,
        key: K,
        options: &ReadOptions,
    ) -> Result<Option<KeyValue>, crate::Error> {
        DbReader::get_key_value_with_options(self, key, options).await
    }

    async fn multi_get_with_options<K>(
        &self,
        keys: &[K],
        options: &ReadOptions,
    ) -> Result<Vec<Option<Bytes>>, crate::Error>
    where
        K: AsRef<[u8]> + Send + Sync,
    {
        DbReader::multi_get_with_options(self, keys, options).await
    }

    async fn scan_with_options<T>(
        &self,
        range: T,
        options: &ScanOptions,
    ) -> Result<DbIterator, crate::Error>
    where
        T: ByteRangeBounds + Send,
    {
        DbReader::scan_with_options(self, range, options).await
    }

    async fn scan_prefix_with_options<P, T>(
        &self,
        prefix: P,
        subrange: T,
        options: &ScanOptions,
    ) -> Result<DbIterator, crate::Error>
    where
        P: AsRef<[u8]> + Send,
        T: ByteRangeBounds + Send,
    {
        DbReader::scan_prefix_with_options(self, prefix, subrange, options).await
    }
}

impl DbMetadataOps for DbReader {
    fn manifest(&self) -> VersionedManifest {
        let state = Arc::clone(&self.inner.state.read());
        VersionedManifest::from(state.as_ref())
    }

    fn subscribe(&self) -> tokio::sync::watch::Receiver<DbStatus> {
        self.inner.status_manager.subscribe()
    }

    fn status(&self) -> DbStatus {
        self.inner.status()
    }
}

impl DbReader {
    /// See [`DbMetadataOps::manifest`].
    pub fn manifest(&self) -> VersionedManifest {
        <Self as DbMetadataOps>::manifest(self)
    }

    /// See [`DbMetadataOps::subscribe`].
    pub fn subscribe(&self) -> tokio::sync::watch::Receiver<DbStatus> {
        <Self as DbMetadataOps>::subscribe(self)
    }

    /// See [`DbMetadataOps::status`].
    pub fn status(&self) -> DbStatus {
        <Self as DbMetadataOps>::status(self)
    }
}

#[async_trait]
impl DbCacheManagerOps for DbReader {
    async fn warm_sst(
        &self,
        sst_id: SsTableId,
        targets: &[CacheTarget],
    ) -> Result<(), crate::Error> {
        self.inner.check_closed()?;
        let manifest = self.manifest();
        db_cache_manager::warm_sst_impl(&self.inner.table_store, &manifest, sst_id, targets).await
    }

    async fn evict_cached_sst(&self, sst_id: SsTableId) -> Result<(), crate::Error> {
        self.inner.check_closed()?;
        db_cache_manager::evict_cached_sst_impl(&self.inner.table_store, sst_id).await
    }
}

/// Checks if the error or any of its sources is an `object_store::Error::NotFound` error.
fn has_not_found_object_store_error(err: &(dyn std::error::Error + 'static)) -> bool {
    let mut current = Some(err);
    while let Some(current_err) = current {
        if current_err
            .downcast_ref::<object_store::Error>()
            .is_some_and(|err| matches!(err, object_store::Error::NotFound { .. }))
            || current_err
                .downcast_ref::<Arc<object_store::Error>>()
                .is_some_and(|err| matches!(err.as_ref(), object_store::Error::NotFound { .. }))
        {
            return true;
        }
        current = current_err.source();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::{CheckpointState, ReaderGeneration, ReplayMemtables};
    use crate::clock::MonotonicClock;
    use crate::config::{
        CheckpointOptions, CheckpointScope, FlushOptions, FlushType, MergeOptions, PutOptions,
        Settings, WriteOptions,
    };
    use crate::db_cache::test_utils::TestCache;
    use crate::db_cache::DbCache;
    use crate::db_reader::{DbReader, DbReaderInner, DbReaderOptions};
    use crate::db_state::SsTableId;
    use crate::db_stats::DbStats;
    use crate::db_status::DbStatusManager;
    use crate::format::sst::SsTableFormat;
    use crate::iter::IterationOrder;
    use crate::manifest::store::{ManifestStore, StoredManifest};
    use crate::manifest::{Manifest, ManifestCore, VersionedManifest};
    use crate::mem_table::{ImmutableMemtable, WritableKVTable};
    use crate::merge_operator::MergeOperatorType;
    use crate::object_stores::ObjectStores;
    use crate::oracle::DbReaderOracle;
    use crate::paths::PathResolver;
    use crate::proptest_util::rng::new_test_rng;
    use crate::proptest_util::sample;
    use crate::reader::Reader;
    use crate::tablestore::{TableStore, TableStoreKind};
    use crate::types::RowEntry;
    use crate::{error::SlateDBError, test_utils, CloseReason, Db};
    use bytes::Bytes;
    use fail_parallel::FailPointRegistry;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use object_store::ObjectStore;
    use rstest::rstest;
    use slatedb_common::clock::{DefaultSystemClock, SystemClock};
    use slatedb_common::DbRand;
    use slatedb_common::MockSystemClock;
    use std::collections::{BTreeMap, VecDeque};
    use std::sync::Arc;
    use std::time::Duration;
    use uuid::Uuid;

    async fn wait_for_reader_generation_change(reader: &DbReader, previous: Uuid) {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if reader.inner.state.read().generation.checkpoint().id != previous {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("reader did not install a new manifest generation");
    }

    #[tokio::test]
    async fn should_get_value_from_db() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_kv_store");
        let test_provider = TestProvider::new(path.clone(), Arc::clone(&object_store));

        let db = test_provider.new_db(Settings::default()).await.unwrap();
        let key = b"test_key";
        let value = b"test_value";

        db.put(key, value).await.unwrap();
        db.flush().await.unwrap();

        let reader = DbReader::open(
            path.clone(),
            Arc::clone(&object_store),
            None,
            DbReaderOptions::default(),
        )
        .await
        .unwrap();

        assert_eq!(
            reader.get(key).await.unwrap(),
            Some(Bytes::from_static(value))
        );
    }

    #[tokio::test]
    async fn empty_database_reader_returns_typed_database_missing() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let error = match DbReader::open(
            "/tmp/test_reader_database_missing",
            object_store,
            None,
            DbReaderOptions::default(),
        )
        .await
        {
            Ok(_) => panic!("empty object store must not open a reader"),
            Err(error) => error,
        };

        assert_eq!(error.kind(), crate::ErrorKind::Data);
        assert_eq!(error.code(), Some(crate::ErrorCode::DatabaseMissing));
    }

    #[tokio::test]
    async fn should_return_current_versioned_manifest() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_reader_manifest_accessor");
        let test_provider = TestProvider::new(path.clone(), Arc::clone(&object_store));

        let db = test_provider.new_db(Settings::default()).await.unwrap();
        db.put(b"test_key", b"test_value").await.unwrap();
        db.flush().await.unwrap();

        let reader = DbReader::open(path, object_store, None, DbReaderOptions::default())
            .await
            .unwrap();

        let manifest = reader.manifest();
        let expected: VersionedManifest =
            VersionedManifest::from(reader.inner.state.read().as_ref());
        assert_eq!(manifest, expected);
    }

    #[tokio::test]
    async fn should_get_latest_value_from_checkpoint() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_kv_store");
        let test_provider = TestProvider::new(path.clone(), Arc::clone(&object_store));

        let db = test_provider.new_db(Settings::default()).await.unwrap();
        let key = b"test_key";
        let value1 = b"test_value";
        let value2 = b"updated_value";

        db.put(key, value1).await.unwrap();
        db.flush().await.unwrap();
        db.put(key, value2).await.unwrap();
        let checkpoint_result = db
            .create_checkpoint(CheckpointScope::All, &CheckpointOptions::default())
            .await
            .unwrap();

        let reader = DbReader::open_internal(
            test_provider.manifest_store(),
            test_provider.table_store(),
            Some(checkpoint_result.id),
            None,
            None,
            DbReaderOptions::default(),
            test_provider.system_clock.clone(),
            test_provider.rand.clone(),
            slatedb_common::metrics::MetricsRecorderHelper::noop(),
        )
        .await
        .unwrap();

        assert_eq!(
            reader.get(key).await.unwrap(),
            Some(Bytes::from_static(value2))
        );
    }

    #[tokio::test]
    async fn should_get_from_checkpoint() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_kv_store");
        let test_provider = TestProvider::new(path.clone(), Arc::clone(&object_store));

        let db = test_provider.new_db(Settings::default()).await.unwrap();
        let key = b"test_key";
        let checkpoint_value = b"test_value";
        let updated_value = b"updated_value";

        db.put(key, checkpoint_value).await.unwrap();
        let checkpoint_result = db
            .create_checkpoint(CheckpointScope::All, &CheckpointOptions::default())
            .await
            .unwrap();
        db.put(key, updated_value).await.unwrap();

        let reader = DbReader::open(
            path.clone(),
            Arc::clone(&object_store),
            Some(checkpoint_result.id),
            DbReaderOptions::default(),
        )
        .await
        .unwrap();

        assert_eq!(
            reader.get(key).await.unwrap(),
            Some(Bytes::from_static(checkpoint_value))
        );
    }

    #[tokio::test]
    async fn should_report_memtable_segments_in_status() {
        // given
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = "/tmp/test_reader_subscribe_reports_memtable_segments";
        let db = Db::builder(path, object_store.clone())
            .with_settings(Settings::default())
            .with_segment_extractor(Arc::new(test_utils::FixedThreeBytePrefixExtractor))
            .build()
            .await
            .unwrap();
        let write_opts = WriteOptions {
            await_durable: false,
            ..Default::default()
        };
        db.put_with_options(b"abc-1", b"v1", &PutOptions::default(), &write_opts)
            .await
            .unwrap();
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::Wal,
        })
        .await
        .unwrap();

        // when
        let reader = DbReader::builder(path, object_store.clone())
            .with_segment_extractor(Arc::new(test_utils::FixedThreeBytePrefixExtractor))
            .build()
            .await
            .unwrap();

        // then
        let prefixes: Vec<Bytes> = reader
            .status()
            .list_segments()
            .into_iter()
            .map(|seg| seg.prefix)
            .collect();
        assert_eq!(prefixes, vec![Bytes::from_static(b"abc")]);
    }

    #[tokio::test]
    async fn should_reject_reader_with_mismatched_extractor() {
        #[derive(Debug)]
        struct OtherExtractor;
        impl crate::prefix_extractor::PrefixExtractor for OtherExtractor {
            fn name(&self) -> &str {
                "other"
            }
            fn prefix_len(&self, _target: &crate::prefix_extractor::PrefixTarget) -> Option<usize> {
                Some(3)
            }
        }

        // given
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = "/tmp/test_reader_rejects_mismatched_extractor";
        let db = Db::builder(path, object_store.clone())
            .with_settings(Settings::default())
            .with_segment_extractor(Arc::new(test_utils::FixedThreeBytePrefixExtractor))
            .build()
            .await
            .unwrap();
        db.close().await.unwrap();

        // when
        let err = match DbReader::builder(path, object_store.clone())
            .with_segment_extractor(Arc::new(OtherExtractor))
            .build()
            .await
        {
            Ok(_) => panic!("expected mismatched-extractor error"),
            Err(err) => err,
        };

        // then
        assert_eq!(err.kind(), crate::ErrorKind::Invalid);

        // when
        // a segmented database also rejects a reader opened without any extractor
        let err = match DbReader::builder(path, object_store).build().await {
            Ok(_) => panic!("expected missing-extractor error"),
            Err(err) => err,
        };

        // then
        assert_eq!(err.kind(), crate::ErrorKind::Invalid);
        assert!(
            err.to_string()
                .contains("segment extractor configuration mismatch"),
            "unexpected error message: {err}"
        );
    }

    #[tokio::test]
    async fn should_list_checkpoint_segments_for_checkpoint_reader() {
        // given
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = "/tmp/test_reader_lists_checkpoint_segments";
        let db = Db::builder(path, object_store.clone())
            .with_settings(Settings::default())
            .with_segment_extractor(Arc::new(test_utils::FixedThreeBytePrefixExtractor))
            .build()
            .await
            .unwrap();
        let write_opts = WriteOptions {
            await_durable: false,
            ..Default::default()
        };
        db.put_with_options(b"abc-1", b"v1", &PutOptions::default(), &write_opts)
            .await
            .unwrap();
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::MemTable,
        })
        .await
        .unwrap();
        let checkpoint = db
            .create_checkpoint(CheckpointScope::All, &CheckpointOptions::default())
            .await
            .unwrap();
        db.put_with_options(b"xyz-1", b"v2", &PutOptions::default(), &write_opts)
            .await
            .unwrap();
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::MemTable,
        })
        .await
        .unwrap();
        db.close().await.unwrap();

        // when
        let reader = DbReader::builder(path, object_store)
            .with_segment_extractor(Arc::new(test_utils::FixedThreeBytePrefixExtractor))
            .with_checkpoint_id(checkpoint.id)
            .build()
            .await
            .unwrap();

        // then
        let segments: Vec<Bytes> = reader
            .status()
            .list_segments()
            .into_iter()
            .map(|seg| seg.prefix)
            .collect();
        assert_eq!(segments, vec![Bytes::from_static(b"abc")]);
    }

    #[tokio::test]
    async fn should_fail_if_db_is_uninitialized() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_kv_store");
        let test_provider = TestProvider::new(path, Arc::clone(&object_store));
        let manifest_store = test_provider.manifest_store();

        let parent_manifest = Manifest::initial(ManifestCore::new());
        let parent_path = "/tmp/parent_store".to_string();
        let source_checkpoint_id = uuid::Uuid::new_v4();

        let _ = StoredManifest::store_uninitialized_clone(
            Arc::clone(&manifest_store),
            Manifest::cloned(
                &parent_manifest,
                parent_path,
                source_checkpoint_id,
                Arc::new(DbRand::default()),
            ),
            Arc::new(DefaultSystemClock::new()),
        )
        .await
        .unwrap();

        let err = test_provider
            .new_db_reader(DbReaderOptions::default(), None, None)
            .await;
        assert!(matches!(err, Err(SlateDBError::InvalidDBState)));
    }

    #[tokio::test]
    async fn should_scan_from_checkpoint() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_kv_store");
        let test_provider = TestProvider::new(path.clone(), Arc::clone(&object_store));

        let db = test_provider.new_db(Settings::default()).await.unwrap();
        let checkpoint_key = b"checkpoint_key";
        let value = b"value";

        db.put(checkpoint_key, value).await.unwrap();
        let checkpoint_result = db
            .create_checkpoint(CheckpointScope::All, &CheckpointOptions::default())
            .await
            .unwrap();

        let post_checkpoint_key = b"post_checkpoint_key";
        db.put(post_checkpoint_key, value).await.unwrap();

        let reader = test_provider
            .new_db_reader(DbReaderOptions::default(), Some(checkpoint_result.id), None)
            .await
            .unwrap();

        let mut db_iter = reader.scan(..).await.unwrap();
        let mut table = BTreeMap::new();
        table.insert(
            Bytes::copy_from_slice(checkpoint_key),
            Bytes::copy_from_slice(value),
        );

        test_utils::assert_ranged_db_scan(&table, .., IterationOrder::Ascending, &mut db_iter)
            .await;
    }

    #[tokio::test(start_paused = true)]
    async fn should_reestablish_reader_checkpoint() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_kv_store");
        let test_provider = TestProvider::new(path.clone(), Arc::clone(&object_store));

        let db_options = Settings {
            l0_sst_size_bytes: 256,
            ..Settings::default()
        };
        let db = test_provider.new_db(db_options).await.unwrap();
        let reader_options = DbReaderOptions {
            manifest_poll_interval: Duration::from_millis(10),
            ..DbReaderOptions::default()
        };
        let reader = test_provider
            .new_db_reader(reader_options, None, None)
            .await
            .unwrap();
        let manifest_store = test_provider.manifest_store();
        let manifest = manifest_store.read_latest_manifest().await.unwrap();
        let initial_checkpoint_id = manifest.manifest.core.checkpoints.first().unwrap().id;

        let mut rng = new_test_rng(None);
        let table = sample::table(&mut rng, 256, 10);
        for (key, value) in &table {
            db.put(key, value).await.unwrap();
        }
        db.flush().await.unwrap();

        tokio::time::sleep(Duration::from_millis(20)).await;
        let mut db_iter = reader.scan(..).await.unwrap();
        test_utils::assert_ranged_db_scan(&table, .., IterationOrder::Ascending, &mut db_iter)
            .await;

        let manifest = manifest_store.read_latest_manifest().await.unwrap();
        assert!(!manifest.manifest.core.checkpoints.is_empty());
        assert_eq!(
            None,
            manifest
                .manifest
                .core
                .find_checkpoint(initial_checkpoint_id)
        );
    }

    #[tokio::test(start_paused = true)]
    async fn should_refresh_reader_checkpoint() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_kv_store");
        let test_provider = TestProvider::new(path.clone(), Arc::clone(&object_store));

        let _db = test_provider.new_db(Settings::default()).await;
        let reader_options = DbReaderOptions {
            manifest_poll_interval: Duration::from_millis(500),
            checkpoint_lifetime: Duration::from_millis(1000),
            ..DbReaderOptions::default()
        };

        let manifest_store = test_provider.manifest_store();
        let reader = test_provider
            .new_db_reader(reader_options, None, None)
            .await
            .unwrap();

        let initial_manifest = manifest_store.read_latest_manifest().await.unwrap();
        assert_eq!(1, initial_manifest.manifest.core.checkpoints.len());
        let initial_reader_checkpoint = initial_manifest
            .manifest
            .core
            .checkpoints
            .first()
            .unwrap()
            .clone();

        tokio::time::sleep(Duration::from_millis(5000)).await;

        let updated_manifest = manifest_store.read_latest_manifest().await.unwrap();
        assert_eq!(1, updated_manifest.manifest.core.checkpoints.len());
        let updated_reader_checkpoint = updated_manifest
            .manifest
            .core
            .checkpoints
            .first()
            .unwrap()
            .clone();
        assert_eq!(initial_reader_checkpoint.id, updated_reader_checkpoint.id);
        assert!(
            updated_reader_checkpoint.expire_time.unwrap()
                > initial_reader_checkpoint.expire_time.unwrap()
        );

        // The checkpoint is removed on shutdown
        reader.close().await.unwrap();
        let updated_manifest = manifest_store.read_latest_manifest().await.unwrap();
        assert_eq!(0, updated_manifest.manifest.core.checkpoints.len());
    }

    // Regression test for https://github.com/slatedb/slatedb/issues/1750.
    // It constructs DbReaderInner without the background manifest poller,
    // advances a mock clock past the first checkpoint refresh deadline, then
    // verifies the next manual poll does not write another manifest before
    // half of the refreshed checkpoint lifetime has elapsed.
    #[tokio::test]
    async fn should_not_refresh_reader_checkpoint_on_every_poll() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from(format!(
            "/tmp/test_db_reader_checkpoint_refresh_cadence_{}",
            Uuid::new_v4()
        ));
        let clock = Arc::new(MockSystemClock::new());
        let mut test_provider = TestProvider::new(path, Arc::clone(&object_store));
        test_provider.system_clock = clock.clone();

        let manifest_store = test_provider.manifest_store();
        let table_store = test_provider.table_store();

        // Seed the DB with its initial manifest. Opening a reader without a
        // user-provided checkpoint will add a second manifest containing the
        // self-established reader checkpoint.
        let stored_manifest = StoredManifest::create_new_db(
            Arc::clone(&manifest_store),
            ManifestCore::new(),
            clock.clone(),
        )
        .await
        .unwrap();
        let recorder = slatedb_common::metrics::MetricsRecorderHelper::noop();

        // Build DbReaderInner directly instead of DbReader so no spawned poller
        // can race the manual calls to maybe_refresh_checkpoint below.
        let inner = DbReaderInner::new(
            Arc::clone(&manifest_store),
            table_store,
            DbReaderOptions {
                manifest_poll_interval: Duration::from_millis(100),
                checkpoint_lifetime: Duration::from_millis(1000),
                ..DbReaderOptions::default()
            },
            None,
            None,
            None,
            clock.clone(),
            test_provider.rand.clone(),
            recorder,
            stored_manifest,
        )
        .await
        .unwrap();

        // Manifest 1 is the initial DB manifest. Manifest 2 is written when
        // DbReaderInner creates its checkpoint.
        let initial_manifests = manifest_store.list_manifests(..).await.unwrap();
        assert_eq!(
            2,
            initial_manifests.len(),
            "expected initial db manifest plus reader checkpoint manifest"
        );

        // Move just past the first refresh deadline: the checkpoint was created
        // at t=0 with a 1000ms lifetime, so checkpoint_lifetime / 2 is 500ms.
        // This refresh is expected and should write exactly one new manifest.
        clock.advance(Duration::from_millis(501)).await;
        let mut stored_manifest = StoredManifest::load(Arc::clone(&manifest_store), clock.clone())
            .await
            .unwrap();
        inner
            .maybe_refresh_checkpoint(&mut stored_manifest)
            .await
            .unwrap();

        let manifests_after_first_refresh = manifest_store.list_manifests(..).await.unwrap();
        assert_eq!(3, manifests_after_first_refresh.len());

        // Simulate the next manifest poll 100ms later. Because the first
        // refresh extended the checkpoint lifetime to t=1501ms, the next
        // refresh deadline should be t=1001ms. A second manifest here shows
        // the in-memory checkpoint expiry was not updated after the refresh.
        clock.advance(Duration::from_millis(100)).await;
        let mut stored_manifest = StoredManifest::load(Arc::clone(&manifest_store), clock.clone())
            .await
            .unwrap();
        inner
            .maybe_refresh_checkpoint(&mut stored_manifest)
            .await
            .unwrap();

        // Correct behavior keeps the manifest count at 3.
        let manifests_after_second_poll = manifest_store.list_manifests(..).await.unwrap();
        assert_eq!(
            manifests_after_first_refresh.len(),
            manifests_after_second_poll.len(),
            "checkpoint refresh should not write another manifest until half of the refreshed lifetime has elapsed"
        );
    }

    // Regression test for https://github.com/slatedb/slatedb/issues/1888.
    // If a poll tick stalls past the refresh deadline (e.g. during an object
    // store brownout), the reader's checkpoint can expire and be reaped by the
    // writer's GC. The next refresh must re-establish a fresh checkpoint
    // instead of failing the reader permanently with CheckpointMissing.
    #[tokio::test]
    async fn should_reestablish_reader_checkpoint_when_missing() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from(format!(
            "/tmp/test_db_reader_reestablish_missing_checkpoint_{}",
            Uuid::new_v4()
        ));
        let clock = Arc::new(MockSystemClock::new());
        let mut test_provider = TestProvider::new(path, Arc::clone(&object_store));
        test_provider.system_clock = clock.clone();

        let manifest_store = test_provider.manifest_store();
        let table_store = test_provider.table_store();

        let stored_manifest = StoredManifest::create_new_db(
            Arc::clone(&manifest_store),
            ManifestCore::new(),
            clock.clone(),
        )
        .await
        .unwrap();
        let recorder = slatedb_common::metrics::MetricsRecorderHelper::noop();

        // Build DbReaderInner directly so no spawned poller can race the
        // manual call to maybe_refresh_checkpoint below.
        let inner = DbReaderInner::new(
            Arc::clone(&manifest_store),
            table_store,
            DbReaderOptions {
                manifest_poll_interval: Duration::from_millis(100),
                checkpoint_lifetime: Duration::from_millis(1000),
                ..DbReaderOptions::default()
            },
            None,
            None,
            None,
            clock.clone(),
            test_provider.rand.clone(),
            recorder,
            stored_manifest,
        )
        .await
        .unwrap();
        let reader_checkpoint_id = inner.state.read().generation.checkpoint().id;

        // Simulate the writer's GC reaping the expired checkpoint.
        let mut stored_manifest = StoredManifest::load(Arc::clone(&manifest_store), clock.clone())
            .await
            .unwrap();
        stored_manifest
            .delete_checkpoint(reader_checkpoint_id)
            .await
            .unwrap();

        // Move past the refresh deadline (checkpoint_lifetime / 2) and poll.
        clock.advance(Duration::from_millis(501)).await;
        let mut stored_manifest = StoredManifest::load(Arc::clone(&manifest_store), clock.clone())
            .await
            .unwrap();
        inner
            .maybe_refresh_checkpoint(&mut stored_manifest)
            .await
            .unwrap();

        // The reader should have replaced the reaped checkpoint with a new one.
        let new_checkpoint_id = inner.state.read().generation.checkpoint().id;
        assert_ne!(reader_checkpoint_id, new_checkpoint_id);
        let latest_manifest = manifest_store.read_latest_manifest().await.unwrap();
        let checkpoints = &latest_manifest.manifest.core.checkpoints;
        assert_eq!(1, checkpoints.len());
        assert_eq!(new_checkpoint_id, checkpoints[0].id);
    }

    // A missing user-supplied checkpoint must still fail loud rather than be
    // silently replaced (RFC-0004): the caller's pinned view is gone.
    #[tokio::test]
    async fn should_fail_refresh_when_user_checkpoint_missing() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from(format!(
            "/tmp/test_db_reader_user_checkpoint_missing_{}",
            Uuid::new_v4()
        ));
        let clock = Arc::new(MockSystemClock::new());
        let mut test_provider = TestProvider::new(path, Arc::clone(&object_store));
        test_provider.system_clock = clock.clone();

        let manifest_store = test_provider.manifest_store();
        let table_store = test_provider.table_store();

        let mut stored_manifest = StoredManifest::create_new_db(
            Arc::clone(&manifest_store),
            ManifestCore::new(),
            clock.clone(),
        )
        .await
        .unwrap();
        let user_checkpoint_id = Uuid::new_v4();
        stored_manifest
            .write_checkpoint(
                user_checkpoint_id,
                &CheckpointOptions {
                    lifetime: Some(Duration::from_millis(1000)),
                    ..CheckpointOptions::default()
                },
            )
            .await
            .unwrap();
        let recorder = slatedb_common::metrics::MetricsRecorderHelper::noop();

        let inner = DbReaderInner::new(
            Arc::clone(&manifest_store),
            table_store,
            DbReaderOptions {
                manifest_poll_interval: Duration::from_millis(100),
                checkpoint_lifetime: Duration::from_millis(1000),
                ..DbReaderOptions::default()
            },
            Some(user_checkpoint_id),
            None,
            None,
            clock.clone(),
            test_provider.rand.clone(),
            recorder,
            stored_manifest,
        )
        .await
        .unwrap();

        let mut stored_manifest = StoredManifest::load(Arc::clone(&manifest_store), clock.clone())
            .await
            .unwrap();
        stored_manifest
            .delete_checkpoint(user_checkpoint_id)
            .await
            .unwrap();

        clock.advance(Duration::from_millis(501)).await;
        let mut stored_manifest = StoredManifest::load(Arc::clone(&manifest_store), clock.clone())
            .await
            .unwrap();
        let result = inner.maybe_refresh_checkpoint(&mut stored_manifest).await;
        assert!(
            matches!(result, Err(SlateDBError::CheckpointMissing(id)) if id == user_checkpoint_id)
        );
    }

    #[tokio::test(start_paused = true)]
    async fn should_replay_new_wals() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_kv_store");
        let test_provider = TestProvider::new(path.clone(), Arc::clone(&object_store));
        let db = test_provider.new_db(Settings::default()).await.unwrap();

        let reader_options = DbReaderOptions {
            manifest_poll_interval: Duration::from_millis(500),
            checkpoint_lifetime: Duration::from_millis(1000),
            ..DbReaderOptions::default()
        };

        let reader = test_provider
            .new_db_reader(reader_options, None, None)
            .await
            .unwrap();
        let key = b"test_key";
        let value = b"test_value";
        db.put(key, value).await.unwrap();
        db.flush().await.unwrap();

        tokio::time::sleep(Duration::from_millis(500)).await;
        assert_eq!(
            reader.get(key).await.unwrap(),
            Some(Bytes::from_static(value))
        );
    }

    #[tokio::test]
    async fn reader_snapshot_should_remain_stable_while_wal_replay_advances() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_db_reader_snapshot_wal_isolation");
        let test_provider = TestProvider::new(path, Arc::clone(&object_store));
        let db = test_provider
            .new_db(Settings {
                flush_interval: None,
                compactor_options: None,
                garbage_collector_options: None,
                ..Settings::default()
            })
            .await
            .unwrap();

        let write_options = WriteOptions {
            await_durable: false,
            ..WriteOptions::default()
        };
        db.put_with_options(b"key", b"v1", &PutOptions::default(), &write_options)
            .await
            .unwrap();
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::Wal,
        })
        .await
        .unwrap();

        let reader = test_provider
            .new_db_reader(
                DbReaderOptions {
                    manifest_poll_interval: Duration::from_millis(10),
                    ..DbReaderOptions::default()
                },
                None,
                None,
            )
            .await
            .unwrap();
        let snapshot = reader.snapshot().await.unwrap();

        db.put_with_options(b"key", b"v2", &PutOptions::default(), &write_options)
            .await
            .unwrap();
        db.put_with_options(b"new", b"v3", &PutOptions::default(), &write_options)
            .await
            .unwrap();
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::Wal,
        })
        .await
        .unwrap();
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if reader.get(b"key").await.unwrap() == Some(Bytes::from_static(b"v2")) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("live reader should replay the second WAL generation");

        assert_eq!(reader.get(b"key").await.unwrap(), Some(Bytes::from("v2")));
        assert_eq!(snapshot.get(b"key").await.unwrap(), Some(Bytes::from("v1")));
        let keys = [&b"key"[..], &b"key"[..], &b"new"[..]];
        assert_eq!(
            snapshot.multi_get(&keys).await.unwrap(),
            vec![
                Some(Bytes::from_static(b"v1")),
                Some(Bytes::from_static(b"v1")),
                None,
            ]
        );
        assert_eq!(
            reader.multi_get(&keys).await.unwrap(),
            vec![
                Some(Bytes::from_static(b"v2")),
                Some(Bytes::from_static(b"v2")),
                Some(Bytes::from_static(b"v3")),
            ]
        );
        assert_eq!(
            reader.snapshot().await.unwrap().get(b"key").await.unwrap(),
            Some(Bytes::from("v2"))
        );
    }

    #[tokio::test]
    async fn reader_snapshot_isolation_should_survive_db_cache_hits_and_eviction() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_db_reader_snapshot_db_cache_isolation");
        let db = Db::builder(path.clone(), Arc::clone(&object_store))
            .with_settings(Settings {
                flush_interval: None,
                compactor_options: None,
                garbage_collector_options: None,
                ..Settings::default()
            })
            .build()
            .await
            .unwrap();
        let write_options = WriteOptions {
            await_durable: false,
            ..WriteOptions::default()
        };
        db.put_with_options(b"key", b"v1", &PutOptions::default(), &write_options)
            .await
            .unwrap();
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::MemTable,
        })
        .await
        .unwrap();

        let cache = Arc::new(TestCache::new());
        let reader = DbReader::builder(path, Arc::clone(&object_store))
            .with_options(DbReaderOptions {
                manifest_poll_interval: Duration::from_millis(10),
                ..DbReaderOptions::default()
            })
            .with_db_cache(cache.clone())
            .build()
            .await
            .unwrap();
        let old_snapshot = reader.snapshot().await.unwrap();
        let old_generation = reader.inner.state.read().generation.checkpoint().id;

        assert_eq!(
            old_snapshot.get(b"key").await.unwrap(),
            Some(Bytes::from_static(b"v1"))
        );
        assert!(
            cache.inserts() > 0,
            "the cold read should populate the cache"
        );
        let hits_after_cold_read = cache.hits();
        assert_eq!(
            old_snapshot.get(b"key").await.unwrap(),
            Some(Bytes::from_static(b"v1"))
        );
        assert!(
            cache.hits() > hits_after_cold_read,
            "the second old-generation read should hit the cache"
        );

        db.put_with_options(b"key", b"v2", &PutOptions::default(), &write_options)
            .await
            .unwrap();
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::MemTable,
        })
        .await
        .unwrap();
        wait_for_reader_generation_change(&reader, old_generation).await;

        assert_eq!(reader.get(b"key").await.unwrap(), Some(Bytes::from("v2")));
        let hits_before_old_generation_read = cache.hits();
        assert_eq!(
            old_snapshot.get(b"key").await.unwrap(),
            Some(Bytes::from_static(b"v1"))
        );
        assert!(
            cache.hits() > hits_before_old_generation_read,
            "warming the new SST must not displace or alias the old SST's cache key"
        );

        cache.clear();
        assert_eq!(cache.entry_count(), 0);
        let misses_before_reload = cache.misses();
        let inserts_before_reload = cache.inserts();
        assert_eq!(
            old_snapshot.get(b"key").await.unwrap(),
            Some(Bytes::from_static(b"v1"))
        );
        assert!(cache.misses() > misses_before_reload);
        assert!(
            cache.inserts() > inserts_before_reload,
            "an evicted old-generation block should reload from its checkpoint-pinned SST"
        );
        assert_eq!(
            reader.snapshot().await.unwrap().get(b"key").await.unwrap(),
            Some(Bytes::from_static(b"v2"))
        );

        drop(old_snapshot);
        reader.close().await.unwrap();
        db.close().await.unwrap();
    }

    #[tokio::test]
    async fn reader_snapshot_isolation_should_survive_warm_object_store_cache_hits() {
        use crate::cached_object_store::stats::PART_HIT_COUNT;
        use slatedb_common::metrics::{lookup_metric, DefaultMetricsRecorder};

        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_db_reader_snapshot_object_cache_isolation");
        let db = Db::builder(path.clone(), Arc::clone(&object_store))
            .with_settings(Settings {
                flush_interval: None,
                compactor_options: None,
                garbage_collector_options: None,
                ..Settings::default()
            })
            .build()
            .await
            .unwrap();
        let write_options = WriteOptions {
            await_durable: false,
            ..WriteOptions::default()
        };
        db.put_with_options(b"key", b"v1", &PutOptions::default(), &write_options)
            .await
            .unwrap();
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::MemTable,
        })
        .await
        .unwrap();

        let cache_dir = tempfile::Builder::new()
            .prefix("dbreader_snapshot_object_cache_")
            .tempdir()
            .unwrap();
        let mut options = DbReaderOptions {
            manifest_poll_interval: Duration::from_millis(10),
            ..DbReaderOptions::default()
        };
        options.object_store_cache_options.root_folder = Some(cache_dir.path().to_path_buf());
        options.object_store_cache_options.part_size_bytes = 1024;
        options.object_store_cache_options.scan_interval = None;
        let metrics = Arc::new(DefaultMetricsRecorder::new());
        let reader = DbReader::builder(path, Arc::clone(&object_store))
            .with_options(options)
            // Force SST reads through the object-store cache instead of allowing
            // the in-memory block cache to satisfy the second read first.
            .with_db_cache_disabled()
            .with_metrics_recorder(metrics.clone())
            .build()
            .await
            .unwrap();
        let old_snapshot = reader.snapshot().await.unwrap();
        let old_generation = reader.inner.state.read().generation.checkpoint().id;

        assert_eq!(
            old_snapshot.get(b"key").await.unwrap(),
            Some(Bytes::from_static(b"v1"))
        );
        let hits_after_cold_read = lookup_metric(&metrics, PART_HIT_COUNT).unwrap_or(0);
        assert_eq!(
            old_snapshot.get(b"key").await.unwrap(),
            Some(Bytes::from_static(b"v1"))
        );
        assert!(
            lookup_metric(&metrics, PART_HIT_COUNT).unwrap_or(0) > hits_after_cold_read,
            "the second read should be served from the local object-store cache"
        );

        db.put_with_options(b"key", b"v2", &PutOptions::default(), &write_options)
            .await
            .unwrap();
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::MemTable,
        })
        .await
        .unwrap();
        wait_for_reader_generation_change(&reader, old_generation).await;
        assert_eq!(reader.get(b"key").await.unwrap(), Some(Bytes::from("v2")));

        let hits_before_old_generation_read = lookup_metric(&metrics, PART_HIT_COUNT).unwrap_or(0);
        assert_eq!(
            old_snapshot.get(b"key").await.unwrap(),
            Some(Bytes::from_static(b"v1"))
        );
        assert!(
            lookup_metric(&metrics, PART_HIT_COUNT).unwrap_or(0) > hits_before_old_generation_read,
            "the old snapshot should still use the cached bytes for its own immutable SST"
        );

        drop(old_snapshot);
        reader.close().await.unwrap();
        db.close().await.unwrap();
    }

    #[tokio::test]
    async fn reader_snapshot_creation_should_do_no_io_and_share_generation_checkpoint() {
        let recording = Arc::new(test_utils::RecordingObjectStore::new(Arc::new(
            InMemory::new(),
        )));
        let object_store: Arc<dyn ObjectStore> = recording.clone();
        let path = Path::from("/tmp/test_db_reader_snapshot_no_io");
        let test_provider = TestProvider::new(path, Arc::clone(&object_store));
        let db = test_provider.new_db(Settings::default()).await.unwrap();
        db.put(b"key", b"value").await.unwrap();
        db.flush().await.unwrap();
        let checkpoint = db
            .create_checkpoint(CheckpointScope::All, &CheckpointOptions::default())
            .await
            .unwrap();
        let reader = test_provider
            .new_db_reader(DbReaderOptions::default(), Some(checkpoint.id), None)
            .await
            .unwrap();
        recording.clear();

        let snapshots = futures::future::try_join_all((0..100).map(|_| reader.snapshot()))
            .await
            .unwrap();

        assert_eq!(100, snapshots.len());
        assert!(recording.get_kinds(false).is_empty());
        assert!(recording.get_kinds(true).is_empty());
        assert!(recording.write_kinds().is_empty());
        let manifest = test_provider
            .manifest_store()
            .read_latest_manifest()
            .await
            .unwrap();
        assert_eq!(1, manifest.manifest.core.checkpoints.len());
    }

    #[tokio::test]
    async fn reader_snapshot_should_reject_an_invalid_current_generation() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_db_reader_snapshot_invalid_generation");
        let test_provider = TestProvider::new(path, Arc::clone(&object_store));
        let db = test_provider.new_db(Settings::default()).await.unwrap();
        let checkpoint = db
            .create_checkpoint(CheckpointScope::All, &CheckpointOptions::default())
            .await
            .unwrap();
        let reader = test_provider
            .new_db_reader(DbReaderOptions::default(), Some(checkpoint.id), None)
            .await
            .unwrap();
        reader.inner.state.read().generation.invalidate();

        let result = reader.snapshot().await;

        let err = match result {
            Ok(_) => panic!("an invalid generation must not produce a snapshot"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("reader checkpoint lease lost"));
    }

    #[tokio::test]
    async fn snapshot_should_keep_old_generation_checkpoint_until_drop() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_db_reader_snapshot_checkpoint_retention");
        let test_provider = TestProvider::new(path, Arc::clone(&object_store));
        let db = test_provider
            .new_db(Settings {
                flush_interval: None,
                compactor_options: None,
                garbage_collector_options: None,
                ..Settings::default()
            })
            .await
            .unwrap();
        let write_options = WriteOptions {
            await_durable: false,
            ..WriteOptions::default()
        };
        db.put_with_options(b"key", b"v1", &PutOptions::default(), &write_options)
            .await
            .unwrap();
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::MemTable,
        })
        .await
        .unwrap();

        let reader = test_provider
            .new_db_reader(
                DbReaderOptions {
                    manifest_poll_interval: Duration::from_millis(10),
                    ..DbReaderOptions::default()
                },
                None,
                None,
            )
            .await
            .unwrap();
        let snapshot = reader.snapshot().await.unwrap();
        let old_checkpoint_id = reader.inner.state.read().generation.checkpoint().id;

        db.put_with_options(b"key", b"v2", &PutOptions::default(), &write_options)
            .await
            .unwrap();
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::MemTable,
        })
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;

        let new_checkpoint_id = reader.inner.state.read().generation.checkpoint().id;
        assert_ne!(old_checkpoint_id, new_checkpoint_id);
        let manifest = test_provider
            .manifest_store()
            .read_latest_manifest()
            .await
            .unwrap();
        assert!(manifest
            .manifest
            .core
            .find_checkpoint(old_checkpoint_id)
            .is_some());
        assert!(manifest
            .manifest
            .core
            .find_checkpoint(new_checkpoint_id)
            .is_some());
        assert_eq!(snapshot.get(b"key").await.unwrap(), Some(Bytes::from("v1")));

        drop(snapshot);
        tokio::time::sleep(Duration::from_millis(20)).await;

        let manifest = test_provider
            .manifest_store()
            .read_latest_manifest()
            .await
            .unwrap();
        assert!(manifest
            .manifest
            .core
            .find_checkpoint(old_checkpoint_id)
            .is_none());
        assert!(manifest
            .manifest
            .core
            .find_checkpoint(new_checkpoint_id)
            .is_some());
    }

    #[tokio::test]
    async fn snapshot_iterator_should_retain_checkpoint_after_snapshot_drop() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_db_reader_snapshot_iterator_retention");
        let test_provider = TestProvider::new(path, Arc::clone(&object_store));
        let db = test_provider
            .new_db(Settings {
                flush_interval: None,
                compactor_options: None,
                garbage_collector_options: None,
                ..Settings::default()
            })
            .await
            .unwrap();
        let write_options = WriteOptions {
            await_durable: false,
            ..WriteOptions::default()
        };
        db.put_with_options(b"a", b"old", &PutOptions::default(), &write_options)
            .await
            .unwrap();
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::MemTable,
        })
        .await
        .unwrap();

        let reader = test_provider
            .new_db_reader(
                DbReaderOptions {
                    manifest_poll_interval: Duration::from_millis(10),
                    ..DbReaderOptions::default()
                },
                None,
                None,
            )
            .await
            .unwrap();
        let snapshot = reader.snapshot().await.unwrap();
        let mut iter = snapshot.scan(..).await.unwrap();
        let old_checkpoint_id = reader.inner.state.read().generation.checkpoint().id;
        drop(snapshot);

        db.put_with_options(b"b", b"new", &PutOptions::default(), &write_options)
            .await
            .unwrap();
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::MemTable,
        })
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;

        let manifest = test_provider
            .manifest_store()
            .read_latest_manifest()
            .await
            .unwrap();
        assert!(manifest
            .manifest
            .core
            .find_checkpoint(old_checkpoint_id)
            .is_some());
        let row = iter.next().await.unwrap().unwrap();
        assert_eq!(row.key, Bytes::from("a"));
        assert_eq!(row.value, Bytes::from("old"));
        assert!(iter.next().await.unwrap().is_none());

        drop(iter);
        tokio::time::sleep(Duration::from_millis(20)).await;
        let manifest = test_provider
            .manifest_store()
            .read_latest_manifest()
            .await
            .unwrap();
        assert!(manifest
            .manifest
            .core
            .find_checkpoint(old_checkpoint_id)
            .is_none());
    }

    #[tokio::test]
    async fn generation_drain_should_wait_for_in_flight_operations_and_reject_new_ones() {
        let clock: Arc<dyn SystemClock> = Arc::new(DefaultSystemClock::new());
        let generation = ReaderGeneration::new(
            test_checkpoint(1, clock),
            Manifest::initial(ManifestCore::new()),
        );
        let permit = generation.acquire().unwrap();
        generation.invalidate();
        let draining_generation = Arc::clone(&generation);
        let drain = tokio::spawn(async move { draining_generation.drain().await });
        tokio::task::yield_now().await;
        assert!(!drain.is_finished());

        drop(permit);
        drain.await.unwrap();
        assert!(matches!(
            generation.acquire(),
            Err(SlateDBError::CheckpointLeaseLost(_))
        ));
    }

    #[tokio::test]
    async fn replay_wal_into_should_use_latest_existing_table_and_keep_newest_first_order() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_db_reader_replay_order");
        let test_provider = TestProvider::new(path, Arc::clone(&object_store));
        let table_store = test_provider.table_store();

        write_wal_sst(
            Arc::clone(&table_store),
            3,
            vec![RowEntry::new_value(b"stale_key", b"stale_value", 3)],
        )
        .await
        .unwrap();
        write_wal_sst(
            Arc::clone(&table_store),
            4,
            vec![RowEntry::new_value(b"fresh_key", b"fresh_value", 4)],
        )
        .await
        .unwrap();

        let mut into_tables: ReplayMemtables = [
            immutable_memtable(
                3,
                vec![RowEntry::new_value(b"stale_key", b"stale_value", 3)],
            ),
            immutable_memtable(
                2,
                vec![RowEntry::new_value(b"older_key", b"older_value", 2)],
            ),
        ]
        .into_iter()
        .collect();

        let mut core = ManifestCore::new();
        core.next_wal_sst_id = 5;

        let (last_wal_id, last_committed_seq) = DbReaderInner::replay_wal_into(
            Arc::clone(&table_store),
            &DbReaderOptions::default(),
            &core,
            &mut into_tables,
            None,
            false,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(last_wal_id, 4);
        assert_eq!(last_committed_seq, 4);

        let newest_replayed = into_tables.front().unwrap();
        assert_eq!(newest_replayed.recent_flushed_wal_id(), 4);

        let newest_table = newest_replayed.table();
        let mut newest_iter = newest_table.iter();
        test_utils::assert_iterator(
            &mut newest_iter,
            vec![RowEntry::new_value(b"fresh_key", b"fresh_value", 4)],
        )
        .await;
    }

    #[test]
    fn replay_memtable_prepend_should_share_the_existing_tail() {
        let mut original = ReplayMemtables::default();
        original.prepend(immutable_memtable(
            1,
            vec![RowEntry::new_value(b"old", b"value", 1)],
        ));
        let original_head = Arc::clone(original.head.as_ref().unwrap());

        let mut extended = original.clone();
        extended.prepend(immutable_memtable(
            2,
            vec![RowEntry::new_value(b"new", b"value", 2)],
        ));

        let shared_tail = extended.head.as_ref().unwrap().older.as_ref().unwrap();
        assert!(Arc::ptr_eq(shared_tail, &original_head));
        assert_eq!(1, original.len());
        assert_eq!(2, extended.len());
    }

    #[tokio::test]
    async fn replay_wal_into_should_publish_each_bounded_batch() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_db_reader_incremental_replay_publication");
        let test_provider = TestProvider::new(path, Arc::clone(&object_store));
        let table_store = test_provider.table_store();
        let rows = (1..=3)
            .map(|seq| RowEntry::new_value(format!("key-{seq}").as_bytes(), &[b'x'; 128], seq))
            .collect::<Vec<_>>();
        for (wal_id, row) in rows.iter().cloned().enumerate() {
            write_wal_sst(Arc::clone(&table_store), wal_id as u64 + 1, vec![row])
                .await
                .unwrap();
        }
        let max_memtable_bytes =
            table_store.estimate_encoded_size_compacted(1, rows[0].estimated_size()) as u64;
        let options = DbReaderOptions {
            max_memtable_bytes,
            ..DbReaderOptions::default()
        };
        let mut core = ManifestCore::new();
        core.next_wal_sst_id = 4;
        let mut into_tables = ReplayMemtables::default();
        let mut publications = Vec::new();
        let mut publish = |tables: &ReplayMemtables, wal_id, seq| {
            publications.push((wal_id, seq, tables.len()));
        };

        let result = DbReaderInner::replay_wal_into(
            Arc::clone(&table_store),
            &options,
            &core,
            &mut into_tables,
            None,
            false,
            None,
            Some(&mut publish),
            None,
        )
        .await
        .unwrap();

        assert_eq!((3, 3), result);
        assert_eq!(vec![(1, 1, 1), (2, 2, 2), (3, 3, 3)], publications);
    }

    #[test]
    fn has_not_found_object_store_error_should_walk_nested_error_sources() {
        let err = crate::Error::from(SlateDBError::from(object_store::Error::NotFound {
            path: "missing-wal".to_string(),
            source: Box::new(std::io::Error::other("missing")),
        }));

        assert!(super::has_not_found_object_store_error(&err));
    }

    #[test]
    fn has_not_found_object_store_error_should_ignore_non_not_found_errors() {
        let err = SlateDBError::from(object_store::Error::NotImplemented {
            operation: "test".to_string(),
            implementer: "test".to_string(),
        });

        assert!(!super::has_not_found_object_store_error(&err));
    }

    #[tokio::test]
    async fn replay_wal_into_should_treat_missing_wal_sst_as_end_of_iteration() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_db_reader_missing_wal");
        let test_provider = TestProvider::new(path, Arc::clone(&object_store));
        let table_store = test_provider.table_store();

        write_wal_sst(
            Arc::clone(&table_store),
            1,
            vec![RowEntry::new_value(b"key", b"value", 1)],
        )
        .await
        .unwrap();

        let mut into_tables = ReplayMemtables::default();
        let mut core = ManifestCore::new();
        core.next_wal_sst_id = 3;

        let (last_wal_id, last_committed_seq) = DbReaderInner::replay_wal_into(
            Arc::clone(&table_store),
            &DbReaderOptions::default(),
            &core,
            &mut into_tables,
            None,
            false,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(last_wal_id, 0);
        assert_eq!(last_committed_seq, 0);
        assert!(into_tables.is_empty());
    }

    #[tokio::test]
    async fn replay_wal_into_should_keep_previously_replayed_tables_before_missing_wal_sst() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_db_reader_missing_wal_after_replay");
        let test_provider = TestProvider::new(path, Arc::clone(&object_store));
        let table_store = test_provider.table_store();

        let wal_1_row = RowEntry::new_value(b"a", &[b'a'; 8], 1);
        let wal_2_row_1 = RowEntry::new_value(b"b", &[b'b'; 40], 2);
        let wal_2_row_2 = RowEntry::new_value(b"c", &[b'c'; 40], 3);

        let max_memtable_bytes = table_store.estimate_encoded_size_compacted(
            2,
            wal_1_row.estimated_size() + wal_2_row_1.estimated_size(),
        ) as u64;

        write_wal_sst(Arc::clone(&table_store), 1, vec![wal_1_row.clone()])
            .await
            .unwrap();
        write_wal_sst(
            Arc::clone(&table_store),
            2,
            vec![wal_2_row_1.clone(), wal_2_row_2.clone()],
        )
        .await
        .unwrap();

        let mut into_tables = ReplayMemtables::default();
        let mut core = ManifestCore::new();
        // Force the reader to attempt to read up to 4 even though 3 and 4 don't exist.
        core.next_wal_sst_id = 4;
        let reader_options = DbReaderOptions {
            max_memtable_bytes,
            ..DbReaderOptions::default()
        };

        let (last_wal_id, last_committed_seq) = DbReaderInner::replay_wal_into(
            Arc::clone(&table_store),
            &reader_options,
            &core,
            &mut into_tables,
            None,
            false,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(last_wal_id, 2);
        assert_eq!(last_committed_seq, 3);
        assert_eq!(into_tables.len(), 1);

        let replayed = into_tables.front().unwrap();
        assert_eq!(replayed.recent_flushed_wal_id(), 2);

        let mut replayed_iter = replayed.table().iter();
        test_utils::assert_iterator(
            &mut replayed_iter,
            vec![wal_1_row, wal_2_row_1, wal_2_row_2],
        )
        .await;
    }

    #[tokio::test]
    async fn replay_wal_into_should_noop_for_fresh_db_with_no_writes() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_db_reader_fresh_db_no_writes");
        let test_provider = TestProvider::new(path, Arc::clone(&object_store));
        let table_store = test_provider.table_store();

        let mut into_tables = ReplayMemtables::default();
        let core = ManifestCore::new();

        let (last_wal_id, last_committed_seq) = DbReaderInner::replay_wal_into(
            Arc::clone(&table_store),
            &DbReaderOptions::default(),
            &core,
            &mut into_tables,
            None,
            true,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(last_wal_id, 0);
        assert_eq!(last_committed_seq, 0);
        assert!(into_tables.is_empty());
    }

    #[tokio::test]
    async fn replay_wal_into_should_replay_single_wal_for_fresh_db() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_db_reader_fresh_db_one_wal");
        let test_provider = TestProvider::new(path, Arc::clone(&object_store));
        let table_store = test_provider.table_store();

        let wal_row = RowEntry::new_value(b"key", b"value", 1);
        write_wal_sst(Arc::clone(&table_store), 1, vec![wal_row.clone()])
            .await
            .unwrap();

        let mut into_tables = ReplayMemtables::default();
        let core = ManifestCore::new();

        let (last_wal_id, last_committed_seq) = DbReaderInner::replay_wal_into(
            Arc::clone(&table_store),
            &DbReaderOptions::default(),
            &core,
            &mut into_tables,
            None,
            true,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(last_wal_id, 1);
        assert_eq!(last_committed_seq, 1);
        assert_eq!(into_tables.len(), 1);

        let replayed = into_tables.front().unwrap();
        assert_eq!(replayed.recent_flushed_wal_id(), 1);

        let mut replayed_iter = replayed.table().iter();
        test_utils::assert_iterator(&mut replayed_iter, vec![wal_row]).await;
    }

    #[tokio::test]
    async fn replay_wal_into_should_preserve_existing_last_committed_seq_for_empty_fence_wal() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_db_reader_empty_fence_wal");
        let test_provider = TestProvider::new(path, Arc::clone(&object_store));
        let table_store = test_provider.table_store();

        write_wal_sst(Arc::clone(&table_store), 6, vec![])
            .await
            .unwrap();

        let mut into_tables = ReplayMemtables::default();
        into_tables.prepend(immutable_memtable(
            5,
            vec![
                RowEntry::new_value(b"existing_key_1", b"existing_value_1", 9),
                RowEntry::new_value(b"existing_key_2", b"existing_value_2", 10),
            ],
        ));

        let mut core = ManifestCore::new();
        core.last_l0_seq = 8;
        core.next_wal_sst_id = 5;

        let (last_wal_id, last_committed_seq) = DbReaderInner::replay_wal_into(
            Arc::clone(&table_store),
            &DbReaderOptions::default(),
            &core,
            &mut into_tables,
            None,
            true,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(last_wal_id, 6);
        assert_eq!(last_committed_seq, 10);

        let head_after_first_replay = Arc::clone(into_tables.head.as_ref().unwrap());
        let (last_wal_id, last_committed_seq) = DbReaderInner::replay_wal_into(
            Arc::clone(&table_store),
            &DbReaderOptions::default(),
            &core,
            &mut into_tables,
            Some((last_wal_id, last_committed_seq)),
            true,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(last_wal_id, 6);
        assert_eq!(last_committed_seq, 10);
        assert!(Arc::ptr_eq(
            into_tables.head.as_ref().unwrap(),
            &head_after_first_replay
        ));
    }

    #[tokio::test(start_paused = true)]
    async fn should_fail_new_reads_if_manifest_poller_crashes() {
        use slatedb_common::metrics::{
            lookup_metric, DefaultMetricsRecorder, MetricLevel, MetricsRecorderHelper,
        };

        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_kv_store");
        let test_provider = TestProvider::new(path.clone(), Arc::clone(&object_store));
        let _db = test_provider.new_db(Settings::default()).await.unwrap();

        let reader_options = DbReaderOptions {
            manifest_poll_interval: Duration::from_millis(500),
            ..DbReaderOptions::default()
        };
        let metrics_recorder = Arc::new(DefaultMetricsRecorder::new());
        let reader = DbReader::open_internal(
            test_provider.manifest_store(),
            test_provider.table_store(),
            None,
            None,
            None,
            reader_options,
            Arc::clone(&test_provider.system_clock),
            Arc::clone(&test_provider.rand),
            MetricsRecorderHelper::new(metrics_recorder.clone(), MetricLevel::default()),
        )
        .await
        .unwrap();
        assert_eq!(
            Some(1),
            lookup_metric(
                &metrics_recorder,
                crate::db_stats::READER_ACTIVE_CHECKPOINTS
            )
        );

        fail_parallel::cfg(
            Arc::clone(&test_provider.fp_registry),
            "probe-wal-ssts",
            "return",
        )
        .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        let result = reader.get(b"key").await.unwrap_err();
        assert_eq!(result.to_string(), "Unavailable error: io error (oops)");
        assert_eq!(
            Some(0),
            lookup_metric(
                &metrics_recorder,
                crate::db_stats::READER_ACTIVE_CHECKPOINTS
            )
        );
    }

    #[tokio::test]
    async fn skip_wal_replay_should_not_see_wal_only_writes() {
        use crate::config::{FlushOptions, FlushType};

        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_kv_store");
        let test_provider = TestProvider::new(path.clone(), Arc::clone(&object_store));

        // Create a DB and write some data, then flush memtable to L0 SSTs
        let db = test_provider.new_db(Settings::default()).await.unwrap();
        let flushed_key = b"flushed_key";
        let flushed_value = b"flushed_value";
        db.put(flushed_key, flushed_value).await.unwrap();
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::MemTable,
        })
        .await
        .unwrap();

        // Write more data that stays in WAL (not flushed to L0)
        let wal_only_key = b"wal_only_key";
        let wal_only_value = b"wal_only_value";
        db.put(wal_only_key, wal_only_value).await.unwrap();
        // Only flush to WAL, not to L0 SSTs
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::Wal,
        })
        .await
        .unwrap();

        // Open a reader with skip_wal_replay=true
        let reader_options = DbReaderOptions {
            skip_wal_replay: true,
            ..DbReaderOptions::default()
        };
        let reader = test_provider
            .new_db_reader(reader_options.clone(), None, None)
            .await
            .unwrap();

        // Should see the L0 flushed data
        assert_eq!(
            reader.get(flushed_key).await.unwrap(),
            Some(Bytes::from_static(flushed_value))
        );

        // Should NOT see the WAL-only data
        assert_eq!(reader.get(wal_only_key).await.unwrap(), None);

        // After flushing memtable to L0, a NEW reader should see the data
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::MemTable,
        })
        .await
        .unwrap();

        // Open a new reader - it should see the newly flushed data
        let reader2 = test_provider
            .new_db_reader(reader_options, None, None)
            .await
            .unwrap();
        assert_eq!(
            reader2.get(wal_only_key).await.unwrap(),
            Some(Bytes::from_static(wal_only_value))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn skip_wal_replay_should_be_respected_during_reestablish_checkpoint() {
        use crate::config::{FlushOptions, FlushType};

        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_kv_store");
        let test_provider = TestProvider::new(path.clone(), Arc::clone(&object_store));

        let db = test_provider.new_db(Settings::default()).await.unwrap();

        // Write initial data and flush to L0 so the reader opens with this state
        db.put(b"key1", b"value1").await.unwrap();
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::MemTable,
        })
        .await
        .unwrap();

        // Open reader with skip_wal_replay=true. The poller's first tick fires
        // immediately (during the next yield) and sees no manifest change.
        let reader_options = DbReaderOptions {
            manifest_poll_interval: Duration::from_millis(100),
            skip_wal_replay: true,
            ..DbReaderOptions::default()
        };
        let reader = test_provider
            .new_db_reader(reader_options, None, None)
            .await
            .unwrap();

        // Capture checkpoint ID before the flush so the wait condition is not
        // affected by a race where the poller replaces the checkpoint during
        // the flush.
        let manifest_store = test_provider.manifest_store();
        let mut stored_manifest =
            StoredManifest::load(manifest_store, test_provider.system_clock.clone())
                .await
                .unwrap();
        let initial_checkpoint_id = stored_manifest
            .manifest()
            .core
            .checkpoints
            .first()
            .unwrap()
            .id;

        // Inject a failpoint on WAL probing before flushing so it is active
        // when the poller fires. With the buggy replay_new_wals=true,
        // reestablish_checkpoint calls last_seen_wal_id() which probes WAL SSTs
        // and hits this failpoint. With the fix (replay_new_wals=false), the
        // WAL probing is skipped entirely.
        fail_parallel::cfg(
            Arc::clone(&test_provider.fp_registry),
            "probe-wal-ssts",
            "return",
        )
        .unwrap();

        // Write more data and flush to L0, changing the manifest's L0 state.
        // This makes should_reestablish_checkpoint() return true on the next poll.
        // Note: the writer uses its own TableStore (not the test_provider's),
        // so the failpoint above does not affect the writer's flush path.
        db.put(b"key2", b"value2").await.unwrap();
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::MemTable,
        })
        .await
        .unwrap();

        // Wait for the manifest poller to see the changed L0 state and
        // reestablish the checkpoint. Without the fix, the poller crashes
        // on the WAL listing failpoint.
        let timeout = Duration::from_secs(5);
        let start = tokio::time::Instant::now();
        loop {
            assert!(
                start.elapsed() < timeout,
                "timed out waiting for checkpoint reestablishment"
            );
            let manifest = stored_manifest.refresh().await.unwrap();
            let current_checkpoint = manifest.core.checkpoints.first().unwrap();
            if current_checkpoint.id != initial_checkpoint_id {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // With the fix, the reader should still work (poller didn't crash).
        // Without the fix, the poller crashes and get() returns an error.
        let result = reader.get(b"key1").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn skip_wal_replay_should_see_l0_data() {
        use crate::config::{FlushOptions, FlushType};

        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_kv_store");
        let test_provider = TestProvider::new(path.clone(), Arc::clone(&object_store));

        // Create a DB and write data, then flush memtable to L0 SSTs
        let db = test_provider.new_db(Settings::default()).await.unwrap();
        let key = b"test_key";
        let value = b"test_value";
        db.put(key, value).await.unwrap();
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::MemTable,
        })
        .await
        .unwrap();

        // Open a reader with skip_wal_replay=true
        let reader_options = DbReaderOptions {
            skip_wal_replay: true,
            ..DbReaderOptions::default()
        };
        let reader = test_provider
            .new_db_reader(reader_options, None, None)
            .await
            .unwrap();

        // Should see the L0 data
        assert_eq!(
            reader.get(key).await.unwrap(),
            Some(Bytes::from_static(value))
        );
    }

    struct TestProvider {
        object_store: Arc<dyn ObjectStore>,
        path: Path,
        fp_registry: Arc<FailPointRegistry>,
        system_clock: Arc<dyn SystemClock>,
        rand: Arc<DbRand>,
    }

    impl TestProvider {
        fn new(path: Path, object_store: Arc<dyn ObjectStore>) -> Self {
            let system_clock = Arc::new(DefaultSystemClock::new());
            let rand = Arc::new(DbRand::default());
            TestProvider {
                object_store,
                path,
                fp_registry: Arc::new(FailPointRegistry::new()),
                system_clock,
                rand,
            }
        }
    }

    impl TestProvider {
        async fn new_db(&self, options: Settings) -> Result<Db, crate::Error> {
            Db::builder(self.path.clone(), self.object_store.clone())
                .with_settings(options)
                .build()
                .await
        }

        async fn new_db_reader(
            &self,
            options: DbReaderOptions,
            checkpoint: Option<Uuid>,
            merge_operator: Option<MergeOperatorType>,
        ) -> Result<DbReader, SlateDBError> {
            DbReader::open_internal(
                self.manifest_store(),
                self.table_store(),
                checkpoint,
                merge_operator,
                None,
                options,
                self.system_clock.clone(),
                self.rand.clone(),
                slatedb_common::metrics::MetricsRecorderHelper::noop(),
            )
            .await
        }
    }

    fn immutable_memtable(
        recent_flushed_wal_id: u64,
        entries: Vec<RowEntry>,
    ) -> Arc<ImmutableMemtable> {
        let table = WritableKVTable::new();
        for entry in entries {
            table.put(entry);
        }
        Arc::new(ImmutableMemtable::new(table, recent_flushed_wal_id))
    }

    async fn write_wal_sst(
        table_store: Arc<TableStore>,
        wal_id: u64,
        entries: Vec<RowEntry>,
    ) -> Result<(), SlateDBError> {
        let mut writer = table_store.table_writer(SsTableId::Wal(wal_id));
        for entry in entries {
            writer.add(entry).await?;
        }
        writer.close().await?;
        Ok(())
    }

    #[derive(Debug)]
    struct InputMemtable {
        recent_flushed_wal_id: u64,
        seqs: Vec<u64>,
    }

    impl InputMemtable {
        fn new(recent_flushed_wal_id: u64, seqs: Vec<u64>) -> Self {
            Self {
                recent_flushed_wal_id,
                seqs,
            }
        }

        fn build(&self) -> Arc<ImmutableMemtable> {
            immutable_memtable(
                self.recent_flushed_wal_id,
                self.seqs
                    .iter()
                    .map(|seq| {
                        let key = format!("key-{seq:020}");
                        let value = format!("value-{seq:020}");
                        RowEntry::new_value(key.as_bytes(), value.as_bytes(), *seq)
                    })
                    .collect(),
            )
        }
    }

    #[derive(Debug)]
    struct RebuildCheckpointCase {
        last_l0_seq: u64,
        tables: Vec<InputMemtable>,
        expected: Vec<InputMemtable>,
    }

    fn test_checkpoint(manifest_id: u64, clock: Arc<dyn SystemClock>) -> crate::Checkpoint {
        crate::Checkpoint {
            id: Uuid::new_v4(),
            manifest_id,
            expire_time: None,
            create_time: clock.now(),
            name: None,
        }
    }

    #[rstest]
    #[case::skips_table_when_last_seq_is_below_last_l0_seq(RebuildCheckpointCase {
        last_l0_seq: 10,
        tables: vec![InputMemtable::new(7, vec![7, 8, 9])],
        expected: vec![],
    })]
    #[case::skips_table_when_last_seq_equals_last_l0_seq(RebuildCheckpointCase {
        last_l0_seq: 10,
        tables: vec![InputMemtable::new(7, vec![10])],
        expected: vec![],
    })]
    #[case::keeps_entire_table_when_first_seq_is_just_after_last_l0_seq(RebuildCheckpointCase {
        last_l0_seq: 10,
        tables: vec![InputMemtable::new(7, vec![11, 12])],
        expected: vec![InputMemtable::new(7, vec![11, 12])],
    })]
    #[case::filters_table_when_first_seq_equals_last_l0_seq(RebuildCheckpointCase {
        last_l0_seq: 10,
        tables: vec![InputMemtable::new(7, vec![10, 11, 12])],
        expected: vec![InputMemtable::new(7, vec![11, 12])],
    })]
    #[case::filters_table_when_only_last_row_is_newer_than_last_l0_seq(RebuildCheckpointCase {
        last_l0_seq: 10,
        tables: vec![InputMemtable::new(7, vec![8, 9, 10, 11])],
        expected: vec![InputMemtable::new(7, vec![11])],
    })]
    #[case::preserves_order_across_keep_filter_and_skip_paths(RebuildCheckpointCase {
        last_l0_seq: 20,
        tables: vec![
            InputMemtable::new(9, vec![25, 26]),
            InputMemtable::new(8, vec![20, 21, 22]),
            InputMemtable::new(7, vec![18, 19, 20]),
            InputMemtable::new(6, vec![21, 23]),
        ],
        expected: vec![
            InputMemtable::new(9, vec![25, 26]),
            InputMemtable::new(8, vec![21, 22]),
            InputMemtable::new(6, vec![21, 23]),
        ],
    })]
    #[tokio::test]
    async fn rebuild_checkpoint_state_should_filter_existing_imm_memtables_by_last_l0_seq(
        #[case] case: RebuildCheckpointCase,
    ) {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from(format!(
            "/tmp/test_db_reader_rebuild_checkpoint_state_{}",
            Uuid::new_v4()
        ));
        let test_provider = TestProvider::new(path, Arc::clone(&object_store));
        let manifest_store = test_provider.manifest_store();
        let table_store = test_provider.table_store();
        let mut stored_manifest = StoredManifest::create_new_db(
            Arc::clone(&manifest_store),
            ManifestCore::new(),
            test_provider.system_clock.clone(),
        )
        .await
        .unwrap();

        // Seed the prior checkpoint state with IMMs.
        let input_tables: Vec<_> = case.tables.iter().map(InputMemtable::build).collect();
        let prior_state = CheckpointState {
            generation: ReaderGeneration::new(
                test_checkpoint(stored_manifest.id(), test_provider.system_clock.clone()),
                stored_manifest.manifest().clone(),
            ),
            imm_memtable: input_tables.iter().cloned().collect(),
            last_wal_id: 0,
            last_remote_persisted_seq: 0,
        };
        let next_wal_sst_id = input_tables
            .iter()
            .map(|table| table.recent_flushed_wal_id())
            .max()
            .unwrap_or(0)
            + 1;

        // Advance only the manifest fields that control the rebuild filter logic.
        // We also pin replay_after_wal_id to the last known WAL so these cases
        // exercise IMM filtering without attempting any fresh WAL replay.
        let mut dirty = stored_manifest.prepare_dirty().unwrap();
        dirty.value.core.last_l0_seq = case.last_l0_seq;
        dirty.value.core.next_wal_sst_id = next_wal_sst_id;
        dirty.value.core.replay_after_wal_id = next_wal_sst_id.saturating_sub(1);
        stored_manifest.update(dirty).await.unwrap();
        let new_manifest_id = stored_manifest.id();

        // Construct just enough DbReaderInner state to call rebuild_checkpoint_state()
        // directly. skip_wal_replay keeps the test scoped to the IMM retention logic.
        let oracle = Arc::new(DbReaderOracle::new(0, DbStatusManager::new(0)));
        let recorder = slatedb_common::metrics::MetricsRecorderHelper::noop();
        let db_stats = DbStats::new(&recorder);
        let reader = Reader::new(
            Arc::clone(&table_store),
            db_stats.clone(),
            Arc::new(MonotonicClock::new(
                test_provider.system_clock.clone(),
                i64::MIN,
            )),
            oracle.clone(),
            None,
        );
        let inner = DbReaderInner {
            manifest_store,
            table_store,
            options: DbReaderOptions {
                skip_wal_replay: true,
                ..DbReaderOptions::default()
            },
            state: parking_lot::RwLock::new(Arc::new(prior_state)),
            system_clock: test_provider.system_clock.clone(),
            user_checkpoint_id: None,
            oracle,
            reader,
            db_stats,
            status_manager: DbStatusManager::new(0),
            segment_extractor: None,
            rand: test_provider.rand.clone(),
            recorder,
        };

        let rebuilt_state = inner
            .rebuild_checkpoint_state(test_checkpoint(
                new_manifest_id,
                test_provider.system_clock.clone(),
            ))
            .await
            .unwrap();

        // The rebuilt checkpoint should reflect the new manifest.
        assert_eq!(
            rebuilt_state.generation.manifest.core.last_l0_seq,
            case.last_l0_seq
        );
        assert_eq!(rebuilt_state.imm_memtable.len(), case.expected.len());

        for (rebuilt_table, expected_table) in
            rebuilt_state.imm_memtable.iter().zip(case.expected.iter())
        {
            let table = rebuilt_table.table();
            let mut iter = table.iter();
            let mut seqs = Vec::new();
            while let Some(entry) = iter.next_sync() {
                seqs.push(entry.seq);
            }

            // Every retained row must be strictly newer than the manifest's last L0 seq.
            assert_eq!(seqs, expected_table.seqs);
            assert!(seqs.iter().all(|seq| *seq > case.last_l0_seq));
            assert_eq!(
                rebuilt_table.recent_flushed_wal_id(),
                expected_table.recent_flushed_wal_id
            );

            // The filtered table's metadata should agree with the rows that survived.
            let metadata = rebuilt_table.table().metadata();
            assert_eq!(metadata.first_seq, *expected_table.seqs.first().unwrap());
            assert_eq!(metadata.last_seq, *expected_table.seqs.last().unwrap());
        }
    }

    fn build_db_reader_inner(
        test_provider: &TestProvider,
        current_core: &ManifestCore,
    ) -> DbReaderInner {
        let manifest_store = test_provider.manifest_store();
        let table_store = test_provider.table_store();

        let prior_state = CheckpointState {
            generation: ReaderGeneration::new(
                test_checkpoint(1, test_provider.system_clock.clone()),
                Manifest::initial(current_core.clone()),
            ),
            imm_memtable: [immutable_memtable(
                1,
                vec![RowEntry::new_value(b"key", b"value", 10)],
            )]
            .into_iter()
            .collect(),
            last_wal_id: 1,
            last_remote_persisted_seq: 10,
        };

        let oracle = Arc::new(DbReaderOracle::new(0, DbStatusManager::new(0)));
        let recorder = slatedb_common::metrics::MetricsRecorderHelper::noop();
        let db_stats = DbStats::new(&recorder);
        let reader = Reader::new(
            Arc::clone(&table_store),
            db_stats.clone(),
            Arc::new(MonotonicClock::new(
                test_provider.system_clock.clone(),
                i64::MIN,
            )),
            oracle.clone(),
            None,
        );
        DbReaderInner {
            manifest_store,
            table_store,
            options: DbReaderOptions::default(),
            state: parking_lot::RwLock::new(Arc::new(prior_state)),
            system_clock: test_provider.system_clock.clone(),
            user_checkpoint_id: None,
            oracle,
            reader,
            db_stats,
            status_manager: DbStatusManager::new(0),
            segment_extractor: None,
            rand: test_provider.rand.clone(),
            recorder,
        }
    }

    #[test]
    fn should_reestablish_checkpoint_when_latest_last_l0_seq_exceeds_last_remote_persisted_seq() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from(format!(
            "/tmp/test_db_reader_should_reestablish_checkpoint_{}",
            Uuid::new_v4()
        ));
        let test_provider = TestProvider::new(path, Arc::clone(&object_store));

        let mut current_core = ManifestCore::new();
        current_core.last_l0_seq = 10;
        current_core.next_wal_sst_id = 2;

        let inner = build_db_reader_inner(&test_provider, &current_core);

        assert!(!inner.should_reestablish_checkpoint(&current_core));

        let mut latest = current_core.clone();
        latest.last_l0_seq = 11;

        assert!(inner.should_reestablish_checkpoint(&latest));
    }

    #[test]
    fn should_reestablish_checkpoint_when_segments_differ() {
        // RFC-0024: per-segment compactions, drains, and segment-set changes
        // are invisible to the root-tree diff. Verify the segments comparison
        // fires on each of those shapes.
        use crate::db_state::{SortedRun, SsTableHandle, SsTableId, SsTableInfo, SsTableView};
        use crate::format::sst::SST_FORMAT_VERSION_LATEST;
        use crate::manifest::{LsmTreeState, Segment};

        fn view(seq: u64) -> SsTableView {
            SsTableView::identity(SsTableHandle::new(
                SsTableId::Compacted(ulid::Ulid::from_parts(seq, 0)),
                SST_FORMAT_VERSION_LATEST,
                SsTableInfo::default(),
            ))
        }
        fn segment_with(prefix: &'static [u8], tree: LsmTreeState) -> Segment {
            Segment {
                prefix: Bytes::from_static(prefix),
                tree: Arc::new(tree),
            }
        }
        fn tree_l0(views: Vec<SsTableView>) -> LsmTreeState {
            LsmTreeState {
                last_compacted_l0_sst_view_id: None,
                last_compacted_l0_sst_id: None,
                l0: VecDeque::from(views),
                compacted: vec![],
            }
        }

        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from(format!(
            "/tmp/test_db_reader_segments_diff_{}",
            Uuid::new_v4()
        ));
        let test_provider = TestProvider::new(path, Arc::clone(&object_store));

        let baseline_segment = segment_with(b"hour=12/", tree_l0(vec![view(1)]));
        let mut current_core = ManifestCore::new();
        current_core.segment_extractor_name = Some("hour".into());
        current_core.segments = vec![baseline_segment.clone()];

        let inner = build_db_reader_inner(&test_provider, &current_core);

        // Identical segments → no refresh.
        assert!(!inner.should_reestablish_checkpoint(&current_core));

        // New segment appears.
        let mut latest = current_core.clone();
        latest
            .segments
            .push(segment_with(b"hour=13/", tree_l0(vec![view(2)])));
        assert!(
            inner.should_reestablish_checkpoint(&latest),
            "adding a segment should retire the snapshot"
        );

        // Segment's L0 changes in place (compaction within a segment).
        let mut latest = current_core.clone();
        latest.segments = vec![segment_with(b"hour=12/", tree_l0(vec![view(1), view(3)]))];
        assert!(
            inner.should_reestablish_checkpoint(&latest),
            "segment L0 change should retire the snapshot"
        );

        // Segment's compacted list changes (sorted-run added).
        let mut latest = current_core.clone();
        latest.segments = vec![segment_with(
            b"hour=12/",
            LsmTreeState {
                last_compacted_l0_sst_view_id: None,
                last_compacted_l0_sst_id: None,
                l0: VecDeque::from(vec![view(1)]),
                compacted: vec![SortedRun {
                    id: 0,
                    sst_views: vec![view(4)],
                }],
            },
        )];
        assert!(
            inner.should_reestablish_checkpoint(&latest),
            "segment compacted change should retire the snapshot"
        );

        // Segment drained (watermark advances).
        let mut latest = current_core.clone();
        latest.segments = vec![segment_with(
            b"hour=12/",
            LsmTreeState {
                last_compacted_l0_sst_view_id: Some(ulid::Ulid::from_parts(99, 0)),
                last_compacted_l0_sst_id: None,
                l0: VecDeque::new(),
                compacted: vec![],
            },
        )];
        assert!(
            inner.should_reestablish_checkpoint(&latest),
            "segment drain marker should retire the snapshot"
        );
    }

    #[tokio::test]
    async fn should_populate_disk_cache_on_read() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_db_reader_disk_cache");

        // Write data via Db
        let db = Db::builder(path.clone(), Arc::clone(&object_store))
            .with_settings(Settings::default())
            .build()
            .await
            .unwrap();
        db.put(b"key1", b"value1").await.unwrap();
        db.flush().await.unwrap();
        db.close().await.unwrap();

        // Open a DbReader with disk caching enabled
        let cache_dir = tempfile::Builder::new()
            .prefix("dbreader_cache_test_")
            .tempdir()
            .unwrap();
        let cache_path = cache_dir.keep();

        let mut reader_opts = DbReaderOptions::default();
        reader_opts.object_store_cache_options.root_folder = Some(cache_path.clone());
        reader_opts.object_store_cache_options.part_size_bytes = 1024;

        let reader = DbReader::open(path.clone(), Arc::clone(&object_store), None, reader_opts)
            .await
            .unwrap();

        // Read data to populate the cache
        let val = reader.get(b"key1").await.unwrap();
        assert_eq!(val, Some(Bytes::from_static(b"value1")));

        // Verify the cache directory has been populated
        let entries: Vec<_> = std::fs::read_dir(&cache_path).unwrap().collect();
        assert!(
            !entries.is_empty(),
            "Expected disk cache directory to be populated after read"
        );
    }

    #[tokio::test]
    async fn should_record_metrics_with_recorder() {
        use slatedb_common::metrics::{
            lookup_metric, lookup_metric_with_labels, DefaultMetricsRecorder,
        };

        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_db_reader_metrics");

        // Write data via Db so there's an SST to read from
        let db = Db::builder(path.clone(), Arc::clone(&object_store))
            .with_settings(Settings::default())
            .build()
            .await
            .unwrap();
        db.put(b"key1", b"value1").await.unwrap();
        db.flush().await.unwrap();
        db.close().await.unwrap();

        // Open a DbReader with a metrics recorder
        let metrics_recorder = Arc::new(DefaultMetricsRecorder::new());
        let reader = DbReader::builder(path, object_store)
            .with_metrics_recorder(metrics_recorder.clone())
            .build()
            .await
            .unwrap();

        // Verify that get_requests metric is incremented
        let val = reader.get(b"key1").await.unwrap();
        assert_eq!(val, Some(Bytes::from_static(b"value1")));
        assert_eq!(
            lookup_metric_with_labels(
                &metrics_recorder,
                crate::db_stats::REQUEST_COUNT,
                &[("op", "get")]
            ),
            Some(1)
        );
        for _ in 0..1_000 {
            if lookup_metric(&metrics_recorder, crate::db_stats::READER_MANIFEST_POLLS) == Some(1) {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(
            lookup_metric(&metrics_recorder, crate::db_stats::READER_MANIFEST_POLLS,),
            Some(1)
        );
    }

    #[tokio::test]
    async fn should_record_incremental_wal_replay_metrics() {
        use slatedb_common::metrics::{lookup_metric, DefaultMetricsRecorder};

        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_db_reader_wal_replay_metrics");
        let db = Db::builder(path.clone(), Arc::clone(&object_store))
            .with_settings(Settings {
                flush_interval: None,
                compactor_options: None,
                garbage_collector_options: None,
                ..Settings::default()
            })
            .build()
            .await
            .unwrap();
        db.put_with_options(
            b"key",
            b"value",
            &PutOptions::default(),
            &WriteOptions {
                await_durable: false,
                ..WriteOptions::default()
            },
        )
        .await
        .unwrap();
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::Wal,
        })
        .await
        .unwrap();

        let metrics_recorder = Arc::new(DefaultMetricsRecorder::new());
        let reader = DbReader::builder(path, object_store)
            .with_metrics_recorder(metrics_recorder.clone())
            .build()
            .await
            .unwrap();

        assert!(
            lookup_metric(&metrics_recorder, crate::db_stats::READER_WAL_REPLAY_SSTS)
                .is_some_and(|value| value > 0)
        );
        assert!(
            lookup_metric(&metrics_recorder, crate::db_stats::READER_WAL_REPLAY_BYTES)
                .is_some_and(|value| value > 0)
        );
        assert!(lookup_metric(
            &metrics_recorder,
            crate::db_stats::READER_WAL_REPLAY_BATCHES
        )
        .is_some_and(|value| value > 0));
        assert_eq!(
            Some(1),
            lookup_metric(&metrics_recorder, crate::db_stats::READER_REPLAY_MEMTABLES)
        );
        assert_eq!(
            Some(1),
            lookup_metric(
                &metrics_recorder,
                crate::db_stats::READER_ACTIVE_CHECKPOINTS
            )
        );

        reader.close().await.unwrap();
        assert_eq!(
            Some(0),
            lookup_metric(
                &metrics_recorder,
                crate::db_stats::READER_ACTIVE_CHECKPOINTS
            )
        );
        db.close().await.unwrap();
    }

    #[tokio::test]
    async fn fixed_checkpoint_reader_should_not_manage_or_delete_its_checkpoint() {
        use slatedb_common::metrics::{lookup_metric, DefaultMetricsRecorder};

        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_fixed_db_reader_checkpoint_metric");
        let db = Db::builder(path.clone(), Arc::clone(&object_store))
            .with_settings(Settings::default())
            .build()
            .await
            .unwrap();
        db.put(b"key", b"value").await.unwrap();
        let checkpoint = db
            .create_checkpoint(CheckpointScope::All, &CheckpointOptions::default())
            .await
            .unwrap();
        db.close().await.unwrap();

        let metrics_recorder = Arc::new(DefaultMetricsRecorder::new());
        let reader = DbReader::builder(path.clone(), Arc::clone(&object_store))
            .with_checkpoint_id(checkpoint.id)
            .with_metrics_recorder(metrics_recorder.clone())
            .build()
            .await
            .unwrap();

        assert_eq!(
            Some(0),
            lookup_metric(
                &metrics_recorder,
                crate::db_stats::READER_ACTIVE_CHECKPOINTS
            )
        );
        assert_eq!(
            reader.get(b"key").await.unwrap(),
            Some(Bytes::from("value"))
        );

        reader.close().await.unwrap();
        assert_eq!(reader.status().close_reason, Some(CloseReason::Clean));
        assert!(reader.get(b"key").await.is_err());
        assert_eq!(
            Some(0),
            lookup_metric(
                &metrics_recorder,
                crate::db_stats::READER_ACTIVE_CHECKPOINTS
            )
        );

        let manifest_store = ManifestStore::new(&path, Arc::clone(&object_store));
        let manifest = manifest_store.read_latest_manifest().await.unwrap();
        assert!(manifest
            .manifest
            .core
            .find_checkpoint(checkpoint.id)
            .is_some());

        let drop_metrics_recorder = Arc::new(DefaultMetricsRecorder::new());
        let dropped_reader = DbReader::builder(path, Arc::clone(&object_store))
            .with_checkpoint_id(checkpoint.id)
            .with_metrics_recorder(drop_metrics_recorder.clone())
            .build()
            .await
            .unwrap();
        assert_eq!(
            Some(0),
            lookup_metric(
                &drop_metrics_recorder,
                crate::db_stats::READER_ACTIVE_CHECKPOINTS
            )
        );
        drop(dropped_reader);
        assert_eq!(
            Some(0),
            lookup_metric(
                &drop_metrics_recorder,
                crate::db_stats::READER_ACTIVE_CHECKPOINTS
            )
        );

        let manifest = manifest_store.read_latest_manifest().await.unwrap();
        assert!(manifest
            .manifest
            .core
            .find_checkpoint(checkpoint.id)
            .is_some());
    }

    impl TestProvider {
        fn table_store(&self) -> Arc<TableStore> {
            Arc::new(TableStore::new_with_fp_registry(
                ObjectStores::new(Arc::clone(&self.object_store), None),
                SsTableFormat::default(),
                PathResolver::new(self.path.clone()),
                Arc::clone(&self.fp_registry),
                None,
                TableStoreKind::Reader,
            ))
        }

        fn manifest_store(&self) -> Arc<ManifestStore> {
            Arc::new(ManifestStore::new(
                &self.path,
                Arc::clone(&self.object_store),
            ))
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn db_reader_get_returns_correct_merge_result_after_reestablish_from_l0() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_db_reader_merge_reestablish_from_l0");
        let clock = Arc::new(MockSystemClock::new());

        let mut test_provider = TestProvider::new(path.clone(), Arc::clone(&object_store));
        test_provider.system_clock = clock.clone();

        let merge_operator: crate::merge_operator::MergeOperatorType =
            Arc::new(crate::test_utils::StringConcatMergeOperator);

        let db = Db::builder(path.clone(), Arc::clone(&object_store))
            .with_settings(Settings {
                flush_interval: None,
                compactor_options: None,
                garbage_collector_options: None,
                ..Settings::default()
            })
            .with_system_clock(clock.clone())
            .with_merge_operator(merge_operator.clone())
            .build()
            .await
            .unwrap();

        let key = b"k";

        // Phase A: make the reader recover merge operands from WAL into replayed IMMs.
        db.merge_with_options(
            key,
            b"a",
            &MergeOptions::default(),
            &WriteOptions {
                await_durable: false,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        db.merge_with_options(
            key,
            b"b",
            &MergeOptions::default(),
            &WriteOptions {
                await_durable: false,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::Wal,
        })
        .await
        .unwrap();

        let reader = test_provider
            .new_db_reader(
                DbReaderOptions {
                    manifest_poll_interval: Duration::from_millis(100),
                    checkpoint_lifetime: Duration::from_secs(30),
                    ..DbReaderOptions::default()
                },
                None,
                Some(merge_operator),
            )
            .await
            .unwrap();

        assert_eq!(
            reader.get(key).await.unwrap(),
            Some(Bytes::from_static(b"ab"))
        );
        assert!(
            !reader.inner.state.read().imm_memtable.is_empty(),
            "reader should have replayed WAL data into immutable memtables"
        );

        // Let the reader's immediate first ticker fire and settle before phase B.
        tokio::task::yield_now().await;

        // Phase B: write newer merge operands and flush the writer memtable to L0.
        db.merge_with_options(
            key,
            b"c",
            &MergeOptions::default(),
            &WriteOptions {
                await_durable: false,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        db.merge_with_options(
            key,
            b"d",
            &MergeOptions::default(),
            &WriteOptions {
                await_durable: false,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::MemTable,
        })
        .await
        .unwrap();

        let manifest_store = test_provider.manifest_store();
        let mut stored_manifest =
            StoredManifest::load(manifest_store, test_provider.system_clock.clone())
                .await
                .unwrap();

        let start = tokio::time::Instant::now();
        loop {
            let manifest = stored_manifest.refresh().await.unwrap();
            if manifest.core.tree.l0.len() == 1 {
                break;
            }
            assert!(
                start.elapsed() < Duration::from_secs(30),
                "timed out waiting for writer manifest to include the L0 flush"
            );
            tokio::task::yield_now().await;
        }

        let timeout = Duration::from_secs(30);
        let start = tokio::time::Instant::now();
        loop {
            if reader
                .inner
                .state
                .read()
                .generation
                .manifest
                .core
                .tree
                .l0
                .len()
                == 1
            {
                break;
            }
            // The reader poller may observe the pre-flush manifest on one tick and
            // only see the new L0 on a later poll. Keep advancing the mock clock
            // until the reader reestablishes from the updated manifest.
            clock.advance(Duration::from_millis(100)).await;
            assert!(
                start.elapsed() < timeout,
                "timed out waiting for reader to reestablish from the new manifest"
            );
            tokio::task::yield_now().await;
        }

        // Correct behavior: the reader should return the full merged value after reestablish.
        assert_eq!(
            reader.get(key).await.unwrap(),
            Some(Bytes::from_static(b"abcd"))
        );

        reader.close().await.unwrap();
        db.close().await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn should_subscribe_to_durable_seq_updates() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_kv_store");
        let test_provider = TestProvider::new(path.clone(), Arc::clone(&object_store));

        let db = test_provider
            .new_db(Settings {
                l0_sst_size_bytes: 256,
                ..Settings::default()
            })
            .await
            .unwrap();

        // Write initial data and flush so reader can see it.
        db.put(b"k1", b"v1").await.unwrap();
        db.flush().await.unwrap();

        let reader_options = DbReaderOptions {
            manifest_poll_interval: Duration::from_millis(10),
            ..DbReaderOptions::default()
        };
        let reader = test_provider
            .new_db_reader(reader_options, None, None)
            .await
            .unwrap();

        let mut rx = reader.subscribe();
        let initial_seq = rx.borrow().durable_seq;
        assert!(initial_seq > 0);

        // Write more data and flush.
        db.put(b"k2", b"v2").await.unwrap();
        db.flush().await.unwrap();

        // Wait for the reader's manifest poll to pick up the new data.
        tokio::time::sleep(Duration::from_millis(20)).await;

        rx.changed().await.unwrap();
        let updated_seq = rx.borrow().durable_seq;
        assert!(
            updated_seq > initial_seq,
            "durable_seq should advance: {} > {}",
            updated_seq,
            initial_seq
        );

        reader.close().await.unwrap();
        db.close().await.unwrap();
    }

    #[tokio::test]
    async fn should_report_open_status() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_kv_store");
        let test_provider = TestProvider::new(path.clone(), Arc::clone(&object_store));

        let db = test_provider.new_db(Settings::default()).await.unwrap();
        let reader = test_provider
            .new_db_reader(DbReaderOptions::default(), None, None)
            .await
            .unwrap();

        assert_eq!(reader.status().close_reason, None);

        reader.close().await.unwrap();
        db.close().await.unwrap();
    }

    #[tokio::test]
    async fn should_report_closed_status() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_kv_store");
        let test_provider = TestProvider::new(path.clone(), Arc::clone(&object_store));

        let db = test_provider.new_db(Settings::default()).await.unwrap();
        let reader = test_provider
            .new_db_reader(DbReaderOptions::default(), None, None)
            .await
            .unwrap();

        reader.close().await.unwrap();
        assert_eq!(reader.status().close_reason, Some(CloseReason::Clean));

        db.close().await.unwrap();
    }

    #[tokio::test]
    async fn should_report_close_via_subscribe() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/tmp/test_kv_store");
        let test_provider = TestProvider::new(path.clone(), Arc::clone(&object_store));

        let db = test_provider.new_db(Settings::default()).await.unwrap();
        let reader = test_provider
            .new_db_reader(DbReaderOptions::default(), None, None)
            .await
            .unwrap();

        let mut rx = reader.subscribe();
        assert!(rx.borrow().close_reason.is_none());

        reader.close().await.unwrap();

        // The watch channel should report the close.
        rx.changed().await.unwrap();
        assert!(rx.borrow().close_reason.is_some());

        db.close().await.unwrap();
    }
}
