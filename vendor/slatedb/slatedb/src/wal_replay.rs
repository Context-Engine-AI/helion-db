use crate::db_state::SsTableId;
use crate::error::SlateDBError;
use crate::iter::{EmptyIterator, RowEntryIterator};
use crate::manifest::ManifestCore;
use crate::manifest::SsTableView;
use crate::mem_table::WritableKVTable;
use crate::sst_iter::{SstIterator, SstIteratorOptions};
use crate::tablestore::TableStore;
use crate::utils::panic_string;
use log::error;
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;
use tokio::task;
use tokio_util::task::AbortOnDropHandle;

pub(crate) struct WalReplayOptions {
    /// The number of SSTs to preload while replaying
    pub(crate) sst_batch_size: usize,

    /// The target maximum number of bytes in each returned table. WAL replay only
    /// splits between complete WAL SSTs, so a returned table may exceed this if a
    /// single WAL SST is larger.
    pub(crate) max_memtable_bytes: usize,

    /// Options to pass through to underlying SST iterators
    pub(crate) sst_iter_options: SstIteratorOptions,

    /// The minimum seq number to replay. If unset, will replay all
    /// entries after `last_l0_seq` in the manifest.
    pub(crate) min_seq: Option<u64>,
}

impl Default for WalReplayOptions {
    fn default() -> Self {
        Self {
            sst_batch_size: 4,
            max_memtable_bytes: 64 * 1024 * 1024,
            sst_iter_options: SstIteratorOptions::default(),
            min_seq: None,
        }
    }
}

pub(crate) struct ReplayedMemtable {
    pub(crate) table: WritableKVTable,
    pub(crate) last_tick: i64,
    pub(crate) last_seq: u64,
    pub(crate) last_wal_id: u64,
}

struct WalIdAndIter {
    wal_id: u64,
    iter: Box<dyn RowEntryIterator + 'static>,
}

struct IteratorHolder<T> {
    initialized: bool,
    current_iter: Option<T>,
}

impl<T> IteratorHolder<T> {
    fn new() -> Self {
        Self {
            initialized: false,
            current_iter: None,
        }
    }

    fn is_finished(&self) -> bool {
        self.initialized && self.current_iter.is_none()
    }

    fn advance(&mut self, iterator: Option<T>) {
        self.initialized = true;
        self.current_iter = iterator;
    }

    fn reset(&mut self) {
        self.initialized = false;
        self.current_iter = None;
    }
}

pub(crate) struct WalReplayIterator {
    options: WalReplayOptions,
    wal_id_range: Range<u64>,
    table_store: Arc<TableStore>,
    current_iter: IteratorHolder<WalIdAndIter>,
    next_iters: VecDeque<AbortOnDropHandle<Result<Option<WalIdAndIter>, SlateDBError>>>,
    last_tick: i64,
    last_seq: u64,
    min_seq: u64,
    next_wal_id: u64,
}

impl WalReplayIterator {
    pub(crate) async fn range(
        wal_id_range: Range<u64>,
        db_state: &ManifestCore,
        options: WalReplayOptions,
        table_store: Arc<TableStore>,
    ) -> Result<Self, SlateDBError> {
        let sst_batch_size = options.sst_batch_size;
        if sst_batch_size < 1 {
            return Err(SlateDBError::InvalidSSTBatchSize(sst_batch_size));
        }

        // load the last seq number from manifest, and use it as the starting seq number to avoid
        // replaying the entries that are already in the L0 SST. while replaying the WALs, we'll
        // update the last seq number to the max seq number, and this final `last_seq` will be passed
        // to the db_state for the further writes.
        let min_seq = options.min_seq.unwrap_or(db_state.last_l0_seq);
        let last_seq = db_state.last_l0_seq;
        let last_tick = db_state.last_l0_clock_tick;
        let next_wal_id = wal_id_range.start;

        let mut replay_iter = WalReplayIterator {
            options,
            wal_id_range,
            table_store: Arc::clone(&table_store),
            current_iter: IteratorHolder::new(),
            next_iters: VecDeque::new(),
            last_tick,
            last_seq,
            min_seq,
            next_wal_id,
        };

        for _ in 0..sst_batch_size {
            if !replay_iter.maybe_load_next_iter() {
                break;
            }
        }

        Ok(replay_iter)
    }

    fn maybe_load_next_iter(&mut self) -> bool {
        if !self.wal_id_range.contains(&self.next_wal_id)
            || self.next_iters.len() >= self.options.sst_batch_size
        {
            return false;
        }

        let next_wal_id = self.next_wal_id;
        self.next_wal_id += 1;

        async fn load_iter(
            wal_id: u64,
            sst_iter_options: SstIteratorOptions,
            table_store: Arc<TableStore>,
        ) -> Result<Option<WalIdAndIter>, SlateDBError> {
            let sst = match table_store.open_sst(&SsTableId::Wal(wal_id)).await {
                Ok(sst) => sst,
                Err(SlateDBError::EmptySSTable) => {
                    // Zero-byte WAL files are fence markers; replay them as empty WALs
                    // so the last replayed WAL ID still advances past the marker.
                    return Ok(Some(WalIdAndIter {
                        wal_id,
                        iter: Box::new(EmptyIterator::new()),
                    }));
                }
                Err(err) => return Err(err),
            };
            let iter = SstIterator::new_owned_initialized(
                ..,
                SsTableView::identity(sst),
                Arc::clone(&table_store),
                sst_iter_options,
            )
            .await?;
            Ok(iter.map(|iter| WalIdAndIter {
                wal_id,
                iter: Box::new(iter) as Box<dyn RowEntryIterator + 'static>,
            }))
        }

        let handle = task::spawn(load_iter(
            next_wal_id,
            self.options.sst_iter_options.clone(),
            Arc::clone(&self.table_store),
        ));
        self.next_iters.push_back(AbortOnDropHandle::new(handle));
        true
    }

    async fn advance_current_iter(&mut self) -> Result<(), SlateDBError> {
        let next_iter = if let Some(join_handle) = self.next_iters.pop_front() {
            match join_handle.await {
                Ok(Ok(sst_iter)) => sst_iter,
                Ok(Err(slate_err)) => return Err(slate_err),
                Err(join_err) => {
                    let task_name = format!("wal_replay[{:?}]", self.wal_id_range);
                    if let Ok(panic_err) = join_err.try_into_panic() {
                        error!(
                            "wal_replay task panicked unexpectedly. [task_name={}, panic={}]",
                            task_name,
                            panic_string(&panic_err),
                        );
                        return Err(SlateDBError::BackgroundTaskPanic(task_name));
                    }
                    return Err(SlateDBError::BackgroundTaskCancelled(task_name));
                }
            }
        } else {
            None
        };
        self.current_iter.advance(next_iter);
        Ok(())
    }

    /// Get the next table replayed from the WAL. Replay accumulates complete WAL
    /// SSTs until the returned table reaches [`WalReplayOptions::max_memtable_bytes`],
    /// unless it is the final table replayed from the WAL. The final table may even
    /// be empty since writers use an empty WAL to fence zombie writers. The empty
    /// table must still be returned so that replay logic can account for the latest
    /// WAL ID.
    ///
    /// The returned table may exceed [`WalReplayOptions::max_memtable_bytes`] when
    /// a complete WAL SST is larger than the configured target, because replay
    /// must not split a WAL SST across replayed memtables.
    pub(crate) async fn next(&mut self) -> Result<Option<ReplayedMemtable>, SlateDBError> {
        if self.current_iter.is_finished() {
            return Ok(None);
        }

        let table = WritableKVTable::new();
        let mut last_wal_id = 0;

        while !self.current_iter.is_finished() {
            if let Some(wal_id_and_iter) = &mut self.current_iter.current_iter {
                while let Some(row_entry) = wal_id_and_iter.iter.next().await? {
                    // skip the entries that are already in the L0 SST.
                    if row_entry.seq <= self.min_seq {
                        continue;
                    }

                    if let Some(ts) = row_entry.create_ts {
                        self.last_tick = self.last_tick.max(ts);
                    }
                    self.last_seq = self.last_seq.max(row_entry.seq);
                    table.put(row_entry);
                }

                last_wal_id = wal_id_and_iter.wal_id;

                let meta = table.metadata();
                let estimated_bytes = self
                    .table_store
                    .estimate_encoded_size_compacted(meta.entry_num, meta.entries_size_in_bytes);
                if !table.is_empty() && estimated_bytes >= self.options.max_memtable_bytes {
                    self.current_iter.reset();
                    break;
                }
            }

            self.maybe_load_next_iter();
            self.advance_current_iter().await?
        }

        if last_wal_id > 0 {
            Ok(Some(ReplayedMemtable {
                table,
                last_tick: self.last_tick,
                last_seq: self.last_seq,
                last_wal_id,
            }))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{WalReplayIterator, WalReplayOptions};
    use crate::bytes_range::BytesRange;
    use crate::db_state::{SsTableId, SsTableView};
    use crate::format::sst::SsTableFormat;
    use crate::iter::{IterationOrder, RowEntryIterator};
    use crate::manifest::ManifestCore;
    use crate::mem_table::WritableKVTable;
    use crate::object_stores::ObjectStores;
    use crate::proptest_util::{rng, sample};
    use crate::sst_iter::{SstIterator, SstIteratorOptions};
    use crate::tablestore::{TableStore, TableStoreKind};
    use crate::types::RowEntry;
    use crate::{error::SlateDBError, test_utils};
    use async_trait::async_trait;
    use bytes::Bytes;
    use futures::stream::BoxStream;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use object_store::{
        CopyOptions, GetOptions, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
        PutMultipartOptions, PutOptions as OS_PutOptions, PutPayload, PutResult, RenameOptions,
    };
    use proptest::test_runner::TestRng;
    use rand::Rng;
    use std::cmp::min;
    use std::collections::btree_map::Iter;
    use std::collections::BTreeMap;
    use std::fmt;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tokio::sync::Notify;
    use tokio::time::sleep;
    use tokio::time::timeout;

    impl WalReplayIterator {
        async fn all_wal_ids(
            db_state: &ManifestCore,
            options: WalReplayOptions,
            table_store: Arc<TableStore>,
        ) -> Result<Self, SlateDBError> {
            let wal_id_start = db_state.replay_after_wal_id + 1;
            let wal_id_end = table_store
                .last_seen_wal_id(db_state.replay_after_wal_id)
                .await?;
            let wal_id_range = wal_id_start..(wal_id_end + 1);
            Self::range(wal_id_range, db_state, options, table_store).await
        }
    }

    #[tokio::test]
    async fn should_replay_empty_wal() {
        let table_store = test_table_store();
        write_empty_wal(1, Arc::clone(&table_store)).await.unwrap();
        let mut replay_iter = WalReplayIterator::all_wal_ids(
            &ManifestCore::new(),
            WalReplayOptions::default(),
            Arc::clone(&table_store),
        )
        .await
        .unwrap();

        let Some(table) = replay_iter.next().await.unwrap() else {
            panic!("Expected empty table to be returned from iterator")
        };

        assert_eq!(table.last_wal_id, 1);
        assert_eq!(table.last_seq, 0);
        assert!(table.table.is_empty());
        assert_eq!(table.last_tick, i64::MIN);
        assert!(replay_iter.next().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn should_replay_zero_byte_wal_fence() {
        let table_store = test_table_store();
        table_store.write_wal_fence(1).await.unwrap();
        let mut replay_iter = WalReplayIterator::all_wal_ids(
            &ManifestCore::new(),
            WalReplayOptions::default(),
            Arc::clone(&table_store),
        )
        .await
        .unwrap();

        let Some(table) = replay_iter.next().await.unwrap() else {
            panic!("Expected empty table to be returned from iterator")
        };

        assert_eq!(table.last_wal_id, 1);
        assert_eq!(table.last_seq, 0);
        assert!(table.table.is_empty());
        assert_eq!(table.last_tick, i64::MIN);
        assert!(replay_iter.next().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn should_replay_zero_byte_wal_fence_before_real_wal() {
        let table_store = test_table_store();
        table_store.write_wal_fence(1).await.unwrap();

        let row = RowEntry::new_value(b"key", b"value", 1);
        let mut builder = table_store.wal_table_builder();
        builder.add(row.clone()).await.unwrap();
        let encoded_sst = builder.build().await.unwrap();
        table_store
            .write_sst(&SsTableId::Wal(2), &encoded_sst, false)
            .await
            .unwrap();

        let mut replay_iter = WalReplayIterator::all_wal_ids(
            &ManifestCore::new(),
            WalReplayOptions::default(),
            Arc::clone(&table_store),
        )
        .await
        .unwrap();

        let Some(replayed_table) = replay_iter.next().await.unwrap() else {
            panic!("Expected table to be returned from iterator")
        };
        assert_eq!(replayed_table.last_wal_id, 2);
        assert_eq!(replayed_table.last_seq, 1);

        let mut iter = replayed_table.table.table().iter();
        test_utils::assert_iterator(&mut iter, vec![row]).await;
        assert!(replay_iter.next().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn should_replay_all_entries() {
        let table_store = test_table_store();
        let mut rng = rng::new_test_rng(None);
        let entries = sample::table(&mut rng, 1000, 10);
        let next_wal_id = write_wals(&entries, 1, &mut rng, 200, Arc::clone(&table_store))
            .await
            .unwrap();

        let mut replay_iter = WalReplayIterator::all_wal_ids(
            &ManifestCore::new(),
            WalReplayOptions::default(),
            Arc::clone(&table_store),
        )
        .await
        .unwrap();

        let Some(replayed_table) = replay_iter.next().await.unwrap() else {
            panic!("Expected table to be returned from iterator")
        };
        assert_eq!(replayed_table.last_wal_id + 1, next_wal_id);

        let mut imm_table_iter = replayed_table.table.table().iter();
        test_utils::assert_ranged_kv_scan(
            &entries,
            &BytesRange::from(..),
            IterationOrder::Ascending,
            &mut imm_table_iter,
        )
        .await;
        assert!(replay_iter.next().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn should_enforce_max_memtable_bytes() {
        let table_store = test_table_store();
        let mut rng = rng::new_test_rng(None);
        let num_entries = 5000;
        let entries = sample::table(&mut rng, num_entries, 10);
        let next_wal_id = write_wals(&entries, 1, &mut rng, 200, Arc::clone(&table_store))
            .await
            .unwrap();

        let max_memtable_bytes = 1024;
        let mut replay_iter = WalReplayIterator::all_wal_ids(
            &ManifestCore::new(),
            WalReplayOptions {
                max_memtable_bytes,
                ..WalReplayOptions::default()
            },
            Arc::clone(&table_store),
        )
        .await
        .unwrap();

        let full_replayed_table = WritableKVTable::new();
        let mut last_wal_id = 0;
        let mut replayed_entry_count = 0;

        while let Some(replayed_table) = replay_iter.next().await.unwrap() {
            last_wal_id = replayed_table.last_wal_id;
            let metadata = replayed_table.table.metadata();
            replayed_entry_count += metadata.entry_num;

            // The last table may be less than `max_memtable_bytes`.
            if replayed_entry_count < num_entries {
                let estimated_bytes = table_store.estimate_encoded_size_compacted(
                    metadata.entry_num,
                    metadata.entries_size_in_bytes,
                );
                assert!(estimated_bytes >= max_memtable_bytes);
            }

            let mut iter = replayed_table.table.table().iter();
            while let Some(next) = iter.next().await.unwrap() {
                full_replayed_table.put(next);
            }
        }
        assert_eq!(last_wal_id + 1, next_wal_id);

        let mut full_replayed_iter = full_replayed_table.table().iter();
        test_utils::assert_ranged_kv_scan(
            &entries,
            &BytesRange::from(..),
            IterationOrder::Ascending,
            &mut full_replayed_iter,
        )
        .await;
    }

    #[tokio::test]
    async fn should_apply_max_memtable_bytes_at_wal_boundaries() {
        let table_store = test_table_store();
        let wal_entries = [
            vec![RowEntry::new_value(b"key_001", &[b'x'; 128], 1)],
            vec![RowEntry::new_value(b"key_002", &[b'x'; 128], 2)],
            vec![RowEntry::new_value(b"key_003", &[b'x'; 128], 3)],
        ];
        let single_row_size = wal_entries[0][0].estimated_size();
        let max_memtable_bytes =
            table_store.estimate_encoded_size_compacted(1, single_row_size) + 1;

        for (wal_id, entries) in wal_entries.into_iter().enumerate() {
            let mut builder = table_store.wal_table_builder();
            for entry in entries {
                builder.add(entry).await.unwrap();
            }
            let encoded_sst = builder.build().await.unwrap();
            table_store
                .write_sst(&SsTableId::Wal(wal_id as u64 + 1), &encoded_sst, false)
                .await
                .unwrap();
        }

        let mut replay_iter = WalReplayIterator::all_wal_ids(
            &ManifestCore::new(),
            WalReplayOptions {
                max_memtable_bytes,
                ..WalReplayOptions::default()
            },
            Arc::clone(&table_store),
        )
        .await
        .unwrap();

        let mut replayed_last_wal_ids = Vec::new();
        let mut replayed_table_sizes = Vec::new();
        let mut replayed_seqs = Vec::new();

        while let Some(replayed_table) = replay_iter.next().await.unwrap() {
            replayed_last_wal_ids.push(replayed_table.last_wal_id);
            let metadata = replayed_table.table.metadata();
            replayed_table_sizes.push(table_store.estimate_encoded_size_compacted(
                metadata.entry_num,
                metadata.entries_size_in_bytes,
            ));
            let mut iter = replayed_table.table.table().iter();
            while let Some(next) = iter.next().await.unwrap() {
                replayed_seqs.push(next.seq);
            }
        }

        assert_eq!(replayed_last_wal_ids, vec![2, 3]);
        assert!(
            replayed_table_sizes[0] > max_memtable_bytes,
            "first replayed table should exceed the target rather than split a WAL SST"
        );
        assert_eq!(replayed_seqs, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn should_not_split_one_commit_seq_across_replayed_memtables() {
        let table_store = test_table_store();
        let commit_seq = 42;

        // Simulate one committed write batch. Every row gets the same commit
        // sequence, which means replay must not split these rows into separate
        // memtable layers.
        let entries = (0..8)
            .map(|i| {
                RowEntry::new_value(format!("key_{i:03}").as_bytes(), &[b'x'; 128], commit_seq)
            })
            .collect::<Vec<_>>();

        // Size replayed memtables so one real row fits, but the second row
        // overflows into the next replayed memtable.
        let max_memtable_bytes =
            table_store.estimate_encoded_size_compacted(1, entries[0].estimated_size());

        // Use the real WAL SST builder so the fixture matches WAL flushes.
        let mut builder = table_store.wal_table_builder();
        for entry in entries {
            builder.add(entry).await.unwrap();
        }
        let encoded_sst = builder.build().await.unwrap();
        table_store
            .write_sst(&SsTableId::Wal(1), &encoded_sst, false)
            .await
            .unwrap();

        // Replay the single WAL SST into in-memory tables. If the replay code
        // can split a single commit sequence, it will do so here.
        let mut replay_iter = WalReplayIterator::all_wal_ids(
            &ManifestCore::new(),
            WalReplayOptions {
                max_memtable_bytes,
                ..WalReplayOptions::default()
            },
            Arc::clone(&table_store),
        )
        .await
        .unwrap();

        let mut replayed_seq_ranges = Vec::new();
        while let Some(replayed_table) = replay_iter.next().await.unwrap() {
            let metadata = replayed_table.table.metadata();
            replayed_seq_ranges.push((metadata.first_seq, metadata.last_seq));
        }

        // This guards against producing multiple replayed memtables with the same
        // sequence range, which can make later replay logic treat part of the write
        // batch as already committed.
        assert_eq!(
            replayed_seq_ranges,
            vec![(commit_seq, commit_seq)],
            "WAL replay split one commit seq across replayed memtables: {replayed_seq_ranges:?}"
        );
    }

    #[tokio::test]
    async fn should_replay_memtables_in_sequence_order() {
        let table_store = test_table_store();

        // Write one WAL with entries whose sequence numbers do not match key
        // order. Replay must not expose a later memtable whose sequence range
        // starts before the previous memtable's sequence range ends.
        let entries = vec![
            RowEntry::new_value(b"key_000", &[b'x'; 128], 100),
            RowEntry::new_value(b"key_001", &[b'x'; 128], 10),
            RowEntry::new_value(b"key_002", &[b'x'; 128], 110),
        ];

        // Size replayed memtables so one real row fits, but the second row
        // overflows into the next replayed memtable.
        let max_memtable_bytes =
            table_store.estimate_encoded_size_compacted(1, entries[0].estimated_size());

        // Use the real WAL SST builder so replay sees the same entry order as a
        // flushed WAL.
        let mut builder = table_store.wal_table_builder();
        for entry in entries {
            builder.add(entry).await.unwrap();
        }
        let encoded_sst = builder.build().await.unwrap();
        table_store
            .write_sst(&SsTableId::Wal(1), &encoded_sst, false)
            .await
            .unwrap();

        // Replay the single WAL SST into in-memory tables.
        let mut replay_iter = WalReplayIterator::all_wal_ids(
            &ManifestCore::new(),
            WalReplayOptions {
                max_memtable_bytes,
                ..WalReplayOptions::default()
            },
            Arc::clone(&table_store),
        )
        .await
        .unwrap();

        let mut replayed_seq_ranges = Vec::new();
        while let Some(replayed_table) = replay_iter.next().await.unwrap() {
            let metadata = replayed_table.table.metadata();
            replayed_seq_ranges.push((metadata.first_seq, metadata.last_seq));
        }

        // This guards against returning the seq=10 row in a later replayed
        // memtable after already returning seq=100.
        for adjacent in replayed_seq_ranges.windows(2) {
            let previous_last_seq = adjacent[0].1;
            let later_first_seq = adjacent[1].0;
            assert!(
                later_first_seq >= previous_last_seq,
                "WAL replay returned out-of-order memtable sequence ranges: {replayed_seq_ranges:?}"
            );
        }
    }

    #[tokio::test]
    async fn should_only_replay_wals_after_last_l0_flushed_wal_id() {
        let table_store = test_table_store();
        let mut rng = rng::new_test_rng(None);
        let compacted_entries = sample::table(&mut rng, 1000, 10);
        let mut next_wal_id = 1;

        next_wal_id = write_wals(
            &compacted_entries,
            next_wal_id,
            &mut rng,
            200,
            Arc::clone(&table_store),
        )
        .await
        .unwrap();

        let replay_after_wal_id = next_wal_id - 1;
        let non_compacted_entries = sample::table(&mut rng, 1000, 10);
        next_wal_id = write_wals(
            &non_compacted_entries,
            next_wal_id,
            &mut rng,
            200,
            Arc::clone(&table_store),
        )
        .await
        .unwrap();

        let mut db_state = ManifestCore::new();
        db_state.replay_after_wal_id = replay_after_wal_id;
        db_state.next_wal_sst_id = replay_after_wal_id + 1;

        let mut replay_iter = WalReplayIterator::all_wal_ids(
            &db_state,
            WalReplayOptions::default(),
            Arc::clone(&table_store),
        )
        .await
        .unwrap();

        let Some(replayed_table) = replay_iter.next().await.unwrap() else {
            panic!("Expected table to be returned from iterator")
        };
        assert_eq!(replayed_table.last_wal_id + 1, next_wal_id);

        let mut imm_table_iter = replayed_table.table.table().iter();
        test_utils::assert_ranged_kv_scan(
            &non_compacted_entries,
            &BytesRange::from(..),
            IterationOrder::Ascending,
            &mut imm_table_iter,
        )
        .await;
        assert!(replay_iter.next().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn should_replay_wals_after_min_seq() {
        let table_store = test_table_store();
        let mut rng = rng::new_test_rng(None);
        let entries = sample::table(&mut rng, 1000, 10);
        let next_wal_id = write_wals(&entries, 1, &mut rng, 200, Arc::clone(&table_store))
            .await
            .unwrap();

        // Set min_seq to skip the first half of entries
        let min_seq = 500;
        let mut db_state = ManifestCore::new();
        db_state.last_l0_seq = min_seq;
        db_state.last_l0_clock_tick = 0;

        let mut replay_iter = WalReplayIterator::all_wal_ids(
            &db_state,
            WalReplayOptions::default(),
            Arc::clone(&table_store),
        )
        .await
        .unwrap();

        let Some(replayed_table) = replay_iter.next().await.unwrap() else {
            panic!("Expected table to be returned from iterator")
        };
        assert_eq!(replayed_table.last_wal_id + 1, next_wal_id);

        // Verify that only entries with seq > min_seq are replayed
        let mut imm_table_iter = replayed_table.table.table().iter();
        let mut replayed_entries = BTreeMap::new();
        let mut total = 0;
        while let Some(entry) = imm_table_iter.next().await.unwrap() {
            assert!(entry.seq > min_seq);
            replayed_entries.insert(entry.key.clone(), entry.value);
            total += 1;
        }
        assert_eq!(total, 500);
    }

    #[tokio::test]
    async fn should_preserve_replay_results_for_larger_prefetch_batches() {
        let fixture = WalReplayExperimentFixture::new().await;

        let baseline = replay_fixture(&fixture, 4, Duration::ZERO).await;
        let batch_8 = replay_fixture(&fixture, 8, Duration::ZERO).await;
        let batch_16 = replay_fixture(&fixture, 16, Duration::ZERO).await;

        assert_eq!(baseline.rows, fixture.expected_rows);
        assert_eq!(batch_8.rows, baseline.rows);
        assert_eq!(batch_16.rows, baseline.rows);
        assert_eq!(baseline.last_wal_ids, vec![fixture.last_wal_id]);
        assert_eq!(batch_8.last_wal_ids, baseline.last_wal_ids);
        assert_eq!(batch_16.last_wal_ids, baseline.last_wal_ids);
        assert_eq!(baseline.last_seq, fixture.expected_last_seq);
        assert_eq!(batch_8.last_seq, baseline.last_seq);
        assert_eq!(batch_16.last_seq, baseline.last_seq);
        assert_eq!(baseline.last_tick, fixture.expected_last_tick);
        assert_eq!(batch_8.last_tick, baseline.last_tick);
        assert_eq!(batch_16.last_tick, baseline.last_tick);
    }

    #[tokio::test]
    async fn should_compare_prefetch_batch_sizes_under_controlled_io_latency() {
        let fixture = WalReplayExperimentFixture::new().await;
        let read_latency = Duration::from_millis(2);

        let batch_4 = replay_fixture(&fixture, 4, read_latency).await;
        let batch_8 = replay_fixture(&fixture, 8, read_latency).await;
        let batch_16 = replay_fixture(&fixture, 16, read_latency).await;

        for result in [&batch_4, &batch_8, &batch_16] {
            assert_eq!(result.rows, fixture.expected_rows);
            assert_eq!(result.last_wal_ids, vec![fixture.last_wal_id]);
            assert_eq!(result.last_seq, fixture.expected_last_seq);
            assert_eq!(result.last_tick, fixture.expected_last_tick);
            assert!(
                result.max_active_get_opts <= result.batch_size,
                "object-store get_opts calls exceeded prefetch window: {result:?}"
            );
        }

        for result in [&batch_4, &batch_8, &batch_16] {
            println!(
                "wal_replay_prefetch_experiment batch={} elapsed_ms={} get_opts={} max_active_get_opts={} initial_prefetched_wals={} fixture_max_encoded_wal_bytes={}",
                result.batch_size,
                result.elapsed.as_millis(),
                result.total_get_opts,
                result.max_active_get_opts,
                result.initial_prefetched_wals,
                result.fixture_max_encoded_wal_bytes
            );
        }
    }

    #[tokio::test]
    async fn should_abort_queued_prefetched_wal_reads_on_drop() {
        let fixture = WalReplayExperimentFixture::new().await;
        let gate = Arc::new(ReadGate::new());
        let measured_store = Arc::new(LatencyRecordingObjectStore::new_blocked(
            Arc::clone(&fixture.object_store),
            Arc::clone(&gate),
        ));
        let object_store: Arc<dyn ObjectStore> = measured_store.clone();
        let table_store = test_table_store_with_object_store(object_store);
        let replay_iter = WalReplayIterator::range(
            1..fixture.last_wal_id + 1,
            &ManifestCore::new(),
            WalReplayOptions {
                sst_batch_size: 4,
                min_seq: Some(5),
                ..WalReplayOptions::default()
            },
            Arc::clone(&table_store),
        )
        .await
        .unwrap();

        wait_for_active_reads(&measured_store, 4).await;
        drop(replay_iter);
        assert_no_active_reads_after_drop(&measured_store, &gate).await;
    }

    #[tokio::test]
    async fn should_abort_prefetched_wal_read_when_next_future_is_cancelled() {
        let fixture = WalReplayExperimentFixture::new().await;
        let gate = Arc::new(ReadGate::new());
        let measured_store = Arc::new(LatencyRecordingObjectStore::new_blocked(
            Arc::clone(&fixture.object_store),
            Arc::clone(&gate),
        ));
        let object_store: Arc<dyn ObjectStore> = measured_store.clone();
        let table_store = test_table_store_with_object_store(object_store);
        let mut replay_iter = WalReplayIterator::range(
            1..fixture.last_wal_id + 1,
            &ManifestCore::new(),
            WalReplayOptions {
                sst_batch_size: 1,
                min_seq: Some(5),
                ..WalReplayOptions::default()
            },
            Arc::clone(&table_store),
        )
        .await
        .unwrap();

        wait_for_active_reads(&measured_store, 1).await;
        let mut next_future = Box::pin(replay_iter.next());
        assert!(
            timeout(Duration::from_millis(10), &mut next_future)
                .await
                .is_err(),
            "blocked WAL read unexpectedly completed before the gate opened"
        );
        drop(next_future);
        assert_no_active_reads_after_drop(&measured_store, &gate).await;
    }

    #[tokio::test]
    async fn should_abort_sst_block_fetch_on_drop_in_ascending_order() {
        assert_sst_block_fetch_aborted_on_drop(IterationOrder::Ascending).await;
    }

    #[tokio::test]
    async fn should_abort_sst_block_fetch_on_drop_in_descending_order() {
        assert_sst_block_fetch_aborted_on_drop(IterationOrder::Descending).await;
    }

    async fn assert_sst_block_fetch_aborted_on_drop(order: IterationOrder) {
        let gate = Arc::new(ReadGate::new());
        gate.release_all();
        let measured_store = Arc::new(LatencyRecordingObjectStore::new_blocked(
            Arc::new(InMemory::new()),
            Arc::clone(&gate),
        ));
        let object_store: Arc<dyn ObjectStore> = measured_store.clone();
        let table_store = test_table_store_with_object_store(object_store);
        let value = vec![b'x'; 8192];
        let entries = [
            RowEntry::new_value(b"alpha", &value, 1),
            RowEntry::new_value(b"bravo", &value, 2),
            RowEntry::new_value(b"charlie", &value, 3),
        ];
        write_wal_entries(1, &entries, Arc::clone(&table_store))
            .await
            .unwrap();
        let table = table_store.open_sst(&SsTableId::Wal(1)).await.unwrap();
        let mut iterator = SstIterator::new_owned_initialized(
            ..,
            SsTableView::identity(table),
            Arc::clone(&table_store),
            SstIteratorOptions {
                cache_blocks: false,
                cache_metadata: false,
                eager_spawn: false,
                order,
                ..SstIteratorOptions::default()
            },
        )
        .await
        .unwrap()
        .expect("fixture has a non-empty SST");
        gate.released.store(false, Ordering::SeqCst);
        let mut read_future = Box::pin(async { while iterator.next().await.unwrap().is_some() {} });
        assert!(
            timeout(Duration::from_millis(10), &mut read_future)
                .await
                .is_err(),
            "SST iteration completed while its block-read gate was closed"
        );
        wait_for_active_reads(&measured_store, 1).await;

        drop(read_future);
        drop(iterator);

        assert_no_active_reads_after_drop(&measured_store, &gate).await;
    }

    fn test_table_store() -> Arc<TableStore> {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        test_table_store_with_object_store(object_store)
    }

    fn test_table_store_with_object_store(object_store: Arc<dyn ObjectStore>) -> Arc<TableStore> {
        let path = Path::from("/tmp/test_kv_store");
        Arc::new(TableStore::new(
            ObjectStores::new(object_store.clone(), None),
            SsTableFormat::default(),
            path,
            None,
            TableStoreKind::Main,
        ))
    }

    #[derive(Debug)]
    struct WalReplayExperimentFixture {
        object_store: Arc<dyn ObjectStore>,
        expected_rows: Vec<RowEntry>,
        expected_last_seq: u64,
        expected_last_tick: i64,
        last_wal_id: u64,
        max_encoded_wal_bytes: usize,
    }

    impl WalReplayExperimentFixture {
        async fn new() -> Self {
            let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
            let table_store = test_table_store_with_object_store(Arc::clone(&object_store));
            let min_seq = 5;
            let wal_entries = vec![
                vec![RowEntry::new_value(b"alpha", b"skip-old-alpha", 1).with_create_ts(10)],
                vec![RowEntry::new_value(b"bravo", b"skip-old-bravo", 2).with_create_ts(20)],
                vec![RowEntry::new_value(b"charlie", b"skip-old-charlie", 3).with_create_ts(30)],
                vec![RowEntry::new_value(b"delta", b"skip-old-delta", 4).with_create_ts(40)],
                vec![RowEntry::new_value(b"echo", b"skip-old-echo", 5).with_create_ts(50)],
                vec![RowEntry::new_value(b"alpha", b"v1", 6).with_create_ts(60)],
                vec![RowEntry::new_value(b"bravo", b"v1", 7).with_create_ts(70)],
                vec![RowEntry::new_value(b"alpha", b"v2", 8).with_create_ts(80)],
                vec![RowEntry::new_value(b"charlie", b"v1", 9).with_create_ts(90)],
                vec![RowEntry::new_tombstone(b"bravo", 10).with_create_ts(100)],
                Vec::new(),
                vec![RowEntry::new_value(b"delta", b"v1", 11).with_create_ts(110)],
                vec![RowEntry::new_value(b"echo", b"v1", 12).with_create_ts(120)],
                vec![RowEntry::new_value(b"foxtrot", b"v1", 13).with_create_ts(130)],
                vec![RowEntry::new_value(b"golf", b"v1", 14).with_create_ts(140)],
                vec![RowEntry::new_tombstone(b"alpha", 15).with_create_ts(150)],
                vec![RowEntry::new_value(b"hotel", b"v1", 16).with_create_ts(160)],
                vec![RowEntry::new_value(b"india", b"v1", 17).with_create_ts(170)],
                vec![RowEntry::new_value(b"juliet", b"v1", 18).with_create_ts(180)],
                vec![RowEntry::new_value(b"kilo", b"v1", 19).with_create_ts(190)],
                vec![RowEntry::new_value(b"lima", b"v1", 20).with_create_ts(200)],
                vec![RowEntry::new_value(b"mike", b"v1", 21).with_create_ts(210)],
                vec![RowEntry::new_value(b"november", b"v1", 22).with_create_ts(220)],
                Vec::new(),
            ];

            let expected_table = WritableKVTable::new();
            let mut expected_last_seq = 0;
            let mut expected_last_tick = i64::MIN;
            let mut max_encoded_wal_bytes = 0;

            for (offset, entries) in wal_entries.iter().enumerate() {
                let wal_id = offset as u64 + 1;
                if entries.is_empty() {
                    table_store.write_wal_fence(wal_id).await.unwrap();
                    continue;
                }
                let encoded_len = write_wal_entries(wal_id, entries, Arc::clone(&table_store))
                    .await
                    .unwrap();
                max_encoded_wal_bytes = max_encoded_wal_bytes.max(encoded_len);

                for entry in entries {
                    if entry.seq > min_seq {
                        if let Some(create_ts) = entry.create_ts {
                            expected_last_tick = expected_last_tick.max(create_ts);
                        }
                        expected_last_seq = expected_last_seq.max(entry.seq);
                        expected_table.put(entry.clone());
                    }
                }
            }

            Self {
                object_store,
                expected_rows: collect_table_rows(&expected_table).await,
                expected_last_seq,
                expected_last_tick,
                last_wal_id: wal_entries.len() as u64,
                max_encoded_wal_bytes,
            }
        }
    }

    #[derive(Debug)]
    struct ReplayExperimentResult {
        batch_size: usize,
        rows: Vec<RowEntry>,
        last_wal_ids: Vec<u64>,
        last_seq: u64,
        last_tick: i64,
        elapsed: Duration,
        total_get_opts: usize,
        max_active_get_opts: usize,
        initial_prefetched_wals: usize,
        fixture_max_encoded_wal_bytes: usize,
    }

    async fn replay_fixture(
        fixture: &WalReplayExperimentFixture,
        batch_size: usize,
        read_latency: Duration,
    ) -> ReplayExperimentResult {
        let measured_store = Arc::new(LatencyRecordingObjectStore::new(
            Arc::clone(&fixture.object_store),
            read_latency,
        ));
        let object_store: Arc<dyn ObjectStore> = measured_store.clone();
        let table_store = test_table_store_with_object_store(object_store);
        let options = WalReplayOptions {
            sst_batch_size: batch_size,
            min_seq: Some(5),
            ..WalReplayOptions::default()
        };

        let start = Instant::now();
        let mut replay_iter = WalReplayIterator::range(
            1..fixture.last_wal_id + 1,
            &ManifestCore::new(),
            options,
            Arc::clone(&table_store),
        )
        .await
        .unwrap();
        let initial_prefetched_wals = replay_iter.next_iters.len();
        let mut rows = Vec::new();
        let mut last_wal_ids = Vec::new();
        let mut last_seq = 0;
        let mut last_tick = i64::MIN;

        while let Some(replayed_table) = replay_iter.next().await.unwrap() {
            last_wal_ids.push(replayed_table.last_wal_id);
            last_seq = replayed_table.last_seq;
            last_tick = replayed_table.last_tick;
            rows.extend(collect_table_rows(&replayed_table.table).await);
        }

        ReplayExperimentResult {
            batch_size,
            rows,
            last_wal_ids,
            last_seq,
            last_tick,
            elapsed: start.elapsed(),
            total_get_opts: measured_store.total_get_opts(),
            max_active_get_opts: measured_store.max_active_get_opts(),
            initial_prefetched_wals,
            fixture_max_encoded_wal_bytes: fixture.max_encoded_wal_bytes,
        }
    }

    async fn collect_table_rows(table: &WritableKVTable) -> Vec<RowEntry> {
        let mut iter = table.table().iter();
        iter.init().await.unwrap();
        let mut rows = Vec::new();
        while let Some(entry) = iter.next().await.unwrap() {
            rows.push(entry);
        }
        rows
    }

    #[derive(Debug)]
    struct LatencyRecordingObjectStore {
        inner: Arc<dyn ObjectStore>,
        delay: Duration,
        gate: Option<Arc<ReadGate>>,
        active_reads: AtomicUsize,
        max_active_get_opts: AtomicUsize,
        total_get_opts: AtomicUsize,
        reads_changed: Notify,
    }

    impl LatencyRecordingObjectStore {
        fn new(inner: Arc<dyn ObjectStore>, delay: Duration) -> Self {
            Self {
                inner,
                delay,
                gate: None,
                active_reads: AtomicUsize::new(0),
                max_active_get_opts: AtomicUsize::new(0),
                total_get_opts: AtomicUsize::new(0),
                reads_changed: Notify::new(),
            }
        }

        fn new_blocked(inner: Arc<dyn ObjectStore>, gate: Arc<ReadGate>) -> Self {
            Self {
                inner,
                delay: Duration::ZERO,
                gate: Some(gate),
                active_reads: AtomicUsize::new(0),
                max_active_get_opts: AtomicUsize::new(0),
                total_get_opts: AtomicUsize::new(0),
                reads_changed: Notify::new(),
            }
        }

        fn total_get_opts(&self) -> usize {
            self.total_get_opts.load(Ordering::SeqCst)
        }

        fn max_active_get_opts(&self) -> usize {
            self.max_active_get_opts.load(Ordering::SeqCst)
        }

        fn enter_read(&self) -> ActiveReadGuard<'_> {
            self.total_get_opts.fetch_add(1, Ordering::SeqCst);
            let active = self.active_reads.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active_get_opts.fetch_max(active, Ordering::SeqCst);
            self.reads_changed.notify_waiters();
            ActiveReadGuard { store: self }
        }

        async fn wait_for_active_reads(&self, expected: usize) {
            loop {
                let notified = self.reads_changed.notified();
                if self.active_reads.load(Ordering::SeqCst) >= expected {
                    return;
                }
                notified.await;
            }
        }

        async fn wait_for_no_active_reads(&self) {
            loop {
                let notified = self.reads_changed.notified();
                if self.active_reads.load(Ordering::SeqCst) == 0 {
                    return;
                }
                notified.await;
            }
        }
    }

    impl fmt::Display for LatencyRecordingObjectStore {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "LatencyRecordingObjectStore({})", self.inner)
        }
    }

    struct ActiveReadGuard<'a> {
        store: &'a LatencyRecordingObjectStore,
    }

    impl Drop for ActiveReadGuard<'_> {
        fn drop(&mut self) {
            self.store.active_reads.fetch_sub(1, Ordering::SeqCst);
            self.store.reads_changed.notify_waiters();
        }
    }

    #[derive(Debug)]
    struct ReadGate {
        released: AtomicBool,
        notify: Notify,
    }

    impl ReadGate {
        fn new() -> Self {
            Self {
                released: AtomicBool::new(false),
                notify: Notify::new(),
            }
        }

        async fn wait(&self) {
            loop {
                let notified = self.notify.notified();
                if self.released.load(Ordering::SeqCst) {
                    return;
                }
                notified.await;
            }
        }

        fn release_all(&self) {
            self.released.store(true, Ordering::SeqCst);
            self.notify.notify_waiters();
        }
    }

    async fn wait_for_active_reads(store: &LatencyRecordingObjectStore, expected: usize) {
        timeout(
            Duration::from_secs(1),
            store.wait_for_active_reads(expected),
        )
        .await
        .unwrap_or_else(|_| {
            panic!(
                "timed out waiting for {expected} active reads; observed {}",
                store.active_reads.load(Ordering::SeqCst)
            )
        });
    }

    async fn assert_no_active_reads_after_drop(
        store: &LatencyRecordingObjectStore,
        gate: &ReadGate,
    ) {
        if timeout(Duration::from_secs(1), store.wait_for_no_active_reads())
            .await
            .is_err()
        {
            let active_reads = store.active_reads.load(Ordering::SeqCst);
            gate.release_all();
            store.wait_for_no_active_reads().await;
            panic!("prefetched WAL reads leaked after cancellation; active_reads={active_reads}");
        }
    }

    #[async_trait]
    impl ObjectStore for LatencyRecordingObjectStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: OS_PutOptions,
        ) -> object_store::Result<PutResult> {
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> object_store::Result<object_store::GetResult> {
            let _guard = self.enter_read();
            if let Some(gate) = &self.gate {
                gate.wait().await;
            }
            if !self.delay.is_zero() {
                sleep(self.delay).await;
            }
            self.inner.get_opts(location, options).await
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, object_store::Result<Path>>,
        ) -> BoxStream<'static, object_store::Result<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
            self.inner.list(prefix)
        }

        fn list_with_offset(
            &self,
            prefix: Option<&Path>,
            offset: &Path,
        ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
            self.inner.list_with_offset(prefix, offset)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> object_store::Result<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: CopyOptions,
        ) -> object_store::Result<()> {
            self.inner.copy_opts(from, to, options).await
        }

        async fn rename_opts(
            &self,
            from: &Path,
            to: &Path,
            options: RenameOptions,
        ) -> object_store::Result<()> {
            self.inner.rename_opts(from, to, options).await
        }
    }

    /// Write a sequence of WALs with a random (bounded) number of entries.
    /// Return the ID of the next WAL.
    async fn write_wals(
        entries: &BTreeMap<Bytes, Bytes>,
        next_wal_id: u64,
        rng: &mut TestRng,
        max_wal_entries: usize,
        table_store: Arc<TableStore>,
    ) -> Result<u64, SlateDBError> {
        let mut iter = entries.iter();
        let mut next_seq = 1;
        let mut total_wal_entries = 0;
        let mut next_wal_id = next_wal_id;

        while total_wal_entries < entries.len() {
            let wal_entries = min(
                entries.len() - total_wal_entries,
                rng.random_range(0..max_wal_entries),
            );
            next_seq = write_wal(
                next_wal_id,
                next_seq,
                &mut iter,
                wal_entries,
                Arc::clone(&table_store),
            )
            .await?;
            next_wal_id += 1;
            total_wal_entries += wal_entries;
        }
        Ok(next_wal_id)
    }

    async fn write_empty_wal(
        wal_id: u64,
        table_store: Arc<TableStore>,
    ) -> Result<(), SlateDBError> {
        let empty_entries = BTreeMap::new();
        let mut empty_iter = empty_entries.iter();
        let _ = write_wal(wal_id, 0, &mut empty_iter, 0, table_store).await?;
        Ok(())
    }

    async fn write_wal_entries(
        wal_id: u64,
        entries: &[RowEntry],
        table_store: Arc<TableStore>,
    ) -> Result<usize, SlateDBError> {
        let mut builder = table_store.wal_table_builder();
        for entry in entries {
            builder.add(entry.clone()).await?;
        }
        let encoded_sst = builder.build().await?;
        let encoded_len = encoded_sst.remaining_len();
        table_store
            .write_sst(&SsTableId::Wal(wal_id), &encoded_sst, false)
            .await?;
        Ok(encoded_len)
    }

    async fn write_wal(
        wal_id: u64,
        next_seq: u64,
        entries: &mut Iter<'_, Bytes, Bytes>,
        max_entries: usize,
        table_store: Arc<TableStore>,
    ) -> Result<u64, SlateDBError> {
        let mut writer = table_store.table_writer(SsTableId::Wal(wal_id));
        let mut next_seq = next_seq;
        while next_seq < next_seq + (max_entries as u64) {
            let Some((key, value)) = entries.next() else {
                break;
            };
            writer
                .add(RowEntry::new_value(key, value, next_seq))
                .await?;
            next_seq += 1;
        }
        writer.close().await?;
        Ok(next_seq)
    }
}
