use crate::batch::WriteBatchIterator;
use crate::block_iterator::DataBlockIterator;
use crate::bytes_range::BytesRange;
use crate::clock::MonotonicClock;
use crate::config::{DurabilityLevel, ReadOptions, ScanOptions};
use crate::db_iter::{apply_filters, DbRecencyIterator};
use crate::db_stats::DbStats;
use crate::filter_policy::FilterQuery;
use crate::format::block::Block;
use crate::iter::RowEntryIterator;
use crate::manifest::{ManifestCore, Segment};
use crate::mem_table::{ImmutableMemtable, KVTable};
use crate::merge_operator::{instrument_merge_operator, MergeOperatorType};
use crate::oracle::Oracle;
use crate::partitioned_keyspace;
use crate::segment_iterator::{build_segment_iter, SegmentScanContext};
use crate::sorted_run_iterator::SortedRunIterator;
use crate::sst_iter::{SstIterator, SstIteratorOptions};
use crate::tablestore::TableStore;
use crate::types::{KeyValue, RowEntry, ValueDeletable};
use crate::{error::SlateDBError, DbIterator};

use bytes::Bytes;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;

pub(crate) trait DbStateReader {
    fn memtable(&self) -> Arc<KVTable>;
    /// Returns immutable memtables newest-first. The iterator form permits
    /// read-only replicas to use a structurally shared persistent chain.
    fn imm_memtables(&self) -> Box<dyn Iterator<Item = Arc<ImmutableMemtable>> + Send + '_>;
    fn core(&self) -> &ManifestCore;
}

struct IteratorSources {
    write_batch_iter: Option<WriteBatchIterator>,
    mem_iters: Vec<Box<dyn RowEntryIterator + 'static>>,
    /// Single chain over the on-disk LSM state. Each segment's L0 and
    /// sorted-run iterators are merged together inside the chain (see
    /// `SegmentRangeIterator`); the top-level scan path treats the
    /// chain as one merge arm alongside `mem_iters` and `write_batch_iter`.
    segment_iter: Box<dyn RowEntryIterator + 'static>,
}

/// Context for [`Reader::scan_with_options`].
///
/// Groups the state each scan needs from its caller.
pub(crate) struct ScanContext<'a> {
    /// Read-only view over in-memory state (memtables) and access to on-disk
    /// data (level-0 SSTs and compacted sorted runs) used to construct
    /// iterators.
    pub(crate) db_state: &'a (dyn DbStateReader + Sync),
    /// Optional write batch iterator to include in the merged scan. Set only
    /// when the scan is running inside a transaction.
    pub(crate) write_batch_iter: Option<WriteBatchIterator>,
    /// Optional upper bound on sequence number visibility. When set, entries
    /// with a greater sequence number are filtered out by the iterator
    /// construction. Used by snapshots and transactions.
    pub(crate) max_seq: Option<u64>,
    /// Optional scan prefix forwarded to per-SST filter evaluation. When set,
    /// filters are probed with a prefix query so SSTs that do not contain the
    /// prefix can be skipped.
    pub(crate) prefix: Option<Bytes>,
}

pub(crate) struct Reader {
    pub(crate) table_store: Arc<TableStore>,
    pub(crate) db_stats: DbStats,
    #[allow(dead_code)] // unused during DST
    pub(crate) mono_clock: Arc<MonotonicClock>,
    pub(crate) oracle: Arc<dyn Oracle>,
    pub(crate) read_merge_operator: Option<MergeOperatorType>,
}

impl Reader {
    pub(crate) fn new(
        table_store: Arc<TableStore>,
        db_stats: DbStats,
        mono_clock: Arc<MonotonicClock>,
        oracle: Arc<dyn Oracle>,
        merge_operator: Option<MergeOperatorType>,
    ) -> Self {
        let read_merge_operator = merge_operator.map(|merge_operator| {
            instrument_merge_operator(
                merge_operator,
                db_stats.merge_operator_read_operands.clone(),
            )
        });

        Self {
            table_store,
            db_stats,
            mono_clock,
            oracle,
            read_merge_operator,
        }
    }

    /// Determines the maximum sequence number for read operations (get and scan). Read operations will filter
    /// out entries with sequence numbers greater than the returned value.
    ///
    /// The method considers:
    /// - User-provided sequence number (e.g., from a Snapshot)
    /// - Durability requirements (Remote vs Local)
    /// - Whether dirty reads are allowed
    ///
    /// Returns the minimum sequence number that satisfies all constraints, or None (read without filtering max seq)
    /// if no constraints apply.
    fn prepare_max_seq(
        &self,
        max_seq_by_user: Option<u64>,
        durability_filter: DurabilityLevel,
        dirty: bool,
    ) -> Option<u64> {
        let mut max_seq: Option<u64> = None;

        // if it's required to only read persisted data, we can only read up to the last remote persisted seq
        if matches!(durability_filter, DurabilityLevel::Remote) {
            max_seq = Some(self.oracle.last_remote_persisted_seq());
        }

        // if dirty read is not allowed, we can only read up to the last committed seq
        if !dirty {
            max_seq = max_seq
                .map(|seq| seq.min(self.oracle.last_committed_seq()))
                .or(Some(self.oracle.last_committed_seq()));
        }

        // if user provide a max seq (mostly from a Snapshot)
        if let Some(max_seq_by_user) = max_seq_by_user {
            max_seq = max_seq
                .map(|seq| seq.min(max_seq_by_user))
                .or(Some(max_seq_by_user));
        }

        max_seq
    }

    async fn build_iterator_sources(
        &self,
        range: &BytesRange,
        db_state: &(dyn DbStateReader + Sync),
        write_batch_iter: Option<WriteBatchIterator>,
        sst_iter_options: &SstIteratorOptions,
        point_lookup_stats: Option<DbStats>,
        max_seq: Option<u64>,
    ) -> Result<IteratorSources, SlateDBError> {
        let mut memtables = VecDeque::new();
        memtables.push_back(db_state.memtable());
        for memtable in db_state.imm_memtables() {
            memtables.push_back(memtable.table());
        }
        let mem_iters = memtables
            .iter()
            .map(|table| {
                Box::new(table.range(range.clone(), sst_iter_options.order))
                    as Box<dyn RowEntryIterator + 'static>
            })
            .collect::<Vec<_>>();

        let default_seg;
        let segments: &[Segment] = match db_state.core().select_segments(range) {
            Some(segments) => segments,
            None => {
                default_seg = db_state.core().default_segment();
                std::slice::from_ref(&default_seg)
            }
        };
        let total_ssts: usize = segments.iter().map(|s| s.tree.total_ssts()).sum();
        let max_parallel = total_ssts.clamp(1, 4);

        let context = SegmentScanContext {
            table_store: self.table_store.clone(),
            range: range.clone(),
            sst_iter_options: sst_iter_options.clone(),
            max_parallel,
            point_lookup_stats,
            db_stats: self.db_stats.clone(),
        };

        let segment_iter = build_segment_iter(segments, context, max_seq)?;

        Ok(IteratorSources {
            write_batch_iter,
            mem_iters,
            segment_iter,
        })
    }

    /// Get the full row entry for the given key.
    ///
    /// Returns `Ok(Some(entry))` if a non-expired, non-tombstone entry exists
    /// for `key`, `Ok(None)` if the key is deleted or the latest visible value
    /// is expired, and an error if the read fails.
    ///
    /// Arguments:
    /// - `key`: The user key to read. Any type that can be viewed as a byte
    ///   slice is accepted.
    /// - `options`: Options for the read, including durability constraint or
    ///   dirty read or cache blocks.
    /// - `db_state`: Read-only view over in-memory state (memtables) and on-disk
    ///   states (level-0 SSTs and compacted sorted runs).
    /// - `write_batch`: Optional `WriteBatch` to consult first. It's only used when
    ///   operating within a Transaction.
    /// - `max_seq`: Optional upper bound on the sequence number visibility. If
    ///   provided, the read will not return entries with a sequence number
    ///   greater than this value. The final bound is the minimum of this value
    ///   and the bound derived from `options` (e.g., durability, dirty read).
    pub(crate) async fn get_key_value_with_options<K: AsRef<[u8]>>(
        &self,
        key: K,
        options: &ReadOptions,
        db_state: &(dyn DbStateReader + Sync + Send),
        write_batch_iter: Option<WriteBatchIterator>,
        max_seq: Option<u64>,
    ) -> Result<Option<KeyValue>, SlateDBError> {
        self.db_stats.get_requests.increment(1);
        let max_seq = self.prepare_max_seq(max_seq, options.durability_filter, options.dirty);
        let key_slice = key.as_ref();
        let range = BytesRange::from_slice(key_slice..=key_slice);

        let sst_iter_options = SstIteratorOptions {
            cache_blocks: options.cache_blocks,
            eager_spawn: true,
            filter_context: options.filter_context.clone(),
            ..SstIteratorOptions::default()
        };

        let IteratorSources {
            write_batch_iter,
            mem_iters,
            segment_iter,
        } = self
            .build_iterator_sources(
                &range,
                db_state,
                write_batch_iter,
                &sst_iter_options,
                Some(self.db_stats.clone()),
                max_seq,
            )
            .await?;

        let mut iterator = DbIterator::new(
            range,
            write_batch_iter,
            mem_iters,
            segment_iter,
            max_seq,
            self.read_merge_operator.clone(),
            sst_iter_options.order,
        )
        .await?;

        iterator
            .next_entry()
            .await?
            .map(|entry| {
                if entry.value.is_tombstone() {
                    Err(SlateDBError::UnexpectedTombstone)
                } else {
                    Ok(KeyValue::from(entry))
                }
            })
            .transpose()
    }

    pub(crate) async fn multi_get_key_value_with_options(
        &self,
        keys: &[Bytes],
        options: &ReadOptions,
        db_state: &(dyn DbStateReader + Sync + Send),
        max_seq: Option<u64>,
    ) -> Result<Vec<Option<KeyValue>>, SlateDBError> {
        self.db_stats.get_requests.increment(1);
        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let prepared_max_seq =
            self.prepare_max_seq(max_seq, options.durability_filter, options.dirty);
        let merge_operator_enabled = self.read_merge_operator.is_some();

        let mut key_to_idx = HashMap::<Bytes, usize>::with_capacity(keys.len());
        let mut unique_keys = Vec::<Bytes>::with_capacity(keys.len());
        let mut output_positions = Vec::<Vec<usize>>::with_capacity(keys.len());

        for (output_idx, key) in keys.iter().enumerate() {
            if let Some(existing_idx) = key_to_idx.get(key).copied() {
                output_positions[existing_idx].push(output_idx);
                continue;
            }

            let key_idx = unique_keys.len();
            key_to_idx.insert(key.clone(), key_idx);
            unique_keys.push(key.clone());
            output_positions.push(vec![output_idx]);
        }

        let mut resolved = vec![false; unique_keys.len()];
        let mut values = vec![None; unique_keys.len()];
        let mut fallback_to_point_get = vec![false; unique_keys.len()];

        self.resolve_rows_from_memtable_for_keys(
            db_state.memtable(),
            &unique_keys,
            prepared_max_seq,
            merge_operator_enabled,
            &mut resolved,
            &mut values,
            &mut fallback_to_point_get,
        )
        .await?;

        for imm in db_state.imm_memtables() {
            if Self::all_done(&resolved, &fallback_to_point_get) {
                break;
            }

            self.resolve_rows_from_memtable_for_keys(
                imm.table(),
                &unique_keys,
                prepared_max_seq,
                merge_operator_enabled,
                &mut resolved,
                &mut values,
                &mut fallback_to_point_get,
            )
            .await?;
        }

        self.resolve_rows_from_segments_for_keys(
            db_state.core(),
            &unique_keys,
            prepared_max_seq,
            options,
            merge_operator_enabled,
            &mut resolved,
            &mut values,
            &mut fallback_to_point_get,
        )
        .await?;

        for key_idx in fallback_to_point_get
            .iter()
            .enumerate()
            .filter_map(|(idx, should_fallback)| should_fallback.then_some(idx))
        {
            values[key_idx] = self
                .get_key_value_with_options(
                    unique_keys[key_idx].as_ref(),
                    options,
                    db_state,
                    None,
                    max_seq,
                )
                .await?;
            resolved[key_idx] = true;
        }

        let mut result = vec![None; keys.len()];
        for (key_idx, positions) in output_positions.into_iter().enumerate() {
            let value = if resolved[key_idx] {
                values[key_idx].clone()
            } else {
                None
            };

            for position in positions {
                result[position] = value.clone();
            }
        }

        Ok(result)
    }

    fn all_done(resolved: &[bool], fallback_to_point_get: &[bool]) -> bool {
        resolved
            .iter()
            .zip(fallback_to_point_get)
            .all(|(resolved, fallback)| *resolved || *fallback)
    }

    async fn resolve_rows_from_memtable_for_keys(
        &self,
        table: Arc<KVTable>,
        unique_keys: &[Bytes],
        max_seq: Option<u64>,
        merge_operator_enabled: bool,
        resolved: &mut [bool],
        values: &mut [Option<KeyValue>],
        fallback_to_point_get: &mut [bool],
    ) -> Result<(), SlateDBError> {
        if table.is_empty() {
            return Ok(());
        }

        for (key_idx, key) in unique_keys.iter().enumerate() {
            if resolved[key_idx] || fallback_to_point_get[key_idx] {
                continue;
            }

            let mut iter = table.range_ascending(key.clone()..=key.clone());
            while let Some(row) = iter.next().await? {
                if !Self::row_visible(&row, max_seq) {
                    continue;
                }

                Self::apply_source_row_result(
                    key_idx,
                    row,
                    merge_operator_enabled,
                    resolved,
                    values,
                    fallback_to_point_get,
                )?;
                break;
            }
        }

        Ok(())
    }

    async fn resolve_rows_from_segments_for_keys(
        &self,
        core: &ManifestCore,
        unique_keys: &[Bytes],
        max_seq: Option<u64>,
        options: &ReadOptions,
        merge_operator_enabled: bool,
        resolved: &mut [bool],
        values: &mut [Option<KeyValue>],
        fallback_to_point_get: &mut [bool],
    ) -> Result<(), SlateDBError> {
        if Self::all_done(resolved, fallback_to_point_get) {
            return Ok(());
        }

        let mut groups = BTreeMap::<Bytes, (Segment, Vec<usize>)>::new();
        for (key_idx, key) in unique_keys.iter().enumerate() {
            if resolved[key_idx] || fallback_to_point_get[key_idx] {
                continue;
            }

            let range = BytesRange::from_slice(key.as_ref()..=key.as_ref());
            let segment = match core.select_segments(&range) {
                None => core.default_segment(),
                Some(segments) => match segments.first() {
                    Some(segment) => segment.clone(),
                    None => continue,
                },
            };

            groups
                .entry(segment.prefix.clone())
                .or_insert_with(|| (segment, Vec::new()))
                .1
                .push(key_idx);
        }

        for (_, (segment, key_indices)) in groups {
            self.resolve_rows_from_lsm_tree_for_keys(
                &segment,
                &key_indices,
                unique_keys,
                max_seq,
                options,
                merge_operator_enabled,
                resolved,
                values,
                fallback_to_point_get,
            )
            .await?;
        }

        Ok(())
    }

    async fn resolve_rows_from_lsm_tree_for_keys(
        &self,
        segment: &Segment,
        key_indices: &[usize],
        unique_keys: &[Bytes],
        max_seq: Option<u64>,
        options: &ReadOptions,
        merge_operator_enabled: bool,
        resolved: &mut [bool],
        values: &mut [Option<KeyValue>],
        fallback_to_point_get: &mut [bool],
    ) -> Result<(), SlateDBError> {
        let mut unresolved = key_indices
            .iter()
            .copied()
            .filter(|key_idx| !resolved[*key_idx] && !fallback_to_point_get[*key_idx])
            .collect::<Vec<_>>();

        for sst in segment.tree.l0.iter() {
            if unresolved.is_empty() {
                return Ok(());
            }

            self.resolve_rows_from_sst_for_keys(
                sst,
                &unresolved,
                unique_keys,
                max_seq,
                options,
                merge_operator_enabled,
                resolved,
                values,
                fallback_to_point_get,
            )
            .await?;
            unresolved.retain(|idx| !resolved[*idx] && !fallback_to_point_get[*idx]);
        }

        for sr in segment.tree.compacted.iter() {
            if unresolved.is_empty() {
                return Ok(());
            }

            self.resolve_rows_from_sorted_run_for_keys(
                sr,
                &unresolved,
                unique_keys,
                max_seq,
                options,
                merge_operator_enabled,
                resolved,
                values,
                fallback_to_point_get,
            )
            .await?;
            unresolved.retain(|idx| !resolved[*idx] && !fallback_to_point_get[*idx]);
        }

        Ok(())
    }

    async fn resolve_rows_from_sorted_run_for_keys(
        &self,
        sorted_run: &crate::db_state::SortedRun,
        key_indices: &[usize],
        unique_keys: &[Bytes],
        max_seq: Option<u64>,
        options: &ReadOptions,
        merge_operator_enabled: bool,
        resolved: &mut [bool],
        values: &mut [Option<KeyValue>],
        fallback_to_point_get: &mut [bool],
    ) -> Result<(), SlateDBError> {
        let mut groups = BTreeMap::<ulid::Ulid, (crate::db_state::SsTableView, Vec<usize>)>::new();
        for key_idx in key_indices.iter().copied() {
            if resolved[key_idx] || fallback_to_point_get[key_idx] {
                continue;
            }

            for sst in sorted_run.tables_covering_point_key(unique_keys[key_idx].as_ref()) {
                groups
                    .entry(sst.id)
                    .or_insert_with(|| (sst.clone(), Vec::new()))
                    .1
                    .push(key_idx);
            }
        }

        for (_, (sst, group_key_indices)) in groups {
            self.resolve_rows_from_sst_for_keys(
                &sst,
                &group_key_indices,
                unique_keys,
                max_seq,
                options,
                merge_operator_enabled,
                resolved,
                values,
                fallback_to_point_get,
            )
            .await?;
        }

        Ok(())
    }

    async fn resolve_rows_from_sst_for_keys(
        &self,
        sst: &crate::db_state::SsTableView,
        key_indices: &[usize],
        unique_keys: &[Bytes],
        max_seq: Option<u64>,
        options: &ReadOptions,
        merge_operator_enabled: bool,
        resolved: &mut [bool],
        values: &mut [Option<KeyValue>],
        fallback_to_point_get: &mut [bool],
    ) -> Result<(), SlateDBError> {
        let mut candidate_key_indices = Vec::with_capacity(key_indices.len());
        let filters = self.table_store.read_filters(&sst.sst, true).await?;

        for &key_idx in key_indices {
            if resolved[key_idx] || fallback_to_point_get[key_idx] {
                continue;
            }

            let key = unique_keys[key_idx].as_ref();
            if sst
                .calculate_view_range(BytesRange::from_slice(key..=key))
                .is_none()
            {
                continue;
            }

            if filters.is_empty() {
                candidate_key_indices.push(key_idx);
                continue;
            }

            let query = FilterQuery::point(unique_keys[key_idx].clone())
                .with_context(options.filter_context.clone());
            if filters.iter().all(|named| named.filter.might_match(&query)) {
                self.db_stats.sst_filter_point_positives.increment(1);
                candidate_key_indices.push(key_idx);
            } else {
                self.db_stats.sst_filter_point_negatives.increment(1);
            }
        }

        if candidate_key_indices.is_empty() {
            return Ok(());
        }

        let index = self.table_store.read_index(&sst.sst, true).await?;
        let block_to_key_indices = {
            let index_ref = index.borrow();
            let mut block_to_key_indices = BTreeMap::<usize, Vec<usize>>::new();
            for &key_idx in &candidate_key_indices {
                let key = unique_keys[key_idx].as_ref();
                let start_block =
                    partitioned_keyspace::first_partition_including_or_after_key(&index_ref, key);
                let end_block_exclusive =
                    partitioned_keyspace::last_partition_including_key(&index_ref, key)
                        .map(|last| last + 1)
                        .unwrap_or(start_block);

                if start_block >= end_block_exclusive {
                    continue;
                }

                for block_idx in start_block..end_block_exclusive {
                    block_to_key_indices
                        .entry(block_idx)
                        .or_default()
                        .push(key_idx);
                }
            }
            block_to_key_indices
        };

        if block_to_key_indices.is_empty() {
            return Ok(());
        }

        let mut block_ranges = Vec::new();
        let mut range_start = None;
        let mut previous_block = 0;
        for &block_idx in block_to_key_indices.keys() {
            if let Some(start) = range_start {
                if block_idx == previous_block + 1 {
                    previous_block = block_idx;
                } else {
                    block_ranges.push(start..(previous_block + 1));
                    range_start = Some(block_idx);
                    previous_block = block_idx;
                }
            } else {
                range_start = Some(block_idx);
                previous_block = block_idx;
            }
        }
        if let Some(start) = range_start {
            block_ranges.push(start..(previous_block + 1));
        }

        let mut found_in_sst = HashSet::<usize>::new();
        for block_range in block_ranges {
            let blocks = self
                .table_store
                .read_blocks_using_index(
                    &sst.sst,
                    index.clone(),
                    block_range.clone(),
                    options.cache_blocks,
                )
                .await?;

            for (offset, block) in blocks.into_iter().enumerate() {
                let block_idx = block_range.start + offset;
                let Some(key_idxs) = block_to_key_indices.get(&block_idx) else {
                    continue;
                };

                self.resolve_rows_from_block_for_keys(
                    block,
                    sst.sst.format_version,
                    key_idxs,
                    unique_keys,
                    max_seq,
                    merge_operator_enabled,
                    resolved,
                    values,
                    fallback_to_point_get,
                    &mut found_in_sst,
                )
                .await?;
            }
        }

        if !filters.is_empty() {
            let false_positives = candidate_key_indices
                .iter()
                .filter(|key_idx| !found_in_sst.contains(key_idx))
                .count() as u64;
            if false_positives > 0 {
                self.db_stats
                    .sst_filter_point_false_positives
                    .increment(false_positives);
            }
        }

        Ok(())
    }

    async fn resolve_rows_from_block_for_keys(
        &self,
        block: Arc<Block>,
        sst_version: u16,
        key_indices: &[usize],
        unique_keys: &[Bytes],
        max_seq: Option<u64>,
        merge_operator_enabled: bool,
        resolved: &mut [bool],
        values: &mut [Option<KeyValue>],
        fallback_to_point_get: &mut [bool],
        found_in_sst: &mut HashSet<usize>,
    ) -> Result<(), SlateDBError> {
        let key_lookup = key_indices
            .iter()
            .copied()
            .map(|key_idx| (unique_keys[key_idx].clone(), key_idx))
            .collect::<HashMap<_, _>>();

        let mut iter =
            DataBlockIterator::new(block, sst_version, crate::iter::IterationOrder::Ascending)?;
        while let Some(row) = iter.next().await? {
            let Some(&key_idx) = key_lookup.get(&row.key) else {
                continue;
            };
            if found_in_sst.contains(&key_idx) || !Self::row_visible(&row, max_seq) {
                continue;
            }

            found_in_sst.insert(key_idx);
            Self::apply_source_row_result(
                key_idx,
                row,
                merge_operator_enabled,
                resolved,
                values,
                fallback_to_point_get,
            )?;
        }

        Ok(())
    }

    fn row_visible(row: &RowEntry, max_seq: Option<u64>) -> bool {
        match max_seq {
            Some(max_seq) => row.seq <= max_seq,
            None => true,
        }
    }

    fn apply_source_row_result(
        key_idx: usize,
        row: RowEntry,
        merge_operator_enabled: bool,
        resolved: &mut [bool],
        values: &mut [Option<KeyValue>],
        fallback_to_point_get: &mut [bool],
    ) -> Result<(), SlateDBError> {
        match &row.value {
            ValueDeletable::Value(_) => {
                resolved[key_idx] = true;
                values[key_idx] = Some(KeyValue::from(row));
            }
            ValueDeletable::Tombstone => {
                resolved[key_idx] = true;
                values[key_idx] = None;
            }
            ValueDeletable::Merge(_) => {
                if !merge_operator_enabled {
                    return Err(SlateDBError::MergeOperatorMissing);
                }
                fallback_to_point_get[key_idx] = true;
            }
        }

        Ok(())
    }

    /// Create an iterator over a key range.
    ///
    /// Produces a merged iterator over the provided `write_batch` (if any),
    /// in-memory memtables, level-0 SSTs, and compacted sorted runs, honoring
    /// the maximum visible sequence number. The iterator yields only non-
    /// expired, non-tombstone values.
    ///
    /// Arguments
    /// - `range`: The half-open key range to scan (start inclusive, end
    ///   exclusive).
    /// - `options`: Options for the scan, including read-ahead, caching, and the
    ///   maximum number of concurrent fetch tasks.
    /// - `ctx`: Caller-threaded state. See [`ScanContext`] for the details.
    pub(crate) async fn scan_with_options(
        &self,
        range: BytesRange,
        options: &ScanOptions,
        ctx: ScanContext<'_>,
    ) -> Result<DbIterator, SlateDBError> {
        self.db_stats.scan_requests.increment(1);
        let max_seq = self.prepare_max_seq(ctx.max_seq, options.durability_filter, options.dirty);
        let read_ahead_blocks = self.table_store.bytes_to_blocks(options.read_ahead_bytes);

        let sst_iter_options = SstIteratorOptions {
            max_fetch_tasks: options.max_fetch_tasks,
            blocks_to_fetch: read_ahead_blocks,
            cache_blocks: options.cache_blocks,
            cache_metadata: true,
            eager_spawn: true,
            order: options.order,
            prefix: ctx.prefix,
            filter_context: options.filter_context.clone(),
        };

        let IteratorSources {
            write_batch_iter,
            mem_iters,
            segment_iter,
        } = self
            .build_iterator_sources(
                &range,
                ctx.db_state,
                ctx.write_batch_iter,
                &sst_iter_options,
                None,
                max_seq,
            )
            .await?;

        DbIterator::new(
            range,
            write_batch_iter,
            mem_iters,
            segment_iter,
            max_seq,
            self.read_merge_operator.clone(),
            options.order,
        )
        .await
    }

    /// See [`crate::Db::scan_prefix_by_recency`].
    ///
    /// Builds a flat chain of per-source iterators newest-first so the
    /// outer [`RecencyIterator`] can drain them in order, lazily
    /// initializing each source only when the recency walk reaches it.
    /// Cross-source order is always newest-first (active memtable, immutable
    /// memtables, then within the single matching segment that segment's L0
    /// SSTs newest-first followed by its sorted runs newest-first).
    /// `options.order` controls iteration order *within* each source. Each
    /// source is fully drained before moving to the next.
    ///
    /// Returns [`SlateDBError::RecencyScanPrefixSpansMultipleSegments`] if
    /// the prefix overlaps more than one segment, since the recency
    /// guarantee is only well-defined within a single segment.
    pub(crate) async fn scan_prefix_by_recency_with_options(
        &self,
        prefix: Bytes,
        options: &ScanOptions,
        db_state: &(dyn DbStateReader + Sync),
    ) -> Result<DbRecencyIterator, SlateDBError> {
        self.db_stats.scan_requests.increment(1);
        let max_seq = self.prepare_max_seq(None, options.durability_filter, options.dirty);
        let read_ahead_blocks = self.table_store.bytes_to_blocks(options.read_ahead_bytes);

        let range = BytesRange::from_prefix(prefix.as_ref());
        let sst_iter_options = SstIteratorOptions {
            max_fetch_tasks: options.max_fetch_tasks,
            blocks_to_fetch: read_ahead_blocks,
            cache_blocks: options.cache_blocks,
            cache_metadata: true,
            // Recency scans are designed for early-stop. Eager spawning would
            // proactively fetch blocks for sources the caller may never reach,
            // defeating the early-stop savings.
            eager_spawn: false,
            order: options.order,
            prefix: Some(prefix),
            filter_context: options.filter_context.clone(),
        };

        // Cross-segment recency is not well-defined: walking segment A's
        // oldest sorted run before touching segment B's freshest data
        // breaks the freshest-first contract callers expect.
        // `select_segments` returns `None` when unconfigured; `default_segment`
        // provides the fallback unsegmented tree.
        let segments = db_state.core().select_segments(&range);
        if segments.is_some_and(|s| s.len() > 1) {
            return Err(SlateDBError::RecencyScanPrefixSpansMultipleSegments);
        }
        let segment = match segments {
            Some(segments) => segments.first().cloned(),
            None => Some(db_state.core().default_segment()),
        };

        let mut all_iters: Vec<Box<dyn RowEntryIterator + 'static>> = Vec::new();

        // Memtables drain first (newest data, always in memory).
        all_iters.push(Box::new(
            db_state
                .memtable()
                .range(range.clone(), sst_iter_options.order),
        ));
        for memtable in db_state.imm_memtables() {
            all_iters.push(Box::new(
                memtable
                    .table()
                    .range(range.clone(), sst_iter_options.order),
            ));
        }

        // Single-segment chain: L0 newest-first, then sorted runs newest-first.
        // SST and SortedRun iterators are constructed without `init`, so the
        // filter check / index load / first-block fetch only happens when the
        // recency walk reaches the source.
        if let Some(segment) = segment {
            for sst in segment.tree.l0.iter().cloned() {
                let iter = SstIterator::new_owned_with_stats(
                    range.clone(),
                    sst,
                    self.table_store.clone(),
                    sst_iter_options.clone(),
                    Some(self.db_stats.clone()),
                )?;
                if let Some(iter) = iter {
                    all_iters.push(Box::new(iter));
                }
            }
            for sr in segment
                .tree
                .compacted
                .iter()
                .filter(|sr| sr.overlaps_range(&range))
                .cloned()
            {
                let iter = SortedRunIterator::new_owned(
                    range.clone(),
                    sr,
                    self.table_store.clone(),
                    sst_iter_options.clone(),
                    Some(self.db_stats.clone()),
                )
                .await?;
                all_iters.push(Box::new(iter));
            }
        }

        let filtered = apply_filters(all_iters, max_seq);
        Ok(DbRecencyIterator::new(VecDeque::from(filtered)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merge_operator::{
        MergeOperator, MergeOperatorError, MERGE_OPERATOR_FLUSH_PATH, MERGE_OPERATOR_READ_PATH,
    };
    use crate::test_utils::lookup_merge_operator_operands;
    use crate::types::{RowEntry, ValueDeletable};
    use bytes::Bytes;
    use rstest::rstest;
    use slatedb_common::clock::{MockSystemClock, SystemClock};

    use crate::batch::WriteBatch;
    use crate::clock::MonotonicClock;
    use crate::iter::IterationOrder;

    fn wb_point_iter(write_batch: &Option<WriteBatch>, key: &[u8]) -> Option<WriteBatchIterator> {
        write_batch.as_ref().map(|wb| {
            WriteBatchIterator::new(
                wb,
                BytesRange::from_slice(key..=key),
                IterationOrder::Ascending,
                u64::MAX,
                None,
                None,
            )
        })
    }

    fn wb_range_iter(
        write_batch: &Option<WriteBatch>,
        range: &BytesRange,
        order: IterationOrder,
    ) -> Option<WriteBatchIterator> {
        write_batch
            .as_ref()
            .map(|wb| WriteBatchIterator::new(wb, range.clone(), order, u64::MAX, None, None))
    }
    use crate::db_state::{SortedRun, SsTableHandle, SsTableId};
    use crate::db_status::DbStatusManager;
    use crate::format::sst::SsTableFormat;
    use crate::manifest::SsTableView;
    use crate::object_stores::ObjectStores;
    use crate::oracle::DbReaderOracle;
    use crate::tablestore::{TableStore, TableStoreKind};
    use object_store::{memory::InMemory, path::Path, ObjectStore};
    use slatedb_common::metrics::{
        lookup_metric_with_labels, DefaultMetricsRecorder, MetricLevel, MetricsRecorder,
        MetricsRecorderHelper,
    };
    use std::collections::HashMap;
    use std::sync::Arc;
    use ulid::Ulid;

    /// A simple merge operator for testing that concatenates byte strings
    struct StringConcatMergeOperator;

    impl MergeOperator for StringConcatMergeOperator {
        fn merge(
            &self,
            _key: &Bytes,
            existing_value: Option<Bytes>,
            operand: Bytes,
        ) -> Result<Bytes, MergeOperatorError> {
            match existing_value {
                Some(base) => {
                    let mut merged = base.to_vec();
                    merged.extend_from_slice(&operand);
                    Ok(Bytes::from(merged))
                }
                None => Ok(operand),
            }
        }
    }

    /// Test database state that can be populated with entries
    struct TestDbState {
        memtable: Arc<KVTable>,
        imm_memtable: VecDeque<Arc<ImmutableMemtable>>,
        core: ManifestCore,
        table_store: Arc<TableStore>,
    }

    impl TestDbState {
        async fn new() -> Self {
            let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
            let table_store = Arc::new(TableStore::new(
                ObjectStores::new(object_store, None),
                SsTableFormat::default(),
                Path::from("/test"),
                None,
                TableStoreKind::Main,
            ));

            Self {
                memtable: Arc::new(KVTable::new()),
                imm_memtable: VecDeque::new(),
                core: ManifestCore::new(),
                table_store,
            }
        }

        /// Add entries to the memtable
        fn add_to_memtable(&mut self, entries: Vec<RowEntry>) {
            for entry in entries {
                self.memtable.put(entry);
            }
        }

        /// Create an immutable memtable from entries
        fn add_immutable_memtable(&mut self, entries: Vec<RowEntry>) {
            let writable = crate::mem_table::WritableKVTable::new();
            for entry in entries {
                writable.put(entry);
            }
            let imm = Arc::new(ImmutableMemtable::new(writable, 0));
            self.imm_memtable.push_back(imm);
        }

        /// Create an SST from entries and add to L0
        async fn add_to_l0(&mut self, entries: Vec<RowEntry>) -> Result<(), SlateDBError> {
            if entries.is_empty() {
                return Ok(());
            }
            let sst_handle = self.build_sst(entries).await?;
            Arc::make_mut(&mut self.core.tree)
                .l0
                .push_front(SsTableView::identity(sst_handle));
            Ok(())
        }

        /// Create an SST from entries and add to a sorted run
        async fn add_to_sorted_run(
            &mut self,
            sr_id: u32,
            entries: Vec<RowEntry>,
        ) -> Result<(), SlateDBError> {
            if entries.is_empty() {
                return Ok(());
            }
            let sst_handle = self.build_sst(entries).await?;

            // Find or create the sorted run
            let tree = Arc::make_mut(&mut self.core.tree);
            if let Some(sr) = tree.compacted.iter_mut().find(|sr| sr.id == sr_id) {
                sr.sst_views.push(SsTableView::identity(sst_handle));
            } else {
                let new_sr = SortedRun {
                    id: sr_id,
                    sst_views: vec![SsTableView::identity(sst_handle)],
                };
                tree.compacted.push(new_sr);
            }
            Ok(())
        }

        /// Build an SST with the given entries
        async fn build_sst(
            &self,
            mut entries: Vec<RowEntry>,
        ) -> Result<SsTableHandle, SlateDBError> {
            // Sort entries by key (required for SST builder)
            entries.sort_by(|a, b| a.key.cmp(&b.key));

            let mut builder = self.table_store.table_builder();

            for entry in entries {
                builder.add(entry).await?;
            }

            let encoded = builder.build().await?;
            let id = SsTableId::Compacted(Ulid::new());
            self.table_store.write_sst(&id, &encoded, false).await
        }
    }

    impl DbStateReader for TestDbState {
        fn memtable(&self) -> Arc<KVTable> {
            self.memtable.clone()
        }

        fn imm_memtables(&self) -> Box<dyn Iterator<Item = Arc<ImmutableMemtable>> + Send + '_> {
            Box::new(self.imm_memtable.iter().cloned())
        }

        fn core(&self) -> &ManifestCore {
            &self.core
        }
    }

    /// Enum to specify which layer an entry belongs to
    #[derive(Debug, Clone)]
    enum LayerLocation {
        WriteBatch,
        Memtable,
        ImmutableMemtable(usize), // index in the imm queue
        L0Sst(usize),             // SST index in L0 (0 = oldest)
        SortedRun(u32),           // sorted run ID (0 = oldest)
    }

    /// A test entry with its location and data
    #[derive(Debug, Clone)]
    struct TestEntry {
        location: LayerLocation,
        key: &'static [u8],
        value: ValueDeletable,
        seq: u64,
        expire_ts: Option<i64>, // None = no expiration
    }

    impl TestEntry {
        fn value(key: &'static [u8], val: &'static [u8], seq: u64) -> Self {
            Self {
                location: LayerLocation::Memtable,
                key,
                value: ValueDeletable::Value(Bytes::from_static(val)),
                seq,
                expire_ts: None,
            }
        }

        fn tombstone(key: &'static [u8], seq: u64) -> Self {
            Self {
                location: LayerLocation::Memtable,
                key,
                value: ValueDeletable::Tombstone,
                seq,
                expire_ts: None,
            }
        }

        fn merge(key: &'static [u8], val: &'static [u8], seq: u64) -> Self {
            Self {
                location: LayerLocation::Memtable,
                key,
                value: ValueDeletable::Merge(Bytes::from_static(val)),
                seq,
                expire_ts: None,
            }
        }

        fn with_location(mut self, location: LayerLocation) -> Self {
            self.location = location;
            self
        }

        fn with_expire_ts(mut self, expire_ts: i64) -> Self {
            self.expire_ts = Some(expire_ts);
            self
        }

        fn to_row_entry(&self) -> RowEntry {
            RowEntry::new(
                Bytes::from_static(self.key),
                self.value.clone(),
                self.seq,
                None,
                self.expire_ts,
            )
        }
    }

    /// Helper to populate a TestDbState with entries organized by layer
    async fn populate_db_state(
        test_db_state: &mut TestDbState,
        entries: Vec<TestEntry>,
    ) -> Result<Option<WriteBatch>, SlateDBError> {
        // Group entries by layer
        let mut wb_batch: Option<WriteBatch> = None;
        let mut mem_entries = Vec::new();
        let mut imm_entries: HashMap<usize, Vec<RowEntry>> = HashMap::new();
        let mut l0_entries: HashMap<usize, Vec<RowEntry>> = HashMap::new();
        let mut sr_entries: HashMap<u32, Vec<RowEntry>> = HashMap::new();

        for entry in entries {
            let row_entry = entry.to_row_entry();
            match entry.location {
                LayerLocation::WriteBatch => {
                    if wb_batch.is_none() {
                        wb_batch = Some(WriteBatch::new());
                    }
                    if let Some(ref mut batch) = wb_batch {
                        match &entry.value {
                            ValueDeletable::Value(v) => {
                                batch.put(entry.key, v.as_ref());
                            }
                            ValueDeletable::Tombstone => {
                                batch.delete(entry.key);
                            }
                            ValueDeletable::Merge(v) => {
                                batch.merge(entry.key, v.as_ref());
                            }
                        }
                    }
                }
                LayerLocation::Memtable => mem_entries.push(row_entry),
                LayerLocation::ImmutableMemtable(idx) => {
                    imm_entries.entry(idx).or_default().push(row_entry);
                }
                LayerLocation::L0Sst(idx) => {
                    l0_entries.entry(idx).or_default().push(row_entry);
                }
                LayerLocation::SortedRun(sr_id) => {
                    sr_entries.entry(sr_id).or_default().push(row_entry);
                }
            }
        }

        // Populate the database state

        // Add to memtable
        if !mem_entries.is_empty() {
            test_db_state.add_to_memtable(mem_entries);
        }

        // Add immutable memtables (in order)
        let mut imm_indices: Vec<_> = imm_entries.keys().copied().collect();
        imm_indices.sort();
        for idx in imm_indices {
            if let Some(entries) = imm_entries.remove(&idx) {
                test_db_state.add_immutable_memtable(entries);
            }
        }

        // Add L0 SSTs (higher index = newer, add in ascending order so highest ends up at front)
        let mut l0_indices: Vec<_> = l0_entries.keys().copied().collect();
        l0_indices.sort();
        for idx in l0_indices {
            if let Some(entries) = l0_entries.remove(&idx) {
                test_db_state.add_to_l0(entries).await?;
            }
        }

        // Add sorted runs (higher ID = newer, add in descending order so highest is checked first)
        let mut sr_ids: Vec<_> = sr_entries.keys().copied().collect();
        sr_ids.sort();
        sr_ids.reverse();
        for sr_id in sr_ids {
            if let Some(entries) = sr_entries.remove(&sr_id) {
                test_db_state.add_to_sorted_run(sr_id, entries).await?;
            }
        }

        Ok(wb_batch)
    }

    struct LayerPriorityTestCase {
        /// Test entries with their layer locations
        entries: Vec<TestEntry>,
        /// Key to query
        query_key: &'static [u8],
        /// Expected result
        expected: Option<&'static [u8]>,
        /// Test description
        description: &'static str,
        /// Whether to allow dirty reads (default: false for realistic testing)
        dirty: bool,
        /// Oracle's last committed sequence (None = u64::MAX for all committed)
        last_committed_seq: Option<u64>,
        /// Maximum sequence number to read (for snapshot testing, None = no limit)
        max_seq: Option<u64>,
    }

    #[tokio::test]
    #[rstest]
    // Test 1: Write batch overrides all other layers
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"wb_value", 100).with_location(LayerLocation::WriteBatch),
            TestEntry::value(b"key1", b"mem_value", 50),
            TestEntry::value(b"key1", b"l0_value", 40).with_location(LayerLocation::L0Sst(0)),
        ],
        query_key: b"key1",
        expected: Some(b"wb_value"),
        description: "write batch should override memtable and L0", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 2: Memtable overrides L0 and sorted runs
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"mem_value", 50),
            TestEntry::value(b"key1", b"l0_value", 40).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key1", b"sr_value", 30).with_location(LayerLocation::SortedRun(0)),
        ],
        query_key: b"key1",
        expected: Some(b"mem_value"),
        description: "memtable should override L0 and sorted run", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 3: Tombstone in write batch hides all lower layers
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::tombstone(b"key1", 100).with_location(LayerLocation::WriteBatch),
            TestEntry::value(b"key1", b"mem_value", 50),
            TestEntry::value(b"key1", b"l0_value", 40).with_location(LayerLocation::L0Sst(0)),
        ],
        query_key: b"key1",
        expected: None,
        description: "tombstone in write batch should hide all values", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 4: Tombstone in memtable hides L0
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::tombstone(b"key1", 50),
            TestEntry::value(b"key1", b"l0_value", 40).with_location(LayerLocation::L0Sst(0)),
        ],
        query_key: b"key1",
        expected: None,
        description: "tombstone in memtable should hide L0 value", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 5: Tombstone in L0 hides sorted run
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::tombstone(b"key1", 40).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key1", b"sr_value", 30).with_location(LayerLocation::SortedRun(0)),
        ],
        query_key: b"key1",
        expected: None,
        description: "tombstone in L0 should hide sorted run value", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 6: Value after tombstone (higher seq number)
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"new_value", 60),
            TestEntry::tombstone(b"key1", 50).with_location(LayerLocation::L0Sst(0)),
        ],
        query_key: b"key1",
        expected: Some(b"new_value"),
        description: "newer value should override older tombstone", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 7: Multiple L0 SSTs - newest wins
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"l0_newer", 45).with_location(LayerLocation::L0Sst(1)),
            TestEntry::value(b"key1", b"l0_older", 40).with_location(LayerLocation::L0Sst(0)),
        ],
        query_key: b"key1",
        expected: Some(b"l0_newer"),
        description: "newer L0 SST should win over older L0 SST", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 8: L0 overrides sorted run
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"l0_value", 40).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key1", b"sr_value", 30).with_location(LayerLocation::SortedRun(0)),
        ],
        query_key: b"key1",
        expected: Some(b"l0_value"),
        description: "L0 value should override sorted run value", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 9: Nonexistent key returns None
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::value(b"other_key", b"value", 50),
        ],
        query_key: b"key1",
        expected: None,
        description: "nonexistent key should return None", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 10: Only tombstone, no value
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::tombstone(b"key1", 50),
        ],
        query_key: b"key1",
        expected: None,
        description: "tombstone with no previous value should return None", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 11: Multiple layers all with same key, write batch wins
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"wb", 100).with_location(LayerLocation::WriteBatch),
            TestEntry::value(b"key1", b"mem", 90),
            TestEntry::value(b"key1", b"imm", 80).with_location(LayerLocation::ImmutableMemtable(0)),
            TestEntry::value(b"key1", b"l0", 70).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key1", b"sr", 60).with_location(LayerLocation::SortedRun(0)),
        ],
        query_key: b"key1",
        expected: Some(b"wb"),
        description: "write batch should win", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 12: Multiple entries per L0 SST
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"l0_0_val1", 50).with_location(LayerLocation::L0Sst(1)),
            TestEntry::value(b"key2", b"l0_0_val2", 51).with_location(LayerLocation::L0Sst(1)),
            TestEntry::value(b"key1", b"l0_1_val1", 40).with_location(LayerLocation::L0Sst(0)),
        ],
        query_key: b"key1",
        expected: Some(b"l0_0_val1"),
        description: "first L0 SST entry should win", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 13: Multiple sorted runs
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"sr0", 30).with_location(LayerLocation::SortedRun(1)),
            TestEntry::value(b"key1", b"sr1", 20).with_location(LayerLocation::SortedRun(0)),
        ],
        query_key: b"key1",
        expected: Some(b"sr0"),
        description: "first sorted run should win", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 14: Multiple immutable memtables
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"imm0", 60).with_location(LayerLocation::ImmutableMemtable(0)),
            TestEntry::value(b"key1", b"imm1", 50).with_location(LayerLocation::ImmutableMemtable(1)),
            TestEntry::value(b"key1", b"l0", 40).with_location(LayerLocation::L0Sst(0)),
        ],
        query_key: b"key1",
        expected: Some(b"imm0"),
        description: "first immutable memtable should win over second and L0", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 15: Complex scenario with multiple entries across all layers
    #[case(LayerPriorityTestCase {
        entries: vec![
            // WriteBatch has multiple keys but not key1
            TestEntry::value(b"key2", b"wb2", 100).with_location(LayerLocation::WriteBatch),
            // Memtable has key1 with high seq
            TestEntry::value(b"key1", b"mem", 90),
            TestEntry::value(b"key3", b"mem3", 91),
            // L0 has older versions
            TestEntry::value(b"key1", b"l0_0", 70).with_location(LayerLocation::L0Sst(1)),
            TestEntry::value(b"key1", b"l0_1", 60).with_location(LayerLocation::L0Sst(0)),
            // SR has oldest
            TestEntry::value(b"key1", b"sr", 50).with_location(LayerLocation::SortedRun(0)),
        ],
        query_key: b"key1",
        expected: Some(b"mem"),
        description: "memtable value should win in complex multi-layer scenario", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 16: Tombstone prevents revival even when newer value expires
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::tombstone(b"key1", 60).with_location(LayerLocation::L0Sst(1)),
            TestEntry::value(b"key1", b"l0_old_value", 50).with_location(LayerLocation::L0Sst(0)),
        ],
        query_key: b"key1",
        expected: None,
        description: "tombstone should prevent returning older value regardless of TTL", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 22: Value with no expiration should always be returned
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"mem_value", 50),
        ],
        query_key: b"key1",
        expected: Some(b"mem_value"),
        description: "value with no expiration should be returned at any time", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 23: Committed read filters out uncommitted data in memtable
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"uncommitted", 100),
        ],
        query_key: b"key1",
        expected: None,
        description: "committed read should not see uncommitted data", dirty: false, last_committed_seq: Some(50), max_seq: None,
    })]
    // Test 24: Committed read sees committed data
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"committed", 40),
        ],
        query_key: b"key1",
        expected: Some(b"committed"),
        description: "committed read should see data within committed range", dirty: false, last_committed_seq: Some(50), max_seq: None,
    })]
    // Test 25: Uncommitted value doesn't hide older committed value
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"uncommitted", 100),
            TestEntry::value(b"key1", b"committed", 40).with_location(LayerLocation::L0Sst(0)),
        ],
        query_key: b"key1",
        expected: Some(b"committed"),
        description: "committed read should see older committed value when newer is uncommitted", dirty: false, last_committed_seq: Some(50), max_seq: None,
    })]
    // Test 26: Snapshot with max_seq filters newer values
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"newer", 80),
            TestEntry::value(b"key1", b"older", 50).with_location(LayerLocation::L0Sst(0)),
        ],
        query_key: b"key1",
        expected: Some(b"older"),
        description: "snapshot read should only see values up to max_seq", dirty: true, last_committed_seq: None, max_seq: Some(60),
    })]
    // Test 27: Snapshot with max_seq returns None when all values are newer
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"newer", 80),
        ],
        query_key: b"key1",
        expected: None,
        description: "snapshot should return None when all values exceed max_seq", dirty: true, last_committed_seq: None, max_seq: Some(60),
    })]
    // Test 28: Combined max_seq and last_committed_seq filtering
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"v1", 100),
            TestEntry::value(b"key1", b"v2", 70).with_location(LayerLocation::L0Sst(1)),
            TestEntry::value(b"key1", b"v3", 40).with_location(LayerLocation::L0Sst(0)),
        ],
        query_key: b"key1",
        expected: Some(b"v3"),
        description: "should respect both max_seq and committed_seq constraints", dirty: false, last_committed_seq: Some(50), max_seq: Some(60),
    })]
    // Test 29: Tombstone within sequence bounds hides older values
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::tombstone(b"key1", 45).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key1", b"old_value", 30).with_location(LayerLocation::SortedRun(0)),
        ],
        query_key: b"key1",
        expected: None,
        description: "tombstone within seq bounds should hide older values", dirty: false, last_committed_seq: Some(50), max_seq: None,
    })]
    // Test 30: Newer tombstone filtered out doesn't prevent reading older value
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::tombstone(b"key1", 100),
            TestEntry::value(b"key1", b"old_value", 40).with_location(LayerLocation::L0Sst(0)),
        ],
        query_key: b"key1",
        expected: Some(b"old_value"),
        description: "filtered tombstone should not hide visible older value", dirty: false, last_committed_seq: Some(50), max_seq: None,
    })]
    // Test 31: Sequence filtering works across all layers
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"mem", 90),
            TestEntry::value(b"key1", b"imm", 70).with_location(LayerLocation::ImmutableMemtable(0)),
            TestEntry::value(b"key1", b"l0", 50).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key1", b"sr", 30).with_location(LayerLocation::SortedRun(0)),
        ],
        query_key: b"key1",
        expected: Some(b"l0"),
        description: "sequence filtering should work uniformly across all layers", dirty: false, last_committed_seq: Some(60), max_seq: None,
    })]
    // Test 32: Dirty read sees all uncommitted data
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"uncommitted", 100),
        ],
        query_key: b"key1",
        expected: Some(b"uncommitted"),
        description: "dirty read should see uncommitted data", dirty: true, last_committed_seq: Some(50), max_seq: None,
    })]
    // NOTE: for tests that use WriteBatch, the order in which the merge operations are listed
    // in the test case is important. The first merge should be the "oldest" that happened because
    // the WriteBatch assumes all sequence numbers are u64::MAX.
    // Test 33: WriteBatch with merge operations merges with memtable value
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"wb_merge", u64::MAX).with_location(LayerLocation::WriteBatch),
            TestEntry::value(b"key1", b"mem_value", 50),
            TestEntry::value(b"key1", b"l0_value", 40).with_location(LayerLocation::L0Sst(0)),
        ],
        query_key: b"key1",
        expected: Some(b"mem_valuewb_merge"),  // Merge("mem_value", "wb_merge") = "mem_valuewb_merge"
        description: "[MERGE] write batch merge should merge with memtable value", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 34: WriteBatch with multiple merge operations for same key
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"merge1", u64::MAX).with_location(LayerLocation::WriteBatch),
            TestEntry::merge(b"key1", b"merge2", u64::MAX).with_location(LayerLocation::WriteBatch),
            TestEntry::value(b"key1", b"base", 50).with_location(LayerLocation::L0Sst(0)),
        ],
        query_key: b"key1",
        expected: Some(b"basemerge1merge2"),  // Merge(base, merge1, merge2) = "basemerge1merge2"
        description: "[MERGE] multiple write batch merges should merge with base value", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 35: WriteBatch merge with tombstone clears history
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"old_merge", u64::MAX).with_location(LayerLocation::WriteBatch),
            TestEntry::tombstone(b"key1", 90).with_location(LayerLocation::WriteBatch),
            TestEntry::merge(b"key1", b"new_merge", u64::MAX).with_location(LayerLocation::WriteBatch),
            TestEntry::value(b"key1", b"base", 50).with_location(LayerLocation::L0Sst(0)),
        ],
        query_key: b"key1",
        expected: Some(b"new_merge"),  // Tombstone clears history, only merge after tombstone applies
        description: "[MERGE] write batch tombstone should clear merge history", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 36: WriteBatch merge with value acts as new base
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"old_merge", u64::MAX).with_location(LayerLocation::WriteBatch),
            TestEntry::value(b"key1", b"new_base", 90).with_location(LayerLocation::WriteBatch),
            TestEntry::merge(b"key1", b"merge", u64::MAX).with_location(LayerLocation::WriteBatch),
            TestEntry::value(b"key1", b"old_base", 50).with_location(LayerLocation::L0Sst(0)),
        ],
        query_key: b"key1",
        expected: Some(b"new_basemerge"),  // Value acts as barrier, only Merge("new_base", "merge") applies
        description: "[MERGE] write batch value should act as new base for subsequent merges", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 37: WriteBatch merges without base values
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"merge1", u64::MAX).with_location(LayerLocation::WriteBatch),
            TestEntry::merge(b"key1", b"merge2", u64::MAX).with_location(LayerLocation::WriteBatch),
        ],
        query_key: b"key1",
        expected: Some(b"merge1merge2"),  // Only merge operands, no base value
        description: "[MERGE] write batch merges without base values should merge together", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 33: Single merge operand without base value acts as base
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"a", 50),
        ],
        query_key: b"key1",
        expected: Some(b"a"),  // Single merge operand with no base returns the operand
        description: "[MERGE] single merge operand without base should be returned as-is", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 34: Merge operand with base value should merge
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"b", 60),
            TestEntry::value(b"key1", b"a", 50).with_location(LayerLocation::L0Sst(0)),
        ],
        query_key: b"key1",
        expected: Some(b"ab"),  // Merge("a", "b") = "ab" (concatenation)
        description: "[MERGE] merge operand should be merged with base value", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 35: Multiple merge operands should be applied in sequence number order
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"c", 70),
            TestEntry::merge(b"key1", b"b", 60).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key1", b"a", 50).with_location(LayerLocation::SortedRun(0)),
        ],
        query_key: b"key1",
        expected: Some(b"abc"),  // Merge(Merge("a", "b"), "c") = "abc"
        description: "[MERGE] multiple merge operands should be applied in sequence order", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 36: Tombstone clears all merge history
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"d", 80),
            TestEntry::tombstone(b"key1", 70).with_location(LayerLocation::L0Sst(1)),
            TestEntry::merge(b"key1", b"c", 60).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key1", b"ab", 50).with_location(LayerLocation::SortedRun(0)),
        ],
        query_key: b"key1",
        expected: Some(b"d"),  // Tombstone acts as barrier, only merge after tombstone is applied
        description: "[MERGE] tombstone should clear merge history", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 37: Value after merges acts as new base
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"z", 90),
            TestEntry::value(b"key1", b"y", 80).with_location(LayerLocation::L0Sst(1)),
            TestEntry::merge(b"key1", b"x", 70).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key1", b"w", 60).with_location(LayerLocation::SortedRun(0)),
        ],
        query_key: b"key1",
        expected: Some(b"yz"),  // Value("y") acts as barrier, only Merge("y", "z") is applied
        description: "[MERGE] value should act as new base for subsequent merges", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 38: Merges across multiple layers
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"d", 100),
            TestEntry::merge(b"key1", b"c", 90).with_location(LayerLocation::ImmutableMemtable(0)),
            TestEntry::merge(b"key1", b"b", 80).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key1", b"a", 70).with_location(LayerLocation::SortedRun(0)),
        ],
        query_key: b"key1",
        expected: Some(b"abcd"),  // Merge(Merge(Merge("a", "b"), "c"), "d") = "abcd"
        description: "[MERGE] merges should work across all layers", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 39: Merge with sequence filtering
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"c", 100),  // Filtered (uncommitted)
            TestEntry::merge(b"key1", b"b", 45),  // Visible
            TestEntry::value(b"key1", b"a", 40).with_location(LayerLocation::L0Sst(0)),  // Visible
        ],
        query_key: b"key1",
        expected: Some(b"ab"),  // Merge("a", "b") = "ab" (seq 100 filtered)
        description: "[MERGE] merge operands should respect sequence filtering", dirty: false, last_committed_seq: Some(50), max_seq: None,
    })]
    // Test 43: Snapshot isolation with merge operands
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"c", 80),  // Filtered by snapshot
            TestEntry::merge(b"key1", b"b", 55),  // Visible in snapshot
            TestEntry::value(b"key1", b"a", 50).with_location(LayerLocation::L0Sst(0)),  // Visible in snapshot
        ],
        query_key: b"key1",
        expected: Some(b"ab"),  // Merge("a", "b") = "ab" (seq 80 filtered by snapshot)
        description: "[MERGE] snapshot should filter merge operands by max_seq", dirty: true, last_committed_seq: None, max_seq: Some(60),
    })]
    // Test 44: Only merge operands, no tombstone/value base
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"c", 70),
            TestEntry::merge(b"key1", b"b", 60).with_location(LayerLocation::L0Sst(0)),
            TestEntry::merge(b"key1", b"a", 50).with_location(LayerLocation::SortedRun(0)),
        ],
        query_key: b"key1",
        expected: Some(b"abc"),  // String concatenation: "" + "a" + "b" + "c" = "abc"
        description: "[MERGE] multiple merge operands without base should merge together", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 45: Tombstone after value prevents merge from applying to old value
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"new", 80),
            TestEntry::tombstone(b"key1", 70).with_location(LayerLocation::L0Sst(1)),
            TestEntry::value(b"key1", b"old", 60).with_location(LayerLocation::L0Sst(0)),
        ],
        query_key: b"key1",
        expected: Some(b"new"),  // Tombstone blocks old value, merge becomes new base
        description: "[MERGE] tombstone should prevent merge from applying to older value", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 46: Merge operands in same layer (memtable)
    #[case(LayerPriorityTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"c", 70),
            TestEntry::merge(b"key1", b"b", 60),
            TestEntry::merge(b"key1", b"a", 50),
        ],
        query_key: b"key1",
        expected: Some(b"abc"),  // All merges in memtable: Merge(Merge("a", "b"), "c") = "abc"
        description: "[MERGE] multiple merge operands in same layer should merge in seq order", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    async fn test_get_with_options_layer_priority(
        #[case] test_case: LayerPriorityTestCase,
    ) -> Result<(), SlateDBError> {
        // Create test database state and populate it

        let mut test_db_state = TestDbState::new().await;
        let write_batch = populate_db_state(&mut test_db_state, test_case.entries).await?;

        // Create Reader with test clock
        let recorder = slatedb_common::metrics::MetricsRecorderHelper::noop();
        let db_stats = DbStats::new(&recorder);
        let test_clock = Arc::new(MockSystemClock::new());
        let mono_clock = Arc::new(MonotonicClock::new(test_clock as Arc<dyn SystemClock>, 0));

        // Create Oracle with appropriate last_committed_seq
        let last_committed_seq = test_case.last_committed_seq.unwrap_or(u64::MAX);
        let oracle = Arc::new(DbReaderOracle::new(
            last_committed_seq,
            DbStatusManager::new(0),
        ));

        // Enable merge operator if the test description contains "[MERGE]"
        let merge_operator = if test_case.description.contains("[MERGE]") {
            Some(Arc::new(StringConcatMergeOperator) as Arc<dyn MergeOperator + Send + Sync>)
        } else {
            None
        };

        let reader = Reader::new(
            test_db_state.table_store.clone(),
            db_stats,
            mono_clock,
            oracle,
            merge_operator,
        );

        // Call the actual get_key_value_with_options method
        let read_options = ReadOptions::default().with_dirty(test_case.dirty);
        let result = reader
            .get_key_value_with_options(
                test_case.query_key,
                &read_options,
                &test_db_state,
                wb_point_iter(&write_batch, test_case.query_key),
                test_case.max_seq,
            )
            .await?;

        let actual = result.as_ref().map(|kv| kv.value.as_ref());
        let expected = test_case.expected;
        assert_eq!(
            actual,
            expected,
            "Failed test: {}\nActual: {:?}\nExpected: {:?}",
            test_case.description,
            actual.map(|b| String::from_utf8_lossy(b)),
            expected.map(|b| String::from_utf8_lossy(b))
        );

        Ok(())
    }

    #[test]
    fn test_scan_options_builder_pattern() {
        // Test that the builder pattern works correctly for max_fetch_tasks
        let options = ScanOptions::default()
            .with_max_fetch_tasks(4)
            .with_cache_blocks(true)
            .with_read_ahead_bytes(1024);

        assert_eq!(options.max_fetch_tasks, 4);
        assert!(options.cache_blocks);
        assert_eq!(options.read_ahead_bytes, 1024);
    }

    /// Test case for scan_with_options
    struct ScanTestCase {
        /// Test entries with their layer locations
        entries: Vec<TestEntry>,
        /// Start of range (inclusive)
        range_start: &'static [u8],
        /// End of range (exclusive)
        range_end: &'static [u8],
        /// Expected results in order
        expected: Vec<(&'static [u8], &'static [u8])>,
        /// Test description
        description: &'static str,
        /// Whether to allow dirty reads (default: false for realistic testing)
        dirty: bool,
        /// Oracle's last committed sequence (None = u64::MAX for all committed)
        last_committed_seq: Option<u64>,
        /// Maximum sequence number to read (for snapshot testing, None = no limit)
        max_seq: Option<u64>,
    }

    #[tokio::test]
    #[rstest]
    // Test 1: Scan returns keys in order from single layer
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"val1", 10),
            TestEntry::value(b"key2", b"val2", 10),
            TestEntry::value(b"key3", b"val3", 10),
        ],
        range_start: b"key1",
        range_end: b"key4",
        expected: vec![(b"key1", b"val1"), (b"key2", b"val2"), (b"key3", b"val3")],
        description: "scan should return keys in order", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 2: Scan respects range boundaries
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"val1", 10),
            TestEntry::value(b"key2", b"val2", 10),
            TestEntry::value(b"key3", b"val3", 10),
        ],
        range_start: b"key2",
        range_end: b"key3",
        expected: vec![(b"key2", b"val2")],
        description: "scan should respect range boundaries (end exclusive)", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 3: Higher layer values override lower layers
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"mem_val", 50),
            TestEntry::value(b"key1", b"l0_val", 40).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key2", b"l0_val2", 40).with_location(LayerLocation::L0Sst(0)),
        ],
        range_start: b"key1",
        range_end: b"key3",
        expected: vec![(b"key1", b"mem_val"), (b"key2", b"l0_val2")],
        description: "scan should prefer higher layer values", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 4: Tombstones hide values in lower layers
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"val1", 50),
            TestEntry::tombstone(b"key2", 50),
            TestEntry::value(b"key2", b"old_val2", 40).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key3", b"val3", 50),
        ],
        range_start: b"key1",
        range_end: b"key4",
        expected: vec![(b"key1", b"val1"), (b"key3", b"val3")],
        description: "tombstones should hide values from lower layers", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 5: Uncommitted values filtered in committed read
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"committed", 40),
            TestEntry::value(b"key2", b"uncommitted", 60),
            TestEntry::value(b"key3", b"committed", 30),
        ],
        range_start: b"key1",
        range_end: b"key4",
        expected: vec![(b"key1", b"committed"), (b"key3", b"committed")],
        description: "committed scan should filter uncommitted values", dirty: false, last_committed_seq: Some(50), max_seq: None,
    })]
    // Test 8: Uncommitted value doesn't hide older committed value
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"uncommitted", 60),
            TestEntry::value(b"key1", b"committed", 40).with_location(LayerLocation::L0Sst(0)),
        ],
        range_start: b"key1",
        range_end: b"key2",
        expected: vec![(b"key1", b"committed")],
        description: "uncommitted value should not hide committed value in scan", dirty: false, last_committed_seq: Some(50), max_seq: None,
    })]
    // Test 9: Snapshot with max_seq
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"v1_new", 70),
            TestEntry::value(b"key1", b"v1_old", 50).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key2", b"v2", 55),
        ],
        range_start: b"key1",
        range_end: b"key3",
        expected: vec![(b"key1", b"v1_old"), (b"key2", b"v2")],
        description: "snapshot scan should respect max_seq", dirty: true, last_committed_seq: None, max_seq: Some(60),
    })]
    // Test 10: Multiple layers with proper deduplication
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"mem", 80),
            TestEntry::value(b"key2", b"imm", 70).with_location(LayerLocation::ImmutableMemtable(0)),
            TestEntry::value(b"key3", b"l0_new", 60).with_location(LayerLocation::L0Sst(1)),
            TestEntry::value(b"key3", b"l0_old", 50).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key4", b"sr", 40).with_location(LayerLocation::SortedRun(0)),
        ],
        range_start: b"key1",
        range_end: b"key5",
        expected: vec![(b"key1", b"mem"), (b"key2", b"imm"), (b"key3", b"l0_new"), (b"key4", b"sr")],
        description: "scan should properly deduplicate across all layers", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 11: Empty range returns no results
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"val1", 10),
        ],
        range_start: b"key5",
        range_end: b"key9",
        expected: vec![],
        description: "scan of empty range should return no results", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 12: Scan with all keys deleted
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::tombstone(b"key1", 50),
            TestEntry::tombstone(b"key2", 50),
        ],
        range_start: b"key1",
        range_end: b"key3",
        expected: vec![],
        description: "scan with all tombstones should return empty", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 13: Complex sequence filtering across range
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::value(b"key1", b"uncommitted1", 70),
            TestEntry::value(b"key1", b"committed1", 40).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key2", b"committed2", 45),
            TestEntry::value(b"key3", b"uncommitted3", 80),
        ],
        range_start: b"key1",
        range_end: b"key4",
        expected: vec![(b"key1", b"committed1"), (b"key2", b"committed2")],
        description: "complex committed scan should filter correctly", dirty: false, last_committed_seq: Some(50), max_seq: None,
    })]
    // Test 14: Tombstone within seq bounds prevents reading old value
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::tombstone(b"key1", 45).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key1", b"old", 30).with_location(LayerLocation::SortedRun(0)),
            TestEntry::value(b"key2", b"val2", 40),
        ],
        range_start: b"key1",
        range_end: b"key3",
        expected: vec![(b"key2", b"val2")],
        description: "tombstone in bounds should hide older value", dirty: false, last_committed_seq: Some(50), max_seq: None,
    })]
    // Test 15: Filtered tombstone doesn't hide visible value
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::tombstone(b"key1", 70),
            TestEntry::value(b"key1", b"visible", 40).with_location(LayerLocation::L0Sst(0)),
        ],
        range_start: b"key1",
        range_end: b"key2",
        expected: vec![(b"key1", b"visible")],
        description: "filtered tombstone should not hide visible value", dirty: false, last_committed_seq: Some(50), max_seq: None,
    })]
    // ========================================
    // MERGE OPERATOR TESTS FOR SCAN OPERATIONS
    // ========================================
    // Test 16: Scan with merge operands
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"b", 60),
            TestEntry::value(b"key1", b"a", 50).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key2", b"x", 50).with_location(LayerLocation::L0Sst(0)),
        ],
        range_start: b"key1",
        range_end: b"key3",
        expected: vec![(b"key1", b"ab"), (b"key2", b"x")],  // key1: Merge("a", "b") = "ab"
        description: "[MERGE SCAN] should merge operands with base values during scan", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 17: Scan with multiple keys having merge operands
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"2", 60),
            TestEntry::value(b"key1", b"1", 50).with_location(LayerLocation::L0Sst(0)),
            TestEntry::merge(b"key2", b"b", 60),
            TestEntry::value(b"key2", b"a", 50).with_location(LayerLocation::L0Sst(0)),
            TestEntry::merge(b"key3", b"y", 60),
            TestEntry::value(b"key3", b"x", 50).with_location(LayerLocation::L0Sst(0)),
        ],
        range_start: b"key1",
        range_end: b"key4",
        expected: vec![(b"key1", b"12"), (b"key2", b"ab"), (b"key3", b"xy")],
        description: "[MERGE SCAN] should merge operands for multiple keys", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 18: Scan with tombstone clearing merge history
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"new", 70),
            TestEntry::tombstone(b"key1", 60).with_location(LayerLocation::L0Sst(1)),
            TestEntry::merge(b"key1", b"c", 50).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key2", b"x", 50).with_location(LayerLocation::L0Sst(0)),
        ],
        range_start: b"key1",
        range_end: b"key3",
        expected: vec![(b"key1", b"new"), (b"key2", b"x")],  // Tombstone clears history
        description: "[MERGE SCAN] tombstone should clear merge history in scan", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 19: Scan with only merge operands (no base values)
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"b", 60),
            TestEntry::merge(b"key1", b"a", 50).with_location(LayerLocation::L0Sst(0)),
            TestEntry::merge(b"key2", b"y", 60),
            TestEntry::merge(b"key2", b"x", 50).with_location(LayerLocation::L0Sst(0)),
        ],
        range_start: b"key1",
        range_end: b"key3",
        expected: vec![(b"key1", b"ab"), (b"key2", b"xy")],
        description: "[MERGE SCAN] should merge operands without base values", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 21: Scan with merge operands across layers
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"d", 80),
            TestEntry::merge(b"key1", b"c", 70).with_location(LayerLocation::ImmutableMemtable(0)),
            TestEntry::merge(b"key1", b"b", 60).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key1", b"a", 50).with_location(LayerLocation::SortedRun(0)),
        ],
        range_start: b"key1",
        range_end: b"key2",
        expected: vec![(b"key1", b"abcd")],  // Merge across all layers: "a"+"b"+"c"+"d" = "abcd"
        description: "[MERGE SCAN] should merge operands across multiple layers", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 22: Scan with sequence filtering and merge operands
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"c", 100),  // Filtered (uncommitted)
            TestEntry::merge(b"key1", b"b", 45),
            TestEntry::value(b"key1", b"a", 40).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key2", b"x", 40).with_location(LayerLocation::L0Sst(0)),
        ],
        range_start: b"key1",
        range_end: b"key3",
        expected: vec![(b"key1", b"ab"), (b"key2", b"x")],  // seq 100 filtered
        description: "[MERGE SCAN] should filter merge operands by sequence number", dirty: false, last_committed_seq: Some(50), max_seq: None,
    })]
    // Test 23: Scan with mixed values, merges, and tombstones
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"b", 60),
            TestEntry::value(b"key1", b"a", 50).with_location(LayerLocation::L0Sst(0)),
            TestEntry::tombstone(b"key2", 60),
            TestEntry::merge(b"key2", b"z", 50).with_location(LayerLocation::L0Sst(0)),
            TestEntry::merge(b"key3", b"y", 60),
            TestEntry::merge(b"key3", b"x", 50).with_location(LayerLocation::L0Sst(0)),
        ],
        range_start: b"key1",
        range_end: b"key4",
        expected: vec![(b"key1", b"ab"), (b"key3", b"xy")],  // key2 tombstoned
        description: "[MERGE SCAN] should handle mix of values, merges, and tombstones", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 24: Scan with WriteBatch merge operations
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"wb_merge", 100).with_location(LayerLocation::WriteBatch),
            TestEntry::value(b"key1", b"base", 50).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key2", b"normal", 50).with_location(LayerLocation::L0Sst(0)),
        ],
        range_start: b"key1",
        range_end: b"key3",
        expected: vec![(b"key1", b"basewb_merge"), (b"key2", b"normal")],  // key1: Merge("base", "wb_merge")
        description: "[MERGE SCAN] should merge write batch operands with base values", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 25: Scan with multiple WriteBatch merge operations for same key
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"merge1", 80).with_location(LayerLocation::WriteBatch),
            TestEntry::merge(b"key1", b"merge2", 90).with_location(LayerLocation::WriteBatch),
            TestEntry::merge(b"key1", b"merge3", 100).with_location(LayerLocation::WriteBatch),
            TestEntry::value(b"key1", b"base", 50).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key2", b"normal", 50).with_location(LayerLocation::L0Sst(0)),
        ],
        range_start: b"key1",
        range_end: b"key3",
        expected: vec![(b"key1", b"basemerge1merge2merge3"), (b"key2", b"normal")],  // Multiple merges applied in sequence
        description: "[MERGE SCAN] should apply multiple write batch merges in sequence", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 26: Scan with WriteBatch tombstone clearing merge history
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"old_merge", 80).with_location(LayerLocation::WriteBatch),
            TestEntry::tombstone(b"key1", 90).with_location(LayerLocation::WriteBatch),
            TestEntry::merge(b"key1", b"new_merge", 100).with_location(LayerLocation::WriteBatch),
            TestEntry::value(b"key1", b"base", 50).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key2", b"normal", 50).with_location(LayerLocation::L0Sst(0)),
        ],
        range_start: b"key1",
        range_end: b"key3",
        expected: vec![(b"key1", b"new_merge"), (b"key2", b"normal")],  // Tombstone clears history
        description: "[MERGE SCAN] write batch tombstone should clear merge history", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 27: Scan with WriteBatch value acting as new base for merges
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"old_merge", 80).with_location(LayerLocation::WriteBatch),
            TestEntry::value(b"key1", b"new_base", 90).with_location(LayerLocation::WriteBatch),
            TestEntry::merge(b"key1", b"final_merge", 100).with_location(LayerLocation::WriteBatch),
            TestEntry::value(b"key1", b"old_base", 50).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key2", b"normal", 50).with_location(LayerLocation::L0Sst(0)),
        ],
        range_start: b"key1",
        range_end: b"key3",
        expected: vec![(b"key1", b"new_basefinal_merge"), (b"key2", b"normal")],  // Value acts as barrier
        description: "[MERGE SCAN] write batch value should act as new base for subsequent merges", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 28: Scan with WriteBatch merges across multiple keys
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"1_merge", 100).with_location(LayerLocation::WriteBatch),
            TestEntry::value(b"key1", b"1_base", 50).with_location(LayerLocation::L0Sst(0)),
            TestEntry::merge(b"key2", b"2_merge", 100).with_location(LayerLocation::WriteBatch),
            TestEntry::value(b"key2", b"2_base", 50).with_location(LayerLocation::L0Sst(0)),
            TestEntry::merge(b"key3", b"3_merge", 100).with_location(LayerLocation::WriteBatch),
            TestEntry::value(b"key3", b"3_base", 50).with_location(LayerLocation::L0Sst(0)),
        ],
        range_start: b"key1",
        range_end: b"key4",
        expected: vec![(b"key1", b"1_base1_merge"), (b"key2", b"2_base2_merge"), (b"key3", b"3_base3_merge")],
        description: "[MERGE SCAN] should merge write batch operands for multiple keys", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    // Test 29: Scan with WriteBatch merges without base values
    #[case(ScanTestCase {
        entries: vec![
            TestEntry::merge(b"key1", b"merge1", 90).with_location(LayerLocation::WriteBatch),
            TestEntry::merge(b"key1", b"merge2", 100).with_location(LayerLocation::WriteBatch),
            TestEntry::value(b"key2", b"normal", 50).with_location(LayerLocation::L0Sst(0)),
        ],
        range_start: b"key1",
        range_end: b"key3",
        expected: vec![(b"key1", b"merge1merge2"), (b"key2", b"normal")],  // Only merge operands, no base value
        description: "[MERGE SCAN] should merge write batch operands without base values", dirty: true, last_committed_seq: None, max_seq: None,
    })]
    async fn test_scan_with_options_layer_priority(
        #[case] test_case: ScanTestCase,
    ) -> Result<(), SlateDBError> {
        // Create test database state and populate it

        let mut test_db_state = TestDbState::new().await;
        let write_batch = populate_db_state(&mut test_db_state, test_case.entries).await?;

        // Create Reader with test clock
        let recorder = slatedb_common::metrics::MetricsRecorderHelper::noop();
        let db_stats = DbStats::new(&recorder);
        let test_clock = Arc::new(MockSystemClock::new());
        let mono_clock = Arc::new(MonotonicClock::new(test_clock as Arc<dyn SystemClock>, 0));

        // Create Oracle with appropriate last_committed_seq
        let last_committed_seq = test_case.last_committed_seq.unwrap_or(u64::MAX);
        let oracle = Arc::new(DbReaderOracle::new(
            last_committed_seq,
            DbStatusManager::new(0),
        ));

        // Enable merge operator if the test description contains "[MERGE"
        let merge_operator = if test_case.description.contains("[MERGE") {
            Some(Arc::new(StringConcatMergeOperator) as Arc<dyn MergeOperator + Send + Sync>)
        } else {
            None
        };

        let reader = Reader::new(
            test_db_state.table_store.clone(),
            db_stats,
            mono_clock,
            oracle,
            merge_operator,
        );

        // Create range
        let range = BytesRange::from_slice(test_case.range_start..test_case.range_end);

        // Call the actual scan_with_options method
        let scan_options = ScanOptions::default().with_dirty(test_case.dirty);
        let wb_iter = wb_range_iter(&write_batch, &range, scan_options.order);
        let mut iter = reader
            .scan_with_options(
                range,
                &scan_options,
                ScanContext {
                    db_state: &test_db_state,
                    write_batch_iter: wb_iter,
                    max_seq: test_case.max_seq,
                    prefix: None,
                },
            )
            .await?;

        // Collect results
        let mut actual = Vec::new();
        while let Some(kv) = iter
            .next()
            .await
            .map_err(|e| SlateDBError::IoError(Arc::new(std::io::Error::other(e))))?
        {
            actual.push((kv.key.to_vec(), kv.value.to_vec()));
        }

        // Compare with expected
        let expected: Vec<(Vec<u8>, Vec<u8>)> = test_case
            .expected
            .into_iter()
            .map(|(k, v)| (k.to_vec(), v.to_vec()))
            .collect();

        assert_eq!(
            actual,
            expected,
            "Failed test: {}\nActual entries: {:?}\nExpected entries: {:?}",
            test_case.description,
            actual
                .iter()
                .map(|(k, v)| format!(
                    "({}, {})",
                    String::from_utf8_lossy(k),
                    String::from_utf8_lossy(v)
                ))
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|(k, v)| format!(
                    "({}, {})",
                    String::from_utf8_lossy(k),
                    String::from_utf8_lossy(v)
                ))
                .collect::<Vec<_>>()
        );

        Ok(())
    }

    /// Helper to build a Reader for expire_ts tests.
    async fn build_reader(
        test_db_state: &TestDbState,
        db_stats: DbStats,
        with_merge: bool,
    ) -> Reader {
        let test_clock = Arc::new(MockSystemClock::new());
        let mono_clock = Arc::new(MonotonicClock::new(test_clock as Arc<dyn SystemClock>, 0));
        let oracle = Arc::new(DbReaderOracle::new(u64::MAX, DbStatusManager::new(0)));
        let merge_operator = if with_merge {
            Some(Arc::new(StringConcatMergeOperator) as Arc<dyn MergeOperator + Send + Sync>)
        } else {
            None
        };
        Reader::new(
            test_db_state.table_store.clone(),
            db_stats,
            mono_clock,
            oracle,
            merge_operator,
        )
    }

    #[tokio::test]
    async fn should_record_merge_operator_operands_on_read_path() -> Result<(), SlateDBError> {
        let entries = vec![
            TestEntry::merge(b"key1", b"b", 2),
            TestEntry::merge(b"key1", b"a", 1),
            TestEntry::value(b"key2", b"value2", 1),
        ];

        let mut test_db_state = TestDbState::new().await;
        let write_batch = populate_db_state(&mut test_db_state, entries).await?;

        let metrics_recorder = Arc::new(DefaultMetricsRecorder::new());
        let recorder = MetricsRecorderHelper::new(
            metrics_recorder.clone() as Arc<dyn MetricsRecorder>,
            MetricLevel::default(),
        );
        let db_stats = DbStats::new(&recorder);
        let reader = build_reader(&test_db_state, db_stats, true).await;

        assert_eq!(
            lookup_merge_operator_operands(metrics_recorder.as_ref(), MERGE_OPERATOR_READ_PATH),
            Some(0)
        );
        assert_eq!(
            lookup_merge_operator_operands(metrics_recorder.as_ref(), MERGE_OPERATOR_FLUSH_PATH,),
            Some(0)
        );

        let result = reader
            .get_key_value_with_options(
                b"key1",
                &ReadOptions::default().with_dirty(true),
                &test_db_state,
                wb_point_iter(&write_batch, b"key1"),
                None,
            )
            .await?;
        assert_eq!(result.unwrap().value, Bytes::from_static(b"ab"));
        assert_eq!(
            lookup_merge_operator_operands(metrics_recorder.as_ref(), MERGE_OPERATOR_READ_PATH),
            Some(3)
        );
        assert_eq!(
            lookup_merge_operator_operands(metrics_recorder.as_ref(), MERGE_OPERATOR_FLUSH_PATH,),
            Some(0)
        );

        let scan_range = BytesRange::from_slice(b"key1".as_ref()..b"key3".as_ref());
        let scan_options = ScanOptions::default().with_dirty(true);
        let wb_iter = wb_range_iter(&write_batch, &scan_range, scan_options.order);
        let mut iter = reader
            .scan_with_options(
                scan_range,
                &scan_options,
                ScanContext {
                    db_state: &test_db_state,
                    write_batch_iter: wb_iter,
                    max_seq: None,
                    prefix: None,
                },
            )
            .await?;

        let first = iter
            .next()
            .await
            .map_err(|e| SlateDBError::IoError(Arc::new(std::io::Error::other(e))))?
            .expect("should have key1");
        assert_eq!(first.key.as_ref(), b"key1");
        assert_eq!(first.value.as_ref(), b"ab");

        let second = iter
            .next()
            .await
            .map_err(|e| SlateDBError::IoError(Arc::new(std::io::Error::other(e))))?
            .expect("should have key2");
        assert_eq!(second.key.as_ref(), b"key2");
        assert_eq!(second.value.as_ref(), b"value2");

        assert!(iter
            .next()
            .await
            .map_err(|e| SlateDBError::IoError(Arc::new(std::io::Error::other(e))))?
            .is_none());

        assert_eq!(
            lookup_merge_operator_operands(metrics_recorder.as_ref(), MERGE_OPERATOR_READ_PATH),
            Some(6)
        );
        assert_eq!(
            lookup_merge_operator_operands(metrics_recorder.as_ref(), MERGE_OPERATOR_FLUSH_PATH,),
            Some(0)
        );

        Ok(())
    }

    #[tokio::test]
    async fn should_record_bloom_filter_negative_for_sorted_run_point_lookup(
    ) -> Result<(), SlateDBError> {
        let entries = vec![
            TestEntry::value(b"key1", b"value1", 50).with_location(LayerLocation::SortedRun(0)),
            TestEntry::value(b"key3", b"value3", 40).with_location(LayerLocation::SortedRun(0)),
        ];

        let mut test_db_state = TestDbState::new().await;
        let write_batch = populate_db_state(&mut test_db_state, entries).await?;

        let recorder = Arc::new(DefaultMetricsRecorder::new());
        let helper = MetricsRecorderHelper::new(recorder.clone(), MetricLevel::default());
        let db_stats = DbStats::new(&helper);
        let reader = build_reader(&test_db_state, db_stats, false).await;

        let result = reader
            .get_key_value_with_options(
                b"key2",
                &ReadOptions::default().with_dirty(true),
                &test_db_state,
                wb_point_iter(&write_batch, b"key2"),
                None,
            )
            .await?;

        assert!(result.is_none());
        let point_labels = &[(
            crate::db_stats::FILTER_KIND_LABEL,
            crate::db_stats::FILTER_KIND_POINT,
        )];
        assert_eq!(
            lookup_metric_with_labels(
                &recorder,
                crate::db_stats::SST_FILTER_NEGATIVE_COUNT,
                point_labels,
            ),
            Some(1)
        );
        assert_eq!(
            lookup_metric_with_labels(
                &recorder,
                crate::db_stats::SST_FILTER_POSITIVE_COUNT,
                point_labels,
            ),
            Some(0)
        );
        assert_eq!(
            lookup_metric_with_labels(
                &recorder,
                crate::db_stats::SST_FILTER_FALSE_POSITIVE_COUNT,
                point_labels,
            ),
            Some(0)
        );

        Ok(())
    }

    #[tokio::test]
    async fn should_return_expire_ts_on_get() -> Result<(), SlateDBError> {
        // given: a value with an expire_ts
        let entries = vec![
            TestEntry::value(b"key1", b"value1", 50).with_expire_ts(500),
            TestEntry::value(b"key2", b"value2", 50),
        ];

        let mut test_db_state = TestDbState::new().await;
        let write_batch = populate_db_state(&mut test_db_state, entries).await?;

        let recorder = slatedb_common::metrics::MetricsRecorderHelper::noop();
        let db_stats = DbStats::new(&recorder);
        let reader = build_reader(&test_db_state, db_stats, false).await;

        // when/then: get key1 should have expire_ts
        let result = reader
            .get_key_value_with_options(
                b"key1",
                &ReadOptions::default().with_dirty(true),
                &test_db_state,
                wb_point_iter(&write_batch, b"key1"),
                None,
            )
            .await?;
        let kv = result.expect("should return value");
        assert_eq!(kv.value.as_ref(), b"value1");
        assert_eq!(kv.expire_ts, Some(500));

        // when/then: get key2 should have no expire_ts
        let result = reader
            .get_key_value_with_options(
                b"key2",
                &ReadOptions::default().with_dirty(true),
                &test_db_state,
                wb_point_iter(&write_batch, b"key2"),
                None,
            )
            .await?;
        let kv = result.expect("should return value");
        assert_eq!(kv.value.as_ref(), b"value2");
        assert_eq!(kv.expire_ts, None);

        Ok(())
    }

    #[tokio::test]
    async fn should_return_expire_ts_on_scan() -> Result<(), SlateDBError> {
        // given: entries with mixed expire_ts
        let entries = vec![
            TestEntry::value(b"key1", b"val1", 50).with_expire_ts(100),
            TestEntry::value(b"key2", b"val2", 50),
            TestEntry::value(b"key3", b"val3", 50).with_expire_ts(300),
        ];

        let mut test_db_state = TestDbState::new().await;
        let write_batch = populate_db_state(&mut test_db_state, entries).await?;

        let recorder = slatedb_common::metrics::MetricsRecorderHelper::noop();
        let db_stats = DbStats::new(&recorder);
        let reader = build_reader(&test_db_state, db_stats, false).await;

        // when: scanning all keys
        let range = BytesRange::from_slice(b"key1".as_ref()..b"key4".as_ref());
        let scan_options = ScanOptions::default().with_dirty(true);
        let wb_iter = wb_range_iter(&write_batch, &range, scan_options.order);
        let mut iter = reader
            .scan_with_options(
                range,
                &scan_options,
                ScanContext {
                    db_state: &test_db_state,
                    write_batch_iter: wb_iter,
                    max_seq: None,
                    prefix: None,
                },
            )
            .await?;

        // then: each result should carry its expire_ts
        let kv1 = iter
            .next()
            .await
            .map_err(|e| SlateDBError::IoError(Arc::new(std::io::Error::other(e))))?
            .expect("should have key1");
        assert_eq!(kv1.key.as_ref(), b"key1");
        assert_eq!(kv1.expire_ts, Some(100));

        let kv2 = iter
            .next()
            .await
            .map_err(|e| SlateDBError::IoError(Arc::new(std::io::Error::other(e))))?
            .expect("should have key2");
        assert_eq!(kv2.key.as_ref(), b"key2");
        assert_eq!(kv2.expire_ts, None);

        let kv3 = iter
            .next()
            .await
            .map_err(|e| SlateDBError::IoError(Arc::new(std::io::Error::other(e))))?
            .expect("should have key3");
        assert_eq!(kv3.key.as_ref(), b"key3");
        assert_eq!(kv3.expire_ts, Some(300));

        Ok(())
    }

    #[tokio::test]
    async fn should_return_min_expire_ts_from_merged_entries() -> Result<(), SlateDBError> {
        // given: merge operands with different expire_ts across layers
        let entries = vec![
            TestEntry::merge(b"key1", b"d", 80).with_expire_ts(200),
            TestEntry::merge(b"key1", b"c", 70)
                .with_location(LayerLocation::L0Sst(1))
                .with_expire_ts(100),
            TestEntry::merge(b"key1", b"b", 60).with_location(LayerLocation::L0Sst(0)),
            TestEntry::value(b"key1", b"a", 50)
                .with_location(LayerLocation::SortedRun(0))
                .with_expire_ts(300),
        ];

        let mut test_db_state = TestDbState::new().await;
        let write_batch = populate_db_state(&mut test_db_state, entries).await?;

        let recorder = slatedb_common::metrics::MetricsRecorderHelper::noop();
        let db_stats = DbStats::new(&recorder);
        let reader = build_reader(&test_db_state, db_stats, true).await;

        // when: reading the merged key
        let result = reader
            .get_key_value_with_options(
                b"key1",
                &ReadOptions::default().with_dirty(true),
                &test_db_state,
                wb_point_iter(&write_batch, b"key1"),
                None,
            )
            .await?;

        // then: the result should have the min expire_ts across all operands
        let kv = result.expect("should return merged value");
        assert_eq!(kv.value.as_ref(), b"abcd");
        assert_eq!(
            kv.expire_ts,
            Some(100),
            "expire_ts should be the minimum across all merge operands"
        );

        Ok(())
    }

    #[tokio::test]
    async fn should_return_base_expire_ts_when_it_is_earliest() -> Result<(), SlateDBError> {
        // given: a base value whose expire_ts is earlier than any merge operand
        let entries = vec![
            TestEntry::merge(b"key1", b"b", 60).with_expire_ts(300),
            TestEntry::value(b"key1", b"a", 50)
                .with_location(LayerLocation::L0Sst(0))
                .with_expire_ts(50),
        ];

        let mut test_db_state = TestDbState::new().await;
        let write_batch = populate_db_state(&mut test_db_state, entries).await?;

        let recorder = slatedb_common::metrics::MetricsRecorderHelper::noop();
        let db_stats = DbStats::new(&recorder);
        let reader = build_reader(&test_db_state, db_stats, true).await;

        // when: reading the merged key
        let result = reader
            .get_key_value_with_options(
                b"key1",
                &ReadOptions::default().with_dirty(true),
                &test_db_state,
                wb_point_iter(&write_batch, b"key1"),
                None,
            )
            .await?;

        // then: expire_ts should come from the base value since it's the earliest
        let kv = result.expect("should return merged value");
        assert_eq!(kv.value.as_ref(), b"ab");
        assert_eq!(
            kv.expire_ts,
            Some(50),
            "expire_ts should be the base value's expire_ts since it is the earliest"
        );

        Ok(())
    }
}
