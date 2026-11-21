//! Online backfill of [`PAYLOAD_INDEX_PLAIN_ROW`] markers on LSM keyword
//! payload-index rows written before markers existed, so exact text walks
//! reach existing data without an index rebuild.
//!
//! Writer only, queued when a collection opens. Each batch scans node rows
//! from a cursor inside one [`LsmCheckedTxn`], marks the existing unmarked
//! row of every node whose value is a top-level string, and commits with
//! those node keys marked read: an upsert or delete of such a node after the
//! batch began refuses the commit (the batch is retried at half the size; a
//! single node that keeps conflicting is left unmarked), so a value turned
//! into an array never ends with a marked row. The commit runs under the
//! payload-index build lock once each index is still the same Ready handle,
//! so a dropped or rebuilt index never gets markers. The cursor, then
//! `done`, commits in the same transaction; an index this binary builds is
//! registered `done`.

use super::*;
use crate::helix_engine::storage_core::backend_lsm::LsmCheckedTxn;

/// Metadata key per field. Re-running the backfill does not repair a
/// rollback to a binary before markers: that binary neither maintains a
/// dropped index's (marked) rows nor clears them when it rebuilds, so exact
/// walks would trust stale markers. After rolling forward from one, set
/// `HELIX_TEXT_WALK_EXACT=0`, `HELIX_DENSE_TEXT_PREFILTER_MAX=0`,
/// `HELIX_SPARSE_LAZY_FILTER=0` and `HELIX_PAYLOAD_MARKER_BACKFILL=0` until
/// each index it touched is rebuilt on this binary.
const PROGRESS_KEY_PREFIX: &str = "payload_marker_backfill:v1:";
const PROGRESS_DONE: &[u8] = b"done";
/// Marker-free batches between cursor commits.
const PROGRESS_EVERY: u32 = 32;
const QUEUE_CAP: usize = 4096;

fn backfill_enabled() -> bool {
    env_bool("HELIX_PAYLOAD_MARKER_BACKFILL", !cfg!(test))
}

/// Fields `match.text` filters target (the client: directory and repo scopes).
fn backfill_fields() -> Vec<String> {
    std::env::var("HELIX_PAYLOAD_MARKER_BACKFILL_FIELDS")
        .unwrap_or_else(|_| "metadata.path,metadata.repo".to_string())
        .split(',')
        .map(str::trim)
        .filter(|field| !field.is_empty())
        .map(str::to_string)
        .collect()
}

fn batch_nodes() -> usize {
    env_usize_at_least("HELIX_PAYLOAD_MARKER_BACKFILL_BATCH", 512, 1)
}

fn batch_pause() -> Duration {
    Duration::from_millis(
        env_usize_at_least("HELIX_PAYLOAD_MARKER_BACKFILL_PAUSE_MS", 100, 0) as u64,
    )
}

fn progress_key(field: &str) -> Vec<u8> {
    format!("{PROGRESS_KEY_PREFIX}{field}").into_bytes()
}

/// The Metadata row, key and value, that records `field` as needing no
/// backfill.
pub(super) fn marker_backfill_done_row(field: &str) -> (Vec<u8>, &'static [u8]) {
    (progress_key(field), PROGRESS_DONE)
}

struct BackfillJob {
    collection: String,
    storage: Weak<HelixGraphStorage>,
}

static BACKFILL_QUEUE: LazyLock<Option<flume::Sender<BackfillJob>>> = LazyLock::new(|| {
    let (sender, receiver) = flume::bounded::<BackfillJob>(QUEUE_CAP);
    let spawned = std::thread::Builder::new()
        .name("helix-payload-marker-backfill".to_string())
        .spawn(move || {
            for job in receiver {
                let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    HelixGraphStorage::run_payload_marker_backfill(&job.storage)
                }));
                match run {
                    Ok(Ok(run)) if run.marked > 0 || run.completed => tracing::info!(
                        collection = %job.collection,
                        scanned = run.scanned,
                        marked = run.marked,
                        conflicts = run.conflicts,
                        skipped = run.skipped,
                        completed = run.completed,
                        "payload marker backfill ran"
                    ),
                    Ok(Ok(_)) => {}
                    Ok(Err(error)) => {
                        metrics::counter!("helix_payload_marker_backfill_errors_total")
                            .increment(1);
                        tracing::warn!(
                            collection = %job.collection,
                            %error,
                            "payload marker backfill stopped; the next open resumes it"
                        );
                    }
                    Err(_) => {
                        metrics::counter!("helix_payload_marker_backfill_errors_total")
                            .increment(1);
                        tracing::error!(
                            collection = %job.collection,
                            "payload marker backfill panicked; worker continuing"
                        );
                    }
                }
            }
        });
    match spawned {
        Ok(_) => Some(sender),
        Err(error) => {
            tracing::error!(%error, "failed to start payload marker backfill worker");
            None
        }
    }
});

/// What one [`HelixGraphStorage::run_payload_marker_backfill`] did.
#[derive(Debug, Default)]
pub(crate) struct MarkerBackfillRun {
    pub(crate) scanned: usize,
    pub(crate) marked: usize,
    pub(crate) conflicts: usize,
    pub(crate) completed: bool,
    /// Nodes left unmarked because writes to them kept refusing the commit.
    pub(crate) skipped: usize,
    /// An index changed generation (dropped, rebuilt) before a commit.
    pub(crate) superseded: bool,
    /// Marker-free batches since the last commit.
    uncommitted: u32,
}

#[cfg(test)]
thread_local! {
    static BEFORE_COMMIT_FOR_TEST: std::cell::RefCell<Option<Box<dyn FnMut()>>> =
        std::cell::RefCell::new(None);
}

/// Test-only: run `hook` between each batch's reads and its commit.
#[cfg(test)]
pub(crate) fn with_marker_backfill_hook_for_test<T>(
    hook: impl FnMut() + 'static,
    f: impl FnOnce() -> T,
) -> T {
    BEFORE_COMMIT_FOR_TEST.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    let result = f();
    BEFORE_COMMIT_FOR_TEST.with(|slot| *slot.borrow_mut() = None);
    result
}

impl HelixGraphStorage {
    /// Queue the marker backfill for a collection just opened on the LSM
    /// writer, when one of its backfilled fields has a Ready keyword index.
    /// Returns whether a job was queued.
    pub(crate) fn submit_payload_marker_backfill(self: &Arc<Self>, collection: &str) -> bool {
        if !matches!(self.backend.as_ref(), AnyBackend::Lsm(_))
            || !backfill_enabled()
            || self.marker_backfill_handles().is_empty()
        {
            return false;
        }
        let job = BackfillJob {
            collection: collection.to_string(),
            storage: Arc::downgrade(self),
        };
        let queued = BACKFILL_QUEUE
            .as_ref()
            .is_some_and(|queue| queue.try_send(job).is_ok());
        if !queued {
            tracing::warn!(
                collection,
                "payload marker backfill queue unavailable; the next open retries"
            );
        }
        queued
    }

    fn marker_backfill_handles(&self) -> Vec<(String, PayloadIndexHandle)> {
        let Ok(indices) = self.payload_indices.read() else {
            return Vec::new();
        };
        backfill_fields()
            .into_iter()
            .filter_map(|field| {
                let handle = indices.get(&field).filter(|handle| {
                    handle.schema == PayloadIndexSchema::Keyword && handle.is_ready()
                })?;
                Some((field, handle.clone()))
            })
            .collect()
    }

    /// Mark every pending field's rows, batch by batch, until done, the
    /// collection closes, or an index changes generation. Holds the storage
    /// only while a batch runs.
    pub(crate) fn run_payload_marker_backfill(
        storage: &Weak<Self>,
    ) -> Result<MarkerBackfillRun, GraphError> {
        let mut run = MarkerBackfillRun::default();
        let Some((fields, mut cursor)) = storage
            .upgrade()
            .map(|storage| storage.pending_marker_fields())
            .transpose()?
        else {
            return Ok(run);
        };
        if fields.is_empty() {
            return Ok(run);
        }
        loop {
            // The cache holds one reference: alone, the collection was evicted.
            let Some(storage) = storage.upgrade().filter(|s| Arc::strong_count(s) > 1) else {
                return Ok(run);
            };
            if !fields
                .iter()
                .all(|(field, handle)| storage.is_payload_index_generation(field, handle))
            {
                run.superseded = true;
                return Ok(run);
            }
            let Some((last, done)) = storage.payload_marker_batch(&fields, cursor, &mut run)?
            else {
                run.superseded = true;
                return Ok(run);
            };
            if done {
                metrics::counter!("helix_payload_marker_backfill_collections_completed_total")
                    .increment(1);
                run.completed = true;
                return Ok(run);
            }
            cursor = last;
            drop(storage);
            std::thread::sleep(batch_pause());
        }
    }

    /// Ready keyword fields not yet done, and the cursor to resume from (the
    /// least advanced field's).
    fn pending_marker_fields(
        &self,
    ) -> Result<(Vec<(String, PayloadIndexHandle)>, Option<u128>), GraphError> {
        let read = self
            .backend
            .begin_read()
            .map_err(graph_error_from_backend_error)?;
        let mut pending = Vec::new();
        let mut cursors = Vec::new();
        for (field, handle) in self.marker_backfill_handles() {
            let progress = self
                .backend
                .get_with(&read, Namespace::Metadata, &progress_key(&field), |v| {
                    v.map(<[u8]>::to_vec)
                })
                .map_err(graph_error_from_backend_error)?;
            if progress.as_deref() == Some(PROGRESS_DONE) {
                continue;
            }
            cursors.push(
                progress
                    .and_then(|bytes| <[u8; 16]>::try_from(bytes.as_slice()).ok())
                    .map(u128::from_be_bytes),
            );
            pending.push((field, handle));
        }
        // `None` (from the start) sorts first.
        Ok((pending, cursors.into_iter().min().flatten()))
    }

    /// One batch of up to [`batch_nodes`] nodes after `cursor`, retried on
    /// conflicts at half the size down to one node, which is then skipped
    /// unmarked (its writer marks it); commits markers, completion, or every
    /// [`PROGRESS_EVERY`]th cursor. `Some((last node id, done))` once
    /// through; `None` when an index is no longer the generation `fields`
    /// holds.
    /// Storage errors never mark the collection degraded: a drop can purge
    /// the files under a running scan.
    fn payload_marker_batch(
        &self,
        fields: &[(String, PayloadIndexHandle)],
        cursor: Option<u128>,
        run: &mut MarkerBackfillRun,
    ) -> Result<Option<(Option<u128>, bool)>, GraphError> {
        let AnyBackend::Lsm(lsm) = self.backend.as_ref() else {
            return Ok(None);
        };
        let db_names: Vec<String> = fields
            .iter()
            .map(|(field, handle)| Self::payload_index_db_name(field, &handle.schema))
            .collect();
        let mut limit = batch_nodes();
        for attempt in 1u64.. {
            let txn = lsm
                .begin_checked()
                .map_err(graph_error_from_backend_error)?;
            let (scanned, last, marked) = self
                .stage_payload_markers(&txn, fields, &db_names, cursor, limit)
                .map_err(graph_error_from_backend_error)?;
            let done = scanned < limit;
            if marked == 0 && !done && run.uncommitted + 1 < PROGRESS_EVERY {
                // Nothing to write; an interrupted run rescans from the last commit.
                run.uncommitted += 1;
                run.scanned += scanned;
                return Ok(Some((last, false)));
            }
            let progress = match last {
                Some(last) if !done => last.to_be_bytes().to_vec(),
                _ => PROGRESS_DONE.to_vec(),
            };
            for (field, _) in fields {
                txn.put(Namespace::Metadata, &progress_key(field), &progress)
                    .map_err(graph_error_from_backend_error)?;
            }
            #[cfg(test)]
            BEFORE_COMMIT_FOR_TEST.with(|slot| {
                let hook = slot.borrow_mut().take();
                if let Some(mut hook) = hook {
                    hook();
                    *slot.borrow_mut() = Some(hook);
                }
            });
            let committed = {
                let _build = self.payload_index_build_lock()?;
                if !fields
                    .iter()
                    .all(|(field, handle)| self.is_payload_index_generation(field, handle))
                {
                    return Ok(None);
                }
                txn.commit().map_err(graph_error_from_backend_error)?
            };
            if committed {
                metrics::counter!("helix_payload_marker_backfill_rows_total")
                    .increment(marked as u64);
                run.uncommitted = 0;
                run.scanned += scanned;
                run.marked += marked;
                return Ok(Some((last, done)));
            }
            run.conflicts += 1;
            if limit == 1 {
                // A conflict needs a staged marker: the one node scanned.
                run.skipped += 1;
                run.scanned += scanned;
                return Ok(Some((last, false)));
            }
            limit /= 2;
            std::thread::sleep(Duration::from_millis(10 * attempt));
        }
        unreachable!("the attempt loop returns")
    }

    /// Record `field` as needing no backfill, in the write that registers an
    /// index this binary built.
    pub(super) fn put_payload_marker_backfill_done(
        &self,
        w: &mut AnyWrite<'_>,
        field: &str,
    ) -> Result<(), GraphError> {
        let (key, done) = marker_backfill_done_row(field);
        self.backend
            .put(w, Namespace::Metadata, &key, done)
            .map_err(graph_error_from_backend_error)
    }

    /// Scan up to `limit` nodes after `cursor` and stage a marker on each
    /// unmarked row `(key of a top-level string value, node id)`, with the
    /// node key marked read. Returns (nodes scanned, last id, rows staged).
    fn stage_payload_markers(
        &self,
        txn: &LsmCheckedTxn<'_>,
        fields: &[(String, PayloadIndexHandle)],
        db_names: &[String],
        cursor: Option<u128>,
        limit: usize,
    ) -> Result<(usize, Option<u128>, usize), BackendError> {
        let range = KeyRange {
            start: cursor.map_or(Bound::Unbounded, |id| {
                Bound::Excluded(id.to_be_bytes().to_vec())
            }),
            end: Bound::Unbounded,
        };
        let mut rows: Vec<Vec<(Vec<u8>, [u8; 16])>> = vec![Vec::new(); fields.len()];
        let (mut scanned, mut last) = (0usize, None);
        txn.scan(Namespace::Nodes, range, |key, bytes| {
            let Ok(id) = <[u8; 16]>::try_from(key) else {
                return true;
            };
            scanned += 1;
            last = Some(u128::from_be_bytes(id));
            if let Ok(node) = SerializedNode::decode_node(bytes, u128::from_be_bytes(id)) {
                for ((field, _), rows) in fields.iter().zip(rows.iter_mut()) {
                    if let Some(value @ Value::String(_)) =
                        Self::payload_value_for_key(&node.properties, field)
                    {
                        if let Ok(key) = Self::stable_index_key_for_value(value) {
                            rows.push((key, id));
                        }
                    }
                }
            }
            scanned < limit
        })?;
        let mut marked = 0usize;
        for (rows, db_name) in rows.iter().zip(db_names) {
            let ns = Namespace::PayloadIndex(db_name);
            let lookups: Vec<(&[u8], &[u8])> = rows
                .iter()
                .map(|(key, id)| (key.as_slice(), id.as_slice()))
                .collect();
            for ((key, id), stored) in rows.iter().zip(txn.get_dup_stored(ns, &lookups)?) {
                // Rows older binaries wrote store nothing; absent rows stay absent.
                if stored.is_some_and(|stored| stored.is_empty()) {
                    txn.mark_read(Namespace::Nodes, id)?;
                    txn.put_dup_stored(ns, key, id, PAYLOAD_INDEX_PLAIN_ROW)?;
                    marked += 1;
                }
            }
        }
        Ok((scanned, last, marked))
    }

    /// `handle` is still `field`'s index and Ready. Under the build lock this
    /// holds until the lock is released: only a drop (under it) removes a
    /// Ready index, and only an absent one is replaced.
    fn is_payload_index_generation(&self, field: &str, handle: &PayloadIndexHandle) -> bool {
        self.payload_indices.read().is_ok_and(|indices| {
            indices.get(field).is_some_and(|current| {
                Arc::ptr_eq(&current.state, &handle.state) && current.is_ready()
            })
        })
    }

    /// Test-only: rewrite every row of `field`'s keyword index the way
    /// binaries before markers wrote it (no stored value, no progress key).
    #[cfg(test)]
    pub(crate) fn strip_payload_markers_for_test(&self, field: &str) {
        let db_name = Self::payload_index_db_name(field, &PayloadIndexSchema::Keyword);
        let ns = Namespace::PayloadIndex(&db_name);
        let read = self.backend.begin_read().unwrap();
        let mut rows = Vec::new();
        self.backend
            .scan_dup_stored(&read, ns, KeyRange::all(), |key, id, _| {
                rows.push((key.to_vec(), id.to_vec()));
                true
            })
            .unwrap();
        drop(read);
        let mut write = self.backend.begin_write().unwrap();
        for (key, id) in rows {
            self.backend.put_dup(&mut write, ns, &key, &id).unwrap();
        }
        self.backend.commit(write).unwrap();
        self.forget_payload_marker_backfill_for_test(field);
    }

    /// Test-only: drop `field`'s backfill progress, as binaries before
    /// markers never wrote it.
    #[cfg(test)]
    pub(crate) fn forget_payload_marker_backfill_for_test(&self, field: &str) {
        let mut write = self.backend.begin_write().unwrap();
        self.backend
            .delete(&mut write, Namespace::Metadata, &progress_key(field))
            .unwrap();
        self.backend.commit(write).unwrap();
    }
}

#[cfg(test)]
#[path = "payload_marker_backfill_tests.rs"]
mod tests;
