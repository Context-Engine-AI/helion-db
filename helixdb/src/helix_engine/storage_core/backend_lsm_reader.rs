//! `LsmReader` — a read-only handle over a SlateDB database (Phase 5: the
//! writer/reader split).
//!
//! SlateDB ships a [`slatedb::DbReader`] that opens a database read-only against
//! object storage and serves *committed* state — the basis for horizontally
//! scaling readers that consume the writer's data off S3 without holding the
//! single-writer lock. This module wraps `DbReader` the same way
//! [`super::backend_lsm::LsmBackend`] wraps `Db`: a dedicated multi-thread tokio
//! runtime bridges SlateDB's async API to Helion's synchronous call sites via
//! `rt.block_on(...)`, and keys are prefixed with the EXACT same `<ns_name>\0`
//! scheme so a reader observes the writer's keys byte-for-byte.
//!
//! Read-visibility semantics (SlateDB 0.13.1): the writer's commit goes through
//! `Db::write` with `WriteOptions { await_durable: true }`, and the default
//! `Settings` has `wal_enabled = true`, so a commit only returns once the batch
//! is durably persisted as a WAL SST on object storage. A freshly-opened
//! `DbReader` (default `DbReaderOptions`, `skip_wal_replay = false`) replays new
//! WALs on open, so it sees the writer's just-committed data even before that
//! data is compacted into L0. While the reader stays open it polls the manifest
//! / WAL every `manifest_poll_interval` (default 10s) to pick up later writes.

#![allow(dead_code)]

use std::ops::Bound;
use std::sync::Arc;

use futures::{future::try_join_all, stream, StreamExt, TryStreamExt};
use object_store::aws::AmazonS3Builder;
use object_store::ObjectStore;
use slatedb::config::ScanOptions;
use slatedb::{DbReader, DbSnapshot};
use tokio::runtime::Handle;

use super::backend::{is_dup, next_prefix, BackendError, KeyRange, Namespace};
use super::backend_lsm::{
    block_on_lsm, block_on_lsm_mapped, block_on_lsm_read, block_on_lsm_read_mapped,
    collections_root_prefix, decode_dup_full_key, dup_composite, dup_key_prefix,
    ensure_lsm_read_not_cancelled, list_child_collection_names, logical_key_in_range,
    lsm_multi_get_enabled, lsm_scan_options, lsm_streaming_scan_options, ns_prefix,
    prefix_filter_policies_from_env, prefixed, range_covers_all_logical_keys,
    reader_options_from_env, scan_prefix_for_namespace_range, shared_db_cache, shared_lsm_handle,
    shared_merge_operator, LsmBackend,
};

fn reader_io<E: std::fmt::Display>(e: E) -> BackendError {
    let msg = e.to_string();
    // The vendored SlateDB fork's snapshot generations fail LOUDLY when their
    // checkpoint lease is reaped ("reader checkpoint lease lost") instead of
    // silently reading a truncated view. Surface it distinctly so the gateway's
    // reader-eviction/fallback path (which reopens a fresh checkpoint) is
    // attributable in metrics.
    if msg.contains("checkpoint lease lost") {
        metrics::counter!("helix_lsm_reader_snapshot_lease_lost_total").increment(1);
    }
    BackendError::Io(msg)
}

/// Stable marker carried by the reader-open error when SlateDB reports
/// [`slatedb::ErrorCode::DatabaseMissing`] (no manifest at the collection
/// prefix). The error is stringified on its way through `GraphError`, so the
/// collection manager matches this marker — set only by the typed check in
/// [`reader_open_err`] — to surface a collection-not-found (404). Must stay
/// clear of `lsm_object_reference_is_missing` phrasing so it never quarantines.
pub const LSM_READER_DATABASE_MISSING: &str = "LSM reader: database missing";

fn reader_open_err(e: slatedb::Error) -> BackendError {
    if e.code() == Some(slatedb::ErrorCode::DatabaseMissing) {
        return BackendError::Io(format!("{LSM_READER_DATABASE_MISSING}: {e}"));
    }
    BackendError::Io(e.to_string())
}

const LSM_READER_BATCH_POINT_READ_CONCURRENCY: usize = 96;

/// Env knob: pin every reader-replica read transaction to a point-in-time
/// `DbSnapshot` (vendored fork `DbReader::snapshot()`, an O(1) local
/// operation) so all reads within one logical read txn observe ONE committed
/// state instead of racing the manifest poller's checkpoint swaps mid-request.
/// Default OFF: flip on only after the fleet runs the vendored fork.
const ENV_READER_SNAPSHOT_READS: &str = "HELIX_LSM_READER_SNAPSHOT_READS";

pub(crate) fn reader_snapshot_reads_enabled() -> bool {
    std::env::var(ENV_READER_SNAPSHOT_READS)
        .ok()
        .map(|v| matches!(v.trim(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

/// Read-only handle over a SlateDB database on object storage.
///
/// Reads committed state written by an [`super::backend_lsm::LsmBackend`] writer
/// (or any SlateDB writer using the same key encoding). Holds no writer lock, so
/// many `LsmReader`s can read the same database concurrently with the writer.
pub struct LsmReader {
    /// Hot-swappable so the reader-replica change-feed poller can reopen the
    /// `DbReader` (fresh checkpoint + new `manifest_poll_interval` tier) in
    /// place while in-flight reads keep the previous instance alive via `Arc`.
    reader: std::sync::RwLock<Arc<DbReader>>,
    /// Bumped under the `reader` write lock on every swap. A single
    /// `DbReader`'s view only moves forward, but a swapped-in reader may sit
    /// at an OLDER state than the one it replaces, so callers that validate
    /// a sequence of latest-state reads (e.g. the sparse cache epoch check)
    /// must also see an unchanged generation.
    generation: std::sync::atomic::AtomicU64,
    /// `Handle` to the process-shared LSM runtime (see
    /// [`super::backend_lsm::shared_lsm_runtime`]), NOT an owned runtime — so N
    /// reader replicas / collections share one worker pool.
    rt: Handle,
    /// Retained object store + SlateDB prefix (= `<root>/<collection>`) so a
    /// reader replica can enumerate the object-store catalog
    /// ([`Self::list_collection_prefixes`]) without reopening the store.
    store: Arc<dyn ObjectStore>,
    path: String,
}

impl LsmReader {
    /// Open a read-only SlateDB reader at `path` backed by `store`.
    ///
    /// The database must already be initialized by a writer (the underlying
    /// `DbReader` errors with `InvalidDBState` against an empty path). Opening
    /// with default [`DbReaderOptions`] establishes a self-refreshing checkpoint
    /// against the latest manifest and replays committed WALs so the reader sees
    /// the writer's durable state.
    /// When `HELIX_LSM_CACHE_DIR` is set, the reader wraps its object store in a
    /// local-filesystem cache (via `DbReaderOptions.object_store_cache_options`)
    /// so reads are served off the SSD mount and S3 is hit only on cache miss —
    /// the same hot-cache config the writer uses. Unset → cache-less open
    /// (current behavior).
    pub fn open_with_store(path: &str, store: Arc<dyn ObjectStore>) -> Result<Self, BackendError> {
        let rt = shared_lsm_handle();
        let options = reader_options_from_env(path);
        let reader = Self::build_reader(&rt, path, store.clone(), options)?;
        Ok(Self {
            reader: std::sync::RwLock::new(Arc::new(reader)),
            generation: std::sync::atomic::AtomicU64::new(0),
            rt,
            store,
            path: path.to_string(),
        })
    }

    /// Build a `DbReader` for `path` with the shared cache/merge-operator/filter
    /// wiring. Shared by [`Self::open_with_store`] and
    /// [`Self::refresh_with_poll_interval_inner`] so both open identically.
    /// The build itself is never dropped: it writes a checkpoint and can start
    /// a manifest poller before resolving. Cancellation is sampled immediately
    /// before and after the build.
    fn build_reader(
        rt: &Handle,
        path: &str,
        store: Arc<dyn ObjectStore>,
        options: slatedb::config::DbReaderOptions,
    ) -> Result<DbReader, BackendError> {
        let mut builder = DbReader::builder(path, store)
            .with_options(options)
            .with_db_cache(shared_db_cache())
            .with_merge_operator(shared_merge_operator());
        if let Some(policies) = prefix_filter_policies_from_env() {
            builder = builder.with_filter_policies(policies);
        }
        ensure_lsm_read_not_cancelled()?;
        let reader = block_on_lsm_mapped(rt, builder.build(), reader_open_err)?;
        if let Err(cancelled) = ensure_lsm_read_not_cancelled() {
            // The side-effecting build completed; explicitly tear down its
            // checkpoint/poller before surfacing request cancellation.
            let _ = block_on_lsm(rt, reader.close());
            return Err(cancelled);
        }
        Ok(reader)
    }

    /// The current `DbReader`, cloned out of the swap slot so reads never hold
    /// the lock across an await.
    fn current_reader(&self) -> Arc<DbReader> {
        self.reader
            .read()
            .map(|guard| Arc::clone(&guard))
            .unwrap_or_else(|poisoned| Arc::clone(&poisoned.into_inner()))
    }

    /// Reader-swap generation; see the `generation` field.
    pub(crate) fn generation(&self) -> u64 {
        self.generation.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Reopen the underlying `DbReader` against the CURRENT manifest with the
    /// given `manifest_poll_interval`, swapping it in atomically. Opening a
    /// fresh reader establishes a new checkpoint immediately, so this doubles
    /// as an on-demand refresh (SlateDB exposes no manual-refresh API). The
    /// replaced reader is closed on a delayed background task: `DbReader` has
    /// no `Drop` shutdown, so skipping `close()` would leak its poll task —
    /// but closing immediately could fail reads still borrowing the old `Arc`.
    /// The delay outlasts any synchronous read bridged via `block_on_lsm`.
    pub fn refresh_with_poll_interval(
        &self,
        interval: std::time::Duration,
    ) -> Result<(), BackendError> {
        self.refresh_with_poll_interval_inner(interval)
    }

    /// Like [`Self::refresh_with_poll_interval`]. `budget` remains in the API
    /// for compatibility, but a reader build cannot be aborted safely after it
    /// writes its checkpoint/starts its poller, so the caller waits for it to
    /// complete. The per-collection poll-tier transition gate still prevents
    /// duplicate concurrent refreshes for the same collection.
    pub fn refresh_with_poll_interval_bounded(
        &self,
        interval: std::time::Duration,
        _budget: std::time::Duration,
    ) -> Result<(), BackendError> {
        self.refresh_with_poll_interval_inner(interval)
    }

    fn refresh_with_poll_interval_inner(
        &self,
        interval: std::time::Duration,
    ) -> Result<(), BackendError> {
        let mut options = reader_options_from_env(&self.path);
        options.manifest_poll_interval = interval;
        super::backend_lsm::clamp_reader_checkpoint_lifetime(&mut options);
        let fresh = Self::build_reader(&self.rt, &self.path, self.store.clone(), options)?;
        let previous = {
            let mut slot = self
                .reader
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            self.generation
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            std::mem::replace(&mut *slot, Arc::new(fresh))
        };
        self.rt.spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            if let Err(error) = previous.close().await {
                tracing::debug!(%error, "closing replaced DbReader after refresh failed");
            }
        });
        Ok(())
    }

    /// List the sibling collection names in the object store: the immediate
    /// children of this reader's root prefix (the parent of `self.path`). A
    /// reader replica uses this to enumerate the writer's catalog off the shared
    /// object store.
    pub fn list_collection_prefixes(&self) -> Result<Vec<String>, BackendError> {
        let root = collections_root_prefix(&self.path);
        list_child_collection_names(&self.rt, &self.store, root)
    }

    pub fn storage_bytes(&self) -> Result<u64, BackendError> {
        let store = Arc::clone(&self.store);
        let prefix = object_store::path::Path::from(self.path.trim_matches('/'));
        block_on_lsm_read(&self.rt, async move {
            let mut entries = store.list(Some(&prefix));
            let mut total = 0_u64;
            while let Some(entry) = entries.try_next().await.map_err(reader_io)? {
                total = total.saturating_add(entry.size as u64);
            }
            Ok::<u64, BackendError>(total)
        })
    }

    pub(crate) fn close_reason(&self) -> Option<slatedb::CloseReason> {
        self.current_reader().status().close_reason
    }

    /// Open a reader replica on AWS S3 (or an S3-compatible endpoint), the
    /// read-only twin of [`super::backend_lsm::LsmBackend::open_s3`]. Resolves
    /// credentials from the environment and reads the writer's committed state at
    /// `prefix` in `bucket`. `endpoint`/`allow_http` target MinIO and other
    /// S3-compatible stores; `region` overrides the environment when provided.
    pub fn open_s3_with_options(
        bucket: &str,
        prefix: &str,
        region: Option<&str>,
        endpoint: Option<&str>,
        allow_http: bool,
    ) -> Result<Self, BackendError> {
        let mut builder = AmazonS3Builder::from_env().with_bucket_name(bucket);
        if let Some(r) = region {
            builder = builder.with_region(r);
        }
        if let Some(endpoint) = endpoint {
            builder = builder.with_endpoint(endpoint);
        }
        if allow_http {
            builder = builder.with_allow_http(true);
        }
        builder = super::backend_lsm::apply_s3_timeout(builder);
        let store = builder
            .build()
            .map_err(|e| BackendError::Io(e.to_string()))?;
        Self::open_with_store(prefix, Arc::new(store))
    }

    /// Open a reader replica on AWS S3 with environment/IRSA credentials.
    pub fn open_s3(bucket: &str, prefix: &str, region: Option<&str>) -> Result<Self, BackendError> {
        Self::open_s3_with_options(bucket, prefix, region, None, false)
    }

    /// Pin a point-in-time snapshot of the reader's latest fully-applied state
    /// (vendored fork `DbReader::snapshot()`): an O(1) local operation — no
    /// object-store I/O — whose checkpoint generation is lease-managed by the
    /// reader's manifest poller (one BATCHED manifest append per poll tick
    /// regardless of live snapshot count). All reads served through the
    /// returned handle observe one committed sequence; if the generation's
    /// checkpoint is reaped mid-read the fork fails loudly with
    /// "reader checkpoint lease lost" (see [`reader_io`]).
    pub(crate) fn begin_snapshot(&self) -> Result<Arc<DbSnapshot>, BackendError> {
        let reader = self.current_reader();
        block_on_lsm_read_mapped(&self.rt, reader.snapshot(), reader_io)
    }

    /// Read the value at (`ns`, `key`) from committed state, handing the borrowed
    /// bytes (or `None`) to `f`. Keys are prefixed identically to
    /// [`super::backend_lsm::LsmBackend`] so the reader resolves the same physical
    /// key the writer wrote.
    pub fn get_with<R>(
        &self,
        ns: Namespace<'_>,
        key: &[u8],
        f: impl FnOnce(Option<&[u8]>) -> R,
    ) -> Result<R, BackendError> {
        self.get_with_at(None, ns, key, f)
    }

    /// [`Self::get_with`] served from `snap` when present (snapshot-pinned read
    /// txn), else from the reader's live latest-checkpoint state.
    pub(crate) fn get_with_at<R>(
        &self,
        snap: Option<&Arc<DbSnapshot>>,
        ns: Namespace<'_>,
        key: &[u8],
        f: impl FnOnce(Option<&[u8]>) -> R,
    ) -> Result<R, BackendError> {
        let full = prefixed(ns, key);
        let val = match snap {
            Some(snapshot) => {
                block_on_lsm_read_mapped(&self.rt, snapshot.get(full.as_slice()), reader_io)?
            }
            None => {
                let reader = self.current_reader();
                block_on_lsm_read(&self.rt, reader.get(full.as_slice()))?
            }
        };
        Ok(f(val.as_deref()))
    }

    pub(crate) fn collect_values_many_with(
        &self,
        ns: Namespace<'_>,
        keys: &[Vec<u8>],
    ) -> Result<Vec<Option<Vec<u8>>>, BackendError> {
        self.collect_values_many_with_at(None, ns, keys)
    }

    pub(crate) fn collect_values_many_with_at(
        &self,
        snap: Option<&Arc<DbSnapshot>>,
        ns: Namespace<'_>,
        keys: &[Vec<u8>],
    ) -> Result<Vec<Option<Vec<u8>>>, BackendError> {
        let full_keys: Vec<Vec<u8>> = keys.iter().map(|key| prefixed(ns, key)).collect();
        if lsm_multi_get_enabled() {
            // One ordered multi_get pass (fork API): shares block fetches
            // across keys and preserves input order.
            let fetched = match snap {
                Some(snapshot) => {
                    let snapshot = Arc::clone(snapshot);
                    block_on_lsm_read_mapped(
                        &self.rt,
                        async move { snapshot.multi_get(&full_keys).await },
                        reader_io,
                    )?
                }
                None => {
                    let reader = self.current_reader();
                    block_on_lsm_read_mapped(
                        &self.rt,
                        async move { reader.multi_get(&full_keys).await },
                        reader_io,
                    )?
                }
            };
            return Ok(fetched
                .into_iter()
                .map(|value| value.map(|bytes| bytes.to_vec()))
                .collect());
        }
        match snap {
            Some(snapshot) => {
                let snapshot = Arc::clone(snapshot);
                block_on_lsm_read_mapped(
                    &self.rt,
                    async move {
                        let reads = full_keys.into_iter().map(|full| {
                            let snapshot = Arc::clone(&snapshot);
                            async move {
                                snapshot
                                    .get(full.as_slice())
                                    .await
                                    .map(|value| value.map(|bytes| bytes.to_vec()))
                                    .map_err(reader_io)
                            }
                        });
                        stream::iter(reads)
                            .buffered(LSM_READER_BATCH_POINT_READ_CONCURRENCY)
                            .try_collect::<Vec<_>>()
                            .await
                    },
                    |e| e,
                )
            }
            None => {
                let reader = self.current_reader();
                block_on_lsm_read_mapped(
                    &self.rt,
                    async {
                        let reads = full_keys.into_iter().map(|full| {
                            let reader = Arc::clone(&reader);
                            async move {
                                reader
                                    .get(full.as_slice())
                                    .await
                                    .map(|value| value.map(|bytes| bytes.to_vec()))
                                    .map_err(reader_io)
                            }
                        });
                        stream::iter(reads)
                            .buffered(LSM_READER_BATCH_POINT_READ_CONCURRENCY)
                            .try_collect::<Vec<_>>()
                            .await
                    },
                    |e| e,
                )
            }
        }
    }

    /// True if the multi-value namespace `ns` holds `value` under `key`.
    ///
    /// The writer stores each dup pair as a composite key `(key, value) -> []`
    /// (see [`super::backend_lsm::dup_composite`]), so existence is a point lookup
    /// of that composite key — decoded byte-for-byte the way the writer wrote it.
    pub fn contains_dup(
        &self,
        ns: Namespace<'_>,
        key: &[u8],
        value: &[u8],
    ) -> Result<bool, BackendError> {
        self.contains_dup_at(None, ns, key, value)
    }

    pub(crate) fn contains_dup_at(
        &self,
        snap: Option<&Arc<DbSnapshot>>,
        ns: Namespace<'_>,
        key: &[u8],
        value: &[u8],
    ) -> Result<bool, BackendError> {
        let ck = dup_composite(ns, key, value);
        let got = match snap {
            Some(snapshot) => {
                block_on_lsm_read_mapped(&self.rt, snapshot.get(ck.as_slice()), reader_io)?
            }
            None => {
                let reader = self.current_reader();
                block_on_lsm_read(&self.rt, reader.get(ck.as_slice()))?
            }
        };
        Ok(got.is_some())
    }

    /// Visit every value stored under (`ns`, `key`) in a multi-value namespace,
    /// in key order, until `visit` returns `false`. Mirrors the writer's
    /// `for_each_dup`: scans the `<ns><key_len><key>` prefix and hands back the
    /// trailing value bytes.
    pub fn for_each_dup_with(
        &self,
        ns: Namespace<'_>,
        key: &[u8],
        visit: impl FnMut(&[u8]) -> bool,
    ) -> Result<(), BackendError> {
        self.for_each_dup_with_at(None, ns, key, visit)
    }

    pub(crate) fn for_each_dup_with_at(
        &self,
        snap: Option<&Arc<DbSnapshot>>,
        ns: Namespace<'_>,
        key: &[u8],
        mut visit: impl FnMut(&[u8]) -> bool,
    ) -> Result<(), BackendError> {
        let kp = dup_key_prefix(ns, key);
        let reader = self.current_reader();
        block_on_lsm_read_mapped(
            &self.rt,
            async {
                let mut iter = match snap {
                    Some(snapshot) => {
                        snapshot
                            .scan_prefix_with_options(kp.clone(), .., &lsm_scan_options())
                            .await
                    }
                    None => {
                        reader
                            .scan_prefix_with_options(kp.clone(), .., &lsm_scan_options())
                            .await
                    }
                }
                .map_err(reader_io)?;
                while let Some(kv) = iter.next().await.map_err(reader_io)? {
                    let k = kv.key.as_ref();
                    let value: &[u8] = if k.len() >= kp.len() {
                        &k[kp.len()..]
                    } else {
                        &[]
                    };
                    if !visit(value) {
                        break;
                    }
                }
                Ok::<(), BackendError>(())
            },
            |e| e,
        )
    }

    pub(crate) fn collect_dup_values_many_with(
        &self,
        ns: Namespace<'_>,
        keys: &[Vec<u8>],
    ) -> Result<Vec<Vec<Vec<u8>>>, BackendError> {
        self.collect_dup_values_many_with_at(None, ns, keys)
    }

    pub(crate) fn collect_dup_values_many_with_at(
        &self,
        snap: Option<&Arc<DbSnapshot>>,
        ns: Namespace<'_>,
        keys: &[Vec<u8>],
    ) -> Result<Vec<Vec<Vec<u8>>>, BackendError> {
        let prefixes: Vec<Vec<u8>> = keys.iter().map(|key| dup_key_prefix(ns, key)).collect();
        let reader = self.current_reader();
        block_on_lsm_read_mapped(
            &self.rt,
            async {
                let scans = prefixes.into_iter().map(|kp| {
                    let reader = Arc::clone(&reader);
                    let snap = snap.map(Arc::clone);
                    async move {
                        let mut values = Vec::new();
                        let mut iter = match &snap {
                            Some(snapshot) => {
                                snapshot
                                    .scan_prefix_with_options(kp.clone(), .., &lsm_scan_options())
                                    .await
                            }
                            None => {
                                reader
                                    .scan_prefix_with_options(kp.clone(), .., &lsm_scan_options())
                                    .await
                            }
                        }
                        .map_err(reader_io)?;
                        while let Some(kv) = iter.next().await.map_err(reader_io)? {
                            let k = kv.key.as_ref();
                            let value = if k.len() >= kp.len() {
                                &k[kp.len()..]
                            } else {
                                &[]
                            };
                            values.push(value.to_vec());
                        }
                        Ok::<Vec<Vec<u8>>, BackendError>(values)
                    }
                });
                try_join_all(scans).await
            },
            |e| e,
        )
    }

    /// Scan a multi-value namespace over `range`, visiting each
    /// `(logical_key, logical_value)` in key order until `visit` returns `false`.
    /// Mirrors the writer's `scan_dup_namespace` and decodes every composite row
    /// with the shared [`super::backend_lsm::decode_dup_full_key`], so the replica
    /// reproduces the writer's dup rows byte-for-byte.
    fn scan_dup_with_options(
        &self,
        snap: Option<&Arc<DbSnapshot>>,
        ns: Namespace<'_>,
        range: KeyRange,
        scan_options: ScanOptions,
        mut visit: impl FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        let prefix = ns_prefix(ns);
        let lo: Bound<Vec<u8>> = Bound::Included(prefix.clone());
        let hi: Bound<Vec<u8>> = match next_prefix(&prefix) {
            Some(end) => Bound::Excluded(end),
            None => Bound::Unbounded,
        };
        let scan_prefix = if range_covers_all_logical_keys(&range) {
            Some(prefix.clone())
        } else {
            None
        };
        let reader = self.current_reader();
        block_on_lsm_read_mapped(
            &self.rt,
            async {
                let mut iter = match (snap, scan_prefix) {
                    (Some(snapshot), Some(scan_prefix)) => {
                        snapshot
                            .scan_prefix_with_options(scan_prefix, .., &scan_options)
                            .await
                    }
                    (Some(snapshot), None) => {
                        snapshot.scan_with_options((lo, hi), &scan_options).await
                    }
                    (None, Some(scan_prefix)) => {
                        reader
                            .scan_prefix_with_options(scan_prefix, .., &scan_options)
                            .await
                    }
                    (None, None) => reader.scan_with_options((lo, hi), &scan_options).await,
                }
                .map_err(reader_io)?;
                while let Some(kv) = iter.next().await.map_err(reader_io)? {
                    let (logical_key, logical_value) =
                        match decode_dup_full_key(kv.key.as_ref(), &prefix)? {
                            Some(parts) => parts,
                            None => continue,
                        };
                    if !logical_key_in_range(logical_key, &range) {
                        continue;
                    }
                    if !visit(logical_key, logical_value) {
                        break;
                    }
                }
                Ok::<(), BackendError>(())
            },
            |e| e,
        )
    }

    pub fn scan_dup_with(
        &self,
        ns: Namespace<'_>,
        range: KeyRange,
        visit: impl FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        self.scan_dup_with_options(None, ns, range, lsm_scan_options(), visit)
    }

    pub(crate) fn scan_streaming(
        &self,
        ns: Namespace<'_>,
        range: KeyRange,
        visit: impl FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        self.scan_streaming_at(None, ns, range, visit)
    }

    pub(crate) fn scan_streaming_at(
        &self,
        snap: Option<&Arc<DbSnapshot>>,
        ns: Namespace<'_>,
        range: KeyRange,
        visit: impl FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        self.scan_with_lsm_options(snap, ns, range, lsm_streaming_scan_options(), visit)
    }

    fn scan_with_lsm_options(
        &self,
        snap: Option<&Arc<DbSnapshot>>,
        ns: Namespace<'_>,
        range: KeyRange,
        scan_options: ScanOptions,
        mut visit: impl FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        if is_dup(ns) {
            return self.scan_dup_with_options(snap, ns, range, scan_options, visit);
        }
        let prefix = ns_prefix(ns);
        let lo: Bound<Vec<u8>> = match &range.start {
            Bound::Unbounded => Bound::Included(prefix.clone()),
            other => LsmBackend::prefixed_bound(&prefix, other),
        };
        let hi: Bound<Vec<u8>> = match &range.end {
            Bound::Unbounded => match next_prefix(&prefix) {
                Some(end) => Bound::Excluded(end),
                None => Bound::Unbounded,
            },
            other => LsmBackend::prefixed_bound(&prefix, other),
        };
        let scan_prefix = scan_prefix_for_namespace_range(&prefix, &range);
        let reader = self.current_reader();
        block_on_lsm_read_mapped(
            &self.rt,
            async {
                let mut iter = match (snap, scan_prefix) {
                    (Some(snapshot), Some(scan_prefix)) => {
                        snapshot
                            .scan_prefix_with_options(scan_prefix, .., &scan_options)
                            .await
                    }
                    (Some(snapshot), None) => {
                        snapshot.scan_with_options((lo, hi), &scan_options).await
                    }
                    (None, Some(scan_prefix)) => {
                        reader
                            .scan_prefix_with_options(scan_prefix, .., &scan_options)
                            .await
                    }
                    (None, None) => reader.scan_with_options((lo, hi), &scan_options).await,
                }
                .map_err(reader_io)?;
                while let Some(kv) = iter.next().await.map_err(reader_io)? {
                    let k = kv.key.as_ref();
                    let stripped = if k.len() >= prefix.len() {
                        &k[prefix.len()..]
                    } else {
                        k
                    };
                    if !visit(stripped, kv.value.as_ref()) {
                        break;
                    }
                }
                Ok::<(), BackendError>(())
            },
            |e| e,
        )
    }

    /// Ordered scan over `range` in `ns`, mirroring the writer's `scan`: dup
    /// namespaces decode composite rows via [`scan_dup_with`](Self::scan_dup_with);
    /// plain namespaces strip the namespace prefix and hand back the stored
    /// `(key, value)`. Slices are borrowed for the duration of `visit`; return
    /// `false` to stop early. Reuses the writer's `prefixed_bound` so range bounds
    /// are encoded identically.
    pub fn scan_with(
        &self,
        ns: Namespace<'_>,
        range: KeyRange,
        visit: impl FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        self.scan_with_at(None, ns, range, visit)
    }

    /// [`Self::scan_with`] served from `snap` when present (snapshot-pinned
    /// read txn), else from the reader's live latest-checkpoint state.
    pub(crate) fn scan_with_at(
        &self,
        snap: Option<&Arc<DbSnapshot>>,
        ns: Namespace<'_>,
        range: KeyRange,
        visit: impl FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        self.scan_with_lsm_options(snap, ns, range, lsm_scan_options(), visit)
    }

    /// [`Self::scan_dup_with`] served from `snap` when present.
    pub(crate) fn scan_dup_with_at(
        &self,
        snap: Option<&Arc<DbSnapshot>>,
        ns: Namespace<'_>,
        range: KeyRange,
        visit: impl FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        self.scan_dup_with_options(snap, ns, range, lsm_scan_options(), visit)
    }
}

/// `DbReader` has no internal `Drop` shutdown: dropping the last `Arc`
/// leaks its manifest-poll/checkpoint tasks, which keep CAS-writing
/// checkpoint state to the object store. A leaked task belonging to an
/// evicted reader resurrected a dropped collection's purged S3 prefix in
/// production, so the wrapper must close the reader explicitly. The close
/// is detached and delayed (mirroring the refresh swap path) because `Drop`
/// cannot block and in-flight reads may still hold the inner `Arc`.
impl Drop for LsmReader {
    fn drop(&mut self) {
        let reader = {
            let slot = self
                .reader
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            Arc::clone(slot)
        };
        if reader.status().close_reason.is_some() {
            return;
        }
        let path = self.path.clone();
        self.rt.spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            if let Err(error) = reader.close().await {
                tracing::debug!(%error, path, "closing DbReader on LsmReader drop failed");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helix_engine::storage_core::backend::{Namespace, StorageBackend};
    use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
    use object_store::aws::AmazonS3Builder;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// End-to-end writer/reader split on real S3 (MinIO at :9000): a separate
    /// `LsmReader` instance, opened against its OWN object_store, must read back
    /// the value a separate `LsmBackend` writer committed to the same db path.
    /// Proves SlateDB's `DbReader` sees the writer's committed data off S3.
    ///
    /// Needs MinIO up with the `helion-test` bucket; run with
    /// `cargo test -p helixdb lsm_reader_sees_writer_on_minio_s3 -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn lsm_reader_sees_writer_on_minio_s3() {
        // Two independent store instances — the writer and the reader each get
        // their own client, mirroring separate processes.
        let mk_store = || {
            AmazonS3Builder::new()
                .with_endpoint("http://localhost:9000")
                .with_region("us-east-1")
                .with_bucket_name("helion-test")
                .with_access_key_id("test")
                .with_secret_access_key("testtest123")
                .with_allow_http(true)
                .build()
                .expect("build MinIO S3 store")
        };
        let writer_store = Arc::new(mk_store());
        let reader_store = Arc::new(mk_store());

        // Unique db path per run so retained bucket state can't mask a bug.
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = format!("reader-{nanos}");

        // Writer: open, put under Nodes, commit. Commit awaits durability
        // (await_durable=true) with WAL enabled, so the value is on S3 once
        // commit returns.
        let writer = LsmBackend::open_with_store(&path, writer_store)
            .expect("open SlateDB writer on MinIO S3");
        let mut w = writer.begin_write().unwrap();
        writer
            .put(&mut w, Namespace::Nodes, b"rk1", b"rv1")
            .unwrap();
        writer.commit(w).unwrap();

        // Reader: a fresh DbReader on its own store, opened AFTER the durable
        // commit, replays the committed WAL and must see the value.
        let reader = LsmReader::open_with_store(&path, reader_store)
            .expect("open SlateDB reader on MinIO S3");
        let got = reader
            .get_with(Namespace::Nodes, b"rk1", |v| v.map(|b| b.to_vec()))
            .unwrap();
        assert_eq!(
            got,
            Some(b"rv1".to_vec()),
            "LsmReader must see the writer's committed value off S3"
        );

        // A key the writer never wrote must read back as None.
        let missing = reader
            .get_with(Namespace::Nodes, b"nope", |v| v.map(|b| b.to_vec()))
            .unwrap();
        assert_eq!(missing, None);
    }

    /// Dup-namespace round-trip across the writer/reader split: the writer commits
    /// several `(key, value)` pairs into a DUP namespace (`OutEdges`) and a
    /// separate `LsmReader` must read them back identically through the shared dup
    /// decoders — `for_each_dup_with` (values under one key), `scan_dup_with`
    /// (whole-namespace), and `contains_dup` (point membership). Proves the
    /// replica reproduces the writer's composite dup rows byte-for-byte off S3.
    ///
    /// Needs MinIO up with the `helion-test` bucket; run with
    /// `cargo test -p helixdb lsm_reader_sees_writer_dup_on_minio_s3 -- --ignored --nocapture`.
    /// A reader opened on a prefix no writer ever created (collection missing
    /// in object storage) must carry the typed `DatabaseMissing` marker so the
    /// collection manager maps it to not-found (404) instead of a 500.
    #[test]
    fn lsm_reader_open_missing_database_is_tagged() {
        use object_store::memory::InMemory;
        use object_store::ObjectStore;

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let err = match LsmReader::open_with_store("never-created", store) {
            Ok(_) => panic!("reader open on an empty prefix must fail"),
            Err(err) => err,
        };
        assert!(
            matches!(&err, BackendError::Io(msg) if msg.starts_with(LSM_READER_DATABASE_MISSING)),
            "missing database must be tagged, got: {err}"
        );
    }

    /// HA / disposable-reader proof: object storage is the durable record; a
    /// reader's local state (DbReader snapshot + cache) is throwaway. A writer
    /// commits and is dropped; reader A reads then is dropped (simulating a reader
    /// node dying with full cache loss); a brand-new reader B with NO local state,
    /// opened on the same object storage, must reconstruct and serve identical
    /// results. Non-ignored: uses a shared in-memory object store as the durable
    /// system of record.
    #[test]
    fn lsm_reader_recovers_from_object_storage_after_cache_loss() {
        use object_store::memory::InMemory;
        use object_store::ObjectStore;
        use std::sync::Arc;

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = "reader-recovery-test";

        // Writer commits a plain node + a dup edge, then is dropped entirely.
        {
            let writer = LsmBackend::open_with_store(path, store.clone()).expect("open writer");
            let mut w = writer.begin_write().unwrap();
            writer.put(&mut w, Namespace::Nodes, b"k1", b"v1").unwrap();
            writer
                .put_dup(&mut w, Namespace::OutEdges, b"n1", b"a")
                .unwrap();
            writer.commit(w).unwrap();
        }

        // Reader A reads the committed data, then is dropped — its snapshot/cache
        // (all local state) is lost, as if the reader node died.
        {
            let reader_a = LsmReader::open_with_store(path, store.clone()).expect("open reader A");
            assert_eq!(
                reader_a
                    .get_with(Namespace::Nodes, b"k1", |v| v.map(|b| b.to_vec()))
                    .unwrap(),
                Some(b"v1".to_vec())
            );
        }

        // Reader B: a fresh replica with no local state, opened on the same object
        // storage. Must serve identical results purely from object storage.
        let reader_b =
            LsmReader::open_with_store(path, store.clone()).expect("open fresh reader B");
        assert_eq!(
            reader_b
                .get_with(Namespace::Nodes, b"k1", |v| v.map(|b| b.to_vec()))
                .unwrap(),
            Some(b"v1".to_vec()),
            "fresh reader must read the committed node off object storage after cache loss"
        );
        let batch_keys = vec![b"k1".to_vec(), b"missing".to_vec()];
        let batch = reader_b
            .collect_values_many_with(Namespace::Nodes, &batch_keys)
            .unwrap();
        assert_eq!(
            batch,
            vec![Some(b"v1".to_vec()), None],
            "batch reader helper must preserve point-read values and misses"
        );
        let mut edges = Vec::new();
        reader_b
            .for_each_dup_with(Namespace::OutEdges, b"n1", |v| {
                edges.push(v.to_vec());
                true
            })
            .unwrap();
        assert_eq!(
            edges,
            vec![b"a".to_vec()],
            "fresh reader must read committed dup rows off object storage after cache loss"
        );
    }

    /// Change-feed promotion path: `refresh_with_poll_interval` reopens the
    /// inner `DbReader` against the CURRENT manifest, so a reader whose own
    /// poll interval is effectively never (1h) still observes a commit that
    /// landed after it opened. In-memory object store; no MinIO needed.
    #[test]
    #[serial_test::serial]
    fn lsm_reader_refresh_sees_later_commit_without_polling() {
        use object_store::memory::InMemory;

        struct VarGuard(&'static str);
        impl VarGuard {
            fn set(key: &'static str, value: &str) -> Self {
                std::env::set_var(key, value);
                Self(key)
            }
        }
        impl Drop for VarGuard {
            fn drop(&mut self) {
                std::env::remove_var(self.0);
            }
        }
        // 60s poll: far beyond this test's runtime, so the DbReader never
        // self-refreshes and any visibility change is attributable to
        // refresh_with_poll_interval. Kept modest (not hours) because env vars
        // leak to concurrently-running tests that also open readers.
        let _poll = VarGuard::set("HELIX_LSM_READER_MANIFEST_POLL_MS", "60000");

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = "reader-refresh-test";

        let writer = LsmBackend::open_with_store(path, store.clone()).expect("open writer");
        let mut w = writer.begin_write().unwrap();
        writer.put(&mut w, Namespace::Nodes, b"k1", b"v1").unwrap();
        writer.commit(w).unwrap();

        let reader = LsmReader::open_with_store(path, store.clone()).expect("open reader");
        assert_eq!(
            reader
                .get_with(Namespace::Nodes, b"k1", |v| v.map(|b| b.to_vec()))
                .unwrap(),
            Some(b"v1".to_vec())
        );

        // Commit AFTER the reader opened: invisible until a refresh.
        let mut w = writer.begin_write().unwrap();
        writer.put(&mut w, Namespace::Nodes, b"k2", b"v2").unwrap();
        writer.commit(w).unwrap();
        assert_eq!(
            reader
                .get_with(Namespace::Nodes, b"k2", |v| v.map(|b| b.to_vec()))
                .unwrap(),
            None,
            "1h-poll reader must not see the later commit before refresh"
        );

        reader
            .refresh_with_poll_interval(std::time::Duration::from_secs(3600))
            .expect("refresh reader");
        assert_eq!(
            reader
                .get_with(Namespace::Nodes, b"k2", |v| v.map(|b| b.to_vec()))
                .unwrap(),
            Some(b"v2".to_vec()),
            "refreshed reader must see the commit that landed after open"
        );
        // Pre-existing keys still read through the swapped-in reader.
        assert_eq!(
            reader
                .get_with(Namespace::Nodes, b"k1", |v| v.map(|b| b.to_vec()))
                .unwrap(),
            Some(b"v1".to_vec())
        );
    }

    /// The bounded promotion primitive (`refresh_with_poll_interval_bounded`,
    /// used by the reader-replica read-gated poll-tier promotion) must behave
    /// exactly like the unbounded `refresh_with_poll_interval` when the
    /// budget isn't exceeded: it sees a commit that landed after the reader
    /// opened. Mirrors `lsm_reader_refresh_sees_later_commit_without_polling`.
    #[test]
    #[serial_test::serial]
    fn lsm_reader_bounded_refresh_sees_later_commit_within_budget() {
        use object_store::memory::InMemory;

        struct VarGuard(&'static str);
        impl VarGuard {
            fn set(key: &'static str, value: &str) -> Self {
                std::env::set_var(key, value);
                Self(key)
            }
        }
        impl Drop for VarGuard {
            fn drop(&mut self) {
                std::env::remove_var(self.0);
            }
        }
        let _poll = VarGuard::set("HELIX_LSM_READER_MANIFEST_POLL_MS", "60000");

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = "reader-bounded-refresh-test";

        let writer = LsmBackend::open_with_store(path, store.clone()).expect("open writer");
        let mut w = writer.begin_write().unwrap();
        writer.put(&mut w, Namespace::Nodes, b"k1", b"v1").unwrap();
        writer.commit(w).unwrap();

        let reader = LsmReader::open_with_store(path, store.clone()).expect("open reader");

        // Commit AFTER the reader opened: invisible until a refresh.
        let mut w = writer.begin_write().unwrap();
        writer.put(&mut w, Namespace::Nodes, b"k2", b"v2").unwrap();
        writer.commit(w).unwrap();
        assert_eq!(
            reader
                .get_with(Namespace::Nodes, b"k2", |v| v.map(|b| b.to_vec()))
                .unwrap(),
            None,
            "1h-poll reader must not see the later commit before a bounded refresh"
        );

        // Generous budget: an InMemory-store open is effectively instant, so
        // this proves wiring correctness, not the timeout path itself (an
        // InMemory store cannot be made to miss a real deadline).
        reader
            .refresh_with_poll_interval_bounded(
                std::time::Duration::from_secs(3600),
                std::time::Duration::from_secs(2),
            )
            .expect("bounded refresh within budget must succeed");
        assert_eq!(
            reader
                .get_with(Namespace::Nodes, b"k2", |v| v.map(|b| b.to_vec()))
                .unwrap(),
            Some(b"v2".to_vec()),
            "bounded refresh must see the commit that landed after open, same as unbounded"
        );
        assert_eq!(
            reader
                .get_with(Namespace::Nodes, b"k1", |v| v.map(|b| b.to_vec()))
                .unwrap(),
            Some(b"v1".to_vec())
        );
    }

    /// End-to-end shape of the poll-tiering review's HIGH #2 fix: a
    /// reader-replica cold-opened while the write-feed is configured takes
    /// the IDLE poll tier (`reader_options_from_env`), so a commit that lands
    /// after open is invisible until something promotes it — exactly the
    /// situation `HelixGraphStorage::note_reader_read_and_maybe_promote`
    /// handles on the first read, now that its `reader_poll_tier_fast` flag
    /// correctly starts `false` in this mode (`reader_cold_open_starts_fast`)
    /// instead of always assuming fast. Simulates that first-read promotion
    /// directly against the `LsmReader` primitive (bounded refresh to a fast
    /// interval) and asserts the later commit becomes visible — same shape as
    /// `lsm_reader_bounded_refresh_sees_later_commit_within_budget`, but
    /// starting from a feed-mode cold open instead of an explicit long poll.
    #[test]
    #[serial_test::serial]
    fn lsm_reader_feed_mode_cold_open_promotes_to_see_commit_on_first_read() {
        use object_store::memory::InMemory;

        struct VarGuard(&'static str);
        impl VarGuard {
            fn set(key: &'static str, value: &str) -> Self {
                std::env::set_var(key, value);
                Self(key)
            }
        }
        impl Drop for VarGuard {
            fn drop(&mut self) {
                std::env::remove_var(self.0);
            }
        }
        // Cold opens under the write-feed take the idle tier automatically
        // (see `reader_options_from_env`) — no explicit poll-interval knob
        // needed to set up the "idle-tier cold open" precondition.
        let _feed = VarGuard::set("HELIX_LSM_WRITER_FEED_URL", "http://helix:6969");

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = "reader-feed-mode-cold-open-test";

        let writer = LsmBackend::open_with_store(path, store.clone()).expect("open writer");
        let mut w = writer.begin_write().unwrap();
        writer.put(&mut w, Namespace::Nodes, b"k1", b"v1").unwrap();
        writer.commit(w).unwrap();

        // Cold open while the feed URL is set: takes the idle poll interval.
        let reader = LsmReader::open_with_store(path, store.clone()).expect("open reader");

        // Commit AFTER the reader opened, on the idle tier: invisible until a
        // read-driven promotion refreshes the checkpoint.
        let mut w = writer.begin_write().unwrap();
        writer.put(&mut w, Namespace::Nodes, b"k2", b"v2").unwrap();
        writer.commit(w).unwrap();
        assert_eq!(
            reader
                .get_with(Namespace::Nodes, b"k2", |v| v.map(|b| b.to_vec()))
                .unwrap(),
            None,
            "idle-tier feed-mode cold open must not see the later commit before promotion"
        );

        // Simulate the first-read promotion
        // `note_reader_read_and_maybe_promote` performs: a bounded refresh to
        // the fast/serving interval.
        reader
            .refresh_with_poll_interval_bounded(
                std::time::Duration::from_secs(5),
                std::time::Duration::from_secs(2),
            )
            .expect("first-read promotion must succeed");
        assert_eq!(
            reader
                .get_with(Namespace::Nodes, b"k2", |v| v.map(|b| b.to_vec()))
                .unwrap(),
            Some(b"v2".to_vec()),
            "promotion on first read must make the later commit visible"
        );
    }

    /// The read-gated promotion path (`HelixGraphStorage::note_reader_read_and_maybe_promote`)
    /// relies on `HelixGraphStorage`'s own CAS to keep only one promotion in
    /// flight per collection, but the underlying swap primitive itself must
    /// not corrupt reader state or panic if it were ever invoked
    /// concurrently. Several threads calling the bounded refresh at once on
    /// the same `LsmReader` must all return `Ok` and leave the reader serving
    /// consistent, correct data.
    #[test]
    #[serial_test::serial]
    fn lsm_reader_bounded_refresh_is_safe_under_concurrent_calls() {
        use object_store::memory::InMemory;
        use std::sync::Barrier;

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = "reader-bounded-refresh-concurrency-test";

        let writer = LsmBackend::open_with_store(path, store.clone()).expect("open writer");
        let mut w = writer.begin_write().unwrap();
        writer.put(&mut w, Namespace::Nodes, b"k1", b"v1").unwrap();
        writer.commit(w).unwrap();

        let reader =
            Arc::new(LsmReader::open_with_store(path, store.clone()).expect("open reader"));

        let threads = 8;
        let barrier = Arc::new(Barrier::new(threads));
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let reader = Arc::clone(&reader);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    reader.refresh_with_poll_interval_bounded(
                        std::time::Duration::from_secs(5),
                        std::time::Duration::from_secs(2),
                    )
                })
            })
            .collect();

        for handle in handles {
            handle
                .join()
                .expect("refresh thread must not panic")
                .expect("concurrent bounded refresh must not error");
        }

        // The reader must still serve correct data after the concurrent swaps.
        assert_eq!(
            reader
                .get_with(Namespace::Nodes, b"k1", |v| v.map(|b| b.to_vec()))
                .unwrap(),
            Some(b"v1".to_vec()),
            "reader must serve correct data after concurrent poll-tier swaps"
        );
    }

    #[test]
    #[ignore]
    fn lsm_reader_sees_writer_dup_on_minio_s3() {
        use crate::helix_engine::storage_core::backend::KeyRange;

        let mk_store = || {
            AmazonS3Builder::new()
                .with_endpoint("http://localhost:9000")
                .with_region("us-east-1")
                .with_bucket_name("helion-test")
                .with_access_key_id("test")
                .with_secret_access_key("testtest123")
                .with_allow_http(true)
                .build()
                .expect("build MinIO S3 store")
        };
        let writer_store = Arc::new(mk_store());
        let reader_store = Arc::new(mk_store());

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = format!("reader-dup-{nanos}");

        // Writer: three values under "n1", one under "n2", in a DUP namespace.
        // Values chosen so composite-key (value-appended) order is a, b, c.
        let writer = LsmBackend::open_with_store(&path, writer_store)
            .expect("open SlateDB writer on MinIO S3");
        let mut w = writer.begin_write().unwrap();
        writer
            .put_dup(&mut w, Namespace::OutEdges, b"n1", b"a")
            .unwrap();
        writer
            .put_dup(&mut w, Namespace::OutEdges, b"n1", b"b")
            .unwrap();
        writer
            .put_dup(&mut w, Namespace::OutEdges, b"n1", b"c")
            .unwrap();
        writer
            .put_dup(&mut w, Namespace::OutEdges, b"n2", b"d")
            .unwrap();
        writer.commit(w).unwrap();

        let reader = LsmReader::open_with_store(&path, reader_store)
            .expect("open SlateDB reader on MinIO S3");

        // Values under one key, in key order.
        let mut n1_vals: Vec<Vec<u8>> = Vec::new();
        reader
            .for_each_dup_with(Namespace::OutEdges, b"n1", |v| {
                n1_vals.push(v.to_vec());
                true
            })
            .unwrap();
        assert_eq!(
            n1_vals,
            vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()],
            "for_each_dup_with must return every value the writer put under n1"
        );

        // Whole-namespace scan returns each (key, value) pair the writer wrote.
        let mut all: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        reader
            .scan_dup_with(Namespace::OutEdges, KeyRange::all(), |k, v| {
                all.push((k.to_vec(), v.to_vec()));
                true
            })
            .unwrap();
        assert_eq!(
            all,
            vec![
                (b"n1".to_vec(), b"a".to_vec()),
                (b"n1".to_vec(), b"b".to_vec()),
                (b"n1".to_vec(), b"c".to_vec()),
                (b"n2".to_vec(), b"d".to_vec()),
            ],
            "scan_dup_with must reproduce the writer's dup rows in key order"
        );

        // Point membership.
        assert!(reader
            .contains_dup(Namespace::OutEdges, b"n1", b"b")
            .unwrap());
        assert!(!reader
            .contains_dup(Namespace::OutEdges, b"n1", b"zzz")
            .unwrap());
    }
}
