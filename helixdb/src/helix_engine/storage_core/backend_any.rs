//! `AnyBackend` — runtime enum dispatch over [`StorageBackend`] implementations.
//!
//! The [`StorageBackend`] trait is GAT-based and therefore not object-safe, so
//! runtime backend selection cannot use `dyn StorageBackend`. `AnyBackend` is
//! the concrete type the migrated call sites hold: it dispatches every operation
//! to the active engine — LMDB (`heed3`) or the SlateDB-backed LSM-on-object-
//! storage engine, chosen via [`BackendKind`].

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(test)]
use std::sync::Mutex;

use heed3::{Env, RoTxn, RwTxn, WithTls};
#[cfg(test)]
use slatedb::object_store::ObjectStore;
use slatedb::{bytes::Bytes, CloseReason};

use super::backend::{
    BackendError, BackendKind, KeyRange, Namespace, ReadTxn, StorageBackend, StorageBackendConfig,
    WriteTxn,
};
use super::backend_lmdb::{LmdbBackend, LmdbRead, LmdbWrite};
use super::backend_lsm::{
    cache_root_from_env, cache_subtree_for_lsm_path, clear_s3_drop_tombstone,
    list_s3_collection_prefixes, persist_s3_drop_tombstone, read_s3_drop_tombstone,
    read_s3_metadata_sidecar, s3_collection_prefix_exists, LsmBackend, LsmRead, LsmWrite,
    S3StoreConfig,
};
#[cfg(test)]
use super::backend_lsm::{list_child_collection_names, purge_prefix_from_store, shared_lsm_handle};
use super::backend_lsm_reader::LsmReader;

fn unhealthy_lsm_writer_close_reason_from(reason: Option<CloseReason>) -> Option<CloseReason> {
    match reason {
        reason @ Some(CloseReason::Fenced | CloseReason::Panic) => reason,
        Some(CloseReason::Clean) | None => None,
        Some(_) => None,
    }
}

/// Error returned by every write method on the LSM reader-replica arm. Reader
/// nodes serve reads off `DbReader` and must never mutate, so writes are rejected
/// at the backend layer (defense-in-depth, independent of gateway routing).
pub(crate) const LSM_READER_READONLY: &str =
    "read-only LSM reader replica; writes must go to the writer node";

#[cfg(test)]
static LSM_TEST_OBJECT_STORE: std::sync::LazyLock<Mutex<Option<Arc<dyn ObjectStore>>>> =
    std::sync::LazyLock::new(|| Mutex::new(None));

#[cfg(test)]
pub(crate) struct LsmTestObjectStoreGuard {
    previous: Option<Arc<dyn ObjectStore>>,
}

#[cfg(test)]
impl Drop for LsmTestObjectStoreGuard {
    fn drop(&mut self) {
        if let Ok(mut slot) = LSM_TEST_OBJECT_STORE.lock() {
            *slot = self.previous.take();
        }
    }
}

#[cfg(test)]
pub(crate) fn set_lsm_test_object_store(store: Arc<dyn ObjectStore>) -> LsmTestObjectStoreGuard {
    let previous = LSM_TEST_OBJECT_STORE
        .lock()
        .map(|mut slot| slot.replace(store))
        .unwrap_or(None);
    LsmTestObjectStoreGuard { previous }
}

#[cfg(test)]
fn lsm_test_object_store() -> Option<Arc<dyn ObjectStore>> {
    LSM_TEST_OBJECT_STORE
        .lock()
        .ok()
        .and_then(|slot| slot.as_ref().map(Arc::clone))
}

/// The active storage engine.
pub enum AnyBackend {
    Lmdb(LmdbBackend),
    Lsm(LsmBackend),
    /// Read-only replica over SlateDB's `DbReader` (`HELIX_LSM_ROLE=reader`):
    /// serves committed reads off object storage + the SSD cache; writes rejected.
    ///
    /// Topology & consistency contract. Each collection has exactly ONE writer
    /// ([`AnyBackend::Lsm`]) and any number of horizontally-scalable readers over
    /// the SAME object-store prefix (`HELIX_LSM_PREFIX/<collection>`). A reader is
    /// *eventually fresh*, NOT linearizable: it resolves a self-refreshing
    /// checkpoint of the writer's committed state and picks up later commits on the
    /// next manifest refresh (see [`LsmReader`]), so a read may briefly lag the
    /// writer. Reads routed to the writer node see committed writes immediately.
    /// Local SSD/in-memory caches affect latency only, never correctness.
    LsmReader(LsmReader),
}

/// Dispatched read snapshot.
pub enum AnyRead<'s> {
    Lmdb(LmdbRead<'s>),
    Lsm(LsmRead),
    /// Read handle for the reader replica. `None`: reads are served from the
    /// `DbReader`'s live latest-checkpoint state (each read may observe a
    /// newer manifest refresh than the last — the legacy behavior). `Some`: a
    /// point-in-time `DbSnapshot` pinned at `begin_read` so every read in this
    /// logical txn observes ONE committed sequence (vendored SlateDB fork;
    /// enabled via `HELIX_LSM_READER_SNAPSHOT_READS`).
    LsmReader(Option<Arc<slatedb::DbSnapshot>>),
    /// A read handle whose underlying snapshot failed to open (e.g. the writer
    /// was fenced by a newer SlateDB client). The error is carried so the first
    /// read returns it cleanly rather than panicking at handle-construction time
    /// (see [`AnyBackend::read_borrowed`], which returns `AnyRead` by value and so
    /// cannot surface an `Err` directly).
    Failed(BackendError),
}
impl ReadTxn for AnyRead<'_> {}

/// Dispatched write batch.
pub enum AnyWrite<'s> {
    Lmdb(LmdbWrite<'s>),
    Lsm(LsmWrite),
}
impl WriteTxn for AnyWrite<'_> {}

impl AnyBackend {
    /// Open the LMDB backend at `path`.
    pub fn open_lmdb(path: &Path, max_dbs: u32, map_size: usize) -> Result<Self, BackendError> {
        Ok(AnyBackend::Lmdb(LmdbBackend::open(
            path, max_dbs, map_size,
        )?))
    }

    /// Wrap an already-open LMDB env.
    pub fn from_lmdb_env(env: Env<WithTls>) -> Self {
        AnyBackend::Lmdb(LmdbBackend::from_env(env))
    }

    /// Open the SlateDB backend on an in-memory object store (tests/dev).
    pub fn open_lsm_in_memory(path: &str) -> Result<Self, BackendError> {
        Ok(AnyBackend::Lsm(LsmBackend::open_in_memory(path)?))
    }

    /// Open the SlateDB backend selected for one collection from environment.
    ///
    /// Required for `HELIX_STORAGE_BACKEND=lsm`:
    /// - `HELIX_LSM_BUCKET`
    ///
    /// Optional:
    /// - `HELIX_LSM_PREFIX` (default `helion`)
    /// - `HELIX_LSM_REGION`
    /// - `HELIX_LSM_ENDPOINT` + `HELIX_LSM_ALLOW_HTTP=1` for MinIO/S3-compatible stores
    pub fn open_lsm_from_env(
        collection_path: &Path,
        config: StorageBackendConfig,
    ) -> Result<Self, BackendError> {
        if config.is_lsm_in_memory() {
            return Self::open_lsm_in_memory(&collection_lsm_path(collection_path));
        }

        let prefix = collection_lsm_path(collection_path);

        #[cfg(test)]
        if let Some(store) = lsm_test_object_store() {
            if config.is_reader() {
                return Err(BackendError::Unsupported(
                    "test object-store override does not support LSM reader replicas".to_string(),
                ));
            }
            return Ok(AnyBackend::Lsm(LsmBackend::open_with_store(
                &prefix, store,
            )?));
        }

        let bucket = std::env::var("HELIX_LSM_BUCKET").map_err(|_| {
            BackendError::Io(
                "HELIX_STORAGE_BACKEND=lsm requires HELIX_LSM_BUCKET (or HELIX_LSM_IN_MEMORY=1 for tests/dev)"
                    .to_string(),
            )
        })?;
        let region = std::env::var("HELIX_LSM_REGION").ok();
        let endpoint = std::env::var("HELIX_LSM_ENDPOINT").ok();
        let allow_http = env_flag("HELIX_LSM_ALLOW_HTTP");

        // `HELIX_LSM_ROLE=reader` opens a read-only replica over the writer's
        // committed state at the same bucket/prefix (self-refreshing `DbReader` +
        // SSD cache); the default role opens the read-write LSM engine. Reads are
        // byte-identical because both arms share the namespace key encoding.
        if config.is_reader() {
            return Ok(AnyBackend::LsmReader(LsmReader::open_s3_with_options(
                &bucket,
                &prefix,
                region.as_deref(),
                endpoint.as_deref(),
                allow_http,
            )?));
        }

        Ok(AnyBackend::Lsm(LsmBackend::open_s3_with_options(
            &bucket,
            &prefix,
            region.as_deref(),
            endpoint.as_deref(),
            allow_http,
        )?))
    }

    pub fn open_selected_for_collection(
        config: StorageBackendConfig,
        lmdb_env: Option<Env<WithTls>>,
        collection_path: &Path,
    ) -> Result<Self, BackendError> {
        match config {
            StorageBackendConfig::Lmdb => lmdb_env
                .map(Self::from_lmdb_env)
                .ok_or_else(|| BackendError::Io("LMDB backend requires a heed env".to_string())),
            StorageBackendConfig::Lsm { .. } => Self::open_lsm_from_env(collection_path, config),
        }
    }

    /// The kind of the active backend.
    pub fn kind(&self) -> BackendKind {
        match self {
            AnyBackend::Lmdb(_) => BackendKind::Lmdb,
            AnyBackend::Lsm(_) => BackendKind::Lsm,
            // A reader replica IS the LSM engine, read-only — no separate kind so
            // existing `kind()==Lsm` branches treat it as LSM (writes are gated by
            // the read-only handlers, not by kind).
            AnyBackend::LsmReader(_) => BackendKind::Lsm,
        }
    }

    /// True for an eventually-consistent reader replica (`HELIX_LSM_ROLE=reader`)
    /// over a `DbReader` checkpoint. Such a node never commits locally, so any
    /// in-memory state cached at open time (e.g. the metadata snapshot / point
    /// count) must be re-read from committed object-store state to stay current,
    /// whereas the writer refreshes that state on every commit.
    pub fn is_reader_replica(&self) -> bool {
        matches!(self, AnyBackend::LsmReader(_))
    }

    pub(crate) fn unhealthy_lsm_writer_close_reason(&self) -> Option<CloseReason> {
        match self {
            AnyBackend::Lsm(backend) => {
                unhealthy_lsm_writer_close_reason_from(backend.close_reason())
            }
            AnyBackend::LsmReader(reader) => {
                unhealthy_lsm_writer_close_reason_from(reader.close_reason())
            }
            AnyBackend::Lmdb(_) => None,
        }
    }

    /// Reopen a reader replica's `DbReader` against the current manifest with
    /// the given poll interval (see [`LsmReader::refresh_with_poll_interval`]).
    /// Returns `Ok(false)` on non-reader arms so the change-feed poller can
    /// skip them without downcasting.
    pub fn refresh_lsm_reader_with_poll_interval(
        &self,
        interval: std::time::Duration,
    ) -> Result<bool, super::backend::BackendError> {
        match self {
            AnyBackend::LsmReader(reader) => {
                reader.refresh_with_poll_interval(interval).map(|()| true)
            }
            AnyBackend::Lmdb(_) | AnyBackend::Lsm(_) => Ok(false),
        }
    }

    /// Compatibility wrapper used by read-gated poll-tier promotion. The
    /// `budget` is forwarded, but SlateDB reader build is side-effecting and is
    /// therefore allowed to finish instead of being dropped on timeout (see
    /// [`LsmReader::refresh_with_poll_interval_bounded`]).
    pub fn refresh_lsm_reader_with_poll_interval_bounded(
        &self,
        interval: std::time::Duration,
        budget: std::time::Duration,
    ) -> Result<bool, super::backend::BackendError> {
        match self {
            AnyBackend::LsmReader(reader) => reader
                .refresh_with_poll_interval_bounded(interval, budget)
                .map(|()| true),
            AnyBackend::Lmdb(_) | AnyBackend::Lsm(_) => Ok(false),
        }
    }

    pub(crate) fn update_lsm_key_transactional<T>(
        &self,
        ns: Namespace<'_>,
        key: &[u8],
        update: impl FnMut(Option<&[u8]>) -> Result<(Vec<u8>, T), BackendError>,
    ) -> Result<T, BackendError> {
        match self {
            AnyBackend::Lsm(backend) => backend.update_key_transactional(ns, key, update),
            AnyBackend::Lmdb(_) => Err(BackendError::Unsupported(
                "LMDB metadata updates use heed write transactions".to_string(),
            )),
            AnyBackend::LsmReader(_) => {
                Err(BackendError::Unsupported(LSM_READER_READONLY.to_string()))
            }
        }
    }

    /// Destroy persistent object-store state for a collection being dropped.
    ///
    /// On the LSM writer arm this quiesces SlateDB (stopping the background
    /// compactor/GC) and then deletes every object under the collection's
    /// S3 prefix. LMDB and reader-replica arms are no-ops: LMDB relies on
    /// the caller's `fs::remove_dir_all` for cleanup, and reader replicas
    /// never own the prefix.
    pub(crate) fn destroy_lsm(&self) -> Result<(), super::backend::BackendError> {
        match self {
            AnyBackend::Lsm(b) => b.destroy(),
            AnyBackend::Lmdb(_) | AnyBackend::LsmReader(_) => Ok(()),
        }
    }

    /// Quiesce an LSM writer without deleting its object-store prefix.
    ///
    /// Cache eviction and ordinary `HelixGraphStorage` drops must stop SlateDB's
    /// background writer/compactor before a later cold-open creates another
    /// writer for the same prefix. LMDB and reader replicas do not own a writer
    /// epoch, so they are no-ops here.
    pub(crate) fn close_lsm_for_cache_eviction(&self) -> Result<(), BackendError> {
        match self {
            AnyBackend::Lsm(b) => match b.close() {
                Err(BackendError::Io(message)) if message.contains("db is closed") => Ok(()),
                result => result,
            },
            AnyBackend::Lmdb(_) | AnyBackend::LsmReader(_) => Ok(()),
        }
    }

    /// List the collection names present in the object store under the
    /// collections root prefix. LMDB has no object-store catalog (collections are
    /// local directories), so it returns empty; the LSM writer and reader-replica
    /// arms list the shared bucket via their retained store handle. Used to
    /// rediscover collections that exist in S3 but not in a freshly-started
    /// node's local registry.
    pub(crate) fn list_collection_prefixes(&self) -> Result<Vec<String>, BackendError> {
        match self {
            AnyBackend::Lmdb(_) => Ok(Vec::new()),
            AnyBackend::Lsm(b) => b.list_collection_prefixes(),
            AnyBackend::LsmReader(b) => b.list_collection_prefixes(),
        }
    }

    pub(crate) fn lsm_storage_bytes(&self) -> Result<Option<u64>, BackendError> {
        match self {
            AnyBackend::Lmdb(_) => Ok(None),
            AnyBackend::Lsm(b) => b.storage_bytes().map(Some),
            AnyBackend::LsmReader(b) => b.storage_bytes().map(Some),
        }
    }

    pub(crate) fn persist_lsm_metadata_sidecar(&self, bytes: Vec<u8>) -> Result<(), BackendError> {
        match self {
            AnyBackend::Lsm(b) => b.persist_metadata_sidecar(bytes),
            AnyBackend::Lmdb(_) | AnyBackend::LsmReader(_) => Ok(()),
        }
    }
}

/// Best-effort LSM catalog rediscovery: list the collection names that exist in
/// the object store under the configured root prefix, building the store from the
/// SAME env knobs as [`AnyBackend::open_lsm_from_env`] (`HELIX_LSM_BUCKET`,
/// `HELIX_LSM_PREFIX`, region/endpoint/allow-http). No `Db`/reader is opened.
///
/// Returns an empty list for the in-memory backend (`HELIX_LSM_IN_MEMORY=1`):
/// each in-memory collection gets an isolated store, so there is no shared
/// catalog to scan. Callers MUST treat any error as non-fatal (log + fall back to
/// the local registry); a listing failure must never block startup or a request.
pub fn list_collection_prefixes_from_env(
    config: StorageBackendConfig,
) -> Result<Vec<String>, BackendError> {
    if config.is_lsm_in_memory() {
        return Ok(Vec::new());
    }
    let root = std::env::var("HELIX_LSM_PREFIX").unwrap_or_else(|_| "helion".to_string());
    let root = root.trim_matches('/').to_string();
    #[cfg(test)]
    if let Some(store) = lsm_test_object_store() {
        return list_child_collection_names(&shared_lsm_handle(), &store, &root);
    }
    let bucket = std::env::var("HELIX_LSM_BUCKET").map_err(|_| {
        BackendError::Io(
            "HELIX_STORAGE_BACKEND=lsm catalog rediscovery requires HELIX_LSM_BUCKET".to_string(),
        )
    })?;
    let region = std::env::var("HELIX_LSM_REGION").ok();
    let endpoint = std::env::var("HELIX_LSM_ENDPOINT").ok();
    let allow_http = env_flag("HELIX_LSM_ALLOW_HTTP");
    list_s3_collection_prefixes(
        &bucket,
        &root,
        region.as_deref(),
        endpoint.as_deref(),
        allow_http,
    )
}

pub fn read_metadata_sidecar_from_lsm_env(
    collection_path: &Path,
    config: StorageBackendConfig,
) -> Result<Option<Vec<u8>>, BackendError> {
    if config.is_lsm_in_memory() {
        return Ok(None);
    }
    let prefix = collection_lsm_path(collection_path);
    #[cfg(test)]
    if lsm_test_object_store().is_some() {
        return Ok(None);
    }
    let bucket = std::env::var("HELIX_LSM_BUCKET").map_err(|_| {
        BackendError::Io(
            "HELIX_STORAGE_BACKEND=lsm metadata sidecar lookup requires HELIX_LSM_BUCKET"
                .to_string(),
        )
    })?;
    let region = std::env::var("HELIX_LSM_REGION").ok();
    let endpoint = std::env::var("HELIX_LSM_ENDPOINT").ok();
    let allow_http = env_flag("HELIX_LSM_ALLOW_HTTP");
    let config = S3StoreConfig {
        bucket: &bucket,
        region: region.as_deref(),
        endpoint: endpoint.as_deref(),
        allow_http,
    };
    read_s3_metadata_sidecar(&config, &prefix)
}

pub(crate) fn lsm_collection_prefix_exists_from_env(
    collection_path: &Path,
    config: StorageBackendConfig,
) -> Result<bool, BackendError> {
    if !config.is_lsm() || config.is_lsm_in_memory() || config.is_reader() {
        return Ok(false);
    }
    let prefix = collection_lsm_path(collection_path);
    #[cfg(test)]
    if let Some(store) = lsm_test_object_store() {
        return super::backend_lsm::collection_prefix_exists_with_store(store, &prefix);
    }
    let bucket = std::env::var("HELIX_LSM_BUCKET").map_err(|_| {
        BackendError::Io(
            "HELIX_STORAGE_BACKEND=lsm prefix probe requires HELIX_LSM_BUCKET".to_string(),
        )
    })?;
    let region = std::env::var("HELIX_LSM_REGION").ok();
    let endpoint = std::env::var("HELIX_LSM_ENDPOINT").ok();
    let config = S3StoreConfig {
        bucket: &bucket,
        region: region.as_deref(),
        endpoint: endpoint.as_deref(),
        allow_http: env_flag("HELIX_LSM_ALLOW_HTTP"),
    };
    s3_collection_prefix_exists(&config, &prefix)
}

pub(crate) fn persist_lsm_drop_tombstone_from_env(
    collection_path: &Path,
    config: StorageBackendConfig,
) -> Result<(), BackendError> {
    if !config.is_lsm() || config.is_lsm_in_memory() || config.is_reader() {
        return Ok(());
    }
    let prefix = collection_lsm_path(collection_path);
    #[cfg(test)]
    if let Some(store) = lsm_test_object_store() {
        return super::backend_lsm::persist_drop_tombstone_with_store(store, &prefix);
    }
    let bucket = std::env::var("HELIX_LSM_BUCKET").map_err(|_| {
        BackendError::Io(
            "HELIX_STORAGE_BACKEND=lsm drop tombstone persist requires HELIX_LSM_BUCKET"
                .to_string(),
        )
    })?;
    let region = std::env::var("HELIX_LSM_REGION").ok();
    let endpoint = std::env::var("HELIX_LSM_ENDPOINT").ok();
    let allow_http = env_flag("HELIX_LSM_ALLOW_HTTP");
    let config = S3StoreConfig {
        bucket: &bucket,
        region: region.as_deref(),
        endpoint: endpoint.as_deref(),
        allow_http,
    };
    persist_s3_drop_tombstone(&config, &prefix)
}

pub(crate) fn lsm_drop_tombstone_exists_from_env(
    collection_path: &Path,
    config: StorageBackendConfig,
) -> Result<bool, BackendError> {
    if !config.is_lsm() || config.is_lsm_in_memory() || config.is_reader() {
        return Ok(false);
    }
    let prefix = collection_lsm_path(collection_path);
    #[cfg(test)]
    if let Some(store) = lsm_test_object_store() {
        return super::backend_lsm::read_drop_tombstone_with_store(store, &prefix);
    }
    let bucket = std::env::var("HELIX_LSM_BUCKET").map_err(|_| {
        BackendError::Io(
            "HELIX_STORAGE_BACKEND=lsm drop tombstone lookup requires HELIX_LSM_BUCKET".to_string(),
        )
    })?;
    let region = std::env::var("HELIX_LSM_REGION").ok();
    let endpoint = std::env::var("HELIX_LSM_ENDPOINT").ok();
    let allow_http = env_flag("HELIX_LSM_ALLOW_HTTP");
    let config = S3StoreConfig {
        bucket: &bucket,
        region: region.as_deref(),
        endpoint: endpoint.as_deref(),
        allow_http,
    };
    read_s3_drop_tombstone(&config, &prefix)
}

pub(crate) fn clear_lsm_drop_tombstone_from_env(
    collection_path: &Path,
    config: StorageBackendConfig,
) -> Result<(), BackendError> {
    if !config.is_lsm() || config.is_lsm_in_memory() || config.is_reader() {
        return Ok(());
    }
    let prefix = collection_lsm_path(collection_path);
    #[cfg(test)]
    if let Some(store) = lsm_test_object_store() {
        return super::backend_lsm::clear_drop_tombstone_with_store(store, &prefix);
    }
    let bucket = std::env::var("HELIX_LSM_BUCKET").map_err(|_| {
        BackendError::Io(
            "HELIX_STORAGE_BACKEND=lsm drop tombstone clear requires HELIX_LSM_BUCKET".to_string(),
        )
    })?;
    let region = std::env::var("HELIX_LSM_REGION").ok();
    let endpoint = std::env::var("HELIX_LSM_ENDPOINT").ok();
    let allow_http = env_flag("HELIX_LSM_ALLOW_HTTP");
    let config = S3StoreConfig {
        bucket: &bucket,
        region: region.as_deref(),
        endpoint: endpoint.as_deref(),
        allow_http,
    };
    clear_s3_drop_tombstone(&config, &prefix)
}

pub(crate) fn purge_lsm_prefix_from_env(
    collection_path: &Path,
    config: StorageBackendConfig,
) -> Result<(), BackendError> {
    if !config.is_lsm() || config.is_lsm_in_memory() || config.is_reader() {
        return Ok(());
    }
    let prefix = collection_lsm_path(collection_path);
    #[cfg(test)]
    if std::env::var("HELIX_LSM_TEST_FAIL_PURGE").is_ok() {
        return Err(BackendError::Io(
            "injected test object-store purge failure".to_string(),
        ));
    }
    #[cfg(test)]
    if let Some(store) = lsm_test_object_store() {
        return purge_prefix_from_store(shared_lsm_handle(), store, &prefix);
    }
    let bucket = std::env::var("HELIX_LSM_BUCKET").map_err(|_| {
        BackendError::Io(
            "HELIX_STORAGE_BACKEND=lsm prefix purge requires HELIX_LSM_BUCKET".to_string(),
        )
    })?;
    let region = std::env::var("HELIX_LSM_REGION").ok();
    let endpoint = std::env::var("HELIX_LSM_ENDPOINT").ok();
    let allow_http = env_flag("HELIX_LSM_ALLOW_HTTP");
    LsmBackend::purge_s3_prefix_with_options(
        &bucket,
        &prefix,
        region.as_deref(),
        endpoint.as_deref(),
        allow_http,
    )
}

impl<'s> AnyWrite<'s> {
    /// Wrap a heed write txn as the LMDB write handle (resize-safe write-view
    /// entry). The traversal holds `&mut AnyWrite`, so the whole write query
    /// shares ONE batch committed at the end.
    pub(crate) fn from_lmdb_txn(txn: RwTxn<'s>) -> Self {
        AnyWrite::Lmdb(LmdbWrite::from_txn(txn))
    }

    /// Mutable access to the underlying heed write txn when this is the LMDB
    /// arm; `None` on LSM.
    ///
    /// Bridge for write paths still typed on `&mut RwTxn` (vector `insert`,
    /// bulk APPEND fast path) during the field flip: on LMDB they write into
    /// the SAME txn the seam writes (`AnyWrite::Lmdb` owns it), so a query's
    /// graph + vector writes commit atomically together. The LSM write path for
    /// those ops is wired separately (Unit 4); until then callers surface a
    /// clear error on the `None` arm.
    pub(crate) fn lmdb_rw_mut(&mut self) -> Option<&mut RwTxn<'s>> {
        match self {
            AnyWrite::Lmdb(w) => Some(w.txn_mut()),
            AnyWrite::Lsm(_) => None,
        }
    }

    /// Shared read of the underlying heed write txn (read-your-writes) when this
    /// is the LMDB arm; `None` on LSM. Bridge for read-within-write paths still
    /// typed on `&RoTxn` (vector existence check during the field flip).
    pub(crate) fn lmdb_ro(&self) -> Option<&RwTxn<'s>> {
        match self {
            AnyWrite::Lmdb(w) => Some(w.txn_ref()),
            AnyWrite::Lsm(_) => None,
        }
    }

    /// The LSM batch's buffered effects (full SlateDB key -> Some(value)|None)
    /// when this is the LSM arm; `None` on LMDB. `WriteView::read_view` clones
    /// this into a fresh-snapshot read view to give read-your-writes on LSM.
    pub(crate) fn lsm_pending(&self) -> Option<&HashMap<Vec<u8>, Option<Bytes>>> {
        match self {
            AnyWrite::Lmdb(_) => None,
            AnyWrite::Lsm(w) => Some(w.pending()),
        }
    }
}

impl<'s> AnyRead<'s> {
    /// The underlying heed read txn when this read handle is the LMDB arm;
    /// `None` on LSM.
    ///
    /// Bridge for read paths still typed on `&RoTxn` (notably HNSW vector
    /// `search`) during the traversal field flip: an LMDB traversal hands its
    /// shared snapshot's `RoTxn` straight through, so vector search stays
    /// byte-identical. The LSM vector read path is wired separately (Unit 4);
    /// until then callers surface a clear error on the `None` arm.
    pub(crate) fn lmdb_ro(&self) -> Option<&RoTxn<'_>> {
        match self {
            AnyRead::Lmdb(r) => Some(r.ro()),
            AnyRead::Lsm(_) => None,
            AnyRead::LsmReader(_) => None,
            AnyRead::Failed(_) => None,
        }
    }

    /// True when every read through this handle observes one committed
    /// state: LMDB read txns and LSM writer snapshots always, reader-replica
    /// handles only when a `DbSnapshot` was pinned. A latest-state reader
    /// handle (`LsmReader(None)`, including the snapshot-open failure
    /// fallback) may observe a manifest refresh between two reads.
    pub(crate) fn is_snapshot_pinned(&self) -> bool {
        match self {
            AnyRead::Lmdb(_) | AnyRead::Lsm(_) => true,
            AnyRead::LsmReader(snapshot) => snapshot.is_some(),
            AnyRead::Failed(_) => false,
        }
    }
}

/// Build the reader-replica read handle: a snapshot-pinned handle when
/// `HELIX_LSM_READER_SNAPSHOT_READS` is on, else the live latest-state handle.
/// A snapshot-open failure degrades to the live handle (with a warning +
/// metric) rather than failing the read txn: the legacy path serves correctly,
/// and the gateway's reader eviction/fallback already covers persistent reader
/// failures.
fn lsm_reader_read_handle(reader: &LsmReader) -> AnyRead<'static> {
    use super::backend_lsm_reader::reader_snapshot_reads_enabled;
    if !reader_snapshot_reads_enabled() {
        return AnyRead::LsmReader(None);
    }
    match reader.begin_snapshot() {
        Ok(snapshot) => AnyRead::LsmReader(Some(snapshot)),
        Err(error) => {
            metrics::counter!("helix_lsm_reader_snapshot_open_failed_total").increment(1);
            tracing::warn!(%error, "reader snapshot open failed; serving latest-state reads");
            AnyRead::LsmReader(None)
        }
    }
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

pub(crate) fn collection_lsm_path(collection_path: &Path) -> String {
    let collection = collection_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("collection");
    let prefix = std::env::var("HELIX_LSM_PREFIX").unwrap_or_else(|_| "helion".to_string());
    format!("{}/{}", prefix.trim_matches('/'), collection)
}

pub(crate) fn cache_subtree_for_collection_path(collection_path: &Path) -> Option<PathBuf> {
    cache_subtree_for_lsm_path(&collection_lsm_path(collection_path))
}

pub(crate) fn lsm_cache_root_from_env() -> Option<PathBuf> {
    cache_root_from_env()
}

impl AnyBackend {
    /// Wrap an existing heed read txn as a borrowed read handle (LMDB only).
    ///
    /// Lets a call site that still threads a heed `&RoTxn` read through the
    /// backend's handle API (`get_with` and the `_be` twins) during the read-path
    /// flip, without opening a second snapshot. The LSM arm is unreachable until
    /// construction sites thread `AnyRead` directly (no cutover), so it panics.
    pub(crate) fn read_borrowed<'a>(&self, txn: &'a RoTxn<'a>) -> AnyRead<'a> {
        match self {
            AnyBackend::Lmdb(_) => AnyRead::Lmdb(LmdbRead::borrowed(txn)),
            // `begin_read` can legitimately fail in production — e.g. a fenced
            // writer whose SlateDB manifest was taken by a newer client returns
            // "Closed error: detected newer DB client". `read_borrowed` returns
            // `AnyRead` by value, so we cannot surface the `Err` here; carry it in
            // `AnyRead::Failed` so the first read returns a clean error instead of
            // panicking the worker on a routine op.
            AnyBackend::Lsm(b) => match b.begin_read() {
                Ok(r) => AnyRead::Lsm(r),
                Err(e) => AnyRead::Failed(e),
            },
            // No borrowed heed txn to wrap on the reader replica; construct the
            // same handle `begin_read` would (snapshot-pinned when enabled).
            AnyBackend::LsmReader(reader) => lsm_reader_read_handle(reader),
        }
    }

    /// Build an LSM read handle whose point reads first consult `pending` — a
    /// fresh committed snapshot with the live write batch's buffered effects
    /// overlaid (read-your-writes). Only called on the LSM path by
    /// `WriteView::read_view`; the LMDB arm is `unreachable!()`.
    pub(crate) fn lsm_read_with_pending(
        &self,
        pending: HashMap<Vec<u8>, Option<Bytes>>,
    ) -> Result<AnyRead<'_>, BackendError> {
        match self {
            AnyBackend::Lmdb(_) => unreachable!("lsm_read_with_pending is LSM-only"),
            AnyBackend::Lsm(b) => Ok(AnyRead::Lsm(b.begin_read_with_pending(pending)?)),
            // Read-your-writes overlay is a writer-batch concern; a reader replica
            // never holds a write batch.
            AnyBackend::LsmReader(_) => unreachable!("lsm_read_with_pending is writer-only"),
        }
    }

    pub(crate) fn get_with_heed<R>(
        &self,
        txn: &RoTxn,
        ns: Namespace<'_>,
        key: &[u8],
        f: impl FnOnce(Option<&[u8]>) -> R,
    ) -> Result<R, BackendError> {
        match self {
            AnyBackend::Lmdb(b) => b.get_with_raw(txn, ns, key, f),
            AnyBackend::Lsm(b) => {
                let r = b.begin_read()?;
                b.get_with(&r, ns, key, f)
            }
            AnyBackend::LsmReader(b) => b.get_with(ns, key, f),
        }
    }

    pub(crate) fn scan_heed(
        &self,
        txn: &RoTxn,
        ns: Namespace<'_>,
        range: KeyRange,
        visit: impl FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        match self {
            AnyBackend::Lmdb(b) => b.scan_raw(txn, ns, range, visit),
            AnyBackend::Lsm(b) => {
                let r = b.begin_read()?;
                b.scan(&r, ns, range, visit)
            }
            AnyBackend::LsmReader(b) => b.scan_with(ns, range, visit),
        }
    }

    pub(crate) fn scan_streaming(
        &self,
        txn: &AnyRead<'_>,
        ns: Namespace<'_>,
        range: KeyRange,
        visit: impl FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        match self {
            AnyBackend::Lmdb(b) => {
                let AnyRead::Lmdb(r) = txn else {
                    unreachable!("read handle does not match active backend")
                };
                b.scan(r, ns, range, visit)
            }
            AnyBackend::Lsm(b) => match txn {
                AnyRead::Lsm(r) => b.scan_streaming(r, ns, range, visit),
                AnyRead::Failed(e) => Err(e.clone()),
                _ => unreachable!("read handle does not match active backend"),
            },
            AnyBackend::LsmReader(b) => {
                let AnyRead::LsmReader(snap) = txn else {
                    unreachable!("read handle does not match active backend")
                };
                b.scan_streaming_at(snap.as_ref(), ns, range, visit)
            }
        }
    }

    pub(crate) fn for_each_dup_heed(
        &self,
        txn: &RoTxn,
        ns: Namespace<'_>,
        key: &[u8],
        visit: impl FnMut(&[u8]) -> bool,
    ) -> Result<(), BackendError> {
        match self {
            AnyBackend::Lmdb(b) => b.for_each_dup_raw(txn, ns, key, visit),
            AnyBackend::Lsm(b) => {
                let r = b.begin_read()?;
                b.for_each_dup(&r, ns, key, visit)
            }
            AnyBackend::LsmReader(b) => b.for_each_dup_with(ns, key, visit),
        }
    }

    pub(crate) fn put_heed(
        &self,
        txn: &mut RwTxn,
        ns: Namespace<'_>,
        key: &[u8],
        val: &[u8],
    ) -> Result<(), BackendError> {
        match self {
            AnyBackend::Lmdb(b) => b.put_raw(txn, ns, key, val),
            AnyBackend::Lsm(_) => unreachable!("heed write path is LMDB-only"),
            AnyBackend::LsmReader(_) => unreachable!("heed write path is LMDB-only"),
        }
    }

    pub(crate) fn delete_heed(
        &self,
        txn: &mut RwTxn,
        ns: Namespace<'_>,
        key: &[u8],
    ) -> Result<(), BackendError> {
        match self {
            AnyBackend::Lmdb(b) => b.delete_raw(txn, ns, key),
            AnyBackend::Lsm(_) => unreachable!("heed write path is LMDB-only"),
            AnyBackend::LsmReader(_) => unreachable!("heed write path is LMDB-only"),
        }
    }

    pub(crate) fn put_dup_heed(
        &self,
        txn: &mut RwTxn,
        ns: Namespace<'_>,
        key: &[u8],
        value: &[u8],
    ) -> Result<(), BackendError> {
        match self {
            AnyBackend::Lmdb(b) => b.put_dup_raw(txn, ns, key, value),
            AnyBackend::Lsm(_) => unreachable!("heed write path is LMDB-only"),
            AnyBackend::LsmReader(_) => unreachable!("heed write path is LMDB-only"),
        }
    }

    pub(crate) fn delete_dup_heed(
        &self,
        txn: &mut RwTxn,
        ns: Namespace<'_>,
        key: &[u8],
        value: &[u8],
    ) -> Result<(), BackendError> {
        match self {
            AnyBackend::Lmdb(b) => b.delete_dup_raw(txn, ns, key, value),
            AnyBackend::Lsm(_) => unreachable!("heed write path is LMDB-only"),
            AnyBackend::LsmReader(_) => unreachable!("heed write path is LMDB-only"),
        }
    }

    pub(crate) fn get_for_update_heed<R>(
        &self,
        txn: &RwTxn,
        ns: Namespace<'_>,
        key: &[u8],
        f: impl FnOnce(Option<&[u8]>) -> R,
    ) -> Result<R, BackendError> {
        match self {
            AnyBackend::Lmdb(b) => b.get_for_update_raw(txn, ns, key, f),
            AnyBackend::Lsm(_) => unreachable!("heed write path is LMDB-only"),
            AnyBackend::LsmReader(_) => unreachable!("heed write path is LMDB-only"),
        }
    }
}

impl StorageBackend for AnyBackend {
    type Read<'s>
        = AnyRead<'s>
    where
        Self: 's;
    type Write<'s>
        = AnyWrite<'s>
    where
        Self: 's;

    fn begin_read(&self) -> Result<Self::Read<'_>, BackendError> {
        match self {
            AnyBackend::Lmdb(b) => Ok(AnyRead::Lmdb(b.begin_read()?)),
            AnyBackend::Lsm(b) => Ok(AnyRead::Lsm(b.begin_read()?)),
            AnyBackend::LsmReader(reader) => Ok(lsm_reader_read_handle(reader)),
        }
    }

    fn begin_write(&self) -> Result<Self::Write<'_>, BackendError> {
        match self {
            AnyBackend::Lmdb(b) => Ok(AnyWrite::Lmdb(b.begin_write()?)),
            AnyBackend::Lsm(b) => Ok(AnyWrite::Lsm(b.begin_write()?)),
            AnyBackend::LsmReader(_) => {
                Err(BackendError::Unsupported(LSM_READER_READONLY.to_string()))
            }
        }
    }

    fn commit(&self, txn: Self::Write<'_>) -> Result<(), BackendError> {
        match self {
            AnyBackend::Lmdb(b) => {
                let AnyWrite::Lmdb(w) = txn else {
                    unreachable!("write handle does not match active backend")
                };
                b.commit(w)
            }
            AnyBackend::Lsm(b) => {
                let AnyWrite::Lsm(w) = txn else {
                    unreachable!("write handle does not match active backend")
                };
                b.commit(w)
            }
            AnyBackend::LsmReader(_) => {
                Err(BackendError::Unsupported(LSM_READER_READONLY.to_string()))
            }
        }
    }

    fn commit_buffered(&self, txn: Self::Write<'_>) -> Result<(), BackendError> {
        match self {
            AnyBackend::Lmdb(b) => {
                let AnyWrite::Lmdb(w) = txn else {
                    unreachable!("write handle does not match active backend")
                };
                b.commit_buffered(w)
            }
            AnyBackend::Lsm(b) => {
                let AnyWrite::Lsm(w) = txn else {
                    unreachable!("write handle does not match active backend")
                };
                b.commit_buffered(w)
            }
            AnyBackend::LsmReader(_) => {
                Err(BackendError::Unsupported(LSM_READER_READONLY.to_string()))
            }
        }
    }

    fn flush_durable(&self) -> Result<(), BackendError> {
        match self {
            AnyBackend::Lmdb(b) => b.flush_durable(),
            AnyBackend::Lsm(b) => b.flush_durable(),
            // A read-only replica has no buffered writes to make durable.
            AnyBackend::LsmReader(_) => Ok(()),
        }
    }

    fn get_with<R>(
        &self,
        txn: &Self::Read<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        f: impl FnOnce(Option<&[u8]>) -> R,
    ) -> Result<R, BackendError> {
        match self {
            AnyBackend::Lmdb(b) => {
                let AnyRead::Lmdb(r) = txn else {
                    unreachable!("read handle does not match active backend")
                };
                b.get_with(r, ns, key, f)
            }
            AnyBackend::Lsm(b) => match txn {
                AnyRead::Lsm(r) => b.get_with(r, ns, key, f),
                // Deferred open failure (e.g. a fenced writer): surface it cleanly
                // instead of panicking on the `let-else`.
                AnyRead::Failed(e) => Err(e.clone()),
                _ => unreachable!("read handle does not match active backend"),
            },
            AnyBackend::LsmReader(b) => {
                let AnyRead::LsmReader(snap) = txn else {
                    unreachable!("read handle does not match active backend")
                };
                b.get_with_at(snap.as_ref(), ns, key, f)
            }
        }
    }

    fn scan(
        &self,
        txn: &Self::Read<'_>,
        ns: Namespace<'_>,
        range: KeyRange,
        visit: impl FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        match self {
            AnyBackend::Lmdb(b) => {
                let AnyRead::Lmdb(r) = txn else {
                    unreachable!("read handle does not match active backend")
                };
                b.scan(r, ns, range, visit)
            }
            AnyBackend::Lsm(b) => match txn {
                AnyRead::Lsm(r) => b.scan(r, ns, range, visit),
                // Deferred open failure (e.g. a fenced writer): surface it cleanly
                // instead of panicking on the `let-else`.
                AnyRead::Failed(e) => Err(e.clone()),
                _ => unreachable!("read handle does not match active backend"),
            },
            AnyBackend::LsmReader(b) => {
                let AnyRead::LsmReader(snap) = txn else {
                    unreachable!("read handle does not match active backend")
                };
                b.scan_with_at(snap.as_ref(), ns, range, visit)
            }
        }
    }

    fn put(
        &self,
        txn: &mut Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        val: &[u8],
    ) -> Result<(), BackendError> {
        match self {
            AnyBackend::Lmdb(b) => {
                let AnyWrite::Lmdb(w) = txn else {
                    unreachable!("write handle does not match active backend")
                };
                b.put(w, ns, key, val)
            }
            AnyBackend::Lsm(b) => {
                let AnyWrite::Lsm(w) = txn else {
                    unreachable!("write handle does not match active backend")
                };
                b.put(w, ns, key, val)
            }
            AnyBackend::LsmReader(_) => {
                Err(BackendError::Unsupported(LSM_READER_READONLY.to_string()))
            }
        }
    }

    fn delete(
        &self,
        txn: &mut Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
    ) -> Result<(), BackendError> {
        match self {
            AnyBackend::Lmdb(b) => {
                let AnyWrite::Lmdb(w) = txn else {
                    unreachable!("write handle does not match active backend")
                };
                b.delete(w, ns, key)
            }
            AnyBackend::Lsm(b) => {
                let AnyWrite::Lsm(w) = txn else {
                    unreachable!("write handle does not match active backend")
                };
                b.delete(w, ns, key)
            }
            AnyBackend::LsmReader(_) => {
                Err(BackendError::Unsupported(LSM_READER_READONLY.to_string()))
            }
        }
    }

    fn merge(
        &self,
        txn: &mut Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        val: &[u8],
    ) -> Result<(), BackendError> {
        match self {
            AnyBackend::Lmdb(b) => {
                let AnyWrite::Lmdb(w) = txn else {
                    unreachable!("write handle does not match active backend")
                };
                b.merge(w, ns, key, val)
            }
            AnyBackend::Lsm(b) => {
                let AnyWrite::Lsm(w) = txn else {
                    unreachable!("write handle does not match active backend")
                };
                b.merge(w, ns, key, val)
            }
            AnyBackend::LsmReader(_) => {
                Err(BackendError::Unsupported(LSM_READER_READONLY.to_string()))
            }
        }
    }

    fn merge_commutative(
        &self,
        txn: &mut Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        val: &[u8],
    ) -> Result<(), BackendError> {
        match self {
            AnyBackend::Lmdb(b) => {
                let AnyWrite::Lmdb(w) = txn else {
                    unreachable!("write handle does not match active backend")
                };
                b.merge_commutative(w, ns, key, val)
            }
            AnyBackend::Lsm(b) => {
                let AnyWrite::Lsm(w) = txn else {
                    unreachable!("write handle does not match active backend")
                };
                b.merge_commutative(w, ns, key, val)
            }
            AnyBackend::LsmReader(_) => {
                Err(BackendError::Unsupported(LSM_READER_READONLY.to_string()))
            }
        }
    }

    fn put_dup(
        &self,
        txn: &mut Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        value: &[u8],
    ) -> Result<(), BackendError> {
        match self {
            AnyBackend::Lmdb(b) => {
                let AnyWrite::Lmdb(w) = txn else {
                    unreachable!("write handle does not match active backend")
                };
                b.put_dup(w, ns, key, value)
            }
            AnyBackend::Lsm(b) => {
                let AnyWrite::Lsm(w) = txn else {
                    unreachable!("write handle does not match active backend")
                };
                b.put_dup(w, ns, key, value)
            }
            AnyBackend::LsmReader(_) => {
                Err(BackendError::Unsupported(LSM_READER_READONLY.to_string()))
            }
        }
    }

    fn delete_dup(
        &self,
        txn: &mut Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        value: &[u8],
    ) -> Result<(), BackendError> {
        match self {
            AnyBackend::Lmdb(b) => {
                let AnyWrite::Lmdb(w) = txn else {
                    unreachable!("write handle does not match active backend")
                };
                b.delete_dup(w, ns, key, value)
            }
            AnyBackend::Lsm(b) => {
                let AnyWrite::Lsm(w) = txn else {
                    unreachable!("write handle does not match active backend")
                };
                b.delete_dup(w, ns, key, value)
            }
            AnyBackend::LsmReader(_) => {
                Err(BackendError::Unsupported(LSM_READER_READONLY.to_string()))
            }
        }
    }

    fn for_each_dup(
        &self,
        txn: &Self::Read<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        visit: impl FnMut(&[u8]) -> bool,
    ) -> Result<(), BackendError> {
        match self {
            AnyBackend::Lmdb(b) => {
                let AnyRead::Lmdb(r) = txn else {
                    unreachable!("read handle does not match active backend")
                };
                b.for_each_dup(r, ns, key, visit)
            }
            AnyBackend::Lsm(b) => match txn {
                AnyRead::Lsm(r) => b.for_each_dup(r, ns, key, visit),
                // Deferred open failure (e.g. a fenced writer): surface it cleanly
                // instead of panicking on the `let-else`.
                AnyRead::Failed(e) => Err(e.clone()),
                _ => unreachable!("read handle does not match active backend"),
            },
            AnyBackend::LsmReader(b) => {
                let AnyRead::LsmReader(snap) = txn else {
                    unreachable!("read handle does not match active backend")
                };
                b.for_each_dup_with_at(snap.as_ref(), ns, key, visit)
            }
        }
    }

    fn get_for_update<R>(
        &self,
        txn: &Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        f: impl FnOnce(Option<&[u8]>) -> R,
    ) -> Result<R, BackendError> {
        match self {
            AnyBackend::Lmdb(b) => {
                let AnyWrite::Lmdb(w) = txn else {
                    unreachable!("write handle does not match active backend")
                };
                b.get_for_update(w, ns, key, f)
            }
            AnyBackend::Lsm(b) => {
                let AnyWrite::Lsm(w) = txn else {
                    unreachable!("write handle does not match active backend")
                };
                b.get_for_update(w, ns, key, f)
            }
            AnyBackend::LsmReader(_) => {
                Err(BackendError::Unsupported(LSM_READER_READONLY.to_string()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn round_trip(be: &AnyBackend) {
        let mut w = be.begin_write().unwrap();
        be.put(&mut w, Namespace::Nodes, b"k1", b"v1").unwrap();
        be.commit(w).unwrap();

        let r = be.begin_read().unwrap();
        let got = be
            .get_with(&r, Namespace::Nodes, b"k1", |v| v.map(|b| b.to_vec()))
            .unwrap();
        assert_eq!(got, Some(b"v1".to_vec()));

        let mut keys = Vec::new();
        be.scan(&r, Namespace::Nodes, KeyRange::all(), |k, _v| {
            keys.push(k.to_vec());
            true
        })
        .unwrap();
        assert_eq!(keys, vec![b"k1".to_vec()]);
    }

    #[test]
    fn any_backend_lmdb_dispatch() {
        let dir = TempDir::new().unwrap();
        let be = AnyBackend::open_lmdb(dir.path(), 64, 10 * 1024 * 1024).unwrap();
        assert_eq!(be.kind(), BackendKind::Lmdb);
        round_trip(&be);
    }

    #[test]
    fn read_borrowed_reads_through_existing_txn_lmdb() {
        use heed3::EnvOpenOptions;
        let dir = TempDir::new().unwrap();
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(10 * 1024 * 1024)
                .max_dbs(64)
                .open(dir.path())
                .unwrap()
        };
        let be = AnyBackend::Lmdb(LmdbBackend::from_env(env.clone()));
        let mut w = be.begin_write().unwrap();
        be.put(&mut w, Namespace::Nodes, b"k", b"v").unwrap();
        be.commit(w).unwrap();

        // Borrow an existing RoTxn (as a mid-migration read op would) and read
        // through the handle API.
        let ro = env.read_txn().unwrap();
        let r = be.read_borrowed(&ro);
        let got = be
            .get_with(&r, Namespace::Nodes, b"k", |v| v.map(|b| b.to_vec()))
            .unwrap();
        assert_eq!(got, Some(b"v".to_vec()));
    }

    #[test]
    fn any_backend_heed_round_trip_lmdb() {
        use heed3::EnvOpenOptions;
        let dir = TempDir::new().unwrap();
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(10 * 1024 * 1024)
                .max_dbs(64)
                .open(dir.path())
                .unwrap()
        };
        let be = AnyBackend::Lmdb(LmdbBackend::from_env(env.clone()));
        {
            let mut w = env.write_txn().unwrap();
            be.put_heed(&mut w, Namespace::Nodes, b"k1", b"v1").unwrap();
            let seen = be
                .get_for_update_heed(&w, Namespace::Nodes, b"k1", |v| v.map(|b| b.to_vec()))
                .unwrap();
            assert_eq!(seen, Some(b"v1".to_vec()));
            w.commit().unwrap();
        }
        {
            let ro = env.read_txn().unwrap();
            let got = be
                .get_with_heed(&ro, Namespace::Nodes, b"k1", |v| v.map(|b| b.to_vec()))
                .unwrap();
            assert_eq!(got, Some(b"v1".to_vec()));
        }
    }

    #[test]
    fn any_backend_lsm_dispatch() {
        let be = AnyBackend::open_lsm_in_memory("/any-lsm-test").unwrap();
        assert_eq!(be.kind(), BackendKind::Lsm);
        round_trip(&be);
    }

    #[test]
    fn close_lsm_for_cache_eviction_quiesces_writer_only() {
        let be = AnyBackend::open_lsm_in_memory("/any-lsm-close-test").unwrap();
        round_trip(&be);
        be.close_lsm_for_cache_eviction()
            .expect("LSM writer close should succeed");
        be.close_lsm_for_cache_eviction()
            .expect("LSM writer close should be idempotent");

        let mut closed_write = be
            .begin_write()
            .expect("SlateDB may still construct a local batch after close");
        be.put(&mut closed_write, Namespace::Nodes, b"after-close", b"v")
            .expect("buffering into a local batch should not require the writer task");
        let commit = be.commit(closed_write);
        assert!(
            commit.is_err(),
            "closed LSM writer must reject durable commits"
        );

        let dir = TempDir::new().unwrap();
        let lmdb =
            AnyBackend::open_lmdb(dir.path(), 16, 1024 * 1024).expect("LMDB backend should open");
        lmdb.close_lsm_for_cache_eviction()
            .expect("LMDB close helper is a no-op");
    }

    /// C1: a borrowed-read handle whose snapshot failed to open (e.g. a fenced
    /// writer whose SlateDB manifest was taken by a newer client — `begin_read`
    /// returns `Err`) must surface that error cleanly through every read method
    /// instead of panicking the worker. `read_borrowed` carries the open error in
    /// `AnyRead::Failed`; the first read returns it.
    #[test]
    fn failed_read_handle_surfaces_error_not_panic() {
        let be = AnyBackend::open_lsm_in_memory("/failed-read-handle-test").unwrap();
        let failed: AnyRead<'_> = AnyRead::Failed(BackendError::Conflict(
            "detected newer DB client".to_string(),
        ));

        // Point read returns the carried error rather than panicking.
        let got = be.get_with(&failed, Namespace::Nodes, b"k", |v| v.map(|b| b.to_vec()));
        assert!(
            matches!(got, Err(BackendError::Conflict(_))),
            "got: {got:?}"
        );

        // Ordered scan returns the carried error rather than panicking.
        let scan = be.scan(&failed, Namespace::Nodes, KeyRange::all(), |_k, _v| true);
        assert!(
            matches!(scan, Err(BackendError::Conflict(_))),
            "scan: {scan:?}"
        );

        // Dup read returns the carried error rather than panicking.
        let dup = be.for_each_dup(&failed, Namespace::OutEdges, b"n1", |_v| true);
        assert!(
            matches!(dup, Err(BackendError::Conflict(_))),
            "dup: {dup:?}"
        );

        // A failed handle exposes no heed txn; callers needing one error out.
        assert!(failed.lmdb_ro().is_none());
    }

    /// Reader-replica arm end-to-end: an `LsmBackend` writer commits a plain key
    /// and dup rows to a shared in-memory object store; an `AnyBackend::LsmReader`
    /// opened on the SAME store reads them back through the seam
    /// (`begin_read`+`get_with`+`scan`+`for_each_dup`) byte-identically, and every
    /// write entry point returns the read-only error. The shared `Arc<InMemory>`
    /// stands in for a writer + reader replica over shared object storage.
    #[test]
    fn lsm_reader_backend_serves_writer_data() {
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use crate::helix_engine::storage_core::backend_lsm_reader::LsmReader;
        use object_store::memory::InMemory;
        use object_store::ObjectStore;
        use std::sync::Arc;

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = "reader-arm-test";

        // Writer: a plain Nodes key + three OutEdges dup values under "n1".
        let writer =
            LsmBackend::open_with_store(path, store.clone()).expect("open writer on shared store");
        let mut w = writer.begin_write().unwrap();
        writer.put(&mut w, Namespace::Nodes, b"k1", b"v1").unwrap();
        writer
            .put_dup(&mut w, Namespace::OutEdges, b"n1", b"a")
            .unwrap();
        writer
            .put_dup(&mut w, Namespace::OutEdges, b"n1", b"b")
            .unwrap();
        writer
            .put_dup(&mut w, Namespace::OutEdges, b"n1", b"c")
            .unwrap();
        writer.commit(w).unwrap();

        // Reader replica over the SAME store, through the AnyBackend seam.
        let reader = AnyBackend::LsmReader(
            LsmReader::open_with_store(path, store.clone()).expect("open reader on shared store"),
        );
        assert_eq!(reader.kind(), BackendKind::Lsm);

        let r = reader.begin_read().unwrap();

        // Plain-namespace point read.
        let got = reader
            .get_with(&r, Namespace::Nodes, b"k1", |v| v.map(|b| b.to_vec()))
            .unwrap();
        assert_eq!(
            got,
            Some(b"v1".to_vec()),
            "reader must see the writer's committed Nodes value"
        );

        // Plain-namespace scan strips the prefix and returns the stored pair.
        let mut nodes = Vec::new();
        reader
            .scan(&r, Namespace::Nodes, KeyRange::all(), |k, v| {
                nodes.push((k.to_vec(), v.to_vec()));
                true
            })
            .unwrap();
        assert_eq!(nodes, vec![(b"k1".to_vec(), b"v1".to_vec())]);

        // Dup-namespace read returns every value under the key, in order.
        let mut edges = Vec::new();
        reader
            .for_each_dup(&r, Namespace::OutEdges, b"n1", |v| {
                edges.push(v.to_vec());
                true
            })
            .unwrap();
        assert_eq!(edges, vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]);

        // Every write entry point on the reader arm is rejected at the backend.
        assert!(
            matches!(reader.begin_write(), Err(BackendError::Unsupported(_))),
            "reader replica must reject begin_write"
        );
    }

    /// The read-gated poll-tier promotion/demotion calls
    /// (`refresh_lsm_reader_with_poll_interval[_bounded]`) must be a pure
    /// no-op on the writer (`Lsm`) and legacy (`Lmdb`) arms — reader poll
    /// tiering only makes sense for reader replicas. `Ok(false)` (not an
    /// error) so callers can skip non-reader collections without downcasting.
    #[test]
    fn reader_poll_tier_refresh_is_noop_on_writer_and_lmdb_arms() {
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use object_store::memory::InMemory;
        use object_store::ObjectStore;
        use std::sync::Arc;

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let writer = AnyBackend::Lsm(
            LsmBackend::open_with_store("poll-tier-noop-test", store).expect("open writer"),
        );
        assert!(
            !writer
                .refresh_lsm_reader_with_poll_interval(std::time::Duration::from_secs(5))
                .unwrap(),
            "writer arm must be a no-op, not an error"
        );
        assert!(
            !writer
                .refresh_lsm_reader_with_poll_interval_bounded(
                    std::time::Duration::from_secs(5),
                    std::time::Duration::from_secs(2)
                )
                .unwrap(),
            "writer arm must be a no-op, not an error"
        );
    }

    #[test]
    fn unhealthy_lsm_writer_close_reason_flags_only_terminal_writer_failures() {
        assert_eq!(
            unhealthy_lsm_writer_close_reason_from(Some(CloseReason::Fenced)),
            Some(CloseReason::Fenced)
        );
        assert_eq!(
            unhealthy_lsm_writer_close_reason_from(Some(CloseReason::Panic)),
            Some(CloseReason::Panic)
        );
        assert_eq!(unhealthy_lsm_writer_close_reason_from(None), None);
        assert_eq!(
            unhealthy_lsm_writer_close_reason_from(Some(CloseReason::Clean)),
            None
        );
    }
}
