//! `LmdbBackend` — the [`StorageBackend`] implementation over LMDB (`heed3`).
//!
//! This is the default backend and preserves today's behavior: every
//! [`Namespace`] maps to a `heed3::Database<Bytes, Bytes>`, reads borrow from
//! the read txn (zero-copy via the `get_with` / `scan` visitors), and writes go
//! through a `heed3::RwTxn` committed atomically.
//!
//! Namespace -> database-name mapping intentionally matches the existing on-disk
//! names (`vectors_{seg}`, `hnsw_out_{seg}`, ...) so this backend can later be
//! pointed at the real collection env during the call-site migration (US-004+).

#![allow(dead_code)]

use std::collections::HashMap;
use std::ops::Bound;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::Duration;

use heed3::types::Bytes;
use heed3::{Database, DatabaseFlags, Env, EnvOpenOptions, PutFlags, RoTxn, RwTxn, WithTls};

use super::backend::{
    is_dup, BackendError, KeyRange, Namespace, ReadTxn, SegmentDb, SparseDb, StorageBackend,
    WriteHint, WriteTxn,
};

const CATALOG_REFRESH_INTERVAL_MS: u64 = 250;
const CATALOG_WAIT_BUDGET_MS: u64 = 50;

struct CatalogSync {
    state: Mutex<CatalogState>,
    wake: Condvar,
    published: Condvar,
}

struct CatalogState {
    requests: u64,
    served: u64,
}

impl CatalogSync {
    fn new() -> Self {
        CatalogSync {
            state: Mutex::new(CatalogState {
                requests: 1,
                served: 0,
            }),
            wake: Condvar::new(),
            published: Condvar::new(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, CatalogState> {
        match self.state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn request_and_wait(&self) {
        let mut state = self.lock();
        state.requests += 1;
        let mine = state.requests;
        self.wake.notify_one();
        let deadline = Duration::from_millis(CATALOG_WAIT_BUDGET_MS);
        let (guard, _timeout) = match self
            .published
            .wait_timeout_while(state, deadline, |s| s.served < mine)
        {
            Ok(pair) => pair,
            Err(poisoned) => poisoned.into_inner(),
        };
        drop(guard);
    }

    fn await_work(&self) {
        let state = self.lock();
        if state.served < state.requests {
            return;
        }
        let interval = Duration::from_millis(CATALOG_REFRESH_INTERVAL_MS);
        let (guard, _timeout) = match self
            .wake
            .wait_timeout_while(state, interval, |s| s.served >= s.requests)
        {
            Ok(pair) => pair,
            Err(poisoned) => poisoned.into_inner(),
        };
        drop(guard);
    }

    fn begin_pass(&self) -> u64 {
        self.lock().requests
    }

    fn publish_pass(&self, snapshot: u64) {
        let mut state = self.lock();
        if state.served < snapshot {
            state.served = snapshot;
        }
        self.published.notify_all();
    }
}

fn io<E: std::fmt::Display>(e: E) -> BackendError {
    BackendError::Io(e.to_string())
}

fn io_at<E: std::fmt::Display>(site: &'static str) -> impl Fn(E) -> BackendError {
    move |e| BackendError::Io(format!("[{site}] {e}"))
}

/// Resolve a [`Namespace`] to a stable name. Shared with the LSM backend (which
/// uses it as a key prefix) so both engines address namespaces identically.
pub(crate) fn ns_name(ns: Namespace<'_>) -> String {
    match ns {
        Namespace::Nodes => "nodes".to_string(),
        Namespace::Edges => "edges".to_string(),
        Namespace::OutEdges => "out_edges".to_string(),
        Namespace::InEdges => "in_edges".to_string(),
        Namespace::EdgePathIdx => "edge_path_idx".to_string(),
        Namespace::Metadata => "metadata".to_string(),
        Namespace::DenseTombstones => "dense_tombstones".to_string(),
        Namespace::SecondaryIndex(name) => name.to_string(),
        Namespace::PayloadIndex(name) => name.to_string(),
        // Build-delta sidecar for an online LSM payload-index build. The
        // `name` here is the SAME physical index db name as PayloadIndex, so
        // the two stay paired; the `pidxb_` prefix keeps the sidecar in its
        // own keyspace. LMDB never writes it (builds there hold the write
        // gate), but the mapping must exist so the namespace is total.
        Namespace::PayloadIndexBuild(name) => format!("pidxb_{name}"),
        // Matches the on-disk name used by production multi-indices
        // (`midx_name` auto-provisioned in storage_core.rs; `create_multi_index`
        // uses `format!("midx_{name}")`). Must stay `midx_` so the backend reads
        // the SAME DUP_SORT database, not a divergent empty one.
        Namespace::MultiIndex(name) => format!("midx_{name}"),
        Namespace::Segment { physical_name, db } => {
            let prefix = match db {
                SegmentDb::Vectors => "vectors",
                SegmentDb::VectorData => "vector_data",
                SegmentDb::HnswOut => "hnsw_out",
                SegmentDb::HnswNeighbors => "hnsw_neighbors",
                SegmentDb::Ordinals => "vector_ordinals",
                SegmentDb::IvfCentroids => "ivf_centroids",
                SegmentDb::IvfPostings => "ivf_postings",
                SegmentDb::SimHash => "hnsw_simhash",
            };
            format!("{prefix}_{physical_name}")
        }
        // Byte-identical to the heed DB names created in SparseVectorCore::new
        // (`sparse_inv_{name}` / `sparse_fwd_{name}` / `sparse_meta_{name}`), so the
        // backend addresses the SAME databases — the inverted index stays the
        // DUP_SORT|DUP_FIXED dup namespace (see `is_dup`).
        Namespace::SparseSegment { physical_name, db } => {
            let prefix = match db {
                SparseDb::Inv => "sparse_inv",
                SparseDb::Fwd => "sparse_fwd",
                SparseDb::Meta => "sparse_meta",
            };
            format!("{prefix}_{physical_name}")
        }
    }
}

/// LMDB-backed storage engine.
pub struct LmdbBackend {
    env: Env<WithTls>,
    /// Cache of opened `Database` handles keyed by namespace name. `heed3`
    /// `Database` is `Copy` and valid for the env lifetime once opened.
    dbs: Arc<RwLock<HashMap<String, Database<Bytes, Bytes>>>>,
    refresher_alive: Arc<AtomicBool>,
    catalog_sync: Arc<CatalogSync>,
}

impl Drop for LmdbBackend {
    fn drop(&mut self) {
        self.refresher_alive.store(false, Ordering::Release);
    }
}

/// Read snapshot: either an owned `heed3` read txn (from `begin_read`) or a
/// borrowed one — an existing `&RoTxn` threaded by a call site mid-migration via
/// [`AnyBackend::read_borrowed`]. Both expose the txn through [`LmdbRead::ro`].
pub struct LmdbRead<'s> {
    txn: RoHandle<'s>,
}
impl ReadTxn for LmdbRead<'_> {}

enum RoHandle<'s> {
    Owned(RoTxn<'s, WithTls>),
    // heed's read methods all take `&RoTxn` (default `AnyTls`); the traversal
    // layer threads `RoTxn<AnyTls>`. Borrow it in that erased form so call sites
    // can hand us their existing snapshot without a TLS-marker mismatch.
    Borrowed(&'s RoTxn<'s>),
}

impl<'s> LmdbRead<'s> {
    /// Wrap an existing read txn without taking ownership (zero-copy borrow).
    pub(crate) fn borrowed(txn: &'s RoTxn<'s>) -> Self {
        Self {
            txn: RoHandle::Borrowed(txn),
        }
    }

    /// The underlying heed read txn (erased `AnyTls` form, which every heed read
    /// method accepts), whether owned or borrowed.
    #[inline]
    pub(crate) fn ro(&self) -> &RoTxn<'_> {
        match &self.txn {
            // `RoTxn<WithTls>` derefs to `RoTxn<AnyTls>`.
            RoHandle::Owned(t) => t,
            RoHandle::Borrowed(t) => t,
        }
    }
}

/// Write batch: wraps a `heed3` write transaction (committed atomically).
pub struct LmdbWrite<'s> {
    txn: RwTxn<'s>,
}
impl WriteTxn for LmdbWrite<'_> {}

impl<'s> LmdbWrite<'s> {
    /// Wrap an existing heed write txn (used by the resize-safe write-view entry
    /// so `AnyWrite::Lmdb` owns the same `RwTxn` the resize fence tracks).
    pub(crate) fn from_txn(txn: RwTxn<'s>) -> Self {
        Self { txn }
    }

    /// Mutable access to the underlying heed write txn. Bridge for write paths
    /// not yet on the seam (vector insert, bulk APPEND) during the field flip.
    pub(crate) fn txn_mut(&mut self) -> &mut RwTxn<'s> {
        &mut self.txn
    }

    /// Shared access to the underlying heed write txn (read-your-writes). A heed
    /// `RwTxn` derefs to `RoTxn`, so this serves read paths still typed on
    /// `&RoTxn` (vector existence check) that must observe this batch's writes.
    pub(crate) fn txn_ref(&self) -> &RwTxn<'s> {
        &self.txn
    }
}

impl LmdbBackend {
    /// Wrap an already-open environment. `heed3::Env` is `Arc`-backed, so this
    /// shares the same underlying LMDB env cheaply — used to route Helion's
    /// existing collection env through the backend during call-site migration.
    pub fn from_env(env: Env<WithTls>) -> Self {
        let dbs: Arc<RwLock<HashMap<String, Database<Bytes, Bytes>>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let refresher_alive = Arc::new(AtomicBool::new(true));
        let catalog_sync = Arc::new(CatalogSync::new());
        {
            let env = env.clone();
            let dbs = Arc::clone(&dbs);
            let alive = Arc::clone(&refresher_alive);
            let sync = Arc::clone(&catalog_sync);
            std::thread::spawn(move || {
                while alive.load(Ordering::Acquire) {
                    let snapshot = sync.begin_pass();
                    let pass = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        Self::open_catalog(&env)
                    }));
                    match pass {
                        Ok(Ok(found)) => {
                            let mut cache = match dbs.write() {
                                Ok(guard) => guard,
                                Err(poisoned) => poisoned.into_inner(),
                            };
                            for (name, db) in found {
                                cache.insert(name, db);
                            }
                        }
                        Ok(Err(error)) => {
                            tracing::warn!(%error, "catalog scan failed; retrying on the next pass");
                        }
                        Err(_) => {
                            tracing::warn!("catalog scan panicked; retrying on the next pass");
                        }
                    }
                    sync.publish_pass(snapshot);
                    sync.await_work();
                }
            });
        }
        Self {
            env,
            dbs,
            refresher_alive,
            catalog_sync,
        }
    }

    /// Open (creating if absent) an LMDB environment at `path`.
    pub fn open(path: &Path, max_dbs: u32, map_size: usize) -> Result<Self, BackendError> {
        std::fs::create_dir_all(path).map_err(io)?;
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(map_size)
                .max_dbs(max_dbs)
                .open(path)
                .map_err(io)?
        };
        Ok(Self::from_env(env))
    }

    /// Resolve a namespace to its database for reading. `Ok(None)` if the
    /// database has not been created yet (treat as empty).
    fn resolve_read(
        &self,
        txn: &RoTxn,
        ns: Namespace<'_>,
    ) -> Result<Option<Database<Bytes, Bytes>>, BackendError> {
        let name = ns_name(ns);
        if let Some(db) = self.dbs.read().unwrap().get(&name).copied() {
            return Ok(Some(db));
        }
        if let Some(db) = self
            .env
            .open_database::<Bytes, Bytes>(txn, Some(&name))
            .map_err(io_at("open_database"))?
        {
            return Ok(Some(db));
        }
        self.catalog_sync.request_and_wait();
        Ok(self.dbs.read().unwrap().get(&name).copied())
    }

    fn open_catalog(
        env: &Env<WithTls>,
    ) -> Result<Vec<(String, Database<Bytes, Bytes>)>, BackendError> {
        let mut wtxn = env.write_txn().map_err(io_at("catalog_txn"))?;
        let names: Vec<String> = match env
            .open_database::<Bytes, Bytes>(&wtxn, None)
            .map_err(io_at("catalog_main"))?
        {
            Some(main) => {
                let mut names = Vec::new();
                for item in main.iter(&wtxn).map_err(io_at("catalog_iter"))? {
                    let (key, _) = item.map_err(io_at("catalog_next"))?;
                    let key = key.strip_suffix(&[0u8]).unwrap_or(key);
                    if key.is_empty() || key.contains(&0u8) {
                        continue;
                    }
                    if let Ok(name) = std::str::from_utf8(key) {
                        names.push(name.to_string());
                    }
                }
                names
            }
            None => Vec::new(),
        };
        let mut found = Vec::with_capacity(names.len());
        for name in names {
            if let Some(db) = env
                .open_database::<Bytes, Bytes>(&wtxn, Some(&name))
                .map_err(io_at("catalog_open"))?
            {
                found.push((name, db));
            }
        }
        wtxn.commit().map_err(io_at("catalog_commit"))?;
        Ok(found)
    }

    /// Resolve a namespace to its database for writing, creating it if absent.
    fn resolve_write(
        &self,
        txn: &mut RwTxn,
        ns: Namespace<'_>,
    ) -> Result<Database<Bytes, Bytes>, BackendError> {
        let name = ns_name(ns);
        if let Some(db) = self.dbs.read().unwrap().get(&name).copied() {
            return Ok(db);
        }
        let db = if is_dup(ns) {
            // Multi-value (adjacency / multi-index) namespace. MUST match the
            // flags the struct creates these DBIs with: out_edges/in_edges
            // (storage_core.rs ~1356/1362) and midx_* (~5554) are all
            // DUP_SORT | DUP_FIXED — fixed-width dup values (32-byte adjacency
            // packs, 16-byte index ids) that enable LMDB's MDB_GET_MULTIPLE bulk
            // cursor path. Opening with DUP_SORT alone here would, if this is the
            // first opener of the name, persist the DBI without DUP_FIXED (or
            // raise MDB_INCOMPATIBLE against an existing one), breaking bulk reads.
            self.env
                .database_options()
                .types::<Bytes, Bytes>()
                .flags(DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED)
                .name(&name)
                .create(txn)
                .map_err(io)?
        } else {
            self.env
                .create_database::<Bytes, Bytes>(txn, Some(&name))
                .map_err(io)?
        };
        Ok(db)
    }
}

impl LmdbBackend {
    // ---- raw heed-txn pass-throughs (call-site migration scaffold) ----
    //
    // These route KV access through the backend while the caller still holds the
    // heed txn it already threads (storage/vector methods during US-004..006).
    // They take heed txns directly, so no owned/borrowed handle or lifetime
    // unification is needed — the threaded `&RoTxn` / `&mut RwTxn` is passed
    // straight through. The handle-based trait methods (`get_with`/`put`/...) are
    // the long-term seam used once the threaded txn type is flipped to
    // `AnyRead` / `AnyWrite`; until then these `_raw` methods carry the LMDB path
    // and behave identically (same `resolve_read`/`resolve_write` + heed calls).

    pub(crate) fn get_with_raw<R>(
        &self,
        txn: &RoTxn,
        ns: Namespace<'_>,
        key: &[u8],
        f: impl FnOnce(Option<&[u8]>) -> R,
    ) -> Result<R, BackendError> {
        let value = match self.resolve_read(txn, ns)? {
            Some(db) => db.get(txn, key).map_err(io_at("get_raw"))?,
            None => None,
        };
        Ok(f(value))
    }

    pub(crate) fn scan_raw(
        &self,
        txn: &RoTxn,
        ns: Namespace<'_>,
        range: KeyRange,
        mut visit: impl FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        let db = match self.resolve_read(txn, ns)? {
            Some(db) => db,
            None => return Ok(()),
        };
        let lo: Bound<&[u8]> = match &range.start {
            Bound::Included(v) => Bound::Included(v.as_slice()),
            Bound::Excluded(v) => Bound::Excluded(v.as_slice()),
            Bound::Unbounded => Bound::Unbounded,
        };
        let hi: Bound<&[u8]> = match &range.end {
            Bound::Included(v) => Bound::Included(v.as_slice()),
            Bound::Excluded(v) => Bound::Excluded(v.as_slice()),
            Bound::Unbounded => Bound::Unbounded,
        };
        for item in db.range(txn, &(lo, hi)).map_err(io_at("range_raw"))? {
            let (k, v) = item.map_err(io_at("range_next_raw"))?;
            if !visit(k, v) {
                break;
            }
        }
        Ok(())
    }

    pub(crate) fn for_each_dup_raw(
        &self,
        txn: &RoTxn,
        ns: Namespace<'_>,
        key: &[u8],
        mut visit: impl FnMut(&[u8]) -> bool,
    ) -> Result<(), BackendError> {
        let db = match self.resolve_read(txn, ns)? {
            Some(db) => db,
            None => return Ok(()),
        };
        if let Some(iter) = db.get_duplicates(txn, key).map_err(io_at("dup"))? {
            for item in iter {
                let (_k, v) = item.map_err(io_at("dup_next"))?;
                if !visit(v) {
                    break;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn put_raw(
        &self,
        txn: &mut RwTxn,
        ns: Namespace<'_>,
        key: &[u8],
        val: &[u8],
    ) -> Result<(), BackendError> {
        let db = self.resolve_write(txn, ns)?;
        db.put(txn, key, val).map_err(io)
    }

    pub(crate) fn delete_raw(
        &self,
        txn: &mut RwTxn,
        ns: Namespace<'_>,
        key: &[u8],
    ) -> Result<(), BackendError> {
        let db = self.resolve_write(txn, ns)?;
        db.delete(txn, key).map_err(io)?;
        Ok(())
    }

    /// Put with a [`WriteHint`], preserving LMDB's `APPEND`/`APPEND_DUP` sorted
    /// bulk-insert fast path. Behaves exactly like the existing
    /// `db.put_with_flags(PutFlags::APPEND.., key, val)` call sites.
    pub(crate) fn put_hinted_raw(
        &self,
        txn: &mut RwTxn,
        ns: Namespace<'_>,
        key: &[u8],
        val: &[u8],
        hint: WriteHint,
    ) -> Result<(), BackendError> {
        let db = self.resolve_write(txn, ns)?;
        match hint {
            WriteHint::None => db.put(txn, key, val).map_err(io),
            WriteHint::Append => db
                .put_with_flags(txn, PutFlags::APPEND, key, val)
                .map_err(io),
            WriteHint::AppendDup => db
                .put_with_flags(txn, PutFlags::APPEND_DUP, key, val)
                .map_err(io),
        }
    }

    pub(crate) fn put_dup_raw(
        &self,
        txn: &mut RwTxn,
        ns: Namespace<'_>,
        key: &[u8],
        value: &[u8],
    ) -> Result<(), BackendError> {
        let db = self.resolve_write(txn, ns)?;
        db.put(txn, key, value).map_err(io)?;
        Ok(())
    }

    pub(crate) fn delete_dup_raw(
        &self,
        txn: &mut RwTxn,
        ns: Namespace<'_>,
        key: &[u8],
        value: &[u8],
    ) -> Result<(), BackendError> {
        let db = self.resolve_write(txn, ns)?;
        db.delete_one_duplicate(txn, key, value).map_err(io)?;
        Ok(())
    }

    pub(crate) fn get_for_update_raw<R>(
        &self,
        txn: &RwTxn,
        ns: Namespace<'_>,
        key: &[u8],
        f: impl FnOnce(Option<&[u8]>) -> R,
    ) -> Result<R, BackendError> {
        // Read through the write txn (read-your-writes), mirroring `get_for_update`.
        let name = ns_name(ns);
        let cached = self.dbs.read().unwrap().get(&name).copied();
        let db = match cached {
            Some(db) => Some(db),
            None => self
                .env
                .open_database::<Bytes, Bytes>(txn, Some(&name))
                .map_err(io)?,
        };
        let value = match db {
            Some(db) => db.get(txn, key).map_err(io)?,
            None => None,
        };
        Ok(f(value))
    }
}

impl StorageBackend for LmdbBackend {
    type Read<'s>
        = LmdbRead<'s>
    where
        Self: 's;
    type Write<'s>
        = LmdbWrite<'s>
    where
        Self: 's;

    fn begin_read(&self) -> Result<Self::Read<'_>, BackendError> {
        let txn = self.env.read_txn().map_err(io_at("read_txn"))?;
        Ok(LmdbRead {
            txn: RoHandle::Owned(txn),
        })
    }

    fn begin_write(&self) -> Result<Self::Write<'_>, BackendError> {
        let txn = self.env.write_txn().map_err(io)?;
        Ok(LmdbWrite { txn })
    }

    fn commit(&self, txn: Self::Write<'_>) -> Result<(), BackendError> {
        txn.txn.commit().map_err(io)
    }

    fn get_with<R>(
        &self,
        txn: &Self::Read<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        f: impl FnOnce(Option<&[u8]>) -> R,
    ) -> Result<R, BackendError> {
        let value = match self.resolve_read(txn.ro(), ns)? {
            Some(db) => db.get(txn.ro(), key).map_err(io_at("get"))?,
            None => None,
        };
        Ok(f(value))
    }

    fn scan(
        &self,
        txn: &Self::Read<'_>,
        ns: Namespace<'_>,
        range: KeyRange,
        mut visit: impl FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError> {
        let db = match self.resolve_read(txn.ro(), ns)? {
            Some(db) => db,
            None => return Ok(()),
        };
        let lo: Bound<&[u8]> = match &range.start {
            Bound::Included(v) => Bound::Included(v.as_slice()),
            Bound::Excluded(v) => Bound::Excluded(v.as_slice()),
            Bound::Unbounded => Bound::Unbounded,
        };
        let hi: Bound<&[u8]> = match &range.end {
            Bound::Included(v) => Bound::Included(v.as_slice()),
            Bound::Excluded(v) => Bound::Excluded(v.as_slice()),
            Bound::Unbounded => Bound::Unbounded,
        };
        for item in db.range(txn.ro(), &(lo, hi)).map_err(io_at("range"))? {
            let (k, v) = item.map_err(io_at("range_next"))?;
            if !visit(k, v) {
                break;
            }
        }
        Ok(())
    }

    fn put(
        &self,
        txn: &mut Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        val: &[u8],
    ) -> Result<(), BackendError> {
        let db = self.resolve_write(&mut txn.txn, ns)?;
        db.put(&mut txn.txn, key, val).map_err(io)
    }

    fn delete(
        &self,
        txn: &mut Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
    ) -> Result<(), BackendError> {
        let db = self.resolve_write(&mut txn.txn, ns)?;
        db.delete(&mut txn.txn, key).map_err(io)?;
        Ok(())
    }

    fn put_dup(
        &self,
        txn: &mut Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        value: &[u8],
    ) -> Result<(), BackendError> {
        let db = self.resolve_write(&mut txn.txn, ns)?;
        // On a DUP_SORT db, `put` appends a value to the key's sorted set.
        db.put(&mut txn.txn, key, value).map_err(io)?;
        Ok(())
    }

    fn delete_dup(
        &self,
        txn: &mut Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        value: &[u8],
    ) -> Result<(), BackendError> {
        let db = self.resolve_write(&mut txn.txn, ns)?;
        db.delete_one_duplicate(&mut txn.txn, key, value)
            .map_err(io)?;
        Ok(())
    }

    fn for_each_dup(
        &self,
        txn: &Self::Read<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        mut visit: impl FnMut(&[u8]) -> bool,
    ) -> Result<(), BackendError> {
        let db = match self.resolve_read(txn.ro(), ns)? {
            Some(db) => db,
            None => return Ok(()),
        };
        if let Some(iter) = db.get_duplicates(txn.ro(), key).map_err(io)? {
            for item in iter {
                let (_k, v) = item.map_err(io)?;
                if !visit(v) {
                    break;
                }
            }
        }
        Ok(())
    }

    fn get_for_update<R>(
        &self,
        txn: &Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        f: impl FnOnce(Option<&[u8]>) -> R,
    ) -> Result<R, BackendError> {
        // A heed `RwTxn` reads its own uncommitted writes (read-your-writes).
        // We resolve the db against the write txn directly — `open_database`/`get`
        // accept a `&RwTxn` (generic over TLS), whereas `resolve_read`'s concrete
        // `&RoTxn<WithTls>` param does not.
        let name = ns_name(ns);
        let cached = self.dbs.read().unwrap().get(&name).copied();
        let db = match cached {
            Some(db) => Some(db),
            None => self
                .env
                .open_database::<Bytes, Bytes>(&txn.txn, Some(&name))
                .map_err(io)?,
        };
        let value = match db {
            Some(db) => db.get(&txn.txn, key).map_err(io)?,
            None => None,
        };
        Ok(f(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn open_tmp() -> (TempDir, LmdbBackend) {
        let dir = TempDir::new().unwrap();
        let be = LmdbBackend::open(dir.path(), 64, 10 * 1024 * 1024).unwrap();
        (dir, be)
    }

    fn get_owned(
        be: &LmdbBackend,
        r: &LmdbRead<'_>,
        ns: Namespace<'_>,
        k: &[u8],
    ) -> Option<Vec<u8>> {
        be.get_with(r, ns, k, |v| v.map(|b| b.to_vec())).unwrap()
    }

    #[test]
    fn from_env_shares_underlying_data() {
        use heed3::types::Bytes as B;
        use heed3::{Database, EnvOpenOptions};
        let dir = TempDir::new().unwrap();
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(10 * 1024 * 1024)
                .max_dbs(64)
                .open(dir.path())
                .unwrap()
        };
        // Write directly via raw heed into a "nodes" db.
        {
            let mut w = env.write_txn().unwrap();
            let db: Database<B, B> = env.create_database(&mut w, Some("nodes")).unwrap();
            db.put(&mut w, b"k1".as_slice(), b"v1".as_slice()).unwrap();
            w.commit().unwrap();
        }
        // The backend, wrapping a clone of that env, reads the same data.
        let be = LmdbBackend::from_env(env.clone());
        let r = be.begin_read().unwrap();
        let got = be
            .get_with(&r, Namespace::Nodes, b"k1", |v| v.map(|b| b.to_vec()))
            .unwrap();
        assert_eq!(got, Some(b"v1".to_vec()));
    }

    #[test]
    fn put_get_delete_round_trip() {
        let (_dir, be) = open_tmp();

        let mut w = be.begin_write().unwrap();
        be.put(&mut w, Namespace::Nodes, b"k1", b"v1").unwrap();
        be.put(&mut w, Namespace::Nodes, b"k2", b"v2").unwrap();
        be.commit(w).unwrap();

        let r = be.begin_read().unwrap();
        assert_eq!(
            get_owned(&be, &r, Namespace::Nodes, b"k1"),
            Some(b"v1".to_vec())
        );
        assert_eq!(
            get_owned(&be, &r, Namespace::Nodes, b"k2"),
            Some(b"v2".to_vec())
        );
        assert_eq!(get_owned(&be, &r, Namespace::Nodes, b"missing"), None);
        drop(r);

        let mut w = be.begin_write().unwrap();
        be.delete(&mut w, Namespace::Nodes, b"k1").unwrap();
        be.commit(w).unwrap();

        let r = be.begin_read().unwrap();
        assert_eq!(get_owned(&be, &r, Namespace::Nodes, b"k1"), None);
        assert_eq!(
            get_owned(&be, &r, Namespace::Nodes, b"k2"),
            Some(b"v2".to_vec())
        );
    }

    #[test]
    fn scan_returns_ordered_entries() {
        let (_dir, be) = open_tmp();
        let mut w = be.begin_write().unwrap();
        for k in [b"b", b"a", b"c"] {
            be.put(&mut w, Namespace::Metadata, k, b"x").unwrap();
        }
        be.commit(w).unwrap();

        let r = be.begin_read().unwrap();
        let mut keys = Vec::new();
        be.scan(&r, Namespace::Metadata, KeyRange::all(), |k, _v| {
            keys.push(k.to_vec());
            true
        })
        .unwrap();
        assert_eq!(keys, vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]);
    }

    #[test]
    fn scan_prefix_bounds_the_range() {
        let (_dir, be) = open_tmp();
        let mut w = be.begin_write().unwrap();
        be.put(&mut w, Namespace::Metadata, b"app:1", b"x").unwrap();
        be.put(&mut w, Namespace::Metadata, b"app:2", b"x").unwrap();
        be.put(&mut w, Namespace::Metadata, b"zzz:1", b"x").unwrap();
        be.commit(w).unwrap();

        let r = be.begin_read().unwrap();
        let mut keys = Vec::new();
        be.scan(
            &r,
            Namespace::Metadata,
            KeyRange::prefix(b"app:"),
            |k, _v| {
                keys.push(k.to_vec());
                true
            },
        )
        .unwrap();
        assert_eq!(keys, vec![b"app:1".to_vec(), b"app:2".to_vec()]);
    }

    #[test]
    fn scan_can_stop_early() {
        let (_dir, be) = open_tmp();
        let mut w = be.begin_write().unwrap();
        for k in [b"a", b"b", b"c", b"d"] {
            be.put(&mut w, Namespace::Metadata, k, b"x").unwrap();
        }
        be.commit(w).unwrap();

        let r = be.begin_read().unwrap();
        let mut seen = 0;
        be.scan(&r, Namespace::Metadata, KeyRange::all(), |_k, _v| {
            seen += 1;
            seen < 2
        })
        .unwrap();
        assert_eq!(seen, 2);
    }

    #[test]
    fn dup_namespace_multi_value() {
        let (_dir, be) = open_tmp();
        let mut w = be.begin_write().unwrap();
        be.put_dup(&mut w, Namespace::OutEdges, b"node1", b"edgeA")
            .unwrap();
        be.put_dup(&mut w, Namespace::OutEdges, b"node1", b"edgeB")
            .unwrap();
        be.put_dup(&mut w, Namespace::OutEdges, b"node2", b"edgeC")
            .unwrap();
        be.commit(w).unwrap();

        let r = be.begin_read().unwrap();
        let mut vals = Vec::new();
        be.for_each_dup(&r, Namespace::OutEdges, b"node1", |v| {
            vals.push(v.to_vec());
            true
        })
        .unwrap();
        // DUP_SORT returns values in sorted order, and only node1's values.
        assert_eq!(vals, vec![b"edgeA".to_vec(), b"edgeB".to_vec()]);
        drop(r);

        let mut w = be.begin_write().unwrap();
        be.delete_dup(&mut w, Namespace::OutEdges, b"node1", b"edgeA")
            .unwrap();
        be.commit(w).unwrap();

        let r = be.begin_read().unwrap();
        let mut vals2 = Vec::new();
        be.for_each_dup(&r, Namespace::OutEdges, b"node1", |v| {
            vals2.push(v.to_vec());
            true
        })
        .unwrap();
        assert_eq!(vals2, vec![b"edgeB".to_vec()]);
    }

    #[test]
    fn put_hinted_raw_append_preserves_order_and_enforces_contract() {
        use heed3::EnvOpenOptions;
        let dir = TempDir::new().unwrap();
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(10 * 1024 * 1024)
                .max_dbs(64)
                .open(dir.path())
                .unwrap()
        };
        let be = LmdbBackend::from_env(env.clone());
        {
            let mut w = env.write_txn().unwrap();
            // Plain put, then ascending APPENDs — mirrors add_n/bulk insert order.
            be.put_hinted_raw(
                &mut w,
                Namespace::Nodes,
                &0u128.to_be_bytes(),
                b"z",
                WriteHint::None,
            )
            .unwrap();
            be.put_hinted_raw(
                &mut w,
                Namespace::Nodes,
                &1u128.to_be_bytes(),
                b"a",
                WriteHint::Append,
            )
            .unwrap();
            be.put_hinted_raw(
                &mut w,
                Namespace::Nodes,
                &2u128.to_be_bytes(),
                b"b",
                WriteHint::Append,
            )
            .unwrap();
            w.commit().unwrap();
        }
        {
            let ro = env.read_txn().unwrap();
            let mut keys = Vec::new();
            be.scan_raw(&ro, Namespace::Nodes, KeyRange::all(), |k, _v| {
                keys.push(k.to_vec());
                true
            })
            .unwrap();
            assert_eq!(
                keys,
                vec![
                    0u128.to_be_bytes().to_vec(),
                    1u128.to_be_bytes().to_vec(),
                    2u128.to_be_bytes().to_vec(),
                ]
            );
        }
        // APPEND of a non-max key violates LMDB's ordering contract -> error
        // (proves the hint maps to PutFlags::APPEND, not a plain overwrite).
        {
            let mut w = env.write_txn().unwrap();
            let res = be.put_hinted_raw(
                &mut w,
                Namespace::Nodes,
                &0u128.to_be_bytes(),
                b"x",
                WriteHint::Append,
            );
            assert!(res.is_err(), "APPEND of a non-max key must error");
        }
    }

    #[test]
    fn raw_methods_round_trip_via_threaded_heed_txn() {
        use heed3::EnvOpenOptions;
        let dir = TempDir::new().unwrap();
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(10 * 1024 * 1024)
                .max_dbs(64)
                .open(dir.path())
                .unwrap()
        };
        let be = LmdbBackend::from_env(env.clone());

        // Writes via a caller-held RwTxn, plus read-your-writes via get_for_update_raw.
        {
            let mut w = env.write_txn().unwrap();
            be.put_raw(&mut w, Namespace::Nodes, b"k1", b"v1").unwrap();
            be.put_raw(&mut w, Namespace::Nodes, b"k2", b"v2").unwrap();
            let seen = be
                .get_for_update_raw(&w, Namespace::Nodes, b"k1", |v| v.map(|b| b.to_vec()))
                .unwrap();
            assert_eq!(seen, Some(b"v1".to_vec()));
            w.commit().unwrap();
        }

        // Reads via a caller-held RoTxn.
        {
            let ro = env.read_txn().unwrap();
            let got = be
                .get_with_raw(&ro, Namespace::Nodes, b"k1", |v| v.map(|b| b.to_vec()))
                .unwrap();
            assert_eq!(got, Some(b"v1".to_vec()));
            let mut keys = Vec::new();
            be.scan_raw(&ro, Namespace::Nodes, KeyRange::all(), |k, _v| {
                keys.push(k.to_vec());
                true
            })
            .unwrap();
            assert_eq!(keys, vec![b"k1".to_vec(), b"k2".to_vec()]);
        }

        // Dup namespace via raw.
        {
            let mut w = env.write_txn().unwrap();
            be.put_dup_raw(&mut w, Namespace::OutEdges, b"n1", b"eA")
                .unwrap();
            be.put_dup_raw(&mut w, Namespace::OutEdges, b"n1", b"eB")
                .unwrap();
            w.commit().unwrap();
        }
        {
            let ro = env.read_txn().unwrap();
            let mut vals = Vec::new();
            be.for_each_dup_raw(&ro, Namespace::OutEdges, b"n1", |v| {
                vals.push(v.to_vec());
                true
            })
            .unwrap();
            assert_eq!(vals, vec![b"eA".to_vec(), b"eB".to_vec()]);
        }

        // Deletes via raw (point + dup).
        {
            let mut w = env.write_txn().unwrap();
            be.delete_raw(&mut w, Namespace::Nodes, b"k2").unwrap();
            be.delete_dup_raw(&mut w, Namespace::OutEdges, b"n1", b"eA")
                .unwrap();
            w.commit().unwrap();
        }
        {
            let ro = env.read_txn().unwrap();
            assert_eq!(
                be.get_with_raw(&ro, Namespace::Nodes, b"k2", |v| v.map(|b| b.to_vec()))
                    .unwrap(),
                None
            );
            let mut vals = Vec::new();
            be.for_each_dup_raw(&ro, Namespace::OutEdges, b"n1", |v| {
                vals.push(v.to_vec());
                true
            })
            .unwrap();
            assert_eq!(vals, vec![b"eB".to_vec()]);
        }
    }

    #[test]
    fn missing_namespace_reads_empty() {
        let (_dir, be) = open_tmp();
        let r = be.begin_read().unwrap();
        assert_eq!(get_owned(&be, &r, Namespace::Nodes, b"k1"), None);
        let mut n = 0;
        be.scan(&r, Namespace::Edges, KeyRange::all(), |_k, _v| {
            n += 1;
            true
        })
        .unwrap();
        assert_eq!(n, 0);
    }
}
