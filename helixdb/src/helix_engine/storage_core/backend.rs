//! `StorageBackend` — pluggable storage abstraction (Phase 1 scaffold).
//!
//! Lets Helion run on either the existing LMDB engine (`heed3`) or a future
//! SlateDB-backed LSM-on-object-storage engine, selected at runtime via
//! `HELIX_STORAGE_BACKEND` (`lmdb` default | `lsm`). This module defines the
//! seam only; the LMDB impl wraps the existing `HelixGraphStorage` txn helpers
//! (`with_read_txn` / `with_write_txn` / `begin_resize_safe_*`), and call sites
//! migrate onto it incrementally so the build stays green at every step.
//!
//! Design notes (from the seam census):
//! - heed returns borrowed `&[u8]` valid for the read txn; returning owned
//!   `Vec<u8>` everywhere would regress the hot read path. We keep zero-copy by
//!   exposing reads/scans through *visitor closures* (`get_with` / `scan`) whose
//!   borrows are scoped to the callback — mirroring the existing
//!   `with_vec_slice` / `with_read_txn` patterns — rather than a lending
//!   iterator / GAT-iterator. SlateDB (owned `Bytes`) satisfies the same
//!   visitor trivially.
//! - Logical keyspaces (heed named DBs, the 5 per-dense-segment DBs) are
//!   addressed by [`Namespace`] instead of concrete `heed3::Database<..>`
//!   handles. The LMDB backend resolves a `Namespace` to a `Database`; the LSM
//!   backend resolves it to a key prefix in SlateDB's single flat keyspace.
//! - This trait is intentionally NOT object-safe (GATs + generic methods).
//!   Runtime backend selection is via an enum (`Lmdb(..) | Lsm(..)`) that
//!   matches and dispatches, not `dyn StorageBackend`.

#![allow(dead_code)]

use std::ops::Bound;

/// Error surfaced by a storage backend. Will gain `From<heed3::Error>` /
/// `From<GraphError>` conversions as call sites migrate; kept standalone in the
/// scaffold to keep the seam decoupled from the engine error types.
#[derive(Debug, Clone)]
pub enum BackendError {
    NotFound,
    /// Optimistic-concurrency / CAS conflict or epoch-fencing on the LSM backend
    /// (a newer writer took the manifest, or a conditional write lost). Carries
    /// the underlying SlateDB message so the write layer can detect split-brain /
    /// failover. LMDB never raises this.
    Conflict(String),
    /// The selected backend cannot safely serve this operation yet.
    Unsupported(String),
    Io(String),
    Corruption(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::NotFound => write!(f, "not found"),
            BackendError::Conflict(m) => write!(f, "write conflict: {m}"),
            BackendError::Unsupported(m) => write!(f, "unsupported operation: {m}"),
            BackendError::Io(m) => write!(f, "io error: {m}"),
            BackendError::Corruption(m) => write!(f, "corruption: {m}"),
        }
    }
}

impl std::error::Error for BackendError {}

/// The LMDB sub-databases that make up one dense vector segment
/// (`vectors_*`, `vector_data_*`, `hnsw_out_*`, `hnsw_neighbors_*`,
/// `vector_ordinals_*`, plus the opt-in IVF posting-list index
/// `ivf_centroids_*` / `ivf_postings_*`). On the LSM backend these collapse
/// to key prefixes under a single segment keyspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentDb {
    Vectors,
    VectorData,
    HnswOut,
    HnswNeighbors,
    Ordinals,
    /// IVF centroid table + build metadata (plain KV, write-once blobs).
    IvfCentroids,
    /// IVF posting lists keyed by u32-BE centroid id (plain KV, write-once
    /// blobs — intentionally NOT dup-sorted to avoid DUP-flag pitfalls).
    IvfPostings,
    /// Per-vector SimHash rows (`id -> [simhash u64 BE][order_code u64 BE]`),
    /// written flag-gated (`HELIX_VECTOR_SIMHASH`) for NEWLY inserted vectors
    /// on the LSM backend. Storage foundation for LSM-locality vector layout
    /// and Hamming-prefix probing (see `vector_core::simhash`); nothing reads
    /// it on the search path yet. Absent for vectors written before the flag
    /// was enabled — readers must treat a missing row as "no code".
    SimHash,
}

/// The three LMDB sub-databases that make up one sparse vector space
/// (`sparse_inv_*`, `sparse_fwd_*`, `sparse_meta_*`). `Inv` is the DUP_SORT
/// inverted index (term_id → posting entries); `Fwd` is the forward index
/// (doc_id → terms); `Meta` holds doc_count + per-term df/max. On the LSM
/// backend these collapse to key prefixes under a single sparse keyspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SparseDb {
    Inv,
    Fwd,
    Meta,
}

/// A logical keyspace. Replaces direct `heed3::Database<..>` handles and the
/// per-segment named DBs. Dynamic names are borrowed from the caller so the
/// enum stays `Copy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Namespace<'a> {
    Nodes,
    Edges,
    OutEdges,
    InEdges,
    EdgePathIdx,
    Metadata,
    SecondaryIndex(&'a str),
    MultiIndex(&'a str),
    PayloadIndex(&'a str),
    /// Build-delta ("dirty row") sidecar for an in-flight ONLINE payload-index
    /// build on the LSM backend, addressed by the same physical index db name
    /// as [`Namespace::PayloadIndex`]. While an index is Building, foreground
    /// node mutations record `(node_id, [seq][old_index_key])` rows (an empty
    /// key tail is the "reindex me" marker) in the same write batch as the
    /// node row; the builder drains them after its scan so a backfill chunk
    /// that raced a foreground write cannot resurrect stale index rows. The
    /// per-process seq makes each row unique, so a drain pass consumes only
    /// rows its own snapshot observed and concurrent writes always leave
    /// evidence that forces another pass; replay order is irrelevant. Unused
    /// on LMDB (builds there hold the write gate, so the race does not exist).
    PayloadIndexBuild(&'a str),
    DenseTombstones,
    /// One sub-DB of a dense vector segment, addressed by physical segment name.
    Segment {
        physical_name: &'a str,
        db: SegmentDb,
    },
    /// One sub-DB of a sparse vector space, addressed by physical sparse-space name.
    SparseSegment {
        physical_name: &'a str,
        db: SparseDb,
    },
}

/// A key range for ordered scans. A prefix scan is expressed as
/// `[prefix, prefix++)`; helpers will build these from a prefix.
pub struct KeyRange {
    pub start: Bound<Vec<u8>>,
    pub end: Bound<Vec<u8>>,
}

impl KeyRange {
    /// Full keyspace.
    pub fn all() -> Self {
        Self {
            start: Bound::Unbounded,
            end: Bound::Unbounded,
        }
    }

    /// Half-open range covering every key starting with `prefix`.
    pub fn prefix(prefix: &[u8]) -> Self {
        let start = Bound::Included(prefix.to_vec());
        match next_prefix(prefix) {
            Some(end) => Self {
                start,
                end: Bound::Excluded(end),
            },
            // All-0xFF prefix: no successor, scan to the end.
            None => Self {
                start,
                end: Bound::Unbounded,
            },
        }
    }
}

/// Smallest byte string strictly greater than every key with `prefix`
/// (increment the last non-0xFF byte, drop the trailing 0xFFs). `None` when
/// `prefix` is all `0xFF` (no finite successor). Shared with the LSM backend.
pub(crate) fn next_prefix(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut end = prefix.to_vec();
    while let Some(last) = end.last_mut() {
        if *last != 0xFF {
            *last += 1;
            return Some(end);
        }
        end.pop();
    }
    None
}

/// Whether a namespace stores multiple values per key (`DUP_SORT` on LMDB:
/// adjacency `out_edges`/`in_edges` and multi-value secondary indices).
pub(crate) fn is_dup(ns: Namespace<'_>) -> bool {
    matches!(
        ns,
        Namespace::OutEdges
            | Namespace::InEdges
            | Namespace::MultiIndex(_)
            | Namespace::PayloadIndex(_)
            | Namespace::PayloadIndexBuild(_)
            | Namespace::SparseSegment {
                db: SparseDb::Inv,
                ..
            }
    )
}

/// Backend-neutral hint for a write. Preserves the sorted bulk-insert fast paths
/// the engine relies on (`add_n`/`bulk_add_*` insert ids in ascending order):
/// the LMDB backend maps these to `heed3::PutFlags` (`APPEND` / `APPEND_DUP`),
/// which skip the B-tree search and append at the rightmost leaf. The LSM
/// backend ignores the hint (its memtable insert is already an append), so the
/// abstraction stays engine-neutral while keeping LMDB's optimization.
///
/// Note: `Append`/`AppendDup` carry LMDB's strict-ordering contract — heed
/// returns `MDB_KEYEXIST` if the key/value is not greater than the current max.
/// Use them only where the existing `put_with_flags(APPEND..)` call sites do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteHint {
    /// Ordinary put (search + insert/overwrite).
    None,
    /// Key is greater than every existing key (sorted append).
    Append,
    /// Value is greater than every existing duplicate under `key` (sorted
    /// append within a `DUP_SORT` key).
    AppendDup,
}

/// Read-only, point-in-time snapshot. LMDB: `RoTxn`. SlateDB: `DbSnapshot`/reader.
pub trait ReadTxn {}

/// A write transaction / batch committed atomically. LMDB: `RwTxn`.
/// SlateDB: `WriteBatch`.
pub trait WriteTxn {}

/// The storage seam both LMDB and the SlateDB-backed LSM engine implement.
///
/// Reads are visitor-based to preserve LMDB zero-copy (the value/key slices are
/// only valid inside the closure). Writes go through an explicit batch that is
/// `commit`ted, matching both `heed3::RwTxn` and SlateDB `WriteBatch`.
pub trait StorageBackend: Send + Sync {
    type Read<'s>: ReadTxn
    where
        Self: 's;
    type Write<'s>: WriteTxn
    where
        Self: 's;

    /// Begin a read snapshot (resize-safe on LMDB).
    fn begin_read(&self) -> Result<Self::Read<'_>, BackendError>;

    /// Begin a write batch/transaction.
    fn begin_write(&self) -> Result<Self::Write<'_>, BackendError>;

    /// Commit a write batch atomically. May return [`BackendError::Conflict`]
    /// on the LSM backend (CAS retry); never on LMDB.
    fn commit(&self, txn: Self::Write<'_>) -> Result<(), BackendError>;

    /// Commit a write batch atomically and make it VISIBLE to subsequent reads,
    /// WITHOUT waiting for it to become durable on the object store. Pair with
    /// [`StorageBackend::flush_durable`] to provide a single durability barrier
    /// over many buffered commits (group-commit): N logical commits then ONE
    /// durable flush, instead of N durable flushes.
    ///
    /// Atomicity per call is unchanged. The ONLY relaxation vs [`commit`] is that
    /// durability is deferred to the next `flush_durable`. Callers MUST NOT ack a
    /// write to a client until a later `flush_durable` succeeds.
    ///
    /// Default = [`commit`] (fully durable) — correct for backends like LMDB
    /// whose commits are already durable (msync per txn) and where deferral has
    /// no benefit.
    fn commit_buffered(&self, txn: Self::Write<'_>) -> Result<(), BackendError> {
        self.commit(txn)
    }

    /// Durability barrier: block until every prior [`commit_buffered`] on this
    /// backend is durably persisted to the object store. Returning `Ok` means all
    /// previously-buffered writes are durable (safe to ack). On the LSM backend
    /// this is one object-store flush amortized across all buffered commits.
    ///
    /// Default = no-op: backends whose [`commit`] is already durable (LMDB) have
    /// nothing to flush.
    fn flush_durable(&self) -> Result<(), BackendError> {
        Ok(())
    }

    /// Zero-copy point read: the value is handed to `f` borrowed for the read's
    /// lifetime (`None` if the key is absent).
    fn get_with<R>(
        &self,
        txn: &Self::Read<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        f: impl FnOnce(Option<&[u8]>) -> R,
    ) -> Result<R, BackendError>;

    /// Ordered scan over `range`; `visit(key, value)` is called per entry with
    /// slices borrowed for the read's lifetime. Return `false` to stop early.
    fn scan(
        &self,
        txn: &Self::Read<'_>,
        ns: Namespace<'_>,
        range: KeyRange,
        visit: impl FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<(), BackendError>;

    fn put(
        &self,
        txn: &mut Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        val: &[u8],
    ) -> Result<(), BackendError>;

    fn delete(
        &self,
        txn: &mut Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
    ) -> Result<(), BackendError>;

    fn merge(
        &self,
        _txn: &mut Self::Write<'_>,
        _ns: Namespace<'_>,
        _key: &[u8],
        _val: &[u8],
    ) -> Result<(), BackendError> {
        Err(BackendError::Unsupported(
            "backend merge operation unsupported".to_string(),
        ))
    }

    /// [`StorageBackend::merge`] whose operands the CALLER guarantees are
    /// commutative (order-independent AND concurrency-safe, e.g. counter
    /// deltas). Semantically this classifies the key so a conflict-checked
    /// write path (SlateDB `DbTransaction::merge_commutative` in the vendored
    /// fork) can skip SSI write-write conflicts on it. Today Helion commits
    /// through non-transactional `WriteBatch`es, which never run conflict
    /// checks, so this default simply delegates to [`StorageBackend::merge`];
    /// call sites use it to carry the commutativity contract forward to a
    /// future transactional write path.
    fn merge_commutative(
        &self,
        txn: &mut Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        val: &[u8],
    ) -> Result<(), BackendError> {
        self.merge(txn, ns, key, val)
    }

    /// Append `value` under `key` in a multi-value ([`is_dup`]) namespace.
    fn put_dup(
        &self,
        txn: &mut Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        value: &[u8],
    ) -> Result<(), BackendError>;

    /// Remove a specific `(key, value)` pair from a multi-value namespace.
    fn delete_dup(
        &self,
        txn: &mut Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        value: &[u8],
    ) -> Result<(), BackendError>;

    /// Visit every value stored under `key` in a multi-value namespace, in
    /// sorted order. Return `false` from `visit` to stop early.
    fn for_each_dup(
        &self,
        txn: &Self::Read<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        visit: impl FnMut(&[u8]) -> bool,
    ) -> Result<(), BackendError>;

    /// Read a key through the WRITE handle, observing this batch's own buffered
    /// writes (read-your-writes) on top of committed state. Required for correct
    /// read-modify-write within a multi-op batch (e.g. metadata counters): a
    /// plain `begin_read()` snapshot would not see earlier writes in the same
    /// batch and would mis-count.
    fn get_for_update<R>(
        &self,
        txn: &Self::Write<'_>,
        ns: Namespace<'_>,
        key: &[u8],
        f: impl FnOnce(Option<&[u8]>) -> R,
    ) -> Result<R, BackendError>;
}

/// Which storage engine backs Helion, selected at runtime via
/// [`BackendKind::ENV_VAR`] (`lmdb` default | `lsm`). The LSM arm is wired to
/// the SlateDB-backed engine in a later phase; until then `lsm` parses and is
/// recorded, and the dispatch site falls back to LMDB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendKind {
    #[default]
    Lmdb,
    Lsm,
}

impl BackendKind {
    /// Environment variable that selects the storage backend.
    pub const ENV_VAR: &'static str = "HELIX_STORAGE_BACKEND";

    /// Resolve the backend from the process environment (defaults to LMDB).
    pub fn from_env() -> Self {
        match std::env::var(Self::ENV_VAR) {
            Ok(raw) => Self::parse(&raw),
            Err(_) => Self::Lmdb,
        }
    }

    /// Parse a backend selector. Unknown values warn and fall back to LMDB so a
    /// typo never silently switches engines.
    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "lsm" | "slatedb" => Self::Lsm,
            "" | "lmdb" => Self::Lmdb,
            other => {
                tracing::warn!(
                    value = other,
                    "unknown {}; defaulting to lmdb",
                    Self::ENV_VAR
                );
                Self::Lmdb
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Bound;

    #[test]
    fn backend_kind_parses_and_defaults() {
        assert_eq!(BackendKind::parse("lsm"), BackendKind::Lsm);
        assert_eq!(BackendKind::parse("slatedb"), BackendKind::Lsm);
        assert_eq!(BackendKind::parse("  LMDB "), BackendKind::Lmdb);
        assert_eq!(BackendKind::parse(""), BackendKind::Lmdb);
        assert_eq!(BackendKind::parse("nonsense"), BackendKind::Lmdb);
        assert_eq!(BackendKind::default(), BackendKind::Lmdb);
    }

    #[test]
    fn prefix_range_increments_last_byte() {
        let r = KeyRange::prefix(b"abc");
        assert_eq!(r.start, Bound::Included(b"abc".to_vec()));
        assert_eq!(r.end, Bound::Excluded(b"abd".to_vec()));
    }

    #[test]
    fn prefix_range_rolls_over_trailing_0xff() {
        let r = KeyRange::prefix(&[0x01, 0xFF]);
        assert_eq!(r.start, Bound::Included(vec![0x01, 0xFF]));
        assert_eq!(r.end, Bound::Excluded(vec![0x02]));
    }

    #[test]
    fn all_0xff_prefix_has_no_upper_bound() {
        let r = KeyRange::prefix(&[0xFF, 0xFF]);
        assert_eq!(r.end, Bound::Unbounded);
    }
}
