//! `StorageBackend` → `StorageBackend` migration helpers (Phase 6).
//!
//! Copies the contents of a logical [`Namespace`] from one backend to another,
//! regardless of the concrete engines on either side (LMDB ↔ LSM ↔ any). This is
//! how a collection is moved between engines (e.g. lifting an existing LMDB
//! collection onto the SlateDB-backed LSM backend) without going through the
//! higher-level graph/vector layers.
//!
//! Strategy: **collect-then-write**. We open a read snapshot on the source, scan
//! the whole namespace into an owned `Vec<(Vec<u8>, Vec<u8>)>` inside the scan
//! visitor (the borrowed slices are only valid for the read's lifetime, so we
//! clone them), then drop the read snapshot and replay every pair through a
//! single destination write batch. Holding a read txn open across a write batch
//! is avoided — important on LMDB where a long-lived reader pins the map and
//! blocks reclamation.
//!
//! Scope note: this copies logical backend keyspaces only. It does not discover
//! or move LMDB dense-vector mmap sidecars (`.hvec`/`.hvs8`/`.hvtq`) or convert
//! prebuilt HNSW segment layout into the LSM backend. Vector cutover needs a
//! separate sidecar/tiering migration before LMDB can be deleted.

#![allow(dead_code)]

use super::backend::{is_dup, BackendError, KeyRange, Namespace, StorageBackend};

/// Copy every `(key, value)` pair in `ns` from `src` to `dst` and return the
/// number of pairs copied.
///
/// Both sides may be any [`StorageBackend`] impl; the source and destination
/// engines need not be the same type.
///
/// Multi-value ([`is_dup`]) namespaces (`OutEdges` / `InEdges` / `MultiIndex`)
/// are copied faithfully — every duplicate value under each key is enumerated
/// via `for_each_dup` and replayed with `put_dup`. Single-value namespaces
/// (`Nodes`, `Edges`, `Metadata`, segment DBs, …) copy via `scan` + `put`.
pub fn copy_namespace<S: StorageBackend, D: StorageBackend>(
    src: &S,
    dst: &D,
    ns: Namespace<'_>,
) -> Result<usize, BackendError> {
    if is_dup(ns) {
        return copy_dup_namespace(src, dst, ns);
    }

    // Phase 1: collect into owned pairs under a scoped read snapshot.
    let mut pairs: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    {
        let r = src.begin_read()?;
        src.scan(&r, ns, KeyRange::all(), |k, v| {
            pairs.push((k.to_vec(), v.to_vec()));
            true // continue scanning
        })?;
    } // read snapshot dropped here, before the write batch opens.

    // Phase 2: replay through one destination write batch.
    let mut w = dst.begin_write()?;
    for (k, v) in &pairs {
        dst.put(&mut w, ns, k, v)?;
    }
    dst.commit(w)?;

    Ok(pairs.len())
}

/// Faithful copy of a `DUP_SORT` (multi-value) namespace: collect the distinct
/// keys, enumerate ALL duplicate values under each via `for_each_dup`, then
/// replay each pair with `put_dup`. `scan`'s behavior on a dup DB is engine-
/// dependent (heed yields every dup as a separate entry), so we dedupe keys and
/// re-enumerate to be correct regardless of engine.
fn copy_dup_namespace<S: StorageBackend, D: StorageBackend>(
    src: &S,
    dst: &D,
    ns: Namespace<'_>,
) -> Result<usize, BackendError> {
    // Phase 1a: distinct keys (scan may surface a key once or once-per-dup; the
    // last-key dedupe handles both since scan order is key-ascending).
    let mut keys: Vec<Vec<u8>> = Vec::new();
    {
        let r = src.begin_read()?;
        let mut last: Option<Vec<u8>> = None;
        src.scan(&r, ns, KeyRange::all(), |k, _v| {
            if last.as_deref() != Some(k) {
                keys.push(k.to_vec());
                last = Some(k.to_vec());
            }
            true
        })?;
    }

    // Phase 1b: every dup value under each key.
    let mut pairs: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    {
        let r = src.begin_read()?;
        for k in &keys {
            src.for_each_dup(&r, ns, k, |v| {
                pairs.push((k.clone(), v.to_vec()));
                true
            })?;
        }
    }

    // Phase 2: replay through one destination write batch.
    let mut w = dst.begin_write()?;
    for (k, v) in &pairs {
        dst.put_dup(&mut w, ns, k, v)?;
    }
    dst.commit(w)?;

    Ok(pairs.len())
}

/// Copy each namespace in `namespaces` from `src` to `dst`, returning the total
/// number of pairs copied across all of them.
pub fn copy_namespaces<S: StorageBackend, D: StorageBackend>(
    src: &S,
    dst: &D,
    namespaces: &[Namespace<'_>],
) -> Result<usize, BackendError> {
    let mut total = 0;
    for &ns in namespaces {
        total += copy_namespace(src, dst, ns)?;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::super::backend::{Namespace, StorageBackend};
    use super::super::backend_lmdb::LmdbBackend;
    use super::super::backend_lsm::LsmBackend;
    use super::copy_namespace;
    use std::time::Instant;
    use tempfile::TempDir;

    /// 64 named DBs is plenty for a single-namespace test; 10 MiB map size.
    const MAX_DBS: u32 = 64;
    const MAP_SIZE: usize = 10 * 1024 * 1024;

    #[test]
    fn copy_namespace_lmdb_to_lsm() {
        // Source: LMDB with ~5 keys under Nodes.
        let dir = TempDir::new().unwrap();
        let lmdb = LmdbBackend::open(dir.path(), MAX_DBS, MAP_SIZE).unwrap();

        let originals: Vec<(Vec<u8>, Vec<u8>)> = (0..5)
            .map(|i| {
                (
                    format!("node:{i}").into_bytes(),
                    format!("payload-{i}").into_bytes(),
                )
            })
            .collect();

        let mut w = lmdb.begin_write().unwrap();
        for (k, v) in &originals {
            lmdb.put(&mut w, Namespace::Nodes, k, v).unwrap();
        }
        lmdb.commit(w).unwrap();

        // Destination: in-memory LSM.
        let lsm = LsmBackend::open_in_memory("/mig").unwrap();

        let count = copy_namespace(&lmdb, &lsm, Namespace::Nodes).unwrap();
        assert_eq!(count, 5, "expected to copy all 5 Nodes keys");

        // Read each key back out of the LSM backend and compare values.
        let r = lsm.begin_read().unwrap();
        for (k, expected) in &originals {
            let got = lsm
                .get_with(&r, Namespace::Nodes, k, |v| v.map(|b| b.to_vec()))
                .unwrap();
            assert_eq!(
                got.as_deref(),
                Some(expected.as_slice()),
                "value mismatch for key {k:?} after migration",
            );
        }
    }

    #[test]
    fn copy_dup_namespace_lmdb_to_lsm_preserves_all_duplicates() {
        // Adjacency-shaped: two keys, multiple sorted dup values each
        // (DUP_SORT). node1 -> {edgeA, edgeB, edgeC}; node2 -> {edgeD}.
        let dir = TempDir::new().unwrap();
        let lmdb = LmdbBackend::open(dir.path(), MAX_DBS, MAP_SIZE).unwrap();
        let mut w = lmdb.begin_write().unwrap();
        for v in [b"edgeA".as_slice(), b"edgeB", b"edgeC"] {
            lmdb.put_dup(&mut w, Namespace::OutEdges, b"node1", v)
                .unwrap();
        }
        lmdb.put_dup(&mut w, Namespace::OutEdges, b"node2", b"edgeD")
            .unwrap();
        lmdb.commit(w).unwrap();

        // Copy the dup namespace to an in-memory LSM backend.
        let lsm = LsmBackend::open_in_memory("/mig-dup").unwrap();
        let count = copy_namespace(&lmdb, &lsm, Namespace::OutEdges).unwrap();
        assert_eq!(count, 4, "expected all 4 (key,dup-value) pairs copied");

        // Every duplicate must survive on the LSM side, in sorted order.
        let r = lsm.begin_read().unwrap();
        let mut node1 = Vec::new();
        lsm.for_each_dup(&r, Namespace::OutEdges, b"node1", |v| {
            node1.push(v.to_vec());
            true
        })
        .unwrap();
        assert_eq!(
            node1,
            vec![b"edgeA".to_vec(), b"edgeB".to_vec(), b"edgeC".to_vec()],
            "all duplicates under node1 must migrate, sorted"
        );
        let mut node2 = Vec::new();
        lsm.for_each_dup(&r, Namespace::OutEdges, b"node2", |v| {
            node2.push(v.to_vec());
            true
        })
        .unwrap();
        assert_eq!(node2, vec![b"edgeD".to_vec()]);
    }

    #[test]
    fn migration_bench_smoke() {
        const N: usize = 500;

        let pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..N)
            .map(|i| {
                (
                    format!("k:{i:08}").into_bytes(),
                    format!("val-{i}").into_bytes(),
                )
            })
            .collect();

        // LMDB insert timing.
        let dir = TempDir::new().unwrap();
        let lmdb = LmdbBackend::open(dir.path(), MAX_DBS, MAP_SIZE).unwrap();
        let t0 = Instant::now();
        {
            let mut w = lmdb.begin_write().unwrap();
            for (k, v) in &pairs {
                lmdb.put(&mut w, Namespace::Nodes, k, v).unwrap();
            }
            lmdb.commit(w).unwrap();
        }
        let lmdb_dur = t0.elapsed();

        // LSM insert timing.
        let lsm = LsmBackend::open_in_memory("/bench").unwrap();
        let t1 = Instant::now();
        {
            let mut w = lsm.begin_write().unwrap();
            for (k, v) in &pairs {
                lsm.put(&mut w, Namespace::Nodes, k, v).unwrap();
            }
            lsm.commit(w).unwrap();
        }
        let lsm_dur = t1.elapsed();

        // Informational only — no timing assertions (machine-dependent).
        println!("migration_bench_smoke: inserted {N} keys");
        println!("  LMDB: {lmdb_dur:?}");
        println!("  LSM:  {lsm_dur:?}");
    }
}
