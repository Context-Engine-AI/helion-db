//! Backend-routed adjacency (out/in-edge) building blocks for the migration
//! (US-004). The adjacency databases are DUP_SORT (one node key -> many edge
//! IDs), so these route through `self.backend.for_each_dup(...)`. They mirror the
//! heed traversal reads byte-for-byte.
//!
//! ## On-disk encoding (verified against `storage_core.rs` write path and the
//! ## `graph_core/ops/out|in_` read path)
//!
//! `out_edges` / `in_edges` are LMDB `DUP_SORT | DUP_FIXED` databases. One key
//! holds many sorted values, so `for_each_dup` (which calls `get_duplicates` on
//! the exact key) visits exactly the same set the heed traversals do via
//! `get_duplicates(&prefix)`.
//!
//! KEY (20 bytes), built by `HelixGraphStorage::out_edge_key` /
//! `in_edge_key`:
//!   * `[0..16]`  node id, big-endian `u128` (from-node for out, to-node for in)
//!   * `[16..20]` `hash_label(edge_label, None)` (`[u8; 4]`)
//!
//! VALUE (32 bytes), built by `HelixGraphStorage::pack_edge_data`:
//!   * `[0..16]`  edge id, big-endian `u128`
//!   * `[16..32]` peer node id, big-endian `u128` (to-node for out, from-node
//!     for in)
//!
//! `unpack_adj_edge_data` decodes a value to `(peer_node_id, edge_id)`.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};

use heed3::RoTxn;

use crate::helix_engine::types::GraphError;
use crate::protocol::label_hash::hash_label;

use super::backend::{BackendKind, KeyRange, Namespace, StorageBackend};
use super::backend_any::{AnyRead, AnyWrite};
use super::storage_core::HelixGraphStorage;

#[cfg(test)]
thread_local! {
    /// Test-only: what this thread's next orphan check inside a write batch
    /// runs once before it reads, standing in for another writer whose
    /// commit lands after the batch has staged its edge drops.
    static BEFORE_ORPHAN_CHECK_FOR_TEST: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

/// Test-only: has the current thread's next
/// [`HelixGraphStorage::nodes_orphaned_by_dropping_be`] run `other_writer`
/// before it reads.
#[cfg(test)]
pub(crate) fn before_next_orphan_check_for_test(other_writer: impl FnOnce() + 'static) {
    BEFORE_ORPHAN_CHECK_FOR_TEST.with(|hook| *hook.borrow_mut() = Some(Box::new(other_writer)));
}

impl HelixGraphStorage {
    /// Backend-routed enumeration of outgoing edge IDs for `from_node` under
    /// `edge_label`. Byte-identical to the heed `out_e` traversal: it reads the
    /// `out_edges` DUP_SORT database with the exact 20-byte `out_edge_key` and
    /// decodes each 32-byte value's leading edge id. Values come back in LMDB
    /// dup-sorted order, matching the heed iterator.
    pub fn out_edges_be(
        &self,
        r: &AnyRead<'_>,
        from_node: &u128,
        edge_label: &str,
    ) -> Result<Vec<u128>, GraphError> {
        let label_hash = hash_label(edge_label, None);
        let key = Self::out_edge_key(from_node, &label_hash);
        self.collect_adj_edge_ids(r, Namespace::OutEdges, &key)
    }

    /// Backend-routed enumeration of incoming edge IDs for `to_node` under
    /// `edge_label`. Byte-identical to the heed `in_e` traversal: reads the
    /// `in_edges` DUP_SORT database with the exact 20-byte `in_edge_key`.
    pub fn in_edges_be(
        &self,
        r: &AnyRead<'_>,
        to_node: &u128,
        edge_label: &str,
    ) -> Result<Vec<u128>, GraphError> {
        let label_hash = hash_label(edge_label, None);
        let key = Self::in_edge_key(to_node, &label_hash);
        self.collect_adj_edge_ids(r, Namespace::InEdges, &key)
    }

    /// Backend-routed enumeration of `(peer_node_id, edge_id)` for outgoing
    /// edges of `from_node` under `edge_label`. The peer is the to-node. Mirrors
    /// the heed `out` traversal, which decodes the peer node id from each value.
    pub fn out_adj_be(
        &self,
        r: &AnyRead<'_>,
        from_node: &u128,
        edge_label: &str,
    ) -> Result<Vec<(u128, u128)>, GraphError> {
        let label_hash = hash_label(edge_label, None);
        let key = Self::out_edge_key(from_node, &label_hash);
        self.collect_adj_pairs(r, Namespace::OutEdges, &key)
    }

    /// Backend-routed enumeration of `(peer_node_id, edge_id)` for incoming
    /// edges of `to_node` under `edge_label`. The peer is the from-node.
    pub fn in_adj_be(
        &self,
        r: &AnyRead<'_>,
        to_node: &u128,
        edge_label: &str,
    ) -> Result<Vec<(u128, u128)>, GraphError> {
        let label_hash = hash_label(edge_label, None);
        let key = Self::in_edge_key(to_node, &label_hash);
        self.collect_adj_pairs(r, Namespace::InEdges, &key)
    }

    /// Visit every adjacency value under `key` and collect the decoded edge ids.
    /// Decode failures abort with the first error, matching the heed path's
    /// `unpack_adj_edge_data(...)?`.
    fn collect_adj_edge_ids(
        &self,
        r: &AnyRead<'_>,
        ns: Namespace<'_>,
        key: &[u8],
    ) -> Result<Vec<u128>, GraphError> {
        let mut ids = Vec::new();
        let mut decode_err: Option<GraphError> = None;
        self.backend
            .for_each_dup(r, ns, key, |value| {
                match Self::unpack_adj_edge_data(value) {
                    Ok((_peer, edge_id)) => {
                        ids.push(edge_id);
                        true
                    }
                    Err(e) => {
                        decode_err = Some(e);
                        false
                    }
                }
            })
            .map_err(|e| GraphError::New(e.to_string()))?;
        match decode_err {
            Some(e) => Err(e),
            None => Ok(ids),
        }
    }

    /// Visit every adjacency value under `key` and collect `(peer_node, edge_id)`.
    fn collect_adj_pairs(
        &self,
        r: &AnyRead<'_>,
        ns: Namespace<'_>,
        key: &[u8],
    ) -> Result<Vec<(u128, u128)>, GraphError> {
        let mut pairs = Vec::new();
        let mut decode_err: Option<GraphError> = None;
        self.backend
            .for_each_dup(r, ns, key, |value| {
                match Self::unpack_adj_edge_data(value) {
                    Ok((peer, edge_id)) => {
                        pairs.push((peer, edge_id));
                        true
                    }
                    Err(e) => {
                        decode_err = Some(e);
                        false
                    }
                }
            })
            .map_err(|e| GraphError::New(e.to_string()))?;
        match decode_err {
            Some(e) => Err(e),
            None => Ok(pairs),
        }
    }

    /// Backend-routed adjacency read returning `(peer_node_id, edge_id)` pairs for
    /// `node_id` under `label_hash`, in dup-sorted order — the routing front door
    /// for graph traversals (bfs / cycles / algorithms / subgraph).
    ///
    /// On the **LMDB** backend this iterates the caller's heed txn cursor,
    /// byte-identical to the prior `out_edges_db.get_duplicates(txn, &key)` path
    /// (same read snapshot, same order, same `>= 32` value guard). On the **LSM**
    /// backend it opens a fresh backend read snapshot and reads the same DUP
    /// namespace through the seam (`collect_adj_pairs`), so adjacency persisted to
    /// SlateDB/S3 is visible and the identical code path works on a reader replica
    /// (which has no local `graph_env`). Opening a per-call snapshot on LSM matches
    /// the existing `get_with_heed` read model used by `get_node`/`get_edge`.
    ///
    /// `out == true` reads OutEdges (peer = to-node); `out == false` reads InEdges
    /// (peer = from-node).
    pub fn adjacency_pairs(
        &self,
        txn: &RoTxn,
        node_id: u128,
        label_hash: &[u8; 4],
        out: bool,
    ) -> Result<Vec<(u128, u128)>, GraphError> {
        if self.backend.kind() == BackendKind::Lsm {
            let r = self
                .backend
                .begin_read()
                .map_err(|e| GraphError::New(e.to_string()))?;
            return self.adjacency_pairs_be(&r, node_id, label_hash, out);
        }

        let (key, db) = if out {
            (Self::out_edge_key(&node_id, label_hash), &self.out_edges_db)
        } else {
            (Self::in_edge_key(&node_id, label_hash), &self.in_edges_db)
        };
        let db = db.ok_or_else(|| {
            GraphError::StorageError(
                "LMDB adjacency DB handle is unavailable on this backend".to_string(),
            )
        })?;
        let mut pairs = Vec::new();
        if let Some(dup_iter) = db.get_duplicates(txn, &key)? {
            for item in dup_iter {
                let (_, val) = item?;
                if val.len() >= 32 {
                    pairs.push(Self::unpack_adj_edge_data(val)?);
                }
            }
        }
        Ok(pairs)
    }

    /// Backend-routed adjacency read returning `(peer_node_id, edge_id)` pairs for
    /// `node_id` under `label_hash`, using the caller's shared [`AnyRead`]
    /// snapshot. This is the zero-LMDB twin used by native graph API traversals.
    pub fn adjacency_pairs_be(
        &self,
        r: &AnyRead<'_>,
        node_id: u128,
        label_hash: &[u8; 4],
        out: bool,
    ) -> Result<Vec<(u128, u128)>, GraphError> {
        let (ns, key) = if out {
            (
                Namespace::OutEdges,
                Self::out_edge_key(&node_id, label_hash),
            )
        } else {
            (Namespace::InEdges, Self::in_edge_key(&node_id, label_hash))
        };
        self.collect_adj_pairs(r, ns, &key)
    }

    /// `true` if `node_id` has at least one out (or in) adjacency row across ALL
    /// edge labels — a prefix existence check over the 16-byte node id. Routed so
    /// orphan-node cleanup stays correct after the edge-write fix: LMDB uses the
    /// heed `prefix_iter` on the caller txn (byte-identical to the prior check);
    /// LSM scans the DUP namespace prefix through a fresh backend snapshot (which
    /// sees edge deletes already committed by `drop_edge_be`). Without this, a node
    /// whose edges live in SlateDB would read as edge-less on LSM and be wrongly
    /// orphan-deleted.
    pub fn node_has_adjacency_be(
        &self,
        r: &AnyRead<'_>,
        node_id: u128,
        out: bool,
    ) -> Result<bool, GraphError> {
        let ns = if out {
            Namespace::OutEdges
        } else {
            Namespace::InEdges
        };
        let mut found = false;
        self.backend
            .scan(r, ns, KeyRange::prefix(&node_id.to_be_bytes()), |_k, _v| {
                found = true;
                false
            })
            .map_err(|e| GraphError::New(e.to_string()))?;
        Ok(found)
    }

    /// [`Self::node_has_adjacency_be`] for many nodes, aligned with `node_ids`:
    /// ONE round trip that reads at most one row per node where the backend
    /// batches. The per-node check answers elsewhere (LMDB, the batch-read kill
    /// switch, a snapshot with a write overlay) and when the batched read
    /// fails, so a failure is always the per-node check's own.
    pub(crate) fn nodes_have_adjacency_be(
        &self,
        r: &AnyRead<'_>,
        node_ids: &[u128],
        out: bool,
    ) -> Result<Vec<bool>, GraphError> {
        let ns = if out {
            Namespace::OutEdges
        } else {
            Namespace::InEdges
        };
        let prefixes: Vec<Vec<u8>> = node_ids
            .iter()
            .map(|id| id.to_be_bytes().to_vec())
            .collect();
        match self
            .backend
            .scan_dup_prefix_values_many(r, ns, &prefixes, Some(1))
        {
            Ok(Some(rows)) => return Ok(rows.iter().map(|row| !row.is_empty()).collect()),
            Ok(None) => {}
            Err(error) => {
                tracing::warn!(
                    collection = self.collection_display_name(),
                    nodes = node_ids.len(),
                    error = %error,
                    "batched adjacency check failed; checking one node at a time"
                );
                metrics::counter!("helix_graph_delete_batch_fallback_total", "step" => "check")
                    .increment(1);
            }
        }
        node_ids
            .iter()
            .map(|id| self.node_has_adjacency_be(r, *id, out))
            .collect()
    }

    /// The ids of `node_ids` with no out and no in adjacency row on `r`, in
    /// input order: the nodes an orphan cleanup may remove.
    pub(crate) fn nodes_without_adjacency_be(
        &self,
        r: &AnyRead<'_>,
        node_ids: &[u128],
    ) -> Result<Vec<u128>, GraphError> {
        let has_out = self.nodes_have_adjacency_be(r, node_ids, true)?;
        let has_in = self.nodes_have_adjacency_be(r, node_ids, false)?;
        Ok(node_ids
            .iter()
            .zip(has_out.iter().zip(&has_in))
            .filter_map(|(id, (has_out, has_in))| (!has_out && !has_in).then_some(*id))
            .collect())
    }

    /// The ids of `node_ids` whose adjacency rows all belong to edges of
    /// `dropped` (an edge id to its two ends), in input order: the nodes left
    /// without an edge once the write batch `staged`, which has staged the
    /// drop of those edges, is stored. The batch asks here, after staging,
    /// and the rows are read on a snapshot taken now: it holds what other
    /// writers have committed since the edges were chosen, so an edge that
    /// has come keeps its ends and one that has gone keeps none. It predates
    /// the batch and still holds the rows of the dropped edges, so a node is
    /// read past them: one row more than the dropped edges that share an end
    /// is enough to tell.
    ///
    /// shortcut: a commit between this snapshot and the batch's own is not
    /// seen: an edge that comes then loses its end's row, and of two batches
    /// that each drop a node's last other edge both keep the node. Closing
    /// that needs a per-collection write gate.
    pub(crate) fn nodes_orphaned_by_dropping_be(
        &self,
        _staged: &AnyWrite<'_>,
        node_ids: &[u128],
        dropped: &BTreeMap<u128, (u128, u128)>,
    ) -> Result<Vec<u128>, GraphError> {
        if node_ids.is_empty() {
            return Ok(Vec::new());
        }
        #[cfg(test)]
        if let Some(other_writer) =
            BEFORE_ORPHAN_CHECK_FOR_TEST.with(|hook| hook.borrow_mut().take())
        {
            other_writer();
        }
        let r = self
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))?;
        let keeps_out = self.nodes_keep_adjacency_be(&r, node_ids, dropped, true)?;
        let keeps_in = self.nodes_keep_adjacency_be(&r, node_ids, dropped, false)?;
        Ok(node_ids
            .iter()
            .zip(keeps_out.iter().zip(&keeps_in))
            .filter_map(|(id, (keeps_out, keeps_in))| (!keeps_out && !keeps_in).then_some(*id))
            .collect())
    }

    /// For each of `node_ids`, whether it has an out (or in) adjacency row on
    /// `r` of an edge that is not in `dropped`.
    fn nodes_keep_adjacency_be(
        &self,
        r: &AnyRead<'_>,
        node_ids: &[u128],
        dropped: &BTreeMap<u128, (u128, u128)>,
        out: bool,
    ) -> Result<Vec<bool>, GraphError> {
        let ns = if out {
            Namespace::OutEdges
        } else {
            Namespace::InEdges
        };
        // The most rows of dropped edges one node holds on this side: a node
        // with one row more has a row of another edge among them.
        let mut rows_dropped: HashMap<u128, usize> = HashMap::new();
        for (from, to) in dropped.values() {
            *rows_dropped
                .entry(if out { *from } else { *to })
                .or_default() += 1;
        }
        let row_cap = rows_dropped.values().max().copied().unwrap_or(0) + 1;
        // A row too short to name its edge keeps its node.
        let keeps = |row: &[u8]| match row.get(..16) {
            Some(edge_id) => {
                let mut raw = [0u8; 16];
                raw.copy_from_slice(edge_id);
                !dropped.contains_key(&u128::from_be_bytes(raw))
            }
            None => true,
        };
        let prefixes: Vec<Vec<u8>> = node_ids
            .iter()
            .map(|id| id.to_be_bytes().to_vec())
            .collect();
        match self
            .backend
            .scan_dup_prefix_values_many(r, ns, &prefixes, Some(row_cap))
        {
            Ok(Some(rows)) => {
                return Ok(rows
                    .iter()
                    .map(|of_node| of_node.iter().any(|row| keeps(row.as_slice())))
                    .collect());
            }
            Ok(None) => {}
            Err(error) => {
                tracing::warn!(
                    collection = self.collection_display_name(),
                    nodes = node_ids.len(),
                    error = %error,
                    "batched adjacency check failed; checking one node at a time"
                );
                metrics::counter!("helix_graph_delete_batch_fallback_total", "step" => "check")
                    .increment(1);
            }
        }
        node_ids
            .iter()
            .map(|id| -> Result<bool, GraphError> {
                let mut kept = false;
                self.backend
                    .scan(r, ns, KeyRange::prefix(&id.to_be_bytes()), |_key, row| {
                        kept = keeps(row);
                        !kept
                    })
                    .map_err(|e| GraphError::New(e.to_string()))?;
                Ok(kept)
            })
            .collect()
    }

    pub fn node_has_adjacency(
        &self,
        txn: &RoTxn,
        node_id: u128,
        out: bool,
    ) -> Result<bool, GraphError> {
        if self.backend.kind() == BackendKind::Lsm {
            let r = self
                .backend
                .begin_read()
                .map_err(|e| GraphError::New(e.to_string()))?;
            return self.node_has_adjacency_be(&r, node_id, out);
        }

        let db = if out {
            self.out_edges_db
        } else {
            self.in_edges_db
        }
        .ok_or_else(|| {
            GraphError::StorageError(
                "LMDB adjacency DB handle is unavailable on this backend".to_string(),
            )
        })?;
        let has = db
            .lazily_decode_data()
            .prefix_iter(txn, &node_id.to_be_bytes())
            .map(|mut it| it.next().is_some())
            .unwrap_or(false);
        Ok(has)
    }
}

#[cfg(test)]
mod tests {
    use super::super::backend::StorageBackend;
    use super::super::storage_methods::StorageMethods;
    use super::*;
    use crate::helix_engine::graph_core::config::Config;
    use crate::protocol::value::Value;

    fn test_config() -> Config {
        Config::new(8, 32, 64, 1)
    }

    /// Read the out/in adjacency edge ids via the raw heed `out_edges_db` /
    /// `in_edges_db` handle — the same `get_duplicates` path the production
    /// traversals (`out_e`/`in_e`) use — to establish ground truth.
    fn heed_adj_edge_ids(
        storage: &HelixGraphStorage,
        out: bool,
        node: &u128,
        edge_label: &str,
    ) -> Vec<u128> {
        storage
            .with_read_txn(|rtxn| {
                let label_hash = hash_label(edge_label, None);
                let (db, key) = if out {
                    (
                        storage.out_edges_db.as_ref().unwrap(),
                        HelixGraphStorage::out_edge_key(node, &label_hash),
                    )
                } else {
                    (
                        storage.in_edges_db.as_ref().unwrap(),
                        HelixGraphStorage::in_edge_key(node, &label_hash),
                    )
                };
                let mut ids = Vec::new();
                if let Some(iter) = db.get_duplicates(rtxn, &key)? {
                    for item in iter {
                        let (_k, v) = item?;
                        let (_peer, edge_id) = HelixGraphStorage::unpack_adj_edge_data(v)?;
                        ids.push(edge_id);
                    }
                }
                Ok(ids)
            })
            .unwrap()
    }

    /// One edge: `out_edges_be` returns the same single edge id the heed
    /// out-traversal yields, and likewise for the in-direction.
    #[test]
    fn out_in_edges_be_match_heed_single_edge() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
        let from: u128 = 0xA1;
        let to: u128 = 0xA2;
        let edge_id = storage
            .with_write_txn(|wtxn| {
                storage.create_node(wtxn, "Label", vec![], None, Some(from))?;
                storage.create_node(wtxn, "Label", vec![], None, Some(to))?;
                let edge = storage.create_edge(
                    wtxn,
                    "REL",
                    &from,
                    &to,
                    vec![("w".to_string(), Value::I32(7))],
                )?;
                Ok(edge.id)
            })
            .unwrap();

        let heed_out = heed_adj_edge_ids(&storage, true, &from, "REL");
        let heed_in = heed_adj_edge_ids(&storage, false, &to, "REL");
        assert_eq!(heed_out, vec![edge_id], "heed ground truth (out)");
        assert_eq!(heed_in, vec![edge_id], "heed ground truth (in)");

        let r = storage.backend.begin_read().unwrap();
        let be_out = storage.out_edges_be(&r, &from, "REL").unwrap();
        let be_in = storage.in_edges_be(&r, &to, "REL").unwrap();

        assert_eq!(
            be_out, heed_out,
            "out_edges_be must match the heed out-traversal byte-for-byte"
        );
        assert_eq!(
            be_in, heed_in,
            "in_edges_be must match the heed in-traversal byte-for-byte"
        );
    }

    /// Multiple edges of the same label from one node land under the same
    /// DUP_SORT key; `out_edges_be` returns all of them in the same LMDB
    /// dup-sorted order the heed traversal sees.
    #[test]
    fn out_edges_be_matches_heed_multiple_edges() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
        let from: u128 = 0xB0;
        let to1: u128 = 0xB1;
        let to2: u128 = 0xB2;
        let to3: u128 = 0xB3;
        storage
            .with_write_txn(|wtxn| {
                storage.create_node(wtxn, "Label", vec![], None, Some(from))?;
                for to in [to1, to2, to3] {
                    storage.create_node(wtxn, "Label", vec![], None, Some(to))?;
                    storage.create_edge(wtxn, "REL", &from, &to, vec![])?;
                }
                Ok(())
            })
            .unwrap();

        let heed_out = heed_adj_edge_ids(&storage, true, &from, "REL");
        assert_eq!(heed_out.len(), 3, "expected three out edges");

        let r = storage.backend.begin_read().unwrap();
        let be_out = storage.out_edges_be(&r, &from, "REL").unwrap();
        assert_eq!(
            be_out, heed_out,
            "out_edges_be must match heed for multiple edges, in dup-sorted order"
        );
    }

    /// `*_adj_be` decodes the peer node id alongside the edge id, matching the
    /// `out`/`in_` traversals which resolve the peer node from each value.
    #[test]
    fn adj_be_decodes_peer_node_ids() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
        let from: u128 = 0xC0;
        let to: u128 = 0xC1;
        let edge_id = storage
            .with_write_txn(|wtxn| {
                storage.create_node(wtxn, "Label", vec![], None, Some(from))?;
                storage.create_node(wtxn, "Label", vec![], None, Some(to))?;
                let edge = storage.create_edge(wtxn, "REL", &from, &to, vec![])?;
                Ok(edge.id)
            })
            .unwrap();

        let r = storage.backend.begin_read().unwrap();
        let out_pairs = storage.out_adj_be(&r, &from, "REL").unwrap();
        let in_pairs = storage.in_adj_be(&r, &to, "REL").unwrap();

        // out: peer is the to-node; in: peer is the from-node.
        assert_eq!(out_pairs, vec![(to, edge_id)]);
        assert_eq!(in_pairs, vec![(from, edge_id)]);
    }

    /// A node with no edges of the requested label reads empty on the backend
    /// path (missing DUP_SORT key), never erroring.
    #[test]
    fn edges_be_empty_for_node_without_edges() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
        let lonely: u128 = 0xD0;
        storage
            .with_write_txn(|wtxn| {
                storage.create_node(wtxn, "Label", vec![], None, Some(lonely))?;
                Ok(())
            })
            .unwrap();

        let r = storage.backend.begin_read().unwrap();
        assert!(storage.out_edges_be(&r, &lonely, "REL").unwrap().is_empty());
        assert!(storage.in_edges_be(&r, &lonely, "REL").unwrap().is_empty());
        // Wrong label also yields nothing (distinct 20-byte key).
        assert!(storage
            .out_edges_be(&r, &lonely, "OTHER")
            .unwrap()
            .is_empty());
    }

    /// An in-memory LSM store holding `edges` as `(edge id, from, to, label)`.
    fn lsm_storage_with_edges(
        edges: &[(u128, u128, u128, &str)],
    ) -> (tempfile::TempDir, HelixGraphStorage) {
        use super::super::upsert::EdgeUpsert;

        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(
            dir.path().to_str().unwrap(),
            test_config().with_lsm_in_memory(),
        )
        .unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);
        let mut w = storage.backend.begin_write().unwrap();
        for (id, from_node, to_node, label) in edges {
            let edge = EdgeUpsert {
                id: *id,
                label: label.to_string(),
                from_node: *from_node,
                to_node: *to_node,
                properties: Default::default(),
            };
            storage.upsert_edge_be(&mut w, &edge).unwrap();
        }
        storage.backend.commit(w).unwrap();
        (dir, storage)
    }

    /// The batched check answers what the per-node check answers, in input
    /// order, in one round trip per direction and without reading a node's
    /// rows past the first: out edges only, in edges only, both (under two
    /// labels), a node with no edges, an id with no node row, the lowest and
    /// the all-0xFF id, and a node with hundreds of edges.
    #[test]
    #[serial_test::serial]
    fn lsm_nodes_have_adjacency_be_matches_the_per_node_check() {
        use super::super::backend_lsm::{
            dup_scan_rows_for_test as dup_scan_rows, lsm_round_trips_for_test as round_trips,
        };

        const OUT_ONLY: u128 = 0x10;
        const IN_ONLY: u128 = 0x20;
        const BOTH: u128 = 0x30;
        const NO_EDGES: u128 = 0x40;
        const NO_ROW: u128 = 0x50;
        const HUB: u128 = 0x60;
        const HUB_PEER: u128 = 0x61;
        let mut edges = vec![
            (0xE1, OUT_ONLY, IN_ONLY, "CALLS"),
            (0xE2, BOTH, IN_ONLY, "CALLS"),
            (0xE3, OUT_ONLY, BOTH, "IMPORTS"),
            (0xE4, u128::MAX, 0, "CALLS"),
        ];
        edges.extend((0..300u128).map(|i| (0x1000 + i, HUB_PEER, HUB, "CALLS")));
        let (_dir, storage) = lsm_storage_with_edges(&edges);
        let mut w = storage.backend.begin_write().unwrap();
        storage
            .create_node_be(&mut w, "Sym", [], None, Some(NO_EDGES))
            .unwrap();
        storage.backend.commit(w).unwrap();

        // Ids just below and above the hub pair must not see its rows.
        let ids = [
            BOTH,
            OUT_ONLY,
            u128::MAX,
            NO_ROW,
            HUB,
            IN_ONLY,
            NO_EDGES,
            0,
            HUB_PEER,
            HUB - 1,
            HUB_PEER + 1,
        ];
        let has_out = [
            true, true, true, false, false, false, false, false, true, false, false,
        ];
        let has_in = [
            true, false, false, false, true, true, false, true, false, false, false,
        ];

        let r = storage.backend.begin_read().unwrap();
        for (out, expected) in [(true, has_out), (false, has_in)] {
            let (trips, rows) = (round_trips(), dup_scan_rows());
            let batched = storage.nodes_have_adjacency_be(&r, &ids, out).unwrap();
            let (trips, rows) = (round_trips() - trips, dup_scan_rows() - rows);
            let per_node: Vec<bool> = ids
                .iter()
                .map(|id| storage.node_has_adjacency_be(&r, *id, out).unwrap())
                .collect();
            assert_eq!(batched, per_node, "out={out}");
            assert_eq!(batched, expected, "out={out}");
            assert_eq!(trips, 1, "out={out}: one round trip for the batch");
            // One probe row for the one key length, at most one row per node.
            assert!(
                rows <= ids.len() + 1,
                "out={out}: the batched check pulled {rows} rows for {} nodes",
                ids.len()
            );
        }
        assert!(storage
            .nodes_have_adjacency_be(&r, &[], true)
            .unwrap()
            .is_empty());
    }

    /// A dup row whose declared key length overruns it fails the batched read
    /// where its key-length probe meets the row. The check then answers one
    /// node at a time: the per-node answers when a node's walk never lands on
    /// the row, the per-node failure when it does, and never "no adjacency"
    /// for a read that failed.
    #[test]
    #[serial_test::serial]
    fn lsm_nodes_have_adjacency_be_answers_per_node_when_the_batched_read_fails() {
        let (_dir, storage) = lsm_storage_with_edges(&[(0xE1, 0x10, 0x20, "CALLS")]);
        // A plain put stores the bytes as they are; no dup write produces
        // such a row.
        let put_raw = |raw: &[u8]| {
            let mut w = storage.backend.begin_write().unwrap();
            storage
                .backend
                .put(&mut w, Namespace::OutEdges, raw, b"")
                .unwrap();
            storage.backend.commit(w).unwrap();
        };
        let ids = [0x10u128, 0x20, 0x30];
        let prefixes: Vec<Vec<u8>> = ids.iter().map(|id| id.to_be_bytes().to_vec()).collect();
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();

        metrics::with_local_recorder(&recorder, || {
            // First of the 16-byte keys: the probe starts on it, a node's own
            // walk starts past it.
            put_raw(&[0, 0, 0, 16, 0]);
            let r = storage.backend.begin_read().unwrap();
            assert!(
                storage
                    .backend
                    .scan_dup_prefix_values_many(&r, Namespace::OutEdges, &prefixes, Some(1))
                    .is_err(),
                "fixture: the batched read must fail on the truncated row"
            );
            let per_node: Vec<bool> = ids
                .iter()
                .map(|id| storage.node_has_adjacency_be(&r, *id, true).unwrap())
                .collect();
            assert_eq!(per_node, [true, false, false]);
            assert_eq!(
                storage.nodes_have_adjacency_be(&r, &ids, true).unwrap(),
                per_node
            );
            drop(r);

            // First of the 40-byte keys: the walk of a node without rows runs
            // into it after the 20-byte keys.
            put_raw(&[0, 0, 0, 40, 0]);
            let r = storage.backend.begin_read().unwrap();
            assert!(storage.node_has_adjacency_be(&r, 0x10, true).unwrap());
            assert!(storage.node_has_adjacency_be(&r, 0x20, true).is_err());
            assert_eq!(
                storage.nodes_have_adjacency_be(&r, &[0x10], true).unwrap(),
                [true]
            );
            assert!(storage.nodes_have_adjacency_be(&r, &ids, true).is_err());
        });
        // Three batched checks failed over to the per-node check.
        let rendered = handle.render();
        assert!(
            rendered
                .lines()
                .any(|line| line == "helix_graph_delete_batch_fallback_total{step=\"check\"} 3"),
            "every failed batched check is counted once: {rendered}"
        );
    }

    /// The same parity on flushed data, where production adjacency lives:
    /// rows in an SST, then deletes that leave dead rows ahead of a node's
    /// first live row (the hub's first live row sits a thousand dead rows,
    /// several blocks, into its range), a node whose rows are all dead, and
    /// those tombstones flushed into an SST of their own.
    #[test]
    #[serial_test::serial]
    fn lsm_nodes_have_adjacency_be_matches_the_per_node_check_on_flushed_rows() {
        use super::super::backend_any::AnyBackend;
        use super::super::backend_lsm::lsm_round_trips_for_test as round_trips;

        const HUB: u128 = 0x60;
        const HUB_PEER: u128 = 0x61;
        const FEW: u128 = 0x70;
        const DEAD: u128 = 0x80;
        const SOURCE: u128 = 0x90;
        const NO_EDGES: u128 = 0xA0;
        // The rows of one node and label sort by edge id.
        let hub_edges = 0x1000..0x1000 + 1200u128;
        let mut edges: Vec<(u128, u128, u128, &str)> = hub_edges
            .clone()
            .map(|id| (id, HUB_PEER, HUB, "CALLS"))
            .collect();
        edges.extend((1..=3u128).map(|i| (0x2000 + i, SOURCE, FEW, "CALLS")));
        edges.extend((1..=3u128).map(|i| (0x3000 + i, SOURCE, DEAD, "CALLS")));
        let (_dir, storage) = lsm_storage_with_edges(&edges);
        let flush = || {
            let AnyBackend::Lsm(writer) = &*storage.backend else {
                unreachable!("the fixture opens an LSM writer")
            };
            writer.flush_memtable_for_test();
        };
        let ids = [HUB, HUB_PEER, FEW, DEAD, SOURCE, NO_EDGES];
        let check = |stage: &str, has_out: [bool; 6], has_in: [bool; 6]| {
            let r = storage.backend.begin_read().unwrap();
            for (out, expected) in [(true, has_out), (false, has_in)] {
                let trips = round_trips();
                let batched = storage.nodes_have_adjacency_be(&r, &ids, out).unwrap();
                assert_eq!(
                    round_trips() - trips,
                    1,
                    "{stage}, out={out}: one round trip for the batch"
                );
                let per_node: Vec<bool> = ids
                    .iter()
                    .map(|id| storage.node_has_adjacency_be(&r, *id, out).unwrap())
                    .collect();
                assert_eq!(batched, per_node, "{stage}, out={out}");
                assert_eq!(batched, expected, "{stage}, out={out}");
            }
        };

        flush();
        check(
            "rows in an SST",
            [false, true, false, false, true, false],
            [true, false, true, true, false, false],
        );

        // The hub's first thousand rows, the first two of FEW, all of DEAD.
        let mut w = storage.backend.begin_write().unwrap();
        for id in hub_edges
            .take(1000)
            .chain([0x2001, 0x2002, 0x3001, 0x3002, 0x3003])
        {
            storage.drop_edge_be(&mut w, &id).unwrap();
        }
        storage.backend.commit(w).unwrap();
        let has_out = [false, true, false, false, true, false];
        let has_in = [true, false, true, false, false, false];
        check("tombstones in the memtable", has_out, has_in);
        flush();
        check("tombstones in a second SST", has_out, has_in);
    }
}
