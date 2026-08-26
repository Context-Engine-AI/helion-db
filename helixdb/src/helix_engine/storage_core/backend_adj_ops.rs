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

use heed3::RoTxn;

use crate::helix_engine::types::GraphError;
use crate::protocol::label_hash::hash_label;

use super::backend::{BackendKind, KeyRange, Namespace, StorageBackend};
use super::backend_any::AnyRead;
use super::storage_core::HelixGraphStorage;

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
}
