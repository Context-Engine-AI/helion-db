//! Backend-routed edge WRITE building blocks for the migration (US-004).
//! Mirror `StorageMethods::create_edge` / `drop_edge` byte-for-byte (including
//! out/in adjacency DUP_SORT writes) but route through `self.backend`.
//! ADDITIVE. Filled in incrementally.
//!
//! ## What `create_edge`/`drop_edge` write, and how each is reproduced here
//!
//! `create_edge` (heed) performs these effects, in order:
//!   1. READ-only existence check of both endpoints in `nodes_db`
//!      (`NodeNotFound` if either is absent). Reproduced via
//!      `backend.get_with(Namespace::Nodes, &node_id.to_be_bytes())`. The heed
//!      `nodes_db` is `U128<BE>`-keyed, so the on-disk key bytes are exactly
//!      `node_id.to_be_bytes()` — identical to what the backend writes/reads.
//!   2. Edge bytes: `edges_db.put(edge_id, encode_edge(&edge))`. The heed
//!      `edges_db` is `U128<BE>`-keyed (key bytes = `edge_id.to_be_bytes()`),
//!      value = bincode big-endian fixint of the edge (label/from/to/props;
//!      `Edge::id` is `#[serde(skip)]`). Reproduced via
//!      `backend.put(Namespace::Edges, &edge_id.to_be_bytes(), &bytes)`.
//!   3. out_edges adjacency: `out_edges_db.put(out_edge_key(from, label_hash),
//!      pack_edge_data(to, edge_id))` on a DUP_SORT db. Reproduced via
//!      `backend.put_dup(Namespace::OutEdges, &out_key, &packed)`.
//!   4. in_edges adjacency: `in_edges_db.put(in_edge_key(to, label_hash),
//!      pack_edge_data(from, edge_id))`. Reproduced via
//!      `backend.put_dup(Namespace::InEdges, &in_key, &packed)`.
//!   5. `index_edge_paths`: for each present `from_path`/`to_path` string
//!      property, `edge_path_idx.put(edge_path_index_key(dir, path, edge_id),
//!      edge_id.to_be_bytes())`. Reproduced via
//!      `backend.put(Namespace::EdgePathIdx, &key, &edge_id.to_be_bytes())`.
//!
//! `drop_edge` (heed) performs:
//!   a. READ edge bytes (`EdgeNotFound` if absent), decode to recover
//!      label/from/to for the adjacency keys.
//!   b. `delete_edge_paths`: delete the same EdgePathIdx keys `index_edge_paths`
//!      wrote.
//!   c. `edges_db.delete(edge_id)`.
//!   d. `out_edges_db.delete(out_edge_key(from, label_hash))` — note heed
//!      `Database::delete` on a DUP_SORT db removes *every* duplicate under the
//!      key. To stay byte-identical we delete ONLY the one packed value for this
//!      edge via `delete_dup`, so co-located edges of the same (from,label) keep
//!      their adjacency entries. See the divergence note in the report.
//!   e. `in_edges_db.delete(in_edge_key(to, label_hash))` — same treatment.
//!   f. `adjust_metadata_counter(Edges, -1)`.

#![allow(dead_code)]

use std::collections::HashMap;
use std::hash::Hasher;

use twox_hash::XxHash64;

use crate::helix_engine::types::GraphError;
use crate::protocol::items::{v6_uuid, Edge, SerializedEdge};
use crate::protocol::label_hash::hash_label;
use crate::protocol::value::Value;

use super::backend::{Namespace, StorageBackend};
use super::backend_any::AnyWrite;
use super::metadata::MetadataCounter;
use super::storage_core::HelixGraphStorage;
use super::upsert::EdgeUpsert;

/// Compact-key prefix (`COMPACT_INDEX_KEY_PREFIX` in `storage_core.rs`) emitted
/// when an edge-path component exceeds `MAX_EDGE_PATH_COMPONENT_BYTES`.
const COMPACT_INDEX_KEY_PREFIX: &[u8] = b"\xffhxk1";

/// Raw-length cap for a single edge-path component before it is hashed
/// (`MAX_EDGE_PATH_COMPONENT_BYTES`).
const MAX_EDGE_PATH_COMPONENT_BYTES: usize = 400;

impl HelixGraphStorage {
    /// Backend-routed twin of `StorageMethods::create_edge`. Performs EXACTLY
    /// the same DB writes (edge bytes, out/in adjacency, edge-path index, edge
    /// counter), all through `self.backend` on the supplied write handle, so the
    /// caller controls the atomic unit. Byte-identical on-disk to the heed path.
    pub fn create_edge_be(
        &self,
        w: &mut AnyWrite<'_>,
        label: &str,
        from_node: &u128,
        to_node: &u128,
        properties: impl IntoIterator<Item = (String, Value)>,
    ) -> Result<Edge, GraphError> {
        // (1) Both endpoints must exist (matches the heed `nodes_db.get` check).
        if self.node_missing_be(from_node)? || self.node_missing_be(to_node)? {
            return Err(GraphError::NodeNotFound);
        }

        let edge = Edge {
            id: v6_uuid(),
            label: label.to_string(),
            from_node: *from_node,
            to_node: *to_node,
            properties: HashMap::from_iter(properties),
        };

        // (2) Edge bytes under the 16-byte big-endian edge id.
        let edge_bytes = SerializedEdge::encode_edge(&edge)?;
        self.backend
            .put(w, Namespace::Edges, &edge.id.to_be_bytes(), &edge_bytes)
            .map_err(|e| GraphError::New(e.to_string()))?;

        let label_hash = hash_label(label, None);

        // (3) out_edges: key = from|label_hash, value = edge_id|to.
        let out_key = Self::out_edge_key(from_node, &label_hash);
        let out_val = Self::pack_edge_data(to_node, &edge.id);
        self.backend
            .put_dup(w, Namespace::OutEdges, &out_key, &out_val)
            .map_err(|e| GraphError::New(e.to_string()))?;

        // (4) in_edges: key = to|label_hash, value = edge_id|from.
        let in_key = Self::in_edge_key(to_node, &label_hash);
        let in_val = Self::pack_edge_data(from_node, &edge.id);
        self.backend
            .put_dup(w, Namespace::InEdges, &in_key, &in_val)
            .map_err(|e| GraphError::New(e.to_string()))?;

        // (5) edge_path_idx (conditional, only for non-empty string paths).
        self.index_edge_paths_be(w, &edge)?;

        // (6) metadata edge counter += 1.
        self.bump_edge_counter_be(w, 1)?;

        Ok(edge)
    }

    /// Backend-routed twin of `upsert::upsert_edge` (deterministic-id edge
    /// upsert, the CE graph-ingest write path). Mirrors `upsert_edge`'s exact
    /// effects — primary edge bytes, out/in adjacency with the
    /// put-new-before-delete-old ordering and the idempotent-repair no-op branch,
    /// edge-path index, edge counter — but routes every write through
    /// `self.backend`, so on the LSM backend edges + adjacency land in SlateDB/S3
    /// (durable + reader-visible) instead of the local LMDB `graph_env`.
    /// Byte-identical on-disk to the heed `upsert_edge` path.
    pub(crate) fn upsert_edge_be(
        &self,
        w: &mut AnyWrite<'_>,
        upsert: &EdgeUpsert,
    ) -> Result<Edge, GraphError> {
        // Existing edge (read-your-writes through the write handle), mirroring the
        // heed `edges_db.get(&txn, edge_key)` lookup. `Namespace::Edges` is not a
        // DUP namespace, so this is a single-value get.
        let existing = self
            .backend
            .get_for_update(w, Namespace::Edges, &upsert.id.to_be_bytes(), |v| {
                v.map(|bytes| SerializedEdge::decode_edge(bytes, upsert.id))
                    .transpose()
            })
            .map_err(|e| GraphError::New(e.to_string()))??;

        let edge = Edge {
            id: upsert.id,
            label: upsert.label.clone(),
            from_node: upsert.from_node,
            to_node: upsert.to_node,
            properties: upsert.properties.clone(),
        };

        // Adjacency keys derive from (label, from, to); only redo adjacency when
        // one of those changed (identical to the heed path's `adj_changed`).
        let adj_changed = match &existing {
            None => true,
            Some(e) => {
                e.label != edge.label || e.from_node != edge.from_node || e.to_node != edge.to_node
            }
        };

        let label_hash = hash_label(&edge.label, None);
        let encoded_edge = SerializedEdge::encode_edge(&edge)?;

        // Primary write FIRST: if this fails, no adjacency state has been touched.
        self.backend
            .put(w, Namespace::Edges, &edge.id.to_be_bytes(), &encoded_edge)
            .map_err(|e| GraphError::New(e.to_string()))?;

        if adj_changed {
            // Put NEW adjacency before deleting OLD (same failure-window reasoning
            // as the heed path).
            self.backend
                .put_dup(
                    w,
                    Namespace::OutEdges,
                    &Self::out_edge_key(&edge.from_node, &label_hash),
                    &Self::pack_edge_data(&edge.to_node, &edge.id),
                )
                .map_err(|e| GraphError::New(e.to_string()))?;
            self.backend
                .put_dup(
                    w,
                    Namespace::InEdges,
                    &Self::in_edge_key(&edge.to_node, &label_hash),
                    &Self::pack_edge_data(&edge.from_node, &edge.id),
                )
                .map_err(|e| GraphError::New(e.to_string()))?;
            self.index_edge_paths_be(w, &edge)?;

            if let Some(existing) = &existing {
                let old_label_hash = hash_label(&existing.label, None);
                self.delete_edge_paths_be(w, existing)?;
                self.backend
                    .delete_dup(
                        w,
                        Namespace::OutEdges,
                        &Self::out_edge_key(&existing.from_node, &old_label_hash),
                        &Self::pack_edge_data(&existing.to_node, &existing.id),
                    )
                    .map_err(|e| GraphError::New(e.to_string()))?;
                self.backend
                    .delete_dup(
                        w,
                        Namespace::InEdges,
                        &Self::in_edge_key(&existing.to_node, &old_label_hash),
                        &Self::pack_edge_data(&existing.from_node, &existing.id),
                    )
                    .map_err(|e| GraphError::New(e.to_string()))?;
            }
        } else {
            // Idempotent repair (mirrors the heed no-op branch): re-put adjacency
            // so a historically-missing row is healed on reingest. DUP put of the
            // same (key,value) is idempotent; nothing is deleted here.
            self.backend
                .put_dup(
                    w,
                    Namespace::OutEdges,
                    &Self::out_edge_key(&edge.from_node, &label_hash),
                    &Self::pack_edge_data(&edge.to_node, &edge.id),
                )
                .map_err(|e| GraphError::New(e.to_string()))?;
            self.backend
                .put_dup(
                    w,
                    Namespace::InEdges,
                    &Self::in_edge_key(&edge.to_node, &label_hash),
                    &Self::pack_edge_data(&edge.from_node, &edge.id),
                )
                .map_err(|e| GraphError::New(e.to_string()))?;
            self.index_edge_paths_be(w, &edge)?;
        }

        if existing.is_none() {
            self.bump_edge_counter_be(w, 1)?;
        }

        Ok(edge)
    }

    /// Backend-routed twin of `StorageMethods::drop_edge`. Removes the edge
    /// bytes, both adjacency entries (only the duplicate for THIS edge), the
    /// edge-path index entries, and decrements the edge counter — all on the
    /// supplied write handle.
    pub fn drop_edge_be(&self, w: &mut AnyWrite<'_>, id: &u128) -> Result<(), GraphError> {
        // (a) Load + decode the edge to recover label/from/to (EdgeNotFound if
        // absent — same as the heed path).
        let edge = self.read_edge_for_drop_be(id)?;
        let label_hash = hash_label(&edge.label, None);

        // (b) edge_path_idx deletes (mirror `delete_edge_paths`).
        self.delete_edge_paths_be(w, &edge)?;

        // (c) edge bytes.
        self.backend
            .delete(w, Namespace::Edges, &id.to_be_bytes())
            .map_err(|e| GraphError::New(e.to_string()))?;

        // (d) out_edges: delete ONLY this edge's packed value.
        let out_key = Self::out_edge_key(&edge.from_node, &label_hash);
        let out_val = Self::pack_edge_data(&edge.to_node, id);
        self.backend
            .delete_dup(w, Namespace::OutEdges, &out_key, &out_val)
            .map_err(|e| GraphError::New(e.to_string()))?;

        // (e) in_edges: delete ONLY this edge's packed value.
        let in_key = Self::in_edge_key(&edge.to_node, &label_hash);
        let in_val = Self::pack_edge_data(&edge.from_node, id);
        self.backend
            .delete_dup(w, Namespace::InEdges, &in_key, &in_val)
            .map_err(|e| GraphError::New(e.to_string()))?;

        // (f) metadata edge counter -= 1.
        self.bump_edge_counter_be(w, -1)?;

        Ok(())
    }

    /// `true` if `nodes_db` has no entry for `node_id` (matches the heed
    /// `nodes_db.get(...).is_none()` existence check). Opens a short backend read
    /// snapshot, which sees state committed before this write batch.
    fn node_missing_be(&self, node_id: &u128) -> Result<bool, GraphError> {
        let r = self
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))?;
        let present = self
            .backend
            .get_with(&r, Namespace::Nodes, &node_id.to_be_bytes(), |v| {
                v.is_some()
            })
            .map_err(|e| GraphError::New(e.to_string()))?;
        Ok(!present)
    }

    /// Load + decode the edge being dropped via a fresh read snapshot, which
    /// sees writes committed before this drop (the heed path likewise reads the
    /// edge before mutating it).
    fn read_edge_for_drop_be(&self, id: &u128) -> Result<Edge, GraphError> {
        let r = self
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))?;
        let bytes = self
            .backend
            .get_with(&r, Namespace::Edges, &id.to_be_bytes(), |v| {
                v.map(|b| b.to_vec())
            })
            .map_err(|e| GraphError::New(e.to_string()))?;
        match bytes {
            Some(b) => SerializedEdge::decode_edge(&b, *id),
            None => Err(GraphError::EdgeNotFound),
        }
    }

    /// Mirror of `index_edge_paths`: write the from/to path index keys when the
    /// edge carries non-empty `from_path`/`to_path` string properties.
    /// `pub(crate)` so `backfill_edge_path_index_batch_be` (storage_core.rs)
    /// can reuse it instead of duplicating the key layout.
    pub(crate) fn index_edge_paths_be(
        &self,
        w: &mut AnyWrite<'_>,
        edge: &Edge,
    ) -> Result<(), GraphError> {
        if let Some(path) = edge_path_property(edge, "from_path") {
            let key = edge_path_index_key(1, &path, &edge.id);
            self.backend
                .put(w, Namespace::EdgePathIdx, &key, &edge.id.to_be_bytes())
                .map_err(|e| GraphError::New(e.to_string()))?;
        }
        if let Some(path) = edge_path_property(edge, "to_path") {
            let key = edge_path_index_key(2, &path, &edge.id);
            self.backend
                .put(w, Namespace::EdgePathIdx, &key, &edge.id.to_be_bytes())
                .map_err(|e| GraphError::New(e.to_string()))?;
        }
        Ok(())
    }

    /// Mirror of `delete_edge_paths`. `pub(crate)` for the LSM backfill test.
    pub(crate) fn delete_edge_paths_be(
        &self,
        w: &mut AnyWrite<'_>,
        edge: &Edge,
    ) -> Result<(), GraphError> {
        if let Some(path) = edge_path_property(edge, "from_path") {
            let key = edge_path_index_key(1, &path, &edge.id);
            self.backend
                .delete(w, Namespace::EdgePathIdx, &key)
                .map_err(|e| GraphError::New(e.to_string()))?;
        }
        if let Some(path) = edge_path_property(edge, "to_path") {
            let key = edge_path_index_key(2, &path, &edge.id);
            self.backend
                .delete(w, Namespace::EdgePathIdx, &key)
                .map_err(|e| GraphError::New(e.to_string()))?;
        }
        Ok(())
    }

    fn bump_edge_counter_be(&self, w: &mut AnyWrite<'_>, delta: i64) -> Result<(), GraphError> {
        self.adjust_metadata_counter_be(w, MetadataCounter::Edges, delta)
    }
}

/// `edge.properties.get(key)` as a non-empty owned `String` — faithful copy of
/// the module-private `HelixGraphStorage::edge_path_property`.
fn edge_path_property(edge: &Edge, key: &str) -> Option<String> {
    match edge.properties.get(key) {
        Some(Value::String(path)) if !path.is_empty() => Some(path.clone()),
        _ => None,
    }
}

/// Faithful copy of the module-private `HelixGraphStorage::compact_index_bytes`.
/// Short inputs pass through verbatim; oversized inputs collapse to a
/// fixed-width `prefix|len|h1|h2` digest. MUST stay byte-identical to the heed
/// builder or the heed path-index readers will not find these keys.
fn compact_index_bytes(bytes: &[u8], max_raw_len: usize) -> Vec<u8> {
    if bytes.len() <= max_raw_len {
        return bytes.to_vec();
    }
    let mut h1 = XxHash64::with_seed(0x4845_4c49_585f_4b31);
    h1.write(bytes);
    let mut h2 = XxHash64::with_seed(0x4345_5f4b_4559_5f32);
    h2.write(bytes);

    let mut out = Vec::with_capacity(COMPACT_INDEX_KEY_PREFIX.len() + 24);
    out.extend_from_slice(COMPACT_INDEX_KEY_PREFIX);
    out.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
    out.extend_from_slice(&h1.finish().to_be_bytes());
    out.extend_from_slice(&h2.finish().to_be_bytes());
    out
}

/// `[dir] ++ compact(path) ++ [0]` — copy of `edge_path_index_prefix`.
fn edge_path_index_prefix(dir: u8, path: &str) -> Vec<u8> {
    let path = compact_index_bytes(path.as_bytes(), MAX_EDGE_PATH_COMPONENT_BYTES);
    let mut key = Vec::with_capacity(1 + path.len() + 1);
    key.push(dir);
    key.extend_from_slice(&path);
    key.push(0);
    key
}

/// `edge_path_index_prefix(dir, path) ++ edge_id.to_be_bytes()` — copy of
/// `edge_path_index_key`.
fn edge_path_index_key(dir: u8, path: &str, edge_id: &u128) -> Vec<u8> {
    let mut key = edge_path_index_prefix(dir, path);
    key.extend_from_slice(&edge_id.to_be_bytes());
    key
}

#[cfg(test)]
mod tests {
    use super::super::backend::StorageBackend;
    use super::super::storage_methods::{BasicStorageMethods, StorageMethods};
    use super::*;
    use crate::helix_engine::graph_core::config::Config;

    fn test_config() -> Config {
        Config::new(8, 32, 64, 1)
    }

    /// Raw `out_edges`/`in_edges` duplicate set for a (node, label) key, read via
    /// the heed `out_edges_db`/`in_edges_db` handle — ground truth for adjacency.
    fn heed_adj_raw(
        storage: &HelixGraphStorage,
        out: bool,
        node: &u128,
        edge_label: &str,
    ) -> Vec<Vec<u8>> {
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
                let mut vals = Vec::new();
                if let Some(iter) = db.get_duplicates(rtxn, &key)? {
                    for item in iter {
                        let (_k, v) = item?;
                        vals.push(v.to_vec());
                    }
                }
                Ok(vals)
            })
            .unwrap()
    }

    /// (1) Same-store readback: an edge created via `create_edge_be` is readable
    /// via the heed `get_edge`, and shows up in BOTH adjacency directions with
    /// the right peer/edge ids and node-existence semantics.
    #[test]
    fn create_edge_be_same_store_readback() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
        let from: u128 = 0xA1;
        let to: u128 = 0xA2;

        // Endpoints first (heed create_node), committed before the edge write.
        // Only one LMDB write txn can be open at a time, so the node txn must
        // close before `begin_write` for the edge.
        storage
            .with_write_txn(|wtxn| {
                storage.create_node(wtxn, "Label", vec![], None, Some(from))?;
                storage.create_node(wtxn, "Label", vec![], None, Some(to))?;
                Ok(())
            })
            .unwrap();

        let mut w = storage.backend.begin_write().unwrap();
        let edge = storage
            .create_edge_be(
                &mut w,
                "REL",
                &from,
                &to,
                vec![("w".to_string(), Value::I32(7))],
            )
            .unwrap();
        storage.backend.commit(w).unwrap();

        // Readback via heed get_edge.
        let read = storage
            .with_read_txn(|rtxn| storage.get_edge(rtxn, &edge.id))
            .unwrap();
        assert_eq!(read.id, edge.id);
        assert_eq!(read.label, "REL");
        assert_eq!(read.from_node, from);
        assert_eq!(read.to_node, to);
        assert_eq!(read.properties.get("w"), Some(&Value::I32(7)));

        // Adjacency via the existing backend-routed readers.
        let r = storage.backend.begin_read().unwrap();
        assert_eq!(
            storage.out_edges_be(&r, &from, "REL").unwrap(),
            vec![edge.id]
        );
        assert_eq!(storage.in_edges_be(&r, &to, "REL").unwrap(), vec![edge.id]);
        assert_eq!(
            storage.out_adj_be(&r, &from, "REL").unwrap(),
            vec![(to, edge.id)]
        );
        assert_eq!(
            storage.in_adj_be(&r, &to, "REL").unwrap(),
            vec![(from, edge.id)]
        );
    }

    /// `create_edge_be` rejects a missing endpoint with `NodeNotFound`, like the
    /// heed `create_edge`.
    #[test]
    fn create_edge_be_missing_node_errors() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
        let from: u128 = 0xB1;
        let to: u128 = 0xB2;
        storage
            .with_write_txn(|wtxn| {
                storage.create_node(wtxn, "Label", vec![], None, Some(from))?;
                Ok(())
            })
            .unwrap();

        let mut w = storage.backend.begin_write().unwrap();
        let res = storage.create_edge_be(&mut w, "REL", &from, &to, vec![]);
        assert!(matches!(res, Err(GraphError::NodeNotFound)));
    }

    /// (2) Twin-store byte equality: the same edge via heed `create_edge`
    /// (store A) vs `create_edge_be` (store B). The raw `edges_db` value bytes
    /// and the raw out/in adjacency duplicate sets must be identical. Edge ids
    /// are `v6_uuid()` (time-based) so we force the SAME id by writing the heed
    /// edge first and reusing its id is impossible (private constructor); instead
    /// we compare the structural bytes by normalizing the edge id out.
    #[test]
    fn create_edge_be_twin_store_byte_equal() {
        let from: u128 = 0xC1;
        let to: u128 = 0xC2;
        let props = || vec![("w".to_string(), Value::I32(42))];

        // Store A: heed create_edge.
        let dir_a = tempfile::TempDir::new().unwrap();
        let store_a =
            HelixGraphStorage::new(dir_a.path().to_str().unwrap(), test_config()).unwrap();
        let edge_a = store_a
            .with_write_txn(|wtxn| {
                store_a.create_node(wtxn, "Label", vec![], None, Some(from))?;
                store_a.create_node(wtxn, "Label", vec![], None, Some(to))?;
                store_a.create_edge(wtxn, "REL", &from, &to, props())
            })
            .unwrap();

        // Store B: create_edge_be.
        let dir_b = tempfile::TempDir::new().unwrap();
        let store_b =
            HelixGraphStorage::new(dir_b.path().to_str().unwrap(), test_config()).unwrap();
        store_b
            .with_write_txn(|wtxn| {
                store_b.create_node(wtxn, "Label", vec![], None, Some(from))?;
                store_b.create_node(wtxn, "Label", vec![], None, Some(to))?;
                Ok(())
            })
            .unwrap();
        let mut w = store_b.backend.begin_write().unwrap();
        let edge_b = store_b
            .create_edge_be(&mut w, "REL", &from, &to, props())
            .unwrap();
        store_b.backend.commit(w).unwrap();

        // --- edges_db value bytes ---
        // The edge value bytes are independent of the edge id (`Edge::id` is
        // `#[serde(skip)]`), so the two encodings must be byte-identical.
        let raw_a = store_a
            .with_read_txn(|rtxn| store_a.get_temp_edge(rtxn, &edge_a.id).map(|s| s.to_vec()))
            .unwrap();
        let raw_b = store_b
            .with_read_txn(|rtxn| store_b.get_temp_edge(rtxn, &edge_b.id).map(|s| s.to_vec()))
            .unwrap();
        assert_eq!(raw_a, raw_b, "edges_db value bytes must be byte-identical");

        // --- adjacency duplicate sets ---
        // Each adjacency value is edge_id(16 BE) | peer(16 BE). The edge ids
        // differ between stores, so compare the peer half (bytes [16..32]) which
        // is the id-independent, structural part. Both must hold exactly one
        // entry with the same peer.
        let out_a = heed_adj_raw(&store_a, true, &from, "REL");
        let out_b = heed_adj_raw(&store_b, true, &from, "REL");
        assert_eq!(out_a.len(), 1);
        assert_eq!(out_b.len(), 1);
        assert_eq!(out_a[0].len(), 32, "out adj value is 32 bytes");
        assert_eq!(out_b[0].len(), 32, "out adj value is 32 bytes");
        assert_eq!(
            &out_a[0][16..32],
            &out_b[0][16..32],
            "out adjacency peer bytes must be identical"
        );
        // And the leading edge-id half equals each store's own edge id.
        assert_eq!(&out_a[0][0..16], &edge_a.id.to_be_bytes());
        assert_eq!(&out_b[0][0..16], &edge_b.id.to_be_bytes());

        let in_a = heed_adj_raw(&store_a, false, &to, "REL");
        let in_b = heed_adj_raw(&store_b, false, &to, "REL");
        assert_eq!(in_a.len(), 1);
        assert_eq!(in_b.len(), 1);
        assert_eq!(
            &in_a[0][16..32],
            &in_b[0][16..32],
            "in adjacency peer bytes must be identical"
        );
        assert_eq!(&in_a[0][0..16], &edge_a.id.to_be_bytes());
        assert_eq!(&in_b[0][0..16], &edge_b.id.to_be_bytes());
    }

    /// `upsert_edge_be` (the LSM CE-ingest edge write path) produces on-disk state
    /// byte-identical to the heed `upsert_edge`. Because upsert uses a
    /// DETERMINISTIC edge id, both stores get the SAME id — so edge bytes AND full
    /// adjacency values (all 32 bytes) must match exactly. Covers all three
    /// branches: new edge, idempotent re-upsert (no-op), endpoint change
    /// (adj_changed delete-old → put-new).
    #[test]
    fn upsert_edge_be_twin_matches_heed() {
        use crate::helix_engine::storage_core::upsert::EdgeUpsert;
        let eid: u128 = 0xE150;
        let from: u128 = 0xE1;
        let to: u128 = 0xE2;
        let mk = |from: u128, to: u128| EdgeUpsert {
            id: eid,
            label: "REL".to_string(),
            from_node: from,
            to_node: to,
            properties: std::collections::HashMap::from_iter([("w".to_string(), Value::I32(42))]),
        };

        // Store A: heed upsert_edge.
        let dir_a = tempfile::TempDir::new().unwrap();
        let store_a =
            HelixGraphStorage::new(dir_a.path().to_str().unwrap(), test_config()).unwrap();
        store_a
            .with_write_txn(|wtxn| {
                store_a.create_node(wtxn, "Label", vec![], None, Some(from))?;
                store_a.create_node(wtxn, "Label", vec![], None, Some(to))?;
                store_a.upsert_edge(wtxn, &mk(from, to))?;
                Ok(())
            })
            .unwrap();

        // Store B: upsert_edge_be on the backend write handle.
        let dir_b = tempfile::TempDir::new().unwrap();
        let store_b =
            HelixGraphStorage::new(dir_b.path().to_str().unwrap(), test_config()).unwrap();
        store_b
            .with_write_txn(|wtxn| {
                store_b.create_node(wtxn, "Label", vec![], None, Some(from))?;
                store_b.create_node(wtxn, "Label", vec![], None, Some(to))?;
                Ok(())
            })
            .unwrap();
        let mut w = store_b.backend.begin_write().unwrap();
        store_b.upsert_edge_be(&mut w, &mk(from, to)).unwrap();
        store_b.backend.commit(w).unwrap();

        // Edge bytes identical (same deterministic id).
        let raw_a = store_a
            .with_read_txn(|rtxn| store_a.get_temp_edge(rtxn, &eid).map(|s| s.to_vec()))
            .unwrap();
        let raw_b = store_b
            .with_read_txn(|rtxn| store_b.get_temp_edge(rtxn, &eid).map(|s| s.to_vec()))
            .unwrap();
        assert_eq!(raw_a, raw_b, "edges_db value bytes must be byte-identical");

        // Full adjacency values identical (same id → compare all 32 bytes).
        assert_eq!(
            heed_adj_raw(&store_a, true, &from, "REL"),
            heed_adj_raw(&store_b, true, &from, "REL"),
            "out adjacency must be byte-identical to heed"
        );
        assert_eq!(
            heed_adj_raw(&store_a, false, &to, "REL"),
            heed_adj_raw(&store_b, false, &to, "REL"),
            "in adjacency must be byte-identical to heed"
        );

        // Idempotent re-upsert (no-op branch): adjacency stays exactly one entry.
        let mut w = store_b.backend.begin_write().unwrap();
        store_b.upsert_edge_be(&mut w, &mk(from, to)).unwrap();
        store_b.backend.commit(w).unwrap();
        assert_eq!(
            heed_adj_raw(&store_b, true, &from, "REL").len(),
            1,
            "idempotent re-upsert must not duplicate out adjacency"
        );

        // Endpoint change (adj_changed): re-point the SAME edge id to a new to-node.
        let to2: u128 = 0xE3;
        store_b
            .with_write_txn(|wtxn| {
                store_b
                    .create_node(wtxn, "Label", vec![], None, Some(to2))
                    .map(|_| ())
            })
            .unwrap();
        let mut w = store_b.backend.begin_write().unwrap();
        store_b.upsert_edge_be(&mut w, &mk(from, to2)).unwrap();
        store_b.backend.commit(w).unwrap();
        assert!(
            heed_adj_raw(&store_b, false, &to, "REL").is_empty(),
            "old in-adjacency must be removed after endpoint change"
        );
        assert_eq!(
            heed_adj_raw(&store_b, false, &to2, "REL").len(),
            1,
            "new in-adjacency must be present after endpoint change"
        );
        assert_eq!(
            heed_adj_raw(&store_b, true, &from, "REL").len(),
            1,
            "out adjacency must still hold exactly one entry after endpoint change"
        );
    }

    /// `drop_edge_be` removes the edge bytes AND both adjacency entries, while
    /// leaving a co-located sibling edge (same from+label) untouched — proving we
    /// delete only the one duplicate, not the whole DUP_SORT key.
    #[test]
    fn drop_edge_be_removes_edge_and_adjacency() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
        let from: u128 = 0xD0;
        let to1: u128 = 0xD1;
        let to2: u128 = 0xD2;
        storage
            .with_write_txn(|wtxn| {
                storage.create_node(wtxn, "Label", vec![], None, Some(from))?;
                storage.create_node(wtxn, "Label", vec![], None, Some(to1))?;
                storage.create_node(wtxn, "Label", vec![], None, Some(to2))?;
                Ok(())
            })
            .unwrap();

        let mut w = storage.backend.begin_write().unwrap();
        let e1 = storage
            .create_edge_be(&mut w, "REL", &from, &to1, vec![])
            .unwrap();
        let e2 = storage
            .create_edge_be(&mut w, "REL", &from, &to2, vec![])
            .unwrap();
        storage.backend.commit(w).unwrap();

        // Drop only e1.
        let mut w = storage.backend.begin_write().unwrap();
        storage.drop_edge_be(&mut w, &e1.id).unwrap();
        storage.backend.commit(w).unwrap();

        // e1 edge bytes gone; e2 still present.
        assert!(storage
            .with_read_txn(|rtxn| storage.get_edge(rtxn, &e1.id))
            .is_err());
        assert!(storage
            .with_read_txn(|rtxn| storage.get_edge(rtxn, &e2.id))
            .is_ok());

        // out adjacency for `from` now holds only e2; in adjacency cleaned up.
        let r = storage.backend.begin_read().unwrap();
        let out = storage.out_edges_be(&r, &from, "REL").unwrap();
        assert_eq!(out, vec![e2.id], "only the sibling edge remains outgoing");
        assert!(
            storage.in_edges_be(&r, &to1, "REL").unwrap().is_empty(),
            "dropped edge's in-adjacency removed"
        );
        assert_eq!(
            storage.in_edges_be(&r, &to2, "REL").unwrap(),
            vec![e2.id],
            "sibling edge's in-adjacency intact"
        );
    }

    /// `drop_edge_be` on a missing id surfaces `EdgeNotFound`.
    #[test]
    fn drop_edge_be_missing_errors() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
        let mut w = storage.backend.begin_write().unwrap();
        let res = storage.drop_edge_be(&mut w, &0xDEADu128);
        assert!(matches!(res, Err(GraphError::EdgeNotFound)));
    }

    /// The edge-path index is written for non-empty `from_path`/`to_path` string
    /// properties and is byte-identical to the heed `index_edge_paths` output;
    /// `drop_edge_be` removes those index entries again.
    #[test]
    fn create_edge_be_indexes_edge_paths() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
        let from: u128 = 0xE0;
        let to: u128 = 0xE1;
        storage
            .with_write_txn(|wtxn| {
                storage.create_node(wtxn, "Label", vec![], None, Some(from))?;
                storage.create_node(wtxn, "Label", vec![], None, Some(to))?;
                Ok(())
            })
            .unwrap();

        let mut w = storage.backend.begin_write().unwrap();
        let edge = storage
            .create_edge_be(
                &mut w,
                "REL",
                &from,
                &to,
                vec![
                    (
                        "from_path".to_string(),
                        Value::String("src/a.rs".to_string()),
                    ),
                    ("to_path".to_string(), Value::String("src/b.rs".to_string())),
                ],
            )
            .unwrap();
        storage.backend.commit(w).unwrap();

        // Both paths resolve to this edge via the heed path-index reader.
        let ids_from = storage
            .with_read_txn(|rtxn| storage.edge_ids_for_path(rtxn, "src/a.rs", 16))
            .unwrap();
        let ids_to = storage
            .with_read_txn(|rtxn| storage.edge_ids_for_path(rtxn, "src/b.rs", 16))
            .unwrap();
        assert!(ids_from.contains(&edge.id), "from_path indexed");
        assert!(ids_to.contains(&edge.id), "to_path indexed");

        // Drop removes the index entries.
        let mut w = storage.backend.begin_write().unwrap();
        storage.drop_edge_be(&mut w, &edge.id).unwrap();
        storage.backend.commit(w).unwrap();

        let ids_from = storage
            .with_read_txn(|rtxn| storage.edge_ids_for_path(rtxn, "src/a.rs", 16))
            .unwrap();
        assert!(!ids_from.contains(&edge.id), "from_path de-indexed on drop");
    }

    /// Twin-store proof that the edge-path index KEY BYTES match the heed
    /// builder (validating the inline `compact_index_bytes`/`edge_path_index_key`
    /// copies): create the same path-bearing edge via heed `create_edge` (A) and
    /// `create_edge_be` (B), then compare the raw `edge_path_idx` key set with
    /// each store's trailing edge id normalized to zero.
    #[test]
    fn edge_path_index_key_bytes_match_heed() {
        let from: u128 = 0xE5;
        let to: u128 = 0xE6;
        let path_props = || {
            vec![
                (
                    "from_path".to_string(),
                    Value::String("pkg/x.rs".to_string()),
                ),
                ("to_path".to_string(), Value::String("pkg/y.rs".to_string())),
            ]
        };

        let collect_idx = |storage: &HelixGraphStorage, edge_id: u128| -> Vec<Vec<u8>> {
            storage
                .with_read_txn(|rtxn| {
                    let mut keys = Vec::new();
                    let iter = storage.lmdb_edge_path_idx().unwrap().iter(rtxn)?;
                    for item in iter {
                        let (k, _v) = item?;
                        let mut k = k.to_vec();
                        let n = k.len();
                        assert!(n >= 16);
                        // Trailing 16 bytes are this store's edge id.
                        assert_eq!(&k[n - 16..], &edge_id.to_be_bytes());
                        // Normalize the id out so the comparison is id-independent.
                        for b in &mut k[n - 16..] {
                            *b = 0;
                        }
                        keys.push(k);
                    }
                    keys.sort();
                    Ok(keys)
                })
                .unwrap()
        };

        let dir_a = tempfile::TempDir::new().unwrap();
        let store_a =
            HelixGraphStorage::new(dir_a.path().to_str().unwrap(), test_config()).unwrap();
        let edge_a = store_a
            .with_write_txn(|wtxn| {
                store_a.create_node(wtxn, "Label", vec![], None, Some(from))?;
                store_a.create_node(wtxn, "Label", vec![], None, Some(to))?;
                store_a.create_edge(wtxn, "REL", &from, &to, path_props())
            })
            .unwrap();

        let dir_b = tempfile::TempDir::new().unwrap();
        let store_b =
            HelixGraphStorage::new(dir_b.path().to_str().unwrap(), test_config()).unwrap();
        store_b
            .with_write_txn(|wtxn| {
                store_b.create_node(wtxn, "Label", vec![], None, Some(from))?;
                store_b.create_node(wtxn, "Label", vec![], None, Some(to))?;
                Ok(())
            })
            .unwrap();
        let mut w = store_b.backend.begin_write().unwrap();
        let edge_b = store_b
            .create_edge_be(&mut w, "REL", &from, &to, path_props())
            .unwrap();
        store_b.backend.commit(w).unwrap();

        assert_eq!(
            collect_idx(&store_a, edge_a.id),
            collect_idx(&store_b, edge_b.id),
            "edge_path_idx key bytes must match heed (id-normalized)"
        );
    }

    /// The metadata edge counter goes up on create and down on drop, matching
    /// the heed `adjust_metadata_counter(Edges, ±1)` side-effect.
    #[test]
    fn create_drop_edge_be_track_edge_counter() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
        let from: u128 = 0xF0;
        let to: u128 = 0xF1;
        storage
            .with_write_txn(|wtxn| {
                storage.create_node(wtxn, "Label", vec![], None, Some(from))?;
                storage.create_node(wtxn, "Label", vec![], None, Some(to))?;
                Ok(())
            })
            .unwrap();

        let base = storage
            .with_read_txn(|rtxn| storage.get_metadata(rtxn))
            .unwrap()
            .stats
            .edge_count;

        let mut w = storage.backend.begin_write().unwrap();
        let edge = storage
            .create_edge_be(&mut w, "REL", &from, &to, vec![])
            .unwrap();
        storage.backend.commit(w).unwrap();

        let after_create = storage
            .with_read_txn(|rtxn| storage.get_metadata(rtxn))
            .unwrap()
            .stats
            .edge_count;
        assert_eq!(after_create, base + 1, "edge counter incremented on create");

        let mut w = storage.backend.begin_write().unwrap();
        storage.drop_edge_be(&mut w, &edge.id).unwrap();
        storage.backend.commit(w).unwrap();

        let after_drop = storage
            .with_read_txn(|rtxn| storage.get_metadata(rtxn))
            .unwrap()
            .stats
            .edge_count;
        assert_eq!(after_drop, base, "edge counter decremented on drop");
    }
}
