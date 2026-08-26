use std::collections::HashMap;

use heed3::{RoTxn, RwTxn};

use crate::helix_engine::storage_core::{
    backend::{BackendKind, Namespace, StorageBackend},
    backend_any::AnyWrite,
    metadata::MetadataCounter,
    storage_core::HelixGraphStorage,
};
use crate::helix_engine::types::GraphError;
use crate::protocol::items::{Edge, Node};
use crate::protocol::items::{SerializedEdge, SerializedNode};
use crate::protocol::label_hash::hash_label;
use crate::protocol::value::Value;

fn is_mdb_problem(error: &heed3::Error) -> bool {
    matches!(error, heed3::Error::Mdb(heed3::MdbError::Problem))
}

/// Input for upserting a node with a deterministic ID.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NodeUpsert {
    pub id: u128,
    pub label: String,
    pub properties: HashMap<String, Value>,
}

/// Input for upserting an edge with a deterministic ID.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EdgeUpsert {
    pub id: u128,
    pub label: String,
    pub from_node: u128,
    pub to_node: u128,
    pub properties: HashMap<String, Value>,
}

impl HelixGraphStorage {
    pub(crate) fn upsert_node_be(
        &self,
        w: &mut AnyWrite<'_>,
        upsert: &NodeUpsert,
    ) -> Result<Node, GraphError> {
        let existing = self
            .backend
            .get_for_update(w, Namespace::Nodes, &upsert.id.to_be_bytes(), |v| {
                v.map(|bytes| SerializedNode::decode_node(bytes, upsert.id))
                    .transpose()
            })
            .map_err(|e| GraphError::New(e.to_string()))??;
        self.upsert_node_be_with_existing(w, upsert, existing)
    }

    pub(crate) fn upsert_node_be_with_existing(
        &self,
        w: &mut AnyWrite<'_>,
        upsert: &NodeUpsert,
        existing: Option<Node>,
    ) -> Result<Node, GraphError> {
        let node = Node {
            id: upsert.id,
            label: upsert.label.clone(),
            properties: upsert.properties.clone(),
        };

        if existing.as_ref().is_some_and(|existing| {
            existing.label == node.label && existing.properties == node.properties
        }) {
            return Ok(node);
        }

        if let Some(existing) = &existing {
            for idx_name in self.multi_indices.keys() {
                let old_val = existing.properties.get(idx_name);
                let new_val = node.properties.get(idx_name);
                if old_val == new_val {
                    continue;
                }
                if let Some(old_val) = old_val {
                    let old_key = Self::stable_index_key_for_value(old_val)?;
                    self.backend
                        .delete_dup(
                            w,
                            Namespace::MultiIndex(idx_name),
                            &old_key,
                            &node.id.to_be_bytes(),
                        )
                        .map_err(|e| GraphError::New(e.to_string()))?;
                }
                if let Some(new_val) = new_val {
                    let new_key = Self::stable_index_key_for_value(new_val)?;
                    self.backend
                        .put_dup(
                            w,
                            Namespace::MultiIndex(idx_name),
                            &new_key,
                            &node.id.to_be_bytes(),
                        )
                        .map_err(|e| GraphError::New(e.to_string()))?;
                }
            }
            let payload_indices = self
                .payload_indices
                .read()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            for (idx_name, handle) in payload_indices.iter() {
                if !handle.accepts_writes() {
                    continue;
                }
                let old_val = Self::payload_value_for_key(&existing.properties, idx_name);
                let new_val = Self::payload_value_for_key(&node.properties, idx_name);
                if old_val == new_val {
                    continue;
                }
                self.deindex_node_payload_field_be(w, idx_name, &handle.schema, node.id, old_val)?;
                self.index_node_payload_field_be(w, idx_name, &handle.schema, node.id, new_val)?;
            }
        } else {
            for idx_name in self.multi_indices.keys() {
                if let Some(val) = node.properties.get(idx_name) {
                    let key = Self::stable_index_key_for_value(val)?;
                    self.backend
                        .put_dup(
                            w,
                            Namespace::MultiIndex(idx_name),
                            &key,
                            &node.id.to_be_bytes(),
                        )
                        .map_err(|e| GraphError::New(e.to_string()))?;
                }
            }
            let payload_indices = self
                .payload_indices
                .read()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            for (idx_name, handle) in payload_indices.iter() {
                if !handle.accepts_writes() {
                    continue;
                }
                self.index_node_payload_field_be(
                    w,
                    idx_name,
                    &handle.schema,
                    node.id,
                    Self::payload_value_for_key(&node.properties, idx_name),
                )?;
            }
        }

        let encoded_node = SerializedNode::encode_node(&node)?;
        self.backend
            .put(w, Namespace::Nodes, &node.id.to_be_bytes(), &encoded_node)
            .map_err(|e| GraphError::New(e.to_string()))?;

        if existing.is_none() {
            self.adjust_metadata_counter_be(w, MetadataCounter::Nodes, 1)?;
        }

        Ok(node)
    }

    /// Upsert a single node by deterministic ID.
    /// `put()` overwrites existing data if the ID already exists.
    pub fn upsert_node(&self, txn: &mut RwTxn, upsert: &NodeUpsert) -> Result<Node, GraphError> {
        if self.backend.kind() == BackendKind::Lsm {
            return self.with_write_backend(|w| self.upsert_node_be(w, upsert));
        }

        let nodes_db = self.lmdb_nodes_db()?;
        let existing = nodes_db
            .get(txn, Self::node_key(&upsert.id))?
            .map(|bytes| SerializedNode::decode_node(bytes, upsert.id))
            .transpose()?;

        let node = Node {
            id: upsert.id,
            label: upsert.label.clone(),
            properties: upsert.properties.clone(),
        };

        if existing.as_ref().is_some_and(|existing| {
            existing.label == node.label && existing.properties == node.properties
        }) {
            return Ok(node);
        }

        let encoded_node = SerializedNode::encode_node(&node)?;

        // Update only changed multi-value index entries. Re-upserts are common during
        // ingest; deleting and rewriting unchanged DUP_SORT entries adds LMDB churn.
        //
        // Index update order follows the upstream rollback pattern from
        // `traversal_core/ops/util/upsert.rs`:
        //   1. Delete the OLD entry first (the entry being replaced).
        //   2. Try to put the NEW entry.
        //   3. If the new put fails, restore the old entry (best-effort) and
        //      return the original error.
        //
        // The earlier order ("put new, then delete old") could leave both
        // entries committed if the second step failed. The rollback path
        // here is best-effort: if `put(old)` itself fails (e.g. txn quota),
        // the caller drops the txn and the whole batch reverts — same end
        // state, just slower.
        //
        // For the create path (else branch below) the primary write moves
        // BEFORE the index inserts so an index-side failure leaves no
        // orphan index entries pointing at a node that does not exist.
        if let Some(existing) = &existing {
            for (idx_name, idx_db) in &self.multi_indices {
                let idx_db = idx_db.ok_or_else(|| {
                    GraphError::StorageError(
                        "LMDB multi-index DB handle is unavailable on this backend".to_string(),
                    )
                })?;
                let old_val = existing.properties.get(idx_name);
                let new_val = node.properties.get(idx_name);
                if old_val == new_val {
                    continue;
                }
                let old_key_bytes = match old_val {
                    Some(v) => Some(Self::stable_index_key_for_value(v)?),
                    None => None,
                };
                if let Some(ref old_key) = old_key_bytes {
                    idx_db.delete_one_duplicate(txn, old_key, &node.id.to_be_bytes())?;
                }
                if let Some(val) = new_val {
                    let key = Self::stable_index_key_for_value(val)?;
                    if let Err(e) = idx_db.put(txn, &key, &node.id.to_be_bytes()) {
                        if is_mdb_problem(&e) {
                            return Err(e.into());
                        }
                        // Restore the old entry. Best-effort — if this
                        // fails, txn rollback by the caller still reverts
                        // the (already-applied) delete.
                        if let Some(ref old_key) = old_key_bytes {
                            let _ = idx_db.put(txn, old_key, &node.id.to_be_bytes());
                        }
                        return Err(e.into());
                    }
                }
            }
            let payload_indices = self
                .payload_indices
                .read()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            for (idx_name, handle) in payload_indices.iter() {
                if !handle.accepts_writes() {
                    continue;
                }
                let old_val = Self::payload_value_for_key(&existing.properties, idx_name);
                let new_val = Self::payload_value_for_key(&node.properties, idx_name);
                if old_val == new_val {
                    continue;
                }
                let db = handle.lmdb_db()?;
                self.deindex_node_payload_field(txn, db, &handle.schema, node.id, old_val)?;
                if let Err(e) =
                    self.index_node_payload_field(txn, db, &handle.schema, node.id, new_val)
                {
                    if e.is_fatal_collection_storage() {
                        return Err(e);
                    }
                    // Best-effort restore of the old payload-index entry.
                    let _ =
                        self.index_node_payload_field(txn, db, &handle.schema, node.id, old_val);
                    return Err(e);
                }
            }

            // Primary write last on the update path: a reader sees either
            // {old primary, old indexes} or (after commit) {new primary,
            // new indexes}. Delete before put forces LMDB to allocate a fresh
            // overflow page instead of reusing the old value's overflow pages
            // in-place; production cores showed a SIGSEGV in LMDB's overflow
            // memcpy path for large node payload rewrites.
            let node_key = Self::node_key(&node.id);
            nodes_db.delete(txn, &node_key)?;
            nodes_db.put(txn, &node_key, &encoded_node)?;
        } else {
            // Create path: write the primary record FIRST. An index failure
            // after primary write leaves a node with missing-but-recoverable
            // index entries (re-upsert reconstructs them) instead of orphan
            // index entries pointing at a node that nodes_db does not have.
            nodes_db.put(txn, &Self::node_key(&node.id), &encoded_node)?;

            for (idx_name, idx_db) in &self.multi_indices {
                let idx_db = idx_db.ok_or_else(|| {
                    GraphError::StorageError(
                        "LMDB multi-index DB handle is unavailable on this backend".to_string(),
                    )
                })?;
                if let Some(val) = node.properties.get(idx_name) {
                    let key = Self::stable_index_key_for_value(val)?;
                    idx_db.put(txn, &key, &node.id.to_be_bytes())?;
                }
            }
            let payload_indices = self
                .payload_indices
                .read()
                .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
            for (idx_name, handle) in payload_indices.iter() {
                if !handle.accepts_writes() {
                    continue;
                }
                let db = handle.lmdb_db()?;
                self.index_node_payload_field(
                    txn,
                    db,
                    &handle.schema,
                    node.id,
                    Self::payload_value_for_key(&node.properties, idx_name),
                )?;
            }
        }

        if existing.is_none() {
            self.adjust_metadata_counter(txn, MetadataCounter::Nodes, 1)?;
        }

        Ok(node)
    }

    /// Upsert a single edge by deterministic ID.
    /// Overwrites edge data and re-indexes adjacency entries.
    pub fn upsert_edge(&self, txn: &mut RwTxn, upsert: &EdgeUpsert) -> Result<Edge, GraphError> {
        if self.backend.kind() == BackendKind::Lsm {
            return self.with_write_backend(|w| self.upsert_edge_be(w, upsert));
        }

        let edges_db = self.lmdb_edges_db()?;
        let out_edges_db = self.out_edges_db.ok_or_else(|| {
            GraphError::StorageError(
                "LMDB out-edge DB handle is unavailable on this backend".to_string(),
            )
        })?;
        let in_edges_db = self.in_edges_db.ok_or_else(|| {
            GraphError::StorageError(
                "LMDB in-edge DB handle is unavailable on this backend".to_string(),
            )
        })?;

        let existing = edges_db
            .get(txn, Self::edge_key(&upsert.id))?
            .map(|bytes| SerializedEdge::decode_edge(bytes, upsert.id))
            .transpose()?;

        let edge = Edge {
            id: upsert.id,
            label: upsert.label.clone(),
            from_node: upsert.from_node,
            to_node: upsert.to_node,
            properties: upsert.properties.clone(),
        };

        // Adjacency keys are derived from (label, from_node, to_node). If
        // none of those changed, the existing adjacency entries are still
        // correct for this edge_id and can be skipped entirely. Only label
        // / from / to changes require redoing adjacency.
        //
        // The earlier code ran a full delete-old → write-primary → put-new
        // sequence on every upsert, which both wasted work on no-op
        // updates AND opened a failure window where a primary-write or
        // adjacency-put error left the edge with no adjacency at all.
        let adj_changed = match &existing {
            None => true,
            Some(e) => {
                e.label != edge.label || e.from_node != edge.from_node || e.to_node != edge.to_node
            }
        };

        let label_hash = hash_label(&edge.label, None);
        let encoded_edge = SerializedEdge::encode_edge(&edge)?;

        // Primary write FIRST: if encoding or this put fails, no adjacency
        // state has been touched and the existing edge (if any) is intact.
        edges_db.put(txn, &Self::edge_key(&edge.id), &encoded_edge)?;

        if adj_changed {
            // Put NEW adjacency before deleting OLD. Failure modes:
            //   - put-new fails: old adjacency still in place, primary
            //     edge has been overwritten with new properties — same
            //     edge_id resolves through old adjacency to new record,
            //     readable. Caller drops txn and the primary write
            //     reverts.
            //   - put-new succeeds, delete-old fails: BOTH adjacency
            //     entries exist transiently. Readers iterating
            //     out_edges/in_edges may see the edge_id twice; resolution
            //     by edge_id dedupes naturally. Recoverable.
            //   - The reverse (delete-old before put-new) had a worse
            //     failure mode: an erased edge with no adjacency,
            //     unrecoverable without re-upsert.
            out_edges_db.put(
                txn,
                &Self::out_edge_key(&edge.from_node, &label_hash),
                &Self::pack_edge_data(&edge.to_node, &edge.id),
            )?;
            in_edges_db.put(
                txn,
                &Self::in_edge_key(&edge.to_node, &label_hash),
                &Self::pack_edge_data(&edge.from_node, &edge.id),
            )?;
            self.index_edge_paths(txn, &edge)?;

            if let Some(existing) = &existing {
                let old_label_hash = hash_label(&existing.label, None);
                self.delete_edge_paths(txn, existing)?;
                out_edges_db.delete_one_duplicate(
                    txn,
                    &Self::out_edge_key(&existing.from_node, &old_label_hash),
                    &Self::pack_edge_data(&existing.to_node, &existing.id),
                )?;
                in_edges_db.delete_one_duplicate(
                    txn,
                    &Self::in_edge_key(&existing.to_node, &old_label_hash),
                    &Self::pack_edge_data(&existing.from_node, &existing.id),
                )?;
            }
        } else {
            // Existing deployments may have primary edge rows whose
            // adjacency write failed during an older upsert. A no-op reingest
            // must repair those rows; otherwise graph endpoints that traverse
            // out_edges_db/in_edges_db permanently miss an edge that still
            // appears in primary scans. DUP_SORT/DUP_FIXED keeps the same
            // (key, value) put idempotent, and we intentionally do not delete
            // anything in this branch.
            out_edges_db.put(
                txn,
                &Self::out_edge_key(&edge.from_node, &label_hash),
                &Self::pack_edge_data(&edge.to_node, &edge.id),
            )?;
            in_edges_db.put(
                txn,
                &Self::in_edge_key(&edge.to_node, &label_hash),
                &Self::pack_edge_data(&edge.from_node, &edge.id),
            )?;
            self.index_edge_paths(txn, &edge)?;
        }

        if existing.is_none() {
            self.adjust_metadata_counter(txn, MetadataCounter::Edges, 1)?;
        }

        Ok(edge)
    }

    /// Bulk upsert nodes in a single transaction. Returns count of upserted nodes.
    pub fn bulk_upsert_nodes(
        &self,
        txn: &mut RwTxn,
        nodes: &[NodeUpsert],
    ) -> Result<usize, GraphError> {
        for upsert in nodes {
            self.upsert_node(txn, upsert)?;
        }
        Ok(nodes.len())
    }

    /// Bulk upsert edges in a single transaction. Returns count of upserted edges.
    /// Note: does NOT validate that from_node/to_node exist — caller must ensure nodes
    /// are upserted first. This is intentional for bulk ingest performance.
    pub fn bulk_upsert_edges(
        &self,
        txn: &mut RwTxn,
        edges: &[EdgeUpsert],
    ) -> Result<usize, GraphError> {
        for upsert in edges {
            self.upsert_edge(txn, upsert)?;
        }
        Ok(edges.len())
    }

    /// Query a multi-value index: returns all node IDs matching the given value.
    pub fn get_nodes_by_multi_index(
        &self,
        txn: &RoTxn,
        index_name: &str,
        value: &Value,
    ) -> Result<Vec<u128>, GraphError> {
        if self.backend.kind() == BackendKind::Lsm {
            // On LSM the multi-index lives in SlateDB (written by `upsert_node_be`),
            // not the local LMDB `midx_*` handle. Read it through the backend seam.
            let r = self
                .backend
                .begin_read()
                .map_err(|e| GraphError::New(e.to_string()))?;
            return self.get_nodes_by_multi_index_be(&r, index_name, value);
        }

        let idx_db = self
            .multi_indices
            .get(index_name)
            .ok_or_else(|| GraphError::New(format!("Multi-index '{}' not found", index_name)))?
            .ok_or_else(|| {
                GraphError::StorageError(
                    "LMDB multi-index DB handle is unavailable on this backend".to_string(),
                )
            })?;

        let key = Self::stable_index_key_for_value(value)?;
        let mut result = Vec::new();

        let iter = idx_db.get_duplicates(txn, &key)?;
        if let Some(dup_iter) = iter {
            for item in dup_iter {
                let (_, val_bytes) = item?;
                let id = u128::from_be_bytes(
                    val_bytes
                        .try_into()
                        .map_err(|_| GraphError::SliceLengthError)?,
                );
                result.push(id);
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helix_engine::{
        graph_core::config::Config,
        storage_core::{metadata::PayloadIndexSchema, storage_methods::StorageMethods},
    };
    use crate::protocol::deterministic_id;
    use tempfile::TempDir;

    fn setup() -> (HelixGraphStorage, TempDir) {
        let tmp = TempDir::new().unwrap();
        let config = Config::new(16, 128, 768, 1);
        let storage = HelixGraphStorage::new(tmp.path().to_str().unwrap(), config).unwrap();
        (storage, tmp)
    }

    #[test]
    fn test_upsert_node_creates() {
        let (storage, _tmp) = setup();
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

        let id = deterministic_id::node_id("coll", "Symbol", "main", "src/main.rs");
        let upsert = NodeUpsert {
            id,
            label: "Symbol".into(),
            properties: HashMap::from([
                ("name".into(), Value::String("main".into())),
                ("path".into(), Value::String("src/main.rs".into())),
            ]),
        };

        let _node = storage.upsert_node(&mut txn, &upsert).unwrap();
        txn.commit().unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let fetched = storage.get_node(&txn, &id).unwrap();
        assert_eq!(fetched.label, "Symbol");
        assert_eq!(
            fetched.properties.get("name"),
            Some(&Value::String("main".into()))
        );
    }

    #[test]
    fn test_upsert_node_overwrites() {
        let (storage, _tmp) = setup();

        let id = deterministic_id::node_id("coll", "Symbol", "main", "src/main.rs");

        // First upsert
        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            storage
                .upsert_node(
                    &mut txn,
                    &NodeUpsert {
                        id,
                        label: "Symbol".into(),
                        properties: HashMap::from([("name".into(), Value::String("main".into()))]),
                    },
                )
                .unwrap();
            txn.commit().unwrap();
        }

        // Second upsert with updated properties
        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            storage
                .upsert_node(
                    &mut txn,
                    &NodeUpsert {
                        id,
                        label: "Symbol".into(),
                        properties: HashMap::from([
                            ("name".into(), Value::String("main".into())),
                            ("language".into(), Value::String("rust".into())),
                        ]),
                    },
                )
                .unwrap();
            txn.commit().unwrap();
        }

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let node = storage.get_node(&txn, &id).unwrap();
        assert_eq!(
            node.properties.get("language"),
            Some(&Value::String("rust".into()))
        );

        let metadata = storage.get_metadata(&txn).unwrap();
        assert_eq!(metadata.stats.node_count, 1);
    }

    #[test]
    fn test_upsert_node_rewrites_large_overflow_payload() {
        let (storage, _tmp) = setup();

        let id = deterministic_id::node_id("coll", "Symbol", "large", "src/large.rs");
        let large_a = "a".repeat(4096);
        let large_b = "b".repeat(4096);

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            storage
                .upsert_node(
                    &mut txn,
                    &NodeUpsert {
                        id,
                        label: "Symbol".into(),
                        properties: HashMap::from([
                            ("name".into(), Value::String("large".into())),
                            ("body".into(), Value::String(large_a)),
                        ]),
                    },
                )
                .unwrap();
            txn.commit().unwrap();
        }

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            storage
                .upsert_node(
                    &mut txn,
                    &NodeUpsert {
                        id,
                        label: "Symbol".into(),
                        properties: HashMap::from([
                            ("name".into(), Value::String("large".into())),
                            ("body".into(), Value::String(large_b.clone())),
                        ]),
                    },
                )
                .unwrap();
            txn.commit().unwrap();
        }

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let node = storage.get_node(&txn, &id).unwrap();
        assert_eq!(node.properties.get("body"), Some(&Value::String(large_b)));

        let metadata = storage.get_metadata(&txn).unwrap();
        assert_eq!(metadata.stats.node_count, 1);
    }

    #[test]
    fn test_upsert_edge() {
        let (storage, _tmp) = setup();
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

        // Create two nodes first
        let n1 = deterministic_id::node_id("coll", "Symbol", "main", "a.rs");
        let n2 = deterministic_id::node_id("coll", "Symbol", "helper", "b.rs");
        storage
            .upsert_node(
                &mut txn,
                &NodeUpsert {
                    id: n1,
                    label: "Symbol".into(),
                    properties: HashMap::new(),
                },
            )
            .unwrap();
        storage
            .upsert_node(
                &mut txn,
                &NodeUpsert {
                    id: n2,
                    label: "Symbol".into(),
                    properties: HashMap::new(),
                },
            )
            .unwrap();

        // Upsert edge
        let eid = deterministic_id::edge_id("coll", "CALLS", "main", "helper", "a.rs", "b.rs");
        storage
            .upsert_edge(
                &mut txn,
                &EdgeUpsert {
                    id: eid,
                    label: "CALLS".into(),
                    from_node: n1,
                    to_node: n2,
                    properties: HashMap::from([(
                        "edge_type".into(),
                        Value::String("CALLS".into()),
                    )]),
                },
            )
            .unwrap();
        txn.commit().unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let edge = storage.get_edge(&txn, &eid).unwrap();
        assert_eq!(edge.label, "CALLS");
        assert_eq!(edge.from_node, n1);
        assert_eq!(edge.to_node, n2);

        let metadata = storage.get_metadata(&txn).unwrap();
        assert_eq!(metadata.stats.edge_count, 1);
    }

    #[test]
    fn test_bulk_upsert() {
        let (storage, _tmp) = setup();
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();

        let nodes: Vec<NodeUpsert> = (0..100)
            .map(|i| NodeUpsert {
                id: deterministic_id::node_id("coll", "Symbol", &format!("fn_{}", i), "test.rs"),
                label: "Symbol".into(),
                properties: HashMap::from([("name".into(), Value::String(format!("fn_{}", i)))]),
            })
            .collect();

        let count = storage.bulk_upsert_nodes(&mut txn, &nodes).unwrap();
        txn.commit().unwrap();

        assert_eq!(count, 100);

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        assert_eq!(storage.lmdb_nodes_db().unwrap().len(&txn).unwrap(), 100);

        let metadata = storage.get_metadata(&txn).unwrap();
        assert_eq!(metadata.stats.node_count, 100);
    }

    #[test]
    fn test_upsert_node_rewrites_multi_index_without_count_drift() {
        let (mut storage, _tmp) = setup();
        storage.create_multi_index("name").unwrap();

        let id = deterministic_id::node_id("coll", "Symbol", "main", "src/main.rs");

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            storage
                .upsert_node(
                    &mut txn,
                    &NodeUpsert {
                        id,
                        label: "Symbol".into(),
                        properties: HashMap::from([("name".into(), Value::String("main".into()))]),
                    },
                )
                .unwrap();
            txn.commit().unwrap();
        }

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            storage
                .upsert_node(
                    &mut txn,
                    &NodeUpsert {
                        id,
                        label: "Symbol".into(),
                        properties: HashMap::from([("name".into(), Value::String("entry".into()))]),
                    },
                )
                .unwrap();
            txn.commit().unwrap();
        }

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let old_hits = storage
            .get_nodes_by_multi_index(&txn, "name", &Value::String("main".into()))
            .unwrap();
        let new_hits = storage
            .get_nodes_by_multi_index(&txn, "name", &Value::String("entry".into()))
            .unwrap();
        let metadata = storage.get_metadata(&txn).unwrap();

        assert!(old_hits.is_empty());
        assert_eq!(new_hits, vec![id]);
        assert_eq!(metadata.stats.node_count, 1);
    }

    #[test]
    fn test_upsert_node_rewrites_payload_index_without_count_drift() {
        let (storage, _tmp) = setup();
        storage
            .create_payload_index("repo", PayloadIndexSchema::Keyword)
            .unwrap();

        let id = deterministic_id::node_id("coll", "Symbol", "main", "src/main.rs");

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            storage
                .upsert_node(
                    &mut txn,
                    &NodeUpsert {
                        id,
                        label: "Symbol".into(),
                        properties: HashMap::from([
                            ("name".into(), Value::String("main".into())),
                            ("repo".into(), Value::String("old".into())),
                        ]),
                    },
                )
                .unwrap();
            txn.commit().unwrap();
        }

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            storage
                .upsert_node(
                    &mut txn,
                    &NodeUpsert {
                        id,
                        label: "Symbol".into(),
                        properties: HashMap::from([
                            ("name".into(), Value::String("main".into())),
                            ("repo".into(), Value::String("new".into())),
                        ]),
                    },
                )
                .unwrap();
            txn.commit().unwrap();
        }

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let old_hits = storage
            .get_nodes_by_payload_value(&txn, "repo", &Value::String("old".into()))
            .unwrap();
        let new_hits = storage
            .get_nodes_by_payload_value(&txn, "repo", &Value::String("new".into()))
            .unwrap();
        let metadata = storage.get_metadata(&txn).unwrap();

        assert!(old_hits.is_empty());
        assert_eq!(new_hits, vec![id]);
        assert_eq!(metadata.stats.node_count, 1);
    }

    #[test]
    fn test_upsert_node_removes_payload_index_when_field_removed() {
        let (storage, _tmp) = setup();
        storage
            .create_payload_index("repo", PayloadIndexSchema::Keyword)
            .unwrap();

        let id = deterministic_id::node_id("coll", "Symbol", "main", "src/main.rs");

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            storage
                .upsert_node(
                    &mut txn,
                    &NodeUpsert {
                        id,
                        label: "Symbol".into(),
                        properties: HashMap::from([
                            ("name".into(), Value::String("main".into())),
                            ("repo".into(), Value::String("old".into())),
                        ]),
                    },
                )
                .unwrap();
            txn.commit().unwrap();
        }

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            storage
                .upsert_node(
                    &mut txn,
                    &NodeUpsert {
                        id,
                        label: "Symbol".into(),
                        properties: HashMap::from([("name".into(), Value::String("main".into()))]),
                    },
                )
                .unwrap();
            txn.commit().unwrap();
        }

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let old_hits = storage
            .get_nodes_by_payload_value(&txn, "repo", &Value::String("old".into()))
            .unwrap();
        let metadata = storage.get_metadata(&txn).unwrap();

        assert!(old_hits.is_empty());
        assert_eq!(metadata.stats.node_count, 1);
    }

    /// Ghost payload-index regression: `drop_node`'s de-index step used a flat
    /// `node.properties.get(idx_name)` lookup, which returns `None` for a
    /// dotted index name like `metadata.repo` (the value lives nested at
    /// `properties["metadata"]["repo"]`), so the de-index step silently no-op'd
    /// on every delete and the dup entry leaked forever. Fixed by switching to
    /// `payload_value_for_key`, the same nested-aware resolver every
    /// insert/update path already uses. A flat-named index must keep working
    /// too (sibling assertion).
    #[test]
    fn test_drop_node_removes_nested_payload_index_ghost_entry() {
        use crate::helix_engine::storage_core::storage_methods::StorageMethods;

        let (storage, _tmp) = setup();
        storage
            .create_payload_index("metadata.repo", PayloadIndexSchema::Keyword)
            .unwrap();
        storage
            .create_payload_index("flat_name", PayloadIndexSchema::Keyword)
            .unwrap();

        let id = deterministic_id::node_id("coll", "Symbol", "main", "src/main.rs");
        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            storage
                .upsert_node(
                    &mut txn,
                    &NodeUpsert {
                        id,
                        label: "Symbol".into(),
                        properties: HashMap::from([
                            (
                                "metadata".into(),
                                Value::Object(HashMap::from([(
                                    "repo".into(),
                                    Value::String("nested-repo".into()),
                                )])),
                            ),
                            ("flat_name".into(), Value::String("flat-value".into())),
                        ]),
                    },
                )
                .unwrap();
            txn.commit().unwrap();
        }

        // Before delete: both index entries resolve.
        {
            let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
            assert_eq!(
                storage
                    .get_nodes_by_payload_value(
                        &txn,
                        "metadata.repo",
                        &Value::String("nested-repo".into())
                    )
                    .unwrap(),
                vec![id]
            );
            assert_eq!(
                storage
                    .get_nodes_by_payload_value(
                        &txn,
                        "flat_name",
                        &Value::String("flat-value".into())
                    )
                    .unwrap(),
                vec![id]
            );
        }

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            storage.drop_node(&mut txn, &id).unwrap();
            txn.commit().unwrap();
        }

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        assert!(
            storage
                .get_nodes_by_payload_value(
                    &txn,
                    "metadata.repo",
                    &Value::String("nested-repo".into())
                )
                .unwrap()
                .is_empty(),
            "nested-field index must not leak a ghost dup entry after delete"
        );
        assert!(
            storage
                .get_nodes_by_payload_value(&txn, "flat_name", &Value::String("flat-value".into()))
                .unwrap()
                .is_empty(),
            "flat-field index must still clean up correctly"
        );
    }

    #[test]
    fn test_keyword_payload_index_matches_numeric_equivalent_values() {
        let (storage, _tmp) = setup();
        storage
            .create_payload_index("rank", PayloadIndexSchema::Keyword)
            .unwrap();

        let id = deterministic_id::node_id("coll", "Symbol", "ranked", "src/ranked.rs");

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            storage
                .upsert_node(
                    &mut txn,
                    &NodeUpsert {
                        id,
                        label: "Symbol".into(),
                        properties: HashMap::from([
                            ("name".into(), Value::String("ranked".into())),
                            ("rank".into(), Value::U64(42)),
                        ]),
                    },
                )
                .unwrap();
            txn.commit().unwrap();
        }

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        for equivalent in [
            Value::I8(42),
            Value::I32(42),
            Value::I64(42),
            Value::U8(42),
            Value::U128(42),
            Value::F32(42.0),
            Value::F64(42.0),
        ] {
            assert_eq!(
                storage
                    .get_nodes_by_payload_value(&txn, "rank", &equivalent)
                    .unwrap(),
                vec![id],
                "query value {equivalent:?} should match stored U64"
            );
        }

        assert!(storage
            .get_nodes_by_payload_value(&txn, "rank", &Value::I64(43))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn test_upsert_edge_rewrites_adjacency_without_count_drift() {
        let (storage, _tmp) = setup();

        let n1 = deterministic_id::node_id("coll", "Symbol", "main", "a.rs");
        let n2 = deterministic_id::node_id("coll", "Symbol", "helper", "b.rs");
        let n3 = deterministic_id::node_id("coll", "Symbol", "other", "c.rs");
        let eid = deterministic_id::edge_id("coll", "CALLS", "main", "helper", "a.rs", "b.rs");

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            for id in [n1, n2, n3] {
                storage
                    .upsert_node(
                        &mut txn,
                        &NodeUpsert {
                            id,
                            label: "Symbol".into(),
                            properties: HashMap::new(),
                        },
                    )
                    .unwrap();
            }
            storage
                .upsert_edge(
                    &mut txn,
                    &EdgeUpsert {
                        id: eid,
                        label: "CALLS".into(),
                        from_node: n1,
                        to_node: n2,
                        properties: HashMap::new(),
                    },
                )
                .unwrap();
            txn.commit().unwrap();
        }

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            storage
                .upsert_edge(
                    &mut txn,
                    &EdgeUpsert {
                        id: eid,
                        label: "CALLS".into(),
                        from_node: n1,
                        to_node: n3,
                        properties: HashMap::from([(
                            "kind".into(),
                            Value::String("updated".into()),
                        )]),
                    },
                )
                .unwrap();
            txn.commit().unwrap();
        }

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let edge = storage.get_edge(&txn, &eid).unwrap();
        let old_key = HelixGraphStorage::out_edge_key(&n1, &hash_label("CALLS", None));
        let old_dups = storage
            .out_edges_db
            .as_ref()
            .unwrap()
            .get_duplicates(&txn, &old_key)
            .unwrap()
            .unwrap();
        let mut packed = Vec::new();
        for item in old_dups {
            let (_, value) = item.unwrap();
            packed.push(value.to_vec());
        }
        let metadata = storage.get_metadata(&txn).unwrap();

        assert_eq!(edge.to_node, n3);
        assert_eq!(packed.len(), 1);
        assert_eq!(
            packed[0],
            HelixGraphStorage::pack_edge_data(&n3, &eid).to_vec()
        );
        assert_eq!(metadata.stats.edge_count, 1);
    }

    #[test]
    fn test_upsert_edge_repairs_missing_adjacency_on_noop_update() {
        let (storage, _tmp) = setup();

        let n1 = deterministic_id::node_id("coll", "Symbol", "main", "a.rs");
        let n2 = deterministic_id::node_id("coll", "Symbol", "helper", "b.rs");
        let eid = deterministic_id::edge_id("coll", "CALLS", "main", "helper", "a.rs", "b.rs");

        let upsert = EdgeUpsert {
            id: eid,
            label: "CALLS".into(),
            from_node: n1,
            to_node: n2,
            properties: HashMap::from([
                ("from_path".into(), Value::String("a.rs".into())),
                ("to_path".into(), Value::String("b.rs".into())),
            ]),
        };

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            for id in [n1, n2] {
                storage
                    .upsert_node(
                        &mut txn,
                        &NodeUpsert {
                            id,
                            label: "Symbol".into(),
                            properties: HashMap::new(),
                        },
                    )
                    .unwrap();
            }
            storage.upsert_edge(&mut txn, &upsert).unwrap();

            let label_hash = hash_label("CALLS", None);
            storage
                .out_edges_db
                .as_ref()
                .unwrap()
                .delete_one_duplicate(
                    &mut txn,
                    &HelixGraphStorage::out_edge_key(&n1, &label_hash),
                    &HelixGraphStorage::pack_edge_data(&n2, &eid),
                )
                .unwrap();
            storage
                .in_edges_db
                .as_ref()
                .unwrap()
                .delete_one_duplicate(
                    &mut txn,
                    &HelixGraphStorage::in_edge_key(&n2, &label_hash),
                    &HelixGraphStorage::pack_edge_data(&n1, &eid),
                )
                .unwrap();
            txn.commit().unwrap();
        }

        {
            let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
            assert!(storage.get_edge(&txn, &eid).is_ok());
            let label_hash = hash_label("CALLS", None);
            assert!(storage
                .out_edges_db
                .as_ref()
                .unwrap()
                .get_duplicates(&txn, &HelixGraphStorage::out_edge_key(&n1, &label_hash))
                .unwrap()
                .is_none());
        }

        {
            let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
            storage.upsert_edge(&mut txn, &upsert).unwrap();
            txn.commit().unwrap();
        }

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let label_hash = hash_label("CALLS", None);
        let out_key = HelixGraphStorage::out_edge_key(&n1, &label_hash);
        let out_dups = storage
            .out_edges_db
            .as_ref()
            .unwrap()
            .get_duplicates(&txn, &out_key)
            .unwrap()
            .unwrap();
        let mut packed = Vec::new();
        for item in out_dups {
            let (_, value) = item.unwrap();
            packed.push(value.to_vec());
        }

        let metadata = storage.get_metadata(&txn).unwrap();
        assert_eq!(
            packed,
            vec![HelixGraphStorage::pack_edge_data(&n2, &eid).to_vec()]
        );
        assert_eq!(
            storage.edge_ids_for_path(&txn, "a.rs", 10).unwrap(),
            vec![eid]
        );
        assert_eq!(metadata.stats.edge_count, 1);
    }
}
