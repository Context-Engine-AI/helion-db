//! Backend-routed node WRITE building blocks for the migration (US-004).
//! Mirror `StorageMethods::create_node` / `drop_node` byte-for-byte but route
//! writes through `self.backend`. ADDITIVE. Filled in incrementally.
//!
//! ## Write parity with the heed path
//!
//! `create_node` (storage_core.rs) performs exactly three kinds of write, all of
//! which `create_node_be` reproduces through the backend write API against the
//! SAME shared LMDB env (`LmdbBackend::from_env(graph_env.clone())`):
//!
//!   1. Node bytes — `nodes_db.put(node_key(id), SerializedNode::encode_node)`.
//!      The nodes DB is `Database<U128<BE>, Bytes>`, so the on-disk key is
//!      `id.to_be_bytes()`; the value is the identical `SerializedNode` encoding.
//!      Reproduced via `put(Namespace::Nodes, &id.to_be_bytes(), &encode_node)`.
//!   2. Single-value secondary indices — for each requested index name, look it
//!      up in `self.secondary_indices` (error if unregistered), read the node's
//!      property of that name (error if missing), then
//!      `db.put(stable_index_key_for_value(value), id.to_be_bytes())`. The
//!      backend resolves `Namespace::SecondaryIndex(name)` to the SAME named DB
//!      (`ns_name` returns `name`), so `put` lands byte-identical bytes.
//!   3. Node counter — `adjust_metadata_counter(Nodes, +1)`, a read-modify-write
//!      of `metadata_db["current"]`. Reproduced via the backend `Namespace::Metadata`
//!      (which `ns_name` maps to the SAME `"metadata"` DB; heed's `Str` codec for
//!      the key is raw UTF-8, matching the backend's `Bytes` key `b"current"`).
//!      We read+decode with `try_deserialize_metadata`, call the same
//!      `adjust_counter`, and write back with the SAME delete-then-put sequence
//!      and `serialize_metadata` codec as `put_metadata`.
//!
//! Order matches `create_node` exactly (node bytes, then the index loop which may
//! error mid-way, then the counter) so failure behavior is identical.
//!
//! `drop_node` is reproduced for the writes the backend `Namespace` enum can
//! address: edge teardown (edge bytes + out/in adjacency deletes), node-bytes
//! delete, the `multi_indices` dup-deletes, payload-index de-indexing, and both
//! metadata counter adjustments.

#![allow(dead_code)]

use crate::helix_engine::storage_core::metadata::{
    counter_label, encode_lsm_counter_delta, encode_lsm_counter_value, lsm_counter_key,
    MetadataCounter,
};
use crate::helix_engine::types::GraphError;
use crate::protocol::filterable::Filterable;
use crate::protocol::items::{v6_uuid, Edge, Node, SerializedEdge, SerializedNode};
use crate::protocol::label_hash::hash_label;
use crate::protocol::value::Value;

use super::backend::{BackendError, BackendKind, Namespace, StorageBackend};
use super::backend_any::{AnyRead, AnyWrite};
use super::metadata::StorageMetadata;
use super::storage_core::HelixGraphStorage;

/// Storage-metadata codec, byte-identical to the private
/// `HelixGraphStorage::serialize_metadata` (a `0x01` version prefix followed by
/// the bincode encoding). Reproduced here because that method is module-private
/// and this migration file must not edit `storage_core.rs`.
const METADATA_CODEC_V1: u8 = 0x01;

fn serialize_metadata(meta: &StorageMetadata) -> Result<Vec<u8>, GraphError> {
    let mut buf = vec![METADATA_CODEC_V1];
    let encoded = bincode::serialize(meta)?;
    buf.extend_from_slice(&encoded);
    Ok(buf)
}

/// Mirror of the private `HelixGraphStorage::try_deserialize_metadata`: handles
/// both the versioned (`0x01` prefix) and the legacy (no-prefix) bincode forms.
fn deserialize_metadata(bytes: &[u8]) -> Result<StorageMetadata, GraphError> {
    if bytes.is_empty() {
        return Err(GraphError::StorageError("empty metadata bytes".to_string()));
    }
    let result = if bytes[0] == METADATA_CODEC_V1 {
        bincode::deserialize(&bytes[1..])
    } else {
        bincode::deserialize(bytes)
    };
    result.map_err(|e| GraphError::StorageError(format!("metadata decode: {}", e)))
}

impl HelixGraphStorage {
    /// Backend-routed twin of `StorageMethods::create_node`.
    ///
    /// Performs byte-identical writes to the heed path, in the same order:
    /// node bytes (`Namespace::Nodes`), then each requested single-value
    /// secondary index (`Namespace::SecondaryIndex`), then the node counter
    /// (`Namespace::Metadata`). The caller owns the write batch lifecycle
    /// (`begin_write` -> `create_node_be` -> `commit`).
    pub fn create_node_be(
        &self,
        w: &mut AnyWrite<'_>,
        label: &str,
        properties: impl IntoIterator<Item = (String, Value)>,
        secondary_indices: Option<&[String]>,
        id: Option<u128>,
    ) -> Result<Node, GraphError> {
        let node = Node {
            id: id.unwrap_or(v6_uuid()),
            label: label.to_string(),
            properties: std::collections::HashMap::from_iter(properties),
        };

        // 1. Validate + precompute secondary-index keys BEFORE any write. A
        //    requested index must be registered and the node must carry the
        //    indexed property (identical `GraphError::New` messages to
        //    `create_node`). Validating first means a missing property leaves NO
        //    buffered node in the batch — matching `add_n`'s pre-write validation
        //    so an errored insert never persists an orphan node. The final
        //    on-disk bytes on the success path are unchanged (same key/values).
        let indices = secondary_indices.unwrap_or(&[]);
        let mut index_keys: Vec<(&str, Vec<u8>)> = Vec::with_capacity(indices.len());
        for index in indices {
            if !self.secondary_indices.contains_key(index) {
                return Err(GraphError::New(format!(
                    "Secondary Index {} not found",
                    index
                )));
            }
            let value = match node.check_property(index) {
                Some(value) => value,
                None => {
                    return Err(GraphError::New(format!(
                        "Secondary Index {} not found",
                        index
                    )))
                }
            };
            index_keys.push((index.as_str(), Self::stable_index_key_for_value(value)?));
        }

        // 2. Store node data — same key (id big-endian) and value (SerializedNode).
        self.backend
            .put(
                w,
                Namespace::Nodes,
                &node.id.to_be_bytes(),
                &SerializedNode::encode_node(&node)?,
            )
            .map_err(|e| GraphError::New(e.to_string()))?;

        // 3. Single-value secondary indices (validated above).
        for (index, key) in &index_keys {
            self.backend
                .put(
                    w,
                    Namespace::SecondaryIndex(index),
                    key,
                    &node.id.to_be_bytes(),
                )
                .map_err(|e| GraphError::New(e.to_string()))?;
        }

        // 4. Bump the node counter in the storage metadata.
        self.adjust_metadata_counter_be(w, MetadataCounter::Nodes, 1)?;

        Ok(node)
    }

    pub fn update_node_be(
        &self,
        w: &mut AnyWrite<'_>,
        id: &u128,
        props: &[(String, Value)],
    ) -> Result<Node, GraphError> {
        let mut updated_node = self
            .backend
            .get_for_update(w, Namespace::Nodes, &id.to_be_bytes(), |v| match v {
                Some(data) => SerializedNode::decode_node(data, *id),
                None => Err(GraphError::NodeNotFound),
            })
            .map_err(|e| GraphError::New(e.to_string()))??;

        for (key, value) in props {
            let old_value = updated_node.properties.get(key);
            if old_value == Some(value) {
                continue;
            }

            if self.secondary_indices.contains_key(key) {
                let new_index_key = Self::stable_index_key_for_value(value)?;
                if let Some(old_value) = old_value {
                    let old_index_key = Self::stable_index_key_for_value(old_value)?;
                    if old_index_key != new_index_key {
                        let id_bytes = id.to_be_bytes();
                        let points_to_node = self
                            .backend
                            .get_for_update(
                                w,
                                Namespace::SecondaryIndex(key),
                                &old_index_key,
                                |existing| {
                                    existing.is_some_and(|bytes| bytes == id_bytes.as_slice())
                                },
                            )
                            .map_err(|e| GraphError::New(e.to_string()))?;
                        if points_to_node {
                            self.backend
                                .delete(w, Namespace::SecondaryIndex(key), &old_index_key)
                                .map_err(|e| GraphError::New(e.to_string()))?;
                        }
                    }
                }
                self.backend
                    .put(
                        w,
                        Namespace::SecondaryIndex(key),
                        &new_index_key,
                        &id.to_be_bytes(),
                    )
                    .map_err(|e| GraphError::New(e.to_string()))?;
            }

            updated_node.properties.insert(key.clone(), value.clone());
        }

        let encoded_node = SerializedNode::encode_node(&updated_node)?;
        self.backend
            .put(w, Namespace::Nodes, &id.to_be_bytes(), &encoded_node)
            .map_err(|e| GraphError::New(e.to_string()))?;

        Ok(updated_node)
    }

    pub fn update_edge_be(
        &self,
        w: &mut AnyWrite<'_>,
        id: &u128,
        props: &[(String, Value)],
    ) -> Result<Edge, GraphError> {
        let mut updated_edge = self
            .backend
            .get_for_update(w, Namespace::Edges, &id.to_be_bytes(), |v| match v {
                Some(data) => SerializedEdge::decode_edge(data, *id),
                None => Err(GraphError::EdgeNotFound),
            })
            .map_err(|e| GraphError::New(e.to_string()))??;

        for (key, value) in props {
            updated_edge.properties.insert(key.clone(), value.clone());
        }

        let encoded_edge = SerializedEdge::encode_edge(&updated_edge)?;
        self.backend
            .put(w, Namespace::Edges, &id.to_be_bytes(), &encoded_edge)
            .map_err(|e| GraphError::New(e.to_string()))?;

        Ok(updated_edge)
    }

    /// Backend-routed twin of `StorageMethods::drop_node`.
    ///
    /// Reproduces the heed deletes the backend `Namespace` enum can address:
    /// edge teardown (edge bytes via `Namespace::Edges`, out/in adjacency via
    /// `Namespace::OutEdges`/`InEdges`), the node-bytes delete
    /// (`Namespace::Nodes`), the `multi_indices` dup-deletes
    /// (`Namespace::MultiIndex`), and both metadata counter adjustments. Edge and
    /// node decoding, key construction, counter deltas, and the
    /// "only adjust the node counter if the node row actually existed" guard all
    /// match `drop_node`.
    ///
    /// Payload-index de-indexing is backend-routed through
    /// `Namespace::PayloadIndex`, so filtered delete/count do not observe stale
    /// index entries after an LSM node drop.
    pub fn drop_node_be(&self, w: &mut AnyWrite<'_>, id: &u128) -> Result<(), GraphError> {
        // Snapshot the node row (for the multi-index dup-deletes) and the edges
        // to remove, using a read snapshot — drop_node reads inside the same
        // write txn; here the writes are buffered in `w` and reads see committed
        // state, which is equivalent because a node's own edges/properties are
        // not mutated earlier in this call.
        let r = self
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))?;

        let existing_node: Option<Node> = self
            .backend
            .get_with(&r, Namespace::Nodes, &id.to_be_bytes(), |v| match v {
                Some(data) => Some(SerializedNode::decode_node(data, *id)),
                None => None,
            })
            .map_err(|e| GraphError::New(e.to_string()))?
            .transpose()?;

        // Collect outgoing then incoming edges (full edge records), mirroring
        // drop_node's two prefix scans + edge-data lookups.
        let out_edges = self.collect_incident_edges(&r, Namespace::OutEdges, id)?;
        let in_edges = self.collect_incident_edges(&r, Namespace::InEdges, id)?;
        drop(r);

        let removed_edge_ids = out_edges
            .iter()
            .chain(in_edges.iter())
            .map(|edge| edge.id)
            .collect::<std::collections::HashSet<_>>();

        // Delete all related edge data: edge bytes + BOTH adjacency sides. The
        // adjacency deletes remove ONLY this edge's packed dup value
        // (delete_dup), not every dup under the (node|label) key — matching the
        // corrected drop_node / drop_edge so dropping a node never wipes the
        // peers' unrelated same-label adjacency.
        for edge in out_edges.iter().chain(in_edges.iter()) {
            let label_hash = hash_label(&edge.label, None);
            self.backend
                .delete(w, Namespace::Edges, &edge.id.to_be_bytes())
                .map_err(|e| GraphError::New(e.to_string()))?;
            self.backend
                .delete_dup(
                    w,
                    Namespace::OutEdges,
                    &Self::out_edge_key(&edge.from_node, &label_hash),
                    &Self::pack_edge_data(&edge.to_node, &edge.id),
                )
                .map_err(|e| GraphError::New(e.to_string()))?;
            self.backend
                .delete_dup(
                    w,
                    Namespace::InEdges,
                    &Self::in_edge_key(&edge.to_node, &label_hash),
                    &Self::pack_edge_data(&edge.from_node, &edge.id),
                )
                .map_err(|e| GraphError::New(e.to_string()))?;
        }

        // Delete node data — only adjust the node counter if the row existed.
        let deleted = self.backend_delete_returning(w, Namespace::Nodes, &id.to_be_bytes())?;
        if deleted {
            if let Some(node) = &existing_node {
                for (idx_name, _idx_db) in &self.multi_indices {
                    if let Some(value) = Self::payload_value_for_key(&node.properties, idx_name) {
                        let key = Self::stable_index_key_for_value(value)?;
                        // `Namespace::MultiIndex(idx_name)` resolves to the on-disk
                        // `midx_{idx_name}` DB (e.g. the auto-provisioned `name`
                        // index at `midx_name`), so this dup-delete removes the real
                        // entry, matching `drop_node`.
                        self.backend
                            .delete_dup(w, Namespace::MultiIndex(idx_name), &key, &id.to_be_bytes())
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
                    self.deindex_node_payload_field_be(
                        w,
                        idx_name,
                        &handle.schema,
                        *id,
                        Self::payload_value_for_key(&node.properties, idx_name),
                    )?;
                }
            }
            self.adjust_metadata_counter_be(w, MetadataCounter::Nodes, -1)?;
        }
        if !removed_edge_ids.is_empty() {
            self.adjust_metadata_counter_be(
                w,
                MetadataCounter::Edges,
                -(removed_edge_ids.len() as i64),
            )?;
        }

        Ok(())
    }

    /// Read-modify-write of the storage metadata counter through the backend,
    /// byte-identical to `adjust_metadata_counter` -> `update_metadata` ->
    /// `put_metadata`: read `metadata["current"]`, decode with the same codec,
    /// `adjust_counter`, then delete-then-put the re-serialized bytes.
    pub(crate) fn adjust_metadata_counter_be(
        &self,
        w: &mut AnyWrite<'_>,
        counter: MetadataCounter,
        delta: i64,
    ) -> Result<(), GraphError> {
        const METADATA_CURRENT_KEY: &[u8] = b"current";

        if self.backend.kind() == BackendKind::Lsm {
            self.seed_lsm_counter_key_for_write(w, counter)?;
            let operand = encode_lsm_counter_delta(delta);
            // Counter deltas are commutative (order-independent i64 adds), so
            // classify them as such: on a conflict-checked write path this key
            // must never produce an SSI write-write conflict. Today's
            // WriteBatch commit path treats this identically to `merge`.
            self.backend
                .merge_commutative(w, Namespace::Metadata, lsm_counter_key(counter), &operand)
                .map_err(|e| GraphError::New(e.to_string()))?;
            return Ok(());
        }

        // Read current metadata via a fresh read snapshot. `update_metadata`
        // reads inside the write txn; on LMDB an uncommitted put in `w` is not
        // visible to a separate read snapshot, but the metadata key is written at
        // most once per `create_node_be`/`drop_node_be` call, so a single read
        // here sees the committed prior value — matching update_metadata's
        // read-then-write within one logical operation.
        // Read-your-writes through the write handle so multi-op batches (e.g.
        // bulk_add_n) count correctly — earlier buffered counter writes in `w`
        // are visible here, matching heed's update-within-the-write-txn.
        let mut metadata = self
            .backend
            .get_for_update(w, Namespace::Metadata, METADATA_CURRENT_KEY, |v| {
                v.map(deserialize_metadata)
            })
            .map_err(|e| GraphError::New(e.to_string()))?
            .transpose()?
            .unwrap_or_else(|| {
                crate::helix_engine::storage_core::metadata::StorageMetadata::new(Vec::new())
            });

        metadata.adjust_counter(counter, delta)?;

        let bytes = serialize_metadata(&metadata)?;
        // Delete-before-put, exactly as `put_metadata`.
        self.backend
            .delete(w, Namespace::Metadata, METADATA_CURRENT_KEY)
            .map_err(|e| GraphError::New(e.to_string()))?;
        self.backend
            .put(w, Namespace::Metadata, METADATA_CURRENT_KEY, &bytes)
            .map_err(|e| GraphError::New(e.to_string()))?;
        Ok(())
    }

    fn seed_lsm_counter_key_for_write(
        &self,
        w: &mut AnyWrite<'_>,
        counter: MetadataCounter,
    ) -> Result<(), GraphError> {
        const METADATA_CURRENT_KEY: &[u8] = b"current";

        let exists = self
            .backend
            .get_for_update(w, Namespace::Metadata, lsm_counter_key(counter), |v| {
                v.is_some()
            })
            .map_err(|e| GraphError::New(e.to_string()))?;
        if exists {
            return Ok(());
        }

        let blob_baseline = self
            .backend
            .get_for_update(w, Namespace::Metadata, METADATA_CURRENT_KEY, |v| {
                v.map(deserialize_metadata)
            })
            .map_err(|e| GraphError::New(e.to_string()))?
            .transpose()?
            .map(|metadata| match counter {
                MetadataCounter::Nodes => metadata.stats.node_count,
                MetadataCounter::Edges => metadata.stats.edge_count,
                MetadataCounter::Vectors => metadata.stats.vector_count,
            })
            .unwrap_or(0);

        // The scan needs a `Read` handle, not the in-flight `Write` batch (the
        // trait has no scan-for-update); a fresh `begin_read()` is safe to open
        // here since `begin_write()` only builds a local batch with no DB-level
        // exclusivity. It won't see this batch's own buffered writes, which is
        // correct: we want the baseline BEFORE this write's delta is merged in.
        let seeded_value = match self.backend.begin_read() {
            Ok(read) => self.seed_value_from_scan_or_blob(&read, counter, blob_baseline, "write"),
            Err(error) => {
                tracing::warn!(
                    collection = self.collection_display_name(),
                    counter = counter_label(counter),
                    context = "write",
                    source = "blob_fallback",
                    blob_baseline,
                    error = %error,
                    "LSM counter scan snapshot unavailable; falling back to stale blob baseline"
                );
                blob_baseline
            }
        };

        self.backend
            .put(
                w,
                Namespace::Metadata,
                lsm_counter_key(counter),
                &encode_lsm_counter_value(seeded_value),
            )
            .map_err(|e| GraphError::New(e.to_string()))
    }

    /// Backend-routed twin of `put_metadata`. Encodes `metadata` with the SAME
    /// codec/version prefix (`serialize_metadata`, byte-identical to the private
    /// `HelixGraphStorage::serialize_metadata`) so LMDB and LSM bytes match.
    ///
    /// The delete-before-put in `put_metadata` is an LMDB-only SIGSEGV
    /// workaround (it forces heed's fresh-allocation path to avoid an in-place
    /// overwrite onto a spilled read-only overflow page). On the LSM backend
    /// there is no mmap fault, so a plain `put` suffices; we keep the
    /// delete-then-put sequence for symmetry with `put_metadata` and
    /// `adjust_metadata_counter_be` (the seam `delete` is a cheap tombstone on
    /// LSM and a no-op net effect with the immediately-following `put`).
    pub fn put_metadata_be(
        &self,
        w: &mut AnyWrite<'_>,
        metadata: &StorageMetadata,
    ) -> Result<(), GraphError> {
        const METADATA_CURRENT_KEY: &[u8] = b"current";
        let bytes = serialize_metadata(metadata)?;
        self.backend
            .delete(w, Namespace::Metadata, METADATA_CURRENT_KEY)
            .map_err(|e| GraphError::New(e.to_string()))?;
        self.backend
            .put(w, Namespace::Metadata, METADATA_CURRENT_KEY, &bytes)
            .map_err(|e| GraphError::New(e.to_string()))?;
        Ok(())
    }

    /// Backend-routed twin of `update_metadata`: read the current metadata
    /// through the write handle (read-your-writes, matching
    /// `adjust_metadata_counter_be`), apply `update`, then `put_metadata_be`.
    pub fn update_metadata_be<F>(
        &self,
        w: &mut AnyWrite<'_>,
        mut update: F,
    ) -> Result<StorageMetadata, GraphError>
    where
        F: FnMut(&mut StorageMetadata) -> Result<(), GraphError>,
    {
        const METADATA_CURRENT_KEY: &[u8] = b"current";
        if self.backend.kind() == BackendKind::Lsm {
            return self
                .backend
                .update_lsm_key_transactional(Namespace::Metadata, METADATA_CURRENT_KEY, |value| {
                    let mut metadata = value
                        .map(deserialize_metadata)
                        .transpose()
                        .map_err(|e| BackendError::Corruption(e.to_string()))?
                        .unwrap_or_else(|| StorageMetadata::new(Vec::new()));
                    update(&mut metadata).map_err(|e| BackendError::Io(e.to_string()))?;
                    let bytes = serialize_metadata(&metadata)
                        .map_err(|e| BackendError::Corruption(e.to_string()))?;
                    Ok((bytes, metadata))
                })
                .map_err(|e| GraphError::New(e.to_string()));
        }

        let mut metadata = self
            .backend
            .get_for_update(w, Namespace::Metadata, METADATA_CURRENT_KEY, |v| {
                v.map(deserialize_metadata)
            })
            .map_err(|e| GraphError::New(e.to_string()))?
            .transpose()?
            .unwrap_or_else(|| StorageMetadata::new(Vec::new()));
        update(&mut metadata)?;
        self.put_metadata_be(w, &metadata)?;
        Ok(metadata)
    }

    /// Backend-routed twin of `set_named_vectors_metadata`.
    pub fn set_named_vectors_metadata_be(
        &self,
        w: &mut AnyWrite<'_>,
        named_vectors: std::collections::HashMap<
            String,
            crate::helix_engine::vector_core::named_vectors::NamedVectorConfig,
        >,
    ) -> Result<StorageMetadata, GraphError> {
        self.update_metadata_be(w, |metadata| {
            metadata.set_named_vectors(named_vectors.clone());
            Ok(())
        })
    }

    /// Backend-routed twin of `set_dense_vector_spaces_metadata`.
    pub fn set_dense_vector_spaces_metadata_be(
        &self,
        w: &mut AnyWrite<'_>,
        dense_vector_spaces: std::collections::HashMap<
            String,
            crate::helix_engine::vector_core::named_vectors::DenseVectorSpaceMetadata,
        >,
    ) -> Result<StorageMetadata, GraphError> {
        self.update_metadata_be(w, |metadata| {
            metadata.set_dense_vector_spaces(dense_vector_spaces.clone());
            Ok(())
        })
    }

    /// Backend-routed twin of `set_sparse_vectors_metadata`.
    pub fn set_sparse_vectors_metadata_be(
        &self,
        w: &mut AnyWrite<'_>,
        sparse_vectors: std::collections::HashMap<
            String,
            crate::helix_engine::vector_core::sparse::SparseVectorConfig,
        >,
    ) -> Result<StorageMetadata, GraphError> {
        self.update_metadata_be(w, |metadata| {
            metadata.set_sparse_vectors(sparse_vectors.clone());
            Ok(())
        })
    }

    /// Backend-routed twin of `set_hnsw_overrides`. Empty overrides are deleted
    /// (zero on-disk footprint), matching the heed path. Keyed by the same
    /// `"hnsw_overrides"` key under `Namespace::Metadata`, encoded with the same
    /// bare-bincode codec (no version prefix — `set_hnsw_overrides` does not add
    /// one).
    pub fn set_hnsw_overrides_be(
        &self,
        w: &mut AnyWrite<'_>,
        overrides: &crate::helix_engine::vector_core::vector_core::HnswOverrides,
    ) -> Result<(), GraphError> {
        const HNSW_OVERRIDES_KEY: &[u8] = b"hnsw_overrides";
        if overrides.is_empty() {
            self.backend
                .delete(w, Namespace::Metadata, HNSW_OVERRIDES_KEY)
                .map_err(|e| GraphError::New(e.to_string()))?;
            return Ok(());
        }
        let bytes = bincode::serialize(overrides)
            .map_err(|e| GraphError::StorageError(format!("hnsw overrides serialize: {}", e)))?;
        self.backend
            .put(w, Namespace::Metadata, HNSW_OVERRIDES_KEY, &bytes)
            .map_err(|e| GraphError::New(e.to_string()))?;
        Ok(())
    }

    /// Collect the full edge records incident to `node` on one adjacency side,
    /// mirroring `drop_node`'s prefix scan + edge-data lookup. The heed path
    /// prefix-iterates `out_edges_db`/`in_edges_db` by the 16-byte node id; the
    /// backend exposes adjacency as DUP_SORT keyed by the full 20-byte
    /// `out_edge_key`/`in_edge_key`, so we scan the node-id prefix and decode
    /// each 32-byte value's edge id, then load the edge record.
    fn collect_incident_edges(
        &self,
        r: &AnyRead<'_>,
        ns: Namespace<'_>,
        node: &u128,
    ) -> Result<Vec<crate::protocol::items::Edge>, GraphError> {
        use super::backend::KeyRange;

        // Gather edge ids from every adjacency value under this node's prefix.
        let mut edge_ids: Vec<u128> = Vec::new();
        let mut decode_err: Option<GraphError> = None;
        self.backend
            .scan(r, ns, KeyRange::prefix(&node.to_be_bytes()), |_k, v| {
                match Self::unpack_adj_edge_data(v) {
                    Ok((_peer, edge_id)) => {
                        edge_ids.push(edge_id);
                        true
                    }
                    Err(e) => {
                        decode_err = Some(e);
                        false
                    }
                }
            })
            .map_err(|e| GraphError::New(e.to_string()))?;
        if let Some(e) = decode_err {
            return Err(e);
        }

        let mut edges = Vec::with_capacity(edge_ids.len());
        for edge_id in edge_ids {
            let edge = self
                .backend
                .get_with(r, Namespace::Edges, &edge_id.to_be_bytes(), |v| {
                    v.map(|data| SerializedEdge::decode_edge(data, edge_id))
                })
                .map_err(|e| GraphError::New(e.to_string()))?;
            if let Some(edge) = edge {
                edges.push(edge?);
            }
        }
        Ok(edges)
    }

    /// Backend `delete` that reports whether a row existed, mirroring heed's
    /// `Database::delete` return value (used by `drop_node` to gate the counter).
    /// The backend `delete` returns `()`, so we probe existence first via a read
    /// snapshot, then delete.
    fn backend_delete_returning(
        &self,
        w: &mut AnyWrite<'_>,
        ns: Namespace<'_>,
        key: &[u8],
    ) -> Result<bool, GraphError> {
        let r = self
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))?;
        let existed = self
            .backend
            .get_with(&r, ns, key, |v| v.is_some())
            .map_err(|e| GraphError::New(e.to_string()))?;
        drop(r);
        self.backend
            .delete(w, ns, key)
            .map_err(|e| GraphError::New(e.to_string()))?;
        Ok(existed)
    }
}

#[cfg(test)]
mod tests {
    use super::super::backend::StorageBackend;
    use super::super::storage_methods::{DBMethods, StorageMethods};
    use super::*;
    use crate::helix_engine::graph_core::config::Config;

    fn test_config() -> Config {
        Config::new(8, 32, 64, 1)
    }

    /// Read the raw `nodes_db` value bytes for `id` via the heed handle — the
    /// authoritative on-disk node encoding both write paths must produce.
    fn heed_node_bytes(storage: &HelixGraphStorage, id: &u128) -> Option<Vec<u8>> {
        storage
            .with_read_txn(|rtxn| {
                Ok(storage
                    .lmdb_nodes_db()
                    .unwrap()
                    .get(rtxn, HelixGraphStorage::node_key(id))?
                    .map(|b| b.to_vec()))
            })
            .unwrap()
    }

    /// Read the raw secondary-index value bytes (the stored node id) for a given
    /// index name + indexed `Value`, via the heed secondary-index DB handle.
    fn heed_secondary_index_bytes(
        storage: &HelixGraphStorage,
        index: &str,
        value: &Value,
    ) -> Option<Vec<u8>> {
        storage
            .with_read_txn(|rtxn| {
                let db = storage
                    .secondary_indices
                    .get(index)
                    .unwrap()
                    .as_ref()
                    .unwrap();
                let key = HelixGraphStorage::stable_index_key_for_value(value)?;
                Ok(db.get(rtxn, &key)?.map(|b| b.to_vec()))
            })
            .unwrap()
    }

    /// (1) Same-store readback: a node written via `create_node_be` is readable
    /// through the heed `get_node` and resolvable through the heed
    /// `get_node_by_secondary_index`, and the node counter advanced.
    /// Multi-op batch correctness: two nodes created in ONE write batch via
    /// `create_node_be` must leave the SAME metadata (node counter) as two heed
    /// `create_node` calls in one write txn. This is exactly what the old
    /// fresh-snapshot RMW under-counted (it would write counter=1); the
    /// `get_for_update` read-your-writes fix makes it counter=2.
    #[test]
    fn create_node_be_counter_correct_for_multi_op_batch() {
        let mk = || {
            let dir = tempfile::TempDir::new().unwrap();
            let storage =
                HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
            (dir, storage)
        };

        // Store A: two heed creates in one write txn (ground truth).
        let (_da, a) = mk();
        a.with_write_txn(|wtxn| {
            a.create_node(wtxn, "L", Vec::<(String, Value)>::new(), None, Some(1u128))?;
            a.create_node(wtxn, "L", Vec::<(String, Value)>::new(), None, Some(2u128))?;
            Ok(())
        })
        .unwrap();

        // Store B: two create_node_be in one backend write batch.
        let (_db, b) = mk();
        let mut w = b.backend.begin_write().unwrap();
        b.create_node_be(
            &mut w,
            "L",
            Vec::<(String, Value)>::new(),
            None,
            Some(1u128),
        )
        .unwrap();
        b.create_node_be(
            &mut w,
            "L",
            Vec::<(String, Value)>::new(),
            None,
            Some(2u128),
        )
        .unwrap();
        b.backend.commit(w).unwrap();

        // Both nodes landed.
        assert!(b.with_read_txn(|rtxn| b.get_node(rtxn, &1u128)).is_ok());
        assert!(b.with_read_txn(|rtxn| b.get_node(rtxn, &2u128)).is_ok());

        // Compare the decoded NODE COUNTER, not raw metadata bytes: the whole
        // record also carries wall-clock created/updated timestamps that differ
        // between the two stores. The counter is what the multi-op batch must
        // get right (the old fresh-snapshot RMW would leave it at 1).
        let node_count = |s: &HelixGraphStorage| -> u64 {
            let bytes = s
                .with_read_txn(|rtxn| {
                    Ok(s.lmdb_metadata_db()
                        .unwrap()
                        .get(rtxn, "current")?
                        .map(|x| x.to_vec()))
                })
                .unwrap()
                .expect("metadata current record present");
            deserialize_metadata(&bytes).unwrap().stats.node_count
        };
        assert_eq!(node_count(&a), 2, "heed baseline must count 2 nodes");
        assert_eq!(
            node_count(&b),
            node_count(&a),
            "create_node_be must count a 2-node batch as 2 (read-your-writes), not 1"
        );
    }

    #[test]
    fn create_node_be_readback_via_heed() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut storage =
            HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
        storage.create_secondary_index("k").unwrap();

        let id: u128 = 0x1234_5678;
        let value = Value::String("alice".to_string());

        let nodes_before = storage
            .with_read_txn(|rtxn| storage.get_metadata(rtxn))
            .unwrap()
            .stats
            .node_count;

        let mut w = storage.backend.begin_write().unwrap();
        let created = storage
            .create_node_be(
                &mut w,
                "Label",
                vec![("k".to_string(), value.clone())],
                Some(&["k".to_string()]),
                Some(id),
            )
            .unwrap();
        storage.backend.commit(w).unwrap();
        assert_eq!(created.id, id);

        // Node round-trips through the heed read path.
        let node = storage
            .with_read_txn(|rtxn| storage.get_node(rtxn, &id))
            .unwrap();
        assert_eq!(node.id, id);
        assert_eq!(node.label, "Label");
        assert_eq!(node.properties.get("k"), Some(&value));

        // Indexed lookup resolves to the same node via the heed index path.
        let by_index = storage
            .with_read_txn(|rtxn| storage.get_node_by_secondary_index(rtxn, "k", &value))
            .unwrap();
        assert_eq!(by_index.id, id);

        // The node counter advanced by exactly one.
        let nodes_after = storage
            .with_read_txn(|rtxn| storage.get_metadata(rtxn))
            .unwrap()
            .stats
            .node_count;
        assert_eq!(nodes_after, nodes_before + 1);
    }

    /// (2) Twin-store byte equality: the SAME node written via heed `create_node`
    /// into store A and via `create_node_be` into store B yields byte-identical
    /// `nodes_db` value bytes AND byte-identical secondary-index value bytes.
    #[test]
    fn create_node_be_byte_identical_to_heed() {
        let id: u128 = 0x0BAD_F00D_DEAD_BEEF;
        let label = "Person";
        let value = Value::String("alice".to_string());
        let props = || vec![("k".to_string(), value.clone())];
        let indices = ["k".to_string()];

        // Store A: heed create_node.
        let dir_a = tempfile::TempDir::new().unwrap();
        let mut store_a =
            HelixGraphStorage::new(dir_a.path().to_str().unwrap(), test_config()).unwrap();
        store_a.create_secondary_index("k").unwrap();
        store_a
            .with_write_txn(|wtxn| {
                store_a.create_node(wtxn, label, props(), Some(&indices), Some(id))?;
                Ok(())
            })
            .unwrap();

        // Store B: backend create_node_be.
        let dir_b = tempfile::TempDir::new().unwrap();
        let mut store_b =
            HelixGraphStorage::new(dir_b.path().to_str().unwrap(), test_config()).unwrap();
        store_b.create_secondary_index("k").unwrap();
        let mut w = store_b.backend.begin_write().unwrap();
        store_b
            .create_node_be(&mut w, label, props(), Some(&indices), Some(id))
            .unwrap();
        store_b.backend.commit(w).unwrap();

        // Node value bytes are byte-identical.
        let bytes_a = heed_node_bytes(&store_a, &id);
        let bytes_b = heed_node_bytes(&store_b, &id);
        assert!(bytes_a.is_some(), "store A must have the node row");
        assert_eq!(
            bytes_a, bytes_b,
            "nodes_db value bytes must be byte-identical across heed and backend writes"
        );

        // Secondary-index value bytes (the stored node id) are byte-identical.
        let idx_a = heed_secondary_index_bytes(&store_a, "k", &value);
        let idx_b = heed_secondary_index_bytes(&store_b, "k", &value);
        assert!(idx_a.is_some(), "store A must have the index entry");
        assert_eq!(
            idx_a,
            Some(id.to_be_bytes().to_vec()),
            "index entry must store the node id big-endian"
        );
        assert_eq!(
            idx_a, idx_b,
            "secondary-index value bytes must be byte-identical across both write paths"
        );
    }

    /// An unregistered secondary index errors the same way `create_node` does,
    /// and a registered index whose property is absent on the node also errors.
    #[test]
    fn create_node_be_index_validation_matches_heed() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut storage =
            HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
        storage.create_secondary_index("k").unwrap();

        // Unregistered index name.
        let mut w = storage.backend.begin_write().unwrap();
        let err = storage.create_node_be(
            &mut w,
            "Label",
            vec![("k".to_string(), Value::String("x".to_string()))],
            Some(&["missing".to_string()]),
            Some(1),
        );
        assert!(err.is_err());
        drop(w);

        // Registered index but the node lacks that property.
        let mut w = storage.backend.begin_write().unwrap();
        let err = storage.create_node_be(
            &mut w,
            "Label",
            vec![("other".to_string(), Value::String("x".to_string()))],
            Some(&["k".to_string()]),
            Some(2),
        );
        assert!(err.is_err());
    }

    /// `drop_node_be` removes the node row and decrements the node counter,
    /// matching heed `drop_node` for a simple (no-edge, no-payload-index,
    /// no-name-multi-index) node. Verified by reading back through heed.
    #[test]
    fn drop_node_be_removes_node_and_decrements_counter() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut storage =
            HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
        storage.create_secondary_index("k").unwrap();

        let id: u128 = 0xFEED;
        // Seed via heed create_node so the counter starts at 1.
        storage
            .with_write_txn(|wtxn| {
                storage.create_node(
                    wtxn,
                    "Label",
                    vec![("k".to_string(), Value::String("x".to_string()))],
                    Some(&["k".to_string()]),
                    Some(id),
                )?;
                Ok(())
            })
            .unwrap();
        let before = storage
            .with_read_txn(|rtxn| storage.get_metadata(rtxn))
            .unwrap()
            .stats
            .node_count;
        assert_eq!(before, 1);
        assert!(heed_node_bytes(&storage, &id).is_some());

        // Drop via the backend path.
        let mut w = storage.backend.begin_write().unwrap();
        storage.drop_node_be(&mut w, &id).unwrap();
        storage.backend.commit(w).unwrap();

        // Node row is gone and the counter is back to zero.
        assert!(heed_node_bytes(&storage, &id).is_none());
        assert!(storage
            .with_read_txn(|rtxn| storage.get_node(rtxn, &id))
            .is_err());
        let after = storage
            .with_read_txn(|rtxn| storage.get_metadata(rtxn))
            .unwrap()
            .stats
            .node_count;
        assert_eq!(after, 0);
    }

    /// Dropping a missing node is a no-op on the counter (the node row never
    /// existed), matching `drop_node`'s `if deleted { ... }` guard.
    #[test]
    fn drop_node_be_missing_is_counter_noop() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();

        let before = storage
            .with_read_txn(|rtxn| storage.get_metadata(rtxn))
            .unwrap()
            .stats
            .node_count;

        let mut w = storage.backend.begin_write().unwrap();
        storage.drop_node_be(&mut w, &0xDEAD_BEEF_u128).unwrap();
        storage.backend.commit(w).unwrap();

        let after = storage
            .with_read_txn(|rtxn| storage.get_metadata(rtxn))
            .unwrap()
            .stats
            .node_count;
        assert_eq!(
            before, after,
            "dropping a missing node must not move the counter"
        );
    }

    /// Ghost payload-index regression (LSM/`_be` counterpart of
    /// `upsert::tests::test_drop_node_removes_nested_payload_index_ghost_entry`):
    /// `drop_node_be`'s de-index step used a flat `node.properties.get(idx_name)`
    /// lookup, which returns `None` for a dotted index name like `metadata.repo`,
    /// so the dup entry leaked forever. Fixed via `payload_value_for_key`.
    #[test]
    fn drop_node_be_removes_nested_payload_index_ghost_entry() {
        use crate::helix_engine::storage_core::metadata::PayloadIndexSchema;
        use crate::helix_engine::storage_core::upsert::NodeUpsert;

        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
        storage
            .create_payload_index("metadata.repo", PayloadIndexSchema::Keyword)
            .unwrap();

        let id: u128 = 0xBEEF;
        let upsert = NodeUpsert {
            id,
            label: "Symbol".to_string(),
            properties: std::collections::HashMap::from([(
                "metadata".to_string(),
                Value::Object(std::collections::HashMap::from([(
                    "repo".to_string(),
                    Value::String("nested-repo".to_string()),
                )])),
            )]),
        };
        let mut w = storage.backend.begin_write().unwrap();
        storage.upsert_node_be(&mut w, &upsert).unwrap();
        storage.backend.commit(w).unwrap();

        let r = storage.backend.begin_read().unwrap();
        assert_eq!(
            storage
                .get_nodes_by_payload_value_be(
                    &r,
                    "metadata.repo",
                    &Value::String("nested-repo".to_string())
                )
                .unwrap(),
            vec![id],
            "sanity: index entry exists before delete"
        );
        drop(r);

        let mut w = storage.backend.begin_write().unwrap();
        storage.drop_node_be(&mut w, &id).unwrap();
        storage.backend.commit(w).unwrap();

        let r = storage.backend.begin_read().unwrap();
        assert!(
            storage
                .get_nodes_by_payload_value_be(
                    &r,
                    "metadata.repo",
                    &Value::String("nested-repo".to_string())
                )
                .unwrap()
                .is_empty(),
            "nested-field index must not leak a ghost dup entry after drop_node_be"
        );
    }
}
