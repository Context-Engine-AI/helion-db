use super::super::tr_val::TraversalVal;
use crate::helix_engine::storage_core::backend::{Namespace, StorageBackend};
use crate::{
    helix_engine::{
        graph_core::traversal_iter::RwTraversalIterator,
        storage_core::storage_core::HelixGraphStorage, types::GraphError, vector_core::hnsw::HNSW,
    },
    protocol::{
        items::{v6_uuid, Edge, SerializedEdge},
        label_hash::hash_label,
        value::Value,
    },
};

pub enum EdgeType {
    Vec,
    Std,
}
pub struct AddE {
    inner: std::iter::Once<Result<TraversalVal, GraphError>>,
}

impl Iterator for AddE {
    type Item = Result<TraversalVal, GraphError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

pub trait AddEAdapter<'a, 'b>: Iterator<Item = Result<TraversalVal, GraphError>> + Sized {
    fn add_e(
        self,
        label: &'a str,
        properties: Vec<(String, Value)>,
        id: Option<u128>,
        from_node: u128,
        to_node: u128,
        should_check: bool,
        edge_type: EdgeType,
    ) -> RwTraversalIterator<'a, 'b, impl Iterator<Item = Result<TraversalVal, GraphError>>>;

    fn node_vec_exists(&self, node_vec_id: &u128, edge_type: EdgeType) -> bool;
}

impl<'a, 'b, I: Iterator<Item = Result<TraversalVal, GraphError>>> AddEAdapter<'a, 'b>
    for RwTraversalIterator<'a, 'b, I>
{
    fn add_e(
        self,
        label: &'a str,
        properties: Vec<(String, Value)>,
        id: Option<u128>,
        from_node: u128,
        to_node: u128,
        should_check: bool,
        edge_type: EdgeType,
    ) -> RwTraversalIterator<'a, 'b, impl Iterator<Item = Result<TraversalVal, GraphError>>> {
        // Field flip: write edge bytes + out/in adjacency into the traversal's
        // single shared `AnyWrite` batch via the backend seam. Preserves the
        // existing `add_e` semantics exactly (existence check only for Std edges;
        // no edge-path index / edge counter — `add_e` is the light edge insert,
        // unlike `create_edge`). Byte-identical on LMDB (same keys/values as the
        // prior `edges_db.put` / `out_edges_db.put` / `in_edges_db.put`).

        // Node-existence check (Std only): fail before any writes.
        if let EdgeType::Std = edge_type {
            if should_check
                && !(self.node_vec_exists(&from_node, EdgeType::Std)
                    && self.node_vec_exists(&to_node, EdgeType::Std))
            {
                return RwTraversalIterator {
                    inner: std::iter::once(Err(GraphError::NodeNotFound)),
                    storage: self.storage,
                    txn: self.txn,
                };
            }
        }

        let edge = Edge {
            id: id.unwrap_or(v6_uuid()),
            label: label.to_string(),
            properties: properties.into_iter().collect(),
            from_node,
            to_node,
        };
        let label_hash = hash_label(edge.label.as_str(), None);

        let result = (|| -> Result<TraversalVal, GraphError> {
            let bytes = SerializedEdge::encode_edge(&edge)?;
            self.storage
                .backend
                .put(self.txn, Namespace::Edges, &edge.id.to_be_bytes(), &bytes)
                .map_err(|e| GraphError::New(e.to_string()))?;
            self.storage
                .backend
                .put_dup(
                    self.txn,
                    Namespace::OutEdges,
                    &HelixGraphStorage::out_edge_key(&from_node, &label_hash),
                    &HelixGraphStorage::pack_edge_data(&to_node, &edge.id),
                )
                .map_err(|e| GraphError::New(e.to_string()))?;
            self.storage
                .backend
                .put_dup(
                    self.txn,
                    Namespace::InEdges,
                    &HelixGraphStorage::in_edge_key(&to_node, &label_hash),
                    &HelixGraphStorage::pack_edge_data(&from_node, &edge.id),
                )
                .map_err(|e| GraphError::New(e.to_string()))?;
            Ok(TraversalVal::Edge(edge))
        })();

        RwTraversalIterator {
            inner: std::iter::once(result),
            storage: self.storage,
            txn: self.txn,
        }
    }

    fn node_vec_exists(&self, node_vec_id: &u128, edge_type: EdgeType) -> bool {
        match edge_type {
            // Read-your-writes existence check through the shared write batch
            // (engine-neutral). Byte-identical key (u128 big-endian) to the
            // prior `nodes_db.get(node_key(..))`; existence == Some.
            EdgeType::Std => self
                .storage
                .backend
                .get_for_update(
                    &*self.txn,
                    Namespace::Nodes,
                    &node_vec_id.to_be_bytes(),
                    |node| node.is_some(),
                )
                .unwrap_or(false),
            // Vector existence still reads the heed vector core (Unit 4). On LMDB
            // read-your-writes via the shared txn; on LSM there is no heed txn,
            // so report absent.
            EdgeType::Vec => match self.txn.lmdb_ro() {
                Some(rw) => {
                    let rd = self.storage.vectors.backend.read_borrowed(rw);
                    self.storage
                        .vectors
                        .get_vector(&rd, *node_vec_id, 0, false)
                        .is_ok()
                }
                None => false,
            },
        }
    }
}
