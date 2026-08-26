use std::collections::HashMap;

use crate::{
    helix_engine::{
        graph_core::traversal_iter::RwTraversalIterator,
        storage_core::backend::{Namespace, StorageBackend},
        storage_core::storage_core::HelixGraphStorage,
        types::GraphError,
    },
    protocol::{
        items::{Edge, SerializedEdge},
        label_hash::hash_label,
    },
};

use super::super::tr_val::TraversalVal;

pub struct BulkAddE {
    inner: std::iter::Once<Result<TraversalVal, GraphError>>,
}

impl Iterator for BulkAddE {
    type Item = Result<TraversalVal, GraphError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

pub trait BulkAddEAdapter<'a, 'b>:
    Iterator<Item = Result<TraversalVal, GraphError>> + Sized
{
    fn bulk_add_e(
        self,
        edges: Vec<(u128, u128, u128)>,
        should_check_nodes: bool,
        chunk_size: usize,
    ) -> RwTraversalIterator<'a, 'b, impl Iterator<Item = Result<TraversalVal, GraphError>>>;
}

impl<'a, 'b, I: Iterator<Item = Result<TraversalVal, GraphError>>> BulkAddEAdapter<'a, 'b>
    for RwTraversalIterator<'a, 'b, I>
{
    fn bulk_add_e(
        self,
        mut edges: Vec<(u128, u128, u128)>,
        should_check_nodes: bool,
        _chunk_size: usize,
    ) -> RwTraversalIterator<'a, 'b, impl Iterator<Item = Result<TraversalVal, GraphError>>> {
        // sort by id
        edges.sort_unstable_by(|(_, _, id), (_, _, id_)| id.cmp(id_));
        let label_hash = hash_label("knows", None);

        // Field flip: write all edges + out/in adjacency into the traversal's
        // single shared `AnyWrite` batch via the seam (engine-neutral). Same
        // keys/values as before; bulk_add_e writes NO edge-path index and does
        // NOT bump the edge counter (matches prior behavior). Drops the LMDB
        // APPEND fast path (re-add via a hinted seam put if bench perf needs it).
        let result = (|| -> Result<TraversalVal, GraphError> {
            for (e_from, e_to, e_id) in &edges {
                if should_check_nodes
                    && (self
                        .storage
                        .backend
                        .get_for_update(&*self.txn, Namespace::Nodes, &e_from.to_be_bytes(), |n| {
                            n.is_none()
                        })
                        .map_err(|e| GraphError::New(e.to_string()))?
                        || self
                            .storage
                            .backend
                            .get_for_update(
                                &*self.txn,
                                Namespace::Nodes,
                                &e_to.to_be_bytes(),
                                |n| n.is_none(),
                            )
                            .map_err(|e| GraphError::New(e.to_string()))?)
                {
                    return Err(GraphError::NodeNotFound);
                }
                let edge = Edge {
                    id: *e_id,
                    label: "knows".to_string(),
                    properties: HashMap::new(),
                    from_node: *e_from,
                    to_node: *e_to,
                };
                let bytes = SerializedEdge::encode_edge(&edge)?;
                self.storage
                    .backend
                    .put(self.txn, Namespace::Edges, &e_id.to_be_bytes(), &bytes)
                    .map_err(|e| GraphError::New(e.to_string()))?;
            }

            for (from_node, to_node, id) in &edges {
                self.storage
                    .backend
                    .put_dup(
                        self.txn,
                        Namespace::OutEdges,
                        &HelixGraphStorage::out_edge_key(from_node, &label_hash),
                        &HelixGraphStorage::pack_edge_data(to_node, id),
                    )
                    .map_err(|e| GraphError::New(e.to_string()))?;
            }

            for (from_node, to_node, id) in &edges {
                self.storage
                    .backend
                    .put_dup(
                        self.txn,
                        Namespace::InEdges,
                        &HelixGraphStorage::in_edge_key(to_node, &label_hash),
                        &HelixGraphStorage::pack_edge_data(from_node, id),
                    )
                    .map_err(|e| GraphError::New(e.to_string()))?;
            }

            Ok(TraversalVal::Empty)
        })();

        RwTraversalIterator {
            inner: std::iter::once(result), // TODO: change to support adding multiple edges
            storage: self.storage,
            txn: self.txn,
        }
    }
}
