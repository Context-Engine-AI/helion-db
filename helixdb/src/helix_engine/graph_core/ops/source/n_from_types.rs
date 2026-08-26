use crate::{
    helix_engine::{
        graph_core::traversal_iter::RoTraversalIterator,
        storage_core::backend::{KeyRange, Namespace, StorageBackend},
        types::GraphError,
    },
    protocol::items::SerializedNode,
};

use super::super::tr_val::TraversalVal;

/// Full node-table scan filtered by label, routed through `backend.scan` over
/// the traversal's existing read snapshot. Eagerly materialised per label (the
/// seam has no lending iterator); on LMDB the matched nodes and their
/// ascending-key ordering are byte-identical to the prior heed `nodes_db` scan.
/// Error semantics preserved from the original `NFromTypes` iterator: a decode
/// failure yields `ConversionError(e.to_string())`, label mismatches are skipped.
pub struct NFromTypes {
    iter: std::vec::IntoIter<Result<TraversalVal, GraphError>>,
}

impl Iterator for NFromTypes {
    type Item = Result<TraversalVal, GraphError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

pub trait NFromTypesAdapter<'a>: Iterator<Item = Result<TraversalVal, GraphError>> + Sized {
    fn n_from_types(
        self,
        types: &'a [&'a str],
    ) -> RoTraversalIterator<'a, impl Iterator<Item = Result<TraversalVal, GraphError>>>;
}
impl<'a, I: Iterator<Item = Result<TraversalVal, GraphError>>> NFromTypesAdapter<'a>
    for RoTraversalIterator<'a, I>
{
    fn n_from_types(
        self,
        types: &'a [&'a str],
    ) -> RoTraversalIterator<'a, impl Iterator<Item = Result<TraversalVal, GraphError>>> {
        let db = self.storage.clone();
        let txn = self.txn;
        let iter = types.iter().flat_map(move |label| {
            // One full node scan per label, routed through the backend seam over
            // the traversal's single shared snapshot. Matched nodes are buffered
            // (eager); byte-identical content/order to the heed scan on LMDB.
            let mut items: Vec<Result<TraversalVal, GraphError>> = Vec::new();
            let scan = db
                .backend
                .scan(txn, Namespace::Nodes, KeyRange::all(), |key, value| {
                    let id = match <[u8; 16]>::try_from(key) {
                        Ok(b) => u128::from_be_bytes(b),
                        Err(_) => {
                            items.push(Err(GraphError::SliceLengthError));
                            return true;
                        }
                    };
                    match SerializedNode::decode_node(value, id) {
                        Ok(node) => {
                            if node.label.as_str() == *label {
                                items.push(Ok(TraversalVal::Node(node)));
                            }
                        }
                        Err(e) => items.push(Err(GraphError::ConversionError(e.to_string()))),
                    }
                    true
                });
            if let Err(e) = scan {
                items.push(Err(GraphError::New(e.to_string())));
            }
            items.into_iter()
        });
        RoTraversalIterator {
            inner: iter,
            storage: self.storage,
            txn: self.txn,
        }
    }
}
