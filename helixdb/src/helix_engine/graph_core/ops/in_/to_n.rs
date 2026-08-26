use crate::helix_engine::storage_core::backend_any::AnyRead;
use crate::helix_engine::{
    graph_core::{ops::tr_val::TraversalVal, traversal_iter::RoTraversalIterator},
    storage_core::{storage_core::HelixGraphStorage, storage_methods::StorageMethods},
    types::GraphError,
};
use heed3::RwTxn;
use std::sync::Arc;

pub struct ToNIterator<'a, I, T> {
    iter: I,
    storage: Arc<HelixGraphStorage>,
    txn: &'a T,
}

// implementing iterator for OutIterator
impl<'a, I> Iterator for ToNIterator<'a, I, AnyRead<'a>>
where
    I: Iterator<Item = Result<TraversalVal, GraphError>>,
{
    type Item = Result<TraversalVal, GraphError>;

    /// Returns the next outgoing node by decoding the edge id and then getting the edge and node
    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some(item) => match item {
                Ok(TraversalVal::Edge(item)) => {
                    // Read through the backend twin (`get_node_be`) over this
                    // traversal's single shared read snapshot. Byte-identical to
                    // `get_node` on LMDB; backend-routed via the storage seam.
                    Some(Ok(TraversalVal::Node(
                        self.storage.get_node_be(self.txn, &item.to_node).unwrap(),
                    ))) // TODO: handle unwrap
                }
                _ => return None,
            },
            None => None,
        }
    }
}

impl<'a, I> Iterator for ToNIterator<'a, I, RwTxn<'a>>
where
    I: Iterator<Item = Result<TraversalVal, GraphError>>,
{
    type Item = Result<TraversalVal, GraphError>;

    /// Returns the next outgoing node by decoding the edge id and then getting the edge and node
    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some(item) => match item {
                Ok(TraversalVal::Edge(item)) => Some(Ok(TraversalVal::Node(
                    self.storage.get_node(self.txn, &item.to_node).unwrap(),
                ))), // TODO: handle unwrap
                _ => return None,
            },
            None => None,
        }
    }
}
pub trait ToNAdapter<'a, T>: Iterator<Item = Result<TraversalVal, GraphError>> + Sized {
    fn to_n(
        self,
    ) -> RoTraversalIterator<'a, impl Iterator<Item = Result<TraversalVal, GraphError>>>;
}

impl<'a, I: Iterator<Item = Result<TraversalVal, GraphError>> + 'a> ToNAdapter<'a, AnyRead<'a>>
    for RoTraversalIterator<'a, I>
{
    fn to_n(
        self,
    ) -> RoTraversalIterator<'a, impl Iterator<Item = Result<TraversalVal, GraphError>>> {
        let iter = ToNIterator {
            iter: self.inner,
            storage: Arc::clone(&self.storage),
            txn: self.txn,
        };
        RoTraversalIterator {
            inner: iter,
            storage: self.storage,
            txn: self.txn,
        }
    }
}

// impl<'a, I: Iterator<Item = Result<TraversalVal, GraphError>> + 'a> InVAdapter<'a, RwTxn<'a>>
//     for RwTraversalIterator<'a, I>
// {
//     fn in_v(
//         self,
//         db: Arc<HelixGraphStorage>,
//         txn: &'a RwTxn<'a>,
//     ) -> InVIterator<'a, Self, RwTxn<'a>> {
//         InVIterator {
//             iter: self,
//             storage: db,
//             txn,
//         }
//     }
// }
