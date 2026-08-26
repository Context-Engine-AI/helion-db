use std::sync::Arc;

use heed3::RwTxn;

use crate::helix_engine::{
    graph_core::traversal_iter::RoTraversalIterator,
    storage_core::backend_any::AnyRead,
    storage_core::{storage_core::HelixGraphStorage, storage_methods::StorageMethods},
    types::GraphError,
};

use super::super::tr_val::{Traversable, TraversalVal};

pub struct OutEdgesIterator<'a, T> {
    /// Outgoing edge ids for one source node under the requested label, in LMDB
    /// dup-sorted order — eagerly materialised by `out_edges_be` (the backend
    /// twin of the heed `get_duplicates` enumeration). The seam has no lending
    /// iterator, so one source node's fan-out is buffered.
    edge_ids: std::vec::IntoIter<u128>,
    storage: Arc<HelixGraphStorage>,
    txn: &'a T,
}

// implementing iterator for OutIterator
impl<'a> Iterator for OutEdgesIterator<'a, AnyRead<'a>> {
    type Item = Result<TraversalVal, GraphError>;

    /// Returns the next outgoing edge, loaded through the backend twin
    /// (`get_edge_be`) over this traversal's single shared read snapshot.
    /// Byte-identical to `get_edge` on LMDB.
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(edge_id) = self.edge_ids.next() {
            if let Ok(edge) = self.storage.get_edge_be(self.txn, &edge_id) {
                return Some(Ok(TraversalVal::Edge(edge)));
            }
        }
        None
    }
}
impl<'a> Iterator for OutEdgesIterator<'a, RwTxn<'a>> {
    type Item = Result<TraversalVal, GraphError>;

    /// Write-txn variant. No live constructor — the Rw `out_e` adapter below is
    /// not implemented — so this reads via heed `get_edge` to observe same-txn
    /// writes if it is ever wired up.
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(edge_id) = self.edge_ids.next() {
            if let Ok(edge) = self.storage.get_edge(self.txn, &edge_id) {
                return Some(Ok(TraversalVal::Edge(edge)));
            }
        }
        None
    }
}
pub trait OutEdgesAdapter<'a, T>:
    Iterator<Item = Result<TraversalVal, GraphError>> + Sized
{
    fn out_e(
        self,
        edge_label: &'a str,
    ) -> RoTraversalIterator<'a, impl Iterator<Item = Result<TraversalVal, GraphError>>>;
}

impl<'a, I: Iterator<Item = Result<TraversalVal, GraphError>> + 'a> OutEdgesAdapter<'a, AnyRead<'a>>
    for RoTraversalIterator<'a, I>
{
    fn out_e(
        self,
        edge_label: &'a str,
    ) -> RoTraversalIterator<'a, impl Iterator<Item = Result<TraversalVal, GraphError>>> {
        // iterate through the iterator and create a new iterator on the out edges
        let db = Arc::clone(&self.storage);
        let storage = Arc::clone(&self.storage);
        let txn = self.txn;
        let iter = self
            .inner
            .filter_map(move |item| {
                // Enumerate this node's outgoing edge ids through the backend
                // twin (`out_edges_be`), reading the SAME out_edges DUP_SORT key
                // in dup-sorted order as the heed `get_duplicates` path, over the
                // traversal's single shared read snapshot. Eager: one source
                // node's fan-out is buffered (seam has no lending iterator);
                // happy-path byte-identical on LMDB.
                match db.out_edges_be(txn, &item.unwrap().id(), edge_label) {
                    Ok(edge_ids) => Some(OutEdgesIterator {
                        edge_ids: edge_ids.into_iter(),
                        storage: Arc::clone(&db),
                        txn,
                    }),
                    Err(e) => {
                        println!("Error getting out edges: {:?}", e);
                        None
                    }
                }
            })
            .flatten();
        RoTraversalIterator {
            inner: iter,
            storage,
            txn,
        }
    }
}

// impl<'a, I: Iterator<Item = Result<TraversalVal, GraphError>> + 'a> OutEdgesAdapter<'a, RwTxn<'a>>
//     for RwTraversalIterator<'a, I>
// {
//     fn out_edges(
//         self,
//         edge_label: &'a str,
//     ) -> OutEdges<
//         'a,
//         Self,
//         impl FnMut(Result<TraversalVal, GraphError>) -> OutEdgesIterator<'a, RwTxn<'a>>,
//         RwTxn<'a>,
//     > {
//         {
//             // iterate through the iterator and create a new iterator on the out edges
//             let db = Arc::clone(&self.storage);
//             let storage = Arc::clone(&self.storage);
//             let txn = self.txn;
//             let iter = self
//                 .map(move |item| {
//                     let prefix = HelixGraphStorage::out_edge_key(item.unwrap().id(), "");
//                     let iter = db
//                         .out_edges_db
//                         .lazily_decode_data()
//                         .prefix_iter(txn, &prefix)
//                         .unwrap();

//                     OutEdgesIterator {
//                         iter,
//                         storage: Arc::clone(&storage),
//                         txn,
//                         edge_label,
//                     }
//                 })
//                 .flatten();
//             OutEdges { iter }
//         }
//     }
// }
