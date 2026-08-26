use std::sync::Arc;

use heed3::RwTxn;

use crate::helix_engine::{
    graph_core::traversal_iter::RoTraversalIterator,
    storage_core::backend_any::AnyRead,
    storage_core::{storage_core::HelixGraphStorage, storage_methods::StorageMethods},
    types::GraphError,
};

use super::super::tr_val::{Traversable, TraversalVal};

pub struct OutNodesIterator<'a, T> {
    /// Peer (to-node) ids for one source node's out edges of the requested
    /// label, in LMDB dup-sorted order — eagerly materialised by `out_adj_be`
    /// (the backend twin of the heed `get_duplicates` enumeration). The seam
    /// has no lending iterator, so one source node's fan-out is buffered.
    peers: std::vec::IntoIter<u128>,
    storage: Arc<HelixGraphStorage>,
    txn: &'a T,
}

// implementing iterator for OutIterator
impl<'a> Iterator for OutNodesIterator<'a, AnyRead<'a>> {
    type Item = Result<TraversalVal, GraphError>;

    /// Returns the next outgoing node, loaded through the backend twin
    /// (`get_node_be`) over this traversal's single shared read snapshot.
    /// Byte-identical to `get_node` on LMDB.
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node_id) = self.peers.next() {
            if let Ok(node) = self.storage.get_node_be(self.txn, &node_id) {
                return Some(Ok(TraversalVal::Node(node)));
            }
        }
        None
    }
}
impl<'a> Iterator for OutNodesIterator<'a, RwTxn<'a>> {
    type Item = Result<TraversalVal, GraphError>;

    /// Write-txn variant. No live constructor — the Rw `out` adapter below is
    /// not implemented — so this reads via heed `get_node` to observe same-txn
    /// writes if it is ever wired up.
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node_id) = self.peers.next() {
            if let Ok(node) = self.storage.get_node(self.txn, &node_id) {
                return Some(Ok(TraversalVal::Node(node)));
            }
        }
        None
    }
}

pub trait OutAdapter<'a, T>: Iterator<Item = Result<TraversalVal, GraphError>> + Sized {
    fn out(
        self,
        edge_label: &'a str,
    ) -> RoTraversalIterator<'a, impl Iterator<Item = Result<TraversalVal, GraphError>>>;
    // where
    //     OutNodesIterator<'a, T>: std::iter::Iterator,
    //     T: 'a;
}

impl<'a, I: Iterator<Item = Result<TraversalVal, GraphError>> + 'a> OutAdapter<'a, AnyRead<'a>>
    for RoTraversalIterator<'a, I>
{
    fn out(
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
                // Enumerate this node's out-edge peers through the backend twin
                // (`out_adj_be`), which reads the SAME out_edges DUP_SORT key in
                // dup-sorted order as the heed `get_duplicates` path, over the
                // traversal's single shared read snapshot. Eagerly materialised:
                // the seam has no lending iterator, so one source node's edge
                // fan-out is buffered (vs. the prior lazy heed iter).
                match db.out_adj_be(txn, &item.unwrap().id(), edge_label) {
                    Ok(pairs) => Some(OutNodesIterator {
                        peers: pairs
                            .into_iter()
                            .map(|(peer, _edge_id)| peer)
                            .collect::<Vec<_>>()
                            .into_iter(),
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

// impl<'a, I: Iterator<Item = Result<TraversalVal, GraphError>> + 'a> OutAdapter<'a, RwTxn<'a>>
//     for RwTraversalIterator<'a, I>
// {
//     fn out(
//         self,
//         edge_label: &'a str,
//     ) -> OutNodes<
//         'a,
//         Self,
//         impl FnMut(Result<TraversalVal, GraphError>) -> OutNodesIterator<'a, RwTxn<'a>>,
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

//                     OutNodesIterator {
//                         iter,
//                         storage: Arc::clone(&storage),
//                         txn,
//                         edge_label,
//                     }
//                 })
//                 .flatten();
//             OutNodes { iter }
//         }
//     }
// }
