use std::sync::Arc;

use heed3::RwTxn;

use crate::helix_engine::{
    graph_core::traversal_iter::RoTraversalIterator,
    storage_core::backend_any::AnyRead,
    storage_core::{storage_core::HelixGraphStorage, storage_methods::StorageMethods},
    types::GraphError,
};

use super::super::tr_val::{Traversable, TraversalVal};

pub struct InNodesIterator<'a, T> {
    /// Peer (from-node) ids for one node's incoming edges of the requested
    /// label, in LMDB dup-sorted order — eagerly materialised by `in_adj_be`
    /// (the backend twin of the heed `get_duplicates` enumeration). The seam
    /// has no lending iterator, so one node's in-degree fan-out is buffered.
    peers: std::vec::IntoIter<u128>,
    storage: Arc<HelixGraphStorage>,
    txn: &'a T,
}

// implementing iterator for OutIterator
impl<'a> Iterator for InNodesIterator<'a, AnyRead<'a>> {
    type Item = Result<TraversalVal, GraphError>;

    /// Returns the next incoming node, loaded through the backend twin
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
impl<'a> Iterator for InNodesIterator<'a, RwTxn<'a>> {
    type Item = Result<TraversalVal, GraphError>;

    /// Write-txn variant. No live constructor — the Rw `in_` adapter below is
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

pub trait InAdapter<'a, T>: Iterator<Item = Result<TraversalVal, GraphError>> + Sized {
    fn in_(
        self,
        edge_label: &'a str,
    ) -> RoTraversalIterator<'a, impl Iterator<Item = Result<TraversalVal, GraphError>>>;
}

impl<'a, I: Iterator<Item = Result<TraversalVal, GraphError>> + 'a> InAdapter<'a, AnyRead<'a>>
    for RoTraversalIterator<'a, I>
{
    fn in_(
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
                // Enumerate this node's incoming-edge peers through the backend
                // twin (`in_adj_be`), reading the SAME in_edges DUP_SORT key in
                // dup-sorted order as the heed `get_duplicates` path, over the
                // traversal's single shared read snapshot. (in_edge_key and
                // out_edge_key are byte-identical, so the key matches.) Eager:
                // one node's in-degree fan-out is buffered (seam has no lending
                // iterator); happy-path byte-identical on LMDB.
                match db.in_adj_be(txn, &item.unwrap().id(), edge_label) {
                    Ok(pairs) => Some(InNodesIterator {
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

// impl<'a, I: Iterator<Item = Result<TraversalVal, GraphError>> + 'a> InAdapter<'a, RwTxn<'a>>
//     for RwTraversalIterator<'a, I>
// {
//     fn in_(
//         self,
//         edge_label: &'a str,
//     ) -> InNodes<
//         'a,
//         Self,
//         impl FnMut(Result<TraversalVal, GraphError>) -> InNodesIterator<'a, RwTxn<'a>>,
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
//                         .in_edges_db
//                         .lazily_decode_data()
//                         .prefix_iter(txn, &prefix)
//                         .unwrap();

//                     InNodesIterator {
//                         iter,
//                         storage: Arc::clone(&storage),
//                         txn,
//                         edge_label,
//                     }
//                 })
//                 .flatten();
//             InNodes { iter }
//         }
//     }
// }
