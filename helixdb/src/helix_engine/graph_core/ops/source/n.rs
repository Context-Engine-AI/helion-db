use crate::helix_engine::{graph_core::traversal_iter::RoTraversalIterator, types::GraphError};

use super::super::tr_val::TraversalVal;

/// Full node-table scan, routed through the backend twin `scan_all_nodes_be`.
/// Eagerly materialised (the storage seam is visitor-based, no lending
/// iterator); on LMDB the decoded nodes and their ascending-key ordering are
/// byte-identical to the prior heed `nodes_db.iter(txn)` scan.
pub struct N {
    iter: std::vec::IntoIter<Result<TraversalVal, GraphError>>,
}

impl Iterator for N {
    type Item = Result<TraversalVal, GraphError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

pub trait NAdapter<'a>: Iterator<Item = Result<TraversalVal, GraphError>> + Sized {
    fn n(self) -> RoTraversalIterator<'a, impl Iterator<Item = Result<TraversalVal, GraphError>>>;
}

impl<'a, I: Iterator<Item = Result<TraversalVal, GraphError>> + 'a> NAdapter<'a>
    for RoTraversalIterator<'a, I>
{
    fn n(self) -> RoTraversalIterator<'a, impl Iterator<Item = Result<TraversalVal, GraphError>>> {
        // Route the full node scan through the backend twin over this traversal's
        // single shared read snapshot. Byte-identical to the heed nodes_db scan on
        // LMDB; a scan-setup error surfaces as a single Err item (the prior code
        // panicked via `.iter(txn).unwrap()`).
        let items: Vec<Result<TraversalVal, GraphError>> =
            match self.storage.scan_all_nodes_be(self.txn) {
                Ok(nodes) => nodes
                    .into_iter()
                    .map(|res| res.map(TraversalVal::Node))
                    .collect(),
                Err(e) => vec![Err(e)],
            };

        RoTraversalIterator {
            inner: N {
                iter: items.into_iter(),
            },
            storage: self.storage,
            txn: self.txn,
        }
    }
}
