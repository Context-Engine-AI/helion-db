use super::super::tr_val::TraversalVal;
use crate::helix_engine::{
    graph_core::traversal_iter::{RoTraversalIterator, RwTraversalIterator},
    storage_core::backend::BackendKind,
    types::GraphError,
};

/// Full edge-table scan (read path), routed through the backend twin
/// `scan_all_edges_be`. Eagerly materialised (the storage seam is visitor-based,
/// no lending iterator); on LMDB the decoded edges and their ascending-key
/// ordering are byte-identical to the prior heed `edges_db.iter(txn)` scan.
pub struct E {
    iter: std::vec::IntoIter<Result<TraversalVal, GraphError>>,
}

impl Iterator for E {
    type Item = Result<TraversalVal, GraphError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

pub trait EAdapter<'a>: Iterator<Item = Result<TraversalVal, GraphError>> + Sized {
    fn e(self) -> RoTraversalIterator<'a, impl Iterator<Item = Result<TraversalVal, GraphError>>>;
}

impl<'a, I: Iterator<Item = Result<TraversalVal, GraphError>> + 'a> EAdapter<'a>
    for RoTraversalIterator<'a, I>
{
    fn e(self) -> RoTraversalIterator<'a, impl Iterator<Item = Result<TraversalVal, GraphError>>> {
        // Route the full edge scan through the backend twin over this traversal's
        // existing read snapshot. Byte-identical to the heed edges_db scan on
        // LMDB; a scan-setup error surfaces as a single Err item (the prior code
        // panicked via `.iter(txn).unwrap()`).
        let items: Vec<Result<TraversalVal, GraphError>> =
            match self.storage.scan_all_edges_be(self.txn) {
                Ok(edges) => edges
                    .into_iter()
                    .map(|res| res.map(TraversalVal::Edge))
                    .collect(),
                Err(e) => vec![Err(e)],
            };
        RoTraversalIterator {
            inner: E {
                iter: items.into_iter(),
            },
            storage: self.storage,
            txn: self.txn,
        }
    }
}

pub trait RwEAdapter<'a, 'b>: Iterator<Item = Result<TraversalVal, GraphError>> + Sized {
    fn e(
        self,
    ) -> RwTraversalIterator<'a, 'b, impl Iterator<Item = Result<TraversalVal, GraphError>>>;
}

impl<'a, 'b, I: Iterator<Item = Result<TraversalVal, GraphError>> + 'a> RwEAdapter<'a, 'b>
    for RwTraversalIterator<'a, 'b, I>
{
    fn e(
        self,
    ) -> RwTraversalIterator<'a, 'b, impl Iterator<Item = Result<TraversalVal, GraphError>>> {
        // Read-your-writes edge scan, routed through the same backend seam as the
        // read path (`scan_all_edges_be`). The read view observes the live write
        // batch's buffered writes: on LMDB it borrows the batch `RwTxn`; on LSM it
        // overlays a clone of the batch's `pending` map on a fresh committed
        // snapshot (mirrors `WriteView::read_view`). Eagerly collected so the read
        // view drops before `txn` is threaded into the returned iterator.
        let storage = self.storage;
        let txn = self.txn;

        // Build a read-your-writes view of the live write batch, then run the same
        // backend edge scan the read path uses. The read view (which borrows
        // `storage`/`txn`) is confined to this inner block so it is fully dropped
        // before `storage`/`txn` are threaded into the returned iterator; errors
        // fold into the item stream rather than early-returning.
        let items: Vec<Result<TraversalVal, GraphError>> = {
            let read_result = match storage.backend.kind() {
                BackendKind::Lmdb => match txn.lmdb_ro() {
                    Some(rw) => Ok(storage.backend.read_borrowed(rw)),
                    None => Err(GraphError::New(
                        "LMDB write batch must hold an RwTxn for read-your-writes".to_string(),
                    )),
                },
                BackendKind::Lsm => {
                    let pending = txn.lsm_pending().cloned().unwrap_or_default();
                    storage
                        .backend
                        .lsm_read_with_pending(pending)
                        .map_err(|e| GraphError::New(format!("LSM read-your-writes view: {e}")))
                }
            };
            match read_result {
                Ok(read) => match storage.scan_all_edges_be(&read) {
                    Ok(edges) => edges
                        .into_iter()
                        .map(|res| res.map(TraversalVal::Edge))
                        .collect(),
                    Err(e) => vec![Err(e)],
                },
                Err(e) => vec![Err(e)],
            }
        };

        RwTraversalIterator {
            inner: items.into_iter(),
            storage,
            txn,
        }
    }
}
