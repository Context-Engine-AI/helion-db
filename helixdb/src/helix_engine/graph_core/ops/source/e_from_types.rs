use std::sync::Arc;

use heed3::RoTxn;

use crate::{
    helix_engine::{
        storage_core::{
            backend::{KeyRange, Namespace},
            storage_core::HelixGraphStorage,
        },
        types::GraphError,
    },
    protocol::items::SerializedEdge,
};

use super::super::tr_val::TraversalVal;

/// Full edge-table scan filtered by label. Eagerly materialised (the storage
/// seam is visitor-based, no lending iterator); on LMDB the matched edges and
/// their ascending-key ordering are byte-identical to the prior heed `edges_db`
/// scan. Error semantics preserved from the original iterator: a decode failure
/// yields `ConversionError(e.to_string())`, label mismatches are skipped.
pub struct EFromTypes {
    iter: std::vec::IntoIter<Result<TraversalVal, GraphError>>,
}

impl Iterator for EFromTypes {
    type Item = Result<TraversalVal, GraphError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl EFromTypes {
    pub fn new(storage: &Arc<HelixGraphStorage>, txn: &RoTxn, label: &str) -> Self {
        // Routed through the backend `scan_raw` (the heed-txn pass-through used
        // by call sites that still thread a `&RoTxn`) instead of a direct
        // `edges_db.iter`. Matched edges are buffered; byte-identical to the heed
        // scan on LMDB.
        let mut items: Vec<Result<TraversalVal, GraphError>> = Vec::new();
        let scan =
            storage
                .backend
                .scan_heed(txn, Namespace::Edges, KeyRange::all(), |key, value| {
                    let id = match <[u8; 16]>::try_from(key) {
                        Ok(b) => u128::from_be_bytes(b),
                        Err(_) => {
                            items.push(Err(GraphError::SliceLengthError));
                            return true;
                        }
                    };
                    match SerializedEdge::decode_edge(value, id) {
                        Ok(edge) => {
                            if edge.label.as_str() == label {
                                items.push(Ok(TraversalVal::Edge(edge)));
                            }
                        }
                        Err(e) => items.push(Err(GraphError::ConversionError(e.to_string()))),
                    }
                    true
                });
        if let Err(e) = scan {
            items.push(Err(GraphError::New(e.to_string())));
        }
        EFromTypes {
            iter: items.into_iter(),
        }
    }
}
