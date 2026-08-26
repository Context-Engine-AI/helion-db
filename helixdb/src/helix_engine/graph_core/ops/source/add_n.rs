use super::super::tr_val::TraversalVal;
use crate::{
    helix_engine::{graph_core::traversal_iter::RwTraversalIterator, types::GraphError},
    protocol::value::Value,
};

pub struct AddNIterator {
    inner: std::iter::Once<Result<TraversalVal, GraphError>>,
}

impl Iterator for AddNIterator {
    type Item = Result<TraversalVal, GraphError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

pub trait AddNAdapter<'a, 'b>: Iterator<Item = Result<TraversalVal, GraphError>> + Sized {
    fn add_n(
        self,
        label: &'a str,
        properties: Vec<(String, Value)>,
        secondary_indices: Option<&'a [String]>,
        id: Option<u128>,
    ) -> RwTraversalIterator<'a, 'b, std::iter::Once<Result<TraversalVal, GraphError>>>;
}

impl<'a, 'b, I: Iterator<Item = Result<TraversalVal, GraphError>>> AddNAdapter<'a, 'b>
    for RwTraversalIterator<'a, 'b, I>
{
    fn add_n(
        self,
        label: &'a str,
        properties: Vec<(String, Value)>,
        secondary_indices: Option<&'a [String]>,
        id: Option<u128>, // TODO: can't be an option has to generated because always needs to be in order
    ) -> RwTraversalIterator<'a, 'b, std::iter::Once<Result<TraversalVal, GraphError>>> {
        // Field flip: write node bytes + secondary indices + node counter into
        // the traversal's single shared `AnyWrite` batch via the backend twin.
        // One engine-neutral path (LMDB or LSM); byte-identical on LMDB
        // (proven by `create_node_be_byte_identical_to_heed`).
        let result = self
            .storage
            .create_node_be(self.txn, label, properties, secondary_indices, id)
            .map(TraversalVal::Node);

        RwTraversalIterator {
            inner: std::iter::once(result),
            storage: self.storage,
            txn: self.txn,
        }
    }
}
