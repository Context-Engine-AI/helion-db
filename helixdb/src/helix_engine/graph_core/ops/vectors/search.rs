use super::super::tr_val::TraversalVal;
use crate::helix_engine::{
    graph_core::traversal_iter::RoTraversalIterator,
    types::{GraphError, VectorError},
    vector_core::{hnsw::HNSW, vector::HVector},
};
use std::iter::once;

pub struct SearchV<I: Iterator<Item = Result<TraversalVal, GraphError>>> {
    iter: I,
}

// implementing iterator for OutIterator
impl<I: Iterator<Item = Result<TraversalVal, GraphError>>> Iterator for SearchV<I> {
    type Item = Result<TraversalVal, GraphError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

pub trait SearchVAdapter<'a>: Iterator<Item = Result<TraversalVal, GraphError>> + Sized {
    fn search_v<F>(
        self,
        query: &Vec<f32>,
        k: usize,
        filter: Option<&[F]>,
    ) -> RoTraversalIterator<'a, impl Iterator<Item = Result<TraversalVal, GraphError>>>
    where
        F: Fn(&HVector) -> bool;
}

impl<'a, I: Iterator<Item = Result<TraversalVal, GraphError>> + 'a> SearchVAdapter<'a>
    for RoTraversalIterator<'a, I>
{
    fn search_v<F>(
        self,
        query: &Vec<f32>,
        k: usize,
        filter: Option<&[F]>,
    ) -> RoTraversalIterator<'a, impl Iterator<Item = Result<TraversalVal, GraphError>>>
    where
        F: Fn(&HVector) -> bool,
    {
        // `VectorCore::search` reads vector + index data through its own backend
        // handle. The traversal already holds the shared read handle (`self.txn`
        // is an `&AnyRead`), so hand it straight through: on LMDB the search reads
        // through that snapshot's `RoTxn` (byte-identical); on LSM it reads the
        // same SlateDB snapshot — a single snapshot, no placeholder txn, so no
        // second RoTxn is opened on the per-thread graph env.
        let vectors = self
            .storage
            .vectors
            .search(self.txn, &query, k, filter, false);

        let iter = match vectors {
            Ok(vectors) => vectors
                .into_iter()
                .map(|vector| Ok::<TraversalVal, GraphError>(TraversalVal::Vector(vector)))
                .collect::<Vec<_>>()
                .into_iter(),
            //Err(VectorError::VectorNotFound()) =>
            //Err(VectorError::InvalidVectorData) =>
            //Err(VectorError::InvalidVectorId) =>
            //Err(VectorError::InvalidVectorLevel) =>
            //Err(VectorError::InvalidEntryPoint) =>
            //Err(VectorError::EntryPointNotFound) =>
            //Err(VectorError::InvalidVectorCoreConfig) =>
            //Err(VectorError::ConversionError()) =>
            //Err(VectorError::VectorCoreError()) =>
            Err(VectorError::InvalidVectorLength) => {
                let error = GraphError::VectorError("invalid vector dimensions!".to_string());
                once(Err(error)).collect::<Vec<_>>().into_iter()
            }
            Err(e) => once(Err(GraphError::VectorError(format!(
                "vector search: {e:?}"
            ))))
            .collect::<Vec<_>>()
            .into_iter(),
        };

        let iter = SearchV { iter };
        // Wrap it with the RoTraversalIterator adapter
        RoTraversalIterator {
            inner: iter,
            storage: self.storage,
            txn: self.txn,
        }
    }
}
