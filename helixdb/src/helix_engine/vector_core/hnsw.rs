use std::collections::HashMap;

use crate::helix_engine::storage_core::backend_any::AnyRead;
use crate::helix_engine::vector_core::vector::HVector;
use crate::{helix_engine::types::VectorError, protocol::value::Value};
use heed3::RwTxn;

pub trait HNSW {
    /// Search for the k nearest neighbors of a query vector
    ///
    /// # Arguments
    ///
    /// * `txn` - The transaction to use
    /// * `query` - The query vector
    /// * `k` - The number of nearest neighbors to search for
    ///
    /// # Returns
    ///
    /// A vector of tuples containing the id and distance of the nearest neighbors
    fn search<F>(
        &self,
        r: &AnyRead<'_>,
        query: &[f32],
        k: usize,
        filter: Option<&[F]>,
        should_trickle: bool,
    ) -> Result<Vec<HVector>, VectorError>
    where
        F: Fn(&HVector) -> bool;

    /// Insert a new vector into the index
    ///
    /// # Arguments
    ///
    /// * `txn` - The transaction to use
    /// * `id` - The id of the vector
    /// * `data` - The vector data
    ///
    /// # Returns
    ///
    /// An HVector of the data inserted
    fn insert<F>(
        &self,
        txn: &mut RwTxn,
        data: &[f32],
        nid: Option<u128>,
        fields: Option<HashMap<String, Value>>,
    ) -> Result<HVector, VectorError>
    where
        F: Fn(&HVector) -> bool;

    /// Load a full hnsw index with all vectors at once
    ///
    /// # Arguments
    ///
    /// * `txn` - The transaction to use
    /// * `id` - The id of the vector
    /// * `data` - A Vec of all the vectors to insert
    ///
    /// # Returns
    ///
    /// An emtpy tuple
    fn load<F>(&self, txn: &mut RwTxn, data: Vec<&[f32]>) -> Result<(), VectorError>
    where
        F: Fn(&HVector) -> bool;

    /// Get all vectors from the index at a specific level
    ///
    /// # Arguments
    ///
    /// * `txn` - The read-only transaction to use for retrieving vectors
    /// * `level` - A usize for which level to get all vectors from
    ///
    /// # Returns
    ///
    /// A `Result` containing a `Vec` of `HVector` if successful
    fn get_all_vectors(
        &self,
        r: &AnyRead<'_>,
        level: Option<usize>,
    ) -> Result<Vec<HVector>, VectorError>;

    /// Get specific vector based on id and level
    ///
    /// # Arguments
    ///
    /// * `txn` - The transaction to use
    /// * `id` - The id of the vector
    /// * `level` - Which level to get the vector from
    /// * `with_data` - Whether or not to fetch the vector with data
    ///
    /// # Returns
    ///
    /// A `Result` containing a `Vec` of `HVector` if successful
    fn get_vector(
        &self,
        r: &AnyRead<'_>,
        id: u128,
        level: usize,
        with_data: bool,
    ) -> Result<HVector, VectorError>;
}
