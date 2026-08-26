use std::sync::Arc;

use super::ops::tr_val::TraversalVal;
use crate::helix_engine::{
    storage_core::backend_any::{AnyRead, AnyWrite},
    storage_core::storage_core::HelixGraphStorage,
    types::GraphError,
};
use itertools::Itertools;

pub struct RoTraversalIterator<'a, I> {
    pub inner: I,
    pub storage: Arc<HelixGraphStorage>,
    /// Single read snapshot threaded across the whole traversal (US field flip).
    /// `AnyRead` dispatches to LMDB (a borrowed resize-safe `RoTxn`) or LSM (a
    /// SlateDB snapshot); both are shared by `&` so every op reads the SAME
    /// point-in-time view.
    pub txn: &'a AnyRead<'a>,
}

// implementing iterator for TraversalIterator
impl<'a, I> Iterator for RoTraversalIterator<'a, I>
where
    I: Iterator<Item = Result<TraversalVal, GraphError>>,
{
    type Item = Result<TraversalVal, GraphError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<'a, I: Iterator<Item = Result<TraversalVal, GraphError>>> RoTraversalIterator<'a, I> {
    pub fn collect_to<B: FromIterator<TraversalVal>>(self) -> B {
        self.inner.filter_map(|item| item.ok()).collect::<B>()
    }

    pub fn collect_dedup<B: FromIterator<TraversalVal>>(self) -> B {
        self.inner
            .filter_map(|item| item.ok())
            .unique()
            .collect::<B>()
    }

    pub fn collect_to_obj(self) -> Option<TraversalVal> {
        self.inner.filter_map(|item| item.ok()).take(1).next()
    }
}
pub struct RwTraversalIterator<'a, 'b, I> {
    pub inner: I,
    pub storage: Arc<HelixGraphStorage>,
    /// Single write batch threaded across the whole write traversal (field
    /// flip). `AnyWrite` dispatches to LMDB (a resize-safe `RwTxn`, owned by the
    /// caller's `WriteView`) or LSM (a SlateDB batch); every write op writes
    /// into this ONE batch, committed once by `WriteView::commit`.
    pub txn: &'b mut AnyWrite<'a>,
}

// implementing iterator for TraversalIterator
impl<'a, 'b, I> Iterator for RwTraversalIterator<'a, 'b, I>
where
    I: Iterator<Item = Result<TraversalVal, GraphError>>,
{
    type Item = Result<TraversalVal, GraphError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}
impl<'a, 'b, I: Iterator> RwTraversalIterator<'a, 'b, I> {
    pub fn new(storage: Arc<HelixGraphStorage>, txn: &'b mut AnyWrite<'a>, inner: I) -> Self {
        Self {
            inner,
            storage,
            txn,
        }
    }

    pub fn collect_to<B: FromIterator<TraversalVal>>(self) -> B
    where
        I: Iterator<Item = Result<TraversalVal, GraphError>>,
    {
        self.inner.filter_map(|item| item.ok()).collect::<B>()
    }
}
// pub trait TraversalIteratorMut<'a> {
//     type Inner: Iterator<Item = Result<TraversalVal, GraphError>>;

//     fn next<'b>(
//         &mut self,
//         storage: Arc<HelixGraphStorage>,
//         txn: &'b mut RwTxn<'a>,
//     ) -> Option<Result<TraversalVal, GraphError>>;

// }
