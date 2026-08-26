use std::collections::HashMap;
use tracing::info;

use crate::helix_engine::storage_core::backend::{Namespace, StorageBackend};
use crate::{
    helix_engine::{graph_core::traversal_iter::RwTraversalIterator, types::GraphError},
    protocol::items::{Node, SerializedNode},
};

use super::super::tr_val::TraversalVal;

pub struct BulkAddN {
    inner: std::iter::Once<Result<TraversalVal, GraphError>>,
}

impl Iterator for BulkAddN {
    type Item = Result<TraversalVal, GraphError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

pub trait BulkAddNAdapter<'a, 'b>:
    Iterator<Item = Result<TraversalVal, GraphError>> + Sized
{
    fn bulk_add_n(
        self,
        nodes: &mut [u128],
        secondary_indices: Option<&[String]>,
        chunk_size: usize,
    ) -> RwTraversalIterator<'a, 'b, impl Iterator<Item = Result<TraversalVal, GraphError>>>;
}

impl<'a, 'b, I: Iterator<Item = Result<TraversalVal, GraphError>>> BulkAddNAdapter<'a, 'b>
    for RwTraversalIterator<'a, 'b, I>
{
    fn bulk_add_n(
        self,
        nodes: &mut [u128],
        secondary_indices: Option<&[String]>,
        chunk_size: usize,
    ) -> RwTraversalIterator<'a, 'b, impl Iterator<Item = Result<TraversalVal, GraphError>>> {
        let mut result: Result<TraversalVal, GraphError> = Ok(TraversalVal::Empty);
        if secondary_indices
            .map(|indices| !indices.is_empty())
            .unwrap_or(false)
        {
            result = Err(GraphError::New(
                "bulk_add_n cannot update secondary indices without node properties".to_string(),
            ));
        }
        nodes.sort_unstable_by_key(|node| *node);

        // Field flip: write node bytes into the traversal's single shared
        // `AnyWrite` batch via the seam (engine-neutral). Byte-identical node
        // value on LMDB; bulk_add_n deliberately writes NO secondary index and
        // does NOT bump the node counter (matches the prior behavior). Drops the
        // LMDB APPEND fast path (re-add via a hinted seam put if bench perf needs
        // it).
        if result.is_ok() {
            let chunk_size = chunk_size.max(1);
            let mut count = 0;
            'outer: for chunk in nodes.chunks_mut(chunk_size) {
                for node_id in chunk {
                    // NOTE: label is hardcoded to "user" — this function is used
                    // only in benchmarks/bulk-load paths where a label is not
                    // supplied by the caller.
                    let node = Node {
                        id: *node_id,
                        label: "user".to_string(),
                        properties: HashMap::new(),
                    };
                    match SerializedNode::encode_node(&node) {
                        Ok(bytes) => {
                            if let Err(e) = self
                                .storage
                                .backend
                                .put(self.txn, Namespace::Nodes, &node.id.to_be_bytes(), &bytes)
                                .map_err(|e| GraphError::New(e.to_string()))
                            {
                                result = Err(e);
                                break 'outer;
                            }
                        }
                        Err(e) => {
                            result = Err(GraphError::from(e));
                            break 'outer;
                        }
                    }
                    count += 1;
                }
                if count % 1000000 == 0 {
                    info!("bulk_add_n processed: {}", count);
                }
            }
        }
        RwTraversalIterator {
            inner: std::iter::once(result), // TODO: change to support adding multiple edges
            storage: self.storage,
            txn: self.txn,
        }
    }
}
