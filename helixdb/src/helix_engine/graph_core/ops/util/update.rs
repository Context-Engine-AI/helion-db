use crate::{
    helix_engine::{
        graph_core::{
            ops::util::secondary_index::{
                restore_secondary_snapshots, secondary_key_points_to_node, snapshot_secondary_key,
            },
            traversal_iter::RwTraversalIterator,
        },
        storage_core::{storage_core::HelixGraphStorage, storage_methods::StorageMethods},
        types::GraphError,
    },
    protocol::{
        items::{SerializedEdge, SerializedNode},
        value::Value,
    },
};

use super::super::tr_val::TraversalVal;

pub struct Update<I> {
    iter: I,
}

impl<I> Iterator for Update<I>
where
    I: Iterator<Item = Result<TraversalVal, GraphError>>,
{
    type Item = Result<TraversalVal, GraphError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

pub trait UpdateAdapter<'a, 'b>: Iterator + Sized {
    fn update(
        self,
        props: Vec<(String, Value)>,
    ) -> RwTraversalIterator<'a, 'b, impl Iterator<Item = Result<TraversalVal, GraphError>>>;
}

impl<'a, 'b, I: Iterator<Item = Result<TraversalVal, GraphError>>> UpdateAdapter<'a, 'b>
    for RwTraversalIterator<'a, 'b, I>
{
    fn update(
        self,
        props: Vec<(String, Value)>,
    ) -> RwTraversalIterator<'a, 'b, impl Iterator<Item = Result<TraversalVal, GraphError>>> {
        let storage = self.storage.clone();

        let capacity = match self.inner.size_hint() {
            (_, Some(upper)) => upper,
            (lower, None) => lower,
        };
        let mut vec = Vec::with_capacity(capacity);

        if let Some(txn) = self.txn.lmdb_rw_mut() {
            for item in self.inner {
                match item {
                    Ok(TraversalVal::Node(node)) => {
                        let result = (|| -> Result<TraversalVal, GraphError> {
                            let mut updated_node = storage.get_node(txn, &node.id)?;
                            let mut snapshots = Vec::new();

                            for (k, v) in props.iter() {
                                let old_value = updated_node.properties.get(k);
                                if old_value == Some(v) {
                                    continue;
                                }

                                if let Some(Some(db)) = storage.secondary_indices.get(k) {
                                    let new_key = HelixGraphStorage::stable_index_key_for_value(v)?;
                                    if let Some(old_value) = old_value {
                                        let old_key =
                                            HelixGraphStorage::stable_index_key_for_value(
                                                old_value,
                                            )?;
                                        if old_key != new_key {
                                            snapshots.push(snapshot_secondary_key(
                                                txn,
                                                *db,
                                                old_key.clone(),
                                            )?);
                                            if secondary_key_points_to_node(
                                                txn,
                                                *db,
                                                old_key.as_slice(),
                                                node.id,
                                            )? {
                                                if let Err(e) = db.delete(txn, old_key.as_slice()) {
                                                    restore_secondary_snapshots(txn, &snapshots)?;
                                                    return Err(GraphError::from(e));
                                                }
                                            }
                                        }
                                    }

                                    snapshots.push(snapshot_secondary_key(
                                        txn,
                                        *db,
                                        new_key.clone(),
                                    )?);
                                    let id_bytes = node.id.to_be_bytes();
                                    if let Err(e) =
                                        db.put(txn, new_key.as_slice(), id_bytes.as_slice())
                                    {
                                        restore_secondary_snapshots(txn, &snapshots)?;
                                        return Err(GraphError::from(e));
                                    }
                                }

                                updated_node.properties.insert(k.clone(), v.clone());
                            }

                            let encoded_node = SerializedNode::encode_node(&updated_node)?;
                            let nodes_db = storage.nodes_db.ok_or_else(|| {
                                GraphError::StorageError(
                                    "LMDB node DB handle is unavailable on this backend"
                                        .to_string(),
                                )
                            })?;
                            if let Err(e) = nodes_db.put(
                                txn,
                                &HelixGraphStorage::node_key(&updated_node.id),
                                &encoded_node,
                            ) {
                                restore_secondary_snapshots(txn, &snapshots)?;
                                return Err(GraphError::from(e));
                            }

                            Ok(TraversalVal::Node(updated_node))
                        })();
                        vec.push(result);
                    }
                    Ok(TraversalVal::Edge(edge)) => match storage.get_edge(txn, &edge.id) {
                        Ok(mut updated_edge) => {
                            for (k, v) in props.iter() {
                                updated_edge.properties.insert(k.clone(), v.clone());
                            }
                            match SerializedEdge::encode_edge(&updated_edge) {
                                Ok(serialized) => {
                                    let edges_db = match storage.edges_db {
                                        Some(db) => db,
                                        None => {
                                            vec.push(Err(GraphError::StorageError(
                                                "LMDB edge DB handle is unavailable on this backend"
                                                    .to_string(),
                                            )));
                                            continue;
                                        }
                                    };
                                    match edges_db.put(
                                        txn,
                                        &HelixGraphStorage::edge_key(&updated_edge.id),
                                        &serialized,
                                    ) {
                                        Ok(_) => vec.push(Ok(TraversalVal::Edge(updated_edge))),
                                        Err(e) => vec.push(Err(GraphError::from(e))),
                                    }
                                }
                                Err(e) => vec.push(Err(GraphError::from(e))),
                            }
                        }
                        Err(e) => vec.push(Err(e)),
                    },
                    _ => vec.push(Err(GraphError::New("Unsupported value type".to_string()))),
                }
            }
        } else {
            for item in self.inner {
                match item {
                    Ok(TraversalVal::Node(node)) => {
                        let result = storage
                            .update_node_be(self.txn, &node.id, &props)
                            .map(TraversalVal::Node);
                        vec.push(result);
                    }
                    Ok(TraversalVal::Edge(edge)) => {
                        let result = storage
                            .update_edge_be(self.txn, &edge.id, &props)
                            .map(TraversalVal::Edge);
                        vec.push(result);
                    }
                    _ => vec.push(Err(GraphError::New("Unsupported value type".to_string()))),
                }
            }
        }

        RwTraversalIterator {
            inner: Update {
                iter: vec.into_iter(),
            },
            storage: self.storage,
            txn: self.txn,
        }
    }
}
