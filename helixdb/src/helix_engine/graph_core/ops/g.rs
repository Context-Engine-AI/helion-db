use super::tr_val::TraversalVal;
use crate::{
    helix_engine::{
        graph_core::traversal_iter::{RoTraversalIterator, RwTraversalIterator},
        storage_core::backend::{BackendKind, Namespace, StorageBackend},
        storage_core::backend_any::{AnyRead, AnyWrite},
        storage_core::storage_core::HelixGraphStorage,
        types::GraphError,
    },
    protocol::{
        items::{Edge, SerializedEdge},
        label_hash::hash_label,
    },
};
use heed3::PutFlags;
use std::{collections::HashMap, sync::Arc};
use tracing::{info, warn};

pub struct G {
    iter: std::iter::Once<Result<TraversalVal, GraphError>>,
}

// implementing iterator for OutIterator
impl Iterator for G {
    type Item = Result<TraversalVal, GraphError>;

    /// Returns the next outgoing node by decoding the edge id and then getting the edge and node
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl G {
    pub fn new<'a>(
        storage: Arc<HelixGraphStorage>,
        txn: &'a AnyRead<'a>,
    ) -> RoTraversalIterator<'a, impl Iterator<Item = Result<TraversalVal, GraphError>>>
    where
        Self: Sized,
    {
        let iter = std::iter::once(Ok(TraversalVal::Empty));
        RoTraversalIterator {
            inner: iter,
            storage,
            txn,
        }
    }

    pub fn new_mut<'a, 'b>(
        storage: Arc<HelixGraphStorage>,
        txn: &'b mut AnyWrite<'a>,
    ) -> RwTraversalIterator<'a, 'b, impl Iterator<Item = Result<TraversalVal, GraphError>>>
    where
        Self: Sized,
    {
        let iter = std::iter::once(Ok(TraversalVal::Empty));
        RwTraversalIterator {
            inner: iter,
            storage,
            txn,
        }
    }

    pub fn new_from<'a>(
        storage: Arc<HelixGraphStorage>,
        txn: &'a AnyRead<'a>,
        vals: Vec<TraversalVal>,
    ) -> RoTraversalIterator<'a, impl Iterator<Item = Result<TraversalVal, GraphError>>> {
        RoTraversalIterator {
            inner: vals.into_iter().map(|val| Ok(val)),
            storage,
            txn,
        }
    }

    pub fn bulk_add_e(
        storage: Arc<HelixGraphStorage>,
        mut edges: Vec<(u128, u128, u128)>,
        should_check_nodes: bool,
        chunk_size: usize,
    ) -> Result<(), GraphError> {
        let chunk_size = chunk_size.max(1);
        // sort by id
        edges.sort_unstable_by(|(_, _, id), (_, _, id_)| id.cmp(id_));

        if storage.backend.kind() == BackendKind::Lsm {
            return storage.with_write_backend(|w| {
                for (e_from, e_to, e_id) in &edges {
                    if should_check_nodes
                        && (storage
                            .backend
                            .get_for_update(w, Namespace::Nodes, &e_from.to_be_bytes(), |node| {
                                node.is_none()
                            })
                            .map_err(|e| GraphError::New(e.to_string()))?
                            || storage
                                .backend
                                .get_for_update(w, Namespace::Nodes, &e_to.to_be_bytes(), |node| {
                                    node.is_none()
                                })
                                .map_err(|e| GraphError::New(e.to_string()))?)
                    {
                        return Err(GraphError::NodeNotFound);
                    }
                    let edge = Edge {
                        id: *e_id,
                        label: "knows".to_string(),
                        properties: HashMap::new(),
                        from_node: *e_from,
                        to_node: *e_to,
                    };
                    let bytes = SerializedEdge::encode_edge(&edge)?;
                    storage
                        .backend
                        .put(w, Namespace::Edges, &e_id.to_be_bytes(), &bytes)
                        .map_err(|e| GraphError::New(e.to_string()))?;
                }

                for (from_node, to_node, id) in &edges {
                    storage
                        .backend
                        .put_dup(
                            w,
                            Namespace::OutEdges,
                            &HelixGraphStorage::out_edge_key(from_node, &hash_label("knows", None)),
                            &HelixGraphStorage::pack_edge_data(to_node, id),
                        )
                        .map_err(|e| GraphError::New(e.to_string()))?;
                }

                for (from_node, to_node, id) in &edges {
                    storage
                        .backend
                        .put_dup(
                            w,
                            Namespace::InEdges,
                            &HelixGraphStorage::in_edge_key(to_node, &hash_label("knows", None)),
                            &HelixGraphStorage::pack_edge_data(from_node, id),
                        )
                        .map_err(|e| GraphError::New(e.to_string()))?;
                }

                Ok(())
            });
        }

        let mut count = 0;
        info!("Adding edges");
        // EDGES
        let chunks = edges.chunks_mut(chunk_size);
        for chunk in chunks {
            let chunk_len = chunk.len();
            storage.with_write_txn(|txn| {
                let nodes_db = storage.nodes_db.ok_or_else(|| {
                    GraphError::StorageError(
                        "LMDB node DB handle is unavailable on this backend".to_string(),
                    )
                })?;
                let edges_db = storage.edges_db.ok_or_else(|| {
                    GraphError::StorageError(
                        "LMDB edge DB handle is unavailable on this backend".to_string(),
                    )
                })?;
                for (e_from, e_to, e_id) in chunk.iter() {
                    if should_check_nodes
                        && (nodes_db
                            .get(txn, &HelixGraphStorage::node_key(e_from))?
                            .is_none()
                            || nodes_db
                                .get(txn, &HelixGraphStorage::node_key(e_to))?
                                .is_none())
                    {
                        return Err(GraphError::NodeNotFound);
                    }

                    match SerializedEdge::encode_edge(&Edge {
                        id: *e_id,
                        label: "knows".to_string(),
                        properties: HashMap::new(),
                        from_node: *e_from,
                        to_node: *e_to,
                    }) {
                        Ok(bytes) => {
                            if let Err(e) = edges_db.put_with_flags(
                                txn,
                                PutFlags::APPEND,
                                &HelixGraphStorage::edge_key(e_id),
                                &bytes,
                            ) {
                                warn!("G::bulk_add_e error adding edge: {:?}", e);
                                return Err(GraphError::from(e));
                            }
                        }
                        Err(e) => {
                            warn!("G::bulk_add_e error serializing edge: {:?}", e);
                            return Err(GraphError::from(e));
                        }
                    }
                }
                Ok(())
            })?;
            count += chunk_len;
            if count % 1000000 == 0 {
                info!("G::bulk_add_e processed {} chunks", count);
            }
        }

        count = 0;
        info!("Adding out edges");
        // OUT EDGES
        let mut prev_out = None;

        edges.sort_unstable_by(|(from, _, id), (from_, _, id_)| {
            if from == from_ {
                id.cmp(id_)
            } else {
                from.cmp(from_)
            }
        });

        let chunks = edges.chunks_mut(chunk_size);
        for chunk in chunks {
            let chunk_len = chunk.len();
            let next_prev_out = chunk.last().map(|(from_node, _, _)| from_node);
            storage.with_write_txn(|txn| {
                let out_edges_db = storage.out_edges_db.ok_or_else(|| {
                    GraphError::StorageError(
                        "LMDB out-edge DB handle is unavailable on this backend".to_string(),
                    )
                })?;
                let mut local_prev_out = prev_out;
                for (from_node, to_node, id) in chunk.iter() {
                    // OUT EDGES
                    let out_flag = if Some(from_node) == local_prev_out {
                        PutFlags::APPEND_DUP
                    } else {
                        local_prev_out = Some(from_node);
                        PutFlags::APPEND
                    };

                    match out_edges_db.put_with_flags(
                        txn,
                        out_flag,
                        &HelixGraphStorage::out_edge_key(from_node, &hash_label("knows", None)),
                        &HelixGraphStorage::pack_edge_data(to_node, id),
                    ) {
                        Ok(_) => {}
                        Err(e) => {
                            warn!("G::bulk_add_e error adding out edge: {:?}", e);
                            return Err(GraphError::from(e));
                        }
                    }
                }
                Ok(())
            })?;
            prev_out = next_prev_out;
            count += chunk_len;
            if count % 1000000 == 0 {
                info!("G::bulk_add_e processed {} chunks", count);
            }
        }

        count = 0;
        info!("Adding in edges");
        // IN EDGES
        edges.sort_unstable_by(
            |(_, to, id), (_, to_, id_)| {
                if to == to_ {
                    id.cmp(id_)
                } else {
                    to.cmp(to_)
                }
            },
        );
        let mut prev_in = None;
        let chunks = edges.chunks_mut(chunk_size);
        for chunk in chunks {
            let chunk_len = chunk.len();
            let next_prev_in = chunk.last().map(|(_, to_node, _)| to_node);
            storage.with_write_txn(|txn| {
                let in_edges_db = storage.in_edges_db.ok_or_else(|| {
                    GraphError::StorageError(
                        "LMDB in-edge DB handle is unavailable on this backend".to_string(),
                    )
                })?;
                let mut local_prev_in = prev_in;
                for (from_node, to_node, id) in chunk.iter() {
                    // IN EDGES
                    let in_flag = if Some(to_node) == local_prev_in {
                        PutFlags::APPEND_DUP
                    } else {
                        local_prev_in = Some(to_node);
                        PutFlags::APPEND
                    };

                    match in_edges_db.put_with_flags(
                        txn,
                        in_flag,
                        &HelixGraphStorage::in_edge_key(to_node, &hash_label("knows", None)),
                        &HelixGraphStorage::pack_edge_data(from_node, id),
                    ) {
                        Ok(_) => {}
                        Err(e) => {
                            warn!("G::bulk_add_e error adding in edge: {:?}", e);
                            return Err(GraphError::from(e));
                        }
                    }
                }
                Ok(())
            })?;
            prev_in = next_prev_in;
            count += chunk_len;
            if count % 1000000 == 0 {
                info!("G::bulk_add_e processed {} chunks", count);
            }
        }
        Ok(())
    }
}
