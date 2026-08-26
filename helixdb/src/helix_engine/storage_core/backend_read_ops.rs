//! Backend-routed read building blocks for the LMDB->SlateDB migration (US-004).
//!
//! Methods here mirror the heed `StorageMethods`/`BasicStorageMethods` reads but
//! route through `self.backend` (the pluggable `StorageBackend`). They read the
//! SAME databases with byte-identical keys, so they coexist with the heed
//! methods during the call-site flip. Filled in incrementally.

#![allow(dead_code)]

use crate::helix_engine::types::GraphError;
use crate::protocol::items::{Edge, Node, SerializedEdge, SerializedNode};
use crate::protocol::value::Value;

use super::backend::{KeyRange, Namespace, StorageBackend};
use super::backend_any::AnyRead;
use super::storage_core::HelixGraphStorage;

/// Backend-routed (`_be`) variants of the temp/secondary-index reads (US-004).
/// These read the SAME databases with byte-identical keys as their heed
/// counterparts but route through the pluggable `StorageBackend`. The heed
/// methods hand back borrowed `&[u8]` valid for the read txn; the backend is
/// visitor-scoped, so the temp variants return OWNED `Vec<u8>` instead.
impl HelixGraphStorage {
    /// Backend-routed variant of `BasicStorageMethods::get_temp_node`: the raw
    /// (still-encoded) node bytes. Heed returns a borrowed `&[u8]`; the backend
    /// closure borrow is scoped, so we copy out an owned `Vec<u8>`.
    #[inline]
    pub fn get_temp_node_be(&self, r: &AnyRead<'_>, id: &u128) -> Result<Vec<u8>, GraphError> {
        self.backend
            .get_with(r, Namespace::Nodes, &id.to_be_bytes(), |v| match v {
                Some(data) => Ok(data.to_vec()),
                None => Err(GraphError::NodeNotFound),
            })
            .map_err(|e| GraphError::New(e.to_string()))?
    }

    /// Backend-routed variant of `BasicStorageMethods::get_temp_edge`: the raw
    /// (still-encoded) edge bytes, owned for the same reason as the node variant.
    #[inline]
    pub fn get_temp_edge_be(&self, r: &AnyRead<'_>, id: &u128) -> Result<Vec<u8>, GraphError> {
        self.backend
            .get_with(r, Namespace::Edges, &id.to_be_bytes(), |v| match v {
                Some(data) => Ok(data.to_vec()),
                None => Err(GraphError::EdgeNotFound),
            })
            .map_err(|e| GraphError::New(e.to_string()))?
    }

    /// Backend-routed variant of `StorageMethods::get_node_by_secondary_index`.
    ///
    /// Byte-identical to the heed path: the index DB is keyed by
    /// `stable_index_key_for_value(value)` (bincode-encoded `Value` run through
    /// the compact-key hasher) and stores the node id as 16-byte big-endian. We
    /// resolve the index namespace, decode the stored id, then load the node via
    /// the backend-routed `get_node_be`.
    pub fn get_node_by_secondary_index_be(
        &self,
        r: &AnyRead<'_>,
        index: &str,
        value: &Value,
    ) -> Result<Node, GraphError> {
        let key = Self::stable_index_key_for_value(value)?;
        let node_id = self
            .backend
            .get_with(r, Namespace::SecondaryIndex(index), &key, |v| match v {
                Some(bytes) => Some(Self::get_u128_from_bytes(bytes)),
                None => None,
            })
            .map_err(|e| GraphError::New(e.to_string()))?
            .ok_or(GraphError::NodeNotFound)??;
        self.get_node_be(r, &node_id)
    }

    /// Backend-routed variant of `get_nodes_by_multi_index` (upsert.rs).
    ///
    /// Byte-identical to the heed path: the multi-index is `DUP_SORT`, keyed by
    /// `stable_index_key_for_value(value)`, with 16-byte big-endian node ids as
    /// the duplicate values. `Namespace::MultiIndex(name)` resolves to the same
    /// on-disk `midx_{name}` database, so this reads the identical entries.
    pub fn get_nodes_by_multi_index_be(
        &self,
        r: &AnyRead<'_>,
        index_name: &str,
        value: &Value,
    ) -> Result<Vec<u128>, GraphError> {
        if !self.multi_indices.contains_key(index_name) {
            return Err(GraphError::New(format!(
                "Multi-index '{}' not found",
                index_name
            )));
        }
        let key = Self::stable_index_key_for_value(value)?;
        let mut result = Vec::new();
        let mut decode_err: Option<GraphError> = None;
        self.backend
            .for_each_dup(r, Namespace::MultiIndex(index_name), &key, |val_bytes| {
                match <[u8; 16]>::try_from(val_bytes) {
                    Ok(arr) => {
                        result.push(u128::from_be_bytes(arr));
                        true
                    }
                    Err(_) => {
                        decode_err = Some(GraphError::SliceLengthError);
                        false
                    }
                }
            })
            .map_err(|e| GraphError::New(e.to_string()))?;
        if let Some(e) = decode_err {
            return Err(e);
        }
        Ok(result)
    }

    /// Backend-routed full scan of the node keyspace (`Namespace::Nodes`),
    /// decoding each entry into a `Node`. Mirrors the heed `n()` source scan
    /// (`nodes_db.lazily_decode_data().iter(txn)`): ascending key order,
    /// per-entry decode errors surfaced as `Err` (the scan continues), empty
    /// values surfaced as a conversion error — NOT a whole-scan failure.
    ///
    /// Eager: results are materialised into a `Vec` because the seam is
    /// visitor-based (no lending iterator). Byte-identical content/order to the
    /// heed scan on LMDB.
    pub fn scan_all_nodes_be(
        &self,
        r: &AnyRead<'_>,
    ) -> Result<Vec<Result<Node, GraphError>>, GraphError> {
        let mut out: Vec<Result<Node, GraphError>> = Vec::new();
        self.backend
            .scan(r, Namespace::Nodes, KeyRange::all(), |key, value| {
                out.push(Self::decode_scanned_node(key, value));
                true
            })
            .map_err(|e| GraphError::New(e.to_string()))?;
        Ok(out)
    }

    /// Backend-routed full scan of the edge keyspace (`Namespace::Edges`).
    /// Same semantics as [`scan_all_nodes_be`] for edges; mirrors heed `e()`.
    pub fn scan_all_edges_be(
        &self,
        r: &AnyRead<'_>,
    ) -> Result<Vec<Result<Edge, GraphError>>, GraphError> {
        let mut out: Vec<Result<Edge, GraphError>> = Vec::new();
        self.backend
            .scan(r, Namespace::Edges, KeyRange::all(), |key, value| {
                out.push(Self::decode_scanned_edge(key, value));
                true
            })
            .map_err(|e| GraphError::New(e.to_string()))?;
        Ok(out)
    }

    /// Decode one scanned node entry exactly as the heed `N` iterator does:
    /// 16-byte big-endian key -> u128 id; empty value -> conversion error.
    fn decode_scanned_node(key: &[u8], value: &[u8]) -> Result<Node, GraphError> {
        let id = Self::scanned_id(key)?;
        if value.is_empty() {
            return Err(GraphError::ConversionError(
                "Error deserializing node".to_string(),
            ));
        }
        SerializedNode::decode_node(value, id)
            .map_err(|e| GraphError::ConversionError(format!("Error deserializing node: {}", e)))
    }

    /// Decode one scanned edge entry exactly as the heed `E` iterator does.
    fn decode_scanned_edge(key: &[u8], value: &[u8]) -> Result<Edge, GraphError> {
        let id = Self::scanned_id(key)?;
        if value.is_empty() {
            return Err(GraphError::ConversionError(
                "Error deserializing edge".to_string(),
            ));
        }
        SerializedEdge::decode_edge(value, id)
            .map_err(|e| GraphError::ConversionError(format!("Error deserializing edge: {}", e)))
    }

    /// A node/edge primary key is a 16-byte big-endian `u128` (heed `U128<BE>`).
    fn scanned_id(key: &[u8]) -> Result<u128, GraphError> {
        <[u8; 16]>::try_from(key)
            .map(u128::from_be_bytes)
            .map_err(|_| GraphError::SliceLengthError)
    }
}

#[cfg(test)]
mod tests {
    use super::super::storage_methods::{BasicStorageMethods, DBMethods, StorageMethods};
    use super::*;
    use crate::helix_engine::graph_core::config::Config;

    fn test_config() -> Config {
        Config::new(8, 32, 64, 1)
    }

    /// Build a storage instance with a single registered secondary index `k`
    /// and one node whose `k` property is indexed, returning the storage and the
    /// node id that was written. `create_node` indexes by looking up the property
    /// whose name equals the index name, so the index and the property share the
    /// name `k`.
    fn storage_with_indexed_node(value: Value) -> (HelixGraphStorage, u128, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let mut storage =
            HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
        storage.create_secondary_index("k").unwrap();
        let id: u128 = 0x1234_5678;
        storage
            .with_write_txn(|wtxn| {
                storage.create_node(
                    wtxn,
                    "Label",
                    vec![("k".to_string(), value.clone())],
                    Some(&["k".to_string()]),
                    Some(id),
                )?;
                Ok(())
            })
            .unwrap();
        (storage, id, dir)
    }

    #[test]
    fn get_temp_node_be_matches_get_temp_node() {
        let (storage, id, _dir) = storage_with_indexed_node(Value::String("alice".to_string()));

        let old = storage
            .with_read_txn(|rtxn| storage.get_temp_node(rtxn, &id).map(|s| s.to_vec()))
            .unwrap();
        let r = storage.backend.begin_read().unwrap();
        let new = storage.get_temp_node_be(&r, &id).unwrap();
        assert_eq!(old, new, "temp node bytes must be byte-identical");

        // Missing id surfaces NodeNotFound on the backend path too.
        assert!(storage.get_temp_node_be(&r, &0xDEADu128).is_err());
    }

    #[test]
    fn get_temp_edge_be_matches_get_temp_edge() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
        let from: u128 = 0xA1;
        let to: u128 = 0xA2;
        let edge_id = storage
            .with_write_txn(|wtxn| {
                storage.create_node(wtxn, "Label", vec![], None, Some(from))?;
                storage.create_node(wtxn, "Label", vec![], None, Some(to))?;
                let edge = storage.create_edge(
                    wtxn,
                    "REL",
                    &from,
                    &to,
                    vec![("w".to_string(), Value::I32(7))],
                )?;
                Ok(edge.id)
            })
            .unwrap();

        let old = storage
            .with_read_txn(|rtxn| storage.get_temp_edge(rtxn, &edge_id).map(|s| s.to_vec()))
            .unwrap();
        let r = storage.backend.begin_read().unwrap();
        let new = storage.get_temp_edge_be(&r, &edge_id).unwrap();
        assert_eq!(old, new, "temp edge bytes must be byte-identical");

        assert!(storage.get_temp_edge_be(&r, &0xDEADu128).is_err());
    }

    #[test]
    fn get_node_by_secondary_index_be_matches_heed() {
        let value = Value::String("alice".to_string());
        let (storage, id, _dir) = storage_with_indexed_node(value.clone());

        let old = storage
            .with_read_txn(|rtxn| storage.get_node_by_secondary_index(rtxn, "k", &value))
            .unwrap();
        let r = storage.backend.begin_read().unwrap();
        let new = storage
            .get_node_by_secondary_index_be(&r, "k", &value)
            .unwrap();

        assert_eq!(old.id, id);
        assert_eq!(old.id, new.id);
        assert_eq!(old.label, new.label);
        assert_eq!(
            old.properties.get("k"),
            new.properties.get("k"),
            "indexed property must round-trip identically"
        );
    }

    #[test]
    fn get_node_by_secondary_index_be_missing_value_matches_heed() {
        let value = Value::String("alice".to_string());
        let (storage, _id, _dir) = storage_with_indexed_node(value);

        // A value that was never indexed must be NotFound on BOTH paths.
        let absent = Value::String("nobody".to_string());
        let old =
            storage.with_read_txn(|rtxn| storage.get_node_by_secondary_index(rtxn, "k", &absent));
        let r = storage.backend.begin_read().unwrap();
        let new = storage.get_node_by_secondary_index_be(&r, "k", &absent);

        assert!(old.is_err());
        assert!(new.is_err());
    }

    #[test]
    fn get_nodes_by_multi_index_be_matches_heed() {
        // The auto-provisioned `name` multi-index is DUP_SORT|DUP_FIXED on-disk as
        // `midx_name`. Populate it the way production does (16-byte BE node ids
        // under the stable index key) and confirm the backend twin reads exactly
        // what production `get_nodes_by_multi_index` reads — which also proves
        // `Namespace::MultiIndex("name")` now resolves to `midx_name`.
        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();

        let name = Value::String("main".to_string());
        let key = HelixGraphStorage::stable_index_key_for_value(&name).unwrap();
        let id1: u128 = 0x11;
        let id2: u128 = 0x22;
        {
            let mut w = storage.lmdb_env().unwrap().write_txn().unwrap();
            let idx_db = storage.multi_indices.get("name").unwrap().as_ref().unwrap();
            idx_db
                .put(&mut w, key.as_slice(), id1.to_be_bytes().as_slice())
                .unwrap();
            idx_db
                .put(&mut w, key.as_slice(), id2.to_be_bytes().as_slice())
                .unwrap();
            w.commit().unwrap();
        }

        let prod = storage
            .with_read_txn(|rtxn| storage.get_nodes_by_multi_index(rtxn, "name", &name))
            .unwrap();
        let r = storage.backend.begin_read().unwrap();
        let twin = storage
            .get_nodes_by_multi_index_be(&r, "name", &name)
            .unwrap();

        assert_eq!(prod.len(), 2, "production sees both ids");
        assert_eq!(
            prod, twin,
            "backend twin reads byte-identical multi-index ids"
        );

        // An unregistered index errors on both paths.
        let bad =
            storage.with_read_txn(|rtxn| storage.get_nodes_by_multi_index(rtxn, "nope", &name));
        let bad_twin = storage.get_nodes_by_multi_index_be(&r, "nope", &name);
        assert!(bad.is_err());
        assert!(bad_twin.is_err());
    }

    #[test]
    fn scan_all_nodes_be_matches_heed_full_scan() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
        // Insert ids out of order to exercise ascending-key ordering.
        let ids = [0x30u128, 0x10u128, 0x20u128];
        storage
            .with_write_txn(|wtxn| {
                for (i, id) in ids.iter().enumerate() {
                    storage.create_node(
                        wtxn,
                        "L",
                        vec![("i".to_string(), Value::I32(i as i32))],
                        None,
                        Some(*id),
                    )?;
                }
                Ok(())
            })
            .unwrap();

        // Heed ground truth: full ordered scan of nodes_db (the heed `n()` path).
        let heed_ids: Vec<u128> = storage
            .with_read_txn(|rtxn| {
                let mut v = Vec::new();
                for item in storage
                    .lmdb_nodes_db()
                    .unwrap()
                    .lazily_decode_data()
                    .iter(rtxn)?
                {
                    let (k, _) = item?;
                    v.push(k);
                }
                Ok(v)
            })
            .unwrap();
        assert_eq!(
            heed_ids,
            vec![0x10, 0x20, 0x30],
            "heed scans ascending key order"
        );

        let r = storage.backend.begin_read().unwrap();
        let scanned = storage.scan_all_nodes_be(&r).unwrap();
        let scanned_ids: Vec<u128> = scanned.iter().map(|res| res.as_ref().unwrap().id).collect();
        assert_eq!(
            scanned_ids, heed_ids,
            "scan_all_nodes_be must match heed full-scan order + ids"
        );
        assert!(scanned.iter().all(|res| res.as_ref().unwrap().label == "L"));
    }

    #[test]
    fn scan_all_edges_be_matches_heed_full_scan() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(dir.path().to_str().unwrap(), test_config()).unwrap();
        let a: u128 = 0xA1;
        let b: u128 = 0xA2;
        let c: u128 = 0xA3;
        let mut edge_ids = storage
            .with_write_txn(|wtxn| {
                storage.create_node(wtxn, "L", vec![], None, Some(a))?;
                storage.create_node(wtxn, "L", vec![], None, Some(b))?;
                storage.create_node(wtxn, "L", vec![], None, Some(c))?;
                let e1 = storage.create_edge(wtxn, "R", &a, &b, vec![])?;
                let e2 = storage.create_edge(wtxn, "R", &b, &c, vec![])?;
                Ok(vec![e1.id, e2.id])
            })
            .unwrap();
        edge_ids.sort_unstable();

        let heed_ids: Vec<u128> = storage
            .with_read_txn(|rtxn| {
                let mut v = Vec::new();
                for item in storage
                    .lmdb_edges_db()
                    .unwrap()
                    .lazily_decode_data()
                    .iter(rtxn)?
                {
                    let (k, _) = item?;
                    v.push(k);
                }
                Ok(v)
            })
            .unwrap();

        let r = storage.backend.begin_read().unwrap();
        let scanned = storage.scan_all_edges_be(&r).unwrap();
        let scanned_ids: Vec<u128> = scanned.iter().map(|res| res.as_ref().unwrap().id).collect();

        assert_eq!(
            scanned_ids, heed_ids,
            "scan_all_edges_be must match heed full-scan order + ids"
        );
        assert_eq!(
            scanned_ids, edge_ids,
            "all created edges scanned in ascending id order"
        );
        assert!(scanned.iter().all(|res| res.as_ref().unwrap().label == "R"));
    }
}
