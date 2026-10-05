use crate::helix_engine::types::GraphError;
use crate::helix_engine::vector_core::named_vectors::{
    DenseVectorSpaceMetadata, NamedVectorConfig,
};
use crate::helix_engine::vector_core::sparse::SparseVectorConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const CURRENT_STORAGE_SCHEMA_VERSION: u32 = 2;
pub const STORAGE_METADATA_SIDECAR_FILE: &str = "metadata.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PayloadIndexSchema {
    Keyword,
    Integer,
    Float,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataCounter {
    Nodes,
    Edges,
    Vectors,
}

const LSM_COUNTER_NODES_KEY: &[u8] = b"__helix_lsm_counter_nodes_v1";
const LSM_COUNTER_EDGES_KEY: &[u8] = b"__helix_lsm_counter_edges_v1";
const LSM_COUNTER_VECTORS_KEY: &[u8] = b"__helix_lsm_counter_vectors_v1";

pub fn lsm_counter_key(counter: MetadataCounter) -> &'static [u8] {
    match counter {
        MetadataCounter::Nodes => LSM_COUNTER_NODES_KEY,
        MetadataCounter::Edges => LSM_COUNTER_EDGES_KEY,
        MetadataCounter::Vectors => LSM_COUNTER_VECTORS_KEY,
    }
}

/// Stable metric/log label for a counter (`helix_lsm_counter_seed_total{counter=...}`).
pub fn counter_label(counter: MetadataCounter) -> &'static str {
    match counter {
        MetadataCounter::Nodes => "nodes",
        MetadataCounter::Edges => "edges",
        MetadataCounter::Vectors => "vectors",
    }
}

pub fn encode_lsm_counter_value(value: u64) -> [u8; 8] {
    value.to_le_bytes()
}

pub fn decode_lsm_counter_value(bytes: &[u8]) -> Result<u64, GraphError> {
    let bytes: [u8; 8] = bytes.try_into().map_err(|_| {
        GraphError::StorageError(format!("invalid LSM counter value length {}", bytes.len()))
    })?;
    Ok(u64::from_le_bytes(bytes))
}

pub fn encode_lsm_counter_delta(delta: i64) -> [u8; 8] {
    delta.to_le_bytes()
}

pub fn decode_lsm_counter_delta(bytes: &[u8]) -> Result<i64, GraphError> {
    let bytes: [u8; 8] = bytes.try_into().map_err(|_| {
        GraphError::StorageError(format!("invalid LSM counter delta length {}", bytes.len()))
    })?;
    Ok(i64::from_le_bytes(bytes))
}

pub fn apply_lsm_counter_delta(value: u64, delta: i64) -> Result<u64, GraphError> {
    if delta >= 0 {
        value
            .checked_add(delta as u64)
            .ok_or_else(|| GraphError::StorageError("metadata counter overflow".to_string()))
    } else {
        Ok(value.saturating_sub(delta.unsigned_abs()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct StorageStats {
    pub node_count: u64,
    pub edge_count: u64,
    pub vector_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StorageMetadata {
    pub schema_version: u32,
    pub created_at_millis: i64,
    pub updated_at_millis: i64,
    pub secondary_indices: Vec<String>,
    #[serde(default)]
    pub payload_indices: HashMap<String, PayloadIndexSchema>,
    #[serde(default)]
    pub named_vectors: HashMap<String, NamedVectorConfig>,
    #[serde(default)]
    pub dense_vector_spaces: HashMap<String, DenseVectorSpaceMetadata>,
    #[serde(default)]
    pub sparse_vectors: HashMap<String, SparseVectorConfig>,
    #[serde(default)]
    pub indexing_threshold_override: Option<usize>,
    pub stats: StorageStats,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StorageMetadataSidecar {
    pub sidecar_version: u32,
    pub written_at_millis: i64,
    pub data_mdb_bytes: u64,
    #[serde(default)]
    pub map_size_bytes: Option<u64>,
    pub metadata: StorageMetadata,
}

impl StorageMetadataSidecar {
    pub const VERSION: u32 = 1;

    pub fn new(
        metadata: StorageMetadata,
        data_mdb_bytes: u64,
        map_size_bytes: Option<u64>,
    ) -> Self {
        Self {
            sidecar_version: Self::VERSION,
            written_at_millis: chrono::Utc::now().timestamp_millis(),
            data_mdb_bytes,
            map_size_bytes,
            metadata,
        }
    }
}

impl StorageMetadata {
    pub fn new(secondary_indices: Vec<String>) -> Self {
        let now = chrono::Utc::now().timestamp_millis();
        let mut secondary_indices = secondary_indices;
        secondary_indices.sort();
        secondary_indices.dedup();

        Self {
            schema_version: CURRENT_STORAGE_SCHEMA_VERSION,
            created_at_millis: now,
            updated_at_millis: now,
            secondary_indices,
            payload_indices: HashMap::new(),
            named_vectors: HashMap::new(),
            dense_vector_spaces: HashMap::new(),
            sparse_vectors: HashMap::new(),
            indexing_threshold_override: None,
            stats: StorageStats::default(),
        }
    }

    pub fn touch(&mut self) {
        self.updated_at_millis = chrono::Utc::now().timestamp_millis();
    }

    pub fn set_secondary_indices(&mut self, secondary_indices: Vec<String>) {
        let mut secondary_indices = secondary_indices;
        secondary_indices.sort();
        secondary_indices.dedup();
        self.secondary_indices = secondary_indices;
        self.touch();
    }

    pub fn set_named_vectors(&mut self, named_vectors: HashMap<String, NamedVectorConfig>) {
        self.named_vectors = named_vectors;
        self.touch();
    }

    pub fn set_dense_vector_spaces(
        &mut self,
        dense_vector_spaces: HashMap<String, DenseVectorSpaceMetadata>,
    ) {
        self.dense_vector_spaces = dense_vector_spaces;
        self.touch();
    }

    pub fn set_payload_indices(&mut self, payload_indices: HashMap<String, PayloadIndexSchema>) {
        self.payload_indices = payload_indices;
        self.touch();
    }

    pub fn set_sparse_vectors(&mut self, sparse_vectors: HashMap<String, SparseVectorConfig>) {
        self.sparse_vectors = sparse_vectors;
        self.touch();
    }

    pub fn set_indexing_threshold_override(&mut self, indexing_threshold_override: Option<usize>) {
        self.indexing_threshold_override = indexing_threshold_override;
        self.touch();
    }

    pub fn adjust_counter(
        &mut self,
        counter: MetadataCounter,
        delta: i64,
    ) -> Result<(), GraphError> {
        let slot = match counter {
            MetadataCounter::Nodes => &mut self.stats.node_count,
            MetadataCounter::Edges => &mut self.stats.edge_count,
            MetadataCounter::Vectors => &mut self.stats.vector_count,
        };

        if delta >= 0 {
            *slot = slot
                .checked_add(delta as u64)
                .ok_or_else(|| GraphError::StorageError("metadata counter overflow".to_string()))?;
        } else {
            *slot = slot.checked_sub(delta.unsigned_abs()).ok_or_else(|| {
                GraphError::StorageError("metadata counter underflow".to_string())
            })?;
        }

        self.touch();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helix_engine::{
        graph_core::config::{Config, GraphConfig},
        storage_core::{
            storage_core::HelixGraphStorage,
            storage_methods::{DBMethods, StorageMethods},
        },
    };
    use tempfile::TempDir;

    fn setup_storage(config: Config) -> (HelixGraphStorage, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let storage = HelixGraphStorage::new(temp_dir.path().to_str().unwrap(), config).unwrap();
        (storage, temp_dir)
    }

    #[test]
    fn initializes_metadata_with_schema_version_and_indices() {
        let config = Config {
            vector_config: Config::default().vector_config,
            graph_config: GraphConfig {
                secondary_indices: Some(vec!["repo".to_string(), "name".to_string()]),
                snapshot_interval_secs: Some(3600),
                snapshot_keep_last: Some(3),
                raft: Default::default(),
            },
            storage_backend: Default::default(),
        };

        let (storage, _temp_dir) = setup_storage(config);
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let metadata = storage.get_metadata(&txn).unwrap();

        assert_eq!(metadata.schema_version, CURRENT_STORAGE_SCHEMA_VERSION);
        assert_eq!(
            metadata.secondary_indices,
            vec!["name".to_string(), "repo".to_string()]
        );
        assert_eq!(metadata.stats, StorageStats::default());
    }

    #[test]
    fn tracks_node_and_edge_counts() {
        let (storage, _temp_dir) = setup_storage(Config::default());

        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let node_a = storage
            .create_node(&mut txn, "User", vec![], None, None)
            .unwrap();
        let node_b = storage
            .create_node(&mut txn, "User", vec![], None, None)
            .unwrap();
        let edge = storage
            .create_edge(&mut txn, "Follows", &node_a.id, &node_b.id, vec![])
            .unwrap();
        txn.commit().unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let metadata = storage.get_metadata(&txn).unwrap();
        assert_eq!(edge.from_node, node_a.id);
        assert_eq!(edge.to_node, node_b.id);
        assert_eq!(metadata.stats.node_count, 2);
        assert_eq!(metadata.stats.edge_count, 1);
        assert_eq!(metadata.stats.vector_count, 0);
    }

    #[test]
    fn tracks_node_deletion() {
        let (storage, _temp_dir) = setup_storage(Config::default());

        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let node = storage
            .create_node(&mut txn, "User", vec![], None, None)
            .unwrap();
        storage.drop_node(&mut txn, &node.id).unwrap();
        txn.commit().unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let metadata = storage.get_metadata(&txn).unwrap();
        assert_eq!(metadata.stats.node_count, 0);
        assert_eq!(metadata.stats.edge_count, 0);
        assert_eq!(metadata.stats.vector_count, 0);
    }

    #[test]
    fn tracks_edge_deletion() {
        let (storage, _temp_dir) = setup_storage(Config::default());

        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        let node_a = storage
            .create_node(&mut txn, "User", vec![], None, None)
            .unwrap();
        let node_b = storage
            .create_node(&mut txn, "User", vec![], None, None)
            .unwrap();
        let edge = storage
            .create_edge(&mut txn, "Follows", &node_a.id, &node_b.id, vec![])
            .unwrap();
        storage.drop_edge(&mut txn, &edge.id).unwrap();
        txn.commit().unwrap();

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let metadata = storage.get_metadata(&txn).unwrap();
        assert_eq!(metadata.stats.node_count, 2);
        assert_eq!(metadata.stats.edge_count, 0);
        assert_eq!(metadata.stats.vector_count, 0);
    }

    #[test]
    fn updates_secondary_index_metadata() {
        let (mut storage, _temp_dir) = setup_storage(Config::default());

        storage.create_secondary_index("path").unwrap();
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let metadata = storage.get_metadata(&txn).unwrap();
        assert_eq!(metadata.secondary_indices, vec!["path".to_string()]);
        drop(txn);

        storage.drop_secondary_index("path").unwrap();
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let metadata = storage.get_metadata(&txn).unwrap();
        assert!(metadata.secondary_indices.is_empty());
    }
}
