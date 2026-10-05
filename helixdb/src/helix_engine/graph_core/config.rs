use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::helix_engine::storage_core::backend::StorageBackendConfig;
use crate::helix_engine::types::GraphError;

pub const DEFAULT_SNAPSHOT_INTERVAL_SECS: u64 = 3600;
pub const DEFAULT_SNAPSHOT_KEEP_LAST: usize = 1;
pub const DEFAULT_RAFT_SNAPSHOT_ENTRIES: u64 = 1024;
pub const DEFAULT_RAFT_SNAPSHOT_CATCHUP_ENTRIES: u64 = 128;
pub const DEFAULT_VECTOR_FLAT_SCAN_THRESHOLD: usize = 16_384;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VectorConfig {
    // Maximum number of bi-directional links per element
    pub m: Option<usize>,

    // Size of dynamic candidate list for graph construction
    pub ef_construction: Option<usize>,

    // Size of dynamic candidate list for graph search
    pub ef_search: Option<usize>,

    // Database in GB
    pub db_max_size: Option<usize>,

    // Keep fresh collections in flat-search mode until they are large enough
    // to justify building HNSW once, instead of rewiring the graph per insert.
    #[serde(default)]
    pub flat_scan_threshold: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GraphConfig {
    pub secondary_indices: Option<Vec<String>>,
    #[serde(default)]
    pub snapshot_interval_secs: Option<u64>,
    #[serde(default)]
    pub snapshot_keep_last: Option<usize>,
    #[serde(default)]
    pub raft: RaftConfig,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct RaftPeerConfig {
    pub id: u64,
    pub address: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct RaftConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub node_id: Option<u64>,
    #[serde(default)]
    pub bind_address: Option<String>,
    #[serde(default)]
    pub peers: Vec<RaftPeerConfig>,
    #[serde(default)]
    pub snapshot_entries: Option<u64>,
    #[serde(default)]
    pub snapshot_catchup_entries: Option<u64>,
    /// Shared secret for authenticating inter-node Raft RPCs.
    /// If set, all `/_raft/*` requests must include a matching
    /// `X-Raft-Secret` header. Falls back to `HELIX_RAFT_SECRET` env var.
    #[serde(default)]
    pub raft_secret: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Config {
    pub vector_config: VectorConfig,
    pub graph_config: GraphConfig,
    #[serde(skip)]
    pub storage_backend: StorageBackendConfig,
    // // Path to the database
    // pub db_path: String,

    // // Port to run the server on
    // pub port: usize,
}

impl Config {
    pub fn new(m: usize, ef_construction: usize, ef_search: usize, db_max_size: usize) -> Self {
        Self {
            vector_config: VectorConfig {
                m: Some(m),
                ef_construction: Some(ef_construction),
                ef_search: Some(ef_search),
                db_max_size: Some(db_max_size),
                flat_scan_threshold: Some(DEFAULT_VECTOR_FLAT_SCAN_THRESHOLD),
            },
            graph_config: GraphConfig {
                secondary_indices: None,
                snapshot_interval_secs: Some(DEFAULT_SNAPSHOT_INTERVAL_SECS),
                snapshot_keep_last: Some(DEFAULT_SNAPSHOT_KEEP_LAST),
                raft: RaftConfig::default(),
            },
            storage_backend: default_storage_backend(),
        }
    }

    pub fn with_storage_backend(mut self, storage_backend: StorageBackendConfig) -> Self {
        self.storage_backend = storage_backend;
        self
    }

    pub fn with_lsm_in_memory(self) -> Self {
        self.with_storage_backend(StorageBackendConfig::lsm_in_memory())
    }

    pub fn with_env_storage_overrides(mut self) -> Self {
        self.storage_backend = StorageBackendConfig::from_env();
        self
    }

    pub fn snapshot_interval_secs(&self) -> u64 {
        if let Some(value) = env_u64("HELIX_SNAPSHOT_INTERVAL_SECS") {
            return value;
        }
        self.graph_config
            .snapshot_interval_secs
            .unwrap_or(DEFAULT_SNAPSHOT_INTERVAL_SECS)
            .max(1)
    }

    pub fn vector_flat_scan_threshold(&self) -> usize {
        self.vector_config
            .flat_scan_threshold
            .unwrap_or(DEFAULT_VECTOR_FLAT_SCAN_THRESHOLD)
    }

    pub fn snapshot_keep_last(&self) -> usize {
        if let Some(value) = env_usize("HELIX_SNAPSHOT_KEEP_LAST") {
            return value.max(1);
        }
        self.graph_config
            .snapshot_keep_last
            .unwrap_or(DEFAULT_SNAPSHOT_KEEP_LAST)
            .max(1)
    }

    pub fn snapshot_on_shutdown(&self) -> bool {
        env_bool("HELIX_SNAPSHOT_ON_SHUTDOWN").unwrap_or(true)
    }

    pub fn raft_snapshot_entries(&self) -> u64 {
        self.graph_config
            .raft
            .snapshot_entries
            .unwrap_or(DEFAULT_RAFT_SNAPSHOT_ENTRIES)
            .max(1)
    }

    pub fn raft_snapshot_catchup_entries(&self) -> u64 {
        self.graph_config
            .raft
            .snapshot_catchup_entries
            .unwrap_or(DEFAULT_RAFT_SNAPSHOT_CATCHUP_ENTRIES)
            .max(1)
    }

    /// Returns the Raft shared secret for inter-node auth.
    /// Prefers the config field, falls back to `HELIX_RAFT_SECRET` env var.
    pub fn raft_secret(&self) -> Option<String> {
        self.graph_config
            .raft
            .raft_secret
            .clone()
            .or_else(|| std::env::var("HELIX_RAFT_SECRET").ok())
    }

    pub fn from_config_file(input_path: PathBuf) -> Result<Self, GraphError> {
        if !input_path.exists() {
            return Err(GraphError::ConfigFileNotFound);
        }
        let config = std::fs::read_to_string(input_path)?;
        let config = sonic_rs::from_str::<Config>(&config)?.with_env_storage_overrides();

        Ok(config)
    }

    pub fn init_config() -> String {
        r#"
        {
    "vector_config": {
        "m": 16,
        "ef_construction": 128,
        "ef_search": 128,
        "db_max_size": 100,
        "flat_scan_threshold": 4096
    },
    "graph_config": {
        "secondary_indices": [],
        "snapshot_interval_secs": 3600,
        "snapshot_keep_last": 1,
        "raft": {
            "enabled": false,
            "node_id": 1,
            "bind_address": "http://127.0.0.1:6969",
            "snapshot_entries": 1024,
            "snapshot_catchup_entries": 128,
            "peers": [
                { "id": 1, "address": "http://127.0.0.1:6969" }
            ]
        }
    }
}
"#
        .to_string()
    }
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.parse::<u64>().ok()
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse::<usize>().ok()
}

fn env_bool(name: &str) -> Option<bool> {
    let value = std::env::var(name).ok()?;
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
fn default_storage_backend() -> StorageBackendConfig {
    StorageBackendConfig::Lmdb
}

#[cfg(not(test))]
fn default_storage_backend() -> StorageBackendConfig {
    StorageBackendConfig::from_env()
}

/// Default initial LMDB map size in MB.  Sparse on 64-bit (only committed
/// pages consume RAM), but each open env reserves virtual address space.
/// At 5 000+ collections the total VA must stay within the OS limit
/// (~128 TB on Linux, ~18 TB on macOS). 64 MB × 5 000 = 320 GB VA which
/// is safe everywhere, and grow_map expands by HELIX_MAP_GROW_MB on MapFull.
pub const DEFAULT_DB_INITIAL_MAP_MB: usize = 64;

impl Default for Config {
    fn default() -> Self {
        Self {
            vector_config: VectorConfig {
                m: Some(25),
                ef_construction: Some(512),
                ef_search: Some(128),
                db_max_size: None, // use DEFAULT_DB_INITIAL_MAP_MB via storage_core
                flat_scan_threshold: Some(DEFAULT_VECTOR_FLAT_SCAN_THRESHOLD),
            },
            graph_config: GraphConfig {
                secondary_indices: None,
                snapshot_interval_secs: Some(DEFAULT_SNAPSHOT_INTERVAL_SECS),
                snapshot_keep_last: Some(DEFAULT_SNAPSHOT_KEEP_LAST),
                raft: RaftConfig::default(),
            },
            storage_backend: default_storage_backend(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    #[test]
    fn raft_snapshot_thresholds_are_clamped() {
        let mut config = Config::default();
        config.graph_config.raft.snapshot_entries = Some(0);
        config.graph_config.raft.snapshot_catchup_entries = Some(0);

        assert_eq!(config.raft_snapshot_entries(), 1);
        assert_eq!(config.raft_snapshot_catchup_entries(), 1);
    }

    #[test]
    fn snapshot_interval_is_clamped_to_one_second() {
        let _guard = env_lock();
        std::env::remove_var("HELIX_SNAPSHOT_INTERVAL_SECS");
        let config = Config {
            vector_config: VectorConfig {
                m: None,
                ef_construction: None,
                ef_search: None,
                db_max_size: None,
                flat_scan_threshold: None,
            },
            graph_config: GraphConfig {
                secondary_indices: None,
                snapshot_interval_secs: Some(0),
                snapshot_keep_last: None,
                raft: RaftConfig::default(),
            },
            storage_backend: StorageBackendConfig::Lmdb,
        };

        assert_eq!(config.snapshot_interval_secs(), 1);
    }

    #[test]
    fn snapshot_env_overrides_config() {
        let _guard = env_lock();
        std::env::set_var("HELIX_SNAPSHOT_INTERVAL_SECS", "0");
        std::env::set_var("HELIX_SNAPSHOT_KEEP_LAST", "2");
        std::env::set_var("HELIX_SNAPSHOT_ON_SHUTDOWN", "0");

        let config = Config::default();
        assert_eq!(config.snapshot_interval_secs(), 0);
        assert_eq!(config.snapshot_keep_last(), 2);
        assert!(!config.snapshot_on_shutdown());

        std::env::remove_var("HELIX_SNAPSHOT_INTERVAL_SECS");
        std::env::remove_var("HELIX_SNAPSHOT_KEEP_LAST");
        std::env::remove_var("HELIX_SNAPSHOT_ON_SHUTDOWN");
    }
}
