use crate::helix_engine::storage_core::storage_core::HelixGraphStorage;
use crate::helix_engine::types::GraphError;
use std::path::Path;
use std::sync::Arc;

use crate::helix_engine::graph_core::config::Config;
use crate::helix_engine::storage_core::replication::{
    attach_dirty_retired_reaper_hook, submit_post_open_vector_maintenance,
};

#[derive(Debug)]
pub enum QueryInput {
    StringValue { value: String },
    IntegerValue { value: i32 },
    FloatValue { value: f64 },
    BooleanValue { value: bool },
}

pub struct HelixGraphEngine {
    pub storage: Arc<HelixGraphStorage>,
}

pub struct HelixGraphEngineOpts {
    pub path: String,
    pub config: Config,
}

impl HelixGraphEngineOpts {
    pub fn default() -> Self {
        Self {
            path: String::new(),
            config: Config::default(),
        }
    }
    pub fn with_path(path: String) -> Self {
        Self {
            path,
            config: Config::default(),
        }
    }
}

impl HelixGraphEngine {
    pub fn new(opts: HelixGraphEngineOpts) -> Result<HelixGraphEngine, GraphError> {
        let collection_name = Path::new(&opts.path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("default")
            .to_string();
        let storage = match HelixGraphStorage::new(opts.path.as_str(), opts.config.clone()) {
            Ok(db) => Arc::new(db),
            Err(err) => return Err(err),
        };
        attach_dirty_retired_reaper_hook(&storage);
        submit_post_open_vector_maintenance(&storage, &opts.config, &collection_name)?;
        Ok(Self { storage })
    }

    pub fn query(&self, _query: String, _params: Vec<QueryInput>) -> Result<String, GraphError> {
        Ok(String::new())
    }
}
