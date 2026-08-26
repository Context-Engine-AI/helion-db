use std::collections::HashMap;
use std::io::{BufRead, Cursor};
use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt};

use crate::helix_engine::storage_core::backend_lsm::allow_lsm_blocking;
use crate::helix_engine::storage_core::metadata::PayloadIndexSchema;
use crate::helix_engine::storage_core::storage_core::HelixGraphStorage;
use crate::helix_engine::storage_core::upsert::{EdgeUpsert, NodeUpsert};
use crate::helix_engine::storage_core::{
    collection_manager::CollectionManager,
    replication::{ReplicatedIngestOp, ReplicatedMutation, ReplicationManager},
};

/// Keyword payload fields that every graph ingest stream guarantees exist.
/// The `/v1/graph/distinct` endpoint requires a keyword payload index to
/// serve queries without a full scan (see handle_distinct_values), and
/// clients historically had to remember to PUT /collections/{name}/index
/// for each field — a contract that legacy collections kept violating
/// (observed 2026-04-18: _graph collections in prod missing the `kind`
/// index). Auto-creating these alongside the ingest stream closes the gap:
/// if your data is coming through handle_ingest_stream, distinct queries
/// will work immediately and stay working across reboots.
const GRAPH_NODE_KEYWORD_INDEX_FIELDS: &[&str] = &["repo", "kind", "label", "path"];
use crate::helix_engine::types::GraphError;
use crate::helix_gateway::router::router::HandlerInput;
use crate::protocol::deterministic_id;
use crate::protocol::request::RequestHead;
use crate::protocol::response::Response;
use crate::protocol::value::Value;

const STREAM_BATCH_SIZE: usize = 1_000;

#[derive(Deserialize)]
pub struct IngestNodesRequest {
    pub collection: String,
    pub nodes: Vec<NodeInput>,
}

#[derive(Deserialize)]
pub struct NodeInput {
    pub label: String,
    pub name: String,
    pub path: String,
    #[serde(default)]
    pub properties: HashMap<String, Value>,
}

#[derive(Deserialize)]
pub struct IngestEdgesRequest {
    pub collection: String,
    pub edges: Vec<EdgeInput>,
}

#[derive(Deserialize)]
pub struct EdgeInput {
    pub edge_type: String,
    pub from_name: String,
    pub to_name: String,
    pub from_path: String,
    pub to_path: String,
    /// Label of the from node (default: "Symbol")
    #[serde(default = "default_label")]
    pub from_label: String,
    /// Label of the to node (default: "Symbol")
    #[serde(default = "default_label")]
    pub to_label: String,
    #[serde(default)]
    pub repo: Option<String>,
    #[serde(default)]
    pub git_branches: Vec<String>,
    #[serde(default)]
    pub properties: HashMap<String, Value>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamRecord {
    Node(NodeInput),
    Edge(EdgeInput),
}

#[derive(Clone)]
enum StreamOp {
    Node(NodeUpsert),
    Edge(EdgeUpsert),
}

fn default_label() -> String {
    "Symbol".into()
}

fn json_response(
    response: &mut Response,
    status: u16,
    body: &impl serde::Serialize,
) -> Result<(), GraphError> {
    response.status = status;
    response.body = sonic_rs::to_vec(body)?;
    response
        .headers
        .insert("Content-Type".to_string(), "application/json".to_string());
    Ok(())
}

fn json_error(response: &mut Response, status: u16, msg: &str) -> Result<(), GraphError> {
    json_response(response, status, &sonic_rs::json!({ "error": msg }))
}

fn build_node_upsert(collection: &str, node: &NodeInput) -> NodeUpsert {
    let id = deterministic_id::node_id(collection, &node.label, &node.name, &node.path);
    let mut props = node.properties.clone();
    // Ensure core fields are in properties for querying.
    props
        .entry("name".into())
        .or_insert_with(|| Value::String(node.name.clone()));
    props
        .entry("path".into())
        .or_insert_with(|| Value::String(node.path.clone()));
    props
        .entry("label".into())
        .or_insert_with(|| Value::String(node.label.clone()));
    props
        .entry("collection".into())
        .or_insert_with(|| Value::String(collection.to_string()));

    NodeUpsert {
        id,
        label: node.label.clone(),
        properties: props,
    }
}

fn branches_value(branches: &[String]) -> Value {
    Value::Array(branches.iter().cloned().map(Value::String).collect())
}

/// Resolve a node ID: if `path` is non-empty use the deterministic ID.
/// Otherwise try the `name` multi-index to find an existing node first.
fn resolve_node_id(
    collection: &str,
    label: &str,
    name: &str,
    path: &str,
    storage: Option<&HelixGraphStorage>,
) -> u128 {
    if path.is_empty() {
        if let Some(s) = storage {
            if let Ok(Some(id)) = s.with_read_txn(|txn| {
                let ids =
                    s.get_nodes_by_multi_index(txn, "name", &Value::String(name.to_string()))?;
                Ok(ids.first().copied())
            }) {
                return id;
            }
        }
    }
    deterministic_id::node_id(collection, label, name, path)
}

fn build_edge_upsert(
    collection: &str,
    edge: &EdgeInput,
    storage: Option<&HelixGraphStorage>,
) -> EdgeUpsert {
    let from_node = resolve_node_id(
        collection,
        &edge.from_label,
        &edge.from_name,
        &edge.from_path,
        storage,
    );
    let to_node = resolve_node_id(
        collection,
        &edge.to_label,
        &edge.to_name,
        &edge.to_path,
        storage,
    );
    let id = deterministic_id::edge_id(
        collection,
        &edge.edge_type,
        &edge.from_name,
        &edge.to_name,
        &edge.from_path,
        &edge.to_path,
    );

    let mut props = edge.properties.clone();
    props
        .entry("edge_type".into())
        .or_insert_with(|| Value::String(edge.edge_type.clone()));
    props
        .entry("from_name".into())
        .or_insert_with(|| Value::String(edge.from_name.clone()));
    props
        .entry("to_name".into())
        .or_insert_with(|| Value::String(edge.to_name.clone()));
    props
        .entry("from_path".into())
        .or_insert_with(|| Value::String(edge.from_path.clone()));
    props
        .entry("to_path".into())
        .or_insert_with(|| Value::String(edge.to_path.clone()));
    if let Some(ref repo) = edge.repo {
        props
            .entry("repo".into())
            .or_insert_with(|| Value::String(repo.clone()));
    }
    if !edge.git_branches.is_empty() {
        props.insert("git_branches".into(), branches_value(&edge.git_branches));
    }

    EdgeUpsert {
        id,
        label: edge.edge_type.clone(),
        from_node,
        to_node,
        properties: props,
    }
}

fn flush_stream_batch(
    replication: &Arc<ReplicationManager>,
    collection: &str,
    batch: &mut Vec<StreamOp>,
) -> Result<(usize, usize), GraphError> {
    if batch.is_empty() {
        return Ok((0, 0));
    }

    let ops = std::mem::take(batch);
    let mut nodes_upserted = 0usize;
    let mut edges_upserted = 0usize;
    let replicated_ops: Vec<ReplicatedIngestOp> = ops
        .into_iter()
        .map(|op| match op {
            StreamOp::Node(node) => {
                nodes_upserted += 1;
                ReplicatedIngestOp::Node(node)
            }
            StreamOp::Edge(edge) => {
                edges_upserted += 1;
                ReplicatedIngestOp::Edge(edge)
            }
        })
        .collect();

    // apply_ingest persists the batch atomically and returns the applied count;
    // on any failure it returns Err (propagated below), so the per-kind counts
    // are only reported when the whole batch landed.
    let applied = replication.apply_ingest(collection.to_string(), replicated_ops)?;
    debug_assert_eq!(applied, nodes_upserted + edges_upserted);
    Ok((nodes_upserted, edges_upserted))
}

async fn flush_stream_batch_async(
    replication: Arc<ReplicationManager>,
    collection: String,
    batch: &mut Vec<StreamOp>,
) -> Result<(usize, usize), GraphError> {
    let mut to_flush = std::mem::take(batch);
    tokio::task::spawn_blocking(move || {
        allow_lsm_blocking(|| flush_stream_batch(&replication, &collection, &mut to_flush))
    })
    .await
    .map_err(|e| GraphError::New(format!("stream batch task failed: {}", e)))?
}

fn build_stream_ingest_response(
    processed: usize,
    nodes_upserted: usize,
    edges_upserted: usize,
    batches: usize,
) -> Result<Response, GraphError> {
    let mut response = Response::new();
    json_response(
        &mut response,
        200,
        &sonic_rs::json!({
            "processed": processed,
            "nodes_upserted": nodes_upserted,
            "edges_upserted": edges_upserted,
            "batches": batches,
        }),
    )?;
    Ok(response)
}

fn build_error_response(status: u16, msg: &str) -> Result<Response, GraphError> {
    let mut response = Response::new();
    json_error(&mut response, status, msg)?;
    Ok(response)
}

fn ensure_collection_exists(
    replication: &ReplicationManager,
    collection: &str,
) -> Result<(), GraphError> {
    replication.apply(ReplicatedMutation::CreateCollection {
        name: collection.to_string(),
        vectors: std::collections::HashMap::new(),
        sparse_vectors: std::collections::HashMap::new(),
        hnsw_overrides: None,
    })?;
    ensure_graph_node_keyword_indexes(replication, collection);
    Ok(())
}

/// Best-effort: ensure a keyword payload index exists on every field in
/// GRAPH_NODE_KEYWORD_INDEX_FIELDS. CreatePayloadIndex is idempotent at
/// the storage layer, so this is a cheap no-op on established collections
/// and a self-heal for legacy ones that predate a given field being added
/// to this list. Failures are logged (they'd surface as later distinct
/// queries returning 400) but never block ingest — the point of ingest is
/// to land the data; a missing index only degrades a secondary endpoint.
fn ensure_graph_node_keyword_indexes(replication: &ReplicationManager, collection: &str) {
    for field_name in GRAPH_NODE_KEYWORD_INDEX_FIELDS {
        if let Err(e) = replication.apply(ReplicatedMutation::CreatePayloadIndex {
            collection: collection.to_string(),
            field_name: (*field_name).to_string(),
            schema: PayloadIndexSchema::Keyword,
        }) {
            tracing::warn!(
                "ensure_graph_node_keyword_indexes: {} on {} failed (non-fatal): {}",
                field_name,
                collection,
                e
            );
        }
    }
}

async fn ensure_collection_exists_async(
    replication: Arc<ReplicationManager>,
    collection: String,
) -> Result<(), GraphError> {
    tokio::task::spawn_blocking(move || {
        allow_lsm_blocking(|| ensure_collection_exists(&replication, &collection))
    })
    .await
    .map_err(|e| GraphError::New(format!("ensure collection task failed: {}", e)))?
}

pub fn stream_collection_name(path: &str) -> Option<String> {
    let trimmed = path.trim_matches('/');
    let mut segments = trimmed.split('/');
    match (
        segments.next(),
        segments.next(),
        segments.next(),
        segments.next(),
        segments.next(),
        segments.next(),
    ) {
        (Some("v1"), Some("collections"), Some(name), Some("ingest"), Some("stream"), None)
            if !name.is_empty() =>
        {
            Some(name.to_string())
        }
        _ => None,
    }
}

pub async fn handle_ingest_stream_socket<R: AsyncBufRead + Unpin>(
    request_head: RequestHead,
    reader: &mut R,
    collection: String,
    collections: Arc<CollectionManager>,
    replication: Arc<ReplicationManager>,
) -> Result<Response, GraphError> {
    let Some(content_length) = request_head.content_length() else {
        return build_error_response(400, "Missing Content-Length header");
    };

    ensure_collection_exists_async(Arc::clone(&replication), collection.clone()).await?;
    // Get storage reference for name-based node resolution
    let storage = collections.get_collection(&collection).ok();
    let mut limited = AsyncReadExt::take(reader, content_length as u64);
    let mut batch = Vec::with_capacity(STREAM_BATCH_SIZE);
    let mut nodes_upserted = 0usize;
    let mut edges_upserted = 0usize;
    let mut processed = 0usize;
    let mut batches = 0usize;
    let mut line_no = 0usize;
    let mut line = String::new();

    loop {
        line.clear();
        let read_result =
            tokio::time::timeout(Duration::from_secs(30), limited.read_line(&mut line))
                .await
                .map_err(|_| GraphError::New("Timed out reading NDJSON stream".into()))?;
        let bytes_read = read_result?;
        if bytes_read == 0 {
            break;
        }

        line_no += 1;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let record: StreamRecord = match sonic_rs::from_str(trimmed) {
            Ok(record) => record,
            Err(e) => {
                return build_error_response(
                    400,
                    &format!("Invalid NDJSON record on line {}: {}", line_no, e),
                )
            }
        };

        match record {
            StreamRecord::Node(node) => {
                batch.push(StreamOp::Node(build_node_upsert(&collection, &node)))
            }
            StreamRecord::Edge(edge) => batch.push(StreamOp::Edge(build_edge_upsert(
                &collection,
                &edge,
                storage.as_deref(),
            ))),
        }
        processed += 1;

        if batch.len() >= STREAM_BATCH_SIZE {
            let (nodes, edges) =
                flush_stream_batch_async(Arc::clone(&replication), collection.clone(), &mut batch)
                    .await?;
            nodes_upserted += nodes;
            edges_upserted += edges;
            batches += 1;
        }
    }

    if !batch.is_empty() {
        let (nodes, edges) =
            flush_stream_batch_async(replication, collection.clone(), &mut batch).await?;
        nodes_upserted += nodes;
        edges_upserted += edges;
        batches += 1;
    }

    build_stream_ingest_response(processed, nodes_upserted, edges_upserted, batches)
}

pub fn handle_ingest_nodes(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let req: IngestNodesRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_error(response, 400, &format!("Invalid JSON: {}", e)),
    };

    if let Err(e) = ensure_collection_exists(&input.replication, &req.collection) {
        return json_error(response, 500, &e.to_string());
    }

    let ops: Vec<ReplicatedIngestOp> = req
        .nodes
        .iter()
        .map(|n| ReplicatedIngestOp::Node(build_node_upsert(&req.collection, n)))
        .collect();
    // Report the count the apply actually persisted, not a blind req.len(): a
    // failed batch returns Err (surfaced to the caller) instead of a false
    // success.
    let count = input.replication.apply_ingest(req.collection, ops)?;

    json_response(response, 200, &sonic_rs::json!({ "upserted": count }))
}

pub fn handle_ingest_edges(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let req: IngestEdgesRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_error(response, 400, &format!("Invalid JSON: {}", e)),
    };

    if let Err(e) = ensure_collection_exists(&input.replication, &req.collection) {
        return json_error(response, 500, &e.to_string());
    }

    let storage = input.collections.get_collection(&req.collection).ok();
    let ops: Vec<ReplicatedIngestOp> = req
        .edges
        .iter()
        .map(|e| {
            ReplicatedIngestOp::Edge(build_edge_upsert(&req.collection, e, storage.as_deref()))
        })
        .collect();
    // Report the count the apply actually persisted, not a blind req.len(): a
    // failed batch returns Err (surfaced to the caller) instead of a false
    // success.
    let count = input.replication.apply_ingest(req.collection, ops)?;

    json_response(response, 200, &sonic_rs::json!({ "upserted": count }))
}

/// POST /v1/collections/{name}/ingest/stream
///
/// Line-oriented bulk ingest using NDJSON. Each line must be either:
/// `{"type":"node", ...}` or `{"type":"edge", ...}`.
pub fn handle_ingest_stream(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let collection = match input.path_params.get("name") {
        Some(name) => name.clone(),
        None => return json_error(response, 400, "Missing collection name"),
    };

    if let Err(e) = ensure_collection_exists(&input.replication, &collection) {
        return json_error(response, 500, &e.to_string());
    }

    let storage = input.collections.get_collection(&collection).ok();
    let reader = Cursor::new(&input.request.body);
    let mut batch = Vec::with_capacity(STREAM_BATCH_SIZE);
    let mut nodes_upserted = 0usize;
    let mut edges_upserted = 0usize;
    let mut processed = 0usize;
    let mut batches = 0usize;

    for (line_no, line) in BufRead::lines(reader).enumerate() {
        let line = match line {
            Ok(line) => line,
            Err(e) => {
                return json_error(
                    response,
                    400,
                    &format!("Invalid stream line {}: {}", line_no + 1, e),
                )
            }
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let record: StreamRecord = match sonic_rs::from_str(line) {
            Ok(record) => record,
            Err(e) => {
                return json_error(
                    response,
                    400,
                    &format!("Invalid NDJSON record on line {}: {}", line_no + 1, e),
                )
            }
        };

        match record {
            StreamRecord::Node(node) => {
                batch.push(StreamOp::Node(build_node_upsert(&collection, &node)))
            }
            StreamRecord::Edge(edge) => batch.push(StreamOp::Edge(build_edge_upsert(
                &collection,
                &edge,
                storage.as_deref(),
            ))),
        }
        processed += 1;

        if batch.len() >= STREAM_BATCH_SIZE {
            let (nodes, edges) = flush_stream_batch(&input.replication, &collection, &mut batch)?;
            nodes_upserted += nodes;
            edges_upserted += edges;
            batches += 1;
        }
    }

    if !batch.is_empty() {
        let (nodes, edges) = flush_stream_batch(&input.replication, &collection, &mut batch)?;
        nodes_upserted += nodes;
        edges_upserted += edges;
        batches += 1;
    }

    let summary = build_stream_ingest_response(processed, nodes_upserted, edges_upserted, batches)?;
    response.status = summary.status;
    response.headers = summary.headers;
    response.body = summary.body;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helix_engine::graph_core::config::Config;
    use crate::helix_engine::graph_core::graph_core::{HelixGraphEngine, HelixGraphEngineOpts};
    use crate::helix_engine::storage_core::backend::{BackendKind, StorageBackend};
    use crate::helix_engine::storage_core::storage_methods::StorageMethods;
    use crate::helix_engine::storage_core::{
        collection_manager::CollectionManager, replication::ReplicationManager,
    };
    use crate::helix_gateway::router::router::HandlerInput;
    use crate::protocol::request::{Request, RequestHead};
    use sonic_rs::JsonValueTrait;
    use tempfile::TempDir;
    use tokio::io::BufReader;

    /// Serializes tests that mutate process-global storage-backend env vars
    /// (`HELIX_STORAGE_BACKEND`, `HELIX_LSM_IN_MEMORY`). Without it, an LSM test's
    /// env leaks into a concurrently-running heed test (or vice versa) because
    /// `std::env::set_var` is process-wide. Mirrors the `ENV_LOCK` used in
    /// `replication.rs`. Lock it BEFORE constructing the `EnvGuard`s and hold the
    /// guard for the whole test body.
    static ENV_LOCK: std::sync::LazyLock<std::sync::Mutex<()>> =
        std::sync::LazyLock::new(|| std::sync::Mutex::new(()));

    struct EnvGuard(&'static str, Option<String>);
    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self(key, prev)
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.1 {
                Some(v) => std::env::set_var(self.0, v),
                None => std::env::remove_var(self.0),
            }
        }
    }

    struct TestContext {
        _tmp: TempDir,
        graph: Arc<HelixGraphEngine>,
        collections: Arc<CollectionManager>,
        replication: Arc<ReplicationManager>,
    }

    fn setup() -> TestContext {
        let tmp = TempDir::new().unwrap();
        let graph_path = tmp.path().join("graph");
        let collections_path = tmp.path().join("data");
        let graph = Arc::new(
            HelixGraphEngine::new(HelixGraphEngineOpts {
                path: graph_path.display().to_string(),
                config: Config::default(),
            })
            .unwrap(),
        );
        let collections =
            Arc::new(CollectionManager::new(collections_path, Config::default()).unwrap());
        let replication =
            Arc::new(ReplicationManager::new(Arc::clone(&collections), Config::default()).unwrap());
        TestContext {
            _tmp: tmp,
            graph,
            collections,
            replication,
        }
    }

    fn make_input(
        ctx: &TestContext,
        path: &str,
        body: Vec<u8>,
        path_params: HashMap<String, String>,
    ) -> HandlerInput {
        HandlerInput {
            request: Request {
                method: "POST".into(),
                headers: HashMap::new(),
                path: path.into(),
                body,
            },
            graph: Arc::clone(&ctx.graph),
            collections: Arc::clone(&ctx.collections),
            replication: Arc::clone(&ctx.replication),
            path_params,
        }
    }

    #[test]
    fn edge_ingest_preserves_git_branches() {
        let _env = ENV_LOCK.lock().unwrap();
        let ctx = setup();
        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "collection": "repo",
            "edges": [{
                "edge_type": "CALLS",
                "from_name": "main",
                "to_name": "helper",
                "from_path": "src/main.rs",
                "to_path": "src/lib.rs",
                "git_branches": ["main", "feature/auth"]
            }]
        }))
        .unwrap();
        let input = make_input(&ctx, "/v1/ingest/edges", body, HashMap::new());
        let mut response = Response::new();

        handle_ingest_edges(&input, &mut response).unwrap();
        assert_eq!(response.status, 200);

        let storage = ctx.collections.get_collection("repo").unwrap();
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let edge_id = deterministic_id::edge_id(
            "repo",
            "CALLS",
            "main",
            "helper",
            "src/main.rs",
            "src/lib.rs",
        );
        let edge = storage.get_edge(&txn, &edge_id).unwrap();
        assert_eq!(
            edge.properties.get("git_branches"),
            Some(&Value::Array(vec![
                Value::String("main".into()),
                Value::String("feature/auth".into()),
            ]))
        );
    }

    #[test]
    fn stream_ingest_processes_nodes_and_edges_in_batches() {
        let _env = ENV_LOCK.lock().unwrap();
        let ctx = setup();
        let body = [
            r#"{"type":"node","label":"Symbol","name":"main","path":"src/main.rs"}"#,
            r#"{"type":"node","label":"Symbol","name":"helper","path":"src/lib.rs"}"#,
            r#"{"type":"edge","edge_type":"CALLS","from_name":"main","to_name":"helper","from_path":"src/main.rs","to_path":"src/lib.rs","git_branches":["main"]}"#,
        ]
        .join("\n")
        .into_bytes();
        let input = make_input(
            &ctx,
            "/v1/collections/repo/ingest/stream",
            body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut response = Response::new();

        handle_ingest_stream(&input, &mut response).unwrap();
        assert_eq!(response.status, 200);

        let storage = ctx.collections.get_collection("repo").unwrap();
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let metadata = storage.get_metadata(&txn).unwrap();
        assert_eq!(metadata.stats.node_count, 2);
        assert_eq!(metadata.stats.edge_count, 1);
    }

    #[test]
    fn stream_ingest_writes_nodes_and_edges_on_lsm_backend() {
        let _env = ENV_LOCK.lock().unwrap();
        let _backend = EnvGuard::set("HELIX_STORAGE_BACKEND", "lsm");
        let _in_mem = EnvGuard::set("HELIX_LSM_IN_MEMORY", "1");
        let ctx = setup();
        let body = [
            r#"{"type":"node","label":"Symbol","name":"main","path":"src/main.rs","properties":{"repo":"repo-a"}}"#,
            r#"{"type":"node","label":"Symbol","name":"helper","path":"src/lib.rs","properties":{"repo":"repo-a"}}"#,
            r#"{"type":"edge","edge_type":"CALLS","from_name":"main","to_name":"helper","from_path":"src/main.rs","to_path":"src/lib.rs","repo":"repo-a","git_branches":["main"]}"#,
        ]
        .join("\n")
        .into_bytes();
        let input = make_input(
            &ctx,
            "/v1/collections/repo_lsm/ingest/stream",
            body,
            HashMap::from([("name".into(), "repo_lsm".into())]),
        );
        let mut response = Response::new();

        handle_ingest_stream(&input, &mut response).unwrap();
        assert_eq!(response.status, 200);

        let storage = ctx.collections.get_collection("repo_lsm").unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);
        let r = storage.backend.begin_read().unwrap();
        let main_id = deterministic_id::node_id("repo_lsm", "Symbol", "main", "src/main.rs");
        let helper_id = deterministic_id::node_id("repo_lsm", "Symbol", "helper", "src/lib.rs");
        let edge_id = deterministic_id::edge_id(
            "repo_lsm",
            "CALLS",
            "main",
            "helper",
            "src/main.rs",
            "src/lib.rs",
        );

        assert!(storage.get_node_be(&r, &main_id).is_ok());
        assert!(storage.get_node_be(&r, &helper_id).is_ok());
        let edge = storage.get_edge_be(&r, &edge_id).unwrap();
        assert_eq!(
            edge.properties.get("repo"),
            Some(&Value::String("repo-a".into()))
        );
    }

    /// Regression for the live "fresh LSM collection ingests 2 nodes + 1 edge,
    /// handler returns upserted success, but collection stats stay 0/0" anomaly.
    /// The LSM write path bumps the metadata counters correctly, but
    /// `collection_stats` reads the in-memory `metadata_snapshot`, which was
    /// never refreshed after an LSM backend commit. `collection_stats` must
    /// reflect the just-ingested counts (and the handler must report the real
    /// applied count).
    #[test]
    fn stream_ingest_refreshes_collection_stats_on_lsm_backend() {
        let _env = ENV_LOCK.lock().unwrap();
        let _backend = EnvGuard::set("HELIX_STORAGE_BACKEND", "lsm");
        let _in_mem = EnvGuard::set("HELIX_LSM_IN_MEMORY", "1");
        let ctx = setup();
        let body = [
            r#"{"type":"node","label":"Symbol","name":"main","path":"src/main.rs"}"#,
            r#"{"type":"node","label":"Symbol","name":"helper","path":"src/lib.rs"}"#,
            r#"{"type":"edge","edge_type":"CALLS","from_name":"main","to_name":"helper","from_path":"src/main.rs","to_path":"src/lib.rs"}"#,
        ]
        .join("\n")
        .into_bytes();
        let input = make_input(
            &ctx,
            "/v1/collections/repo_stats/ingest/stream",
            body,
            HashMap::from([("name".into(), "repo_stats".into())]),
        );
        let mut response = Response::new();

        handle_ingest_stream(&input, &mut response).unwrap();
        assert_eq!(response.status, 200);

        let stats = ctx.collections.collection_stats("repo_stats").unwrap();
        assert_eq!(stats.node_count, 2, "node_count must reflect LSM ingest");
        assert_eq!(stats.edge_count, 1, "edge_count must reflect LSM ingest");
    }

    /// The node/edge ingest handlers report the count the apply actually
    /// persisted, and `collection_stats` reflects it, on the LSM backend.
    #[test]
    fn handle_ingest_nodes_edges_report_applied_count_on_lsm_backend() {
        let _env = ENV_LOCK.lock().unwrap();
        let _backend = EnvGuard::set("HELIX_STORAGE_BACKEND", "lsm");
        let _in_mem = EnvGuard::set("HELIX_LSM_IN_MEMORY", "1");
        let ctx = setup();

        let nodes_body = sonic_rs::to_vec(&sonic_rs::json!({
            "collection": "repo_count",
            "nodes": [
                {"label":"Symbol","name":"main","path":"src/main.rs"},
                {"label":"Symbol","name":"helper","path":"src/lib.rs"}
            ]
        }))
        .unwrap();
        let nodes_input = make_input(&ctx, "/v1/ingest/nodes", nodes_body, HashMap::new());
        let mut nodes_response = Response::new();
        handle_ingest_nodes(&nodes_input, &mut nodes_response).unwrap();
        assert_eq!(nodes_response.status, 200);
        let nodes_json: sonic_rs::Value = sonic_rs::from_slice(&nodes_response.body).unwrap();
        assert_eq!(nodes_json["upserted"].as_u64(), Some(2));

        let edges_body = sonic_rs::to_vec(&sonic_rs::json!({
            "collection": "repo_count",
            "edges": [{
                "edge_type":"CALLS",
                "from_name":"main","to_name":"helper",
                "from_path":"src/main.rs","to_path":"src/lib.rs"
            }]
        }))
        .unwrap();
        let edges_input = make_input(&ctx, "/v1/ingest/edges", edges_body, HashMap::new());
        let mut edges_response = Response::new();
        handle_ingest_edges(&edges_input, &mut edges_response).unwrap();
        assert_eq!(edges_response.status, 200);
        let edges_json: sonic_rs::Value = sonic_rs::from_slice(&edges_response.body).unwrap();
        assert_eq!(edges_json["upserted"].as_u64(), Some(1));

        let stats = ctx.collections.collection_stats("repo_count").unwrap();
        assert_eq!(stats.node_count, 2);
        assert_eq!(stats.edge_count, 1);
    }

    #[tokio::test]
    async fn stream_socket_ingest_processes_body_incrementally() {
        let _env = ENV_LOCK.lock().unwrap();
        let ctx = setup();
        let body = [
            r#"{"type":"node","label":"Symbol","name":"main","path":"src/main.rs"}"#,
            r#"{"type":"node","label":"Symbol","name":"helper","path":"src/lib.rs"}"#,
            r#"{"type":"edge","edge_type":"CALLS","from_name":"main","to_name":"helper","from_path":"src/main.rs","to_path":"src/lib.rs","git_branches":["main"]}"#,
        ]
        .join("\n");
        let bytes = body.as_bytes().to_vec();
        let mut reader = BufReader::new(Cursor::new(bytes.clone()));
        let request_head = RequestHead {
            method: "POST".into(),
            headers: HashMap::from([("content-length".into(), bytes.len().to_string())]),
            path: "/v1/collections/repo/ingest/stream".into(),
        };

        let response = handle_ingest_stream_socket(
            request_head,
            &mut reader,
            "repo".into(),
            Arc::clone(&ctx.collections),
            Arc::clone(&ctx.replication),
        )
        .await
        .unwrap();

        assert_eq!(response.status, 200);

        let storage = ctx.collections.get_collection("repo").unwrap();
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let metadata = storage.get_metadata(&txn).unwrap();
        assert_eq!(metadata.stats.node_count, 2);
        assert_eq!(metadata.stats.edge_count, 1);
    }
}
