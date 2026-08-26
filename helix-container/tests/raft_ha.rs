use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

static TEST_MUTEX: Mutex<()> = Mutex::new(());

const STARTUP_TIMEOUT: Duration = Duration::from_secs(20);
const ELECTION_TIMEOUT: Duration = Duration::from_secs(20);
const REPLICATION_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Deserialize)]
struct StatusResponse {
    enabled: bool,
    node_id: Option<u64>,
    leader_id: Option<u64>,
    term: u64,
    is_leader: bool,
    commit_index: u64,
    applied_index: u64,
    first_index: u64,
    last_index: u64,
    snapshot_index: u64,
}

#[derive(Debug, Clone)]
struct NodeSpec {
    node_id: u64,
    port: u16,
    data_dir: PathBuf,
    config_path: PathBuf,
}

struct NodeProcess {
    spec: NodeSpec,
    log_path: PathBuf,
    child: Child,
}

impl NodeProcess {
    fn start(spec: NodeSpec) -> Self {
        let log_path = spec.data_dir.join("helix-container.log");
        let child = Command::new(env!("CARGO_BIN_EXE_helix-container"))
            .env("HELIX_CONFIG_PATH", &spec.config_path)
            .env("HELIX_DATA_DIR", &spec.data_dir)
            .env("HELIX_PORT", spec.port.to_string())
            .env("RUST_BACKTRACE", "1")
            .stdout(Stdio::from(
                fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&log_path)
                    .unwrap(),
            ))
            .stderr(Stdio::from(
                fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&log_path)
                    .unwrap(),
            ))
            .spawn()
            .expect("failed to start helix-container");
        Self {
            spec,
            log_path,
            child,
        }
    }

    fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.spec.port)
    }

    fn stop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }

    fn restart(&mut self) {
        self.stop();
        let _ = fs::remove_file(&self.log_path);
        self.child = Command::new(env!("CARGO_BIN_EXE_helix-container"))
            .env("HELIX_CONFIG_PATH", &self.spec.config_path)
            .env("HELIX_DATA_DIR", &self.spec.data_dir)
            .env("HELIX_PORT", self.spec.port.to_string())
            .env("RUST_BACKTRACE", "1")
            .stdout(Stdio::from(
                fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.log_path)
                    .unwrap(),
            ))
            .stderr(Stdio::from(
                fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.log_path)
                    .unwrap(),
            ))
            .spawn()
            .expect("failed to restart helix-container");
    }

    fn recent_logs(&self) -> String {
        fs::read_to_string(&self.log_path)
            .unwrap_or_default()
            .lines()
            .rev()
            .take(20)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Drop for NodeProcess {
    fn drop(&mut self) {
        self.stop();
    }
}

fn http_client() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("reqwest client")
}

fn find_free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind ephemeral port")
        .local_addr()
        .expect("local addr")
        .port()
}

fn write_config(path: &Path, node_id: u64, port: u16, peers: &[(u64, u16)]) {
    let peer_values: Vec<Value> = peers
        .iter()
        .map(|(peer_id, peer_port)| {
            json!({
                "id": peer_id,
                "address": format!("http://127.0.0.1:{}", peer_port),
            })
        })
        .collect();

    let config = json!({
        "vector_config": {
            "m": 16,
            "ef_construction": 128,
            "ef_search": 256,
            "db_max_size": 1
        },
        "graph_config": {
            "secondary_indices": [],
            "snapshot_interval_secs": 3600,
            "snapshot_keep_last": 3,
            "raft": {
                "enabled": true,
                "node_id": node_id,
                "bind_address": format!("http://127.0.0.1:{}", port),
                "snapshot_entries": 8,
                "snapshot_catchup_entries": 2,
                "raft_secret": "test-cluster-secret-42",
                "peers": peer_values
            }
        }
    });

    fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
}

fn build_cluster(temp: &TempDir) -> Vec<NodeProcess> {
    let ports = [find_free_port(), find_free_port(), find_free_port()];
    let peers = vec![(1, ports[0]), (2, ports[1]), (3, ports[2])];

    let mut nodes = Vec::new();
    for (node_id, port) in peers.iter().copied() {
        let node_dir = temp.path().join(format!("node-{}", node_id));
        fs::create_dir_all(&node_dir).unwrap();
        let config_path = node_dir.join("config.hx.json");
        write_config(&config_path, node_id, port, &peers);
        nodes.push(NodeProcess::start(NodeSpec {
            node_id,
            port,
            data_dir: node_dir,
            config_path,
        }));
    }
    nodes
}

fn wait_until<F>(timeout: Duration, mut predicate: F)
where
    F: FnMut() -> bool,
{
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if predicate() {
            return;
        }
        thread::sleep(Duration::from_millis(100));
    }
    panic!("condition not satisfied before timeout");
}

fn wait_for_healthy(node: &NodeProcess) {
    let client = http_client();
    let url = format!("{}/health", node.base_url());
    let deadline = Instant::now() + STARTUP_TIMEOUT;
    while Instant::now() < deadline {
        if client
            .get(&url)
            .send()
            .map(|response| response.status().is_success())
            .unwrap_or(false)
        {
            return;
        }
        thread::sleep(Duration::from_millis(100));
    }
    panic!(
        "node {} did not become healthy in time\nlogs:\n{}",
        node.spec.node_id,
        node.recent_logs()
    );
}

fn get_status(node: &NodeProcess) -> Option<StatusResponse> {
    http_client()
        .get(format!("{}/_raft/status", node.base_url()))
        .send()
        .ok()?
        .json::<StatusResponse>()
        .ok()
}

fn wait_for_leader(nodes: &[NodeProcess]) -> StatusResponse {
    let deadline = Instant::now() + ELECTION_TIMEOUT;
    while Instant::now() < deadline {
        for node in nodes {
            if let Some(status) = get_status(node) {
                if status.enabled && status.is_leader {
                    return status;
                }
            }
        }
        thread::sleep(Duration::from_millis(100));
    }
    panic!("leader election timed out");
}

fn request_json(node: &NodeProcess, method: reqwest::Method, path: &str, body: Value) -> Value {
    let client = http_client();
    let response = client
        .request(method, format!("{}{}", node.base_url(), path))
        .json(&body)
        .send()
        .unwrap_or_else(|err| panic!("request to {}{} failed: {}", node.base_url(), path, err));
    let status = response.status();
    let text = response.text().unwrap();
    assert!(
        status.is_success(),
        "request {}{} failed with {}: {}\nlogs:\n{}",
        node.base_url(),
        path,
        status,
        text,
        node.recent_logs()
    );
    serde_json::from_str(&text).unwrap()
}

fn create_collection(node: &NodeProcess) {
    request_json(
        node,
        reqwest::Method::PUT,
        "/collections/repo",
        json!({
            "vectors": {
                "dense": { "size": 3, "distance": "Cosine" }
            }
        }),
    );
}

fn upsert_point(node: &NodeProcess, point_id: u64, vector: [f32; 3], tag: &str) {
    request_json(
        node,
        reqwest::Method::PUT,
        "/collections/repo/points",
        json!({
            "points": [{
                "id": point_id,
                "vector": { "dense": vector },
                "payload": { "tag": tag }
            }]
        }),
    );
}

fn search_tags(node: &NodeProcess, vector: [f32; 3]) -> Vec<String> {
    let body = request_json(
        node,
        reqwest::Method::POST,
        "/collections/repo/points/search",
        json!({
            "vector": vector,
            "limit": 8,
            "with_payload": true
        }),
    );
    body["result"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|value| value["payload"]["tag"].as_str().map(|tag| tag.to_string()))
        .collect()
}

fn node_by_id(nodes: &[NodeProcess], node_id: u64) -> &NodeProcess {
    nodes
        .iter()
        .find(|node| node.spec.node_id == node_id)
        .expect("node id present")
}

fn node_by_id_mut(nodes: &mut [NodeProcess], node_id: u64) -> &mut NodeProcess {
    nodes
        .iter_mut()
        .find(|node| node.spec.node_id == node_id)
        .expect("node id present")
}

#[test]
fn cluster_fails_over_reads_and_writes() {
    let _guard = TEST_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let temp = TempDir::new().unwrap();
    let mut nodes = build_cluster(&temp);
    for node in &nodes {
        wait_for_healthy(node);
    }

    let leader = wait_for_leader(&nodes);
    let follower_id = nodes
        .iter()
        .map(|node| node.spec.node_id)
        .find(|node_id| *node_id != leader.node_id.unwrap())
        .unwrap();
    let reader_id = nodes
        .iter()
        .map(|node| node.spec.node_id)
        .find(|node_id| *node_id != leader.node_id.unwrap() && *node_id != follower_id)
        .unwrap();

    create_collection(node_by_id(&nodes, follower_id));
    upsert_point(
        node_by_id(&nodes, follower_id),
        1,
        [1.0, 0.0, 0.0],
        "before-failover",
    );
    wait_until(REPLICATION_TIMEOUT, || {
        search_tags(node_by_id(&nodes, reader_id), [1.0, 0.0, 0.0])
            .iter()
            .any(|tag| tag == "before-failover")
    });

    node_by_id_mut(&mut nodes, leader.node_id.unwrap()).stop();

    let new_leader = wait_for_leader(&nodes);
    assert_ne!(new_leader.node_id, leader.node_id);

    let writer_id = nodes
        .iter()
        .filter_map(|node| {
            get_status(node)
                .filter(|_| Some(node.spec.node_id) != new_leader.node_id)
                .map(|_| node.spec.node_id)
        })
        .next()
        .unwrap();
    upsert_point(
        node_by_id(&nodes, writer_id),
        2,
        [0.0, 1.0, 0.0],
        "after-failover",
    );

    wait_until(REPLICATION_TIMEOUT, || {
        search_tags(node_by_id(&nodes, writer_id), [0.0, 1.0, 0.0])
            .iter()
            .any(|tag| tag == "after-failover")
    });
}

#[test]
fn restarted_follower_catches_up_after_log_compaction() {
    let _guard = TEST_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let temp = TempDir::new().unwrap();
    let mut nodes = build_cluster(&temp);
    for node in &nodes {
        wait_for_healthy(node);
    }

    let leader = wait_for_leader(&nodes);
    create_collection(node_by_id(&nodes, leader.node_id.unwrap()));
    upsert_point(
        node_by_id(&nodes, leader.node_id.unwrap()),
        1,
        [1.0, 0.0, 0.0],
        "seed",
    );

    let stopped_follower_id = nodes
        .iter()
        .map(|node| node.spec.node_id)
        .find(|node_id| *node_id != leader.node_id.unwrap())
        .unwrap();
    let stopped_last_index = get_status(node_by_id(&nodes, stopped_follower_id))
        .unwrap()
        .last_index;
    node_by_id_mut(&mut nodes, stopped_follower_id).stop();

    for point_id in 2..=24 {
        upsert_point(
            node_by_id(&nodes, leader.node_id.unwrap()),
            point_id,
            [1.0, 0.0, 0.0],
            &format!("p{}", point_id),
        );
    }

    wait_until(REPLICATION_TIMEOUT, || {
        get_status(node_by_id(&nodes, leader.node_id.unwrap()))
            .map(|status| status.first_index > stopped_last_index)
            .unwrap_or(false)
    });

    node_by_id_mut(&mut nodes, stopped_follower_id).restart();
    wait_for_healthy(node_by_id(&nodes, stopped_follower_id));

    let target_last_index = wait_for_leader(&nodes).last_index;
    wait_until(REPLICATION_TIMEOUT, || {
        get_status(node_by_id(&nodes, stopped_follower_id))
            .map(|status| status.applied_index >= target_last_index && status.snapshot_index > 0)
            .unwrap_or(false)
    });

    upsert_point(
        node_by_id(&nodes, stopped_follower_id),
        25,
        [1.0, 0.0, 0.0],
        "post-restart",
    );
    wait_until(REPLICATION_TIMEOUT, || {
        get_status(node_by_id(&nodes, stopped_follower_id))
            .map(|status| status.last_index >= target_last_index + 1)
            .unwrap_or(false)
    });
}
