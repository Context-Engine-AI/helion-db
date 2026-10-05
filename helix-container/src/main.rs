// jemalloc replaces the default glibc allocator. On ingest-heavy workloads
// (bincode serialize per upsert, HNSW priority queues, LMDB/hyper/bytes
// copies, rayon work-stealing) the default ptmalloc arena design becomes a
// scalability ceiling; jemalloc's per-thread arenas + decay-based purge
// reduce lock contention and fragmentation. Tune via `_RJEM_MALLOC_CONF`
// in the deploy manifest (not plain `MALLOC_CONF` -- tikv-jemallocator
// namespaces its symbols with the `_rjem_` prefix).
#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use helixdb::helix_engine::graph_core::config::Config;
use helixdb::helix_engine::graph_core::graph_core::{HelixGraphEngine, HelixGraphEngineOpts};
use helixdb::helix_engine::storage_core::{
    backend_lsm_reader::LSM_READER_DATABASE_MISSING, collection_manager::CollectionManager,
    reader_warm, replication::ReplicationManager,
};
use helixdb::helix_gateway::{
    api::register::register_api_routes,
    gateway::{GatewayOpts, HelixGateway},
    router::router::{HandlerFn, HandlerSubmission, HelixRouter},
};
use inventory;
use std::{collections::HashMap, sync::Arc};
use tokio::runtime::Builder;
use tracing::{info, warn};

mod queries;

fn env_usize(name: &str, default: usize, min: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value >= min)
        .unwrap_or(default)
}

/// Open the graph engine, tolerating a not-yet-initialized object store.
///
/// On the LSM backend a node — especially a `HELIX_LSM_ROLE=reader` — can start
/// before any writer has created the SlateDB manifest in the bucket. SlateDB then
/// fails to open with "failed to find latest transactional object (e.g.
/// manifest)". Rather than crash-loop on `.unwrap()`, wait with bounded backoff
/// for the writer to initialize the store and retry. Any other error fails fast,
/// and the success-on-first-try path is unchanged (no added latency / no sleep).
///
/// Tunables: `HELIX_LSM_INIT_WAIT_SECS` (total wait, default 120) and
/// `HELIX_LSM_INIT_RETRY_SECS` (retry interval, default 2, min 1).
fn open_graph_engine(path: String, config: Config) -> HelixGraphEngine {
    let wait = std::time::Duration::from_secs(env_usize("HELIX_LSM_INIT_WAIT_SECS", 120, 0) as u64);
    let retry_interval =
        std::time::Duration::from_secs(env_usize("HELIX_LSM_INIT_RETRY_SECS", 2, 1) as u64);
    let deadline = std::time::Instant::now() + wait;
    loop {
        let opts = HelixGraphEngineOpts {
            path: path.clone(),
            config: config.clone(),
        };
        match HelixGraphEngine::new(opts) {
            Ok(engine) => return engine,
            Err(e) => {
                let msg = e.to_string();
                // SlateDB phrases an uninitialized bucket as "failed to find
                // latest transactional object (e.g. manifest)". Only retry that
                // specific not-yet-initialized condition; everything else fails
                // fast (preserving the prior `.unwrap()` semantics).
                let lower = msg.to_ascii_lowercase();
                // Match SlateDB's specific "missing manifest" error string only —
                // a bare "manifest" substring could match unrelated errors and turn
                // an immediate failure into a 120s-delayed panic.
                // A reader replica reports the same condition as SlateDB's typed
                // `DatabaseMissing`, tagged by `LsmReader` with its marker.
                let uninitialized = lower.contains("latest transactional object")
                    || msg.contains(LSM_READER_DATABASE_MISSING);
                if uninitialized && std::time::Instant::now() < deadline {
                    warn!(
                        error = %msg,
                        retry_in_secs = retry_interval.as_secs(),
                        "waiting for writer to initialize object store"
                    );
                    std::thread::sleep(retry_interval);
                    continue;
                }
                panic!("Failed to open graph engine: {msg}");
            }
        }
    }
}

/// Raise the soft open-file limit (`RLIMIT_NOFILE`) to the hard limit at startup.
///
/// SlateDB opens up to ~1000 file handles per collection, and with
/// `HELIX_MAX_OPEN_COLLECTIONS` (~64) the writer can approach the 65536 soft
/// `nofile` cap; S3 socket/DNS FDs then tip it into EMFILE ("Too many open
/// files", errno 24), stalling `/health` until the liveness probe SIGKILLs the
/// pod (exit 137). The hard limit (1048576, 16x headroom) is otherwise unused, so
/// lift the soft limit to it. Best-effort and non-fatal: any failure is logged
/// and startup continues — the server must still come up.
#[cfg(unix)]
fn raise_fd_limit() {
    unsafe {
        let mut lim = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut lim) != 0 {
            eprintln!("getrlimit NOFILE failed; continuing");
            return;
        }
        let old = lim.rlim_cur;
        if lim.rlim_cur < lim.rlim_max {
            lim.rlim_cur = lim.rlim_max;
            if libc::setrlimit(libc::RLIMIT_NOFILE, &lim) != 0 {
                eprintln!("setrlimit NOFILE failed; continuing with soft={}", old);
                return;
            }
            eprintln!("raised RLIMIT_NOFILE soft {} -> {}", old, lim.rlim_cur);
        }
    }
}

#[cfg(not(unix))]
fn raise_fd_limit() {}

fn main() {
    // Lift the soft open-file limit before any storage/server setup so SlateDB's
    // per-collection FD demand can scale (non-fatal; logs via eprintln! since
    // tracing is not yet initialized).
    raise_fd_limit();

    // ── Structured logging ──
    // HELIX_LOG controls verbosity: trace, debug, info (default), warn, error
    // Example: HELIX_LOG=debug helix-container
    let log_filter = std::env::var("HELIX_LOG").unwrap_or_else(|_| "info".to_string());
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_new(&log_filter)
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .with_thread_ids(false)
        .compact()
        .init();

    // Install the Prometheus metrics recorder early so startup-path metrics
    // (collection scan, WAL recovery) are captured. No-op if already installed.
    helixdb::helix_gateway::api::metrics::init();

    // read from config.hx.json
    let config_path = match std::env::var("HELIX_CONFIG_PATH") {
        Ok(path) => std::path::PathBuf::from(path),
        Err(_) => {
            let home = dirs::home_dir().expect("Could not retrieve home directory");
            home.join(".helix/repo/helix-db/helix-container/src/config.hx.json")
        }
    };
    let config = match Config::from_config_file(config_path) {
        Ok(config) => config,
        Err(e) => {
            warn!(error = %e, "Config file not found, using defaults");
            Config::default()
        }
    };

    let path = match std::env::var("HELIX_DATA_DIR") {
        Ok(val) => std::path::PathBuf::from(val).join("user"),
        Err(_) => {
            info!("HELIX_DATA_DIR not set, using default ~/.helix/user");
            let home = dirs::home_dir().expect("Could not retrieve home directory");
            home.join(".helix/user")
        }
    };
    let port = match std::env::var("HELIX_PORT") {
        Ok(val) => match val.parse::<u16>() {
            Ok(port) => port,
            Err(e) => {
                warn!(value = %val, error = %e, "Invalid HELIX_PORT, using default");
                6969
            }
        },
        Err(_) => 6969,
    };
    use helixdb::helix_engine::graph_core::config::DEFAULT_DB_INITIAL_MAP_MB;
    let db_initial_map_mb = config
        .vector_config
        .db_max_size
        .map(|gb| gb * 1024)
        .unwrap_or(DEFAULT_DB_INITIAL_MAP_MB);
    let max_open = std::env::var("HELIX_MAX_OPEN_COLLECTIONS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(256);
    info!(
        path = %path.display(),
        port = port,
        initial_map_mb = db_initial_map_mb,
        max_open_collections = max_open,
        "HelixDB starting"
    );
    let path_str = path.to_str().expect("Could not convert path to string");
    let collections_config = config.clone();
    let graph = Arc::new(open_graph_engine(path_str.to_string(), config));

    // Create collection manager for per-tenant LMDB environments
    let collection_manager = Arc::new(
        CollectionManager::new(path.clone(), collections_config)
            .expect("Failed to create CollectionManager"),
    );
    let replication_manager = Arc::new(
        ReplicationManager::new(
            Arc::clone(&collection_manager),
            collection_manager.config().clone(),
        )
        .expect("Failed to create ReplicationManager"),
    );

    // generates routes from handler proc macro
    let submissions: Vec<_> = inventory::iter::<HandlerSubmission>.into_iter().collect();
    info!(
        helixql_handlers = submissions.len(),
        "Collecting HelixQL routes"
    );

    let routes = HashMap::from_iter(
        submissions
            .into_iter()
            .map(|submission| {
                let handler = &submission.0;
                let func: HandlerFn =
                    Arc::new(move |input, response| (handler.func)(input, response));
                (
                    (
                        "post".to_ascii_uppercase().to_string(),
                        format!("/{}", handler.name.to_string()),
                    ),
                    func,
                )
            })
            .collect::<Vec<((String, String), HandlerFn)>>(),
    );

    // Build router with both HelixQL compiled routes and CE REST API routes
    let mut router = HelixRouter::new(Some(routes));
    register_api_routes(&mut router);
    info!(total_routes = router.routes.len(), "Router ready");

    let pool_size = GatewayOpts::pool_size();
    let default_runtime_workers = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(4);
    let tokio_worker_threads = env_usize("HELIX_TOKIO_WORKER_THREADS", default_runtime_workers, 2);
    let max_blocking_threads = env_usize("HELIX_TOKIO_MAX_BLOCKING_THREADS", 512, 64);
    let runtime = Builder::new_multi_thread()
        .worker_threads(tokio_worker_threads)
        .max_blocking_threads(max_blocking_threads)
        .enable_all()
        .build()
        .expect("Failed to create Tokio runtime");

    // Reader replicas re-open their persisted hot collections before reporting
    // ready on /readyz/warm (no-op for writers). Spawned before the listener
    // binds so the very first readiness probe already sees the warming state.
    {
        let _runtime_guard = runtime.enter();
        reader_warm::spawn_reader_startup_warm(Arc::clone(&collection_manager));
    }

    // create gateway
    let gateway = runtime.block_on(HelixGateway::new_with_router(
        &format!("0.0.0.0:{}", port),
        graph,
        collection_manager,
        replication_manager,
        pool_size,
        router,
    ));

    info!(
        address = format!("0.0.0.0:{}", port),
        workers = pool_size,
        tokio_worker_threads = tokio_worker_threads,
        tokio_max_blocking_threads = max_blocking_threads,
        "HelixDB listening"
    );
    let handle = runtime
        .block_on(gateway.connection_handler.accept_conns())
        .unwrap();
    runtime.block_on(handle).unwrap();
}
