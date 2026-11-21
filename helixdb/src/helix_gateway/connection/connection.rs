use crate::helix_engine::graph_core::graph_core::HelixGraphEngine;
use crate::helix_engine::storage_core::{
    backend_lsm::allow_lsm_blocking, collection_manager::CollectionManager,
    replication::ReplicationManager,
};
use crate::helix_engine::types::GraphError;
use crate::helix_gateway::{
    async_gateway::{self, AsyncGatewayState, WriteSubmitter},
    router::router::HelixRouter,
    thread_pool::thread_pool::ThreadPool,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tokio::{
    net::TcpListener,
    task::JoinHandle,
    time::{Duration, MissedTickBehavior},
};
use tracing::{error, info, warn};

/// In-flight requests get this long after the shutdown signal. Past it the
/// storage is closed under them: they end with a closed connection or a
/// "db is closed" error, as they did at the kill.
const SHUTDOWN_REQUEST_CAP: Duration = Duration::from_secs(8);

/// `HELIX_SHUTDOWN_CLOSE_MS`: time from the shutdown signal to the end of the
/// close step, inside the pod's 30 s grace period. In it in-flight requests
/// get [`SHUTDOWN_REQUEST_CAP`], then every open LSM writer is closed
/// (`CollectionManager::close_lsm_writers`). Default 25000. 0 waits for every
/// request and closes nothing, as before.
///
/// A process whose root is not an LSM writer (a reader replica, LMDB) has no
/// writer to close, so it gets no budget and no cap: its requests in flight,
/// a reader's cold opens among them, run to their end.
fn shutdown_close_budget(lsm_writer: bool) -> Option<Duration> {
    close_budget_from(
        std::env::var("HELIX_SHUTDOWN_CLOSE_MS").ok().as_deref(),
        lsm_writer,
    )
}

fn close_budget_from(setting: Option<&str>, lsm_writer: bool) -> Option<Duration> {
    let ms = setting
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(25_000);
    (ms > 0 && lsm_writer).then(|| Duration::from_millis(ms))
}

/// The last storage step of a shutdown: close every open LSM writer by
/// `deadline`, off the runtime's workers.
async fn close_lsm_writers_for_shutdown(
    collections: Arc<CollectionManager>,
    graph: Arc<HelixGraphEngine>,
    deadline: std::time::Instant,
) {
    if let Err(err) = tokio::task::spawn_blocking(move || {
        allow_lsm_blocking(|| collections.close_lsm_writers(Some(&graph.storage), deadline))
    })
    .await
    {
        error!(error = %err, "LSM writer close task failed");
    }
}

pub struct ConnectionHandler {
    pub address: String,
    pub graph: Arc<HelixGraphEngine>,
    pub collections: Arc<CollectionManager>,
    pub replication: Arc<ReplicationManager>,
    pub router: Arc<HelixRouter>,
    pub thread_pool: ThreadPool,
}

impl ConnectionHandler {
    pub fn new(
        address: &str,
        graph: Arc<HelixGraphEngine>,
        collections: Arc<CollectionManager>,
        replication: Arc<ReplicationManager>,
        size: usize,
        router: HelixRouter,
    ) -> Result<Self, GraphError> {
        let router = Arc::new(router);
        Ok(Self {
            address: address.to_string(),
            graph: Arc::clone(&graph),
            collections: Arc::clone(&collections),
            replication: Arc::clone(&replication),
            router: Arc::clone(&router),
            thread_pool: ThreadPool::new(size, graph, collections, replication, router)?,
        })
    }

    pub async fn accept_conns(&self) -> Result<JoinHandle<()>, GraphError> {
        if !std::env::var("HELIX_GATEWAY_IMPL")
            .map(|value| value.eq_ignore_ascii_case("legacy"))
            .unwrap_or(false)
        {
            return self.accept_conns_axum().await;
        }

        let listener = TcpListener::bind(&self.address).await.map_err(|e| {
            error!(address = %self.address, error = %e, "Failed to bind");
            GraphError::GraphConnectionError("Failed to bind to address".to_string(), e)
        })?;

        let collections = Arc::clone(&self.collections);
        let graph = Arc::clone(&self.graph);
        let thread_pool_sender = self.thread_pool.sender.clone();
        let snapshot_interval_secs = collections.snapshot_interval_secs();
        let snapshot_on_shutdown = collections.config().snapshot_on_shutdown();

        let handle = tokio::spawn(async move {
            // Listen for both SIGINT (ctrl_c) and SIGTERM (k8s pod eviction).
            let mut sigterm = {
                #[cfg(unix)]
                {
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("Failed to register SIGTERM handler")
                }
            };
            let mut snapshot_interval =
                tokio::time::interval(Duration::from_secs(snapshot_interval_secs.max(1)));
            snapshot_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
            snapshot_interval.tick().await;
            let snapshot_in_progress = Arc::new(AtomicBool::new(false));

            loop {
                tokio::select! {
                    accepted = listener.accept() => {
                        match accepted {
                            Ok((stream, _addr)) => {
                                if let Err(e) = stream.set_nodelay(true) {
                                    warn!(error = %e, "Failed to set TCP_NODELAY");
                                }

                                // Non-blocking send: if the work queue is full, reject
                                // with 503 instead of stalling the accept loop.
                                let queue_len = thread_pool_sender.len();
                                let queue_cap = thread_pool_sender.capacity().unwrap_or(0);
                                metrics::gauge!("helix_thread_pool_queue_depth").set(queue_len as f64);
                                if queue_cap > 0 {
                                    metrics::gauge!("helix_thread_pool_queue_capacity")
                                        .set(queue_cap as f64);
                                }
                                match thread_pool_sender.try_send(stream) {
                                    Ok(_) => (),
                                    Err(flume::TrySendError::Full(mut stream)) => {
                                        metrics::counter!(
                                            "helix_thread_pool_queue_rejected_total"
                                        )
                                        .increment(1);
                                        warn!(
                                            queue_len,
                                            queue_cap,
                                            "Thread pool queue full; returning 503"
                                        );
                                        let body = b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\nContent-Length: 16\r\n\r\nServer too busy\n";
                                        let _ = tokio::io::AsyncWriteExt::write_all(&mut stream, body).await;
                                    }
                                    Err(flume::TrySendError::Disconnected(_)) => {
                                        error!("Thread pool disconnected");
                                    }
                                }
                            }
                            Err(e) => {
                                error!(error = %e, "Error accepting connection");
                            }
                        }
                        continue;
                    }
                    _ = tokio::signal::ctrl_c() => {
                        info!("Received SIGINT, shutting down...");
                    }
                    _ = sigterm.recv() => {
                        info!("Received SIGTERM (pod eviction), shutting down...");
                    }
                    _ = snapshot_interval.tick(), if snapshot_interval_secs > 0 => {
                        if snapshot_in_progress
                            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                            .is_ok()
                        {
                            let collections = Arc::clone(&collections);
                            let snapshot_in_progress = Arc::clone(&snapshot_in_progress);
                            tokio::task::spawn_blocking(move || {
                                allow_lsm_blocking(|| {
                                    if let Err(err) = collections.snapshot_loaded_collections() {
                                        error!(error = %err, "Periodic snapshot failed");
                                    }
                                    snapshot_in_progress.store(false, Ordering::Release);
                                });
                            });
                        }
                        continue;
                    }
                }

                let signal_at = std::time::Instant::now();
                info!("Draining in-flight requests (5s grace period)...");
                drop(thread_pool_sender);
                tokio::time::sleep(Duration::from_secs(5)).await;

                while snapshot_in_progress.load(Ordering::Acquire) {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }

                // Flush WAL to ensure all pending writes are durable.
                let collections_for_wal = Arc::clone(&collections);
                match tokio::task::spawn_blocking(move || {
                    allow_lsm_blocking(|| collections_for_wal.flush_all_wals())
                })
                .await
                {
                    Ok(Ok(count)) => {
                        if count > 0 {
                            info!(collections = count, "WAL flush complete");
                        }
                    }
                    Ok(Err(err)) => error!(error = %err, "WAL flush failed"),
                    Err(err) => error!(error = %err, "WAL flush task failed"),
                }

                if snapshot_on_shutdown {
                    let collections_for_snapshot = Arc::clone(&collections);
                    match tokio::task::spawn_blocking(move || {
                        allow_lsm_blocking(|| {
                            collections_for_snapshot.snapshot_loaded_collections()
                        })
                    })
                    .await
                    {
                        Ok(Ok(snapshots)) => {
                            if !snapshots.is_empty() {
                                info!(count = snapshots.len(), "Shutdown snapshots created");
                            }
                        }
                        Ok(Err(err)) => error!(error = %err, "Shutdown snapshot failed"),
                        Err(err) => error!(error = %err, "Shutdown snapshot task failed"),
                    }
                }

                if let Some(budget) = shutdown_close_budget(graph.storage.backend.is_lsm_writer()) {
                    close_lsm_writers_for_shutdown(
                        Arc::clone(&collections),
                        Arc::clone(&graph),
                        signal_at + budget,
                    )
                    .await;
                }

                info!("Shutdown complete");
                break;
            }
        });

        Ok(handle)
    }

    async fn accept_conns_axum(&self) -> Result<JoinHandle<()>, GraphError> {
        let listener = TcpListener::bind(&self.address).await.map_err(|e| {
            error!(address = %self.address, error = %e, "Failed to bind");
            GraphError::GraphConnectionError("Failed to bind to address".to_string(), e)
        })?;

        let state = AsyncGatewayState {
            graph: Arc::clone(&self.graph),
            collections: Arc::clone(&self.collections),
            replication: Arc::clone(&self.replication),
            router: Arc::clone(&self.router),
            write_submitter: WriteSubmitter::new(),
        };
        // Background liveness loop: re-evaluates tripped segment breakers even
        // when clients back off on 503 so collections keep draining and the
        // breaker can close without further upsert traffic.
        async_gateway::spawn_segment_breaker_reeval_task(state.clone());
        let collections = Arc::clone(&self.collections);
        let graph = Arc::clone(&self.graph);
        let snapshot_interval_secs = collections.snapshot_interval_secs();
        let snapshot_on_shutdown = collections.config().snapshot_on_shutdown();

        let handle = tokio::spawn(async move {
            let snapshot_stop = Arc::new(AtomicBool::new(false));
            let snapshot_in_progress = Arc::new(AtomicBool::new(false));
            let snapshot_stop_task = Arc::clone(&snapshot_stop);
            let snapshot_in_progress_task = Arc::clone(&snapshot_in_progress);
            let collections_for_periodic = Arc::clone(&collections);
            if snapshot_interval_secs > 0 {
                tokio::spawn(async move {
                    let mut snapshot_interval =
                        tokio::time::interval(Duration::from_secs(snapshot_interval_secs));
                    snapshot_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
                    snapshot_interval.tick().await;
                    while !snapshot_stop_task.load(Ordering::Acquire) {
                        snapshot_interval.tick().await;
                        if snapshot_in_progress_task
                            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                            .is_ok()
                        {
                            let collections = Arc::clone(&collections_for_periodic);
                            let snapshot_in_progress = Arc::clone(&snapshot_in_progress_task);
                            tokio::task::spawn_blocking(move || {
                                allow_lsm_blocking(|| {
                                    if let Err(err) = collections.snapshot_loaded_collections() {
                                        error!(error = %err, "Periodic snapshot failed");
                                    }
                                    snapshot_in_progress.store(false, Ordering::Release);
                                });
                            });
                        }
                    }
                });
            }

            let close_budget = shutdown_close_budget(graph.storage.backend.is_lsm_writer());
            let signal_at = Arc::new(std::sync::OnceLock::new());
            let signalled = Arc::new(tokio::sync::Notify::new());
            let (signal_at_set, signal_seen) = (Arc::clone(&signal_at), Arc::clone(&signalled));
            let shutdown = async move {
                #[cfg(unix)]
                let mut sigterm =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("Failed to register SIGTERM handler");
                #[cfg(unix)]
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => info!("Received SIGINT, shutting down..."),
                    _ = sigterm.recv() => info!("Received SIGTERM (pod eviction), shutting down..."),
                }
                #[cfg(not(unix))]
                {
                    let _ = tokio::signal::ctrl_c().await;
                    info!("Received SIGINT, shutting down...");
                }
                let _ = signal_at_set.set(std::time::Instant::now());
                signal_seen.notify_one();
            };

            // Without the cap the graceful shutdown waits for every request,
            // and the pod's kill comes before any writer is closed.
            let cap = close_budget.map(|budget| budget.min(SHUTDOWN_REQUEST_CAP));
            let request_cap = async {
                signalled.notified().await;
                match cap {
                    Some(cap) => tokio::time::sleep(cap).await,
                    None => std::future::pending::<()>().await,
                }
            };
            tokio::select! {
                served = async_gateway::serve(listener, state, shutdown) => {
                    if let Err(err) = served {
                        error!(error = %err, "Axum gateway failed");
                    }
                }
                _ = request_cap => {
                    warn!(
                        cap_ms = cap.unwrap_or_default().as_millis() as u64,
                        "in-flight requests outlived the shutdown cap; closing storage under them"
                    );
                }
            }
            let signal_at = signal_at
                .get()
                .copied()
                .unwrap_or_else(std::time::Instant::now);
            snapshot_stop.store(true, Ordering::Release);

            while snapshot_in_progress.load(Ordering::Acquire) {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }

            let collections_for_wal = Arc::clone(&collections);
            match tokio::task::spawn_blocking(move || {
                allow_lsm_blocking(|| collections_for_wal.flush_all_wals())
            })
            .await
            {
                Ok(Ok(count)) => {
                    if count > 0 {
                        info!(collections = count, "WAL flush complete");
                    }
                }
                Ok(Err(err)) => error!(error = %err, "WAL flush failed"),
                Err(err) => error!(error = %err, "WAL flush task failed"),
            }

            if snapshot_on_shutdown {
                let collections_for_snapshot = Arc::clone(&collections);
                match tokio::task::spawn_blocking(move || {
                    allow_lsm_blocking(|| collections_for_snapshot.snapshot_loaded_collections())
                })
                .await
                {
                    Ok(Ok(snapshots)) => {
                        if !snapshots.is_empty() {
                            info!(count = snapshots.len(), "Shutdown snapshots created");
                        }
                    }
                    Ok(Err(err)) => error!(error = %err, "Shutdown snapshot failed"),
                    Err(err) => error!(error = %err, "Shutdown snapshot task failed"),
                }
            }

            if let Some(budget) = close_budget {
                close_lsm_writers_for_shutdown(collections, graph, signal_at + budget).await;
            }

            info!("Shutdown complete");
        });

        Ok(handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_lsm_writer_gets_the_shutdown_budget_and_any_other_process_none() {
        let seconds = |setting, writer| close_budget_from(setting, writer).map(|d| d.as_secs());
        // An LSM writer: 25 s unless set, and 0 is the shutdown before the close.
        assert_eq!(seconds(None, true), Some(25));
        assert_eq!(seconds(Some("12000"), true), Some(12));
        assert_eq!(seconds(Some("0"), true), None);
        assert_eq!(seconds(Some("soon"), true), Some(25));
        // A reader replica or an LMDB process has no writer to close: no
        // budget, so no cap on its requests in flight either.
        assert_eq!(seconds(None, false), None);
        assert_eq!(seconds(Some("12000"), false), None);
    }
}
