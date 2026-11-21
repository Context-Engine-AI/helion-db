//! Warm-before-ready for LSM reader replicas (`HELIX_LSM_ROLE=reader`).
//!
//! The SlateDB object-store cache under `HELIX_LSM_CACHE_DIR` survives a pod
//! restart, but the in-memory collection open (manifest, filters, index blocks,
//! vector caches) does not, so the first request per collection after a reader
//! restart pays a full cold open. Readers therefore:
//!
//! 1. periodically persist the names of their recently used collections
//!    (most-recent first, capped) to `<cache dir>/.helix_hot_collections.json`;
//! 2. on startup, re-open those collections through the normal
//!    [`CollectionManager::get_collection`] path with bounded concurrency and a
//!    total time budget, then reopen each warmed `DbReader` against the current
//!    manifest so Ready means caught up with the writer (a write landing while
//!    an open is in flight is skipped by the feed poller as not-yet-resident); and
//! 3. report not-ready on `/readyz/warm` (and `/ready`, `/readyz`) until that
//!    warm and catch-up finish or the budget is exhausted.
//!
//! Each reader only learns what it served, so the startup warm also unions in
//! Ready peers' lists, fetched from their in-memory snapshot at
//! `GET /readyz/warm/hot_collections` (own list first, then peers round-robin).
//! Peers come from `HELIX_READER_WARM_PEER_URLS`, else from the StatefulSet's
//! headless service (derived from the kubelet `/etc/hosts` FQDN, so no manifest
//! env is needed). Peer-learned names are not persisted unless actually loaded.
//!
//! Liveness (`/health`) is never gated. Writers and readers without a cache dir
//! never enter the warming state, so their readiness is unchanged.
//!
//! Tunables:
//! - `HELIX_READER_WARM_ENABLED` (default on): startup warm + readiness gate.
//! - `HELIX_READER_WARM_BUDGET_SECS` (default 120): total warm budget.
//! - `HELIX_READER_WARM_CONCURRENCY` (default 2, max 8): parallel opens.
//! - `HELIX_READER_HOT_LIST_MAX` (default 64): persisted names; the warm set is
//!   further capped at half of `HELIX_MAX_OPEN_COLLECTIONS` (min 1) so live
//!   traffic has LRU headroom and does not immediately evict warmed collections.
//! - `HELIX_READER_HOT_LIST_INTERVAL_SECS` (default 60, 0 disables persisting).
//! - `HELIX_READER_HOT_LIST_MAX_AGE_SECS` (default 86400, 0 disables): a hot-list
//!   older than this is ignored (nothing warmed) rather than reopening stale names.
//!   Applied per source: own file and each peer's `written_at_ms`.
//! - `HELIX_READER_WARM_PEER_TIMEOUT_MS` (default 2000): total peer discovery +
//!   fetch time, also clamped to the remaining warm budget.
//! - `HELIX_READER_WARM_PEER_URLS` (unset → StatefulSet discovery; `off`
//!   disables): comma-separated peer base URLs; entries naming this pod are dropped.
//!
//! Operational note: the gate is per pod. If BOTH readers restart at once (node
//! drain, manual delete — not the one-at-a-time StatefulSet rollout), the reader
//! Service has no ready endpoints for up to the warm budget, and the client's reader
//! client falls back to the writer (`reader_fallback_retryable`) meanwhile.

use super::{
    backend_lsm::{
        allow_lsm_blocking, cache_root_from_env, reader_active_poll_interval_from_env,
        reader_idle_poll_interval_from_env,
    },
    collection_manager::CollectionManager,
};
use crate::helix_engine::types::GraphError;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tracing::{debug, info, warn};

pub const HOT_LIST_FILE_NAME: &str = ".helix_hot_collections.json";
const HOT_LIST_VERSION: u32 = 1;
const DEFAULT_HOT_LIST_MAX: usize = 64;
const DEFAULT_HOT_LIST_INTERVAL_SECS: u64 = 60;
const DEFAULT_HOT_LIST_MAX_AGE_SECS: u64 = 86_400;
const DEFAULT_WARM_BUDGET_SECS: u64 = 120;
const DEFAULT_WARM_CONCURRENCY: usize = 2;
const MAX_WARM_CONCURRENCY: usize = 8;
const MAX_COLLECTION_NAME_BYTES: usize = 255;

/// True while a reader's startup warm is in progress. Defaults to false so
/// writers, tests, and ungated readers always report ready.
static STARTUP_WARM_PENDING: AtomicBool = AtomicBool::new(false);

/// Serializes every test that flips or asserts [`STARTUP_WARM_PENDING`]: it is
/// process-global and lib tests run in parallel threads of one binary.
#[cfg(test)]
pub(crate) static WARM_GATE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(crate) fn warm_gate_test_lock() -> std::sync::MutexGuard<'static, ()> {
    WARM_GATE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Readiness gate for `/ready`, `/readyz`, and `/readyz/warm`.
pub fn startup_warm_ready() -> bool {
    !STARTUP_WARM_PENDING.load(Ordering::Acquire)
}

pub(crate) fn set_startup_warm_pending(pending: bool) {
    STARTUP_WARM_PENDING.store(pending, Ordering::Release);
    metrics::gauge!("helix_reader_warm_ready").set(if pending { 0.0 } else { 1.0 });
}

/// Clears the readiness gate on drop so a panic or early return in the warm
/// task can never leave the pod permanently not-ready.
struct ClearWarmPendingOnDrop;

impl Drop for ClearWarmPendingOnDrop {
    fn drop(&mut self) {
        set_startup_warm_pending(false);
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn warm_enabled() -> bool {
    std::env::var("HELIX_READER_WARM_ENABLED")
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true)
}

fn hot_list_max() -> usize {
    env_u64("HELIX_READER_HOT_LIST_MAX", DEFAULT_HOT_LIST_MAX as u64).max(1) as usize
}

fn warm_concurrency() -> usize {
    (env_u64(
        "HELIX_READER_WARM_CONCURRENCY",
        DEFAULT_WARM_CONCURRENCY as u64,
    ) as usize)
        .clamp(1, MAX_WARM_CONCURRENCY)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct HotListFile {
    version: u32,
    written_at_ms: u64,
    collections: Vec<String>,
}

/// Collection names come back from disk: reject anything that could escape
/// the collections dir or is not a plausible single path segment.
fn plausible_collection_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_COLLECTION_NAME_BYTES
        && name != "."
        && name != ".."
        && !name.contains(['/', '\\', '\0'])
}

/// Dedupe (first occurrence wins), drop implausible names, and cap.
fn normalize_names(names: impl IntoIterator<Item = String>, cap: usize) -> Vec<String> {
    let mut seen = HashSet::new();
    names
        .into_iter()
        .filter(|name| plausible_collection_name(name))
        .filter(|name| seen.insert(name.clone()))
        .take(cap)
        .collect()
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis().min(u64::MAX as u128) as u64)
        .unwrap_or(0)
}

/// Validate a serialized hot-list (own file or a peer's response): version,
/// staleness against `written_at_ms` (future timestamps from clock skew count as
/// fresh), and per-name validation + dedupe + cap. `Err` is the metric outcome.
fn parse_hot_list(
    bytes: &[u8],
    cap: usize,
    max_age: Option<Duration>,
) -> Result<HotListFile, (&'static str, String)> {
    let file =
        serde_json::from_slice::<HotListFile>(bytes).map_err(|err| ("corrupt", err.to_string()))?;
    if file.version != HOT_LIST_VERSION {
        return Err(("corrupt", format!("unsupported version {}", file.version)));
    }
    let age_ms = now_millis().saturating_sub(file.written_at_ms);
    if max_age.is_some_and(|max_age| u128::from(age_ms) > max_age.as_millis()) {
        return Err((
            "stale",
            format!(
                "{}s old exceeds HELIX_READER_HOT_LIST_MAX_AGE_SECS",
                age_ms / 1000
            ),
        ));
    }
    Ok(HotListFile {
        version: file.version,
        written_at_ms: file.written_at_ms,
        collections: normalize_names(file.collections, cap),
    })
}

/// Load the persisted hot-list. Missing, unreadable, corrupt, or (with
/// `max_age`) stale files yield `None` (logged + counted) — never an error.
fn load_hot_list(path: &Path, cap: usize, max_age: Option<Duration>) -> Option<HotListFile> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            metrics::counter!("helix_reader_hot_list_reads_total", "outcome" => "missing")
                .increment(1);
            return None;
        }
        Err(err) => {
            metrics::counter!("helix_reader_hot_list_reads_total", "outcome" => "error")
                .increment(1);
            warn!(path = %path.display(), error = %err, "failed to read reader hot-list; skipping warm");
            return None;
        }
    };
    match parse_hot_list(&bytes, cap, max_age) {
        Ok(file) => {
            metrics::counter!("helix_reader_hot_list_reads_total", "outcome" => "ok").increment(1);
            Some(file)
        }
        Err((outcome, reason)) => {
            metrics::counter!("helix_reader_hot_list_reads_total", "outcome" => outcome)
                .increment(1);
            warn!(path = %path.display(), outcome, reason = %reason, "ignoring reader hot-list");
            None
        }
    }
}

/// Persisted hot-list names, or empty when missing/corrupt/stale.
pub fn read_hot_list(path: &Path, cap: usize, max_age: Option<Duration>) -> Vec<String> {
    load_hot_list(path, cap, max_age)
        .map(|file| file.collections)
        .unwrap_or_default()
}

/// Atomically replace the hot-list (write tmp, fsync, rename).
pub fn write_hot_list(path: &Path, names: &[String], cap: usize) -> std::io::Result<()> {
    let file = HotListFile {
        version: HOT_LIST_VERSION,
        written_at_ms: now_millis(),
        collections: normalize_names(names.iter().cloned(), cap),
    };
    let bytes = serde_json::to_vec(&file).map_err(std::io::Error::other)?;
    let mut tmp_name = path.as_os_str().to_owned();
    tmp_name.push(".tmp");
    let tmp_path = PathBuf::from(tmp_name);
    let result = (|| {
        let mut tmp = fs::File::create(&tmp_path)?;
        tmp.write_all(&bytes)?;
        tmp.sync_all()?;
        fs::rename(&tmp_path, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp_path);
    }
    result
}

/// The hot-list this reader serves to peers on
/// `GET /readyz/warm/hot_collections`. Only a reader's warm task publishes it,
/// so writers (and readers without a cache dir) answer 404.
static SERVED_HOT_LIST: std::sync::RwLock<Option<HotListFile>> = std::sync::RwLock::new(None);

#[cfg(test)]
pub(crate) fn publish_hot_list_for_test(collections: Option<Vec<String>>) {
    publish_hot_list(collections.map(|collections| HotListFile {
        version: HOT_LIST_VERSION,
        written_at_ms: now_millis(),
        collections,
    }));
}

pub const HOT_COLLECTIONS_PATH: &str = "/readyz/warm/hot_collections";

fn publish_hot_list(file: Option<HotListFile>) {
    let mut served = SERVED_HOT_LIST
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *served = file;
}

/// JSON body for `GET /readyz/warm/hot_collections` (same format as the
/// on-disk file), or `None` → 404 when this process is not a warming reader.
/// Reads an in-memory snapshot only; never touches storage or the
/// collection map.
pub fn hot_collections_response_body() -> Option<Vec<u8>> {
    let served = SERVED_HOT_LIST
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    served
        .as_ref()
        .and_then(|file| serde_json::to_vec(file).ok())
}

const DEFAULT_PEER_TIMEOUT_MS: u64 = 2_000;
const DEFAULT_HELIX_PORT: u16 = 6969;

fn helix_port() -> u16 {
    std::env::var("HELIX_PORT")
        .ok()
        .and_then(|value| value.trim().parse::<u16>().ok())
        .unwrap_or(DEFAULT_HELIX_PORT)
}

/// This pod's IP and StatefulSet governing-service domain, from the kubelet-
/// managed `/etc/hosts` line `<pod ip> <hostname>.<service>.<ns>.svc... <hostname>`.
/// Kubelet writes that FQDN for every pod with `hostname` + `subdomain` set,
/// which a StatefulSet always does (subdomain = `serviceName`), so this follows
/// the real headless service without hardcoding its name.
fn statefulset_identity(etc_hosts: &str, hostname: &str) -> Option<(String, String)> {
    let prefix = format!("{hostname}.");
    etc_hosts.lines().find_map(|line| {
        let line = line.split('#').next().unwrap_or("");
        let mut tokens = line.split_whitespace();
        let ip = tokens.next()?;
        let domain = tokens.find_map(|token| token.strip_prefix(prefix.as_str()))?;
        (!domain.is_empty()).then(|| (ip.to_string(), domain.trim_end_matches('.').to_string()))
    })
}

/// Host part of an `http://host[:port][/...]` base URL.
fn url_host(url: &str) -> &str {
    let rest = url.split_once("://").map(|(_, rest)| rest).unwrap_or(url);
    let authority = rest.split('/').next().unwrap_or(rest);
    if let Some(bracketed) = authority.strip_prefix('[') {
        return bracketed.split(']').next().unwrap_or(bracketed);
    }
    authority.split(':').next().unwrap_or(authority)
}

/// Explicit peers from `HELIX_READER_WARM_PEER_URLS` (comma-separated base
/// URLs), minus any that name this pod. `Some(empty)` disables peer fetch
/// (`off`/`none`); `None` means unset → StatefulSet discovery.
fn explicit_peer_urls(raw: Option<&str>, self_names: &[&str]) -> Option<Vec<String>> {
    let raw = raw?.trim();
    if raw.is_empty() {
        return None;
    }
    if matches!(raw.to_ascii_lowercase().as_str(), "off" | "none" | "0") {
        return Some(Vec::new());
    }
    Some(
        raw.split(',')
            .map(|url| url.trim().trim_end_matches('/').to_string())
            .filter(|url| !url.is_empty())
            .filter(|url| {
                let host = url_host(url);
                !self_names.iter().any(|name| {
                    !name.is_empty() && (host == *name || host.split('.').next() == Some(name))
                })
            })
            .collect(),
    )
}

/// Peer base URLs from StatefulSet discovery: every address the governing
/// headless service resolves to, except this pod. Headless DNS only publishes
/// Ready pods, so a peer that is itself still warming (or down) is not asked.
/// `explicit` (the raw `HELIX_READER_WARM_PEER_URLS`) wins when set.
async fn discover_peer_urls(explicit: Option<&str>) -> Result<Vec<String>, String> {
    let hostname = std::env::var("HOSTNAME").unwrap_or_default();
    let etc_hosts = tokio::fs::read_to_string("/etc/hosts")
        .await
        .unwrap_or_default();
    let identity = statefulset_identity(&etc_hosts, hostname.trim());
    let self_ip = identity.as_ref().map(|(ip, _)| ip.as_str()).unwrap_or("");
    if let Some(urls) = explicit_peer_urls(explicit, &[hostname.trim(), self_ip]) {
        return Ok(urls);
    }
    let Some((self_ip, service_domain)) = identity else {
        return Err("no StatefulSet FQDN for HOSTNAME in /etc/hosts".to_string());
    };
    let resolved = tokio::net::lookup_host((service_domain.clone(), helix_port())).await;
    let addrs = resolved.map_err(|err| format!("resolve {service_domain}: {err}"))?;
    let mut urls: Vec<String> = addrs
        .filter(|addr| addr.ip().to_string() != self_ip)
        .map(|addr| format!("http://{addr}"))
        .collect();
    urls.sort();
    urls.dedup();
    Ok(urls)
}

/// Upper bound on a peer hot-list response (a full 64-name list is a few KiB).
const MAX_PEER_RESPONSE_BYTES: usize = 256 * 1024;

/// Peer client: no proxy (in-cluster pod IPs), short connect timeout. `None`
/// if the client cannot be built — peers are then skipped, never a panic.
static PEER_HTTP_CLIENT: std::sync::LazyLock<Option<reqwest::Client>> =
    std::sync::LazyLock::new(|| {
        reqwest::Client::builder()
            .no_proxy()
            .connect_timeout(Duration::from_secs(2))
            .build()
            .map_err(|err| warn!(error = %err, "reader warm: peer HTTP client unavailable"))
            .ok()
    });

/// GET a peer's hot-list body, reading at most [`MAX_PEER_RESPONSE_BYTES`].
async fn fetch_peer_body(
    client: &reqwest::Client,
    url: &str,
) -> Result<Vec<u8>, (&'static str, String)> {
    let peer_error = |err: reqwest::Error| ("peer_error", err.to_string());
    let mut response = client.get(url).send().await.map_err(peer_error)?;
    let status = response.status();
    if !status.is_success() {
        return Err(("peer_error", format!("HTTP {status}")));
    }
    if response
        .content_length()
        .is_some_and(|len| len > MAX_PEER_RESPONSE_BYTES as u64)
    {
        return Err(("peer_error", "response exceeds size limit".to_string()));
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(peer_error)? {
        if body.len() + chunk.len() > MAX_PEER_RESPONSE_BYTES {
            return Err(("peer_error", "response exceeds size limit".to_string()));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// Fetch one peer's hot-list. Every failure is logged + counted and yields
/// `None`; it can never fail the warm.
async fn fetch_peer_hot_list(
    client: &reqwest::Client,
    base_url: &str,
    timeout: Duration,
    cap: usize,
    max_age: Option<Duration>,
) -> Option<Vec<String>> {
    let url = format!("{}{}", base_url.trim_end_matches('/'), HOT_COLLECTIONS_PATH);
    let result = match tokio::time::timeout(timeout, fetch_peer_body(client, &url)).await {
        Err(_) => Err(("peer_timeout", "timed out".to_string())),
        Ok(Err(err)) => Err(err),
        Ok(Ok(body)) => match parse_hot_list(&body, cap, max_age) {
            Ok(file) => Ok(file.collections),
            Err(("stale", reason)) => Err(("peer_stale", reason)),
            Err((_, reason)) => Err(("peer_error", reason)),
        },
    };
    let outcome = match &result {
        Ok(_) => "peer_ok",
        Err((outcome, _)) => outcome,
    };
    metrics::counter!("helix_reader_warm_peer_fetches_total", "outcome" => outcome).increment(1);
    match result {
        Ok(names) => {
            debug!(peer = %base_url, collections = names.len(), "reader warm: fetched peer hot-list");
            Some(names)
        }
        Err((outcome, reason)) => {
            warn!(peer = %base_url, outcome, reason = %reason, "reader warm: ignoring peer hot-list");
            None
        }
    }
}

/// Result of the peer phase of a startup warm.
#[derive(Debug, Default, PartialEq, Eq)]
struct PeerHotLists {
    discovered: usize,
    lists: Vec<Vec<String>>,
}

/// Discover peers (explicit URLs or StatefulSet DNS) and fetch their
/// hot-lists concurrently, all within `timeout` (the per-peer timeout,
/// already clamped to the remaining warm budget). Lists follow the (sorted)
/// peer URL order. Discovery outcomes are counted separately from fetches.
async fn fetch_peer_hot_lists(
    explicit: Option<&str>,
    timeout: Duration,
    cap: usize,
    max_age: Option<Duration>,
) -> PeerHotLists {
    let count = |outcome: &'static str| {
        metrics::counter!("helix_reader_warm_peer_fetches_total", "outcome" => outcome).increment(1)
    };
    if explicit_peer_urls(explicit, &[]).is_some_and(|urls| urls.is_empty()) {
        debug!("reader warm: peer hot-lists disabled by HELIX_READER_WARM_PEER_URLS");
        return PeerHotLists::default();
    }
    let Some(client) = PEER_HTTP_CLIENT.as_ref() else {
        count("discovery_error");
        return PeerHotLists::default();
    };
    let started = Instant::now();
    let peers = match tokio::time::timeout(timeout, discover_peer_urls(explicit)).await {
        Err(_) => {
            count("discovery_timeout");
            warn!("reader warm: peer discovery timed out");
            return PeerHotLists::default();
        }
        Ok(Err(reason)) => {
            count("discovery_error");
            warn!(reason = %reason, "reader warm: peer discovery failed");
            return PeerHotLists::default();
        }
        Ok(Ok(peers)) if peers.is_empty() => {
            count("discovery_none");
            return PeerHotLists::default();
        }
        Ok(Ok(peers)) => peers,
    };
    let remaining = timeout.saturating_sub(started.elapsed());
    let fetches = peers
        .iter()
        .map(|peer| fetch_peer_hot_list(client, peer, remaining, cap, max_age));
    let lists = futures::future::join_all(fetches)
        .await
        .into_iter()
        .flatten()
        .collect();
    PeerHotLists {
        discovered: peers.len(),
        lists,
    }
}

/// Startup warm order: round-robin by rank across this reader's own list
/// (rank 0 source) and each peer's list — own[0], peerA[0], peerB[0], own[1],
/// … — deduped, then capped. Interleaving (not own-first) guarantees peers'
/// hottest names make the cut even when the own list alone fills the cap.
/// Retouching in this order after warm makes the LRU match it.
pub fn merge_warm_sources(own: &[String], peers: &[Vec<String>], cap: usize) -> Vec<String> {
    let sources: Vec<&[String]> = std::iter::once(own)
        .chain(peers.iter().map(Vec::as_slice))
        .collect();
    let longest = sources.iter().map(|source| source.len()).max().unwrap_or(0);
    let interleaved = (0..longest).flat_map(|rank| {
        sources
            .iter()
            .filter_map(move |source| source.get(rank))
            .cloned()
            .collect::<Vec<_>>()
    });
    normalize_names(interleaved, cap)
}

/// Currently loaded collections (most-recent first) followed by previously
/// persisted names that are no longer loaded, deduped and capped. Keeping the
/// older names means an idle-evicted-but-recently-hot collection still gets
/// warmed after a restart.
pub fn merge_hot_list(
    current_recent_first: Vec<String>,
    previous: &[String],
    cap: usize,
) -> Vec<String> {
    normalize_names(
        current_recent_first
            .into_iter()
            .chain(previous.iter().cloned()),
        cap,
    )
}

/// Outcome of warming one collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmOutcome {
    Warmed,
    /// Not found / dropped / degraded — nothing to warm, not an error.
    Skipped,
    Failed,
}

/// Classify a `get_collection` result. Not-found errors (the same shapes the
/// gateway maps to 404) are skips; anything quarantine-worthy or otherwise
/// unexpected is a failure.
pub fn classify_open_result<T>(result: &Result<T, GraphError>) -> WarmOutcome {
    match result {
        Ok(_) => WarmOutcome::Warmed,
        Err(err) => {
            let msg = err.to_string();
            if !err.should_quarantine_collection()
                && (msg.contains("not found")
                    || msg.contains("does not exist")
                    || msg.contains("NotFound"))
            {
                WarmOutcome::Skipped
            } else {
                WarmOutcome::Failed
            }
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct WarmReport {
    pub requested: usize,
    pub warmed: usize,
    pub skipped: usize,
    pub failed: usize,
    pub budget_exhausted: bool,
}

impl WarmReport {
    /// Requested names that did not complete within the budget.
    pub fn incomplete(&self) -> usize {
        self.requested
            .saturating_sub(self.warmed + self.skipped + self.failed)
    }
}

#[derive(Default)]
struct WarmCounts {
    warmed: AtomicUsize,
    skipped: AtomicUsize,
    failed: AtomicUsize,
}

/// Warm `names` (in order) with at most `concurrency` blocking opens in flight,
/// giving up after `budget`. Opens already running when the budget expires are
/// left to finish in the background (they are ordinary opens); no new ones start.
pub async fn run_startup_warm<F>(
    names: Vec<String>,
    concurrency: usize,
    budget: Duration,
    open: F,
) -> WarmReport
where
    F: Fn(&str) -> WarmOutcome + Send + Sync + 'static,
{
    let requested = names.len();
    let open = Arc::new(open);
    let counts = Arc::new(WarmCounts::default());
    let stop = Arc::new(AtomicBool::new(false));
    let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency.max(1)));

    let work = {
        let counts = Arc::clone(&counts);
        let stop = Arc::clone(&stop);
        async move {
            let mut handles = Vec::with_capacity(names.len());
            for name in names {
                let Ok(permit) = Arc::clone(&semaphore).acquire_owned().await else {
                    break;
                };
                let open = Arc::clone(&open);
                let counts = Arc::clone(&counts);
                let stop = Arc::clone(&stop);
                handles.push(tokio::task::spawn_blocking(move || {
                    let _permit = permit;
                    if stop.load(Ordering::Acquire) {
                        return;
                    }
                    let counter = match open(&name) {
                        WarmOutcome::Warmed => &counts.warmed,
                        WarmOutcome::Skipped => &counts.skipped,
                        WarmOutcome::Failed => &counts.failed,
                    };
                    counter.fetch_add(1, Ordering::AcqRel);
                }));
            }
            for handle in handles {
                let _ = handle.await;
            }
        }
    };
    let budget_exhausted = tokio::time::timeout(budget, work).await.is_err();
    stop.store(true, Ordering::Release);
    WarmReport {
        requested,
        warmed: counts.warmed.load(Ordering::Acquire),
        skipped: counts.skipped.load(Ordering::Acquire),
        failed: counts.failed.load(Ordering::Acquire),
        budget_exhausted,
    }
}

fn open_for_warm(collections: &CollectionManager, name: &str) -> WarmOutcome {
    if let Some((kind, _)) = collections.collection_degraded_marker(name) {
        debug!(
            collection = name,
            kind, "reader warm skipping degraded collection"
        );
        return WarmOutcome::Skipped;
    }
    let started = Instant::now();
    let result = allow_lsm_blocking(|| collections.get_collection(name).map(drop));
    let outcome = classify_open_result(&result);
    match (&result, outcome) {
        (Err(err), WarmOutcome::Failed) => warn!(
            collection = name,
            error = %err,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "reader warm failed to open collection"
        ),
        _ => debug!(
            collection = name,
            outcome = ?outcome,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "reader warm opened collection"
        ),
    }
    outcome
}

/// Bring a warmed collection's `DbReader` up to the current manifest, on its
/// current poll tier, so Ready implies "caught up to the end of warm": one
/// manifest poll, or a reopen where tier changes are reopens. A poll-tier
/// transition already in flight catches up by itself: skip it.
fn refresh_for_warm(collections: &CollectionManager, name: &str) -> WarmOutcome {
    let Ok(Some(storage)) = collections.get_loaded_collection(name) else {
        return WarmOutcome::Skipped;
    };
    if !storage.try_begin_poll_tier_transition() {
        return WarmOutcome::Skipped;
    }
    let interval = if storage.reader_poll_tier_is_fast() {
        reader_active_poll_interval_from_env()
    } else {
        reader_idle_poll_interval_from_env()
    };
    let result = allow_lsm_blocking(|| storage.backend.retier_lsm_reader(interval));
    storage.end_poll_tier_transition();
    match result {
        Ok(_) => WarmOutcome::Warmed,
        Err(err) => {
            warn!(
                collection = name,
                error = %err,
                "reader warm catch-up refresh failed"
            );
            WarmOutcome::Failed
        }
    }
}

/// Warm at most half the open-collection LRU (min 1) so the first live cold
/// opens after ready have headroom instead of evicting warmed collections.
fn warm_cap_for_open_limit(max_open: usize) -> usize {
    (max_open / 2).max(1)
}

/// Warming opens hottest-first, which leaves the hottest collection with the
/// OLDEST LRU stamp (and would flip the persisted order every restart).
/// Re-touch coldest→hottest through the cached-only lookup so the final LRU
/// order matches hot-list order. Never opens storage; unloaded names are no-ops.
fn retouch_in_hot_order(collections: &CollectionManager, names_hot_first: &[String]) {
    for name in names_hot_first.iter().rev() {
        let _ = collections.get_loaded_collection(name);
    }
}

/// `report` is the open phase, `catch_up` the refresh of every warmed
/// collection that follows it; both run inside the one warm budget.
fn record_warm_report(report: &WarmReport, catch_up: &WarmReport, elapsed: Duration) {
    metrics::histogram!("helix_reader_warm_duration_ms").record(elapsed.as_secs_f64() * 1000.0);
    for (outcome, value) in [
        ("warmed", report.warmed),
        ("skipped", report.skipped),
        ("failed", report.failed),
        ("incomplete", report.incomplete()),
        ("refreshed", catch_up.warmed),
        ("refresh_failed", catch_up.failed),
        ("refresh_incomplete", catch_up.incomplete()),
    ] {
        metrics::counter!("helix_reader_warm_collections_total", "outcome" => outcome)
            .increment(value as u64);
    }
    let budget_exhausted = report.budget_exhausted || catch_up.budget_exhausted;
    metrics::gauge!("helix_reader_warm_budget_exhausted").set(if budget_exhausted {
        1.0
    } else {
        0.0
    });
    info!(
        requested = report.requested,
        warmed = report.warmed,
        skipped = report.skipped,
        failed = report.failed,
        incomplete = report.incomplete(),
        refreshed = catch_up.warmed,
        refresh_skipped = catch_up.skipped,
        refresh_failed = catch_up.failed,
        refresh_incomplete = catch_up.incomplete(),
        budget_exhausted,
        elapsed_ms = elapsed.as_millis() as u64,
        "reader startup warm finished; reporting ready"
    );
}

/// Persist the hot-list every `interval` until the process exits.
async fn persist_hot_list_loop(
    collections: Arc<CollectionManager>,
    path: PathBuf,
    mut known: Vec<String>,
    cap: usize,
    interval: Duration,
) {
    let mut last_written: Option<Vec<String>> = None;
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    ticker.tick().await;
    loop {
        ticker.tick().await;
        let collections = Arc::clone(&collections);
        let path = path.clone();
        let previous = std::mem::take(&mut known);
        let unchanged = last_written.take();
        let joined = tokio::task::spawn_blocking(move || {
            let current = match collections.loaded_collection_names_by_recency() {
                Ok(current) => current,
                Err(err) => {
                    warn!(error = %err, "reader hot-list: failed to list loaded collections");
                    return (previous, unchanged);
                }
            };
            let merged = merge_hot_list(current, &previous, cap);
            publish_hot_list(Some(HotListFile {
                version: HOT_LIST_VERSION,
                written_at_ms: now_millis(),
                collections: merged.clone(),
            }));
            if unchanged.as_ref() == Some(&merged) {
                return (merged, unchanged);
            }
            match write_hot_list(&path, &merged, cap) {
                Ok(()) => {
                    metrics::counter!("helix_reader_hot_list_writes_total", "outcome" => "ok")
                        .increment(1);
                    (merged.clone(), Some(merged))
                }
                Err(err) => {
                    metrics::counter!("helix_reader_hot_list_writes_total", "outcome" => "error")
                        .increment(1);
                    warn!(path = %path.display(), error = %err, "failed to persist reader hot-list");
                    (merged, None)
                }
            }
        })
        .await;
        match joined {
            Ok((merged, written)) => {
                known = merged;
                last_written = written;
            }
            Err(err) => warn!(error = %err, "reader hot-list persist task failed"),
        }
    }
}

/// Start the reader warm + hot-list persistence. No-op for writers / LMDB and
/// for readers without `HELIX_LSM_CACHE_DIR`. Must be called inside a Tokio
/// runtime, before the listener starts accepting, so `/readyz/warm` reports
/// not-ready from the first probe.
pub fn spawn_reader_startup_warm(collections: Arc<CollectionManager>) {
    if !collections.config().storage_backend.is_reader() {
        return;
    }
    let Some(cache_root) = cache_root_from_env() else {
        info!("reader warm disabled: HELIX_LSM_CACHE_DIR unset");
        return;
    };
    let path = cache_root.join(HOT_LIST_FILE_NAME);
    let cap = hot_list_max();
    let enabled = warm_enabled();
    let budget = Duration::from_secs(env_u64(
        "HELIX_READER_WARM_BUDGET_SECS",
        DEFAULT_WARM_BUDGET_SECS,
    ));
    let interval_secs = env_u64(
        "HELIX_READER_HOT_LIST_INTERVAL_SECS",
        DEFAULT_HOT_LIST_INTERVAL_SECS,
    );
    let max_age = match env_u64(
        "HELIX_READER_HOT_LIST_MAX_AGE_SECS",
        DEFAULT_HOT_LIST_MAX_AGE_SECS,
    ) {
        0 => None,
        secs => Some(Duration::from_secs(secs)),
    };
    let peer_timeout = Duration::from_millis(env_u64(
        "HELIX_READER_WARM_PEER_TIMEOUT_MS",
        DEFAULT_PEER_TIMEOUT_MS,
    ));
    let peer_urls = std::env::var("HELIX_READER_WARM_PEER_URLS").ok();
    let gate = enabled.then(|| {
        set_startup_warm_pending(true);
        ClearWarmPendingOnDrop
    });

    tokio::spawn(async move {
        let started = Instant::now();
        let read_path = path.clone();
        let own = tokio::time::timeout(
            budget,
            tokio::task::spawn_blocking(move || load_hot_list(&read_path, cap, max_age)),
        )
        .await
        .ok()
        .and_then(Result::ok)
        .flatten();
        let previous = own
            .as_ref()
            .map(|file| file.collections.clone())
            .unwrap_or_default();
        // Serve our own list to peers right away (also while warming), keeping
        // the file's timestamp so peers apply the same staleness rule.
        publish_hot_list(Some(own.unwrap_or(HotListFile {
            version: HOT_LIST_VERSION,
            written_at_ms: now_millis(),
            collections: Vec::new(),
        })));

        if let Some(gate) = gate {
            let warm_cap = cap.min(warm_cap_for_open_limit(
                collections.max_open_collections_limit(),
            ));
            // Union with Ready peers' hot-lists: each reader only learns what
            // it served, so a collection hot on a peer would otherwise be cold
            // here. Bounded by the per-peer timeout AND the remaining budget.
            let peers = fetch_peer_hot_lists(
                peer_urls.as_deref(),
                peer_timeout.min(budget.saturating_sub(started.elapsed())),
                cap,
                max_age,
            )
            .await;
            let names = merge_warm_sources(&previous, &peers.lists, warm_cap);
            let retouch_names = names.clone();
            info!(
                own = previous.len(),
                discovered = peers.discovered,
                peers_ok = peers.lists.len(),
                collections = names.len(),
                budget_secs = budget.as_secs(),
                concurrency = warm_concurrency(),
                "reader startup warm starting"
            );
            let warm_collections = Arc::clone(&collections);
            let warmed = Arc::new(std::sync::Mutex::new(Vec::new()));
            let warmed_sink = Arc::clone(&warmed);
            let report = run_startup_warm(
                names,
                warm_concurrency(),
                budget.saturating_sub(started.elapsed()),
                move |name| {
                    let outcome = open_for_warm(&warm_collections, name);
                    if outcome == WarmOutcome::Warmed {
                        if let Ok(mut warmed) = warmed_sink.lock() {
                            warmed.push(name.to_string());
                        }
                    }
                    outcome
                },
            )
            .await;
            // Catch up before reporting ready: each warmed DbReader's view
            // dates from its open, and the feed poller skips a collection that
            // was still opening when its write was signalled.
            let warmed_names = warmed
                .lock()
                .map(|mut warmed| std::mem::take(&mut *warmed))
                .unwrap_or_default();
            let refresh_collections = Arc::clone(&collections);
            let catch_up = run_startup_warm(
                warmed_names,
                warm_concurrency(),
                budget.saturating_sub(started.elapsed()),
                move |name| refresh_for_warm(&refresh_collections, name),
            )
            .await;
            let retouch_collections = Arc::clone(&collections);
            let _ = tokio::task::spawn_blocking(move || {
                retouch_in_hot_order(&retouch_collections, &retouch_names)
            })
            .await;
            record_warm_report(&report, &catch_up, started.elapsed());
            drop(gate);
        }

        if interval_secs > 0 {
            persist_hot_list_loop(
                collections,
                path,
                previous,
                cap,
                Duration::from_secs(interval_secs),
            )
            .await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helix_engine::graph_core::config::Config;
    use tempfile::TempDir;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn hot_list_round_trip_preserves_order_and_caps() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join(HOT_LIST_FILE_NAME);
        write_hot_list(&path, &names(&["c", "a", "b", "a", "d"]), 3).unwrap();
        assert_eq!(read_hot_list(&path, 64, None), names(&["c", "a", "b"]));
        assert_eq!(read_hot_list(&path, 2, None), names(&["c", "a"]));
        assert!(
            !dir.path()
                .join(format!("{HOT_LIST_FILE_NAME}.tmp"))
                .exists(),
            "atomic write must not leave the tmp file behind"
        );
        // Overwrite replaces the whole list.
        write_hot_list(&path, &names(&["z"]), 3).unwrap();
        assert_eq!(read_hot_list(&path, 64, None), names(&["z"]));
    }

    #[test]
    fn hot_list_missing_or_corrupt_file_is_ignored() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join(HOT_LIST_FILE_NAME);
        assert!(read_hot_list(&path, 64, None).is_empty());
        fs::write(&path, b"{not json").unwrap();
        assert!(read_hot_list(&path, 64, None).is_empty());
        fs::write(
            &path,
            br#"{"version":99,"written_at_ms":0,"collections":["a"]}"#,
        )
        .unwrap();
        assert!(read_hot_list(&path, 64, None).is_empty());
    }

    #[test]
    fn hot_list_rejects_path_like_names() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join(HOT_LIST_FILE_NAME);
        fs::write(
            &path,
            br#"{"version":1,"written_at_ms":0,"collections":["ok","","..","a/b","..\\x","fine"]}"#,
        )
        .unwrap();
        assert_eq!(read_hot_list(&path, 64, None), names(&["ok", "fine"]));
    }

    #[test]
    fn hot_list_older_than_max_age_is_skipped() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join(HOT_LIST_FILE_NAME);
        let day_ago = now_millis() - 86_400_000 - 1_000;
        fs::write(
            &path,
            format!(r#"{{"version":1,"written_at_ms":{day_ago},"collections":["old"]}}"#),
        )
        .unwrap();
        let max_age = Some(Duration::from_secs(86_400));
        assert!(read_hot_list(&path, 64, max_age).is_empty());
        assert_eq!(read_hot_list(&path, 64, None), names(&["old"]));
        // Fresh and future (clock-skewed) timestamps are both accepted.
        write_hot_list(&path, &names(&["fresh"]), 3).unwrap();
        assert_eq!(read_hot_list(&path, 64, max_age), names(&["fresh"]));
        let future = now_millis() + 3_600_000;
        fs::write(
            &path,
            format!(r#"{{"version":1,"written_at_ms":{future},"collections":["skewed"]}}"#),
        )
        .unwrap();
        assert_eq!(read_hot_list(&path, 64, max_age), names(&["skewed"]));
    }

    #[test]
    fn warm_cap_leaves_lru_headroom() {
        assert_eq!(warm_cap_for_open_limit(32), 16);
        assert_eq!(warm_cap_for_open_limit(3), 1);
        assert_eq!(warm_cap_for_open_limit(1), 1);
        assert_eq!(warm_cap_for_open_limit(0), 1);
    }

    #[test]
    fn hot_list_write_error_is_reported_not_panicked() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("missing-dir").join(HOT_LIST_FILE_NAME);
        assert!(write_hot_list(&path, &names(&["a"]), 3).is_err());
    }

    #[test]
    fn merge_puts_current_first_then_previous_and_caps() {
        let merged = merge_hot_list(names(&["b", "x"]), &names(&["a", "b", "c"]), 4);
        assert_eq!(merged, names(&["b", "x", "a", "c"]));
        let merged = merge_hot_list(names(&["b", "x"]), &names(&["a", "b", "c"]), 2);
        assert_eq!(merged, names(&["b", "x"]));
    }

    #[test]
    fn classify_not_found_as_skip_and_quarantine_as_failure() {
        let ok: Result<(), GraphError> = Ok(());
        assert_eq!(classify_open_result(&ok), WarmOutcome::Warmed);
        let not_found: Result<(), GraphError> = Err(GraphError::New(
            "Collection 'gone' not found (no manifest in object store)".to_string(),
        ));
        assert_eq!(classify_open_result(&not_found), WarmOutcome::Skipped);
        let missing_sst: Result<(), GraphError> = Err(GraphError::New(
            "Storage error: io error: Data error: object store error (Object at location prod/repo/compacted/01ABC.sst not found: <Error><Code>NoSuchKey</Code></Error>)".to_string(),
        ));
        assert_eq!(classify_open_result(&missing_sst), WarmOutcome::Failed);
        let other: Result<(), GraphError> = Err(GraphError::New("boom".to_string()));
        assert_eq!(classify_open_result(&other), WarmOutcome::Failed);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn warm_counts_outcomes_and_finishes_within_budget() {
        let report = run_startup_warm(
            names(&["ok1", "gone", "bad", "ok2"]),
            2,
            Duration::from_secs(10),
            |name| match name {
                "gone" => WarmOutcome::Skipped,
                "bad" => WarmOutcome::Failed,
                _ => WarmOutcome::Warmed,
            },
        )
        .await;
        assert_eq!(
            report,
            WarmReport {
                requested: 4,
                warmed: 2,
                skipped: 1,
                failed: 1,
                budget_exhausted: false,
            }
        );
        assert_eq!(report.incomplete(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn warm_gives_up_at_budget_and_respects_concurrency() {
        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));
        let (in_flight_c, max_seen_c) = (Arc::clone(&in_flight), Arc::clone(&max_seen));
        let started = Instant::now();
        let report = run_startup_warm(
            names(&["a", "b", "c", "d", "e", "f"]),
            2,
            Duration::from_millis(100),
            move |_| {
                let now = in_flight_c.fetch_add(1, Ordering::AcqRel) + 1;
                max_seen_c.fetch_max(now, Ordering::AcqRel);
                std::thread::sleep(Duration::from_millis(3_000));
                in_flight_c.fetch_sub(1, Ordering::AcqRel);
                WarmOutcome::Warmed
            },
        )
        .await;
        assert!(report.budget_exhausted);
        // Wide margin: returning at all before the 3s opens finish proves the
        // budget (not the opens) ended the warm, even on a loaded CI box.
        assert!(started.elapsed() < Duration::from_millis(2_500));
        assert_eq!(report.warmed, 0);
        assert_eq!(report.incomplete(), 6);
        // Let the in-flight opens drain; queued names must never start.
        let drain_deadline = Instant::now() + Duration::from_secs(20);
        while in_flight.load(Ordering::Acquire) > 0 && Instant::now() < drain_deadline {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(in_flight.load(Ordering::Acquire), 0);
        assert!(max_seen.load(Ordering::Acquire) <= 2);
    }

    /// Minimal one-route HTTP server: answers every connection with `status`
    /// and `body` after `delay`. Returns its base URL.
    async fn peer_server(status: u16, body: String, delay: Duration) -> String {
        peer_server_with_length(status, body, delay, true).await
    }

    /// `with_length = false` omits `Content-Length` (close-delimited body), so
    /// the size cap must be enforced while streaming rather than up front.
    async fn peer_server_with_length(
        status: u16,
        body: String,
        delay: Duration,
        with_length: bool,
    ) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let body = body.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 2048];
                    let _ = socket.read(&mut buf).await;
                    tokio::time::sleep(delay).await;
                    let length = if with_length {
                        format!("Content-Length: {}\r\n", body.len())
                    } else {
                        String::new()
                    };
                    let response = format!(
                        "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\n{length}Connection: close\r\n\r\n{body}"
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                });
            }
        });
        format!("http://{addr}")
    }

    fn hot_list_json(written_at_ms: u64, collections: &[&str]) -> String {
        serde_json::to_string(&HotListFile {
            version: HOT_LIST_VERSION,
            written_at_ms,
            collections: names(collections),
        })
        .unwrap()
    }

    #[test]
    fn merge_warm_sources_round_robins_own_and_peers_dedup_cap() {
        let own = names(&["a", "b"]);
        let peers = vec![names(&["x", "a", "y"]), names(&["p", "q"])];
        assert_eq!(
            merge_warm_sources(&own, &peers, 16),
            names(&["a", "x", "p", "b", "q", "y"])
        );
        assert_eq!(merge_warm_sources(&own, &peers, 3), names(&["a", "x", "p"]));
        assert_eq!(merge_warm_sources(&[], &peers, 2), names(&["x", "p"]));
        assert_eq!(merge_warm_sources(&own, &[], 16), own);
        let bad_peer = vec![names(&["..", "a/b", "ok"])];
        assert_eq!(merge_warm_sources(&[], &bad_peer, 16), names(&["ok"]));
    }

    #[test]
    fn peer_names_make_the_cut_when_own_list_fills_the_warm_cap() {
        let warm_cap = 16;
        let own: Vec<String> = (0..20).map(|i| format!("own{i}")).collect();
        let peers = vec![names(&["peer0", "peer1", "peer2"])];
        let merged = merge_warm_sources(&own, &peers, warm_cap);
        assert_eq!(merged.len(), warm_cap);
        for peer in ["peer0", "peer1", "peer2"] {
            assert!(
                merged.contains(&peer.to_string()),
                "{peer} missing: {merged:?}"
            );
        }
        assert_eq!(
            &merged[..4],
            &names(&["own0", "peer0", "own1", "peer1"])[..]
        );
    }

    #[test]
    fn statefulset_identity_comes_from_kubelet_etc_hosts() {
        let hosts = "# Kubernetes-managed hosts file.\n127.0.0.1\tlocalhost\n::1\tlocalhost ip6-localhost\n10.0.0.7\thelion-reader-1.helion-reader-headless.helion.svc.cluster.local\thelion-reader-1\n";
        assert_eq!(
            statefulset_identity(hosts, "helion-reader-1"),
            Some((
                "10.0.0.7".to_string(),
                "helion-reader-headless.helion.svc.cluster.local".to_string()
            ))
        );
        assert_eq!(statefulset_identity(hosts, "helion-reader-0"), None);
        assert_eq!(
            statefulset_identity("127.0.0.1 localhost\n", "laptop"),
            None
        );
    }

    #[test]
    fn explicit_peer_urls_drop_self_and_support_off() {
        let selves = ["helion-reader-0", "10.0.0.6"];
        assert_eq!(explicit_peer_urls(None, &selves), None);
        assert_eq!(explicit_peer_urls(Some("  "), &selves), None);
        assert_eq!(explicit_peer_urls(Some("off"), &selves), Some(Vec::new()));
        let urls = explicit_peer_urls(
            Some("http://helion-reader-0.svc:6969, http://helion-reader-1.svc:6969/,http://10.0.0.6:6969,,http://[fd00::1]:6969"),
            &selves,
        )
        .unwrap();
        assert_eq!(
            urls,
            vec![
                "http://helion-reader-1.svc:6969".to_string(),
                "http://[fd00::1]:6969".to_string()
            ]
        );
    }

    #[test]
    fn peer_hot_list_validation_drops_bad_names_and_stale_lists() {
        let fresh = hot_list_json(now_millis(), &["ok", "..", "a/b", "ok", "fine"]);
        let parsed = parse_hot_list(fresh.as_bytes(), 64, Some(Duration::from_secs(60))).unwrap();
        assert_eq!(parsed.collections, names(&["ok", "fine"]));
        let stale = hot_list_json(now_millis() - 120_000, &["old"]);
        assert_eq!(
            parse_hot_list(stale.as_bytes(), 64, Some(Duration::from_secs(60)))
                .unwrap_err()
                .0,
            "stale"
        );
        assert_eq!(
            parse_hot_list(b"<html>", 64, None).unwrap_err().0,
            "corrupt"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn peer_fetch_errors_and_timeouts_are_ignored_within_bound() {
        let ok = peer_server(
            200,
            hot_list_json(now_millis(), &["p1", "../x", "p2"]),
            Duration::ZERO,
        )
        .await;
        let not_found = peer_server(404, "{}".to_string(), Duration::ZERO).await;
        let garbage = peer_server(200, "not json".to_string(), Duration::ZERO).await;
        let slow = peer_server(
            200,
            hot_list_json(now_millis(), &["late"]),
            Duration::from_secs(10),
        )
        .await;
        let timeout = Duration::from_millis(300);
        let client = PEER_HTTP_CLIENT.as_ref().unwrap();
        assert_eq!(
            fetch_peer_hot_list(client, &ok, timeout, 64, None).await,
            Some(names(&["p1", "p2"]))
        );
        assert_eq!(
            fetch_peer_hot_list(client, &not_found, timeout, 64, None).await,
            None
        );
        assert_eq!(
            fetch_peer_hot_list(client, &garbage, timeout, 64, None).await,
            None
        );
        // Nothing listens on port 9 locally → connection error, not a panic.
        assert_eq!(
            fetch_peer_hot_list(client, "http://127.0.0.1:9", timeout, 64, None).await,
            None
        );
        let started = Instant::now();
        assert_eq!(
            fetch_peer_hot_list(client, &slow, timeout, 64, None).await,
            None
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "timeout must bound the fetch"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn oversized_peer_responses_are_rejected() {
        let client = PEER_HTTP_CLIENT.as_ref().unwrap();
        let timeout = Duration::from_secs(5);
        let huge: Vec<String> = (0..20_000).map(|i| format!("collection_{i:06}")).collect();
        let body = serde_json::to_string(&HotListFile {
            version: HOT_LIST_VERSION,
            written_at_ms: now_millis(),
            collections: huge,
        })
        .unwrap();
        assert!(body.len() > MAX_PEER_RESPONSE_BYTES);
        let declared = peer_server_with_length(200, body.clone(), Duration::ZERO, true).await;
        let streamed = peer_server_with_length(200, body, Duration::ZERO, false).await;
        assert_eq!(
            fetch_peer_hot_list(client, &declared, timeout, 64, None).await,
            None
        );
        assert_eq!(
            fetch_peer_hot_list(client, &streamed, timeout, 64, None).await,
            None
        );
        // Close-delimited bodies under the cap still parse.
        let small = peer_server_with_length(
            200,
            hot_list_json(now_millis(), &["ok"]),
            Duration::ZERO,
            false,
        )
        .await;
        assert_eq!(
            fetch_peer_hot_list(client, &small, timeout, 64, None).await,
            Some(names(&["ok"]))
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn peer_hot_lists_union_from_explicit_urls_and_bounded_by_timeout() {
        let a = peer_server(
            200,
            hot_list_json(now_millis(), &["a1", "shared"]),
            Duration::ZERO,
        )
        .await;
        let b = peer_server(
            200,
            hot_list_json(now_millis(), &["shared", "b1"]),
            Duration::ZERO,
        )
        .await;
        let slow = peer_server(
            200,
            hot_list_json(now_millis(), &["late"]),
            Duration::from_secs(10),
        )
        .await;
        let explicit = format!("{a},{slow},{b}");
        let started = Instant::now();
        let peers =
            fetch_peer_hot_lists(Some(&explicit), Duration::from_millis(500), 64, None).await;
        let elapsed = started.elapsed();
        assert_eq!(peers.discovered, 3);
        let lists = peers.lists;
        assert!(
            elapsed < Duration::from_secs(5),
            "slow peer must not hold the warm"
        );
        assert_eq!(
            lists,
            vec![names(&["a1", "shared"]), names(&["shared", "b1"])]
        );
        assert_eq!(
            merge_warm_sources(&names(&["own"]), &lists, 16),
            names(&["own", "a1", "shared", "b1"])
        );
        // `off` disables peers entirely.
        assert_eq!(
            fetch_peer_hot_lists(Some("off"), Duration::from_millis(500), 64, None).await,
            PeerHotLists::default()
        );
    }

    #[test]
    fn hot_collections_endpoint_serves_only_published_reader_list() {
        let _gate_lock = warm_gate_test_lock();
        publish_hot_list(None);
        assert_eq!(
            hot_collections_response_body(),
            None,
            "writer/no-warm → 404"
        );
        let file = HotListFile {
            version: HOT_LIST_VERSION,
            written_at_ms: 42,
            collections: names(&["a", "b"]),
        };
        publish_hot_list(Some(file.clone()));
        let body = hot_collections_response_body().unwrap();
        publish_hot_list(None);
        assert_eq!(serde_json::from_slice::<HotListFile>(&body).unwrap(), file);
    }

    #[test]
    fn readiness_gate_defaults_ready_and_clears_on_drop() {
        let _gate_lock = warm_gate_test_lock();
        assert!(startup_warm_ready(), "writers/tests start ready");
        set_startup_warm_pending(true);
        let gate = ClearWarmPendingOnDrop;
        assert!(!startup_warm_ready());
        drop(gate);
        assert!(startup_warm_ready());
    }

    #[test]
    fn writer_role_never_enters_warming() {
        let _gate_lock = warm_gate_test_lock();
        let tmp = TempDir::new().unwrap();
        let mgr = Arc::new(
            CollectionManager::new(
                tmp.path().to_path_buf(),
                Config::new(16, 128, 768, 1).with_lsm_in_memory(),
            )
            .unwrap(),
        );
        spawn_reader_startup_warm(mgr);
        assert!(startup_warm_ready());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_open_warms_existing_and_skips_missing_collections() {
        let tmp = TempDir::new().unwrap();
        let mgr = Arc::new(
            CollectionManager::new(
                tmp.path().to_path_buf(),
                Config::new(16, 128, 768, 1).with_lsm_in_memory(),
            )
            .unwrap(),
        );
        let setup_mgr = Arc::clone(&mgr);
        tokio::task::spawn_blocking(move || {
            allow_lsm_blocking(|| {
                drop(setup_mgr.create_collection("warm_a").unwrap());
                drop(setup_mgr.create_collection("warm_b").unwrap());
                setup_mgr.get_collection("warm_a").unwrap();
                assert_eq!(
                    setup_mgr.loaded_collection_names_by_recency().unwrap(),
                    names(&["warm_a", "warm_b"])
                );
                setup_mgr.evict_collection("warm_a");
                setup_mgr.evict_collection("warm_b");
                assert_eq!(setup_mgr.loaded_count(), 0);
            })
        })
        .await
        .unwrap();

        // Hot-list order (hottest first) deliberately differs from the
        // creation order; concurrency 1 opens strictly in list order, which on
        // its own would leave warm_b (hottest) with the OLDEST LRU stamp.
        let hot_list = names(&["warm_b", "warm_missing", "warm_a"]);
        let warm_mgr = Arc::clone(&mgr);
        let report = run_startup_warm(hot_list.clone(), 1, Duration::from_secs(30), move |name| {
            open_for_warm(&warm_mgr, name)
        })
        .await;
        assert_eq!(report.warmed, 2);
        assert_eq!(report.skipped, 1);
        assert_eq!(report.failed, 0);
        assert!(!report.budget_exhausted);
        assert_eq!(mgr.loaded_count(), 2);

        let retouch_mgr = Arc::clone(&mgr);
        let recency = tokio::task::spawn_blocking(move || {
            retouch_in_hot_order(&retouch_mgr, &hot_list);
            retouch_mgr.loaded_collection_names_by_recency().unwrap()
        })
        .await
        .unwrap();
        assert_eq!(recency, names(&["warm_b", "warm_a"]));
    }

    use crate::helix_engine::storage_core::backend::StorageBackend;

    struct EnvVar(&'static str, Option<String>);
    impl EnvVar {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self(key, previous)
        }
    }
    impl Drop for EnvVar {
        fn drop(&mut self) {
            match &self.1 {
                Some(value) => std::env::set_var(self.0, value),
                None => std::env::remove_var(self.0),
            }
        }
    }
    fn put_node(writer: &CollectionManager, id: u128) {
        let storage = writer.get_or_create_collection("hot").unwrap();
        let mut w = storage.backend.begin_write().unwrap();
        storage
            .create_node_be(
                &mut w,
                "Sym",
                Vec::<(String, crate::protocol::value::Value)>::new(),
                None,
                Some(id),
            )
            .unwrap();
        storage.backend.commit(w).unwrap();
    }
    fn reader_sees(reader: &CollectionManager, id: u128) -> bool {
        let storage = reader.get_collection("hot").unwrap();
        let r = storage.backend.begin_read().unwrap();
        storage.get_node_be(&r, &id).is_ok()
    }
    async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        tokio::task::spawn_blocking(move || allow_lsm_blocking(f))
            .await
            .unwrap()
    }

    /// Ready must mean "caught up with the writer". A warmed collection's
    /// `DbReader` view dates from its open, and the feed poller skips a
    /// collection that is still opening when its write is signalled, so a
    /// write landing between the open and the end of warm stays invisible
    /// until the next manifest poll. Modeled with the collection resident
    /// before warm starts and the reader's own poll far beyond the test.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[serial_test::serial]
    async fn startup_warm_catches_up_resident_collections_before_ready() {
        use crate::helix_engine::storage_core::backend::{
            LsmRole, LsmStorage, StorageBackendConfig,
        };
        use crate::helix_engine::storage_core::backend_any;

        let _gate_lock = warm_gate_test_lock();
        let cache_dir = TempDir::new().unwrap();
        let _cache = EnvVar::set("HELIX_LSM_CACHE_DIR", cache_dir.path().to_str().unwrap());
        let _prefix = EnvVar::set("HELIX_LSM_PREFIX", "warm_catch_up");
        // Far beyond this test's runtime: any visibility change is the warm's.
        let _poll = EnvVar::set("HELIX_LSM_READER_MANIFEST_POLL_MS", "60000");
        let _peers = EnvVar::set("HELIX_READER_WARM_PEER_URLS", "off");
        let _persist = EnvVar::set("HELIX_READER_HOT_LIST_INTERVAL_SECS", "0");
        let store: Arc<dyn slatedb::object_store::ObjectStore> =
            Arc::new(slatedb::object_store::memory::InMemory::new());
        let _store = backend_any::set_lsm_test_object_store(store);
        let backend = |role| StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role,
        };
        let writer_dir = TempDir::new().unwrap();
        let reader_dir = TempDir::new().unwrap();
        let writer = Arc::new(
            CollectionManager::new(
                writer_dir.path().to_path_buf(),
                Config::new(16, 128, 768, 1).with_storage_backend(backend(LsmRole::Writer)),
            )
            .unwrap(),
        );
        let reader = Arc::new(
            CollectionManager::new(
                reader_dir.path().to_path_buf(),
                Config::new(16, 128, 768, 1).with_storage_backend(backend(LsmRole::Reader)),
            )
            .unwrap(),
        );

        // The reader opens the collection at a view holding node 1 only ...
        let (w, r) = (Arc::clone(&writer), Arc::clone(&reader));
        blocking(move || {
            put_node(&w, 1);
            assert!(reader_sees(&r, 1));
            assert!(!reader_sees(&r, 2));
        })
        .await;
        // ... then a write lands that the resident DbReader has not polled.
        let (w, r) = (Arc::clone(&writer), Arc::clone(&reader));
        assert!(
            !blocking(move || {
                put_node(&w, 2);
                reader_sees(&r, 2)
            })
            .await,
            "precondition: the resident DbReader must still serve the old view"
        );
        write_hot_list(
            &cache_dir.path().join(HOT_LIST_FILE_NAME),
            &names(&["hot"]),
            64,
        )
        .unwrap();

        spawn_reader_startup_warm(Arc::clone(&reader));
        assert!(
            !startup_warm_ready(),
            "warm must gate readiness from the first probe"
        );
        let started = Instant::now();
        while !startup_warm_ready() {
            assert!(
                started.elapsed() < Duration::from_secs(60),
                "warm did not finish"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let r = Arc::clone(&reader);
        assert!(
            blocking(move || reader_sees(&r, 2)).await,
            "Ready must imply the warmed collection is caught up with the writer"
        );
    }

    async fn wait_for_warm_ready() {
        let started = Instant::now();
        while !startup_warm_ready() {
            assert!(
                started.elapsed() < Duration::from_secs(60),
                "warm did not finish"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// The catch-up reopen must keep the collection on the tier its belief
    /// says (a fast belief over an idle `DbReader` never promotes, so writes
    /// stay invisible for the whole idle poll), and a warm open must count as
    /// a read so the sweeper does not demote it on its first tick.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[serial_test::serial]
    async fn startup_warm_keeps_fast_tier_and_read_window() {
        use crate::helix_engine::storage_core::backend::{
            LsmRole, LsmStorage, StorageBackendConfig,
        };
        use crate::helix_engine::storage_core::backend_any;

        let _gate_lock = warm_gate_test_lock();
        let cache_dir = TempDir::new().unwrap();
        let _cache = EnvVar::set("HELIX_LSM_CACHE_DIR", cache_dir.path().to_str().unwrap());
        let _prefix = EnvVar::set("HELIX_LSM_PREFIX", "warm_tier");
        let _fast = EnvVar::set("HELIX_LSM_READER_MANIFEST_POLL_MS", "200");
        let _idle = EnvVar::set("HELIX_LSM_READER_IDLE_POLL_MS", "600000");
        // Sweeper tick = idle_after / 4 = 2s; a never-read open is demoted on it.
        let _idle_after = EnvVar::set("HELIX_LSM_READER_IDLE_AFTER_MS", "8000");
        let _peers = EnvVar::set("HELIX_READER_WARM_PEER_URLS", "off");
        let _persist = EnvVar::set("HELIX_READER_HOT_LIST_INTERVAL_SECS", "0");
        let store: Arc<dyn slatedb::object_store::ObjectStore> =
            Arc::new(slatedb::object_store::memory::InMemory::new());
        let _store = backend_any::set_lsm_test_object_store(store);
        let backend = |role| StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role,
        };
        let (writer_dir, reader_dir) = (TempDir::new().unwrap(), TempDir::new().unwrap());
        let writer = Arc::new(
            CollectionManager::new(
                writer_dir.path().to_path_buf(),
                Config::new(16, 128, 768, 1).with_storage_backend(backend(LsmRole::Writer)),
            )
            .unwrap(),
        );
        let reader = Arc::new(
            CollectionManager::new(
                reader_dir.path().to_path_buf(),
                Config::new(16, 128, 768, 1).with_storage_backend(backend(LsmRole::Reader)),
            )
            .unwrap(),
        );
        let w = Arc::clone(&writer);
        blocking(move || put_node(&w, 1)).await;
        write_hot_list(
            &cache_dir.path().join(HOT_LIST_FILE_NAME),
            &names(&["hot"]),
            64,
        )
        .unwrap();

        // Warm does the cold open itself (nothing resident), as after a restart.
        spawn_reader_startup_warm(Arc::clone(&reader));
        wait_for_warm_ready().await;
        tokio::time::sleep(Duration::from_millis(3_500)).await;
        let loaded =
            |reader: &CollectionManager| reader.get_loaded_collection("hot").unwrap().unwrap();
        assert!(
            loaded(&reader).reader_poll_tier_is_fast(),
            "a warm open is read activity: the sweeper must not demote it on its first tick"
        );

        let w = Arc::clone(&writer);
        blocking(move || put_node(&w, 2)).await;
        let started = Instant::now();
        loop {
            let r = Arc::clone(&reader);
            let visible = blocking(move || {
                let storage = loaded(&r);
                let txn = storage.backend.begin_read().unwrap();
                storage.get_node_be(&txn, &2).is_ok()
            })
            .await;
            if visible {
                break;
            }
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "after catch-up the DbReader must poll at the fast tier its belief claims"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}
