# Helion — Operations Guide

## Building

Requires Rust toolchain (stable). The workspace has five crates:

```
helixdb/          -- core library (engine, gateway, protocol)
helix-container/  -- binary entrypoint
helix-cli/        -- CLI tool (init, build, start, stop)
hbuild/           -- schema compiler
get_routes/       -- proc-macro for route generation
```

Build:
```bash
cargo build --release -p helix-container
```

Run tests:
```bash
cargo test --workspace
```

Run vector benchmarks:
```bash
cargo bench -p helixdb --bench vector_benchmarks
```

The resulting binary is `target/release/helix-container`.

---

## Environment Variables

| Variable | Default | Notes |
|---|---|---|
| `HELIX_CONFIG_PATH` | `~/.helix/repo/helix-db/helix-container/src/config.hx.json` | Path to `config.hx.json` |
| `HELIX_DATA_DIR` | `~/.helix` | Root data dir. Collections stored under `{HELIX_DATA_DIR}/user/collections/` |
| `HELIX_PORT` | `6969` | TCP listen port |
| `HELIX_REQUEST_TIMEOUT_SECS` | `300` | Max non-stream request handler time before returning `504`. Clamped to min 1 second |
| `HELIX_RAFT_SECRET` | (none) | Shared secret for Raft inter-node auth. Fallback when `raft_secret` is not in config |
| `HELIX_LOG` | `info` | Log level filter for structured tracing output: `trace`, `debug`, `info`, `warn`, `error` |
| `HELIX_POOL_SIZE` | `2048` | Worker thread pool size. Each worker is a dedicated OS thread reading from a flume channel |
| `HELIX_GATEWAY_IMPL` | `axum` | Axum/Hyper/Tower gateway. Set to `legacy` to force the old socket + thread-pool gateway |
| `HELIX_GATEWAY_WRITER_URL` | unset | Enables Helix Cloud gateway routing when set. All write-like, index, maintenance, and admin client routes proxy to this writer base URL before local handlers can run. Aliases: `HELIX_CLOUD_WRITER_URL`, `HELIX_WRITER_URL` |
| `HELIX_GATEWAY_READER_URLS` | unset | Comma-separated reader base URLs. Read, search, and scan routes start on healthy readers in round-robin order; writes are never routed to these URLs. Aliases: `HELIX_CLOUD_READER_URLS`, `HELIX_READER_URLS` |
| `HELIX_GATEWAY_READER_EVICT_MS` | `5000` | Temporary reader eviction cooldown after a proxy failure or retryable reader status (`429`, `502`, `503`, `504`) |
| `HELIX_GATEWAY_READS_INCLUDE_WRITER` | `false` | Append the writer to normal read routing. Even when false, reads fall back to the writer if no configured reader is currently healthy. Alias: `HELIX_CLOUD_READS_INCLUDE_WRITER` |
| `HELIX_GATEWAY_BUFFER_DIR` | unset | Enables the Helix Cloud durable mutation buffer when cloud gateway routing is active. Writer proxy failures and retryable writer statuses are fsynced here and acknowledged with `202 Accepted`; replay is at-least-once and request-hash coalesced |
| `HELIX_GATEWAY_BUFFER_MAX_ENTRIES` | `10000` | Maximum number of buffered mutation files before new writer-failure buffers are rejected with retryable overload |
| `HELIX_GATEWAY_BUFFER_MAX_BYTES` | `1073741824` | Maximum total buffered mutation bytes |
| `HELIX_GATEWAY_BUFFER_MAX_REQUEST_BYTES` | `HELIX_MAX_BODY_MB` limit | Maximum single buffered request body size |
| `HELIX_GATEWAY_BUFFER_REPLAY_MS` | `1000` | Replay worker sleep between spool scans |
| `HELIX_WRITE_QUEUE_DIR` | unset | Legacy alias for `HELIX_GATEWAY_BUFFER_DIR`. It applies only to Helix Cloud gateway writer-failure buffering, not to the local Axum submit path |
| `HELIX_WRITE_QUEUE_WORKERS` | `4` | Reserved for Layer 6 fair write queue workers; not wired in the current Axum submit path |
| `HELIX_WRITE_QUEUE_PER_COLLECTION_CAP` | `1024` | Reserved for the Layer 6 per-collection pending write cap; not wired in the current Axum submit path |
| `HELIX_WRITE_QUEUE_QUANTUM` | `16` | Reserved for Layer 6 deficit-round-robin scheduling; not wired in the current Axum submit path |
| `HELIX_KEEP_ALIVE_SECS` | `30` | Idle connection timeout in seconds. Connections idle longer than this are closed |
| `HELIX_PER_COLLECTION_INFLIGHT_CAP` | `0` | Fallback per-collection in-flight cap. Keep `0` in multi-tenant production unless read routes have separate capacity, because the fallback also caps reads |
| `HELIX_WRITE_INFLIGHT_CAP` | unset | Process-wide write admission cap. Applies only to write routes and rejects before request body mutation work with retryable overload |
| `HELIX_WRITE_RATE_PER_COLLECTION_PER_SEC` | unset | Governor-backed keyed write rate limit per collection/user. Applies before mutation work; rejected writes return retryable overload |
| `HELIX_WRITE_RATE_BURST` | same as rate | Burst size for the keyed write rate limiter |
| `HELIX_INDEX_INFLIGHT_CAP` | unset | Process-wide index admission cap for create/drop index routes |
| `HELIX_INDEX_EXECUTOR_WORKERS` | `2` | Global bounded background index executor worker count |
| `HELIX_INDEX_EXECUTOR_QUEUE_CAP` | `1024` | Global background index executor pending job cap |
| `HELIX_INDEX_JOB_MAX_RUNTIME_MS` | `5000` | Max wall-clock time for one background optimizer turn before it reschedules remaining dense build/merge work. Set `0` to disable the runtime budget |
| `HELIX_INDEX_JOB_MAX_BUILD_SEGMENTS` | `1` | Max building dense segments flushed per optimizer turn |
| `HELIX_INDEX_JOB_MAX_MERGES` | `1` | Max dense merge publishes per optimizer turn |
| `HELIX_MAX_CONCURRENT_BUILDS` | derived from CPU, capped at `2` | Global HNSW build semaphore. Covers both Building→Indexed flushes and dense merge HNSW preparation, including vector export for merge builds, to bound peak RSS |
| `HELIX_PER_COLLECTION_SEGMENT_LIMIT` | unset | Point-upsert circuit breaker. When set, rejects `PUT /collections/{name}/points` with retryable 503 if the loaded collection has more indexed dense segments than the limit. Deletes bypass this breaker |
| `HELIX_PER_COLLECTION_SEGMENT_RETRY_SECS` | `5` | `Retry-After` seconds returned by the per-collection segment circuit breaker |
| `HELIX_SEGMENT_BREAKER_RECOVERY_RATIO` | `0.75` | Hysteresis ratio for the per-collection segment circuit breaker. Once tripped (segments > limit), the breaker only closes when segments drop to `limit * ratio`. Prevents stuck open/close loops when segment drain rate barely clears the trip threshold |
| `HELIX_SEGMENT_BREAKER_OPTIMIZER_THROTTLE_SECS` | `30` | Minimum seconds between merge optimizer submissions for the same collection when the circuit breaker is tripped |
| `HELIX_MAINTENANCE_INFLIGHT_CAP` | unset | Process-wide maintenance/admin admission cap for snapshots, Raft, and backfills |
| `HELIX_INFLIGHT_CAP_WRITES` | unset | Per-collection cap for write/index routes such as point upserts, payload updates, deletes, and `/collections/{name}/index` |
| `HELIX_INFLIGHT_CAP_GRAPH_WRITES` | unset | Per-collection cap for native graph ingest/write routes |
| `HELIX_INFLIGHT_CAP_SEARCH`, `HELIX_INFLIGHT_CAP_SCROLL`, `HELIX_INFLIGHT_CAP_COUNT`, `HELIX_INFLIGHT_CAP_FACET`, `HELIX_INFLIGHT_CAP_COLLECTION_INFO` | unset | Optional per-collection caps for read/search routes. Use in SaaS to keep one hot collection from monopolizing global search/scan/read slots |
| `HELIX_SEGMENT_FORMAT` | unset | Set to `hvs8` to publish newly merged dense mmap sidecars in scalar-quantized HVS8 format. Existing HVEC sidecars convert only when their segments are merged |
| `HELIX_DENSE_MAX_INDEXED_SEGMENTS` | `16` | Target max indexed dense vector segments per vector space before background merge selection. Lower values reduce search fanout at the cost of more merge CPU |
| `HELIX_DENSE_MERGE_MAX_FANIN` | `8` | Max indexed dense segments merged in one optimizer merge operation. Bounds one merge's memory, CPU, and sidecar rewrite time |
| `HELIX_DENSE_DRAIN_MERGE_MAX_FANIN` | `2` | Max indexed dense segments merged in one optimizer operation while the collection is at or above the segment-breaker recovery floor. This breaker-drain override favors small publishable merges under cold I/O contention and leaves normal merge fan-in unchanged |
| `HELIX_SCAN_PARTITION_MAX` | `4096` | Max `partition.total` accepted by `/v1/collections/{name}/points/scan` |
| `HELIX_SCAN_BUDGET_MAX` | `1000000` | Max `max_scan` budget accepted by the native Helion scan endpoint |
| `HELIX_SCAN_INDEX_CANDIDATE_MAX` | `100000` | Max broad indexed-filter candidate set used by scroll/scan before falling back to primary ID-order scanning |
| `HELIX_QDRANT_SCROLL_SCAN_BUDGET_MIN` | `1024` | Minimum points inspected by one filtered Qdrant-compatible scroll call before yielding a continuation offset |
| `HELIX_QDRANT_SCROLL_SCAN_BUDGET_MULTIPLIER` | `64` | Filtered scroll effort multiplier: `limit * multiplier`, clamped by min/max |
| `HELIX_QDRANT_SCROLL_SCAN_BUDGET_MAX` | `4096` | Max points inspected by one filtered Qdrant-compatible scroll call before yielding; set `0` to disable |
| `HELIX_QDRANT_SCROLL_SCAN_BUDGET_MS` | `250` | Wall-clock budget for one filtered Qdrant-compatible scroll call; set `0` to disable |
| `HELIX_SEARCH_BLOCKING_CAP` | unset | Reserved blocking execution slots for client search routes (`/points/search`, `/points/query`, `/points/hybrid_query`, and native graph reads). Use this to prevent status/count/scroll traffic from starving user searches |
| `HELIX_SCAN_BLOCKING_CAP` | unset | Reserved blocking execution slots for scan-style enumeration (`/points/scroll` and native `/v1/collections/{name}/points/scan`) |
| `HELIX_READ_BLOCKING_CAP` | unset | Blocking execution slots for low-priority read probes such as collection info, count, and facet. Keep this lower than search in SaaS so background/status reads shed before client search does |
| `HELIX_SEARCH_BLOCKING_QUEUE_WAIT_MS` | `2000` | How long search requests can wait asynchronously for a search slot before returning overload. Waiting does not consume a blocking thread |
| `HELIX_SCAN_BLOCKING_QUEUE_WAIT_MS` | `2000` | How long scan requests can wait asynchronously for a scan slot before returning overload |
| `HELIX_TOKIO_WORKER_THREADS` | detected CPU count | Tokio runtime worker threads for async accept/read/write work |
| `HELIX_TOKIO_MAX_BLOCKING_THREADS` | `512` | Explicit Tokio blocking pool ceiling. Keep this above the sum of route-class blocking caps plus write/index workers, but do not use it as the primary admission control |
| `HELIX_MAX_READERS` | `2048` | LMDB reader table size for each collection environment. Raise before sustained >1500 concurrent read transactions |
| `HELIX_GRAPH_DEPTH_MAX` | `10` | Max native graph traversal depth accepted by BFS-style routes and shortest-path search. Requests below 1 are raised to 1; larger requests are clamped |
| `HELIX_GRAPH_LIMIT_MAX` | `500` | Max native graph result count accepted by graph query routes, PageRank/community result windows, cycle result count, and subgraph node selection |
| `HELIX_GRAPH_ALGORITHM_ITERATIONS_MAX` | `50` | Max PageRank and label-propagation iterations per request |
| `HELIX_GRAPH_SHORTEST_PATH_VISITED_MAX` | `100000` | Max nodes visited by a shortest-path request before returning no path within budget |
| `HELIX_GRAPH_SYMBOL_RESOLVE_MAX` | `32` | Max pathless symbol matches expanded as graph traversal start nodes. Keep this low in multi-tenant SaaS; high values multiply graph BFS cost |
| `HELIX_GRAPH_ALGORITHM_CACHE_MAX_ENTRIES` | `1024` | In-process materialized PageRank/community cache entries keyed by collection, repo, edge labels, and algorithm parameters. Set `0` to force live computation |
| `HELIX_MAX_OPEN_COLLECTIONS` | `256` | LRU cache capacity for open LMDB environments. When full, 25% of idle collections are batch-evicted (~2 ms estimated reopen cost, depends on working-set heat). Set higher for deployments with many hot collections |
| `HELIX_MAX_MAP_SIZE_GB` | `48` | Per-collection LMDB map size growth cap in GB. The initial map size is 64 MB per collection; it grows by `HELIX_MAP_GROW_MB` on `MapFull` errors up to this cap |
| `HELIX_MAX_DBS` | `65536` | Named LMDB DBI slot ceiling per collection env. Dense-family vector segments allocate several named DBs each; chunked segment cleanup clears DB contents instead of delete-and-close, so churned collections need DBI headroom |
| `HELIX_INITIAL_MAP_MB` | `64` | Initial LMDB map size per new collection |
| `HELIX_MAP_GROW_MB` | `1024` | Fixed LMDB map grow step on `MapFull` and pre-grow. Larger values reduce exclusive resize frequency at the cost of more virtual address space per hot collection |
| `HELIX_REOPEN_MAP_HEADROOM_MB` | `HELIX_MAP_GROW_MB` | Extra map slack reserved when reopening an existing collection so the first post-restart write does not immediately resize |
| `HELIX_UPSERT_HEADROOM_MB` | `8` | Minimum free LMDB headroom reserved before Qdrant point upserts. SaaS uses 1024 MB so large collections grow before late write-time allocation failures |
| `HELIX_UPSERT_APPLY_CHUNK` | `64` | Point upsert ids per write transaction |
| `HELIX_UPSERT_WRITE_BUDGET` | `32768` | Estimated LMDB writes per point-upsert transaction. Applied alongside doc and sparse-posting budgets to split payload-index-heavy, dense-delete-heavy, or sparse-frontier-heavy chunks before they pin the single writer |
| `HELIX_MERGE_FLAT_ROW_CAP` | `512` | Dense flat-merge rows per write transaction, applied in addition to the byte budget. Lower this if `flat_flush` merge chunks still exceed the slow-writer threshold on payload-index-heavy collections |
| `HELIX_DELETE_APPLY_CHUNK` | unset | Explicit point delete ids per chunk. When unset, Helix derives the delete chunk from `HELIX_UPSERT_APPLY_CHUNK`, sqrt dense-segment fanout, and `HELIX_DELETE_APPLY_MIN_CHUNK` |
| `HELIX_DELETE_APPLY_MIN_CHUNK` | `16` | Floor for derived point delete chunk size so high segment fanout does not collapse deletes to one id per write transaction |

### Search and graph performance signals

The dense vector hot path records metrics instead of printing `[PERF]` lines to
stderr. Watch `helix_dense_search_segments`, `helix_dense_segment_search_duration_ms`,
`helix_vector_core_search_stage_ms`, and
`helix_dense_search_segment_fanout_over_target_total` by collection/vector to
identify segment fanout pressure before p99 search latency regresses.
`helix_segment_ids_high_water` should plateau for stable collections; if it keeps
climbing while segment count is bounded, segment-name reuse or reaping is broken.

PageRank and community detection are materialized in-process for normal API
requests and invalidated on successful graph/data writes. Use request
`{"live": true}` or `{"fresh": true}` on those endpoints for canaries and
debugging when you need to bypass `helix_graph_algorithm_cache_*` entries.

The Qdrant-compatible `/collections/{name}/points/scroll` route and the native
`/v1/collections/{name}/points/scan` route share one executor. Watch
`helix_point_scan_requests_total`, `helix_point_scan_duration_ms`,
`helix_point_scan_scanned_points`, `helix_point_scan_returned_points`,
`helix_point_scan_response_bytes`, `helix_point_scan_budget_exhausted_total`,
and `helix_point_scan_generation_mismatch_total` by `endpoint`, `plan`,
`partitioned`, `filtered`, `payload`, and `vectors` labels when comparing the
compatibility path with Helion-native scan before Context Engine integration.

For the longer-term scaling plan, see
[read-write-isolation-plan.md](read-write-isolation-plan.md).

The production EKS manifest is tuned for the current 4 CPU / 32 GiB Helix pod:
write admission is intentionally narrower than read capacity, and index work is
kept to a very small concurrent set.

### Crash forensics toggles

Steady-state production should run with crash forensics off:

- Build stripped release images: leave `HELIX_DEBUG_SYMBOLS` unset or set it to
  `false` when running `deploy/ci-deploy.sh`.
- Keep `HELIX_DIAG_LOG=0`.
- Keep `HELIX_HOT_PATH_METRICS=1`; these are production telemetry for
  write/search/LMDB regressions, not crash-forensic debug.
- Leave `RUST_BACKTRACE`, `MALLOC_CHECK_`, and `LIBC_FATAL_STDERR_` unset on
  the main StatefulSet. `MALLOC_CHECK_=3` is canary-only because it can add
  large allocator overhead on bincode/HNSW-heavy paths.
- Leave node fault-printing sysctls off unless a crash is actively being
  captured.

For a targeted SIGSEGV/SIGBUS capture window:

```bash
# 1. Publish a release image with symbols retained.
NO_CACHE=true HELIX_DEBUG_SYMBOLS=true ./deploy/ci-deploy.sh

# 2. On the node currently running helix-0, enable user-fault logging and
#    give systemd-coredump enough time/space for a 32 GiB process image.
NODE=$(kubectl -n context-engine get pod helix-0 -o jsonpath='{.spec.nodeName}')
kubectl debug -n context-engine "node/$NODE" \
  --image=public.ecr.aws/amazonlinux/amazonlinux:2023 -- chroot /host bash -lc '
    sysctl -w debug.exception-trace=1 kernel.print-fatal-signals=1
    mkdir -p /etc/systemd/coredump.conf.d /etc/systemd/system/systemd-coredump@.service.d
    # NOTE: Compress=yes + a huge ProcessSizeMax makes systemd-coredump xz a
    # multi-GB image and blow past the 5min default RuntimeMaxSec -> the core is
    # DROPPED. Use Compress=no + a bounded ProcessSizeMax so the dump completes.
    # (2026-06-21 incident: the 64G/xz config silently lost every core.)
    cat >/etc/systemd/coredump.conf.d/99-helix-forensics.conf <<EOF
[Coredump]
Storage=external
Compress=no
ProcessSizeMax=8G
ExternalSizeMax=8G
MaxUse=40G
KeepFree=8G
EOF
    systemctl reset-failed 'systemd-coredump@*' 2>/dev/null || true
    cat >/etc/systemd/system/systemd-coredump@.service.d/99-helix-forensics.conf <<EOF
[Service]
RuntimeMaxSec=30min
EOF
    systemctl daemon-reload
  '
```

After the capture window, return the node to steady state. The files can stay
as comments for the next incident, but no active override should remain:

```bash
NODE=$(kubectl -n context-engine get pod helix-0 -o jsonpath='{.spec.nodeName}')
kubectl debug -n context-engine "node/$NODE" \
  --image=public.ecr.aws/amazonlinux/amazonlinux:2023 -- chroot /host bash -lc '
    sysctl -w debug.exception-trace=0 kernel.print-fatal-signals=0
    cat >/etc/systemd/coredump.conf.d/99-helix-forensics.conf <<EOF
# Helix crash forensics disabled for steady state.
# Re-enable during a SIGSEGV/SIGBUS capture window:
# [Coredump]
# Storage=external
# Compress=yes
# ProcessSizeMax=64G
# ExternalSizeMax=64G
# MaxUse=70G
# KeepFree=5G
EOF
    cat >/etc/systemd/system/systemd-coredump@.service.d/99-helix-forensics.conf <<EOF
# Helix crash forensics disabled for steady state.
# Re-enable during capture:
# [Service]
# RuntimeMaxSec=30min
EOF
    systemctl daemon-reload
  '

# Publish a normal stripped release image again.
NO_CACHE=true ./deploy/ci-deploy.sh
```

These node changes are ephemeral for EKS node replacement. Existing core files
are not deleted by the disable step; remove them manually only after preserving
any artifact needed for symbolication.

---

## Docker

The repository includes a multi-stage `Dockerfile` (`rust:1.85-bookworm` builder → `debian:bookworm-slim` runtime).

```bash
# Build
docker build -t helix-db .

# Run with persistent volume
docker run -p 6969:6969 -v helix-data:/data helix-db

# With custom settings
docker run -p 6969:6969 \
  -v helix-data:/data \
  -e HELIX_POOL_SIZE=4096 \
  -e HELIX_MAX_OPEN_COLLECTIONS=512 \
  -e HELIX_LOG=debug \
  helix-db
```

The image runs as a non-root `helix` user with `nofile` limit set to 65536 (each LMDB env uses 2 FDs; at 256+ open collections this headroom is required).

Health check: `GET /health` — the `Dockerfile` configures this as the container health probe with 10 s interval / 3 s timeout.

---

## Kubernetes deployment

Run the LMDB backend as a `StatefulSet` with persistent storage. Size resources,
file-descriptor limits, and collection-cache bounds for your own workload. The
following manifest fragment is an example, not a production prescription:

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1"
  limits:
    memory: "4Gi"
    cpu: "2"

volumeClaimTemplates:
  - metadata:
      name: helix-data
    spec:
      storageClassName: standard
      accessModes: [ReadWriteOnce]
      resources:
        requests:
          storage: 20Gi

env:
  - name: HELIX_DATA_DIR
    value: /data
  - name: HELIX_PORT
    value: "6969"
  - name: HELIX_MAX_OPEN_COLLECTIONS
    value: "32"
  - name: HELIX_POOL_SIZE
    value: "512"
  - name: HELIX_LOG
    value: info

livenessProbe:
  httpGet:
    path: /health
    port: 6969
  initialDelaySeconds: 10
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 6969
  initialDelaySeconds: 5
  periodSeconds: 5
```

Kubernetes-style probe aliases `/healthz`, `/readyz`, and `/livez` are
available for compatibility with common probe configurations.

**Scale notes:**
- Initial LMDB map: 64 MB/collection by default; maps grow on demand in fixed `HELIX_MAP_GROW_MB` increments up to `HELIX_MAX_MAP_SIZE_GB` (default 48 GB).
- LRU evicts 25% of idle envs when cache is full.
- Set `nofile` high enough for the configured collection cache and workload.
- Each collection dir is ~2 FDs (LMDB data + lock); 256 open × 2 = 512 FDs minimum.
- LMDB `max_dbs=65536` by default (configurable with `HELIX_MAX_DBS`; combined with segment-ID reuse in `named_vectors.rs`). `max_readers` defaults to `2048` and can be raised with `HELIX_MAX_READERS`.

For the LSM backend, use the local Compose and Minikube examples under
`deploy/lsm-cloud/`. The provider-neutral ConfigMap template intentionally
contains no account identifiers, bucket names, or credentials.

---

## Configuration

Configuration is loaded from `config.hx.json` (`Config` struct in `helixdb/src/helix_engine/graph_core/config.rs`). If the file is missing or unparseable, `Config::default()` is used.

### Full config shape

```json
{
  "vector_config": {
    "m": 25,
    "ef_construction": 512,
    "ef_search": 128,
    "db_max_size": 10000,
    "flat_scan_threshold": 1024
  },
  "graph_config": {
    "secondary_indices": [],
    "snapshot_interval_secs": 3600,
    "snapshot_keep_last": 3,
    "raft": {
      "enabled": false,
      "node_id": 1,
      "bind_address": "http://127.0.0.1:6969",
      "peers": [
        {"id": 1, "address": "http://127.0.0.1:6969"}
      ],
      "snapshot_entries": 1024,
      "snapshot_catchup_entries": 128,
      "raft_secret": null
    }
  }
}
```

### VectorConfig

| Field | Type | Default | Notes |
|---|---|---|---|
| `m` | `Option<usize>` | 25 | HNSW max bi-directional links per element |
| `ef_construction` | `Option<usize>` | 512 | Dynamic candidate list size for index build |
| `ef_search` | `Option<usize>` | 128 | Dynamic candidate list size for search |
| `db_max_size` | `Option<usize>` | 10000 | LMDB map size in GB |
| `flat_scan_threshold` | `Option<usize>` | 1024 | Keep new collections in bounded flat-scan mode until this many vector points, then build HNSW once. Set `0` to disable |

These feed into `HNSWConfig::new(m, ef_construction, ef_search)` which also derives `m_max_0 = 2*m` and `m_l = 1/ln(m)`.

### GraphConfig

| Field | Type | Default | Notes |
|---|---|---|---|
| `secondary_indices` | `Option<Vec<String>>` | None | Property keys to build secondary indices on |
| `snapshot_interval_secs` | `Option<u64>` | 3600 | Periodic LMDB snapshot interval. Clamped to min 1 |
| `snapshot_keep_last` | `Option<usize>` | 3 | Max snapshots retained per collection. Clamped to min 1 |

### RaftConfig

| Field | Type | Default | Notes |
|---|---|---|---|
| `enabled` | bool | false | Enable Raft consensus replication |
| `node_id` | `Option<u64>` | None | This node's ID. Required when enabled |
| `bind_address` | `Option<String>` | None | This node's externally reachable address |
| `peers` | `Vec<RaftPeerConfig>` | `[]` | All cluster members including self |
| `snapshot_entries` | `Option<u64>` | 1024 | Raft log entries between snapshots. Clamped to min 1 |
| `snapshot_catchup_entries` | `Option<u64>` | 128 | Entries to keep for follower catch-up. Clamped to min 1 |
| `raft_secret` | `Option<String>` | None | Shared secret for `X-Raft-Secret` header. Falls back to `HELIX_RAFT_SECRET` env var |

`RaftPeerConfig`: `{"id": u64, "address": "http://host:port"}`.

---

## Running

Single node:
```bash
HELIX_DATA_DIR=/data/helix HELIX_PORT=6969 ./helix-container
```

The server binds to `0.0.0.0:{HELIX_PORT}`. On startup it:

1. Loads config from `HELIX_CONFIG_PATH` (or default path).
2. Opens the graph engine at `{HELIX_DATA_DIR}/user/`.
3. Creates `CollectionManager` at the same path. Collections live under `{data_dir}/collections/{name}/`.
4. Creates `ReplicationManager`. If `raft.enabled`, spawns a dedicated Raft thread (`raft-node-{id}`).
5. Collects compiled HelixQL handler routes via `inventory`.
6. Registers REST API routes via `register_api_routes()`.
7. Creates `HelixGateway` with a thread pool (default size: 1024 from `GatewayOpts::DEFAULT_POOL_SIZE`).
8. Starts accepting TCP connections with `TCP_NODELAY` enabled.

---

## Raft Cluster Setup

A 3-node cluster example:

**Node 1** (`config.hx.json`):
```json
{
  "vector_config": {"m": 25, "ef_construction": 512, "ef_search": 128, "db_max_size": 10000},
  "graph_config": {
    "raft": {
      "enabled": true,
      "node_id": 1,
      "bind_address": "http://node1:6969",
      "peers": [
        {"id": 1, "address": "http://node1:6969"},
        {"id": 2, "address": "http://node2:6969"},
        {"id": 3, "address": "http://node3:6969"}
      ],
      "raft_secret": "my-cluster-secret"
    }
  }
}
```

Nodes 2 and 3 use identical config but with their own `node_id` and `bind_address`.

Key behaviors:
- The Raft runtime thread runs on a dedicated OS thread (`thread::Builder`), not in the Tokio runtime.
- Writes (`ReplicatedMutation`) are proposed to the leader. If a follower receives a write, it forwards to the leader via `/_raft/propose`.
- Reads on the leader use `linearizable_read()` (ReadIndex-based) for consistency.
- Proposal timeout: 15 seconds (`PROPOSE_TIMEOUT`). Leader wait timeout: 5 seconds (`LEADER_WAIT_TIMEOUT`). Read timeout: 15 seconds.
- Inter-node transport uses `reqwest::blocking::Client` with 5-second timeout over HTTP.
- All `/_raft/message` and `/_raft/propose` requests must include `X-Raft-Secret` header when a secret is configured.
- Check cluster status: `GET /_raft/status` (no auth required).

---

## Snapshots

### Periodic snapshots

The connection handler (`ConnectionHandler::accept_conns`) runs a periodic snapshot timer at `snapshot_interval_secs` (default 3600s). It calls `CollectionManager::snapshot_loaded_collections()` on a blocking Tokio task. Only one snapshot runs at a time (guarded by `AtomicBool`).

### Manual snapshots

Via the Qdrant-compatible API:
- `POST /collections/{name}/snapshots` -- create snapshot
- `GET /collections/{name}/snapshots` -- list snapshots
- `PUT /collections/{name}/snapshots/recover` -- restore (blocked when Raft is enabled, returns 409)

### Snapshot storage layout

Snapshots are stored per-collection at:
```
{data_dir}/collections/{name}/snapshots/{timestamp_millis:020}.mdb
```

Each snapshot has a companion manifest:
```
{data_dir}/collections/{name}/snapshots/{timestamp_millis:020}.json
```

The manifest (`SnapshotManifest`) contains a `SnapshotInfo`:

| Field | Type |
|---|---|
| `name` | string |
| `lsn` | u64 |
| `path` | string |
| `disk_bytes` | u64 |
| `created_at_millis` | i64 |

Old snapshots beyond `snapshot_keep_last` are pruned after each new snapshot.

---

## Write-Ahead Log (WAL)

Source: `helixdb/src/helix_engine/storage_core/wal.rs`

Every mutation is recorded to the WAL before hitting LMDB. On crash, uncommitted entries are replayed.

### Record format

```
[len:u32 LE][payload:bincode][crc32c:u32 LE]
```

CRC-32C (Castagnoli polynomial `0x82F63B78`) integrity check per record.

### Segment files

```
{wal_dir}/{first_lsn:020}.wal
```

Segments rotate at 64 MB (`DEFAULT_SEGMENT_MAX`).

### Sync policies (`SyncPolicy`)

| Policy | Behavior | Default |
|---|---|---|
| `Every` | fsync after every write. Safest, ~1ms overhead | |
| `EveryN(n)` | fsync every N writes | Yes (N=100) |
| `None` | No explicit fsync. OS flushes eventually | |

### WAL operations (`WalOp`)

- `CreateNode`, `UpsertNode`, `DropNode`
- `CreateEdge`, `UpsertEdge`, `DropEdge`
- `InsertVector`
- `TxBegin { tx_id }`, `TxCommit { tx_id }` -- atomic transaction grouping
- `Snapshot { lsn_at_snapshot }` -- marker after LMDB snapshot

### Recovery

`wal::recover(wal_dir, committed_lsn)`:

1. Reads WAL entries from `committed_lsn + 1`.
2. Groups entries by `tx_id`.
3. Only replays transactions with a `TxCommit` marker (incomplete transactions are skipped).
4. Returns `RecoveryReport { replayed, skipped }` and the grouped batches.

### Truncation

`WalWriter::truncate(committed_lsn, keep)` removes old segments whose entries are all below `committed_lsn`, keeping at least `keep` segments for follower catch-up.

---

## Graceful Shutdown

The connection handler listens for both `SIGINT` (Ctrl+C) and `SIGTERM` (k8s pod eviction).

Shutdown sequence (in `ConnectionHandler::accept_conns`):

1. Signal received -- stop accepting new connections.
2. Drop the thread pool sender to drain in-flight requests.
3. Sleep 5 seconds for drain.
4. Wait for any in-progress periodic snapshot to complete.
5. Run a final `snapshot_loaded_collections()` on all loaded collections.
6. Log completion and exit.

---

## Storage Layout

```
{HELIX_DATA_DIR}/user/
  data.mdb              -- global graph engine LMDB
  lock.mdb
  collections/
    {collection_name}/
      data.mdb          -- per-collection LMDB (nodes, edges, vectors, HNSW graph, metadata)
      lock.mdb
      snapshots/
        {millis:020}.mdb
        {millis:020}.json
  aliases.json           -- alias_name -> collection_name mapping
  raft/
    node-{id}/           -- Raft persistent state (when enabled)
```

Each collection LMDB contains these named databases:
- `nodes` -- `Database<U128<BE>, Bytes>`: node data keyed by u128 ID
- `edges` -- `Database<U128<BE>, Bytes>`: edge data keyed by u128 ID
- `out_edges` -- `Database<Bytes, Bytes>`: outgoing edge index
- `in_edges` -- `Database<Bytes, Bytes>`: incoming edge index
- `metadata` -- `Database<Str, Bytes>`: storage metadata under key `"current"`
- `vectors_{name}`, `vector_data_{name}`, `hnsw_out_{name}` -- per named vector index
- `sparse_{name}_postings`, `sparse_{name}_meta` -- per sparse vector index
- `pi_{field}_{kind}` -- payload index databases (kind: `kw`, `i64`, `f64`)

### Metadata Self-Heal

`storage_core.rs` protects against bincode schema drift in the `metadata` database. `try_deserialize_metadata()` is a strict parse that surfaces bincode errors rather than silently reinterpreting bytes. `heal_metadata(&self)` opens its own fresh write txn, serializes `StorageMetadata::default()` into the `"current"` key, and commits — persisting the reset rather than only papering over it in memory.

Two code paths invoke the heal:

- **Open path (`new()`)**: if the existing metadata bytes fail strict deserialization when the environment is first opened, the heal is applied inline in the existing write txn (no extra commit).
- **Read path (`get_metadata()`)**: if deserialization fails on a read, the call atomically gates on a `metadata_corruption_logged: AtomicBool`, invokes `heal_metadata()` to open its own write txn and persist defaults, and logs the event exactly once per collection per process.

Because defaults are persisted, subsequent reads never re-trigger the warning. This keeps a collection accessible across bincode struct changes without losing the environment.

---

## Running Tests

```bash
# Full test suite
cargo test --workspace

# Specific crate
cargo test -p helixdb

# Specific test
cargo test -p helixdb test_name

# With output
cargo test --workspace -- --nocapture
```

Most tests use `tempfile::TempDir` for isolated LMDB environments. Tests that exercise Raft create `ReplicationManager` with `raft.enabled = false` (default) to avoid spawning cluster infrastructure.
