# Helion — API Reference

All endpoints return `Content-Type: application/json` unless noted otherwise.

Qdrant-compatible endpoints wrap results in `{"status":"ok","result":...}`.
On error: `{"status":{"error":"..."},"result":null}`.

Native v1 endpoints return plain JSON (e.g. `{"error":"..."}` on failure).

---

## Health

### `GET /health`

Response:
```json
{"status":"ok"}
```

### `GET /ready`

Response:
```json
{
  "status": "ready",
  "collections_on_disk": 5,
  "collections_loaded": 2
}
```

Fields: `collections_on_disk` (int) -- total collection dirs; `collections_loaded` (int) -- currently open in memory.

### `GET /healthz`, `GET /livez`, `GET /readyz`

Kubernetes-style aliases. `/healthz` and `/livez` mirror `/health`; `/readyz` mirrors `/ready`. All return 200 on success and are provided for compatibility with common probe configs.

---

## Raft Internal

All `/_raft/*` endpoints (except status) require a matching `X-Raft-Secret` header when `raft_secret` is configured.

### `GET /_raft/status`

No auth required. Returns `ReplicationStatus`:

```json
{
  "enabled": true,
  "node_id": 1,
  "leader_id": 1,
  "term": 4,
  "is_leader": true,
  "commit_index": 102,
  "applied_index": 102,
  "first_index": 1,
  "last_index": 102,
  "snapshot_index": 0
}
```

All fields are integers except `enabled` / `is_leader` (bool). `node_id` and `leader_id` are `Option<u64>` (may be null).

### `POST /_raft/message`

Body: protobuf-encoded `raft::prelude::Message`. Response: 204 No Content.

### `POST /_raft/propose`

Body: bincode-encoded `ReplicatedMutation`. Response:

```json
{"status":"ok"}
```

---

## Collections (v1 native)

### `POST /v1/collections/create`

Request:
```json
{"name": "my_collection"}
```

Response (201):
```json
{"created": "my_collection"}
```

### `POST /v1/collections/drop`

Request:
```json
{"name": "my_collection"}
```

Response:
```json
{"dropped": "my_collection"}
```

### `POST /v1/collections/list`

Request: empty body or `{}`.

Response:
```json
{"collections": ["coll_a", "coll_b"]}
```

### `POST /v1/collections/stats`

Request:
```json
{"name": "my_collection"}
```

Response (`CollectionStats`):
```json
{
  "name": "my_collection",
  "schema_version": 2,
  "node_count": 10000,
  "edge_count": 25000,
  "vector_count": 10000,
  "disk_bytes": 52428800
}
```

---

## Collections (Qdrant-compatible)

### `PUT /collections/{name}`

Create a collection with dense and/or sparse vector configs.

Request (`CreateCollectionRequest`):
```json
{
  "vectors": {
    "dense": {
      "size": 768,
      "distance": "Cosine",
      "quantization": {
        "mode": "turbo_prod",
        "keep_original": true,
        "rescore": true,
        "oversampling": 16,
        "binary_dims": 256,
        "turbo_dims": 768
      }
    }
  },
  "sparse_vectors": {
    "lex": {
      "modifier": "idf",
      "index": {
        "full_scan_threshold": 5000
      }
    }
  }
}
```

**`vectors` map values** (`VectorParamsInput`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `size` | usize | required | Dimensionality |
| `distance` | string | `"Cosine"` | `"Cosine"`, `"Dot"`, `"Euclid"` / `"Euclidean"` |
| `quantization` | object or null | null (no compression) | See below |

**`quantization`** (`VectorQuantizationInput`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `mode` | string | `"none"` | `"scalar"` / `"int8"`, `"binary"`, `"turbo"` / `"int4"`, `"turbo_prod"` / `"tq"` |
| `keep_original` | bool or null | (see below) | Store original f32 alongside compressed. SpindleConfig default: `true` |
| `rescore` | bool or null | (see below) | Rescore shortlist with originals. SpindleConfig default: `true` |
| `oversampling` | int or null | 16 | Fetch N*oversampling candidates before rescore. Min 1 |
| `binary_dims` | int or null | 256 | Dims for binary sign mode. Min 1 |
| `turbo_dims` | int or null | 768 | Dims for turbo modes. Min 1 |

When `quantization` is omitted entirely, the Qdrant-compatible path sets `SpindleMode::None` with `keep_original: false`.

**`sparse_vectors` map values** (`SparseVectorParamsInput`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `modifier` | string or null | `"none"` | `"idf"` or `"none"` |
| `index.full_scan_threshold` | usize | 5000 | Term threshold before index-based lookup |

Response: `{"status":"ok","result":true}`

### `GET /collections/{name}`

Response:
```json
{
  "status": "ok",
  "result": {
    "status": "green",
    "config": {
      "params": {
        "vectors": {
          "dense": {
            "size": 768,
            "distance": "Cosine",
            "quantization": {
              "mode": "turbo_prod",
              "keep_original": true,
              "rescore": true,
              "oversampling": 16,
              "binary_dims": 256,
              "turbo_dims": 768
            }
          }
        },
        "sparse_vectors": {
          "lex": {
            "modifier": "idf",
            "index": {"full_scan_threshold": 5000}
          }
        }
      }
    },
    "points_count": 10000,
    "vectors_count": 10000
  }
}
```

### `GET /collections`

Response:
```json
{
  "status": "ok",
  "result": {
    "collections": [
      {"name": "coll_a"},
      {"name": "coll_b"}
    ]
  }
}
```

### `PATCH /collections/{name}`

Add new named vector configs to an existing collection, or set the indexing threshold for bulk ingest mode. Existing vector configs cannot be changed.

Request (`UpdateCollectionRequest`):
```json
{
  "vectors": {
    "mini": {"size": 384, "distance": "Cosine"}
  },
  "sparse_vectors": {},
  "optimizer_config": {
    "indexing_threshold": 0
  }
}
```

| Field | Notes |
|---|---|
| `vectors` | New named vector configs to add. Defaults to `{}`. |
| `sparse_vectors` | New sparse vector configs to add. Defaults to `{}`. |
| `optimizer_config.indexing_threshold` | Set to `0` to defer all indexing (bulk ingest mode). Vectors accumulate in the mutable tail without triggering HNSW builds. Set to a positive value to restore normal threshold-triggered indexing. |

**Bulk mode pattern**: `PATCH` with `indexing_threshold=0` before large ingest → upsert points → `PATCH` with normal threshold (or trigger optimizer manually) to build indexes.

Response: `{"status":"ok","result":true}`

### `DELETE /collections/{name}`

Response: `{"status":"ok","result":true}`

---

## Points

### `PUT /collections/{name}/points`

Upsert points with mixed dense and sparse vectors.

Request (`UpsertPointsRequest`):
```json
{
  "points": [
    {
      "id": "abc-123",
      "vector": {
        "dense": [0.1, 0.2, 0.3],
        "lex": {"indices": [10, 42, 99], "values": [0.5, 0.3, 0.1]}
      },
      "payload": {
        "repo": "myrepo",
        "lang": "rust",
        "score": 42
      }
    }
  ]
}
```

**`PointInput`**:

| Field | Type | Notes |
|---|---|---|
| `id` | string or integer | String IDs are tried as hex u128; if that fails, hashed via double-XxHash64 to u128 |
| `vector` | `map<string, dense_or_sparse>` | Dense = `[f32, ...]`; Sparse = `{"indices":[u32...],"values":[f32...]}` |
| `payload` | `map<string, json_value>` | Arbitrary key-value metadata |

**Sparse vector wire format**: an object with `indices` (array of u32) and `values` (array of f32). Indices and values must have equal length, no duplicate indices, all values finite.

Response: `{"status":"ok","result":{"status":"completed"}}`

### `POST /collections/{name}/points/delete`

Delete by ID list or by filter.

Request (`DeletePointsRequest`):
```json
{
  "points": ["abc-123", "def-456"]
}
```

Or by filter:
```json
{
  "filter": {
    "must": [{"key": "repo", "match": {"value": "old-repo"}}]
  }
}
```

Both `points` and `filter` are optional; provide one. Response: `{"status":"ok","result":{"status":"completed"}}`

### `POST /collections/{name}/points`

Retrieve points by ID list.

Request (`GetPointsRequest`):
```json
{
  "ids": ["abc-123", "def-456"],
  "with_payload": true
}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `ids` | array | required | Point IDs (string or integer) |
| `with_payload` | bool | true | Include payload in results |

Response:
```json
{
  "status": "ok",
  "result": [
    {"id": "0000...001a", "payload": {...}, "vector": {}}
  ]
}
```

Points not found are silently skipped.

### `POST /collections/{name}/points/payload`

Merge payload fields into existing points (set_payload).

Request:
```json
{
  "payload": {"status": "indexed", "version": 2},
  "points": ["abc-123", "def-456"]
}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `payload` | `map<string, json_value>` | required | Fields to merge (existing fields preserved) |
| `points` | array or null | null | Point IDs to update |

Response: `{"status":"ok"}`

Note: filter-based set_payload is not yet supported; use `points` IDs.

### `POST /collections/{name}/points/count`

Return point count, optionally filtered.

Request: empty body or `{}` (filter support planned).

Response:
```json
{"result": {"count": 10000}}
```

### `GET /collections/{name}/exists`

Fast existence check for a collection.

Response:
```json
{"result": {"exists": true}}
```

Returns `{"exists": false}` (not 404) when the collection does not exist.

### `POST /collections/{name}/points/scroll`

Qdrant-compatible paginated point scan with optional filter. This route keeps
Qdrant response semantics stable while internally using the shared Helion point
scan executor, so it benefits from scan-specific indexed-filter planning and
observability without requiring client changes.

Request (`ScrollRequest`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `limit` | usize | 10 | Max points per page |
| `offset` | string (hex) or integer or null | null (start) | Pagination cursor |
| `filter` | Filter or null | null | See Filters section |
| `with_payload` | bool, string array, or `{include, exclude}` | true | Qdrant-compatible payload projection |
| `with_vectors` / `with_vector` | bool or string array | false | Hydrate all or selected dense vectors |

Response:
```json
{
  "status": "ok",
  "result": {
    "points": [
      {"id": "0000...001a", "payload": {...}}
    ],
    "next_page_offset": "0000...001b"
  }
}
```

`next_page_offset` is null only when there are no more results. Filtered
Qdrant-compatible scroll calls are bounded by
`HELIX_QDRANT_SCROLL_SCAN_BUDGET_MIN`,
`HELIX_QDRANT_SCROLL_SCAN_BUDGET_MULTIPLIER`,
`HELIX_QDRANT_SCROLL_SCAN_BUDGET_MAX`, and
`HELIX_QDRANT_SCROLL_SCAN_BUDGET_MS`; when a filtered page uses its budget
before the collection is exhausted, Helion returns the next offset so clients
can continue without one unindexed filter monopolizing the shard.

### `POST /v1/collections/{name}/points/scan`

Helion-native point scan/export endpoint. This is an opt-in superset of
Qdrant-compatible scroll for bulk backfills, diagnostics, and future Context
Engine integration. It shares the same executor as `/points/scroll` but exposes
partitioning, cursor generation drift, budgeted scans, and plan metadata.

Request:

| Field | Type | Default | Notes |
|---|---|---|---|
| `limit` | usize | 10 | Max points returned in this page |
| `offset` | string (hex) or integer or null | null | Initial point-id cursor |
| `cursor` | string or null | null | Resume token returned as `next_cursor` |
| `filter` | Filter or null | null | Same Qdrant-compatible filter structure |
| `with_payload` | bool, string array, or `{include, exclude}` | true | Payload projection |
| `with_vectors` / `with_vector` | bool or string array | false | Hydrate all or selected dense vectors |
| `partition` | `{index,total}` or null | null | Parallel scan partition by point-id range |
| `max_scan` | usize or null | null | Max points to inspect before returning a resume cursor |
| `require_consistent` | bool | false | Return 409 if the cursor's collection generation changed |

Response:
```json
{
  "status": "ok",
  "result": {
    "points": [
      {"id": "8000...0001", "payload": {"repo": "context-engine"}}
    ],
    "next_page_offset": "8000...0002",
    "next_cursor": "hscan1.eyJvZmZzZXQiOi...",
    "scan": {
      "plan": "indexed_candidates",
      "scanned": 32,
      "returned": 10,
      "index_candidates": 214,
      "budget_exhausted": false,
      "limit": 10,
      "max_scan": 10000,
      "partition": {"index": 1, "total": 8},
      "generation": 1234,
      "consistent": true
    }
  }
}
```

Cursor tokens are stateless and validate that the filter and partition are the
same on resume. They are not a long-lived server lease. `generation` is the LMDB
environment transaction generation observed by the scan; use it to compare scan
stability before routing production backfills to the native endpoint.

The native path is bounded by `HELIX_SCAN_PARTITION_MAX`,
`HELIX_SCAN_BUDGET_MAX`, and `HELIX_SCAN_INDEX_CANDIDATE_MAX`. The
Qdrant-compatible scroll path additionally applies the filtered-scroll
budgets described above while preserving the Qdrant response shape.

---

## Search

### `POST /collections/{name}/points/search`

Single-vector dense ANN search.

Request (`SearchRequest`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `vector` | see below | required | Query vector |
| `limit` | usize | 10 | Max results |
| `filter` | Filter or null | null | Applied during HNSW traversal (not post-filter) |
| `with_payload` | bool | true | Include payload |
| `with_vectors` | bool | false | (reserved) |

**`vector`** accepts two shapes (`SearchVector`, serde untagged):
- Named: `{"name": "dense", "vector": [0.1, 0.2, ...]}`
- Raw: `[0.1, 0.2, ...]` (defaults to vector name `"dense"`)

Response:
```json
{
  "status": "ok",
  "result": [
    {"id": "0000...0042", "version": 0, "score": 0.95, "payload": {...}},
    ...
  ]
}
```

When filtering, the engine oversamples by 4x internally and truncates after.

---

## Query (Prefetch + Fusion)

### `POST /collections/{name}/points/query`

Supports three modes depending on the `query` field shape.

Request (`QueryRequest`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `prefetch` | array of `PrefetchQuery` | `[]` | Prefetch stages for fusion |
| `query` | json value or null | null | See three modes below |
| `using` | string or null | null | Named vector for single-vector modes (defaults to `"dense"`) |
| `limit` | usize | 10 | Max results |
| `with_payload` | bool | true | Include payload |
| `filter` | Filter or null | null | Applied during traversal |

**Mode 1 -- Dense single-vector**: `query` is a JSON array of floats.

```json
{
  "query": [0.1, 0.2, 0.3],
  "using": "dense",
  "limit": 10
}
```

**Mode 2 -- Sparse single-vector**: `query` is an object with `indices` and `values`.

```json
{
  "query": {"indices": [10, 42], "values": [0.5, 0.3]},
  "using": "lex",
  "limit": 10
}
```

`using` is required for sparse queries.

**Mode 3 -- Prefetch + RRF fusion**: `query` is `{"fusion": "rrf"}` or omitted (defaults to RRF if `prefetch` is non-empty).

```json
{
  "prefetch": [
    {"query": [0.1, 0.2, ...], "using": "dense", "limit": 100},
    {"query": [0.3, 0.4, ...], "using": "mini", "limit": 100}
  ],
  "query": {"fusion": "rrf"},
  "limit": 10
}
```

**`PrefetchQuery`**:

| Field | Type | Default |
|---|---|---|
| `query` | `[f32, ...]` | required |
| `using` | string | `"dense"` |
| `limit` | usize | 100 |

RRF uses k=60 (Cormack et al. 2009). Score = sum(1/(60+rank)) across lists.

Response (all three modes):
```json
{
  "status": "ok",
  "result": [
    {"id": "...", "version": 0, "score": 0.032, "payload": {...}},
    ...
  ]
}
```

---

## Facet

### `POST /collections/{name}/facet` and `POST /collections/{name}/facet/count`

Aggregate counts of distinct values for a given payload key, optionally constrained by a filter.

Request:
```json
{
  "key": "lang",
  "filter": {"must": [{"key": "repo", "match": {"value": "myrepo"}}]},
  "limit": 10,
  "exact": true
}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `key` | string | required | Payload key to aggregate |
| `filter` | Filter or null | null | Qdrant-style filter applied before counting |
| `limit` | usize | 10 | Max distinct values returned |
| `exact` | bool | true | Hint flag; the implementation returns exact counts |

Response:
```json
{
  "status": "ok",
  "result": {
    "hits": [
      {"value": "rust", "count": 42},
      {"value": "python", "count": 10}
    ]
  }
}
```

Notes:
- O(N) full scan over matching points; when a `filter` is present, scans `indexed_filter_candidates` instead of the full collection.
- Array-valued payload keys count each element separately (Qdrant semantics).
- Hits sort by `count` descending, then by `value` for determinism.
- `limit` truncates the sorted list.
- `/facet/count` is the same endpoint under a second route alias.

---

## Payload Index

### `PUT /collections/{name}/index`

Create a payload field index for accelerated filtering.

Request (`CreateIndexRequest`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `field_name` | string | required | Payload key to index |
| `field_schema` | string | `"keyword"` | `"keyword"`, `"integer"` / `"int"`, `"float"` / `"double"` |

Response: `{"status":"ok","result":{"status":"acknowledged"}}`

---

## Snapshots

### `GET /collections/{name}/snapshots`

List existing snapshots.

Response:
```json
{
  "status": "ok",
  "result": [
    {
      "name": "00000001713000000000.mdb",
      "creation_time": 1713000000000,
      "size": 52428800,
      "lsn": 42,
      "location": "00000001713000000000.mdb"
    }
  ]
}
```

### `POST /collections/{name}/snapshots`

Create a snapshot of the collection (LMDB `copy_to_file`).

Response: same shape as a single snapshot entry above.

### `PUT /collections/{name}/snapshots/recover`

Restore from a named snapshot. Blocked when Raft replication is enabled (returns 409).

Request (`RecoverSnapshotRequest`):
```json
{"location": "00000001713000000000.mdb"}
```

`name` is accepted as an alias for `location`.

Response:
```json
{
  "status": "ok",
  "result": {
    "status": "completed",
    "snapshot": {...}
  }
}
```

---

## Aliases

### `POST /collections/aliases`

Batch create/delete aliases. Actions apply in order; no cross-action transaction.

Request (`AliasActionBatch`):
```json
{
  "actions": [
    {"create_alias": {"collection_name": "repo_v2", "alias_name": "repo"}},
    {"delete_alias": {"alias_name": "old_alias"}}
  ]
}
```

Response: `{"status":"ok","result":true}`

### `GET /aliases`

Response:
```json
{
  "status": "ok",
  "result": {
    "aliases": [
      {"alias_name": "repo", "collection_name": "repo_v2"}
    ]
  }
}
```

### `GET /collections/{name}/aliases`

List aliases pointing to a specific collection.

Response: same shape as `GET /aliases` but filtered.

---

## Ingest (v1 native)

### `POST /v1/ingest/nodes`

Request (`IngestNodesRequest`):
```json
{
  "collection": "repo",
  "nodes": [
    {
      "label": "Symbol",
      "name": "main",
      "path": "src/main.rs",
      "properties": {"lang": "rust"}
    }
  ]
}
```

**`NodeInput`**:

| Field | Type | Default | Notes |
|---|---|---|---|
| `label` | string | required | Node label (e.g. `"Symbol"`, `"File"`) |
| `name` | string | required | Symbol/file name |
| `path` | string | required | File path |
| `properties` | `map<string, Value>` | `{}` | Extra metadata |

Node ID is deterministic: `deterministic_id::node_id(collection, label, name, path)`.

Response: `{"upserted": 2}`

### `POST /v1/ingest/edges`

Request (`IngestEdgesRequest`):
```json
{
  "collection": "repo",
  "edges": [
    {
      "edge_type": "CALLS",
      "from_name": "main",
      "to_name": "helper",
      "from_path": "src/main.rs",
      "to_path": "src/lib.rs",
      "from_label": "Symbol",
      "to_label": "Symbol",
      "git_branches": ["main", "feature/auth"],
      "properties": {}
    }
  ]
}
```

**`EdgeInput`**:

| Field | Type | Default | Notes |
|---|---|---|---|
| `edge_type` | string | required | e.g. `"CALLS"`, `"IMPORTS"`, `"INHERITS_FROM"` |
| `from_name` | string | required | Source symbol name |
| `to_name` | string | required | Target symbol name |
| `from_path` | string | required | Source file path |
| `to_path` | string | required | Target file path |
| `from_label` | string | `"Symbol"` | Label of source node |
| `to_label` | string | `"Symbol"` | Label of target node |
| `git_branches` | `[string]` | `[]` | Branch provenance |
| `properties` | `map<string, Value>` | `{}` | Extra metadata |

Response: `{"upserted": 1}`

### `POST /v1/collections/{name}/ingest/stream`

NDJSON streaming ingest. Each line is a tagged record:

```
{"type":"node","label":"Symbol","name":"main","path":"src/main.rs"}
{"type":"edge","edge_type":"CALLS","from_name":"main","to_name":"helper","from_path":"src/main.rs","to_path":"src/lib.rs"}
```

Records are batched in groups of 1000 (`STREAM_BATCH_SIZE`). Also works as a raw HTTP socket with `Content-Length` header via `handle_ingest_stream_socket` (30s per-line read timeout).

Response:
```json
{
  "processed": 3,
  "nodes_upserted": 2,
  "edges_upserted": 1,
  "batches": 1
}
```

---

## Graph Queries (v1 native)

All graph endpoints accept the same request body (`GraphQuery`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `collection` | string | required | Collection name |
| `symbol` | string | required | Symbol name to resolve |
| `path` | string or null | null | File path filter (used in ID resolution) |
| `depth` | usize | 1 | BFS traversal depth |
| `limit` | usize | 100 | Max results |

Traversal depth is clamped to `HELIX_GRAPH_DEPTH_MAX` (default `10`) and result
limits are clamped to `HELIX_GRAPH_LIMIT_MAX` (default `500`). PageRank and
community iterations are clamped to `HELIX_GRAPH_ALGORITHM_ITERATIONS_MAX`
(default `50`). Shortest-path search also stops at
`HELIX_GRAPH_SHORTEST_PATH_VISITED_MAX` visited nodes (default `100000`).
PageRank and community responses are normally served from an in-process
materialized cache keyed by collection, repo, edge labels, and algorithm
parameters; writes invalidate the collection generation. Send `live: true` (or
`fresh: true`) to bypass the cache for debugging.

Symbol is resolved to a node ID via `deterministic_id::node_id(collection, "Symbol", symbol, path)`.

Response shape (all except `/cycles` and `/definition`):
```json
{
  "results": [
    {
      "id": "0000...00ab",
      "label": "Symbol",
      "properties": {"name": "foo", "path": "src/lib.rs"},
      "depth": 1
    }
  ]
}
```

### `POST /v1/graph/callers`
BFS reverse over `CALLS` edges, depth=1. Returns direct callers.

### `POST /v1/graph/callees`
BFS forward over `CALLS` edges, depth=1. Returns direct callees.

### `POST /v1/graph/importers`
BFS reverse over `IMPORTS` edges, depth=1.

### `POST /v1/graph/definition`
Returns a single node by symbol ID. Response: `{"result": {...}}` (no `depth` field). Returns 404 if not found.

### `POST /v1/graph/transitive_callers`
BFS reverse over `CALLS`, uses `depth` parameter.

### `POST /v1/graph/transitive_callees`
BFS forward over `CALLS`, uses `depth` parameter.

### `POST /v1/graph/impact`
BFS impact analysis (forward + reverse). Uses `depth` and `limit`.

### `POST /v1/graph/dependencies`
BFS forward over both `CALLS` and `IMPORTS` edges.

### `POST /v1/graph/cycles`
Cycle detection starting from symbol over `CALLS` edges.

Response:
```json
{
  "cycles": [
    ["0000...001a", "0000...002b", "0000...001a"]
  ]
}
```

### `POST /v1/graph/subclasses`
BFS reverse over `INHERITS_FROM` edges.

### `POST /v1/graph/base_classes`
BFS forward over `INHERITS_FROM` edges.

### `POST /v1/graph/pagerank`

Iterative power-method PageRank over the graph.

Request:
```json
{"collection": "repo", "edge_labels": ["CALLS", "IMPORTS"], "iterations": 20, "limit": 50, "repo": "my/repo", "live": false}
```

Response: `{"results": [{"id": "...", "label": "Symbol", "rank": 0.042, "properties": {...}}, ...]}` ordered by descending PageRank score.

### `POST /v1/graph/communities`

Label-propagation community detection.

Request:
```json
{"collection": "repo", "edge_labels": ["CALLS", "IMPORTS"], "max_iterations": 50, "limit": 50, "repo": "my/repo", "live": false}
```

Response: `{"communities": [{"community_id": "...", "size": 12, "members": ["..."]}], "total": 1}`.

### `POST /v1/graph/jaccard`

Jaccard similarity between two nodes' neighbor sets.

Request:
```json
{"collection": "repo", "symbol_a": "foo", "path_a": "src/a.rs", "symbol_b": "bar", "path_b": "src/b.rs"}
```

Response: `{"similarity": 0.333}`

### `POST /v1/graph/shortest_path`

BFS shortest path between two nodes.

Request:
```json
{"collection": "repo", "from_symbol": "main", "from_path": "src/main.rs", "to_symbol": "helper", "to_path": "src/lib.rs", "depth": 10}
```

Response: `{"path": [{"id": "0000...001a", "label": "Symbol", "properties": {...}}], "distance": 2}`. Empty array if no path is found within the configured depth/visited-node budgets.

### `POST /v1/graph/subgraph`

Extract a subgraph for visualization. If `node_ids` is empty, returns top-N nodes by PageRank plus their interconnecting edges.

Request:
```json
{
  "collection": "repo",
  "node_ids": ["0000...001a", "0000...002b"],
  "edge_labels": ["CALLS", "IMPORTS"],
  "limit": 50
}
```

Response:
```json
{
  "nodes": [
    {"id": "...", "name": "main", "path": "src/main.rs", "symbol_type": "function", "importance": 0.042}
  ],
  "edges": [
    {"from": "...", "to": "...", "label": "CALLS"}
  ]
}
```

### `POST /v1/graph/delete_by_path`

Delete all nodes (and their edges) whose `path` property matches the given value.

Request:
```json
{"collection": "repo", "path": "src/old_module.rs"}
```

Response: `{"deleted": 12}`

---

## Filters

Qdrant-compatible filter structure, used in search/scroll/delete/query endpoints.

```json
{
  "must": [...],
  "must_not": [...],
  "should": [...]
}
```

All three arrays default to `[]`. Semantics:
- `must`: all conditions must match
- `must_not`: none may match
- `should`: at least one must match (if non-empty)

**`FieldCondition`**:
```json
{"key": "repo", "match": {"value": "myrepo"}}
{"key": "repo", "match": {"any": ["repo_a", "repo_b"]}}
{"key": "content", "match": {"text": "search term"}}
{"key": "score", "range": {"gte": 10.0, "lt": 100.0}}
```

| Condition type | Field | Notes |
|---|---|---|
| `MatchValue` | `match.value` | Exact match (any JSON value) |
| `MatchAny` | `match.any` | Match any value in array |
| `MatchText` | `match.text` | Substring/text match |
| `RangeCondition` | `range.gte`, `range.gt`, `range.lte`, `range.lt` | All optional f64 |

Indexed fields (`PayloadIndexSchema::Keyword`, `Integer`, `Float`) are resolved via payload index before HNSW traversal, avoiding full scans.
