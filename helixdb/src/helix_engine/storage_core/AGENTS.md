<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-06-30 -->

# storage_core

## Purpose

Backend-agnostic storage layer: env management, write-ahead log, replication / Raft, collection lifecycle, payload filtering, and persistent metadata. Persistence is dispatched at runtime through the `StorageBackend` trait / `AnyBackend` enum — the SlateDB-backed LSM engine on object storage (S3) is what production runs; the LMDB (`heed3`) backend is legacy / local-dev only. The foundation under both `vector_core` and `graph_core`.

## Key Files

| File | Description |
|------|-------------|
| `storage_core.rs` | Top-level `HelixGraphStorage` facade; owns the storage env and the active `AnyBackend` |
| `backend.rs` | `StorageBackend` trait — pluggable storage seam; visitor-based zero-copy reads, `Namespace` keyspaces, backend chosen at runtime |
| `backend_any.rs` | `AnyBackend` — runtime enum dispatch over `StorageBackend` impls (`Lmdb` / `Lsm` / `LsmReader`); the trait is GAT-based so not object-safe |
| `backend_lmdb.rs` | `LmdbBackend` — `StorageBackend` over LMDB (`heed3`); legacy / dev-only |
| `backend_lsm.rs` | `LsmBackend` — `StorageBackend` over SlateDB (LSM on object storage / S3); production engine. `Namespace` becomes a key prefix in one flat keyspace |
| `backend_lsm_reader.rs` | `LsmReader` — read-only handle over SlateDB's `DbReader` (writer/reader split); eventually-fresh replica, rejects writes |
| `backend_read_ops.rs` | Backend-routed (`_be`) read building blocks routing through `self.backend` |
| `backend_write_node_ops.rs` | Backend-routed node write building blocks (`create_node_be` / `drop_node_be`) |
| `backend_write_edge_ops.rs` | Backend-routed edge write building blocks (create/drop edge incl. adjacency writes) |
| `backend_adj_ops.rs` | Backend-routed adjacency (out/in-edge) DUP reads via `for_each_dup` |
| `replication.rs` | Merge optimizer, segment flush, Raft replication wiring |
| `wal.rs` | Write-ahead log with CRC-32C checksums (`crc32c` crate) |
| `collection_manager.rs` | Collection lifecycle (create/drop, rename, list) |
| `filters.rs` | Payload filter evaluation |
| `metadata.rs` | `StorageMetadata`, `DenseVectorSpaceMetadata` (bincode-fragile) |
| `properties.rs` | Property storage helpers |
| `raft.rs` | Raft node integration |
| `upsert.rs` | Upsert primitives shared by graph + vector paths |
| `storage_methods.rs` | Storage trait surface |
| `mod.rs` | Module root |
| `loom_tests.rs` | Concurrency tests under `loom` |

## For AI Agents

### Working In This Directory

- `flush_prepared_merge` in `replication.rs` MUST use a manual write txn + pre-grow (NOT `with_write_txn`) because `PreparedMerge` is consumed by value and cannot survive transaction retry.
- `max_dbs = 65536` default; configurable via `HELIX_MAX_DBS`. This is a legacy-LMDB-dev-backend concern: each dense segment creates ~5 named DBs per named vector, and DBI pressure stays bounded only when segment names are reused. The production LSM backend maps every `Namespace` to a key prefix in a single flat SlateDB keyspace, so it has no DBI / `max_dbs` limit.
- `metadata.rs` structs (`StorageMetadata`, `DenseVectorSpaceMetadata`) are bincode-serialized. Adding/removing fields breaks deserialization of existing on-disk data — compute derived state at runtime instead.
- `deserialize_metadata` catches bincode errors and resets to defaults (self-healing). Preserve this fallback when refactoring.
- Check storage-symbol impact before editing — many CE flows depend on storage symbols.

### Testing Requirements

- `cargo test` covers most paths.
- `loom_tests.rs` runs under loom for concurrency correctness.
- Postgres ingestion tests in `helixdb/src/ingestion_engine/` exercise the full write path.

### Common Patterns

- (Legacy LMDB backend) Heed3 transactions wrap LMDB cursors; release them before opening new ones to avoid DBI starvation. The LSM backend bridges SlateDB's async API to the synchronous call sites via a dedicated tokio runtime.
- (Legacy LMDB backend) Segment-named DBs are cleaned by clearing in chunks rather than delete-and-close.

## Dependencies

### Internal

- Used by `vector_core` (segment DBs) and `graph_core` (node/edge stores).
- Replication touches `helix_gateway::api::raft`.

### External

- `slatedb` (production LSM backend), `heed3` (legacy LMDB backend), `bincode`, `crc32c`, `serde`, `loom` (dev).

<!-- MANUAL: -->
