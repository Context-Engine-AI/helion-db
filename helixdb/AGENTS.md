<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-06-30 -->

# helixdb

## Purpose

Core library crate for Helion — the unified graph + vector database. Contains the backend-agnostic storage engine (`AnyBackend` — LMDB via `heed3` for local dev/legacy, SlateDB-on-S3 LSM in production, plus an LSM read-replica arm), HNSW vector index, native graph traversals, HTTP/gRPC gateway, Helix DSL compiler (`helixc`), wire protocol, and Postgres/SQLite ingestion pipelines.

## Key Files

| File | Description |
|------|-------------|
| `Cargo.toml` | Library manifest and feature flags |
| `schema.hx` | Sample Helix DSL schema |
| `run.sh` | Local run helper |
| `ingestion.jsonl` | Ingestion fixture data |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `src/` | Library source — engine, gateway, compiler, protocol, ingestion (see `src/AGENTS.md`) |
| `benches/` | Criterion benchmarks for graph and vector operations (see `benches/AGENTS.md`) |

## For AI Agents

### Working In This Directory

- This is the library crate (`helixdb`); binaries live in `helix-container`, `helix-cli`, `hbuild`.
- Bincode-serialized metadata structs (`DenseVectorSpaceMetadata`, `StorageMetadata`) are fragile — never add or remove fields. Compute derived state at runtime.
- Assess downstream impact before editing public symbols. The library has many downstream callers across the workspace.

### Testing Requirements

- Unit and integration tests colocated with source modules (e.g. `*_tests.rs`).
- Run `cargo test` from the repo root.
- HNSW concurrency tests live in `vector_core/concurrent_tests.rs` and `storage_core/loom_tests.rs`.

### Common Patterns

- Storage dispatched through `AnyBackend` (`storage_core::backend_any`): `Lmdb` (heed3, local-dev/legacy), `Lsm` (SlateDB on S3, production), `LsmReader` (read replica). `HELIX_MAX_DBS` (default 65536) is an LMDB-only DBI cap and does not apply to the LSM path.
- Manual write txn + pre-grow in merge optimizer paths (PreparedMerge consumed by value, cannot survive retry).
- Build concurrency capped by `HELIX_MAX_CONCURRENT_BUILDS` to prevent OOM during HNSW construction.

## Dependencies

### Internal

- `get_routes` — route definitions consumed via the gateway.
- Used by `helix-container`, `helix-cli`, `hbuild`.

### External

- `slatedb` (production LSM on S3), `heed3` (LMDB, local-dev/legacy), `bincode`, `serde`, `crc32c`, `rayon` (HNSW build), `tokio` (async gateway), `pest` (DSL parser).

<!-- MANUAL: -->
