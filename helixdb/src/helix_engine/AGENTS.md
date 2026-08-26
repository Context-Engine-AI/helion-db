<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-06-30 -->

# helix_engine

## Purpose

Core engine for Helion: the backend-agnostic storage layer (`AnyBackend` — LMDB/heed3 for local dev/legacy, SlateDB-on-S3 LSM in production, plus an LSM read-replica arm), the HNSW-based vector index, and the native graph engine. Together these subsystems satisfy queries from the gateway and the DSL-generated handlers.

## Key Files

| File | Description |
|------|-------------|
| `mod.rs` | Module root re-exporting `storage_core`, `vector_core`, `graph_core` |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `storage_core/` | Backend dispatch (`AnyBackend`: LMDB / SlateDB LSM), WAL, replication, collections, payload filters (see `storage_core/AGENTS.md`) |
| `vector_core/` | Dense + sparse vectors, HNSW, mmap segments, SIMD distances (see `vector_core/AGENTS.md`) |
| `graph_core/` | Nodes, edges, traversal iterators and operators (see `graph_core/AGENTS.md`) |

## For AI Agents

### Working In This Directory

- Bincode-fragile structs live under `storage_core` (`StorageMetadata`, `DenseVectorSpaceMetadata`). Do not add or remove fields.
- Segment ID reuse depends on the reaper draining all DBs for the prior physical name — coordinate any change with `vector_core::named_vectors`.
- `HELIX_LSM_ROLE` (writer/reader selection) and `HELIX_MAX_CONCURRENT_BUILDS` (HNSW build cap) are the production-critical envs. `HELIX_MAX_DBS` (default 65536) is an LMDB-only DBI cap and does not apply to the LSM production path.

### Testing Requirements

- `cargo test` in the workspace covers this tree.
- Loom and concurrency tests live in `storage_core/loom_tests.rs` and `vector_core/concurrent_tests.rs`.

### Common Patterns

- Manual write txn + pre-grow on the merge / flush path (PreparedMerge cannot survive retry).
- Iterator-driven traversals — match existing patterns when adding ops.

## Dependencies

### Internal

- Consumed by `helix_gateway` and the generated handlers from `helixc`.

### External

- `slatedb` (production LSM on S3), `heed3` (LMDB, local-dev/legacy), `rayon`, `bincode`, `serde`, `crc32c`.

<!-- MANUAL: -->
