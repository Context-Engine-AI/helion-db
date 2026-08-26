<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-06-30 -->

# ops/vectors

## Purpose

Vector-aware traversal operators that bridge `graph_core` and `vector_core`. Allow DSL queries to insert vectors against a node and search by similarity.

## Key Files

| File | Description |
|------|-------------|
| `insert.rs` | Insert vector(s) attached to nodes |
| `search.rs` | k-NN / hybrid search returning nodes |
| `mod.rs` | Module root |

## For AI Agents

### Working In This Directory

- Search is intentionally non-lazy at the boundary — the HNSW result set is materialized before downstream steps.
- Insert paths must respect `HELIX_MAX_CONCURRENT_BUILDS`.

### Testing Requirements

- Covered by HNSW tests in `vector_core/hnsw_tests.rs` and graph traversal tests.

### Common Patterns

- `insert_v` / `insert_vs` for single / batch insert; `search_v` for k-NN.

## Dependencies

### Internal

- `vector_core` (HNSW + segment lifecycle).
- `storage_core` (for node attachment).

### External

- `rayon`. Vector + node persistence is backend-agnostic via `vector_core` / `storage_core::AnyBackend` (SlateDB/LSM in production, LMDB in dev); `heed3` txn types are still referenced directly at some call sites.

<!-- MANUAL: -->
