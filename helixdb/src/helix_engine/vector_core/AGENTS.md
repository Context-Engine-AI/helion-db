<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-06-30 -->

# vector_core

## Purpose

Vector index implementation: dense segment lifecycle, HNSW graph (parallel `rayon` builder), sparse vectors, mmap-backed storage, SIMD distance kernels, and hybrid (dense + sparse) fusion scoring.

## Key Files

| File | Description |
|------|-------------|
| `vector_core.rs` | Top-level facade |
| `named_vectors.rs` | Dense segment lifecycle, segment ID allocation |
| `hnsw.rs` | HNSW index with rayon-parallel builder |
| `hnsw_tests.rs` | HNSW correctness tests |
| `segments.rs` | Segment state machine: Mutable → Building → Indexed |
| `sparse.rs` | Sparse vector storage and search |
| `mmap_vectors.rs` | Memory-mapped vector storage |
| `simd.rs` | SIMD distance kernels |
| `fusion.rs` | Dense + sparse hybrid score fusion |
| `arena_heap.rs` | Arena-backed heap for HNSW search |
| `heap_utils.rs` | Heap helpers shared by search routines |
| `spindle.rs` | Background indexing coordinator |
| `vector.rs` | Vector primitives |
| `mod.rs` | Module root |
| `concurrent_tests.rs` | Concurrency tests |

## For AI Agents

### Working In This Directory

- `HELIX_MAX_CONCURRENT_BUILDS` caps simultaneous HNSW builds to avoid OOM. Don't bypass.
- A segment ID is reusable only after the reaper drains all named DBs for that physical name. Pending cores and dirty orphan DBs are NOT reusable. No metadata migration needed for old collections.
- `DenseVectorSpaceMetadata` (in `storage_core::metadata`) is bincode-fragile — never add/remove fields here.
- Check segment-lifecycle impact before editing any function touching segment lifecycle.

### Testing Requirements

- `cargo test` covers correctness.
- `concurrent_tests.rs` runs under multi-threaded scenarios — keep deterministic.
- HNSW recall benches in `helixdb/benches/vector_benchmarks.rs`.

### Common Patterns

- Rayon-parallel build with bounded concurrency.
- mmap segments for read-only Indexed state; Mutable-state segment storage is backend-dispatched via `storage_core::AnyBackend` (a shared `Arc<AnyBackend>` threaded into every segment core, US-006), so it persists through the active backend — SlateDB/S3 in production, LMDB in dev — not LMDB-only.
- Sparse + dense hybrid via `fusion.rs` (RRF / weighted).

## Dependencies

### Internal

- Sits on top of `storage_core`. Consumed by `graph_core::ops::vectors` and `helix_gateway::api::qdrant`.

### External

- `rayon`, `bincode`, `memmap2`, SIMD intrinsics via `std::arch`. Segment persistence goes through `storage_core::AnyBackend` (SlateDB/LSM in production, LMDB in dev); `heed3` env/txn types are still threaded through directly.

<!-- MANUAL: -->
