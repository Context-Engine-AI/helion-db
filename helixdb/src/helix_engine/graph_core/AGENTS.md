<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-06-30 -->

# graph_core

## Purpose

Native graph engine — nodes, edges, and the iterator-based traversal DSL that the Helix language compiles to. Wraps `storage_core` for persistence and exposes the `G` builder used by generated handlers.

## Key Files

| File | Description |
|------|-------------|
| `graph_core.rs` | Top-level graph facade |
| `traversal_iter.rs` | Lazy iterator chain types backing the DSL |
| `traversal_tests.rs` | Traversal correctness tests |
| `config.rs` | Graph engine configuration |
| `mod.rs` | Module root |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `traversals/` | Higher-level algorithms — BFS, cycles, shortest path (see `traversals/AGENTS.md`) |
| `ops/` | Step operators that compose into a traversal (see `ops/AGENTS.md`) |

## For AI Agents

### Working In This Directory

- Traversals are lazy iterator chains. Match the existing patterns rather than introducing eager collection.
- Check operator impact before editing — many DSL-generated handlers reference them.
- Don't refactor adjacent unrelated code — surgical changes only.

### Testing Requirements

- `cargo test` covers traversal_tests.rs and the per-algorithm tests under `traversals/`.

### Common Patterns

- `G::new()` builder produces a chain of `ops::*` step operators.
- Traversal values flow through `ops::tr_val::TraversalValue`.

## Dependencies

### Internal

- `storage_core` for node/edge persistence; `vector_core` for `ops::vectors`.

### External

- `serde`, `bincode`. Persistence is backend-agnostic via `storage_core::AnyBackend` (SlateDB/LSM in production, LMDB in dev); `heed3` txn types are still referenced directly at some call sites.

<!-- MANUAL: -->
