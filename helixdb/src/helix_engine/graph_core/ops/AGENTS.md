<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-06-30 -->

# ops

## Purpose

Step operators that compose into a graph traversal. The Helix DSL compiles to chains of these operators (sources → in/out steps → utility/filter → sinks). Lazy by default; results stream through `TraversalValue`.

## Key Files

| File | Description |
|------|-------------|
| `g.rs` | The `G` builder — entry point for traversal construction |
| `tr_val.rs` | `TraversalValue` enum carrying step results |
| `mod.rs` | Module root |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `source/` | Source / creation operators — `n`, `e`, add, bulk add (see `source/AGENTS.md`) |
| `in_/` | Incoming-edge operators (see `in_/AGENTS.md`) |
| `out/` | Outgoing-edge operators (see `out/AGENTS.md`) |
| `util/` | Utility operators — dedup, drop, filter, map, paths, range, update (see `util/AGENTS.md`) |
| `vectors/` | Vector-aware operators — insert / search bridging into `vector_core` (see `vectors/AGENTS.md`) |

## For AI Agents

### Working In This Directory

- Operators are lazy iterator chains. Match the existing iterator-based pattern when adding ops.
- Bulk operators (`bulk_add_e`, `bulk_add_n`) require sorted input keys for `PutFlags::APPEND` correctness.
- Check impact before editing public symbols — DSL-generated handlers reference them.

### Testing Requirements

- `cargo test` covers operator correctness via traversal tests.

### Common Patterns

- Each operator is a small struct implementing `Iterator<Item = TraversalValue>`.
- Compose via `G::new().n().out().filter(...)` style chains.

## Dependencies

### Internal

- `storage_core` for persistence; `vector_core` for `ops::vectors`.

### External

- Standard iterator combinators. Persistence is backend-agnostic via `storage_core::AnyBackend` (SlateDB/LSM in production, LMDB in dev); `heed3` txn types are still referenced directly at some operator call sites.

<!-- MANUAL: -->
