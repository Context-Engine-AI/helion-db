<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# traversals

## Purpose

Higher-level graph algorithms layered on top of the iterator-based traversal ops: BFS, cycle detection, and other graph algorithms exposed through the DSL.

## Key Files

| File | Description |
|------|-------------|
| `bfs.rs` | Breadth-first search (forward / reverse / impact variants) |
| `bfs_tests.rs` | BFS correctness tests |
| `algorithms.rs` | PageRank, label propagation, shortest path, Jaccard, etc. |
| `algorithms_tests.rs` | Algorithm correctness tests |
| `cycles.rs` | Cycle detection |
| `mod.rs` | Module root |

## For AI Agents

### Working In This Directory

- Algorithms are designed to be lazy when possible; eager fan-out should be opt-in.
- Check impact before editing — these are entry points for many DSL queries.

### Testing Requirements

- `cargo test` covers algorithm correctness.
- Add tests next to new algorithms in the matching `*_tests.rs` file.

### Common Patterns

- Generation-based visited caches to avoid repeated allocations across queries.

## Dependencies

### Internal

- `graph_core::ops` for the underlying step operators.

### External

- `serde`, standard library collections.

<!-- MANUAL: -->
