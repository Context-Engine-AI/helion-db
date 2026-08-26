<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# benches

## Purpose

Criterion benchmarks for the engine's hot paths — graph traversal and vector search.

## Key Files

| File | Description |
|------|-------------|
| `graph_benchmarks.rs` | BFS / shortest path / pagerank benches |
| `vector_benchmarks.rs` | HNSW build / query / sparse / fusion benches |

## For AI Agents

### Working In This Directory

- Run with `cargo bench` from the repo root.
- Don't gate refactors on bench numbers without a controlled baseline run.
- Compare numbers before/after for any storage or vector change in PR notes.

### Testing Requirements

- Benches are not part of `cargo test`. Run on demand.

### Common Patterns

- Criterion `BenchmarkGroup` per scenario; warm up before measuring.

## Dependencies

### Internal

- `helixdb` (full library).

### External

- `criterion`.

<!-- MANUAL: -->
