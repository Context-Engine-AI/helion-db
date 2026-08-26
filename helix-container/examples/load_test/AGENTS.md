<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# load_test

## Purpose

Load-test harness for the Qdrant-compatible REST surface. Exercises ingestion + search at scale to validate p95/p99 latency and admission-control behavior.

## Key Files

See directory listing for the current driver (typically `main.rs` or `Cargo.toml` with binary entry).

## For AI Agents

### Working In This Directory

- Don't bake in production endpoints; default to `localhost`.
- Honor `point_mutation_admission` backpressure — clients should respect 429s.

### Testing Requirements

- Manual smoke runs against a local container.

### Common Patterns

- Spawn N async clients, sustain a target QPS, record histograms.

## Dependencies

### Internal

- Calls into `helix-container` over HTTP.

### External

- `tokio`, `reqwest` (or similar), `hdrhistogram`.

<!-- MANUAL: -->
