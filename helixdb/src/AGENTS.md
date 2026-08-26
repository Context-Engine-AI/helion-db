<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-06-30 -->

# helixdb/src

## Purpose

Library source for the Helion engine. Composes the storage core (backend-agnostic graph + vector — LMDB for local dev/legacy, SlateDB-on-S3 LSM in production), HTTP gateway, Helix DSL compiler, wire protocol, and external ingestion pipelines into a single library consumed by `helix-container`.

## Key Files

| File | Description |
|------|-------------|
| `lib.rs` | Crate root — re-exports public surface |
| `grammar.pest` | Pest grammar for the Helix DSL (consumed by `helixc::parser`) |
| `diag.rs` | Diagnostic / error formatting helpers |
| `telemetry.rs` | Tracing / metrics initialization |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `helix_engine/` | Storage, vector, and graph engines (see `helix_engine/AGENTS.md`) |
| `helix_gateway/` | HTTP gateway, admission control, routing (see `helix_gateway/AGENTS.md`) |
| `helixc/` | Helix DSL compiler — parser, analyzer, generator (see `helixc/AGENTS.md`) |
| `protocol/` | Wire types and serde wrappers (see `protocol/AGENTS.md`) |
| `ingestion_engine/` | Postgres / SQLite ingestion (see `ingestion_engine/AGENTS.md`) |

## For AI Agents

### Working In This Directory

- Assess downstream impact before editing public items in `lib.rs`. The crate is consumed by every workspace binary.
- `grammar.pest` is the source of truth for DSL syntax; document any change in `docs/grammar.md` in the same PR.
- Bincode metadata structs are fragile (no add/remove fields) — see `helix_engine/storage_core` and `protocol`.

### Testing Requirements

- `cargo test -p helixdb`. Many modules embed `*_tests.rs` siblings — keep that pattern.

### Common Patterns

- Storage transactions plumbed through `helix_engine::storage_core` via `AnyBackend` (LMDB/heed3 or SlateDB LSM).
- Iterator-based traversal chains under `helix_engine::graph_core::ops`.

## Dependencies

### Internal

- Used by `helix-container`, `hbuild`, `helix-cli`, and tests in `helix-container/tests/`.

### External

- `slatedb`, `heed3`, `bincode`, `serde`, `pest`, `rayon`, `tokio`, `crc32c`, `tracing`.

<!-- MANUAL: -->
