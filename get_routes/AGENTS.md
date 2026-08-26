<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# get_routes

## Purpose

Tiny support crate that owns the route table consumed by `helix-container` and the `helixdb` gateway. Centralizing routes here keeps the binary, the gateway, and the DSL-generated handlers in agreement on the wire surface.

## Key Files

| File | Description |
|------|-------------|
| `Cargo.toml` | Crate manifest |
| `src/lib.rs` | Route definitions and registration helper |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `src/` | Library source (see `src/AGENTS.md`) |

## For AI Agents

### Working In This Directory

- Adding a route here requires a corresponding handler in `helix-container/src/queries.rs` or `helixdb::helix_gateway::api`.
- Keep Qdrant-compatible paths stable — Context Engine pins them.

### Testing Requirements

- Coverage comes from `helix-container` integration tests and downstream CE smoke tests.

### Common Patterns

- Each route is declared once and reused by the binary and the test harness.

## Dependencies

### Internal

- Consumed by `helix-container` and indirectly by `helixdb::helix_gateway`.

### External

- Minimal — typically `serde` and the HTTP framework re-exports.

<!-- MANUAL: -->
