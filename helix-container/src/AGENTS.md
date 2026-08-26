<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# helix-container/src

## Purpose

Source for the production HTTP server binary. Boots the engine, registers routes, and serves the Qdrant-compatible REST surface.

## Key Files

| File | Description |
|------|-------------|
| `main.rs` | Process entry — env wiring, runtime, gateway boot |
| `queries.rs` | Route handlers (Qdrant-compatible + native graph) |

## For AI Agents

### Working In This Directory

- Routes are declared in `get_routes/src/lib.rs`. Add a route there, implement the handler here.
- Check handler impact before editing — many CE flows depend on response shapes.
- Keep `main.rs` thin — push logic into `helixdb` modules.

### Testing Requirements

- `cargo test -p helix-container`.
- HA / Raft tests in `../tests/raft_ha.rs`.

### Common Patterns

- Async handler functions; shared `AppState` carrying the engine handle.

## Dependencies

### Internal

- `helixdb`, `get_routes`.

### External

- `tokio`, HTTP framework, `serde_json`, `tracing`.

<!-- MANUAL: -->
