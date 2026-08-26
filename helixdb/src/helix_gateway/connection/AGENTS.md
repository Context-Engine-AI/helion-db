<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# helix_gateway/connection

## Purpose

Connection lifecycle management — accept, hand off to handlers, track per-connection state, close gracefully on shutdown.

## Key Files

| File | Description |
|------|-------------|
| `connection.rs` | Per-connection state and lifecycle |
| `mod.rs` | Module root |

## For AI Agents

### Working In This Directory

- Keep allocation-light — this code is hot under high QPS.
- Coordinate any timeout / keepalive change with the CE client defaults.

### Testing Requirements

- Indirect coverage via load tests in `helix-container/examples/load_test/`.

### Common Patterns

- Connection state held in `Arc` to share between accept loop and handlers.

## Dependencies

### Internal

- `helix_gateway::router`.

### External

- `tokio` net, `tracing`.

<!-- MANUAL: -->
