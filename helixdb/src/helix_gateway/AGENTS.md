<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# helix_gateway

## Purpose

HTTP/gRPC gateway and request admission control. Hosts the Qdrant-compatible REST surface, the native graph endpoints, Raft RPC, metrics, and the worker thread pool. The CE consumer reaches Helion through this layer.

## Key Files

| File | Description |
|------|-------------|
| `gateway.rs` | Synchronous gateway entry |
| `async_gateway.rs` | Async (tokio) gateway entry |
| `inflight.rs` | In-flight request tracking |
| `point_mutation_admission.rs` | Admission control for point mutations |
| `gateway.png` | Architecture diagram |
| `mod.rs` | Module root |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `api/` | REST handlers — collections, graph, ingest, qdrant, raft, metrics, health (see `api/AGENTS.md`) |
| `router/` | Request routing layer (see `router/AGENTS.md`) |
| `connection/` | Connection lifecycle (see `connection/AGENTS.md`) |
| `thread_pool/` | Worker thread pool (see `thread_pool/AGENTS.md`) |

## For AI Agents

### Working In This Directory

- Production: `QDRANT_URL=http://helix:6333`, `HELIX_GRAPH=1`. Don't change the listening port without coordinating with CE.
- `point_mutation_admission` enforces backpressure on writes; respect its limits when adding new write paths.
- Check handler impact before editing — many CE flows depend on response shapes.

### Testing Requirements

- Integration coverage from `helix-container/tests/` and downstream CE smoke tests.

### Common Patterns

- Async-first via `tokio`; synchronous entry retained for internal harnesses.
- Routes are registered through `get_routes` for consistency between binary and tests.

## Dependencies

### Internal

- `helix_engine` (storage / vector / graph), `protocol`, `get_routes`.

### External

- `tokio`, HTTP framework (axum/actix), `serde_json`, `tracing`.

<!-- MANUAL: -->
