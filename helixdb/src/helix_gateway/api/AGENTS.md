<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-06-30 -->

# helix_gateway/api

## Purpose

REST API endpoint handlers. Implements the Qdrant-compatible surface that Context Engine consumes plus native graph and ops endpoints (health, metrics, Raft, ingest, registration).

## Key Files

| File | Description |
|------|-------------|
| `qdrant.rs` | Qdrant-compatible endpoints — primary CE production seam |
| `collections.rs` | Collection CRUD (Qdrant-shaped) |
| `graph.rs` | Native graph endpoints |
| `ingest.rs` | Bulk ingestion endpoints |
| `health.rs` | Health / readiness probes |
| `metrics.rs` | Prometheus metrics endpoint |
| `raft.rs` | Raft RPC endpoints |
| `register.rs` | Endpoint registration helper |
| `mod.rs` | Module root |

## For AI Agents

### Working In This Directory

- `qdrant.rs` is the production compatibility layer — never change response shapes without CE coordination.
- Review handler impact before editing handlers.
- Health/readiness must remain cheap — no storage writes in those paths.

### Testing Requirements

- `helix-container/tests/raft_ha.rs` for Raft endpoints.
- Downstream CE integration tests cover the Qdrant surface.

### Common Patterns

- Each handler is a small async function returning a JSON response.
- Errors map to consistent HTTP status codes through a shared error type.

## Dependencies

### Internal

- `helix_engine` for execution, `protocol` for wire types, `get_routes` for paths.

### External

- HTTP framework (axum/actix), `serde_json`, `tokio`.

<!-- MANUAL: -->
