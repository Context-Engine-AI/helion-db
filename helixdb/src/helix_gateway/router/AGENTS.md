<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# helix_gateway/router

## Purpose

Request routing layer: maps incoming HTTP paths/methods to handlers registered via `get_routes` and the `api/` modules.

## Key Files

| File | Description |
|------|-------------|
| `router.rs` | Router construction and method dispatch |
| `mod.rs` | Module root |

## For AI Agents

### Working In This Directory

- Routes are owned by `get_routes/src/lib.rs` — the router only wires them up.
- Avoid adding bespoke per-handler middleware here; prefer composing in the gateway entry.

### Testing Requirements

- Indirect coverage through `helix-container/tests/`.

### Common Patterns

- Single router builder consumed by both sync and async gateway entries.

## Dependencies

### Internal

- `get_routes`, `helix_gateway::api`.

### External

- HTTP framework router.

<!-- MANUAL: -->
