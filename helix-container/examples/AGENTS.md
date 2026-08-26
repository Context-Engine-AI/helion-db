<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# helix-container/examples

## Purpose

Example workloads bundled with the production binary. Currently the load-test harness used to exercise the Qdrant-compatible surface under concurrent traffic.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `load_test/` | Load-test harness (see `load_test/AGENTS.md`) |

## For AI Agents

### Working In This Directory

- Examples should be runnable with `cargo run --example <name>` from the workspace root.

### Testing Requirements

- Manual smoke runs against a local `helix-container`.

### Common Patterns

- One subdirectory per scenario.

## Dependencies

### Internal

- `helix-container` running locally.

### External

- `tokio`, HTTP client.

<!-- MANUAL: -->
