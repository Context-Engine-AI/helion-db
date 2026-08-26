<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# helix-container/tests

## Purpose

Integration tests for the production server binary.

## Key Files

| File | Description |
|------|-------------|
| `raft_ha.rs` | Raft high-availability test — leader election, failover, replication |

## For AI Agents

### Working In This Directory

- Add a fixture per scenario; do not share global state between tests.
- Raft tests are slow and stateful — keep them deterministic.

### Testing Requirements

- `cargo test -p helix-container --test raft_ha`.

### Common Patterns

- Spawn multiple in-process gateway instances; drive them with HTTP clients.

## Dependencies

### Internal

- `helixdb`, `helix-container` (main bin code reused via `pub` items).

### External

- `tokio`, HTTP client (`reqwest` or framework test client).

<!-- MANUAL: -->
