<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-06-30 -->

# helix_gateway/thread_pool

## Purpose

Worker thread pool used by the gateway for offloading blocking storage / index work off the async runtime.

## Key Files

| File | Description |
|------|-------------|
| `thread_pool.rs` | Pool construction and submit API |
| `mod.rs` | Module root |

## For AI Agents

### Working In This Directory

- Sized via the existing config knob — don't introduce a parallel sizing mechanism.
- Avoid blocking calls inside async handlers; submit through this pool instead.

### Testing Requirements

- Indirect through gateway integration tests.

### Common Patterns

- Submit-and-await pattern bridging blocking storage (SlateDB LSM / LMDB) and HNSW work to async callers (handlers run under `spawn_blocking` + `allow_lsm_blocking`).

## Dependencies

### Internal

- Used by `helix_gateway::api` for blocking work.

### External

- `rayon` or std threads (per implementation), `tokio` oneshot for completion.

<!-- MANUAL: -->
