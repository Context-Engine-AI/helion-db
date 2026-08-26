<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-06-30 -->

# ops/util

## Purpose

Utility operators that filter, transform, or sink traversal values without changing direction (no in/out movement). Includes mutation helpers for in-traversal updates and drops.

## Key Files

| File | Description |
|------|-------------|
| `dedup.rs` | Stream-dedup (preserves order) |
| `drop.rs` | Delete current items |
| `filter_ref.rs` | Read-only predicate filter |
| `filter_mut.rs` | Mutating-context filter |
| `map.rs` | Map step |
| `paths.rs` | Path projection |
| `range.rs` | Range / pagination slicing |
| `secondary_index.rs` | Lookups via secondary indexes |
| `update.rs` | In-traversal updates |
| `mod.rs` | Module root |

## For AI Agents

### Working In This Directory

- `update` and `drop` mutate during iteration — coordinate with snapshot/restore in `source/`.
- Don't mix `filter_ref` and `filter_mut` semantics — they have different transaction kinds.

### Testing Requirements

- `graph_core/traversal_tests.rs`.

### Common Patterns

- Iterator adapters that wrap upstream `TraversalValue` streams.

## Dependencies

### Internal

- `storage_core` for index lookups and mutation helpers.

### External

- Persistence is backend-agnostic via `storage_core::AnyBackend` (SlateDB/LSM in production, LMDB in dev); `heed3` txn types are still referenced directly at some call sites.

<!-- MANUAL: -->
