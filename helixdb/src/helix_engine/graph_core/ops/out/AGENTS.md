<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-06-30 -->

# ops/out

## Purpose

Outgoing-edge traversal operators. Given a node, walk the outbound adjacency to reach successor nodes or the edges themselves.

## Key Files

| File | Description |
|------|-------------|
| `out.rs` | Successor nodes via outbound edges |
| `out_e.rs` | Outbound edges themselves |
| `from_n.rs` | Source node from a current edge |
| `mod.rs` | Module root |

## For AI Agents

### Working In This Directory

- Mirror any change here in `in_/` for symmetry.
- One-key-to-many-edges adjacency layout — `DUP_SORT | DUP_FIXED` under the legacy LMDB backend, emulated by the production LSM backend via composite-key prefix scans (`for_each_dup`). Preserve the dup-sorted key ordering invariant regardless of backend.

### Testing Requirements

- `graph_core/traversal_tests.rs` exercises both directions.

### Common Patterns

- Iterator adapters: `OutAdapter::out`, `OutEdgesAdapter::out_e`, `FromNAdapter::from_n`.

## Dependencies

### Internal

- `storage_core` adjacency stores.

### External

- Persistence is backend-agnostic via `storage_core::AnyBackend` (SlateDB/LSM in production, LMDB in dev); `heed3` txn types are still referenced directly at some call sites.

<!-- MANUAL: -->
