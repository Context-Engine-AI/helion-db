<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-06-30 -->

# ops/in_

## Purpose

Incoming-edge traversal operators. Given a node, walk the inbound adjacency to reach predecessor nodes or the edges themselves.

## Key Files

| File | Description |
|------|-------------|
| `in_.rs` | Predecessor nodes via inbound edges |
| `in_e.rs` | Inbound edges themselves |
| `to_n.rs` | Target node from a current edge |
| `mod.rs` | Module root |

## For AI Agents

### Working In This Directory

- Adjacency is a one-key-to-many-edges layout: `DUP_SORT | DUP_FIXED` under the legacy LMDB backend, emulated by the production LSM backend as composite `<ns_prefix><key_len><key><value>` keys scanned via `for_each_dup` / `scan_dup_namespace`. Both preserve the SAME dup-sorted iteration order (the `_be` twins mirror the heed `get_duplicates` enumeration byte-for-byte) — honor that ordering invariant regardless of backend when changing iteration order.
- Mirror any change here in `out/` for symmetry.

### Testing Requirements

- Coverage via `graph_core/traversal_tests.rs` and BFS tests in `traversals/`.

### Common Patterns

- Iterator adapters: `InAdapter::in_`, `InEdgesAdapter::in_e`, `ToNAdapter::to_n`.

## Dependencies

### Internal

- `storage_core` adjacency stores.

### External

- Persistence is backend-agnostic via `storage_core::AnyBackend` (SlateDB/LSM in production, LMDB in dev); `heed3` txn types are still referenced directly at some call sites.

<!-- MANUAL: -->
