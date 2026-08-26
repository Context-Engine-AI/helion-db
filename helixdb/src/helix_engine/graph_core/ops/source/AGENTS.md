<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-06-30 -->

# ops/source

## Purpose

Source operators — the entry points of a traversal. Provide all-nodes / all-edges scans, lookups by ID or type, and creation operators (single + bulk add).

## Key Files

| File | Description |
|------|-------------|
| `n.rs` | All-nodes source |
| `e.rs` | All-edges source |
| `n_from_id.rs` | Node lookup by ID |
| `n_from_types.rs` | Node lookup by label/type |
| `e_from_id.rs` | Edge lookup by ID |
| `e_from_types.rs` | Edge lookup by label/type |
| `add_n.rs` | Add a single node |
| `add_e.rs` | Add a single edge |
| `bulk_add_n.rs` | Bulk add nodes (sorted input required) |
| `bulk_add_e.rs` | Bulk add edges (sorted input required) |
| `k_merge.rs` | K-way merge over multiple sources |
| `mod.rs` | Module root |

## For AI Agents

### Working In This Directory

- Bulk-add paths use `PutFlags::APPEND` and require strictly increasing keys — validate or sort upstream.
- Add operators wrap snapshot/restore semantics for rollback on transaction failure.
- Check impact on `add_n` / `add_e` before editing — write paths cross multiple modules.

### Testing Requirements

- Bulk and single-add paths covered in `graph_core/traversal_tests.rs`.

### Common Patterns

- ID lookups go through `storage_core::storage_methods`.
- `k_merge` enables iterator union over heterogeneous sources.

## Dependencies

### Internal

- `storage_core` for persistence and metadata.

### External

- Persistence is backend-agnostic via `storage_core::AnyBackend` (SlateDB/LSM in production, LMDB in dev); `heed3` txn types are still referenced directly at some call sites.

<!-- MANUAL: -->
