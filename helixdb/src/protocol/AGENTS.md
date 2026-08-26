<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# protocol

## Purpose

Wire protocol types and serde wrappers. Crosses the gateway/engine boundary and is referenced by generated handler code.

## Key Files

| File | Description |
|------|-------------|
| `request.rs` | Request envelope types |
| `response.rs` | Response envelope types |
| `value.rs` | Generic value type |
| `traversal_value.rs` | Traversal step value type |
| `items.rs` | Node / edge / item types |
| `count.rs` | Count primitive |
| `deterministic_id.rs` | Deterministic ID derivation |
| `filterable.rs` | Filterable trait for payload filters |
| `label_hash.rs` | Label → hash mapping (must stay stable) |
| `remapping.rs` | Field remapping helpers |
| `serdes.rs` | Custom serde wrappers |
| `return_values.rs` | Return-value packaging |
| `mod.rs` | Module root |

## For AI Agents

### Working In This Directory

- Bincode-sensitive types — never add or remove fields on persisted structs without a migration plan.
- `label_hash.rs` defines the on-disk hash. Changing the hash function silently breaks every existing collection.
- Assess downstream impact before editing any public type — generated code references many of these.

### Testing Requirements

- Round-trip serde tests live alongside type definitions.
- End-to-end coverage via gateway integration tests.

### Common Patterns

- One file per wire concept. Keep modules small and focused.

## Dependencies

### Internal

- Consumed by `helix_engine`, `helix_gateway`, generated code from `helixc`.

### External

- `serde`, `bincode`, `serde_json`.

<!-- MANUAL: -->
