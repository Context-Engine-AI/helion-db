<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# helixc/analyzer

## Purpose

Semantic analysis for the Helix DSL: type checking, scope resolution, structured diagnostics with codes, pretty-printing, and auto-fix suggestions.

## Key Files

| File | Description |
|------|-------------|
| `analyzer.rs` | Main analyzer pass |
| `types.rs` | Type system definitions |
| `error_codes.rs` | Diagnostic error codes (stable IDs) |
| `pretty.rs` | Pretty-printer for diagnostics |
| `fix.rs` | Auto-fix suggestion generation |
| `mod.rs` | Module root |
| `README.md` | Existing analyzer notes (preserve when editing) |

## For AI Agents

### Working In This Directory

- Read `README.md` first for existing design notes; respect those decisions.
- Stable error codes — never re-use a retired code; downstream tools depend on them.
- Auto-fix suggestions must be safe to apply mechanically (no semantics change).

### Testing Requirements

- `cargo test` covers analyzer paths.
- Add a fixture for any new error code.

### Common Patterns

- Visitor over the AST collecting diagnostics into a sink.

## Dependencies

### Internal

- `helixc::parser` (input AST).

### External

- `serde`, `colored`/`owo-colors` (per impl) for terminal output.

<!-- MANUAL: -->
