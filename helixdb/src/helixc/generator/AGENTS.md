<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# helixc/generator

## Purpose

Emits Rust handler source code from a type-checked Helix AST. Output is consumed by `helix-container` (embedded build) and `hbuild` (standalone CLI).

## Key Files

| File | Description |
|------|-------------|
| `generator.rs` | Main code-emission pass |
| `generator_test.rs` | Round-trip tests (parse → analyze → generate → compile) |
| `example.rs` | Example schema-driven generation |
| `mod.rs` | Module root |

## For AI Agents

### Working In This Directory

- Generated code must compile cleanly with the workspace toolchain — run round-trip tests after any change.
- Keep emitted code idiomatic; prefer explicit types over inference where downstream tooling reads the source.

### Testing Requirements

- `generator_test.rs` runs round-trip cases. Extend it for new constructs.

### Common Patterns

- Build-up of a `String` (or `quote!` `TokenStream`) per handler.

## Dependencies

### Internal

- `helixc::analyzer` (input typed AST), `protocol` (referenced wire types).

### External

- `quote`, `proc-macro2`, `syn` (if used).

<!-- MANUAL: -->
