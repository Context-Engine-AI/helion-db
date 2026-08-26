<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# helixc

## Purpose

The Helix DSL compiler. Parses `.hx` schema files (driven by `helixdb/src/grammar.pest`), runs semantic analysis with diagnostics, and emits Rust handler source code consumed by `helix-container` and `hbuild`.

## Key Files

| File | Description |
|------|-------------|
| `mod.rs` | Module root re-exporting parser / analyzer / generator |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `parser/` | Pest-based parser, source span tracking (see `parser/AGENTS.md`) |
| `analyzer/` | Type / scope / error analysis with auto-fix suggestions (see `analyzer/AGENTS.md`) |
| `generator/` | Rust code emission for handlers (see `generator/AGENTS.md`) |

## For AI Agents

### Working In This Directory

- Grammar source of truth: `helixdb/src/grammar.pest`. Update `docs/grammar.md` in the same change.
- Generator output must compile under the workspace toolchain — keep emitted code idiomatic.

### Testing Requirements

- `cargo test -p helixdb helixc::*`.
- Generator round-trip tests in `generator/generator_test.rs`.

### Common Patterns

- Pipeline: parser → analyzer → generator. Each stage owns its own error type.

## Dependencies

### Internal

- `protocol` (for emitted code referencing wire types).

### External

- `pest`, `pest_derive`, `proc-macro2` / `quote` for codegen, `serde`.

<!-- MANUAL: -->
