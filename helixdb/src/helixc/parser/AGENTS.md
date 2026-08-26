<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# helixc/parser

## Purpose

Pest-based parser for the Helix DSL. Produces a typed AST with source-span tracking for diagnostics.

## Key Files

| File | Description |
|------|-------------|
| `helix_parser.rs` | Pest grammar bindings and AST construction |
| `parser_methods.rs` | Helper combinators for parser stages |
| `location.rs` | Source span / location tracking for diagnostics |
| `mod.rs` | Module root |

## For AI Agents

### Working In This Directory

- Grammar lives in `helixdb/src/grammar.pest`. Any rule change must be reflected here AND in `docs/grammar.md`.
- Preserve location info on every AST node — analyzer relies on it for error pretty-printing.

### Testing Requirements

- Parser tests in `helix_parser.rs` or sibling test modules.
- End-to-end coverage via `generator/generator_test.rs`.

### Common Patterns

- Pest `Pair<Rule>` → typed AST conversion functions per rule.

## Dependencies

### Internal

- Consumed by `helixc::analyzer`.

### External

- `pest`, `pest_derive`.

<!-- MANUAL: -->
