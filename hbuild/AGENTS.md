<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# hbuild

## Purpose

Build helper binary that compiles a Helix DSL schema (`.hx`) into Rust handler source for embedding in `helix-container`. Sits between `helixc` (the in-process compiler) and the binary build, used during local development and CI.

## Key Files

| File | Description |
|------|-------------|
| `Cargo.toml` | Binary manifest |
| `src/main.rs` | CLI entry — reads `.hx`, drives `helixc`, writes generated Rust |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `src/` | Binary source (see `src/AGENTS.md`) |

## For AI Agents

### Working In This Directory

- Audit `helix-cli`, `helix-container`, and `hbuild` together when changing repo-layout or runtime-path assumptions.
- Generated output must compile cleanly under the workspace's pinned toolchain — do not silently change codegen format.

### Testing Requirements

- Generator behavior is exercised by `helixdb/src/helixc/generator/generator_test.rs`.
- Smoke: `cargo run --bin hbuild -- <path-to-schema>`.

### Common Patterns

- Pure CLI wrapper around the `helixdb::helixc` API surface.

## Dependencies

### Internal

- `helixdb` (`helixc::parser`, `helixc::analyzer`, `helixc::generator`).

### External

- `clap` (if used), `anyhow`/`thiserror`.

<!-- MANUAL: -->
