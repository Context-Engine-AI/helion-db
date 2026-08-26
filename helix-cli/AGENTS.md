<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# helix-cli

## Purpose

User-facing CLI for managing local Helion instances — start/stop, init schemas, push DSL files, inspect status. Mirrors the upstream HelixDB CLI ergonomics for compatibility with the local-development workflow.

## Key Files

| File | Description |
|------|-------------|
| `Cargo.toml` | Binary manifest |
| `src/main.rs` | CLI entry point and command dispatch |
| `src/args.rs` | Clap command/argument definitions |
| `src/instance_manager.rs` | Local instance lifecycle (start/stop, ports, PIDs) |
| `src/utils.rs` | Path resolution, repo detection, helpers |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `src/` | CLI source (see `src/AGENTS.md`) |

## For AI Agents

### Working In This Directory

- Audit `helix-cli`, `helix-container`, and `hbuild` together when changing install or runtime paths — they share assumptions about the repo layout under `~/.helix/repo/helix-db`.
- Keep compatibility-sensitive names (`helix`, `helixdb`) unless an explicit rename is requested.
- Prefer local/source-based install guidance; the GPL fork does not run a managed installer pipeline.

### Testing Requirements

- `cargo test -p helix-cli`.
- Manual smoke: `cargo run --bin helix -- --help`.

### Common Patterns

- Clap-based subcommands. Keep flags consistent across commands.
- `instance_manager` writes/reads PID files under the user's Helix workdir.

## Dependencies

### Internal

- `helixdb` for engine types when needed.

### External

- `clap`, `serde`, `tokio` (selectively).

<!-- MANUAL: -->
