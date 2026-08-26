<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# rag_demo

## Purpose

Retrieval-augmented generation demo built on Helion: ingests a small corpus, builds vector + graph indices via the Helix DSL, and runs queries from a Rust-flavored Jupyter notebook.

## Key Files

| File | Description |
|------|-------------|
| `rag_rust_demo.ipynb` | Notebook driver — ingest → index → query |
| `README.md` | Setup and run instructions |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `helixdb-cfg/` | Helix DSL schema and config used by the demo |

## For AI Agents

### Working In This Directory

- Keep notebook output committed minimally — strip large blobs before committing.
- DSL schema must compile under the workspace's `helixc`.

### Testing Requirements

- Manual: open and run the notebook.

### Common Patterns

- DSL-first: schema + queries live under `helixdb-cfg/`; notebook drives them.

## Dependencies

### Internal

- `helix-container`, `helixc` for schema compilation.

### External

- Jupyter, `evcxr_jupyter` (Rust kernel), embedding model APIs (per notebook).

<!-- MANUAL: -->
