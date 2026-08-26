<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# examples

## Purpose

End-to-end usage examples for Helion. Currently includes a RAG demo with a Jupyter notebook driver.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `rag_demo/` | RAG demo with Helix DSL schema and notebook (see `rag_demo/AGENTS.md`) |

## For AI Agents

### Working In This Directory

- Examples must run against the current `helix-container` build — pin or update them when the public API changes.
- Don't reintroduce upstream HelixDB managed-service marketing in example READMEs.

### Testing Requirements

- Smoke: run the notebook against a local `helix-container`.

### Common Patterns

- One subdirectory per scenario; each has its own README + DSL config.

## Dependencies

### Internal

- `helix-container` running locally.

### External

- Jupyter (for `.ipynb` demos).

<!-- MANUAL: -->
