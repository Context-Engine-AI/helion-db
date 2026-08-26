<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# docs

## Purpose

User and operator documentation for Helion: API reference, architecture, grammar, operations, postgres tests, and active design specs (sharding, raft scale-up, read/write isolation).

## Key Files

| File | Description |
|------|-------------|
| `api-reference.md` | REST API surface (Qdrant-compatible + native) |
| `architecture.md` | High-level architecture overview |
| `grammar.md` | Helix DSL grammar documentation |
| `operations.md` | Operator runbooks (deploy, scale, recover) |
| `postgres-tests.md` | Postgres ingestion test setup |
| `audit-2026.md` | 2026 audit notes |
| `LAYER_2_SHARDING_SPEC.md` | Sharding design spec |
| `raft-scaleup-plan.md` | Raft scale-up plan |
| `read-write-isolation-plan.md` | Read/write isolation plan |

## For AI Agents

### Working In This Directory

- Update relevant doc in the same change as a behavior or workflow change (per repo `AGENTS.md`).
- Keep `grammar.md` aligned with `helixdb/src/grammar.pest`.
- Active specs (sharding, raft, isolation) are work-in-progress — preserve TODO markers when editing.

### Testing Requirements

- No automated checks. Lint manually for broken cross-links.

### Common Patterns

- Markdown with front-matter when needed; cross-links to sibling docs.

## Dependencies

### Internal

- Reflects behavior of `helixdb`, `helix-container`, `helix-cli`.

### External

- None.

<!-- MANUAL: -->
