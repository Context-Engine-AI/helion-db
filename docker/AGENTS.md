<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# docker

## Purpose

Docker assets in addition to the root `Dockerfile`. Currently hosts test fixtures for Postgres ingestion.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `test-postgres/` | Postgres test container fixtures (see `test-postgres/AGENTS.md`) |

## For AI Agents

### Working In This Directory

- The production image is built from the repo-root `Dockerfile`. This directory is for auxiliary containers only.
- `docker-compose.postgres-tests.yml` (at repo root) consumes fixtures here.

### Testing Requirements

- Smoke: `docker compose -f docker-compose.postgres-tests.yml up`.

### Common Patterns

- One subdirectory per support service.

## Dependencies

### Internal

- Used by `helixdb/src/ingestion_engine` tests.

### External

- Docker / Docker Compose.

<!-- MANUAL: -->
