<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# test-postgres

## Purpose

Postgres container fixtures used by the ingestion-engine integration tests.

## Key Files

See directory listing for current fixtures (init scripts, configuration).

## For AI Agents

### Working In This Directory

- Keep fixtures aligned with `helixdb/src/ingestion_engine/init.sql` — they bootstrap the same schema.
- Tests assume the standard Postgres port; don't change without updating `docker-compose.postgres-tests.yml`.

### Testing Requirements

- Smoke via `docker compose -f docker-compose.postgres-tests.yml up`.

### Common Patterns

- Plain Postgres image with mounted init scripts.

## Dependencies

### Internal

- Consumed by `helixdb/src/ingestion_engine/postgres_tests.rs`.

### External

- `postgres` image.

<!-- MANUAL: -->
