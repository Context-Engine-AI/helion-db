<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 -->

# ingestion_engine

## Purpose

External-source ingestion pipelines. Currently supports Postgres (CDC + bulk SQL) and SQLite. Reads rows from a source database and upserts them into Helion via the engine API.

## Key Files

| File | Description |
|------|-------------|
| `postgres_ingestion.rs` | Postgres ingestion (logical / CDC) |
| `postgres_tests.rs` | Postgres integration tests |
| `sql_ingestion.rs` | Generic SQL ingestion path |
| `sqlite_tests.rs` | SQLite integration tests |
| `init.sql` | Postgres test schema bootstrap |
| `start_pg.sh` | Local Postgres test container helper |
| `mod.rs` | Module root |

## For AI Agents

### Working In This Directory

- `docker-compose.postgres-tests.yml` at the repo root spins up the test Postgres for these tests.
- Don't introduce silent retries on ingestion failures — surface errors to the operator.
- Keep `init.sql` and the test fixtures in sync.

### Testing Requirements

- `cargo test -p helixdb ingestion_engine::*`.
- Postgres tests gated on Docker availability — see `docs/postgres-tests.md`.

### Common Patterns

- Streaming row reader → batched upsert via `helix_engine`.

## Dependencies

### Internal

- `helix_engine::storage_core` (writes), `protocol` (value types).

### External

- `tokio-postgres`, `rusqlite`/`sqlx` (per impl).

<!-- MANUAL: -->
