# Postgres Tests

The PostgreSQL ingestion tests expect a local Postgres instance with the `pgvector` extension enabled.

## Start The Local Test Database

From the repository root:

```bash
docker compose -f docker-compose.postgres-tests.yml up -d --build
```

Or use the helper script:

```bash
./helixdb/src/ingestion_engine/start_pg.sh
```

This starts a local container with these defaults:

- `PGHOST=localhost`
- `PGPORT=5432`
- `PGUSER=postgres`
- `PGPASSWORD=postgres`
- `PGDATABASE=postgres`

The container image enables `pgvector` during initialization.

If `5432` is already in use, override the host port:

```bash
PGPORT=55432 docker compose -f docker-compose.postgres-tests.yml up -d --build
```

Then run tests with the same `PGPORT`:

```bash
PGPORT=55432 cargo test -p helixdb postgres_tests -- --test-threads=1
```

## Run The Postgres Ingestion Tests

```bash
cargo test -p helixdb postgres_tests -- --ignored --test-threads=1
```

The Postgres-backed ingestion tests are ignored by default so the main `cargo test`
suite does not require a local database. Run them explicitly with `--ignored` once
your Postgres test environment is up.

The tests now read the standard Postgres environment variables listed above, so you can point them at a different container or host if needed.

## Stop And Clean Up

```bash
docker compose -f docker-compose.postgres-tests.yml down -v
```
