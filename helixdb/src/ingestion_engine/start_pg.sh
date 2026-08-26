#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PGPORT="${PGPORT:-5432}"

cd "$REPO_ROOT"
docker compose -f docker-compose.postgres-tests.yml up -d --build

echo "Waiting for postgres-test healthcheck..."
until [ "$(docker inspect --format='{{json .State.Health.Status}}' helix-postgres-test 2>/dev/null || echo '\"starting\"')" = "\"healthy\"" ]; do
  sleep 1
done

echo "Postgres test container is ready on localhost:${PGPORT}."
