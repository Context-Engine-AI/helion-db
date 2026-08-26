FROM pgvector/pgvector:pg17

COPY helixdb/src/ingestion_engine/init.sql /docker-entrypoint-initdb.d/00-init.sql

HEALTHCHECK --interval=5s --timeout=3s --retries=20 CMD pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"
