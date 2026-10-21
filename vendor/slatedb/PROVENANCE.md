# Vendored SlateDB (HelixDB fork)

- Source: https://github.com/HelixDB/slatedb (Apache-2.0, see `LICENSE`)
- Pinned rev: `dfe4b4dc38609db8b3fe8dc6f2d3de64bfe63f0e` (also in `PINNED_REV`)
- Upstream lineage: fork of https://github.com/slatedb/slatedb, version 0.14.1
  line, diverged from slatedb main at `6a131a9e` (2026-07-07).
- Vendored crates: `slatedb`, `slatedb-common`, `slatedb-txn-obj` (the fork's
  other workspace members — dst/bencher/cli/bindings/examples — are omitted;
  the root `Cargo.toml` member list is trimmed accordingly).
- Wired into the Helion build via `[patch.crates-io]` in the repo root
  `Cargo.toml`, replacing the crates.io 0.14.1 releases of the same three
  packages.

## Why this fork

Relative to crates.io slatedb 0.14.1 it adds (all format-compatible — no
manifest/SST/WAL byte changes):

- `DbReader` snapshot isolation: `DbReader::snapshot()` returns an
  `Arc<DbSnapshot>` pinned to a reader generation; a reaped checkpoint
  surfaces as a typed `CheckpointLeaseLost` error instead of a silently
  truncated view.
- Batched checkpoint lease management (`manifest/store.rs`
  `delete_checkpoints` / `refresh_checkpoints`): one manifest append per poll
  tick regardless of live snapshot count.
- `multi_get` / `multi_get_with_options` on `Db`, `DbReader`, `DbSnapshot`,
  `DbTransaction` (input order and duplicates preserved).
- `DbTransaction::merge_commutative`: commutative merges skip SSI write-write
  conflicts; the flag is conflict metadata only and is never persisted.
- Typed `DatabaseMissing` reader error; cache usage snapshots; request-scoped
  storage metrics.

## Upstream correctness fixes

- `slatedb/src/reader.rs` and `slatedb/src/db_state.rs`: resolve overlapping
  SST views in stored run order rather than ULID order. This matches point
  reads for updated values, tombstones and merge operands while preserving
  snapshot and projection boundaries. Regressions use real SSTs with reversed
  view IDs. Reference: https://github.com/HelixDB/slatedb/pull/14.
- `slatedb/src/filter.rs`: a Bloom filter with no extracted hashes returns
  false without probing its zero-bit buffer. Queries rejected by the prefix
  extractor remain conservative and keep short stored keys reachable.
  Reference: https://github.com/slatedb/slatedb/pull/1969.

Neither fix changes manifest, SST or WAL encoding. Upload retry timing is unchanged.

## Local modifications

Local patches below are format-compatible; the original cache-lifecycle,
WAL-prefetch and logging patches are marked `HELION LOCAL PATCH` in-source.

- `slatedb/src/cached_object_store/storage_fs.rs`: `impl Drop for
  FsCacheEvictor` aborting the background scan/evict tasks. Upstream's
  RFC-0027 rewrite leaves `background_scan` parked forever on its own
  `reconcile_notify` clone when `scan_interval` is `None`, so every dropped
  `Db`/`DbReader` cache store leaks one task for the life of the process —
  unbounded under Helion's collection-handle LRU churn (guarded by the
  `cache_scan_disabled_reader_close_releases_background_tasks` test).
- `slatedb/src/config.rs` + `slatedb/src/db_reader.rs`:
  `DbReaderOptions::wal_replay_sst_batch_size` (default 4 = upstream's
  hardcoded prefetch window). A reader replays the whole unflushed WAL tail on
  every open and checkpoint refresh; with trickle-write collections holding
  hundreds of tiny WAL SSTs, increasing the prefetch width reduces serial
  object-store waiting. Helion selects 32 by default and caps its override at
  256; the vendored library default remains 4. Guarded by
  `reader_wal_replay_results_are_identical_for_every_prefetch_batch_size`.
- Reader WAL reads deliberately still BYPASS the on-disk object cache
  (`cached_object_store/policy.rs` is unmodified). WAL SST names are
  sequential ids that restart at 1 when a dropped database is re-created at
  the same path, and the cache is keyed by path with no ETag check, so a
  reader-side WAL cache serves the dropped incarnation's WAL for the new one
  (guarded by
  `reader_wal_cache_does_not_serve_a_recreated_databases_reused_wal_ids`).
- `slatedb/src/dispatcher.rs`: `run_lifecycle` logs a task that ends with
  `DatabaseMissing` at `info!` instead of `error!` ("background task panicked
  unexpectedly"). Only `DbReader` raises it (its database was dropped under
  it, the normal reader eviction path); panics and every other error still log
  at `error!`. Logging only, no behaviour change.
- `slatedb/src/db_reader.rs`, `slatedb/src/db_status.rs`, and
  `slatedb/src/error.rs`: missing manifests retire a reader and its checkpoint
  poller with `DatabaseMissing`, preserving a typed close reason instead of
  renewing a deleted database's lease.
- `slatedb-txn-obj/src/lib.rs`, `slatedb/src/manifest/store.rs`, and
  `slatedb/src/db/builder.rs`: optional initial manifest ID floors fence late
  stale appends when a database path is reused. Existing databases and the
  default floor of zero retain their previous behavior. Guarded by
  `test_create_new_db_after_starts_above_floor_and_fences_stale_append`.

## Updating

Fetch the fork, check out the desired rev, re-copy the three crates + root
manifest trim, update `PINNED_REV`, and re-run the format-compatibility
review (manifest FlatBuffers schema, SST/WAL encoding) before shipping.
