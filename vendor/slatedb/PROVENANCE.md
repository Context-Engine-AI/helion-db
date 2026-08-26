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

## Local modifications

One (marked `HELION LOCAL PATCH` in-source):

- `slatedb/src/cached_object_store/storage_fs.rs`: `impl Drop for
  FsCacheEvictor` aborting the background scan/evict tasks. Upstream's
  RFC-0027 rewrite leaves `background_scan` parked forever on its own
  `reconcile_notify` clone when `scan_interval` is `None`, so every dropped
  `Db`/`DbReader` cache store leaks one task for the life of the process —
  unbounded under Helion's collection-handle LRU churn (guarded by the
  `cache_scan_disabled_reader_close_releases_background_tasks` test).

## Updating

Fetch the fork, check out the desired rev, re-copy the three crates + root
manifest trim, update `PINNED_REV`, and re-run the format-compatibility
review (manifest FlatBuffers schema, SST/WAL encoding) before shipping.
