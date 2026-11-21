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
- `slatedb/src/db_reader.rs`: a reader's WAL replay reads its SSTs with
  `cache_blocks: false` (it was `true`), as the writer's replay in `db.rs`
  and the compactor do. A replay reads every block of a WAL SST once, and the
  block cache keys WAL entries by that `DbReader`'s own scope, so no later
  reader finds them. In Helion's one process-wide block cache they only pushed
  out the blocks reads use: about 3 MB of resident memory for every reopen of
  a collection with a 1.8 MB tail, until the cache was full. Guarded by
  helixdb's `reader_wal_replay_leaves_the_block_cache_alone`.
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
- `slatedb/src/db_reader.rs`: a reader ends with `DatabaseMissing` when a
  manifest loaded at its path does not hold the reader's own current
  checkpoint (`DbReaderInner::checkpoint_absent`, which logs how much of the
  lease was left or how long ago it ended). A database dropped and created
  again at the same path has a manifest, so the poll went on in it from the
  reader's own WAL cursor, among WAL ids that began again at 1: the reader
  kept the rows of the dropped database and showed none of the new one
  (Helion drops and creates a collection on a recreate). Within one database
  only the garbage collector takes a reader's checkpoint, once its stored
  expiry has passed, which can be up to a second before the reader's own
  copy of the expiry ends (the manifest stores whole seconds, rounded down)
  and earlier by its clock's lead; so no length of lease left tells a lost
  lease from another database, and both end the reader. Upstream answered a
  lost lease by writing a new checkpoint and going on (slatedb issue 1888);
  here the owner opens a new reader. The lease of an older generation that a
  read still holds is lost as before: that read fails, the reader goes on.
  The check is made in six places:
  - after the poll's first manifest load;
  - after the poll's delete of released checkpoints: that write loads the
    manifest again when its put meets another writer's, and the manifest it
    loads may be another database's;
  - in the poll's write of a new checkpoint
    (`StoredManifest::write_checkpoint_beside`, `manifest/store.rs`): the
    update loop applies a write again to the latest manifest at the path
    after such a conflict, and the reader's new checkpoint went into a
    database it never opened. The write is refused wherever the manifest it
    is applied to does not hold the checkpoint the reader has, on the first
    attempt and on every one after;
  - in the poll's lease refresh, which loads the manifest again;
  - `DbReader::close_if_absent_from_reopened(held, reopened)`, for a second
    reader opened at the path of the first (Helion's reopen to change a
    reader's poll interval). The second reader opens whatever database is
    there and writes its own checkpoint into it, so its polls find nothing
    wrong; the first reader's checkpoint, absent from the manifest the second
    opened, is what shows it. `held` comes from
    `DbReader::current_checkpoint()` before the second open: a checkpoint the
    first reader takes during it is in no manifest that open could read. A
    first reader that closed before the reopen deleted its checkpoint itself
    and is answered the same, under the reason it closed with;
  - `DbReader::close_if_absent_from_manifest()`, one manifest load for a
    caller that cannot wait for the next poll.
  The last two close the reader's status; its owner calls `close` to end the
  poller. Guarded by
  `poll_of_a_database_created_again_at_the_path_closes_the_reader`,
  `lease_refresh_of_a_database_created_again_closes_the_reader`,
  `checkpoint_put_that_meets_a_database_created_again_closes_the_reader`,
  `released_checkpoint_delete_that_meets_a_database_created_again_closes_the_reader`,
  `reader_that_renews_its_checkpoint_is_not_taken_for_another_database`,
  `checkpoint_taken_after_its_lease_ended_closes_the_reader`,
  `checkpoint_taken_before_its_lease_refresh_closes_the_reader`,
  `checkpoint_taken_in_the_last_second_of_its_lease_closes_the_reader`,
  `checkpoints_of_other_readers_do_not_look_like_another_database`,
  `reader_reopened_on_a_database_created_again_closes_the_first`,
  `reader_reopened_on_its_own_database_stays_open`,
  `reopen_of_a_reader_without_its_checkpoint_closes_it_in_its_own_database`
  and `reader_asked_between_polls_closes_on_a_database_created_again`, and by
  helixdb's `collection_created_again_is_not_read_from_the_dropped_one`,
  `collection_created_again_is_not_reopened_under_the_old_storage` and
  `list_walked_in_a_dropped_collection_does_not_answer_for_the_new_one`. No
  workflow runs the fork's tests: run them by hand with
  `cargo test --manifest-path vendor/slatedb/Cargo.toml -p slatedb --locked --lib -- db_reader::`.
- `slatedb/src/db/builder.rs` (+ `error.rs`): `DbBuilder::with_create_only`.
  A writer open takes any manifest it finds at its path. Helion creates a
  collection again at the path of a dropped one right after purging it, and
  a reader of the dropped database can put one more manifest there in
  between (it loads a manifest with one request and writes its checkpoint
  back with another): the open took it, and the collection came back with a
  manifest naming objects the purge had deleted. With the option set the
  build fails with `ErrorCode::DatabaseExists` when a manifest is there,
  before the manifest is fenced or anything is replayed, and also when the
  id of the new database's first manifest was taken between the look for a
  manifest and that write (`TransactionalObjectVersionExists`, which the id
  floor makes the usual place for such a late put). Off by default (upstream
  behaviour). Guarded by
  `create_only_build_creates_a_database_and_refuses_one_that_is_there` (a
  manifest found; the taken first id has no fork test) and, for both, by
  helixdb's `create_after_a_drop_opens_no_manifest_put_back_before_its_open`,
  `create_after_a_drop_opens_no_manifest_put_back_under_its_first_write` and
  `create_after_a_drop_gives_up_when_a_manifest_is_put_back_before_every_open`.
- `slatedb/src/db_cache/mod.rs`: `set_share_compacted_entries(true)` makes
  every `DbCacheWrapper` built afterwards key compacted-SST entries (blocks,
  indexes, filters, stats) by one shared scope instead of its own. Compacted
  SSTs are ULID-named and immutable, so a rebuilt `DbReader` can reuse what
  the previous handle loaded; upstream scoping made every reopen start cold
  and stranded the old entries until eviction. WAL entries stay per-wrapper
  (WAL ids restart per database). Off by default (upstream behaviour); Helion
  turns it on in `shared_db_cache` (`HELIX_LSM_SHARE_SST_CACHE`, default on).
  Guarded by `rebuilt_reader_reuses_shared_cache_entries_for_compacted_ssts`.
- `slatedb/src/cached_object_store/object_store.rs` (+ `config.rs`,
  `stats.rs`, `single_flight.rs`): ranged cache misses,
  `ObjectStoreCacheOptions::ranged_miss` (default on). On a disk-cache miss
  for a read of at most 1/16 of a part of at least 1 MiB, the caller GETs just
  its range while the whole part is fetched and saved by a background task
  through the same `part_flights` SingleFlight, so the cache layout (part
  files, head) is unchanged. A miss that finds a fill in flight waits for it
  as before; a failed ranged GET falls back to the full part; background fills
  share a process-wide 256 MiB budget and are skipped when it is spent. Part
  fetches now reject a body whose length does not match the part (upstream
  saved whatever arrived). Process-wide counters in
  `cached_object_store_stats::ranged_miss_stats()`. Helion sets it from
  `HELIX_LSM_CACHE_RANGED_MISS`. Guarded by the `ranged_miss` tests in
  `cached_object_store/object_store.rs`.
- `slatedb/src/cached_object_store/object_store.rs` (+ `config.rs`):
  concurrent part fetch, `ObjectStoreCacheOptions::part_fetch_concurrency`
  (default 8). A cached read spanning several parts reads up to that many
  parts at once (`buffered`, so they are still yielded in part order and the
  bytes returned are unchanged) instead of one after the other (prod: 240-370
  ms per 4 MiB part, so seconds for a segment's 100 MB+ `.hvec` sidecar blob
  on its first search). The number is lowered until the parts in flight hold
  at most 32 MiB (8 parts of 4 MiB on a reader; one 64 MiB part on the
  writer, i.e. the sequential read). On an error the read fails
  with the first failing part's error as before, but up to that many minus
  one later parts may already have been fetched and cached. 0 or 1 keeps
  upstream's sequential stream. Helion sets it from
  `HELIX_LSM_CACHE_PART_FETCH_CONCURRENCY`. Guarded by the
  `part_fetch_concurrency` tests in `cached_object_store/object_store.rs`.
- `slatedb/src/db_reader.rs`: an open `DbReader` can be polled at another
  interval, or now: `set_manifest_poll_interval`, `manifest_poll_interval`,
  `poll_manifest`. Upstream fixes the interval at the open, so Helion built a
  new reader (checkpoint write, manifest read, replay of the whole WAL tail)
  each time a collection moved between its 5 s and its 300 s poll tier: about
  100 times an hour on a production reader pod on 2026-11-17, 194 GET and 125
  HEAD each. The poll ticker keeps its period
  (`DbReaderOptions::manifest_poll_interval`, which is now also the shortest
  interval the reader can be given) and a poll runs on every n-th tick.
  `poll_manifest` wakes the poller through its message channel, whose sender
  upstream dropped, and returns when a poll that began after the call has
  finished; calls made together share one poll. A poll is what it was:
  manifest load, released checkpoints, a re-established checkpoint or the
  replay of new WAL objects, lease refresh. The rule of the open
  (`checkpoint_lifetime` at least twice the interval) is checked when an
  interval is set. A reader that is never told otherwise polls on every tick,
  as before. Guarded by `should_poll_at_the_interval_set_on_an_open_reader`,
  `should_reject_a_poll_interval_the_checkpoint_cannot_outlive`,
  `should_poll_now_on_request`,
  `should_release_a_waiting_poll_request_when_the_reader_closes` and
  `should_refresh_reader_checkpoint_at_an_interval_set_after_open`. A reader
  that is polled and not reopened keeps one replay table for every poll that
  found WAL objects; `reader_read_time_by_replay_table_count` builds a reader
  with 1, 75 and 1,000 of them and prints the time of a get and of a scan
  (it asserts counts and rows, no time; run it with `--nocapture`). No
  workflow runs the fork's tests; run them by hand with
  `cargo test --manifest-path vendor/slatedb/Cargo.toml -p slatedb --locked --lib -- db_reader::`.
- `slatedb/src/memtable_flusher/mod.rs` + `uploader.rs`, `db/builder.rs`,
  `db.rs`: an L0 flush asks a gate before it encodes its memtable
  (`L0FlushGate`, `DbBuilder::with_l0_flush_gate`). Upstream bounds flushes
  for each database on its own (`l0_flush_parallelism`), and a flush holds
  its whole SST encoded, every block twice, from `build_imm_ssts` until the
  upload ends: twelve Helion collections that reached their flush size in the
  same second took a 6 Gi writer over its limit in 1.45 s (2026-11-18). One
  gate is shared by every database of the process. `upload_with_retry` asks
  it with the database's path and the estimated bytes of the SST (the figure
  the freeze rule goes by) and keeps what it is given until it returns,
  however it returns; the rule itself is the embedder's (helixdb
  `backend_lsm_flush_gate.rs`). `upload_segment_sst` gives it back while an
  upload that failed waits to be tried again, and asks again before the next
  attempt: the flush takes nothing more while it sleeps, and kept through the
  wait the allowance would hold every other database's flush behind one
  upload that cannot get through (one allowance a flush, shared by its
  segments: Helion flushes one). While the gate keeps a flush of a database
  waiting, `maybe_apply_backpressure` holds that database's writes back by
  the wait it already has for `max_unflushed_bytes`, WAL replay at an open
  included: what the flush was kept from taking must not go into the next
  memtable instead. A flush that starts as it asks was never waiting and
  holds no write back. Not asked, or asked without waiting:
  a database with no WAL, where the flush is what makes a write durable, is
  not asked; a database that is closing is counted and does not wait, and a
  flush already waiting when its database starts to close stops waiting. With
  no gate, the default, nothing changes. The WAL, the manifest writer and the
  compactor do not ask it. No on-disk format is touched. Guarded by helixdb's
  `flush_gate` tests (`backend_lsm/tests/flush_gate.rs`,
  `collection_manager/tests/flush_gate.rs`); the fork has no test of its own
  for it.
- `slatedb/src/memtable_flusher/mod.rs` + `uploader.rs`: the gate is told of
  an L0 upload that failed and waits to be tried again
  (`L0FlushGate::upload_failed`, which does nothing unless the embedder
  overrides it). `upload_segment_sst` writes no line for that at any level,
  and tries the same encoded SST again every `manifest_poll_interval`, for
  ever. An SST of up to 10 MiB goes up in one put (the buffer of
  `object_store`'s `BufWriter`), and of that put only the errors
  `RetryingObjectStore` does not retry inside the call come back:
  AlreadyExists, Precondition, NotModified, NotFound, NotImplemented,
  NotSupported; every other one is retried inside the call, with an INFO line
  a retry, and is not told. A larger SST goes up in parts, and the wrapper
  retries only the call that starts them: any error of a part or of the
  completing call comes back. It arrives as SlateDB's `io error` or `object
  store error`, whose text is its class alone, so the gate is given its
  `{:?}`, which holds the store's own. With no gate nothing is told.
  Helion's gate writes a WARN with the collection and counts
  `helix_lsm_flush_upload_failed_total` (helixdb `backend_lsm_flush_gate.rs`).
  Guarded by helixdb's
  `the_gate_is_told_of_each_upload_that_fails_and_waits_to_be_tried_again`
  (`backend_lsm/tests/flush_gate.rs`); the fork has no test of its own for it.
- `slatedb/src/retrying_object_store.rs`: an upload in parts is aborted when
  one of its parts fails (`RetryingMultipartUpload::put_part`: the failed
  part's future aborts, then returns its error). `object_store`'s
  `WriteMultipart::finish` returns a part's error without aborting and takes
  the upload with it, and `write_sst_streaming_in_object_store` aborts
  nothing either, so every failed try of an SST over 10 MiB left its finished
  parts in the bucket. (63 unfinished uploads of SSTs, 4.35 GiB, were found
  on production's bucket on 2026-11-20; which of them a failed part left and
  which a process that died is not known.) A part in flight at the abort may
  still arrive, and a process that dies aborts nothing; a bucket
  lifecycle rule that aborts unfinished uploads is for those. Guarded by
  helixdb's
  `an_upload_in_parts_is_aborted_when_a_part_fails_before_it_is_tried_again`
  (`backend_lsm/tests/flush_gate.rs`); the fork has no test of its own for it.
- `slatedb/src/cached_object_store/storage_fs.rs` (+ `object_store.rs`): a
  disk cache holds parts of one size (`FsCacheStorage::keep_only_part_size`,
  called by `CachedObjectStore::from_config` with
  `ObjectStoreCacheOptions::part_size_bytes`). A part's size is in its file
  name (`_part4mb-…`, `_part1mb-…`) and a read builds the name from the
  configured size, so after a change of part size upstream keeps the old
  files: the scan counts them against the limit and they leave by eviction
  only, which never comes to a folder under its limit. The evictor's scan (at
  the start and at each reconcile) now unlinks a file whose name starts with
  `_part` and not with the configured size's prefix, a part's leftover temp
  file included, before it is counted, and logs how many at INFO. `_head`
  files and every other file are tracked as before, and a storage that is
  told no size (every caller but `from_config`) scans as upstream does. No
  lock is held over the unlinks, and a file that cannot be unlinked is
  counted like any other. Guarded by
  `a_part_file_prefix_names_one_size_only`,
  `a_scan_unlinks_parts_of_another_size_and_tracks_the_rest`,
  `a_scan_told_no_part_size_keeps_every_file`,
  `a_part_that_cannot_be_unlinked_stays_counted`,
  `a_storage_told_its_part_size_drops_the_others_at_its_first_scan`
  (`cached_object_store/storage_fs.rs`) and
  `a_store_built_from_config_drops_parts_of_another_size`
  (`cached_object_store/object_store.rs`).
- `slatedb/src/wal_buffer.rs`: `WalBufferManager::append` takes no entry once
  the WAL flush task has ended, and returns what the task ended with
  (`ended`, set by the task's cleanup before its last freeze). The task ends
  on a fatal error, a fenced WAL put first of all: it marks the database
  closed, then cleans up (freezes the current WAL and tells every waiting WAL
  its error), and only once it has ended does the monitor cancel the batch
  writer. A write that passed `check_closed` before the mark, and that the
  batch writer appended after the cleanup, landed in a WAL nobody flushes or
  tells, and `write_with_options` waits on that WAL's cell alone: a durable
  commit that never returned. (Seen on 2026-11-20 as a helixdb suite run that
  stood still in a fenced writer's commit.) Upstream names the gap: the
  struct's doc says a fatal error is kept and returned by later operations,
  and `append` carried a TODO for the check. A write refused here was never
  acknowledged and is not put into the memtable either. A clean close stops
  the batch writer before the flush task (`Db::close`) and has no such
  window. No on-disk format is touched. Guarded by helixdb's
  `a_durable_write_appended_after_the_wal_flush_task_ended_is_told_its_error`
  (`backend_lsm/tests/fenced_write.rs`) and
  `an_index_job_whose_store_is_appended_after_the_wal_flush_task_ended_leaves_its_worker`
  (`replication/tests/fenced_index_store.rs`); the fork has no test of its
  own for it.
- `slatedb/src/lib.rs`: `pub use slatedb_common;`, beside the crates the
  file re-exports already (bytes, fail_parallel, object_store) and for the
  reason it gives for them. `DbBuilder::with_system_clock` and
  `with_metrics_recorder` take a clock and a recorder whose traits are
  slatedb-common's, and helixdb depends on slatedb alone, so it could hand
  the builder neither. No method and no behavior is added. Its users are
  helixdb's tests: the second of the two above gives one test database a
  clock that keeps the batch writer's read of it waiting.
- `slatedb/src/memtable_flusher/mod.rs` + `db.rs`: a write that
  `maybe_apply_backpressure` held back for a flush waiting at the gate looks
  again when that flush is let in. The count of waiting flushes is a
  `tokio::sync::watch` (it was an atomic), and the wait has one arm more
  (`MemtableFlusher::no_flush_waits_at_gate`), armed only in a turn that
  found a flush waiting. Before, nothing woke that write but the upload of
  its memtable, a WAL flush or the wait's 30 s: it went in one upload later
  than a write that arrived after the flush was let in, and so could be
  acknowledged after one that arrived later. Why a watch: `wait_for` reads
  the count as the wait begins, so a flush let in between the write's look
  and its wait is not missed, as a notify would miss it. Why armed only
  then: a write held for `max_unflushed_bytes` alone finds the count at 0,
  and an arm that is ready at once would turn its wait into a loop. WAL
  replay at an open goes through the same wait and gets the same arm. With
  no gate the count is never above 0 and the arm is never armed. No on-disk
  format is touched. Guarded by helixdb's
  `a_write_held_for_a_waiting_flush_goes_in_when_the_flush_is_let_in`,
  `a_flush_let_in_between_a_writes_look_and_its_wait_is_not_missed` and
  `a_write_held_for_memory_alone_is_not_woken_at_the_gate`
  (`backend_lsm/tests/flush_gate.rs`), which hand the builder a metrics
  recorder and a clock of their own; the fork has no test of its own for
  it, and its own tests open no gate.

## Updating

Fetch the fork, check out the desired rev, re-copy the three crates + root
manifest trim, update `PINNED_REV`, and re-run the format-compatibility
review (manifest FlatBuffers schema, SST/WAL encoding) before shipping.
