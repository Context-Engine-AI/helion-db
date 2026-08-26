# LMDB Resize Fence

How Helion keeps LMDB map growth safe without stalling reads or writes, and the
invariants that must hold for every transaction. Companion to
[read-write-isolation-plan.md](./read-write-isolation-plan.md).

## The constraint

LMDB grows its memory map with `mdb_env_set_mapsize`, which may **only** be
called when there are no open transactions in the process — otherwise a live
txn's pointers can dangle when the map is remapped (UB: `_mdb_cursor_put`
SIGSEGV, `MDB_PROBLEM`). The resize call itself is cheap: `munmap` + `mmap` of a
virtual reservation, microseconds, **size-independent**. Any time a "resize"
appears slow, the cost is *draining open transactions*, never the syscall.

## How the fence works

The fence is a **counter, not a long-held lock**. (The legacy `resize_gate`
RwLock is no longer held for a transaction's lifetime — that caused writer
starvation.)

- Every transaction is opened through a tracked wrapper
  (`with_read_txn` / `begin_resize_safe_read_txn` / `with_write_txn` /
  `begin_tracked_write_txn`). The wrapper increments `active_lmdb_txns` on open
  and decrements it when the txn drops (`begin_lmdb_txn` / `finish_lmdb_txn`).
- A resize marks itself pending (`resize_writers_pending`), which **parks new
  txn admission**, then waits for `active_lmdb_txns` to drain to 0
  (`wait_for_lmdb_txns_to_drain`) before calling `env.resize()`.
- Net effect: new work is throttled for the brief (ms-scale) drain window;
  in-flight short txns finish; the resize fires in microseconds; new work
  resumes. Reads never block behind the resize itself.

**The invariant:** `active_lmdb_txns == 0` must mean *no LMDB txn is live*. That
is only true if **every** production transaction goes through a tracked opener.

## June 2026 incident (resize-fence bypass)

The fence was *opt-in*. Roughly five hot production paths opened raw
`graph_env.read_txn()` and bypassed the counter — the worst was dense
**parallel** search (per-segment txns inside a `rayon` `par_iter`, enabled by
`HELIX_DENSE_PARALLEL_SEARCH=1`), plus the graph-bridge probe,
`env_live_bytes` (auto-compaction), and two replication reads. With those
uncounted, `wait_for_lmdb_txns_to_drain` could observe `0` and remap **under a
live txn**, producing the `_mdb_cursor_put` SIGSEGV / `MDB_PROBLEM` crashes, and
— via the permit held across the LMDB writer mutex — a silent **write-convoy
wedge** (writers pinned at the submit cap, `write_txn_gate` waits of 16–45 s,
reader stalls of 83–86 s, self-healing only at the 120 s write timeout).

Six prior fixes did not stick because **both validators were stale and reported
false-green**: the loom model exercised the abandoned legacy `resize_gate`, not
the active counter; and `repro/monitor.sh` watched retired `FairWriteQueue`
metrics that the direct-submit binary never emits. Fixes were "verified" against
dead signals.

## The fix

1. **Seal the door.** All production reads/writes go through the tracked
   openers so `active_lmdb_txns == 0` is a true barrier. Never call
   `graph_env.read_txn()` / `write_txn()` directly outside tests.
2. **Non-parking nested admission.** Dense parallel search holds a tracked
   *outer* read txn across the `par_iter` while each worker opens its own
   per-segment txn (`RoTxn<WithTls>` is `!Send`, so the outer txn cannot be
   shared across rayon threads). Routing those nested opens through the normal
   admission gate **deadlocks**: the outer txn keeps `active_lmdb_txns >= 1`, a
   pending resize waits for it to reach 0, and the worker opens park on
   resize-pending so the search can never finish. Nested opens therefore use
   `begin_nested_lmdb_txn` — a counted but **non-parking** admission that
   requires `active_before > 0` and increments under `resize_wait_lock`. It is
   safe because a held outer permit proves the resize has not yet remapped, so
   counting one more short-lived nested txn cannot corrupt; it only makes the
   resize wait a few more microseconds. Construct it via
   `storage.nested_dense_read_provider(&outer_txn)`.

Originally shipped in the private development lineage before the public
`helion-db` source release.

## Invariants to preserve

- **Never** open a raw `graph_env.read_txn()` / `write_txn()` on a serving
  path. Use the tracked openers.
- Keep tracked read txns **short**. The resize drain waits for the longest
  in-flight txn; a long-held read re-introduces a stall (not a lock, but a
  drain wait).
- For a txn opened **while another tracked txn is already held on the same
  storage** (e.g. per-segment search under an outer search txn), use the
  non-parking nested admission, never the parking path.
- Resize is microseconds. If you need to cut resize *frequency*, pre-sizing the
  map is disk-free (Helion opens without `MDB_WRITEMAP`, so `map_size` is pure
  virtual reservation — an empty collection is ~16 KiB on disk under a 512 MiB
  map), but a large map costs VmPTE per env. Pre-size only the few giant
  collections, never all of them.

## Operability — verifying on a live pod

Check a serving pod's `/metrics`, **not** `repro/` (its detector watches retired
metrics). Healthy vs wedged:

| Signal | Healthy | Wedged |
| --- | --- | --- |
| `helix_write_txn_gate_wait_ms` | sub-millisecond | 16,000–45,000 ms |
| `helix_resize_active_txn_drain_wait_ms` | ~10^-4 ms (tens of ns) | climbing / seconds |
| `helix_submit_active_writers` | 0–low | pinned at cap (e.g. 24) |
| `helix_submit_available_permits{class="write"}` | at cap (e.g. 24) | 0 |
| `helix_lmdb_active_tracked_txns` | low, oscillating | stuck ≥1 |
| `helix_lmdb_nested_txn_admission_ms` | µs per call (parallel search) | — |
| logs | clean | `write_txn_gate acquisition is slow`, `escalating to blocking resize`, `MDB_PROBLEM`, SIGSEGV |

The wedge has no self-heal until `HELIX_WRITE_REQUEST_TIMEOUT_SECS` (120 s), so a
fresh `/health` (readiness) can stay green while writes stall — watch the write
metrics, not just liveness.
