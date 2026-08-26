# Helix wedge repro harness

Reproduces the production write-queue wedge in Docker so we can find and verify
the fix without touching prod.

## What it does

- Builds the helix binary from local source.
- Starts one helix container with the same gateway/queue config as prod, but
  with caps tightened so the bug surfaces in seconds:
  - `HELIX_MAX_OPEN_COLLECTIONS=4` and `NUM_COLLECTIONS=8` → forces LRU
    eviction every few requests, exercising the close/reopen race.
  - `HELIX_WRITE_BLOCKING_CAP=2` → a single stuck handler is obvious.
  - `HELIX_INITIAL_MAP_MB=8` + small `HELIX_MAP_GROW_MB=16` → frequent
    `grow_map` and `ensure_map_headroom` calls.
- Generates concurrent upserts (steady writers per collection + a "hot" tenant
  doing 50-point bursts to one collection — mirrors a bulk re-index storm).
- Scrapes `/metrics` every 2s and detects a wedge: queue depth ≥ previous
  sample AND `helix_write_queue_upsert_chunk_success_total` not advancing for
  6 consecutive samples.

## Run

```sh
./run.sh
```

Exit codes:
- `0`  — no wedge in the observation window (fix worked, or bug not present)
- `2`  — wedge reproduced
- `1`  — harness setup failed (build / health / etc.)

## Tuning

```sh
LOAD_DURATION=120 MONITOR_SAMPLES=60 ./run.sh   # longer run
KEEP_RUNNING=1 ./run.sh                          # leave container up for inspection
REBUILD=0 ./run.sh                               # skip the rebuild (use last image)
```

## Inspect the wedge state

When `KEEP_RUNNING=1` and the run reports a wedge, the container is still up:

```sh
docker logs helix-repro --tail 100
docker exec helix-repro /bin/sh -c '
  for t in /proc/1/task/*/wchan; do cat "$t"; echo; done | sort | uniq -c | sort -rn
'
curl -sS http://localhost:6970/metrics | grep -E '^helix_(write_queue|blocking_admission)'
```

## Workflow

1. Run the harness. It should reproduce the wedge against current `HEAD`.
2. Apply a candidate fix in helix source.
3. Re-run with `REBUILD=1`. Should exit 0.
4. Run a longer duration (`LOAD_DURATION=300`) to confirm no wedge under
   sustained load.
5. Only then deploy to prod.
