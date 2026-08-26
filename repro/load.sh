#!/usr/bin/env bash
# Concurrent write generator that targets the production wedge pattern:
#   - more collections than HELIX_MAX_OPEN_COLLECTIONS (forces eviction churn)
#   - parallel upserters per collection (forces close/reopen race)
#   - large bodies (exercises chunked path)
#   - bursts of writes to a single "hot" collection (mimics indexing storm)
#
# Bodies are pre-generated to disk via Python at startup. The hot path is then
# pure curl --data-binary @file, no bash string concatenation. (Bash O(n²)
# string-append on 24 KB-per-element vectors at DIM=1536 was killing the host
# CPU before any traffic reached helix.)

set -uo pipefail

HOST="${HOST:-http://localhost:6970}"
NUM_COLLECTIONS="${NUM_COLLECTIONS:-8}"
WRITERS_PER_COLLECTION="${WRITERS_PER_COLLECTION:-3}"
HOT_COLLECTION_BURSTS="${HOT_COLLECTION_BURSTS:-4}"
DURATION_SECS="${DURATION_SECS:-60}"
DIM="${DIM:-1536}"
WRITER_BATCH="${WRITER_BATCH:-200}"
HOT_BURST_BATCH="${HOT_BURST_BATCH:-1000}"
BODY_DIR="${BODY_DIR:-/tmp/helix-repro-bodies}"

log() { echo "[load $(date +%H:%M:%S)] $*"; }

create_collection() {
  local name="$1"
  curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
    -X PUT "$HOST/collections/$name" \
    -H "Content-Type: application/json" -H "Expect:" \
    -d "{\"vectors\":{\"dense\":{\"size\":$DIM,\"distance\":\"Cosine\"}}}"
}

# Pre-generate one payload per (worker, role). Each worker uses a non-overlapping
# id range so concurrent writers to the same collection don't fight on identical
# point ids. We rebuild the file at startup; keep IDs overlapping ACROSS bursts
# from the same writer so the work simulates an indexer reuploading slices.
generate_body() {
  local out="$1" count="$2" base_id="$3"
  python3 - "$out" "$count" "$base_id" "$DIM" <<'PY'
import json, random, sys
out, count, base_id, dim = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
points = [
    {"id": base_id + i,
     "vector": {"dense": [round(random.random(), 4) for _ in range(dim)]},
     "payload": {"k": f"v{i}"}}
    for i in range(count)
]
with open(out, "w") as fh:
    json.dump({"points": points}, fh)
PY
}

upsert_file() {
  local name="$1" body_path="$2"
  curl -s -o /dev/null -w "%{http_code}\n" --max-time 30 \
    -X PUT "$HOST/collections/$name/points" \
    -H "Content-Type: application/json" -H "Expect:" \
    --data-binary @"$body_path"
}

writer_loop() {
  local name="$1" body_path="$2" deadline="$3"
  while [[ $(date +%s) -lt $deadline ]]; do
    upsert_file "$name" "$body_path" >/dev/null 2>&1
  done
}

hot_burst_loop() {
  local name="$1" body_path="$2" deadline="$3"
  while [[ $(date +%s) -lt $deadline ]]; do
    upsert_file "$name" "$body_path" >/dev/null 2>&1
    sleep 0.2
  done
}

mkdir -p "$BODY_DIR"
rm -f "$BODY_DIR"/*.json 2>/dev/null

log "creating $NUM_COLLECTIONS collections"
for ((i=0; i<NUM_COLLECTIONS; i++)); do
  status=$(create_collection "wedge-$i")
  log "  wedge-$i create=$status"
done

log "pre-generating bodies (DIM=$DIM, writer_batch=$WRITER_BATCH, hot_burst=$HOT_BURST_BATCH)"
gen_pids=()
# One writer body per (collection, worker). Distinct id ranges per writer so
# concurrent writers don't pile on the same ids.
for ((i=0; i<NUM_COLLECTIONS; i++)); do
  for ((w=0; w<WRITERS_PER_COLLECTION; w++)); do
    out="$BODY_DIR/wedge-$i-w$w.json"
    base=$(( (i * WRITERS_PER_COLLECTION + w) * 100000 ))
    generate_body "$out" "$WRITER_BATCH" "$base" &
    gen_pids+=($!)
  done
done
# Hot bursters share the same hot id range so they simulate an indexer
# re-pushing slices of one tenant's collection (lots of upsert overwrites).
for ((b=0; b<HOT_COLLECTION_BURSTS; b++)); do
  out="$BODY_DIR/hot-b$b.json"
  base=$(( 999000000 + b * 1000000 ))
  generate_body "$out" "$HOT_BURST_BATCH" "$base" &
  gen_pids+=($!)
done
wait "${gen_pids[@]}" 2>/dev/null
total_size=$(du -sh "$BODY_DIR" 2>/dev/null | awk '{print $1}')
log "bodies ready: $(ls "$BODY_DIR" | wc -l) files, total=$total_size"

deadline=$(($(date +%s) + DURATION_SECS))
log "starting $WRITERS_PER_COLLECTION writers per collection + $HOT_COLLECTION_BURSTS hot bursters for ${DURATION_SECS}s"

PIDS=()
for ((i=0; i<NUM_COLLECTIONS; i++)); do
  for ((w=0; w<WRITERS_PER_COLLECTION; w++)); do
    body="$BODY_DIR/wedge-$i-w$w.json"
    writer_loop "wedge-$i" "$body" "$deadline" &
    PIDS+=($!)
  done
done
for ((b=0; b<HOT_COLLECTION_BURSTS; b++)); do
  body="$BODY_DIR/hot-b$b.json"
  hot_burst_loop "wedge-0" "$body" "$deadline" &
  PIDS+=($!)
done

log "load running, ${#PIDS[@]} writer processes"
wait "${PIDS[@]}" 2>/dev/null
log "load complete"
