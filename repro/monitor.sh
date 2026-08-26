#!/usr/bin/env bash
# Scrapes /metrics every 2s. Detects the wedge by watching:
#   - write_queue_depth: should be flat or shrinking under steady load
#   - blocking_admission_available{class=write}: should oscillate, not pin low
#   - chunk_success_total: should keep ticking up under load
#
# Outputs a CSV trace + a final PASS/FAIL verdict.

set -uo pipefail

HOST="${HOST:-http://localhost:6970}"
SAMPLES="${SAMPLES:-45}"            # ~90s at 2s interval
INTERVAL="${INTERVAL:-2}"
TRACE="${TRACE:-/tmp/helix-repro-trace.csv}"

# Tolerances: how long stalled metrics are allowed before we call it a wedge.
STALL_WINDOW="${STALL_WINDOW:-6}"   # samples (~12s)

scrape() {
  curl -sS --max-time 4 "$HOST/metrics" 2>/dev/null
}

field() {
  local metric="$1" labels="${2:-}"
  awk -v m="$metric" -v l="$labels" '
    $0 ~ "^" m && (l == "" || index($0, l) > 0) {
      n = split($0, parts, " ")
      print parts[n]
      exit
    }
  '
}

echo "ts,queue_depth,write_avail,read_avail,scan_avail,chunk_succ,batch_succ,points,chunked_jobs" >"$TRACE"
last_chunk=0
last_queue=0
stalled=0
high_water_queue=0
seen_chunked_job=0

for ((i=0; i<SAMPLES; i++)); do
  ts=$(date +%s)
  metrics=$(scrape)
  if [[ -z "$metrics" ]]; then
    echo "[mon $(date +%H:%M:%S)] scrape failed"
    sleep "$INTERVAL"
    continue
  fi
  q=$(echo "$metrics"      | field "helix_write_queue_depth ")
  wa=$(echo "$metrics"     | field "helix_blocking_admission_available" 'class="write"')
  ra=$(echo "$metrics"     | field "helix_blocking_admission_available" 'class="read"')
  sa=$(echo "$metrics"     | field "helix_blocking_admission_available" 'class="scan"')
  cs=$(echo "$metrics"     | field "helix_write_queue_upsert_chunk_success_total ")
  bs=$(echo "$metrics"     | field "helix_write_queue_batch_success_total ")
  pts=$(echo "$metrics"    | field "helix_write_queue_points_batched_total ")
  cj=$(echo "$metrics"     | field "helix_write_queue_upsert_chunked_jobs_total ")
  q="${q:-0}"; wa="${wa:-0}"; ra="${ra:-0}"; sa="${sa:-0}"
  cs="${cs:-0}"; bs="${bs:-0}"; pts="${pts:-0}"; cj="${cj:-0}"
  echo "$ts,$q,$wa,$ra,$sa,$cs,$bs,$pts,$cj" >>"$TRACE"

  (( $(echo "$q > $high_water_queue" | bc -l 2>/dev/null || echo 0) )) && high_water_queue=$q
  [[ "$cj" -gt 0 ]] && seen_chunked_job=1

  printf "[mon %s]  q=%-6s wa=%-3s ra=%-3s sa=%-3s chunks=%-4s batches=%-3s pts=%-6s cj=%s\n" \
    "$(date +%H:%M:%S)" "$q" "$wa" "$ra" "$sa" "$cs" "$bs" "$pts" "$cj"

  # Wedge detector: queue stays >0 AND chunk counter doesn't tick AND queue
  # isn't shrinking → likely wedge.
  if [[ "$q" -gt 0 && "$cs" == "$last_chunk" && "$q" -ge "$last_queue" ]]; then
    stalled=$((stalled + 1))
  else
    stalled=0
  fi
  last_chunk="$cs"
  last_queue="$q"

  if [[ "$stalled" -ge "$STALL_WINDOW" ]]; then
    echo
    echo "═══════════════════════════════════════════════════════════════"
    echo "  WEDGE DETECTED at sample $i"
    echo "  queue_depth=$q (high water=$high_water_queue)  chunks_succ=$cs (frozen for ${STALL_WINDOW} samples)"
    echo "  write_admission_available=$wa  read=$ra  scan=$sa"
    echo "  saw chunked jobs at all? $([[ $seen_chunked_job -eq 1 ]] && echo yes || echo no)"
    echo "  trace: $TRACE"
    echo "═══════════════════════════════════════════════════════════════"
    exit 2
  fi
  sleep "$INTERVAL"
done

echo
echo "═══════════════════════════════════════════════════════════════"
echo "  NO WEDGE in $SAMPLES samples (~$((SAMPLES * INTERVAL))s)"
echo "  high water queue depth=$high_water_queue  chunked jobs ran=$seen_chunked_job"
echo "  trace: $TRACE"
echo "═══════════════════════════════════════════════════════════════"
exit 0
