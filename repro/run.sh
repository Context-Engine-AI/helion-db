#!/usr/bin/env bash
# Orchestrator: build → up → load + monitor in parallel → tear down → report.
# Exit 0  = no wedge detected (good — fix worked, or no bug present)
# Exit 2  = wedge detected (repro confirmed)
# Exit 1  = harness setup failed (build, healthcheck, etc.)

set -uo pipefail
cd "$(dirname "$0")"

REBUILD="${REBUILD:-1}"
LOAD_DURATION="${LOAD_DURATION:-60}"
MONITOR_SAMPLES="${MONITOR_SAMPLES:-45}"
KEEP_RUNNING="${KEEP_RUNNING:-0}"

log() { echo "[run $(date +%H:%M:%S)] $*"; }

cleanup() {
  if [[ "$KEEP_RUNNING" != "1" ]]; then
    log "tearing down container"
    docker compose down -v >/dev/null 2>&1 || true
  else
    log "KEEP_RUNNING=1 — container left up. inspect with: docker logs helix-repro"
  fi
}
trap cleanup EXIT

if [[ "$REBUILD" == "1" ]]; then
  log "building helix image (this is the slow part — release build)"
  if ! docker compose build helix; then
    log "build failed"; exit 1
  fi
fi

log "starting helix"
docker compose up -d helix >/dev/null
log "waiting for /health"
for i in {1..60}; do
  if curl -sS --max-time 1 "http://localhost:6970/health" 2>/dev/null | grep -q ok; then
    log "helix healthy after ${i}s"
    break
  fi
  sleep 1
  if [[ $i -eq 60 ]]; then
    log "helix did not become healthy"
    docker logs helix-repro --tail 50
    exit 1
  fi
done

log "starting monitor (${MONITOR_SAMPLES} samples) in background"
DURATION_SECS="$LOAD_DURATION" SAMPLES="$MONITOR_SAMPLES" \
  ./monitor.sh > /tmp/helix-repro-monitor.log 2>&1 &
MON_PID=$!

log "starting load (${LOAD_DURATION}s)"
DURATION_SECS="$LOAD_DURATION" ./load.sh > /tmp/helix-repro-load.log 2>&1
LOAD_RC=$?

log "load done (rc=$LOAD_RC), waiting for monitor"
wait "$MON_PID"
MON_RC=$?

echo
log "═══════════ MONITOR LOG ═══════════"
cat /tmp/helix-repro-monitor.log
echo
log "═══════════ LAST 20 HELIX LOG LINES ═══════════"
docker logs helix-repro --tail 20 2>&1
echo

if [[ "$MON_RC" -eq 2 ]]; then
  log "RESULT: WEDGE REPRODUCED"
elif [[ "$MON_RC" -eq 0 ]]; then
  log "RESULT: no wedge detected"
else
  log "RESULT: harness error (rc=$MON_RC)"
fi
exit "$MON_RC"
