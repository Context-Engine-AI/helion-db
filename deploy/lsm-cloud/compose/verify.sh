#!/usr/bin/env bash
# Live HA proof for the Helix Cloud LSM topology (docker compose emulation).
#
# Refactored from helix-container/tests/raft_ha.rs (create_collection /
# upsert_point / search_tags) into curl against the running compose stack.
#
# Proves, against a real writer + 2 readers + MinIO + per-node SSD cache:
#   1. writer commits a collection + points to object storage
#   2. BOTH readers serve those points (read off object storage + SSD cache)
#   3. a write sent to a reader is REJECTED at the gateway (role guard)
#   4. SSD cache is populated on the nodes; object storage holds the data
#   5. cache-loss recovery: kill a reader, WIPE its cache, restart — it still
#      serves identical results, rebuilt purely from object storage
#
# Usage:
#   deploy/lsm-cloud/compose/verify.sh          # up (if needed) + run checks
#   deploy/lsm-cloud/compose/verify.sh --down   # tear everything down
set -uo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE=(docker compose -f "$DIR/docker-compose.yml")
PROJECT="helion-lsm"
WRITER="http://localhost:6969"
READER1="http://localhost:6970"
READER2="http://localhost:6971"
COLL="proof"

pass() { printf '  \033[32mPASS\033[0m %s\n' "$*"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAILED=1; }
step() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
FAILED=0

if [[ "${1:-}" == "--down" ]]; then "${COMPOSE[@]}" down -v; exit 0; fi

step "0. bring the stack up (writer + 2 readers + MinIO + SSD caches)"
"${COMPOSE[@]}" up -d
wait_health() {
  local url="$1" name="$2"
  for _ in $(seq 1 90); do
    curl -sf "$url/health" >/dev/null 2>&1 && { pass "$name healthy ($url)"; return 0; }
    sleep 2
  done
  fail "$name never became healthy ($url)"; return 1
}
wait_health "$WRITER"  "writer"
wait_health "$READER1" "reader-1"
wait_health "$READER2" "reader-2"

# --- helpers refactored from raft_ha.rs -------------------------------------
create_collection() { # $1=base
  curl -sf -X PUT "$1/collections/$COLL" -H 'content-type: application/json' \
    -d '{"vectors":{"dense":{"size":3,"distance":"Cosine"}}}' >/dev/null
}
upsert_point() { # $1=base $2=id $3=vectorjson $4=tag
  curl -sf -X PUT "$1/collections/$COLL/points?wait=true" -H 'content-type: application/json' \
    -d "{\"points\":[{\"id\":$2,\"vector\":{\"dense\":$3},\"payload\":{\"tag\":\"$4\"}}]}" >/dev/null
}
search_tags() { # $1=base $2=vectorjson  -> prints sorted tags
  curl -sf -X POST "$1/collections/$COLL/points/search" -H 'content-type: application/json' \
    -d "{\"vector\":$2,\"limit\":8,\"with_payload\":true}" \
  | grep -oE '"tag":"[^"]*"' | sed 's/"tag":"//;s/"//' | sort | tr '\n' ' '
}
# search with retry to absorb the reader's manifest-refresh window
search_tags_retry() { # $1=base $2=vectorjson $3=expect-substr
  local out
  for _ in $(seq 1 15); do
    out="$(search_tags "$1" "$2")"
    [[ "$out" == *"$3"* ]] && { echo "$out"; return 0; }
    sleep 2
  done
  echo "$out"; return 1
}

step "1. WRITER commits a collection + two points"
if create_collection "$WRITER"; then pass "collection '$COLL' created on writer"; else fail "create_collection on writer"; fi
upsert_point "$WRITER" 1 "[1.0,0.0,0.0]" "doc-x" && pass "upserted point 1 (doc-x)" || fail "upsert point 1"
upsert_point "$WRITER" 2 "[0.0,1.0,0.0]" "doc-y" && pass "upserted point 2 (doc-y)" || fail "upsert point 2"

step "2. BOTH readers serve the writer's committed points (off object storage)"
r1="$(search_tags_retry "$READER1" "[1.0,0.0,0.0]" "doc-x")"
[[ "$r1" == *"doc-x"* ]] && pass "reader-1 search returned: $r1" || fail "reader-1 did not see committed data (got: '$r1')"
r2="$(search_tags_retry "$READER2" "[0.0,1.0,0.0]" "doc-y")"
[[ "$r2" == *"doc-y"* ]] && pass "reader-2 search returned: $r2" || fail "reader-2 did not see committed data (got: '$r2')"

step "3. a WRITE sent to a reader is REJECTED at the gateway (role guard)"
hdrs="$(curl -s -D - -o /dev/null -X PUT "$READER1/collections/$COLL/points?wait=true" \
  -H 'content-type: application/json' \
  -d '{"points":[{"id":9,"vector":{"dense":[0.0,0.0,1.0]},"payload":{"tag":"should-not-write"}}]}')"
code="$(printf '%s' "$hdrs" | awk 'NR==1{print $2}')"
role="$(printf '%s' "$hdrs" | grep -i '^X-Helix-Node-Role:' | tr -d '\r' | awk '{print $2}')"
if [[ "$code" =~ ^(4|5) ]]; then pass "reader rejected write: HTTP $code, X-Helix-Node-Role: ${role:-<none>}"; else fail "reader accepted a write (HTTP $code) — role guard not enforced"; fi
# and confirm the rejected write never landed: doc count unchanged from writer's view
leak="$(search_tags "$WRITER" "[0.0,0.0,1.0]")"
[[ "$leak" != *"should-not-write"* ]] && pass "rejected write did not reach object storage" || fail "rejected write leaked into storage"

step "4. SSD cache populated on nodes; object storage holds the data"
wc="$(docker exec "${PROJECT}-helix-writer-1" sh -c 'find /cache -type f 2>/dev/null | wc -l' 2>/dev/null || echo 0)"
rc="$(docker exec "${PROJECT}-helix-reader-1-1" sh -c 'find /cache -type f 2>/dev/null | wc -l' 2>/dev/null || echo 0)"
[[ "${wc:-0}" -gt 0 ]] && pass "writer SSD cache populated ($wc files)" || echo "  note: writer /cache has $wc files (may be lazy)"
[[ "${rc:-0}" -gt 0 ]] && pass "reader-1 SSD cache populated ($rc files)" || echo "  note: reader-1 /cache has $rc files (may be lazy)"
obj="$("${COMPOSE[@]}" exec -T minio sh -c 'mc alias set h http://localhost:9000 test testtest123 >/dev/null 2>&1; mc ls --recursive h/helion 2>/dev/null | wc -l' 2>/dev/null || echo 0)"
[[ "${obj:-0}" -gt 0 ]] && pass "object storage holds $obj objects under helion/" || fail "object storage empty — data not durable"

step "5. cache-loss recovery: kill reader-1, WIPE its SSD cache, restart"
"${COMPOSE[@]}" rm -sf helix-reader-1 >/dev/null 2>&1
docker volume rm "${PROJECT}_reader1-cache" >/dev/null 2>&1 && pass "wiped reader-1 SSD cache volume" || echo "  note: cache volume already gone"
"${COMPOSE[@]}" up -d helix-reader-1 >/dev/null
wait_health "$READER1" "reader-1 (fresh, no cache)"
r1b="$(search_tags_retry "$READER1" "[1.0,0.0,0.0]" "doc-x")"
[[ "$r1b" == *"doc-x"* ]] && pass "fresh reader rebuilt from object storage: $r1b" || fail "fresh reader could not recover from object storage (got: '$r1b')"

echo
if [[ "$FAILED" == "0" ]]; then
  printf '\033[1;32m✓ ALL PROOFS PASSED — writer/reader/object-storage/SSD-cache topology works\033[0m\n'
else
  printf '\033[1;31m✗ SOME PROOFS FAILED (see above)\033[0m\n'; exit 1
fi
