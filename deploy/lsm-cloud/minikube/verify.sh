#!/usr/bin/env bash
# Live HA proof for the Helix Cloud LSM topology on minikube.
#
# Brings the topology up on the current cluster, then proves (same checks as the
# compose verify, refactored from helix-container/tests/raft_ha.rs):
#   1. writer commits a collection + points to object storage
#   2. both reader pods serve those points (off object storage + SSD cache)
#   3. a write sent to a reader is REJECTED at the gateway (role guard)
#   4. object storage holds the data
#   5. cache-loss recovery: delete the reader pods (fresh emptyDir caches) — a
#      brand-new reader still serves identical results, rebuilt from object storage
#
# Prereqs:
#   docker build -t helix-db:lsm-local .
#   minikube start
#   minikube image load helix-db:lsm-local
#
# Usage:
#   deploy/lsm-cloud/minikube/verify.sh           # apply + prove
#   deploy/lsm-cloud/minikube/verify.sh --down    # delete the namespace
set -uo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
NS=helion-lsm
COLL=proof
WRITER="http://localhost:7969"
READER="http://localhost:7970"

pass() { printf '  \033[32mPASS\033[0m %s\n' "$*"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAILED=1; }
step() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
FAILED=0
PF_PIDS=()
cleanup() { for p in "${PF_PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

if [[ "${1:-}" == "--down" ]]; then kubectl delete namespace "$NS" --ignore-not-found; exit 0; fi

step "0. apply manifests + wait for rollout"
kubectl apply -f "$DIR/00-namespace.yaml" -f "$DIR/10-minio.yaml" -f "$DIR/20-helix.yaml"
kubectl -n "$NS" rollout status deploy/minio --timeout=120s
kubectl -n "$NS" rollout status deploy/helix-writer --timeout=180s
kubectl -n "$NS" rollout status deploy/helix-reader --timeout=180s
pass "all deployments rolled out"

step "0b. port-forward writer (:7969) and reader service (:7970)"
kubectl -n "$NS" port-forward svc/helix-writer 7969:6969 >/dev/null 2>&1 & PF_PIDS+=($!)
kubectl -n "$NS" port-forward svc/helix-reader 7970:6969 >/dev/null 2>&1 & READER_PF=$!; PF_PIDS+=("$READER_PF")
for url in "$WRITER" "$READER"; do
  for _ in $(seq 1 30); do curl -sf "$url/health" >/dev/null 2>&1 && break; sleep 1; done
done
pass "port-forwards ready"

# --- helpers (curl, refactored from raft_ha.rs) -----------------------------
create_collection() { curl -sf -X PUT "$1/collections/$COLL" -H 'content-type: application/json' \
  -d '{"vectors":{"dense":{"size":3,"distance":"Cosine"}}}' >/dev/null; }
upsert_point() { curl -sf -X PUT "$1/collections/$COLL/points?wait=true" -H 'content-type: application/json' \
  -d "{\"points\":[{\"id\":$2,\"vector\":{\"dense\":$3},\"payload\":{\"tag\":\"$4\"}}]}" >/dev/null; }
search_tags() { curl -sf -X POST "$1/collections/$COLL/points/search" -H 'content-type: application/json' \
  -d "{\"vector\":$2,\"limit\":8,\"with_payload\":true}" | grep -oE '"tag":"[^"]*"' | sed 's/"tag":"//;s/"//' | sort | tr '\n' ' '; }
search_retry() { local out; for _ in $(seq 1 15); do out="$(search_tags "$1" "$2")"; [[ "$out" == *"$3"* ]] && { echo "$out"; return 0; }; sleep 2; done; echo "$out"; return 1; }

step "1. WRITER commits a collection + two points"
create_collection "$WRITER" && pass "collection created" || fail "create collection"
upsert_point "$WRITER" 1 "[1.0,0.0,0.0]" "doc-x" && pass "upserted doc-x" || fail "upsert doc-x"
upsert_point "$WRITER" 2 "[0.0,1.0,0.0]" "doc-y" && pass "upserted doc-y" || fail "upsert doc-y"

step "2. reader pods serve the committed points (off object storage)"
r="$(search_retry "$READER" "[1.0,0.0,0.0]" "doc-x")"
[[ "$r" == *"doc-x"* ]] && pass "reader search returned: $r" || fail "reader did not see committed data (got: '$r')"

step "3. a WRITE sent to a reader is REJECTED at the gateway"
hdrs="$(curl -s -D - -o /dev/null -X PUT "$READER/collections/$COLL/points?wait=true" \
  -H 'content-type: application/json' -d '{"points":[{"id":9,"vector":{"dense":[0,0,1]}}]}')"
code="$(printf '%s' "$hdrs" | awk 'NR==1{print $2}')"
role="$(printf '%s' "$hdrs" | grep -i '^X-Helix-Node-Role:' | tr -d '\r' | awk '{print $2}')"
[[ "$code" =~ ^(4|5) ]] && pass "reader rejected write: HTTP $code, role=${role:-<none>}" || fail "reader accepted a write (HTTP $code)"

step "4. object storage holds the data"
obj="$(kubectl -n "$NS" exec deploy/minio -- sh -c 'mc alias set h http://localhost:9000 test testtest123 >/dev/null 2>&1; mc ls --recursive h/helion 2>/dev/null | wc -l' 2>/dev/null | tr -d '[:space:]')"
[[ "${obj:-0}" -gt 0 ]] && pass "object storage holds $obj objects under helion/" || fail "object storage empty"

step "5. cache-loss recovery: delete reader pods (fresh emptyDir caches), re-read"
kubectl -n "$NS" delete pod -l "role=reader" --wait=true >/dev/null
kubectl -n "$NS" rollout status deploy/helix-reader --timeout=120s >/dev/null
# re-establish reader port-forward to the new pods (bash 3.2-safe: named PID, no negative index)
kill "$READER_PF" 2>/dev/null || true
kubectl -n "$NS" port-forward svc/helix-reader 7970:6969 >/dev/null 2>&1 & READER_PF=$!; PF_PIDS+=("$READER_PF")
for _ in $(seq 1 30); do curl -sf "$READER/health" >/dev/null 2>&1 && break; sleep 1; done
rb="$(search_retry "$READER" "[1.0,0.0,0.0]" "doc-x")"
[[ "$rb" == *"doc-x"* ]] && pass "fresh reader rebuilt from object storage: $rb" || fail "fresh reader could not recover (got: '$rb')"

echo
if [[ "$FAILED" == "0" ]]; then
  printf '\033[1;32m✓ ALL PROOFS PASSED on minikube\033[0m\n'
else
  printf '\033[1;31m✗ SOME PROOFS FAILED\033[0m\n'; exit 1
fi
