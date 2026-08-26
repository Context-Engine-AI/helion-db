# LSM deployment examples

These examples demonstrate Helion's **LSM-on-object-storage** topology: one
fenced writer per collection prefix, disposable read replicas, object storage
as the durable system of record, and a local cache per node.

```
        ┌─────────── Gateway / LB ───────────┐
        │   writes → writer   reads → readers │
   ┌────▼────┐   ┌─────────┐   ┌─────────┐
   │ writer  │   │ reader  │   │ reader  │   …horizontal read scale
   │ ┌─────┐ │   │ ┌─────┐ │   │ ┌─────┐ │
   │ │cache│ │   │ │cache│ │   │ │cache│ │   ← local SSD cache (HELIX_LSM_CACHE_DIR)
   └────┬────┘   └────┬────┘   └────┬────┘
        └───────── object storage (MinIO / S3) ─────────┘   ← durable record
```

Roles are config-driven via **`HELIX_LSM_ROLE`** (`writer` default | `reader`).
Reader nodes serve committed reads off object storage + their SSD cache and
**reject writes at the gateway** (HTTP 503, `X-Helix-Node-Role: reader`).

## Build the image

```bash
docker build -t helix-db:lsm-local .      # from repo root
```

## Option A — docker compose (fastest local proof)

```bash
docker compose -f deploy/lsm-cloud/compose/docker-compose.yml up -d
deploy/lsm-cloud/compose/verify.sh        # live HA proof (all 5 checks)
docker compose -f deploy/lsm-cloud/compose/docker-compose.yml down -v
```

Ports: writer `:6969`, reader-1 `:6970`, reader-2 `:6971`, MinIO console `:9001`.
Compose runs nodes as root because docker named volumes are root-owned.

## Option B — Minikube

```bash
minikube start
minikube image load helix-db:lsm-local
deploy/lsm-cloud/minikube/verify.sh       # apply + port-forward + HA proof
deploy/lsm-cloud/minikube/verify.sh --down
```

K8s uses `securityContext.fsGroup: 1000` (not root) so the image's non-root
`helix` user can write the `emptyDir` cache/data volumes — the production-correct
mechanism. Readers run as a 2-replica Deployment behind a ClusterIP service that
load-balances reads; the SSD cache is an `emptyDir`, so deleting a reader pod
wipes its cache and proves recovery from object storage.

A provider-neutral ConfigMap template is available under `kubernetes/`. It
contains placeholders only; supply credentials through a Secret or workload
identity instead of putting them in the ConfigMap.

## What the proof demonstrates (both options)

1. writer commits a collection + points → persisted to object storage
2. every reader serves those points (read off object storage + SSD cache)
3. a write sent to a reader is rejected at the gateway (role guard)
4. SSD cache populated on nodes; object storage holds the data
5. **cache-loss recovery** — wipe a reader's cache / replace the pod; a fresh
   reader with no local state rebuilds and serves identical results from object
   storage (readers are disposable, object storage is the system of record)
