//! Per-collection async admission for point mutation routes.
//!
//! Same-collection upserts and deletes are also serialized by a synchronous
//! `std::sync::Mutex` acquired deep inside `apply_upsert_points` /
//! `apply_delete_points`, after the request has already taken a
//! `WriteSubmitter` permit, a write blocking-admission slot, and a Tokio
//! blocking thread. Under bursty same-collection writes, canceled clients can
//! retry while their first blocking mutation is still running. Without async
//! admission those duplicates occupy more blocking workers and queue behind
//! the synchronous mutex, starving unrelated collections on the pod.
//!
//! This module moves the serialization point to async context so waiting
//! requests for both LMDB and LSM park on a per-collection
//! `tokio::sync::Mutex` before blocking submission. Different collections use
//! different gates and remain fully parallel. For upserts and deletes, the
//! storage-layer mutex remains a defense-in-depth safety boundary; this gate
//! prevents duplicate same-collection work from reaching it concurrently.

use crate::helix_engine::storage_core::backend::BackendKind;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};
use std::time::Instant;

use tokio::sync::{Mutex, OwnedMutexGuard};

/// Process-wide map of per-collection async point-mutation gates. Entries
/// are created lazily on first request and never removed; each entry costs
/// a `String` + `Arc<Mutex<()>>` so the steady-state memory cost is bound
/// by the number of collections that have ever served a point mutation.
static GATES: LazyLock<RwLock<HashMap<String, Arc<Mutex<()>>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

fn gate_for(collection: &str) -> Arc<Mutex<()>> {
    if let Ok(map) = GATES.read() {
        if let Some(gate) = map.get(collection) {
            return Arc::clone(gate);
        }
    }
    let mut map = match GATES.write() {
        Ok(map) => map,
        // Poisoned lock: hand out a one-shot orphan so the request can
        // proceed; the std mutex inside apply_*_points will still
        // serialize correctly.
        Err(_) => return Arc::new(Mutex::new(())),
    };
    Arc::clone(
        map.entry(collection.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(()))),
    )
}

/// Cloneable RAII guard returned by [`acquire_for_backend`]. Every clone shares
/// one owned Tokio mutex guard, so async request cancellation cannot release
/// admission while a blocking worker clone still executes the mutation. The
/// underlying permit and inflight gauge are released exactly once when the last
/// clone drops.
#[derive(Clone)]
pub struct PointMutationGuard {
    _inner: Arc<PointMutationGuardInner>,
}

struct PointMutationGuardInner {
    _inner: OwnedMutexGuard<()>,
    backend: &'static str,
    collection: String,
}

impl Drop for PointMutationGuardInner {
    fn drop(&mut self) {
        metrics::gauge!(
            "helix_point_mutation_gate_inflight",
            "backend" => self.backend,
            "collection" => self.collection.clone()
        )
        .decrement(1.0);
    }
}

/// Acquire the async point-mutation gate for `collection`. Awaits in place if a
/// same-collection mutation is already in flight; the wait is cheap because it
/// parks the future without consuming a write permit or blocking thread.
pub async fn acquire_for_backend(kind: BackendKind, collection: &str) -> PointMutationGuard {
    let backend = backend_label(kind);
    let gate = gate_for(collection);
    let started = Instant::now();
    metrics::counter!(
        "helix_point_mutation_gate_attempts_total",
        "backend" => backend,
        "collection" => collection.to_string()
    )
    .increment(1);
    let inner = gate.lock_owned().await;
    let waited_ms = started.elapsed().as_secs_f64() * 1000.0;
    metrics::histogram!(
        "helix_point_mutation_gate_wait_ms",
        "backend" => backend,
        "collection" => collection.to_string()
    )
    .record(waited_ms);
    metrics::gauge!(
        "helix_point_mutation_gate_inflight",
        "backend" => backend,
        "collection" => collection.to_string()
    )
    .increment(1.0);
    PointMutationGuard {
        _inner: Arc::new(PointMutationGuardInner {
            _inner: inner,
            backend,
            collection: collection.to_string(),
        }),
    }
}

fn backend_label(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::Lmdb => "lmdb",
        BackendKind::Lsm => "lsm",
    }
}

/// Returns the collection name when `(method, path)` targets a point
/// mutation route. Gateway-side admission parks same-collection mutations in
/// async context before they occupy the blocking pool. Upserts and deletes also
/// retain their storage-layer per-collection write gate as defense in depth.
///
/// Routes covered:
///   * `PUT  /collections/{name}/points`           — upsert
///   * `POST /collections/{name}/points/delete`    — delete by ids/filter
///   * `POST /collections/{name}/points/payload`   — payload patch
///
/// Other write routes (`/index`, snapshot ops, ingest stream) take different
/// code paths and intentionally bypass this admission layer.
pub fn point_mutation_collection<'a>(method: &str, path: &'a str) -> Option<&'a str> {
    let path = path.split('?').next().unwrap_or(path);
    let rest = path.strip_prefix("/collections/")?;
    let (name, suffix) = match rest.find('/') {
        Some(idx) => (&rest[..idx], &rest[idx..]),
        None => (rest, ""),
    };
    if name.is_empty() {
        return None;
    }
    match (method, suffix) {
        ("PUT", "/points") => Some(name),
        ("POST", "/points/delete") => Some(name),
        ("POST", "/points/payload") => Some(name),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_upsert_route() {
        assert_eq!(
            point_mutation_collection("PUT", "/collections/foo/points"),
            Some("foo")
        );
        assert_eq!(
            point_mutation_collection("PUT", "/collections/foo/points?wait=true"),
            Some("foo")
        );
    }

    #[test]
    fn detects_delete_route() {
        assert_eq!(
            point_mutation_collection("POST", "/collections/foo/points/delete"),
            Some("foo")
        );
    }

    #[test]
    fn ignores_unrelated_routes() {
        assert!(point_mutation_collection("POST", "/collections/foo/points/search").is_none());
        assert!(point_mutation_collection("POST", "/collections/foo/points/scroll").is_none());
        assert!(point_mutation_collection("PUT", "/collections/foo/index").is_none());
        assert!(point_mutation_collection("GET", "/collections/foo/points").is_none());
        assert!(point_mutation_collection("PUT", "/collections/").is_none());
        assert!(point_mutation_collection("PUT", "/health").is_none());
    }

    #[test]
    fn detects_payload_patch_route() {
        assert_eq!(
            point_mutation_collection("POST", "/collections/foo/points/payload"),
            Some("foo")
        );
        assert_eq!(
            point_mutation_collection("POST", "/collections/foo/points/payload?wait=true"),
            Some("foo")
        );
    }

    #[tokio::test]
    async fn serializes_same_collection() {
        let _g1 = acquire_for_backend(BackendKind::Lmdb, "test_serialize_same").await;
        let acquired_second = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            acquire_for_backend(BackendKind::Lmdb, "test_serialize_same"),
        )
        .await
        .is_ok();
        assert!(!acquired_second, "second acquire must wait for first guard");
        drop(_g1);
        // After drop, a fresh acquire must succeed promptly.
        let _g2 = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            acquire_for_backend(BackendKind::Lmdb, "test_serialize_same"),
        )
        .await
        .expect("second acquire should proceed after drop");
    }

    #[tokio::test]
    async fn lsm_serializes_same_collection() {
        let first = acquire_for_backend(BackendKind::Lsm, "test_lsm_serialize").await;
        let acquired_second = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            acquire_for_backend(BackendKind::Lsm, "test_lsm_serialize"),
        )
        .await
        .is_ok();
        assert!(
            !acquired_second,
            "second LSM mutation must park asynchronously"
        );

        // `timeout` canceled the queued acquisition. After the active mutation
        // completes, a new request must acquire immediately rather than being
        // stuck behind the abandoned waiter.
        drop(first);
        tokio::time::timeout(
            std::time::Duration::from_millis(50),
            acquire_for_backend(BackendKind::Lsm, "test_lsm_serialize"),
        )
        .await
        .expect("next LSM mutation should proceed after the active guard drops");
    }

    #[tokio::test]
    async fn cancelled_request_keeps_lsm_gate_closed_while_worker_clone_runs() {
        let request_guard =
            acquire_for_backend(BackendKind::Lsm, "test_lsm_active_cancellation").await;
        let worker_guard = request_guard.clone();

        // Model the HTTP/submit future being canceled after spawn_blocking has
        // received its clone. The active worker must remain the sole admitted
        // mutation for this collection.
        drop(request_guard);
        let overlapped = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            acquire_for_backend(BackendKind::Lsm, "test_lsm_active_cancellation"),
        )
        .await
        .is_ok();
        assert!(
            !overlapped,
            "HTTP cancellation must not admit a retry behind active blocking work"
        );

        drop(worker_guard);
        tokio::time::timeout(
            std::time::Duration::from_millis(50),
            acquire_for_backend(BackendKind::Lsm, "test_lsm_active_cancellation"),
        )
        .await
        .expect("retry should proceed after the blocking worker finishes");
    }

    #[tokio::test]
    async fn does_not_serialize_different_collections() {
        let _g1 = acquire_for_backend(BackendKind::Lsm, "test_diff_a").await;
        let _g2 = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            acquire_for_backend(BackendKind::Lsm, "test_diff_b"),
        )
        .await
        .expect("different collection must not block");
    }
}
