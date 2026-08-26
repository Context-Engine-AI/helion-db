//! Opt-in per-collection in-flight request cap.
//!
//! Protects noisy-neighbor scenarios where one tenant's hot collection can
//! monopolize the shared worker pool. When the cap is exceeded, new requests
//! for that collection receive HTTP 429 with `Retry-After`; other collections
//! continue to serve traffic.
//!
//! Default behaviour: **disabled** (cap = 0). Set `HELIX_PER_COLLECTION_INFLIGHT_CAP`
//! to a positive integer to enable. This is a load-shedding tool — setting it
//! too low will surface as 429s under bursty writes.

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, LazyLock, RwLock,
};

/// Configured per-collection cap. `0` ⇒ limiter disabled (all requests pass).
/// Parsed once at process start.
fn per_collection_cap() -> usize {
    static CAP: LazyLock<usize> = LazyLock::new(|| {
        std::env::var("HELIX_PER_COLLECTION_INFLIGHT_CAP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
    });
    *CAP
}

/// Process-wide limiter. Uses an `RwLock<HashMap>` — reads dominate after the
/// first request per collection, and the inner counter is atomic, so the
/// read-lock path is wait-free on the hot axis.
static LIMITER: LazyLock<InflightLimiter> = LazyLock::new(InflightLimiter::new);

struct InflightLimiter {
    counters: RwLock<HashMap<String, Arc<AtomicUsize>>>,
}

impl InflightLimiter {
    fn new() -> Self {
        Self {
            counters: RwLock::new(HashMap::new()),
        }
    }

    fn get_or_insert(&self, collection: &str) -> Arc<AtomicUsize> {
        if let Ok(map) = self.counters.read() {
            if let Some(c) = map.get(collection) {
                return Arc::clone(c);
            }
        }
        // Write-lock slow path — runs at most once per collection lifetime.
        let mut map = match self.counters.write() {
            Ok(guard) => guard,
            // A poisoned lock here shouldn't bring down the request; we fall
            // back to an orphan counter (no cap enforcement) so traffic still
            // flows while ops investigates.
            Err(_) => return Arc::new(AtomicUsize::new(0)),
        };
        Arc::clone(
            map.entry(collection.to_string())
                .or_insert_with(|| Arc::new(AtomicUsize::new(0))),
        )
    }
}

/// RAII guard: decrements the in-flight counter on drop. Always drop order
/// matches request scope — a handler panic still releases the slot because
/// `Drop` runs during stack unwind.
pub struct InflightGuard {
    counter: Arc<AtomicUsize>,
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Outcome of `try_acquire`.
pub enum AcquireOutcome {
    /// Limiter disabled or limit not reached. The returned guard decrements
    /// on drop.
    Admitted(InflightGuard),
    /// Limit reached. The caller should respond 429 and not run the handler.
    /// Carries the current in-flight count for logging / metrics.
    Rejected { current: usize, cap: usize },
}

/// Attempt to admit one more in-flight request for `collection`. When the
/// limiter is disabled (`HELIX_PER_COLLECTION_INFLIGHT_CAP=0` or unset) this
/// is a cheap `Admitted` with no counter bump.
pub fn try_acquire_with_cap(collection: &str, cap: usize) -> AcquireOutcome {
    if cap == 0 {
        // Return a dummy guard that doesn't touch any shared state. This
        // keeps the hot path allocation-free when the feature is disabled.
        static NOOP: LazyLock<Arc<AtomicUsize>> = LazyLock::new(|| Arc::new(AtomicUsize::new(0)));
        NOOP.fetch_add(1, Ordering::Relaxed);
        return AcquireOutcome::Admitted(InflightGuard {
            counter: Arc::clone(&NOOP),
        });
    }
    let counter = LIMITER.get_or_insert(collection);
    // Strict CAS gate: route caps are used to protect LMDB/handler safety
    // under retry storms, so do not briefly overshoot the cap.
    let mut current = counter.load(Ordering::Acquire);
    loop {
        if current >= cap {
            // Emit a counter so ops can see 429 pressure per collection. Label
            // cardinality is bounded by the configured collection count.
            metrics::counter!(
                "helix_inflight_rejected_total",
                "collection" => collection.to_string()
            )
            .increment(1);
            return AcquireOutcome::Rejected { current, cap };
        }
        match counter.compare_exchange_weak(
            current,
            current + 1,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
    let admitted = current + 1;
    if admitted > cap {
        // Emit a counter so ops can see 429 pressure per collection. Label
        // cardinality is bounded by the configured collection count.
        metrics::counter!(
            "helix_inflight_rejected_total",
            "collection" => collection.to_string()
        )
        .increment(1);
        counter.fetch_sub(1, Ordering::AcqRel);
        return AcquireOutcome::Rejected {
            current: admitted,
            cap,
        };
    }
    metrics::gauge!(
        "helix_inflight_current",
        "collection" => collection.to_string()
    )
    .set(admitted as f64);
    AcquireOutcome::Admitted(InflightGuard { counter })
}

pub fn try_acquire(collection: &str) -> AcquireOutcome {
    try_acquire_with_cap(collection, per_collection_cap())
}

/// Drop the per-collection counter when a collection is deleted so the map
/// doesn't accumulate dead entries under tenant churn. The `Arc<AtomicUsize>`
/// stays alive as long as any outstanding `InflightGuard` references it, so
/// in-flight requests still release their slot correctly.
pub fn forget(collection: &str) {
    if let Ok(mut map) = LIMITER.counters.write() {
        map.remove(collection);
    }
}

/// Extract the `{name}` segment from a Qdrant-compatible `/collections/{name}/…`
/// path or native `/v1/collections/{name}/…` path. Returns `None` for
/// non-collection routes; callers skip the limiter in that case.
pub fn collection_from_path(path: &str) -> Option<&str> {
    // Strip query string.
    let path = path.split('?').next().unwrap_or(path);
    let rest = path
        .strip_prefix("/collections/")
        .or_else(|| path.strip_prefix("/v1/collections/"))?;
    // The collection name ends at the first '/' or end of string.
    let end = rest.find('/').unwrap_or(rest.len());
    let name = &rest[..end];
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_collection_names() {
        assert_eq!(collection_from_path("/collections/foo"), Some("foo"));
        assert_eq!(collection_from_path("/v1/collections/foo"), Some("foo"));
        assert_eq!(
            collection_from_path("/collections/foo/points/search"),
            Some("foo")
        );
        assert_eq!(
            collection_from_path("/v1/collections/foo/points/scan"),
            Some("foo")
        );
        assert_eq!(collection_from_path("/collections/"), None);
        assert_eq!(collection_from_path("/v1/collections/"), None);
        assert_eq!(collection_from_path("/health"), None);
        assert_eq!(
            collection_from_path("/collections/foo?wait=true"),
            Some("foo")
        );
        assert_eq!(
            collection_from_path("/v1/collections/foo/points/scan?partition=1"),
            Some("foo")
        );
    }

    #[test]
    fn explicit_cap_rejects_when_collection_is_full() {
        let collection = "unit-test-explicit-cap-rejects";
        let first = try_acquire_with_cap(collection, 1);
        match first {
            AcquireOutcome::Admitted(_guard) => {
                let second = try_acquire_with_cap(collection, 1);
                match second {
                    AcquireOutcome::Rejected { current, cap } => {
                        assert_eq!(current, 1);
                        assert_eq!(cap, 1);
                    }
                    AcquireOutcome::Admitted(_) => panic!("second request should be rejected"),
                }
            }
            AcquireOutcome::Rejected { .. } => panic!("first request should be admitted"),
        }
        forget(collection);
    }

    #[test]
    fn zero_cap_disables_rejection() {
        let collection = "unit-test-zero-cap";
        let _first = try_acquire_with_cap(collection, 0);
        let second = try_acquire_with_cap(collection, 0);
        match second {
            AcquireOutcome::Admitted(_guard) => {}
            AcquireOutcome::Rejected { .. } => panic!("zero cap should not reject"),
        }
        forget(collection);
    }

    #[test]
    fn explicit_cap_is_strict_under_concurrency() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Barrier};
        use std::thread;
        use std::time::Duration;

        let collection = "unit-test-strict-concurrent-cap";
        forget(collection);
        let admitted = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(16));

        let handles: Vec<_> = (0..16)
            .map(|_| {
                let admitted = Arc::clone(&admitted);
                let max_seen = Arc::clone(&max_seen);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    if let AcquireOutcome::Admitted(_guard) = try_acquire_with_cap(collection, 1) {
                        let now = admitted.fetch_add(1, Ordering::AcqRel) + 1;
                        max_seen.fetch_max(now, Ordering::AcqRel);
                        thread::sleep(Duration::from_millis(5));
                        admitted.fetch_sub(1, Ordering::AcqRel);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(max_seen.load(Ordering::Acquire), 1);
        forget(collection);
    }
}
