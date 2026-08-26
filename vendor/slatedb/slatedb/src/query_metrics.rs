//! Request-scoped storage metrics for foreground query execution.
//!
//! The observer is installed with [`scope_query_metrics`]. Storage paths record
//! against the current task scope without reading or subtracting process-wide
//! metrics, so concurrent queries cannot contaminate each other's counters.

use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

tokio::task_local! {
    static QUERY_METRICS: QueryMetricsObserver;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryCacheKind {
    Block,
    Object,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QueryCacheStatistics {
    pub hits: u64,
    pub misses: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QueryMetricsSnapshot {
    pub block_cache: QueryCacheStatistics,
    pub object_cache: QueryCacheStatistics,
    pub object_storage_reads: u64,
}

#[derive(Clone, Debug, Default)]
pub struct QueryMetricsObserver {
    inner: Arc<QueryMetricsInner>,
}

#[derive(Debug, Default)]
struct QueryMetricsInner {
    block_cache_hits: AtomicU64,
    block_cache_misses: AtomicU64,
    object_cache_hits: AtomicU64,
    object_cache_misses: AtomicU64,
    object_storage_reads: AtomicU64,
}

impl QueryMetricsObserver {
    #[must_use]
    pub fn snapshot(&self) -> QueryMetricsSnapshot {
        QueryMetricsSnapshot {
            block_cache: QueryCacheStatistics {
                hits: self.inner.block_cache_hits.load(Ordering::Relaxed),
                misses: self.inner.block_cache_misses.load(Ordering::Relaxed),
            },
            object_cache: QueryCacheStatistics {
                hits: self.inner.object_cache_hits.load(Ordering::Relaxed),
                misses: self.inner.object_cache_misses.load(Ordering::Relaxed),
            },
            object_storage_reads: self.inner.object_storage_reads.load(Ordering::Relaxed),
        }
    }

    fn record_cache(&self, kind: QueryCacheKind, hit: bool) {
        let counter = match (kind, hit) {
            (QueryCacheKind::Block, true) => &self.inner.block_cache_hits,
            (QueryCacheKind::Block, false) => &self.inner.block_cache_misses,
            (QueryCacheKind::Object, true) => &self.inner.object_cache_hits,
            (QueryCacheKind::Object, false) => &self.inner.object_cache_misses,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    fn record_object_storage_read(&self) {
        self.inner
            .object_storage_reads
            .fetch_add(1, Ordering::Relaxed);
    }
}

/// Runs one foreground query with an isolated storage observer.
///
/// ```
/// # tokio_test::block_on(async {
/// use slatedb::{QueryMetricsObserver, scope_query_metrics};
///
/// let observer = QueryMetricsObserver::default();
/// scope_query_metrics(observer.clone(), async {}).await;
/// assert_eq!(observer.snapshot().object_storage_reads, 0);
/// # });
/// ```
pub async fn scope_query_metrics<F>(observer: QueryMetricsObserver, future: F) -> F::Output
where
    F: Future,
{
    QUERY_METRICS.scope(observer, future).await
}

pub(crate) fn record_cache_access(kind: QueryCacheKind, hit: bool) {
    let _ = QUERY_METRICS.try_with(|observer| observer.record_cache(kind, hit));
}

pub(crate) fn record_object_storage_read() {
    let _ = QUERY_METRICS.try_with(QueryMetricsObserver::record_object_storage_read);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn concurrent_scopes_do_not_share_counters() {
        let first = QueryMetricsObserver::default();
        let second = QueryMetricsObserver::default();

        tokio::join!(
            scope_query_metrics(first.clone(), async {
                record_cache_access(QueryCacheKind::Block, true);
                tokio::task::yield_now().await;
                record_object_storage_read();
            }),
            scope_query_metrics(second.clone(), async {
                record_cache_access(QueryCacheKind::Object, false);
                tokio::task::yield_now().await;
                record_cache_access(QueryCacheKind::Object, false);
            }),
        );

        assert_eq!(
            first.snapshot(),
            QueryMetricsSnapshot {
                block_cache: QueryCacheStatistics { hits: 1, misses: 0 },
                object_storage_reads: 1,
                ..QueryMetricsSnapshot::default()
            }
        );
        assert_eq!(
            second.snapshot(),
            QueryMetricsSnapshot {
                object_cache: QueryCacheStatistics { hits: 0, misses: 2 },
                ..QueryMetricsSnapshot::default()
            }
        );
    }

    #[tokio::test]
    async fn records_outside_scope_are_ignored() {
        record_cache_access(QueryCacheKind::Block, false);
        record_object_storage_read();
    }
}
