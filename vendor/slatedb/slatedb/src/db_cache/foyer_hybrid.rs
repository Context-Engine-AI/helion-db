//! # Foyer Hybrid Cache
//!
//! This module provides an implementation of a hybrid (in-memory + on-disk) cache using the Foyer
//! library. The cache is designed to store and retrieve cached blocks, indexes, and filters
//! associated with SSTable IDs.
//!
//! ## Features
//!
//! - **Hybrid Cache**: Caches blocks using a tiered cache that stores data across memory and
//!   local disk.
//! - **Custom Weigher**: Implements a custom weigher to account for the size of cached blocks.
//! - **Flexible Configuration**: The HybridCache instance is passed to the cache, so you are free
//!   to configure it as needed for your use case.
//!
//! ## Notes
//! Foyer HybridCache manages its own memory and disk cache, so when it coexists with `CachedObjectStore`
//! there may be some duplication of cached data. In other words, disk cache may suffer from write
//! amplification. If you don't want to suffer from write amplification, then we recommend that you do not
//! coexist Foyer HybridCache with `CachedObjectStore`.
//!
//! Benefits of coexistence:
//! `CachedObjectStore` provides prefetching, which may speed up your read performance and
//! reduce unnecessary object store accesses.
//!
//! Drawbacks of coexistence:
//! Disk may suffer from write amplification. Data will first be written to `CachedObjectStore`, then
//! flow into the memory part of Foyer HybridCache, and finally written back to the disk area managed
//! by Foyer HybridCache itself. slateDB cannot control Foyer HybridCache to write data to its disk area.
//!
//! ## Examples
//!
//! ```
//! use foyer::{
//!     BlockEngineConfig, DeviceBuilder, FsDeviceBuilder, HybridCacheBuilder, PsyncIoEngineConfig,
//! };
//! use slatedb::Db;
//! use slatedb::db_cache::CachedEntry;
//! use slatedb::db_cache::foyer_hybrid::FoyerHybridCache;
//! use slatedb::object_store::memory::InMemory;
//! use std::sync::Arc;
//!
//! #[::tokio::main]
//! async fn main() {
//!     let object_store = Arc::new(InMemory::new());
//!     let cache = HybridCacheBuilder::new()
//!         .with_name("hybrid_cache")
//!         .memory(1024 * 1024)
//!         .with_weighter(|_, v: &CachedEntry| v.size())
//!         .storage()
//!         .with_io_engine_config(PsyncIoEngineConfig::new())
//!         .with_engine_config(
//!             BlockEngineConfig::new(
//!                 FsDeviceBuilder::new("/tmp/slatedb-cache")
//!                     .with_capacity(1024 * 1024)
//!                     .build()
//!                     .unwrap(),
//!             )
//!             .with_block_size(64 * 1024),
//!         )
//!         .build()
//!         .await
//!         .unwrap();
//!     let cache = Arc::new(FoyerHybridCache::new_with_cache(cache));
//!     let db = Db::builder("path/to/db", object_store)
//!         .with_db_cache(cache)
//!         .build()
//!         .await
//!         .unwrap();
//! }
//! ```
//!

use crate::{
    db_cache::{
        CacheLoader, CacheUsageSnapshot, CachedEntry, CachedKey, DbCache, DbCacheUsageSnapshot,
    },
    error::SlateDBError,
    utils::format_bytes_si,
};
use async_trait::async_trait;
use log::info;
use mixtrics::{
    metrics::{
        BoxedCounterVec, BoxedGauge, BoxedGaugeVec, BoxedHistogramVec, GaugeOps, GaugeVecOps,
        RegistryOps,
    },
    registry::noop::NoopMetricsRegistry,
};
use std::{
    borrow::Cow,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

const FOYER_BLOCK_GAUGE: &str = "foyer_storage_block_engine_block";
const FOYER_BLOCK_SIZE_GAUGE: &str = "foyer_storage_block_engine_block_size_bytes";

#[derive(Debug, Default)]
struct FoyerHybridCacheMetricValues {
    clean_blocks: AtomicU64,
    writing_blocks: AtomicU64,
    evictable_blocks: AtomicU64,
    reclaiming_blocks: AtomicU64,
    block_size_bytes: AtomicU64,
}

/// Retained Foyer block-engine gauges used for synchronous cache accounting.
///
/// Install [`Self::registry`] on the same `HybridCacheBuilder` whose cache is
/// passed to [`FoyerHybridCache::new_with_cache_and_metrics`].
#[derive(Clone, Debug, Default)]
pub struct FoyerHybridCacheMetrics {
    values: Arc<FoyerHybridCacheMetricValues>,
}

impl FoyerHybridCacheMetrics {
    /// Build an empty metrics handle.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the registry passed to `HybridCacheBuilder::with_metrics_registry`.
    pub fn registry(&self) -> mixtrics::metrics::BoxedRegistry {
        Box::new(FoyerCacheMetricsRegistry {
            values: Arc::clone(&self.values),
        })
    }

    fn disk_usage_snapshot(&self) -> CacheUsageSnapshot {
        let block_size_bytes = self.values.block_size_bytes.load(Ordering::Relaxed);
        if block_size_bytes == 0 {
            return CacheUsageSnapshot::Unavailable;
        }
        let clean_blocks = self.values.clean_blocks.load(Ordering::Relaxed);
        let writing_blocks = self.values.writing_blocks.load(Ordering::Relaxed);
        let evictable_blocks = self.values.evictable_blocks.load(Ordering::Relaxed);
        let reclaiming_blocks = self.values.reclaiming_blocks.load(Ordering::Relaxed);
        let used_blocks = writing_blocks
            .saturating_add(evictable_blocks)
            .saturating_add(reclaiming_blocks);
        let total_blocks = clean_blocks.saturating_add(used_blocks);
        CacheUsageSnapshot::Ready {
            used_bytes: used_blocks.saturating_mul(block_size_bytes),
            capacity_bytes: Some(total_blocks.saturating_mul(block_size_bytes)),
        }
    }
}

#[derive(Debug)]
struct FoyerCacheMetricsRegistry {
    values: Arc<FoyerHybridCacheMetricValues>,
}

impl RegistryOps for FoyerCacheMetricsRegistry {
    fn register_counter_vec(
        &self,
        _name: Cow<'static, str>,
        _desc: Cow<'static, str>,
        _label_names: &'static [&'static str],
    ) -> BoxedCounterVec {
        Box::new(NoopMetricsRegistry)
    }

    fn register_gauge_vec(
        &self,
        name: Cow<'static, str>,
        _desc: Cow<'static, str>,
        _label_names: &'static [&'static str],
    ) -> BoxedGaugeVec {
        let kind = match name.as_ref() {
            FOYER_BLOCK_GAUGE => TrackedGaugeKind::BlockState,
            FOYER_BLOCK_SIZE_GAUGE => TrackedGaugeKind::BlockSize,
            _ => return Box::new(NoopMetricsRegistry),
        };
        Box::new(FoyerTrackedGaugeVec {
            kind,
            values: Arc::clone(&self.values),
        })
    }

    fn register_histogram_vec(
        &self,
        _name: Cow<'static, str>,
        _desc: Cow<'static, str>,
        _label_names: &'static [&'static str],
    ) -> BoxedHistogramVec {
        Box::new(NoopMetricsRegistry)
    }

    fn register_histogram_vec_with_buckets(
        &self,
        _name: Cow<'static, str>,
        _desc: Cow<'static, str>,
        _label_names: &'static [&'static str],
        _buckets: Vec<f64>,
    ) -> BoxedHistogramVec {
        Box::new(NoopMetricsRegistry)
    }
}

#[derive(Debug, Clone, Copy)]
enum TrackedGaugeKind {
    BlockState,
    BlockSize,
}

#[derive(Debug)]
struct FoyerTrackedGaugeVec {
    kind: TrackedGaugeKind,
    values: Arc<FoyerHybridCacheMetricValues>,
}

impl GaugeVecOps for FoyerTrackedGaugeVec {
    fn gauge(&self, labels: &[Cow<'static, str>]) -> BoxedGauge {
        Box::new(AtomicGauge {
            value: Arc::clone(&self.values),
            field: match self.kind {
                TrackedGaugeKind::BlockSize => AtomicGaugeField::BlockSize,
                TrackedGaugeKind::BlockState => match labels.get(1).map(Cow::as_ref) {
                    Some("clean") => AtomicGaugeField::Clean,
                    Some("writing") => AtomicGaugeField::Writing,
                    Some("evictable") => AtomicGaugeField::Evictable,
                    Some("reclaiming") => AtomicGaugeField::Reclaiming,
                    _ => return Box::new(NoopMetricsRegistry),
                },
            },
        })
    }
}

#[derive(Debug, Clone, Copy)]
enum AtomicGaugeField {
    Clean,
    Writing,
    Evictable,
    Reclaiming,
    BlockSize,
}

#[derive(Debug)]
struct AtomicGauge {
    value: Arc<FoyerHybridCacheMetricValues>,
    field: AtomicGaugeField,
}

impl AtomicGauge {
    fn atomic(&self) -> &AtomicU64 {
        match self.field {
            AtomicGaugeField::Clean => &self.value.clean_blocks,
            AtomicGaugeField::Writing => &self.value.writing_blocks,
            AtomicGaugeField::Evictable => &self.value.evictable_blocks,
            AtomicGaugeField::Reclaiming => &self.value.reclaiming_blocks,
            AtomicGaugeField::BlockSize => &self.value.block_size_bytes,
        }
    }
}

impl GaugeOps for AtomicGauge {
    fn increase(&self, value: u64) {
        let _ = self
            .atomic()
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_add(value))
            });
    }

    fn decrease(&self, value: u64) {
        let _ = self
            .atomic()
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(value))
            });
    }

    fn absolute(&self, value: u64) {
        self.atomic().store(value, Ordering::Relaxed);
    }
}

pub struct FoyerHybridCache {
    inner: foyer::HybridCache<CachedKey, CachedEntry>,
    metrics: Option<FoyerHybridCacheMetrics>,
}

impl FoyerHybridCache {
    pub fn new_with_cache(cache: foyer::HybridCache<CachedKey, CachedEntry>) -> Self {
        Self {
            inner: cache,
            metrics: None,
        }
    }

    /// Build a hybrid cache with retained block-engine accounting.
    pub fn new_with_cache_and_metrics(
        cache: foyer::HybridCache<CachedKey, CachedEntry>,
        metrics: FoyerHybridCacheMetrics,
    ) -> Self {
        Self {
            inner: cache,
            metrics: Some(metrics),
        }
    }
}

impl FoyerHybridCache {
    async fn get(&self, key: &CachedKey) -> Result<Option<CachedEntry>, crate::Error> {
        self.inner
            .get(key)
            .await
            .map_err(|e| SlateDBError::from(e).into())
            .map(|maybe_v| maybe_v.map(|v| v.value().clone()))
    }
}

#[async_trait]
impl DbCache for FoyerHybridCache {
    async fn get_block(&self, key: &CachedKey) -> Result<Option<CachedEntry>, crate::Error> {
        self.get(key).await
    }

    async fn get_index(&self, key: &CachedKey) -> Result<Option<CachedEntry>, crate::Error> {
        self.get(key).await
    }

    async fn get_filter(&self, key: &CachedKey) -> Result<Option<CachedEntry>, crate::Error> {
        self.get(key).await
    }

    async fn get_stats(&self, key: &CachedKey) -> Result<Option<CachedEntry>, crate::Error> {
        self.get(key).await
    }

    async fn insert(&self, key: CachedKey, value: CachedEntry) {
        self.inner.insert(key, value);
    }

    async fn remove(&self, key: &CachedKey) {
        self.inner.remove(key);
    }

    fn entry_count(&self) -> u64 {
        // foyer cache doesn't support an entry count estimate
        0
    }

    fn usage_snapshot(&self) -> DbCacheUsageSnapshot {
        DbCacheUsageSnapshot {
            memory: CacheUsageSnapshot::Ready {
                used_bytes: self.inner.memory().usage() as u64,
                capacity_bytes: Some(self.inner.memory().capacity() as u64),
            },
            disk: self
                .metrics
                .as_ref()
                .map_or(CacheUsageSnapshot::Unavailable, |metrics| {
                    metrics.disk_usage_snapshot()
                }),
        }
    }

    async fn close(&self) -> Result<(), crate::Error> {
        let memory_bytes = self.inner.memory().usage();
        info!(
            "foyer hybrid cache: closing \
             (flushing {} from memory to disk)",
            format_bytes_si(memory_bytes as u64)
        );
        let result = self
            .inner
            .close()
            .await
            .map_err(|e| crate::Error::from(SlateDBError::from(e)));
        match &result {
            Ok(()) => info!("foyer hybrid cache: closed successfully"),
            Err(e) => info!("foyer hybrid cache: close failed [error={e:?}]"),
        }
        result
    }

    async fn fetch_block(
        &self,
        key: CachedKey,
        loader: CacheLoader,
    ) -> Result<CachedEntry, crate::Error> {
        self.dedup_fetch(key, loader).await
    }

    async fn fetch_index(
        &self,
        key: CachedKey,
        loader: CacheLoader,
    ) -> Result<CachedEntry, crate::Error> {
        self.dedup_fetch(key, loader).await
    }

    async fn fetch_filter(
        &self,
        key: CachedKey,
        loader: CacheLoader,
    ) -> Result<CachedEntry, crate::Error> {
        self.dedup_fetch(key, loader).await
    }

    async fn fetch_stats(
        &self,
        key: CachedKey,
        loader: CacheLoader,
    ) -> Result<CachedEntry, crate::Error> {
        self.dedup_fetch(key, loader).await
    }
}

impl FoyerHybridCache {
    /// Use foyer's `HybridCache::get_or_fetch`, which deduplicates concurrent loads across
    /// memory + disk tiers. See [`crate::db_cache::foyer::FoyerCache::dedup_fetch`] for the
    /// error-handling rationale.
    async fn dedup_fetch(
        &self,
        key: CachedKey,
        loader: CacheLoader,
    ) -> Result<CachedEntry, crate::Error> {
        let fetch = self
            .inner
            .get_or_fetch(&key, move || async move { loader().await });
        match fetch.await {
            Ok(entry) => Ok(entry.value().clone()),
            Err(err) => Err(SlateDBError::FoyerError(Arc::new(err)).into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FoyerCacheMetricsRegistry, FoyerHybridCache, FoyerHybridCacheMetrics, FOYER_BLOCK_GAUGE,
        FOYER_BLOCK_SIZE_GAUGE,
    };
    use crate::db_cache::{CacheUsageSnapshot, CachedEntry, CachedKey, DbCache};
    use crate::db_state::SsTableId;
    use crate::format::sst::BlockBuilder;
    use foyer::{
        BlockEngineConfig, DeviceBuilder, FsDeviceBuilder, HybridCacheBuilder, PsyncIoEngineConfig,
    };
    use rand::RngCore;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tempfile::{tempdir, TempDir};

    const SST_ID: SsTableId = SsTableId::Wal(123);

    #[test]
    fn tracks_pinned_foyer_block_metrics() {
        use mixtrics::metrics::RegistryOps;
        use std::borrow::Cow;

        let metrics = FoyerHybridCacheMetrics::new();
        let registry = FoyerCacheMetricsRegistry {
            values: Arc::clone(&metrics.values),
        };
        let block_states = registry.register_gauge_vec(
            Cow::Borrowed(FOYER_BLOCK_GAUGE),
            Cow::Borrowed(""),
            &["name", "type"],
        );
        block_states
            .gauge(&[Cow::Borrowed("test"), Cow::Borrowed("clean")])
            .absolute(5);
        block_states
            .gauge(&[Cow::Borrowed("test"), Cow::Borrowed("writing")])
            .absolute(1);
        block_states
            .gauge(&[Cow::Borrowed("test"), Cow::Borrowed("evictable")])
            .absolute(2);
        block_states
            .gauge(&[Cow::Borrowed("test"), Cow::Borrowed("reclaiming")])
            .absolute(1);
        registry
            .register_gauge_vec(
                Cow::Borrowed(FOYER_BLOCK_SIZE_GAUGE),
                Cow::Borrowed(""),
                &["name"],
            )
            .gauge(&[Cow::Borrowed("test")])
            .absolute(4096);

        assert_eq!(
            metrics.disk_usage_snapshot(),
            CacheUsageSnapshot::Ready {
                used_bytes: 4 * 4096,
                capacity_bytes: Some(9 * 4096),
            }
        );
    }

    #[tokio::test]
    async fn test_hybrid_cache() {
        let (cache, _dir) = setup().await;
        let mut items = HashMap::new();
        for b in 0u64..256 {
            let k = CachedKey::from((SST_ID, b));
            let v = build_block();
            cache.insert(k.clone(), v.clone()).await;
            items.insert(k, v);
        }
        let mut found = 0;
        let mut notfound = 0;
        for (k, v) in items {
            let cached_v = cache.get_block(&k).await.unwrap();
            if let Some(cached_v) = cached_v {
                assert!(v.block().unwrap().as_ref() == cached_v.block().unwrap().as_ref());
                found += 1;
            } else {
                notfound += 1;
            }
        }
        println!("found: {}, notfound: {}", found, notfound);
        assert!(found > 0);
    }

    fn build_block() -> CachedEntry {
        let mut rng = rand::rng();
        let mut builder = BlockBuilder::new_latest(1024);
        loop {
            let mut k = vec![0u8; 32];
            rng.fill_bytes(&mut k);
            let mut v = vec![0u8; 128];
            rng.fill_bytes(&mut v);
            if builder.add_value(&k, &v, None, None) {
                break;
            }
        }
        let block = Arc::new(builder.build().unwrap());
        CachedEntry::with_block(block)
    }

    async fn setup() -> (FoyerHybridCache, TempDir) {
        let tempdir = tempdir().unwrap();
        let cache = open_cache(tempdir.path()).await;
        (cache, tempdir)
    }

    async fn open_cache(path: &std::path::Path) -> FoyerHybridCache {
        let cache = HybridCacheBuilder::new()
            .with_name("hybrid_cache_test")
            .memory(1024 * 1024)
            .with_weighter(|_, v: &CachedEntry| v.size())
            .storage()
            .with_io_engine_config(PsyncIoEngineConfig::new())
            .with_engine_config(
                BlockEngineConfig::new(
                    FsDeviceBuilder::new(path)
                        .with_capacity(4 * 1024 * 1024)
                        .build()
                        .unwrap(),
                )
                .with_block_size(64 * 1024),
            )
            .build()
            .await
            .unwrap();
        FoyerHybridCache::new_with_cache(cache)
    }

    #[tokio::test]
    async fn should_persist_blocks_to_disk_on_close() {
        // given: a cache with entries
        let dir = tempdir().unwrap();
        let cache = open_cache(dir.path()).await;
        let mut keys = Vec::new();
        for b in 0u64..64 {
            let k = CachedKey::from((SST_ID, b));
            cache.insert(k.clone(), build_block()).await;
            keys.push(k);
        }

        // when: close (flush to disk) and reopen
        cache.close().await.unwrap();
        drop(cache);
        let cache = open_cache(dir.path()).await;

        // then: all entries should be recoverable from disk
        let mut found = 0;
        for k in &keys {
            if cache.get_block(k).await.unwrap().is_some() {
                found += 1;
            }
        }
        assert_eq!(found, keys.len(), "all entries should survive close+reopen");
    }
}
