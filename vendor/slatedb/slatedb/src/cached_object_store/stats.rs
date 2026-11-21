use crate::cached_object_store::storage::PartID;
use object_store::path::Path;
use slatedb_common::metrics::{CounterFn, GaugeFn, MetricsRecorderHelper};
use std::collections::BTreeSet;
use std::fmt::{Debug, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use tokio::sync::Semaphore;

macro_rules! oscache_stat_name {
    ($suffix:expr) => {
        concat!("slatedb.object_store_cache.", $suffix)
    };
}

pub const PART_HIT_COUNT: &str = oscache_stat_name!("part_hit_count");
pub const PART_ACCESS_COUNT: &str = oscache_stat_name!("part_access_count");
pub const CACHE_KEYS: &str = oscache_stat_name!("cache_keys");
pub const CACHE_BYTES: &str = oscache_stat_name!("cache_bytes");
pub const EVICTED_KEYS: &str = oscache_stat_name!("evicted_keys");
pub const EVICTED_BYTES: &str = oscache_stat_name!("evicted_bytes");

#[derive(Clone)]
pub struct CachedObjectStoreStats {
    pub(super) object_store_cache_part_hits: Arc<dyn CounterFn>,
    pub(super) object_store_cache_part_access: Arc<dyn CounterFn>,
    pub(super) object_store_cache_keys: Arc<dyn GaugeFn>,
    pub(super) object_store_cache_bytes: Arc<dyn GaugeFn>,
    pub(super) object_store_cache_evicted_keys: Arc<dyn CounterFn>,
    pub(super) object_store_cache_evicted_bytes: Arc<dyn CounterFn>,
}

impl Debug for CachedObjectStoreStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedObjectStoreStats")
            .field("object_store_cache_part_hits", &"<counter>")
            .field("object_store_cache_part_access", &"<counter>")
            .field("object_store_cache_keys", &"<gauge>")
            .field("object_store_cache_bytes", &"<gauge>")
            .field("object_store_cache_evicted_keys", &"<counter>")
            .field("object_store_cache_evicted_bytes", &"<counter>")
            .finish()
    }
}

impl CachedObjectStoreStats {
    pub(crate) fn new(recorder: &MetricsRecorderHelper) -> Self {
        Self {
            object_store_cache_part_hits: recorder.counter(PART_HIT_COUNT).register(),
            object_store_cache_part_access: recorder.counter(PART_ACCESS_COUNT).register(),
            object_store_cache_keys: recorder.gauge(CACHE_KEYS).register(),
            object_store_cache_bytes: recorder.gauge(CACHE_BYTES).register(),
            object_store_cache_evicted_keys: recorder.counter(EVICTED_KEYS).register(),
            object_store_cache_evicted_bytes: recorder.counter(EVICTED_BYTES).register(),
        }
    }
}

/// HELION LOCAL PATCH: process-wide counters for ranged cache misses
/// (`ObjectStoreCacheOptions::ranged_miss`), summed over every cached object
/// store in the process. Helion exports them on `/metrics`.
#[derive(Debug)]
pub struct RangedMissStats {
    /// Ranged GETs issued for small cache-miss reads (failed ones included).
    pub misses: AtomicU64,
    /// Bytes returned by the ranged GETs that succeeded.
    pub bytes: AtomicU64,
    /// Background full-part fills that fetched their part (saving it is best
    /// effort, as on every cache write).
    pub fills_ok: AtomicU64,
    /// Background fills that failed; the part stays uncached.
    pub fills_err: AtomicU64,
    /// Fills skipped because the in-flight fill budget was spent.
    pub fills_skipped: AtomicU64,
    /// Fills of a database's parts started while a marked batched row read
    /// of that database was in flight ([`Self::batch_row_read`]; Helion's
    /// exact scorer marks its own) and left out: too many fills waited their
    /// turn already, or the budget was spent when the turn came.
    /// `fills_skipped` counts every other skip, as it did: the reads of a
    /// database with no mark in flight, a query's payload reads and a scan's
    /// row batches among them. Either way the read has its bytes and the
    /// part is fetched again by a later read.
    pub fills_skipped_in_batch: AtomicU64,
    /// Small reads that found their part's fill in flight and waited for the
    /// whole part where their own range would have been one small GET.
    pub waits: AtomicU64,
    /// The root of every marked batched row read in flight, one entry a mark.
    batch_row_reads: Mutex<Vec<Path>>,
    /// The turn of the fills started under a mark: one fetches, the others
    /// wait.
    pub(crate) fill_turn: Semaphore,
    /// The parts whose fill waits for that turn.
    pub(crate) fills_waiting: Mutex<BTreeSet<(Path, PartID)>>,
}

/// HELION LOCAL PATCH: a marked batched row read of the database at a root
/// in flight, from [`RangedMissStats::batch_row_read`] until this is dropped.
#[derive(Debug)]
pub struct BatchRowRead(&'static RangedMissStats, Path);

impl Drop for BatchRowRead {
    fn drop(&mut self) {
        let mut roots = self.0.marked_roots();
        if let Some(at) = roots.iter().position(|root| *root == self.1) {
            roots.swap_remove(at);
        }
    }
}

impl RangedMissStats {
    pub(crate) const fn new() -> Self {
        Self {
            misses: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            fills_ok: AtomicU64::new(0),
            fills_err: AtomicU64::new(0),
            fills_skipped: AtomicU64::new(0),
            fills_skipped_in_batch: AtomicU64::new(0),
            waits: AtomicU64::new(0),
            batch_row_reads: Mutex::new(Vec::new()),
            fill_turn: Semaphore::const_new(1),
            fills_waiting: Mutex::new(BTreeSet::new()),
        }
    }

    /// Marks a batched row read of the database at `root` (the path its
    /// objects lie under) in flight until the returned value is dropped. The
    /// fill of a part of that database missed meanwhile, by whatever read,
    /// waits its turn behind the fills started under a mark before it, and
    /// goes on after the mark is gone. A miss of any other database's part
    /// fills at once, as with no mark: a fill that waits is no flight, so
    /// its part costs a round trip a read until its turn comes, and one
    /// database's batch must not put that on the cold reads of the others.
    pub fn batch_row_read(&'static self, root: Path) -> BatchRowRead {
        self.marked_roots().push(root.clone());
        BatchRowRead(self, root)
    }

    /// Whether a marked batched row read of the database that holds
    /// `location` is in flight.
    pub fn in_batch_row_read(&self, location: &Path) -> bool {
        self.marked_roots()
            .iter()
            .any(|root| location.prefix_matches(root))
    }

    /// The marked roots. Nothing panics with the lock held.
    fn marked_roots(&self) -> MutexGuard<'_, Vec<Path>> {
        self.batch_row_reads
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

pub(crate) static RANGED_MISS_STATS: RangedMissStats = RangedMissStats::new();

/// The process-wide [`RangedMissStats`].
pub fn ranged_miss_stats() -> &'static RangedMissStats {
    &RANGED_MISS_STATS
}

/// HELION LOCAL PATCH: what a directory of the on-disk cache holds, as a
/// class. The index order of [`TrimStats`] and [`open_dirs_by_size`].
pub const CACHE_DIR_SIZES: [&str; 4] = ["under_1g", "1g_2g", "2g_4g", "over_4g"];

/// The index in [`CACHE_DIR_SIZES`] of a directory that holds `bytes`.
pub fn cache_dir_size_class(bytes: u64) -> usize {
    const GIB: u64 = 1 << 30;
    match bytes {
        b if b < GIB => 0,
        b if b < 2 * GIB => 1,
        b if b < 4 * GIB => 2,
        _ => 3,
    }
}

/// HELION LOCAL PATCH: process-wide totals of what the on-disk cache's
/// evictors removed from directories over their limit, summed over every
/// cached object store in the process, by what the directory held after the
/// trim ([`CACHE_DIR_SIZES`]). Helion exports them on `/metrics`: a store's
/// own `evicted_*` stats reach no recorder there, and the evictor logs a
/// removed file at debug level only.
#[derive(Debug)]
pub struct TrimStats {
    /// Trims that removed at least one file.
    pub trims: [AtomicU64; 4],
    /// Files removed.
    pub files: [AtomicU64; 4],
    /// Bytes removed, as the evictor counted them.
    pub bytes: [AtomicU64; 4],
}

impl TrimStats {
    pub(crate) const fn new() -> Self {
        Self {
            trims: [const { AtomicU64::new(0) }; 4],
            files: [const { AtomicU64::new(0) }; 4],
            bytes: [const { AtomicU64::new(0) }; 4],
        }
    }

    /// One trim of `files` and `bytes` that left its directory holding `left`.
    pub(crate) fn record(&self, left: u64, files: u64, bytes: u64) {
        let size = cache_dir_size_class(left);
        self.trims[size].fetch_add(1, Ordering::Relaxed);
        self.files[size].fetch_add(files, Ordering::Relaxed);
        self.bytes[size].fetch_add(bytes, Ordering::Relaxed);
    }
}

pub(crate) static TRIM_STATS: TrimStats = TrimStats::new();

/// The process-wide [`TrimStats`].
pub fn trim_stats() -> &'static TrimStats {
    &TRIM_STATS
}

/// HELION LOCAL PATCH: the directories of the evictors alive in the process
/// and the bytes they hold, by [`CACHE_DIR_SIZES`] class: `(directories,
/// bytes)`.
pub fn open_dirs_by_size() -> [(u64, u64); 4] {
    super::storage_fs::OPEN_DIRS.by_size()
}
