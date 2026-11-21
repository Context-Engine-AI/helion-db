use crate::cached_object_store::policy::{
    CachePutConfig, DefaultGetPolicy, DefaultPutPolicy, GetAction, GetPolicy, HeadAction,
    PutAction, PutPolicy,
};
use crate::cached_object_store::stats::{
    CachedObjectStoreStats, RangedMissStats, RANGED_MISS_STATS,
};
use crate::cached_object_store::storage_fs::FsCacheStorage;
use crate::cached_object_store::LocalCacheEntry;
use crate::config::{default_part_fetch_concurrency, ObjectStoreCacheOptions};
use crate::object_store_tag::ObjectStoreCallTag;
use crate::query_metrics::{self, QueryCacheKind};
use bytes::{Bytes, BytesMut};
use futures::{future::BoxFuture, stream, stream::BoxStream, StreamExt};
use object_store::{path::Path, GetOptions, GetResult, ObjectMeta, ObjectStore, ObjectStoreExt};
use object_store::{
    Attributes, CopyOptions, Extensions, GetRange, GetResultPayload, PutMultipartOptions,
    PutResult, RenameOptions,
};
use object_store::{ListResult, MultipartUpload, PutOptions, PutPayload};
use slatedb_common::clock::{DefaultSystemClock, SystemClock};
use slatedb_common::DbRand;
use std::{
    ops::Range,
    sync::{atomic::Ordering, Arc},
};
use tokio::sync::Semaphore;

use crate::single_flight::SingleFlight;

use crate::cached_object_store::storage::{LocalCacheStorage, PartID};
use crate::db_cache::CacheUsageSnapshot;
use crate::error::SlateDBError;
use log::warn;

use crate::utils::build_concurrent;
use slatedb_common::metrics::MetricsRecorderHelper;

// HELION LOCAL PATCH: ranged cache misses (`ObjectStoreCacheOptions::ranged_miss`).
// A part under 1 MiB transfers in about one S3 first-byte latency, so a second
// GET would save nothing.
const RANGED_MISS_MIN_PART_BYTES: u64 = 1024 * 1024;
// Ranges up to 1/16 of the part: the ranged GET re-reads at most 6.25% of the
// part's bytes, and the caller skips at least 15/16 of the part transfer.
const RANGED_MISS_PART_DIVISOR: u64 = 16;
// Background fills hold their part in memory until saved, so they share one
// process-wide budget of 256 MiB (64 reader 4 MiB parts, 4 writer 64 MiB parts).
// One permit is one KiB.
const RANGED_FILL_BUDGET_KIB: usize = 256 * 1024;
static RANGED_FILL_BUDGET: Semaphore = Semaphore::const_new(RANGED_FILL_BUDGET_KIB);
// HELION LOCAL PATCH: the fills of a database's parts started while a marked
// batched row read of it is in flight (`RangedMissStats::batch_row_read`)
// take turns: one fetches, the others wait and hold neither bytes nor budget.
// A batch reads its rows many at a time, and its parts' fills all at once
// would hold up to the whole budget in memory and write a part a row to the
// cache volume in a second or two, where the reads of each id it stands in
// for made one fill after another. Past this much waiting, a fill is left
// out: 1 GiB of parts, about half a minute of turns, which is 1,024 parts
// of 1 MiB and 256 of 4 MiB.
const MARKED_FILLS_WAITING_KIB: usize = 1024 * 1024;
// A fill keeps its turn this long at most. One fetch that hangs would hold
// every waiting fill for as long as the store's client retries it; past
// this it goes on beside the next, still under the budget.
const MARKED_FILL_TURN_MAX: std::time::Duration = std::time::Duration::from_secs(2);
// HELION LOCAL PATCH: the parts of one read that are fetched at once
// (`ObjectStoreCacheOptions::part_fetch_concurrency`) hold at most this much:
// 8 parts of 4 MiB on a reader, one 64 MiB part on the writer.
const PART_FETCH_MAX_BYTES_IN_FLIGHT: usize = 32 * 1024 * 1024;

#[derive(Debug, Clone)]
pub(crate) struct CachedObjectStore {
    object_store: Arc<dyn ObjectStore>,
    part_size_bytes: usize, // expected to be aligned with mb or kb
    pub(crate) cache_storage: Arc<dyn LocalCacheStorage>,
    get_policy: Arc<dyn GetPolicy>,
    put_policy: Arc<dyn PutPolicy>,
    stats: Arc<CachedObjectStoreStats>,
    // Deduplicates concurrent HEAD requests for the same path after a cache miss.
    head_flights: SingleFlight<Path, (ObjectMeta, Attributes, Extensions)>,
    // Deduplicates concurrent prefetch/GET requests for the same path after a cache miss.
    prefetch_flights:
        SingleFlight<(Path, Option<GetRangeKey>), (ObjectMeta, Attributes, Extensions)>,
    // Deduplicates concurrent fetches of the same part after a cache miss.
    // Keyed on (path, part_id) so multiple readers needing the same part share one fetch.
    part_flights: SingleFlight<(Path, PartID), Bytes>,
    // HELION LOCAL PATCH: `Some` when small cache misses use a ranged GET plus a
    // background part fill.
    ranged_miss: Option<RangedMiss>,
    // HELION LOCAL PATCH: parts of one ranged read that are read at once
    // (`ObjectStoreCacheOptions::part_fetch_concurrency`, capped by
    // `effective_part_fetch_concurrency`).
    part_fetch_concurrency: usize,
}

/// HELION LOCAL PATCH: shared state of ranged cache misses. Process-wide in
/// production; tests swap in their own budget and stats.
#[derive(Debug, Clone, Copy)]
struct RangedMiss {
    fill_budget: &'static Semaphore,
    stats: &'static RangedMissStats,
}

impl CachedObjectStore {
    pub(crate) fn usage_snapshot(&self) -> CacheUsageSnapshot {
        self.cache_storage.usage_snapshot()
    }

    /// Default read and put policies, ranged misses on. Production builds go
    /// through [`Self::from_config`].
    #[allow(unused)]
    pub(crate) fn new(
        object_store: Arc<dyn ObjectStore>,
        cache_storage: Arc<dyn LocalCacheStorage>,
        part_size_bytes: usize,
        cache_puts: bool,
        stats: Arc<CachedObjectStoreStats>,
    ) -> Result<Arc<Self>, SlateDBError> {
        Self::new_with_policies(
            object_store,
            cache_storage,
            part_size_bytes,
            stats,
            Arc::new(DefaultGetPolicy),
            Arc::new(DefaultPutPolicy {
                put: CachePutConfig { cache_puts },
            }),
            true,
            default_part_fetch_concurrency(),
        )
    }

    /// Like [`Self::new`], but with caller supplied read and put policies.
    /// `new` installs [`DefaultGetPolicy`] and [`DefaultPutPolicy`].
    pub(crate) fn new_with_policies(
        object_store: Arc<dyn ObjectStore>,
        cache_storage: Arc<dyn LocalCacheStorage>,
        part_size_bytes: usize,
        stats: Arc<CachedObjectStoreStats>,
        get_policy: Arc<dyn GetPolicy>,
        put_policy: Arc<dyn PutPolicy>,
        ranged_miss: bool,
        part_fetch_concurrency: usize,
    ) -> Result<Arc<Self>, SlateDBError> {
        if part_size_bytes == 0 || !part_size_bytes.is_multiple_of(1024) {
            return Err(SlateDBError::InvalidCachePartSize);
        }

        Ok(Arc::new(Self {
            object_store,
            part_size_bytes,
            cache_storage,
            get_policy,
            put_policy,
            stats,
            head_flights: SingleFlight::new(),
            prefetch_flights: SingleFlight::new(),
            part_flights: SingleFlight::new(),
            ranged_miss: ranged_miss.then_some(RangedMiss {
                fill_budget: &RANGED_FILL_BUDGET,
                stats: &RANGED_MISS_STATS,
            }),
            part_fetch_concurrency: effective_part_fetch_concurrency(
                part_fetch_concurrency,
                part_size_bytes,
            ),
        }))
    }

    pub(crate) async fn start_evictor(&self) {
        self.cache_storage.start_evictor().await;
    }

    /// Build a `CachedObjectStore` from `ObjectStoreCacheOptions`, returning `None`
    /// if caching is not configured (i.e. `root_folder` is `None`). When `Some` is
    /// returned the evictor has already been started.
    pub(crate) async fn from_config(
        object_store: Arc<dyn ObjectStore>,
        options: &ObjectStoreCacheOptions,
        recorder: &MetricsRecorderHelper,
        clock: Arc<dyn SystemClock>,
        rand: Arc<DbRand>,
    ) -> Result<Option<Arc<Self>>, SlateDBError> {
        let cache_root_folder = match &options.root_folder {
            None => return Ok(None),
            Some(f) => f,
        };
        let stats = Arc::new(CachedObjectStoreStats::new(recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            cache_root_folder.clone(),
            options.max_cache_size_bytes,
            options.scan_interval,
            stats.clone(),
            clock,
            rand,
            options.max_open_file_handles,
        ));
        // HELION LOCAL PATCH: parts of another size are dead files.
        cache_storage.keep_only_part_size(options.part_size_bytes);
        let cached = Self::new_with_policies(
            object_store,
            cache_storage,
            options.part_size_bytes,
            stats,
            Arc::new(DefaultGetPolicy),
            Arc::new(DefaultPutPolicy {
                put: CachePutConfig {
                    cache_puts: options.cache_puts,
                },
            }),
            options.ranged_miss,
            options.part_fetch_concurrency,
        )?;
        cached.start_evictor().await;
        Ok(Some(cached))
    }

    /// Load files into cache up to a maximum number of bytes.
    /// This method fetches objects from the provided paths and stores them in the cache
    /// until the specified max_bytes limit is reached.
    pub(crate) async fn load_files_to_cache(
        &self,
        file_paths: Vec<Path>,
        max_bytes: usize,
    ) -> Result<(), SlateDBError> {
        if file_paths.is_empty() || max_bytes == 0 {
            return Ok(());
        }

        let mut remaining_bytes = max_bytes;
        let mut files_to_load = Vec::with_capacity(file_paths.len());

        // First pass: sequentially get metadata and select files that fit
        // This is done sequentially because the head calls should be very quick compared to files loading
        for path in file_paths {
            match self.object_store.head(&path).await {
                Ok(meta) => {
                    let file_size = meta.size as usize;
                    if remaining_bytes >= file_size {
                        remaining_bytes -= file_size;
                        files_to_load.push(path);
                    } else {
                        // We can't fit this file, so we stop here
                        break;
                    }
                }
                Err(e) => {
                    // If file doesn't exist or can't be accessed, we stop here
                    warn!("Failed to preload all SSTs to cache: {:?}", e);
                    break;
                }
            }
        }

        // Second pass: load the selected files in bounded parallelism and cache them.
        let degree_of_parallelism = 32;
        let _result = build_concurrent(files_to_load.into_iter(), degree_of_parallelism, |path| {
            let this = self.clone();
            async move {
                match this
                    .maybe_prefetch_range(&path, GetOptions::default())
                    .await
                {
                    Ok(_) => Ok(Some(())),
                    Err(e) => {
                        warn!(
                            "Failed to prefetch file into cache [path={}, error={:?}]",
                            path, e
                        );
                        Ok(None) // best-effort: skip errors
                    }
                }
            }
        })
        .await;

        Ok(())
    }

    pub(crate) async fn cached_head(
        &self,
        location: &Path,
        admit_on_miss: bool,
    ) -> object_store::Result<GetResult> {
        let entry = self.cache_storage.entry(location, self.part_size_bytes);
        if let Ok(Some((meta, attributes))) = entry.read_head().await {
            return Ok(head_only_get_result(meta, attributes, Extensions::new()));
        }

        // Cache miss — deduplicate concurrent HEAD requests for the same path.
        let (meta, attributes, extensions) = self
            .head_flights
            .call(location.clone(), || async {
                let result = self
                    .object_store
                    .get_opts(
                        location,
                        GetOptions {
                            range: None,
                            head: true,
                            ..Default::default()
                        },
                    )
                    .await?;
                let meta = result.meta.clone();
                let attributes = result.attributes.clone();
                let extensions = result.extensions.clone();

                if admit_on_miss {
                    self.save_get_result(location, result).await.ok();
                }
                Ok::<_, object_store::Error>((meta, attributes, extensions))
            })
            .await?;
        Ok(head_only_get_result(meta, attributes, extensions))
    }

    pub(crate) async fn cached_get_opts(
        &self,
        location: &Path,
        opts: GetOptions,
        force_refresh: bool,
    ) -> object_store::Result<GetResult> {
        let PrefetchedHead {
            meta,
            attributes,
            extensions,
            head_source,
        } = self.maybe_prefetch_range(location, opts.clone()).await?;

        let get_range = opts.range.clone();
        let object_size = meta.size;
        let range = self.canonicalize_range(get_range, object_size)?;
        let parts = self.split_range_into_parts(range.clone());

        // Read parts and concatenate them into a single stream. Some parts may not
        // be cached; read_part falls back to the object store for the missing ones.
        let futures = parts
            .into_iter()
            .map(|(part_id, range_in_part)| {
                let this = self.clone();
                let location = location.clone();
                async move {
                    this.stats.object_store_cache_part_access.increment(1);
                    let result = this
                        .read_part(
                            &location,
                            part_id,
                            range_in_part,
                            object_size,
                            force_refresh,
                        )
                        .await;
                    let Ok((bytes, part_source)) = result else {
                        query_metrics::record_cache_access(QueryCacheKind::Object, false);
                        return result.map(|(bytes, _)| bytes);
                    };
                    let hit = head_source == ReadResultSource::Disk
                        && part_source == ReadResultSource::Disk;
                    query_metrics::record_cache_access(QueryCacheKind::Object, hit);
                    if hit {
                        this.stats.object_store_cache_part_hits.increment(1);
                    }
                    Ok::<Bytes, object_store::Error>(bytes)
                }
            })
            .collect::<Vec<_>>();
        // HELION LOCAL PATCH: read up to `part_fetch_concurrency` parts at once.
        // `buffered` yields them in part order, so the stream's items are those
        // of the one-at-a-time read below, and it holds at most that many parts.
        let result_stream = if self.part_fetch_concurrency > 1 {
            stream::iter(futures)
                .buffered(self.part_fetch_concurrency)
                .boxed()
        } else {
            stream::iter(futures).then(|fut| fut).boxed()
        };

        Ok(GetResult {
            meta,
            range,
            attributes,
            payload: GetResultPayload::Stream(result_stream),
            extensions,
        })
    }

    async fn cached_put_opts(
        &self,
        location: &Path,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<PutResult> {
        // The per-call tag decides whether this write is cached.
        let tag = ObjectStoreCallTag::from_extensions(&opts.extensions);
        if self.put_policy.put_action(tag.as_ref()) == PutAction::Skip {
            // Write directly to upstream without caching the payload.
            return self.object_store.put_opts(location, payload, opts).await;
        }

        // Capture the size and attributes before payload/opts are consumed: they
        // go into the head we write below.
        let payload_size = payload.content_length() as u64;
        let attributes = opts.attributes.clone();

        // First, write to the upstream object store.
        let result = self
            .object_store
            .put_opts(location, payload.clone(), opts)
            .await?;

        // Convert PutPayload to stream and save parts to cache.
        let entry = self.cache_storage.entry(location, self.part_size_bytes);
        let stream = stream::iter(payload.into_iter()).map(Ok::<Bytes, object_store::Error>);
        // Save parts, ignoring errors (cache failures must not fail the PUT).
        self.save_parts_stream(entry.as_ref(), stream, 0).await.ok();

        // Make the write visible to reads by writing the head. This is not
        // the actual HEAD response from the upstream store, but a synthesized
        // head with the known size and attributes.
        let meta = build_head(location, payload_size, &result);
        entry.save_head((&meta, &attributes)).await.ok();

        Ok(result)
    }

    // if an object is not cached before, maybe_prefetch_range will try to prefetch the object from the
    // object store and save the parts into the local disk cache. the prefetching is helpful to reduce the
    // number of GET requests to the object store, it'll try to aggregate the parts among the range into a
    // single GET request, and save the related parts into local disks together.
    // when it sends GET requests to the object store, the range is expected to be ALIGNED with the part
    // size.
    async fn maybe_prefetch_range(
        &self,
        location: &Path,
        mut opts: GetOptions,
    ) -> object_store::Result<PrefetchedHead> {
        let entry = self.cache_storage.entry(location, self.part_size_bytes);
        match entry.read_head().await {
            Ok(Some((meta, attrs))) => {
                return Ok(PrefetchedHead {
                    meta,
                    attributes: attrs,
                    extensions: Extensions::new(),
                    head_source: ReadResultSource::Disk,
                })
            }
            Ok(None) => {}
            Err(e) => {
                warn!(
                    "failed to read head from disk cache, will fallback to object store [location={}, error={:?}]",
                    location, e,
                );
            }
        }

        if let Some(range) = &opts.range {
            opts.range = Some(self.align_get_range(range));
        }

        // Cache miss — deduplicate concurrent prefetch requests for the same path.
        // Only one caller performs the fetch+save; others share the metadata result.
        // Parts not covered by the winning caller's range are handled by read_part's
        // own object-store fallback, so correctness is maintained.
        self.prefetch_flights
            .call(
                (location.clone(), opts.range.clone().map(Into::into)),
                || async {
                    let get_result = self.object_store.get_opts(location, opts).await?;
                    let result_meta = get_result.meta.clone();
                    let result_attrs = get_result.attributes.clone();
                    let result_extensions = get_result.extensions.clone();
                    // swallow the error on saving to disk here (the disk might be already full), just fallback
                    // to the object store.
                    // TODO: add a warning log here
                    self.save_get_result(location, get_result).await.ok();
                    Ok((result_meta, result_attrs, result_extensions))
                },
            )
            .await
            .map(|(meta, attributes, extensions)| PrefetchedHead {
                meta,
                attributes,
                extensions,
                head_source: ReadResultSource::Upstream,
            })
    }

    /// save the GetResult to the disk cache, a GetResult may be transformed into multiple part
    /// files and a meta file. please note that the `range` in the GetResult is expected to be
    /// aligned with the part size.
    async fn save_get_result(
        &self,
        cache_location: &Path,
        result: GetResult,
    ) -> object_store::Result<u64> {
        let part_size_bytes_u64 = self.part_size_bytes as u64;
        assert!(result.range.start.is_multiple_of(part_size_bytes_u64));
        assert!(
            result.range.end.is_multiple_of(part_size_bytes_u64)
                || result.range.end == result.meta.size
        );

        let entry = self
            .cache_storage
            .entry(cache_location, self.part_size_bytes);
        let object_size = result.meta.size;

        // Reaching here means the read policy already chose to fill the cache
        // so always save.
        entry.save_head((&result.meta, &result.attributes)).await?;

        let start_part_number = usize::try_from(result.range.start / part_size_bytes_u64)
            .expect("Part number exceeds u32 on a 32-bit system. Try increasing part size.");

        let stream = result.into_stream();

        self.save_parts_stream(entry.as_ref(), stream, start_part_number)
            .await?;

        Ok(object_size)
    }

    /// Save a stream of bytes to cache as parts, starting from the specified part number.
    /// Returns the number of bytes saved.
    /// This method only saves the data parts - the head should be saved separately.
    async fn save_parts_stream<S>(
        &self,
        entry: &dyn LocalCacheEntry,
        mut stream: S,
        start_part_number: usize,
    ) -> object_store::Result<usize>
    where
        S: stream::Stream<Item = Result<Bytes, object_store::Error>> + Unpin,
    {
        let mut buffer = BytesMut::new();
        let mut part_number = start_part_number;
        let mut total_bytes: usize = 0;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            total_bytes += chunk.len();
            buffer.extend_from_slice(&chunk);

            while buffer.len() >= self.part_size_bytes {
                let to_write = buffer.split_to(self.part_size_bytes);
                entry.save_part(part_number, to_write.into()).await?;
                part_number += 1;
            }
        }

        // Save any remaining bytes as the last part
        if !buffer.is_empty() {
            entry.save_part(part_number, buffer.into()).await?;
        }

        Ok(total_bytes)
    }

    // split the range into parts, and return the part id and the range inside the part.
    fn split_range_into_parts(&self, range: Range<u64>) -> Vec<(PartID, Range<usize>)> {
        let part_size_bytes_u64 = self.part_size_bytes as u64;
        let range_aligned = self.align_range(&range, self.part_size_bytes);
        let start_part = range_aligned.start / part_size_bytes_u64;
        let end_part = range_aligned.end / part_size_bytes_u64;
        let mut parts: Vec<_> = (start_part..end_part)
            .map(|part_id| {
                (
                    usize::try_from(part_id).expect("Number of parts exceeds usize"),
                    Range {
                        start: 0,
                        end: self.part_size_bytes,
                    },
                )
            })
            .collect();
        if parts.is_empty() {
            return vec![];
        }
        if let Some(first_part) = parts.first_mut() {
            first_part.1.start = usize::try_from(range.start % part_size_bytes_u64)
                .expect("Part size is too large to fit in a usize");
        }
        if let Some(last_part) = parts.last_mut() {
            if !range.end.is_multiple_of(part_size_bytes_u64) {
                last_part.1.end = usize::try_from(range.end % part_size_bytes_u64)
                    .expect("Part size is too large to fit in a usize");
            }
        }
        parts
    }

    /// Get a part from disk if cached, otherwise start a new GET request.
    ///
    /// IO errors reading the disk cache are ignored and fall back to the object
    /// store.
    ///
    /// Returns the bytes plus where they were served from, so the caller can
    /// classify the read as a hit or a miss.
    fn read_part(
        &self,
        location: &Path,
        part_id: PartID,
        range_in_part: Range<usize>,
        object_size: u64,
        force_refresh: bool,
    ) -> BoxFuture<'static, object_store::Result<(Bytes, ReadResultSource)>> {
        let this = self.clone();
        let location = location.clone();
        Box::pin(async move {
            let entry = this.cache_storage.entry(&location, this.part_size_bytes);
            if !force_refresh {
                if let Ok(Some(bytes)) = entry.read_part(part_id, range_in_part.clone()).await {
                    return Ok((bytes, ReadResultSource::Disk));
                }
            }

            // Cache miss, so we need to fetch from the object store.
            let key = (location.clone(), part_id);
            // HELION LOCAL PATCH: a small read gets its own ranged GET while the
            // whole part fills in the background, unless a fill is already in
            // flight (then it waits for that one, as before).
            if let Some(ranged) = this.ranged_miss {
                if !force_refresh && this.wants_ranged_miss(part_id, &range_in_part, object_size) {
                    if this.part_flights.contains(&key) {
                        ranged.stats.waits.fetch_add(1, Ordering::Relaxed);
                    } else {
                        this.spawn_part_fill(ranged, location.clone(), part_id);
                        match this
                            .get_range_in_part(ranged, &location, part_id, range_in_part.clone())
                            .await
                        {
                            Ok(bytes) => return Ok((bytes, ReadResultSource::Upstream)),
                            Err(e) => warn!(
                                "ranged cache miss failed, waiting for the full part [location={}, part={}, error={:?}]",
                                location, part_id, e,
                            ),
                        }
                    }
                }
            }

            // Read Part — deduplicate concurrent fetches of the same part.
            // The SingleFlight fetches the full part and saves it to cache; each
            // caller then slices out their own range_in_part.
            let bytes = this
                .part_flights
                .call(key, || this.fetch_part(&location, part_id))
                .await?;

            Ok((bytes.slice(range_in_part), ReadResultSource::Upstream))
        })
    }

    /// Fetch one whole part and save it, with the head, to the disk cache. The
    /// body of every `part_flights` call.
    async fn fetch_part(&self, location: &Path, part_id: PartID) -> object_store::Result<Bytes> {
        let part_range = Range {
            start: (part_id * self.part_size_bytes) as u64,
            end: ((part_id + 1) * self.part_size_bytes) as u64,
        };
        let get_result = self
            .object_store
            .get_opts(
                location,
                GetOptions {
                    range: Some(GetRange::Bounded(part_range.clone())),
                    ..Default::default()
                },
            )
            .await?;

        // Save the head and the part to cache for future accesses.
        let entry = self.cache_storage.entry(location, self.part_size_bytes);
        let meta = get_result.meta.clone();
        let attrs = get_result.attributes.clone();
        let bytes = get_result.bytes().await?;
        // HELION LOCAL PATCH: save_part stores whatever it is given, so a short
        // body must never become the part (only the last part is short).
        let expected_len = part_range
            .end
            .min(meta.size)
            .saturating_sub(part_range.start);
        check_body_len(bytes.len(), expected_len)?;
        entry.save_head((&meta, &attrs)).await.ok();
        entry.save_part(part_id, bytes.clone()).await.ok();

        Ok(bytes)
    }

    /// HELION LOCAL PATCH: whether a cache miss of `range_in_part` takes the
    /// ranged path: the range is at most 1/16 of a part of at least 1 MiB. The
    /// last part of an object counts at its real, shorter length.
    fn wants_ranged_miss(
        &self,
        part_id: PartID,
        range_in_part: &Range<usize>,
        object_size: u64,
    ) -> bool {
        let part_start = (part_id * self.part_size_bytes) as u64;
        let part_len = object_size
            .saturating_sub(part_start)
            .min(self.part_size_bytes as u64);
        // An empty range is no valid GET; RetryingObjectStore would retry it forever.
        part_len >= RANGED_MISS_MIN_PART_BYTES
            && !range_in_part.is_empty()
            && range_in_part.len() as u64 * RANGED_MISS_PART_DIVISOR <= part_len
    }

    /// HELION LOCAL PATCH: GET exactly `range_in_part`. Saves nothing to the
    /// cache; the background fill saves the part and the head.
    async fn get_range_in_part(
        &self,
        ranged: RangedMiss,
        location: &Path,
        part_id: PartID,
        range_in_part: Range<usize>,
    ) -> object_store::Result<Bytes> {
        let part_start = (part_id * self.part_size_bytes) as u64;
        let range = Range {
            start: part_start + range_in_part.start as u64,
            end: part_start + range_in_part.end as u64,
        };
        ranged.stats.misses.fetch_add(1, Ordering::Relaxed);
        let bytes = self
            .object_store
            .get_opts(
                location,
                GetOptions {
                    range: Some(GetRange::Bounded(range)),
                    ..Default::default()
                },
            )
            .await?
            .bytes()
            .await?;
        check_body_len(bytes.len(), range_in_part.len() as u64)?;
        ranged
            .stats
            .bytes
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok(bytes)
    }

    /// HELION LOCAL PATCH: fetch and save the whole part in a task the caller
    /// does not await, through `part_flights` so later misses wait for it. When
    /// the fill budget is spent the fill is skipped and the next miss refetches.
    /// While a marked batched row read of the part's database is in flight
    /// the fill waits its turn instead ([`Self::queue_part_fill`]).
    fn spawn_part_fill(&self, ranged: RangedMiss, location: Path, part_id: PartID) {
        if ranged.stats.in_batch_row_read(&location) {
            self.queue_part_fill(ranged, location, part_id);
            return;
        }
        let Ok(permit) = ranged.fill_budget.try_acquire_many(self.fill_permits()) else {
            ranged.stats.fills_skipped.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let this = self.clone();
        tokio::spawn(async move {
            let _permit = permit;
            this.fill_part(ranged, location, part_id).await;
        });
    }

    /// HELION LOCAL PATCH: the fill of a part missed while a marked batched
    /// row read of its database is in flight, whichever read missed it. It
    /// waits for the one turn in a task the caller does not await, and goes
    /// on after the mark is gone. A fill that waits is no flight: a later
    /// miss of its part takes its own range and waits for nothing, and
    /// starts no second fill. Left out, and counted, where
    /// `MARKED_FILLS_WAITING_KIB` of parts wait already or the budget is
    /// spent when its turn comes. A fill that has the turn gives it up after
    /// `MARKED_FILL_TURN_MAX`.
    fn queue_part_fill(&self, ranged: RangedMiss, location: Path, part_id: PartID) {
        let key = (location, part_id);
        {
            let mut waiting = ranged
                .stats
                .fills_waiting
                .lock()
                .expect("waiting fills lock");
            if waiting.contains(&key) {
                return;
            }
            if waiting.len() >= MARKED_FILLS_WAITING_KIB / self.fill_permits().max(1) as usize {
                ranged
                    .stats
                    .fills_skipped_in_batch
                    .fetch_add(1, Ordering::Relaxed);
                return;
            }
            waiting.insert(key.clone());
        }
        let this = self.clone();
        tokio::spawn(async move {
            let turn = ranged.stats.fill_turn.acquire().await;
            ranged
                .stats
                .fills_waiting
                .lock()
                .expect("waiting fills lock")
                .remove(&key);
            let (location, part_id) = key;
            // The turn is never closed. A read with no mark in flight may have
            // filled the part while this waited.
            let cached = this.cache_storage.entry(&location, this.part_size_bytes);
            if turn.is_err() || matches!(cached.read_part(part_id, 0..1).await, Ok(Some(_))) {
                return;
            }
            let Ok(_permit) = ranged.fill_budget.try_acquire_many(this.fill_permits()) else {
                ranged
                    .stats
                    .fills_skipped_in_batch
                    .fetch_add(1, Ordering::Relaxed);
                return;
            };
            let fill = this.fill_part(ranged, location, part_id);
            tokio::pin!(fill);
            let clock = DefaultSystemClock::new();
            tokio::select! {
                () = &mut fill => return,
                () = clock.sleep(MARKED_FILL_TURN_MAX) => {}
            }
            drop(turn);
            fill.await;
        });
    }

    /// The fill budget's permits for one part.
    fn fill_permits(&self) -> u32 {
        let permits = (self.part_size_bytes / 1024).min(RANGED_FILL_BUDGET_KIB);
        u32::try_from(permits).expect("fill budget fits in u32 permits")
    }

    /// The body of a background fill: one part fetched and saved through
    /// `part_flights`, and counted.
    async fn fill_part(&self, ranged: RangedMiss, location: Path, part_id: PartID) {
        let result = self
            .part_flights
            .call((location.clone(), part_id), || {
                self.fetch_part(&location, part_id)
            })
            .await;
        match result {
            Ok(_) => {
                ranged.stats.fills_ok.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                ranged.stats.fills_err.fetch_add(1, Ordering::Relaxed);
                warn!(
                    "background cache part fill failed [location={}, part={}, error={:?}]",
                    location, part_id, e,
                );
            }
        }
    }

    // given the range and object size, return the canonicalized `Range<usize>` with concrete start and
    // end.
    fn canonicalize_range(
        &self,
        range: Option<GetRange>,
        object_size: u64,
    ) -> object_store::Result<Range<u64>> {
        let (start_offset, end_offset) = match range {
            None => (0, object_size),
            Some(range) => match range {
                GetRange::Bounded(range) => {
                    if range.start >= object_size {
                        return Err(object_store::Error::Generic {
                            store: "cached_object_store",
                            source: Box::new(InvalidGetRange::StartTooLarge {
                                requested: range.start,
                                length: object_size,
                            }),
                        });
                    }
                    if range.start >= range.end {
                        return Err(object_store::Error::Generic {
                            store: "cached_object_store",
                            source: Box::new(InvalidGetRange::Inconsistent {
                                start: range.start,
                                end: range.end,
                            }),
                        });
                    }
                    (range.start, range.end.min(object_size))
                }
                GetRange::Offset(offset) => {
                    if offset >= object_size {
                        return Err(object_store::Error::Generic {
                            store: "cached_object_store",
                            source: Box::new(InvalidGetRange::StartTooLarge {
                                requested: offset,
                                length: object_size,
                            }),
                        });
                    }
                    (offset, object_size)
                }
                GetRange::Suffix(suffix) => (object_size.saturating_sub(suffix), object_size),
            },
        };
        Ok(Range {
            start: start_offset,
            end: end_offset,
        })
    }

    fn align_get_range(&self, range: &GetRange) -> GetRange {
        match range {
            GetRange::Bounded(bounded) => {
                let aligned = self.align_range(bounded, self.part_size_bytes);
                GetRange::Bounded(aligned)
            }
            GetRange::Suffix(suffix) => {
                let suffix_aligned = self.align_range(&(0..*suffix), self.part_size_bytes).end;
                GetRange::Suffix(suffix_aligned)
            }
            GetRange::Offset(offset) => {
                let offset_aligned = *offset - *offset % self.part_size_bytes as u64;
                GetRange::Offset(offset_aligned)
            }
        }
    }

    fn align_range(&self, range: &Range<u64>, alignment: usize) -> Range<u64> {
        let alignment = alignment as u64;
        let start_aligned = range.start - range.start % alignment;
        let end_aligned = range.end.div_ceil(alignment) * alignment;
        Range {
            start: start_aligned,
            end: end_aligned,
        }
    }
}

/// HELION LOCAL PATCH: how many parts of one read are fetched at once: the
/// configured number, lowered until they hold at most
/// [`PART_FETCH_MAX_BYTES_IN_FLIGHT`], but at least one. One (or a configured
/// 0) is the sequential read.
fn effective_part_fetch_concurrency(configured: usize, part_size_bytes: usize) -> usize {
    configured.min((PART_FETCH_MAX_BYTES_IN_FLIGHT / part_size_bytes).max(1))
}

/// HELION LOCAL PATCH: reject an upstream body whose length is not the length
/// of the range it answers.
fn check_body_len(actual: usize, expected: u64) -> object_store::Result<()> {
    if actual as u64 == expected {
        return Ok(());
    }
    Err(object_store::Error::Generic {
        store: "cached_object_store",
        source: format!("read {actual} bytes from the object store, expected {expected}").into(),
    })
}

fn head_only_get_result(
    meta: ObjectMeta,
    attributes: Attributes,
    extensions: Extensions,
) -> GetResult {
    GetResult {
        payload: GetResultPayload::Stream(stream::empty().boxed()),
        range: 0..0,
        meta,
        attributes,
        extensions,
    }
}

/// Builds a synthetic head to save on a write, from the upstream `PutResult`
/// and the known object size.
///
/// The head is the cache entry's commit point: cached parts are not usable
/// until a `read_head` succeeds, so writing it last (after the upstream write
/// completes) publishes the entry.
fn build_head(cache_location: &Path, size: u64, result: &PutResult) -> ObjectMeta {
    ObjectMeta {
        location: cache_location.clone(),
        // `last_modified` is not used by the cache, add a stub instead of
        // executing an actual HEAD request after write. If this ever change,
        // the cache should be updated to use the upstream `last_modified`
        // instead of the stub value here.
        last_modified: chrono::DateTime::<chrono::Utc>::from_timestamp(0, 0)
            .expect("unix epoch is a valid timestamp"),
        size,
        e_tag: result.e_tag.clone(),
        version: result.version.clone(),
    }
}

impl std::fmt::Display for CachedObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CachedObjectStore({}, {})",
            self.object_store, self.cache_storage
        )
    }
}

#[async_trait::async_trait]
impl ObjectStore for CachedObjectStore {
    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        let tag = ObjectStoreCallTag::from_extensions(&options.extensions);

        if options.head {
            return match self.get_policy.head_action(tag.as_ref()) {
                HeadAction::Bypass => self.object_store.get_opts(location, options).await,
                HeadAction::Probe => self.cached_head(location, false).await,
                HeadAction::ReadThrough => self.cached_head(location, true).await,
            };
        }
        match self.get_policy.get_action(tag.as_ref()) {
            GetAction::Bypass => self.object_store.get_opts(location, options).await,
            GetAction::Refetch => self.cached_get_opts(location, options, true).await,
            GetAction::ReadThrough => self.cached_get_opts(location, options, false).await,
        }
    }

    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.cached_put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        let tag = ObjectStoreCallTag::from_extensions(&opts.extensions);
        let attributes = opts.attributes.clone();

        let inner = self.object_store.put_multipart_opts(location, opts).await?;

        // Wrap the upload to mirror its parts into the cache, unless skipped.
        if self.put_policy.put_action(tag.as_ref()) == PutAction::Skip {
            return Ok(inner);
        }
        Ok(Box::new(CachingMultipartUpload::new(
            inner,
            Arc::clone(&self.cache_storage),
            location.clone(),
            self.part_size_bytes,
            attributes,
        )))
    }

    /// Deletion of the cache entries associated with the object being
    /// deleted is not atomic with respect to the object deletion from
    /// the underlying object store. So for some period of time after
    /// the deletion, cached object parts are still visible in the cache.
    /// But assuming each object ever created by SlateDB is immutable and
    /// has a unique name, this is not a problem.
    ///
    /// If eviction is enabled, deletion of the associated cache entries
    /// happens asynchronously; when the control returns to the caller,
    /// the entries still might be present in the cache. If eviction is
    /// off, the deletion happens synchronously; when the control returns
    /// to the caller, it is guaranteed no entries present in the cache
    /// (assuming no errors happened during the deletion).
    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        let cache_storage = self.cache_storage.clone();
        let part_size_bytes = self.part_size_bytes;

        self.object_store
            .delete_stream(locations)
            .then(move |result| {
                let cache_storage = cache_storage.clone();
                async move {
                    if let Ok(ref location) = result {
                        let entry = cache_storage.entry(location, part_size_bytes);
                        entry.delete().await;
                    }
                    result
                }
            })
            .boxed()
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.object_store.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.object_store.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        self.object_store.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.object_store.copy_opts(from, to, options).await
    }

    async fn rename_opts(
        &self,
        from: &Path,
        to: &Path,
        options: RenameOptions,
    ) -> object_store::Result<()> {
        self.object_store.rename_opts(from, to, options).await
    }
}

/// A [`MultipartUpload`] that mirrors the uploaded bytes into the local cache as
/// it streams them upstream. Created by [`CachedObjectStore::put_multipart_opts`]
/// when the call policy caches a compacted SST (the path large compacted SSTs
/// take, above BufWriter's multipart threshold).
///
/// The head is the commit point: it is written on `complete` after the
/// upstream upload succeeds, which publishes the parts as a live cache entry.
/// Until then the parts are not usable (a read with no head refetches from
/// upstream).
///
/// Cache writes are best effort: a failed cache write never fails the upload.
///
/// TODO: fix potential part leak: a multipart upload that fails midway
/// (dropped without complete() or abort() leaks its already written cache
/// parts forever when the evictor is disabled. Can happen on a crash or
/// exhuasting retries in one of the uploads.
struct CachingMultipartUpload {
    inner: Box<dyn MultipartUpload>,
    cache_storage: Arc<dyn LocalCacheStorage>,
    cache_location: Path,
    part_size: usize,
    /// In-order bytes observed so far that have not yet filled a cache part.
    buffer: BytesMut,
    /// The next cache part number to write.
    next_part: PartID,
    /// Total bytes teed so far; becomes the committed head's `size`.
    total_len: u64,
    /// Attributes from the upload options, echoed into the committed head.
    attributes: Attributes,
}

impl CachingMultipartUpload {
    fn new(
        inner: Box<dyn MultipartUpload>,
        cache_storage: Arc<dyn LocalCacheStorage>,
        cache_location: Path,
        part_size: usize,
        attributes: Attributes,
    ) -> Self {
        Self {
            inner,
            cache_storage,
            cache_location,
            part_size,
            buffer: BytesMut::new(),
            next_part: 0,
            total_len: 0,
            attributes,
        }
    }
}

impl std::fmt::Debug for CachingMultipartUpload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachingMultipartUpload")
            .field("cache_location", &self.cache_location)
            .finish_non_exhaustive()
    }
}

#[async_trait::async_trait]
impl MultipartUpload for CachingMultipartUpload {
    fn put_part(&mut self, data: PutPayload) -> object_store::UploadPart {
        // Write the payload bytes into the cache buffer, then forward the
        // original payload upstream.
        self.total_len += data.content_length() as u64;
        self.buffer.reserve(data.content_length());
        for bytes in &data {
            self.buffer.extend_from_slice(bytes);
        }
        let mut parts = Vec::new();
        while self.buffer.len() >= self.part_size {
            let chunk = self.buffer.split_to(self.part_size).freeze();
            parts.push((self.next_part, chunk));
            self.next_part += 1;
        }

        let inner_fut = self.inner.put_part(data);
        if parts.is_empty() {
            return inner_fut;
        }
        let cache_storage = Arc::clone(&self.cache_storage);
        let cache_location = self.cache_location.clone();
        let part_size = self.part_size;
        Box::pin(async move {
            // Overlap the cache disk writes with the upstream upload.
            let cache_fut = async {
                let entry = cache_storage.entry(&cache_location, part_size);
                for (part_number, chunk) in parts {
                    // Currently, we ignore errors writing to the cache. This
                    // is best-effort: a failed cache write never fails the
                    // upload.
                    entry.save_part(part_number, chunk).await.ok();
                }
            };
            let (result, ()) = futures::future::join(inner_fut, cache_fut).await;
            result
        })
    }

    async fn complete(&mut self) -> object_store::Result<PutResult> {
        let result = self.inner.complete().await?;
        let entry = self
            .cache_storage
            .entry(&self.cache_location, self.part_size);
        // Flush the trailing partial part once the upload is durable upstream.
        if !self.buffer.is_empty() {
            let chunk = std::mem::take(&mut self.buffer).freeze();
            entry.save_part(self.next_part, chunk).await.ok();
            self.next_part += 1;
        }

        // Commit by writing the head last (after the upstream upload succeeded):
        // it publishes the parts as a live cache entry and lets the first read
        // serve from the cache instead of doing an upstream HEAD and re-prefetch.
        let meta = build_head(&self.cache_location, self.total_len, &result);
        entry.save_head((&meta, &self.attributes)).await.ok();
        Ok(result)
    }

    async fn abort(&mut self) -> object_store::Result<()> {
        let result = self.inner.abort().await;
        // The object will never exist upstream, so drop any cached parts.
        self.cache_storage
            .entry(&self.cache_location, self.part_size)
            .delete()
            .await;
        result
    }
}

/// Where a read (of the object head or of a single part) was served from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ReadResultSource {
    /// Served from the local disk cache.
    Disk,
    /// Fetched from Object Store.
    Upstream,
}

/// The head metadata returned by [`CachedObjectStore::maybe_prefetch_range`],
/// plus where the head was served from (`Disk` = warm, `Upstream` = cold
/// prefetch).
struct PrefetchedHead {
    meta: ObjectMeta,
    attributes: Attributes,
    extensions: Extensions,
    head_source: ReadResultSource,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum InvalidGetRange {
    #[error("Range start too large, requested: {requested}, length: {length}")]
    StartTooLarge { requested: u64, length: u64 },

    #[error("Range started at {start} and ended at {end}")]
    Inconsistent { start: u64, end: u64 },
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
/// A mirror of [`object_store::GetRange`] that implements [`Hash`] and [`Eq`],
/// allowing it to be used as a key in hash-based collections (e.g. `SingleFlight`).
enum GetRangeKey {
    Bounded(Range<u64>),
    Offset(u64),
    Suffix(u64),
}

impl From<GetRange> for GetRangeKey {
    fn from(range: GetRange) -> Self {
        match range {
            GetRange::Bounded(r) => GetRangeKey::Bounded(r),
            GetRange::Offset(o) => GetRangeKey::Offset(o),
            GetRange::Suffix(s) => GetRangeKey::Suffix(s),
        }
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use object_store::{
        path::Path, GetOptions, GetRange, MultipartUpload, ObjectStore, ObjectStoreExt,
        PutMultipartOptions, PutPayload,
    };
    use rand::Rng;
    use rstest::rstest;
    use std::sync::Arc;
    use std::time::Duration;

    use super::CachedObjectStore;
    use crate::cached_object_store::stats::CachedObjectStoreStats;
    use crate::cached_object_store::storage::{LocalCacheStorage, PartID};
    use crate::cached_object_store::storage_fs::FsCacheEntry;
    use crate::cached_object_store::storage_fs::FsCacheStorage;
    use crate::db_state::SstType;
    use crate::object_store_tag::{ObjectStoreCallTag, TableStoreKind};
    use crate::test_utils::{
        gen_rand_bytes, ExtensionMarker, ExtensionObjectStore, FlakyObjectStore, GatedObjectStore,
    };
    use slatedb_common::clock::DefaultSystemClock;
    use slatedb_common::metrics::MetricsRecorderHelper;
    use slatedb_common::DbRand;

    fn new_test_cache_folder() -> std::path::PathBuf {
        let mut rng = rand::rng();
        let dir_name: String = (0..10)
            .map(|_| rng.sample(rand::distr::Alphanumeric) as char)
            .collect();
        let path = format!("/tmp/testcache-{}", dir_name);
        let _ = std::fs::remove_dir_all(&path);
        std::path::PathBuf::from(path)
    }

    fn new_cached_store(object_store: Arc<dyn ObjectStore>) -> Arc<CachedObjectStore> {
        let test_cache_folder = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder,
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));
        CachedObjectStore::new(object_store, cache_storage, 1024, false, stats).unwrap()
    }

    /// The production constructor tells the storage its part size, so the
    /// first scan drops what another part size left in the folder.
    #[tokio::test]
    async fn a_store_built_from_config_drops_parts_of_another_size() {
        let test_cache_folder = new_test_cache_folder();
        let sst = test_cache_folder.join("db/compacted/01ARZ3NDEKTSV4RRFFQ69G5FAV.sst");
        std::fs::create_dir_all(&sst).unwrap();
        for name in ["_part4mb-000000000", "_part1kb-000000000", "_head"] {
            std::fs::write(sst.join(name), [7u8; 16]).unwrap();
        }
        let options = crate::config::ObjectStoreCacheOptions {
            root_folder: Some(test_cache_folder.clone()),
            part_size_bytes: 1024,
            ..Default::default()
        };
        let store = CachedObjectStore::from_config(
            Arc::new(object_store::memory::InMemory::new()),
            &options,
            &MetricsRecorderHelper::noop(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
        )
        .await
        .unwrap()
        .unwrap();

        let mut snapshot = store.cache_storage.usage_snapshot();
        for _ in 0..500 {
            if matches!(snapshot, crate::db_cache::CacheUsageSnapshot::Ready { .. }) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
            snapshot = store.cache_storage.usage_snapshot();
        }
        assert!(
            matches!(
                snapshot,
                crate::db_cache::CacheUsageSnapshot::Ready { used_bytes: 32, .. }
            ),
            "{snapshot:?}"
        );
        assert!(!sst.join("_part4mb-000000000").exists());
        assert!(sst.join("_part1kb-000000000").exists());
        assert!(sst.join("_head").exists());
        let _ = std::fs::remove_dir_all(&test_cache_folder);
    }

    #[tokio::test]
    async fn test_save_result_not_aligned() -> object_store::Result<()> {
        let payload = gen_rand_bytes(1024 * 3 + 32);
        let object_store = Arc::new(object_store::memory::InMemory::new());
        let test_cache_folder = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        object_store
            .put(
                &Path::from("/data/testfile1"),
                PutPayload::from_bytes(payload.clone()),
            )
            .await?;
        let location = Path::from("/data/testfile1");
        let get_result = object_store.get(&location).await?;

        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder.clone(),
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));

        let part_size = 1024;
        let cached_store =
            CachedObjectStore::new(object_store.clone(), cache_storage, part_size, false, stats)
                .unwrap();
        let entry = cached_store.cache_storage.entry(&location, 1024);

        let object_size_hint = cached_store.save_get_result(&location, get_result).await?;
        assert_eq!(object_size_hint, 1024 * 3 + 32);

        // assert the cached meta
        let head = entry.read_head().await?;
        assert_eq!(head.unwrap().0.size, 1024 * 3 + 32);

        // assert the parts
        let cached_parts = entry.cached_parts().await?;
        assert_eq!(cached_parts.len(), 4);
        assert_eq!(
            entry.read_part(0, 0..part_size).await?,
            Some(payload.slice(0..1024))
        );
        assert_eq!(
            entry.read_part(1, 0..part_size).await?,
            Some(payload.slice(1024..2048))
        );
        assert_eq!(
            entry.read_part(2, 0..part_size).await?,
            Some(payload.slice(2048..3072))
        );
        // check that the unaligned part was also cached
        assert_eq!(
            entry.read_part(3, 0..32).await?,
            Some(payload.slice(3072..3104))
        );

        // delete part 2, known_cache_size is still known
        let evict_part_path =
            FsCacheEntry::make_part_path(test_cache_folder.clone(), &location, 2, 1024);
        std::fs::remove_file(evict_part_path).unwrap();
        assert_eq!(entry.read_part(2, 0..part_size).await?, None);
        let cached_parts = entry.cached_parts().await?;
        assert_eq!(cached_parts, vec![0, 1, 3]);

        // delete part 3, known_cache_size become None
        let evict_part_path =
            FsCacheEntry::make_part_path(test_cache_folder.clone(), &location, 3, 1024);
        std::fs::remove_file(evict_part_path).unwrap();
        assert_eq!(entry.read_part(3, 0..part_size).await?, None);
        let cached_parts = entry.cached_parts().await?;
        assert_eq!(cached_parts, vec![0, 1]);
        Ok(())
    }

    #[tokio::test]
    async fn test_save_result_aligned() -> object_store::Result<()> {
        let payload = gen_rand_bytes(1024 * 3);
        let object_store = Arc::new(object_store::memory::InMemory::new());
        let test_cache_folder = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        object_store
            .put(
                &Path::from("/data/testfile1"),
                PutPayload::from_bytes(payload.clone()),
            )
            .await?;
        let location = Path::from("/data/testfile1");
        let get_result = object_store.get(&location).await?;
        let part_size = 1024;

        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder.clone(),
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));

        let cached_store =
            CachedObjectStore::new(object_store, cache_storage, part_size, false, stats).unwrap();
        let entry = cached_store.cache_storage.entry(&location, part_size);
        let object_size_hint = cached_store.save_get_result(&location, get_result).await?;
        assert_eq!(object_size_hint, 1024 * 3);
        let cached_parts = entry.cached_parts().await?;
        assert_eq!(cached_parts.len(), 3);
        assert_eq!(
            entry.read_part(0, 0..part_size).await?,
            Some(payload.slice(0..1024))
        );
        assert_eq!(
            entry.read_part(1, 0..part_size).await?,
            Some(payload.slice(1024..2048))
        );
        assert_eq!(
            entry.read_part(2, 0..part_size).await?,
            Some(payload.slice(2048..3072))
        );

        let evict_part_path =
            FsCacheEntry::make_part_path(test_cache_folder.clone(), &location, 2, part_size);
        std::fs::remove_file(evict_part_path).unwrap();
        assert_eq!(entry.read_part(2, 0..part_size).await?, None);

        let cached_parts = entry.cached_parts().await?;
        assert_eq!(cached_parts.len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn test_cached_get_opts_preserves_extensions_on_cache_miss() {
        let inner: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let location = Path::from("/data/test_extensions_get");
        inner
            .put(
                &location,
                PutPayload::from_bytes(bytes::Bytes::from_static(b"hello world")),
            )
            .await
            .unwrap();

        let marking: Arc<dyn ObjectStore> = Arc::new(ExtensionObjectStore::new(inner));
        let cached_store = new_cached_store(marking);
        let result = cached_store
            .cached_get_opts(
                &location,
                GetOptions {
                    range: Some(GetRange::Bounded(0..5)),
                    ..Default::default()
                },
                false,
            )
            .await
            .expect("cache miss should fetch from inner store");

        assert!(result.extensions.get::<ExtensionMarker>().is_some());
        assert_eq!(
            result.bytes().await.unwrap(),
            bytes::Bytes::from_static(b"hello")
        );
    }

    #[tokio::test]
    async fn test_cached_head_preserves_extensions_on_cache_miss() {
        let inner: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let location = Path::from("/data/test_extensions_head");
        inner
            .put(
                &location,
                PutPayload::from_bytes(bytes::Bytes::from_static(b"hello")),
            )
            .await
            .unwrap();

        let marking: Arc<dyn ObjectStore> = Arc::new(ExtensionObjectStore::new(inner));
        let cached_store = new_cached_store(marking);
        let result = cached_store
            .cached_head(&location, true)
            .await
            .expect("cache miss should fetch head from inner store");

        assert!(result.extensions.get::<ExtensionMarker>().is_some());
    }

    #[test]
    fn test_split_range_into_parts() {
        let object_store = Arc::new(object_store::memory::InMemory::new());
        let test_cache_folder = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder,
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));

        let cached_store =
            CachedObjectStore::new(object_store, cache_storage, 1024, false, stats).unwrap();

        struct Test {
            input: (Option<GetRange>, usize),
            expect: Vec<(PartID, std::ops::Range<usize>)>,
        }
        let tests = [
            Test {
                input: (None, 1024 * 3),
                expect: vec![(0, 0..1024), (1, 0..1024), (2, 0..1024)],
            },
            Test {
                input: (None, 1024 * 3 + 12),
                expect: vec![(0, 0..1024), (1, 0..1024), (2, 0..1024), (3, 0..12)],
            },
            Test {
                input: (None, 12),
                expect: vec![(0, 0..12)],
            },
            Test {
                input: (Some(GetRange::Bounded(0..1024)), 1024),
                expect: vec![(0, 0..1024)],
            },
            Test {
                input: (Some(GetRange::Bounded(128..1024)), 20000),
                expect: vec![(0, 128..1024)],
            },
            Test {
                input: (Some(GetRange::Bounded(128..1024 + 12)), 20000),
                expect: vec![(0, 128..1024), (1, 0..12)],
            },
            Test {
                input: (Some(GetRange::Bounded(128..1024 * 2 + 12)), 20000),
                expect: vec![(0, 128..1024), (1, 0..1024), (2, 0..12)],
            },
            Test {
                input: (Some(GetRange::Bounded(1024 * 2..1024 * 3 + 12)), 200000),
                expect: vec![(2, 0..1024), (3, 0..12)],
            },
            Test {
                input: (Some(GetRange::Bounded(1024 * 2 - 2..1024 * 3 + 12)), 20000),
                expect: vec![(1, 1022..1024), (2, 0..1024), (3, 0..12)],
            },
            Test {
                input: (Some(GetRange::Suffix(128)), 1024),
                expect: vec![(0, 896..1024)],
            },
            Test {
                input: (Some(GetRange::Suffix(1024 * 2 + 8)), 1024 * 4),
                expect: vec![(1, 1016..1024), (2, 0..1024), (3, 0..1024)],
            },
            Test {
                input: (Some(GetRange::Offset(8)), 1024 * 4),
                expect: vec![(0, 8..1024), (1, 0..1024), (2, 0..1024), (3, 0..1024)],
            },
            Test {
                input: (Some(GetRange::Offset(1024 * 2 + 8)), 1024 * 4),
                expect: vec![(2, 8..1024), (3, 0..1024)],
            },
            Test {
                input: (Some(GetRange::Offset(1024 * 2 + 8)), 1024 * 4 + 2),
                expect: vec![(2, 8..1024), (3, 0..1024), (4, 0..2)],
            },
        ];

        for t in tests.iter() {
            let range = cached_store
                .canonicalize_range(t.input.0.clone(), t.input.1 as u64)
                .unwrap();
            let parts = cached_store.split_range_into_parts(range);
            assert_eq!(parts, t.expect, "input: {:?}", t.input);
        }
    }

    #[test]
    fn test_align_range() {
        let object_store = Arc::new(object_store::memory::InMemory::new());
        let test_cache_folder = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder,
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));
        let cached_store =
            CachedObjectStore::new(object_store, cache_storage, 1024, false, stats).unwrap();

        let aligned = cached_store.align_range(&(9..1025), 1024);
        assert_eq!(aligned, 0..2048);
        let aligned = cached_store.align_range(&(1024 + 1..2048 + 4), 1024);
        assert_eq!(aligned, 1024..3072);
    }

    #[test]
    fn test_align_get_range() {
        let object_store = Arc::new(object_store::memory::InMemory::new());
        let test_cache_folder = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder,
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));
        let cached_store =
            CachedObjectStore::new(object_store, cache_storage, 1024, false, stats).unwrap();

        let aligned = cached_store.align_get_range(&GetRange::Bounded(9..1025));
        assert_eq!(aligned, GetRange::Bounded(0..2048));
        let aligned = cached_store.align_get_range(&GetRange::Bounded(9..2048));
        assert_eq!(aligned, GetRange::Bounded(0..2048));
        let aligned = cached_store.align_get_range(&GetRange::Suffix(12));
        assert_eq!(aligned, GetRange::Suffix(1024));
        let aligned = cached_store.align_get_range(&GetRange::Suffix(1024));
        assert_eq!(aligned, GetRange::Suffix(1024));
        let aligned = cached_store.align_get_range(&GetRange::Offset(1024));
        assert_eq!(aligned, GetRange::Offset(1024));
        let aligned = cached_store.align_get_range(&GetRange::Offset(12));
        assert_eq!(aligned, GetRange::Offset(0));
    }

    #[tokio::test]
    async fn test_cached_object_store_impl_object_store() -> object_store::Result<()> {
        let object_store = Arc::new(object_store::memory::InMemory::new());
        let test_cache_folder = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder.clone(),
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));
        let cached_store =
            CachedObjectStore::new(object_store.clone(), cache_storage, 1024, false, stats)
                .unwrap();

        let test_path = Path::from("/data/testdata1");
        let test_payload = gen_rand_bytes(1024 * 3 + 2);
        object_store
            .put(&test_path, PutPayload::from_bytes(test_payload.clone()))
            .await?;

        // test get entire object
        let test_ranges = vec![
            Some(GetRange::Offset(260817)),
            None,
            Some(GetRange::Bounded(1000..2048)),
            Some(GetRange::Bounded(1000..260817)),
            Some(GetRange::Suffix(10)),
            Some(GetRange::Suffix(260817)),
            Some(GetRange::Offset(1000)),
            Some(GetRange::Offset(0)),
            Some(GetRange::Offset(1028)),
            Some(GetRange::Offset(260817)),
            Some(GetRange::Offset(1024 * 3 + 2)),
            Some(GetRange::Offset(1024 * 3 + 1)),
            #[allow(clippy::reversed_empty_ranges)]
            Some(GetRange::Bounded(2900..2048)),
            Some(GetRange::Bounded(10..10)),
        ];

        // test get a range
        for range in test_ranges.iter() {
            let want = object_store
                .get_opts(
                    &test_path,
                    GetOptions {
                        range: range.clone(),
                        ..Default::default()
                    },
                )
                .await;
            let got = cached_store
                .cached_get_opts(
                    &test_path,
                    GetOptions {
                        range: range.clone(),
                        ..Default::default()
                    },
                    false,
                )
                .await;
            match (want, got) {
                (Ok(want), Ok(got)) => {
                    assert_eq!(want.range, got.range);
                    assert_eq!(want.meta, got.meta);
                    assert_eq!(want.bytes().await?, got.bytes().await?);
                }
                (Err(want), Err(got)) => {
                    if want.to_string().to_lowercase().contains("range") {
                        assert!(got.to_string().to_lowercase().contains("range"));
                    }
                }
                (origin_result, cached_result) => {
                    panic!("expect: {:?}, got: {:?}", origin_result, cached_result);
                }
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_preload_cache() {
        let cache_dir = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            cache_dir,
            Some(10 * 1024 * 1024), // 10MB
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));

        let object_store = Arc::new(object_store::memory::InMemory::new());

        let cached_store =
            CachedObjectStore::new(object_store.clone(), cache_storage, 1024, false, stats)
                .unwrap();

        // Create some test files to preload
        let test_paths = vec![
            Path::from("file1.sst"),
            Path::from("file2.sst"),
            Path::from("file3.sst"),
        ];

        let test_data = gen_rand_bytes(2048); // 2KB per file

        // Put test files in object store
        for path in &test_paths {
            object_store
                .put(path, PutPayload::from_bytes(test_data.clone()))
                .await
                .unwrap();
        }

        // Test preloading with max cache size
        cached_store
            .load_files_to_cache(test_paths.clone(), 10 * 1024) // 10KB limit
            .await
            .unwrap();

        // Verify that files are cached by checking if we can read from cache
        for path in &test_paths {
            let entry = cached_store.cache_storage.entry(path, 1024);
            let cached_parts = entry.cached_parts().await.unwrap();
            assert_eq!(cached_parts.len(), 2); // 2KB = 2 parts of 1024 bytes
        }
    }

    #[tokio::test]
    async fn test_preload_cache_above_limit() {
        let cache_dir = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            cache_dir,
            Some(10 * 1024 * 1024), // 10MB
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));

        let object_store = Arc::new(object_store::memory::InMemory::new());

        let cached_store =
            CachedObjectStore::new(object_store.clone(), cache_storage, 1024, false, stats)
                .unwrap();

        // Create some test files
        let test_paths = vec![Path::from("file1.sst"), Path::from("file2.sst")];

        let test_data = gen_rand_bytes(2048); // 2KB per file

        // Put test files in object store
        for path in &test_paths {
            object_store
                .put(path, PutPayload::from_bytes(test_data.clone()))
                .await
                .unwrap();
        }

        // Test load_files_to_cache with 0 bytes limit (should load nothing)
        cached_store
            .load_files_to_cache(test_paths.clone(), 0)
            .await
            .unwrap();

        // Verify that files are NOT cached since preloading was disabled
        for path in &test_paths {
            let entry = cached_store.cache_storage.entry(path, 1024);
            let cached_parts = entry.cached_parts().await.unwrap();
            assert_eq!(cached_parts.len(), 0); // No parts should be cached
        }
    }

    /// Helper to build a CachedObjectStore backed by an InstrumentedObjectStore so
    /// we can assert on the number of actual object-store requests made.
    fn build_instrumented_cached_store(
        inner: Arc<dyn ObjectStore>,
    ) -> (
        Arc<slatedb_common::metrics::DefaultMetricsRecorder>,
        Arc<CachedObjectStore>,
    ) {
        use crate::instrumented_object_store::{InstrumentedObjectStore, ObjectStoreComponent};
        use crate::object_stores::ObjectStoreType;
        use slatedb_common::metrics::test_recorder_helper;

        let (recorder, helper) = test_recorder_helper();
        let instrumented = Arc::new(InstrumentedObjectStore::new(
            inner,
            &helper,
            ObjectStoreComponent::Db,
            ObjectStoreType::Main,
        ));

        let test_cache_folder = new_test_cache_folder();
        let noop_helper = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&noop_helper));
        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder,
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));

        let cached_store = CachedObjectStore::new(
            instrumented as Arc<dyn ObjectStore>,
            cache_storage,
            1024,
            false,
            stats,
        )
        .unwrap();

        (recorder, cached_store)
    }

    fn get_request_count(
        recorder: &slatedb_common::metrics::DefaultMetricsRecorder,
        api: &str,
    ) -> i64 {
        use crate::instrumented_object_store::stats::REQUEST_COUNT;
        use slatedb_common::metrics::lookup_metric_with_labels;

        let labels = [
            ("component", "db"),
            ("store_type", "main"),
            ("op", "get"),
            ("api", api),
        ];
        lookup_metric_with_labels(recorder, REQUEST_COUNT, &labels).unwrap_or(0)
    }

    #[tokio::test]
    async fn test_single_flight_deduplicates_concurrent_head_requests() {
        // Set up an object in the backing store.
        let mem: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let path = Path::from("data/test_head_dedup");
        mem.put(&path, PutPayload::from_bytes(gen_rand_bytes(512)))
            .await
            .unwrap();

        // Wrap with a gate-controlled store so we can block callers deterministically.
        let gated = Arc::new(GatedObjectStore::new(mem));
        gated.head_gate.close();
        let (recorder, cached_store) = build_instrumented_cached_store(gated.clone());

        // Launch many concurrent head requests for the same path.
        let mut handles = Vec::new();
        for _ in 0..10 {
            let store = cached_store.clone();
            let p = path.clone();
            handles.push(tokio::spawn(
                async move { store.cached_head(&p, true).await },
            ));
        }

        // Wait until exactly 1 caller arrives at the gate (SingleFlight dedup
        // ensures only one caller reaches the head gate).
        gated.head_gate.wait_for_arrivals(1).await;
        assert_eq!(
            gated.head_gate.arrivals(),
            1,
            "SingleFlight should let only 1 through"
        );

        // Release the gate — success path.
        gated.head_gate.release();

        for handle in handles {
            handle.await.unwrap().unwrap();
        }

        // SingleFlight should collapse them into a single actual HEAD request.
        let count = get_request_count(&recorder, "head");
        assert_eq!(
            count, 1,
            "expected 1 actual object store request, got {count}"
        );
    }

    #[tokio::test]
    async fn test_single_flight_deduplicates_concurrent_get_opts_requests() {
        let mem: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let path = Path::from("data/test_get_dedup");
        let payload = gen_rand_bytes(2048);
        mem.put(&path, PutPayload::from_bytes(payload.clone()))
            .await
            .unwrap();

        let gated = Arc::new(GatedObjectStore::new(mem));
        gated.get_opts_gate.close();
        let (recorder, cached_store) = build_instrumented_cached_store(gated.clone());

        // Launch many concurrent get_opts requests for the same path and range.
        let mut handles = Vec::new();
        for _ in 0..10 {
            let store = cached_store.clone();
            let p = path.clone();
            handles.push(tokio::spawn(async move {
                let opts = GetOptions {
                    range: Some(GetRange::Bounded(0..1024)),
                    ..Default::default()
                };
                let result = store.cached_get_opts(&p, opts, false).await?;
                result.bytes().await
            }));
        }

        // Wait for the single winning caller to arrive at the gate.
        gated.get_opts_gate.wait_for_arrivals(1).await;
        assert_eq!(
            gated.get_opts_gate.arrivals(),
            1,
            "SingleFlight should let only 1 through"
        );

        // Release — success.
        gated.get_opts_gate.release();

        for handle in handles {
            handle.await.unwrap().unwrap();
        }

        // The prefetch SingleFlight should collapse all prefetch GETs into one.
        // Part reads may also be deduplicated. Total GET count should be much less than 10.
        let count = get_request_count(&recorder, "get_range");
        assert!(
            count <= 2,
            "expected at most 2 actual object store requests (prefetch + maybe 1 part), got {count}"
        );
    }

    #[tokio::test]
    async fn test_single_flight_allows_independent_paths() {
        // Requests to different paths should NOT be deduplicated.
        let mem: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let paths: Vec<Path> = (0..5)
            .map(|i| Path::from(format!("data/independent_{}", i)))
            .collect();
        for p in &paths {
            mem.put(p, PutPayload::from_bytes(gen_rand_bytes(512)))
                .await
                .unwrap();
        }

        let gated = Arc::new(GatedObjectStore::new(mem));
        gated.head_gate.close();
        let (recorder, cached_store) = build_instrumented_cached_store(gated.clone());

        let mut handles = Vec::new();
        for p in &paths {
            let store = cached_store.clone();
            let p = p.clone();
            handles.push(tokio::spawn(
                async move { store.cached_head(&p, true).await },
            ));
        }

        // Each distinct path has its own SingleFlight key, so all 5 should arrive.
        gated.head_gate.wait_for_arrivals(5).await;
        assert_eq!(
            gated.head_gate.arrivals(),
            5,
            "different keys should each pass through SingleFlight independently"
        );

        // Release all.
        gated.head_gate.release();

        for handle in handles {
            handle.await.unwrap().unwrap();
        }

        // Each distinct path should result in its own request.
        let count = get_request_count(&recorder, "head");
        assert_eq!(
            count, 5,
            "expected 5 actual object store requests (one per path), got {count}"
        );
    }

    #[tokio::test]
    async fn test_single_flight_different_ranges_are_independent() {
        // Requests with different ranges should be treated as separate flights.
        let mem: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let path = Path::from("data/test_range_independent");
        let payload = gen_rand_bytes(4096);
        mem.put(&path, PutPayload::from_bytes(payload.clone()))
            .await
            .unwrap();

        let gated = Arc::new(GatedObjectStore::new(mem));
        gated.get_opts_gate.close();
        let (recorder, cached_store) = build_instrumented_cached_store(gated.clone());

        let ranges = vec![
            Some(GetRange::Bounded(0..1024)),
            Some(GetRange::Bounded(1024..2048)),
            Some(GetRange::Suffix(512)),
        ];

        let mut handles = Vec::new();
        for range in ranges {
            let store = cached_store.clone();
            let p = path.clone();
            handles.push(tokio::spawn(async move {
                let opts = GetOptions {
                    range,
                    ..Default::default()
                };
                store.cached_get_opts(&p, opts, false).await
            }));
        }

        // Each distinct range maps to a different key, so all 3 should arrive.
        gated.get_opts_gate.wait_for_arrivals(3).await;
        assert_eq!(
            gated.get_opts_gate.arrivals(),
            3,
            "different ranges should each pass through SingleFlight independently"
        );

        // Release all.
        gated.get_opts_gate.release();

        for handle in handles {
            handle.await.unwrap().unwrap();
        }

        // Each distinct range key should trigger its own prefetch request.
        let count = get_request_count(&recorder, "get_range");
        assert!(
            count >= 3,
            "expected at least 3 object store requests (one per distinct range), got {count}"
        );
    }

    #[tokio::test]
    async fn test_single_flight_concurrent_callers_see_gate_failure() {
        // When the gate is configured to fail, all concurrent waiters on the
        // same SingleFlight key should receive an error (not hang forever).
        let mem: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let path = Path::from("data/test_gate_failure");
        mem.put(&path, PutPayload::from_bytes(gen_rand_bytes(512)))
            .await
            .unwrap();

        let gated = Arc::new(GatedObjectStore::new(mem));
        gated.head_gate.close();
        let (_, cached_store) = build_instrumented_cached_store(gated.clone());

        // Launch concurrent head requests.
        let mut handles = Vec::new();
        for _ in 0..5 {
            let store = cached_store.clone();
            let p = path.clone();
            handles.push(tokio::spawn(
                async move { store.cached_head(&p, true).await },
            ));
        }

        // Wait for the single winning caller to arrive at the gate.
        gated.head_gate.wait_for_arrivals(1).await;

        // Inject failure, then release.
        gated.head_gate.set_error(|| object_store::Error::Generic {
            store: "test",
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "injected test failure",
            )),
        });
        gated.head_gate.release();

        // All callers should see an error.
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_err(), "expected error when gate injects failure");
        }
    }

    #[tokio::test]
    async fn test_single_flight_retries_after_gate_failure() {
        // After a failure, the SingleFlight should not cache the error,
        // allowing the next call to succeed fresh.
        let mem: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let path = Path::from("data/test_retry_after_fail");
        mem.put(&path, PutPayload::from_bytes(gen_rand_bytes(512)))
            .await
            .unwrap();

        let gated = Arc::new(GatedObjectStore::new(mem));
        gated.head_gate.close();
        let (_, cached_store) = build_instrumented_cached_store(gated.clone());

        // First call — configure failure.
        let store = cached_store.clone();
        let p = path.clone();
        let handle = tokio::spawn(async move { store.cached_head(&p, true).await });

        gated.head_gate.wait_for_arrivals(1).await;
        gated.head_gate.set_error(|| object_store::Error::Generic {
            store: "test",
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "injected test failure",
            )),
        });
        gated.head_gate.release();

        let result = handle.await.unwrap();
        assert!(result.is_err(), "first call should fail");

        // Second call — configure success.
        gated.head_gate.clear_error();
        let store = cached_store.clone();
        let p = path.clone();
        let handle = tokio::spawn(async move { store.cached_head(&p, true).await });

        gated.head_gate.wait_for_arrivals(2).await;
        gated.head_gate.release();

        let result = handle.await.unwrap();
        assert!(result.is_ok(), "second call should succeed after retry");
    }

    #[tokio::test]
    async fn test_single_flight_part_fetch_with_get_range_failures() {
        // Validates that when fetching parts fails transiently, the SingleFlight
        // does not permanently cache the failure, and retries succeed.
        let mem: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let path = Path::from("data/test_part_flaky");
        let payload = gen_rand_bytes(4096);
        mem.put(&path, PutPayload::from_bytes(payload.clone()))
            .await
            .unwrap();

        // Use a FlakyObjectStore that fails the first get_range call.
        let flaky = Arc::new(FlakyObjectStore::new(mem, 0).with_get_range_failures(1));
        let (_, cached_store) = build_instrumented_cached_store(flaky.clone());

        // First, prime metadata via a full get (get_opts doesn't use get_range).
        let prime_opts = GetOptions {
            range: None,
            ..Default::default()
        };
        let result = cached_store
            .cached_get_opts(&path, prime_opts, false)
            .await
            .unwrap();
        let _ = result.bytes().await.unwrap();

        // Now try reading — the parts should be cached from the full get above,
        // so even though get_range is flaky, we should succeed from cache.
        let opts = GetOptions {
            range: Some(GetRange::Bounded(0..512)),
            ..Default::default()
        };
        let result = cached_store.cached_get_opts(&path, opts, false).await;
        assert!(result.is_ok());
        let bytes = result.unwrap().bytes().await.unwrap();
        assert_eq!(&bytes[..], &payload[..512]);
    }

    #[rstest::rstest]
    #[case::no_evictor_cached(false, true)]
    #[case::with_evictor_cached(true, true)]
    #[case::no_evictor_uncached(false, false)]
    #[case::with_evictor_uncached(true, false)]
    #[tokio::test]
    async fn test_delete(#[case] evictor: bool, #[case] cached: bool) {
        const PART_SIZE: usize = 1024;

        let location1 = Path::from("/data/testfile1");
        let location2 = Path::from("/data/testfile2");

        let test_cache_folder = new_test_cache_folder();
        let payload = gen_rand_bytes(PART_SIZE * 3);
        let object_store = Arc::new(object_store::memory::InMemory::new());
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));

        object_store
            .put(&location1, PutPayload::from_bytes(payload.clone()))
            .await
            .unwrap();
        object_store
            .put(&location2, PutPayload::from_bytes(payload.clone()))
            .await
            .unwrap();

        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder.clone(),
            evictor.then_some(1024 * 1024),
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));

        let cached_store = CachedObjectStore::new(
            object_store,
            Arc::clone(&cache_storage) as Arc<dyn LocalCacheStorage>,
            PART_SIZE,
            false,
            stats,
        )
        .unwrap();
        cached_store.start_evictor().await;

        // Populate the cache through the normal read path. Untagged reads bypass
        // the cache, so tag these as a cacheable (main, compacted) read.
        let cacheable = || GetOptions {
            extensions: ObjectStoreCallTag::new(TableStoreKind::Main, SstType::Compacted).into(),
            ..GetOptions::default()
        };
        if cached {
            cached_store
                .get_opts(&location1, cacheable())
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap();
        }
        cached_store
            .get_opts(&location2, cacheable())
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();

        let entry1 = cached_store.cache_storage.entry(&location1, PART_SIZE);
        let parts1 = entry1.cached_parts().await.unwrap();
        if cached {
            assert_eq!(parts1.len(), 3, "{parts1:?}");
            assert_eq!(cache_storage.file_handle_cache_population(), 6);
        } else {
            assert_eq!(parts1.len(), 0, "{parts1:?}");
            assert_eq!(cache_storage.file_handle_cache_population(), 3);
        }

        let entry2 = cached_store.cache_storage.entry(&location2, PART_SIZE);
        let parts2 = entry2.cached_parts().await.unwrap();
        assert_eq!(parts2.len(), 3, "{parts2:?}");

        cached_store.delete(&location1).await.unwrap();
        if evictor {
            // XXX: If evictor is running, deletion is performed asynchronously
            //      from the evictor "thread".
            tokio::time::sleep(Duration::from_secs(3)).await;
        }

        let entry1 = cached_store.cache_storage.entry(&location1, PART_SIZE);
        let parts1 = entry1.cached_parts().await.unwrap();
        assert_eq!(parts1.len(), 0, "{parts1:?}");
        assert_eq!(cache_storage.file_handle_cache_population(), 3);

        let entry2 = cached_store.cache_storage.entry(&location2, PART_SIZE);
        let parts2 = entry2.cached_parts().await.unwrap();
        assert_eq!(parts2.len(), 3, "{parts2:?}");

        // verify repeated delete is idempotent
        cached_store.delete(&location1).await.unwrap();
        let entry1 = cached_store.cache_storage.entry(&location1, PART_SIZE);
        let parts1 = entry1.cached_parts().await.unwrap();
        assert_eq!(parts1.len(), 0, "{parts1:?}");
        assert_eq!(cache_storage.file_handle_cache_population(), 3);
    }

    fn policy_test_store(
        upstream: Arc<dyn ObjectStore>,
        cache_puts: bool,
    ) -> Arc<CachedObjectStore> {
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            new_test_cache_folder(),
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));
        CachedObjectStore::new(upstream, cache_storage, 1024, cache_puts, stats).unwrap()
    }

    fn put_opts_tagged(tag: ObjectStoreCallTag) -> object_store::PutOptions {
        object_store::PutOptions {
            extensions: tag.into(),
            ..Default::default()
        }
    }

    fn get_opts_tagged(tag: ObjectStoreCallTag) -> GetOptions {
        GetOptions {
            extensions: tag.into(),
            ..Default::default()
        }
    }

    async fn cached_part_count(store: &CachedObjectStore, location: &Path) -> usize {
        let cache_location = location.clone();
        store
            .cache_storage
            .entry(&cache_location, store.part_size_bytes)
            .cached_parts()
            .await
            .unwrap()
            .len()
    }

    #[rstest]
    // WAL writes are never cached, even with cache_puts on.
    #[case(ObjectStoreCallTag::new(TableStoreKind::Main, SstType::Wal), true, 0)]
    // Compacted writes from the main store (flush) and the compactor are cached
    // when cache_puts is set, and not otherwise.
    #[case(
        ObjectStoreCallTag::new(TableStoreKind::Main, SstType::Compacted),
        true,
        2
    )]
    #[case(
        ObjectStoreCallTag::new(TableStoreKind::Main, SstType::Compacted),
        false,
        0
    )]
    #[case(
        ObjectStoreCallTag::new(TableStoreKind::Compactor, SstType::Compacted),
        true,
        2
    )]
    #[case(
        ObjectStoreCallTag::new(TableStoreKind::Compactor, SstType::Compacted),
        false,
        0
    )]
    #[tokio::test]
    async fn test_put_caching_by_tag(
        #[case] tag: ObjectStoreCallTag,
        #[case] cache_puts: bool,
        #[case] expected_parts: usize,
    ) {
        let upstream: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let store = policy_test_store(upstream.clone(), cache_puts);

        let location = Path::from("compacted/01.sst");
        let payload = gen_rand_bytes(2048); // 2 parts of 1024 bytes
        store
            .put_opts(
                &location,
                PutPayload::from_bytes(payload),
                put_opts_tagged(tag),
            )
            .await
            .unwrap();

        assert_eq!(cached_part_count(&store, &location).await, expected_parts);
    }

    #[tokio::test]
    async fn test_untagged_put_is_not_cached() {
        let upstream: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let store = policy_test_store(upstream.clone(), true);

        // No tag in the options: coordination I/O (manifest, etc.) is never cached.
        let location = Path::from("manifest/01.manifest");
        let payload = gen_rand_bytes(2048);
        store
            .put_opts(
                &location,
                PutPayload::from_bytes(payload),
                object_store::PutOptions::default(),
            )
            .await
            .unwrap();

        assert_eq!(cached_part_count(&store, &location).await, 0);
    }

    #[tokio::test]
    async fn test_compactor_get_bypasses_cache() {
        let upstream: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let store = policy_test_store(upstream.clone(), false);

        let location = Path::from("compacted/01.sst");
        let payload = gen_rand_bytes(2048);
        upstream
            .put(&location, PutPayload::from_bytes(payload.clone()))
            .await
            .unwrap();

        // A compactor read returns the bytes but caches nothing.
        let got = store
            .get_opts(
                &location,
                get_opts_tagged(ObjectStoreCallTag::new(
                    TableStoreKind::Compactor,
                    SstType::Compacted,
                )),
            )
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(got, payload);
        assert_eq!(cached_part_count(&store, &location).await, 0);

        // A main read of the same object caches it: the bypass is compactor specific.
        store
            .get_opts(
                &location,
                get_opts_tagged(ObjectStoreCallTag::new(
                    TableStoreKind::Main,
                    SstType::Compacted,
                )),
            )
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(cached_part_count(&store, &location).await, 2);
    }

    #[rstest]
    #[case::no_evictor(None)]
    #[case::with_evictor(Some(64 * 1024 * 1024))]
    #[tokio::test]
    async fn test_retry_get_refetches_stale_part(#[case] max_cache_size_bytes: Option<usize>) {
        let upstream: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            new_test_cache_folder(),
            max_cache_size_bytes,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
        ));
        let store =
            CachedObjectStore::new(upstream.clone(), cache_storage, 1024, false, stats).unwrap();
        store.start_evictor().await;

        let location = Path::from("compacted/01.sst");
        // 512 bytes is below the 1024 byte part size, so the object is a single
        // part (part 0) and `cached_part_count` is 1.
        let good = gen_rand_bytes(512);
        upstream
            .put(&location, PutPayload::from_bytes(good.clone()))
            .await
            .unwrap();

        // Populate the cache with the correct head and part via a normal read.
        let main_tag = ObjectStoreCallTag::new(TableStoreKind::Main, SstType::Compacted);
        let got = store
            .get_opts(&location, get_opts_tagged(main_tag))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(got, good);
        assert_eq!(cached_part_count(&store, &location).await, 1);

        // Poison part 0 on disk so the cache would otherwise serve corrupt bytes.
        let bad = gen_rand_bytes(512);
        let cache_location = location.clone();
        store
            .cache_storage
            .entry(&cache_location, store.part_size_bytes)
            .save_part(0, bad.clone())
            .await
            .unwrap();

        // A normal read now serves the poisoned bytes (cache hit).
        let served = store
            .get_opts(&location, get_opts_tagged(main_tag))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(served, bad);

        // A reissued (retry) read force-refreshes from upstream, healing the part.
        let retry_tag = ObjectStoreCallTag {
            kind: TableStoreKind::Main,
            sst_type: SstType::Compacted,
            retry: Some(crate::error::RetryReason::CrcMismatch),
        };
        let refetched = store
            .get_opts(&location, get_opts_tagged(retry_tag))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(
            refetched, good,
            "retry should refetch durable upstream bytes"
        );

        // The cache now holds the corrected part: a later normal read serves good bytes.
        let after = store
            .get_opts(&location, get_opts_tagged(main_tag))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(after, good);
    }

    #[tokio::test]
    async fn test_compactor_head_reads_without_admitting() {
        let upstream: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let store = policy_test_store(upstream.clone(), false);

        let location = Path::from("compacted/01.sst");
        let payload = gen_rand_bytes(512);
        upstream
            .put(&location, PutPayload::from_bytes(payload.clone()))
            .await
            .unwrap();
        let cache_location = location.clone();
        let read_head = || {
            let entry = store
                .cache_storage
                .entry(&cache_location, store.part_size_bytes);
            async move { entry.read_head().await.unwrap() }
        };
        let compactor_head = || GetOptions {
            head: true,
            extensions: ObjectStoreCallTag::new(TableStoreKind::Compactor, SstType::Compacted)
                .into(),
            ..Default::default()
        };

        // On a miss the compactor HEAD serves the metadata but must not admit a
        // head-only entry, which would defeat a later foreground range prefetch.
        let result = store.get_opts(&location, compactor_head()).await.unwrap();
        assert_eq!(result.meta.size, payload.len() as u64);
        assert!(
            read_head().await.is_none(),
            "compactor HEAD must not admit a head-only entry on a miss"
        );

        // But it still serves an already-cached head: populate via a main HEAD,
        // then the compactor HEAD is served from it.
        let main_head = GetOptions {
            head: true,
            extensions: ObjectStoreCallTag::new(TableStoreKind::Main, SstType::Compacted).into(),
            ..Default::default()
        };
        store.get_opts(&location, main_head).await.unwrap();
        assert!(
            read_head().await.is_some(),
            "a main HEAD reads through and populates the cache head"
        );
        let served = store.get_opts(&location, compactor_head()).await.unwrap();
        assert_eq!(
            served.meta.size,
            payload.len() as u64,
            "compactor HEAD should serve the already-cached head"
        );
    }

    #[tokio::test]
    async fn test_wal_read_bypasses_cache() {
        let upstream: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let store = policy_test_store(upstream.clone(), false);

        let location = Path::from("wal/00000000000000000001.sst");
        let payload = gen_rand_bytes(2048);
        upstream
            .put(&location, PutPayload::from_bytes(payload.clone()))
            .await
            .unwrap();
        let wal_tag = ObjectStoreCallTag::new(TableStoreKind::Main, SstType::Wal);

        // A WAL data read bypasses: it returns the bytes but caches nothing.
        let got = store
            .get_opts(&location, get_opts_tagged(wal_tag))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(got, payload);
        assert_eq!(cached_part_count(&store, &location).await, 0);

        // A WAL HEAD bypasses too: it does not populate the cache head.
        let wal_head = GetOptions {
            head: true,
            extensions: wal_tag.into(),
            ..Default::default()
        };
        store.get_opts(&location, wal_head).await.unwrap();
        let cache_location = location.clone();
        let head = store
            .cache_storage
            .entry(&cache_location, store.part_size_bytes)
            .read_head()
            .await
            .unwrap();
        assert!(head.is_none(), "WAL HEAD must not populate the cache");
    }

    #[tokio::test]
    async fn test_put_writes_head_and_serves_first_read_from_cache() {
        let upstream: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let store = policy_test_store(upstream.clone(), true);

        // A flush write (main store, compacted) is cached and commits a head.
        let location = Path::from("compacted/01.sst");
        let payload = gen_rand_bytes(2048);
        let tag = ObjectStoreCallTag::new(TableStoreKind::Main, SstType::Compacted);
        store
            .put_opts(
                &location,
                PutPayload::from_bytes(payload.clone()),
                put_opts_tagged(tag),
            )
            .await
            .unwrap();

        // The head is written on the write path, with the right size.
        let cache_location = location.clone();
        let head = store
            .cache_storage
            .entry(&cache_location, store.part_size_bytes)
            .read_head()
            .await
            .unwrap();
        assert_eq!(
            head.expect("head should be written on the put").0.size,
            2048
        );

        // With the head and parts cached, the first read is served entirely from
        // the cache: deleting the object upstream must not affect it. (Without a
        // head, the read would prefetch from the now-missing upstream and fail.)
        upstream.delete(&location).await.unwrap();
        let got = store
            .get_opts(&location, get_opts_tagged(tag))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(got, payload);
    }

    fn multipart_opts(tag: ObjectStoreCallTag) -> PutMultipartOptions {
        PutMultipartOptions {
            extensions: tag.into(),
            ..Default::default()
        }
    }

    #[rstest]
    // Uploaded chunks larger than the cache part size: every put_part flushes
    // a full cache part and carries a remainder into the next call.
    #[case(1500, 2, vec![1024, 1024, 952])]
    // Uploaded chunks equal to the cache part size.
    #[case(1024, 2, vec![1024, 1024])]
    // Uploaded chunks smaller than the cache part size.
    #[case(400, 3, vec![1024, 176])]
    // Total upload smaller than the cache part size.
    #[case(400, 2, vec![800])]
    #[tokio::test]
    async fn test_multipart_compacted_upload_is_cached(
        #[case] chunk_size: usize,
        #[case] num_chunks: usize,
        #[case] expected_part_sizes: Vec<usize>,
    ) {
        let upstream: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let store = policy_test_store(upstream.clone(), true);

        // A compaction output written as a multipart upload (the path large
        // compacted SSTs take). The tag survives multipart init, so no fallback
        // is involved.
        let location = Path::from("compacted/big.sst");
        let chunks: Vec<Bytes> = (0..num_chunks)
            .map(|_| gen_rand_bytes(chunk_size))
            .collect();
        let tag = ObjectStoreCallTag::new(TableStoreKind::Compactor, SstType::Compacted);
        let mut upload = store
            .put_multipart_opts(&location, multipart_opts(tag))
            .await
            .unwrap();
        for chunk in &chunks {
            upload.put_part(chunk.clone().into()).await.unwrap();
        }
        upload.complete().await.unwrap();

        let cache_location = location.clone();
        let entry = store
            .cache_storage
            .entry(&cache_location, store.part_size_bytes);
        let cached = entry.cached_parts().await.unwrap();
        let expected_part_ids: Vec<PartID> = (0..expected_part_sizes.len()).collect();
        assert_eq!(cached, expected_part_ids);

        // The teed bytes round-trip in order through cache parts of the
        // expected sizes.
        let expected: Vec<u8> = chunks.iter().flat_map(|c| c.to_vec()).collect();
        assert_eq!(expected_part_sizes.iter().sum::<usize>(), expected.len());
        let mut offset = 0;
        for (part_id, part_size) in expected_part_sizes.iter().enumerate() {
            let bytes = entry
                .read_part(part_id, 0..*part_size)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(&bytes[..], &expected[offset..offset + part_size]);
            offset += part_size;
        }
    }

    #[rstest]
    // A compacted multipart upload is not cached when cache_puts is off.
    #[case(
        ObjectStoreCallTag::new(TableStoreKind::Compactor, SstType::Compacted),
        false
    )]
    // A WAL multipart upload is never cached, even with cache_puts on.
    #[case(ObjectStoreCallTag::new(TableStoreKind::Main, SstType::Wal), true)]
    #[tokio::test]
    async fn test_multipart_upload_not_cached(
        #[case] tag: ObjectStoreCallTag,
        #[case] cache_puts: bool,
    ) {
        let upstream: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let store = policy_test_store(upstream.clone(), cache_puts);

        let location = Path::from("compacted/big.sst");
        let mut upload = store
            .put_multipart_opts(&location, multipart_opts(tag))
            .await
            .unwrap();
        upload.put_part(gen_rand_bytes(2048).into()).await.unwrap();
        upload.complete().await.unwrap();

        assert_eq!(cached_part_count(&store, &location).await, 0);
    }

    #[tokio::test]
    async fn test_multipart_head_is_the_commit_point() {
        let upstream: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let store = policy_test_store(upstream.clone(), true);

        let location = Path::from("compacted/big.sst");
        let cache_location = location.clone();
        let tag = ObjectStoreCallTag::new(TableStoreKind::Compactor, SstType::Compacted);
        let mut upload = store
            .put_multipart_opts(&location, multipart_opts(tag))
            .await
            .unwrap();
        upload.put_part(gen_rand_bytes(2048).into()).await.unwrap();

        // Before complete, parts may be on disk but the entry is not committed:
        // no head, so a read would treat it as a miss and refetch from upstream.
        let head_before = store
            .cache_storage
            .entry(&cache_location, store.part_size_bytes)
            .read_head()
            .await
            .unwrap();
        assert!(
            head_before.is_none(),
            "entry must not be committed before complete"
        );

        upload.complete().await.unwrap();

        // complete writes the head, committing the entry.
        let head_after = store
            .cache_storage
            .entry(&cache_location, store.part_size_bytes)
            .read_head()
            .await
            .unwrap();
        assert_eq!(
            head_after
                .expect("head should be committed on complete")
                .0
                .size,
            2048
        );
    }

    /// HELION LOCAL PATCH: ranged cache miss tests.
    mod ranged_miss {
        use super::super::{
            RangedMiss, MARKED_FILLS_WAITING_KIB, MARKED_FILL_TURN_MAX, RANGED_FILL_BUDGET_KIB,
        };
        use super::*;
        use crate::cached_object_store::policy::{
            CachePutConfig, DefaultGetPolicy, DefaultPutPolicy,
        };
        use crate::cached_object_store::stats::RangedMissStats;
        use futures::stream::{self, BoxStream, StreamExt};
        use object_store::{
            CopyOptions, GetResult, GetResultPayload, ListResult, ObjectMeta, PutOptions,
            PutResult, RenameOptions,
        };
        use rand::RngCore;
        use std::collections::BTreeSet;
        use std::ops::Range;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::sync::Semaphore;

        const MIB: usize = 1024 * 1024;

        /// Upstream for ranged-miss tests. A GET whose range is not a whole
        /// number of parts is "ranged"; every other GET is a part GET (fills,
        /// fallbacks, prefetches). Injects first-byte plus per-KiB latency, and
        /// fails, shortens, breaks or stalls the next N GETs of a kind.
        #[derive(Debug)]
        struct ScriptedStore {
            inner: Arc<dyn ObjectStore>,
            part_size: u64,
            first_byte: Duration,
            per_kib: Duration,
            ranged_gets: AtomicUsize,
            part_gets: AtomicUsize,
            fail_ranged: AtomicUsize,
            fail_part: AtomicUsize,
            short_part: AtomicUsize,
            broken_part: AtomicUsize,
            stall_part: AtomicUsize,
            /// GETs in flight now, and the most there were at once.
            in_flight: AtomicUsize,
            peak_in_flight: AtomicUsize,
            /// Extra latency of a GET by the part it starts in (earlier parts
            /// can be made to finish last).
            part_delay: std::sync::Mutex<fn(usize) -> Duration>,
            /// Fail every GET that starts in this part or a later one.
            fail_parts_from: AtomicUsize,
            /// Part each GET started in, in the order the GETs finished.
            finished: std::sync::Mutex<Vec<usize>>,
        }

        impl ScriptedStore {
            fn new(part_size: usize, first_byte: Duration, per_kib: Duration) -> Arc<Self> {
                Arc::new(Self {
                    inner: Arc::new(object_store::memory::InMemory::new()),
                    part_size: part_size as u64,
                    first_byte,
                    per_kib,
                    ranged_gets: AtomicUsize::new(0),
                    part_gets: AtomicUsize::new(0),
                    fail_ranged: AtomicUsize::new(0),
                    fail_part: AtomicUsize::new(0),
                    short_part: AtomicUsize::new(0),
                    broken_part: AtomicUsize::new(0),
                    stall_part: AtomicUsize::new(0),
                    in_flight: AtomicUsize::new(0),
                    peak_in_flight: AtomicUsize::new(0),
                    part_delay: std::sync::Mutex::new(|_| Duration::ZERO),
                    fail_parts_from: AtomicUsize::new(usize::MAX),
                    finished: std::sync::Mutex::new(vec![]),
                })
            }

            async fn put_object(&self, name: &str, bytes: Bytes) -> Path {
                let path = Path::from(format!("compacted/{name}.sst"));
                self.inner
                    .put(&path, PutPayload::from_bytes(bytes))
                    .await
                    .unwrap();
                path
            }

            fn ranged_gets(&self) -> usize {
                self.ranged_gets.load(Ordering::SeqCst)
            }

            fn part_gets(&self) -> usize {
                self.part_gets.load(Ordering::SeqCst)
            }
        }

        fn take(counter: &AtomicUsize) -> bool {
            counter
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                .is_ok()
        }

        fn injected(message: &'static str) -> object_store::Error {
            object_store::Error::Generic {
                store: "scripted",
                source: message.into(),
            }
        }

        /// Counts a GET as in flight until dropped (returned or cancelled).
        struct InFlight<'a>(&'a AtomicUsize);

        impl Drop for InFlight<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::SeqCst);
            }
        }

        impl std::fmt::Display for ScriptedStore {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "ScriptedStore({})", self.inner)
            }
        }

        #[async_trait::async_trait]
        impl ObjectStore for ScriptedStore {
            async fn get_opts(
                &self,
                location: &Path,
                options: GetOptions,
            ) -> object_store::Result<GetResult> {
                if options.head {
                    return self.inner.get_opts(location, options).await;
                }
                let ranged = matches!(
                    &options.range,
                    Some(GetRange::Bounded(r)) if (r.end - r.start) % self.part_size != 0
                );
                let (gets, fail) = if ranged {
                    (&self.ranged_gets, &self.fail_ranged)
                } else {
                    (&self.part_gets, &self.fail_part)
                };
                gets.fetch_add(1, Ordering::SeqCst);
                let in_flight = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                let _in_flight = InFlight(&self.in_flight);
                self.peak_in_flight.fetch_max(in_flight, Ordering::SeqCst);
                let result = self.inner.get_opts(location, options).await?;
                let kib = u32::try_from((result.range.end - result.range.start) / 1024)
                    .expect("test bodies are small");
                let part_id = (result.range.start / self.part_size) as usize;
                let part_delay = (*self.part_delay.lock().unwrap())(part_id);
                tokio::time::sleep(self.first_byte + self.per_kib * kib + part_delay).await;
                self.finished.lock().unwrap().push(part_id);
                if take(fail) {
                    return Err(injected("injected GET failure"));
                }
                if part_id >= self.fail_parts_from.load(Ordering::SeqCst) {
                    return Err(object_store::Error::Generic {
                        store: "scripted",
                        source: format!("injected failure of part {part_id}").into(),
                    });
                }
                if ranged {
                    return Ok(result);
                }
                let tail: BoxStream<'static, object_store::Result<Bytes>> =
                    if take(&self.short_part) {
                        stream::empty().boxed()
                    } else if take(&self.broken_part) {
                        stream::once(async { Err(injected("injected mid-body failure")) }).boxed()
                    } else if take(&self.stall_part) {
                        stream::pending().boxed()
                    } else {
                        return Ok(result);
                    };
                // Deliver the first half of the body, then the fault.
                let meta = result.meta.clone();
                let range = result.range.clone();
                let attributes = result.attributes.clone();
                let extensions = result.extensions.clone();
                let body = result.bytes().await?;
                let half = body.slice(..body.len() / 2);
                Ok(GetResult {
                    payload: GetResultPayload::Stream(
                        stream::once(async move { Ok(half) }).chain(tail).boxed(),
                    ),
                    meta,
                    range,
                    attributes,
                    extensions,
                })
            }

            async fn put_opts(
                &self,
                location: &Path,
                payload: PutPayload,
                opts: PutOptions,
            ) -> object_store::Result<PutResult> {
                self.inner.put_opts(location, payload, opts).await
            }

            async fn put_multipart_opts(
                &self,
                location: &Path,
                opts: PutMultipartOptions,
            ) -> object_store::Result<Box<dyn MultipartUpload>> {
                self.inner.put_multipart_opts(location, opts).await
            }

            fn delete_stream(
                &self,
                locations: BoxStream<'static, object_store::Result<Path>>,
            ) -> BoxStream<'static, object_store::Result<Path>> {
                self.inner.delete_stream(locations)
            }

            fn list(
                &self,
                prefix: Option<&Path>,
            ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
                self.inner.list(prefix)
            }

            async fn list_with_delimiter(
                &self,
                prefix: Option<&Path>,
            ) -> object_store::Result<ListResult> {
                self.inner.list_with_delimiter(prefix).await
            }

            async fn copy_opts(
                &self,
                from: &Path,
                to: &Path,
                options: CopyOptions,
            ) -> object_store::Result<()> {
                self.inner.copy_opts(from, to, options).await
            }

            async fn rename_opts(
                &self,
                from: &Path,
                to: &Path,
                options: RenameOptions,
            ) -> object_store::Result<()> {
                self.inner.rename_opts(from, to, options).await
            }
        }

        /// A cached store with its own fill budget and stats (the process-wide
        /// ones are shared by every test in the binary).
        struct Fixture {
            store: Arc<CachedObjectStore>,
            stats: &'static RangedMissStats,
            budget: &'static Semaphore,
            budget_kib: usize,
            root: std::path::PathBuf,
        }

        impl Fixture {
            fn new(
                upstream: Arc<ScriptedStore>,
                part_size: usize,
                ranged_miss: bool,
                budget_kib: usize,
            ) -> Self {
                let root = new_test_cache_folder();
                let recorder = MetricsRecorderHelper::noop();
                let cache_stats = Arc::new(CachedObjectStoreStats::new(&recorder));
                let cache_storage = Arc::new(FsCacheStorage::new(
                    root.clone(),
                    None,
                    None,
                    cache_stats.clone(),
                    Arc::new(DefaultSystemClock::new()),
                    Arc::new(DbRand::default()),
                    1000,
                ));
                let mut store = CachedObjectStore::new_with_policies(
                    upstream,
                    cache_storage,
                    part_size,
                    cache_stats,
                    Arc::new(DefaultGetPolicy),
                    Arc::new(DefaultPutPolicy {
                        put: CachePutConfig { cache_puts: false },
                    }),
                    ranged_miss,
                    crate::config::default_part_fetch_concurrency(),
                )
                .unwrap();
                let stats: &'static RangedMissStats = Box::leak(Box::new(RangedMissStats::new()));
                let budget: &'static Semaphore = Box::leak(Box::new(Semaphore::new(budget_kib)));
                if let Some(ranged) = Arc::get_mut(&mut store)
                    .expect("sole owner")
                    .ranged_miss
                    .as_mut()
                {
                    *ranged = RangedMiss {
                        fill_budget: budget,
                        stats,
                    };
                }
                Self {
                    store,
                    stats,
                    budget,
                    budget_kib,
                    root,
                }
            }

            /// The same store, reading `n` parts of one range at once.
            fn part_fetch_concurrency(mut self, n: usize) -> Self {
                Arc::get_mut(&mut self.store)
                    .expect("sole owner")
                    .part_fetch_concurrency = n;
                self
            }

            /// Cache the head but no parts (as after the parts were evicted), so
            /// reads miss in `read_part` instead of prefetching. `cached_head`
            /// would cache the parts too: InMemory answers a HEAD with the body.
            async fn cache_head(&self, upstream: &ScriptedStore, path: &Path) {
                let meta = upstream.inner.head(path).await.unwrap();
                self.store
                    .cache_storage
                    .entry(path, self.store.part_size_bytes)
                    .save_head((&meta, &object_store::Attributes::new()))
                    .await
                    .unwrap();
            }

            async fn read(
                &self,
                path: &Path,
                range: Range<u64>,
                force_refresh: bool,
            ) -> object_store::Result<Bytes> {
                let opts = GetOptions {
                    range: Some(GetRange::Bounded(range)),
                    ..Default::default()
                };
                self.store
                    .cached_get_opts(path, opts, force_refresh)
                    .await?
                    .bytes()
                    .await
            }

            /// Wait until every background fill has returned its permits.
            async fn wait_for_fills(&self) {
                for _ in 0..2000 {
                    if self.budget.available_permits() == self.budget_kib {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
                panic!("background part fills did not finish");
            }

            /// The part file on disk, whatever its length.
            fn part_on_disk(&self, path: &Path, part_id: PartID) -> Option<Bytes> {
                let file = FsCacheEntry::make_part_path(
                    self.root.clone(),
                    path,
                    part_id,
                    self.store.part_size_bytes,
                );
                std::fs::read(file).ok().map(Bytes::from)
            }

            fn count(counter: &std::sync::atomic::AtomicU64) -> u64 {
                counter.load(Ordering::SeqCst)
            }
        }

        impl Drop for Fixture {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.root);
            }
        }

        fn rand_bytes(rand: &DbRand, len: usize) -> Bytes {
            let mut buf = vec![0; len];
            rand.rng().fill_bytes(&mut buf);
            buf.into()
        }

        fn slice(bytes: &Bytes, range: &Range<u64>) -> Bytes {
            bytes.slice(range.start as usize..range.end as usize)
        }

        /// An object of the database at `root`, its head cached and no part.
        async fn object_under(
            fx: &Fixture,
            upstream: &ScriptedStore,
            root: &str,
            bytes: &Bytes,
        ) -> Path {
            let path = Path::from(format!("{root}/compacted/obj.sst"));
            upstream
                .inner
                .put(&path, PutPayload::from_bytes(bytes.clone()))
                .await
                .unwrap();
            fx.cache_head(upstream, &path).await;
            path
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_ranged_miss_random_concurrent_reads_match_upstream_and_fill_parts() {
            const PART: usize = 2 * MIB;
            let seed = rand::rng().random::<u64>();
            let rand = DbRand::new(seed);
            let upstream =
                ScriptedStore::new(PART, Duration::from_millis(2), Duration::from_micros(1));
            let fx = Fixture::new(upstream.clone(), PART, true, RANGED_FILL_BUDGET_KIB);

            // Part aligned; a short last part over 1 MiB (ranged); a short last
            // part under 1 MiB (never ranged); a single short part.
            let sizes = [
                2 * PART,
                2 * PART + MIB + MIB / 2 + 7,
                PART + 333 * 1024,
                MIB + MIB / 5,
            ];
            let mut objects = vec![];
            for (i, size) in sizes.into_iter().enumerate() {
                let bytes = rand_bytes(&rand, size);
                let path = upstream.put_object(&format!("obj{i}"), bytes.clone()).await;
                fx.cache_head(&upstream, &path).await;
                objects.push((path, bytes));
            }

            // Mostly small reads, every fourth up to 1.5 parts long, every third
            // in the first half of part 0 so concurrent readers share parts.
            let plan: Vec<(usize, Range<u64>)> = (0..96)
                .map(|i| {
                    let mut rng = rand.rng();
                    let obj = rng.random_range(0..objects.len());
                    let size = objects[obj].1.len() as u64;
                    let len = if i % 4 == 0 {
                        rng.random_range(1..=(PART + PART / 2) as u64)
                    } else {
                        rng.random_range(1..=64 * 1024)
                    };
                    let start = if i % 3 == 0 {
                        rng.random_range(0..(PART / 2) as u64)
                    } else {
                        rng.random_range(0..size)
                    };
                    (obj, start..(start + len).min(size))
                })
                .collect();

            let reads: Vec<_> = plan
                .iter()
                .map(|(obj, range)| {
                    let store = fx.store.clone();
                    let path = objects[*obj].0.clone();
                    let opts = GetOptions {
                        range: Some(GetRange::Bounded(range.clone())),
                        ..Default::default()
                    };
                    tokio::spawn(async move {
                        store
                            .cached_get_opts(&path, opts, false)
                            .await?
                            .bytes()
                            .await
                    })
                })
                .collect();
            for ((obj, range), read) in plan.iter().zip(reads) {
                let got = read.await.unwrap().unwrap();
                assert_eq!(
                    got,
                    slice(&objects[*obj].1, range),
                    "seed {seed}: obj{obj} {range:?}"
                );
            }
            fx.wait_for_fills().await;

            // Every part a read touched is on disk, exactly as upstream has it.
            let part = PART as u64;
            let touched: BTreeSet<(usize, usize)> = plan
                .iter()
                .flat_map(|(obj, range)| {
                    (range.start / part..=(range.end - 1) / part).map(|p| (*obj, p as usize))
                })
                .collect();
            for &(obj, part_id) in &touched {
                let bytes = &objects[obj].1;
                let end = ((part_id + 1) * PART).min(bytes.len());
                assert_eq!(
                    fx.part_on_disk(&objects[obj].0, part_id),
                    Some(bytes.slice(part_id * PART..end)),
                    "seed {seed}: obj{obj} part {part_id}"
                );
            }

            let misses = Fixture::count(&fx.stats.misses);
            assert!(misses > 0, "seed {seed}: no read took the ranged path");
            assert_eq!(upstream.ranged_gets() as u64, misses, "seed {seed}");
            assert_eq!(Fixture::count(&fx.stats.fills_ok), misses, "seed {seed}");
            assert_eq!(Fixture::count(&fx.stats.fills_err), 0, "seed {seed}");
            assert_eq!(Fixture::count(&fx.stats.fills_skipped), 0, "seed {seed}");

            // The same reads again are all disk hits.
            let gets = (upstream.ranged_gets(), upstream.part_gets());
            for (obj, range) in &plan {
                let got = fx
                    .read(&objects[*obj].0, range.clone(), false)
                    .await
                    .unwrap();
                assert_eq!(got, slice(&objects[*obj].1, range), "seed {seed}");
            }
            assert_eq!(
                (upstream.ranged_gets(), upstream.part_gets()),
                gets,
                "seed {seed}: warm reads went upstream"
            );
        }

        #[tokio::test]
        async fn test_ranged_miss_failed_get_falls_back_to_the_full_part() {
            const PART: usize = 2 * MIB;
            // The part GET outlasts the ranged GET, so the fallback finds the
            // background fill still in flight.
            let upstream =
                ScriptedStore::new(PART, Duration::from_millis(5), Duration::from_micros(5));
            let bytes = rand_bytes(&DbRand::default(), 3 * PART);
            let path = upstream.put_object("obj", bytes.clone()).await;

            // The fallback waits for the background fill: one part GET.
            let fx = Fixture::new(upstream.clone(), PART, true, RANGED_FILL_BUDGET_KIB);
            fx.cache_head(&upstream, &path).await;
            upstream.fail_ranged.store(1, Ordering::SeqCst);
            let range = PART as u64 + 100..PART as u64 + 4196;
            assert_eq!(
                fx.read(&path, range.clone(), false).await.unwrap(),
                slice(&bytes, &range)
            );
            fx.wait_for_fills().await;
            assert_eq!(upstream.ranged_gets(), 1);
            assert_eq!(
                upstream.part_gets(),
                1,
                "the fallback should share the fill"
            );
            assert_eq!(Fixture::count(&fx.stats.fills_ok), 1);
            assert_eq!(fx.part_on_disk(&path, 1), Some(bytes.slice(PART..2 * PART)));

            // With the fill budget spent, the fallback fetches the part itself.
            let fx = Fixture::new(upstream.clone(), PART, true, 0);
            fx.cache_head(&upstream, &path).await;
            upstream.fail_ranged.store(1, Ordering::SeqCst);
            let range = 100..4196;
            assert_eq!(
                fx.read(&path, range.clone(), false).await.unwrap(),
                slice(&bytes, &range)
            );
            assert_eq!(upstream.ranged_gets(), 2);
            assert_eq!(upstream.part_gets(), 2);
            assert_eq!(Fixture::count(&fx.stats.fills_skipped), 1);
            assert_eq!(fx.part_on_disk(&path, 0), Some(bytes.slice(0..PART)));
        }

        /// Fills started under a marked batched row read of their database
        /// take turns. Three marks and 72 reads at once over 24 parts, three
        /// rows a part as a batch has them, with the turn held by the test:
        /// every read gets its bytes by its own ranged GET and waits for no
        /// part, and each part's fill waits once. Past the most that may wait
        /// a fill is left out and counted. A read with no mark in flight
        /// fills its part the usual way, and that part is not fetched again
        /// when its turn comes. With the turn free the fills run one at a
        /// time, after the marks are gone, until every part is on disk; the
        /// second ask of the same rows then costs no GET. A fill whose turn
        /// comes with the budget spent is left out, and with no mark in
        /// flight a spent budget is `fills_skipped`, as it was.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_fills_started_under_a_batched_row_read_take_turns() {
            const PART: usize = MIB;
            const PARTS: usize = 24;
            /// Waits until no fill waits its turn or has it, and returns the
            /// most of the budget seen in use meanwhile, in KiB.
            async fn turns_taken(fx: &Fixture) -> usize {
                let mut most = 0;
                for _ in 0..8000 {
                    most = most.max(fx.budget_kib - fx.budget.available_permits());
                    if fx.stats.fills_waiting.lock().unwrap().is_empty()
                        && fx.stats.fill_turn.available_permits() == 1
                    {
                        return most;
                    }
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
                panic!("the fills that waited their turn did not finish");
            }
            // A part GET takes about 20 ms and a ranged GET 2 ms.
            let upstream =
                ScriptedStore::new(PART, Duration::from_millis(2), Duration::from_micros(18));
            let bytes = rand_bytes(&DbRand::default(), (PARTS + 2) * PART);
            let fx = Fixture::new(upstream.clone(), PART, true, RANGED_FILL_BUDGET_KIB);
            let root = Path::from("prod/ours");
            let path = object_under(&fx, &upstream, "prod/ours", &bytes).await;
            let row = |part: usize, nth: usize| {
                let start = (part * PART + 100 + nth * 4096) as u64;
                start..start + 3072
            };

            let turn = fx.stats.fill_turn.acquire().await.unwrap();
            // Every task that waits its turn holds one clone of the store.
            let stores = Arc::strong_count(&upstream);
            let marks: Vec<_> = (0..3)
                .map(|_| fx.stats.batch_row_read(root.clone()))
                .collect();
            let reads: Vec<_> = (0..3 * PARTS)
                .map(|at| {
                    let (store, path) = (fx.store.clone(), path.clone());
                    let range = row(at % PARTS, at / PARTS);
                    let opts = GetOptions {
                        range: Some(GetRange::Bounded(range.clone())),
                        ..Default::default()
                    };
                    let read = tokio::spawn(async move {
                        store
                            .cached_get_opts(&path, opts, false)
                            .await?
                            .bytes()
                            .await
                    });
                    (range, read)
                })
                .collect();
            for (range, read) in reads {
                assert_eq!(read.await.unwrap().unwrap(), slice(&bytes, &range));
            }
            assert_eq!(
                (upstream.ranged_gets(), upstream.part_gets()),
                (3 * PARTS, 0),
                "every read took its own range and no part was fetched"
            );
            assert_eq!(fx.stats.fills_waiting.lock().unwrap().len(), PARTS);
            assert_eq!(
                Arc::strong_count(&upstream) - stores,
                PARTS,
                "one fill a part waits its turn, however many rows of the part were read"
            );

            // As many waiting as may wait: the next fill is left out.
            fx.stats.fills_waiting.lock().unwrap().extend(
                (PARTS..MARKED_FILLS_WAITING_KIB / (PART / 1024))
                    .map(|n| (Path::from(format!("filler{n}")), 0)),
            );
            let past = row(PARTS, 0);
            assert_eq!(
                fx.read(&path, past.clone(), false).await.unwrap(),
                slice(&bytes, &past)
            );
            assert_eq!(Fixture::count(&fx.stats.fills_skipped_in_batch), 1);
            fx.stats
                .fills_waiting
                .lock()
                .unwrap()
                .retain(|(waits, _)| *waits == path);
            drop(marks);

            // With no mark in flight a miss fills its part the usual way,
            // though that part's fill still waits its turn.
            let unmarked = row(0, 3);
            assert_eq!(
                fx.read(&path, unmarked.clone(), false).await.unwrap(),
                slice(&bytes, &unmarked)
            );
            fx.wait_for_fills().await;
            assert_eq!(upstream.part_gets(), 1);

            drop(turn);
            assert!(
                turns_taken(&fx).await <= PART / 1024,
                "two fills that waited their turn ran at once"
            );
            assert_eq!(
                upstream.part_gets(),
                PARTS,
                "each part fetched once, part 0 by the read with no mark"
            );
            assert_eq!(Fixture::count(&fx.stats.fills_ok), PARTS as u64);
            for part in 0..PARTS {
                assert!(fx.part_on_disk(&path, part).is_some(), "part {part}");
            }
            let gets = (upstream.ranged_gets(), upstream.part_gets());
            for at in 0..3 * PARTS {
                let range = row(at % PARTS, at / PARTS);
                assert_eq!(
                    fx.read(&path, range.clone(), false).await.unwrap(),
                    slice(&bytes, &range)
                );
            }
            assert_eq!(
                (upstream.ranged_gets(), upstream.part_gets()),
                gets,
                "the second ask reads from disk"
            );

            // A fill whose turn comes with the budget spent is left out.
            let spent = fx.budget.try_acquire_many(fx.budget_kib as u32).unwrap();
            let mark = fx.stats.batch_row_read(root);
            let last = row(PARTS + 1, 0);
            assert_eq!(
                fx.read(&path, last.clone(), false).await.unwrap(),
                slice(&bytes, &last)
            );
            turns_taken(&fx).await;
            drop(mark);
            assert_eq!(Fixture::count(&fx.stats.fills_skipped_in_batch), 2);
            assert_eq!(Fixture::count(&fx.stats.fills_skipped), 0);
            // With no mark in flight a spent budget is `fills_skipped`, as it was.
            let last = row(PARTS + 1, 1);
            assert_eq!(
                fx.read(&path, last.clone(), false).await.unwrap(),
                slice(&bytes, &last)
            );
            drop(spent);
            assert_eq!(Fixture::count(&fx.stats.fills_skipped), 1);
            assert_eq!(Fixture::count(&fx.stats.fills_skipped_in_batch), 2);
            assert_eq!(upstream.part_gets(), PARTS);
        }

        /// A mark is its database's. With the turn held and two marks of one
        /// root in flight, a miss of an object of another root, however alike
        /// the two are spelled, fills its part at once as with no mark, where
        /// a miss of the marked root waits its turn. A mark that goes takes
        /// its own root's mark with it and no other's: the root stays marked
        /// until the last of its marks is gone, and then its misses fill at
        /// once again.
        #[tokio::test]
        async fn test_a_mark_makes_only_the_fills_of_its_own_root_wait() {
            const PART: usize = MIB;
            let upstream =
                ScriptedStore::new(PART, Duration::from_millis(2), Duration::from_micros(18));
            let bytes = rand_bytes(&DbRand::default(), 3 * PART);
            let fx = Fixture::new(upstream.clone(), PART, true, RANGED_FILL_BUDGET_KIB);
            let ours = object_under(&fx, &upstream, "prod/ours", &bytes).await;
            let row = |part: usize| (part * PART + 100) as u64..(part * PART + 3172) as u64;
            let waiting = || fx.stats.fills_waiting.lock().unwrap().clone();

            let marked = |root: &str| {
                let location = Path::from(format!("{root}/compacted/obj.sst"));
                fx.stats.in_batch_row_read(&location)
            };

            let _turn = fx.stats.fill_turn.acquire().await.unwrap();
            let first = fx.stats.batch_row_read("prod/ours".into());
            let other = fx.stats.batch_row_read("prod/else".into());
            let second = fx.stats.batch_row_read("prod/ours".into());
            assert!(marked("prod/ours") && marked("prod/else"));
            for (fetched, root) in ["prod/ours2", "prod/theirs", "prod"]
                .into_iter()
                .enumerate()
            {
                let theirs = object_under(&fx, &upstream, root, &bytes).await;
                assert_eq!(
                    fx.read(&theirs, row(0), false).await.unwrap(),
                    slice(&bytes, &row(0))
                );
                fx.wait_for_fills().await;
                assert_eq!(upstream.part_gets(), fetched + 1, "{root}: fetched at once");
                assert!(fx.part_on_disk(&theirs, 0).is_some(), "{root}");
            }
            assert!(waiting().is_empty());

            // The other root's mark goes, then one of this root's two.
            drop(other);
            assert!(marked("prod/ours") && !marked("prod/else"));
            for (part, mark) in [None, Some(first)].into_iter().enumerate() {
                drop(mark);
                assert_eq!(
                    fx.read(&ours, row(part), false).await.unwrap(),
                    slice(&bytes, &row(part))
                );
                fx.wait_for_fills().await;
                assert_eq!(upstream.part_gets(), 3, "part {part} waits its turn");
                assert_eq!(
                    waiting(),
                    (0..=part).map(|part| (ours.clone(), part)).collect()
                );
            }
            drop(second);
            assert!(!marked("prod/ours"));
            assert_eq!(
                fx.read(&ours, row(2), false).await.unwrap(),
                slice(&bytes, &row(2))
            );
            fx.wait_for_fills().await;
            assert_eq!(upstream.part_gets(), 4, "no mark of the root is left");
            assert!(fx.part_on_disk(&ours, 2).is_some());
        }

        /// A fill keeps its turn `MARKED_FILL_TURN_MAX` at most. With the
        /// fetch of one part hung, the fill of the next waits that long and
        /// no longer, and then runs beside it.
        #[tokio::test]
        async fn test_a_fill_that_hangs_gives_its_turn_up() {
            const PART: usize = MIB;
            let upstream =
                ScriptedStore::new(PART, Duration::from_millis(2), Duration::from_micros(18));
            let bytes = rand_bytes(&DbRand::default(), 2 * PART);
            let fx = Fixture::new(upstream.clone(), PART, true, RANGED_FILL_BUDGET_KIB);
            let path = object_under(&fx, &upstream, "prod/ours", &bytes).await;
            let _mark = fx.stats.batch_row_read("prod/ours".into());
            upstream.stall_part.store(1, Ordering::SeqCst);
            let started = std::time::Instant::now();
            for part in 0..2 {
                let row = (part * PART + 100) as u64..(part * PART + 3172) as u64;
                assert_eq!(
                    fx.read(&path, row.clone(), false).await.unwrap(),
                    slice(&bytes, &row)
                );
                // Part 0's fill has the turn before part 1 is missed.
                while upstream.part_gets() == 0 {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            }
            tokio::time::sleep(MARKED_FILL_TURN_MAX / 4).await;
            assert_eq!(
                upstream.part_gets(),
                1,
                "part 1 waits while part 0 has the turn"
            );
            while Fixture::count(&fx.stats.fills_ok) == 0 {
                assert!(
                    started.elapsed() < MARKED_FILL_TURN_MAX * 5,
                    "part 1 never had its turn"
                );
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            assert!(started.elapsed() >= MARKED_FILL_TURN_MAX);
            assert_eq!(upstream.part_gets(), 2);
            assert!(fx.part_on_disk(&path, 1).is_some());
            assert!(
                fx.part_on_disk(&path, 0).is_none(),
                "part 0's fetch hangs still"
            );
        }

        /// The most that may wait is counted by its bytes: of 4 MiB parts a
        /// quarter as many as of 1 MiB parts.
        #[tokio::test]
        async fn test_the_fills_that_may_wait_are_bounded_by_their_bytes() {
            const PART: usize = 4 * MIB;
            let most = MARKED_FILLS_WAITING_KIB / (PART / 1024);
            assert_eq!(most, 256);
            let upstream = ScriptedStore::new(PART, Duration::ZERO, Duration::ZERO);
            let bytes = rand_bytes(&DbRand::default(), 2 * PART);
            let fx = Fixture::new(upstream.clone(), PART, true, RANGED_FILL_BUDGET_KIB);
            let path = object_under(&fx, &upstream, "prod/ours", &bytes).await;
            let _turn = fx.stats.fill_turn.acquire().await.unwrap();
            let _mark = fx.stats.batch_row_read("prod/ours".into());
            fx.stats
                .fills_waiting
                .lock()
                .unwrap()
                .extend((1..most).map(|n| (Path::from(format!("filler{n}")), 0)));
            for (part, left_out) in [(0, 0), (1, 1)] {
                let row = (part * PART + 100) as u64..(part * PART + 3172) as u64;
                assert_eq!(
                    fx.read(&path, row.clone(), false).await.unwrap(),
                    slice(&bytes, &row)
                );
                assert_eq!(Fixture::count(&fx.stats.fills_skipped_in_batch), left_out);
                assert_eq!(fx.stats.fills_waiting.lock().unwrap().len(), most);
            }
        }

        #[rstest]
        #[case::get_error("error")]
        #[case::short_body("short")]
        #[case::broken_body("broken")]
        #[tokio::test]
        async fn test_ranged_miss_failed_fill_caches_nothing_and_next_miss_refetches(
            #[case] fault: &str,
        ) {
            const PART: usize = 2 * MIB;
            let upstream = ScriptedStore::new(PART, Duration::from_millis(2), Duration::ZERO);
            // The read lands in the short (1 MiB + 5 byte) last part.
            let bytes = rand_bytes(&DbRand::default(), 2 * PART + MIB + 5);
            let path = upstream.put_object("obj", bytes.clone()).await;
            let fx = Fixture::new(upstream.clone(), PART, true, RANGED_FILL_BUDGET_KIB);
            fx.cache_head(&upstream, &path).await;
            let counter = match fault {
                "error" => &upstream.fail_part,
                "short" => &upstream.short_part,
                _ => &upstream.broken_part,
            };
            counter.store(1, Ordering::SeqCst);

            let range = 2 * PART as u64 + 10..2 * PART as u64 + 4106;
            assert_eq!(
                fx.read(&path, range.clone(), false).await.unwrap(),
                slice(&bytes, &range)
            );
            fx.wait_for_fills().await;
            assert_eq!(Fixture::count(&fx.stats.fills_err), 1);
            assert_eq!(fx.part_on_disk(&path, 2), None, "{fault}: bad part cached");

            // The next miss takes the ranged path again and this fill succeeds.
            assert_eq!(
                fx.read(&path, range.clone(), false).await.unwrap(),
                slice(&bytes, &range)
            );
            fx.wait_for_fills().await;
            assert_eq!(upstream.ranged_gets(), 2);
            assert_eq!(Fixture::count(&fx.stats.fills_ok), 1);
            assert_eq!(fx.part_on_disk(&path, 2), Some(bytes.slice(2 * PART..)));

            let gets = (upstream.ranged_gets(), upstream.part_gets());
            assert_eq!(
                fx.read(&path, range.clone(), false).await.unwrap(),
                slice(&bytes, &range)
            );
            assert_eq!((upstream.ranged_gets(), upstream.part_gets()), gets);
        }

        /// A runtime dropped while the fill is mid-body stands in for a crash.
        #[test]
        fn test_ranged_miss_fill_cut_off_mid_body_leaves_no_part() {
            const PART: usize = 2 * MIB;
            let upstream = ScriptedStore::new(PART, Duration::from_millis(1), Duration::ZERO);
            let fx = Fixture::new(upstream.clone(), PART, true, RANGED_FILL_BUDGET_KIB);
            let bytes = rand_bytes(&DbRand::default(), 2 * PART);
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap();
            let path = runtime.block_on(async {
                let path = upstream.put_object("obj", bytes.clone()).await;
                fx.cache_head(&upstream, &path).await;
                upstream.stall_part.store(1, Ordering::SeqCst);
                let range = 100..4196;
                assert_eq!(
                    fx.read(&path, range.clone(), false).await.unwrap(),
                    slice(&bytes, &range)
                );
                while upstream.part_gets() == 0 {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
                path
            });
            assert!(
                fx.budget.available_permits() < fx.budget_kib,
                "the fill should still be in flight"
            );

            drop(runtime);

            assert_eq!(fx.budget.available_permits(), fx.budget_kib);
            assert_eq!(fx.part_on_disk(&path, 0), None);
            let dir = FsCacheEntry::make_part_path(fx.root.clone(), &path, 0, PART);
            let parts: Vec<_> = std::fs::read_dir(dir.parent().unwrap())
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .filter(|name| name.starts_with("_part"))
                .collect();
            assert!(parts.is_empty(), "{parts:?}");
        }

        #[tokio::test]
        async fn test_ranged_miss_waits_for_an_in_flight_fill() {
            const PART: usize = 2 * MIB;
            // Ranged GET ~10 ms, part fill ~30 ms.
            let upstream =
                ScriptedStore::new(PART, Duration::from_millis(10), Duration::from_micros(10));
            let bytes = rand_bytes(&DbRand::default(), PART);
            let path = upstream.put_object("obj", bytes.clone()).await;
            let fx = Fixture::new(upstream.clone(), PART, true, RANGED_FILL_BUDGET_KIB);
            fx.cache_head(&upstream, &path).await;

            let first = 100..4196;
            assert_eq!(
                fx.read(&path, first.clone(), false).await.unwrap(),
                slice(&bytes, &first)
            );
            assert!(fx.store.part_flights.contains(&(path.clone(), 0)));
            assert_eq!(Fixture::count(&fx.stats.waits), 0);

            // A second small miss on the same part waits for the fill.
            let second = 500_000..520_000;
            assert_eq!(
                fx.read(&path, second.clone(), false).await.unwrap(),
                slice(&bytes, &second)
            );
            assert_eq!(Fixture::count(&fx.stats.waits), 1);
            assert_eq!(upstream.ranged_gets(), 1);
            assert_eq!(upstream.part_gets(), 1);
            fx.wait_for_fills().await;
            assert_eq!(fx.part_on_disk(&path, 0), Some(bytes.clone()));
        }

        #[tokio::test]
        async fn test_ranged_miss_only_for_small_reads_of_large_parts() {
            const PART: usize = 2 * MIB;
            let part = PART as u64;
            let upstream = ScriptedStore::new(PART, Duration::from_millis(1), Duration::ZERO);
            // The last part (512 KiB) is under the 1 MiB floor.
            let bytes = rand_bytes(&DbRand::default(), 2 * PART + 512 * 1024);
            let path = upstream.put_object("obj", bytes.clone()).await;
            // (case, ranged_miss option, read, force_refresh, ranged GETs)
            let cases = [
                ("small read", true, 0..4096, false, 1),
                (
                    "exactly 1/16 of the part",
                    true,
                    part..part + part / 16,
                    false,
                    1,
                ),
                ("over 1/16 of the part", true, 0..part / 16 + 1, false, 0),
                ("whole part", true, 0..part, false, 0),
                (
                    "short last part under 1 MiB",
                    true,
                    2 * part..2 * part + 4096,
                    false,
                    0,
                ),
                ("kill switch", false, 0..4096, false, 0),
                ("force refresh", true, 0..4096, true, 0),
            ];
            for (case, ranged_miss, range, force_refresh, ranged_gets) in cases {
                let fx = Fixture::new(upstream.clone(), PART, ranged_miss, RANGED_FILL_BUDGET_KIB);
                fx.cache_head(&upstream, &path).await;
                let before = (upstream.ranged_gets(), upstream.part_gets());
                let got = fx.read(&path, range.clone(), force_refresh).await.unwrap();
                assert_eq!(got, slice(&bytes, &range), "{case}");
                fx.wait_for_fills().await;
                assert_eq!(
                    (
                        upstream.ranged_gets() - before.0,
                        upstream.part_gets() - before.1
                    ),
                    (ranged_gets, 1),
                    "{case}: (ranged, part) GETs"
                );
                let part_id = (range.start / part) as usize;
                assert!(fx.part_on_disk(&path, part_id).is_some(), "{case}");
            }
        }

        /// Effect check with S3-like latency: 20 ms to first byte plus 10 us/KiB.
        /// Run with --nocapture to see the numbers.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_ranged_miss_cuts_small_read_miss_latency() {
            const PART: usize = 4 * MIB;
            const READS: usize = 7;
            const READ: u64 = 16 * 1024;
            let bytes = rand_bytes(&DbRand::default(), READS * PART);
            let mut medians = vec![];
            for ranged_miss in [false, true] {
                let upstream =
                    ScriptedStore::new(PART, Duration::from_millis(20), Duration::from_micros(10));
                let path = upstream.put_object("obj", bytes.clone()).await;
                let fx = Fixture::new(upstream.clone(), PART, ranged_miss, RANGED_FILL_BUDGET_KIB);
                fx.cache_head(&upstream, &path).await;
                let mut latencies = vec![];
                for part_id in 0..READS {
                    let start = (part_id * PART + 12_345) as u64;
                    let range = start..start + READ;
                    let started = std::time::Instant::now();
                    let got = fx.read(&path, range.clone(), false).await.unwrap();
                    latencies.push(started.elapsed());
                    assert_eq!(got, slice(&bytes, &range));
                }
                fx.wait_for_fills().await;
                latencies.sort();
                let median = latencies[READS / 2];
                println!(
                    "ranged_miss={ranged_miss}: {READS} cold 16 KiB reads of 4 MiB parts: \
                     median {median:?}, min {:?}, max {:?}; GETs: {} ranged + {} part; \
                     ranged bytes {}",
                    latencies[0],
                    latencies[READS - 1],
                    upstream.ranged_gets(),
                    upstream.part_gets(),
                    Fixture::count(&fx.stats.bytes),
                );
                assert_eq!(
                    upstream.part_gets(),
                    READS,
                    "one part GET per miss either way"
                );
                medians.push(median);
            }
            assert!(
                medians[1] * 2 < medians[0],
                "ranged {:?} vs full part {:?}",
                medians[1],
                medians[0]
            );
        }

        /// HELION LOCAL PATCH: concurrent part fetch tests
        /// (`ObjectStoreCacheOptions::part_fetch_concurrency`).
        mod part_fetch_concurrency {
            use super::*;

            const KIB: usize = 1024;

            /// An object of `parts` whole parts of `part` bytes plus `tail` bytes
            /// behind a store that reads `concurrency` parts at once, with the
            /// head cached but no part.
            async fn fixture(
                part: usize,
                parts: usize,
                tail: usize,
                ranged_miss: bool,
                concurrency: usize,
            ) -> (Arc<ScriptedStore>, Fixture, Path, Bytes) {
                let upstream = ScriptedStore::new(part, Duration::from_millis(1), Duration::ZERO);
                let bytes = rand_bytes(&DbRand::default(), parts * part + tail);
                let path = upstream.put_object("obj", bytes.clone()).await;
                let fx = Fixture::new(upstream.clone(), part, ranged_miss, RANGED_FILL_BUDGET_KIB)
                    .part_fetch_concurrency(concurrency);
                fx.cache_head(&upstream, &path).await;
                (upstream, fx, path, bytes)
            }

            /// The items of the read's stream, errors as text.
            async fn stream_items(
                fx: &Fixture,
                path: &Path,
                range: Range<u64>,
            ) -> Vec<Result<Bytes, String>> {
                let opts = GetOptions {
                    range: Some(GetRange::Bounded(range)),
                    ..Default::default()
                };
                fx.store
                    .cached_get_opts(path, opts, false)
                    .await
                    .unwrap()
                    .into_stream()
                    .map(|item| item.map_err(|e| e.to_string()))
                    .collect()
                    .await
            }

            /// Ranges that start and end on and off part boundaries, with some
            /// parts already cached and ranged misses on and off: every
            /// concurrency returns upstream's bytes, leaves upstream's parts on
            /// disk and serves the same read again from disk.
            #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
            async fn test_part_fetch_bytes_match_upstream_for_every_concurrency() {
                const PART: usize = MIB;
                const PARTS: usize = 9;
                const TAIL: usize = 300 * KIB;
                let p = PART as u64;
                let size = (PARTS * PART + TAIL) as u64;
                let seed = rand::rng().random::<u64>();
                let rand = DbRand::new(seed);
                let mut ranges = vec![
                    0..size,                   // whole object, short last part
                    2 * p..7 * p,              // part aligned
                    p + 5..p + 9,              // inside one part
                    3 * p - 1..3 * p + 1,      // one byte either side of a boundary
                    p / 2..8 * p + p / 2,      // starts and ends mid-part
                    2 * p - 100..6 * p + 4096, // small first and last pieces (ranged)
                    4 * p + 17..size,          // mid-part to the end
                ];
                for _ in 0..5 {
                    let start = rand.rng().random_range(0..size - 1);
                    let end = rand.rng().random_range(start + 1..=size);
                    ranges.push(start..end);
                }
                for ranged_miss in [false, true] {
                    for concurrency in [0, 1, 2, 8, 64] {
                        for range in &ranges {
                            let ctx = format!(
                                "seed {seed}: {range:?}, concurrency {concurrency}, \
                                 ranged_miss {ranged_miss}"
                            );
                            let (upstream, fx, path, bytes) =
                                fixture(PART, PARTS, TAIL, ranged_miss, concurrency).await;
                            // Some of the parts are on disk already.
                            let entry = fx.store.cache_storage.entry(&path, PART);
                            for part_id in 0..=PARTS {
                                if rand.rng().random_bool(0.3) {
                                    let end = ((part_id + 1) * PART).min(bytes.len());
                                    entry
                                        .save_part(part_id, bytes.slice(part_id * PART..end))
                                        .await
                                        .unwrap();
                                }
                            }
                            // Earlier parts finish later.
                            *upstream.part_delay.lock().unwrap() =
                                |part_id| Duration::from_millis(2 * (PARTS - part_id) as u64);

                            let got = fx.read(&path, range.clone(), false).await.unwrap();
                            assert!(got == slice(&bytes, range), "{ctx}: wrong bytes");
                            fx.wait_for_fills().await;
                            // A ranged miss adds a background fill to its GET, at
                            // most for the first and the last part of the read.
                            let fills = if ranged_miss { 2 } else { 0 };
                            assert!(
                                upstream.peak_in_flight.load(Ordering::SeqCst)
                                    <= concurrency.max(1) + fills,
                                "{ctx}: too many GETs in flight"
                            );
                            let touched =
                                (range.start / p) as usize..=((range.end - 1) / p) as usize;
                            for part_id in touched {
                                let end = ((part_id + 1) * PART).min(bytes.len());
                                assert!(
                                    fx.part_on_disk(&path, part_id)
                                        == Some(bytes.slice(part_id * PART..end)),
                                    "{ctx}: part {part_id} on disk"
                                );
                            }
                            let gets = (upstream.ranged_gets(), upstream.part_gets());
                            let again = fx.read(&path, range.clone(), false).await.unwrap();
                            assert!(again == slice(&bytes, range), "{ctx}: warm read");
                            assert_eq!(
                                (upstream.ranged_gets(), upstream.part_gets()),
                                gets,
                                "{ctx}: warm read went upstream"
                            );
                        }
                    }
                }
            }

            /// Earlier parts take longer, so the GETs finish last part first; the
            /// stream still yields the parts in part order.
            #[tokio::test]
            async fn test_part_fetch_yields_parts_in_order_when_they_finish_out_of_order() {
                const PART: usize = 64 * KIB;
                const PARTS: usize = 6;
                let (upstream, fx, path, bytes) = fixture(PART, PARTS, 0, false, PARTS).await;
                *upstream.part_delay.lock().unwrap() =
                    |part_id| Duration::from_millis(40 * (PARTS - part_id) as u64);

                let items = stream_items(&fx, &path, 0..bytes.len() as u64).await;

                let expected: Vec<Result<Bytes, String>> = (0..PARTS)
                    .map(|part_id| Ok(bytes.slice(part_id * PART..(part_id + 1) * PART)))
                    .collect();
                assert!(items == expected, "parts out of order or wrong");
                assert_eq!(
                    *upstream.finished.lock().unwrap(),
                    (0..PARTS).rev().collect::<Vec<_>>(),
                    "the GETs should overlap and finish last part first"
                );
                assert_eq!(upstream.part_gets(), PARTS, "one GET per part");
            }

            /// Exactly `concurrency` part GETs are in flight at the peak (all of
            /// them when the range has fewer parts), one when it is 0 or 1.
            #[tokio::test]
            async fn test_part_fetch_never_exceeds_its_concurrency() {
                const PART: usize = 64 * KIB;
                const PARTS: usize = 24;
                for (concurrency, peak) in [(0, 1), (1, 1), (3, 3), (8, 8), (64, PARTS)] {
                    let (upstream, fx, path, bytes) =
                        fixture(PART, PARTS, 0, false, concurrency).await;
                    // Long enough for every GET in a window to have started.
                    *upstream.part_delay.lock().unwrap() = |_| Duration::from_millis(30);

                    let got = fx.read(&path, 0..bytes.len() as u64, false).await.unwrap();

                    assert!(got == bytes, "concurrency {concurrency}: wrong bytes");
                    assert_eq!(
                        upstream.peak_in_flight.load(Ordering::SeqCst),
                        peak,
                        "concurrency {concurrency}: peak GETs in flight"
                    );
                    assert_eq!(upstream.part_gets(), PARTS, "concurrency {concurrency}");
                }
            }

            /// The parts fetched at once hold at most 32 MiB: 8 parts of 4 MiB,
            /// two of 16 MiB and one of 64 MiB, which is the sequential read.
            #[tokio::test]
            async fn test_part_fetch_concurrency_is_capped_at_32_mib_in_flight() {
                // (configured, part size in MiB, parts fetched at once)
                for (configured, part_mib, effective) in [
                    (8, 4, 8),
                    (8, 16, 2),
                    (8, 64, 1),
                    (8, 1, 8),
                    (8, 8, 4),
                    (8, 32, 1),
                    (8, 128, 1),
                    (64, 4, 8),
                    (64, 1, 32),
                    (2, 4, 2),
                    (1, 4, 1),
                    (0, 4, 0),
                ] {
                    assert_eq!(
                        super::super::super::effective_part_fetch_concurrency(
                            configured,
                            part_mib * MIB
                        ),
                        effective,
                        "configured {configured}, {part_mib} MiB parts"
                    );
                }
                // A store built with the default of 8.
                for (part_mib, effective) in [(4, 8), (16, 2), (64, 1)] {
                    let upstream =
                        ScriptedStore::new(part_mib * MIB, Duration::ZERO, Duration::ZERO);
                    let fx = Fixture::new(upstream, part_mib * MIB, true, RANGED_FILL_BUDGET_KIB);
                    assert_eq!(
                        fx.store.part_fetch_concurrency, effective,
                        "{part_mib} MiB parts"
                    );
                }
            }

            /// A failing part fails the read with that part's error, whatever
            /// finished first; the stream's items are those of the sequential
            /// read, and the parts that were fetched are good cache entries.
            #[tokio::test]
            async fn test_part_fetch_error_on_one_part_fails_the_read() {
                const PART: usize = 64 * KIB;
                const PARTS: usize = 8;
                const FAILING: usize = 3;
                let mut outcomes_by_concurrency = vec![];
                for concurrency in [1, 8] {
                    let (upstream, fx, path, bytes) =
                        fixture(PART, PARTS, 0, false, concurrency).await;
                    let size = bytes.len() as u64;
                    // Parts 3.. fail, the last of them first.
                    upstream.fail_parts_from.store(FAILING, Ordering::SeqCst);
                    *upstream.part_delay.lock().unwrap() =
                        |part_id| Duration::from_millis(20 * (PARTS - part_id) as u64);

                    let error = fx.read(&path, 0..size, false).await.unwrap_err();
                    assert!(
                        error
                            .to_string()
                            .contains(&format!("injected failure of part {FAILING}")),
                        "concurrency {concurrency}: {error}"
                    );

                    let items = stream_items(&fx, &path, 0..size).await;
                    assert_eq!(items.len(), PARTS, "concurrency {concurrency}");
                    for (part_id, item) in items.iter().enumerate() {
                        match item {
                            Ok(part) => {
                                assert!(part_id < FAILING, "part {part_id} should fail");
                                assert!(*part == bytes.slice(part_id * PART..(part_id + 1) * PART));
                            }
                            Err(e) => assert!(
                                e.contains(&format!("injected failure of part {part_id}")),
                                "concurrency {concurrency}: part {part_id}: {e}"
                            ),
                        }
                    }
                    let outcomes: Vec<Result<usize, String>> = items
                        .into_iter()
                        .map(|item| item.map(|part| part.len()))
                        .collect();
                    outcomes_by_concurrency.push(outcomes);

                    // No failed part was cached; once upstream recovers the read
                    // is whole.
                    for part_id in 0..PARTS {
                        assert_eq!(
                            fx.part_on_disk(&path, part_id).is_some(),
                            part_id < FAILING,
                            "concurrency {concurrency}: part {part_id} on disk"
                        );
                    }
                    upstream.fail_parts_from.store(usize::MAX, Ordering::SeqCst);
                    let got = fx.read(&path, 0..size, false).await.unwrap();
                    assert!(got == bytes, "concurrency {concurrency}: after recovery");
                }
                assert_eq!(
                    outcomes_by_concurrency[0], outcomes_by_concurrency[1],
                    "the concurrent stream differs from the sequential one"
                );
            }
        }

        /// REVIEW (rangedmiss-review, 2026-11-01): adversarial tests for ranged
        /// cache misses. Not part of the change under review.
        mod review {
            use super::*;
            use crate::retrying_object_store::RetryingObjectStore;
            use rand::rngs::StdRng;
            use rand::SeedableRng;
            use std::sync::atomic::AtomicBool;

            const KIB: usize = 1024;

            /// InMemory upstream that counts GETs by kind, adds latency, and can
            /// hold or panic whole-part GETs (what fills and fallbacks issue).
            #[derive(Debug)]
            struct ReviewStore {
                inner: Arc<dyn ObjectStore>,
                part_size: u64,
                latency: Duration,
                ranged_gets: AtomicUsize,
                part_gets: AtomicUsize,
                other_gets: AtomicUsize,
                /// Hold every part GET (after its body was read) until the test
                /// adds a permit to `gate`.
                gate_parts: AtomicBool,
                /// Hold only the next N part GETs.
                gate_next_parts: AtomicUsize,
                gate: Semaphore,
                /// Panic in the next N part GETs (after the latency sleep).
                panic_parts: AtomicUsize,
            }

            impl ReviewStore {
                fn new(part_size: usize, latency: Duration) -> Arc<Self> {
                    Arc::new(Self {
                        inner: Arc::new(object_store::memory::InMemory::new()),
                        part_size: part_size as u64,
                        latency,
                        ranged_gets: AtomicUsize::new(0),
                        part_gets: AtomicUsize::new(0),
                        other_gets: AtomicUsize::new(0),
                        gate_parts: AtomicBool::new(false),
                        gate_next_parts: AtomicUsize::new(0),
                        gate: Semaphore::new(0),
                        panic_parts: AtomicUsize::new(0),
                    })
                }

                async fn put_bytes(&self, path: &Path, bytes: Bytes) {
                    self.inner
                        .put(path, PutPayload::from_bytes(bytes))
                        .await
                        .unwrap();
                }
            }

            fn n(counter: &AtomicUsize) -> usize {
                counter.load(Ordering::SeqCst)
            }

            impl std::fmt::Display for ReviewStore {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "ReviewStore({})", self.inner)
                }
            }

            #[async_trait::async_trait]
            impl ObjectStore for ReviewStore {
                async fn get_opts(
                    &self,
                    location: &Path,
                    options: GetOptions,
                ) -> object_store::Result<GetResult> {
                    if options.head {
                        return self.inner.get_opts(location, options).await;
                    }
                    let part = self.part_size;
                    let is_part = matches!(
                        &options.range,
                        Some(GetRange::Bounded(r)) if r.start % part == 0 && r.end - r.start == part
                    );
                    let is_ranged =
                        !is_part && matches!(&options.range, Some(GetRange::Bounded(_)));
                    if is_part {
                        self.part_gets.fetch_add(1, Ordering::SeqCst);
                    } else if is_ranged {
                        self.ranged_gets.fetch_add(1, Ordering::SeqCst);
                    } else {
                        self.other_gets.fetch_add(1, Ordering::SeqCst);
                    }
                    if !self.latency.is_zero() {
                        tokio::time::sleep(self.latency).await;
                    }
                    if is_part && take(&self.panic_parts) {
                        panic!("review: injected panic in a part GET");
                    }
                    let result = self.inner.get_opts(location, options).await?;
                    let hold = is_part
                        && (self.gate_parts.load(Ordering::SeqCst) || take(&self.gate_next_parts));
                    if !hold {
                        return Ok(result);
                    }
                    // Read the body first: the held GET already has upstream's bytes.
                    let meta = result.meta.clone();
                    let range = result.range.clone();
                    let attributes = result.attributes.clone();
                    let extensions = result.extensions.clone();
                    let body = result.bytes().await?;
                    self.gate.acquire().await.expect("gate open").forget();
                    Ok(GetResult {
                        payload: GetResultPayload::Stream(
                            stream::once(async move { Ok(body) }).boxed(),
                        ),
                        meta,
                        range,
                        attributes,
                        extensions,
                    })
                }

                async fn put_opts(
                    &self,
                    location: &Path,
                    payload: PutPayload,
                    opts: PutOptions,
                ) -> object_store::Result<PutResult> {
                    self.inner.put_opts(location, payload, opts).await
                }

                async fn put_multipart_opts(
                    &self,
                    location: &Path,
                    opts: PutMultipartOptions,
                ) -> object_store::Result<Box<dyn MultipartUpload>> {
                    self.inner.put_multipart_opts(location, opts).await
                }

                fn delete_stream(
                    &self,
                    locations: BoxStream<'static, object_store::Result<Path>>,
                ) -> BoxStream<'static, object_store::Result<Path>> {
                    self.inner.delete_stream(locations)
                }

                fn list(
                    &self,
                    prefix: Option<&Path>,
                ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
                    self.inner.list(prefix)
                }

                async fn list_with_delimiter(
                    &self,
                    prefix: Option<&Path>,
                ) -> object_store::Result<ListResult> {
                    self.inner.list_with_delimiter(prefix).await
                }

                async fn copy_opts(
                    &self,
                    from: &Path,
                    to: &Path,
                    options: CopyOptions,
                ) -> object_store::Result<()> {
                    self.inner.copy_opts(from, to, options).await
                }

                async fn rename_opts(
                    &self,
                    from: &Path,
                    to: &Path,
                    options: RenameOptions,
                ) -> object_store::Result<()> {
                    self.inner.rename_opts(from, to, options).await
                }
            }

            fn retrying(upstream: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
                Arc::new(RetryingObjectStore::new(
                    upstream,
                    Arc::new(DbRand::default()),
                    Arc::new(DefaultSystemClock::new()),
                ))
            }

            struct Rig {
                store: Arc<CachedObjectStore>,
                stats: &'static RangedMissStats,
                budget: &'static Semaphore,
                budget_kib: usize,
                root: std::path::PathBuf,
                owns_root: bool,
            }

            async fn rig(
                upstream: Arc<dyn ObjectStore>,
                root: std::path::PathBuf,
                part: usize,
                ranged: bool,
                budget_kib: usize,
                max_cache_bytes: Option<usize>,
            ) -> Rig {
                let recorder = MetricsRecorderHelper::noop();
                let cache_stats = Arc::new(CachedObjectStoreStats::new(&recorder));
                let cache_storage = Arc::new(FsCacheStorage::new(
                    root.clone(),
                    max_cache_bytes,
                    None,
                    cache_stats.clone(),
                    Arc::new(DefaultSystemClock::new()),
                    Arc::new(DbRand::default()),
                    1000,
                ));
                let mut store = CachedObjectStore::new_with_policies(
                    upstream,
                    cache_storage,
                    part,
                    cache_stats,
                    Arc::new(DefaultGetPolicy),
                    Arc::new(DefaultPutPolicy {
                        put: CachePutConfig { cache_puts: false },
                    }),
                    ranged,
                    crate::config::default_part_fetch_concurrency(),
                )
                .unwrap();
                let stats: &'static RangedMissStats = Box::leak(Box::new(RangedMissStats::new()));
                let budget: &'static Semaphore = Box::leak(Box::new(Semaphore::new(budget_kib)));
                if let Some(r) = Arc::get_mut(&mut store)
                    .expect("sole owner")
                    .ranged_miss
                    .as_mut()
                {
                    *r = RangedMiss {
                        fill_budget: budget,
                        stats,
                    };
                }
                if max_cache_bytes.is_some() {
                    store.start_evictor().await;
                }
                Rig {
                    store,
                    stats,
                    budget,
                    budget_kib,
                    root,
                    owns_root: true,
                }
            }

            impl Rig {
                async fn cache_head(&self, upstream: &dyn ObjectStore, path: &Path) {
                    let meta = upstream.head(path).await.unwrap();
                    self.store
                        .cache_storage
                        .entry(path, self.store.part_size_bytes)
                        .save_head((&meta, &object_store::Attributes::new()))
                        .await
                        .unwrap();
                }

                async fn wait_for_fills(&self) {
                    for _ in 0..4000 {
                        if self.budget.available_permits() == self.budget_kib {
                            return;
                        }
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                    panic!("background part fills did not finish");
                }

                fn part_file(&self, path: &Path, part_id: PartID) -> std::path::PathBuf {
                    FsCacheEntry::make_part_path(
                        self.root.clone(),
                        path,
                        part_id,
                        self.store.part_size_bytes,
                    )
                }

                fn count(counter: &std::sync::atomic::AtomicU64) -> u64 {
                    counter.load(Ordering::SeqCst)
                }
            }

            impl Drop for Rig {
                fn drop(&mut self) {
                    if self.owns_root {
                        let _ = std::fs::remove_dir_all(&self.root);
                    }
                }
            }

            async fn get(
                store: &CachedObjectStore,
                path: &Path,
                range: Option<GetRange>,
                force_refresh: bool,
            ) -> object_store::Result<Bytes> {
                let opts = GetOptions {
                    range,
                    ..Default::default()
                };
                store
                    .cached_get_opts(path, opts, force_refresh)
                    .await?
                    .bytes()
                    .await
            }

            fn bounded(range: Range<u64>) -> Option<GetRange> {
                Some(GetRange::Bounded(range))
            }

            /// What upstream returns for `range` (Bounded ends clamp to the size).
            fn expected(bytes: &Bytes, range: &Option<GetRange>) -> Bytes {
                let size = bytes.len() as u64;
                let r = match range {
                    None => 0..size,
                    Some(GetRange::Bounded(r)) => r.start..r.end.min(size),
                    Some(GetRange::Offset(o)) => *o..size,
                    Some(GetRange::Suffix(s)) => size.saturating_sub(*s)..size,
                };
                bytes.slice(r.start as usize..r.end as usize)
            }

            fn random_bytes(rng: &mut StdRng, len: usize) -> Bytes {
                let mut buf = vec![0; len];
                rng.fill_bytes(&mut buf);
                buf.into()
            }

            /// Every part file and head under the rig's root is exactly upstream's;
            /// no temp file is left behind.
            async fn assert_disk_matches(rig: &Rig, objects: &[(Path, Bytes)], ctx: &str) {
                let part = rig.store.part_size_bytes;
                for (path, bytes) in objects {
                    let dir = rig.root.join(path.to_string());
                    let Ok(entries) = std::fs::read_dir(&dir) else {
                        continue;
                    };
                    for e in entries {
                        let e = e.unwrap();
                        let name = e.file_name().to_string_lossy().into_owned();
                        assert!(!name.contains("_tmp"), "{ctx}: temp file left: {name}");
                        let Some(part_id) = name
                            .strip_prefix("_part")
                            .and_then(|s| s.rsplit('-').next())
                            .and_then(|s| s.parse::<usize>().ok())
                        else {
                            continue;
                        };
                        let Ok(on_disk) = std::fs::read(e.path()) else {
                            continue; // evicted meanwhile
                        };
                        let end = ((part_id + 1) * part).min(bytes.len());
                        assert!(
                            part_id * part < bytes.len(),
                            "{ctx}: {path} part {part_id} past the end"
                        );
                        assert_eq!(
                            Bytes::from(on_disk),
                            bytes.slice(part_id * part..end),
                            "{ctx}: {path} part {part_id}"
                        );
                    }
                    if let Ok(Some((meta, _))) =
                        rig.store.cache_storage.entry(path, part).read_head().await
                    {
                        assert_eq!(meta.size, bytes.len() as u64, "{ctx}: {path} head size");
                    }
                }
            }

            /// Property test: random part sizes, object sizes (aligned, short last
            /// part under and over 1 MiB), cached/uncached heads, pre-warmed parts,
            /// random ranges (small, exactly 1/16, crossing part boundaries, past
            /// the end, offsets, whole object), concurrent batches, random part and
            /// whole-entry evictions between batches, a size-bounded evictor on half
            /// the rounds, and the prod RetryingObjectStore underneath. Ranged on and
            /// off must both return upstream's bytes; every file left on disk must be
            /// upstream's. REVIEW_SEED / REVIEW_ROUNDS override.
            #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
            async fn review_property_random_layouts_match_upstream_ranged_on_and_off() {
                let seed = std::env::var("REVIEW_SEED")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| rand::rng().random::<u64>());
                let rounds: usize = std::env::var("REVIEW_ROUNDS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(16);
                let latency_ms: u64 = std::env::var("REVIEW_LATENCY_MS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                println!("review property seed {seed}, rounds {rounds}, latency {latency_ms} ms");
                let mut rng = StdRng::seed_from_u64(seed);
                let part_sizes = [512 * KIB, MIB, MIB + 64 * KIB, 2 * MIB, 3 * MIB];
                let (mut total_misses, mut total_reads) = (0u64, 0usize);
                for round in 0..rounds {
                    let part = part_sizes[rng.random_range(0..part_sizes.len())];
                    let p = part as u64;
                    let upstream = ReviewStore::new(part, Duration::from_millis(latency_ms));
                    let up: Arc<dyn ObjectStore> = retrying(upstream.clone());
                    let max_cache = rng.random_bool(0.5).then_some(3 * part);
                    let on = rig(
                        up.clone(),
                        new_test_cache_folder(),
                        part,
                        true,
                        RANGED_FILL_BUDGET_KIB,
                        max_cache,
                    )
                    .await;
                    let off = rig(
                        up.clone(),
                        new_test_cache_folder(),
                        part,
                        false,
                        RANGED_FILL_BUDGET_KIB,
                        max_cache,
                    )
                    .await;
                    let ctx = format!("seed {seed} round {round} part {part} cap {max_cache:?}");

                    let mut objects: Vec<(Path, Bytes)> = vec![];
                    for i in 0..3 {
                        let k = rng.random_range(0..4usize);
                        let tail = match rng.random_range(0..4) {
                            0 if k > 0 => 0,
                            1 if part > MIB => MIB + rng.random_range(0..=7),
                            2 => rng.random_range(1..=part / 2),
                            _ => rng.random_range(1..=part),
                        };
                        let size = (k * part + tail).max(1);
                        let path = Path::from(format!("compacted/r{round}o{i}.sst"));
                        let bytes = random_bytes(&mut rng, size);
                        upstream.put_bytes(&path, bytes.clone()).await;
                        if rng.random_bool(0.7) {
                            for r in [&on, &off] {
                                r.cache_head(&*upstream, &path).await;
                            }
                            let parts = size.div_ceil(part);
                            for part_id in 0..parts {
                                if rng.random_bool(0.3) {
                                    let end = ((part_id + 1) * part).min(size);
                                    for r in [&on, &off] {
                                        r.store
                                            .cache_storage
                                            .entry(&path, part)
                                            .save_part(part_id, bytes.slice(part_id * part..end))
                                            .await
                                            .unwrap();
                                    }
                                }
                            }
                        }
                        objects.push((path, bytes));
                    }

                    for _batch in 0..6 {
                        let plan: Vec<(usize, Option<GetRange>, bool)> = (0..8)
                            .map(|_| {
                                let obj = rng.random_range(0..objects.len());
                                let size = objects[obj].1.len() as u64;
                                let range = match rng.random_range(0..10) {
                                    0..=3 => {
                                        let len = rng.random_range(1..=p / 16 + 64);
                                        let start = rng.random_range(0..size);
                                        bounded(start..start + len)
                                    }
                                    4 if size > p => {
                                        let b = p * rng.random_range(1..=(size - 1) / p);
                                        let a = rng.random_range(1..=4096u64);
                                        let c = rng.random_range(1..=4096u64);
                                        bounded(b - a..b + c)
                                    }
                                    5 => {
                                        let start = rng.random_range(0..size);
                                        bounded(start..start + rng.random_range(1..=2 * p))
                                    }
                                    6 => Some(GetRange::Offset(rng.random_range(0..size))),
                                    7 => None,
                                    _ => {
                                        let part_id = rng.random_range(0..size.div_ceil(p));
                                        let part_len = (size - part_id * p).min(p);
                                        let len = (part_len / 16).max(1);
                                        let start =
                                            part_id * p + rng.random_range(0..=part_len - len);
                                        bounded(start..start + len)
                                    }
                                };
                                (obj, range, rng.random_bool(0.05))
                            })
                            .collect();
                        let tasks: Vec<_> = plan
                            .iter()
                            .map(|(obj, range, force)| {
                                let (on_store, off_store) = (on.store.clone(), off.store.clone());
                                let path = objects[*obj].0.clone();
                                let (range, force) = (range.clone(), *force);
                                tokio::spawn(async move {
                                    let a = get(&on_store, &path, range.clone(), force).await;
                                    let b = get(&off_store, &path, range, force).await;
                                    (a, b)
                                })
                            })
                            .collect();
                        for ((obj, range, _), task) in plan.iter().zip(tasks) {
                            let (a, b) = task.await.unwrap();
                            let want = expected(&objects[*obj].1, range);
                            assert_eq!(a.unwrap(), want, "{ctx}: ranged on, obj{obj} {range:?}");
                            assert_eq!(b.unwrap(), want, "{ctx}: ranged off, obj{obj} {range:?}");
                            total_reads += 1;
                        }

                        // Evict between batches, sometimes while fills are in flight.
                        if rng.random_bool(0.5) {
                            let (path, bytes) = &objects[rng.random_range(0..objects.len())];
                            let part_id = rng.random_range(0..bytes.len().div_ceil(part));
                            let _ = std::fs::remove_file(on.part_file(path, part_id));
                        }
                        if rng.random_bool(0.15) {
                            let (path, _) = &objects[rng.random_range(0..objects.len())];
                            on.store.cache_storage.entry(path, part).delete().await;
                        }
                    }
                    on.wait_for_fills().await;
                    assert_eq!(Rig::count(&on.stats.fills_err), 0, "{ctx}");
                    assert_eq!(
                        Rig::count(&off.stats.misses),
                        0,
                        "{ctx}: kill switch took the ranged path"
                    );
                    assert!(
                        n(&upstream.ranged_gets) as u64 >= Rig::count(&on.stats.misses),
                        "{ctx}"
                    );
                    total_misses += Rig::count(&on.stats.misses);
                    assert_disk_matches(&on, &objects, &format!("{ctx} on")).await;
                    assert_disk_matches(&off, &objects, &format!("{ctx} off")).await;
                }
                println!("review property: {total_reads} reads, {total_misses} ranged misses");
                assert!(total_misses > 0, "seed {seed}: the ranged path never ran");
            }

            /// 256 concurrent reads hammer one cold 4 MiB part (plus reads across
            /// both of its boundaries and some large reads that use the part flight
            /// directly). Every read is right, the parts on disk are right, the
            /// budget is returned and the duplicate part GETs from the
            /// `contains()`-then-spawn window stay small.
            #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
            async fn review_concurrent_misses_on_one_part() {
                const PART: usize = 4 * MIB;
                let p = PART as u64;
                let mut rng = StdRng::seed_from_u64(7);
                let upstream = ReviewStore::new(PART, Duration::from_millis(5));
                let bytes = random_bytes(&mut rng, 2 * PART + 2 * MIB);
                let path = Path::from("compacted/hot.sst");
                upstream.put_bytes(&path, bytes.clone()).await;
                let r = rig(
                    retrying(upstream.clone()),
                    new_test_cache_folder(),
                    PART,
                    true,
                    RANGED_FILL_BUDGET_KIB,
                    None,
                )
                .await;
                r.cache_head(&*upstream, &path).await;

                let ranges: Vec<Range<u64>> = (0..256)
                    .map(|i| match i % 8 {
                        0 => p - rng.random_range(1..4096)..p + rng.random_range(1..4096),
                        1 => 2 * p - rng.random_range(1..4096)..2 * p + rng.random_range(1..4096),
                        2 => {
                            let start = p + rng.random_range(0..p / 2);
                            start..start + p / 8
                        }
                        _ => {
                            let start = p + rng.random_range(0..p - 64 * 1024);
                            start..start + rng.random_range(1..64 * 1024)
                        }
                    })
                    .collect();
                let tasks: Vec<_> = ranges
                    .iter()
                    .map(|range| {
                        let (store, path, range) = (r.store.clone(), path.clone(), range.clone());
                        tokio::spawn(async move { get(&store, &path, bounded(range), false).await })
                    })
                    .collect();
                for (range, task) in ranges.iter().zip(tasks) {
                    let got = task.await.unwrap().unwrap();
                    assert_eq!(got, expected(&bytes, &bounded(range.clone())), "{range:?}");
                }
                r.wait_for_fills().await;
                assert_disk_matches(&r, &[(path.clone(), bytes.clone())], "hot part").await;
                for part_id in 0..3 {
                    assert!(
                        r.part_file(&path, part_id).exists(),
                        "part {part_id} not cached"
                    );
                }
                let (ranged, parts) = (n(&upstream.ranged_gets), n(&upstream.part_gets));
                println!(
                    "review concurrent: {ranged} ranged GETs, {parts} part GETs for 3 parts; \
                     fills ok {} skipped {} err {}",
                    Rig::count(&r.stats.fills_ok),
                    Rig::count(&r.stats.fills_skipped),
                    Rig::count(&r.stats.fills_err)
                );
                assert_eq!(Rig::count(&r.stats.fills_err), 0);
                assert!(parts <= 6, "{parts} part GETs for 3 parts");

                // Warm: no more upstream traffic.
                let before = (n(&upstream.ranged_gets), n(&upstream.part_gets));
                for range in ranges.iter().take(32) {
                    let got = get(&r.store, &path, bounded(range.clone()), false)
                        .await
                        .unwrap();
                    assert_eq!(got, expected(&bytes, &bounded(range.clone())));
                }
                assert_eq!((n(&upstream.ranged_gets), n(&upstream.part_gets)), before);
            }

            /// A fill that panics mid-GET returns its budget and clears its flight,
            /// and a reader that joined the fill takes over instead of hanging.
            #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
            async fn review_panicking_fill_releases_budget_and_flight() {
                const PART: usize = 2 * MIB;
                let mut rng = StdRng::seed_from_u64(11);
                let upstream = ReviewStore::new(PART, Duration::from_millis(40));
                let bytes = random_bytes(&mut rng, 2 * PART);
                let path = Path::from("compacted/panics.sst");
                upstream.put_bytes(&path, bytes.clone()).await;
                let r = rig(
                    upstream.clone(),
                    new_test_cache_folder(),
                    PART,
                    true,
                    RANGED_FILL_BUDGET_KIB,
                    None,
                )
                .await;
                r.cache_head(&*upstream, &path).await;
                upstream.panic_parts.store(1, Ordering::SeqCst);

                let first = bounded(100..4196);
                let second = bounded(300_000..310_000);
                let store = r.store.clone();
                let path2 = path.clone();
                let joined = tokio::spawn(async move {
                    // Arrive while the fill is in flight: contains() is true.
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    assert!(store.part_flights.contains(&(path2.clone(), 0)));
                    get(&store, &path2, bounded(300_000..310_000), false).await
                });
                assert_eq!(
                    get(&r.store, &path, first.clone(), false).await.unwrap(),
                    expected(&bytes, &first)
                );
                let got = tokio::time::timeout(Duration::from_secs(10), joined)
                    .await
                    .expect("the joined reader hung on a panicked fill")
                    .unwrap()
                    .unwrap();
                assert_eq!(got, expected(&bytes, &second));
                r.wait_for_fills().await;
                assert!(!r.store.part_flights.contains(&(path.clone(), 0)));
                assert_eq!(
                    n(&upstream.part_gets),
                    2,
                    "panicked fill + the joined reader's own fetch"
                );
                assert_eq!(
                    std::fs::read(r.part_file(&path, 0)).unwrap(),
                    bytes.slice(0..PART)
                );
                println!(
                    "review panic: fills ok {} err {} (a panicked fill is not counted)",
                    Rig::count(&r.stats.fills_ok),
                    Rig::count(&r.stats.fills_err)
                );
            }

            /// Fills that already read upstream are held while the entry is deleted
            /// and the whole cache dir is removed again and again (Helion's volume
            /// evictor / drop purge), with a size-bounded evictor running too. Only
            /// whole, correct parts may remain and later reads stay right.
            #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
            async fn review_eviction_and_dir_removal_during_fills() {
                const PART: usize = 2 * MIB;
                let mut rng = StdRng::seed_from_u64(13);
                let upstream = ReviewStore::new(PART, Duration::ZERO);
                let bytes = random_bytes(&mut rng, 6 * PART + MIB + 3);
                let path = Path::from("compacted/evicted.sst");
                upstream.put_bytes(&path, bytes.clone()).await;
                let r = rig(
                    retrying(upstream.clone()),
                    new_test_cache_folder(),
                    PART,
                    true,
                    RANGED_FILL_BUDGET_KIB,
                    Some(2 * PART + 64 * KIB),
                )
                .await;
                r.cache_head(&*upstream, &path).await;
                upstream.gate_parts.store(true, Ordering::SeqCst);

                let p = PART as u64;
                for part_id in 0..7u64 {
                    let range = bounded(part_id * p + 77..part_id * p + 4173);
                    assert_eq!(
                        get(&r.store, &path, range.clone(), false).await.unwrap(),
                        expected(&bytes, &range)
                    );
                }
                while n(&upstream.part_gets) < 7 {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
                r.store.cache_storage.entry(&path, PART).delete().await;
                let _ = std::fs::remove_dir_all(&r.root);

                // Release the fills one by one while another thread keeps removing the dir.
                let stop = Arc::new(AtomicBool::new(false));
                let remover = {
                    let (root, stop) = (r.root.clone(), stop.clone());
                    std::thread::spawn(move || {
                        let mut removals = 0;
                        while !stop.load(Ordering::SeqCst) {
                            if std::fs::remove_dir_all(&root).is_ok() {
                                removals += 1;
                            }
                            std::thread::sleep(Duration::from_micros(200));
                        }
                        removals
                    })
                };
                for _ in 0..7 {
                    upstream.gate.add_permits(1);
                    tokio::time::sleep(Duration::from_millis(3)).await;
                }
                upstream.gate_parts.store(false, Ordering::SeqCst);
                r.wait_for_fills().await;
                stop.store(true, Ordering::SeqCst);
                let removals = remover.join().unwrap();
                assert_eq!(Rig::count(&r.stats.fills_ok), 7);
                assert_disk_matches(&r, &[(path.clone(), bytes.clone())], "after removals").await;

                for _ in 0..64 {
                    let start = rng.random_range(0..bytes.len() as u64);
                    let range = bounded(start..start + rng.random_range(1..=200_000));
                    assert_eq!(
                        get(&r.store, &path, range.clone(), false).await.unwrap(),
                        expected(&bytes, &range)
                    );
                }
                r.wait_for_fills().await;
                assert_disk_matches(&r, &[(path.clone(), bytes.clone())], "after reads").await;
                println!(
                    "review eviction: {removals} dir removals raced 7 fills; root exists after: {}",
                    r.root.exists()
                );
            }

            /// A fill from a dropped collection's last handle lands after the drop
            /// purge removed the cache dir and the collection was re-created under
            /// the same root. The late fill does resurrect its SST's files, but only
            /// under the dropped SST's own (ULID) path: the new incarnation's SSTs
            /// read correctly, and WAL reads (ids restart on re-create) never touch
            /// the cache. Only a reused compacted-SST path could read stale bytes.
            #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
            async fn review_late_fill_after_drop_and_recreate() {
                const PART: usize = 2 * MIB;
                let mut rng = StdRng::seed_from_u64(17);
                let upstream = ReviewStore::new(PART, Duration::ZERO);
                let root = new_test_cache_folder();
                let old_sst = Path::from("prod/c/compacted/01JA0000000000000000000OLD.sst");
                let old_bytes = random_bytes(&mut rng, 2 * PART);
                upstream.put_bytes(&old_sst, old_bytes.clone()).await;
                let wal = Path::from("prod/c/wal/00000000000000000001.sst");
                upstream
                    .put_bytes(&wal, random_bytes(&mut rng, 3 * MIB))
                    .await;

                let mut a = rig(
                    upstream.clone(),
                    root.clone(),
                    PART,
                    true,
                    RANGED_FILL_BUDGET_KIB,
                    Some(64 * MIB),
                )
                .await;
                a.owns_root = false;
                a.cache_head(&*upstream, &old_sst).await;
                // WAL reads bypass the cache even with ranged misses on.
                let wal_tag = ObjectStoreCallTag::new(TableStoreKind::Reader, SstType::Wal);
                let opts = GetOptions {
                    range: bounded(100..4196),
                    extensions: wal_tag.into(),
                    ..Default::default()
                };
                a.store
                    .get_opts(&wal, opts.clone())
                    .await
                    .unwrap()
                    .bytes()
                    .await
                    .unwrap();
                assert!(!root.join(wal.to_string()).exists(), "WAL read was cached");

                upstream.gate_parts.store(true, Ordering::SeqCst);
                let range = bounded(100..4196);
                assert_eq!(
                    get(&a.store, &old_sst, range.clone(), false).await.unwrap(),
                    expected(&old_bytes, &range)
                );
                while n(&upstream.part_gets) < 1 {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }

                // Drop: purge S3, remove the cache subtree; the handle is gone but
                // its fill (holding its own store clone) is still in flight.
                let a_budget = (a.budget, a.budget_kib);
                drop(a);
                upstream.inner.delete(&old_sst).await.unwrap();
                upstream.inner.delete(&wal).await.unwrap();
                std::fs::remove_dir_all(&root).unwrap();

                // Re-create under the same root: new SST ids, WAL id 1 again.
                let b = rig(
                    upstream.clone(),
                    root.clone(),
                    PART,
                    true,
                    RANGED_FILL_BUDGET_KIB,
                    Some(64 * MIB),
                )
                .await;
                let new_sst = Path::from("prod/c/compacted/01JB0000000000000000000NEW.sst");
                let new_bytes = random_bytes(&mut rng, 3 * PART + 5);
                upstream.put_bytes(&new_sst, new_bytes.clone()).await;
                let new_wal = random_bytes(&mut rng, 3 * MIB);
                upstream.put_bytes(&wal, new_wal.clone()).await;
                b.cache_head(&*upstream, &new_sst).await;

                // The late fill lands now.
                upstream.gate_parts.store(false, Ordering::SeqCst);
                upstream.gate.add_permits(1);
                for _ in 0..4000 {
                    if a_budget.0.available_permits() == a_budget.1 {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
                assert_eq!(a_budget.0.available_permits(), a_budget.1);
                let resurrected = FsCacheEntry::make_part_path(root.clone(), &old_sst, 0, PART);
                assert!(
                    resurrected.exists(),
                    "expected the late fill to re-create the dropped SST's part"
                );

                for _ in 0..32 {
                    let start = rng.random_range(0..new_bytes.len() as u64);
                    let range = bounded(start..start + rng.random_range(1..=100_000));
                    assert_eq!(
                        get(&b.store, &new_sst, range.clone(), false).await.unwrap(),
                        expected(&new_bytes, &range)
                    );
                }
                let got = b
                    .store
                    .get_opts(&wal, opts)
                    .await
                    .unwrap()
                    .bytes()
                    .await
                    .unwrap();
                assert_eq!(
                    got,
                    new_wal.slice(100..4196),
                    "WAL id reuse served the dropped incarnation"
                );
                b.wait_for_fills().await;
                assert_disk_matches(
                    &b,
                    &[
                        (new_sst.clone(), new_bytes.clone()),
                        (old_sst.clone(), old_bytes.clone()),
                    ],
                    "recreated",
                )
                .await;
            }

            /// Callers cancelled mid ranged GET still get their fills; a spent
            /// budget skips fills; the accounting is exact either way.
            #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
            async fn review_cancelled_callers_and_spent_budget() {
                const PART: usize = 2 * MIB;
                let mut rng = StdRng::seed_from_u64(19);
                let upstream = ReviewStore::new(PART, Duration::from_millis(40));
                let bytes = random_bytes(&mut rng, 12 * PART);
                let path = Path::from("compacted/cancel.sst");
                upstream.put_bytes(&path, bytes.clone()).await;
                let budget_kib = 3 * PART / 1024;
                let r = rig(
                    retrying(upstream.clone()),
                    new_test_cache_folder(),
                    PART,
                    true,
                    budget_kib,
                    None,
                )
                .await;
                r.cache_head(&*upstream, &path).await;
                let p = PART as u64;

                let handles: Vec<_> = (0..12u64)
                    .map(|part_id| {
                        let (store, path) = (r.store.clone(), path.clone());
                        tokio::spawn(async move {
                            get(
                                &store,
                                &path,
                                bounded(part_id * p + 5..part_id * p + 4101),
                                false,
                            )
                            .await
                        })
                    })
                    .collect();
                tokio::time::sleep(Duration::from_millis(25)).await;
                for h in &handles {
                    h.abort();
                }
                for h in handles {
                    assert!(h.await.unwrap_err().is_cancelled());
                }
                r.wait_for_fills().await;
                assert_eq!(Rig::count(&r.stats.fills_ok), 3);
                assert_eq!(Rig::count(&r.stats.fills_skipped), 9);
                assert_eq!(r.budget.available_permits(), budget_kib);
                let cached = (0..12)
                    .filter(|id| r.part_file(&path, *id).exists())
                    .count();
                assert_eq!(cached, 3, "fills should outlive their cancelled callers");
                assert_disk_matches(&r, &[(path.clone(), bytes.clone())], "after cancel").await;

                for part_id in 0..12u64 {
                    let range = bounded(part_id * p + 999..part_id * p + 9999);
                    assert_eq!(
                        get(&r.store, &path, range.clone(), false).await.unwrap(),
                        expected(&bytes, &range)
                    );
                }
                r.wait_for_fills().await;
                assert_eq!(r.budget.available_permits(), budget_kib);
                assert_disk_matches(&r, &[(path.clone(), bytes.clone())], "after rereads").await;
            }

            /// Behaviour change: a hung part GET used to be dropped with its
            /// (cancelled) caller, and the next reader started a fresh GET. A
            /// detached fill is never cancelled, so later small misses of that part
            /// wait on the hung fill (until the S3 client timeout in prod).
            #[rstest]
            #[case::ranged_off(false)]
            #[case::ranged_on(true)]
            #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
            async fn review_hung_part_get_after_cancelled_caller(#[case] ranged: bool) {
                const PART: usize = 2 * MIB;
                let mut rng = StdRng::seed_from_u64(23);
                let upstream = ReviewStore::new(PART, Duration::from_millis(5));
                let bytes = random_bytes(&mut rng, 2 * PART);
                let path = Path::from("compacted/hung.sst");
                upstream.put_bytes(&path, bytes.clone()).await;
                let r = rig(
                    upstream.clone(),
                    new_test_cache_folder(),
                    PART,
                    ranged,
                    RANGED_FILL_BUDGET_KIB,
                    None,
                )
                .await;
                r.cache_head(&*upstream, &path).await;
                upstream.gate_next_parts.store(1, Ordering::SeqCst);

                let first = {
                    let (store, path) = (r.store.clone(), path.clone());
                    tokio::spawn(async move { get(&store, &path, bounded(100..4196), false).await })
                };
                tokio::time::sleep(Duration::from_millis(50)).await;
                first.abort(); // the HTTP request behind it was cancelled
                let _ = first.await;

                let second = bounded(500_000..504_096);
                let result = tokio::time::timeout(
                    Duration::from_millis(500),
                    get(&r.store, &path, second.clone(), false),
                )
                .await;
                println!(
                    "review hung (ranged={ranged}): second reader {}",
                    if result.is_ok() {
                        "returned"
                    } else {
                        "timed out"
                    }
                );
                if ranged {
                    assert!(
                        result.is_err(),
                        "expected the second miss to wait on the hung fill"
                    );
                } else {
                    assert_eq!(
                        result.expect("fresh fetch").unwrap(),
                        expected(&bytes, &second)
                    );
                }
                upstream.gate.add_permits(1);
                r.wait_for_fills().await;
            }

            /// Unreachable from SlateDB (it only issues non-empty Bounded ranges),
            /// but CachedObjectStore::get_opts accepts Suffix(0): with a cached head
            /// and a long unaligned last part the empty range takes the ranged
            /// path, and RetryingObjectStore retries the invalid GET forever.
            #[rstest]
            #[case::ranged_off(false)]
            #[case::ranged_on(true)]
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn review_empty_suffix_read(#[case] ranged: bool) {
                const PART: usize = 2 * MIB;
                let mut rng = StdRng::seed_from_u64(29);
                let upstream = ReviewStore::new(PART, Duration::ZERO);
                let bytes = random_bytes(&mut rng, PART + MIB + 5);
                let path = Path::from("compacted/suffix.sst");
                upstream.put_bytes(&path, bytes.clone()).await;
                let r = rig(
                    retrying(upstream.clone()),
                    new_test_cache_folder(),
                    PART,
                    ranged,
                    RANGED_FILL_BUDGET_KIB,
                    None,
                )
                .await;
                r.cache_head(&*upstream, &path).await;
                let result = tokio::time::timeout(
                    Duration::from_secs(3),
                    get(&r.store, &path, Some(GetRange::Suffix(0)), false),
                )
                .await;
                println!(
                    "review empty suffix (ranged={ranged}): {:?}",
                    result.as_ref().map(|r| r.as_ref().map(|b| b.len()))
                );
                assert_eq!(result.expect("empty read hung").unwrap(), Bytes::new());
            }
        }
    }
}
