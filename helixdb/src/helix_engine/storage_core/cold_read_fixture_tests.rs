//! Cold-read fixture for reader replicas: a gitea-shaped collection (point
//! rows, per-segment HNSW neighbour blocks, ordinals, raw vector rows) on a
//! local object store with injected S3 latency, read through reader-replica
//! options and the Helion disk cache, under several shared meta-cache shapes.
//!
//! One "fresh query" is what a fresh mini top-200 costs a reader: 200
//! sequential neighbour-block gets, 200 sequential 14-key ordinal multi-gets
//! and 200 sequential point-row gets (`/points/search` hydration).
//!
//! Ignored (~1 GiB in memory, <=384 MiB disk cache); run by hand:
//! `HELION_COLD_FIXTURE_DIR=/tmp/helion-cold-fixture cargo test -p helixdb --lib \
//!  cold_read_fixture -- --ignored --nocapture --test-threads=1`

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::stream::BoxStream;
use object_store::memory::InMemory;
use object_store::path::Path as ObjPath;
use object_store::ObjectStore;
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use slatedb::db_cache::DbCache;
use slatedb::{scope_query_metrics, DbReader, QueryMetricsObserver};

use super::backend::{Namespace, SegmentDb, StorageBackend};
use super::backend_lsm::{
    build_db_cache, prefix_filter_policies_from_env, prefixed, reader_options_from_env,
    shared_merge_operator, LsmBackend,
};

const MIB: usize = 1024 * 1024;
const POINTS: u64 = 48_000;
const SPACES: [(&str, usize); 3] = [("mini", 64), ("dense", 768), ("lex", 1024)];
const HOPS: usize = 200;

/// S3 stand-in: every GET pays a first-byte delay plus transfer at a per-stream
/// rate, and is counted.
#[derive(Debug)]
struct LatencyStore {
    inner: Arc<dyn ObjectStore>,
    first_byte: Duration,
    bytes_per_sec: f64,
    gets: AtomicU64,
    bytes: AtomicU64,
}

impl std::fmt::Display for LatencyStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LatencyStore({})", self.inner)
    }
}

type Boxed<'a, T> = std::pin::Pin<Box<dyn Future<Output = object_store::Result<T>> + Send + 'a>>;
use std::future::Future;

impl ObjectStore for LatencyStore {
    fn put_opts<'a, 'b, 'c>(
        &'a self,
        location: &'b ObjPath,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> Boxed<'c, object_store::PutResult>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        self.inner.put_opts(location, payload, opts)
    }

    fn put_multipart_opts<'a, 'b, 'c>(
        &'a self,
        location: &'b ObjPath,
        opts: object_store::PutMultipartOptions,
    ) -> Boxed<'c, Box<dyn object_store::MultipartUpload>>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        self.inner.put_multipart_opts(location, opts)
    }

    fn get_opts<'a, 'b, 'c>(
        &'a self,
        location: &'b ObjPath,
        options: object_store::GetOptions,
    ) -> Boxed<'c, object_store::GetResult>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        Box::pin(async move {
            let head = options.head;
            let result = self.inner.get_opts(location, options).await?;
            let len = if head {
                0
            } else {
                result.range.end - result.range.start
            };
            self.gets.fetch_add(1, Ordering::Relaxed);
            self.bytes.fetch_add(len, Ordering::Relaxed);
            let transfer = Duration::from_secs_f64(len as f64 / self.bytes_per_sec);
            tokio::time::sleep(self.first_byte + transfer).await;
            Ok(result)
        })
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<ObjPath>>,
    ) -> BoxStream<'static, object_store::Result<ObjPath>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&ObjPath>,
    ) -> BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_delimiter<'a, 'b, 'c>(
        &'a self,
        prefix: Option<&'b ObjPath>,
    ) -> Boxed<'c, object_store::ListResult>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        self.inner.list_with_delimiter(prefix)
    }

    fn copy_opts<'a, 'b, 'c, 'd>(
        &'a self,
        from: &'b ObjPath,
        to: &'c ObjPath,
        options: object_store::CopyOptions,
    ) -> Boxed<'d, ()>
    where
        'a: 'd,
        'b: 'd,
        'c: 'd,
        Self: 'd,
    {
        self.inner.copy_opts(from, to, options)
    }
}

fn seg(space: &str) -> String {
    format!("{space}__seg_000006")
}

fn point_id(i: u64) -> [u8; 16] {
    // Spread ids across the key space like UUID-derived point ids.
    (u128::from(i).wrapping_mul(0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835)).to_be_bytes()
}

fn neighbor_key(i: u64) -> Vec<u8> {
    let mut key = point_id(i).to_vec();
    key.push(0);
    key
}

fn set_env(pairs: &[(&str, &str)]) {
    for (key, value) in pairs {
        std::env::set_var(key, value);
    }
}

/// Writes the collection in shuffled row order so every L0 spans the whole
/// key range and a point read consults every L0's filter, like gitea's mix of
/// L0s and sorted runs (~7 SSTs per point read).
fn write_fixture(path: &str, store: Arc<dyn ObjectStore>) {
    set_env(&[
        ("HELIX_LSM_COMPRESSION_CODEC", "lz4"),
        ("HELIX_LSM_PREFIX_FILTERS", "1"),
        ("HELIX_LSM_MIN_FILTER_KEYS", "64"),
        ("HELIX_LSM_L0_SST_SIZE_BYTES", &(32 * MIB).to_string()),
        ("HELIX_LSM_MAX_UNFLUSHED_BYTES", &(512 * MIB).to_string()),
        ("HELIX_LSM_L0_MAX_SSTS", "64"),
        ("HELIX_LSM_L0_MAX_SSTS_PER_KEY", "64"),
    ]);
    std::env::remove_var("HELIX_LSM_CACHE_DIR");
    let be = LsmBackend::open_with_store(path, store).expect("open fixture writer");
    let mut rows: Vec<(u8, u64)> = (0..POINTS)
        .flat_map(|i| (0..=(2 * SPACES.len() as u8)).map(move |kind| (kind, i)))
        .collect();
    let mut rng = StdRng::seed_from_u64(7);
    for i in (1..rows.len()).rev() {
        rows.swap(i, rng.random_range(0..=i));
    }
    let mut value = vec![0u8; 4096];
    for chunk in rows.chunks(20_000) {
        let mut w = be.begin_write().unwrap();
        for &(kind, i) in chunk {
            let id = point_id(i);
            rng.fill_bytes(&mut value);
            if kind == 0 {
                be.put(&mut w, Namespace::Nodes, &id, &value[..2400])
                    .unwrap();
                continue;
            }
            let (space, dim) = SPACES[usize::from((kind - 1) / 2)];
            let physical = seg(space);
            if kind % 2 == 1 {
                let ns = Namespace::Segment {
                    physical_name: &physical,
                    db: SegmentDb::HnswNeighbors,
                };
                be.put(&mut w, ns, &neighbor_key(i), &value[..520]).unwrap();
                let ns = Namespace::Segment {
                    physical_name: &physical,
                    db: SegmentDb::Ordinals,
                };
                be.put(&mut w, ns, &id, &i.to_le_bytes()).unwrap();
            } else {
                let ns = Namespace::Segment {
                    physical_name: &physical,
                    db: SegmentDb::VectorData,
                };
                be.put(&mut w, ns, &id, &value[..dim * 4]).unwrap();
            }
        }
        be.commit(w).unwrap();
    }
    be.close().expect("flush fixture memtables to L0");
}

#[derive(Default, Debug, Clone, Copy)]
struct PhaseStats {
    ms: f64,
    mem_misses: u64,
    disk_misses: u64,
}

async fn timed<F: Future<Output = ()>>(f: F) -> PhaseStats {
    let observer = QueryMetricsObserver::default();
    let started = Instant::now();
    scope_query_metrics(observer.clone(), f).await;
    let snap = observer.snapshot();
    PhaseStats {
        ms: started.elapsed().as_secs_f64() * 1000.0,
        mem_misses: snap.block_cache.misses,
        disk_misses: snap.object_cache.misses,
    }
}

/// One fresh query: new random ids for every phase.
async fn fresh_query(reader: &DbReader, seed: u64) -> [PhaseStats; 3] {
    let mut rng = StdRng::seed_from_u64(seed);
    let mini = seg("mini");
    let neighbors = Namespace::Segment {
        physical_name: &mini,
        db: SegmentDb::HnswNeighbors,
    };
    let ordinals = Namespace::Segment {
        physical_name: &mini,
        db: SegmentDb::Ordinals,
    };
    let hnsw = timed(async {
        for _ in 0..HOPS {
            let key = prefixed(neighbors, &neighbor_key(rng.random_range(0..POINTS)));
            assert!(reader.get(&key).await.unwrap().is_some());
        }
    })
    .await;
    let ords = timed(async {
        for _ in 0..HOPS {
            let keys: Vec<Vec<u8>> = (0..14)
                .map(|_| prefixed(ordinals, &point_id(rng.random_range(0..POINTS))))
                .collect();
            let rows = reader.multi_get(&keys).await.unwrap();
            assert!(rows.iter().all(Option::is_some));
        }
    })
    .await;
    let nodes = timed(async {
        for _ in 0..HOPS {
            let key = prefixed(Namespace::Nodes, &point_id(rng.random_range(0..POINTS)));
            assert!(reader.get(&key).await.unwrap().is_some());
        }
    })
    .await;
    [hnsw, ords, nodes]
}

async fn open_reader(path: &str, store: Arc<dyn ObjectStore>, cache: Arc<dyn DbCache>) -> DbReader {
    let mut builder = DbReader::builder(path, store)
        .with_options(reader_options_from_env(path))
        .with_db_cache(cache)
        .with_merge_operator(shared_merge_operator());
    if let Some(policies) = prefix_filter_policies_from_env() {
        builder = builder.with_filter_policies(policies);
    }
    builder.build().await.expect("open fixture reader")
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[test]
#[serial_test::serial]
#[ignore = "~1 GiB in-memory fixture; run by hand, see module docs"]
fn cold_read_fixture() {
    let root = std::path::PathBuf::from(
        std::env::var("HELION_COLD_FIXTURE_DIR").unwrap_or("/tmp/helion-cold-fixture".into()),
    );
    // In-memory "bucket" keeps the ~1 GiB of SSTs + WAL off the laptop disk.
    let local: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "prod/admin_cold_fixture";
    let started = Instant::now();
    write_fixture(path, Arc::clone(&local));
    println!("fixture written in {:.0}s", started.elapsed().as_secs_f64());
    // Prod reader cache shape: 12 GiB / 32 = 384 MiB per collection, 4 MiB parts.
    let cache_dir = root.join("cache");
    set_env(&[
        ("HELIX_LSM_CACHE_DIR", cache_dir.to_str().unwrap()),
        ("HELIX_LSM_CACHE_MAX_BYTES", &(12 * 1024 * MIB).to_string()),
        ("HELIX_LSM_CACHE_COLLECTION_DIVISOR", "32"),
        ("HELIX_LSM_CACHE_MEMORY_MAX_PCT", "0"),
        ("HELIX_LSM_CACHE_PART_BYTES", &(4 * MIB).to_string()),
        ("HELIX_LSM_CACHE_SCAN_SECS", "0"),
        ("HELIX_LSM_CACHE_MAX_OPEN_FILES", "16"),
    ]);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    // (label, meta bytes, meta shards). Gitea on prod: ~28 SSTs, 5-8 MiB decoded
    // indexes, 128 MiB / 16 = 8 MiB shards. The fixture's ~20 32 MiB SSTs carry
    // ~0.6 MiB indexes, so 12 MiB / 16 shards is the same entry-to-shard ratio
    // and 96 MiB / 16 shards scales the 1 GiB fix.
    let configs = [
        ("prod-shape 12MiB/16 shards", 12 * MIB, 16),
        ("same memory, derived 12MiB/1 shard", 12 * MIB, 1),
        ("fix-shape 96MiB/16 shards", 96 * MIB, 16),
    ];
    for (label, meta_bytes, meta_shards) in configs {
        let _ = std::fs::remove_dir_all(&cache_dir);
        let s3 = latency_store(&local);
        rt.block_on(async {
            let cache = build_db_cache(512 * MIB, meta_bytes, meta_shards);
            let reader = open_reader(path, s3.clone(), cache).await;
            let mut warm = [Vec::new(), Vec::new(), Vec::new()];
            let mut misses = [0u64; 3];
            let mut disk_misses = [0u64; 3];
            for q in 0..12u64 {
                let gets_before = s3.gets.load(Ordering::Relaxed);
                let bytes_before = s3.bytes.load(Ordering::Relaxed);
                let phases = fresh_query(&reader, 1_000 + q).await;
                let total: f64 = phases.iter().map(|p| p.ms).sum();
                if q == 0 {
                    println!(
                        "[{label}] cold q0: {total:.0} ms, s3 gets {} ({:.0} MiB), phases {:?}",
                        s3.gets.load(Ordering::Relaxed) - gets_before,
                        (s3.bytes.load(Ordering::Relaxed) - bytes_before) as f64 / MIB as f64,
                        phases.map(|p| p.ms.round()),
                    );
                } else if q >= 2 {
                    for (i, p) in phases.iter().enumerate() {
                        warm[i].push(p.ms);
                        misses[i] += p.mem_misses;
                        disk_misses[i] += p.disk_misses;
                    }
                }
            }
            let n = warm[0].len() as f64;
            let med: Vec<f64> = warm.iter().map(|w| median(w.clone()).round()).collect();
            println!(
                "[{label}] fresh warm-disk median ms hnsw/ords/nodes {med:?} = {:.0}; \
                 mem-cache misses per query {:?}; disk-cache misses per query {:?}; s3 gets total {}",
                med.iter().sum::<f64>(),
                misses.map(|m| (m as f64 / n).round()),
                disk_misses.map(|m| (m as f64 / n).round()),
                s3.gets.load(Ordering::Relaxed),
            );
            reader.close().await.unwrap();
        });
    }
    // Reader rebuild (poll-tier promote/demote, refresh): the next fresh query on
    // the new handle, per-handle cache scopes vs the shared compacted scope.
    for share in [false, true] {
        slatedb::db_cache::set_share_compacted_entries(share);
        let s3 = latency_store(&local);
        rt.block_on(async {
            let cache = build_db_cache(512 * MIB, 96 * MIB, 16);
            let old = open_reader(path, s3.clone(), Arc::clone(&cache)).await;
            for q in 0..3u64 {
                fresh_query(&old, 5_000 + q).await;
            }
            let rebuilt = open_reader(path, s3.clone(), cache).await;
            let phases = fresh_query(&rebuilt, 6_000).await;
            println!(
                "[rebuild share_compacted={share}] first fresh query on the rebuilt reader: {:.0} ms, \
                 mem-cache misses {}",
                phases.iter().map(|p| p.ms).sum::<f64>(),
                phases.iter().map(|p| p.mem_misses).sum::<u64>(),
            );
            old.close().await.unwrap();
            rebuilt.close().await.unwrap();
        });
    }
    slatedb::db_cache::set_share_compacted_entries(false);
}

fn latency_store(inner: &Arc<dyn ObjectStore>) -> Arc<LatencyStore> {
    Arc::new(LatencyStore {
        inner: Arc::clone(inner),
        first_byte: Duration::from_millis(25),
        bytes_per_sec: 150.0 * MIB as f64,
        gets: AtomicU64::new(0),
        bytes: AtomicU64::new(0),
    })
}
