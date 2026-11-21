//! Adversarial review tests for the shared compacted-SST cache scope
//! (`HELIX_LSM_SHARE_SST_CACHE`, vendored `db_cache::set_share_compacted_entries`)
//! and the derived meta-cache shard split (`HELIX_LSM_META_CACHE_SHARDS`).
//!
//! Every test drives real `Db`/`DbReader` handles over an in-memory object
//! store through one shared `DbCache`, the way a reader pod shares
//! `shared_db_cache()` across collections and reopened handles.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use object_store::memory::InMemory;
use object_store::path::Path as ObjPath;
use object_store::{ObjectStore, ObjectStoreExt};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use slatedb::config::{
    DbReaderOptions, FlushOptions, FlushType, GarbageCollectorDirectoryOptions,
    GarbageCollectorOptions, PutOptions, Settings, WriteOptions,
};
use slatedb::db_cache::foyer::{FoyerCache, FoyerCacheOptions};
use slatedb::db_cache::{CacheUsageSnapshot, DbCache, SplitCache};
use slatedb::{scope_query_metrics, Db, DbReader, QueryMetricsObserver};

use super::backend_lsm::build_db_cache;

const MIB: usize = 1024 * 1024;

/// Restores the production default (`shared_db_cache` turns sharing on) so a
/// failing test cannot leak `false` into later tests.
struct ShareGuard;
impl ShareGuard {
    fn set(share: bool) -> Self {
        slatedb::db_cache::set_share_compacted_entries(share);
        ShareGuard
    }
}
impl Drop for ShareGuard {
    fn drop(&mut self) {
        slatedb::db_cache::set_share_compacted_entries(true);
    }
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap()
}

fn writer_settings(compact: bool) -> Settings {
    let mut s = Settings::default();
    s.min_filter_keys = 1;
    s.flush_interval = Some(Duration::from_millis(10));
    if compact {
        let mut c = s.compactor_options.clone().unwrap_or_default();
        c.poll_interval = Duration::from_millis(20);
        c.commit_compacted_interval = Duration::from_millis(20);
        s.compactor_options = Some(c);
        let mut dir = GarbageCollectorDirectoryOptions::default();
        dir.interval = Some(Duration::from_millis(50));
        dir.min_age = Duration::from_millis(0);
        let mut gc = GarbageCollectorOptions::default();
        gc.compacted_options = Some(dir);
        gc.wal_options = Some(dir);
        s.garbage_collector_options = Some(gc);
    } else {
        // No compactor: lift the L0 backpressure cap or memtable flushes stall.
        s.compactor_options = None;
        s.garbage_collector_options = None;
        s.l0_max_ssts = 64;
        s.l0_max_ssts_per_key = 64;
    }
    s
}

async fn open_writer(
    path: &str,
    store: Arc<dyn ObjectStore>,
    cache: Arc<dyn DbCache>,
    compact: bool,
) -> Db {
    Db::builder(path, store)
        .with_settings(writer_settings(compact))
        .with_db_cache(cache)
        .build()
        .await
        .expect("open writer")
}

async fn open_reader(path: &str, store: Arc<dyn ObjectStore>, cache: Arc<dyn DbCache>) -> DbReader {
    let mut options = DbReaderOptions::default();
    options.manifest_poll_interval = Duration::from_millis(20);
    // The default lease (10 min), not a short one. A manifest stores a checkpoint's
    // expiry in whole seconds, so a 1 s lease ends at the next second boundary: a
    // writer's GC reaps it before the reader's half-life refresh, and reads fail
    // with "checkpoint lease lost" until the reader has a new one.
    DbReader::builder(path, store)
        .with_options(options)
        .with_db_cache(cache)
        .build()
        .await
        .expect("open reader")
}

fn lazy_write() -> WriteOptions {
    let mut o = WriteOptions::default();
    o.await_durable = false;
    o
}

/// Non-durable writes; tests make them durable with an explicit [`flush`].
async fn put(db: &Db, k: impl AsRef<[u8]>, v: impl AsRef<[u8]>) {
    db.put_with_options(k, v, &PutOptions::default(), &lazy_write())
        .await
        .expect("put");
}

async fn del(db: &Db, k: impl AsRef<[u8]>) {
    db.delete_with_options(k, &lazy_write())
        .await
        .expect("delete");
}

async fn flush(db: &Db, flush_type: FlushType) {
    db.flush_with_options(FlushOptions { flush_type })
        .await
        .expect("flush");
}

/// Simulates the drop purge: every object under the collection prefix goes.
async fn purge(store: &Arc<dyn ObjectStore>, path: &str) {
    use futures::TryStreamExt;
    let prefix = ObjPath::from(path);
    let listed: Vec<_> = store.list(Some(&prefix)).try_collect().await.unwrap();
    assert!(!listed.is_empty(), "nothing to purge under {path}");
    for meta in listed {
        store.delete(&meta.location).await.unwrap();
    }
}

/// Object names under `{path}/wal/` (WAL SST ids as stored).
async fn wal_names(store: &Arc<dyn ObjectStore>, path: &str) -> std::collections::BTreeSet<String> {
    use futures::TryStreamExt;
    let prefix = ObjPath::from(format!("{path}/wal"));
    let listed: Vec<_> = store.list(Some(&prefix)).try_collect().await.unwrap();
    listed
        .into_iter()
        .map(|m| m.location.filename().unwrap().to_string())
        .collect()
}

fn key(i: usize) -> Vec<u8> {
    format!("key-{i:06}").into_bytes()
}

/// Two collections whose WAL SST ids collide (both start at the same id) and
/// whose keys/value layouts are identical must never read each other's rows
/// through the shared cache, with the WAL tail cached by replay.
#[test]
#[serial_test::serial]
fn shared_scope_keeps_colliding_wal_ids_isolated_across_collections() {
    let _share = ShareGuard::set(true);
    let rt = runtime();
    rt.block_on(async {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let cache = build_db_cache(64 * MIB, 64 * MIB, 4);
        let a = open_writer("review/wal-a", store.clone(), cache.clone(), false).await;
        let b = open_writer("review/wal-b", store.clone(), cache.clone(), false).await;
        // Compacted part (L0) plus a WAL-only tail in each collection.
        for (db, tag) in [(&a, "A"), (&b, "B")] {
            for i in 0..50 {
                put(db, key(i), format!("{tag}-l0-{i:06}")).await;
            }
            flush(db, FlushType::MemTable).await;
            for i in 50..100 {
                put(db, key(i), format!("{tag}-wal-{i:06}")).await;
            }
            flush(db, FlushType::Wal).await;
        }
        let (wa, wb) = (
            wal_names(&store, "review/wal-a").await,
            wal_names(&store, "review/wal-b").await,
        );
        assert!(
            wa.intersection(&wb).count() > 0,
            "fixture must make WAL ids collide: {wa:?} {wb:?}"
        );
        let ra = open_reader("review/wal-a", store.clone(), cache.clone()).await;
        let rb = open_reader("review/wal-b", store.clone(), cache.clone()).await;
        // Interleave so whichever collection loads first would poison the other.
        for round in 0..2 {
            for i in 0..100 {
                for (r, tag) in [(&ra, "A"), (&rb, "B")] {
                    let got = r.get(key(i)).await.unwrap().expect("row present");
                    let want = if i < 50 {
                        format!("{tag}-l0-{i:06}")
                    } else {
                        format!("{tag}-wal-{i:06}")
                    };
                    assert_eq!(got.as_ref(), want.as_bytes(), "round {round} key {i}");
                }
            }
        }
        // A reopened handle (fresh WAL scope, shared compacted scope) too.
        let rb2 = open_reader("review/wal-b", store.clone(), cache.clone()).await;
        for i in 0..100 {
            let got = rb2.get(key(i)).await.unwrap().unwrap();
            assert!(
                got.starts_with(b"B-"),
                "reopened B read {got:?} for key {i}"
            );
        }
        for r in [ra, rb, rb2] {
            r.close().await.unwrap();
        }
        a.close().await.unwrap();
        b.close().await.unwrap();
    });
}

/// Drop + recreate under the same path: WAL ids restart and keys/offsets line
/// up exactly, so any reuse of the old collection's cached WAL/L0 entries
/// would surface the old values or resurrect rows the new collection lacks.
#[test]
#[serial_test::serial]
fn recreated_collection_never_serves_dropped_collection_cache_entries() {
    for share in [false, true] {
        let _share = ShareGuard::set(share);
        let rt = runtime();
        rt.block_on(async {
            let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
            let cache = build_db_cache(64 * MIB, 64 * MIB, 4);
            let path = "review/recreate";

            let w1 = open_writer(path, store.clone(), cache.clone(), false).await;
            for i in 0..60 {
                put(&w1, key(i), format!("old-l0-{i:06}")).await;
            }
            flush(&w1, FlushType::MemTable).await;
            for i in 60..120 {
                put(&w1, key(i), format!("old-wl-{i:06}")).await;
            }
            flush(&w1, FlushType::Wal).await;
            let r1 = open_reader(path, store.clone(), cache.clone()).await;
            let old_wals = wal_names(&store, path).await;
            for i in 0..120 {
                assert!(r1.get(key(i)).await.unwrap().unwrap().starts_with(b"old-"));
            }
            r1.close().await.unwrap();
            w1.close().await.unwrap();

            purge(&store, path).await;

            // Same layout, different bytes, fewer rows (100..120 must stay gone).
            let w2 = open_writer(path, store.clone(), cache.clone(), false).await;
            for i in 0..60 {
                put(&w2, key(i), format!("new-l0-{i:06}")).await;
            }
            flush(&w2, FlushType::MemTable).await;
            for i in 60..100 {
                put(&w2, key(i), format!("new-wl-{i:06}")).await;
            }
            flush(&w2, FlushType::Wal).await;
            let new_wals = wal_names(&store, path).await;
            assert!(
                old_wals.intersection(&new_wals).count() > 0,
                "fixture must reuse the dropped collection's WAL ids: {old_wals:?} vs {new_wals:?}"
            );
            let r2 = open_reader(path, store.clone(), cache.clone()).await;
            for i in 0..120 {
                let got = r2.get(key(i)).await.unwrap();
                if i < 100 {
                    let got = got.unwrap_or_else(|| panic!("share={share} key {i} missing"));
                    assert!(
                        got.starts_with(b"new-"),
                        "share={share} key {i} read {got:?}"
                    );
                } else {
                    assert!(got.is_none(), "share={share} key {i} resurrected: {got:?}");
                }
            }
            let mut it = r2.scan(key(0)..key(1000)).await.unwrap();
            let mut n = 0;
            while let Some(kv) = it.next().await.unwrap() {
                assert!(kv.value.starts_with(b"new-"), "share={share} scan {kv:?}");
                n += 1;
            }
            assert_eq!(n, 100, "share={share}");
            r2.close().await.unwrap();
            w2.close().await.unwrap();
        });
    }
}

fn expected_after_rounds(rounds: usize, keys: usize) -> BTreeMap<Vec<u8>, Vec<u8>> {
    let mut want = BTreeMap::new();
    for round in 0..rounds {
        for i in 0..keys {
            if (i + round) % 7 == 0 {
                want.remove(&key(i));
            } else {
                want.insert(key(i), format!("r{round}-{i:06}").into_bytes());
            }
        }
    }
    want
}

async fn read_all(r: &DbReader, keys: usize) -> BTreeMap<Vec<u8>, Vec<u8>> {
    let mut got = BTreeMap::new();
    for i in 0..keys {
        if let Some(v) = r.get(key(i)).await.unwrap() {
            got.insert(key(i), v.to_vec());
        }
    }
    got
}

/// Compaction rewrites every L0 into new ULID SSTs and GC deletes the inputs.
/// A reader opened before compaction (its entries cached under the shared
/// scope), a reader opened after GC, and a refreshed long-lived reader must
/// all read exactly the latest values, with sharing on and off.
#[test]
#[serial_test::serial]
fn compaction_and_gc_keep_shared_scope_reads_exact() {
    const KEYS: usize = 400;
    const ROUNDS: usize = 6;
    for share in [false, true] {
        let _share = ShareGuard::set(share);
        let rt = runtime();
        rt.block_on(async {
            let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
            let cache = build_db_cache(64 * MIB, 64 * MIB, 4);
            let path = "review/compact";
            let w = open_writer(path, store.clone(), cache.clone(), true).await;
            let early = {
                for i in 0..KEYS {
                    put(&w, key(i), format!("r0-{i:06}")).await;
                }
                flush(&w, FlushType::MemTable).await;
                let r = open_reader(path, store.clone(), cache.clone()).await;
                let initial: BTreeMap<_, _> = (0..KEYS)
                    .map(|i| (key(i), format!("r0-{i:06}").into_bytes()))
                    .collect();
                assert_eq!(read_all(&r, KEYS).await, initial);
                r
            };
            for round in 0..ROUNDS {
                for i in 0..KEYS {
                    if (i + round) % 7 == 0 {
                        del(&w, key(i)).await;
                    } else {
                        put(&w, key(i), format!("r{round}-{i:06}")).await;
                    }
                }
                flush(&w, FlushType::MemTable).await;
            }
            let want = expected_after_rounds(ROUNDS, KEYS);
            let early_l0s: Vec<_> = early.manifest().l0().iter().map(|v| v.sst.id).collect();
            let deadline = Instant::now() + Duration::from_secs(60);
            while w.manifest().compacted().is_empty() || w.manifest().l0().len() >= ROUNDS {
                assert!(Instant::now() < deadline, "no compaction");
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            // The early reader (entries for the pre-compaction L0s cached under
            // its scope or the shared one) polls onto the compacted manifest.
            tokio::time::sleep(Duration::from_millis(300)).await;
            assert_eq!(
                read_all(&early, KEYS).await,
                want,
                "share={share} early reader"
            );
            // Its checkpoint pins the inputs; closing it releases the checkpoint
            // and lets GC delete them.
            early.close().await.unwrap();
            // Leave the writer's GC (50 ms interval) time to do that.
            // These rows land in fresh L0s next to the compacted run.
            tokio::time::sleep(Duration::from_millis(1500)).await;
            for i in 0..KEYS / 2 {
                put(&w, key(i), format!("post-{i:06}")).await;
            }
            flush(&w, FlushType::MemTable).await;
            let mut want = want;
            for i in 0..KEYS / 2 {
                want.insert(key(i), format!("post-{i:06}").into_bytes());
            }
            // GC stand-in: delete every compacted object the latest manifest no
            // longer references, as GC would, however far the writer's own GC
            // has got.
            let m = w.manifest();
            let live: std::collections::HashSet<String> = m
                .l0()
                .iter()
                .chain(m.compacted().iter().flat_map(|sr| sr.sst_views.iter()))
                .map(|v| v.sst.id.unwrap_compacted_id().to_string())
                .collect();
            let compacted_dir = ObjPath::from(format!("{path}/compacted"));
            let early_ids: Vec<String> = early_l0s
                .iter()
                .map(|id| id.unwrap_compacted_id().to_string())
                .collect();
            {
                use futures::TryStreamExt;
                let files: Vec<_> = store
                    .list(Some(&compacted_dir))
                    .try_collect()
                    .await
                    .unwrap();
                let mut deleted = Vec::new();
                for f in files {
                    let name = f
                        .location
                        .filename()
                        .unwrap()
                        .split('.')
                        .next()
                        .unwrap()
                        .to_string();
                    if !live.contains(&name) {
                        store.delete(&f.location).await.unwrap();
                        deleted.push(name);
                    }
                }
                assert!(
                    deleted.iter().any(|d| early_ids.contains(d)),
                    "no SST the early reader cached was rewritten away: deleted {deleted:?}"
                );
                println!("share={share}: GC stand-in deleted {} SSTs", deleted.len());
            }
            let late = open_reader(path, store.clone(), cache.clone()).await;
            assert!(!late.manifest().compacted().is_empty());
            assert_eq!(
                read_all(&late, KEYS).await,
                want,
                "share={share} late reader"
            );
            let late2 = open_reader(path, store.clone(), cache.clone()).await;
            assert_eq!(
                read_all(&late2, KEYS).await,
                want,
                "share={share} reopened reader"
            );
            late2.close().await.unwrap();
            late.close().await.unwrap();
            w.close().await.unwrap();
        });
    }
}

/// Snapshot of every observable read (get, multi_get, scan) over a dataset of
/// overwrites, deletes, several L0s and a WAL tail.
async fn observe(r: &DbReader, keys: usize) -> Vec<(Vec<u8>, Option<Vec<u8>>)> {
    let mut out = Vec::new();
    for i in 0..keys + 10 {
        out.push((key(i), r.get(key(i)).await.unwrap().map(|v| v.to_vec())));
    }
    let batch: Vec<Vec<u8>> = (0..keys + 10).rev().step_by(3).map(key).collect();
    for (k, v) in batch.iter().zip(r.multi_get(&batch).await.unwrap()) {
        out.push((k.clone(), v.map(|v| v.to_vec())));
    }
    let mut it = r.scan(key(0)..key(keys + 10)).await.unwrap();
    while let Some(kv) = it.next().await.unwrap() {
        out.push((kv.key.to_vec(), Some(kv.value.to_vec())));
    }
    out
}

/// Results must be byte-identical with sharing on/off and across meta shard
/// splits, including caches small enough that every index/filter overflows
/// its shard and eviction runs on every read.
#[test]
#[serial_test::serial]
fn reads_are_identical_across_share_and_shard_configs() {
    const KEYS: usize = 3_000;
    let rt = runtime();
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "review/parity";
    rt.block_on(async {
        let w = Db::builder(path, store.clone())
            .with_settings(writer_settings(false))
            .with_db_cache_disabled()
            .build()
            .await
            .unwrap();
        let mut rng = StdRng::seed_from_u64(11);
        for round in 0..5 {
            for _ in 0..KEYS {
                let i = rng.random_range(0..KEYS);
                if rng.random_range(0..10) == 0 {
                    del(&w, key(i)).await;
                } else {
                    let len = rng.random_range(1..300);
                    put(&w, key(i), vec![b'a' + round as u8; len]).await;
                }
            }
            flush(
                &w,
                if round < 4 {
                    FlushType::MemTable
                } else {
                    FlushType::Wal
                },
            )
            .await;
        }
        w.close().await.unwrap();
    });
    let baseline = {
        let _share = ShareGuard::set(false);
        rt.block_on(async {
            let r = DbReader::builder(path, store.clone())
                .with_db_cache_disabled()
                .build()
                .await
                .unwrap();
            let o = observe(&r, KEYS).await;
            r.close().await.unwrap();
            o
        })
    };
    assert!(
        baseline.iter().filter(|(_, v)| v.is_none()).count() > 0,
        "deletes visible"
    );
    // (share, block bytes, meta bytes, meta shards)
    let configs = [
        (false, 64 * MIB, 128 * MIB, 16),
        (true, 64 * MIB, 128 * MIB, 2),
        (true, 64 * MIB, 128 * MIB, 16),
        (true, 64 * 1024, 16 * 1024, 16),
        (false, 64 * 1024, 16 * 1024, 16),
        (true, 64 * 1024, 16 * 1024, 1),
    ];
    for (share, block, meta, shards) in configs {
        let _share = ShareGuard::set(share);
        rt.block_on(async {
            let cache = build_db_cache(block, meta, shards);
            for pass in 0..3 {
                // Reopen each pass: later handles read what earlier ones cached.
                let r = open_reader(path, store.clone(), cache.clone()).await;
                let got = observe(&r, KEYS).await;
                assert!(
                    got == baseline,
                    "share={share} block={block} meta={meta} shards={shards} pass={pass} diverged"
                );
                r.close().await.unwrap();
            }
        });
    }
}

/// Readers on two collections with identical keys churn handles (rebuild +
/// delayed close, like poll-tier promote/demote) under a tiny shared cache
/// while eight tasks read through whatever handle is current. Every read must
/// return its own collection's row; nothing may hang.
#[test]
#[serial_test::serial]
fn handle_churn_under_shared_scope_never_crosses_collections() {
    const KEYS: usize = 2_000;
    let _share = ShareGuard::set(true);
    let rt = runtime();
    rt.block_on(async {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let cache = build_db_cache(256 * 1024, 128 * 1024, 4);
        for tag in ["A", "B"] {
            let w = open_writer(
                &format!("review/churn-{tag}"),
                store.clone(),
                cache.clone(),
                false,
            )
            .await;
            for half in 0..2 {
                for i in (half * KEYS / 2)..((half + 1) * KEYS / 2) {
                    put(&w, key(i), format!("{tag}-{i:06}")).await;
                }
                flush(&w, FlushType::MemTable).await;
            }
            w.close().await.unwrap();
        }
        let slots: Vec<Arc<std::sync::RwLock<Arc<DbReader>>>> =
            futures::future::join_all(["A", "B"].map(|tag| {
                let store = store.clone();
                let cache = cache.clone();
                async move {
                    Arc::new(std::sync::RwLock::new(Arc::new(
                        open_reader(&format!("review/churn-{tag}"), store, cache).await,
                    )))
                }
            }))
            .await;
        let stop = Instant::now() + Duration::from_secs(4);
        let reads = Arc::new(AtomicU64::new(0));
        let mut tasks = Vec::new();
        for (c, tag) in ["A", "B"].into_iter().enumerate() {
            let slot = slots[c].clone();
            let store = store.clone();
            let cache = cache.clone();
            tasks.push(tokio::spawn(async move {
                let mut swaps = 0u64;
                while Instant::now() < stop {
                    let fresh = Arc::new(
                        open_reader(&format!("review/churn-{tag}"), store.clone(), cache.clone())
                            .await,
                    );
                    let old = std::mem::replace(&mut *slot.write().unwrap(), fresh);
                    tokio::spawn(async move {
                        tokio::time::sleep(Duration::from_millis(300)).await;
                        let _ = old.close().await;
                    });
                    swaps += 1;
                    tokio::time::sleep(Duration::from_millis(15)).await;
                }
                swaps
            }));
        }
        for t in 0..8u64 {
            let slots = slots.clone();
            let reads = reads.clone();
            tasks.push(tokio::spawn(async move {
                let mut rng = StdRng::seed_from_u64(t);
                while Instant::now() < stop {
                    let c = rng.random_range(0..2usize);
                    let tag = ["A", "B"][c];
                    let r = slots[c].read().unwrap().clone();
                    let i = rng.random_range(0..KEYS);
                    if rng.random_range(0..4) == 0 {
                        let ks: Vec<Vec<u8>> = (0..8).map(|j| key((i + j * 97) % KEYS)).collect();
                        for (j, v) in r.multi_get(&ks).await.unwrap().into_iter().enumerate() {
                            let want = format!("{tag}-{:06}", (i + j * 97) % KEYS);
                            assert_eq!(v.as_deref(), Some(want.as_bytes()));
                        }
                    } else {
                        let v = r.get(key(i)).await.unwrap();
                        assert_eq!(v.as_deref(), Some(format!("{tag}-{i:06}").as_bytes()));
                    }
                    reads.fetch_add(1, Ordering::Relaxed);
                }
                0
            }));
        }
        let joined =
            tokio::time::timeout(Duration::from_secs(60), futures::future::join_all(tasks))
                .await
                .expect("churn hung");
        let swaps: u64 = joined.into_iter().map(|r| r.expect("task panicked")).sum();
        println!(
            "churn: {swaps} handle swaps, {} reads",
            reads.load(Ordering::Relaxed)
        );
        assert!(swaps > 20 && reads.load(Ordering::Relaxed) > 1_000);
        for slot in slots {
            let r = slot.read().unwrap().clone();
            r.close().await.unwrap();
        }
    });
}

fn used_and_capacity(cache: &dyn DbCache) -> (u64, u64) {
    match cache.usage_snapshot().memory {
        CacheUsageSnapshot::Ready {
            used_bytes,
            capacity_bytes,
        } => (used_bytes, capacity_bytes.unwrap_or(0)),
        other => panic!("unexpected usage snapshot {other:?}"),
    }
}

/// Meta-cache memory bound. Foyer admits an entry larger than its shard by
/// evicting the shard and inserting anyway, so resident bytes are bounded by
/// `max(capacity, shards * largest entry)`. Derived shards keep usage near the
/// configured cap; an over-sharded override (the `HELIX_LSM_META_CACHE_SHARDS`
/// env knob is unclamped) lets usage run to many times the cap.
#[test]
#[serial_test::serial]
fn meta_cache_usage_bound_by_capacity_or_shards_times_largest_entry() {
    let _share = ShareGuard::set(true);
    let rt = runtime();
    rt.block_on(async {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = "review/meta-bound";
        let w = open_writer(path, store.clone(), build_db_cache(MIB, MIB, 1), false).await;
        const SSTS: usize = 24;
        for s in 0..SSTS {
            for i in 0..4_000 {
                put(&w, format!("s{s:02}-{i:06}"), vec![7u8; 64]).await;
            }
            flush(&w, FlushType::MemTable).await;
        }
        w.close().await.unwrap();
        let capacity = 64 * 1024u64;
        let mut results = Vec::new();
        for shards in [1usize, 16, 256] {
            let meta = Arc::new(FoyerCache::new_with_opts(FoyerCacheOptions {
                max_capacity: capacity,
                shards,
            }));
            let cache: Arc<dyn DbCache> = Arc::new(
                SplitCache::new()
                    .with_block_cache(Some(Arc::new(FoyerCache::new_with_opts(
                        FoyerCacheOptions {
                            max_capacity: 4 * MIB as u64,
                            shards: 4,
                        },
                    ))))
                    .with_meta_cache(Some(meta.clone()))
                    .build(),
            );
            let r = open_reader(path, store.clone(), cache).await;
            let mut peak = 0;
            for s in 0..SSTS {
                for i in (0..4_000).step_by(500) {
                    assert!(r.get(format!("s{s:02}-{i:06}")).await.unwrap().is_some());
                    peak = peak.max(used_and_capacity(meta.as_ref()).0);
                }
            }
            r.close().await.unwrap();
            println!(
                "meta cap {capacity} B, {shards} shards: peak used {peak} B ({:.1}x cap)",
                peak as f64 / capacity as f64
            );
            results.push((shards, peak));
        }
        // Derived/single-shard split: at most one oversize entry past the cap.
        assert!(results[0].1 <= capacity * 2, "{results:?}");
        // Over-sharded: shards * largest entry, far past the cap.
        assert!(
            results[2].1 > capacity * 2,
            "expected overshoot evidence: {results:?}"
        );
    });
}

/// Hot-path contention: point reads on a warm reader through a 2-shard vs a
/// 16-shard meta cache (the new derived split of the 128 MiB default vs the
/// old per-CPU split). Foyer's LRU takes the shard write lock on every hit.
#[test]
#[serial_test::serial]
#[ignore = "throughput measurement; run by hand with --ignored --nocapture"]
fn meta_cache_shard_count_contention() {
    const KEYS: usize = 20_000;
    let _share = ShareGuard::set(true);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(16)
        .enable_all()
        .build()
        .unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "review/contention";
    rt.block_on(async {
        let w = open_writer(path, store.clone(), build_db_cache(MIB, MIB, 1), false).await;
        for s in 0..8 {
            for i in 0..KEYS {
                if i % 8 == s {
                    put(&w, key(i), vec![1u8; 100]).await;
                }
            }
            flush(&w, FlushType::MemTable).await;
        }
        w.close().await.unwrap();
    });
    for shards in [2usize, 16, 2, 16] {
        let ops = rt.block_on(async {
            let cache = build_db_cache(512 * MIB, 128 * MIB, shards);
            let r = Arc::new(open_reader(path, store.clone(), cache).await);
            for i in 0..KEYS {
                r.get(key(i)).await.unwrap();
            }
            let stop = Instant::now() + Duration::from_secs(3);
            let total = Arc::new(AtomicU64::new(0));
            let tasks: Vec<_> = (0..32u64)
                .map(|t| {
                    let r = r.clone();
                    let total = total.clone();
                    tokio::spawn(async move {
                        let mut rng = StdRng::seed_from_u64(t);
                        let mut n = 0u64;
                        while Instant::now() < stop {
                            r.get(key(rng.random_range(0..KEYS))).await.unwrap();
                            n += 1;
                        }
                        total.fetch_add(n, Ordering::Relaxed);
                    })
                })
                .collect();
            futures::future::join_all(tasks).await;
            let observer = QueryMetricsObserver::default();
            scope_query_metrics(observer.clone(), r.get(key(1)))
                .await
                .unwrap();
            r.close().await.unwrap();
            total.load(Ordering::Relaxed) / 3
        });
        println!("meta shards {shards}: {ops} point reads/s (32 tasks, 16 workers)");
    }
}
