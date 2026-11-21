//! Adversarial tests for the bounded LSM disk-cache evictor.
//! Run with `--test-threads=1`: tests share process env and the global registry.

use helixdb::helix_engine::storage_core::backend::{Namespace, StorageBackend};
use helixdb::helix_engine::storage_core::backend_lsm::cache_evictor;
use helixdb::helix_engine::storage_core::backend_lsm::LsmBackend;
use helixdb::helix_engine::storage_core::backend_lsm_reader::LsmReader;
use object_store::local::LocalFileSystem;
use object_store::ObjectStore;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Once};
use std::time::{Duration, Instant, SystemTime};
use tempfile::TempDir;

const KEYS: usize = 3000;

/// Boot the shared LSM runtime with NO cache dir so its background evictor never
/// starts (a startup sweep at the laptop's real volume usage would race these
/// tests), and pin the knobs that force reads through the disk cache.
fn setup() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        std::env::remove_var("HELIX_LSM_CACHE_DIR");
        std::env::set_var("HELIX_LSM_BLOCK_CACHE_MB", "1");
        std::env::set_var("HELIX_LSM_SHARE_SST_CACHE", "0");
        std::env::set_var("HELIX_LSM_CACHE_PART_BYTES", "65536");
        std::env::set_var("HELIX_LSM_CACHE_MAX_BYTES", "1073741824");
        std::env::set_var("HELIX_LSM_IDLE_FLUSH_SECS", "1");
        drop(LsmBackend::open_in_memory("boot").unwrap());
    });
}

fn value(i: usize, gen: u8) -> Vec<u8> {
    let mut v = format!("v{gen}-{i:06}-").into_bytes();
    v.resize(512, b'a' + (i % 26) as u8);
    v
}

fn key(i: usize) -> Vec<u8> {
    format!("k{i:06}").into_bytes()
}

fn write_keys(w: &LsmBackend, gen: u8) {
    for chunk in (0..KEYS).collect::<Vec<_>>().chunks(500) {
        let mut txn = w.begin_write().unwrap();
        for &i in chunk {
            w.put(&mut txn, Namespace::Nodes, &key(i), &value(i, gen))
                .unwrap();
        }
        w.commit(txn).unwrap();
    }
}

fn count_parts(dir: &Path) -> usize {
    fn walk(p: &Path, n: &mut usize) {
        let Ok(rd) = fs::read_dir(p) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, n);
            } else if e.file_name().to_string_lossy().starts_with("_part") {
                *n += 1;
            }
        }
    }
    let mut n = 0;
    walk(dir, &mut n);
    n
}

fn part_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            part_files(&p, out);
        } else if e.file_name().to_string_lossy().starts_with("_part") {
            out.push(p);
        }
    }
}

fn wait_for_parts(dir: &Path, at_least: usize) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while count_parts(dir) < at_least {
        assert!(
            Instant::now() < deadline,
            "no cached SST parts under {}",
            dir.display()
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn verify_reader(r: &LsmReader, gen: u8) {
    for i in 0..KEYS {
        let got = r
            .get_with(Namespace::Nodes, &key(i), |v| v.map(|b| b.to_vec()))
            .unwrap();
        assert_eq!(
            got.as_deref(),
            Some(&value(i, gen)[..]),
            "reader key {i} gen {gen}"
        );
    }
}

fn verify_writer(w: &LsmBackend, gen: u8) {
    let txn = w.begin_read().unwrap();
    for i in 0..KEYS {
        let got = w
            .get_with(&txn, Namespace::Nodes, &key(i), |v| v.map(|b| b.to_vec()))
            .unwrap();
        assert_eq!(
            got.as_deref(),
            Some(&value(i, gen)[..]),
            "writer key {i} gen {gen}"
        );
    }
}

struct Fixture {
    _tmp: TempDir,
    cache_root: PathBuf,
    store: Arc<dyn ObjectStore>,
}

fn fixture() -> Fixture {
    setup();
    let tmp = TempDir::new().unwrap();
    let cache_root = tmp.path().join("cache");
    let store_root = tmp.path().join("store");
    fs::create_dir_all(&cache_root).unwrap();
    fs::create_dir_all(&store_root).unwrap();
    std::env::set_var("HELIX_LSM_CACHE_DIR", cache_root.to_str().unwrap());
    let store: Arc<dyn ObjectStore> =
        Arc::new(LocalFileSystem::new_with_prefix(&store_root).unwrap());
    Fixture {
        _tmp: tmp,
        cache_root,
        store,
    }
}

/// Probe 2: the cache subtree of an OPEN writer and reader vanishes (rename, then
/// delete, then partial part deletion) — every read must still return the right
/// bytes (fallback to the object store) and later writes/flushes must succeed.
#[test]
fn open_handles_survive_cache_dir_rename_delete_and_partial_loss() {
    let f = fixture();
    let path = "prod/vanish";
    let subtree = f.cache_root.join("prod_vanish");
    let w = LsmBackend::open_with_store(path, f.store.clone()).unwrap();
    write_keys(&w, 1);
    wait_for_parts(&subtree, 2); // idle flush wrote L0 SSTs through the cache
    let r = LsmReader::open_with_store(path, f.store.clone()).unwrap();
    verify_reader(&r, 1);
    verify_writer(&w, 1);

    // (a) renamed away while open (what the evictor does to a closed dir).
    let moved = f.cache_root.join(".evicting-prod_vanish-1");
    fs::rename(&subtree, &moved).unwrap();
    verify_reader(&r, 1);
    verify_writer(&w, 1);
    // (b) deleted outright while open.
    fs::remove_dir_all(&moved).unwrap();
    if subtree.exists() {
        fs::remove_dir_all(&subtree).unwrap();
    }
    verify_reader(&r, 1);
    verify_writer(&w, 1);
    let fresh = LsmReader::open_with_store(path, f.store.clone()).unwrap();
    verify_reader(&fresh, 1);
    assert!(
        count_parts(&subtree) > 0,
        "fallback reads must refill the cache"
    );

    // (c) half of the parts disappear under a live handle.
    let mut parts = Vec::new();
    part_files(&subtree, &mut parts);
    for p in parts.iter().step_by(2) {
        fs::remove_file(p).unwrap();
    }
    verify_reader(&fresh, 1);
    verify_reader(&r, 1);

    // (d) writes + memtable flush after the dir vanished still work end to end.
    fs::remove_dir_all(&subtree).unwrap();
    write_keys(&w, 2);
    wait_for_parts(&subtree, 1);
    verify_writer(&w, 2);
    let after = LsmReader::open_with_store(path, f.store.clone()).unwrap();
    verify_reader(&after, 2);
}

/// Core claim: a file created in a registered dir is never removed while the
/// guard lives, under a tight register/evict race (no SlateDB timing involved).
#[test]
fn registered_dir_contents_never_removed_under_tight_race() {
    let f = fixture();
    let dir = f.cache_root.join("prod_racer");
    let stop = Arc::new(AtomicBool::new(false));
    let evictions = Arc::new(AtomicUsize::new(0));
    let evictor = {
        let (root, stop, evictions) = (f.cache_root.clone(), stop.clone(), evictions.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                if cache_evictor::evict_lru_dirs(&root, u64::MAX).unwrap_or(0) > 0 {
                    evictions.fetch_add(1, Ordering::Relaxed);
                }
            }
        })
    };
    for i in 0..3000 {
        let guard = cache_evictor::register_cache_dir(dir.clone());
        fs::create_dir_all(dir.join("prod/racer/compacted")).unwrap();
        let file = dir.join(format!("prod/racer/compacted/{i}"));
        fs::write(&file, b"x").unwrap();
        std::thread::yield_now();
        assert!(
            file.exists(),
            "iteration {i}: file in a registered dir was evicted"
        );
        drop(guard);
    }
    stop.store(true, Ordering::Relaxed);
    evictor.join().unwrap();
    assert!(
        evictions.load(Ordering::Relaxed) > 0,
        "race never exercised an eviction"
    );
}

/// Real handles opening/closing while an evictor hammers: every read through a
/// freshly opened reader must return the committed bytes.
#[test]
fn reads_stay_correct_while_evictor_hammers_open_close_cycles() {
    let f = fixture();
    let names: Vec<String> = (0..3).map(|i| format!("prod/hammer{i}")).collect();
    for n in &names {
        let w = LsmBackend::open_with_store(n, f.store.clone()).unwrap();
        write_keys(&w, 7);
        wait_for_parts(&f.cache_root.join(n.replace('/', "_")), 1);
    }
    let stop = Arc::new(AtomicBool::new(false));
    let evictor = {
        let (root, stop) = (f.cache_root.clone(), stop.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                cache_evictor::evict_lru_dirs(&root, u64::MAX).ok();
                std::thread::sleep(Duration::from_millis(2));
            }
        })
    };
    for _ in 0..6 {
        for n in &names {
            let r = LsmReader::open_with_store(n, f.store.clone()).unwrap();
            verify_reader(&r, 7);
        }
    }
    stop.store(true, Ordering::Relaxed);
    evictor.join().unwrap();
}

fn mtime(p: &Path) -> SystemTime {
    fs::metadata(p).unwrap().modified().unwrap()
}

/// LRU inversion across a restart. A hot collection held open for days is only
/// touched at open; the process is then killed (no guard drop, so no close
/// touch). After restart its dir looks OLDER than a cold collection that was
/// opened and closed later, so the startup sweep evicts the hot one first —
/// exactly the dirs reader warm-before-ready reopens. Periodic sweeps must keep
/// open dirs' mtimes current.
#[test]
fn long_open_dir_is_not_lru_oldest_after_process_kill() {
    let f = fixture();
    let hot = f.cache_root.join("prod_hot");
    let cold = f.cache_root.join("prod_cold");
    for d in [&hot, &cold] {
        fs::create_dir_all(d).unwrap();
        fs::write(d.join("data"), vec![0_u8; 4096]).unwrap();
    }
    let hot_guard = cache_evictor::register_cache_dir(hot.clone()); // opened long ago
    std::thread::sleep(Duration::from_millis(1100));
    drop(cache_evictor::register_cache_dir(cold.clone())); // cold open+close, later
    std::thread::sleep(Duration::from_millis(1100));
    // A periodic sweep tick while `hot` is still open (budget 100 = nothing to free).
    std::env::set_var("HELIX_LSM_CACHE_BUDGET_PCT", "100");
    cache_evictor::evict_if_over_budget(&f.cache_root);
    std::env::remove_var("HELIX_LSM_CACHE_BUDGET_PCT");
    std::mem::forget(hot_guard); // SIGKILL / process exit: no close-time touch

    // "Restart": the registry is per-process; a moved root has no registered dirs.
    let restarted = f.cache_root.with_file_name("cache-after-restart");
    fs::rename(&f.cache_root, &restarted).unwrap();
    assert!(
        mtime(&restarted.join("prod_hot")) > mtime(&restarted.join("prod_cold")),
        "a dir open at the last sweep must look more recent than one closed before it"
    );
    cache_evictor::evict_lru_dirs(&restarted, 4096).unwrap();
    assert!(
        restarted.join("prod_hot").exists(),
        "hot dir evicted first after restart"
    );
    assert!(
        !restarted.join("prod_cold").exists(),
        "cold dir should be the LRU victim"
    );
}

/// Degenerate: open dirs alone exceed what must be freed. One pass must return
/// (no spin), never touch the open dir, and a second pass frees nothing.
#[test]
fn unreachable_target_terminates_and_spares_open_dirs() {
    let f = fixture();
    let open = f.cache_root.join("prod_open_big");
    fs::create_dir_all(&open).unwrap();
    fs::write(open.join("data"), vec![0_u8; 1 << 20]).unwrap();
    let _g = cache_evictor::register_cache_dir(open.clone());
    for i in 0..20 {
        let d = f.cache_root.join(format!("prod_closed{i}"));
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join("data"), vec![0_u8; 1024]).unwrap();
    }
    let t = Instant::now();
    let freed = cache_evictor::evict_lru_dirs(&f.cache_root, u64::MAX).unwrap();
    assert_eq!(freed, 20 * 1024);
    assert_eq!(
        cache_evictor::evict_lru_dirs(&f.cache_root, u64::MAX).unwrap(),
        0
    );
    assert!(t.elapsed() < Duration::from_secs(5));
    assert!(open.join("data").exists());
}

/// A sweep over thousands of closed dirs stays fast (runs on spawn_blocking).
#[test]
fn sweep_over_many_dirs_is_bounded() {
    let f = fixture();
    for i in 0..5000 {
        let d = f.cache_root.join(format!("prod_many{i:05}"));
        fs::create_dir_all(d.join("prod/x/compacted")).unwrap();
        fs::write(d.join("prod/x/compacted/_part64kb-000000000"), b"y").unwrap();
    }
    let t = Instant::now();
    let freed = cache_evictor::evict_lru_dirs(&f.cache_root, u64::MAX).unwrap();
    let took = t.elapsed();
    eprintln!("5000-dir sweep took {took:?}, freed {freed}");
    assert_eq!(freed, 5000);
    assert!(took < Duration::from_secs(60));
}

/// statvfs failure (missing cache root) must be a logged no-op, not a panic.
#[test]
fn statvfs_failure_is_noop() {
    setup();
    cache_evictor::evict_if_over_budget(Path::new("/definitely/not/a/cache/root"));
}
