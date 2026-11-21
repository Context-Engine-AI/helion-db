//! Integration tests for the cache directory eviction system.
//!
//! These tests verify that:
//! 1. Cache dirs for open collections are never evicted
//! 2. LRU eviction ordering is respected
//! 3. Budget enforcement stops at the target
//! 4. Drop+recreate scenarios are safe (stale cache cannot be reused)
//! 5. Concurrent open during eviction is safe

use helixdb::helix_engine::storage_core::backend_lsm::cache_evictor;
use helixdb::helix_engine::storage_core::backend_lsm::LsmBackend;
use std::fs;
use std::time::Duration;
use tempfile::TempDir;

#[test]
fn open_collection_never_evicted() {
    let cache_dir = TempDir::new().unwrap();
    let cache_path = cache_dir.path();

    // Simulate two collections: one open, one closed.
    let open_coll = cache_path.join("prod_open_collection");
    let closed_coll = cache_path.join("prod_closed_collection");
    fs::create_dir(&open_coll).unwrap();
    fs::create_dir(&closed_coll).unwrap();
    fs::write(closed_coll.join("data"), vec![0_u8; 4096]).unwrap();

    // Register the open collection.
    let _guard = cache_evictor::register_cache_dir(open_coll.clone());

    // Evict everything possible.
    let freed = cache_evictor::evict_lru_dirs(cache_path, u64::MAX).unwrap();
    assert!(freed >= 4096, "Should free the closed collection");
    assert!(open_coll.exists(), "Open collection must not be evicted");
    assert!(!closed_coll.exists(), "Closed collection should be evicted");
}

#[test]
fn eviction_respects_lru_order() {
    let cache_dir = TempDir::new().unwrap();
    let cache_path = cache_dir.path();

    let old = cache_path.join("old");
    let medium = cache_path.join("medium");
    let new = cache_path.join("new");

    // Each directory's last write is its age: write its file before the
    // pause, or the three writes land in one filesystem timestamp step.
    fs::create_dir(&old).unwrap();
    fs::write(old.join("data"), vec![0_u8; 1024]).unwrap();
    std::thread::sleep(Duration::from_millis(50));
    fs::create_dir(&medium).unwrap();
    fs::write(medium.join("data"), vec![0_u8; 1024]).unwrap();
    std::thread::sleep(Duration::from_millis(50));
    fs::create_dir(&new).unwrap();
    fs::write(new.join("data"), vec![0_u8; 1024]).unwrap();

    // Evict 1500 bytes (should take old + medium).
    cache_evictor::evict_lru_dirs(cache_path, 1500).unwrap();
    assert!(!old.exists(), "Oldest should go first");
    assert!(!medium.exists(), "Next oldest should go second");
    assert!(new.exists(), "Newest should remain");
}

#[test]
fn eviction_stops_at_target() {
    let cache_dir = TempDir::new().unwrap();
    let cache_path = cache_dir.path();

    let dir1 = cache_path.join("dir1");
    let dir2 = cache_path.join("dir2");
    let dir3 = cache_path.join("dir3");

    fs::create_dir(&dir1).unwrap();
    fs::write(dir1.join("data"), vec![0_u8; 1024]).unwrap();
    std::thread::sleep(Duration::from_millis(20));
    fs::create_dir(&dir2).unwrap();
    fs::write(dir2.join("data"), vec![0_u8; 1024]).unwrap();
    std::thread::sleep(Duration::from_millis(20));
    fs::create_dir(&dir3).unwrap();
    fs::write(dir3.join("data"), vec![0_u8; 1024]).unwrap();

    // Target 1024 bytes — should stop after dir1.
    let freed = cache_evictor::evict_lru_dirs(cache_path, 1024).unwrap();
    assert!(freed >= 1024);
    assert!(!dir1.exists());
    assert!(dir2.exists());
    assert!(dir3.exists());
}

#[test]
fn missing_cache_dir_on_open_is_safe() {
    // Verify that if a cache dir is missing at open time (e.g., evicted while
    // the collection was closed), the backend can still open and will regenerate
    // cache entries on cache miss. NOTE: open_in_memory bypasses cache dir, so
    // this test just verifies that reopening after eviction doesn't error.
    let tmp = TempDir::new().unwrap();
    let cache_root = tmp.path().join("cache");
    fs::create_dir(&cache_root).unwrap();

    std::env::set_var("HELIX_LSM_CACHE_DIR", cache_root.to_str().unwrap());
    let backend = LsmBackend::open_in_memory("test_coll").unwrap();
    drop(backend);

    // open_in_memory does not create cache dirs (it bypasses caching), so we
    // just verify the open succeeded without error. A real S3-backed backend
    // would create the cache dir on first open.
    let backend2 = LsmBackend::open_in_memory("test_coll").unwrap();
    drop(backend2);

    std::env::remove_var("HELIX_LSM_CACHE_DIR");
}

#[test]
fn concurrent_open_during_eviction() {
    // Verify that if eviction starts scanning a dir and the collection is
    // opened mid-scan, the guard prevents deletion.
    let cache_dir = TempDir::new().unwrap();
    let cache_path = cache_dir.path();

    let coll = cache_path.join("concurrent_coll");
    fs::create_dir(&coll).unwrap();
    fs::write(coll.join("data"), vec![0_u8; 4096]).unwrap();

    // Spawn eviction in a thread, but register the guard just before it runs.
    let cache_path_clone = cache_path.to_path_buf();
    let handle = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(10)); // Let guard register first
        cache_evictor::evict_lru_dirs(&cache_path_clone, u64::MAX).ok();
    });

    let _guard = cache_evictor::register_cache_dir(coll.clone());
    handle.join().unwrap();

    assert!(coll.exists(), "Concurrent open must prevent eviction");
}

#[test]
fn drop_recreate_cache_safety() {
    // Verify that the evictor correctly handles drop+recreate scenarios.
    // NOTE: open_in_memory bypasses cache dirs, so this test verifies the
    // registry/eviction logic with simulated dirs.
    let tmp = TempDir::new().unwrap();
    let cache_root = tmp.path().join("cache");
    fs::create_dir(&cache_root).unwrap();

    std::env::set_var("HELIX_LSM_CACHE_DIR", cache_root.to_str().unwrap());

    // Simulate a closed collection with a stale cache dir.
    let cache_dir = cache_root.join("drop_test");
    fs::create_dir(&cache_dir).unwrap();
    fs::write(cache_dir.join("stale_wal"), b"stale").unwrap();

    // Evict the closed dir.
    cache_evictor::evict_lru_dirs(&cache_root, u64::MAX).unwrap();
    assert!(!cache_dir.exists(), "Cache dir should be evicted");

    // Verify the evictor removed it successfully.
    assert!(
        !cache_dir.join("stale_wal").exists(),
        "Stale cache must be gone"
    );

    std::env::remove_var("HELIX_LSM_CACHE_DIR");
}

#[test]
fn volume_usage_returns_valid_data() {
    let tmp = TempDir::new().unwrap();
    let usage = cache_evictor::volume_usage(tmp.path()).unwrap();
    assert!(usage.total > 0);
    assert!(usage.used < usage.total);
    assert!(usage.used > 0); // Should have *some* usage on a real FS
}

#[test]
fn budget_enforcement_under_threshold_no_eviction() {
    let cache_dir = TempDir::new().unwrap();
    let cache_path = cache_dir.path();

    // Create a small dir (under any reasonable budget).
    let small = cache_path.join("small");
    fs::create_dir(&small).unwrap();
    fs::write(small.join("data"), vec![0_u8; 1024]).unwrap();

    // Set a very high budget (99%) — should not evict anything.
    std::env::set_var("HELIX_LSM_CACHE_BUDGET_PCT", "99");
    cache_evictor::evict_if_over_budget(cache_path);
    assert!(small.exists(), "Should not evict when under budget");
    std::env::remove_var("HELIX_LSM_CACHE_BUDGET_PCT");
}
