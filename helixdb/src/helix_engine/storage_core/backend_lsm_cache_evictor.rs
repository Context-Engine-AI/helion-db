//! Cache directory eviction to bound total `/cache` disk usage.
//!
//! Each opened collection gets a cache dir at `<HELIX_LSM_CACHE_DIR>/<sanitized_path>/`.
//! Per-collection cache size is capped, but the *sum* across all collections is unbounded:
//! collections that are closed (not dropped) leave their cache dirs behind forever, and
//! with 890+ collections in production (HELIX_MAX_OPEN_COLLECTIONS=32, so ~850 closed),
//! `/cache` grows until the volume is full.
//!
//! This module tracks which collection cache dirs are currently open (process-wide refcount
//! registry) and runs a periodic evictor that:
//! 1. Checks total /cache usage via `statvfs`
//! 2. If usage > budget (default 70% of volume), sorts closed dirs by mtime (LRU first)
//! 3. Sizes and deletes one at a time until under budget
//! 4. Never evicts dirs for collections that are currently open in this process
//!
//! **LRU tracking**: The top-level `/cache/prod_x` dir's mtime is touched on every
//! open and close, and on every sweep while open, so mtime = last use time (survives
//! restarts, including a kill that never runs the close-time touch).
//!
//! **Concurrent open safety**: Each deletion is atomic w.r.t. registration: under the
//! registry lock, re-check the dir is not open and rename to `.evicting-<name>-<nanos>`;
//! release the lock; then remove_dir_all the renamed path. A collection opened during
//! the walk cannot have its live cache dir deleted.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use tokio::runtime::Handle;
use tracing::{debug, info, warn};

/// Env knob: target /cache volume usage as a percentage (0-100). When usage exceeds
/// this, the evictor deletes least-recently-used closed collection cache dirs until
/// usage is [`EVICT_HYSTERESIS_PCT`] points under budget. Default 70% leaves headroom
/// for bursty writes.
const ENV_CACHE_BUDGET_PCT: &str = "HELIX_LSM_CACHE_BUDGET_PCT";
const DEFAULT_CACHE_BUDGET_PCT: u8 = 70;
/// Sweeps that trigger evict down to `budget - EVICT_HYSTERESIS_PCT`.
const EVICT_HYSTERESIS_PCT: u8 = 5;

/// Env knob: cache eviction sweep interval in seconds. The evictor runs once at
/// startup (background, non-blocking) and then every N seconds. Default 300s (5 min).
/// Set 0 to disable periodic sweeps after startup.
const ENV_CACHE_EVICT_INTERVAL_SECS: &str = "HELIX_LSM_CACHE_EVICT_INTERVAL_SECS";
const DEFAULT_CACHE_EVICT_INTERVAL_SECS: u64 = 300;

/// Process-global refcount registry of currently-open collection cache directories.
/// Key = cache dir path, Value = open handle count. When count reaches 0, the dir
/// is eligible for eviction. Handles overlapping opens (writer reopen before deferred
/// close, LsmBackend + LsmReader on same path).
static OPEN_CACHE_DIRS: std::sync::LazyLock<Mutex<HashMap<PathBuf, usize>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// Register a collection cache dir as open. Must be called BEFORE opening the Db/DbReader
/// (so the guard exists during the build). Returns a `CacheDirGuard` that unregisters on drop.
/// Increments the refcount; the dir is only evictable when refcount reaches 0.
pub fn register_cache_dir(dir: PathBuf) -> CacheDirGuard {
    {
        let mut registry = OPEN_CACHE_DIRS.lock();
        *registry.entry(dir.clone()).or_insert(0) += 1;
    }
    // Touch mtime on open so LRU = last use time.
    touch_dir_mtime(&dir);
    CacheDirGuard { dir }
}

/// Whether any writer or reader handle in this process has `dir` registered.
pub(crate) fn cache_dir_is_open(dir: &Path) -> bool {
    OPEN_CACHE_DIRS.lock().contains_key(dir)
}

/// Best-effort touch of a directory's mtime to `now`, so the evictor's LRU ordering
/// reflects last use time (not creation time). Errors are logged but not fatal.
pub fn touch_dir_mtime(dir: &Path) {
    if let Ok(file) = std::fs::File::open(dir) {
        if let Err(e) = file.set_modified(SystemTime::now()) {
            debug!(dir = %dir.display(), error = %e, "Failed to touch cache dir mtime");
        }
    }
}

/// RAII guard that unregisters a cache dir from the open registry on drop.
/// Held by `LsmBackend` and `LsmReader` so the cache dir is unregistered
/// when the handle is dropped. Decrements refcount; dir becomes evictable when count == 0.
pub struct CacheDirGuard {
    dir: PathBuf,
}

impl Drop for CacheDirGuard {
    fn drop(&mut self) {
        // Touch mtime on close so LRU = last use time.
        touch_dir_mtime(&self.dir);

        let mut registry = OPEN_CACHE_DIRS.lock();
        if let Some(count) = registry.get_mut(&self.dir) {
            *count -= 1;
            if *count == 0 {
                registry.remove(&self.dir);
            }
        }
    }
}

/// Start the background cache evictor task on the shared LSM runtime.
/// Runs once immediately (non-blocking) and then every `ENV_CACHE_EVICT_INTERVAL_SECS`.
/// Does nothing if `cache_root` is `None` (caching disabled).
pub(crate) fn start_cache_evictor(rt: &Handle, cache_root: Option<PathBuf>) {
    let Some(root) = cache_root else {
        return;
    };
    let interval_secs = std::env::var(ENV_CACHE_EVICT_INTERVAL_SECS)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_CACHE_EVICT_INTERVAL_SECS);

    rt.spawn(async move {
        // Immediate first run (background, non-blocking for readiness).
        tokio::task::spawn_blocking({
            let root = root.clone();
            move || evict_if_over_budget(&root)
        })
        .await
        .ok();

        if interval_secs == 0 {
            debug!("Cache evictor: periodic sweeps disabled (interval=0)");
            return;
        }

        let tick = Duration::from_secs(interval_secs);
        loop {
            tokio::time::sleep(tick).await;
            let root = root.clone();
            if let Err(e) = tokio::task::spawn_blocking(move || evict_if_over_budget(&root)).await {
                warn!(error = %e, "Cache evictor task panicked");
            }
        }
    });
}

/// Touch every currently-open cache dir. A guard that never drops (SIGKILL, exit
/// without teardown) only leaves its open-time touch, which would make a hot dir held
/// open for days look LRU-oldest to the next process's startup sweep.
fn touch_open_dirs() {
    let open: Vec<PathBuf> = OPEN_CACHE_DIRS.lock().keys().cloned().collect();
    for dir in open {
        touch_dir_mtime(&dir);
    }
}

/// [`ENV_CACHE_BUDGET_PCT`], or the default when it is unset or not a percentage.
fn cache_budget_pct() -> u8 {
    std::env::var(ENV_CACHE_BUDGET_PCT)
        .ok()
        .and_then(|s| s.parse::<u8>().ok())
        .filter(|&p| p <= 100)
        .unwrap_or(DEFAULT_CACHE_BUDGET_PCT)
}

/// The largest per-collection cache share at which `max_open` open collections,
/// each with its directory at the full share, stay within the budget this
/// evictor holds the volume of `cache_root` to. The evictor deletes closed
/// collections' directories only, so a larger share lets the open ones fill the
/// volume, and a full volume is silent: a cache write that fails is dropped and
/// the reads go to the object store. `None` when the volume cannot be read or
/// the budget leaves nothing.
pub(crate) fn share_the_volume_holds(cache_root: &Path, max_open: usize) -> Option<usize> {
    let usage = volume_usage(cache_root).ok()?;
    share_within_budget(usage.total, cache_budget_pct(), max_open)
}

fn share_within_budget(volume_total: u64, budget_pct: u8, max_open: usize) -> Option<usize> {
    let share = volume_total as u128 * budget_pct as u128 / 100 / max_open.max(1) as u128;
    (share > 0).then(|| usize::try_from(share).unwrap_or(usize::MAX))
}

/// Check /cache volume usage and evict LRU closed dirs if over budget.
pub fn evict_if_over_budget(cache_root: &Path) {
    touch_open_dirs();
    let budget_pct = cache_budget_pct();

    let usage = match volume_usage(cache_root) {
        Ok(u) => u,
        Err(e) => {
            warn!(cache_root = %cache_root.display(), error = %e, "Failed to stat cache volume");
            return;
        }
    };

    if usage.total == 0 {
        warn!(cache_root = %cache_root.display(), "Volume total == 0, skipping eviction");
        return;
    }

    let usage_pct = (usage.used * 100) / usage.total;
    if usage_pct <= budget_pct as u64 {
        debug!(
            cache_root = %cache_root.display(),
            used_gb = usage.used / (1 << 30),
            total_gb = usage.total / (1 << 30),
            usage_pct,
            budget_pct,
            "Cache usage under budget"
        );
        return;
    }

    info!(
        cache_root = %cache_root.display(),
        used_gb = usage.used / (1 << 30),
        total_gb = usage.total / (1 << 30),
        usage_pct,
        budget_pct,
        "Cache usage over budget; starting eviction"
    );

    // Evict a few points below the trigger so a volume hovering at the budget does not
    // pay a full sweep every interval.
    let target_bytes = (usage.total * budget_pct.saturating_sub(EVICT_HYSTERESIS_PCT) as u64) / 100;
    let to_free = usage.used.saturating_sub(target_bytes);

    match evict_lru_dirs(cache_root, to_free) {
        Ok(freed) => {
            info!(
                cache_root = %cache_root.display(),
                freed_gb = freed / (1 << 30),
                "Cache eviction completed"
            );
            metrics::counter!("helix_lsm_cache_eviction_total", "outcome" => "success")
                .increment(1);
            metrics::counter!("helix_lsm_cache_eviction_bytes_freed_total").increment(freed);
        }
        Err(e) => {
            warn!(
                cache_root = %cache_root.display(),
                error = %e,
                "Cache eviction failed"
            );
            metrics::counter!("helix_lsm_cache_eviction_total", "outcome" => "error").increment(1);
        }
    }
}

/// Volume usage statistics from `statvfs`.
pub struct VolumeUsage {
    pub used: u64,
    pub total: u64,
}

/// Query volume usage for the filesystem containing `path`.
pub fn volume_usage(path: &Path) -> std::io::Result<VolumeUsage> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let path_cstr = std::ffi::CString::new(path.as_os_str().as_bytes())?;
        let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statvfs(path_cstr.as_ptr(), &mut stat) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        let block_size = stat.f_frsize as u64;
        let total = (stat.f_blocks as u64) * block_size;
        let avail = (stat.f_bavail as u64) * block_size;
        let used = total.saturating_sub(avail);
        Ok(VolumeUsage { used, total })
    }
    #[cfg(not(unix))]
    {
        // Non-Unix: return a dummy "under budget" result so eviction is a no-op.
        // Production runs on Linux; this branch is for test/dev on other platforms.
        let _ = path;
        Ok(VolumeUsage {
            used: 0,
            total: u64::MAX,
        })
    }
}

/// Evict least-recently-used closed collection cache dirs until `to_free` bytes
/// are reclaimed. Returns the total bytes freed.
///
/// Sorts candidates by mtime (oldest first), then sizes and deletes one at a time.
/// Each deletion is atomic w.r.t. concurrent opens: under the registry lock, re-check
/// the dir is not open and rename to `.evicting-<name>-<nanos>`; release lock; then
/// remove_dir_all the renamed path. Leftover `.evicting-*` dirs (from crashes) are
/// removed on the next sweep.
pub fn evict_lru_dirs(cache_root: &Path, to_free: u64) -> std::io::Result<u64> {
    // Collect (dir, mtime) for every subdir (including leftover `.evicting-*` from crashes).
    let mut candidates: Vec<(PathBuf, SystemTime)> = Vec::new();
    for entry in std::fs::read_dir(cache_root)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let mtime = match newest_dir_mtime(&path) {
            Ok(m) => m,
            Err(e) => {
                warn!(dir = %path.display(), error = %e, "Failed to stat cache dir; skipping");
                continue;
            }
        };
        candidates.push((path, mtime));
    }

    // Sort by mtime ascending (oldest first = LRU).
    candidates.sort_by_key(|(_, mtime)| *mtime);

    let mut freed = 0_u64;
    for (dir, _mtime) in candidates {
        if freed >= to_free {
            break;
        }

        // Atomic check-and-rename under registry lock. Leftover `.evicting-*` dirs are
        // never registered, so delete them in place instead of renaming them again.
        let leftover = dir
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with(".evicting-"));
        let renamed = if leftover {
            Some(dir.clone())
        } else {
            let registry = OPEN_CACHE_DIRS.lock();
            if registry.contains_key(&dir) {
                debug!(dir = %dir.display(), "Skipping open cache dir");
                continue;
            }

            // Rename to `.evicting-<name>-<nanos>` while holding the lock, so a concurrent
            // open cannot register this path after we checked.
            let name = dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");
            let nanos = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let renamed = dir.with_file_name(format!(".evicting-{}-{}", name, nanos));
            match std::fs::rename(&dir, &renamed) {
                Ok(()) => Some(renamed),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    // Already gone (concurrent eviction or manual delete).
                    None
                }
                Err(e) => {
                    warn!(dir = %dir.display(), error = %e, "Failed to rename cache dir for eviction");
                    metrics::counter!("helix_lsm_cache_eviction_errors_total").increment(1);
                    None
                }
            }
        }; // Lock released here

        let Some(renamed_path) = renamed else {
            continue;
        };

        // Size the dir now (after rename, so we don't race with writes).
        let size = dir_size(&renamed_path).unwrap_or(0);

        // Remove the renamed dir (no lock held).
        debug!(dir = %renamed_path.display(), size_mb = size / (1 << 20), "Evicting cache dir");
        match std::fs::remove_dir_all(&renamed_path) {
            Ok(()) => {
                freed += size;
                metrics::counter!("helix_lsm_cache_dirs_evicted_total").increment(1);
            }
            Err(e) => {
                warn!(dir = %renamed_path.display(), error = %e, "Failed to remove cache dir");
                metrics::counter!("helix_lsm_cache_eviction_errors_total").increment(1);
            }
        }
    }

    Ok(freed)
}

/// Newest mtime of `dir` and the directories down to `<dir>/prod/<name>/compacted`.
/// The top-level dir only holds `prod/`, so its own mtime is the creation (or last
/// open/close/sweep touch) time; `compacted/` gains an entry whenever an SST gets its
/// first cached part, which tracks real use for dirs that predate the touch-on-open/close
/// bookkeeping. Stops above the per-SST dirs: prod writers hold ~666k of them, and
/// stat-ing each one made the first sweep minutes of EBS inode reads.
fn newest_dir_mtime(dir: &Path) -> std::io::Result<SystemTime> {
    newest_dir_mtime_to_depth(dir, 3)
}

fn newest_dir_mtime_to_depth(dir: &Path, depth: u32) -> std::io::Result<SystemTime> {
    let mut newest = std::fs::metadata(dir)?
        .modified()
        .unwrap_or(SystemTime::UNIX_EPOCH);
    if depth == 0 {
        return Ok(newest);
    }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if entry.file_type().is_ok_and(|t| t.is_dir()) {
                if let Ok(m) = newest_dir_mtime_to_depth(&entry.path(), depth - 1) {
                    newest = newest.max(m);
                }
            }
        }
    }
    Ok(newest)
}

/// Recursively compute the total size of a directory in bytes.
fn dir_size(path: &Path) -> std::io::Result<u64> {
    fn walk_dir(path: &Path, total: &mut u64) -> std::io::Result<()> {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if metadata.is_file() {
                *total += metadata.len();
            } else if metadata.is_dir() {
                walk_dir(&entry.path(), total)?;
            }
        }
        Ok(())
    }

    let mut total = 0_u64;
    walk_dir(path, &mut total)?;
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn register_unregister_cycle() {
        let dir = PathBuf::from("/fake/cache/dir");
        assert!(!OPEN_CACHE_DIRS.lock().contains_key(&dir));

        let guard = register_cache_dir(dir.clone());
        assert_eq!(*OPEN_CACHE_DIRS.lock().get(&dir).unwrap(), 1);

        drop(guard);
        assert!(!OPEN_CACHE_DIRS.lock().contains_key(&dir));
    }

    #[test]
    fn refcount_overlapping_handles() {
        let dir = PathBuf::from("/fake/cache/dir2");

        let guard1 = register_cache_dir(dir.clone());
        assert_eq!(*OPEN_CACHE_DIRS.lock().get(&dir).unwrap(), 1);

        let guard2 = register_cache_dir(dir.clone());
        assert_eq!(*OPEN_CACHE_DIRS.lock().get(&dir).unwrap(), 2);

        drop(guard1);
        assert_eq!(*OPEN_CACHE_DIRS.lock().get(&dir).unwrap(), 1);

        drop(guard2);
        assert!(!OPEN_CACHE_DIRS.lock().contains_key(&dir));
    }

    fn set_dir_mtime(dir: &Path, secs_ago: u64) {
        fs::File::open(dir)
            .unwrap()
            .set_modified(SystemTime::now() - Duration::from_secs(secs_ago))
            .unwrap();
    }

    #[test]
    fn lru_uses_newest_subtree_mtime_not_top_dir_creation() {
        // Pre-upgrade layout: top-level mtimes are creation times. `hot` was created long
        // ago but cached a new SST recently (compacted/ gained an entry); `cold` was
        // created later and never used since. Per-SST dir mtimes are not read.
        let tmp = TempDir::new().unwrap();
        for (name, top_ago, compacted_ago) in [("hot", 9_000, 10), ("cold", 5_000, 5_000)] {
            let sst = tmp
                .path()
                .join(name)
                .join("prod")
                .join(name)
                .join("compacted")
                .join("s1");
            fs::create_dir_all(&sst).unwrap();
            fs::write(sst.join("part"), vec![0_u8; 1024]).unwrap();
            set_dir_mtime(&sst, 1); // newest of all, but below the walk depth
            set_dir_mtime(sst.parent().unwrap(), compacted_ago);
            for up in [2, 3] {
                set_dir_mtime(sst.ancestors().nth(up).unwrap(), top_ago);
            }
            set_dir_mtime(&tmp.path().join(name), top_ago);
        }
        assert!(
            newest_dir_mtime(&tmp.path().join("cold")).unwrap()
                < SystemTime::now() - Duration::from_secs(1_000),
            "per-SST dirs must not be read"
        );
        evict_lru_dirs(tmp.path(), 1).unwrap();
        assert!(
            tmp.path().join("hot").exists(),
            "recently used subtree must survive"
        );
        assert!(
            !tmp.path().join("cold").exists(),
            "least recently used subtree goes first"
        );
    }

    #[test]
    fn leftover_evicting_dir_is_not_renamed_again_when_removal_fails() {
        use std::os::unix::fs::PermissionsExt;
        if unsafe { libc::geteuid() } == 0 {
            return; // root ignores the permission bits this test relies on
        }
        let tmp = TempDir::new().unwrap();
        let leftover = tmp.path().join(".evicting-prod_x-123");
        let stuck = leftover.join("stuck");
        fs::create_dir_all(&stuck).unwrap();
        fs::write(stuck.join("part"), vec![0_u8; 1024]).unwrap();
        fs::set_permissions(&stuck, fs::Permissions::from_mode(0o500)).unwrap();
        evict_lru_dirs(tmp.path(), u64::MAX).unwrap();
        let names: Vec<_> = fs::read_dir(tmp.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        fs::set_permissions(&stuck, fs::Permissions::from_mode(0o700)).unwrap();
        assert_eq!(
            names,
            vec![std::ffi::OsString::from(".evicting-prod_x-123")]
        );
        evict_lru_dirs(tmp.path(), u64::MAX).unwrap();
        assert_eq!(
            fs::read_dir(tmp.path()).unwrap().count(),
            0,
            "removed once deletable"
        );
    }

    #[test]
    fn volume_usage_returns_sensible_values() {
        let tmp = TempDir::new().unwrap();
        let usage = volume_usage(tmp.path()).unwrap();
        assert!(usage.total > 0);
        assert!(usage.used < usage.total);
    }

    #[test]
    fn evict_respects_open_registry() {
        let tmp = TempDir::new().unwrap();
        let open_dir = tmp.path().join("open_coll");
        let closed_dir = tmp.path().join("closed_coll");
        fs::create_dir(&open_dir).unwrap();
        fs::create_dir(&closed_dir).unwrap();
        fs::write(closed_dir.join("data"), vec![0_u8; 1024]).unwrap();

        let _guard = register_cache_dir(open_dir.clone());

        let freed = evict_lru_dirs(tmp.path(), 1024 * 1024).unwrap();
        assert!(freed >= 1024); // closed_dir freed
        assert!(open_dir.exists(), "open dir must not be evicted");
        assert!(!closed_dir.exists(), "closed dir should be evicted");
    }

    #[test]
    fn evict_lru_order() {
        let tmp = TempDir::new().unwrap();
        let old_dir = tmp.path().join("old");
        let new_dir = tmp.path().join("new");
        fs::create_dir(&old_dir).unwrap();
        std::thread::sleep(Duration::from_millis(10));
        fs::create_dir(&new_dir).unwrap();
        fs::write(old_dir.join("data"), vec![0_u8; 1024]).unwrap();
        fs::write(new_dir.join("data"), vec![0_u8; 1024]).unwrap();

        // Touch new_dir to make it more recent.
        touch_dir_mtime(&new_dir);

        // Less than one dir's worth, so exactly one (the LRU one) goes.
        let freed = evict_lru_dirs(tmp.path(), 1000).unwrap();
        assert!(freed >= 1024);
        // Old dir evicted first.
        assert!(!old_dir.exists());
        assert!(new_dir.exists());
    }

    #[test]
    fn evict_stops_at_target() {
        let tmp = TempDir::new().unwrap();
        let dir1 = tmp.path().join("dir1");
        let dir2 = tmp.path().join("dir2");
        fs::create_dir(&dir1).unwrap();
        fs::write(dir1.join("data"), vec![0_u8; 1024]).unwrap();
        // dir1's last write must be older than dir2's by more than the
        // filesystem's timestamp step, or the two tie for least recent.
        std::thread::sleep(Duration::from_millis(20));
        fs::create_dir(&dir2).unwrap();
        fs::write(dir2.join("data"), vec![0_u8; 1024]).unwrap();

        let freed = evict_lru_dirs(tmp.path(), 1024).unwrap();
        assert!(freed >= 1024);
        assert!(!dir1.exists());
        // Should stop after dir1 since target is met.
        assert!(dir2.exists());
    }

    /// The readers' figures of 2026-11-20: 64 handles on a volume of 97.87 GiB
    /// (a 100Gi claim) hold a 1 GiB share and not a 2 GiB one; the same claim
    /// grown to 196.30 GiB holds 2 GiB.
    #[test]
    fn the_share_a_volume_holds_is_its_budget_split_over_the_handles() {
        const GIB: u64 = 1 << 30;
        let small = 97 * GIB + 87 * GIB / 100;
        let grown = 196 * GIB + 30 * GIB / 100;
        let held = share_within_budget(small, 70, 64).unwrap() as u64;
        assert_eq!(held, small * 70 / 100 / 64);
        assert!((GIB..2 * GIB).contains(&held), "{held}");
        assert!(share_within_budget(grown, 70, 64).unwrap() as u64 >= 2 * GIB);
        // No budget or no volume is no bound, and no handle limit counts as one.
        assert_eq!(share_within_budget(small, 0, 64), None);
        assert_eq!(share_within_budget(0, 70, 64), None);
        assert_eq!(
            share_within_budget(small, 70, 0),
            share_within_budget(small, 70, 1)
        );
        // The stand-in total of a platform without statvfs does not overflow.
        assert_eq!(share_within_budget(u64::MAX, 100, 1), Some(usize::MAX));
    }

    struct EnvVar(&'static str, Option<String>);
    impl EnvVar {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self(key, prev)
        }
    }
    impl Drop for EnvVar {
        fn drop(&mut self) {
            match &self.1 {
                Some(v) => std::env::set_var(self.0, v),
                None => std::env::remove_var(self.0),
            }
        }
    }

    fn files_under(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).unwrap().flatten() {
            if entry.file_type().unwrap().is_dir() {
                files_under(&entry.path(), out);
            } else {
                out.push(entry.path());
            }
        }
    }

    /// Real SlateDB FS cache under a prod-shaped `<prefix>/<name>` LSM path: every cached
    /// file sits at `<subtree>/prod/<name>/compacted/<sst>/<file>`, so walk depth 3 is
    /// `compacted/` itself, the per-SST level is never listed, and a newly cached SST
    /// moves `compacted/`'s mtime.
    #[test]
    #[serial_test::serial]
    fn real_cache_layout_puts_compacted_at_walk_depth_and_new_ssts_bump_it() {
        use crate::helix_engine::storage_core::backend::{Namespace, StorageBackend};
        use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
        use slatedb::object_store::{memory::InMemory, ObjectStore};
        use std::sync::Arc;

        let cache = TempDir::new().unwrap();
        let _dir = EnvVar::set("HELIX_LSM_CACHE_DIR", cache.path().to_str().unwrap());
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let write = |key: &[u8]| {
            let be = LsmBackend::open_with_store("prod/depthcheck", store.clone()).unwrap();
            let mut w = be.begin_write().unwrap();
            be.put(&mut w, Namespace::Nodes, key, b"v").unwrap();
            be.commit(w).unwrap();
            be.close().unwrap();
        };
        write(b"k1");

        let subtree = cache.path().join("prod_depthcheck");
        let compacted = subtree.join("prod").join("depthcheck").join("compacted");
        let mut files = Vec::new();
        files_under(&subtree, &mut files);
        assert!(
            !files.is_empty(),
            "close-time L0 flush must populate the cache"
        );
        for f in &files {
            let rel = f.strip_prefix(&subtree).unwrap();
            assert_eq!(
                rel.components().count(),
                5,
                "unexpected cache layout: {rel:?}"
            );
            assert!(
                f.starts_with(&compacted),
                "cached outside compacted/: {rel:?}"
            );
        }
        let sst_dirs = |c: &Path| fs::read_dir(c).unwrap().count();
        let ssts_before = sst_dirs(&compacted);

        // Everything at or above compacted/ is old; the per-SST dirs stay fresh. The walk
        // must report the old time, i.e. never stat the per-SST level.
        for d in [
            &compacted,
            &subtree.join("prod/depthcheck"),
            &subtree.join("prod"),
            &subtree,
        ] {
            set_dir_mtime(d, 10_000);
        }
        assert!(
            newest_dir_mtime(&subtree).unwrap() < SystemTime::now() - Duration::from_secs(5_000),
            "walk read below compacted/ on the real layout"
        );

        write(b"k2");
        assert!(
            sst_dirs(&compacted) > ssts_before,
            "second flush must cache a new SST"
        );
        assert!(
            fs::metadata(&compacted).unwrap().modified().unwrap()
                > SystemTime::now() - Duration::from_secs(600),
            "a newly cached SST must bump compacted/ mtime"
        );
    }

    /// Budget below the hysteresis gap saturates the target to 0: one finite pass that
    /// evicts every closed dir and never an open one. Budget 100 never triggers.
    #[test]
    #[serial_test::serial]
    fn hysteresis_saturates_for_tiny_budget_and_never_evicts_open() {
        let tmp = TempDir::new().unwrap();
        let usage = volume_usage(tmp.path()).unwrap();
        if usage.total == 0 || usage.used * 100 / usage.total <= 3 {
            return; // host volume too empty to cross a 3% budget
        }
        let open_dir = tmp.path().join("open");
        let closed_dir = tmp.path().join("closed");
        for d in [&open_dir, &closed_dir] {
            fs::create_dir(d).unwrap();
            fs::write(d.join("part"), vec![0_u8; 1024]).unwrap();
        }
        let _guard = register_cache_dir(open_dir.clone());

        {
            let _budget = EnvVar::set(ENV_CACHE_BUDGET_PCT, "100");
            evict_if_over_budget(tmp.path());
            assert!(closed_dir.exists(), "budget 100 must never trigger");
        }
        let _budget = EnvVar::set(ENV_CACHE_BUDGET_PCT, "3");
        evict_if_over_budget(tmp.path());
        assert!(open_dir.exists(), "open dir must never be evicted");
        assert!(!closed_dir.exists(), "target 0 evicts every closed dir");
    }

    /// `cargo test -p helixdb --lib sweep_walk_bench -- --ignored --nocapture`
    /// (`SWEEP_BENCH_COLLS` / `SWEEP_BENCH_SSTS` override 200 x 2000).
    #[test]
    #[ignore = "bench"]
    fn sweep_walk_bench() {
        fn full_walk(dir: &Path, visited: &mut u64) -> SystemTime {
            *visited += 1;
            let mut newest = fs::metadata(dir).unwrap().modified().unwrap();
            for entry in fs::read_dir(dir).unwrap().flatten() {
                if entry.file_type().is_ok_and(|t| t.is_dir()) {
                    newest = newest.max(full_walk(&entry.path(), visited));
                }
            }
            newest
        }
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let (colls, ssts) = (env("SWEEP_BENCH_COLLS", 200), env("SWEEP_BENCH_SSTS", 2000));
        let tmp = TempDir::new().unwrap();
        let t = std::time::Instant::now();
        for c in 0..colls {
            let compacted = tmp.path().join(format!("prod_c{c}/prod/c{c}/compacted"));
            fs::create_dir_all(&compacted).unwrap();
            for s in 0..ssts {
                let sst = compacted.join(format!("{s:026}.sst"));
                fs::create_dir(&sst).unwrap();
                fs::write(sst.join("_part1mb-000000000"), b"").unwrap();
            }
        }
        println!("setup {colls}x{ssts}: {:?}", t.elapsed());
        for round in 0..2 {
            let t = std::time::Instant::now();
            let mut visited = 0;
            for entry in fs::read_dir(tmp.path()).unwrap().flatten() {
                full_walk(&entry.path(), &mut visited);
            }
            let old = t.elapsed();
            let t = std::time::Instant::now();
            evict_lru_dirs(tmp.path(), 0).unwrap();
            let new = t.elapsed();
            println!(
                "round {round}: full walk {old:?} ({visited} dirs) | depth-3 walk {new:?} | {:.0}x",
                old.as_secs_f64() / new.as_secs_f64()
            );
        }
    }
}
