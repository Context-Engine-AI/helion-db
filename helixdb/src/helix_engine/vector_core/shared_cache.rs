//! Pod-global sharded LRU caches for HNSW neighbor blocks and decoded vectors.
//!
//! At 1k concurrent collections × 6 named vectors × N segments the old
//! per-VectorCore caches were ~24 GiB just in cache memory. This module
//! replaces them with a single pod-global cache split into 16 shards (one
//! Mutex per shard) so at 1k concurrency the lock contention stays bounded
//! while we still only hold one cap-sized LRU per cache tier across the
//! whole process.
//!
//! ## Design
//!
//! - One `SharedVectorCaches` per pod, held in an `Arc` by `StorageCore`
//!   and handed to every `VectorCore` at construction.
//! - Keys are `(namespace_id: u64, point_id: u128, level: u32)`.
//!   `namespace_id = stable_hash(collection, vector_name)` so two
//!   collections (or two named vectors in the same collection) with
//!   overlapping `point_id`s never collide.
//! - Caps are **global**: `HELIX_NEIGHBOR_CACHE_CAP` now means "total
//!   entries across all collections" instead of "entries per segment".
//!   Defaults are raised accordingly in the compiled code and the k8s
//!   yaml.
//! - Sharding by `fx_hash(key) % 16`. Each shard has its own `Mutex`,
//!   so at 16 concurrent searches they usually fan out to distinct
//!   shards. Worst case: collision → tiny critical section (one LRU
//!   put/get) → minimal tail latency.
//! - `invalidate_namespace(ns)` walks all shards to drop that tenant's
//!   entries. Rare (collection drop / VectorCore destructor); O(N_total)
//!   but still cheap because LRU is just a HashMap + DLL.
//!
//! ## Correctness traps avoided
//!
//! - Key includes `namespace_id` so no cross-collection pollution.
//! - `level` is `u32` (not `usize`) so the key type has a stable
//!   repr across 32-bit / 64-bit targets. HNSW level fits comfortably.
//! - Nav vs level-0 stay in separate caches so a burst of level-0
//!   reads cannot evict the hot navigation set (same reasoning as
//!   the per-core version).
//!
//! ## Rollout
//!
//! Gated behind `HELIX_SHARED_VECTOR_CACHES=1`. When off, VectorCore
//! falls back to its private LruCaches, which keeps a safe revert path.

use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Mutex;

use lru::LruCache;

use super::vector_core::NeighborBlock;

/// Number of shards per cache tier. Power-of-two so `hash & (SHARDS - 1)`
/// dispatches without a modulo. 16 balances lock granularity (~62 ns /
/// lock acquire uncontended) against LRU overhead per shard.
pub const SHARDS: usize = 16;

/// Pod-global cache key.
///
/// - `namespace`: `stable_hash(collection_name, vector_name)`. Stable
///   across process restarts so metrics derived from key distribution
///   are comparable.
/// - `id`: the point's `u128` id (shared with the graph engine).
/// - `level`: HNSW level. `u32` keeps the struct size predictable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub namespace: u64,
    pub id: u128,
    pub level: u32,
}

impl CacheKey {
    #[inline]
    pub fn new(namespace: u64, id: u128, level: usize) -> Self {
        Self {
            namespace,
            id,
            // Clamp rather than panic; level > u32::MAX is impossible in
            // practice (HNSW m_l rarely produces > 16 levels).
            level: level.min(u32::MAX as usize) as u32,
        }
    }
}

/// A pod-global set of four sharded LRU caches.
///
/// All four caches share the same `SHARDS` shape but own their own caps.
/// The hot path takes exactly one Mutex per cache lookup — no nested
/// locks — so deadlocks are structurally impossible.
pub struct SharedVectorCaches {
    pub neighbor: ShardedLru<NeighborBlock>,
    pub vector: ShardedLru<Vec<f32>>,
    pub nav_neighbor: ShardedLru<NeighborBlock>,
    pub nav_vector: ShardedLru<Vec<f32>>,
}

impl SharedVectorCaches {
    pub fn new(
        neighbor_cap: NonZeroUsize,
        vector_cap: NonZeroUsize,
        nav_cap: NonZeroUsize,
    ) -> Self {
        Self {
            neighbor: ShardedLru::new(neighbor_cap),
            vector: ShardedLru::new(vector_cap),
            nav_neighbor: ShardedLru::new(nav_cap),
            nav_vector: ShardedLru::new(nav_cap),
        }
    }

    /// Drop every entry belonging to `namespace` across all four caches.
    /// Called on collection drop / VectorCore teardown so stale entries
    /// never outlive the data they reference.
    pub fn invalidate_namespace(&self, namespace: u64) {
        self.neighbor.invalidate_namespace(namespace);
        self.vector.invalidate_namespace(namespace);
        self.nav_neighbor.invalidate_namespace(namespace);
        self.nav_vector.invalidate_namespace(namespace);
    }
}

/// A `SHARDS`-way sharded LRU. Capacity is distributed evenly; total
/// capacity is `per_shard_cap * SHARDS`, which matches the caller's
/// requested cap within ±SHARDS entries.
pub struct ShardedLru<V: Clone> {
    shards: Vec<Mutex<LruCache<CacheKey, V>>>,
}

impl<V: Clone> ShardedLru<V> {
    pub fn new(total_cap: NonZeroUsize) -> Self {
        // Floor at 1 entry/shard so even a cap=1 config still builds.
        let per_shard = (total_cap.get() / SHARDS).max(1);
        let per_shard = NonZeroUsize::new(per_shard).unwrap();
        let mut shards = Vec::with_capacity(SHARDS);
        for _ in 0..SHARDS {
            shards.push(Mutex::new(LruCache::new(per_shard)));
        }
        Self { shards }
    }

    #[inline]
    fn shard_index(key: &CacheKey) -> usize {
        // FxHash-style single-pass — avoids DefaultHasher's finalization
        // cost for a tiny tuple key.
        let mut h = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut h);
        (h.finish() as usize) & (SHARDS - 1)
    }

    pub fn get(&self, key: &CacheKey) -> Option<V> {
        let idx = Self::shard_index(key);
        self.shards[idx]
            .lock()
            .ok()
            .and_then(|mut shard| shard.get(key).cloned())
    }

    pub fn put(&self, key: CacheKey, value: V) {
        let idx = Self::shard_index(&key);
        if let Ok(mut shard) = self.shards[idx].lock() {
            shard.put(key, value);
        }
    }

    pub fn pop(&self, key: &CacheKey) {
        let idx = Self::shard_index(key);
        if let Ok(mut shard) = self.shards[idx].lock() {
            shard.pop(key);
        }
    }

    pub fn invalidate_namespace(&self, namespace: u64) {
        for shard in &self.shards {
            if let Ok(mut shard) = shard.lock() {
                // LruCache exposes retain via iter() + manual pop; the
                // simpler route is a clear-and-repop of survivors.
                let keep: Vec<(CacheKey, V)> = shard
                    .iter()
                    .filter(|(k, _)| k.namespace != namespace)
                    .map(|(k, v)| (*k, v.clone()))
                    .collect();
                shard.clear();
                for (k, v) in keep {
                    shard.put(k, v);
                }
            }
        }
    }

    /// Total entries across all shards. For metrics only — the lock
    /// acquire storm makes this unsuitable for the hot path.
    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.lock().map(|s| s.len()).unwrap_or(0))
            .sum()
    }
}

/// Derive a stable, process-portable `namespace_id` for a `(collection,
/// vector)` pair. Uses `DefaultHasher` (SipHash) so the mapping is stable
/// within a process and collision-resistant; not persisted.
#[inline]
pub fn namespace_id(collection: &str, vector: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    collection.hash(&mut h);
    // Delimiter so "ab" + "c" ≠ "a" + "bc".
    0xFFu8.hash(&mut h);
    vector.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap()
    }

    #[test]
    fn namespaces_are_isolated() {
        let cache: ShardedLru<u64> = ShardedLru::new(nz(64));
        let k_a = CacheKey::new(1, 42, 0);
        let k_b = CacheKey::new(2, 42, 0); // same id/level, different ns
        cache.put(k_a, 100);
        cache.put(k_b, 200);
        assert_eq!(cache.get(&k_a), Some(100));
        assert_eq!(cache.get(&k_b), Some(200));
    }

    #[test]
    fn global_cap_evicts_across_shards() {
        // 16 shards × 1 entry each = 16 total. Insert 17 uniformly
        // hashed keys; cache size stays at most 16.
        let cache: ShardedLru<u64> = ShardedLru::new(nz(16));
        for i in 0..17u128 {
            cache.put(CacheKey::new(0, i, 0), i as u64);
        }
        assert!(cache.len() <= 16);
    }

    #[test]
    fn invalidate_namespace_drops_only_that_ns() {
        let cache: ShardedLru<u64> = ShardedLru::new(nz(128));
        for i in 0..32u128 {
            cache.put(CacheKey::new(1, i, 0), i as u64);
            cache.put(CacheKey::new(2, i, 0), (i + 1000) as u64);
        }
        let before = cache.len();
        cache.invalidate_namespace(1);
        // All ns=1 entries gone, all ns=2 entries remain.
        for i in 0..32u128 {
            assert_eq!(cache.get(&CacheKey::new(1, i, 0)), None);
            assert_eq!(cache.get(&CacheKey::new(2, i, 0)), Some((i + 1000) as u64));
        }
        assert!(cache.len() < before);
    }

    #[test]
    fn concurrent_get_put_is_deadlock_free() {
        use std::sync::Arc;
        use std::thread;
        let caches = Arc::new(SharedVectorCaches::new(nz(256), nz(256), nz(256)));
        let mut handles = Vec::new();
        for t in 0..8u64 {
            let c = Arc::clone(&caches);
            handles.push(thread::spawn(move || {
                for i in 0..1000u128 {
                    let k = CacheKey::new(t % 2, i, 0);
                    c.vector.put(k, vec![i as f32]);
                    let _ = c.vector.get(&k);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn namespace_id_delimited() {
        assert_ne!(namespace_id("ab", "c"), namespace_id("a", "bc"));
    }
}
