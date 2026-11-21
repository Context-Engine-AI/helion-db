//! Per-collection cache of `match.text` payload-index walks.
//!
//! A text walk scans the field's whole keyword index (~0.4 s on a 48k-point
//! reader), and the client sends the same path needle on every channel of a hybrid
//! query and again on later queries. A walk is a pure function of the index
//! rows a snapshot sees, so it is keyed by the payload-index handle (identity
//! plus the instant it turned Ready) and the snapshot sequence the read is
//! pinned at: equal sequences of one collection see identical rows. Reads that
//! are not pinned to a sequence (LMDB, latest-state reader reads, write views
//! carrying uncommitted puts) never use the cache.

use std::sync::Arc;
use std::time::Instant;

use lru::LruCache;

/// Env knob: per-collection cache budget in KiB. `0` disables the cache.
const ENV_TEXT_WALK_CACHE_KB: &str = "HELIX_TEXT_WALK_CACHE_KB";
const DEFAULT_TEXT_WALK_CACHE_KB: usize = 4096;
/// Rough fixed cost of an entry beyond its id list and key strings.
const ENTRY_OVERHEAD_BYTES: usize = 128;

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct TextWalkKey {
    field: String,
    /// `None` in the key of the field's mark that a walk of it gave up.
    text: Option<String>,
    /// `Arc::as_ptr` of the handle's state: a rebuilt index is a new handle.
    index: usize,
    ready_since: Instant,
    seq: u64,
}

impl TextWalkKey {
    pub(crate) fn new(
        field: &str,
        text: &str,
        index: usize,
        ready_since: Instant,
        seq: u64,
    ) -> Self {
        Self {
            field: field.to_string(),
            text: Some(text.to_string()),
            index,
            ready_since,
            seq,
        }
    }

    /// The key of the mark that a walk of this field gave up at this
    /// snapshot. A walk reads every row of the field's index whatever the
    /// text, so one that gave up there gives up there for every text.
    pub(crate) fn of_field(&self) -> Self {
        Self {
            text: None,
            ..self.clone()
        }
    }

    fn bytes(&self) -> usize {
        self.field.len() + self.text.as_ref().map_or(0, String::len) + ENTRY_OVERHEAD_BYTES
    }
}

/// A completed (uncapped) walk: its ids, exactness, and how many index rows it
/// visited, so a later caller with a smaller walk cap still gets `None`. Under
/// a field's key ([`TextWalkKey::of_field`]), the mark of a walk that gave
/// up: no ids, and the rows it had visited when it did.
pub(crate) struct TextWalk {
    pub(crate) ids: Vec<u128>,
    pub(crate) exact: bool,
    pub(crate) visited: usize,
}

impl TextWalk {
    fn bytes(&self) -> usize {
        self.ids.len() * std::mem::size_of::<u128>()
    }
}

pub(crate) struct TextWalkCache {
    entries: LruCache<TextWalkKey, Arc<TextWalk>>,
    bytes: usize,
    cap_bytes: usize,
    hits: u64,
}

impl TextWalkCache {
    pub(crate) fn from_env() -> Self {
        let kb = std::env::var(ENV_TEXT_WALK_CACHE_KB)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_TEXT_WALK_CACHE_KB);
        Self::with_cap_bytes(kb.saturating_mul(1024))
    }

    pub(crate) fn with_cap_bytes(cap_bytes: usize) -> Self {
        Self {
            entries: LruCache::unbounded(),
            bytes: 0,
            cap_bytes,
            hits: 0,
        }
    }

    pub(crate) fn get(&mut self, key: &TextWalkKey) -> Option<Arc<TextWalk>> {
        let hit = self.entries.get(key).cloned();
        self.hits += u64::from(hit.is_some());
        metrics::counter!(
            "helix_text_walk_cache_total",
            "outcome" => if hit.is_some() { "hit" } else { "miss" }
        )
        .increment(1);
        hit
    }

    /// The rows a walk of the field had visited when it gave up at this
    /// snapshot, if one did.
    pub(crate) fn gave_up(&mut self, field: &TextWalkKey) -> Option<usize> {
        let visited = self.entries.get(field).map(|walk| walk.visited);
        if visited.is_some() {
            metrics::counter!("helix_text_walk_cache_total", "outcome" => "gave_up").increment(1);
        }
        visited
    }

    pub(crate) fn insert(&mut self, key: TextWalkKey, walk: TextWalk) {
        let size = key.bytes() + walk.bytes();
        if size > self.cap_bytes {
            // Such a walk is made again at every ask, and until it is counted
            // nothing tells it from a miss. A budget of 0 is the cache off.
            if self.cap_bytes > 0 {
                metrics::counter!("helix_text_walk_cache_total", "outcome" => "too_large")
                    .increment(1);
            }
            return;
        }
        let key_bytes = key.bytes();
        if let Some(old) = self.entries.put(key, Arc::new(walk)) {
            self.bytes -= key_bytes + old.bytes();
        }
        self.bytes += size;
        while self.bytes > self.cap_bytes {
            let Some((key, old)) = self.entries.pop_lru() else {
                break;
            };
            self.bytes -= key.bytes() + old.bytes();
            metrics::counter!("helix_text_walk_cache_total", "outcome" => "evicted").increment(1);
        }
    }

    #[cfg(test)]
    pub(crate) fn hits(&self) -> u64 {
        self.hits
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walk(n: usize) -> TextWalk {
        TextWalk {
            ids: (0..n as u128).collect(),
            exact: true,
            visited: n,
        }
    }

    #[test]
    fn keys_differ_by_every_component() {
        let now = Instant::now();
        let base = TextWalkKey::new("metadata.path", "routers/", 1, now, 7);
        let mut cache = TextWalkCache::with_cap_bytes(1 << 20);
        cache.insert(base.clone(), walk(3));
        assert!(cache.get(&base).is_some());
        for other in [
            TextWalkKey::new("metadata.repo", "routers/", 1, now, 7),
            TextWalkKey::new("metadata.path", "routers", 1, now, 7),
            TextWalkKey::new("metadata.path", "routers/", 2, now, 7),
            TextWalkKey::new(
                "metadata.path",
                "routers/",
                1,
                now + std::time::Duration::from_nanos(1),
                7,
            ),
            TextWalkKey::new("metadata.path", "routers/", 1, now, 8),
        ] {
            assert!(cache.get(&other).is_none());
        }
    }

    #[test]
    fn budget_evicts_least_recently_used_and_accounts_bytes() {
        let now = Instant::now();
        let key = |seq| TextWalkKey::new("f", "t", 1, now, seq);
        let entry = key(0).bytes() + walk(100).bytes();
        let mut cache = TextWalkCache::with_cap_bytes(entry * 2);
        cache.insert(key(1), walk(100));
        cache.insert(key(2), walk(100));
        assert!(cache.get(&key(1)).is_some()); // 1 is now most recent
        cache.insert(key(3), walk(100));
        assert!(cache.get(&key(2)).is_none(), "LRU entry evicted");
        assert!(cache.get(&key(1)).is_some() && cache.get(&key(3)).is_some());
        assert_eq!(cache.bytes(), entry * 2);
        cache.insert(key(3), walk(10));
        assert_eq!(cache.bytes(), entry + key(0).bytes() + walk(10).bytes());
    }

    #[test]
    fn oversized_entries_and_zero_budget_are_not_cached() {
        let key = TextWalkKey::new("f", "t", 1, Instant::now(), 1);
        let mut cache = TextWalkCache::with_cap_bytes(0);
        cache.insert(key.clone(), walk(1));
        assert_eq!(cache.len(), 0);
        let mut cache = TextWalkCache::with_cap_bytes(64);
        cache.insert(key, walk(100));
        assert_eq!((cache.len(), cache.bytes()), (0, 0));
    }

    /// A walk too large for the budget is counted, so a page shows the
    /// collections whose broad texts are walked at every ask; the cache
    /// switched off counts none.
    #[test]
    fn a_walk_too_large_to_keep_is_counted() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let key = TextWalkKey::new("f", "t", 1, Instant::now(), 1);
            TextWalkCache::with_cap_bytes(0).insert(key.clone(), walk(100));
            TextWalkCache::with_cap_bytes(1 << 20).insert(key.clone(), walk(100));
            TextWalkCache::with_cap_bytes(64).insert(key, walk(100));
        });
        let page = handle.render();
        assert!(
            page.contains(r#"helix_text_walk_cache_total{outcome="too_large"} 1"#),
            "{page}"
        );
    }
}
