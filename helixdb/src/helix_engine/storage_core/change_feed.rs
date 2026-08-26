//! Writer-side durable-commit change feed.
//!
//! Reader replicas discover writer progress by polling S3 manifests per
//! resident collection (`DbReader::manifest_poll_interval`) — an O(residents ×
//! readers) LIST/GET floor that dominates S3 request cost while telling the
//! readers almost nothing (most residents are idle at any instant). The writer
//! already knows exactly which collections changed, so it publishes each
//! durable commit here and serves the feed over `GET /internal/changes`
//! (async_gateway fast path). Readers poll THAT (in-cluster HTTP, zero S3) and
//! refresh only the collections that actually changed; the per-collection S3
//! poll becomes a slow safety net.
//!
//! Publish points are DURABLE commits only — `commit` (awaits durability) and
//! the `flush_durable` group-commit barrier — because a `DbReader` can only
//! observe WAL SSTs that reached the object store; signaling on buffered
//! commits would refresh readers before there is anything new to see.
//!
//! The feed is process-local state: `epoch` identifies this writer process, so
//! a restarted writer (cursor reset to 0) is detected by readers as an epoch
//! change and handled as a full resync instead of silently missed updates.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};

/// Snapshot returned to the feed endpoint: the publisher's epoch, the current
/// cursor, and every collection whose latest durable commit is newer than the
/// caller's `since` cursor. Serde derives so the reader-replica feed poller
/// parses the endpoint's JSON straight into this type.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct FeedChanges {
    pub epoch: u64,
    pub cursor: u64,
    pub changes: Vec<String>,
}

pub struct ChangeFeed {
    /// Identifies this writer process. Readers reset their cursor when it
    /// changes (writer restart ⇒ cursor restarts from 0).
    epoch: u64,
    /// Monotonic sequence; bumped once per published durable commit.
    cursor: AtomicU64,
    /// collection name -> cursor value of its latest durable commit. Bounded
    /// by the number of collections written since process start.
    latest: RwLock<HashMap<String, u64>>,
}

impl ChangeFeed {
    fn new() -> Self {
        // Wall-clock nanos: distinct across writer restarts, which is all the
        // epoch has to guarantee (readers only compare for equality).
        let epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(1)
            .max(1);
        Self {
            epoch,
            cursor: AtomicU64::new(0),
            latest: RwLock::new(HashMap::new()),
        }
    }

    /// Record a durable commit for `collection`. Cheap enough for per-commit
    /// call sites: one atomic increment + one map insert.
    pub fn publish(&self, collection: &str) {
        let seq = self.cursor.fetch_add(1, Ordering::Relaxed) + 1;
        if let Ok(mut latest) = self.latest.write() {
            match latest.get_mut(collection) {
                Some(entry) => *entry = seq,
                None => {
                    latest.insert(collection.to_string(), seq);
                }
            }
        }
    }

    /// Collections whose latest durable commit is newer than `since`, plus the
    /// cursor to pass back on the next poll. `since = 0` returns everything
    /// published since process start.
    pub fn changes_since(&self, since: u64) -> FeedChanges {
        let cursor = self.cursor.load(Ordering::Relaxed);
        let changes = if since >= cursor {
            Vec::new()
        } else {
            self.latest
                .read()
                .map(|latest| {
                    latest
                        .iter()
                        .filter(|(_, seq)| **seq > since)
                        .map(|(name, _)| name.clone())
                        .collect()
                })
                .unwrap_or_default()
        };
        FeedChanges {
            epoch: self.epoch,
            cursor,
            changes,
        }
    }
}

/// The process-global feed. Writers publish into it; the gateway's
/// `/internal/changes` fast path reads from it. On reader/LMDB processes it
/// simply stays empty.
pub fn change_feed() -> &'static ChangeFeed {
    static FEED: OnceLock<ChangeFeed> = OnceLock::new();
    FEED.get_or_init(ChangeFeed::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changes_since_returns_only_newer_collections() {
        let feed = ChangeFeed::new();
        feed.publish("a");
        feed.publish("b");
        let first = feed.changes_since(0);
        assert_eq!(first.cursor, 2);
        let mut names = first.changes.clone();
        names.sort();
        assert_eq!(names, vec!["a".to_string(), "b".to_string()]);

        // A cursor at the head sees nothing new.
        assert!(feed.changes_since(first.cursor).changes.is_empty());

        // Republishing one collection surfaces only that collection.
        feed.publish("b");
        let second = feed.changes_since(first.cursor);
        assert_eq!(second.cursor, 3);
        assert_eq!(second.changes, vec!["b".to_string()]);
    }

    #[test]
    fn republish_dedupes_to_latest_seq() {
        let feed = ChangeFeed::new();
        feed.publish("a");
        feed.publish("a");
        feed.publish("a");
        let snap = feed.changes_since(0);
        assert_eq!(snap.cursor, 3);
        assert_eq!(snap.changes, vec!["a".to_string()]);
    }

    #[test]
    fn epoch_is_stable_within_process() {
        let feed = ChangeFeed::new();
        let a = feed.changes_since(0).epoch;
        feed.publish("x");
        assert_eq!(feed.changes_since(0).epoch, a);
        assert!(a >= 1);
    }
}
