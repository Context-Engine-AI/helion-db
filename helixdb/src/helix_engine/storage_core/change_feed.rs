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
//! Beside the commits the feed names the collections whose drop tombstone
//! the writer cleared (`publish_recreated`): created again over the drop, or
//! released by the purge worker. Once the tombstone is gone nothing else
//! tells a reader of the drop.
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
    /// Collections whose drop tombstone the writer cleared since `since`:
    /// created again over the drop, or released by the purge worker
    /// ([`ChangeFeed::publish_recreated`]). Empty from a writer that does not
    /// send the list; a reader that does not know it reads past it.
    #[serde(default)]
    pub recreated: Vec<String>,
}

pub struct ChangeFeed {
    /// Identifies this writer process. Readers reset their cursor when it
    /// changes (writer restart ⇒ cursor restarts from 0).
    epoch: u64,
    /// Monotonic sequence; bumped once per published durable commit and once
    /// per cleared drop tombstone.
    cursor: AtomicU64,
    /// collection name -> cursor value of its latest durable commit. Bounded
    /// by the number of collections written since process start.
    latest: RwLock<HashMap<String, u64>>,
    /// collection name -> cursor value of the last time its drop tombstone
    /// was cleared. Bounded by the number of names dropped.
    recreated: RwLock<HashMap<String, u64>>,
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
            recreated: RwLock::new(HashMap::new()),
        }
    }

    /// Record a durable commit for `collection`. Cheap enough for per-commit
    /// call sites: one atomic increment + one map insert.
    ///
    /// The number is taken under the map's lock, so a poll that reads a
    /// cursor at or past it finds the name. Taken before the lock, a poll
    /// between the two got the cursor without the name and asked past it
    /// from then on: the commit went untold until the collection's next one
    /// or its reader's own poll, 300 s away on the idle tier.
    pub fn publish(&self, collection: &str) {
        if let Ok(mut latest) = self.latest.write() {
            let seq = self.cursor.fetch_add(1, Ordering::Relaxed) + 1;
            match latest.get_mut(collection) {
                Some(entry) => *entry = seq,
                None => {
                    latest.insert(collection.to_string(), seq);
                }
            }
        }
    }

    /// Record that the writer cleared the drop tombstone of `collection`: it
    /// purged the dropped collection's prefix to create the name there
    /// again, or its purge worker released the name. A drop is told to the
    /// readers by its tombstone: a reader whose sweep did not list it while
    /// it stood learns of neither the drop nor the create, and one that
    /// holds a quarantine on the name holds it for the database that is gone.
    ///
    /// Not a commit, so it stays out of `latest`: there is nothing for a
    /// reader to replay. The number is taken under the map's lock, as
    /// `publish` takes its own.
    pub fn publish_recreated(&self, collection: &str) {
        if let Ok(mut recreated) = self.recreated.write() {
            let seq = self.cursor.fetch_add(1, Ordering::Relaxed) + 1;
            recreated.insert(collection.to_string(), seq);
        }
    }

    /// Collections whose latest durable commit is newer than `since`, and
    /// those created again over a drop after it, plus the cursor to pass back
    /// on the next poll. `since = 0` returns everything published since
    /// process start.
    pub fn changes_since(&self, since: u64) -> FeedChanges {
        let cursor = self.cursor.load(Ordering::Relaxed);
        let newer = |published: &RwLock<HashMap<String, u64>>| -> Vec<String> {
            if since >= cursor {
                return Vec::new();
            }
            published
                .read()
                .map(|published| {
                    published
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
            changes: newer(&self.latest),
            recreated: newer(&self.recreated),
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

    /// A collection created again over a drop is named once, to the polls
    /// behind it, and apart from the commits. What the endpoint writes is
    /// what a reader parses, and a reader reads a writer that sends no list.
    #[test]
    fn recreated_names_reach_the_polls_behind_them_and_cross_the_wire() {
        let feed = ChangeFeed::new();
        feed.publish("written");
        let before = feed.changes_since(0);
        assert!(before.recreated.is_empty());

        feed.publish_recreated("again");
        let after = feed.changes_since(before.cursor);
        assert_eq!(after.cursor, before.cursor + 1);
        assert_eq!(after.recreated, vec!["again".to_string()]);
        assert!(after.changes.is_empty());
        assert!(feed.changes_since(after.cursor).recreated.is_empty());

        let wire = serde_json::to_vec(&after).unwrap();
        let read: FeedChanges = serde_json::from_slice(&wire).unwrap();
        assert_eq!(
            (read.epoch, read.cursor, read.recreated),
            (after.epoch, after.cursor, after.recreated)
        );
        let old_writer: FeedChanges =
            serde_json::from_str(r#"{"epoch":7,"cursor":3,"changes":["written"]}"#).unwrap();
        assert_eq!(old_writer.changes, vec!["written".to_string()]);
        assert!(old_writer.recreated.is_empty());
    }

    /// A poller that asks from the cursor of its last reply is told of every
    /// commit, whatever is published beside its polls. With a commit's number
    /// taken before the map's lock, a poll in between read a cursor that was
    /// past a name not yet in the map, and never asked for it again.
    #[test]
    fn poll_beside_publishers_misses_no_commit() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;
        const PUBLISHERS: usize = 4;
        const EACH: usize = 1_000;

        let feed = Arc::new(ChangeFeed::new());
        let published = Arc::new(AtomicBool::new(false));
        let poller = {
            let (feed, published) = (Arc::clone(&feed), Arc::clone(&published));
            std::thread::spawn(move || {
                let mut told = std::collections::HashSet::new();
                let mut since = 0;
                loop {
                    // Read before the poll: the last poll starts after the
                    // last commit.
                    let last = published.load(Ordering::Acquire);
                    let reply = feed.changes_since(since);
                    since = reply.cursor;
                    told.extend(reply.changes);
                    if last {
                        return told;
                    }
                }
            })
        };
        let publishers: Vec<_> = (0..PUBLISHERS)
            .map(|publisher| {
                let feed = Arc::clone(&feed);
                std::thread::spawn(move || {
                    for commit in 0..EACH {
                        feed.publish(&format!("{publisher}_{commit}"));
                    }
                })
            })
            .collect();
        publishers
            .into_iter()
            .for_each(|publisher| publisher.join().unwrap());
        published.store(true, Ordering::Release);

        let told = poller.join().unwrap();
        assert_eq!(
            told.len(),
            PUBLISHERS * EACH,
            "commits the poller was never told of"
        );
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
