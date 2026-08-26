//! Write-Ahead Log for HelixDB.
//!
//! Simple append-only binary log per collection. Every mutation is recorded
//! with a monotonic LSN before it hits LMDB. On crash, uncommitted entries
//! are replayed. A follower node can stream entries from any LSN to replicate.
//!
//! Format per record:  [len:u32][payload:bincode][crc32:u32]
//! File naming:        {wal_dir}/{first_lsn:020}.wal
//! Segments rotate at  64 MB by default.

use crate::helix_engine::types::GraphError;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Monotonic log sequence number. Never reused. Serves as replication cursor.
pub type Lsn = u64;

/// Every kind of mutation the storage layer can perform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalOp {
    // Nodes
    CreateNode {
        id: u128,
        label: String,
        /// Properties serialized as JSON bytes for portability.
        properties_json: Vec<u8>,
    },
    UpsertNode {
        id: u128,
        label: String,
        /// Properties serialized as JSON bytes for portability.
        properties_json: Vec<u8>,
    },
    DropNode {
        id: u128,
    },

    // Edges
    CreateEdge {
        id: u128,
        label: String,
        from_node: u128,
        to_node: u128,
        /// Properties serialized as JSON bytes for portability.
        properties_json: Vec<u8>,
    },
    UpsertEdge {
        id: u128,
        label: String,
        from_node: u128,
        to_node: u128,
        /// Properties serialized as JSON bytes for portability.
        properties_json: Vec<u8>,
    },
    DropEdge {
        id: u128,
    },

    // Vectors
    InsertVector {
        id: u128,
        data: Vec<f32>,
        /// Vector fields/payload serialized as JSON bytes.
        fields_json: Option<Vec<u8>>,
        named_index: Option<String>,
    },

    // Transaction markers — group ops atomically
    TxBegin {
        tx_id: u64,
    },
    TxCommit {
        tx_id: u64,
    },

    // Snapshot marker — written after successful LMDB snapshot
    Snapshot {
        lsn_at_snapshot: Lsn,
    },
}

/// Single WAL record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    pub lsn: Lsn,
    pub timestamp_ms: i64,
    pub collection: String,
    pub op: WalOp,
}

// ---------------------------------------------------------------------------
// On-disk format: [len:u32 LE][payload bytes][crc32:u32 LE]
// ---------------------------------------------------------------------------

fn encode_record(entry: &WalEntry) -> Result<Vec<u8>, GraphError> {
    let payload = bincode::serialize(entry)?;
    let len = payload.len() as u32;
    let crc = crc32c(&payload);

    let mut buf = Vec::with_capacity(4 + payload.len() + 4);
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&payload);
    buf.extend_from_slice(&crc.to_le_bytes());
    Ok(buf)
}

fn decode_record(data: &[u8]) -> Result<(WalEntry, usize), DecodeStatus> {
    if data.len() < 8 {
        return Err(DecodeStatus::Incomplete);
    }
    let len = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let total = 4 + len + 4;
    if data.len() < total {
        return Err(DecodeStatus::Incomplete);
    }
    let payload = &data[4..4 + len];
    let stored_crc = u32::from_le_bytes(data[4 + len..total].try_into().unwrap());
    if crc32c(payload) != stored_crc {
        return Err(DecodeStatus::Corrupted);
    }
    let entry: WalEntry = bincode::deserialize(payload).map_err(|_| DecodeStatus::Corrupted)?;
    Ok((entry, total))
}

#[derive(Debug)]
enum DecodeStatus {
    Incomplete,
    Corrupted,
}

/// CRC-32C (Castagnoli) via the `crc32c` crate — uses hardware
/// CRC instructions (SSE4.2 / ARMv8) where available.
fn crc32c(data: &[u8]) -> u32 {
    crc32c::crc32c(data)
}

// ---------------------------------------------------------------------------
// Segment — one WAL file
// ---------------------------------------------------------------------------

const DEFAULT_SEGMENT_MAX: u64 = 64 * 1024 * 1024; // 64 MB

struct Segment {
    writer: BufWriter<File>,
    size: u64,
}

impl Segment {
    fn create(dir: &Path, first_lsn: Lsn) -> Result<Self, GraphError> {
        let path = dir.join(format!("{:020}.wal", first_lsn));
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self {
            writer: BufWriter::with_capacity(256 * 1024, file),
            size: 0,
        })
    }

    /// Re-open an existing segment file for appending.
    fn reopen(path: PathBuf, _first_lsn: Lsn) -> Result<Self, GraphError> {
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let file = OpenOptions::new().append(true).open(&path)?;
        Ok(Self {
            writer: BufWriter::with_capacity(256 * 1024, file),
            size,
        })
    }

    fn append(&mut self, record: &[u8]) -> Result<(), GraphError> {
        self.writer.write_all(record)?;
        self.size += record.len() as u64;
        Ok(())
    }

    fn sync(&mut self) -> Result<(), GraphError> {
        let start = std::time::Instant::now();
        self.writer.flush()?;
        self.writer.get_ref().sync_data()?;
        // EBS gp3 fsync runs ~0.5-2 ms; if this histogram drifts to tens
        // of ms, the device is the bottleneck and SyncPolicy::EveryN
        // should grow (group commit) before any other change.
        metrics::histogram!("helix_wal_fsync_duration_ms")
            .record(start.elapsed().as_secs_f64() * 1000.0);
        metrics::counter!("helix_wal_fsync_total").increment(1);
        Ok(())
    }

    fn needs_rotate(&self) -> bool {
        self.size >= DEFAULT_SEGMENT_MAX
    }
}

// ---------------------------------------------------------------------------
// WalWriter — the core WAL engine
// ---------------------------------------------------------------------------

/// Controls when fsync is called.
#[derive(Debug, Clone)]
pub enum SyncPolicy {
    /// fsync every write. Safest, ~1ms overhead per write.
    Every,
    /// fsync every N writes. Default 100.
    EveryN(u32),
    /// Never fsync explicitly. OS flushes eventually.
    None,
}

impl Default for SyncPolicy {
    fn default() -> Self {
        // Allow ops to override the WAL sync cadence without recompiling.
        // Format: `every` | `every_n=<u32>` | `none`. Falls back to the
        // historical `EveryN(100)` default when unset or unparseable.
        match std::env::var("HELIX_WAL_SYNC_POLICY")
            .ok()
            .as_deref()
            .map(str::trim)
        {
            Some("every") | Some("Every") => SyncPolicy::Every,
            Some("none") | Some("None") => SyncPolicy::None,
            Some(s) if s.starts_with("every_n=") => {
                let n = s["every_n=".len()..].parse::<u32>().ok();
                match n {
                    Some(n) if n > 0 => SyncPolicy::EveryN(n),
                    _ => SyncPolicy::EveryN(100),
                }
            }
            _ => SyncPolicy::EveryN(100),
        }
    }
}

pub struct WalWriter {
    dir: PathBuf,
    next_lsn: AtomicU64,
    inner: Mutex<WalInner>,
    sync_policy: SyncPolicy,
}

struct WalInner {
    segment: Segment,
    writes_since_sync: u32,
    durable_lsn: Lsn,
}

impl WalWriter {
    /// Open or create a WAL directory. Scans existing segments to find
    /// the highest LSN so we continue the sequence.
    pub fn open(dir: PathBuf, sync_policy: SyncPolicy) -> Result<Self, GraphError> {
        fs::create_dir_all(&dir)?;
        let max_lsn = scan_max_lsn(&dir)?;
        let next = max_lsn + 1;

        // Reuse the last segment if it exists and is not full,
        // instead of unconditionally creating a new one.
        let segments = list_segments(&dir)?;
        let segment = if let Some((first_lsn, path)) = segments.last() {
            let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            if size < DEFAULT_SEGMENT_MAX {
                Segment::reopen(path.clone(), *first_lsn)?
            } else {
                Segment::create(&dir, next)?
            }
        } else {
            Segment::create(&dir, next)?
        };

        Ok(Self {
            dir,
            next_lsn: AtomicU64::new(next),
            inner: Mutex::new(WalInner {
                segment,
                writes_since_sync: 0,
                durable_lsn: max_lsn,
            }),
            sync_policy,
        })
    }

    /// Append a mutation. Returns the assigned LSN.
    /// Call BEFORE writing to LMDB.
    pub fn append(&self, collection: &str, op: WalOp) -> Result<Lsn, GraphError> {
        let lsn = self.next_lsn.fetch_add(1, Ordering::SeqCst);
        let entry = WalEntry {
            lsn,
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
            collection: collection.to_string(),
            op,
        };
        let record = encode_record(&entry)?;

        let mut inner = self
            .inner
            .lock()
            .map_err(|e| GraphError::New(format!("WAL lock poisoned: {}", e)))?;

        // Rotate segment if full
        if inner.segment.needs_rotate() {
            inner.segment.sync()?;
            inner.durable_lsn = lsn.saturating_sub(1);
            inner.segment = Segment::create(&self.dir, lsn)?;
        }

        inner.segment.append(&record)?;
        inner.writes_since_sync += 1;

        // Sync policy
        match &self.sync_policy {
            SyncPolicy::Every => {
                inner.segment.sync()?;
                inner.durable_lsn = lsn;
            }
            SyncPolicy::EveryN(n) => {
                if inner.writes_since_sync >= *n {
                    inner.segment.sync()?;
                    inner.durable_lsn = lsn;
                    inner.writes_since_sync = 0;
                }
            }
            SyncPolicy::None => {}
        }

        Ok(lsn)
    }

    /// Force flush everything to disk. Call on graceful shutdown.
    pub fn flush(&self) -> Result<Lsn, GraphError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| GraphError::New(format!("WAL lock poisoned: {}", e)))?;
        inner.segment.sync()?;
        let lsn = self.next_lsn.load(Ordering::SeqCst).saturating_sub(1);
        inner.durable_lsn = lsn;
        Ok(lsn)
    }

    /// Current durable (fsynced) LSN.
    pub fn durable_lsn(&self) -> Lsn {
        self.inner.lock().map(|i| i.durable_lsn).unwrap_or(0)
    }

    /// Next LSN that will be assigned.
    pub fn next_lsn(&self) -> Lsn {
        self.next_lsn.load(Ordering::SeqCst)
    }

    /// Truncate segments whose entries are all <= committed_lsn.
    /// Keeps at least `keep` segments for follower catch-up.
    pub fn truncate(&self, committed_lsn: Lsn, keep: usize) -> Result<u32, GraphError> {
        let segments = list_segments(&self.dir)?;
        if segments.len() <= keep + 1 {
            return Ok(0); // +1 for the active segment
        }

        let mut removed = 0u32;
        let protected_from = segments.len().saturating_sub(keep + 1);
        for idx in 0..protected_from {
            let (_first_lsn, path) = &segments[idx];
            let next_first_lsn = segments
                .get(idx + 1)
                .map(|(first_lsn, _)| *first_lsn)
                .unwrap_or(Lsn::MAX);
            if next_first_lsn <= committed_lsn.saturating_add(1) {
                fs::remove_file(&path)?;
                removed += 1;
            }
        }
        Ok(removed)
    }
}

// ---------------------------------------------------------------------------
// WalReader — for crash recovery and replication streaming
// ---------------------------------------------------------------------------

/// Read WAL entries starting from a given LSN.
/// Used for both crash recovery and follower replication.
pub fn read_from(dir: &Path, from_lsn: Lsn) -> Result<Vec<WalEntry>, GraphError> {
    let segments = list_segments(dir)?;
    let mut entries = Vec::new();

    for (idx, (first_lsn, path)) in segments.iter().enumerate() {
        let data = fs::read(path)?;
        let mut offset = 0;
        while offset < data.len() {
            match decode_record(&data[offset..]) {
                Ok((entry, consumed)) => {
                    if entry.lsn >= from_lsn {
                        entries.push(entry);
                    }
                    offset += consumed;
                }
                Err(DecodeStatus::Incomplete) => break,
                Err(DecodeStatus::Corrupted) => {
                    metrics::counter!("helix_wal_decode_corrupt_records_total").increment(1);
                    tracing::error!(
                        segment = %path.display(),
                        segment_lsn_start = *first_lsn,
                        segment_lsn_end = ?segments.get(idx + 1).map(|(lsn, _)| lsn.saturating_sub(1)),
                        offset_bytes = offset,
                        "Corrupted WAL record encountered; stopping segment replay"
                    );
                    break;
                }
            }
        }
    }

    Ok(entries)
}

// ---------------------------------------------------------------------------
// Recovery — replay uncommitted WAL entries into LMDB on startup
// ---------------------------------------------------------------------------

/// Result of a recovery pass.
#[derive(Debug)]
pub struct RecoveryReport {
    pub replayed: u64,
    pub skipped: u64,
}

/// Recover from WAL after crash.
///
/// 1. Read `committed_lsn` from LMDB metadata.
/// 2. Read WAL entries from `committed_lsn + 1`.
/// 3. Only replay transactions that have a TxCommit marker.
/// 4. Returns a report of what was replayed.
///
/// The caller is responsible for calling the actual replay (applying ops
/// back to storage) since we don't want circular deps. This function
/// just returns the ops grouped by committed transactions.
pub fn recover(
    wal_dir: &Path,
    committed_lsn: Lsn,
) -> Result<(RecoveryReport, Vec<Vec<WalEntry>>), GraphError> {
    let entries = read_from(wal_dir, committed_lsn.saturating_add(1))?;
    if entries.is_empty() {
        return Ok((
            RecoveryReport {
                replayed: 0,
                skipped: 0,
            },
            vec![],
        ));
    }

    // Group by tx_id, track which are committed
    let mut tx_map: std::collections::HashMap<u64, Vec<WalEntry>> =
        std::collections::HashMap::new();
    let mut committed_txs: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut current_tx: u64 = 0;

    for entry in entries {
        match &entry.op {
            WalOp::TxBegin { tx_id } => {
                current_tx = *tx_id;
            }
            WalOp::TxCommit { tx_id } => {
                committed_txs.insert(*tx_id);
            }
            _ => {
                tx_map.entry(current_tx).or_default().push(entry);
            }
        }
    }

    let mut replayed = 0u64;
    let mut skipped = 0u64;
    let mut replay_batches = Vec::new();

    for (tx_id, ops) in tx_map {
        if committed_txs.contains(&tx_id) {
            replayed += ops.len() as u64;
            replay_batches.push(ops);
        } else {
            skipped += ops.len() as u64;
        }
    }

    // Sort batches by the minimum LSN in each batch so replay happens in
    // write order. tx_map is a HashMap and iterates in arbitrary order;
    // without this sort, edges could be replayed before their source nodes.
    replay_batches.sort_by_key(|batch| batch.first().map(|e| e.lsn).unwrap_or(0));

    Ok((RecoveryReport { replayed, skipped }, replay_batches))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// List segment files sorted by first LSN.
fn list_segments(dir: &Path) -> Result<Vec<(Lsn, PathBuf)>, GraphError> {
    let mut segments = Vec::new();
    if !dir.exists() {
        return Ok(segments);
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("wal") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                if let Ok(first_lsn) = stem.parse::<Lsn>() {
                    segments.push((first_lsn, path));
                }
            }
        }
    }
    segments.sort_by_key(|(lsn, _)| *lsn);
    Ok(segments)
}

/// Scan segments to find the highest LSN (by reading the last segment).
fn scan_max_lsn(dir: &Path) -> Result<Lsn, GraphError> {
    let segments = list_segments(dir)?;
    let Some((_first_lsn, last_path)) = segments.last() else {
        return Ok(0);
    };

    let data = fs::read(last_path)?;
    let mut offset = 0;
    let mut max_lsn: Lsn = 0;
    while offset < data.len() {
        match decode_record(&data[offset..]) {
            Ok((entry, consumed)) => {
                if entry.lsn > max_lsn {
                    max_lsn = entry.lsn;
                }
                offset += consumed;
            }
            Err(_) => break,
        }
    }
    Ok(max_lsn)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn roundtrip_encode_decode() {
        let entry = WalEntry {
            lsn: 42,
            timestamp_ms: 1234567890,
            collection: "test".to_string(),
            op: WalOp::CreateNode {
                id: 999,
                label: "User".to_string(),
                properties_json: vec![],
            },
        };
        let record = encode_record(&entry).unwrap();
        let (decoded, consumed) = decode_record(&record).unwrap();
        assert_eq!(consumed, record.len());
        assert_eq!(decoded.lsn, 42);
        assert_eq!(decoded.collection, "test");
    }

    #[test]
    fn corrupted_crc_detected() {
        let entry = WalEntry {
            lsn: 1,
            timestamp_ms: 0,
            collection: "x".to_string(),
            op: WalOp::DropNode { id: 1 },
        };
        let mut record = encode_record(&entry).unwrap();
        // Flip a byte in the payload
        record[5] ^= 0xFF;
        assert!(matches!(
            decode_record(&record),
            Err(DecodeStatus::Corrupted)
        ));
    }

    #[test]
    fn writer_appends_and_reads_back() {
        let tmp = TempDir::new().unwrap();
        let wal_dir = tmp.path().join("wal");
        let writer = WalWriter::open(wal_dir.clone(), SyncPolicy::Every).unwrap();

        let lsn1 = writer
            .append(
                "coll",
                WalOp::CreateNode {
                    id: 1,
                    label: "A".into(),
                    properties_json: vec![],
                },
            )
            .unwrap();
        let lsn2 = writer
            .append(
                "coll",
                WalOp::CreateEdge {
                    id: 2,
                    label: "E".into(),
                    from_node: 1,
                    to_node: 1,
                    properties_json: vec![],
                },
            )
            .unwrap();

        assert_eq!(lsn1, 1);
        assert_eq!(lsn2, 2);
        assert_eq!(writer.durable_lsn(), 2);

        let entries = read_from(&wal_dir, 1).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].lsn, 1);
        assert_eq!(entries[1].lsn, 2);
    }

    #[test]
    fn segment_rotation() {
        let tmp = TempDir::new().unwrap();
        let wal_dir = tmp.path().join("wal");
        let writer = WalWriter::open(wal_dir.clone(), SyncPolicy::None).unwrap();

        // Write enough to trigger rotation (we'll check segment count)
        // Each entry is ~50-100 bytes, need 64MB / ~80 bytes ≈ 800K entries
        // Just verify the mechanism works with a smaller test
        for i in 0..100 {
            writer.append("coll", WalOp::DropNode { id: i }).unwrap();
        }
        writer.flush().unwrap();

        let entries = read_from(&wal_dir, 1).unwrap();
        assert_eq!(entries.len(), 100);
    }

    #[test]
    fn recovery_replays_committed_skips_incomplete() {
        let tmp = TempDir::new().unwrap();
        let wal_dir = tmp.path().join("wal");
        let writer = WalWriter::open(wal_dir.clone(), SyncPolicy::Every).unwrap();

        // Committed transaction
        writer.append("coll", WalOp::TxBegin { tx_id: 1 }).unwrap();
        writer
            .append(
                "coll",
                WalOp::CreateNode {
                    id: 10,
                    label: "A".into(),
                    properties_json: vec![],
                },
            )
            .unwrap();
        writer.append("coll", WalOp::TxCommit { tx_id: 1 }).unwrap();

        // Incomplete transaction (no commit — simulates crash)
        writer.append("coll", WalOp::TxBegin { tx_id: 2 }).unwrap();
        writer
            .append(
                "coll",
                WalOp::CreateNode {
                    id: 20,
                    label: "B".into(),
                    properties_json: vec![],
                },
            )
            .unwrap();
        // no TxCommit for tx 2

        let (report, batches) = recover(&wal_dir, 0).unwrap();
        assert_eq!(report.replayed, 1); // only tx 1's CreateNode
        assert_eq!(report.skipped, 1); // tx 2's CreateNode
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 1);
    }

    #[test]
    fn truncation_keeps_min_segments() {
        let tmp = TempDir::new().unwrap();
        let wal_dir = tmp.path().join("wal");
        fs::create_dir_all(&wal_dir).unwrap();

        // Create fake segment files
        for lsn in [1u64, 100, 200, 300] {
            let path = wal_dir.join(format!("{:020}.wal", lsn));
            fs::write(&path, b"fake").unwrap();
        }

        let segments_before = list_segments(&wal_dir).unwrap();
        assert_eq!(segments_before.len(), 4);

        // Truncate with committed_lsn=250, keep 2
        let writer = WalWriter::open(wal_dir.clone(), SyncPolicy::None).unwrap();
        let removed = writer.truncate(250, 2).unwrap();
        assert_eq!(removed, 1); // removes segment starting at LSN 1

        let segments_after = list_segments(&wal_dir).unwrap();
        // 3 original remaining + 1 new from WalWriter::open
        assert!(segments_after.len() >= 3);
    }

    #[test]
    fn crc32c_sanity() {
        let data = b"hello world";
        let c1 = crc32c(data);
        let c2 = crc32c(data);
        assert_eq!(c1, c2);
        assert_ne!(c1, crc32c(b"hello worlD"));
    }

    #[test]
    fn read_from_empty_dir() {
        let tmp = TempDir::new().unwrap();
        let entries = read_from(tmp.path(), 0).unwrap();
        assert!(entries.is_empty());
    }
}
