//! Raft consensus for HelixDB using TiKV's raft-rs.
//!
//! Every mutation goes through Raft before hitting LMDB:
//!   propose(bytes) → leader replicates to majority → commit → apply to LMDB
//!
//! The proposal payload is opaque to this module. Callers serialize their own
//! mutation type into proposal bytes and provide an apply callback.
//!
//! Transport is pluggable — the RaftNode drives ticks and messages,
//! the caller provides a send function for inter-node communication.

use prost::Message as ProstMessage;
use raft::prelude::*;
use raft::{Config as RaftConfig, RawNode, StateRole};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Proposal — what gets proposed to Raft and applied to LMDB
// ---------------------------------------------------------------------------

/// A proposed mutation. Serialized into Raft log entry data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    /// Unique proposal ID for response routing.
    pub id: u64,
    /// Opaque payload bytes supplied by the caller.
    pub data: Vec<u8>,
}

impl Proposal {
    pub fn encode(&self) -> Vec<u8> {
        bincode::serialize(self).expect("proposal serialization should not fail")
    }

    pub fn decode(data: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(data)
    }
}

// ---------------------------------------------------------------------------
// In-memory Raft storage backed by our WAL
// ---------------------------------------------------------------------------

/// Simple in-memory storage implementing raft::Storage trait.
/// WAL segments on disk provide durability (from wal.rs).
/// This keeps the raft log in memory for fast access.
#[derive(Clone)]
pub struct MemStorage {
    inner: Arc<Mutex<MemStorageInner>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedMemStorage {
    hard_state: Vec<u8>,
    conf_state: Vec<u8>,
    snapshot: Vec<u8>,
    entries: Vec<Vec<u8>>,
    offset: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedRaftState {
    storage: PersistedMemStorage,
    last_applied: u64,
}

struct MemStorageInner {
    hard_state: HardState,
    conf_state: ConfState,
    snapshot: Snapshot,
    entries: Vec<Entry>,
    /// Offset: entries[0].index = offset
    offset: u64,
}

impl MemStorage {
    pub fn new(voters: Vec<u64>) -> Self {
        let mut cs = ConfState::default();
        cs.voters = voters;

        Self {
            inner: Arc::new(Mutex::new(MemStorageInner {
                hard_state: HardState::default(),
                conf_state: cs,
                snapshot: Snapshot::default(),
                entries: vec![],
                offset: 1,
            })),
        }
    }

    pub fn from_persisted(state: PersistedMemStorage) -> Result<Self, String> {
        let hard_state =
            HardState::decode(state.hard_state.as_slice()).map_err(|e| e.to_string())?;
        let conf_state =
            ConfState::decode(state.conf_state.as_slice()).map_err(|e| e.to_string())?;
        let snapshot = Snapshot::decode(state.snapshot.as_slice()).map_err(|e| e.to_string())?;
        let mut entries = Vec::with_capacity(state.entries.len());
        for bytes in state.entries {
            let entry = Entry::decode(bytes.as_slice()).map_err(|e| e.to_string())?;
            entries.push(entry);
        }

        Ok(Self {
            inner: Arc::new(Mutex::new(MemStorageInner {
                hard_state,
                conf_state,
                snapshot,
                entries,
                offset: state.offset,
            })),
        })
    }

    /// Append entries to log. Called after Raft ready.
    pub fn append(&self, entries: &[Entry]) {
        if entries.is_empty() {
            return;
        }
        let mut inner = self.inner.lock().unwrap();
        let already_compacted = inner.offset.saturating_sub(entries[0].index) as usize;
        let entries = if already_compacted >= entries.len() {
            return;
        } else {
            &entries[already_compacted..]
        };
        let first = entries[0].index;

        if inner.entries.is_empty() {
            inner.offset = first;
        }

        // Truncate any conflicting entries
        let start = (first - inner.offset) as usize;
        if start < inner.entries.len() {
            inner.entries.truncate(start);
        }
        inner.entries.extend_from_slice(entries);
    }

    /// Set hard state (term, vote, commit).
    pub fn set_hard_state(&self, hs: HardState) {
        self.inner.lock().unwrap().hard_state = hs;
    }

    /// Set conf state.
    pub fn set_conf_state(&self, cs: ConfState) {
        self.inner.lock().unwrap().conf_state = cs;
    }

    pub fn apply_snapshot(&self, snapshot: Snapshot) {
        let mut inner = self.inner.lock().unwrap();
        inner.snapshot = snapshot.clone();
        inner.conf_state = snapshot
            .get_metadata()
            .conf_state
            .clone()
            .unwrap_or_default();
        inner.offset = snapshot.get_metadata().index.saturating_add(1);
        inner.entries.clear();
    }

    pub fn create_snapshot(&self, index: u64, term: u64, data: Vec<u8>) {
        let mut inner = self.inner.lock().unwrap();
        if index < inner.snapshot.get_metadata().index {
            return;
        }

        let mut snapshot = Snapshot::default();
        snapshot.mut_metadata().index = index;
        snapshot.mut_metadata().term = term;
        snapshot
            .mut_metadata()
            .set_conf_state(inner.conf_state.clone());
        snapshot.data = data.into();
        inner.snapshot = snapshot;
    }

    /// Compact log up to index (inclusive). Keeps entries after index.
    pub fn compact(&self, index: u64) {
        let mut inner = self.inner.lock().unwrap();
        let first_index = inner.offset;
        if index < first_index {
            return;
        }

        let new_offset = index.saturating_add(1);
        if inner.entries.is_empty() {
            inner.offset = inner.offset.max(new_offset);
            return;
        }

        let drain = (new_offset.saturating_sub(first_index)) as usize;
        if drain >= inner.entries.len() {
            inner.entries.clear();
        } else {
            inner.entries.drain(..drain);
        }
        inner.offset = new_offset;
    }

    pub fn first_index(&self) -> u64 {
        self.inner.lock().unwrap().offset
    }

    /// Last index in the log.
    pub fn last_index(&self) -> u64 {
        let inner = self.inner.lock().unwrap();
        if inner.entries.is_empty() {
            inner.offset.saturating_sub(1)
        } else {
            inner.entries.last().unwrap().index
        }
    }

    pub fn snapshot_index(&self) -> u64 {
        self.inner.lock().unwrap().snapshot.get_metadata().index
    }

    pub fn to_persisted(&self) -> PersistedMemStorage {
        let inner = self.inner.lock().unwrap();
        PersistedMemStorage {
            hard_state: inner.hard_state.encode_to_vec(),
            conf_state: inner.conf_state.encode_to_vec(),
            snapshot: inner.snapshot.encode_to_vec(),
            entries: inner.entries.iter().map(Entry::encode_to_vec).collect(),
            offset: inner.offset,
        }
    }
}

impl raft::Storage for MemStorage {
    fn initial_state(&self) -> raft::Result<RaftState> {
        let inner = self.inner.lock().unwrap();
        Ok(RaftState {
            hard_state: inner.hard_state.clone(),
            conf_state: inner.conf_state.clone(),
        })
    }

    fn entries(
        &self,
        low: u64,
        high: u64,
        max_size: impl Into<Option<u64>>,
        _context: raft::GetEntriesContext,
    ) -> raft::Result<Vec<Entry>> {
        let inner = self.inner.lock().unwrap();
        let max_size = max_size.into().unwrap_or(u64::MAX) as usize;
        let first_index = inner.offset;
        let last_index = if inner.entries.is_empty() {
            inner.offset.saturating_sub(1)
        } else {
            inner.entries.last().unwrap().index
        };

        if low < first_index {
            return Err(raft::Error::Store(raft::StorageError::Compacted));
        }

        if high > last_index.saturating_add(1) {
            return Err(raft::Error::Store(raft::StorageError::Unavailable));
        }

        if low == high {
            return Ok(Vec::new());
        }

        if inner.entries.is_empty() {
            return Err(raft::Error::Store(raft::StorageError::Unavailable));
        }

        let lo = (low - inner.offset) as usize;
        let hi = (high - inner.offset) as usize;

        if lo >= inner.entries.len() {
            return Err(raft::Error::Store(raft::StorageError::Unavailable));
        }

        let hi = hi.min(inner.entries.len());
        let mut result = Vec::new();
        let mut size = 0usize;

        for entry in &inner.entries[lo..hi] {
            size += entry.data.len();
            if !result.is_empty() && size > max_size {
                break;
            }
            result.push(entry.clone());
        }
        Ok(result)
    }

    fn term(&self, idx: u64) -> raft::Result<u64> {
        let inner = self.inner.lock().unwrap();
        if idx == 0 {
            return Ok(0);
        }
        let snap_idx = inner.snapshot.get_metadata().index;
        if idx == snap_idx {
            return Ok(inner.snapshot.get_metadata().term);
        }
        if idx < inner.offset {
            return Err(raft::Error::Store(raft::StorageError::Compacted));
        }
        if !inner.entries.is_empty() {
            let i = (idx - inner.offset) as usize;
            if i < inner.entries.len() {
                return Ok(inner.entries[i].term);
            }
        }
        Err(raft::Error::Store(raft::StorageError::Unavailable))
    }

    fn first_index(&self) -> raft::Result<u64> {
        Ok(self.first_index())
    }

    fn last_index(&self) -> raft::Result<u64> {
        Ok(self.last_index())
    }

    fn snapshot(&self, request_index: u64, _to: u64) -> raft::Result<Snapshot> {
        let inner = self.inner.lock().unwrap();
        let snapshot = inner.snapshot.clone();
        if snapshot.get_metadata().index < request_index {
            return Err(raft::Error::Store(
                raft::StorageError::SnapshotTemporarilyUnavailable,
            ));
        }
        Ok(snapshot)
    }
}

// ---------------------------------------------------------------------------
// RaftNode — drives the Raft state machine
// ---------------------------------------------------------------------------

/// Wraps a RawNode and provides a simple propose/tick/apply interface.
pub struct RaftNode {
    pub node: RawNode<MemStorage>,
    pub storage: MemStorage,
    /// Callback for applying committed entries to the state machine.
    apply_fn: Option<Arc<dyn Fn(&[u8]) -> Result<(), String> + Send + Sync>>,
    /// Callback for restoring a full state-machine snapshot.
    snapshot_fn: Option<Arc<dyn Fn(&[u8]) -> Result<(), String> + Send + Sync>>,
    /// Last applied index (= our LSN for replication).
    pub last_applied: u64,
    last_tick: Instant,
}

#[derive(Debug, Clone)]
pub struct AppliedProposal {
    pub id: u64,
    pub index: u64,
    pub result: Result<(), String>,
}

pub struct ReadyOutcome {
    pub messages: Vec<Message>,
    pub applied: Vec<AppliedProposal>,
    pub read_states: Vec<ReadState>,
    pub should_persist: bool,
}

impl RaftNode {
    /// Create a new Raft node.
    ///
    /// `id`: unique node ID (1, 2, 3)
    /// `peers`: all voter node IDs including self
    pub fn new(id: u64, peers: Vec<u64>) -> Result<Self, raft::Error> {
        Self::new_with_storage(id, MemStorage::new(peers), 0)
    }

    pub fn new_with_storage(
        id: u64,
        storage: MemStorage,
        last_applied: u64,
    ) -> Result<Self, raft::Error> {
        let config = RaftConfig {
            id,
            applied: last_applied,
            election_tick: 10, // 10 * tick_interval = 1s with 100ms ticks
            heartbeat_tick: 3, // 3 * tick_interval = 300ms
            max_size_per_msg: 1024 * 1024, // 1MB
            max_inflight_msgs: 256,
            ..Default::default()
        };
        config.validate()?;

        let node = RawNode::new(
            &config,
            storage.clone(),
            &slog::Logger::root(slog::Discard, slog::o!()),
        )?;

        Ok(Self {
            node,
            storage,
            apply_fn: None,
            snapshot_fn: None,
            last_applied,
            last_tick: Instant::now(),
        })
    }

    pub fn restore_from_path<P: AsRef<Path>>(
        id: u64,
        peers: Vec<u64>,
        path: P,
    ) -> Result<Self, String> {
        let path = path.as_ref();
        if !path.exists() {
            return Self::new(id, peers).map_err(|e| e.to_string());
        }
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        let state: PersistedRaftState = bincode::deserialize(&bytes).map_err(|e| e.to_string())?;
        let storage = MemStorage::from_persisted(state.storage)?;
        Self::new_with_storage(id, storage, state.last_applied).map_err(|e| e.to_string())
    }

    /// Register the LMDB apply callback.
    pub fn set_apply_fn<F>(&mut self, f: F)
    where
        F: Fn(&[u8]) -> Result<(), String> + Send + Sync + 'static,
    {
        self.apply_fn = Some(Arc::new(f));
    }

    /// Register the state-machine snapshot restore callback.
    pub fn set_snapshot_fn<F>(&mut self, f: F)
    where
        F: Fn(&[u8]) -> Result<(), String> + Send + Sync + 'static,
    {
        self.snapshot_fn = Some(Arc::new(f));
    }

    /// Propose a mutation. Only succeeds on the leader.
    /// Returns the proposal ID for tracking.
    pub fn propose(&mut self, proposal: Proposal) -> Result<u64, String> {
        if self.node.raft.state != StateRole::Leader {
            return Err("not leader".to_string());
        }
        let id = proposal.id;
        let data = proposal.encode();
        self.node
            .propose(vec![], data)
            .map_err(|e| format!("propose failed: {}", e))?;
        Ok(id)
    }

    pub fn request_read_index(&mut self, context: Vec<u8>) -> Result<(), String> {
        if self.node.raft.state != StateRole::Leader {
            return Err("not leader".to_string());
        }
        self.node.read_index(context);
        Ok(())
    }

    /// Tick the Raft node. Call every ~100ms.
    pub fn tick(&mut self) {
        self.node.tick();
    }

    /// Check if enough time has passed for a tick (~100ms).
    pub fn should_tick(&self) -> bool {
        self.last_tick.elapsed() >= Duration::from_millis(100)
    }

    /// Mark tick as done.
    pub fn mark_ticked(&mut self) {
        self.last_tick = Instant::now();
    }

    /// Process Raft ready state: persist entries, apply committed,
    /// return messages to send to peers.
    ///
    /// Returns (messages_to_send, applied_proposals).
    pub fn process_ready(&mut self) -> Result<ReadyOutcome, String> {
        if !self.node.has_ready() {
            return Ok(ReadyOutcome {
                messages: vec![],
                applied: vec![],
                read_states: vec![],
                should_persist: false,
            });
        }

        let mut ready = self.node.ready();
        let mut applied = Vec::new();
        let should_persist =
            ready.hs().is_some() || !ready.entries().is_empty() || !ready.snapshot().is_empty();

        if !ready.snapshot().is_empty() {
            let snapshot = ready.snapshot().clone();
            if let Some(f) = &self.snapshot_fn {
                f(snapshot.data.as_slice())?;
            }
            self.storage.apply_snapshot(snapshot.clone());
            self.last_applied = snapshot.get_metadata().index;
        }

        // 1. Persist hard state
        if let Some(hs) = ready.hs() {
            self.storage.set_hard_state(hs.clone());
        }

        // 2. Append new entries to log
        if !ready.entries().is_empty() {
            self.storage.append(ready.entries());
        }

        // 3. Apply committed entries to LMDB
        for entry in ready.take_committed_entries() {
            let index = entry.index;
            if entry.data.is_empty() {
                // Config change or empty entry
                self.last_applied = index;
                continue;
            }

            match Proposal::decode(&entry.data) {
                Ok(proposal) => {
                    let result = if let Some(f) = &self.apply_fn {
                        f(&proposal.data)
                    } else {
                        Ok(())
                    };
                    if let Err(err) = &result {
                        eprintln!("[raft] apply error at index {}: {}", index, err);
                    }
                    applied.push(AppliedProposal {
                        id: proposal.id,
                        index,
                        result,
                    });
                }
                Err(e) => {
                    eprintln!("[raft] failed to decode entry at index {}: {}", index, e);
                }
            }

            self.last_applied = index;
        }

        // 4. Collect messages to send to peers. raft-rs can split outbound
        // traffic into pre-persist and post-persist buckets; we need both.
        let read_states = ready.take_read_states();
        let mut messages = ready.take_messages();
        messages.extend(ready.take_persisted_messages());

        // 5. Advance Raft
        let mut light_rd = self.node.advance(ready);

        // Process light ready too
        if let Some(ref hs) = light_rd.commit_index() {
            let mut current_hs = self.storage.inner.lock().unwrap().hard_state.clone();
            current_hs.commit = *hs;
            self.storage.set_hard_state(current_hs);
        }

        if !light_rd.committed_entries().is_empty() {
            for entry in light_rd.take_committed_entries() {
                let index = entry.index;
                if !entry.data.is_empty() {
                    if let Ok(proposal) = Proposal::decode(&entry.data) {
                        let result = if let Some(f) = &self.apply_fn {
                            f(&proposal.data)
                        } else {
                            Ok(())
                        };
                        if let Err(err) = &result {
                            eprintln!("[raft] apply error at index {}: {}", index, err);
                        }
                        applied.push(AppliedProposal {
                            id: proposal.id,
                            index,
                            result,
                        });
                    }
                }
                self.last_applied = index;
            }
        }

        messages.extend(light_rd.take_messages());

        self.node.advance_apply();

        Ok(ReadyOutcome {
            messages,
            applied,
            read_states,
            should_persist,
        })
    }

    /// Feed a message from a peer (append entries, vote, etc).
    pub fn step(&mut self, msg: Message) -> Result<(), raft::Error> {
        self.node.step(msg)
    }

    /// Is this node the current leader?
    pub fn is_leader(&self) -> bool {
        self.node.raft.state == StateRole::Leader
    }

    /// Current leader ID (0 if unknown).
    pub fn leader_id(&self) -> u64 {
        self.node.raft.leader_id
    }

    /// Current term.
    pub fn term(&self) -> u64 {
        self.node.raft.term
    }

    pub fn commit_index(&self) -> u64 {
        self.node.raft.raft_log.committed
    }

    pub fn first_index(&self) -> u64 {
        self.storage.first_index()
    }

    pub fn last_index(&self) -> u64 {
        self.storage.last_index()
    }

    pub fn snapshot_index(&self) -> u64 {
        self.storage.snapshot_index()
    }

    pub fn create_snapshot(&self, index: u64, data: Vec<u8>) -> Result<(), String> {
        let term = raft::Storage::term(&self.storage, index).map_err(|e| e.to_string())?;
        self.storage.create_snapshot(index, term, data);
        Ok(())
    }

    pub fn compact_to(&self, index: u64) {
        self.storage.compact(index);
    }

    /// Campaign to become leader (used for single-node bootstrap).
    pub fn campaign(&mut self) -> Result<(), raft::Error> {
        self.node.campaign()
    }

    pub fn persist_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let path = path.as_ref();
        let parent = path
            .parent()
            .ok_or_else(|| format!("raft state path has no parent: {}", path.display()))?;
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        let state = PersistedRaftState {
            storage: self.storage.to_persisted(),
            last_applied: self.last_applied,
        };
        let bytes = bincode::serialize(&state).map_err(|e| e.to_string())?;
        let tmp = path.with_extension("tmp");
        // Write + fsync the tmp file so the bytes are durable before rename.
        // Without this, a crash between write() and rename() can leave a
        // zero-length tmp file or leave a renamed but empty file on disk.
        {
            let mut f = fs::File::create(&tmp).map_err(|e| e.to_string())?;
            std::io::Write::write_all(&mut f, &bytes).map_err(|e| e.to_string())?;
            f.sync_all().map_err(|e| e.to_string())?;
        }
        fs::rename(&tmp, path).map_err(|e| e.to_string())?;
        // fsync the parent directory so the rename is durable; otherwise
        // the directory entry can be lost on crash even though the file
        // contents survived.
        if let Ok(dir) = fs::File::open(parent) {
            let _ = dir.sync_all();
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn single_node_propose_and_apply() {
        let mut node = RaftNode::new(1, vec![1]).unwrap();

        // Track applied ops
        let applied = Arc::new(Mutex::new(Vec::<Vec<u8>>::new()));
        let applied_clone = Arc::clone(&applied);
        node.set_apply_fn(move |data| {
            applied_clone.lock().unwrap().push(data.to_vec());
            Ok(())
        });

        // Bootstrap: campaign to become leader
        node.campaign().unwrap();

        // Process until leader
        for _ in 0..20 {
            node.tick();
            node.process_ready().unwrap();
        }
        assert!(node.is_leader(), "node should be leader");

        // Propose a mutation
        let proposal = Proposal {
            id: 1,
            data: b"hello".to_vec(),
        };
        node.propose(proposal).unwrap();

        // Process ready to commit and apply
        for _ in 0..10 {
            node.tick();
            let outcome = node.process_ready().unwrap();
            assert!(outcome.messages.is_empty(), "no peers, no messages");
            if !outcome.applied.is_empty() {
                break;
            }
        }

        // Verify it was applied
        let ops = applied.lock().unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0], b"hello".to_vec());
        assert!(node.last_applied > 0);
    }

    #[test]
    fn proposal_roundtrip() {
        let p = Proposal {
            id: 99,
            data: b"payload".to_vec(),
        };
        let encoded = p.encode();
        let decoded = Proposal::decode(&encoded).unwrap();
        assert_eq!(decoded.id, 99);
        assert_eq!(decoded.data, b"payload".to_vec());
    }

    #[test]
    fn non_leader_rejects_proposal() {
        // Single node that hasn't campaigned yet — not a leader
        let mut node = RaftNode::new(1, vec![1, 2, 3]).unwrap();
        let result = node.propose(Proposal {
            id: 1,
            data: b"x".to_vec(),
        });
        assert!(result.is_err());
    }

    #[test]
    fn mem_storage_append_and_retrieve() {
        let storage = MemStorage::new(vec![1]);

        let mut entry = Entry::default();
        entry.index = 1;
        entry.term = 1;
        entry.data = b"hello".to_vec().into();

        storage.append(&[entry]);
        assert_eq!(storage.last_index(), 1);

        let entries =
            raft::Storage::entries(&storage, 1, 2, None, raft::GetEntriesContext::empty(false))
                .unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(&entries[0].data[..], b"hello");
    }

    #[test]
    fn raft_state_persists_and_restores() {
        let tmp = TempDir::new().unwrap();
        let state_path = tmp.path().join("raft").join("state.bin");

        let mut node = RaftNode::new(1, vec![1]).unwrap();
        node.set_apply_fn(|_| Ok(()));
        node.campaign().unwrap();
        for _ in 0..20 {
            node.tick();
            node.process_ready().unwrap();
        }
        assert!(node.is_leader());

        node.propose(Proposal {
            id: 7,
            data: b"persist-me".to_vec(),
        })
        .unwrap();
        for _ in 0..10 {
            node.tick();
            let outcome = node.process_ready().unwrap();
            if !outcome.applied.is_empty() {
                break;
            }
        }
        assert!(node.last_applied > 0);
        node.persist_to_path(&state_path).unwrap();

        let restored = RaftNode::restore_from_path(1, vec![1], &state_path).unwrap();
        assert_eq!(restored.last_applied, node.last_applied);
        assert_eq!(restored.storage.last_index(), node.storage.last_index());
        assert_eq!(restored.term(), node.term());
    }

    #[test]
    fn compaction_returns_compacted_for_old_entries() {
        let storage = MemStorage::new(vec![1]);
        let mut entries = Vec::new();
        for index in 1..=5 {
            let mut entry = Entry::default();
            entry.index = index;
            entry.term = 1;
            entries.push(entry);
        }
        storage.append(&entries);
        storage.create_snapshot(5, 1, Vec::new());
        storage.compact(3);

        let compacted =
            raft::Storage::entries(&storage, 1, 2, None, raft::GetEntriesContext::empty(false))
                .unwrap_err();
        assert!(matches!(
            compacted,
            raft::Error::Store(raft::StorageError::Compacted)
        ));
        assert_eq!(raft::Storage::term(&storage, 5).unwrap(), 1);
        assert_eq!(storage.first_index(), 4);
    }
}
