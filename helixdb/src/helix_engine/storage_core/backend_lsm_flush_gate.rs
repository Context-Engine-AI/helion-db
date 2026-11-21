//! The gate every memtable flush of the process passes before it encodes.
//!
//! A flush holds its memtable's whole SST encoded, every block twice, until
//! the upload ends, and SlateDB bounds flushes for each database on its own
//! (`l0_flush_parallelism`). Twelve collections that reached their flush size
//! in the same second took a 6 Gi writer from 4,473 to 6,121 MiB in 1.45 s
//! and the kernel killed it (pod smoke, 2026-11-18).
//!
//! The rule. A flush is allowed [`ALLOWANCE_TIMES`] the bytes of its SST. It
//! starts when the memory the pod has free as it asks holds that on top of
//! the whole allowance of every flush still running; else it waits, first
//! come first served, and asks again each time a flush ends and every
//! [`LOOK_AGAIN`]. With no flush running the first in line starts whatever
//! the memory: a flush is the only way a memtable's memory comes back.
//!
//! A running flush counts whole. What it has taken of its allowance already
//! shows in the free memory read, so for a moment that part is counted twice:
//! what it has yet to take cannot be read apart from it, and flushes started
//! on one reading have taken nothing a second reading would show. Too few may
//! start, never too many.
//!
//! One collection's upload that cannot get through must not hold the others'
//! flushes. SlateDB gives the allowance back while a failed upload waits to be
//! tried again; most errors it retries inside the store call, which never
//! returns, so the bytes of a flush that has held its allowance for
//! [`OVERDUE`] count no longer either: it encoded long before, and what it
//! took shows in the memory read. It still counts as running: a store down
//! for every collection would else start the first in line with nothing free
//! each time the flush before it passed that age.

use futures::future::BoxFuture;
use slatedb::L0FlushGate;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

/// What a flush is allowed, in times the bytes of the SST it writes. The
/// close of a memtable 95% full took 3.07 times its bytes at its highest
/// (375 MiB for 122, pod smoke behind 9a8d50ad); the fourth is for what its
/// collection writes into the next memtable while the flush runs.
pub(super) const ALLOWANCE_TIMES: usize = 4;
/// A waiting flush reads the free memory again this often, beside each time
/// a flush ends: memory also comes back from what is not a flush.
const LOOK_AGAIN: Duration = Duration::from_secs(1);
/// A flush that has waited this long is said so, once in as long, and one
/// that has held its allowance this long is said so and its bytes count no
/// longer.
pub(super) const OVERDUE: Duration = Duration::from_secs(60);

/// Bytes the pod has free under its memory limit, read now. `None` where its
/// use cannot be read: there no flush waits, as before the gate.
pub(crate) type FreeMemory = Box<dyn Fn() -> Option<usize> + Send + Sync>;

/// Whether a flush allowed `allowance` bytes may start, with `free` bytes
/// free and `running` flushes holding `reserved` bytes between them.
pub(super) fn may_start(
    free: Option<usize>,
    reserved: usize,
    running: usize,
    allowance: usize,
) -> bool {
    running == 0 || free.is_none_or(|free| free >= reserved.saturating_add(allowance))
}

/// Whether to say now that the flush at the head of the line, which asked at
/// `asked`, is still waiting; `warned` is when it was last said.
pub(super) fn overdue(asked: Instant, warned: Option<Instant>, now: Instant) -> bool {
    now.duration_since(asked) >= OVERDUE
        && warned.is_none_or(|at| now.duration_since(at) >= OVERDUE)
}

/// See the module.
pub(crate) struct FlushGate(Arc<Line>);

struct Line {
    /// Held by the flush at the head of the line for as long as it waits, so
    /// that a later, smaller flush does not start past it.
    turn: tokio::sync::Mutex<()>,
    /// A flush has given its allowance back.
    ended: tokio::sync::Notify,
    state: Mutex<State>,
    free: FreeMemory,
    look_again: Duration,
    /// [`OVERDUE`], or less in a test.
    counted_for: Duration,
}

#[derive(Default)]
struct State {
    /// The flushes holding an allowance: encoding, or uploading.
    holding: Vec<Holding>,
    /// How many flushes have started: the next one's id.
    next_id: u64,
    /// The database of each flush that has asked and not started. Its writes
    /// wait with it.
    waiting: Vec<String>,
    /// How many of the started had to wait.
    waited: u64,
    warned: Option<Instant>,
    /// How many failed uploads SlateDB has told of.
    upload_failures: u64,
    /// The last of them: its database, the error, the wait before the next
    /// attempt.
    #[cfg(test)]
    last_upload_failure: Option<(String, String, Duration)>,
}

struct Holding {
    id: u64,
    /// The database's path, for the log.
    db: String,
    allowance: usize,
    since: Instant,
    /// Held for `counted_for` and said so: its bytes count no longer.
    overdue: bool,
}

impl State {
    #[cfg(test)]
    fn reserved(&self) -> usize {
        self.holding
            .iter()
            .fold(0, |sum, holding| sum.saturating_add(holding.allowance))
    }

    /// How many flushes run as one asks at `now`, and the bytes that count
    /// against it: the allowances held for less than `counted_for`. A flush
    /// that passes it is said, once, and still runs.
    fn counted(&mut self, now: Instant, counted_for: Duration) -> (usize, usize) {
        let mut reserved = 0usize;
        for holding in &mut self.holding {
            let held = now.duration_since(holding.since);
            if held < counted_for {
                reserved = reserved.saturating_add(holding.allowance);
            } else if !std::mem::replace(&mut holding.overdue, true) {
                tracing::warn!(
                    collection = %holding.db,
                    held_secs = held.as_secs(),
                    allowance_mib = (holding.allowance >> 20) as u64,
                    "LSM memtable flush has held its allowance too long to be an upload \
                     that gets through: its bytes no longer count against other flushes"
                );
            }
        }
        (self.holding.len(), reserved)
    }
}

/// A flush's allowance, given back when it is dropped.
struct Held {
    line: Arc<Line>,
    id: u64,
}

impl Drop for Held {
    fn drop(&mut self) {
        let mut state = self.line.state();
        // How long the flush held its allowance: one encode and one try of
        // its upload. Nothing else says how long an L0 upload takes.
        if let Some(at) = state
            .holding
            .iter()
            .position(|holding| holding.id == self.id)
        {
            let held = state.holding.remove(at).since.elapsed();
            metrics::counter!("helix_lsm_flush_gate_held_total").increment(1);
            metrics::counter!("helix_lsm_flush_gate_held_ms_total")
                .increment(held.as_millis() as u64);
        }
        metrics::gauge!("helix_lsm_flush_gate_running").set(state.holding.len() as f64);
        drop(state);
        self.line.ended.notify_one();
    }
}

/// A flush of the database at `db` in the line, until it starts or gives up.
struct Waiting<'a> {
    line: &'a Line,
    db: &'a str,
}

impl<'a> Waiting<'a> {
    fn new(line: &'a Line, db: &'a str) -> Self {
        let mut state = line.state();
        state.waiting.push(db.to_string());
        metrics::gauge!("helix_lsm_flush_gate_waiting").set(state.waiting.len() as f64);
        Self { line, db }
    }
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        let mut state = self.line.state();
        if let Some(at) = state.waiting.iter().position(|db| db == self.db) {
            state.waiting.swap_remove(at);
        }
        metrics::gauge!("helix_lsm_flush_gate_waiting").set(state.waiting.len() as f64);
    }
}

impl Line {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn hold(self: &Arc<Self>, state: &mut State, db: &str, allowance: usize) -> Held {
        let id = state.next_id;
        state.next_id += 1;
        state.holding.push(Holding {
            id,
            db: db.to_string(),
            allowance,
            since: Instant::now(),
            overdue: false,
        });
        metrics::gauge!("helix_lsm_flush_gate_running").set(state.holding.len() as f64);
        Held {
            line: Arc::clone(self),
            id,
        }
    }

    /// The allowance, where [`may_start`] says so for the memory as it reads
    /// now; else that reading.
    fn start_if_room(self: &Arc<Self>, db: &str, allowance: usize) -> Result<Held, Option<usize>> {
        let free = (self.free)();
        let mut state = self.state();
        let (running, reserved) = state.counted(Instant::now(), self.counted_for);
        if may_start(free, reserved, running, allowance) {
            Ok(self.hold(&mut state, db, allowance))
        } else {
            Err(free)
        }
    }

    async fn admit(self: Arc<Self>, db: String, sst_bytes: usize) -> Held {
        let allowance = sst_bytes.saturating_mul(ALLOWANCE_TIMES);
        let asked = Instant::now();
        let waiting = Waiting::new(&self, &db);
        let mut waited = false;
        let _turn = match self.turn.try_lock() {
            Ok(turn) => turn,
            Err(_) => {
                waited = true;
                self.turn.lock().await
            }
        };
        loop {
            // In place before the look: a flush that ends between the look
            // and the wait is not missed.
            let ended = self.ended.notified();
            tokio::pin!(ended);
            ended.as_mut().enable();
            match self.start_if_room(&db, allowance) {
                Ok(held) => {
                    drop(waiting);
                    if waited {
                        self.say_waited(asked, sst_bytes);
                    }
                    return held;
                }
                Err(free) => {
                    waited = true;
                    self.warn_when_overdue(asked, free);
                }
            }
            tokio::select! {
                _ = &mut ended => {}
                _ = tokio::time::sleep(self.look_again) => {}
            }
        }
    }

    fn say_waited(&self, asked: Instant, sst_bytes: usize) {
        metrics::counter!("helix_lsm_flush_gate_waited_total").increment(1);
        let mut state = self.state();
        state.waited += 1;
        tracing::info!(
            waited_ms = asked.elapsed().as_millis() as u64,
            sst_mib = (sst_bytes >> 20) as u64,
            running = state.holding.len(),
            waiting = state.waiting.len(),
            "LSM memtable flush waited for free memory"
        );
    }

    fn warn_when_overdue(&self, asked: Instant, free: Option<usize>) {
        let now = Instant::now();
        let mut state = self.state();
        if !overdue(asked, state.warned, now) {
            return;
        }
        state.warned = Some(now);
        tracing::warn!(
            // Each takes no writes while its flush waits.
            waiting = state.waiting.len(),
            collections = ?state.waiting,
            running = state.holding.len(),
            free_mib = free.map(|bytes| (bytes >> 20) as u64),
            waited_secs = now.duration_since(asked).as_secs(),
            // A flush that never ends keeps its allowance: an upload that
            // cannot get through is the one this names.
            oldest_running_secs = state
                .holding
                .iter()
                .map(|holding| now.duration_since(holding.since).as_secs())
                .max(),
            "LSM memtable flushes have waited over a minute for free memory"
        );
    }
}

impl FlushGate {
    pub(crate) fn new(free: FreeMemory) -> Self {
        Self::timed(free, LOOK_AGAIN, OVERDUE)
    }

    /// [`Self::new`] with other times than [`LOOK_AGAIN`] and [`OVERDUE`], for
    /// a test: how often a waiting flush looks again, and for how long the
    /// bytes of one that holds its allowance count.
    pub(crate) fn timed(free: FreeMemory, look_again: Duration, counted_for: Duration) -> Self {
        Self(Arc::new(Line {
            turn: tokio::sync::Mutex::new(()),
            ended: tokio::sync::Notify::new(),
            state: Mutex::new(State::default()),
            free,
            look_again,
            counted_for,
        }))
    }

    /// How many failed uploads the gate was told of, and the last one.
    #[cfg(test)]
    pub(crate) fn upload_failures(&self) -> (u64, Option<(String, String, Duration)>) {
        let state = self.0.state();
        (state.upload_failures, state.last_upload_failure.clone())
    }

    #[cfg(test)]
    pub(crate) fn counts(&self) -> Counts {
        let state = self.0.state();
        Counts {
            running: state.holding.len(),
            reserved: state.reserved(),
            waiting: state.waiting.len(),
            started: state.next_id,
            waited: state.waited,
        }
    }
}

/// What a test reads off a gate.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Counts {
    /// Flushes holding an allowance.
    pub(crate) running: usize,
    /// The bytes they hold between them.
    pub(crate) reserved: usize,
    /// Flushes that have asked and not started.
    pub(crate) waiting: usize,
    /// Flushes that have started since the gate was made.
    pub(crate) started: u64,
    /// Those of them that had to wait.
    pub(crate) waited: u64,
}

#[cfg(test)]
#[path = "backend_lsm/tests/flush_gate_support.rs"]
pub(crate) mod test_support;

impl L0FlushGate for FlushGate {
    fn admit(&self, db: &str, sst_bytes: usize) -> BoxFuture<'_, Box<dyn Send>> {
        let (line, db) = (Arc::clone(&self.0), db.to_string());
        Box::pin(async move { Box::new(line.admit(db, sst_bytes).await) as Box<dyn Send> })
    }

    fn start(&self, db: &str, sst_bytes: usize) -> Box<dyn Send> {
        let mut state = self.0.state();
        Box::new(
            self.0
                .hold(&mut state, db, sst_bytes.saturating_mul(ALLOWANCE_TIMES)),
        )
    }

    /// SlateDB writes nothing for an upload it will try again itself, and
    /// the gate's gauges read 0 in that pause: this line and this count are
    /// all an operator sees of it. The memtable and its encoded SST stay in
    /// memory, the collection's later memtables wait behind it, and its rows
    /// stay acked in the WAL, where the next open reads them.
    fn upload_failed(&self, db: &str, error: &str, retry_in: Duration) {
        metrics::counter!("helix_lsm_flush_upload_failed_total").increment(1);
        let mut state = self.0.state();
        state.upload_failures += 1;
        #[cfg(test)]
        {
            state.last_upload_failure = Some((db.to_string(), error.to_string(), retry_in));
        }
        tracing::warn!(
            collection = %db,
            error,
            retry_secs = retry_in.as_secs_f64(),
            failed_uploads_since_start = state.upload_failures,
            "LSM memtable flush could not upload its SST and waits to try again: \
             its rows stay acked in the WAL, and its collection's later flushes wait behind it"
        );
    }
}
