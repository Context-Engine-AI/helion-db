//! Writer-only sweep that garbage-collects collections with no open handle.
//!
//! SlateDB collects garbage only inside an open writer `Db`. The writer keeps a
//! bounded number of collections open and closes one after an idle hour, while
//! garbage must be a day old before it may go, so whatever a collection stopped
//! referencing before it went quiet was never deleted unless the collection was
//! reopened. This sweep visits every collection the writer does not have loaded
//! and runs one pass for it without opening it.
//!
//! It is off unless the interval knob switches it on, and its passes are dry
//! runs unless the dry-run knob switches deletion on: in every collection, or
//! only in the collections the delete-prefixes knob names.
//!
//! Where a knob of its own asks for it (`HELIX_LSM_IDLE_GC_PUT_METADATA_SIDECARS`)
//! and a pass may delete, the sweep also puts a closed collection's metadata
//! sidecar into the object store when the store has none, from the writer's
//! local copy (`put_missing_metadata_sidecar`). A reader answers the info
//! route of a collection it has not loaded from that object alone.
//!
//! For the operator: keep the sweep off while a collection is copied or
//! restored into the swept root, and after an interrupted copy until it is
//! complete. A pass takes the highest manifest in a prefix for the
//! collection's last and deletes the SSTs it does not name. It stays away
//! from a prefix that was written within the quiet period (the SST minimum
//! age, or what `HELIX_LSM_IDLE_GC_QUIET_SECS` sets) and keeps the SSTs that
//! arrived within the SST minimum age, which covers a copy while it runs. A
//! copy that has been quiet for longer than the first, with SSTs that arrived
//! longer ago than the second and manifests still to come, looks like a
//! closed collection.

use super::*;
use crate::helix_engine::storage_core::backend_lsm::{
    closed_gc_deletes, closed_gc_plan, closed_gc_puts_metadata_sidecars, ClosedGcOutcome,
    ClosedGcPass, ClosedGcPlan, GateWanted,
};

/// Seconds between sweeps. The sweeper runs only when this is a positive whole
/// number; unset, `0`, empty or anything else leaves it off, so rolling out an
/// image never starts a sweep by itself.
const ENV_IDLE_GC_SWEEP_SECS: &str = "HELIX_LSM_IDLE_GC_SWEEP_SECS";

/// The role knob. Every value but `reader` makes a writer, an unset or
/// misspelt one included; the sweeper wants it spelt out.
const ENV_LSM_ROLE: &str = "HELIX_LSM_ROLE";

/// Longest wait before the first sweep: boot is over by then, and a writer that
/// restarts more often than the interval still sweeps.
const IDLE_GC_FIRST_SWEEP_DELAY: Duration = Duration::from_secs(600);

/// The sweep interval the knob's `value` asks for, or why it asks for none:
/// `unset`, `zero` or `unparsable`.
pub(super) fn sweep_interval(value: Option<&str>) -> Result<Duration, &'static str> {
    match value.map(|value| value.trim().parse::<u64>()) {
        None => Err("unset"),
        Some(Ok(0)) => Err("zero"),
        Some(Ok(secs)) => Ok(Duration::from_secs(secs)),
        Some(Err(_)) => Err("unparsable"),
    }
}

/// Whether a writer runs the sweeper, or why not. The interval knob has to ask
/// for it. The role knob has to say `writer` (`role_not_writer` otherwise): a
/// pod that is a writer only because nothing told it what it is must not
/// delete. And a pass has to be able to run at all: `plan` refuses without a
/// cache directory (`cache_dir_unset`), without minimum ages from the env
/// (`min_age_unset`), with one under the floor (`min_age_under_900s`) or with
/// a quiet period under it (`quiet_under_900s`).
pub(super) fn sweeper_start(
    value: Option<&str>,
    role: Option<&str>,
    plan: impl FnOnce() -> Result<ClosedGcPlan, &'static str>,
) -> Result<(Duration, ClosedGcPlan), &'static str> {
    let interval = sweep_interval(value)?;
    if !role.is_some_and(|role| role.trim().eq_ignore_ascii_case("writer")) {
        return Err("role_not_writer");
    }
    Ok((interval, plan()?))
}

/// Start the sweeper on an object-store writer, if [`sweeper_start`] allows.
/// Readers never collect: they own no prefix, and a pass writes manifests.
///
/// A writer logs one line either way. Started: the interval, the wait before
/// the first sweep, whether passes are dry runs, where they delete (`none`,
/// `all`, or the delete prefixes), whether the sweep collects WAL fences and
/// puts missing metadata sidecars, the most a pass deletes per
/// directory, the pass deadline, how long after a reader lease lapses a
/// collection is still left alone, how long after its last SST or WAL write,
/// and the age an object must reach in each directory. Not started: that the
/// sweeper is off, and why.
///
/// `dry_run=true` means no pass changes the object store: a dry run reads,
/// and logs what a real pass would delete. What a real pass writes before it
/// deletes (the latest manifest without its expired checkpoints, and the
/// `gc/*.boundary` objects) a dry run keeps in memory.
pub(crate) fn spawn_idle_gc_sweeper(
    manager: &Arc<CollectionManager>,
) -> Option<thread::JoinHandle<()>> {
    let backend = manager.config.storage_backend;
    if !backend.is_writer() || backend.is_lsm_in_memory() {
        return None;
    }
    let value = std::env::var(ENV_IDLE_GC_SWEEP_SECS).ok();
    let role = std::env::var(ENV_LSM_ROLE).ok();
    let start = sweeper_start(value.as_deref(), role.as_deref(), closed_gc_plan);
    let (interval, plan) = match start {
        Ok(start) => start,
        Err(reason @ ("unset" | "zero")) => {
            info!(
                knob = ENV_IDLE_GC_SWEEP_SECS,
                reason, "idle GC sweeper is off"
            );
            return None;
        }
        // Asked for, and refused.
        Err(reason) => {
            warn!(
                knob = ENV_IDLE_GC_SWEEP_SECS,
                value = ?value,
                role = ?role,
                reason,
                "idle GC sweeper is off"
            );
            return None;
        }
    };
    let first_sweep = interval.min(IDLE_GC_FIRST_SWEEP_DELAY);
    let manager = Arc::downgrade(manager);
    let spawned = thread::Builder::new()
        .name("helix-idle-gc".to_string())
        .spawn(move || {
            thread::sleep(first_sweep);
            let mut reached = None;
            while run_idle_gc_sweep(&manager, &mut reached).is_some() {
                thread::sleep(interval);
            }
        });
    match spawned {
        Ok(handle) => {
            info!(
                interval_secs = interval.as_secs(),
                first_sweep_secs = first_sweep.as_secs(),
                dry_run = plan.dry_run,
                deletes = %closed_gc_deletes(),
                wal_fences = plan.wal_fences,
                puts_metadata_sidecars = plan.puts_metadata_sidecars,
                max_deletes_per_directory = plan.max_deletes_per_directory,
                pass_deadline_secs = plan.deadline.as_secs(),
                lapsed_lease_grace_secs = plan.lapsed_lease_grace.as_secs(),
                quiet_secs = plan.quiet_period.as_secs(),
                manifest_min_age_secs = plan.manifest_min_age.as_secs(),
                wal_min_age_secs = plan.wal_min_age.as_secs(),
                compacted_min_age_secs = plan.compacted_min_age.as_secs(),
                compactions_min_age_secs = plan.compactions_min_age.as_secs(),
                "idle GC sweeper started"
            );
            Some(handle)
        }
        Err(error) => {
            warn!(%error, "failed to start the idle GC sweeper");
            None
        }
    }
}

/// What a sweep's passes found, and of that what the real passes handed to
/// deletion and what the dry runs left in place. The rest is what a real pass
/// found over its limit per directory.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct SweepTotals {
    pub(super) candidates: u64,
    pub(super) candidate_bytes: u64,
    pub(super) deleted: u64,
    pub(super) deleted_bytes: u64,
    /// Of `deleted`, the objects whose delete a pass that was cut short no
    /// longer started. Their bytes are still in `deleted_bytes`.
    pub(super) deletes_not_started: u64,
    pub(super) dry_run_candidates: u64,
    pub(super) dry_run_candidate_bytes: u64,
}

/// One sequential sweep over the object-store catalog. Returns its totals, or
/// `None` once the manager is gone.
///
/// `reached` is the last collection the previous sweep got to. This one starts
/// after it, so a sweep that ends early does not keep the rest of the catalog
/// waiting; with nothing reached yet (after a restart) it starts anywhere.
pub(super) fn run_idle_gc_sweep(
    manager: &Weak<CollectionManager>,
    reached: &mut Option<String>,
) -> Option<SweepTotals> {
    let backend = manager.upgrade()?.config.storage_backend;
    let started = Instant::now();
    // The children of this writer's own root prefix, in name order, already
    // without the names that carry a drop tombstone.
    let mut names = match backend_any::list_collection_catalog_from_env(backend) {
        Ok(catalog) => catalog.names,
        Err(error) => {
            warn!(%error, "idle GC sweep could not list collections");
            metrics::counter!("helix_lsm_idle_gc_sweeps_total", "outcome" => "list_error")
                .increment(1);
            return Some(SweepTotals::default());
        }
    };
    let first = match reached.as_deref() {
        Some(last) => names.partition_point(|name| name.as_str() <= last),
        None => {
            let anywhere = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |since| since.subsec_nanos() as usize);
            anywhere % names.len().max(1)
        }
    };
    names.rotate_left(first);
    let mut outcomes = std::collections::BTreeMap::<&'static str, u64>::new();
    // Closed collections by what was done about a metadata sidecar the object
    // store may lack (`put_missing_metadata_sidecar`); all four are logged.
    let mut metadata_sidecars = std::collections::BTreeMap::from(
        ["uploaded", "present", "no_local_copy", "failed"].map(|outcome| (outcome, 0_u64)),
    );
    let mut totals = SweepTotals::default();
    let mut outcome = "completed";
    let mut failures = 0;
    for name in &names {
        // Upgraded per collection so the loop ends between passes once the
        // manager is dropped.
        let manager = manager.upgrade()?;
        let (label, pass) = manager.gc_collection_if_idle(name);
        *reached = Some(name.clone());
        match label {
            // A pass that ran, failed or found the collection broken names it
            // in its own lines.
            "collected" | "timed_out" | "yielded" | "incomplete" | "error" | "panicked"
            | "not_intact" => {}
            // These keep their garbage until someone acts on the collection.
            "quarantined" => {
                info!(collection = %name, reason = label, "idle GC skipped a collection")
            }
            "lock_poisoned" => {
                warn!(collection = %name, reason = label, "idle GC skipped a collection")
            }
            _ => debug!(collection = %name, reason = label, "idle GC skipped a collection"),
        }
        metrics::counter!("helix_lsm_idle_gc_collections_total", "outcome" => label).increment(1);
        *outcomes.entry(label).or_default() += 1;
        if let Some(outcome) = pass.metadata_sidecar {
            metrics::counter!("helix_lsm_idle_gc_metadata_sidecars_total", "outcome" => outcome)
                .increment(1);
            *metadata_sidecars.entry(outcome).or_default() += 1;
        }
        // The pass logs what it found per directory; these are the totals.
        totals.candidates += pass.candidates;
        totals.candidate_bytes += pass.candidate_bytes;
        totals.deleted += pass.objects;
        totals.deleted_bytes += pass.bytes;
        totals.deletes_not_started += pass.deletes_not_started;
        if pass.dry_run {
            totals.dry_run_candidates += pass.candidates;
            totals.dry_run_candidate_bytes += pass.candidate_bytes;
        }
        metrics::counter!("helix_lsm_idle_gc_objects_total").increment(pass.objects);
        metrics::counter!("helix_lsm_idle_gc_bytes_total").increment(pass.bytes);
        match label {
            // Two passes that fail outright, with none succeeding in between,
            // is an object store failing for everyone. One cut short at its
            // deadline or by a request does not count: a large backlog and a
            // busy collection do that too.
            "error" | "incomplete" => {
                failures += 1;
                if failures == 2 {
                    warn!(collection = %name, "idle GC passes keep failing; ending this sweep");
                    outcome = "aborted";
                    break;
                }
            }
            // The object store answered for this one.
            "collected" | "tombstoned" | "no_manifest" | "lease_lapsing" | "empty"
            | "recently_written" | "not_intact" => failures = 0,
            _ => {}
        }
    }
    metrics::counter!("helix_lsm_idle_gc_sweeps_total", "outcome" => outcome).increment(1);
    info!(
        collections = names.len(),
        outcomes = ?outcomes,
        metadata_sidecars = ?metadata_sidecars,
        deletes = %closed_gc_deletes(),
        candidates = totals.candidates,
        candidate_bytes = totals.candidate_bytes,
        deleted = totals.deleted,
        deleted_bytes = totals.deleted_bytes,
        deletes_not_started = totals.deletes_not_started,
        dry_run_candidates = totals.dry_run_candidates,
        dry_run_candidate_bytes = totals.dry_run_candidate_bytes,
        outcome,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "idle GC sweep finished"
    );
    Some(totals)
}

impl CollectionManager {
    /// Run one closed-collection GC pass for `name` unless a handle, an open or
    /// a drop could overlap it. Returns the outcome label and what the pass
    /// handed to deletion.
    ///
    /// The collection's open gate is held for the whole pass. Create, cold
    /// open, drop, the purge retry and the drop-grace sweep all take that
    /// gate, so none of them can interleave with the pass's deletes or with
    /// the manifest and boundary objects it writes: a writer never opens onto
    /// a view the pass is deleting from, and nothing lands in a prefix after
    /// its purge. `try_lock`, because a held gate means one of them is already
    /// under way.
    ///
    /// The pass gives the gate back as soon as one of them wants it: it stops
    /// asking the object store for anything and returns.
    pub(super) fn gc_collection_if_idle(&self, name: &str) -> (&'static str, ClosedGcPass) {
        let none = ClosedGcPass::default();
        // The root graph engine keeps its own database open under the same root
        // prefix for the life of the process, outside this manager.
        if self.is_root_engine_name(name) {
            return ("root_engine", none);
        }
        // Listed names are raw object-store segments. One that is not its own
        // canonical segment lives under a re-encoded prefix SlateDB's collector
        // cannot parse table ids from, and `collection_path` would resolve
        // `a/b` or `..` to some other collection's prefix.
        if name.is_empty() || name.contains('/') || ObjPath::from(name).as_ref() != name {
            return ("unsupported_name", none);
        }
        // Not `open_gate`: that hands out an orphan gate when the registry is
        // poisoned, which would exclude nobody.
        let gate = match self.open_gates.write() {
            Ok(mut gates) => Arc::clone(
                gates
                    .entry(name.to_string())
                    .or_insert_with(|| Arc::new(Mutex::new(()))),
            ),
            Err(_) => return ("lock_poisoned", none),
        };
        let result = match gate.try_lock() {
            Ok(_guard) => {
                // Whoever wants this gate takes a clone of it from the registry
                // before waiting on it. The registry holds one and this call
                // holds one, so a third is someone waiting, or a drop's purge
                // worker, which keeps one for as long as it runs.
                let holders = Arc::downgrade(&gate);
                self.gc_collection_under_gate(name, Arc::new(move || holders.strong_count() > 2))
            }
            Err(_) => ("busy", none),
        };
        self.cleanup_open_gate_if_idle(name, &gate);
        result
    }

    fn gc_collection_under_gate(
        &self,
        name: &str,
        gate_wanted: GateWanted,
    ) -> (&'static str, ClosedGcPass) {
        let none = ClosedGcPass::default();
        let path = self.collection_path(name);
        // Every check fails closed: a poisoned registry proves nothing.
        if self
            .dropping
            .read()
            .map_or(true, |dropping| dropping.contains_key(name))
            || lsm_purge_pending(&path)
        {
            return ("dropping", none);
        }
        if self
            .collections
            .read()
            .map_or(true, |loaded| loaded.contains_key(name))
            || self.closing.read().map_or(true, |closing| {
                closing
                    .get(name)
                    .is_some_and(|weak| weak.strong_count() > 0)
            })
        {
            return ("loaded", none);
        }
        // A collection that failed to open or was marked degraded keeps every
        // object, old manifests included, for whoever repairs it.
        if OPEN_FAILURES
            .lock()
            .map_or(true, |failures| failures.contains_key(name))
            || path
                .to_str()
                .is_none_or(|path| HelixGraphStorage::read_degraded_marker(path).is_some())
        {
            return ("quarantined", none);
        }
        let started = Instant::now();
        let backend = self.config.storage_backend;
        let asked_for = Arc::clone(&gate_wanted);
        // Caught so a panic neither poisons the gate nor ends the sweeper.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            backend_any::gc_closed_lsm_collection_from_env(&path, backend, gate_wanted)
        }));
        let (label, pass) = match outcome {
            Ok(Ok(ClosedGcOutcome::Collected(pass))) => {
                metrics::histogram!("helix_lsm_idle_gc_pass_ms")
                    .record(started.elapsed().as_secs_f64() * 1000.0);
                // The one line a pass that was cut short leaves: SlateDB no
                // longer logs each delete such a pass did not start.
                if pass.yielded {
                    info!(
                        collection = %name,
                        deleting = pass.objects,
                        deletes_not_started = pass.deletes_not_started,
                        "idle GC pass gave way to a request for the collection"
                    );
                    ("yielded", pass)
                } else if pass.timed_out {
                    warn!(
                        collection = %name,
                        deleting = pass.objects,
                        deletes_not_started = pass.deletes_not_started,
                        "idle GC pass hit its deadline before finishing"
                    );
                    ("timed_out", pass)
                } else if pass.incomplete {
                    warn!(collection = %name, "idle GC pass did not get through every directory");
                    ("incomplete", pass)
                } else {
                    ("collected", pass)
                }
            }
            Ok(Ok(ClosedGcOutcome::Skipped(reason))) => (reason, none),
            Ok(Err(error)) => {
                warn!(collection = %name, %error, "idle GC pass could not start");
                ("error", none)
            }
            Err(_) => {
                warn!(collection = %name, "idle GC pass panicked");
                ("panicked", none)
            }
        };
        // A pass that ended by itself as one of these has read the drop
        // tombstone and then the manifest under the gate: the collection is
        // there, closed and not dropped (`empty`: its manifest names no SST,
        // so there was nothing to collect). The sweep's own settings start no
        // new kind of request: this takes its knob. A dry run writes nothing.
        let look = matches!(label, "collected" | "empty")
            && closed_gc_puts_metadata_sidecars()
            && closed_gc_deletes().allows(name);
        let metadata_sidecar = look
            .then(|| self.put_missing_metadata_sidecar(name, &path, asked_for))
            .flatten();
        (
            label,
            ClosedGcPass {
                metadata_sidecar,
                ..pass
            },
        )
    }

    /// Puts the writer's local copy of a closed collection's metadata sidecar
    /// into the object store when the store has none, and says which it was:
    /// `uploaded`, `present`, `no_local_copy` or `failed`; `None` when it gave
    /// the gate to something that asked for it. The caller holds the
    /// collection's open gate, and `gate_wanted` says when to give it back.
    ///
    /// A sidecar goes up from an open collection only, at its open and when
    /// its metadata changes. So a collection the writer has not opened since
    /// sidecars are kept in the object store has this local copy and no
    /// object, and a reader, which has nothing else to answer the info route
    /// from for a collection it has not loaded, answers 503 for it until a
    /// search opens it there.
    ///
    /// Never without the local copy. A drop removes it, and an object put
    /// under a purged prefix would list the name in the catalog again.
    fn put_missing_metadata_sidecar(
        &self,
        name: &str,
        path: &Path,
        gate_wanted: GateWanted,
    ) -> Option<&'static str> {
        let put = || -> Result<Option<&'static str>, String> {
            let bytes = match fs::read(HelixGraphStorage::metadata_sidecar_path(path)) {
                Ok(bytes) => bytes,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    return Ok(Some("no_local_copy"))
                }
                Err(error) => return Err(error.to_string()),
            };
            // A reader is handed these bytes: they have to read as a sidecar.
            sonic_rs::from_slice::<StorageMetadataSidecar>(&bytes)
                .map_err(|error| error.to_string())?;
            backend_any::put_missing_lsm_metadata_sidecar_from_env(
                path,
                self.config.storage_backend,
                bytes,
                gate_wanted,
            )
            .map(|put| put.map(|put| if put { "uploaded" } else { "present" }))
            .map_err(|error| error.to_string())
        };
        // Caught for the pass's reason: a panic here would poison the gate.
        let put = std::panic::catch_unwind(std::panic::AssertUnwindSafe(put))
            .unwrap_or_else(|_| Err("panicked".to_string()));
        put.unwrap_or_else(|error| {
            debug!(
                collection = %name,
                %error,
                "idle GC could not put a collection's missing metadata sidecar"
            );
            Some("failed")
        })
    }
}
