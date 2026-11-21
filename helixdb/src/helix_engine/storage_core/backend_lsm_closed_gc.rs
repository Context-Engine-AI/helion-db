//! One SlateDB garbage-collection pass over a collection this process holds
//! no handle for.
//!
//! SlateDB collects garbage only inside an open writer `Db`. The writer closes
//! a collection after an idle hour while garbage must be a day old to go, so
//! whatever a collection stopped referencing before it went quiet stayed in
//! the bucket until someone reopened it. The collection manager's idle sweep
//! calls this for each closed collection instead. No `Db` is opened: no writer
//! epoch bump, no fencing, no cache directory.
//!
//! A pass still writes. It appends a manifest version when it reaps an expired
//! checkpoint, and it advances `gc/manifest.boundary` and
//! `gc/compactions.boundary` before deleting. The caller must therefore keep
//! every open, drop and purge of the collection out until the pass returns.
//! A dry run makes the same pass over a store that keeps those writes in
//! memory ([`DryRunStore`]), so it sends the bucket nothing.
//!
//! Before any of that, a pass reads the latest manifest and leaves the
//! collection alone unless it can show three things. No reader lease in the
//! manifest has only just lapsed. Nothing was written under `compacted/` or
//! `wal/` for the quiet period (the SST minimum age, unless
//! [`ENV_IDLE_GC_QUIET_SECS`] sets another), so no writer and no copy is still
//! putting objects into the prefix. And the collection is whole, with every
//! SST and WAL object the manifest needs present in the bucket: for a broken
//! collection the old manifests and the objects they name are all a repair
//! can start from, and a pass would delete them.
//!
//! A copy can also begin once those checks are done. The SSTs it brings are
//! old by their names and in no manifest yet, so [`ClosedGcFilter`] keeps
//! every SST that has not been in the bucket for the SST minimum age.
//!
//! Every request goes through [`PassStore`], which stops the pass as soon as
//! anything else wants the collection's open gate.

use super::*;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::time::{Duration, Instant};

/// Most objects one pass deletes per directory (manifests, WAL, SSTs,
/// compaction state). The caller holds the collection's open gate for the
/// whole pass, so this bounds how long a backlog keeps the collection from
/// opening; later sweeps take the rest.
pub(super) const CLOSED_GC_MAX_DELETES_PER_DIRECTORY: usize = 2_000;

/// How long a pass may keep issuing object-store requests.
const CLOSED_GC_PASS_DEADLINE: Duration = Duration::from_secs(120);

/// How long after a checkpoint in the latest manifest expired a pass still
/// leaves the collection alone. Such a checkpoint may be the lease of a reader
/// that is only late refreshing it (its poll stalled, or the object store
/// did), or the compactor's on a collection the writer has just closed. A pass
/// strips expired checkpoints, and with them what kept that reader's files in
/// the bucket.
///
/// A checkpoint that outlives the pass holds none back: SlateDB strips only
/// expired ones, and keeps the manifest a live one names with every SST and
/// WAL object that manifest needs.
const CLOSED_GC_LAPSED_LEASE_GRACE: Duration = Duration::from_secs(20 * 60);

/// The least minimum age a pass accepts, for any directory, and the least
/// quiet period. A pass takes the manifest it reads for the one that counts,
/// and waits the quiet period after the last SST or WAL write before it trusts
/// that. Much shorter, and a copy or a second writer that merely paused would
/// pass for gone.
const CLOSED_GC_MIN_AGE_FLOOR: Duration = Duration::from_secs(15 * 60);

/// Test builds only: seconds to use for [`CLOSED_GC_MIN_AGE_FLOOR`] instead.
#[cfg(test)]
pub(super) const ENV_TEST_MIN_AGE_FLOOR_SECS: &str = "HELIX_LSM_TEST_IDLE_GC_MIN_AGE_FLOOR_SECS";

fn min_age_floor() -> Duration {
    // A test cannot wait fifteen minutes for its garbage to age.
    #[cfg(test)]
    if let Some(floor) = duration_secs_from_env(ENV_TEST_MIN_AGE_FLOOR_SECS) {
        return floor;
    }
    CLOSED_GC_MIN_AGE_FLOOR
}

/// How often a read in flight looks whether the pass has to stop.
const CLOSED_GC_STOP_POLL: Duration = Duration::from_millis(25);

/// The directories a pass collects: manifests, WAL, SSTs, compaction state.
/// SlateDB consults the filter once for each one it gets through, and once
/// more for the WAL fences where a pass collects those.
const CLOSED_GC_DIRECTORIES: u64 = 4;

/// Env knob: where passes delete.
/// - An explicit `0`, `false`, `no` or `off`: in every collection. With
///   [`ENV_IDLE_GC_DELETE_PREFIXES`] set as well, nowhere: one says
///   everywhere and the other names a part, and nobody means both.
/// - `prefixes`: only in the collections [`ENV_IDLE_GC_DELETE_PREFIXES`]
///   names.
/// - Unset, empty or anything else: nowhere. Every pass is a dry run, so
///   deleting takes two deliberate settings: this one and the sweep interval.
///
/// An image from before `prefixes` existed takes that word for "anything
/// else" and makes dry runs. It takes an off value for deleting in every
/// collection, whether the prefixes are set or not.
///
/// A dry run logs what a real pass would delete. It deletes nothing and
/// writes nothing to the object store ([`DryRunStore`]).
pub(super) const ENV_IDLE_GC_DRY_RUN: &str = "HELIX_LSM_IDLE_GC_DRY_RUN";

/// Env knob: the collections passes delete in while the dry-run knob says
/// `prefixes`, as name prefixes separated by commas (`admin_,acme_`). The
/// name is the last segment of a collection's path, so `admin_` covers
/// `prod/admin_docs`. A pass over any other collection is a dry run.
///
/// Unset, empty, with an empty item, or with a character outside
/// `A-Z a-z 0-9 _ - .`, it covers no collection: neither a typo nor a
/// misspelt key may widen what is deleted. Next to an off value on the
/// dry-run knob its presence alone, whatever it holds, stops every delete.
pub(super) const ENV_IDLE_GC_DELETE_PREFIXES: &str = "HELIX_LSM_IDLE_GC_DELETE_PREFIXES";

/// Env knob: an explicit `1`, `true`, `yes` or `on` makes a pass collect the
/// zero-byte WAL fence objects below the manifest's replay position as well,
/// under the WAL's minimum age and every other rule of a pass. A writer puts
/// one at each open and SlateDB's WAL task takes only objects that hold data,
/// so they pile up for as long as the collection lives. Unset or anything
/// else, a pass leaves them alone and does not list the WAL a second time.
///
/// A fence stops a writer that has been superseded from writing at its
/// position. It may go once no such writer can still be up: the WAL minimum
/// age has to be longer than a superseded writer could stay alive without
/// writing, which is why SlateDB's own fence collector only logs.
pub(super) const ENV_IDLE_GC_WAL_FENCES: &str = "HELIX_LSM_IDLE_GC_WAL_FENCES";

/// Env knob: the quiet period, in seconds. A pass leaves a collection alone
/// until nothing has been written under its `compacted/` or `wal/` for that
/// long. Unset, empty, `0` or not a number, it is the SST minimum age, as it
/// was before the knob existed; below [`CLOSED_GC_MIN_AGE_FLOOR`] no pass runs
/// (`quiet_under_900s`).
///
/// The quiet period decides when a pass may look, not what goes. Every object
/// still has to reach its directory's minimum age, and an SST to have been in
/// the bucket for the SST minimum age. With a shorter quiet period a pass
/// takes, from a collection written since, no more than that collection's own
/// collector would take at its next open.
pub(super) const ENV_IDLE_GC_QUIET_SECS: &str = "HELIX_LSM_IDLE_GC_QUIET_SECS";

/// Env knob: an explicit `1`, `true`, `yes` or `on` makes the sweep put a
/// closed collection's metadata sidecar into the object store where the store
/// holds none, from the writer's local copy, in every collection a pass may
/// delete in. Unset or anything else, the sweep asks the object store nothing
/// about a sidecar and puts none.
///
/// Only an open collection puts its sidecar. One the writer has not opened
/// since sidecars are kept in the object store has it on the writer's disk
/// alone, and a reader, which answers the info route of a collection it has
/// not loaded from the object, answers 503 until a search opens it there.
pub(super) const ENV_IDLE_GC_PUT_METADATA_SIDECARS: &str =
    "HELIX_LSM_IDLE_GC_PUT_METADATA_SIDECARS";

/// Whether the sweep puts missing metadata sidecars
/// ([`ENV_IDLE_GC_PUT_METADATA_SIDECARS`]).
pub(crate) fn closed_gc_puts_metadata_sidecars() -> bool {
    env_bool_opt(ENV_IDLE_GC_PUT_METADATA_SIDECARS) == Some(true)
}

/// Where passes delete. Shown as `none`, `all` or the prefixes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ClosedGcDeletes {
    Nowhere,
    Everywhere,
    /// In the collections whose name starts with one of these.
    Under(Vec<String>),
}

impl ClosedGcDeletes {
    /// Whether a pass over the collection at `collection_prefix` deletes.
    pub(crate) fn allows(&self, collection_prefix: &str) -> bool {
        let name = collection_prefix.rsplit('/').next().unwrap_or_default();
        match self {
            Self::Nowhere => false,
            Self::Everywhere => true,
            Self::Under(prefixes) => prefixes
                .iter()
                .any(|prefix| name.starts_with(prefix.as_str())),
        }
    }
}

impl std::fmt::Display for ClosedGcDeletes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nowhere => f.write_str("none"),
            Self::Everywhere => f.write_str("all"),
            Self::Under(prefixes) => f.write_str(&prefixes.join(",")),
        }
    }
}

/// Where passes delete on what the env gives: [`ENV_IDLE_GC_DRY_RUN`] and,
/// when that says `prefixes`, [`ENV_IDLE_GC_DELETE_PREFIXES`].
pub(crate) fn closed_gc_deletes() -> ClosedGcDeletes {
    if env_bool_opt(ENV_IDLE_GC_DRY_RUN) == Some(false) {
        return if std::env::var_os(ENV_IDLE_GC_DELETE_PREFIXES).is_some() {
            ClosedGcDeletes::Nowhere
        } else {
            ClosedGcDeletes::Everywhere
        };
    }
    if !std::env::var(ENV_IDLE_GC_DRY_RUN)
        .is_ok_and(|knob| knob.trim().eq_ignore_ascii_case("prefixes"))
    {
        return ClosedGcDeletes::Nowhere;
    }
    // Unset or not UTF-8 reads as empty, which is malformed.
    let value = std::env::var(ENV_IDLE_GC_DELETE_PREFIXES).unwrap_or_default();
    let prefixes: Vec<&str> = value.split(',').map(str::trim).collect();
    let well_formed = prefixes.iter().all(|prefix| {
        !prefix.is_empty()
            && prefix
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
    });
    if well_formed {
        ClosedGcDeletes::Under(prefixes.into_iter().map(String::from).collect())
    } else {
        ClosedGcDeletes::Nowhere
    }
}

/// The in-process collector's options without the two tasks that never delete
/// here, the age an object must reach in each directory (manifests, WAL, SSTs,
/// compaction state), and the quiet period: [`ENV_IDLE_GC_QUIET_SECS`], else
/// the SST minimum age. Or why no pass may run on what the env gives:
/// `min_age_unset` unless it sets every directory's age (left to SlateDB's own
/// default, a pass would delete what is five minutes old), `min_age_under_900s`
/// when one of them is below [`CLOSED_GC_MIN_AGE_FLOOR`], `quiet_under_900s`
/// when the quiet period is.
fn closed_gc_options() -> Result<(GarbageCollectorOptions, [Duration; 4], Duration), &'static str> {
    let mut options = gc_options_from_env().ok_or("min_age_unset")?;
    let default_min_age = duration_secs_from_env(ENV_GC_MIN_AGE_SECS);
    let every_age_is_set = [
        ENV_GC_MANIFEST_MIN_AGE_SECS,
        ENV_GC_WAL_MIN_AGE_SECS,
        ENV_GC_COMPACTED_MIN_AGE_SECS,
        ENV_GC_COMPACTIONS_MIN_AGE_SECS,
    ]
    .into_iter()
    .all(|key| duration_secs_from_env(key).or(default_min_age).is_some());
    if !every_age_is_set {
        return Err("min_age_unset");
    }
    let [Some(manifests), Some(wal), Some(ssts), Some(compactions)] = [
        options.manifest_options,
        options.wal_options,
        options.compacted_options,
        options.compactions_options,
    ] else {
        return Err("min_age_unset");
    };
    let min_ages = [
        manifests.min_age,
        wal.min_age,
        ssts.min_age,
        compactions.min_age,
    ];
    if min_ages.iter().any(|age| *age < min_age_floor()) {
        return Err("min_age_under_900s");
    }
    let quiet_period = duration_secs_from_env(ENV_IDLE_GC_QUIET_SECS).unwrap_or(ssts.min_age);
    if quiet_period < min_age_floor() {
        return Err("quiet_under_900s");
    }
    // Helion never clones a database. The fence task lists the WAL a second
    // time, so it runs only where the knob asks for it; then it deletes like
    // the WAL task (SlateDB's own setting for it is log-only).
    options.wal_fence_options = (env_bool_opt(ENV_IDLE_GC_WAL_FENCES) == Some(true)).then_some(
        GarbageCollectorDirectoryOptions {
            dry_run: false,
            ..wal
        },
    );
    options.detach_options = None;
    Ok((options, min_ages, quiet_period))
}

/// What every pass applies. The sweeper logs it once when it starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ClosedGcPlan {
    /// No pass deletes anywhere.
    pub(crate) dry_run: bool,
    /// Passes collect zero-byte WAL fences too ([`ENV_IDLE_GC_WAL_FENCES`]).
    pub(crate) wal_fences: bool,
    /// The sweep puts missing metadata sidecars
    /// ([`ENV_IDLE_GC_PUT_METADATA_SIDECARS`]).
    pub(crate) puts_metadata_sidecars: bool,
    pub(crate) max_deletes_per_directory: usize,
    pub(crate) deadline: Duration,
    pub(crate) lapsed_lease_grace: Duration,
    /// How long nothing may have been written under `compacted/` or `wal/`.
    pub(crate) quiet_period: Duration,
    /// The age an object must reach before it may go.
    pub(crate) manifest_min_age: Duration,
    pub(crate) wal_min_age: Duration,
    pub(crate) compacted_min_age: Duration,
    pub(crate) compactions_min_age: Duration,
}

/// What every pass will apply, or why no pass can run in this process: the
/// refusals [`gc_closed_collection_until`] makes before it reads anything.
pub(crate) fn closed_gc_plan() -> Result<ClosedGcPlan, &'static str> {
    if cache_root_from_env().is_none() {
        return Err("cache_dir_unset");
    }
    let (
        options,
        [manifest_min_age, wal_min_age, compacted_min_age, compactions_min_age],
        quiet_period,
    ) = closed_gc_options()?;
    Ok(ClosedGcPlan {
        dry_run: closed_gc_deletes() == ClosedGcDeletes::Nowhere,
        wal_fences: options.wal_fence_options.is_some(),
        puts_metadata_sidecars: closed_gc_puts_metadata_sidecars(),
        max_deletes_per_directory: CLOSED_GC_MAX_DELETES_PER_DIRECTORY,
        deadline: CLOSED_GC_PASS_DEADLINE,
        lapsed_lease_grace: CLOSED_GC_LAPSED_LEASE_GRACE,
        quiet_period,
        manifest_min_age,
        wal_min_age,
        compacted_min_age,
        compactions_min_age,
    })
}

/// What one pass did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ClosedGcPass {
    /// Objects SlateDB found unreferenced and old enough to delete.
    pub(crate) candidates: u64,
    pub(crate) candidate_bytes: u64,
    /// The candidates handed back for deletion: none in a dry run, at most
    /// [`CLOSED_GC_MAX_DELETES_PER_DIRECTORY`] per directory otherwise. A
    /// delete that then fails is logged by SlateDB, not returned, and one that
    /// a pass cut short never gets to is counted in `deletes_not_started`;
    /// either object is a candidate again in the next pass, so for a pass that
    /// was cut short this is an upper bound.
    pub(crate) objects: u64,
    pub(crate) bytes: u64,
    /// The pass was a dry run: passes delete nowhere, or only under prefixes
    /// that do not cover the collection.
    pub(crate) dry_run: bool,
    /// The pass was cut short at its deadline: the object store is slow or
    /// failing, or the backlog is large.
    pub(crate) timed_out: bool,
    /// The pass was cut short because something else wanted the collection.
    pub(crate) yielded: bool,
    /// Nothing cut the pass short, yet it did not get through every directory.
    pub(crate) incomplete: bool,
    /// Of `objects`, the deletes a pass that was cut short no longer started.
    pub(crate) deletes_not_started: u64,
    /// What the sweep did after the pass about a metadata sidecar the object
    /// store may lack: `uploaded`, `present`, `no_local_copy` or `failed`.
    /// The collection manager sets it; `None` where it did not look.
    pub(crate) metadata_sidecar: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClosedGcOutcome {
    Collected(ClosedGcPass),
    /// Nothing was written or deleted; the reason is a metric label.
    Skipped(&'static str),
}

/// Counts what a pass found, caps what it may delete, and keeps the SSTs that
/// arrived less than the SST minimum age ago.
#[derive(Default)]
pub(super) struct ClosedGcFilter {
    collection: String,
    dry_run: bool,
    sst_min_age: Duration,
    candidates: AtomicU64,
    candidate_bytes: AtomicU64,
    objects: AtomicU64,
    bytes: AtomicU64,
    /// Directories SlateDB got as far as asking about.
    directories: AtomicU64,
}

impl ClosedGcFilter {
    pub(super) fn new(collection: &str, dry_run: bool, sst_min_age: Duration) -> Self {
        Self {
            collection: collection.to_string(),
            dry_run,
            sst_min_age,
            ..Self::default()
        }
    }
}

// `GcFilter` is an `async_trait`; this is its desugared signature.
impl slatedb::GcFilter for ClosedGcFilter {
    fn filter<'a, 'b>(
        &'a self,
        candidates: HashSet<slatedb::ObjectMetadata>,
    ) -> std::pin::Pin<Box<dyn Future<Output = HashSet<slatedb::ObjectMetadata>> + Send + 'b>>
    where
        'a: 'b,
        Self: 'b,
    {
        self.directories.fetch_add(1, Ordering::Relaxed);
        // SlateDB asks once per directory, so one candidate names it.
        let directory = candidates
            .iter()
            .next()
            .and_then(|object| object.location.as_ref().rsplit('/').nth(1))
            .unwrap_or_default()
            .to_string();
        // The fence task asks about the WAL directory too, and only about its
        // zero-byte objects, which the WAL task never asks about.
        let directory = if directory == "wal" && candidates.iter().all(|object| object.size == 0) {
            "wal_fence".to_string()
        } else {
            directory
        };
        // SlateDB ages a manifest, a WAL object and a compactions file by
        // LastModified, each against its own minimum age, but an SST by the
        // time in its name. So only an SST can be a candidate moments after it
        // arrived: one a copy put into the prefix after this pass looked, old
        // by its name and in no manifest until the copy has brought that too.
        let now = chrono::Utc::now();
        let (kept, candidates): (HashSet<_>, HashSet<_>) =
            candidates.into_iter().partition(|object| {
                directory == "compacted"
                    && now
                        .signed_duration_since(object.last_modified)
                        .to_std()
                        .map_or(true, |in_bucket_for| in_bucket_for < self.sst_min_age)
            });
        if !kept.is_empty() {
            tracing::warn!(
                collection = %self.collection,
                kept = kept.len() as u64,
                kept_bytes = kept.iter().map(|object| object.size).sum::<u64>(),
                dry_run = self.dry_run,
                "closed-collection GC kept unreferenced SSTs written less than the minimum age ago: the collection may be the target of a copy"
            );
        }
        let found = candidates.len() as u64;
        let found_bytes = candidates.iter().map(|object| object.size).sum::<u64>();
        let limit = if self.dry_run {
            0
        } else {
            CLOSED_GC_MAX_DELETES_PER_DIRECTORY
        };
        let allowed: HashSet<slatedb::ObjectMetadata> =
            candidates.into_iter().take(limit).collect();
        let deleting = allowed.len() as u64;
        let deleting_bytes = allowed.iter().map(|object| object.size).sum::<u64>();
        self.candidates.fetch_add(found, Ordering::Relaxed);
        self.candidate_bytes
            .fetch_add(found_bytes, Ordering::Relaxed);
        self.objects.fetch_add(deleting, Ordering::Relaxed);
        self.bytes.fetch_add(deleting_bytes, Ordering::Relaxed);
        if found > 0 {
            tracing::info!(
                collection = %self.collection,
                directory = %directory,
                candidates = found,
                candidate_bytes = found_bytes,
                deleting,
                deleting_bytes,
                dry_run = self.dry_run,
                "closed-collection GC found unreferenced objects"
            );
        }
        Box::pin(async move { allowed })
    }
}

/// Says whether anything else wants the collection's open gate.
pub(crate) type GateWanted = Arc<dyn Fn() -> bool + Send + Sync>;

const STOPPED_FOR_THE_GATE: u8 = 1;
const STOPPED_AT_THE_DEADLINE: u8 = 2;

/// When a pass has to stop asking the object store for anything.
struct PassStop {
    deadline: Instant,
    gate_wanted: GateWanted,
    /// Why the first call was refused or abandoned; 0 while none has been.
    stopped: AtomicU8,
    /// Deletes asked for after the pass had to stop, each refused before
    /// anything went to the bucket.
    deletes_not_started: AtomicU64,
}

impl PassStop {
    /// The refusal for a call the pass may no longer make, if it may not.
    fn refusal(&self) -> Option<ObjectStoreError> {
        let (why, text) = if (self.gate_wanted)() {
            (
                STOPPED_FOR_THE_GATE,
                "closed-collection GC pass gave way to a request for the collection",
            )
        } else if Instant::now() >= self.deadline {
            (
                STOPPED_AT_THE_DEADLINE,
                "closed-collection GC pass ran past its deadline",
            )
        } else {
            return None;
        };
        // The first reason stands.
        let _ = self
            .stopped
            .compare_exchange(0, why, Ordering::Relaxed, Ordering::Relaxed);
        // One of the few errors SlateDB's retry wrapper gives up on.
        Some(ObjectStoreError::NotSupported {
            source: text.into(),
        })
    }

    /// Resolves with the refusal once the pass has to stop.
    async fn reached(&self) -> ObjectStoreError {
        loop {
            if let Some(refusal) = self.refusal() {
                return refusal;
            }
            tokio::time::sleep(CLOSED_GC_STOP_POLL).await;
        }
    }
}

/// The object store one pass sees. Once something else wants the collection's
/// open gate, or the deadline has passed, it answers nothing more.
///
/// The pass runs under that gate, so every open, create and drop of the
/// collection waits for it. SlateDB retries a failed call forever and
/// object_store retries each request for minutes underneath, so without this a
/// slow bucket or a large backlog would hold those requests for as long.
///
/// A read still waiting for its answer is abandoned when the pass has to stop;
/// a body already arriving is read to its end. A write in flight is left to
/// finish: a put or delete that landed after the gate was released could hit a
/// collection dropped and recreated in between. A listing that is stopped ends
/// in an error, never just short, so it cannot pass for a complete one.
pub(super) struct PassStore {
    inner: Arc<dyn ObjectStore>,
    stop: Arc<PassStop>,
}

impl PassStore {
    pub(super) fn new(
        inner: Arc<dyn ObjectStore>,
        deadline: Instant,
        gate_wanted: GateWanted,
    ) -> Self {
        Self {
            inner,
            stop: Arc::new(PassStop {
                deadline,
                gate_wanted,
                stopped: AtomicU8::new(0),
                deletes_not_started: AtomicU64::new(0),
            }),
        }
    }

    /// Deletes this store refused because the pass had already stopped.
    #[cfg(test)]
    pub(super) fn deletes_not_started(&self) -> u64 {
        self.stop.deletes_not_started.load(Ordering::Relaxed)
    }

    async fn read<T>(
        &self,
        call: impl Future<Output = object_store::Result<T>>,
    ) -> object_store::Result<T> {
        tokio::select! {
            biased;
            refusal = self.stop.reached() => Err(refusal),
            result = call => result,
        }
    }
}

impl std::fmt::Debug for PassStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PassStore({:?})", self.inner)
    }
}

impl std::fmt::Display for PassStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PassStore({})", self.inner)
    }
}

type Boxed<'a, T> = std::pin::Pin<Box<dyn Future<Output = object_store::Result<T>> + Send + 'a>>;

fn refused<'a, T: 'a>(error: ObjectStoreError) -> Boxed<'a, T> {
    Box::pin(async move { Err(error) })
}

impl ObjectStore for PassStore {
    fn put_opts<'a, 'b, 'c>(
        &'a self,
        location: &'b ObjPath,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> Boxed<'c, object_store::PutResult>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        match self.stop.refusal() {
            None => self.inner.put_opts(location, payload, opts),
            Some(refusal) => refused(refusal),
        }
    }

    fn put_multipart_opts<'a, 'b, 'c>(
        &'a self,
        location: &'b ObjPath,
        opts: object_store::PutMultipartOptions,
    ) -> Boxed<'c, Box<dyn object_store::MultipartUpload>>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        match self.stop.refusal() {
            None => self.inner.put_multipart_opts(location, opts),
            Some(refusal) => refused(refusal),
        }
    }

    fn get_opts<'a, 'b, 'c>(
        &'a self,
        location: &'b ObjPath,
        options: object_store::GetOptions,
    ) -> Boxed<'c, object_store::GetResult>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        Box::pin(self.read(self.inner.get_opts(location, options)))
    }

    fn delete_stream(
        &self,
        locations: stream::BoxStream<'static, object_store::Result<ObjPath>>,
    ) -> stream::BoxStream<'static, object_store::Result<ObjPath>> {
        match self.stop.refusal() {
            None => self.inner.delete_stream(locations),
            Some(refusal) => {
                // SlateDB asks for one object a call.
                self.stop
                    .deletes_not_started
                    .fetch_add(1, Ordering::Relaxed);
                stream::once(async move { Err(refusal) }).boxed()
            }
        }
    }

    fn list(
        &self,
        prefix: Option<&ObjPath>,
    ) -> stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        let stop = Arc::clone(&self.stop);
        stream::unfold(Some(self.inner.list(prefix)), move |listing| {
            let stop = Arc::clone(&stop);
            async move {
                let Some(mut listing) = listing else {
                    return None;
                };
                tokio::select! {
                    biased;
                    // The error is the listing's last item.
                    refusal = stop.reached() => Some((Err(refusal), None)),
                    object = listing.next() => object.map(|object| (object, Some(listing))),
                }
            }
        })
        .boxed()
    }

    fn list_with_delimiter<'a, 'b, 'c>(
        &'a self,
        prefix: Option<&'b ObjPath>,
    ) -> Boxed<'c, object_store::ListResult>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        Box::pin(self.read(self.inner.list_with_delimiter(prefix)))
    }

    fn copy_opts<'a, 'b, 'c, 'd>(
        &'a self,
        from: &'b ObjPath,
        to: &'c ObjPath,
        options: object_store::CopyOptions,
    ) -> Boxed<'d, ()>
    where
        'a: 'd,
        'b: 'd,
        'c: 'd,
        Self: 'd,
    {
        match self.stop.refusal() {
            None => self.inner.copy_opts(from, to, options),
            Some(refusal) => refused(refusal),
        }
    }
}

/// The object store a dry run gives SlateDB in place of the bucket. Reads come
/// from the bucket; whatever the pass writes stays in memory, where its own
/// later reads find it.
///
/// SlateDB writes before it asks which objects may go: the latest manifest
/// without its expired checkpoints, then the GC boundaries. Refusing those
/// writes would end the pass before it had counted anything. Dropping them
/// would leave out of the count whatever only a lapsed checkpoint still
/// protects, which includes the inputs of the collection's last compaction.
/// Kept in memory, they give the counts a real pass would act on, and nothing
/// reaches the bucket: no method here forwards a put, a delete or a copy.
#[derive(Debug)]
pub(super) struct DryRunStore {
    bucket: Arc<dyn ObjectStore>,
    written: InMemory,
}

impl DryRunStore {
    pub(super) fn new(bucket: Arc<dyn ObjectStore>) -> Self {
        Self {
            bucket,
            written: InMemory::new(),
        }
    }
}

impl std::fmt::Display for DryRunStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DryRunStore({})", self.bucket)
    }
}

fn dry_run_refuses(operation: &str) -> ObjectStoreError {
    ObjectStoreError::NotSupported {
        source: format!("a dry-run GC pass does not {operation}").into(),
    }
}

impl ObjectStore for DryRunStore {
    fn put_opts<'a, 'b, 'c>(
        &'a self,
        location: &'b ObjPath,
        payload: object_store::PutPayload,
        mut opts: object_store::PutOptions,
    ) -> Boxed<'c, object_store::PutResult>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        Box::pin(async move {
            // A conditional put is judged against what the pass sees at
            // `location`: its own earlier write, else the object in the bucket.
            if self.written.head(location).await.is_err() {
                let in_bucket = match self.bucket.head(location).await {
                    Ok(object) => Some(object),
                    Err(ObjectStoreError::NotFound { .. }) => None,
                    Err(error) => return Err(error),
                };
                match (&opts.mode, in_bucket) {
                    (object_store::PutMode::Create, Some(_)) => {
                        return Err(ObjectStoreError::AlreadyExists {
                            path: location.to_string(),
                            source: "the bucket holds this object".into(),
                        });
                    }
                    (object_store::PutMode::Update(expected), object)
                        if object.as_ref().is_none_or(|object| {
                            object.e_tag != expected.e_tag || object.version != expected.version
                        }) =>
                    {
                        return Err(ObjectStoreError::Precondition {
                            path: location.to_string(),
                            source: "the bucket holds another version of this object".into(),
                        });
                    }
                    _ => {}
                }
                // The pass's first write here: memory holds nothing to compare.
                opts.mode = object_store::PutMode::Overwrite;
            }
            self.written.put_opts(location, payload, opts).await
        })
    }

    fn put_multipart_opts<'a, 'b, 'c>(
        &'a self,
        _location: &'b ObjPath,
        _opts: object_store::PutMultipartOptions,
    ) -> Boxed<'c, Box<dyn object_store::MultipartUpload>>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        refused(dry_run_refuses("upload in parts"))
    }

    fn get_opts<'a, 'b, 'c>(
        &'a self,
        location: &'b ObjPath,
        options: object_store::GetOptions,
    ) -> Boxed<'c, object_store::GetResult>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        Box::pin(async move {
            match self.written.get_opts(location, options.clone()).await {
                Err(ObjectStoreError::NotFound { .. }) => {
                    self.bucket.get_opts(location, options).await
                }
                written => written,
            }
        })
    }

    fn delete_stream(
        &self,
        _locations: stream::BoxStream<'static, object_store::Result<ObjPath>>,
    ) -> stream::BoxStream<'static, object_store::Result<ObjPath>> {
        stream::once(async { Err(dry_run_refuses("delete")) }).boxed()
    }

    fn list(
        &self,
        prefix: Option<&ObjPath>,
    ) -> stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        let written = self.written.clone();
        // The pass's own version of an object stands in for the bucket's.
        let in_bucket = self.bucket.list(prefix).try_filter(move |object| {
            let written = written.clone();
            let location = object.location.clone();
            async move { written.head(&location).await.is_err() }
        });
        in_bucket.chain(self.written.list(prefix)).boxed()
    }

    fn list_with_delimiter<'a, 'b, 'c>(
        &'a self,
        _prefix: Option<&'b ObjPath>,
    ) -> Boxed<'c, object_store::ListResult>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        // Unused by a GC pass, and a bucket-only answer would hide its writes.
        refused(dry_run_refuses("list by delimiter"))
    }

    fn copy_opts<'a, 'b, 'c, 'd>(
        &'a self,
        _from: &'b ObjPath,
        _to: &'c ObjPath,
        _options: object_store::CopyOptions,
    ) -> Boxed<'d, ()>
    where
        'a: 'd,
        'b: 'd,
        'c: 'd,
        Self: 'd,
    {
        refused(dry_run_refuses("copy"))
    }
}

/// Run one garbage-collection pass over the closed collection at
/// `collection_prefix`, with the options the in-process collector uses
/// ([`closed_gc_options`]).
///
/// The caller must hold the collection's open gate and have checked that the
/// collection is neither loaded nor being dropped; see the module docs.
/// `gate_wanted` says when the caller wants the gate back.
pub(crate) fn gc_closed_collection_with_store(
    store: Arc<dyn ObjectStore>,
    collection_prefix: &str,
    gate_wanted: GateWanted,
) -> Result<ClosedGcOutcome, BackendError> {
    gc_closed_collection_until(
        store,
        collection_prefix,
        Instant::now() + CLOSED_GC_PASS_DEADLINE,
        CLOSED_GC_LAPSED_LEASE_GRACE,
        gate_wanted,
    )
}

pub(crate) fn gc_closed_s3_collection(
    config: &S3StoreConfig<'_>,
    collection_prefix: &str,
    gate_wanted: GateWanted,
) -> Result<ClosedGcOutcome, BackendError> {
    let store = build_s3_store(config)?;
    gc_closed_collection_with_store(store, collection_prefix, gate_wanted)
}

pub(crate) fn put_missing_s3_metadata_sidecar(
    config: &S3StoreConfig<'_>,
    collection_prefix: &str,
    bytes: Vec<u8>,
    gate_wanted: GateWanted,
) -> Result<Option<bool>, BackendError> {
    let store = build_s3_store(config)?;
    put_missing_metadata_sidecar_with_store(store, collection_prefix, bytes, gate_wanted)
}

/// Puts `bytes` as the closed collection's metadata sidecar unless the store
/// holds one: `Some(true)` when this call wrote it, `Some(false)` when one is
/// there, `None` when it had to stop first. One HEAD, and a create-only put
/// where that finds none, so a sidecar somebody put since the look stays.
///
/// The caller holds the collection's open gate, as for a pass, so the calls
/// go through a [`PassStore`] as a pass's own do: the look is given up once
/// something else wants the gate, and a put that has started runs to its end.
pub(crate) fn put_missing_metadata_sidecar_with_store(
    store: Arc<dyn ObjectStore>,
    collection_prefix: &str,
    bytes: Vec<u8>,
    gate_wanted: GateWanted,
) -> Result<Option<bool>, BackendError> {
    let store = PassStore::new(store, Instant::now() + CLOSED_GC_PASS_DEADLINE, gate_wanted);
    let stop = Arc::clone(&store.stop);
    let store: Arc<dyn ObjectStore> = Arc::new(store);
    let path = metadata_sidecar_object_path(collection_prefix);
    let put = block_on_lsm(&shared_lsm_handle(), async move {
        match store.head(&path).await {
            Ok(_) => return Ok(false),
            Err(ObjectStoreError::NotFound { .. }) => {}
            Err(err) => return Err(io(err)),
        }
        let create_only = object_store::PutOptions::from(object_store::PutMode::Create);
        match store.put_opts(&path, bytes.into(), create_only).await {
            Ok(_) => Ok(true),
            Err(ObjectStoreError::AlreadyExists { .. }) => Ok(false),
            Err(err) => Err(io(err)),
        }
    });
    match put {
        Ok(put) => Ok(Some(put)),
        // A call the store refused comes back as an error. The stop is the
        // outcome then, not the error.
        Err(_) if stop.stopped.load(Ordering::Relaxed) != 0 => Ok(None),
        Err(error) => Err(error),
    }
}

pub(super) fn gc_closed_collection_until(
    store: Arc<dyn ObjectStore>,
    collection_prefix: &str,
    deadline: Instant,
    lapsed_lease_grace: Duration,
    gate_wanted: GateWanted,
) -> Result<ClosedGcOutcome, BackendError> {
    // An evicted handle a request still holds is open without being in the
    // collection manager's map. Its cache directory stays registered, and
    // that registry is the only place it shows: no cache directory, no pass.
    let Some(cache_dir) = cache_subtree_for_lsm_path(collection_prefix) else {
        return Ok(ClosedGcOutcome::Skipped("cache_dir_unset"));
    };
    if cache_evictor::cache_dir_is_open(&cache_dir) {
        return Ok(ClosedGcOutcome::Skipped("handle_open"));
    }
    let (options, [_, _, sst_min_age, _], quiet_period) = match closed_gc_options() {
        Ok(options) => options,
        Err(reason) => return Ok(ClosedGcOutcome::Skipped(reason)),
    };
    let directories = CLOSED_GC_DIRECTORIES + u64::from(options.wal_fence_options.is_some());
    let dry_run = !closed_gc_deletes().allows(collection_prefix);
    let store: Arc<dyn ObjectStore> = if dry_run {
        Arc::new(DryRunStore::new(store))
    } else {
        store
    };
    let store = PassStore::new(store, deadline, gate_wanted);
    let stop = Arc::clone(&store.stop);
    let store: Arc<dyn ObjectStore> = Arc::new(store);
    let filter = Arc::new(ClosedGcFilter::new(collection_prefix, dry_run, sst_min_age));

    let ran = examine_and_collect(
        &store,
        collection_prefix,
        options,
        quiet_period,
        lapsed_lease_grace,
        deadline,
        &filter,
    );
    let stopped = stop.stopped.load(Ordering::Relaxed);
    match ran {
        Ok(Some(reason)) => Ok(ClosedGcOutcome::Skipped(reason)),
        // A call the pass was refused comes back as an error. The stop is the
        // outcome then, not the error.
        Err(error) if stopped == 0 => Err(error),
        Ok(None) | Err(_) => Ok(ClosedGcOutcome::Collected(ClosedGcPass {
            candidates: filter.candidates.load(Ordering::Relaxed),
            candidate_bytes: filter.candidate_bytes.load(Ordering::Relaxed),
            objects: filter.objects.load(Ordering::Relaxed),
            bytes: filter.bytes.load(Ordering::Relaxed),
            dry_run,
            timed_out: stopped == STOPPED_AT_THE_DEADLINE,
            yielded: stopped == STOPPED_FOR_THE_GATE,
            incomplete: stopped == 0 && filter.directories.load(Ordering::Relaxed) < directories,
            deletes_not_started: stop.deletes_not_started.load(Ordering::Relaxed),
            metadata_sidecar: None,
        })),
    }
}

/// Decide whether the collection may be collected and, if so, run the pass.
/// Returns why it was left alone, if it was. Nothing is written or deleted
/// before every check here has passed.
fn examine_and_collect(
    store: &Arc<dyn ObjectStore>,
    collection_prefix: &str,
    options: GarbageCollectorOptions,
    quiet_period: Duration,
    lapsed_lease_grace: Duration,
    deadline: Instant,
    filter: &Arc<ClosedGcFilter>,
) -> Result<Option<&'static str>, BackendError> {
    if read_drop_tombstone_with_store(Arc::clone(store), collection_prefix)? {
        return Ok(Some("tombstoned"));
    }
    let rt = shared_lsm_handle();
    let admin =
        slatedb::admin::AdminBuilder::new(collection_prefix.to_string(), Arc::clone(store)).build();
    let Some(manifest) = block_on_lsm(&rt, admin.read_manifest(None))? else {
        return Ok(Some("no_manifest"));
    };
    // SlateDB strips expired checkpoints again before each directory, so the
    // whole time the pass has left counts, not only this moment.
    let time_left = deadline.saturating_duration_since(Instant::now());
    let now = chrono::Utc::now();
    if manifest.checkpoints().iter().any(|checkpoint| {
        checkpoint
            .expire_time
            .is_some_and(|expires| lapsing(expires, now, lapsed_lease_grace, time_left))
    }) {
        return Ok(Some("lease_lapsing"));
    }

    // Both listings are taken here, under the caller's gate, before the pass
    // makes its first write.
    let root = ObjPath::from(collection_prefix);
    let needed = needed_ssts(&root, &manifest);
    if needed.is_empty() {
        // With no SST to find, nothing shows this manifest belongs to the
        // objects around it.
        return Ok(Some("empty"));
    }
    let (ssts, newest_sst) = list_directory(&rt, store, root.clone().join("compacted"))?;
    let (wal, newest_wal) = list_directory(&rt, store, root.clone().join("wal"))?;
    // SlateDB reads the latest manifest again for each directory and ages an
    // SST by the time in its name. In a prefix something is still writing, a
    // copy or restore into it or a second writer, the manifest read above may
    // not be the last, and SSTs only a later one names would go as garbage.
    // Nothing written under either directory for the quiet period is what
    // shows that nothing is. A time ahead of this clock counts as just now.
    if newest_sst.max(newest_wal).is_some_and(|written| {
        now.signed_duration_since(written)
            .to_std()
            .map_or(true, |quiet_for| quiet_for < quiet_period)
    }) {
        return Ok(Some("recently_written"));
    }
    let missing = needed
        .into_iter()
        .find(|sst| !ssts.contains(sst) && !wal.contains(sst))
        .or_else(|| first_wal_gap(&root, manifest.replay_after_wal_id(), &wal));
    if let Some(missing) = missing {
        tracing::warn!(
            collection = %collection_prefix,
            manifest = manifest.id(),
            missing = %missing,
            "closed-collection GC left a collection alone: its latest manifest needs an object that is gone"
        );
        return Ok(Some("not_intact"));
    }

    let gc_filter: Arc<dyn slatedb::GcFilter> = filter.clone();
    let store = Arc::clone(store);
    let path = collection_prefix.to_string();
    // Runs to completion: dropping it would abandon deletes and manifest writes
    // already on the wire, to land after the caller released the gate.
    block_on_lsm(&rt, async move {
        slatedb::GarbageCollectorBuilder::new(path, store)
            .with_options(options)
            .with_gc_filter(gc_filter)
            .build()
            .run_gc_once()
            .await;
        Ok::<(), String>(())
    })?;
    Ok(None)
}

/// Whether a pass that strips expired checkpoints from `now` until `time_left`
/// later could strip one that expires at `expires` less than `grace` after it
/// did: it expired less than `grace` ago, or expires before the pass must end.
pub(super) fn lapsing(
    expires: chrono::DateTime<chrono::Utc>,
    now: chrono::DateTime<chrono::Utc>,
    grace: Duration,
    time_left: Duration,
) -> bool {
    let millis = |duration: Duration| i64::try_from(duration.as_millis()).unwrap_or(i64::MAX);
    // Negative for as long as the checkpoint has not expired.
    let lapsed_for = now.signed_duration_since(expires).num_milliseconds();
    lapsed_for > -millis(time_left) && lapsed_for < millis(grace)
}

/// Every SST `manifest` names, as the object the bucket should hold for it.
fn needed_ssts(root: &ObjPath, manifest: &slatedb::VersionedManifest) -> Vec<String> {
    let trees = std::iter::once((manifest.l0(), manifest.compacted())).chain(
        manifest
            .segments()
            .iter()
            .map(|segment| (segment.l0(), segment.compacted())),
    );
    trees
        .flat_map(|(l0, runs)| {
            l0.iter()
                .chain(runs.iter().flat_map(|run| run.sst_views.iter()))
        })
        .map(|view| match view.sst.id {
            slatedb::manifest::SsTableId::Compacted(id) => format!("{root}/compacted/{id}.sst"),
            slatedb::manifest::SsTableId::Wal(id) => wal_object(root, id),
        })
        .collect()
}

fn wal_object(root: &ObjPath, id: u64) -> String {
    format!("{root}/wal/{id:020}.sst")
}

/// Every object in `directory`, and when the newest of them was written. A
/// listing that was cut short is an error.
fn list_directory(
    rt: &Handle,
    store: &Arc<dyn ObjectStore>,
    directory: ObjPath,
) -> Result<(HashSet<String>, Option<chrono::DateTime<chrono::Utc>>), BackendError> {
    let store = Arc::clone(store);
    block_on_lsm(rt, async move {
        let mut objects = HashSet::new();
        let mut newest: Option<chrono::DateTime<chrono::Utc>> = None;
        let mut listing = store.list(Some(&directory));
        while let Some(object) = listing.try_next().await? {
            newest = newest.max(Some(object.last_modified));
            objects.insert(object.location.to_string());
        }
        Ok::<_, ObjectStoreError>((objects, newest))
    })
}

/// The first WAL object an open would need and not find. An open replays
/// every WAL object after `replay_after`, taking their ids to be contiguous,
/// so one missing below another that is there is a hole in the log.
fn first_wal_gap(root: &ObjPath, replay_after: u64, wal: &HashSet<String>) -> Option<String> {
    let prefix = format!("{root}/wal/");
    let unreplayed: BTreeSet<u64> = wal
        .iter()
        .filter_map(|object| {
            object
                .strip_prefix(&prefix)?
                .strip_suffix(".sst")?
                .parse::<u64>()
                .ok()
        })
        .filter(|id| *id > replay_after)
        .collect();
    unreplayed
        .iter()
        .zip(replay_after.saturating_add(1)..)
        .find(|(present, expected)| *present != expected)
        .map(|(_, expected)| wal_object(root, expected))
}
