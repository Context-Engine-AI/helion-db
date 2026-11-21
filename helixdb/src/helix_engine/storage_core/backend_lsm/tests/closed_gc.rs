//! The closed-collection GC pass (`backend_lsm_closed_gc.rs`) against real
//! SlateDB state on an in-memory object store.

use super::super::closed_gc::{
    gc_closed_collection_until, lapsing, ClosedGcDeletes, ClosedGcFilter, DryRunStore, PassStore,
    CLOSED_GC_MAX_DELETES_PER_DIRECTORY, ENV_IDLE_GC_DELETE_PREFIXES, ENV_IDLE_GC_DRY_RUN,
    ENV_IDLE_GC_QUIET_SECS, ENV_IDLE_GC_WAL_FENCES, ENV_TEST_MIN_AGE_FLOOR_SECS,
};
use super::*;
use std::time::{Duration, Instant};

type Objects = BTreeMap<String, (u64, Option<String>)>;

/// Every object under `prefix`: location to (size, e_tag). The e_tag changes
/// when an object is rewritten, so equal maps mean nothing was written either.
fn objects(store: &Arc<dyn ObjectStore>, prefix: &str) -> Objects {
    block_on_lsm(&shared_lsm_handle(), async {
        store
            .list(Some(&ObjPath::from(prefix)))
            .map_ok(|meta| (meta.location.to_string(), (meta.size, meta.e_tag)))
            .try_collect::<Objects>()
            .await
            .map_err(|error| error.to_string())
    })
    .unwrap()
}

/// Nothing else wants the collection's open gate.
fn unwanted() -> GateWanted {
    Arc::new(|| false)
}

/// What a pass needs from the env that these tests do not vary. A cache
/// directory: the registry of open cache directories is how a pass sees a
/// handle that is still open. And a one-second floor under the minimum age, in
/// place of fifteen minutes, so that a test's garbage is old enough in a second.
fn pass_env() -> (EnvGuard, EnvGuard, tempfile::TempDir) {
    let dir = tempfile::TempDir::new().unwrap();
    (
        EnvGuard::set(ENV_CACHE_DIR, dir.path().to_str().unwrap()),
        EnvGuard::set(ENV_TEST_MIN_AGE_FLOOR_SECS, "1"),
        dir,
    )
}

/// One pass with the usual deadline. The fixtures lapse their checkpoints
/// moments before the pass, so it does not give a lapsed lease the twenty
/// minutes a pass otherwise leaves its reader to come back.
fn pass(store: &Arc<dyn ObjectStore>, path: &str) -> ClosedGcOutcome {
    gc_closed_collection_until(
        Arc::clone(store),
        path,
        Instant::now() + Duration::from_secs(120),
        Duration::ZERO,
        unwanted(),
    )
    .unwrap()
}

/// One pass that ran and got through every directory.
fn collect(store: &Arc<dyn ObjectStore>, path: &str) -> ClosedGcPass {
    match pass(store, path) {
        ClosedGcOutcome::Collected(pass) => {
            assert!(
                !pass.timed_out && !pass.yielded && !pass.incomplete,
                "pass cut short: {pass:?}"
            );
            pass
        }
        ClosedGcOutcome::Skipped(reason) => panic!("pass skipped: {reason}"),
    }
}

/// The highest WAL id in the bucket.
fn last_wal_id(store: &Arc<dyn ObjectStore>, path: &str) -> u64 {
    let wal = format!("{path}/wal/");
    objects(store, path)
        .keys()
        .filter_map(|location| {
            location
                .strip_prefix(&wal)?
                .strip_suffix(".sst")?
                .parse::<u64>()
                .ok()
        })
        .max()
        .unwrap_or(0)
}

/// A closed collection that was opened twice, so it holds manifest versions
/// nothing references, all older than a one-second minimum age.
fn collection_with_stale_manifests(store: &Arc<dyn ObjectStore>, path: &str) {
    {
        // A writer's own collector ticks at open; under the callers' one-second
        // minimum age the second open could take the first one's manifests.
        let _keep = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "3600");
        for key in [b"k1", b"k2"] {
            let be = LsmBackend::open_with_store(path, Arc::clone(store)).unwrap();
            put_one(&be, key);
            be.close().unwrap();
        }
    }
    std::thread::sleep(Duration::from_millis(1_200));
}

/// A writer that compacts promptly and whose own collector, with an hour's
/// interval and minimum age, leaves everything the fixture produces in place.
fn compacting_writer_env() -> Vec<EnvGuard> {
    vec![
        EnvGuard::set(ENV_GC_INTERVAL_SECS, "3600"),
        EnvGuard::set(ENV_GC_MIN_AGE_SECS, "3600"),
        EnvGuard::set(ENV_COMPACTOR_POLL_MS, "50"),
        EnvGuard::set(ENV_COMPACTOR_COMMIT_COMPACTED_MS, "50"),
        EnvGuard::set(ENV_COMPACTOR_WORKER_POLL_MS, "50"),
        EnvGuard::set(ENV_MANIFEST_POLL_MS, "50"),
    ]
}

/// A closed collection whose compaction, memtable flushes and manifest updates
/// each left objects nothing references any more. Returns the rows it holds.
fn compacted_collection(store: &Arc<dyn ObjectStore>, path: &str) -> BTreeMap<Vec<u8>, Vec<u8>> {
    const ROUNDS: usize = 6;
    const KEYS: usize = 200;
    let rt = shared_lsm_handle();
    let be = LsmBackend::open_with_store(path, Arc::clone(store)).unwrap();
    let mut rows = BTreeMap::new();
    let mut write_l0 = |round: usize| {
        let mut w = be.begin_write().unwrap();
        for i in 0..KEYS {
            let key = format!("k{i:04}").into_bytes();
            let value = format!("round{round}-{i:04}").into_bytes();
            be.put(&mut w, Namespace::Nodes, &key, &value).unwrap();
            rows.insert(key, value);
        }
        be.commit(w).unwrap();
        block_on_lsm(
            &rt,
            be.db().flush_with_options(slatedb::config::FlushOptions {
                flush_type: slatedb::config::FlushType::MemTable,
            }),
        )
        .unwrap();
    };
    for round in 0..ROUNDS {
        write_l0(round);
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    while be.db().manifest().compacted().is_empty() || be.db().manifest().l0().len() >= ROUNDS {
        assert!(Instant::now() < deadline, "no compaction");
        std::thread::sleep(Duration::from_millis(50));
    }
    // SlateDB never deletes an SST as new as the newest L0 it knows of; this
    // one is newer than everything the compaction consumed.
    write_l0(ROUNDS);
    be.close().unwrap();
    drop(be);

    // The compactor pins what it just compacted away with a 15-minute
    // checkpoint. Lapse it instead of waiting.
    let admin = slatedb::admin::AdminBuilder::new(path, Arc::clone(store)).build();
    for checkpoint in block_on_lsm(&rt, admin.list_checkpoints(None)).unwrap() {
        block_on_lsm(
            &rt,
            admin.refresh_checkpoint(checkpoint.id, Some(Duration::ZERO)),
        )
        .unwrap();
    }
    std::thread::sleep(Duration::from_millis(1_200));
    rows
}

#[test]
#[serial]
fn pass_removes_unreferenced_objects_and_every_row_reads_back() {
    let _env = compacting_writer_env();
    let _pass_env = pass_env();
    let _deleting = EnvGuard::set(ENV_IDLE_GC_DRY_RUN, "0");
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "helion/closed_gc_reclaims";
    let rt = shared_lsm_handle();
    let rows = compacted_collection(&store, path);

    // The ages are the in-process collector's. Under its one-hour minimum the
    // collection was written too recently for a pass to touch it.
    let before = objects(&store, path);
    assert_eq!(
        pass(&store, path),
        ClosedGcOutcome::Skipped("recently_written")
    );
    assert_eq!(objects(&store, path), before);

    let _min_age = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "1");
    let pass = collect(&store, path);
    let after = objects(&store, path);
    let removed: Vec<&String> = before
        .keys()
        .filter(|location| !after.contains_key(*location))
        .collect();
    assert_eq!(pass.objects as usize, removed.len());
    assert_eq!(
        (pass.candidates, pass.candidate_bytes),
        (pass.objects, pass.bytes)
    );
    assert_eq!(
        pass.bytes,
        removed
            .iter()
            .map(|location| before[*location].0)
            .sum::<u64>()
    );
    for directory in ["compacted", "wal", "manifest"] {
        assert!(
            removed
                .iter()
                .any(|location| location.starts_with(&format!("{path}/{directory}/"))),
            "nothing was collected under {directory}/"
        );
    }

    // Everything the latest manifest references is still there.
    let admin = slatedb::admin::AdminBuilder::new(path, Arc::clone(&store)).build();
    let manifest = block_on_lsm(&rt, admin.read_manifest(None))
        .unwrap()
        .expect("latest manifest");
    assert_eq!(manifest_ids(&rt, &store, path), vec![manifest.id()]);
    let live_ssts = manifest.l0().iter().chain(
        manifest
            .compacted()
            .iter()
            .flat_map(|run| run.sst_views.iter()),
    );
    for view in live_ssts {
        let location = format!("{path}/compacted/{}.sst", view.sst.id.unwrap_compacted_id());
        assert!(after.contains_key(&location), "live SST {location} deleted");
    }
    let wal_prefix = format!("{path}/wal/");
    for location in before.keys() {
        let wal_id = location
            .strip_prefix(&wal_prefix)
            .and_then(|file| file.strip_suffix(".sst"))
            .and_then(|id| id.parse::<u64>().ok());
        if wal_id.is_some_and(|id| id > manifest.replay_after_wal_id()) {
            assert!(
                after.contains_key(location),
                "unreplayed WAL {location} deleted"
            );
        }
    }

    let be = LsmBackend::open_with_store(path, Arc::clone(&store)).unwrap();
    let r = be.begin_read().unwrap();
    let mut read_back = BTreeMap::new();
    be.scan(&r, Namespace::Nodes, KeyRange::all(), |key, value| {
        read_back.insert(key.to_vec(), value.to_vec());
        true
    })
    .unwrap();
    assert_eq!(read_back, rows);
}

/// A writer leaves a zero-byte fence in the WAL at each open, and SlateDB's
/// WAL task takes only objects that hold data. A pass leaves the fences below
/// the replay position where they are unless the knob asks for them as well.
#[test]
#[serial]
fn pass_takes_wal_fences_only_where_the_knob_asks_for_them() {
    let _env = compacting_writer_env();
    let _pass_env = pass_env();
    let _deleting = EnvGuard::set(ENV_IDLE_GC_DRY_RUN, "0");
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "helion/closed_gc_fences";
    let rows = compacted_collection(&store, path);
    let _min_age = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "1");

    let admin = slatedb::admin::AdminBuilder::new(path, Arc::clone(&store)).build();
    let replay_after = block_on_lsm(&shared_lsm_handle(), admin.read_manifest(None))
        .unwrap()
        .expect("latest manifest")
        .replay_after_wal_id();
    let wal_prefix = format!("{path}/wal/");
    let wal_id = |location: &str| {
        location
            .strip_prefix(&wal_prefix)?
            .strip_suffix(".sst")?
            .parse::<u64>()
            .ok()
    };
    // The ids of the zero-byte WAL objects: below the replay position, and
    // from it on.
    let fences = || -> (Vec<u64>, Vec<u64>) {
        objects(&store, path)
            .into_iter()
            .filter(|(_, (size, _))| *size == 0)
            .filter_map(|(location, _)| wal_id(&location))
            .partition(|id| *id < replay_after)
    };
    // An old WAL object, emptied, stands in for the fence of an earlier open.
    let old_wal = objects(&store, path)
        .into_keys()
        .find(|location| wal_id(location).is_some_and(|id| id < replay_after))
        .expect("a WAL object below the replay position");
    put_object(&store, &old_wal, Vec::new());
    std::thread::sleep(Duration::from_millis(1_200));
    let (below, from_it_on) = fences();
    assert!(!below.is_empty());

    // Without the knob a pass takes the WAL objects that hold data and no fence.
    assert!(collect(&store, path).objects > 0);
    assert_eq!(fences(), (below.clone(), from_it_on.clone()));

    let _fences = EnvGuard::set(ENV_IDLE_GC_WAL_FENCES, "1");
    assert!(collect(&store, path).objects >= below.len() as u64);
    assert_eq!(fences(), (Vec::new(), from_it_on));

    let be = LsmBackend::open_with_store(path, Arc::clone(&store)).unwrap();
    let r = be.begin_read().unwrap();
    let mut read_back = BTreeMap::new();
    be.scan(&r, Namespace::Nodes, KeyRange::all(), |key, value| {
        read_back.insert(key.to_vec(), value.to_vec());
        true
    })
    .unwrap();
    assert_eq!(read_back, rows);
}

#[test]
#[serial]
fn pass_leaves_a_dropped_or_manifestless_prefix_untouched() {
    let _deleting = EnvGuard::set(ENV_IDLE_GC_DRY_RUN, "0");
    let _pass_env = pass_env();
    let _interval = EnvGuard::set(ENV_GC_INTERVAL_SECS, "3600");
    let _min_age = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "1");
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let root = "closed_gc_untouched";

    // A prefix mid-create, or one that is not a collection at all.
    let stray = format!("{root}/stray");
    let stray_object = ObjPath::from(format!("{stray}/wal/00000000000000000001.sst"));
    shared_lsm_handle().block_on(async {
        store
            .put(&stray_object, b"x".to_vec().into())
            .await
            .unwrap();
    });
    let dropped = format!("{root}/dropped");
    collection_with_stale_manifests(&store, &dropped);
    persist_drop_tombstone_with_store(Arc::clone(&store), &dropped).unwrap();

    let before = objects(&store, root);
    assert_eq!(
        pass(&store, &stray),
        ClosedGcOutcome::Skipped("no_manifest")
    );
    assert_eq!(
        pass(&store, &dropped),
        ClosedGcOutcome::Skipped("tombstoned")
    );
    assert_eq!(objects(&store, root), before);

    // Only the tombstone held the pass back.
    clear_drop_tombstone_with_store(Arc::clone(&store), &dropped).unwrap();
    assert!(collect(&store, &dropped).objects > 0);
}

#[test]
#[serial]
fn pass_skips_a_collection_whose_handle_is_still_open() {
    let cache_dir = tempfile::TempDir::new().unwrap();
    let _cache = EnvGuard::set(ENV_CACHE_DIR, cache_dir.path().to_str().unwrap());
    let _min_age = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "3600");
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "helion/closed_gc_open_handle";
    let be = LsmBackend::open_with_store(path, Arc::clone(&store)).unwrap();
    put_one(&be, b"k");
    assert_eq!(pass(&store, path), ClosedGcOutcome::Skipped("handle_open"));

    // Closed, the handle no longer holds a pass back. That the collection was
    // written a moment ago does.
    be.close().unwrap();
    drop(be);
    assert_eq!(
        pass(&store, path),
        ClosedGcOutcome::Skipped("recently_written")
    );
}

#[test]
#[serial]
fn pass_past_its_deadline_stops_without_deleting_or_writing() {
    let _deleting = EnvGuard::set(ENV_IDLE_GC_DRY_RUN, "0");
    let _pass_env = pass_env();
    let _interval = EnvGuard::set(ENV_GC_INTERVAL_SECS, "3600");
    let _min_age = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "1");
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "helion/closed_gc_deadline";
    collection_with_stale_manifests(&store, path);

    let before = objects(&store, path);
    assert_eq!(
        gc_closed_collection_until(
            Arc::clone(&store),
            path,
            Instant::now(),
            Duration::ZERO,
            unwanted()
        )
        .unwrap(),
        ClosedGcOutcome::Collected(ClosedGcPass {
            timed_out: true,
            ..ClosedGcPass::default()
        })
    );
    assert_eq!(objects(&store, path), before);

    // There was garbage to take.
    assert!(collect(&store, path).objects > 0);
}

#[test]
#[serial]
fn pass_is_a_dry_run_until_deletion_is_switched_on() {
    let _pass_env = pass_env();
    let _interval = EnvGuard::set(ENV_GC_INTERVAL_SECS, "3600");
    let _min_age = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "1");
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "helion/closed_gc_dry_run";

    let dry_then_real = || {
        let before = objects(&store, path);
        let dry = {
            // With the knob unset a pass only reports what it would delete.
            let _unset = EnvGuard::remove(ENV_IDLE_GC_DRY_RUN);
            let dry = collect(&store, path);
            // A value that is not an explicit "off" does not switch deletion on.
            let _typo = EnvGuard::set(ENV_IDLE_GC_DRY_RUN, "flase");
            assert_eq!(collect(&store, path), dry);
            dry
        };
        assert!(dry.candidates > 0 && dry.candidate_bytes > 0);
        assert_eq!((dry.objects, dry.bytes), (0, 0));
        // Not one object was written, rewritten or deleted.
        assert_eq!(objects(&store, path), before);

        // The real pass deletes exactly what the dry run reported.
        let _deleting = EnvGuard::set(ENV_IDLE_GC_DRY_RUN, "0");
        let real = collect(&store, path);
        assert_eq!(
            (real.objects, real.bytes),
            (dry.candidates, dry.candidate_bytes)
        );
    };
    // First with no GC boundary object in the bucket, so the pass creates one;
    // then with the one the first real pass left, which the pass advances.
    collection_with_stale_manifests(&store, path);
    dry_then_real();
    collection_with_stale_manifests(&store, path);
    dry_then_real();
}

#[test]
#[serial]
fn dry_run_counts_what_a_lapsed_checkpoint_still_protects() {
    let _env = compacting_writer_env();
    let _pass_env = pass_env();
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "helion/closed_gc_dry_checkpoint";
    compacted_collection(&store, path);
    let _min_age = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "1");

    // The latest manifest still holds the compactor's lapsed checkpoint, which
    // a pass removes before it looks for garbage. A dry run removes it in
    // memory only, and still counts the SSTs nothing else protected.
    let before = objects(&store, path);
    let dry = {
        let _unset = EnvGuard::remove(ENV_IDLE_GC_DRY_RUN);
        collect(&store, path)
    };
    assert_eq!((dry.objects, dry.bytes), (0, 0));
    assert_eq!(objects(&store, path), before);

    let _deleting = EnvGuard::set(ENV_IDLE_GC_DRY_RUN, "0");
    let real = collect(&store, path);
    assert_eq!(
        (real.objects, real.bytes),
        (dry.candidates, dry.candidate_bytes)
    );
    let after = objects(&store, path);
    let ssts = format!("{path}/compacted/");
    assert!(
        before
            .keys()
            .any(|location| location.starts_with(&ssts) && !after.contains_key(location)),
        "no SST was collected"
    );
}

#[test]
fn dry_run_store_keeps_writes_in_memory_and_judges_them_against_the_bucket() {
    use object_store::{PutMode, PutOptions, UpdateVersion};

    let bucket: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let boundary = ObjPath::from("c/gc/manifest.boundary");
    let manifest = ObjPath::from("c/manifest/00000000000000000002.manifest");
    let payload = |bytes: &[u8]| object_store::PutPayload::from(bytes.to_vec());
    shared_lsm_handle().block_on(async {
        let in_bucket = bucket.put(&boundary, payload(b"1")).await.unwrap();
        let before = bucket.head(&boundary).await.unwrap();
        let store = DryRunStore::new(Arc::clone(&bucket));

        // Create: refused where the bucket has the object, kept where it has
        // none, and refused again once the pass has written there itself.
        let create = || PutOptions::from(PutMode::Create);
        assert!(matches!(
            store.put_opts(&boundary, payload(b"2"), create()).await,
            Err(ObjectStoreError::AlreadyExists { .. })
        ));
        store
            .put_opts(&manifest, payload(b"m"), create())
            .await
            .unwrap();
        assert!(matches!(
            store.put_opts(&manifest, payload(b"m"), create()).await,
            Err(ObjectStoreError::AlreadyExists { .. })
        ));

        // Update: only over the version the bucket holds.
        let stale = UpdateVersion {
            e_tag: Some("another".to_string()),
            version: None,
        };
        assert!(matches!(
            store
                .put_opts(&boundary, payload(b"2"), PutMode::Update(stale).into())
                .await,
            Err(ObjectStoreError::Precondition { .. })
        ));
        store
            .put_opts(
                &boundary,
                payload(b"2"),
                PutMode::Update(in_bucket.into()).into(),
            )
            .await
            .unwrap();

        // The pass reads its own writes, each object once.
        let read = store.get(&boundary).await.unwrap().bytes().await.unwrap();
        assert_eq!(read.to_vec(), b"2".to_vec());
        let mut listed: Vec<String> = store
            .list(Some(&ObjPath::from("c")))
            .map_ok(|object| object.location.to_string())
            .try_collect()
            .await
            .unwrap();
        listed.sort();
        assert_eq!(listed, vec![boundary.to_string(), manifest.to_string()]);

        // It cannot delete, and the bucket holds exactly what it held.
        assert!(store.delete(&boundary).await.is_err());
        let left: Vec<object_store::ObjectMeta> = bucket.list(None).try_collect().await.unwrap();
        assert_eq!(left, vec![before]);
    });
}

/// The dry-run knob says where passes delete: an off value everywhere,
/// `prefixes` only under the delete prefixes, anything else nowhere. The
/// prefixes cover a collection by its name, the last segment of its path.
/// Under `prefixes` a missing, empty or malformed value covers nothing, and
/// next to an off value a value of any kind stops every delete.
#[test]
#[serial]
fn dry_run_knob_says_where_passes_delete() {
    let deletes = |knob: Option<&str>, prefixes: Option<&str>| {
        let _knob = match knob {
            Some(knob) => EnvGuard::set(ENV_IDLE_GC_DRY_RUN, knob),
            None => EnvGuard::remove(ENV_IDLE_GC_DRY_RUN),
        };
        let _prefixes = match prefixes {
            Some(prefixes) => EnvGuard::set(ENV_IDLE_GC_DELETE_PREFIXES, prefixes),
            None => EnvGuard::remove(ENV_IDLE_GC_DELETE_PREFIXES),
        };
        closed_gc_deletes()
    };

    for (prefixes, collection, covered) in [
        ("admin_", "prod/admin_docs", true),
        ("admin_", "prod/admin_", true),
        ("admin_", "prod/acme_docs", false),
        ("admin_", "prod/xadmin_docs", false),
        ("admin_", "prod/Admin_docs", false),
        // A root that starts with a prefix covers nothing under it.
        ("admin_", "admin_root/acme_docs", false),
        (" admin_ , acme.io- ", "prod/acme.io-docs", true),
        ("admin_,acme_", "prod/other", false),
    ] {
        assert_eq!(
            deletes(Some("prefixes"), Some(prefixes)).allows(collection),
            covered,
            "{prefixes:?} {collection}"
        );
    }
    assert_eq!(
        deletes(Some(" Prefixes "), Some("admin_, acme_")),
        ClosedGcDeletes::Under(vec!["admin_".to_string(), "acme_".to_string()])
    );

    // `prefixes` without the variable, or with an empty or malformed one.
    for prefixes in [
        None,
        Some(""),
        Some(" "),
        Some(","),
        Some("admin_,"),
        Some(",admin_"),
        Some("admin_,,acme_"),
        Some("*"),
        Some("admin_*"),
        Some("prod/admin_"),
        Some("admin_ x"),
        Some("\"admin_\""),
    ] {
        let got = deletes(Some("prefixes"), prefixes);
        assert_eq!(got, ClosedGcDeletes::Nowhere, "{prefixes:?}");
        assert!(!got.allows("prod/admin_docs"), "{prefixes:?}");
    }

    // An off value: everywhere. With the variable set as well, whatever it
    // holds, the two contradict each other: nowhere.
    for knob in ["0", "false", "no", "off", " OFF "] {
        let got = deletes(Some(knob), None);
        assert_eq!(got, ClosedGcDeletes::Everywhere, "{knob:?}");
        assert!(got.allows("prod/acme_docs"), "{knob:?}");
        for prefixes in ["admin_", "", "admin_*"] {
            let got = deletes(Some(knob), Some(prefixes));
            assert_eq!(got, ClosedGcDeletes::Nowhere, "{knob:?} {prefixes:?}");
            assert!(!got.allows("prod/admin_docs"), "{knob:?} {prefixes:?}");
        }
    }

    // Unset or any other word: nowhere, whatever the prefixes say.
    for knob in [
        None,
        Some(""),
        Some("1"),
        Some("true"),
        Some("prefix"),
        Some("prefixes,0"),
        Some("prefixes=admin_"),
        Some("admin_"),
        Some("all"),
        Some("canary"),
    ] {
        let got = deletes(knob, Some("admin_"));
        assert_eq!(got, ClosedGcDeletes::Nowhere, "{knob:?}");
        assert!(!got.allows("prod/admin_docs"), "{knob:?}");
    }
}

#[test]
#[serial]
fn only_an_explicit_off_value_lets_a_pass_delete() {
    let _pass_env = pass_env();
    let _min_age = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "3600");
    let dry_run = || closed_gc_plan().unwrap().dry_run;
    let _unset = EnvGuard::remove(ENV_IDLE_GC_DRY_RUN);
    let _no_prefixes = EnvGuard::remove(ENV_IDLE_GC_DELETE_PREFIXES);
    assert!(dry_run());
    // `prefixes` is among them: by itself that word covers no collection.
    for value in [
        "", " ", "1", "true", "yes", "on", "2", "flase", "disabled", "0.0", "prefixes",
    ] {
        let _value = EnvGuard::set(ENV_IDLE_GC_DRY_RUN, value);
        assert!(dry_run(), "{value:?}");
    }
    for value in ["0", "false", "no", "off", "OFF", " False "] {
        let _value = EnvGuard::set(ENV_IDLE_GC_DRY_RUN, value);
        assert!(!dry_run(), "{value:?}");
    }
}

#[test]
#[serial]
fn plan_names_what_a_pass_applies_and_there_is_none_without_a_cache_or_ages() {
    let _pass_env = pass_env();
    let _interval = EnvGuard::set(ENV_GC_INTERVAL_SECS, "3600");
    let _min_age = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "86400");
    let _wal_min_age = EnvGuard::set(ENV_GC_WAL_MIN_AGE_SECS, "60");
    let plan = closed_gc_plan().unwrap();
    let day = Duration::from_secs(86_400);
    assert_eq!(
        (
            plan.manifest_min_age,
            plan.compacted_min_age,
            plan.compactions_min_age
        ),
        (day, day, day)
    );
    assert_eq!(plan.wal_min_age, Duration::from_secs(60));
    assert_eq!(
        plan.max_deletes_per_directory,
        CLOSED_GC_MAX_DELETES_PER_DIRECTORY
    );
    assert_eq!(plan.deadline, Duration::from_secs(120));
    assert_eq!(plan.lapsed_lease_grace, Duration::from_secs(1_200));
    // Without its knob the quiet period is the SST minimum age.
    assert_eq!(plan.quiet_period, day);
    {
        let _quiet = EnvGuard::set(ENV_IDLE_GC_QUIET_SECS, "3600");
        let hour = Duration::from_secs(3_600);
        assert_eq!(closed_gc_plan().unwrap().quiet_period, hour);
    }

    // Without a cache directory a pass cannot see a handle that is still
    // open, and without an age from the env SlateDB's own five minutes would
    // apply: no plan, so no sweeper, and no pass either.
    {
        let _no_cache = EnvGuard::set(ENV_CACHE_DIR, "");
        assert_refused("cache_dir_unset");
    }
    {
        // The WAL age alone does not cover the other directories.
        let _no_age = EnvGuard::remove(ENV_GC_MIN_AGE_SECS);
        assert_refused("min_age_unset");
        let _no_wal_age = EnvGuard::remove(ENV_GC_WAL_MIN_AGE_SECS);
        assert_refused("min_age_unset");
        let _no_interval = EnvGuard::remove(ENV_GC_INTERVAL_SECS);
        assert_refused("min_age_unset");
    }
}

/// There is no plan, and no pass either, for `reason`.
fn assert_refused(reason: &'static str) {
    assert_eq!(closed_gc_plan(), Err(reason));
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    assert_eq!(
        gc_closed_collection_with_store(store, "helion/closed_gc_any", unwanted()).unwrap(),
        ClosedGcOutcome::Skipped(reason)
    );
}

#[test]
#[serial]
fn minimum_age_under_fifteen_minutes_is_refused() {
    let cache_dir = tempfile::TempDir::new().unwrap();
    let _cache = EnvGuard::set(ENV_CACHE_DIR, cache_dir.path().to_str().unwrap());
    let _production_floor = EnvGuard::remove(ENV_TEST_MIN_AGE_FLOOR_SECS);
    let _interval = EnvGuard::set(ENV_GC_INTERVAL_SECS, "3600");
    {
        let _min_age = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "899");
        assert_refused("min_age_under_900s");
    }
    let _min_age = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "900");
    let quarter_hour = Duration::from_secs(900);
    assert_eq!(closed_gc_plan().unwrap().compacted_min_age, quarter_hour);

    // One directory with its own age under the floor is enough.
    for directory in [
        ENV_GC_MANIFEST_MIN_AGE_SECS,
        ENV_GC_WAL_MIN_AGE_SECS,
        ENV_GC_COMPACTED_MIN_AGE_SECS,
        ENV_GC_COMPACTIONS_MIN_AGE_SECS,
    ] {
        let _short = EnvGuard::set(directory, "60");
        assert_refused("min_age_under_900s");
    }
}

#[test]
#[serial]
fn quiet_period_under_fifteen_minutes_is_refused() {
    let cache_dir = tempfile::TempDir::new().unwrap();
    let _cache = EnvGuard::set(ENV_CACHE_DIR, cache_dir.path().to_str().unwrap());
    let _production_floor = EnvGuard::remove(ENV_TEST_MIN_AGE_FLOOR_SECS);
    let _interval = EnvGuard::set(ENV_GC_INTERVAL_SECS, "3600");
    // However old an object has to be.
    let _min_age = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "86400");
    {
        let _quiet = EnvGuard::set(ENV_IDLE_GC_QUIET_SECS, "899");
        assert_refused("quiet_under_900s");
    }
    let _quiet = EnvGuard::set(ENV_IDLE_GC_QUIET_SECS, "900");
    let quarter_hour = Duration::from_secs(900);
    assert_eq!(closed_gc_plan().unwrap().quiet_period, quarter_hour);
}

#[test]
fn filter_hands_over_a_bounded_number_of_deletions_per_directory() {
    use slatedb::GcFilter;

    let candidates: HashSet<slatedb::ObjectMetadata> = (0..CLOSED_GC_MAX_DELETES_PER_DIRECTORY + 7)
        .map(|id| slatedb::ObjectMetadata {
            location: ObjPath::from(format!("helion/c/wal/{id:020}.sst")),
            last_modified: chrono::DateTime::<chrono::Utc>::UNIX_EPOCH,
            size: 3,
            e_tag: None,
            version: None,
        })
        .collect();
    let hour = Duration::from_secs(3_600);
    let filter = ClosedGcFilter::new("helion/c", false, hour);
    // One call per directory, each capped on its own.
    for _directory in 0..2 {
        let allowed = shared_lsm_handle().block_on(filter.filter(candidates.clone()));
        assert_eq!(allowed.len(), CLOSED_GC_MAX_DELETES_PER_DIRECTORY);
        assert!(allowed.is_subset(&candidates));
    }
    let dry_run = ClosedGcFilter::new("helion/c", true, hour);
    assert!(shared_lsm_handle()
        .block_on(dry_run.filter(candidates))
        .is_empty());
}

#[test]
fn filter_keeps_only_the_ssts_that_arrived_within_the_minimum_age() {
    use slatedb::GcFilter;

    let object =
        |location: String, last_modified: chrono::DateTime<chrono::Utc>| slatedb::ObjectMetadata {
            location: ObjPath::from(location),
            last_modified,
            size: 3,
            e_tag: None,
            version: None,
        };
    let now = chrono::Utc::now();
    let long_ago = chrono::DateTime::<chrono::Utc>::UNIX_EPOCH;
    let filter = ClosedGcFilter::new("helion/c", false, Duration::from_secs(3_600));
    let allowed = |candidates: Vec<slatedb::ObjectMetadata>| {
        shared_lsm_handle().block_on(filter.filter(candidates.into_iter().collect()))
    };

    // SlateDB calls an SST old by the time in its name, however lately the
    // object arrived.
    let sst =
        |ulid: &str, last_modified| object(format!("helion/c/compacted/{ulid}.sst"), last_modified);
    let arrived = sst("00000000010000000000000000", now);
    let ahead_of_this_clock = sst(
        "00000000020000000000000000",
        now + chrono::Duration::hours(1),
    );
    let stale = sst("00000000030000000000000000", long_ago);
    assert_eq!(
        allowed(vec![arrived, ahead_of_this_clock, stale.clone()]),
        HashSet::from([stale])
    );

    // The other three it ages by LastModified itself, each by its own minimum
    // age, which may be shorter than the SSTs'. What it hands over goes.
    for directory in ["manifest", "wal", "compactions"] {
        let candidate = object(format!("helion/c/{directory}/00000000000000000001"), now);
        assert_eq!(
            allowed(vec![candidate.clone()]),
            HashSet::from([candidate]),
            "{directory}"
        );
    }
}

fn delete_object(store: &Arc<dyn ObjectStore>, location: &str) {
    shared_lsm_handle().block_on(async {
        store.delete(&ObjPath::from(location)).await.unwrap();
    });
}

fn put_object(store: &Arc<dyn ObjectStore>, location: &str, bytes: Vec<u8>) {
    shared_lsm_handle().block_on(async {
        store
            .put(&ObjPath::from(location), bytes.into())
            .await
            .unwrap();
    });
}

fn object_bytes(store: &Arc<dyn ObjectStore>, location: &str) -> Vec<u8> {
    let object = block_on_lsm(&shared_lsm_handle(), async {
        store.get(&ObjPath::from(location)).await?.bytes().await
    });
    object.unwrap().to_vec()
}

/// The pass leaves the collection alone, in a dry run and in a real one, and
/// neither writes nor deletes anything.
fn assert_left_alone(store: &Arc<dyn ObjectStore>, path: &str, reason: &'static str) {
    let before = objects(store, path);
    for dry_run in ["1", "0"] {
        let _mode = EnvGuard::set(ENV_IDLE_GC_DRY_RUN, dry_run);
        assert_eq!(
            pass(store, path),
            ClosedGcOutcome::Skipped(reason),
            "dry_run={dry_run}"
        );
        assert_eq!(objects(store, path), before, "dry_run={dry_run}");
    }
}

#[test]
#[serial]
fn collection_missing_an_object_its_manifest_needs_is_left_alone() {
    let _pass_env = pass_env();
    let _interval = EnvGuard::set(ENV_GC_INTERVAL_SECS, "3600");
    let _min_age = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "1");
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "helion/closed_gc_not_intact";
    collection_with_stale_manifests(&store, path);
    let rt = shared_lsm_handle();
    let admin = slatedb::admin::AdminBuilder::new(path, Arc::clone(&store)).build();
    let manifest = block_on_lsm(&rt, admin.read_manifest(None))
        .unwrap()
        .expect("latest manifest");

    // An SST the latest manifest names is gone.
    let sst = format!(
        "{path}/compacted/{}.sst",
        manifest.l0()[0].sst.id.unwrap_compacted_id()
    );
    let bytes = block_on_lsm(&rt, async {
        store.get(&ObjPath::from(sst.as_str())).await?.bytes().await
    })
    .unwrap();
    delete_object(&store, &sst);
    assert_left_alone(&store, path, "not_intact");
    put_object(&store, &sst, bytes.to_vec());

    // A WAL object an open would replay is gone: the one after it is there.
    // Both objects just put have to be past the minimum age for the pass to
    // get as far as looking for it.
    let last = last_wal_id(&store, path).max(manifest.replay_after_wal_id());
    let after_the_hole = format!("{path}/wal/{:020}.sst", last + 2);
    put_object(&store, &after_the_hole, b"wal".to_vec());
    std::thread::sleep(Duration::from_millis(1_200));
    assert_left_alone(&store, path, "not_intact");
    delete_object(&store, &after_the_hole);

    // Whole again, it is collected.
    let _deleting = EnvGuard::set(ENV_IDLE_GC_DRY_RUN, "0");
    assert!(collect(&store, path).objects > 0);
}

/// A prefix that got an SST or a WAL object less than the SST minimum age ago
/// is one a writer may still have open, or a copy may still be filling. The
/// manifest a pass reads there may not be the last one.
#[test]
#[serial]
fn collection_written_within_the_minimum_age_is_left_alone() {
    let _pass_env = pass_env();
    let _interval = EnvGuard::set(ENV_GC_INTERVAL_SECS, "3600");
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "helion/closed_gc_recent";
    collection_with_stale_manifests(&store, path);

    // Closed seconds ago: under an hour's minimum age that is too recent.
    {
        let _hour = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "3600");
        assert_left_alone(&store, path, "recently_written");
    }

    // Quiet for the minimum age, a pass would run. One object put again under
    // either directory, as a copy or a restore does, starts the wait over.
    let _min_age = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "3");
    for directory in ["compacted", "wal"] {
        std::thread::sleep(Duration::from_millis(3_200));
        let dry = collect(&store, path);
        assert!(dry.candidates > 0 && dry.objects == 0, "{dry:?}");
        let rewritten = objects(&store, path)
            .into_keys()
            .find(|location| location.starts_with(&format!("{path}/{directory}/")))
            .expect("an object to put again");
        put_object(&store, &rewritten, object_bytes(&store, &rewritten));
        assert_left_alone(&store, path, "recently_written");
    }

    std::thread::sleep(Duration::from_millis(3_200));
    let _deleting = EnvGuard::set(ENV_IDLE_GC_DRY_RUN, "0");
    assert!(collect(&store, path).objects > 0);
}

/// The wait after the last SST or WAL write has a knob of its own. A shorter
/// one lets a pass look sooner; the minimum ages still decide what goes.
#[test]
#[serial]
fn quiet_period_knob_sets_the_wait_after_the_last_write() {
    // The writer's ages are the pass's: an hour for every directory.
    let _env = compacting_writer_env();
    let _pass_env = pass_env();
    let _deleting = EnvGuard::set(ENV_IDLE_GC_DRY_RUN, "0");
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "helion/closed_gc_quiet";
    compacted_collection(&store, path);
    let arrival = format!("{path}/compacted/00000000010000000000000000.sst");

    // Closed a second ago. Without the knob the wait is the SST minimum age.
    assert_left_alone(&store, path, "recently_written");

    // An SST that is old by its name and in no manifest arrives, as a copy
    // would bring it. SlateDB takes it for garbage at once. For the SST age
    // the wait kept a pass away from it; under a shorter wait only the filter
    // does.
    put_object(&store, &arrival, b"sst".to_vec());
    std::thread::sleep(Duration::from_millis(1_200));
    {
        // With the knob a second of quiet is enough. The pass runs and takes
        // nothing: no object is an hour old, and the SST has not been in the
        // bucket for an hour.
        let _quiet = EnvGuard::set(ENV_IDLE_GC_QUIET_SECS, "1");
        let before = objects(&store, path);
        let pass = collect(&store, path);
        assert_eq!((pass.candidates, pass.objects), (0, 0), "{pass:?}");
        let after = objects(&store, path);
        assert!(after.contains_key(&arrival), "the SST that arrived went");
        assert!(before.keys().all(|location| after.contains_key(location)));
    }

    // The knob can ask for a longer wait than the SST minimum age as well.
    let _min_age = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "1");
    {
        let _quiet = EnvGuard::set(ENV_IDLE_GC_QUIET_SECS, "3600");
        assert_left_alone(&store, path, "recently_written");
    }

    // Not a number, it is the SST minimum age again. What the collection
    // stopped referencing goes, and the SST with it: in the bucket for its age
    // by now, it was a candidate all along.
    let _quiet = EnvGuard::set(ENV_IDLE_GC_QUIET_SECS, "soon");
    assert!(collect(&store, path).objects > 0);
    assert!(!objects(&store, path).contains_key(&arrival));
}

/// A copy over the collection that begins once the pass has looked at it. The
/// SST it brings is old by its name and in no manifest the prefix holds yet,
/// so SlateDB takes it for garbage the moment its SST task lists it.
#[test]
#[serial]
fn sst_that_arrives_during_a_pass_is_kept_until_it_has_aged() {
    let _env = compacting_writer_env();
    let _pass_env = pass_env();
    let bucket: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "helion/closed_gc_arrival";
    compacted_collection(&bucket, path);
    let _min_age = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "3");
    let _deleting = EnvGuard::set(ENV_IDLE_GC_DRY_RUN, "0");
    // The fixture slept 1.2 s. All it wrote has to be three seconds old.
    std::thread::sleep(Duration::from_millis(2_200));

    // Named for the first millisecond of 1970. The store puts it on the pass's
    // first delete, which is the manifest task's, two tasks ahead of the SSTs'.
    let arrival = ObjPath::from(format!("{path}/compacted/00000000010000000000000000.sst"));
    let deleted = Arc::new(std::sync::Mutex::new(Vec::new()));
    let store: Arc<dyn ObjectStore> = Arc::new(RacingReaderStore {
        inner: Arc::clone(&bucket),
        zombie: arrival.clone(),
        append_after_lists: 0,
        fail_deletes_after_lists: usize::MAX,
        lists: Default::default(),
        appended: Default::default(),
        deleted: Arc::clone(&deleted),
        delete_calls: Default::default(),
    });
    let ssts = format!("{path}/compacted/");

    collect(&store, path);
    let went = deleted.lock().unwrap().clone();
    assert!(
        went.first()
            .is_some_and(|first| !first.as_ref().starts_with(&ssts)),
        "the SST arrived after the SST task had listed: {went:?}"
    );
    // The older SSTs nothing references went. The one that had just arrived
    // stayed.
    assert!(
        went.iter()
            .any(|location| location.as_ref().starts_with(&ssts)),
        "no older SST was collected beside it: {went:?}"
    );
    assert!(!went.contains(&arrival));
    assert!(objects(&bucket, path).contains_key(arrival.as_ref()));

    // It was a candidate all along: in the bucket for the minimum age, it goes.
    std::thread::sleep(Duration::from_millis(3_200));
    collect(&store, path);
    assert!(!objects(&bucket, path).contains_key(arrival.as_ref()));
}

#[test]
#[serial]
fn collection_whose_manifest_names_no_sst_is_left_alone() {
    let _pass_env = pass_env();
    let _interval = EnvGuard::set(ENV_GC_INTERVAL_SECS, "3600");
    let _min_age = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "1");
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "helion/closed_gc_empty";
    {
        // As in `collection_with_stale_manifests`, without the rows.
        let _keep = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "3600");
        for _open in 0..2 {
            let be = LsmBackend::open_with_store(path, Arc::clone(&store)).unwrap();
            be.close().unwrap();
        }
    }
    std::thread::sleep(Duration::from_millis(1_200));
    // There are stale manifests, but no SST to show the latest one belongs to
    // the objects around it.
    assert!(manifest_ids(&shared_lsm_handle(), &store, path).len() > 1);
    assert_left_alone(&store, path, "empty");
}

/// A collection dropped and created again, with a manifest of the dropped one
/// landing afterwards, above every manifest of the new one: a reader of the
/// old incarnation appending its lease late. SlateDB takes the highest
/// manifest for the latest, so a pass would reap from it, then delete the new
/// incarnation's WAL and its real manifests as garbage.
#[test]
#[serial]
fn manifest_of_a_dropped_incarnation_above_a_recreated_one_stops_the_pass() {
    let dropped_env = pass_env();
    let _interval = EnvGuard::set(ENV_GC_INTERVAL_SECS, "3600");
    let _min_age = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "1");
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "helion/closed_gc_shadowed";
    let rt = shared_lsm_handle();
    let manifest_object = |id: u64| format!("{path}/manifest/{id:020}.manifest");

    collection_with_stale_manifests(&store, path);
    let dropped = *manifest_ids(&rt, &store, path).last().unwrap();
    let late_append = block_on_lsm(&rt, async {
        store
            .get(&ObjPath::from(manifest_object(dropped)))
            .await?
            .bytes()
            .await
    })
    .unwrap();
    // Purged by an earlier writer process: this one holds no record of the
    // manifest ids the prefix had, and a purge takes the cache directory too.
    for location in objects(&store, path).keys() {
        delete_object(&store, location);
    }
    drop(dropped_env);
    let _pass_env = pass_env();

    // Created again: manifest and WAL ids start over.
    collection_with_stale_manifests(&store, path);
    let recreated = manifest_ids(&rt, &store, path);
    let stale = recreated.last().unwrap() + 100;
    put_object(&store, &manifest_object(stale), late_append.to_vec());

    // The SSTs the stale manifest names went with the drop.
    assert_left_alone(&store, path, "not_intact");

    // What the pass left in place is what brings the collection back: without
    // the stale manifest it opens on the rows of the new incarnation.
    delete_object(&store, &manifest_object(stale));
    let be = LsmBackend::open_with_store(path, Arc::clone(&store)).unwrap();
    let r = be.begin_read().unwrap();
    let mut keys = Vec::new();
    be.scan(&r, Namespace::Nodes, KeyRange::all(), |key, _value| {
        keys.push(key.to_vec());
        true
    })
    .unwrap();
    assert_eq!(keys, vec![b"k1".to_vec(), b"k2".to_vec()]);
}

/// What a pass does about a reader's lease: a checkpoint with an expiry in the
/// latest manifest. The fixture's is the compactor's, which names the manifest
/// from before the compaction and so holds SSTs the latest one no longer does.
#[test]
#[serial]
fn pass_keeps_what_a_live_lease_holds_and_waits_out_one_that_just_lapsed() {
    let _env = compacting_writer_env();
    let _pass_env = pass_env();
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = "helion/closed_gc_lease";
    compacted_collection(&store, path);
    let _min_age = EnvGuard::set(ENV_GC_MIN_AGE_SECS, "1");
    let _deleting = EnvGuard::set(ENV_IDLE_GC_DRY_RUN, "0");
    let rt = shared_lsm_handle();
    let admin = slatedb::admin::AdminBuilder::new(path, Arc::clone(&store)).build();
    let leases = || block_on_lsm(&rt, admin.list_checkpoints(None)).unwrap();
    let set_lifetime = |lifetime: Duration| {
        for lease in leases() {
            block_on_lsm(&rt, admin.refresh_checkpoint(lease.id, Some(lifetime))).unwrap();
        }
    };
    // A pass as the sweep makes it: twenty minutes' grace, two to finish in.
    let sweep = || gc_closed_collection_with_store(Arc::clone(&store), path, unwanted()).unwrap();

    // The fixture's leases ran out a moment ago. A reader may only be late
    // refreshing one, so the collection waits for a later sweep.
    let before = objects(&store, path);
    assert_eq!(sweep(), ClosedGcOutcome::Skipped("lease_lapsing"));
    assert_eq!(objects(&store, path), before);

    // A live lease does not hold the pass back, and loses nothing to it: the
    // lease stays, with the manifest it names and every SST that one needs.
    set_lifetime(Duration::from_secs(600));
    let held = leases();
    assert!(!held.is_empty());
    let ClosedGcOutcome::Collected(swept) = sweep() else {
        panic!("a live lease held the pass back");
    };
    assert!(swept.objects > 0 && !swept.incomplete, "{swept:?}");
    assert_eq!(leases(), held);
    let after = objects(&store, path);
    let mut held_ssts = Vec::new();
    for lease in &held {
        let manifest = block_on_lsm(&rt, admin.read_manifest(Some(lease.manifest_id)))
            .unwrap()
            .expect("the manifest a live lease names");
        let views = manifest.l0().iter().chain(
            manifest
                .compacted()
                .iter()
                .flat_map(|run| run.sst_views.iter()),
        );
        for view in views {
            held_ssts.push(format!(
                "{path}/compacted/{}.sst",
                view.sst.id.unwrap_compacted_id()
            ));
        }
    }
    for sst in &held_ssts {
        assert!(
            after.contains_key(sst),
            "{sst} went while a live lease held it"
        );
    }

    // Lapsed for longer than the grace, the lease goes, and with it the SSTs
    // only it held.
    set_lifetime(Duration::ZERO);
    std::thread::sleep(Duration::from_millis(1_200));
    let outcome = gc_closed_collection_until(
        Arc::clone(&store),
        path,
        Instant::now() + Duration::from_secs(120),
        Duration::from_secs(1),
        unwanted(),
    )
    .unwrap();
    assert!(
        matches!(outcome, ClosedGcOutcome::Collected(pass) if pass.objects > 0),
        "{outcome:?}"
    );
    assert!(leases().is_empty());
    let after = objects(&store, path);
    assert!(
        held_ssts.iter().any(|sst| !after.contains_key(sst)),
        "the lapsed lease still holds every SST it named"
    );
}

#[test]
fn lease_holds_a_pass_back_only_while_it_is_lapsing() {
    let now = chrono::Utc::now();
    let expires_in = |secs: i64| now + chrono::Duration::seconds(secs);
    let grace = Duration::from_secs(20 * 60);
    let time_left = Duration::from_secs(120);
    let holds = |expires| lapsing(expires, now, grace, time_left);

    // A live lease: SlateDB keeps what it names.
    assert!(!holds(expires_in(600)));
    assert!(!holds(expires_in(121)));
    // One that runs out under the pass, which strips expired leases before
    // each directory.
    assert!(holds(expires_in(119)));
    assert!(holds(expires_in(0)));
    // Expired five minutes ago: its reader may only be late.
    assert!(holds(expires_in(-5 * 60)));
    assert!(holds(expires_in(-19 * 60)));
    // Expired an hour ago: its reader is gone.
    assert!(!holds(expires_in(-20 * 60)));
    assert!(!holds(expires_in(-3_600)));
}

#[test]
fn listing_the_pass_has_to_stop_ends_in_an_error_and_nothing_more_is_answered() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let bucket: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let wanted = Arc::new(AtomicBool::new(false));
    let gate_wanted = {
        let wanted = Arc::clone(&wanted);
        Arc::new(move || wanted.load(Ordering::SeqCst))
    };
    let store = PassStore::new(
        Arc::clone(&bucket),
        Instant::now() + Duration::from_secs(60),
        gate_wanted,
    );
    let object = |id: u64| ObjPath::from(format!("c/wal/{id:020}.sst"));
    shared_lsm_handle().block_on(async {
        for id in 0..5 {
            bucket
                .put(&object(id), b"wal".to_vec().into())
                .await
                .unwrap();
        }
        let mut listing = store.list(Some(&ObjPath::from("c")));
        assert!(listing.next().await.unwrap().is_ok());
        assert!(listing.next().await.unwrap().is_ok());

        // Something wants the collection while the listing is under way. The
        // listing ends in an error: cut short, it must not read as complete.
        wanted.store(true, Ordering::SeqCst);
        assert!(listing.next().await.unwrap().is_err());
        assert!(listing.next().await.is_none());

        assert!(store.head(&object(0)).await.is_err());
        assert!(store.put(&object(9), b"wal".to_vec().into()).await.is_err());
        assert!(store.delete(&object(0)).await.is_err());
        // Counted, so that the sweep can say how many deletes the pass left.
        assert!(store.delete(&object(1)).await.is_err());
        assert_eq!(store.deletes_not_started(), 2);
        let left: Vec<object_store::ObjectMeta> = bucket.list(None).try_collect().await.unwrap();
        assert_eq!(left.len(), 5);
    });
}
