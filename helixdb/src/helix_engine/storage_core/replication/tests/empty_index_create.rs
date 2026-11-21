//! An index create on an empty LSM collection is one write, flushed before it
//! is acknowledged: it does not wait out the WAL flush interval twice.

use super::*;
use futures::TryStreamExt;
use object_store::{memory::InMemory, ObjectStore};

/// The fields the client creates for a new repository: 25 on its code collection and
/// 8 on its graph collection.
const FIELDS: usize = 33;
const TICK: Duration = Duration::from_millis(500);

fn lsm(role: LsmRole) -> Config {
    Config::new(16, 64, 128, 1).with_storage_backend(StorageBackendConfig::Lsm {
        storage: LsmStorage::ObjectStore,
        role,
    })
}

fn wal_objects(store: &InMemory) -> usize {
    futures::executor::block_on(store.list(None).try_collect::<Vec<_>>())
        .unwrap()
        .iter()
        .filter(|object| object.location.as_ref().contains("/wal/"))
        .count()
}

fn create_index(manager: &Arc<CollectionManager>, config: &Config, name: &str, field: &str) {
    apply_mutation(
        manager,
        config,
        &ReplicatedMutation::CreatePayloadIndex {
            collection: name.into(),
            field_name: field.into(),
            schema: PayloadIndexSchema::Keyword,
        },
    )
    .unwrap();
}

/// The record that `field`'s rows need no marker backfill, as stored.
fn marker(storage: &HelixGraphStorage, field: &str) -> Option<Vec<u8>> {
    let read = storage.backend.begin_read().unwrap();
    storage
        .backend
        .get_with(
            &read,
            Namespace::Metadata,
            format!("payload_marker_backfill:v1:{field}").as_bytes(),
            |row| row.map(<[u8]>::to_vec),
        )
        .unwrap()
}

/// The keyword indexes `storage` serves, by field.
fn indexed(storage: &HelixGraphStorage) -> Vec<String> {
    let indexes = storage.list_payload_indexes().unwrap();
    assert!(indexes
        .values()
        .all(|schema| *schema == PayloadIndexSchema::Keyword));
    let mut fields: Vec<String> = indexes.into_keys().collect();
    fields.sort();
    fields
}

#[test]
fn an_index_create_on_an_empty_collection_is_one_flush() {
    use crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store;
    let _guard = ENV_LOCK.lock().unwrap();
    let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "empty_index_create_lsm");
    let _interval = EnvVarGuard::set("HELIX_LSM_FLUSH_INTERVAL_MS", "500");
    let _switch = EnvVarGuard::unset("HELIX_LSM_FLUSH_POINT_WRITES");
    let store = Arc::new(InMemory::new());
    let _store = set_lsm_test_object_store(Arc::clone(&store) as Arc<dyn ObjectStore>);
    let (writer_dir, reader_dir) = (TempDir::new().unwrap(), TempDir::new().unwrap());
    let config = lsm(LsmRole::Writer);
    let writer =
        Arc::new(CollectionManager::new(writer_dir.path().to_path_buf(), config.clone()).unwrap());
    // A code collection with a dense space, and its graph collection, which
    // has no vectors and so no metadata at all before its first index.
    let (code, graph) = ("new_repo", "new_repo_graph");
    for (name, vectors) in [
        (
            code,
            HashMap::from([(
                "dense".to_string(),
                NamedVectorConfig {
                    size: 16,
                    distance: DistanceMetric::Cosine,
                    spindle: Default::default(),
                },
            )]),
        ),
        (graph, HashMap::new()),
    ] {
        apply_mutation(
            &writer,
            &config,
            &ReplicatedMutation::CreateCollection {
                name: name.into(),
                vectors,
                sparse_vectors: HashMap::new(),
                hnsw_overrides: None,
            },
        )
        .unwrap();
    }
    // A reader that opened the code collection before it had an index.
    let reader = Arc::new(
        CollectionManager::new(reader_dir.path().to_path_buf(), lsm(LsmRole::Reader)).unwrap(),
    );
    let replica = reader.get_collection(code).unwrap();
    assert!(replica.backend.is_reader_replica());
    assert!(indexed(&replica).is_empty());

    // (elapsed, WAL objects written) for these creates in a row.
    let creates = |fields: &[(&str, String)]| {
        let before = wal_objects(&store);
        let started = Instant::now();
        for (name, field) in fields {
            create_index(&writer, &config, name, field);
        }
        (started.elapsed(), wal_objects(&store) - before)
    };
    let named = |name: &'static str, prefix: &str, count: usize| -> Vec<(&'static str, String)> {
        (0..count)
            .map(|i| (name, format!("{prefix}{i:02}")))
            .collect()
    };
    let waited_fields = named(code, "waited_", 3);
    let waited = {
        let _off = EnvVarGuard::set("HELIX_LSM_FLUSH_POINT_WRITES", "0");
        creates(&waited_fields)
    };
    // The create above returned on a WAL tick, so the next tick is a whole
    // interval away: a WAL object written by these is their own flush.
    let flushed_fields = [named(code, "field_", 25), named(graph, "edge_", 8)].concat();
    assert_eq!(flushed_fields.len(), FIELDS);
    let flushed = creates(&flushed_fields);
    // What the object store holds when the last create is acknowledged.
    let acknowledged = store.fork();
    println!(
        "index creates on an empty collection: 3 with two durable commits each {:?} and {} WAL \
         objects; {FIELDS} with one flushed write each {:?} and {} WAL objects",
        waited.0, waited.1, flushed.0, flushed.1
    );

    let on = |name: &str| {
        let mut fields: Vec<String> = waited_fields
            .iter()
            .chain(&flushed_fields)
            .filter(|(collection, _)| *collection == name)
            .map(|(_, field)| field.clone())
            .collect();
        fields.sort();
        fields
    };
    assert_eq!(indexed(&writer.get_collection(code).unwrap()), on(code));
    assert_eq!(indexed(&writer.get_collection(graph).unwrap()), on(graph));
    assert_eq!((on(code).len(), on(graph).len()), (28, 8));

    // Two commits a create, each on its own tick: six, of which the first
    // may find a tick already close.
    assert!(waited.0 >= TICK * 4, "{waited:?}");
    assert!(waited.1 >= 6, "two WAL objects a create: {waited:?}");
    // All of them in less time than three took, and in less than a tick each.
    assert!(
        flushed.0 < waited.0 && flushed.0 < TICK * FIELDS as u32,
        "waited {waited:?}, flushed {flushed:?}"
    );
    assert!(
        (1..=FIELDS).contains(&flushed.1),
        "at most one WAL object a create: {flushed:?}"
    );

    // The reader that follows the writer serves them after its next refresh.
    replica
        .backend
        .refresh_lsm_reader_with_poll_interval(Duration::from_millis(100))
        .unwrap();
    replica.refresh_reader_view_best_effort();
    assert_eq!(indexed(&replica), on(code));

    // A writer that opens what the store held at the acknowledgement has
    // every index with its marker record: nothing was still only buffered.
    drop(replica);
    drop(reader);
    drop(writer);
    let _crash = set_lsm_test_object_store(Arc::new(acknowledged));
    let reopened_dir = TempDir::new().unwrap();
    let reopened =
        CollectionManager::new(reopened_dir.path().to_path_buf(), config.clone()).unwrap();
    for name in [code, graph] {
        let recovered = reopened.get_collection(name).unwrap();
        assert_eq!(indexed(&recovered), on(name), "{name}");
        for field in on(name) {
            assert_eq!(
                marker(&recovered, &field).as_deref(),
                Some(b"done".as_slice()),
                "{name} {field}"
            );
        }
    }
}

/// A collection with points takes the other route, a build in the
/// background, and still registers its index by durable commits.
#[test]
fn an_index_create_on_a_collection_with_points_is_built_as_before() {
    use crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store;
    let _guard = ENV_LOCK.lock().unwrap();
    let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "index_create_with_points_lsm");
    let _interval = EnvVarGuard::set("HELIX_LSM_FLUSH_INTERVAL_MS", "500");
    let _switch = EnvVarGuard::unset("HELIX_LSM_FLUSH_POINT_WRITES");
    let _store = set_lsm_test_object_store(Arc::new(InMemory::new()));
    let dir = TempDir::new().unwrap();
    let name = "repo_with_points";
    let (writer, config, storage, rows) =
        sealed_lsm_collection(dir.path(), name, &["dense"], 16, 8);

    let started = Instant::now();
    create_index(&writer, &config, name, "path");
    let deadline = started + Duration::from_secs(60);
    let status = loop {
        let status = storage.payload_index_status("path").unwrap().unwrap();
        if status.status != "building" {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "the index never finished building"
        );
        thread::sleep(Duration::from_millis(10));
    };
    let built_in = started.elapsed();
    println!(
        "index build on {} points: {built_in:?}, {status:?}",
        rows.len()
    );

    assert_eq!(status.status, "ready", "{:?}", status.error);
    // The build read every point, which the empty route never does.
    assert_eq!(status.indexed_nodes, rows.len() as u64);
    // Its registration is the two durable commits: the second waits for a
    // whole WAL flush interval.
    assert!(built_in >= TICK, "{built_in:?}");
    assert_eq!(
        marker(&storage, "path").as_deref(),
        Some(b"done".as_slice())
    );
}

/// A create whose registration fails leaves no index in memory: a retry has to
/// register it, and must not be answered "ready" from a handle that nothing on
/// the store backs.
#[test]
fn a_failed_index_registration_leaves_no_ready_index() {
    use crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store;
    let _guard = ENV_LOCK.lock().unwrap();
    let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "failed_index_registration_lsm");
    let _switch = EnvVarGuard::unset("HELIX_LSM_FLUSH_POINT_WRITES");
    let _store = set_lsm_test_object_store(Arc::new(InMemory::new()));
    let dir = TempDir::new().unwrap();
    let config = lsm(LsmRole::Writer);
    let writer =
        Arc::new(CollectionManager::new(dir.path().to_path_buf(), config.clone()).unwrap());
    let name = "failed_registration";
    apply_mutation(
        &writer,
        &config,
        &ReplicatedMutation::CreateCollection {
            name: name.into(),
            vectors: HashMap::new(),
            sparse_vectors: HashMap::new(),
            hnsw_overrides: None,
        },
    )
    .unwrap();
    create_index(&writer, &config, name, "kept");
    let storage = writer.get_collection(name).unwrap();

    // From here on the collection refuses every write, and so the registration
    // of the next index.
    storage.mark_collection_degraded(
        &GraphError::FatalCollectionStorage {
            code: "test_writes_refused",
            message: "writes refused".into(),
        },
        "test",
    );
    for attempt in ["create", "retry"] {
        let answer =
            storage.enqueue_payload_index_build(name, "lost", PayloadIndexSchema::Keyword, true);
        assert!(
            answer.is_err(),
            "{attempt}: {:?}",
            answer.map(|status| status.status)
        );
        assert!(
            storage.payload_index_status("lost").unwrap().is_none(),
            "{attempt}: the unregistered index is still in memory"
        );
    }
    assert_eq!(indexed(&storage), ["kept"]);
}
