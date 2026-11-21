//! A create that follows a drop in the same process: the database it opens
//! must be one it made, whatever is put under the collection's prefix after
//! its purge (`CollectionManager::open_new_database_after_purge`). And a
//! drop whose purge had to be retried: its name is released by a grace
//! sweep, as any drop's is, and not by the retried purge. And the two
//! poisoned locks that used to end in that create's purge of a collection
//! that was never dropped.

use super::idle_gc::{put_row, Writer, ROWS};
use super::*;
use crate::helix_engine::storage_core::backend::StorageBackend;
use crate::helix_engine::storage_core::backend_lsm::shared_lsm_handle;
use futures::{stream, StreamExt};
use slatedb::object_store::memory::InMemory;
use slatedb::object_store::ObjectStore;
use std::future::Future;
use std::sync::atomic::AtomicUsize;

const NAME: &str = "again";

/// What a reader of a dropped collection can do to the collection created
/// after it. Its poll loaded the dropped collection's manifest, and the
/// checkpoint it then writes is that manifest again, one id on, put with a
/// second request: one that can land in the prefix the drop has emptied.
///
/// This store remembers the last manifest put under the collection until a
/// test arms it, and from then on puts that manifest back one id on, either
/// when the last listing of a purge has been read or under the first
/// manifest a new database writes. Clones share their state, so a test keeps
/// one to steer the store it gave the manager.
#[derive(Debug, Clone)]
struct LatePut {
    inner: Arc<dyn ObjectStore>,
    /// `<root>/<collection>`, as a purge lists it.
    collection: ObjPath,
    last_manifest: Arc<Mutex<Option<(u64, object_store::PutPayload)>>>,
    /// The dropped collection's last manifest, once a test has armed.
    late: Arc<Mutex<Option<(u64, object_store::PutPayload)>>>,
    /// Purges still to be followed by a late put.
    after_purges: Arc<AtomicUsize>,
    /// First manifest writes still to find a late put at their id.
    at_first_writes: Arc<AtomicUsize>,
    collection_listings: Arc<AtomicUsize>,
    put_back: Arc<AtomicUsize>,
}

impl LatePut {
    fn under(root: &str) -> Self {
        Self {
            inner: Arc::new(InMemory::new()),
            collection: ObjPath::from(format!("{root}/{NAME}")),
            last_manifest: Arc::default(),
            late: Arc::default(),
            after_purges: Arc::default(),
            at_first_writes: Arc::default(),
            collection_listings: Arc::default(),
            put_back: Arc::default(),
        }
    }

    fn arm(&self) {
        let last = self.last_manifest.lock().unwrap().clone();
        assert!(
            last.is_some(),
            "no manifest was put under {}",
            self.collection
        );
        *self.late.lock().unwrap() = last;
        self.collection_listings.store(0, Ordering::SeqCst);
    }

    /// The next `purges` purges of the collection are each followed by a
    /// late put: it lands when the purge's last listing has been read. A
    /// purge lists the whole collection twice and nothing else lists it.
    fn late_put_after_purges(&self, purges: usize) {
        self.arm();
        self.after_purges.store(purges, Ordering::SeqCst);
    }

    /// The next new database finds a late put at the id of its first
    /// manifest: it landed after the open looked for a manifest.
    fn late_put_under_the_first_manifest(&self) {
        self.arm();
        self.at_first_writes.store(1, Ordering::SeqCst);
    }

    fn put_back(&self) -> usize {
        self.put_back.load(Ordering::SeqCst)
    }

    fn manifest_id(&self, location: &ObjPath) -> Option<u64> {
        location
            .as_ref()
            .strip_prefix(&format!("{}/manifest/", self.collection))?
            .strip_suffix(".manifest")?
            .parse()
            .ok()
    }

    fn take(count: &AtomicUsize) -> bool {
        count
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                left.checked_sub(1)
            })
            .is_ok()
    }
}

impl std::fmt::Display for LatePut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LatePut({})", self.inner)
    }
}

type Boxed<'a, T> = std::pin::Pin<Box<dyn Future<Output = object_store::Result<T>> + Send + 'a>>;

impl ObjectStore for LatePut {
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
        let Some(id) = self.manifest_id(location) else {
            return self.inner.put_opts(location, payload, opts);
        };
        let late = self.late.lock().unwrap().clone();
        let Some((_, stale)) = late else {
            *self.last_manifest.lock().unwrap() = Some((id, payload.clone()));
            return self.inner.put_opts(location, payload, opts);
        };
        if opts.mode != object_store::PutMode::Create || !Self::take(&self.at_first_writes) {
            return self.inner.put_opts(location, payload, opts);
        }
        Box::pin(async move {
            self.inner
                .put_opts(location, stale, object_store::PutOptions::default())
                .await?;
            self.put_back.fetch_add(1, Ordering::SeqCst);
            self.inner.put_opts(location, payload, opts).await
        })
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
        self.inner.put_multipart_opts(location, opts)
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
        self.inner.get_opts(location, options)
    }

    fn delete_stream(
        &self,
        locations: stream::BoxStream<'static, object_store::Result<ObjPath>>,
    ) -> stream::BoxStream<'static, object_store::Result<ObjPath>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&ObjPath>,
    ) -> stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        let listing = self.inner.list(prefix);
        if prefix != Some(&self.collection) {
            return listing;
        }
        let nth = self.collection_listings.fetch_add(1, Ordering::SeqCst) + 1;
        if nth % 2 != 0 || !Self::take(&self.after_purges) {
            return listing;
        }
        let Some((id, stale)) = self.late.lock().unwrap().clone() else {
            return listing;
        };
        let location = ObjPath::from(format!(
            "{}/manifest/{:020}.manifest",
            self.collection,
            id + 1
        ));
        let (inner, put_back) = (Arc::clone(&self.inner), Arc::clone(&self.put_back));
        stream::once(async move {
            let listed: Vec<_> = listing.collect().await;
            inner
                .put_opts(&location, stale, object_store::PutOptions::default())
                .await
                .expect("late put");
            put_back.fetch_add(1, Ordering::SeqCst);
            stream::iter(listed)
        })
        .flatten()
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
        self.inner.list_with_delimiter(prefix)
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
        self.inner.copy_opts(from, to, options)
    }
}

/// What a read of `key` answers: `Ok(true)`, `Ok(false)` or the error.
fn shows(storage: &HelixGraphStorage, key: &[u8]) -> String {
    match storage.backend.begin_read() {
        Err(error) => format!("{error:?}"),
        Ok(txn) => format!(
            "{:?}",
            storage
                .backend
                .get_with(&txn, ROWS, key, |value| value.is_some())
        ),
    }
}

/// A writer whose collection held the row `before` and has been dropped.
fn dropped(root: &'static str) -> (Writer, LatePut) {
    let store = LatePut::under(root);
    let writer = Writer::over(root, Arc::new(store.clone()));
    let source = writer.manager.create_collection(NAME).unwrap();
    put_row(&source, b"before", b"v");
    drop(source);
    writer.manager.drop_collection(NAME).unwrap();
    (writer, store)
}

/// The created collection is empty, takes a row, and is what a later open
/// of the name finds.
fn assert_new(writer: &Writer, created: Arc<HelixGraphStorage>) {
    assert_eq!(
        shows(&created, b"before"),
        "Ok(false)",
        "the create opened the dropped collection's manifest"
    );
    put_row(&created, b"after", b"v");
    drop(created);
    writer.close(NAME);
    let reopened = writer.manager.get_collection(NAME).unwrap();
    assert_eq!(shows(&reopened, b"after"), "Ok(true)");
    assert_eq!(shows(&reopened, b"before"), "Ok(false)");
}

/// The late put lands after the create's purge and before its open looks
/// for a manifest. The open used to take what it found: the collection came
/// back with the manifest of the dropped one, whose objects are gone.
#[test]
fn create_after_a_drop_opens_no_manifest_put_back_before_its_open() {
    let (writer, store) = dropped("create_after_drop_before_open");
    store.late_put_after_purges(1);

    let created = writer.manager.create_collection(NAME).unwrap();

    assert_eq!(store.put_back(), 1, "the late put never landed");
    assert_new(&writer, created);
}

/// The late put lands after the open found no manifest and before it wrote
/// its first: at that manifest's id, as the first id of a database created
/// after a purge is one above the highest the purge removed. The open
/// failed, and the name's next open took the late put.
#[test]
fn create_after_a_drop_opens_no_manifest_put_back_under_its_first_write() {
    let (writer, store) = dropped("create_after_drop_first_write");
    store.late_put_under_the_first_manifest();

    let created = writer.manager.create_collection(NAME).unwrap();

    assert_eq!(store.put_back(), 1, "the late put never landed");
    assert_new(&writer, created);
}

/// A manifest put back before every open: the create gives up, and it has
/// opened none of them. The name is then where the drop left it, so a read
/// does not open what is under the prefix, and the next create purges
/// before it opens.
#[test]
fn create_after_a_drop_gives_up_when_a_manifest_is_put_back_before_every_open() {
    let (writer, store) = dropped("create_after_drop_every_open");
    store.late_put_after_purges(usize::MAX);

    let refused = writer
        .manager
        .create_collection(NAME)
        .map(|_| ())
        .expect_err("the create opened a manifest that was put back");

    assert!(
        matches!(refused, GraphError::PurgePending(_)),
        "{refused:?}"
    );
    // One after the create's own purge, and one after each of the three
    // purges that follow an open that found a manifest.
    assert_eq!(store.put_back(), 4);
    assert!(writer
        .manager
        .get_loaded_collection(NAME)
        .unwrap()
        .is_none());
    // Under its tombstone, for a writer that starts after this one, and
    // under this process's pending mark.
    let path = writer.manager.collection_path(NAME);
    assert!(backend_any::lsm_drop_tombstone_exists_from_env(
        &path,
        writer.manager.config.storage_backend
    )
    .unwrap());
    assert!(lsm_purge_pending(&path));
    let read = writer
        .manager
        .get_collection(NAME)
        .map(|_| ())
        .expect_err("a read opened the manifest that was put back");
    assert!(read.to_string().contains("purge pending"), "{read:?}");

    // When the late puts stop, the last of them is still under the prefix.
    store.late_put_after_purges(0);
    let created = writer.manager.create_collection(NAME).unwrap();
    assert_new(&writer, created);
}

/// The other side of the rule: a collection that is there, opened with no
/// drop pending in this process, is opened as it is. Nothing is purged and
/// its rows are read, as when a writer opens a collection after a restart
/// or one it no longer has loaded.
#[test]
fn open_of_a_collection_with_no_drop_pending_keeps_what_is_there() {
    let writer = Writer::new("create_after_drop_no_drop");
    // The writer's own collector ticks at an open: keep it off these objects.
    let _keep = EnvRestore::set("HELIX_LSM_GC_MIN_AGE_SECS", "3600");
    let source = writer.manager.create_collection("kept").unwrap();
    put_row(&source, b"before", b"v");
    drop(source);
    writer.close("kept");
    let closed = writer.objects();

    let opened = writer.manager.get_or_create_collection("kept").unwrap();

    assert_eq!(shows(&opened, b"before"), "Ok(true)");
    let now = writer.objects();
    let deleted: Vec<&String> = closed
        .keys()
        .filter(|key| !now.contains_key(*key))
        .collect();
    assert!(deleted.is_empty(), "the open deleted {deleted:?}");
}

/// A drop whose purge failed gets its first purge from the retry worker. As
/// after a drop whose own purge went through, that purge does not release
/// the name: the tombstone and the pending mark stay for the grace, and the
/// sweep that ends it, on the same pending entry, is what releases it. The
/// readers are told when the tombstone is cleared: once, by that sweep.
#[test]
fn retried_purge_keeps_the_name_for_its_grace() {
    const RETRIED: &str = "retried_purge_keeps_the_name";
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("collections").join(RETRIED);
    let gate = Mutex::new(());
    let token = lsm_purge_pending_register(&path);
    let feed = crate::helix_engine::storage_core::change_feed::change_feed();
    let polled = feed.changes_since(0).cursor;
    let told = || {
        let released = feed.changes_since(polled).recreated;
        released.iter().filter(|name| *name == RETRIED).count()
    };

    let step = lsm_purge_retry_attempt(
        &path,
        token,
        LsmPurgeWorker::Retry,
        &gate,
        || Ok(true),
        || Ok(()),
        || panic!("the retried purge released the name"),
        RETRIED,
        1,
    );
    assert_eq!(step, LsmPurgeRetryStep::Purged);
    assert!(lsm_purge_pending(&path));
    assert_eq!(told(), 0, "the retried purge told the readers");

    let step = lsm_purge_retry_attempt(
        &path,
        token,
        LsmPurgeWorker::Grace,
        &gate,
        || Ok(true),
        || Ok(()),
        || Ok(()),
        RETRIED,
        1,
    );
    assert_eq!(step, LsmPurgeRetryStep::Purged);
    assert!(!lsm_purge_pending(&path));
    assert_eq!(told(), 1);
}

fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !done() {
        assert!(Instant::now() < deadline, "{what}");
        thread::sleep(Duration::from_millis(20));
    }
}

/// The same through the worker. A reader that loaded the manifest before
/// the retried purge can put one more version under the emptied prefix.
/// With the name released at that purge, the next create found no drop
/// pending and opened it. The grace sweep now comes after a retried purge
/// too, and takes the late put with it.
#[test]
fn drop_whose_purge_was_retried_is_swept_again_before_its_name_is_released() {
    let root = "create_after_drop_retried_purge";
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let writer = Writer::over(root, Arc::clone(&store));
    let _grace = EnvRestore::set("HELIX_LSM_DROP_TOMBSTONE_GRACE_MS", "2000");
    let _first = EnvRestore::set("HELIX_LSM_TEST_PURGE_RETRY_FIRST_DELAY_MS", "200");
    let source = writer.manager.create_collection(NAME).unwrap();
    put_row(&source, b"before", b"v");
    drop(source);
    let path = writer.manager.collection_path(NAME);
    let backend = writer.manager.config.storage_backend;
    let tombstone = || backend_any::lsm_drop_tombstone_exists_from_env(&path, backend).unwrap();
    let prefix = format!("{root}/{NAME}/");
    let under_prefix = || {
        writer
            .objects()
            .keys()
            .filter(|key| key.starts_with(&prefix))
            .count()
    };

    // Given a drop whose purge fails, and the retry that then goes through.
    {
        let _fail = EnvRestore::set("HELIX_LSM_TEST_FAIL_PURGE", "1");
        writer.manager.drop_collection(NAME).unwrap();
    }
    assert!(under_prefix() > 0, "the drop's purge went through");
    wait_until("the retry never purged", || under_prefix() == 0);
    thread::sleep(Duration::from_millis(200));

    // Then the name is not released yet.
    assert!(tombstone(), "the retried purge released the name");
    assert!(lsm_purge_pending(&path));

    // When a reader's late put lands under the prefix.
    let late = ObjPath::from(format!("{prefix}manifest/00000000000000000099.manifest"));
    shared_lsm_handle().block_on(async {
        store
            .put_opts(
                &late,
                b"late".to_vec().into(),
                object_store::PutOptions::default(),
            )
            .await
            .unwrap();
    });

    // Then the sweep at the end of the grace takes it, and releases the name.
    wait_until("the grace sweep never ran", || !lsm_purge_pending(&path));
    assert!(!tombstone());
    assert_eq!(under_prefix(), 0, "the name was released over a late put");
    let created = writer.manager.create_collection(NAME).unwrap();
    assert_new(&writer, created);
}

/// Leaves `lock` poisoned, as a thread that panicked while it held it does.
fn poison<T: Send>(lock: &Mutex<T>) {
    thread::scope(|scope| {
        let panicked = scope
            .spawn(|| {
                let _held = lock.lock().unwrap();
                panic!("poisons the lock");
            })
            .join();
        assert!(panicked.is_err());
    });
}

/// A create purges on the word of the registry of pending purges. Poisoned,
/// the registry cannot say, and it used to be read as a yes: a purge of
/// whatever collection is at the next name created or got-or-created, live
/// or not. The create is refused instead.
#[test]
fn poisoned_registry_of_pending_purges_refuses_the_create_that_would_purge() {
    let registry = Mutex::new(HashMap::new());
    let path = Path::new("collections/live");
    assert!(!lsm_purge_known_pending(&registry, path, "live").unwrap());
    registry.lock().unwrap().insert(path.to_path_buf(), 1);
    assert!(lsm_purge_known_pending(&registry, path, "live").unwrap());

    poison(&registry);

    let refused = lsm_purge_known_pending(&registry, path, "live")
        .expect_err("a poisoned registry was taken for a pending purge");
    assert!(refused.to_string().contains("poisoned"), "{refused:?}");
}

/// A drop that fails on a poisoned collection map leaves the collection
/// loaded and serving. It wrote the drop tombstone before it met the map,
/// and the tombstone stayed: the next create of the name, once the
/// collection had left the cache, purged it, though the drop had answered an
/// error. The drop now looks at the map first and writes nothing.
#[test]
fn drop_refused_by_a_poisoned_collection_map_leaves_no_tombstone() {
    let writer = Writer::new("create_after_drop_poisoned_map");
    let source = writer.manager.create_collection(NAME).unwrap();
    put_row(&source, b"before", b"v");
    let path = writer.manager.collection_path(NAME);
    thread::scope(|scope| {
        let panicked = scope
            .spawn(|| {
                let _held = writer.manager.collections.write().unwrap();
                panic!("poisons the collection map");
            })
            .join();
        assert!(panicked.is_err());
    });

    let refused = writer
        .manager
        .drop_collection(NAME)
        .expect_err("the drop went through a poisoned collection map");

    assert!(refused.to_string().contains("poisoned"), "{refused:?}");
    assert!(
        !backend_any::lsm_drop_tombstone_exists_from_env(
            &path,
            writer.manager.config.storage_backend
        )
        .unwrap(),
        "the refused drop left its tombstone over a loaded collection"
    );
    assert!(!lsm_purge_pending(&path));
    assert!(!writer.manager.is_dropping(NAME));
    assert_eq!(shows(&source, b"before"), "Ok(true)");
}
