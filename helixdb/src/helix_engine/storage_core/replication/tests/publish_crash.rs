use super::*;
use futures::stream::BoxStream;
use object_store::{memory::InMemory, path::Path as ObjPath, ObjectStore};
use std::future::Future;
use std::sync::atomic::AtomicBool;

/// An in-memory store that, while `armed`, keeps a copy of itself as it is
/// after each WAL object: what a writer that died there left behind.
#[derive(Debug, Default)]
pub(super) struct WalSnapshots {
    pub(super) inner: InMemory,
    pub(super) armed: AtomicBool,
    pub(super) snapshots: Mutex<Vec<InMemory>>,
}

impl std::fmt::Display for WalSnapshots {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WalSnapshots({})", self.inner)
    }
}

type Boxed<'a, T> = std::pin::Pin<Box<dyn Future<Output = object_store::Result<T>> + Send + 'a>>;

impl ObjectStore for WalSnapshots {
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
        Box::pin(async move {
            let result = self.inner.put_opts(location, payload, opts).await?;
            if self.armed.load(Ordering::Acquire) && location.as_ref().contains("/wal/") {
                self.snapshots.lock().unwrap().push(self.inner.fork());
            }
            Ok(result)
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
        opts: object_store::GetOptions,
    ) -> Boxed<'c, object_store::GetResult>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        self.inner.get_opts(location, opts)
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<ObjPath>>,
    ) -> BoxStream<'static, object_store::Result<ObjPath>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&ObjPath>,
    ) -> BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(prefix)
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
        opts: object_store::CopyOptions,
    ) -> Boxed<'d, ()>
    where
        'a: 'd,
        'b: 'd,
        'c: 'd,
        Self: 'd,
    {
        self.inner.copy_opts(from, to, opts)
    }
}

#[test]
fn real_index_publish_keeps_every_durable_entry_point_backed_by_the_full_graph() {
    use crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store;
    let _guard = ENV_LOCK.lock().unwrap();
    let _prefix = EnvVarGuard::set("HELIX_LSM_PREFIX", "real_publish_crash_test");
    let _group = EnvVarGuard::unset("HELIX_LSM_GROUP_COMMIT");
    let store = Arc::new(WalSnapshots::default());
    let _store = set_lsm_test_object_store(Arc::clone(&store) as Arc<dyn ObjectStore>);
    let dir = TempDir::new().unwrap();
    let (writer, _, storage, _) =
        sealed_lsm_collection(dir.path(), "real_publish_crash", &["dense"], 16, 64);
    let segment = storage
        .named_vectors
        .building_segment_names("dense")
        .remove(0);
    let prepared = {
        let r = storage.backend.begin_read().unwrap();
        storage
            .named_vectors
            .prepare_building_indices_limited_be(&r, "dense", 1)
            .unwrap()
    };
    assert_eq!(prepared.len(), 1);
    let before = store.inner.fork();

    store.armed.store(true, Ordering::Release);
    assert!(run_chunked_indices_publish(&storage, "dense", prepared).unwrap());
    store.armed.store(false, Ordering::Release);

    let full_graph = segment_dump(&storage, &segment);
    let mut snapshots = std::mem::take(&mut *store.snapshots.lock().unwrap());
    assert!(
        snapshots.len() >= 2,
        "must observe chunk flush and entry-point commit"
    );
    snapshots.insert(0, before);
    drop(storage);
    drop(writer);
    let mut published = 0;
    for snapshot in snapshots {
        let _crash = set_lsm_test_object_store(Arc::new(snapshot));
        let dir = TempDir::new().unwrap();
        let reader = CollectionManager::new(
            dir.path().to_path_buf(),
            Config::new(16, 64, 128, 1).with_storage_backend(StorageBackendConfig::Lsm {
                storage: LsmStorage::ObjectStore,
                role: LsmRole::Reader,
            }),
        )
        .unwrap();
        let replica = reader.get_collection("real_publish_crash").unwrap();
        let r = replica.backend.begin_read().unwrap();
        let indexed = replica.named_vectors.cores_read().unwrap()[&segment]
            .has_index(&r)
            .unwrap();
        drop(r);
        if indexed {
            assert_eq!(segment_dump(&replica, &segment), full_graph);
            published += 1;
        }
    }
    assert!(published > 0, "the durable finalization must be observed");
}
