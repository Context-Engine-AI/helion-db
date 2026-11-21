//! What the flush gate's tests share: an object store that keeps SST uploads
//! waiting and counts them, and a gate that reads its free memory from the
//! test.

use super::FlushGate;
use futures::stream;
use object_store::path::Path as ObjPath;
use slatedb::object_store::memory::InMemory;
use slatedb::object_store::ObjectStore;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

type Boxed<'a, T> = std::pin::Pin<Box<dyn Future<Output = object_store::Result<T>> + Send + 'a>>;

/// An in-memory object store in which the upload of an L0 SST waits while
/// its path is held, or fails while its path is refused, in one put or in
/// its first part. Everything else, the WAL and the manifest first, goes
/// through.
#[derive(Debug)]
pub(crate) struct HeldSsts {
    inner: InMemory,
    /// The uploads this store keeps waiting or fails: an L0 SST's
    /// (`/compacted/`), or a WAL's for [`Self::holding_wal`].
    within: &'static str,
    /// SST uploads whose path contains this wait to be released.
    hold: Mutex<Option<String>>,
    /// SST uploads whose path contains this fail.
    refuse: Mutex<Option<String>>,
    released: tokio::sync::Semaphore,
    held: AtomicUsize,
    most_held: AtomicUsize,
    refused: AtomicUsize,
    /// SST uploads in parts whose path contains this have their first part
    /// fail.
    refuse_part: Mutex<Option<String>>,
    part_uploads: Arc<PartUploads>,
}

/// How the uploads of SSTs in parts went: begun, aborted, completed.
#[derive(Debug, Default)]
struct PartUploads {
    begun: AtomicUsize,
    aborted: AtomicUsize,
    completed: AtomicUsize,
}

impl HeldSsts {
    /// A store that holds every SST upload whose path contains `path`.
    pub(crate) fn holding(path: &str) -> Arc<Self> {
        Self::holding_within("/compacted/", path)
    }

    /// A store that holds every WAL upload whose path contains `path`, and
    /// no L0 SST.
    pub(crate) fn holding_wal(path: &str) -> Arc<Self> {
        Self::holding_within("/wal/", path)
    }

    fn holding_within(within: &'static str, path: &str) -> Arc<Self> {
        Arc::new(Self {
            inner: InMemory::new(),
            within,
            hold: Mutex::new(Some(path.to_string())),
            refuse: Mutex::new(None),
            released: tokio::sync::Semaphore::new(0),
            held: AtomicUsize::new(0),
            most_held: AtomicUsize::new(0),
            refused: AtomicUsize::new(0),
            refuse_part: Mutex::new(None),
            part_uploads: Arc::default(),
        })
    }

    pub(crate) fn hold(&self, path: &str) {
        *self.hold.lock().unwrap() = Some(path.to_string());
    }

    /// Fail every SST upload whose path contains `path`; `None` ends it.
    pub(crate) fn refuse(&self, path: Option<&str>) {
        *self.refuse.lock().unwrap() = path.map(str::to_string);
    }

    /// Fail the first part of every SST upload in parts that begins while
    /// its path contains `path`; `None` ends it.
    pub(crate) fn refuse_part(&self, path: Option<&str>) {
        *self.refuse_part.lock().unwrap() = path.map(str::to_string);
    }

    /// SST uploads in parts so far: begun, aborted, completed.
    pub(crate) fn part_uploads(&self) -> (usize, usize, usize) {
        let uploads = &self.part_uploads;
        (
            uploads.begun.load(SeqCst),
            uploads.aborted.load(SeqCst),
            uploads.completed.load(SeqCst),
        )
    }

    /// Let `uploads` of the waiting SST uploads through.
    pub(crate) fn release(&self, uploads: usize) {
        self.released.add_permits(uploads);
    }

    /// Hold nothing more, and let every waiting upload through.
    pub(crate) fn release_all(&self) {
        *self.hold.lock().unwrap() = None;
        self.released.add_permits(1 << 16);
    }

    /// SST uploads that have arrived and wait.
    pub(crate) fn held(&self) -> usize {
        self.held.load(SeqCst)
    }

    /// The most SST uploads that waited at the same moment.
    pub(crate) fn most_held(&self) -> usize {
        self.most_held.load(SeqCst)
    }

    /// SST uploads failed so far.
    pub(crate) fn refused(&self) -> usize {
        self.refused.load(SeqCst)
    }

    /// Leaves an object where the one writer of this store puts its next
    /// WAL, as the open of a newer writer of its path does: the put that
    /// finds it is fenced. (A newer writer's open also raises the epoch in
    /// the manifest, which the stale writer's own read of the manifest can
    /// find first; this leaves the WAL put as the only way it learns.)
    pub(crate) fn fence_the_next_wal(&self) {
        use futures::TryStreamExt;
        use slatedb::object_store::ObjectStoreExt;
        let last = futures::executor::block_on(self.inner.list(None).try_collect::<Vec<_>>())
            .unwrap()
            .into_iter()
            .map(|object| object.location.to_string())
            .filter(|location| location.contains("/wal/"))
            .max()
            .expect("a writer's open leaves a WAL");
        let (wals, name) = last.rsplit_once('/').unwrap();
        let id = name.strip_suffix(".sst").unwrap();
        let next = format!(
            "{:0width$}",
            id.parse::<u64>().unwrap() + 1,
            width = id.len()
        );
        let next = ObjPath::from(format!("{wals}/{next}.sst"));
        futures::executor::block_on(self.inner.put(&next, object_store::PutPayload::default()))
            .unwrap();
    }

    /// SSTs under `path`'s `compacted/`: what its flushed memtables left.
    pub(crate) fn ssts(&self, path: &str) -> usize {
        use futures::TryStreamExt;
        let prefix = ObjPath::from(format!("{path}/compacted"));
        futures::executor::block_on(self.inner.list(Some(&prefix)).try_collect::<Vec<_>>())
            .unwrap()
            .len()
    }

    fn matches(slot: &Mutex<Option<String>>, path: &str) -> bool {
        slot.lock()
            .unwrap()
            .as_deref()
            .is_some_and(|part| path.contains(part))
    }
}

impl std::fmt::Display for HeldSsts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HeldSsts({})", self.inner)
    }
}

impl ObjectStore for HeldSsts {
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
            let path = location.as_ref();
            if path.contains(self.within) {
                if Self::matches(&self.refuse, path) {
                    self.refused.fetch_add(1, SeqCst);
                    return Err(object_store::Error::NotSupported {
                        source: "refused for the test".into(),
                    });
                }
                if Self::matches(&self.hold, path) {
                    let held = self.held.fetch_add(1, SeqCst) + 1;
                    self.most_held.fetch_max(held, SeqCst);
                    let released = self.released.acquire().await;
                    self.held.fetch_sub(1, SeqCst);
                    released.expect("never closed").forget();
                }
            }
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
        Box::pin(async move {
            let inner = self.inner.put_multipart_opts(location, opts).await?;
            let path = location.as_ref();
            if !path.contains("/compacted/") {
                return Ok(inner);
            }
            self.part_uploads.begun.fetch_add(1, SeqCst);
            Ok(Box::new(CountedUpload {
                inner,
                refuse_first_part: Self::matches(&self.refuse_part, path),
                parts: 0,
                uploads: Arc::clone(&self.part_uploads),
            }) as Box<dyn object_store::MultipartUpload>)
        })
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

/// An SST's upload in parts that counts how it ends, and whose first part
/// fails if the store was told to refuse it when the upload began.
#[derive(Debug)]
struct CountedUpload {
    inner: Box<dyn object_store::MultipartUpload>,
    refuse_first_part: bool,
    parts: usize,
    uploads: Arc<PartUploads>,
}

impl object_store::MultipartUpload for CountedUpload {
    fn put_part(&mut self, data: object_store::PutPayload) -> object_store::UploadPart {
        self.parts += 1;
        if self.refuse_first_part && self.parts == 1 {
            return Box::pin(async {
                Err(object_store::Error::Generic {
                    store: "HeldSsts",
                    source: "part refused for the test".into(),
                })
            });
        }
        self.inner.put_part(data)
    }

    fn complete<'a, 'b>(&'a mut self) -> Boxed<'b, object_store::PutResult>
    where
        'a: 'b,
        Self: 'b,
    {
        Box::pin(async move {
            let result = self.inner.complete().await;
            if result.is_ok() {
                self.uploads.completed.fetch_add(1, SeqCst);
            }
            result
        })
    }

    fn abort<'a, 'b>(&'a mut self) -> Boxed<'b, ()>
    where
        'a: 'b,
        Self: 'b,
    {
        Box::pin(async move {
            self.uploads.aborted.fetch_add(1, SeqCst);
            self.inner.abort().await
        })
    }
}

/// The free memory a test's gate reads: bytes, or `None` for a pod whose use
/// cannot be read.
#[derive(Clone)]
pub(crate) struct Memory(Arc<Mutex<Option<usize>>>);

impl Memory {
    pub(crate) fn set(&self, free: Option<usize>) {
        *self.0.lock().unwrap() = free;
    }
}

/// A gate that reads `free` and looks again every 10 ms, and the handle a
/// test changes that reading by.
pub(crate) fn gate_reading(free: Option<usize>) -> (Arc<FlushGate>, Memory) {
    gate_counting_for(free, super::OVERDUE)
}

/// [`gate_reading`], with the bytes of a flush that holds its allowance
/// counted for `counted_for` and no longer.
pub(crate) fn gate_counting_for(
    free: Option<usize>,
    counted_for: Duration,
) -> (Arc<FlushGate>, Memory) {
    let memory = Memory(Arc::new(Mutex::new(free)));
    let read = memory.clone();
    let gate = FlushGate::timed(
        Box::new(move || *read.0.lock().unwrap()),
        Duration::from_millis(10),
        counted_for,
    );
    (Arc::new(gate), memory)
}

/// Waits until `reached`, for 30 s at most: `what` it never did, else.
pub(crate) fn wait_until(what: &str, mut reached: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !reached() {
        assert!(Instant::now() < deadline, "never: {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// What `thread` returns, once it has: within 30 s, as [`wait_until`].
pub(crate) fn joined<T>(what: &str, thread: std::thread::JoinHandle<T>) -> T {
    wait_until(what, || thread.is_finished());
    thread.join().unwrap()
}
