//! One GET for the ranged reads of a WAL object, for a reader's `DbReader`.
//!
//! A reader replays its WAL tail at every open, at every reopen and, for the
//! objects written since, at every poll. SlateDB reads a WAL object that holds
//! data with five requests one after the other: a HEAD for its length, then
//! the footer, the info, the index and the blocks as four ranges. None comes
//! from its disk cache, which leaves WAL reads alone because WAL ids start at 1
//! again when a dropped collection is created anew. So each is a round trip to
//! the object store: about 105 ms an object read alone, about 8 ms an object
//! through the 32-wide replay window.
//!
//! [`WalWholeGet`] sits between `DbReader` and the store. The HEAD goes
//! through as it is, and its size and tags are noted. The first ranged read of
//! that object is then sent as one GET of the whole object, and that read and
//! the ones after it are cut from the bytes: five requests become two. A
//! zero-byte object (the fence a writer open leaves, one HEAD and nothing
//! more) and an object over [`MAX_OBJECT_BYTES`] are read as before.
//!
//! The bytes are held for one replay of the object and never for a second:
//!
//! - they are dropped once the range that starts at offset 0 has been served,
//!   which is the blocks, the last thing SlateDB reads of an object;
//! - a new HEAD of the path starts over, also for a fetch still on its way:
//!   nothing a re-created collection wrote under an old id is answered from
//!   bytes fetched before that HEAD;
//! - a range asked a second time is SlateDB's retry after a failed validation:
//!   the bytes are dropped and the read goes to the store for fresh ones;
//! - after [`MAX_AGE`], or when [`MAX_HELD`] newer objects are held.
//!
//! So a `DbReader` holds at most [`MAX_HELD`] objects of [`MAX_OBJECT_BYTES`]
//! (16 MiB), and in practice what its replay window has between footer and
//! blocks. The age is looked at whenever a WAL request passes, and every poll
//! of a `DbReader` sends HEADs for new WAL objects, so bytes whose blocks were
//! never asked for are gone a poll interval after [`MAX_AGE`] at the latest.
//!
//! Anything else goes to the store as it was sent: a fetch that fails, a fetch
//! whose size or tags are not the HEAD's, a range the bytes do not cover, a
//! conditional read. So this adds no failure a reader does not have without it.

use std::future::Future;
use std::ops::Range;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use futures::stream::{self, BoxStream};
use futures::StreamExt;
use object_store::path::Path;
use object_store::{
    Attributes, CopyOptions, GetOptions, GetRange, GetResult, GetResultPayload, ListResult,
    MultipartUpload, ObjectMeta, ObjectStore, PutMultipartOptions, PutOptions, PutPayload,
    PutResult, RenameOptions,
};
use slatedb::bytes::Bytes;

use super::backend_lsm::reader_wal_whole_get_enabled;

/// The largest object fetched whole. On 2026-11-17 the WAL tails in the bucket
/// averaged 4.6 KB an object (the 42 long ones) to 104 KB (data objects of the
/// short ones); the objects of an index run are larger and are left alone.
const MAX_OBJECT_BYTES: u64 = 256 * 1024;
/// Objects noted or held at once: twice the replay window's default width.
/// A window set wider than this reads what does not fit as before.
const MAX_HELD: usize = 64;
/// How long after its HEAD what an object's slot holds is kept. SlateDB's
/// reads of one object follow its HEAD at once; this is for those that never
/// come.
const MAX_AGE: Duration = Duration::from_secs(30);

/// `store` as a reader's `DbReader` should see it: behind [`WalWholeGet`],
/// unless `HELIX_LSM_READER_WAL_WHOLE_GET` is `0`.
pub(crate) fn for_reader(store: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
    if reader_wal_whole_get_enabled() {
        Arc::new(WalWholeGet::new(store))
    } else {
        store
    }
}

/// See the module documentation.
pub(crate) struct WalWholeGet {
    inner: Arc<dyn ObjectStore>,
    /// Least recently touched first. At most [`MAX_HELD`], so a scan is
    /// cheaper than a map.
    slots: Mutex<Vec<Slot>>,
    notes: AtomicU64,
}

/// One answered HEAD of a path, and what has come of it since.
struct Slot {
    path: Path,
    /// Which HEAD this is: a fetch is kept only for the one it followed.
    note: u64,
    at: Instant,
    held: Held,
}

enum Held {
    /// Nothing yet: what the object must look like when it is fetched.
    Sized(Expected),
    /// A ranged read is fetching the object; any other goes to the store.
    Fetching,
    /// The object, and the ranges answered from it so far.
    Whole {
        meta: ObjectMeta,
        attributes: Attributes,
        bytes: Bytes,
        served: Vec<Range<u64>>,
    },
}

/// What a HEAD said of an object.
struct Expected {
    size: u64,
    e_tag: Option<String>,
    version: Option<String>,
}

/// What a ranged read finds for its path.
enum Lookup {
    Served(GetResult),
    /// Only the HEAD was seen: fetch the object for slot `note`.
    Fetch {
        note: u64,
        expected: Expected,
    },
    /// Nothing that answers it: the read goes to the store.
    Miss,
}

impl WalWholeGet {
    pub(crate) fn new(inner: Arc<dyn ObjectStore>) -> Self {
        Self {
            inner,
            slots: Mutex::new(Vec::new()),
            notes: AtomicU64::new(0),
        }
    }

    fn slots(&self) -> MutexGuard<'_, Vec<Slot>> {
        let mut slots = self
            .slots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        slots.retain(|slot| slot.at.elapsed() < MAX_AGE);
        slots
    }

    /// A HEAD of `path` was answered with `meta`, or failed. Either way it
    /// starts over: what an earlier HEAD of the path led to is dropped.
    fn note_head(&self, path: &Path, meta: Option<&ObjectMeta>) {
        let mut slots = self.slots();
        slots.retain(|slot| slot.path != *path);
        let Some(meta) = meta.filter(|meta| (1..=MAX_OBJECT_BYTES).contains(&meta.size)) else {
            return;
        };
        if slots.len() >= MAX_HELD {
            slots.remove(0);
        }
        slots.push(Slot {
            path: path.clone(),
            note: self.notes.fetch_add(1, Ordering::Relaxed),
            at: Instant::now(),
            held: Held::Sized(Expected {
                size: meta.size,
                e_tag: meta.e_tag.clone(),
                version: meta.version.clone(),
            }),
        });
    }

    fn lookup(&self, path: &Path, range: &Range<u64>) -> Lookup {
        let mut slots = self.slots();
        let Some(index) = slots.iter().position(|slot| slot.path == *path) else {
            return Lookup::Miss;
        };
        // Taken out, and put back last with what it holds next, if anything.
        let slot = slots.remove(index);
        let (held, lookup) = match slot.held {
            Held::Sized(expected) => {
                let note = slot.note;
                (Some(Held::Fetching), Lookup::Fetch { note, expected })
            }
            Held::Fetching => (Some(Held::Fetching), Lookup::Miss),
            Held::Whole {
                meta,
                attributes,
                bytes,
                mut served,
            } => {
                if range.end > meta.size || served.contains(range) {
                    (None, Lookup::Miss)
                } else {
                    let result = cut(&meta, &attributes, &bytes, range.clone());
                    served.push(range.clone());
                    let whole = Held::Whole {
                        meta,
                        attributes,
                        bytes,
                        served,
                    };
                    ((range.start != 0).then_some(whole), Lookup::Served(result))
                }
            }
        };
        if let Some(held) = held {
            slots.push(Slot {
                path: slot.path,
                note: slot.note,
                at: slot.at,
                held,
            });
        }
        lookup
    }

    /// The fetch for slot `note` is over. Its bytes are kept only if that slot
    /// is still there: a HEAD answered in the meantime has replaced it, and
    /// the reads after that HEAD must not be answered from before it.
    fn settle(&self, note: u64, whole: Option<Held>) {
        let mut slots = self.slots();
        let Some(index) = slots.iter().position(|slot| slot.note == note) else {
            return;
        };
        match whole {
            Some(whole) => slots[index].held = whole,
            None => drop(slots.remove(index)),
        }
    }

    /// The whole object, when it is the one the HEAD described.
    async fn fetch(
        &self,
        path: &Path,
        options: &GetOptions,
        expected: &Expected,
    ) -> Option<(ObjectMeta, Attributes, Bytes)> {
        let whole = GetOptions {
            extensions: options.extensions.clone(),
            ..GetOptions::default()
        };
        let result = self.inner.get_opts(path, whole).await.ok()?;
        let (meta, attributes) = (result.meta.clone(), result.attributes.clone());
        if meta.size != expected.size
            || meta.e_tag != expected.e_tag
            || meta.version != expected.version
        {
            return None;
        }
        let bytes = result.bytes().await.ok()?;
        (bytes.len() as u64 == meta.size).then_some((meta, attributes, bytes))
    }

    async fn wal_get(&self, path: &Path, options: GetOptions) -> object_store::Result<GetResult> {
        if options.head {
            let result = self.inner.get_opts(path, options).await;
            self.note_head(path, result.as_ref().ok().map(|result| &result.meta));
            return result;
        }
        let Some(range) = plain_range(&options) else {
            return self.inner.get_opts(path, options).await;
        };
        let (note, expected) = match self.lookup(path, &range) {
            Lookup::Served(result) => return Ok(result),
            Lookup::Miss => return self.inner.get_opts(path, options).await,
            Lookup::Fetch { note, expected } => (note, expected),
        };
        match self.fetch(path, &options, &expected).await {
            Some((meta, attributes, bytes)) if range.end <= meta.size => {
                metrics::counter!("helix_lsm_reader_wal_whole_get_total", "outcome" => "fetched")
                    .increment(1);
                let result = cut(&meta, &attributes, &bytes, range.clone());
                let whole = (range.start != 0).then(|| Held::Whole {
                    meta,
                    attributes,
                    bytes,
                    served: vec![range],
                });
                self.settle(note, whole);
                Ok(result)
            }
            _ => {
                metrics::counter!("helix_lsm_reader_wal_whole_get_total", "outcome" => "fell_back")
                    .increment(1);
                self.settle(note, None);
                self.inner.get_opts(path, options).await
            }
        }
    }
}

/// `<root>/wal/<digits>.sst`: the only objects SlateDB reads this way.
fn is_wal_object(path: &Path) -> bool {
    let path: &str = path.as_ref();
    let Some((dir, file)) = path.rsplit_once('/') else {
        return false;
    };
    (dir == "wal" || dir.ends_with("/wal"))
        && file
            .strip_suffix(".sst")
            .is_some_and(|id| !id.is_empty() && id.bytes().all(|byte| byte.is_ascii_digit()))
}

/// The range of a read that asks for nothing but a non-empty byte range.
fn plain_range(options: &GetOptions) -> Option<Range<u64>> {
    let conditional = options.if_match.is_some()
        || options.if_none_match.is_some()
        || options.if_modified_since.is_some()
        || options.if_unmodified_since.is_some()
        || options.version.is_some();
    match &options.range {
        Some(GetRange::Bounded(range)) if !conditional && range.start < range.end => {
            Some(range.clone())
        }
        _ => None,
    }
}

/// The answer a ranged GET of `range` gets from the store, from `bytes`.
fn cut(meta: &ObjectMeta, attributes: &Attributes, bytes: &Bytes, range: Range<u64>) -> GetResult {
    let part = bytes.slice(range.start as usize..range.end as usize);
    GetResult {
        payload: GetResultPayload::Stream(
            stream::once(async move { Ok::<_, object_store::Error>(part) }).boxed(),
        ),
        meta: meta.clone(),
        range,
        attributes: attributes.clone(),
        extensions: Default::default(),
    }
}

impl std::fmt::Debug for WalWholeGet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WalWholeGet")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Display for WalWholeGet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WalWholeGet({})", self.inner)
    }
}

type Boxed<'a, T> = Pin<Box<dyn Future<Output = object_store::Result<T>> + Send + 'a>>;

/// Every call but a read of a WAL object is the store's own, the provided
/// methods included: its listing from an offset and its ranges are not the
/// trait's defaults.
impl ObjectStore for WalWholeGet {
    fn put_opts<'a, 'b, 'c>(
        &'a self,
        location: &'b Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Boxed<'c, PutResult>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        self.inner.put_opts(location, payload, opts)
    }

    fn put_multipart_opts<'a, 'b, 'c>(
        &'a self,
        location: &'b Path,
        opts: PutMultipartOptions,
    ) -> Boxed<'c, Box<dyn MultipartUpload>>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        self.inner.put_multipart_opts(location, opts)
    }

    fn get_opts<'a, 'b, 'c>(
        &'a self,
        location: &'b Path,
        options: GetOptions,
    ) -> Boxed<'c, GetResult>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        if is_wal_object(location) {
            Box::pin(self.wal_get(location, options))
        } else {
            self.inner.get_opts(location, options)
        }
    }

    fn get_ranges<'a, 'b, 'c, 'd>(
        &'a self,
        location: &'b Path,
        ranges: &'c [Range<u64>],
    ) -> Boxed<'d, Vec<Bytes>>
    where
        'a: 'd,
        'b: 'd,
        'c: 'd,
        Self: 'd,
    {
        self.inner.get_ranges(location, ranges)
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }

    fn list_with_delimiter<'a, 'b, 'c>(&'a self, prefix: Option<&'b Path>) -> Boxed<'c, ListResult>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        self.inner.list_with_delimiter(prefix)
    }

    fn copy_opts<'a, 'b, 'c, 'd>(
        &'a self,
        from: &'b Path,
        to: &'c Path,
        options: CopyOptions,
    ) -> Boxed<'d, ()>
    where
        'a: 'd,
        'b: 'd,
        'c: 'd,
        Self: 'd,
    {
        self.inner.copy_opts(from, to, options)
    }

    fn rename_opts<'a, 'b, 'c, 'd>(
        &'a self,
        from: &'b Path,
        to: &'c Path,
        options: RenameOptions,
    ) -> Boxed<'d, ()>
    where
        'a: 'd,
        'b: 'd,
        'c: 'd,
        Self: 'd,
    {
        self.inner.rename_opts(from, to, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helix_engine::storage_core::backend::{Namespace, StorageBackend};
    use crate::helix_engine::storage_core::backend_lsm::{shared_lsm_handle, LsmBackend};
    use crate::helix_engine::storage_core::backend_lsm_reader::LsmReader;
    use futures::TryStreamExt;
    use object_store::memory::InMemory;
    use object_store::ObjectStoreExt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Semaphore;

    /// Counts the requests for WAL objects that pass through it, by kind.
    /// Given a `reference`, it also compares every ranged answer it hands up
    /// with that store's answer to the same read.
    #[derive(Debug)]
    struct Probe {
        inner: Arc<dyn ObjectStore>,
        reference: Option<Arc<dyn ObjectStore>>,
        /// A whole GET's answer is handed up only once this has a permit.
        gate: Option<Arc<Semaphore>>,
        heads: AtomicUsize,
        wholes: AtomicUsize,
        ranged: AtomicUsize,
        differing: Mutex<Vec<String>>,
    }

    impl Probe {
        fn new(inner: Arc<dyn ObjectStore>, reference: Option<Arc<dyn ObjectStore>>) -> Arc<Self> {
            Self::gated(inner, reference, None)
        }

        fn gated(
            inner: Arc<dyn ObjectStore>,
            reference: Option<Arc<dyn ObjectStore>>,
            gate: Option<Arc<Semaphore>>,
        ) -> Arc<Self> {
            Arc::new(Self {
                inner,
                reference,
                gate,
                heads: AtomicUsize::new(0),
                wholes: AtomicUsize::new(0),
                ranged: AtomicUsize::new(0),
                differing: Mutex::new(Vec::new()),
            })
        }

        /// (HEADs, whole GETs, ranged GETs) of WAL objects so far.
        fn counts(&self) -> (usize, usize, usize) {
            (
                self.heads.load(Ordering::SeqCst),
                self.wholes.load(Ordering::SeqCst),
                self.ranged.load(Ordering::SeqCst),
            )
        }

        async fn wal_get(
            &self,
            path: &Path,
            options: GetOptions,
        ) -> object_store::Result<GetResult> {
            let (head, asked) = (options.head, options.range.clone());
            let counter = match (head, &asked) {
                (true, _) => &self.heads,
                (false, None) => &self.wholes,
                (false, Some(_)) => &self.ranged,
            };
            counter.fetch_add(1, Ordering::SeqCst);
            let result = self.inner.get_opts(path, options).await?;
            if let (Some(gate), false, None) = (&self.gate, head, &asked) {
                // The answer has left the store; the test says when it arrives.
                let _permit = gate.acquire().await;
            }
            let (Some(reference), Some(asked), false) = (&self.reference, asked, head) else {
                return Ok(result);
            };
            let same_read = GetOptions {
                range: Some(asked.clone()),
                ..GetOptions::default()
            };
            let expected = reference.get_opts(path, same_read).await?;
            let (meta, range, attributes) = (
                result.meta.clone(),
                result.range.clone(),
                result.attributes.clone(),
            );
            let same = meta == expected.meta
                && range == expected.range
                && attributes == expected.attributes;
            let (part, expected_part) = (result.bytes().await?, expected.bytes().await?);
            if !same || part != expected_part {
                let mut differing = self.differing.lock().unwrap();
                differing.push(format!("{path} {asked:?}"));
            }
            Ok(GetResult {
                payload: GetResultPayload::Stream(
                    stream::once(async move { Ok::<_, object_store::Error>(part) }).boxed(),
                ),
                meta,
                range,
                attributes,
                extensions: Default::default(),
            })
        }
    }

    impl std::fmt::Display for Probe {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Probe({})", self.inner)
        }
    }

    impl ObjectStore for Probe {
        fn put_opts<'a, 'b, 'c>(
            &'a self,
            location: &'b Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> Boxed<'c, PutResult>
        where
            'a: 'c,
            'b: 'c,
            Self: 'c,
        {
            self.inner.put_opts(location, payload, opts)
        }

        fn put_multipart_opts<'a, 'b, 'c>(
            &'a self,
            location: &'b Path,
            opts: PutMultipartOptions,
        ) -> Boxed<'c, Box<dyn MultipartUpload>>
        where
            'a: 'c,
            'b: 'c,
            Self: 'c,
        {
            self.inner.put_multipart_opts(location, opts)
        }

        fn get_opts<'a, 'b, 'c>(
            &'a self,
            location: &'b Path,
            options: GetOptions,
        ) -> Boxed<'c, GetResult>
        where
            'a: 'c,
            'b: 'c,
            Self: 'c,
        {
            if is_wal_object(location) {
                Box::pin(self.wal_get(location, options))
            } else {
                self.inner.get_opts(location, options)
            }
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, object_store::Result<Path>>,
        ) -> BoxStream<'static, object_store::Result<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
            self.inner.list(prefix)
        }

        fn list_with_delimiter<'a, 'b, 'c>(
            &'a self,
            prefix: Option<&'b Path>,
        ) -> Boxed<'c, ListResult>
        where
            'a: 'c,
            'b: 'c,
            Self: 'c,
        {
            self.inner.list_with_delimiter(prefix)
        }

        fn copy_opts<'a, 'b, 'c, 'd>(
            &'a self,
            from: &'b Path,
            to: &'c Path,
            options: CopyOptions,
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

    // SlateDB's ranged reads of a 1,000-byte WAL object, in its order.
    const FOOTER: Range<u64> = 990..1000;
    const INFO: Range<u64> = 900..990;
    const INDEX: Range<u64> = 800..900;
    const BLOCKS: Range<u64> = 0..800;

    fn wal(id: u64) -> Path {
        Path::from(format!("db/wal/{id:020}.sst"))
    }

    /// 1,000 bytes that differ from one offset to the next, and with `salt`.
    fn body(salt: u8) -> Vec<u8> {
        (0..1000u32).map(|i| (i % 251) as u8 ^ salt).collect()
    }

    /// A store with [`body`] as WAL object 2, the probe over it, and the
    /// wrapper over the probe.
    async fn fixture() -> (Arc<InMemory>, Arc<Probe>, WalWholeGet) {
        let raw = Arc::new(InMemory::new());
        raw.put(&wal(2), body(0).into()).await.expect("put");
        let below = Probe::new(raw.clone(), None);
        let wrapper = WalWholeGet::new(below.clone());
        (raw, below, wrapper)
    }

    async fn head(store: &dyn ObjectStore, path: &Path) -> ObjectMeta {
        let options = GetOptions {
            head: true,
            ..GetOptions::default()
        };
        store.get_opts(path, options).await.expect("head").meta
    }

    /// A ranged read as SlateDB sends it, and all that its answer carries.
    async fn ranged(
        store: &dyn ObjectStore,
        path: &Path,
        range: Range<u64>,
    ) -> (ObjectMeta, Range<u64>, Attributes, Bytes) {
        let options = GetOptions {
            range: Some(GetRange::Bounded(range)),
            ..GetOptions::default()
        };
        let result = store.get_opts(path, options).await.expect("ranged read");
        let (meta, range, attributes) = (
            result.meta.clone(),
            result.range.clone(),
            result.attributes.clone(),
        );
        (
            meta,
            range,
            attributes,
            result.bytes().await.expect("bytes"),
        )
    }

    fn held_bytes(wrapper: &WalWholeGet) -> usize {
        let slots = wrapper.slots();
        slots
            .iter()
            .map(|slot| match &slot.held {
                Held::Whole { bytes, .. } => bytes.len(),
                Held::Sized(_) | Held::Fetching => 0,
            })
            .sum()
    }

    #[tokio::test]
    async fn the_ranged_reads_of_a_wal_object_reach_the_store_as_one_get() {
        let (raw, below, wrapper) = fixture().await;

        assert_eq!(head(&wrapper, &wal(2)).await, head(&*raw, &wal(2)).await);
        for range in [FOOTER, INFO, INDEX] {
            assert_eq!(
                ranged(&wrapper, &wal(2), range.clone()).await,
                ranged(&*raw, &wal(2), range).await,
            );
        }
        assert_eq!(held_bytes(&wrapper), 1000);
        assert_eq!(
            ranged(&wrapper, &wal(2), BLOCKS).await,
            ranged(&*raw, &wal(2), BLOCKS).await,
        );
        assert_eq!(
            below.counts(),
            (1, 1, 0),
            "(HEADs, whole GETs, ranged GETs)"
        );
        assert_eq!(held_bytes(&wrapper), 0, "the blocks are the last read");

        // Nothing is kept for a second replay: the same read again is the store's.
        ranged(&wrapper, &wal(2), FOOTER).await;
        assert_eq!(below.counts(), (1, 1, 1));
    }

    #[tokio::test]
    async fn a_fence_and_an_object_over_the_bound_are_read_as_before() {
        let raw = Arc::new(InMemory::new());
        let at_bound = vec![7u8; MAX_OBJECT_BYTES as usize];
        let over_bound = vec![7u8; MAX_OBJECT_BYTES as usize + 1];
        raw.put(&wal(1), Vec::<u8>::new().into())
            .await
            .expect("put");
        raw.put(&wal(2), over_bound.into()).await.expect("put");
        raw.put(&wal(3), at_bound.into()).await.expect("put");
        let below = Probe::new(raw.clone(), None);
        let wrapper = WalWholeGet::new(below.clone());

        assert_eq!(head(&wrapper, &wal(1)).await.size, 0);
        head(&wrapper, &wal(2)).await;
        assert!(wrapper.slots().is_empty(), "neither is noted for a fetch");
        assert_eq!(
            ranged(&wrapper, &wal(2), 10..20).await,
            ranged(&*raw, &wal(2), 10..20).await,
        );
        assert_eq!(below.counts(), (2, 0, 1));

        head(&wrapper, &wal(3)).await;
        assert_eq!(
            ranged(&wrapper, &wal(3), 10..20).await,
            ranged(&*raw, &wal(3), 10..20).await,
        );
        assert_eq!(
            below.counts(),
            (3, 1, 1),
            "an object at the bound is fetched"
        );
    }

    #[tokio::test]
    async fn a_range_asked_twice_is_read_from_the_store() {
        let (raw, below, wrapper) = fixture().await;

        // SlateDB asks again for a range whose first answer failed its checksum.
        head(&wrapper, &wal(2)).await;
        for range in [FOOTER, INFO, INDEX, INDEX, BLOCKS] {
            assert_eq!(
                ranged(&wrapper, &wal(2), range.clone()).await,
                ranged(&*raw, &wal(2), range).await,
            );
        }
        assert_eq!(below.counts(), (1, 1, 2), "the retry and the read after it");
        assert_eq!(held_bytes(&wrapper), 0);

        // The same for the read the object was fetched for.
        head(&wrapper, &wal(2)).await;
        ranged(&wrapper, &wal(2), FOOTER).await;
        ranged(&wrapper, &wal(2), FOOTER).await;
        assert_eq!(below.counts(), (2, 2, 3));
        assert_eq!(held_bytes(&wrapper), 0);
    }

    #[tokio::test]
    async fn an_object_replaced_after_its_head_is_read_from_the_store() {
        let (raw, below, wrapper) = fixture().await;

        head(&wrapper, &wal(2)).await;
        // A collection dropped and created anew writes its WAL ids again.
        raw.put(&wal(2), body(7).into()).await.expect("put");
        for range in [FOOTER, INFO] {
            assert_eq!(
                ranged(&wrapper, &wal(2), range.clone()).await,
                ranged(&*raw, &wal(2), range).await,
            );
        }
        assert_eq!(
            below.counts(),
            (1, 1, 2),
            "the fetch that was not the HEAD's object, then both reads"
        );
        assert_eq!(held_bytes(&wrapper), 0);
    }

    #[tokio::test]
    async fn a_new_head_starts_over() {
        let (raw, below, wrapper) = fixture().await;

        head(&wrapper, &wal(2)).await;
        ranged(&wrapper, &wal(2), FOOTER).await;
        raw.put(&wal(2), body(7).into()).await.expect("put");
        head(&wrapper, &wal(2)).await;
        assert_eq!(
            ranged(&wrapper, &wal(2), FOOTER).await,
            ranged(&*raw, &wal(2), FOOTER).await,
            "the footer of the new object, not of the first fetch"
        );
        assert_eq!(below.counts(), (2, 2, 0));
    }

    #[tokio::test]
    async fn a_fetch_is_kept_only_for_the_head_it_followed() {
        let (raw, path) = (Arc::new(InMemory::new()), wal(2));
        raw.put(&path, body(0).into()).await.expect("put");
        let gate = Arc::new(Semaphore::new(0));
        let below = Probe::gated(raw.clone(), None, Some(gate.clone()));
        let wrapper = WalWholeGet::new(below.clone());

        head(&wrapper, &path).await;
        let first_footer = ranged(&*raw, &path, FOOTER).await;
        // The first object's bytes have left the store when it is replaced ...
        let first = ranged(&wrapper, &path, FOOTER);
        tokio::pin!(first);
        assert!(futures::poll!(first.as_mut()).is_pending());
        raw.put(&path, body(7).into()).await.expect("put");
        // ... and arrive after the HEAD of the second.
        head(&wrapper, &path).await;
        gate.add_permits(1);
        assert_eq!(first.await, first_footer, "the read that began before");
        assert_eq!(
            ranged(&wrapper, &path, FOOTER).await,
            ranged(&*raw, &path, FOOTER).await,
            "the second object's footer for the read after its HEAD"
        );
        assert_eq!(below.counts(), (2, 2, 0));
    }

    #[tokio::test]
    async fn only_plain_ranges_of_wal_objects_are_answered_from_a_fetch() {
        assert!(is_wal_object(&Path::from(
            "prod/c/wal/00000000000000000042.sst"
        )));
        assert!(is_wal_object(&Path::from("wal/7.sst")));
        for other in [
            "prod/c/compacted/01JD0000000000000000000000.sst",
            "prod/c/manifest/00000000000000000042.manifest",
            "prod/c/journal/00000000000000000042.sst",
            "prod/c/wal/current.sst",
            "prod/c/wal/.sst",
            "prod/wal",
        ] {
            assert!(!is_wal_object(&Path::from(other)), "{other}");
        }

        let (_raw, below, wrapper) = fixture().await;
        head(&wrapper, &wal(2)).await;
        let whole = wrapper.get_opts(&wal(2), GetOptions::default()).await;
        assert_eq!(whole.expect("get").bytes().await.expect("bytes"), body(0));
        let suffix = GetOptions {
            range: Some(GetRange::Suffix(10)),
            ..GetOptions::default()
        };
        let suffix = wrapper.get_opts(&wal(2), suffix).await.expect("suffix");
        assert_eq!(suffix.bytes().await.expect("bytes")[..], body(0)[990..]);
        let conditional = GetOptions {
            range: Some(GetRange::Bounded(FOOTER)),
            if_match: Some("not its tag".to_string()),
            ..GetOptions::default()
        };
        assert!(wrapper.get_opts(&wal(2), conditional).await.is_err());
        assert_eq!(below.counts(), (1, 1, 2), "each went to the store as sent");
        assert_eq!(held_bytes(&wrapper), 0);
    }

    #[tokio::test]
    async fn no_more_than_the_bound_is_noted_at_once() {
        let raw = Arc::new(InMemory::new());
        let ids = 0..=MAX_HELD as u64;
        for id in ids.clone() {
            raw.put(&wal(id), body(0).into()).await.expect("put");
        }
        let below = Probe::new(raw.clone(), None);
        let wrapper = WalWholeGet::new(below.clone());

        for id in ids {
            head(&wrapper, &wal(id)).await;
        }
        assert_eq!(wrapper.slots().len(), MAX_HELD);
        ranged(&wrapper, &wal(0), FOOTER).await;
        ranged(&wrapper, &wal(MAX_HELD as u64), FOOTER).await;
        assert_eq!(
            below.counts(),
            (MAX_HELD + 1, 1, 1),
            "the first note made room for the last"
        );
    }

    struct VarGuard(&'static str);
    impl VarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            std::env::set_var(key, value);
            Self(key)
        }
    }
    impl Drop for VarGuard {
        fn drop(&mut self) {
            std::env::remove_var(self.0);
        }
    }

    const SWITCH: &str = "HELIX_LSM_READER_WAL_WHOLE_GET";
    const KEYS: [&[u8]; 3] = [b"k1", b"k2", b"k3"];

    fn value(i: usize) -> Vec<u8> {
        vec![i as u8; 2000]
    }

    /// The WAL objects in `raw` that a reader fetches whole.
    fn data_objects(raw: &Arc<dyn ObjectStore>) -> usize {
        let listed: Vec<ObjectMeta> = shared_lsm_handle()
            .block_on(raw.list(None).try_collect())
            .expect("list");
        listed
            .iter()
            .filter(|meta| is_wal_object(&meta.location))
            .filter(|meta| (1..=MAX_OBJECT_BYTES).contains(&meta.size))
            .count()
    }

    /// What a reader opened over `store` reads of [`KEYS`]. No row has left
    /// the WAL, so all of it comes from the replay at its open.
    fn read_rows(path: &str, store: Arc<dyn ObjectStore>) -> Vec<Option<Vec<u8>>> {
        let reader = LsmReader::open_with_store(path, store).expect("open reader");
        KEYS.iter()
            .map(|key| {
                reader
                    .get_with(Namespace::Nodes, key, |v| v.map(|b| b.to_vec()))
                    .expect("get")
            })
            .collect()
    }

    /// The real replay, over the real writer's WAL objects. The counts are
    /// exact on purpose: a SlateDB that reads a WAL object in another order or
    /// in other pieces is a reason to look at when the bytes are dropped.
    #[test]
    #[serial_test::serial]
    fn a_reader_replays_its_wal_tail_with_one_get_an_object() {
        let _poll = VarGuard::set("HELIX_LSM_READER_MANIFEST_POLL_MS", "60000");
        let raw: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = "wal-whole-get-test";

        // The writer stays open and flushes nothing, so every reader below
        // replays the same tail.
        let writer = LsmBackend::open_with_store(path, raw.clone()).expect("open writer");
        for (i, key) in KEYS.iter().enumerate() {
            let mut w = writer.begin_write().unwrap();
            writer
                .put(&mut w, Namespace::Nodes, key, &value(i))
                .unwrap();
            writer.commit(w).unwrap();
        }
        let data = data_objects(&raw);
        assert!(data > 0, "the writer left no WAL object with data");
        let written: Vec<_> = (0..KEYS.len()).map(|i| Some(value(i))).collect();

        // Switched off: four ranged reads an object, as before.
        {
            let _off = VarGuard::set(SWITCH, "0");
            let probe = Probe::new(raw.clone(), None);
            assert_eq!(read_rows(path, probe.clone()), written);
            let (_, wholes, ranged) = probe.counts();
            assert_eq!((wholes, ranged), (0, 4 * data), "(whole, ranged), off");
        }

        // By default: one GET an object, and the same rows.
        {
            let probe = Probe::new(raw.clone(), None);
            assert_eq!(read_rows(path, probe.clone()), written);
            let (_, wholes, ranged) = probe.counts();
            assert_eq!((wholes, ranged), (data, 0), "(whole, ranged), by default");
        }

        // Every answer cut from a fetch is the store's answer to that read,
        // and nothing stays held once the tail is replayed.
        {
            let _off = VarGuard::set(SWITCH, "0");
            let below = Probe::new(raw.clone(), None);
            let wrapper = Arc::new(WalWholeGet::new(below.clone()));
            let above = Probe::new(wrapper.clone(), Some(raw.clone()));
            assert_eq!(read_rows(path, above.clone()), written);
            assert_eq!(above.counts().2, 4 * data, "ranged reads SlateDB sent");
            let (_, wholes, ranged) = below.counts();
            assert_eq!((wholes, ranged), (data, 0), "(whole, ranged) at the store");
            assert_eq!(*above.differing.lock().unwrap(), Vec::<String>::new());
            assert_eq!(held_bytes(&wrapper), 0);
        }

        assert_eq!(data_objects(&raw), data, "the tail moved under the test");
    }
}
