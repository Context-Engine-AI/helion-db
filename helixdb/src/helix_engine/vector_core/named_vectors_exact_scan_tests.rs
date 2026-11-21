//! The two exact scorers of a set of ids, the read per id
//! (`dense_search_candidate_ids_exact_be`) and the sidecar scan
//! (`dense_search_broad_prefilter_scan_be`), on a space as production has it:
//! raw f32 rows and an exact sidecar file a segment, searched on the writer
//! and on a reader replica through a read pinned to its snapshot. Each id is
//! scored where its newest copy is, and the two return the same rows in the
//! same order.

use super::*;
use crate::helix_engine::storage_core::backend::SegmentDb;
use crate::helix_engine::storage_core::backend_lsm::{
    key_batches_marked_for_test, with_lsm_multi_get_for_test, LsmBackend,
};
use crate::helix_engine::storage_core::backend_lsm_reader::LsmReader;
use crate::helix_engine::vector_core::spindle::SpindleMode;
use object_store::{memory::InMemory, ObjectStore};
use rand::{rngs::StdRng, Rng, SeedableRng};
use tempfile::TempDir;

type Row = (u128, [f32; 3]);
/// `(id, bits of the distance, bits of the vector)` of every hit, in the
/// order returned.
type Hits = Vec<(u128, u32, Vec<u32>)>;

const QUERY: [f32; 3] = [1.0, 0.0, 0.0];

/// A segment core of the space, without a quantizer. `dir: None` opens it
/// with no sidecar, so its inserts write rows only.
fn raw_core(name: &str, dir: Option<&Path>, backend: &Arc<AnyBackend>) -> VectorCore {
    VectorCore::new_named_lsm_with_dir(
        name,
        HNSWConfig::new(Some(8), Some(32), Some(64)),
        DistanceMetric::Cosine,
        SpindleConfig {
            mode: SpindleMode::None,
            keep_original: false,
            rescore: false,
            ..SpindleConfig::default()
        },
        dir,
        3,
        Arc::clone(backend),
    )
    .unwrap()
}

fn insert(core: &VectorCore, backend: &AnyBackend, rows: &[Row]) {
    let mut write = backend.begin_write().unwrap();
    for (id, vector) in rows {
        core.insert_flat_be(&mut write, vector, Some(*id), None)
            .unwrap();
    }
    backend.commit(write).unwrap();
}

/// The space `dense` as one server has it: the writer, or a reader replica
/// of its store with sidecar files of its own.
struct Server {
    backend: Arc<AnyBackend>,
    manager: NamedVectorManager,
    dir: TempDir,
}

impl Server {
    fn new(backend: Arc<AnyBackend>, collection: &str) -> Self {
        let manager = NamedVectorManager::new(collection.to_string());
        manager.attach_backend(Arc::clone(&backend));
        manager.dense_spaces.write().unwrap().insert(
            "dense".to_string(),
            DenseVectorSpaceMetadata {
                next_segment_id: 0,
                segments: Vec::new(),
            },
        );
        Self {
            backend,
            manager,
            dir: TempDir::new().unwrap(),
        }
    }

    /// Opens segment `name` here and lists it last in the space's layout.
    fn open(&self, name: &str, role: DenseVectorSegmentRole) {
        let core = raw_core(name, Some(self.dir.path()), &self.backend);
        self.manager
            .dense_spaces
            .write()
            .unwrap()
            .get_mut("dense")
            .unwrap()
            .segments
            .push(DenseVectorSegmentMetadata {
                physical_name: name.to_string(),
                role,
            });
        self.manager
            .cores
            .write()
            .unwrap()
            .insert(name.to_string(), core);
    }

    /// A read of what is committed now; on a replica, pinned to its snapshot
    /// as production's reads are.
    fn read(&self) -> AnyRead<'_> {
        match &*self.backend {
            AnyBackend::LsmReader(reader) => {
                self.backend
                    .refresh_lsm_reader_with_poll_interval(std::time::Duration::from_millis(100))
                    .unwrap();
                AnyRead::LsmReader(Some(reader.begin_snapshot().unwrap()))
            }
            writer => writer.begin_read().unwrap(),
        }
    }

    /// The nearest `k` of `set` by the read per id.
    fn by_id(&self, query: [f32; 3], k: usize, set: &HashSet<u128>) -> Hits {
        let read = self.read();
        hits(
            self.manager
                .dense_search_candidate_ids_exact_be(
                    &read,
                    "dense",
                    &query,
                    k,
                    set.iter().copied(),
                    None,
                )
                .unwrap(),
        )
    }

    /// The nearest `k` of `set` by the sidecar scan; `None` where it
    /// declines, which it does for a segment with rows and no sidecar file.
    fn scan(&self, query: [f32; 3], k: usize, set: &HashSet<u128>) -> Option<Hits> {
        let read = self.read();
        self.manager
            .dense_search_broad_prefilter_scan_be::<fn(u128) -> bool>(
                &read, "dense", &query, k, set, None, None,
            )
            .unwrap()
            .map(hits)
    }

    fn by_scan(&self, query: [f32; 3], k: usize, set: &HashSet<u128>) -> Hits {
        self.scan(query, k, set)
            .expect("every segment has a sidecar file, so the scan serves")
    }

    /// The scan as the gateway asks it for a filter no index answers
    /// exactly: handed a recheck, it answers only an id of `passes`.
    fn scan_rechecked(
        &self,
        query: [f32; 3],
        k: usize,
        set: &HashSet<u128>,
        passes: &HashSet<u128>,
    ) -> Option<Hits> {
        let read = self.read();
        let recheck = [|id: u128| passes.contains(&id)];
        self.manager
            .dense_search_broad_prefilter_scan_be(
                &read,
                "dense",
                &query,
                k,
                set,
                Some(&recheck[..]),
                None,
            )
            .unwrap()
            .map(hits)
    }
}

/// The space on an LSM writer over an in-memory object store. The module's
/// other rigs make a TurboProd space, which has no slots and no ordinal
/// rows, so neither scorer reads a sidecar there.
struct Space {
    store: Arc<dyn ObjectStore>,
    path: String,
    writer: Server,
}

impl Space {
    fn new(test: &str) -> Self {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = format!("{test}-{}", uuid::Uuid::new_v4());
        let backend = Arc::new(AnyBackend::Lsm(
            LsmBackend::open_with_store(&path, Arc::clone(&store)).unwrap(),
        ));
        Self {
            store,
            path,
            writer: Server::new(backend, test),
        }
    }

    /// Adds the space's next segment, newer than every one before it, with
    /// `rows`, and returns its name.
    fn segment(&self, role: DenseVectorSegmentRole, rows: &[Row]) -> String {
        let name = {
            let mut spaces = self.writer.manager.dense_spaces.write().unwrap();
            let space = spaces.get_mut("dense").unwrap();
            space.next_segment_id += 1;
            format!("dense__seg_{:06}", space.next_segment_id - 1)
        };
        self.writer.open(&name, role);
        self.write(&name, rows);
        assert!(
            rows.is_empty() || self.writer.manager.cores.read().unwrap()[&name].has_mmap_sidecar(),
            "{name} must score from a sidecar file, as a production segment does"
        );
        name
    }

    /// Writes `rows` into `segment`, each with its slot and its ordinal row.
    fn write(&self, segment: &str, rows: &[Row]) {
        let cores = self.writer.manager.cores.read().unwrap();
        insert(&cores[segment], &self.writer.backend, rows);
    }

    /// Removes the rows of `ids` from `segment`, as a delete of those points
    /// or a changed re-upsert does.
    fn delete(&self, segment: &str, ids: &[u128]) {
        let cores = self.writer.manager.cores.read().unwrap();
        let mut write = self.writer.backend.begin_write().unwrap();
        cores[segment]
            .delete_vectors_batch_be(&mut write, ids)
            .unwrap();
        self.writer.backend.commit(write).unwrap();
    }

    /// Lists the space's segments in the order of `places`, which names
    /// each segment's present place.
    fn relist(&self, places: &[usize]) {
        let mut spaces = self.writer.manager.dense_spaces.write().unwrap();
        let segments = &mut spaces.get_mut("dense").unwrap().segments;
        *segments = places.iter().map(|&at| segments[at].clone()).collect();
    }

    /// A reader replica of the store as it is now: the writer's layout, and
    /// each segment's sidecar file laid out from the rows it holds now. Rows
    /// written later are past the end of those files.
    fn replica(&self) -> Server {
        self.replica_with(true)
    }

    /// A reader replica that has laid out its sidecar files (`files`), or
    /// one that has opened its segments and not read from them yet.
    fn replica_with(&self, files: bool) -> Server {
        let backend = Arc::new(AnyBackend::LsmReader(
            LsmReader::open_with_store(&self.path, Arc::clone(&self.store)).unwrap(),
        ));
        let replica = Server::new(backend, "replica");
        let layout = self.writer.manager.dense_spaces.read().unwrap()["dense"]
            .segments
            .clone();
        for segment in layout {
            replica.open(&segment.physical_name, segment.role);
            assert!(
                !files
                    || replica.manager.cores.read().unwrap()[&segment.physical_name]
                        .ensure_lsm_mmap_sidecar()
                        .unwrap(),
                "the replica must lay out the sidecar file of {}",
                segment.physical_name
            );
        }
        replica
    }

    /// A write to one namespace of `segment` alone, which no insert or
    /// delete of the engine makes: `value: None` deletes `key`.
    fn raw(&self, segment: &str, db: SegmentDb, key: &[u8], value: Option<&[u8]>) {
        let backend = &self.writer.backend;
        let namespace = Namespace::Segment {
            physical_name: segment,
            db,
        };
        let mut write = backend.begin_write().unwrap();
        match value {
            Some(value) => backend.put(&mut write, namespace, key, value),
            None => backend.delete(&mut write, namespace, key),
        }
        .unwrap();
        backend.commit(write).unwrap();
    }

    /// Takes the vector row of `id` out of `segment` and leaves its ordinal
    /// row and its slot.
    fn drop_row_alone(&self, segment: &str, id: u128) {
        self.row(segment, id, None);
    }

    /// Writes the vector row of `id` in `segment` as `value`, or takes it
    /// out, and leaves its ordinal row and its slot as they are.
    fn row(&self, segment: &str, id: u128, value: Option<&[u8]>) {
        let key = [b"v:".as_slice(), &id.to_be_bytes(), &0usize.to_be_bytes()].concat();
        self.raw(segment, SegmentDb::Vectors, &key, value);
    }

    /// The name of the segment the layout lists last.
    fn last(&self) -> String {
        let spaces = self.writer.manager.dense_spaces.read().unwrap();
        spaces["dense"]
            .segments
            .last()
            .unwrap()
            .physical_name
            .clone()
    }
}

fn hits(found: Vec<HVector>) -> Hits {
    found
        .iter()
        .map(|hit| {
            (
                hit.get_id(),
                hit.get_distance().to_bits(),
                hit.get_data().iter().map(|value| value.to_bits()).collect(),
            )
        })
        .collect()
}

fn ids(found: &Hits) -> Vec<u128> {
    found.iter().map(|hit| hit.0).collect()
}

/// `f`'s value and the metrics it counted on this thread, rendered.
fn counted<R>(f: impl FnOnce() -> R) -> (R, String) {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let out = metrics::with_local_recorder(&recorder, f);
    (out, handle.render())
}

/// Id 5 was rewritten twice: its first row, in the oldest segment, is the
/// nearest row there is, its second is the next nearest, and its newest, in
/// the tail, is the farthest; 6 lies between the second and the newest. The
/// nearest one of {5, 6} is 6 by the read per id (c7654bd75), and the scan
/// has to say so too: on the writer and on a replica, and whatever order the
/// layout lists the segments in. Listed second, tail, oldest, a walk in the
/// layout's order answers from the second segment and a walk in its reverse
/// from the oldest.
#[test]
fn the_scan_scores_an_id_where_its_newest_copy_is() {
    for places in [[0, 1, 2], [1, 2, 0]] {
        let space = Space::new("scan_newest_copy");
        space.segment(DenseVectorSegmentRole::Indexed, &[(5, [1.0, 0.0, 0.0])]);
        space.segment(DenseVectorSegmentRole::Indexed, &[(5, [0.8, 0.6, 0.0])]);
        space.segment(
            DenseVectorSegmentRole::Mutable,
            &[(5, [0.0, 1.0, 0.0]), (6, [0.6, 0.8, 0.0])],
        );
        space.relist(&places);
        let replica = space.replica();
        let set = HashSet::from([5, 6]);
        for (server, on) in [(&space.writer, "the writer"), (&replica, "a replica")] {
            assert_eq!(ids(&server.by_id(QUERY, 1, &set)), [6], "{on}, {places:?}");
            assert_eq!(
                ids(&server.by_scan(QUERY, 1, &set)),
                [6],
                "on {on}, listed {places:?}, the scan returned a row of id 5 that a newer row replaced"
            );
            assert_eq!(
                server.by_scan(QUERY, 2, &set),
                server.by_id(QUERY, 2, &set),
                "{on}, {places:?}"
            );
            // What the contest reads: the tail's row of 5, once, when the
            // second segment would answer with its own. 6 lies once, and a
            // set of it alone has no such read.
            let (_, twice) = counted(|| server.by_scan(QUERY, 1, &set));
            assert!(
                twice.contains(r#"helix_broad_scan_row_lookups_total{outcome="newer_copy"} 1"#),
                "{on}, {places:?}: {twice}"
            );
            let (_, once) = counted(|| server.by_scan(QUERY, 1, &HashSet::from([6])));
            assert!(!once.contains("newer_copy"), "{on}, {places:?}: {once}");
        }
    }
}

/// What the scan can see without a read of its own: the tail lists id 5
/// with a slot past the end of its file, and the row that slot stood for is
/// gone. The read the scan makes for every slot past the file finds no row,
/// so the listing stands for nothing, it is counted, and the older segment
/// answers for 5 as it does by the read per id.
#[test]
fn a_listing_past_the_file_with_no_row_leaves_the_id_to_the_older_copy() {
    let space = Space::new("scan_listing_past_file");
    space.segment(DenseVectorSegmentRole::Indexed, &[(5, [1.0, 0.0, 0.0])]);
    let tail = space.segment(DenseVectorSegmentRole::Mutable, &[(6, [0.9, 0.1, 0.0])]);
    let replica = space.replica();
    space.raw(
        &tail,
        SegmentDb::Ordinals,
        &5u128.to_be_bytes(),
        Some(&1_000u64.to_le_bytes()),
    );
    let set = HashSet::from([5, 6]);
    for (server, on) in [(&space.writer, "the writer"), (&replica, "a replica")] {
        let by_id = server.by_id(QUERY, 2, &set);
        assert_eq!(ids(&by_id), [5, 6], "{on}");
        assert_eq!(server.by_scan(QUERY, 2, &set), by_id, "{on}");
    }
    // A replica walks the ordinal rows at its snapshot, so it meets the
    // listing whatever a writer's cached ordinals hold.
    key_batches_marked_for_test();
    let (_, metrics) = counted(|| replica.by_scan(QUERY, 2, &set));
    assert!(
        metrics.contains("helix_broad_scan_listing_without_row_total 1"),
        "{metrics}"
    );
    // The mark of a batched row read is the exact scorer's alone.
    let batches = key_batches_marked_for_test();
    assert!(
        !batches.is_empty() && !batches.contains(&true),
        "the scan's row batches carry no mark: {batches:?}"
    );
}

/// An ordinal row and its slot whose vector row is gone, which no insert or
/// delete of the engine leaves in a space without a quantized store (both
/// rows go into one batch and leave in one). The scan scores a slot inside
/// its file without reading the row, so it cannot tell such a listing from
/// any other until it reads the row, and it reads the row of a newer listing
/// only when an older copy of that id is near enough to be answered.
///
/// 5 has such a listing in the tail and a row in the older segment: the read
/// finds the tail's listing void, counts it, and 5 is answered with its
/// older row, as by the read per id. 9 has such a listing and no other copy:
/// nothing asks for its row, and the scan answers it from the slot, which
/// the read per id does not. `GET /collections/{name}/unreached` counts
/// these listings (`ordinals_without_row`).
#[test]
fn a_listing_whose_row_is_gone_does_not_hide_the_older_copy() {
    let space = Space::new("scan_listing_without_row");
    space.segment(
        DenseVectorSegmentRole::Indexed,
        &[(5, [1.0, 0.0, 0.0]), (8, [0.0, 0.0, 1.0])],
    );
    let tail = space.segment(
        DenseVectorSegmentRole::Mutable,
        &[
            (5, [0.0, 1.0, 0.0]),
            (6, [0.9, 0.1, 0.0]),
            (9, [0.5, 0.5, 0.0]),
        ],
    );
    let replica = space.replica();
    space.drop_row_alone(&tail, 5);
    space.drop_row_alone(&tail, 9);
    let set = HashSet::from([5, 6, 8, 9]);
    for (server, on) in [(&space.writer, "the writer"), (&replica, "a replica")] {
        let by_id = server.by_id(QUERY, 4, &set);
        assert_eq!(ids(&by_id), [5, 6, 8], "{on}");
        let (by_scan, metrics) = counted(|| server.by_scan(QUERY, 4, &set));
        assert_eq!(ids(&by_scan), [5, 6, 9, 8], "{on}");
        assert_eq!(by_scan[0], by_id[0], "{on}: 5 with its older row");
        assert!(
            metrics.contains("helix_broad_scan_listing_without_row_total 1"),
            "{on}: {metrics}"
        );
        // Four rows asked for and not there: 8 of the tail, 6 and 9 of the
        // older segment, and the tail's row of 5 when the contest asks.
        assert!(
            metrics.contains(r#"helix_broad_scan_row_lookups_total{outcome="absent"} 4"#),
            "{on}: {metrics}"
        );
        assert!(!metrics.contains("newer_copy"), "{on}: {metrics}");
    }

    // What that read comes too late for: here the void listing is the tail's
    // nearest row, so the tail has cut its real row 6 for it before the
    // older segment asks. The hit is taken out and the row it displaced is
    // not put back: nearest one, the scan answers 5 with its older row
    // where the read per id answers 6. Nearest two, both answer both.
    let space = Space::new("scan_listing_without_row_cut");
    space.segment(DenseVectorSegmentRole::Indexed, &[(5, [0.6, 0.8, 0.0])]);
    let tail = space.segment(
        DenseVectorSegmentRole::Mutable,
        &[(5, [1.0, 0.0, 0.0]), (6, [0.8, 0.6, 0.0])],
    );
    let replica = space.replica();
    space.drop_row_alone(&tail, 5);
    let set = HashSet::from([5, 6]);
    for (server, on) in [(&space.writer, "the writer"), (&replica, "a replica")] {
        assert_eq!(ids(&server.by_id(QUERY, 1, &set)), [6], "{on}");
        assert_eq!(ids(&server.by_scan(QUERY, 1, &set)), [5], "{on}");
        assert_eq!(
            server.by_scan(QUERY, 2, &set),
            server.by_id(QUERY, 2, &set),
            "{on}"
        );
    }
}

/// 5 lies in three segments: the tail lists it with no row, the middle
/// segment holds a row of it that is far, and the oldest's row is the nearest
/// there is. Nearest one, the middle segment answers 7 and never asks for
/// the tail's row of 5, so the oldest's question is the first: it finds the
/// tail's listing void and has to go on to the middle one's, whose row is
/// there. 5 is the middle segment's to answer, as by the read per id.
#[test]
fn a_void_listing_does_not_end_the_search_for_a_newer_copy() {
    let space = Space::new("scan_void_then_held");
    space.segment(DenseVectorSegmentRole::Indexed, &[(5, [1.0, 0.0, 0.0])]);
    space.segment(
        DenseVectorSegmentRole::Indexed,
        &[(5, [0.1, 0.9, 0.0]), (7, [0.9, 0.1, 0.0])],
    );
    let tail = space.segment(
        DenseVectorSegmentRole::Mutable,
        &[(5, [0.0, 1.0, 0.0]), (6, [0.6, 0.8, 0.0])],
    );
    let replica = space.replica();
    space.drop_row_alone(&tail, 5);
    let set = HashSet::from([5, 6, 7]);
    for (server, on) in [(&space.writer, "the writer"), (&replica, "a replica")] {
        for (k, nearest) in [(1, &[7][..]), (3, &[7, 6, 5][..])] {
            let by_id = server.by_id(QUERY, k, &set);
            assert_eq!(ids(&by_id), nearest, "{on}, nearest {k}");
            assert_eq!(server.by_scan(QUERY, k, &set), by_id, "{on}, nearest {k}");
        }
    }
}

/// Three segments of 200 ids: 20 ids in the two older ones, 10 in the two
/// newer ones and 10 in the oldest and the tail, each copy with a vector of
/// its own; three rows removed, a newest copy among them; a row with a zero
/// vector. Then a replica is opened, and after it the tail gets what a
/// replica reads by row: rows past the end of its file (211 to 215, and 20
/// rewritten), a row with no ordinal row (216) and rows with an ordinal that
/// names no slot (217, and 30 rewritten). `vector` draws a row's vector.
fn three_segments(test: &str, mut vector: impl FnMut() -> [f32; 3]) -> (Space, Server) {
    let space = Space::new(test);
    let mut rows = |ids: &mut dyn Iterator<Item = u128>| -> Vec<Row> {
        ids.map(|id| (id, vector())).collect()
    };
    let oldest = space.segment(DenseVectorSegmentRole::Indexed, &rows(&mut (1..=120u128)));
    space.segment(DenseVectorSegmentRole::Indexed, &rows(&mut (101..=180u128)));
    let tail = space.segment(
        DenseVectorSegmentRole::Mutable,
        &rows(&mut (171..=200u128).chain(1..=10)),
    );
    // 5 and 175 lose their newest copy and are served from an older one; 110
    // loses its older copy.
    space.delete(&tail, &[5, 175]);
    space.delete(&oldest, &[110]);
    space.write(&tail, &[(201, [0.0, 0.0, 0.0])]);

    let replica = space.replica();
    space.write(&tail, &rows(&mut (211..=215u128).chain([20])));
    let writer = &space.writer.backend;
    // Written with no sidecar attached: no slot and no ordinal row.
    insert(
        &raw_core(&tail, None, writer),
        writer,
        &rows(&mut (216..=216u128)),
    );
    // That row makes a rebuild of the file refuse, so a core that has lost
    // its file stores rows with an ordinal that names no slot.
    let lost = TempDir::new().unwrap();
    let row_only = raw_core(&tail, Some(lost.path()), writer);
    insert(&row_only, writer, &rows(&mut [217u128, 30].into_iter()));
    assert!(!row_only.has_mmap_sidecar());
    (space, replica)
}

/// What differs, if anything, between the two scorers over seeded queries,
/// sets and limits. A set also names ids the space does not hold, and a
/// limit goes past the size of any set. With `passing`, the scan is handed a
/// recheck that lets that share of each set through, and the read per id is
/// given the ids that pass.
fn differences(server: &Server, seed: u64, passing: Option<f32>) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut wrong = Vec::new();
    for case in 0..120 {
        let query: [f32; 3] = std::array::from_fn(|_| rng.random_range(-1.0f32..1.0));
        let keep = [1.0, 0.5, 0.1, 0.02][case % 4];
        let set: HashSet<u128> = (1..=230u128)
            .filter(|_| rng.random_range(0.0f32..1.0) < keep)
            .collect();
        let k = [1, 7, 40, 500][(case / 4) % 4];
        let passes: Option<HashSet<u128>> = passing.map(|share| {
            (1..=230u128)
                .filter(|id| set.contains(id) && rng.random_range(0.0f32..1.0) < share)
                .collect()
        });
        // The scan first: a read per id may lay a sidecar file out again.
        let scanned = match &passes {
            Some(passes) => server.scan_rechecked(query, k, &set, passes),
            None => server.scan(query, k, &set),
        };
        let Some(by_scan) = scanned else {
            wrong.push(format!("case {case}: the scan declined"));
            continue;
        };
        let by_id = server.by_id(query, k, passes.as_ref().unwrap_or(&set));
        if by_id != by_scan {
            let first = by_id
                .iter()
                .zip(&by_scan)
                .position(|(a, b)| a != b)
                .unwrap_or(by_id.len().min(by_scan.len()));
            wrong.push(format!(
                "case {case} ({} ids, nearest {k}): {} rows by id, {} by scan, first difference at {first}: {:?} against {:?}",
                set.len(),
                by_id.len(),
                by_scan.len(),
                by_id.get(first),
                by_scan.get(first),
            ));
        }
    }
    wrong
}

/// No two rows are equally far here but the copies of the zero vector's
/// distance, so the order of equal distances plays no part: what the two
/// scorers can differ in is which copy of an id they score and where they
/// read it from.
#[test]
fn both_scorers_return_the_same_rows_where_no_two_are_equally_far() {
    let mut rng = StdRng::seed_from_u64(2026_1120);
    let (space, replica) = three_segments("scorers_equal_distinct", || {
        std::array::from_fn(|_| rng.random_range(-1.0f32..1.0))
    });
    for (server, on) in [(&space.writer, "the writer"), (&replica, "a replica")] {
        let wrong = differences(server, 7, None);
        assert!(
            wrong.is_empty(),
            "on {on}, {} of 120 cases: {wrong:#?}",
            wrong.len()
        );
    }
}

/// Twelve vectors for 250 rows: most rows have others equally far from any
/// query, in their own segment and in the others.
#[test]
fn both_scorers_return_the_same_rows_where_many_are_equally_far() {
    let mut rng = StdRng::seed_from_u64(2026_1121);
    let pool: Vec<[f32; 3]> = (0..12)
        .map(|_| std::array::from_fn(|_| rng.random_range(-1.0f32..1.0)))
        .collect();
    let (space, replica) = three_segments("scorers_equal_ties", || {
        pool[rng.random_range(0..pool.len())]
    });
    for (server, on) in [(&space.writer, "the writer"), (&replica, "a replica")] {
        let wrong = differences(server, 11, None);
        assert!(
            wrong.is_empty(),
            "on {on}, {} of 120 cases: {wrong:#?}",
            wrong.len()
        );
    }
}

/// The scan with a recheck, as the gateway asks it for a filter no index
/// answers exactly: it answers what a read of each id answers for the ids
/// that pass. An id a newer segment holds is passed over in an older one
/// before the recheck is asked, and that costs the older segment no row: it
/// still returns the nearest `k` of its own members that pass.
///
/// 5 and 9 lie twice, their older rows the two nearest rows there are. 5
/// fails the recheck and 9 passes; 7 and 8 are the older segment's own and
/// pass. Nearest one, the older segment has to go past both to 7, and 9 is
/// answered where its newer row is: last.
#[test]
fn a_scan_that_rechecks_answers_as_a_read_of_each_id_that_passes() {
    let space = Space::new("scan_recheck_pairs");
    space.segment(
        DenseVectorSegmentRole::Indexed,
        &[
            (5, [1.0, 0.0, 0.0]),
            (9, [0.95, 0.05, 0.0]),
            (7, [0.9, 0.1, 0.0]),
            (8, [0.8, 0.2, 0.0]),
        ],
    );
    space.segment(
        DenseVectorSegmentRole::Mutable,
        &[
            (5, [0.0, 1.0, 0.0]),
            (9, [0.1, 0.9, 0.0]),
            (6, [0.6, 0.8, 0.0]),
        ],
    );
    let replica = space.replica();
    let set = HashSet::from([5, 6, 7, 8, 9]);
    let passes = HashSet::from([6, 7, 8, 9]);
    for (server, on) in [(&space.writer, "the writer"), (&replica, "a replica")] {
        for k in 1..=5 {
            let by_scan = server.scan_rechecked(QUERY, k, &set, &passes).unwrap();
            assert_eq!(ids(&by_scan), [7, 8, 6, 9][..k.min(4)], "{on}, nearest {k}");
            assert_eq!(
                by_scan,
                server.by_id(QUERY, k, &passes),
                "{on}, nearest {k}"
            );
        }
    }

    let mut rng = StdRng::seed_from_u64(2026_1123);
    let (space, replica) = three_segments("scan_recheck_seeded", || {
        std::array::from_fn(|_| rng.random_range(-1.0f32..1.0))
    });
    for (server, on) in [(&space.writer, "the writer"), (&replica, "a replica")] {
        for passing in [0.5, 0.2] {
            let wrong = differences(server, 13, Some(passing));
            assert!(
                wrong.is_empty(),
                "on {on}, {passing} of a set passing, {} of 120 cases: {wrong:#?}",
                wrong.len()
            );
        }
    }
}

/// `f`'s value and which way the exact scorer went for the one set it
/// scored: the `way` label of `helix_dense_exact_scorer_total`.
fn way<R>(f: impl FnOnce() -> R) -> (R, String) {
    let (out, metrics) = counted(f);
    let ways: Vec<&str> = metrics
        .lines()
        .filter(|line| line.starts_with("helix_dense_exact_scorer_total{"))
        .collect();
    assert_eq!(ways.len(), 1, "one set, one way: {metrics}");
    let way = ways[0].split('"').nth(1).unwrap_or_default().to_string();
    assert!(ways[0].ends_with(" 1"), "{}", ways[0]);
    (out, way)
}

/// 1,200 rows in a sealed segment and 10 in the tail, on the writer and on a
/// replica whose files hold all 1,210.
fn large_space(test: &str) -> (Space, Server) {
    let space = Space::new(test);
    let mut rng = StdRng::seed_from_u64(2026_1122);
    let mut rows = |ids: std::ops::RangeInclusive<u128>| -> Vec<Row> {
        ids.map(|id| (id, std::array::from_fn(|_| rng.random_range(-1.0f32..1.0))))
            .collect()
    };
    space.segment(DenseVectorSegmentRole::Indexed, &rows(1..=1_200));
    space.segment(DenseVectorSegmentRole::Mutable, &rows(1_201..=1_210));
    let replica = space.replica();
    (space, replica)
}

/// The rule at its bounds, through the scorer on a replica: a set of 512 ids
/// is read id by id whatever the setting, a larger one is scanned where the
/// space holds no more sidecar rows an id than the setting allows, and the
/// rows are the same either way. The setting is taken at 0 (never), at
/// values on both sides of the space's 1,210 rows, and unset.
#[test]
fn a_set_is_scanned_by_its_size_and_the_rows_of_its_space() {
    let (space, replica) = large_space("exact_scan_rule");
    let asked = |server: &Server, rows_per_id: Option<usize>, ids: u128| -> (Hits, String) {
        let set: HashSet<u128> = (1..=ids).collect();
        let scored = || way(|| server.by_id(QUERY, 40, &set));
        match rows_per_id {
            Some(rows) => super::exact_search::with_exact_scan_rows_per_id_for_test(rows, scored),
            None => scored(),
        }
    };

    let (read_each, why) = asked(&replica, Some(0), 600);
    assert_eq!(why, "per_id_off");
    assert_eq!(read_each.len(), 40);
    // 600 ids in 1,210 rows: a little over two rows an id.
    for (rows_per_id, expected) in [
        (Some(1), "per_id_rows"),
        (Some(2), "per_id_rows"),
        (Some(3), "scan"),
        (Some(100_000), "scan"),
        (None, "scan"),
    ] {
        let (found, why) = asked(&replica, rows_per_id, 600);
        assert_eq!(why, expected, "{rows_per_id:?} rows an id");
        assert_eq!(found, read_each, "{rows_per_id:?} rows an id");
    }
    // The floor, and the first size over it: 513 ids are 2.36 rows an id.
    for rows_per_id in [None, Some(100_000)] {
        assert_eq!(asked(&replica, rows_per_id, 512).1, "per_id_small");
        assert_eq!(asked(&replica, rows_per_id, 513).1, "scan");
    }
    assert_eq!(asked(&replica, Some(2), 513).1, "per_id_rows");
    assert_eq!(asked(&replica, Some(3), 513).1, "scan");
    // A writer's read is never scanned, and it returns the same rows.
    let (found, why) = asked(&space.writer, None, 600);
    assert_eq!(why, "per_id_handle");
    assert_eq!(found, read_each);
    // Nor is a replica's read that is pinned to no snapshot.
    let set: HashSet<u128> = (1..=600).collect();
    let (found, why) = way(|| {
        replica
            .manager
            .dense_search_candidate_ids_exact_be(
                &AnyRead::LsmReader(None),
                "dense",
                &QUERY,
                40,
                set.iter().copied(),
                None,
            )
            .map(hits)
            .unwrap()
    });
    assert_eq!(why, "per_id_handle");
    assert_eq!(found, read_each);
}

/// A replica that has not laid out a segment's sidecar file reads the set
/// id by id: the scan would decline, and the set is not handed to it.
#[test]
fn a_set_is_read_id_by_id_where_a_segment_has_no_sidecar_file() {
    let (space, _) = large_space("exact_scan_no_file");
    let fresh = space.replica_with(false);
    let set: HashSet<u128> = (1..=600).collect();
    let (found, why) = way(|| fresh.by_id(QUERY, 40, &set));
    assert_eq!(why, "per_id_segment");
    assert_eq!(ids(&found), ids(&space.writer.by_id(QUERY, 40, &set)));
}

/// What a segment says of itself to the rule: the slots of its exact file,
/// 0 with no file and no rows, and nothing where a scan does not return
/// what a read of each id returns.
#[test]
fn a_segment_says_whether_a_scan_can_stand_in_for_a_read_of_each_id() {
    let space = Space::new("exact_scan_slots");
    let sealed = space.segment(
        DenseVectorSegmentRole::Indexed,
        &[
            (1, [1.0, 0.0, 0.0]),
            (2, [0.0, 1.0, 0.0]),
            (3, [0.0, 0.0, 1.0]),
        ],
    );
    let empty = space.segment(DenseVectorSegmentRole::Mutable, &[]);
    let merged = space.segment(DenseVectorSegmentRole::Indexed, &[(4, [0.5, 0.5, 0.0])]);
    let backend = &space.writer.backend;
    let slots = |core: &VectorCore| core.exact_scan_slots(&space.writer.read()).unwrap();
    let cores = space.writer.manager.cores.read().unwrap();

    assert_eq!(slots(&cores[&sealed]), Some(3));
    assert_eq!(slots(&cores[&empty]), Some(0), "no file and no rows");
    assert_eq!(
        slots(&raw_core(&sealed, None, backend)),
        None,
        "rows and no file open"
    );
    let quantized = VectorCore::new_named_lsm_with_dir(
        "dense__seg_000009",
        HNSWConfig::new(Some(8), Some(32), Some(64)),
        DistanceMetric::Cosine,
        SpindleConfig::default(),
        Some(space.writer.dir.path()),
        3,
        Arc::clone(backend),
    )
    .unwrap();
    assert_eq!(slots(&quantized), None, "a quantized store");
    // A merged segment's file written scalar-quantized (HELIX_SEGMENT_FORMAT
    // =hvs8) in a space that has no quantizer: its slots are not the rows.
    assert_eq!(slots(&cores[&merged]), Some(1));
    assert!(cores[&merged].quantize_mmap().unwrap());
    assert_eq!(slots(&cores[&merged]), None, "a scalar-quantized file");

    let mut write = backend.begin_write().unwrap();
    let dead = cores[&sealed]
        .tombstone_delete_batch_be(&mut write, &[2])
        .unwrap();
    backend.commit(write).unwrap();
    cores[&sealed].apply_delete_tombstones(&dead);
    assert_eq!(slots(&cores[&sealed]), None, "a row deleted by tombstone");
}

/// A scan whose reads are cancelled with its request ends the search: no
/// read of each id follows it for a caller that has left.
#[test]
fn a_cancelled_scan_is_followed_by_no_read_of_each_id() {
    use crate::helix_engine::storage_core::backend_lsm::{
        allow_lsm_blocking_cancellable, LsmReadCancellation,
    };

    let (_space, replica) = large_space("exact_scan_cancelled");
    let set: HashSet<u128> = (1..=600).collect();
    let read = replica.read();
    let gone = LsmReadCancellation::new();
    gone.cancel();
    let (outcome, metrics) = counted(|| {
        allow_lsm_blocking_cancellable(gone, || {
            replica.manager.dense_search_candidate_ids_exact_be(
                &read,
                "dense",
                &QUERY,
                40,
                set.iter().copied(),
                None,
            )
        })
    });
    assert!(outcome.is_err(), "rows were scored for a caller that left");
    assert!(
        metrics.contains(r#"helix_dense_exact_scorer_total{way="scan_cancelled"} 1"#),
        "{metrics}"
    );
    assert!(!metrics.contains("per_id"), "{metrics}");
}

/// The nearest `k` of `set` by a read of each id, the way every segment went
/// before its rows were read in one batch: `HELIX_DENSE_EXACT_ROWS=0`.
fn by_each_id(server: &Server, query: [f32; 3], k: usize, set: &HashSet<u128>) -> Hits {
    crate::helix_engine::vector_core::vector_core::with_exact_rows_for_test(false, || {
        server.by_id(query, k, set)
    })
}

/// The seeded queries, sets and limits of `differences`.
fn seeded_cases(seed: u64) -> Vec<([f32; 3], usize, HashSet<u128>)> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..120)
        .map(|case| {
            let query: [f32; 3] = std::array::from_fn(|_| rng.random_range(-1.0f32..1.0));
            let keep = [1.0, 0.5, 0.1, 0.02][case % 4];
            let set: HashSet<u128> = (1..=230u128)
                .filter(|_| rng.random_range(0.0f32..1.0) < keep)
                .collect();
            (query, [1, 7, 40, 500][(case / 4) % 4], set)
        })
        .collect()
}

/// The rows a batched read of a segment's rows scores are the rows a read of
/// each id scores: the same ids in the same order, each with the same
/// distance and the same vector to the bit. On the writer and on a replica
/// whose files end before the tail's last rows, the read of each id scores
/// from the sidecar files, so that is the row's score against the
/// sidecar's, to the last bit. And on a replica that has laid out no file,
/// where the batched way is asked first for every case: it must answer from
/// the rows and lay out none, which the read of each id then does.
#[test]
#[serial_test::serial]
fn a_batched_read_of_the_rows_returns_what_a_read_of_each_id_returns() {
    let mut rng = StdRng::seed_from_u64(2026_1123);
    let pool: Vec<[f32; 3]> = (0..40)
        .map(|_| std::array::from_fn(|_| rng.random_range(-1.0f32..1.0)))
        .collect();
    // Most rows have a vector of their own; a fifth share one of 40.
    let (space, replica) = three_segments("exact_rows_equal", || {
        if rng.random_range(0..5) == 0 {
            pool[rng.random_range(0..pool.len())]
        } else {
            std::array::from_fn(|_| rng.random_range(-1.0f32..1.0))
        }
    });
    let fresh = space.replica_with(false);
    let no_file_open = |server: &Server| {
        let cores = server.manager.cores.read().unwrap();
        cores.values().all(|core| !core.has_mmap_sidecar())
    };
    for (server, on) in [
        (&fresh, "a replica with no file laid out"),
        (&space.writer, "the writer"),
        (&replica, "a replica"),
    ] {
        let cases = seeded_cases(13);
        let unread = no_file_open(server);
        assert_eq!(
            unread,
            std::ptr::eq(server, &fresh),
            "on {on}: the read of each id scores from sidecar files, but for the fresh replica"
        );
        // A replica's batch as production's readers read it: single gets, 96
        // in flight (`HELIX_LSM_MULTI_GET=0`). The writer's is a multi-get,
        // and so are the reads of seven ids below, on every server.
        let multi_get = std::ptr::eq(server, &space.writer);
        let batched: Vec<Hits> = cases
            .iter()
            .map(|(query, k, set)| {
                with_lsm_multi_get_for_test(multi_get, || server.by_id(*query, *k, set))
            })
            .collect();
        assert_eq!(
            no_file_open(server),
            unread,
            "on {on}, the batched read laid out a vector file"
        );
        for (case, ((query, k, set), batched)) in cases.iter().zip(&batched).enumerate() {
            let each = by_each_id(server, *query, *k, set);
            assert_eq!(
                batched,
                &each,
                "on {on}, case {case} ({} ids, nearest {k})",
                set.len()
            );
            // The same rows when a set takes many reads: seven ids a read.
            let in_sevens =
                crate::helix_engine::vector_core::vector_core::with_flat_scan_row_batch_for_test(
                    7,
                    || server.by_id(*query, *k, set),
                );
            assert_eq!(
                &in_sevens, &each,
                "on {on}, case {case} in reads of seven ids"
            );
        }
        assert!(
            batched.iter().any(|hits| hits.len() > 1),
            "on {on}, no case returned rows"
        );
    }
    assert!(
        !no_file_open(&fresh),
        "the read of each id is what lays a replica's files out"
    );
}

/// What the batch is for: a set costs a segment one round trip to the store
/// where it cost two reads an id (the row, to learn the id is there, and its
/// ordinal). Eight ids in one sealed segment on a replica that reads as
/// production's readers do (single gets): 15 round trips fewer, and the same
/// rows.
#[test]
fn a_set_costs_a_segment_one_round_trip() {
    use crate::helix_engine::storage_core::backend_lsm::lsm_round_trips_for_test;

    let space = Space::new("exact_rows_round_trips");
    let rows: Vec<Row> = (1..=8u128)
        .map(|id| (id, [1.0, id as f32 / 10.0, 0.0]))
        .collect();
    space.segment(DenseVectorSegmentRole::Indexed, &rows);
    let replica = space.replica();
    let set: HashSet<u128> = (1..=8).collect();
    let read = replica.read();
    let asked = || {
        key_batches_marked_for_test();
        let before = lsm_round_trips_for_test();
        let found = replica
            .manager
            .dense_search_candidate_ids_exact_be(
                &read,
                "dense",
                &QUERY,
                3,
                set.iter().copied(),
                None,
            )
            .unwrap();
        (
            lsm_round_trips_for_test() - before,
            hits(found),
            key_batches_marked_for_test(),
        )
    };
    let (batched_trips, batched, marked) = with_lsm_multi_get_for_test(false, asked);
    let (each_trips, each, unmarked) =
        crate::helix_engine::vector_core::vector_core::with_exact_rows_for_test(false, asked);
    // The part cache makes a batch's fills take turns by this mark alone.
    assert!(
        !marked.is_empty() && !marked.contains(&false),
        "the batch reads its rows under a mark of its collection: {marked:?}"
    );
    assert!(
        !unmarked.contains(&true),
        "the read of each id marks nothing"
    );
    assert!(
        !slatedb::cached_object_store_stats::ranged_miss_stats()
            .in_batch_row_read(&format!("{}/compacted/x.sst", space.path).as_str().into()),
        "no mark outlives its read"
    );
    assert_eq!(batched, each);
    assert_eq!(ids(&batched), [1, 2, 3]);
    assert_eq!(
        each_trips - batched_trips,
        15,
        "{each_trips} round trips by a read of each id, {batched_trips} by the batch"
    );
}

/// Where the row's score is not the store's, or the batch is switched off,
/// the batched read declines before it scores anything and the segment's
/// ids are read one by one as before.
#[test]
fn a_segment_whose_rows_do_not_score_as_its_store_keeps_the_read_of_each_id() {
    let space = Space::new("exact_rows_declined");
    let sealed = space.segment(
        DenseVectorSegmentRole::Indexed,
        &[(1, [1.0, 0.0, 0.0]), (2, [0.0, 1.0, 0.0])],
    );
    let backend = &space.writer.backend;
    let read = space.writer.read();
    let cores = space.writer.manager.cores.read().unwrap();
    let from_rows = |core: &VectorCore| {
        core.exact_hits_from_rows(&read, &QUERY, &[1, 2, 3], 2, None)
            .unwrap()
    };

    key_batches_marked_for_test();
    let (scored, metrics) = counted(|| from_rows(&cores[&sealed]));
    assert_eq!(
        key_batches_marked_for_test(),
        [true],
        "a writer's batch reads its rows under a mark of its collection"
    );
    let (present, found, _) = scored.expect("an exact store: scored from rows");
    assert!(
        metrics.contains(r#"helix_dense_exact_rows_total{outcome="scored"} 1"#),
        "{metrics}"
    );
    assert_eq!(present, [1, 2], "3 has no row here");
    assert_eq!(ids(&hits(found)), [1, 2]);
    assert!(
        crate::helix_engine::vector_core::vector_core::with_flat_scan_row_batch_for_test(0, || {
            from_rows(&cores[&sealed])
        })
        .is_none(),
        "the batch switched off"
    );
    assert!(
        crate::helix_engine::vector_core::vector_core::with_exact_rows_for_test(false, || {
            from_rows(&cores[&sealed])
        })
        .is_none(),
        "the scorer's own switch off"
    );
    assert!(
        cores[&sealed]
            .exact_hits_from_rows(&read, &QUERY[..2], &[1, 2, 3], 2, None)
            .unwrap()
            .is_none(),
        "rows that are not vectors of the query's length"
    );

    // A quantized payload stored in a segment without a quantizer is not the
    // vector: the segment goes to the read of each id.
    let payload = crate::helix_engine::vector_core::spindle::encode_vector(
        &[1.0, 0.0, 0.0],
        &SpindleConfig::scalar_int8(),
    )
    .unwrap();
    let key = [
        b"v:".as_slice(),
        &4u128.to_be_bytes(),
        &0usize.to_be_bytes(),
    ]
    .concat();
    space.raw(&sealed, SegmentDb::Vectors, &key, Some(payload.as_slice()));
    let with_payload = space.writer.read();
    assert!(
        cores[&sealed]
            .exact_hits_from_rows(&with_payload, &QUERY, &[1, 4], 2, None)
            .unwrap()
            .is_none(),
        "a quantized payload among the rows"
    );
    space.raw(&sealed, SegmentDb::Vectors, &key, None);

    // A row deleted by tombstone is the segment's and is not scored, as by
    // the read of each id.
    let mut write = backend.begin_write().unwrap();
    let dead = cores[&sealed]
        .tombstone_delete_batch_be(&mut write, &[2])
        .unwrap();
    backend.commit(write).unwrap();
    cores[&sealed].apply_delete_tombstones(&dead);
    let (present, found, _) = from_rows(&cores[&sealed]).expect("still an exact store");
    assert_eq!(present, [1, 2], "a tombstoned row is the segment's");
    assert_eq!(ids(&hits(found)), [1], "and is not scored");
    let each = cores[&sealed]
        .search_exact_ids_with_selectivity(&read, &QUERY, [1u128, 2, 3], 2, None)
        .unwrap();
    assert_eq!(ids(&hits(each)), [1], "as by the read of each id");

    let quantized = VectorCore::new_named_lsm_with_dir(
        "dense__seg_000009",
        HNSWConfig::new(Some(8), Some(32), Some(64)),
        DistanceMetric::Cosine,
        SpindleConfig::default(),
        Some(space.writer.dir.path()),
        3,
        Arc::clone(backend),
    )
    .unwrap();
    assert!(from_rows(&quantized).is_none(), "a quantized store");

    // And a vector file that is not the exact one in a space without a
    // quantizer, as a merge target quantized at its publish has: the file's
    // score is not the row's.
    assert!(
        cores[&sealed].quantize_mmap().unwrap(),
        "the sealed segment's file is quantized"
    );
    assert!(
        from_rows(&cores[&sealed]).is_none(),
        "a file that is not the exact one"
    );
}

/// The same equality at a width the kernel's vector loops run at: 35 floats
/// are four 8-wide steps (eight 4-wide) and a 3-float tail, where the seeded
/// cases above are 3 floats and never leave the scalar tail. One segment of
/// 300 rows, 30 of them sharing six vectors: the rows scored from the batched
/// read are the rows the vector file scores, each with the same distance and
/// the same vector to the bit, on the writer and on a replica that scores
/// from its mapped file.
#[test]
#[serial_test::serial]
fn rows_past_the_kernels_vector_loops_score_as_the_vector_file_scores_them() {
    const DIM: usize = 35;
    let wide = |rng: &mut StdRng| -> Vec<f32> {
        (0..DIM).map(|_| rng.random_range(-1.0f32..1.0)).collect()
    };
    let mut rng = StdRng::seed_from_u64(2026_1121);
    let pool: Vec<Vec<f32>> = (0..12).map(|_| wide(&mut rng)).collect();
    let rows: Vec<(u128, Vec<f32>)> = (1..=300u128)
        .map(|id| match id % 10 {
            0 => (id, pool[id as usize % 12].clone()),
            _ => (id, wide(&mut rng)),
        })
        .collect();

    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = format!("exact_rows_wide-{}", uuid::Uuid::new_v4());
    let core_of = |server: &Server| {
        VectorCore::new_named_lsm_with_dir(
            "dense__seg_000000",
            HNSWConfig::new(Some(8), Some(32), Some(64)),
            DistanceMetric::Cosine,
            SpindleConfig {
                mode: SpindleMode::None,
                keep_original: false,
                rescore: false,
                ..SpindleConfig::default()
            },
            Some(server.dir.path()),
            DIM,
            Arc::clone(&server.backend),
        )
        .unwrap()
    };
    let writer = Server::new(
        Arc::new(AnyBackend::Lsm(
            LsmBackend::open_with_store(&path, Arc::clone(&store)).unwrap(),
        )),
        "wide",
    );
    let written = core_of(&writer);
    let mut write = writer.backend.begin_write().unwrap();
    for (id, vector) in &rows {
        written
            .insert_flat_be(&mut write, vector, Some(*id), None)
            .unwrap();
    }
    writer.backend.commit(write).unwrap();
    let replica = Server::new(
        Arc::new(AnyBackend::LsmReader(
            LsmReader::open_with_store(&path, Arc::clone(&store)).unwrap(),
        )),
        "wide replica",
    );
    let laid_out = core_of(&replica);
    assert!(
        laid_out.ensure_lsm_mmap_sidecar().unwrap(),
        "the replica lays out its vector file"
    );

    for (server, core, on) in [
        (&writer, &written, "the writer"),
        (&replica, &laid_out, "a replica"),
    ] {
        assert!(
            core.has_mmap_sidecar(),
            "on {on}, the read of each id scores from the vector file"
        );
        let read = server.read();
        let mut equally_far = false;
        for case in 0..48 {
            let query = wide(&mut rng);
            let keep = [1.0, 0.5, 0.1, 0.02][case % 4];
            // 301 to 320 have no row.
            let set: Vec<u128> = (1..=320u128)
                .filter(|_| rng.random_range(0.0f32..1.0) < keep)
                .collect();
            let k = [1, 7, 40, 500][(case / 4) % 4];
            let (present, from_rows, _) = core
                .exact_hits_from_rows(&read, &query, &set, k, None)
                .unwrap()
                .expect("a space without a quantizer: scored from its rows");
            let held: Vec<u128> = set.iter().copied().filter(|id| *id <= 300).collect();
            assert_eq!(present, held, "on {on}, case {case}");
            let from_file = hits(
                core.search_exact_ids_with_selectivity(&read, &query, held, k, None)
                    .unwrap(),
            );
            assert_eq!(
                hits(from_rows),
                from_file,
                "on {on}, case {case} ({} ids, nearest {k})",
                set.len()
            );
            equally_far |= from_file.windows(2).any(|pair| pair[0].1 == pair[1].1);
        }
        assert!(
            equally_far,
            "on {on}, no case returned two equally far rows"
        );
    }
}

/// A scan that fails, other than by its request being cancelled, is followed
/// by the read of each id, and the search answers what it answers with the
/// rule off. 1,300's row is written after the replica laid its file out, so
/// its slot is past the file, and is then made the marker of an externalized
/// vector (an empty row): the scan cannot resolve it under its own guard
/// and fails, and the read of each id passes the marker over.
#[test]
fn a_set_whose_scan_fails_is_read_id_by_id() {
    let (space, replica) = large_space("exact_scan_failed");
    let tail = space.last();
    space.write(&tail, &[(1_300, [0.5, 0.5, 0.0])]);
    space.row(&tail, 1_300, Some(b""));
    let set: HashSet<u128> = (1..=600).chain([1_300]).collect();
    let scored = || way(|| replica.by_id(QUERY, 40, &set));
    let (read_each, why) = super::exact_search::with_exact_scan_rows_per_id_for_test(0, scored);
    assert_eq!(why, "per_id_off");
    assert_eq!(read_each.len(), 40);
    let (found, why) = scored();
    assert_eq!(why, "scan_failed");
    assert_eq!(found, read_each);
}

/// A scan that declines is followed by the read of each id too. The rule
/// hands a set to a scan only where every segment has its file open, so a
/// scan declines only if a file goes in between: the tail's is closed there.
#[test]
fn a_set_whose_scan_declines_is_read_id_by_id() {
    let (space, replica) = large_space("exact_scan_declined");
    let set: HashSet<u128> = (1..=600).collect();
    let scored = || way(|| replica.by_id(QUERY, 40, &set));
    let (read_each, why) = super::exact_search::with_exact_scan_rows_per_id_for_test(0, scored);
    assert_eq!(why, "per_id_off");
    super::exact_search::close_before_exact_scan_for_test(&space.last());
    let (found, why) = scored();
    assert_eq!(why, "scan_declined");
    assert_eq!(found, read_each);
    // The file is gone now, and the rule says so before any scan starts.
    assert_eq!(scored().1, "per_id_segment");
}
