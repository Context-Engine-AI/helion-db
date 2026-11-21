//! The stored row wins over the writer's local sidecar slot.
//!
//! A segment's vectors are in the store as rows and, for scoring, in a flat
//! file on the writer's disk. A seal publishes that file whole as the
//! segment's blob, which every reader restores and scores from, and a merge
//! writes the target's rows from what it exports of its sources. A node's
//! death can leave zero blocks inside the file's length, so a slot that
//! differs from its row must reach neither the blob nor a merged row.

use super::*;
use crate::helix_engine::storage_core::backend::SegmentDb;
use crate::helix_engine::vector_core::spindle::{decode_vector, SpindleConfig, SpindleMode};
use crate::helix_engine::vector_core::vector_core::HVEC_SIDECAR_BLOB_KEY;

const DIM: usize = 16;
const ROWS: usize = 64;
/// The point whose sidecar slot is zeroed. Its row is `rows[DAMAGED - 1]`.
const DAMAGED: u128 = 7;
/// An `.hvec` file is this header and then one slot of `DIM` floats a row.
const HVEC_HEADER: usize = 16;

fn hnsw() -> HNSWConfig {
    HNSWConfig::new(Some(16), Some(64), Some(128))
}

/// A writer's collection as the client's are: a create without a
/// quantization block stores raw f32 rows and an exact sidecar. (The
/// helpers of this module's parent make a TurboProd space, which has
/// neither.) `ROWS` seeded rows, ids from 1, in the tail, sealed.
fn sealed_raw_collection(
    dir: &std::path::Path,
    name: &str,
) -> (
    Arc<CollectionManager>,
    Arc<HelixGraphStorage>,
    Vec<Vec<f32>>,
) {
    use rand::{Rng, SeedableRng};
    let mut config = Config::new(16, 64, 128, 1).with_storage_backend(StorageBackendConfig::Lsm {
        storage: LsmStorage::ObjectStore,
        role: LsmRole::Writer,
    });
    config.vector_config.flat_scan_threshold = Some(0);
    let writer = Arc::new(CollectionManager::new(dir.to_path_buf(), config.clone()).unwrap());
    let raw = NamedVectorConfig {
        size: DIM,
        distance: DistanceMetric::Cosine,
        spindle: SpindleConfig {
            mode: SpindleMode::None,
            keep_original: false,
            rescore: false,
            oversampling: 1,
            binary_dims: 1,
            turbo_dims: 1,
        },
    };
    apply_mutation(
        &writer,
        &config,
        &ReplicatedMutation::CreateCollection {
            name: name.into(),
            vectors: HashMap::from([("dense".to_string(), raw)]),
            sparse_vectors: HashMap::new(),
            hnsw_overrides: None,
        },
    )
    .unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(5);
    let rows: Vec<Vec<f32>> = (0..ROWS)
        .map(|_| (0..DIM).map(|_| rng.random_range(-1.0f32..1.0)).collect())
        .collect();
    apply_mutation(
        &writer,
        &config,
        &ReplicatedMutation::UpsertPoints {
            collection: name.into(),
            points: rows
                .iter()
                .enumerate()
                .map(|(at, row)| ReplicatedPoint {
                    id: at as u128 + 1,
                    vectors: HashMap::from([("dense".to_string(), row.clone())]),
                    sparse_vectors: HashMap::new(),
                    payload: HashMap::new(),
                })
                .collect(),
        },
    )
    .unwrap();
    let storage = writer.get_collection(name).unwrap();
    // An index job queued by the upsert would seal and build by itself.
    wait_for_index_executor(name);
    assert!(seal_lsm_dense_tail(&storage, "dense", &hnsw()).unwrap());
    (writer, storage, rows)
}

fn segment_value(
    storage: &HelixGraphStorage,
    physical_name: &str,
    db: SegmentDb,
    key: &[u8],
) -> Option<Vec<u8>> {
    let r = storage.backend.begin_read().unwrap();
    storage
        .backend
        .get_with(&r, Namespace::Segment { physical_name, db }, key, |row| {
            row.map(|bytes| bytes.to_vec())
        })
        .unwrap()
}

fn slot_range(storage: &HelixGraphStorage, segment: &str, id: u128) -> std::ops::Range<usize> {
    let ordinal = segment_value(storage, segment, SegmentDb::Ordinals, &id.to_be_bytes())
        .expect("the point has a sidecar slot");
    let start = HVEC_HEADER + u64::from_le_bytes(ordinal.try_into().unwrap()) as usize * DIM * 4;
    start..start + DIM * 4
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|value| f32::from_ne_bytes(value.try_into().unwrap()))
        .collect()
}

fn sidecar_file(storage: &HelixGraphStorage, segment: &str) -> std::path::PathBuf {
    storage
        .collection_path()
        .join(segment)
        .with_extension("hvec")
}

/// Zeroes `id`'s slot in the writer's local file, inside its length, and
/// returns the file as it then is.
fn zero_sidecar_slot(storage: &HelixGraphStorage, segment: &str, id: u128) -> Vec<u8> {
    use std::io::{Seek, SeekFrom, Write};
    let path = sidecar_file(storage, segment);
    let slot = slot_range(storage, segment, id);
    assert!(std::fs::read(&path).unwrap()[slot.clone()]
        .iter()
        .any(|byte| *byte != 0));
    let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(slot.start as u64)).unwrap();
    file.write_all(&vec![0u8; slot.len()]).unwrap();
    file.sync_all().unwrap();
    std::fs::read(&path).unwrap()
}

/// `id`'s slot in the blob the store holds for `segment`.
fn blob_slot(storage: &HelixGraphStorage, segment: &str, id: u128) -> Vec<f32> {
    let blob = segment_value(
        storage,
        segment,
        SegmentDb::VectorData,
        HVEC_SIDECAR_BLOB_KEY,
    )
    .expect("the segment has a published blob");
    floats(&blob[slot_range(storage, segment, id)])
}

fn stored_row(storage: &HelixGraphStorage, segment: &str, id: u128) -> Vec<f32> {
    let row = segment_value(
        storage,
        segment,
        SegmentDb::Vectors,
        &vector_key_for_test(id, 0),
    )
    .expect("the point has a level-0 row");
    decode_vector(&row).unwrap()
}

/// How often a build found the sidecar at odds with its rows while `run` ran
/// on this thread.
fn build_mismatches<T>(run: impl FnOnce() -> T) -> (T, usize) {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let out = metrics::with_local_recorder(&recorder, run);
    let counted = handle
        .render()
        .lines()
        .filter(|line| {
            line.starts_with("helix_lsm_sidecar_verify_total") && line.contains("build_mismatch")
        })
        .map(|line| line.rsplit(' ').next().unwrap().parse::<usize>().unwrap())
        .sum();
    (out, counted)
}

fn in_memory_store(prefix: &'static str) -> (EnvVarGuard, impl Drop) {
    let store: Arc<dyn slatedb::object_store::ObjectStore> =
        Arc::new(slatedb::object_store::memory::InMemory::new());
    (
        EnvVarGuard::set("HELIX_LSM_PREFIX", prefix),
        crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store(store),
    )
}

/// Seals a tail whose local file has one zeroed slot (`damaged`), or none.
/// Returns the stored rows, the file as it was before the build, the blob the
/// build published, and how many times the build said the file differed.
fn seal(name: &'static str, damaged: bool) -> (Vec<Vec<f32>>, Vec<u8>, Vec<u8>, usize) {
    let (_prefix, _store) = in_memory_store(name);
    let dir = TempDir::new().unwrap();
    let (_writer, storage, rows) = sealed_raw_collection(dir.path(), name);
    let tail = storage
        .named_vectors
        .building_segment_names("dense")
        .remove(0);
    let file = if damaged {
        zero_sidecar_slot(&storage, &tail, DAMAGED)
    } else {
        storage.named_vectors.cores_read().unwrap()[&tail]
            .flush_mmap()
            .unwrap();
        std::fs::read(sidecar_file(&storage, &tail)).unwrap()
    };

    let (prepared, mismatches) = build_mismatches(|| {
        let r = storage.backend.begin_read().unwrap();
        storage
            .named_vectors
            .prepare_building_indices_limited_be(&r, "dense", 1)
            .unwrap()
    });
    assert_eq!(prepared.len(), 1);
    // The graph is built around the stored rows.
    let index = &prepared[0].1;
    for (at, id) in index.point_ids.iter().enumerate() {
        assert_eq!(
            index.raw_at(at).unwrap(),
            rows[*id as usize - 1].as_slice(),
            "the build took point {id} from elsewhere than its row"
        );
    }
    assert!(run_chunked_indices_publish(&storage, "dense", prepared).unwrap());

    for (at, row) in rows.iter().enumerate() {
        let id = at as u128 + 1;
        assert_eq!(stored_row(&storage, &tail, id), *row);
        assert_eq!(
            blob_slot(&storage, &tail, id),
            *row,
            "the published blob's slot for point {id} is not its stored row"
        );
    }
    let blob = segment_value(
        &storage,
        &tail,
        SegmentDb::VectorData,
        HVEC_SIDECAR_BLOB_KEY,
    )
    .unwrap();

    // A reader has no local file: it restores the blob and scores from it.
    let reader_dir = TempDir::new().unwrap();
    let reader = CollectionManager::new(
        reader_dir.path().to_path_buf(),
        Config::new(16, 64, 128, 1).with_storage_backend(StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role: LsmRole::Reader,
        }),
    )
    .unwrap();
    let replica = reader.get_collection(name).unwrap();
    let read = replica.backend.begin_read().unwrap();
    let hits = replica
        .named_vectors
        .dense_search_with_id_filter_ef_be::<fn(u128) -> bool>(
            &read,
            "dense",
            &rows[DAMAGED as usize - 1],
            ROWS,
            None,
            true,
            None,
            None,
        )
        .unwrap();
    let own = hits
        .iter()
        .find(|hit| hit.get_id() == DAMAGED)
        .expect("the reader finds the point");
    assert!(
        own.get_distance().abs() < 1e-4,
        "the reader scored point {DAMAGED} against its own vector at {}",
        own.get_distance()
    );
    (rows, file, blob, mismatches)
}

/// Red before the change twice over: the graph was built around the zeroed
/// slot, and the blob the readers score from held it.
#[test]
fn seal_publishes_the_stored_row_where_the_tail_sidecar_slot_differs() {
    let _guard = ENV_LOCK.lock().unwrap();
    let (rows, file, blob, mismatches) = seal("stored_row_wins_seal", true);
    assert_eq!(mismatches, 1, "one build, said once");
    // The file was built again from the rows: only the zeroed slot's bytes
    // had to change, and every live slot is where its ordinal puts it.
    assert_eq!(blob.len(), file.len());
    assert_eq!(blob.len(), HVEC_HEADER + rows.len() * DIM * 4);
}

/// Why the change is safe to take: where no slot differs, nothing is rebuilt
/// and the blob is the writer's file byte for byte, as it was before.
#[test]
fn seal_of_a_clean_sidecar_publishes_the_file_as_it_is() {
    let _guard = ENV_LOCK.lock().unwrap();
    let (_rows, file, blob, mismatches) = seal("stored_row_wins_clean", false);
    assert_eq!(mismatches, 0);
    assert!(blob == file, "the blob is not the writer's file");
}

/// A segment sealed before this change can have a wrong slot in its local
/// file and in its blob while its row is right. Its next merge writes the
/// row. Red before the change: the merged row and the merged segment's blob
/// held the zeroed slot, and the source with the right row was retired.
#[test]
fn merge_writes_the_stored_row_where_a_source_sidecar_slot_differs() {
    let _guard = ENV_LOCK.lock().unwrap();
    let name = "stored_row_wins_merge";
    let (_prefix, _store) = in_memory_store(name);
    let dir = TempDir::new().unwrap();
    let (_writer, storage, rows) = sealed_raw_collection(dir.path(), name);
    run_index_job(IndexJob {
        collection: name.into(),
        storage: Arc::downgrade(&storage),
        spaces_to_check: vec!["dense".into()],
        hnsw_config: hnsw(),
        flat_scan_threshold: 1_000,
    });
    let source = sole_indexed_segment(&storage);
    zero_sidecar_slot(&storage, &source, DAMAGED);

    let target = write_merge_target(
        &storage,
        "dense",
        &hnsw(),
        prepared_merge_of(&storage, &source),
    )
    .unwrap();

    for (at, row) in rows.iter().enumerate() {
        let id = at as u128 + 1;
        assert_eq!(
            stored_row(&storage, &target, id),
            *row,
            "the merged row of point {id} is not the source's row"
        );
        assert_eq!(
            blob_slot(&storage, &target, id),
            *row,
            "the merged segment's blob slot for point {id} is not the source's row"
        );
    }
}
