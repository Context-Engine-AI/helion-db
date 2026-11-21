//! The flush gate (`backend_lsm_flush_gate.rs`) under the collection manager:
//! the shutdown step, an eviction, a seal and a merge beside a memtable flush
//! that waits for memory.

use super::idle_gc::{get_row, put_row, Writer};
use super::*;
use crate::helix_engine::storage_core::backend_any::AnyBackend;
use crate::helix_engine::storage_core::backend_lsm::flush_gate::test_support::{
    gate_reading, joined, wait_until, HeldSsts,
};
use crate::helix_engine::storage_core::backend_lsm::{
    allow_lsm_blocking, with_flush_gate_for_test, LsmBackend,
};
use slatedb::object_store::ObjectStore;
use slatedb::L0FlushGate;

fn lsm(storage: &HelixGraphStorage) -> &LsmBackend {
    match &*storage.backend {
        AnyBackend::Lsm(backend) => backend,
        _ => panic!("not an LSM writer"),
    }
}

/// A memtable flush of `storage`, asked for on a thread of its own: it
/// returns when the memtable is in L0.
fn flush_on_a_thread(storage: &Arc<HelixGraphStorage>) -> thread::JoinHandle<()> {
    let storage = Arc::clone(storage);
    thread::spawn(move || lsm(&storage).flush_memtable_for_test())
}

/// Evicts `name` and waits until its `Db` is closed. The eviction runs on a
/// thread: one that waited at the gate, holding the map of loaded
/// collections, fails the test and does not hang it.
fn evict(writer: &Writer, name: &'static str) {
    let manager = Arc::clone(&writer.manager);
    joined(
        "the eviction",
        thread::spawn(move || manager.evict_collection(name)),
    );
    writer.close(name);
}

/// The shutdown step as the gateway runs it: on the blocking pool of a
/// runtime.
fn shutdown_close(manager: &Arc<CollectionManager>, within: Duration) -> ShutdownClose {
    let manager = Arc::clone(manager);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    runtime
        .block_on(async move {
            tokio::task::spawn_blocking(move || {
                allow_lsm_blocking(|| manager.close_lsm_writers(None, Instant::now() + within))
            })
            .await
        })
        .unwrap()
}

/// The shutdown step ends at its deadline with its closes' uploads still
/// running, none of them held back at the gate, and the process that leaves
/// then has lost nothing: the next one reads every acked row from the WAL.
#[test]
fn the_shutdown_close_keeps_its_deadline_and_the_wal_keeps_every_acked_row() {
    let store = HeldSsts::holding("/compacted/");
    let writer = Writer::over(
        "flush_gate_shutdown",
        Arc::clone(&store) as Arc<dyn ObjectStore>,
    );
    // No memory free.
    let (gate, _memory) = gate_reading(Some(0));
    let names = ["a", "b", "c"];
    with_flush_gate_for_test(Some(Arc::clone(&gate) as Arc<dyn L0FlushGate>), || {
        for name in names {
            let storage = writer.manager.create_collection(name).unwrap();
            put_row(&storage, b"k1", b"v1");
            put_row(&storage, b"k2", b"v2");
        }
    });

    // 4 s to the deadline: 1 s for the collections, the rest kept for the
    // root database.
    let started = Instant::now();
    let report = shutdown_close(&writer.manager, Duration::from_secs(4));
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "the step ran past its deadline: {:?}",
        started.elapsed()
    );
    assert_eq!(
        report,
        ShutdownClose {
            closed: 0,
            left: names.map(str::to_string).to_vec(),
            left_for: Some("deadline"),
        }
    );
    // Its closes got to their uploads: counted at the gate, and none waiting.
    wait_until("every started close at its upload", || {
        store.held() > 0 && store.held() == gate.counts().running
    });
    assert_eq!(gate.counts().waiting, 0);

    // The process leaves here. The next one opens the collections.
    let next_dir = TempDir::new().unwrap();
    let next = CollectionManager::new(
        next_dir.path().to_path_buf(),
        test_config().with_storage_backend(StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role: LsmRole::Writer,
        }),
    )
    .unwrap();
    for name in names {
        let storage = next.get_collection(name).unwrap();
        assert_eq!(get_row(&storage, b"k1").as_deref(), Some(&b"v1"[..]));
        assert_eq!(get_row(&storage, b"k2").as_deref(), Some(&b"v2"[..]));
    }
    store.release_all();
    wait_until("the closes of the first process over", || {
        store.held() == 0 && gate.counts().running == 0
    });
}

/// An eviction closes its writer while it holds the map of loaded
/// collections. One with nothing to flush asks the gate nothing; one with
/// rows to flush is counted there and does not wait behind another
/// collection's flush, so the map is held for its own upload and no longer.
#[test]
fn an_eviction_does_not_wait_at_the_gate() {
    let store = HeldSsts::holding("/busy/compacted/");
    let writer = Writer::over(
        "flush_gate_evict",
        Arc::clone(&store) as Arc<dyn ObjectStore>,
    );
    let (gate, memory) = gate_reading(None);
    let names = ["flushed", "unflushed", "busy"];
    let [flushed, unflushed, busy] =
        with_flush_gate_for_test(Some(Arc::clone(&gate) as Arc<dyn L0FlushGate>), || {
            names.map(|name| {
                let storage = writer.manager.create_collection(name).unwrap();
                put_row(&storage, b"k1", b"v1");
                storage
            })
        });
    lsm(&flushed).flush_memtable_for_test();
    // The cache holds the last reference of each: its eviction closes it.
    drop((flushed, unflushed));

    // No memory free, and another collection's flush running.
    memory.set(Some(0));
    let busy_flush = flush_on_a_thread(&busy);
    wait_until("the running flush at its upload", || store.held() == 1);
    let before = gate.counts();

    evict(&writer, "flushed");
    assert_eq!(
        gate.counts(),
        before,
        "a close with nothing to flush asked the gate"
    );

    evict(&writer, "unflushed");
    let after = gate.counts();
    assert_eq!(
        (after.started, after.waited, after.waiting, after.running),
        (before.started + 1, before.waited, 0, 1)
    );
    assert_eq!(store.held(), 1);

    store.release_all();
    joined("the running flush", busy_flush);
    for name in ["flushed", "unflushed"] {
        let storage = writer.manager.get_collection(name).unwrap();
        assert_eq!(get_row(&storage, b"k1").as_deref(), Some(&b"v1"[..]));
    }
}

/// A seal and a merge run on a collection whose memtable flush waits at the
/// gate, each with whatever it holds, and both end once that flush is let
/// through: nothing a waiting flush needs is held by them. That their writes
/// wait meanwhile is `backend_lsm`'s test of the park: this one passes with
/// the park taken out too.
#[test]
fn a_seal_and_a_merge_wait_out_a_flush_that_waits_at_the_gate() {
    use crate::helix_engine::storage_core::replication::{
        apply_mutation, index_job_queued_or_active, test_support, ReplicatedMutation,
        ReplicatedPoint,
    };
    use crate::helix_engine::vector_core::named_vectors::{DistanceMetric, NamedVectorConfig};
    /// Rows enough in a segment for a regraph to take it.
    const POINTS: u128 = 520;
    const NAME: &str = "flush_gate_sealed";

    let store = HeldSsts::holding("/busy/compacted/");
    let writer = Writer::over(
        "flush_gate_seal",
        Arc::clone(&store) as Arc<dyn ObjectStore>,
    );
    let (gate, memory) = gate_reading(None);
    // 0 keeps upserts from queueing background index jobs.
    let mut config = writer.manager.config.clone();
    config.vector_config.flat_scan_threshold = Some(0);
    let upsert = |from: u128| {
        let points = (from..from + POINTS)
            .map(|id| ReplicatedPoint {
                id,
                vectors: HashMap::from([(
                    "dense".to_string(),
                    (0..16u128)
                        .map(|d| ((id * 31 + d * 17) % 97) as f32 / 97.0 + 0.01)
                        .collect(),
                )]),
                sparse_vectors: HashMap::new(),
                payload: HashMap::from([(
                    "path".to_string(),
                    Value::String(format!("src/f{id}.rs")),
                )]),
            })
            .collect();
        let mutation = ReplicatedMutation::UpsertPoints {
            collection: NAME.into(),
            points,
        };
        apply_mutation(&writer.manager, &config, &mutation).unwrap();
    };
    let idle_executor = || {
        wait_until("no index job on the collection", || {
            !index_job_queued_or_active(NAME)
        })
    };

    // One indexed segment, and a tail to seal.
    let (sealed, busy) =
        with_flush_gate_for_test(Some(Arc::clone(&gate) as Arc<dyn L0FlushGate>), || {
            let create = ReplicatedMutation::CreateCollection {
                name: NAME.into(),
                vectors: HashMap::from([(
                    "dense".to_string(),
                    NamedVectorConfig {
                        size: 16,
                        distance: DistanceMetric::Cosine,
                        spindle: Default::default(),
                    },
                )]),
                sparse_vectors: HashMap::new(),
                hnsw_overrides: None,
            };
            apply_mutation(&writer.manager, &config, &create).unwrap();
            upsert(1);
            (
                writer.manager.get_collection(NAME).unwrap(),
                writer.manager.create_collection("busy").unwrap(),
            )
        });
    idle_executor();
    test_support::seal_and_index(&sealed, "dense");
    upsert(1 + POINTS);
    idle_executor();
    put_row(&sealed, b"k1", b"v1");

    // From here no memory is free: one flush at a time. `step` runs on the
    // collection while its flush waits at the gate behind another's.
    memory.set(Some(0));
    let beside_its_waiting_flush = |what: &str, step: Box<dyn FnOnce() + Send>| {
        put_row(&busy, what.as_bytes(), b"v");
        let busy_flush = flush_on_a_thread(&busy);
        wait_until("the other collection's flush at its upload", || {
            store.held() == 1
        });
        let own_flush = flush_on_a_thread(&sealed);
        wait_until("the collection's own flush at the gate", || {
            gate.counts().waiting == 1
        });
        let step = thread::spawn(step);
        // Time for it to reach its first write.
        thread::sleep(Duration::from_millis(300));
        store.release(1);
        for (thread, what) in [
            (step, what),
            (busy_flush, "a flush"),
            (own_flush, "a flush"),
        ] {
            joined(what, thread);
        }
    };
    beside_its_waiting_flush("a seal", {
        let sealed = Arc::clone(&sealed);
        Box::new(move || test_support::seal_tail(&sealed, "dense"))
    });
    idle_executor();
    beside_its_waiting_flush("a merge", {
        let sealed = Arc::clone(&sealed);
        // A regraph writes its target as a merge does (`write_merge_target`).
        Box::new(move || {
            let status = test_support::regraph(&sealed, NAME, "dense");
            assert!(status.segments_rebuilt >= 1, "the regraph rebuilt nothing");
        })
    });

    assert_eq!(get_row(&sealed, b"k1").as_deref(), Some(&b"v1"[..]));
    let counts = gate.counts();
    assert_eq!((counts.running, counts.waiting, counts.waited), (0, 0, 2));
    store.release_all();
}
