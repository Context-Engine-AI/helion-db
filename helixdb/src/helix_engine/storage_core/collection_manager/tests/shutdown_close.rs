//! The shutdown step that closes every loaded LSM writer
//! (`CollectionManager::close_lsm_writers`).

use super::idle_gc::{get_row, put_row, Writer, ROWS};
use super::*;
use crate::helix_engine::storage_core::backend_lsm::allow_lsm_blocking;

/// SSTs under `name`'s `compacted/`: what a flushed memtable leaves.
fn ssts(writer: &Writer, name: &str) -> usize {
    let prefix = format!("{}/{name}/compacted/", writer.root);
    writer
        .objects()
        .keys()
        .filter(|location| location.starts_with(&prefix))
        .count()
}

/// The step as the gateway runs it: on the blocking pool of a runtime.
fn shutdown_close(writer: &Writer, within: Duration) -> ShutdownClose {
    let manager = Arc::clone(&writer.manager);
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

#[test]
fn shutdown_close_flushes_the_memtable_of_every_loaded_writer() {
    let writer = Writer::new("shutdown_close_all");
    // Given two loaded collections whose acked rows are in the WAL and the
    // memtable only.
    for name in ["a", "b"] {
        let storage = writer.manager.create_collection(name).unwrap();
        put_row(&storage, b"k1", b"v1");
        put_row(&storage, b"k2", b"v2");
        assert_eq!(ssts(&writer, name), 0);
    }

    // When the shutdown step runs with time to spare.
    let report = shutdown_close(&writer, Duration::from_secs(30));

    // Then each memtable is an SST, and the step counts both writers closed.
    for name in ["a", "b"] {
        assert!(ssts(&writer, name) > 0, "{name}: memtable not flushed");
    }
    assert_eq!(
        report,
        ShutdownClose {
            closed: 2,
            left: Vec::new(),
            left_for: None
        }
    );
    // And a writer that opens them afterwards reads the rows.
    for name in ["a", "b"] {
        writer.close(name);
        let storage = writer.manager.get_collection(name).unwrap();
        assert_eq!(get_row(&storage, b"k1").as_deref(), Some(&b"v1"[..]));
        assert_eq!(get_row(&storage, b"k2").as_deref(), Some(&b"v2"[..]));
    }
}

#[test]
fn shutdown_close_past_its_deadline_closes_nothing_and_names_what_it_left() {
    let writer = Writer::new("shutdown_close_late");
    // Given a loaded collection with unflushed rows.
    let storage = writer.manager.create_collection("a").unwrap();
    put_row(&storage, b"k1", b"v1");

    // When the step starts with its time already used up.
    let report = shutdown_close(&writer, Duration::ZERO);

    // Then the writer is left as it was: named, with why, open and unflushed.
    assert_eq!(
        report,
        ShutdownClose {
            closed: 0,
            left: vec!["a".to_string()],
            left_for: Some("deadline")
        }
    );
    assert_eq!(ssts(&writer, "a"), 0);
    put_row(&storage, b"k2", b"v2");
    assert_eq!(get_row(&storage, b"k2").as_deref(), Some(&b"v2"[..]));
}

#[test]
fn an_error_under_the_shutdown_close_marks_no_collection_degraded() {
    let writer = Writer::new("shutdown_close_no_marker");
    // Given a loaded collection, and the kind of error a read cut in the
    // middle could give: one that reads like corruption.
    let storage = writer.manager.create_collection("a").unwrap();
    put_row(&storage, b"k1", b"v1");
    let cut = GraphError::FatalCollectionStorage {
        code: "collection_degraded",
        message: "unexpected eof".to_string(),
    };
    let marker = || HelixGraphStorage::read_degraded_marker(storage.path().to_str().unwrap());

    // When the shutdown step has closed the writer and a request that is
    // still running on it fails that way.
    shutdown_close(&writer, Duration::from_secs(30));
    storage.mark_collection_degraded(&cut, "test");

    // Then the collection is not degraded, and no marker waits in its
    // directory for the next start.
    assert!(storage.ensure_not_degraded().is_ok());
    assert!(marker().is_none(), "{:?}", marker());
}

#[test]
fn shutdown_close_runs_as_many_closes_at_once_as_free_memory_holds() {
    const GIB: usize = 1 << 30;
    // Twice production's unflushed allowance of 256 MiB.
    let reserve = 512 << 20;

    // No usage to read: a few, not the 32 the kernel killed a pod for.
    // Seven are the fewest that still close production's 32 loaded
    // collections in the 14 s the step has at the least, at the 2.5 s the
    // one measured round of closes took. (A limit is always there to read:
    // without a cgroup's it is the machine's memory.)
    assert_eq!(shutdown_close_workers(Some(GIB), None, reserve), 7);
    // The production writer, 5 GiB in use of 32: every thread.
    assert_eq!(
        shutdown_close_workers(Some(32 * GIB), Some(5 * GIB), reserve),
        SHUTDOWN_CLOSE_THREADS
    );
    // The pod the kernel killed, 4.63 GB in use of 6 GiB: three, not 24.
    assert_eq!(
        shutdown_close_workers(Some(6 * GIB), Some(4_632_000_000), reserve),
        3
    );
    // No room for one writer's unflushed bytes: none.
    assert_eq!(
        shutdown_close_workers(Some(6 * GIB), Some(6 * GIB - reserve + 1), reserve),
        0
    );
    assert_eq!(
        shutdown_close_workers(Some(6 * GIB), Some(7 * GIB), reserve),
        0
    );
}

/// What the step's rule does on a pod that takes `took` bytes for a close as
/// it starts and has `gave_back` of them again as it ends, the closes ending
/// one at a time: the closes started at each asking, the writers closed and
/// left, and the most memory in use.
fn closes_on_a_pod(
    (limit, in_use): (usize, usize),
    to_close: usize,
    (took, gave_back): (usize, usize),
) -> (Vec<usize>, usize, usize, usize) {
    let (mut in_use, mut left) = (in_use, to_close);
    let (mut running, mut closed, mut highest) = (0, 0, in_use);
    let mut started = Vec::new();
    loop {
        let starts = shutdown_close_starts(Some(limit), Some(in_use), 512 << 20, running).min(left);
        started.push(starts);
        (left, running, in_use) = (left - starts, running + starts, in_use + starts * took);
        highest = highest.max(in_use);
        if running == 0 {
            return (started, closed, left, highest);
        }
        (running, closed, in_use) = (running - 1, closed + 1, in_use - gave_back);
    }
}

#[test]
fn shutdown_close_starts_a_close_only_into_the_free_memory_read_as_it_starts() {
    const MIB: usize = 1 << 20;
    // The pod the kernel killed inside the step (pod smoke, 2026-11-18): 24
    // writers to close, 4,447 MiB in use of 6,144. Its first three closes took
    // 1,305 MiB between them and gave 390 back as they ended; three more,
    // started on the figure read once at the step's start, reached the limit.
    let pod = (6_144 * MIB, 4_447 * MIB);
    let (started, closed, left, highest) = closes_on_a_pod(pod, 24, (435 * MIB, 130 * MIB));

    // Three start, as they did. No fourth fits while one of them runs; with
    // all three ended 782 MiB are free and one more starts. After it 477 MiB
    // are free, short of a close's allowance with none running to wait for:
    // the step ends and the other 20 keep their WAL tails.
    assert_eq!(started, [3, 0, 0, 1, 0]);
    assert_eq!((closed, left), (4, 20));
    // The pod stays under its limit.
    assert_eq!(highest, 5_797 * MIB);
    // A close just started has taken nothing a reading shows yet: the same
    // reading with the three running lets no fourth start.
    assert_eq!(
        shutdown_close_starts(Some(pod.0), Some(pod.1), 512 * MIB, 3),
        0
    );
    // No usage to read: seven at a time, as before.
    assert_eq!(shutdown_close_starts(Some(pod.0), None, 512 * MIB, 0), 7);
    assert_eq!(shutdown_close_starts(Some(pod.0), None, 512 * MIB, 6), 1);
    assert_eq!(shutdown_close_starts(Some(pod.0), None, 512 * MIB, 7), 0);
    // The reading of this machine takes the running closes off as well.
    assert_eq!(shutdown_close_starts_now(usize::MAX).0, 0);
}

#[test]
fn shutdown_close_with_ample_memory_starts_as_many_closes_at_once_as_it_did() {
    const MIB: usize = 1 << 20;
    // The production writer, 9.5 GiB in use of 32.
    let pod = (32_768 * MIB, 9_728 * MIB);

    // The same 24 writers all start on the first reading and all close.
    let (started, closed, left, highest) = closes_on_a_pod(pod, 24, (435 * MIB, 130 * MIB));
    assert_eq!(started[0], 24);
    assert_eq!((closed, left), (24, 0));
    assert!(highest < pod.0, "{highest}");
    // With more to close than threads, every thread starts at once, the
    // figure the step had when it read the memory once.
    let (started, closed, left, _) = closes_on_a_pod(pod, 40, (435 * MIB, 130 * MIB));
    assert_eq!(started[0], SHUTDOWN_CLOSE_THREADS);
    assert_eq!(
        started[0],
        shutdown_close_workers(Some(pod.0), Some(pod.1), 512 * MIB)
    );
    assert_eq!((closed, left), (40, 0));
}

#[test]
fn shutdown_close_closes_every_writer_where_the_memory_of_a_close_comes_back() {
    const MIB: usize = 1 << 20;
    // The 6 Gi pod again, but a close gives back all it took.
    let (started, closed, left, highest) =
        closes_on_a_pod((6_144 * MIB, 4_447 * MIB), 24, (435 * MIB, 435 * MIB));

    // All 24 close. The first of the three to end leaves 827 MiB with two
    // running, room for none; from the second on each close that ends makes
    // room for the next, and no more memory is in use than under the three.
    assert_eq!((closed, left), (24, 0));
    assert_eq!(started[..4], [3, 0, 1, 1]);
    assert_eq!(highest, 5_752 * MIB);
}

/// Loaded writers under `names`, each with an unflushed row, in the order the
/// step is given them: it closes from the end.
fn loaded(writer: &Writer, names: &[&str]) -> Vec<(String, Arc<HelixGraphStorage>)> {
    names
        .iter()
        .map(|name| {
            let storage = writer.manager.create_collection(name).unwrap();
            put_row(&storage, b"k1", b"v1");
            (name.to_string(), storage)
        })
        .collect()
}

fn in_30_s() -> Instant {
    Instant::now() + Duration::from_secs(30)
}

#[test]
fn shutdown_close_with_room_for_none_leaves_every_writer_and_with_room_for_one_closes_them_in_turn()
{
    let writer = Writer::new("shutdown_close_bounded");
    // Given three loaded writers with unflushed rows.
    let storages = loaded(&writer, &["a", "b", "c"]);
    let names = || vec!["a".to_string(), "b".to_string(), "c".to_string()];

    // When the pod has no room for a close.
    let asked = Instant::now();
    let none = close_lsm_writers_by(storages.clone(), in_30_s(), |_| 0);

    // Then none is started and all are named, at once and not at the deadline.
    assert!(
        asked.elapsed() < Duration::from_secs(10),
        "waited {:?} of a 30 s deadline",
        asked.elapsed()
    );
    assert_eq!(
        none,
        ShutdownClose {
            closed: 0,
            left: names(),
            left_for: Some("free_memory")
        }
    );
    for name in names() {
        assert_eq!(ssts(&writer, &name), 0);
    }

    // When it has room for one at a time.
    let one = close_lsm_writers_by(storages, in_30_s(), |running| {
        1usize.saturating_sub(running)
    });

    // Then every writer is closed, its memtable an SST.
    assert_eq!(
        one,
        ShutdownClose {
            closed: 3,
            left: Vec::new(),
            left_for: None
        }
    );
    for name in names() {
        assert!(ssts(&writer, &name) > 0, "{name}: memtable not flushed");
    }
}

#[test]
fn shutdown_close_with_room_for_one_and_then_none_closes_one_writer_and_leaves_the_rest() {
    let writer = Writer::new("shutdown_close_no_room_again");
    // Given three loaded writers, and a pod with room for one close that the
    // close does not give back.
    let storages = loaded(&writer, &["a", "b", "c"]);
    let mut room = [1, 0].into_iter();
    let mut asked_with = Vec::new();

    // When the step runs.
    let asked = Instant::now();
    let report = close_lsm_writers_by(storages, in_30_s(), |running| {
        asked_with.push(running);
        room.next().expect("asked a third time")
    });

    // Then the last writer of the list is closed and the others are left as
    // they were and named, with why, at once and not at the deadline.
    assert_eq!(
        report,
        ShutdownClose {
            closed: 1,
            left: vec!["a".to_string(), "b".to_string()],
            left_for: Some("free_memory")
        }
    );
    assert!(
        asked.elapsed() < Duration::from_secs(10),
        "waited {:?} of a 30 s deadline",
        asked.elapsed()
    );
    assert!(ssts(&writer, "c") > 0, "c: memtable not flushed");
    for name in ["a", "b"] {
        assert_eq!(ssts(&writer, name), 0, "{name}");
    }
    // It asked again when the one close had ended, with none running.
    assert_eq!(asked_with, [0, 0]);
}

#[test]
fn a_shutdown_close_with_no_room_waits_for_a_running_close_and_starts_when_there_is_room() {
    let writer = Writer::new("shutdown_close_room_comes_back");
    // Given three loaded writers, and a pod with room for two closes, for no
    // third while one of the two still runs, and for one when both have ended.
    let storages = loaded(&writer, &["a", "b", "c"]);
    let mut room = [2, 0, 1].into_iter();
    let mut asked_with = Vec::new();

    // When the step runs.
    let report = close_lsm_writers_by(storages, in_30_s(), |running| {
        asked_with.push(running);
        room.next().expect("asked a fourth time")
    });

    // Then it does not end while a close runs, and closes all three.
    assert_eq!(
        report,
        ShutdownClose {
            closed: 3,
            left: Vec::new(),
            left_for: None
        }
    );
    for name in ["a", "b", "c"] {
        assert!(ssts(&writer, name) > 0, "{name}: memtable not flushed");
    }
    // It asked as each of the two ended, with the closes still running, and
    // not again with nothing left to start.
    assert_eq!(asked_with, [0, 1, 0]);
}

/// Writes in flight while the step closes their writer. Every write that was
/// acked must be there after a reopen, and no write begun after the step
/// returned may be acked.
#[test]
fn a_write_acked_while_the_shutdown_close_runs_is_there_after_a_reopen() {
    use crate::helix_engine::storage_core::backend::StorageBackend;
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    let writer = Writer::new("shutdown_close_under_writes");
    for round in 0..6u64 {
        let name = format!("p{round}");
        let storage = writer.manager.create_collection(&name).unwrap();
        put_row(&storage, b"seed", b"v");
        let acked: Arc<Mutex<Vec<Vec<u8>>>> = Arc::new(Mutex::new(Vec::new()));
        let refused = Arc::new(AtomicUsize::new(0));
        let late = Arc::new(AtomicUsize::new(0));
        let closed = Arc::new(AtomicBool::new(false));
        let threads: Vec<_> = (0..8u8)
            .map(|t| {
                let storage = Arc::clone(&storage);
                let acked = Arc::clone(&acked);
                let refused = Arc::clone(&refused);
                let late = Arc::clone(&late);
                let closed = Arc::clone(&closed);
                thread::spawn(move || {
                    for i in 0..10_000u32 {
                        let key = format!("t{t}-{i}").into_bytes();
                        let begun_after_close = closed.load(Ordering::Acquire);
                        let result = (|| -> Result<(), String> {
                            let mut txn =
                                storage.backend.begin_write().map_err(|e| e.to_string())?;
                            storage
                                .backend
                                .put(&mut txn, ROWS, &key, b"v")
                                .map_err(|e| e.to_string())?;
                            if t % 2 == 0 {
                                storage.backend.commit(txn).map_err(|e| e.to_string())
                            } else {
                                storage
                                    .backend
                                    .commit_buffered(txn)
                                    .map_err(|e| e.to_string())?;
                                storage.backend.flush_durable().map_err(|e| e.to_string())
                            }
                        })();
                        match result {
                            Ok(()) => {
                                if begun_after_close {
                                    late.fetch_add(1, Ordering::AcqRel);
                                }
                                acked.lock().unwrap().push(key);
                            }
                            Err(_) => {
                                refused.fetch_add(1, Ordering::AcqRel);
                                if closed.load(Ordering::Acquire) {
                                    break;
                                }
                            }
                        }
                    }
                })
            })
            .collect();
        thread::sleep(Duration::from_millis(130 + 37 * round));
        let before_close = acked.lock().unwrap().len();
        let report = shutdown_close(&writer, Duration::from_secs(30));
        closed.store(true, Ordering::Release);
        for handle in threads {
            handle.join().unwrap();
        }
        assert!(report.left.is_empty(), "{report:?}");
        let acked = acked.lock().unwrap().clone();
        drop(storage);
        writer.close(&name);
        let reopened = writer.manager.get_collection(&name).unwrap();
        let missing: Vec<String> = acked
            .iter()
            .filter(|key| get_row(&reopened, key).is_none())
            .map(|key| String::from_utf8_lossy(key).into_owned())
            .collect();
        assert!(missing.is_empty(), "acked and gone: {missing:?}");
        assert_eq!(late.load(Ordering::Acquire), 0);
        assert!(acked.len() > before_close || refused.load(Ordering::Acquire) > 0);
        drop(reopened);
        writer.close(&name);
    }
}
