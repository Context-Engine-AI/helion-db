//! The flush gate (`backend_lsm_flush_gate.rs`): its rule, and writers on an
//! in-memory object store whose memtable flushes ask it.

use super::super::flush_gate::test_support::{
    gate_counting_for, gate_reading, joined, wait_until, HeldSsts,
};
use super::super::flush_gate::{may_start, overdue, Counts, FlushGate, ALLOWANCE_TIMES, OVERDUE};
use super::*;
use slatedb::slatedb_common::clock::{DefaultSystemClock, SystemClock, SystemClockTicker};
use slatedb::slatedb_common::metrics::{DefaultMetricsRecorder, MetricValue, MetricsRecorder};
use slatedb::L0FlushGate;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

const MIB: usize = 1 << 20;

/// A writer that freezes its memtable at every write, as one does at its
/// flush size, and that SlateDB's own bound on unflushed bytes never holds.
fn flushing_at_every_write() -> Settings {
    let mut settings = Settings::default();
    settings.l0_sst_size_bytes = 1;
    settings.flush_interval = Some(Duration::from_millis(5));
    settings
}

/// A writer of `path` whose flushes ask `gate`, or nothing for `None`.
fn open(
    store: &Arc<HeldSsts>,
    gate: Option<&Arc<FlushGate>>,
    path: &str,
    settings: Settings,
) -> LsmBackend {
    let gate = gate.map(|gate| Arc::clone(gate) as Arc<dyn L0FlushGate>);
    with_flush_gate_for_test(gate, || {
        LsmBackend::open_with_store_settings(
            path,
            Arc::clone(store) as Arc<dyn ObjectStore>,
            Some(settings),
        )
    })
    .unwrap()
}

/// The value `put_one` wrote under `key`, if it is there.
fn row(be: &LsmBackend, key: &[u8]) -> Option<Vec<u8>> {
    let read = be.begin_read().unwrap();
    be.get_with(&read, Namespace::Nodes, key, |value| {
        value.map(<[u8]>::to_vec)
    })
    .unwrap()
}

fn assert_rows(be: &LsmBackend, keys: &[&[u8]]) {
    for key in keys {
        assert_eq!(
            row(be, key).as_deref(),
            Some(&b"v"[..]),
            "an acked row is gone: {}",
            String::from_utf8_lossy(key)
        );
    }
}

/// SSTs in L0, as the writer's manifest has them.
fn l0(be: &LsmBackend) -> usize {
    be.db().manifest().l0().len()
}

async fn until(gate: &FlushGate, what: &str, reached: impl Fn(Counts) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !reached(gate.counts()) {
        assert!(
            Instant::now() < deadline,
            "never: {what}: {:?}",
            gate.counts()
        );
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

/// The allowance of a flush that asks now and starts as it asks.
async fn starts_now(gate: &FlushGate, sst_bytes: usize) -> Box<dyn Send> {
    tokio::time::timeout(Duration::from_secs(30), gate.admit("a", sst_bytes))
        .await
        .expect("a flush that starts as it asks waited")
}

// ── The rule ───────────────────────────────────────────────────────────────

#[test]
fn a_flush_starts_into_free_memory_that_holds_it_beside_the_running_ones() {
    let full = 512 * MIB;
    // Nothing running: it starts whatever the memory, with none at all.
    assert!(may_start(Some(0), 0, 0, full));
    // One running: a second needs room for the whole of both.
    assert!(!may_start(Some(2 * full - 1), full, 1, full));
    assert!(may_start(Some(2 * full), full, 1, full));
    // The pod of the smoke, 1,671 MiB free of 6 Gi and twelve memtables at
    // their flush size in the same moment: three start, where all twelve did.
    let started = (0..12usize)
        .filter(|running| may_start(Some(1_671 * MIB), running * full, *running, full))
        .count();
    assert_eq!(started, 3);
    // A small one fits where a large one does not.
    assert!(may_start(Some(2 * full + 32 * MIB), 2 * full, 2, 32 * MIB));
    assert!(!may_start(Some(2 * full + 32 * MIB), 2 * full, 2, full));
    // Memory use that cannot be read holds no flush back.
    assert!(may_start(None, usize::MAX, 99, usize::MAX));
}

#[test]
fn a_flush_that_waits_is_said_once_a_minute_from_its_first_minute() {
    let asked = Instant::now();
    let at = |secs: u64| asked + Duration::from_secs(secs);
    assert_eq!(OVERDUE, Duration::from_secs(60));
    assert!(!overdue(asked, None, at(59)));
    assert!(overdue(asked, None, at(60)));
    assert!(!overdue(asked, Some(at(60)), at(119)));
    assert!(overdue(asked, Some(at(60)), at(120)));
}

#[tokio::test]
async fn flushes_start_in_the_order_they_asked_and_only_into_free_memory() {
    let (gate, memory) = gate_reading(Some(0));
    // No flush runs: the first starts with no memory free, allowed four
    // times its SST.
    let first = starts_now(&gate, 100).await;
    assert_eq!(ALLOWANCE_TIMES, 4);
    assert_eq!(gate.counts().reserved, 400);

    // A large flush asks, and a small one after it. Neither has room.
    let asking = |sst_bytes: usize| {
        let gate = Arc::clone(&gate);
        tokio::spawn(async move { gate.admit("a", sst_bytes).await })
    };
    let large = asking(1_000);
    until(&gate, "the large flush in the line", |c| c.waiting == 1).await;
    let small = asking(1);
    until(&gate, "the small flush in the line", |c| c.waiting == 2).await;

    // Room for the small one beside the first, none for the large one: the
    // small one does not start past the large one.
    memory.set(Some(400 + 4));
    tokio::time::sleep(Duration::from_millis(100)).await;
    let counts = gate.counts();
    assert_eq!((counts.running, counts.waiting), (1, 2));

    // Room for the large one beside the first: it starts. The small one then
    // needs room beside both.
    memory.set(Some(400 + 4_000));
    until(&gate, "the large flush started", |c| c.running == 2).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    let counts = gate.counts();
    assert_eq!((counts.reserved, counts.waiting), (4_400, 1));

    // The first ends: now the small one has it.
    drop(first);
    until(&gate, "the small flush started", |c| c.waiting == 0).await;
    assert_eq!(gate.counts().reserved, 4_004);

    drop((large.await.unwrap(), small.await.unwrap()));
    assert_eq!(
        gate.counts(),
        Counts {
            running: 0,
            reserved: 0,
            waiting: 0,
            started: 3,
            waited: 2,
        }
    );
}

#[tokio::test]
async fn a_flush_ended_any_way_leaves_nothing_counted() {
    let (gate, _memory) = gate_reading(Some(0));
    let running = starts_now(&gate, 100).await;

    // One gives up while it waits: its task is cancelled.
    let cancelled = {
        let gate = Arc::clone(&gate);
        tokio::spawn(async move { gate.admit("a", 100).await })
    };
    until(&gate, "a flush in the line", |c| c.waiting == 1).await;
    cancelled.abort();
    assert!(cancelled.await.is_err_and(|ended| ended.is_cancelled()));
    assert_eq!(gate.counts().waiting, 0);

    // One dies holding its allowance.
    drop(running);
    let died = {
        let gate = Arc::clone(&gate);
        tokio::spawn(async move {
            let _held = starts_now(&gate, 100).await;
            panic!("a flush that dies with its allowance");
        })
    };
    assert!(died.await.unwrap_err().is_panic());
    let counts = gate.counts();
    assert_eq!((counts.running, counts.reserved, counts.waiting), (0, 0, 0));

    // And the line goes on behind them.
    let _next = starts_now(&gate, 100).await;
    assert_eq!(gate.counts().running, 1);
}

#[tokio::test]
async fn a_flush_of_a_closing_writer_starts_at_once_and_is_counted() {
    let (gate, _memory) = gate_reading(Some(0));
    let running = starts_now(&gate, 100).await;
    let waiting = {
        let gate = Arc::clone(&gate);
        tokio::spawn(async move { gate.admit("a", 100).await })
    };
    until(&gate, "a flush in the line", |c| c.waiting == 1).await;

    // No room, and one in the line before it: it starts all the same.
    let closing = gate.start("closing", 50);
    let counts = gate.counts();
    assert_eq!(
        (counts.running, counts.reserved, counts.waiting),
        (2, 600, 1)
    );

    // The one in the line waits for it as for any running flush.
    drop(running);
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(gate.counts().waiting, 1);
    drop(closing);
    until(&gate, "the waiting flush started", |c| c.waiting == 0).await;
    drop(waiting.await.unwrap());
}

#[tokio::test]
async fn where_the_pods_memory_use_cannot_be_read_no_flush_waits() {
    let (gate, _memory) = gate_reading(None);
    let mut held = Vec::new();
    for _ in 0..12 {
        held.push(starts_now(&gate, 128 * MIB).await);
    }
    let counts = gate.counts();
    assert_eq!((counts.running, counts.waited), (12, 0));
}

#[test]
#[serial]
fn the_flush_gate_is_on_unless_its_switch_is_set_to_a_false_value() {
    {
        let _unset = EnvGuard::remove(ENV_FLUSH_GATE);
        assert!(flush_gate_enabled());
    }
    for off in ["0", "false", "off", "no", " OFF "] {
        let _off = EnvGuard::set(ENV_FLUSH_GATE, off);
        assert!(!flush_gate_enabled(), "{off:?}");
    }
    for on in ["1", "true", "on", "", "never"] {
        let _on = EnvGuard::set(ENV_FLUSH_GATE, on);
        assert!(flush_gate_enabled(), "{on:?}");
    }
}

// ── Writers ────────────────────────────────────────────────────────────────

/// Six collections reach their flush size in the same moment. Their flushes
/// start only into the memory that is free, one after the other as each ends,
/// and every acked row is read back.
#[test]
fn flushes_of_many_collections_start_only_into_the_memory_that_is_free() {
    const COLLECTIONS: usize = 6;
    let store = HeldSsts::holding("/compacted/");
    // No memory free.
    let (gate, memory) = gate_reading(Some(0));
    let paths: Vec<String> = (0..COLLECTIONS).map(|i| format!("gate/many-{i}")).collect();
    let collections: Vec<LsmBackend> = paths
        .iter()
        .map(|path| open(&store, Some(&gate), path, flushing_at_every_write()))
        .collect();
    for be in &collections {
        put_one(be, b"acked");
    }

    // Every flush has got as far as it can: to its upload, with its SST
    // encoded, or to the gate.
    wait_until("six flushes at the store or at the gate", || {
        store.held() + gate.counts().waiting == COLLECTIONS
    });
    assert_eq!(
        store.held(),
        1,
        "memtables encoded at the same moment with no memory free"
    );

    // Room for three of them.
    let allowance = gate.counts().reserved;
    assert!(allowance > 0);
    memory.set(Some(3 * allowance));
    wait_until("three flushes at the store", || store.held() == 3);
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!((store.held(), gate.counts().waiting), (3, 3));

    // Each one that ends lets one more start.
    for ended in 1..=COLLECTIONS {
        store.release(1);
        let left = COLLECTIONS - ended;
        wait_until("the next flush at the store", || {
            store.held() == left.min(3) && gate.counts().waiting == left.saturating_sub(3)
        });
    }
    assert_eq!(store.most_held(), 3);
    wait_until("every memtable in L0", || {
        collections.iter().all(|be| l0(be) == 1)
    });
    assert_eq!(
        gate.counts(),
        Counts {
            running: 0,
            reserved: 0,
            waiting: 0,
            started: COLLECTIONS as u64,
            waited: COLLECTIONS as u64 - 1,
        }
    );

    // Every acked row, from the writer and from the next one to open it.
    store.release_all();
    for (be, path) in collections.iter().zip(&paths) {
        assert_rows(be, &[b"acked"]);
        be.close().unwrap();
        let next = open(&store, None, path, Settings::default());
        assert_rows(&next, &[b"acked"]);
        next.close().unwrap();
    }
}

/// What `HELIX_FLUSH_GATE=0` gives, and what there was before the gate: with
/// no gate every collection encodes at the same moment.
#[test]
fn without_a_gate_every_collection_flushes_at_the_same_moment() {
    const COLLECTIONS: usize = 6;
    let store = HeldSsts::holding("/compacted/");
    let collections: Vec<LsmBackend> = (0..COLLECTIONS)
        .map(|i| {
            let path = format!("gate/ungated-{i}");
            open(&store, None, &path, flushing_at_every_write())
        })
        .collect();
    for be in &collections {
        put_one(be, b"acked");
    }
    wait_until("six flushes at the store", || store.held() == COLLECTIONS);
    store.release_all();
    wait_until("every memtable in L0", || {
        collections.iter().all(|be| l0(be) == 1)
    });
    for be in &collections {
        assert_rows(be, &[b"acked"]);
        be.close().unwrap();
    }
}

/// The memory a waiting flush was kept from taking does not go into the next
/// memtable instead: its collection takes no write while that flush waits.
/// Collections with no flush waiting are written as ever.
#[test]
fn a_collection_whose_flush_waits_takes_no_write_while_it_waits() {
    let store = HeldSsts::holding("/compacted/");
    let (gate, _memory) = gate_reading(Some(0));
    let running = open(
        &store,
        Some(&gate),
        "gate/park-running",
        flushing_at_every_write(),
    );
    let waiting = Arc::new(open(
        &store,
        Some(&gate),
        "gate/park-waiting",
        flushing_at_every_write(),
    ));
    // At SlateDB's own flush size: nothing of it is ever frozen here.
    let quiet = open(&store, Some(&gate), "gate/park-quiet", Settings::default());
    put_one(&running, b"k1");
    wait_until("the first flush at the store", || store.held() == 1);
    put_one(&waiting, b"k1");
    wait_until("the second flush at the gate", || {
        gate.counts().waiting == 1
    });

    let parked = {
        let waiting = Arc::clone(&waiting);
        std::thread::spawn(move || put_one(&waiting, b"k2"))
    };
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        !parked.is_finished(),
        "a write went into a collection whose flush waits for memory"
    );
    // One whose flush runs, and one with nothing to flush.
    put_one(&running, b"k2");
    put_one(&quiet, b"k1");
    // And it is read as ever: the row whose flush waits, from its memtable.
    assert_rows(&waiting, &[b"k1"]);

    store.release_all();
    joined("the parked write", parked);
    assert_rows(&running, &[b"k1", b"k2"]);
    assert_rows(&waiting, &[b"k1", b"k2"]);
    assert_rows(&quiet, &[b"k1"]);
    wait_until("nothing left at the gate", || {
        let counts = gate.counts();
        counts.running == 0 && counts.waiting == 0
    });
    for be in [&running, &*waiting, &quiet] {
        be.close().unwrap();
    }
}

/// What is acked is in the WAL before it is acked, whatever its flush does:
/// a writer that dies with one flush encoding and one waiting has lost
/// nothing.
#[test]
fn rows_acked_by_a_writer_that_died_with_a_flush_waiting_are_read_by_the_next() {
    let store = HeldSsts::holding("/compacted/");
    let (gate, _memory) = gate_reading(Some(0));
    let paths = ["gate/died-running", "gate/died-waiting"];
    let died: Vec<LsmBackend> = paths
        .iter()
        .map(|path| open(&store, Some(&gate), path, flushing_at_every_write()))
        .collect();
    put_one(&died[0], b"k1");
    wait_until("the first flush at the store", || store.held() == 1);
    put_one(&died[1], b"k1");
    wait_until("the second flush at the gate", || {
        gate.counts().waiting == 1
    });
    assert_eq!(paths.map(|path| store.ssts(path)), [0, 0]);

    // Nothing is closed and no SST was written. The next writer of each
    // collection fences the one that died and replays its WAL.
    let next: Vec<LsmBackend> = paths
        .iter()
        .map(|path| open(&store, None, path, Settings::default()))
        .collect();
    for be in &next {
        assert_rows(be, &[b"k1"]);
    }
    store.release_all();
    for be in &next {
        be.close().unwrap();
    }
}

/// A flush that gives its allowance back is counted, with the time it held
/// it: its encode and its upload, which nothing else measures. The two
/// counters are read as the process's scrape gives them.
#[test]
fn a_flush_that_ends_is_counted_with_the_time_it_held_its_allowance() {
    let (gate, _memory) = gate_reading(None);
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let kept = Duration::from_millis(20);
    metrics::with_local_recorder(&recorder, || {
        let held = gate.start("a", 100);
        std::thread::sleep(kept);
        drop(held);
    });
    let scrape = recorder.handle().render();
    let counter = |name: &str| -> u64 {
        let series = format!("{name} ");
        scrape
            .lines()
            .find_map(|line| line.strip_prefix(series.as_str()))
            .map_or(0, |count| count.parse().expect("a counter value"))
    };
    assert_eq!(counter("helix_lsm_flush_gate_held_total"), 1, "{scrape}");
    assert!(
        counter("helix_lsm_flush_gate_held_ms_total") >= kept.as_millis() as u64,
        "{scrape}"
    );
}

/// One collection's flushes, one after the other, start at once: with no
/// memory free, with all of it, and where it cannot be read.
#[test]
fn a_collection_that_flushes_alone_never_waits() {
    let store = HeldSsts::holding("nothing is held");
    for (i, free) in [Some(0), Some(usize::MAX), None].into_iter().enumerate() {
        let (gate, _memory) = gate_reading(free);
        let be = open(
            &store,
            Some(&gate),
            &format!("gate/alone-{i}"),
            flushing_at_every_write(),
        );
        for flushed in 1..=3 {
            put_one(&be, format!("k{flushed}").as_bytes());
            wait_until("its memtable in L0", || l0(&be) == flushed);
        }
        let counts = gate.counts();
        assert_eq!((counts.started, counts.waited), (3, 0), "{free:?}");
        assert_rows(&be, &[b"k1", b"k2", b"k3"]);
        be.close().unwrap();
    }
}

/// A close holds what its caller holds (an eviction, the map of loaded
/// collections) and has a deadline of its own at shutdown: its flush is
/// counted and does not wait, and one that was waiting stops waiting.
#[test]
fn a_close_does_not_wait_at_the_gate() {
    let store = HeldSsts::holding("gate/close-running/");
    let (gate, _memory) = gate_reading(Some(0));
    let running = open(
        &store,
        Some(&gate),
        "gate/close-running",
        flushing_at_every_write(),
    );
    let closing = Arc::new(open(
        &store,
        Some(&gate),
        "gate/close-closing",
        flushing_at_every_write(),
    ));
    put_one(&running, b"k1");
    wait_until("the first flush at the store", || store.held() == 1);
    put_one(&closing, b"k1");
    wait_until("the second flush at the gate", || {
        gate.counts().waiting == 1
    });
    // A write parked behind that waiting flush.
    let parked = {
        let closing = Arc::clone(&closing);
        std::thread::spawn(move || {
            let mut write = closing.begin_write().unwrap();
            closing
                .put(&mut write, Namespace::Nodes, b"k2", b"v")
                .unwrap();
            closing.commit(write)
        })
    };
    std::thread::sleep(Duration::from_millis(100));
    assert!(!parked.is_finished());

    let closed = {
        let closing = Arc::clone(&closing);
        std::thread::spawn(move || closing.close())
    };
    joined("the close", closed).unwrap();
    // The other flush still runs, and nothing waits behind it.
    let counts = gate.counts();
    assert_eq!((store.held(), counts.running, counts.waiting), (1, 1, 0));
    assert_eq!(store.ssts("gate/close-closing"), 1);
    // The parked write was not acked: its writer closed under it.
    assert!(joined("the parked write", parked).is_err());

    let next = open(&store, None, "gate/close-closing", Settings::default());
    assert_rows(&next, &[b"k1"]);
    next.close().unwrap();
    store.release_all();
    wait_until("the first flush in L0", || l0(&running) == 1);
    running.close().unwrap();
}

/// A writer that opens a collection with a WAL tail replays it into memtables
/// of the flush size and freezes all but the last. Their flushes ask the gate
/// like any other: none starts while there is no room. (The tables of a tail
/// this small are all read before the first flush has asked, so how many
/// wait is not fixed, and the replay is not seen to stop for them.)
#[test]
fn a_wal_tail_replayed_at_an_open_is_flushed_through_the_gate() {
    let store = HeldSsts::holding("gate/replay-running/");
    let (gate, _memory) = gate_reading(Some(0));
    // Two WAL objects and nothing in L0, left by a writer that is gone.
    let gone = open(&store, None, "gate/replay-tail", Settings::default());
    put_one(&gone, b"k1");
    put_one(&gone, b"k2");
    assert_eq!(store.ssts("gate/replay-tail"), 0);
    let running = open(
        &store,
        Some(&gate),
        "gate/replay-running",
        flushing_at_every_write(),
    );
    put_one(&running, b"k1");
    wait_until("the first flush at the store", || store.held() == 1);

    let opened = {
        let (store, gate) = (Arc::clone(&store), Arc::clone(&gate));
        std::thread::spawn(move || {
            open(
                &store,
                Some(&gate),
                "gate/replay-tail",
                flushing_at_every_write(),
            )
        })
    };
    wait_until("a replayed memtable at the gate", || {
        gate.counts().waiting >= 1
    });
    std::thread::sleep(Duration::from_millis(100));
    let counts = gate.counts();
    assert_eq!(
        (counts.running, store.ssts("gate/replay-tail")),
        (1, 0),
        "{counts:?}"
    );

    store.release_all();
    let next = joined("the open", opened);
    assert_rows(&next, &[b"k1", b"k2"]);
    wait_until("the replayed memtable in L0", || l0(&next) >= 1);
    assert!(gate.counts().waited >= 1);
    next.close().unwrap();
    running.close().unwrap();
    drop(gone);
}

/// One collection's uploads fail for ever, with no memory free. Its flush
/// gives its allowance back while it waits to try again, so three other
/// collections go on inside that one wait: their memtables reach L0 and their
/// writes go in. Its own rows are in the WAL, and the next writer reads them.
#[test]
fn a_flush_whose_upload_keeps_failing_does_not_hold_the_others() {
    let store = HeldSsts::holding("nothing is held");
    store.refuse(Some("gate/failing/"));
    let (gate, _memory) = gate_reading(Some(0));
    let mut settings = flushing_at_every_write();
    // The pause between two attempts at an upload, as in production.
    settings.manifest_poll_interval = Duration::from_secs(30);
    let failing = open(&store, Some(&gate), "gate/failing", settings);
    put_one(&failing, b"k1");
    wait_until("its upload failed, and its allowance given back", || {
        store.refused() == 1 && gate.counts().running == 0
    });

    let others: Vec<Arc<LsmBackend>> = (0..3)
        .map(|i| {
            let path = format!("gate/other-{i}");
            Arc::new(open(&store, Some(&gate), &path, flushing_at_every_write()))
        })
        .collect();
    let writers: Vec<_> = others
        .iter()
        .map(|be| {
            let be = Arc::clone(be);
            std::thread::spawn(move || {
                for key in [b"k1", b"k2", b"k3"] {
                    put_one(&be, key);
                }
            })
        })
        .collect();
    for writer in writers {
        joined("another collection's writes", writer);
    }
    wait_until("the others' memtables in L0", || {
        others.iter().all(|be| l0(be) == 3)
    });
    assert_eq!(
        store.refused(),
        1,
        "the others went on only once the failing upload was tried again"
    );
    let counts = gate.counts();
    assert_eq!((counts.running, counts.waiting), (0, 0));
    for be in &others {
        assert_rows(be, &[b"k1", b"k2", b"k3"]);
        be.close().unwrap();
    }

    // What the failing flush holds was acked from the WAL, and is read from
    // it by the writer that fences this one.
    assert_eq!(store.ssts("gate/failing"), 0);
    store.refuse(None);
    let next = open(&store, None, "gate/failing", Settings::default());
    assert_rows(&next, &[b"k1"]);
    next.close().unwrap();
    drop(failing);
}

/// An upload the store does not try again inside its own call, here one it
/// refuses outright, comes back to SlateDB's uploader. That waits its poll
/// interval and tries the same SST again, for ever, and writes no line. The
/// gate is told each time, with the collection and the store's error, and
/// of nothing once the upload gets through.
#[test]
fn the_gate_is_told_of_each_upload_that_fails_and_waits_to_be_tried_again() {
    let store = HeldSsts::holding("nothing is held");
    store.refuse(Some("gate/refused/"));
    let (gate, _memory) = gate_reading(None);
    let mut settings = flushing_at_every_write();
    // The pause between two attempts at an upload: short, to see a second.
    let pause = Duration::from_millis(50);
    settings.manifest_poll_interval = pause;
    let refused = open(&store, Some(&gate), "gate/refused", settings);
    put_one(&refused, b"k1");
    wait_until("the gate told of a second failed upload", || {
        gate.upload_failures().0 >= 2
    });
    let (db, error, retry_in) = gate.upload_failures().1.expect("a failure was told");
    assert_eq!((db.as_str(), retry_in), ("gate/refused", pause));
    assert!(error.contains("refused for the test"), "{error}");

    store.refuse(None);
    wait_until("the memtable in L0 once the store takes it", || {
        l0(&refused) == 1
    });
    assert_eq!(
        gate.upload_failures().0,
        store.refused() as u64,
        "one failure told for each upload the store refused"
    );
    assert_rows(&refused, &[b"k1"]);
    refused.close().unwrap();
}

/// `len` bytes no codec makes smaller, so that their SST is as large.
fn unpacked(len: usize) -> Vec<u8> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut bytes = Vec::with_capacity(len + 8);
    while bytes.len() < len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        bytes.extend_from_slice(&state.to_le_bytes());
    }
    bytes.truncate(len);
    bytes
}

/// An SST over the 10 MiB one put may hold goes up in parts. A part that
/// fails ends that upload, and after its pause the uploader begins another
/// for the same SST. Unless the one that failed is aborted the bucket keeps
/// the parts of it that got through, of every try, where no listing shows
/// them.
#[test]
fn an_upload_in_parts_is_aborted_when_a_part_fails_before_it_is_tried_again() {
    let store = HeldSsts::holding("nothing is held");
    store.refuse_part(Some("parts/refused/"));
    let mut settings = flushing_at_every_write();
    settings.manifest_poll_interval = Duration::from_millis(50);
    let be = open(&store, None, "parts/refused", settings);
    let value = unpacked(11 * MIB);
    let mut w = be.begin_write().unwrap();
    be.put(&mut w, Namespace::Nodes, b"big", &value).unwrap();
    be.commit(w).unwrap();

    // A second upload has begun: the first failed and its error went back.
    wait_until("the upload was tried a second time", || {
        store.part_uploads().0 >= 2
    });
    let (begun, aborted, completed) = store.part_uploads();
    assert!(
        aborted + 1 >= begun,
        "of {begun} uploads begun, {aborted} were aborted and {completed} completed"
    );

    store.refuse_part(None);
    wait_until("the SST in L0 once its parts get through", || l0(&be) == 1);
    let (begun, aborted, completed) = store.part_uploads();
    assert_eq!(
        (completed, begun - aborted),
        (1, 1),
        "an upload was left unfinished: {begun} begun, {aborted} aborted"
    );
    assert_eq!(row(&be, b"big").map(|read| read.len()), Some(value.len()));
    be.close().unwrap();
}

/// An upload the store tries again without end never gives its allowance
/// back. Once a flush has held it too long to be an upload that gets through
/// its bytes count no longer: the one behind it starts into memory that holds
/// its own allowance, with the first still holding. It still runs: into less
/// than that the one behind does not start, as it would with no flush running.
#[tokio::test]
async fn a_flush_that_holds_its_allowance_too_long_still_runs_and_its_bytes_count_no_longer() {
    let (gate, memory) = gate_counting_for(Some(399), Duration::from_millis(200));
    let stuck = starts_now(&gate, 100).await;
    let behind = {
        let gate = Arc::clone(&gate);
        tokio::spawn(async move { gate.admit("b", 100).await })
    };
    until(&gate, "a flush in the line", |c| c.waiting == 1).await;
    tokio::time::sleep(Duration::from_millis(600)).await;
    let counts = gate.counts();
    assert_eq!((counts.running, counts.waiting), (1, 1));

    memory.set(Some(400));
    until(&gate, "the flush behind it started", |c| c.waiting == 0).await;
    let counts = gate.counts();
    assert_eq!(
        (counts.running, counts.reserved, counts.waited),
        (2, 800, 1)
    );
    drop((stuck, behind.await.unwrap()));
    assert_eq!(gate.counts().running, 0);
}

/// A flush that starts as it asks never waited. A write of its collection
/// that looks while the gate is still reading the pod's memory is not held
/// back for it: counted as waiting from the moment it asks, the flush would
/// send that write into the wait for its whole upload.
#[test]
fn a_write_that_looks_while_its_collections_flush_asks_is_not_held_back() {
    use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
    // Uploads are held: a write held back for the flush would stay so.
    let store = HeldSsts::holding("/compacted/");
    // A gate whose reading of the memory takes until the test lets it go.
    let (asking, let_go) = (
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
    );
    let gate = Arc::new(FlushGate::timed(
        Box::new({
            let (asking, let_go) = (Arc::clone(&asking), Arc::clone(&let_go));
            move || {
                asking.store(true, SeqCst);
                // Longer than `joined` waits for the write below, and than a
                // held write's own 30 s look: held back for this reading, the
                // write is still held when `joined` gives up.
                let deadline = Instant::now() + Duration::from_secs(60);
                while !let_go.load(SeqCst) && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Some(usize::MAX)
            }
        }),
        Duration::from_millis(10),
        OVERDUE,
    ));
    let be = Arc::new(open(
        &store,
        Some(&gate),
        "gate/asking",
        flushing_at_every_write(),
    ));
    put_one(&be, b"k1");
    wait_until("the flush asking", || asking.load(SeqCst));

    let write = {
        let be = Arc::clone(&be);
        std::thread::spawn(move || put_one(&be, b"k2"))
    };
    joined("a write while the flush asks", write);

    let_go.store(true, SeqCst);
    store.release_all();
    wait_until("both memtables in L0", || l0(&be) == 2);
    assert_rows(&be, &[b"k1", b"k2"]);
    be.close().unwrap();
}

// ── The wake of a held write ───────────────────────────────────────────────

/// How often a write of one writer has been held back so far: SlateDB's
/// `db/backpressure_count`, counted once for each look that found the write
/// had to wait. The engine reads it nowhere: a test hands the builder a
/// recorder of its own and reads its snapshot.
struct Parks(Arc<DefaultMetricsRecorder>);

impl Parks {
    fn count(&self) -> u64 {
        self.0
            .snapshot()
            .by_name(slatedb::db_stats::BACKPRESSURE_COUNT)
            .iter()
            .map(|metric| match &metric.value {
                MetricValue::Counter(count) => *count,
                _ => 0,
            })
            .sum()
    }
}

/// A clock that keeps one 30 s sleep from being made until the test lets it
/// go: the next one asked of it once the test says so. SlateDB asks its clock
/// for a sleep of 30 s in one place, the bound of a held write's wait, which
/// it makes after the write has looked at the gate and before it waits.
#[derive(Debug)]
struct HeldSleep {
    clock: DefaultSystemClock,
    /// 0: none is held. 1: the next one is to be. 2: it waits. 3: let go.
    state: Mutex<u8>,
    changed: Condvar,
}

impl HeldSleep {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            clock: DefaultSystemClock::new(),
            state: Mutex::new(0),
            changed: Condvar::new(),
        })
    }

    fn hold_the_next(&self) {
        *self.state.lock().unwrap() = 1;
    }

    fn one_waits(&self) -> bool {
        *self.state.lock().unwrap() == 2
    }

    fn let_go(&self) {
        *self.state.lock().unwrap() = 3;
        self.changed.notify_all();
    }
}

impl SystemClock for HeldSleep {
    fn now(&self) -> chrono::DateTime<chrono::Utc> {
        self.clock.now()
    }

    fn sleep<'a>(&'a self, duration: Duration) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        if duration == Duration::from_secs(30) {
            let mut state = self.state.lock().unwrap();
            if *state == 1 {
                *state = 2;
                // Bounded: a test that fails before `let_go` leaves no
                // runtime thread waiting for good.
                let _state = self
                    .changed
                    .wait_timeout_while(state, Duration::from_secs(120), |state| *state != 3)
                    .unwrap();
            }
        }
        self.clock.sleep(duration)
    }

    fn ticker<'a>(&'a self, duration: Duration) -> SystemClockTicker<'a> {
        SystemClockTicker::new(self, duration)
    }
}

/// [`open`], with the writer's held writes counted and, for `clock`, its
/// 30 s sleeps asked of that.
fn open_watched(
    store: &Arc<HeldSsts>,
    gate: &Arc<FlushGate>,
    path: &str,
    settings: Settings,
    clock: Option<&Arc<HeldSleep>>,
) -> (LsmBackend, Parks) {
    let recorder = Arc::new(DefaultMetricsRecorder::new());
    let be = {
        let (recorder, clock) = (Arc::clone(&recorder), clock.cloned());
        with_db_builder_for_test(
            move |builder| {
                let builder = builder
                    .with_metrics_recorder(Arc::clone(&recorder) as Arc<dyn MetricsRecorder>);
                match &clock {
                    Some(clock) => {
                        builder.with_system_clock(Arc::clone(clock) as Arc<dyn SystemClock>)
                    }
                    None => builder,
                }
            },
            || open(store, Some(gate), path, settings),
        )
    };
    (be, Parks(recorder))
}

/// Two writers on a gate that reads no memory free: the first one's flush is
/// at the store, where every upload is held, and the second one's waits at
/// the gate. The second, and the count of its held writes.
fn a_flush_waiting_behind_one_at_the_store(
    store: &Arc<HeldSsts>,
    gate: &Arc<FlushGate>,
    name: &str,
    clock: Option<&Arc<HeldSleep>>,
) -> (LsmBackend, Arc<LsmBackend>, Parks) {
    let running = open(
        store,
        Some(gate),
        &format!("gate/{name}-running"),
        flushing_at_every_write(),
    );
    let (waiting, parks) = open_watched(
        store,
        gate,
        &format!("gate/{name}-waiting"),
        flushing_at_every_write(),
        clock,
    );
    put_one(&running, b"k1");
    wait_until("the first flush at the store", || store.held() == 1);
    put_one(&waiting, b"k1");
    wait_until("the second flush at the gate", || {
        gate.counts().waiting == 1
    });
    (running, Arc::new(waiting), parks)
}

/// Waits until `reached`, for 10 s at most. A held write that nothing wakes
/// looks again after 30 s by itself: a bound under that tells the two apart.
fn within_ten_seconds(what: &str, mut reached: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !reached() {
        assert!(Instant::now() < deadline, "not within 10 s: {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn put_on_a_thread(be: &Arc<LsmBackend>, key: &'static [u8]) -> std::thread::JoinHandle<()> {
    let be = Arc::clone(be);
    std::thread::spawn(move || put_one(&be, key))
}

/// A write held back for a flush that waited at the gate goes in when that
/// flush is let in, as a write does that arrives after it. It does not wait
/// for the upload: here no upload ends before the test lets it.
#[test]
fn a_write_held_for_a_waiting_flush_goes_in_when_the_flush_is_let_in() {
    let store = HeldSsts::holding("/compacted/");
    let (gate, memory) = gate_reading(Some(0));
    let (running, waiting, parks) =
        a_flush_waiting_behind_one_at_the_store(&store, &gate, "wake", None);
    let held = put_on_a_thread(&waiting, b"k2");
    wait_until("the write held back", || parks.count() == 1);

    // Room for both: the waiting flush is let in, and the write goes in.
    memory.set(Some(usize::MAX));
    within_ten_seconds("the held write goes in", || held.is_finished());
    held.join().unwrap();
    assert_eq!(parks.count(), 1, "the write was held back a second time");
    // No upload is over, the test has let none through: the store keeps
    // the first flush's and the one that was let in.
    wait_until("both uploads at the store", || store.held() >= 2);

    store.release_all();
    assert_rows(&waiting, &[b"k1", b"k2"]);
    for be in [&running, &*waiting] {
        be.close().unwrap();
    }
}

/// The flush is let in after the write has looked at the gate and before it
/// waits: the write reads the count again as its wait begins, and goes in.
#[test]
fn a_flush_let_in_between_a_writes_look_and_its_wait_is_not_missed() {
    let store = HeldSsts::holding("/compacted/");
    let (gate, memory) = gate_reading(Some(0));
    let clock = HeldSleep::new();
    let (running, waiting, parks) =
        a_flush_waiting_behind_one_at_the_store(&store, &gate, "wake-look", Some(&clock));
    clock.hold_the_next();
    let held = put_on_a_thread(&waiting, b"k2");
    wait_until("the write between its look and its wait", || {
        clock.one_waits()
    });
    assert_eq!(parks.count(), 1);

    memory.set(Some(usize::MAX));
    wait_until("both uploads at the store", || store.held() == 2);
    clock.let_go();
    within_ten_seconds("the held write goes in", || held.is_finished());
    assert!(
        store.held() >= 2,
        "an upload ended before the write went in"
    );
    held.join().unwrap();

    store.release_all();
    assert_rows(&waiting, &[b"k1", b"k2"]);
    for be in [&running, &*waiting] {
        be.close().unwrap();
    }
}

/// A commit that does not wait for its WAL, as an upsert's is.
fn put_buffered(be: &LsmBackend, key: &[u8]) {
    let mut w = be.begin_write().unwrap();
    be.put(&mut w, Namespace::Nodes, key, b"v").unwrap();
    be.commit_buffered(w).unwrap();
}

/// A write held back for memory alone, with no flush of its collection at
/// the gate, looks again when a WAL or a memtable of its own is flushed and
/// at nothing the gate does: another collection's flush being let in does
/// not wake it, and it does not look again and again.
#[test]
fn a_write_held_for_memory_alone_is_not_woken_at_the_gate() {
    let store = HeldSsts::holding("/compacted/");
    let (gate, memory) = gate_reading(Some(0));
    let (running, waiting, _) =
        a_flush_waiting_behind_one_at_the_store(&store, &gate, "wake-full", None);
    // A writer that holds a write back at its first unflushed byte, on a
    // store of its own where its WAL is held: nothing of it is flushed.
    let wal = HeldSsts::holding_wal("no WAL is held yet");
    let mut settings = Settings::default();
    settings.flush_interval = Some(Duration::from_millis(5));
    settings.max_unflushed_bytes = 1;
    let (full, parks) = open_watched(&wal, &gate, "gate/wake-full-alone", settings, None);
    let full = Arc::new(full);
    wal.hold("/wal/");
    put_buffered(&full, b"k1");
    let held = {
        let full = Arc::clone(&full);
        std::thread::spawn(move || put_buffered(&full, b"k2"))
    };
    wait_until("the write held back for memory", || parks.count() >= 1);

    memory.set(Some(usize::MAX));
    wait_until("the other collection's flush let in", || store.held() == 2);
    assert_eq!(
        parks.count(),
        1,
        "a write held for memory alone looked again, and nothing of its own was flushed"
    );
    assert!(!held.is_finished());

    // Its own WAL flushed, it goes in.
    wal.release_all();
    joined("the write held for memory", held);
    full.flush_durable().unwrap();
    assert_rows(&full, &[b"k1", b"k2"]);
    store.release_all();
    for be in [&running, &*waiting, &*full] {
        be.close().unwrap();
    }
}
