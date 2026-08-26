//! End-to-end scaling benchmarks for reader-backed snapshots and incremental WAL replay.
//!
//! This target deliberately uses fixed repetitions instead of Criterion's adaptive
//! iteration count. Replay, checkpoint rollover, and generation cleanup mutate state,
//! and their untimed fixture setup is substantially more expensive than the operation
//! under test. Fixed repetitions keep those costs out of the samples without causing
//! Criterion to request millions of fixture rebuilds.
//!
//! Run the standard matrix:
//!   cargo bench -p slatedb --bench db_reader_scaling --features test-util
//!
//! Run the extended edge-case matrix:
//!   SLATEDB_READER_BENCH_FULL=1 cargo bench -p slatedb --bench db_reader_scaling --features test-util

use std::future::Future;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};
use slatedb::config::{
    CheckpointOptions, CheckpointScope, DbReaderOptions, FlushOptions, FlushType, PutOptions,
    Settings, WriteOptions,
};
use slatedb::instrumented_object_store_stats;
use slatedb::{Db, DbReader, DbSnapshot, PrefixExtractor, PrefixTarget};
use slatedb_common::metrics::{lookup_metric, lookup_metric_with_labels, DefaultMetricsRecorder};
use slatedb_common::{MockSystemClock, SystemClock};
use tokio::sync::Barrier;
use uuid::Uuid;

const POLL_INTERVAL: Duration = Duration::from_millis(10);
const REPLAY_SSTS: &str = "slatedb.db.reader_wal_replay_ssts";
const MANIFEST_POLLS: &str = "slatedb.db.reader_manifest_polls";

#[derive(Clone, Copy)]
enum Segmentation {
    None,
    Fixed4,
}

impl Segmentation {
    fn label(self) -> &'static str {
        match self {
            Self::None => "unsegmented",
            Self::Fixed4 => "segmented",
        }
    }
}

struct Fixed4Extractor;

impl PrefixExtractor for Fixed4Extractor {
    fn name(&self) -> &str {
        "reader-bench-fixed4"
    }

    fn prefix_len(&self, target: &PrefixTarget) -> Option<usize> {
        let len = match target {
            PrefixTarget::Point(key) | PrefixTarget::Prefix(key) => key.len(),
        };
        (len >= 4).then_some(4)
    }
}

#[derive(Debug)]
struct Summary {
    reps: usize,
    min_us: f64,
    p50_us: f64,
    p95_us: f64,
    mean_us: f64,
    max_us: f64,
}

impl Summary {
    fn from_durations(samples: Vec<Duration>) -> Self {
        assert!(!samples.is_empty());
        let mut micros = samples
            .into_iter()
            .map(|sample| sample.as_secs_f64() * 1_000_000.0)
            .collect::<Vec<_>>();
        micros.sort_by(f64::total_cmp);
        let reps = micros.len();
        let percentile = |p: f64| {
            let index = ((reps - 1) as f64 * p).round() as usize;
            micros[index]
        };
        Self {
            reps,
            min_us: micros[0],
            p50_us: percentile(0.50),
            p95_us: percentile(0.95),
            mean_us: micros.iter().sum::<f64>() / reps as f64,
            max_us: micros[reps - 1],
        }
    }
}

fn print_summary(suite: &str, case: &str, scale: usize, summary: &Summary, notes: &str) {
    println!(
        "RESULT\t{suite}\t{case}\t{scale}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{notes}",
        summary.reps,
        summary.min_us,
        summary.p50_us,
        summary.p95_us,
        summary.mean_us,
        summary.max_us,
    );
}

async fn measure_async<F, Fut>(warmup: usize, reps: usize, mut operation: F) -> Summary
where
    F: FnMut() -> Fut,
    Fut: Future<Output = ()>,
{
    for _ in 0..warmup {
        operation().await;
    }
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        operation().await;
        samples.push(start.elapsed());
    }
    Summary::from_durations(samples)
}

fn writer_settings() -> Settings {
    Settings {
        flush_interval: None,
        compactor_options: None,
        garbage_collector_options: None,
        l0_sst_size_bytes: 256 * 1024 * 1024,
        l0_max_ssts: 16_384,
        l0_max_ssts_per_key: 16_384,
        ..Settings::default()
    }
}

fn quiet_reader_options(max_memtable_bytes: u64) -> DbReaderOptions {
    DbReaderOptions {
        manifest_poll_interval: Duration::from_secs(60 * 60),
        checkpoint_lifetime: Duration::from_secs(3 * 60 * 60),
        max_memtable_bytes,
        ..DbReaderOptions::default()
    }
}

fn polling_reader_options(
    max_memtable_bytes: u64,
    checkpoint_lifetime: Duration,
) -> DbReaderOptions {
    DbReaderOptions {
        manifest_poll_interval: POLL_INTERVAL,
        checkpoint_lifetime,
        max_memtable_bytes,
        ..DbReaderOptions::default()
    }
}

fn key_for(index: usize, segmentation: Segmentation) -> Bytes {
    match segmentation {
        Segmentation::None => Bytes::from(format!("key-{index:08}")),
        Segmentation::Fixed4 => Bytes::from(format!("{index:04}-key-{index:08}")),
    }
}

async fn write_wal(db: &Db, index: usize, segmentation: Segmentation) -> Bytes {
    let key = key_for(index, segmentation);
    db.put_with_options(
        &key,
        b"value-value-value-value-value",
        &PutOptions::default(),
        &WriteOptions {
            await_durable: false,
            ..WriteOptions::default()
        },
    )
    .await
    .expect("put failed");
    db.flush_with_options(FlushOptions {
        flush_type: FlushType::Wal,
    })
    .await
    .expect("WAL flush failed");
    key
}

async fn open_db(
    path: &str,
    store: Arc<InMemory>,
    clock: Option<Arc<MockSystemClock>>,
    segmentation: Segmentation,
) -> Db {
    let mut builder = Db::builder(path, store).with_settings(writer_settings());
    if let Some(clock) = clock {
        builder = builder.with_system_clock(clock);
    }
    if matches!(segmentation, Segmentation::Fixed4) {
        builder = builder.with_segment_extractor(Arc::new(Fixed4Extractor));
    }
    builder.build().await.expect("DB open failed")
}

async fn open_reader(
    path: &str,
    store: Arc<InMemory>,
    clock: Option<Arc<MockSystemClock>>,
    segmentation: Segmentation,
    options: DbReaderOptions,
    recorder: Arc<DefaultMetricsRecorder>,
) -> DbReader {
    let mut builder = DbReader::builder(path, store)
        .with_options(options)
        .with_metrics_recorder(recorder.clone())
        .with_db_cache_disabled();
    if let Some(clock) = clock {
        builder = builder.with_system_clock(clock);
    }
    if matches!(segmentation, Segmentation::Fixed4) {
        builder = builder.with_segment_extractor(Arc::new(Fixed4Extractor));
    }
    let reader = builder.build().await.expect("reader open failed");
    // The dispatcher's first ticker fires immediately. Wait for the complete
    // startup poll, not merely its first object-store request, so the first
    // timed sample cannot race startup work.
    wait_for_counter(|| scalar(&recorder, MANIFEST_POLLS), 1).await;
    reader
}

struct ReaderFixture {
    path: String,
    store: Arc<InMemory>,
    db: Db,
    reader: Arc<DbReader>,
    recorder: Arc<DefaultMetricsRecorder>,
    clock: Option<Arc<MockSystemClock>>,
    segmentation: Segmentation,
    next_key: usize,
    latest_key: Option<Bytes>,
}

impl ReaderFixture {
    async fn wal_history(
        wal_count: usize,
        segmentation: Segmentation,
        polling: bool,
        checkpoint_lifetime: Duration,
    ) -> Self {
        let path = format!("bench/db-reader-scaling/{}", Uuid::new_v4());
        let store = Arc::new(InMemory::new());
        let clock = polling.then(|| Arc::new(MockSystemClock::new()));
        let db = open_db(&path, Arc::clone(&store), clock.clone(), segmentation).await;
        let mut latest_key = None;
        for index in 0..wal_count {
            latest_key = Some(write_wal(&db, index, segmentation).await);
        }
        let recorder = Arc::new(DefaultMetricsRecorder::new());
        let options = if polling {
            polling_reader_options(1, checkpoint_lifetime)
        } else {
            quiet_reader_options(1)
        };
        let reader = Arc::new(
            open_reader(
                &path,
                Arc::clone(&store),
                clock.clone(),
                segmentation,
                options,
                Arc::clone(&recorder),
            )
            .await,
        );
        Self {
            path,
            store,
            db,
            reader,
            recorder,
            clock,
            segmentation,
            next_key: wal_count,
            latest_key,
        }
    }

    async fn append_wals(&mut self, count: usize) {
        for _ in 0..count {
            self.latest_key = Some(write_wal(&self.db, self.next_key, self.segmentation).await);
            self.next_key += 1;
        }
    }

    async fn close(self) {
        self.reader.close().await.expect("reader close failed");
        self.db.close().await.expect("DB close failed");
    }
}

fn scalar(recorder: &DefaultMetricsRecorder, name: &str) -> u64 {
    lookup_metric(recorder, name).unwrap_or(0) as u64
}

fn object_store_requests(recorder: &DefaultMetricsRecorder, api: &'static str) -> u64 {
    lookup_metric_with_labels(
        recorder,
        instrumented_object_store_stats::REQUEST_COUNT,
        &[
            ("component", "reader"),
            ("store_type", "main"),
            ("op", if api == "put" { "put" } else { "get" }),
            ("api", api),
        ],
    )
    .unwrap_or(0) as u64
}

#[derive(Clone, Copy, Default)]
struct ReaderCounters {
    lists: u64,
    heads: u64,
    gets: u64,
    get_ranges: u64,
    puts: u64,
    replay_ssts: u64,
}

impl ReaderCounters {
    fn capture(recorder: &DefaultMetricsRecorder) -> Self {
        Self {
            lists: object_store_requests(recorder, "list"),
            heads: object_store_requests(recorder, "head"),
            gets: object_store_requests(recorder, "get"),
            get_ranges: object_store_requests(recorder, "get_range"),
            puts: object_store_requests(recorder, "put"),
            replay_ssts: scalar(recorder, REPLAY_SSTS),
        }
    }

    fn delta(self, before: Self) -> Self {
        Self {
            lists: self.lists - before.lists,
            heads: self.heads - before.heads,
            gets: self.gets - before.gets,
            get_ranges: self.get_ranges - before.get_ranges,
            puts: self.puts - before.puts,
            replay_ssts: self.replay_ssts - before.replay_ssts,
        }
    }

    fn note(self, reps: usize) -> String {
        let reps = reps as f64;
        format!(
            "list/op={:.2};head/op={:.2};get/op={:.2};range/op={:.2};put/op={:.2};replay_sst/op={:.2}",
            self.lists as f64 / reps,
            self.heads as f64 / reps,
            self.gets as f64 / reps,
            self.get_ranges as f64 / reps,
            self.puts as f64 / reps,
            self.replay_ssts as f64 / reps,
        )
    }
}

async fn wait_for_counter<F>(mut current: F, target: u64)
where
    F: FnMut() -> u64,
{
    for _ in 0..100_000 {
        if current() >= target {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!(
        "timed out waiting for reader poll completion: current={}, target={target}",
        current()
    );
}

async fn trigger_poll(fixture: &ReaderFixture, expected_new_wals: u64) {
    advance_and_wait_poll(fixture, POLL_INTERVAL, expected_new_wals).await;
}

async fn advance_and_wait_poll(fixture: &ReaderFixture, advance: Duration, expected_new_wals: u64) {
    let clock = fixture
        .clock
        .as_ref()
        .expect("polling fixture has no clock");
    let completed_poll_target = scalar(&fixture.recorder, MANIFEST_POLLS) + 1;
    let replay_target = scalar(&fixture.recorder, REPLAY_SSTS) + expected_new_wals;
    clock.advance(advance).await;
    wait_for_counter(
        || scalar(&fixture.recorder, MANIFEST_POLLS),
        completed_poll_target,
    )
    .await;
    if expected_new_wals > 0 {
        wait_for_counter(|| scalar(&fixture.recorder, REPLAY_SSTS), replay_target).await;
    }
}

async fn benchmark_usual_paths(scales: &[usize], full: bool) {
    println!("SECTION\tusual_paths");
    for &wal_count in scales {
        let fixture = ReaderFixture::wal_history(
            wal_count,
            Segmentation::None,
            false,
            Duration::from_secs(60),
        )
        .await;
        let snapshot = fixture.reader.snapshot().await.expect("snapshot failed");
        let latest = fixture
            .latest_key
            .clone()
            .unwrap_or_else(|| Bytes::from_static(b"missing"));
        let snapshot_reps = if full { 20_000 } else { 5_000 };
        let read_reps = if full { 2_000 } else { 500 };
        let scan_reps = if full { 250 } else { 75 };

        let summary = measure_async(200, snapshot_reps, || async {
            let snapshot = fixture.reader.snapshot().await.expect("snapshot failed");
            black_box(snapshot.seq());
        })
        .await;
        print_summary("usual", "snapshot_create_drop", wal_count, &summary, "");

        let summary = measure_async(30, read_reps, || async {
            black_box(fixture.reader.get(&latest).await.expect("get failed"));
        })
        .await;
        print_summary("usual", "reader_get_latest", wal_count, &summary, "");

        let summary = measure_async(30, read_reps, || async {
            black_box(
                fixture
                    .reader
                    .get(b"definitely-missing")
                    .await
                    .expect("missing get failed"),
            );
        })
        .await;
        print_summary("usual", "reader_get_missing", wal_count, &summary, "");

        let summary = measure_async(30, read_reps, || async {
            black_box(snapshot.get(&latest).await.expect("snapshot get failed"));
        })
        .await;
        print_summary("usual", "snapshot_get_latest", wal_count, &summary, "");

        let summary = measure_async(10, scan_reps, || async {
            let mut iter = fixture.reader.scan(..).await.expect("scan failed");
            black_box(iter.next().await.expect("scan next failed"));
        })
        .await;
        print_summary("usual", "reader_scan_first", wal_count, &summary, "");

        let summary = measure_async(10, scan_reps, || async {
            let mut iter = snapshot.scan(..).await.expect("snapshot scan failed");
            black_box(iter.next().await.expect("snapshot scan next failed"));
        })
        .await;
        print_summary("usual", "snapshot_scan_first", wal_count, &summary, "");

        let full_scan_reps = if wal_count <= 32 { 50 } else { 15 };
        let summary = measure_async(3, full_scan_reps, || async {
            let mut iter = fixture.reader.scan(..).await.expect("scan failed");
            let mut count = 0usize;
            while iter.next().await.expect("scan next failed").is_some() {
                count += 1;
            }
            black_box(count);
        })
        .await;
        print_summary("usual", "reader_scan_full", wal_count, &summary, "");

        drop(snapshot);
        fixture.close().await;
    }
}

async fn benchmark_bulk_iterator(full: bool) {
    println!("SECTION\tbulk_iterator");
    let row_count = if full { 20_000 } else { 5_000 };
    let path = format!("bench/db-reader-bulk/{}", Uuid::new_v4());
    let store = Arc::new(InMemory::new());
    let db = open_db(&path, Arc::clone(&store), None, Segmentation::None).await;
    for index in 0..row_count {
        let key = key_for(index, Segmentation::None);
        db.put_with_options(
            &key,
            b"value",
            &PutOptions::default(),
            &WriteOptions {
                await_durable: false,
                ..WriteOptions::default()
            },
        )
        .await
        .expect("put failed");
    }
    db.flush_with_options(FlushOptions {
        flush_type: FlushType::Wal,
    })
    .await
    .expect("WAL flush failed");
    let recorder = Arc::new(DefaultMetricsRecorder::new());
    let reader = open_reader(
        &path,
        Arc::clone(&store),
        None,
        Segmentation::None,
        quiet_reader_options(u64::MAX),
        recorder,
    )
    .await;

    let summary = measure_async(2, if full { 12 } else { 5 }, || async {
        let mut iter = db.scan(..).await.expect("DB scan failed");
        let mut count = 0usize;
        while iter.next().await.expect("DB scan next failed").is_some() {
            count += 1;
        }
        assert_eq!(count, row_count);
    })
    .await;
    print_summary("bulk", "db_scan_one_memtable", row_count, &summary, "");

    let summary = measure_async(2, if full { 12 } else { 5 }, || async {
        let mut iter = reader.scan(..).await.expect("reader scan failed");
        let mut count = 0usize;
        while iter
            .next()
            .await
            .expect("reader scan next failed")
            .is_some()
        {
            count += 1;
        }
        assert_eq!(count, row_count);
    })
    .await;
    print_summary(
        "bulk",
        "reader_scan_one_replay_memtable",
        row_count,
        &summary,
        "",
    );

    let snapshot = reader.snapshot().await.expect("snapshot failed");
    let summary = measure_async(2, if full { 12 } else { 5 }, || async {
        let mut iter = snapshot.scan(..).await.expect("snapshot scan failed");
        let mut count = 0usize;
        while iter
            .next()
            .await
            .expect("snapshot scan next failed")
            .is_some()
        {
            count += 1;
        }
        assert_eq!(count, row_count);
    })
    .await;
    print_summary(
        "bulk",
        "snapshot_scan_one_replay_memtable",
        row_count,
        &summary,
        "",
    );
    drop(snapshot);
    reader.close().await.expect("reader close failed");
    db.close().await.expect("DB close failed");
}

async fn benchmark_iterator_contention(full: bool) {
    println!("SECTION\titerator_contention");
    let row_count = if full { 5_000 } else { 2_000 };
    let path = format!("bench/db-reader-iterator-contention/{}", Uuid::new_v4());
    let store = Arc::new(InMemory::new());
    let db = Arc::new(open_db(&path, Arc::clone(&store), None, Segmentation::None).await);
    for index in 0..row_count {
        db.put_with_options(
            &key_for(index, Segmentation::None),
            b"value",
            &PutOptions::default(),
            &WriteOptions {
                await_durable: false,
                ..WriteOptions::default()
            },
        )
        .await
        .expect("put failed");
    }
    db.flush_with_options(FlushOptions {
        flush_type: FlushType::Wal,
    })
    .await
    .expect("WAL flush failed");
    let reader = Arc::new(
        open_reader(
            &path,
            Arc::clone(&store),
            None,
            Segmentation::None,
            quiet_reader_options(u64::MAX),
            Arc::new(DefaultMetricsRecorder::new()),
        )
        .await,
    );

    // Populate both read paths' caches before comparing their concurrent scans.
    let mut db_warmup = db.scan(..).await.expect("DB warmup scan failed");
    while db_warmup
        .next()
        .await
        .expect("DB warmup next failed")
        .is_some()
    {}
    let mut reader_warmup = reader.scan(..).await.expect("reader warmup scan failed");
    while reader_warmup
        .next()
        .await
        .expect("reader warmup next failed")
        .is_some()
    {}

    let task_counts = if full {
        vec![1, 2, 4, 8, 16, 32]
    } else {
        vec![1, 4, 16]
    };
    for task_count in task_counts {
        let barrier = Arc::new(Barrier::new(task_count + 1));
        let mut tasks = Vec::with_capacity(task_count);
        for _ in 0..task_count {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            tasks.push(tokio::spawn(async move {
                barrier.wait().await;
                let mut iter = db.scan(..).await.expect("DB scan failed");
                let mut count = 0usize;
                while iter.next().await.expect("DB scan next failed").is_some() {
                    count += 1;
                }
                count
            }));
        }
        barrier.wait().await;
        let start = Instant::now();
        for task in tasks {
            assert_eq!(task.await.expect("DB scan task failed"), row_count);
        }
        let elapsed = start.elapsed();
        let total_rows = task_count * row_count;
        let summary = Summary::from_durations(vec![elapsed.div_f64(total_rows as f64)]);
        print_summary(
            "contention",
            "db_scan_per_row",
            task_count,
            &summary,
            &format!(
                "throughput_rows_s={:.0};total_rows={total_rows}",
                total_rows as f64 / elapsed.as_secs_f64()
            ),
        );

        let barrier = Arc::new(Barrier::new(task_count + 1));
        let mut tasks = Vec::with_capacity(task_count);
        for _ in 0..task_count {
            let reader = Arc::clone(&reader);
            let barrier = Arc::clone(&barrier);
            tasks.push(tokio::spawn(async move {
                barrier.wait().await;
                let mut iter = reader.scan(..).await.expect("reader scan failed");
                let mut count = 0usize;
                while iter
                    .next()
                    .await
                    .expect("reader scan next failed")
                    .is_some()
                {
                    count += 1;
                }
                count
            }));
        }
        barrier.wait().await;
        let start = Instant::now();
        for task in tasks {
            assert_eq!(task.await.expect("reader scan task failed"), row_count);
        }
        let elapsed = start.elapsed();
        let total_rows = task_count * row_count;
        let summary = Summary::from_durations(vec![elapsed.div_f64(total_rows as f64)]);
        print_summary(
            "contention",
            "reader_scan_per_row",
            task_count,
            &summary,
            &format!(
                "throughput_rows_s={:.0};total_rows={total_rows}",
                total_rows as f64 / elapsed.as_secs_f64()
            ),
        );
    }

    reader.close().await.expect("reader close failed");
    db.close().await.expect("DB close failed");
}

async fn benchmark_incremental_replay(histories: &[usize], deltas: &[usize], full: bool) {
    println!("SECTION\tincremental_replay");
    for segmentation in [Segmentation::None, Segmentation::Fixed4] {
        for &history in histories {
            for &delta in deltas {
                let reps = if full { 12 } else { 5 };
                let mut samples = Vec::with_capacity(reps);
                let mut counter_total = ReaderCounters::default();
                for _ in 0..reps {
                    // Rebuild outside the timed interval so every sample starts
                    // from exactly `history`, rather than accumulating each
                    // preceding sample's delta into its advertised baseline.
                    let mut fixture = ReaderFixture::wal_history(
                        history,
                        segmentation,
                        true,
                        Duration::from_secs(60),
                    )
                    .await;
                    fixture.append_wals(delta).await;
                    let before = ReaderCounters::capture(&fixture.recorder);
                    let start = Instant::now();
                    trigger_poll(&fixture, delta as u64).await;
                    samples.push(start.elapsed());
                    let counters = ReaderCounters::capture(&fixture.recorder).delta(before);
                    counter_total.lists += counters.lists;
                    counter_total.heads += counters.heads;
                    counter_total.gets += counters.gets;
                    counter_total.get_ranges += counters.get_ranges;
                    counter_total.puts += counters.puts;
                    counter_total.replay_ssts += counters.replay_ssts;
                    fixture.close().await;
                }
                let summary = Summary::from_durations(samples);
                let notes = format!("mode={};{}", segmentation.label(), counter_total.note(reps));
                print_summary(
                    "replay",
                    "poll_delta",
                    history * 10_000 + delta,
                    &summary,
                    &notes,
                );
            }
        }
    }
}

async fn benchmark_generation_rollover(scales: &[usize], full: bool) {
    println!("SECTION\tgeneration_rollover");
    for &history in scales.iter().filter(|&&value| value > 0) {
        let reps = if full && history <= 32 { 5 } else { 2 };
        let mut samples = Vec::with_capacity(reps);
        let mut counter_total = ReaderCounters {
            lists: 0,
            heads: 0,
            gets: 0,
            get_ranges: 0,
            puts: 0,
            replay_ssts: 0,
        };
        for _ in 0..reps {
            let fixture = ReaderFixture::wal_history(
                history,
                Segmentation::None,
                true,
                Duration::from_secs(60),
            )
            .await;
            let before_manifest = fixture.reader.manifest().id();
            fixture
                .db
                .flush_with_options(FlushOptions {
                    flush_type: FlushType::MemTable,
                })
                .await
                .expect("L0 flush failed");
            let before = ReaderCounters::capture(&fixture.recorder);
            let start = Instant::now();
            advance_and_wait_poll(&fixture, POLL_INTERVAL, 0).await;
            assert!(fixture.reader.manifest().id() > before_manifest);
            samples.push(start.elapsed());
            let delta = ReaderCounters::capture(&fixture.recorder).delta(before);
            counter_total.lists += delta.lists;
            counter_total.heads += delta.heads;
            counter_total.gets += delta.gets;
            counter_total.get_ranges += delta.get_ranges;
            counter_total.puts += delta.puts;
            counter_total.replay_ssts += delta.replay_ssts;
            fixture.close().await;
        }
        let summary = Summary::from_durations(samples);
        print_summary(
            "rollover",
            "flush_all_replayed_wals_to_l0",
            history,
            &summary,
            &counter_total.note(reps),
        );
    }
}

async fn benchmark_final_old_snapshot_drop(scales: &[usize], full: bool) {
    println!("SECTION\tfinal_old_snapshot_drop");
    for &history in scales.iter().filter(|&&value| value > 0) {
        let reps = if full && history <= 32 { 8 } else { 3 };
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let fixture = ReaderFixture::wal_history(
                history,
                Segmentation::None,
                true,
                Duration::from_secs(60),
            )
            .await;
            let old_snapshot = fixture.reader.snapshot().await.expect("snapshot failed");
            let before_manifest = fixture.reader.manifest().id();
            fixture
                .db
                .flush_with_options(FlushOptions {
                    flush_type: FlushType::MemTable,
                })
                .await
                .expect("L0 flush failed");
            advance_and_wait_poll(&fixture, POLL_INTERVAL, 0).await;
            assert!(fixture.reader.manifest().id() > before_manifest);

            // The reader has moved to an L0-backed generation, so this is the
            // last strong owner of the old persistent replay-memtable chain.
            let start = Instant::now();
            drop(old_snapshot);
            samples.push(start.elapsed());
            fixture.close().await;
        }
        let summary = Summary::from_durations(samples);
        print_summary(
            "cleanup",
            "drop_final_old_snapshot",
            history,
            &summary,
            "old_generation_replay_memtables",
        );
    }
}

async fn benchmark_snapshot_contention(full: bool) {
    println!("SECTION\tsnapshot_contention");
    let fixture =
        ReaderFixture::wal_history(128, Segmentation::None, false, Duration::from_secs(60)).await;
    let per_task = if full { 25_000 } else { 5_000 };
    let task_counts = if full {
        vec![1, 2, 4, 8, 16, 32]
    } else {
        vec![1, 4, 16]
    };
    for task_count in task_counts {
        let barrier = Arc::new(Barrier::new(task_count + 1));
        let mut tasks = Vec::with_capacity(task_count);
        for _ in 0..task_count {
            let reader = Arc::clone(&fixture.reader);
            let barrier = Arc::clone(&barrier);
            tasks.push(tokio::spawn(async move {
                barrier.wait().await;
                for _ in 0..per_task {
                    let snapshot = reader.snapshot().await.expect("snapshot failed");
                    black_box(snapshot.seq());
                }
            }));
        }
        barrier.wait().await;
        let start = Instant::now();
        for task in tasks {
            task.await.expect("snapshot task failed");
        }
        let elapsed = start.elapsed();
        let operations = task_count * per_task;
        let per_operation = elapsed.div_f64(operations as f64);
        let summary = Summary::from_durations(vec![per_operation]);
        let throughput = operations as f64 / elapsed.as_secs_f64();
        print_summary(
            "contention",
            "snapshot_create_drop",
            task_count,
            &summary,
            &format!("throughput_ops_s={throughput:.0};total_ops={operations}"),
        );
    }
    fixture.close().await;
}

async fn benchmark_manifest_file_listing(scales: &[usize], full: bool) {
    println!("SECTION\tmanifest_file_listing");
    for &file_count in scales {
        let fixture =
            ReaderFixture::wal_history(0, Segmentation::None, true, Duration::from_secs(60)).await;
        let source_id = fixture.reader.manifest().id();
        // Opening a managed reader creates a checkpoint and therefore advances
        // the manifest ID. Small requested scales can be below that fixture
        // baseline, so report and populate the actual retained-file count.
        let target_file_count = file_count.max(source_id as usize);
        let source = Path::from(format!(
            "{}/manifest/{source_id:020}.manifest",
            fixture.path
        ));
        for id in (source_id + 1)..=target_file_count as u64 {
            let destination = Path::from(format!("{}/manifest/{id:020}.manifest", fixture.path));
            fixture
                .store
                .copy(&source, &destination)
                .await
                .expect("manifest copy failed");
        }
        let reps = if full { 25 } else { 8 };
        for _ in 0..2 {
            trigger_poll(&fixture, 0).await;
        }
        let before = ReaderCounters::capture(&fixture.recorder);
        let summary = measure_async(0, reps, || async {
            trigger_poll(&fixture, 0).await;
        })
        .await;
        let counters = ReaderCounters::capture(&fixture.recorder).delta(before);
        print_summary(
            "manifest",
            "no_change_poll_by_retained_files",
            target_file_count,
            &summary,
            &counters.note(reps),
        );
        fixture.close().await;
    }
}

async fn benchmark_checkpoint_heavy_manifest(scales: &[usize], full: bool) {
    println!("SECTION\tcheckpoint_heavy_manifest");
    for &checkpoint_count in scales {
        let path = format!("bench/db-reader-checkpoints/{}", Uuid::new_v4());
        let store = Arc::new(InMemory::new());
        let clock = Arc::new(MockSystemClock::new());
        let db = open_db(
            &path,
            Arc::clone(&store),
            Some(Arc::clone(&clock)),
            Segmentation::None,
        )
        .await;
        for _ in 0..checkpoint_count {
            db.create_checkpoint(CheckpointScope::Durable, &CheckpointOptions::default())
                .await
                .expect("user checkpoint creation failed");
        }
        let recorder = Arc::new(DefaultMetricsRecorder::new());
        let reader = open_reader(
            &path,
            Arc::clone(&store),
            Some(Arc::clone(&clock)),
            Segmentation::None,
            polling_reader_options(1, Duration::from_secs(1)),
            Arc::clone(&recorder),
        )
        .await;
        let fixture = ReaderFixture {
            path,
            store,
            db,
            reader: Arc::new(reader),
            recorder,
            clock: Some(clock),
            segmentation: Segmentation::None,
            next_key: 0,
            latest_key: None,
        };

        let reps = if full { 15 } else { 5 };
        for _ in 0..2 {
            trigger_poll(&fixture, 0).await;
        }
        let before = ReaderCounters::capture(&fixture.recorder);
        let summary = measure_async(0, reps, || async {
            trigger_poll(&fixture, 0).await;
        })
        .await;
        let counters = ReaderCounters::capture(&fixture.recorder).delta(before);
        print_summary(
            "manifest",
            "no_change_poll_by_checkpoint_count",
            checkpoint_count,
            &summary,
            &counters.note(reps),
        );

        let refresh_reps = if full { 7 } else { 3 };
        let before = ReaderCounters::capture(&fixture.recorder);
        let mut samples = Vec::with_capacity(refresh_reps);
        for _ in 0..refresh_reps {
            let start = Instant::now();
            advance_and_wait_poll(&fixture, Duration::from_millis(501), 0).await;
            samples.push(start.elapsed());
        }
        let counters = ReaderCounters::capture(&fixture.recorder).delta(before);
        let summary = Summary::from_durations(samples);
        print_summary(
            "manifest",
            "refresh_one_managed_checkpoint",
            checkpoint_count,
            &summary,
            &counters.note(refresh_reps),
        );
        fixture.close().await;
    }
}

async fn build_pinned_generations(
    generation_count: usize,
) -> (ReaderFixture, Vec<Arc<DbSnapshot>>) {
    let fixture =
        ReaderFixture::wal_history(0, Segmentation::None, true, Duration::from_secs(60)).await;
    let mut snapshots = vec![fixture.reader.snapshot().await.expect("snapshot failed")];
    for index in 1..generation_count {
        let key = key_for(index, Segmentation::None);
        fixture
            .db
            .put_with_options(
                &key,
                b"value",
                &PutOptions::default(),
                &WriteOptions {
                    await_durable: false,
                    ..WriteOptions::default()
                },
            )
            .await
            .expect("put failed");
        fixture
            .db
            .flush_with_options(FlushOptions {
                flush_type: FlushType::MemTable,
            })
            .await
            .expect("L0 flush failed");
        let old_manifest = fixture.reader.manifest().id();
        advance_and_wait_poll(&fixture, POLL_INTERVAL, 0).await;
        assert!(fixture.reader.manifest().id() > old_manifest);
        snapshots.push(fixture.reader.snapshot().await.expect("snapshot failed"));
    }
    (fixture, snapshots)
}

async fn benchmark_pinned_generations(scales: &[usize], full: bool) {
    println!("SECTION\tpinned_generations");
    for &generation_count in scales {
        let (fixture, snapshots) = build_pinned_generations(generation_count).await;
        let poll_reps = if full { 15 } else { 5 };
        for _ in 0..2 {
            trigger_poll(&fixture, 0).await;
        }
        let before = ReaderCounters::capture(&fixture.recorder);
        let summary = measure_async(0, poll_reps, || async {
            trigger_poll(&fixture, 0).await;
        })
        .await;
        let counters = ReaderCounters::capture(&fixture.recorder).delta(before);
        print_summary(
            "generation",
            "no_change_poll_while_pinned",
            generation_count,
            &summary,
            &counters.note(poll_reps),
        );

        let refresh_reps = if full { 5 } else { 2 };
        let before = ReaderCounters::capture(&fixture.recorder);
        let mut refresh_samples = Vec::with_capacity(refresh_reps);
        for _ in 0..refresh_reps {
            let start = Instant::now();
            advance_and_wait_poll(&fixture, Duration::from_secs(31), 0).await;
            refresh_samples.push(start.elapsed());
        }
        let counters = ReaderCounters::capture(&fixture.recorder).delta(before);
        let summary = Summary::from_durations(refresh_samples);
        print_summary(
            "generation",
            "refresh_all_pinned",
            generation_count,
            &summary,
            &counters.note(refresh_reps),
        );

        drop(snapshots);
        let before = ReaderCounters::capture(&fixture.recorder);
        let start = Instant::now();
        trigger_poll(&fixture, 0).await;
        let summary = Summary::from_durations(vec![start.elapsed()]);
        let counters = ReaderCounters::capture(&fixture.recorder).delta(before);
        print_summary(
            "generation",
            "delete_released",
            generation_count,
            &summary,
            &counters.note(1),
        );
        fixture.close().await;
    }
}

async fn benchmark_fixed_reader_open(scales: &[usize], full: bool) {
    println!("SECTION\tfixed_reader_open");
    for &wal_count in scales {
        let path = format!("bench/db-reader-fixed/{}", Uuid::new_v4());
        let store = Arc::new(InMemory::new());
        let db = open_db(&path, Arc::clone(&store), None, Segmentation::None).await;
        for index in 0..wal_count {
            write_wal(&db, index, Segmentation::None).await;
        }
        let checkpoint = db
            .create_checkpoint(CheckpointScope::Durable, &CheckpointOptions::default())
            .await
            .expect("checkpoint creation failed");
        let reps = if full { 10 } else { 3 };
        let mut samples = Vec::with_capacity(reps);
        let mut counter_total = ReaderCounters {
            lists: 0,
            heads: 0,
            gets: 0,
            get_ranges: 0,
            puts: 0,
            replay_ssts: 0,
        };
        for _ in 0..reps {
            let recorder = Arc::new(DefaultMetricsRecorder::new());
            let before = ReaderCounters::capture(&recorder);
            let object_store: Arc<dyn ObjectStore> = store.clone();
            let start = Instant::now();
            let reader = DbReader::builder(path.as_str(), object_store)
                .with_checkpoint_id(checkpoint.id)
                .with_options(quiet_reader_options(1))
                .with_metrics_recorder(recorder.clone())
                .with_db_cache_disabled()
                .build()
                .await
                .expect("fixed reader open failed");
            samples.push(start.elapsed());
            let delta = ReaderCounters::capture(&recorder).delta(before);
            counter_total.lists += delta.lists;
            counter_total.heads += delta.heads;
            counter_total.gets += delta.gets;
            counter_total.get_ranges += delta.get_ranges;
            counter_total.puts += delta.puts;
            counter_total.replay_ssts += delta.replay_ssts;
            reader.close().await.expect("fixed reader close failed");
        }
        let summary = Summary::from_durations(samples);
        print_summary(
            "open",
            "fixed_checkpoint_reader",
            wal_count,
            &summary,
            &counter_total.note(reps),
        );
        db.close().await.expect("DB close failed");
    }
}

async fn run() {
    let full = std::env::var_os("SLATEDB_READER_BENCH_FULL").is_some();
    println!(
        "CONFIG\tprofile={}\tfull={full}",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    println!("HEADER\tsuite\tcase\tscale\treps\tmin_us\tp50_us\tp95_us\tmean_us\tmax_us\tnotes");

    let usual_scales = if full {
        vec![0, 1, 8, 32, 128, 512]
    } else {
        vec![0, 8, 128]
    };
    benchmark_usual_paths(&usual_scales, full).await;
    benchmark_bulk_iterator(full).await;
    benchmark_iterator_contention(full).await;

    let replay_histories = if full {
        vec![0, 32, 128, 512]
    } else {
        vec![0, 128]
    };
    let replay_deltas = if full { vec![1, 8, 32] } else { vec![1, 8] };
    benchmark_incremental_replay(&replay_histories, &replay_deltas, full).await;
    benchmark_generation_rollover(&usual_scales, full).await;
    benchmark_final_old_snapshot_drop(&usual_scales, full).await;
    benchmark_snapshot_contention(full).await;

    let manifest_files = if full {
        vec![2, 16, 64, 256, 1_024, 4_096]
    } else {
        vec![2, 64, 1_024]
    };
    benchmark_manifest_file_listing(&manifest_files, full).await;

    let checkpoint_counts = if full {
        vec![0, 8, 32, 128]
    } else {
        vec![0, 32]
    };
    benchmark_checkpoint_heavy_manifest(&checkpoint_counts, full).await;

    let generation_counts = if full {
        vec![1, 8, 32, 128]
    } else {
        vec![1, 8]
    };
    benchmark_pinned_generations(&generation_counts, full).await;

    let fixed_open_scales = if full {
        vec![0, 1, 8, 32, 128]
    } else {
        vec![0, 8, 32]
    };
    benchmark_fixed_reader_open(&fixed_open_scales, full).await;
}

fn main() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build Tokio runtime");
    runtime.block_on(run());
}
