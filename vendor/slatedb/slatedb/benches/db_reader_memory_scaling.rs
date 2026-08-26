//! Heap-memory scaling benchmarks for DbReader snapshots and incremental WAL replay.
//!
//! This is separate from `db_reader_scaling` because its global allocator performs
//! atomic accounting on every allocation. Keeping it isolated prevents memory
//! instrumentation from distorting the latency benchmark.
//!
//! Run the standard matrix:
//!   cargo bench -p slatedb --bench db_reader_memory_scaling --features test-util
//!
//! Run the extended matrix:
//!   SLATEDB_READER_BENCH_FULL=1 cargo bench -p slatedb --bench db_reader_memory_scaling --features test-util

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use object_store::memory::InMemory;
use slatedb::config::{
    DbReaderOptions, FlushOptions, FlushType, PutOptions, Settings, WriteOptions,
};
use slatedb::{Db, DbReader, DbSnapshot, PrefixExtractor, PrefixTarget};
use slatedb_common::metrics::{lookup_metric, DefaultMetricsRecorder};
use slatedb_common::{MockSystemClock, SystemClock};
use uuid::Uuid;

const POLL_INTERVAL: Duration = Duration::from_millis(10);
const MANIFEST_POLLS: &str = "slatedb.db.reader_manifest_polls";
const REPLAY_SSTS: &str = "slatedb.db.reader_wal_replay_ssts";

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static ALLOCATION_CALLS: AtomicUsize = AtomicUsize::new(0);

struct TrackingAllocator;

fn record_allocation(size: usize) {
    ALLOCATED_BYTES.fetch_add(size, Ordering::Relaxed);
    ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
    let live = LIVE_BYTES.fetch_add(size, Ordering::Relaxed) + size;
    PEAK_LIVE_BYTES.fetch_max(live, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Delegates the exact layout to the system allocator.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Delegates the exact layout to the system allocator.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: `pointer` was allocated with this allocator and `layout` is unchanged.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, old: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: Delegates the original pointer/layout and requested size unchanged.
        let new_pointer = unsafe { System.realloc(pointer, old, new_size) };
        if !new_pointer.is_null() {
            ALLOCATED_BYTES.fetch_add(new_size, Ordering::Relaxed);
            ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
            let live = if new_size >= old.size() {
                LIVE_BYTES.fetch_add(new_size - old.size(), Ordering::Relaxed) + new_size
                    - old.size()
            } else {
                LIVE_BYTES.fetch_sub(old.size() - new_size, Ordering::Relaxed)
                    - (old.size() - new_size)
            };
            PEAK_LIVE_BYTES.fetch_max(live, Ordering::Relaxed);
        }
        new_pointer
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: TrackingAllocator = TrackingAllocator;

#[derive(Clone, Copy, Debug)]
struct MemoryWindow {
    live_before: usize,
    allocated_before: usize,
    calls_before: usize,
}

#[derive(Clone, Copy, Debug)]
struct MemorySample {
    retained_bytes: i64,
    peak_extra_bytes: usize,
    allocated_bytes: usize,
    allocation_calls: usize,
}

impl MemoryWindow {
    fn start() -> Self {
        let live_before = LIVE_BYTES.load(Ordering::SeqCst);
        PEAK_LIVE_BYTES.store(live_before, Ordering::SeqCst);
        Self {
            live_before,
            allocated_before: ALLOCATED_BYTES.load(Ordering::SeqCst),
            calls_before: ALLOCATION_CALLS.load(Ordering::SeqCst),
        }
    }

    fn finish(self) -> MemorySample {
        let live_after = LIVE_BYTES.load(Ordering::SeqCst);
        MemorySample {
            retained_bytes: live_after as i64 - self.live_before as i64,
            peak_extra_bytes: PEAK_LIVE_BYTES
                .load(Ordering::SeqCst)
                .saturating_sub(self.live_before),
            allocated_bytes: ALLOCATED_BYTES
                .load(Ordering::SeqCst)
                .saturating_sub(self.allocated_before),
            allocation_calls: ALLOCATION_CALLS
                .load(Ordering::SeqCst)
                .saturating_sub(self.calls_before),
        }
    }
}

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
        "reader-memory-bench-fixed4"
    }

    fn prefix_len(&self, target: &PrefixTarget) -> Option<usize> {
        let len = match target {
            PrefixTarget::Point(key) | PrefixTarget::Prefix(key) => key.len(),
        };
        (len >= 4).then_some(4)
    }
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

fn reader_options(max_memtable_bytes: u64) -> DbReaderOptions {
    DbReaderOptions {
        manifest_poll_interval: POLL_INTERVAL,
        checkpoint_lifetime: Duration::from_secs(60),
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

async fn open_db(
    path: &str,
    store: Arc<InMemory>,
    clock: Arc<MockSystemClock>,
    segmentation: Segmentation,
) -> Db {
    let mut builder = Db::builder(path, store)
        .with_settings(writer_settings())
        .with_system_clock(clock);
    if matches!(segmentation, Segmentation::Fixed4) {
        builder = builder.with_segment_extractor(Arc::new(Fixed4Extractor));
    }
    builder.build().await.expect("DB open failed")
}

async fn open_reader(
    path: &str,
    store: Arc<InMemory>,
    clock: Arc<MockSystemClock>,
    segmentation: Segmentation,
    recorder: Arc<DefaultMetricsRecorder>,
) -> DbReader {
    let mut builder = DbReader::builder(path, store)
        .with_options(reader_options(1))
        .with_system_clock(clock)
        .with_metrics_recorder(recorder.clone())
        .with_db_cache_disabled();
    if matches!(segmentation, Segmentation::Fixed4) {
        builder = builder.with_segment_extractor(Arc::new(Fixed4Extractor));
    }
    let reader = builder.build().await.expect("reader open failed");
    wait_for_counter(|| scalar(&recorder, MANIFEST_POLLS), 1).await;
    reader
}

async fn write_wal(db: &Db, index: usize, segmentation: Segmentation) {
    db.put_with_options(
        &key_for(index, segmentation),
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
}

fn scalar(recorder: &DefaultMetricsRecorder, name: &str) -> u64 {
    lookup_metric(recorder, name).unwrap_or(0) as u64
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
        "timed out waiting for reader poll: current={}, target={target}",
        current()
    );
}

async fn settle() {
    for _ in 0..100 {
        tokio::task::yield_now().await;
    }
}

struct ReaderFixture {
    db: Db,
    reader: Arc<DbReader>,
    recorder: Arc<DefaultMetricsRecorder>,
    clock: Arc<MockSystemClock>,
    segmentation: Segmentation,
    next_key: usize,
}

impl ReaderFixture {
    async fn new(history: usize, segmentation: Segmentation) -> Self {
        let path = format!("bench/db-reader-memory/{}", Uuid::new_v4());
        let store = Arc::new(InMemory::new());
        let clock = Arc::new(MockSystemClock::new());
        let db = open_db(&path, Arc::clone(&store), Arc::clone(&clock), segmentation).await;
        for index in 0..history {
            write_wal(&db, index, segmentation).await;
        }
        let recorder = Arc::new(DefaultMetricsRecorder::new());
        let reader = Arc::new(
            open_reader(
                &path,
                store,
                Arc::clone(&clock),
                segmentation,
                Arc::clone(&recorder),
            )
            .await,
        );
        Self {
            db,
            reader,
            recorder,
            clock,
            segmentation,
            next_key: history,
        }
    }

    async fn append_wals(&mut self, count: usize) {
        for _ in 0..count {
            write_wal(&self.db, self.next_key, self.segmentation).await;
            self.next_key += 1;
        }
    }

    async fn poll(&self, expected_new_wals: u64) {
        let poll_target = scalar(&self.recorder, MANIFEST_POLLS) + 1;
        let replay_target = scalar(&self.recorder, REPLAY_SSTS) + expected_new_wals;
        self.clock.advance(POLL_INTERVAL).await;
        wait_for_counter(|| scalar(&self.recorder, MANIFEST_POLLS), poll_target).await;
        if expected_new_wals > 0 {
            wait_for_counter(|| scalar(&self.recorder, REPLAY_SSTS), replay_target).await;
        }
    }

    async fn close(self) {
        self.reader.close().await.expect("reader close failed");
        self.db.close().await.expect("DB close failed");
    }
}

fn percentile_i64(samples: &[MemorySample], field: impl Fn(&MemorySample) -> i64) -> i64 {
    let mut values = samples.iter().map(field).collect::<Vec<_>>();
    values.sort_unstable();
    values[(values.len() - 1) / 2]
}

fn percentile_usize(samples: &[MemorySample], field: impl Fn(&MemorySample) -> usize) -> usize {
    let mut values = samples.iter().map(field).collect::<Vec<_>>();
    values.sort_unstable();
    values[(values.len() - 1) / 2]
}

fn print_samples(
    suite: &str,
    case: &str,
    history: usize,
    delta: usize,
    samples: &[MemorySample],
    notes: &str,
) {
    println!(
        "MEMORY\t{suite}\t{case}\t{history}\t{delta}\t{}\t{}\t{}\t{}\t{}\t{notes}",
        samples.len(),
        percentile_i64(samples, |sample| sample.retained_bytes),
        percentile_usize(samples, |sample| sample.peak_extra_bytes),
        percentile_usize(samples, |sample| sample.allocated_bytes),
        percentile_usize(samples, |sample| sample.allocation_calls),
    );
}

async fn benchmark_incremental_replay_memory(full: bool) {
    println!("SECTION\tincremental_replay_memory");
    let histories = if full {
        vec![0, 32, 128, 512]
    } else {
        vec![0, 128]
    };
    let deltas = if full { vec![1, 8, 32] } else { vec![1, 8] };
    let reps = if full { 9 } else { 5 };
    for segmentation in [Segmentation::None, Segmentation::Fixed4] {
        for &history in &histories {
            for &delta in &deltas {
                let mut samples = Vec::with_capacity(reps);
                for _ in 0..reps {
                    let mut fixture = ReaderFixture::new(history, segmentation).await;
                    fixture.append_wals(delta).await;
                    settle().await;
                    let window = MemoryWindow::start();
                    fixture.poll(delta as u64).await;
                    settle().await;
                    samples.push(window.finish());
                    fixture.close().await;
                }
                print_samples(
                    "replay",
                    "poll_delta",
                    history,
                    delta,
                    &samples,
                    segmentation.label(),
                );
            }
        }
    }
}

async fn benchmark_snapshot_memory(full: bool) {
    println!("SECTION\tsnapshot_memory");
    let histories = if full {
        vec![0, 128, 512]
    } else {
        vec![0, 128]
    };
    let snapshot_count = if full { 10_000 } else { 2_000 };
    for history in histories {
        let fixture = ReaderFixture::new(history, Segmentation::None).await;
        let mut snapshots: Vec<Arc<DbSnapshot>> = Vec::with_capacity(snapshot_count);
        settle().await;
        let window = MemoryWindow::start();
        for _ in 0..snapshot_count {
            snapshots.push(
                fixture
                    .reader
                    .snapshot()
                    .await
                    .expect("snapshot creation failed"),
            );
        }
        black_box(&snapshots);
        settle().await;
        let held = window.finish();
        let live_before_drop = LIVE_BYTES.load(Ordering::SeqCst);
        snapshots.clear();
        settle().await;
        let released = live_before_drop.saturating_sub(LIVE_BYTES.load(Ordering::SeqCst));
        println!(
            "SNAPSHOT\thistory={history}\tcount={snapshot_count}\tretained_bytes={}\tbytes_per_snapshot={:.2}\talloc_calls_per_snapshot={:.2}\treleased_bytes={released}",
            held.retained_bytes,
            held.retained_bytes as f64 / snapshot_count as f64,
            held.allocation_calls as f64 / snapshot_count as f64,
        );
        fixture.close().await;
    }
}

async fn benchmark_old_generation_release(full: bool) {
    println!("SECTION\told_generation_release");
    let histories = if full {
        vec![1, 8, 32, 128, 512]
    } else {
        vec![8, 128]
    };
    let reps = if full { 5 } else { 3 };
    for history in histories {
        let mut released_samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let fixture = ReaderFixture::new(history, Segmentation::None).await;
            let old_snapshot = fixture.reader.snapshot().await.expect("snapshot failed");
            fixture
                .db
                .flush_with_options(FlushOptions {
                    flush_type: FlushType::MemTable,
                })
                .await
                .expect("L0 flush failed");
            fixture.poll(0).await;
            settle().await;
            let before = LIVE_BYTES.load(Ordering::SeqCst);
            drop(old_snapshot);
            settle().await;
            let after = LIVE_BYTES.load(Ordering::SeqCst);
            released_samples.push(before.saturating_sub(after));
            fixture.close().await;
        }
        released_samples.sort_unstable();
        println!(
            "RELEASE\thistory={history}\treps={reps}\treleased_bytes_p50={}",
            released_samples[(released_samples.len() - 1) / 2]
        );
    }
}

async fn run() {
    let full = std::env::var_os("SLATEDB_READER_BENCH_FULL").is_some();
    println!(
        "CONFIG\tprofile={}\tfull={full}\tallocator=system-counting",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    println!("HEADER\tsuite\tcase\thistory\tdelta\treps\tretained_p50_bytes\tpeak_extra_p50_bytes\tallocated_p50_bytes\tallocation_calls_p50\tnotes");
    benchmark_incremental_replay_memory(full).await;
    benchmark_snapshot_memory(full).await;
    benchmark_old_generation_release(full).await;
}

fn main() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build Tokio runtime");
    runtime.block_on(run());
}
