//! A panic in an index job or in a reaper turn ended its worker thread, and
//! with it more than that job. An index job's collection stayed in `active`,
//! where every later submit for it is parked and never taken. A reaper job's
//! key stayed in `queued_or_running`, so its segment was never submitted
//! again. And each panic left a worker fewer for every other collection.

use super::*;

const WAIT: Duration = Duration::from_secs(10);

fn wait_until(what: &str, done: impl Fn() -> bool) {
    let started = Instant::now();
    while !done() {
        assert!(started.elapsed() < WAIT, "never: {what}");
        thread::sleep(Duration::from_millis(10));
    }
}

/// Each of `names` panics once more, as an index job or as a reaper turn.
fn panics(names: &[String]) {
    WORKER_PANICS_FOR_TEST
        .lock()
        .unwrap()
        .extend(names.iter().cloned());
}

/// How many panics `name` still has to make.
fn panics_left(name: &str) -> usize {
    let left = WORKER_PANICS_FOR_TEST.lock().unwrap();
    left.iter().filter(|listed| *listed == name).count()
}

/// A job that ends as soon as it is taken: its collection is not loaded.
fn job(collection: &str) -> IndexJob {
    IndexJob {
        collection: collection.to_string(),
        storage: Weak::new(),
        spaces_to_check: Vec::new(),
        hnsw_config: HNSWConfig::new(Some(8), Some(32), Some(64)),
        flat_scan_threshold: 1,
    }
}

fn nothing_queued_or_running(executor: &IndexExecutor) -> bool {
    let state = executor.state.0.lock().unwrap();
    state.queued_jobs.is_empty() && state.active.is_empty()
}

#[test]
fn an_index_job_that_panics_leaves_its_worker_and_its_collection_to_the_next_job() {
    let executor = IndexExecutor::new();
    // As many panics as there are workers: uncaught, no worker is left.
    let panicking: Vec<String> = (0..index_executor_worker_count())
        .map(|worker| format!("index_job_panics_{worker}"))
        .collect();
    panics(&panicking);
    for collection in &panicking {
        executor.submit(job(collection)).unwrap();
    }
    wait_until("every one of the jobs panicked", || {
        panicking.iter().all(|name| panics_left(name) == 0)
    });

    // The same collections again, and one that never had a job.
    let after = panicking
        .iter()
        .map(String::as_str)
        .chain(["index_job_runs_after_the_panics"]);
    for collection in after {
        executor.submit(job(collection)).unwrap();
    }
    wait_until("the jobs submitted after the panics ran", || {
        nothing_queued_or_running(&executor)
    });
}

#[test]
fn a_reaper_turn_that_panics_leaves_its_worker_and_its_job_takes_the_failed_path() {
    let dir = TempDir::new().unwrap();
    let storage = Arc::new(
        HelixGraphStorage::new(
            dir.path().to_str().unwrap(),
            Config::new(8, 32, 64, 1).with_lsm_in_memory(),
        )
        .unwrap(),
    );
    // One worker: uncaught, the first panic leaves none.
    let reaper = SegmentReaper::with_workers(1);
    let panicking = "reaper_turn_panics__seg_000001";
    // Its first turn and each turn the failed path gives it after, and a
    // panic more than those turns can make.
    let turns = segment_reaper_retry_attempts() + 1;
    panics(&vec![panicking.to_string(); turns + 1]);
    reaper.submit(Arc::clone(&storage), panicking.to_string());
    wait_until("the job's first turn panicked", || {
        panics_left(panicking) <= turns
    });

    // A segment the collection does not hold: every step finds it drained.
    reaper.submit(
        Arc::clone(&storage),
        "reaper_turn_runs__seg_000001".to_string(),
    );
    wait_until("both jobs left the reaper", || reaper.idle());
    // Tried as often as a job that fails is, then given up, with the last
    // panic unmade. A turn taken for one that yielded is given another turn
    // each time, and makes that one too.
    let unmade = panics_left(panicking);
    WORKER_PANICS_FOR_TEST
        .lock()
        .unwrap()
        .retain(|listed| listed != panicking);
    assert_eq!(unmade, 1);
}
