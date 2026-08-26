//! Multi-threaded HNSW stress and consistency tests.
//!
//! The fork's existing vector_core test suite is single-threaded. This module
//! adds coverage for concurrent readers — the workload Context Engine
//! actually runs in production (many search threads against a sealed segment).
//!
//! What's intentionally NOT here:
//!   - Concurrent inserts. The fork serializes writes through the write_queue
//!     and per-env writers; the unit-test layer doesn't exercise that path.
//!     Concurrent insert correctness is tested at the gateway layer.
//!   - Loom model checking. The fork has fork-specific concurrency primitives
//!     (resize gate, build permits, write queue) that loom would need
//!     instrumented; that's a separate effort.
//!
//! Each test is `#[serial(lmdb_stress)]` to keep multiple LMDB-touching tests
//! from contending for the same TempDir/page cache while running in the
//! cargo-test thread pool.

use rand::{Rng, SeedableRng};
use serial_test::serial;
use std::sync::{Arc, Barrier};
use std::thread;
use tempfile::TempDir;

use crate::helix_engine::storage_core::backend_any::AnyBackend;
use crate::helix_engine::storage_core::backend_lmdb::LmdbBackend;
use crate::helix_engine::vector_core::hnsw::HNSW;
use crate::helix_engine::vector_core::named_vectors::DistanceMetric;
use crate::helix_engine::vector_core::spindle::{SpindleConfig, SpindleMode};
use crate::helix_engine::vector_core::vector::HVector;
use crate::helix_engine::vector_core::vector_core::{HNSWConfig, VectorCore};

type VF = fn(&HVector) -> bool;

fn setup_env() -> (heed3::Env, TempDir) {
    let dir = TempDir::new().unwrap();
    let env = unsafe {
        heed3::EnvOpenOptions::new()
            .map_size(256 * 1024 * 1024)
            .max_dbs(32)
            .max_readers(64)
            .open(dir.path())
            .unwrap()
    };
    (env, dir)
}

fn build_core(env: &heed3::Env, name: &str) -> VectorCore {
    let mut txn = env.write_txn().unwrap();
    let backend = Arc::new(AnyBackend::Lmdb(LmdbBackend::from_env(env.clone())));
    let core = VectorCore::new_named(
        env,
        &mut txn,
        name,
        HNSWConfig::new(Some(8), Some(32), Some(64)),
        DistanceMetric::Cosine,
        SpindleConfig {
            mode: SpindleMode::None,
            ..SpindleConfig::default()
        },
        backend,
    )
    .unwrap();
    txn.commit().unwrap();
    core
}

fn random_vector(rng: &mut impl Rng, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|_| rng.random_range(-1.0f32..1.0f32))
        .collect()
}

/// Insert `n` random vectors, then commit. Returns nothing — caller searches
/// against the resulting sealed state.
fn populate(env: &heed3::Env, core: &VectorCore, n: usize, dim: usize, seed: u64) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut txn = env.write_txn().unwrap();
    for i in 0..n {
        let data = random_vector(&mut rng, dim);
        core.insert::<VF>(&mut txn, &data, Some((i + 1) as u128), None)
            .unwrap();
    }
    txn.commit().unwrap();
}

#[test]
#[serial(lmdb_stress)]
fn concurrent_readers_return_consistent_topk() {
    // N threads each issue many searches against the same sealed index
    // simultaneously. We assert two invariants:
    //   1. No thread panics or returns Err — heed3 MVCC must hold up under
    //      concurrent RoTxn open/close from multiple threads.
    //   2. Every search returns exactly k results (the index has >> k
    //      vectors, so partial results would indicate a graph integrity bug).
    let (env, _dir) = setup_env();
    let core = build_core(&env, "conc_readers");
    populate(&env, &core, 300, 16, 0xc0ffee);

    let env = Arc::new(env);
    let core = Arc::new(core);
    const THREADS: usize = 8;
    const SEARCHES_PER_THREAD: usize = 25;
    const K: usize = 10;
    let barrier = Arc::new(Barrier::new(THREADS));

    let handles: Vec<_> = (0..THREADS)
        .map(|tid| {
            let env = Arc::clone(&env);
            let core = Arc::clone(&core);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let mut rng = rand::rngs::StdRng::seed_from_u64(0xbeef + tid as u64);
                for _ in 0..SEARCHES_PER_THREAD {
                    let txn = env.read_txn().unwrap();
                    let query = random_vector(&mut rng, 16);
                    let res = core
                        .search::<VF>(&core.backend.read_borrowed(&txn), &query, K, None, false)
                        .expect("search should not error under concurrent readers");
                    assert_eq!(
                        res.len(),
                        K,
                        "thread {tid}: expected {K} results, got {}",
                        res.len()
                    );
                    // Distance ordering must hold within each result set.
                    for w in res.windows(2) {
                        assert!(
                            w[1].get_distance() >= w[0].get_distance(),
                            "thread {tid}: results not in ascending distance"
                        );
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("worker thread should not panic");
    }
}

#[test]
#[serial(lmdb_stress)]
fn concurrent_readers_with_distinct_queries_yield_stable_ids() {
    // Two passes, single-threaded then multi-threaded, with the same
    // deterministic queries. Result IDs must match — proves concurrent
    // reads don't corrupt state or perturb topology.
    let (env, _dir) = setup_env();
    let core = build_core(&env, "conc_stable");
    populate(&env, &core, 200, 16, 0xdead_d00d);

    // Generate a fixed bank of queries.
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x515c01d);
    let queries: Vec<Vec<f32>> = (0..16).map(|_| random_vector(&mut rng, 16)).collect();
    const K: usize = 8;

    // Single-threaded baseline.
    let baseline: Vec<Vec<u128>> = {
        let txn = env.read_txn().unwrap();
        queries
            .iter()
            .map(|q| {
                core.search::<VF>(&core.backend.read_borrowed(&txn), q, K, None, false)
                    .unwrap()
                    .iter()
                    .map(|h| h.get_id())
                    .collect()
            })
            .collect()
    };

    // Multi-threaded: each thread re-runs all queries; ids must match baseline.
    let env = Arc::new(env);
    let core = Arc::new(core);
    let queries = Arc::new(queries);
    let baseline = Arc::new(baseline);
    const THREADS: usize = 6;
    let barrier = Arc::new(Barrier::new(THREADS));

    let handles: Vec<_> = (0..THREADS)
        .map(|tid| {
            let env = Arc::clone(&env);
            let core = Arc::clone(&core);
            let queries = Arc::clone(&queries);
            let baseline = Arc::clone(&baseline);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let txn = env.read_txn().unwrap();
                for (qi, q) in queries.iter().enumerate() {
                    let res = core
                        .search::<VF>(&core.backend.read_borrowed(&txn), q, K, None, false)
                        .unwrap();
                    let ids: Vec<u128> = res.iter().map(|h| h.get_id()).collect();
                    assert_eq!(
                        ids, baseline[qi],
                        "thread {tid} query {qi}: id list diverged from single-threaded baseline"
                    );
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
#[serial(lmdb_stress)]
fn delete_then_concurrent_search_excludes_tombstoned_id() {
    // Delete a vector, then have N readers search across many queries.
    // None of them should ever surface the deleted id — covers the
    // "tombstone visible across MVCC snapshot" class of bug at the unit
    // level.
    let (env, _dir) = setup_env();
    let core = build_core(&env, "conc_delete");
    populate(&env, &core, 150, 16, 0xfeed_face);

    // Delete a known id (vector 7) and commit.
    let mut wtxn = env.write_txn().unwrap();
    core.delete_vector(&mut wtxn, 7).unwrap();
    wtxn.commit().unwrap();

    let env = Arc::new(env);
    let core = Arc::new(core);
    const THREADS: usize = 4;
    const SEARCHES_PER_THREAD: usize = 20;
    let barrier = Arc::new(Barrier::new(THREADS));

    let handles: Vec<_> = (0..THREADS)
        .map(|tid| {
            let env = Arc::clone(&env);
            let core = Arc::clone(&core);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let mut rng = rand::rngs::StdRng::seed_from_u64(0x1234 + tid as u64);
                let txn = env.read_txn().unwrap();
                for _ in 0..SEARCHES_PER_THREAD {
                    let query = random_vector(&mut rng, 16);
                    let res = core
                        .search::<VF>(&core.backend.read_borrowed(&txn), &query, 50, None, false)
                        .unwrap();
                    for hit in &res {
                        assert_ne!(
                            hit.get_id(),
                            7,
                            "thread {tid}: deleted id 7 leaked into search results"
                        );
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
