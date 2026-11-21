//! Hybrid requests run their dense and sparse channels at the same time
//! (`HELIX_HYBRID_PARALLEL_CHANNELS`); the response must be the one the serial
//! path returns.

use super::tests::{make_input, setup_with_config, upsert_mixed_points, TestContext};
use super::text_prefilter_tests::{dir_fixture_with, filter_shapes, path_text, requests};
use super::*;
use crate::helix_engine::graph_core::config::Config;
use crate::helix_engine::storage_core::backend::{
    BackendError, LsmRole, LsmStorage, Namespace, StorageBackend, StorageBackendConfig,
};
use crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store;
use crate::helix_engine::storage_core::backend_lsm::{
    allow_lsm_blocking, allow_lsm_blocking_cancellable, block_on_lsm_read,
    lsm_round_trips_for_test, shared_lsm_handle, LsmReadCancellation,
};
use rand::{Rng, SeedableRng};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// One hybrid request against the `dirs` fixture with the channels run in
/// parallel or not: the handler's error, or the status and the whole body.
fn hybrid(
    ctx: &TestContext,
    body: &serde_json::Value,
    parallel: bool,
) -> Result<(u16, serde_json::Value), String> {
    hybrid_on(ctx, "dirs", body, parallel)
}

fn hybrid_on(
    ctx: &TestContext,
    collection: &str,
    body: &serde_json::Value,
    parallel: bool,
) -> Result<(u16, serde_json::Value), String> {
    let input = make_input(
        ctx,
        "POST",
        &format!("/collections/{collection}/points/hybrid_query"),
        serde_json::to_vec(body).unwrap(),
        HashMap::from([("name".into(), collection.into())]),
    );
    let mut response = Response::new();
    // As the gateway runs a handler: reads nested in a scan are allowed.
    allow_lsm_blocking(|| {
        with_hybrid_parallel_channels_for_test(parallel, || {
            handle_hybrid_query_points(&input, &mut response)
        })
    })
    .map_err(|e| e.to_string())?;
    let body = serde_json::from_slice(&response.body).unwrap();
    Ok((response.status, body))
}

/// The fixture's hybrid request: two dense and two sparse channels, one of
/// each with a filter of its own, channel points on.
fn hybrid_request(filter: &serde_json::Value, v: [f32; 3], w: [f32; 3]) -> serde_json::Value {
    requests(filter, v, w)
        .into_iter()
        .find(|(endpoint, _)| *endpoint == "query/hybrid")
        .unwrap()
        .1
}

fn channel_points(body: &serde_json::Value, kind: &str, channel: usize) -> usize {
    body["result"]["channels"][kind][channel]["points"]
        .as_array()
        .unwrap()
        .len()
}

#[test]
fn parallel_channels_return_the_serial_response_for_every_filter_shape() {
    let (ctx, _storage) = dir_fixture_with(true);
    let mut rng = rand::rngs::StdRng::seed_from_u64(41);
    let mut vector = || -> [f32; 3] { std::array::from_fn(|_| rng.random_range(-1.0f32..1.0)) };
    let queries: Vec<([f32; 3], [f32; 3])> = (0..2).map(|_| (vector(), vector())).collect();
    let mut shapes: Vec<(String, Option<serde_json::Value>)> = filter_shapes()
        .into_iter()
        .map(|(label, filter)| (label, Some(filter)))
        .collect();
    shapes.push(("no filter".into(), None));
    let mut both_kinds_answered = 0usize;
    for (label, filter) in shapes {
        for (v, w) in &queries {
            let mut plain = hybrid_request(&serde_json::json!({}), *v, *w);
            match &filter {
                Some(filter) => plain["filter"] = filter.clone(),
                None => {
                    plain.as_object_mut().unwrap().remove("filter");
                }
            }
            let mut reranked = plain.clone();
            reranked["with_payload"] = serde_json::json!(true);
            reranked["rrf_k"] = serde_json::json!(20.0);
            reranked["mmr"] = serde_json::json!({"lambda": 0.5});
            for body in [plain, reranked] {
                let serial = hybrid(&ctx, &body, false).unwrap();
                let parallel = hybrid(&ctx, &body, true).unwrap();
                assert_eq!(serial.0, 200, "{label}: {}", serial.1);
                assert_eq!(parallel, serial, "{label}: {body}");
                if channel_points(&serial.1, "dense", 0) > 0
                    && channel_points(&serial.1, "sparse", 0) > 0
                {
                    both_kinds_answered += 1;
                }
            }
        }
    }
    assert!(
        both_kinds_answered > 20,
        "only {both_kinds_answered} requests had dense and sparse hits to fuse"
    );
}

#[test]
fn parallel_channels_return_the_serial_response_when_one_kind_is_empty() {
    let (ctx, _storage) = dir_fixture_with(true);
    let (v, w) = ([0.3, -0.8, 0.5], [-0.6, 0.1, 0.9]);
    let nothing = serde_json::json!({"must": [path_text("no_such_dir/")]});
    let full = hybrid_request(&serde_json::json!({"must": [path_text("/work/")]}), v, w);

    // No channel of one kind in the request.
    for kind in ["dense", "sparse"] {
        let mut body = full.clone();
        body[kind] = serde_json::json!([]);
        let serial = hybrid(&ctx, &body, false).unwrap();
        assert_eq!(serial.0, 200, "no {kind}: {}", serial.1);
        assert!(!serial.1["result"]["points"].as_array().unwrap().is_empty());
        assert_eq!(hybrid(&ctx, &body, true).unwrap(), serial, "no {kind}");
    }

    // Every channel of one kind finds nothing.
    for (kind, other) in [("dense", "sparse"), ("sparse", "dense")] {
        let mut body = full.clone();
        for channel in body[kind].as_array_mut().unwrap() {
            channel["filter"] = nothing.clone();
        }
        let serial = hybrid(&ctx, &body, false).unwrap();
        assert_eq!(serial.0, 200, "empty {kind}: {}", serial.1);
        assert_eq!(channel_points(&serial.1, kind, 0), 0, "empty {kind}");
        assert_eq!(channel_points(&serial.1, kind, 1), 0, "empty {kind}");
        assert!(channel_points(&serial.1, other, 0) > 0, "empty {kind}");
        assert_eq!(hybrid(&ctx, &body, true).unwrap(), serial, "empty {kind}");
    }
}

/// A failing channel surfaces what the serial path surfaces: its own error,
/// and the dense one when both kinds fail.
#[test]
fn parallel_channels_surface_the_serial_error() {
    let (ctx, _storage) = dir_fixture_with(true);
    let (v, w) = ([0.3, -0.8, 0.5], [-0.6, 0.1, 0.9]);
    let full = hybrid_request(&serde_json::json!({"must": [path_text("/work/")]}), v, w);
    let with = |edits: &[(&str, usize, &str, serde_json::Value)]| {
        let mut body = full.clone();
        for (kind, channel, key, value) in edits {
            body[*kind][*channel][*key] = value.clone();
        }
        body
    };
    let unknown = serde_json::json!("no_such_vector");
    let empty_dense = serde_json::json!([]);
    let empty_sparse = serde_json::json!({"indices": [], "values": []});
    let uneven_sparse = serde_json::json!({"indices": [3, 102], "values": [1.0]});
    let cases = [
        (
            "sparse fails",
            with(&[("sparse", 1, "using", unknown.clone())]),
            "no_such_vector",
        ),
        (
            "dense fails",
            with(&[("dense", 1, "using", unknown.clone())]),
            "no_such_vector",
        ),
        (
            "both fail",
            with(&[
                ("dense", 0, "using", serde_json::json!("no_dense")),
                ("sparse", 0, "using", serde_json::json!("no_sparse")),
            ]),
            "no_dense",
        ),
        (
            "empty dense query",
            with(&[("dense", 1, "query", empty_dense.clone())]),
            "Dense query vector must not be empty",
        ),
        (
            "empty sparse query",
            with(&[("sparse", 1, "query", empty_sparse.clone())]),
            "Sparse query vector must not be empty",
        ),
        (
            "uneven sparse query",
            with(&[("sparse", 0, "query", uneven_sparse)]),
            "Sparse query vector indices and values must have the same length",
        ),
        (
            "empty dense query, sparse fails",
            with(&[
                ("dense", 0, "query", empty_dense),
                ("sparse", 0, "using", unknown.clone()),
            ]),
            "Dense query vector must not be empty",
        ),
        (
            "dense fails, empty sparse query",
            with(&[
                ("dense", 0, "using", serde_json::json!("no_dense")),
                ("sparse", 0, "query", empty_sparse),
            ]),
            "no_dense",
        ),
    ];
    for (case, body, expected) in cases {
        let serial = hybrid(&ctx, &body, false);
        let text = match &serial {
            Ok((status, body)) => {
                assert_eq!(*status, 400, "{case}: {body}");
                body.to_string()
            }
            Err(error) => error.clone(),
        };
        assert!(text.contains(expected), "{case}: {text}");
        assert_eq!(hybrid(&ctx, &body, true), serial, "{case}");
    }
}

/// SlateDB round trips are counted per thread: with the channels in parallel
/// the sparse ones leave the handler thread, with the kill switch they stay.
#[test]
fn sparse_channels_run_off_the_handler_thread_unless_switched_off() {
    let (ctx, _storage) = dir_fixture_with(true);
    let (v, w) = ([0.3, -0.8, 0.5], [-0.6, 0.1, 0.9]);
    let body = hybrid_request(&serde_json::json!({"must": [path_text("/work/")]}), v, w);
    let mut dense_only = body.clone();
    dense_only["sparse"] = serde_json::json!([]);
    let trips = |body: &serde_json::Value, parallel: bool| {
        let before = lsm_round_trips_for_test();
        hybrid(&ctx, body, parallel).unwrap();
        lsm_round_trips_for_test() - before
    };
    // Warm every cache first, so the counts below differ by thread only.
    for body in [&body, &dense_only] {
        for parallel in [false, true] {
            trips(body, parallel);
        }
    }

    let serial = trips(&body, false);
    let parallel = trips(&body, true);
    assert_eq!(
        trips(&body, false),
        serial,
        "the serial count is not stable"
    );
    assert_eq!(
        trips(&body, true),
        parallel,
        "the parallel count is not stable"
    );
    assert!(
        parallel < serial,
        "handler thread round trips: {parallel} in parallel, {serial} serial"
    );

    // One kind of channel only: nothing to run beside it.
    assert_eq!(trips(&dense_only, true), trips(&dense_only, false));
}

/// Each side waits for the other to have started, so this only passes when the
/// two really run at the same time; a dense error wins over a sparse one.
#[test]
fn run_dense_beside_sparse_overlaps_the_two_and_prefers_the_dense_error() {
    let (dense_started, sparse_started) = (AtomicBool::new(false), AtomicBool::new(false));
    let meets = |own: &AtomicBool, other: &AtomicBool| {
        own.store(true, Ordering::SeqCst);
        let deadline = Instant::now() + Duration::from_secs(5);
        while !other.load(Ordering::SeqCst) {
            if Instant::now() > deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        true
    };
    let handler_thread = std::thread::current().id();
    let outcome: Result<(bool, (bool, bool)), ()> = run_dense_beside_sparse(
        || Ok(meets(&dense_started, &sparse_started)),
        || {
            Ok((
                meets(&sparse_started, &dense_started),
                std::thread::current().id() != handler_thread,
            ))
        },
    );
    assert_eq!(
        outcome,
        Ok((true, (true, true))),
        "(dense met sparse, (sparse met dense, sparse on a helper thread))"
    );

    type Outcome = Result<((), ()), &'static str>;
    let both: Outcome = run_dense_beside_sparse(|| Err("dense"), || Err("sparse"));
    assert_eq!(both, Err("dense"));
    let sparse_only: Outcome = run_dense_beside_sparse(|| Ok(()), || Err("sparse"));
    assert_eq!(sparse_only, Err("sparse"));
    let dense_only: Outcome = run_dense_beside_sparse(|| Err("dense"), || Ok(()));
    assert_eq!(dense_only, Err("dense"));
}

/// A panic in the sparse channels reaches the handler thread as that panic
/// (the gateway answers 500, as when they ran there) once dense has finished.
/// Dense is not left waiting on a lock the panicking side held, and a dense
/// panic waits for the helper instead of leaving it behind.
#[test]
fn run_dense_beside_sparse_raises_a_panic_from_either_side_without_hanging() {
    fn message(panic: Box<dyn std::any::Any + Send>) -> String {
        panic
            .downcast_ref::<&str>()
            .map(|message| (*message).to_string())
            .unwrap_or_default()
    }
    let wait_for = |flag: &AtomicBool| {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !flag.load(Ordering::SeqCst) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        flag.load(Ordering::SeqCst)
    };

    // Sparse panics while it holds the payload cache lock.
    let cache = parking_lot::Mutex::new(());
    let (sparse_locked, dense_finished) = (AtomicBool::new(false), AtomicBool::new(false));
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_dense_beside_sparse::<(), (), ()>(
            || {
                assert!(wait_for(&sparse_locked), "sparse never started");
                assert!(
                    cache.try_lock_for(Duration::from_secs(5)).is_some(),
                    "the panicking side kept the lock"
                );
                dense_finished.store(true, Ordering::SeqCst);
                Ok(())
            },
            || {
                let _held = cache.lock();
                sparse_locked.store(true, Ordering::SeqCst);
                panic!("sparse channel panicked")
            },
        )
    }));
    assert_eq!(
        message(caught.expect_err("the sparse panic was swallowed")),
        "sparse channel panicked"
    );
    assert!(dense_finished.load(Ordering::SeqCst), "dense was cut short");

    // Dense panics while sparse is still running.
    let (dense_panicking, sparse_finished) = (AtomicBool::new(false), AtomicBool::new(false));
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_dense_beside_sparse::<(), (), ()>(
            || {
                dense_panicking.store(true, Ordering::SeqCst);
                panic!("dense channel panicked")
            },
            || {
                assert!(wait_for(&dense_panicking), "dense never started");
                std::thread::sleep(Duration::from_millis(50));
                sparse_finished.store(true, Ordering::SeqCst);
                Ok(())
            },
        )
    }));
    assert_eq!(
        message(caught.expect_err("the dense panic was swallowed")),
        "dense channel panicked"
    );
    assert!(
        sparse_finished.load(Ordering::SeqCst),
        "the helper thread was left running"
    );
}

/// Cancelling the request stops the reads of both sides: the helper thread
/// runs in the handler thread's LSM context, so its pending read is dropped
/// like the handler's own.
#[test]
fn cancelling_the_request_stops_the_dense_and_the_sparse_side() {
    let rt = shared_lsm_handle();
    let cancellation = LsmReadCancellation::new();
    let sparse_reading = AtomicBool::new(false);
    // Whether the sparse read ended cancelled.
    let sparse_cancelled = parking_lot::Mutex::new(None);
    // A read that takes 30 s unless the request is cancelled.
    let slow_read = || {
        block_on_lsm_read(&rt, async {
            tokio::time::sleep(Duration::from_secs(30)).await;
            Ok::<(), &'static str>(())
        })
    };

    let started = Instant::now();
    let outcome = allow_lsm_blocking_cancellable(cancellation.clone(), || {
        run_dense_beside_sparse(
            || {
                let deadline = Instant::now() + Duration::from_secs(5);
                while !sparse_reading.load(Ordering::SeqCst) && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(1));
                }
                cancellation.cancel();
                slow_read()
            },
            || {
                sparse_reading.store(true, Ordering::SeqCst);
                let outcome = slow_read();
                *sparse_cancelled.lock() =
                    Some(matches!(&outcome, Err(error) if error.is_lsm_read_cancelled()));
                outcome
            },
        )
    });

    assert!(
        matches!(&outcome, Err(error) if error.is_lsm_read_cancelled()),
        "dense read: {outcome:?}"
    );
    assert_eq!(
        *sparse_cancelled.lock(),
        Some(true),
        "the sparse read outlived the request"
    );
    assert!(started.elapsed() < Duration::from_secs(10));
}

/// The helper must run in the request thread's LSM context: nested blocking
/// reads are legal in the gateway, and the sparse side can do them too.
#[test]
fn run_dense_beside_sparse_carries_lsm_blocking_context_to_the_helper() {
    let lsm_rt = shared_lsm_handle();
    let outer_rt = tokio::runtime::Runtime::new().unwrap();
    let outcome = allow_lsm_blocking(|| {
        run_dense_beside_sparse(
            || Ok::<(), BackendError>(()),
            || {
                outer_rt.block_on(async {
                    block_on_lsm_read(&lsm_rt, async { Ok::<(), &'static str>(()) })
                })
            },
        )
    });

    assert!(
        outcome.is_ok(),
        "sparse helper lost the gateway's LSM blocking context: {outcome:?}"
    );
}

/// Helper-thread code can read through an already-captured backend handle even
/// after newer writer state exists.
#[test]
fn run_dense_beside_sparse_accepts_a_captured_read_handle_on_the_helper() {
    let path = format!("hybrid-parallel-pinned-read-{}", std::process::id());
    let store = Arc::new(object_store::memory::InMemory::new());
    let writer =
        crate::helix_engine::storage_core::backend_lsm::LsmBackend::open_with_store(&path, store)
            .unwrap();
    {
        let mut write = writer.begin_write().unwrap();
        writer
            .put(&mut write, Namespace::Nodes, b"point", b"request")
            .unwrap();
        writer.commit(write).unwrap();
    }
    let request_read = writer.begin_read().unwrap();
    {
        let mut write = writer.begin_write().unwrap();
        writer
            .put(&mut write, Namespace::Nodes, b"point", b"newer")
            .unwrap();
        writer.commit(write).unwrap();
    }

    let outcome = allow_lsm_blocking(|| {
        run_dense_beside_sparse(
            || Ok::<(), BackendError>(()),
            || {
                writer.get_with(&request_read, Namespace::Nodes, b"point", |value| {
                    value.map(<[u8]>::to_vec)
                })
            },
        )
    })
    .unwrap();

    assert_eq!(outcome.1.as_deref(), Some(&b"request"[..]));
}

/// Sets an env var for the test and restores it after.
pub(super) struct EnvVar(&'static str, Option<String>);

impl EnvVar {
    pub(super) fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        std::env::set_var(key, value);
        Self(key, previous)
    }
}

impl Drop for EnvVar {
    fn drop(&mut self) {
        match &self.1 {
            Some(value) => std::env::set_var(self.0, value),
            None => std::env::remove_var(self.0),
        }
    }
}

/// Production serves hybrid from reader replicas: the helper thread reads the
/// request's pinned snapshot, or the replica's live state when reads are not
/// pinned, and the response is the serial one either way.
#[test]
#[serial_test::serial]
fn parallel_channels_return_the_serial_response_on_a_reader_replica() {
    let _store = set_lsm_test_object_store(Arc::new(object_store::memory::InMemory::new()));
    let _prefix = EnvVar::set("HELIX_LSM_PREFIX", "hybrid_parallel_reader");
    let config = |role| {
        Config::default().with_storage_backend(StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role,
        })
    };
    let writer = setup_with_config(config(LsmRole::Writer));
    let input = make_input(
        &writer,
        "PUT",
        "/collections/replica",
        serde_json::to_vec(&serde_json::json!({
            "vectors": {"dense": {"size": 3, "distance": "Cosine"}},
            "sparse_vectors": {"lex_sparse": {"modifier": "idf"}}
        }))
        .unwrap(),
        HashMap::from([("name".into(), "replica".into())]),
    );
    let mut response = Response::new();
    handle_create_collection(&input, &mut response).unwrap();
    assert!(response.status == 200 || response.status == 201);
    let points = (1..=300u32)
        .map(|id| {
            let angle = id as f32 * 0.37;
            serde_json::json!({
                "id": id,
                "vector": {
                    "dense": [angle.cos(), angle.sin(), (id % 17) as f32 / 17.0],
                    "lex_sparse": {"indices": [id % 13, 100 + id % 7], "values": [1.0, 0.5]}
                },
                "payload": {"language": if id % 2 == 0 { "rust" } else { "go" }}
            })
        })
        .collect::<Vec<_>>();
    upsert_mixed_points(&writer, "replica", serde_json::Value::Array(points));
    // Replicas search published segments: index the dense one.
    let storage = writer.collections.get_collection("replica").unwrap();
    storage
        .named_vectors
        .with_core("dense", |core| {
            let read = storage.backend.begin_read().unwrap();
            let mut prepared = core.prepare_index_from_flat(&read)?.unwrap();
            drop(read);
            let mut write = storage.backend.begin_write().unwrap();
            let total = prepared.point_ids.len();
            core.flush_prepared_index_chunk_be(&mut write, &mut prepared, 0, total)?;
            core.finalize_prepared_index_entry_be(&mut write, &prepared)?;
            storage.backend.commit(write).unwrap();
            Ok(())
        })
        .unwrap();

    let reader = setup_with_config(config(LsmRole::Reader));
    let storage = reader.collections.get_collection("replica").unwrap();
    assert!(storage.backend.is_reader_replica());
    let sparse = serde_json::json!({"indices": [3, 102], "values": [1.0, 1.0]});
    let body = serde_json::json!({
        "dense": [
            {"query": [0.3, -0.8, 0.5], "using": "dense", "limit": 12},
            {"query": [-0.6, 0.1, 0.9], "using": "dense", "limit": 12}],
        "sparse": [{"query": sparse, "using": "lex_sparse", "limit": 12}],
        "filter": {"must": [{"key": "language", "match": {"value": "rust"}}]},
        "limit": 12, "with_payload": true, "with_channel_points": true
    });
    for pinned in ["1", "0"] {
        let _pinned = EnvVar::set("HELIX_LSM_READER_SNAPSHOT_READS", pinned);
        let serial = hybrid_on(&reader, "replica", &body, false).unwrap();
        assert_eq!(serial.0, 200, "pinned {pinned}: {}", serial.1);
        assert!(channel_points(&serial.1, "dense", 0) > 0, "pinned {pinned}");
        assert!(
            channel_points(&serial.1, "sparse", 0) > 0,
            "pinned {pinned}"
        );
        let parallel = hybrid_on(&reader, "replica", &body, true).unwrap();
        assert_eq!(parallel, serial, "pinned {pinned}");
    }
}
