use super::tests::{make_input, setup_with_config, upsert_points, TestContext};
use super::text_prefilter_tests::run;
use super::*;
use crate::helix_engine::graph_core::config::Config;
use crate::helix_engine::vector_core::vector_core::{broad_scan_completions_for_test, HNSWConfig};
use rand::{Rng, SeedableRng};
use std::sync::Arc;

const DIM: usize = 16;
const TOPICS: usize = 6;
const K: usize = 10;

/// The client's scoped, test-excluding search: `path_scopes` as `must` text,
/// `not_path_text` as `must_not` text.
fn scoped_filter() -> serde_json::Value {
    serde_json::json!({
        "must": [{"key": "metadata.path", "match": {"text": "src"}}],
        "must_not": [{"key": "metadata.path", "match": {"text": "test"}}]})
}

fn normalized(v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.into_iter().map(|x| x / norm).collect()
}

fn near(rng: &mut rand::rngs::StdRng, center: &[f32], spread: f32) -> Vec<f32> {
    normalized(
        center
            .iter()
            .map(|x| x + rng.random_range(-spread..spread))
            .collect(),
    )
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Per topic, `[tests, members, docs]` points: tests sit tightest around the
/// topic center (a feature's tests are its nearest neighbours), so the
/// `path~src` superset's nearest points to a topic query are all excluded.
fn add_batch(
    rng: &mut rand::rngs::StdRng,
    centers: &[Vec<f32>],
    counts: [usize; 3],
    next_id: &mut u32,
    vectors: &mut HashMap<u128, Vec<f32>>,
) -> serde_json::Value {
    let mut points = Vec::new();
    for (topic, center) in centers.iter().enumerate() {
        for (kind, count) in counts.into_iter().enumerate() {
            for _ in 0..count {
                let (path, spread) = match kind {
                    0 => (
                        format!("/work/repo/src/tests/t{topic}/{next_id}.test.ts"),
                        0.02,
                    ),
                    1 => (format!("/work/repo/src/lib/t{topic}/{next_id}.ts"), 0.12),
                    _ => (format!("/work/repo/docs/t{topic}/{next_id}.md"), 0.12),
                };
                let vector = near(rng, center, spread);
                points.push(
                    serde_json::json!({"id": *next_id, "vector": {"dense": vector},
                    "payload": {"metadata": {"path": path}}}),
                );
                vectors.insert(u128::from(*next_id), vector);
                *next_id += 1;
            }
        }
    }
    serde_json::Value::Array(points)
}

/// `(ctx, storage, vector by id, topic centers)`.
type Fixture = (
    TestContext,
    Arc<HelixGraphStorage>,
    HashMap<u128, Vec<f32>>,
    Vec<Vec<f32>>,
);

/// A sealed HNSW segment plus a mutable flat one (prod's segment shape), each
/// with a materialized sidecar so dense search can take the broad-filter scan.
fn fixture() -> Fixture {
    let mut config = Config::default().with_lsm_in_memory();
    config.vector_config.flat_scan_threshold = Some(0);
    let ctx = setup_with_config(config);
    let input = make_input(
        &ctx,
        "PUT",
        "/collections/dirs",
        serde_json::to_vec(
            &serde_json::json!({"vectors": {"dense": {"size": DIM, "distance": "Cosine"}}}),
        )
        .unwrap(),
        HashMap::from([("name".into(), "dirs".into())]),
    );
    let mut response = Response::new();
    handle_create_collection(&input, &mut response).unwrap();
    assert!(response.status == 200 || response.status == 201);
    let storage = ctx.collections.get_collection("dirs").unwrap();
    assert_eq!(storage.backend.kind(), BackendKind::Lsm);
    storage
        .create_payload_index("metadata.path", PayloadIndexSchema::Keyword)
        .unwrap();

    let mut rng = rand::rngs::StdRng::seed_from_u64(41);
    let centers: Vec<Vec<f32>> = (0..TOPICS)
        .map(|_| normalized((0..DIM).map(|_| rng.random_range(-1.0f32..1.0)).collect()))
        .collect();
    let mut vectors = HashMap::new();
    let mut next_id = 1;
    let sealed = add_batch(
        &mut rng,
        &centers,
        [450, 400, 40],
        &mut next_id,
        &mut vectors,
    );
    upsert_points(&ctx, "dirs", sealed);
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
    let tail = storage
        .named_vectors
        .mutable_dense_segment_name("dense")
        .unwrap();
    storage
        .with_write_backend(|w| {
            storage
                .named_vectors
                .mark_sealed_tail_be(w, &tail)
                .map_err(GraphError::from)
        })
        .unwrap();
    let read = storage.backend.begin_read().unwrap();
    storage
        .named_vectors
        .seal_mutable_tail_lsm(
            &read,
            storage.collection_path(),
            "dense",
            HNSWConfig::new(Some(16), Some(128), Some(64)),
        )
        .unwrap();
    drop(read);
    storage
        .with_write_backend(|w| {
            storage.set_dense_vector_spaces_metadata_be(
                w,
                storage.named_vectors.list_dense_vector_spaces(),
            )
        })
        .unwrap();
    let mutable = add_batch(&mut rng, &centers, [20, 15, 5], &mut next_id, &mut vectors);
    upsert_points(&ctx, "dirs", mutable);

    let segments = storage.named_vectors.list_dense_vector_spaces()["dense"]
        .segments
        .clone();
    assert_eq!(segments.len(), 2, "dense segments");
    let read = storage.backend.begin_read().unwrap();
    for segment in &segments {
        storage
            .named_vectors
            .with_core(&segment.physical_name, |core| {
                assert!(!core.is_empty(&read)?);
                core.ensure_lsm_mmap_sidecar()?;
                assert!(core.has_mmap_sidecar(), "{}", segment.physical_name);
                Ok(())
            })
            .unwrap();
    }
    drop(read);
    (ctx, storage, vectors, centers)
}

fn top_ids(endpoint: &str, result: &serde_json::Value) -> Vec<u128> {
    let points = match endpoint {
        "query/hybrid" => &result["channels"]["dense"][0]["points"],
        _ if result.is_array() => result,
        _ => &result["points"],
    };
    points
        .as_array()
        .unwrap()
        .iter()
        .take(K)
        .map(|point| point_id_to_u128(&point["id"]))
        .collect()
}

/// Directory-scoped, test-excluding dense search on a multi-segment
/// collection: recall@10 against brute force on all three routes.
///
/// The broad-filter scan kept only its 4 x target_k nearest superset members
/// and rechecked those, so when the excluded tests crowd the query it found
/// no match in the sealed segment: recall 0 on `/points/query` and the hybrid
/// dense channel, whose planner hands the scan the `path~src` superset, while
/// `/points/search` kept the set and ran filtered HNSW. With exact text walks
/// (the default) the planner now also subtracts the `must_not` walk, so every
/// route scans the exact match set. With walks unproven the superset still
/// goes to the scan, which must recheck past its window. No asserted arm
/// touches HNSW, so rng-seeded levels cannot flake it; `/points/search`
/// without exact walks runs filtered HNSW and is only reported.
#[test]
fn scoped_test_excluding_dense_search_matches_brute_force_on_every_route() {
    let (ctx, storage, vectors, centers) = fixture();
    let filter_json = scoped_filter();
    let filter: Filter = serde_json::from_value(filter_json.clone()).unwrap();
    let read = storage.backend.begin_read().unwrap();
    let members: Vec<u128> = vectors
        .keys()
        .copied()
        .filter(|id| {
            let node = storage.get_node_be(&read, id).unwrap();
            filter.matches_point(Some(*id), &node.properties)
        })
        .collect();
    drop(read);
    assert_eq!(members.len(), TOPICS * 415);

    let mut rng = rand::rngs::StdRng::seed_from_u64(43);
    // Topic queries (tests crowd them) plus midpoints of two members.
    let mut queries: Vec<Vec<f32>> = centers.iter().map(|c| near(&mut rng, c, 0.01)).collect();
    for _ in 0..12 {
        let a = &vectors[&members[rng.random_range(0..members.len())]];
        let b = &vectors[&members[rng.random_range(0..members.len())]];
        queries.push(normalized(a.iter().zip(b).map(|(x, y)| x + y).collect()));
    }
    // The cosine of each query's K-th best member: a hit that reaches it is
    // right. The two members of a midpoint query have the same cosine, and
    // when they sit at ranks K and K + 1 an exact search may return either
    // (a set of the first K ids made about one run in twenty fail).
    let floors: Vec<f32> = queries
        .iter()
        .map(|query| {
            let mut ranked: Vec<f32> = members
                .iter()
                .map(|id| cosine(query, &vectors[id]))
                .collect();
            ranked.sort_by(|a, b| b.total_cmp(a));
            ranked[K - 1] - 1e-5
        })
        .collect();

    let mut failures = Vec::new();
    for walk_exact in [true, false] {
        for endpoint in ["search", "query", "query/hybrid"] {
            let (mut recalls, mut node_reads, mut micros) = (Vec::new(), 0usize, 0u128);
            let scans_before = broad_scan_completions_for_test();
            for (query, floor) in queries.iter().zip(&floors) {
                let body = match endpoint {
                    "search" => serde_json::json!({"vector": {"name": "dense", "vector": query},
                        "limit": K, "filter": filter_json}),
                    "query" => serde_json::json!({"query": query, "using": "dense", "limit": K,
                        "filter": filter_json}),
                    _ => serde_json::json!({"dense": [{"query": query, "using": "dense",
                        "limit": 24}], "limit": 24, "filter": filter_json,
                        "with_channel_points": true}),
                };
                let started = Instant::now();
                let (result, cost) =
                    with_text_walk_exact_for_test(walk_exact, || run(&ctx, endpoint, &body));
                micros += started.elapsed().as_micros();
                node_reads += cost[0];
                let hits = top_ids(endpoint, &result);
                assert!(
                    hits.iter().all(|id| members.contains(id)),
                    "{endpoint}: non-member"
                );
                let right = hits
                    .iter()
                    .filter(|&&id| cosine(query, &vectors[&id]) >= *floor)
                    .count();
                recalls.push(right as f64 / K as f64);
            }
            let mean = recalls.iter().sum::<f64>() / recalls.len() as f64;
            println!(
                "walk_exact={walk_exact} {endpoint:>12}: recall@10 mean {mean:.3} topic queries \
                 {:?} | {node_reads} node reads, {} ms over {} queries",
                &recalls[..TOPICS],
                micros / 1000,
                queries.len()
            );
            if !walk_exact && endpoint == "search" {
                continue;
            }
            if broad_scan_completions_for_test() == scans_before {
                failures.push(format!("walk_exact={walk_exact} {endpoint}: no broad scan"));
            }
            if recalls.iter().any(|recall| *recall < 1.0) {
                failures.push(format!("walk_exact={walk_exact} {endpoint}: {recalls:?}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The client's test exclusion is subtracted when its walk is exact, so the scoped
/// set is exactly the matches; an unproven walk keeps the superset.
#[test]
fn exact_must_not_text_walk_is_subtracted() {
    let (_ctx, storage, _vectors, _centers) = fixture();
    let filter: Filter = serde_json::from_value(scoped_filter()).unwrap();
    let read = storage.backend.begin_read().unwrap();
    let (ids, exact) = indexed_filter_ids_be(&storage, &read, &filter, 0)
        .unwrap()
        .unwrap();
    assert!(exact);
    assert_eq!(ids.len(), TOPICS * 415);
    let (ids, exact) = with_text_walk_exact_for_test(false, || {
        indexed_filter_ids_be(&storage, &read, &filter, 0)
            .unwrap()
            .unwrap()
    });
    assert!(!exact);
    assert_eq!(ids.len(), TOPICS * (415 + 470));
    // A must_not walk is skipped like a `must` one behind a small set.
    let (_, exact) = indexed_filter_ids_be(&storage, &read, &filter, TOPICS * 900)
        .unwrap()
        .unwrap();
    assert!(!exact);
}
