//! Bulk-build neighbor diversity (`select_neighbor_ords` with `diverse`):
//! selection invariants, a clustered-data recall regression, and an
//! `#[ignore]` offline eval over real dense vectors.

use super::*;
use crate::helix_engine::storage_core::backend_lsm::LsmBackend;
use crate::helix_engine::storage_core::backend_lsm_reader::LsmReader;
use object_store::{memory::InMemory, ObjectStore};
use rand::SeedableRng;
use tempfile::TempDir;

const K: usize = 10;

fn euclid(points: &[Vec<f32>]) -> impl Fn(usize, usize) -> f32 + '_ {
    move |a, b| {
        points[a]
            .iter()
            .zip(&points[b])
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }
}

fn ranked_from(base: usize, points: &[Vec<f32>]) -> Vec<(usize, f32)> {
    let dist = euclid(points);
    let mut ranked: Vec<(usize, f32)> = (0..points.len())
        .filter(|&ord| ord != base)
        .map(|ord| (ord, dist(base, ord)))
        .collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    ranked
}

#[test]
fn diverse_selection_skips_a_candidate_shadowed_by_a_kept_neighbor() {
    // Base at the origin. `b` is closer to `a` than to the base, so only the
    // fill may bring it back; nearest-M keeps it ahead of `c`.
    let points = vec![
        vec![0.0, 0.0],  // base
        vec![1.0, 0.0],  // a
        vec![1.1, 0.1],  // b
        vec![-1.2, 0.0], // c
    ];
    let dist = euclid(&points);
    let ranked = ranked_from(0, &points);
    assert_eq!(
        select_neighbor_ords(&ranked, 3, 4, Some(false), &dist),
        [1, 3]
    );
    assert_eq!(
        select_neighbor_ords(&ranked, 3, 4, Some(true), &dist),
        [1, 3, 2]
    );
    assert_eq!(
        select_neighbor_ords(&ranked, 2, 4, Some(true), &dist),
        [1, 3]
    );
    assert_eq!(select_neighbor_ords(&ranked, 2, 4, None, &dist), [1, 2]);
}

#[test]
fn diverse_selection_invariants_on_random_points() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let points: Vec<Vec<f32>> = (0..300)
        .map(|_| (0..2).map(|_| rng.random_range(-1.0f32..1.0)).collect())
        .collect();
    let dist = euclid(&points);
    let cap = 16;
    let mut filled_some = false;
    for base in 0..40 {
        let ranked = ranked_from(base, &points);
        let strict = select_neighbor_ords(&ranked, cap, points.len(), Some(false), &dist);
        assert_eq!(strict[0] as usize, ranked[0].0, "nearest is always kept");
        assert!(strict.len() <= cap);
        for (i, &kept) in strict.iter().enumerate() {
            let to_base = dist(kept as usize, base);
            for &earlier in &strict[..i] {
                assert!(
                    dist(kept as usize, earlier as usize) >= to_base,
                    "base {base}: kept {kept} is closer to kept {earlier} than to the base"
                );
            }
        }

        // keepPruned: same diverse prefix, then the nearest rejected
        // candidates up to the cap.
        let filled = select_neighbor_ords(&ranked, cap, points.len(), Some(true), &dist);
        assert_eq!(filled.len(), cap);
        assert_eq!(filled[..strict.len()], strict[..]);
        let nearest_rejected: Vec<u32> = ranked
            .iter()
            .map(|&(ord, _)| ord as u32)
            .filter(|ord| !strict.contains(ord))
            .take(cap - strict.len())
            .collect();
        assert_eq!(filled[strict.len()..], nearest_rejected[..]);
        filled_some |= strict.len() < cap;

        let nearest = select_neighbor_ords(&ranked, cap, points.len(), None, &dist);
        let expected: Vec<u32> = ranked[..cap].iter().map(|&(ord, _)| ord as u32).collect();
        assert_eq!(nearest, expected);
    }
    assert!(
        filled_some,
        "2-D diverse sets should be smaller than the cap"
    );
}

fn normalized(v: Vec<f32>) -> Vec<f32> {
    let norm = v
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
        .max(f32::MIN_POSITIVE);
    v.into_iter().map(|x| x / norm).collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt()).max(f32::MIN_POSITIVE)
}

/// Normalized midpoints of two random vectors from `pool` (ordinals), plus
/// their exact cosine top-K ids within `pool` (ids are `ordinal + 1`).
fn midpoint_queries(
    rows: &[Vec<f32>],
    pool: &[usize],
    count: usize,
    seed: u64,
) -> Vec<(Vec<f32>, Vec<u128>)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let a = &rows[pool[rng.random_range(0..pool.len())]];
            let b = &rows[pool[rng.random_range(0..pool.len())]];
            let query = normalized(a.iter().zip(b).map(|(x, y)| x + y).collect());
            let mut scored: Vec<(usize, f32)> = pool
                .iter()
                .map(|&ord| (ord, cosine_similarity(&query, &rows[ord])))
                .collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1));
            let truth = scored[..K]
                .iter()
                .map(|&(ord, _)| ord as u128 + 1)
                .collect();
            (query, truth)
        })
        .collect()
}

struct BuiltSegment {
    backend: Arc<AnyBackend>,
    /// The writer's store and path, for opening reader replicas.
    store: Arc<dyn ObjectStore>,
    path: String,
    core: VectorCore,
    _dir: TempDir,
    build_secs: f64,
    neighbor_ids: usize,
    level0_degree: f64,
    max_level0: usize,
    max_upper: usize,
    /// Points no level-0 walk from the entry point can reach.
    unreachable: usize,
}

/// Flat-insert `rows` into an in-memory LSM segment with the production dense
/// settings (cosine, no spindle) and build HNSW the way the LSM index job does:
/// `prepare_index_from_flat` -> `flush_prepared_index_chunk_be` ->
/// `finalize_prepared_index_entry_be`.
fn build_segment(
    name: &str,
    hnsw: HNSWConfig,
    rows: &[Vec<f32>],
    diverse: Option<bool>,
) -> BuiltSegment {
    build_segment_stranding(name, hnsw, rows, diverse, &HashSet::new())
}

/// `build_segment` with every link to `strand` (bar the entry point) cut at
/// every level before the flush, so no walk reaches those points.
fn build_segment_stranding(
    name: &str,
    hnsw: HNSWConfig,
    rows: &[Vec<f32>],
    diverse: Option<bool>,
    strand: &HashSet<u128>,
) -> BuiltSegment {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = format!("{name}-{}", uuid::Uuid::new_v4());
    let backend = Arc::new(AnyBackend::Lsm(
        LsmBackend::open_with_store(&path, Arc::clone(&store)).unwrap(),
    ));
    let dir = TempDir::new().unwrap();
    let spindle = SpindleConfig {
        mode: SpindleMode::None,
        keep_original: false,
        rescore: false,
        oversampling: 1,
        binary_dims: 1,
        turbo_dims: 1,
    };
    let core = VectorCore::new_named_lsm_with_dir(
        name,
        hnsw,
        DistanceMetric::Cosine,
        spindle,
        Some(dir.path()),
        rows[0].len(),
        Arc::clone(&backend),
    )
    .unwrap();
    for (chunk_index, chunk) in rows.chunks(4096).enumerate() {
        let mut w = backend.begin_write().unwrap();
        for (i, row) in chunk.iter().enumerate() {
            let id = (chunk_index * 4096 + i + 1) as u128;
            core.insert_flat_be(&mut w, row, Some(id), None).unwrap();
        }
        backend.commit(w).unwrap();
    }

    let ro = backend.begin_read().unwrap();
    let started = std::time::Instant::now();
    let mut prepared =
        with_diverse_neighbors_for_test(diverse, || core.prepare_index_from_flat(&ro))
            .unwrap()
            .unwrap();
    let build_secs = started.elapsed().as_secs_f64();
    drop(ro);
    let stranded: HashSet<u32> = (0..prepared.point_ids.len())
        .filter(|&ord| ord != prepared.entry_ord && strand.contains(&prepared.point_ids[ord]))
        .map(|ord| ord as u32)
        .collect();
    for levels in prepared.adjacency.iter_mut().flatten() {
        for list in levels.iter_mut() {
            list.retain(|ord| !stranded.contains(ord));
        }
    }

    let lists = prepared
        .adjacency
        .iter()
        .map(|levels| levels.as_ref().unwrap());
    let neighbor_ids = lists.clone().flatten().map(Vec::len).sum();
    let level0: usize = lists.clone().map(|levels| levels[0].len()).sum();
    let max_level0 = lists.clone().map(|levels| levels[0].len()).max().unwrap();
    let max_upper = lists
        .flat_map(|levels| levels[1..].iter().map(Vec::len))
        .max()
        .unwrap_or(0);
    let mut seen = vec![false; prepared.adjacency.len()];
    let mut stack = vec![prepared.entry_ord];
    seen[prepared.entry_ord] = true;
    while let Some(ord) = stack.pop() {
        for &next in &prepared.adjacency[ord].as_ref().unwrap()[0] {
            if !std::mem::replace(&mut seen[next as usize], true) {
                stack.push(next as usize);
            }
        }
    }
    let unreachable = seen.iter().filter(|&&reached| !reached).count();

    let mut w = backend.begin_write().unwrap();
    let total = prepared.point_ids.len();
    core.flush_prepared_index_chunk_be(&mut w, &mut prepared, 0, total)
        .unwrap();
    core.finalize_prepared_index_entry_be(&mut w, &prepared)
        .unwrap();
    backend.commit(w).unwrap();

    BuiltSegment {
        backend,
        store,
        path,
        core,
        _dir: dir,
        build_secs,
        neighbor_ids,
        level0_degree: level0 as f64 / rows.len() as f64,
        max_level0,
        max_upper,
        unreachable,
    }
}

/// A reader replica's core over `segment`'s committed state.
fn reader_core(segment: &BuiltSegment, dim: usize) -> (VectorCore, TempDir) {
    reader_at(&segment.core, &segment.store, &segment.path, dim)
}

/// A reader replica's core over the committed state of the writer `core`,
/// whose backend was opened on `store` at `path`.
fn reader_at(
    core: &VectorCore,
    store: &Arc<dyn ObjectStore>,
    path: &str,
    dim: usize,
) -> (VectorCore, TempDir) {
    let dir = TempDir::new().unwrap();
    let reader = reader_in(core, store, path, dim, dir.path());
    (reader, dir)
}

/// `reader_at` with the reader's local directory given: a replica opened
/// again over the files an earlier one left there.
fn reader_in(
    core: &VectorCore,
    store: &Arc<dyn ObjectStore>,
    path: &str,
    dim: usize,
    dir: &std::path::Path,
) -> VectorCore {
    VectorCore::new_named_lsm_with_dir(
        &core.physical_name,
        core.config.clone(),
        core.distance_metric.clone(),
        core.spindle.clone(),
        Some(dir),
        dim,
        Arc::new(AnyBackend::LsmReader(
            LsmReader::open_with_store(path, Arc::clone(store)).unwrap(),
        )),
    )
    .unwrap()
}

struct SearchStats {
    recalls: Vec<f64>,
    /// Means per query, all levels, from the search's own
    /// `helix_vector_core_search_items` counts.
    visited: f64,
    neighbor_blocks: f64,
    millis: f64,
}

/// Per-query recall@K at `ef`, optionally through an id filter with its
/// selectivity hint (the named-vector layer's filtered dense path).
fn recall_at(
    core: &VectorCore,
    queries: &[(Vec<f32>, Vec<u128>)],
    ef: usize,
    allowed: Option<(&HashSet<u128>, f32)>,
) -> SearchStats {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let labels = VectorSearchMetrics {
        collection: "eval",
        vector: "dense",
        segment_count: 1,
        segment_index: 0,
        filtered: allowed.is_some(),
    };
    let pred = |id: u128| allowed.is_none_or(|(set, _)| set.contains(&id));
    let filter = allowed.map(|_| std::slice::from_ref(&pred));
    let ro = core.backend.begin_read().unwrap();
    let started = std::time::Instant::now();
    let recalls = metrics::with_local_recorder(&recorder, || {
        queries
            .iter()
            .map(|(query, truth)| {
                let hits = core
                    .search_with_id_filter_ef_observed(
                        &ro,
                        query,
                        K,
                        filter,
                        true,
                        allowed.map(|(_, selectivity)| selectivity),
                        Some(ef),
                        Some(labels),
                    )
                    .unwrap();
                hits.iter()
                    .filter(|hit| truth.contains(&hit.get_id()))
                    .count() as f64
                    / K as f64
            })
            .collect::<Vec<_>>()
    });
    let millis = started.elapsed().as_secs_f64() * 1000.0 / queries.len() as f64;
    let rendered = handle.render();
    let per_query = |measurement: &str| {
        let label = format!("measurement=\"{measurement}\"");
        rendered
            .lines()
            .filter(|line| {
                line.starts_with("helix_vector_core_search_items_sum{") && line.contains(&label)
            })
            .filter_map(|line| line.rsplit(' ').next()?.parse::<f64>().ok())
            .sum::<f64>()
            / queries.len() as f64
    };
    SearchStats {
        visited: per_query("visited"),
        neighbor_blocks: per_query("neighbor_blocks"),
        millis,
        recalls,
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Tight topic clusters (code embeddings cluster hard by file and topic):
/// nearest-M links stay inside clusters, so between-cluster queries strand.
#[test]
fn diverse_neighbors_lift_recall_on_clustered_data() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(41);
    let (dim, topics, spread) = (32usize, 60usize, 0.05f32);
    let centers: Vec<Vec<f32>> = (0..topics)
        .map(|_| normalized((0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect()))
        .collect();
    let rows: Vec<Vec<f32>> = (0..3000)
        .map(|_| {
            let center = &centers[rng.random_range(0..topics)];
            normalized(
                center
                    .iter()
                    .map(|c| c + spread * rng.random_range(-1.0f32..1.0))
                    .collect(),
            )
        })
        .collect();
    let all: Vec<usize> = (0..rows.len()).collect();
    let queries = midpoint_queries(&rows, &all, 60, 43);
    let hnsw = HNSWConfig::new(Some(16), Some(64), Some(128));

    let mut recall = HashMap::new();
    for (label, diverse) in [("off", None), ("on", Some(HNSW_KEEP_PRUNED_CONNECTIONS))] {
        let segment = build_segment(&format!("clustered_{label}"), hnsw.clone(), &rows, diverse);
        assert!(segment.max_level0 <= hnsw.m_max_0 && segment.max_upper <= hnsw.m);
        let recalls = recall_at(&segment.core, &queries, 128, None).recalls;
        println!("clustered {label}: recall@{K} mean {:.3}", mean(&recalls));
        recall.insert(label, mean(&recalls));
    }
    // Graph builds are random (level draws, parallel insert order); over
    // 40 review runs OFF stayed in 0.33-0.56 and ON in 0.73-0.88.
    assert!(
        recall["on"] >= 0.65 && recall["on"] >= recall["off"] + 0.15,
        "{recall:?}"
    );
}

/// Ids (`ordinal + 1`) of `rows` passing `keep`, nearest `query` first.
fn ranked_ids(rows: &[Vec<f32>], query: &[f32], keep: impl Fn(u128) -> bool) -> Vec<u128> {
    let mut scored: Vec<(u128, f32)> = rows
        .iter()
        .enumerate()
        .map(|(ord, row)| (ord as u128 + 1, cosine_similarity(query, row)))
        .filter(|&(id, _)| keep(id))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.into_iter().map(|(id, _)| id).collect()
}

/// Sorted hit ids of the unfiltered, the payload-filtered and the
/// id-filtered route, then of the id route with no filter (the entry an
/// unfiltered request takes); both filters keep the even ids.
fn route_ids(core: &VectorCore, query: &[f32]) -> [Vec<u128>; 4] {
    let even = |id: u128| id % 2 == 0;
    let even_vector = |v: &HVector| v.get_id() % 2 == 0;
    let ro = core.backend.begin_read().unwrap();
    let ids = |hits: Vec<HVector>| {
        let mut ids: Vec<u128> = hits.iter().map(HVector::get_id).collect();
        ids.sort();
        ids
    };
    [
        ids(core
            .search_with_selectivity_ef_observed::<fn(&HVector) -> bool>(
                &ro,
                query,
                K,
                None,
                false,
                None,
                Some(128),
                None,
            )
            .unwrap()),
        ids(core
            .search_with_selectivity_ef_observed(
                &ro,
                query,
                K,
                Some(&[even_vector]),
                true,
                Some(0.5),
                Some(128),
                None,
            )
            .unwrap()),
        ids(core
            .search_with_id_filter_ef_observed(
                &ro,
                query,
                K,
                Some(&[even]),
                true,
                Some(0.5),
                Some(128),
                None,
            )
            .unwrap()),
        ids(core
            .search_with_id_filter_ef_observed::<fn(u128) -> bool>(
                &ro,
                query,
                K,
                None,
                false,
                None,
                Some(128),
                None,
            )
            .unwrap()),
    ]
}

/// A small HNSW segment whose nearest points to the query no list links to
/// (the islands nearest-`M` leaves on clustered code vectors): the
/// unfiltered, payload-filtered and id-filtered routes return the exact
/// top-K, also after a tombstone and a hard delete, and
/// `HELIX_SMALL_SEGMENT_EXACT=0` restores the walk that misses them.
#[test]
fn small_hnsw_segments_search_exactly_past_unreachable_points() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(29);
    let mut random_unit = || normalized((0..16).map(|_| rng.random_range(-1.0f32..1.0)).collect());
    let rows: Vec<Vec<f32>> = (0..300).map(|_| random_unit()).collect();
    let query = random_unit();
    let even = |id: u128| id % 2 == 0;
    let nearest = ranked_ids(&rows, &query, |_| true);
    let nearest_even = ranked_ids(&rows, &query, even);
    let strand: HashSet<u128> = nearest[..2 * K]
        .iter()
        .chain(&nearest_even[..2 * K])
        .copied()
        .collect();
    let segment = build_segment_stranding(
        "small_exact",
        HNSWConfig::new(Some(16), Some(64), Some(128)),
        &rows,
        None,
        &strand,
    );
    assert!(segment.unreachable >= strand.len() - 1);
    let (reader_core, _reader_dir) = reader_core(&segment, rows[0].len());

    let routes = |core: &VectorCore| route_ids(core, &query);
    let expected = |deleted: &[u128]| {
        let top = |ranked: &[u128]| {
            let mut ids: Vec<u128> = ranked
                .iter()
                .copied()
                .filter(|id| !deleted.contains(id))
                .take(K)
                .collect();
            ids.sort();
            ids
        };
        [
            top(&nearest),
            top(&nearest_even),
            top(&nearest_even),
            top(&nearest),
        ]
    };

    for core in [&segment.core, &reader_core] {
        for hits in with_small_segment_exact_for_test(false, || routes(core)) {
            assert!(
                hits.len() == K && hits.iter().filter(|id| strand.contains(id)).count() <= 1,
                "the walk returns K hits, none stranded bar the entry point: {hits:?}"
            );
        }
        assert_eq!(
            with_small_segment_exact_for_test(true, || routes(core)),
            expected(&[])
        );
    }

    // A deferred-repair (tombstone) delete and a hard delete on the writer.
    let (tombstoned, hard_deleted) = (nearest[0], nearest_even[1]);
    segment.core.apply_delete_tombstones(&[tombstoned]);
    let mut w = segment.backend.begin_write().unwrap();
    let levels = segment
        .core
        .delete_vectors_batch_be(&mut w, &[hard_deleted])
        .unwrap();
    segment.backend.commit(w).unwrap();
    segment
        .core
        .apply_hard_deletes_post_commit(&[hard_deleted], &levels);
    assert_eq!(
        with_small_segment_exact_for_test(true, || routes(&segment.core)),
        expected(&[tombstoned, hard_deleted])
    );
}

/// The exact search's row limit at its default
/// (`HELIX_SMALL_SEGMENT_EXACT_MAX_ROWS`): an unfiltered search of a segment
/// holding exactly the limit returns the exact top-K, on the writer and on a
/// reader, although no list links to those points, and with one row more the
/// graph is walked. A filtered search keeps the rule from before the limit:
/// exact below `HNSW_DIVERSE_MIN_POINTS` rows, walked from there on.
#[test]
fn hnsw_segment_at_the_exact_row_limit_is_scored_and_one_row_more_is_walked() {
    assert_eq!(
        small_segment_exact_max_rows(),
        SMALL_SEGMENT_EXACT_MAX_ROWS,
        "run without HELIX_SMALL_SEGMENT_EXACT_MAX_ROWS"
    );
    let limit = SMALL_SEGMENT_EXACT_MAX_ROWS as usize;
    let mut rng = rand::rngs::StdRng::seed_from_u64(31);
    let mut random_unit = || normalized((0..16).map(|_| rng.random_range(-1.0f32..1.0)).collect());
    let all_rows: Vec<Vec<f32>> = (0..=limit).map(|_| random_unit()).collect();
    let query = random_unit();
    let more_queries: Vec<Vec<f32>> = (0..4).map(|_| random_unit()).collect();
    let even = |id: u128| id % 2 == 0;
    let top = |ranked: &[u128]| {
        let mut ids = ranked[..K].to_vec();
        ids.sort();
        ids
    };

    // (rows, unfiltered search is exact, filtered search is exact)
    for (count, unfiltered_exact, filtered_exact) in [
        (HNSW_DIVERSE_MIN_POINTS - 1, true, true),
        (HNSW_DIVERSE_MIN_POINTS, true, false),
        (limit, true, false),
        (limit + 1, false, false),
    ] {
        let rows = &all_rows[..count];
        let nearest = ranked_ids(rows, &query, |_| true);
        let nearest_even = ranked_ids(rows, &query, even);
        let strand: HashSet<u128> = nearest[..2 * K]
            .iter()
            .chain(&nearest_even[..2 * K])
            .copied()
            .collect();
        let segment = build_segment_stranding(
            "exact_row_limit",
            HNSWConfig::new(Some(16), Some(64), Some(128)),
            rows,
            None,
            &strand,
        );
        let (reader, _reader_dir) = reader_core(&segment, rows[0].len());
        for (role, core) in [("writer", &segment.core), ("reader", &reader)] {
            let [unfiltered, by_payload, by_id, unfiltered_by_id] =
                with_small_segment_exact_for_test(true, || route_ids(core, &query));
            for (route, hits, exact, expected) in [
                ("unfiltered", unfiltered, unfiltered_exact, top(&nearest)),
                (
                    "payload filter",
                    by_payload,
                    filtered_exact,
                    top(&nearest_even),
                ),
                ("id filter", by_id, filtered_exact, top(&nearest_even)),
                (
                    "id route, no filter",
                    unfiltered_by_id,
                    unfiltered_exact,
                    top(&nearest),
                ),
            ] {
                if exact {
                    assert_eq!(hits, expected, "{count} rows, {role}, {route}");
                } else {
                    assert!(
                        hits.len() == K
                            && hits.iter().filter(|id| strand.contains(id)).count() <= 1,
                        "{count} rows, {role}, {route}: the walk reaches no stranded point \
                         bar the entry point: {hits:?}"
                    );
                }
            }
            if count == limit {
                for query in &more_queries {
                    let [unfiltered, .., unfiltered_by_id] =
                        with_small_segment_exact_for_test(true, || route_ids(core, query));
                    let expected = top(&ranked_ids(rows, query, |_| true));
                    assert_eq!(
                        (&unfiltered, &unfiltered_by_id),
                        (&expected, &expected),
                        "{count} rows, {role}: recall of the exact search"
                    );
                }
            }
        }
    }
}

/// The limit's setting: a number is taken as given down to the rule from
/// before it (a lower one would walk the graphs that strand points), and
/// anything else is the default.
#[test]
fn exact_row_limit_setting_takes_a_number_no_lower_than_the_old_rule() {
    assert_eq!(small_segment_exact_max_rows_from_raw(None), 2048);
    assert_eq!(small_segment_exact_max_rows_from_raw(Some(" 4096\n")), 4096);
    assert_eq!(small_segment_exact_max_rows_from_raw(Some("-1")), 2048);
    for low in ["511", "100", "0"] {
        assert_eq!(small_segment_exact_max_rows_from_raw(Some(low)), 511);
    }
}

/// Real prod dense vectors: `<name>.f32` files (768-d f32, row-major, no
/// header) in `HNSW_EVAL_DIR` (default `/tmp/hnsw-vecs`).
fn eval_inputs() -> impl Iterator<Item = (String, Vec<Vec<f32>>)> {
    const DIM: usize = 768;
    let dir = std::env::var("HNSW_EVAL_DIR").unwrap_or_else(|_| "/tmp/hnsw-vecs".into());
    let mut files: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap()
        .filter_map(|entry| Some(entry.ok()?.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "f32"))
        .collect();
    files.sort();
    files.into_iter().map(|path| {
        let bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.len() % (DIM * 4), 0, "{path:?}");
        let rows = bytes
            .chunks_exact(DIM * 4)
            .map(|row| {
                row.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            })
            .collect();
        (
            path.file_stem().unwrap().to_string_lossy().to_string(),
            rows,
        )
    })
}

/// `HNSW_EVAL_BUILDS` repeats each build (level draws and parallel insert
/// order are random).
fn eval_builds() -> usize {
    std::env::var("HNSW_EVAL_BUILDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
}

const EVAL_MODES: [(&str, Option<bool>); 3] = [
    ("off", None),
    ("on-strict", Some(false)),
    ("on-fill", Some(true)),
];

/// The client creates collections with hnsw_config {m: 16,
/// ef_construct: 256}; search ef defaults to 128.
fn eval_hnsw() -> HNSWConfig {
    HNSWConfig::new(Some(16), Some(256), Some(128))
}

/// Offline eval over `eval_inputs`, unfiltered. Run with
/// `cargo test -p helixdb --lib hnsw_diversity_tests::eval -- --ignored --nocapture`.
#[test]
#[ignore = "offline eval; needs HNSW_EVAL_DIR"]
fn eval_diverse_neighbors_on_real_vectors() {
    let builds = eval_builds();
    let hnsw = eval_hnsw();
    let modes = EVAL_MODES;
    println!(
        "config m={} m_max_0={} ef_construct={} (bulk ef_construction={}) builds={builds}",
        hnsw.m,
        hnsw.m_max_0,
        hnsw.ef_construct,
        (hnsw.ef_construct / 2).max(32)
    );
    for (name, rows) in eval_inputs() {
        let all: Vec<usize> = (0..rows.len()).collect();
        let queries = midpoint_queries(&rows, &all, 200, 20261101);
        let mut pooled: HashMap<(&str, usize), Vec<SearchStats>> = HashMap::new();
        let mut build_stats: HashMap<&str, Vec<(f64, usize, f64, usize)>> = HashMap::new();
        for build in 0..builds {
            for &(label, diverse) in &modes {
                let segment = build_segment("eval", hnsw.clone(), &rows, diverse);
                assert!(segment.max_level0 <= hnsw.m_max_0 && segment.max_upper <= hnsw.m);
                build_stats.entry(label).or_default().push((
                    segment.build_secs,
                    segment.neighbor_ids,
                    segment.level0_degree,
                    segment.unreachable,
                ));
                for ef in [64, 128, 256] {
                    let stats = recall_at(&segment.core, &queries, ef, None);
                    println!(
                        "BUILD {name} n={} build={build} {label} ef={ef}: recall {:.4} visited {:.0}",
                        rows.len(),
                        mean(&stats.recalls),
                        stats.visited
                    );
                    pooled.entry((label, ef)).or_default().push(stats);
                }
            }
        }
        for &(label, _) in &modes {
            let stats = &build_stats[label];
            let secs: Vec<f64> = stats.iter().map(|s| s.0).collect();
            let ids: Vec<f64> = stats.iter().map(|s| s.1 as f64).collect();
            let degree: Vec<f64> = stats.iter().map(|s| s.2).collect();
            let unreachable: Vec<usize> = stats.iter().map(|s| s.3).collect();
            println!(
                "GRAPH {name} n={} {label}: build_secs {:.2} {secs:.2?} neighbor_ids {:.0} level0_degree {:.2} unreachable {unreachable:?}",
                rows.len(),
                mean(&secs),
                mean(&ids),
                mean(&degree)
            );
            for ef in [64, 128, 256] {
                let runs = &pooled[&(label, ef)];
                let build_means: Vec<f64> = runs.iter().map(|run| mean(&run.recalls)).collect();
                let mut sorted: Vec<f64> =
                    runs.iter().flat_map(|run| run.recalls.clone()).collect();
                sorted.sort_by(f64::total_cmp);
                let avg = |field: fn(&SearchStats) -> f64| {
                    mean(&runs.iter().map(field).collect::<Vec<_>>())
                };
                println!(
                    "ROW {name} {label} ef={ef}: mean {:.4} p10 {:.2} min {:.2} visited {:.0} blocks {:.0} ms {:.3} per-build {build_means:.4?}",
                    mean(&build_means),
                    sorted[sorted.len() / 10],
                    sorted[0],
                    avg(|run| run.visited),
                    avg(|run| run.neighbor_blocks),
                    avg(|run| run.millis)
                );
            }
        }
    }
}

/// Filtered dense search (the shape of the client's repo/path filters) and
/// tiny segments over `eval_inputs`. `rand10` keeps a random 10% of ids,
/// `near10` the 10% nearest a random anchor (one directory); filtered queries
/// are drawn from inside the filter.
#[test]
#[ignore = "offline eval; needs HNSW_EVAL_DIR"]
fn eval_diverse_neighbors_filtered_and_small() {
    let hnsw = eval_hnsw();
    for (name, rows) in eval_inputs() {
        let n = rows.len();
        let mut rng = rand::rngs::StdRng::seed_from_u64(20261102);
        let rand10: Vec<usize> = (0..n).filter(|_| rng.random_range(0..10) == 0).collect();
        let anchor = rows[rng.random_range(0..n)].clone();
        let mut near10: Vec<usize> = (0..n).collect();
        near10.sort_by(|&a, &b| {
            cosine_similarity(&anchor, &rows[b]).total_cmp(&cosine_similarity(&anchor, &rows[a]))
        });
        near10.truncate(n / 10);
        let filters = [("rand10", rand10), ("near10", near10)].map(|(label, pool)| {
            let allowed: HashSet<u128> = pool.iter().map(|&ord| ord as u128 + 1).collect();
            (
                label,
                allowed,
                midpoint_queries(&rows, &pool, 100, 20261103),
            )
        });
        let smalls = [100usize, 300].map(|size| {
            let small = rows[..size].to_vec();
            let all: Vec<usize> = (0..size).collect();
            let queries = midpoint_queries(&small, &all, 100, 20261104);
            (small, queries)
        });
        for build in 0..eval_builds() {
            for &(mode, diverse) in &EVAL_MODES {
                let segment = build_segment("eval", hnsw.clone(), &rows, diverse);
                for (label, allowed, queries) in &filters {
                    let selectivity = allowed.len() as f32 / n as f32;
                    let stats =
                        recall_at(&segment.core, queries, 128, Some((allowed, selectivity)));
                    println!(
                        "FILTER {name} {label} build={build} {mode} ef=128: mean {:.4} min {:.2} visited {:.0}",
                        mean(&stats.recalls),
                        stats.recalls.iter().copied().fold(f64::INFINITY, f64::min),
                        stats.visited
                    );
                }
                drop(segment);
                for (small, queries) in &smalls {
                    let segment = build_segment("eval_small", hnsw.clone(), small, diverse);
                    let stats = recall_at(&segment.core, queries, 128, None);
                    println!(
                        "SMALL {name} n={} build={build} {mode} ef=128: mean {:.4} min {:.2} unreachable {} level0_degree {:.2}",
                        small.len(),
                        mean(&stats.recalls),
                        stats.recalls.iter().copied().fold(f64::INFINITY, f64::min),
                        segment.unreachable,
                        segment.level0_degree
                    );
                }
            }
        }
    }
}

/// Small segments over `eval_inputs` (the first `n` rows, and a random `n`
/// per build), unfiltered at ef=128, on the writer and on a fresh reader
/// replica: the walk (`HELIX_SMALL_SEGMENT_EXACT=0`) against the exact
/// small-segment search, which scores all `n` rows and reads no neighbor
/// blocks. `cold_ms` is the first pass over a core, `ms` the second. Run with
/// `cargo test -p helixdb --lib hnsw_diversity_tests::eval_small -- --ignored --nocapture`.
#[test]
#[ignore = "offline eval; needs HNSW_EVAL_DIR"]
fn eval_small_segment_exact_on_real_vectors() {
    let hnsw = eval_hnsw();
    for (name, rows) in eval_inputs() {
        for size in [300usize, 511] {
            for build in 0..eval_builds() {
                let mut rng = rand::rngs::StdRng::seed_from_u64(1000 + build as u64);
                let picks = [
                    ("head", (0..size).collect::<Vec<_>>()),
                    (
                        "random",
                        rand::seq::index::sample(&mut rng, rows.len(), size).into_vec(),
                    ),
                ];
                for (pick, ords) in picks {
                    let small: Vec<Vec<f32>> = ords.iter().map(|&ord| rows[ord].clone()).collect();
                    let all: Vec<usize> = (0..size).collect();
                    let queries = midpoint_queries(&small, &all, 200, 20261104 + build as u64);
                    let segment = build_segment("eval_small", hnsw.clone(), &small, None);
                    for exact in [false, true] {
                        let (reader, _reader_dir) = reader_core(&segment, small[0].len());
                        for (role, core) in [("writer", &segment.core), ("reader", &reader)] {
                            let run = || recall_at(core, &queries, 128, None);
                            let cold = with_small_segment_exact_for_test(exact, run);
                            let stats = with_small_segment_exact_for_test(exact, run);
                            println!(
                                "SMALLX {name} {pick} n={size} build={build} {role} exact={exact} ef=128: mean {:.4} min {:.2} visited {:.0} blocks {:.0} ms {:.3} cold_ms {:.3} unreachable {}",
                                mean(&stats.recalls),
                                stats.recalls.iter().copied().fold(f64::INFINITY, f64::min),
                                stats.visited,
                                stats.neighbor_blocks,
                                stats.millis,
                                cold.millis,
                                segment.unreachable
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Writer-built HNSW segment on `store` (prod dense settings, hash-spread
/// ids like UUID-derived point ids), memtable flushed and the writer closed
/// so a reader serves it from SSTs. `writer_dir: false` builds with no
/// sidecar dir, so no per-id `Ordinals` rows are written.
fn build_closed_segment(
    store: Arc<dyn ObjectStore>,
    path: &str,
    name: &str,
    rows: &[Vec<f32>],
    writer_dir: bool,
) {
    let backend = Arc::new(AnyBackend::Lsm(
        LsmBackend::open_with_store(path, store).unwrap(),
    ));
    let dir = TempDir::new().unwrap();
    let core = VectorCore::new_named_lsm_with_dir(
        name,
        HNSWConfig::new(Some(16), Some(64), Some(128)),
        DistanceMetric::Cosine,
        SpindleConfig {
            mode: SpindleMode::None,
            keep_original: false,
            rescore: false,
            oversampling: 1,
            binary_dims: 1,
            turbo_dims: 1,
        },
        writer_dir.then(|| dir.path()),
        rows[0].len(),
        Arc::clone(&backend),
    )
    .unwrap();
    let mut w = backend.begin_write().unwrap();
    for (i, row) in rows.iter().enumerate() {
        let id = (i as u128 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835) | 1;
        core.insert_flat_be(&mut w, row, Some(id), None).unwrap();
    }
    backend.commit(w).unwrap();
    let ro = backend.begin_read().unwrap();
    let mut prepared = core.prepare_index_from_flat(&ro).unwrap().unwrap();
    drop(ro);
    let mut w = backend.begin_write().unwrap();
    let total = prepared.point_ids.len();
    core.flush_prepared_index_chunk_be(&mut w, &mut prepared, 0, total)
        .unwrap();
    core.finalize_prepared_index_entry_be(&mut w, &prepared)
        .unwrap();
    backend.commit(w).unwrap();
    drop(core);
    let AnyBackend::Lsm(lsm) = &*backend else {
        unreachable!()
    };
    lsm.flush_memtable_and_close_for_test();
}

/// A small HNSW segment whose writer stored no per-id `Ordinals` rows (a
/// core without a sidecar dir) served by a reader replica that has one: the
/// exact path must return what the walk returns, on every query, not just
/// the first one (the first exact query attaches the reader's sidecar).
#[test]
fn reader_exact_without_persisted_ordinals_matches_walk() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(5);
    let rows: Vec<Vec<f32>> = (0..300)
        .map(|_| normalized((0..16).map(|_| rng.random_range(-1.0f32..1.0)).collect()))
        .collect();
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = format!("noord-{}", uuid::Uuid::new_v4());
    let name = "dense__seg_000002";
    build_closed_segment(Arc::clone(&store), &path, name, &rows, false);
    let dir = TempDir::new().unwrap();
    let reader = VectorCore::new_named_lsm_with_dir(
        name,
        HNSWConfig::new(Some(16), Some(64), Some(128)),
        DistanceMetric::Cosine,
        SpindleConfig {
            mode: SpindleMode::None,
            keep_original: false,
            rescore: false,
            oversampling: 1,
            binary_dims: 1,
            turbo_dims: 1,
        },
        Some(dir.path()),
        16,
        Arc::new(AnyBackend::LsmReader(
            LsmReader::open_with_store(&path, store).unwrap(),
        )),
    )
    .unwrap();
    let ro = reader.backend.begin_read().unwrap();
    for (i, query) in rows.iter().take(4).enumerate() {
        let walk = with_small_segment_exact_for_test(false, || {
            reader.search_with_id_filter_ef_observed::<fn(u128) -> bool>(
                &ro,
                query,
                K,
                None,
                false,
                None,
                Some(128),
                None,
            )
        })
        .unwrap();
        let exact = with_small_segment_exact_for_test(true, || {
            reader.search_with_id_filter_ef_observed::<fn(u128) -> bool>(
                &ro,
                query,
                K,
                None,
                false,
                None,
                Some(128),
                None,
            )
        })
        .unwrap();
        let unfiltered = with_small_segment_exact_for_test(true, || {
            reader.search_with_selectivity_ef_observed::<fn(&HVector) -> bool>(
                &ro,
                query,
                K,
                None,
                false,
                None,
                Some(128),
                None,
            )
        })
        .unwrap();
        assert_eq!(walk.len(), K, "query {i}: walk");
        assert_eq!(exact.len(), K, "query {i}: exact id-filter route lost rows");
        assert_eq!(
            unfiltered.len(),
            K,
            "query {i}: exact unfiltered route lost rows"
        );
    }
}

/// Process-level kill switch: no thread-local override, so the
/// `HELIX_SMALL_SEGMENT_EXACT` env var decides which path serves a small
/// HNSW segment (flat-search metric mode) and whether its stranded nearest
/// points are found. Run once with `HELIX_SMALL_SEGMENT_EXACT=0` to check the
/// walk comes back.
#[test]
fn env_kill_switch_selects_path() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(29);
    let mut random_unit = || normalized((0..16).map(|_| rng.random_range(-1.0f32..1.0)).collect());
    let rows: Vec<Vec<f32>> = (0..300).map(|_| random_unit()).collect();
    let query = random_unit();
    let nearest = ranked_ids(&rows, &query, |_| true);
    let strand: HashSet<u128> = nearest[..2 * K].iter().copied().collect();
    let segment = build_segment_stranding(
        "env_kill_switch",
        HNSWConfig::new(Some(16), Some(64), Some(128)),
        &rows,
        None,
        &strand,
    );
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let labels = VectorSearchMetrics {
        collection: "env_kill_switch",
        vector: "dense",
        segment_count: 1,
        segment_index: 0,
        filtered: false,
    };
    let ro = segment.backend.begin_read().unwrap();
    let hits = metrics::with_local_recorder(&recorder, || {
        segment
            .core
            .search_with_id_filter_ef_observed::<fn(u128) -> bool>(
                &ro,
                &query,
                K,
                None,
                false,
                None,
                Some(128),
                Some(labels),
            )
            .unwrap()
    });
    let stranded_found = hits.iter().filter(|h| strand.contains(&h.get_id())).count();
    let exact_mode = handle
        .render()
        .lines()
        .any(|line| line.contains("mode=\"small_hnsw_id_filter\""));
    let env = std::env::var("HELIX_SMALL_SEGMENT_EXACT").unwrap_or_else(|_| "<unset>".into());
    println!("HELIX_SMALL_SEGMENT_EXACT={env}: exact_mode={exact_mode} stranded_found={stranded_found}/{K}");
    if env.trim() == "0" {
        assert!(
            !exact_mode && stranded_found <= 1,
            "=0 must restore the walk"
        );
        assert_eq!(
            segment
                .core
                .hnsw_rows_upto_small
                .load(AtomicOrdering::Relaxed),
            u64::MAX
        );
    } else {
        assert!(
            exact_mode && stranded_found == K,
            "default must search exactly"
        );
    }
}

/// A segment as its writer holds it, with the store and path a reader
/// replica opens.
struct Written {
    core: VectorCore,
    _dir: TempDir,
    store: Arc<dyn ObjectStore>,
    path: String,
}

impl Written {
    /// A reader replica's core over what is committed now.
    fn reader(&self, dim: usize) -> (VectorCore, TempDir) {
        reader_at(&self.core, &self.store, &self.path, dim)
    }

    /// A reader replica opened again in `dir`, over the files an earlier one
    /// left there.
    fn reader_in(&self, dim: usize, dir: &std::path::Path) -> VectorCore {
        reader_in(&self.core, &self.store, &self.path, dim, dir)
    }

    /// A hard delete as the writer applies it: the batch, its commit, then
    /// the post-commit cache step. The rows go, and their own lists stay or
    /// go with them as `with_deleted_lists_for_test` says (they stay unless
    /// it says otherwise); the lists that name them stay as they are.
    fn hard_delete(&self, ids: &[u128]) {
        let mut w = self.core.backend.begin_write().unwrap();
        let levels = self.core.delete_vectors_batch_be(&mut w, ids).unwrap();
        self.core.backend.commit(w).unwrap();
        self.core.apply_hard_deletes_post_commit(ids, &levels);
    }
}

/// A segment written from the lists it is given (`adjacency[ordinal][level]`,
/// in ordinals; ids are `ordinal + 1`) and searched like any segment.
fn write_segment_with_lists(
    name: &str,
    hnsw: &HNSWConfig,
    rows: &[Vec<f32>],
    levels: &[usize],
    adjacency: Vec<Vec<Vec<u32>>>,
    entry_ord: usize,
) -> Written {
    write_segment_with_lists_under(
        DistanceMetric::Cosine,
        name,
        hnsw,
        rows,
        levels,
        adjacency,
        entry_ord,
    )
}

/// As `write_segment_with_lists`, its rows compared by `metric`.
fn write_segment_with_lists_under(
    metric: DistanceMetric,
    name: &str,
    hnsw: &HNSWConfig,
    rows: &[Vec<f32>],
    levels: &[usize],
    adjacency: Vec<Vec<Vec<u32>>>,
    entry_ord: usize,
) -> Written {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = format!("{name}-{}", uuid::Uuid::new_v4());
    let backend = Arc::new(AnyBackend::Lsm(
        LsmBackend::open_with_store(&path, Arc::clone(&store)).unwrap(),
    ));
    let dir = TempDir::new().unwrap();
    let spindle = SpindleConfig {
        mode: SpindleMode::None,
        keep_original: false,
        rescore: false,
        oversampling: 1,
        binary_dims: 1,
        turbo_dims: 1,
    };
    let (n, dim) = (rows.len(), rows[0].len());
    let core = VectorCore::new_named_lsm_with_dir(
        name,
        hnsw.clone(),
        metric,
        spindle,
        Some(dir.path()),
        dim,
        Arc::clone(&backend),
    )
    .unwrap();
    let mut w = backend.begin_write().unwrap();
    for (ord, row) in rows.iter().enumerate() {
        core.insert_flat_be(&mut w, row, Some(ord as u128 + 1), None)
            .unwrap();
    }
    backend.commit(w).unwrap();

    let mut prepared = PreparedIndex {
        point_ids: (1..=n as u128).collect(),
        original_data: rows.iter().cloned().map(Some).collect(),
        encoded_data: vec![None; n],
        point_fields: vec![HashMap::new(); n],
        levels: levels.to_vec(),
        adjacency: adjacency.into_iter().map(Some).collect(),
        entry_ord,
        level_zero_preflushed: true,
        unreached: 0,
    };
    let mut w = backend.begin_write().unwrap();
    core.flush_prepared_index_chunk_be(&mut w, &mut prepared, 0, n)
        .unwrap();
    core.finalize_prepared_index_entry_be(&mut w, &prepared)
        .unwrap();
    backend.commit(w).unwrap();
    Written {
        core,
        _dir: dir,
        store,
        path,
    }
}

/// A segment whose graph the test builds point by point, on one thread and
/// at the levels it is given, with what every bulk builder runs: the SQ8
/// cosine of the merge build, `insert_point_ord` for each point and
/// `reach_every_point` at the end. It is written and searched like any
/// segment. Also returns each point's level-0 list (ordinals; ids are
/// `ordinal + 1`).
fn build_segment_at_levels(
    name: &str,
    hnsw: &HNSWConfig,
    rows: &[Vec<f32>],
    levels: &[usize],
) -> (Written, Vec<Vec<u32>>) {
    let (n, dim) = (rows.len(), rows[0].len());
    let codes = crate::helix_engine::vector_core::simd::SQ8Params::fit(rows)
        .quantize_bulk(rows)
        .0;
    let distance = |a: usize, b: usize| -> f32 {
        merge_build_distance(&DistanceMetric::Cosine, rows, &codes, dim, a, b)
    };
    let max_level = levels.iter().copied().max().unwrap();
    let entry_ord = levels.iter().position(|&level| level == max_level).unwrap();
    let adjacency: Vec<StdRwLock<Vec<Vec<u32>>>> = levels
        .iter()
        .map(|&level| StdRwLock::new(vec![Vec::new(); level + 1]))
        .collect();
    let ef_construction = (hnsw.ef_construct / 2).max(32);
    for ord in (0..n).filter(|&ord| ord != entry_ord) {
        insert_point_ord(
            ord,
            entry_ord,
            max_level,
            levels,
            &adjacency,
            ef_construction,
            hnsw.m,
            hnsw.m_max_0,
            Some(HNSW_KEEP_PRUNED_CONNECTIONS),
            &distance,
        );
    }
    reach_every_point(
        "merge",
        &adjacency,
        entry_ord,
        max_level,
        hnsw.m_max_0,
        &distance,
    );
    let adjacency: Vec<Vec<Vec<u32>>> = adjacency
        .into_iter()
        .map(|lists| lists.into_inner().unwrap())
        .collect();
    let level_zero: Vec<Vec<u32>> = adjacency.iter().map(|lists| lists[0].clone()).collect();
    let written = write_segment_with_lists(name, hnsw, rows, levels, adjacency, entry_ord);
    (written, level_zero)
}

/// Hits of a search with no filter by the two routes it can take (the
/// payload route and the id route), nearest first, each as its id and the
/// bits of its distance. `ef: None` is the segment's own.
fn unfiltered_hits(
    core: &VectorCore,
    query: &[f32],
    k: usize,
    ef: Option<usize>,
) -> [Vec<(u128, u32)>; 2] {
    unfiltered_hits_at(core, &core.backend.begin_read().unwrap(), query, k, ef)
}

/// `unfiltered_hits` through the read `ro`.
fn unfiltered_hits_at(
    core: &VectorCore,
    ro: &AnyRead<'_>,
    query: &[f32],
    k: usize,
    ef: Option<usize>,
) -> [Vec<(u128, u32)>; 2] {
    let hits = |hits: Vec<HVector>| -> Vec<(u128, u32)> {
        hits.iter()
            .map(|hit| (hit.get_id(), hit.get_distance().to_bits()))
            .collect()
    };
    [
        hits(
            core.search_with_selectivity_ef_observed::<fn(&HVector) -> bool>(
                ro, query, k, None, false, None, ef, None,
            )
            .unwrap(),
        ),
        hits(
            core.search_with_id_filter_ef_observed::<fn(u128) -> bool>(
                ro, query, k, None, false, None, ef, None,
            )
            .unwrap(),
        ),
    ]
}

/// Hits of a search by the id route under a filter applied in the walk, at
/// the segment's ef, as `unfiltered_hits` gives them.
fn id_filtered_hits(
    core: &VectorCore,
    query: &[f32],
    k: usize,
    keep: &dyn Fn(u128) -> bool,
) -> Vec<(u128, u32)> {
    let ro = core.backend.begin_read().unwrap();
    let keep: &[&dyn Fn(u128) -> bool] = &[keep];
    let hits = core
        .search_with_id_filter_ef_observed(&ro, query, k, Some(keep), true, None, None, None)
        .unwrap();
    hits.iter()
        .map(|hit| (hit.get_id(), hit.get_distance().to_bits()))
        .collect()
}

/// Runs `f` and returns, with what it returns, the level-0 walks that ran
/// out of candidates below ef and went on while it ran, by outcome:
/// [nothing_new, filled, short].
fn counting_restarts<T>(f: impl FnOnce() -> T) -> ([usize; 3], T) {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let out = metrics::with_local_recorder(&recorder, f);
    let rendered = handle.render();
    let count = |outcome: &str| -> usize {
        rendered
            .lines()
            .filter(|line| {
                line.starts_with("helix_vector_core_hnsw_walk_restarts_total{")
                    && line.contains(&format!("outcome=\"{outcome}\""))
            })
            .filter_map(|line| line.rsplit(' ').next()?.parse::<usize>().ok())
            .sum()
    };
    ([count("nothing_new"), count("filled"), count("short")], out)
}

/// One vector held by more points than a level-0 list holds (48 copies, 32
/// places), two of the copies on level 1, and 40 ordinary points close
/// around that vector, built with the linking pass on and off (off is the
/// kind of graph the stores hold today).
///
/// The copies fill each other's lists (a copy is the nearest there is and
/// passes the diversity test against every kept copy), so no list of the
/// group leads out of it. The descent of a query near the group ends on a
/// copy, and the level-0 walk from there has only copies to visit: it runs
/// out of candidates below ef. Ended there, as it was before the restart, a
/// search for the own vector of each of the 40 returns copies and not the
/// point. Gone on from its start's level-1 list, it returns the point
/// first: at the segment's ef and at one above the segment's size by both
/// routes, and under a filter that the 40 and the copies pass.
///
/// Below the size of the group (ef 32) the copies fill the results, the
/// walk is not below ef when it runs out, and the 40 stay lost: that case
/// is the build's to cure.
///
/// A walk that was not shut in is the walk it was. Ordinary points at the
/// segment's ef end by the stop rule and never get to the restart. Asked
/// for all their walk reaches, or under a filter five points pass, they run
/// out below ef, find their start's level-1 list visited, score nothing
/// more (`nothing_new`), and return the reply they return with the restart
/// off, to the bit.
#[test]
fn points_beside_a_group_of_equal_vectors_come_back_for_their_own_vector() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(53);
    let (dim, ordinary, beside, copies) = (32usize, 2300usize, 40usize, 48usize);
    let hnsw = HNSWConfig::new(Some(16), Some(256), Some(128));
    assert!(copies > hnsw.m_max_0);
    let centers: Vec<Vec<f32>> = (0..60)
        .map(|_| normalized((0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect()))
        .collect();
    let mut rows: Vec<Vec<f32>> = (0..ordinary)
        .map(|_| {
            let center = &centers[rng.random_range(0..centers.len())];
            normalized(
                center
                    .iter()
                    .map(|c| c + 0.05 * rng.random_range(-1.0f32..1.0))
                    .collect(),
            )
        })
        .collect();
    let shared = normalized((0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect());
    // Cosine about 0.97 to the shared vector.
    for _ in 0..beside {
        rows.push(normalized(
            shared
                .iter()
                .map(|c| c + 0.0766 * rng.random_range(-1.0f32..1.0))
                .collect(),
        ));
    }
    let first_copy = rows.len();
    rows.extend((0..copies).map(|_| shared.clone()));
    // The clustered points draw their levels as a build does; the points
    // beside the group stay on level 0 and two copies go up one level.
    let mut levels: Vec<usize> = (0..rows.len())
        .map(|ord| {
            if ord < ordinary {
                assign_hnsw_level(rng.random::<f64>(), hnsw.m_l)
            } else {
                0
            }
        })
        .collect();
    levels[first_copy] = 1;
    levels[first_copy + 1] = 1;

    // Ids are ordinal + 1.
    let own_first =
        |hits: &[(u128, u32)], ord: usize| hits.first().map(|hit| hit.0) == Some(ord as u128 + 1);
    let only_copies_back =
        |hits: &[(u128, u32)]| hits.iter().all(|hit| hit.0 as usize > first_copy);
    let group_and_beside = |id: u128| id as usize > ordinary;
    let five = |id: u128| (id as usize - 1) < ordinary && (id as usize - 1) % 460 == 0;
    let sample = || (0..ordinary).step_by(191);
    let wide = Some(rows.len() + 1);
    let searches = 2 * beside; // each of the 40 by both routes
    let open = 2 * sample().count();

    // Every case of both builds is judged before the test fails.
    let mut failures: Vec<String> = Vec::new();
    for pass in [true, false] {
        let (segment, level_zero) = with_link_unreached_for_test(pass, || {
            build_segment_at_levels(&format!("copies_{pass}"), &hnsw, &rows, &levels)
        });
        let core = &segment.core;
        let (reader, _reader_dir) = segment.reader(dim);
        let only_copies = (first_copy..rows.len())
            .filter(|&ord| level_zero[ord].iter().all(|&to| to as usize >= first_copy))
            .count();
        // The 40 points' own vectors: [searches that did not return the
        // point first, of those the replies that held only copies].
        let lost = |search: &dyn Fn(usize) -> Vec<Vec<(u128, u32)>>, restart: bool| {
            counting_restarts(|| {
                let mut count = [0usize; 2];
                for ord in ordinary..first_copy {
                    for hits in with_walk_restart_for_test(restart, || search(ord)) {
                        if !own_first(&hits, ord) {
                            count[0] += 1;
                            count[1] += usize::from(only_copies_back(&hits));
                        }
                    }
                }
                count
            })
        };
        // Ordinary points' own vectors with the restart on: [replies that
        // are not, to the bit, the reply with the restart off; 0].
        let changed = |search: &dyn Fn(usize) -> Vec<Vec<(u128, u32)>>| {
            let replies = |restart: bool| -> Vec<Vec<Vec<(u128, u32)>>> {
                with_walk_restart_for_test(restart, || sample().map(search).collect())
            };
            let off = replies(false);
            counting_restarts(|| {
                let on = replies(true);
                let pairs = on.iter().flatten().zip(off.iter().flatten());
                [pairs.filter(|(on, off)| on != off).count(), 0]
            })
        };
        let at_segment_ef = |ord: usize| unfiltered_hits(&core, &rows[ord], K, None).to_vec();
        let at_wide_ef = |ord: usize| unfiltered_hits(&core, &rows[ord], K, wide).to_vec();
        let at_ef_32 = |ord: usize| unfiltered_hits(&core, &rows[ord], K, Some(32)).to_vec();
        let all_reached =
            |ord: usize| unfiltered_hits(&core, &rows[ord], rows.len(), wide).to_vec();
        let group_passes =
            |ord: usize| vec![id_filtered_hits(&core, &rows[ord], K, &group_and_beside)];
        let five_pass = |ord: usize| vec![id_filtered_hits(&core, &rows[ord], rows.len(), &five)];
        // The same segment through a reader replica: the sidecar's scores,
        // the ordinal prefetch and the reader's caches.
        let on_reader = |ord: usize| unfiltered_hits(&reader, &rows[ord], K, None).to_vec();
        let all_reached_on_reader =
            |ord: usize| unfiltered_hits(&reader, &rows[ord], rows.len(), wide).to_vec();
        for (case, got, expected) in [
            (
                "the 40, restart off",
                lost(&at_segment_ef, false),
                ([0, 0, 0], [searches, searches]),
            ),
            (
                "the 40, restart on",
                lost(&at_segment_ef, true),
                ([0, searches, 0], [0, 0]),
            ),
            (
                "the 40, restart on, wide ef",
                lost(&at_wide_ef, true),
                ([0, 0, searches], [0, 0]),
            ),
            // A level-1 neighbour of the start is never nearer than the
            // copies, so the reply is this one with or without the "below
            // ef" rule; that no walk went on is what the count shows. The
            // drawn graph of the next test has the case where the rule
            // decides the reply.
            (
                "the 40, restart on, ef 32",
                lost(&at_ef_32, true),
                ([0, 0, 0], [searches, searches]),
            ),
            (
                "the 40 under a filter they and the copies pass, restart off",
                lost(&group_passes, false),
                ([0, 0, 0], [beside, beside]),
            ),
            (
                "the 40 under a filter they and the copies pass, restart on",
                lost(&group_passes, true),
                ([0, 0, beside], [0, 0]),
            ),
            (
                "ordinary points at the segment's ef",
                changed(&at_segment_ef),
                ([0, 0, 0], [0, 0]),
            ),
            (
                "ordinary points, all they reach",
                changed(&all_reached),
                ([open, 0, 0], [0, 0]),
            ),
            (
                "ordinary points under a filter five pass",
                changed(&five_pass),
                ([open / 2, 0, 0], [0, 0]),
            ),
            (
                "the 40 on a reader replica, restart off",
                lost(&on_reader, false),
                ([0, 0, 0], [searches, searches]),
            ),
            (
                "the 40 on a reader replica, restart on",
                lost(&on_reader, true),
                ([0, searches, 0], [0, 0]),
            ),
            (
                "ordinary points on a reader replica, all they reach",
                changed(&all_reached_on_reader),
                ([open, 0, 0], [0, 0]),
            ),
        ] {
            if got != expected {
                failures.push(format!(
                    "linking pass {pass}, {case}: {got:?}, expected {expected:?}; {only_copies} of \
                     the {copies} copies list only copies"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "expected (walks gone on as [nothing_new, filled, short], [replies lost or changed, of \
         the lost those of copies only]) as listed: {}",
        failures.join(" || ")
    );
}

/// Which list a walk goes on from, and when, on a graph drawn by hand:
/// points on a circle, named by their angle.
///
/// The query is the vector of the point at 0. Its descent ends on the point
/// at 10 (the start), which is shut in with the points at 40 and 50. The
/// start's level-1 list leads to the point at 30, whose level-0 list leads
/// to the point at 0. The entry's level-1 list leads only to the start and
/// to the point at 120, which leads back to the entry.
///
/// With room for one more result than the three shut in, the walk goes on
/// and returns the point at 0 first: it read the start's list, not the
/// entry's. With room for three it holds ef results when it runs out and
/// ends there, although a nearer point lies behind the start's list: a
/// closed set of ef points or more keeps the walk, copies or not.
#[test]
fn a_walk_out_of_candidates_goes_on_from_its_starts_level_one_list() {
    let at = |degrees: f32| vec![degrees.to_radians().cos(), degrees.to_radians().sin()];
    // Ids 1 to 7: the entry, the start, the two shut in with it, the point
    // its level-1 list leads to, the point asked for, the entry's other
    // neighbour.
    let rows = [90.0, 10.0, 40.0, 50.0, 30.0, 0.0, 120.0].map(at);
    let levels = [2, 1, 0, 0, 1, 0, 1];
    let adjacency = vec![
        vec![vec![6], vec![1, 6], vec![]],
        vec![vec![2, 3], vec![0, 4]],
        vec![vec![1, 3]],
        vec![vec![1, 2]],
        vec![vec![5, 1], vec![1]],
        vec![vec![4]],
        vec![vec![0], vec![0]],
    ];
    let hnsw = HNSWConfig::new(Some(2), Some(8), Some(4));
    let segment = write_segment_with_lists("drawn", &hnsw, &rows, &levels, adjacency, 0);
    let (reader, _reader_dir) = segment.reader(2);
    // On the writer's core and on a reader replica's, which scores from the
    // sidecar and prefetches ordinals: the same replies and the same counts.
    for (role, core) in [("writer", &segment.core), ("reader", &reader)] {
        let reply = |restart: bool, ef: usize| drawn_reply(core, &rows[5], restart, ef);
        let shut_in = [vec![2, 3, 4], vec![2, 3, 4]];
        assert_eq!(reply(false, 4), ([0, 0, 0], shut_in.clone()), "{role}");
        assert_eq!(
            reply(true, 4),
            ([0, 2, 0], [vec![6, 2, 5], vec![6, 2, 5]]),
            "{role}"
        );
        assert_eq!(reply(true, 3), ([0, 0, 0], shut_in), "{role}");
    }
}

/// The three nearest to `query` on a drawn graph by both routes, as ids, and
/// the walks that went on, as `counting_restarts` gives them. A segment this
/// small is scored exactly unless the test says otherwise.
fn drawn_reply(
    core: &VectorCore,
    query: &[f32],
    restart: bool,
    ef: usize,
) -> ([usize; 3], [Vec<u128>; 2]) {
    counting_restarts(|| {
        with_small_segment_exact_for_test(false, || {
            with_walk_restart_for_test(restart, || unfiltered_hits(core, query, 3, Some(ef)))
        })
        .map(|hits| hits.iter().map(|hit| hit.0).collect::<Vec<u128>>())
    })
}

/// A hole left by a delete that took the row's own lists with it, as every
/// delete did before the lists were kept and as the segments stored before
/// then have it, on a graph drawn by hand: points on a circle, named by
/// their angle. The lists that name the row stay. (Where the row's own
/// lists stay too the walk gets through them:
/// `a_walk_gets_through_the_lists_a_hard_delete_left`.)
///
/// The query is the vector of the point at 0. Its descent ends on the point
/// at 10 (the start), whose level-0 list leads to the point at 0 through
/// the point at 20, and whose level-1 list names the point at 30, which
/// also links to the point at 0.
///
/// Whole, the walk finds the point at 0 and ends by its stop rule, the
/// restart on or off. With the point at 20 deleted the start's list names a
/// row that is gone: the walk runs out of candidates with two results, and
/// gone on from the start's level-1 list it returns the point at 0 first.
/// With the point at 30 deleted as well, the level-1 list names only a row
/// that is gone: it is skipped without an error, nothing is scored, and the
/// reply is the one without the restart.
#[test]
fn a_walk_cut_off_by_a_deleted_row_goes_on_from_its_starts_level_one_list() {
    let (rows, segment) = holes("holes");
    let reply = |restart: bool| drawn_reply(&segment.core, &rows[5], restart, 4);
    let both = |ids: &[u128]| [ids.to_vec(), ids.to_vec()];
    let delete = |ids: &[u128]| with_deleted_lists_for_test(false, 0, || segment.hard_delete(ids));

    assert_eq!(reply(false), ([0, 0, 0], both(&[6, 2, 4])));
    assert_eq!(reply(true), ([0, 0, 0], both(&[6, 2, 4])));

    delete(&[4]);
    assert_eq!(reply(false), ([0, 0, 0], both(&[2, 3])));
    assert_eq!(reply(true), ([0, 2, 0], both(&[6, 2, 5])));

    delete(&[5]);
    assert_eq!(reply(false), ([0, 0, 0], both(&[2, 3])));
    assert_eq!(reply(true), ([2, 0, 0], both(&[2, 3])));
}

/// The drawn graph of the two tests on holes, written as a segment, and its
/// points' vectors.
fn holes(name: &str) -> ([Vec<f32>; 7], Written) {
    let at = |degrees: f32| vec![degrees.to_radians().cos(), degrees.to_radians().sin()];
    // Ids 1 to 7: the entry, the start, a point only the start links to,
    // the point on the way, the point the start's level-1 list names, the
    // point asked for, the entry's other neighbour.
    let rows = [90.0, 10.0, 40.0, 20.0, 30.0, 0.0, 120.0].map(at);
    let levels = [2, 1, 0, 0, 1, 0, 1];
    let adjacency = vec![
        vec![vec![6], vec![1, 6], vec![]],
        vec![vec![2, 3], vec![4]],
        vec![vec![1]],
        vec![vec![5, 1]],
        vec![vec![5, 1], vec![1]],
        vec![vec![3, 4]],
        vec![vec![0], vec![0]],
    ];
    let hnsw = HNSWConfig::new(Some(2), Some(8), Some(4));
    let segment = write_segment_with_lists(name, &hnsw, &rows, &levels, adjacency, 0);
    (rows, segment)
}

/// The walk that goes on from its start's level-1 list, where that list
/// names a deleted point whose lists were kept: the graph of
/// `a_walk_cut_off_by_a_deleted_row_goes_on_from_its_starts_level_one_list`,
/// with the point at 20 deleted as the stores have it (its lists taken) and
/// the point at 30 deleted with its lists kept.
///
/// The level-0 walk from the start runs out with two results: the point at
/// 20 has no row, and for a walk that gets through one such id, no list to
/// read. Gone on from the start's level-1 list it meets the point at 30,
/// which has no row either. A walk that gets through none ends there with
/// nothing new. One that gets through one reads its level-0 list and
/// returns the point at 0 first, with three results of the four asked for.
#[test]
fn a_walk_gone_on_from_its_starts_level_one_list_gets_through_a_kept_list() {
    let (rows, segment) = holes("kept_holes");
    with_deleted_lists_for_test(false, 0, || segment.hard_delete(&[4]));
    with_deleted_lists_for_test(true, 0, || segment.hard_delete(&[5]));
    let reply = |restart: bool, crossed: usize| {
        with_deleted_lists_for_test(true, crossed, || {
            drawn_reply(&segment.core, &rows[5], restart, 4)
        })
    };
    let both = |ids: &[u128]| [ids.to_vec(), ids.to_vec()];

    assert_eq!(reply(true, 0), ([2, 0, 0], both(&[2, 3])));
    assert_eq!(reply(false, 1), ([0, 0, 0], both(&[2, 3])));
    assert_eq!(reply(true, 1), ([0, 0, 2], both(&[6, 2, 3])));
}

/// The rows no stored link leads to, read from a graph as it is stored
/// (`VectorCore::stored_reach`), on a graph drawn by hand: ids 1 to 8.
///
/// The entry (1) leads to 2 on level 1, and on level 0 to 2 and 3. 2 leads
/// on to 4, and 4 to 5. No list names 6. 7 and 8 name each other and no
/// other list names them. So 6, 7 and 8 are the rows no walk arrives at.
///
/// With 2 and 6 replaced by another segment's copies, six rows are served:
/// 2 is scored by a walk all the same and still leads on to 4, and 6 is
/// counted apart from 7 and 8.
///
/// A hard delete of 4 takes its row and leaves its lists. 5 was reached
/// through 4 alone: it joins them for a walk that ends at an id with no
/// row, and does not for one that gets through 4's list. With 2 deleted as
/// well there are two such lists in a row between the entry and 5, and it
/// takes a walk that gets through two. The count says so for 0, 1, 2 and 4
/// lists whatever the walk here does.
///
/// A segment this small is scored row by row unless the test says
/// otherwise, and then its lists are not read.
#[test]
fn stored_reach_lists_the_unreached_rows_of_a_stored_graph() {
    let at = |degrees: f32| vec![degrees.to_radians().cos(), degrees.to_radians().sin()];
    let rows = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0].map(at);
    let levels = [1, 1, 0, 0, 0, 0, 0, 0];
    let adjacency = vec![
        vec![vec![1, 2], vec![1]],
        vec![vec![0, 3], vec![0]],
        vec![vec![0]],
        vec![vec![1, 4]],
        vec![vec![3]],
        vec![vec![0]],
        vec![vec![7]],
        vec![vec![6]],
    ];
    let hnsw = HNSWConfig::new(Some(2), Some(8), Some(4));
    let segment = write_segment_with_lists("reach", &hnsw, &rows, &levels, adjacency, 0);
    let core = &segment.core;
    // What is read when a search walks the graph (`Some`: by a walk that
    // gets through that many rowless lists in a row) or scores the rows.
    let reach = |walk: Option<usize>, replaced: &[u128]| {
        let ro = core.backend.begin_read().unwrap();
        with_deleted_lists_for_test(true, walk.unwrap_or(0), || {
            with_small_segment_exact_for_test(walk.is_none(), || {
                core.stored_reach(&ro, &|id: u128| -> Result<bool, VectorError> {
                    Ok(replaced.contains(&id))
                })
                .unwrap()
            })
        })
    };
    // The ids, how many are served, and where the graph is walked: the
    // served rows no link leads to, the count of the others, and the served
    // rows unreached through 0, 1, 2 and 4 rowless lists.
    type Unreached = (Vec<u128>, usize, [usize; 4]);
    let seen = |ids: &[u128], served: usize, unreached: Option<Unreached>| StoredReach {
        ids: ids.to_vec(),
        served,
        walked: unreached.is_some(),
        walked_filtered: unreached.is_some(),
        unreached_unserved: unreached.as_ref().map(|(_, unserved, _)| *unserved),
        unreached_crossing: unreached
            .as_ref()
            .map(|(_, _, rows)| LISTS_CROSSED_COUNTED.into_iter().zip(*rows).collect()),
        unreached: unreached.map(|(ids, _, _)| ids),
        // An insert writes a row with its ordinal row and a hard delete
        // takes both, so neither is ever without the other here.
        ordinals_without_row: 0,
        rows_without_ordinal: 0,
    };
    let all: Vec<u128> = (1..=8).collect();
    assert_eq!(reach(None, &[]), seen(&all, 8, None));
    assert_eq!(
        reach(Some(2), &[]),
        seen(&all, 8, Some((vec![6, 7, 8], 0, [3; 4])))
    );
    assert_eq!(reach(None, &[2, 6]), seen(&all, 6, None));
    assert_eq!(
        reach(Some(2), &[2, 6]),
        seen(&all, 6, Some((vec![7, 8], 1, [2; 4])))
    );

    let delete = |ids: &[u128]| with_deleted_lists_for_test(true, 0, || segment.hard_delete(ids));
    delete(&[4]);
    let left = [1, 2, 3, 5, 6, 7, 8];
    let by_lists = [4, 3, 3, 3];
    assert_eq!(
        reach(Some(0), &[]),
        seen(&left, 7, Some((vec![5, 6, 7, 8], 0, by_lists)))
    );
    assert_eq!(
        reach(Some(1), &[]),
        seen(&left, 7, Some((vec![6, 7, 8], 0, by_lists)))
    );

    delete(&[2]);
    let left = [1, 3, 5, 6, 7, 8];
    let by_lists = [4, 4, 3, 3];
    assert_eq!(
        reach(Some(1), &[]),
        seen(&left, 6, Some((vec![5, 6, 7, 8], 0, by_lists)))
    );
    assert_eq!(
        reach(Some(2), &[]),
        seen(&left, 6, Some((vec![6, 7, 8], 0, by_lists)))
    );
}

/// The reach read holds a segment's ordinal rows against its level-0 rows
/// and counts each kind left without the other. No insert or delete of the
/// engine leaves one in a space without a quantized store, so the two are
/// made here by writes to one namespace alone: 3 loses its ordinal row, and
/// 99, which has no row, gets one.
#[test]
fn stored_reach_counts_ordinal_rows_and_rows_without_their_pair() {
    let rows = [0.0f32, 1.0, 2.0, 3.0].map(|x| vec![x, 1.0]);
    let hnsw = HNSWConfig::new(Some(2), Some(8), Some(4));
    let lists = vec![vec![vec![1]], vec![vec![0]], vec![vec![1]], vec![vec![2]]];
    let segment = write_segment_with_lists("pairs", &hnsw, &rows, &[0; 4], lists, 0);
    let core = &segment.core;
    // (rows, ordinal rows without a row, rows without an ordinal row)
    let counts = || {
        let ro = core.backend.begin_read().unwrap();
        let reach = core
            .stored_reach(&ro, &|_: u128| -> Result<bool, VectorError> { Ok(false) })
            .unwrap();
        (
            reach.ids.len(),
            reach.ordinals_without_row,
            reach.rows_without_ordinal,
        )
    };
    assert_eq!(counts(), (4, 0, 0), "as the engine writes a segment");

    let ordinals = core.seg_ns(SegmentDb::Ordinals);
    let mut w = core.backend.begin_write().unwrap();
    core.backend
        .delete(&mut w, ordinals, &3u128.to_be_bytes())
        .unwrap();
    core.backend.commit(w).unwrap();
    assert_eq!(counts(), (4, 0, 1), "3 has a row and no ordinal row");

    let mut w = core.backend.begin_write().unwrap();
    core.backend
        .put(&mut w, ordinals, &99u128.to_be_bytes(), &7u64.to_le_bytes())
        .unwrap();
    core.backend.commit(w).unwrap();
    assert_eq!(counts(), (4, 1, 1), "99 has an ordinal row and no row");

    segment.hard_delete(&[1]);
    assert_eq!(counts(), (3, 1, 1), "a hard delete takes both rows of 1");
}

/// A walk gets through the lists a hard delete leaves. Ids 1 to 5 in a line:
/// each point's list names the one before it and the one after, the entry
/// is 1 and the query is 5's own vector, so the walk passes every point
/// between.
///
/// With 4 deleted and its lists kept, 5 comes back first for a walk that
/// gets through one rowless list, and not at all for one that gets through
/// none. That one is the walk of an image from before the lists were kept:
/// it returns what it returns from a segment whose delete took the lists
/// too. With 3 deleted as well, two rowless lists in a row lie between 2
/// and 5: one is not enough, two are, and a walk that may read only one
/// list ends after 3's. Where the lists went with the rows a walk that would
/// read them finds none, and returns what the other does.
///
/// The same holds for a search under a filter, by either route (the one
/// that hands the filter the row and the one that hands it the id), and on
/// a reader replica through a read pinned to its snapshot, whether it was
/// opened after the delete or before it.
#[test]
fn a_walk_gets_through_the_lists_a_hard_delete_left() {
    let at = |degrees: f32| vec![degrees.to_radians().cos(), degrees.to_radians().sin()];
    let rows = [80.0, 60.0, 40.0, 20.0, 0.0].map(at);
    let levels = [1, 0, 0, 0, 0];
    let line = || {
        vec![
            vec![vec![1], vec![]],
            vec![vec![0, 2]],
            vec![vec![1, 3]],
            vec![vec![2, 4]],
            vec![vec![3]],
        ]
    };
    let hnsw = HNSWConfig::new(Some(2), Some(8), Some(4));
    // The line with `gone` deleted, their lists kept or taken with them.
    let segment = |name: &str, keep: bool, gone: &[u128]| {
        let written = write_segment_with_lists(name, &hnsw, &rows, &levels, line(), 0);
        with_deleted_lists_for_test(keep, 0, || written.hard_delete(gone));
        written
    };
    // For 5's vector, by a walk that reads as many lists as it comes to.
    let found =
        |core: &VectorCore, crossed: usize| ids_through_lists(core, &rows[4], crossed, usize::MAX);
    let both = |ids: &[u128]| [ids.to_vec(), ids.to_vec()];

    let kept = segment("kept", true, &[4]);
    let taken = segment("taken", false, &[4]);
    assert_eq!(found(&kept.core, 1), both(&[5, 3, 2, 1]));
    assert_eq!(found(&kept.core, 0), both(&[3, 2, 1]));
    assert_eq!(found(&taken.core, 0), found(&kept.core, 0));
    assert_eq!(found(&taken.core, 2), found(&taken.core, 0));

    let chain = segment("chain", true, &[3, 4]);
    assert_eq!(found(&chain.core, 1), both(&[2, 1]));
    assert_eq!(found(&chain.core, 2), both(&[5, 2, 1]));
    let reading = |lists: usize| ids_through_lists(&chain.core, &rows[4], 2, lists);
    assert_eq!(reading(1), both(&[2, 1]));
    assert_eq!(reading(2), both(&[5, 2, 1]));

    // Under a filter that 2 does not pass, by the payload route and by the
    // id route: 2 is still the way to 3, and 4's list the way to 5.
    let filtered = |crossed: usize| {
        let keep = |id: u128| id != 2;
        let keep_row = |row: &HVector| keep(row.get_id());
        let ro = kept.core.backend.begin_read().unwrap();
        let ids = |hits: Vec<HVector>| hits.iter().map(HVector::get_id).collect::<Vec<u128>>();
        with_deleted_lists_for_test(true, crossed, || {
            with_small_segment_exact_for_test(false, || {
                with_walk_restart_for_test(false, || {
                    let by_row = kept.core.search_with_selectivity_ef_observed(
                        &ro,
                        &rows[4],
                        5,
                        Some(&[keep_row]),
                        true,
                        None,
                        Some(8),
                        None,
                    );
                    let by_id = kept.core.search_with_id_filter_ef_observed(
                        &ro,
                        &rows[4],
                        5,
                        Some(&[keep]),
                        true,
                        None,
                        Some(8),
                        None,
                    );
                    [ids(by_row.unwrap()), ids(by_id.unwrap())]
                })
            })
        })
    };
    assert_eq!(filtered(1), both(&[5, 3, 1]));
    assert_eq!(filtered(0), both(&[3, 1]));

    // On a reader replica, through a read pinned to the snapshot it has
    // now, as production's reads are.
    let pinned = |reader: &VectorCore, crossed: usize| {
        let AnyBackend::LsmReader(lsm_reader) = &*reader.backend else {
            unreachable!()
        };
        let read = AnyRead::LsmReader(Some(lsm_reader.begin_snapshot().unwrap()));
        with_deleted_lists_for_test(true, crossed, || {
            with_small_segment_exact_for_test(false, || {
                with_walk_restart_for_test(false, || {
                    unfiltered_hits_at(reader, &read, &rows[4], 5, Some(8))
                })
            })
        })
        .map(|hits| hits.iter().map(|hit| hit.0).collect::<Vec<u128>>())
    };
    // One opened after the delete finds no row for 4 and reads the same
    // list.
    let (reader, _reader_dir) = kept.reader(2);
    assert_eq!(pinned(&reader, 1), both(&[5, 3, 2, 1]));
    assert_eq!(pinned(&reader, 0), both(&[3, 2, 1]));

    // One that was open before the delete: it has walked the whole line, so
    // it holds every list and a vector file with 4 in it. Once its view has
    // moved past the delete, 4 has no row at its snapshot, and the list it
    // holds for 4 is the one the delete left.
    let whole = write_segment_with_lists("open_before", &hnsw, &rows, &levels, line(), 0);
    let (open, _open_dir) = whole.reader(2);
    assert_eq!(pinned(&open, 1), both(&[5, 4, 3, 2, 1]));
    with_deleted_lists_for_test(true, 0, || whole.hard_delete(&[4]));
    open.backend
        .refresh_lsm_reader_with_poll_interval(std::time::Duration::from_millis(100))
        .unwrap();
    assert_eq!(pinned(&open, 1), both(&[5, 3, 2, 1]));
    assert_eq!(pinned(&open, 0), both(&[3, 2, 1]));
}

/// The ids both routes return for `query`, nearest first, by a walk that
/// gets through `crossed` rowless lists in a row and reads `budget` of them.
fn ids_through_lists(
    core: &VectorCore,
    query: &[f32],
    crossed: usize,
    budget: usize,
) -> [Vec<u128>; 2] {
    with_deleted_lists_budget_for_test(true, crossed, budget, || {
        with_small_segment_exact_for_test(false, || {
            with_walk_restart_for_test(false, || unfiltered_hits(core, query, 5, Some(8)))
        })
    })
    .map(|hits| hits.iter().map(|hit| hit.0).collect::<Vec<u128>>())
}

/// The walk and the count agree on what the lists a hard delete left lead
/// to, on a graph drawn by hand. Ids 1 to 7: A (the entry), B, three deleted
/// points X1, X2 and X3, and T and U. The query is T's vector, A is nearer
/// to it than B.
///
/// On level 0, A names X1 and B, X1 names X2, X2 names X3, B names X3 too,
/// and X3 names T. So T lies behind one deleted point by way of B and behind
/// three in a row by way of X1. A walk takes X1 and X2 before B, since they
/// stand at A's distance. One that gets through two in a row stops at X3
/// when X2 names it, and takes it up when B names it after: the shortest
/// way to an id decides, as it does in the count.
///
/// U is named by X1's level-1 list alone. No walk reads a deleted point's
/// list above level 0, so U is reached by none and the count says so for
/// any number of lists.
#[test]
fn a_walk_and_the_count_take_the_shortest_way_through_deleted_lists() {
    let at = |degrees: f32| vec![degrees.to_radians().cos(), degrees.to_radians().sin()];
    let rows = [20.0, 40.0, 60.0, 70.0, 80.0, 0.0, 90.0].map(at);
    let levels = [1, 0, 1, 0, 0, 0, 1];
    let adjacency = vec![
        vec![vec![2, 1], vec![2]],
        vec![vec![4]],
        vec![vec![3], vec![6]],
        vec![vec![4]],
        vec![vec![5]],
        vec![vec![4]],
        vec![vec![], vec![]],
    ];
    let hnsw = HNSWConfig::new(Some(2), Some(8), Some(4));
    let segment = write_segment_with_lists("shortest", &hnsw, &rows, &levels, adjacency, 0);
    with_deleted_lists_for_test(true, 0, || segment.hard_delete(&[3, 4, 5]));
    let core = &segment.core;

    let found = |crossed: usize| ids_through_lists(core, &rows[5], crossed, usize::MAX);
    let both = |ids: &[u128]| [ids.to_vec(), ids.to_vec()];
    assert_eq!(found(0), both(&[1, 2]));
    assert_eq!(found(1), both(&[6, 1, 2]));
    assert_eq!(found(2), both(&[6, 1, 2]));

    // The served rows the count calls unreached by a walk that gets through
    // `crossed` lists in a row, and how many for 0, 1, 2 and 4.
    let counted = |crossed: usize| {
        let ro = core.backend.begin_read().unwrap();
        let reach = with_deleted_lists_for_test(true, crossed, || {
            with_small_segment_exact_for_test(false, || {
                core.stored_reach(&ro, &|_: u128| -> Result<bool, VectorError> { Ok(false) })
                    .unwrap()
            })
        });
        (reach.unreached.unwrap(), reach.unreached_crossing.unwrap())
    };
    let by_lists: Vec<(usize, usize)> = vec![(0, 2), (1, 1), (2, 1), (4, 1)];
    assert_eq!(counted(0), (vec![6, 7], by_lists.clone()));
    assert_eq!(counted(1), (vec![7], by_lists.clone()));
    assert_eq!(counted(2), (vec![7], by_lists));
}

/// The link pass over a graph as it is stored, on the drawn graph of
/// `stored_reach_lists_the_unreached_rows_of_a_stored_graph`: no list names
/// 6, and 7 and 8 name only each other.
///
/// A walk for 6's vector, and one for 7's, arrives at 5 first of the rows
/// with room, so in one plan both go at the end of 5's list. 8 gets no
/// link: 7 leads to it.
///
/// Planned and written a link a call, the first call links 6 and does not
/// mark the segment, and the second links 7 from 6, which a walk now
/// arrives at. After it no row is unreached, the segment carries the pass's
/// mark, and 6 comes back for its own vector.
#[test]
fn the_link_pass_links_the_unreached_rows_of_a_stored_graph() {
    let at = |degrees: f32| vec![degrees.to_radians().cos(), degrees.to_radians().sin()];
    let rows = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0].map(at);
    let levels = [1, 1, 0, 0, 0, 0, 0, 0];
    let adjacency = vec![
        vec![vec![1, 2], vec![1]],
        vec![vec![0, 3], vec![0]],
        vec![vec![0]],
        vec![vec![1, 4]],
        vec![vec![3]],
        vec![vec![0]],
        vec![vec![7]],
        vec![vec![6]],
    ];
    let hnsw = HNSWConfig::new(Some(2), Some(8), Some(4));
    let segment = write_segment_with_lists("linked", &hnsw, &rows, &levels, adjacency, 0);
    let core = &segment.core;
    // The served rows no link leads to, and whether the pass has been here.
    let state = || {
        let ro = core.backend.begin_read().unwrap();
        let reach = with_small_segment_exact_for_test(false, || {
            core.stored_reach(&ro, &|_: u128| -> Result<bool, VectorError> { Ok(false) })
                .unwrap()
        });
        (reach.unreached, core.link_pass_done(&ro).unwrap())
    };
    assert_eq!(state(), (Some(vec![6, 7, 8]), false));

    // What a call that may look at `max_rows` rows plans for `unreached`,
    // from the row the pass has come to: the links, how many rows found no
    // room, and the row it stopped at where rows are left.
    let plan = |unreached: &[u128], max_rows: usize| {
        let ro = core.backend.begin_read().unwrap();
        with_small_segment_exact_for_test(false, || {
            let after = core.link_pass_after(&ro).unwrap();
            core.plan_stored_links(&ro, unreached, after, max_rows)
                .unwrap()
        })
    };
    // That plan written as the writer writes it; the rows whose list changed.
    let write = |(links, _, upto): (Vec<(u128, u128)>, usize, Option<u128>)| {
        let mut w = core.backend.begin_write().unwrap();
        let changed = core
            .apply_stored_links_be(&mut w, &links, upto, began_at(core))
            .unwrap();
        core.backend.commit(w).unwrap();
        core.forget_neighbor_lists(&changed);
        changed
    };
    assert_eq!(
        plan(&[6, 7, 8], usize::MAX),
        (vec![(5, 6), (5, 7)], 0, None)
    );

    // A row a call: the first links 6, stops there and leaves the segment
    // unmarked. The second goes on after 6, finds 6 the nearest row with
    // room for 7, and comes to the end.
    let first = plan(&[6, 7, 8], 1);
    assert_eq!(first, (vec![(5, 6)], 0, Some(6)));
    assert_eq!(write(first), [5]);
    assert_eq!(state(), (Some(vec![7, 8]), false));
    let second = plan(&[7, 8], 1);
    assert_eq!(second, (vec![(6, 7)], 0, None));
    assert_eq!(write(second), [6]);
    assert_eq!(state(), (Some(vec![]), true));

    let found = with_small_segment_exact_for_test(false, || {
        with_walk_restart_for_test(false, || unfiltered_hits(core, &rows[5], 1, Some(8)))
    });
    assert_eq!(found.map(|hits| hits[0].0), [6, 6]);
}

/// `hosts` rows that each name all the others (ids 1 to `hosts`, ten degrees
/// apart, the first of them the entry and alone on level 1), and after them
/// `more` rows that name the first and that no list names.
fn each_names_the_others(name: &str, hnsw: &HNSWConfig, hosts: u32, more: u32) -> Written {
    hosts_and_rows_beside(name, hnsw, hosts, more, &[0])
}

/// As `each_names_the_others`, the `more` rows each naming the rows at the
/// ordinals `named`.
fn hosts_and_rows_beside(
    name: &str,
    hnsw: &HNSWConfig,
    hosts: u32,
    more: u32,
    named: &[u32],
) -> Written {
    let at = |degrees: f32| vec![degrees.to_radians().cos(), degrees.to_radians().sin()];
    let rows: Vec<Vec<f32>> = (0..hosts + more).map(|ord| at(ord as f32 * 10.0)).collect();
    let mut levels = vec![0; rows.len()];
    levels[0] = 1;
    let adjacency = (0..hosts + more)
        .map(|ord| {
            let level_zero: Vec<u32> = if ord < hosts {
                (0..hosts).filter(|&other| other != ord).collect()
            } else {
                named.to_vec()
            };
            if ord == 0 {
                vec![level_zero, Vec::new()]
            } else {
                vec![level_zero]
            }
        })
        .collect();
    write_segment_with_lists(name, hnsw, &rows, &levels, adjacency, 0)
}

/// The level-0 list of each of `ids`, as the store has it.
fn level_zero_lists(
    core: &VectorCore,
    ids: impl IntoIterator<Item = u128>,
) -> HashMap<u128, Vec<u128>> {
    let ro = core.backend.begin_read().unwrap();
    ids.into_iter()
        .map(|id| {
            let list = core.get_neighbor_block_uncached(&ro, id, 0).unwrap();
            (id, list.ids)
        })
        .collect()
}

/// The count of deleted rows a call of the pass over this segment writes
/// with its mark or its place, as the plan reads it.
fn began_at(core: &VectorCore) -> u64 {
    let ro = core.backend.begin_read().unwrap();
    core.link_pass_counts(&ro).unwrap().0
}

/// The served rows of a segment no stored link leads to, its graph walked.
fn stored_unreached(core: &VectorCore) -> Vec<u128> {
    let ro = core.backend.begin_read().unwrap();
    with_small_segment_exact_for_test(false, || {
        core.stored_reach(&ro, &|_: u128| -> Result<bool, VectorError> { Ok(false) })
            .unwrap()
    })
    .unreached
    .unwrap()
}

/// The link pass over one segment as the writer runs it, `max_rows`
/// unreached rows a call: the unreached rows are read and their links
/// planned, the links are written with the mark and committed, and the
/// changed lists are dropped from the cache, until a call comes to the end.
/// Returns the links it wrote and how many rows found no room.
fn run_link_pass(core: &VectorCore, max_rows: usize) -> (Vec<(u128, u128)>, usize) {
    let mut linked = Vec::new();
    let mut without_room = 0;
    loop {
        let unreached = stored_unreached(core);
        let ro = core.backend.begin_read().unwrap();
        let (links, no_room, upto) = with_small_segment_exact_for_test(false, || {
            let after = core.link_pass_after(&ro).unwrap();
            core.plan_stored_links(&ro, &unreached, after, max_rows)
                .unwrap()
        });
        drop(ro);
        let mut w = core.backend.begin_write().unwrap();
        let changed = core
            .apply_stored_links_be(&mut w, &links, upto, began_at(core))
            .unwrap();
        core.backend.commit(w).unwrap();
        core.forget_neighbor_lists(&changed);
        assert_eq!(changed.len(), links.len(), "a planned link was not written");
        linked.extend(links);
        without_room += no_room;
        if upto.is_none() {
            return (linked, without_room);
        }
    }
}

/// What a call of the link pass that may look at `max_rows` rows plans for
/// `unreached`, from the place the pass has come to.
fn plan_link_pass(
    core: &VectorCore,
    unreached: &[u128],
    max_rows: usize,
) -> (Vec<(u128, u128)>, usize, Option<u128>) {
    let ro = core.backend.begin_read().unwrap();
    with_small_segment_exact_for_test(false, || {
        let after = core.link_pass_after(&ro).unwrap();
        core.plan_stored_links(&ro, unreached, after, max_rows)
            .unwrap()
    })
}

/// A row gets no link when no row has room, near it or below: it is
/// counted, and looked at once. A list has room up to a quarter past what a
/// build fills it to (`link_pass_list_bound`), here five ids where a build
/// writes four.
///
/// Ids 1 to 8: each of 1 to 6 names the five others, a list at its bound,
/// and no list names 7 or 8. A walk for either's vector returns those six
/// and none has room, and the way down from the nearest goes from one of
/// the six to another until it gives up. One plan comes to the end with two
/// rows left without a link. A row a call, the first looks at 7 and stops
/// there; the second goes on after 7, looks at 8 alone and comes to the end.
#[test]
fn the_link_pass_counts_a_row_no_row_near_or_below_has_room_for() {
    let hnsw = HNSWConfig::new(Some(2), Some(8), Some(4));
    assert_eq!((hnsw.m_max_0, link_pass_list_bound(hnsw.m_max_0)), (4, 5));
    let at_bound = each_names_the_others("at_bound", &hnsw, 6, 2);
    let core = &at_bound.core;
    assert_eq!(plan_link_pass(core, &[7, 8], usize::MAX), (vec![], 2, None));
    let first = plan_link_pass(core, &[7, 8], 1);
    assert_eq!(first, (vec![], 1, Some(7)));
    let mut w = core.backend.begin_write().unwrap();
    core.apply_stored_links_be(&mut w, &first.0, first.2, began_at(core))
        .unwrap();
    core.backend.commit(w).unwrap();
    assert_eq!(plan_link_pass(core, &[7, 8], 1), (vec![], 1, None));
}

/// Where every row near a row is at its bound, its link comes from a row
/// the pass has linked below the nearest of them, and the same call can
/// have linked that row. Ids 1 to 12: each of 1 to 5 names the four others
/// and each of 6 to 12 names four rows, as a build leaves them, and no list
/// names 6 to 12.
///
/// One plan gives each of the five lists one link, the nearest first, which
/// brings it to its bound. The five are all that a walk for 11 or for 12
/// returns, and 5 is the nearest to both. Below 5 is 6, which has room for
/// one: 11 goes at the end of 6's list, and 12, a row further down, at the
/// end of 11's. The one call leaves no row.
///
/// Where 6 to 12 each name five rows, a list at its bound, 6 has no room
/// and below 6 is 5 again. The way down gives up after
/// `LINK_PASS_STEPS_DOWN` rows, and 11 and 12 are counted and left.
#[test]
fn a_row_whose_near_rows_are_at_their_bound_is_linked_from_a_row_below() {
    let hnsw = HNSWConfig::new(Some(2), Some(8), Some(4));
    assert_eq!((hnsw.m_max_0, link_pass_list_bound(hnsw.m_max_0)), (4, 5));
    let from_the_five = [(5, 6), (4, 7), (3, 8), (2, 9), (1, 10)];
    let cases: [(&str, &[u32], &[(u128, u128)], Vec<u128>); 2] = [
        ("room_below", &[0, 1, 2, 3], &[(6, 11), (11, 12)], vec![]),
        ("none_below", &[0, 1, 2, 3, 4], &[], vec![11, 12]),
    ];
    for (name, named, from_below, left) in cases {
        let segment = hosts_and_rows_beside(name, &hnsw, 5, 7, named);
        let core = &segment.core;
        let unreached: Vec<u128> = (6..=12).collect();
        assert_eq!(stored_unreached(core), unreached, "{name}");
        let (links, without_room, upto) = plan_link_pass(core, &unreached, usize::MAX);
        assert_eq!(
            (links.as_slice(), without_room, upto),
            (
                &[&from_the_five[..], from_below].concat()[..],
                left.len(),
                None
            ),
            "{name}"
        );
        let mut w = core.backend.begin_write().unwrap();
        let changed = core
            .apply_stored_links_be(&mut w, &links, upto, began_at(core))
            .unwrap();
        core.backend.commit(w).unwrap();
        core.forget_neighbor_lists(&changed);
        assert_eq!(changed.len(), links.len(), "{name}");
        let ro = core.backend.begin_read().unwrap();
        assert_eq!(
            (stored_unreached(core), core.link_pass_done(&ro).unwrap()),
            (left, true),
            "{name}"
        );
    }
}

/// A row a delete has taken is no row to link from, and where the rows below
/// a row are all gone the way down goes on below the next nearest. The five
/// and the seven of the test above, the seven with room for one. A first
/// call, a row a call, links 6 from 5, and a delete takes 6: 5's list goes
/// on naming it. The next call links 7 to 10 from the other four. For 11 and
/// 12 the five are then at their bound and below 5, the nearest, is 6 alone,
/// which is gone, and a link from it would lead to them from no row. Below
/// 4, the next nearest, is 7: 11 goes at the end of 7's list and 12 at the
/// end of 11's, and no row is left.
#[test]
fn a_row_a_delete_has_taken_is_not_linked_from() {
    let hnsw = HNSWConfig::new(Some(2), Some(8), Some(4));
    let segment = hosts_and_rows_beside("gone_below", &hnsw, 5, 7, &[0, 1, 2, 3]);
    let core = &segment.core;
    // A call as the writer runs it: what it planned, written.
    let call = |max_rows: usize| {
        let plan = plan_link_pass(core, &stored_unreached(core), max_rows);
        let mut w = core.backend.begin_write().unwrap();
        let changed = core
            .apply_stored_links_be(&mut w, &plan.0, plan.2, began_at(core))
            .unwrap();
        core.backend.commit(w).unwrap();
        core.forget_neighbor_lists(&changed);
        plan
    };
    assert_eq!(call(1), (vec![(5, 6)], 0, Some(6)));
    segment.hard_delete(&[6]);
    assert_eq!(level_zero_lists(core, [5])[&5], [1, 2, 3, 4, 6]);
    assert_eq!(stored_unreached(core), (7..=12).collect::<Vec<u128>>());
    assert_eq!(
        call(usize::MAX),
        (
            vec![(4, 7), (3, 8), (2, 9), (1, 10), (7, 11), (11, 12)],
            0,
            None
        )
    );
    assert_eq!(stored_unreached(core), Vec::<u128>::new());
}

/// Eight rows ten degrees apart, as
/// `the_link_pass_links_the_unreached_rows_of_a_stored_graph` draws them:
/// ids 1 to 5 are reached from the entry, 1, each by one way (3 from 1, 4
/// from 2, 5 from 4), 7 and 8 name each other, and no list names 6, 7 or 8.
fn eight_on_an_arc(name: &str) -> ([Vec<f32>; 8], Written) {
    let at = |degrees: f32| vec![degrees.to_radians().cos(), degrees.to_radians().sin()];
    let rows = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0].map(at);
    let levels = [1, 1, 0, 0, 0, 0, 0, 0];
    let adjacency = vec![
        vec![vec![1, 2], vec![1]],
        vec![vec![0, 3], vec![0]],
        vec![vec![0]],
        vec![vec![1, 4]],
        vec![vec![3]],
        vec![vec![0]],
        vec![vec![7]],
        vec![vec![6]],
    ];
    let hnsw = HNSWConfig::new(Some(2), Some(8), Some(4));
    let segment = write_segment_with_lists(name, &hnsw, &rows, &levels, adjacency, 0);
    (rows, segment)
}

/// The rows hard deletes have taken from a segment's graph, whether the link
/// pass has been over it since the last of them, and whether it is due.
fn deleted_done_due(core: &VectorCore) -> (u64, bool, bool) {
    let ro = core.backend.begin_read().unwrap();
    with_small_segment_exact_for_test(false, || {
        (
            core.deleted_graph_rows(&ro).unwrap(),
            core.link_pass_done(&ro).unwrap(),
            core.link_pass_due(&ro).unwrap(),
        )
    })
}

/// A segment the pass has marked is due again when a hard delete has taken
/// a row of its graph since: the row may be one a link came from, and what
/// that link led to is unreached again. The mark carries the count of
/// deleted rows the pass began at, so rows deleted before the pass do not
/// make a segment due.
///
/// A delete takes 3 before the pass, which links 6 and 7 from 5 and marks
/// the segment: one row deleted, and not due. A delete then takes 5. The
/// lists of 6 and 7 are as they were and no row leads to them, nor to 8,
/// which 7 names: the segment is due, and the pass links 6 and 7 from 4.
#[test]
fn a_marked_segment_is_due_again_when_a_delete_has_taken_a_row_since() {
    let (rows, segment) = eight_on_an_arc("due_again");
    let core = &segment.core;
    assert_eq!(deleted_done_due(core), (0, false, true));
    segment.hard_delete(&[3]);
    assert_eq!(run_link_pass(core, usize::MAX), (vec![(5, 6), (5, 7)], 0));
    assert_eq!(
        (deleted_done_due(core), stored_unreached(core)),
        ((1, true, false), vec![])
    );

    segment.hard_delete(&[5]);
    assert_eq!(
        (deleted_done_due(core), stored_unreached(core)),
        ((2, false, true), vec![6, 7, 8])
    );
    assert_eq!(run_link_pass(core, usize::MAX), (vec![(4, 6), (4, 7)], 0));
    assert_eq!(
        (deleted_done_due(core), stored_unreached(core)),
        ((2, true, false), vec![])
    );
    let found = with_small_segment_exact_for_test(false, || {
        with_walk_restart_for_test(false, || unfiltered_hits(core, &rows[5], 1, Some(8)))
    });
    assert_eq!(found.map(|hits| hits[0].0), [6, 6]);
}

/// The count a mark carries is the one the pass began at in the segment, not
/// the one its last call found. A first call, a row a call, links 6 from 5
/// and keeps its place at 6. A delete then takes 4, the one row that led to
/// 5: 5 and 6 are unreached, and both lie before the place. The second call
/// goes on after 6, links 7 and comes to the end. It leaves 5 and 6, which
/// it did not look at, and the segment is due at once: the mark has the
/// count of the first call. The next call links 5, whose list leads to 6.
#[test]
fn a_delete_between_two_calls_leaves_the_segment_due() {
    let (_, segment) = eight_on_an_arc("deleted_between");
    let core = &segment.core;
    // A call as the writer runs it: what it planned, written.
    let call = |max_rows: usize| {
        let plan = plan_link_pass(core, &stored_unreached(core), max_rows);
        let mut w = core.backend.begin_write().unwrap();
        let changed = core
            .apply_stored_links_be(&mut w, &plan.0, plan.2, began_at(core))
            .unwrap();
        core.backend.commit(w).unwrap();
        core.forget_neighbor_lists(&changed);
        plan
    };
    assert_eq!(call(1), (vec![(5, 6)], 0, Some(6)));
    segment.hard_delete(&[4]);
    assert_eq!(stored_unreached(core), [5, 6, 7, 8]);
    assert_eq!(call(1), (vec![(3, 7)], 0, None));
    assert_eq!(
        (deleted_done_due(core), stored_unreached(core)),
        ((1, false, true), vec![5, 6])
    );

    let (links, without_room, upto) = call(usize::MAX);
    assert_eq!(
        (
            links.iter().map(|link| link.1).collect::<Vec<_>>(),
            without_room,
            upto
        ),
        (vec![5], 0, None)
    );
    assert_eq!(
        (deleted_done_due(core), stored_unreached(core)),
        ((1, true, false), vec![])
    );
}

/// A read that began before the pass's write reads a list as it was, and
/// does not put it back into the writer's cache, which dropped the list at
/// the write: a walk would stop short of the link the pass added for as long
/// as the cache kept the older list. The pass links 6 and 7 from 5. A read
/// from before then reads 5's list, without the two. A read from after gets
/// the list with them, and a walk for 6's vector arrives at 6.
#[test]
fn a_read_from_before_the_pass_does_not_cache_the_older_list() {
    let (rows, segment) = eight_on_an_arc("older_list");
    let core = &segment.core;
    let before = core.backend.begin_read().unwrap();
    assert_eq!(run_link_pass(core, usize::MAX), (vec![(5, 6), (5, 7)], 0));
    assert_eq!(core.get_neighbor_block(&before, 5, 0).unwrap().ids, [4]);
    drop(before);
    let after = core.backend.begin_read().unwrap();
    assert_eq!(
        core.get_neighbor_block(&after, 5, 0).unwrap().ids,
        [4, 6, 7]
    );
    drop(after);
    let found = with_small_segment_exact_for_test(false, || {
        with_walk_restart_for_test(false, || unfiltered_hits(core, &rows[5], 1, Some(8)))
    });
    assert_eq!(found.map(|hits| hits[0].0), [6, 6]);
}

/// A row that keeps its list a stored row an edge, the layout before the
/// packed one, is no row to link from. A walk reads either layout, but the
/// write lengthens a packed list and no other: a link planned from such a
/// row was dropped at the write, with the segment marked and the point left
/// uncounted.
///
/// With 5's list in that layout the pass links 6 and 7 from 4, the next
/// nearest. With the lists of all five in it no row can take a link: every
/// row the pass looks at is counted and left.
#[test]
fn a_row_without_a_packed_list_is_not_linked_from() {
    type Case = (&'static str, &'static [u128], Option<Vec<(u128, u128)>>);
    let cases: [Case; 2] = [
        ("an_edge_a_row", &[5], Some(vec![(4, 6), (4, 7)])),
        ("all_an_edge_a_row", &[1, 2, 3, 4, 5], None),
    ];
    for (name, edge_a_row, linked) in cases {
        let (_, segment) = eight_on_an_arc(name);
        let core = &segment.core;
        let lists = level_zero_lists(core, edge_a_row.iter().copied());
        let mut w = core.backend.begin_write().unwrap();
        for (&row, list) in &lists {
            core.backend
                .delete(
                    &mut w,
                    core.seg_ns(SegmentDb::HnswNeighbors),
                    &VectorCore::neighbor_list_key(row, 0),
                )
                .unwrap();
            for &named in list {
                core.backend
                    .put(
                        &mut w,
                        core.seg_ns(SegmentDb::HnswOut),
                        &VectorCore::out_edges_key(row, 0, Some(named)),
                        &[],
                    )
                    .unwrap();
            }
        }
        core.backend.commit(w).unwrap();
        assert_eq!(
            level_zero_lists(core, edge_a_row.iter().copied()),
            lists,
            "{name}: the lists as a walk reads them"
        );

        let unreached = stored_unreached(core);
        let (links, without_room) = run_link_pass(core, usize::MAX);
        match linked {
            Some(linked) => {
                assert_eq!((links, without_room), (linked, 0), "{name}");
                assert_eq!(stored_unreached(core), Vec::<u128>::new(), "{name}");
            }
            None => {
                assert_eq!((links, without_room), (vec![], unreached.len()), "{name}");
                assert_eq!(stored_unreached(core), unreached, "{name}");
            }
        }
    }
}

/// Rows on the ring of `full_ring`.
const RING: usize = 64;

/// Where the rows beside a ring of [`RING`] rows lie, in degrees: twelve in
/// the gap after the ring's eleventh row, all nearer to it than to the
/// twelfth, and eight more in a gap each.
fn beside_the_ring() -> Vec<f32> {
    let step = 360.0 / RING as f32;
    (1..=12)
        .map(|k| 10.0 * step + 0.2 * k as f32)
        .chain((20..60).step_by(5).map(|gap| gap as f32 * step + 2.8))
        .collect()
}

/// A stored graph with every list full, as the builds from before the
/// linking step left them: `ring` rows around a circle (ids 1 to `ring`),
/// each naming the `m_max_0` rows nearest to it, and after them a row at
/// each of `beside` (degrees), which names the `m_max_0` ring rows nearest
/// to it and which no list names. The entry is row 1, alone on level 1.
/// Returns the rows' vectors with the segment.
fn full_ring(
    name: &str,
    hnsw: &HNSWConfig,
    ring: usize,
    beside: &[f32],
) -> (Vec<Vec<f32>>, Written) {
    let step = 360.0 / ring as f32;
    let angles: Vec<f32> = (0..ring)
        .map(|ord| ord as f32 * step)
        .chain(beside.iter().copied())
        .collect();
    let rows: Vec<Vec<f32>> = angles
        .iter()
        .map(|angle| vec![angle.to_radians().cos(), angle.to_radians().sin()])
        .collect();
    // The ring rows nearest to a row, nearest first, itself left out.
    let nearest = |ord: usize| -> Vec<u32> {
        let apart = |other: usize| {
            let turn = (angles[ord] - angles[other]).abs() % 360.0;
            turn.min(360.0 - turn)
        };
        let mut others: Vec<usize> = (0..ring).filter(|&other| other != ord).collect();
        others.sort_by(|&a, &b| apart(a).total_cmp(&apart(b)));
        others.truncate(hnsw.m_max_0);
        others.into_iter().map(|other| other as u32).collect()
    };
    let adjacency = (0..rows.len())
        .map(|ord| match ord {
            0 => vec![nearest(ord), Vec::new()],
            _ => vec![nearest(ord)],
        })
        .collect();
    let mut levels = vec![0; rows.len()];
    levels[0] = 1;
    let written = write_segment_with_lists(name, hnsw, &rows, &levels, adjacency, 0);
    (rows, written)
}

/// The graphs built before the builds linked their own unreached points
/// have every near list full, and the pass as it was linked only under what
/// a build fills a list to: it looked at their unreached rows, linked none
/// and marked the segments. It now puts its link past a full list, as far
/// as the bound.
///
/// A ring of 64 rows that each name their 32 nearest, and 20 rows beside it
/// with full lists of their own that no list names: twelve in one gap of
/// the ring, all nearest to the same ring row, and eight in a gap each.
/// After the pass no row is unreached, none was left without room, and the
/// segment is marked. Every list begins with the list it was, so a walk
/// that holds an older copy follows nothing that is not there, and none is
/// past the bound.
///
/// In one call the twelve take the room of their nearest ring row, eight
/// places, and the last four go to the next nearest. Five rows a call, a row
/// an earlier call linked is itself a row with room for the next.
#[test]
fn the_link_pass_links_past_the_lists_a_build_filled() {
    let hnsw = HNSWConfig::new(Some(16), Some(64), Some(64));
    let bound = link_pass_list_bound(hnsw.m_max_0);
    assert_eq!((hnsw.m_max_0, bound), (32, 40));
    let unreached: Vec<u128> = (65..=84).collect();
    for (name, max_rows) in [("full_one_call", usize::MAX), ("full_five_a_call", 5)] {
        let (_, segment) = full_ring(name, &hnsw, RING, &beside_the_ring());
        let core = &segment.core;
        let before = level_zero_lists(core, 1..=84);
        assert!(
            before.values().all(|list| list.len() == hnsw.m_max_0),
            "{name}: a list is not full"
        );
        assert_eq!(stored_unreached(core), unreached, "{name}");

        let (links, without_room) = run_link_pass(core, max_rows);
        let after = level_zero_lists(core, 1..=84);
        let mut wrong = Vec::new();
        if (links.len(), without_room) != (20, 0) {
            wrong.push(format!(
                "{} rows linked, {without_room} without room",
                links.len()
            ));
        }
        let left = stored_unreached(core);
        if !left.is_empty() {
            wrong.push(format!("still unreached: {left:?}"));
        }
        if !core
            .link_pass_done(&core.backend.begin_read().unwrap())
            .unwrap()
        {
            wrong.push("the segment is not marked".to_string());
        }
        let mut added = 0;
        for (id, was) in &before {
            let now = &after[id];
            if !now.starts_with(was) {
                wrong.push(format!("{id}: {was:?} is not how {now:?} begins"));
            }
            if now.len() > bound {
                wrong.push(format!("{id}: {} ids, past the bound", now.len()));
            }
            added += now.len().saturating_sub(was.len());
        }
        if added != links.len() {
            wrong.push(format!("{added} ids added for {} links", links.len()));
        }
        let the_twelve = (after[&11].len(), after[&12].len());
        if max_rows == usize::MAX && the_twelve != (40, 36) {
            wrong.push(format!("the twelve were given {the_twelve:?} places"));
        }
        assert!(wrong.is_empty(), "{name}: {wrong:#?}");
    }
}

/// A read pinned to the snapshot a reader replica has now, as production's
/// reads are.
fn pinned(reader: &VectorCore) -> AnyRead<'_> {
    let AnyBackend::LsmReader(lsm_reader) = &*reader.backend else {
        unreachable!()
    };
    AnyRead::LsmReader(Some(lsm_reader.begin_snapshot().unwrap()))
}

/// A row the pass linked past a full list comes back for its own vector: on
/// the writer, which drops the lists it changed from its cache, and on a
/// reader replica that built its file of vectors before the pass ran, as it
/// is and opened again in the directory it left. The ring and the 20 rows
/// beside it of `the_link_pass_links_past_the_lists_a_build_filled`.
///
/// A reader that walked the graph before the pass holds the lists it read as
/// they were. Its count of unreached rows is read from the store and falls
/// to 0 with the writer's, while its walks follow the older lists and return
/// none of the 20: that is what the readers' restart after a run of the pass
/// is for. Opened again it returns them all.
#[test]
fn a_row_linked_past_a_full_list_comes_back_for_its_own_vector() {
    let hnsw = HNSWConfig::new(Some(16), Some(64), Some(64));
    let (vectors, segment) = full_ring("own_vector", &hnsw, RING, &beside_the_ring());
    let core = &segment.core;
    let linked: Vec<u128> = (65..=84).collect();
    // Of the 20, how many come back first for their own vector by both
    // routes of a search through `read`, the graph walked.
    let found = |core: &VectorCore, read: &AnyRead<'_>| {
        linked
            .iter()
            .filter(|&&id| {
                let hits = with_small_segment_exact_for_test(false, || {
                    with_walk_restart_for_test(false, || {
                        unfiltered_hits_at(core, read, &vectors[id as usize - 1], 1, Some(64))
                    })
                });
                hits.iter()
                    .all(|route| route.first().map(|hit| hit.0) == Some(id))
            })
            .count()
    };
    let on_writer = || found(core, &core.backend.begin_read().unwrap());
    let refresh = |reader: &VectorCore| {
        reader
            .backend
            .refresh_lsm_reader_with_poll_interval(std::time::Duration::from_millis(100))
            .unwrap()
    };

    // Two readers open before the pass. One builds its file of vectors and
    // reads no list. The other walks for each of the 20, so it holds the
    // lists of the rows nearest to them.
    let (reader, reader_dir) = segment.reader(2);
    assert!(
        reader.ensure_lsm_mmap_sidecar().unwrap() && reader.has_mmap_sidecar(),
        "the reader built no file of vectors"
    );
    let rows_in_file = reader.mmap_store_count();
    let (walked, walked_dir) = segment.reader(2);
    assert_eq!((on_writer(), found(&walked, &pinned(&walked))), (0, 0));

    assert_eq!(run_link_pass(core, usize::MAX).0.len(), 20);
    assert_eq!(on_writer(), 20, "on the writer");

    refresh(&reader);
    assert_eq!(
        found(&reader, &pinned(&reader)),
        20,
        "on a reader with its file from before the pass"
    );
    assert_eq!(
        reader.mmap_store_count(),
        rows_in_file,
        "rows in the reader's file of vectors"
    );
    drop(reader);
    let reopened = segment.reader_in(2, reader_dir.path());
    assert_eq!(
        found(&reopened, &pinned(&reopened)),
        20,
        "on that reader opened again"
    );

    refresh(&walked);
    assert_eq!(
        (
            stored_unreached(&walked).len(),
            found(&walked, &pinned(&walked))
        ),
        (0, 0),
        "on a reader that holds the older lists"
    );
    drop(walked);
    let reopened = segment.reader_in(2, walked_dir.path());
    assert_eq!(
        found(&reopened, &pinned(&reopened)),
        20,
        "on the reader that held them, opened again"
    );
}

/// A row gets no link once eight rows with its own vector are arrived at
/// (`LINK_PASS_TWINS`): a group of rows with one vector that are all linked
/// takes every place of a walk that comes by, and on a copy of a production
/// collection replies lost rows of their nearest ten that way. A search with
/// no filter is answered by the eight; the others are counted and left.
///
/// 333 rows with one vector. The first 33 each name the 32 others, as full
/// as a build leaves a list, and are what a walk arrives at. Each of the
/// other 300 names 32 of them and no list names it. A walk for the vector
/// returns eight rows with it: none of the 300 is linked, no list changes,
/// and the segment is marked.
#[test]
fn the_link_pass_leaves_a_row_eight_rows_of_whose_vector_are_arrived_at() {
    let hnsw = HNSWConfig::new(Some(16), Some(64), Some(64));
    let (reached, more) = (33u32, 300u32);
    let rows = vec![vec![1.0, 0.0]; (reached + more) as usize];
    let mut levels = vec![0; rows.len()];
    levels[0] = 1;
    let adjacency = (0..reached + more)
        .map(|ord| {
            let level_zero: Vec<u32> = (0..reached)
                .filter(|&other| other != ord)
                .take(hnsw.m_max_0)
                .collect();
            match ord {
                0 => vec![level_zero, Vec::new()],
                _ => vec![level_zero],
            }
        })
        .collect();
    let segment = write_segment_with_lists("one_vector", &hnsw, &rows, &levels, adjacency, 0);
    let core = &segment.core;
    let unreached: Vec<u128> = (34..=333).collect();
    let before = level_zero_lists(core, 1..=333);
    assert_eq!(stored_unreached(core), unreached);

    let (links, without_room) = run_link_pass(core, 256);
    assert_eq!((links, without_room), (vec![], 300));
    assert_eq!(
        (stored_unreached(core), level_zero_lists(core, 1..=333)),
        (unreached, before)
    );
    let ro = core.backend.begin_read().unwrap();
    assert!(core.link_pass_done(&ro).unwrap());
}

/// The pass links rows with one vector until eight of them are arrived at,
/// the rows a call has linked counted with those a walk returns, which does
/// not see them before the call's write.
///
/// Twelve rows ten degrees apart that each name the eleven others, and 40
/// rows with one vector fifteen degrees past the twelfth, which no list
/// names. The first eight get a link and the other 32 are counted and left:
/// in one call, and three rows a call, where the walks of a later call
/// return the rows the earlier ones linked.
#[test]
fn the_link_pass_links_rows_with_one_vector_until_eight_are_arrived_at() {
    let hnsw = HNSWConfig::new(Some(16), Some(64), Some(64));
    let at = |degrees: f32| vec![degrees.to_radians().cos(), degrees.to_radians().sin()];
    let (others, twins) = (12u32, 40u32);
    let rows: Vec<Vec<f32>> = (0..others)
        .map(|ord| at(ord as f32 * 10.0))
        .chain((0..twins).map(|_| at(125.0)))
        .collect();
    let mut levels = vec![0; rows.len()];
    levels[0] = 1;
    for (name, max_rows) in [("twins_one_call", 256), ("twins_three_a_call", 3)] {
        let adjacency = (0..others + twins)
            .map(|ord| {
                let level_zero: Vec<u32> = if ord < others {
                    (0..others).filter(|&other| other != ord).collect()
                } else {
                    vec![0, 1, 2, 3]
                };
                match ord {
                    0 => vec![level_zero, Vec::new()],
                    _ => vec![level_zero],
                }
            })
            .collect();
        let segment = write_segment_with_lists(name, &hnsw, &rows, &levels, adjacency, 0);
        let core = &segment.core;
        assert_eq!(
            stored_unreached(core),
            (13..=52).collect::<Vec<u128>>(),
            "{name}"
        );
        let (links, without_room) = run_link_pass(core, max_rows);
        assert_eq!(
            (
                links.iter().map(|link| link.1).collect::<Vec<_>>(),
                without_room
            ),
            ((13..=20).collect::<Vec<u128>>(), 32),
            "{name}"
        );
        assert_eq!(
            stored_unreached(core),
            (21..=52).collect::<Vec<u128>>(),
            "{name}"
        );
    }
}

/// Under a dot product the distance of two rows is their product negated,
/// and says nothing of their having one vector: the pass counts no row
/// there, and links as it did. Twelve rows ten degrees apart that each name
/// the eleven others, and a row five degrees past the twelfth that no list
/// names: its product with each of the eight nearest is above zero.
#[test]
fn the_link_pass_counts_no_rows_with_one_vector_under_a_dot_product() {
    let hnsw = HNSWConfig::new(Some(16), Some(64), Some(64));
    let at = |degrees: f32| vec![degrees.to_radians().cos(), degrees.to_radians().sin()];
    let rows: Vec<Vec<f32>> = (0..12)
        .map(|ord| at(ord as f32 * 10.0))
        .chain([at(115.0)])
        .collect();
    let mut levels = vec![0; rows.len()];
    levels[0] = 1;
    let adjacency = (0..13u32)
        .map(|ord| {
            let level_zero: Vec<u32> = if ord < 12 {
                (0..12).filter(|&other| other != ord).collect()
            } else {
                vec![0, 1, 2, 3]
            };
            match ord {
                0 => vec![level_zero, Vec::new()],
                _ => vec![level_zero],
            }
        })
        .collect();
    let segment = write_segment_with_lists_under(
        DistanceMetric::Dot,
        "dot_product",
        &hnsw,
        &rows,
        &levels,
        adjacency,
        0,
    );
    let core = &segment.core;
    assert_eq!(stored_unreached(core), [13]);
    let (links, without_room) = run_link_pass(core, usize::MAX);
    assert_eq!(
        (
            links.iter().map(|link| link.1).collect::<Vec<_>>(),
            without_room
        ),
        (vec![13], 0)
    );
}

/// A row linked from a row below its nearest comes back for its own vector.
/// A ring of 24 rows that each name their 16 nearest, and 36 rows beside the
/// eleventh, all nearer to it than to the twelfth, with full lists of their
/// own that no list names. A list has room for four ids past the 16.
///
/// The 36 have the same eight nearest ring rows, in the same order. The
/// first 32 take the four places of each, the nearest ring row first. For
/// the last four all eight are at their bound, and their links come from
/// the four rows the eleventh was given. No row is left, no list is past its
/// bound and each begins with the list it was. Every one of the 36 then
/// comes back first for its own vector, by both routes, from a search that
/// ends with as many rows as the segment's ef.
#[test]
fn a_row_linked_from_a_row_below_comes_back_for_its_own_vector() {
    let hnsw = HNSWConfig::new(Some(8), Some(64), Some(128));
    let bound = link_pass_list_bound(hnsw.m_max_0);
    assert_eq!((hnsw.m_max_0, bound), (16, 20));
    let ring = 24;
    let step = 360.0 / ring as f32;
    let beside: Vec<f32> = (1..=36).map(|k| 10.0 * step + 0.2 * k as f32).collect();
    let (vectors, segment) = full_ring("below", &hnsw, ring, &beside);
    let core = &segment.core;
    let rows: Vec<u128> = (25..=60).collect();
    let before = level_zero_lists(core, 1..=60);
    assert!(before.values().all(|list| list.len() == hnsw.m_max_0));
    assert_eq!(stored_unreached(core), rows);

    let (links, without_room) = run_link_pass(core, usize::MAX);
    assert_eq!(
        (links.len(), without_room, stored_unreached(core)),
        (36, 0, vec![])
    );
    // The rows the links come from, each once, ascending.
    let hosts = |links: &[(u128, u128)]| -> Vec<u128> {
        let mut hosts: Vec<u128> = links.iter().map(|link| link.0).collect();
        hosts.sort_unstable();
        hosts.dedup();
        hosts
    };
    let (from_the_ring, from_below) = links.split_at(32);
    assert_eq!(hosts(from_the_ring), (8..=15).collect::<Vec<u128>>());
    assert_eq!(from_the_ring[..4], [(11, 25), (11, 26), (11, 27), (11, 28)]);
    assert!(
        hosts(from_below)
            .iter()
            .all(|host| (25..=28).contains(host)),
        "{from_below:?}"
    );
    let after = level_zero_lists(core, 1..=60);
    let changed_wrongly: Vec<&u128> = before
        .iter()
        .filter(|&(id, was)| !after[id].starts_with(was) || after[id].len() > bound)
        .map(|(id, _)| id)
        .collect();
    assert!(changed_wrongly.is_empty(), "{changed_wrongly:?}");

    // The rows that do not come back first for their own vector by both
    // routes.
    let not_first: Vec<u128> = rows
        .into_iter()
        .filter(|&row| {
            let found = with_small_segment_exact_for_test(false, || {
                with_walk_restart_for_test(false, || {
                    unfiltered_hits(core, &vectors[row as usize - 1], 1, None)
                })
            });
            found.map(|hits| hits[0].0) != [row, row]
        })
        .collect();
    assert!(not_first.is_empty(), "{not_first:?}");
}

/// A segment an earlier pass has been over is due again, and one it left
/// part of the way through starts again from its first row: the first pass
/// linked only under what a build fills a list to and the second tried a
/// row's eight nearest rows alone, so the rows they looked at and left are
/// still to do. Ids 1 to 7: each of 1 to 5 names the four others, as many as
/// a build writes, and no list names 6 or 7. The first pass found no room
/// for either and marked the segment.
///
/// This pass keeps its place with its own number, takes it up at the next
/// call, and marks the segment when it comes to the end. A mark of a later
/// pass reads as done.
#[test]
fn a_segment_an_earlier_pass_marked_is_due_again() {
    let hnsw = HNSWConfig::new(Some(2), Some(8), Some(4));
    let segment = each_names_the_others("marked", &hnsw, 5, 2);
    let core = &segment.core;
    let stage = |mark: &[u8]| {
        let mut w = core.backend.begin_write().unwrap();
        core.put_link_pass_mark_for_test(&mut w, mark);
        core.backend.commit(w).unwrap();
    };
    // Whether the pass has been here, whether the segment is due, and the
    // row a call would go on after.
    let state = || {
        let ro = core.backend.begin_read().unwrap();
        with_small_segment_exact_for_test(false, || {
            (
                core.link_pass_done(&ro).unwrap(),
                core.link_pass_due(&ro).unwrap(),
                core.link_pass_after(&ro).unwrap(),
            )
        })
    };
    const DUE: (bool, bool, Option<u128>) = (false, true, None);
    assert_eq!(state(), DUE, "with no mark");
    for pass in 1..LINK_PASS_NUMBER {
        stage(&[pass]);
        assert_eq!(state(), DUE, "marked by pass {pass}");
    }
    stage(&[&[0u8][..], &6u128.to_be_bytes()].concat());
    assert_eq!(state(), DUE, "left at row 6 by the first pass");
    stage(&[&[0u8][..], &6u128.to_be_bytes(), &[2]].concat());
    assert_eq!(state(), DUE, "left at row 6 by the second pass");
    // This pass's own row without a count of deleted rows, as it wrote it
    // before it kept one: the place is taken up, and the mark reads as done
    // where no row has been deleted.
    stage(&[&[0u8][..], &6u128.to_be_bytes(), &[LINK_PASS_NUMBER]].concat());
    assert_eq!(state(), (false, true, Some(6)), "a place without a count");
    stage(&[LINK_PASS_NUMBER]);
    assert_eq!(state(), (true, false, None), "a mark without a count");
    stage(&[2]);

    // One call of this pass, a row a call: the links it wrote.
    let call = || {
        let unreached = stored_unreached(core);
        let ro = core.backend.begin_read().unwrap();
        let (links, _, upto) = with_small_segment_exact_for_test(false, || {
            let after = core.link_pass_after(&ro).unwrap();
            core.plan_stored_links(&ro, &unreached, after, 1).unwrap()
        });
        drop(ro);
        let mut w = core.backend.begin_write().unwrap();
        let changed = core
            .apply_stored_links_be(&mut w, &links, upto, began_at(core))
            .unwrap();
        core.backend.commit(w).unwrap();
        core.forget_neighbor_lists(&changed);
        links
    };
    assert_eq!(call(), [(5, 6)]);
    assert_eq!(
        state(),
        (false, true, Some(6)),
        "left at row 6 by this pass"
    );
    assert_eq!(call(), [(6, 7)]);
    assert_eq!(
        (state(), stored_unreached(core)),
        ((true, false, None), vec![])
    );
    stage(&[LINK_PASS_NUMBER + 1]);
    assert_eq!(state(), (true, false, None), "marked by a later pass");
}

/// What the pass planned for a list is written at the end of the list as it
/// is found at the write, and not at all when by then the list is at its
/// bound, names the point, or is gone. The drawn graph of
/// `the_link_pass_links_the_unreached_rows_of_a_stored_graph`, where one
/// plan puts 6 and 7 at the end of 5's list, and that list as each case
/// finds it at the write: the rows the write changed, and the list after.
#[test]
fn a_list_at_its_bound_by_the_time_of_the_write_is_left() {
    let at = |degrees: f32| vec![degrees.to_radians().cos(), degrees.to_radians().sin()];
    let rows = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0].map(at);
    let levels = [1, 1, 0, 0, 0, 0, 0, 0];
    let adjacency = || {
        vec![
            vec![vec![1, 2], vec![1]],
            vec![vec![0, 3], vec![0]],
            vec![vec![0]],
            vec![vec![1, 4]],
            vec![vec![3]],
            vec![vec![0]],
            vec![vec![7]],
            vec![vec![6]],
        ]
    };
    let hnsw = HNSWConfig::new(Some(2), Some(8), Some(4));
    assert_eq!(link_pass_list_bound(hnsw.m_max_0), 5);
    // What the write finds, what it changes, what it leaves.
    type Case = (
        &'static str,
        Option<Vec<u128>>,
        Vec<u128>,
        Option<Vec<u128>>,
    );
    let cases: [Case; 5] = [
        (
            "as it was planned",
            Some(vec![4]),
            vec![5, 5],
            Some(vec![4, 6, 7]),
        ),
        (
            "as full as a build leaves one",
            Some(vec![4, 1, 2, 3]),
            vec![5],
            Some(vec![4, 1, 2, 3, 6]),
        ),
        (
            "at its bound",
            Some(vec![4, 1, 2, 3, 8]),
            vec![],
            Some(vec![4, 1, 2, 3, 8]),
        ),
        ("naming 6", Some(vec![4, 6]), vec![5], Some(vec![4, 6, 7])),
        ("gone", None, vec![], None),
    ];
    for (index, (case, found, changed, left)) in cases.into_iter().enumerate() {
        let name = format!("write_finds_{index}");
        let segment = write_segment_with_lists(&name, &hnsw, &rows, &levels, adjacency(), 0);
        let core = &segment.core;
        let ro = core.backend.begin_read().unwrap();
        let (links, _, upto) = with_small_segment_exact_for_test(false, || {
            core.plan_stored_links(&ro, &[6, 7, 8], None, usize::MAX)
                .unwrap()
        });
        drop(ro);
        assert_eq!(links, [(5, 6), (5, 7)], "{case}");

        let key = VectorCore::neighbor_list_key(5, 0);
        let mut w = core.backend.begin_write().unwrap();
        match found {
            Some(ids) => core.backend.put(
                &mut w,
                core.seg_ns(SegmentDb::HnswNeighbors),
                &key,
                &VectorCore::encode_neighbor_block(&NeighborBlock::from_ids(ids)),
            ),
            None => core
                .backend
                .delete(&mut w, core.seg_ns(SegmentDb::HnswNeighbors), &key),
        }
        .unwrap();
        core.backend.commit(w).unwrap();

        let mut w = core.backend.begin_write().unwrap();
        let wrote = core
            .apply_stored_links_be(&mut w, &links, upto, began_at(core))
            .unwrap();
        core.backend.commit(w).unwrap();
        let ro = core.backend.begin_read().unwrap();
        let now = core
            .packed_neighbor_block(&ro, 5, 0)
            .unwrap()
            .map(|list| list.ids);
        assert_eq!((wrote, now), (changed, left), "5's list {case}");
    }
}

/// The rows hard deletes take from a graph are counted in the segment
/// (`deleted_graph_rows`), a row once however often it is deleted, for the
/// ceiling past which the segment is rewritten. Ten points, each one's list
/// naming the next: two gone is a fifth of what was built and not past a
/// ceiling of a fifth, three is. Asked for a segment that still has eight
/// rows, the seven that are left are too few and it is not past it.
#[test]
fn hard_deletes_of_graph_rows_are_counted_for_the_merge_ceiling() {
    let at = |degrees: f32| vec![degrees.to_radians().cos(), degrees.to_radians().sin()];
    let rows: Vec<Vec<f32>> = (0..10).map(|point| at(point as f32 * 8.0)).collect();
    let mut levels = vec![0; 10];
    levels[0] = 1;
    let adjacency: Vec<Vec<Vec<u32>>> = (0..10u32)
        .map(|ord| match ord {
            0 => vec![vec![1], vec![]],
            _ => vec![vec![(ord + 1) % 10]],
        })
        .collect();
    let hnsw = HNSWConfig::new(Some(2), Some(8), Some(4));
    let segment = write_segment_with_lists("counted", &hnsw, &rows, &levels, adjacency, 0);
    let core = &segment.core;
    let delete = |ids: &[u128]| segment.hard_delete(ids);
    // Whether it is past a ceiling of a fifth with `min_left` rows or more
    // left.
    let over = |min_left: u64| {
        let ro = core.backend.begin_read().unwrap();
        core.deleted_graph_rows_over(&ro, 0.2, min_left).unwrap()
    };
    // The count, and whether it is past the ceiling however few are left.
    let counted = || {
        let ro = core.backend.begin_read().unwrap();
        (core.deleted_graph_rows(&ro).unwrap(), over(0))
    };
    assert_eq!(counted(), (0, false));
    // 4 is named twice in the batch and is one row.
    delete(&[4, 5, 4]);
    assert_eq!(counted(), (2, false));
    delete(&[5]);
    assert_eq!(counted(), (2, false));
    delete(&[6]);
    assert_eq!(counted(), (3, true));
    assert_eq!((over(7), over(8)), (true, false));
}

/// The entry point after a hard delete or a publish, on this module's
/// segment and recall harness.
#[path = "entry_repoint_tests.rs"]
mod entry_repoint_tests;
