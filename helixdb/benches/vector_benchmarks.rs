use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use heed3::{Env, EnvOpenOptions};
use helixdb::helix_engine::storage_core::backend::StorageBackend;
use helixdb::helix_engine::storage_core::backend_any::AnyBackend;
use helixdb::helix_engine::storage_core::backend_lmdb::LmdbBackend;
use helixdb::helix_engine::vector_core::{
    hnsw::HNSW,
    named_vectors::DistanceMetric,
    spindle::{decode_vector, encode_vector, project_for_search, SpindleConfig, SpindleMode},
    vector::HVector,
    vector_core::{HNSWConfig, VectorCore},
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cell::OnceCell;
use std::collections::HashSet;
use std::mem::size_of;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;

type VectorFilter = fn(&HVector) -> bool;

/// Shared backend handle for bench cores (US-006 plumbing). 6a is pure
/// plumbing — cores hold this Arc but no KV access routes through it yet.
fn bench_backend(env: &Env) -> Arc<AnyBackend> {
    Arc::new(AnyBackend::Lmdb(LmdbBackend::from_env(env.clone())))
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

#[derive(Clone)]
struct BenchVariant {
    name: String,
    spindle: SpindleConfig,
    stored_dims: usize,
}

fn raw_variant(dim: usize) -> BenchVariant {
    BenchVariant {
        name: "raw_full".to_string(),
        spindle: SpindleConfig {
            keep_original: false,
            rescore: false,
            ..SpindleConfig::default()
        },
        stored_dims: dim,
    }
}

fn scalar_variant(dim: usize, keep_original: bool) -> BenchVariant {
    BenchVariant {
        name: format!(
            "scalar_int8_full_{}",
            if keep_original { "exact" } else { "approx" }
        ),
        spindle: SpindleConfig {
            mode: SpindleMode::ScalarInt8,
            keep_original,
            rescore: keep_original,
            ..SpindleConfig::default()
        },
        stored_dims: dim,
    }
}

fn binary_variant(stored_dims: usize, keep_original: bool) -> BenchVariant {
    BenchVariant {
        name: format!(
            "binary_sign_{stored_dims}d_{}",
            if keep_original { "exact" } else { "approx" }
        ),
        spindle: SpindleConfig {
            mode: SpindleMode::BinarySign,
            keep_original,
            rescore: true,
            binary_dims: stored_dims,
            ..SpindleConfig::default()
        },
        stored_dims,
    }
}

fn turbo_variant(stored_dims: usize, keep_original: bool) -> BenchVariant {
    BenchVariant {
        name: format!(
            "turbo_int4_{stored_dims}d_{}",
            if keep_original { "exact" } else { "approx" }
        ),
        spindle: SpindleConfig {
            mode: SpindleMode::TurboInt4,
            keep_original,
            rescore: true,
            binary_dims: stored_dims,
            turbo_dims: stored_dims,
            ..SpindleConfig::default()
        },
        stored_dims,
    }
}

fn turbo_prod_variant(stored_dims: usize, keep_original: bool) -> BenchVariant {
    BenchVariant {
        name: format!(
            "turbo_prod_{stored_dims}d_{}",
            if keep_original { "exact" } else { "approx" }
        ),
        spindle: SpindleConfig {
            mode: SpindleMode::TurboProd,
            keep_original,
            rescore: true,
            turbo_dims: stored_dims,
            ..SpindleConfig::default()
        },
        stored_dims,
    }
}

fn setup_temp_env() -> (Env, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().to_str().unwrap();

    let env = unsafe {
        EnvOpenOptions::new()
            .map_size(1024 * 1024 * 1024)
            .max_dbs(64)
            .open(path)
            .unwrap()
    };

    (env, temp_dir)
}

fn setup_incremental_search_core(
    count: usize,
    dim: usize,
    seed: u64,
    ef: usize,
) -> (Env, TempDir, VectorCore, Vec<Vec<f32>>) {
    let (env, temp_dir) = setup_temp_env();
    let mut txn = env.write_txn().unwrap();
    let core = VectorCore::new(
        &env,
        &mut txn,
        HNSWConfig::new(Some(16), Some(128), Some(ef)),
        bench_backend(&env),
    )
    .unwrap();

    for (i, vector) in generate_random_vectors(count, dim, seed)
        .into_iter()
        .enumerate()
    {
        core.insert::<VectorFilter>(&mut txn, &vector, Some((i + 1) as u128), None)
            .unwrap();
    }
    txn.commit().unwrap();

    let queries = generate_random_vectors(20, dim, 99);
    (env, temp_dir, core, queries)
}

fn setup_indexed_search_core(
    count: usize,
    dim: usize,
    seed: u64,
    mmap: bool,
    name: &str,
) -> (Env, TempDir, VectorCore, Vec<Vec<f32>>) {
    let (env, temp_dir) = setup_temp_env();
    let mut txn = env.write_txn().unwrap();
    let config = HNSWConfig::new(Some(16), Some(128), Some(128));
    let core = if mmap {
        VectorCore::new_named_with_dir(
            &env,
            &mut txn,
            name,
            config,
            DistanceMetric::Cosine,
            SpindleConfig {
                mode: SpindleMode::None,
                keep_original: false,
                ..SpindleConfig::default()
            },
            Some(temp_dir.path()),
            dim,
            None,
            bench_backend(&env),
        )
        .unwrap()
    } else {
        VectorCore::new(&env, &mut txn, config, bench_backend(&env)).unwrap()
    };

    for (i, vector) in generate_random_vectors(count, dim, seed)
        .into_iter()
        .enumerate()
    {
        core.insert_flat(&mut txn, &vector, Some((i + 1) as u128), None)
            .unwrap();
    }
    core.build_index_from_flat(&mut txn).unwrap();
    txn.commit().unwrap();
    core.flush_mmap().unwrap();

    let queries = generate_random_vectors(20, dim, 99);
    (env, temp_dir, core, queries)
}

fn setup_lsm_vector_core(name: &str) -> (Env, TempDir, Arc<AnyBackend>, VectorCore) {
    let (env, temp_dir) = setup_temp_env();
    let backend = Arc::new(
        AnyBackend::open_lsm_in_memory(&format!("/vector-bench-{name}"))
            .expect("open in-memory LSM benchmark backend"),
    );
    let mut txn = env.write_txn().unwrap();
    let core = VectorCore::new_named(
        &env,
        &mut txn,
        name,
        HNSWConfig::new(Some(16), Some(128), Some(128)),
        DistanceMetric::Cosine,
        SpindleConfig {
            mode: SpindleMode::None,
            keep_original: false,
            ..SpindleConfig::default()
        },
        Arc::clone(&backend),
    )
    .unwrap();
    txn.commit().unwrap();
    (env, temp_dir, backend, core)
}

fn setup_lsm_flat_search_core(
    count: usize,
    dim: usize,
    query_count: usize,
) -> (Env, TempDir, Arc<AnyBackend>, VectorCore, Vec<Vec<f32>>) {
    let (env, temp_dir, backend, core) =
        setup_lsm_vector_core(&format!("lsm_flat_search_{count}_{dim}"));
    let vectors = generate_random_vectors(count, dim, 42);
    let mut w = backend.begin_write().unwrap();
    for (i, vector) in vectors.iter().enumerate() {
        core.insert_write_context::<VectorFilter>(&mut w, vector, Some((i + 1) as u128), None)
            .unwrap();
    }
    backend.commit(w).unwrap();
    let queries = generate_random_vectors(query_count, dim, 99);
    (env, temp_dir, backend, core, queries)
}

fn even_candidate_set(count: usize) -> HashSet<u128> {
    (1..=count as u128).filter(|id| id % 2 == 0).collect()
}

fn generate_random_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vectors = Vec::with_capacity(count);

    for _ in 0..count {
        let data: Vec<f32> = (0..dim)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect();
        vectors.push(data);
    }

    vectors
}

/// Generate low-rank vectors that mimic real embeddings (e.g. OpenAI, GloVe).
/// Real 768d embeddings have ~50-100 effective dimensions — most variance lives in
/// a low-rank subspace. This gives cosine 0.3-0.9 for true neighbors vs ~0.0 for
/// random pairs, creating the wide margins where quantization works well.
fn generate_lowrank_vectors(count: usize, dim: usize, rank: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);

    // Random projection: rank → dim (use f64 for precision, narrow at end)
    let proj: Vec<Vec<f64>> = (0..rank)
        .map(|_| {
            (0..dim)
                .map(|_| rng.random_range(-1.0..1.0) / (dim as f64).sqrt())
                .collect()
        })
        .collect();

    let mut vectors = Vec::with_capacity(count);
    for _ in 0..count {
        let latent: Vec<f64> = (0..rank).map(|_| rng.random_range(-1.0..1.0)).collect();
        let mut vec = vec![0.0f64; dim];
        for (r, &lat) in latent.iter().enumerate() {
            for d in 0..dim {
                vec[d] += lat * proj[r][d];
            }
        }
        for d in 0..dim {
            vec[d] += rng.random_range(-0.01..0.01);
        }
        let norm = vec.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }
        vectors.push(vec.into_iter().map(|v| v as f32).collect());
    }
    vectors
}

/// Generate clustered vectors that mimic real embedding distributions.
/// Creates `n_clusters` random cluster centers on the unit sphere, then
/// generates vectors near each center with `spread` controlling intra-cluster noise.
/// Result: true neighbors share a cluster (high cosine ~0.7-0.95), cross-cluster pairs
/// have low cosine (~0.0-0.3), giving the wide distance margins that real embeddings have.
fn generate_clustered_vectors(
    count: usize,
    dim: usize,
    n_clusters: usize,
    spread: f64,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);

    // random unit-sphere cluster centers (f64 for precision)
    let centers: Vec<Vec<f64>> = (0..n_clusters)
        .map(|_| {
            let raw: Vec<f64> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let norm = raw.iter().map(|v| v * v).sum::<f64>().sqrt();
            raw.into_iter().map(|v| v / norm).collect()
        })
        .collect();

    let mut vectors = Vec::with_capacity(count);
    for _ in 0..count {
        let ci = rng.random_range(0..n_clusters);
        let center = &centers[ci];
        let data: Vec<f64> = center
            .iter()
            .map(|c| c + rng.random_range(-spread..spread))
            .collect();
        // normalize to unit sphere
        let norm = data.iter().map(|v| v * v).sum::<f64>().sqrt();
        vectors.push(data.into_iter().map(|v| (v / norm) as f32).collect());
    }
    vectors
}

fn tracked_dims(dim: usize) -> Vec<usize> {
    let mut dims = vec![dim.min(256), dim.min(512), dim];
    dims.sort_unstable();
    dims.dedup();
    dims
}

fn spindle_variants(dim: usize) -> Vec<BenchVariant> {
    let mut variants = vec![raw_variant(dim)];
    variants.push(BenchVariant {
        name: "scalar_int8_full".to_string(),
        spindle: SpindleConfig {
            mode: SpindleMode::ScalarInt8,
            keep_original: false,
            rescore: false,
            ..SpindleConfig::default()
        },
        stored_dims: dim,
    });

    for stored_dims in tracked_dims(dim) {
        variants.push(binary_variant(stored_dims, false));
        variants.push(turbo_variant(stored_dims, false));
    }

    variants
}

fn spindle_search_variants(dim: usize) -> Vec<BenchVariant> {
    let mut variants = vec![
        raw_variant(dim),
        scalar_variant(dim, false),
        scalar_variant(dim, true),
    ];
    for stored_dims in tracked_dims(dim) {
        variants.push(binary_variant(stored_dims, false));
        variants.push(binary_variant(stored_dims, true));
        variants.push(turbo_variant(stored_dims, false));
        variants.push(turbo_variant(stored_dims, true));
        variants.push(turbo_prod_variant(stored_dims, false));
        variants.push(turbo_prod_variant(stored_dims, true));
    }
    variants
}

fn average_encoded_size(vectors: &[Vec<f32>], spindle: &SpindleConfig) -> usize {
    let total: usize = vectors
        .iter()
        .map(|vector| encode_vector(vector, spindle).unwrap().len())
        .sum();
    total / vectors.len().max(1)
}

fn average_logical_retained_size(vectors: &[Vec<f32>], spindle: &SpindleConfig) -> usize {
    let encoded = average_encoded_size(vectors, spindle);
    let original = if spindle.keep_original {
        vectors
            .first()
            .map(|vector| vector.len() * size_of::<f32>())
            .unwrap_or(0)
    } else {
        0
    };
    encoded + original
}

fn cosine_similarity(lhs: &[f32], rhs: &[f32]) -> f64 {
    let mut dot: f64 = 0.0;
    let mut lhs_norm: f64 = 0.0;
    let mut rhs_norm: f64 = 0.0;

    for idx in 0..lhs.len() {
        let l = lhs[idx] as f64;
        let r = rhs[idx] as f64;
        dot += l * r;
        lhs_norm += l * l;
        rhs_norm += r * r;
    }

    if lhs_norm == 0.0 || rhs_norm == 0.0 {
        0.0
    } else {
        dot / (lhs_norm.sqrt() * rhs_norm.sqrt())
    }
}

fn top_k_ids(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f64)> = vectors
        .iter()
        .enumerate()
        .map(|(idx, vector)| (idx, cosine_similarity(vector, query)))
        .collect();
    scored.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap());
    scored.into_iter().take(k).map(|(idx, _)| idx).collect()
}

fn average_overlap_at_k(
    baseline: &[Vec<usize>],
    candidate_vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
) -> f64 {
    let mut total = 0.0;
    for (baseline_ids, query) in baseline.iter().zip(queries.iter()) {
        let candidate_ids = top_k_ids(candidate_vectors, query, k);
        let overlap = baseline_ids
            .iter()
            .filter(|id| candidate_ids.contains(id))
            .count();
        total += overlap as f64 / k as f64;
    }
    total / baseline.len().max(1) as f64
}

fn average_id_overlap_at_k(baseline: &[Vec<usize>], candidate: &[Vec<usize>], k: usize) -> f64 {
    let mut total = 0.0;
    for (baseline_ids, candidate_ids) in baseline.iter().zip(candidate.iter()) {
        let overlap = baseline_ids
            .iter()
            .filter(|id| candidate_ids.contains(id))
            .count();
        total += overlap as f64 / k as f64;
    }
    total / baseline.len().max(1) as f64
}

fn decode_dataset(vectors: &[Vec<f32>], spindle: &SpindleConfig) -> Vec<Vec<f32>> {
    vectors
        .iter()
        .map(|vector| decode_vector(&encode_vector(vector, spindle).unwrap()).unwrap())
        .collect()
}

fn search_top_k_ids(
    env: &Env,
    core: &VectorCore,
    queries: &[Vec<f32>],
    k: usize,
) -> Vec<Vec<usize>> {
    let backend = bench_backend(env);
    let r = backend.begin_read().unwrap();
    let no_filter: Option<&[VectorFilter]> = None;
    queries
        .iter()
        .map(|query| {
            core.search::<VectorFilter>(&r, query, k, no_filter, false)
                .unwrap()
                .into_iter()
                .map(|result| result.get_id() as usize - 1)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn setup_spindle_core(env: &Env, name: &str, spindle: SpindleConfig) -> VectorCore {
    let mut txn = env.write_txn().unwrap();
    let core = VectorCore::new_named(
        env,
        &mut txn,
        name,
        HNSWConfig::new(Some(16), Some(128), Some(256)),
        DistanceMetric::Cosine,
        spindle,
        bench_backend(env),
    )
    .unwrap();
    txn.commit().unwrap();
    core
}

fn bench_vector_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_insertion");
    group.measurement_time(Duration::from_secs(10));

    for &dim in &[128usize, 768, 1536] {
        group.bench_with_input(BenchmarkId::new("insert_200", dim), &dim, |b, &dim| {
            b.iter_batched(
                || {
                    let (env, temp_dir) = setup_temp_env();
                    let mut txn = env.write_txn().unwrap();
                    let core = VectorCore::new(
                        &env,
                        &mut txn,
                        HNSWConfig::new(Some(16), Some(128), Some(128)),
                        bench_backend(&env),
                    )
                    .unwrap();
                    txn.commit().unwrap();
                    let vectors = generate_random_vectors(200, dim, 42);
                    (env, temp_dir, core, vectors)
                },
                |(env, _temp_dir, core, vectors)| {
                    let mut txn = env.write_txn().unwrap();
                    for vector in &vectors {
                        core.insert::<VectorFilter>(&mut txn, vector, None, None)
                            .unwrap();
                    }
                    txn.commit().unwrap();
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_lsm_vector_flat(c: &mut Criterion) {
    let points = env_usize("HELIX_LSM_BENCH_POINTS", 1_000);
    let dim = env_usize("HELIX_LSM_BENCH_DIM", 128);
    let query_count = env_usize("HELIX_LSM_BENCH_QUERIES", 20);
    let mut group = c.benchmark_group("lsm_vector_flat");
    group.measurement_time(Duration::from_secs(10));
    group.throughput(Throughput::Elements(points as u64));

    group.bench_function(BenchmarkId::new("insert", format!("{points}x{dim}")), |b| {
        b.iter_batched(
            || {
                let vectors = generate_random_vectors(points, dim, 42);
                let (env, temp_dir, backend, core) =
                    setup_lsm_vector_core(&format!("lsm_flat_insert_{points}_{dim}"));
                (env, temp_dir, backend, core, vectors)
            },
            |(_env, _temp_dir, backend, core, vectors)| {
                let mut w = backend.begin_write().unwrap();
                for (i, vector) in vectors.iter().enumerate() {
                    core.insert_write_context::<VectorFilter>(
                        &mut w,
                        vector,
                        Some((i + 1) as u128),
                        None,
                    )
                    .unwrap();
                }
                backend.commit(w).unwrap();
            },
            BatchSize::LargeInput,
        );
    });

    let search_state: OnceCell<(Env, TempDir, Arc<AnyBackend>, VectorCore, Vec<Vec<f32>>)> =
        OnceCell::new();
    group.bench_function(
        BenchmarkId::new("flat_search", format!("{points}x{dim}")),
        |b| {
            let (_env, _temp_dir, backend, core, queries) =
                search_state.get_or_init(|| setup_lsm_flat_search_core(points, dim, query_count));
            b.iter(|| {
                let r = backend.begin_read().unwrap();
                let no_filter: Option<&[VectorFilter]> = None;
                for query in queries {
                    let results = core
                        .search::<VectorFilter>(&r, query, 10, no_filter, false)
                        .unwrap();
                    black_box(results.len());
                }
            });
        },
    );

    group.finish();
}

fn bench_spindle_codec(c: &mut Criterion) {
    let mut group = c.benchmark_group("spindle_codec");
    for &dim in &[768usize] {
        let raw_bytes = (dim * size_of::<f32>()) as u64;
        group.throughput(Throughput::Bytes(raw_bytes));

        for variant in spindle_variants(dim) {
            let bench_name = variant.name.clone();
            group.bench_function(BenchmarkId::new("encode_decode", &bench_name), move |b| {
                let vector = generate_random_vectors(1, dim, 123).pop().unwrap();
                let encoded = encode_vector(&vector, &variant.spindle).unwrap();
                eprintln!(
                    "spindle_codec/{}: dims={} raw_bytes={} encoded_bytes={} compression={:.2}x",
                    variant.name,
                    variant.stored_dims,
                    raw_bytes,
                    encoded.len(),
                    raw_bytes as f64 / encoded.len() as f64
                );
                let spindle = variant.spindle.clone();
                b.iter(|| {
                    let bytes = encode_vector(black_box(&vector), black_box(&spindle)).unwrap();
                    let decoded = decode_vector(black_box(&bytes)).unwrap();
                    black_box(decoded.len());
                });
            });
        }
    }

    group.finish();
}

fn bench_vector_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search");
    group.measurement_time(Duration::from_secs(10));

    for &dim in &[128usize, 768] {
        group.bench_function(BenchmarkId::new("search_1000", dim), |b| {
            let (env, _temp_dir, core, queries) = setup_incremental_search_core(1_000, dim, 7, 256);
            let backend = bench_backend(&env);
            b.iter(|| {
                let r = backend.begin_read().unwrap();
                let no_filter: Option<&[VectorFilter]> = None;
                for query in &queries {
                    let results = core
                        .search::<VectorFilter>(&r, query, 10, no_filter, false)
                        .unwrap();
                    black_box(results.len());
                }
            });
        });
    }

    group.finish();
}

fn bench_bulk_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_bulk_build");
    group.measurement_time(Duration::from_secs(10));

    for &count in &[2_000usize, 10_000usize] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("build_768", count), &count, |b, &count| {
            b.iter_batched(
                || {
                    let (env, temp_dir) = setup_temp_env();
                    let mut txn = env.write_txn().unwrap();
                    let core = VectorCore::new(
                        &env,
                        &mut txn,
                        HNSWConfig::new(Some(16), Some(128), Some(128)),
                        bench_backend(&env),
                    )
                    .unwrap();
                    let vectors = generate_random_vectors(count, 768, 42);
                    for (i, vector) in vectors.iter().enumerate() {
                        core.insert_flat(&mut txn, vector, Some((i + 1) as u128), None)
                            .unwrap();
                    }
                    txn.commit().unwrap();
                    (env, temp_dir, core)
                },
                |(env, _temp_dir, core)| {
                    let mut txn = env.write_txn().unwrap();
                    core.build_index_from_flat(&mut txn).unwrap();
                    txn.commit().unwrap();
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

fn bench_vector_search_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search_large");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("indexed_search_10000_768", |b| {
        let (env, _temp_dir, core, queries) =
            setup_indexed_search_core(10_000, 768, 17, false, "legacy_large");
        let backend = bench_backend(&env);
        b.iter(|| {
            let r = backend.begin_read().unwrap();
            let no_filter: Option<&[VectorFilter]> = None;
            for query in &queries {
                let results = core
                    .search::<VectorFilter>(&r, query, 10, no_filter, false)
                    .unwrap();
                black_box(results.len());
            }
        });
    });

    group.finish();
}

fn bench_vector_search_mmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search_mmap");
    group.measurement_time(Duration::from_secs(10));
    let legacy_state: OnceCell<(Env, TempDir, VectorCore, Vec<Vec<f32>>)> = OnceCell::new();
    let mmap_state: OnceCell<(Env, TempDir, VectorCore, Vec<Vec<f32>>)> = OnceCell::new();

    group.bench_function("legacy_unfiltered_10000_768", |b| {
        let (env, _temp_dir, core, queries) = legacy_state
            .get_or_init(|| setup_indexed_search_core(10_000, 768, 17, false, "legacy_unfiltered"));
        let backend = bench_backend(env);
        b.iter(|| {
            let r = backend.begin_read().unwrap();
            let no_filter: Option<&[VectorFilter]> = None;
            for query in queries {
                let results = core
                    .search::<VectorFilter>(&r, query, 10, no_filter, false)
                    .unwrap();
                black_box(results.len());
            }
        });
    });

    group.bench_function("mmap_unfiltered_10000_768", |b| {
        let (env, _temp_dir, core, queries) =
            mmap_state.get_or_init(|| setup_indexed_search_core(10_000, 768, 17, true, "mmap"));
        let backend = bench_backend(env);
        b.iter(|| {
            let r = backend.begin_read().unwrap();
            let no_filter: Option<&[VectorFilter]> = None;
            for query in queries {
                let results = core
                    .search::<VectorFilter>(&r, query, 10, no_filter, false)
                    .unwrap();
                black_box(results.len());
            }
        });
    });

    group.bench_function("mmap_hvector_hash_filter_even_10000_768_limit40", |b| {
        let (env, _temp_dir, core, queries) =
            mmap_state.get_or_init(|| setup_indexed_search_core(10_000, 768, 17, true, "mmap"));
        let candidates = even_candidate_set(10_000);
        let filter = |vector: &HVector| candidates.contains(&vector.get_id());
        let backend = bench_backend(env);
        b.iter(|| {
            let r = backend.begin_read().unwrap();
            for query in queries {
                let results = core
                    .search(&r, query, 40, Some(std::slice::from_ref(&filter)), true)
                    .unwrap();
                black_box(results.len());
            }
        });
    });

    group.bench_function("mmap_id_hash_filter_even_10000_768_limit40", |b| {
        let (env, _temp_dir, core, queries) =
            mmap_state.get_or_init(|| setup_indexed_search_core(10_000, 768, 17, true, "mmap"));
        let candidates = even_candidate_set(10_000);
        let filter = |id: u128| candidates.contains(&id);
        let backend = bench_backend(env);
        b.iter(|| {
            let r = backend.begin_read().unwrap();
            for query in queries {
                let results = core
                    .search_with_id_filter_ef_observed(
                        &r,
                        query,
                        40,
                        Some(std::slice::from_ref(&filter)),
                        true,
                        Some(0.5),
                        None,
                        None,
                    )
                    .unwrap();
                black_box(results.len());
            }
        });
    });

    group.finish();
}

fn bench_spindle_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("spindle_search");
    group.measurement_time(Duration::from_secs(10));

    let k = 10usize;
    for &dim in &[768usize] {
        for variant in spindle_search_variants(dim) {
            let bench_name = variant.name.clone();
            group.bench_function(BenchmarkId::new("search_2000x20", &bench_name), move |b| {
                let base_vectors = generate_random_vectors(2_000, dim, 7);
                let queries = generate_random_vectors(20, dim, 99);
                let baseline: Vec<Vec<usize>> = queries
                    .iter()
                    .map(|query| top_k_ids(&base_vectors, query, k))
                    .collect();
                let raw_bytes = dim * size_of::<f32>();
                let (env, _temp_dir) = setup_temp_env();
                let core = setup_spindle_core(
                    &env,
                    &format!("bench_{}", variant.name),
                    variant.spindle.clone(),
                );
                let avg_encoded_bytes =
                    average_encoded_size(&base_vectors[..128], &variant.spindle);
                let avg_logical_bytes =
                    average_logical_retained_size(&base_vectors[..128], &variant.spindle);
                let decoded_vectors = decode_dataset(&base_vectors, &variant.spindle);
                let projected_queries = queries
                    .iter()
                    .map(|query| project_for_search(query, &variant.spindle).unwrap())
                    .collect::<Vec<_>>();
                let codec_overlap_at_k =
                    average_overlap_at_k(&baseline, &decoded_vectors, &projected_queries, k);

                eprintln!(
                    "spindle_search/{}: dims={} raw_bytes={} avg_encoded_bytes={} logical_bytes={} codec_compression={:.2}x logical_compression={:.2}x codec_overlap@{}={:.3}",
                    variant.name,
                    variant.stored_dims,
                    raw_bytes,
                    avg_encoded_bytes,
                    avg_logical_bytes,
                    raw_bytes as f64 / avg_encoded_bytes as f64,
                    raw_bytes as f64 / avg_logical_bytes as f64,
                    k,
                    codec_overlap_at_k,
                );

                let mut txn = env.write_txn().unwrap();
                for (i, vector) in base_vectors.iter().enumerate() {
                    core.insert::<VectorFilter>(&mut txn, vector, Some((i + 1) as u128), None)
                        .unwrap();
                }
                txn.commit().unwrap();

                let search_ids = search_top_k_ids(&env, &core, &queries, k);
                let search_overlap_at_k = average_id_overlap_at_k(&baseline, &search_ids, k);
                eprintln!(
                    "spindle_search/{}: search_overlap@{}={:.3}",
                    variant.name, k, search_overlap_at_k,
                );

                let backend = bench_backend(&env);
                b.iter(|| {
                    let r = backend.begin_read().unwrap();
                    let no_filter: Option<&[VectorFilter]> = None;
                    for query in &queries {
                        let results = core
                            .search::<VectorFilter>(&r, query, k, no_filter, false)
                            .unwrap();
                        black_box(results.len());
                    }
                });
            });
        }
    }

    group.finish();
}

fn bench_spindle_search_clustered(c: &mut Criterion) {
    let mut group = c.benchmark_group("spindle_search_clustered");
    group.measurement_time(Duration::from_secs(10));

    let k = 10usize;
    for &dim in &[768usize] {
        for variant in spindle_search_variants(dim) {
            let bench_name = variant.name.clone();
            group.bench_function(BenchmarkId::new("search_2000x20", &bench_name), move |b| {
                // 40 clusters, spread=0.35: intra-cluster cosine is high and
                // cross-cluster pairs are low, close to real embedding shape.
                let base_vectors = generate_clustered_vectors(2_000, dim, 40, 0.35, 7);
                let queries = generate_clustered_vectors(20, dim, 40, 0.35, 99);
                let baseline: Vec<Vec<usize>> = queries
                    .iter()
                    .map(|query| top_k_ids(&base_vectors, query, k))
                    .collect();
                let raw_bytes = dim * size_of::<f32>();
                let (env, _temp_dir) = setup_temp_env();
                let core = setup_spindle_core(
                    &env,
                    &format!("bench_{}", variant.name),
                    variant.spindle.clone(),
                );
                let avg_encoded_bytes =
                    average_encoded_size(&base_vectors[..128], &variant.spindle);
                let avg_logical_bytes =
                    average_logical_retained_size(&base_vectors[..128], &variant.spindle);
                let decoded_vectors = decode_dataset(&base_vectors, &variant.spindle);
                let projected_queries = queries
                    .iter()
                    .map(|query| project_for_search(query, &variant.spindle).unwrap())
                    .collect::<Vec<_>>();
                let codec_overlap_at_k =
                    average_overlap_at_k(&baseline, &decoded_vectors, &projected_queries, k);

                eprintln!(
                    "spindle_search_clustered/{}: dims={} encoded_bytes={} logical_bytes={} codec_compression={:.2}x logical_compression={:.2}x codec_overlap@{}={:.3}",
                    variant.name,
                    variant.stored_dims,
                    avg_encoded_bytes,
                    avg_logical_bytes,
                    raw_bytes as f64 / avg_encoded_bytes as f64,
                    raw_bytes as f64 / avg_logical_bytes as f64,
                    k,
                    codec_overlap_at_k,
                );

                let mut txn = env.write_txn().unwrap();
                for (i, vector) in base_vectors.iter().enumerate() {
                    core.insert::<VectorFilter>(&mut txn, vector, Some((i + 1) as u128), None)
                        .unwrap();
                }
                txn.commit().unwrap();

                let search_ids = search_top_k_ids(&env, &core, &queries, k);
                let search_overlap_at_k = average_id_overlap_at_k(&baseline, &search_ids, k);
                eprintln!(
                    "spindle_search_clustered/{}: search_overlap@{}={:.3}",
                    variant.name, k, search_overlap_at_k,
                );

                let backend = bench_backend(&env);
                b.iter(|| {
                    let r = backend.begin_read().unwrap();
                    let no_filter: Option<&[VectorFilter]> = None;
                    for query in &queries {
                        let results = core
                            .search::<VectorFilter>(&r, query, k, no_filter, false)
                            .unwrap();
                        black_box(results.len());
                    }
                });
            });
        }
    }

    group.finish();
}

fn bench_spindle_search_lowrank(c: &mut Criterion) {
    let mut group = c.benchmark_group("spindle_search_lowrank");
    group.measurement_time(Duration::from_secs(10));

    let k = 10usize;
    for &dim in &[768usize] {
        // Only test the key variants: raw, int8, turbo_int4 full-dim, turbo_prod full-dim
        let variants = vec![
            raw_variant(dim),
            scalar_variant(dim, false),
            scalar_variant(dim, true),
            turbo_variant(dim, false),
            turbo_variant(dim, true),
            turbo_prod_variant(dim, false),
            turbo_prod_variant(dim, true),
        ];

        for variant in variants {
            let bench_name = variant.name.clone();
            group.bench_function(BenchmarkId::new("search_2000x20", &bench_name), move |b| {
                // rank=50 in 768d mimics real embedding structure.
                let base_vectors = generate_lowrank_vectors(2_000, dim, 50, 7);
                let queries = generate_lowrank_vectors(20, dim, 50, 99);
                let baseline: Vec<Vec<usize>> = queries
                    .iter()
                    .map(|query| top_k_ids(&base_vectors, query, k))
                    .collect();
                let raw_bytes = dim * size_of::<f32>();
                let (env, _temp_dir) = setup_temp_env();
                let core = setup_spindle_core(
                    &env,
                    &format!("bench_{}", variant.name),
                    variant.spindle.clone(),
                );
                let avg_encoded_bytes =
                    average_encoded_size(&base_vectors[..128], &variant.spindle);
                let avg_logical_bytes =
                    average_logical_retained_size(&base_vectors[..128], &variant.spindle);
                let decoded_vectors = decode_dataset(&base_vectors, &variant.spindle);
                let projected_queries = queries
                    .iter()
                    .map(|query| project_for_search(query, &variant.spindle).unwrap())
                    .collect::<Vec<_>>();
                let codec_overlap_at_k =
                    average_overlap_at_k(&baseline, &decoded_vectors, &projected_queries, k);

                eprintln!(
                    "spindle_search_lowrank/{}: dims={} encoded={} logical={} compression={:.2}x codec_overlap@{}={:.3}",
                    variant.name, variant.stored_dims, avg_encoded_bytes, avg_logical_bytes,
                    raw_bytes as f64 / avg_encoded_bytes as f64, k, codec_overlap_at_k,
                );

                let mut txn = env.write_txn().unwrap();
                for (i, vector) in base_vectors.iter().enumerate() {
                    core.insert::<VectorFilter>(&mut txn, vector, Some((i + 1) as u128), None)
                        .unwrap();
                }
                txn.commit().unwrap();

                let search_ids = search_top_k_ids(&env, &core, &queries, k);
                let search_overlap_at_k = average_id_overlap_at_k(&baseline, &search_ids, k);
                eprintln!(
                    "spindle_search_lowrank/{}: search_overlap@{}={:.3}",
                    variant.name, k, search_overlap_at_k,
                );

                let backend = bench_backend(&env);
                b.iter(|| {
                    let r = backend.begin_read().unwrap();
                    let no_filter: Option<&[VectorFilter]> = None;
                    for query in &queries {
                        let results = core
                            .search::<VectorFilter>(&r, query, k, no_filter, false)
                            .unwrap();
                        black_box(results.len());
                    }
                });
            });
        }
    }

    group.finish();
}

/// Sweep oversampling for turbo_prod vs int8 on low-rank (realistic) data.
/// Proves that turbo_prod + higher oversampling recovers int8-quality recall.
fn bench_oversampling_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("oversampling_sweep");
    group.measurement_time(Duration::from_secs(10));

    let dim = 768usize;
    let k = 10usize;

    for oversampling in [1usize, 2, 4, 8, 16] {
        let variants: Vec<(&str, SpindleConfig)> = vec![
            (
                "scalar_int8",
                SpindleConfig {
                    mode: SpindleMode::ScalarInt8,
                    keep_original: true,
                    rescore: true,
                    oversampling,
                    ..SpindleConfig::default()
                },
            ),
            (
                "turbo_prod",
                SpindleConfig {
                    mode: SpindleMode::TurboProd,
                    keep_original: true,
                    rescore: true,
                    oversampling,
                    turbo_dims: dim,
                    ..SpindleConfig::default()
                },
            ),
        ];

        for (label, spindle) in variants {
            let name = format!("{label}_os{oversampling}");
            group.bench_function(BenchmarkId::new("search", &name), move |b| {
                let base_vectors = generate_lowrank_vectors(2_000, dim, 50, 7);
                let queries = generate_lowrank_vectors(20, dim, 50, 99);
                let baseline: Vec<Vec<usize>> = queries
                    .iter()
                    .map(|query| top_k_ids(&base_vectors, query, k))
                    .collect();
                let (env, _temp_dir) = setup_temp_env();
                let core = setup_spindle_core(&env, &format!("bench_{name}"), spindle.clone());

                let mut txn = env.write_txn().unwrap();
                for (i, vector) in base_vectors.iter().enumerate() {
                    core.insert::<VectorFilter>(&mut txn, vector, Some((i + 1) as u128), None)
                        .unwrap();
                }
                txn.commit().unwrap();

                let search_ids = search_top_k_ids(&env, &core, &queries, k);
                let overlap = average_id_overlap_at_k(&baseline, &search_ids, k);
                eprintln!("oversampling_sweep/{name}: overlap@{k}={overlap:.3}");

                let backend = bench_backend(&env);
                b.iter(|| {
                    let r = backend.begin_read().unwrap();
                    let no_filter: Option<&[VectorFilter]> = None;
                    for query in &queries {
                        let results = core
                            .search::<VectorFilter>(&r, query, k, no_filter, false)
                            .unwrap();
                        black_box(results.len());
                    }
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_vector_insertion,
    bench_lsm_vector_flat,
    bench_bulk_build,
    bench_vector_search_large,
    bench_vector_search_mmap,
    bench_spindle_codec,
    bench_vector_search,
    bench_spindle_search,
    bench_spindle_search_clustered,
    bench_spindle_search_lowrank,
    bench_oversampling_sweep
);
criterion_main!(benches);
