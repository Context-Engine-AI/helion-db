use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use helixdb::helix_engine::{
    graph_core::{
        config::Config,
        traversals::{
            bfs::{bfs_forward, bfs_reverse},
            cycles::detect_cycles,
        },
    },
    storage_core::{
        collection_manager::CollectionManager,
        storage_core::HelixGraphStorage,
        upsert::{EdgeUpsert, NodeUpsert},
    },
};
use helixdb::protocol::{deterministic_id, value::Value};
use std::collections::HashMap;
use std::time::Duration;
use tempfile::TempDir;

fn test_config() -> Config {
    Config::new(16, 128, 768, 1)
}

fn make_node(collection: &str, name: &str, path: &str) -> NodeUpsert {
    NodeUpsert {
        id: deterministic_id::node_id(collection, "Symbol", name, path),
        label: "Symbol".to_string(),
        properties: HashMap::from([
            ("name".to_string(), Value::String(name.to_string())),
            ("path".to_string(), Value::String(path.to_string())),
        ]),
    }
}

fn make_edge(
    collection: &str,
    label: &str,
    from_name: &str,
    to_name: &str,
    from_path: &str,
    to_path: &str,
) -> EdgeUpsert {
    EdgeUpsert {
        id: deterministic_id::edge_id(collection, label, from_name, to_name, from_path, to_path),
        label: label.to_string(),
        from_node: deterministic_id::node_id(collection, "Symbol", from_name, from_path),
        to_node: deterministic_id::node_id(collection, "Symbol", to_name, to_path),
        properties: HashMap::from([("edge_type".to_string(), Value::String(label.to_string()))]),
    }
}

fn setup_storage() -> (HelixGraphStorage, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let storage = HelixGraphStorage::new(temp_dir.path().to_str().unwrap(), test_config()).unwrap();
    (storage, temp_dir)
}

fn build_chain_graph(storage: &HelixGraphStorage, collection: &str, size: usize) -> (u128, u128) {
    let mut txn = storage.graph_env.write_txn().unwrap();

    for i in 0..size {
        let name = format!("sym_{i}");
        let path = format!("src/file_{i}.rs");
        storage
            .upsert_node(&mut txn, &make_node(collection, &name, &path))
            .unwrap();
    }

    for i in 0..size.saturating_sub(1) {
        let from_name = format!("sym_{i}");
        let to_name = format!("sym_{}", i + 1);
        let from_path = format!("src/file_{i}.rs");
        let to_path = format!("src/file_{}.rs", i + 1);
        storage
            .upsert_edge(
                &mut txn,
                &make_edge(
                    collection, "CALLS", &from_name, &to_name, &from_path, &to_path,
                ),
            )
            .unwrap();
    }

    txn.commit().unwrap();

    let start = deterministic_id::node_id(collection, "Symbol", "sym_0", "src/file_0.rs");
    let end = deterministic_id::node_id(
        collection,
        "Symbol",
        &format!("sym_{}", size.saturating_sub(1)),
        &format!("src/file_{}.rs", size.saturating_sub(1)),
    );

    (start, end)
}

fn build_cycle_graph(storage: &HelixGraphStorage, collection: &str, size: usize) -> u128 {
    let mut txn = storage.graph_env.write_txn().unwrap();

    for i in 0..size {
        let name = format!("cyc_{i}");
        let path = format!("src/cycle_{i}.rs");
        storage
            .upsert_node(&mut txn, &make_node(collection, &name, &path))
            .unwrap();
    }

    for i in 0..size {
        let next = (i + 1) % size;
        let from_name = format!("cyc_{i}");
        let to_name = format!("cyc_{next}");
        let from_path = format!("src/cycle_{i}.rs");
        let to_path = format!("src/cycle_{next}.rs");
        storage
            .upsert_edge(
                &mut txn,
                &make_edge(
                    collection, "CALLS", &from_name, &to_name, &from_path, &to_path,
                ),
            )
            .unwrap();
    }

    txn.commit().unwrap();
    deterministic_id::node_id(collection, "Symbol", "cyc_0", "src/cycle_0.rs")
}

fn bench_collection_manager(c: &mut Criterion) {
    let mut group = c.benchmark_group("phase1_collection_manager");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("create_collection", |b| {
        b.iter_batched(
            || {
                let temp_dir = TempDir::new().unwrap();
                let manager =
                    CollectionManager::new(temp_dir.path().to_path_buf(), test_config()).unwrap();
                (temp_dir, manager)
            },
            |(_temp_dir, manager)| {
                let storage = manager.create_collection("tenant_a").unwrap();
                black_box(storage);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("collection_stats", |b| {
        let temp_dir = TempDir::new().unwrap();
        let manager = CollectionManager::new(temp_dir.path().to_path_buf(), test_config()).unwrap();
        manager.create_collection("tenant_stats").unwrap();

        b.iter(|| {
            let stats = manager.collection_stats("tenant_stats").unwrap();
            black_box(stats);
        });
    });

    group.finish();
}

fn bench_bulk_upserts(c: &mut Criterion) {
    let mut group = c.benchmark_group("phase1_bulk_upserts");
    group.measurement_time(Duration::from_secs(10));

    for &count in &[100usize, 1_000, 5_000] {
        group.bench_with_input(BenchmarkId::new("nodes", count), &count, |b, &count| {
            b.iter_batched(
                || {
                    let (storage, temp_dir) = setup_storage();
                    let nodes: Vec<NodeUpsert> = (0..count)
                        .map(|i| {
                            make_node("bench", &format!("node_{i}"), &format!("src/node_{i}.rs"))
                        })
                        .collect();
                    (storage, temp_dir, nodes)
                },
                |(storage, _temp_dir, nodes)| {
                    let mut txn = storage.graph_env.write_txn().unwrap();
                    let upserted = storage.bulk_upsert_nodes(&mut txn, &nodes).unwrap();
                    txn.commit().unwrap();
                    black_box(upserted);
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.bench_function("edges_5000", |b| {
        b.iter_batched(
            || {
                let (storage, temp_dir) = setup_storage();
                let mut txn = storage.graph_env.write_txn().unwrap();
                let nodes: Vec<NodeUpsert> = (0..1_000)
                    .map(|i| make_node("bench", &format!("node_{i}"), &format!("src/node_{i}.rs")))
                    .collect();
                storage.bulk_upsert_nodes(&mut txn, &nodes).unwrap();
                txn.commit().unwrap();

                let edges: Vec<EdgeUpsert> = (0..5_000)
                    .map(|i| {
                        let from = i % 1_000;
                        let to = (i + 1) % 1_000;
                        make_edge(
                            "bench",
                            "CALLS",
                            &format!("node_{from}"),
                            &format!("node_{to}"),
                            &format!("src/node_{from}.rs"),
                            &format!("src/node_{to}.rs"),
                        )
                    })
                    .collect();
                (storage, temp_dir, edges)
            },
            |(storage, _temp_dir, edges)| {
                let mut txn = storage.graph_env.write_txn().unwrap();
                let upserted = storage.bulk_upsert_edges(&mut txn, &edges).unwrap();
                txn.commit().unwrap();
                black_box(upserted);
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_graph_traversals(c: &mut Criterion) {
    let mut group = c.benchmark_group("phase1_graph_traversals");
    group.measurement_time(Duration::from_secs(10));

    let (storage, _temp_dir) = setup_storage();
    let (start, end) = build_chain_graph(&storage, "bench", 2_000);

    group.bench_function("bfs_forward_depth_8", |b| {
        b.iter(|| {
            let txn = storage.graph_env.read_txn().unwrap();
            let results =
                bfs_forward(&storage, &txn, &[start], &["CALLS"], 8, 10_000, None).unwrap();
            black_box(results.len());
        });
    });

    group.bench_function("bfs_reverse_depth_8", |b| {
        b.iter(|| {
            let txn = storage.graph_env.read_txn().unwrap();
            let results = bfs_reverse(&storage, &txn, &[end], &["CALLS"], 8, 10_000, None).unwrap();
            black_box(results.len());
        });
    });

    let (cycle_storage, _cycle_tmp) = setup_storage();
    let cycle_start = build_cycle_graph(&cycle_storage, "cycle_bench", 256);
    group.bench_function("detect_cycles", |b| {
        b.iter(|| {
            let txn = cycle_storage.graph_env.read_txn().unwrap();
            let cycles = detect_cycles(&cycle_storage, &txn, cycle_start, &["CALLS"], 8).unwrap();
            black_box(cycles.len());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_collection_manager,
    bench_bulk_upserts,
    bench_graph_traversals
);
criterion_main!(benches);
