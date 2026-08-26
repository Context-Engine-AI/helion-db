//! Mixed-workload load test for a running helix-container.
//!
//! Usage:
//!   # terminal 1: start server
//!   cargo run -p helix-container --release
//!
//!   # terminal 2: drive load
//!   HELIX_URL=http://127.0.0.1:6969 \
//!   HELIX_DURATION_SECS=30 HELIX_CONCURRENCY=32 \
//!   HELIX_WORKLOAD=mixed \
//!   cargo run -p helix-container --example load_test --release
//!
//! Env overrides:
//!   HELIX_URL            (default http://127.0.0.1:6969)
//!   HELIX_COLLECTION     (default loadtest_<pid>)
//!   HELIX_DIM            (default 768)
//!   HELIX_DURATION_SECS  (default 30)
//!   HELIX_CONCURRENCY    (default 32)
//!   HELIX_UPSERT_BATCH   (default 16)
//!   HELIX_WORKLOAD       (writes_only | reads_only | mixed | coalesce_stress)
//!
//! Reports p50/p95/p99 latency and QPS per op class (upsert, search, filtered
//! search, scroll, count). Run once with the default coalescing settings,
//! once with `HELIX_WRITE_QUEUE_COALESCE=0` to measure the delta.
//!
//! Kept deliberately simple: one file, only `reqwest` (dev-dep) + `tokio`
//! (runtime) + `serde_json` — no new crates, no new workspace members.

use std::env;
use std::process;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use reqwest::Client;
use serde_json::{json, Value};
use tokio::sync::Mutex;

mod support;
use support::{hex_id, rand_vec, OpStats, Workload};

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let url = env::var("HELIX_URL").unwrap_or_else(|_| "http://127.0.0.1:6969".into());
    let coll =
        env::var("HELIX_COLLECTION").unwrap_or_else(|_| format!("loadtest_{}", process::id()));
    let dim: usize = env::var("HELIX_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(768);
    let secs: u64 = env::var("HELIX_DURATION_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let conc: usize = env::var("HELIX_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let batch: usize = env::var("HELIX_UPSERT_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let wl = Workload::from_env(&env::var("HELIX_WORKLOAD").unwrap_or_else(|_| "mixed".into()));

    println!("helix load test");
    println!(
        "  url={}  collection={}  dim={}  conc={}  duration={}s  batch={}  workload={:?}",
        url, coll, dim, conc, secs, batch, wl
    );

    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .pool_max_idle_per_host(conc * 2)
        .build()
        .expect("reqwest client");

    // Create collection (idempotent: 4xx if it exists, we just proceed).
    let create = json!({"vectors": {"dense": {"size": dim, "distance": "Cosine"}}});
    let resp = client
        .put(format!("{}/collections/{}", url, coll))
        .json(&create)
        .send()
        .await
        .expect("create collection");
    if !resp.status().is_success() && resp.status().as_u16() != 409 {
        eprintln!("create collection failed: {}", resp.status());
    }
    // Keyword payload index for tenant_id so filtered search has an index.
    let idx = json!({"field_name": "tenant_id", "field_schema": "keyword"});
    let _ = client
        .put(format!("{}/collections/{}/index", url, coll))
        .json(&idx)
        .send()
        .await;

    // Seed so search/scroll have something to hit.
    let seed_points: Vec<Value> = (0..256)
        .map(|i| {
            json!({
                "id": hex_id(i as u128),
                "vector": {"dense": rand_vec(dim, i as u64)},
                "payload": {"tenant_id": format!("t{}", i % 8), "seq": i},
            })
        })
        .collect();
    let _ = client
        .put(format!("{}/collections/{}/points", url, coll))
        .json(&json!({"points": seed_points}))
        .send()
        .await;

    let stop = Arc::new(AtomicBool::new(false));
    let next_id = Arc::new(AtomicU64::new(1_000));
    let up = Arc::new(OpStats::default());
    let se = Arc::new(OpStats::default());
    let fs = Arc::new(OpStats::default());
    let sc = Arc::new(OpStats::default());
    let cn = Arc::new(OpStats::default());
    let t0 = Instant::now();

    let mut tasks = Vec::with_capacity(conc);
    for w in 0..conc {
        let (client, url, coll) = (client.clone(), url.clone(), coll.clone());
        let (stop, next_id) = (stop.clone(), next_id.clone());
        let (up, se, fs, sc, cn) = (up.clone(), se.clone(), fs.clone(), sc.clone(), cn.clone());
        tasks.push(tokio::spawn(async move {
            support::worker_loop(
                w, client, url, coll, dim, batch, wl, stop, next_id, up, se, fs, sc, cn,
            )
            .await;
        }));
    }

    tokio::time::sleep(Duration::from_secs(secs)).await;
    stop.store(true, Ordering::SeqCst);
    for t in tasks {
        let _ = t.await;
    }
    let wall = t0.elapsed().as_secs_f64();

    println!("\nresults after {:.1}s:", wall);
    up.summary("upsert", wall).await;
    se.summary("search", wall).await;
    fs.summary("search+filt", wall).await;
    sc.summary("scroll", wall).await;
    cn.summary("count", wall).await;

    // Self-cleanup: drop the benchmark collection so a load/ingest run never
    // leaves disk full (LMDB files / SlateDB objects + SSD cache). Opt out with
    // HELIX_KEEP_COLLECTION=1 to keep it for inspection.
    if env::var("HELIX_KEEP_COLLECTION").ok().as_deref() == Some("1") {
        println!(
            "\ndone. collection `{}` left in place (HELIX_KEEP_COLLECTION=1).",
            coll
        );
    } else {
        match client
            .delete(format!("{}/collections/{}", url, coll))
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => {
                println!("\ndone. collection `{}` deleted (cleanup).", coll)
            }
            Ok(r) => println!(
                "\ndone. collection `{}` cleanup returned HTTP {} — verify disk.",
                coll,
                r.status()
            ),
            Err(e) => println!(
                "\ndone. collection `{}` cleanup FAILED ({e}) — verify disk.",
                coll
            ),
        }
    }
}

// Required by the OpStats::summary signature — suppress dead_code lints so
// this example builds clean when `cargo clippy` is run.
#[allow(dead_code)]
fn _phantom(_: &Mutex<Vec<u64>>) {}
