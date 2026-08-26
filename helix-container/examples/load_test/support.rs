//! Support module for examples/load_test.rs. Kept separate so the driver
//! stays under 150 lines.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use reqwest::Client;
use serde_json::{json, Value};
use tokio::sync::Mutex;

#[derive(Default)]
pub struct OpStats {
    pub count: AtomicU64,
    pub errors: AtomicU64,
    pub lat: Mutex<Vec<u64>>,
}

impl OpStats {
    pub async fn record(&self, ok: bool, elapsed: Duration) {
        self.count.fetch_add(1, Ordering::Relaxed);
        if !ok {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
        let mut v = self.lat.lock().await;
        v.push(elapsed.as_micros() as u64);
    }
    pub async fn summary(&self, name: &str, wall: f64) {
        let mut v = self.lat.lock().await;
        v.sort_unstable();
        let n = v.len();
        if n == 0 {
            println!("  {:<12} (no samples)", name);
            return;
        }
        let p = |q: f64| v[((n as f64 - 1.0) * q) as usize] as f64 / 1000.0;
        let err = self.errors.load(Ordering::Relaxed);
        println!(
            "  {:<12} n={:>7}  err={:>4}  qps={:>8.1}  p50={:>6.2}ms  p95={:>6.2}ms  p99={:>6.2}ms",
            name,
            n,
            err,
            n as f64 / wall,
            p(0.5),
            p(0.95),
            p(0.99)
        );
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Workload {
    WritesOnly,
    ReadsOnly,
    Mixed,
    CoalesceStress,
}

impl Workload {
    pub fn from_env(s: &str) -> Self {
        match s {
            "writes_only" => Self::WritesOnly,
            "reads_only" => Self::ReadsOnly,
            "coalesce_stress" => Self::CoalesceStress,
            _ => Self::Mixed,
        }
    }
    /// Return (upsert_weight, search_weight, filt_weight, scroll_weight, count_weight).
    fn weights(&self) -> (u32, u32, u32, u32, u32) {
        match self {
            Self::WritesOnly => (10, 0, 0, 0, 0),
            Self::ReadsOnly => (0, 6, 3, 1, 1),
            Self::Mixed => (4, 3, 2, 1, 1),
            Self::CoalesceStress => (20, 0, 0, 0, 0),
        }
    }
}

/// Deterministic f32 vector from a seed. Normalized to unit length for cosine.
pub fn rand_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut out = Vec::with_capacity(dim);
    let mut norm = 0.0f64;
    for _ in 0..dim {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = ((s >> 33) as f32) / (1u64 << 31) as f32;
        let v = u * 2.0 - 1.0;
        out.push(v);
        norm += (v as f64) * (v as f64);
    }
    let inv = (1.0 / norm.sqrt()) as f32;
    for x in &mut out {
        *x *= inv;
    }
    out
}

pub fn hex_id(id: u128) -> String {
    format!("{:032x}", id)
}

#[allow(clippy::too_many_arguments)]
pub async fn worker_loop(
    worker: usize,
    client: Client,
    url: String,
    coll: String,
    dim: usize,
    batch: usize,
    wl: Workload,
    stop: Arc<AtomicBool>,
    next_id: Arc<AtomicU64>,
    up: Arc<OpStats>,
    se: Arc<OpStats>,
    fs: Arc<OpStats>,
    sc: Arc<OpStats>,
    cn: Arc<OpStats>,
) {
    let (wu, ws, wf, wsc, wc) = wl.weights();
    let total: u32 = wu + ws + wf + wsc + wc;
    let mut rng: u64 = 0xCAFEBABE ^ (worker as u64).wrapping_mul(0x9E3779B97F4A7C15);
    while !stop.load(Ordering::Relaxed) {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let pick = ((rng >> 33) as u32) % total.max(1);
        let t0 = Instant::now();
        let (ok, which) = if pick < wu {
            (
                do_upsert(&client, &url, &coll, dim, batch, &next_id, worker).await,
                &up,
            )
        } else if pick < wu + ws {
            (do_search(&client, &url, &coll, dim, rng).await, &se)
        } else if pick < wu + ws + wf {
            (
                do_filtered_search(&client, &url, &coll, dim, rng).await,
                &fs,
            )
        } else if pick < wu + ws + wf + wsc {
            (do_scroll(&client, &url, &coll).await, &sc)
        } else {
            (do_count(&client, &url, &coll).await, &cn)
        };
        which.record(ok, t0.elapsed()).await;
    }
}

async fn do_upsert(
    c: &Client,
    url: &str,
    coll: &str,
    dim: usize,
    batch: usize,
    next: &AtomicU64,
    w: usize,
) -> bool {
    let mut pts = Vec::with_capacity(batch);
    for _ in 0..batch {
        let id = next.fetch_add(1, Ordering::Relaxed) as u128 + ((w as u128) << 64);
        pts.push(json!({"id": hex_id(id),
            "vector": {"dense": rand_vec(dim, id as u64)},
            "payload": {"tenant_id": format!("t{}", id % 8), "w": w}}));
    }
    let body = json!({"points": pts});
    c.put(format!("{}/collections/{}/points", url, coll))
        .json(&body)
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

async fn do_search(c: &Client, url: &str, coll: &str, dim: usize, seed: u64) -> bool {
    let body = json!({"vector": rand_vec(dim, seed), "limit": 10, "with_payload": true});
    c.post(format!("{}/collections/{}/points/search", url, coll))
        .json(&body)
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

async fn do_filtered_search(c: &Client, url: &str, coll: &str, dim: usize, seed: u64) -> bool {
    let tenant = format!("t{}", seed % 8);
    let body = json!({"vector": rand_vec(dim, seed), "limit": 10,
        "filter": {"must": [{"key": "tenant_id", "match": {"value": tenant}}]}});
    c.post(format!("{}/collections/{}/points/search", url, coll))
        .json(&body)
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

async fn do_scroll(c: &Client, url: &str, coll: &str) -> bool {
    let body = json!({"limit": 50, "with_payload": true});
    c.post(format!("{}/collections/{}/points/scroll", url, coll))
        .json(&body)
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

async fn do_count(c: &Client, url: &str, coll: &str) -> bool {
    let body: Value = json!({});
    c.post(format!("{}/collections/{}/points/count", url, coll))
        .json(&body)
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}
