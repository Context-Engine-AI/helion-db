//! IVF posting-list ANN — pure math and blob (de)serialization.
//!
//! Opt-in alternative to HNSW for dense segments, in the style of
//! SPANN (NeurIPS 2021) / classic IVF posting-list ANN: a small in-memory
//! centroid table is probed exhaustively, then only the posting lists of the
//! `nprobe` nearest centroids are read and scored. Unlike HNSW's random graph
//! hops — pathological on S3-resident SlateDB — posting-list reads are a few
//! sequential blob fetches per query.
//!
//! This module has no storage dependencies: k-means and the blob encodings are
//! pure functions. Storage wiring (build/search/reaper) lives in
//! `vector_core.rs`; the env-gated mode switch lives in `named_vectors.rs`.
//!
//! Determinism: initialization is an evenly-strided sample of the input (no
//! RNG), Lloyd assignment is order-independent per point, and centroid
//! recomputation / empty-cluster reseeding iterate points in index order — so
//! identical input always yields identical centroids and assignments.

use crate::helix_engine::types::VectorError;
use rayon::prelude::*;

/// Magic prefix shared by the centroids and meta blobs.
pub const IVF_MAGIC: &[u8; 4] = b"IVF1";
/// Current blob format version.
pub const IVF_FORMAT_VERSION: u8 = 1;

/// Hard cap on the number of centroids for a single segment.
pub const IVF_MAX_K: usize = 4096;

/// Default centroid count for `n` stored vectors: `clamp(isqrt(n), 1, 4096)`.
pub fn default_k(n: usize) -> usize {
    isqrt(n).clamp(1, IVF_MAX_K)
}

/// Default number of probed posting lists for `k` centroids:
/// `clamp(k / 8, 1, 64)`. The floor of 1 keeps tiny segments searchable.
pub fn default_nprobe(k: usize) -> usize {
    (k / 8).clamp(1, 64)
}

/// Integer square root (floor). `usize::isqrt` needs Rust 1.84; keep a local
/// Newton iteration so the crate's MSRV is untouched.
fn isqrt(n: usize) -> usize {
    if n < 2 {
        return n;
    }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

/// Decoded `ivf:meta` blob: enough to plan a search without touching the
/// centroid table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IvfMeta {
    pub k: u32,
    pub dim: u32,
    pub default_nprobe: u32,
}

/// Encode the meta blob: magic, version u8, k u32-LE, dim u32-LE,
/// default_nprobe u32-LE.
pub fn encode_meta(meta: &IvfMeta) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + 1 + 12);
    buf.extend_from_slice(IVF_MAGIC);
    buf.push(IVF_FORMAT_VERSION);
    buf.extend_from_slice(&meta.k.to_le_bytes());
    buf.extend_from_slice(&meta.dim.to_le_bytes());
    buf.extend_from_slice(&meta.default_nprobe.to_le_bytes());
    buf
}

pub fn decode_meta(buf: &[u8]) -> Result<IvfMeta, VectorError> {
    let rest = check_header(buf, "ivf meta")?;
    if rest.len() < 12 {
        return Err(VectorError::VectorCoreError(
            "ivf meta blob truncated".to_string(),
        ));
    }
    let k = u32::from_le_bytes(rest[0..4].try_into().expect("length checked"));
    let dim = u32::from_le_bytes(rest[4..8].try_into().expect("length checked"));
    let default_nprobe = u32::from_le_bytes(rest[8..12].try_into().expect("length checked"));
    Ok(IvfMeta {
        k,
        dim,
        default_nprobe,
    })
}

/// Encode the centroid table: magic, version u8, dim u32-LE, k u32-LE, then
/// `k * dim` f32-LE values in centroid order.
pub fn encode_centroids(dim: usize, centroids: &[Vec<f32>]) -> Result<Vec<u8>, VectorError> {
    let mut buf = Vec::with_capacity(4 + 1 + 8 + centroids.len() * dim * 4);
    buf.extend_from_slice(IVF_MAGIC);
    buf.push(IVF_FORMAT_VERSION);
    buf.extend_from_slice(&(dim as u32).to_le_bytes());
    buf.extend_from_slice(&(centroids.len() as u32).to_le_bytes());
    for centroid in centroids {
        if centroid.len() != dim {
            return Err(VectorError::InvalidVectorLength);
        }
        for value in centroid {
            buf.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(buf)
}

pub fn decode_centroids(buf: &[u8]) -> Result<(usize, Vec<Vec<f32>>), VectorError> {
    let rest = check_header(buf, "ivf centroids")?;
    if rest.len() < 8 {
        return Err(VectorError::VectorCoreError(
            "ivf centroids blob truncated".to_string(),
        ));
    }
    let dim = u32::from_le_bytes(rest[0..4].try_into().expect("length checked")) as usize;
    let k = u32::from_le_bytes(rest[4..8].try_into().expect("length checked")) as usize;
    let payload = &rest[8..];
    if payload.len() != k * dim * 4 {
        return Err(VectorError::VectorCoreError(format!(
            "ivf centroids blob length mismatch: expected {} payload bytes, got {}",
            k * dim * 4,
            payload.len()
        )));
    }
    let mut centroids = Vec::with_capacity(k);
    for c in 0..k {
        let mut centroid = Vec::with_capacity(dim);
        for d in 0..dim {
            let at = (c * dim + d) * 4;
            centroid.push(f32::from_le_bytes(
                payload[at..at + 4].try_into().expect("length checked"),
            ));
        }
        centroids.push(centroid);
    }
    Ok((dim, centroids))
}

/// Encode one posting list: concatenated u128-BE point ids.
pub fn encode_posting(ids: &[u128]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(ids.len() * 16);
    for id in ids {
        buf.extend_from_slice(&id.to_be_bytes());
    }
    buf
}

pub fn decode_posting(buf: &[u8]) -> Result<Vec<u128>, VectorError> {
    if buf.len() % 16 != 0 {
        return Err(VectorError::VectorCoreError(
            "ivf posting blob length is not a multiple of 16".to_string(),
        ));
    }
    Ok(buf
        .chunks_exact(16)
        .map(|chunk| u128::from_be_bytes(chunk.try_into().expect("chunks_exact(16)")))
        .collect())
}

fn check_header<'a>(buf: &'a [u8], what: &str) -> Result<&'a [u8], VectorError> {
    if buf.len() < 5 {
        return Err(VectorError::VectorCoreError(format!(
            "{what} blob truncated"
        )));
    }
    if &buf[0..4] != IVF_MAGIC {
        return Err(VectorError::VectorCoreError(format!(
            "{what} blob bad magic"
        )));
    }
    if buf[4] != IVF_FORMAT_VERSION {
        return Err(VectorError::VectorCoreError(format!(
            "{what} blob unsupported version {}",
            buf[4]
        )));
    }
    Ok(&buf[5..])
}

/// Output of [`kmeans`]: `centroids[c]` is the centroid vector for cluster
/// `c`, `assignments[i]` is the cluster of input vector `i`.
pub struct KmeansResult {
    pub centroids: Vec<Vec<f32>>,
    pub assignments: Vec<u32>,
}

/// Deterministic Lloyd k-means over `vectors` with the caller's distance
/// function (must match the metric the segment scores with).
///
/// - init: evenly-strided sample (every `n/k`-th input vector)
/// - at most [`KMEANS_MAX_ITERS`] Lloyd iterations, or until assignments are
///   stable
/// - assignment is rayon-parallel; centroid recomputation is serial in index
///   order for determinism
/// - an empty cluster is reseeded from the not-yet-reseeded point farthest
///   from its assigned centroid
///
/// `k` is clamped to `[1, n]`. Empty input yields an empty result.
pub fn kmeans<D>(vectors: &[Vec<f32>], k: usize, dist: D) -> KmeansResult
where
    D: Fn(&[f32], &[f32]) -> f32 + Sync,
{
    let n = vectors.len();
    if n == 0 {
        return KmeansResult {
            centroids: Vec::new(),
            assignments: Vec::new(),
        };
    }
    let k = k.clamp(1, n);
    let dim = vectors[0].len();

    // Evenly-strided init: deterministic and spread across insertion order.
    let stride = (n / k).max(1);
    let mut centroids: Vec<Vec<f32>> = (0..k).map(|c| vectors[c * stride].clone()).collect();

    let mut assignments: Vec<u32> = vec![u32::MAX; n];
    for _ in 0..KMEANS_MAX_ITERS {
        // Assignment: nearest centroid, lowest index wins ties (strict `<`).
        let next: Vec<u32> = vectors
            .par_iter()
            .map(|v| nearest_centroid(v, &centroids, &dist))
            .collect();
        let stable = next == assignments;
        assignments = next;
        if stable {
            break;
        }

        // Recompute centroids as per-cluster means, serially in index order.
        let mut sums: Vec<Vec<f64>> = vec![vec![0.0; dim]; k];
        let mut counts: Vec<usize> = vec![0; k];
        for (v, &a) in vectors.iter().zip(assignments.iter()) {
            let c = a as usize;
            counts[c] += 1;
            for (acc, value) in sums[c].iter_mut().zip(v.iter()) {
                *acc += f64::from(*value);
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                centroids[c] = sums[c]
                    .iter()
                    .map(|acc| (*acc / counts[c] as f64) as f32)
                    .collect();
            }
        }

        // Reseed each empty cluster from the not-yet-reseeded point farthest
        // from its assigned centroid.
        let empty: Vec<usize> = (0..k).filter(|&c| counts[c] == 0).collect();
        if !empty.is_empty() {
            let mut reseeded: Vec<usize> = Vec::with_capacity(empty.len());
            for c in empty {
                let mut farthest: Option<(f32, usize)> = None;
                for (i, v) in vectors.iter().enumerate() {
                    if reseeded.contains(&i) {
                        continue;
                    }
                    let d = dist(v, &centroids[assignments[i] as usize]);
                    // Strict `>` keeps the lowest index on ties (deterministic).
                    if farthest.map(|(best, _)| d > best).unwrap_or(true) {
                        farthest = Some((d, i));
                    }
                }
                if let Some((_, i)) = farthest {
                    centroids[c] = vectors[i].clone();
                    reseeded.push(i);
                }
            }
        }
    }

    // Final assignment against the final centroids so callers can build
    // posting lists that match the persisted centroid table exactly.
    let assignments: Vec<u32> = vectors
        .par_iter()
        .map(|v| nearest_centroid(v, &centroids, &dist))
        .collect();

    KmeansResult {
        centroids,
        assignments,
    }
}

/// Lloyd iteration cap; assignment stability usually terminates earlier.
const KMEANS_MAX_ITERS: usize = 10;

fn nearest_centroid<D>(v: &[f32], centroids: &[Vec<f32>], dist: &D) -> u32
where
    D: Fn(&[f32], &[f32]) -> f32 + Sync,
{
    let mut best = 0u32;
    let mut best_dist = f32::INFINITY;
    for (c, centroid) in centroids.iter().enumerate() {
        let d = dist(v, centroid);
        if d < best_dist {
            best_dist = d;
            best = c as u32;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn euclid(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    /// Seeded LCG so test data is deterministic without the rand crate.
    fn lcg_vectors(seed: u64, n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (u32::MAX >> 1) as f32) - 1.0
        };
        (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
    }

    #[test]
    fn kmeans_is_deterministic() {
        let vectors = lcg_vectors(7, 300, 8);
        let a = kmeans(&vectors, 12, euclid);
        let b = kmeans(&vectors, 12, euclid);
        assert_eq!(a.centroids, b.centroids);
        assert_eq!(a.assignments, b.assignments);
        assert_eq!(a.centroids.len(), 12);
        assert_eq!(a.assignments.len(), 300);
        assert!(a.assignments.iter().all(|&c| (c as usize) < 12));
    }

    #[test]
    fn kmeans_reseeds_empty_clusters() {
        // 99 duplicates of one point plus one distant outlier: strided init
        // seeds every centroid on the duplicate, so all but cluster 0 start
        // empty and must be reseeded (the outlier is the farthest point).
        let mut vectors = vec![vec![0.0f32, 0.0]; 99];
        vectors.push(vec![100.0f32, 100.0]);
        let result = kmeans(&vectors, 4, euclid);
        assert_eq!(result.centroids.len(), 4);
        // The outlier must end up in a cluster whose centroid is the outlier
        // itself, i.e. at distance 0.
        let outlier_cluster = result.assignments[99] as usize;
        assert_eq!(
            euclid(&vectors[99], &result.centroids[outlier_cluster]),
            0.0
        );
        // And the duplicates must not share it.
        assert_ne!(result.assignments[0], result.assignments[99]);
    }

    #[test]
    fn kmeans_clamps_k_and_handles_small_n() {
        let vectors = lcg_vectors(3, 5, 4);
        let result = kmeans(&vectors, 64, euclid);
        assert_eq!(result.centroids.len(), 5);
        assert_eq!(result.assignments.len(), 5);

        let empty = kmeans(&[], 8, euclid);
        assert!(empty.centroids.is_empty());
        assert!(empty.assignments.is_empty());
    }

    #[test]
    fn meta_blob_round_trips() {
        let meta = IvfMeta {
            k: 44,
            dim: 16,
            default_nprobe: 5,
        };
        assert_eq!(decode_meta(&encode_meta(&meta)).unwrap(), meta);
        assert!(decode_meta(b"nope").is_err());
        assert!(decode_meta(b"IVF1").is_err());
    }

    #[test]
    fn centroids_blob_round_trips() {
        let centroids = lcg_vectors(11, 9, 6);
        let blob = encode_centroids(6, &centroids).unwrap();
        let (dim, decoded) = decode_centroids(&blob).unwrap();
        assert_eq!(dim, 6);
        assert_eq!(decoded, centroids);
        assert!(decode_centroids(&blob[..blob.len() - 1]).is_err());
        assert!(encode_centroids(5, &centroids).is_err());
    }

    #[test]
    fn posting_blob_round_trips() {
        let ids: Vec<u128> = vec![1, u128::MAX, 42, 0];
        let blob = encode_posting(&ids);
        assert_eq!(decode_posting(&blob).unwrap(), ids);
        assert!(decode_posting(&blob[..blob.len() - 1]).is_err());
        assert!(decode_posting(&[]).unwrap().is_empty());
    }

    #[test]
    fn default_sizing() {
        assert_eq!(default_k(0), 1);
        assert_eq!(default_k(5), 2);
        assert_eq!(default_k(2000), 44);
        assert_eq!(default_k(100_000_000), IVF_MAX_K);
        assert_eq!(default_nprobe(1), 1);
        assert_eq!(default_nprobe(44), 5);
        assert_eq!(default_nprobe(4096), 64);
    }
}
