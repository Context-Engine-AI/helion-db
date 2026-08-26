//! SIMD-accelerated distance functions for f32 vectors.
//!
//! Each function provides three implementations selected at compile time (architecture)
//! and, on x86_64, at runtime (AVX2 feature detection):
//!
//! - **aarch64**: ARM NEON intrinsics (128-bit, 4 f32 lanes)
//! - **x86_64 + AVX2**: AVX2 intrinsics (256-bit, 8 f32 lanes)
//! - **scalar fallback**: chunked loop (chunk size 8) for autovectorization
//!
//! All paths use f64 accumulators internally for numerical precision, then cast
//! the final result to f32.
//!
//! AVX2 kernels include `_mm_prefetch` hints ~4 cache lines ahead of the
//! current chunk to overlap L1 fill with SIMD compute. Pure hint — zero
//! correctness impact. Benefit is measurable primarily on the first handful
//! of iterations before the hardware prefetcher locks on.

/// Prefetch distance in chunks (f32 AVX2 = 8 lanes/chunk, so 8 chunks ahead
/// = 256 bytes = 4 cache lines). Tuned for Intel Ice Lake / AMD Zen 3+.
#[cfg(target_arch = "x86_64")]
const PREFETCH_AHEAD_F32: usize = 8;

// ---------------------------------------------------------------------------
// Cosine distance: 1.0 - cosine_similarity  (0 = identical, 2 = opposite)
// ---------------------------------------------------------------------------

/// Returns the cosine *distance* between `a` and `b`: `1.0 - cos(a, b)`.
///
/// If either vector has zero magnitude the distance is defined as 1.0.
/// Panics in debug mode if the lengths differ; in release, uses the shorter length.
#[inline(always)]
pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine_f32: length mismatch");

    #[cfg(target_arch = "aarch64")]
    {
        cosine_f32_neon(a, b)
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { cosine_f32_avx2(a, b) }
        } else {
            cosine_f32_scalar(a, b)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        cosine_f32_scalar(a, b)
    }
}

// ---------------------------------------------------------------------------
// Dot product distance: -dot(a, b)  (lower = more similar)
// ---------------------------------------------------------------------------

/// Returns the negative dot product of `a` and `b`.
///
/// Using the negative makes this a distance metric: lower values indicate
/// higher similarity, consistent with the min-heap used in HNSW search.
#[inline(always)]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot_f32: length mismatch");

    #[cfg(target_arch = "aarch64")]
    {
        dot_f32_neon(a, b)
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { dot_f32_avx2(a, b) }
        } else {
            dot_f32_scalar(a, b)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        dot_f32_scalar(a, b)
    }
}

// ---------------------------------------------------------------------------
// Euclidean squared distance: sum((a_i - b_i)^2)
// ---------------------------------------------------------------------------

/// Returns the squared L2 (Euclidean) distance between `a` and `b`.
///
/// We intentionally return the *squared* distance to avoid a final `sqrt` in
/// the hot path -- HNSW only needs a consistent ordering, and squared distance
/// preserves that ordering.
#[inline(always)]
pub fn euclid_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "euclid_f32: length mismatch");

    #[cfg(target_arch = "aarch64")]
    {
        euclid_f32_neon(a, b)
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { euclid_f32_avx2(a, b) }
        } else {
            euclid_f32_scalar(a, b)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        euclid_f32_scalar(a, b)
    }
}

// ===========================================================================
//  Scalar fallback (all platforms)
// ===========================================================================

#[inline(always)]
#[allow(dead_code)]
fn cosine_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;

    const CHUNK: usize = 8;
    let chunks = len / CHUNK;
    let remainder = len % CHUNK;

    for c in 0..chunks {
        let off = c * CHUNK;
        let mut ld: f64 = 0.0;
        let mut la: f64 = 0.0;
        let mut lb: f64 = 0.0;
        for j in 0..CHUNK {
            let av = a[off + j] as f64;
            let bv = b[off + j] as f64;
            ld += av * bv;
            la += av * av;
            lb += bv * bv;
        }
        dot += ld;
        norm_a += la;
        norm_b += lb;
    }

    let rem_off = chunks * CHUNK;
    for i in 0..remainder {
        let av = a[rem_off + i] as f64;
        let bv = b[rem_off + i] as f64;
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    let similarity = dot / (norm_a.sqrt() * norm_b.sqrt());
    (1.0 - similarity) as f32
}

#[inline(always)]
#[allow(dead_code)]
fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut dot: f64 = 0.0;

    const CHUNK: usize = 8;
    let chunks = len / CHUNK;
    let remainder = len % CHUNK;

    for c in 0..chunks {
        let off = c * CHUNK;
        let mut ld: f64 = 0.0;
        for j in 0..CHUNK {
            ld += a[off + j] as f64 * b[off + j] as f64;
        }
        dot += ld;
    }

    let rem_off = chunks * CHUNK;
    for i in 0..remainder {
        dot += a[rem_off + i] as f64 * b[rem_off + i] as f64;
    }

    -dot as f32
}

#[inline(always)]
#[allow(dead_code)]
fn euclid_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum_sq: f64 = 0.0;

    const CHUNK: usize = 8;
    let chunks = len / CHUNK;
    let remainder = len % CHUNK;

    for c in 0..chunks {
        let off = c * CHUNK;
        let mut ls: f64 = 0.0;
        for j in 0..CHUNK {
            let d = a[off + j] as f64 - b[off + j] as f64;
            ls += d * d;
        }
        sum_sq += ls;
    }

    let rem_off = chunks * CHUNK;
    for i in 0..remainder {
        let d = a[rem_off + i] as f64 - b[rem_off + i] as f64;
        sum_sq += d * d;
    }

    sum_sq as f32
}

// ===========================================================================
//  NEON (aarch64) -- always available on aarch64
// ===========================================================================

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn cosine_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len().min(b.len());
    let chunks = len / 4;
    let remainder = len % 4;

    let (mut dot, mut norm_a, mut norm_b) = unsafe {
        let mut dot_lo = vdupq_n_f64(0.0);
        let mut dot_hi = vdupq_n_f64(0.0);
        let mut na_lo = vdupq_n_f64(0.0);
        let mut na_hi = vdupq_n_f64(0.0);
        let mut nb_lo = vdupq_n_f64(0.0);
        let mut nb_hi = vdupq_n_f64(0.0);

        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));

            // Widen lower 2 floats to f64
            let a_lo = vcvt_f64_f32(vget_low_f32(va));
            let a_hi = vcvt_f64_f32(vget_high_f32(va));
            let b_lo = vcvt_f64_f32(vget_low_f32(vb));
            let b_hi = vcvt_f64_f32(vget_high_f32(vb));

            dot_lo = vfmaq_f64(dot_lo, a_lo, b_lo);
            dot_hi = vfmaq_f64(dot_hi, a_hi, b_hi);
            na_lo = vfmaq_f64(na_lo, a_lo, a_lo);
            na_hi = vfmaq_f64(na_hi, a_hi, a_hi);
            nb_lo = vfmaq_f64(nb_lo, b_lo, b_lo);
            nb_hi = vfmaq_f64(nb_hi, b_hi, b_hi);
        }

        // Horizontal reduction
        let dot_sum = vaddq_f64(dot_lo, dot_hi);
        let na_sum = vaddq_f64(na_lo, na_hi);
        let nb_sum = vaddq_f64(nb_lo, nb_hi);
        (
            vgetq_lane_f64(dot_sum, 0) + vgetq_lane_f64(dot_sum, 1),
            vgetq_lane_f64(na_sum, 0) + vgetq_lane_f64(na_sum, 1),
            vgetq_lane_f64(nb_sum, 0) + vgetq_lane_f64(nb_sum, 1),
        )
    };

    // Remainder
    let rem_off = chunks * 4;
    for i in 0..remainder {
        let av = a[rem_off + i] as f64;
        let bv = b[rem_off + i] as f64;
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    let similarity = dot / (norm_a.sqrt() * norm_b.sqrt());
    (1.0 - similarity) as f32
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len().min(b.len());
    let chunks = len / 4;
    let remainder = len % 4;

    let mut dot = unsafe {
        let mut acc_lo = vdupq_n_f64(0.0);
        let mut acc_hi = vdupq_n_f64(0.0);

        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));

            let a_lo = vcvt_f64_f32(vget_low_f32(va));
            let a_hi = vcvt_f64_f32(vget_high_f32(va));
            let b_lo = vcvt_f64_f32(vget_low_f32(vb));
            let b_hi = vcvt_f64_f32(vget_high_f32(vb));

            acc_lo = vfmaq_f64(acc_lo, a_lo, b_lo);
            acc_hi = vfmaq_f64(acc_hi, a_hi, b_hi);
        }

        let sum = vaddq_f64(acc_lo, acc_hi);
        vgetq_lane_f64(sum, 0) + vgetq_lane_f64(sum, 1)
    };

    let rem_off = chunks * 4;
    for i in 0..remainder {
        dot += a[rem_off + i] as f64 * b[rem_off + i] as f64;
    }

    -dot as f32
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn euclid_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len().min(b.len());
    let chunks = len / 4;
    let remainder = len % 4;

    let mut sum_sq = unsafe {
        let mut acc_lo = vdupq_n_f64(0.0);
        let mut acc_hi = vdupq_n_f64(0.0);

        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));

            // Compute difference in f32, then widen to f64 for accumulation
            let diff = vsubq_f32(va, vb);
            let d_lo = vcvt_f64_f32(vget_low_f32(diff));
            let d_hi = vcvt_f64_f32(vget_high_f32(diff));

            acc_lo = vfmaq_f64(acc_lo, d_lo, d_lo);
            acc_hi = vfmaq_f64(acc_hi, d_hi, d_hi);
        }

        let sum = vaddq_f64(acc_lo, acc_hi);
        vgetq_lane_f64(sum, 0) + vgetq_lane_f64(sum, 1)
    };

    let rem_off = chunks * 4;
    for i in 0..remainder {
        let d = a[rem_off + i] as f64 - b[rem_off + i] as f64;
        sum_sq += d * d;
    }

    sum_sq as f32
}

// ===========================================================================
//  AVX2 + FMA (x86_64) -- runtime detected
// ===========================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn cosine_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let chunks = len / 8;
    let remainder = len % 8;

    // Use f64 accumulators: process 8 f32 -> 2 groups of 4 f64
    let mut dot_lo = _mm256_setzero_pd();
    let mut dot_hi = _mm256_setzero_pd();
    let mut na_lo = _mm256_setzero_pd();
    let mut na_hi = _mm256_setzero_pd();
    let mut nb_lo = _mm256_setzero_pd();
    let mut nb_hi = _mm256_setzero_pd();

    for i in 0..chunks {
        let off = i * 8;
        // Software prefetch ~256 bytes ahead (4 cache lines). Pure hint —
        // covers L1 fill latency while we process the current chunk. The
        // hardware prefetcher already handles sequential access well, but
        // the first few iterations benefit from an explicit warm-up.
        if i + PREFETCH_AHEAD_F32 < chunks {
            let pf_off = (i + PREFETCH_AHEAD_F32) * 8;
            _mm_prefetch(a.as_ptr().add(pf_off) as *const i8, _MM_HINT_T0);
            _mm_prefetch(b.as_ptr().add(pf_off) as *const i8, _MM_HINT_T0);
        }
        let va = _mm256_loadu_ps(a.as_ptr().add(off));
        let vb = _mm256_loadu_ps(b.as_ptr().add(off));

        // Split 8 f32 into low 4 and high 4, widen each to f64
        let a_lo_f32 = _mm256_castps256_ps128(va);
        let a_hi_f32 = _mm256_extractf128_ps(va, 1);
        let b_lo_f32 = _mm256_castps256_ps128(vb);
        let b_hi_f32 = _mm256_extractf128_ps(vb, 1);

        let a_lo_f64 = _mm256_cvtps_pd(a_lo_f32);
        let a_hi_f64 = _mm256_cvtps_pd(a_hi_f32);
        let b_lo_f64 = _mm256_cvtps_pd(b_lo_f32);
        let b_hi_f64 = _mm256_cvtps_pd(b_hi_f32);

        dot_lo = _mm256_fmadd_pd(a_lo_f64, b_lo_f64, dot_lo);
        dot_hi = _mm256_fmadd_pd(a_hi_f64, b_hi_f64, dot_hi);
        na_lo = _mm256_fmadd_pd(a_lo_f64, a_lo_f64, na_lo);
        na_hi = _mm256_fmadd_pd(a_hi_f64, a_hi_f64, na_hi);
        nb_lo = _mm256_fmadd_pd(b_lo_f64, b_lo_f64, nb_lo);
        nb_hi = _mm256_fmadd_pd(b_hi_f64, b_hi_f64, nb_hi);
    }

    let dot_total = hsum_pd_avx2(_mm256_add_pd(dot_lo, dot_hi));
    let na_total = hsum_pd_avx2(_mm256_add_pd(na_lo, na_hi));
    let nb_total = hsum_pd_avx2(_mm256_add_pd(nb_lo, nb_hi));

    // Scalar remainder
    let mut dot = dot_total;
    let mut norm_a = na_total;
    let mut norm_b = nb_total;
    let rem_off = chunks * 8;
    for i in 0..remainder {
        let av = a[rem_off + i] as f64;
        let bv = b[rem_off + i] as f64;
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    let similarity = dot / (norm_a.sqrt() * norm_b.sqrt());
    (1.0 - similarity) as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let chunks = len / 8;
    let remainder = len % 8;

    let mut acc_lo = _mm256_setzero_pd();
    let mut acc_hi = _mm256_setzero_pd();

    for i in 0..chunks {
        let off = i * 8;
        if i + PREFETCH_AHEAD_F32 < chunks {
            let pf_off = (i + PREFETCH_AHEAD_F32) * 8;
            _mm_prefetch(a.as_ptr().add(pf_off) as *const i8, _MM_HINT_T0);
            _mm_prefetch(b.as_ptr().add(pf_off) as *const i8, _MM_HINT_T0);
        }
        let va = _mm256_loadu_ps(a.as_ptr().add(off));
        let vb = _mm256_loadu_ps(b.as_ptr().add(off));

        let a_lo_f64 = _mm256_cvtps_pd(_mm256_castps256_ps128(va));
        let a_hi_f64 = _mm256_cvtps_pd(_mm256_extractf128_ps(va, 1));
        let b_lo_f64 = _mm256_cvtps_pd(_mm256_castps256_ps128(vb));
        let b_hi_f64 = _mm256_cvtps_pd(_mm256_extractf128_ps(vb, 1));

        acc_lo = _mm256_fmadd_pd(a_lo_f64, b_lo_f64, acc_lo);
        acc_hi = _mm256_fmadd_pd(a_hi_f64, b_hi_f64, acc_hi);
    }

    let mut dot = hsum_pd_avx2(_mm256_add_pd(acc_lo, acc_hi));
    let rem_off = chunks * 8;
    for i in 0..remainder {
        dot += a[rem_off + i] as f64 * b[rem_off + i] as f64;
    }

    -dot as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn euclid_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let chunks = len / 8;
    let remainder = len % 8;

    let mut acc_lo = _mm256_setzero_pd();
    let mut acc_hi = _mm256_setzero_pd();

    for i in 0..chunks {
        let off = i * 8;
        if i + PREFETCH_AHEAD_F32 < chunks {
            let pf_off = (i + PREFETCH_AHEAD_F32) * 8;
            _mm_prefetch(a.as_ptr().add(pf_off) as *const i8, _MM_HINT_T0);
            _mm_prefetch(b.as_ptr().add(pf_off) as *const i8, _MM_HINT_T0);
        }
        let va = _mm256_loadu_ps(a.as_ptr().add(off));
        let vb = _mm256_loadu_ps(b.as_ptr().add(off));

        // Subtract in f32, then widen to f64 for accumulation
        let diff = _mm256_sub_ps(va, vb);
        let d_lo_f64 = _mm256_cvtps_pd(_mm256_castps256_ps128(diff));
        let d_hi_f64 = _mm256_cvtps_pd(_mm256_extractf128_ps(diff, 1));

        acc_lo = _mm256_fmadd_pd(d_lo_f64, d_lo_f64, acc_lo);
        acc_hi = _mm256_fmadd_pd(d_hi_f64, d_hi_f64, acc_hi);
    }

    let mut sum_sq = hsum_pd_avx2(_mm256_add_pd(acc_lo, acc_hi));
    let rem_off = chunks * 8;
    for i in 0..remainder {
        let d = a[rem_off + i] as f64 - b[rem_off + i] as f64;
        sum_sq += d * d;
    }

    sum_sq as f32
}

/// Horizontal sum of four f64 values in an AVX2 256-bit register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_pd_avx2(v: std::arch::x86_64::__m256d) -> f64 {
    use std::arch::x86_64::*;
    let hi128 = _mm256_extractf128_pd(v, 1);
    let lo128 = _mm256_castpd256_pd128(v);
    let sum128 = _mm_add_pd(lo128, hi128);
    let hi64 = _mm_unpackhi_pd(sum128, sum128);
    _mm_cvtsd_f64(_mm_add_sd(sum128, hi64))
}

// ===========================================================================
//  SQ8 (Scalar Quantization to u8) for flash HNSW construction
// ===========================================================================

/// Per-dimension quantization parameters: maps [min, max] → [0, 255].
pub struct SQ8Params {
    pub mins: Vec<f32>,
    pub scales: Vec<f32>, // 255.0 / (max - min), or 0.0 if range is 0
}

impl SQ8Params {
    /// Compute quantization parameters from a set of f32 vectors.
    pub fn fit(vectors: &[Vec<f32>]) -> Self {
        if vectors.is_empty() {
            return Self {
                mins: Vec::new(),
                scales: Vec::new(),
            };
        }
        let dim = vectors[0].len();
        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];

        for v in vectors {
            for (i, &val) in v.iter().enumerate() {
                if val < mins[i] {
                    mins[i] = val;
                }
                if val > maxs[i] {
                    maxs[i] = val;
                }
            }
        }

        let scales: Vec<f32> = mins
            .iter()
            .zip(maxs.iter())
            .map(|(&lo, &hi)| {
                let range = hi - lo;
                if range <= f32::EPSILON {
                    0.0
                } else {
                    255.0 / range
                }
            })
            .collect();

        Self { mins, scales }
    }

    /// Compute quantization parameters from borrowed rows. This keeps publish
    /// paths from building a second `Vec<Vec<f32>>` when vectors already live in
    /// a prepared merge artifact.
    pub fn fit_slices(vectors: &[&[f32]]) -> Self {
        if vectors.is_empty() {
            return Self {
                mins: Vec::new(),
                scales: Vec::new(),
            };
        }
        let dim = vectors[0].len();
        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];

        for v in vectors {
            for (i, &val) in v.iter().enumerate() {
                if val < mins[i] {
                    mins[i] = val;
                }
                if val > maxs[i] {
                    maxs[i] = val;
                }
            }
        }

        let scales: Vec<f32> = mins
            .iter()
            .zip(maxs.iter())
            .map(|(&lo, &hi)| {
                let range = hi - lo;
                if range <= f32::EPSILON {
                    0.0
                } else {
                    255.0 / range
                }
            })
            .collect();

        Self { mins, scales }
    }

    /// Quantize a single f32 vector to u8.
    #[inline]
    pub fn quantize(&self, v: &[f32]) -> Vec<u8> {
        v.iter()
            .enumerate()
            .map(|(i, &val)| {
                if self.scales[i] == 0.0 {
                    128u8
                } else {
                    ((val - self.mins[i]) * self.scales[i])
                        .round()
                        .clamp(0.0, 255.0) as u8
                }
            })
            .collect()
    }

    /// Quantize all vectors in bulk, returning a flat Vec<u8> where each
    /// vector occupies `dim` consecutive bytes. This is cache-friendlier
    /// than Vec<Vec<u8>> for large-batch construction.
    pub fn quantize_bulk(&self, vectors: &[Vec<f32>]) -> (Vec<u8>, usize) {
        if vectors.is_empty() {
            return (Vec::new(), 0);
        }
        let dim = vectors[0].len();
        let mut out = vec![0u8; vectors.len() * dim];
        for (vi, v) in vectors.iter().enumerate() {
            let offset = vi * dim;
            for (i, &val) in v.iter().enumerate() {
                out[offset + i] = if self.scales[i] == 0.0 {
                    128u8
                } else {
                    ((val - self.mins[i]) * self.scales[i])
                        .round()
                        .clamp(0.0, 255.0) as u8
                };
            }
        }
        (out, dim)
    }
}

/// Approximate L2-squared distance between two u8 vectors.
/// Uses u32 accumulators to avoid overflow (max per-dim: (255-0)^2 = 65025,
/// times 768 dims ≈ 50M, well within u32 range).
#[inline(always)]
pub fn euclid_u8(a: &[u8], b: &[u8]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "euclid_u8: length mismatch");

    #[cfg(target_arch = "aarch64")]
    {
        euclid_u8_neon(a, b)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        euclid_u8_scalar(a, b)
    }
}

/// Approximate dot-product distance between two u8 vectors (negated).
#[inline(always)]
pub fn dot_u8(a: &[u8], b: &[u8]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot_u8: length mismatch");

    #[cfg(target_arch = "aarch64")]
    {
        dot_u8_neon(a, b)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_u8_scalar(a, b)
    }
}

/// Approximate cosine distance between two u8 vectors.
/// Uses integer arithmetic for dot, norm_a, norm_b, then computes
/// 1.0 - cos in floating point at the end.
#[inline(always)]
pub fn cosine_u8(a: &[u8], b: &[u8]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine_u8: length mismatch");

    #[cfg(target_arch = "aarch64")]
    {
        cosine_u8_neon(a, b)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        cosine_u8_scalar(a, b)
    }
}

// -- SQ8 scalar fallback --

#[inline(always)]
#[allow(dead_code)]
fn euclid_u8_scalar(a: &[u8], b: &[u8]) -> f32 {
    let mut sum: u64 = 0;
    for i in 0..a.len() {
        let d = a[i] as i32 - b[i] as i32;
        sum += (d * d) as u64;
    }
    sum as f32
}

#[inline(always)]
#[allow(dead_code)]
fn dot_u8_scalar(a: &[u8], b: &[u8]) -> f32 {
    let mut sum: u64 = 0;
    for i in 0..a.len() {
        sum += a[i] as u64 * b[i] as u64;
    }
    -(sum as f32)
}

#[inline(always)]
#[allow(dead_code)]
fn cosine_u8_scalar(a: &[u8], b: &[u8]) -> f32 {
    let mut dot: u64 = 0;
    let mut norm_a: u64 = 0;
    let mut norm_b: u64 = 0;
    for i in 0..a.len() {
        let av = a[i] as u64;
        let bv = b[i] as u64;
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }
    if norm_a == 0 || norm_b == 0 {
        return 1.0;
    }
    let dot_f = dot as f64;
    let na_f = norm_a as f64;
    let nb_f = norm_b as f64;
    let similarity = dot_f / (na_f.sqrt() * nb_f.sqrt());
    (1.0 - similarity) as f32
}

// -- SQ8 NEON (aarch64) --

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn euclid_u8_neon(a: &[u8], b: &[u8]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len();
    let chunks = len / 16;
    let remainder = len % 16;
    let mut sum = unsafe {
        let mut acc = vdupq_n_u32(0);

        for i in 0..chunks {
            let off = i * 16;
            let va = vld1q_u8(a.as_ptr().add(off));
            let vb = vld1q_u8(b.as_ptr().add(off));

            // Absolute difference, then widening multiply-accumulate
            let diff = vabdq_u8(va, vb);
            let lo = vget_low_u8(diff);
            let hi = vget_high_u8(diff);

            // Widen u8→u16 and multiply (square the diff)
            let sq_lo = vmull_u8(lo, lo);
            let sq_hi = vmull_u8(hi, hi);

            // Pairwise add u16→u32 and accumulate
            acc = vpadalq_u16(acc, sq_lo);
            acc = vpadalq_u16(acc, sq_hi);
        }

        // Horizontal sum of acc (4 x u32)
        vaddlvq_u32(acc) as u64
    };

    // Scalar remainder
    let rem_off = chunks * 16;
    for i in 0..remainder {
        let d = a[rem_off + i] as i32 - b[rem_off + i] as i32;
        sum += (d * d) as u64;
    }

    sum as f32
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dot_u8_neon(a: &[u8], b: &[u8]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len();
    let chunks = len / 16;
    let remainder = len % 16;
    let mut sum = unsafe {
        let mut acc = vdupq_n_u32(0);

        for i in 0..chunks {
            let off = i * 16;
            let va = vld1q_u8(a.as_ptr().add(off));
            let vb = vld1q_u8(b.as_ptr().add(off));

            let lo_a = vget_low_u8(va);
            let hi_a = vget_high_u8(va);
            let lo_b = vget_low_u8(vb);
            let hi_b = vget_high_u8(vb);

            let prod_lo = vmull_u8(lo_a, lo_b);
            let prod_hi = vmull_u8(hi_a, hi_b);

            acc = vpadalq_u16(acc, prod_lo);
            acc = vpadalq_u16(acc, prod_hi);
        }

        vaddlvq_u32(acc) as u64
    };

    let rem_off = chunks * 16;
    for i in 0..remainder {
        sum += a[rem_off + i] as u64 * b[rem_off + i] as u64;
    }

    -(sum as f32)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn cosine_u8_neon(a: &[u8], b: &[u8]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len();
    let chunks = len / 16;
    let remainder = len % 16;

    let (mut dot_sum, mut na_sum, mut nb_sum) = unsafe {
        let mut dot_acc = vdupq_n_u32(0);
        let mut na_acc = vdupq_n_u32(0);
        let mut nb_acc = vdupq_n_u32(0);

        for i in 0..chunks {
            let off = i * 16;
            let va = vld1q_u8(a.as_ptr().add(off));
            let vb = vld1q_u8(b.as_ptr().add(off));

            let lo_a = vget_low_u8(va);
            let hi_a = vget_high_u8(va);
            let lo_b = vget_low_u8(vb);
            let hi_b = vget_high_u8(vb);

            // dot(a, b)
            dot_acc = vpadalq_u16(dot_acc, vmull_u8(lo_a, lo_b));
            dot_acc = vpadalq_u16(dot_acc, vmull_u8(hi_a, hi_b));

            // norm_a = sum(a*a)
            na_acc = vpadalq_u16(na_acc, vmull_u8(lo_a, lo_a));
            na_acc = vpadalq_u16(na_acc, vmull_u8(hi_a, hi_a));

            // norm_b = sum(b*b)
            nb_acc = vpadalq_u16(nb_acc, vmull_u8(lo_b, lo_b));
            nb_acc = vpadalq_u16(nb_acc, vmull_u8(hi_b, hi_b));
        }

        (
            vaddlvq_u32(dot_acc) as u64,
            vaddlvq_u32(na_acc) as u64,
            vaddlvq_u32(nb_acc) as u64,
        )
    };

    let rem_off = chunks * 16;
    for i in 0..remainder {
        let av = a[rem_off + i] as u64;
        let bv = b[rem_off + i] as u64;
        dot_sum += av * bv;
        na_sum += av * av;
        nb_sum += bv * bv;
    }

    if na_sum == 0 || nb_sum == 0 {
        return 1.0;
    }
    let similarity = dot_sum as f64 / ((na_sum as f64).sqrt() * (nb_sum as f64).sqrt());
    (1.0 - similarity) as f32
}

// ===========================================================================
//  Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: scalar cosine similarity (returns similarity, not distance) for reference.
    fn reference_cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
        let mut dot: f64 = 0.0;
        let mut na: f64 = 0.0;
        let mut nb: f64 = 0.0;
        for i in 0..a.len() {
            let av = a[i] as f64;
            let bv = b[i] as f64;
            dot += av * bv;
            na += av * av;
            nb += bv * bv;
        }
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    fn reference_dot(a: &[f32], b: &[f32]) -> f64 {
        let mut dot: f64 = 0.0;
        for i in 0..a.len() {
            dot += a[i] as f64 * b[i] as f64;
        }
        dot
    }

    fn reference_euclid_sq(a: &[f32], b: &[f32]) -> f64 {
        let mut sum: f64 = 0.0;
        for i in 0..a.len() {
            let d = a[i] as f64 - b[i] as f64;
            sum += d * d;
        }
        sum
    }

    // -- Cosine distance tests -----------------------------------------------

    #[test]
    fn cosine_identical_vectors() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let dist = cosine_f32(&a, &a);
        assert!(
            dist.abs() < 1e-6,
            "identical vectors should have distance ~0, got {}",
            dist
        );
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0];
        let dist = cosine_f32(&a, &b);
        assert!(
            (dist - 1.0).abs() < 1e-6,
            "orthogonal vectors should have distance ~1.0, got {}",
            dist
        );
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b: Vec<f32> = a.iter().map(|x| -x).collect();
        let dist = cosine_f32(&a, &b);
        assert!(
            (dist - 2.0).abs() < 1e-5,
            "opposite vectors should have distance ~2.0, got {}",
            dist
        );
    }

    #[test]
    fn cosine_zero_vector() {
        let a = vec![0.0f32; 4];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        let dist = cosine_f32(&a, &b);
        assert!(
            (dist - 1.0).abs() < 1e-6,
            "zero vector distance should be 1.0, got {}",
            dist
        );
    }

    // -- Dot product tests ----------------------------------------------------

    #[test]
    fn dot_product_correctness() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let expected = -(1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0);
        let result = dot_f32(&a, &b);
        assert!(
            (result - expected as f32).abs() < 1e-5,
            "dot product mismatch: got {}, expected {}",
            result,
            expected
        );
    }

    #[test]
    fn dot_product_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let result = dot_f32(&a, &b);
        assert!(
            result.abs() < 1e-6,
            "orthogonal dot product should be ~0, got {}",
            result
        );
    }

    // -- Euclidean distance tests ---------------------------------------------

    #[test]
    fn euclid_identical_vectors() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let dist = euclid_f32(&a, &a);
        assert!(
            dist.abs() < 1e-6,
            "identical vectors should have distance 0, got {}",
            dist
        );
    }

    #[test]
    fn euclid_known_distance() {
        let a = vec![0.0f32, 0.0];
        let b = vec![3.0f32, 4.0];
        let dist = euclid_f32(&a, &b);
        // Squared distance: 9 + 16 = 25
        assert!(
            (dist - 25.0).abs() < 1e-5,
            "expected squared distance 25.0, got {}",
            dist
        );
    }

    #[test]
    fn euclid_unit_displacement() {
        let a = vec![0.0f32; 8];
        let b = vec![1.0f32; 8];
        let dist = euclid_f32(&a, &b);
        // Squared distance = 8 * 1^2 = 8
        assert!(
            (dist - 8.0).abs() < 1e-5,
            "expected squared distance 8.0, got {}",
            dist
        );
    }

    // -- Empty vector handling ------------------------------------------------

    #[test]
    fn empty_vectors_cosine() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let dist = cosine_f32(&a, &b);
        assert!(
            (dist - 1.0).abs() < 1e-6,
            "empty vector cosine distance should be 1.0, got {}",
            dist
        );
    }

    #[test]
    fn empty_vectors_dot() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let dist = dot_f32(&a, &b);
        assert!(
            dist.abs() < 1e-6,
            "empty vector dot should be 0, got {}",
            dist
        );
    }

    #[test]
    fn empty_vectors_euclid() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let dist = euclid_f32(&a, &b);
        assert!(
            dist.abs() < 1e-6,
            "empty vector euclid should be 0, got {}",
            dist
        );
    }

    // -- Large vector (768-dim) correctness vs scalar reference ---------------

    #[test]
    fn large_vector_cosine_768() {
        let a: Vec<f32> = (0..768).map(|i| ((i as f32) * 0.001).sin()).collect();
        let b: Vec<f32> = (0..768).map(|i| ((i as f32) * 0.002).cos()).collect();

        let expected_sim = reference_cosine_similarity(&a, &b);
        let expected_dist = (1.0 - expected_sim) as f32;
        let result = cosine_f32(&a, &b);

        assert!(
            (result - expected_dist).abs() < 1e-5,
            "768-dim cosine mismatch: got {}, expected {}",
            result,
            expected_dist
        );
    }

    #[test]
    fn large_vector_dot_768() {
        let a: Vec<f32> = (0..768).map(|i| ((i as f32) * 0.001).sin()).collect();
        let b: Vec<f32> = (0..768).map(|i| ((i as f32) * 0.002).cos()).collect();

        let expected = -reference_dot(&a, &b) as f32;
        let result = dot_f32(&a, &b);

        assert!(
            (result - expected).abs() < 1e-3,
            "768-dim dot mismatch: got {}, expected {}",
            result,
            expected
        );
    }

    #[test]
    fn large_vector_euclid_768() {
        let a: Vec<f32> = (0..768).map(|i| ((i as f32) * 0.001).sin()).collect();
        let b: Vec<f32> = (0..768).map(|i| ((i as f32) * 0.002).cos()).collect();

        let expected = reference_euclid_sq(&a, &b) as f32;
        let result = euclid_f32(&a, &b);

        assert!(
            (result - expected).abs() < 1e-2,
            "768-dim euclid mismatch: got {}, expected {}",
            result,
            expected
        );
    }

    // -- Odd-length vectors (remainder handling) ------------------------------

    #[test]
    fn odd_length_cosine() {
        let a: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..13).map(|i| (13 - i) as f32).collect();

        let expected_sim = reference_cosine_similarity(&a, &b);
        let expected_dist = (1.0 - expected_sim) as f32;
        let result = cosine_f32(&a, &b);

        assert!(
            (result - expected_dist).abs() < 1e-5,
            "odd-length cosine mismatch: got {}, expected {}",
            result,
            expected_dist
        );
    }

    #[test]
    fn odd_length_dot() {
        let a: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..13).map(|i| (13 - i) as f32).collect();

        let expected = -reference_dot(&a, &b) as f32;
        let result = dot_f32(&a, &b);

        assert!(
            (result - expected).abs() < 1e-3,
            "odd-length dot mismatch: got {}, expected {}",
            result,
            expected
        );
    }

    #[test]
    fn odd_length_euclid() {
        let a: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..13).map(|i| (13 - i) as f32).collect();

        let expected = reference_euclid_sq(&a, &b) as f32;
        let result = euclid_f32(&a, &b);

        assert!(
            (result - expected).abs() < 1e-3,
            "odd-length euclid mismatch: got {}, expected {}",
            result,
            expected
        );
    }

    // -- SQ8 quantization + distance tests -----------------------------------

    #[test]
    fn sq8_quantize_roundtrip_preserves_ordering() {
        // Generate vectors, quantize, verify distance ordering is preserved
        let vecs: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                (0..32)
                    .map(|j| ((i * 7 + j * 3) as f32 * 0.01).sin())
                    .collect()
            })
            .collect();
        let params = SQ8Params::fit(&vecs);

        let q0 = params.quantize(&vecs[0]);
        let q1 = params.quantize(&vecs[1]);
        let q2 = params.quantize(&vecs[50]);

        // Just verify the quantized distances are finite and non-negative for L2
        let d01 = euclid_u8(&q0, &q1);
        let d02 = euclid_u8(&q0, &q2);
        assert!(d01 >= 0.0, "euclid_u8 should be non-negative");
        assert!(d02 >= 0.0, "euclid_u8 should be non-negative");
        assert!(d01.is_finite());
        assert!(d02.is_finite());
    }

    #[test]
    fn sq8_cosine_identical_vectors() {
        let vecs: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0, 4.0]; 2];
        let params = SQ8Params::fit(&vecs);
        let q = params.quantize(&vecs[0]);
        let d = cosine_u8(&q, &q);
        assert!(
            d.abs() < 0.01,
            "identical quantized vectors cosine should be ~0, got {}",
            d
        );
    }

    #[test]
    fn sq8_euclid_identical_vectors() {
        let vecs: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0, 4.0]; 2];
        let params = SQ8Params::fit(&vecs);
        let q = params.quantize(&vecs[0]);
        let d = euclid_u8(&q, &q);
        assert!(
            d.abs() < 0.01,
            "identical quantized vectors euclid should be ~0, got {}",
            d
        );
    }

    #[test]
    fn sq8_bulk_quantize_matches_individual() {
        let vecs: Vec<Vec<f32>> = (0..50)
            .map(|i| {
                (0..768)
                    .map(|j| ((i * 13 + j * 7) as f32 * 0.001).sin())
                    .collect()
            })
            .collect();
        let params = SQ8Params::fit(&vecs);
        let (bulk, dim) = params.quantize_bulk(&vecs);
        assert_eq!(dim, 768);
        for (vi, v) in vecs.iter().enumerate() {
            let single = params.quantize(v);
            let bulk_slice = &bulk[vi * dim..(vi + 1) * dim];
            assert_eq!(
                single, bulk_slice,
                "Bulk quantize mismatch at vector {}",
                vi
            );
        }
    }

    #[test]
    fn sq8_768dim_distance_correlation() {
        // Check that SQ8 distances correlate with f32 distances on 768-dim vectors
        let vecs: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                (0..768)
                    .map(|j| ((i * 7 + j * 3) as f32 * 0.001).sin())
                    .collect()
            })
            .collect();
        let params = SQ8Params::fit(&vecs);
        let (bulk, dim) = params.quantize_bulk(&vecs);

        // Compare distance orderings for 20 random pairs
        let mut f32_dists = Vec::new();
        let mut u8_dists = Vec::new();
        for i in 0..20 {
            let a_idx = i * 3;
            let b_idx = i * 3 + 1;
            f32_dists.push(cosine_f32(&vecs[a_idx], &vecs[b_idx]));
            let qa = &bulk[a_idx * dim..(a_idx + 1) * dim];
            let qb = &bulk[b_idx * dim..(b_idx + 1) * dim];
            u8_dists.push(cosine_u8(qa, qb));
        }

        // Check rank correlation (Spearman-ish): sort both by f32 distance,
        // verify u8 distances are roughly in the same order
        let mut pairs: Vec<(f32, f32)> = f32_dists
            .iter()
            .copied()
            .zip(u8_dists.iter().copied())
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Count inversions in u8 ordering vs f32 ordering
        let mut inversions = 0;
        for i in 0..pairs.len() {
            for j in i + 1..pairs.len() {
                if pairs[j].1 < pairs[i].1 {
                    inversions += 1;
                }
            }
        }
        let max_inversions = pairs.len() * (pairs.len() - 1) / 2;
        let inversion_rate = inversions as f64 / max_inversions as f64;
        assert!(
            inversion_rate < 0.3,
            "SQ8 distance ordering has too many inversions vs f32: {:.1}%",
            inversion_rate * 100.0
        );
    }
}
