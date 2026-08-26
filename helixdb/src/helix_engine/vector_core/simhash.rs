//! SimHash codes for dense vectors — the storage foundation for
//! LSM-locality-aware vector layout and cheap angular pre-filtering.
//!
//! Ported (as a mechanism, not code) from upstream HelixDB's SimHash
//! directory design: a vector's 64-bit SimHash is a random-hyperplane
//! signature whose Hamming distance estimates angular distance
//! (Goemans–Williamson: `P[bit collision] = 1 - θ/π`). The **order code**
//! bit-interleaves the hash's four 16-bit planes (Morton/Z-order), so a
//! shared order-code prefix implies signature agreement spread across the
//! whole hash rather than concentrated in one quarter — which is what makes
//! an order-code prefix range usable as an LSH bucket and makes
//! order-code-sorted storage keys place cosine-similar vectors physically
//! adjacent in SSTables.
//!
//! Current use (flag-gated, additive): the LSM dense insert path persists a
//! per-vector `simhash ++ order_code` row in the segment's
//! [`super::super::storage_core::backend::SegmentDb::SimHash`] keyspace for
//! NEWLY written vectors. Nothing reads it on the search path yet — enabling
//! search-side use (locality-sorted fetches, Hamming-ordered probe seeding,
//! distance-skip filtering) is retrieval-affecting and goes through the eval
//! gate separately. Existing segments are untouched: no reindex, no
//! migration.

/// Deployed hyperplane seed. Changing it invalidates every persisted SimHash
/// row, so it is a constant, not a knob.
pub const SIMHASH_SEED: u64 = 42;

/// Env knob: persist a SimHash row for each NEWLY inserted dense vector on
/// the LSM backend (`SegmentDb::SimHash`). Default OFF. Purely additive —
/// existing vectors/segments are untouched and nothing on the search path
/// reads the rows yet.
const ENV_VECTOR_SIMHASH: &str = "HELIX_VECTOR_SIMHASH";

pub fn simhash_rows_enabled() -> bool {
    std::env::var(ENV_VECTOR_SIMHASH)
        .ok()
        .map(|v| matches!(v.trim(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

/// SplitMix64 — deterministic, dependency-free PRNG for hyperplane
/// components. Statistically solid for sign-projection purposes.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Map a u64 to a standard-ish normal-ish component in (-1, 1). A uniform
/// component is sufficient for random-hyperplane SimHash (only the sign of
/// the dot product matters and uniform hyperplanes preserve the angular
/// collision identity in expectation).
#[inline]
fn plane_component(raw: u64) -> f32 {
    // Uniform in (-1, 1), never exactly 0.
    let unit = ((raw >> 11) as f32) * (1.0 / (1u64 << 53) as f32);
    unit.mul_add(2.0, -1.0)
}

/// 64-bit random-hyperplane SimHash of `vector` under `seed`. Deterministic
/// in (seed, dimension, values); O(64 × d), computed once per insert.
pub fn simhash64(vector: &[f32], seed: u64) -> u64 {
    let mut bits = 0u64;
    for bit in 0..64u64 {
        // Each hyperplane's component stream is keyed by (seed, bit) so a
        // vector's bits are independent and dimension-stable.
        let mut state = seed ^ (bit.wrapping_mul(0xA24B_AED4_963E_E407));
        let mut dot = 0.0f32;
        for &component in vector {
            dot += component * plane_component(splitmix64(&mut state));
        }
        if dot >= 0.0 {
            bits |= 1 << (63 - bit);
        }
    }
    bits
}

/// Z-order (Morton) interleave of the SimHash's four 16-bit planes. The high
/// bits of the result carry the TOP bits of ALL four planes, so a shared
/// order-code prefix means agreement spread across the whole signature —
/// the property that makes prefix ranges meaningful LSH buckets and makes
/// order-code-sorted keys locality-preserving for cosine similarity.
pub fn order_code_from_simhash_bits(bits: u64) -> u64 {
    let b0 = ((bits >> 48) & 0xFFFF) as u16;
    let b1 = ((bits >> 32) & 0xFFFF) as u16;
    let b2 = ((bits >> 16) & 0xFFFF) as u16;
    let b3 = (bits & 0xFFFF) as u16;
    let mut code = 0u64;
    for bit in (0..16).rev() {
        code = (code << 1) | u64::from((b0 >> bit) & 1);
        code = (code << 1) | u64::from((b1 >> bit) & 1);
        code = (code << 1) | u64::from((b2 >> bit) & 1);
        code = (code << 1) | u64::from((b3 >> bit) & 1);
    }
    code
}

/// Every u16 prefix offset ordered by ascending Hamming weight (then value):
/// probing `query_prefix ^ offsets[i]` visits candidate order-code prefixes
/// in ascending Hamming distance — multi-probe LSH as plain ordered range
/// scans. The first 17 entries are the exact prefix plus all one-bit flips.
pub fn directory_prefix_offsets() -> &'static [u16] {
    use std::sync::OnceLock;
    static OFFSETS: OnceLock<Vec<u16>> = OnceLock::new();
    OFFSETS.get_or_init(|| {
        let mut offsets: Vec<u16> = (u16::MIN..=u16::MAX).collect();
        offsets.sort_unstable_by_key(|offset| (offset.count_ones(), *offset));
        offsets
    })
}

/// Version tag on every persisted SimHash row, so a future format change
/// (wider hashes, different seed lineage) is detectable instead of misread.
const SIMHASH_ROW_VERSION_V1: u8 = 0x01;

/// Encoded per-vector SimHash row value:
/// `[version: u8][simhash: u64 BE][order_code: u64 BE]`.
pub fn encode_simhash_row(simhash: u64) -> [u8; 17] {
    let order_code = order_code_from_simhash_bits(simhash);
    let mut row = [0u8; 17];
    row[0] = SIMHASH_ROW_VERSION_V1;
    row[1..9].copy_from_slice(&simhash.to_be_bytes());
    row[9..].copy_from_slice(&order_code.to_be_bytes());
    row
}

/// Decode a persisted SimHash row back into `(simhash, order_code)`.
/// `None` for unknown versions or malformed rows (callers treat both as
/// "vector has no code").
pub fn decode_simhash_row(row: &[u8]) -> Option<(u64, u64)> {
    if row.len() != 17 || row[0] != SIMHASH_ROW_VERSION_V1 {
        return None;
    }
    let simhash = u64::from_be_bytes(row[1..9].try_into().ok()?);
    let order_code = u64::from_be_bytes(row[9..].try_into().ok()?);
    Some((simhash, order_code))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simhash_is_deterministic_and_scale_invariant() {
        let v: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let scaled: Vec<f32> = v.iter().map(|x| x * 3.5).collect();
        assert_eq!(simhash64(&v, SIMHASH_SEED), simhash64(&v, SIMHASH_SEED));
        // Sign projections are scale-invariant for positive scaling.
        assert_eq!(
            simhash64(&v, SIMHASH_SEED),
            simhash64(&scaled, SIMHASH_SEED)
        );
    }

    #[test]
    fn similar_vectors_have_closer_simhash_than_dissimilar() {
        let base: Vec<f32> = (0..128).map(|i| ((i * 7 + 3) as f32).sin()).collect();
        let near: Vec<f32> = base
            .iter()
            .enumerate()
            .map(|(i, x)| x + 0.01 * ((i * 13) as f32).cos())
            .collect();
        let far: Vec<f32> = base.iter().map(|x| -x).collect();
        let h_base = simhash64(&base, SIMHASH_SEED);
        let h_near = simhash64(&near, SIMHASH_SEED);
        let h_far = simhash64(&far, SIMHASH_SEED);
        let d_near = (h_base ^ h_near).count_ones();
        let d_far = (h_base ^ h_far).count_ones();
        assert!(
            d_near < d_far,
            "near Hamming {d_near} must beat far Hamming {d_far}"
        );
        // A negated vector flips every hyperplane sign.
        assert_eq!(d_far, 64);
    }

    #[test]
    fn order_code_interleaves_all_four_planes() {
        // Top bit of each 16-bit plane must land in the order code's top
        // nibble (one bit from each plane), not in one contiguous quarter.
        assert_eq!(order_code_from_simhash_bits(1 << 63), 1 << 63);
        assert_eq!(order_code_from_simhash_bits(1 << 47), 1 << 62);
        assert_eq!(order_code_from_simhash_bits(1 << 31), 1 << 61);
        assert_eq!(order_code_from_simhash_bits(1 << 15), 1 << 60);
    }

    #[test]
    fn directory_offsets_are_hamming_ordered() {
        let offsets = directory_prefix_offsets();
        assert_eq!(offsets.len(), 65_536);
        assert_eq!(offsets[0], 0);
        // 1..=16 are exactly the one-bit flips.
        for offset in &offsets[1..17] {
            assert_eq!(offset.count_ones(), 1);
        }
        assert!(offsets
            .windows(2)
            .all(|w| (w[0].count_ones(), w[0]) < (w[1].count_ones(), w[1])));
    }

    #[test]
    fn simhash_row_round_trips() {
        let simhash = 0xDEAD_BEEF_1234_5678u64;
        let row = encode_simhash_row(simhash);
        let (decoded, order_code) = decode_simhash_row(&row).expect("row decodes");
        assert_eq!(decoded, simhash);
        assert_eq!(order_code, order_code_from_simhash_bits(simhash));
        assert_eq!(decode_simhash_row(&row[..16]), None, "truncated row");
        let mut unknown_version = row;
        unknown_version[0] = 0x7F;
        assert_eq!(decode_simhash_row(&unknown_version), None, "future version");
    }
}
