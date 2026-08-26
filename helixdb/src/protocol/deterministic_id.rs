use std::hash::Hasher;
use twox_hash::XxHash64;

/// Compute a deterministic u128 ID from composite key parts.
///
/// Uses two XxHash64 passes with different seeds to produce a 128-bit ID.
/// This makes upserts trivial: same natural key always produces the same ID,
/// so `put()` with the deterministic ID overwrites existing data.
///
/// Collision probability is ~10^-20 at 100M items — effectively zero.
#[inline]
pub fn deterministic_id(parts: &[&str]) -> u128 {
    let hi = hash_with_seed(parts, 0);
    let lo = hash_with_seed(parts, 0x9E3779B97F4A7C15); // golden ratio constant
    ((hi as u128) << 64) | (lo as u128)
}

#[inline]
fn hash_with_seed(parts: &[&str], seed: u64) -> u64 {
    let mut hasher = XxHash64::with_seed(seed);
    for (i, part) in parts.iter().enumerate() {
        if i > 0 {
            hasher.write_u8(0xFF); // separator byte (not valid UTF-8 in strings)
        }
        hasher.write(part.as_bytes());
    }
    hasher.finish()
}

/// Compute deterministic node ID from natural key.
/// Key: node:{collection}:{label}:{name}:{path}
#[inline]
pub fn node_id(collection: &str, label: &str, name: &str, path: &str) -> u128 {
    deterministic_id(&["node", collection, label, name, path])
}

/// Compute deterministic edge ID from natural key.
/// Key: edge:{collection}:{edge_type}:{caller_symbol}:{callee_symbol}:{caller_path}:{callee_path}
#[inline]
pub fn edge_id(
    collection: &str,
    edge_type: &str,
    from_symbol: &str,
    to_symbol: &str,
    from_path: &str,
    to_path: &str,
) -> u128 {
    deterministic_id(&[
        "edge",
        collection,
        edge_type,
        from_symbol,
        to_symbol,
        from_path,
        to_path,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_stability() {
        let id1 = node_id("coll_abc", "Symbol", "main", "src/main.rs");
        let id2 = node_id("coll_abc", "Symbol", "main", "src/main.rs");
        assert_eq!(id1, id2, "Same inputs must produce same ID");
    }

    #[test]
    fn test_different_inputs_differ() {
        let id1 = node_id("coll_abc", "Symbol", "main", "src/main.rs");
        let id2 = node_id("coll_abc", "Symbol", "helper", "src/main.rs");
        assert_ne!(id1, id2, "Different inputs must produce different IDs");
    }

    #[test]
    fn test_collection_scoping() {
        let id1 = node_id("coll_aaa", "Symbol", "main", "src/main.rs");
        let id2 = node_id("coll_bbb", "Symbol", "main", "src/main.rs");
        assert_ne!(id1, id2, "Different collections must produce different IDs");
    }

    #[test]
    fn test_edge_id_stability() {
        let id1 = edge_id(
            "coll",
            "CALLS",
            "main",
            "helper",
            "src/main.rs",
            "src/util.rs",
        );
        let id2 = edge_id(
            "coll",
            "CALLS",
            "main",
            "helper",
            "src/main.rs",
            "src/util.rs",
        );
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_edge_id_direction_matters() {
        let id1 = edge_id("coll", "CALLS", "main", "helper", "a.rs", "b.rs");
        let id2 = edge_id("coll", "CALLS", "helper", "main", "b.rs", "a.rs");
        assert_ne!(id1, id2, "Edge direction must matter");
    }

    #[test]
    fn test_separator_prevents_collision() {
        // "a" + "bc" vs "ab" + "c" should differ
        let id1 = deterministic_id(&["a", "bc"]);
        let id2 = deterministic_id(&["ab", "c"]);
        assert_ne!(
            id1, id2,
            "Separator must prevent key-part boundary collisions"
        );
    }

    #[test]
    fn test_nonzero() {
        let id = node_id("c", "S", "n", "p");
        assert_ne!(id, 0, "ID should not be zero");
    }
}
