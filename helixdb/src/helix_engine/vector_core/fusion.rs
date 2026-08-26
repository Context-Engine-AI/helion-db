use std::collections::HashMap;
use std::collections::HashSet;

/// Reciprocal Rank Fusion (RRF) for combining multiple ranked result lists.
///
/// CE uses this for hybrid search: combining dense vector results with lexical results.
/// Score for each item = sum(1 / (k + rank_in_list)) across all lists containing it.
///
/// Standard k=60 (from the original RRF paper: Cormack, Clarke, Buettcher 2009).

const DEFAULT_K: f64 = 60.0;

/// A ranked item: point ID + score.
#[derive(Debug, Clone)]
pub struct RankedItem {
    pub id: u128,
    pub score: f64,
}

/// A ranked item that also carries the dense vector used for diversity rerank.
#[derive(Debug, Clone)]
pub struct VectorRankedItem {
    pub id: u128,
    pub score: f64,
    pub vector: Vec<f32>,
}

struct VectorRankedItemRef<'a> {
    id: u128,
    score: f64,
    vector: &'a [f32],
}

/// Normalization strategies for score ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationMethod {
    MinMax,
    ZScore,
    None,
}

/// Normalize scores using the requested method.
///
/// Empty input returns an empty vector. Constant min-max scores become 0.5 so
/// callers can still blend them with other normalized channels.
pub fn normalize_scores(scores: &[f64], method: NormalizationMethod) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }

    match method {
        NormalizationMethod::MinMax => {
            let min = scores
                .iter()
                .fold(f64::INFINITY, |acc, score| acc.min(*score));
            let max = scores
                .iter()
                .fold(f64::NEG_INFINITY, |acc, score| acc.max(*score));
            let range = max - min;
            if range == 0.0 {
                vec![0.5; scores.len()]
            } else {
                scores.iter().map(|score| (score - min) / range).collect()
            }
        }
        NormalizationMethod::ZScore => {
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            let variance = scores
                .iter()
                .map(|score| (score - mean).powi(2))
                .sum::<f64>()
                / scores.len() as f64;
            let stddev = variance.sqrt();
            if stddev == 0.0 {
                vec![0.0; scores.len()]
            } else {
                scores.iter().map(|score| (score - mean) / stddev).collect()
            }
        }
        NormalizationMethod::None => scores.to_vec(),
    }
}

/// Return a copy of the ranked items with normalized scores.
pub fn normalize_ranked_scores(
    items: &[RankedItem],
    method: NormalizationMethod,
) -> Vec<RankedItem> {
    let scores: Vec<f64> = items.iter().map(|item| item.score).collect();
    let normalized = normalize_scores(&scores, method);
    items
        .iter()
        .zip(normalized)
        .map(|(item, score)| RankedItem { id: item.id, score })
        .collect()
}

/// Fuse multiple ranked lists using Reciprocal Rank Fusion.
///
/// Each input list is ordered by relevance (best first).
/// Returns a merged list sorted by RRF score (highest first), limited to `limit` results.
pub fn rrf_fusion(ranked_lists: &[Vec<RankedItem>], limit: usize) -> Vec<RankedItem> {
    rrf_fusion_with_k(ranked_lists, DEFAULT_K, limit)
}

/// RRF with configurable k parameter.
pub fn rrf_fusion_with_k(
    ranked_lists: &[Vec<RankedItem>],
    k: f64,
    limit: usize,
) -> Vec<RankedItem> {
    let mut scores: HashMap<u128, f64> = HashMap::new();

    for list in ranked_lists {
        for (rank, item) in list.iter().enumerate() {
            let rrf_score = 1.0 / (k + (rank + 1) as f64);
            *scores.entry(item.id).or_insert(0.0) += rrf_score;
        }
    }

    let mut results: Vec<RankedItem> = scores
        .into_iter()
        .map(|(id, score)| RankedItem { id, score })
        .collect();

    // Sort by score descending
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    results.truncate(limit);
    results
}

/// Maximal Marginal Relevance reranking for dense-vector candidates.
///
/// `lambda` is clamped to [0.0, 1.0]. Higher values favor original relevance;
/// lower values favor diversity against the already selected candidates.
pub fn mmr_rerank(candidates: &[VectorRankedItem], limit: usize, lambda: f64) -> Vec<RankedItem> {
    let borrowed: Vec<VectorRankedItemRef<'_>> = candidates
        .iter()
        .map(|candidate| VectorRankedItemRef {
            id: candidate.id,
            score: candidate.score,
            vector: &candidate.vector,
        })
        .collect();
    mmr_rerank_refs(&borrowed, limit, lambda)
}

fn mmr_rerank_refs(
    candidates: &[VectorRankedItemRef<'_>],
    limit: usize,
    lambda: f64,
) -> Vec<RankedItem> {
    if candidates.is_empty() || limit == 0 {
        return Vec::new();
    }

    let lambda = lambda.clamp(0.0, 1.0);
    let mut remaining: Vec<usize> = (0..candidates.len()).collect();
    remaining.sort_by(|a, b| compare_ranked(candidates[*a].score, candidates[*b].score));

    let take = limit.min(candidates.len());
    let mut selected: Vec<usize> = Vec::with_capacity(take);
    let mut output = Vec::with_capacity(take);

    let first = remaining.remove(0);
    selected.push(first);
    output.push(RankedItem {
        id: candidates[first].id,
        score: candidates[first].score,
    });

    while output.len() < take && !remaining.is_empty() {
        let mut best_pos = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_relevance = f64::NEG_INFINITY;

        for (pos, idx) in remaining.iter().copied().enumerate() {
            let relevance = candidates[idx].score;
            let diversity = selected
                .iter()
                .copied()
                .map(|selected_idx| {
                    cosine_similarity(candidates[idx].vector, candidates[selected_idx].vector)
                })
                .fold(f64::NEG_INFINITY, f64::max)
                .max(0.0);
            let mmr_score = lambda * relevance - (1.0 - lambda) * diversity;

            let better_score = mmr_score > best_score;
            let tied_score = (mmr_score - best_score).abs() <= f64::EPSILON;
            if better_score || (tied_score && relevance > best_relevance) {
                best_pos = pos;
                best_score = mmr_score;
                best_relevance = relevance;
            }
        }

        let idx = remaining.remove(best_pos);
        selected.push(idx);
        output.push(RankedItem {
            id: candidates[idx].id,
            score: best_score,
        });
    }

    output
}

/// Apply MMR to the vector-backed subset of a fused list and append any
/// non-vector-backed items in their original fused order.
pub fn mmr_rerank_fused(
    fused: &[RankedItem],
    dense_vectors: &HashMap<u128, Vec<f32>>,
    limit: usize,
    lambda: f64,
) -> (Vec<RankedItem>, usize) {
    if fused.is_empty() || limit == 0 {
        return (Vec::new(), 0);
    }

    let vector_candidates: Vec<VectorRankedItemRef<'_>> = fused
        .iter()
        .filter_map(|item| {
            dense_vectors
                .get(&item.id)
                .map(|vector| VectorRankedItemRef {
                    id: item.id,
                    score: item.score,
                    vector,
                })
        })
        .collect();

    if vector_candidates.len() < 2 {
        return (
            fused.iter().take(limit).cloned().collect(),
            vector_candidates.len(),
        );
    }

    let mut reranked = mmr_rerank_refs(&vector_candidates, vector_candidates.len(), lambda);
    let mut seen: HashSet<u128> = reranked.iter().map(|item| item.id).collect();

    for item in fused {
        if seen.insert(item.id) {
            reranked.push(item.clone());
        }
        if reranked.len() >= limit {
            break;
        }
    }

    reranked.truncate(limit);
    (reranked, vector_candidates.len())
}

fn compare_ranked(lhs: f64, rhs: f64) -> std::cmp::Ordering {
    rhs.partial_cmp(&lhs).unwrap_or(std::cmp::Ordering::Equal)
}

fn cosine_similarity(lhs: &[f32], rhs: &[f32]) -> f64 {
    if lhs.len() != rhs.len() || lhs.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f64;
    let mut lhs_norm = 0.0f64;
    let mut rhs_norm = 0.0f64;
    for (a, b) in lhs.iter().zip(rhs.iter()) {
        let a = *a as f64;
        let b = *b as f64;
        dot += a * b;
        lhs_norm += a * a;
        rhs_norm += b * b;
    }

    if lhs_norm == 0.0 || rhs_norm == 0.0 {
        0.0
    } else {
        dot / (lhs_norm.sqrt() * rhs_norm.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_single_list() {
        let list = vec![
            RankedItem { id: 1, score: 0.9 },
            RankedItem { id: 2, score: 0.8 },
            RankedItem { id: 3, score: 0.7 },
        ];
        let result = rrf_fusion(&[list], 10);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].id, 1); // rank 1 → highest RRF score
        assert_eq!(result[1].id, 2);
        assert_eq!(result[2].id, 3);
    }

    #[test]
    fn test_rrf_two_lists_overlap() {
        // Item 2 appears in both lists → gets boosted
        let list1 = vec![
            RankedItem { id: 1, score: 0.9 },
            RankedItem { id: 2, score: 0.8 },
        ];
        let list2 = vec![
            RankedItem { id: 2, score: 0.95 },
            RankedItem { id: 3, score: 0.7 },
        ];
        let result = rrf_fusion(&[list1, list2], 10);

        // Item 2 has score from both lists, should be first
        assert_eq!(result[0].id, 2);
    }

    #[test]
    fn test_rrf_limit() {
        let list = vec![
            RankedItem { id: 1, score: 0.9 },
            RankedItem { id: 2, score: 0.8 },
            RankedItem { id: 3, score: 0.7 },
        ];
        let result = rrf_fusion(&[list], 2);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_rrf_empty() {
        let result = rrf_fusion(&[], 10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_rrf_no_overlap() {
        let list1 = vec![RankedItem { id: 1, score: 0.9 }];
        let list2 = vec![RankedItem { id: 2, score: 0.8 }];
        let result = rrf_fusion(&[list1, list2], 10);
        assert_eq!(result.len(), 2);
        // Both have same RRF score (1/(60+1) each from rank 1)
        // Order is non-deterministic for equal scores but both should be present
    }

    #[test]
    fn test_minmax_normalization() {
        let normalized = normalize_scores(&[1.0, 2.0, 3.0], NormalizationMethod::MinMax);
        assert_eq!(normalized, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_minmax_constant_scores() {
        let normalized = normalize_scores(&[5.0, 5.0], NormalizationMethod::MinMax);
        assert_eq!(normalized, vec![0.5, 0.5]);
    }

    #[test]
    fn test_zscore_normalization() {
        let normalized = normalize_scores(&[1.0, 2.0, 3.0], NormalizationMethod::ZScore);
        let mean = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_mmr_prefers_diverse_second_result_when_lambda_is_low() {
        let candidates = vec![
            VectorRankedItem {
                id: 1,
                score: 1.0,
                vector: vec![1.0, 0.0, 0.0],
            },
            VectorRankedItem {
                id: 2,
                score: 0.95,
                vector: vec![0.99, 0.01, 0.0],
            },
            VectorRankedItem {
                id: 3,
                score: 0.8,
                vector: vec![0.0, 1.0, 0.0],
            },
        ];

        let reranked = mmr_rerank(&candidates, 3, 0.1);

        assert_eq!(reranked[0].id, 1);
        assert_eq!(reranked[1].id, 3);
        assert_eq!(reranked[2].id, 2);
    }

    #[test]
    fn test_mmr_rerank_fused_appends_sparse_only_items() {
        let fused = vec![
            RankedItem { id: 1, score: 1.0 },
            RankedItem { id: 2, score: 0.9 },
            RankedItem { id: 3, score: 0.8 },
        ];
        let dense_vectors = HashMap::from([(1, vec![1.0, 0.0]), (2, vec![0.0, 1.0])]);

        let (reranked, candidate_count) = mmr_rerank_fused(&fused, &dense_vectors, 3, 0.5);

        assert_eq!(candidate_count, 2);
        assert_eq!(reranked.len(), 3);
        assert_eq!(reranked[2].id, 3);
    }
}
