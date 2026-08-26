use std::collections::{HashMap, HashSet, VecDeque};

use super::fusion::RankedItem;

/// Bounded Personalized PageRank (PPR) via the Andersen-Chung-Lang forward-push
/// algorithm, used to rank graph neighbors of a seed set for graph-signal fusion
/// in hybrid search.
///
/// Forward push approximates the PPR vector without materializing the full graph:
/// starting from a seed distribution, it repeatedly moves a fraction `alpha` of a
/// node's residual mass into its permanent estimate `p`, and pushes the remainder
/// `(1 - alpha)` evenly across outgoing neighbors as new residual. A node is only
/// pushed once its residual clears `eps`, which bounds total work independent of
/// graph size.
///
/// Reference: Andersen, Chung, Lang 2006 (FOCS) forward-push approximate PPR.

/// Which edge direction to walk when computing PPR over the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PprDirection {
    Out,
    In,
    Both,
}

/// Tunable parameters for bounded forward-push PPR.
#[derive(Debug, Clone)]
pub struct PprParams {
    /// Restart probability: fraction of residual mass converted to a permanent
    /// score estimate on each push.
    pub alpha: f64,
    /// Minimum residual mass required for a node to be pushed.
    pub eps: f64,
    /// Budget on distinct nodes touched (seeded or reached via a push) before
    /// the push loop stops early.
    pub max_pushed_nodes: usize,
    /// Hard cap on the number of push operations before the loop stops early.
    pub max_pushes: usize,
    /// Max number of ranked results to return.
    pub limit: usize,
}

impl Default for PprParams {
    fn default() -> Self {
        PprParams {
            alpha: 0.15,
            eps: 1e-4,
            max_pushed_nodes: 10_000,
            max_pushes: 100_000,
            limit: 100,
        }
    }
}

/// Merge out- and in-neighbor lookups into a single adjacency list per
/// `PprDirection`. `Both` concatenates out-then-in and dedups, keeping the
/// first occurrence of each id.
pub fn merge_directions(dir: PprDirection, out: Vec<u128>, inn: Vec<u128>) -> Vec<u128> {
    match dir {
        PprDirection::Out => out,
        PprDirection::In => inn,
        PprDirection::Both => {
            let mut seen: HashSet<u128> = HashSet::new();
            let mut merged = Vec::with_capacity(out.len() + inn.len());
            for id in out.into_iter().chain(inn) {
                if seen.insert(id) {
                    merged.push(id);
                }
            }
            merged
        }
    }
}

/// Compute a bounded, deterministic approximate Personalized PageRank over a
/// graph reachable through `neighbors`, seeded at `seeds` with per-seed weights.
///
/// `neighbors(node)` returns the adjacency for `node` in whatever direction the
/// caller has chosen (see `merge_directions`); this function treats it as an
/// out-neighborhood and never inspects direction itself, keeping it storage-agnostic.
///
/// Deterministic by construction: nodes are pushed in FIFO order from a work
/// queue seeded in input order, and each node's neighbors are visited in the
/// order the closure returns them. Ties in the output are broken by ascending id.
pub fn personalized_pagerank<F>(
    seeds: &[(u128, f64)],
    mut neighbors: F,
    params: &PprParams,
) -> Vec<RankedItem>
where
    F: FnMut(u128) -> Vec<u128>,
{
    // Dedup seeds, preserving first-occurrence order, summing weights.
    let mut seed_order: Vec<u128> = Vec::new();
    let mut seed_weight: HashMap<u128, f64> = HashMap::new();
    for (id, weight) in seeds {
        if !seed_weight.contains_key(id) {
            seed_order.push(*id);
        }
        *seed_weight.entry(*id).or_insert(0.0) += weight;
    }

    let total: f64 = seed_weight.values().sum();
    if seed_order.is_empty() || total <= 0.0 {
        return Vec::new();
    }

    let mut p: HashMap<u128, f64> = HashMap::new();
    let mut r: HashMap<u128, f64> = HashMap::new();
    let mut touched: HashSet<u128> = HashSet::new();
    let mut queued: HashSet<u128> = HashSet::new();
    let mut queue: VecDeque<u128> = VecDeque::new();
    let mut adjacency: HashMap<u128, Vec<u128>> = HashMap::new();
    let mut push_count: usize = 0;

    for id in &seed_order {
        let weight = seed_weight[id] / total;
        if weight <= 0.0 {
            continue;
        }
        r.insert(*id, weight);
        touched.insert(*id);
        if weight >= params.eps && queued.insert(*id) {
            queue.push_back(*id);
        }
    }

    loop {
        if touched.len() > params.max_pushed_nodes || push_count > params.max_pushes {
            break;
        }
        let v = match queue.pop_front() {
            Some(v) => v,
            None => break,
        };
        queued.remove(&v);

        let residual = r.get(&v).copied().unwrap_or(0.0);
        if residual < params.eps {
            continue;
        }
        push_count += 1;

        *p.entry(v).or_insert(0.0) += params.alpha * residual;
        let spread = (1.0 - params.alpha) * residual;
        r.insert(v, 0.0);

        let out_neighbors = adjacency.entry(v).or_insert_with(|| neighbors(v));
        let degree = out_neighbors.len();

        if degree == 0 {
            *p.entry(v).or_insert(0.0) += spread;
            continue;
        }

        let share = spread / degree as f64;
        for &u in out_neighbors.iter() {
            touched.insert(u);
            let new_residual = r.get(&u).copied().unwrap_or(0.0) + share;
            r.insert(u, new_residual);
            if new_residual >= params.eps && queued.insert(u) {
                queue.push_back(u);
            }
        }
    }

    let mut results: Vec<RankedItem> = p
        .into_iter()
        .filter(|(_, score)| *score > 0.0)
        .map(|(id, score)| RankedItem { id, score })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    results.truncate(params.limit);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn star_graph_hub_ranks_first() {
        // Five leaves each point to a shared hub; the hub itself is dangling.
        let seeds: Vec<(u128, f64)> = (1..=5).map(|id| (id, 1.0)).collect();
        let neighbors = |node: u128| -> Vec<u128> {
            if (1..=5).contains(&node) {
                vec![100]
            } else {
                vec![]
            }
        };
        let params = PprParams::default();
        let results = personalized_pagerank(&seeds, neighbors, &params);

        let hub = results
            .iter()
            .find(|item| item.id == 100)
            .expect("hub present in results");
        assert_eq!(results[0].id, 100, "hub should rank first");
        for leaf_id in 1..=5u128 {
            let leaf_score = results
                .iter()
                .find(|item| item.id == leaf_id)
                .map(|item| item.score)
                .unwrap_or(0.0);
            assert!(hub.score > leaf_score);
        }
    }

    #[test]
    fn chain_proximity_decays() {
        // Chain 1->2->3->4->5, with node 5 draining into an unasserted escape
        // node 6 so the tail of the chain isn't a dangling sink that would
        // absorb all remaining mass in one shot.
        let seeds = vec![(1u128, 1.0)];
        let neighbors = |node: u128| -> Vec<u128> {
            match node {
                1 => vec![2],
                2 => vec![3],
                3 => vec![4],
                4 => vec![5],
                5 => vec![6],
                _ => vec![],
            }
        };
        let params = PprParams::default();
        let results = personalized_pagerank(&seeds, neighbors, &params);
        let score = |id: u128| {
            results
                .iter()
                .find(|item| item.id == id)
                .map(|item| item.score)
                .unwrap_or(0.0)
        };

        let p1 = score(1);
        let p2 = score(2);
        let p3 = score(3);
        let p4 = score(4);
        let p5 = score(5);
        assert!(p1 > p2);
        assert!(p2 > p3);
        assert!(p3 > p4);
        assert!(p4 > p5);
    }

    #[test]
    fn empty_adjacency_returns_seed_mass() {
        let seeds = vec![(1u128, 2.0), (2u128, 3.0), (3u128, 5.0)];
        let neighbors = |_node: u128| -> Vec<u128> { vec![] };
        let params = PprParams::default();
        let results = personalized_pagerank(&seeds, neighbors, &params);

        assert_eq!(results.len(), 3);
        let expected = [(3u128, 0.5), (2u128, 0.3), (1u128, 0.2)];
        for (item, (id, weight)) in results.iter().zip(expected.iter()) {
            assert_eq!(item.id, *id);
            assert!((item.score - weight).abs() < 1e-9);
        }
    }

    #[test]
    fn determinism() {
        let seeds = vec![(1u128, 0.6), (3u128, 0.4)];
        let neighbors = |node: u128| -> Vec<u128> {
            match node {
                1 => vec![2, 3],
                2 => vec![3],
                3 => vec![1],
                _ => vec![],
            }
        };
        let params = PprParams::default();
        let run1 = personalized_pagerank(&seeds, neighbors, &params);
        let run2 = personalized_pagerank(&seeds, neighbors, &params);

        let bits = |items: &[RankedItem]| -> Vec<(u128, u64)> {
            items
                .iter()
                .map(|item| (item.id, item.score.to_bits()))
                .collect()
        };
        assert_eq!(bits(&run1), bits(&run2));
    }

    #[test]
    fn budget_respected() {
        // A 10,000-node chain with a tiny node budget must stop early.
        let seeds = vec![(0u128, 1.0)];
        let neighbors = |node: u128| -> Vec<u128> {
            if node + 1 < 10_000 {
                vec![node + 1]
            } else {
                vec![]
            }
        };
        let params = PprParams {
            max_pushed_nodes: 8,
            ..PprParams::default()
        };
        let results = personalized_pagerank(&seeds, neighbors, &params);
        assert!(results.len() <= 8);
    }

    #[test]
    fn weighted_seeds_bias_ranking() {
        // Two disjoint stars; seed A carries far more weight than seed B.
        let seeds = vec![(1u128, 0.9), (2u128, 0.1)];
        let neighbors = |node: u128| -> Vec<u128> {
            match node {
                1 => vec![100],
                2 => vec![200],
                _ => vec![],
            }
        };
        let params = PprParams::default();
        let results = personalized_pagerank(&seeds, neighbors, &params);
        let score = |id: u128| {
            results
                .iter()
                .find(|item| item.id == id)
                .map(|item| item.score)
                .unwrap_or(0.0)
        };
        assert!(score(100) > score(200));
    }

    #[test]
    fn merge_directions_dedup() {
        let out = vec![1u128, 2, 3];
        let inn = vec![2u128, 4, 1];

        let merged = merge_directions(PprDirection::Both, out.clone(), inn.clone());
        assert_eq!(merged, vec![1, 2, 3, 4]);

        assert_eq!(
            merge_directions(PprDirection::Out, out.clone(), inn.clone()),
            out
        );
        assert_eq!(merge_directions(PprDirection::In, out, inn.clone()), inn);
    }
}
