use super::graph_signal::{personalized_pagerank, personalized_pagerank_filtered, PprParams};

#[test]
fn adjacency_reads_stop_when_push_budget_is_exhausted() {
    // Given: a chain whose residual stays above epsilon beyond each budget.
    let budgets = [0, 1, 8];
    let mut observations = Vec::new();
    for max_pushes in budgets {
        let params = PprParams {
            alpha: 0.15,
            eps: 1e-4,
            max_pushed_nodes: 100,
            max_pushes,
            limit: 100,
        };
        let mut reads = Vec::new();

        // When: the graph channel walks with a hard push budget.
        let results = personalized_pagerank(
            &[(0, 1.0)],
            |id| {
                reads.push(id);
                vec![id + 1]
            },
            &params,
        );
        observations.push((
            reads,
            results.into_iter().map(|item| item.id).collect::<Vec<_>>(),
        ));
    }

    // Then: neither adjacency reads nor ranked nodes exceed the budget.
    println!("push budgets {budgets:?}: {observations:?}");
    assert_eq!(
        observations,
        vec![
            (vec![], vec![]),
            (vec![0], vec![0]),
            ((0..8).collect(), (0..8).collect()),
        ]
    );
}

#[test]
fn cached_adjacency_still_consumes_push_budget() {
    // Given: a self-loop that revisits the same cached adjacency.
    let params = PprParams {
        alpha: 0.5,
        eps: 1e-4,
        max_pushed_nodes: 100,
        max_pushes: 2,
        limit: 100,
    };
    let mut reads = Vec::new();

    // When: two pushes move half the residual into the seed's score each time.
    let results = personalized_pagerank_filtered(
        &[(7, 1.0)],
        |id| {
            reads.push(id);
            vec![id]
        },
        |_| true,
        &params,
    );

    // Then: the score is 1/2 + 1/4, with no third push adding another 1/8.
    println!("two-push self-loop result: {results:?}, adjacency reads: {reads:?}");
    assert_eq!(reads, vec![7]);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 7);
    assert!((results[0].score - 0.75).abs() < 1e-12);
}
