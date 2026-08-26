use crate::helix_gateway::router::router::HelixRouter;

use super::collections;
use super::graph;
use super::health;
use super::ingest;
use super::metrics;
use super::qdrant;
use super::raft;

/// Register all CE REST API routes on the router.
pub fn register_api_routes(router: &mut HelixRouter) {
    // Health — also register GET / so Qdrant-compat wait scripts work.
    // `/readyz` and `/livez` mirror Kubernetes probe conventions so
    // misconfigured probes or sidecars don't flood logs with 404s.
    router.add_route("GET", "/", health::handle_health);
    router.add_route("GET", "/health", health::handle_health);
    router.add_route("GET", "/healthz", health::handle_health);
    router.add_route("GET", "/livez", health::handle_health);
    router.add_route("GET", "/ready", health::handle_ready);
    router.add_route("GET", "/readyz", health::handle_ready);
    // Prometheus metrics scrape endpoint.
    router.add_route("GET", "/metrics", metrics::handle_metrics);
    router.add_route("GET", "/_raft/status", raft::handle_status);
    router.add_route("POST", "/_raft/message", raft::handle_message);
    router.add_route("POST", "/_raft/propose", raft::handle_propose);

    // Collections
    router.add_route("POST", "/v1/collections/create", collections::handle_create);
    router.add_route("POST", "/v1/collections/drop", collections::handle_drop);
    router.add_route("POST", "/v1/collections/list", collections::handle_list);
    router.add_route("POST", "/v1/collections/stats", collections::handle_stats);
    router.add_route(
        "POST",
        "/v1/collections/storage_bytes",
        collections::handle_storage_bytes,
    );
    router.add_route(
        "POST",
        "/v1/collections/recount",
        collections::handle_recount,
    );
    router.add_route(
        "POST",
        "/v1/collections/gc_payload_index",
        collections::handle_gc_payload_index,
    );
    router.add_route(
        "POST",
        "/v1/collections/maintenance",
        collections::handle_maintenance,
    );

    // Ingest
    router.add_route("POST", "/v1/ingest/nodes", ingest::handle_ingest_nodes);
    router.add_route("POST", "/v1/ingest/edges", ingest::handle_ingest_edges);
    router.add_pattern_route(
        "POST",
        "/v1/collections/{name}/ingest/stream",
        ingest::handle_ingest_stream,
    );

    // Graph queries
    router.add_route("POST", "/v1/graph/callers", graph::handle_callers);
    router.add_route("POST", "/v1/graph/callees", graph::handle_callees);
    router.add_route("POST", "/v1/graph/importers", graph::handle_importers);
    router.add_route("POST", "/v1/graph/definition", graph::handle_definition);
    router.add_route(
        "POST",
        "/v1/graph/transitive_callers",
        graph::handle_transitive_callers,
    );
    router.add_route(
        "POST",
        "/v1/graph/transitive_callees",
        graph::handle_transitive_callees,
    );
    router.add_route("POST", "/v1/graph/impact", graph::handle_impact);
    router.add_route("POST", "/v1/graph/dependencies", graph::handle_dependencies);
    router.add_route("POST", "/v1/graph/cycles", graph::handle_cycles);
    router.add_route("POST", "/v1/graph/subclasses", graph::handle_subclasses);
    router.add_route("POST", "/v1/graph/base_classes", graph::handle_base_classes);

    // Graph algorithms
    router.add_route("POST", "/v1/graph/pagerank", graph::handle_pagerank);
    router.add_route("POST", "/v1/graph/communities", graph::handle_communities);
    router.add_route(
        "POST",
        "/v1/graph/shortest_path",
        graph::handle_shortest_path,
    );
    router.add_route("POST", "/v1/graph/jaccard", graph::handle_jaccard);
    router.add_route("POST", "/v1/graph/subgraph", graph::handle_subgraph);
    router.add_route("POST", "/v1/graph/distinct", graph::handle_distinct_values);

    // Graph mutations
    router.add_route(
        "POST",
        "/v1/graph/delete_by_path",
        graph::handle_delete_by_path,
    );
    router.add_route(
        "POST",
        "/v1/graph/delete_by_paths",
        graph::handle_delete_by_paths,
    );
    router.add_route(
        "POST",
        "/v1/graph/backfill_edge_path_index",
        graph::handle_backfill_edge_path_index,
    );
    router.add_route(
        "POST",
        "/v1/graph/backfill_adjacency_from_points",
        graph::handle_backfill_adjacency_from_points,
    );
    router.add_route(
        "POST",
        "/v1/graph/rebuild_adjacency_for_paths",
        graph::handle_rebuild_adjacency_for_paths,
    );

    // Qdrant-compatible REST API (pattern routes with path params)
    router.add_pattern_route(
        "PUT",
        "/collections/{name}",
        qdrant::handle_create_collection,
    );
    router.add_pattern_route("GET", "/collections/{name}", qdrant::handle_get_collection);
    router.add_pattern_route(
        "PATCH",
        "/collections/{name}",
        qdrant::handle_update_collection,
    );
    router.add_pattern_route(
        "DELETE",
        "/collections/{name}",
        qdrant::handle_delete_collection,
    );
    router.add_route("GET", "/collections", qdrant::handle_list_collections);
    router.add_pattern_route(
        "PUT",
        "/collections/{name}/points",
        qdrant::handle_upsert_points,
    );
    router.add_pattern_route(
        "POST",
        "/collections/{name}/points/search",
        qdrant::handle_search_points,
    );
    router.add_pattern_route(
        "POST",
        "/collections/{name}/points/scroll",
        qdrant::handle_scroll_points,
    );
    router.add_pattern_route(
        "POST",
        "/v1/collections/{name}/points/scan",
        qdrant::handle_scan_points,
    );
    router.add_pattern_route(
        "POST",
        "/collections/{name}/points/delete",
        qdrant::handle_delete_points,
    );
    router.add_pattern_route(
        "POST",
        "/collections/{name}/points/query",
        qdrant::handle_query_points,
    );
    router.add_pattern_route(
        "POST",
        "/collections/{name}/points/hybrid_query",
        qdrant::handle_hybrid_query_points,
    );
    router.add_pattern_route(
        "PUT",
        "/collections/{name}/index",
        qdrant::handle_create_index,
    );
    router.add_pattern_route(
        "DELETE",
        "/collections/{name}/index/{field_name}",
        qdrant::handle_delete_index,
    );
    router.add_pattern_route(
        "GET",
        "/collections/{name}/snapshots",
        qdrant::handle_list_snapshots,
    );
    router.add_pattern_route(
        "POST",
        "/collections/{name}/snapshots",
        qdrant::handle_create_snapshot,
    );
    router.add_pattern_route(
        "PUT",
        "/collections/{name}/snapshots/recover",
        qdrant::handle_recover_snapshot,
    );

    // Point operations: payload, count, retrieve, exists
    router.add_pattern_route(
        "POST",
        "/collections/{name}/points/payload",
        qdrant::handle_set_payload,
    );
    router.add_pattern_route(
        "POST",
        "/collections/{name}/points/count",
        qdrant::handle_count_points,
    );
    router.add_pattern_route("POST", "/collections/{name}/facet", qdrant::handle_facet);
    router.add_pattern_route(
        "POST",
        "/collections/{name}/facet/count",
        qdrant::handle_facet,
    );
    router.add_pattern_route(
        "POST",
        "/collections/{name}/points",
        qdrant::handle_get_points,
    );
    router.add_pattern_route(
        "GET",
        "/collections/{name}/exists",
        qdrant::handle_collection_exists,
    );

    // Collection aliases
    router.add_route("POST", "/collections/aliases", qdrant::handle_alias_actions);
    router.add_route("GET", "/aliases", qdrant::handle_list_aliases);
    router.add_pattern_route(
        "GET",
        "/collections/{name}/aliases",
        qdrant::handle_collection_aliases,
    );
}
