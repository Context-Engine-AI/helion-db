use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::Hasher;
use std::ops::Bound;
use std::sync::{OnceLock, RwLock};
use std::time::{Duration, Instant};

use crate::helix_engine::storage_core::backend::{
    BackendError, BackendKind, KeyRange, Namespace, StorageBackend,
};
use crate::helix_engine::storage_core::backend_any::{AnyBackend, AnyRead};
use crate::helix_engine::storage_core::collection_manager::reader_refresh_ttl_ms;
use crate::helix_engine::storage_core::filters::{
    Condition, FieldCondition, Filter, MatchCondition, RangeCondition,
};
use crate::helix_engine::storage_core::metadata::PayloadIndexSchema;
use crate::helix_engine::storage_core::replication::{ReplicatedMutation, ReplicatedPoint};
use crate::helix_engine::storage_core::storage_core::HelixGraphStorage;
use crate::helix_engine::storage_core::storage_methods::StorageMethods;
use crate::helix_engine::types::{graph_error_from_backend_error, GraphError};
use crate::helix_engine::vector_core::fusion::{
    mmr_rerank_fused, rrf_fusion, rrf_fusion_with_k, RankedItem,
};
use crate::helix_engine::vector_core::graph_signal::{
    merge_directions, personalized_pagerank_filtered, PprDirection, PprParams,
};
use crate::helix_engine::vector_core::hnsw::HNSW;
use crate::helix_engine::vector_core::named_vectors::{
    DenseVectorSegmentRole, DistanceMetric, NamedVectorConfig,
};
use crate::helix_engine::vector_core::sparse::{SparseModifier, SparseVector, SparseVectorConfig};
use crate::helix_engine::vector_core::spindle::{SpindleConfig, SpindleMode};
use crate::helix_engine::vector_core::vector::HVector;
use crate::helix_engine::vector_core::vector_core::HnswOverrides;
use crate::helix_gateway::router::router::HandlerInput;
use crate::protocol::items::{Node, SerializedNode};
use crate::protocol::label_hash::hash_label;
use crate::protocol::response::Response;
use crate::protocol::value::Value;

#[cfg(test)]
#[path = "qdrant_payload_cache_handler_tests.rs"]
mod qdrant_payload_cache_handler_tests;
#[cfg(test)]
#[path = "qdrant_payload_cache_tests.rs"]
mod qdrant_payload_cache_tests;

// ─── Helpers ───

fn json_ok<T: Serialize>(response: &mut Response, body: &T) -> Result<(), GraphError> {
    response.status = 200;
    response.body = sonic_rs::to_vec(&QdrantResponse {
        status: "ok",
        result: body,
    })?;
    response
        .headers
        .insert("Content-Type".to_string(), "application/json".to_string());
    Ok(())
}

fn json_created<T: Serialize>(response: &mut Response, body: &T) -> Result<(), GraphError> {
    response.status = 200;
    response.body = sonic_rs::to_vec(&QdrantResponse {
        status: "ok",
        result: body,
    })?;
    response
        .headers
        .insert("Content-Type".to_string(), "application/json".to_string());
    Ok(())
}

fn json_err(response: &mut Response, status: u16, msg: &str) -> Result<(), GraphError> {
    response.status = status;
    response.body = sonic_rs::to_vec(&sonic_rs::json!({
        "status": {"error": msg},
        "result": null,
    }))?;
    response
        .headers
        .insert("Content-Type".to_string(), "application/json".to_string());
    Ok(())
}

fn backend_label(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::Lmdb => "lmdb",
        BackendKind::Lsm => "lsm",
    }
}

fn query_stage_elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn record_qdrant_query_stage(
    collection: &str,
    endpoint: &'static str,
    backend: &'static str,
    query_kind: &'static str,
    stage: &'static str,
    started: Instant,
) {
    metrics::histogram!(
        "helix_qdrant_query_stage_ms",
        "collection" => collection.to_string(),
        "endpoint" => endpoint,
        "backend" => backend,
        "query_kind" => query_kind,
        "stage" => stage,
    )
    .record(query_stage_elapsed_ms(started));
}

fn record_qdrant_query_outcome(
    collection: &str,
    endpoint: &'static str,
    backend: &'static str,
    query_kind: &'static str,
    outcome: &'static str,
    started: Instant,
) {
    let elapsed_ms = query_stage_elapsed_ms(started);
    metrics::histogram!(
        "helix_qdrant_query_total_ms",
        "collection" => collection.to_string(),
        "endpoint" => endpoint,
        "backend" => backend,
        "query_kind" => query_kind,
        "outcome" => outcome,
    )
    .record(elapsed_ms);
    metrics::counter!(
        "helix_qdrant_query_total",
        "collection" => collection.to_string(),
        "endpoint" => endpoint,
        "backend" => backend,
        "query_kind" => query_kind,
        "outcome" => outcome,
    )
    .increment(1);
}

fn record_qdrant_query_count(
    collection: &str,
    endpoint: &'static str,
    backend: &'static str,
    query_kind: &'static str,
    measurement: &'static str,
    value: usize,
) {
    metrics::histogram!(
        "helix_qdrant_query_items",
        "collection" => collection.to_string(),
        "endpoint" => endpoint,
        "backend" => backend,
        "query_kind" => query_kind,
        "measurement" => measurement,
    )
    .record(value as f64);
}

fn record_qdrant_payload_cache_stage(
    collection: &str,
    backend: &'static str,
    outcome: &'static str,
    started: Instant,
) {
    metrics::histogram!(
        "helix_qdrant_payload_cache_ms",
        "collection" => collection.to_string(),
        "backend" => backend,
        "outcome" => outcome,
    )
    .record(query_stage_elapsed_ms(started));
}

fn record_qdrant_payload_cache_count(
    collection: &str,
    backend: &'static str,
    measurement: &'static str,
    value: usize,
) {
    metrics::histogram!(
        "helix_qdrant_payload_cache_items",
        "collection" => collection.to_string(),
        "backend" => backend,
        "measurement" => measurement,
    )
    .record(value as f64);
}

fn query_kind_label(kind: &QueryKind) -> &'static str {
    match kind {
        QueryKind::DenseVector(_) => "dense",
        QueryKind::SparseVector(_) => "sparse",
        QueryKind::Fusion(_) => "fusion",
    }
}

fn get_collection_name(input: &HandlerInput) -> Option<String> {
    input.path_params.get("name").cloned()
}

fn get_field_name(input: &HandlerInput) -> Option<String> {
    input.path_params.get("field_name").cloned()
}

fn parse_payload_index_schema(schema: &str) -> Option<PayloadIndexSchema> {
    match schema.trim().to_lowercase().as_str() {
        "keyword" => Some(PayloadIndexSchema::Keyword),
        "integer" | "int" => Some(PayloadIndexSchema::Integer),
        "float" | "double" => Some(PayloadIndexSchema::Float),
        _ => None,
    }
}

fn payload_index_schema_json(schema: &PayloadIndexSchema) -> serde_json::Value {
    let data_type = match schema {
        PayloadIndexSchema::Keyword => "keyword",
        PayloadIndexSchema::Integer => "integer",
        PayloadIndexSchema::Float => "float",
    };
    serde_json::json!({ "data_type": data_type, "points": 0 })
}

fn indexed_condition_ids(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    txn: &heed3::RoTxn,
    condition: &FieldCondition,
) -> Result<Option<HashSet<u128>>, GraphError> {
    let Some(schema) = storage.has_payload_index(&condition.key) else {
        return Ok(None);
    };

    let ids = match schema {
        PayloadIndexSchema::Keyword => match &condition.match_cond {
            Some(MatchCondition::Value(value)) => {
                let value = json_to_value(&value.value);
                storage.get_nodes_by_payload_value(txn, &condition.key, &value)?
            }
            Some(MatchCondition::Any(values)) => {
                let mut ids = HashSet::new();
                for value in &values.any {
                    let value = json_to_value(value);
                    ids.extend(storage.get_nodes_by_payload_value(txn, &condition.key, &value)?);
                }
                return Ok(Some(ids));
            }
            Some(MatchCondition::Text(value)) => {
                let Some(ids) = storage.get_nodes_by_payload_text(
                    txn,
                    &condition.key,
                    &value.text,
                    text_candidate_walk_max(),
                )?
                else {
                    return Ok(None);
                };
                ids
            }
            _ => return Ok(None),
        },
        PayloadIndexSchema::Integer | PayloadIndexSchema::Float => {
            let Some(range) = &condition.range else {
                return Ok(None);
            };
            storage.get_nodes_by_payload_range(
                txn,
                &condition.key,
                range_lower_bound(range),
                range_upper_bound(range),
            )?
        }
    };

    Ok(Some(ids.into_iter().collect()))
}

fn indexed_condition_ids_be(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    condition: &FieldCondition,
) -> Result<Option<HashSet<u128>>, GraphError> {
    let Some(schema) = storage.has_payload_index(&condition.key) else {
        return Ok(None);
    };

    let ids = match schema {
        PayloadIndexSchema::Keyword => match &condition.match_cond {
            Some(MatchCondition::Value(value)) => {
                let value = json_to_value(&value.value);
                storage.get_nodes_by_payload_value_be(r, &condition.key, &value)?
            }
            Some(MatchCondition::Any(values)) => {
                let mut ids = HashSet::new();
                for value in &values.any {
                    let value = json_to_value(value);
                    ids.extend(storage.get_nodes_by_payload_value_be(r, &condition.key, &value)?);
                }
                return Ok(Some(ids));
            }
            Some(MatchCondition::Text(value)) => {
                let Some(ids) = storage.get_nodes_by_payload_text_be(
                    r,
                    &condition.key,
                    &value.text,
                    text_candidate_walk_max(),
                )?
                else {
                    return Ok(None);
                };
                ids
            }
            _ => return Ok(None),
        },
        PayloadIndexSchema::Integer | PayloadIndexSchema::Float => {
            let Some(range) = &condition.range else {
                return Ok(None);
            };
            storage.get_nodes_by_payload_range_be(
                r,
                &condition.key,
                range_lower_bound(range),
                range_upper_bound(range),
            )?
        }
    };

    Ok(Some(ids.into_iter().collect()))
}

fn indexed_filter_candidates(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    txn: &heed3::RoTxn,
    filter: &Filter,
) -> Result<Option<HashSet<u128>>, GraphError> {
    let mut candidates: Option<HashSet<u128>> = None;

    // Text conditions are resolved by walking index terms, so materialize
    // the cheap keyword/range/nested conditions first and skip the walk when
    // they are already selective: the candidate set stays a superset and the
    // per-point recheck applies the text predicate.
    let mut text_conditions = Vec::new();
    for condition in &filter.must {
        if is_text_field_condition(condition) {
            text_conditions.push(condition);
            continue;
        }
        let Some(ids) = indexed_condition_candidates(storage, txn, condition)? else {
            continue;
        };
        candidates = intersect_candidate_sets(candidates, ids);
    }
    for condition in text_conditions {
        if text_walk_skippable(candidates.as_ref()) {
            continue;
        }
        let Some(ids) = indexed_condition_candidates(storage, txn, condition)? else {
            continue;
        };
        candidates = intersect_candidate_sets(candidates, ids);
    }

    // A single unindexable branch discards the whole should set, so check
    // every branch before walking any (text walks are not free).
    if !filter.should.is_empty()
        && filter
            .should
            .iter()
            .all(|condition| condition_may_yield_candidates(storage, condition))
    {
        let mut should_candidates: Option<HashSet<u128>> = None;

        for condition in &filter.should {
            let Some(ids) = indexed_condition_candidates(storage, txn, condition)? else {
                should_candidates = None;
                break;
            };
            should_candidates = union_candidate_sets(should_candidates, ids);
        }

        if let Some(ids) = should_candidates {
            candidates = intersect_candidate_sets(candidates, ids);
        }
    }

    // must_not subtraction on indexed fields.
    //
    // Before this change, any filter with a `must_not` clause on an
    // indexed field still ran through the indexed-candidate path but
    // the must_not predicate was only checked during post-scan filter
    // application (scan loop `filter.matches_point`). For a common CE
    // pattern like `must=[repo=X] must_not=[branch=feature/abc]` that
    // meant scanning every point in the positive set just to reject
    // a small excluded subset.
    //
    // Here we compute the must_not candidate set the same way we
    // compute `must` (per-condition ids, union), then subtract from
    // `candidates` if everything in must_not was indexable. Any
    // must_not clause that isn't indexed forces us to skip the
    // subtraction — we can't safely exclude on a predicate we can't
    // evaluate in the planner, so we leave the superset for the scan
    // loop to filter. Correctness is preserved either way.
    //
    // We only subtract when `candidates` is already Some (we have a
    // positive candidate set to prune). Pure must_not-only queries
    // still fall through to the primary-scan plan; deriving a full
    // "all indexed points in collection" set for that case costs a
    // full nodes_db key iteration and is deferred until we see a
    // workload that needs it.
    if !filter.must_not.is_empty() && candidates.is_some() {
        let mut must_not_candidates: Option<HashSet<u128>> = None;
        let mut any_unindexed = false;
        for condition in &filter.must_not {
            // A nested filter's candidate set may be a superset (unindexed
            // inner clauses are skipped); subtracting a superset would drop
            // points that do not actually match the exclusion.
            if !indexed_condition_candidates_are_exact(storage, condition) {
                any_unindexed = true;
                break;
            }
            match indexed_condition_candidates(storage, txn, condition)? {
                Some(ids) => {
                    must_not_candidates = union_candidate_sets(must_not_candidates, ids);
                }
                None => {
                    any_unindexed = true;
                    break;
                }
            }
        }
        if !any_unindexed {
            if let (Some(cands), Some(excludes)) = (candidates.as_mut(), must_not_candidates) {
                cands.retain(|id| !excludes.contains(id));
            }
        }
    }

    Ok(candidates)
}

fn indexed_filter_candidates_be(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    filter: &Filter,
) -> Result<Option<HashSet<u128>>, GraphError> {
    let mut candidates: Option<HashSet<u128>> = None;

    // Text conditions are resolved by walking index terms, so materialize
    // the cheap keyword/range/nested conditions first and skip the walk when
    // they are already selective: the candidate set stays a superset and the
    // per-point recheck applies the text predicate.
    let mut text_conditions = Vec::new();
    for condition in &filter.must {
        if is_text_field_condition(condition) {
            text_conditions.push(condition);
            continue;
        }
        let Some(ids) = indexed_condition_candidates_be(storage, r, condition)? else {
            continue;
        };
        candidates = intersect_candidate_sets(candidates, ids);
    }
    for condition in text_conditions {
        if text_walk_skippable(candidates.as_ref()) {
            continue;
        }
        let Some(ids) = indexed_condition_candidates_be(storage, r, condition)? else {
            continue;
        };
        candidates = intersect_candidate_sets(candidates, ids);
    }

    // A single unindexable branch discards the whole should set, so check
    // every branch before walking any (text walks are not free).
    if !filter.should.is_empty()
        && filter
            .should
            .iter()
            .all(|condition| condition_may_yield_candidates(storage, condition))
    {
        let mut should_candidates: Option<HashSet<u128>> = None;

        for condition in &filter.should {
            let Some(ids) = indexed_condition_candidates_be(storage, r, condition)? else {
                should_candidates = None;
                break;
            };
            should_candidates = union_candidate_sets(should_candidates, ids);
        }

        if let Some(ids) = should_candidates {
            candidates = intersect_candidate_sets(candidates, ids);
        }
    }

    if !filter.must_not.is_empty() && candidates.is_some() {
        let mut must_not_candidates: Option<HashSet<u128>> = None;
        let mut any_unindexed = false;
        for condition in &filter.must_not {
            // A nested filter's candidate set may be a superset (unindexed
            // inner clauses are skipped); subtracting a superset would drop
            // points that do not actually match the exclusion.
            if !indexed_condition_candidates_are_exact(storage, condition) {
                any_unindexed = true;
                break;
            }
            match indexed_condition_candidates_be(storage, r, condition)? {
                Some(ids) => {
                    must_not_candidates = union_candidate_sets(must_not_candidates, ids);
                }
                None => {
                    any_unindexed = true;
                    break;
                }
            }
        }
        if !any_unindexed {
            if let (Some(cands), Some(excludes)) = (candidates.as_mut(), must_not_candidates) {
                cands.retain(|id| !excludes.contains(id));
            }
        }
    }

    Ok(candidates)
}

/// Entries a single `match.text` index-term walk may visit before the planner
/// gives up on it (the condition is then treated as unindexed and the scan
/// recheck applies it). `0` disables the cap.
fn text_candidate_walk_max() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| env_usize_or("HELIX_TEXT_CANDIDATE_WALK_MAX", 200_000))
}

/// When the non-text `must` conditions already narrowed candidates to at most
/// this many ids, a `must` text condition is not walked at all.
fn text_candidate_skip_max() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| env_usize_or("HELIX_TEXT_CANDIDATE_SKIP_MAX", 50_000))
}

fn is_text_field_condition(condition: &Condition) -> bool {
    matches!(
        condition,
        Condition::Field(field) if matches!(field.match_cond, Some(MatchCondition::Text(_)))
    )
}

fn text_walk_skippable(candidates: Option<&HashSet<u128>>) -> bool {
    let skip = candidates.is_some_and(|ids| ids.len() <= text_candidate_skip_max());
    if skip {
        metrics::counter!("helix_payload_text_walk_total", "outcome" => "skipped_selective")
            .increment(1);
    }
    skip
}

/// Cheap, read-free mirror of the planner: could `condition` produce an
/// indexed candidate set? Used to avoid walking `should` branches whose
/// siblings would discard the set anyway.
fn condition_may_yield_candidates(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    condition: &Condition,
) -> bool {
    match condition {
        Condition::Field(field) => indexed_field_condition_supported(storage, field),
        Condition::Nested(filter) => {
            filter
                .must
                .iter()
                .any(|condition| condition_may_yield_candidates(storage, condition))
                || (!filter.should.is_empty()
                    && filter
                        .should
                        .iter()
                        .all(|condition| condition_may_yield_candidates(storage, condition)))
        }
        Condition::HasId(_) | Condition::IsEmpty(_) | Condition::IsNull(_) => false,
    }
}

fn indexed_field_condition_supported(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    condition: &FieldCondition,
) -> bool {
    let Some(schema) = storage.has_payload_index(&condition.key) else {
        return false;
    };

    match schema {
        PayloadIndexSchema::Keyword => matches!(
            &condition.match_cond,
            Some(MatchCondition::Value(_))
                | Some(MatchCondition::Any(_))
                | Some(MatchCondition::Text(_))
        ),
        PayloadIndexSchema::Integer | PayloadIndexSchema::Float => condition.range.is_some(),
    }
}

fn indexed_condition_candidates_are_exact(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    condition: &Condition,
) -> bool {
    match condition {
        // Text candidates are resolved by a substring walk over index terms
        // (historically they were only the exact keyword hits — a subset).
        // Keep them out of must_not subtraction; the scan recheck applies the
        // exclusion, which is always correct.
        Condition::Field(field) => {
            indexed_field_condition_supported(storage, field)
                && !matches!(field.match_cond, Some(MatchCondition::Text(_)))
        }
        Condition::Nested(filter) => indexed_filter_candidates_are_exact(storage, filter),
        Condition::HasId(_) | Condition::IsEmpty(_) | Condition::IsNull(_) => false,
    }
}

fn indexed_filter_candidates_are_exact(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    filter: &Filter,
) -> bool {
    filter
        .must
        .iter()
        .all(|condition| indexed_condition_candidates_are_exact(storage, condition))
        && filter
            .must_not
            .iter()
            .all(|condition| indexed_condition_candidates_are_exact(storage, condition))
        && filter
            .should
            .iter()
            .all(|condition| indexed_condition_candidates_are_exact(storage, condition))
        // Candidate derivation ignores `min_should`, so the set is a superset.
        && filter.min_should.is_none()
}

fn indexed_condition_candidates(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    txn: &heed3::RoTxn,
    condition: &Condition,
) -> Result<Option<HashSet<u128>>, GraphError> {
    match condition {
        Condition::Field(field) => indexed_condition_ids(storage, txn, field),
        Condition::Nested(filter) => indexed_filter_candidates(storage, txn, filter),
        Condition::HasId(_) | Condition::IsEmpty(_) | Condition::IsNull(_) => Ok(None),
    }
}

fn indexed_condition_candidates_be(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    condition: &Condition,
) -> Result<Option<HashSet<u128>>, GraphError> {
    match condition {
        Condition::Field(field) => indexed_condition_ids_be(storage, r, field),
        Condition::Nested(filter) => indexed_filter_candidates_be(storage, r, filter),
        Condition::HasId(_) | Condition::IsEmpty(_) | Condition::IsNull(_) => Ok(None),
    }
}

fn intersect_candidate_sets(
    existing: Option<HashSet<u128>>,
    ids: HashSet<u128>,
) -> Option<HashSet<u128>> {
    Some(match existing {
        Some(existing) => existing.intersection(&ids).copied().collect(),
        None => ids,
    })
}

fn union_candidate_sets(
    existing: Option<HashSet<u128>>,
    ids: HashSet<u128>,
) -> Option<HashSet<u128>> {
    Some(match existing {
        Some(mut existing) => {
            existing.extend(ids);
            existing
        }
        None => ids,
    })
}

/// Kill switch for the pre-materialization filter cardinality probe.
/// `HELIX_FILTER_COUNT_PROBE_ENABLED=0|false` restores the old
/// materialize-always behavior.
fn filter_count_probe_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HELIX_FILTER_COUNT_PROBE_ENABLED")
            .ok()
            .map(|v| !matches!(v.trim(), "0" | "false" | "FALSE" | "no"))
            .unwrap_or(true)
    })
}

/// Largest candidate-set size a `/query`-style dense search keeps. A payload
/// index helps vector search only when it is selective. For broad filters
/// like language=python on code collections, building and checking a 50k+ id
/// HashSet is slower than letting HNSW search normally and applying the same
/// filter predicate to candidate points. Keep the index gate for truly
/// selective filters where it preserves quality and reduces work.
/// (`count > total/4 || count > 20_000` collapses to `count > cap`.)
fn vector_query_broad_candidate_cap(total: usize) -> usize {
    (total / 4).min(20_000)
}

fn vector_query_candidates_too_broad(candidate_count: usize, total: usize) -> bool {
    total > 0 && candidate_count > vector_query_broad_candidate_cap(total)
}

/// Count-only mirror of `indexed_condition_ids_be` for the cardinality probe.
/// Returns `None` when the condition cannot be probed cheaply (no ready
/// payload index, or an unsupported match kind) — the planner treats that as
/// unknown cardinality. Counts saturate at `cap + 1`, meaning "exceeds cap".
fn indexed_condition_count_be(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    condition: &FieldCondition,
    cap: usize,
) -> Result<Option<usize>, GraphError> {
    let Some(schema) = storage.has_payload_index(&condition.key) else {
        return Ok(None);
    };

    let count = match schema {
        PayloadIndexSchema::Keyword => match &condition.match_cond {
            Some(MatchCondition::Value(value)) => {
                let value = json_to_value(&value.value);
                storage.count_nodes_by_payload_value_be(r, &condition.key, &value, cap)?
            }
            Some(MatchCondition::Any(values)) => {
                let mut count = 0usize;
                for value in &values.any {
                    if count > cap {
                        break;
                    }
                    let value = json_to_value(value);
                    count = count.saturating_add(storage.count_nodes_by_payload_value_be(
                        r,
                        &condition.key,
                        &value,
                        cap,
                    )?);
                }
                count
            }
            Some(MatchCondition::Text(value)) => {
                // The text reader walks every index term for substring hits;
                // the exact-keyword count is only a cheap LOWER bound. That is
                // still safe for the too-broad skip (lower bound > cap implies
                // the true count > cap); an underestimate only forgoes a skip.
                // Abstain when there is no exact hit to count.
                let count = storage.count_nodes_by_payload_value_be(
                    r,
                    &condition.key,
                    &Value::String(value.text.clone()),
                    cap,
                )?;
                if count == 0 {
                    return Ok(None);
                }
                count
            }
            _ => return Ok(None),
        },
        PayloadIndexSchema::Integer | PayloadIndexSchema::Float => {
            let Some(range) = &condition.range else {
                return Ok(None);
            };
            storage.count_nodes_by_payload_range_be(
                r,
                &condition.key,
                range_lower_bound(range),
                range_upper_bound(range),
                cap,
            )?
        }
    };

    Ok(Some(count.min(cap.saturating_add(1))))
}

/// Upper-bound cardinality estimate for the indexed candidate set a filter
/// would materialize: the minimum count across probeable `must` conditions
/// (each counted with early abort, saturating at `cap + 1`). `None` when no
/// must condition is probeable.
///
/// Deliberate trade-off: min-over-must-counts is an upper bound of the
/// intersection. When every must condition is individually broad but the
/// intersection is tiny, the probe skips materialization and the search runs
/// as in-traversal filtered HNSW with the `matches_point` recheck instead of
/// finding the tiny intersection — results stay correct via the recheck, only
/// the perf path differs. The common CE case (one selective scoping key such
/// as a repo filter) keeps the fast candidate path.
fn indexed_filter_must_count_estimate_be(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    filter: &Filter,
    cap: usize,
) -> Result<Option<usize>, GraphError> {
    let mut estimate: Option<usize> = None;
    for condition in &filter.must {
        let count = match condition {
            Condition::Field(field) => indexed_condition_count_be(storage, r, field, cap)?,
            Condition::Nested(nested) => {
                indexed_filter_must_count_estimate_be(storage, r, nested, cap)?
            }
            Condition::HasId(_) | Condition::IsEmpty(_) | Condition::IsNull(_) => None,
        };
        if let Some(count) = count {
            estimate = Some(estimate.map_or(count, |existing| existing.min(count)));
        }
    }
    Ok(estimate)
}

/// Result of a probed candidate resolution.
struct ProbedCandidates {
    candidates: Option<HashSet<u128>>,
    /// Upper-bound cardinality estimate from the count probe (saturates at
    /// the probe cap + 1); `None` when the probe did not run or abstained.
    probe_estimate: Option<usize>,
    /// True when the probe concluded the candidate set would be dropped as
    /// too broad after materialization, so materialization was skipped.
    probe_skipped: bool,
}

fn vector_query_indexed_filter_candidates(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    txn: &heed3::RoTxn,
    filter: &Filter,
) -> Result<Option<HashSet<u128>>, GraphError> {
    Ok(vector_query_indexed_filter_candidates_probed(
        storage,
        txn,
        filter,
        filter_count_probe_enabled(),
    )?
    .candidates)
}

fn vector_query_indexed_filter_candidates_probed(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    txn: &heed3::RoTxn,
    filter: &Filter,
    probe_enabled: bool,
) -> Result<ProbedCandidates, GraphError> {
    let total = storage
        .get_metadata(txn)
        .map(|m| m.stats.vector_count)
        .unwrap_or(0) as usize;

    // Pre-materialization count probe: the too-broad drop below used to be
    // decided only AFTER building the candidate HashSet, so a filter matching
    // 500k points built a 500k-id set just to throw it away. Probe the
    // payload indexes for a count-only upper bound first and skip
    // materialization when the drop is already certain (see
    // `indexed_filter_must_count_estimate_be` for the upper-bound trade-off).
    let mut probe_estimate = None;
    if probe_enabled && total > 0 && !filter.must.is_empty() {
        let cap = vector_query_broad_candidate_cap(total);
        let r = storage.backend.read_borrowed(txn);
        probe_estimate = indexed_filter_must_count_estimate_be(storage, &r, filter, cap)?;
        if let Some(estimate) = probe_estimate {
            if vector_query_candidates_too_broad(estimate, total) {
                return Ok(ProbedCandidates {
                    candidates: None,
                    probe_estimate,
                    probe_skipped: true,
                });
            }
        }
    }

    let Some(candidates) = indexed_filter_candidates(storage, txn, filter)? else {
        return Ok(ProbedCandidates {
            candidates: None,
            probe_estimate,
            probe_skipped: false,
        });
    };
    if vector_query_candidates_too_broad(candidates.len(), total) {
        return Ok(ProbedCandidates {
            candidates: None,
            probe_estimate,
            probe_skipped: false,
        });
    }
    Ok(ProbedCandidates {
        candidates: Some(candidates),
        probe_estimate,
        probe_skipped: false,
    })
}

fn vector_query_indexed_filter_candidates_be_probed(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    filter: &Filter,
    probe_enabled: bool,
) -> Result<ProbedCandidates, GraphError> {
    let total = collection_vector_count_be(storage) as usize;

    let mut probe_estimate = None;
    if probe_enabled && total > 0 && !filter.must.is_empty() {
        let cap = vector_query_broad_candidate_cap(total);
        probe_estimate = indexed_filter_must_count_estimate_be(storage, r, filter, cap)?;
        if let Some(estimate) = probe_estimate {
            if vector_query_candidates_too_broad(estimate, total) {
                return Ok(ProbedCandidates {
                    candidates: None,
                    probe_estimate,
                    probe_skipped: true,
                });
            }
        }
    }

    let Some(candidates) = indexed_filter_candidates_be(storage, r, filter)? else {
        return Ok(ProbedCandidates {
            candidates: None,
            probe_estimate,
            probe_skipped: false,
        });
    };
    if vector_query_candidates_too_broad(candidates.len(), total) {
        return Ok(ProbedCandidates {
            candidates: None,
            probe_estimate,
            probe_skipped: false,
        });
    }
    Ok(ProbedCandidates {
        candidates: Some(candidates),
        probe_estimate,
        probe_skipped: false,
    })
}

fn max_scan_index_candidates() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HELIX_SCAN_INDEX_CANDIDATE_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(100_000)
    })
}

/// Largest candidate-set size a scroll/scan keeps: candidate paging beats a
/// primary scan until the set is both huge and a large share of the
/// collection. (`count > max_scan && count*2 > total` collapses to
/// `count > cap`.)
fn scan_broad_candidate_cap(total: usize) -> usize {
    max_scan_index_candidates().max(total / 2)
}

fn scan_candidates_too_broad(candidate_count: usize, total: usize) -> bool {
    total > 0 && candidate_count > scan_broad_candidate_cap(total)
}

fn scan_indexed_filter_candidates(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    txn: &heed3::RoTxn,
    filter: &Filter,
) -> Result<Option<HashSet<u128>>, GraphError> {
    if filter.is_empty() {
        return Ok(None);
    }
    let Some(candidates) = indexed_filter_candidates(storage, txn, filter)? else {
        return Ok(None);
    };
    let candidate_count = candidates.len();
    if candidate_count == 0 {
        return Ok(Some(candidates));
    }

    let total = storage
        .get_metadata(txn)
        .map(|m| m.stats.node_count)
        .unwrap_or(0) as usize;
    if scan_candidates_too_broad(candidate_count, total) {
        return Ok(None);
    }

    Ok(Some(candidates))
}

fn scan_indexed_filter_candidates_be(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    filter: &Filter,
) -> Result<Option<HashSet<u128>>, GraphError> {
    Ok(
        scan_indexed_filter_candidates_be_probed(storage, r, filter, filter_count_probe_enabled())?
            .candidates,
    )
}

fn scan_indexed_filter_candidates_be_probed(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    filter: &Filter,
    probe_enabled: bool,
) -> Result<ProbedCandidates, GraphError> {
    if filter.is_empty() {
        return Ok(ProbedCandidates {
            candidates: None,
            probe_estimate: None,
            probe_skipped: false,
        });
    }

    let total = storage
        .get_metadata_be(r)
        .map(|m| m.stats.node_count)
        .unwrap_or(0) as usize;

    // Pre-materialization count probe; same rationale and trade-off as
    // `vector_query_indexed_filter_candidates_probed`, with the scan-path
    // broadness threshold.
    let mut probe_estimate = None;
    if probe_enabled && total > 0 && !filter.must.is_empty() {
        let cap = scan_broad_candidate_cap(total);
        probe_estimate = indexed_filter_must_count_estimate_be(storage, r, filter, cap)?;
        if let Some(estimate) = probe_estimate {
            if scan_candidates_too_broad(estimate, total) {
                tracing::debug!(
                    estimated_cardinality = estimate,
                    total_nodes = total,
                    "scan filter count probe skipped candidate materialization"
                );
                return Ok(ProbedCandidates {
                    candidates: None,
                    probe_estimate,
                    probe_skipped: true,
                });
            }
        }
    }

    let Some(candidates) = indexed_filter_candidates_be(storage, r, filter)? else {
        return Ok(ProbedCandidates {
            candidates: None,
            probe_estimate,
            probe_skipped: false,
        });
    };
    let candidate_count = candidates.len();
    if candidate_count > 0 && scan_candidates_too_broad(candidate_count, total) {
        return Ok(ProbedCandidates {
            candidates: None,
            probe_estimate,
            probe_skipped: false,
        });
    }

    Ok(ProbedCandidates {
        candidates: Some(candidates),
        probe_estimate,
        probe_skipped: false,
    })
}

/// When both inclusive and exclusive bounds are present, the index scan must
/// use the stricter one: candidates are treated as exact (e.g. for must_not
/// subtraction), so a looser bound would over-select.
fn range_lower_bound(range: &RangeCondition) -> Option<(f64, bool)> {
    match (range.gte, range.gt) {
        (Some(gte), Some(gt)) if gt >= gte => Some((gt, false)),
        (Some(gte), _) => Some((gte, true)),
        (None, gt) => gt.map(|value| (value, false)),
    }
}

fn range_upper_bound(range: &RangeCondition) -> Option<(f64, bool)> {
    match (range.lte, range.lt) {
        (Some(lte), Some(lt)) if lt <= lte => Some((lt, false)),
        (Some(lte), _) => Some((lte, true)),
        (None, lt) => lt.map(|value| (value, false)),
    }
}

#[derive(Serialize)]
struct QdrantResponse<'a, T: Serialize> {
    status: &'a str,
    result: T,
}

// ─── Collection Management ───

#[derive(Deserialize)]
struct CreateCollectionRequest {
    vectors: HashMap<String, VectorParamsInput>,
    #[serde(default)]
    sparse_vectors: HashMap<String, SparseVectorParamsInput>,
    /// Qdrant-native collection-level quantization. Applied to every named
    /// vector that didn't set its own `quantization` block. Accepts the
    /// tagged form `{"scalar": {...}}`, `{"binary": {...}}`, or
    /// `{"product": {...}}` (see Qdrant REST docs) and translates it to
    /// our internal Spindle config. Per-vector `quantization` always wins.
    #[serde(default)]
    quantization_config: Option<CollectionQuantizationConfig>,
    /// Qdrant-native collection-level HNSW tuning. Any subset of
    /// `m` / `ef_construct` is accepted; unknown fields are ignored for
    /// forward-compat. `ef_construct` is mapped to our internal
    /// `ef_construction` name. Persisted per-collection and layered over
    /// the global `HELIX_HNSW_*` defaults at index-build time.
    #[serde(default)]
    hnsw_config: Option<HnswConfigInput>,
}

/// Qdrant-style `hnsw_config` block accepted on create_collection. All
/// fields optional so partial tuning is allowed. Unknown fields tolerated
/// since Qdrant has ~8 HNSW knobs and we only implement three.
#[derive(Deserialize, Default)]
struct HnswConfigInput {
    #[serde(default)]
    m: Option<usize>,
    /// Qdrant's camelCase-free name. Maps to our `ef_construction`.
    #[serde(default)]
    ef_construct: Option<usize>,
    /// Optional search-time default ef. Qdrant doesn't expose this at
    /// create-time but we do, so callers can tune once instead of passing
    /// `params.hnsw_ef` on every query.
    #[serde(default)]
    ef: Option<usize>,
}

impl HnswConfigInput {
    fn to_overrides(&self) -> Result<HnswOverrides, String> {
        let overrides = HnswOverrides {
            m: self.m,
            ef_construction: self.ef_construct,
            ef: self.ef,
        };
        overrides.validate().map_err(|e| e.to_string())?;
        Ok(overrides)
    }
}

/// Qdrant's tagged collection-level quantization. Exactly one variant is
/// expected; additional fields are ignored for forward-compatibility.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CollectionQuantizationConfig {
    #[serde(default)]
    scalar: Option<ScalarQuantizationBody>,
    #[serde(default)]
    binary: Option<BinaryQuantizationBody>,
    #[serde(default)]
    product: Option<ProductQuantizationBody>,
}

#[derive(Deserialize)]
struct ScalarQuantizationBody {
    // Qdrant only defines `int8` today; we accept any string and map
    // unknown values to ScalarInt8 since we don't support other widths.
    #[serde(rename = "type", default)]
    _type: Option<String>,
    // Qdrant-only tuning knobs we don't model. Accept + ignore for compat.
    #[serde(default)]
    _quantile: Option<f32>,
    #[serde(default)]
    _always_ram: Option<bool>,
}

#[derive(Deserialize)]
struct BinaryQuantizationBody {
    #[serde(default)]
    _always_ram: Option<bool>,
}

#[derive(Deserialize)]
struct ProductQuantizationBody {
    // Qdrant uses strings like "x4", "x8", "x16", "x32", "x64". We map these
    // to our TurboProd mode, which is our nearest PQ-family equivalent.
    #[serde(default)]
    _compression: Option<String>,
    #[serde(default)]
    _always_ram: Option<bool>,
}

impl CollectionQuantizationConfig {
    /// Translate the Qdrant tagged form into our per-vector input struct.
    /// Returns `None` when no variant was populated (caller falls back to
    /// "no quantization").
    fn to_vector_input(&self) -> Option<VectorQuantizationInput> {
        if self.scalar.is_some() {
            return Some(VectorQuantizationInput {
                mode: "scalar".into(),
                keep_original: None,
                rescore: None,
                oversampling: None,
                binary_dims: None,
                turbo_dims: None,
            });
        }
        if self.binary.is_some() {
            return Some(VectorQuantizationInput {
                mode: "binary".into(),
                keep_original: None,
                rescore: None,
                oversampling: None,
                binary_dims: None,
                turbo_dims: None,
            });
        }
        if self.product.is_some() {
            return Some(VectorQuantizationInput {
                mode: "turbo_prod".into(),
                keep_original: None,
                rescore: None,
                oversampling: None,
                binary_dims: None,
                turbo_dims: None,
            });
        }
        None
    }
}

#[derive(Deserialize)]
struct SparseVectorParamsInput {
    #[serde(default)]
    index: Option<SparseIndexParamsInput>,
    #[serde(default)]
    modifier: Option<String>,
}

#[derive(Deserialize)]
struct SparseIndexParamsInput {
    #[serde(default = "default_full_scan_threshold")]
    full_scan_threshold: usize,
}

fn default_full_scan_threshold() -> usize {
    5000
}

#[derive(Deserialize)]
struct VectorParamsInput {
    size: usize,
    #[serde(default = "default_distance")]
    distance: String,
    /// Helix-native per-vector quantization: `{"mode": "scalar", ...}`.
    #[serde(default)]
    quantization: Option<VectorQuantizationInput>,
    /// Qdrant-native per-vector quantization: `{"scalar": {...}}` etc.
    /// `qdrant-client` serialises `VectorParams(quantization_config=...)`
    /// into this field; without this, the qdrant-client payload reached
    /// here and was silently dropped because the unknown field was ignored.
    /// When both `quantization` and `quantization_config` are present the
    /// Helix-native field wins.
    #[serde(default)]
    quantization_config: Option<CollectionQuantizationConfig>,
}

fn default_distance() -> String {
    "Cosine".into()
}

#[derive(Deserialize)]
struct VectorQuantizationInput {
    #[serde(default = "default_quantization_mode")]
    mode: String,
    #[serde(default)]
    keep_original: Option<bool>,
    #[serde(default)]
    rescore: Option<bool>,
    #[serde(default)]
    oversampling: Option<usize>,
    #[serde(default)]
    binary_dims: Option<usize>,
    #[serde(default)]
    turbo_dims: Option<usize>,
}

fn default_quantization_mode() -> String {
    "none".into()
}

/// Cheap by-value clone for `VectorQuantizationInput`. Used to thread the
/// same struct through multiple resolution fallbacks without deriving Clone
/// on a public-shaped type (which would change its API surface).
fn clone_vector_quantization_input(q: &VectorQuantizationInput) -> VectorQuantizationInput {
    VectorQuantizationInput {
        mode: q.mode.clone(),
        keep_original: q.keep_original,
        rescore: q.rescore,
        oversampling: q.oversampling,
        binary_dims: q.binary_dims,
        turbo_dims: q.turbo_dims,
    }
}

/// Resolve a `SpindleMode` from the `HELIX_SPINDLE_MODE` env var.
///
/// This is the deployment-wide default applied to collections created
/// WITHOUT an explicit `quantization` block in the create-collection
/// request. Context Engine's watcher creates collections through the
/// Qdrant-compatible endpoint without a quantization field, so before this
/// existed every new collection silently fell back to `SpindleMode::None`
/// (raw f32) regardless of `HELIX_SEGMENT_FORMAT` — `HELIX_SEGMENT_FORMAT`
/// only governs MERGE-output sidecar conversion, never the write path.
///
/// Default is `none` so deploying this code is a no-op until the env is
/// set; existing collections are never touched (segment files are
/// self-describing via their HVEC/HVS8 magic, so mixed-format collections
/// already decode correctly). An explicit `quantization` block in the
/// request always wins over this env default.
fn spindle_mode_from_env() -> SpindleMode {
    match std::env::var("HELIX_SPINDLE_MODE")
        .ok()
        .map(|s| s.trim().to_lowercase())
        .as_deref()
    {
        Some("scalar" | "scalar_int8" | "int8") => SpindleMode::ScalarInt8,
        Some("binary" | "binary_sign") => SpindleMode::BinarySign,
        Some("turbo" | "turbo_int4" | "int4" | "mrc" | "mrc_int4") => SpindleMode::TurboInt4,
        Some("turbo_prod" | "turboquant" | "tq") => SpindleMode::TurboProd,
        Some("none" | "") | None => SpindleMode::None,
        Some(other) => {
            tracing::warn!(
                mode = %other,
                "Unknown HELIX_SPINDLE_MODE; defaulting to none"
            );
            SpindleMode::None
        }
    }
}

fn spindle_bool_from_env(name: &str, default: bool) -> bool {
    match std::env::var(name)
        .ok()
        .map(|s| s.trim().to_lowercase())
        .as_deref()
    {
        Some("1" | "true" | "yes" | "on") => true,
        Some("0" | "false" | "no" | "off") => false,
        Some(other) => {
            tracing::warn!(
                env = name,
                value = %other,
                default,
                "Invalid spindle boolean env; using default"
            );
            default
        }
        None => default,
    }
}

fn spindle_usize_from_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn default_spindle_config_for_env(
    mode: SpindleMode,
    keep_original: bool,
    rescore: bool,
    oversampling: usize,
) -> SpindleConfig {
    if mode == SpindleMode::None {
        return SpindleConfig {
            mode: SpindleMode::None,
            keep_original: false,
            rescore: false,
            oversampling: 1,
            binary_dims: 1,
            turbo_dims: 1,
        };
    }

    SpindleConfig {
        mode,
        keep_original,
        rescore,
        oversampling: oversampling.max(1),
        ..SpindleConfig::default()
    }
}

/// Build the default `SpindleConfig` used when a create-collection request
/// carries no explicit `quantization` block. Returns a fully-disabled
/// config when `HELIX_SPINDLE_MODE` is unset/none (byte-identical to the
/// historical behavior). When enabled, the deployment must choose compact
/// storage vs exact rerank with explicit env:
///
/// * `HELIX_SPINDLE_KEEP_ORIGINAL=false` stores only quantized vectors in LMDB.
/// * `HELIX_SPINDLE_KEEP_ORIGINAL=true` retains f32 originals for rescore,
///   improving recall but preserving much of the LMDB storage pressure.
fn env_default_spindle_config() -> SpindleConfig {
    let mode = spindle_mode_from_env();
    let keep_original = spindle_bool_from_env("HELIX_SPINDLE_KEEP_ORIGINAL", false);
    let requested_rescore = spindle_bool_from_env("HELIX_SPINDLE_RESCORE", keep_original);
    if requested_rescore && !keep_original {
        tracing::warn!(
            "HELIX_SPINDLE_RESCORE=true requires HELIX_SPINDLE_KEEP_ORIGINAL=true; disabling env-default rescore"
        );
    }
    let rescore = requested_rescore && keep_original;
    let oversampling_default = if keep_original { 16 } else { 1 };
    default_spindle_config_for_env(
        mode,
        keep_original,
        rescore,
        spindle_usize_from_env("HELIX_SPINDLE_OVERSAMPLING", oversampling_default),
    )
}

fn parse_spindle_config(input: &Option<VectorQuantizationInput>) -> SpindleConfig {
    // A disabled quantizer must be internally consistent: rescore/oversampling
    // are only meaningful when there's a lossy codec to rescore against.
    // Inheriting SpindleConfig::default() here would store `oversampling: 16`
    // on the collection and surface on /info, which is misleading even though
    // the search paths already clamp oversampling to 1 when !is_enabled().
    fn disabled() -> SpindleConfig {
        // Every field named explicitly: spreading `..SpindleConfig::default()`
        // would pull in TurboProd-flavoured `binary_dims`/`turbo_dims` which
        // have no meaning when mode is None. Using a fixed pair keeps
        // /info output predictable and decouples us from future default
        // tweaks that shouldn't affect disabled collections.
        SpindleConfig {
            mode: SpindleMode::None,
            keep_original: false,
            rescore: false,
            oversampling: 1,
            binary_dims: 1,
            turbo_dims: 1,
        }
    }

    let Some(input) = input.as_ref() else {
        // No explicit quantization block in the request. This is the
        // Context Engine watcher path. Fall back to the deployment-wide
        // env default (HELIX_SPINDLE_MODE) instead of unconditionally
        // disabling — that env defaults to none, so behavior is unchanged
        // until an operator opts in, and no CE change is required to start
        // quantizing newly created / re-ingested collections.
        return env_default_spindle_config();
    };

    let mode = match input.mode.trim().to_lowercase().as_str() {
        "scalar" | "scalar_int8" | "int8" => SpindleMode::ScalarInt8,
        "binary" | "binary_sign" => SpindleMode::BinarySign,
        "turbo" | "turbo_int4" | "int4" | "mrc" | "mrc_int4" => SpindleMode::TurboInt4,
        "turbo_prod" | "turboquant" | "tq" => SpindleMode::TurboProd,
        "none" | "" => SpindleMode::None,
        other => {
            // Unknown mode strings are a config-drift footgun: they silently
            // become None and the user thinks they have quantization. Warn
            // so ops sees this in logs, then honour the safe default.
            tracing::warn!(
                mode = %other,
                "Unknown quantization mode; defaulting to none"
            );
            SpindleMode::None
        }
    };

    if mode == SpindleMode::None {
        return disabled();
    }

    let mut config = SpindleConfig {
        mode,
        keep_original: false,
        rescore: false,
        oversampling: 1,
        ..SpindleConfig::default()
    };

    if let Some(value) = input.keep_original {
        config.keep_original = value;
    }
    if let Some(value) = input.rescore {
        config.rescore = value;
    }
    if let Some(value) = input.oversampling {
        config.oversampling = value.max(1);
    }
    if let Some(value) = input.binary_dims {
        config.binary_dims = value.max(1);
    }
    if let Some(value) = input.turbo_dims {
        config.turbo_dims = value.max(1);
    }

    // rescore_results takes a different branch when !keep_original and
    // re-scores against the same quantized bytes the search used. That's a
    // no-op cycle that produces no fidelity improvement; warn so the user
    // either turns off rescore or persists the original vectors. Dedup with
    // an AtomicBool so logs stay clean if many misconfigured tenants land in
    // the same process.
    if config.rescore && !config.keep_original {
        use std::sync::atomic::{AtomicBool, Ordering};
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                mode = ?config.mode,
                "Spindle config has rescore=true but keep_original=false; rescore will re-check against the same quantized bytes (no-op). Set keep_original=true or rescore=false. (further occurrences suppressed this process)"
            );
        }
    }

    config
}

fn spindle_config_json(config: &SpindleConfig) -> serde_json::Value {
    serde_json::json!({
        "mode": config.mode.as_str(),
        "keep_original": config.keep_original,
        "rescore": config.rescore,
        "oversampling": config.oversampling,
        "binary_dims": config.binary_dims,
        "turbo_dims": config.turbo_dims,
    })
}

fn qdrant_quantization_config_json(config: &SpindleConfig) -> Option<serde_json::Value> {
    match config.mode {
        SpindleMode::None => None,
        SpindleMode::ScalarInt8 => Some(serde_json::json!({
            "scalar": {
                "type": "int8",
                "always_ram": true,
            }
        })),
        SpindleMode::BinarySign => Some(serde_json::json!({
            "binary": {
                "always_ram": true,
            }
        })),
        // Qdrant has no typed VectorParams shape for Helix/TurboQuant's
        // TurboInt4 or TurboProd payloads. Keep GET /collections parseable by
        // qdrant-client and expose the exact Helix-native shape via
        // config.metadata.helix_quantization instead.
        SpindleMode::TurboInt4 | SpindleMode::TurboProd => None,
    }
}

/// PUT /collections/{name}
pub fn handle_create_collection(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let name = match get_collection_name(input) {
        Some(n) => n,
        None => return json_err(response, 400, "Missing collection name"),
    };
    let req: CreateCollectionRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_err(response, 400, &format!("Invalid JSON: {}", e)),
    };

    // Collection-level Qdrant quantization_config applies to every named
    // vector that didn't set its own quantization block. Per-vector input
    // still wins on conflict so existing Helix-native payloads keep working.
    let collection_fallback = req
        .quantization_config
        .as_ref()
        .and_then(|cfg| cfg.to_vector_input());

    let mut vectors = HashMap::new();
    for (vec_name, params) in &req.vectors {
        let distance = match params.distance.to_lowercase().as_str() {
            "cosine" => DistanceMetric::Cosine,
            "dot" => DistanceMetric::Dot,
            "euclid" | "euclidean" => DistanceMetric::Euclid,
            _ => DistanceMetric::Cosine,
        };

        // Resolution order (first match wins):
        //   1. Helix-native per-vector `quantization` (explicit knobs).
        //   2. Qdrant-native per-vector `quantization_config` (what the
        //      python qdrant-client emits when you pass quantization to
        //      models.VectorParams — this is the real-world default).
        //   3. Collection-level `quantization_config`.
        //   4. None.
        let quant = params
            .quantization
            .as_ref()
            .map(clone_vector_quantization_input)
            .or_else(|| {
                params
                    .quantization_config
                    .as_ref()
                    .and_then(|cfg| cfg.to_vector_input())
            })
            .or_else(|| {
                collection_fallback
                    .as_ref()
                    .map(clone_vector_quantization_input)
            });

        vectors.insert(
            vec_name.clone(),
            NamedVectorConfig {
                size: params.size,
                distance,
                spindle: parse_spindle_config(&quant),
            },
        );
    }

    let mut sparse_vectors = HashMap::new();
    for (sp_name, sp_params) in &req.sparse_vectors {
        let modifier = match sp_params.modifier.as_deref() {
            Some("idf") | Some("IDF") | Some("Idf") => SparseModifier::Idf,
            _ => SparseModifier::None,
        };
        let full_scan_threshold = sp_params
            .index
            .as_ref()
            .map(|idx| idx.full_scan_threshold)
            .unwrap_or(5000);
        sparse_vectors.insert(
            sp_name.clone(),
            SparseVectorConfig {
                full_scan_threshold,
                modifier,
                ..Default::default()
            },
        );
    }

    let hnsw_overrides = match req.hnsw_config.as_ref().map(HnswConfigInput::to_overrides) {
        Some(Ok(overrides)) if !overrides.is_empty() => Some(overrides),
        Some(Ok(_)) | None => None,
        Some(Err(message)) => return json_err(response, 400, &message),
    };

    // Idempotent fast-path for existing collections.
    //
    // CE's watcher / index-worker call PUT /collections/<name> on every
    // startup + periodic backfill tick to "ensure" companion graph
    // stores. Under a deep write queue these calls were piling up behind
    // real upserts — the client's 30 s read timeout was firing because
    // the idempotent PUT ended up queued for >30 s before Raft applied
    // it as a no-op CreateCollection mutation.
    //
    let schema_bearing_create =
        !vectors.is_empty() || !sparse_vectors.is_empty() || hnsw_overrides.is_some();
    // Schema-bearing creates always fall through to the real create so they can
    // (re)materialize/repair metadata. Bare "ensure" probes short-circuit
    // success, but only when materialization is cheaply confirmed — never via a
    // cold env open (that would stampede under startup/backfill load).
    //
    // A collection *directory* existing does NOT prove it was materialized:
    // `HelixGraphStorage::new` runs `create_dir_all` before the schema/metadata
    // write commits, so a create that failed mid-way (MapFull, resize/memory
    // backpressure, or a fatal storage error) leaves an orphan or degraded
    // directory behind. ACKing those as success strands the collection
    // registered-but-empty forever, because callers treat the ACK as "indexed"
    // and never re-send the schema. So short-circuit success only when
    // materialization is cheaply confirmed (loaded healthy env, or a readable
    // metadata sidecar); surface the fatal error for a degraded collection; and
    // let a bare/orphan directory fall through to the real create below, which
    // repairs it.
    if !schema_bearing_create {
        match input.collections.get_loaded_collection(&name)? {
            Some(storage) => {
                if let Some(err) = storage.degraded_collection_error() {
                    return Err(err);
                }
                metrics::counter!("helix_create_collection_idempotent_total").increment(1);
                return json_created(response, &true);
            }
            None => {
                if let Some((code, message)) = input.collections.collection_degraded_marker(&name) {
                    return Err(GraphError::FatalCollectionStorage { code, message });
                }
                if input
                    .collections
                    .collection_metadata_sidecar(&name)?
                    .is_some()
                {
                    metrics::counter!("helix_create_collection_idempotent_total").increment(1);
                    return json_created(response, &true);
                }
            }
        }
    }

    input
        .replication
        .apply(ReplicatedMutation::CreateCollection {
            name,
            vectors,
            sparse_vectors,
            hnsw_overrides,
        })?;

    json_created(response, &true)
}

/// GET /collections/{name}
pub fn handle_get_collection(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let name = match get_collection_name(input) {
        Some(n) => n,
        None => return json_err(response, 400, "Missing collection name"),
    };
    let request_started = Instant::now();

    // Qdrant clients call this endpoint as a cheap schema/capability probe.
    // Keep it on metadata/in-memory snapshots only: no recursive dir_size(),
    // no live dense-segment scans, and no HNSW/file existence diagnostics.
    // Deep checks belong on admin diagnostics endpoints, not this hot path.
    let loaded_storage = input.collections.get_loaded_collection(&name)?;
    let (metadata, cold_metadata) = match &loaded_storage {
        Some(storage) => {
            // A reader replica's metadata snapshot is pinned at open time — it
            // never commits locally, so points_count would otherwise stay frozen
            // at the first-open value while its search path is eventually
            // consistent. Re-read committed metadata from object storage (through
            // the self-refreshing DbReader checkpoint) without blocking this
            // schema probe on dense/HNSW sidecar reconciliation.
            if storage.backend.is_reader_replica() {
                storage.refresh_metadata_snapshot_best_effort();
                storage.maybe_refresh_reader_view(reader_refresh_ttl_ms());
                // No poll-tier promotion on this path: `get_loaded_collection`
                // above is the cache-only lookup (no cold-open, doesn't route
                // through `get_collection`), and this handler is a deliberately
                // cheap schema/capability probe — the single promotion site is
                // `get_collection`'s post-lock path, used by the actual
                // search/scroll/query read handlers this poll tier exists for.
            }
            (storage.metadata_snapshot()?, false)
        }
        None => match input.collections.collection_metadata_sidecar(&name)? {
            Some(sidecar) => (sidecar.metadata, true),
            None if input.collections.collection_exists(&name)? => {
                return json_err(
                    response,
                    503,
                    &format!("Collection '{}' metadata sidecar is not available", name),
                );
            }
            None => {
                return json_err(
                    response,
                    404,
                    &format!("Collection '{}' does not exist", name),
                )
            }
        },
    };
    let mut core_names = HashSet::new();
    let mut core_names_available = false;
    let mut indexed_core_names = HashSet::new();
    let mut live_status_skipped = Vec::new();
    if let Some(storage) = &loaded_storage {
        let core_status_started = Instant::now();
        match storage
            .named_vectors
            .try_core_names()
            .map_err(GraphError::from)?
        {
            Some(names) => {
                core_names = names;
                core_names_available = true;
                metrics::histogram!(
                    "helix_qdrant_collection_info_phase_ms",
                    "phase" => "core_names",
                    "outcome" => "ok"
                )
                .record(core_status_started.elapsed().as_secs_f64() * 1000.0);
            }
            None => {
                live_status_skipped.push("VECTOR_CORE_STATUS_BUSY");
                metrics::counter!(
                    "helix_qdrant_collection_info_live_status_skipped_total",
                    "reason" => "vector_core_busy"
                )
                .increment(1);
                metrics::histogram!(
                    "helix_qdrant_collection_info_phase_ms",
                    "phase" => "core_names",
                    "outcome" => "busy"
                )
                .record(core_status_started.elapsed().as_secs_f64() * 1000.0);
            }
        }
        for space in metadata.dense_vector_spaces.values() {
            indexed_core_names.extend(
                space
                    .segments
                    .iter()
                    .filter(|segment| segment.role == DenseVectorSegmentRole::Indexed)
                    .map(|segment| segment.physical_name.clone()),
            );
        }
    } else {
        for space in metadata.dense_vector_spaces.values() {
            indexed_core_names.extend(
                space
                    .segments
                    .iter()
                    .filter(|segment| segment.role == DenseVectorSegmentRole::Indexed)
                    .map(|segment| segment.physical_name.clone()),
            );
        }
    }
    let vector_configs = metadata.named_vectors.clone();
    let sparse_configs = metadata.sparse_vectors.clone();
    let dense_spaces = metadata.dense_vector_spaces.clone();
    let global_threshold = input.collections.config().vector_flat_scan_threshold();
    let flat_scan_threshold = loaded_storage
        .as_ref()
        .map(|storage| {
            storage
                .named_vectors
                .effective_indexing_threshold(global_threshold)
        })
        .unwrap_or_else(|| {
            metadata
                .indexing_threshold_override
                .unwrap_or(global_threshold)
        });

    let dense_space_count = vector_configs.len().max(1) as u64;
    let vectors_count = metadata
        .stats
        .vector_count
        .saturating_mul(dense_space_count);
    let mut segments_count = 0usize;
    let mut missing_vector_segments = Vec::new();
    let mut any_dense_segment_indexed = false;
    for (vname, space) in &dense_spaces {
        segments_count += space.segments.len();
        for segment in &space.segments {
            if indexed_core_names.contains(&segment.physical_name) {
                any_dense_segment_indexed = true;
            }
            if core_names_available && !core_names.contains(&segment.physical_name) {
                missing_vector_segments.push(serde_json::json!({
                    "vector": vname,
                    "segment": segment.physical_name,
                }));
            }
        }
    }
    segments_count = segments_count.max(vector_configs.len()).max(1);
    let missing_segments_count = missing_vector_segments.len();
    let has_missing_segments = missing_segments_count > 0;
    let indexed_vectors_count = if any_dense_segment_indexed && !has_missing_segments {
        vectors_count
    } else {
        0
    };

    let mut vectors_info: HashMap<String, serde_json::Value> = HashMap::new();
    let mut helix_quantization_info: HashMap<String, serde_json::Value> = HashMap::new();
    for (vname, vconfig) in &vector_configs {
        let mut info = serde_json::json!({
            "size": vconfig.size,
            "distance": format!("{:?}", vconfig.distance),
        });
        if vconfig.spindle.is_enabled() {
            helix_quantization_info.insert(vname.clone(), spindle_config_json(&vconfig.spindle));
            if let serde_json::Value::Object(ref mut object) = info {
                if let Some(qdrant_quantization) = qdrant_quantization_config_json(&vconfig.spindle)
                {
                    object.insert("quantization_config".to_string(), qdrant_quantization);
                }
            }
        }
        vectors_info.insert(vname.clone(), info);
    }

    let mut sparse_info: HashMap<String, serde_json::Value> = HashMap::new();
    for (sp_name, sp_config) in &sparse_configs {
        let modifier_str = match sp_config.modifier {
            SparseModifier::Idf => "idf",
            SparseModifier::None => "none",
        };
        sparse_info.insert(
            sp_name.clone(),
            serde_json::json!({
                "modifier": modifier_str,
                "index": {
                    "full_scan_threshold": sp_config.full_scan_threshold,
                },
            }),
        );
    }

    let mut payload_schema: HashMap<String, serde_json::Value> = metadata
        .payload_indices
        .iter()
        .map(|(name, schema)| (name.clone(), payload_index_schema_json(schema)))
        .collect();
    if let Some(storage) = &loaded_storage {
        let payload_status_started = Instant::now();
        match storage.try_list_payload_index_statuses()? {
            Some(statuses) => {
                for status in statuses {
                    if matches!(status.status, "cancelled") {
                        continue;
                    }
                    let mut entry = payload_index_schema_json(&status.schema);
                    if let serde_json::Value::Object(ref mut object) = entry {
                        object.insert(
                            "status".to_string(),
                            serde_json::Value::String(status.status.to_string()),
                        );
                        object.insert(
                            "points".to_string(),
                            serde_json::json!(status.indexed_nodes),
                        );
                    }
                    payload_schema.insert(status.field_name, entry);
                }
                metrics::histogram!(
                    "helix_qdrant_collection_info_phase_ms",
                    "phase" => "payload_index_status",
                    "outcome" => "ok"
                )
                .record(payload_status_started.elapsed().as_secs_f64() * 1000.0);
            }
            None => {
                live_status_skipped.push("PAYLOAD_INDEX_STATUS_BUSY");
                metrics::counter!(
                    "helix_qdrant_collection_info_live_status_skipped_total",
                    "reason" => "payload_index_busy"
                )
                .increment(1);
                metrics::histogram!(
                    "helix_qdrant_collection_info_phase_ms",
                    "phase" => "payload_index_status",
                    "outcome" => "busy"
                )
                .record(payload_status_started.elapsed().as_secs_f64() * 1000.0);
            }
        }
    }

    let mut warnings = Vec::new();
    if cold_metadata {
        warnings.push(serde_json::json!({
            "code": "COLD_COLLECTION_METADATA",
            "message": "Collection info was served from persisted metadata without opening storage."
        }));
    }
    if has_missing_segments {
        warnings.push(serde_json::json!({
            "code": "MISSING_VECTOR_SEGMENTS",
            "message": "Collection metadata references missing vector segments; collection should be reindexed."
        }));
    }
    for code in live_status_skipped {
        warnings.push(serde_json::json!({
            "code": code,
            "message": "Live collection diagnostics were busy; collection info used committed metadata for this field."
        }));
    }

    let result = json_ok(
        response,
        &serde_json::json!({
            "status": if has_missing_segments { "yellow" } else { "green" },
            // Qdrant clients parse this as either the enum string "ok" or a
            // structured optimizer error object. Keep compatibility here and
            // expose Helix-specific repair detail through `warnings` below.
            "optimizer_status": "ok",
            "segments_count": segments_count,
            "config": {
                "params": {
                    "vectors": vectors_info,
                    "sparse_vectors": sparse_info,
                    "shard_number": 1,
                    "replication_factor": 1,
                    "write_consistency_factor": 1,
                    "on_disk_payload": true,
                },
                "hnsw_config": {
                    "m": 16,
                    "ef_construct": 100,
                    "full_scan_threshold": flat_scan_threshold,
                    "max_indexing_threads": 0,
                    "on_disk": false,
                    "payload_m": null,
                },
                "optimizer_config": {
                    "deleted_threshold": 0.2,
                    "vacuum_min_vector_number": 1000,
                    "default_segment_number": 1,
                    "max_segment_size": null,
                    "memmap_threshold": null,
                    "indexing_threshold": flat_scan_threshold,
                    "flush_interval_sec": 5,
                    "max_optimization_threads": 1,
                },
                "metadata": {
                    "helix_quantization": helix_quantization_info,
                },
            },
            "payload_schema": payload_schema,
            "points_count": metadata.stats.vector_count,
            "vectors_count": vectors_count,
            "indexed_vectors_count": indexed_vectors_count,
            "missing_vector_segments_count": missing_segments_count,
            "missing_vector_segments": missing_vector_segments,
            "warnings": warnings,
        }),
    );
    metrics::histogram!(
        "helix_qdrant_collection_info_phase_ms",
        "phase" => "total",
        "outcome" => if result.is_ok() { "ok" } else { "error" }
    )
    .record(request_started.elapsed().as_secs_f64() * 1000.0);
    result
}

/// GET /collections
pub fn handle_list_collections(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let names = input.collections.list_collections()?;
    let collections: Vec<serde_json::Value> = names
        .iter()
        .map(|n| serde_json::json!({"name": n}))
        .collect();
    json_ok(response, &serde_json::json!({"collections": collections}))
}

/// DELETE /collections/{name}
pub fn handle_delete_collection(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let name = match get_collection_name(input) {
        Some(n) => n,
        None => return json_err(response, 400, "Missing collection name"),
    };

    match input
        .replication
        .apply(ReplicatedMutation::DeleteCollection { name: name.clone() })
    {
        Ok(_) => json_ok(response, &true),
        Err(e) => json_err(response, 404, &e.to_string()),
    }
}

/// PATCH /collections/{name}
/// Update a collection by adding new named (dense or sparse) vectors.
/// Existing vector configs cannot be changed.
#[derive(Deserialize)]
struct UpdateCollectionRequest {
    #[serde(default)]
    vectors: HashMap<String, VectorParamsInput>,
    #[serde(default)]
    sparse_vectors: HashMap<String, SparseVectorParamsInput>,
    #[serde(default)]
    optimizer_config: Option<OptimizerConfigUpdate>,
}

#[derive(Deserialize)]
struct OptimizerConfigUpdate {
    /// Set to 0 to enter bulk mode (defer all indexing).
    /// Set to a positive value to override the indexing threshold.
    /// Set to null to reset to the global default.
    /// Omit `optimizer_config` entirely to leave the current value unchanged.
    #[serde(default)]
    indexing_threshold: Option<usize>,
}

pub fn handle_update_collection(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let name = match get_collection_name(input) {
        Some(n) => n,
        None => return json_err(response, 400, "Missing collection name"),
    };

    let req: UpdateCollectionRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_err(response, 400, &format!("Invalid JSON: {}", e)),
    };

    let mut dense_configs = HashMap::new();
    for (vec_name, params) in &req.vectors {
        let distance = match params.distance.to_lowercase().as_str() {
            "cosine" => DistanceMetric::Cosine,
            "dot" => DistanceMetric::Dot,
            "euclid" | "euclidean" => DistanceMetric::Euclid,
            _ => DistanceMetric::Cosine,
        };
        let quant = params
            .quantization
            .as_ref()
            .map(clone_vector_quantization_input)
            .or_else(|| {
                params
                    .quantization_config
                    .as_ref()
                    .and_then(|cfg| cfg.to_vector_input())
            });

        dense_configs.insert(
            vec_name.clone(),
            NamedVectorConfig {
                size: params.size,
                distance,
                spindle: parse_spindle_config(&quant),
            },
        );
    }

    let mut sparse_configs = HashMap::new();
    for (sp_name, sp_params) in &req.sparse_vectors {
        let modifier = match sp_params.modifier.as_deref() {
            Some("idf") | Some("IDF") | Some("Idf") => SparseModifier::Idf,
            _ => SparseModifier::None,
        };
        let full_scan_threshold = sp_params
            .index
            .as_ref()
            .map(|idx| idx.full_scan_threshold)
            .unwrap_or(5000);
        sparse_configs.insert(
            sp_name.clone(),
            SparseVectorConfig {
                full_scan_threshold,
                modifier,
                ..Default::default()
            },
        );
    }

    input
        .replication
        .apply(ReplicatedMutation::UpdateCollection {
            name: name.clone(),
            vectors: dense_configs,
            sparse_vectors: sparse_configs,
        })?;

    // Apply per-collection indexing threshold override (bulk mode support).
    // CE can PATCH with `{"optimizer_config": {"indexing_threshold": 0}}` before
    // bulk upload, then restore with a positive threshold later, or omit
    // `optimizer_config` entirely to leave the current mode unchanged.
    if let Some(opt_config) = &req.optimizer_config {
        if let Ok(storage) = input.collections.get_collection(&name) {
            match opt_config.indexing_threshold {
                Some(threshold) => {
                    let effective = if threshold == 0 { 0 } else { threshold };
                    storage.named_vectors.set_indexing_threshold(effective);
                    storage.with_write_txn(|txn| {
                        storage.set_indexing_threshold_override_metadata(txn, Some(effective))?;
                        Ok(())
                    })?;
                }
                None => {
                    storage.named_vectors.set_indexing_threshold(usize::MAX);
                    storage.with_write_txn(|txn| {
                        storage.set_indexing_threshold_override_metadata(txn, None)?;
                        Ok(())
                    })?;
                }
            }
        }
    }

    json_ok(response, &true)
}

// ─── Point Operations ───

#[derive(Deserialize)]
struct UpsertPointsRequest {
    points: Vec<PointInput>,
}

/// Point input where `vector` can contain both dense (array of floats) and
/// sparse (object with indices/values) vectors under the same map.
/// We accept a raw JSON map and split them into dense vs sparse at parse time.
#[derive(Clone, Deserialize)]
struct PointInput {
    id: serde_json::Value, // String or integer
    #[serde(default)]
    vector: HashMap<String, serde_json::Value>,
    #[serde(default)]
    payload: HashMap<String, serde_json::Value>,
}

#[derive(Clone)]
struct PendingPoint {
    input: PointInput,
    recorded_at: Instant,
    approx_bytes: usize,
}

static WAL_AHEAD_PENDING_POINTS: OnceLock<RwLock<HashMap<String, HashMap<u128, PendingPoint>>>> =
    OnceLock::new();

#[cfg(test)]
static WAL_AHEAD_PENDING_TEST_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn wal_ahead_pending_points() -> &'static RwLock<HashMap<String, HashMap<u128, PendingPoint>>> {
    WAL_AHEAD_PENDING_POINTS.get_or_init(|| RwLock::new(HashMap::new()))
}

#[cfg(test)]
fn enable_wal_ahead_pending_for_test() {
    WAL_AHEAD_PENDING_TEST_ENABLED.store(true, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(test)]
fn collection_from_points_path(path: &str) -> Option<&str> {
    let path = path.split('?').next().unwrap_or(path);
    let rest = path.strip_prefix("/collections/")?;
    let (name, tail) = rest.split_once('/')?;
    if !name.is_empty() && tail == "points" {
        Some(name)
    } else {
        None
    }
}

/// Master kill-switch for the WAL-ahead in-memory pending cache.
///
/// Defaults to **off**. The cache duplicates the durable WAL spool in
/// process RAM purely to give read-your-writes visibility while the
/// background apply catches up. Most workloads (vector search, batch
/// ingest) don't actually depend on read-your-writes within the drain
/// window — the WAL is durable and reads see committed LMDB state — so
/// the cache is pure RAM tax. Cached at first read; toggle requires a
/// pod restart, which is also when the cache is empty anyway.
fn wal_ahead_pending_enabled() -> bool {
    #[cfg(test)]
    if WAL_AHEAD_PENDING_TEST_ENABLED.load(std::sync::atomic::Ordering::SeqCst) {
        return true;
    }

    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HELIX_WAL_AHEAD_PENDING_ENABLED")
            .ok()
            .map(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes"))
            .unwrap_or(false)
    })
}

fn wal_ahead_pending_max_points() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HELIX_WAL_AHEAD_PENDING_MAX_POINTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(100_000)
    })
}

fn wal_ahead_pending_max_bytes() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HELIX_WAL_AHEAD_PENDING_MAX_BYTES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(512 * 1024 * 1024)
    })
}

fn wal_ahead_pending_max_age() -> Duration {
    static CACHED: OnceLock<Duration> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let secs = std::env::var("HELIX_WAL_AHEAD_PENDING_MAX_AGE_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(600);
        Duration::from_secs(secs)
    })
}

#[cfg(test)]
fn pending_point_approx_bytes(point: &PointInput) -> usize {
    serde_json::to_vec(&point.id).map(|v| v.len()).unwrap_or(16)
        + serde_json::to_vec(&point.vector)
            .map(|v| v.len())
            .unwrap_or_default()
        + serde_json::to_vec(&point.payload)
            .map(|v| v.len())
            .unwrap_or_default()
}

fn prune_wal_ahead_pending_locked(pending: &mut HashMap<String, HashMap<u128, PendingPoint>>) {
    let now = Instant::now();
    let max_age = wal_ahead_pending_max_age();

    pending.retain(|_, points| {
        points.retain(|_, point| now.duration_since(point.recorded_at) <= max_age);
        !points.is_empty()
    });

    let mut entries: Vec<(String, u128, Instant, usize)> = pending
        .iter()
        .flat_map(|(collection, points)| {
            points.iter().map(|(id, point)| {
                (
                    collection.clone(),
                    *id,
                    point.recorded_at,
                    point.approx_bytes,
                )
            })
        })
        .collect();

    let max_points = wal_ahead_pending_max_points();
    let max_bytes = wal_ahead_pending_max_bytes();
    let mut total_bytes = entries.iter().map(|(_, _, _, bytes)| *bytes).sum::<usize>();

    if entries.len() <= max_points && total_bytes <= max_bytes {
        return;
    }

    entries.sort_by_key(|(_, _, recorded_at, _)| *recorded_at);
    let mut total_points = entries.len();
    for (collection, id, _, bytes) in entries {
        if total_points <= max_points && total_bytes <= max_bytes {
            break;
        }
        if let Some(points) = pending.get_mut(&collection) {
            if points.remove(&id).is_some() {
                total_points = total_points.saturating_sub(1);
                total_bytes = total_bytes.saturating_sub(bytes);
            }
            if points.is_empty() {
                pending.remove(&collection);
            }
        }
    }
}

fn prune_wal_ahead_pending_points() {
    let Ok(mut pending) = wal_ahead_pending_points().write() else {
        return;
    };
    prune_wal_ahead_pending_locked(&mut pending);
}

#[cfg(test)]
pub(crate) fn record_wal_ahead_points(path: &str, body: &[u8]) {
    if !wal_ahead_pending_enabled() {
        return;
    }
    let Some(collection) = collection_from_points_path(path) else {
        return;
    };
    let Ok(req) = sonic_rs::from_slice::<UpsertPointsRequest>(body) else {
        return;
    };
    let Ok(mut pending) = wal_ahead_pending_points().write() else {
        return;
    };
    prune_wal_ahead_pending_locked(&mut pending);
    let by_collection = pending.entry(collection.to_string()).or_default();
    for point in req.points {
        let approx_bytes = pending_point_approx_bytes(&point);
        by_collection.insert(
            point_id_to_u128(&point.id),
            PendingPoint {
                input: point,
                recorded_at: Instant::now(),
                approx_bytes,
            },
        );
    }
    prune_wal_ahead_pending_locked(&mut pending);
}

#[cfg(test)]
pub(crate) fn clear_wal_ahead_points(path: &str, body: &[u8]) {
    if !wal_ahead_pending_enabled() {
        return;
    }
    let Some(collection) = collection_from_points_path(path) else {
        return;
    };
    let Ok(req) = sonic_rs::from_slice::<UpsertPointsRequest>(body) else {
        return;
    };
    let Ok(mut pending) = wal_ahead_pending_points().write() else {
        return;
    };
    let Some(by_collection) = pending.get_mut(collection) else {
        return;
    };
    for point in req.points {
        by_collection.remove(&point_id_to_u128(&point.id));
    }
    if by_collection.is_empty() {
        pending.remove(collection);
    }
}

fn pending_point(collection: &str, id: u128) -> Option<PendingPoint> {
    if !wal_ahead_pending_enabled() {
        return None;
    }
    prune_wal_ahead_pending_points();
    wal_ahead_pending_points().read().ok().and_then(|pending| {
        pending
            .get(collection)
            .and_then(|points| points.get(&id).cloned())
    })
}

fn pending_points_for_collection(collection: &str) -> Vec<(u128, PendingPoint)> {
    if !wal_ahead_pending_enabled() {
        return Vec::new();
    }
    prune_wal_ahead_pending_points();
    let mut points: Vec<(u128, PendingPoint)> = wal_ahead_pending_points()
        .read()
        .ok()
        .and_then(|pending| pending.get(collection).cloned())
        .unwrap_or_default()
        .into_iter()
        .collect();
    points.sort_by_key(|(id, _)| *id);
    points
}

fn clear_pending_ids(collection: &str, ids: &[u128]) {
    if !wal_ahead_pending_enabled() {
        return;
    }
    let Ok(mut pending) = wal_ahead_pending_points().write() else {
        return;
    };
    let Some(by_collection) = pending.get_mut(collection) else {
        return;
    };
    for id in ids {
        by_collection.remove(id);
    }
    if by_collection.is_empty() {
        pending.remove(collection);
    }
}

fn pending_ids_matching_filter(collection: &str, filter: &Filter) -> Vec<u128> {
    pending_points_for_collection(collection)
        .into_iter()
        .filter_map(|(id, point)| {
            if pending_matches_filter(id, &point, filter) {
                Some(id)
            } else {
                None
            }
        })
        .collect()
}

fn count_exact_index_candidates_with_pending(
    candidates: &HashSet<u128>,
    pending_points: &[(u128, PendingPoint)],
    filter: &Filter,
) -> u64 {
    let pending_existing = pending_points
        .iter()
        .filter(|(id, _)| candidates.contains(id))
        .count() as u64;
    let pending_matching = pending_points
        .iter()
        .filter(|(id, point)| pending_matches_filter(*id, point, filter))
        .count() as u64;

    (candidates.len() as u64)
        .saturating_sub(pending_existing)
        .saturating_add(pending_matching)
}

fn pending_payload_json(pending: &PendingPoint) -> serde_json::Value {
    serde_json::Value::Object(
        pending
            .input
            .payload
            .clone()
            .into_iter()
            .collect::<serde_json::Map<_, _>>(),
    )
}

fn pending_vector_json(pending: &PendingPoint, vec_names: &[String]) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    for name in vec_names {
        if let Some(value @ serde_json::Value::Array(_)) = pending.input.vector.get(name) {
            out.insert(name.clone(), value.clone());
        }
    }
    serde_json::Value::Object(out)
}

fn pending_payload_value_map(pending: &PendingPoint) -> HashMap<String, Value> {
    json_payload_to_value_map(&pending.input.payload)
}

fn pending_matches_filter(id: u128, pending: &PendingPoint, filter: &Filter) -> bool {
    filter.is_empty() || filter.matches_point(Some(id), &pending_payload_value_map(pending))
}

fn build_pending_scan_point(
    id: u128,
    pending: &PendingPoint,
    with_payload: &WithPayload,
    hydrate_vectors: bool,
    vec_names: &[String],
) -> serde_json::Value {
    let mut point = serde_json::json!({
        "id": format_point_id(id),
    });

    if with_payload.is_active() {
        point["payload"] = with_payload.project(pending_payload_json(pending));
    }

    if hydrate_vectors {
        point["vector"] = pending_vector_json(pending, vec_names);
    }

    point
}

fn pending_dense_vector(pending: &PendingPoint, vec_name: &str) -> Option<Vec<f32>> {
    let serde_json::Value::Array(values) = pending.input.vector.get(vec_name)? else {
        return None;
    };
    Some(
        values
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect(),
    )
}

fn cosine_similarity(lhs: &[f32], rhs: &[f32]) -> Option<f32> {
    if lhs.len() != rhs.len() || lhs.is_empty() {
        return None;
    }
    let mut dot = 0.0f32;
    let mut lhs_norm = 0.0f32;
    let mut rhs_norm = 0.0f32;
    for (l, r) in lhs.iter().zip(rhs.iter()) {
        dot += l * r;
        lhs_norm += l * l;
        rhs_norm += r * r;
    }
    if lhs_norm <= f32::EPSILON || rhs_norm <= f32::EPSILON {
        return None;
    }
    Some(dot / (lhs_norm.sqrt() * rhs_norm.sqrt()))
}

fn pending_sparse_vector(pending: &PendingPoint, vec_name: &str) -> Option<SparseVector> {
    let serde_json::Value::Object(obj) = pending.input.vector.get(vec_name)? else {
        return None;
    };
    let indices: Vec<u32> = obj
        .get("indices")?
        .as_array()?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as u32)
        .collect();
    let values: Vec<f32> = obj
        .get("values")?
        .as_array()?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
        .collect();
    if indices.len() != values.len() || indices.is_empty() {
        return None;
    }
    Some(SparseVector { indices, values })
}

fn sparse_dot(lhs: &SparseVector, rhs: &SparseVector) -> f64 {
    let mut rhs_values: HashMap<u32, f32> = HashMap::with_capacity(rhs.indices.len());
    for (idx, value) in rhs.indices.iter().zip(rhs.values.iter()) {
        rhs_values.insert(*idx, *value);
    }
    lhs.indices
        .iter()
        .zip(lhs.values.iter())
        .filter_map(|(idx, value)| {
            rhs_values
                .get(idx)
                .map(|rhs| (*value as f64) * (*rhs as f64))
        })
        .sum()
}

fn fuse_pending_dense_results(
    collection: &str,
    filter: &Filter,
    vec_name: &str,
    query_vec: &[f32],
    limit: usize,
    results: &mut Vec<HVector>,
) {
    let mut pending_ids = HashSet::new();
    for (id, pending) in pending_points_for_collection(collection) {
        if !pending_matches_filter(id, &pending, filter) {
            continue;
        }
        let Some(vector) = pending_dense_vector(&pending, vec_name) else {
            continue;
        };
        let Some(score) = cosine_similarity(query_vec, &vector) else {
            continue;
        };
        pending_ids.insert(id);
        let mut hvec = HVector::new(id, vector);
        hvec.distance = Some(score);
        results.push(hvec);
    }
    if pending_ids.is_empty() {
        return;
    }
    results.retain(|hvec| {
        pending_ids.contains(&hvec.id) || pending_point(collection, hvec.id).is_none()
    });
    results.sort_by(|a, b| {
        b.distance
            .unwrap_or(0.0)
            .partial_cmp(&a.distance.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    results.truncate(limit);
}

fn fuse_pending_sparse_results(
    collection: &str,
    filter: &Filter,
    vec_name: &str,
    query: &SparseVector,
    limit: usize,
    results: &mut Vec<(u128, f64)>,
) {
    let mut pending_ids = HashSet::new();
    for (id, pending) in pending_points_for_collection(collection) {
        if !pending_matches_filter(id, &pending, filter) {
            continue;
        }
        let Some(vector) = pending_sparse_vector(&pending, vec_name) else {
            continue;
        };
        let score = sparse_dot(query, &vector);
        if score <= 0.0 {
            continue;
        }
        pending_ids.insert(id);
        results.push((id, score));
    }
    if pending_ids.is_empty() {
        return;
    }
    results.retain(|(id, _)| pending_ids.contains(id) || pending_point(collection, *id).is_none());
    results.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    results.truncate(limit);
}

/// Split mixed vector map into dense and sparse.
fn split_vectors(
    raw: &HashMap<String, serde_json::Value>,
) -> Result<(HashMap<String, Vec<f32>>, HashMap<String, SparseVector>), String> {
    let mut dense = HashMap::new();
    let mut sparse = HashMap::new();
    for (name, val) in raw {
        match val {
            serde_json::Value::Array(arr) => {
                let floats: Vec<f32> = arr
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect();
                dense.insert(name.clone(), floats);
            }
            serde_json::Value::Object(obj) => {
                if let (Some(indices_val), Some(values_val)) =
                    (obj.get("indices"), obj.get("values"))
                {
                    let indices: Vec<u32> = indices_val
                        .as_array()
                        .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as u32).collect())
                        .unwrap_or_default();
                    let values: Vec<f32> = values_val
                        .as_array()
                        .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect())
                        .unwrap_or_default();
                    sparse.insert(name.clone(), SparseVector { indices, values });
                } else {
                    return Err(format!(
                        "Vector '{}': object must have 'indices' and 'values' fields",
                        name
                    ));
                }
            }
            _ => {
                return Err(format!(
                    "Vector '{}': expected array (dense) or object (sparse)",
                    name
                ));
            }
        }
    }
    Ok((dense, sparse))
}

fn estimate_value_bytes(value: &Value) -> usize {
    match value {
        Value::String(s) => s.len().saturating_add(16),
        Value::Array(items) => items
            .iter()
            .map(estimate_value_bytes)
            .sum::<usize>()
            .saturating_add(16),
        Value::Object(items) => items
            .iter()
            .map(|(key, value)| key.len().saturating_add(estimate_value_bytes(value)))
            .sum::<usize>()
            .saturating_add(32),
        Value::U128(_) => 16,
        Value::I64(_) | Value::U64(_) | Value::F64(_) => 8,
        Value::I32(_) | Value::U32(_) | Value::F32(_) => 4,
        Value::I16(_) | Value::U16(_) => 2,
        Value::I8(_) | Value::U8(_) | Value::Boolean(_) => 1,
        Value::Empty => 0,
    }
}

fn estimate_upsert_headroom_bytes(body_len: usize, points: &[ReplicatedPoint]) -> usize {
    let min_headroom = std::env::var("HELIX_UPSERT_HEADROOM_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v >= 1)
        .unwrap_or(8)
        .saturating_mul(1024)
        .saturating_mul(1024);
    let vectors = points
        .iter()
        .flat_map(|point| point.vectors.values())
        .map(|vector| vector.len().saturating_mul(std::mem::size_of::<f32>()))
        .sum::<usize>();
    let sparse = points
        .iter()
        .flat_map(|point| point.sparse_vectors.values())
        .map(|vector| {
            vector
                .indices
                .len()
                .saturating_add(vector.values.len())
                .saturating_mul(std::mem::size_of::<f32>())
        })
        .sum::<usize>();
    let payload = points
        .iter()
        .map(|point| {
            point
                .payload
                .iter()
                .map(|(key, value)| key.len().saturating_add(estimate_value_bytes(value)))
                .sum::<usize>()
        })
        .sum::<usize>();

    // Include the JSON request body and over-provision for LMDB pages,
    // metadata, payload indices, vector sidecars, and HNSW adjacency.
    body_len
        .saturating_add(vectors)
        .saturating_add(sparse)
        .saturating_add(payload)
        .saturating_mul(3)
        .max(min_headroom)
}

/// PUT /collections/{name}/points
pub fn handle_upsert_points(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let name = match get_collection_name(input) {
        Some(n) => n,
        None => return json_err(response, 400, "Missing collection name"),
    };

    // Phase-level instrumentation paired with `helix_apply_upsert_phase_ms` in
    // replication.rs. Bounds the gap between gateway-level `chunk_commit_ms`
    // (~30 s p50) and inner LMDB work (~150 ms p50).
    let phase_start = std::time::Instant::now();
    let req: UpsertPointsRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_err(response, 400, &format!("Invalid JSON: {}", e)),
    };

    // NOTE: No local get_collection check here. In cluster mode followers can lag
    // behind the leader. The Raft apply path on the leader validates collection
    // existence; rejecting here would cause spurious 404s on followers.

    let mut points = Vec::with_capacity(req.points.len());
    for point in req.points {
        let (dense, sparse) = match split_vectors(&point.vector) {
            Ok(v) => v,
            Err(e) => return json_err(response, 400, &e),
        };
        points.push(ReplicatedPoint {
            id: point_id_to_u128(&point.id),
            vectors: dense,
            sparse_vectors: sparse,
            payload: json_payload_to_value_map(&point.payload),
        });
    }
    let observe_hot_metrics = crate::telemetry::hot_path_metrics_enabled();
    if observe_hot_metrics {
        metrics::histogram!("helix_qdrant_upsert_phase_ms", "phase" => "parse_body")
            .record(phase_start.elapsed().as_secs_f64() * 1000.0);
    }

    let phase_start = std::time::Instant::now();
    let estimated_headroom = estimate_upsert_headroom_bytes(input.request.body.len(), &points);
    if let Some(storage) = input.collections.get_loaded_collection(&name)? {
        if let Err(err) = storage.ensure_map_headroom(estimated_headroom) {
            if matches!(err, GraphError::ResizeBackpressure(_)) {
                response
                    .headers
                    .insert("Retry-After".to_string(), "1".to_string());
                return json_err(response, 503, &err.to_string());
            }
            tracing::debug!(
                collection = %name,
                estimated_headroom,
                error = %err,
                "pre-upsert ensure_map_headroom failed; proceeding with retry-capable write path"
            );
        }
    }
    if observe_hot_metrics {
        metrics::histogram!("helix_qdrant_upsert_phase_ms", "phase" => "ensure_headroom")
            .record(phase_start.elapsed().as_secs_f64() * 1000.0);
    }

    let mutation = ReplicatedMutation::UpsertPoints {
        collection: name.clone(),
        points,
    };
    let phase_start = std::time::Instant::now();
    match input.replication.apply(mutation.clone()) {
        Ok(()) => {}
        Err(GraphError::MapFull) => {
            if let Ok(storage) = input.collections.get_collection(&name) {
                storage.grow_map()?;
            }
            input.replication.apply(mutation)?;
        }
        Err(err) if err.is_retryable_lmdb_invalid_argument() => {
            tracing::warn!(
                collection = %name,
                error = %err,
                "retrying upsert after retryable LMDB invalid argument"
            );
            if let Ok(storage) = input.collections.get_collection(&name) {
                storage.grow_map()?;
            }
            input.replication.apply(mutation)?;
        }
        Err(err) => {
            if let Ok(storage) = input.collections.get_collection(&name) {
                if let Ok(env) = storage.lmdb_env() {
                    let info = env.info();
                    let page_size = page_size::get();
                    let used_bytes = info.last_page_number.saturating_mul(page_size);
                    tracing::error!(
                        collection = %name,
                        error = %err,
                        map_size_bytes = info.map_size,
                        used_bytes,
                        page_size,
                        last_page_number = info.last_page_number,
                        estimated_headroom,
                        "upsert apply failed after headroom preflight"
                    );
                } else {
                    tracing::error!(
                        collection = %name,
                        error = %err,
                        estimated_headroom,
                        "upsert apply failed after headroom preflight; LMDB stats unavailable"
                    );
                }
            } else {
                tracing::error!(
                    collection = %name,
                    error = %err,
                    estimated_headroom,
                    "upsert apply failed after headroom preflight; collection stats unavailable"
                );
            }
            return Err(err);
        }
    }
    if observe_hot_metrics {
        metrics::histogram!("helix_qdrant_upsert_phase_ms", "phase" => "replication_apply")
            .record(phase_start.elapsed().as_secs_f64() * 1000.0);
    }
    json_ok(response, &serde_json::json!({"status": "completed"}))
}

// ─── Snapshots ───

#[derive(Deserialize)]
struct RecoverSnapshotRequest {
    #[serde(alias = "name")]
    location: String,
}

fn snapshot_result(
    snapshot: &crate::helix_engine::storage_core::collection_manager::SnapshotInfo,
) -> serde_json::Value {
    serde_json::json!({
        "name": snapshot.name.clone(),
        "creation_time": snapshot.created_at_millis,
        "size": snapshot.disk_bytes,
        "lsn": snapshot.lsn,
        // Keep `location` for compatibility, but expose a snapshot identifier rather than a host path.
        "location": snapshot.name.clone(),
    })
}

/// GET /collections/{name}/snapshots
pub fn handle_list_snapshots(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let name = match get_collection_name(input) {
        Some(n) => n,
        None => return json_err(response, 400, "Missing collection name"),
    };

    match input.collections.list_snapshots(&name) {
        Ok(snapshots) => {
            let result: Vec<serde_json::Value> = snapshots.iter().map(snapshot_result).collect();
            json_ok(response, &result)
        }
        Err(e) => json_err(response, 404, &e.to_string()),
    }
}

/// POST /collections/{name}/snapshots
pub fn handle_create_snapshot(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let name = match get_collection_name(input) {
        Some(n) => n,
        None => return json_err(response, 400, "Missing collection name"),
    };

    match input.collections.snapshot_collection_async(&name) {
        Ok(snapshot) => json_ok(response, &snapshot_result(&snapshot)),
        Err(e) => json_err(response, 500, &e.to_string()),
    }
}

/// PUT /collections/{name}/snapshots/recover
pub fn handle_recover_snapshot(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let name = match get_collection_name(input) {
        Some(n) => n,
        None => return json_err(response, 400, "Missing collection name"),
    };

    if input.replication.status()?.enabled {
        return json_err(
            response,
            409,
            "Snapshot recovery is not supported through the HTTP API while Raft replication is enabled",
        );
    }

    let req: RecoverSnapshotRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_err(response, 400, &format!("Invalid JSON: {}", e)),
    };

    match input
        .collections
        .restore_collection_from_snapshot(&name, &req.location)
    {
        Ok(snapshot) => json_ok(
            response,
            &serde_json::json!({
                "status": "completed",
                "snapshot": snapshot_result(&snapshot),
            }),
        ),
        Err(e) => json_err(response, 500, &e.to_string()),
    }
}

// ─── Search ───

#[derive(Deserialize)]
struct SearchRequest {
    vector: SearchVector,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    offset: usize,
    #[serde(default)]
    score_threshold: Option<f32>,
    #[serde(default)]
    filter: Option<Filter>,
    #[serde(default = "default_true")]
    with_payload: bool,
    /// Accepts `false | true | ["name"]`. See `WithVectors` docs.
    #[serde(default, alias = "with_vector")]
    with_vectors: WithVectors,
    #[serde(default)]
    params: Option<SearchParams>,
}

/// Qdrant-compatible search_params. We honour `hnsw_ef`; other fields are
/// accepted for compatibility but ignored (quantization is handled by Spindle).
#[derive(Deserialize, Default)]
struct SearchParams {
    #[serde(default)]
    hnsw_ef: Option<usize>,
    #[serde(default, rename = "quantization")]
    _quantization: Option<serde_json::Value>, // accepted, ignored
}

#[derive(Deserialize)]
#[serde(untagged)]
enum SearchVector {
    NamedSparse {
        name: String,
        vector: SparseVectorInput,
    },
    Named {
        name: String,
        vector: Vec<f32>,
    },
    Raw(Vec<f32>),
}

#[derive(Deserialize)]
struct SparseVectorInput {
    indices: Vec<u32>,
    values: Vec<f32>,
}

fn default_limit() -> usize {
    10
}
fn default_true() -> bool {
    true
}

/// Hard cap for any client-supplied `limit` field on read endpoints.
/// Prevents DoS via heap allocations (e.g. `limit: 10_000_000` triggers a
/// large `Vec::truncate` and result-buffer growth). Override with
/// `HELIX_MAX_SEARCH_LIMIT`. Cached on first call.
fn max_search_limit() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HELIX_MAX_SEARCH_LIMIT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(10_000)
    })
}

/// Clamp a client-supplied limit to the configured maximum. Zero is treated
/// as the default (10) to match Qdrant behaviour where omitted limit ⇒ 10.
fn clamp_limit(limit: usize) -> usize {
    let cap = max_search_limit();
    if limit == 0 {
        default_limit()
    } else {
        limit.min(cap)
    }
}

fn clamp_offset(offset: usize) -> usize {
    offset.min(max_search_limit())
}

fn result_window(limit: usize, offset: usize) -> usize {
    limit.saturating_add(offset).min(max_search_limit())
}

fn channel_result_window(channel_limit: usize, result_window: usize, offset: usize) -> usize {
    if offset > 0 {
        channel_limit.max(result_window).min(max_search_limit())
    } else {
        channel_limit
    }
}

fn validate_score_threshold(threshold: Option<f32>) -> Result<(), &'static str> {
    if threshold.is_some_and(|score| !score.is_finite()) {
        Err("score_threshold must be finite")
    } else {
        Ok(())
    }
}

fn dense_score_threshold_for_metric(
    storage: &HelixGraphStorage,
    vector_name: &str,
    threshold: Option<f32>,
) -> Option<f32> {
    let threshold = threshold?;
    let Some(config) = storage.named_vectors.get_config(vector_name) else {
        return Some(threshold);
    };
    match config.distance {
        DistanceMetric::Euclid => Some(-threshold),
        DistanceMetric::Cosine | DistanceMetric::Dot => Some(threshold),
    }
}

fn score_passes_threshold(score: f32, threshold: Option<f32>) -> bool {
    threshold.is_none_or(|min_score| score >= min_score)
}

fn apply_hvector_score_threshold_and_offset(
    results: &mut Vec<HVector>,
    threshold: Option<f32>,
    offset: usize,
    limit: usize,
) {
    results.retain(|result| score_passes_threshold(result.distance.unwrap_or(0.0), threshold));
    if offset > 0 {
        if offset >= results.len() {
            results.clear();
        } else {
            results.drain(..offset);
        }
    }
    results.truncate(limit);
}

fn apply_sparse_score_threshold_and_offset(
    results: &mut Vec<(u128, f64)>,
    threshold: Option<f32>,
    offset: usize,
    limit: usize,
) {
    results.retain(|(_, score)| score_passes_threshold(*score as f32, threshold));
    if offset > 0 {
        if offset >= results.len() {
            results.clear();
        } else {
            results.drain(..offset);
        }
    }
    results.truncate(limit);
}

fn apply_ranked_score_threshold_and_offset(
    results: &mut Vec<RankedItem>,
    threshold: Option<f32>,
    offset: usize,
    limit: usize,
) {
    results.retain(|item| score_passes_threshold(item.score as f32, threshold));
    if offset > 0 {
        if offset >= results.len() {
            results.clear();
        } else {
            results.drain(..offset);
        }
    }
    results.truncate(limit);
}

fn dense_exact_index_candidate_max() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HELIX_DENSE_EXACT_INDEX_CANDIDATE_MAX")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(2048)
    })
}

fn should_exact_score_index_candidates_with_limit(
    candidates: &HashSet<u128>,
    total_vectors: u64,
    max_exact_candidates: usize,
) -> bool {
    let len = candidates.len();
    if len <= max_exact_candidates {
        return true;
    }

    // Allow a small selectivity bonus for highly selective indexed filters, but
    // keep an absolute request-time scoring ceiling on very large collections.
    let selectivity_cap = max_exact_candidates.saturating_mul(8);
    total_vectors > 0 && len <= selectivity_cap && (len as u64).saturating_mul(20) <= total_vectors
}

fn should_exact_score_index_candidates(candidates: &HashSet<u128>, total_vectors: u64) -> bool {
    should_exact_score_index_candidates_with_limit(
        candidates,
        total_vectors,
        dense_exact_index_candidate_max(),
    )
}

fn collection_vector_count(storage: &HelixGraphStorage, txn: &heed3::RoTxn) -> u64 {
    storage
        .get_metadata(txn)
        .map(|metadata| metadata.stats.vector_count)
        .unwrap_or(0)
}

fn collection_vector_count_be(storage: &HelixGraphStorage) -> u64 {
    storage
        .metadata_snapshot()
        .map(|metadata| metadata.stats.vector_count)
        .unwrap_or(0)
}

fn exact_dense_candidate_ids_be(
    storage: &HelixGraphStorage,
    r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    indexed_candidates: Option<&HashSet<u128>>,
    filter: &Filter,
    recheck_filter: bool,
    total_vectors: u64,
) -> Option<Vec<u128>> {
    bounded_filter_candidate_ids_be(
        storage,
        r,
        indexed_candidates,
        filter,
        recheck_filter,
        total_vectors,
    )
}

fn exact_index_filter_candidates<'a>(
    has_filter: bool,
    filter_recheck_required: bool,
    indexed_candidates: Option<&'a HashSet<u128>>,
) -> Option<&'a HashSet<u128>> {
    if has_filter && !filter_recheck_required {
        indexed_candidates
    } else {
        None
    }
}

fn bounded_filter_candidate_ids_be(
    storage: &HelixGraphStorage,
    r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    indexed_candidates: Option<&HashSet<u128>>,
    filter: &Filter,
    recheck_filter: bool,
    total_vectors: u64,
) -> Option<Vec<u128>> {
    indexed_candidates
        .filter(|candidates| should_exact_score_index_candidates(candidates, total_vectors))
        .and_then(|_| {
            filter_candidate_ids_be(storage, r, indexed_candidates, filter, recheck_filter)
        })
}

fn filter_candidate_ids_be(
    storage: &HelixGraphStorage,
    r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    indexed_candidates: Option<&HashSet<u128>>,
    filter: &Filter,
    recheck_filter: bool,
) -> Option<Vec<u128>> {
    indexed_candidates.map(|candidates| {
        candidates
            .iter()
            .copied()
            .filter(|id| {
                if !recheck_filter {
                    return true;
                }
                match storage.get_node_be(r, id) {
                    Ok(node) => filter.matches_point(Some(*id), &node.properties),
                    Err(_) => false,
                }
            })
            .collect()
    })
}

/// Dense-search analogue of `PointScanPlan`: which arm of the filtered
/// dense-search planner served the request.
#[derive(Debug, Clone, Copy)]
enum DenseSearchPlan {
    ExactCandidates,
    CandidateSetHnsw,
    FilteredHnsw,
    Unfiltered,
    ProbeSkippedIndex,
}

impl DenseSearchPlan {
    fn as_str(self) -> &'static str {
        match self {
            DenseSearchPlan::ExactCandidates => "exact_candidates",
            DenseSearchPlan::CandidateSetHnsw => "candidate_set_hnsw",
            DenseSearchPlan::FilteredHnsw => "filtered_hnsw",
            DenseSearchPlan::Unfiltered => "unfiltered",
            DenseSearchPlan::ProbeSkippedIndex => "probe_skipped_index",
        }
    }
}

/// Mirrors the dispatch order of the dense-search decision trees: exact
/// scoring over candidates, candidate-set-gated HNSW, then in-traversal
/// filtered HNSW (labelled `ProbeSkippedIndex` when the cardinality probe
/// skipped candidate materialization), else unfiltered HNSW.
fn select_dense_search_plan(
    exact_candidates: bool,
    candidate_set_hnsw: bool,
    probe_skipped: bool,
    dense_filter_active: bool,
) -> DenseSearchPlan {
    if exact_candidates {
        DenseSearchPlan::ExactCandidates
    } else if candidate_set_hnsw {
        DenseSearchPlan::CandidateSetHnsw
    } else if probe_skipped {
        DenseSearchPlan::ProbeSkippedIndex
    } else if dense_filter_active {
        DenseSearchPlan::FilteredHnsw
    } else {
        DenseSearchPlan::Unfiltered
    }
}

fn observe_dense_search_plan(
    collection: &str,
    plan: DenseSearchPlan,
    probe_estimate: Option<usize>,
) {
    if let Some(estimate) = probe_estimate {
        tracing::debug!(
            collection = collection,
            plan = plan.as_str(),
            estimated_cardinality = estimate,
            "dense search filter count probe"
        );
    }
    if !crate::telemetry::hot_path_metrics_enabled() {
        return;
    }
    metrics::counter!(
        "helix_dense_search_requests_total",
        "collection" => collection.to_string(),
        "plan" => plan.as_str(),
    )
    .increment(1);
}

/// Candidate resolution for the LSM `/points/search` branch with the
/// pre-materialization cardinality probe. This site historically materialized
/// the candidate set unconditionally (no post-materialization drop), so the
/// probe applies the same broadness threshold as
/// `vector_query_indexed_filter_candidates` before paying for
/// materialization; when it skips, downstream falls to the in-traversal
/// filtered arm with the `matches_point` recheck (see
/// `indexed_filter_must_count_estimate_be` for the upper-bound trade-off).
fn search_indexed_filter_candidates_be(
    storage: &HelixGraphStorage,
    r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    filter: &Filter,
    probe_enabled: bool,
) -> Result<ProbedCandidates, GraphError> {
    let mut probe_estimate = None;
    if probe_enabled && !filter.must.is_empty() {
        let total = collection_vector_count_be(storage) as usize;
        if total > 0 {
            let cap = vector_query_broad_candidate_cap(total);
            probe_estimate = indexed_filter_must_count_estimate_be(storage, r, filter, cap)?;
            if let Some(estimate) = probe_estimate {
                if vector_query_candidates_too_broad(estimate, total) {
                    return Ok(ProbedCandidates {
                        candidates: None,
                        probe_estimate,
                        probe_skipped: true,
                    });
                }
            }
        }
    }
    Ok(ProbedCandidates {
        candidates: indexed_filter_candidates_be(storage, r, filter)?,
        probe_estimate,
        probe_skipped: false,
    })
}

/// POST /collections/{name}/points/search
pub fn handle_search_points(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let name = match get_collection_name(input) {
        Some(n) => n,
        None => return json_err(response, 400, "Missing collection name"),
    };

    let mut req: SearchRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_err(response, 400, &format!("Invalid JSON: {}", e)),
    };
    if let Err(message) = validate_score_threshold(req.score_threshold) {
        return json_err(response, 400, message);
    }
    req.limit = clamp_limit(req.limit);
    req.offset = clamp_offset(req.offset);
    let result_window = result_window(req.limit, req.offset);

    let storage = match input.collections.get_collection(&name) {
        Ok(s) => s,
        Err(e) => return json_err(response, 404, &e.to_string()),
    };

    if storage.backend.kind() == BackendKind::Lsm {
        let filter = req.filter.unwrap_or_default();
        let has_filter = !filter.is_empty();
        let r = storage
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))?;
        let ProbedCandidates {
            candidates: indexed_candidates,
            probe_estimate,
            probe_skipped,
        } = search_indexed_filter_candidates_be(
            &storage,
            &r,
            &filter,
            filter_count_probe_enabled(),
        )?;
        let filter_index_exact =
            indexed_candidates.is_some() && indexed_filter_candidates_are_exact(&storage, &filter);
        let filter_recheck_required = has_filter && !filter_index_exact;
        let total_vectors = indexed_candidates
            .as_ref()
            .map(|_| collection_vector_count_be(&storage))
            .unwrap_or(0);
        let filter_fn = |id: u128| -> bool {
            if let Some(candidates) = &indexed_candidates {
                if !candidates.contains(&id) {
                    return false;
                }
            }
            if !filter_recheck_required {
                return true;
            }
            match storage.get_node_be(&r, &id) {
                Ok(node) => filter.matches_point(Some(id), &node.properties),
                Err(_) => false,
            }
        };

        if let SearchVector::NamedSparse {
            name: using,
            vector,
        } = &req.vector
        {
            let query = SparseVector {
                indices: vector.indices.clone(),
                values: vector.values.clone(),
            };
            let search_limit = if has_filter || indexed_candidates.is_some() {
                result_window.saturating_mul(4)
            } else {
                result_window
            };
            let exact_candidate_ids = bounded_filter_candidate_ids_be(
                &storage,
                &r,
                indexed_candidates.as_ref(),
                &filter,
                filter_recheck_required,
                total_vectors,
            );
            let mut sparse_results = storage
                .named_vectors
                .with_sparse_core(using, |core| {
                    let mut results = if let Some(candidate_ids) = exact_candidate_ids.as_ref() {
                        core.search_candidate_ids_exact_be(
                            &r,
                            &query,
                            search_limit,
                            candidate_ids.iter().copied(),
                        )?
                    } else {
                        core.search_be(&r, &query, search_limit, Some(&filter_fn))?
                    };
                    results.truncate(result_window);
                    Ok(results)
                })
                .map_err(GraphError::from)?;
            fuse_pending_sparse_results(
                &name,
                &filter,
                using,
                &query,
                result_window,
                &mut sparse_results,
            );
            apply_sparse_score_threshold_and_offset(
                &mut sparse_results,
                req.score_threshold,
                req.offset,
                req.limit,
            );
            let mut results = Vec::new();
            for (id, score) in sparse_results {
                let node = match storage.get_node_be(&r, &id) {
                    Ok(n) => n,
                    Err(_) => continue,
                };
                let mut point = serde_json::json!({
                    "id": format_point_id(id),
                    "version": 0,
                    "score": score,
                });
                if req.with_payload {
                    point["payload"] = value_map_to_json(&node.properties);
                }
                if req.with_vectors.is_active() {
                    let vec_names = req
                        .with_vectors
                        .resolved_names(&known_vector_names(&storage));
                    point["vector"] = fetch_named_vectors_be(&storage, &r, id, &vec_names);
                }
                results.push(point);
            }
            return json_ok(response, &results);
        }

        let (vec_name, query_vec) = match &req.vector {
            SearchVector::Named { name, vector } => (name.clone(), vector.clone()),
            SearchVector::Raw(vector) => ("dense".into(), vector.clone()),
            SearchVector::NamedSparse { .. } => unreachable!(),
        };
        let score_threshold =
            dense_score_threshold_for_metric(&storage, &vec_name, req.score_threshold);
        let dense_filter_active = has_filter || indexed_candidates.is_some();
        let exact_index_filter_candidates = exact_index_filter_candidates(
            has_filter,
            filter_recheck_required,
            indexed_candidates.as_ref(),
        );
        let search_limit = if dense_filter_active {
            result_window.saturating_mul(4)
        } else {
            result_window
        };
        let selectivity_hint = if has_filter {
            indexed_candidates.as_ref().and_then(|candidates| {
                (total_vectors > 0).then_some(candidates.len() as f32 / total_vectors as f32)
            })
        } else {
            None
        };
        let ef_override = req.params.as_ref().and_then(|p| p.hnsw_ef);
        let exact_candidate_ids = exact_dense_candidate_ids_be(
            &storage,
            &r,
            indexed_candidates.as_ref(),
            &filter,
            filter_recheck_required,
            total_vectors,
        );
        let plan = select_dense_search_plan(
            exact_candidate_ids.is_some(),
            exact_index_filter_candidates.is_some(),
            probe_skipped,
            dense_filter_active,
        );
        observe_dense_search_plan(&name, plan, probe_estimate);
        let mut search_results = if let Some(candidate_ids) = exact_candidate_ids {
            storage.named_vectors.dense_search_candidate_ids_exact_be(
                &r,
                &vec_name,
                &query_vec,
                search_limit,
                candidate_ids.iter().copied(),
                selectivity_hint,
            )
        } else if let Some(candidate_ids) = exact_index_filter_candidates {
            storage
                .named_vectors
                .dense_search_candidate_set_filter_ef_be(
                    &r,
                    &vec_name,
                    &query_vec,
                    search_limit,
                    candidate_ids,
                    true,
                    selectivity_hint,
                    ef_override,
                )
        } else {
            storage.named_vectors.dense_search_with_id_filter_ef_be(
                &r,
                &vec_name,
                &query_vec,
                search_limit,
                dense_filter_active.then_some(&[filter_fn][..]),
                true,
                selectivity_hint,
                ef_override,
            )
        }
        .map_err(GraphError::from)?;
        search_results.truncate(result_window);
        for result in &mut search_results {
            if let Some(distance) = result.distance {
                result.distance = Some(
                    storage
                        .named_vectors
                        .public_score(&vec_name, distance)
                        .map_err(GraphError::from)?,
                );
            }
        }
        fuse_pending_dense_results(
            &name,
            &filter,
            &vec_name,
            &query_vec,
            result_window,
            &mut search_results,
        );
        apply_hvector_score_threshold_and_offset(
            &mut search_results,
            score_threshold,
            req.offset,
            req.limit,
        );

        let mut results: Vec<serde_json::Value> = Vec::new();
        for hvec in search_results {
            let node = match storage.get_node_be(&r, &hvec.id) {
                Ok(n) => n,
                Err(_) => continue,
            };
            let mut point = serde_json::json!({
                "id": format_point_id(hvec.id),
                "version": 0,
                "score": hvec.distance.unwrap_or(0.0),
            });
            if req.with_payload {
                point["payload"] = value_map_to_json(&node.properties);
            }
            results.push(point);
        }
        return json_ok(response, &results);
    }

    let txn = storage.begin_resize_safe_read_txn()?;
    let dense_read_txns = storage.nested_dense_read_provider(&txn);
    let filter = req.filter.unwrap_or_default();
    let ProbedCandidates {
        candidates: indexed_candidates,
        probe_estimate,
        probe_skipped,
    } = vector_query_indexed_filter_candidates_probed(
        &storage,
        &txn,
        &filter,
        filter_count_probe_enabled(),
    )?;
    let has_filter = !filter.is_empty();
    let pending_dense_query: Option<(String, Vec<f32>)> = match &req.vector {
        SearchVector::Named { name, vector } => Some((name.clone(), vector.clone())),
        SearchVector::Raw(vector) => Some(("dense".into(), vector.clone())),
        SearchVector::NamedSparse { .. } => None,
    };

    // Build a filter closure for both dense and sparse paths.
    let filter_fn = |id: u128| -> bool {
        if let Some(candidates) = &indexed_candidates {
            if !candidates.contains(&id) {
                return false;
            }
        }
        if !has_filter {
            return true;
        }
        match storage.get_node(&txn, &id) {
            Ok(node) => filter.matches_point(Some(id), &node.properties),
            Err(_) => false,
        }
    };

    // Branch: sparse vs dense search
    let search_results: Vec<HVector> = if let SearchVector::NamedSparse {
        name: sp_name,
        vector,
    } = &req.vector
    {
        // ── Sparse vector search (WAND) ──
        let sp_name = sp_name.clone();
        let sp_query = SparseVector {
            indices: vector.indices.clone(),
            values: vector.values.clone(),
        };
        let sparse_results = storage
            .named_vectors
            .with_sparse_core(&sp_name, |core| {
                let mut results = core.search(&txn, &sp_query, result_window, Some(&filter_fn))?;
                results.truncate(result_window);
                Ok(results)
            })
            .map_err(GraphError::from)?;
        // Convert (u128, f64) → HVector
        sparse_results
            .into_iter()
            .map(|(id, score)| {
                let mut v = HVector::new(id, vec![]);
                v.distance = Some(score as f32);
                v
            })
            .collect()
    } else {
        // ── Dense vector search (HNSW) ──
        let (vec_name, query_vec) = match &req.vector {
            SearchVector::Named { name, vector } => (name.clone(), vector.clone()),
            SearchVector::Raw(v) => ("dense".into(), v.clone()),
            SearchVector::NamedSparse { .. } => {
                // Unreachable in practice: matched in the `if` branch above.
                return json_err(response, 400, "invalid vector variant for dense search");
            }
        };
        let search_limit = if has_filter {
            result_window.saturating_mul(4)
        } else {
            result_window
        };
        let total_vectors = indexed_candidates
            .as_ref()
            .map(|_| collection_vector_count(&storage, &txn))
            .unwrap_or(0);
        let selectivity_hint: Option<f32> = if has_filter {
            if let Some(candidates) = &indexed_candidates {
                if total_vectors > 0 {
                    Some(candidates.len() as f32 / total_vectors as f32)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        let ef_override = req.params.as_ref().and_then(|p| p.hnsw_ef);
        let dense_filter_active = has_filter || indexed_candidates.is_some();
        let exact_candidate_ids = indexed_candidates
            .as_ref()
            .filter(|candidates| should_exact_score_index_candidates(candidates, total_vectors))
            .map(|candidates| {
                candidates
                    .iter()
                    .copied()
                    .filter(|id| filter_fn(*id))
                    .collect::<Vec<u128>>()
            });
        let plan = select_dense_search_plan(
            exact_candidate_ids.is_some(),
            false,
            probe_skipped,
            dense_filter_active,
        );
        observe_dense_search_plan(&name, plan, probe_estimate);
        let mut results = if let Some(candidate_ids) = exact_candidate_ids {
            storage.named_vectors.dense_search_candidate_ids_exact(
                &txn,
                &vec_name,
                &query_vec,
                search_limit,
                candidate_ids.iter().copied(),
                selectivity_hint,
            )
        } else if dense_filter_active {
            storage.named_vectors.dense_search_with_id_filter_ef(
                &txn,
                &vec_name,
                &query_vec,
                search_limit,
                Some(&[filter_fn]),
                true,
                selectivity_hint,
                ef_override,
            )
        } else {
            storage
                .named_vectors
                .dense_search_with_selectivity_ef_with_provider::<fn(&HVector) -> bool, _>(
                    &dense_read_txns,
                    &txn,
                    &vec_name,
                    &query_vec,
                    search_limit,
                    None,
                    true,
                    selectivity_hint,
                    ef_override,
                )
        }
        .map_err(GraphError::from)?;
        results.truncate(result_window);
        for result in &mut results {
            if let Some(distance) = result.distance {
                result.distance = Some(
                    storage
                        .named_vectors
                        .public_score(&vec_name, distance)
                        .map_err(GraphError::from)?,
                );
            }
        }
        results
    };

    let score_threshold = pending_dense_query
        .as_ref()
        .map_or(req.score_threshold, |(vec_name, _)| {
            dense_score_threshold_for_metric(&storage, vec_name, req.score_threshold)
        });
    let mut search_results = search_results;
    if let Some((vec_name, query_vec)) = pending_dense_query {
        fuse_pending_dense_results(
            &name,
            &filter,
            &vec_name,
            &query_vec,
            result_window,
            &mut search_results,
        );
    }
    apply_hvector_score_threshold_and_offset(
        &mut search_results,
        score_threshold,
        req.offset,
        req.limit,
    );

    let mut results: Vec<serde_json::Value> = Vec::new();

    let vec_names: Vec<String> = if req.with_vectors.is_active() {
        req.with_vectors
            .resolved_names(&known_vector_names(&storage))
    } else {
        Vec::new()
    };
    let hydrate_vectors = !vec_names.is_empty();

    for hvec in search_results {
        if let Some(pending) = pending_point(&name, hvec.id) {
            let mut point = serde_json::json!({
                "id": format_point_id(hvec.id),
                "version": 0,
                "score": hvec.distance.unwrap_or(0.0),
            });

            if req.with_payload {
                point["payload"] = pending_payload_json(&pending);
            }

            if hydrate_vectors {
                point["vector"] = pending_vector_json(&pending, &vec_names);
            }

            results.push(point);
            continue;
        }

        let node = match storage.get_node(&txn, &hvec.id) {
            Ok(n) => n,
            Err(_) => continue,
        };

        let mut point = serde_json::json!({
            "id": format_point_id(hvec.id),
            "version": 0,
            "score": hvec.distance.unwrap_or(0.0),
        });

        if req.with_payload {
            point["payload"] = value_map_to_json(&node.properties);
        }

        if hydrate_vectors {
            point["vector"] = fetch_named_vectors(&storage, &txn, hvec.id, &vec_names);
        }

        results.push(point);
    }

    json_ok(response, &results)
}

// ─── Scroll ───

const SCAN_CURSOR_PREFIX: &str = "hscan1.";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
struct ScanPartition {
    index: usize,
    total: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScanCursor {
    offset: String,
    generation: usize,
    filter_hash: u64,
    #[serde(default)]
    partition: Option<ScanPartition>,
}

#[derive(Debug, Clone, Copy)]
enum PointScanEndpoint {
    QdrantScroll,
    HelionScan,
}

impl PointScanEndpoint {
    fn as_str(self) -> &'static str {
        match self {
            PointScanEndpoint::QdrantScroll => "qdrant_scroll",
            PointScanEndpoint::HelionScan => "helion_scan",
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum PointScanPlan {
    Primary,
    IndexedCandidates,
}

impl PointScanPlan {
    fn as_str(self) -> &'static str {
        match self {
            PointScanPlan::Primary => "primary",
            PointScanPlan::IndexedCandidates => "indexed_candidates",
        }
    }
}

struct PointScanOptions<'a> {
    collection: &'a str,
    limit: usize,
    offset_id: u128,
    filter: &'a Filter,
    with_payload: &'a WithPayload,
    with_vectors: &'a WithVectors,
    partition: Option<ScanPartition>,
    max_scan: Option<usize>,
    deadline: Option<Instant>,
    filter_hash: u64,
    cursor_generation: Option<usize>,
    fuse_pending: bool,
}

struct PointScanResult {
    points: Vec<serde_json::Value>,
    next_offset: Option<String>,
    next_cursor: Option<String>,
    plan: PointScanPlan,
    scanned: usize,
    returned: usize,
    index_candidates: Option<usize>,
    budget_exhausted: bool,
    generation: usize,
    generation_mismatch: bool,
}

#[derive(Deserialize)]
struct ScrollRequest {
    #[serde(default = "default_scroll_limit")]
    limit: usize,
    #[serde(default)]
    offset: Option<serde_json::Value>,
    #[serde(default)]
    filter: Option<Filter>,
    /// Accepts `bool | [str] | {include?: [str], exclude?: [str]}` — full
    /// Qdrant `WithPayload` shape. See `WithPayload` docs.
    #[serde(default)]
    with_payload: WithPayload,
    /// Accepts `false | true | ["name"]`. See `WithVectors` docs.
    #[serde(default, alias = "with_vector")]
    with_vectors: WithVectors,
}

#[derive(Deserialize)]
struct HelionScanRequest {
    #[serde(default = "default_scroll_limit")]
    limit: usize,
    #[serde(default)]
    offset: Option<serde_json::Value>,
    #[serde(default)]
    cursor: Option<String>,
    #[serde(default)]
    filter: Option<Filter>,
    #[serde(default)]
    with_payload: WithPayload,
    #[serde(default, alias = "with_vector")]
    with_vectors: WithVectors,
    #[serde(default)]
    partition: Option<ScanPartition>,
    #[serde(default)]
    max_scan: Option<usize>,
    #[serde(default)]
    require_consistent: bool,
}

fn default_scroll_limit() -> usize {
    10
}

fn max_scan_partitions() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HELIX_SCAN_PARTITION_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(4096)
    })
}

fn max_scan_budget() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HELIX_SCAN_BUDGET_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(1_000_000)
    })
}

const DEFAULT_QDRANT_SCROLL_SCAN_BUDGET_MIN: usize = 1_024;
const DEFAULT_QDRANT_SCROLL_SCAN_BUDGET_MULTIPLIER: usize = 64;
const DEFAULT_QDRANT_SCROLL_SCAN_BUDGET_MAX: usize = 4_096;
const DEFAULT_QDRANT_SCROLL_SCAN_BUDGET_MS: usize = 250;

fn env_usize_or(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn qdrant_scroll_scan_budget_for(
    limit: usize,
    filtered: bool,
    min_budget: usize,
    multiplier: usize,
    max_budget: usize,
    global_max: usize,
) -> Option<usize> {
    if !filtered || max_budget == 0 || global_max == 0 {
        return None;
    }
    let requested = limit
        .max(1)
        .saturating_mul(multiplier.max(1))
        .max(min_budget);
    Some(requested.min(max_budget).min(global_max).max(1))
}

fn qdrant_scroll_scan_budget(limit: usize, filter: &Filter) -> Option<usize> {
    qdrant_scroll_scan_budget_for(
        limit,
        !filter.is_empty(),
        env_usize_or(
            "HELIX_QDRANT_SCROLL_SCAN_BUDGET_MIN",
            DEFAULT_QDRANT_SCROLL_SCAN_BUDGET_MIN,
        ),
        env_usize_or(
            "HELIX_QDRANT_SCROLL_SCAN_BUDGET_MULTIPLIER",
            DEFAULT_QDRANT_SCROLL_SCAN_BUDGET_MULTIPLIER,
        ),
        env_usize_or(
            "HELIX_QDRANT_SCROLL_SCAN_BUDGET_MAX",
            DEFAULT_QDRANT_SCROLL_SCAN_BUDGET_MAX,
        ),
        max_scan_budget(),
    )
}

fn qdrant_scroll_scan_deadline(filter: &Filter) -> Option<Instant> {
    if filter.is_empty() {
        return None;
    }
    let budget_ms = env_usize_or(
        "HELIX_QDRANT_SCROLL_SCAN_BUDGET_MS",
        DEFAULT_QDRANT_SCROLL_SCAN_BUDGET_MS,
    );
    if budget_ms == 0 {
        return None;
    }
    Instant::now().checked_add(Duration::from_millis(budget_ms as u64))
}

fn normalize_scan_budget(max_scan: Option<usize>) -> Option<usize> {
    max_scan
        .filter(|&n| n > 0)
        .map(|n| n.min(max_scan_budget()))
}

fn validate_scan_partition(
    partition: Option<ScanPartition>,
) -> Result<Option<ScanPartition>, String> {
    let Some(partition) = partition else {
        return Ok(None);
    };
    if partition.total == 0 {
        return Err("partition.total must be greater than zero".to_string());
    }
    if partition.index >= partition.total {
        return Err("partition.index must be less than partition.total".to_string());
    }
    if partition.total > max_scan_partitions() {
        return Err(format!(
            "partition.total exceeds HELIX_SCAN_PARTITION_MAX ({})",
            max_scan_partitions()
        ));
    }
    Ok(Some(partition))
}

fn scan_partition_width(total: usize) -> u128 {
    (u128::MAX / total as u128).saturating_add(1)
}

fn scan_partition_start(partition: Option<ScanPartition>) -> u128 {
    let Some(partition) = partition else {
        return 0;
    };
    if partition.total <= 1 {
        return 0;
    }
    scan_partition_width(partition.total).saturating_mul(partition.index as u128)
}

fn scan_partition_end_exclusive(partition: Option<ScanPartition>) -> Option<u128> {
    let partition = partition?;
    if partition.total <= 1 || partition.index + 1 >= partition.total {
        return None;
    }
    Some(scan_partition_width(partition.total).saturating_mul((partition.index + 1) as u128))
}

fn parse_scroll_offset(offset: Option<&serde_json::Value>) -> u128 {
    match offset {
        Some(serde_json::Value::String(s)) => {
            let clean = s.trim().trim_start_matches("0x");
            u128::from_str_radix(clean, 16)
                .ok()
                .or_else(|| clean.parse::<u128>().ok())
                .unwrap_or(0)
        }
        Some(serde_json::Value::Number(n)) => n.as_u64().unwrap_or(0) as u128,
        _ => 0,
    }
}

fn next_scan_offset(id: u128) -> Option<String> {
    id.checked_add(1).map(format_point_id)
}

fn scan_filter_hash(filter: &Filter) -> u64 {
    if filter.is_empty() {
        return 0;
    }
    let encoded = serde_json::to_vec(filter).unwrap_or_default();
    let mut hasher = twox_hash::XxHash64::with_seed(0xC0DEC0DE_51A9_0001);
    hasher.write(&encoded);
    hasher.finish()
}

fn filter_field_keys(filter: &Filter) -> String {
    let mut keys = BTreeSet::new();
    collect_filter_field_keys(filter, &mut keys);
    if keys.is_empty() {
        "none".to_string()
    } else {
        keys.into_iter().collect::<Vec<_>>().join(",")
    }
}

fn collect_filter_field_keys(filter: &Filter, keys: &mut BTreeSet<String>) {
    for condition in filter
        .must
        .iter()
        .chain(filter.should.iter())
        .chain(filter.must_not.iter())
        .chain(filter.min_should.iter().flat_map(|ms| ms.conditions.iter()))
    {
        collect_condition_field_keys(condition, keys);
    }
}

fn collect_condition_field_keys(condition: &Condition, keys: &mut BTreeSet<String>) {
    match condition {
        Condition::Field(field) => {
            keys.insert(field.key.clone());
        }
        Condition::IsEmpty(cond) => {
            keys.insert(cond.is_empty.key.clone());
        }
        Condition::IsNull(cond) => {
            keys.insert(cond.is_null.key.clone());
        }
        Condition::Nested(filter) => collect_filter_field_keys(filter, keys),
        Condition::HasId(_) => {}
    }
}

fn encode_scan_cursor(
    offset: &str,
    generation: usize,
    filter_hash: u64,
    partition: Option<ScanPartition>,
) -> Option<String> {
    let cursor = ScanCursor {
        offset: offset.to_string(),
        generation,
        filter_hash,
        partition,
    };
    serde_json::to_vec(&cursor)
        .ok()
        .map(|bytes| format!("{}{}", SCAN_CURSOR_PREFIX, URL_SAFE_NO_PAD.encode(bytes)))
}

fn decode_scan_cursor(cursor: &str) -> Result<ScanCursor, String> {
    let encoded = cursor
        .strip_prefix(SCAN_CURSOR_PREFIX)
        .ok_or_else(|| "invalid cursor prefix".to_string())?;
    let bytes = URL_SAFE_NO_PAD
        .decode(encoded)
        .map_err(|e| format!("invalid cursor encoding: {}", e))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("invalid cursor payload: {}", e))
}

fn build_scan_point(
    storage: &HelixGraphStorage,
    txn: &heed3::RoTxn,
    id: u128,
    node: &Node,
    with_payload: &WithPayload,
    hydrate_vectors: bool,
    vec_names: &[String],
) -> serde_json::Value {
    let mut point = serde_json::json!({
        "id": format_point_id(id),
    });

    if with_payload.is_active() {
        point["payload"] = with_payload.project(value_map_to_json(&node.properties));
    }

    if hydrate_vectors {
        point["vector"] = fetch_named_vectors(storage, txn, id, vec_names);
    }

    point
}

fn build_scan_point_be(
    storage: &HelixGraphStorage,
    r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    id: u128,
    node: &Node,
    with_payload: &WithPayload,
    hydrate_vectors: bool,
    vec_names: &[String],
) -> serde_json::Value {
    let mut point = serde_json::json!({
        "id": format_point_id(id),
    });

    if with_payload.is_active() {
        point["payload"] = with_payload.project(value_map_to_json(&node.properties));
    }

    if hydrate_vectors {
        point["vector"] = fetch_named_vectors_be(storage, r, id, vec_names);
    }

    point
}

fn point_json_id(point: &serde_json::Value) -> Option<u128> {
    point.get("id").map(point_id_to_u128)
}

fn point_scan_budget_exhausted(scanned: usize, opts: &PointScanOptions<'_>) -> bool {
    opts.max_scan.map(|max| scanned >= max).unwrap_or(false)
        || opts
            .deadline
            .map(|deadline| Instant::now() >= deadline)
            .unwrap_or(false)
}

fn fuse_pending_scan_points(
    points: &mut Vec<serde_json::Value>,
    next_offset: &mut Option<String>,
    opts: &PointScanOptions<'_>,
    hydrate_vectors: bool,
    vec_names: &[String],
) {
    let start_id = opts.offset_id.max(scan_partition_start(opts.partition));
    let end_exclusive = scan_partition_end_exclusive(opts.partition);
    let mut pending_json = Vec::new();
    let mut pending_ids = HashSet::new();

    for (id, pending) in pending_points_for_collection(opts.collection) {
        if id < start_id {
            continue;
        }
        if let Some(end) = end_exclusive {
            if id >= end {
                break;
            }
        }
        if !pending_matches_filter(id, &pending, opts.filter) {
            continue;
        }
        pending_ids.insert(id);
        pending_json.push(build_pending_scan_point(
            id,
            &pending,
            opts.with_payload,
            hydrate_vectors,
            vec_names,
        ));
    }

    if pending_json.is_empty() {
        return;
    }

    points.retain(|point| {
        point_json_id(point)
            .map(|id| !pending_ids.contains(&id))
            .unwrap_or(true)
    });
    points.extend(pending_json);
    points.sort_by_key(|point| point_json_id(point).unwrap_or(0));

    if points.len() > opts.limit {
        points.truncate(opts.limit);
        if let Some(last_id) = points.last().and_then(point_json_id) {
            *next_offset = next_scan_offset(last_id);
        }
    }
}

fn execute_point_scan(
    storage: &HelixGraphStorage,
    txn: &heed3::RoTxn,
    opts: &PointScanOptions<'_>,
) -> Result<PointScanResult, GraphError> {
    let generation = storage.lmdb_env()?.info().last_txn_id;
    let generation_mismatch = opts
        .cursor_generation
        .map(|cursor_generation| cursor_generation != generation)
        .unwrap_or(false);
    let start_id = opts.offset_id.max(scan_partition_start(opts.partition));
    let end_exclusive = scan_partition_end_exclusive(opts.partition);
    let indexed_candidates = scan_indexed_filter_candidates(storage, txn, opts.filter)?;
    let plan = if indexed_candidates.is_some() {
        PointScanPlan::IndexedCandidates
    } else {
        PointScanPlan::Primary
    };
    let index_candidates = indexed_candidates.as_ref().map(HashSet::len);

    let vec_names: Vec<String> = if opts.with_vectors.is_active() {
        opts.with_vectors
            .resolved_names(&known_vector_names(&storage))
    } else {
        Vec::new()
    };
    let hydrate_vectors = !vec_names.is_empty();

    let mut points: Vec<serde_json::Value> = Vec::new();
    let mut next_offset: Option<String> = None;
    let mut scanned = 0usize;
    let mut budget_exhausted = false;

    if let Some(candidates) = indexed_candidates {
        let mut ids: Vec<u128> = candidates.into_iter().collect();
        ids.sort_unstable();

        for id in ids {
            if id < start_id {
                continue;
            }
            if let Some(end) = end_exclusive {
                if id >= end {
                    break;
                }
            }

            scanned = scanned.saturating_add(1);
            let node = match storage.get_node(txn, &id) {
                Ok(node) => node,
                // An indexed candidate whose node is gone is skipped; any other
                // error (e.g. object storage unreachable on an uncached LSM read,
                // or decode corruption) must propagate, not be swallowed into a
                // silent empty result.
                Err(GraphError::NodeNotFound) => continue,
                Err(e) => return Err(e),
            };
            if !opts.filter.is_empty() && !opts.filter.matches_point(Some(id), &node.properties) {
                if point_scan_budget_exhausted(scanned, opts) {
                    budget_exhausted = true;
                    next_offset = next_scan_offset(id);
                    break;
                }
                continue;
            }

            points.push(build_scan_point(
                storage,
                txn,
                id,
                &node,
                opts.with_payload,
                hydrate_vectors,
                &vec_names,
            ));

            if points.len() >= opts.limit {
                next_offset = next_scan_offset(id);
                break;
            }
            if point_scan_budget_exhausted(scanned, opts) {
                budget_exhausted = true;
                next_offset = next_scan_offset(id);
                break;
            }
        }
    } else {
        // Primary plan: ordered scan over Namespace::Nodes through the backend
        // seam (NOT `storage.nodes_db.range`, which reads the heed DBI directly
        // and returns nothing on the LSM backend where nodes live in SlateDB).
        // Node ids are 16-byte big-endian keys (`Database<U128<BE>, Bytes>`), so
        // the byte range is identical on LMDB and decode is byte-for-byte the
        // same as `get_node`'s seam read.
        use crate::helix_engine::storage_core::backend::{KeyRange, Namespace};

        let range = KeyRange {
            start: Bound::Included(start_id.to_be_bytes().to_vec()),
            end: match end_exclusive.as_ref() {
                Some(end) => Bound::Excluded(end.to_be_bytes().to_vec()),
                None => Bound::Unbounded,
            },
        };

        let mut scan_err: Option<GraphError> = None;
        storage
            .backend
            .scan_heed(txn, Namespace::Nodes, range, |k, data| {
                let mut bytes = [0u8; 16];
                bytes.copy_from_slice(k);
                let id = u128::from_be_bytes(bytes);
                scanned = scanned.saturating_add(1);
                let node = match SerializedNode::decode_node(data, id) {
                    Ok(node) => node,
                    Err(e) => {
                        scan_err = Some(e);
                        return false;
                    }
                };
                if !opts.filter.is_empty() && !opts.filter.matches_point(Some(id), &node.properties)
                {
                    if point_scan_budget_exhausted(scanned, opts) {
                        budget_exhausted = true;
                        next_offset = next_scan_offset(id);
                        return false;
                    }
                    return true;
                }

                points.push(build_scan_point(
                    storage,
                    txn,
                    id,
                    &node,
                    opts.with_payload,
                    hydrate_vectors,
                    &vec_names,
                ));

                if points.len() >= opts.limit {
                    next_offset = next_scan_offset(id);
                    return false;
                }
                if point_scan_budget_exhausted(scanned, opts) {
                    budget_exhausted = true;
                    next_offset = next_scan_offset(id);
                    return false;
                }
                true
            })
            .map_err(|e| GraphError::New(e.to_string()))?;
        if let Some(e) = scan_err {
            return Err(e);
        }
    }

    if opts.fuse_pending {
        fuse_pending_scan_points(
            &mut points,
            &mut next_offset,
            opts,
            hydrate_vectors,
            &vec_names,
        );
    }

    let next_cursor = next_offset.as_ref().and_then(|offset| {
        encode_scan_cursor(offset, generation, opts.filter_hash, opts.partition)
    });
    let returned = points.len();
    Ok(PointScanResult {
        points,
        next_offset,
        next_cursor,
        plan,
        scanned,
        returned,
        index_candidates,
        budget_exhausted,
        generation,
        generation_mismatch,
    })
}

fn execute_point_scan_be(
    storage: &HelixGraphStorage,
    r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    opts: &PointScanOptions<'_>,
) -> Result<PointScanResult, GraphError> {
    let generation = 0usize;
    let generation_mismatch = opts
        .cursor_generation
        .map(|cursor_generation| cursor_generation != generation)
        .unwrap_or(false);
    let start_id = opts.offset_id.max(scan_partition_start(opts.partition));
    let end_exclusive = scan_partition_end_exclusive(opts.partition);
    let indexed_candidates = scan_indexed_filter_candidates_be(storage, r, opts.filter)?;
    let plan = if indexed_candidates.is_some() {
        PointScanPlan::IndexedCandidates
    } else {
        PointScanPlan::Primary
    };
    let index_candidates = indexed_candidates.as_ref().map(HashSet::len);
    let vec_names: Vec<String> = if opts.with_vectors.is_active() {
        opts.with_vectors
            .resolved_names(&known_vector_names(&storage))
    } else {
        Vec::new()
    };
    let hydrate_vectors = !vec_names.is_empty();

    let mut points: Vec<serde_json::Value> = Vec::new();
    let mut next_offset: Option<String> = None;
    let mut scanned = 0usize;
    let mut budget_exhausted = false;
    if let Some(candidates) = indexed_candidates {
        let mut ids: Vec<u128> = candidates.into_iter().collect();
        ids.sort_unstable();

        for id in ids {
            if id < start_id {
                continue;
            }
            if let Some(end) = end_exclusive {
                if id >= end {
                    break;
                }
            }

            scanned = scanned.saturating_add(1);
            let node = match storage.get_node_be(r, &id) {
                Ok(node) => node,
                Err(GraphError::NodeNotFound) => {
                    // A ghost payload-index entry (or merely a sparse filter)
                    // lands here. Without a budget check, a long run of
                    // missing candidates never terminates the page — this is
                    // the generic runaway-scan protection every other branch
                    // below already gets.
                    if point_scan_budget_exhausted(scanned, opts) {
                        budget_exhausted = true;
                        next_offset = next_scan_offset(id);
                        break;
                    }
                    continue;
                }
                Err(e) => return Err(e),
            };
            if !opts.filter.is_empty() && !opts.filter.matches_point(Some(id), &node.properties) {
                if point_scan_budget_exhausted(scanned, opts) {
                    budget_exhausted = true;
                    next_offset = next_scan_offset(id);
                    break;
                }
                continue;
            }

            points.push(build_scan_point_be(
                storage,
                r,
                id,
                &node,
                opts.with_payload,
                hydrate_vectors,
                &vec_names,
            ));

            if points.len() >= opts.limit {
                next_offset = next_scan_offset(id);
                break;
            }
            if point_scan_budget_exhausted(scanned, opts) {
                budget_exhausted = true;
                next_offset = next_scan_offset(id);
                break;
            }
        }
    } else {
        let range = KeyRange {
            start: Bound::Included(start_id.to_be_bytes().to_vec()),
            end: match end_exclusive.as_ref() {
                Some(end) => Bound::Excluded(end.to_be_bytes().to_vec()),
                None => Bound::Unbounded,
            },
        };

        let mut scan_err: Option<GraphError> = None;
        storage
            .backend
            .scan(r, Namespace::Nodes, range, |k, data| {
                let Ok(bytes) = <[u8; 16]>::try_from(k) else {
                    scan_err = Some(GraphError::New(
                        "invalid node key length during LSM scan".into(),
                    ));
                    return false;
                };
                let id = u128::from_be_bytes(bytes);
                scanned = scanned.saturating_add(1);
                let node = match SerializedNode::decode_node(data, id) {
                    Ok(node) => node,
                    Err(e) => {
                        scan_err = Some(e);
                        return false;
                    }
                };
                if !opts.filter.is_empty() && !opts.filter.matches_point(Some(id), &node.properties)
                {
                    if point_scan_budget_exhausted(scanned, opts) {
                        budget_exhausted = true;
                        next_offset = next_scan_offset(id);
                        return false;
                    }
                    return true;
                }

                points.push(build_scan_point_be(
                    storage,
                    r,
                    id,
                    &node,
                    opts.with_payload,
                    hydrate_vectors,
                    &vec_names,
                ));

                if points.len() >= opts.limit {
                    next_offset = next_scan_offset(id);
                    return false;
                }
                if point_scan_budget_exhausted(scanned, opts) {
                    budget_exhausted = true;
                    next_offset = next_scan_offset(id);
                    return false;
                }
                true
            })
            .map_err(|e| GraphError::New(e.to_string()))?;
        if let Some(e) = scan_err {
            return Err(e);
        }
    }

    if opts.fuse_pending {
        fuse_pending_scan_points(
            &mut points,
            &mut next_offset,
            opts,
            hydrate_vectors,
            &vec_names,
        );
    }

    let next_cursor = next_offset.as_ref().and_then(|offset| {
        encode_scan_cursor(offset, generation, opts.filter_hash, opts.partition)
    });
    let returned = points.len();
    Ok(PointScanResult {
        points,
        next_offset,
        next_cursor,
        plan,
        scanned,
        returned,
        index_candidates,
        budget_exhausted,
        generation,
        generation_mismatch,
    })
}

fn primary_scan_warn_threshold() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HELIX_SCAN_PRIMARY_WARN_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(50_000)
    })
}

fn observe_point_scan_metrics(
    collection: &str,
    endpoint: PointScanEndpoint,
    plan: PointScanPlan,
    partitioned: bool,
    filtered: bool,
    filter: &Filter,
    with_payload: &WithPayload,
    with_vectors: &WithVectors,
    elapsed_ms: f64,
    scanned: usize,
    returned: usize,
    response_bytes: usize,
    budget_exhausted: bool,
    generation_mismatch: bool,
) {
    let endpoint_label = endpoint.as_str();
    let plan_label = plan.as_str();
    let partitioned_label = if partitioned { "true" } else { "false" };
    let filtered_label = if filtered { "true" } else { "false" };
    let payload_label = with_payload.label();
    let vectors_label = with_vectors.label();

    // Always-on visibility for runaway primary-plan scans, even when
    // hot-path metrics are disabled. These are the requests where a
    // filter missed every payload index and Helix walked the collection.
    let primary_warn_threshold = primary_scan_warn_threshold();
    if matches!(plan, PointScanPlan::Primary)
        && filtered
        && primary_warn_threshold > 0
        && scanned >= primary_warn_threshold
    {
        let filter_keys = filter_field_keys(filter);
        tracing::warn!(
            collection = collection,
            endpoint = endpoint_label,
            scanned = scanned,
            returned = returned,
            elapsed_ms = elapsed_ms,
            response_bytes = response_bytes,
            filter_keys = %filter_keys,
            payload = payload_label,
            vectors = vectors_label,
            budget_exhausted = budget_exhausted,
            "filtered scan fell back to primary plan; filter has no payload-index match — caller should add a payload index or page by indexed candidates"
        );
    }

    if !crate::telemetry::hot_path_metrics_enabled() {
        return;
    }

    metrics::counter!(
        "helix_point_scan_requests_total",
        "collection" => collection.to_string(),
        "endpoint" => endpoint_label,
        "plan" => plan_label,
        "partitioned" => partitioned_label,
        "filtered" => filtered_label,
        "payload" => payload_label,
        "vectors" => vectors_label,
    )
    .increment(1);
    metrics::histogram!(
        "helix_point_scan_duration_ms",
        "collection" => collection.to_string(),
        "endpoint" => endpoint_label,
        "plan" => plan_label,
        "partitioned" => partitioned_label,
        "filtered" => filtered_label,
        "payload" => payload_label,
        "vectors" => vectors_label,
    )
    .record(elapsed_ms);
    metrics::histogram!(
        "helix_point_scan_scanned_points",
        "collection" => collection.to_string(),
        "endpoint" => endpoint_label,
        "plan" => plan_label,
        "partitioned" => partitioned_label,
        "filtered" => filtered_label,
    )
    .record(scanned as f64);
    metrics::histogram!(
        "helix_point_scan_returned_points",
        "collection" => collection.to_string(),
        "endpoint" => endpoint_label,
        "plan" => plan_label,
        "partitioned" => partitioned_label,
        "filtered" => filtered_label,
    )
    .record(returned as f64);
    metrics::histogram!(
        "helix_point_scan_response_bytes",
        "collection" => collection.to_string(),
        "endpoint" => endpoint_label,
        "plan" => plan_label,
        "partitioned" => partitioned_label,
        "filtered" => filtered_label,
    )
    .record(response_bytes as f64);

    if budget_exhausted {
        metrics::counter!(
            "helix_point_scan_budget_exhausted_total",
            "collection" => collection.to_string(),
            "endpoint" => endpoint_label,
            "plan" => plan_label,
        )
        .increment(1);
    }
    if generation_mismatch {
        metrics::counter!(
            "helix_point_scan_generation_mismatch_total",
            "collection" => collection.to_string(),
            "endpoint" => endpoint_label,
            "plan" => plan_label,
        )
        .increment(1);
    }
}

/// POST /collections/{name}/points/scroll
pub fn handle_scroll_points(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let name = match get_collection_name(input) {
        Some(n) => n,
        None => return json_err(response, 400, "Missing collection name"),
    };

    let mut req: ScrollRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_err(response, 400, &format!("Invalid JSON: {}", e)),
    };
    req.limit = clamp_limit(req.limit);

    let storage = match input.collections.get_collection(&name) {
        Ok(s) => s,
        Err(e) => return json_err(response, 404, &e.to_string()),
    };

    let filter = req.filter.unwrap_or_default();
    let opts = PointScanOptions {
        collection: &name,
        limit: req.limit,
        offset_id: parse_scroll_offset(req.offset.as_ref()),
        filter: &filter,
        with_payload: &req.with_payload,
        with_vectors: &req.with_vectors,
        partition: None,
        max_scan: qdrant_scroll_scan_budget(req.limit, &filter),
        deadline: qdrant_scroll_scan_deadline(&filter),
        filter_hash: scan_filter_hash(&filter),
        cursor_generation: None,
        fuse_pending: true,
    };

    if storage.backend.kind() == BackendKind::Lsm {
        let r = storage
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))?;
        let started = Instant::now();
        let scan = execute_point_scan_be(&storage, &r, &opts)?;
        let plan = scan.plan;
        let scanned = scan.scanned;
        let returned = scan.returned;
        let budget_exhausted = scan.budget_exhausted;
        let generation_mismatch = scan.generation_mismatch;
        let result = json_ok(
            response,
            &serde_json::json!({
                "points": scan.points,
                "next_page_offset": scan.next_offset,
            }),
        );
        if result.is_ok() {
            observe_point_scan_metrics(
                &name,
                PointScanEndpoint::QdrantScroll,
                plan,
                false,
                !filter.is_empty(),
                &filter,
                &req.with_payload,
                &req.with_vectors,
                started.elapsed().as_secs_f64() * 1000.0,
                scanned,
                returned,
                response.body.len(),
                budget_exhausted,
                generation_mismatch,
            );
        }
        return result;
    }

    let txn = storage.begin_resize_safe_read_txn()?;
    let started = Instant::now();
    let scan = execute_point_scan(&storage, &txn, &opts)?;
    let plan = scan.plan;
    let scanned = scan.scanned;
    let returned = scan.returned;
    let budget_exhausted = scan.budget_exhausted;
    let generation_mismatch = scan.generation_mismatch;
    let result = json_ok(
        response,
        &serde_json::json!({
            "points": scan.points,
            "next_page_offset": scan.next_offset,
        }),
    );
    if result.is_ok() {
        observe_point_scan_metrics(
            &name,
            PointScanEndpoint::QdrantScroll,
            plan,
            false,
            !filter.is_empty(),
            &filter,
            &req.with_payload,
            &req.with_vectors,
            started.elapsed().as_secs_f64() * 1000.0,
            scanned,
            returned,
            response.body.len(),
            budget_exhausted,
            generation_mismatch,
        );
    }
    result
}

/// POST /v1/collections/{name}/points/scan
pub fn handle_scan_points(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    let name = match get_collection_name(input) {
        Some(n) => n,
        None => return json_err(response, 400, "Missing collection name"),
    };

    let mut req: HelionScanRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_err(response, 400, &format!("Invalid JSON: {}", e)),
    };
    req.limit = clamp_limit(req.limit);
    req.max_scan = normalize_scan_budget(req.max_scan);

    let filter = req.filter.unwrap_or_default();
    let filter_hash = scan_filter_hash(&filter);
    let mut offset_id = parse_scroll_offset(req.offset.as_ref());
    let mut cursor_generation = None;
    let mut partition = match validate_scan_partition(req.partition) {
        Ok(partition) => partition,
        Err(e) => return json_err(response, 400, &e),
    };
    let request_partition = partition;

    if let Some(raw_cursor) = req.cursor.as_deref() {
        let cursor = match decode_scan_cursor(raw_cursor) {
            Ok(cursor) => cursor,
            Err(e) => return json_err(response, 400, &e),
        };
        if cursor.filter_hash != filter_hash {
            return json_err(response, 400, "cursor does not match the supplied filter");
        }
        if request_partition.is_some() && request_partition != cursor.partition {
            return json_err(
                response,
                400,
                "cursor does not match the supplied partition",
            );
        }
        if let Some(cursor_partition) = cursor.partition {
            partition = Some(cursor_partition);
        }
        offset_id = parse_scroll_offset(Some(&serde_json::Value::String(cursor.offset)));
        cursor_generation = Some(cursor.generation);
    }

    let storage = match input.collections.get_collection(&name) {
        Ok(s) => s,
        Err(e) => return json_err(response, 404, &e.to_string()),
    };

    let opts = PointScanOptions {
        collection: &name,
        limit: req.limit,
        offset_id,
        filter: &filter,
        with_payload: &req.with_payload,
        with_vectors: &req.with_vectors,
        partition,
        max_scan: req.max_scan,
        deadline: None,
        filter_hash,
        cursor_generation,
        fuse_pending: true,
    };

    if storage.backend.kind() == BackendKind::Lsm {
        let r = storage
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))?;
        let started = Instant::now();
        let scan = execute_point_scan_be(&storage, &r, &opts)?;
        let plan = scan.plan;
        let scanned = scan.scanned;
        let returned = scan.returned;
        let index_candidates = scan.index_candidates;
        let budget_exhausted = scan.budget_exhausted;
        let generation = scan.generation;
        let generation_mismatch = scan.generation_mismatch;
        if req.require_consistent && generation_mismatch {
            observe_point_scan_metrics(
                &name,
                PointScanEndpoint::HelionScan,
                plan,
                partition.is_some(),
                !filter.is_empty(),
                &filter,
                &req.with_payload,
                &req.with_vectors,
                started.elapsed().as_secs_f64() * 1000.0,
                scanned,
                returned,
                0,
                budget_exhausted,
                generation_mismatch,
            );
            return json_err(
                response,
                409,
                "scan cursor generation changed; restart scan for a consistent export",
            );
        }

        let next_cursor = scan.next_cursor.clone();
        let next_offset = scan.next_offset.clone();
        let result = json_ok(
            response,
            &serde_json::json!({
                "points": scan.points,
                "next_page_offset": next_offset,
                "next_cursor": next_cursor,
                "scan": {
                    "plan": plan.as_str(),
                    "scanned": scanned,
                    "returned": returned,
                    "index_candidates": index_candidates,
                    "budget_exhausted": budget_exhausted,
                    "generation": generation,
                    "generation_mismatch": generation_mismatch,
                }
            }),
        );
        if result.is_ok() {
            observe_point_scan_metrics(
                &name,
                PointScanEndpoint::HelionScan,
                plan,
                partition.is_some(),
                !filter.is_empty(),
                &filter,
                &req.with_payload,
                &req.with_vectors,
                started.elapsed().as_secs_f64() * 1000.0,
                scanned,
                returned,
                response.body.len(),
                budget_exhausted,
                generation_mismatch,
            );
        }
        return result;
    }

    let txn = storage.begin_resize_safe_read_txn()?;
    let started = Instant::now();
    let scan = execute_point_scan(&storage, &txn, &opts)?;
    let plan = scan.plan;
    let scanned = scan.scanned;
    let returned = scan.returned;
    let index_candidates = scan.index_candidates;
    let budget_exhausted = scan.budget_exhausted;
    let generation = scan.generation;
    let generation_mismatch = scan.generation_mismatch;
    if req.require_consistent && generation_mismatch {
        observe_point_scan_metrics(
            &name,
            PointScanEndpoint::HelionScan,
            plan,
            partition.is_some(),
            !filter.is_empty(),
            &filter,
            &req.with_payload,
            &req.with_vectors,
            started.elapsed().as_secs_f64() * 1000.0,
            scanned,
            returned,
            0,
            budget_exhausted,
            generation_mismatch,
        );
        return json_err(
            response,
            409,
            "scan cursor generation changed; restart scan for a consistent export",
        );
    }

    let next_cursor = scan.next_cursor.clone();
    let next_offset = scan.next_offset.clone();
    let result = json_ok(
        response,
        &serde_json::json!({
            "points": scan.points,
            "next_page_offset": next_offset,
            "next_cursor": next_cursor,
            "scan": {
                "plan": plan.as_str(),
                "scanned": scanned,
                "returned": returned,
                "index_candidates": index_candidates,
                "budget_exhausted": budget_exhausted,
                "limit": req.limit,
                "max_scan": req.max_scan,
                "partition": partition,
                "generation": generation,
                "consistent": !generation_mismatch,
            }
        }),
    );
    if result.is_ok() {
        observe_point_scan_metrics(
            &name,
            PointScanEndpoint::HelionScan,
            plan,
            partition.is_some(),
            !filter.is_empty(),
            &filter,
            &req.with_payload,
            &req.with_vectors,
            started.elapsed().as_secs_f64() * 1000.0,
            scanned,
            returned,
            response.body.len(),
            budget_exhausted,
            generation_mismatch,
        );
    }
    result
}

// ─── Facet ───

#[derive(Deserialize)]
struct FacetRequest {
    key: String,
    #[serde(default)]
    filter: Option<Filter>,
    #[serde(default = "default_facet_limit")]
    limit: usize,
    #[serde(default)]
    exact: Option<bool>,
}

fn default_facet_limit() -> usize {
    10
}

#[derive(Default)]
struct FacetTelemetry {
    entries_scanned: usize,
    candidates: usize,
    matched_items: usize,
}

fn record_facet_phase(collection: &str, plan: &'static str, phase: &'static str, started: Instant) {
    if !crate::telemetry::hot_path_metrics_enabled() {
        return;
    }
    metrics::histogram!(
        "helix_facet_phase_ms",
        "collection" => collection.to_string(),
        "plan" => plan,
        "phase" => phase,
    )
    .record(started.elapsed().as_secs_f64() * 1000.0);
}

fn observe_facet_result(
    collection: &str,
    plan: &'static str,
    filtered: bool,
    exact: bool,
    telemetry: FacetTelemetry,
    distinct_values: usize,
    returned_hits: usize,
    response_bytes: usize,
) {
    if !crate::telemetry::hot_path_metrics_enabled() {
        return;
    }
    let filtered_label = if filtered { "true" } else { "false" };
    let exact_label = if exact { "true" } else { "false" };
    metrics::counter!(
        "helix_facet_requests_total",
        "collection" => collection.to_string(),
        "plan" => plan,
        "filtered" => filtered_label,
        "exact" => exact_label,
    )
    .increment(1);
    metrics::histogram!(
        "helix_facet_entries_scanned",
        "collection" => collection.to_string(),
        "plan" => plan,
        "filtered" => filtered_label,
    )
    .record(telemetry.entries_scanned as f64);
    metrics::histogram!(
        "helix_facet_candidates",
        "collection" => collection.to_string(),
        "plan" => plan,
        "filtered" => filtered_label,
    )
    .record(telemetry.candidates as f64);
    metrics::histogram!(
        "helix_facet_matched_items",
        "collection" => collection.to_string(),
        "plan" => plan,
        "filtered" => filtered_label,
    )
    .record(telemetry.matched_items as f64);
    metrics::histogram!(
        "helix_facet_distinct_values",
        "collection" => collection.to_string(),
        "plan" => plan,
        "filtered" => filtered_label,
    )
    .record(distinct_values as f64);
    metrics::histogram!(
        "helix_facet_returned_hits",
        "collection" => collection.to_string(),
        "plan" => plan,
        "filtered" => filtered_label,
    )
    .record(returned_hits as f64);
    metrics::histogram!(
        "helix_facet_response_bytes",
        "collection" => collection.to_string(),
        "plan" => plan,
        "filtered" => filtered_label,
    )
    .record(response_bytes as f64);
}

fn indexed_keyword_facet_counts(
    storage: &HelixGraphStorage,
    txn: Option<&heed3::RoTxn>,
    key: &str,
    candidates: Option<&HashSet<u128>>,
    filter: Option<&Filter>,
) -> Result<(HashMap<String, (serde_json::Value, u64)>, FacetTelemetry), GraphError> {
    let payload_indices = storage
        .payload_indices
        .read()
        .map_err(|e| GraphError::New(format!("Lock poisoned: {}", e)))?;
    let handle = payload_indices.get(key).ok_or_else(|| {
        GraphError::New(format!("Payload index required for facet key '{}'", key))
    })?;
    if !handle.is_ready() {
        return Err(GraphError::New(format!(
            "Payload index required for facet key '{}'",
            key
        )));
    }
    if handle.schema != PayloadIndexSchema::Keyword {
        return Err(GraphError::New(format!(
            "Keyword payload index required for facet key '{}'",
            key
        )));
    }

    let mut counts: HashMap<String, (serde_json::Value, u64)> = HashMap::new();
    let mut telemetry = FacetTelemetry {
        candidates: candidates.map(HashSet::len).unwrap_or(0),
        ..FacetTelemetry::default()
    };

    if storage.backend.kind() == BackendKind::Lsm {
        let db_name = HelixGraphStorage::payload_index_db_name(key, &handle.schema);
        let r = storage
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))?;

        if candidates.is_none() && filter.is_none() {
            let mut current_encoded: Option<Vec<u8>> = None;
            let mut current_count = 0_u64;
            let mut last_point_for_value: Option<u128> = None;

            storage
                .backend
                .scan(
                    &r,
                    Namespace::PayloadIndex(&db_name),
                    KeyRange::all(),
                    |encoded_value, id_bytes| {
                        telemetry.entries_scanned = telemetry.entries_scanned.saturating_add(1);
                        let Ok(id_arr) = <[u8; 16]>::try_from(id_bytes) else {
                            return true;
                        };
                        let point_id = u128::from_be_bytes(id_arr);
                        let is_new_value = current_encoded
                            .as_deref()
                            .map_or(true, |current| current != encoded_value);
                        if is_new_value {
                            if let Some(encoded) = current_encoded.take() {
                                record_encoded_facet_count(&mut counts, &encoded, current_count);
                            }
                            current_encoded = Some(encoded_value.to_vec());
                            current_count = 0;
                            last_point_for_value = None;
                        }
                        if last_point_for_value == Some(point_id) {
                            return true;
                        }
                        last_point_for_value = Some(point_id);
                        telemetry.matched_items = telemetry.matched_items.saturating_add(1);
                        current_count = current_count.saturating_add(1);
                        true
                    },
                )
                .map_err(|e| GraphError::New(e.to_string()))?;

            if let Some(encoded) = current_encoded {
                record_encoded_facet_count(&mut counts, &encoded, current_count);
            }
            return Ok((counts, telemetry));
        }

        let mut last_index_entry: Option<(Vec<u8>, u128)> = None;
        storage
            .backend
            .scan(
                &r,
                Namespace::PayloadIndex(&db_name),
                KeyRange::all(),
                |encoded_value, id_bytes| {
                    telemetry.entries_scanned = telemetry.entries_scanned.saturating_add(1);
                    let Ok(id_arr) = <[u8; 16]>::try_from(id_bytes) else {
                        return true;
                    };
                    let point_id = u128::from_be_bytes(id_arr);
                    if last_index_entry
                        .as_ref()
                        .is_some_and(|(last_value, last_id)| {
                            *last_id == point_id && last_value.as_slice() == encoded_value
                        })
                    {
                        return true;
                    }
                    last_index_entry = Some((encoded_value.to_vec(), point_id));
                    if let Some(candidates) = candidates {
                        if !candidates.contains(&point_id) {
                            return true;
                        }
                    }
                    if let Some(filter) = filter {
                        let node = match storage.get_node_be(&r, &point_id) {
                            Ok(node) => node,
                            Err(_) => return true,
                        };
                        if !filter.matches_point(Some(point_id), &node.properties) {
                            return true;
                        }
                    }
                    telemetry.matched_items = telemetry.matched_items.saturating_add(1);
                    let value: Value = match bincode::deserialize(encoded_value) {
                        Ok(value) => value,
                        Err(_) => return true,
                    };
                    let json_value = value_to_json(&value);
                    let count_key = json_value.to_string();
                    counts
                        .entry(count_key)
                        .and_modify(|entry| entry.1 += 1)
                        .or_insert((json_value, 1));
                    true
                },
            )
            .map_err(|e| GraphError::New(e.to_string()))?;
        return Ok((counts, telemetry));
    }

    let txn = txn.ok_or_else(|| {
        GraphError::StorageError("LMDB read transaction required for LMDB facet scan".to_string())
    })?;

    if candidates.is_none() && filter.is_none() {
        let mut current_encoded: Option<Vec<u8>> = None;
        let mut current_count = 0_u64;
        let mut last_point_for_value: Option<u128> = None;

        for item in handle.lmdb_db()?.iter(txn)? {
            telemetry.entries_scanned = telemetry.entries_scanned.saturating_add(1);
            let (encoded_value, id_bytes) = item?;
            let point_id = u128::from_be_bytes(
                id_bytes
                    .try_into()
                    .map_err(|_| GraphError::SliceLengthError)?,
            );
            let is_new_value = current_encoded
                .as_deref()
                .map_or(true, |current| current != encoded_value);
            if is_new_value {
                if let Some(encoded) = current_encoded.take() {
                    record_encoded_facet_count(&mut counts, &encoded, current_count);
                }
                current_encoded = Some(encoded_value.to_vec());
                current_count = 0;
                last_point_for_value = None;
            }
            if last_point_for_value == Some(point_id) {
                continue;
            }
            last_point_for_value = Some(point_id);
            telemetry.matched_items = telemetry.matched_items.saturating_add(1);
            current_count = current_count.saturating_add(1);
        }

        if let Some(encoded) = current_encoded {
            record_encoded_facet_count(&mut counts, &encoded, current_count);
        }
        return Ok((counts, telemetry));
    }

    let mut last_index_entry: Option<(Vec<u8>, u128)> = None;
    for item in handle.lmdb_db()?.iter(txn)? {
        telemetry.entries_scanned = telemetry.entries_scanned.saturating_add(1);
        let (encoded_value, id_bytes) = item?;
        let point_id = u128::from_be_bytes(
            id_bytes
                .try_into()
                .map_err(|_| GraphError::SliceLengthError)?,
        );
        if last_index_entry
            .as_ref()
            .is_some_and(|(last_value, last_id)| {
                *last_id == point_id && last_value.as_slice() == encoded_value
            })
        {
            continue;
        }
        last_index_entry = Some((encoded_value.to_vec(), point_id));
        if let Some(candidates) = candidates {
            if !candidates.contains(&point_id) {
                continue;
            }
        }
        if let Some(filter) = filter {
            let node = match storage.get_node(txn, &point_id) {
                Ok(node) => node,
                Err(_) => continue,
            };
            if !filter.matches_point(Some(point_id), &node.properties) {
                continue;
            }
        }
        telemetry.matched_items = telemetry.matched_items.saturating_add(1);
        let value: Value = match bincode::deserialize(encoded_value) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let json_value = value_to_json(&value);
        let count_key = json_value.to_string();
        counts
            .entry(count_key)
            .and_modify(|entry| entry.1 += 1)
            .or_insert((json_value, 1));
    }
    Ok((counts, telemetry))
}

fn record_encoded_facet_count(
    counts: &mut HashMap<String, (serde_json::Value, u64)>,
    encoded_value: &[u8],
    count: u64,
) {
    if count == 0 {
        return;
    }
    let Ok(value) = bincode::deserialize::<Value>(encoded_value) else {
        return;
    };
    let json_value = value_to_json(&value);
    let count_key = json_value.to_string();
    counts
        .entry(count_key)
        .and_modify(|entry| entry.1 = entry.1.saturating_add(count))
        .or_insert((json_value, count));
}

fn record_facet_value(counts: &mut HashMap<String, (serde_json::Value, u64)>, value: &Value) {
    let json_value = value_to_json(value);
    let count_key = json_value.to_string();
    counts
        .entry(count_key)
        .and_modify(|entry| entry.1 += 1)
        .or_insert((json_value, 1));
}

fn record_point_facet_values(
    counts: &mut HashMap<String, (serde_json::Value, u64)>,
    value: &Value,
) {
    match value {
        Value::Array(values) => {
            let mut seen = HashSet::new();
            for value in values {
                let json_value = value_to_json(value);
                let count_key = json_value.to_string();
                if !seen.insert(count_key.clone()) {
                    continue;
                }
                counts
                    .entry(count_key)
                    .and_modify(|entry| entry.1 += 1)
                    .or_insert((json_value, 1));
            }
        }
        _ => record_facet_value(counts, value),
    }
}

fn exact_keyword_facet_counts(
    storage: &HelixGraphStorage,
    txn: &heed3::RoTxn,
    key: &str,
    filter: Option<&Filter>,
) -> Result<(HashMap<String, (serde_json::Value, u64)>, FacetTelemetry), GraphError> {
    let mut counts: HashMap<String, (serde_json::Value, u64)> = HashMap::new();
    let mut telemetry = FacetTelemetry::default();
    let iter = storage.lmdb_nodes_db()?.iter(txn)?;
    for item in iter {
        telemetry.entries_scanned = telemetry.entries_scanned.saturating_add(1);
        let (id, data) = item?;
        let node = SerializedNode::decode_node(data, id)?;
        if let Some(filter) = filter {
            if !filter.matches_point(Some(id), &node.properties) {
                continue;
            }
        }
        let Some(value) = HelixGraphStorage::payload_value_for_key(&node.properties, key) else {
            continue;
        };
        telemetry.matched_items = telemetry.matched_items.saturating_add(1);
        record_point_facet_values(&mut counts, value);
    }
    Ok((counts, telemetry))
}

/// `POST /collections/{name}/facet` — count distinct values of a payload key.
///
/// Request body:
/// ```json
/// { "key": "language", "filter": { ... }, "limit": 10, "exact": true }
/// ```
/// Response:
/// ```json
/// { "hits": [ {"value": "rust", "count": 42}, {"value": "python", "count": 10} ] }
/// ```
pub fn handle_facet(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    let facet_started = Instant::now();
    let name = match get_collection_name(input) {
        Some(n) => n,
        None => return json_err(response, 400, "Missing collection name"),
    };
    let phase_start = Instant::now();
    let mut req: FacetRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_err(response, 400, &format!("Invalid JSON: {}", e)),
    };
    record_facet_phase(&name, "unknown", "parse_body", phase_start);
    if req.key.is_empty() {
        return json_err(response, 400, "Missing 'key' field");
    }
    req.limit = clamp_limit(req.limit);
    let exact = req.exact.unwrap_or(false);

    let phase_start = Instant::now();
    let storage = match input.collections.get_collection(&name) {
        Ok(s) => s,
        Err(e) => return json_err(response, 404, &e.to_string()),
    };
    record_facet_phase(&name, "unknown", "get_collection", phase_start);

    let filter = req.filter.unwrap_or_default();
    let filtered = !filter.is_empty();
    let phase_start = Instant::now();
    let (counts, telemetry, plan): (
        HashMap<String, (serde_json::Value, u64)>,
        FacetTelemetry,
        &'static str,
    ) = if storage.backend.kind() == BackendKind::Lsm {
        let plan = match (exact, filtered) {
            (true, true) => "indexed_exact_filtered",
            (false, true) => "indexed_filtered",
            (true, false) => "indexed_exact",
            (false, false) => "indexed",
        };
        match indexed_keyword_facet_counts(
            &storage,
            None,
            &req.key,
            None,
            filtered.then_some(&filter),
        ) {
            Ok((counts, telemetry)) => (counts, telemetry, plan),
            Err(e) => return json_err(response, 422, &e.to_string()),
        }
    } else {
        let txn = storage.begin_resize_safe_read_txn()?;
        record_facet_phase(&name, "unknown", "read_txn", phase_start);
        let indexed_candidates = if filtered {
            indexed_filter_candidates(&storage, &txn, &filter)?
        } else {
            None
        };
        record_facet_phase(&name, "unknown", "indexed_candidates", phase_start);

        if !filtered {
            let plan = if exact { "indexed_exact" } else { "indexed" };
            match indexed_keyword_facet_counts(&storage, Some(&txn), &req.key, None, None) {
                Ok((counts, telemetry)) => (counts, telemetry, plan),
                Err(_) => match exact_keyword_facet_counts(&storage, &txn, &req.key, None) {
                    Ok((counts, telemetry)) => (counts, telemetry, "exact"),
                    Err(e) => return json_err(response, 422, &e.to_string()),
                },
            }
        } else if let Some(candidates) = indexed_candidates.as_ref() {
            let plan = if exact {
                "indexed_exact_filtered"
            } else {
                "indexed_filtered"
            };
            match indexed_keyword_facet_counts(
                &storage,
                Some(&txn),
                &req.key,
                Some(candidates),
                Some(&filter),
            ) {
                Ok((counts, telemetry)) => (counts, telemetry, plan),
                Err(_) => match exact_keyword_facet_counts(&storage, &txn, &req.key, Some(&filter))
                {
                    Ok((counts, telemetry)) => (counts, telemetry, "exact"),
                    Err(e) => return json_err(response, 422, &e.to_string()),
                },
            }
        } else {
            match exact_keyword_facet_counts(&storage, &txn, &req.key, Some(&filter)) {
                Ok((counts, telemetry)) => (counts, telemetry, "exact"),
                Err(e) => return json_err(response, 422, &e.to_string()),
            }
        }
    };
    record_facet_phase(&name, plan, "count", phase_start);

    let phase_start = Instant::now();
    let mut hits: Vec<(serde_json::Value, u64)> = counts.into_values().collect();
    let distinct_values = hits.len();
    hits.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| a.0.to_string().cmp(&b.0.to_string()))
    });
    hits.truncate(req.limit);
    let returned_hits = hits.len();

    let hits_json: Vec<serde_json::Value> = hits
        .into_iter()
        .map(|(v, c)| serde_json::json!({ "value": v, "count": c }))
        .collect();
    record_facet_phase(&name, plan, "sort_response", phase_start);

    let result = json_ok(response, &serde_json::json!({ "hits": hits_json }));
    if result.is_ok() {
        observe_facet_result(
            &name,
            plan,
            filtered,
            exact,
            telemetry,
            distinct_values,
            returned_hits,
            response.body.len(),
        );
        record_facet_phase(&name, plan, "total", facet_started);
    }
    result
}

// ─── Delete ───

#[derive(Deserialize)]
struct DeletePointsRequest {
    #[serde(default)]
    points: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    filter: Option<Filter>,
}

/// POST /collections/{name}/points/delete
pub fn handle_delete_points(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let name = match get_collection_name(input) {
        Some(n) => n,
        None => return json_err(response, 400, "Missing collection name"),
    };

    let req: DeletePointsRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_err(response, 400, &format!("Invalid JSON: {}", e)),
    };
    let observe_hot_metrics = crate::telemetry::hot_path_metrics_enabled();
    let delete_started = Instant::now();
    let mut total_plan = "empty";

    if let Some(point_ids) = &req.points {
        total_plan = "direct";
        // Direct ID delete: pure mutation, no local state dependency.
        // Safe to forward to leader without local collection check.
        let ids: Vec<u128> = point_ids.iter().map(point_id_to_u128).collect();
        if observe_hot_metrics {
            metrics::histogram!("helix_qdrant_delete_matched_ids", "source" => "direct")
                .record(ids.len() as f64);
        }
        let phase_start = Instant::now();
        input.replication.apply(ReplicatedMutation::DeletePoints {
            collection: name.clone(),
            ids: ids.clone(),
        })?;
        if observe_hot_metrics {
            metrics::histogram!("helix_qdrant_delete_phase_ms", "phase" => "replication_apply", "plan" => "direct")
                .record(phase_start.elapsed().as_secs_f64() * 1000.0);
        }
        clear_pending_ids(&name, &ids);
    } else if let Some(filter) = &req.filter {
        // Filter-based delete: must read local state to resolve matching IDs.
        // In cluster mode, this runs on the leader via request forwarding.
        let phase_start = Instant::now();
        let storage = match input.collections.get_collection(&name) {
            Ok(s) => s,
            Err(e) => return json_err(response, 404, &e.to_string()),
        };
        if storage.backend.kind() == BackendKind::Lsm {
            let r = storage
                .backend
                .begin_read()
                .map_err(|e| GraphError::New(e.to_string()))?;
            let mut to_delete = Vec::new();
            let mut resolve_plan = "scan";
            if let Some(candidates) = indexed_filter_candidates_be(&storage, &r, filter)? {
                resolve_plan = "indexed";
                for id in candidates {
                    let node = match storage.get_node_be(&r, &id) {
                        Ok(node) => node,
                        Err(_) => continue,
                    };
                    if filter.matches_point(Some(id), &node.properties) {
                        to_delete.push(id);
                    }
                }
            } else {
                let mut scan_err: Option<GraphError> = None;
                storage
                    .backend
                    .scan(&r, Namespace::Nodes, KeyRange::all(), |k, data| {
                        let Ok(bytes) = <[u8; 16]>::try_from(k) else {
                            scan_err = Some(GraphError::New(
                                "invalid node key length during LSM delete scan".into(),
                            ));
                            return false;
                        };
                        let id = u128::from_be_bytes(bytes);
                        match SerializedNode::decode_node(data, id) {
                            Ok(node) => {
                                if filter.matches_point(Some(id), &node.properties) {
                                    to_delete.push(id);
                                }
                                true
                            }
                            Err(e) => {
                                scan_err = Some(e);
                                false
                            }
                        }
                    })
                    .map_err(|e| GraphError::New(e.to_string()))?;
                if let Some(e) = scan_err {
                    return Err(e);
                }
            }
            let pending_to_delete = pending_ids_matching_filter(&name, filter);
            to_delete.extend(pending_to_delete);
            to_delete.sort_unstable();
            to_delete.dedup();
            if observe_hot_metrics {
                metrics::histogram!("helix_qdrant_delete_phase_ms", "phase" => "filter_resolve", "plan" => resolve_plan)
                    .record(phase_start.elapsed().as_secs_f64() * 1000.0);
                metrics::histogram!("helix_qdrant_delete_matched_ids", "source" => resolve_plan)
                    .record(to_delete.len() as f64);
            }
            drop(r);

            total_plan = resolve_plan;
            let phase_start = Instant::now();
            input.replication.apply(ReplicatedMutation::DeletePoints {
                collection: name.clone(),
                ids: to_delete.clone(),
            })?;
            if observe_hot_metrics {
                metrics::histogram!("helix_qdrant_delete_phase_ms", "phase" => "replication_apply", "plan" => resolve_plan)
                    .record(phase_start.elapsed().as_secs_f64() * 1000.0);
            }
            clear_pending_ids(&name, &to_delete);
            if observe_hot_metrics {
                metrics::histogram!(
                    "helix_qdrant_delete_phase_ms",
                    "phase" => "total",
                    "plan" => total_plan
                )
                .record(delete_started.elapsed().as_secs_f64() * 1000.0);
            }
            return json_ok(response, &serde_json::json!({"status": "completed"}));
        }
        let rtxn = storage.begin_resize_safe_read_txn()?;
        let mut to_delete = Vec::new();
        let mut resolve_plan = "scan";
        if let Some(candidates) = indexed_filter_candidates(&storage, &rtxn, filter)? {
            resolve_plan = "indexed";
            for id in candidates {
                let node = match storage.get_node(&rtxn, &id) {
                    Ok(node) => node,
                    Err(_) => continue,
                };
                if filter.matches_point(Some(id), &node.properties) {
                    to_delete.push(id);
                }
            }
        } else {
            let iter = storage.lmdb_nodes_db()?.iter(&rtxn)?;
            for item in iter {
                let (id, data) = item?;
                let node = SerializedNode::decode_node(data, id)?;
                if filter.matches_point(Some(id), &node.properties) {
                    to_delete.push(id);
                }
            }
        }
        let pending_to_delete = pending_ids_matching_filter(&name, filter);
        to_delete.extend(pending_to_delete);
        to_delete.sort_unstable();
        to_delete.dedup();
        if observe_hot_metrics {
            metrics::histogram!("helix_qdrant_delete_phase_ms", "phase" => "filter_resolve", "plan" => resolve_plan)
                .record(phase_start.elapsed().as_secs_f64() * 1000.0);
            metrics::histogram!("helix_qdrant_delete_matched_ids", "source" => resolve_plan)
                .record(to_delete.len() as f64);
        }
        drop(rtxn);

        total_plan = resolve_plan;
        let phase_start = Instant::now();
        input.replication.apply(ReplicatedMutation::DeletePoints {
            collection: name.clone(),
            ids: to_delete.clone(),
        })?;
        if observe_hot_metrics {
            metrics::histogram!("helix_qdrant_delete_phase_ms", "phase" => "replication_apply", "plan" => resolve_plan)
                .record(phase_start.elapsed().as_secs_f64() * 1000.0);
        }
        clear_pending_ids(&name, &to_delete);
    }
    if observe_hot_metrics {
        metrics::histogram!(
            "helix_qdrant_delete_phase_ms",
            "phase" => "total",
            "plan" => total_plan
        )
        .record(delete_started.elapsed().as_secs_f64() * 1000.0);
    }
    json_ok(response, &serde_json::json!({"status": "completed"}))
}

// ─── Payload Index ───

#[derive(Deserialize)]
#[serde(untagged)]
enum FieldSchemaInput {
    Name(String),
    Object(FieldSchemaObject),
}

impl FieldSchemaInput {
    fn as_str(&self) -> &str {
        match self {
            FieldSchemaInput::Name(name) => name.as_str(),
            FieldSchemaInput::Object(obj) => obj.r#type.as_str(),
        }
    }
}

#[derive(Deserialize)]
struct FieldSchemaObject {
    #[serde(rename = "type")]
    r#type: String,
}

fn default_schema_input() -> FieldSchemaInput {
    FieldSchemaInput::Name("keyword".into())
}

#[derive(Deserialize)]
struct CreateIndexRequest {
    field_name: String,
    #[serde(default = "default_schema_input")]
    field_schema: FieldSchemaInput,
}

// ─── Set Payload ─────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct SetPayloadRequest {
    payload: HashMap<String, Value>,
    #[serde(default)]
    points: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    filter: Option<serde_json::Value>,
    #[serde(default)]
    key: Option<String>,
}

/// POST /collections/{name}/points/payload — merge payload fields into existing points.
pub fn handle_set_payload(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    let name = match get_collection_name(input) {
        Some(n) => n,
        None => return json_err(response, 400, "Missing collection name"),
    };
    let req: SetPayloadRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_err(response, 400, &format!("Invalid JSON: {}", e)),
    };

    let storage = input.collections.get_collection(&name)?;

    // Resolve target point IDs
    let point_ids: Vec<u128> = if let Some(points) = &req.points {
        points.iter().map(|v| point_id_to_u128(v)).collect()
    } else if let Some(_filter) = &req.filter {
        // Filter-based: scroll all points and apply filter
        // For now, return error — CE primarily uses point-ID-based set_payload
        return json_err(response, 400, "Filter-based set_payload not yet supported");
    } else {
        return json_err(response, 400, "Must provide 'points' or 'filter'");
    };

    if storage.backend.kind() == BackendKind::Lsm {
        let r = storage
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))?;
        let mut upserts = Vec::with_capacity(point_ids.len());
        for id in &point_ids {
            let mut props = storage.get_node_be(&r, id)?.properties;
            if let Some(payload_key) = req.key.as_deref().map(str::trim).filter(|k| !k.is_empty()) {
                let nested = props
                    .entry(payload_key.to_string())
                    .or_insert_with(|| Value::Object(HashMap::new()));
                if !matches!(nested, Value::Object(_)) {
                    *nested = Value::Object(HashMap::new());
                }
                if let Value::Object(existing) = nested {
                    for (key, value) in &req.payload {
                        existing.insert(key.clone(), value.clone());
                    }
                }
            } else {
                for (key, value) in &req.payload {
                    props.insert(key.clone(), value.clone());
                }
            }
            upserts.push(crate::helix_engine::storage_core::upsert::NodeUpsert {
                id: *id,
                label: "point".into(),
                properties: props,
            });
        }

        storage.with_write_backend(|w| {
            for upsert in &upserts {
                storage.upsert_node_be(w, upsert)?;
            }
            Ok(())
        })?;

        return json_ok(response, &sonic_rs::json!({"status": "completed"}));
    }

    storage.with_write_txn(|txn| {
        for id in &point_ids {
            // Read existing payload
            let mut props = storage.get_node(txn, id)?.properties;
            // Merge new payload fields. Qdrant's `key` parameter scopes the
            // patch under a payload object (CE uses key="metadata" for
            // backfill markers). Preserve nested object fields instead of
            // writing marker fields at the top level.
            if let Some(payload_key) = req.key.as_deref().map(str::trim).filter(|k| !k.is_empty()) {
                let nested = props
                    .entry(payload_key.to_string())
                    .or_insert_with(|| Value::Object(HashMap::new()));
                if !matches!(nested, Value::Object(_)) {
                    *nested = Value::Object(HashMap::new());
                }
                if let Value::Object(existing) = nested {
                    for (key, value) in &req.payload {
                        existing.insert(key.clone(), value.clone());
                    }
                }
            } else {
                for (key, value) in &req.payload {
                    props.insert(key.clone(), value.clone());
                }
            }
            // Write back
            let upsert = crate::helix_engine::storage_core::upsert::NodeUpsert {
                id: *id,
                label: "point".into(),
                properties: props,
            };
            storage.upsert_node(txn, &upsert)?;
        }
        Ok(())
    })?;

    json_ok(response, &sonic_rs::json!({"status": "completed"}))
}

// ─── Count Points ────────────────────────────────────────────────────────

#[derive(Deserialize, Default)]
struct CountPointsRequest {
    #[serde(default)]
    filter: Option<Filter>,
    #[serde(default)]
    exact: Option<bool>,
}

/// POST /collections/{name}/points/count — return point count with optional filter.
///
/// Qdrant parity: when a `filter` is supplied, the count must reflect only
/// matching points. Previously this handler ignored the body and always
/// returned the collection-wide `vector_count`, silently breaking any caller
/// relying on filter-scoped counts (tenant isolation, ingestion health checks,
/// per-path audits, etc.).
///
/// `exact: true` on an empty filter must also bypass the cached
/// `vector_count` shortcut: that counter is the LSM merge-key counter, which
/// can drift from the true point count (see the recount/reseed fix). Route it
/// into the same scan/candidate path used for a non-empty filter — an empty
/// `Filter::matches_point` is vacuously true for every point, so that path
/// already computes the exact total.
pub fn handle_count_points(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let name = match get_collection_name(input) {
        Some(n) => n,
        None => return json_err(response, 400, "Missing collection name"),
    };

    // Body is optional. Empty body means "count everything".
    let req: CountPointsRequest = if input.request.body.is_empty() {
        CountPointsRequest::default()
    } else {
        match sonic_rs::from_slice(&input.request.body) {
            Ok(r) => r,
            Err(e) => return json_err(response, 400, &format!("Invalid JSON: {}", e)),
        }
    };

    let storage = input.collections.get_collection(&name)?;
    let filter = req.filter.unwrap_or_default();
    let exact = req.exact.unwrap_or(false);
    let pending_points = pending_points_for_collection(&name);
    let pending_ids: HashSet<u128> = pending_points.iter().map(|(id, _)| *id).collect();

    if storage.backend.kind() == BackendKind::Lsm {
        let r = storage
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))?;
        if filter.is_empty() && !exact {
            let stats = storage.metadata_snapshot()?;
            let pending_new = pending_points
                .iter()
                .filter(|(id, _)| storage.get_node_be(&r, id).is_err())
                .count() as u64;
            return json_ok(
                response,
                &sonic_rs::json!({"count": stats.stats.vector_count.saturating_add(pending_new)}),
            );
        }

        let indexed_candidates = indexed_filter_candidates_be(&storage, &r, &filter)?;
        let mut count: u64 = 0;
        if let Some(candidates) = indexed_candidates {
            if indexed_filter_candidates_are_exact(&storage, &filter) {
                return json_ok(
                    response,
                    &sonic_rs::json!({
                        "count": count_exact_index_candidates_with_pending(
                            &candidates,
                            &pending_points,
                            &filter,
                        )
                    }),
                );
            }
            for id in candidates {
                let node = match storage.get_node_be(&r, &id) {
                    Ok(n) => n,
                    Err(_) => continue,
                };
                if !pending_ids.contains(&id) && filter.matches_point(Some(id), &node.properties) {
                    count += 1;
                }
            }
        } else {
            let mut scan_err: Option<GraphError> = None;
            storage
                .backend
                .scan(&r, Namespace::Nodes, KeyRange::all(), |k, data| {
                    let Ok(bytes) = <[u8; 16]>::try_from(k) else {
                        scan_err = Some(GraphError::New(
                            "invalid node key length during LSM count scan".into(),
                        ));
                        return false;
                    };
                    let id = u128::from_be_bytes(bytes);
                    match SerializedNode::decode_node(data, id) {
                        Ok(node) => {
                            if !pending_ids.contains(&id)
                                && filter.matches_point(Some(id), &node.properties)
                            {
                                count += 1;
                            }
                            true
                        }
                        Err(e) => {
                            scan_err = Some(e);
                            false
                        }
                    }
                })
                .map_err(|e| GraphError::New(e.to_string()))?;
            if let Some(e) = scan_err {
                return Err(e);
            }
        }

        count = count.saturating_add(
            pending_points
                .iter()
                .filter(|(id, point)| pending_matches_filter(*id, point, &filter))
                .count() as u64,
        );
        return json_ok(response, &sonic_rs::json!({"count": count}));
    }

    let txn = storage.begin_resize_safe_read_txn()?;

    if filter.is_empty() && !exact {
        let stats = storage.metadata_snapshot()?;
        let pending_new = pending_points
            .iter()
            .filter(|(id, _)| storage.get_node(&txn, id).is_err())
            .count() as u64;
        return json_ok(
            response,
            &sonic_rs::json!({"count": stats.stats.vector_count.saturating_add(pending_new)}),
        );
    }

    let indexed_candidates = indexed_filter_candidates(&storage, &txn, &filter)?;
    let mut count: u64 = 0;
    if let Some(candidates) = indexed_candidates {
        if indexed_filter_candidates_are_exact(&storage, &filter) {
            return json_ok(
                response,
                &sonic_rs::json!({
                    "count": count_exact_index_candidates_with_pending(
                        &candidates,
                        &pending_points,
                        &filter,
                    )
                }),
            );
        }
        for id in candidates {
            let node = match storage.get_node(&txn, &id) {
                Ok(n) => n,
                Err(_) => continue,
            };
            if !pending_ids.contains(&id) && filter.matches_point(Some(id), &node.properties) {
                count += 1;
            }
        }
    } else {
        let iter = storage.lmdb_nodes_db()?.iter(&txn)?;
        for item in iter {
            let (id, data) = item?;
            let node = SerializedNode::decode_node(data, id)?;
            if !pending_ids.contains(&id) && filter.matches_point(Some(id), &node.properties) {
                count += 1;
            }
        }
    }

    count = count.saturating_add(
        pending_points
            .iter()
            .filter(|(id, point)| pending_matches_filter(*id, point, &filter))
            .count() as u64,
    );

    json_ok(response, &sonic_rs::json!({"count": count}))
}

// ─── Get Points by ID ────────────────────────────────────────────────────

#[derive(Deserialize)]
struct GetPointsRequest {
    ids: Vec<serde_json::Value>,
    #[serde(default)]
    with_payload: Option<bool>,
    #[serde(default, alias = "with_vector")]
    with_vectors: WithVectors,
}

/// POST /collections/{name}/points — retrieve points by ID.
pub fn handle_get_points(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    let name = match get_collection_name(input) {
        Some(n) => n,
        None => return json_err(response, 400, "Missing collection name"),
    };
    let req: GetPointsRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_err(response, 400, &format!("Invalid JSON: {}", e)),
    };

    let storage = input.collections.get_collection(&name)?;
    if storage.backend.kind() == BackendKind::Lsm {
        let with_payload = req.with_payload.unwrap_or(true);
        let vec_names: Vec<String> = if req.with_vectors.is_active() {
            req.with_vectors
                .resolved_names(&known_vector_names(&storage))
        } else {
            Vec::new()
        };
        let hydrate_vectors = !vec_names.is_empty();
        let r = storage
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))?;

        let mut results: Vec<sonic_rs::Value> = Vec::new();
        for id_val in &req.ids {
            let id = point_id_to_u128(id_val);

            if let Some(pending) = pending_point(&name, id) {
                let payload = if with_payload {
                    serde_json_value_to_sonic(serde_json::Value::Object(
                        pending
                            .input
                            .payload
                            .clone()
                            .into_iter()
                            .collect::<serde_json::Map<_, _>>(),
                    ))
                } else {
                    sonic_rs::json!({})
                };
                let vector_json: sonic_rs::Value = if hydrate_vectors {
                    let mut filtered = serde_json::Map::new();
                    for vec_name in &vec_names {
                        if let Some(v) = pending.input.vector.get(vec_name) {
                            filtered.insert(vec_name.clone(), v.clone());
                        }
                    }
                    serde_json_value_to_sonic(serde_json::Value::Object(filtered))
                } else {
                    sonic_rs::json!({})
                };
                let id_str = format!("{:032x}", id);
                results.push(sonic_rs::json!({
                    "id": id_str,
                    "payload": payload,
                    "vector": vector_json,
                }));
                continue;
            }

            let node = match storage.get_node_be(&r, &id) {
                Ok(n) => n,
                Err(GraphError::NodeNotFound) => continue,
                Err(e) => return Err(e),
            };
            let payload = if with_payload {
                sonic_rs::to_value(&node.properties).unwrap_or(sonic_rs::json!({}))
            } else {
                sonic_rs::json!({})
            };
            let vector_json: sonic_rs::Value = if hydrate_vectors {
                serde_json_value_to_sonic(fetch_named_vectors_be(&storage, &r, id, &vec_names))
            } else {
                sonic_rs::json!({})
            };
            let id_str = format!("{:032x}", id);
            results.push(sonic_rs::json!({
                "id": id_str,
                "payload": payload,
                "vector": vector_json,
            }));
        }

        return json_ok(response, &results);
    }

    let txn = storage.begin_resize_safe_read_txn()?;
    let with_payload = req.with_payload.unwrap_or(true);

    let vec_names: Vec<String> = if req.with_vectors.is_active() {
        req.with_vectors
            .resolved_names(&known_vector_names(&storage))
    } else {
        Vec::new()
    };
    let hydrate_vectors = !vec_names.is_empty();

    let mut results: Vec<sonic_rs::Value> = Vec::new();
    for id_val in &req.ids {
        let id = point_id_to_u128(id_val);

        // WAL-ahead read fusion: a point that was acknowledged via the
        // wal-ahead path may not have landed in LMDB yet. Consult the
        // in-memory pending cache first so direct ID lookups return the
        // latest write (read-your-writes consistency on this endpoint).
        // Pending wins over LMDB by definition: if a write is pending it
        // is newer than what is in LMDB.
        if let Some(pending) = pending_point(&name, id) {
            let payload = if with_payload {
                serde_json_value_to_sonic(serde_json::Value::Object(
                    pending
                        .input
                        .payload
                        .clone()
                        .into_iter()
                        .collect::<serde_json::Map<_, _>>(),
                ))
            } else {
                sonic_rs::json!({})
            };
            let vector_json: sonic_rs::Value = if hydrate_vectors {
                let mut filtered = serde_json::Map::new();
                for vec_name in &vec_names {
                    if let Some(v) = pending.input.vector.get(vec_name) {
                        filtered.insert(vec_name.clone(), v.clone());
                    }
                }
                serde_json_value_to_sonic(serde_json::Value::Object(filtered))
            } else {
                sonic_rs::json!({})
            };
            let id_str = format!("{:032x}", id);
            results.push(sonic_rs::json!({
                "id": id_str,
                "payload": payload,
                "vector": vector_json,
            }));
            continue;
        }

        let node = match storage.get_node(&txn, &id) {
            Ok(n) => n,
            // A genuinely absent point is skipped (Qdrant get-by-id semantics).
            Err(GraphError::NodeNotFound) => continue,
            // A read that could not reach committed state (e.g. object storage
            // unreachable on an uncached LSM read, or decode corruption) must
            // surface as an error rather than be silently swallowed into a 200
            // with the point quietly missing.
            Err(e) => return Err(e),
        };
        let payload = if with_payload {
            sonic_rs::to_value(&node.properties).unwrap_or(sonic_rs::json!({}))
        } else {
            sonic_rs::json!({})
        };
        let vector_json: sonic_rs::Value = if hydrate_vectors {
            // fetch_named_vectors returns serde_json::Value; translate to sonic_rs::Value
            // through a single JSON round-trip (handler uses sonic_rs::json! elsewhere).
            let sj = fetch_named_vectors(&storage, &txn, id, &vec_names);
            serde_json_value_to_sonic(sj)
        } else {
            sonic_rs::json!({})
        };
        let id_str = format!("{:032x}", id);
        results.push(sonic_rs::json!({
            "id": id_str,
            "payload": payload,
            "vector": vector_json,
        }));
    }

    json_ok(response, &results)
}

/// Convert a `serde_json::Value` to a `sonic_rs::Value` via a single JSON
/// round-trip. Used by handlers that mostly speak `sonic_rs` but receive
/// data from helpers (`fetch_named_vectors`, WAL-ahead pending cache)
/// that return `serde_json::Value`.
fn serde_json_value_to_sonic(value: serde_json::Value) -> sonic_rs::Value {
    sonic_rs::from_slice(&serde_json::to_vec(&value).unwrap_or_default())
        .unwrap_or(sonic_rs::json!({}))
}

// ─── Collection Exists ───────────────────────────────────────────────────

/// GET /collections/{name}/exists — fast existence check.
pub fn handle_collection_exists(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let name = match get_collection_name(input) {
        Some(n) => n,
        None => return json_err(response, 400, "Missing collection name"),
    };
    let exists = input.collections.collection_exists(&name)?;
    json_ok(response, &sonic_rs::json!({"exists": exists}))
}

/// PUT /collections/{name}/index
pub fn handle_create_index(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let name = match get_collection_name(input) {
        Some(n) => n,
        None => return json_err(response, 400, "Missing collection name"),
    };

    let req: CreateIndexRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_err(response, 400, &format!("Invalid JSON: {}", e)),
    };

    // NOTE: No local get_collection check. The Raft apply on the leader
    // validates collection existence. Removing this prevents followers
    // from rejecting valid requests when their local state lags.

    let field_schema = req.field_schema.as_str();
    let Some(schema) = parse_payload_index_schema(field_schema) else {
        return json_err(
            response,
            400,
            &format!("Unsupported field_schema '{}'", field_schema),
        );
    };

    let field_name = req.field_name;
    let result = input
        .replication
        .apply(ReplicatedMutation::CreatePayloadIndex {
            collection: name.clone(),
            field_name: field_name.clone(),
            schema,
        });
    match result {
        Ok(()) => {
            let index_status = input
                .collections
                .get_loaded_collection(&name)?
                .and_then(|storage| storage.payload_index_status(&field_name).ok().flatten());

            json_ok(
                response,
                &serde_json::json!({
                    "status": "acknowledged",
                    "index": index_status,
                }),
            )
        }
        Err(e @ GraphError::ResizeBackpressure(_)) => json_err(response, 429, &e.to_string()),
        Err(e) => json_err(response, 500, &e.to_string()),
    }
}

/// DELETE /collections/{name}/index/{field_name}
pub fn handle_delete_index(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let name = match get_collection_name(input) {
        Some(n) => n,
        None => return json_err(response, 400, "Missing collection name"),
    };
    let field_name = match get_field_name(input) {
        Some(n) if !n.is_empty() => n,
        _ => return json_err(response, 400, "Missing field name"),
    };

    match input
        .replication
        .apply(ReplicatedMutation::DeletePayloadIndex {
            collection: name,
            field_name,
        }) {
        Ok(()) => json_ok(response, &serde_json::json!({ "status": "acknowledged" })),
        Err(e) => json_err(response, 500, &e.to_string()),
    }
}

// ─── Query (Prefetch + Fusion) ───

#[derive(Deserialize)]
struct QueryRequest {
    #[serde(default)]
    prefetch: Vec<PrefetchQuery>,
    /// Accepts three shapes:
    /// - `{fusion: "rrf"}` — fuse prefetch results
    /// - `[0.1, 0.2, ...]` — dense single-vector query
    /// - `{indices: [...], values: [...]}` — sparse single-vector query
    #[serde(default)]
    query: Option<serde_json::Value>,
    /// Named vector space to search (for single-vector mode).
    #[serde(default)]
    using: Option<String>,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    offset: usize,
    #[serde(default)]
    score_threshold: Option<f32>,
    #[serde(default = "default_true")]
    with_payload: bool,
    #[serde(default)]
    filter: Option<Filter>,
    #[serde(default)]
    params: Option<SearchParams>,
}

#[derive(Deserialize)]
struct PrefetchQuery {
    query: Vec<f32>,
    #[serde(default = "default_dense")]
    using: String,
    #[serde(default = "default_prefetch_limit")]
    limit: usize,
    #[serde(default)]
    params: Option<SearchParams>,
}

#[derive(Deserialize)]
struct HybridQueryRequest {
    #[serde(default)]
    dense: Vec<HybridDenseQuery>,
    #[serde(default)]
    sparse: Vec<HybridSparseQuery>,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default = "default_true")]
    with_payload: bool,
    #[serde(default)]
    with_channel_points: bool,
    #[serde(default)]
    filter: Option<Filter>,
    #[serde(default)]
    rrf_k: Option<f64>,
    #[serde(default)]
    mmr: Option<HybridMmrRequest>,
    #[serde(default)]
    graph: Option<HybridGraphRequest>,
}

#[derive(Deserialize)]
struct HybridDenseQuery {
    query: Vec<f32>,
    #[serde(default = "default_dense")]
    using: String,
    #[serde(default = "default_prefetch_limit")]
    limit: usize,
    #[serde(default)]
    params: Option<SearchParams>,
    /// Optional per-entry filter. When present, REPLACES the request-level
    /// `filter` for this channel (matches Qdrant Query API prefetch semantics).
    /// This lets callers scope lexical-only restrictions (e.g. pid pre-gating
    /// from ReFRAG) to dense channels without affecting sparse recall.
    #[serde(default)]
    filter: Option<Filter>,
}

#[derive(Deserialize)]
struct HybridSparseQuery {
    query: SparseVectorInput,
    using: String,
    #[serde(default = "default_prefetch_limit")]
    limit: usize,
    /// See `HybridDenseQuery::filter`.
    #[serde(default)]
    filter: Option<Filter>,
}

#[derive(Clone, Deserialize)]
#[serde(untagged)]
enum HybridMmrRequest {
    Enabled(bool),
    Config(HybridMmrConfig),
}

#[derive(Clone, Deserialize)]
struct HybridMmrConfig {
    #[serde(default = "default_true")]
    enabled: bool,
    #[serde(default = "default_mmr_lambda")]
    lambda: f64,
}

impl HybridMmrRequest {
    fn config(&self) -> Option<HybridMmrConfig> {
        match self {
            HybridMmrRequest::Enabled(false) => None,
            HybridMmrRequest::Enabled(true) => Some(HybridMmrConfig {
                enabled: true,
                lambda: default_mmr_lambda(),
            }),
            HybridMmrRequest::Config(config) if config.enabled => Some(config.clone()),
            HybridMmrRequest::Config(_) => None,
        }
    }
}

/// Opt-in GRAPH-SIGNAL third fusion channel. Graph node ids and vector point
/// ids are the same deterministic u128, so the top hybrid hits seed a bounded
/// personalized PageRank over the code graph; the PPR ranking joins dense and
/// sparse as a third list in Reciprocal Rank Fusion — structural relevance
/// fused with semantic+lexical relevance.
#[derive(Clone, Deserialize)]
#[serde(untagged)]
enum HybridGraphRequest {
    Enabled(bool),
    Config(HybridGraphConfig),
}

#[derive(Clone, Default, Deserialize)]
struct HybridGraphConfig {
    #[serde(default = "default_true")]
    enabled: bool,
    /// Edge labels to walk. Absent means the CE code-graph defaults
    /// (CALLS + IMPORTS, mirroring /graph/pagerank); an explicit empty list
    /// walks nothing, so the channel degrades to two-channel fusion.
    #[serde(default)]
    edge_labels: Option<Vec<String>>,
    #[serde(default)]
    alpha: Option<f64>,
    #[serde(default)]
    direction: Option<String>,
    #[serde(default)]
    seed_count: Option<usize>,
    #[serde(default)]
    max_nodes: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
}

impl HybridGraphRequest {
    fn config(&self) -> Option<HybridGraphConfig> {
        match self {
            HybridGraphRequest::Enabled(false) => None,
            HybridGraphRequest::Enabled(true) => Some(HybridGraphConfig {
                enabled: true,
                ..HybridGraphConfig::default()
            }),
            HybridGraphRequest::Config(config) if config.enabled => Some(config.clone()),
            HybridGraphRequest::Config(_) => None,
        }
    }
}

/// Validated, defaulted graph-channel parameters.
struct HybridGraphResolved {
    edge_labels: Vec<String>,
    alpha: f64,
    direction: PprDirection,
    direction_label: &'static str,
    seed_count: usize,
    max_nodes: usize,
    limit: usize,
}

fn resolve_hybrid_graph_config(config: &HybridGraphConfig) -> Result<HybridGraphResolved, String> {
    let alpha = config.alpha.unwrap_or(0.15);
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err("graph.alpha must be between 0.0 and 1.0 (exclusive)".to_string());
    }
    let (direction, direction_label) = match config.direction.as_deref() {
        None | Some("both") => (PprDirection::Both, "both"),
        Some("out") => (PprDirection::Out, "out"),
        Some("in") => (PprDirection::In, "in"),
        Some(_) => {
            return Err("graph.direction must be one of \"out\", \"in\", or \"both\"".to_string())
        }
    };
    Ok(HybridGraphResolved {
        edge_labels: config
            .edge_labels
            .clone()
            .unwrap_or_else(|| vec!["CALLS".to_string(), "IMPORTS".to_string()]),
        alpha,
        direction,
        direction_label,
        seed_count: config.seed_count.unwrap_or(10).clamp(1, 100),
        max_nodes: config.max_nodes.unwrap_or(10_000).min(100_000),
        limit: config.limit.unwrap_or(100).clamp(1, 1000),
    })
}

/// Execute the graph-signal channel: seed a bounded personalized PageRank with
/// the preliminary RRF fusion of the dense+sparse ranked lists and return the
/// PPR ranking as a third list for the final fusion, plus the response summary.
///
/// `neighbors` unions adjacency for one node across the resolved labels and
/// direction. A walk that never crossed an edge is pure seed echo (PPR returns
/// the seed mass unchanged), so the list is emptied and the final fusion is
/// identical to the two-channel result.
///
/// `admit` gates which PPR-ranked ids may enter the fusion: adjacency is not
/// filter-aware, so without it a neighbor from another repo (or a deleted
/// point still referenced by a stale edge) would bypass the request filter.
fn run_hybrid_graph_channel<F, A>(
    ranked_lists: &[Vec<RankedItem>],
    rrf_k: Option<f64>,
    resolved: &HybridGraphResolved,
    mut neighbors: F,
    admit: A,
) -> (Vec<RankedItem>, serde_json::Value)
where
    F: FnMut(u128) -> Vec<u128>,
    A: FnMut(u128) -> bool,
{
    let preliminary = match rrf_k {
        Some(k) => rrf_fusion_with_k(ranked_lists, k, resolved.seed_count),
        None => rrf_fusion(ranked_lists, resolved.seed_count),
    };
    let seeds: Vec<(u128, f64)> = preliminary
        .into_iter()
        .map(|item| (item.id, item.score))
        .collect();
    let params = PprParams {
        alpha: resolved.alpha,
        max_pushed_nodes: resolved.max_nodes,
        limit: resolved.limit,
        ..PprParams::default()
    };
    let mut saw_edges = false;
    let mut graph_list = personalized_pagerank_filtered(
        &seeds,
        |node| {
            let peers = neighbors(node);
            if !peers.is_empty() {
                saw_edges = true;
            }
            peers
        },
        admit,
        &params,
    );
    if !saw_edges {
        graph_list.clear();
    }
    let summary = serde_json::json!({
        "seeds": seeds.len(),
        "alpha": resolved.alpha,
        "direction": resolved.direction_label,
        "edge_labels": resolved.edge_labels,
        "returned": graph_list.len(),
    });
    (graph_list, summary)
}

fn default_dense() -> String {
    "dense".into()
}
fn default_prefetch_limit() -> usize {
    100
}
fn default_mmr_lambda() -> f64 {
    0.7
}

/// Detect the kind of query from the JSON value.
enum QueryKind {
    #[allow(dead_code)]
    Fusion(String),
    DenseVector(Vec<f32>),
    SparseVector(SparseVector),
}

fn classify_query(val: &serde_json::Value) -> Result<QueryKind, String> {
    // Array → dense vector
    if let Some(arr) = val.as_array() {
        let v: Vec<f32> = arr
            .iter()
            .map(|x| x.as_f64().unwrap_or(0.0) as f32)
            .collect();
        return Ok(QueryKind::DenseVector(v));
    }
    // Object — check for indices/values (sparse) or fusion
    if let Some(obj) = val.as_object() {
        if let Some(nearest) = obj.get("nearest") {
            return classify_query(nearest);
        }
        if obj.contains_key("indices") && obj.contains_key("values") {
            let indices: Vec<u32> = obj["indices"]
                .as_array()
                .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as u32).collect())
                .unwrap_or_default();
            let values: Vec<f32> = obj["values"]
                .as_array()
                .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect())
                .unwrap_or_default();
            return Ok(QueryKind::SparseVector(SparseVector { indices, values }));
        }
        if let Some(fusion_val) = obj.get("fusion") {
            let fusion = fusion_val.as_str().unwrap_or("rrf").to_string();
            return Ok(QueryKind::Fusion(fusion));
        }
    }
    Err("query must be a dense vector array, sparse {indices, values} object, or {fusion: \"rrf\"} object".into())
}

/// POST /collections/{name}/points/query
pub fn handle_query_points(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    const ENDPOINT: &str = "/collections/{collection}/points/query";
    let request_start = Instant::now();
    let name = match get_collection_name(input) {
        Some(n) => n,
        None => return json_err(response, 400, "Missing collection name"),
    };

    let parse_start = Instant::now();
    let mut req: QueryRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => {
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                "unknown",
                "unknown",
                "parse_body",
                parse_start,
            );
            record_qdrant_query_outcome(
                &name,
                ENDPOINT,
                "unknown",
                "unknown",
                "bad_request",
                request_start,
            );
            return json_err(response, 400, &format!("Invalid JSON: {}", e));
        }
    };
    record_qdrant_query_stage(
        &name,
        ENDPOINT,
        "unknown",
        "unknown",
        "parse_body",
        parse_start,
    );
    if let Err(message) = validate_score_threshold(req.score_threshold) {
        return json_err(response, 400, message);
    }
    req.limit = clamp_limit(req.limit);
    req.offset = clamp_offset(req.offset);
    let result_window = result_window(req.limit, req.offset);
    for p in req.prefetch.iter_mut() {
        p.limit = clamp_limit(p.limit);
    }

    let collection_start = Instant::now();
    let storage = match input.collections.get_collection(&name) {
        Ok(s) => {
            if s.backend.kind() == BackendKind::Lsm {
                let backend = backend_label(s.backend.kind());
                record_qdrant_query_stage(
                    &name,
                    ENDPOINT,
                    backend,
                    "unknown",
                    "collection_get",
                    collection_start,
                );
            }
            s
        }
        Err(e) => {
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                "unknown",
                "unknown",
                "collection_get",
                collection_start,
            );
            record_qdrant_query_outcome(
                &name,
                ENDPOINT,
                "unknown",
                "unknown",
                "collection_not_found",
                request_start,
            );
            return json_err(response, 404, &e.to_string());
        }
    };
    let backend = backend_label(storage.backend.kind());

    if storage.backend.kind() == BackendKind::Lsm {
        let read_start = Instant::now();
        let r = storage
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))?;
        record_qdrant_query_stage(
            &name,
            ENDPOINT,
            backend,
            "unknown",
            "begin_read",
            read_start,
        );
        let filter = req.filter.unwrap_or_default();
        let filter_start = Instant::now();
        let ProbedCandidates {
            candidates: indexed_candidates,
            probe_estimate,
            probe_skipped,
        } = vector_query_indexed_filter_candidates_be_probed(
            &storage,
            &r,
            &filter,
            filter_count_probe_enabled(),
        )?;
        record_qdrant_query_stage(
            &name,
            ENDPOINT,
            backend,
            "unknown",
            "filter_candidates",
            filter_start,
        );
        let has_filter = !filter.is_empty();
        let filter_index_exact =
            indexed_candidates.is_some() && indexed_filter_candidates_are_exact(&storage, &filter);
        let filter_recheck_required = has_filter && !filter_index_exact;
        let classify_start = Instant::now();
        let query_kind = match &req.query {
            Some(val) => match classify_query(val) {
                Ok(k) => Some(k),
                Err(e) => {
                    record_qdrant_query_stage(
                        &name,
                        ENDPOINT,
                        backend,
                        "unknown",
                        "classify_query",
                        classify_start,
                    );
                    record_qdrant_query_outcome(
                        &name,
                        ENDPOINT,
                        backend,
                        "unknown",
                        "bad_request",
                        request_start,
                    );
                    return json_err(response, 400, &e);
                }
            },
            None => {
                if req.prefetch.is_empty() {
                    record_qdrant_query_stage(
                        &name,
                        ENDPOINT,
                        backend,
                        "unknown",
                        "classify_query",
                        classify_start,
                    );
                    record_qdrant_query_outcome(
                        &name,
                        ENDPOINT,
                        backend,
                        "unknown",
                        "bad_request",
                        request_start,
                    );
                    return json_err(response, 400, "Either 'query' or 'prefetch' is required");
                }
                Some(QueryKind::Fusion("rrf".into()))
            }
        };
        let query_kind_label = query_kind
            .as_ref()
            .map(query_kind_label)
            .unwrap_or("unknown");
        record_qdrant_query_stage(
            &name,
            ENDPOINT,
            backend,
            query_kind_label,
            "classify_query",
            classify_start,
        );
        if let Some(candidates) = &indexed_candidates {
            record_qdrant_query_count(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "filter_candidates",
                candidates.len(),
            );
        }
        if let Some(estimate) = probe_estimate {
            record_qdrant_query_count(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "filter_candidate_estimate",
                estimate,
            );
        }
        record_qdrant_query_count(
            &name,
            ENDPOINT,
            backend,
            query_kind_label,
            "limit",
            req.limit,
        );
        record_qdrant_query_count(
            &name,
            ENDPOINT,
            backend,
            query_kind_label,
            "prefetch_channels",
            req.prefetch.len(),
        );
        let total_vectors = indexed_candidates
            .as_ref()
            .map(|_| collection_vector_count_be(&storage))
            .unwrap_or(0);
        if total_vectors > 0 {
            record_qdrant_query_count(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "total_vectors",
                total_vectors as usize,
            );
        }
        let selectivity_hint = if has_filter {
            indexed_candidates.as_ref().and_then(|candidates| {
                (total_vectors > 0).then_some(candidates.len() as f32 / total_vectors as f32)
            })
        } else {
            None
        };

        if let Some(QueryKind::DenseVector(dense_vec)) = &query_kind {
            if dense_vec.is_empty() {
                return json_err(response, 400, "Query vector must not be empty");
            }
            let using = req.using.as_deref().unwrap_or("dense");
            let score_threshold =
                dense_score_threshold_for_metric(&storage, using, req.score_threshold);
            let search_limit = if has_filter || indexed_candidates.is_some() {
                result_window.saturating_mul(4)
            } else {
                result_window
            };
            let ef_override = req.params.as_ref().and_then(|p| p.hnsw_ef);
            let filter_fn = |id: u128| -> bool {
                if let Some(candidates) = &indexed_candidates {
                    if !candidates.contains(&id) {
                        return false;
                    }
                }
                if !filter_recheck_required {
                    return true;
                }
                match storage.get_node_be(&r, &id) {
                    Ok(node) => filter.matches_point(Some(id), &node.properties),
                    Err(_) => false,
                }
            };
            let dense_filter_active = has_filter || indexed_candidates.is_some();
            let exact_index_filter_candidates = exact_index_filter_candidates(
                has_filter,
                filter_recheck_required,
                indexed_candidates.as_ref(),
            );
            let candidate_start = Instant::now();
            let exact_candidate_ids = exact_dense_candidate_ids_be(
                &storage,
                &r,
                indexed_candidates.as_ref(),
                &filter,
                filter_recheck_required,
                total_vectors,
            );
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "dense_candidate_plan",
                candidate_start,
            );
            if let Some(candidate_ids) = &exact_candidate_ids {
                record_qdrant_query_count(
                    &name,
                    ENDPOINT,
                    backend,
                    query_kind_label,
                    "exact_candidate_ids",
                    candidate_ids.len(),
                );
            }
            observe_dense_search_plan(
                &name,
                select_dense_search_plan(
                    exact_candidate_ids.is_some(),
                    exact_index_filter_candidates.is_some(),
                    probe_skipped,
                    dense_filter_active,
                ),
                probe_estimate,
            );
            let search_start = Instant::now();
            let mut results = if let Some(candidate_ids) = exact_candidate_ids {
                storage.named_vectors.dense_search_candidate_ids_exact_be(
                    &r,
                    using,
                    dense_vec,
                    search_limit,
                    candidate_ids.iter().copied(),
                    selectivity_hint,
                )
            } else if let Some(candidate_ids) = exact_index_filter_candidates {
                storage
                    .named_vectors
                    .dense_search_candidate_set_filter_ef_be(
                        &r,
                        using,
                        dense_vec,
                        search_limit,
                        candidate_ids,
                        true,
                        selectivity_hint,
                        ef_override,
                    )
            } else {
                storage.named_vectors.dense_search_with_id_filter_ef_be(
                    &r,
                    using,
                    dense_vec,
                    search_limit,
                    dense_filter_active.then_some(&[filter_fn][..]),
                    true,
                    selectivity_hint,
                    ef_override,
                )
            }
            .map_err(GraphError::from)?;
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "dense_search",
                search_start,
            );
            record_qdrant_query_count(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "search_results_raw",
                results.len(),
            );
            results.truncate(result_window);
            let score_start = Instant::now();
            for result in &mut results {
                if let Some(distance) = result.distance {
                    result.distance = Some(
                        storage
                            .named_vectors
                            .public_score(using, distance)
                            .map_err(GraphError::from)?,
                    );
                }
            }
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "score_normalize",
                score_start,
            );
            let fuse_start = Instant::now();
            fuse_pending_dense_results(
                &name,
                &filter,
                using,
                dense_vec,
                result_window,
                &mut results,
            );
            apply_hvector_score_threshold_and_offset(
                &mut results,
                score_threshold,
                req.offset,
                req.limit,
            );
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "pending_fuse",
                fuse_start,
            );
            record_qdrant_query_count(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "results_returned",
                results.len(),
            );
            let response_start = Instant::now();
            let result =
                build_query_response_be(response, &name, &storage, &r, &results, req.with_payload);
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "response_build",
                response_start,
            );
            record_qdrant_query_outcome(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                if result.is_ok() { "ok" } else { "error" },
                request_start,
            );
            return result;
        }

        if let Some(QueryKind::SparseVector(sparse_vec)) = &query_kind {
            let Some(using) = req.using.as_deref() else {
                return json_err(
                    response,
                    400,
                    "'using' is required for sparse vector queries",
                );
            };
            let filter_fn = |doc_id: u128| -> bool {
                if let Some(candidates) = &indexed_candidates {
                    if !candidates.contains(&doc_id) {
                        return false;
                    }
                }
                if !filter_recheck_required {
                    return true;
                }
                match storage.get_node_be(&r, &doc_id) {
                    Ok(node) => filter.matches_point(Some(doc_id), &node.properties),
                    Err(_) => false,
                }
            };
            let search_limit = if has_filter || indexed_candidates.is_some() {
                result_window.saturating_mul(4)
            } else {
                result_window
            };
            let candidate_start = Instant::now();
            let exact_candidate_ids = bounded_filter_candidate_ids_be(
                &storage,
                &r,
                indexed_candidates.as_ref(),
                &filter,
                filter_recheck_required,
                total_vectors,
            );
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "sparse_candidate_plan",
                candidate_start,
            );
            if let Some(candidate_ids) = &exact_candidate_ids {
                record_qdrant_query_count(
                    &name,
                    ENDPOINT,
                    backend,
                    query_kind_label,
                    "exact_candidate_ids",
                    candidate_ids.len(),
                );
            }
            let search_start = Instant::now();
            let mut scored = storage
                .named_vectors
                .with_sparse_core(using, |core| {
                    let mut results = if let Some(candidate_ids) = exact_candidate_ids.as_ref() {
                        core.search_candidate_ids_exact_be(
                            &r,
                            sparse_vec,
                            search_limit,
                            candidate_ids.iter().copied(),
                        )?
                    } else {
                        core.search_be(&r, sparse_vec, search_limit, Some(&filter_fn))?
                    };
                    results.truncate(result_window);
                    Ok(results)
                })
                .map_err(GraphError::from)?;
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "sparse_search",
                search_start,
            );
            record_qdrant_query_count(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "search_results_raw",
                scored.len(),
            );
            let fuse_start = Instant::now();
            fuse_pending_sparse_results(
                &name,
                &filter,
                using,
                sparse_vec,
                result_window,
                &mut scored,
            );
            apply_sparse_score_threshold_and_offset(
                &mut scored,
                req.score_threshold,
                req.offset,
                req.limit,
            );
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "pending_fuse",
                fuse_start,
            );
            record_qdrant_query_count(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "results_returned",
                scored.len(),
            );
            let response_start = Instant::now();
            let result = build_sparse_query_response_be(
                response,
                &name,
                &storage,
                &r,
                &scored,
                req.with_payload,
            );
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "response_build",
                response_start,
            );
            record_qdrant_query_outcome(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                if result.is_ok() { "ok" } else { "error" },
                request_start,
            );
            return result;
        }

        let mut ranked_lists: Vec<Vec<RankedItem>> = Vec::new();
        for pf in &req.prefetch {
            let channel_window = channel_result_window(pf.limit, result_window, req.offset);
            let search_limit = if has_filter || indexed_candidates.is_some() {
                channel_window.saturating_mul(4)
            } else {
                channel_window
            };
            let pf_ef = pf.params.as_ref().and_then(|p| p.hnsw_ef);
            let filter_fn = |id: u128| -> bool {
                if let Some(candidates) = &indexed_candidates {
                    if !candidates.contains(&id) {
                        return false;
                    }
                }
                if !filter_recheck_required {
                    return true;
                }
                match storage.get_node_be(&r, &id) {
                    Ok(node) => filter.matches_point(Some(id), &node.properties),
                    Err(_) => false,
                }
            };
            let dense_filter_active = has_filter || indexed_candidates.is_some();
            let exact_index_filter_candidates = exact_index_filter_candidates(
                has_filter,
                filter_recheck_required,
                indexed_candidates.as_ref(),
            );
            let candidate_start = Instant::now();
            let exact_candidate_ids = exact_dense_candidate_ids_be(
                &storage,
                &r,
                indexed_candidates.as_ref(),
                &filter,
                filter_recheck_required,
                total_vectors,
            );
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "fusion_dense_candidate_plan",
                candidate_start,
            );
            if let Some(candidate_ids) = &exact_candidate_ids {
                record_qdrant_query_count(
                    &name,
                    ENDPOINT,
                    backend,
                    query_kind_label,
                    "exact_candidate_ids",
                    candidate_ids.len(),
                );
            }
            let search_start = Instant::now();
            let mut results = if let Some(candidate_ids) = exact_candidate_ids {
                storage.named_vectors.dense_search_candidate_ids_exact_be(
                    &r,
                    &pf.using,
                    &pf.query,
                    search_limit,
                    candidate_ids.iter().copied(),
                    selectivity_hint,
                )
            } else if let Some(candidate_ids) = exact_index_filter_candidates {
                storage
                    .named_vectors
                    .dense_search_candidate_set_filter_ef_be(
                        &r,
                        &pf.using,
                        &pf.query,
                        search_limit,
                        candidate_ids,
                        true,
                        selectivity_hint,
                        pf_ef,
                    )
            } else {
                storage.named_vectors.dense_search_with_id_filter_ef_be(
                    &r,
                    &pf.using,
                    &pf.query,
                    search_limit,
                    dense_filter_active.then_some(&[filter_fn][..]),
                    true,
                    selectivity_hint,
                    pf_ef,
                )
            }
            .map_err(GraphError::from)?;
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "fusion_dense_search",
                search_start,
            );
            record_qdrant_query_count(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "search_results_raw",
                results.len(),
            );
            results.truncate(channel_window);
            let score_start = Instant::now();
            for result in &mut results {
                if let Some(distance) = result.distance {
                    result.distance = Some(
                        storage
                            .named_vectors
                            .public_score(&pf.using, distance)
                            .map_err(GraphError::from)?,
                    );
                }
            }
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "score_normalize",
                score_start,
            );
            let fuse_start = Instant::now();
            fuse_pending_dense_results(
                &name,
                &filter,
                &pf.using,
                &pf.query,
                channel_window,
                &mut results,
            );
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "pending_fuse",
                fuse_start,
            );
            ranked_lists.push(
                results
                    .into_iter()
                    .map(|v| RankedItem {
                        id: v.id,
                        score: v.distance.unwrap_or(0.0) as f64,
                    })
                    .collect(),
            );
        }

        let mut fused = if ranked_lists.len() > 1 {
            rrf_fusion(&ranked_lists, result_window)
        } else if ranked_lists.len() == 1 {
            ranked_lists
                .into_iter()
                .next()
                .unwrap_or_default()
                .into_iter()
                .take(result_window)
                .collect()
        } else {
            Vec::new()
        };
        apply_ranked_score_threshold_and_offset(
            &mut fused,
            req.score_threshold,
            req.offset,
            req.limit,
        );

        record_qdrant_query_count(
            &name,
            ENDPOINT,
            backend,
            query_kind_label,
            "results_returned",
            fused.len(),
        );
        let response_start = Instant::now();
        let payload_cache = if req.with_payload {
            lsm_payload_cache_be(&name, &storage, &r, fused.iter().map(|item| item.id))
                .map_err(graph_error_from_backend_error)?
        } else {
            None
        };
        let results: Vec<serde_json::Value> = fused
            .iter()
            .map(|item| {
                query_point_json_be(
                    &name,
                    &storage,
                    &r,
                    item.id,
                    item.score,
                    req.with_payload,
                    payload_cache.as_ref(),
                )
            })
            .collect();
        let result = json_ok(response, &serde_json::json!({ "points": results }));
        record_qdrant_query_stage(
            &name,
            ENDPOINT,
            backend,
            query_kind_label,
            "response_build",
            response_start,
        );
        record_qdrant_query_outcome(
            &name,
            ENDPOINT,
            backend,
            query_kind_label,
            if result.is_ok() { "ok" } else { "error" },
            request_start,
        );
        return result;
    }

    let txn = storage.begin_resize_safe_read_txn()?;
    let dense_read_txns = storage.nested_dense_read_provider(&txn);
    let filter = req.filter.unwrap_or_default();
    // Use the simpler indexed_filter_candidates planner.
    //
    // Earlier we swapped this for `vector_query_indexed_filter_candidates`
    // (the broad-filter cutoff used by /points/search and
    // /points/hybrid_query). Under live traffic that change correlated
    // with SIGSEGVs every ~4 minutes (image b51f6837): the pod entered
    // a readiness-failure loop, /health stopped responding, and helix-0
    // crashed and restarted. Rolled back. The planner swap is still a
    // desirable change but needs an isolated repro + a guard for the
    // Some(empty HashSet) corner case before re-attempting.
    let indexed_candidates = indexed_filter_candidates(&storage, &txn, &filter)?;
    let has_filter = !filter.is_empty();

    // Classify the top-level query to determine single-vector vs fusion mode.
    let query_kind = match &req.query {
        Some(val) => match classify_query(val) {
            Ok(k) => Some(k),
            Err(e) => return json_err(response, 400, &e),
        },
        None => {
            // No query at all — default to fusion if prefetch present, else error.
            if req.prefetch.is_empty() {
                return json_err(response, 400, "Either 'query' or 'prefetch' is required");
            }
            Some(QueryKind::Fusion("rrf".into()))
        }
    };

    // Compute filter selectivity for adaptive ef in dense search paths.
    let total_vectors = indexed_candidates
        .as_ref()
        .map(|_| collection_vector_count(&storage, &txn))
        .unwrap_or(0);
    let selectivity_hint: Option<f32> = if has_filter {
        if let Some(candidates) = &indexed_candidates {
            if total_vectors > 0 {
                Some(candidates.len() as f32 / total_vectors as f32)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // ── Single-vector dense search ──
    if let Some(QueryKind::DenseVector(dense_vec)) = &query_kind {
        if dense_vec.is_empty() {
            return json_err(response, 400, "Query vector must not be empty");
        }
        let using = req.using.as_deref().unwrap_or("dense");
        let score_threshold =
            dense_score_threshold_for_metric(&storage, using, req.score_threshold);
        let search_limit = if has_filter {
            result_window.saturating_mul(4)
        } else {
            result_window
        };

        let ef_override = req.params.as_ref().and_then(|p| p.hnsw_ef);
        let dense_filter_active = has_filter || indexed_candidates.is_some();
        let exact_candidate_ids = indexed_candidates
            .as_ref()
            .filter(|candidates| should_exact_score_index_candidates(candidates, total_vectors))
            .map(|candidates| {
                candidates
                    .iter()
                    .copied()
                    .filter(|id| {
                        if !has_filter {
                            return true;
                        }
                        match storage.get_node(&txn, id) {
                            Ok(node) => filter.matches_point(Some(*id), &node.properties),
                            Err(_) => false,
                        }
                    })
                    .collect::<Vec<u128>>()
            });
        let mut results = if let Some(candidate_ids) = exact_candidate_ids {
            storage.named_vectors.dense_search_candidate_ids_exact(
                &txn,
                using,
                dense_vec,
                search_limit,
                candidate_ids.iter().copied(),
                selectivity_hint,
            )
        } else if dense_filter_active {
            storage.named_vectors.dense_search_with_id_filter_ef(
                &txn,
                using,
                dense_vec,
                search_limit,
                Some(&[|id: u128| {
                    if let Some(candidates) = &indexed_candidates {
                        if !candidates.contains(&id) {
                            return false;
                        }
                    }
                    if !has_filter {
                        return true;
                    }
                    match storage.get_node(&txn, &id) {
                        Ok(node) => filter.matches_point(Some(id), &node.properties),
                        Err(_) => false,
                    }
                }]),
                true,
                selectivity_hint,
                ef_override,
            )
        } else {
            storage
                .named_vectors
                .dense_search_with_selectivity_ef_with_provider::<fn(&HVector) -> bool, _>(
                    &dense_read_txns,
                    &txn,
                    using,
                    dense_vec,
                    search_limit,
                    None,
                    true,
                    selectivity_hint,
                    ef_override,
                )
        }
        .map_err(GraphError::from)?;
        results.truncate(result_window);
        for result in &mut results {
            if let Some(distance) = result.distance {
                result.distance = Some(
                    storage
                        .named_vectors
                        .public_score(using, distance)
                        .map_err(GraphError::from)?,
                );
            }
        }
        fuse_pending_dense_results(
            &name,
            &filter,
            using,
            dense_vec,
            result_window,
            &mut results,
        );
        apply_hvector_score_threshold_and_offset(
            &mut results,
            score_threshold,
            req.offset,
            req.limit,
        );

        return build_query_response(response, &name, &storage, &txn, &results, req.with_payload);
    }

    // ── Single-vector sparse search ──
    if let Some(QueryKind::SparseVector(sparse_vec)) = &query_kind {
        let Some(using) = req.using.as_deref() else {
            return json_err(
                response,
                400,
                "'using' is required for sparse vector queries",
            );
        };

        let filter_fn = |doc_id: u128| -> bool {
            if let Some(candidates) = &indexed_candidates {
                if !candidates.contains(&doc_id) {
                    return false;
                }
            }
            if !has_filter {
                return true;
            }
            match storage.get_node(&txn, &doc_id) {
                Ok(node) => filter.matches_point(Some(doc_id), &node.properties),
                Err(_) => false,
            }
        };

        let search_limit = if has_filter {
            result_window.saturating_mul(4)
        } else {
            result_window
        };

        let mut scored = storage
            .named_vectors
            .with_sparse_core(using, |core| {
                let mut results = core.search(&txn, sparse_vec, search_limit, Some(&filter_fn))?;
                results.truncate(result_window);
                Ok(results)
            })
            .map_err(GraphError::from)?;
        fuse_pending_sparse_results(
            &name,
            &filter,
            using,
            sparse_vec,
            result_window,
            &mut scored,
        );
        apply_sparse_score_threshold_and_offset(
            &mut scored,
            req.score_threshold,
            req.offset,
            req.limit,
        );

        return build_sparse_query_response(
            response,
            &name,
            &storage,
            &txn,
            &scored,
            req.with_payload,
        );
    }

    // ── Prefetch + Fusion mode (existing behavior) ──
    let mut ranked_lists: Vec<Vec<RankedItem>> = Vec::new();

    for pf in &req.prefetch {
        let channel_window = channel_result_window(pf.limit, result_window, req.offset);
        let search_limit = if has_filter {
            channel_window.saturating_mul(4)
        } else {
            channel_window
        };

        let pf_ef = pf.params.as_ref().and_then(|p| p.hnsw_ef);
        let dense_filter_active = has_filter || indexed_candidates.is_some();
        let exact_candidate_ids = indexed_candidates
            .as_ref()
            .filter(|candidates| should_exact_score_index_candidates(candidates, total_vectors))
            .map(|candidates| {
                candidates
                    .iter()
                    .copied()
                    .filter(|id| {
                        if !has_filter {
                            return true;
                        }
                        match storage.get_node(&txn, id) {
                            Ok(node) => filter.matches_point(Some(*id), &node.properties),
                            Err(_) => false,
                        }
                    })
                    .collect::<Vec<u128>>()
            });
        let mut results = if let Some(candidate_ids) = exact_candidate_ids {
            storage.named_vectors.dense_search_candidate_ids_exact(
                &txn,
                &pf.using,
                &pf.query,
                search_limit,
                candidate_ids.iter().copied(),
                selectivity_hint,
            )
        } else if dense_filter_active {
            storage.named_vectors.dense_search_with_id_filter_ef(
                &txn,
                &pf.using,
                &pf.query,
                search_limit,
                Some(&[|id: u128| {
                    if let Some(candidates) = &indexed_candidates {
                        if !candidates.contains(&id) {
                            return false;
                        }
                    }
                    if !has_filter {
                        return true;
                    }
                    match storage.get_node(&txn, &id) {
                        Ok(node) => filter.matches_point(Some(id), &node.properties),
                        Err(_) => false,
                    }
                }]),
                true,
                selectivity_hint,
                pf_ef,
            )
        } else {
            storage
                .named_vectors
                .dense_search_with_selectivity_ef_with_provider::<fn(&HVector) -> bool, _>(
                    &dense_read_txns,
                    &txn,
                    &pf.using,
                    &pf.query,
                    search_limit,
                    None,
                    true,
                    selectivity_hint,
                    pf_ef,
                )
        }
        .map_err(GraphError::from)?;
        results.truncate(channel_window);
        for result in &mut results {
            if let Some(distance) = result.distance {
                result.distance = Some(
                    storage
                        .named_vectors
                        .public_score(&pf.using, distance)
                        .map_err(GraphError::from)?,
                );
            }
        }
        fuse_pending_dense_results(
            &name,
            &filter,
            &pf.using,
            &pf.query,
            channel_window,
            &mut results,
        );

        let ranked: Vec<RankedItem> = results
            .into_iter()
            .map(|v| RankedItem {
                id: v.id,
                score: v.distance.unwrap_or(0.0) as f64,
            })
            .collect();
        ranked_lists.push(ranked);
    }

    // Apply fusion
    let mut fused = if ranked_lists.len() > 1 {
        rrf_fusion(&ranked_lists, result_window)
    } else if ranked_lists.len() == 1 {
        ranked_lists
            .into_iter()
            .next()
            .unwrap_or_default()
            .into_iter()
            .take(result_window)
            .collect()
    } else {
        Vec::new()
    };
    apply_ranked_score_threshold_and_offset(&mut fused, req.score_threshold, req.offset, req.limit);

    // Build response
    let mut results: Vec<serde_json::Value> = Vec::new();
    for item in &fused {
        let mut point = serde_json::json!({
            "id": format_point_id(item.id),
            "version": 0,
            "score": item.score,
        });

        if req.with_payload {
            if let Some(pending) = pending_point(&name, item.id) {
                point["payload"] = pending_payload_json(&pending);
            } else if let Ok(node) = storage.get_node(&txn, &item.id) {
                point["payload"] = value_map_to_json(&node.properties);
            }
        }

        results.push(point);
    }

    json_ok(response, &serde_json::json!({ "points": results }))
}

/// POST /collections/{name}/points/hybrid_query
///
/// Server-side multi-channel retrieval for CE fast paths. The request carries
/// client-computed dense and sparse query vectors; Helix executes each channel
/// under one read transaction and fuses the ranked lists with RRF.
pub fn handle_hybrid_query_points(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    const ENDPOINT: &str = "/collections/{collection}/points/hybrid_query";
    let request_start = Instant::now();
    let name = match get_collection_name(input) {
        Some(n) => n,
        None => return json_err(response, 400, "Missing collection name"),
    };

    let mut req: HybridQueryRequest = match sonic_rs::from_slice(&input.request.body) {
        Ok(r) => r,
        Err(e) => return json_err(response, 400, &format!("Invalid JSON: {}", e)),
    };
    req.limit = clamp_limit(req.limit);
    for dense in req.dense.iter_mut() {
        dense.limit = clamp_limit(dense.limit);
    }
    for sparse in req.sparse.iter_mut() {
        sparse.limit = clamp_limit(sparse.limit);
    }
    if req.dense.is_empty() && req.sparse.is_empty() {
        return json_err(
            response,
            400,
            "At least one dense or sparse query is required",
        );
    }
    if let Some(rrf_k) = req.rrf_k {
        if !rrf_k.is_finite() || rrf_k <= 0.0 {
            return json_err(response, 400, "rrf_k must be a finite positive number");
        }
    }
    let mmr_config = req.mmr.as_ref().and_then(HybridMmrRequest::config);
    if let Some(config) = &mmr_config {
        if !config.lambda.is_finite() || !(0.0..=1.0).contains(&config.lambda) {
            return json_err(response, 400, "mmr.lambda must be between 0.0 and 1.0");
        }
    }
    // Ops kill-switch: when off, the graph field is ignored entirely (no
    // validation, no summary) so requests behave as if the field were absent.
    let graph_config = if spindle_bool_from_env("HELIX_HYBRID_GRAPH_CHANNEL_ENABLED", true) {
        req.graph.as_ref().and_then(HybridGraphRequest::config)
    } else {
        None
    };
    let graph_resolved = match &graph_config {
        Some(config) => match resolve_hybrid_graph_config(config) {
            Ok(resolved) => Some(resolved),
            Err(message) => return json_err(response, 400, &message),
        },
        None => None,
    };

    let collection_start = Instant::now();
    let storage = match input.collections.get_collection(&name) {
        Ok(s) => {
            if s.backend.kind() == BackendKind::Lsm {
                record_qdrant_query_stage(
                    &name,
                    ENDPOINT,
                    "lsm",
                    "hybrid",
                    "collection_get",
                    collection_start,
                );
            }
            s
        }
        Err(e) => return json_err(response, 404, &e.to_string()),
    };

    if storage.backend.kind() == BackendKind::Lsm {
        let backend = backend_label(storage.backend.kind());
        let query_kind_label = "hybrid";
        record_qdrant_query_count(
            &name,
            ENDPOINT,
            backend,
            query_kind_label,
            "limit",
            req.limit,
        );
        record_qdrant_query_count(
            &name,
            ENDPOINT,
            backend,
            query_kind_label,
            "dense_channels",
            req.dense.len(),
        );
        record_qdrant_query_count(
            &name,
            ENDPOINT,
            backend,
            query_kind_label,
            "sparse_channels",
            req.sparse.len(),
        );
        let read_start = Instant::now();
        let r = storage
            .backend
            .begin_read()
            .map_err(|e| GraphError::New(e.to_string()))?;
        record_qdrant_query_stage(
            &name,
            ENDPOINT,
            backend,
            query_kind_label,
            "begin_read",
            read_start,
        );
        let filter = req.filter.unwrap_or_default();
        let filter_start = Instant::now();
        let ProbedCandidates {
            candidates: indexed_candidates,
            probe_estimate,
            probe_skipped: _,
        } = vector_query_indexed_filter_candidates_be_probed(
            &storage,
            &r,
            &filter,
            filter_count_probe_enabled(),
        )?;
        record_qdrant_query_stage(
            &name,
            ENDPOINT,
            backend,
            query_kind_label,
            "filter_candidates",
            filter_start,
        );
        if let Some(candidates) = &indexed_candidates {
            record_qdrant_query_count(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "filter_candidates",
                candidates.len(),
            );
        }
        if let Some(estimate) = probe_estimate {
            record_qdrant_query_count(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "filter_candidate_estimate",
                estimate,
            );
        }
        let has_filter = !filter.is_empty();
        let total_vectors = if has_filter && indexed_candidates.is_some() {
            collection_vector_count_be(&storage)
        } else {
            0
        };
        if total_vectors > 0 {
            record_qdrant_query_count(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "total_vectors",
                total_vectors as usize,
            );
        }
        let selectivity_hint = if has_filter {
            indexed_candidates.as_ref().and_then(|candidates| {
                (total_vectors > 0).then_some(candidates.len() as f32 / total_vectors as f32)
            })
        } else {
            None
        };

        let collect_mmr_vectors = mmr_config.is_some();
        let mut ranked_lists: Vec<Vec<RankedItem>> = Vec::new();
        let mut dense_channels: Vec<serde_json::Value> = Vec::new();
        let mut sparse_channels: Vec<serde_json::Value> = Vec::new();
        let mut dense_vectors: HashMap<u128, Vec<f32>> = HashMap::new();
        let mut request_payload_cache = RequestPayloadCache::default();

        for dense in &req.dense {
            if dense.query.is_empty() {
                record_qdrant_query_outcome(
                    &name,
                    ENDPOINT,
                    backend,
                    query_kind_label,
                    "bad_request",
                    request_start,
                );
                return json_err(response, 400, "Dense query vector must not be empty");
            }
            let eff_filter: &Filter = dense.filter.as_ref().unwrap_or(&filter);
            let eff_has_filter = !eff_filter.is_empty();
            let channel_filter_start = Instant::now();
            let eff_indexed_candidates = if dense.filter.is_some() {
                vector_query_indexed_filter_candidates_be_probed(
                    &storage,
                    &r,
                    eff_filter,
                    filter_count_probe_enabled(),
                )?
                .candidates
            } else {
                indexed_candidates.clone()
            };
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "hybrid_dense_filter_candidates",
                channel_filter_start,
            );
            if let Some(candidates) = &eff_indexed_candidates {
                record_qdrant_query_count(
                    &name,
                    ENDPOINT,
                    backend,
                    query_kind_label,
                    "hybrid_dense_filter_candidates",
                    candidates.len(),
                );
            }
            let eff_filter_index_exact = eff_indexed_candidates.is_some()
                && indexed_filter_candidates_are_exact(&storage, eff_filter);
            let eff_filter_recheck_required = eff_has_filter && !eff_filter_index_exact;
            let eff_total_vectors = eff_indexed_candidates
                .as_ref()
                .map(|_| collection_vector_count_be(&storage))
                .unwrap_or(0);
            let eff_selectivity_hint = if dense.filter.is_some() && eff_has_filter {
                eff_indexed_candidates.as_ref().and_then(|candidates| {
                    (eff_total_vectors > 0)
                        .then_some(candidates.len() as f32 / eff_total_vectors as f32)
                })
            } else {
                selectivity_hint
            };
            let filter_fn = |id: u128| -> bool {
                if let Some(candidates) = &eff_indexed_candidates {
                    if !candidates.contains(&id) {
                        return false;
                    }
                }
                if !eff_filter_recheck_required {
                    return true;
                }
                match storage.get_node_be(&r, &id) {
                    Ok(node) => eff_filter.matches_point(Some(id), &node.properties),
                    Err(_) => false,
                }
            };
            let search_limit = if eff_has_filter || eff_indexed_candidates.is_some() {
                dense.limit.saturating_mul(4)
            } else {
                dense.limit
            };
            let ef_override = dense.params.as_ref().and_then(|p| p.hnsw_ef);
            let dense_filter_active = eff_has_filter || eff_indexed_candidates.is_some();
            let exact_index_filter_candidates = exact_index_filter_candidates(
                eff_has_filter,
                eff_filter_recheck_required,
                eff_indexed_candidates.as_ref(),
            );
            let candidate_start = Instant::now();
            let exact_candidate_ids = exact_dense_candidate_ids_be(
                &storage,
                &r,
                eff_indexed_candidates.as_ref(),
                &eff_filter,
                eff_filter_recheck_required,
                eff_total_vectors,
            );
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "hybrid_dense_candidate_plan",
                candidate_start,
            );
            if let Some(candidate_ids) = &exact_candidate_ids {
                record_qdrant_query_count(
                    &name,
                    ENDPOINT,
                    backend,
                    query_kind_label,
                    "exact_candidate_ids",
                    candidate_ids.len(),
                );
            }
            let search_start = Instant::now();
            let mut results = if let Some(candidate_ids) = exact_candidate_ids {
                storage.named_vectors.dense_search_candidate_ids_exact_be(
                    &r,
                    &dense.using,
                    &dense.query,
                    search_limit,
                    candidate_ids.iter().copied(),
                    eff_selectivity_hint,
                )
            } else if let Some(candidate_ids) = exact_index_filter_candidates {
                storage
                    .named_vectors
                    .dense_search_candidate_set_filter_ef_be(
                        &r,
                        &dense.using,
                        &dense.query,
                        search_limit,
                        candidate_ids,
                        true,
                        eff_selectivity_hint,
                        ef_override,
                    )
            } else {
                storage.named_vectors.dense_search_with_id_filter_ef_be(
                    &r,
                    &dense.using,
                    &dense.query,
                    search_limit,
                    dense_filter_active.then_some(&[filter_fn][..]),
                    true,
                    eff_selectivity_hint,
                    ef_override,
                )
            }
            .map_err(GraphError::from)?;
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "hybrid_dense_search",
                search_start,
            );
            record_qdrant_query_count(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "search_results_raw",
                results.len(),
            );
            results.truncate(dense.limit);
            let score_start = Instant::now();
            for result in &mut results {
                if let Some(distance) = result.distance {
                    result.distance = Some(
                        storage
                            .named_vectors
                            .public_score(&dense.using, distance)
                            .map_err(GraphError::from)?,
                    );
                }
            }
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "hybrid_dense_score_normalize",
                score_start,
            );
            let fuse_start = Instant::now();
            fuse_pending_dense_results(
                &name,
                eff_filter,
                &dense.using,
                &dense.query,
                dense.limit,
                &mut results,
            );
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "hybrid_dense_pending_fuse",
                fuse_start,
            );
            record_qdrant_query_count(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "hybrid_dense_results",
                results.len(),
            );
            if collect_mmr_vectors {
                for vector in &results {
                    dense_vectors
                        .entry(vector.id)
                        .or_insert_with(|| vector.get_data().to_vec());
                }
            }
            if req.with_channel_points {
                let response_start = Instant::now();
                let payload_cache = if req.with_payload {
                    request_payload_cache
                        .get_or_load(&name, &storage, &r, results.iter().map(|v| v.id))
                        .map_err(graph_error_from_backend_error)?
                } else {
                    None
                };
                let points: Vec<serde_json::Value> = results
                    .iter()
                    .map(|v| {
                        query_point_json_be(
                            &name,
                            &storage,
                            &r,
                            v.id,
                            v.distance.unwrap_or(0.0) as f64,
                            req.with_payload,
                            payload_cache,
                        )
                    })
                    .collect();
                record_qdrant_query_stage(
                    &name,
                    ENDPOINT,
                    backend,
                    query_kind_label,
                    "hybrid_dense_channel_response",
                    response_start,
                );
                dense_channels.push(serde_json::json!({
                    "using": dense.using,
                    "points": points,
                }));
            }
            ranked_lists.push(
                results
                    .into_iter()
                    .map(|v| RankedItem {
                        id: v.id,
                        score: v.distance.unwrap_or(0.0) as f64,
                    })
                    .collect(),
            );
        }

        for sparse in &req.sparse {
            if sparse.query.indices.is_empty() || sparse.query.values.is_empty() {
                record_qdrant_query_outcome(
                    &name,
                    ENDPOINT,
                    backend,
                    query_kind_label,
                    "bad_request",
                    request_start,
                );
                return json_err(response, 400, "Sparse query vector must not be empty");
            }
            if sparse.query.indices.len() != sparse.query.values.len() {
                record_qdrant_query_outcome(
                    &name,
                    ENDPOINT,
                    backend,
                    query_kind_label,
                    "bad_request",
                    request_start,
                );
                return json_err(
                    response,
                    400,
                    "Sparse query vector indices and values must have the same length",
                );
            }
            let eff_filter: &Filter = sparse.filter.as_ref().unwrap_or(&filter);
            let eff_has_filter = !eff_filter.is_empty();
            let channel_filter_start = Instant::now();
            let eff_indexed_candidates = if sparse.filter.is_some() {
                vector_query_indexed_filter_candidates_be_probed(
                    &storage,
                    &r,
                    eff_filter,
                    filter_count_probe_enabled(),
                )?
                .candidates
            } else {
                indexed_candidates.clone()
            };
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "hybrid_sparse_filter_candidates",
                channel_filter_start,
            );
            if let Some(candidates) = &eff_indexed_candidates {
                record_qdrant_query_count(
                    &name,
                    ENDPOINT,
                    backend,
                    query_kind_label,
                    "hybrid_sparse_filter_candidates",
                    candidates.len(),
                );
            }
            let eff_filter_index_exact = eff_indexed_candidates.is_some()
                && indexed_filter_candidates_are_exact(&storage, eff_filter);
            let eff_filter_recheck_required = eff_has_filter && !eff_filter_index_exact;
            let filter_fn = |doc_id: u128| -> bool {
                if let Some(candidates) = &eff_indexed_candidates {
                    if !candidates.contains(&doc_id) {
                        return false;
                    }
                }
                if !eff_filter_recheck_required {
                    return true;
                }
                match storage.get_node_be(&r, &doc_id) {
                    Ok(node) => eff_filter.matches_point(Some(doc_id), &node.properties),
                    Err(_) => false,
                }
            };
            let search_limit = if eff_has_filter || eff_indexed_candidates.is_some() {
                sparse.limit.saturating_mul(4)
            } else {
                sparse.limit
            };
            let query = SparseVector {
                indices: sparse.query.indices.clone(),
                values: sparse.query.values.clone(),
            };
            let eff_total_vectors = eff_indexed_candidates
                .as_ref()
                .map(|_| collection_vector_count_be(&storage))
                .unwrap_or(0);
            let candidate_start = Instant::now();
            let exact_candidate_ids = bounded_filter_candidate_ids_be(
                &storage,
                &r,
                eff_indexed_candidates.as_ref(),
                eff_filter,
                eff_filter_recheck_required,
                eff_total_vectors,
            );
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "hybrid_sparse_candidate_plan",
                candidate_start,
            );
            if let Some(candidate_ids) = &exact_candidate_ids {
                record_qdrant_query_count(
                    &name,
                    ENDPOINT,
                    backend,
                    query_kind_label,
                    "exact_candidate_ids",
                    candidate_ids.len(),
                );
            }
            let search_start = Instant::now();
            let mut results = storage
                .named_vectors
                .with_sparse_core(&sparse.using, |core| {
                    if let Some(candidate_ids) = exact_candidate_ids.as_ref() {
                        core.search_candidate_ids_exact_be(
                            &r,
                            &query,
                            search_limit,
                            candidate_ids.iter().copied(),
                        )
                    } else {
                        core.search_be(&r, &query, search_limit, Some(&filter_fn))
                    }
                })
                .map_err(GraphError::from)?;
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "hybrid_sparse_search",
                search_start,
            );
            record_qdrant_query_count(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "search_results_raw",
                results.len(),
            );
            results.truncate(sparse.limit);
            let fuse_start = Instant::now();
            fuse_pending_sparse_results(
                &name,
                eff_filter,
                &sparse.using,
                &query,
                sparse.limit,
                &mut results,
            );
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "hybrid_sparse_pending_fuse",
                fuse_start,
            );
            record_qdrant_query_count(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "hybrid_sparse_results",
                results.len(),
            );
            if req.with_channel_points {
                let response_start = Instant::now();
                let payload_cache = if req.with_payload {
                    request_payload_cache
                        .get_or_load(&name, &storage, &r, results.iter().map(|(id, _)| *id))
                        .map_err(graph_error_from_backend_error)?
                } else {
                    None
                };
                let points: Vec<serde_json::Value> = results
                    .iter()
                    .map(|(id, score)| {
                        query_point_json_be(
                            &name,
                            &storage,
                            &r,
                            *id,
                            *score,
                            req.with_payload,
                            payload_cache,
                        )
                    })
                    .collect();
                record_qdrant_query_stage(
                    &name,
                    ENDPOINT,
                    backend,
                    query_kind_label,
                    "hybrid_sparse_channel_response",
                    response_start,
                );
                sparse_channels.push(serde_json::json!({
                    "using": sparse.using,
                    "points": points,
                }));
            }
            ranked_lists.push(
                results
                    .into_iter()
                    .map(|(id, score)| RankedItem { id, score })
                    .collect(),
            );
        }

        let mut graph_summary: Option<serde_json::Value> = None;
        let mut graph_channels: Vec<serde_json::Value> = Vec::new();
        if let Some(resolved) = &graph_resolved {
            let graph_start = Instant::now();
            let label_hashes: Vec<[u8; 4]> = resolved
                .edge_labels
                .iter()
                .map(|label| hash_label(label, None))
                .collect();
            let neighbors = |node: u128| -> Vec<u128> {
                let mut out_peers: Vec<u128> = Vec::new();
                let mut in_peers: Vec<u128> = Vec::new();
                for label_hash in &label_hashes {
                    if resolved.direction != PprDirection::In {
                        match storage.adjacency_pairs_be(&r, node, label_hash, true) {
                            Ok(pairs) => out_peers.extend(pairs.into_iter().map(|(peer, _)| peer)),
                            Err(e) => tracing::warn!(
                                collection = %name,
                                error = %e,
                                "hybrid graph channel out-adjacency read failed"
                            ),
                        }
                    }
                    if resolved.direction != PprDirection::Out {
                        match storage.adjacency_pairs_be(&r, node, label_hash, false) {
                            Ok(pairs) => in_peers.extend(pairs.into_iter().map(|(peer, _)| peer)),
                            Err(e) => tracing::warn!(
                                collection = %name,
                                error = %e,
                                "hybrid graph channel in-adjacency read failed"
                            ),
                        }
                    }
                }
                merge_directions(resolved.direction, out_peers, in_peers)
            };
            // Same top-level filter the fused result must satisfy; nodes that
            // no longer exist (deleted points behind stale edges) are dropped.
            let admit = |id: u128| -> bool {
                if let Some(pending) = pending_point(&name, id) {
                    return pending_matches_filter(id, &pending, &filter);
                }
                match storage.get_node_be(&r, &id) {
                    Ok(node) => {
                        filter.is_empty() || filter.matches_point(Some(id), &node.properties)
                    }
                    Err(_) => false,
                }
            };
            let (graph_list, summary) =
                run_hybrid_graph_channel(&ranked_lists, req.rrf_k, resolved, neighbors, admit);
            record_qdrant_query_stage(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "hybrid_graph_channel",
                graph_start,
            );
            record_qdrant_query_count(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "hybrid_graph_results",
                graph_list.len(),
            );
            graph_summary = Some(summary);
            if req.with_channel_points {
                let payload_cache = if req.with_payload {
                    request_payload_cache
                        .get_or_load(&name, &storage, &r, graph_list.iter().map(|v| v.id))
                        .map_err(graph_error_from_backend_error)?
                } else {
                    None
                };
                let points: Vec<serde_json::Value> = graph_list
                    .iter()
                    .map(|item| {
                        query_point_json_be(
                            &name,
                            &storage,
                            &r,
                            item.id,
                            item.score,
                            req.with_payload,
                            payload_cache,
                        )
                    })
                    .collect();
                graph_channels.push(serde_json::json!({ "points": points }));
            }
            if !graph_list.is_empty() {
                ranked_lists.push(graph_list);
            }
        }

        let fusion_start = Instant::now();
        let mut fused = match ranked_lists.len() {
            0 => Vec::new(),
            1 => ranked_lists
                .into_iter()
                .next()
                .unwrap_or_default()
                .into_iter()
                .take(req.limit)
                .collect(),
            _ => match req.rrf_k {
                Some(k) => rrf_fusion_with_k(&ranked_lists, k, req.limit),
                None => rrf_fusion(&ranked_lists, req.limit),
            },
        };
        record_qdrant_query_stage(
            &name,
            ENDPOINT,
            backend,
            query_kind_label,
            "fusion",
            fusion_start,
        );
        record_qdrant_query_count(
            &name,
            ENDPOINT,
            backend,
            query_kind_label,
            "fusion_candidates",
            fused.len(),
        );
        let mut mmr_summary: Option<serde_json::Value> = None;
        if let Some(config) = &mmr_config {
            let mmr_start = Instant::now();
            let (reranked, candidate_vectors) =
                mmr_rerank_fused(&fused, &dense_vectors, req.limit, config.lambda);
            let applied = candidate_vectors >= 2;
            fused = reranked;
            record_qdrant_query_stage(&name, ENDPOINT, backend, query_kind_label, "mmr", mmr_start);
            record_qdrant_query_count(
                &name,
                ENDPOINT,
                backend,
                query_kind_label,
                "mmr_candidate_vectors",
                candidate_vectors,
            );
            mmr_summary = Some(serde_json::json!({
                "enabled": true,
                "applied": applied,
                "lambda": config.lambda,
                "candidate_vectors": candidate_vectors,
            }));
        } else if req.mmr.is_some() {
            mmr_summary = Some(serde_json::json!({
                "enabled": false,
                "applied": false,
            }));
        }

        record_qdrant_query_count(
            &name,
            ENDPOINT,
            backend,
            query_kind_label,
            "results_returned",
            fused.len(),
        );
        let response_start = Instant::now();
        let payload_cache = if req.with_payload {
            request_payload_cache
                .get_or_load(&name, &storage, &r, fused.iter().map(|item| item.id))
                .map_err(graph_error_from_backend_error)?
        } else {
            None
        };
        let points: Vec<serde_json::Value> = fused
            .iter()
            .map(|item| {
                query_point_json_be(
                    &name,
                    &storage,
                    &r,
                    item.id,
                    item.score,
                    req.with_payload,
                    payload_cache,
                )
            })
            .collect();
        let mut fusion = serde_json::json!({
            "dense": req.dense.len(),
            "sparse": req.sparse.len(),
            "rrf_k": req.rrf_k.unwrap_or(60.0),
        });
        if let Some(summary) = mmr_summary {
            fusion["mmr"] = summary;
        }
        if let Some(summary) = graph_summary {
            fusion["graph"] = summary;
        }
        let mut result = serde_json::json!({
            "points": points,
            "fusion": fusion
        });
        if req.with_channel_points {
            let mut channels = serde_json::json!({
                "dense": dense_channels,
                "sparse": sparse_channels,
            });
            if !graph_channels.is_empty() {
                channels["graph"] = serde_json::json!(graph_channels);
            }
            result["channels"] = channels;
        }
        let result = json_ok(response, &result);
        record_qdrant_query_stage(
            &name,
            ENDPOINT,
            backend,
            query_kind_label,
            "response_build",
            response_start,
        );
        record_qdrant_query_outcome(
            &name,
            ENDPOINT,
            backend,
            query_kind_label,
            if result.is_ok() { "ok" } else { "error" },
            request_start,
        );
        return result;
    }

    let txn = storage.begin_resize_safe_read_txn()?;
    let dense_read_txns = storage.nested_dense_read_provider(&txn);
    let filter = req.filter.unwrap_or_default();
    let indexed_candidates = vector_query_indexed_filter_candidates(&storage, &txn, &filter)?;
    let has_filter = !filter.is_empty();
    let selectivity_hint: Option<f32> = if has_filter {
        if let Some(candidates) = &indexed_candidates {
            let total = storage
                .get_metadata(&txn)
                .map(|m| m.stats.vector_count)
                .unwrap_or(0);
            if total > 0 {
                Some(candidates.len() as f32 / total as f32)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    let mut ranked_lists: Vec<Vec<RankedItem>> =
        Vec::with_capacity(req.dense.len() + req.sparse.len());
    let collect_mmr_vectors = mmr_config.is_some();
    let mut dense_vectors: HashMap<u128, Vec<f32>> = if collect_mmr_vectors {
        HashMap::with_capacity(req.limit.saturating_mul(req.dense.len().max(1)))
    } else {
        HashMap::new()
    };
    let mut dense_channels: Vec<serde_json::Value> = Vec::new();
    let mut sparse_channels: Vec<serde_json::Value> = Vec::new();

    for dense in &req.dense {
        if dense.query.is_empty() {
            return json_err(response, 400, "Dense query vector must not be empty");
        }

        // Per-entry filter REPLACES the top-level filter for this channel
        // (Qdrant prefetch semantics). Fall back to top-level when absent.
        let eff_filter: &Filter = match &dense.filter {
            Some(f) => f,
            None => &filter,
        };
        let eff_has_filter = !eff_filter.is_empty();
        let eff_indexed_candidates = if dense.filter.is_some() {
            vector_query_indexed_filter_candidates(&storage, &txn, eff_filter)?
        } else {
            indexed_candidates.clone()
        };
        let eff_total_vectors = eff_indexed_candidates
            .as_ref()
            .map(|_| collection_vector_count(&storage, &txn))
            .unwrap_or(0);
        let eff_selectivity_hint: Option<f32> = if dense.filter.is_some() {
            if eff_has_filter {
                if let Some(candidates) = &eff_indexed_candidates {
                    if eff_total_vectors > 0 {
                        Some(candidates.len() as f32 / eff_total_vectors as f32)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            selectivity_hint
        };

        let search_limit = if eff_has_filter {
            dense.limit.saturating_mul(4)
        } else {
            dense.limit
        };
        let ef_override = dense.params.as_ref().and_then(|p| p.hnsw_ef);
        let dense_filter_active = eff_has_filter || eff_indexed_candidates.is_some();
        let exact_candidate_ids = eff_indexed_candidates
            .as_ref()
            .filter(|candidates| should_exact_score_index_candidates(candidates, eff_total_vectors))
            .map(|candidates| {
                candidates
                    .iter()
                    .copied()
                    .filter(|id| {
                        if !eff_has_filter {
                            return true;
                        }
                        match storage.get_node(&txn, id) {
                            Ok(node) => eff_filter.matches_point(Some(*id), &node.properties),
                            Err(_) => false,
                        }
                    })
                    .collect::<Vec<u128>>()
            });
        let mut results = if let Some(candidate_ids) = exact_candidate_ids {
            storage.named_vectors.dense_search_candidate_ids_exact(
                &txn,
                &dense.using,
                &dense.query,
                search_limit,
                candidate_ids.iter().copied(),
                eff_selectivity_hint,
            )
        } else if dense_filter_active {
            storage.named_vectors.dense_search_with_id_filter_ef(
                &txn,
                &dense.using,
                &dense.query,
                search_limit,
                Some(&[|id: u128| {
                    if let Some(candidates) = &eff_indexed_candidates {
                        if !candidates.contains(&id) {
                            return false;
                        }
                    }
                    if !eff_has_filter {
                        return true;
                    }
                    match storage.get_node(&txn, &id) {
                        Ok(node) => eff_filter.matches_point(Some(id), &node.properties),
                        Err(_) => false,
                    }
                }]),
                true,
                eff_selectivity_hint,
                ef_override,
            )
        } else {
            storage
                .named_vectors
                .dense_search_with_selectivity_ef_with_provider::<fn(&HVector) -> bool, _>(
                    &dense_read_txns,
                    &txn,
                    &dense.using,
                    &dense.query,
                    search_limit,
                    None,
                    true,
                    eff_selectivity_hint,
                    ef_override,
                )
        }
        .map_err(GraphError::from)?;
        results.truncate(dense.limit);
        for result in &mut results {
            if let Some(distance) = result.distance {
                result.distance = Some(
                    storage
                        .named_vectors
                        .public_score(&dense.using, distance)
                        .map_err(GraphError::from)?,
                );
            }
        }
        fuse_pending_dense_results(
            &name,
            eff_filter,
            &dense.using,
            &dense.query,
            dense.limit,
            &mut results,
        );
        if collect_mmr_vectors {
            for vector in &results {
                dense_vectors
                    .entry(vector.id)
                    .or_insert_with(|| vector.get_data().to_vec());
            }
        }
        if req.with_channel_points {
            let points: Vec<serde_json::Value> = results
                .iter()
                .map(|v| {
                    query_point_json(
                        &name,
                        &storage,
                        &txn,
                        v.id,
                        v.distance.unwrap_or(0.0) as f64,
                        req.with_payload,
                    )
                })
                .collect();
            dense_channels.push(serde_json::json!({
                "using": dense.using,
                "points": points,
            }));
        }
        ranked_lists.push(
            results
                .into_iter()
                .map(|v| RankedItem {
                    id: v.id,
                    score: v.distance.unwrap_or(0.0) as f64,
                })
                .collect(),
        );
    }

    for sparse in &req.sparse {
        if sparse.query.indices.is_empty() || sparse.query.values.is_empty() {
            return json_err(response, 400, "Sparse query vector must not be empty");
        }
        if sparse.query.indices.len() != sparse.query.values.len() {
            return json_err(
                response,
                400,
                "Sparse query vector indices and values must have the same length",
            );
        }

        // Per-entry filter REPLACES the top-level filter for this channel
        // (Qdrant prefetch semantics). Fall back to top-level when absent.
        let eff_filter: &Filter = match &sparse.filter {
            Some(f) => f,
            None => &filter,
        };
        let eff_has_filter = !eff_filter.is_empty();
        let eff_indexed_candidates = if sparse.filter.is_some() {
            vector_query_indexed_filter_candidates(&storage, &txn, eff_filter)?
        } else {
            indexed_candidates.clone()
        };

        let filter_fn = |doc_id: u128| -> bool {
            if let Some(candidates) = &eff_indexed_candidates {
                if !candidates.contains(&doc_id) {
                    return false;
                }
            }
            if !eff_has_filter {
                return true;
            }
            match storage.get_node(&txn, &doc_id) {
                Ok(node) => eff_filter.matches_point(Some(doc_id), &node.properties),
                Err(_) => false,
            }
        };
        let search_limit = if eff_has_filter {
            sparse.limit.saturating_mul(4)
        } else {
            sparse.limit
        };
        let query = SparseVector {
            indices: sparse.query.indices.clone(),
            values: sparse.query.values.clone(),
        };
        let mut results = storage
            .named_vectors
            .with_sparse_core(&sparse.using, |core| {
                core.search(&txn, &query, search_limit, Some(&filter_fn))
            })
            .map_err(GraphError::from)?;
        results.truncate(sparse.limit);
        fuse_pending_sparse_results(
            &name,
            eff_filter,
            &sparse.using,
            &query,
            sparse.limit,
            &mut results,
        );
        if req.with_channel_points {
            let points: Vec<serde_json::Value> = results
                .iter()
                .map(|(id, score)| {
                    query_point_json(&name, &storage, &txn, *id, *score, req.with_payload)
                })
                .collect();
            sparse_channels.push(serde_json::json!({
                "using": sparse.using,
                "points": points,
            }));
        }
        ranked_lists.push(
            results
                .into_iter()
                .map(|(id, score)| RankedItem { id, score })
                .collect(),
        );
    }

    let mut graph_summary: Option<serde_json::Value> = None;
    let mut graph_channels: Vec<serde_json::Value> = Vec::new();
    if let Some(resolved) = &graph_resolved {
        let label_hashes: Vec<[u8; 4]> = resolved
            .edge_labels
            .iter()
            .map(|label| hash_label(label, None))
            .collect();
        let neighbors = |node: u128| -> Vec<u128> {
            let mut out_peers: Vec<u128> = Vec::new();
            let mut in_peers: Vec<u128> = Vec::new();
            for label_hash in &label_hashes {
                if resolved.direction != PprDirection::In {
                    match storage.adjacency_pairs(&txn, node, label_hash, true) {
                        Ok(pairs) => out_peers.extend(pairs.into_iter().map(|(peer, _)| peer)),
                        Err(e) => tracing::warn!(
                            collection = %name,
                            error = %e,
                            "hybrid graph channel out-adjacency read failed"
                        ),
                    }
                }
                if resolved.direction != PprDirection::Out {
                    match storage.adjacency_pairs(&txn, node, label_hash, false) {
                        Ok(pairs) => in_peers.extend(pairs.into_iter().map(|(peer, _)| peer)),
                        Err(e) => tracing::warn!(
                            collection = %name,
                            error = %e,
                            "hybrid graph channel in-adjacency read failed"
                        ),
                    }
                }
            }
            merge_directions(resolved.direction, out_peers, in_peers)
        };
        // Same top-level filter the fused result must satisfy; nodes that no
        // longer exist (deleted points behind stale edges) are dropped.
        let admit = |id: u128| -> bool {
            if let Some(pending) = pending_point(&name, id) {
                return pending_matches_filter(id, &pending, &filter);
            }
            match storage.get_node(&txn, &id) {
                Ok(node) => filter.is_empty() || filter.matches_point(Some(id), &node.properties),
                Err(_) => false,
            }
        };
        let (graph_list, summary) =
            run_hybrid_graph_channel(&ranked_lists, req.rrf_k, resolved, neighbors, admit);
        graph_summary = Some(summary);
        if req.with_channel_points {
            let points: Vec<serde_json::Value> = graph_list
                .iter()
                .map(|item| {
                    query_point_json(&name, &storage, &txn, item.id, item.score, req.with_payload)
                })
                .collect();
            graph_channels.push(serde_json::json!({ "points": points }));
        }
        if !graph_list.is_empty() {
            ranked_lists.push(graph_list);
        }
    }

    let mut fused = match ranked_lists.len() {
        0 => Vec::new(),
        1 => ranked_lists
            .into_iter()
            .next()
            .unwrap_or_default()
            .into_iter()
            .take(req.limit)
            .collect(),
        _ => match req.rrf_k {
            Some(k) => rrf_fusion_with_k(&ranked_lists, k, req.limit),
            None => rrf_fusion(&ranked_lists, req.limit),
        },
    };
    let mut mmr_summary: Option<serde_json::Value> = None;
    if let Some(config) = &mmr_config {
        let (reranked, candidate_vectors) =
            mmr_rerank_fused(&fused, &dense_vectors, req.limit, config.lambda);
        let applied = candidate_vectors >= 2;
        fused = reranked;
        mmr_summary = Some(serde_json::json!({
            "enabled": true,
            "applied": applied,
            "lambda": config.lambda,
            "candidate_vectors": candidate_vectors,
        }));
    } else if req.mmr.is_some() {
        mmr_summary = Some(serde_json::json!({
            "enabled": false,
            "applied": false,
        }));
    }

    let mut points: Vec<serde_json::Value> = Vec::with_capacity(fused.len());
    for item in &fused {
        let mut point = serde_json::json!({
            "id": format_point_id(item.id),
            "version": 0,
            "score": item.score,
        });
        if req.with_payload {
            if let Some(pending) = pending_point(&name, item.id) {
                point["payload"] = pending_payload_json(&pending);
            } else if let Ok(node) = storage.get_node(&txn, &item.id) {
                point["payload"] = value_map_to_json(&node.properties);
            }
        }
        points.push(point);
    }

    let mut fusion = serde_json::json!({
        "dense": req.dense.len(),
        "sparse": req.sparse.len(),
        "rrf_k": req.rrf_k.unwrap_or(60.0),
    });
    if let Some(summary) = mmr_summary {
        fusion["mmr"] = summary;
    }
    if let Some(summary) = graph_summary {
        fusion["graph"] = summary;
    }

    let mut result = serde_json::json!({
        "points": points,
        "fusion": fusion
    });
    if req.with_channel_points {
        let mut channels = serde_json::json!({
            "dense": dense_channels,
            "sparse": sparse_channels,
        });
        if !graph_channels.is_empty() {
            channels["graph"] = serde_json::json!(graph_channels);
        }
        result["channels"] = channels;
    }

    json_ok(response, &result)
}

fn query_point_json(
    collection: &str,
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    txn: &heed3::RoTxn,
    id: u128,
    score: f64,
    with_payload: bool,
) -> serde_json::Value {
    let mut point = serde_json::json!({
        "id": format_point_id(id),
        "version": 0,
        "score": score,
    });
    if with_payload {
        if let Some(pending) = pending_point(collection, id) {
            point["payload"] = pending_payload_json(&pending);
        } else if let Ok(node) = storage.get_node(txn, &id) {
            point["payload"] = value_map_to_json(&node.properties);
        }
    }
    point
}

fn query_point_json_be(
    collection: &str,
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    id: u128,
    score: f64,
    with_payload: bool,
    payload_cache: Option<&HashMap<u128, serde_json::Value>>,
) -> serde_json::Value {
    let mut point = serde_json::json!({
        "id": format_point_id(id),
        "version": 0,
        "score": score,
    });
    if with_payload {
        if let Some(pending) = pending_point(collection, id) {
            point["payload"] = pending_payload_json(&pending);
        } else if let Some(payload) = payload_cache.and_then(|cache| cache.get(&id)) {
            point["payload"] = payload.clone();
        } else {
            let reason = if payload_cache.is_some() {
                "cache_miss"
            } else {
                "cache_unavailable"
            };
            metrics::counter!(
                "helix_qdrant_payload_cache_serial_fallback_total",
                "collection" => collection.to_string(),
                "reason" => reason,
            )
            .increment(1);
            if let Ok(node) = storage.get_node_be(r, &id) {
                point["payload"] = value_map_to_json(&node.properties);
            }
        }
    }
    point
}

#[derive(Default)]
struct RequestPayloadCache {
    payloads: HashMap<u128, serde_json::Value>,
}

impl RequestPayloadCache {
    fn get_or_load<I>(
        &mut self,
        collection: &str,
        storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
        r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
        ids: I,
    ) -> Result<Option<&HashMap<u128, serde_json::Value>>, BackendError>
    where
        I: IntoIterator<Item = u128>,
    {
        let missing_ids: Vec<u128> = ids
            .into_iter()
            .filter(|id| !self.payloads.contains_key(id))
            .collect();
        if missing_ids.is_empty() {
            return Ok(Some(&self.payloads));
        }
        let Some(payloads) = lsm_payload_cache_be(collection, storage, r, missing_ids)? else {
            return Ok(None);
        };
        self.payloads.extend(payloads);
        Ok(Some(&self.payloads))
    }
}

fn lsm_payload_cache_be<I>(
    collection: &str,
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    ids: I,
) -> Result<Option<HashMap<u128, serde_json::Value>>, BackendError>
where
    I: IntoIterator<Item = u128>,
{
    let started = Instant::now();
    let mut requested = 0usize;
    let mut duplicate = 0usize;
    let mut pending = 0usize;
    let mut seen = HashSet::new();
    let mut unique_ids = Vec::new();
    let mut keys = Vec::new();
    for id in ids {
        requested += 1;
        if !seen.insert(id) {
            duplicate += 1;
            continue;
        }
        if pending_point(collection, id).is_some() {
            pending += 1;
            continue;
        }
        unique_ids.push(id);
        keys.push(id.to_be_bytes().to_vec());
    }
    if unique_ids.is_empty() {
        record_qdrant_payload_cache_count(collection, "none", "requested", requested);
        record_qdrant_payload_cache_count(collection, "none", "duplicate", duplicate);
        record_qdrant_payload_cache_count(collection, "none", "pending", pending);
        record_qdrant_payload_cache_count(collection, "none", "batched", 0);
        record_qdrant_payload_cache_stage(collection, "none", "empty", started);
        return Ok(Some(HashMap::new()));
    }

    let (backend, raw_values) = match (&*storage.backend, r) {
        (AnyBackend::Lsm(writer), AnyRead::Lsm(read)) => {
            match writer.collect_values_many_with(read, Namespace::Nodes, &keys) {
                Ok(values) => ("lsm", values),
                Err(error) if error.is_lsm_read_cancelled() => {
                    tracing::debug!(collection, "cancelled batched LSM payload cache build");
                    record_qdrant_payload_cache_stage(collection, "lsm", "cancelled", started);
                    return Err(error);
                }
                Err(error) => {
                    tracing::warn!(
                        collection,
                        error = %error,
                        "failed to build batched LSM payload cache"
                    );
                    record_qdrant_payload_cache_stage(collection, "lsm", "error", started);
                    return Ok(None);
                }
            }
        }
        (AnyBackend::LsmReader(reader), AnyRead::LsmReader(snap)) => {
            match reader.collect_values_many_with_at(snap.as_ref(), Namespace::Nodes, &keys) {
                Ok(values) => ("lsm_reader", values),
                Err(error) if error.is_lsm_read_cancelled() => {
                    tracing::debug!(
                        collection,
                        "cancelled batched LSM-reader payload cache build"
                    );
                    record_qdrant_payload_cache_stage(
                        collection,
                        "lsm_reader",
                        "cancelled",
                        started,
                    );
                    return Err(error);
                }
                Err(error) => {
                    tracing::warn!(
                        collection,
                        error = %error,
                        "failed to build batched LSM-reader payload cache"
                    );
                    record_qdrant_payload_cache_stage(collection, "lsm_reader", "error", started);
                    return Ok(None);
                }
            }
        }
        _ => {
            record_qdrant_payload_cache_stage(collection, "other", "unsupported", started);
            return Ok(None);
        }
    };
    record_qdrant_payload_cache_count(collection, backend, "requested", requested);
    record_qdrant_payload_cache_count(collection, backend, "duplicate", duplicate);
    record_qdrant_payload_cache_count(collection, backend, "pending", pending);
    record_qdrant_payload_cache_count(collection, backend, "batched", unique_ids.len());
    let mut payloads = HashMap::with_capacity(raw_values.len());
    for (id, raw) in unique_ids.into_iter().zip(raw_values) {
        let Some(bytes) = raw else {
            continue;
        };
        let Ok(node) = SerializedNode::decode_node(bytes.as_slice(), id) else {
            continue;
        };
        payloads.insert(id, value_map_to_json(&node.properties));
    }
    record_qdrant_payload_cache_count(collection, backend, "fetched", payloads.len());
    record_qdrant_payload_cache_stage(collection, backend, "ok", started);
    Ok(Some(payloads))
}

/// Build JSON response for dense vector query results.
fn build_query_response(
    response: &mut Response,
    collection: &str,
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    txn: &heed3::RoTxn,
    results: &[crate::helix_engine::vector_core::vector::HVector],
    with_payload: bool,
) -> Result<(), GraphError> {
    let mut out: Vec<serde_json::Value> = Vec::new();
    for v in results {
        let mut point = serde_json::json!({
            "id": format_point_id(v.id),
            "version": 0,
            "score": v.distance.unwrap_or(0.0),
        });
        if with_payload {
            if let Some(pending) = pending_point(collection, v.id) {
                point["payload"] = pending_payload_json(&pending);
            } else if let Ok(node) = storage.get_node(txn, &v.id) {
                point["payload"] = value_map_to_json(&node.properties);
            }
        }
        out.push(point);
    }
    json_ok(response, &serde_json::json!({ "points": out }))
}

fn build_query_response_be(
    response: &mut Response,
    collection: &str,
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    results: &[crate::helix_engine::vector_core::vector::HVector],
    with_payload: bool,
) -> Result<(), GraphError> {
    let payload_cache = if with_payload {
        lsm_payload_cache_be(collection, storage, r, results.iter().map(|v| v.id))
            .map_err(graph_error_from_backend_error)?
    } else {
        None
    };
    let mut out: Vec<serde_json::Value> = Vec::new();
    for v in results {
        out.push(query_point_json_be(
            collection,
            storage,
            r,
            v.id,
            v.distance.unwrap_or(0.0) as f64,
            with_payload,
            payload_cache.as_ref(),
        ));
    }
    json_ok(response, &serde_json::json!({ "points": out }))
}

/// Build JSON response for sparse vector query results.
fn build_sparse_query_response(
    response: &mut Response,
    collection: &str,
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    txn: &heed3::RoTxn,
    results: &[(u128, f64)],
    with_payload: bool,
) -> Result<(), GraphError> {
    let mut out: Vec<serde_json::Value> = Vec::new();
    for &(id, score) in results {
        let mut point = serde_json::json!({
            "id": format_point_id(id),
            "version": 0,
            "score": score,
        });
        if with_payload {
            if let Some(pending) = pending_point(collection, id) {
                point["payload"] = pending_payload_json(&pending);
            } else if let Ok(node) = storage.get_node(txn, &id) {
                point["payload"] = value_map_to_json(&node.properties);
            }
        }
        out.push(point);
    }
    json_ok(response, &serde_json::json!({ "points": out }))
}

fn build_sparse_query_response_be(
    response: &mut Response,
    collection: &str,
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    results: &[(u128, f64)],
    with_payload: bool,
) -> Result<(), GraphError> {
    let payload_cache = if with_payload {
        lsm_payload_cache_be(collection, storage, r, results.iter().map(|(id, _)| *id))
            .map_err(graph_error_from_backend_error)?
    } else {
        None
    };
    let mut out: Vec<serde_json::Value> = Vec::new();
    for &(id, score) in results {
        out.push(query_point_json_be(
            collection,
            storage,
            r,
            id,
            score,
            with_payload,
            payload_cache.as_ref(),
        ));
    }
    json_ok(response, &serde_json::json!({ "points": out }))
}

// ─── Utility Functions ───

/// Derive a u128 id from a string point id: hex first, then hash the string
/// deterministically. Exposed so other call sites that only have a point id
/// as a string (not a full `serde_json::Value`) can derive the same id
/// `point_id_to_u128` would have assigned on ingest.
pub(crate) fn string_id_to_u128(s: &str) -> u128 {
    u128::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or_else(|_| {
        use std::hash::Hasher;
        let hi = {
            let mut h = twox_hash::XxHash64::with_seed(0);
            h.write(s.as_bytes());
            h.finish()
        };
        let lo = {
            let mut h = twox_hash::XxHash64::with_seed(0x9E3779B97F4A7C15);
            h.write(s.as_bytes());
            h.finish()
        };
        ((hi as u128) << 64) | (lo as u128)
    })
}

fn point_id_to_u128(id: &serde_json::Value) -> u128 {
    match id {
        serde_json::Value::String(s) => string_id_to_u128(s),
        serde_json::Value::Number(n) => n.as_u64().unwrap_or(0) as u128,
        _ => 0,
    }
}

fn format_point_id(id: u128) -> String {
    format!("{:032x}", id)
}

// ─── Vector hydration on read paths ─────────────────────────────────────
//
// Qdrant's `with_vectors` supports `false | true | ["name1", "name2"]`. Prior
// versions of this handler parsed the field into a plain `bool` and then never
// used it — reads always returned an empty `vector: {}`. That broke several
// client flows (e.g. the context-engine segment-centroid-gate backfill, which
// scrolls points to pool per-path centroids, and any agent that wants to
// re-embed or re-rank using the stored vector).
//
// `WithVectors` accepts all three shapes, and `fetch_named_vectors` hydrates
// the requested vectors from the dense segments. Sparse vectors are intentionally
// skipped here — clients that need them can still drive the sparse search path.

/// Qdrant-compatible `with_payload` selector. Accepts:
/// - `bool` — true returns full payload, false returns none
/// - `[str]` — return only the named top-level fields (treated as include)
/// - `{include?: [str], exclude?: [str]}` — explicit include/exclude lists
///
/// Without this, qdrant-client's `PayloadSelectorInclude` shape was rejected
/// with `Invalid JSON: invalid type: map, expected a boolean`, surfacing as
/// 400s on every scroll-with-payload request from the upload-service.
#[derive(Default, Debug)]
enum WithPayload {
    #[default]
    All,
    None,
    Selector {
        include: Vec<String>,
        exclude: Vec<String>,
    },
}

impl<'de> serde::Deserialize<'de> for WithPayload {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = serde_json::Value::deserialize(deserializer)?;
        Ok(match raw {
            serde_json::Value::Null => WithPayload::All,
            serde_json::Value::Bool(true) => WithPayload::All,
            serde_json::Value::Bool(false) => WithPayload::None,
            serde_json::Value::Array(items) => {
                let include: Vec<String> = items
                    .into_iter()
                    .filter_map(|v| match v {
                        serde_json::Value::String(s) => Some(s),
                        _ => None,
                    })
                    .collect();
                if include.is_empty() {
                    WithPayload::None
                } else {
                    WithPayload::Selector {
                        include,
                        exclude: Vec::new(),
                    }
                }
            }
            serde_json::Value::Object(map) => {
                let to_strs = |v: Option<&serde_json::Value>| -> Vec<String> {
                    v.and_then(|x| x.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                                .collect()
                        })
                        .unwrap_or_default()
                };
                let include = to_strs(map.get("include"));
                let exclude = to_strs(map.get("exclude"));
                if include.is_empty() && exclude.is_empty() {
                    WithPayload::All
                } else {
                    WithPayload::Selector { include, exclude }
                }
            }
            // Unknown shape → be lenient and return everything rather than 400.
            _ => WithPayload::All,
        })
    }
}

impl WithPayload {
    fn is_active(&self) -> bool {
        !matches!(self, WithPayload::None)
    }

    fn label(&self) -> &'static str {
        match self {
            WithPayload::All => "all",
            WithPayload::None => "none",
            WithPayload::Selector { .. } => "selector",
        }
    }

    /// Project a JSON payload object according to the selector. For `All`,
    /// returns the input untouched; for `Selector`, returns a new object
    /// containing only the included fields with excluded fields removed.
    fn project(&self, payload: serde_json::Value) -> serde_json::Value {
        match self {
            WithPayload::All => payload,
            WithPayload::None => serde_json::Value::Null,
            WithPayload::Selector { include, exclude } => {
                let serde_json::Value::Object(map) = payload else {
                    return serde_json::Value::Null;
                };
                let mut filtered = serde_json::Map::new();
                if include.is_empty() {
                    // Exclude-only mode: copy everything except excluded keys.
                    for (k, v) in map {
                        if !exclude.iter().any(|e| key_matches(e, &k)) {
                            filtered.insert(k, v);
                        }
                    }
                } else {
                    for inc in include {
                        if let Some(v) = lookup_dotted(&map, inc) {
                            insert_dotted(&mut filtered, inc, v);
                        }
                    }
                    if !exclude.is_empty() {
                        let keys: Vec<String> = filtered.keys().cloned().collect();
                        for k in keys {
                            if exclude.iter().any(|e| key_matches(e, &k)) {
                                filtered.remove(&k);
                            }
                        }
                    }
                }
                serde_json::Value::Object(filtered)
            }
        }
    }
}

/// Match an exclude pattern against a payload key. Both exact match and the
/// dotted-prefix shape (`metadata.repo` excludes the `metadata.repo` projection)
/// are accepted.
fn key_matches(pattern: &str, key: &str) -> bool {
    pattern == key || pattern.starts_with(&format!("{}.", key))
}

/// Walk a dotted path (`metadata.repo`) into a nested JSON object and return
/// the leaf value if present.
fn lookup_dotted(
    map: &serde_json::Map<String, serde_json::Value>,
    path: &str,
) -> Option<serde_json::Value> {
    let mut current: serde_json::Value = serde_json::Value::Object(map.clone());
    for seg in path.split('.') {
        let next = current.as_object().and_then(|o| o.get(seg).cloned())?;
        current = next;
    }
    Some(current)
}

/// Insert a value at a dotted path, creating intermediate objects as needed.
fn insert_dotted(
    map: &mut serde_json::Map<String, serde_json::Value>,
    path: &str,
    value: serde_json::Value,
) {
    let segs: Vec<&str> = path.split('.').collect();
    if segs.len() == 1 {
        map.insert(segs[0].to_string(), value);
        return;
    }
    let head = segs[0];
    let entry = map
        .entry(head.to_string())
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    if let serde_json::Value::Object(child) = entry {
        insert_dotted(child, &segs[1..].join("."), value);
    }
}

#[derive(Default, Debug)]
enum WithVectors {
    #[default]
    None,
    All,
    Named(Vec<String>),
}

impl<'de> serde::Deserialize<'de> for WithVectors {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Accept: missing/null → None, bool → All/None, [str] → Named.
        let raw = serde_json::Value::deserialize(deserializer)?;
        Ok(match raw {
            serde_json::Value::Null => WithVectors::None,
            serde_json::Value::Bool(true) => WithVectors::All,
            serde_json::Value::Bool(false) => WithVectors::None,
            serde_json::Value::Array(items) => {
                let names: Vec<String> = items
                    .into_iter()
                    .filter_map(|v| match v {
                        serde_json::Value::String(s) => Some(s),
                        _ => None,
                    })
                    .collect();
                if names.is_empty() {
                    WithVectors::None
                } else {
                    WithVectors::Named(names)
                }
            }
            // Unknown shape → be lenient and skip hydration rather than 400.
            _ => WithVectors::None,
        })
    }
}

impl WithVectors {
    fn is_active(&self) -> bool {
        !matches!(self, WithVectors::None)
    }

    fn label(&self) -> &'static str {
        match self {
            WithVectors::None => "none",
            WithVectors::All => "all",
            WithVectors::Named(_) => "named",
        }
    }

    /// Return the set of logical vector names to hydrate, given the collection's
    /// known dense vector spaces. `All` expands to every dense name.
    fn resolved_names(&self, known: &[String]) -> Vec<String> {
        match self {
            WithVectors::None => Vec::new(),
            WithVectors::All => known.to_vec(),
            WithVectors::Named(names) => names
                .iter()
                .filter(|n| known.iter().any(|k| k == *n))
                .cloned()
                .collect(),
        }
    }
}

/// All hydratable vector names for a collection: dense spaces plus sparse
/// vectors. `WithVectors::resolved_names` filters requested names against this,
/// so `with_vector:true`/`all` and named requests resolve sparse vectors too —
/// not just dense. Without the sparse names here, scroll/get/scan silently drop
/// sparse vectors, which is what made the scroll->upsert migrator lose them.
fn known_vector_names(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
) -> Vec<String> {
    let mut names: Vec<String> = storage
        .named_vectors
        .list_vectors()
        .keys()
        .cloned()
        .collect();
    names.extend(storage.named_vectors.list_sparse_vectors().into_keys());
    names
}

/// Fetch named vector data for a point id, returning a JSON object of
/// `{ name: [f32, ...] }`. Missing vectors are simply absent.
///
/// Iterates the dense segments for each requested logical name and stops at the
/// first segment that owns the id (single-segment collections short-circuit on
/// the first try). Errors during lookup degrade to absent rather than failing
/// the whole read — an agent fetching one vector out of many shouldn't 500 the
/// request if one segment hiccups.
fn fetch_named_vectors(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    txn: &heed3::RoTxn,
    id: u128,
    names: &[String],
) -> serde_json::Value {
    let mut out = serde_json::Map::with_capacity(names.len());
    if names.is_empty() {
        return serde_json::Value::Object(out);
    }
    // Read lock held for the lifetime of all lookups — cheaper than reacquiring.
    let spaces = storage.named_vectors.list_dense_vector_spaces();
    let cores = match storage.named_vectors.cores_read() {
        Ok(c) => c,
        Err(_) => return serde_json::Value::Object(out),
    };
    for name in names {
        let Some(space) = spaces.get(name) else {
            // Not a dense space — try it as a sparse vector. The forward index
            // stores the doc's terms; export them in the same
            // `{indices, values}` object shape `split_mixed_vectors` parses on
            // upsert, so a scroll->upsert migration round-trips sparse vectors
            // (previously lost because only dense names were hydrated). Errors
            // (e.g. name is neither dense nor sparse) degrade to absent.
            if let Ok(Some(sv)) = storage
                .named_vectors
                .with_sparse_core(name, |core| core.get_doc_vector(txn, id))
            {
                out.insert(
                    name.clone(),
                    serde_json::json!({ "indices": sv.indices, "values": sv.values }),
                );
            }
            continue;
        };
        let mut found: Option<Vec<f32>> = None;
        for seg in &space.segments {
            let Some(core) = cores.get(&seg.physical_name) else {
                continue;
            };
            let rd = core.backend.read_borrowed(txn);
            match core.get_vector(&rd, id, 0, true) {
                Ok(hv) => {
                    let data = hv.get_data();
                    if !data.is_empty() {
                        found = Some(data.to_vec());
                        break;
                    }
                }
                Err(_) => continue, // id not in this segment, try the next
            }
        }
        if let Some(vec) = found {
            out.insert(name.clone(), serde_json::json!(vec));
        }
    }
    serde_json::Value::Object(out)
}

fn fetch_named_vectors_be(
    storage: &crate::helix_engine::storage_core::storage_core::HelixGraphStorage,
    r: &crate::helix_engine::storage_core::backend_any::AnyRead<'_>,
    id: u128,
    names: &[String],
) -> serde_json::Value {
    let mut out = serde_json::Map::with_capacity(names.len());
    if names.is_empty() {
        return serde_json::Value::Object(out);
    }
    let spaces = storage.named_vectors.list_dense_vector_spaces();
    let cores = match storage.named_vectors.cores_read() {
        Ok(c) => c,
        Err(_) => return serde_json::Value::Object(out),
    };
    for name in names {
        let Some(space) = spaces.get(name) else {
            if let Ok(Some(sv)) = storage
                .named_vectors
                .with_sparse_core(name, |core| core.get_doc_vector_be(r, id))
            {
                out.insert(
                    name.clone(),
                    serde_json::json!({ "indices": sv.indices, "values": sv.values }),
                );
            }
            continue;
        };
        let mut found: Option<Vec<f32>> = None;
        for seg in &space.segments {
            let Some(core) = cores.get(&seg.physical_name) else {
                continue;
            };
            match core.get_vector(r, id, 0, true) {
                Ok(hv) => {
                    let data = hv.get_data();
                    if !data.is_empty() {
                        found = Some(data.to_vec());
                        break;
                    }
                }
                Err(_) => continue,
            }
        }
        if let Some(vec) = found {
            out.insert(name.clone(), serde_json::json!(vec));
        }
    }
    serde_json::Value::Object(out)
}

fn json_payload_to_value_map(
    payload: &HashMap<String, serde_json::Value>,
) -> HashMap<String, Value> {
    payload
        .iter()
        .map(|(k, v)| (k.clone(), json_to_value(v)))
        .collect()
}

fn json_to_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::I64(i)
            } else if let Some(f) = n.as_f64() {
                Value::F64(f)
            } else {
                Value::String(n.to_string())
            }
        }
        serde_json::Value::Bool(b) => Value::Boolean(*b),
        serde_json::Value::Array(arr) => Value::Array(arr.iter().map(json_to_value).collect()),
        serde_json::Value::Object(obj) => {
            let map: HashMap<String, Value> = obj
                .iter()
                .map(|(k, v)| (k.clone(), json_to_value(v)))
                .collect();
            Value::Object(map)
        }
        serde_json::Value::Null => Value::String("null".into()),
    }
}

fn value_map_to_json(props: &HashMap<String, Value>) -> serde_json::Value {
    let map: serde_json::Map<String, serde_json::Value> = props
        .iter()
        .map(|(k, v)| (k.clone(), value_to_json(v)))
        .collect();
    serde_json::Value::Object(map)
}

fn value_to_json(v: &Value) -> serde_json::Value {
    match v {
        Value::String(s) => serde_json::Value::String(s.clone()),
        Value::I32(n) => serde_json::json!(*n),
        Value::I64(n) => serde_json::json!(*n),
        Value::F32(n) => serde_json::json!(*n),
        Value::F64(n) => serde_json::json!(*n),
        Value::U32(n) => serde_json::json!(*n),
        Value::U64(n) => serde_json::json!(*n),
        Value::Boolean(b) => serde_json::json!(*b),
        Value::Array(arr) => serde_json::Value::Array(arr.iter().map(value_to_json).collect()),
        Value::Object(map) => value_map_to_json(map),
        _ => serde_json::Value::Null,
    }
}

// ─── Alias Endpoints ───

/// Qdrant-compatible alias action.
#[derive(Deserialize)]
struct AliasActionBatch {
    actions: Vec<AliasAction>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum AliasAction {
    CreateAlias {
        collection_name: String,
        alias_name: String,
    },
    DeleteAlias {
        alias_name: String,
    },
}

/// `POST /collections/aliases` — batch alias create/delete (Qdrant-compatible).
///
/// Request body:
/// ```json
/// { "actions": [
///     { "create_alias": { "collection_name": "repo_v2", "alias_name": "repo" } },
///     { "delete_alias": { "alias_name": "old_repo" } }
/// ] }
/// ```
///
/// Actions are applied in order. If any action fails, earlier actions ARE already
/// committed (no transaction across actions — matches Qdrant semantics).
pub fn handle_alias_actions(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let batch: AliasActionBatch = sonic_rs::from_slice(&input.request.body)
        .map_err(|e| GraphError::New(format!("invalid alias request: {}", e)))?;

    for action in &batch.actions {
        match action {
            AliasAction::CreateAlias {
                collection_name,
                alias_name,
            } => {
                input
                    .collections
                    .create_alias(alias_name, collection_name)?;
            }
            AliasAction::DeleteAlias { alias_name } => {
                input.collections.delete_alias(alias_name)?;
            }
        }
    }

    json_ok(response, &true)
}

/// `GET /aliases` — list all aliases.
///
/// Response: `{ "status": "ok", "result": { "aliases": [ { "alias_name": "repo", "collection_name": "repo_v2" }, ... ] } }`
pub fn handle_list_aliases(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let aliases = input.collections.list_aliases()?;
    let alias_list: Vec<serde_json::Value> = aliases
        .into_iter()
        .map(|(alias_name, collection_name)| {
            serde_json::json!({
                "alias_name": alias_name,
                "collection_name": collection_name,
            })
        })
        .collect();
    json_ok(response, &serde_json::json!({ "aliases": alias_list }))
}

/// `GET /collections/{name}/aliases` — list aliases for a specific collection.
///
/// Response: `{ "status": "ok", "result": { "aliases": [ { "alias_name": "repo", "collection_name": "repo_v2" } ] } }`
pub fn handle_collection_aliases(
    input: &HandlerInput,
    response: &mut Response,
) -> Result<(), GraphError> {
    let name = input
        .path_params
        .get("name")
        .ok_or_else(|| GraphError::New("missing collection name".into()))?;

    let aliases = input.collections.aliases_for_collection(name)?;
    let alias_list: Vec<serde_json::Value> = aliases
        .into_iter()
        .map(|alias_name| {
            serde_json::json!({
                "alias_name": alias_name,
                "collection_name": name,
            })
        })
        .collect();
    json_ok(response, &serde_json::json!({ "aliases": alias_list }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helix_engine::graph_core::config::Config;
    use crate::helix_engine::graph_core::graph_core::{HelixGraphEngine, HelixGraphEngineOpts};
    use crate::helix_engine::storage_core::backend_lsm::{
        allow_lsm_blocking_cancellable, LsmReadCancellation,
    };
    use crate::helix_engine::storage_core::filters::{HasIdCondition, MatchValue};
    use crate::helix_engine::storage_core::metadata::STORAGE_METADATA_SIDECAR_FILE;
    use crate::helix_engine::storage_core::storage_methods::StorageMethods;
    use crate::helix_engine::storage_core::upsert::{EdgeUpsert, NodeUpsert};
    use crate::helix_engine::storage_core::{
        collection_manager::CollectionManager, replication::ReplicationManager,
    };
    use crate::helix_engine::vector_core::named_vectors::{DistanceMetric, NamedVectorConfig};
    use crate::helix_engine::vector_core::spindle::SpindleConfig;
    use crate::helix_engine::vector_core::vector_core::HNSWConfig;
    use crate::helix_gateway::api::graph::handle_rebuild_adjacency_for_paths;
    use crate::helix_gateway::router::router::HandlerInput;
    use crate::protocol::request::Request;
    use serial_test::serial;
    use tempfile::TempDir;

    pub(super) struct TestContext {
        _tmp: TempDir,
        graph: std::sync::Arc<HelixGraphEngine>,
        pub(super) collections: std::sync::Arc<CollectionManager>,
        replication: std::sync::Arc<ReplicationManager>,
    }

    fn setup() -> TestContext {
        setup_with_config(Config::default())
    }

    pub(super) fn setup_with_config(config: Config) -> TestContext {
        let tmp = TempDir::new().unwrap();
        let graph_path = tmp.path().join("graph");
        let collections_path = tmp.path().join("data");
        let graph = std::sync::Arc::new(
            HelixGraphEngine::new(HelixGraphEngineOpts {
                path: graph_path.display().to_string(),
                config: config.clone(),
            })
            .unwrap(),
        );
        let collections =
            std::sync::Arc::new(CollectionManager::new(collections_path, config.clone()).unwrap());
        let replication = std::sync::Arc::new(
            ReplicationManager::new(std::sync::Arc::clone(&collections), config).unwrap(),
        );
        TestContext {
            _tmp: tmp,
            graph,
            collections,
            replication,
        }
    }

    #[test]
    fn cancelled_lsm_payload_cache_stops_response_hydration() {
        let ctx = setup_with_config(Config::default().with_lsm_in_memory());
        let storage = ctx
            .collections
            .create_collection("cancelled_payload_cache")
            .unwrap();
        let read = storage.backend.begin_read().unwrap();
        let cancellation = LsmReadCancellation::new();
        cancellation.cancel();
        let mut response = Response::new();

        let result = allow_lsm_blocking_cancellable(cancellation, || {
            build_query_response_be(
                &mut response,
                "cancelled_payload_cache",
                &storage,
                &read,
                &[HVector::new(1, Vec::new())],
                true,
            )
        });

        assert!(
            matches!(result, Err(ref error) if error.to_string().contains("LSM read request cancelled")),
            "cancelled batched payload reads must terminate hydration instead of entering serial fallback"
        );
    }

    pub(super) fn make_input(
        ctx: &TestContext,
        method: &str,
        path: &str,
        body: Vec<u8>,
        path_params: HashMap<String, String>,
    ) -> HandlerInput {
        HandlerInput {
            request: Request {
                method: method.into(),
                headers: HashMap::new(),
                path: path.into(),
                body,
            },
            graph: std::sync::Arc::clone(&ctx.graph),
            collections: std::sync::Arc::clone(&ctx.collections),
            replication: std::sync::Arc::clone(&ctx.replication),
            path_params,
        }
    }

    fn wait_for_payload_index_ready(ctx: &TestContext, collection: &str, field: &str) {
        let storage = ctx.collections.get_collection(collection).unwrap();
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        while storage.has_payload_index(field).is_none() {
            assert!(
                Instant::now() < deadline,
                "payload index '{}' on '{}' did not become ready",
                field,
                collection
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    fn create_dense_collection(ctx: &TestContext, name: &str) {
        let storage = ctx.collections.create_collection(name).unwrap();
        let mut wtxn = storage.lmdb_env().unwrap().write_txn().unwrap();
        storage
            .named_vectors
            .create_vector_index(
                storage.lmdb_env().unwrap(),
                &mut wtxn,
                "dense",
                NamedVectorConfig {
                    size: 3,
                    distance: DistanceMetric::Cosine,
                    spindle: SpindleConfig::default(),
                },
                HNSWConfig::new(Some(16), Some(128), Some(768)),
            )
            .unwrap();
        storage
            .set_named_vectors_metadata(&mut wtxn, storage.named_vectors.list_vectors())
            .unwrap();
        storage
            .set_dense_vector_spaces_metadata(
                &mut wtxn,
                storage.named_vectors.list_dense_vector_spaces(),
            )
            .unwrap();
        wtxn.commit().unwrap();
        storage.refresh_metadata_snapshot().unwrap();
    }

    fn create_euclid_collection(ctx: &TestContext, name: &str) {
        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "vectors": {"dense": {"size": 3, "distance": "Euclid"}}
        }))
        .unwrap();
        let input = make_input(
            ctx,
            "PUT",
            &format!("/collections/{}", name),
            body,
            HashMap::from([("name".into(), name.into())]),
        );
        let mut response = Response::new();
        handle_create_collection(&input, &mut response).unwrap();
        assert!(
            response.status == 200 || response.status == 201,
            "create Euclid collection failed: {}",
            String::from_utf8_lossy(&response.body)
        );
    }

    /// Create a single-dense-vector collection on the LSM backend through the
    /// REAL Qdrant create-collection handler (`handle_create_collection` →
    /// `ReplicatedMutation::CreateCollection` → `apply_create_collection`). This
    /// exercises the production metadata/config persistence path, which now
    /// routes the named-vector / dense-space metadata writes through the backend
    /// seam (`set_named_vectors_metadata_be` etc.) on LSM instead of the
    /// heed-only `put_metadata` that `unreachable!`s there. The snapshot refresh
    /// mirrors what the LMDB `create_dense_collection` helper does (the
    /// in-process replication apply path does not refresh the snapshot for
    /// CreateCollection).
    pub(super) fn create_dense_collection_lsm(ctx: &TestContext, name: &str) {
        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "vectors": {
                "dense": {"size": 3, "distance": "Cosine"}
            }
        }))
        .unwrap();
        let input = make_input(
            ctx,
            "PUT",
            &format!("/collections/{}", name),
            body,
            HashMap::from([("name".into(), name.into())]),
        );
        let mut resp = Response::new();
        handle_create_collection(&input, &mut resp).unwrap();
        assert!(
            resp.status == 200 || resp.status == 201,
            "create-collection on LSM failed: status={} body={}",
            resp.status,
            String::from_utf8_lossy(&resp.body)
        );
        let storage = ctx.collections.get_collection(name).unwrap();
        if storage.backend.kind() != BackendKind::Lsm {
            storage.refresh_metadata_snapshot().unwrap();
        }
    }

    fn upsert_points(ctx: &TestContext, name: &str, points: serde_json::Value) {
        let body = sonic_rs::to_vec(&sonic_rs::json!({ "points": points })).unwrap();
        let input = make_input(
            ctx,
            "PUT",
            &format!("/collections/{}/points", name),
            body,
            HashMap::from([("name".into(), name.into())]),
        );
        let mut response = Response::new();
        handle_upsert_points(&input, &mut response).unwrap();
        assert_eq!(response.status, 200);
    }

    #[test]
    fn list_and_exists_handlers_do_not_open_cold_collections() {
        let ctx = setup();
        create_dense_collection(&ctx, "alpha");
        create_dense_collection(&ctx, "beta");
        upsert_points(
            &ctx,
            "alpha",
            serde_json::json!([{
                "id": "doc-1",
                "vector": {"dense": [0.1, 0.2, 0.3]},
                "payload": {"path": "src/main.rs"}
            }]),
        );
        ctx.collections.evict_collection("alpha");
        ctx.collections.evict_collection("beta");
        assert_eq!(ctx.collections.loaded_count(), 0);

        let list_input = make_input(&ctx, "GET", "/collections", Vec::new(), HashMap::new());
        let mut list_response = Response::new();
        handle_list_collections(&list_input, &mut list_response).unwrap();
        assert_eq!(list_response.status, 200);
        assert_eq!(ctx.collections.loaded_count(), 0);

        let exists_input = make_input(
            &ctx,
            "GET",
            "/collections/alpha/exists",
            Vec::new(),
            HashMap::from([("name".into(), "alpha".into())]),
        );
        let mut exists_response = Response::new();
        handle_collection_exists(&exists_input, &mut exists_response).unwrap();
        assert_eq!(exists_response.status, 200);
        assert_eq!(ctx.collections.loaded_count(), 0);
        let body: serde_json::Value = serde_json::from_slice(&exists_response.body).unwrap();
        assert_eq!(body["result"]["exists"], serde_json::json!(true));

        let info_input = make_input(
            &ctx,
            "GET",
            "/collections/alpha",
            Vec::new(),
            HashMap::from([("name".into(), "alpha".into())]),
        );
        let mut info_response = Response::new();
        handle_get_collection(&info_input, &mut info_response).unwrap();
        assert_eq!(info_response.status, 200);
        assert_eq!(ctx.collections.loaded_count(), 0);
        let body: serde_json::Value = serde_json::from_slice(&info_response.body).unwrap();
        assert_eq!(body["result"]["points_count"], serde_json::json!(1));
        assert_eq!(
            body["result"]["config"]["params"]["vectors"]["dense"]["size"],
            3
        );
        assert_eq!(
            body["result"]["warnings"][0]["code"],
            "COLD_COLLECTION_METADATA"
        );
    }

    #[test]
    fn collection_info_missing_sidecar_does_not_open_cold_collection() {
        let ctx = setup();
        create_dense_collection(&ctx, "missing_meta");
        ctx.collections.evict_collection("missing_meta");
        let sidecar_path = ctx
            ._tmp
            .path()
            .join("data")
            .join("collections")
            .join("missing_meta")
            .join(STORAGE_METADATA_SIDECAR_FILE);
        std::fs::remove_file(sidecar_path).unwrap();
        assert_eq!(ctx.collections.loaded_count(), 0);

        let info_input = make_input(
            &ctx,
            "GET",
            "/collections/missing_meta",
            Vec::new(),
            HashMap::from([("name".into(), "missing_meta".into())]),
        );
        let mut info_response = Response::new();
        handle_get_collection(&info_input, &mut info_response).unwrap();

        assert_eq!(info_response.status, 503);
        assert_eq!(ctx.collections.loaded_count(), 0);
        let body: serde_json::Value = serde_json::from_slice(&info_response.body).unwrap();
        assert!(body["status"]["error"]
            .as_str()
            .unwrap()
            .contains("metadata sidecar is not available"));
    }

    #[test]
    fn create_rejects_degenerate_hnsw_m_with_400() {
        let ctx = setup();
        let create = |name: &str, m: u64| {
            let body = sonic_rs::to_vec(&sonic_rs::json!({
                "vectors": { "dense": { "size": 3, "distance": "cosine" } },
                "hnsw_config": { "m": m }
            }))
            .unwrap();
            let input = make_input(
                &ctx,
                "PUT",
                &format!("/collections/{name}"),
                body,
                HashMap::from([("name".into(), name.into())]),
            );
            let mut response = Response::new();
            handle_create_collection(&input, &mut response).unwrap();
            response.status
        };

        assert_eq!(create("bad-m", 1), 400);
        assert!(ctx.collections.get_collection("bad-m").is_err());
        assert_eq!(create("huge-m", 129), 400);
        assert!(ctx.collections.get_collection("huge-m").is_err());
        let ok = create("good-m", 16);
        assert!(ok == 200 || ok == 201);
    }

    #[test]
    fn empty_idempotent_create_does_not_open_existing_cold_collection() {
        let ctx = setup();
        create_dense_collection(&ctx, "repo");
        ctx.collections.evict_collection("repo");
        assert_eq!(ctx.collections.loaded_count(), 0);

        let body = sonic_rs::to_vec(&sonic_rs::json!({ "vectors": {} })).unwrap();
        let input = make_input(
            &ctx,
            "PUT",
            "/collections/repo",
            body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut response = Response::new();
        handle_create_collection(&input, &mut response).unwrap();
        assert!(response.status == 200 || response.status == 201);
        assert_eq!(ctx.collections.loaded_count(), 0);
    }

    #[test]
    fn schema_idempotent_create_repairs_existing_collection_metadata() {
        let ctx = setup();
        ctx.collections.create_collection("repo").unwrap();
        ctx.collections.evict_collection("repo");
        assert_eq!(ctx.collections.loaded_count(), 0);

        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "vectors": {
                "dense": {
                    "size": 3,
                    "distance": "cosine"
                }
            }
        }))
        .unwrap();
        let input = make_input(
            &ctx,
            "PUT",
            "/collections/repo",
            body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut response = Response::new();
        handle_create_collection(&input, &mut response).unwrap();
        assert!(response.status == 200 || response.status == 201);

        let storage = ctx.collections.get_collection("repo").unwrap();
        assert_eq!(ctx.collections.loaded_count(), 1);
        let dense = storage
            .named_vectors
            .get_config("dense")
            .expect("schema-bearing idempotent create should repair dense vector metadata");
        assert_eq!(dense.size, 3);
        assert_eq!(dense.distance, DistanceMetric::Cosine);
    }

    #[test]
    fn set_payload_returns_qdrant_update_result_status() {
        let ctx = setup();
        create_dense_collection(&ctx, "repo");
        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([{
                "id": "doc-1",
                "vector": {"dense": [0.1, 0.2, 0.3]},
                "payload": {"path": "src/main.rs"}
            }]),
        );

        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "payload": {"_graph_backfilled": true},
            "key": "metadata",
            "points": ["doc-1"]
        }))
        .unwrap();
        let input = make_input(
            &ctx,
            "POST",
            "/collections/repo/points/payload?wait=true",
            body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut response = Response::new();
        handle_set_payload(&input, &mut response).unwrap();
        assert_eq!(response.status, 200);

        let body: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(body["status"], "ok");
        assert_eq!(body["result"]["status"], "completed");
        assert_ne!(body["result"]["status"], "ok");

        let scroll = run_scroll(&ctx, "repo", serde_json::json!(true));
        let payload = &scroll["result"]["points"][0]["payload"];
        assert_eq!(payload["metadata"]["_graph_backfilled"], true);
        assert!(
            payload.get("_graph_backfilled").is_none() || payload["_graph_backfilled"].is_null()
        );
    }

    #[test]
    fn set_payload_missing_point_does_not_create_point() {
        let ctx = setup();
        create_dense_collection(&ctx, "repo");

        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "payload": {"_pending_prune": true},
            "points": ["missing-doc"]
        }))
        .unwrap();
        let input = make_input(
            &ctx,
            "POST",
            "/collections/repo/points/payload?wait=true",
            body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut response = Response::new();
        let err = handle_set_payload(&input, &mut response).unwrap_err();
        assert!(matches!(err, GraphError::NodeNotFound), "err: {err:?}");

        let scroll = run_scroll(&ctx, "repo", serde_json::json!(true));
        let points = scroll["result"]["points"].as_array().unwrap();
        assert!(points.is_empty(), "unexpected point created: {points:?}");
    }

    #[test]
    fn upsert_points_persists_payload_fields_into_named_vectors() {
        let ctx = setup();
        create_dense_collection(&ctx, "repo");

        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "points": [{
                "id": "doc-1",
                "vector": {"dense": [0.1, 0.2, 0.3]},
                "payload": {
                    "path": "src/main.rs",
                    "git_branches": ["main", "feature/auth"]
                }
            }]
        }))
        .unwrap();
        let input = make_input(
            &ctx,
            "PUT",
            "/collections/repo/points",
            body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut response = Response::new();

        handle_upsert_points(&input, &mut response).unwrap();
        assert_eq!(response.status, 200);

        let storage = ctx.collections.get_collection("repo").unwrap();
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let point_id = point_id_to_u128(&serde_json::Value::String("doc-1".into()));
        let fields = storage
            .named_vectors
            .debug_fields(&txn, "dense", point_id)
            .unwrap();

        assert_eq!(
            fields.get("git_branches"),
            Some(&Value::Array(vec![
                Value::String("main".into()),
                Value::String("feature/auth".into()),
            ]))
        );
        assert_eq!(
            fields.get("path"),
            Some(&Value::String("src/main.rs".into()))
        );
    }

    #[test]
    fn get_collection_reports_vector_points_not_total_nodes() {
        let ctx = setup();
        create_dense_collection(&ctx, "repo");

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([{
                "id": "doc-1",
                "vector": {"dense": [0.1, 0.2, 0.3]},
                "payload": {"path": "src/main.rs"}
            }]),
        );

        let storage = ctx.collections.get_collection("repo").unwrap();
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        storage
            .create_node(
                &mut txn,
                "Symbol",
                vec![("name".into(), Value::String("helper".into()))],
                None,
                None,
            )
            .unwrap();
        txn.commit().unwrap();
        storage.refresh_metadata_snapshot().unwrap();

        let stats = ctx.collections.collection_stats("repo").unwrap();
        assert_eq!(stats.node_count, 2);
        assert_eq!(stats.vector_count, 1);

        let input = make_input(
            &ctx,
            "GET",
            "/collections/repo",
            vec![],
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut response = Response::new();
        handle_get_collection(&input, &mut response).unwrap();
        assert_eq!(response.status, 200);

        let body: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(body["result"]["points_count"], serde_json::json!(1));
        assert_eq!(body["result"]["vectors_count"], serde_json::json!(1));
        assert_eq!(
            body["result"]["indexed_vectors_count"],
            serde_json::json!(0)
        );
    }

    #[test]
    fn get_collection_reports_indexed_vectors_after_threshold_build() {
        let mut config = Config::default();
        config.vector_config.flat_scan_threshold = Some(1);
        let ctx = setup_with_config(config);
        create_dense_collection(&ctx, "repo");

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([{
                "id": "doc-1",
                "vector": {"dense": [0.1, 0.2, 0.3]},
                "payload": {"path": "src/main.rs"}
            }]),
        );

        let input = make_input(
            &ctx,
            "GET",
            "/collections/repo",
            vec![],
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut indexed_vectors_count = serde_json::json!(0);
        for _ in 0..50 {
            let mut response = Response::new();
            handle_get_collection(&input, &mut response).unwrap();
            assert_eq!(response.status, 200);

            let body: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
            indexed_vectors_count = body["result"]["indexed_vectors_count"].clone();
            if indexed_vectors_count == serde_json::json!(1) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        assert_eq!(indexed_vectors_count, serde_json::json!(1));
    }

    #[test]
    fn create_keyword_index_backfills_arrays_for_scroll_and_delete() {
        let ctx = setup();
        ctx.collections.create_collection("repo").unwrap();

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {
                    "id": "doc-1",
                    "payload": {
                        "git_branches": ["main", "feature/auth"],
                        "repo": "context-engine"
                    }
                },
                {
                    "id": "doc-2",
                    "payload": {
                        "git_branches": ["release"],
                        "repo": "context-engine"
                    }
                },
                {
                    "id": "doc-3",
                    "payload": {
                        "git_branches": ["main"],
                        "repo": "helix"
                    }
                }
            ]),
        );

        let create_index_body = sonic_rs::to_vec(&sonic_rs::json!({
            "field_name": "git_branches",
            "field_schema": "keyword",
        }))
        .unwrap();
        let create_index_input = make_input(
            &ctx,
            "PUT",
            "/collections/repo/index",
            create_index_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut create_index_response = Response::new();
        handle_create_index(&create_index_input, &mut create_index_response).unwrap();
        assert_eq!(create_index_response.status, 200);

        let scroll_body = sonic_rs::to_vec(&sonic_rs::json!({
            "limit": 10,
            "with_payload": true,
            "filter": {
                "must": [{
                    "key": "git_branches",
                    "match": {"value": "main"}
                }]
            }
        }))
        .unwrap();
        let scroll_input = make_input(
            &ctx,
            "POST",
            "/collections/repo/points/scroll",
            scroll_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut scroll_response = Response::new();
        handle_scroll_points(&scroll_input, &mut scroll_response).unwrap();
        assert_eq!(scroll_response.status, 200);

        let scroll_json: serde_json::Value = serde_json::from_slice(&scroll_response.body).unwrap();
        let points = scroll_json["result"]["points"].as_array().unwrap();
        assert_eq!(points.len(), 2);
        let returned_ids: HashSet<String> = points
            .iter()
            .filter_map(|point| point["id"].as_str().map(|id| id.to_string()))
            .collect();
        assert_eq!(returned_ids.len(), 2);

        let delete_body = sonic_rs::to_vec(&sonic_rs::json!({
            "filter": {
                "must": [{
                    "key": "git_branches",
                    "match": {"value": "main"}
                }]
            }
        }))
        .unwrap();
        let delete_input = make_input(
            &ctx,
            "POST",
            "/collections/repo/points/delete",
            delete_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut delete_response = Response::new();
        handle_delete_points(&delete_input, &mut delete_response).unwrap();
        assert_eq!(delete_response.status, 200);

        let stats = ctx.collections.collection_stats("repo").unwrap();
        assert_eq!(stats.node_count, 1);
    }

    #[test]
    fn filter_delete_releases_resize_guard_before_apply() {
        let ctx = setup();
        ctx.collections.create_collection("repo").unwrap();

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {"id": "doc-1", "payload": {"repo": "context-engine"}},
                {"id": "doc-2", "payload": {"repo": "helix"}}
            ]),
        );

        let delete_body = sonic_rs::to_vec(&sonic_rs::json!({
            "filter": {
                "must": [{"key": "repo", "match": {"value": "context-engine"}}]
            }
        }))
        .unwrap();
        let delete_input = make_input(
            &ctx,
            "POST",
            "/collections/repo/points/delete",
            delete_body,
            HashMap::from([("name".into(), "repo".into())]),
        );

        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let delete_thread = std::thread::spawn(move || {
            let mut response = Response::new();
            let result = handle_delete_points(&delete_input, &mut response)
                .map(|_| response.status)
                .map_err(|e| e.to_string());
            let _ = done_tx.send(result);
        });

        let status = done_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("filter delete deadlocked while applying replicated delete");
        delete_thread.join().unwrap();
        assert_eq!(status.unwrap(), 200);

        let stats = ctx.collections.collection_stats("repo").unwrap();
        assert_eq!(stats.node_count, 1);
    }

    #[test]
    fn create_numeric_index_supports_range_scroll() {
        let ctx = setup();
        ctx.collections.create_collection("repo").unwrap();

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {
                    "id": "doc-1",
                    "payload": {
                        "line_count": 12
                    }
                },
                {
                    "id": "doc-2",
                    "payload": {
                        "line_count": 42
                    }
                },
                {
                    "id": "doc-3",
                    "payload": {
                        "line_count": 80
                    }
                }
            ]),
        );

        let create_index_body = sonic_rs::to_vec(&sonic_rs::json!({
            "field_name": "line_count",
            "field_schema": "integer",
        }))
        .unwrap();
        let create_index_input = make_input(
            &ctx,
            "PUT",
            "/collections/repo/index",
            create_index_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut create_index_response = Response::new();
        handle_create_index(&create_index_input, &mut create_index_response).unwrap();
        assert_eq!(create_index_response.status, 200);

        let scroll_body = sonic_rs::to_vec(&sonic_rs::json!({
            "limit": 10,
            "with_payload": true,
            "filter": {
                "must": [{
                    "key": "line_count",
                    "range": {"gte": 20, "lt": 70}
                }]
            }
        }))
        .unwrap();
        let scroll_input = make_input(
            &ctx,
            "POST",
            "/collections/repo/points/scroll",
            scroll_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut scroll_response = Response::new();
        handle_scroll_points(&scroll_input, &mut scroll_response).unwrap();
        assert_eq!(scroll_response.status, 200);

        let scroll_json: serde_json::Value = serde_json::from_slice(&scroll_response.body).unwrap();
        let points = scroll_json["result"]["points"].as_array().unwrap();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0]["payload"]["line_count"], serde_json::json!(42));
    }

    #[test]
    fn unfiltered_scroll_honors_offset_cursor() {
        let ctx = setup();
        ctx.collections.create_collection("repo").unwrap();

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {"id": 1, "payload": {"seq": 1}},
                {"id": 2, "payload": {"seq": 2}},
                {"id": 3, "payload": {"seq": 3}},
                {"id": 4, "payload": {"seq": 4}}
            ]),
        );

        let scroll_body = sonic_rs::to_vec(&sonic_rs::json!({
            "limit": 2,
            "offset": 3,
            "with_payload": true,
        }))
        .unwrap();
        let scroll_input = make_input(
            &ctx,
            "POST",
            "/collections/repo/points/scroll",
            scroll_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut scroll_response = Response::new();
        handle_scroll_points(&scroll_input, &mut scroll_response).unwrap();
        assert_eq!(scroll_response.status, 200);

        let scroll_json: serde_json::Value = serde_json::from_slice(&scroll_response.body).unwrap();
        let points = scroll_json["result"]["points"].as_array().unwrap();
        assert_eq!(points.len(), 2);
        assert_eq!(
            points[0]["id"],
            serde_json::json!("00000000000000000000000000000003")
        );
        assert_eq!(
            points[1]["id"],
            serde_json::json!("00000000000000000000000000000004")
        );
    }

    #[test]
    fn qdrant_scroll_scan_budget_only_caps_filtered_requests() {
        assert_eq!(
            qdrant_scroll_scan_budget_for(10, false, 1_024, 64, 4_096, 100_000),
            None
        );
        assert_eq!(
            qdrant_scroll_scan_budget_for(1, true, 1_024, 64, 4_096, 100_000),
            Some(1_024)
        );
        assert_eq!(
            qdrant_scroll_scan_budget_for(100, true, 1_024, 64, 4_096, 100_000),
            Some(4_096)
        );
        assert_eq!(
            qdrant_scroll_scan_budget_for(100, true, 1_024, 64, 4_096, 2_000),
            Some(2_000)
        );
        assert_eq!(
            qdrant_scroll_scan_budget_for(100, true, 1_024, 64, 0, 100_000),
            None
        );
    }

    #[test]
    fn qdrant_filtered_scroll_yields_resume_offset_when_budget_spent() {
        let ctx = setup();
        ctx.collections.create_collection("repo").unwrap();

        let points: Vec<serde_json::Value> = (1..=1025)
            .map(|id| {
                serde_json::json!({
                    "id": id,
                    "payload": {
                        "repo": "context-engine",
                        "seq": id
                    }
                })
            })
            .collect();
        upsert_points(&ctx, "repo", serde_json::Value::Array(points));

        let scroll_body = sonic_rs::to_vec(&sonic_rs::json!({
            "limit": 1,
            "with_payload": true,
            "filter": {
                "must": [{
                    "key": "repo",
                    "match": {"value": "missing"}
                }]
            }
        }))
        .unwrap();
        let scroll_input = make_input(
            &ctx,
            "POST",
            "/collections/repo/points/scroll",
            scroll_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut scroll_response = Response::new();
        handle_scroll_points(&scroll_input, &mut scroll_response).unwrap();
        assert_eq!(scroll_response.status, 200);

        let first: serde_json::Value = serde_json::from_slice(&scroll_response.body).unwrap();
        assert_eq!(first["result"]["points"].as_array().unwrap().len(), 0);
        assert_eq!(
            first["result"]["next_page_offset"],
            serde_json::json!("00000000000000000000000000000401")
        );

        let next_body = sonic_rs::to_vec(&sonic_rs::json!({
            "limit": 1,
            "offset": first["result"]["next_page_offset"].as_str().unwrap(),
            "with_payload": true,
            "filter": {
                "must": [{
                    "key": "repo",
                    "match": {"value": "missing"}
                }]
            }
        }))
        .unwrap();
        let next_input = make_input(
            &ctx,
            "POST",
            "/collections/repo/points/scroll",
            next_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut next_response = Response::new();
        handle_scroll_points(&next_input, &mut next_response).unwrap();
        assert_eq!(next_response.status, 200);

        let second: serde_json::Value = serde_json::from_slice(&next_response.body).unwrap();
        assert_eq!(second["result"]["points"].as_array().unwrap().len(), 0);
        assert!(second["result"]["next_page_offset"].is_null());
    }

    #[test]
    fn helion_scan_pages_partition_with_cursor() {
        let ctx = setup();
        ctx.collections.create_collection("repo").unwrap();

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {"id": 1, "payload": {"repo": "context-engine", "seq": 1}},
                {
                    "id": "80000000000000000000000000000001",
                    "payload": {"repo": "context-engine", "seq": 2}
                },
                {
                    "id": "80000000000000000000000000000002",
                    "payload": {"repo": "context-engine", "seq": 3}
                },
                {
                    "id": "90000000000000000000000000000000",
                    "payload": {"repo": "helix", "seq": 4}
                }
            ]),
        );

        let create_index_body = sonic_rs::to_vec(&sonic_rs::json!({
            "field_name": "repo",
            "field_schema": "keyword",
        }))
        .unwrap();
        let create_index_input = make_input(
            &ctx,
            "PUT",
            "/collections/repo/index",
            create_index_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut create_index_response = Response::new();
        handle_create_index(&create_index_input, &mut create_index_response).unwrap();
        assert_eq!(create_index_response.status, 200);
        wait_for_payload_index_ready(&ctx, "repo", "repo");

        let scan_body = sonic_rs::to_vec(&sonic_rs::json!({
            "limit": 1,
            "partition": {"index": 1, "total": 2},
            "max_scan": 10,
            "with_payload": ["repo", "seq"],
            "filter": {
                "must": [{
                    "key": "repo",
                    "match": {"value": "context-engine"}
                }]
            }
        }))
        .unwrap();
        let scan_input = make_input(
            &ctx,
            "POST",
            "/v1/collections/repo/points/scan",
            scan_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut scan_response = Response::new();
        handle_scan_points(&scan_input, &mut scan_response).unwrap();
        assert_eq!(scan_response.status, 200);

        let first: serde_json::Value = serde_json::from_slice(&scan_response.body).unwrap();
        assert_eq!(
            first["result"]["scan"]["plan"],
            serde_json::json!("indexed_candidates")
        );
        assert_eq!(first["result"]["scan"]["returned"], serde_json::json!(1));
        assert_eq!(
            first["result"]["scan"]["partition"],
            serde_json::json!({"index": 1, "total": 2})
        );
        assert_eq!(
            first["result"]["points"][0]["id"],
            serde_json::json!("80000000000000000000000000000001")
        );
        assert_eq!(
            first["result"]["points"][0]["payload"]["seq"],
            serde_json::json!(2)
        );
        let cursor = first["result"]["next_cursor"].as_str().unwrap().to_string();
        assert!(cursor.starts_with(SCAN_CURSOR_PREFIX));

        let next_body = sonic_rs::to_vec(&sonic_rs::json!({
            "limit": 1,
            "cursor": cursor,
            "with_payload": ["seq"],
            "filter": {
                "must": [{
                    "key": "repo",
                    "match": {"value": "context-engine"}
                }]
            }
        }))
        .unwrap();
        let next_input = make_input(
            &ctx,
            "POST",
            "/v1/collections/repo/points/scan",
            next_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut next_response = Response::new();
        handle_scan_points(&next_input, &mut next_response).unwrap();
        assert_eq!(next_response.status, 200);

        let second: serde_json::Value = serde_json::from_slice(&next_response.body).unwrap();
        assert_eq!(
            second["result"]["points"][0]["id"],
            serde_json::json!("80000000000000000000000000000002")
        );
        assert_eq!(
            second["result"]["scan"]["partition"],
            serde_json::json!({"index": 1, "total": 2})
        );
    }

    #[test]
    fn helion_scan_budget_returns_resume_cursor() {
        let ctx = setup();
        ctx.collections.create_collection("repo").unwrap();

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {"id": 1, "payload": {"seq": 1}},
                {"id": 2, "payload": {"seq": 2}},
                {"id": 3, "payload": {"seq": 3}}
            ]),
        );

        let scan_body = sonic_rs::to_vec(&sonic_rs::json!({
            "limit": 3,
            "max_scan": 1,
            "with_payload": true
        }))
        .unwrap();
        let scan_input = make_input(
            &ctx,
            "POST",
            "/v1/collections/repo/points/scan",
            scan_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut scan_response = Response::new();
        handle_scan_points(&scan_input, &mut scan_response).unwrap();
        assert_eq!(scan_response.status, 200);

        let first: serde_json::Value = serde_json::from_slice(&scan_response.body).unwrap();
        assert_eq!(first["result"]["points"].as_array().unwrap().len(), 1);
        assert_eq!(first["result"]["scan"]["scanned"], serde_json::json!(1));
        assert_eq!(
            first["result"]["scan"]["budget_exhausted"],
            serde_json::json!(true)
        );
        let cursor = first["result"]["next_cursor"].as_str().unwrap().to_string();

        let next_body = sonic_rs::to_vec(&sonic_rs::json!({
            "limit": 3,
            "cursor": cursor,
            "max_scan": 10,
            "with_payload": true
        }))
        .unwrap();
        let next_input = make_input(
            &ctx,
            "POST",
            "/v1/collections/repo/points/scan",
            next_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut next_response = Response::new();
        handle_scan_points(&next_input, &mut next_response).unwrap();
        assert_eq!(next_response.status, 200);

        let second: serde_json::Value = serde_json::from_slice(&next_response.body).unwrap();
        let points = second["result"]["points"].as_array().unwrap();
        assert_eq!(points.len(), 2);
        assert_eq!(
            points[0]["id"],
            serde_json::json!("00000000000000000000000000000002")
        );
        assert_eq!(
            points[1]["id"],
            serde_json::json!("00000000000000000000000000000003")
        );
    }

    #[test]
    fn helion_scan_rejects_cursor_partition_mismatch() {
        let ctx = setup();
        ctx.collections.create_collection("repo").unwrap();

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {"id": 1, "payload": {"seq": 1}},
                {
                    "id": "80000000000000000000000000000001",
                    "payload": {"seq": 2}
                }
            ]),
        );

        let scan_body = sonic_rs::to_vec(&sonic_rs::json!({
            "limit": 1,
            "partition": {"index": 1, "total": 2}
        }))
        .unwrap();
        let scan_input = make_input(
            &ctx,
            "POST",
            "/v1/collections/repo/points/scan",
            scan_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut scan_response = Response::new();
        handle_scan_points(&scan_input, &mut scan_response).unwrap();
        assert_eq!(scan_response.status, 200);

        let first: serde_json::Value = serde_json::from_slice(&scan_response.body).unwrap();
        let cursor = first["result"]["next_cursor"].as_str().unwrap();

        let mismatch_body = sonic_rs::to_vec(&sonic_rs::json!({
            "limit": 1,
            "cursor": cursor,
            "partition": {"index": 0, "total": 2}
        }))
        .unwrap();
        let mismatch_input = make_input(
            &ctx,
            "POST",
            "/v1/collections/repo/points/scan",
            mismatch_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut mismatch_response = Response::new();
        handle_scan_points(&mismatch_input, &mut mismatch_response).unwrap();
        assert_eq!(mismatch_response.status, 400);

        let body: serde_json::Value = serde_json::from_slice(&mismatch_response.body).unwrap();
        assert_eq!(
            body["status"]["error"],
            serde_json::json!("cursor does not match the supplied partition")
        );
    }

    #[test]
    fn fully_indexed_count_returns_filter_scoped_count() {
        let ctx = setup();
        ctx.collections.create_collection("repo").unwrap();

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {"id": 1, "payload": {"repo": "context-engine", "branch": "main"}},
                {"id": 2, "payload": {"repo": "context-engine", "branch": "release"}},
                {"id": 3, "payload": {"repo": "helix", "branch": "main"}}
            ]),
        );

        for field in ["repo", "branch"] {
            let create_index_body = sonic_rs::to_vec(&sonic_rs::json!({
                "field_name": field,
                "field_schema": "keyword",
            }))
            .unwrap();
            let create_index_input = make_input(
                &ctx,
                "PUT",
                "/collections/repo/index",
                create_index_body,
                HashMap::from([("name".into(), "repo".into())]),
            );
            let mut create_index_response = Response::new();
            handle_create_index(&create_index_input, &mut create_index_response).unwrap();
            assert_eq!(create_index_response.status, 200);
        }

        let count_body = sonic_rs::to_vec(&sonic_rs::json!({
            "filter": {
                "must": [
                    {"key": "repo", "match": {"value": "context-engine"}},
                    {"key": "branch", "match": {"value": "main"}}
                ]
            }
        }))
        .unwrap();
        let count_input = make_input(
            &ctx,
            "POST",
            "/collections/repo/points/count",
            count_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut count_response = Response::new();
        handle_count_points(&count_input, &mut count_response).unwrap();
        assert_eq!(count_response.status, 200);

        let count_json: serde_json::Value = serde_json::from_slice(&count_response.body).unwrap();
        assert_eq!(count_json["result"]["count"], serde_json::json!(1));
    }

    /// Root-cause regression: on the LSM backend, `collection_stats`/count's
    /// unfiltered fast path reads the cached `metadata_snapshot` counter, which
    /// mirrors the separate merge-key counter and can drift from the true point
    /// count (see the recount/reseed fix). `exact: true` on an empty filter
    /// must bypass that cache and fall through to the same scan path a
    /// non-empty filter already uses.
    #[test]
    fn count_points_exact_true_bypasses_stale_cached_counter() {
        use crate::helix_engine::storage_core::backend::{BackendKind, Namespace, StorageBackend};
        use crate::helix_engine::storage_core::metadata::{
            encode_lsm_counter_value, lsm_counter_key, MetadataCounter,
        };

        let ctx = setup_with_config(Config::default().with_lsm_in_memory());
        ctx.collections.create_collection("repo_exact").unwrap();
        let storage = ctx.collections.get_collection("repo_exact").unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);

        upsert_points(
            &ctx,
            "repo_exact",
            serde_json::json!([
                {"id": 1, "payload": {"repo": "context-engine"}},
                {"id": 2, "payload": {"repo": "context-engine"}},
                {"id": 3, "payload": {"repo": "context-engine"}}
            ]),
        );

        // Corrupt the cached counter to a wrong low value, then force the
        // cached `metadata_snapshot` to pick it up — reproducing the real
        // fleet corruption, where a served `vector_count` reflects a stale
        // merge-key counter rather than the true point count.
        let mut w = storage.backend.begin_write().unwrap();
        storage
            .backend
            .put(
                &mut w,
                Namespace::Metadata,
                lsm_counter_key(MetadataCounter::Vectors),
                &encode_lsm_counter_value(1),
            )
            .unwrap();
        storage.backend.commit(w).unwrap();
        storage.refresh_metadata_snapshot().unwrap();

        // (b) exact omitted: unchanged — still serves the (now stale) cache.
        let cached_input = make_input(
            &ctx,
            "POST",
            "/collections/repo_exact/points/count",
            sonic_rs::to_vec(&sonic_rs::json!({})).unwrap(),
            HashMap::from([("name".into(), "repo_exact".into())]),
        );
        let mut cached_response = Response::new();
        handle_count_points(&cached_input, &mut cached_response).unwrap();
        assert_eq!(cached_response.status, 200);
        let cached_json: serde_json::Value = serde_json::from_slice(&cached_response.body).unwrap();
        assert_eq!(
            cached_json["result"]["count"],
            serde_json::json!(1),
            "without exact, the stale cached counter must be returned unchanged"
        );

        // (a) exact: true bypasses the cache and returns the true scan count.
        let exact_input = make_input(
            &ctx,
            "POST",
            "/collections/repo_exact/points/count",
            sonic_rs::to_vec(&sonic_rs::json!({"exact": true})).unwrap(),
            HashMap::from([("name".into(), "repo_exact".into())]),
        );
        let mut exact_response = Response::new();
        handle_count_points(&exact_input, &mut exact_response).unwrap();
        assert_eq!(exact_response.status, 200);
        let exact_json: serde_json::Value = serde_json::from_slice(&exact_response.body).unwrap();
        assert_eq!(
            exact_json["result"]["count"],
            serde_json::json!(3),
            "exact:true must scan and return the true point count, not the corrupted cache"
        );
    }

    #[test]
    fn exact_index_count_fast_path_applies_pending_overlay() {
        let candidates = HashSet::from([1u128, 2u128, 3u128]);
        let filter = Filter {
            must: vec![Condition::Field(FieldCondition {
                key: "repo".into(),
                match_cond: Some(MatchCondition::Value(MatchValue {
                    value: serde_json::json!("ce"),
                })),
                range: None,
            })],
            must_not: Vec::new(),
            should: Vec::new(),
            min_should: None,
        };
        let pending_points = vec![
            (
                2u128,
                PendingPoint {
                    input: PointInput {
                        id: serde_json::json!(2),
                        vector: HashMap::new(),
                        payload: HashMap::from([("repo".into(), serde_json::json!("helix"))]),
                    },
                    recorded_at: Instant::now(),
                    approx_bytes: 0,
                },
            ),
            (
                4u128,
                PendingPoint {
                    input: PointInput {
                        id: serde_json::json!(4),
                        vector: HashMap::new(),
                        payload: HashMap::from([("repo".into(), serde_json::json!("ce"))]),
                    },
                    recorded_at: Instant::now(),
                    approx_bytes: 0,
                },
            ),
        ];

        assert_eq!(
            count_exact_index_candidates_with_pending(&candidates, &pending_points, &filter),
            3
        );
    }

    #[test]
    fn must_not_indexed_clause_is_subtracted_from_scroll_candidates() {
        // Regression for the plan="primary" fallback on filters that
        // combined `must` with `must_not`. Pre-fix the candidate set
        // was computed from `must` only and the `must_not` predicate
        // was evaluated in the post-scan filter loop, so a narrow
        // exclusion still forced a full scan. After the fix the
        // planner subtracts `must_not` candidates up front and keeps
        // plan="indexed_candidates".
        let ctx = setup();
        ctx.collections.create_collection("repo").unwrap();

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {"id": 1, "payload": {"repo": "ce", "branch": "main"}},
                {"id": 2, "payload": {"repo": "ce", "branch": "feature/x"}},
                {"id": 3, "payload": {"repo": "ce", "branch": "release"}},
                {"id": 4, "payload": {"repo": "helix", "branch": "main"}},
            ]),
        );

        for field in ["repo", "branch"] {
            let body = sonic_rs::to_vec(&sonic_rs::json!({
                "field_name": field,
                "field_schema": "keyword",
            }))
            .unwrap();
            let input = make_input(
                &ctx,
                "PUT",
                "/collections/repo/index",
                body,
                HashMap::from([("name".into(), "repo".into())]),
            );
            let mut resp = Response::new();
            handle_create_index(&input, &mut resp).unwrap();
            assert_eq!(resp.status, 200);
            wait_for_payload_index_ready(&ctx, "repo", field);
        }

        // repo=ce AND NOT branch=feature/x → expect {1, 3}.
        let storage = ctx.collections.get_collection("repo").unwrap();
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let filter = Filter {
            must: vec![Condition::Field(FieldCondition {
                key: "repo".into(),
                match_cond: Some(MatchCondition::Value(MatchValue {
                    value: serde_json::json!("ce"),
                })),
                range: None,
            })],
            must_not: vec![Condition::Field(FieldCondition {
                key: "branch".into(),
                match_cond: Some(MatchCondition::Value(MatchValue {
                    value: serde_json::json!("feature/x"),
                })),
                range: None,
            })],
            should: Vec::new(),
            min_should: None,
        };
        let candidates = indexed_filter_candidates(&storage, &txn, &filter)
            .unwrap()
            .expect("indexed candidate set required");
        let mut got: Vec<u128> = candidates.into_iter().collect();
        got.sort();
        assert_eq!(got, vec![1u128, 3u128]);
    }

    #[test]
    fn must_not_unindexed_clause_falls_back_to_full_candidate_set() {
        // If any `must_not` clause targets an unindexed field we can't
        // safely subtract from the candidate set — the predicate is
        // only evaluable on full payloads. Fall back to returning the
        // unchanged `must` candidate superset and let the scan loop
        // do the final filtering. Correctness-preserving behavior.
        let ctx = setup();
        ctx.collections.create_collection("repo").unwrap();

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {"id": 1, "payload": {"repo": "ce", "draft": true}},
                {"id": 2, "payload": {"repo": "ce", "draft": false}},
            ]),
        );

        // Only `repo` is indexed; `draft` is not.
        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "field_name": "repo",
            "field_schema": "keyword",
        }))
        .unwrap();
        let input = make_input(
            &ctx,
            "PUT",
            "/collections/repo/index",
            body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut resp = Response::new();
        handle_create_index(&input, &mut resp).unwrap();
        assert_eq!(resp.status, 200);
        wait_for_payload_index_ready(&ctx, "repo", "repo");

        let storage = ctx.collections.get_collection("repo").unwrap();
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let filter = Filter {
            must: vec![Condition::Field(FieldCondition {
                key: "repo".into(),
                match_cond: Some(MatchCondition::Value(MatchValue {
                    value: serde_json::json!("ce"),
                })),
                range: None,
            })],
            must_not: vec![Condition::Field(FieldCondition {
                key: "draft".into(),
                match_cond: Some(MatchCondition::Value(MatchValue {
                    value: serde_json::json!(true),
                })),
                range: None,
            })],
            should: Vec::new(),
            min_should: None,
        };
        let candidates = indexed_filter_candidates(&storage, &txn, &filter)
            .unwrap()
            .expect("indexed must=repo should still derive a candidate set");
        // Both points match `repo=ce`; the unindexed must_not can't be
        // subtracted at planner time, so the scan loop gets the full
        // positive superset and applies the exclusion during matching.
        let mut got: Vec<u128> = candidates.into_iter().collect();
        got.sort();
        assert_eq!(got, vec![1u128, 2u128]);
    }

    #[test]
    fn must_not_nested_superset_is_not_subtracted_as_exact() {
        // `indexed_condition_candidates(Nested)` skips unindexed inner must
        // clauses, so the nested set {2, 3} is a superset of the real
        // exclusion {3}. Subtracting it used to drop point 2.
        let ctx = setup();
        ctx.collections.create_collection("repo").unwrap();
        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {"id": 1, "payload": {"repo": "ce", "branch": "main", "draft": true}},
                {"id": 2, "payload": {"repo": "ce", "branch": "feature/x", "draft": false}},
                {"id": 3, "payload": {"repo": "ce", "branch": "feature/x", "draft": true}},
            ]),
        );
        for field in ["repo", "branch"] {
            let body = sonic_rs::to_vec(&sonic_rs::json!({
                "field_name": field,
                "field_schema": "keyword",
            }))
            .unwrap();
            let input = make_input(
                &ctx,
                "PUT",
                "/collections/repo/index",
                body,
                HashMap::from([("name".into(), "repo".into())]),
            );
            let mut resp = Response::new();
            handle_create_index(&input, &mut resp).unwrap();
            assert_eq!(resp.status, 200);
            wait_for_payload_index_ready(&ctx, "repo", field);
        }

        let filter: Filter = serde_json::from_value(serde_json::json!({
            "must": [{"key": "repo", "match": {"value": "ce"}}],
            "must_not": [{"must": [
                {"key": "branch", "match": {"value": "feature/x"}},
                {"key": "draft", "match": {"value": true}}
            ]}]
        }))
        .unwrap();
        let storage = ctx.collections.get_collection("repo").unwrap();
        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let candidates = indexed_filter_candidates(&storage, &txn, &filter)
            .unwrap()
            .expect("indexed must=repo should still derive a candidate set");
        let mut got: Vec<u128> = candidates.into_iter().collect();
        got.sort();
        assert_eq!(got, vec![1u128, 2u128, 3u128]);
        drop(txn);

        let body = sonic_rs::to_vec(&sonic_rs::json!({"filter": filter, "limit": 10})).unwrap();
        let input = make_input(
            &ctx,
            "POST",
            "/collections/repo/points/scroll",
            body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut resp = Response::new();
        handle_scroll_points(&input, &mut resp).unwrap();
        assert_eq!(resp.status, 200);
        let json: serde_json::Value = serde_json::from_slice(&resp.body).unwrap();
        let mut ids: Vec<String> = json["result"]["points"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["id"].as_str().unwrap().to_string())
            .collect();
        ids.sort();
        assert_eq!(ids, vec![format_point_id(1), format_point_id(2)]);
    }

    #[test]
    fn must_not_text_match_excludes_every_substring_hit() {
        // The keyword index answers a text match with only the exact hit
        // ("api") when one exists, so subtracting it as exact used to keep
        // "api-gateway" even though it contains the excluded text.
        let ctx = setup();
        ctx.collections.create_collection("repo").unwrap();
        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {"id": 1, "payload": {"repo": "api"}},
                {"id": 2, "payload": {"repo": "api-gateway"}},
                {"id": 3, "payload": {"repo": "web"}},
            ]),
        );
        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "field_name": "repo",
            "field_schema": "keyword",
        }))
        .unwrap();
        let input = make_input(
            &ctx,
            "PUT",
            "/collections/repo/index",
            body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut resp = Response::new();
        handle_create_index(&input, &mut resp).unwrap();
        assert_eq!(resp.status, 200);
        wait_for_payload_index_ready(&ctx, "repo", "repo");

        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "filter": {"must_not": [{"key": "repo", "match": {"text": "api"}}]},
            "limit": 10
        }))
        .unwrap();
        let input = make_input(
            &ctx,
            "POST",
            "/collections/repo/points/scroll",
            body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut resp = Response::new();
        handle_scroll_points(&input, &mut resp).unwrap();
        assert_eq!(resp.status, 200);
        let json: serde_json::Value = serde_json::from_slice(&resp.body).unwrap();
        let ids: Vec<String> = json["result"]["points"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["id"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(ids, vec![format_point_id(3)]);
    }

    /// Regression: a positive `match.text` filter on a keyword-indexed field
    /// used to resolve candidates to ONLY the exact keyword hits whenever one
    /// existed, so scroll returned a truncated page with `next_page_offset:
    /// null` and exact count under-reported — every point whose value merely
    /// contained the text was silently dropped (prod: text "VectorError" on
    /// `callee_symbol` counted 5 vs 179 true matches).
    fn assert_text_match_returns_every_substring_hit(ctx: &TestContext) {
        ctx.collections.create_collection("graph").unwrap();
        upsert_points(
            ctx,
            "graph",
            serde_json::json!([
                {"id": 1, "payload": {"callee_symbol": "VectorError"}},
                {"id": 2, "payload": {"callee_symbol": "VectorError::VectorCoreError"}},
                {"id": 3, "payload": {"callee_symbol": "crate::vectorerror::Io"}},
                {"id": 4, "payload": {"callee_symbol": "Unrelated"}},
            ]),
        );
        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "field_name": "callee_symbol",
            "field_schema": "keyword",
        }))
        .unwrap();
        let input = make_input(
            ctx,
            "PUT",
            "/collections/graph/index",
            body,
            HashMap::from([("name".into(), "graph".into())]),
        );
        let mut resp = Response::new();
        handle_create_index(&input, &mut resp).unwrap();
        assert_eq!(resp.status, 200);
        wait_for_payload_index_ready(ctx, "graph", "callee_symbol");

        let filter = serde_json::json!({
            "must": [{"key": "callee_symbol", "match": {"text": "VectorError"}}]
        });
        let body = sonic_rs::to_vec(&sonic_rs::json!({"filter": filter, "limit": 10})).unwrap();
        let input = make_input(
            ctx,
            "POST",
            "/collections/graph/points/scroll",
            body,
            HashMap::from([("name".into(), "graph".into())]),
        );
        let mut resp = Response::new();
        handle_scroll_points(&input, &mut resp).unwrap();
        assert_eq!(resp.status, 200);
        let json: serde_json::Value = serde_json::from_slice(&resp.body).unwrap();
        let ids: Vec<String> = json["result"]["points"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["id"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            ids,
            vec![format_point_id(1), format_point_id(2), format_point_id(3)],
            "text match must return every substring hit, not just the exact keyword hit"
        );
        assert!(json["result"]["next_page_offset"].is_null());

        let body = sonic_rs::to_vec(&sonic_rs::json!({"filter": filter, "exact": true})).unwrap();
        let input = make_input(
            ctx,
            "POST",
            "/collections/graph/points/count",
            body,
            HashMap::from([("name".into(), "graph".into())]),
        );
        let mut resp = Response::new();
        handle_count_points(&input, &mut resp).unwrap();
        assert_eq!(resp.status, 200);
        let json: serde_json::Value = serde_json::from_slice(&resp.body).unwrap();
        assert_eq!(json["result"]["count"], serde_json::json!(3));
    }

    #[test]
    fn text_match_returns_every_substring_hit_lmdb() {
        assert_text_match_returns_every_substring_hit(&setup());
    }

    #[test]
    fn text_match_returns_every_substring_hit_lsm() {
        let ctx = setup_with_config(Config::default().with_lsm_in_memory());
        assert_text_match_returns_every_substring_hit(&ctx);
        let storage = ctx.collections.get_collection("graph").unwrap();
        assert_eq!(storage.backend.kind(), BackendKind::Lsm);
    }

    fn text_walk_ctx(ctx: &TestContext) -> std::sync::Arc<HelixGraphStorage> {
        let storage = ctx.collections.create_collection("tw").unwrap();
        upsert_points(
            ctx,
            "tw",
            serde_json::json!([
                {"id": 1, "payload": {"sym": "VectorError", "repo": "a"}},
                {"id": 2, "payload": {"sym": "VectorError::Io", "repo": "b"}},
                {"id": 3, "payload": {"sym": "crate::vectorerror::X", "repo": "b"}},
                {"id": 4, "payload": {"sym": "Unrelated", "repo": "a"}},
                {"id": 5, "payload": {"sym": "Other", "repo": "b"}},
            ]),
        );
        for field in ["sym", "repo"] {
            storage
                .create_payload_index(field, PayloadIndexSchema::Keyword)
                .unwrap();
            wait_for_payload_index_ready(ctx, "tw", field);
        }
        storage
    }

    fn text_cond(key: &str, text: &str) -> Condition {
        use crate::helix_engine::storage_core::filters::MatchText;
        FieldCondition {
            key: key.into(),
            match_cond: Some(MatchCondition::Text(MatchText { text: text.into() })),
            range: None,
        }
        .into()
    }

    fn keyword_cond(key: &str, value: &str) -> Condition {
        FieldCondition {
            key: key.into(),
            match_cond: Some(MatchCondition::Value(MatchValue {
                value: serde_json::json!(value),
            })),
            range: None,
        }
        .into()
    }

    fn scan_opts<'a>(filter: &'a Filter, limit: usize, offset_id: u128) -> PointScanOptions<'a> {
        PointScanOptions {
            collection: "tw",
            limit,
            offset_id,
            filter,
            with_payload: &WithPayload::All,
            with_vectors: &WithVectors::None,
            partition: None,
            max_scan: None,
            deadline: None,
            filter_hash: scan_filter_hash(filter),
            cursor_generation: None,
            fuse_pending: false,
        }
    }

    fn scan_ids(scan: &PointScanResult) -> Vec<u128> {
        scan.points.iter().filter_map(point_json_id).collect()
    }

    #[test]
    fn must_text_walk_skipped_when_keyword_must_is_selective() {
        let ctx = setup_with_config(Config::default().with_lsm_in_memory());
        let storage = text_walk_ctx(&ctx);
        let filter = Filter {
            must: vec![text_cond("sym", "vectorerror"), keyword_cond("repo", "b")],
            ..Default::default()
        };
        let r = storage.backend.begin_read().unwrap();
        let scan = execute_point_scan_be(&storage, &r, &scan_opts(&filter, 10, 0)).unwrap();
        assert!(matches!(scan.plan, PointScanPlan::IndexedCandidates));
        // repo=b alone (3 ids): the text condition was not walked/intersected
        // (that would give 2) — the recheck applies it instead.
        assert_eq!(scan.index_candidates, Some(3));
        assert_eq!(scan_ids(&scan), vec![2, 3]);
        assert!(scan.next_offset.is_none());
    }

    #[test]
    fn should_branches_prechecked_before_any_text_walk() {
        let ctx = setup_with_config(Config::default().with_lsm_in_memory());
        let storage = text_walk_ctx(&ctx);
        let unindexed = keyword_cond("not_indexed", "x");
        assert!(condition_may_yield_candidates(
            &storage,
            &text_cond("sym", "io")
        ));
        assert!(!condition_may_yield_candidates(&storage, &unindexed));
        let nested = Condition::Nested(Filter {
            should: vec![text_cond("sym", "io"), unindexed.clone()],
            ..Default::default()
        });
        assert!(!condition_may_yield_candidates(&storage, &nested));

        let filter = Filter {
            should: vec![text_cond("sym", "vectorerror"), unindexed],
            ..Default::default()
        };
        let r = storage.backend.begin_read().unwrap();
        assert!(indexed_filter_candidates_be(&storage, &r, &filter)
            .unwrap()
            .is_none());
        let scan = execute_point_scan_be(&storage, &r, &scan_opts(&filter, 10, 0)).unwrap();
        assert!(matches!(scan.plan, PointScanPlan::Primary));
        assert_eq!(scan_ids(&scan), vec![1, 2, 3]);
    }

    fn assert_text_walk_cap(storage: &HelixGraphStorage) {
        let r = storage.backend.begin_read().unwrap();
        let all = storage
            .get_nodes_by_payload_text_be(&r, "sym", "vectorerror", 0)
            .unwrap();
        assert_eq!(all, Some(vec![1, 2, 3]));
        assert_eq!(
            storage
                .get_nodes_by_payload_text_be(&r, "sym", "vectorerror", 100)
                .unwrap(),
            Some(vec![1, 2, 3])
        );
        assert_eq!(
            storage
                .get_nodes_by_payload_text_be(&r, "sym", "vectorerror", 2)
                .unwrap(),
            None,
            "a walk over the cap must fall back to the unindexed plan"
        );
    }

    #[test]
    fn text_walk_cap_returns_none_lsm() {
        let ctx = setup_with_config(Config::default().with_lsm_in_memory());
        assert_text_walk_cap(&text_walk_ctx(&ctx));
    }

    #[test]
    fn text_walk_cap_returns_none_lmdb() {
        let ctx = setup();
        let storage = text_walk_ctx(&ctx);
        let txn = storage.begin_resize_safe_read_txn().unwrap();
        assert_eq!(
            storage
                .get_nodes_by_payload_text(&txn, "sym", "vectorerror", 0)
                .unwrap(),
            Some(vec![1, 2, 3])
        );
        assert_eq!(
            storage
                .get_nodes_by_payload_text(&txn, "sym", "vectorerror", 2)
                .unwrap(),
            None
        );
        drop(txn);
        assert_text_walk_cap(&storage);
    }

    /// Matching terms with several duplicate ids exercise the LMDB
    /// `move_between_keys` + `get_duplicates` path and the LSM per-term
    /// verdict reuse.
    #[test]
    fn text_walk_collects_every_dup_of_matching_terms() {
        for lsm in [false, true] {
            let ctx = if lsm {
                setup_with_config(Config::default().with_lsm_in_memory())
            } else {
                setup()
            };
            let storage = text_walk_ctx(&ctx);
            upsert_points(
                &ctx,
                "tw",
                serde_json::json!([
                    {"id": 6, "payload": {"sym": "VectorError", "repo": "a"}},
                    {"id": 7, "payload": {"sym": "VectorError", "repo": "b"}},
                    {"id": 8, "payload": {"sym": "Unrelated", "repo": "b"}},
                ]),
            );
            let expected = Some(vec![1, 2, 3, 6, 7]);
            if !lsm {
                let txn = storage.begin_resize_safe_read_txn().unwrap();
                assert_eq!(
                    storage
                        .get_nodes_by_payload_text(&txn, "sym", "vectorerror", 0)
                        .unwrap(),
                    expected,
                    "lmdb heed walk"
                );
            }
            let r = storage.backend.begin_read().unwrap();
            assert_eq!(
                storage
                    .get_nodes_by_payload_text_be(&r, "sym", "vectorerror", 0)
                    .unwrap(),
                expected,
                "backend walk (lsm={lsm})"
            );
            let filter = Filter {
                must: vec![text_cond("sym", "VectorError")],
                ..Default::default()
            };
            let scan = execute_point_scan_be(&storage, &r, &scan_opts(&filter, 10, 0)).unwrap();
            assert_eq!(scan_ids(&scan), vec![1, 2, 3, 6, 7], "scan (lsm={lsm})");
        }
    }

    /// A value whose encoding exceeds the raw-key limit is stored under a
    /// hashed `\xffhxk1` key that cannot be decoded for substring matching:
    /// the walk must abstain (None) and the query must still be answered
    /// correctly by the recheck scan.
    #[test]
    fn text_walk_abstains_on_hashed_keys_and_recheck_stays_correct() {
        for lsm in [false, true] {
            let ctx = if lsm {
                setup_with_config(Config::default().with_lsm_in_memory())
            } else {
                setup()
            };
            let storage = text_walk_ctx(&ctx);
            let long = format!("VectorError::{}", "x".repeat(600));
            upsert_points(
                &ctx,
                "tw",
                serde_json::json!([{"id": 9, "payload": {"sym": long, "repo": "a"}}]),
            );
            if !lsm {
                let txn = storage.begin_resize_safe_read_txn().unwrap();
                assert_eq!(
                    storage
                        .get_nodes_by_payload_text(&txn, "sym", "vectorerror", 0)
                        .unwrap(),
                    None,
                    "lmdb heed walk must abstain on a hashed key"
                );
            }
            let r = storage.backend.begin_read().unwrap();
            assert_eq!(
                storage
                    .get_nodes_by_payload_text_be(&r, "sym", "vectorerror", 0)
                    .unwrap(),
                None,
                "hashed key must make the walk abstain (lsm={lsm})"
            );
            let filter = Filter {
                must: vec![text_cond("sym", "VectorError")],
                ..Default::default()
            };
            let scan = execute_point_scan_be(&storage, &r, &scan_opts(&filter, 10, 0)).unwrap();
            assert!(matches!(scan.plan, PointScanPlan::Primary));
            assert_eq!(scan_ids(&scan), vec![1, 2, 3, 9], "recheck (lsm={lsm})");
            assert!(scan.next_offset.is_none());
        }
    }

    #[test]
    fn multi_page_text_scroll_returns_every_hit() {
        let ctx = setup_with_config(Config::default().with_lsm_in_memory());
        let storage = text_walk_ctx(&ctx);
        let filter = Filter {
            must: vec![text_cond("sym", "VectorError")],
            ..Default::default()
        };
        let r = storage.backend.begin_read().unwrap();
        let mut offset = 0u128;
        let mut seen = Vec::new();
        for _ in 0..10 {
            let scan = execute_point_scan_be(&storage, &r, &scan_opts(&filter, 1, offset)).unwrap();
            seen.extend(scan_ids(&scan));
            match scan.next_offset {
                Some(next) => offset = parse_scroll_offset(Some(&serde_json::json!(next))),
                None => break,
            }
        }
        assert_eq!(seen, vec![1, 2, 3]);
    }

    #[test]
    fn scroll_honors_min_should() {
        let ctx = setup();
        ctx.collections.create_collection("repo").unwrap();
        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {"id": 1, "payload": {"lang": "rust", "kind": "fn", "pub": true}},
                {"id": 2, "payload": {"lang": "rust", "kind": "struct", "pub": false}},
                {"id": 3, "payload": {"lang": "go", "kind": "fn", "pub": false}},
            ]),
        );
        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "filter": {"min_should": {
                "conditions": [
                    {"key": "lang", "match": {"value": "rust"}},
                    {"key": "kind", "match": {"value": "fn"}},
                    {"key": "pub", "match": {"value": true}}
                ],
                "min_count": 2
            }},
            "limit": 10
        }))
        .unwrap();
        let input = make_input(
            &ctx,
            "POST",
            "/collections/repo/points/scroll",
            body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut resp = Response::new();
        handle_scroll_points(&input, &mut resp).unwrap();
        assert_eq!(resp.status, 200, "{}", String::from_utf8_lossy(&resp.body));
        let json: serde_json::Value = serde_json::from_slice(&resp.body).unwrap();
        let ids: Vec<String> = json["result"]["points"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["id"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(ids, vec![format_point_id(1)]);
    }

    #[test]
    fn range_index_bounds_use_the_stricter_of_inclusive_and_exclusive() {
        let range = |gte, gt, lte, lt| RangeCondition { gte, gt, lte, lt };
        assert_eq!(
            range_lower_bound(&range(Some(10.0), Some(20.0), None, None)),
            Some((20.0, false))
        );
        assert_eq!(
            range_lower_bound(&range(Some(20.0), Some(10.0), None, None)),
            Some((20.0, true))
        );
        assert_eq!(
            range_lower_bound(&range(Some(5.0), Some(5.0), None, None)),
            Some((5.0, false))
        );
        assert_eq!(
            range_lower_bound(&range(None, Some(3.0), None, None)),
            Some((3.0, false))
        );
        assert_eq!(
            range_upper_bound(&range(None, None, Some(20.0), Some(10.0))),
            Some((10.0, false))
        );
        assert_eq!(
            range_upper_bound(&range(None, None, Some(10.0), Some(20.0))),
            Some((10.0, true))
        );
        assert_eq!(
            range_upper_bound(&range(None, None, Some(5.0), Some(5.0))),
            Some((5.0, false))
        );
        assert_eq!(range_upper_bound(&range(None, None, None, None)), None);
    }

    #[test]
    fn unsupported_filter_condition_is_rejected_with_400() {
        let ctx = setup();
        create_dense_collection(&ctx, "repo");
        for filter in [
            serde_json::json!({"must": [{"key": "t", "range": {"gte": "2024-01-01T00:00:00Z"}}]}),
            serde_json::json!({"must_not": [{"key": "lang", "match": {"phrase": "rust"}}]}),
            serde_json::json!({"must": [{"nested": {"key": "a", "filter": {}}}]}),
        ] {
            let body = sonic_rs::to_vec(&sonic_rs::json!({
                "vector": {"name": "dense", "vector": [1.0, 0.0, 0.0]},
                "filter": filter,
            }))
            .unwrap();
            let input = make_input(
                &ctx,
                "POST",
                "/collections/repo/points/search",
                body,
                HashMap::from([("name".into(), "repo".into())]),
            );
            let mut resp = Response::new();
            handle_search_points(&input, &mut resp).unwrap();
            assert_eq!(resp.status, 400, "{filter}");
        }
    }

    #[test]
    fn search_and_query_honor_offset_and_score_threshold() {
        for lsm in [false, true] {
            let ctx = if lsm {
                setup_with_config(Config::default().with_lsm_in_memory())
            } else {
                setup()
            };
            if lsm {
                create_dense_collection_lsm(&ctx, "repo");
            } else {
                create_dense_collection(&ctx, "repo");
            }
            upsert_points(
                &ctx,
                "repo",
                serde_json::json!([
                    {"id": 1, "vector": {"dense": [1.0, 0.0, 0.0]}, "payload": {"lang": "rust"}},
                    {"id": 2, "vector": {"dense": [0.9, 0.1, 0.0]}, "payload": {"lang": "rust"}},
                    {"id": 3, "vector": {"dense": [0.0, 1.0, 0.0]}, "payload": {"lang": "go"}},
                ]),
            );
            let run = |path: &str, body: serde_json::Value| -> Vec<(String, f64)> {
                let input = make_input(
                    &ctx,
                    "POST",
                    path,
                    sonic_rs::to_vec(&body).unwrap(),
                    HashMap::from([("name".into(), "repo".into())]),
                );
                let mut resp = Response::new();
                if path.ends_with("/search") {
                    handle_search_points(&input, &mut resp).unwrap();
                } else {
                    handle_query_points(&input, &mut resp).unwrap();
                }
                assert_eq!(resp.status, 200, "{}", String::from_utf8_lossy(&resp.body));
                let json: serde_json::Value = serde_json::from_slice(&resp.body).unwrap();
                let points = json["result"]
                    .get("points")
                    .unwrap_or(&json["result"])
                    .as_array()
                    .unwrap()
                    .clone();
                points
                    .iter()
                    .map(|p| {
                        (
                            p["id"].as_str().unwrap().to_string(),
                            p["score"].as_f64().unwrap(),
                        )
                    })
                    .collect()
            };
            let ids = |hits: &[(String, f64)]| hits.iter().map(|h| h.0.clone()).collect::<Vec<_>>();
            let search = "/collections/repo/points/search";
            let query = "/collections/repo/points/query";

            let page = run(
                search,
                serde_json::json!({"vector": {"name": "dense", "vector": [1.0, 0.0, 0.0]}, "limit": 1, "offset": 1}),
            );
            assert_eq!(
                ids(&page),
                vec![format_point_id(2)],
                "search offset lsm={lsm}"
            );
            let thresholded = run(
                search,
                serde_json::json!({"vector": {"name": "dense", "vector": [1.0, 0.0, 0.0]}, "limit": 10, "score_threshold": 0.5}),
            );
            assert_eq!(
                ids(&thresholded),
                vec![format_point_id(1), format_point_id(2)],
                "search threshold lsm={lsm}"
            );
            assert!(thresholded.iter().all(|h| h.1 >= 0.5));

            let page = run(
                query,
                serde_json::json!({"query": [1.0, 0.0, 0.0], "using": "dense", "limit": 1, "offset": 1}),
            );
            assert_eq!(
                ids(&page),
                vec![format_point_id(2)],
                "query offset lsm={lsm}"
            );
            let thresholded = run(
                query,
                serde_json::json!({"query": [1.0, 0.0, 0.0], "using": "dense", "limit": 10, "score_threshold": 0.5}),
            );
            assert_eq!(
                ids(&thresholded),
                vec![format_point_id(1), format_point_id(2)],
                "query threshold lsm={lsm}"
            );

            // Fusion path: threshold/offset apply to the fused score.
            let fused = run(
                query,
                serde_json::json!({
                    "prefetch": [
                        {"using": "dense", "query": [1.0, 0.0, 0.0], "limit": 5},
                        {"using": "dense", "query": [0.9, 0.1, 0.0], "limit": 5}
                    ],
                    "query": {"fusion": "rrf"},
                    "limit": 10
                }),
            );
            assert_eq!(fused.len(), 3, "fusion baseline lsm={lsm}");
            let page = run(
                query,
                serde_json::json!({
                    "prefetch": [
                        {"using": "dense", "query": [1.0, 0.0, 0.0], "limit": 5},
                        {"using": "dense", "query": [0.9, 0.1, 0.0], "limit": 5}
                    ],
                    "query": {"fusion": "rrf"},
                    "limit": 1,
                    "offset": 1
                }),
            );
            assert_eq!(
                ids(&page),
                vec![fused[1].0.clone()],
                "fusion offset lsm={lsm}"
            );
            let cut = fused[1].1;
            let thresholded = run(
                query,
                serde_json::json!({
                    "prefetch": [
                        {"using": "dense", "query": [1.0, 0.0, 0.0], "limit": 5},
                        {"using": "dense", "query": [0.9, 0.1, 0.0], "limit": 5}
                    ],
                    "query": {"fusion": "rrf"},
                    "limit": 10,
                    "score_threshold": cut
                }),
            );
            assert_eq!(
                ids(&thresholded),
                ids(&fused[..2]),
                "fusion threshold lsm={lsm}"
            );
        }
    }

    #[test]
    fn facet_rechecks_should_candidates_against_must_not() {
        let ctx = setup();
        ctx.collections.create_collection("repo").unwrap();

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {
                    "id": 1,
                    "payload": {
                        "language": "rust",
                        "metadata": {"symbol": "HelixGraph", "kind": "struct"}
                    }
                },
                {
                    "id": 2,
                    "payload": {
                        "language": "rust",
                        "metadata": {"symbol": "QueryInput", "kind": "enum"}
                    }
                },
                {
                    "id": 3,
                    "payload": {
                        "language": "python",
                        "metadata": {"symbol": "StorageCore", "kind": "struct"}
                    }
                }
            ]),
        );

        for field in ["language", "metadata.symbol", "metadata.kind"] {
            let create_index_body = sonic_rs::to_vec(&sonic_rs::json!({
                "field_name": field,
                "field_schema": "keyword",
            }))
            .unwrap();
            let create_index_input = make_input(
                &ctx,
                "PUT",
                "/collections/repo/index",
                create_index_body,
                HashMap::from([("name".into(), "repo".into())]),
            );
            let mut create_index_response = Response::new();
            handle_create_index(&create_index_input, &mut create_index_response).unwrap();
            assert_eq!(create_index_response.status, 200);
        }

        let facet_body = sonic_rs::to_vec(&sonic_rs::json!({
            "key": "language",
            "filter": {
                "should": [
                    {"key": "metadata.symbol", "match": {"value": "HelixGraph"}},
                    {"key": "metadata.symbol", "match": {"value": "QueryInput"}}
                ],
                "must_not": [
                    {"key": "metadata.kind", "match": {"value": "enum"}}
                ]
            }
        }))
        .unwrap();
        let facet_input = make_input(
            &ctx,
            "POST",
            "/collections/repo/facet",
            facet_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut facet_response = Response::new();
        handle_facet(&facet_input, &mut facet_response).unwrap();
        assert_eq!(facet_response.status, 200);

        let facet_json: serde_json::Value = serde_json::from_slice(&facet_response.body).unwrap();
        assert_eq!(
            facet_json["result"]["hits"],
            serde_json::json!([{"count": 1, "value": "rust"}])
        );
    }

    #[test]
    fn facet_rechecks_nested_should_candidates_against_must_not() {
        let ctx = setup();
        ctx.collections.create_collection("repo").unwrap();

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {
                    "id": 1,
                    "payload": {
                        "language": "rust",
                        "metadata": {"symbol": "HelixGraph", "kind": "struct"}
                    }
                },
                {
                    "id": 2,
                    "payload": {
                        "language": "rust",
                        "metadata": {"symbol": "QueryInput", "kind": "enum"}
                    }
                }
            ]),
        );

        for field in ["language", "metadata.symbol", "metadata.kind"] {
            let create_index_body = sonic_rs::to_vec(&sonic_rs::json!({
                "field_name": field,
                "field_schema": "keyword",
            }))
            .unwrap();
            let create_index_input = make_input(
                &ctx,
                "PUT",
                "/collections/repo/index",
                create_index_body,
                HashMap::from([("name".into(), "repo".into())]),
            );
            let mut create_index_response = Response::new();
            handle_create_index(&create_index_input, &mut create_index_response).unwrap();
            assert_eq!(create_index_response.status, 200);
        }

        let facet_body = sonic_rs::to_vec(&sonic_rs::json!({
            "key": "language",
            "filter": {
                "must": [{
                    "should": [
                        {"key": "metadata.symbol", "match": {"value": "HelixGraph"}},
                        {"key": "metadata.symbol", "match": {"value": "QueryInput"}}
                    ],
                    "must_not": [
                        {"key": "metadata.kind", "match": {"value": "enum"}}
                    ]
                }]
            }
        }))
        .unwrap();
        let facet_input = make_input(
            &ctx,
            "POST",
            "/collections/repo/facet",
            facet_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut facet_response = Response::new();
        handle_facet(&facet_input, &mut facet_response).unwrap();
        assert_eq!(facet_response.status, 200);

        let facet_json: serde_json::Value = serde_json::from_slice(&facet_response.body).unwrap();
        assert_eq!(
            facet_json["result"]["hits"],
            serde_json::json!([{"count": 1, "value": "rust"}])
        );
    }

    #[test]
    fn facet_exact_true_scans_primary_for_unindexed_filters() {
        let ctx = setup();
        ctx.collections.create_collection("repo").unwrap();

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {"id": 1, "payload": {"language": "rust", "repo": "helix"}},
                {"id": 2, "payload": {"language": "python", "repo": "helix"}},
                {"id": 3, "payload": {"language": "rust", "repo": "context-engine"}}
            ]),
        );

        let create_index_body = sonic_rs::to_vec(&sonic_rs::json!({
            "field_name": "language",
            "field_schema": "keyword",
        }))
        .unwrap();
        let create_index_input = make_input(
            &ctx,
            "PUT",
            "/collections/repo/index",
            create_index_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut create_index_response = Response::new();
        handle_create_index(&create_index_input, &mut create_index_response).unwrap();
        assert_eq!(create_index_response.status, 200);

        let facet_body = sonic_rs::to_vec(&sonic_rs::json!({
            "key": "language",
            "exact": true,
            "filter": {
                "must": [
                    {"key": "repo", "match": {"value": "helix"}}
                ]
            }
        }))
        .unwrap();
        let facet_input = make_input(
            &ctx,
            "POST",
            "/collections/repo/facet",
            facet_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut facet_response = Response::new();
        handle_facet(&facet_input, &mut facet_response).unwrap();
        assert_eq!(facet_response.status, 200);

        let facet_json: serde_json::Value = serde_json::from_slice(&facet_response.body).unwrap();
        assert_eq!(
            facet_json["result"]["hits"],
            serde_json::json!([
                {"count": 1, "value": "python"},
                {"count": 1, "value": "rust"}
            ])
        );
    }

    #[test]
    fn facet_exact_true_counts_duplicate_array_values_once_per_point() {
        let ctx = setup();
        ctx.collections.create_collection("repo").unwrap();

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {"id": 1, "payload": {"tags": ["alpha", "alpha", "beta"]}},
                {"id": 2, "payload": {"tags": ["alpha"]}}
            ]),
        );

        let create_index_body = sonic_rs::to_vec(&sonic_rs::json!({
            "field_name": "tags",
            "field_schema": "keyword",
        }))
        .unwrap();
        let create_index_input = make_input(
            &ctx,
            "PUT",
            "/collections/repo/index",
            create_index_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut create_index_response = Response::new();
        handle_create_index(&create_index_input, &mut create_index_response).unwrap();
        assert_eq!(create_index_response.status, 200);

        let facet_body = sonic_rs::to_vec(&sonic_rs::json!({
            "key": "tags",
            "exact": true
        }))
        .unwrap();
        let facet_input = make_input(
            &ctx,
            "POST",
            "/collections/repo/facet",
            facet_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut facet_response = Response::new();
        handle_facet(&facet_input, &mut facet_response).unwrap();
        assert_eq!(facet_response.status, 200);

        let facet_json: serde_json::Value = serde_json::from_slice(&facet_response.body).unwrap();
        assert_eq!(
            facet_json["result"]["hits"],
            serde_json::json!([
                {"count": 2, "value": "alpha"},
                {"count": 1, "value": "beta"}
            ])
        );
    }

    #[test]
    fn exact_index_candidate_scoring_stays_bounded() {
        fn candidates(count: usize) -> HashSet<u128> {
            (0..count).map(|id| id as u128).collect()
        }

        let exact_cap = 4;
        assert!(should_exact_score_index_candidates_with_limit(
            &candidates(exact_cap),
            0,
            exact_cap
        ));
        assert!(should_exact_score_index_candidates_with_limit(
            &candidates(exact_cap * 8),
            10_000,
            exact_cap
        ));
        assert!(!should_exact_score_index_candidates_with_limit(
            &candidates(exact_cap * 8 + 1),
            10_000_000,
            exact_cap
        ));
        assert!(!should_exact_score_index_candidates_with_limit(
            &candidates(exact_cap + 1),
            50,
            exact_cap
        ));
    }

    #[test]
    fn indexed_filter_exactness_requires_all_clauses_indexed() {
        let ctx = setup();
        let storage = ctx.collections.create_collection("repo").unwrap();
        storage
            .create_payload_index("language", PayloadIndexSchema::Keyword)
            .unwrap();

        fn keyword_condition(key: &str, value: &str) -> Condition {
            FieldCondition {
                key: key.into(),
                match_cond: Some(MatchCondition::Value(MatchValue {
                    value: serde_json::json!(value),
                })),
                range: None,
            }
            .into()
        }

        let indexed_filter = Filter {
            must: vec![keyword_condition("language", "rust")],
            ..Default::default()
        };
        assert!(indexed_filter_candidates_are_exact(
            &storage,
            &indexed_filter
        ));

        let mixed_filter = Filter {
            must: vec![
                keyword_condition("language", "rust"),
                keyword_condition("path", "src/lib.rs"),
            ],
            ..Default::default()
        };
        assert!(!indexed_filter_candidates_are_exact(
            &storage,
            &mixed_filter
        ));
    }

    #[test]
    fn backend_exact_candidate_dense_search_scores_indexed_subset() {
        let ctx = setup();
        create_dense_collection(&ctx, "repo");
        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {"id": 1, "vector": {"dense": [1.0, 0.0, 0.0]}, "payload": {"language": "rust"}},
                {"id": 2, "vector": {"dense": [0.0, 1.0, 0.0]}, "payload": {"language": "python"}},
                {"id": 3, "vector": {"dense": [0.9, 0.1, 0.0]}, "payload": {"language": "rust"}}
            ]),
        );

        let storage = ctx.collections.get_collection("repo").unwrap();
        storage
            .create_payload_index("language", PayloadIndexSchema::Keyword)
            .unwrap();
        wait_for_payload_index_ready(&ctx, "repo", "language");

        let filter = Filter {
            must: vec![FieldCondition {
                key: "language".into(),
                match_cond: Some(MatchCondition::Value(MatchValue {
                    value: serde_json::json!("rust"),
                })),
                range: None,
            }
            .into()],
            ..Default::default()
        };
        let r = storage.backend.begin_read().unwrap();
        let candidates = indexed_filter_candidates_be(&storage, &r, &filter)
            .unwrap()
            .expect("indexed filter must produce candidates");

        let results = storage
            .named_vectors
            .dense_search_candidate_ids_exact_be(
                &r,
                "dense",
                &[1.0, 0.0, 0.0],
                10,
                candidates.iter().copied(),
                Some(candidates.len() as f32 / 3.0),
            )
            .unwrap();

        let ids: Vec<u128> = results.into_iter().map(|hit| hit.id).collect();
        assert_eq!(ids, vec![1, 3]);
    }

    #[test]
    fn indexed_filter_candidates_uses_indexed_should_union() {
        let ctx = setup();
        let storage = ctx.collections.create_collection("repo").unwrap();

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {"id": 1, "payload": {"metadata": {"symbol": "HelixGraph"}}},
                {"id": 2, "payload": {"metadata": {"symbol": "QueryInput"}}},
                {"id": 3, "payload": {"metadata": {"symbol": "StorageCore"}}}
            ]),
        );

        storage
            .create_payload_index("metadata.symbol", PayloadIndexSchema::Keyword)
            .unwrap();

        let filter = Filter {
            should: vec![
                FieldCondition {
                    key: "metadata.symbol".into(),
                    match_cond: Some(MatchCondition::Value(MatchValue {
                        value: serde_json::json!("HelixGraph"),
                    })),
                    range: None,
                }
                .into(),
                FieldCondition {
                    key: "metadata.symbol".into(),
                    match_cond: Some(MatchCondition::Value(MatchValue {
                        value: serde_json::json!("QueryInput"),
                    })),
                    range: None,
                }
                .into(),
            ],
            ..Default::default()
        };

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let candidates = indexed_filter_candidates(&storage, &txn, &filter)
            .unwrap()
            .unwrap();

        assert_eq!(candidates, HashSet::from([1_u128, 2_u128]));
    }

    #[test]
    fn exact_index_filter_candidates_require_no_payload_recheck() {
        let candidates = HashSet::from([1_u128, 2_u128]);

        assert!(exact_index_filter_candidates(true, false, Some(&candidates)).is_some());
        assert!(exact_index_filter_candidates(true, true, Some(&candidates)).is_none());
        assert!(exact_index_filter_candidates(false, false, Some(&candidates)).is_none());
        assert!(exact_index_filter_candidates(true, false, None).is_none());
    }

    #[test]
    fn backend_point_scan_uses_indexed_filter_candidates() {
        let ctx = setup();
        let storage = ctx.collections.create_collection("repo").unwrap();

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {"id": 1, "payload": {"metadata": {"symbol": "HelixGraph"}}},
                {"id": 2, "payload": {"metadata": {"symbol": "QueryInput"}}},
                {"id": 3, "payload": {"metadata": {"symbol": "StorageCore"}}}
            ]),
        );

        storage
            .create_payload_index("metadata.symbol", PayloadIndexSchema::Keyword)
            .unwrap();
        wait_for_payload_index_ready(&ctx, "repo", "metadata.symbol");

        let filter = Filter {
            must: vec![FieldCondition {
                key: "metadata.symbol".into(),
                match_cond: Some(MatchCondition::Value(MatchValue {
                    value: serde_json::json!("QueryInput"),
                })),
                range: None,
            }
            .into()],
            ..Default::default()
        };
        let opts = PointScanOptions {
            collection: "repo",
            limit: 10,
            offset_id: 0,
            filter: &filter,
            with_payload: &WithPayload::All,
            with_vectors: &WithVectors::None,
            partition: None,
            max_scan: None,
            deadline: None,
            filter_hash: scan_filter_hash(&filter),
            cursor_generation: None,
            fuse_pending: false,
        };
        let r = storage.backend.begin_read().unwrap();
        let scan = execute_point_scan_be(&storage, &r, &opts).unwrap();

        assert!(matches!(scan.plan, PointScanPlan::IndexedCandidates));
        assert_eq!(scan.scanned, 1);
        assert_eq!(scan.returned, 1);
        assert_eq!(scan.index_candidates, Some(1));
    }

    /// Scan-budget blind spot: `execute_point_scan_be`'s `IndexedCandidates`
    /// branch only checked `point_scan_budget_exhausted` after a filter
    /// mismatch or a successful push — never on the `NodeNotFound` arm, the
    /// branch every ghost payload-index entry (or merely a sparse/poisoned
    /// filter) hits. A long run of missing candidates walked the whole run
    /// with no page boundary. Simulates the poisoned-index scenario by
    /// injecting raw index dup entries for ids that never became real nodes,
    /// sorted before the one real match, and asserts a tight `max_scan`
    /// budget still terminates the page with a resumable cursor instead of
    /// walking the whole run.
    #[test]
    fn indexed_scan_budget_terminates_on_long_run_of_missing_candidates() {
        let ctx = setup();
        let storage = ctx.collections.create_collection("repo").unwrap();

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {"id": 1000, "payload": {"metadata": {"symbol": "QueryInput"}}}
            ]),
        );

        storage
            .create_payload_index("metadata.symbol", PayloadIndexSchema::Keyword)
            .unwrap();
        wait_for_payload_index_ready(&ctx, "repo", "metadata.symbol");

        // Ghost entries: real dup entries in the index for ids that never
        // became nodes, sorted before id 1000 (execute_point_scan_be sorts
        // candidates ascending), so the scan must walk through all of them
        // before reaching the one real match.
        {
            let mut w = storage.backend.begin_write().unwrap();
            for ghost_id in 1_u128..=100_u128 {
                storage
                    .index_node_payload_field_be(
                        &mut w,
                        "metadata.symbol",
                        &PayloadIndexSchema::Keyword,
                        ghost_id,
                        Some(&Value::String("QueryInput".to_string())),
                    )
                    .unwrap();
            }
            storage.backend.commit(w).unwrap();
        }

        let filter = Filter {
            must: vec![FieldCondition {
                key: "metadata.symbol".into(),
                match_cond: Some(MatchCondition::Value(MatchValue {
                    value: serde_json::json!("QueryInput"),
                })),
                range: None,
            }
            .into()],
            ..Default::default()
        };
        let opts = PointScanOptions {
            collection: "repo",
            limit: 10,
            offset_id: 0,
            filter: &filter,
            with_payload: &WithPayload::All,
            with_vectors: &WithVectors::None,
            partition: None,
            max_scan: Some(10),
            deadline: None,
            filter_hash: scan_filter_hash(&filter),
            cursor_generation: None,
            fuse_pending: false,
        };
        let r = storage.backend.begin_read().unwrap();
        let scan = execute_point_scan_be(&storage, &r, &opts).unwrap();

        assert_eq!(
            scan.index_candidates,
            Some(101),
            "101 candidates: 100 ghosts + 1 real"
        );
        assert!(
            scan.budget_exhausted,
            "must stop once the scan budget is hit rather than walk the whole ghost run"
        );
        assert!(
            scan.next_cursor.is_some(),
            "a budget-exhausted page must carry a resumable next_page_offset"
        );
        assert!(
            scan.scanned <= 11,
            "must not scan far past the budget (max_scan=10, checked after incrementing): scanned={}",
            scan.scanned
        );
        assert!(
            scan.returned == 0,
            "the real match (id 1000) sorts after all 100 ghosts, so a budget of 10 must not reach it"
        );
    }

    #[test]
    fn backend_point_scan_uses_indexed_text_match_candidates() {
        use crate::helix_engine::storage_core::filters::MatchText;

        let ctx = setup();
        let storage = ctx.collections.create_collection("repo").unwrap();

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {"id": 1, "payload": {"metadata": {"symbol": "HelixGraph"}}},
                {"id": 2, "payload": {"metadata": {"symbol": "GraphQueryInput"}}},
                {"id": 3, "payload": {"metadata": {"symbol": "StorageCore"}}}
            ]),
        );

        storage
            .create_payload_index("metadata.symbol", PayloadIndexSchema::Keyword)
            .unwrap();
        wait_for_payload_index_ready(&ctx, "repo", "metadata.symbol");

        let filter = Filter {
            must: vec![FieldCondition {
                key: "metadata.symbol".into(),
                match_cond: Some(MatchCondition::Text(MatchText {
                    text: "graph".into(),
                })),
                range: None,
            }
            .into()],
            ..Default::default()
        };
        let opts = PointScanOptions {
            collection: "repo",
            limit: 10,
            offset_id: 0,
            filter: &filter,
            with_payload: &WithPayload::All,
            with_vectors: &WithVectors::None,
            partition: None,
            max_scan: None,
            deadline: None,
            filter_hash: scan_filter_hash(&filter),
            cursor_generation: None,
            fuse_pending: false,
        };
        let r = storage.backend.begin_read().unwrap();
        let scan = execute_point_scan_be(&storage, &r, &opts).unwrap();

        assert!(matches!(scan.plan, PointScanPlan::IndexedCandidates));
        assert_eq!(scan.scanned, 2);
        assert_eq!(scan.returned, 2);
        assert_eq!(scan.index_candidates, Some(2));

        let exact_filter = Filter {
            must: vec![FieldCondition {
                key: "metadata.symbol".into(),
                match_cond: Some(MatchCondition::Text(MatchText {
                    text: "HelixGraph".into(),
                })),
                range: None,
            }
            .into()],
            ..Default::default()
        };
        let exact_opts = PointScanOptions {
            filter: &exact_filter,
            filter_hash: scan_filter_hash(&exact_filter),
            ..opts
        };
        let exact_scan = execute_point_scan_be(&storage, &r, &exact_opts).unwrap();

        assert!(matches!(exact_scan.plan, PointScanPlan::IndexedCandidates));
        assert_eq!(exact_scan.scanned, 1);
        assert_eq!(exact_scan.returned, 1);
        assert_eq!(exact_scan.index_candidates, Some(1));
    }

    fn broad_keyword_filter(key: &str, value: &str) -> Filter {
        Filter {
            must: vec![FieldCondition {
                key: key.into(),
                match_cond: Some(MatchCondition::Value(MatchValue {
                    value: serde_json::json!(value),
                })),
                range: None,
            }
            .into()],
            ..Default::default()
        }
    }

    #[test]
    fn broad_filter_skips_materialization_results_identical() {
        let ctx = setup();
        create_dense_collection(&ctx, "repo");
        let points: Vec<serde_json::Value> = (0..12_i64)
            .map(|i| {
                let angle = i as f32 * 0.1;
                serde_json::json!({
                    "id": i + 1,
                    "vector": {"dense": [angle.cos(), angle.sin(), 0.0]},
                    "payload": {"broad": "yes"}
                })
            })
            .collect();
        upsert_points(&ctx, "repo", serde_json::Value::Array(points));

        let storage = ctx.collections.get_collection("repo").unwrap();
        storage.refresh_metadata_snapshot().unwrap();
        storage
            .create_payload_index("broad", PayloadIndexSchema::Keyword)
            .unwrap();
        wait_for_payload_index_ready(&ctx, "repo", "broad");

        let filter = broad_keyword_filter("broad", "yes");
        let r = storage.backend.begin_read().unwrap();

        // Probe enabled: every point matches broad=yes, so counting aborts at
        // the broadness cap (12/4 = 3) and materialization is skipped.
        let probed = search_indexed_filter_candidates_be(&storage, &r, &filter, true).unwrap();
        assert!(probed.probe_skipped);
        assert!(probed.candidates.is_none());
        assert_eq!(
            probed.probe_estimate,
            Some(vector_query_broad_candidate_cap(12) + 1)
        );

        // Kill switch: materialize-always restores today's behavior.
        let unprobed = search_indexed_filter_candidates_be(&storage, &r, &filter, false).unwrap();
        assert!(!unprobed.probe_skipped);
        let candidates = unprobed
            .candidates
            .expect("probe disabled must materialize the candidate set");
        assert_eq!(candidates.len(), 12);

        // Replicate the handler's downstream wiring for both paths. Probe-off:
        // all clauses indexed → no recheck → exact scoring over candidates.
        assert!(indexed_filter_candidates_are_exact(&storage, &filter));
        let total_vectors = collection_vector_count_be(&storage);
        let limit = 12;
        let query = [1.0_f32, 0.0, 0.0];
        let exact_candidate_ids = exact_dense_candidate_ids_be(
            &storage,
            &r,
            Some(&candidates),
            &filter,
            false,
            total_vectors,
        )
        .expect("small candidate set must use exact scoring");
        let plan_off = select_dense_search_plan(true, false, unprobed.probe_skipped, true);
        assert!(matches!(plan_off, DenseSearchPlan::ExactCandidates));
        let exact_results = storage
            .named_vectors
            .dense_search_candidate_ids_exact_be(
                &r,
                "dense",
                &query,
                limit,
                exact_candidate_ids.iter().copied(),
                None,
            )
            .unwrap();

        // Probe-on: no candidate set → in-traversal filtered HNSW with the
        // matches_point recheck.
        let filter_fn = |id: u128| -> bool {
            match storage.get_node_be(&r, &id) {
                Ok(node) => filter.matches_point(Some(id), &node.properties),
                Err(_) => false,
            }
        };
        let plan_on = select_dense_search_plan(false, false, probed.probe_skipped, true);
        assert!(matches!(plan_on, DenseSearchPlan::ProbeSkippedIndex));
        let hnsw_results = storage
            .named_vectors
            .dense_search_with_id_filter_ef_be(
                &r,
                "dense",
                &query,
                limit,
                Some(&[filter_fn][..]),
                true,
                None,
                None,
            )
            .unwrap();

        let exact_ids: Vec<u128> = exact_results.iter().map(|hit| hit.id).collect();
        let hnsw_ids: Vec<u128> = hnsw_results.iter().map(|hit| hit.id).collect();
        assert_eq!(
            exact_ids, hnsw_ids,
            "both plans must return the same points"
        );
        for (exact, hnsw) in exact_results.iter().zip(hnsw_results.iter()) {
            let a = exact.distance.unwrap_or(0.0);
            let b = hnsw.distance.unwrap_or(0.0);
            assert!(
                (a - b).abs() < 1e-5,
                "score mismatch for {}: {} vs {}",
                exact.id,
                a,
                b
            );
        }
    }

    #[test]
    fn narrow_filter_still_exact() {
        let ctx = setup();
        create_dense_collection(&ctx, "repo");
        let mut points: Vec<serde_json::Value> = vec![
            serde_json::json!({"id": 1, "vector": {"dense": [1.0, 0.0, 0.0]}, "payload": {"language": "rust"}}),
            serde_json::json!({"id": 2, "vector": {"dense": [0.9, 0.1, 0.0]}, "payload": {"language": "rust"}}),
        ];
        for id in 3..=12_i64 {
            points.push(serde_json::json!({
                "id": id,
                "vector": {"dense": [0.0, 1.0, (id as f32) * 0.01]},
                "payload": {"language": "python"}
            }));
        }
        upsert_points(&ctx, "repo", serde_json::Value::Array(points));

        let storage = ctx.collections.get_collection("repo").unwrap();
        storage.refresh_metadata_snapshot().unwrap();
        storage
            .create_payload_index("language", PayloadIndexSchema::Keyword)
            .unwrap();
        wait_for_payload_index_ready(&ctx, "repo", "language");

        let filter = broad_keyword_filter("language", "rust");
        let r = storage.backend.begin_read().unwrap();

        let probed = search_indexed_filter_candidates_be(&storage, &r, &filter, true).unwrap();
        assert!(!probed.probe_skipped);
        assert_eq!(probed.probe_estimate, Some(2));
        let candidates = probed
            .candidates
            .expect("selective filter must still materialize candidates");
        assert_eq!(candidates.len(), 2);

        let total_vectors = collection_vector_count_be(&storage);
        let exact_candidate_ids = exact_dense_candidate_ids_be(
            &storage,
            &r,
            Some(&candidates),
            &filter,
            false,
            total_vectors,
        )
        .expect("selective candidates must use exact scoring");
        let plan = select_dense_search_plan(true, false, probed.probe_skipped, true);
        assert!(matches!(plan, DenseSearchPlan::ExactCandidates));

        // Brute-force reference: id 1 is exactly the query direction, id 2 is
        // the only other rust point.
        let results = storage
            .named_vectors
            .dense_search_candidate_ids_exact_be(
                &r,
                "dense",
                &[1.0, 0.0, 0.0],
                10,
                exact_candidate_ids.iter().copied(),
                None,
            )
            .unwrap();
        let ids: Vec<u128> = results.into_iter().map(|hit| hit.id).collect();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn must_not_only_filter_not_probed() {
        let ctx = setup();
        create_dense_collection(&ctx, "repo");
        let points: Vec<serde_json::Value> = (1..=4_i64)
            .map(|id| {
                serde_json::json!({
                    "id": id,
                    "vector": {"dense": [id as f32, 1.0, 0.0]},
                    "payload": {"broad": "yes"}
                })
            })
            .collect();
        upsert_points(&ctx, "repo", serde_json::Value::Array(points));

        let storage = ctx.collections.get_collection("repo").unwrap();
        storage.refresh_metadata_snapshot().unwrap();
        storage
            .create_payload_index("broad", PayloadIndexSchema::Keyword)
            .unwrap();
        wait_for_payload_index_ready(&ctx, "repo", "broad");

        let filter = Filter {
            must_not: vec![FieldCondition {
                key: "broad".into(),
                match_cond: Some(MatchCondition::Value(MatchValue {
                    value: serde_json::json!("yes"),
                })),
                range: None,
            }
            .into()],
            ..Default::default()
        };
        let r = storage.backend.begin_read().unwrap();

        // Only `must` conditions are probed; a must_not-only filter yields no
        // estimate and the probe abstains.
        let estimate = indexed_filter_must_count_estimate_be(&storage, &r, &filter, 100).unwrap();
        assert!(estimate.is_none());

        let probed = search_indexed_filter_candidates_be(&storage, &r, &filter, true).unwrap();
        let unprobed = search_indexed_filter_candidates_be(&storage, &r, &filter, false).unwrap();
        assert!(!probed.probe_skipped);
        assert!(probed.probe_estimate.is_none());
        // must_not-only filters produce no positive candidate set today either;
        // behavior is unchanged with the probe enabled.
        assert!(probed.candidates.is_none());
        assert!(unprobed.candidates.is_none());
    }

    #[test]
    fn lsm_query_broad_filter_skips_materialization_and_still_filters() {
        let ctx = setup_with_config(Config::default().with_lsm_in_memory());
        create_collection_with_sparse(&ctx, "query_broad_lsm");

        let points: Vec<serde_json::Value> = (0..12_i64)
            .map(|i| {
                let angle = i as f32 * 0.1;
                serde_json::json!({
                    "id": i + 1,
                    "vector": {
                        "dense": [angle.cos(), angle.sin(), 0.0],
                        "lex_sparse": {"indices": [7], "values": [1.0]}
                    },
                    "payload": {"scope": "all"}
                })
            })
            .collect();
        upsert_mixed_points(&ctx, "query_broad_lsm", serde_json::Value::Array(points));

        let storage = ctx.collections.get_collection("query_broad_lsm").unwrap();
        storage.refresh_metadata_snapshot().unwrap();
        storage
            .create_payload_index("scope", PayloadIndexSchema::Keyword)
            .unwrap();
        wait_for_payload_index_ready(&ctx, "query_broad_lsm", "scope");

        let filter = broad_keyword_filter("scope", "all");
        let r = storage.backend.begin_read().unwrap();
        let probed =
            vector_query_indexed_filter_candidates_be_probed(&storage, &r, &filter, true).unwrap();
        assert!(probed.probe_skipped);
        assert!(probed.candidates.is_none());
        assert_eq!(
            probed.probe_estimate,
            Some(vector_query_broad_candidate_cap(12) + 1)
        );

        let (status, body) = query_points_raw(
            &ctx,
            "query_broad_lsm",
            serde_json::json!({
                "query": {"indices": [7], "values": [1.0]},
                "using": "lex_sparse",
                "limit": 5,
                "with_payload": true,
                "filter": {"must": [{"key": "scope", "match": {"value": "all"}}]}
            }),
        );
        assert_eq!(status, 200, "query failed: {body}");
        let points = body["result"]["points"].as_array().unwrap();
        assert_eq!(points.len(), 5, "query should still return filtered hits");
        assert!(points
            .iter()
            .all(|point| point["payload"]["scope"] == serde_json::json!("all")));
    }

    #[test]
    fn lsm_hybrid_channel_filter_rechecks_when_probe_skips_candidates() {
        let ctx = setup_with_config(Config::default().with_lsm_in_memory());
        create_collection_with_sparse(&ctx, "hybrid_channel_lsm");

        let mut points = Vec::new();
        for id in 1..=12_i64 {
            let scope = if id <= 2 { "top" } else { "channel" };
            points.push(serde_json::json!({
                "id": id,
                "vector": {
                    "dense": [1.0 - (id as f32 * 0.01), id as f32 * 0.01, 0.0],
                    "lex_sparse": {"indices": [7], "values": [1.0]}
                },
                "payload": {"scope": scope}
            }));
        }
        upsert_mixed_points(&ctx, "hybrid_channel_lsm", serde_json::Value::Array(points));

        let storage = ctx
            .collections
            .get_collection("hybrid_channel_lsm")
            .unwrap();
        storage.refresh_metadata_snapshot().unwrap();
        storage
            .create_payload_index("scope", PayloadIndexSchema::Keyword)
            .unwrap();
        wait_for_payload_index_ready(&ctx, "hybrid_channel_lsm", "scope");

        let channel_filter = broad_keyword_filter("scope", "channel");
        let r = storage.backend.begin_read().unwrap();
        let probed =
            vector_query_indexed_filter_candidates_be_probed(&storage, &r, &channel_filter, true)
                .unwrap();
        assert!(probed.probe_skipped);
        assert!(probed.candidates.is_none());

        let top_filter = broad_keyword_filter("scope", "top");
        let narrow =
            vector_query_indexed_filter_candidates_be_probed(&storage, &r, &top_filter, true)
                .unwrap();
        assert!(!narrow.probe_skipped);
        assert_eq!(
            narrow
                .candidates
                .expect("narrow filter should materialize")
                .len(),
            2
        );

        let (status, body) = hybrid_query_points_raw(
            &ctx,
            "hybrid_channel_lsm",
            serde_json::json!({
                "dense": [{
                    "using": "dense",
                    "query": [1.0, 0.0, 0.0],
                    "limit": 5,
                    "filter": {"must": [{"key": "scope", "match": {"value": "channel"}}]}
                }],
                "sparse": [{
                    "using": "lex_sparse",
                    "query": {"indices": [7], "values": [1.0]},
                    "limit": 5,
                    "filter": {"must": [{"key": "scope", "match": {"value": "channel"}}]}
                }],
                "filter": {"must": [{"key": "scope", "match": {"value": "top"}}]},
                "limit": 5,
                "with_payload": true,
                "with_channel_points": true
            }),
        );
        assert_eq!(status, 200, "hybrid query failed: {body}");
        let fused_points = body["result"]["points"].as_array().unwrap();
        assert!(
            !fused_points.is_empty(),
            "hybrid query should return channel-filtered hits: {body}"
        );
        for point in fused_points {
            assert_eq!(point["payload"]["scope"], serde_json::json!("channel"));
        }
        for group in ["dense", "sparse"] {
            for channel in body["result"]["channels"][group].as_array().unwrap() {
                for point in channel["points"].as_array().unwrap() {
                    assert_eq!(point["payload"]["scope"], serde_json::json!("channel"));
                }
            }
        }
    }

    #[test]
    fn filter_field_keys_reports_nested_fields_for_runaway_scan_logs() {
        let filter = Filter {
            must: vec![Condition::Nested(Filter {
                should: vec![
                    FieldCondition {
                        key: "metadata.symbol".into(),
                        match_cond: Some(MatchCondition::Value(MatchValue {
                            value: serde_json::json!("HelixGraph"),
                        })),
                        range: None,
                    }
                    .into(),
                    FieldCondition {
                        key: "metadata.kind".into(),
                        match_cond: Some(MatchCondition::Value(MatchValue {
                            value: serde_json::json!("struct"),
                        })),
                        range: None,
                    }
                    .into(),
                ],
                ..Default::default()
            })],
            must_not: vec![Condition::HasId(HasIdCondition {
                has_id: vec![serde_json::json!(7)],
            })],
            ..Default::default()
        };

        assert_eq!(filter_field_keys(&filter), "metadata.kind,metadata.symbol");
    }

    #[test]
    fn indexed_filter_candidates_recurses_nested_filters() {
        let ctx = setup();
        let storage = ctx.collections.create_collection("repo").unwrap();

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {"id": 1, "payload": {"metadata": {"symbol": "HelixGraph", "kind": "struct"}}},
                {"id": 2, "payload": {"metadata": {"symbol": "QueryInput", "kind": "enum"}}},
                {"id": 3, "payload": {"metadata": {"symbol": "StorageCore", "kind": "struct"}}}
            ]),
        );

        for field in ["metadata.symbol", "metadata.kind"] {
            storage
                .create_payload_index(field, PayloadIndexSchema::Keyword)
                .unwrap();
        }

        let filter = Filter {
            must: vec![
                Condition::Nested(Filter {
                    should: vec![
                        FieldCondition {
                            key: "metadata.symbol".into(),
                            match_cond: Some(MatchCondition::Value(MatchValue {
                                value: serde_json::json!("HelixGraph"),
                            })),
                            range: None,
                        }
                        .into(),
                        FieldCondition {
                            key: "metadata.symbol".into(),
                            match_cond: Some(MatchCondition::Value(MatchValue {
                                value: serde_json::json!("QueryInput"),
                            })),
                            range: None,
                        }
                        .into(),
                    ],
                    ..Default::default()
                }),
                FieldCondition {
                    key: "metadata.kind".into(),
                    match_cond: Some(MatchCondition::Value(MatchValue {
                        value: serde_json::json!("struct"),
                    })),
                    range: None,
                }
                .into(),
            ],
            ..Default::default()
        };

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let candidates = indexed_filter_candidates(&storage, &txn, &filter)
            .unwrap()
            .unwrap();

        assert_eq!(candidates, HashSet::from([1_u128]));
    }

    #[test]
    fn indexed_filter_candidates_skips_partially_unindexed_should() {
        let ctx = setup();
        let storage = ctx.collections.create_collection("repo").unwrap();

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {"id": 1, "payload": {"metadata": {"symbol": "HelixGraph", "kind": "struct"}}},
                {"id": 2, "payload": {"metadata": {"symbol": "QueryInput", "kind": "enum"}}},
                {"id": 3, "payload": {"metadata": {"symbol": "StorageCore", "kind": "struct"}}}
            ]),
        );

        storage
            .create_payload_index("metadata.symbol", PayloadIndexSchema::Keyword)
            .unwrap();

        let filter = Filter {
            should: vec![
                FieldCondition {
                    key: "metadata.symbol".into(),
                    match_cond: Some(MatchCondition::Value(MatchValue {
                        value: serde_json::json!("HelixGraph"),
                    })),
                    range: None,
                }
                .into(),
                FieldCondition {
                    key: "metadata.kind".into(),
                    match_cond: Some(MatchCondition::Value(MatchValue {
                        value: serde_json::json!("struct"),
                    })),
                    range: None,
                }
                .into(),
            ],
            ..Default::default()
        };

        let txn = storage.lmdb_env().unwrap().read_txn().unwrap();
        let candidates = indexed_filter_candidates(&storage, &txn, &filter).unwrap();

        assert!(candidates.is_none());
    }

    #[test]
    fn create_keyword_index_backfills_nested_payload_keys_for_scroll() {
        let ctx = setup();
        ctx.collections.create_collection("repo").unwrap();

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {
                    "id": "doc-1",
                    "payload": {
                        "metadata": {
                            "symbol": "HelixGraphEngine",
                            "kind": "struct"
                        }
                    }
                },
                {
                    "id": "doc-2",
                    "payload": {
                        "metadata": {
                            "symbol": "QueryInput",
                            "kind": "enum"
                        }
                    }
                }
            ]),
        );

        let create_index_body = sonic_rs::to_vec(&sonic_rs::json!({
            "field_name": "metadata.symbol",
            "field_schema": "keyword",
        }))
        .unwrap();
        let create_index_input = make_input(
            &ctx,
            "PUT",
            "/collections/repo/index",
            create_index_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut create_index_response = Response::new();
        handle_create_index(&create_index_input, &mut create_index_response).unwrap();
        assert_eq!(create_index_response.status, 200);

        let scroll_body = sonic_rs::to_vec(&sonic_rs::json!({
            "limit": 10,
            "with_payload": true,
            "filter": {
                "must": [{
                    "key": "metadata.symbol",
                    "match": {"value": "HelixGraphEngine"}
                }]
            }
        }))
        .unwrap();
        let scroll_input = make_input(
            &ctx,
            "POST",
            "/collections/repo/points/scroll",
            scroll_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut scroll_response = Response::new();
        handle_scroll_points(&scroll_input, &mut scroll_response).unwrap();
        assert_eq!(scroll_response.status, 200);

        let scroll_json: serde_json::Value = serde_json::from_slice(&scroll_response.body).unwrap();
        let points = scroll_json["result"]["points"].as_array().unwrap();
        assert_eq!(points.len(), 1);
        assert_eq!(
            points[0]["payload"]["metadata"]["symbol"],
            serde_json::json!("HelixGraphEngine")
        );
    }

    #[test]
    fn delete_payload_index_allows_recreate_for_nested_payload_keys() {
        let ctx = setup();
        ctx.collections.create_collection("repo").unwrap();

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {
                    "id": "doc-1",
                    "payload": {
                        "metadata": {
                            "symbol": "HelixGraphEngine"
                        }
                    }
                }
            ]),
        );

        let create_index_body = sonic_rs::to_vec(&sonic_rs::json!({
            "field_name": "metadata.symbol",
            "field_schema": "keyword",
        }))
        .unwrap();
        let create_index_input = make_input(
            &ctx,
            "PUT",
            "/collections/repo/index",
            create_index_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut create_index_response = Response::new();
        handle_create_index(&create_index_input, &mut create_index_response).unwrap();
        assert_eq!(create_index_response.status, 200);

        let delete_index_input = make_input(
            &ctx,
            "DELETE",
            "/collections/repo/index/metadata.symbol",
            Vec::new(),
            HashMap::from([
                ("name".into(), "repo".into()),
                ("field_name".into(), "metadata.symbol".into()),
            ]),
        );
        let mut delete_index_response = Response::new();
        handle_delete_index(&delete_index_input, &mut delete_index_response).unwrap();
        assert_eq!(delete_index_response.status, 200);

        let create_index_input_2 = make_input(
            &ctx,
            "PUT",
            "/collections/repo/index",
            sonic_rs::to_vec(&sonic_rs::json!({
                "field_name": "metadata.symbol",
                "field_schema": "keyword",
            }))
            .unwrap(),
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut create_index_response_2 = Response::new();
        handle_create_index(&create_index_input_2, &mut create_index_response_2).unwrap();
        assert_eq!(create_index_response_2.status, 200);

        let scroll_body = sonic_rs::to_vec(&sonic_rs::json!({
            "limit": 10,
            "with_payload": true,
            "filter": {
                "must": [{
                    "key": "metadata.symbol",
                    "match": {"value": "HelixGraphEngine"}
                }]
            }
        }))
        .unwrap();
        let scroll_input = make_input(
            &ctx,
            "POST",
            "/collections/repo/points/scroll",
            scroll_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut scroll_response = Response::new();
        handle_scroll_points(&scroll_input, &mut scroll_response).unwrap();
        assert_eq!(scroll_response.status, 200);
        let scroll_json: serde_json::Value = serde_json::from_slice(&scroll_response.body).unwrap();
        let points = scroll_json["result"]["points"].as_array().unwrap();
        assert_eq!(points.len(), 1);
    }

    #[test]
    fn search_and_query_respect_indexed_keyword_filters() {
        let ctx = setup();
        create_dense_collection(&ctx, "repo");

        upsert_points(
            &ctx,
            "repo",
            serde_json::json!([
                {
                    "id": "doc-1",
                    "vector": {"dense": [1.0, 0.0, 0.0]},
                    "payload": {
                        "git_branches": ["main"],
                        "path": "src/main.rs"
                    }
                },
                {
                    "id": "doc-2",
                    "vector": {"dense": [0.0, 1.0, 0.0]},
                    "payload": {
                        "git_branches": ["release"],
                        "path": "src/lib.rs"
                    }
                },
                {
                    "id": "doc-3",
                    "vector": {"dense": [0.9, 0.1, 0.0]},
                    "payload": {
                        "git_branches": ["main", "feature/auth"],
                        "path": "src/auth.rs"
                    }
                }
            ]),
        );

        let create_index_body = sonic_rs::to_vec(&sonic_rs::json!({
            "field_name": "git_branches",
            "field_schema": "keyword",
        }))
        .unwrap();
        let create_index_input = make_input(
            &ctx,
            "PUT",
            "/collections/repo/index",
            create_index_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut create_index_response = Response::new();
        handle_create_index(&create_index_input, &mut create_index_response).unwrap();
        assert_eq!(create_index_response.status, 200);

        let search_body = sonic_rs::to_vec(&sonic_rs::json!({
            "vector": {"name": "dense", "vector": [1.0, 0.0, 0.0]},
            "limit": 5,
            "with_payload": true,
            "filter": {
                "must": [{
                    "key": "git_branches",
                    "match": {"value": "main"}
                }]
            }
        }))
        .unwrap();
        let search_input = make_input(
            &ctx,
            "POST",
            "/collections/repo/points/search",
            search_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut search_response = Response::new();
        handle_search_points(&search_input, &mut search_response).unwrap();
        assert_eq!(search_response.status, 200);

        let search_json: serde_json::Value = serde_json::from_slice(&search_response.body).unwrap();
        let search_points = search_json["result"].as_array().unwrap();
        assert_eq!(search_points.len(), 2);
        assert!(search_points
            .iter()
            .all(|point| point["payload"]["git_branches"]
                .to_string()
                .contains("main")));

        let query_body = sonic_rs::to_vec(&sonic_rs::json!({
            "prefetch": [{
                "using": "dense",
                "query": [1.0, 0.0, 0.0],
                "limit": 5
            }],
            "limit": 5,
            "with_payload": true,
            "filter": {
                "must": [{
                    "key": "git_branches",
                    "match": {"value": "main"}
                }]
            }
        }))
        .unwrap();
        let query_input = make_input(
            &ctx,
            "POST",
            "/collections/repo/points/query",
            query_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut query_response = Response::new();
        handle_query_points(&query_input, &mut query_response).unwrap();
        assert_eq!(query_response.status, 200);

        let query_json: serde_json::Value = serde_json::from_slice(&query_response.body).unwrap();
        let query_points = query_json["result"]["points"].as_array().unwrap();
        assert_eq!(query_points.len(), 2);
        assert!(query_points
            .iter()
            .all(|point| point["payload"]["git_branches"]
                .to_string()
                .contains("main")));
    }

    #[test]
    fn snapshot_handlers_create_list_and_recover_collection() {
        let ctx = setup();
        let storage = ctx.collections.create_collection("repo").unwrap();

        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        storage
            .create_node(
                &mut txn,
                "Symbol",
                vec![("name".into(), Value::String("main".into()))],
                None,
                None,
            )
            .unwrap();
        txn.commit().unwrap();
        drop(storage);

        let create_input = make_input(
            &ctx,
            "POST",
            "/collections/repo/snapshots",
            Vec::new(),
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut create_response = Response::new();
        handle_create_snapshot(&create_input, &mut create_response).unwrap();
        assert_eq!(create_response.status, 200);

        let create_json: serde_json::Value = serde_json::from_slice(&create_response.body).unwrap();
        assert_eq!(
            create_json["result"]["location"],
            create_json["result"]["name"]
        );
        let snapshot_location = create_json["result"]["location"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(!snapshot_location.contains(std::path::MAIN_SEPARATOR));

        // Wait for background copy to complete so the snapshot file exists
        // for the recovery step below. Poll the snapshot list API until we
        // see a non-zero disk_bytes for the snapshot we just created.
        use std::time::{Duration, Instant};
        let started = Instant::now();
        loop {
            let list_resp = make_input(
                &ctx,
                "GET",
                "/collections/repo/snapshots",
                Vec::new(),
                HashMap::from([("name".into(), "repo".into())]),
            );
            let mut lr = Response::new();
            handle_list_snapshots(&list_resp, &mut lr).unwrap();
            let list_json: serde_json::Value =
                serde_json::from_slice(&lr.body).unwrap_or(serde_json::Value::Null);
            let disks: Vec<u64> = list_json["result"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .map(|v| v["size"].as_u64().unwrap_or(0))
                        .collect()
                })
                .unwrap_or_default();
            if disks.iter().any(|&b| b > 0) {
                break;
            }
            if started.elapsed() > Duration::from_secs(5) {
                panic!("Snapshot background copy did not complete within 5s");
            }
            std::thread::sleep(Duration::from_millis(50));
        }

        let storage = ctx.collections.get_collection("repo").unwrap();
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        storage
            .create_node(
                &mut txn,
                "Symbol",
                vec![("name".into(), Value::String("helper".into()))],
                None,
                None,
            )
            .unwrap();
        txn.commit().unwrap();
        drop(storage);

        let list_input = make_input(
            &ctx,
            "GET",
            "/collections/repo/snapshots",
            Vec::new(),
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut list_response = Response::new();
        handle_list_snapshots(&list_input, &mut list_response).unwrap();
        assert_eq!(list_response.status, 200);
        let list_json: serde_json::Value = serde_json::from_slice(&list_response.body).unwrap();
        assert_eq!(list_json["result"].as_array().unwrap().len(), 1);

        let recover_body = sonic_rs::to_vec(&sonic_rs::json!({
            "location": snapshot_location,
        }))
        .unwrap();
        let recover_input = make_input(
            &ctx,
            "PUT",
            "/collections/repo/snapshots/recover",
            recover_body,
            HashMap::from([("name".into(), "repo".into())]),
        );
        let mut recover_response = Response::new();
        handle_recover_snapshot(&recover_input, &mut recover_response).unwrap();
        assert_eq!(recover_response.status, 200);

        let stats = ctx.collections.collection_stats("repo").unwrap();
        assert_eq!(stats.node_count, 1);
    }

    // ── Helpers for sparse vector tests ──

    fn create_collection_with_sparse(ctx: &TestContext, name: &str) {
        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "vectors": {
                "dense": {"size": 3, "distance": "Cosine"}
            },
            "sparse_vectors": {
                "lex_sparse": {"modifier": "idf"}
            }
        }))
        .unwrap();
        let input = make_input(
            ctx,
            "PUT",
            &format!("/collections/{}", name),
            body,
            HashMap::from([("name".into(), name.into())]),
        );
        let mut resp = Response::new();
        handle_create_collection(&input, &mut resp).unwrap();
        assert!(
            resp.status == 200 || resp.status == 201,
            "status={}",
            resp.status
        );
    }

    pub(super) fn upsert_mixed_points(ctx: &TestContext, name: &str, points: serde_json::Value) {
        let body = sonic_rs::to_vec(&sonic_rs::json!({ "points": points })).unwrap();
        let input = make_input(
            ctx,
            "PUT",
            &format!("/collections/{}/points", name),
            body,
            HashMap::from([("name".into(), name.into())]),
        );
        let mut resp = Response::new();
        handle_upsert_points(&input, &mut resp).unwrap();
        assert_eq!(
            resp.status,
            200,
            "upsert body: {}",
            String::from_utf8_lossy(&resp.body)
        );
    }

    fn query_points_raw(
        ctx: &TestContext,
        name: &str,
        body: serde_json::Value,
    ) -> (u16, serde_json::Value) {
        let body_bytes = sonic_rs::to_vec(&body).unwrap();
        let input = make_input(
            ctx,
            "POST",
            &format!("/collections/{}/points/query", name),
            body_bytes,
            HashMap::from([("name".into(), name.into())]),
        );
        let mut resp = Response::new();
        handle_query_points(&input, &mut resp).unwrap();
        let json: serde_json::Value =
            serde_json::from_slice(&resp.body).unwrap_or(serde_json::Value::Null);
        (resp.status, json)
    }

    pub(super) fn hybrid_query_points_raw(
        ctx: &TestContext,
        name: &str,
        body: serde_json::Value,
    ) -> (u16, serde_json::Value) {
        let body_bytes = sonic_rs::to_vec(&body).unwrap();
        let input = make_input(
            ctx,
            "POST",
            &format!("/collections/{}/points/hybrid_query", name),
            body_bytes,
            HashMap::from([("name".into(), name.into())]),
        );
        let mut resp = Response::new();
        handle_hybrid_query_points(&input, &mut resp).unwrap();
        let json: serde_json::Value =
            serde_json::from_slice(&resp.body).unwrap_or(serde_json::Value::Null);
        (resp.status, json)
    }

    // ── Tests ──

    #[test]
    fn create_collection_with_sparse_vectors() {
        let ctx = setup();
        create_collection_with_sparse(&ctx, "sparse_test");

        // Verify via get_collection that sparse is reported.
        let input = make_input(
            &ctx,
            "GET",
            "/collections/sparse_test",
            vec![],
            HashMap::from([("name".into(), "sparse_test".into())]),
        );
        let mut resp = Response::new();
        handle_get_collection(&input, &mut resp).unwrap();
        assert_eq!(resp.status, 200);
        let body: serde_json::Value = serde_json::from_slice(&resp.body).unwrap();
        let result = &body["result"];
        assert!(result["config"]["params"]["sparse_vectors"]["lex_sparse"].is_object());
        assert_eq!(
            result["config"]["params"]["sparse_vectors"]["lex_sparse"]["modifier"],
            "idf"
        );
    }

    #[test]
    fn create_index_accepts_object_field_schema() {
        let ctx = setup();
        create_dense_collection(&ctx, "idxobj");

        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "field_name": "path",
            "field_schema": {"type": "keyword"},
        }))
        .unwrap();
        let input = make_input(
            &ctx,
            "PUT",
            "/collections/idxobj/index",
            body,
            HashMap::from([("name".into(), "idxobj".into())]),
        );
        let mut resp = Response::new();
        handle_create_index(&input, &mut resp).unwrap();
        assert_eq!(
            resp.status,
            200,
            "body: {}",
            String::from_utf8_lossy(&resp.body)
        );

        let get_input = make_input(
            &ctx,
            "GET",
            "/collections/idxobj",
            vec![],
            HashMap::from([("name".into(), "idxobj".into())]),
        );
        let mut get_resp = Response::new();
        handle_get_collection(&get_input, &mut get_resp).unwrap();
        assert_eq!(get_resp.status, 200);
        let body: serde_json::Value = serde_json::from_slice(&get_resp.body).unwrap();
        assert_eq!(
            body["result"]["payload_schema"]["path"]["data_type"],
            "keyword"
        );
        assert_eq!(body["result"]["payload_schema"]["path"]["points"], 0);
    }

    #[test]
    fn upsert_and_query_sparse_vectors() {
        let ctx = setup();
        create_collection_with_sparse(&ctx, "sq");

        // Upsert points with both dense + sparse vectors.
        upsert_mixed_points(
            &ctx,
            "sq",
            serde_json::json!([
                {
                    "id": "doc-1",
                    "vector": {
                        "dense": [1.0, 0.0, 0.0],
                        "lex_sparse": {"indices": [10, 20], "values": [1.0, 2.0]}
                    },
                    "payload": {"topic": "rust"}
                },
                {
                    "id": "doc-2",
                    "vector": {
                        "dense": [0.0, 1.0, 0.0],
                        "lex_sparse": {"indices": [20, 30], "values": [5.0, 1.0]}
                    },
                    "payload": {"topic": "python"}
                }
            ]),
        );

        // Single-vector sparse query on term 20.
        let (status, body) = query_points_raw(
            &ctx,
            "sq",
            serde_json::json!({
                "query": {"indices": [20], "values": [1.0]},
                "using": "lex_sparse",
                "limit": 10,
                "with_payload": true
            }),
        );
        assert_eq!(status, 200);
        let results = body["result"]["points"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        // doc-2 has value 5.0 on term 20 (higher than doc-1's 2.0 after IDF).
        let first_id = results[0]["id"].as_str().unwrap();
        assert!(
            first_id.contains(&format!(
                "{:032x}",
                point_id_to_u128(&serde_json::Value::String("doc-2".into()))
            )) || results[0]["payload"]["topic"] == "python",
            "Expected doc-2 first, got: {:?}",
            results[0]
        );
    }

    #[test]
    fn sparse_query_with_indexed_filter_scores_candidate_ids() {
        let ctx = setup();
        create_collection_with_sparse(&ctx, "sq_filter");

        upsert_mixed_points(
            &ctx,
            "sq_filter",
            serde_json::json!([
                {
                    "id": "doc-rust-a",
                    "vector": {
                        "dense": [1.0, 0.0, 0.0],
                        "lex_sparse": {"indices": [20], "values": [2.0]}
                    },
                    "payload": {"topic": "rust"}
                },
                {
                    "id": "doc-python",
                    "vector": {
                        "dense": [0.0, 1.0, 0.0],
                        "lex_sparse": {"indices": [20], "values": [10.0]}
                    },
                    "payload": {"topic": "python"}
                },
                {
                    "id": "doc-rust-b",
                    "vector": {
                        "dense": [0.0, 0.0, 1.0],
                        "lex_sparse": {"indices": [20], "values": [1.0]}
                    },
                    "payload": {"topic": "rust"}
                }
            ]),
        );

        let storage = ctx.collections.get_collection("sq_filter").unwrap();
        storage
            .create_payload_index("topic", PayloadIndexSchema::Keyword)
            .unwrap();
        wait_for_payload_index_ready(&ctx, "sq_filter", "topic");

        let (status, body) = query_points_raw(
            &ctx,
            "sq_filter",
            serde_json::json!({
                "query": {"indices": [20], "values": [1.0]},
                "using": "lex_sparse",
                "limit": 10,
                "with_payload": true,
                "filter": {
                    "must": [{
                        "key": "topic",
                        "match": {"value": "rust"}
                    }]
                }
            }),
        );
        assert_eq!(status, 200, "body: {body}");
        let results = body["result"]["points"].as_array().unwrap();
        assert_eq!(results.len(), 2, "body: {body}");
        assert!(results
            .iter()
            .all(|point| point["payload"]["topic"] == "rust"));
        assert_eq!(results[0]["payload"]["topic"], "rust");
    }

    /// Sparse-vector export round-trip: a point upserted with a sparse vector
    /// must come back through BOTH scroll and get when `with_vector` is set.
    /// This is the migration bug — before `get_doc_vector` + the
    /// `fetch_named_vectors` sparse branch + sparse names in
    /// `known_vector_names`, scroll/get hydrated only dense vectors, so a
    /// scroll->upsert migration silently dropped `lex_sparse`. Runs on the
    /// default LMDB backend: the sparse write path (`put_heed`) is
    /// `unreachable!()` on the in-memory LSM backend, so a full sparse upsert
    /// cannot execute there (a separate known issue). The export read path
    /// itself is backend-agnostic — it routes through `get_with_heed`, the same
    /// seam LSM reads use.
    #[test]
    fn scroll_and_get_export_sparse_vectors() {
        let ctx = setup();
        create_collection_with_sparse(&ctx, "sparse_export");

        upsert_mixed_points(
            &ctx,
            "sparse_export",
            serde_json::json!([{
                "id": "doc-1",
                "vector": {
                    "dense": [1.0, 0.0, 0.0],
                    "lex_sparse": {"indices": [10, 20, 30], "values": [1.5, 2.5, 3.5]}
                },
                "payload": {"topic": "rust"}
            }]),
        );

        // Order-robust comparison: the stored forward index may reorder terms.
        let expected: Vec<(u64, String)> =
            vec![(10, "1.5".into()), (20, "2.5".into()), (30, "3.5".into())];
        let extract_sparse = |point: &serde_json::Value| -> Vec<(u64, String)> {
            let sparse = &point["vector"]["lex_sparse"];
            let indices = sparse["indices"]
                .as_array()
                .unwrap_or_else(|| panic!("sparse indices missing in {point}"));
            let values = sparse["values"]
                .as_array()
                .unwrap_or_else(|| panic!("sparse values missing in {point}"));
            assert_eq!(indices.len(), values.len());
            let mut pairs: Vec<(u64, String)> = indices
                .iter()
                .zip(values.iter())
                .map(|(i, v)| (i.as_u64().unwrap(), v.as_f64().unwrap().to_string()))
                .collect();
            pairs.sort_by_key(|(i, _)| *i);
            pairs
        };

        // (a) Scroll with with_vector=true must carry the sparse vector.
        let scroll_body = serde_json::to_vec(&serde_json::json!({
            "limit": 10,
            "with_payload": true,
            "with_vector": true,
        }))
        .unwrap();
        let scroll_input = make_input(
            &ctx,
            "POST",
            "/collections/sparse_export/points/scroll",
            scroll_body,
            HashMap::from([("name".into(), "sparse_export".into())]),
        );
        let mut scroll_response = Response::new();
        handle_scroll_points(&scroll_input, &mut scroll_response).unwrap();
        assert_eq!(
            scroll_response.status,
            200,
            "scroll body: {}",
            String::from_utf8_lossy(&scroll_response.body)
        );
        let scroll_json: serde_json::Value = serde_json::from_slice(&scroll_response.body).unwrap();
        let scrolled = scroll_json["result"]["points"].as_array().unwrap();
        assert_eq!(
            scrolled.len(),
            1,
            "scroll must return the point: {scroll_json}"
        );
        assert_eq!(
            extract_sparse(&scrolled[0]),
            expected,
            "scroll must export the sparse vector",
        );

        // (b) Get by id with with_vector=true must also carry the sparse vector.
        let get_body = serde_json::to_vec(&serde_json::json!({
            "ids": ["doc-1"],
            "with_payload": true,
            "with_vector": true,
        }))
        .unwrap();
        let get_input = make_input(
            &ctx,
            "POST",
            "/collections/sparse_export/points",
            get_body,
            HashMap::from([("name".into(), "sparse_export".into())]),
        );
        let mut get_response = Response::new();
        handle_get_points(&get_input, &mut get_response).unwrap();
        assert_eq!(get_response.status, 200);
        let get_json: serde_json::Value = serde_json::from_slice(&get_response.body).unwrap();
        let retrieved = get_json["result"].as_array().unwrap();
        assert_eq!(retrieved.len(), 1, "get must return the point: {get_json}");
        assert_eq!(
            extract_sparse(&retrieved[0]),
            expected,
            "get must export the sparse vector",
        );
    }

    #[test]
    fn hybrid_query_fuses_dense_and_sparse_channels() {
        let ctx = setup();
        create_collection_with_sparse(&ctx, "hybrid");
        upsert_mixed_points(
            &ctx,
            "hybrid",
            serde_json::json!([
                {
                    "id": "dense-only",
                    "vector": {
                        "dense": [0.95, 0.05, 0.0],
                        "lex_sparse": {"indices": [7], "values": [1.0]}
                    },
                    "payload": {"label": "dense-only", "group": "keep"}
                },
                {
                    "id": "overlap",
                    "vector": {
                        "dense": [1.0, 0.0, 0.0],
                        "lex_sparse": {"indices": [99], "values": [10.0]}
                    },
                    "payload": {"label": "overlap", "group": "keep"}
                },
                {
                    "id": "sparse-only",
                    "vector": {
                        "dense": [0.0, 1.0, 0.0],
                        "lex_sparse": {"indices": [99], "values": [9.0]}
                    },
                    "payload": {"label": "sparse-only", "group": "drop"}
                }
            ]),
        );

        let (status, body) = hybrid_query_points_raw(
            &ctx,
            "hybrid",
            serde_json::json!({
                "dense": [{
                    "using": "dense",
                    "query": [1.0, 0.0, 0.0],
                    "limit": 3
                }],
                "sparse": [{
                    "using": "lex_sparse",
                    "query": {"indices": [99], "values": [1.0]},
                    "limit": 3
                }],
                "limit": 3,
                "with_payload": true,
                "with_channel_points": true
            }),
        );
        assert_eq!(
            status,
            200,
            "body: {}",
            serde_json::to_string_pretty(&body).unwrap()
        );
        let results = body["result"]["points"].as_array().unwrap();
        assert_eq!(results[0]["payload"]["label"], "overlap");
        assert_eq!(body["result"]["fusion"]["dense"], 1);
        assert_eq!(body["result"]["fusion"]["sparse"], 1);
        assert_eq!(
            body["result"]["channels"]["dense"][0]["points"][0]["payload"]["label"],
            "overlap"
        );
        assert_eq!(
            body["result"]["channels"]["sparse"][0]["points"][0]["payload"]["label"],
            "overlap"
        );
    }

    #[test]
    fn hybrid_query_mmr_diversifies_dense_candidates() {
        let ctx = setup();
        create_dense_collection(&ctx, "hybrid_mmr");
        upsert_points(
            &ctx,
            "hybrid_mmr",
            serde_json::json!([
                {
                    "id": "anchor",
                    "vector": {"dense": [1.0, 0.0, 0.0]},
                    "payload": {"label": "anchor"}
                },
                {
                    "id": "near-duplicate",
                    "vector": {"dense": [0.99, 0.01, 0.0]},
                    "payload": {"label": "near-duplicate"}
                },
                {
                    "id": "diverse",
                    "vector": {"dense": [0.0, 1.0, 0.0]},
                    "payload": {"label": "diverse"}
                }
            ]),
        );

        let (status, body) = hybrid_query_points_raw(
            &ctx,
            "hybrid_mmr",
            serde_json::json!({
                "dense": [{
                    "using": "dense",
                    "query": [1.0, 0.0, 0.0],
                    "limit": 3
                }],
                "limit": 3,
                "with_payload": true,
                "mmr": {"lambda": 0.1}
            }),
        );
        assert_eq!(
            status,
            200,
            "body: {}",
            serde_json::to_string_pretty(&body).unwrap()
        );
        let results = body["result"]["points"].as_array().unwrap();
        assert_eq!(results[0]["payload"]["label"], "anchor");
        assert_eq!(results[1]["payload"]["label"], "diverse");
        assert_eq!(body["result"]["fusion"]["mmr"]["applied"], true);
        assert_eq!(body["result"]["fusion"]["mmr"]["candidate_vectors"], 3);
    }

    // ── Helpers for hybrid graph-channel tests ──

    fn point_uid(id: &str) -> u128 {
        point_id_to_u128(&serde_json::Value::String(id.into()))
    }

    fn upsert_graph_edge(ctx: &TestContext, collection: &str, label: &str, from: u128, to: u128) {
        let storage = ctx.collections.get_collection(collection).unwrap();
        let mut txn = storage.lmdb_env().unwrap().write_txn().unwrap();
        storage
            .upsert_edge(
                &mut txn,
                &EdgeUpsert {
                    id: from ^ to.rotate_left(64),
                    label: label.into(),
                    from_node: from,
                    to_node: to,
                    properties: HashMap::new(),
                },
            )
            .unwrap();
        txn.commit().unwrap();
    }

    fn hybrid_query_points_bytes(
        ctx: &TestContext,
        name: &str,
        body: serde_json::Value,
    ) -> (u16, Vec<u8>) {
        let body_bytes = sonic_rs::to_vec(&body).unwrap();
        let input = make_input(
            ctx,
            "POST",
            &format!("/collections/{}/points/hybrid_query", name),
            body_bytes,
            HashMap::from([("name".into(), name.into())]),
        );
        let mut resp = Response::new();
        handle_hybrid_query_points(&input, &mut resp).unwrap();
        (resp.status, resp.body)
    }

    #[test]
    fn hybrid_query_graph_channel_boosts_connected_point() {
        let ctx = setup();
        create_dense_collection(&ctx, "hybrid_graph");
        upsert_points(
            &ctx,
            "hybrid_graph",
            serde_json::json!([
                {"id": "a", "vector": {"dense": [1.0, 0.0, 0.0]}, "payload": {"label": "a"}},
                {"id": "b", "vector": {"dense": [0.80, 0.60, 0.0]}, "payload": {"label": "b"}},
                {"id": "c", "vector": {"dense": [0.82, 0.57, 0.0]}, "payload": {"label": "c"}}
            ]),
        );
        let a = point_uid("a");
        let b = point_uid("b");
        let c = point_uid("c");

        let request = serde_json::json!({
            "dense": [{"using": "dense", "query": [1.0, 0.0, 0.0], "limit": 3}],
            "limit": 3,
            "with_payload": true
        });

        // Baseline: c out-scores b on cosine, so the dense order is a, c, b.
        let (status, body) = hybrid_query_points_raw(&ctx, "hybrid_graph", request.clone());
        assert_eq!(status, 200);
        let baseline: Vec<String> = body["result"]["points"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["payload"]["label"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(baseline, ["a", "c", "b"]);

        // b is the structural hub: both the top hit a and the runner-up seed c
        // call into it, so the PPR channel ranks b above c.
        upsert_graph_edge(&ctx, "hybrid_graph", "calls", a, b);
        upsert_graph_edge(&ctx, "hybrid_graph", "calls", c, b);

        let mut with_graph = request;
        with_graph["graph"] = serde_json::json!({"edge_labels": ["calls"], "seed_count": 2});
        let (status, body) = hybrid_query_points_raw(&ctx, "hybrid_graph", with_graph);
        assert_eq!(
            status,
            200,
            "body: {}",
            serde_json::to_string_pretty(&body).unwrap()
        );
        let results = body["result"]["points"].as_array().unwrap();
        let pos = |label: &str| {
            results
                .iter()
                .position(|p| p["payload"]["label"] == label)
                .unwrap_or_else(|| panic!("{} missing from {:?}", label, results))
        };
        assert!(
            pos("b") < pos("c"),
            "structural boost should rank b above c: {:?}",
            results
        );
        let graph = &body["result"]["fusion"]["graph"];
        assert!(graph["seeds"].as_u64().unwrap() >= 1, "graph: {}", graph);
        assert!(graph["returned"].as_u64().unwrap() >= 1, "graph: {}", graph);
    }

    /// PPR neighbors must satisfy the request filter and still exist: a
    /// repo-B neighbor and a stale edge to a deleted point must not leak into
    /// a repo-A-filtered hybrid query through the graph channel.
    fn assert_graph_channel_respects_filter_and_existence(
        ctx: &TestContext,
        name: &str,
        create: fn(&TestContext, &str),
    ) {
        create(ctx, name);
        upsert_points(
            ctx,
            name,
            serde_json::json!([
                {"id": "a", "vector": {"dense": [1.0, 0.0, 0.0]}, "payload": {"label": "a", "repo": "A"}},
                {"id": "c", "vector": {"dense": [0.82, 0.57, 0.0]}, "payload": {"label": "c", "repo": "A"}},
                {"id": "b", "vector": {"dense": [0.80, 0.60, 0.0]}, "payload": {"label": "b", "repo": "B"}}
            ]),
        );
        let (a, b, c) = (point_uid("a"), point_uid("b"), point_uid("c"));
        let ghost = point_uid("ghost");
        let storage = ctx.collections.get_collection(name).unwrap();
        storage
            .with_write_backend(|w| {
                for (from, to) in [(a, b), (c, b), (a, ghost), (c, ghost)] {
                    storage.upsert_edge_be(
                        w,
                        &EdgeUpsert {
                            id: from ^ to.rotate_left(64),
                            label: "calls".into(),
                            from_node: from,
                            to_node: to,
                            properties: HashMap::new(),
                        },
                    )?;
                }
                Ok(())
            })
            .unwrap();

        let (status, body) = hybrid_query_points_raw(
            ctx,
            name,
            serde_json::json!({
                "dense": [{"using": "dense", "query": [1.0, 0.0, 0.0], "limit": 3}],
                "limit": 5,
                "with_payload": true,
                "filter": {"must": [{"key": "repo", "match": {"value": "A"}}]},
                "graph": {"edge_labels": ["calls"], "seed_count": 2}
            }),
        );
        assert_eq!(
            status,
            200,
            "body: {}",
            serde_json::to_string_pretty(&body).unwrap()
        );
        let points = body["result"]["points"].as_array().unwrap();
        assert!(!points.is_empty(), "body: {}", body);
        for point in points {
            assert_eq!(
                point["payload"]["repo"], "A",
                "graph channel leaked a filtered-out or deleted point: {:?}",
                points
            );
        }
    }

    #[test]
    fn hybrid_query_graph_channel_respects_filter_and_existence() {
        let ctx = setup();
        assert_graph_channel_respects_filter_and_existence(
            &ctx,
            "hybrid_graph_filter",
            create_dense_collection,
        );
    }

    #[test]
    fn hybrid_query_graph_channel_respects_filter_and_existence_lsm() {
        let ctx = setup_with_config(Config::default().with_lsm_in_memory());
        assert_graph_channel_respects_filter_and_existence(
            &ctx,
            "hybrid_graph_filter_lsm",
            create_dense_collection_lsm,
        );
    }

    #[test]
    fn hybrid_query_graph_absent_is_byte_identical() {
        let ctx = setup();
        create_dense_collection(&ctx, "hybrid_graph_absent");
        upsert_points(
            &ctx,
            "hybrid_graph_absent",
            serde_json::json!([
                {"id": "a", "vector": {"dense": [1.0, 0.0, 0.0]}, "payload": {"label": "a"}},
                {"id": "b", "vector": {"dense": [0.80, 0.60, 0.0]}, "payload": {"label": "b"}},
                {"id": "c", "vector": {"dense": [0.82, 0.57, 0.0]}, "payload": {"label": "c"}}
            ]),
        );
        let request = serde_json::json!({
            "dense": [{"using": "dense", "query": [1.0, 0.0, 0.0], "limit": 3}],
            "limit": 3,
            "with_payload": true
        });

        let (status_before, body_before) =
            hybrid_query_points_bytes(&ctx, "hybrid_graph_absent", request.clone());
        assert_eq!(status_before, 200);

        upsert_graph_edge(
            &ctx,
            "hybrid_graph_absent",
            "calls",
            point_uid("a"),
            point_uid("b"),
        );

        let (status_after, body_after) =
            hybrid_query_points_bytes(&ctx, "hybrid_graph_absent", request);
        assert_eq!(status_after, 200);
        assert_eq!(
            body_before, body_after,
            "graph-absent responses must be byte-identical with and without edges"
        );
        let json: serde_json::Value = serde_json::from_slice(&body_before).unwrap();
        assert!(json["result"]["fusion"].get("graph").is_none());
    }

    #[test]
    fn hybrid_query_graph_on_edgeless_collection_degrades() {
        let ctx = setup();
        create_dense_collection(&ctx, "hybrid_graph_edgeless");
        upsert_points(
            &ctx,
            "hybrid_graph_edgeless",
            serde_json::json!([
                {"id": "a", "vector": {"dense": [1.0, 0.0, 0.0]}, "payload": {"label": "a"}},
                {"id": "b", "vector": {"dense": [0.80, 0.60, 0.0]}, "payload": {"label": "b"}},
                {"id": "c", "vector": {"dense": [0.82, 0.57, 0.0]}, "payload": {"label": "c"}}
            ]),
        );
        let request = serde_json::json!({
            "dense": [{"using": "dense", "query": [1.0, 0.0, 0.0], "limit": 3}],
            "limit": 3,
            "with_payload": true
        });

        let (status, base) =
            hybrid_query_points_raw(&ctx, "hybrid_graph_edgeless", request.clone());
        assert_eq!(status, 200);
        assert!(base["result"]["fusion"].get("graph").is_none());

        let mut with_graph = request;
        with_graph["graph"] = serde_json::json!({"edge_labels": ["calls"], "seed_count": 2});
        let (status, body) = hybrid_query_points_raw(&ctx, "hybrid_graph_edgeless", with_graph);
        assert_eq!(
            status,
            200,
            "body: {}",
            serde_json::to_string_pretty(&body).unwrap()
        );
        assert_eq!(
            body["result"]["points"], base["result"]["points"],
            "edgeless graph channel must degrade to the two-channel result"
        );
        let graph = &body["result"]["fusion"]["graph"];
        assert_eq!(graph["seeds"], 2, "graph: {}", graph);
        assert_eq!(graph["returned"], 0, "graph: {}", graph);
    }

    #[test]
    fn hybrid_query_graph_invalid_direction_rejected() {
        let ctx = setup();
        create_dense_collection(&ctx, "hybrid_graph_direction");

        let (status, body) = hybrid_query_points_raw(
            &ctx,
            "hybrid_graph_direction",
            serde_json::json!({
                "dense": [{"using": "dense", "query": [1.0, 0.0, 0.0], "limit": 3}],
                "limit": 3,
                "graph": {"direction": "sideways"}
            }),
        );
        assert_eq!(
            status,
            400,
            "body: {}",
            serde_json::to_string_pretty(&body).unwrap()
        );
        assert!(
            body["status"]["error"]
                .as_str()
                .unwrap()
                .contains("graph.direction"),
            "body: {}",
            body
        );
    }

    // ── Helpers + tests for chunk-level edges (`base_collection` on rebuild) ──

    fn rebuild_adjacency_input(
        ctx: &TestContext,
        collection: &str,
        paths: &[&str],
        base_collection: Option<&str>,
    ) -> HandlerInput {
        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "collection": collection,
            "paths": paths,
            "base_collection": base_collection,
        }))
        .unwrap();
        make_input(
            ctx,
            "POST",
            "/v1/graph/rebuild_adjacency_for_paths",
            body,
            HashMap::new(),
        )
    }

    #[test]
    fn rebuild_adjacency_chunk_edges_created_in_base_collection() {
        let ctx = setup();
        create_dense_collection(&ctx, "chunk_edges_base");
        upsert_points(
            &ctx,
            "chunk_edges_base",
            serde_json::json!([
                {"id": "chunk_a", "vector": {"dense": [1.0, 0.0, 0.0]}, "payload": {"path": "src/a.rs"}},
                {"id": "chunk_b", "vector": {"dense": [0.0, 1.0, 0.0]}, "payload": {"path": "src/b.rs"}},
                {"id": "chunk_c", "vector": {"dense": [0.0, 0.0, 1.0]}, "payload": {"path": "src/c.rs"}}
            ]),
        );
        let a = point_uid("chunk_a");
        let b = point_uid("chunk_b");

        let graph_collection = "chunk_edges_base_graph";
        let graph_storage = ctx.collections.create_collection(graph_collection).unwrap();
        let mut txn = graph_storage.lmdb_env().unwrap().write_txn().unwrap();
        graph_storage
            .upsert_node(
                &mut txn,
                &NodeUpsert {
                    id: 0xC0FFEE,
                    label: "point".into(),
                    properties: HashMap::from([
                        ("caller_point_id".into(), Value::String(a.to_string())),
                        ("callee_point_id".into(), Value::String(b.to_string())),
                        ("caller_path".into(), Value::String("src/a.rs".into())),
                        ("callee_path".into(), Value::String("src/b.rs".into())),
                        ("edge_type".into(), Value::String("calls".into())),
                    ]),
                },
            )
            .unwrap();
        txn.commit().unwrap();

        let input = rebuild_adjacency_input(
            &ctx,
            graph_collection,
            &["src/a.rs"],
            Some("chunk_edges_base"),
        );
        let mut response = Response::new();
        handle_rebuild_adjacency_for_paths(&input, &mut response).unwrap();
        assert_eq!(response.status, 200);
        let payload: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(payload["chunk_edges_upserted"], 1, "body: {}", payload);

        let base_storage = ctx.collections.get_collection("chunk_edges_base").unwrap();
        let r = base_storage.backend.begin_read().unwrap();
        let label_hash = hash_label("CALLS", None);
        let out_peers: Vec<u128> = base_storage
            .adjacency_pairs_be(&r, a, &label_hash, true)
            .unwrap()
            .into_iter()
            .map(|(peer, _)| peer)
            .collect();
        assert_eq!(out_peers, vec![b]);
        let in_peers: Vec<u128> = base_storage
            .adjacency_pairs_be(&r, b, &label_hash, false)
            .unwrap()
            .into_iter()
            .map(|(peer, _)| peer)
            .collect();
        assert_eq!(in_peers, vec![a]);
    }

    #[test]
    fn rebuild_adjacency_chunk_edge_skips_missing_callee_node() {
        let ctx = setup();
        create_dense_collection(&ctx, "chunk_edges_missing");
        upsert_points(
            &ctx,
            "chunk_edges_missing",
            serde_json::json!([
                {"id": "only_chunk", "vector": {"dense": [1.0, 0.0, 0.0]}, "payload": {"path": "src/a.rs"}}
            ]),
        );
        let a = point_uid("only_chunk");
        let missing = point_uid("does_not_exist");

        let graph_collection = "chunk_edges_missing_graph";
        let graph_storage = ctx.collections.create_collection(graph_collection).unwrap();
        let mut txn = graph_storage.lmdb_env().unwrap().write_txn().unwrap();
        graph_storage
            .upsert_node(
                &mut txn,
                &NodeUpsert {
                    id: 0xDEAD,
                    label: "point".into(),
                    properties: HashMap::from([
                        ("caller_point_id".into(), Value::String(a.to_string())),
                        ("callee_point_id".into(), Value::String(missing.to_string())),
                        ("caller_path".into(), Value::String("src/a.rs".into())),
                        ("callee_path".into(), Value::String("src/nope.rs".into())),
                        ("edge_type".into(), Value::String("calls".into())),
                    ]),
                },
            )
            .unwrap();
        txn.commit().unwrap();

        let input = rebuild_adjacency_input(
            &ctx,
            graph_collection,
            &["src/a.rs"],
            Some("chunk_edges_missing"),
        );
        let mut response = Response::new();
        handle_rebuild_adjacency_for_paths(&input, &mut response).unwrap();
        assert_eq!(response.status, 200);
        let payload: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(payload["chunk_edges_upserted"], 0, "body: {}", payload);
        assert!(payload["warning"].is_null(), "body: {}", payload);
    }

    #[test]
    fn rebuild_adjacency_chunk_edges_idempotent() {
        let ctx = setup();
        create_dense_collection(&ctx, "chunk_edges_idem");
        upsert_points(
            &ctx,
            "chunk_edges_idem",
            serde_json::json!([
                {"id": "idem_a", "vector": {"dense": [1.0, 0.0, 0.0]}, "payload": {"path": "src/a.rs"}},
                {"id": "idem_b", "vector": {"dense": [0.0, 1.0, 0.0]}, "payload": {"path": "src/b.rs"}}
            ]),
        );
        let a = point_uid("idem_a");
        let b = point_uid("idem_b");

        let graph_collection = "chunk_edges_idem_graph";
        let graph_storage = ctx.collections.create_collection(graph_collection).unwrap();
        let mut txn = graph_storage.lmdb_env().unwrap().write_txn().unwrap();
        graph_storage
            .upsert_node(
                &mut txn,
                &NodeUpsert {
                    id: 0xABCD,
                    label: "point".into(),
                    properties: HashMap::from([
                        ("caller_point_id".into(), Value::String(a.to_string())),
                        ("callee_point_id".into(), Value::String(b.to_string())),
                        ("caller_path".into(), Value::String("src/a.rs".into())),
                        ("callee_path".into(), Value::String("src/b.rs".into())),
                        ("edge_type".into(), Value::String("calls".into())),
                    ]),
                },
            )
            .unwrap();
        txn.commit().unwrap();

        for _ in 0..2 {
            let input = rebuild_adjacency_input(
                &ctx,
                graph_collection,
                &["src/a.rs"],
                Some("chunk_edges_idem"),
            );
            let mut response = Response::new();
            handle_rebuild_adjacency_for_paths(&input, &mut response).unwrap();
            assert_eq!(response.status, 200);
            let payload: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
            assert_eq!(payload["chunk_edges_upserted"], 1, "body: {}", payload);
        }

        let base_storage = ctx.collections.get_collection("chunk_edges_idem").unwrap();
        let txn = base_storage.lmdb_env().unwrap().read_txn().unwrap();
        let metadata = base_storage.get_metadata(&txn).unwrap();
        assert_eq!(metadata.stats.edge_count, 1);
    }

    #[test]
    fn rebuild_adjacency_chunk_edge_decimal_point_id_not_misparsed_as_hex() {
        let ctx = setup();
        create_dense_collection(&ctx, "chunk_edges_numeric");
        upsert_points(
            &ctx,
            "chunk_edges_numeric",
            serde_json::json!([
                {"id": 5001, "vector": {"dense": [1.0, 0.0, 0.0]}, "payload": {"path": "src/num_a.rs"}},
                {"id": 5002, "vector": {"dense": [0.0, 1.0, 0.0]}, "payload": {"path": "src/num_b.rs"}}
            ]),
        );

        let graph_collection = "chunk_edges_numeric_graph";
        let graph_storage = ctx.collections.create_collection(graph_collection).unwrap();
        let mut txn = graph_storage.lmdb_env().unwrap().write_txn().unwrap();
        // "5001"/"5002" are also valid hex (0x5001/0x5002 = 20481/20482) — a
        // hex-first parse would silently miss both real nodes and skip the
        // edge. Decimal-first must resolve them to the real point ids.
        graph_storage
            .upsert_node(
                &mut txn,
                &NodeUpsert {
                    id: 0xFACE,
                    label: "point".into(),
                    properties: HashMap::from([
                        ("caller_point_id".into(), Value::String("5001".into())),
                        ("callee_point_id".into(), Value::String("5002".into())),
                        ("caller_path".into(), Value::String("src/num_a.rs".into())),
                        ("callee_path".into(), Value::String("src/num_b.rs".into())),
                        ("edge_type".into(), Value::String("calls".into())),
                    ]),
                },
            )
            .unwrap();
        txn.commit().unwrap();

        let input = rebuild_adjacency_input(
            &ctx,
            graph_collection,
            &["src/num_a.rs"],
            Some("chunk_edges_numeric"),
        );
        let mut response = Response::new();
        handle_rebuild_adjacency_for_paths(&input, &mut response).unwrap();
        assert_eq!(response.status, 200);
        let payload: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(
            payload["chunk_edges_upserted"], 1,
            "decimal point ids '5001'/'5002' must resolve to the real numeric \
             point ids (5001/5002), not their hex misreading: body: {}",
            payload
        );

        let base_storage = ctx
            .collections
            .get_collection("chunk_edges_numeric")
            .unwrap();
        let r = base_storage.backend.begin_read().unwrap();
        let label_hash = hash_label("CALLS", None);
        let peers: Vec<u128> = base_storage
            .adjacency_pairs_be(&r, 5001, &label_hash, true)
            .unwrap()
            .into_iter()
            .map(|(peer, _)| peer)
            .collect();
        assert_eq!(peers, vec![5002]);
    }

    #[test]
    fn hybrid_query_graph_channel_uses_rebuilt_chunk_edges() {
        let ctx = setup();
        create_dense_collection(&ctx, "chunk_hybrid");
        upsert_points(
            &ctx,
            "chunk_hybrid",
            serde_json::json!([
                {"id": "a", "vector": {"dense": [1.0, 0.0, 0.0]}, "payload": {"label": "a", "path": "src/a.rs"}},
                {"id": "b", "vector": {"dense": [0.80, 0.60, 0.0]}, "payload": {"label": "b", "path": "src/b.rs"}},
                {"id": "c", "vector": {"dense": [0.82, 0.57, 0.0]}, "payload": {"label": "c", "path": "src/c.rs"}}
            ]),
        );
        let a = point_uid("a");
        let b = point_uid("b");
        let c = point_uid("c");

        let graph_collection = "chunk_hybrid_graph";
        let graph_storage = ctx.collections.create_collection(graph_collection).unwrap();
        let mut txn = graph_storage.lmdb_env().unwrap().write_txn().unwrap();
        // a calls b, c calls b — mirrors the PPR-hub scenario in
        // `hybrid_query_graph_channel_boosts_connected_point`, but the edges
        // here are derived chunk edges from a rebuild, not hand-inserted.
        graph_storage
            .upsert_node(
                &mut txn,
                &NodeUpsert {
                    id: 0x1001,
                    label: "point".into(),
                    properties: HashMap::from([
                        ("caller_point_id".into(), Value::String(a.to_string())),
                        ("callee_point_id".into(), Value::String(b.to_string())),
                        ("caller_path".into(), Value::String("src/a.rs".into())),
                        ("callee_path".into(), Value::String("src/b.rs".into())),
                        ("edge_type".into(), Value::String("calls".into())),
                    ]),
                },
            )
            .unwrap();
        graph_storage
            .upsert_node(
                &mut txn,
                &NodeUpsert {
                    id: 0x1002,
                    label: "point".into(),
                    properties: HashMap::from([
                        ("caller_point_id".into(), Value::String(c.to_string())),
                        ("callee_point_id".into(), Value::String(b.to_string())),
                        ("caller_path".into(), Value::String("src/c.rs".into())),
                        ("callee_path".into(), Value::String("src/b.rs".into())),
                        ("edge_type".into(), Value::String("calls".into())),
                    ]),
                },
            )
            .unwrap();
        txn.commit().unwrap();

        let input = rebuild_adjacency_input(
            &ctx,
            graph_collection,
            &["src/a.rs", "src/c.rs"],
            Some("chunk_hybrid"),
        );
        let mut rebuild_response = Response::new();
        handle_rebuild_adjacency_for_paths(&input, &mut rebuild_response).unwrap();
        assert_eq!(rebuild_response.status, 200);
        let rebuild_payload: serde_json::Value =
            serde_json::from_slice(&rebuild_response.body).unwrap();
        assert_eq!(
            rebuild_payload["chunk_edges_upserted"], 2,
            "body: {}",
            rebuild_payload
        );

        let request = serde_json::json!({
            "dense": [{"using": "dense", "query": [1.0, 0.0, 0.0], "limit": 3}],
            "limit": 3,
            "with_payload": true,
            // Chunk edges always store the label upper-cased (CALLS/IMPORTS/
            // INHERITS_FROM), unlike the hand-inserted "calls" edges in the
            // PPR test above — request the matching case.
            "graph": {"edge_labels": ["CALLS"], "seed_count": 2}
        });
        let (status, body) = hybrid_query_points_raw(&ctx, "chunk_hybrid", request);
        assert_eq!(
            status,
            200,
            "body: {}",
            serde_json::to_string_pretty(&body).unwrap()
        );
        let graph = &body["result"]["fusion"]["graph"];
        assert!(
            graph["returned"].as_u64().unwrap_or(0) > 0,
            "graph channel should return the CALLS-derived chunk-edge peer(s): {}",
            graph
        );
    }

    #[test]
    fn hybrid_query_applies_filter_to_all_channels() {
        let ctx = setup();
        create_collection_with_sparse(&ctx, "hybrid_filter");
        upsert_mixed_points(
            &ctx,
            "hybrid_filter",
            serde_json::json!([
                {
                    "id": "keep",
                    "vector": {
                        "dense": [1.0, 0.0, 0.0],
                        "lex_sparse": {"indices": [42], "values": [1.0]}
                    },
                    "payload": {"label": "keep", "group": "keep"}
                },
                {
                    "id": "drop",
                    "vector": {
                        "dense": [0.99, 0.01, 0.0],
                        "lex_sparse": {"indices": [42], "values": [10.0]}
                    },
                    "payload": {"label": "drop", "group": "drop"}
                }
            ]),
        );

        let (status, body) = hybrid_query_points_raw(
            &ctx,
            "hybrid_filter",
            serde_json::json!({
                "dense": [{
                    "using": "dense",
                    "query": [1.0, 0.0, 0.0],
                    "limit": 5
                }],
                "sparse": [{
                    "using": "lex_sparse",
                    "query": {"indices": [42], "values": [1.0]},
                    "limit": 5
                }],
                "limit": 5,
                "with_payload": true,
                "filter": {
                    "must": [{
                        "key": "group",
                        "match": {"value": "keep"}
                    }]
                }
            }),
        );
        assert_eq!(
            status,
            200,
            "body: {}",
            serde_json::to_string_pretty(&body).unwrap()
        );
        let results = body["result"]["points"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["payload"]["label"], "keep");
    }

    #[test]
    fn hybrid_query_honors_per_entry_filter_per_channel() {
        let ctx = setup();
        create_collection_with_sparse(&ctx, "hybrid_entry_filter");
        upsert_mixed_points(
            &ctx,
            "hybrid_entry_filter",
            serde_json::json!([
                {
                    "id": "dense-match",
                    "vector": {
                        "dense": [1.0, 0.0, 0.0],
                        "lex_sparse": {"indices": [42], "values": [1.0]}
                    },
                    "payload": {"label": "dense-match", "group": "keep"}
                },
                {
                    "id": "sparse-match",
                    "vector": {
                        "dense": [0.0, 1.0, 0.0],
                        "lex_sparse": {"indices": [42], "values": [10.0]}
                    },
                    "payload": {"label": "sparse-match", "group": "keep"}
                }
            ]),
        );

        let (status, body) = hybrid_query_points_raw(
            &ctx,
            "hybrid_entry_filter",
            serde_json::json!({
                "dense": [{
                    "using": "dense",
                    "query": [1.0, 0.0, 0.0],
                    "limit": 5,
                    "filter": {
                        "must": [{
                            "key": "group",
                            "match": {"value": "missing"}
                        }]
                    }
                }],
                "sparse": [{
                    "using": "lex_sparse",
                    "query": {"indices": [42], "values": [1.0]},
                    "limit": 5
                }],
                "limit": 5,
                "with_payload": true,
                "with_channel_points": true
            }),
        );
        assert_eq!(
            status,
            200,
            "body: {}",
            serde_json::to_string_pretty(&body).unwrap()
        );
        assert!(body["result"]["channels"]["dense"][0]["points"]
            .as_array()
            .unwrap()
            .is_empty());
        assert_eq!(
            body["result"]["channels"]["sparse"][0]["points"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(body["result"]["points"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn hybrid_query_requires_at_least_one_channel() {
        let ctx = setup();
        create_collection_with_sparse(&ctx, "hybrid_empty");

        let (status, body) = hybrid_query_points_raw(
            &ctx,
            "hybrid_empty",
            serde_json::json!({
                "limit": 5
            }),
        );
        assert_eq!(status, 400);
        assert!(body["status"]["error"]
            .as_str()
            .unwrap()
            .contains("At least one dense or sparse query"));
    }

    #[test]
    fn single_vector_dense_query() {
        let ctx = setup();
        create_dense_collection(&ctx, "svd");
        upsert_points(
            &ctx,
            "svd",
            serde_json::json!([
                {"id": 1, "vector": {"dense": [1.0, 0.0, 0.0]}, "payload": {"label": "x"}},
                {"id": 2, "vector": {"dense": [0.0, 1.0, 0.0]}, "payload": {"label": "y"}}
            ]),
        );

        // Single-vector dense query (no prefetch, no fusion).
        let (status, body) = query_points_raw(
            &ctx,
            "svd",
            serde_json::json!({
                "query": [1.0, 0.0, 0.0],
                "using": "dense",
                "limit": 2,
                "with_payload": true
            }),
        );
        assert_eq!(status, 200);
        let results = body["result"]["points"].as_array().unwrap();
        assert!(!results.is_empty());
        // First result should be the closest to [1, 0, 0].
        assert_eq!(results[0]["payload"]["label"], "x");
    }

    #[test]
    fn single_vector_dense_query_accepts_nearest_wrapper() {
        let ctx = setup();
        create_dense_collection(&ctx, "svd_nearest");
        upsert_points(
            &ctx,
            "svd_nearest",
            serde_json::json!([
                {"id": 1, "vector": {"dense": [1.0, 0.0, 0.0]}, "payload": {"label": "x"}},
                {"id": 2, "vector": {"dense": [0.0, 1.0, 0.0]}, "payload": {"label": "y"}}
            ]),
        );

        let (status, body) = query_points_raw(
            &ctx,
            "svd_nearest",
            serde_json::json!({
                "query": {"nearest": [1.0, 0.0, 0.0]},
                "using": "dense",
                "limit": 2,
                "with_payload": true
            }),
        );
        assert_eq!(status, 200);
        let results = body["result"]["points"].as_array().unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0]["payload"]["label"], "x");
    }

    #[test]
    fn update_collection_adds_sparse_vector() {
        let ctx = setup();
        // Start with dense only.
        create_dense_collection(&ctx, "upd");

        // PATCH to add sparse vector.
        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "sparse_vectors": {
                "lex_sparse": {"modifier": "idf"}
            }
        }))
        .unwrap();
        let input = make_input(
            &ctx,
            "PATCH",
            "/collections/upd",
            body,
            HashMap::from([("name".into(), "upd".into())]),
        );
        let mut resp = Response::new();
        handle_update_collection(&input, &mut resp).unwrap();
        assert_eq!(
            resp.status,
            200,
            "body: {}",
            String::from_utf8_lossy(&resp.body)
        );

        // Now upsert points with both dense + sparse.
        upsert_mixed_points(
            &ctx,
            "upd",
            serde_json::json!([{
                "id": "p1",
                "vector": {
                    "dense": [1.0, 0.0, 0.0],
                    "lex_sparse": {"indices": [42], "values": [3.14]}
                },
                "payload": {"tag": "first"}
            }]),
        );

        // Query the sparse space.
        let (status, body) = query_points_raw(
            &ctx,
            "upd",
            serde_json::json!({
                "query": {"indices": [42], "values": [1.0]},
                "using": "lex_sparse",
                "limit": 5,
                "with_payload": true
            }),
        );
        assert_eq!(status, 200);
        let results = body["result"]["points"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["payload"]["tag"], "first");
    }

    #[test]
    fn update_collection_qdrant_quantization_adds_compact_vector() {
        let ctx = setup();
        create_dense_collection(&ctx, "upd_quant");

        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "vectors": {
                "qd_dense": {
                    "size": 3,
                    "distance": "Cosine",
                    "quantization_config": {"scalar": {"type": "int8"}}
                }
            }
        }))
        .unwrap();
        let input = make_input(
            &ctx,
            "PATCH",
            "/collections/upd_quant",
            body,
            HashMap::from([("name".into(), "upd_quant".into())]),
        );
        let mut resp = Response::new();
        handle_update_collection(&input, &mut resp).unwrap();
        assert_eq!(
            resp.status,
            200,
            "body: {}",
            String::from_utf8_lossy(&resp.body)
        );

        let storage = ctx.collections.get_collection("upd_quant").unwrap();
        let spindle = storage
            .named_vectors
            .get_config("qd_dense")
            .expect("named vector 'qd_dense' missing")
            .spindle;
        assert_eq!(spindle.mode, SpindleMode::ScalarInt8);
        assert!(!spindle.keep_original);
        assert!(!spindle.rescore);
        assert_eq!(spindle.oversampling, 1);
    }

    #[test]
    fn update_collection_idempotent_same_config() {
        let ctx = setup();
        create_collection_with_sparse(&ctx, "idem");

        // PATCH with exact same sparse config should be a no-op.
        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "sparse_vectors": {
                "lex_sparse": {"modifier": "idf"}
            }
        }))
        .unwrap();
        let input = make_input(
            &ctx,
            "PATCH",
            "/collections/idem",
            body,
            HashMap::from([("name".into(), "idem".into())]),
        );
        let mut resp = Response::new();
        handle_update_collection(&input, &mut resp).unwrap();
        assert_eq!(resp.status, 200);
    }

    #[test]
    fn update_collection_optimizer_threshold_zero_enables_bulk_mode() {
        let ctx = setup();
        create_dense_collection(&ctx, "bulk_mode");

        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "optimizer_config": {
                "indexing_threshold": 0
            }
        }))
        .unwrap();
        let input = make_input(
            &ctx,
            "PATCH",
            "/collections/bulk_mode",
            body,
            HashMap::from([("name".into(), "bulk_mode".into())]),
        );
        let mut resp = Response::new();
        handle_update_collection(&input, &mut resp).unwrap();
        assert_eq!(resp.status, 200);

        upsert_points(
            &ctx,
            "bulk_mode",
            serde_json::json!([
                {"id": 1, "vector": {"dense": [1.0, 0.0, 0.0]}, "payload": {"label": "x"}},
                {"id": 2, "vector": {"dense": [0.0, 1.0, 0.0]}, "payload": {"label": "y"}}
            ]),
        );

        let input = make_input(
            &ctx,
            "GET",
            "/collections/bulk_mode",
            vec![],
            HashMap::from([("name".into(), "bulk_mode".into())]),
        );
        let mut resp = Response::new();
        handle_get_collection(&input, &mut resp).unwrap();
        assert_eq!(resp.status, 200);
        let body: serde_json::Value = serde_json::from_slice(&resp.body).unwrap();
        assert_eq!(
            body["result"]["config"]["optimizer_config"]["indexing_threshold"],
            0
        );
        assert_eq!(body["result"]["indexed_vectors_count"], 0);
        assert_eq!(body["result"]["points_count"], 2);
    }

    #[test]
    fn update_collection_optimizer_threshold_null_resets_to_global() {
        let ctx = setup();
        create_dense_collection(&ctx, "bulk_reset");

        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "optimizer_config": {
                "indexing_threshold": 0
            }
        }))
        .unwrap();
        let input = make_input(
            &ctx,
            "PATCH",
            "/collections/bulk_reset",
            body,
            HashMap::from([("name".into(), "bulk_reset".into())]),
        );
        let mut resp = Response::new();
        handle_update_collection(&input, &mut resp).unwrap();
        assert_eq!(resp.status, 200);

        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "optimizer_config": {
                "indexing_threshold": null
            }
        }))
        .unwrap();
        let input = make_input(
            &ctx,
            "PATCH",
            "/collections/bulk_reset",
            body,
            HashMap::from([("name".into(), "bulk_reset".into())]),
        );
        let mut resp = Response::new();
        handle_update_collection(&input, &mut resp).unwrap();
        assert_eq!(resp.status, 200);

        let input = make_input(
            &ctx,
            "GET",
            "/collections/bulk_reset",
            vec![],
            HashMap::from([("name".into(), "bulk_reset".into())]),
        );
        let mut resp = Response::new();
        handle_get_collection(&input, &mut resp).unwrap();
        assert_eq!(resp.status, 200);
        let body: serde_json::Value = serde_json::from_slice(&resp.body).unwrap();
        // After a null-reset the effective threshold should fall back to the
        // global default, not the previously-set override (0). We compare
        // against the actual default constant so this test doesn't break if
        // the default is ever tuned.
        assert_eq!(
            body["result"]["config"]["optimizer_config"]["indexing_threshold"],
            crate::helix_engine::graph_core::config::DEFAULT_VECTOR_FLAT_SCAN_THRESHOLD
        );
    }

    #[test]
    #[serial]
    fn estimate_upsert_headroom_respects_env_floor() {
        let key = "HELIX_UPSERT_HEADROOM_MB";
        let previous = std::env::var_os(key);
        unsafe {
            std::env::set_var(key, "64");
        }

        let point = ReplicatedPoint {
            id: 1,
            vectors: HashMap::new(),
            sparse_vectors: HashMap::new(),
            payload: HashMap::new(),
        };
        assert_eq!(
            estimate_upsert_headroom_bytes(1, &[point]),
            64 * 1024 * 1024
        );

        unsafe {
            match previous {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }

    #[test]
    fn clamp_limit_zero_returns_default_and_huge_is_capped() {
        // Zero ⇒ Qdrant default of 10.
        assert_eq!(clamp_limit(0), default_limit());
        // Values under the cap pass through unchanged.
        assert_eq!(clamp_limit(50), 50);
        // Absurd values get clamped to the configured cap. We don't assume a
        // specific cap value — just that the result doesn't exceed it and is
        // smaller than the raw input.
        let cap = max_search_limit();
        let clamped = clamp_limit(usize::MAX);
        assert_eq!(clamped, cap);
        assert!(clamped < usize::MAX);
        // Exactly at cap is preserved.
        assert_eq!(clamp_limit(cap), cap);
    }

    #[test]
    fn channel_result_window_preserves_zero_offset_prefetch_limit() {
        assert_eq!(channel_result_window(1, 2, 0), 1);
    }

    #[test]
    fn channel_result_window_expands_for_offset() {
        assert_eq!(channel_result_window(1, 2, 1), 2);
        assert_eq!(channel_result_window(5, 2, 1), 5);
    }

    #[test]
    fn score_threshold_rejects_non_finite_values() {
        assert_eq!(validate_score_threshold(Some(0.5)), Ok(()));
        assert_eq!(
            validate_score_threshold(Some(f32::INFINITY)),
            Err("score_threshold must be finite")
        );
        assert_eq!(
            validate_score_threshold(Some(f32::NAN)),
            Err("score_threshold must be finite")
        );
    }

    #[test]
    fn search_points_rejects_parsed_non_finite_score_threshold() {
        let ctx = setup();
        create_dense_collection(&ctx, "bad_search_threshold");
        let input = make_input(
            &ctx,
            "POST",
            "/collections/bad_search_threshold/points/search",
            br#"{
                "vector": {"name": "dense", "vector": [1.0, 0.0, 0.0]},
                "score_threshold": 3.5e38
            }"#
            .to_vec(),
            HashMap::from([("name".into(), "bad_search_threshold".into())]),
        );
        let mut response = Response::new();

        handle_search_points(&input, &mut response).unwrap();

        assert_eq!(response.status, 400);
        let body: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert!(body["status"]["error"]
            .as_str()
            .unwrap()
            .contains("score_threshold must be finite"));
    }

    #[test]
    fn query_points_rejects_parsed_non_finite_score_threshold() {
        let ctx = setup();
        create_dense_collection(&ctx, "bad_query_threshold");

        let (status, body) = query_points_raw(
            &ctx,
            "bad_query_threshold",
            serde_json::json!({
                "query": [1.0, 0.0, 0.0],
                "score_threshold": 3.5e38
            }),
        );

        assert_eq!(status, 400);
        assert!(body["status"]["error"]
            .as_str()
            .unwrap()
            .contains("score_threshold must be finite"));
    }

    #[test]
    fn hnsw_config_rejects_m_below_two() {
        let ctx = setup();
        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "vectors": {"dense": {"size": 3, "distance": "Cosine"}},
            "hnsw_config": {"m": 1}
        }))
        .unwrap();
        let input = make_input(
            &ctx,
            "PUT",
            "/collections/bad_hnsw",
            body,
            HashMap::from([("name".into(), "bad_hnsw".into())]),
        );
        let mut response = Response::new();

        handle_create_collection(&input, &mut response).unwrap();

        assert_eq!(response.status, 400);
        let body: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert!(body["status"]["error"]
            .as_str()
            .unwrap()
            .contains("hnsw_config.m must be at least 2"));
    }

    #[test]
    fn search_points_applies_score_threshold_and_offset() {
        let ctx = setup();
        create_dense_collection(&ctx, "score_page");
        upsert_points(
            &ctx,
            "score_page",
            serde_json::json!([
                {"id": 1, "vector": {"dense": [1.0, 0.0, 0.0]}, "payload": {}},
                {"id": 2, "vector": {"dense": [0.9, 0.1, 0.0]}, "payload": {}},
                {"id": 3, "vector": {"dense": [0.0, 1.0, 0.0]}, "payload": {}}
            ]),
        );
        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "vector": {"name": "dense", "vector": [1.0, 0.0, 0.0]},
            "limit": 1,
            "offset": 1,
            "score_threshold": 0.9,
            "with_payload": false
        }))
        .unwrap();
        let input = make_input(
            &ctx,
            "POST",
            "/collections/score_page/points/search",
            body,
            HashMap::from([("name".into(), "score_page".into())]),
        );
        let mut response = Response::new();

        handle_search_points(&input, &mut response).unwrap();

        assert_eq!(response.status, 200);
        let body: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        let points = body["result"].as_array().unwrap();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0]["id"], format_point_id(2));
        assert!(points[0]["score"].as_f64().unwrap() >= 0.9);
    }

    #[test]
    fn search_points_euclid_score_threshold_keeps_distances_within_threshold() {
        let ctx = setup();
        create_euclid_collection(&ctx, "euclid_search_threshold");
        upsert_points(
            &ctx,
            "euclid_search_threshold",
            serde_json::json!([
                {"id": "near", "vector": {"dense": [0.5, 0.0, 0.0]}, "payload": {"label": "near"}},
                {"id": "far", "vector": {"dense": [1.0, 0.0, 0.0]}, "payload": {"label": "far"}}
            ]),
        );
        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "vector": {"name": "dense", "vector": [0.0, 0.0, 0.0]},
            "limit": 10,
            "score_threshold": 0.5,
            "with_payload": true
        }))
        .unwrap();
        let input = make_input(
            &ctx,
            "POST",
            "/collections/euclid_search_threshold/points/search",
            body,
            HashMap::from([("name".into(), "euclid_search_threshold".into())]),
        );
        let mut response = Response::new();

        handle_search_points(&input, &mut response).unwrap();

        assert_eq!(response.status, 200);
        let body: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        let points = body["result"].as_array().unwrap();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0]["payload"]["label"], serde_json::json!("near"));
        let score = points[0]["score"].as_f64().unwrap();
        assert!((score + 0.5).abs() < 0.000001);
    }

    #[test]
    fn query_points_euclid_score_threshold_keeps_distances_within_threshold() {
        let ctx = setup();
        create_euclid_collection(&ctx, "euclid_query_threshold");
        upsert_points(
            &ctx,
            "euclid_query_threshold",
            serde_json::json!([
                {"id": "near", "vector": {"dense": [0.5, 0.0, 0.0]}, "payload": {"label": "near"}},
                {"id": "far", "vector": {"dense": [1.0, 0.0, 0.0]}, "payload": {"label": "far"}}
            ]),
        );

        let (status, body) = query_points_raw(
            &ctx,
            "euclid_query_threshold",
            serde_json::json!({
                "query": [0.0, 0.0, 0.0],
                "using": "dense",
                "limit": 10,
                "score_threshold": 0.5,
                "with_payload": true
            }),
        );

        assert_eq!(status, 200, "query failed: {body}");
        let points = body["result"]["points"].as_array().unwrap();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0]["payload"]["label"], serde_json::json!("near"));
        let score = points[0]["score"].as_f64().unwrap();
        assert!((score + 0.5).abs() < 0.000001);
    }

    #[test]
    fn search_points_sparse_offset_is_applied_once() {
        let ctx = setup();
        create_collection_with_sparse(&ctx, "sparse_page");
        upsert_mixed_points(
            &ctx,
            "sparse_page",
            serde_json::json!([
                {
                    "id": "sparse-a",
                    "vector": {
                        "dense": [1.0, 0.0, 0.0],
                        "lex_sparse": {"indices": [7], "values": [3.0]}
                    },
                    "payload": {"rank": 1}
                },
                {
                    "id": "sparse-b",
                    "vector": {
                        "dense": [0.0, 1.0, 0.0],
                        "lex_sparse": {"indices": [7], "values": [2.0]}
                    },
                    "payload": {"rank": 2}
                },
                {
                    "id": "sparse-c",
                    "vector": {
                        "dense": [0.0, 0.0, 1.0],
                        "lex_sparse": {"indices": [7], "values": [1.0]}
                    },
                    "payload": {"rank": 3}
                }
            ]),
        );
        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "vector": {
                "name": "lex_sparse",
                "vector": {"indices": [7], "values": [1.0]}
            },
            "limit": 1,
            "offset": 1,
            "with_payload": true
        }))
        .unwrap();
        let input = make_input(
            &ctx,
            "POST",
            "/collections/sparse_page/points/search",
            body,
            HashMap::from([("name".into(), "sparse_page".into())]),
        );
        let mut response = Response::new();

        handle_search_points(&input, &mut response).unwrap();

        assert_eq!(response.status, 200);
        let body: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        let points = body["result"].as_array().unwrap();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0]["payload"]["rank"], serde_json::json!(2));
    }

    #[test]
    fn query_fusion_prefetch_expands_channel_window_for_offset() {
        let ctx = setup();
        create_dense_collection(&ctx, "fusion_page");
        upsert_points(
            &ctx,
            "fusion_page",
            serde_json::json!([
                {"id": "dense-a", "vector": {"dense": [1.0, 0.0, 0.0]}, "payload": {"rank": 1}},
                {"id": "dense-b", "vector": {"dense": [0.9, 0.1, 0.0]}, "payload": {"rank": 2}},
                {"id": "dense-c", "vector": {"dense": [0.0, 1.0, 0.0]}, "payload": {"rank": 3}}
            ]),
        );

        let (status, body) = query_points_raw(
            &ctx,
            "fusion_page",
            serde_json::json!({
                "prefetch": [{
                    "using": "dense",
                    "query": [1.0, 0.0, 0.0],
                    "limit": 1
                }],
                "query": {"fusion": "rrf"},
                "limit": 1,
                "offset": 1,
                "with_payload": true
            }),
        );

        assert_eq!(status, 200, "query failed: {body}");
        let points = body["result"]["points"].as_array().unwrap();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0]["payload"]["rank"], serde_json::json!(2));
    }

    /// Helper: create a collection via handle_create_collection using the
    /// provided JSON body and return the resulting SpindleConfig for the
    /// named vector "dense".
    fn create_and_get_spindle(
        ctx: &TestContext,
        name: &str,
        body: sonic_rs::Value,
    ) -> SpindleConfig {
        let bytes = sonic_rs::to_vec(&body).unwrap();
        let input = make_input(
            ctx,
            "PUT",
            &format!("/collections/{}", name),
            bytes,
            HashMap::from([("name".into(), name.into())]),
        );
        let mut resp = Response::new();
        handle_create_collection(&input, &mut resp).unwrap();
        assert!(
            resp.status == 200 || resp.status == 201,
            "create failed: {}",
            String::from_utf8_lossy(&resp.body)
        );
        let storage = ctx.collections.get_collection(name).unwrap();
        storage
            .named_vectors
            .get_config("dense")
            .expect("named vector 'dense' missing")
            .spindle
    }

    #[test]
    #[serial]
    fn quantization_none_clamps_rescore_and_oversampling() {
        // F1: when no quantization is requested, the stored SpindleConfig
        // must be internally consistent — no inherited oversampling:16 or
        // rescore:true from SpindleConfig::default().
        let previous = std::env::var_os("HELIX_SPINDLE_MODE");
        unsafe {
            std::env::remove_var("HELIX_SPINDLE_MODE");
        }

        let ctx = setup();
        let spindle = create_and_get_spindle(
            &ctx,
            "none_default",
            sonic_rs::json!({
                "vectors": {"dense": {"size": 3, "distance": "Cosine"}}
            }),
        );
        assert_eq!(spindle.mode, SpindleMode::None);
        assert!(!spindle.keep_original);
        assert!(!spindle.rescore);
        assert_eq!(spindle.oversampling, 1);

        unsafe {
            match previous {
                Some(value) => std::env::set_var("HELIX_SPINDLE_MODE", value),
                None => std::env::remove_var("HELIX_SPINDLE_MODE"),
            }
        }
    }

    #[test]
    fn quantization_env_default_compact_scalar_config() {
        let spindle = default_spindle_config_for_env(SpindleMode::ScalarInt8, false, false, 1);

        assert_eq!(spindle.mode, SpindleMode::ScalarInt8);
        assert!(!spindle.keep_original);
        assert!(!spindle.rescore);
        assert_eq!(spindle.oversampling, 1);
    }

    #[test]
    fn quantization_env_default_can_retain_originals_for_rescore() {
        let spindle = default_spindle_config_for_env(SpindleMode::TurboProd, true, true, 8);

        assert_eq!(spindle.mode, SpindleMode::TurboProd);
        assert!(spindle.keep_original);
        assert!(spindle.rescore);
        assert_eq!(spindle.oversampling, 8);
    }

    #[test]
    #[serial]
    fn get_collection_reports_env_default_quantization() {
        let previous_mode = std::env::var_os("HELIX_SPINDLE_MODE");
        let previous_keep = std::env::var_os("HELIX_SPINDLE_KEEP_ORIGINAL");
        let previous_rescore = std::env::var_os("HELIX_SPINDLE_RESCORE");
        unsafe {
            std::env::set_var("HELIX_SPINDLE_MODE", "scalar_int8");
            std::env::set_var("HELIX_SPINDLE_KEEP_ORIGINAL", "false");
            std::env::set_var("HELIX_SPINDLE_RESCORE", "false");
        }

        let ctx = setup();
        let _spindle = create_and_get_spindle(
            &ctx,
            "env_quant_info",
            sonic_rs::json!({
                "vectors": {"dense": {"size": 3, "distance": "Cosine"}}
            }),
        );
        let input = make_input(
            &ctx,
            "GET",
            "/collections/env_quant_info",
            vec![],
            HashMap::from([("name".into(), "env_quant_info".into())]),
        );
        let mut resp = Response::new();
        handle_get_collection(&input, &mut resp).unwrap();
        assert_eq!(resp.status, 200);
        let body: serde_json::Value = serde_json::from_slice(&resp.body).unwrap();
        assert_eq!(
            body["result"]["config"]["params"]["vectors"]["dense"]["quantization_config"]["scalar"]
                ["type"],
            "int8"
        );
        assert!(body["result"]["config"]["params"]["vectors"]["dense"]["quantization"].is_null());
        assert_eq!(
            body["result"]["config"]["metadata"]["helix_quantization"]["dense"]["mode"],
            "scalar_int8"
        );
        assert_eq!(
            body["result"]["config"]["metadata"]["helix_quantization"]["dense"]["keep_original"],
            false
        );

        unsafe {
            match previous_mode {
                Some(value) => std::env::set_var("HELIX_SPINDLE_MODE", value),
                None => std::env::remove_var("HELIX_SPINDLE_MODE"),
            }
            match previous_keep {
                Some(value) => std::env::set_var("HELIX_SPINDLE_KEEP_ORIGINAL", value),
                None => std::env::remove_var("HELIX_SPINDLE_KEEP_ORIGINAL"),
            }
            match previous_rescore {
                Some(value) => std::env::set_var("HELIX_SPINDLE_RESCORE", value),
                None => std::env::remove_var("HELIX_SPINDLE_RESCORE"),
            }
        }
    }

    #[test]
    fn quantization_collection_level_qdrant_scalar_applies() {
        // F2 (existing): top-level quantization_config with {"scalar": {...}}
        // is the Qdrant REST wire format. Must translate to ScalarInt8.
        let ctx = setup();
        let spindle = create_and_get_spindle(
            &ctx,
            "qd_scalar",
            sonic_rs::json!({
                "vectors": {"dense": {"size": 3, "distance": "Cosine"}},
                "quantization_config": {
                    "scalar": {"type": "int8", "quantile": 0.99, "always_ram": true}
                }
            }),
        );
        assert_eq!(spindle.mode, SpindleMode::ScalarInt8);
        assert!(!spindle.keep_original);
        assert!(!spindle.rescore);
        assert_eq!(spindle.oversampling, 1);
    }

    #[test]
    fn quantization_explicit_originals_and_rescore_are_preserved() {
        // Compact storage is the default for quantized collections, but
        // callers that explicitly request DB-side exact rerank still get it.
        let ctx = setup();
        let spindle = create_and_get_spindle(
            &ctx,
            "explicit_rescore",
            sonic_rs::json!({
                "vectors": {
                    "dense": {
                        "size": 3,
                        "distance": "Cosine",
                        "quantization": {
                            "mode": "turbo_prod",
                            "keep_original": true,
                            "rescore": true,
                            "oversampling": 8
                        }
                    }
                }
            }),
        );
        assert_eq!(spindle.mode, SpindleMode::TurboProd);
        assert!(spindle.keep_original);
        assert!(spindle.rescore);
        assert_eq!(spindle.oversampling, 8);
    }

    #[test]
    fn quantization_per_vector_qdrant_native_applies() {
        // F2 extension: qdrant-client puts quantization inside VectorParams,
        // not at the top level. VectorParamsInput.quantization_config is the
        // new field that catches those payloads. Must still reach Helix.
        let ctx = setup();
        let spindle = create_and_get_spindle(
            &ctx,
            "qd_scalar_pv",
            sonic_rs::json!({
                "vectors": {
                    "dense": {
                        "size": 3,
                        "distance": "Cosine",
                        "quantization_config": {"binary": {"always_ram": true}}
                    }
                }
            }),
        );
        assert_eq!(spindle.mode, SpindleMode::BinarySign);
    }

    #[test]
    fn quantization_per_vector_helix_native_wins_over_qdrant() {
        // Resolution order: helix-native `quantization` beats qdrant-native
        // `quantization_config` when both are present on the same vector.
        let ctx = setup();
        let spindle = create_and_get_spindle(
            &ctx,
            "qd_conflict",
            sonic_rs::json!({
                "vectors": {
                    "dense": {
                        "size": 3,
                        "distance": "Cosine",
                        "quantization": {"mode": "scalar"},
                        "quantization_config": {"binary": {}}
                    }
                }
            }),
        );
        assert_eq!(spindle.mode, SpindleMode::ScalarInt8);
    }

    #[test]
    fn quantization_per_vector_wins_over_collection_level() {
        // Resolution order: a per-vector quantization field (either shape)
        // must win over the collection-level fallback.
        let ctx = setup();
        let spindle = create_and_get_spindle(
            &ctx,
            "qd_override",
            sonic_rs::json!({
                "vectors": {
                    "dense": {
                        "size": 3,
                        "distance": "Cosine",
                        "quantization_config": {"scalar": {"type": "int8"}}
                    }
                },
                "quantization_config": {"binary": {}}
            }),
        );
        assert_eq!(spindle.mode, SpindleMode::ScalarInt8);
    }

    /// Drive a scroll request and return the parsed JSON body.
    /// The caller supplies the `with_payload` value verbatim so we can prove
    /// every Qdrant-compatible shape is accepted (bool, list, object).
    fn run_scroll(
        ctx: &TestContext,
        name: &str,
        with_payload: serde_json::Value,
    ) -> serde_json::Value {
        let body = serde_json::to_vec(&serde_json::json!({
            "limit": 10,
            "with_payload": with_payload,
        }))
        .unwrap();
        let input = make_input(
            ctx,
            "POST",
            &format!("/collections/{}/points/scroll", name),
            body,
            HashMap::from([("name".into(), name.into())]),
        );
        let mut response = Response::new();
        handle_scroll_points(&input, &mut response).unwrap();
        assert_eq!(
            response.status,
            200,
            "scroll must return 200 for with_payload={:?}, body={}",
            with_payload,
            String::from_utf8_lossy(&response.body),
        );
        serde_json::from_slice(&response.body).unwrap()
    }

    #[test]
    fn scroll_with_payload_accepts_qdrant_compat_shapes() {
        // Regression: qdrant-client sends `with_payload: {"include": [...]}` for
        // PayloadSelectorInclude. Helix used to reject it as
        // `Invalid JSON: invalid type: map, expected a boolean`, surfacing as
        // 400s on every scroll-with-payload from the upload-service. The new
        // WithPayload deserializer must accept bool, list, and object shapes.
        let ctx = setup();
        create_dense_collection(&ctx, "scroll_compat");
        upsert_points(
            &ctx,
            "scroll_compat",
            serde_json::json!([{
                "id": "doc-1",
                "vector": {"dense": [0.1, 0.2, 0.3]},
                "payload": {
                    "path": "src/main.rs",
                    "metadata": {"repo": "demo", "branch": "main"},
                    "secret": "leak-me-not"
                }
            }]),
        );

        // Shape 1: bool true → full payload
        let bool_true = run_scroll(&ctx, "scroll_compat", serde_json::json!(true));
        let payload = &bool_true["result"]["points"][0]["payload"];
        assert_eq!(payload["path"], "src/main.rs");
        assert_eq!(payload["secret"], "leak-me-not");

        // Shape 2: bool false → no payload
        let bool_false = run_scroll(&ctx, "scroll_compat", serde_json::json!(false));
        assert!(
            bool_false["result"]["points"][0]["payload"].is_null()
                || bool_false["result"]["points"][0].get("payload").is_none(),
        );

        // Shape 3: array → include-only those top-level keys
        let array_form = run_scroll(
            &ctx,
            "scroll_compat",
            serde_json::json!(["path", "metadata"]),
        );
        let payload = &array_form["result"]["points"][0]["payload"];
        assert_eq!(payload["path"], "src/main.rs");
        assert_eq!(payload["metadata"]["repo"], "demo");
        assert!(
            payload.get("secret").is_none() || payload["secret"].is_null(),
            "array selector must drop excluded keys",
        );

        // Shape 4: object {include: [...]} — qdrant-client PayloadSelectorInclude
        let object_include = run_scroll(
            &ctx,
            "scroll_compat",
            serde_json::json!({"include": ["metadata.repo"]}),
        );
        let payload = &object_include["result"]["points"][0]["payload"];
        assert_eq!(payload["metadata"]["repo"], "demo");
        assert!(
            payload.get("path").is_none() || payload["path"].is_null(),
            "object include selector must drop excluded keys",
        );
        assert!(payload.get("secret").is_none() || payload["secret"].is_null(),);

        // Shape 5: object {exclude: [...]} — explicit drop list
        let object_exclude = run_scroll(
            &ctx,
            "scroll_compat",
            serde_json::json!({"exclude": ["secret"]}),
        );
        let payload = &object_exclude["result"]["points"][0]["payload"];
        assert_eq!(payload["path"], "src/main.rs");
        assert_eq!(payload["metadata"]["repo"], "demo");
        assert!(payload.get("secret").is_none() || payload["secret"].is_null(),);
    }

    #[test]
    fn wal_ahead_pending_fuses_get_scroll_count_and_delete() {
        enable_wal_ahead_pending_for_test();
        let ctx = setup();
        create_dense_collection(&ctx, "wal_fuse_rw");

        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "points": [
                {
                    "id": "wal-1",
                    "vector": {"dense": [1.0, 0.0, 0.0]},
                    "payload": {"repo": "context-engine", "seq": 1}
                },
                {
                    "id": "wal-2",
                    "vector": {"dense": [0.0, 1.0, 0.0]},
                    "payload": {"repo": "context-engine", "seq": 2}
                }
            ]
        }))
        .unwrap();
        record_wal_ahead_points("/collections/wal_fuse_rw/points", &body);

        let get_body = sonic_rs::to_vec(&sonic_rs::json!({
            "ids": ["wal-1"],
            "with_payload": true,
            "with_vector": true
        }))
        .unwrap();
        let get_input = make_input(
            &ctx,
            "POST",
            "/collections/wal_fuse_rw/points",
            get_body,
            HashMap::from([("name".into(), "wal_fuse_rw".into())]),
        );
        let mut get_response = Response::new();
        handle_get_points(&get_input, &mut get_response).unwrap();
        assert_eq!(get_response.status, 200);
        let get_json: serde_json::Value = serde_json::from_slice(&get_response.body).unwrap();
        assert_eq!(
            get_json["result"][0]["payload"]["seq"],
            serde_json::json!(1)
        );
        assert_eq!(
            get_json["result"][0]["vector"]["dense"],
            serde_json::json!([1.0, 0.0, 0.0])
        );

        let scroll = run_scroll(&ctx, "wal_fuse_rw", serde_json::json!(true));
        assert_eq!(scroll["result"]["points"].as_array().unwrap().len(), 2);

        let count_input = make_input(
            &ctx,
            "POST",
            "/collections/wal_fuse_rw/points/count",
            Vec::new(),
            HashMap::from([("name".into(), "wal_fuse_rw".into())]),
        );
        let mut count_response = Response::new();
        handle_count_points(&count_input, &mut count_response).unwrap();
        let count_json: serde_json::Value = serde_json::from_slice(&count_response.body).unwrap();
        assert_eq!(count_json["result"]["count"], serde_json::json!(2));

        let delete_one_body = sonic_rs::to_vec(&sonic_rs::json!({"points": ["wal-1"]})).unwrap();
        let delete_one_input = make_input(
            &ctx,
            "POST",
            "/collections/wal_fuse_rw/points/delete",
            delete_one_body,
            HashMap::from([("name".into(), "wal_fuse_rw".into())]),
        );
        let mut delete_one_response = Response::new();
        handle_delete_points(&delete_one_input, &mut delete_one_response).unwrap();
        assert_eq!(delete_one_response.status, 200);
        assert!(
            pending_point("wal_fuse_rw", point_id_to_u128(&serde_json::json!("wal-1"))).is_none()
        );
        assert!(
            pending_point("wal_fuse_rw", point_id_to_u128(&serde_json::json!("wal-2"))).is_some()
        );

        let delete_filter_body = sonic_rs::to_vec(&sonic_rs::json!({
            "filter": {
                "must": [{"key": "repo", "match": {"value": "context-engine"}}]
            }
        }))
        .unwrap();
        let delete_filter_input = make_input(
            &ctx,
            "POST",
            "/collections/wal_fuse_rw/points/delete",
            delete_filter_body,
            HashMap::from([("name".into(), "wal_fuse_rw".into())]),
        );
        let mut delete_filter_response = Response::new();
        handle_delete_points(&delete_filter_input, &mut delete_filter_response).unwrap();
        assert_eq!(delete_filter_response.status, 200);
        assert!(pending_points_for_collection("wal_fuse_rw").is_empty());
    }

    #[test]
    fn wal_ahead_pending_fuses_search_query_and_hybrid_paths() {
        enable_wal_ahead_pending_for_test();
        let ctx = setup();
        create_collection_with_sparse(&ctx, "wal_fuse_search");

        let body = sonic_rs::to_vec(&sonic_rs::json!({
            "points": [{
                "id": "wal-search-1",
                "vector": {
                    "dense": [1.0, 0.0, 0.0],
                    "lex_sparse": {"indices": [7], "values": [3.0]}
                },
                "payload": {"repo": "context-engine", "path": "pending.rs"}
            }]
        }))
        .unwrap();
        record_wal_ahead_points("/collections/wal_fuse_search/points", &body);

        let search_input = make_input(
            &ctx,
            "POST",
            "/collections/wal_fuse_search/points/search",
            sonic_rs::to_vec(&sonic_rs::json!({
                "vector": {"name": "dense", "vector": [1.0, 0.0, 0.0]},
                "limit": 5,
                "with_payload": true
            }))
            .unwrap(),
            HashMap::from([("name".into(), "wal_fuse_search".into())]),
        );
        let mut search_response = Response::new();
        handle_search_points(&search_input, &mut search_response).unwrap();
        assert_eq!(search_response.status, 200);
        let search_json: serde_json::Value = serde_json::from_slice(&search_response.body).unwrap();
        assert_eq!(
            search_json["result"][0]["payload"]["path"],
            serde_json::json!("pending.rs")
        );

        let (query_status, query_json) = query_points_raw(
            &ctx,
            "wal_fuse_search",
            serde_json::json!({
                "query": [1.0, 0.0, 0.0],
                "using": "dense",
                "limit": 5,
                "with_payload": true
            }),
        );
        assert_eq!(query_status, 200);
        assert_eq!(
            query_json["result"]["points"][0]["payload"]["path"],
            serde_json::json!("pending.rs")
        );

        let (hybrid_status, hybrid_json) = hybrid_query_points_raw(
            &ctx,
            "wal_fuse_search",
            serde_json::json!({
                "dense": [{"using": "dense", "query": [1.0, 0.0, 0.0], "limit": 5}],
                "sparse": [{"using": "lex_sparse", "query": {"indices": [7], "values": [1.0]}, "limit": 5}],
                "limit": 5,
                "with_payload": true,
                "with_channel_points": true
            }),
        );
        assert_eq!(hybrid_status, 200);
        assert_eq!(
            hybrid_json["result"]["points"][0]["payload"]["path"],
            serde_json::json!("pending.rs")
        );
        assert!(!hybrid_json["result"]["channels"]["sparse"][0]["points"]
            .as_array()
            .unwrap()
            .is_empty());

        clear_wal_ahead_points("/collections/wal_fuse_search/points", &body);
    }

    /// Full Qdrant upsert → search/retrieve round-trip on the in-memory LSM
    /// backend. Proves the Qdrant REST write path persists vectors+payload to
    /// LSM (via `apply_upsert_points_lsm_chunk`) and the read path reads them
    /// back through the backend seam (`read_borrowed`/`get_node`), not LMDB.
    #[test]
    fn qdrant_api_round_trip_on_lsm() {
        use crate::helix_engine::storage_core::backend::BackendKind;

        let ctx = setup_with_config(Config::default().with_lsm_in_memory());
        create_dense_collection_lsm(&ctx, "lsm_round_trip");

        // The per-collection storage must have selected the in-memory LSM
        // backend; otherwise this test would silently pass on LMDB.
        let storage = ctx.collections.get_collection("lsm_round_trip").unwrap();
        assert_eq!(
            storage.backend.kind(),
            BackendKind::Lsm,
            "collection storage must run on the LSM backend for this test to be meaningful"
        );

        let update_body = sonic_rs::to_vec(&sonic_rs::json!({
            "sparse_vectors": {
                "lex_sparse": {"modifier": "idf"}
            }
        }))
        .unwrap();
        let update_input = make_input(
            &ctx,
            "PATCH",
            "/collections/lsm_round_trip",
            update_body,
            HashMap::from([("name".into(), "lsm_round_trip".into())]),
        );
        let mut update_response = Response::new();
        handle_update_collection(&update_input, &mut update_response).unwrap();
        assert_eq!(
            update_response.status,
            200,
            "update-collection add sparse failed on LSM: {}",
            String::from_utf8_lossy(&update_response.body)
        );

        // Upsert two points with distinct dense vectors and payloads through
        // the real Qdrant write handler.
        upsert_mixed_points(
            &ctx,
            "lsm_round_trip",
            serde_json::json!([
                {
                    "id": "doc-x",
                    "vector": {
                        "dense": [1.0, 0.0, 0.0],
                        "lex_sparse": {"indices": [7, 11], "values": [3.0, 1.0]}
                    },
                    "payload": {"repo": "context-engine", "path": "src/x.rs"}
                },
                {
                    "id": "doc-y",
                    "vector": {
                        "dense": [0.0, 1.0, 0.0],
                        "lex_sparse": {"indices": [7, 13], "values": [1.0, 5.0]}
                    },
                    "payload": {"repo": "context-engine", "path": "src/y.rs"}
                }
            ]),
        );

        // (a) Filtered nearest-neighbour search. The query is closest to
        // doc-x; the filter forces the backend-seam flat scan over the
        // LSM-stored vectors. If the upsert did not persist to LSM, the flat
        // segment would be empty and this would return nothing.
        let search_body = sonic_rs::to_vec(&sonic_rs::json!({
            "vector": {"name": "dense", "vector": [1.0, 0.0, 0.0]},
            "limit": 1,
            "with_payload": true,
            "filter": {
                "must": [{"key": "repo", "match": {"value": "context-engine"}}]
            }
        }))
        .unwrap();
        let search_input = make_input(
            &ctx,
            "POST",
            "/collections/lsm_round_trip/points/search",
            search_body,
            HashMap::from([("name".into(), "lsm_round_trip".into())]),
        );
        let mut search_response = Response::new();
        handle_search_points(&search_input, &mut search_response).unwrap();
        assert_eq!(
            search_response.status,
            200,
            "search failed on LSM: {}",
            String::from_utf8_lossy(&search_response.body)
        );
        let search_json: serde_json::Value = serde_json::from_slice(&search_response.body).unwrap();
        let search_points = search_json["result"].as_array().unwrap();
        assert_eq!(
            search_points.len(),
            1,
            "expected exactly one nearest point from LSM, got: {search_json}"
        );
        assert_eq!(
            search_points[0]["payload"]["path"],
            serde_json::json!("src/x.rs"),
            "nearest point must be doc-x, read back from LSM"
        );

        let sparse_search_body = sonic_rs::to_vec(&sonic_rs::json!({
            "vector": {
                "name": "lex_sparse",
                "vector": {"indices": [13], "values": [1.0]}
            },
            "limit": 1,
            "with_payload": true
        }))
        .unwrap();
        let sparse_search_input = make_input(
            &ctx,
            "POST",
            "/collections/lsm_round_trip/points/search",
            sparse_search_body,
            HashMap::from([("name".into(), "lsm_round_trip".into())]),
        );
        let mut sparse_search_response = Response::new();
        handle_search_points(&sparse_search_input, &mut sparse_search_response).unwrap();
        assert_eq!(
            sparse_search_response.status,
            200,
            "sparse search failed on LSM: {}",
            String::from_utf8_lossy(&sparse_search_response.body)
        );
        let sparse_search_json: serde_json::Value =
            serde_json::from_slice(&sparse_search_response.body).unwrap();
        assert_eq!(
            sparse_search_json["result"][0]["payload"]["path"],
            serde_json::json!("src/y.rs"),
            "sparse search must read postings from LSM"
        );

        let (query_status, query_json) = query_points_raw(
            &ctx,
            "lsm_round_trip",
            serde_json::json!({
                "query": {"indices": [13], "values": [1.0]},
                "using": "lex_sparse",
                "limit": 1,
                "with_payload": true
            }),
        );
        assert_eq!(query_status, 200, "sparse query failed: {query_json}");
        assert_eq!(
            query_json["result"]["points"][0]["payload"]["path"],
            "src/y.rs"
        );

        let (hybrid_status, hybrid_json) = hybrid_query_points_raw(
            &ctx,
            "lsm_round_trip",
            serde_json::json!({
                "dense": [{"using": "dense", "query": [1.0, 0.0, 0.0], "limit": 2}],
                "sparse": [{"using": "lex_sparse", "query": {"indices": [13], "values": [1.0]}, "limit": 2}],
                "limit": 2,
                "with_payload": true,
                "with_channel_points": true
            }),
        );
        assert_eq!(hybrid_status, 200, "hybrid query failed: {hybrid_json}");
        assert!(!hybrid_json["result"]["channels"]["sparse"][0]["points"]
            .as_array()
            .unwrap()
            .is_empty());

        // (b) Retrieve doc-y by id. The payload is read through the backend
        // seam (`get_node` → `backend.get_with_heed`), proving the second point
        // also persisted to and reads back from LSM.
        let get_body = sonic_rs::to_vec(&sonic_rs::json!({
            "ids": ["doc-y"],
            "with_payload": true,
            "with_vector": true
        }))
        .unwrap();
        let get_input = make_input(
            &ctx,
            "POST",
            "/collections/lsm_round_trip/points",
            get_body,
            HashMap::from([("name".into(), "lsm_round_trip".into())]),
        );
        let mut get_response = Response::new();
        handle_get_points(&get_input, &mut get_response).unwrap();
        assert_eq!(get_response.status, 200);
        let get_json: serde_json::Value = serde_json::from_slice(&get_response.body).unwrap();
        let retrieved = get_json["result"].as_array().unwrap();
        assert_eq!(
            retrieved.len(),
            1,
            "retrieve by id must return the point from LSM, got: {get_json}"
        );
        assert_eq!(
            retrieved[0]["payload"]["path"],
            serde_json::json!("src/y.rs"),
            "retrieved payload must round-trip through LSM"
        );
        // A dense vector column IS hydrated for the retrieved point. We assert
        // presence/shape, not exact bytes: on LSM the pure-backend flat insert
        // does not populate the mmap sidecar, and `get_vector`'s mmap fast path
        // (ordinal → mmap slot) returns an unpopulated slot rather than falling
        // through to the backend B-tree. Populating the mmap sidecar on LSM flat
        // inserts is an orthogonal follow-up (see the same caveat on the
        // graph_core LSM end-to-end test); the persistence proof here is the
        // search-by-vector hit (a) and the payload round-trip above.
        let dense = retrieved[0]["vector"]["dense"].as_array().unwrap();
        assert_eq!(
            dense.len(),
            3,
            "retrieved dense vector must have the configured dimensionality"
        );
        assert_eq!(
            retrieved[0]["vector"]["lex_sparse"]["indices"],
            serde_json::json!([7, 13]),
            "retrieved sparse vector must hydrate through the LSM backend"
        );

        let set_payload_body = sonic_rs::to_vec(&sonic_rs::json!({
            "points": ["doc-y"],
            "payload": {"tag": "patched"}
        }))
        .unwrap();
        let set_payload_input = make_input(
            &ctx,
            "POST",
            "/collections/lsm_round_trip/points/payload",
            set_payload_body,
            HashMap::from([("name".into(), "lsm_round_trip".into())]),
        );
        let mut set_payload_response = Response::new();
        handle_set_payload(&set_payload_input, &mut set_payload_response).unwrap();
        assert_eq!(set_payload_response.status, 200);

        let mut patched_get_response = Response::new();
        handle_get_points(&get_input, &mut patched_get_response).unwrap();
        assert_eq!(patched_get_response.status, 200);
        let patched_get_json: serde_json::Value =
            serde_json::from_slice(&patched_get_response.body).unwrap();
        assert_eq!(
            patched_get_json["result"][0]["payload"]["tag"],
            serde_json::json!("patched"),
            "set_payload must update point payload through the LSM backend"
        );

        // (b2) Retrieving a genuinely-absent id must still succeed with an empty
        // result (NodeNotFound is a skip, not an error). This guards the #7 fix's
        // error branch: only NodeNotFound is swallowed; a real backend read error
        // would now propagate instead of being masked as an empty 200.
        let missing_body = sonic_rs::to_vec(&sonic_rs::json!({
            "ids": ["doc-does-not-exist"],
            "with_payload": true
        }))
        .unwrap();
        let missing_input = make_input(
            &ctx,
            "POST",
            "/collections/lsm_round_trip/points",
            missing_body,
            HashMap::from([("name".into(), "lsm_round_trip".into())]),
        );
        let mut missing_response = Response::new();
        handle_get_points(&missing_input, &mut missing_response).unwrap();
        assert_eq!(missing_response.status, 200);
        let missing_json: serde_json::Value =
            serde_json::from_slice(&missing_response.body).unwrap();
        assert!(
            missing_json["result"].as_array().unwrap().is_empty(),
            "absent id must be skipped (empty result), got: {missing_json}"
        );

        // (c) Scroll must enumerate BOTH upserted points on the LSM backend. The
        // scroll/scan primary plan walks Namespace::Nodes through the backend
        // seam; before the fix it read the empty heed `nodes_db` directly and
        // returned `{"points": []}` even though the data was in SlateDB.
        let scroll = run_scroll(&ctx, "lsm_round_trip", serde_json::json!(true));
        let scrolled = scroll["result"]["points"].as_array().unwrap();
        assert_eq!(
            scrolled.len(),
            2,
            "scroll must enumerate both upserted points on LSM, got: {scroll}"
        );
        let mut paths: Vec<String> = scrolled
            .iter()
            .map(|p| {
                p["payload"]["path"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string()
            })
            .collect();
        paths.sort();
        assert_eq!(
            paths,
            vec!["src/x.rs".to_string(), "src/y.rs".to_string()],
            "scroll must return both points' payloads read back from LSM"
        );

        let count_body = sonic_rs::to_vec(&sonic_rs::json!({
            "filter": {"must": [{"key": "path", "match": {"value": "src/x.rs"}}]}
        }))
        .unwrap();
        let count_input = make_input(
            &ctx,
            "POST",
            "/collections/lsm_round_trip/points/count",
            count_body,
            HashMap::from([("name".into(), "lsm_round_trip".into())]),
        );
        let mut count_response = Response::new();
        handle_count_points(&count_input, &mut count_response).unwrap();
        assert_eq!(count_response.status, 200);
        let count_json: serde_json::Value = serde_json::from_slice(&count_response.body).unwrap();
        assert_eq!(count_json["result"]["count"], serde_json::json!(1));

        storage
            .create_payload_index("path", PayloadIndexSchema::Keyword)
            .unwrap();
        let delete_body = sonic_rs::to_vec(&sonic_rs::json!({
            "filter": {"must": [{"key": "path", "match": {"value": "src/x.rs"}}]}
        }))
        .unwrap();
        let delete_input = make_input(
            &ctx,
            "POST",
            "/collections/lsm_round_trip/points/delete",
            delete_body,
            HashMap::from([("name".into(), "lsm_round_trip".into())]),
        );
        let mut delete_response = Response::new();
        handle_delete_points(&delete_input, &mut delete_response).unwrap();
        assert_eq!(delete_response.status, 200);
        let r = storage.backend.begin_read().unwrap();
        let stale_ids = storage
            .get_nodes_by_payload_value_be(&r, "path", &Value::String("src/x.rs".into()))
            .unwrap();
        assert!(
            stale_ids.is_empty(),
            "drop_node_be must de-index payload index entries on delete, got stale ids: {stale_ids:?}"
        );
    }
}
