//! One live copy per point and dense space on the LSM backend: a write that
//! replaces or deletes a point leaves no older copy for search, the by-id
//! read or a merge to pick, also while a merge is building from that copy.
//! Rows written before that held (a second copy, or the row of a point that
//! is gone) are no hit, and go when their segment merges.

use super::tests::{
    create_collection_with_sparse, create_dense_collection_lsm, make_input, setup_with_config,
    TestContext,
};
use super::*;
use crate::helix_engine::graph_core::config::Config;
use crate::helix_engine::storage_core::backend::{LsmRole, LsmStorage, StorageBackendConfig};
use crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store;
use crate::helix_engine::storage_core::backend_lsm::{
    allow_lsm_blocking, allow_lsm_blocking_cancellable, LsmReadCancellation,
};
use crate::helix_engine::storage_core::replication::test_support::{
    optimize, optimize_with, regraph, row_keeps_a_digest, seal_and_index, seal_tail, with_hook,
};
use crate::helix_engine::vector_core::named_vectors::NamedVectorManager;
use serde_json::json;
use std::rc::Rc;
use std::sync::Arc;

/// The point every test rewrites or deletes.
const X: u64 = 1;
/// A point in the mutable tail while a merge runs.
const TAIL: u64 = 1_000;
/// An id no point has: a dense row under it is one a merge left behind.
const GONE: u64 = 77;
const OLD: [f32; 3] = [1.0, 0.0, 0.0];
/// Cosine 0.99875 to `OLD`: near enough that a search meets both copies, far
/// enough to tell which one it scored.
const NEW: [f32; 3] = [0.998_752_3, 0.049_937_62, 0.0];
const FAR: [f32; 3] = [0.0, 0.0, 1.0];

/// Sets an environment variable for a test and restores it after.
struct EnvVar(&'static str, Option<String>);

impl EnvVar {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        std::env::set_var(key, value);
        Self(key, previous)
    }
}

impl Drop for EnvVar {
    fn drop(&mut self) {
        match &self.1 {
            Some(value) => std::env::set_var(self.0, value),
            None => std::env::remove_var(self.0),
        }
    }
}

fn same(a: &[f32], b: &[f32]) -> bool {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    a.len() == b.len() && dot / (norm(a) * norm(b)) > 0.9999
}

/// The score of an exact match, as a similarity (1) or a distance (0); the
/// stale copy scores 0.99875 or 0.00125.
fn exact(score: f64) -> bool {
    (score - 1.0).abs() < 5e-4 || score.abs() < 5e-4
}

/// Ids `merge_ready` writes.
fn merge_ids() -> HashSet<u128> {
    let sources = NamedVectorManager::max_indexed_segments_target() as u64 + 1;
    (1..=sources).chain([TAIL]).map(u128::from).collect()
}

/// `(segment, role, vector)` of every stored copy, by point id.
type Rows = HashMap<u128, Vec<(String, DenseVectorSegmentRole, Vec<f32>)>>;

/// The payload of write number `write`, with what a client's JSON can hold:
/// nested values, a null, a float and an integer above `i64`.
fn payload_of(write: u64) -> serde_json::Value {
    json!({"write": write, "ratio": 0.5, "big": 18446744073709551615u64,
        "list": [1, 2.5, "a", null], "nested": {"k": [1, {"z": 1.0}]}})
}

/// What a stored row keeps of the payload it was written with, `payload`.
fn shape(fields: &HashMap<String, Value>, payload: &serde_json::Value) -> &'static str {
    let payload: HashMap<String, serde_json::Value> =
        serde_json::from_value(payload.clone()).unwrap();
    if row_keeps_a_digest(fields) {
        "digest"
    } else if *fields == json_payload_to_value_map(&payload) {
        "payload"
    } else if fields.is_empty() {
        "nothing"
    } else {
        "another payload"
    }
}

/// An LSM writer collection with one 3-wide cosine space, `dense`, whose
/// optimizer runs only when the test calls it.
struct Points {
    ctx: TestContext,
    name: &'static str,
    storage: Arc<HelixGraphStorage>,
}

impl Points {
    fn new(name: &'static str) -> Rc<Self> {
        let mut config = Config::default().with_lsm_in_memory();
        // 0 keeps an upsert from queueing an index job of its own.
        config.vector_config.flat_scan_threshold = Some(0);
        let ctx = setup_with_config(config);
        create_dense_collection_lsm(&ctx, name);
        Self::open(ctx, name)
    }

    /// The collection as `ctx`, a writer or a reader replica, has it.
    fn open(ctx: TestContext, name: &'static str) -> Rc<Self> {
        let storage = ctx.collections.get_collection(name).unwrap();
        Rc::new(Self { ctx, name, storage })
    }

    /// One handler call; `result` of its 200 response.
    fn call(&self, route: &str, body: serde_json::Value) -> serde_json::Value {
        self.try_call(route, body).unwrap()
    }

    fn try_call(&self, route: &str, body: serde_json::Value) -> Result<serde_json::Value, String> {
        let name = self.name;
        let (method, path) = match route {
            "upsert" => ("PUT", format!("/collections/{name}/points")),
            "get" => ("POST", format!("/collections/{name}/points")),
            other => ("POST", format!("/collections/{name}/points/{other}")),
        };
        let input = make_input(
            &self.ctx,
            method,
            &path,
            serde_json::to_vec(&body).unwrap(),
            HashMap::from([("name".to_string(), name.to_string())]),
        );
        let mut response = Response::new();
        let mut handle = || match route {
            "upsert" => handle_upsert_points(&input, &mut response),
            "delete" => handle_delete_points(&input, &mut response),
            "get" => handle_get_points(&input, &mut response),
            "scroll" => handle_scroll_points(&input, &mut response),
            "search" => handle_search_points(&input, &mut response),
            "query" => handle_query_points(&input, &mut response),
            // On this thread, as in the other gateway tests.
            "hybrid_query" => with_hybrid_parallel_channels_for_test(false, || {
                handle_hybrid_query_points(&input, &mut response)
            }),
            _ => unreachable!(),
        };
        // A replica's reads run as the gateway runs a handler.
        if self.storage.backend.is_reader_replica() {
            allow_lsm_blocking(handle)
        } else {
            handle()
        }
        .map_err(|error| format!("{route}: {error}"))?;
        let text = String::from_utf8_lossy(&response.body);
        if response.status != 200 {
            return Err(format!("{route}: {} {text}", response.status));
        }
        let body: serde_json::Value = serde_json::from_str(&text).unwrap();
        Ok(body["result"].clone())
    }

    fn upsert(&self, points: &[(u64, [f32; 3])]) {
        let points: Vec<serde_json::Value> = points
            .iter()
            .map(|(id, vector)| json!({"id": id, "vector": {"dense": vector}}))
            .collect();
        self.call("upsert", json!({ "points": points }));
    }

    /// `X` with a payload that says which write it was.
    fn write_x(&self, vector: [f32; 3], write: u64) {
        self.write(X, vector, write);
    }

    fn write(&self, id: u64, vector: [f32; 3], write: u64) {
        let point = json!({"id": id, "vector": {"dense": vector}, "payload": payload_of(write)});
        self.call("upsert", json!({ "points": [point] }));
    }

    /// `X`'s payload as the by-id read, the scroll and the two searches
    /// return it, when `X` is the only point.
    fn payloads_of_x(&self, query: [f32; 3]) -> Vec<serde_json::Value> {
        let mut seen = vec![
            self.call("get", json!({"ids": [X], "with_payload": true}))[0]["payload"].clone(),
            self.call("scroll", json!({"limit": 1, "with_payload": true}))["points"][0]["payload"]
                .clone(),
        ];
        for route in ["search", "query"] {
            seen.push(self.found(route, query, 1).unwrap()[0]["payload"].clone());
        }
        seen
    }

    /// A dense row in the mutable tail with no point behind it, as a merge
    /// left the row of a point deleted while it was building.
    fn row_without_a_point(&self, id: u64, vector: [f32; 3]) {
        self.storage
            .with_write_backend(|w| {
                self.storage
                    .named_vectors
                    .dense_append_to_mutable_be(w, "dense", &vector, u128::from(id), HashMap::new())
                    .map(drop)
                    .map_err(GraphError::from)
            })
            .unwrap();
    }

    fn seal(&self) {
        seal_and_index(&self.storage, "dense");
    }

    fn segments(&self) -> Vec<String> {
        self.storage.named_vectors.list_dense_vector_spaces()["dense"]
            .segments
            .iter()
            .map(|segment| segment.physical_name.clone())
            .collect()
    }

    /// One indexed segment per id, one more than the optimizer keeps, so its
    /// next job merges the oldest; the first holds `X` = `OLD`. Then `TAIL`
    /// in the mutable tail. Returns the first segment's name.
    fn merge_ready(&self) -> String {
        let sources = NamedVectorManager::max_indexed_segments_target() as u64 + 1;
        for id in 1..=sources {
            self.upsert(&[(id, if id == X { OLD } else { FAR })]);
            self.seal();
        }
        self.upsert(&[(TAIL, FAR)]);
        self.segments().remove(0)
    }

    fn rows(&self) -> Rows {
        let mut rows = Rows::new();
        for (id, segment, role, vector, _) in self.copies() {
            rows.entry(id).or_default().push((segment, role, vector));
        }
        rows
    }

    /// `(point id, segment, role, vector, what the row keeps of its payload)`
    /// of every stored copy.
    #[allow(clippy::type_complexity)]
    fn copies(
        &self,
    ) -> Vec<(
        u128,
        String,
        DenseVectorSegmentRole,
        Vec<f32>,
        HashMap<String, Value>,
    )> {
        let spaces = self.storage.named_vectors.list_dense_vector_spaces();
        let read = self.storage.backend.begin_read().unwrap();
        let cores = self.storage.named_vectors.cores_read().unwrap();
        let mut copies = Vec::new();
        for segment in &spaces["dense"].segments {
            let Some(core) = cores.get(&segment.physical_name) else {
                continue;
            };
            for (id, vector, fields) in core.export_level_zero(&read).unwrap() {
                copies.push((
                    id,
                    segment.physical_name.clone(),
                    segment.role,
                    vector,
                    fields,
                ));
            }
        }
        copies
    }

    /// What each stored copy of `id` keeps of `payload`, by name.
    fn shapes(&self, id: u64, payload: &serde_json::Value) -> Vec<&'static str> {
        self.copies()
            .iter()
            .filter(|copy| copy.0 == u128::from(id))
            .map(|copy| shape(&copy.4, payload))
            .collect()
    }

    /// The first `limit` hits for `query` on `route`, with their payload.
    fn found(
        &self,
        route: &str,
        query: [f32; 3],
        limit: usize,
    ) -> Result<Vec<serde_json::Value>, String> {
        let result = self.try_call(
            route,
            match route {
                "search" => json!({"vector": {"name": "dense", "vector": query},
                    "limit": limit, "with_payload": true}),
                "query" => json!({"query": query, "using": "dense", "limit": limit,
                    "with_payload": true}),
                _ => json!({"dense": [{"query": query, "using": "dense", "limit": limit}],
                    "limit": limit, "with_payload": true, "with_channel_points": true}),
            },
        )?;
        let points = match route {
            "search" => &result,
            "query" => &result["points"],
            _ => &result["channels"]["dense"][0]["points"],
        };
        points
            .as_array()
            .cloned()
            .ok_or_else(|| format!("{route}: no points in {result}"))
    }

    /// `(id, score)` of every hit for `query` on `route`.
    fn hits(&self, route: &str, query: [f32; 3]) -> Result<Vec<(u128, f64)>, String> {
        Ok(self
            .found(route, query, 64)?
            .iter()
            .map(|point| {
                (
                    point_id_to_u128(&point["id"]),
                    point["score"].as_f64().unwrap_or(f64::NAN),
                )
            })
            .collect())
    }

    /// What is wrong when the first two hits for `query` must be `X` and
    /// another point, each with its payload.
    fn not_x_and_another(&self, route: &str, query: [f32; 3]) -> Option<String> {
        match self.found(route, query, 2) {
            Ok(hits) => {
                let full = hits.len() == 2
                    && point_id_to_u128(&hits[0]["id"]) == u128::from(X)
                    && hits.iter().all(|hit| hit["payload"].is_object());
                (!full).then(|| format!("{route} returns {hits:?}"))
            }
            Err(error) => Some(error),
        }
    }

    /// The points the by-id read returns for `X`, with their vectors.
    fn by_id(&self) -> Vec<serde_json::Value> {
        self.call(
            "get",
            json!({"ids": [X], "with_payload": false, "with_vector": true}),
        )
        .as_array()
        .unwrap()
        .clone()
    }

    /// What is wrong, if anything, with how `X` is stored and served when it
    /// was last written with `NEW`. `one_copy`: the write removed the old row.
    fn stale(&self, one_copy: bool) -> Vec<String> {
        self.served(NEW, one_copy)
    }

    /// What is wrong, if anything, when `X` must be stored with `expected`,
    /// in its only row if `one_copy`, and read and searched with it.
    fn served(&self, expected: [f32; 3], one_copy: bool) -> Vec<String> {
        let mut wrong = Vec::new();
        let copies = self.rows().remove(&u128::from(X)).unwrap_or_default();
        let fresh = copies
            .iter()
            .filter(|(_, _, vector)| same(vector, &expected))
            .count();
        if fresh != 1 || (one_copy && copies.len() != 1) {
            wrong.push(format!("stored rows {copies:?}"));
        }
        let read = self.by_id();
        let vector = read.first().map(|point| point["vector"]["dense"].clone());
        let fresh = vector
            .clone()
            .and_then(|vector| serde_json::from_value::<Vec<f32>>(vector).ok())
            .map(|vector| same(&vector, &expected))
            .unwrap_or(false);
        if !fresh {
            wrong.push(format!("the by-id read returns {vector:?}"));
        }
        for route in ["search", "query"] {
            match self.hits(route, expected) {
                Ok(hits) => {
                    let scores: Vec<f64> = hits
                        .into_iter()
                        .filter(|(id, _)| *id == u128::from(X))
                        .map(|(_, score)| score)
                        .collect();
                    if scores.len() != 1 || !exact(scores[0]) {
                        wrong.push(format!("{route} scores it {scores:?}"));
                    }
                }
                Err(error) => wrong.push(error),
            }
        }
        wrong
    }

    /// What is wrong when exactly `live` must be stored, one row each, and
    /// scrolled.
    fn lost_or_extra(&self, live: &HashSet<u128>) -> Vec<String> {
        let mut wrong = Vec::new();
        let rows = self.rows();
        let stored: HashSet<u128> = rows.keys().copied().collect();
        if stored != *live || rows.values().any(|copies| copies.len() != 1) {
            let mut rows: Vec<_> = rows.into_iter().collect();
            rows.sort_by_key(|(id, _)| *id);
            wrong.push(format!("stored rows {rows:?}"));
        }
        let scrolled: HashSet<u128> = self
            .call("scroll", json!({"limit": 256, "with_payload": false}))["points"]
            .as_array()
            .unwrap()
            .iter()
            .map(|point| point_id_to_u128(&point["id"]))
            .collect();
        if scrolled != *live {
            let mut scrolled: Vec<_> = scrolled.into_iter().collect();
            scrolled.sort();
            wrong.push(format!("scroll returns {scrolled:?}"));
        }
        wrong
    }
}

/// A changed re-upsert of a point whose first copy is in a sealed, indexed
/// segment. With the switch at `0` the write leaves that copy in place, as
/// data written before the fix has it.
#[test]
#[serial_test::serial]
fn changed_reupsert_is_read_with_its_new_vector() {
    let mut wrong = Vec::new();
    for (switch, name) in [("1", "reupsert_removes"), ("0", "reupsert_keeps")] {
        let _switch = EnvVar::set("HELIX_LSM_REUPSERT_DELETE", switch);
        let points = Points::new(name);
        points.upsert(&[(X, OLD), (2, FAR)]);
        points.seal();
        points.upsert(&[(X, NEW)]);
        let found = points.stale(switch == "1");
        wrong.extend(
            found
                .into_iter()
                .map(|what| format!("switch {switch}: {what}")),
        );
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Two writes of a point into one mutable tail: the second overwrites the
/// tail's row, and that row stays in the writer's own search of the tail.
#[test]
#[serial_test::serial]
fn changed_reupsert_within_one_tail_stays_searchable() {
    let points = Points::new("reupsert_in_tail");
    points.upsert(&[(X, OLD), (2, FAR)]);
    points.upsert(&[(X, NEW)]);
    let mut wrong = points.stale(true);
    wrong.extend(points.lost_or_extra(&HashSet::from([u128::from(X), 2])));
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// The tail that holds a point is sealed while a changed re-upsert of it is
/// under way, after the write has read the point and before it appends: the
/// new row goes to the next tail, and the sealed one's is an older copy.
#[test]
#[serial_test::serial]
fn changed_reupsert_across_a_tail_seal_leaves_no_older_row() {
    let points = Points::new("reupsert_across_seal");
    points.upsert(&[(X, OLD), (2, FAR)]);
    let hook = Rc::clone(&points);
    let sealed = points.segments();
    with_hook(
        "upsert_prepared",
        move || hook.seal(),
        || points.upsert(&[(X, NEW)]),
    );
    assert_ne!(points.segments(), sealed, "no seal");
    let mut wrong = points.stale(true);
    wrong.extend(points.lost_or_extra(&HashSet::from([u128::from(X), 2])));
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Writes that replace nothing read the store no more than with the removal
/// switched off: new points, an unchanged re-upsert, and the same content
/// under a new id after a delete.
#[test]
#[serial_test::serial]
fn writes_that_replace_nothing_read_nothing_more() {
    use crate::helix_engine::storage_core::backend_lsm::lsm_round_trips_for_test as round_trips;
    // The sidecar's put to the object store is a round trip on this thread
    // and goes by the clock (one every 10 s a collection): a phase that
    // crossed that mark on a slow runner counted one more than the other.
    let _no_sidecar_put = EnvVar::set("HELIX_LSM_METADATA_SIDECAR_PUT_INTERVAL_MS", "3600000");
    let spent = |switch: &str, name: &'static str| {
        let _switch = EnvVar::set("HELIX_LSM_REUPSERT_DELETE", switch);
        let points = Points::new(name);
        points.upsert(&[(X, OLD), (2, FAR)]);
        points.seal();
        let before = round_trips();
        points.upsert(&[(3, FAR), (4, NEW)]);
        points.upsert(&[(X, OLD), (2, FAR)]);
        let upserts = round_trips() - before;
        // The delete is left out. It reads more when the point is the entry
        // of its graph, and more again per level the point has above 0
        // (`repoint_entry_before_delete_be`), and HNSW draws those levels at
        // random: 20, 26 or 27 round trips, whatever the switch says.
        points.call("delete", json!({"points": [X]}));
        let before = round_trips();
        points.upsert(&[(5, OLD)]);
        upserts + round_trips() - before
    };
    let on = spent("1", "replaces_nothing_on");
    let off = spent("0", "replaces_nothing_off");
    assert!(off > 0, "the writes must run on this thread");
    assert!(
        on <= off,
        "store round trips: {on} with the removal, {off} without"
    );
}

/// A merged segment's id is above the id of the tail that was mutable when it
/// was made, so once that tail is sealed search takes the merged segment for
/// the newer of the two.
#[test]
#[serial_test::serial]
fn changed_reupsert_is_searched_with_its_new_vector_once_its_tail_is_sealed() {
    let _quiesced = EnvVar::set("HELIX_INDEX_SUBTHRESHOLD_QUIESCED_TAILS", "0");
    let points = Points::new("reupsert_after_merge");
    let first = points.merge_ready();
    optimize(&points.storage, points.name, "dense");
    assert!(!points.segments().contains(&first), "no merge");
    points.upsert(&[(X, NEW)]);
    points.seal();
    let mut wrong = points.stale(true);
    wrong.extend(points.lost_or_extra(&merge_ids()));
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Two copies written before the fix (switch at `0`), the old one alone in
/// the smallest segment and the new one in a larger, later segment; a merge
/// takes its sources smallest first.
#[test]
#[serial_test::serial]
fn merge_keeps_the_newer_of_two_copies() {
    let _quiesced = EnvVar::set("HELIX_INDEX_SUBTHRESHOLD_QUIESCED_TAILS", "0");
    let _switch = EnvVar::set("HELIX_LSM_REUPSERT_DELETE", "0");
    let points = Points::new("merge_two_copies");
    points.upsert(&[(X, OLD)]);
    points.seal();
    points.upsert(&[(X, NEW), (2, FAR)]);
    points.seal();
    let pair = points.segments();
    let mut live: HashSet<u128> = HashSet::from([u128::from(X), 2]);
    let mut id = 10;
    for _ in 2..=NamedVectorManager::max_indexed_segments_target() {
        points.upsert(&[(id, FAR), (id + 1, FAR), (id + 2, FAR)]);
        points.seal();
        live.extend((id..id + 3).map(u128::from));
        id += 3;
    }
    optimize(&points.storage, points.name, "dense");
    let merged = points.segments();
    assert!(
        pair.iter().all(|name| !merged.contains(name)),
        "the merge must take both copies' segments: {pair:?} in {merged:?}"
    );
    let mut wrong = points.stale(true);
    wrong.extend(points.lost_or_extra(&live));
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// What the count route reads (`stored_reach_be`) of two copies written
/// before the fix (switch at `0`): the tail's row is served, and of the
/// sealed segment's three rows the one the tail's copy has replaced is not.
#[test]
#[serial_test::serial]
fn the_count_route_tells_a_replaced_row_from_a_served_one() {
    let _switch = EnvVar::set("HELIX_LSM_REUPSERT_DELETE", "0");
    let points = Points::new("reach_served");
    points.upsert(&[(X, OLD), (2, FAR), (3, FAR)]);
    points.seal();
    points.upsert(&[(X, NEW)]);
    let read = points.storage.backend.begin_read().unwrap();
    let rows_and_served: Vec<(usize, usize)> = points
        .storage
        .named_vectors
        .stored_reach_be(&read)
        .unwrap()
        .iter()
        .map(|(_, _, reach)| (reach.ids.len(), reach.served))
        .filter(|&(rows, _)| rows > 0)
        .collect();
    assert_eq!(rows_and_served, [(1, 1), (3, 2)], "newest segment first");
}

/// A point deleted while a merge builds from its row: at "exported" the merge
/// has read its sources, at "flat_written" its target has rows and no graph,
/// at "written" the target is whole, at "drained" the merge has closed its
/// window and only the swap is left. At none has it published.
#[test]
#[serial_test::serial]
fn point_deleted_during_a_merge_is_not_republished() {
    let _quiesced = EnvVar::set("HELIX_INDEX_SUBTHRESHOLD_QUIESCED_TAILS", "0");
    let mut wrong = Vec::new();
    for (stage, name) in [
        ("exported", "delete_at_exported"),
        ("flat_written", "delete_at_flat_written"),
        ("written", "delete_at_written"),
        ("drained", "delete_at_drained"),
    ] {
        let points = Points::new(name);
        let first = points.merge_ready();
        let hook = Rc::clone(&points);
        optimize_with(&points.storage, name, "dense", stage, move || {
            hook.call("delete", json!({"points": [X]}));
        });
        assert!(!points.segments().contains(&first), "{stage}: no merge");
        let mut live = merge_ids();
        live.remove(&u128::from(X));
        let mut found = points.lost_or_extra(&live);
        for route in ["search", "query", "hybrid_query"] {
            match points.hits(route, OLD) {
                Ok(hits) => {
                    if let Some((_, score)) = hits.iter().find(|(id, _)| *id == u128::from(X)) {
                        found.push(format!("{route} returns it, score {score}"));
                    }
                }
                Err(error) => found.push(error),
            }
        }
        let read = points.by_id();
        if !read.is_empty() {
            found.push(format!("the by-id read returns {read:?}"));
        }
        wrong.extend(found.into_iter().map(|what| format!("{stage}: {what}")));
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// A point re-upserted with a changed vector while a merge builds from its
/// old row. The tail is sealed afterwards, which is when a republished old
/// row outranks the new one.
#[test]
#[serial_test::serial]
fn point_reupserted_during_a_merge_is_served_with_its_new_vector() {
    let _quiesced = EnvVar::set("HELIX_INDEX_SUBTHRESHOLD_QUIESCED_TAILS", "0");
    let mut wrong = Vec::new();
    for (stage, name) in [
        ("exported", "reupsert_at_exported"),
        ("flat_written", "reupsert_at_flat_written"),
        ("written", "reupsert_at_written"),
        ("drained", "reupsert_at_drained"),
    ] {
        let points = Points::new(name);
        let first = points.merge_ready();
        let hook = Rc::clone(&points);
        optimize_with(&points.storage, name, "dense", stage, move || {
            hook.upsert(&[(X, NEW)]);
        });
        assert!(!points.segments().contains(&first), "{stage}: no merge");
        points.seal();
        let mut found = points.stale(true);
        found.extend(points.lost_or_extra(&merge_ids()));
        wrong.extend(found.into_iter().map(|what| format!("{stage}: {what}")));
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// A dense row whose point is gone, in the mutable tail and then in an
/// indexed segment: `/points/query` and the hybrid dense channel leave it out
/// and still return as many hits as were asked for.
#[test]
#[serial_test::serial]
fn row_without_a_point_is_no_hit_and_the_reply_stays_full() {
    let points = Points::new("row_without_point");
    points.upsert(&[(X, OLD), (2, FAR), (3, FAR)]);
    points.row_without_a_point(GONE, NEW);
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        if sealed {
            points.seal();
        }
        for route in ["query", "hybrid_query"] {
            let found = points.not_x_and_another(route, NEW);
            wrong.extend(found.map(|what| format!("sealed {sealed}: {what}")));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// A row whose point is gone, alone in a segment a merge takes: the merge
/// leaves it out, so such rows go as their segments merge.
#[test]
#[serial_test::serial]
fn merge_leaves_out_a_row_without_a_point() {
    let _quiesced = EnvVar::set("HELIX_INDEX_SUBTHRESHOLD_QUIESCED_TAILS", "0");
    let points = Points::new("merge_row_without_point");
    points.row_without_a_point(GONE, FAR);
    points.seal();
    let first = points.segments().remove(0);
    points.merge_ready();
    optimize(&points.storage, points.name, "dense");
    assert!(!points.segments().contains(&first), "no merge");
    let wrong = points.lost_or_extra(&merge_ids());
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// An LSM collection manager on the test object store, in `role`.
fn object_store_config(role: LsmRole) -> Config {
    let mut config = Config::default().with_storage_backend(StorageBackendConfig::Lsm {
        storage: LsmStorage::ObjectStore,
        role,
    });
    config.vector_config.flat_scan_threshold = Some(0);
    config
}

/// A chunk that rewrites a point and then fails on a later one applies
/// nothing, and the writer still finds the point by its stored row. The row
/// is in a sealed tail without an index. The writer scans that flat by its
/// cached ordinals once it has them whole, which is after it opened the
/// collection from the store: so a second writer opens it. The later point
/// fails on a dense name, then on a sparse vector's values.
#[test]
#[serial_test::serial]
fn failed_chunk_keeps_the_row_it_would_have_replaced() {
    let _store = set_lsm_test_object_store(Arc::new(object_store::memory::InMemory::new()));
    let _prefix = EnvVar::set("HELIX_LSM_PREFIX", "reupsert_copy_failed_chunk");
    let _quiesced = EnvVar::set("HELIX_INDEX_SUBTHRESHOLD_QUIESCED_TAILS", "0");
    let name = "failed_chunk";
    let writer = setup_with_config(object_store_config(LsmRole::Writer));
    create_collection_with_sparse(&writer, name);
    let points = Points::open(writer, name);
    points.upsert(&[(X, OLD), (2, FAR)]);
    seal_tail(&points.storage, "dense");
    drop(points);

    let points = Points::open(
        setup_with_config(object_store_config(LsmRole::Writer)),
        name,
    );
    let live = HashSet::from([u128::from(X), 2]);
    let mut wrong = points.served(OLD, true);
    for (what, later) in [
        ("a dense name", json!({"no_such_space": FAR})),
        (
            "sparse values",
            json!({"dense": FAR, "lex_sparse": {"indices": [7, 7], "values": [1.0, 1.0]}}),
        ),
    ] {
        let failed = points.try_call(
            "upsert",
            json!({"points": [
                {"id": X, "vector": {"dense": NEW}},
                {"id": 3, "vector": later},
            ]}),
        );
        assert!(failed.is_err(), "{what}: {failed:?}");
        let mut found = points.served(OLD, true);
        found.extend(points.lost_or_extra(&live));
        wrong.extend(
            found
                .into_iter()
                .map(|found| format!("after {what}: {found}")),
        );
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// One sealed segment of nine new points.
fn larger_segment(points: &Points, next: &mut u64, live: &mut HashSet<u128>) {
    let rows: Vec<(u64, [f32; 3])> = (*next..*next + 9).map(|id| (id, FAR)).collect();
    points.upsert(&rows);
    points.seal();
    live.extend(rows.iter().map(|(id, _)| u128::from(*id)));
    *next += 9;
}

/// The arrangement the defect left behind (switch at `0`): the old copy of
/// `X` in a merged segment numbered above the sealed tail that holds the new
/// one (written as `second_write`), `TAIL` and `fill`, so search serves the
/// old one. `row_payload` is `HELIX_DENSE_ROW_PAYLOAD` at the old write, at
/// the new write and at the merge that rewrites the old copy alone. Returns
/// the collection, its live ids and the next id to write.
fn old_copy_numbered_above(
    name: &'static str,
    second_write: u64,
    fill: &[(u64, [f32; 3])],
    row_payload: [&str; 3],
) -> (Rc<Points>, HashSet<u128>, u64) {
    let [old, new, alone] = row_payload;
    let _shape = EnvVar::set("HELIX_DENSE_ROW_PAYLOAD", old);
    let points = Points::new(name);
    let mut live = HashSet::from([u128::from(X)]);
    // Eight one-row segments, the first with the old copy, and enough
    // larger ones that the next merge takes exactly the eight.
    points.write_x(OLD, 1);
    points.seal();
    for id in 2..=8 {
        points.upsert(&[(id, FAR)]);
        points.seal();
        live.insert(u128::from(id));
    }
    let mut next = 100;
    for _ in 8..=NamedVectorManager::max_indexed_segments_target() {
        larger_segment(&points, &mut next, &mut live);
    }
    // The new copy goes to the tail that is mutable while they merge.
    let _shape = EnvVar::set("HELIX_DENSE_ROW_PAYLOAD", new);
    points.write_x(NEW, second_write);
    let mut tail = vec![(TAIL, FAR)];
    tail.extend_from_slice(fill);
    points.upsert(&tail);
    live.extend(tail.iter().map(|(id, _)| u128::from(*id)));
    let _shape = EnvVar::set("HELIX_DENSE_ROW_PAYLOAD", alone);
    optimize(&points.storage, name, "dense");
    points.seal();
    let arranged = points.served(OLD, false);
    assert!(
        arranged.is_empty(),
        "{name}: not the arrangement: {arranged:#?}"
    );
    (points, live, next)
}

/// A merge of the two copies' segments keeps the copy written with the
/// point's payload; where both were, it keeps the one search serves.
///
/// A row keeps that payload or, under `HELIX_DENSE_ROW_PAYLOAD=0`, its
/// digest, and the merge that makes the arrangement rewrites the old copy
/// away from the new one: the choice is the same whichever of the two each
/// row keeps by the time they meet.
#[test]
#[serial_test::serial]
fn merge_keeps_the_copy_written_with_the_points_payload() {
    let _quiesced = EnvVar::set("HELIX_INDEX_SUBTHRESHOLD_QUIESCED_TAILS", "0");
    let _switch = EnvVar::set("HELIX_LSM_REUPSERT_DELETE", "0");
    let mut wrong = Vec::new();
    // `HELIX_DENSE_ROW_PAYLOAD` at the old write, at the new write, at the
    // merge that rewrites the old copy alone and at the merge of the two.
    for (names, old, new, alone, both) in [
        (
            ["payloads_told_apart", "payloads_alike"],
            "1",
            "1",
            "1",
            "1",
        ),
        (["digests_told_apart", "digests_alike"], "0", "0", "0", "0"),
        // A pair written before the switch, one row digested since.
        (
            ["digested_alone_told_apart", "digested_alone_alike"],
            "1",
            "1",
            "0",
            "0",
        ),
        (
            ["new_digest_told_apart", "new_digest_alike"],
            "1",
            "0",
            "1",
            "1",
        ),
    ] {
        for (name, second_write, kept) in [(names[0], 2, NEW), (names[1], 1, OLD)] {
            let (points, mut live, mut next) =
                old_copy_numbered_above(name, second_write, &[], [old, new, alone]);
            let _shape = EnvVar::set("HELIX_DENSE_ROW_PAYLOAD", alone);
            // Six more larger segments: the next merge takes that tail and
            // the merged eight among its eight smallest.
            for _ in 0..6 {
                larger_segment(&points, &mut next, &mut live);
            }
            let _shape = EnvVar::set("HELIX_DENSE_ROW_PAYLOAD", both);
            optimize(&points.storage, name, "dense");
            let mut found = points.served(kept, true);
            found.extend(points.lost_or_extra(&live));
            let shapes = points.shapes(X, &payload_of(second_write));
            if both == "0" && shapes != ["digest"] {
                found.push(format!("the merged row keeps {shapes:?}"));
            }
            wrong.extend(found.into_iter().map(|what| format!("{name}: {what}")));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// An upsert writes the point's payload beside its dense vector, or under
/// `HELIX_DENSE_ROW_PAYLOAD=0` that payload's digest. The index build leaves
/// the row as it is, and the routes read the point and answer the same.
#[test]
#[serial_test::serial]
fn an_upsert_writes_the_payload_or_its_digest_beside_the_vector() {
    let mut answers = Vec::new();
    for (name, switch, kept) in [
        ("row_keeps_payload", "1", "payload"),
        ("row_keeps_digest", "0", "digest"),
    ] {
        let _shape = EnvVar::set("HELIX_DENSE_ROW_PAYLOAD", switch);
        let points = Points::new(name);
        points.write_x(OLD, 1);
        assert_eq!(points.shapes(X, &payload_of(1)), [kept], "{name}: tail");
        let seen = points.payloads_of_x(OLD);
        assert!(
            seen[0]["write"] == 1 && seen.iter().all(|payload| *payload == seen[0]),
            "{name}: {seen:#?}"
        );
        points.seal();
        assert_eq!(points.shapes(X, &payload_of(1)), [kept], "{name}: indexed");
        assert_eq!(points.payloads_of_x(OLD), seen, "{name}: indexed");
        answers.push(seen);
    }
    assert_eq!(answers[0], answers[1]);
}

/// A merge under `HELIX_DENSE_ROW_PAYLOAD=0` writes the digest for a row
/// that kept its payload; by default it carries the payload. A row it does
/// not take keeps what it has.
#[test]
#[serial_test::serial]
fn a_merge_writes_the_digest_for_a_row_that_kept_its_payload() {
    let _quiesced = EnvVar::set("HELIX_INDEX_SUBTHRESHOLD_QUIESCED_TAILS", "0");
    for (name, switch, rewritten) in [
        ("merge_carries_payload", "1", "payload"),
        ("merge_writes_digest", "0", "digest"),
    ] {
        let _shape = EnvVar::set("HELIX_DENSE_ROW_PAYLOAD", "1");
        let points = Points::new(name);
        // One indexed segment a point, one more than the optimizer keeps.
        let sources = NamedVectorManager::max_indexed_segments_target() as u64 + 1;
        for id in 1..=sources {
            points.write(id, FAR, id);
            points.seal();
        }
        points.write(TAIL, FAR, TAIL);
        let before = points.segments();
        let _shape = EnvVar::set("HELIX_DENSE_ROW_PAYLOAD", switch);
        optimize(&points.storage, name, "dense");
        let mut merged = 0;
        for (id, segment, _, _, fields) in points.copies() {
            let expected = if before.contains(&segment) {
                "payload"
            } else {
                merged += 1;
                rewritten
            };
            assert_eq!(
                shape(&fields, &payload_of(id as u64)),
                expected,
                "{name}: point {id} in {segment}"
            );
        }
        assert!(merged > 1, "{name}: the merge rewrote {merged} rows");
        assert!(points.lost_or_extra(&merge_ids()).is_empty(), "{name}");
    }
}

/// A digest is a `U64` alone under its reserved key, and no number of a
/// client's payload is stored as a `U64`: a payload that uses the key is
/// kept, and compared, as the payload it is.
#[test]
#[serial_test::serial]
fn a_payload_under_the_digests_key_is_kept_as_a_payload() {
    let _shape = EnvVar::set("HELIX_DENSE_ROW_PAYLOAD", "1");
    let points = Points::new("payload_under_digest_key");
    for (id, number) in [
        (1, json!(5)),
        (2, json!(18446744073709551615u64)),
        (3, json!(5.0)),
        (4, json!(-5)),
    ] {
        let payload = json!({ "__helix_row_payload_digest__": number });
        let point = json!({"id": id, "vector": {"dense": FAR}, "payload": payload});
        points.call("upsert", json!({ "points": [point] }));
        assert_eq!(points.shapes(id, &payload), ["payload"], "{payload}");
    }
}

/// Rows enough in a segment for a regraph to take it (520, ids from 10,000),
/// spread over the sphere.
fn regraph_fill() -> Vec<(u64, [f32; 3])> {
    (0..520u64)
        .map(|i| {
            let z = 1.0 - (2 * i + 1) as f32 / 520.0;
            let (sin, cos) = (i as f32 * 2.4).sin_cos();
            let r = (1.0 - z * z).sqrt();
            (10_000 + i, [r * cos, r * sin, z])
        })
        .collect()
}

/// A regraph takes a segment out when a rebuild would keep no row of it, and
/// rebuilds one it would keep a single row of. 520 points are written and
/// sealed, then written again with another payload and sealed (switch at
/// `0`, so the first rows stay): every row of the first segment has its
/// point's own copy in the second, which search serves. The first segment
/// goes with nothing in its place and the second is rebuilt. With one more
/// point in the first write alone, the first segment is rebuilt to that one
/// row, and a second regraph leaves it be. Either way each point is left one
/// row.
#[test]
#[serial_test::serial]
fn regraph_retires_a_segment_it_would_keep_no_row_of() {
    let _quiesced = EnvVar::set("HELIX_INDEX_SUBTHRESHOLD_QUIESCED_TAILS", "0");
    let _switch = EnvVar::set("HELIX_LSM_REUPSERT_DELETE", "0");
    let fill = regraph_fill();
    let mut wrong = Vec::new();
    // What the status says: segments retired, segments rebuilt, points
    // rebuilt, rows left out for a more recent copy. The same where the rows
    // keep digests of their payloads (`HELIX_DENSE_ROW_PAYLOAD=0`).
    for (name, row_payload, alone, said) in [
        ("regraph_retires", "1", None, (1, 1, 520, 520)),
        ("regraph_keeps_one_row", "1", Some(9_999), (0, 2, 521, 520)),
        ("regraph_retires_digests", "0", None, (1, 1, 520, 520)),
        (
            "regraph_keeps_one_row_digests",
            "0",
            Some(9_999),
            (0, 2, 521, 520),
        ),
    ] {
        let _shape = EnvVar::set("HELIX_DENSE_ROW_PAYLOAD", row_payload);
        let points = Points::new(name);
        let sealed = |ids: Vec<(u64, [f32; 3])>, write: u64| {
            let body: Vec<serde_json::Value> = ids
                .iter()
                .map(|(id, vector)| {
                    json!({"id": id, "vector": {"dense": vector}, "payload": payload_of(write)})
                })
                .collect();
            points.call("upsert", json!({ "points": body }));
            points.seal();
        };
        let mut first = fill.clone();
        first.extend(alone.map(|id| (id, FAR)));
        let ids = first.len();
        sealed(first, 1);
        sealed(fill.clone(), 2);
        let before = points.segments();
        let status = regraph(&points.storage, name, "dense");
        let after = points.segments();
        let rows = points.rows();
        let seen = (
            status.segments_retired,
            status.segments_rebuilt,
            status.points_rebuilt,
            status.superseded_dropped,
        );
        // The two sealed segments are the first two, and neither is left.
        if seen != said
            || before.iter().take(2).any(|segment| after.contains(segment))
            || rows.len() != ids
            || rows.values().any(|copies| copies.len() != 1)
        {
            wrong.push(format!(
                "{name}: {before:?} to {after:?}, {} ids stored, {status:?}",
                rows.len()
            ));
        }
        // The one row a rebuild kept: a second regraph leaves its segment as
        // it is, which stores too few rows to be walked, and a search still
        // returns the point.
        if let Some(id) = alone.map(u128::from) {
            let segment_of = |rows: &Rows| rows.get(&id).map(|copies| copies[0].0.clone());
            let kept = segment_of(&rows);
            let again = regraph(&points.storage, name, "dense");
            let found = points.hits("search", FAR).unwrap_or_default();
            if (again.segments_small, again.segments_retired) != (1, 0)
                || kept.is_none()
                || segment_of(&points.rows()) != kept
                || !found.iter().any(|(hit, _)| *hit == id)
            {
                wrong.push(format!(
                    "{name}, a second regraph: {again:?}, {kept:?}, found {found:?}"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// A regraph rebuilds the sealed tail alone, and what it rebuilds takes the
/// newest number: the tail's copy goes along when it is the one written with
/// the point's payload, and stays out where both copies were, so that search
/// serves the same one as before.
///
/// The regraph reads what each row keeps of its payload as a merge does, so
/// the choice is the same where the rows keep digests
/// (`HELIX_DENSE_ROW_PAYLOAD=0`), and where a pair written with payloads is
/// met by a regraph under the switch. A regraph under it writes digests.
#[test]
#[serial_test::serial]
fn regraph_keeps_the_copy_written_with_the_points_payload() {
    let _quiesced = EnvVar::set("HELIX_INDEX_SUBTHRESHOLD_QUIESCED_TAILS", "0");
    let _switch = EnvVar::set("HELIX_LSM_REUPSERT_DELETE", "0");
    let fill = regraph_fill();
    let mut wrong = Vec::new();
    // `HELIX_DENSE_ROW_PAYLOAD` at the two writes and the merge that makes
    // the arrangement, and at the regraph.
    for (names, written, at_regraph) in [
        (
            ["regraph_copies_told_apart", "regraph_copies_alike"],
            "1",
            "1",
        ),
        (
            ["regraph_digests_told_apart", "regraph_digests_alike"],
            "0",
            "0",
        ),
        (["regraph_late_told_apart", "regraph_late_alike"], "1", "0"),
    ] {
        for (name, second_write, kept, left_out) in [(names[0], 2, NEW, 0), (names[1], 1, OLD, 1)] {
            let (points, live, _) =
                old_copy_numbered_above(name, second_write, &fill, [written; 3]);
            let _shape = EnvVar::set("HELIX_DENSE_ROW_PAYLOAD", at_regraph);
            let status = regraph(&points.storage, name, "dense");
            let mut found = points.served(kept, left_out == 1);
            let stored = points.rows().len();
            if (status.segments_rebuilt, status.superseded_dropped) != (1, left_out)
                || stored != live.len()
            {
                found.push(format!("{stored} of {} ids stored, {status:?}", live.len()));
            }
            // The tail's copy of `X`, where it went along: the row the
            // regraph wrote for it.
            if left_out == 0 {
                let rebuilt: Vec<&'static str> = points
                    .copies()
                    .iter()
                    .filter(|copy| copy.0 == u128::from(X) && same(&copy.3, &NEW))
                    .map(|copy| shape(&copy.4, &payload_of(second_write)))
                    .collect();
                let expected = if at_regraph == "0" {
                    "digest"
                } else {
                    "payload"
                };
                if rebuilt != [expected] {
                    found.push(format!("the rebuilt row keeps {rebuilt:?}"));
                }
            }
            wrong.extend(found.into_iter().map(|what| format!("{name}: {what}")));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Read through a reader replica that opens the collection after the
/// writes: it serves a rewritten point's new row alone, and no row without a
/// point.
#[test]
#[serial_test::serial]
fn reader_replica_serves_the_new_row_and_no_row_without_a_point() {
    let _store = set_lsm_test_object_store(Arc::new(object_store::memory::InMemory::new()));
    let _prefix = EnvVar::set("HELIX_LSM_PREFIX", "reupsert_copy_reader");
    let name = "replica_rows";
    let writer = setup_with_config(object_store_config(LsmRole::Writer));
    create_dense_collection_lsm(&writer, name);
    let writer = Points::open(writer, name);
    writer.upsert(&[(X, OLD), (2, FAR), (3, FAR)]);
    writer.seal();
    writer.upsert(&[(X, NEW)]);
    writer.row_without_a_point(GONE, NEW);
    writer.seal();

    let reader = Points::open(
        setup_with_config(object_store_config(LsmRole::Reader)),
        name,
    );
    assert!(reader.storage.backend.is_reader_replica());
    let mut wrong = Vec::new();
    let read = reader.by_id();
    let vector = read.first().map(|point| point["vector"]["dense"].clone());
    let fresh = vector
        .clone()
        .and_then(|vector| serde_json::from_value::<Vec<f32>>(vector).ok())
        .is_some_and(|vector| same(&vector, &NEW));
    if !fresh {
        wrong.push(format!("the by-id read returns {vector:?}"));
    }
    for route in ["query", "hybrid_query"] {
        wrong.extend(reader.not_x_and_another(route, NEW));
        match reader.hits(route, NEW) {
            Ok(hits) => {
                let scores: Vec<f64> = hits
                    .into_iter()
                    .filter(|(id, _)| *id == u128::from(X))
                    .map(|(_, score)| score)
                    .collect();
                if scores.len() != 1 || !exact(scores[0]) {
                    wrong.push(format!("{route} scores it {scores:?}"));
                }
            }
            Err(error) => wrong.push(error),
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Two copies written before the fix (switch at `0`), the old one in a sealed
/// segment and the new one in the tail. The broad prefilter scan asks each
/// segment for the members the other one holds; it serves `X` with the newer
/// copy and returns the same hits, to the bit, whether it reads those rows in
/// batches or one at a time.
#[test]
#[serial_test::serial]
fn broad_scan_over_two_copies_is_the_same_batched_and_one_by_one() {
    use crate::helix_engine::vector_core::vector_core::with_flat_scan_row_batch_for_test;

    let _switch = EnvVar::set("HELIX_LSM_REUPSERT_DELETE", "0");
    let points = Points::new("broad_scan_two_copies");
    points.upsert(&[(X, OLD), (2, FAR)]);
    points.seal();
    points.upsert(&[(X, NEW), (3, FAR)]);
    let copies = points.rows().remove(&u128::from(X)).unwrap_or_default();
    assert_eq!(copies.len(), 2, "an old and a new copy: {copies:?}");

    let read = points.storage.backend.begin_read().unwrap();
    for segment in points.segments() {
        points
            .storage
            .named_vectors
            .with_core(&segment, |core| {
                if core.is_empty(&read)? {
                    return Ok(());
                }
                core.ensure_lsm_mmap_sidecar()?;
                assert!(core.has_mmap_sidecar(), "{segment} has no sidecar");
                Ok(())
            })
            .unwrap();
    }
    let members: HashSet<u128> = [X, 2, 3, GONE].into_iter().map(u128::from).collect();
    let scan = |batch: usize| {
        with_flat_scan_row_batch_for_test(batch, || {
            points
                .storage
                .named_vectors
                .dense_search_broad_prefilter_scan_be::<fn(u128) -> bool>(
                    &read, "dense", &NEW, 8, &members, None, None,
                )
                .unwrap()
                .expect("both segments have a sidecar")
                .into_iter()
                .map(|hit| {
                    let vector: Vec<u32> =
                        hit.get_data().iter().map(|value| value.to_bits()).collect();
                    (hit.id, hit.get_distance().to_bits(), vector)
                })
                .collect::<Vec<_>>()
        })
    };

    let one_by_one = scan(0);
    let ids: Vec<u128> = one_by_one.iter().map(|hit| hit.0).collect();
    assert_eq!(ids.len(), 3, "X, 2 and 3 once each: {ids:?}");
    assert_eq!(ids[0], u128::from(X), "{ids:?}");
    let served: Vec<f32> = one_by_one[0]
        .2
        .iter()
        .map(|bits| f32::from_bits(*bits))
        .collect();
    assert!(same(&served, &NEW), "X is served with {served:?}");
    for batch in [1, 2, 2048] {
        assert_eq!(scan(batch), one_by_one, "batch {batch}");
    }
    // Production readers run with multi-get off.
    {
        let _multi_get = EnvVar::set("HELIX_LSM_MULTI_GET", "0");
        for batch in [0, 1, 2, 2048] {
            assert_eq!(scan(batch), one_by_one, "multi-get off, batch {batch}");
        }
    }

    // Every caller answers a scan that failed from the graph walk, so the
    // engine counts how each scan ended.
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        assert_eq!(scan(2048), one_by_one);
        let no_such_space = points
            .storage
            .named_vectors
            .dense_search_broad_prefilter_scan_be::<fn(u128) -> bool>(
                &read, "absent", &NEW, 8, &members, None, None,
            );
        assert!(no_such_space.is_err());
        // A caller that left: its reads are cancelled, which is no failure
        // of the scan.
        let gone = LsmReadCancellation::new();
        gone.cancel();
        let for_nobody = allow_lsm_blocking_cancellable(gone, || {
            points
                .storage
                .named_vectors
                .dense_search_broad_prefilter_scan_be::<fn(u128) -> bool>(
                    &read, "dense", &NEW, 8, &members, None, None,
                )
        });
        assert!(
            for_nobody.is_err(),
            "a scan read rows for a caller that left"
        );
    });
    let rendered = handle.render();
    for outcome in ["served", "failed", "cancelled"] {
        let line = format!(r#"helix_broad_scan_total{{outcome="{outcome}"}} 1"#);
        assert!(rendered.contains(&line), "{line} not in:\n{rendered}");
    }
}

/// The link pass takes a segment a call: one that a search walks and the
/// pass has not marked. Two sealed segments of 520 points each: the first
/// call reads one of them and marks it, with one left to do, the second
/// reads the other, and a third finds none left and reads nothing.
///
/// A segment that carries the mark of the pass before this one is one the
/// pass has still to do: that pass tried a row's eight nearest rows alone,
/// and its mark does not say whether it left rows. The next call reads the
/// segment, and the one after finds none left.
///
/// So is a segment this pass has marked, once a hard delete has taken a row
/// of its graph: what that row alone led to is unreached again. It is read
/// once more and marked with the count of deleted rows it then has.
#[test]
#[serial_test::serial]
fn the_link_pass_takes_one_segment_a_call() {
    use crate::helix_engine::storage_core::replication::{
        link_stored_unreached, LINK_PASS_ROWS_A_CALL,
    };
    let name = "link_pass_calls";
    let points = Points::new(name);
    for first in [10_000u64, 20_000] {
        // Spread over the sphere.
        let rows: Vec<(u64, [f32; 3])> = (0..520u64)
            .map(|i| {
                let z = 1.0 - (2 * i + 1) as f32 / 520.0;
                let (sin, cos) = (i as f32 * 2.4).sin_cos();
                let r = (1.0 - z * z).sqrt();
                (first + i, [r * cos, r * sin, z])
            })
            .collect();
        points.upsert(&rows);
        points.seal();
    }
    let sealed = points.segments();
    // The segment a call read, and what it says is left.
    let call = || {
        let status = link_stored_unreached(
            &points.ctx.collections,
            points.ctx.collections.config(),
            name,
            LINK_PASS_ROWS_A_CALL,
        )
        .unwrap();
        assert_eq!(
            (status.state, status.readers_see_after),
            ("done", "restart"),
            "{status:?}"
        );
        (status.segment, status.remaining)
    };
    let (first, second, third) = (call(), call(), call());
    assert_eq!((first.1, second.1, third), (1, 0, (None, 0)));
    for read in [&first.0, &second.0] {
        assert!(
            read.as_ref().is_some_and(|read| sealed.contains(read)),
            "{read:?} of {sealed:?}"
        );
    }
    assert_ne!(first.0, second.0);

    let marked_by_the_pass_before = second.0.unwrap();
    points
        .storage
        .with_write_backend(|w| {
            points
                .storage
                .named_vectors
                .with_core(&marked_by_the_pass_before, |core| {
                    core.put_link_pass_mark_for_test(w, &[2]);
                    Ok(())
                })
                .map_err(GraphError::from)
        })
        .unwrap();
    assert_eq!(
        (call(), call()),
        ((Some(marked_by_the_pass_before.clone()), 0), (None, 0))
    );

    let deleted_from_since = marked_by_the_pass_before;
    points
        .storage
        .with_write_backend(|w| {
            points
                .storage
                .named_vectors
                .with_core(&deleted_from_since, |core| {
                    core.put_deleted_graph_rows_for_test(w, 1);
                    Ok(())
                })
                .map_err(GraphError::from)
        })
        .unwrap();
    assert_eq!(
        (call(), call()),
        ((Some(deleted_from_since.clone()), 0), (None, 0))
    );

    // A delete took a row between two calls of a pass over that segment: the
    // place it kept has the count it began at, 1, and the segment has 2 by
    // now. The call that comes to the end leaves the segment due and counts
    // it as remaining; the next reads it once more.
    let between_two_calls = deleted_from_since;
    points
        .storage
        .with_write_backend(|w| {
            points
                .storage
                .named_vectors
                .with_core(&between_two_calls, |core| {
                    let place =
                        [&[0u8][..], &0u128.to_be_bytes(), &1u64.to_be_bytes(), &[3]].concat();
                    core.put_link_pass_mark_for_test(w, &place);
                    core.put_deleted_graph_rows_for_test(w, 2);
                    Ok(())
                })
                .map_err(GraphError::from)
        })
        .unwrap();
    assert_eq!(
        (call(), call(), call()),
        (
            (Some(between_two_calls.clone()), 1),
            (Some(between_two_calls), 0),
            (None, 0)
        )
    );
}

/// A poisoned lock on the vector configs is no answer to whether a vector is
/// quantized: the route refuses on it, where it would have taken the
/// collection for one without.
#[test]
#[serial_test::serial]
fn the_link_route_refuses_when_it_cannot_tell_a_quantized_vector() {
    use crate::helix_gateway::api::collections::handle_link_unreached;
    let name = "link_pass_poisoned";
    let points = Points::new(name);
    points.storage.named_vectors.poison_configs_for_test();
    let input = make_input(
        &points.ctx,
        "POST",
        &format!("/collections/{name}/unreached/link"),
        Vec::new(),
        HashMap::from([("name".to_string(), name.to_string())]),
    );
    let mut response = Response::new();
    handle_link_unreached(&input, &mut response).unwrap();
    let body = String::from_utf8_lossy(&response.body);
    assert_eq!(response.status, 503, "{body}");
    assert!(body.contains("poisoned"), "{body}");
}

/// The link pass leaves a collection with a quantized named vector alone.
/// The route refuses it with an error and no status, so that no caller reads
/// the reply as one of a call to repeat; a collection without one is taken
/// as before.
#[test]
#[serial_test::serial]
fn the_link_route_refuses_a_collection_with_a_quantized_vector() {
    use crate::helix_gateway::api::collections::handle_link_unreached;
    // The route's reply for a collection made while new collections quantize
    // as `mode` says.
    let reply = |name: &'static str, mode: &str| {
        let points = {
            let _held = super::quantizer_env_for_test::hold();
            let _mode = EnvVar::set("HELIX_SPINDLE_MODE", mode);
            Points::new(name)
        };
        let input = make_input(
            &points.ctx,
            "POST",
            &format!("/collections/{name}/unreached/link"),
            Vec::new(),
            HashMap::from([("name".to_string(), name.to_string())]),
        );
        let mut response = Response::new();
        handle_link_unreached(&input, &mut response).unwrap();
        let body: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        (response.status, body)
    };
    let (status, body) = reply("link_pass_quantized", "scalar");
    assert_eq!(status, 409, "{body}");
    assert!(
        body["error"]
            .as_str()
            .is_some_and(|error| error.contains("quantized")),
        "{body}"
    );
    let (status, body) = reply("link_pass_plain", "none");
    assert_eq!((status, &body["state"]), (200, &json!("done")), "{body}");
}

/// `f`'s value and which way the exact scorer went for the one set it
/// scored: the `way` label of `helix_dense_exact_scorer_total`.
fn exact_scorer_way<R>(f: impl FnOnce() -> R) -> (R, String) {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let out = metrics::with_local_recorder(&recorder, f);
    let rendered = handle.render();
    let ways: Vec<&str> = rendered
        .lines()
        .filter(|line| line.starts_with("helix_dense_exact_scorer_total{"))
        .collect();
    assert_eq!(ways.len(), 1, "one set, one way: {rendered}");
    (
        out,
        ways[0].split('"').nth(1).unwrap_or_default().to_string(),
    )
}

/// A filtered search on a reader replica whose filter an index answers with
/// 700 of the collection's 900 points: the handlers hand that set to the
/// exact scorer, which scans it (1.4 rows an id) where the setting allows
/// and reads each id at 0, and the reply is the same either way: ids, scores,
/// payloads and vectors, in the same order. Sixty of the points were written
/// twice with the re-upsert delete off, as a store written before it has
/// them, so their older rows are still in the sealed segment.
///
/// An id list of the client's own sizes (its gate's, up to 500, and its
/// symbol lookup's 512) is read id by id whatever the setting. A longer one,
/// which no caller sends as the gate is set today, is a set like any other.
#[test]
#[serial_test::serial]
fn a_large_filtered_set_is_answered_alike_by_a_scan_and_by_a_read_of_each_id() {
    use crate::helix_engine::vector_core::named_vectors::with_exact_scan_rows_per_id_for_test;

    let _store = set_lsm_test_object_store(Arc::new(object_store::memory::InMemory::new()));
    let _prefix = EnvVar::set("HELIX_LSM_PREFIX", "exact_scan_replica");
    let _switch = EnvVar::set("HELIX_LSM_REUPSERT_DELETE", "0");
    // As the readers' manifest sets it: a read pinned to one snapshot.
    let _pinned = EnvVar::set("HELIX_LSM_READER_SNAPSHOT_READS", "1");
    let name = "exact_scan";
    let writer = setup_with_config(object_store_config(LsmRole::Writer));
    create_dense_collection_lsm(&writer, name);
    let writer = Points::open(writer, name);
    writer
        .storage
        .create_payload_index("kind", PayloadIndexSchema::Keyword)
        .unwrap();
    let points = |ids: std::ops::RangeInclusive<u64>, turn: f32| -> serde_json::Value {
        ids.map(|id| {
            let angle = id as f32 * 0.37 + turn;
            json!({
                "id": id,
                "vector": {"dense": [angle.cos(), angle.sin(), (id % 17) as f32 / 17.0]},
                "payload": {"kind": if id % 9 < 7 { "a" } else { "b" }}
            })
        })
        .collect()
    };
    writer.call("upsert", json!({"points": points(1..=900, 0.0)}));
    writer.seal();
    writer.call("upsert", json!({"points": points(1..=60, 1.3)}));
    let copies = writer.rows().remove(&1).unwrap_or_default();
    assert_eq!(copies.len(), 2, "an old and a new copy: {copies:?}");

    let reader = Points::open(
        setup_with_config(object_store_config(LsmRole::Reader)),
        name,
    );
    assert!(reader.storage.backend.is_reader_replica());
    // A replica lays a segment's sidecar file out at its first read of it;
    // here before the searches, so that both are asked with the files open.
    allow_lsm_blocking(|| {
        let read = reader.storage.backend.begin_read().unwrap();
        for segment in reader.segments() {
            reader
                .storage
                .named_vectors
                .with_core(&segment, |core| {
                    if !core.is_empty(&read)? {
                        core.ensure_lsm_mmap_sidecar()?;
                        assert!(core.has_mmap_sidecar(), "{segment} has no sidecar");
                    }
                    Ok(())
                })
                .unwrap();
        }
    });
    let asked = |route: &str, filter: &serde_json::Value, rows_per_id: Option<usize>| {
        let query = [0.3, -0.8, 0.5];
        let body = match route {
            "query" => json!({"query": query, "using": "dense", "limit": 30, "filter": filter,
                "with_payload": true, "with_vector": true}),
            _ => json!({"dense": [{"query": query, "using": "dense", "limit": 30}],
                "filter": filter, "limit": 30, "with_payload": true,
                "with_channel_points": true}),
        };
        let call = || exact_scorer_way(|| reader.call(route, body));
        match rows_per_id {
            Some(rows) => with_exact_scan_rows_per_id_for_test(rows, call),
            None => call(),
        }
    };

    let kind_a = json!({"must": [{"key": "kind", "match": {"value": "a"}}]});
    for route in ["query", "hybrid_query"] {
        let (read_each, why) = asked(route, &kind_a, Some(0));
        assert_eq!(why, "per_id_off", "{route}");
        let (scanned, why) = asked(route, &kind_a, None);
        assert_eq!(why, "scan", "{route}");
        assert_eq!(scanned, read_each, "{route}");
    }

    let listed = |ids: u64| json!({"must": [{"has_id": (1..=ids).collect::<Vec<u64>>()}]});
    for ids in [200, 512] {
        assert_eq!(
            asked("query", &listed(ids), None).1,
            "per_id_small",
            "{ids} ids"
        );
    }
    let (read_each, _) = asked("query", &listed(600), Some(0));
    let (scanned, why) = asked("query", &listed(600), None);
    assert_eq!(why, "scan");
    assert_eq!(scanned, read_each);

    // A reader whose reads pin no snapshot, which is the key's default in
    // the code, reads each id: a scan's walk and its row reads could see two
    // states of the store there.
    let _unpinned = EnvVar::set("HELIX_LSM_READER_SNAPSHOT_READS", "0");
    let (unpinned, why) = asked("query", &listed(600), None);
    assert_eq!(why, "per_id_handle");
    assert_eq!(unpinned, read_each);
}
