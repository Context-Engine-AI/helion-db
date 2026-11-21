//! A rebuild leaves an edge that is stored exactly as its row derives it.
//!
//! `upsert_edge_be` read the stored edge of every derived edge and then wrote
//! its row, both adjacency keys and both path keys whatever it had read. On
//! LSM the rebuild for paths, the chunk-edge half of its request and the
//! backfill from points now read the rows of a call in one batched read and
//! the keys of the equal ones in a second
//! (`HelixGraphStorage::upsert_derived_edges_be`), and stage nothing for an
//! edge whose stored row equals the derived one and whose keys are stored.
//! `HELIX_GRAPH_REWRITE_UNCHANGED_EDGES=1` keeps the loop that writes every
//! edge: the tests run one store each way and hold the two against each
//! other key by key.
//!
//! Nothing of the old repair is given up, and that is pinned here: an equal
//! row that lacks any one of its keys is written again, and so is the absent
//! node row of an equal edge's end. The keys are read only for an equal row.
//! Of two rows of one call that derive one id the later is the edge, as the
//! loop ends: they are one before the read.
//!
//! The tests are serial among themselves, as each reads the switch and some
//! set it. They do not hold the graph tests' `ENV_LOCK`: one of them failing
//! under it would poison it for the tests that unwrap it.

use super::tests::{
    make_relationship_point, post_json, setup_context_with_config, EnvVarGuard, TestContext,
};
use super::*;
use crate::helix_engine::graph_core::config::Config;
use crate::helix_engine::storage_core::backend_any::AnyBackend;
use crate::helix_engine::storage_core::backend_lsm::{
    fail_next_present_for_update_for_test, lsm_round_trips_for_test as round_trips,
};
use crate::helix_engine::storage_core::upsert::NodeUpsert;
use crate::protocol::items::SerializedNode;
use serde_json::json;

const SWITCH: &str = "HELIX_GRAPH_REWRITE_UNCHANGED_EDGES";
const GRAPH: &str = "repo_graph";
const BASE: &str = "repo";

fn lsm() -> TestContext {
    setup_context_with_config(Config::default().with_lsm_in_memory())
}

/// A relationship row with the keys the client stores on one: what an edge is derived
/// from, and what the edge row carries along (its lines, a list, the stamp).
fn row(
    id: u128,
    caller: &str,
    callee: &str,
    caller_path: &str,
    callee_path: &str,
    line: i64,
) -> NodeUpsert {
    let mut point =
        make_relationship_point(id, caller, callee, caller_path, callee_path, "calls", "r");
    let text = |value: &str| Value::String(value.to_string());
    point.properties.extend([
        ("confidence".to_string(), text("high")),
        ("language".to_string(), text("rust")),
        ("start_line".to_string(), Value::I64(line)),
        ("end_line".to_string(), Value::I64(line + 2)),
        ("start_col".to_string(), Value::I64(4)),
        ("end_col".to_string(), Value::I64(21)),
        (
            "callee_receivers".to_string(),
            Value::Array(vec![text("self"), text("store")]),
        ),
        (
            "__helix_content_hash__".to_string(),
            Value::U64(0x9E37_79B9_7F4A_7C15 ^ id as u64),
        ),
    ]);
    point
}

fn with(mut point: NodeUpsert, key: &str, value: Value) -> NodeUpsert {
    point.properties.insert(key.to_string(), value);
    point
}

fn store(storage: &HelixGraphStorage, points: &[NodeUpsert]) {
    storage
        .with_write_backend(|w| {
            for point in points {
                storage.upsert_node_be(w, point)?;
            }
            Ok(())
        })
        .unwrap();
}

fn remove(storage: &HelixGraphStorage, ids: &[u128]) {
    storage
        .with_write_backend(|w| storage.drop_node_rows_be(w, ids).map(|_| ()))
        .unwrap();
}

fn rebuild_body(ctx: &TestContext, body: serde_json::Value) -> serde_json::Value {
    let (status, reply) = post_json(ctx, handle_rebuild_adjacency_for_paths, body);
    assert_eq!(status, 200, "{reply}");
    reply
}

fn rebuild(ctx: &TestContext, paths: &[&str], replace: bool) -> serde_json::Value {
    rebuild_body(
        ctx,
        json!({"collection": GRAPH, "paths": paths, "replace_outgoing": replace}),
    )
}

fn backfill(ctx: &TestContext, body: serde_json::Value) -> serde_json::Value {
    let (status, reply) = post_json(ctx, handle_backfill_adjacency_from_points, body);
    assert_eq!(status, 200, "{reply}");
    reply
}

/// One forced call that takes every row: a pass from its start to its mark.
fn forced_pass(ctx: &TestContext) -> serde_json::Value {
    let reply = backfill(ctx, json!({"collection": GRAPH, "force": true}));
    assert_eq!(reply["complete"], true, "{reply}");
    reply
}

/// The mark of a finished pass, as stored.
fn mark(storage: &HelixGraphStorage) -> Option<Vec<u8>> {
    let r = storage.backend.begin_read().unwrap();
    storage
        .backend
        .get_with(
            &r,
            Namespace::Metadata,
            b"adj_backfill_from_points:complete",
            |value| value.map(<[u8]>::to_vec),
        )
        .unwrap()
}

/// The edges a rebuild or a pass derived, and how many of them it left.
fn edges(reply: &serde_json::Value) -> (u64, u64) {
    (
        reply["edges_upserted"].as_u64().unwrap(),
        reply["edges_unchanged"].as_u64().unwrap(),
    )
}

fn chunk_edges(reply: &serde_json::Value) -> (u64, u64) {
    (
        reply["chunk_edges_upserted"].as_u64().unwrap(),
        reply["chunk_edges_unchanged"].as_u64().unwrap(),
    )
}

/// A value with its type, so that `5` and `5.0` are told apart here though
/// `Value` holds them equal.
fn typed(value: &Value) -> String {
    match value {
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(typed).collect::<Vec<_>>().join(",")
        ),
        Value::Object(map) => format!("{{{}}}", properties(map)),
        other => format!("{other:?}"),
    }
}

fn properties(map: &HashMap<String, Value>) -> String {
    let mut entries: Vec<String> = map
        .iter()
        .map(|(key, value)| format!("{key}={}", typed(value)))
        .collect();
    entries.sort();
    entries.join(" ")
}

/// Every node and edge row decoded, its properties in key order, and every
/// adjacency and path key as stored.
#[derive(PartialEq)]
struct Whole {
    nodes: Vec<String>,
    edges: Vec<String>,
    keys: Vec<(usize, Vec<u8>, Vec<u8>)>,
}

fn whole(storage: &HelixGraphStorage) -> Whole {
    let r = storage.backend.begin_read().unwrap();
    let id = |key: &[u8]| u128::from_be_bytes(key.try_into().unwrap());
    let (mut nodes, mut edges, mut keys) = (Vec::new(), Vec::new(), Vec::new());
    storage
        .backend
        .scan(&r, Namespace::Nodes, KeyRange::all(), |key, value| {
            let node = SerializedNode::decode_node(value, id(key)).unwrap();
            nodes.push(format!(
                "{:032x} {} {}",
                node.id,
                node.label,
                properties(&node.properties)
            ));
            true
        })
        .unwrap();
    storage
        .backend
        .scan(&r, Namespace::Edges, KeyRange::all(), |key, value| {
            let edge = SerializedEdge::decode_edge(value, id(key)).unwrap();
            edges.push(format!(
                "{:032x} {} {:032x} {:032x} {}",
                edge.id,
                edge.label,
                edge.from_node,
                edge.to_node,
                properties(&edge.properties)
            ));
            true
        })
        .unwrap();
    for (which, namespace) in [
        Namespace::OutEdges,
        Namespace::InEdges,
        Namespace::EdgePathIdx,
    ]
    .into_iter()
    .enumerate()
    {
        storage
            .backend
            .scan(&r, namespace, KeyRange::all(), |key, value| {
                keys.push((which, key.to_vec(), value.to_vec()));
                true
            })
            .unwrap();
    }
    Whole { nodes, edges, keys }
}

fn three_rows_of_a_and_one_of_b() -> [NodeUpsert; 4] {
    [
        row(1, "a_fn", "b_fn", "src/a.rs", "src/b.rs", 10),
        row(2, "a_fn", "c_fn", "src/a.rs", "src/c.rs", 14),
        row(3, "a2_fn", "ext", "src/a.rs", "<external>/lib", 30),
        row(4, "b_fn", "a_fn", "src/b.rs", "src/a.rs", 5),
    ]
}

/// The edge `0xE1`, with its two path properties and one more.
fn confident(confidence: &str) -> EdgeUpsert {
    EdgeUpsert {
        id: 0xE1,
        label: "CALLS".into(),
        from_node: 0xA,
        to_node: 0xB,
        properties: HashMap::from([
            ("from_path".to_string(), Value::String("src/a.rs".into())),
            ("to_path".to_string(), Value::String("src/b.rs".into())),
            ("confidence".to_string(), Value::String(confidence.into())),
        ]),
    }
}

fn store_by_the_loop(storage: &HelixGraphStorage, upserts: &[EdgeUpsert]) {
    storage
        .with_write_backend(|w| {
            for upsert in upserts {
                storage.upsert_edge_be(w, upsert)?;
            }
            Ok(())
        })
        .unwrap();
}

/// The keys a write handle has staged.
fn pending(w: &AnyWrite<'_>) -> usize {
    let AnyWrite::Lsm(batch) = w else {
        panic!("an LSM collection")
    };
    batch.pending().len()
}

/// The edges one call leaves and the keys it stages, in a batch of its own
/// that is never stored.
fn staged(storage: &HelixGraphStorage, upserts: &[EdgeUpsert]) -> (usize, usize) {
    let mut w = storage.backend.begin_write().unwrap();
    let left = storage.upsert_derived_edges_be(&mut w, upserts).unwrap();
    (left, pending(&w))
}

/// The stored `confidence` of the edge `0xE1`, with its type.
fn stored_confidence(storage: &HelixGraphStorage) -> String {
    let edge = storage
        .with_read_backend(|r| storage.get_edge_be(r, &0xE1))
        .unwrap();
    typed(&edge.properties["confidence"])
}

/// An edge stored as derived stages nothing; with one property changed it
/// stages its five keys, as the loop a caller off the leave keeps stages
/// them whatever is stored. And the compare is made on what the batch holds:
/// a row the batch has changed is the row, and an edge the batch itself has
/// staged, row and keys, is left though the store does not hold it yet. On
/// LMDB, and with the switch, a caller keeps the loop.
#[test]
#[serial_test::serial]
fn an_edge_that_is_stored_as_derived_stages_nothing() {
    let ctx = lsm();
    let storage = ctx.collections.create_collection(GRAPH).unwrap();
    store_by_the_loop(&storage, &[confident("high")]);

    assert_eq!(
        staged(&storage, &[confident("high")]),
        (1, 0),
        "stored as derived"
    );
    assert_eq!(
        staged(&storage, &[confident("low")]),
        (0, 5),
        "one property differs: the row, two adjacency keys and two path keys"
    );
    let by_the_loop = {
        let mut w = storage.backend.begin_write().unwrap();
        storage.upsert_edge_be(&mut w, &confident("high")).unwrap();
        pending(&w)
    };
    assert_eq!(by_the_loop, 5, "the loop writes an equal edge again");
    assert_eq!(staged(&storage, &[]), (0, 0));

    storage
        .with_write_backend(|w| {
            assert_eq!(storage.upsert_derived_edges_be(w, &[confident("low")])?, 0);
            assert_eq!(
                storage.upsert_derived_edges_be(w, &[confident("high")])?,
                0,
                "the batch holds the other row by now"
            );
            Ok(())
        })
        .unwrap();
    assert!(stored_confidence(&storage).contains("high"));

    let unstored = EdgeUpsert {
        id: 0xE2,
        ..confident("high")
    };
    storage
        .with_write_backend(|w| {
            assert_eq!(
                storage.upsert_derived_edges_be(w, std::slice::from_ref(&unstored))?,
                0
            );
            assert_eq!(
                storage.upsert_derived_edges_be(w, std::slice::from_ref(&unstored))?,
                1,
                "the batch holds its row and its keys by now"
            );
            Ok(())
        })
        .unwrap();

    assert!(storage.leaves_edges_stored_as_derived());
    {
        let _rewrite = EnvVarGuard::set(SWITCH, "1");
        assert!(!storage.leaves_edges_stored_as_derived());
    }
    let lmdb = setup_context_with_config(Config::default());
    let on_lmdb = lmdb.collections.create_collection(GRAPH).unwrap();
    assert_eq!(on_lmdb.backend.kind(), BackendKind::Lmdb);
    assert!(
        !on_lmdb.leaves_edges_stored_as_derived(),
        "LMDB keeps its repair"
    );
    store(&on_lmdb, &three_rows_of_a_and_one_of_b());
    for _ in 0..2 {
        assert_eq!(
            edges(&rebuild(&lmdb, &["src/a.rs", "src/b.rs"], false)),
            (4, 0),
            "LMDB: every edge is written, by the loop"
        );
        assert_eq!(edges(&forced_pass(&lmdb)), (4, 0), "LMDB: a pass");
    }
}

/// Two rows of one call that derive one id are one edge before the store is
/// read: the later, as a loop of upserts ends. Where the stored row equals
/// the later nothing is staged, though the earlier differs; where it equals
/// the earlier the later is staged, once. Either way the store ends as the
/// later has it, and holds what a store written by the loop holds.
#[test]
#[serial_test::serial]
fn of_two_rows_of_one_call_that_derive_one_id_the_later_is_the_edge() {
    let ctx = lsm();
    let storage = ctx.collections.create_collection(GRAPH).unwrap();
    store_by_the_loop(&storage, &[confident("high")]);

    assert_eq!(
        staged(&storage, &[confident("low"), confident("high")]),
        (2, 0),
        "the stored row equals the later: nothing is staged, though the earlier differs"
    );
    assert_eq!(
        staged(&storage, &[confident("high"), confident("low")]),
        (0, 5),
        "the later differs: its five keys, once"
    );
    assert_eq!(
        staged(
            &storage,
            &[confident("low"), confident("medium"), confident("high")]
        ),
        (3, 0),
        "three rows"
    );

    let calls: [&[EdgeUpsert]; 3] = [
        &[confident("low"), confident("high")],
        &[confident("high"), confident("low")],
        &[confident("medium"), confident("low")],
    ];
    let by_the_loop = lsm();
    let looped = by_the_loop.collections.create_collection(GRAPH).unwrap();
    store_by_the_loop(&looped, &[confident("high")]);
    for (call, (left, ends)) in calls.into_iter().zip([(2, "high"), (0, "low"), (2, "low")]) {
        let answered = storage
            .with_write_backend(|w| storage.upsert_derived_edges_be(w, call))
            .unwrap();
        assert_eq!(answered, left, "ends {ends}");
        assert!(
            stored_confidence(&storage).contains(ends),
            "as the later has it"
        );
        store_by_the_loop(&looped, call);
        assert!(whole(&storage) == whole(&looped), "ends {ends}: the loop");
    }
}

/// What a call reads. The rows come from one batched read, with its
/// snapshot, in place of a get an edge, and each id is looked up once; the
/// keys of the equal rows come from a second, four an edge here. So a call
/// whose edges are all stored as derived costs four round trips whatever
/// its size, two for its rows and two for their keys: more than the loop's
/// N + 1, a get an edge and the commit its request then makes, for a call
/// of one or two edges, equal at three and fewer from four. No key is read
/// for a row that is new or changed, and nothing for no row.
#[test]
#[serial_test::serial]
fn a_call_reads_its_rows_in_one_read_and_the_keys_of_its_equal_rows_in_a_second() {
    let ctx = lsm();
    let storage = ctx.collections.create_collection(GRAPH).unwrap();
    let AnyBackend::Lsm(backend) = &*storage.backend else {
        panic!("an LSM collection")
    };
    let edge = |n: u128, confidence: &str| EdgeUpsert {
        id: 0xE0 + n,
        to_node: 0xB0 + n,
        ..confident(confidence)
    };
    let first =
        |edges: u128, confidence: &str| (0..edges).map(|n| edge(n, confidence)).collect::<Vec<_>>();
    store_by_the_loop(&storage, &first(1_000, "high"));

    // Round trips, keys looked up and edges left, in a batch never stored.
    let reads = |upserts: &[EdgeUpsert]| {
        let mut w = storage.backend.begin_write().unwrap();
        backend.reset_read_count();
        let before = round_trips();
        let left = storage.upsert_derived_edges_be(&mut w, upserts).unwrap();
        (round_trips() - before, backend.read_count(), left)
    };
    // The loop's round trips over the same edges, in a batch never stored.
    let by_the_loop = |upserts: &[EdgeUpsert]| {
        let mut w = storage.backend.begin_write().unwrap();
        let before = round_trips();
        for upsert in upserts {
            storage.upsert_edge_be(&mut w, upsert).unwrap();
        }
        round_trips() - before
    };
    for edges in [1usize, 2, 3, 1_000] {
        let equal = first(edges as u128, "high");
        assert_eq!(
            reads(&equal),
            (4, edges + edges * 4, edges),
            "{edges} equal edges: four round trips, a lookup an id and four keys an edge"
        );
        assert_eq!(by_the_loop(&equal), edges, "{edges} edges: a get an edge");
        assert_eq!(
            reads(&first(edges as u128, "low")),
            (2, edges, 0),
            "{edges} changed edges: no key is read"
        );
    }

    let mut one_changed = first(5, "high");
    one_changed[2] = edge(2, "low");
    assert_eq!(
        reads(&one_changed),
        (4, 5 + 4 * 4, 4),
        "one of five changed"
    );
    let twice = [first(5, "high"), first(5, "high")].concat();
    assert_eq!(
        reads(&twice),
        (4, 5 + 5 * 4, 10),
        "an id that two rows derive is looked up once"
    );
    assert_eq!(reads(&[]), (0, 0, 0));

    // New edges, against the loop: both seed the edge counter alike.
    let new = (1_000..1_005).map(|n| edge(n, "high")).collect::<Vec<_>>();
    let by_the_call = reads(&new);
    assert_eq!(
        (by_the_call.1, by_the_call.2),
        (5, 0),
        "no key of a new edge is read"
    );
    assert_eq!(
        by_the_call.0 + 5,
        by_the_loop(&new) + 2,
        "one read of five rows, with its snapshot, for five gets"
    );
}

/// A second rebuild of rows that did not change says so and makes no write
/// batch, with or without `replace_outgoing`. It reads the four rows in one
/// read and their sixteen keys in a second, a snapshot each: four round
/// trips where the same request with the switch set makes five, a get an
/// edge and the commit. Either way the store holds what the first rebuild
/// wrote.
#[test]
#[serial_test::serial]
fn a_rebuild_leaves_the_edges_that_are_stored_as_their_rows_derive_them() {
    let ctx = lsm();
    let storage = ctx.collections.create_collection(GRAPH).unwrap();
    store(&storage, &three_rows_of_a_and_one_of_b());
    let paths = ["src/a.rs", "src/b.rs"];

    assert_eq!(
        edges(&rebuild(&ctx, &paths, false)),
        (4, 0),
        "no edge is stored yet"
    );
    let written = whole(&storage);
    assert_eq!(written.edges.len(), 4);
    assert_eq!(
        written.keys.len(),
        16,
        "an out, an in and two path keys an edge"
    );

    for replace in [false, true] {
        // Once unmeasured, so that neither measured request is the first of
        // its kind.
        rebuild(&ctx, &paths, replace);
        let before = round_trips();
        assert_eq!(
            edges(&rebuild(&ctx, &paths, replace)),
            (4, 4),
            "replace={replace}"
        );
        let left = round_trips() - before;
        assert!(whole(&storage) == written, "replace={replace}: left");

        let _rewrite = EnvVarGuard::set(SWITCH, "1");
        let before = round_trips();
        assert_eq!(
            edges(&rebuild(&ctx, &paths, replace)),
            (4, 0),
            "replace={replace}: the switch"
        );
        assert_eq!(
            round_trips() - before + (2 + 2),
            left + (4 + 1),
            "replace={replace}: a get an edge and the commit, against the two batched reads"
        );
        assert!(whole(&storage) == written, "replace={replace}: rewritten");
    }
}

const FILES: usize = 4;
const ROWS: usize = 6;

fn file(f: usize) -> String {
    format!("src/f{f}.rs")
}

fn point_id(f: usize, i: usize) -> u128 {
    0x1000 * (f as u128 + 1) + i as u128
}

/// Row `i` of file `f`: three callers a file, a callee of its own, every
/// third one outside the repository and so shared by the files.
fn file_row(f: usize, i: usize) -> NodeUpsert {
    let callee_path = if i % 3 == 2 {
        "<external>/lib".to_string()
    } else {
        file((f + 1 + i % 2) % FILES)
    };
    row(
        point_id(f, i),
        &format!("f{f}_c{}", i % 3),
        &format!("t{i}"),
        &file(f),
        &callee_path,
        10 * (i as i64 + 1),
    )
}

/// One flush as Helion receives it: what changed in the stored rows, then
/// the rebuild of the files named.
struct Step {
    what: &'static str,
    change: fn(&HelixGraphStorage),
    files: &'static [usize],
    replace: bool,
}

fn steps() -> Vec<Step> {
    vec![
        Step {
            what: "a first index",
            change: |storage| {
                let rows: Vec<NodeUpsert> = (0..FILES)
                    .flat_map(|f| (0..ROWS).map(move |i| file_row(f, i)))
                    .collect();
                store(storage, &rows);
            },
            files: &[0, 1, 2, 3],
            replace: false,
        },
        Step {
            what: "nothing changed",
            change: |_| {},
            files: &[0, 1, 2, 3],
            replace: false,
        },
        Step {
            what: "two rows change in payload only",
            change: |storage| {
                store(
                    storage,
                    &[
                        with(file_row(0, 1), "confidence", Value::String("low".into())),
                        with(
                            file_row(2, 3),
                            "callee_point_id",
                            Value::String("77".into()),
                        ),
                    ],
                );
            },
            files: &[0, 2],
            replace: true,
        },
        Step {
            what: "two rows name another callee file",
            change: |storage| {
                let moved = || Value::String("src/moved.rs".into());
                store(
                    storage,
                    &[
                        with(file_row(0, 3), "callee_path", moved()),
                        with(file_row(1, 0), "callee_path", moved()),
                    ],
                );
            },
            files: &[0, 1],
            replace: true,
        },
        Step {
            what: "an edit moves the lines of every row below it",
            change: |storage| {
                let rows: Vec<NodeUpsert> = (2..ROWS)
                    .map(|i| {
                        let line = 10 * (i as i64 + 1) + 5;
                        with(
                            with(file_row(3, i), "start_line", Value::I64(line)),
                            "end_line",
                            Value::I64(line + 2),
                        )
                    })
                    .collect();
                store(storage, &rows);
            },
            files: &[3],
            replace: true,
        },
        Step {
            what: "two rows are gone",
            change: |storage| remove(storage, &[point_id(1, 4), point_id(1, 5)]),
            files: &[1],
            replace: true,
        },
        Step {
            what: "two rows are new",
            change: |storage| store(storage, &[file_row(2, 6), file_row(2, 7)]),
            files: &[2],
            replace: false,
        },
        Step {
            what: "a second row derives the id of a stored edge",
            change: |storage| {
                let mut twin = with(file_row(0, 0), "confidence", Value::String("low".into()));
                twin.id = point_id(0, 0) + 0x800;
                store(storage, &[twin]);
            },
            files: &[0],
            replace: true,
        },
        Step {
            what: "the first of the two rows changes, the second is as stored",
            change: |storage| {
                store(
                    storage,
                    &[with(
                        file_row(0, 0),
                        "confidence",
                        Value::String("medium".into()),
                    )],
                );
            },
            files: &[0],
            replace: true,
        },
        Step {
            what: "nothing changed, every file",
            change: |_| {},
            files: &[0, 1, 2, 3],
            replace: false,
        },
    ]
}

/// What one store holds after each step, and what each rebuild answered.
struct Run {
    after: Vec<Whole>,
    edges: Vec<(u64, u64)>,
    dropped: Vec<(u64, u64)>,
    counters: (u64, u64, u64, u64),
}

fn run(rewrite: bool) -> Run {
    let _rewrite = rewrite.then(|| EnvVarGuard::set(SWITCH, "1"));
    let ctx = lsm();
    let storage = ctx.collections.create_collection(GRAPH).unwrap();
    storage
        .create_payload_index("caller_path", PayloadIndexSchema::Keyword)
        .unwrap();
    // A covered graph: a replace finds its edges through the path index.
    while !storage.backfill_edge_path_index_batch(4).unwrap().1 {}

    let (mut after, mut derived, mut dropped) = (Vec::new(), Vec::new(), Vec::new());
    for step in steps() {
        (step.change)(&storage);
        let paths: Vec<String> = step.files.iter().map(|f| file(*f)).collect();
        let reply = rebuild_body(
            &ctx,
            json!({"collection": GRAPH, "paths": paths, "replace_outgoing": step.replace}),
        );
        after.push(whole(&storage));
        derived.push(edges(&reply));
        dropped.push((
            reply["edges_deleted"].as_u64().unwrap(),
            reply["nodes_deleted"].as_u64().unwrap(),
        ));
    }
    let recount = storage.recount_lsm_counters().unwrap();
    Run {
        after,
        edges: derived,
        dropped,
        counters: (
            recount.nodes.before,
            recount.nodes.after,
            recount.edges.before,
            recount.edges.after,
        ),
    }
}

/// Two stores take the same flushes, one writing every derived edge again
/// and one leaving those stored as derived. After every flush they hold the
/// same node rows, edge rows, adjacency keys and path keys, and at the end
/// the same counters, which a recount confirms. The flushes: a first index,
/// rows changed in payload only, rows pointed at another file, an edit that
/// moves the lines of the rows below it (the share of a file's rows that is
/// left is the share above the edit), rows gone, rows new, and one id
/// derived by two rows.
#[test]
#[serial_test::serial]
fn a_store_that_leaves_equal_edges_holds_what_one_that_rewrites_them_holds() {
    let rewritten = run(true);
    let left = run(false);

    for (at, step) in steps().iter().enumerate() {
        assert!(
            left.after[at] == rewritten.after[at],
            "{}: the two stores hold the same rows and keys",
            step.what
        );
        assert_eq!(left.dropped[at], rewritten.dropped[at], "{}", step.what);
        assert_eq!(rewritten.edges[at].1, 0, "{}: the switch", step.what);
    }
    assert_eq!(
        rewritten.edges.iter().map(|e| e.0).collect::<Vec<_>>(),
        [24, 24, 12, 12, 6, 4, 8, 7, 7, 25],
        "rows derived"
    );
    assert_eq!(
        left.edges,
        [
            (24, 0),
            (24, 24),
            (12, 10),
            (12, 10),
            // The two rows above the edit.
            (6, 2),
            (4, 4),
            (8, 6),
            // The twin is the later of the two rows and differs from the
            // store: their edge is written, as the twin has it.
            (7, 5),
            // The later of the two is as stored: their edge is left, though
            // the first has changed.
            (7, 7),
            (25, 25),
        ],
        "rows derived, and of them left"
    );
    assert_eq!(left.dropped[3], (2, 2), "the two edges re-pointed away");
    assert_eq!(left.dropped[5].0, 2, "the two rows that are gone");
    assert_eq!(left.counters, rewritten.counters);
    assert_eq!(
        (left.counters.0, left.counters.2),
        (left.counters.1, left.counters.3),
        "stored node and edge counters against a recount"
    );
}

/// A replace among unchanged rows: the row that is gone loses its edge and
/// the node that leaves alone, the row that changed is written, and the row
/// that did neither is left.
#[test]
#[serial_test::serial]
fn a_replace_among_unchanged_rows_writes_only_what_changed() {
    let ctx = lsm();
    let storage = ctx.collections.create_collection(GRAPH).unwrap();
    let rows = three_rows_of_a_and_one_of_b();
    store(&storage, &rows);
    rebuild(&ctx, &["src/a.rs", "src/b.rs"], false);
    let before = whole(&storage);

    remove(&storage, &[2]);
    store(
        &storage,
        &[with(rows[2].clone(), "start_line", Value::I64(31))],
    );
    let reply = rebuild(&ctx, &["src/a.rs"], true);
    assert_eq!(edges(&reply), (2, 1), "{reply}");
    assert_eq!(
        (
            reply["edges_deleted"].as_u64(),
            reply["nodes_deleted"].as_u64()
        ),
        (Some(1), Some(1)),
        "the edge to `c_fn`, and `c_fn`: {reply}"
    );

    let after = whole(&storage);
    assert_eq!(after.edges.len(), 3);
    assert_eq!(after.keys.len(), 12, "the gone edge's four keys with it");
    let edge_of = |caller: &str, callee: &str, callee_path: &str| {
        deterministic_id::edge_id(GRAPH, "CALLS", caller, callee, "src/a.rs", callee_path)
    };
    let r = storage.backend.begin_read().unwrap();
    let stored = |id: u128| storage.get_edge_be(&r, &id);
    assert!(stored(edge_of("a_fn", "c_fn", "src/c.rs")).is_err());
    assert!(
        typed(
            &stored(edge_of("a2_fn", "ext", "<external>/lib"))
                .unwrap()
                .properties["start_line"]
        )
        .contains("31"),
        "the changed row's edge is written"
    );
    let kept = format!("{:032x} ", edge_of("a_fn", "b_fn", "src/b.rs"));
    let row_of = |state: &Whole| {
        state
            .edges
            .iter()
            .find(|edge| edge.starts_with(&kept))
            .cloned()
    };
    assert!(row_of(&after).is_some() && row_of(&after) == row_of(&before));
}

/// The edge `a_fn` to `b_fn` of [`three_rows_of_a_and_one_of_b`], as stored.
fn stored_edge_a_to_b(storage: &HelixGraphStorage) -> Edge {
    stored_edge(storage, ["a_fn", "b_fn", "src/a.rs", "src/b.rs"])
}

/// The stored call edge of a caller, a callee and their two paths.
fn stored_edge(storage: &HelixGraphStorage, [caller, callee, from, to]: [&str; 4]) -> Edge {
    let id = deterministic_id::edge_id(GRAPH, "CALLS", caller, callee, from, to);
    storage
        .with_read_backend(|r| storage.get_edge_be(r, &id))
        .unwrap()
}

/// The upsert that derives `edge` as it is stored.
fn as_upsert(edge: &Edge) -> EdgeUpsert {
    EdgeUpsert {
        id: edge.id,
        label: edge.label.clone(),
        from_node: edge.from_node,
        to_node: edge.to_node,
        properties: edge.properties.clone(),
    }
}

/// Stages the removal of one key of the stored `edge`: 0 its outgoing
/// adjacency row, 1 its incoming one, 2 its from-path key, 3 its to-path key.
fn take_key(
    storage: &HelixGraphStorage,
    w: &mut AnyWrite<'_>,
    edge: &Edge,
    which: usize,
) -> Result<(), GraphError> {
    let label = hash_label(&edge.label, None);
    match which {
        0 => storage.backend.delete_dup(
            w,
            Namespace::OutEdges,
            &HelixGraphStorage::out_edge_key(&edge.from_node, &label),
            &HelixGraphStorage::pack_edge_data(&edge.to_node, &edge.id),
        ),
        1 => storage.backend.delete_dup(
            w,
            Namespace::InEdges,
            &HelixGraphStorage::in_edge_key(&edge.to_node, &label),
            &HelixGraphStorage::pack_edge_data(&edge.from_node, &edge.id),
        ),
        _ => {
            // A path key is its direction, the path, a zero and the edge id.
            let direction = which as u8 - 1;
            let key = whole(storage)
                .keys
                .into_iter()
                .find(|(namespace, key, _)| {
                    *namespace == 2 && key[0] == direction && key.ends_with(&edge.id.to_be_bytes())
                })
                .expect("the edge has that path key")
                .1;
            storage.backend.delete(w, Namespace::EdgePathIdx, &key)
        }
    }
    .map_err(|e| GraphError::New(e.to_string()))
}

/// What `call` counted, as the metrics endpoint renders it.
fn metrics_of<T>(call: impl FnOnce() -> T) -> (T, String) {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let answer = metrics::with_local_recorder(&recorder, call);
    (answer, handle.render())
}

const ONE_EDGE_WRITTEN_FOR_A_KEY: &str =
    "helix_graph_equal_edge_rewritten_total{why=\"key_absent\"} 1";

/// An equal edge that lacks `which` of its four keys (see [`take_key`]) is
/// written again by a rebuild, which puts the key back: the repair an upsert
/// has always made, counted with its reason. The three edges that are whole
/// are left.
fn a_rebuild_writes_again_an_equal_edge_without(which: usize) {
    let ctx = lsm();
    let storage = ctx.collections.create_collection(GRAPH).unwrap();
    store(&storage, &three_rows_of_a_and_one_of_b());
    let paths = ["src/a.rs", "src/b.rs"];
    rebuild(&ctx, &paths, false);
    let healthy = whole(&storage);

    let edge = stored_edge_a_to_b(&storage);
    storage
        .with_write_backend(|w| take_key(&storage, w, &edge, which))
        .unwrap();
    let short = whole(&storage);
    assert_eq!(short.keys.len() + 1, healthy.keys.len());
    assert!(short.edges == healthy.edges, "the edge row is there");

    let (reply, counted) = metrics_of(|| rebuild(&ctx, &paths, false));
    assert_eq!(edges(&reply), (4, 3));
    assert!(whole(&storage) == healthy, "the key is back");
    assert!(
        counted
            .lines()
            .any(|line| line == ONE_EDGE_WRITTEN_FOR_A_KEY),
        "{counted}"
    );
}

#[test]
#[serial_test::serial]
fn a_rebuild_writes_again_an_equal_edge_without_its_outgoing_key() {
    a_rebuild_writes_again_an_equal_edge_without(0);
}

#[test]
#[serial_test::serial]
fn a_rebuild_writes_again_an_equal_edge_without_its_incoming_key() {
    a_rebuild_writes_again_an_equal_edge_without(1);
}

#[test]
#[serial_test::serial]
fn a_rebuild_writes_again_an_equal_edge_without_its_from_path_key() {
    a_rebuild_writes_again_an_equal_edge_without(2);
}

#[test]
#[serial_test::serial]
fn a_rebuild_writes_again_an_equal_edge_without_its_to_path_key() {
    a_rebuild_writes_again_an_equal_edge_without(3);
}

/// The keys are read through the write handle: one the batch has deleted is
/// gone though the store still holds it, so the equal edge is written and
/// the batch leaves the key where it was.
#[test]
#[serial_test::serial]
fn an_equal_edge_whose_key_its_batch_deletes_is_written_again() {
    let ctx = lsm();
    let storage = ctx.collections.create_collection(GRAPH).unwrap();
    store(&storage, &three_rows_of_a_and_one_of_b());
    rebuild(&ctx, &["src/a.rs", "src/b.rs"], false);
    let healthy = whole(&storage);

    let edge = stored_edge_a_to_b(&storage);
    let upsert = as_upsert(&edge);
    assert_eq!(staged(&storage, std::slice::from_ref(&upsert)), (1, 0));
    for which in 0..4 {
        storage
            .with_write_backend(|w| {
                take_key(&storage, w, &edge, which)?;
                assert_eq!(
                    storage.upsert_derived_edges_be(w, std::slice::from_ref(&upsert))?,
                    0,
                    "key {which} is deleted in this batch"
                );
                Ok(())
            })
            .unwrap();
        assert!(whole(&storage) == healthy, "key {which}");
    }
}

/// Every equal edge of a call takes its own answers of the one read of the
/// keys. Of two equal edges of one call, the first without its outgoing key
/// and the second without its to-path key, both are written again and both
/// keys are back: an edge that stopped at its first absent key would leave
/// the rest of its answers to the next, which would then be left as whole.
#[test]
#[serial_test::serial]
fn each_equal_edge_of_a_call_takes_its_own_answers_of_the_keys_read() {
    let ctx = lsm();
    let storage = ctx.collections.create_collection(GRAPH).unwrap();
    store(&storage, &three_rows_of_a_and_one_of_b());
    rebuild(&ctx, &["src/a.rs", "src/b.rs"], false);
    let healthy = whole(&storage);

    let first = stored_edge_a_to_b(&storage);
    let second = stored_edge(&storage, ["b_fn", "a_fn", "src/b.rs", "src/a.rs"]);
    storage
        .with_write_backend(|w| {
            take_key(&storage, w, &first, 0)?;
            take_key(&storage, w, &second, 3)
        })
        .unwrap();
    assert_eq!(whole(&storage).keys.len() + 2, healthy.keys.len());

    let call = [as_upsert(&first), as_upsert(&second)];
    let (left, counted) = metrics_of(|| {
        storage
            .with_write_backend(|w| storage.upsert_derived_edges_be(w, &call))
            .unwrap()
    });
    assert_eq!(left, 0, "each lacks a key");
    assert!(whole(&storage) == healthy, "both keys are back");
    let two_written = "helix_graph_equal_edge_rewritten_total{why=\"key_absent\"} 2";
    assert!(counted.lines().any(|line| line == two_written), "{counted}");
}

/// A read of the keys that fails is a key not seen: the rebuild writes every
/// equal edge of that call, as the loop writes it, and counts each with that
/// reason. A key that was absent is back, and the next rebuild reads its
/// keys and leaves all four.
#[test]
#[serial_test::serial]
fn a_rebuild_whose_read_of_the_keys_fails_writes_its_equal_edges() {
    let ctx = lsm();
    let storage = ctx.collections.create_collection(GRAPH).unwrap();
    store(&storage, &three_rows_of_a_and_one_of_b());
    let paths = ["src/a.rs", "src/b.rs"];
    rebuild(&ctx, &paths, false);
    let healthy = whole(&storage);
    let edge = stored_edge_a_to_b(&storage);
    storage
        .with_write_backend(|w| take_key(&storage, w, &edge, 0))
        .unwrap();

    fail_next_present_for_update_for_test();
    let (reply, counted) = metrics_of(|| rebuild(&ctx, &paths, false));
    assert_eq!(edges(&reply), (4, 0));
    assert!(whole(&storage) == healthy, "the key is back");
    let four_unread = "helix_graph_equal_edge_rewritten_total{why=\"keys_unread\"} 4";
    assert!(counted.lines().any(|line| line == four_unread), "{counted}");
    assert_eq!(edges(&rebuild(&ctx, &paths, false)), (4, 4));
}

/// The node rows are written outside the edge's leave: where the node row
/// of an equal edge's end is gone, the rebuild writes it again, with its
/// counter, and leaves the edge.
#[test]
#[serial_test::serial]
fn a_rebuild_writes_the_absent_node_row_of_an_equal_edges_end() {
    let ctx = lsm();
    let storage = ctx.collections.create_collection(GRAPH).unwrap();
    store(&storage, &three_rows_of_a_and_one_of_b());
    rebuild(&ctx, &["src/a.rs", "src/b.rs"], false);
    let healthy = whole(&storage);

    // `c_fn` is called by `a_fn` and is the end of no other edge.
    let callee = deterministic_id::node_id(GRAPH, "Symbol", "c_fn", "src/c.rs");
    remove(&storage, &[callee]);
    let without = whole(&storage);
    assert_eq!(without.nodes.len() + 1, healthy.nodes.len());
    assert!(without.edges == healthy.edges && without.keys == healthy.keys);

    assert_eq!(edges(&rebuild(&ctx, &["src/a.rs"], false)), (3, 3));
    assert!(whole(&storage) == healthy, "the node row is back");
    let recount = storage.recount_lsm_counters().unwrap();
    assert_eq!(
        (recount.nodes.before, recount.edges.before),
        (recount.nodes.after, recount.edges.after),
        "stored node and edge counters against a recount"
    );
}

/// A rebuild whose edges are all new reads no key. Against the same rebuild
/// with the switch set, which reads a row a get, it reads the four rows in
/// one read with its snapshot: two round trips for four, and the four rows
/// on the counter of keys that a batched read looks up.
#[test]
#[serial_test::serial]
fn a_rebuild_of_new_edges_reads_their_rows_in_one_read_and_no_key() {
    let first_rebuild = |rewrite: bool| {
        let _rewrite = rewrite.then(|| EnvVarGuard::set(SWITCH, "1"));
        let ctx = lsm();
        let storage = ctx.collections.create_collection(GRAPH).unwrap();
        let AnyBackend::Lsm(backend) = &*storage.backend else {
            panic!("an LSM collection")
        };
        store(&storage, &three_rows_of_a_and_one_of_b());
        backend.reset_read_count();
        let before = round_trips();
        assert_eq!(
            edges(&rebuild(&ctx, &["src/a.rs", "src/b.rs"], false)),
            (4, 0)
        );
        (round_trips() - before, backend.read_count())
    };
    let (left, rewritten) = (first_rebuild(false), first_rebuild(true));
    assert_eq!(left.0 + 4, rewritten.0 + 2, "round trips");
    assert_eq!(left.1, rewritten.1 + 4, "the four rows, and no key");
}

/// The chunk edges of the base collection: one that is stored, with its two
/// adjacency keys, is left, and a request whose chunk edges are all stored
/// makes no write batch there either. A candidate whose point is not there
/// is no edge, and is not read as one. Each collection's edges cost two
/// batched reads, a snapshot each, where the switch's request makes a get an
/// edge and a commit: three and one for the graph, two and one for the base.
/// A chunk edge that is gone, as after its point was deleted, is written.
#[test]
#[serial_test::serial]
fn a_rebuild_leaves_the_chunk_edges_that_are_stored() {
    let ctx = lsm();
    let storage = ctx.collections.create_collection(GRAPH).unwrap();
    let base = ctx.collections.create_collection(BASE).unwrap();
    let chunk = |id: u128| NodeUpsert {
        id,
        label: "point".into(),
        properties: HashMap::from([("path".to_string(), Value::String("src/a.rs".into()))]),
    };
    store(&base, &[chunk(11), chunk(12), chunk(13)]);
    let between = |point: NodeUpsert, callee: &str| {
        with(
            with(point, "caller_point_id", Value::String("11".into())),
            "callee_point_id",
            Value::String(callee.into()),
        )
    };
    let rows = three_rows_of_a_and_one_of_b();
    store(
        &storage,
        &[
            between(rows[0].clone(), "12"),
            between(rows[1].clone(), "13"),
            // No point 99: no chunk edge.
            between(rows[2].clone(), "99"),
        ],
    );
    let body = json!({"collection": GRAPH, "paths": ["src/a.rs"], "base_collection": BASE});

    let reply = rebuild_body(&ctx, body.clone());
    assert_eq!((edges(&reply), chunk_edges(&reply)), ((3, 0), (2, 0)));
    let written = whole(&base);
    assert_eq!((written.edges.len(), written.keys.len()), (2, 4));

    let before = round_trips();
    let reply = rebuild_body(&ctx, body.clone());
    let left = round_trips() - before;
    assert_eq!((edges(&reply), chunk_edges(&reply)), ((3, 3), (2, 2)));
    assert!(whole(&base) == written);
    {
        let _rewrite = EnvVarGuard::set(SWITCH, "1");
        let before = round_trips();
        let reply = rebuild_body(&ctx, body.clone());
        assert_eq!((edges(&reply), chunk_edges(&reply)), ((3, 0), (2, 0)));
        assert_eq!(
            round_trips() - before + (2 + 2) + (2 + 2),
            left + (3 + 1) + (2 + 1)
        );
        assert!(whole(&base) == written);
    }

    let gone = deterministic_id::edge_id(BASE, "CALLS", "11", "12", "src/a.rs", "src/b.rs");
    base.with_write_backend(|w| base.drop_edge_be(w, &gone))
        .unwrap();
    assert_eq!(whole(&base).edges.len(), 1);
    let reply = rebuild_body(&ctx, body);
    assert_eq!(chunk_edges(&reply), (2, 1), "{reply}");
    assert!(whole(&base) == written, "the edge that was gone is back");
}

/// The same number under another type is not a change, as it is none for a
/// node: the stored edge keeps the type it was written with. Another number
/// is a change.
#[test]
#[serial_test::serial]
fn the_same_number_under_another_type_is_not_a_change() {
    let ctx = lsm();
    let storage = ctx.collections.create_collection(GRAPH).unwrap();
    let point = || row(1, "a_fn", "b_fn", "src/a.rs", "src/b.rs", 10);
    let edge_id = deterministic_id::edge_id(GRAPH, "CALLS", "a_fn", "b_fn", "src/a.rs", "src/b.rs");
    let stored_line = || {
        let edge = storage
            .with_read_backend(|r| storage.get_edge_be(r, &edge_id))
            .unwrap();
        typed(&edge.properties["start_line"])
    };
    store(&storage, &[point()]);
    assert_eq!(edges(&rebuild(&ctx, &["src/a.rs"], false)), (1, 0));
    assert_eq!(stored_line(), "I64(10)");

    // A row written anew with the line as a float: an upsert over the stored
    // row would leave the row itself as it is, by the same rule.
    remove(&storage, &[1]);
    store(&storage, &[with(point(), "start_line", Value::F64(10.0))]);
    let row_line = storage
        .with_read_backend(|r| storage.get_node_be(r, &1))
        .unwrap();
    assert_eq!(typed(&row_line.properties["start_line"]), "F64(10.0)");
    assert_eq!(edges(&rebuild(&ctx, &["src/a.rs"], false)), (1, 1));
    assert_eq!(stored_line(), "I64(10)");

    store(&storage, &[with(point(), "start_line", Value::I64(11))]);
    assert_eq!(edges(&rebuild(&ctx, &["src/a.rs"], false)), (1, 0));
    assert_eq!(stored_line(), "I64(11)");
}

/// The rows of a pass: [`three_rows_of_a_and_one_of_b`] and one whose id is
/// after every Symbol node's, so that a pass meets rows on both sides of the
/// nodes it writes.
fn rows_of_a_pass() -> Vec<NodeUpsert> {
    let mut rows = three_rows_of_a_and_one_of_b().to_vec();
    rows.push(row(u128::MAX, "z_fn", "a_fn", "src/z.rs", "src/a.rs", 3));
    rows
}

/// A pass over a graph that needs no repair leaves every edge and still
/// walks to its end. A row a call here: each call's cursor is the row it
/// took, in the order a cached scan of the nodes meets the rows (the pass
/// streams them), and no call but the last sets the mark. After it a call
/// is answered by the mark.
#[test]
#[serial_test::serial]
fn a_pass_over_a_graph_that_needs_no_repair_walks_to_its_mark() {
    let ctx = lsm();
    let storage = ctx.collections.create_collection(GRAPH).unwrap();
    store(&storage, &rows_of_a_pass());
    assert_eq!(edges(&forced_pass(&ctx)), (5, 0), "no edge is stored yet");
    let healthy = whole(&storage);
    assert_eq!(healthy.edges.len(), 5);

    let mut rows = Vec::new();
    {
        let r = storage.backend.begin_read().unwrap();
        storage
            .backend
            .scan(&r, Namespace::Nodes, KeyRange::all(), |key, value| {
                let id = u128::from_be_bytes(key.try_into().unwrap());
                let node = SerializedNode::decode_node(value, id).unwrap();
                if node.properties.contains_key("caller_symbol") {
                    rows.push(id);
                }
                true
            })
            .unwrap();
    }
    assert_eq!(rows, [1, 2, 3, 4, u128::MAX]);

    let call = |force: bool| {
        backfill(
            &ctx,
            json!({"collection": GRAPH, "batch_size": 1, "force": force}),
        )
    };
    for (at, row) in rows.iter().enumerate() {
        let reply = call(at == 0);
        assert_eq!(edges(&reply), (1, 1), "call {at}: {reply}");
        assert_eq!(reply["complete"], false, "call {at}: {reply}");
        assert_eq!(reply["cursor"], row.to_string(), "call {at}: {reply}");
        assert_eq!(mark(&storage), None, "call {at}");
    }
    let last = call(false);
    assert_eq!(edges(&last), (0, 0), "{last}");
    assert_eq!(last["complete"], true, "{last}");
    assert_eq!(mark(&storage).as_deref(), Some(&b"1"[..]));

    let marked = call(false);
    assert_eq!(edges(&marked), (0, 0), "{marked}");
    assert_eq!(marked["complete"], true, "{marked}");
    assert!(marked["cursor"].is_null(), "{marked}");
    assert!(whole(&storage) == healthy);
}

/// What a forced pass puts back: each of an equal edge's four keys, an edge
/// that is gone with its keys, and the node row of an equal edge's end. An
/// edge short of a key or gone is written and the other four are left; for
/// the node row every edge is left. After each the graph is whole, and at
/// the end the counters are what a recount finds.
#[test]
#[serial_test::serial]
fn a_forced_pass_puts_back_what_a_graph_lacks() {
    let ctx = lsm();
    let storage = ctx.collections.create_collection(GRAPH).unwrap();
    store(&storage, &rows_of_a_pass());
    forced_pass(&ctx);
    let healthy = whole(&storage);
    let edge = stored_edge_a_to_b(&storage);

    for which in 0..4 {
        storage
            .with_write_backend(|w| take_key(&storage, w, &edge, which))
            .unwrap();
        assert_eq!(
            whole(&storage).keys.len() + 1,
            healthy.keys.len(),
            "key {which}"
        );
        let (reply, counted) = metrics_of(|| forced_pass(&ctx));
        assert_eq!(edges(&reply), (5, 4), "key {which}: {reply}");
        assert!(whole(&storage) == healthy, "key {which} is back");
        assert!(
            counted
                .lines()
                .any(|line| line == ONE_EDGE_WRITTEN_FOR_A_KEY),
            "key {which}: {counted}"
        );
    }

    storage
        .with_write_backend(|w| storage.drop_edge_be(w, &edge.id))
        .unwrap();
    assert_eq!(whole(&storage).edges.len() + 1, healthy.edges.len());
    assert_eq!(edges(&forced_pass(&ctx)), (5, 4), "an edge that is gone");
    assert!(whole(&storage) == healthy, "the edge is back");

    // `c_fn` is called by `a_fn` and is the end of no other edge.
    remove(
        &storage,
        &[deterministic_id::node_id(
            GRAPH, "Symbol", "c_fn", "src/c.rs",
        )],
    );
    assert_eq!(whole(&storage).nodes.len() + 1, healthy.nodes.len());
    assert_eq!(edges(&forced_pass(&ctx)), (5, 5), "a node row that is gone");
    assert!(whole(&storage) == healthy, "the node row is back");

    let recount = storage.recount_lsm_counters().unwrap();
    assert_eq!(
        (recount.nodes.before, recount.edges.before),
        (recount.nodes.after, recount.edges.after),
        "stored node and edge counters against a recount"
    );
}

/// With the switch set, or with the request's `rewrite_unchanged_edges`, a
/// pass writes every edge it derives by the loop every pass ran: a get an
/// edge, five, where the leave makes its two batched reads, a snapshot
/// each. The graph is whole after each.
#[test]
#[serial_test::serial]
fn a_pass_told_to_rewrite_runs_the_loop_that_every_pass_ran() {
    let ctx = lsm();
    let storage = ctx.collections.create_collection(GRAPH).unwrap();
    store(&storage, &rows_of_a_pass());
    forced_pass(&ctx);
    let healthy = whole(&storage);

    let pass = |rewrite_unchanged_edges: bool| {
        let before = round_trips();
        let reply = backfill(
            &ctx,
            json!({
                "collection": GRAPH,
                "force": true,
                "rewrite_unchanged_edges": rewrite_unchanged_edges,
            }),
        );
        assert!(whole(&storage) == healthy, "{reply}");
        (edges(&reply), round_trips() - before)
    };
    // Once unmeasured, so that no measured pass is the first of its kind.
    pass(false);
    let left = pass(false);
    assert_eq!(left.0, (5, 5));
    let asked = pass(true);
    assert_eq!(asked.0, (5, 0), "the request's field");
    let switched = {
        let _rewrite = EnvVarGuard::set(SWITCH, "1");
        pass(false)
    };
    assert_eq!(switched.0, (5, 0), "the switch");
    assert_eq!(asked.1, switched.1);
    assert_eq!(switched.1 + (2 + 2), left.1 + 5, "round trips");
}

/// Two rows that one call of a pass takes and that derive one id: the store
/// ends as the later has it, as the store of the loop does, and a second
/// pass leaves the edge.
#[test]
#[serial_test::serial]
fn a_pass_takes_two_rows_that_derive_one_id_as_the_later() {
    let passes = |rewrite: bool| {
        let _rewrite = rewrite.then(|| EnvVarGuard::set(SWITCH, "1"));
        let ctx = lsm();
        let storage = ctx.collections.create_collection(GRAPH).unwrap();
        let mut rows = rows_of_a_pass();
        let mut twin = with(rows[0].clone(), "confidence", Value::String("low".into()));
        twin.id = 5;
        rows.push(twin);
        store(&storage, &rows);
        let first = edges(&forced_pass(&ctx));
        let second = edges(&forced_pass(&ctx));
        (first, second, whole(&storage))
    };
    let (left, rewritten) = (passes(false), passes(true));
    assert_eq!((left.0, left.1), ((6, 0), (6, 6)));
    assert_eq!((rewritten.0, rewritten.1), ((6, 0), (6, 0)));
    assert!(left.2 == rewritten.2, "the two stores hold the same");
    assert_eq!(left.2.edges.len(), 5);
    assert_eq!(
        left.2
            .edges
            .iter()
            .filter(|edge| edge.contains("confidence=String(\"low\")"))
            .count(),
        1,
        "the twin's row is the edge"
    );
}
