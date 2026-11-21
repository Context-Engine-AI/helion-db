//! What a dense row keeps of the payload it was written with (the payload,
//! or its digest under `HELIX_DENSE_ROW_PAYLOAD=0`) and how a merge that
//! holds two rows of one point tells them apart.

use super::*;

fn payload(write: i64) -> HashMap<String, Value> {
    HashMap::from([
        ("write".to_string(), Value::I64(write)),
        ("ratio".to_string(), Value::F64(0.5)),
        (
            "list".to_string(),
            Value::Array(vec![
                Value::I64(1),
                Value::F64(2.5),
                Value::String("a".into()),
            ]),
        ),
        (
            "metadata".to_string(),
            Value::Object(HashMap::from([
                ("kind".to_string(), Value::String("function".into())),
                ("ingested_at".to_string(), Value::I64(write)),
            ])),
        ),
    ])
}

/// The point's payload as its node has it: the content hash stamped on, and
/// the keys a payload-only update may have rewritten since.
fn node_payload(write: i64) -> HashMap<String, Value> {
    let mut stored = payload(write);
    stored.insert(CONTENT_HASH_KEY.to_string(), Value::U64(7));
    stored.insert("_pending_prune".to_string(), Value::Boolean(true));
    if let Some(Value::Object(metadata)) = stored.get_mut("metadata") {
        metadata.insert("ingested_at".to_string(), Value::I64(99));
    }
    stored
}

fn said(fields: &HashMap<String, Value>, node: &HashMap<String, Value>) -> WrittenWith {
    row_written_with(
        fields,
        node,
        &payload_without_content_hash_payload_only_keys(node),
        &mut None,
    )
}

/// The rule of every image before the digest: a row is the point's when the
/// payload it keeps equals the point's, and never "says nothing".
fn older_image_said(fields: &HashMap<String, Value>, node: &HashMap<String, Value>) -> WrittenWith {
    if payload_without_content_hash_payload_only_keys(fields)
        == payload_without_content_hash_payload_only_keys(node)
    {
        WrittenWith::ThePointsPayload
    } else {
        WrittenWith::AnotherPayload
    }
}

#[test]
fn a_row_is_told_by_the_payload_it_keeps_or_by_its_digest() {
    let node = node_payload(2);
    for (what, row, expected) in [
        (
            "the point's payload",
            payload(2),
            WrittenWith::ThePointsPayload,
        ),
        ("another payload", payload(1), WrittenWith::AnotherPayload),
        (
            "the digest of the point's payload",
            row_digest_fields(&payload(2)),
            WrittenWith::ThePointsPayload,
        ),
        (
            "the digest of another payload",
            row_digest_fields(&payload(1)),
            WrittenWith::AnotherPayload,
        ),
        ("nothing", HashMap::new(), WrittenWith::Unknown),
    ] {
        assert_eq!(said(&row, &node), expected, "a row that keeps {what}");
    }
    // A digest is that of the keys the content hash covers, as the node has them.
    let digest = |payload: &HashMap<String, Value>| row_payload_digest(KeyLists::CURRENT, payload);
    assert_eq!(digest(&payload(2)), digest(&node));
    assert_ne!(digest(&payload(2)), digest(&payload(1)));
}

#[test]
fn only_a_u64_under_the_reserved_key_alone_is_a_digest() {
    let key = KeyLists::CURRENT.row_digest_key().to_string();
    let digest = HashMap::from([(key.clone(), Value::U64(5))]);
    assert_eq!(row_keeps(&digest), RowKeeps::Digest(KeyLists::CURRENT, 5));
    assert_eq!(row_keeps(&HashMap::new()), RowKeeps::Nothing);
    // What the gateway makes of a client's number under that key, and the
    // key beside another.
    for value in [Value::I64(5), Value::F64(5.0), Value::String("5".into())] {
        let copy = HashMap::from([(key.clone(), value)]);
        assert_eq!(row_keeps(&copy), RowKeeps::Payload(&copy));
        // Such a payload is compared as the payload it is.
        assert_eq!(said(&copy, &copy), WrittenWith::ThePointsPayload);
        assert_eq!(said(&copy, &node_payload(2)), WrittenWith::AnotherPayload);
    }
    let beside = HashMap::from([(key, Value::U64(5)), ("write".to_string(), Value::I64(1))]);
    assert_eq!(row_keeps(&beside), RowKeeps::Payload(&beside));
}

/// A digest made by another rule is kept under another key of the family.
/// This image cannot check one: the row says nothing, so its pair stays
/// undecided.
#[test]
fn a_digest_under_another_key_of_the_family_says_nothing() {
    let later_key = format!("{ROW_PAYLOAD_DIGEST_FAMILY}_3__");
    let later = HashMap::from([(
        later_key.clone(),
        Value::U64(row_payload_digest(KeyLists::CURRENT, &payload(2))),
    )]);
    assert_eq!(row_keeps(&later), RowKeeps::DigestOfAnotherRule);
    let node = node_payload(2);
    let now = |row: &HashMap<String, Value>| said(row, &node);
    assert_eq!(now(&later), WrittenWith::Unknown);
    assert_eq!(
        copy_to_keep(now(&later), &[now(&payload(2))]),
        KeptCopy::Undecided
    );
    assert_eq!(
        copy_to_keep(now(&payload(1)), &[now(&later)]),
        KeptCopy::Undecided
    );
    // A client's number under such a key is a payload, as under this image's.
    let clients = HashMap::from([(later_key, Value::I64(5))]);
    assert_eq!(row_keeps(&clients), RowKeeps::Payload(&clients));
}

/// A stored digest is frozen. This pins what a payload's digest is, with
/// every kind of value an upsert stores and keys from the lists the content
/// hash leaves out, and pins the lists and the key a digest of each is kept
/// under. If it fails, rows digested so far no longer match their points:
/// keep the digest of the lists there are as it was, and give a new rule a
/// new member of `KeyLists` with a key that begins with
/// `ROW_PAYLOAD_DIGEST_FAMILY`, as the second lists were added. Do not change
/// a literal or a list.
#[test]
fn the_digest_of_a_payload_does_not_change_once_rows_keep_it() {
    let text = |s: &str| Value::String(s.to_string());
    let pinned = HashMap::from([
        ("path".to_string(), text("src/lib.rs")),
        ("lines".to_string(), Value::I64(42)),
        ("neg".to_string(), Value::I64(-7)),
        ("ratio".to_string(), Value::F64(0.5)),
        // A JSON integer past `i64`, and a JSON null, as the gateway stores them.
        ("big".to_string(), Value::F64(u64::MAX as f64)),
        ("none".to_string(), text("null")),
        ("live".to_string(), Value::Boolean(true)),
        (
            "list".to_string(),
            Value::Array(vec![
                Value::I64(1),
                Value::F64(2.5),
                text("a"),
                text("null"),
            ]),
        ),
        (
            "metadata".to_string(),
            Value::Object(HashMap::from([
                ("kind".to_string(), text("function")),
                (
                    "symbol".to_string(),
                    Value::Object(HashMap::from([
                        ("name".to_string(), text("f")),
                        ("depth".to_string(), Value::I64(2)),
                    ])),
                ),
                ("ingested_at".to_string(), Value::I64(1)),
                ("git_branches".to_string(), Value::Array(vec![text("main")])),
                ("start_line".to_string(), Value::I64(10)),
            ])),
        ),
        ("_pending_prune".to_string(), Value::Boolean(true)),
        ("caller_point_id".to_string(), text("x")),
        ("start_line".to_string(), Value::I64(3)),
        (CONTENT_HASH_KEY.to_string(), Value::U64(7)),
    ]);
    assert_eq!(
        row_payload_digest(KeyLists::First, &pinned),
        0x5FF5_38C3_1354_98E4,
        "the digest of a payload changed: {:#018X}",
        row_payload_digest(KeyLists::First, &pinned)
    );
    assert_eq!(
        KeyLists::First.row_digest_key(),
        "__helix_row_payload_digest__"
    );
    // The second lists leave out more and hash alike. A chunk the client moved, with
    // every key only they leave out: its digest is that of the payload
    // without those keys, and the first lists still cover every one.
    let mut moved = pinned.clone();
    let line = text("f in src/lib.rs, lines 10 to 12");
    moved.insert("document".to_string(), line.clone());
    moved.insert("information".to_string(), line);
    if let Some(Value::Object(metadata)) = moved.get_mut("metadata") {
        metadata.extend([
            ("symbol_start_line".to_string(), Value::I64(10)),
            ("symbol_end_line".to_string(), Value::I64(12)),
            (
                "import_map".to_string(),
                Value::Object(HashMap::from([("a".to_string(), text("crate::a"))])),
            ),
            ("inheritance_map".to_string(), Value::Object(HashMap::new())),
            ("_graph_repair_backfilled".to_string(), Value::Boolean(true)),
            (
                "_graph_repair_backfilled_version".to_string(),
                Value::I64(3),
            ),
        ]);
    }
    for payload in [&pinned, &moved] {
        assert_eq!(
            row_payload_digest(KeyLists::Second, payload),
            0x5FF5_38C3_1354_98E4,
            "the digest of a payload changed: {:#018X}",
            row_payload_digest(KeyLists::Second, payload)
        );
    }
    assert_eq!(
        row_payload_digest(KeyLists::First, &moved),
        0x8ADF_072A_42DC_8B24,
        "the first lists' digest of a payload changed: {:#018X}",
        row_payload_digest(KeyLists::First, &moved)
    );
    assert_eq!(
        row_digest_fields(&moved),
        HashMap::from([(
            "__helix_row_payload_digest_2__".to_string(),
            Value::U64(0x5FF5_38C3_1354_98E4)
        )])
    );
    assert_eq!(CONTENT_HASH_KEY, "__helix_content_hash__");
    assert_eq!(CONTENT_HASH_METADATA_KEY, "metadata");
    assert_eq!(
        CONTENT_HASH_PAYLOAD_ONLY_KEYS.to_vec(),
        vec![
            "caller_point_id",
            "_pending_prune",
            "_pending_prune_at",
            "start_line",
            "end_line",
            "start_col",
            "end_col",
        ],
        "a key the digest covers or leaves out changed"
    );
    assert_eq!(
        CONTENT_HASH_METADATA_PAYLOAD_ONLY_SUBKEYS.to_vec(),
        vec![
            "ingested_at",
            "churn_count",
            "author_count",
            "_graph_backfilled",
            "_graph_backfilled_version",
            "_pseudo_backfilled",
            "host_path",
            "container_path",
            "file_hash",
            "file_size_bytes",
            "last_modified_at",
            "indexed_branch",
            "git_branches",
            "start_line",
            "end_line",
            "start_col",
            "end_col",
        ],
        "a metadata key the digest covers or leaves out changed"
    );
    assert_eq!(
        CONTENT_HASH_MOVED_CHUNK_KEYS.to_vec(),
        vec!["document", "information"],
        "a key the second lists leave out changed"
    );
    assert_eq!(
        CONTENT_HASH_METADATA_MOVED_CHUNK_SUBKEYS.to_vec(),
        vec![
            "symbol_start_line",
            "symbol_end_line",
            "import_map",
            "inheritance_map",
            "_graph_repair_backfilled",
            "_graph_repair_backfilled_version",
        ],
        "a metadata key the second lists leave out changed"
    );
}

/// A row digested under the first lists, by an image that had only those. A
/// node write can since have changed a key they cover and today's do not,
/// with no row written: a first digest that is not the node's says nothing,
/// where one of today's lists says "another payload". A payload copy is
/// compared on the keys today's lists cover and still tells.
#[test]
fn a_digest_of_the_first_lists_is_the_points_or_says_nothing() {
    let line = |n: i64| Value::String(format!("f, line {n}"));
    let written = {
        let mut written = payload(2);
        written.insert("document".to_string(), line(10));
        written
    };
    let first = |payload: &HashMap<String, Value>| {
        HashMap::from([(
            KeyLists::First.row_digest_key().to_string(),
            Value::U64(row_payload_digest(KeyLists::First, payload)),
        )])
    };
    assert_eq!(
        row_keeps(&first(&written)),
        RowKeeps::Digest(
            KeyLists::First,
            row_payload_digest(KeyLists::First, &written)
        )
    );
    let mut node = written.clone();
    node.insert(CONTENT_HASH_KEY.to_string(), Value::U64(7));
    let another = payload(1);
    for (row, expected) in [
        (first(&written), WrittenWith::ThePointsPayload),
        (first(&another), WrittenWith::Unknown),
        (row_digest_fields(&written), WrittenWith::ThePointsPayload),
        (row_digest_fields(&another), WrittenWith::AnotherPayload),
    ] {
        assert_eq!(said(&row, &node), expected, "as written: {row:?}");
    }
    // The chunk moved: a node write, and no row.
    node.insert("document".to_string(), line(20));
    for (row, expected) in [
        (first(&written), WrittenWith::Unknown),
        (row_digest_fields(&written), WrittenWith::ThePointsPayload),
        (written.clone(), WrittenWith::ThePointsPayload),
        (another, WrittenWith::AnotherPayload),
    ] {
        assert_eq!(said(&row, &node), expected, "moved since: {row:?}");
    }
    // A merge on this image leaves a first digest as it found it.
    let _guard = ENV_LOCK.lock().unwrap();
    let _digest = EnvVarGuard::set("HELIX_DENSE_ROW_PAYLOAD", "0");
    let mut rows: Vec<ExportedRow> = vec![(1, vec![1.0], first(&written))];
    digest_in_place_of_a_payload_copy(&mut rows);
    assert_eq!(rows, vec![(1, vec![1.0], first(&written))]);
}

#[test]
fn a_merge_keeps_the_one_copy_written_with_the_points_payload() {
    use KeptCopy::{Other, Search, Undecided};
    use WrittenWith::{AnotherPayload as Another, ThePointsPayload as Points, Unknown};
    for (served, others, kept) in [
        (Points, vec![Another], Search),
        (Another, vec![Points], Other(0)),
        (Another, vec![Another, Points], Other(1)),
        // Both, neither or two others: the rows do not tell.
        (Points, vec![Points], Undecided),
        (Another, vec![Another], Undecided),
        (Another, vec![Points, Points], Undecided),
        // A row that keeps nothing is not "another payload".
        (Unknown, vec![Points], Undecided),
        (Points, vec![Unknown], Undecided),
        (Another, vec![Points, Unknown], Undecided),
    ] {
        assert_eq!(copy_to_keep(served, &others), kept, "{served:?} {others:?}");
    }
}

/// Why the digest is not written by default yet. A pair whose rows were both
/// written with the point's payload, one rewritten since with its digest: an
/// image without this rule reads the digest as another payload and keeps the
/// full row whichever of the two search serves. This image leaves the one
/// search serves, as that image does while both rows are full.
#[test]
fn an_older_image_keeps_the_full_row_of_a_pair_when_the_other_has_a_digest() {
    let node = node_payload(2);
    let (full, digest) = (payload(2), row_digest_fields(&payload(2)));
    let older = |row: &HashMap<String, Value>| older_image_said(row, &node);
    assert_eq!(
        copy_to_keep(older(&digest), &[older(&full)]),
        KeptCopy::Other(0)
    );
    assert_eq!(
        copy_to_keep(older(&full), &[older(&digest)]),
        KeptCopy::Search
    );
    assert_eq!(
        copy_to_keep(older(&full), &[older(&full)]),
        KeptCopy::Undecided
    );
    let now = |row: &HashMap<String, Value>| said(row, &node);
    assert_eq!(
        copy_to_keep(now(&digest), &[now(&full)]),
        KeptCopy::Undecided
    );
    assert_eq!(
        copy_to_keep(now(&full), &[now(&digest)]),
        KeptCopy::Undecided
    );
    // Where the row with the digest was written with another payload, both
    // images keep the full row, and they are right.
    let stale = row_digest_fields(&payload(1));
    assert_eq!(
        copy_to_keep(older(&stale), &[older(&full)]),
        KeptCopy::Other(0)
    );
    assert_eq!(copy_to_keep(now(&stale), &[now(&full)]), KeptCopy::Other(0));
}

#[test]
fn the_switch_decides_what_an_upsert_and_a_rewrite_write() {
    let _guard = ENV_LOCK.lock().unwrap();
    // A digest of a rule this image does not have: never digested again.
    let later = HashMap::from([(format!("{ROW_PAYLOAD_DIGEST_FAMILY}_3__"), Value::U64(5))]);
    let rows = || -> Vec<ExportedRow> {
        vec![
            (1, vec![1.0], payload(1)),
            (2, vec![1.0], row_digest_fields(&payload(2))),
            (3, vec![1.0], HashMap::new()),
            (4, vec![1.0], later.clone()),
        ]
    };
    {
        // The copy is the default, and a rewrite carries what a row has.
        let _default = EnvVarGuard::unset("HELIX_DENSE_ROW_PAYLOAD");
        assert_eq!(dense_row_fields(&payload(1)), payload(1));
        let mut carried = rows();
        digest_in_place_of_a_payload_copy(&mut carried);
        assert_eq!(carried, rows());
    }
    let _digest = EnvVarGuard::set("HELIX_DENSE_ROW_PAYLOAD", "0");
    assert_eq!(
        dense_row_fields(&payload(1)),
        row_digest_fields(&payload(1))
    );
    let mut rewritten = rows();
    digest_in_place_of_a_payload_copy(&mut rewritten);
    assert_eq!(
        rewritten,
        vec![
            (1, vec![1.0], row_digest_fields(&payload(1))),
            (2, vec![1.0], row_digest_fields(&payload(2))),
            (3, vec![1.0], HashMap::new()),
            (4, vec![1.0], later.clone()),
        ]
    );
}
