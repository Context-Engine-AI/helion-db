//! Which stored stamps let an upsert leave a point's dense rows and sparse
//! postings as they are (`keeps_its_vectors`): a stamp of today's lists or
//! of the first, made with the vectors and the covered payload that come in.
//! Anything else is a full write.

use super::*;

/// A chunk as the client writes one, with `spaces` dense spaces and one sparse.
/// `line` is where it sits in its file: it moves keys only the second lists
/// leave out.
fn chunk(spaces: usize, line: i64) -> ReplicatedPoint {
    let text = |s: &str| Value::String(s.to_string());
    ReplicatedPoint {
        id: 7,
        vectors: (0..spaces)
            .map(|space| (format!("dense_{space}"), vec![0.25, 0.5 + space as f32]))
            .collect(),
        sparse_vectors: HashMap::from([(
            "lex_sparse".to_string(),
            SparseVector {
                indices: vec![3, 9],
                values: vec![1.0, 0.5],
            },
        )]),
        payload: HashMap::from([
            ("document".to_string(), text(&format!("f, line {line}"))),
            ("path".to_string(), text("src/lib.rs")),
            (
                "metadata".to_string(),
                Value::Object(HashMap::from([
                    ("calls".to_string(), Value::Array(vec![text("g")])),
                    ("symbol_start_line".to_string(), Value::I64(line)),
                ])),
            ),
        ]),
    }
}

/// The node an upsert of `point` left, stamped under `lists`.
fn node(point: &ReplicatedPoint, lists: KeyLists) -> HashMap<String, Value> {
    let stamp = content_hash_of(lists, &point.payload, &point.vectors, &point.sparse_vectors);
    let mut node = point.payload.clone();
    node.insert(CONTENT_HASH_KEY.to_string(), Value::U64(stamp));
    node
}

fn keeps(stored: &HashMap<String, Value>, incoming: &ReplicatedPoint) -> bool {
    keeps_its_vectors(stored, incoming, compute_content_hash(incoming))
}

#[test]
fn a_stamp_of_either_lists_keeps_the_vectors_of_a_moved_chunk() {
    for lists in [KeyLists::First, KeyLists::Second] {
        let stored = node(&chunk(6, 10), lists);
        assert!(keeps(&stored, &chunk(6, 10)), "{lists:?}: as stored");
        assert!(keeps(&stored, &chunk(6, 20)), "{lists:?}: moved");

        let mut another_vector = chunk(6, 20);
        another_vector
            .vectors
            .insert("dense_0".into(), vec![0.5, 0.5]);
        assert!(!keeps(&stored, &another_vector), "{lists:?}: a vector");
        let mut another_term = chunk(6, 20);
        another_term.sparse_vectors.insert(
            "lex_sparse".into(),
            SparseVector {
                indices: vec![3, 10],
                values: vec![1.0, 0.5],
            },
        );
        assert!(!keeps(&stored, &another_term), "{lists:?}: a sparse term");
        let mut another_callee = chunk(6, 20);
        if let Some(Value::Object(metadata)) = another_callee.payload.get_mut("metadata") {
            metadata.insert("calls".into(), Value::Array(Vec::new()));
        }
        assert!(!keeps(&stored, &another_callee), "{lists:?}: a covered key");
    }
}

/// An upsert replaces every vector of its point with those it names. The client
/// stopped writing two of its six dense spaces while points with six were
/// stored: an upsert that names another set of spaces than the stamp was
/// made with is a full write, whichever lists the stamp is of.
#[test]
fn another_number_of_vector_spaces_is_a_full_write() {
    for lists in [KeyLists::First, KeyLists::Second] {
        for (had, comes) in [(6, 4), (4, 6)] {
            let stored = node(&chunk(had, 10), lists);
            for line in [10, 20] {
                assert!(
                    !keeps(&stored, &chunk(comes, line)),
                    "{lists:?}: {had} spaces stored, {comes} come, line {line}"
                );
            }
        }
        let mut no_sparse = chunk(6, 10);
        no_sparse.sparse_vectors.clear();
        assert!(
            !keeps(&node(&chunk(6, 10), lists), &no_sparse),
            "{lists:?}: the sparse space does not come"
        );
        assert!(
            !keeps(&node(&no_sparse, lists), &chunk(6, 10)),
            "{lists:?}: a sparse space comes"
        );
    }
}

/// Only a `U64` is a stamp: an upsert writes one and a `set_payload` stores a
/// non-negative integer as one. A node without a stamp, or with the stamp's
/// number as another kind of value, says nothing of its rows.
#[test]
fn no_stamp_or_one_that_is_not_a_u64_is_a_full_write() {
    let incoming = chunk(6, 10);
    assert!(!keeps(&incoming.payload, &incoming), "no stamp");
    for lists in [KeyLists::First, KeyLists::Second] {
        let mut stored = node(&incoming, lists);
        assert!(keeps(&stored, &incoming), "{lists:?}: the stamp");
        let Some(Value::U64(stamp)) = stored.get(CONTENT_HASH_KEY).cloned() else {
            panic!("no stamp on the node");
        };
        for not_a_stamp in [
            Value::String(stamp.to_string()),
            Value::I64(stamp as i64),
            Value::F64(stamp as f64),
            Value::U128(u128::from(stamp)),
            Value::Boolean(true),
            Value::Empty,
        ] {
            stored.insert(CONTENT_HASH_KEY.to_string(), not_a_stamp.clone());
            assert!(!keeps(&stored, &incoming), "{lists:?}: {not_a_stamp:?}");
        }
        stored.insert(CONTENT_HASH_KEY.to_string(), Value::U64(stamp ^ 1));
        assert!(!keeps(&stored, &incoming), "{lists:?}: another number");
    }
}
