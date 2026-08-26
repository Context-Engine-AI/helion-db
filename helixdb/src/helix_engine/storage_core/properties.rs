//! Zero-copy property storage for HelixDB.
//!
//! Two complementary types:
//!
//! - `PropertiesView<'a>`: zero-copy read of properties directly from LMDB's
//!   memory-mapped pages. No deserialization, no allocation. Binary search for
//!   key lookups. Used on the hot read/query path.
//!
//! - `PropertiesArena`: bump allocator for batching writes. All property strings
//!   and values are packed into a single contiguous buffer. One allocation per
//!   batch instead of one per property. Serializes to the flat format that
//!   PropertiesView reads.
//!
//! ## Flat binary format (what's stored in LMDB)
//!
//! ```text
//! [count: u16 LE]
//! [key_entry × count]:   [key_off: u32 LE][key_len: u16 LE]
//! [val_entry × count]:   [val_off: u32 LE][val_len: u16 LE][val_type: u8]
//! [... key bytes (sorted) ...]
//! [... value bytes ...]
//! ```
//!
//! Keys are stored sorted so PropertiesView can binary-search them.
//! Values use a type tag for reconstruction without full deserialization.

use crate::protocol::value::Value;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Value type tags — one byte per value
// ---------------------------------------------------------------------------

const TAG_STRING: u8 = 1;
const TAG_F32: u8 = 2;
const TAG_F64: u8 = 3;
const TAG_I32: u8 = 4;
const TAG_I64: u8 = 5;
const TAG_U32: u8 = 6;
const TAG_U64: u8 = 7;
const TAG_U128: u8 = 8;
const TAG_BOOL_TRUE: u8 = 9;
const TAG_BOOL_FALSE: u8 = 10;
const TAG_EMPTY: u8 = 11;
const TAG_JSON: u8 = 12; // fallback: serde_json for Array/Object/other

// ---------------------------------------------------------------------------
// PropertiesView — zero-copy reads from LMDB mmap
// ---------------------------------------------------------------------------

/// Zero-copy view over the flat properties format stored in LMDB.
/// Borrows the raw bytes — no allocation, no deserialization.
#[derive(Clone, Copy)]
pub struct PropertiesView<'a> {
    data: &'a [u8],
    count: usize,
}

/// A single key-value entry in the view (still borrows from LMDB).
pub struct PropertyRef<'a> {
    pub key: &'a str,
    pub value: Value,
}

impl<'a> PropertiesView<'a> {
    /// Wrap raw LMDB bytes as a properties view.
    /// Returns None if the data is too short or malformed.
    pub fn from_bytes(data: &'a [u8]) -> Option<Self> {
        if data.len() < 2 {
            return None;
        }
        let count = u16::from_le_bytes([data[0], data[1]]) as usize;
        // Validate minimum size: header + key entries + val entries
        let min_size = 2 + count * 6 + count * 7;
        if data.len() < min_size {
            return None;
        }
        Some(Self { data, count })
    }

    /// Number of properties.
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Look up a property by key. O(log n) binary search.
    pub fn get(&self, key: &str) -> Option<Value> {
        if self.count == 0 {
            return None;
        }

        let mut lo = 0usize;
        let mut hi = self.count;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let k = self.key_at(mid)?;
            match k.cmp(key) {
                std::cmp::Ordering::Equal => return self.value_at(mid),
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
            }
        }
        None
    }

    /// Iterate all properties.
    pub fn iter(&self) -> impl Iterator<Item = PropertyRef<'a>> + 'a {
        let view = *self;
        (0..self.count).filter_map(move |i| {
            let key = view.key_at(i)?;
            let value = view.value_at(i)?;
            Some(PropertyRef { key, value })
        })
    }

    /// Convert to a HashMap (for compatibility with existing code).
    pub fn to_hashmap(&self) -> HashMap<String, Value> {
        self.iter().map(|p| (p.key.to_string(), p.value)).collect()
    }

    // -- internal helpers --

    fn key_entry_offset(&self, i: usize) -> usize {
        2 + i * 6
    }

    fn val_entry_offset(&self, i: usize) -> usize {
        2 + self.count * 6 + i * 7
    }

    fn key_at(&self, i: usize) -> Option<&'a str> {
        let off = self.key_entry_offset(i);
        let key_off = u32::from_le_bytes(self.data[off..off + 4].try_into().ok()?) as usize;
        let key_len = u16::from_le_bytes(self.data[off + 4..off + 6].try_into().ok()?) as usize;
        let key_bytes = self.data.get(key_off..key_off + key_len)?;
        std::str::from_utf8(key_bytes).ok()
    }

    fn value_at(&self, i: usize) -> Option<Value> {
        let off = self.val_entry_offset(i);
        let val_off = u32::from_le_bytes(self.data[off..off + 4].try_into().ok()?) as usize;
        let val_len = u16::from_le_bytes(self.data[off + 4..off + 6].try_into().ok()?) as usize;
        let val_tag = self.data[off + 6];
        let val_bytes = self.data.get(val_off..val_off + val_len)?;
        decode_value(val_tag, val_bytes)
    }
}

fn decode_value(tag: u8, bytes: &[u8]) -> Option<Value> {
    match tag {
        TAG_STRING => std::str::from_utf8(bytes)
            .ok()
            .map(|s| Value::String(s.to_string())),
        TAG_F32 => bytes
            .try_into()
            .ok()
            .map(|b| Value::F32(f32::from_le_bytes(b))),
        TAG_F64 => bytes
            .try_into()
            .ok()
            .map(|b| Value::F64(f64::from_le_bytes(b))),
        TAG_I32 => bytes
            .try_into()
            .ok()
            .map(|b| Value::I32(i32::from_le_bytes(b))),
        TAG_I64 => bytes
            .try_into()
            .ok()
            .map(|b| Value::I64(i64::from_le_bytes(b))),
        TAG_U32 => bytes
            .try_into()
            .ok()
            .map(|b| Value::U32(u32::from_le_bytes(b))),
        TAG_U64 => bytes
            .try_into()
            .ok()
            .map(|b| Value::U64(u64::from_le_bytes(b))),
        TAG_U128 => bytes
            .try_into()
            .ok()
            .map(|b| Value::U128(u128::from_le_bytes(b))),
        TAG_BOOL_TRUE => Some(Value::Boolean(true)),
        TAG_BOOL_FALSE => Some(Value::Boolean(false)),
        TAG_EMPTY => Some(Value::Empty),
        TAG_JSON => serde_json::from_slice(bytes).ok().map(json_to_value),
        _ => None,
    }
}

fn json_to_value(j: serde_json::Value) -> Value {
    match j {
        serde_json::Value::String(s) => Value::String(s),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::I64(i)
            } else if let Some(f) = n.as_f64() {
                Value::F64(f)
            } else {
                Value::Empty
            }
        }
        serde_json::Value::Bool(b) => Value::Boolean(b),
        serde_json::Value::Null => Value::Empty,
        serde_json::Value::Array(arr) => Value::Array(arr.into_iter().map(json_to_value).collect()),
        serde_json::Value::Object(obj) => Value::Object(
            obj.into_iter()
                .map(|(k, v)| (k, json_to_value(v)))
                .collect(),
        ),
    }
}

// ---------------------------------------------------------------------------
// PropertiesArena — bump allocator for batch writes
// ---------------------------------------------------------------------------

/// Simple bump allocator for building properties during write batches.
/// All strings and value bytes are packed into one contiguous buffer.
/// One allocation for the whole batch instead of one per property.
pub struct PropertiesArena {
    buf: Vec<u8>,
    /// Entries: (key_start, key_len, val_start, val_len, val_tag)
    entries: Vec<(u32, u16, u32, u16, u8)>,
}

impl PropertiesArena {
    /// Create a new arena with estimated capacity.
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(4096),
            entries: Vec::with_capacity(32),
        }
    }

    /// Create with a capacity hint (total expected properties across batch).
    pub fn with_capacity(properties_hint: usize) -> Self {
        Self {
            buf: Vec::with_capacity(properties_hint * 64),
            entries: Vec::with_capacity(properties_hint),
        }
    }

    /// Add a key-value pair. Returns the entry index.
    pub fn push(&mut self, key: &str, value: &Value) -> usize {
        let key_start = self.buf.len() as u32;
        self.buf.extend_from_slice(key.as_bytes());
        let key_len = key.len() as u16;

        let val_start = self.buf.len() as u32;
        let val_tag = encode_value(value, &mut self.buf);
        let val_len = (self.buf.len() as u32 - val_start) as u16;

        let idx = self.entries.len();
        self.entries
            .push((key_start, key_len, val_start, val_len, val_tag));
        idx
    }

    /// Add all properties from a HashMap.
    pub fn push_map(&mut self, props: &HashMap<String, Value>) {
        for (k, v) in props {
            self.push(k, v);
        }
    }

    /// Serialize to flat binary format (what gets stored in LMDB).
    /// Keys are sorted for binary search by PropertiesView.
    pub fn serialize(&self) -> Vec<u8> {
        // Sort entries by key
        let mut sorted: Vec<usize> = (0..self.entries.len()).collect();
        sorted.sort_by(|&a, &b| {
            let (ak_start, ak_len, _, _, _) = self.entries[a];
            let (bk_start, bk_len, _, _, _) = self.entries[b];
            let ak = &self.buf[ak_start as usize..(ak_start as usize + ak_len as usize)];
            let bk = &self.buf[bk_start as usize..(bk_start as usize + bk_len as usize)];
            ak.cmp(bk)
        });

        let count = sorted.len();
        let header_size = 2 + count * 6 + count * 7;

        // Calculate total data size
        let mut data_size = 0usize;
        for &i in &sorted {
            let (_, kl, _, vl, _) = self.entries[i];
            data_size += kl as usize + vl as usize;
        }

        let mut out = Vec::with_capacity(header_size + data_size);

        // Write count
        out.extend_from_slice(&(count as u16).to_le_bytes());

        // First pass: compute data offsets
        let data_start = header_size;
        let mut key_offsets = Vec::with_capacity(count);
        let mut val_offsets = Vec::with_capacity(count);
        let mut cursor = data_start;

        for &i in &sorted {
            let (_, kl, _, vl, _) = self.entries[i];
            key_offsets.push((cursor as u32, kl));
            cursor += kl as usize;
            val_offsets.push((cursor as u32, vl));
            cursor += vl as usize;
        }

        // Write key entries
        for &(off, len) in &key_offsets {
            out.extend_from_slice(&off.to_le_bytes());
            out.extend_from_slice(&len.to_le_bytes());
        }

        // Write value entries
        for (idx, &(off, len)) in val_offsets.iter().enumerate() {
            let (_, _, _, _, tag) = self.entries[sorted[idx]];
            out.extend_from_slice(&off.to_le_bytes());
            out.extend_from_slice(&len.to_le_bytes());
            out.push(tag);
        }

        // Write key data then value data (interleaved per entry)
        for &i in &sorted {
            let (ks, kl, _, _, _) = self.entries[i];
            out.extend_from_slice(&self.buf[ks as usize..(ks as usize + kl as usize)]);
            let (_, _, vs, vl, _) = self.entries[i];
            out.extend_from_slice(&self.buf[vs as usize..(vs as usize + vl as usize)]);
        }

        out
    }

    /// Total bytes used in the arena buffer.
    pub fn bytes_used(&self) -> usize {
        self.buf.len()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Reset for reuse (keeps allocated capacity).
    pub fn clear(&mut self) {
        self.buf.clear();
        self.entries.clear();
    }
}

fn encode_value(value: &Value, buf: &mut Vec<u8>) -> u8 {
    match value {
        Value::String(s) => {
            buf.extend_from_slice(s.as_bytes());
            TAG_STRING
        }
        Value::F32(v) => {
            buf.extend_from_slice(&v.to_le_bytes());
            TAG_F32
        }
        Value::F64(v) => {
            buf.extend_from_slice(&v.to_le_bytes());
            TAG_F64
        }
        Value::I32(v) => {
            buf.extend_from_slice(&v.to_le_bytes());
            TAG_I32
        }
        Value::I64(v) => {
            buf.extend_from_slice(&v.to_le_bytes());
            TAG_I64
        }
        Value::U32(v) => {
            buf.extend_from_slice(&v.to_le_bytes());
            TAG_U32
        }
        Value::U64(v) => {
            buf.extend_from_slice(&v.to_le_bytes());
            TAG_U64
        }
        Value::U128(v) => {
            buf.extend_from_slice(&v.to_le_bytes());
            TAG_U128
        }
        Value::Boolean(true) => TAG_BOOL_TRUE,
        Value::Boolean(false) => TAG_BOOL_FALSE,
        Value::Empty => TAG_EMPTY,
        Value::I8(v) => {
            buf.extend_from_slice(&(*v as i32).to_le_bytes());
            TAG_I32
        }
        Value::I16(v) => {
            buf.extend_from_slice(&(*v as i32).to_le_bytes());
            TAG_I32
        }
        Value::U8(v) => {
            buf.extend_from_slice(&(*v as u32).to_le_bytes());
            TAG_U32
        }
        Value::U16(v) => {
            buf.extend_from_slice(&(*v as u32).to_le_bytes());
            TAG_U32
        }
        Value::Array(_) | Value::Object(_) => {
            // Fallback to JSON for complex types
            let json = value_to_json(value);
            let bytes = serde_json::to_vec(&json).unwrap_or_default();
            buf.extend_from_slice(&bytes);
            TAG_JSON
        }
    }
}

fn value_to_json(v: &Value) -> serde_json::Value {
    match v {
        Value::String(s) => serde_json::Value::String(s.clone()),
        Value::F32(n) => serde_json::json!(*n),
        Value::F64(n) => serde_json::json!(*n),
        Value::I8(n) => serde_json::json!(*n),
        Value::I16(n) => serde_json::json!(*n),
        Value::I32(n) => serde_json::json!(*n),
        Value::I64(n) => serde_json::json!(*n),
        Value::U8(n) => serde_json::json!(*n),
        Value::U16(n) => serde_json::json!(*n),
        Value::U32(n) => serde_json::json!(*n),
        Value::U64(n) => serde_json::json!(*n),
        Value::U128(n) => serde_json::json!(n.to_string()),
        Value::Boolean(b) => serde_json::Value::Bool(*b),
        Value::Empty => serde_json::Value::Null,
        Value::Array(arr) => serde_json::Value::Array(arr.iter().map(value_to_json).collect()),
        Value::Object(obj) => serde_json::Value::Object(
            obj.iter()
                .map(|(k, v)| (k.clone(), value_to_json(v)))
                .collect(),
        ),
    }
}

// ---------------------------------------------------------------------------
// Convenience: serialize a HashMap directly (for migration/compat)
// ---------------------------------------------------------------------------

/// Serialize a HashMap<String, Value> to the flat binary format.
pub fn serialize_properties(props: &HashMap<String, Value>) -> Vec<u8> {
    if props.is_empty() {
        return vec![0, 0]; // count = 0
    }
    let mut arena = PropertiesArena::with_capacity(props.len());
    arena.push_map(props);
    arena.serialize()
}

/// Deserialize flat binary format back to HashMap (for compat).
pub fn deserialize_properties(data: &[u8]) -> HashMap<String, Value> {
    PropertiesView::from_bytes(data)
        .map(|v| v.to_hashmap())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_simple_properties() {
        let mut props = HashMap::new();
        props.insert("name".to_string(), Value::String("foo".to_string()));
        props.insert("count".to_string(), Value::I64(42));
        props.insert("active".to_string(), Value::Boolean(true));

        let bytes = serialize_properties(&props);
        let view = PropertiesView::from_bytes(&bytes).unwrap();

        assert_eq!(view.len(), 3);
        assert_eq!(view.get("name"), Some(Value::String("foo".to_string())));
        assert_eq!(view.get("count"), Some(Value::I64(42)));
        assert_eq!(view.get("active"), Some(Value::Boolean(true)));
        assert_eq!(view.get("missing"), None);
    }

    #[test]
    fn roundtrip_numeric_types() {
        let mut props = HashMap::new();
        props.insert("f32".to_string(), Value::F32(3.14));
        props.insert("f64".to_string(), Value::F64(2.718281828));
        props.insert("u32".to_string(), Value::U32(999));
        props.insert("u64".to_string(), Value::U64(u64::MAX));
        props.insert("u128".to_string(), Value::U128(u128::MAX));
        props.insert("i32".to_string(), Value::I32(-42));

        let bytes = serialize_properties(&props);
        let view = PropertiesView::from_bytes(&bytes).unwrap();

        assert_eq!(view.get("f64"), Some(Value::F64(2.718281828)));
        assert_eq!(view.get("u32"), Some(Value::U32(999)));
        assert_eq!(view.get("u64"), Some(Value::U64(u64::MAX)));
        assert_eq!(view.get("u128"), Some(Value::U128(u128::MAX)));
        assert_eq!(view.get("i32"), Some(Value::I32(-42)));
        // f32 loses precision so check approximately
        if let Some(Value::F32(v)) = view.get("f32") {
            assert!((v - 3.14).abs() < 0.001);
        } else {
            panic!("f32 not found");
        }
    }

    #[test]
    fn roundtrip_complex_json_fallback() {
        let mut inner = HashMap::new();
        inner.insert("nested".to_string(), Value::I64(1));

        let mut props = HashMap::new();
        props.insert("obj".to_string(), Value::Object(inner));
        props.insert(
            "arr".to_string(),
            Value::Array(vec![Value::I64(1), Value::I64(2)]),
        );

        let bytes = serialize_properties(&props);
        let view = PropertiesView::from_bytes(&bytes).unwrap();

        assert_eq!(view.len(), 2);
        // Complex types round-trip through JSON
        assert!(view.get("arr").is_some());
        assert!(view.get("obj").is_some());
    }

    #[test]
    fn empty_properties() {
        let props = HashMap::new();
        let bytes = serialize_properties(&props);
        assert_eq!(bytes, vec![0, 0]);

        let view = PropertiesView::from_bytes(&bytes).unwrap();
        assert_eq!(view.len(), 0);
        assert!(view.is_empty());
        assert_eq!(view.get("anything"), None);
    }

    #[test]
    fn arena_reuse() {
        let mut arena = PropertiesArena::new();

        // First batch
        arena.push("a", &Value::I64(1));
        arena.push("b", &Value::I64(2));
        let bytes1 = arena.serialize();

        // Reset and reuse
        arena.clear();
        arena.push("x", &Value::String("hello".into()));
        let bytes2 = arena.serialize();

        let v1 = PropertiesView::from_bytes(&bytes1).unwrap();
        let v2 = PropertiesView::from_bytes(&bytes2).unwrap();

        assert_eq!(v1.len(), 2);
        assert_eq!(v1.get("a"), Some(Value::I64(1)));
        assert_eq!(v2.len(), 1);
        assert_eq!(v2.get("x"), Some(Value::String("hello".into())));
    }

    #[test]
    fn keys_are_sorted_for_binary_search() {
        let mut arena = PropertiesArena::new();
        // Insert in reverse order
        arena.push("zebra", &Value::I64(3));
        arena.push("apple", &Value::I64(1));
        arena.push("mango", &Value::I64(2));

        let bytes = arena.serialize();
        let view = PropertiesView::from_bytes(&bytes).unwrap();

        // All keys should be findable via binary search
        assert_eq!(view.get("apple"), Some(Value::I64(1)));
        assert_eq!(view.get("mango"), Some(Value::I64(2)));
        assert_eq!(view.get("zebra"), Some(Value::I64(3)));

        // Verify iteration is sorted
        let keys: Vec<&str> = view.iter().map(|p| p.key).collect();
        assert_eq!(keys, vec!["apple", "mango", "zebra"]);
    }

    #[test]
    fn deserialize_properties_compat() {
        let mut props = HashMap::new();
        props.insert("key".to_string(), Value::String("val".to_string()));

        let bytes = serialize_properties(&props);
        let roundtrip = deserialize_properties(&bytes);

        assert_eq!(roundtrip.len(), 1);
        assert_eq!(
            roundtrip.get("key"),
            Some(&Value::String("val".to_string()))
        );
    }

    #[test]
    fn malformed_data_returns_none() {
        assert!(PropertiesView::from_bytes(&[]).is_none());
        assert!(PropertiesView::from_bytes(&[0]).is_none());
        // Count says 1 but no entries follow
        assert!(PropertiesView::from_bytes(&[1, 0]).is_none());
    }

    #[test]
    fn large_batch_arena_performance() {
        let mut arena = PropertiesArena::with_capacity(1000);

        // Simulate a 1000-property batch (e.g., bulk ingest)
        for i in 0..1000 {
            arena.push(
                &format!("prop_{:04}", i),
                &Value::String(format!("value_{}", i)),
            );
        }

        let bytes = arena.serialize();
        let view = PropertiesView::from_bytes(&bytes).unwrap();

        assert_eq!(view.len(), 1000);
        assert_eq!(
            view.get("prop_0500"),
            Some(Value::String("value_500".into()))
        );
        assert_eq!(view.get("prop_9999"), None);

        // Arena reuse
        assert!(arena.bytes_used() > 10000);
        arena.clear();
        assert_eq!(arena.len(), 0);
    }
}
