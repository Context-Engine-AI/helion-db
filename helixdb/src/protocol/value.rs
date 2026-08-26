use crate::helix_engine::types::GraphError;
use serde::{
    de::{DeserializeSeed, VariantAccess, Visitor},
    Deserializer, Serializer,
};
use serde_json::Value as JsonValue;
use sonic_rs::{Deserialize, Serialize};
use std::{cmp::Ordering, collections::HashMap, fmt};

/// A flexible value type that can represent various property values in nodes and edges.
/// Handles both JSON and binary serialisation formats via custom implementaions of the Serialize and Deserialize traits.
#[derive(Clone, Debug)]
pub enum Value {
    String(String),
    F32(f32),
    F64(f64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
    Boolean(bool),
    Array(Vec<Value>),
    Object(HashMap<String, Value>),
    Empty,
}

impl Value {
    fn variant_order(&self) -> u8 {
        match self {
            Value::String(_) => 0,
            Value::F32(_) => 1,
            Value::F64(_) => 2,
            Value::I8(_) => 3,
            Value::I16(_) => 4,
            Value::I32(_) => 5,
            Value::I64(_) => 6,
            Value::U8(_) => 7,
            Value::U16(_) => 8,
            Value::U32(_) => 9,
            Value::U64(_) => 10,
            Value::U128(_) => 11,
            Value::Boolean(_) => 12,
            Value::Array(_) => 13,
            Value::Object(_) => 14,
            Value::Empty => 15,
        }
    }

    fn integer_to_i128(&self) -> Option<i128> {
        match self {
            Value::I8(v) => Some(*v as i128),
            Value::I16(v) => Some(*v as i128),
            Value::I32(v) => Some(*v as i128),
            Value::I64(v) => Some(*v as i128),
            Value::U8(v) => Some(*v as i128),
            Value::U16(v) => Some(*v as i128),
            Value::U32(v) => Some(*v as i128),
            Value::U64(v) => Some(*v as i128),
            Value::U128(v) if *v <= i128::MAX as u128 => Some(*v as i128),
            Value::U128(_) => None,
            _ => None,
        }
    }

    fn is_integer(&self) -> bool {
        matches!(
            self,
            Value::I8(_)
                | Value::I16(_)
                | Value::I32(_)
                | Value::I64(_)
                | Value::U8(_)
                | Value::U16(_)
                | Value::U32(_)
                | Value::U64(_)
                | Value::U128(_)
        )
    }

    fn numeric_to_f64(&self) -> Option<f64> {
        match self {
            Value::I8(v) => Some(*v as f64),
            Value::I16(v) => Some(*v as f64),
            Value::I32(v) => Some(*v as f64),
            Value::I64(v) => Some(*v as f64),
            Value::U8(v) => Some(*v as f64),
            Value::U16(v) => Some(*v as f64),
            Value::U32(v) => Some(*v as f64),
            Value::U64(v) => Some(*v as f64),
            Value::U128(v) => Some(*v as f64),
            Value::F32(v) => Some(*v as f64),
            Value::F64(v) => Some(*v),
            _ => None,
        }
    }

    fn is_numeric(&self) -> bool {
        self.numeric_to_f64().is_some()
    }

    fn cmp_f64(left: f64, right: f64) -> Ordering {
        match left.partial_cmp(&right) {
            Some(ordering) => ordering,
            None if left.is_nan() && right.is_nan() => Ordering::Equal,
            None if left.is_nan() => Ordering::Greater,
            None => Ordering::Less,
        }
    }

    fn cmp_numeric(&self, other: &Self) -> Option<Ordering> {
        if self.is_integer() && other.is_integer() {
            return Some(match (self.integer_to_i128(), other.integer_to_i128()) {
                (Some(left), Some(right)) => left.cmp(&right),
                (None, Some(_)) => Ordering::Greater,
                (Some(_), None) => Ordering::Less,
                (None, None) => match (self, other) {
                    (Value::U128(left), Value::U128(right)) => left.cmp(right),
                    _ => unreachable!(),
                },
            });
        }

        match (self.numeric_to_f64(), other.numeric_to_f64()) {
            (Some(left), Some(right)) => Some(Self::cmp_f64(left, right)),
            _ => None,
        }
    }
}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Value::String(left), Value::String(right)) => left.cmp(right),
            (Value::F32(left), Value::F32(right)) => Self::cmp_f64(*left as f64, *right as f64),
            (Value::F64(left), Value::F64(right)) => Self::cmp_f64(*left, *right),
            (Value::Boolean(left), Value::Boolean(right)) => left.cmp(right),
            (Value::Array(left), Value::Array(right)) => left.cmp(right),
            (Value::Object(left), Value::Object(right)) => cmp_value_maps(left, right),
            (Value::Empty, Value::Empty) => Ordering::Equal,
            (Value::Empty, _) => Ordering::Less,
            (_, Value::Empty) => Ordering::Greater,
            _ => self
                .cmp_numeric(other)
                .unwrap_or_else(|| self.variant_order().cmp(&other.variant_order())),
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Value {}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::String(left), Value::String(right)) => left == right,
            (Value::Boolean(left), Value::Boolean(right)) => left == right,
            (Value::Array(left), Value::Array(right)) => left == right,
            (Value::Object(left), Value::Object(right)) => left == right,
            (Value::Empty, Value::Empty) => true,
            (Value::Empty, _) | (_, Value::Empty) => false,
            (left, right) if left.is_numeric() && right.is_numeric() => {
                left.cmp_numeric(right) == Some(Ordering::Equal)
            }
            _ => false,
        }
    }
}

fn cmp_value_maps(left: &HashMap<String, Value>, right: &HashMap<String, Value>) -> Ordering {
    let mut left_entries = left.iter().collect::<Vec<_>>();
    let mut right_entries = right.iter().collect::<Vec<_>>();
    left_entries.sort_unstable_by(|(left_key, _), (right_key, _)| left_key.cmp(right_key));
    right_entries.sort_unstable_by(|(left_key, _), (right_key, _)| left_key.cmp(right_key));

    for ((left_key, left_value), (right_key, right_value)) in
        left_entries.into_iter().zip(right_entries)
    {
        let ordering = left_key
            .cmp(right_key)
            .then_with(|| left_value.cmp(right_value));
        if ordering != Ordering::Equal {
            return ordering;
        }
    }

    left.len().cmp(&right.len())
}

impl PartialEq<i32> for Value {
    fn eq(&self, other: &i32) -> bool {
        match self {
            Value::I32(i) => i == other,
            _ => false,
        }
    }
}
impl PartialEq<i64> for Value {
    fn eq(&self, other: &i64) -> bool {
        match self {
            Value::I64(i) => i == other,
            _ => false,
        }
    }
}

impl PartialEq<f64> for Value {
    fn eq(&self, other: &f64) -> bool {
        match self {
            Value::F64(f) => f == other,
            _ => false,
        }
    }
}

impl PartialEq<String> for Value {
    fn eq(&self, other: &String) -> bool {
        match self {
            Value::String(s) => s == other,
            _ => false,
        }
    }
}

impl PartialOrd<i64> for Value {
    fn partial_cmp(&self, other: &i64) -> Option<Ordering> {
        match self {
            Value::I64(i) => i.partial_cmp(other),
            _ => None,
        }
    }
}

impl PartialOrd<i32> for Value {
    fn partial_cmp(&self, other: &i32) -> Option<Ordering> {
        match self {
            Value::I32(i) => i.partial_cmp(other),
            _ => None,
        }
    }
}
impl PartialOrd<f64> for Value {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        match self {
            Value::F64(f) => f.partial_cmp(other),
            _ => None,
        }
    }
}

/// Custom serialisation implementation for Value that removes enum variant names in JSON
/// whilst preserving them for binary formats like bincode.
impl Serialize for Value {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            match self {
                Value::String(s) => s.serialize(serializer),
                Value::F32(f) => f.serialize(serializer),
                Value::F64(f) => f.serialize(serializer),
                Value::I8(i) => i.serialize(serializer),
                Value::I16(i) => i.serialize(serializer),
                Value::I32(i) => i.serialize(serializer),
                Value::I64(i) => i.serialize(serializer),
                Value::U8(i) => i.serialize(serializer),
                Value::U16(i) => i.serialize(serializer),
                Value::U32(i) => i.serialize(serializer),
                Value::U64(i) => i.serialize(serializer),
                Value::U128(i) => i.serialize(serializer),
                Value::Boolean(b) => b.serialize(serializer),
                Value::Array(arr) => {
                    use serde::ser::SerializeSeq;
                    let mut seq = serializer.serialize_seq(Some(arr.len()))?;
                    for value in arr {
                        seq.serialize_element(&value)?;
                    }
                    seq.end()
                }
                Value::Object(obj) => {
                    use serde::ser::SerializeMap;
                    let mut map = serializer.serialize_map(Some(obj.len()))?;
                    for (k, v) in obj {
                        map.serialize_entry(k, v)?;
                    }
                    map.end()
                }
                Value::Empty => serializer.serialize_none(),
            }
        } else {
            match self {
                Value::String(s) => serializer.serialize_newtype_variant("Value", 0, "String", s),
                Value::F32(f) => serializer.serialize_newtype_variant("Value", 1, "F32", f),
                Value::F64(f) => serializer.serialize_newtype_variant("Value", 2, "F64", f),
                Value::I8(i) => serializer.serialize_newtype_variant("Value", 3, "I8", i),
                Value::I16(i) => serializer.serialize_newtype_variant("Value", 4, "I16", i),
                Value::I32(i) => serializer.serialize_newtype_variant("Value", 5, "I32", i),
                Value::I64(i) => serializer.serialize_newtype_variant("Value", 6, "I64", i),
                Value::U8(i) => serializer.serialize_newtype_variant("Value", 7, "U8", i),
                Value::U16(i) => serializer.serialize_newtype_variant("Value", 8, "U16", i),
                Value::U32(i) => serializer.serialize_newtype_variant("Value", 9, "U32", i),
                Value::U64(i) => serializer.serialize_newtype_variant("Value", 10, "U64", i),
                Value::U128(i) => serializer.serialize_newtype_variant("Value", 11, "U128", i),
                Value::Boolean(b) => {
                    serializer.serialize_newtype_variant("Value", 12, "Boolean", b)
                }
                Value::Array(a) => serializer.serialize_newtype_variant("Value", 13, "Array", a),
                Value::Object(obj) => {
                    serializer.serialize_newtype_variant("Value", 14, "Object", obj)
                }
                Value::Empty => serializer.serialize_unit_variant("Value", 15, "Empty"),
            }
        }
    }
}

/// Custom deserialisation implementation for Value that handles both JSON and binary formats.
/// For JSON, parses raw values directly.
/// For binary formats like bincode, reconstructs the full enum structure.
impl<'de> Deserialize<'de> for Value {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        /// Visitor implementation that handles conversion of raw values into Value enum variants.
        /// Supports both direct value parsing for JSON and enum variant parsing for binary formats.
        struct ValueVisitor;

        impl<'de> Visitor<'de> for ValueVisitor {
            type Value = Value;

            #[inline]
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a string, number, boolean, array, null, or Value enum")
            }

            #[inline]
            fn visit_str<E>(self, value: &str) -> Result<Value, E>
            where
                E: serde::de::Error,
            {
                Ok(Value::String(value.to_owned()))
            }

            #[inline]
            fn visit_string<E>(self, value: String) -> Result<Value, E>
            where
                E: serde::de::Error,
            {
                Ok(Value::String(value))
            }

            #[inline]
            fn visit_f32<E>(self, value: f32) -> Result<Value, E>
            where
                E: serde::de::Error,
            {
                Ok(Value::F32(value))
            }

            #[inline]
            fn visit_f64<E>(self, value: f64) -> Result<Value, E>
            where
                E: serde::de::Error,
            {
                Ok(Value::F64(value))
            }

            #[inline]
            fn visit_i8<E>(self, value: i8) -> Result<Value, E>
            where
                E: serde::de::Error,
            {
                Ok(Value::I8(value))
            }

            #[inline]
            fn visit_i16<E>(self, value: i16) -> Result<Value, E>
            where
                E: serde::de::Error,
            {
                Ok(Value::I16(value))
            }

            #[inline]
            fn visit_i32<E>(self, value: i32) -> Result<Value, E>
            where
                E: serde::de::Error,
            {
                Ok(Value::I32(value))
            }

            #[inline]
            fn visit_i64<E>(self, value: i64) -> Result<Value, E>
            where
                E: serde::de::Error,
            {
                Ok(Value::I64(value))
            }

            #[inline]
            fn visit_u8<E>(self, value: u8) -> Result<Value, E>
            where
                E: serde::de::Error,
            {
                Ok(Value::U8(value))
            }

            #[inline]
            fn visit_u16<E>(self, value: u16) -> Result<Value, E>
            where
                E: serde::de::Error,
            {
                Ok(Value::U16(value))
            }

            #[inline]
            fn visit_u32<E>(self, value: u32) -> Result<Value, E>
            where
                E: serde::de::Error,
            {
                Ok(Value::U32(value))
            }

            #[inline]
            fn visit_u64<E>(self, value: u64) -> Result<Value, E>
            where
                E: serde::de::Error,
            {
                Ok(Value::U64(value))
            }

            #[inline]
            fn visit_u128<E>(self, value: u128) -> Result<Value, E>
            where
                E: serde::de::Error,
            {
                Ok(Value::U128(value))
            }

            #[inline]
            fn visit_bool<E>(self, value: bool) -> Result<Value, E>
            where
                E: serde::de::Error,
            {
                Ok(Value::Boolean(value))
            }

            #[inline]
            fn visit_none<E>(self) -> Result<Value, E>
            where
                E: serde::de::Error,
            {
                Ok(Value::Empty)
            }

            /// Handles array values by recursively deserialising each element
            fn visit_seq<A>(self, mut seq: A) -> Result<Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut values = Vec::new();
                while let Some(value) = seq.next_element()? {
                    values.push(value);
                }
                Ok(Value::Array(values))
            }

            /// Handles binary format deserialisation using numeric indices to identify variants
            /// Maps indices 0-5 to corresponding Value enum variants
            fn visit_enum<A>(self, data: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::EnumAccess<'de>,
            {
                let (variant_idx, variant_data) = data.variant_seed(VariantIdxDeserializer)?;
                match variant_idx {
                    0 => Ok(Value::String(variant_data.newtype_variant()?)),
                    1 => Ok(Value::F32(variant_data.newtype_variant()?)),
                    2 => Ok(Value::F64(variant_data.newtype_variant()?)),
                    3 => Ok(Value::I8(variant_data.newtype_variant()?)),
                    4 => Ok(Value::I16(variant_data.newtype_variant()?)),
                    5 => Ok(Value::I32(variant_data.newtype_variant()?)),
                    6 => Ok(Value::I64(variant_data.newtype_variant()?)),
                    7 => Ok(Value::U8(variant_data.newtype_variant()?)),
                    8 => Ok(Value::U16(variant_data.newtype_variant()?)),
                    9 => Ok(Value::U32(variant_data.newtype_variant()?)),
                    10 => Ok(Value::U64(variant_data.newtype_variant()?)),
                    11 => Ok(Value::U128(variant_data.newtype_variant()?)),
                    12 => Ok(Value::Boolean(variant_data.newtype_variant()?)),
                    13 => Ok(Value::Array(variant_data.newtype_variant()?)),
                    14 => Ok(Value::Object(variant_data.newtype_variant()?)),
                    15 => {
                        variant_data.unit_variant()?;
                        Ok(Value::Empty)
                    }
                    _ => Err(serde::de::Error::invalid_value(
                        serde::de::Unexpected::Unsigned(variant_idx as u64),
                        &"variant index 0 through 5",
                    )),
                }
            }
        }

        /// Helper deserialiser for handling numeric variant indices in binary format
        struct VariantIdxDeserializer;

        impl<'de> DeserializeSeed<'de> for VariantIdxDeserializer {
            type Value = u32;
            #[inline]
            fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where
                D: Deserializer<'de>,
            {
                deserializer.deserialize_u32(self)
            }
        }

        impl<'de> Visitor<'de> for VariantIdxDeserializer {
            type Value = u32;

            #[inline]
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("variant index")
            }

            #[inline]
            fn visit_u32<E>(self, v: u32) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(v)
            }
        }
        // Choose deserialisation strategy based on format
        if deserializer.is_human_readable() {
            // For JSON, accept any value type
            deserializer.deserialize_any(ValueVisitor)
        } else {
            // For binary, use enum variant indices
            deserializer.deserialize_enum(
                "Value",
                &[
                    "String", "F32", "F64", "I8", "I16", "I32", "I64", "U8", "U16", "U32", "U64",
                    "U128", "Boolean", "Array", "Object", "Empty",
                ],
                ValueVisitor,
            )
        }
    }
}

/// Module for custom serialisation of property hashmaps
/// Ensures consistent handling of Value enum serialisation within property maps
pub mod properties_format {
    use super::*;

    #[inline]
    pub fn serialize<S>(
        properties: &HashMap<String, Value>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(properties.len()))?;
        for (k, v) in properties {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }

    #[inline]
    pub fn deserialize<'de, D>(deserializer: D) -> Result<HashMap<String, Value>, D::Error>
    where
        D: Deserializer<'de>,
    {
        HashMap::deserialize(deserializer)
    }
}

impl From<&str> for Value {
    #[inline]
    fn from(s: &str) -> Self {
        Value::String(s.trim_matches('"').to_string())
    }
}

impl From<String> for Value {
    #[inline]
    fn from(s: String) -> Self {
        Value::String(s.trim_matches('"').to_string())
    }
}
impl From<bool> for Value {
    #[inline]
    fn from(b: bool) -> Self {
        Value::Boolean(b)
    }
}

impl From<f32> for Value {
    #[inline]
    fn from(f: f32) -> Self {
        Value::F32(f)
    }
}

impl From<f64> for Value {
    #[inline]
    fn from(f: f64) -> Self {
        Value::F64(f)
    }
}

impl From<i8> for Value {
    #[inline]
    fn from(i: i8) -> Self {
        Value::I8(i)
    }
}

impl From<i16> for Value {
    #[inline]
    fn from(i: i16) -> Self {
        Value::I16(i)
    }
}

impl From<i32> for Value {
    #[inline]
    fn from(i: i32) -> Self {
        Value::I32(i)
    }
}

impl From<i64> for Value {
    #[inline]
    fn from(i: i64) -> Self {
        Value::I64(i)
    }
}

impl From<u8> for Value {
    #[inline]
    fn from(i: u8) -> Self {
        Value::U8(i)
    }
}

impl From<u16> for Value {
    #[inline]
    fn from(i: u16) -> Self {
        Value::U16(i)
    }
}

impl From<u32> for Value {
    #[inline]
    fn from(i: u32) -> Self {
        Value::U32(i)
    }
}

impl From<u64> for Value {
    #[inline]
    fn from(i: u64) -> Self {
        Value::U64(i)
    }
}

impl From<u128> for Value {
    #[inline]
    fn from(i: u128) -> Self {
        Value::U128(i)
    }
}

impl From<Vec<Value>> for Value {
    #[inline]
    fn from(v: Vec<Value>) -> Self {
        Value::Array(v)
    }
}

impl From<usize> for Value {
    #[inline]
    fn from(v: usize) -> Self {
        if cfg!(target_pointer_width = "64") {
            Value::U64(v as u64)
        } else {
            Value::U128(v as u128)
        }
    }
}

impl From<Value> for String {
    #[inline]
    fn from(v: Value) -> Self {
        match v {
            Value::String(s) => s,
            _ => panic!("Value is not a string"),
        }
    }
}
impl From<JsonValue> for Value {
    #[inline]
    fn from(v: JsonValue) -> Self {
        match v {
            JsonValue::String(s) => Value::String(s),
            JsonValue::Number(n) => {
                if n.is_u64() {
                    Value::U64(n.as_u64().unwrap() as u64)
                } else if n.is_i64() {
                    Value::I64(n.as_i64().unwrap())
                } else {
                    Value::F64(n.as_f64().unwrap())
                }
            }
            JsonValue::Bool(b) => Value::Boolean(b),
            JsonValue::Array(a) => Value::Array(a.into_iter().map(|v| v.into()).collect()),
            JsonValue::Object(o) => {
                Value::Object(o.into_iter().map(|(k, v)| (k, v.into())).collect())
            }
            JsonValue::Null => Value::Empty,
        }
    }
}

pub trait Encodings {
    fn decode_properties(bytes: &[u8]) -> Result<HashMap<String, Value>, GraphError>;
    fn encode_properties(&self) -> Result<Vec<u8>, GraphError>;
}

impl Encodings for HashMap<String, Value> {
    fn decode_properties(bytes: &[u8]) -> Result<HashMap<String, Value>, GraphError> {
        match bincode::deserialize(bytes) {
            Ok(properties) => Ok(properties),
            Err(e) => Err(GraphError::ConversionError(format!(
                "Error deserializing properties: {}",
                e
            ))),
        }
    }

    fn encode_properties(&self) -> Result<Vec<u8>, GraphError> {
        match bincode::serialize(self) {
            Ok(bytes) => Ok(bytes),
            Err(e) => Err(GraphError::ConversionError(format!(
                "Error serializing properties: {}",
                e
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bincode_variant_tag(value: &Value) -> u32 {
        let encoded = bincode::serialize(value).unwrap();
        u32::from_le_bytes(encoded[0..4].try_into().unwrap())
    }

    #[test]
    fn value_bincode_variant_tags_remain_stable() {
        let mut object = HashMap::new();
        object.insert("k".to_string(), Value::I32(1));

        let cases = [
            (Value::String("x".into()), 0),
            (Value::F32(1.0), 1),
            (Value::F64(1.0), 2),
            (Value::I8(1), 3),
            (Value::I16(1), 4),
            (Value::I32(1), 5),
            (Value::I64(1), 6),
            (Value::U8(1), 7),
            (Value::U16(1), 8),
            (Value::U32(1), 9),
            (Value::U64(1), 10),
            (Value::U128(1), 11),
            (Value::Boolean(true), 12),
            (Value::Array(vec![Value::I32(1)]), 13),
            (Value::Object(object), 14),
            (Value::Empty, 15),
        ];

        for (value, expected_tag) in cases {
            assert_eq!(bincode_variant_tag(&value), expected_tag, "{value:?}");
            let decoded: Value = bincode::deserialize(&bincode::serialize(&value).unwrap())
                .expect("value should round-trip through bincode");
            assert_eq!(decoded, value);
        }
    }

    #[test]
    fn value_orders_integer_widths_by_numeric_value() {
        assert_eq!(Value::I8(7).cmp(&Value::I64(7)), Ordering::Equal);
        assert_eq!(Value::U16(9).cmp(&Value::I32(10)), Ordering::Less);
        assert_eq!(
            Value::U128(i128::MAX as u128 + 1).cmp(&Value::I64(i64::MAX)),
            Ordering::Greater
        );
        assert_eq!(
            Value::U128(u128::MAX).cmp(&Value::U128(u128::MAX - 1)),
            Ordering::Greater
        );
    }

    #[test]
    fn value_numeric_equality_crosses_widths() {
        assert_eq!(Value::I32(42), Value::I64(42));
        assert_eq!(Value::U8(42), Value::I16(42));
        assert_eq!(Value::F32(42.0), Value::I64(42));
        assert_ne!(Value::F64(42.5), Value::I64(42));
    }

    #[test]
    fn value_mixed_types_have_stable_order() {
        assert!(Value::Empty < Value::String("a".into()));
        assert!(Value::String("a".into()) < Value::Boolean(false));
        assert!(Value::Boolean(false) < Value::Array(vec![]));
        assert!(Value::Array(vec![]) < Value::Object(HashMap::new()));
    }

    #[test]
    fn value_objects_compare_structurally() {
        let mut left = HashMap::new();
        left.insert("repo".to_string(), Value::String("context-engine".into()));
        left.insert("score".to_string(), Value::I32(10));

        let mut same = HashMap::new();
        same.insert("score".to_string(), Value::I64(10));
        same.insert("repo".to_string(), Value::String("context-engine".into()));

        let mut different = same.clone();
        different.insert("score".to_string(), Value::I64(11));

        let left = Value::Object(left);
        let same = Value::Object(same);
        let different = Value::Object(different);

        assert_eq!(left, same);
        assert_eq!(left.cmp(&same), Ordering::Equal);
        assert_ne!(left, different);
        assert!(left < different);
    }
}
