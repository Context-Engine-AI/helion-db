use crate::helix_gateway::api::qdrant::string_id_to_u128;
use crate::protocol::value::Value;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;

/// Qdrant-compatible filter structures.
/// Evaluates filter conditions against node properties (payloads).

/// Unknown keys (e.g. `nested`, geo conditions) are rejected rather than
/// ignored: an ignored clause silently widens or narrows results.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Filter {
    #[serde(default)]
    pub must: Vec<Condition>,
    #[serde(default)]
    pub must_not: Vec<Condition>,
    #[serde(default)]
    pub should: Vec<Condition>,
    /// Qdrant `min_should`: at least `min_count` of `conditions` must match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_should: Option<MinShould>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MinShould {
    pub conditions: Vec<Condition>,
    pub min_count: usize,
}

/// Deserialized by shape (see `Condition::from_json`) instead of
/// `#[serde(untagged)]`: untagged parsing let unsupported shapes fall through
/// to an empty `Nested` filter (match-all in `must`, match-none in
/// `must_not`) or a bare field-exists check.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Condition {
    Field(FieldCondition),
    HasId(HasIdCondition),
    IsEmpty(IsEmptyCondition),
    IsNull(IsNullCondition),
    Nested(Filter),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HasIdCondition {
    pub has_id: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadKey {
    pub key: String,
}

/// Qdrant `{"is_empty": {"key": ..}}`: field missing, null, or `[]`.
#[derive(Debug, Clone, Serialize)]
pub struct IsEmptyCondition {
    pub is_empty: PayloadKey,
}

/// Qdrant `{"is_null": {"key": ..}}`: field present with a null value.
#[derive(Debug, Clone, Serialize)]
pub struct IsNullCondition {
    pub is_null: PayloadKey,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FieldCondition {
    pub key: String,
    #[serde(rename = "match")]
    pub match_cond: Option<MatchCondition>,
    pub range: Option<RangeCondition>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MatchCondition {
    Value(MatchValue),
    Any(MatchAny),
    Text(MatchText),
    Except(MatchExcept),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MatchValue {
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MatchAny {
    pub any: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MatchText {
    pub text: String,
}

/// Qdrant `match.except`: value is none of the listed values. Like Qdrant,
/// a missing/null/empty field matches, and an array matches when at least
/// one element is outside the list.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MatchExcept {
    pub except: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RangeCondition {
    #[serde(default)]
    pub gte: Option<f64>,
    #[serde(default)]
    pub gt: Option<f64>,
    #[serde(default)]
    pub lte: Option<f64>,
    #[serde(default)]
    pub lt: Option<f64>,
}

impl<'de> Deserialize<'de> for Condition {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = serde_json::Value::deserialize(deserializer)?;
        Condition::from_json(raw).map_err(serde::de::Error::custom)
    }
}

impl Condition {
    fn from_json(raw: serde_json::Value) -> Result<Self, String> {
        let serde_json::Value::Object(mut object) = raw else {
            return Err("filter condition must be a JSON object".to_string());
        };
        if object.contains_key("key") {
            return Self::field_from_json(object);
        }
        for (name, build) in [
            (
                "has_id",
                Self::has_id_from_json as fn(_) -> Result<Self, String>,
            ),
            ("is_empty", Self::is_empty_from_json),
            ("is_null", Self::is_null_from_json),
        ] {
            if let Some(value) = object.remove(name) {
                if let Some(extra) = object.keys().next() {
                    return Err(format!(
                        "unsupported key `{extra}` alongside `{name}` in filter condition"
                    ));
                }
                return build(value);
            }
        }
        serde_json::from_value::<Filter>(serde_json::Value::Object(object))
            .map(Condition::Nested)
            .map_err(|e| format!("unsupported filter condition: {e}"))
    }

    fn has_id_from_json(value: serde_json::Value) -> Result<Self, String> {
        serde_json::from_value(value)
            .map(|has_id| Condition::HasId(HasIdCondition { has_id }))
            .map_err(|e| format!("invalid has_id condition: {e}"))
    }

    fn is_empty_from_json(value: serde_json::Value) -> Result<Self, String> {
        serde_json::from_value(value)
            .map(|is_empty| Condition::IsEmpty(IsEmptyCondition { is_empty }))
            .map_err(|e| format!("invalid is_empty condition: {e}"))
    }

    fn is_null_from_json(value: serde_json::Value) -> Result<Self, String> {
        serde_json::from_value(value)
            .map(|is_null| Condition::IsNull(IsNullCondition { is_null }))
            .map_err(|e| format!("invalid is_null condition: {e}"))
    }

    /// Field condition keyed by `key`. Besides Qdrant's `match`/`range`, this
    /// accepts the gRPC-style `"is_empty": bool` / `"is_null": bool` flags
    /// Context Engine emits on a field condition. Multiple clauses in one
    /// object are ANDed.
    fn field_from_json(object: serde_json::Map<String, serde_json::Value>) -> Result<Self, String> {
        let mut key = None;
        let mut match_cond = None;
        let mut range = None;
        let mut flags: Vec<(bool, bool)> = Vec::new(); // (is_null?, expected)
        for (name, value) in object {
            match name.as_str() {
                "key" => match value {
                    serde_json::Value::String(k) => key = Some(k),
                    _ => return Err("field condition `key` must be a string".to_string()),
                },
                _ if value.is_null() => {}
                "match" => {
                    match_cond = Some(
                        serde_json::from_value::<MatchCondition>(value.clone())
                            .map_err(|_| format!("unsupported match condition: {value}"))?,
                    )
                }
                "range" => {
                    range = Some(
                        serde_json::from_value::<RangeCondition>(value.clone())
                            .map_err(|_| format!("unsupported range condition: {value}"))?,
                    )
                }
                "is_empty" | "is_null" => match value {
                    serde_json::Value::Bool(expected) => flags.push((name == "is_null", expected)),
                    _ => return Err(format!("field condition `{name}` must be a boolean")),
                },
                other => {
                    return Err(format!("unsupported field condition key `{other}`"));
                }
            }
        }
        let key = key.ok_or_else(|| "field condition `key` must be a string".to_string())?;

        let mut parts = Vec::new();
        if match_cond.is_some() {
            parts.push(Condition::Field(FieldCondition {
                key: key.clone(),
                match_cond,
                range: None,
            }));
        }
        if range.is_some() {
            parts.push(Condition::Field(FieldCondition {
                key: key.clone(),
                match_cond: None,
                range,
            }));
        }
        for (is_null, expected) in flags {
            let payload_key = PayloadKey { key: key.clone() };
            let condition = if is_null {
                Condition::IsNull(IsNullCondition {
                    is_null: payload_key,
                })
            } else {
                Condition::IsEmpty(IsEmptyCondition {
                    is_empty: payload_key,
                })
            };
            parts.push(if expected {
                condition
            } else {
                Condition::Nested(Filter {
                    must_not: vec![condition],
                    ..Default::default()
                })
            });
        }
        Ok(match parts.len() {
            // Bare `{"key": ..}` keeps the legacy field-exists check.
            0 => Condition::Field(FieldCondition {
                key,
                match_cond: None,
                range: None,
            }),
            1 => parts.pop().expect("one part"),
            _ => Condition::Nested(Filter {
                must: parts,
                ..Default::default()
            }),
        })
    }
}

impl Filter {
    /// Evaluate the filter against a set of properties (payload).
    /// Returns true if the properties pass all filter conditions.
    pub fn matches(&self, properties: &HashMap<String, Value>) -> bool {
        self.matches_point(None, properties)
    }

    /// Evaluate the filter against point id + properties.
    ///
    /// Qdrant clients can put `has_id` and nested boolean filters inside
    /// `must`/`should`/`must_not`. The original flat field-only parser rejected
    /// those shapes, which broke compatibility for client-side helpers that
    /// exclude already-seen ids or group OR clauses under a nested filter.
    pub fn matches_point(&self, id: Option<u128>, properties: &HashMap<String, Value>) -> bool {
        // All must conditions must match
        if !self
            .must
            .iter()
            .all(|cond| cond.matches_point(id, properties))
        {
            return false;
        }
        // No must_not conditions should match
        if self
            .must_not
            .iter()
            .any(|cond| cond.matches_point(id, properties))
        {
            return false;
        }
        // If should is non-empty, at least one should match
        if !self.should.is_empty()
            && !self
                .should
                .iter()
                .any(|cond| cond.matches_point(id, properties))
        {
            return false;
        }
        if let Some(min_should) = &self.min_should {
            let mut matched = 0usize;
            for cond in &min_should.conditions {
                if matched >= min_should.min_count {
                    break;
                }
                if cond.matches_point(id, properties) {
                    matched += 1;
                }
            }
            if matched < min_should.min_count {
                return false;
            }
        }
        true
    }

    pub fn is_empty(&self) -> bool {
        self.must.is_empty()
            && self.must_not.is_empty()
            && self.should.is_empty()
            && self.min_should.is_none()
    }
}

impl Condition {
    fn matches_point(&self, id: Option<u128>, properties: &HashMap<String, Value>) -> bool {
        match self {
            Condition::Field(cond) => cond.matches(properties),
            Condition::HasId(cond) => id
                .map(|point_id| cond.matches_id(point_id))
                .unwrap_or(false),
            Condition::IsEmpty(cond) => match resolve_nested_key(properties, &cond.is_empty.key) {
                None | Some(Value::Empty) => true,
                Some(Value::Array(values)) => values.is_empty(),
                Some(_) => false,
            },
            Condition::IsNull(cond) => matches!(
                resolve_nested_key(properties, &cond.is_null.key),
                Some(Value::Empty)
            ),
            Condition::Nested(filter) => filter.matches_point(id, properties),
        }
    }

    pub fn as_field(&self) -> Option<&FieldCondition> {
        match self {
            Condition::Field(cond) => Some(cond),
            _ => None,
        }
    }
}

impl From<FieldCondition> for Condition {
    fn from(value: FieldCondition) -> Self {
        Condition::Field(value)
    }
}

impl HasIdCondition {
    fn matches_id(&self, id: u128) -> bool {
        self.has_id
            .iter()
            .any(|raw| point_id_json_to_u128(raw) == Some(id))
    }
}

impl FieldCondition {
    fn matches(&self, properties: &HashMap<String, Value>) -> bool {
        let value = resolve_nested_key(properties, &self.key);

        if let Some(match_cond) = &self.match_cond {
            match value {
                Some(val) => match_cond.matches(val),
                // A missing field is "none of" any list.
                None => matches!(match_cond, MatchCondition::Except(_)),
            }
        } else if let Some(range) = &self.range {
            match value {
                Some(val) => range.matches(val),
                None => false,
            }
        } else {
            // No condition specified — just check field exists
            value.is_some()
        }
    }
}

fn point_id_json_to_u128(value: &serde_json::Value) -> Option<u128> {
    match value {
        serde_json::Value::Number(n) => n.as_u64().map(|v| v as u128),
        // Must mirror ingest (`point_id_to_u128`): hex, else hashed string
        // (UUIDs and other non-hex ids).
        serde_json::Value::String(s) => Some(string_id_to_u128(s)),
        _ => None,
    }
}

impl MatchCondition {
    fn matches(&self, value: &Value) -> bool {
        match self {
            MatchCondition::Value(mv) => value_matches_json(value, &mv.value),
            MatchCondition::Any(ma) => ma.any.iter().any(|v| value_matches_json(value, v)),
            MatchCondition::Text(mt) => match value {
                Value::String(s) => s.to_lowercase().contains(&mt.text.to_lowercase()),
                _ => false,
            },
            MatchCondition::Except(me) => match value {
                Value::Empty => true,
                Value::Array(values) => {
                    values.is_empty()
                        || values
                            .iter()
                            .any(|v| !me.except.iter().any(|e| value_matches_json(v, e)))
                }
                _ => !me.except.iter().any(|e| value_matches_json(value, e)),
            },
        }
    }
}

impl RangeCondition {
    fn matches(&self, value: &Value) -> bool {
        let num = value_to_f64(value);
        match num {
            Some(n) => {
                if let Some(gte) = self.gte {
                    if n < gte {
                        return false;
                    }
                }
                if let Some(gt) = self.gt {
                    if n <= gt {
                        return false;
                    }
                }
                if let Some(lte) = self.lte {
                    if n > lte {
                        return false;
                    }
                }
                if let Some(lt) = self.lt {
                    if n >= lt {
                        return false;
                    }
                }
                true
            }
            None => false,
        }
    }
}

/// Resolve a dotted key path like "metadata.language" against nested properties.
fn resolve_nested_key<'a>(properties: &'a HashMap<String, Value>, key: &str) -> Option<&'a Value> {
    let parts: Vec<&str> = key.split('.').collect();
    if parts.is_empty() {
        return None;
    }

    let mut current = properties.get(parts[0])?;
    for &part in &parts[1..] {
        match current {
            Value::Object(map) => {
                current = map.get(part)?;
            }
            _ => return None,
        }
    }
    Some(current)
}

/// Compare a Value against a serde_json::Value.
fn value_matches_json(val: &Value, json_val: &serde_json::Value) -> bool {
    match val {
        Value::Array(values) => values
            .iter()
            .any(|value| value_matches_json(value, json_val)),
        _ => val == &Value::from(json_val.clone()),
    }
}

fn value_to_f64(val: &Value) -> Option<f64> {
    let number = match val {
        Value::F32(n) => Some(*n as f64),
        Value::F64(n) => Some(*n),
        Value::I32(n) => Some(*n as f64),
        Value::I64(n) => Some(*n as f64),
        Value::U32(n) => Some(*n as f64),
        Value::U64(n) => Some(*n as f64),
        _ => None,
    };

    number.filter(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_props() -> HashMap<String, Value> {
        let mut metadata = HashMap::new();
        metadata.insert("language".into(), Value::String("python".into()));
        metadata.insert("repo".into(), Value::String("context-engine".into()));
        metadata.insert("kind".into(), Value::String("function".into()));
        metadata.insert("line_count".into(), Value::I32(42));
        metadata.insert("visits".into(), Value::U64(42));

        let mut props = HashMap::new();
        props.insert("content".into(), Value::String("def main():".into()));
        props.insert("metadata".into(), Value::Object(metadata));
        props
    }

    #[test]
    fn test_match_value() {
        let filter = Filter {
            must: vec![FieldCondition {
                key: "metadata.language".into(),
                match_cond: Some(MatchCondition::Value(MatchValue {
                    value: serde_json::Value::String("python".into()),
                })),
                range: None,
            }
            .into()],
            ..Default::default()
        };
        assert!(filter.matches(&test_props()));
    }

    #[test]
    fn test_match_value_uses_value_numeric_equality() {
        let filter = Filter {
            must: vec![FieldCondition {
                key: "metadata.visits".into(),
                match_cond: Some(MatchCondition::Value(MatchValue {
                    value: serde_json::json!(42),
                })),
                range: None,
            }
            .into()],
            ..Default::default()
        };
        assert!(filter.matches(&test_props()));
    }

    #[test]
    fn test_match_value_no_match() {
        let filter = Filter {
            must: vec![FieldCondition {
                key: "metadata.language".into(),
                match_cond: Some(MatchCondition::Value(MatchValue {
                    value: serde_json::Value::String("rust".into()),
                })),
                range: None,
            }
            .into()],
            ..Default::default()
        };
        assert!(!filter.matches(&test_props()));
    }

    #[test]
    fn test_match_any() {
        let filter = Filter {
            must: vec![FieldCondition {
                key: "metadata.kind".into(),
                match_cond: Some(MatchCondition::Any(MatchAny {
                    any: vec![
                        serde_json::Value::String("function".into()),
                        serde_json::Value::String("class".into()),
                    ],
                })),
                range: None,
            }
            .into()],
            ..Default::default()
        };
        assert!(filter.matches(&test_props()));
    }

    #[test]
    fn test_must_not() {
        let filter = Filter {
            must_not: vec![FieldCondition {
                key: "metadata.language".into(),
                match_cond: Some(MatchCondition::Value(MatchValue {
                    value: serde_json::Value::String("python".into()),
                })),
                range: None,
            }
            .into()],
            ..Default::default()
        };
        assert!(!filter.matches(&test_props()));
    }

    #[test]
    fn test_range_filter() {
        let filter = Filter {
            must: vec![FieldCondition {
                key: "metadata.line_count".into(),
                match_cond: None,
                range: Some(RangeCondition {
                    gte: Some(10.0),
                    lte: Some(100.0),
                    gt: None,
                    lt: None,
                }),
            }
            .into()],
            ..Default::default()
        };
        assert!(filter.matches(&test_props()));
    }

    #[test]
    fn range_filter_rejects_non_finite_float_values() {
        let mut props = test_props();
        let metadata = props.get_mut("metadata").unwrap();
        let Value::Object(metadata) = metadata else {
            panic!("metadata must be an object");
        };
        metadata.insert("line_count".into(), Value::F64(f64::NAN));

        let filter = Filter {
            must: vec![FieldCondition {
                key: "metadata.line_count".into(),
                match_cond: None,
                range: Some(RangeCondition {
                    gte: Some(10.0),
                    lte: Some(100.0),
                    gt: None,
                    lt: None,
                }),
            }
            .into()],
            ..Default::default()
        };
        assert!(!filter.matches(&props));
    }

    #[test]
    fn test_empty_filter_matches_all() {
        let filter = Filter::default();
        assert!(filter.matches(&test_props()));
    }

    #[test]
    fn test_nested_key_missing() {
        let filter = Filter {
            must: vec![FieldCondition {
                key: "metadata.nonexistent".into(),
                match_cond: Some(MatchCondition::Value(MatchValue {
                    value: serde_json::Value::String("x".into()),
                })),
                range: None,
            }
            .into()],
            ..Default::default()
        };
        assert!(!filter.matches(&test_props()));
    }

    #[test]
    fn test_has_id_condition() {
        let filter = Filter {
            must_not: vec![Condition::HasId(HasIdCondition {
                has_id: vec![serde_json::Value::String(
                    "0000000000000000000000000000002a".into(),
                )],
            })],
            ..Default::default()
        };
        assert!(!filter.matches_point(Some(42), &test_props()));
        assert!(filter.matches_point(Some(43), &test_props()));
    }

    #[test]
    fn test_nested_should_filter() {
        let filter = Filter {
            must: vec![Condition::Nested(Filter {
                should: vec![
                    FieldCondition {
                        key: "metadata.language".into(),
                        match_cond: Some(MatchCondition::Value(MatchValue {
                            value: serde_json::Value::String("rust".into()),
                        })),
                        range: None,
                    }
                    .into(),
                    FieldCondition {
                        key: "metadata.language".into(),
                        match_cond: Some(MatchCondition::Value(MatchValue {
                            value: serde_json::Value::String("python".into()),
                        })),
                        range: None,
                    }
                    .into(),
                ],
                ..Default::default()
            })],
            ..Default::default()
        };
        assert!(filter.matches(&test_props()));
    }

    fn parse(json: serde_json::Value) -> Result<Filter, serde_json::Error> {
        serde_json::from_value(json)
    }

    #[test]
    fn parses_every_condition_shape_context_engine_emits() {
        // Mirrors ctxce-search rest.rs filter_to_json / match_to_json /
        // range_to_json / has_id output.
        let filter = parse(serde_json::json!({
            "must": [
                {"key": "metadata.language", "match": {"value": "python"}},
                {"key": "metadata.line_count", "match": {"value": 42}},
                {"key": "metadata.flag", "match": {"value": true}},
                {"key": "metadata.repo", "match": {"text": "context"}},
                {"key": "metadata.kind", "match": {"any": ["function", "class"]}},
                {"key": "metadata.visits", "match": {"any": [41, 42]}},
                {"key": "metadata.kind", "match": {"except": ["struct"]}},
                {"key": "metadata.visits", "match": {"except": [7]}},
                {"key": "metadata.line_count", "range": {"lt": 100, "lte": 100, "gt": 1, "gte": 1}},
                {"key": "metadata.missing", "is_empty": true},
                {"key": "metadata.language", "is_empty": false},
                {"key": "metadata.language", "is_null": false},
                {"has_id": ["0000000000000000000000000000002a", 42]},
                {"should": [{"key": "metadata.language", "match": {"value": "python"}}]}
            ],
            "should": [{"key": "metadata.kind", "match": {"value": "function"}}],
            "must_not": [{"key": "metadata.path", "match": {"text": "vendor/"}}]
        }))
        .expect("CE filter shapes must parse");
        assert_eq!(filter.must.len(), 14);
        let mut props = test_props();
        let Some(Value::Object(metadata)) = props.get_mut("metadata") else {
            panic!("metadata must be an object");
        };
        metadata.insert("flag".into(), Value::Boolean(true));
        assert!(filter.matches_point(Some(42), &props));
        assert!(!filter.matches_point(Some(43), &props));
    }

    #[test]
    fn rejects_unsupported_condition_shapes_instead_of_matching_all() {
        for json in [
            serde_json::json!({"must": [{"key": "x", "match": {"phrase": "a"}}]}),
            serde_json::json!({"must": [{"key": "x", "match": {"value": "a", "any": ["b"]}}]}),
            serde_json::json!({"must": [{"key": "t", "range": {"gte": "2024-01-01T00:00:00Z"}}]}),
            serde_json::json!({"must": [{"key": "x", "values_count": {"gt": 1}}]}),
            serde_json::json!({"must": [{"key": "x", "is_empty": "yes"}]}),
            serde_json::json!({"must": [{"nested": {"key": "a", "filter": {}}}]}),
            serde_json::json!({"must": [{"geo_radius": {"center": {}, "radius": 1.0}}]}),
            serde_json::json!({"must": [{"is_empty": {"key": "x"}, "extra": 1}]}),
            serde_json::json!({"must_not": [{"is_empty": {"field": "x"}}]}),
            serde_json::json!({"must": ["x"]}),
            serde_json::json!({"min_should": {"conditions": [], "min_count": 1, "extra": 1}}),
            serde_json::json!({"min_should": {"conditions": []}}),
        ] {
            assert!(parse(json.clone()).is_err(), "must reject {json}");
        }
    }

    #[test]
    fn min_should_requires_min_count_matching_conditions() {
        let props = test_props();
        let filter = |min_count: usize| {
            parse(serde_json::json!({
                "min_should": {
                    "conditions": [
                        {"key": "metadata.language", "match": {"value": "python"}},
                        {"key": "metadata.visits", "match": {"value": 42}},
                        {"key": "metadata.language", "match": {"value": "go"}}
                    ],
                    "min_count": min_count
                }
            }))
            .unwrap()
        };
        assert!(filter(0).matches(&props));
        assert!(filter(1).matches(&props));
        assert!(filter(2).matches(&props));
        assert!(!filter(3).matches(&props));
        assert!(!filter(2).is_empty());

        // Combined with must, both constraints apply.
        let combined = parse(serde_json::json!({
            "must": [{"key": "metadata.language", "match": {"value": "python"}}],
            "min_should": {
                "conditions": [{"key": "metadata.language", "match": {"value": "go"}}],
                "min_count": 1
            }
        }))
        .unwrap();
        assert!(!combined.matches(&props));
        // Nested filters accept min_should too.
        let nested = parse(serde_json::json!({
            "must": [{"min_should": {
                "conditions": [{"key": "metadata.visits", "match": {"value": 42}}],
                "min_count": 1
            }}]
        }))
        .unwrap();
        assert!(nested.matches(&props));
    }

    #[test]
    fn match_except_follows_qdrant_semantics() {
        let mut props = test_props();
        props.insert(
            "tags".into(),
            Value::Array(vec![Value::String("a".into()), Value::String("b".into())]),
        );
        props.insert("empty_tags".into(), Value::Array(Vec::new()));
        let matches = |cond: serde_json::Value| {
            parse(serde_json::json!({"must": [cond]}))
                .unwrap()
                .matches(&props)
        };
        assert!(!matches(
            serde_json::json!({"key": "metadata.language", "match": {"except": ["python", "rust"]}})
        ));
        assert!(matches(
            serde_json::json!({"key": "metadata.language", "match": {"except": ["rust"]}})
        ));
        assert!(!matches(
            serde_json::json!({"key": "metadata.visits", "match": {"except": [42]}})
        ));
        assert!(matches(
            serde_json::json!({"key": "metadata.missing", "match": {"except": ["x"]}})
        ));
        assert!(matches(
            serde_json::json!({"key": "tags", "match": {"except": ["a"]}})
        ));
        assert!(!matches(
            serde_json::json!({"key": "tags", "match": {"except": ["a", "b"]}})
        ));
        assert!(matches(
            serde_json::json!({"key": "empty_tags", "match": {"except": ["a"]}})
        ));
        // In must_not, except is no longer an empty (match-none) filter.
        let filter = parse(serde_json::json!({
            "must_not": [{"key": "metadata.language", "match": {"except": ["python"]}}]
        }))
        .unwrap();
        assert!(filter.matches(&props));
    }

    #[test]
    fn is_empty_and_is_null_follow_qdrant_semantics() {
        let mut props = test_props();
        props.insert("null_field".into(), Value::Empty);
        props.insert("empty_list".into(), Value::Array(Vec::new()));
        props.insert("list".into(), Value::Array(vec![Value::I64(1)]));
        let matches = |cond: serde_json::Value| {
            parse(serde_json::json!({"must": [cond]}))
                .unwrap()
                .matches(&props)
        };
        for key in ["missing", "null_field", "empty_list"] {
            assert!(
                matches(serde_json::json!({"is_empty": {"key": key}})),
                "{key}"
            );
            assert!(
                matches(serde_json::json!({"key": key, "is_empty": true})),
                "{key}"
            );
            assert!(
                !matches(serde_json::json!({"key": key, "is_empty": false})),
                "{key}"
            );
        }
        for key in ["list", "metadata.language"] {
            assert!(
                !matches(serde_json::json!({"is_empty": {"key": key}})),
                "{key}"
            );
            assert!(
                matches(serde_json::json!({"key": key, "is_empty": false})),
                "{key}"
            );
        }
        assert!(matches(
            serde_json::json!({"is_null": {"key": "null_field"}})
        ));
        assert!(matches(
            serde_json::json!({"key": "null_field", "is_null": true})
        ));
        for key in ["missing", "empty_list", "metadata.language"] {
            assert!(
                !matches(serde_json::json!({"is_null": {"key": key}})),
                "{key}"
            );
            assert!(
                matches(serde_json::json!({"key": key, "is_null": false})),
                "{key}"
            );
        }
        // Previously parsed as an empty nested filter, i.e. match-none here.
        let filter =
            parse(serde_json::json!({"must_not": [{"is_empty": {"key": "missing"}}]})).unwrap();
        assert!(!filter.matches(&props));
        let filter =
            parse(serde_json::json!({"must_not": [{"is_empty": {"key": "list"}}]})).unwrap();
        assert!(filter.matches(&props));
    }

    #[test]
    fn has_id_matches_hashed_string_ids_like_ingest() {
        let uuid = "550e8400-e29b-41d4-a716-446655440000";
        let filter =
            parse(serde_json::json!({"must": [{"has_id": [uuid, "chunk:src/lib.rs#3"]}]})).unwrap();
        assert!(filter.matches_point(Some(string_id_to_u128(uuid)), &test_props()));
        assert!(filter.matches_point(Some(string_id_to_u128("chunk:src/lib.rs#3")), &test_props()));
        assert!(!filter.matches_point(Some(42), &test_props()));
        let hex = parse(serde_json::json!({"must": [{"has_id": ["0x2a", 7]}]})).unwrap();
        assert!(hex.matches_point(Some(42), &test_props()));
        assert!(hex.matches_point(Some(7), &test_props()));
    }
}
