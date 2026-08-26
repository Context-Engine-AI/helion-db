use crate::protocol::value::Value;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Qdrant-compatible filter structures.
/// Evaluates filter conditions against node properties (payloads).

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Filter {
    #[serde(default)]
    pub must: Vec<Condition>,
    #[serde(default)]
    pub must_not: Vec<Condition>,
    #[serde(default)]
    pub should: Vec<Condition>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Condition {
    Field(FieldCondition),
    HasId(HasIdCondition),
    Nested(Filter),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HasIdCondition {
    pub has_id: Vec<serde_json::Value>,
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
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MatchValue {
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MatchAny {
    pub any: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MatchText {
    pub text: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
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
        true
    }

    pub fn is_empty(&self) -> bool {
        self.must.is_empty() && self.must_not.is_empty() && self.should.is_empty()
    }
}

impl Condition {
    fn matches_point(&self, id: Option<u128>, properties: &HashMap<String, Value>) -> bool {
        match self {
            Condition::Field(cond) => cond.matches(properties),
            Condition::HasId(cond) => id
                .map(|point_id| cond.matches_id(point_id))
                .unwrap_or(false),
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
                None => false,
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
        serde_json::Value::String(s) => {
            let clean = s.trim().trim_start_matches("0x");
            u128::from_str_radix(clean, 16)
                .ok()
                .or_else(|| clean.parse::<u128>().ok())
        }
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
}
