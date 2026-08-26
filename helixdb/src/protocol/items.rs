use super::value::{properties_format, Value};
use crate::helix_engine::types::GraphError;
use bincode::Options;
use sonic_rs::{Deserialize, Serialize};
use std::{cmp::Ordering, collections::HashMap};
use uuid::Uuid;

/// A node in the graph containing an ID, label, and property map.
/// Properties are serialised without enum variant names in JSON format.
#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub struct Node {
    #[serde(skip)]
    pub id: u128,
    pub label: String,
    #[serde(with = "properties_format")]
    pub properties: HashMap<String, Value>,
}

impl Eq for Node {}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Node {
    pub const NUM_PROPERTIES: usize = 2;
    pub fn new(label: &str, properties: Vec<(String, Value)>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().as_u128(),
            label: label.to_string(),
            properties: HashMap::from_iter(properties),
        }
    }
    pub fn new_with_id(label: &str, properties: Vec<(String, Value)>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().as_u128(),
            label: label.to_string(),
            properties: HashMap::from_iter(properties),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct SerializedNode {
    pub label: String,
    #[serde(with = "properties_format")]
    pub properties: HashMap<String, Value>,
}

impl SerializedNode {
    pub fn decode_node(bytes: &[u8], id: u128) -> Result<Node, GraphError> {
        let cfg = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_big_endian();

        match cfg.deserialize::<SerializedNode>(bytes) {
            Ok(node) => {
                let node = Node {
                    id,
                    label: node.label,
                    properties: node.properties,
                };
                Ok(node)
            }
            Err(e) => Err(GraphError::ConversionError(format!(
                "Error deserializing node: {}",
                e
            ))),
        }
    }

    pub fn encode_node(node: &Node) -> Result<Vec<u8>, GraphError> {
        let cfg = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_big_endian();

        cfg.serialize(node)
            .map_err(|e| GraphError::ConversionError(format!("Error serializing node: {}", e)))
    }
}
impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{ id: {}, label: {}, properties: {:?} }}",
            uuid::Uuid::from_u128(self.id).to_string(),
            self.label,
            self.properties
        )
    }
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{ id: {}, label: {}, properties: {:?} }}",
            uuid::Uuid::from_u128(self.id).to_string(),
            self.label,
            self.properties
        )
    }
}

/// An edge in the graph connecting two nodes with an ID, label, and property map.
/// Properties are serialised without enum variant names in JSON format.
#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct Edge {
    #[serde(skip)]
    pub id: u128, // TODO: change to uuid::Uuid and implement SERDE manually
    pub label: String,
    pub from_node: u128,
    pub to_node: u128,
    #[serde(with = "properties_format")]
    pub properties: HashMap<String, Value>,
}

impl Eq for Edge {}

impl Ord for Edge {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for Edge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Edge {
    pub const NUM_PROPERTIES: usize = 4;
    pub fn new(label: &str, properties: Vec<(String, Value)>) -> Self {
        Self {
            id: v6_uuid(),
            label: label.to_string(),
            from_node: 0,
            to_node: 0,
            properties: HashMap::from_iter(properties),
        }
    }

    pub fn new_with_id(label: &str, properties: Vec<(String, Value)>) -> Self {
        Self {
            id: v6_uuid(),
            label: label.to_string(),
            from_node: 0,
            to_node: 0,
            properties: HashMap::from_iter(properties),
        }
    }
}

impl std::fmt::Display for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{ id: {}, label: {}, from_node: {}, to_node: {}, properties: {:?} }}",
            uuid::Uuid::from_u128(self.id).to_string(),
            self.label,
            uuid::Uuid::from_u128(self.from_node).to_string(),
            uuid::Uuid::from_u128(self.to_node).to_string(),
            self.properties
        )
    }
}

impl std::fmt::Debug for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{ id: {}, label: {}, from_node: {}, to_node: {}, properties: {:?} }}",
            uuid::Uuid::from_u128(self.id).to_string(),
            self.label,
            uuid::Uuid::from_u128(self.from_node).to_string(),
            uuid::Uuid::from_u128(self.to_node).to_string(),
            self.properties
        )
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct SerializedEdge {
    pub label: String,
    pub from_node: u128,
    pub to_node: u128,
    #[serde(with = "properties_format")]
    pub properties: HashMap<String, Value>,
}

impl SerializedEdge {
    pub fn decode_edge(bytes: &[u8], id: u128) -> Result<Edge, GraphError> {
        let cfg = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_big_endian();

        match cfg.deserialize::<SerializedEdge>(bytes) {
            Ok(edge) => {
                let edge = Edge {
                    id,
                    label: edge.label,
                    from_node: edge.from_node,
                    to_node: edge.to_node,
                    properties: edge.properties,
                };
                Ok(edge)
            }
            Err(e) => Err(GraphError::ConversionError(format!(
                "Error deserializing edge: {}",
                e
            ))),
        }
    }

    pub fn encode_edge(edge: &Edge) -> Result<Vec<u8>, GraphError> {
        let cfg = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_big_endian();

        cfg.serialize(edge)
            .map_err(|e| GraphError::ConversionError(format!("Error serializing edge: {}", e)))
    }
}

#[inline(always)]
pub fn v6_uuid() -> u128 {
    Uuid::now_v6(&[1, 2, 3, 4, 5, 6]).as_u128()
}
