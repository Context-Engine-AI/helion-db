// router

// takes in raw [u8] data
// parses to request type

// then locks graph and passes parsed data and graph to handler to execute query

// returns response

use crate::helix_engine::{
    graph_core::graph_core::HelixGraphEngine,
    storage_core::{collection_manager::CollectionManager, replication::ReplicationManager},
    types::GraphError,
};
use core::fmt;
use std::{collections::HashMap, sync::Arc};

use crate::protocol::{request::Request, response::Response};

pub struct HandlerInput {
    pub request: Request,
    pub graph: Arc<HelixGraphEngine>,
    pub collections: Arc<CollectionManager>,
    pub replication: Arc<ReplicationManager>,
    /// Path parameters extracted from pattern routes (e.g., "name" from /collections/{name}/...)
    pub path_params: HashMap<String, String>,
}

// basic type for function pointer
pub type BasicHandlerFn = fn(&HandlerInput, &mut Response) -> Result<(), GraphError>;

// thread safe type for multi threaded use
pub type HandlerFn =
    Arc<dyn Fn(&HandlerInput, &mut Response) -> Result<(), GraphError> + Send + Sync>;

#[derive(Clone, Debug)]
pub struct HandlerSubmission(pub Handler);

#[derive(Clone, Debug)]
pub struct Handler {
    pub name: &'static str,
    pub func: BasicHandlerFn,
}

impl Handler {
    pub const fn new(name: &'static str, func: BasicHandlerFn) -> Self {
        Self { name, func }
    }
}

inventory::collect!(HandlerSubmission);

/// A pattern route entry: pattern segments + named param positions + handler.
struct PatternRoute {
    method: String,
    /// Path segments, e.g., ["collections", "{name}", "points", "search"]
    segments: Vec<String>,
    handler: HandlerFn,
}

pub struct HelixRouter {
    /// Method+Path => Function (exact match, fast path)
    pub routes: HashMap<(String, String), HandlerFn>,
    /// Pattern routes with path parameters (fallback)
    pattern_routes: Vec<PatternRoute>,
}

impl HelixRouter {
    /// Create a new router with a set of routes
    pub fn new(routes: Option<HashMap<(String, String), HandlerFn>>) -> Self {
        let rts = match routes {
            Some(routes) => routes,
            None => HashMap::new(),
        };
        Self {
            routes: rts,
            pattern_routes: Vec::new(),
        }
    }

    /// Add an exact route to the router
    pub fn add_route(&mut self, method: &str, path: &str, handler: BasicHandlerFn) {
        self.routes
            .insert((method.to_uppercase(), path.to_string()), Arc::new(handler));
    }

    /// Add a pattern route with path parameters.
    /// Use `{param_name}` for path parameters, e.g., `/collections/{name}/points/search`.
    pub fn add_pattern_route(&mut self, method: &str, pattern: &str, handler: BasicHandlerFn) {
        let segments: Vec<String> = pattern
            .trim_start_matches('/')
            .split('/')
            .map(|s| s.to_string())
            .collect();
        self.pattern_routes.push(PatternRoute {
            method: method.to_uppercase(),
            segments,
            handler: Arc::new(handler),
        });
    }

    /// Handle a request by finding the appropriate handler and executing it
    ///
    /// ## Arguments
    ///
    /// * `graph_access` - A reference to the graph engine
    /// * `request` - The request to handle
    /// * `response` - The response to write to
    ///
    /// ## Returns
    ///
    /// * `Ok(())` if the request was handled successfully
    /// * `Err(RouterError)` if there was an error handling the request
    pub fn handle(
        &self,
        graph_access: Arc<HelixGraphEngine>,
        collections: Arc<CollectionManager>,
        replication: Arc<ReplicationManager>,
        request: Request,
        response: &mut Response,
    ) -> Result<(), GraphError> {
        if let Some(proxy_response) = replication.prepare_cluster_request(&request)? {
            *response = proxy_response;
            return Ok(());
        }

        // Fast path: exact match
        let route_key = (request.method.clone(), request.path.clone());
        if let Some(handler) = self.routes.get(&route_key) {
            let input = HandlerInput {
                request,
                graph: Arc::clone(&graph_access),
                collections,
                replication,
                path_params: HashMap::new(),
            };
            return handler(&input, response);
        }

        // Slow path: pattern matching
        let req_segments: Vec<&str> = request.path.trim_start_matches('/').split('/').collect();

        for pattern in &self.pattern_routes {
            if pattern.method != request.method {
                continue;
            }
            if pattern.segments.len() != req_segments.len() {
                continue;
            }
            if let Some(params) = match_pattern_params(&pattern.segments, &req_segments) {
                let input = HandlerInput {
                    request,
                    graph: Arc::clone(&graph_access),
                    collections,
                    replication,
                    path_params: params,
                };
                return (pattern.handler)(&input, response);
            }
        }

        response.status = 404;
        response.body = b"404 - Not Found".to_vec();
        Ok(())
    }
}

fn match_pattern_params(
    pattern_segments: &[String],
    req_segments: &[&str],
) -> Option<HashMap<String, String>> {
    if pattern_segments.len() != req_segments.len() {
        return None;
    }
    let mut params = HashMap::new();
    for (pat_seg, req_seg) in pattern_segments.iter().zip(req_segments.iter()) {
        if pat_seg.starts_with('{') && pat_seg.ends_with('}') {
            let param_name = &pat_seg[1..pat_seg.len() - 1];
            params.insert(param_name.to_string(), decode_path_param_segment(req_seg));
        } else if pat_seg != req_seg {
            return None;
        }
    }
    Some(params)
}

fn decode_path_param_segment(segment: &str) -> String {
    let bytes = segment.as_bytes();
    if !bytes.contains(&b'%') {
        return segment.to_string();
    }

    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            if let (Some(high), Some(low)) =
                (hex_value(bytes[index + 1]), hex_value(bytes[index + 2]))
            {
                let value = (high << 4) | low;
                if matches!(value, b'/' | b'\\' | 0) {
                    decoded.extend_from_slice(&bytes[index..index + 3]);
                } else {
                    decoded.push(value);
                }
                index += 3;
                continue;
            }
        }
        decoded.push(bytes[index]);
        index += 1;
    }

    String::from_utf8(decoded).unwrap_or_else(|_| segment.to_string())
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[derive(Debug)]
pub enum RouterError {
    Io(std::io::Error),
    New(String),
}

impl fmt::Display for RouterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RouterError::Io(e) => write!(f, "IO error: {}", e),
            RouterError::New(msg) => write!(f, "Graph error: {}", msg),
        }
    }
}

impl From<String> for RouterError {
    fn from(error: String) -> Self {
        RouterError::New(error)
    }
}

impl From<std::io::Error> for RouterError {
    fn from(error: std::io::Error) -> Self {
        RouterError::Io(error)
    }
}

impl From<GraphError> for RouterError {
    fn from(error: GraphError) -> Self {
        RouterError::New(error.to_string())
    }
}

impl From<RouterError> for GraphError {
    fn from(error: RouterError) -> Self {
        GraphError::New(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn pattern_params_decode_percent_encoded_segment_once() {
        let pattern = segments(&["collections", "{name}", "index"]);
        let params = match_pattern_params(
            &pattern,
            &["collections", "wv1_transcript%20md%20test", "index"],
        )
        .unwrap();

        assert_eq!(params.get("name").unwrap(), "wv1_transcript md test");
    }

    #[test]
    fn pattern_params_keep_literal_percent_after_one_decode() {
        let pattern = segments(&["collections", "{name}"]);
        let params = match_pattern_params(&pattern, &["collections", "repo%2520name"]).unwrap();

        assert_eq!(params.get("name").unwrap(), "repo%20name");
    }

    #[test]
    fn pattern_params_do_not_decode_path_separators() {
        assert_eq!(decode_path_param_segment("a%2Fb%5Cc"), "a%2Fb%5Cc");
    }

    #[test]
    fn pattern_params_leave_malformed_percent_literals() {
        assert_eq!(decode_path_param_segment("bad%zz%"), "bad%zz%");
    }
}
