use prost::Message as ProstMessage;
use raft::prelude::Message;

use crate::helix_engine::storage_core::replication::{ReplicatedMutation, RAFT_SECRET_HEADER};
use crate::helix_engine::types::GraphError;
use crate::helix_gateway::router::router::HandlerInput;
use crate::protocol::response::Response;

fn json_response(
    response: &mut Response,
    status: u16,
    body: &impl serde::Serialize,
) -> Result<(), GraphError> {
    response.status = status;
    response.body = sonic_rs::to_vec(body)?;
    response
        .headers
        .insert("Content-Type".to_string(), "application/json".to_string());
    Ok(())
}

/// Validate the shared Raft secret on incoming internal RPCs.
/// If a secret is configured, the request must include a matching
/// `X-Raft-Secret` header; otherwise the request is rejected with 403.
/// If no secret is configured, all requests are allowed (dev/test mode).
fn verify_raft_secret(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    if let Some(expected) = input.replication.raft_secret() {
        // Header keys are stored lower-cased by the request parser.
        let provided = input
            .request
            .headers
            .get(&RAFT_SECRET_HEADER.to_ascii_lowercase());
        match provided {
            Some(val) if val == &expected => Ok(()),
            _ => {
                response.status = 403;
                response.body = b"forbidden: invalid or missing raft secret".to_vec();
                Err(GraphError::New("raft secret mismatch".into()))
            }
        }
    } else {
        Ok(())
    }
}

pub fn handle_status(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    // Status is informational — no auth required.
    json_response(response, 200, &input.replication.status()?)
}

pub fn handle_message(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    verify_raft_secret(input, response)?;
    let message = Message::decode(input.request.body.as_slice())
        .map_err(|e| GraphError::New(format!("invalid raft message: {}", e)))?;
    input.replication.receive_raft_message(message)?;
    response.status = 204;
    response.body.clear();
    Ok(())
}

pub fn handle_propose(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    verify_raft_secret(input, response)?;
    let mutation: ReplicatedMutation = bincode::deserialize(&input.request.body)
        .map_err(|e| GraphError::New(format!("invalid raft proposal: {}", e)))?;
    input.replication.propose_internal(mutation)?;
    json_response(response, 200, &sonic_rs::json!({ "status": "ok" }))
}
