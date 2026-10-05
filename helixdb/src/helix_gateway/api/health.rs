use crate::helix_engine::storage_core::reader_warm;
use crate::helix_engine::types::GraphError;
use crate::helix_gateway::router::router::HandlerInput;
use crate::protocol::response::Response;

pub fn handle_health(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    if write_probe_unhealthy_lsm_writer(input, response, 500)? {
        return Ok(());
    }
    response.status = 200;
    response.body = b"{\"status\":\"ok\"}".to_vec();
    response
        .headers
        .insert("Content-Type".to_string(), "application/json".to_string());
    Ok(())
}

pub fn handle_ready(input: &HandlerInput, response: &mut Response) -> Result<(), GraphError> {
    if write_probe_unhealthy_lsm_writer(input, response, 503)? {
        return Ok(());
    }
    if !reader_warm::startup_warm_ready() {
        response.status = 503;
        response.body = b"{\"status\":\"warming\"}".to_vec();
        response
            .headers
            .insert("Content-Type".to_string(), "application/json".to_string());
        return Ok(());
    }
    let collections = &input.collections;
    let list = collections.list_collections().unwrap_or_default();
    response.status = 200;
    let body = sonic_rs::json!({
        "status": "ready",
        "collections_on_disk": list.len(),
        "collections_loaded": collections.loaded_count(),
    });
    response.body = sonic_rs::to_vec(&body)?;
    response
        .headers
        .insert("Content-Type".to_string(), "application/json".to_string());
    Ok(())
}

fn write_probe_unhealthy_lsm_writer(
    input: &HandlerInput,
    response: &mut Response,
    status: u16,
) -> Result<bool, GraphError> {
    let unhealthy = input.collections.unhealthy_lsm_writer_close_reasons()?;
    let Some((collection, reason)) = unhealthy.first() else {
        return Ok(false);
    };
    response.status = status;
    let body = sonic_rs::json!({
        "status": "unhealthy",
        "reason": "lsm_writer_closed",
        "collection": collection,
        "close_reason": format!("{reason:?}"),
        "unhealthy_loaded_collections": unhealthy.len(),
    });
    response.body = sonic_rs::to_vec(&body)?;
    response
        .headers
        .insert("Content-Type".to_string(), "application/json".to_string());
    Ok(true)
}
