use super::*;
use crate::protocol::{request::Request, response::Response};
use std::{collections::HashMap, time::Duration};
use tempfile::TempDir;

fn config(dir: &TempDir) -> GatewayBufferConfig {
    GatewayBufferConfig {
        dir: dir.path().to_path_buf(),
        max_entries: 8,
        max_bytes: 1024 * 1024,
        max_request_bytes: 1024,
        replay_interval: Duration::from_millis(10),
    }
}

fn mutation_request(body: &[u8]) -> Request {
    Request {
        method: "PUT".to_string(),
        headers: HashMap::from([("content-type".to_string(), "application/json".to_string())]),
        path: "/collections/repo/points".to_string(),
        body: body.to_vec(),
    }
}

#[test]
fn durable_buffer_coalesces_duplicate_requests_by_hash() {
    let dir = TempDir::new().unwrap();
    let buffer = GatewayDurableBuffer::new(config(&dir));
    let request = mutation_request(br#"{"points":[{"id":1}]}"#);

    let first = buffer.enqueue(&request).unwrap();
    let second = buffer.enqueue(&request).unwrap();

    assert_eq!(first.id, second.id);
    assert!(!first.duplicate);
    assert!(second.duplicate);
    assert_eq!(buffer.queue_depth().unwrap().entries, 1);
}

#[test]
fn durable_buffer_rejects_requests_over_configured_body_cap() {
    let dir = TempDir::new().unwrap();
    let mut cfg = config(&dir);
    cfg.max_request_bytes = 4;
    let buffer = GatewayDurableBuffer::new(cfg);

    let err = buffer.enqueue(&mutation_request(b"too-large")).unwrap_err();

    assert!(err.to_string().contains("exceeds gateway buffer cap"));
    assert_eq!(buffer.queue_depth().unwrap().entries, 0);
}

#[test]
fn durable_buffer_rejects_when_entry_cap_is_full() {
    let dir = TempDir::new().unwrap();
    let mut cfg = config(&dir);
    cfg.max_entries = 1;
    let buffer = GatewayDurableBuffer::new(cfg);

    buffer.enqueue(&mutation_request(b"one")).unwrap();
    let err = buffer.enqueue(&mutation_request(b"two")).unwrap_err();

    assert!(err.to_string().contains("gateway buffer is full"));
    assert_eq!(buffer.queue_depth().unwrap().entries, 1);
}

#[test]
fn durable_buffer_drops_file_after_successful_replay() {
    let dir = TempDir::new().unwrap();
    let buffer = GatewayDurableBuffer::new(config(&dir));
    buffer.enqueue(&mutation_request(b"one")).unwrap();

    let report = buffer
        .drain_once(|request| {
            assert_eq!(request.method, "PUT");
            let mut response = Response::new();
            response.status = 200;
            Ok(response)
        })
        .unwrap();

    assert_eq!(report.delivered, 1);
    assert_eq!(report.retained, 0);
    assert_eq!(buffer.queue_depth().unwrap().entries, 0);
}

#[test]
fn durable_buffer_replays_after_buffer_object_recreation() {
    let dir = TempDir::new().unwrap();
    let first_buffer = GatewayDurableBuffer::new(config(&dir));
    first_buffer.enqueue(&mutation_request(b"one")).unwrap();
    drop(first_buffer);

    let second_buffer = GatewayDurableBuffer::new(config(&dir));
    let report = second_buffer
        .drain_once(|request| {
            assert_eq!(request.body, b"one");
            let mut response = Response::new();
            response.status = 200;
            Ok(response)
        })
        .unwrap();

    assert_eq!(report.delivered, 1);
    assert_eq!(second_buffer.queue_depth().unwrap().entries, 0);
}

#[test]
fn durable_buffer_retains_file_after_retryable_replay_status() {
    let dir = TempDir::new().unwrap();
    let buffer = GatewayDurableBuffer::new(config(&dir));
    buffer.enqueue(&mutation_request(b"one")).unwrap();

    let report = buffer
        .drain_once(|_| {
            let mut response = Response::new();
            response.status = 503;
            Ok(response)
        })
        .unwrap();

    assert_eq!(report.delivered, 0);
    assert_eq!(report.retained, 1);
    assert_eq!(buffer.queue_depth().unwrap().entries, 1);
}
