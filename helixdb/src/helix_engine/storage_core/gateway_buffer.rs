use crate::{
    helix_engine::types::GraphError,
    protocol::{request::Request, response::Response},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    hash::{Hash, Hasher},
    io::Write,
    path::{Path, PathBuf},
    time::Duration,
};
use twox_hash::XxHash64;

const BUFFER_FILE_EXTENSION: &str = "bin";

#[derive(Clone, Debug)]
pub(crate) struct GatewayBufferConfig {
    pub(crate) dir: PathBuf,
    pub(crate) max_entries: usize,
    pub(crate) max_bytes: u64,
    pub(crate) max_request_bytes: usize,
    pub(crate) replay_interval: Duration,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct GatewayBufferAck {
    pub(crate) id: String,
    pub(crate) duplicate: bool,
}

#[derive(Debug, Default, Eq, PartialEq)]
pub(crate) struct GatewayBufferDepth {
    pub(crate) entries: usize,
    pub(crate) bytes: u64,
}

#[derive(Debug, Default, Eq, PartialEq)]
pub(crate) struct GatewayBufferDrainReport {
    pub(crate) delivered: usize,
    pub(crate) retained: usize,
}

#[derive(Clone)]
pub(crate) struct GatewayDurableBuffer {
    config: GatewayBufferConfig,
}

#[derive(Serialize, Deserialize)]
struct BufferedGatewayRequest {
    id: String,
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

impl GatewayDurableBuffer {
    pub(crate) fn new(config: GatewayBufferConfig) -> Self {
        Self { config }
    }

    pub(crate) fn replay_interval(&self) -> Duration {
        self.config.replay_interval
    }

    pub(crate) fn enqueue(&self, request: &Request) -> Result<GatewayBufferAck, GraphError> {
        if request.body.len() > self.config.max_request_bytes {
            return Err(GraphError::New(format!(
                "request body {} bytes exceeds gateway buffer cap {} bytes",
                request.body.len(),
                self.config.max_request_bytes
            )));
        }

        self.ensure_dir()?;
        let id = buffered_request_id(request);
        let final_path = self.request_path(&id);
        if final_path.exists() {
            return Ok(GatewayBufferAck {
                id,
                duplicate: true,
            });
        }

        let envelope = BufferedGatewayRequest {
            id: id.clone(),
            method: request.method.clone(),
            path: request.path.clone(),
            headers: request.headers.clone(),
            body: request.body.clone(),
        };
        let payload = bincode::serialize(&envelope)?;
        let queued = self.queue_depth()?;
        if queued.entries >= self.config.max_entries
            || queued.bytes.saturating_add(payload.len() as u64) > self.config.max_bytes
        {
            return Err(GraphError::New(format!(
                "gateway buffer is full: entries={}/{} bytes={}/{}",
                queued.entries, self.config.max_entries, queued.bytes, self.config.max_bytes
            )));
        }

        let tmp_path = self.tmp_path(&id);
        let write_result = write_file_durable(&tmp_path, &payload)
            .and_then(|()| match fs::rename(&tmp_path, &final_path) {
                Ok(()) => Ok(()),
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
                Err(err) => Err(GraphError::from(err)),
            })
            .and_then(|()| sync_dir_durable(&self.config.dir));

        match write_result {
            Ok(()) => Ok(GatewayBufferAck {
                id,
                duplicate: false,
            }),
            Err(err) => {
                let _ = fs::remove_file(&tmp_path);
                if final_path.exists() {
                    Ok(GatewayBufferAck {
                        id,
                        duplicate: true,
                    })
                } else {
                    Err(err)
                }
            }
        }
    }

    pub(crate) fn queue_depth(&self) -> Result<GatewayBufferDepth, GraphError> {
        if !self.config.dir.exists() {
            return Ok(GatewayBufferDepth::default());
        }

        let mut depth = GatewayBufferDepth::default();
        for path in self.queued_paths()? {
            let metadata = fs::metadata(path)?;
            depth.entries = depth.entries.saturating_add(1);
            depth.bytes = depth.bytes.saturating_add(metadata.len());
        }
        Ok(depth)
    }

    pub(crate) fn drain_once<F>(&self, mut send: F) -> Result<GatewayBufferDrainReport, GraphError>
    where
        F: FnMut(&Request) -> Result<Response, GraphError>,
    {
        let mut report = GatewayBufferDrainReport::default();
        for path in self.queued_paths()? {
            let payload = fs::read(&path)?;
            let envelope: BufferedGatewayRequest = bincode::deserialize(&payload)?;
            let request = Request {
                method: envelope.method,
                headers: envelope.headers,
                path: envelope.path,
                body: envelope.body,
            };

            match send(&request) {
                Ok(response) if !retryable_replay_status(response.status) => {
                    match fs::remove_file(&path) {
                        Ok(()) => {
                            report.delivered = report.delivered.saturating_add(1);
                            sync_dir_durable(&self.config.dir)?;
                        }
                        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                            report.delivered = report.delivered.saturating_add(1);
                        }
                        Err(err) => return Err(GraphError::from(err)),
                    }
                }
                Ok(_) | Err(_) => {
                    report.retained = report.retained.saturating_add(1);
                }
            }
        }
        Ok(report)
    }

    fn ensure_dir(&self) -> Result<(), GraphError> {
        fs::create_dir_all(&self.config.dir)?;
        sync_dir_durable(&self.config.dir)
    }

    fn queued_paths(&self) -> Result<Vec<PathBuf>, GraphError> {
        if !self.config.dir.exists() {
            return Ok(Vec::new());
        }

        let mut paths = Vec::new();
        for entry in fs::read_dir(&self.config.dir)? {
            let path = entry?.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some(BUFFER_FILE_EXTENSION) {
                paths.push(path);
            }
        }
        paths.sort();
        Ok(paths)
    }

    fn request_path(&self, id: &str) -> PathBuf {
        self.config
            .dir
            .join(format!("{id}.{BUFFER_FILE_EXTENSION}"))
    }

    fn tmp_path(&self, id: &str) -> PathBuf {
        self.config
            .dir
            .join(format!(".{id}.{}.tmp", uuid::Uuid::new_v4()))
    }
}

fn buffered_request_id(request: &Request) -> String {
    let mut h1 = XxHash64::with_seed(0x4845_4c49_585f_4757);
    let mut h2 = XxHash64::with_seed(0x4345_4757_4255_4632);
    request.method.hash(&mut h1);
    request.path.hash(&mut h1);
    request.method.hash(&mut h2);
    request.path.hash(&mut h2);

    if let Some(key) = request
        .headers
        .get("idempotency-key")
        .or_else(|| request.headers.get("Idempotency-Key"))
    {
        key.hash(&mut h1);
        key.hash(&mut h2);
    } else {
        request.body.hash(&mut h1);
        request.body.hash(&mut h2);
    }

    format!("{:016x}{:016x}", h1.finish(), h2.finish())
}

fn retryable_replay_status(status: u16) -> bool {
    matches!(status, 429 | 502 | 503 | 504) || status >= 500
}

fn write_file_durable(path: &Path, payload: &[u8]) -> Result<(), GraphError> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(payload)?;
    file.sync_all()?;
    Ok(())
}

fn sync_dir_durable(dir: &Path) -> Result<(), GraphError> {
    match File::open(dir) {
        Ok(handle) => {
            let _ = handle.sync_all();
            Ok(())
        }
        Err(_) => Ok(()),
    }
}

#[cfg(test)]
#[path = "gateway_buffer_tests.rs"]
mod tests;
