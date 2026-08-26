//! `SegmentTier` — tiers dense vector sidecar files (`.hvec`/`.hvs8`/`.hvtq`)
//! to/from object storage (S3) so segments are durable and rehydratable.
//!
//! Vectors live in mmap sidecar files under the collection data dir (see
//! [`mmap_vectors`](super::mmap_vectors)). This module uploads those files to an
//! object store keyed by `<prefix>/<physical_name>.<ext>` and downloads them back
//! on demand, so a segment can be evicted locally and rehydrated later.
//!
//! `object_store` is async; Helion's call sites here are synchronous. Mirroring
//! [`LsmBackend`](crate::helix_engine::storage_core::backend_lsm), we bridge with
//! a dedicated tokio runtime owned by the struct: every operation runs via
//! `rt.block_on(...)`. Call sites must invoke these from blocking threads, never
//! from inside an async runtime.

#![allow(dead_code)]

use std::path::Path as FsPath;
use std::sync::Arc;

use object_store::aws::AmazonS3Builder;
use object_store::memory::InMemory;
use object_store::path::Path as ObjectPath;
use object_store::{Error as ObjectStoreError, ObjectStore, ObjectStoreExt};
use tokio::runtime::Runtime;

use crate::helix_engine::types::VectorError;

/// Tiers dense vector sidecar files to/from an object store (S3, MinIO, …).
///
/// Object keys are `<prefix>/<physical_name>.<ext>`. The owned [`Runtime`]
/// bridges the async `object_store` API to Helion's synchronous call sites.
pub struct SegmentTier {
    store: Arc<dyn ObjectStore>,
    prefix: String,
    rt: Runtime,
}

impl SegmentTier {
    /// Build a tier over `store`, keying objects under `prefix`.
    pub fn new(store: Arc<dyn ObjectStore>, prefix: &str) -> Result<Self, VectorError> {
        let rt = Runtime::new().map_err(|e| {
            VectorError::VectorCoreError(format!("tokio runtime build failed: {e}"))
        })?;
        Ok(Self {
            store,
            prefix: prefix.to_string(),
            rt,
        })
    }

    /// Build a tier backed by an in-memory object store (for tests / no infra).
    pub fn new_in_memory(prefix: &str) -> Result<Self, VectorError> {
        Self::new(Arc::new(InMemory::new()), prefix)
    }

    /// Build a tier on AWS S3 for `bucket`, keyed under `prefix`. Credentials and
    /// region resolve from the standard AWS environment (env vars, shared config,
    /// or the instance/IRSA role).
    pub fn new_s3(bucket: &str, prefix: &str) -> Result<Self, VectorError> {
        let store = AmazonS3Builder::from_env()
            .with_bucket_name(bucket)
            .build()
            .map_err(|e| VectorError::VectorCoreError(format!("S3 store build failed: {e}")))?;
        Self::new(Arc::new(store), prefix)
    }

    /// Object key for a segment sidecar: `<prefix>/<physical_name>.<ext>`.
    fn object_key(&self, physical_name: &str, ext: &str) -> ObjectPath {
        ObjectPath::from(format!("{}/{}.{}", self.prefix, physical_name, ext))
    }

    /// Upload the local sidecar file at `local_path` to the object store under
    /// the key for `(physical_name, ext)`.
    pub fn upload_segment(
        &self,
        physical_name: &str,
        ext: &str,
        local_path: &FsPath,
    ) -> Result<(), VectorError> {
        let bytes = std::fs::read(local_path).map_err(|e| {
            VectorError::VectorCoreError(format!(
                "read local segment {} failed: {e}",
                local_path.display()
            ))
        })?;
        let key = self.object_key(physical_name, ext);
        self.rt
            .block_on(self.store.put(&key, bytes.into()))
            .map_err(|e| VectorError::VectorCoreError(format!("upload {key} failed: {e}")))?;
        Ok(())
    }

    /// Download the object for `(physical_name, ext)` to `dest_path`.
    ///
    /// Returns `Ok(true)` on success, `Ok(false)` if the object does not exist
    /// (object store reports `NotFound`).
    pub fn download_segment(
        &self,
        physical_name: &str,
        ext: &str,
        dest_path: &FsPath,
    ) -> Result<bool, VectorError> {
        let key = self.object_key(physical_name, ext);
        let get_result = match self.rt.block_on(self.store.get(&key)) {
            Ok(r) => r,
            Err(ObjectStoreError::NotFound { .. }) => return Ok(false),
            Err(e) => {
                return Err(VectorError::VectorCoreError(format!(
                    "download {key} failed: {e}"
                )))
            }
        };
        let bytes = self.rt.block_on(get_result.bytes()).map_err(|e| {
            VectorError::VectorCoreError(format!("read body for {key} failed: {e}"))
        })?;
        std::fs::write(dest_path, &bytes).map_err(|e| {
            VectorError::VectorCoreError(format!(
                "write segment {} failed: {e}",
                dest_path.display()
            ))
        })?;
        Ok(true)
    }

    /// Return whether the object for `(physical_name, ext)` exists in the store.
    pub fn segment_exists(&self, physical_name: &str, ext: &str) -> Result<bool, VectorError> {
        let key = self.object_key(physical_name, ext);
        match self.rt.block_on(self.store.head(&key)) {
            Ok(_) => Ok(true),
            Err(ObjectStoreError::NotFound { .. }) => Ok(false),
            Err(e) => Err(VectorError::VectorCoreError(format!(
                "head {key} failed: {e}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const SAMPLE: &[u8] = b"helion-dense-segment-\x00\x01\x02\xff bytes";

    /// Default test (no infra): round-trips a sidecar file through an in-memory
    /// object store, and checks existence + missing-object handling.
    #[test]
    fn segment_tier_in_memory_round_trip() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("seg.hvec");
        std::fs::write(&src, SAMPLE).unwrap();

        let tier = SegmentTier::new_in_memory("segments").unwrap();

        // Upload, then download to a fresh path and assert byte-identical.
        tier.upload_segment("seg-0001", "hvec", &src).unwrap();
        let dest = dir.path().join("seg.rehydrated.hvec");
        let ok = tier.download_segment("seg-0001", "hvec", &dest).unwrap();
        assert!(ok, "download of an existing segment must return Ok(true)");
        assert_eq!(std::fs::read(&dest).unwrap(), SAMPLE);

        // segment_exists: true for the uploaded one, false for a missing one.
        assert!(tier.segment_exists("seg-0001", "hvec").unwrap());
        assert!(!tier.segment_exists("seg-9999", "hvec").unwrap());
        // Same physical name, different ext is a different (missing) object.
        assert!(!tier.segment_exists("seg-0001", "hvs8").unwrap());

        // Downloading a missing object returns Ok(false) and writes nothing.
        let missing_dest = dir.path().join("missing.hvec");
        let downloaded = tier
            .download_segment("seg-9999", "hvec", &missing_dest)
            .unwrap();
        assert!(
            !downloaded,
            "download of a missing segment must return Ok(false)"
        );
        assert!(
            !missing_dest.exists(),
            "no file should be written on NotFound"
        );
    }

    /// MinIO round-trip. Ignored by default; needs MinIO on :9000 with the
    /// `helion-test` bucket. Run with:
    /// `cargo test -p helixdb segment_tier_minio -- --ignored`.
    #[test]
    #[ignore]
    fn segment_tier_minio_round_trip() {
        let store = AmazonS3Builder::new()
            .with_endpoint("http://localhost:9000")
            .with_region("us-east-1")
            .with_bucket_name("helion-test")
            .with_access_key_id("test")
            .with_secret_access_key("testtest123")
            .with_allow_http(true)
            .build()
            .expect("build MinIO S3 store");

        // Unique prefix per run so retained bucket state doesn't interfere.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let prefix = format!("segment-tier-test/{nanos}");
        let tier = SegmentTier::new(Arc::new(store), &prefix).unwrap();

        let dir = TempDir::new().unwrap();
        let src = dir.path().join("seg.hvec");
        std::fs::write(&src, SAMPLE).unwrap();

        assert!(!tier.segment_exists("seg-0001", "hvec").unwrap());
        tier.upload_segment("seg-0001", "hvec", &src).unwrap();
        assert!(tier.segment_exists("seg-0001", "hvec").unwrap());

        let dest = dir.path().join("seg.rehydrated.hvec");
        let ok = tier.download_segment("seg-0001", "hvec", &dest).unwrap();
        assert!(ok);
        assert_eq!(std::fs::read(&dest).unwrap(), SAMPLE);

        let missing_dest = dir.path().join("missing.hvec");
        assert!(!tier
            .download_segment("seg-9999", "hvec", &missing_dest)
            .unwrap());
    }
}
