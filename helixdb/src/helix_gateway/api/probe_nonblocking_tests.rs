//! `/metrics` and `/readyz` must answer while the writer's object-store catalog
//! is slow and a point mutation holds its collection gate. Both handlers used to
//! call `list_collections()` inline — one S3 LIST plus a drop-tombstone probe
//! per collection — so a wedged writer stopped being scrapeable and failed its
//! readiness probe.

use super::tests::{make_input, setup_with_config};
use crate::helix_engine::graph_core::config::Config;
use crate::helix_engine::storage_core::backend::{LsmRole, LsmStorage, StorageBackendConfig};
use crate::helix_engine::storage_core::backend_any::set_lsm_test_object_store;
use crate::helix_gateway::api::{health, metrics};
use crate::protocol::response::Response;
use futures::stream::BoxStream;
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Delegates to an in-memory store but holds every catalog listing for
/// `stall`, standing in for a saturated S3 path on the writer.
#[derive(Debug)]
struct StalledCatalogStore {
    inner: Arc<dyn ObjectStore>,
    stall: Duration,
}

impl std::fmt::Display for StalledCatalogStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StalledCatalogStore({})", self.inner)
    }
}

type Boxed<'a, T> = Pin<Box<dyn Future<Output = object_store::Result<T>> + Send + 'a>>;

// `ObjectStore` is an `async_trait` trait and helixdb does not depend on the
// macro, so the boxed signatures are written out by hand.
impl ObjectStore for StalledCatalogStore {
    fn put_opts<'a, 'b, 'c>(
        &'a self,
        location: &'b Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Boxed<'c, PutResult>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        self.inner.put_opts(location, payload, opts)
    }

    fn put_multipart_opts<'a, 'b, 'c>(
        &'a self,
        location: &'b Path,
        opts: PutMultipartOptions,
    ) -> Boxed<'c, Box<dyn MultipartUpload>>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        self.inner.put_multipart_opts(location, opts)
    }

    fn get_opts<'a, 'b, 'c>(
        &'a self,
        location: &'b Path,
        options: GetOptions,
    ) -> Boxed<'c, GetResult>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        self.inner.get_opts(location, options)
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_delimiter<'a, 'b, 'c>(&'a self, prefix: Option<&'b Path>) -> Boxed<'c, ListResult>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        let inner = Arc::clone(&self.inner);
        let prefix = prefix.cloned();
        let stall = self.stall;
        Box::pin(async move {
            tokio::time::sleep(stall).await;
            inner.list_with_delimiter(prefix.as_ref()).await
        })
    }

    fn copy_opts<'a, 'b, 'c, 'd>(
        &'a self,
        from: &'b Path,
        to: &'c Path,
        options: CopyOptions,
    ) -> Boxed<'d, ()>
    where
        'a: 'd,
        'b: 'd,
        'c: 'd,
        Self: 'd,
    {
        self.inner.copy_opts(from, to, options)
    }
}

const STALL: Duration = Duration::from_secs(4);
const PROMPT: Duration = Duration::from_secs(1);

#[test]
fn metrics_and_readiness_answer_while_catalog_listing_stalls_and_gate_is_held() {
    let fast: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let _fast = set_lsm_test_object_store(Arc::clone(&fast));
    let ctx = setup_with_config(Config::default().with_storage_backend(
        StorageBackendConfig::Lsm {
            storage: LsmStorage::ObjectStore,
            role: LsmRole::Writer,
        },
    ));

    // From here on every catalog listing takes STALL, and a mutation holds the
    // collection's point gate as a batch delete would.
    let _stalled = set_lsm_test_object_store(Arc::new(StalledCatalogStore {
        inner: fast,
        stall: STALL,
    }));
    let gate = ctx.collections.point_mutation_gate("repo");
    let _held = gate.lock().unwrap();

    let started = Instant::now();
    let mut response = Response::new();
    let input = make_input(&ctx, "GET", "/metrics", Vec::new(), HashMap::new());
    metrics::handle_metrics(&input, &mut response).unwrap();
    assert_eq!(response.status, 200);
    assert!(
        response.body.starts_with(b"# ") || response.body.is_empty(),
        "metrics body must be Prometheus text"
    );
    let metrics_elapsed = started.elapsed();

    let started = Instant::now();
    let mut response = Response::new();
    let input = make_input(&ctx, "GET", "/readyz", Vec::new(), HashMap::new());
    health::handle_ready(&input, &mut response).unwrap();
    assert_eq!(response.status, 200);
    let ready_elapsed = started.elapsed();

    assert!(
        metrics_elapsed < PROMPT && ready_elapsed < PROMPT,
        "probes waited on the catalog listing: /metrics {metrics_elapsed:?}, /readyz {ready_elapsed:?}"
    );
}
