use crate::{
    helix_engine::storage_core::backend::BackendError, helixc::parser::parser_methods::ParserError,
    protocol::traversal_value::TraversalValueError,
};
use core::fmt;
use heed3::Error as HeedError;
use sonic_rs::Error as SonicError;
use std::{net::AddrParseError, str::Utf8Error, string::FromUtf8Error};

#[derive(Debug)]
pub enum GraphError {
    Io(std::io::Error),
    GraphConnectionError(String, std::io::Error),
    StorageConnectionError(String, std::io::Error),
    StorageError(String),
    TraversalError(String),
    ConversionError(String),
    EdgeNotFound,
    NodeNotFound,
    LabelNotFound,
    VectorError(String),
    /// LMDB map is full — the caller should resize and retry.
    MapFull,
    /// LMDB reported a fatal collection-local storage problem. The current
    /// transaction must abort; callers should quarantine/rebuild the collection.
    FatalCollectionStorage {
        code: &'static str,
        message: String,
    },
    /// A collection needs LMDB resize runway but the resize gate is busy.
    ///
    /// This is not a storage corruption/error condition. Callers should
    /// return retryable per-collection backpressure instead of queueing a
    /// long pending resize writer that blocks unrelated reads.
    ResizeBackpressure(String),
    /// LMDB env for this path is still open elsewhere in the process —
    /// the caller should retry after a short backoff (another thread/task
    /// holds an `Arc<HelixGraphStorage>` that hasn't dropped yet).
    EnvAlreadyOpen,
    Default,
    New(String),
    Empty,
    MultipleNodesWithSameId,
    MultipleEdgesWithSameId,
    InvalidNode,
    ConfigFileNotFound,
    SliceLengthError,
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::Io(e) => write!(f, "IO error: {}", e),
            GraphError::StorageConnectionError(msg, e) => {
                write!(f, "Error: {}", format!("{} {}", msg, e))
            }
            GraphError::GraphConnectionError(msg, e) => {
                write!(f, "Error: {}", format!("{} {}", msg, e))
            }
            GraphError::TraversalError(msg) => write!(f, "Traversal error: {}", msg),
            GraphError::StorageError(msg) => write!(f, "Storage error: {}", msg),
            GraphError::ConversionError(msg) => write!(f, "Conversion error: {}", msg),
            GraphError::EdgeNotFound => write!(f, "Edge not found"),
            GraphError::NodeNotFound => write!(f, "Node not found"),
            GraphError::LabelNotFound => write!(f, "Label not found"),
            GraphError::New(msg) => write!(f, "Graph error: {}", msg),
            GraphError::Default => write!(f, "Graph error"),
            GraphError::Empty => write!(f, "No Error"),
            GraphError::MultipleNodesWithSameId => write!(f, "Multiple nodes with same id"),
            GraphError::MultipleEdgesWithSameId => write!(f, "Multiple edges with same id"),
            GraphError::InvalidNode => write!(f, "Invalid node"),
            GraphError::ConfigFileNotFound => write!(f, "Config file not found"),
            GraphError::SliceLengthError => write!(f, "Slice length error"),
            GraphError::MapFull => write!(f, "LMDB map full — resize needed"),
            GraphError::FatalCollectionStorage { code, message } => {
                write!(f, "Fatal collection storage error ({}): {}", code, message)
            }
            GraphError::ResizeBackpressure(msg) => write!(f, "Resize backpressure: {}", msg),
            GraphError::EnvAlreadyOpen => write!(
                f,
                "LMDB environment already open in this process — retry after close"
            ),
            GraphError::VectorError(msg) => write!(f, "Vector error: {}", msg),
        }
    }
}

// impl From<rocksdb::Error> for GraphError {
//     fn from(error: rocksdb::Error) -> Self {
//         GraphError::New(error.into_string())
//     }
// }

fn heed_error_is_retryable_lmdb_invalid_argument(error: &HeedError) -> bool {
    matches!(error, HeedError::Io(io_error) if io_error.raw_os_error() == Some(22))
}

fn log_unclassified_heed_error(context: &'static str, error: &HeedError) {
    match error {
        HeedError::Io(io_error) if heed_error_is_retryable_lmdb_invalid_argument(error) => {
            tracing::warn!(
                context,
                raw_os_error = ?io_error.raw_os_error(),
                error_kind = ?io_error.kind(),
                error_message = %io_error,
                "retryable heed IO invalid argument"
            );
        }
        HeedError::Io(io_error) => {
            tracing::error!(
                context,
                raw_os_error = ?io_error.raw_os_error(),
                error_kind = ?io_error.kind(),
                error_message = %io_error,
                "unclassified heed IO error"
            );
        }
        HeedError::Mdb(mdb_error) => {
            tracing::error!(
                context,
                mdb_error = ?mdb_error,
                error_message = %mdb_error,
                "unclassified heed MDB error"
            );
        }
        HeedError::Encoding(encoding_error) => {
            tracing::error!(
                context,
                error_message = %encoding_error,
                "heed encoding error"
            );
        }
        HeedError::Decoding(decoding_error) => {
            tracing::error!(
                context,
                error_message = %decoding_error,
                "heed decoding error"
            );
        }
        HeedError::EnvAlreadyOpened => {
            tracing::error!(context, "heed environment already open");
        }
    }
}

fn fatal_collection_storage_from_mdb_error(
    error: heed3::MdbError,
) -> Option<(&'static str, &'static str)> {
    match error {
        heed3::MdbError::Problem => Some((
            "mdb_problem",
            "LMDB MDB_PROBLEM: txn should abort; collection requires quarantine/rebuild",
        )),
        heed3::MdbError::PageNotFound => Some((
            "mdb_page_not_found",
            "LMDB MDB_PAGE_NOTFOUND: requested page not found; collection requires quarantine/rebuild",
        )),
        heed3::MdbError::Corrupted => Some((
            "mdb_corrupted",
            "LMDB MDB_CORRUPTED: located page was wrong type; collection requires quarantine/rebuild",
        )),
        heed3::MdbError::Panic => Some((
            "mdb_panic",
            "LMDB MDB_PANIC: collection environment had a fatal error; collection requires quarantine/rebuild",
        )),
        _ => None,
    }
}

/// A SlateDB/object-store `Io` message is treated as fatal (quarantine the
/// collection) only when it clearly signals unrecoverable on-disk/object
/// damage. Transient I/O — timeouts, throttling, connection resets, broken
/// pipes, object-store 5xx — is intentionally excluded so a healthy collection
/// is never quarantined on a network blip (those should be retried). Mirrors the
/// conservative substring matching `backend_lsm::map_slatedb_err` already uses to
/// separate fencing from generic I/O.
fn backend_io_is_fatal(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    if lsm_object_reference_is_missing(&lower) {
        return true;
    }
    const FATAL_IO_SIGNALS: &[&str] = &[
        "corrupt",
        "checksum",
        "crc mismatch",
        "malformed",
        "invalid sst",
        "invalid manifest",
        "invalid block",
        "unexpected eof",
        "decode error",
    ];
    FATAL_IO_SIGNALS.iter().any(|sig| lower.contains(sig))
}

fn lsm_object_reference_is_missing(lower: &str) -> bool {
    lower.contains("nosuchkey")
        || lower.contains("manifestmissing")
        || lower.contains("manifest missing")
        || (lower.contains("object at location")
            && (lower.contains(".sst not found")
                || lower.contains("/manifest")
                || lower.contains("manifest/")))
        || (lower.contains("compacted/") && lower.contains(".sst") && lower.contains("not found"))
}

/// Classify a fatal LSM/SlateDB backend error into a quarantine-worthy
/// `(code, message)` pair, the backend-routed analogue of
/// `fatal_collection_storage_from_mdb_error` for the heed path. Optimistic
/// write conflicts (`Conflict`) are retryable at the request/handle layer: a
/// SlateDB CAS loss or fenced stale client is not evidence that collection bytes
/// are corrupt. Explicit `Corruption` is always fatal; `Io` is fatal only for
/// clearly-unrecoverable cases (see [`backend_io_is_fatal`]). `NotFound`,
/// `Unsupported`, and `Conflict` are never fatal.
fn fatal_collection_storage_from_backend_error(
    error: &BackendError,
) -> Option<(&'static str, String)> {
    match error {
        BackendError::Corruption(msg) => Some((
            "lsm_corrupted",
            format!("LSM corruption: {msg}; collection requires quarantine/rebuild"),
        )),
        BackendError::Io(msg) if backend_io_is_fatal(msg) => Some((
            "lsm_io",
            format!("LSM fatal I/O: {msg}; collection requires quarantine/rebuild"),
        )),
        _ => None,
    }
}

/// Convert a [`BackendError`] to a [`GraphError`], promoting only fatal LSM
/// failures (corruption and clearly-fatal I/O) to
/// [`GraphError::FatalCollectionStorage`] so callers quarantine the collection,
/// while retryable/non-fatal errors keep the previous `GraphError::New(..)`
/// shape.
pub(crate) fn graph_error_from_backend_error(error: BackendError) -> GraphError {
    if let Some((code, message)) = fatal_collection_storage_from_backend_error(&error) {
        return GraphError::FatalCollectionStorage { code, message };
    }
    GraphError::New(error.to_string())
}

impl From<HeedError> for GraphError {
    fn from(error: HeedError) -> Self {
        if matches!(error, HeedError::Mdb(heed3::MdbError::MapFull)) {
            return GraphError::MapFull;
        }
        if let HeedError::Mdb(mdb_error) = error {
            if let Some((code, message)) = fatal_collection_storage_from_mdb_error(mdb_error) {
                return GraphError::FatalCollectionStorage {
                    code,
                    message: message.into(),
                };
            }
        }
        if matches!(error, HeedError::EnvAlreadyOpened) {
            return GraphError::EnvAlreadyOpen;
        }
        log_unclassified_heed_error("graph", &error);
        GraphError::StorageError(error.to_string())
    }
}

impl From<std::io::Error> for GraphError {
    fn from(error: std::io::Error) -> Self {
        GraphError::Io(error)
    }
}

impl From<AddrParseError> for GraphError {
    fn from(error: AddrParseError) -> Self {
        GraphError::ConversionError(format!("AddrParseError: {}", error.to_string()))
    }
}

impl From<SonicError> for GraphError {
    fn from(error: SonicError) -> Self {
        GraphError::ConversionError(format!("sonic error: {}", error.to_string()))
    }
}

impl From<FromUtf8Error> for GraphError {
    fn from(error: FromUtf8Error) -> Self {
        GraphError::ConversionError(format!("FromUtf8Error: {}", error.to_string()))
    }
}

impl From<&'static str> for GraphError {
    fn from(error: &'static str) -> Self {
        GraphError::New(error.to_string())
    }
}

impl From<String> for GraphError {
    fn from(error: String) -> Self {
        GraphError::New(error.to_string())
    }
}

impl From<bincode::Error> for GraphError {
    fn from(error: bincode::Error) -> Self {
        GraphError::ConversionError(format!("bincode error: {}", error.to_string()))
    }
}

impl From<ParserError> for GraphError {
    fn from(error: ParserError) -> Self {
        GraphError::ConversionError(format!("ParserError: {}", error.to_string()))
    }
}

impl From<Utf8Error> for GraphError {
    fn from(error: Utf8Error) -> Self {
        GraphError::ConversionError(format!("Utf8Error: {}", error.to_string()))
    }
}

impl From<uuid::Error> for GraphError {
    fn from(error: uuid::Error) -> Self {
        GraphError::ConversionError(format!("uuid error: {}", error.to_string()))
    }
}

impl From<TraversalValueError> for GraphError {
    fn from(error: TraversalValueError) -> Self {
        GraphError::ConversionError(format!("TraversalValueError: {}", error.to_string()))
    }
}

impl From<VectorError> for GraphError {
    fn from(error: VectorError) -> Self {
        if matches!(error, VectorError::MapFull) {
            return GraphError::MapFull;
        }
        if let VectorError::FatalCollectionStorage { code, message } = error {
            return GraphError::FatalCollectionStorage { code, message };
        }
        GraphError::VectorError(format!("VectorError: {}", error.to_string()))
    }
}

impl GraphError {
    pub fn is_fatal_collection_storage(&self) -> bool {
        matches!(self, GraphError::FatalCollectionStorage { .. })
    }

    pub fn should_quarantine_collection(&self) -> bool {
        if self.is_fatal_collection_storage() {
            return true;
        }
        match self {
            GraphError::New(message)
            | GraphError::StorageError(message)
            | GraphError::VectorError(message) => {
                lsm_object_reference_is_missing(&message.to_ascii_lowercase())
            }
            _ => false,
        }
    }

    pub fn is_retryable_lmdb_invalid_argument(&self) -> bool {
        match self {
            GraphError::VectorError(msg) => {
                msg.contains("heed error")
                    && msg.contains("Invalid argument")
                    && msg.contains("os error 22")
            }
            GraphError::StorageError(msg) => {
                msg.contains("Invalid argument") && msg.contains("os error 22")
            }
            GraphError::Io(err) => err.raw_os_error() == Some(22),
            _ => false,
        }
    }

    pub fn fatal_collection_response_body(&self) -> Option<Vec<u8>> {
        match self {
            GraphError::FatalCollectionStorage { code, message } => {
                sonic_rs::to_vec(&sonic_rs::json!({
                    "status": "error",
                    "code": code,
                    "fatal": true,
                    "scope": "collection",
                    "action": "quarantine_rebuild",
                    "message": message,
                }))
                .ok()
            }
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum VectorError {
    VectorNotFound(String),
    InvalidVectorLength,
    InvalidVectorData,
    InvalidVectorId,
    InvalidVectorLevel,
    InvalidEntryPoint,
    EntryPointNotFound,
    InvalidVectorCoreConfig,
    ConversionError(String),
    VectorCoreError(String),
    /// LMDB map is full — propagates to GraphError::MapFull for retry.
    MapFull,
    FatalCollectionStorage {
        code: &'static str,
        message: String,
    },
}

impl fmt::Display for VectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorError::VectorNotFound(id) => write!(f, "Vector not found: {}", id),
            VectorError::InvalidVectorLength => write!(f, "Invalid vector length"),
            VectorError::InvalidVectorData => write!(f, "Invalid vector data"),
            VectorError::InvalidVectorId => write!(f, "Invalid vector id"),
            VectorError::InvalidVectorLevel => write!(f, "Invalid vector level"),
            VectorError::InvalidEntryPoint => write!(f, "Invalid entry point"),
            VectorError::EntryPointNotFound => write!(f, "Entry point not found"),
            VectorError::InvalidVectorCoreConfig => write!(f, "Invalid vector core config"),
            VectorError::ConversionError(msg) => write!(f, "Conversion error: {}", msg),
            VectorError::VectorCoreError(msg) => write!(f, "Vector core error: {}", msg),
            VectorError::MapFull => write!(f, "LMDB map full — resize needed"),
            VectorError::FatalCollectionStorage { code, message } => {
                write!(f, "Fatal collection storage error ({}): {}", code, message)
            }
        }
    }
}

impl From<HeedError> for VectorError {
    fn from(error: HeedError) -> Self {
        if matches!(error, HeedError::Mdb(heed3::MdbError::MapFull)) {
            return VectorError::MapFull;
        }
        if let HeedError::Mdb(mdb_error) = error {
            if let Some((code, message)) = fatal_collection_storage_from_mdb_error(mdb_error) {
                return VectorError::FatalCollectionStorage {
                    code,
                    message: message.into(),
                };
            }
        }
        log_unclassified_heed_error("vector", &error);
        VectorError::VectorCoreError(format!("heed error: {}", error.to_string()))
    }
}

impl From<FromUtf8Error> for VectorError {
    fn from(error: FromUtf8Error) -> Self {
        VectorError::ConversionError(format!("FromUtf8Error: {}", error.to_string()))
    }
}

impl From<Utf8Error> for VectorError {
    fn from(error: Utf8Error) -> Self {
        VectorError::ConversionError(format!("Utf8Error: {}", error.to_string()))
    }
}

impl From<SonicError> for VectorError {
    fn from(error: SonicError) -> Self {
        VectorError::ConversionError(format!("SonicError: {}", error.to_string()))
    }
}

impl std::error::Error for GraphError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GraphError::Io(e) => Some(e),
            GraphError::GraphConnectionError(_, e) | GraphError::StorageConnectionError(_, e) => {
                Some(e)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{GraphError, HeedError, VectorError};

    #[test]
    fn heed_mdb_problem_maps_to_fatal_collection_storage() {
        let err = GraphError::from(HeedError::Mdb(heed3::MdbError::Problem));

        assert!(err.is_fatal_collection_storage());
        let body = String::from_utf8(err.fatal_collection_response_body().unwrap()).unwrap();
        assert!(body.contains("\"code\":\"mdb_problem\""), "body: {body}");
        assert!(body.contains("\"fatal\":true"), "body: {body}");
        assert!(
            body.contains("\"action\":\"quarantine_rebuild\""),
            "body: {body}"
        );
    }

    #[test]
    fn heed_mdb_page_not_found_maps_to_fatal_collection_storage() {
        let err = GraphError::from(HeedError::Mdb(heed3::MdbError::PageNotFound));

        assert!(err.is_fatal_collection_storage());
        let body = String::from_utf8(err.fatal_collection_response_body().unwrap()).unwrap();
        assert!(
            body.contains("\"code\":\"mdb_page_not_found\""),
            "body: {body}"
        );
        assert!(
            body.contains("MDB_PAGE_NOTFOUND"),
            "body should preserve the LMDB class: {body}"
        );
    }

    #[test]
    fn vector_mdb_problem_promotes_to_graph_fatal_collection_storage() {
        let vector = VectorError::from(HeedError::Mdb(heed3::MdbError::Problem));
        let graph = GraphError::from(vector);

        assert!(graph.is_fatal_collection_storage());
    }

    #[test]
    fn vector_mdb_page_not_found_promotes_to_graph_fatal_collection_storage() {
        let vector = VectorError::from(HeedError::Mdb(heed3::MdbError::PageNotFound));
        let graph = GraphError::from(vector);

        assert!(graph.is_fatal_collection_storage());
        let body = String::from_utf8(graph.fatal_collection_response_body().unwrap()).unwrap();
        assert!(
            body.contains("\"code\":\"mdb_page_not_found\""),
            "body: {body}"
        );
    }

    #[test]
    fn backend_conflict_maps_to_retryable_graph_error() {
        use super::{graph_error_from_backend_error, BackendError};

        let err = graph_error_from_backend_error(BackendError::Conflict(
            "detected newer DB client".to_string(),
        ));

        assert!(
            !err.is_fatal_collection_storage(),
            "SlateDB fencing/CAS conflicts are retryable, not collection corruption: {err}"
        );
        assert!(
            err.fatal_collection_response_body().is_none(),
            "retryable conflicts must not produce quarantine response bodies"
        );
        assert!(
            err.to_string().contains("detected newer DB client"),
            "retryable error should preserve the SlateDB message: {err}"
        );
    }

    #[test]
    fn backend_corruption_maps_to_fatal_collection_storage() {
        use super::{graph_error_from_backend_error, BackendError};

        let err = graph_error_from_backend_error(BackendError::Corruption("bad block".to_string()));

        assert!(err.is_fatal_collection_storage());
    }

    #[test]
    fn backend_fatal_io_maps_to_fatal_collection_storage() {
        use super::{graph_error_from_backend_error, BackendError};

        let err = graph_error_from_backend_error(BackendError::Io(
            "sst checksum mismatch on read".to_string(),
        ));

        assert!(err.is_fatal_collection_storage());
        let body = String::from_utf8(err.fatal_collection_response_body().unwrap()).unwrap();
        assert!(body.contains("\"code\":\"lsm_io\""), "body: {body}");
    }

    #[test]
    fn missing_referenced_lsm_sst_maps_to_collection_quarantine() {
        use super::{graph_error_from_backend_error, BackendError};

        let message = "object store error (Object at location prod/repo/compacted/01ABC.sst not found: Server returned non-2xx status code: 404 Not Found: <Error><Code>NoSuchKey</Code></Error>)";
        let err = graph_error_from_backend_error(BackendError::Io(message.to_string()));

        assert!(err.is_fatal_collection_storage());
        assert!(err.should_quarantine_collection());
    }

    #[test]
    fn wrapped_missing_referenced_lsm_sst_marks_handler_error_quarantineable() {
        let err = GraphError::New("Storage error: io error: Data error: object store error (Object at location prod/repo/compacted/01ABC.sst not found: <Error><Code>NoSuchKey</Code></Error>)".to_string());

        assert!(err.should_quarantine_collection());
        assert!(!err.is_fatal_collection_storage());
    }

    #[test]
    fn backend_transient_io_and_notfound_are_not_fatal() {
        use super::{graph_error_from_backend_error, BackendError};

        // Transient object-store/network I/O must NOT quarantine the collection.
        assert!(!graph_error_from_backend_error(BackendError::Io(
            "connection reset by peer (broken pipe)".to_string()
        ))
        .is_fatal_collection_storage());
        assert!(
            !graph_error_from_backend_error(BackendError::NotFound).is_fatal_collection_storage()
        );
        assert!(!graph_error_from_backend_error(BackendError::Unsupported(
            "blocking thread".to_string()
        ))
        .is_fatal_collection_storage());
    }

    #[test]
    fn retryable_lmdb_invalid_argument_matches_observed_heed_error() {
        let graph = GraphError::VectorError(
            "VectorError: Vector core error: heed error: Invalid argument (os error 22)"
                .to_string(),
        );

        assert!(graph.is_retryable_lmdb_invalid_argument());
        assert!(
            GraphError::StorageError("Invalid argument (os error 22)".to_string())
                .is_retryable_lmdb_invalid_argument()
        );
        assert!(GraphError::Io(std::io::Error::from_raw_os_error(22))
            .is_retryable_lmdb_invalid_argument());
        assert!(!GraphError::VectorError("other vector error".to_string())
            .is_retryable_lmdb_invalid_argument());
    }
}

impl std::error::Error for VectorError {}

impl From<bincode::Error> for VectorError {
    fn from(error: bincode::Error) -> Self {
        VectorError::ConversionError(format!("bincode error: {}", error.to_string()))
    }
}
