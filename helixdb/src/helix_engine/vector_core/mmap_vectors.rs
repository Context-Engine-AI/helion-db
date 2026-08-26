//! Flat mmap'd vector storage for O(1) reads on the search hot path.
//!
//! Vectors are stored contiguously in a memory-mapped file:
//!   [header: 16 bytes][vector_0: dim×4 bytes][vector_1: dim×4 bytes]...
//!
//! Header layout:
//!   magic: b"HVEC" (4 bytes)
//!   dim:   u32 LE  (4 bytes)
//!   count: u64 LE  (8 bytes)
//!
//! Each vector is `dim` f32 values in native endian.
//! Read is pure pointer arithmetic: `&mmap[HEADER + ordinal * dim * 4]`.
//!
//! The OS page cache manages residency — no extra RAM budget beyond what
//! the kernel would cache anyway. Same memory model as Qdrant's mmap storage.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use memmap2::{Advice, Mmap, UncheckedAdvice};

use crate::helix_engine::{
    types::VectorError,
    vector_core::spindle::{
        decode_vector, encode_vector, score_encoded, ApproximateInnerProduct, PreparedSpindleQuery,
        SpindleConfig, SpindleMode, MSE_LEVELS,
    },
};

const MAGIC: &[u8; 4] = b"HVEC";
const HEADER_SIZE: usize = 16; // 4 magic + 4 dim + 8 count

fn advise_random_access(mmap: &Mmap) {
    let _ = mmap.advise(Advice::Random);
    #[cfg(target_os = "linux")]
    if !mmap_hugepage_enabled() {
        let _ = mmap.advise(Advice::NoHugePage);
    }
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn mmap_hugepage_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("HELIX_MMAP_HUGEPAGE")
            .ok()
            .map(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
            .unwrap_or(false)
    })
}

fn mmap_willneed_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("HELIX_MMAP_WILLNEED")
            .ok()
            .map(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
            .unwrap_or(false)
    })
}

fn advise_sequential_scan(mmap: &Mmap) {
    let _ = mmap.advise(Advice::Sequential);
    if mmap_willneed_enabled() {
        let _ = mmap.advise(Advice::WillNeed);
    }
    #[cfg(target_os = "linux")]
    if mmap_hugepage_enabled() {
        let _ = mmap.advise(Advice::HugePage);
    }
}

#[cfg(target_os = "linux")]
fn advise_file_access(file: &File, advice: libc::c_int) {
    use std::os::fd::AsRawFd;

    let _ = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, advice) };
}

#[cfg(target_os = "linux")]
fn advise_file_random_access(file: &File) {
    advise_file_access(file, libc::POSIX_FADV_RANDOM);
}

#[cfg(not(target_os = "linux"))]
fn advise_file_random_access(_file: &File) {}

#[cfg(target_os = "linux")]
fn advise_file_sequential_scan(file: &File) {
    advise_file_access(file, libc::POSIX_FADV_SEQUENTIAL);
    if mmap_willneed_enabled() {
        advise_file_access(file, libc::POSIX_FADV_WILLNEED);
    }
}

#[cfg(not(target_os = "linux"))]
fn advise_file_sequential_scan(_file: &File) {}

#[cfg(target_os = "linux")]
fn advise_file_done_with_pages(file: &File) {
    advise_file_access(file, libc::POSIX_FADV_DONTNEED);
}

#[cfg(not(target_os = "linux"))]
fn advise_file_done_with_pages(_file: &File) {}

/// Open a brand-new sidecar file at a possibly reused path.
///
/// Unlinks any existing file first instead of truncating in place: segment
/// physical names are reused, and `truncate(true)` on a path whose old inode
/// is still mmapped by an in-flight core invalidates that mapping's pages
/// (SIGBUS/SIGSEGV on the next dereference). Unlinking leaves the old inode
/// intact for existing mmap holders while the new file gets a fresh inode.
fn create_fresh_file(path: &Path) -> io::Result<File> {
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)
}

struct MmapScanAdvice<'a> {
    mmap: Option<&'a Mmap>,
    file: &'a File,
}

impl Drop for MmapScanAdvice<'_> {
    fn drop(&mut self) {
        if let Some(mmap) = self.mmap {
            advise_random_access(mmap);
        }
        advise_file_random_access(self.file);
    }
}

/// Flat mmap'd vector file. Supports append (during ingest) and O(1) reads
/// (during search). Thread-safe for concurrent reads; writes require external
/// synchronization (guaranteed by LMDB's single-writer model).
pub struct MmapVectorStore {
    path: PathBuf,
    dim: usize,
    count: AtomicU64,
    /// Read-only mmap for search. Remapped after writes.
    mmap: Option<Mmap>,
    /// The backing file handle.
    file: File,
    /// Set on append; cleared after a successful flush. flush() short-
    /// circuits when false so per-batch flush passes that touch only
    /// N of M cores in a collection don't pay N fsync calls on the
    /// (M-N) untouched cores. flush_mmap_stores fans out to every
    /// core regardless of which got writes; this gate makes the
    /// no-write case a true no-op.
    dirty: AtomicBool,
}

impl MmapVectorStore {
    fn data_end_offset(&self, count: u64) -> u64 {
        HEADER_SIZE as u64 + count * (self.dim as u64) * 4
    }

    fn position_write_cursor(&mut self, count: u64) -> Result<(), VectorError> {
        self.file
            .seek(SeekFrom::Start(self.data_end_offset(count)))
            .map_err(|e| VectorError::VectorCoreError(format!("seek failed: {}", e)))?;
        Ok(())
    }

    /// Create a new mmap vector file at `path` for vectors of dimension `dim`.
    pub fn create(path: &Path, dim: usize) -> Result<Self, VectorError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| VectorError::VectorCoreError(format!("mkdir failed: {}", e)))?;
        }

        let mut file = create_fresh_file(path)
            .map_err(|e| VectorError::VectorCoreError(format!("mmap create failed: {}", e)))?;

        // Write header
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(MAGIC);
        header[4..8].copy_from_slice(&(dim as u32).to_le_bytes());
        header[8..16].copy_from_slice(&0u64.to_le_bytes());
        file.write_all(&header)
            .map_err(|e| VectorError::VectorCoreError(format!("header write failed: {}", e)))?;
        file.sync_all()
            .map_err(|e| VectorError::VectorCoreError(format!("sync failed: {}", e)))?;

        let mut store = Self {
            path: path.to_path_buf(),
            dim,
            count: AtomicU64::new(0),
            mmap: None,
            file,
            dirty: AtomicBool::new(false),
        };
        store.position_write_cursor(0)?;
        Ok(store)
    }

    /// Open an existing mmap vector file.
    pub fn open(path: &Path) -> Result<Self, VectorError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| VectorError::VectorCoreError(format!("mmap open failed: {}", e)))?;

        let metadata = file
            .metadata()
            .map_err(|e| VectorError::VectorCoreError(format!("metadata failed: {}", e)))?;
        let file_len = metadata.len() as usize;

        if file_len < HEADER_SIZE {
            return Err(VectorError::VectorCoreError(
                "mmap file too small for header".into(),
            ));
        }

        // Read header via temporary mmap
        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| VectorError::VectorCoreError(format!("mmap map failed: {}", e)))?
        };
        advise_random_access(&mmap);
        advise_file_random_access(&file);

        if &mmap[0..4] != MAGIC {
            return Err(VectorError::VectorCoreError(
                "mmap file has invalid magic".into(),
            ));
        }

        let dim = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
        let count = u64::from_le_bytes(mmap[8..16].try_into().unwrap());

        // Validate file size
        let expected = HEADER_SIZE + (count as usize) * dim * 4;
        if file_len < expected {
            return Err(VectorError::VectorCoreError(format!(
                "mmap file truncated: expected {} bytes, got {}",
                expected, file_len
            )));
        }

        let mut store = Self {
            path: path.to_path_buf(),
            dim,
            count: AtomicU64::new(count),
            mmap: Some(mmap),
            file,
            dirty: AtomicBool::new(false),
        };
        store.position_write_cursor(count)?;
        Ok(store)
    }

    /// Open if exists, create if not.
    pub fn open_or_create(path: &Path, dim: usize) -> Result<Self, VectorError> {
        if path.exists() {
            Self::open(path)
        } else {
            Self::create(path, dim)
        }
    }

    /// Create a read-mapped exact HVEC file from a complete borrowed row set.
    ///
    /// This is the cold-materialization path for immutable SlateDB segments:
    /// scan durable rows once, write a local exact f32 mmap sidecar, then keep
    /// HNSW scoring off the object-store read path.
    pub fn create_from_slices(path: &Path, vectors: &[&[f32]]) -> Result<Self, VectorError> {
        if vectors.is_empty() {
            return Err(VectorError::VectorCoreError(
                "MmapVectorStore::create_from_slices requires at least one vector".into(),
            ));
        }
        let dim = vectors[0].len();
        if dim == 0 {
            return Err(VectorError::VectorCoreError(
                "vector dim must be > 0".into(),
            ));
        }
        let dim_u32 = u32::try_from(dim).map_err(|_| {
            VectorError::VectorCoreError(format!("vector dim {dim} exceeds HVEC header limit"))
        })?;
        let count = u64::try_from(vectors.len())
            .map_err(|_| VectorError::VectorCoreError("too many vectors for HVEC header".into()))?;
        for (index, vector) in vectors.iter().enumerate() {
            if vector.len() != dim {
                return Err(VectorError::VectorCoreError(format!(
                    "dim mismatch at vector {index}: expected {dim}, got {}",
                    vector.len()
                )));
            }
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| VectorError::VectorCoreError(format!("mkdir failed: {e}")))?;
        }
        let mut file = create_fresh_file(path)
            .map_err(|e| VectorError::VectorCoreError(format!("hvec create failed: {e}")))?;

        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(MAGIC);
        header[4..8].copy_from_slice(&dim_u32.to_le_bytes());
        header[8..16].copy_from_slice(&count.to_le_bytes());
        file.write_all(&header)
            .map_err(|e| VectorError::VectorCoreError(format!("hvec header write: {e}")))?;

        let mut row = Vec::with_capacity(dim * std::mem::size_of::<f32>());
        for vector in vectors {
            row.clear();
            for value in *vector {
                row.extend_from_slice(&value.to_ne_bytes());
            }
            file.write_all(&row)
                .map_err(|e| VectorError::VectorCoreError(format!("hvec data write: {e}")))?;
        }
        file.sync_all()
            .map_err(|e| VectorError::VectorCoreError(format!("hvec sync: {e}")))?;

        drop(file);
        let mut store = Self::open(path)?;
        store.position_write_cursor(count)?;
        Ok(store)
    }

    /// Temporarily mark the mapping as a linear scan. The drop guard restores
    /// random advice so normal HNSW/search access does not keep aggressive
    /// readahead after conversion work finishes.
    fn sequential_scan_advice(&self) -> MmapScanAdvice<'_> {
        let mmap = self.mmap.as_ref();
        if let Some(mmap) = mmap {
            advise_sequential_scan(mmap);
        }
        advise_file_sequential_scan(&self.file);
        MmapScanAdvice {
            mmap,
            file: &self.file,
        }
    }

    /// The caller must hold exclusive access to this store and must not keep any
    /// borrowed slices into the mmap. Used after one-shot conversion when the
    /// legacy HVEC source is about to be dropped/removed.
    fn advise_done_with_pages(&self) {
        if let Some(mmap) = self.mmap.as_ref() {
            // SAFETY: callers hold &mut access through MmapBackend conversion,
            // and this method is called only after scan-local slices have gone
            // out of scope. For a shared file mapping, MADV_DONTNEED discards
            // resident pages and reloads unchanged bytes from the file on any
            // later access.
            let _ = unsafe { mmap.unchecked_advise(UncheckedAdvice::DontNeed) };
        }
        advise_file_done_with_pages(&self.file);
    }

    /// Append a vector and return its ordinal index.
    /// Caller must ensure `data.len() == self.dim`.
    pub fn append(&mut self, data: &[f32]) -> Result<u64, VectorError> {
        if data.len() != self.dim {
            return Err(VectorError::VectorCoreError(format!(
                "dimension mismatch: expected {}, got {}",
                self.dim,
                data.len()
            )));
        }

        let ordinal = self.count.load(Ordering::Acquire);
        self.position_write_cursor(ordinal)?;

        // Write vector bytes
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        io::Write::write_all(&mut self.file, bytes)
            .map_err(|e| VectorError::VectorCoreError(format!("vector write failed: {}", e)))?;

        // Release ensures the vector bytes are visible before count update.
        self.count.store(ordinal + 1, Ordering::Release);

        // Invalidate read mmap — will be remapped on next read
        self.mmap = None;

        // Marks this store as needing flush. Read in flush() to skip
        // the fsync + remap when no append happened since last flush.
        self.dirty.store(true, Ordering::Release);

        Ok(ordinal)
    }

    /// Flush writes and update the header count. Call after a batch of appends.
    /// No-op if no append happened since the last successful flush.
    pub fn flush(&mut self) -> Result<(), VectorError> {
        // Per-batch fan-out fsync gate. flush_mmap_stores iterates every
        // core in the collection regardless of which got writes; without
        // this gate that costs one fsync per untouched core (1.26 s/call
        // p99 on prod with ~16 cores per collection). Load (not swap) so
        // a flush failure leaves dirty=true for the next attempt.
        if !self.dirty.load(Ordering::Acquire) {
            metrics::counter!("helix_mmap_flush_skipped_total").increment(1);
            return Ok(());
        }
        let count = self.count.load(Ordering::Acquire);

        // Update count in header
        self.file
            .seek(SeekFrom::Start(8))
            .map_err(|e| VectorError::VectorCoreError(format!("seek failed: {}", e)))?;
        io::Write::write_all(&mut self.file, &count.to_le_bytes())
            .map_err(|e| VectorError::VectorCoreError(format!("count write failed: {}", e)))?;
        self.file
            .sync_all()
            .map_err(|e| VectorError::VectorCoreError(format!("sync failed: {}", e)))?;

        // Remap for reads
        self.remap()?;
        self.position_write_cursor(count)?;

        // Clear dirty only after every step succeeded. An early ? above
        // leaves dirty=true so the next flush retries.
        self.dirty.store(false, Ordering::Release);
        metrics::counter!("helix_mmap_flush_executed_total").increment(1);

        Ok(())
    }

    /// Remap dirty appended bytes for same-process reads without forcing
    /// durability. SlateDB remains the durable source of truth; this sidecar is
    /// a local hot cache and `flush()` can persist its header later in batches.
    pub fn refresh_read_mmap(&mut self) -> Result<(), VectorError> {
        if self.mmap.is_none() {
            self.remap()?;
        }
        Ok(())
    }

    /// Remap the read-only mmap to reflect current file contents.
    fn remap(&mut self) -> Result<(), VectorError> {
        let mmap = unsafe {
            Mmap::map(&self.file)
                .map_err(|e| VectorError::VectorCoreError(format!("remap failed: {}", e)))?
        };
        advise_random_access(&mmap);
        advise_file_random_access(&self.file);
        self.mmap = Some(mmap);
        Ok(())
    }

    /// Read a vector by ordinal. Returns a zero-copy slice into the mmap.
    /// Returns None if the mmap isn't mapped or ordinal is out of bounds.
    #[inline]
    pub fn get(&self, ordinal: u64) -> Option<&[f32]> {
        let mmap = self.mmap.as_ref()?;
        let ordinal = ordinal as usize;
        let count = self.count.load(Ordering::Acquire) as usize;
        if ordinal >= count {
            return None;
        }
        let offset = HEADER_SIZE + ordinal * self.dim * 4;
        let end = offset + self.dim * 4;
        if end > mmap.len() {
            return None;
        }
        let bytes = &mmap[offset..end];
        let slice = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.dim) };
        Some(slice)
    }

    /// Number of vectors stored.
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Acquire)
    }

    /// Vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// File path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

// ============================================================================
// HVS8: at-rest SQ8-quantized vector store
// ============================================================================
//
// HVS8 file layout:
//   [header: 24 bytes]
//     magic:   b"HVS8" (4 bytes)
//     dim:     u32 LE  (4 bytes)
//     count:   u64 LE  (8 bytes)
//     flags:   u64 LE  (8 bytes; reserved for future format extensions)
//   [params: dim * 8 bytes]
//     for each dim i: f32 mins[i] (4 bytes), f32 scales[i] (4 bytes)
//   [data: count * dim bytes]
//     each vector quantized to u8 per dim
//
// The reader auto-detects format by magic bytes, so HVS8 files coexist
// with legacy HVEC files in the same collection. Distance functions can
// use cosine_u8 directly on the stored bytes (no per-search dequantize).
// HNSW rebuild on merge dequantizes transiently into f32 buffers — same
// fidelity loss as the SQ8-quantized HNSW build path that already runs
// at line 2118 of vector_core.rs.

const QUANTIZED_MAGIC: &[u8; 4] = b"HVS8";
const QUANTIZED_HEADER_SIZE: usize = 24; // 4 magic + 4 dim + 8 count + 8 flags
const TURBO_QUANT_MAGIC: &[u8; 4] = b"HVTQ";
const TURBO_QUANT_HEADER_SIZE: usize = 32; // 4 magic + 4 dim + 8 count + 8 flags + 4 record_len + 4 reserved

/// On-disk SQ8-quantized vector store. Created in one shot from a complete
/// vector set (since SQ8 params depend on the full distribution); no
/// streaming append. Use this for indexed segments produced by the merge
/// optimizer — the legacy `MmapVectorStore` continues to back mutable
/// segments where vectors arrive one at a time.
pub struct MmapQuantizedStore {
    path: PathBuf,
    dim: usize,
    count: u64,
    /// Per-dim mins (length == dim). Used to dequantize back to f32.
    mins: Vec<f32>,
    /// Per-dim scales (length == dim). 255.0 / (max - min), or 0.0 if range is 0.
    scales: Vec<f32>,
    /// Read-only mmap covering the entire file.
    mmap: Mmap,
    /// Offset in the mmap where vector u8 data begins.
    data_offset: usize,
}

/// Distance selector used by mmap-backed sidecars when scoring without first
/// materializing an owned `Vec<f32>`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MmapDistanceMetric {
    Cosine,
    Dot,
    Euclid,
}

impl MmapQuantizedStore {
    /// Create a new HVS8 file from a complete set of f32 vectors. Fits SQ8
    /// params on the input distribution, quantizes, and writes header +
    /// params + data in one shot. The returned store is read-only.
    ///
    /// Caller guarantees all vectors have length `dim` and that the input
    /// is non-empty. An empty input is rejected (an empty quantized file
    /// has no SQ8 params to derive).
    pub fn create_from(path: &Path, vectors: &[Vec<f32>]) -> Result<Self, VectorError> {
        let rows = vectors.iter().map(Vec::as_slice).collect::<Vec<_>>();
        Self::create_from_slices(path, &rows)
    }

    /// Borrowed-row variant of `create_from`. Used by HNSW publish so prepared
    /// rows can be quantized into HVS8 without another full vector copy.
    pub fn create_from_slices(path: &Path, vectors: &[&[f32]]) -> Result<Self, VectorError> {
        if vectors.is_empty() {
            return Err(VectorError::VectorCoreError(
                "MmapQuantizedStore::create_from requires at least one vector".into(),
            ));
        }
        let dim = vectors[0].len();
        if dim == 0 {
            return Err(VectorError::VectorCoreError(
                "vector dim must be > 0".into(),
            ));
        }
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != dim {
                return Err(VectorError::VectorCoreError(format!(
                    "dim mismatch at vector {}: expected {}, got {}",
                    i,
                    dim,
                    v.len()
                )));
            }
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| VectorError::VectorCoreError(format!("mkdir failed: {}", e)))?;
        }

        // Fit SQ8 params on the full input distribution.
        use super::simd::SQ8Params;
        let params = SQ8Params::fit_slices(vectors);
        let count = vectors.len() as u64;

        let mut file = create_fresh_file(path)
            .map_err(|e| VectorError::VectorCoreError(format!("hvs8 create failed: {}", e)))?;

        // Header
        let mut header = [0u8; QUANTIZED_HEADER_SIZE];
        header[0..4].copy_from_slice(QUANTIZED_MAGIC);
        header[4..8].copy_from_slice(&(dim as u32).to_le_bytes());
        header[8..16].copy_from_slice(&count.to_le_bytes());
        header[16..24].copy_from_slice(&0u64.to_le_bytes());
        file.write_all(&header)
            .map_err(|e| VectorError::VectorCoreError(format!("hvs8 header write: {}", e)))?;

        // Params blob (interleaved [min, scale] per dim).
        let mut params_buf = Vec::with_capacity(dim * 8);
        for i in 0..dim {
            params_buf.extend_from_slice(&params.mins[i].to_le_bytes());
            params_buf.extend_from_slice(&params.scales[i].to_le_bytes());
        }
        file.write_all(&params_buf)
            .map_err(|e| VectorError::VectorCoreError(format!("hvs8 params write: {}", e)))?;

        // Quantize and write vector data one row at a time. Avoid holding an
        // extra count*dim u8 buffer while a large merge already has the f32
        // build rows resident.
        for v in vectors {
            let quantized = params.quantize(v);
            file.write_all(&quantized)
                .map_err(|e| VectorError::VectorCoreError(format!("hvs8 data write: {}", e)))?;
        }

        file.sync_all()
            .map_err(|e| VectorError::VectorCoreError(format!("hvs8 sync: {}", e)))?;

        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| VectorError::VectorCoreError(format!("hvs8 mmap: {}", e)))?
        };
        advise_random_access(&mmap);

        let data_offset = QUANTIZED_HEADER_SIZE + dim * 8;

        Ok(Self {
            path: path.to_path_buf(),
            dim,
            count,
            mins: params.mins,
            scales: params.scales,
            mmap,
            data_offset,
        })
    }

    /// Create a new HVS8 file from an HVEC mmap source without materializing
    /// all vectors on the heap. This does two linear mmap scans: one to fit
    /// per-dimension SQ8 params and one to write quantized bytes.
    pub fn create_from_hvec(path: &Path, src: &MmapVectorStore) -> Result<Self, VectorError> {
        let count = src.count();
        if count == 0 {
            return Err(VectorError::VectorCoreError(
                "MmapQuantizedStore::create_from_hvec requires at least one vector".into(),
            ));
        }
        let dim = src.dim();
        if dim == 0 {
            return Err(VectorError::VectorCoreError(
                "vector dim must be > 0".into(),
            ));
        }

        let _scan_advice = src.sequential_scan_advice();

        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];
        for ord in 0..count {
            let v = src.get(ord).ok_or_else(|| {
                VectorError::VectorCoreError(format!(
                    "create_from_hvec: ordinal {} out of bounds",
                    ord
                ))
            })?;
            for (i, &val) in v.iter().enumerate() {
                mins[i] = mins[i].min(val);
                maxs[i] = maxs[i].max(val);
            }
        }

        let scales: Vec<f32> = mins
            .iter()
            .zip(maxs.iter())
            .map(|(&lo, &hi)| {
                let range = hi - lo;
                if range <= f32::EPSILON {
                    0.0
                } else {
                    255.0 / range
                }
            })
            .collect();
        let params = super::simd::SQ8Params { mins, scales };

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| VectorError::VectorCoreError(format!("mkdir failed: {}", e)))?;
        }
        let mut file = create_fresh_file(path)
            .map_err(|e| VectorError::VectorCoreError(format!("hvs8 create failed: {}", e)))?;

        let mut header = [0u8; QUANTIZED_HEADER_SIZE];
        header[0..4].copy_from_slice(QUANTIZED_MAGIC);
        header[4..8].copy_from_slice(&(dim as u32).to_le_bytes());
        header[8..16].copy_from_slice(&count.to_le_bytes());
        header[16..24].copy_from_slice(&0u64.to_le_bytes());
        file.write_all(&header)
            .map_err(|e| VectorError::VectorCoreError(format!("hvs8 header write: {}", e)))?;

        let mut params_buf = Vec::with_capacity(dim * 8);
        for i in 0..dim {
            params_buf.extend_from_slice(&params.mins[i].to_le_bytes());
            params_buf.extend_from_slice(&params.scales[i].to_le_bytes());
        }
        file.write_all(&params_buf)
            .map_err(|e| VectorError::VectorCoreError(format!("hvs8 params write: {}", e)))?;

        for ord in 0..count {
            let v = src.get(ord).ok_or_else(|| {
                VectorError::VectorCoreError(format!(
                    "create_from_hvec: ordinal {} out of bounds",
                    ord
                ))
            })?;
            let quantized = params.quantize(v);
            file.write_all(&quantized)
                .map_err(|e| VectorError::VectorCoreError(format!("hvs8 data write: {}", e)))?;
        }

        file.sync_all()
            .map_err(|e| VectorError::VectorCoreError(format!("hvs8 sync: {}", e)))?;
        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| VectorError::VectorCoreError(format!("hvs8 mmap: {}", e)))?
        };
        advise_random_access(&mmap);

        Ok(Self {
            path: path.to_path_buf(),
            dim,
            count,
            data_offset: QUANTIZED_HEADER_SIZE + dim * 8,
            mins: params.mins,
            scales: params.scales,
            mmap,
        })
    }

    /// Open an existing HVS8 file for reads.
    pub fn open(path: &Path) -> Result<Self, VectorError> {
        let file = OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(|e| VectorError::VectorCoreError(format!("hvs8 open: {}", e)))?;
        let metadata = file
            .metadata()
            .map_err(|e| VectorError::VectorCoreError(format!("hvs8 metadata: {}", e)))?;
        let file_len = metadata.len() as usize;

        if file_len < QUANTIZED_HEADER_SIZE {
            return Err(VectorError::VectorCoreError(
                "hvs8 file too small for header".into(),
            ));
        }

        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| VectorError::VectorCoreError(format!("hvs8 mmap: {}", e)))?
        };
        advise_random_access(&mmap);

        if &mmap[0..4] != QUANTIZED_MAGIC {
            return Err(VectorError::VectorCoreError(format!(
                "hvs8 magic mismatch: expected HVS8, got {:?}",
                &mmap[0..4]
            )));
        }
        let dim = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
        let count = u64::from_le_bytes(mmap[8..16].try_into().unwrap());
        // mmap[16..24] = flags, ignored for now.

        let data_offset = QUANTIZED_HEADER_SIZE + dim * 8;
        let expected = data_offset + (count as usize) * dim;
        if file_len < expected {
            return Err(VectorError::VectorCoreError(format!(
                "hvs8 truncated: expected {} bytes, got {}",
                expected, file_len
            )));
        }

        // Decode params.
        let mut mins = Vec::with_capacity(dim);
        let mut scales = Vec::with_capacity(dim);
        for i in 0..dim {
            let off = QUANTIZED_HEADER_SIZE + i * 8;
            mins.push(f32::from_le_bytes(mmap[off..off + 4].try_into().unwrap()));
            scales.push(f32::from_le_bytes(
                mmap[off + 4..off + 8].try_into().unwrap(),
            ));
        }

        Ok(Self {
            path: path.to_path_buf(),
            dim,
            count,
            mins,
            scales,
            mmap,
            data_offset,
        })
    }

    /// Auto-detect on-disk format. Returns the raw magic bytes so callers
    /// can dispatch between legacy `MmapVectorStore` and `MmapQuantizedStore`
    /// without opening the file twice.
    pub fn peek_magic(path: &Path) -> Result<[u8; 4], VectorError> {
        use std::io::Read;
        let mut file = OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(|e| VectorError::VectorCoreError(format!("peek open: {}", e)))?;
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)
            .map_err(|e| VectorError::VectorCoreError(format!("peek read: {}", e)))?;
        Ok(buf)
    }

    /// Zero-copy view of the quantized u8 vector at the given ordinal.
    /// Returns None if ordinal is out of bounds.
    #[inline]
    pub fn get_quantized(&self, ordinal: u64) -> Option<&[u8]> {
        if ordinal >= self.count {
            return None;
        }
        let start = self.data_offset + (ordinal as usize) * self.dim;
        let end = start + self.dim;
        if end > self.mmap.len() {
            return None;
        }
        Some(&self.mmap[start..end])
    }

    /// Dequantize the vector at `ordinal` back to f32. Allocates a Vec<f32>
    /// per call — use `get_quantized` + `cosine_u8` on the search hot path
    /// to avoid this cost. Reserved for HNSW rebuild and external callers
    /// that need raw f32 (e.g. exact-precision distance for re-ranking).
    pub fn get_dequantized(&self, ordinal: u64) -> Option<Vec<f32>> {
        let q = self.get_quantized(ordinal)?;
        let mut out = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let scale = self.scales[i];
            let min = self.mins[i];
            let v = if scale == 0.0 {
                // All values were equal; the quantized byte is sentinel 128.
                min
            } else {
                min + (q[i] as f32) / scale
            };
            out.push(v);
        }
        Some(out)
    }

    /// Score one HVS8 row against an f32 query without allocating a temporary
    /// dequantized vector. This is exact with respect to `get_dequantized`:
    /// each byte is expanded through the same per-dimension min/scale pair, but
    /// only for the lifetime of the distance accumulator.
    #[inline]
    pub fn distance_to(
        &self,
        ordinal: u64,
        query: &[f32],
        metric: MmapDistanceMetric,
    ) -> Option<Result<f32, VectorError>> {
        if query.len() != self.dim {
            return Some(Err(VectorError::InvalidVectorLength));
        }
        let q = self.get_quantized(ordinal)?;
        Some(Ok(match metric {
            MmapDistanceMetric::Cosine => self.cosine_distance(q, query),
            MmapDistanceMetric::Dot => self.dot_distance(q, query),
            MmapDistanceMetric::Euclid => self.euclid_distance(q, query),
        }))
    }

    #[inline]
    fn dequantized_value(&self, q: &[u8], i: usize) -> f32 {
        let scale = self.scales[i];
        if scale == 0.0 {
            self.mins[i]
        } else {
            self.mins[i] + (q[i] as f32) / scale
        }
    }

    #[inline]
    fn cosine_distance(&self, q: &[u8], query: &[f32]) -> f32 {
        let mut dot = 0.0f64;
        let mut norm_doc = 0.0f64;
        let mut norm_query = 0.0f64;

        for i in 0..self.dim {
            let doc = self.dequantized_value(q, i) as f64;
            let query = query[i] as f64;
            dot += doc * query;
            norm_doc += doc * doc;
            norm_query += query * query;
        }

        if norm_doc == 0.0 || norm_query == 0.0 {
            return 1.0;
        }
        (1.0 - dot / (norm_doc.sqrt() * norm_query.sqrt())) as f32
    }

    #[inline]
    fn dot_distance(&self, q: &[u8], query: &[f32]) -> f32 {
        let mut dot = 0.0f64;
        for i in 0..self.dim {
            dot += self.dequantized_value(q, i) as f64 * query[i] as f64;
        }
        -dot as f32
    }

    #[inline]
    fn euclid_distance(&self, q: &[u8], query: &[f32]) -> f32 {
        let mut sum_sq = 0.0f64;
        for i in 0..self.dim {
            let delta = self.dequantized_value(q, i) as f64 - query[i] as f64;
            sum_sq += delta * delta;
        }
        sum_sq.sqrt() as f32
    }

    /// Number of vectors stored.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Per-dim quantization mins. Length == `dim`.
    pub fn mins(&self) -> &[f32] {
        &self.mins
    }

    /// Per-dim quantization scales. Length == `dim`. A scale of 0.0 means
    /// the input had a constant value at that dim; the byte is 128 and
    /// dequantizes back to `mins[i]`.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// File path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

// ============================================================================
// HVTQ: at-rest TurboProd-encoded vector store
// ============================================================================
//
// HVTQ file layout:
//   [header: 32 bytes]
//     magic:      b"HVTQ" (4 bytes)
//     dim:        u32 LE  (4 bytes)
//     count:      u64 LE  (8 bytes)
//     flags:      u64 LE  (8 bytes; reserved)
//     record_len: u32 LE  (4 bytes)
//     reserved:   u32 LE  (4 bytes)
//   [data: count * record_len bytes]
//     each row is one complete SPN1/TurboProd payload
//
// TurboProd payload length is fixed for a fixed vector dimension and config,
// so ordinal lookup stays O(1) like HVEC/HVS8. The sidecar stores the same
// bytes the compact LMDB path would otherwise store per vector; merged TQ
// segments can therefore use LMDB markers instead of duplicating payloads.

pub struct MmapTurboQuantStore {
    path: PathBuf,
    dim: usize,
    count: u64,
    record_len: usize,
    mmap: Mmap,
    data_offset: usize,
    fastscan: Mutex<Option<TurboQuantFastScanIndex>>,
}

pub struct MmapSpindleStore {
    path: PathBuf,
    dim: usize,
    count: u64,
    record_len: usize,
    mmap: Mmap,
    data_offset: usize,
}

const SPINDLE_ENCODED_MAGIC: &[u8; 4] = b"HSPN";
const SPINDLE_ENCODED_HEADER_SIZE: usize = 32;
const SCALAR_INT8_TAG: u8 = 1;
const TURBO_PROD_TAG: u8 = 4;
const HVTQ_FASTSCAN_BLOCK: usize = 16;

struct TurboQuantRecordParts<'a> {
    stored_dim: usize,
    doc_norm: f32,
    residual_norm: f32,
    mse_bytes: &'a [u8],
    qjl_bytes: &'a [u8],
}

struct TurboQuantFastScanIndex {
    dim: usize,
    stored_dim: usize,
    count: usize,
    block_count: usize,
    mse_codes: Vec<[u8; HVTQ_FASTSCAN_BLOCK]>,
    mse_bitplanes: Vec<[u16; 3]>,
    qjl_bitplanes: Vec<u16>,
    doc_norms: Vec<f32>,
    residual_norms: Vec<f32>,
}

impl TurboQuantFastScanIndex {
    fn score_all(
        &self,
        query: &crate::helix_engine::vector_core::spindle::PreparedTurboProdQuery,
    ) -> Result<Vec<ApproximateInnerProduct>, VectorError> {
        let compare_mse_dim = self.stored_dim.min(query.mse_dim).min(query.mse_lut.len());
        let compare_qjl_dim = self.dim.min(query.q_transformed.len());
        let mut out = Vec::with_capacity(self.count);

        for block in 0..self.block_count {
            let block_start = block * HVTQ_FASTSCAN_BLOCK;
            let block_len = (self.count - block_start).min(HVTQ_FASTSCAN_BLOCK);
            let mut mse = [0.0f64; HVTQ_FASTSCAN_BLOCK];
            let mut qjl = [0.0f64; HVTQ_FASTSCAN_BLOCK];

            let mse_base = block * self.stored_dim;
            for dim_idx in 0..compare_mse_dim {
                let planes = self.mse_bitplanes[mse_base + dim_idx];
                let lut = &query.mse_lut[dim_idx];
                for (lane, score) in mse.iter_mut().enumerate().take(block_len) {
                    let mask = 1u16 << lane;
                    let code = ((planes[0] & mask != 0) as usize)
                        | (((planes[1] & mask != 0) as usize) << 1)
                        | (((planes[2] & mask != 0) as usize) << 2);
                    *score += lut[code.min(MSE_LEVELS - 1)];
                }
            }

            let qjl_base = block * self.dim;
            for dim_idx in 0..compare_qjl_dim {
                let plane = self.qjl_bitplanes[qjl_base + dim_idx];
                let q = query.q_transformed[dim_idx];
                for (lane, score) in qjl.iter_mut().enumerate().take(block_len) {
                    if plane & (1u16 << lane) != 0 {
                        *score += q;
                    } else {
                        *score -= q;
                    }
                }
            }

            for lane in 0..block_len {
                let ordinal = block_start + lane;
                let residual_norm = self.residual_norms[ordinal] as f64;
                let qjl_dot = if residual_norm > 0.0 && self.dim > 0 {
                    let scale =
                        (std::f64::consts::PI / 2.0).sqrt() * residual_norm / self.dim as f64;
                    qjl[lane] * scale
                } else {
                    0.0
                };
                out.push(ApproximateInnerProduct {
                    unit_dot: (mse[lane] + qjl_dot).clamp(-1.0, 1.0),
                    query_norm: query.query_norm,
                    doc_norm: self.doc_norms[ordinal] as f64,
                });
            }
        }

        Ok(out)
    }

    #[cfg(all(test, target_arch = "aarch64"))]
    fn score_all_i8_lut_neon(
        &self,
        query: &crate::helix_engine::vector_core::spindle::PreparedTurboProdQuery,
    ) -> Result<Vec<ApproximateInnerProduct>, VectorError> {
        let compare_mse_dim = self.stored_dim.min(query.mse_dim).min(query.mse_lut.len());
        let compare_qjl_dim = self.dim.min(query.q_transformed.len());
        let mut max_abs = 0.0f64;
        for row in query.mse_lut.iter().take(compare_mse_dim) {
            for value in row {
                max_abs = max_abs.max(value.abs());
            }
        }
        let quant_scale = if max_abs > 0.0 { 127.0 / max_abs } else { 1.0 };
        let inv_quant_scale = 1.0 / quant_scale;
        let qlut = query
            .mse_lut
            .iter()
            .take(compare_mse_dim)
            .map(|row| {
                let mut encoded = [0u8; 16];
                for code in 0..MSE_LEVELS {
                    let q = (row[code] * quant_scale).round().clamp(-127.0, 127.0) as i8;
                    encoded[code] = q as u8;
                }
                encoded
            })
            .collect::<Vec<_>>();
        let mut out = Vec::with_capacity(self.count);

        for block in 0..self.block_count {
            let block_start = block * HVTQ_FASTSCAN_BLOCK;
            let block_len = (self.count - block_start).min(HVTQ_FASTSCAN_BLOCK);
            let mse_i32 = unsafe { self.mse_block_i8_lut_neon(block, &qlut) };
            let mut qjl = [0.0f64; HVTQ_FASTSCAN_BLOCK];

            let qjl_base = block * self.dim;
            for dim_idx in 0..compare_qjl_dim {
                let plane = self.qjl_bitplanes[qjl_base + dim_idx];
                let q = query.q_transformed[dim_idx];
                for (lane, score) in qjl.iter_mut().enumerate().take(block_len) {
                    if plane & (1u16 << lane) != 0 {
                        *score += q;
                    } else {
                        *score -= q;
                    }
                }
            }

            for lane in 0..block_len {
                let ordinal = block_start + lane;
                let residual_norm = self.residual_norms[ordinal] as f64;
                let qjl_dot = if residual_norm > 0.0 && self.dim > 0 {
                    let scale =
                        (std::f64::consts::PI / 2.0).sqrt() * residual_norm / self.dim as f64;
                    qjl[lane] * scale
                } else {
                    0.0
                };
                out.push(ApproximateInnerProduct {
                    unit_dot: (mse_i32[lane] as f64 * inv_quant_scale + qjl_dot).clamp(-1.0, 1.0),
                    query_norm: query.query_norm,
                    doc_norm: self.doc_norms[ordinal] as f64,
                });
            }
        }

        Ok(out)
    }

    #[cfg(all(test, target_arch = "aarch64"))]
    unsafe fn mse_block_i8_lut_neon(&self, block: usize, qlut: &[[u8; 16]]) -> [i32; 16] {
        use std::arch::aarch64::*;

        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut acc2 = vdupq_n_s32(0);
        let mut acc3 = vdupq_n_s32(0);
        let masks = vld1q_u8([1u8, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128].as_ptr());
        let zero = vdupq_n_u8(0);
        let one = vdupq_n_u8(1);
        let two = vdupq_n_u8(2);
        let four = vdupq_n_u8(4);
        let mse_base = block * self.stored_dim;

        for (dim_idx, table_bytes) in qlut.iter().enumerate() {
            let planes = self.mse_bitplanes[mse_base + dim_idx];
            let p0 = vcombine_u8(
                vdup_n_u8((planes[0] & 0xff) as u8),
                vdup_n_u8((planes[0] >> 8) as u8),
            );
            let p1 = vcombine_u8(
                vdup_n_u8((planes[1] & 0xff) as u8),
                vdup_n_u8((planes[1] >> 8) as u8),
            );
            let p2 = vcombine_u8(
                vdup_n_u8((planes[2] & 0xff) as u8),
                vdup_n_u8((planes[2] >> 8) as u8),
            );
            let b0 = vandq_u8(vcgtq_u8(vandq_u8(p0, masks), zero), one);
            let b1 = vandq_u8(vcgtq_u8(vandq_u8(p1, masks), zero), two);
            let b2 = vandq_u8(vcgtq_u8(vandq_u8(p2, masks), zero), four);
            let codes = vorrq_u8(vorrq_u8(b0, b1), b2);
            let table = vld1q_u8(table_bytes.as_ptr());
            let vals = vreinterpretq_s8_u8(vqtbl1q_u8(table, codes));
            let lo = vmovl_s8(vget_low_s8(vals));
            let hi = vmovl_s8(vget_high_s8(vals));
            acc0 = vaddq_s32(acc0, vmovl_s16(vget_low_s16(lo)));
            acc1 = vaddq_s32(acc1, vmovl_s16(vget_high_s16(lo)));
            acc2 = vaddq_s32(acc2, vmovl_s16(vget_low_s16(hi)));
            acc3 = vaddq_s32(acc3, vmovl_s16(vget_high_s16(hi)));
        }

        let mut out = [0i32; HVTQ_FASTSCAN_BLOCK];
        vst1q_s32(out.as_mut_ptr(), acc0);
        vst1q_s32(out.as_mut_ptr().add(4), acc1);
        vst1q_s32(out.as_mut_ptr().add(8), acc2);
        vst1q_s32(out.as_mut_ptr().add(12), acc3);
        out
    }

    #[cfg(target_arch = "aarch64")]
    fn score_all_mse_code_i8_lut_neon(
        &self,
        query: &crate::helix_engine::vector_core::spindle::PreparedTurboProdQuery,
    ) -> Result<Vec<ApproximateInnerProduct>, VectorError> {
        let compare_mse_dim = self.stored_dim.min(query.mse_dim).min(query.mse_lut.len());
        let mut max_abs = 0.0f64;
        for row in query.mse_lut.iter().take(compare_mse_dim) {
            for value in row {
                max_abs = max_abs.max(value.abs());
            }
        }
        let quant_scale = if max_abs > 0.0 { 127.0 / max_abs } else { 1.0 };
        let inv_quant_scale = 1.0 / quant_scale;
        let qlut = query
            .mse_lut
            .iter()
            .take(compare_mse_dim)
            .map(|row| {
                let mut encoded = [0u8; 16];
                for code in 0..MSE_LEVELS {
                    let q = (row[code] * quant_scale).round().clamp(-127.0, 127.0) as i8;
                    encoded[code] = q as u8;
                }
                encoded
            })
            .collect::<Vec<_>>();
        let mut out = Vec::with_capacity(self.count);

        for block in 0..self.block_count {
            let block_start = block * HVTQ_FASTSCAN_BLOCK;
            let block_len = (self.count - block_start).min(HVTQ_FASTSCAN_BLOCK);
            let mse_i32 = unsafe { self.mse_code_block_i8_lut_neon(block, &qlut) };
            for (lane, mse_score) in mse_i32.iter().enumerate().take(block_len) {
                let ordinal = block_start + lane;
                out.push(ApproximateInnerProduct {
                    unit_dot: (*mse_score as f64 * inv_quant_scale).clamp(-1.0, 1.0),
                    query_norm: query.query_norm,
                    doc_norm: self.doc_norms[ordinal] as f64,
                });
            }
        }

        Ok(out)
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn mse_code_block_i8_lut_neon(&self, block: usize, qlut: &[[u8; 16]]) -> [i32; 16] {
        use std::arch::aarch64::*;

        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut acc2 = vdupq_n_s32(0);
        let mut acc3 = vdupq_n_s32(0);
        let mse_base = block * self.stored_dim;

        for (dim_idx, table_bytes) in qlut.iter().enumerate() {
            let codes = vld1q_u8(self.mse_codes[mse_base + dim_idx].as_ptr());
            let table = vld1q_u8(table_bytes.as_ptr());
            let vals = vreinterpretq_s8_u8(vqtbl1q_u8(table, codes));
            let lo = vmovl_s8(vget_low_s8(vals));
            let hi = vmovl_s8(vget_high_s8(vals));
            acc0 = vaddq_s32(acc0, vmovl_s16(vget_low_s16(lo)));
            acc1 = vaddq_s32(acc1, vmovl_s16(vget_high_s16(lo)));
            acc2 = vaddq_s32(acc2, vmovl_s16(vget_low_s16(hi)));
            acc3 = vaddq_s32(acc3, vmovl_s16(vget_high_s16(hi)));
        }

        let mut out = [0i32; HVTQ_FASTSCAN_BLOCK];
        vst1q_s32(out.as_mut_ptr(), acc0);
        vst1q_s32(out.as_mut_ptr().add(4), acc1);
        vst1q_s32(out.as_mut_ptr().add(8), acc2);
        vst1q_s32(out.as_mut_ptr().add(12), acc3);
        out
    }
}

impl MmapSpindleStore {
    pub fn create_from_encoded(path: &Path, rows: &[&[u8]]) -> Result<Self, VectorError> {
        if rows.is_empty() {
            return Err(VectorError::VectorCoreError(
                "MmapSpindleStore::create_from_encoded requires at least one row".into(),
            ));
        }
        let first_decoded = decode_vector(rows[0])?;
        let dim = first_decoded.len();
        if dim == 0 {
            return Err(VectorError::VectorCoreError(
                "vector dim must be > 0".into(),
            ));
        }
        let record_len = rows[0].len();
        if record_len == 0 {
            return Err(VectorError::InvalidVectorData);
        }
        for (i, row) in rows.iter().enumerate() {
            if row.len() != record_len {
                return Err(VectorError::VectorCoreError(format!(
                    "hspn record length mismatch at row {}: expected {}, got {}",
                    i,
                    record_len,
                    row.len()
                )));
            }
            let decoded = decode_vector(row)?;
            if decoded.len() != dim {
                return Err(VectorError::VectorCoreError(format!(
                    "hspn dim mismatch at row {}: expected {}, got {}",
                    i,
                    dim,
                    decoded.len()
                )));
            }
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| VectorError::VectorCoreError(format!("mkdir failed: {}", e)))?;
        }

        let count = rows.len() as u64;
        let mut file = create_fresh_file(path)
            .map_err(|e| VectorError::VectorCoreError(format!("hspn create failed: {}", e)))?;

        let mut header = [0u8; SPINDLE_ENCODED_HEADER_SIZE];
        header[0..4].copy_from_slice(SPINDLE_ENCODED_MAGIC);
        header[4..8].copy_from_slice(&(dim as u32).to_le_bytes());
        header[8..16].copy_from_slice(&count.to_le_bytes());
        header[16..24].copy_from_slice(&0u64.to_le_bytes());
        header[24..28].copy_from_slice(&(record_len as u32).to_le_bytes());
        header[28..32].copy_from_slice(&0u32.to_le_bytes());
        file.write_all(&header)
            .map_err(|e| VectorError::VectorCoreError(format!("hspn header write: {}", e)))?;
        for row in rows {
            file.write_all(row)
                .map_err(|e| VectorError::VectorCoreError(format!("hspn data write: {}", e)))?;
        }
        file.sync_all()
            .map_err(|e| VectorError::VectorCoreError(format!("hspn sync: {}", e)))?;
        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| VectorError::VectorCoreError(format!("hspn mmap: {}", e)))?
        };
        advise_random_access(&mmap);

        Ok(Self {
            path: path.to_path_buf(),
            dim,
            count,
            record_len,
            mmap,
            data_offset: SPINDLE_ENCODED_HEADER_SIZE,
        })
    }

    pub fn open(path: &Path) -> Result<Self, VectorError> {
        let file = OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(|e| VectorError::VectorCoreError(format!("hspn open: {}", e)))?;
        let metadata = file
            .metadata()
            .map_err(|e| VectorError::VectorCoreError(format!("hspn metadata: {}", e)))?;
        let file_len = metadata.len() as usize;
        if file_len < SPINDLE_ENCODED_HEADER_SIZE {
            return Err(VectorError::VectorCoreError(
                "hspn file too small for header".into(),
            ));
        }

        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| VectorError::VectorCoreError(format!("hspn mmap: {}", e)))?
        };
        advise_random_access(&mmap);

        if &mmap[0..4] != SPINDLE_ENCODED_MAGIC {
            return Err(VectorError::VectorCoreError(format!(
                "hspn magic mismatch: expected HSPN, got {:?}",
                &mmap[0..4]
            )));
        }
        let dim = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
        let count = u64::from_le_bytes(mmap[8..16].try_into().unwrap());
        let record_len = u32::from_le_bytes(mmap[24..28].try_into().unwrap()) as usize;
        if dim == 0 || record_len == 0 {
            return Err(VectorError::InvalidVectorData);
        }
        let data_offset = SPINDLE_ENCODED_HEADER_SIZE;
        let expected = data_offset + (count as usize) * record_len;
        if file_len < expected {
            return Err(VectorError::VectorCoreError(format!(
                "hspn truncated: expected {} bytes, got {}",
                expected, file_len
            )));
        }

        Ok(Self {
            path: path.to_path_buf(),
            dim,
            count,
            record_len,
            mmap,
            data_offset,
        })
    }

    #[inline]
    pub fn get_encoded(&self, ordinal: u64) -> Option<&[u8]> {
        if ordinal >= self.count {
            return None;
        }
        let start = self.data_offset + (ordinal as usize) * self.record_len;
        let end = start + self.record_len;
        if end > self.mmap.len() {
            return None;
        }
        Some(&self.mmap[start..end])
    }

    #[inline]
    pub fn get_decoded(&self, ordinal: u64) -> Option<Vec<f32>> {
        self.get_encoded(ordinal)
            .and_then(|bytes| decode_vector(bytes).ok())
    }

    #[inline]
    pub fn score_encoded_to(
        &self,
        ordinal: u64,
        prepared: &PreparedSpindleQuery,
    ) -> Option<Result<ApproximateInnerProduct, VectorError>> {
        let bytes = self.get_encoded(ordinal)?;
        Some(match score_encoded(prepared, bytes) {
            Ok(Some(approx)) => Ok(approx),
            Ok(None) => Err(VectorError::VectorCoreError(
                "prepared query did not match HSPN payload".into(),
            )),
            Err(err) => Err(err),
        })
    }

    #[inline]
    pub fn distance_to(
        &self,
        ordinal: u64,
        query: &[f32],
        metric: MmapDistanceMetric,
    ) -> Option<Result<f32, VectorError>> {
        if query.len() != self.dim {
            return Some(Err(VectorError::InvalidVectorLength));
        }
        let bytes = self.get_encoded(ordinal)?;
        if let Some((scale, quantized)) = Self::scalar_int8_parts(bytes, self.dim) {
            return Some(Ok(match metric {
                MmapDistanceMetric::Cosine => Self::scalar_int8_cosine(scale, quantized, query),
                MmapDistanceMetric::Dot => Self::scalar_int8_dot(scale, quantized, query),
                MmapDistanceMetric::Euclid => Self::scalar_int8_euclid(scale, quantized, query),
            }));
        }

        Some(decode_vector(bytes).and_then(|decoded| {
            if decoded.len() != query.len() {
                return Err(VectorError::InvalidVectorLength);
            }
            Ok(match metric {
                MmapDistanceMetric::Cosine => super::simd::cosine_f32(&decoded, query),
                MmapDistanceMetric::Dot => super::simd::dot_f32(&decoded, query),
                MmapDistanceMetric::Euclid => super::simd::euclid_f32(&decoded, query).sqrt(),
            })
        }))
    }

    #[inline]
    fn scalar_int8_parts(bytes: &[u8], expected_dim: usize) -> Option<(f32, &[u8])> {
        if bytes.len() != 13 + expected_dim || !bytes.starts_with(b"SPN1") {
            return None;
        }
        if bytes[4] != SCALAR_INT8_TAG {
            return None;
        }
        let dim = u32::from_le_bytes(bytes[5..9].try_into().ok()?) as usize;
        if dim != expected_dim {
            return None;
        }
        let scale = f32::from_le_bytes(bytes[9..13].try_into().ok()?);
        Some((scale, &bytes[13..]))
    }

    #[inline]
    fn scalar_int8_doc_value(scale: f32, byte: u8) -> f64 {
        (byte as i8) as f64 * scale as f64
    }

    #[inline]
    fn scalar_int8_cosine(scale: f32, quantized: &[u8], query: &[f32]) -> f32 {
        let mut dot = 0.0f64;
        let mut norm_doc = 0.0f64;
        let mut norm_query = 0.0f64;
        for (byte, query_value) in quantized.iter().zip(query.iter()) {
            let doc = Self::scalar_int8_doc_value(scale, *byte);
            let query = *query_value as f64;
            dot += doc * query;
            norm_doc += doc * doc;
            norm_query += query * query;
        }
        if norm_doc == 0.0 || norm_query == 0.0 {
            return 1.0;
        }
        (1.0 - dot / (norm_doc.sqrt() * norm_query.sqrt())) as f32
    }

    #[inline]
    fn scalar_int8_dot(scale: f32, quantized: &[u8], query: &[f32]) -> f32 {
        let mut dot = 0.0f64;
        for (byte, query_value) in quantized.iter().zip(query.iter()) {
            dot += Self::scalar_int8_doc_value(scale, *byte) * *query_value as f64;
        }
        -dot as f32
    }

    #[inline]
    fn scalar_int8_euclid(scale: f32, quantized: &[u8], query: &[f32]) -> f32 {
        let mut sum_sq = 0.0f64;
        for (byte, query_value) in quantized.iter().zip(query.iter()) {
            let delta = Self::scalar_int8_doc_value(scale, *byte) - *query_value as f64;
            sum_sq += delta * delta;
        }
        sum_sq.sqrt() as f32
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn record_len(&self) -> usize {
        self.record_len
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl MmapTurboQuantStore {
    /// Create a new HVTQ file from complete f32 rows by encoding each row with
    /// the supplied TurboProd config. The returned store is immutable/read-only.
    pub fn create_from_slices(
        path: &Path,
        vectors: &[&[f32]],
        config: &SpindleConfig,
    ) -> Result<Self, VectorError> {
        if config.mode != SpindleMode::TurboProd {
            return Err(VectorError::VectorCoreError(
                "MmapTurboQuantStore requires SpindleMode::TurboProd".into(),
            ));
        }
        if vectors.is_empty() {
            return Err(VectorError::VectorCoreError(
                "MmapTurboQuantStore::create_from_slices requires at least one vector".into(),
            ));
        }
        let dim = vectors[0].len();
        if dim == 0 {
            return Err(VectorError::VectorCoreError(
                "vector dim must be > 0".into(),
            ));
        }
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != dim {
                return Err(VectorError::VectorCoreError(format!(
                    "dim mismatch at vector {}: expected {}, got {}",
                    i,
                    dim,
                    v.len()
                )));
            }
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| VectorError::VectorCoreError(format!("mkdir failed: {}", e)))?;
        }

        let first = encode_vector(vectors[0], config)?;
        let record_len = first.len();
        if record_len == 0 {
            return Err(VectorError::InvalidVectorData);
        }
        let count = vectors.len() as u64;

        let mut file = create_fresh_file(path)
            .map_err(|e| VectorError::VectorCoreError(format!("hvtq create failed: {}", e)))?;

        let mut header = [0u8; TURBO_QUANT_HEADER_SIZE];
        header[0..4].copy_from_slice(TURBO_QUANT_MAGIC);
        header[4..8].copy_from_slice(&(dim as u32).to_le_bytes());
        header[8..16].copy_from_slice(&count.to_le_bytes());
        header[16..24].copy_from_slice(&0u64.to_le_bytes());
        header[24..28].copy_from_slice(&(record_len as u32).to_le_bytes());
        header[28..32].copy_from_slice(&0u32.to_le_bytes());
        file.write_all(&header)
            .map_err(|e| VectorError::VectorCoreError(format!("hvtq header write: {}", e)))?;
        file.write_all(&first)
            .map_err(|e| VectorError::VectorCoreError(format!("hvtq data write: {}", e)))?;

        for v in vectors.iter().skip(1) {
            let encoded = encode_vector(v, config)?;
            if encoded.len() != record_len {
                return Err(VectorError::VectorCoreError(format!(
                    "hvtq record length changed: expected {}, got {}",
                    record_len,
                    encoded.len()
                )));
            }
            file.write_all(&encoded)
                .map_err(|e| VectorError::VectorCoreError(format!("hvtq data write: {}", e)))?;
        }

        file.sync_all()
            .map_err(|e| VectorError::VectorCoreError(format!("hvtq sync: {}", e)))?;
        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| VectorError::VectorCoreError(format!("hvtq mmap: {}", e)))?
        };
        advise_random_access(&mmap);

        Ok(Self {
            path: path.to_path_buf(),
            dim,
            count,
            record_len,
            mmap,
            data_offset: TURBO_QUANT_HEADER_SIZE,
            fastscan: Mutex::new(None),
        })
    }

    /// Create a new HVTQ file by copying already-encoded TurboProd payloads.
    /// This is preferred when compact vectors already live in LMDB because it
    /// avoids a lossy decode/re-encode cycle while moving payloads into mmap.
    pub fn create_from_encoded(path: &Path, rows: &[&[u8]]) -> Result<Self, VectorError> {
        if rows.is_empty() {
            return Err(VectorError::VectorCoreError(
                "MmapTurboQuantStore::create_from_encoded requires at least one row".into(),
            ));
        }
        let first_decoded = decode_vector(rows[0])?;
        let dim = first_decoded.len();
        if dim == 0 {
            return Err(VectorError::VectorCoreError(
                "vector dim must be > 0".into(),
            ));
        }
        let record_len = rows[0].len();
        if record_len == 0 {
            return Err(VectorError::InvalidVectorData);
        }
        for (i, row) in rows.iter().enumerate() {
            if row.len() != record_len {
                return Err(VectorError::VectorCoreError(format!(
                    "hvtq record length mismatch at row {}: expected {}, got {}",
                    i,
                    record_len,
                    row.len()
                )));
            }
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| VectorError::VectorCoreError(format!("mkdir failed: {}", e)))?;
        }

        let count = rows.len() as u64;
        let mut file = create_fresh_file(path)
            .map_err(|e| VectorError::VectorCoreError(format!("hvtq create failed: {}", e)))?;

        let mut header = [0u8; TURBO_QUANT_HEADER_SIZE];
        header[0..4].copy_from_slice(TURBO_QUANT_MAGIC);
        header[4..8].copy_from_slice(&(dim as u32).to_le_bytes());
        header[8..16].copy_from_slice(&count.to_le_bytes());
        header[16..24].copy_from_slice(&0u64.to_le_bytes());
        header[24..28].copy_from_slice(&(record_len as u32).to_le_bytes());
        header[28..32].copy_from_slice(&0u32.to_le_bytes());
        file.write_all(&header)
            .map_err(|e| VectorError::VectorCoreError(format!("hvtq header write: {}", e)))?;
        for row in rows {
            file.write_all(row)
                .map_err(|e| VectorError::VectorCoreError(format!("hvtq data write: {}", e)))?;
        }
        file.sync_all()
            .map_err(|e| VectorError::VectorCoreError(format!("hvtq sync: {}", e)))?;
        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| VectorError::VectorCoreError(format!("hvtq mmap: {}", e)))?
        };
        advise_random_access(&mmap);

        Ok(Self {
            path: path.to_path_buf(),
            dim,
            count,
            record_len,
            mmap,
            data_offset: TURBO_QUANT_HEADER_SIZE,
            fastscan: Mutex::new(None),
        })
    }

    /// Open an existing HVTQ file for reads.
    pub fn open(path: &Path) -> Result<Self, VectorError> {
        let file = OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(|e| VectorError::VectorCoreError(format!("hvtq open: {}", e)))?;
        let metadata = file
            .metadata()
            .map_err(|e| VectorError::VectorCoreError(format!("hvtq metadata: {}", e)))?;
        let file_len = metadata.len() as usize;
        if file_len < TURBO_QUANT_HEADER_SIZE {
            return Err(VectorError::VectorCoreError(
                "hvtq file too small for header".into(),
            ));
        }

        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| VectorError::VectorCoreError(format!("hvtq mmap: {}", e)))?
        };
        advise_random_access(&mmap);

        if &mmap[0..4] != TURBO_QUANT_MAGIC {
            return Err(VectorError::VectorCoreError(format!(
                "hvtq magic mismatch: expected HVTQ, got {:?}",
                &mmap[0..4]
            )));
        }
        let dim = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
        let count = u64::from_le_bytes(mmap[8..16].try_into().unwrap());
        let record_len = u32::from_le_bytes(mmap[24..28].try_into().unwrap()) as usize;
        if dim == 0 || record_len == 0 {
            return Err(VectorError::InvalidVectorData);
        }
        let data_offset = TURBO_QUANT_HEADER_SIZE;
        let expected = data_offset + (count as usize) * record_len;
        if file_len < expected {
            return Err(VectorError::VectorCoreError(format!(
                "hvtq truncated: expected {} bytes, got {}",
                expected, file_len
            )));
        }

        Ok(Self {
            path: path.to_path_buf(),
            dim,
            count,
            record_len,
            mmap,
            data_offset,
            fastscan: Mutex::new(None),
        })
    }

    /// Zero-copy view of the encoded TurboProd payload at the given ordinal.
    #[inline]
    pub fn get_encoded(&self, ordinal: u64) -> Option<&[u8]> {
        if ordinal >= self.count {
            return None;
        }
        let start = self.data_offset + (ordinal as usize) * self.record_len;
        let end = start + self.record_len;
        if end > self.mmap.len() {
            return None;
        }
        Some(&self.mmap[start..end])
    }

    fn parse_record<'a>(&self, bytes: &'a [u8]) -> Result<TurboQuantRecordParts<'a>, VectorError> {
        if bytes.len() != self.record_len
            || bytes.len() < 21
            || !bytes.starts_with(b"SPN1")
            || bytes[4] != TURBO_PROD_TAG
        {
            return Err(VectorError::InvalidVectorData);
        }
        let full_dim = u32::from_le_bytes(
            bytes[5..9]
                .try_into()
                .map_err(|_| VectorError::InvalidVectorData)?,
        ) as usize;
        if full_dim != self.dim {
            return Err(VectorError::InvalidVectorData);
        }
        let payload = &bytes[9..];
        if payload.len() < 12 {
            return Err(VectorError::InvalidVectorData);
        }
        let stored_dim = u32::from_le_bytes(payload[..4].try_into().unwrap()) as usize;
        if stored_dim == 0 || stored_dim > self.dim {
            return Err(VectorError::InvalidVectorData);
        }
        let doc_norm = f32::from_le_bytes(payload[4..8].try_into().unwrap());
        let residual_norm = f32::from_le_bytes(payload[8..12].try_into().unwrap());
        let mse_packed_len = (stored_dim * 3).div_ceil(8);
        let qjl_packed_len = self.dim.div_ceil(8);
        if payload.len() != 12 + mse_packed_len + qjl_packed_len {
            return Err(VectorError::InvalidVectorData);
        }
        Ok(TurboQuantRecordParts {
            stored_dim,
            doc_norm,
            residual_norm,
            mse_bytes: &payload[12..12 + mse_packed_len],
            qjl_bytes: &payload[12 + mse_packed_len..],
        })
    }

    #[inline]
    fn packed_3bit_code(bytes: &[u8], idx: usize) -> u8 {
        let bit_pos = idx * 3;
        let byte_idx = bit_pos / 8;
        let bit_offset = bit_pos % 8;
        let mut code = (bytes[byte_idx] >> bit_offset) & 0x07;
        if bit_offset > 5 && byte_idx + 1 < bytes.len() {
            code |= (bytes[byte_idx + 1] << (8 - bit_offset)) & 0x07;
        }
        code
    }

    fn build_fastscan_index(&self) -> Result<TurboQuantFastScanIndex, VectorError> {
        let count = self.count as usize;
        let block_count = count.div_ceil(HVTQ_FASTSCAN_BLOCK);
        let first = self
            .get_encoded(0)
            .ok_or(VectorError::InvalidVectorData)
            .and_then(|bytes| self.parse_record(bytes))?;
        let stored_dim = first.stored_dim;
        let mut index = TurboQuantFastScanIndex {
            dim: self.dim,
            stored_dim,
            count,
            block_count,
            mse_codes: vec![[0u8; HVTQ_FASTSCAN_BLOCK]; block_count * stored_dim],
            mse_bitplanes: vec![[0u16; 3]; block_count * stored_dim],
            qjl_bitplanes: vec![0u16; block_count * self.dim],
            doc_norms: vec![0.0; count],
            residual_norms: vec![0.0; count],
        };

        for ordinal in 0..count {
            let bytes = self
                .get_encoded(ordinal as u64)
                .ok_or(VectorError::InvalidVectorData)?;
            let parts = self.parse_record(bytes)?;
            if parts.stored_dim != stored_dim {
                return Err(VectorError::VectorCoreError(format!(
                    "hvtq stored dim changed: expected {}, got {}",
                    stored_dim, parts.stored_dim
                )));
            }
            let block = ordinal / HVTQ_FASTSCAN_BLOCK;
            let lane = ordinal % HVTQ_FASTSCAN_BLOCK;
            let lane_mask = 1u16 << lane;
            index.doc_norms[ordinal] = parts.doc_norm;
            index.residual_norms[ordinal] = parts.residual_norm;

            let mse_base = block * stored_dim;
            for dim_idx in 0..stored_dim {
                let code = Self::packed_3bit_code(parts.mse_bytes, dim_idx);
                index.mse_codes[mse_base + dim_idx][lane] = code;
                let planes = &mut index.mse_bitplanes[mse_base + dim_idx];
                if code & 0x01 != 0 {
                    planes[0] |= lane_mask;
                }
                if code & 0x02 != 0 {
                    planes[1] |= lane_mask;
                }
                if code & 0x04 != 0 {
                    planes[2] |= lane_mask;
                }
            }

            let qjl_base = block * self.dim;
            for dim_idx in 0..self.dim {
                if (parts.qjl_bytes[dim_idx / 8] >> (dim_idx % 8)) & 1 == 1 {
                    index.qjl_bitplanes[qjl_base + dim_idx] |= lane_mask;
                }
            }
        }

        Ok(index)
    }

    pub fn score_all_fastscan(
        &self,
        prepared: &PreparedSpindleQuery,
    ) -> Option<Result<Vec<ApproximateInnerProduct>, VectorError>> {
        let query = prepared.as_turbo_prod()?;
        Some((|| {
            let mut guard = self.fastscan.lock().map_err(|e| {
                VectorError::VectorCoreError(format!("hvtq fastscan lock poisoned: {}", e))
            })?;
            if guard.is_none() {
                *guard = Some(self.build_fastscan_index()?);
            }
            let index = guard.as_ref().unwrap();
            index.score_all(query)
        })())
    }

    #[cfg(all(test, target_arch = "aarch64"))]
    pub fn score_all_fastscan_i8_lut_neon(
        &self,
        prepared: &PreparedSpindleQuery,
    ) -> Option<Result<Vec<ApproximateInnerProduct>, VectorError>> {
        let query = prepared.as_turbo_prod()?;
        Some((|| {
            let mut guard = self.fastscan.lock().map_err(|e| {
                VectorError::VectorCoreError(format!("hvtq fastscan lock poisoned: {}", e))
            })?;
            if guard.is_none() {
                *guard = Some(self.build_fastscan_index()?);
            }
            let index = guard.as_ref().unwrap();
            index.score_all_i8_lut_neon(query)
        })())
    }

    #[cfg(target_arch = "aarch64")]
    pub fn score_all_mse_code_i8_lut_neon(
        &self,
        prepared: &PreparedSpindleQuery,
    ) -> Option<Result<Vec<ApproximateInnerProduct>, VectorError>> {
        let query = prepared.as_turbo_prod()?;
        Some((|| {
            let mut guard = self.fastscan.lock().map_err(|e| {
                VectorError::VectorCoreError(format!("hvtq fastscan lock poisoned: {}", e))
            })?;
            if guard.is_none() {
                *guard = Some(self.build_fastscan_index()?);
            }
            let index = guard.as_ref().unwrap();
            index.score_all_mse_code_i8_lut_neon(query)
        })())
    }

    #[cfg(not(target_arch = "aarch64"))]
    pub fn score_all_mse_code_i8_lut_neon(
        &self,
        _prepared: &PreparedSpindleQuery,
    ) -> Option<Result<Vec<ApproximateInnerProduct>, VectorError>> {
        None
    }

    #[inline]
    pub fn get_decoded(&self, ordinal: u64) -> Option<Vec<f32>> {
        self.get_encoded(ordinal)
            .and_then(|bytes| decode_vector(bytes).ok())
    }

    #[inline]
    pub fn score_encoded_to(
        &self,
        ordinal: u64,
        prepared: &PreparedSpindleQuery,
    ) -> Option<Result<ApproximateInnerProduct, VectorError>> {
        let bytes = self.get_encoded(ordinal)?;
        Some(match score_encoded(prepared, bytes) {
            Ok(Some(approx)) => Ok(approx),
            Ok(None) => Err(VectorError::VectorCoreError(
                "prepared query did not match HVTQ payload".into(),
            )),
            Err(err) => Err(err),
        })
    }

    #[inline]
    pub fn distance_to(
        &self,
        ordinal: u64,
        query: &[f32],
        metric: MmapDistanceMetric,
    ) -> Option<Result<f32, VectorError>> {
        if query.len() != self.dim {
            return Some(Err(VectorError::InvalidVectorLength));
        }
        let decoded = match self.get_decoded(ordinal) {
            Some(decoded) => decoded,
            None => return None,
        };
        Some(Ok(match metric {
            MmapDistanceMetric::Cosine => super::simd::cosine_f32(&decoded, query),
            MmapDistanceMetric::Dot => super::simd::dot_f32(&decoded, query),
            MmapDistanceMetric::Euclid => super::simd::euclid_f32(&decoded, query).sqrt(),
        }))
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn record_len(&self) -> usize {
        self.record_len
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// One-shot conversion of a legacy HVEC file to a fresh HVS8 file.
///
/// Scans the source `MmapVectorStore` twice, fits SQ8 params on the full
/// distribution, and writes the quantized result to `dst_path` without
/// materializing the full source in heap memory.
/// The source file is left untouched — the caller is responsible for the
/// atomic rename + cleanup of the legacy file once the new one is published.
///
/// Returned `MmapQuantizedStore` is read-ready. This is the foundation for
/// the segment-publish path: when a segment transitions Building → Indexed,
/// the merge optimizer will call this to lay down a quantized segment file
/// in parallel with the LMDB metadata swap.
///
/// **Customer-safe:** does not modify the source. A failure leaves the
/// destination file in whatever state it reached (caller must clean up via
/// fs::remove_file if `Err`). The source HVEC is byte-identical at exit.
pub fn convert_hvec_to_hvs8(
    src: &MmapVectorStore,
    dst_path: &Path,
) -> Result<MmapQuantizedStore, VectorError> {
    let count = src.count();
    if count == 0 {
        return Err(VectorError::VectorCoreError(
            "convert_hvec_to_hvs8: source store is empty".into(),
        ));
    }
    MmapQuantizedStore::create_from_hvec(dst_path, src)
}

/// Dispatch enum for the mmap-backed vector sidecar.
///
/// `Hvec` is the legacy f32 layout (mutable, supports append). `Hvs8` is the
/// scalar-quantized read-only layout produced by the merge optimizer at segment
/// publish. `Hvtq` is the compact TurboProd payload layout for merged
/// TurboProd segments. The dispatch lets callers ask for a vector by ordinal
/// without caring about the on-disk format; lossy stores decode per-call.
///
/// Append/flush are HVEC-only by construction: HVS8 is created once from a
/// finalized HVEC source and is then immutable. Calling `append` on an
/// `Hvs8`-backed store returns an error rather than silently dropping writes.
pub enum MmapBackend {
    Hvec(MmapVectorStore),
    Hvs8(MmapQuantizedStore),
    Hvtq(MmapTurboQuantStore),
    Hspn(MmapSpindleStore),
}

impl MmapBackend {
    #[inline]
    pub fn dim(&self) -> usize {
        match self {
            Self::Hvec(s) => s.dim(),
            Self::Hvs8(s) => s.dim(),
            Self::Hvtq(s) => s.dim(),
            Self::Hspn(s) => s.dim(),
        }
    }

    /// Score every HVTQ ordinal through the block-transposed FastScan cache.
    /// Returns `None` for non-HVTQ backends or non-TurboProd prepared queries.
    #[inline]
    pub fn score_all_fastscan(
        &self,
        prepared: &PreparedSpindleQuery,
    ) -> Option<Result<Vec<ApproximateInnerProduct>, VectorError>> {
        match self {
            Self::Hvtq(s) => s.score_all_fastscan(prepared),
            Self::Hvec(_) | Self::Hvs8(_) | Self::Hspn(_) => None,
        }
    }

    /// Fast first-pass HVTQ scan: block-transposed byte codes + NEON LUT.
    /// Scores only the TurboProd MSE lane; callers should exact-rerank the
    /// returned candidates with `score_encoded_to`.
    #[inline]
    pub fn score_all_mse_code_i8_lut_neon(
        &self,
        prepared: &PreparedSpindleQuery,
    ) -> Option<Result<Vec<ApproximateInnerProduct>, VectorError>> {
        match self {
            Self::Hvtq(s) => s.score_all_mse_code_i8_lut_neon(prepared),
            Self::Hvec(_) | Self::Hvs8(_) | Self::Hspn(_) => None,
        }
    }

    #[inline]
    pub fn count(&self) -> u64 {
        match self {
            Self::Hvec(s) => s.count(),
            Self::Hvs8(s) => s.count(),
            Self::Hvtq(s) => s.count(),
            Self::Hspn(s) => s.count(),
        }
    }

    /// Returns the vector at `ordinal` as an owned `Vec<f32>`.
    /// HVEC copies from the mmap; HVS8 dequantizes from u8 + per-dim params;
    /// HVTQ decodes the stored TurboProd payload.
    /// Returns `None` if `ordinal` is out of bounds.
    #[inline]
    pub fn get_vec(&self, ordinal: u64) -> Option<Vec<f32>> {
        match self {
            Self::Hvec(s) => s.get(ordinal).map(|v| v.to_vec()),
            Self::Hvs8(s) => s.get_dequantized(ordinal),
            Self::Hvtq(s) => s.get_decoded(ordinal),
            Self::Hspn(s) => s.get_decoded(ordinal),
        }
    }

    /// Returns the encoded vector payload at `ordinal` when the sidecar itself
    /// owns encoded Spindle bytes. This is the no-duplication path for compact
    /// TurboProd merge targets: LMDB stores markers and HVTQ stores payloads.
    #[inline]
    pub fn get_encoded_vec(&self, ordinal: u64) -> Option<Vec<u8>> {
        match self {
            Self::Hvtq(s) => s.get_encoded(ordinal).map(|bytes| bytes.to_vec()),
            Self::Hspn(s) => s.get_encoded(ordinal).map(|bytes| bytes.to_vec()),
            Self::Hvec(_) | Self::Hvs8(_) => None,
        }
    }

    /// Score an encoded sidecar row without first copying it out of the mmap.
    #[inline]
    pub fn score_encoded_to(
        &self,
        ordinal: u64,
        prepared: &PreparedSpindleQuery,
    ) -> Option<Result<ApproximateInnerProduct, VectorError>> {
        match self {
            Self::Hvtq(s) => s.score_encoded_to(ordinal, prepared),
            Self::Hspn(s) => s.score_encoded_to(ordinal, prepared),
            Self::Hvec(_) | Self::Hvs8(_) => None,
        }
    }

    /// Borrowed-slice access to an HVEC vector for zero-copy scoring on the
    /// search hot path. Returns `Some(f(slice))` for HVEC backends, where
    /// `slice` is a `&[f32]` borrowed directly from the mmap. Returns `None`
    /// for HVS8 (which must dequantize) or when the ordinal is out of bounds.
    ///
    /// The closure form keeps the mmap borrow scoped to the call — no
    /// lifetimes leak to callers — while still letting them avoid the
    /// per-neighbor `Vec<f32>` allocation that `get_vec` does. Distance
    /// kernels that take `&[f32]` (e.g. `simd::cosine_f32`) compose
    /// naturally inside the closure.
    #[inline]
    pub fn with_vec_slice<F, R>(&self, ordinal: u64, f: F) -> Option<R>
    where
        F: FnOnce(&[f32]) -> R,
    {
        match self {
            Self::Hvec(s) => s.get(ordinal).map(f),
            Self::Hvs8(_) | Self::Hvtq(_) | Self::Hspn(_) => None,
        }
    }

    /// Score a row without allocating an owned vector. HVEC borrows the f32 mmap
    /// row directly; HVS8 streams quantized bytes through its min/scale table.
    #[inline]
    pub fn distance_to(
        &self,
        ordinal: u64,
        query: &[f32],
        metric: MmapDistanceMetric,
    ) -> Option<Result<f32, VectorError>> {
        match self {
            Self::Hvec(s) => s.get(ordinal).map(|slice| {
                if slice.len() != query.len() {
                    return Err(VectorError::InvalidVectorLength);
                }
                Ok(match metric {
                    MmapDistanceMetric::Cosine => super::simd::cosine_f32(slice, query),
                    MmapDistanceMetric::Dot => super::simd::dot_f32(slice, query),
                    MmapDistanceMetric::Euclid => super::simd::euclid_f32(slice, query).sqrt(),
                })
            }),
            Self::Hvs8(s) => s.distance_to(ordinal, query, metric),
            Self::Hvtq(s) => s.distance_to(ordinal, query, metric),
            Self::Hspn(s) => s.distance_to(ordinal, query, metric),
        }
    }

    /// Append a vector. HVEC only. Quantized backends are read-only.
    #[inline]
    pub fn append(&mut self, data: &[f32]) -> Result<u64, VectorError> {
        match self {
            Self::Hvec(s) => s.append(data),
            Self::Hvs8(_) | Self::Hvtq(_) | Self::Hspn(_) => Err(VectorError::VectorCoreError(
                "cannot append to read-only mmap backend".into(),
            )),
        }
    }

    /// Flush pending writes. HVS8 is immutable so flush is a no-op.
    #[inline]
    pub fn flush(&mut self) -> Result<(), VectorError> {
        match self {
            Self::Hvec(s) => s.flush(),
            Self::Hvs8(_) | Self::Hvtq(_) | Self::Hspn(_) => Ok(()),
        }
    }

    #[inline]
    pub fn refresh_read_mmap(&mut self) -> Result<(), VectorError> {
        match self {
            Self::Hvec(s) => s.refresh_read_mmap(),
            Self::Hvs8(_) | Self::Hvtq(_) | Self::Hspn(_) => Ok(()),
        }
    }

    /// Path to the on-disk file backing this store.
    #[inline]
    pub fn path(&self) -> &Path {
        match self {
            Self::Hvec(s) => s.path(),
            Self::Hvs8(s) => s.path(),
            Self::Hvtq(s) => s.path(),
            Self::Hspn(s) => s.path(),
        }
    }

    #[inline]
    pub fn format_label(&self) -> &'static str {
        match self {
            Self::Hvec(_) => "hvec",
            Self::Hvs8(_) => "hvs8",
            Self::Hvtq(_) => "hvtq",
            Self::Hspn(_) => "hspn",
        }
    }

    /// True if this backend is the read-only HVS8 quantized format.
    #[inline]
    pub fn is_hvs8(&self) -> bool {
        matches!(self, Self::Hvs8(_))
    }

    /// True if this backend is the read-only HVTQ TurboProd sidecar format.
    #[inline]
    pub fn is_hvtq(&self) -> bool {
        matches!(self, Self::Hvtq(_))
    }

    #[inline]
    pub fn is_hspn(&self) -> bool {
        matches!(self, Self::Hspn(_))
    }

    /// True if this backend can provide exact f32 rows. Lossy sidecars are not
    /// suitable as keep_original storage.
    #[inline]
    pub fn is_exact(&self) -> bool {
        matches!(self, Self::Hvec(_))
    }

    /// Convert an HVEC backend in-place to HVS8.
    ///
    /// Behavior:
    ///   - If already HVS8, returns `Ok(false)` (no-op).
    ///   - Otherwise: flushes the source HVEC, writes a sibling `.hvs8`
    ///     file derived from the source path, replaces `self` with the
    ///     new HVS8 backend, and best-effort deletes the source `.hvec`.
    ///   - The source path's extension is replaced with `.hvs8`. If the
    ///     source has no extension or a non-`.hvec` extension, the new
    ///     file is `<source>.hvs8` (extension appended).
    ///
    /// Caller must hold the write lock — the function takes `&mut self`.
    /// The conversion is in-process: failure leaves the backend
    /// unchanged (still Hvec) and the destination file may be partial
    /// (caller can retry, or the segment reaper will reclaim it).
    pub fn convert_to_hvs8(&mut self) -> Result<bool, VectorError> {
        let src = match self {
            Self::Hvs8(_) | Self::Hvtq(_) | Self::Hspn(_) => return Ok(false),
            Self::Hvec(s) => s,
        };
        src.flush()?;
        if src.count() == 0 {
            return Ok(false);
        }

        // Derive destination path: same stem with .hvs8 extension.
        let src_path = src.path().to_path_buf();
        let dst_path = src_path.with_extension("hvs8");

        let quantized = convert_hvec_to_hvs8(src, &dst_path)?;
        src.advise_done_with_pages();

        // Swap backend; the old MmapVectorStore is dropped, releasing
        // the file handle. Then best-effort remove the legacy .hvec so
        // the next open doesn't see two formats for the same segment.
        *self = Self::Hvs8(quantized);
        let _ = fs::remove_file(&src_path);
        let _ = fs::remove_file(src_path.with_extension("hvtq"));

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn create_append_read_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.hvec");
        let dim = 4;

        let mut store = MmapVectorStore::create(&path, dim).unwrap();
        let v0 = vec![1.0f32, 2.0, 3.0, 4.0];
        let v1 = vec![5.0f32, 6.0, 7.0, 8.0];

        let ord0 = store.append(&v0).unwrap();
        let ord1 = store.append(&v1).unwrap();
        assert_eq!(ord0, 0);
        assert_eq!(ord1, 1);

        store.flush().unwrap();

        let got0 = store.get(0).unwrap();
        let got1 = store.get(1).unwrap();
        assert_eq!(got0, &v0[..]);
        assert_eq!(got1, &v1[..]);
        assert!(store.get(2).is_none());
    }

    #[test]
    fn create_from_slices_writes_exact_hvec_without_append_loop() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("exact.hvec");
        let rows = [
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0f32, 6.0, 7.0, 8.0],
            [-1.0f32, -2.0, -3.0, -4.0],
        ];
        let slices = rows.iter().map(|row| row.as_slice()).collect::<Vec<_>>();

        let store = MmapVectorStore::create_from_slices(&path, &slices).unwrap();

        assert_eq!(store.dim(), 4);
        assert_eq!(store.count(), 3);
        assert_eq!(store.get(0).unwrap(), rows[0].as_slice());
        assert_eq!(store.get(1).unwrap(), rows[1].as_slice());
        assert_eq!(store.get(2).unwrap(), rows[2].as_slice());
    }

    #[test]
    fn reopen_preserves_data() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("persist.hvec");
        let dim = 3;

        {
            let mut store = MmapVectorStore::create(&path, dim).unwrap();
            store.append(&[1.0, 2.0, 3.0]).unwrap();
            store.append(&[4.0, 5.0, 6.0]).unwrap();
            store.flush().unwrap();
        }

        let store = MmapVectorStore::open(&path).unwrap();
        assert_eq!(store.count(), 2);
        assert_eq!(store.dim(), 3);
        assert_eq!(store.get(0).unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(store.get(1).unwrap(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn recreate_same_path_preserves_existing_mmap_holders() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("reuse.hvec");

        let mut old = MmapVectorStore::create(&path, 2).unwrap();
        old.append(&[1.0, 2.0]).unwrap();
        old.flush().unwrap();
        assert_eq!(old.get(0).unwrap(), &[1.0, 2.0]);

        let mut replacement = MmapVectorStore::create(&path, 2).unwrap();
        replacement.append(&[3.0, 4.0]).unwrap();
        replacement.flush().unwrap();

        assert_eq!(old.get(0).unwrap(), &[1.0, 2.0]);
        assert_eq!(replacement.get(0).unwrap(), &[3.0, 4.0]);

        drop(replacement);
        let reopened = MmapVectorStore::open(&path).unwrap();
        assert_eq!(reopened.get(0).unwrap(), &[3.0, 4.0]);
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad.hvec");

        let mut store = MmapVectorStore::create(&path, 4).unwrap();
        let result = store.append(&[1.0, 2.0, 3.0]); // dim 3, expected 4
        assert!(result.is_err());
    }

    #[test]
    fn open_or_create_works() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("ooc.hvec");

        // Create
        {
            let mut store = MmapVectorStore::open_or_create(&path, 2).unwrap();
            store.append(&[1.0, 2.0]).unwrap();
            store.flush().unwrap();
        }

        // Re-open
        let store = MmapVectorStore::open_or_create(&path, 2).unwrap();
        assert_eq!(store.count(), 1);
        assert_eq!(store.get(0).unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn large_batch_768_dim() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("large.hvec");
        let dim = 768;
        let n = 1000;

        let mut store = MmapVectorStore::create(&path, dim).unwrap();
        let mut rng = 42u64;
        for i in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|_| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (rng >> 33) as f32 / (1u64 << 31) as f32
                })
                .collect();
            let ord = store.append(&v).unwrap();
            assert_eq!(ord, i);
        }
        store.flush().unwrap();

        assert_eq!(store.count(), n);
        // Verify first and last
        assert!(store.get(0).is_some());
        assert!(store.get(n - 1).is_some());
        assert!(store.get(n).is_none());
    }

    #[test]
    fn append_after_flush_preserves_existing_vectors() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("append_after_flush.hvec");

        let mut store = MmapVectorStore::create(&path, 2).unwrap();
        assert_eq!(store.append(&[1.0, 2.0]).unwrap(), 0);
        store.flush().unwrap();

        assert_eq!(store.append(&[3.0, 4.0]).unwrap(), 1);
        store.flush().unwrap();

        assert_eq!(store.count(), 2);
        assert_eq!(store.get(0).unwrap(), &[1.0, 2.0]);
        assert_eq!(store.get(1).unwrap(), &[3.0, 4.0]);
    }

    #[test]
    fn reopen_then_append_preserves_existing_vectors() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("reopen_append.hvec");

        {
            let mut store = MmapVectorStore::create(&path, 2).unwrap();
            store.append(&[1.0, 2.0]).unwrap();
            store.flush().unwrap();
        }

        let mut store = MmapVectorStore::open(&path).unwrap();
        assert_eq!(store.append(&[3.0, 4.0]).unwrap(), 1);
        store.flush().unwrap();

        assert_eq!(store.count(), 2);
        assert_eq!(store.get(0).unwrap(), &[1.0, 2.0]);
        assert_eq!(store.get(1).unwrap(), &[3.0, 4.0]);
    }

    #[test]
    fn quantized_create_open_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.hvs8");
        let dim = 4;

        let v0: Vec<f32> = vec![0.10, 0.20, 0.30, 0.40];
        let v1: Vec<f32> = vec![0.90, 0.80, 0.70, 0.60];
        let v2: Vec<f32> = vec![0.50, 0.50, 0.50, 0.50];
        let vecs = vec![v0.clone(), v1.clone(), v2.clone()];

        let store = MmapQuantizedStore::create_from(&path, &vecs).unwrap();
        assert_eq!(store.count(), 3);
        assert_eq!(store.dim(), dim);
        assert_eq!(store.mins().len(), dim);
        assert_eq!(store.scales().len(), dim);

        // Quantized bytes should be non-empty for each vector.
        let q0 = store.get_quantized(0).unwrap();
        assert_eq!(q0.len(), dim);

        // Out-of-bounds returns None.
        assert!(store.get_quantized(3).is_none());

        // Re-open and verify the data round-trips.
        drop(store);
        let reopened = MmapQuantizedStore::open(&path).unwrap();
        assert_eq!(reopened.count(), 3);
        assert_eq!(reopened.dim(), dim);

        let dq0 = reopened.get_dequantized(0).unwrap();
        let dq1 = reopened.get_dequantized(1).unwrap();
        let dq2 = reopened.get_dequantized(2).unwrap();
        assert_eq!(dq0.len(), dim);
        // SQ8 is lossy with ~1/255 of the per-dim range as quantization error.
        // Check the dequantized values are close to the originals.
        for i in 0..dim {
            assert!(
                (dq0[i] - v0[i]).abs() < 0.005,
                "v0[{}] dequant drift: orig={} got={}",
                i,
                v0[i],
                dq0[i]
            );
            assert!(
                (dq1[i] - v1[i]).abs() < 0.005,
                "v1[{}] dequant drift: orig={} got={}",
                i,
                v1[i],
                dq1[i]
            );
            assert!(
                (dq2[i] - v2[i]).abs() < 0.005,
                "v2[{}] dequant drift: orig={} got={}",
                i,
                v2[i],
                dq2[i]
            );
        }
    }

    #[test]
    fn spindle_encoded_scalar_int8_scores_from_mmap_row() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.hspn");
        let config = SpindleConfig::scalar_int8();
        let rows = [
            [1.0f32, 0.0, 0.0, 0.0],
            [0.0f32, 1.0, 0.0, 0.0],
            [-0.25f32, 0.5, 0.75, -1.0],
        ];
        let encoded = rows
            .iter()
            .map(|row| encode_vector(row, &config).unwrap())
            .collect::<Vec<_>>();
        let encoded_slices = encoded.iter().map(|row| row.as_slice()).collect::<Vec<_>>();

        let store = MmapSpindleStore::create_from_encoded(&path, &encoded_slices).unwrap();

        assert_eq!(store.dim(), 4);
        assert_eq!(store.count(), 3);
        assert_eq!(store.get_encoded(1).unwrap(), encoded[1].as_slice());
        let query = decode_vector(&encode_vector(&rows[1], &config).unwrap()).unwrap();
        let decoded = decode_vector(&encoded[1]).unwrap();
        let expected = crate::helix_engine::vector_core::simd::cosine_f32(&decoded, &query);
        let got = store
            .distance_to(1, &query, MmapDistanceMetric::Cosine)
            .unwrap()
            .unwrap();
        assert!((got - expected).abs() < 1e-6);

        drop(store);
        let reopened = MmapSpindleStore::open(&path).unwrap();
        assert_eq!(reopened.record_len(), encoded[0].len());
        assert_eq!(
            reopened.get_decoded(2).unwrap(),
            decode_vector(&encoded[2]).unwrap()
        );
    }

    #[test]
    fn quantized_peek_magic_distinguishes_formats() {
        let dir = TempDir::new().unwrap();
        let hvec_path = dir.path().join("legacy.hvec");
        let hvs8_path = dir.path().join("quantized.hvs8");

        // Legacy HVEC file.
        let mut legacy = MmapVectorStore::create(&hvec_path, 2).unwrap();
        legacy.append(&[1.0, 2.0]).unwrap();
        legacy.flush().unwrap();

        // New HVS8 file.
        let _quantized =
            MmapQuantizedStore::create_from(&hvs8_path, &[vec![0.1f32, 0.2], vec![0.3, 0.4]])
                .unwrap();

        assert_eq!(&MmapQuantizedStore::peek_magic(&hvec_path).unwrap(), MAGIC);
        assert_eq!(
            &MmapQuantizedStore::peek_magic(&hvs8_path).unwrap(),
            QUANTIZED_MAGIC
        );
    }

    #[test]
    fn turbo_quant_create_open_roundtrip_and_scores() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.hvtq");
        let config = SpindleConfig::turbo_prod_compact(4);
        let rows: Vec<Vec<f32>> = vec![
            vec![0.10, 0.20, 0.30, 0.40],
            vec![0.90, 0.80, 0.70, 0.60],
            vec![-0.20, 0.15, -0.35, 0.55],
        ];
        let slices = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();

        let store = MmapTurboQuantStore::create_from_slices(&path, &slices, &config).unwrap();
        assert_eq!(store.count(), 3);
        assert_eq!(store.dim(), 4);
        assert!(store.record_len() > 0);
        assert!(store.get_encoded(0).unwrap().starts_with(b"SPN1"));
        assert_eq!(store.get_decoded(1).unwrap().len(), 4);
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            (TURBO_QUANT_HEADER_SIZE + rows.len() * store.record_len()) as u64
        );

        drop(store);
        let reopened = MmapTurboQuantStore::open(&path).unwrap();
        assert_eq!(reopened.count(), 3);
        assert_eq!(reopened.dim(), 4);
        assert_eq!(reopened.get_decoded(2).unwrap().len(), 4);

        let query = vec![0.10, 0.20, 0.30, 0.40];
        let prepared =
            crate::helix_engine::vector_core::spindle::prepare_query(&query, &config).unwrap();
        let approx = reopened.score_encoded_to(0, &prepared).unwrap().unwrap();
        assert!(approx.unit_dot.is_finite());
        assert!(approx.query_norm.is_finite());
        assert!(approx.doc_norm.is_finite());
    }

    #[test]
    fn turbo_quant_fastscan_matches_row_scorer() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("fastscan.hvtq");
        let config = SpindleConfig::turbo_prod_compact(16);
        let rows: Vec<Vec<f32>> = (0..19)
            .map(|row| {
                (0..16)
                    .map(|dim| ((row * 17 + dim * 31) as f32 * 0.013).sin())
                    .collect()
            })
            .collect();
        let slices = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let store = MmapTurboQuantStore::create_from_slices(&path, &slices, &config).unwrap();
        let query: Vec<f32> = (0..16).map(|dim| (dim as f32 * 0.071).cos()).collect();
        let prepared =
            crate::helix_engine::vector_core::spindle::prepare_query(&query, &config).unwrap();
        let fast = store.score_all_fastscan(&prepared).unwrap().unwrap();

        assert_eq!(fast.len(), rows.len());
        for (ordinal, fast_score) in fast.iter().enumerate() {
            let row_score = store
                .score_encoded_to(ordinal as u64, &prepared)
                .unwrap()
                .unwrap();
            assert!(
                (fast_score.unit_dot - row_score.unit_dot).abs() < 1e-9,
                "unit dot mismatch at ordinal {ordinal}: fast={} row={}",
                fast_score.unit_dot,
                row_score.unit_dot
            );
            assert_eq!(fast_score.query_norm, row_score.query_norm);
            assert_eq!(fast_score.doc_norm, row_score.doc_norm);
        }
    }

    fn top_k_indices(scores: &[f64], k: usize) -> Vec<usize> {
        let mut scored = scores.iter().copied().enumerate().collect::<Vec<_>>();
        scored.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap());
        scored.into_iter().take(k).map(|(idx, _)| idx).collect()
    }

    fn top_k_indices_from_pairs(scores: &[(usize, f64)], k: usize) -> Vec<usize> {
        let mut scored = scores.to_vec();
        scored.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap());
        scored.into_iter().take(k).map(|(idx, _)| idx).collect()
    }

    fn top_k_overlap(lhs: &[usize], rhs: &[usize]) -> f64 {
        lhs.iter().filter(|idx| rhs.contains(idx)).count() as f64 / lhs.len().max(1) as f64
    }

    #[test]
    #[ignore = "local HVTQ scan benchmark; run explicitly before FastScan rollout"]
    fn bench_hvtq_current_vs_fastscan_scoring() {
        let dim = std::env::var("HELIX_HVTQ_BENCH_DIM")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(512);
        let doc_count = std::env::var("HELIX_HVTQ_BENCH_DOCS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(4096);
        let query_count = std::env::var("HELIX_HVTQ_BENCH_QUERIES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(16);
        let rerank_mult = std::env::var("HELIX_HVTQ_BENCH_RERANK_MULT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(16)
            .max(1);
        let k = 10usize;
        let rerank_k = (k * rerank_mult).min(doc_count);

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bench.hvtq");
        let config = SpindleConfig::turbo_prod_compact(dim);
        let rows: Vec<Vec<f32>> = (0..doc_count)
            .map(|row| {
                (0..dim)
                    .map(|col| ((row * 131 + col * 17) as f32 * 0.00137).sin())
                    .collect()
            })
            .collect();
        let slices = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let store = MmapTurboQuantStore::create_from_slices(&path, &slices, &config).unwrap();
        let queries: Vec<Vec<f32>> = (0..query_count)
            .map(|row| {
                (0..dim)
                    .map(|col| ((row * 97 + col * 29) as f32 * 0.00211).cos())
                    .collect()
            })
            .collect();
        let warm_prepared =
            crate::helix_engine::vector_core::spindle::prepare_query(&queries[0], &config).unwrap();
        let _ = store.score_all_fastscan(&warm_prepared).unwrap().unwrap();

        let row_started = std::time::Instant::now();
        let mut row_checksum = 0.0f64;
        let mut row_topk = Vec::with_capacity(query_count);
        for query in &queries {
            let prepared =
                crate::helix_engine::vector_core::spindle::prepare_query(query, &config).unwrap();
            let mut query_scores = Vec::with_capacity(doc_count);
            for ordinal in 0..doc_count {
                let score = store
                    .score_encoded_to(ordinal as u64, &prepared)
                    .unwrap()
                    .unwrap()
                    .unit_dot;
                row_checksum += score;
                query_scores.push(score);
            }
            row_topk.push(top_k_indices(&query_scores, k));
        }
        let row_elapsed = row_started.elapsed();

        let fast_started = std::time::Instant::now();
        let mut fast_checksum = 0.0f64;
        for query in &queries {
            let prepared =
                crate::helix_engine::vector_core::spindle::prepare_query(query, &config).unwrap();
            let scores = store.score_all_fastscan(&prepared).unwrap().unwrap();
            for score in scores {
                fast_checksum += score.unit_dot;
            }
        }
        let fast_elapsed = fast_started.elapsed();

        #[cfg(target_arch = "aarch64")]
        let (neon_elapsed, neon_checksum, neon_speedup) = {
            let neon_started = std::time::Instant::now();
            let mut checksum = 0.0f64;
            for query in &queries {
                let prepared =
                    crate::helix_engine::vector_core::spindle::prepare_query(query, &config)
                        .unwrap();
                let scores = store
                    .score_all_fastscan_i8_lut_neon(&prepared)
                    .unwrap()
                    .unwrap();
                for score in scores {
                    checksum += score.unit_dot;
                }
            }
            let elapsed = neon_started.elapsed();
            (
                format!("{:?}", elapsed),
                checksum,
                row_elapsed.as_secs_f64() / elapsed.as_secs_f64().max(f64::EPSILON),
            )
        };
        #[cfg(not(target_arch = "aarch64"))]
        let (neon_elapsed, neon_checksum, neon_speedup) = ("n/a".to_string(), fast_checksum, 0.0);

        #[cfg(target_arch = "aarch64")]
        let (code_elapsed, code_checksum, code_speedup, code_overlap) = {
            let code_started = std::time::Instant::now();
            let mut checksum = 0.0f64;
            let mut overlap = 0.0;
            for (query_idx, query) in queries.iter().enumerate() {
                let prepared =
                    crate::helix_engine::vector_core::spindle::prepare_query(query, &config)
                        .unwrap();
                let scores = store
                    .score_all_mse_code_i8_lut_neon(&prepared)
                    .unwrap()
                    .unwrap();
                let query_scores = scores
                    .into_iter()
                    .map(|score| {
                        checksum += score.unit_dot;
                        score.unit_dot
                    })
                    .collect::<Vec<_>>();
                overlap += top_k_overlap(&row_topk[query_idx], &top_k_indices(&query_scores, k));
            }
            let elapsed = code_started.elapsed();
            (
                format!("{:?}", elapsed),
                checksum,
                row_elapsed.as_secs_f64() / elapsed.as_secs_f64().max(f64::EPSILON),
                overlap / query_count as f64,
            )
        };
        #[cfg(not(target_arch = "aarch64"))]
        let (code_elapsed, code_checksum, code_speedup, code_overlap) =
            ("n/a".to_string(), fast_checksum, 0.0, 0.0);

        #[cfg(target_arch = "aarch64")]
        let (rerank_elapsed, rerank_speedup, rerank_overlap) = {
            let rerank_started = std::time::Instant::now();
            let mut overlap = 0.0;
            for (query_idx, query) in queries.iter().enumerate() {
                let prepared =
                    crate::helix_engine::vector_core::spindle::prepare_query(query, &config)
                        .unwrap();
                let scores = store
                    .score_all_mse_code_i8_lut_neon(&prepared)
                    .unwrap()
                    .unwrap();
                let code_scores = scores
                    .iter()
                    .enumerate()
                    .map(|(idx, score)| (idx, score.unit_dot))
                    .collect::<Vec<_>>();
                let candidates = top_k_indices_from_pairs(&code_scores, rerank_k);
                let mut reranked = Vec::with_capacity(candidates.len());
                for ordinal in candidates {
                    let exact = store
                        .score_encoded_to(ordinal as u64, &prepared)
                        .unwrap()
                        .unwrap()
                        .unit_dot;
                    reranked.push((ordinal, exact));
                }
                overlap += top_k_overlap(
                    &row_topk[query_idx],
                    &top_k_indices_from_pairs(&reranked, k),
                );
            }
            let elapsed = rerank_started.elapsed();
            (
                format!("{:?}", elapsed),
                row_elapsed.as_secs_f64() / elapsed.as_secs_f64().max(f64::EPSILON),
                overlap / query_count as f64,
            )
        };
        #[cfg(not(target_arch = "aarch64"))]
        let (rerank_elapsed, rerank_speedup, rerank_overlap) = ("n/a".to_string(), 0.0, 0.0);

        println!(
            "HVTQ score benchmark: row={:?} exact_fastscan={:?} exact_speedup={:.2}x neon_bitplane_i8={} neon_bitplane_speedup={:.2}x neon_code_i8={} neon_code_speedup={:.2}x code_overlap@10={:.3} code_rerank={} code_rerank_speedup={:.2}x code_rerank_overlap@10={:.3} rerank_k={} docs={} queries={} dim={} exact_delta={:.6} bitplane_delta={:.6} code_delta={:.6}",
            row_elapsed,
            fast_elapsed,
            row_elapsed.as_secs_f64() / fast_elapsed.as_secs_f64().max(f64::EPSILON),
            neon_elapsed,
            neon_speedup,
            code_elapsed,
            code_speedup,
            code_overlap,
            rerank_elapsed,
            rerank_speedup,
            rerank_overlap,
            rerank_k,
            doc_count,
            query_count,
            dim,
            (row_checksum - fast_checksum).abs(),
            (row_checksum - neon_checksum).abs(),
            (row_checksum - code_checksum).abs()
        );
    }

    #[test]
    fn quantized_rejects_empty_input() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty.hvs8");
        let res = MmapQuantizedStore::create_from(&path, &[]);
        assert!(res.is_err());
    }

    #[test]
    fn quantized_rejects_dim_mismatch() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("mismatch.hvs8");
        let res = MmapQuantizedStore::create_from(&path, &[vec![1.0, 2.0, 3.0], vec![1.0, 2.0]]);
        assert!(res.is_err());
    }

    #[test]
    fn convert_hvec_to_hvs8_roundtrip() {
        let dir = TempDir::new().unwrap();
        let hvec_path = dir.path().join("source.hvec");
        let hvs8_path = dir.path().join("converted.hvs8");
        let dim = 4;

        // Build a legacy HVEC store with a few vectors.
        let v0: Vec<f32> = vec![0.10, 0.20, 0.30, 0.40];
        let v1: Vec<f32> = vec![0.90, 0.80, 0.70, 0.60];
        let v2: Vec<f32> = vec![0.50, 0.50, 0.50, 0.50];
        let mut legacy = MmapVectorStore::create(&hvec_path, dim).unwrap();
        legacy.append(&v0).unwrap();
        legacy.append(&v1).unwrap();
        legacy.append(&v2).unwrap();
        legacy.flush().unwrap();

        // Convert.
        let converted = convert_hvec_to_hvs8(&legacy, &hvs8_path).unwrap();
        assert_eq!(converted.count(), 3);
        assert_eq!(converted.dim(), dim);

        // Dequantized values should be close to the originals.
        for (ord, orig) in [(0, &v0), (1, &v1), (2, &v2)] {
            let dq = converted.get_dequantized(ord).unwrap();
            for i in 0..dim {
                assert!(
                    (dq[i] - orig[i]).abs() < 0.005,
                    "vec {} dim {} drift: orig={} got={}",
                    ord,
                    i,
                    orig[i],
                    dq[i]
                );
            }
        }

        // Source HVEC is unchanged — byte-identical reads.
        let reread = MmapVectorStore::open(&hvec_path).unwrap();
        assert_eq!(reread.count(), 3);
        assert_eq!(reread.get(0).unwrap(), v0.as_slice());
        assert_eq!(reread.get(1).unwrap(), v1.as_slice());
        assert_eq!(reread.get(2).unwrap(), v2.as_slice());
    }

    #[test]
    fn convert_hvec_to_hvs8_rejects_empty_source() {
        let dir = TempDir::new().unwrap();
        let hvec_path = dir.path().join("empty.hvec");
        let hvs8_path = dir.path().join("out.hvs8");
        let _empty = MmapVectorStore::create(&hvec_path, 4).unwrap();
        let legacy = MmapVectorStore::open(&hvec_path).unwrap();
        let res = convert_hvec_to_hvs8(&legacy, &hvs8_path);
        assert!(res.is_err());
    }

    #[test]
    fn quantized_constant_vector_dequantizes_to_min() {
        // When all input values at a given dim are equal, scale is 0.0.
        // Dequantization must round-trip to the original constant value.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("constant.hvs8");
        let vecs = vec![
            vec![0.5f32, 0.1, 0.5, 0.9],
            vec![0.5f32, 0.2, 0.5, 0.8],
            vec![0.5f32, 0.3, 0.5, 0.7],
        ];
        let store = MmapQuantizedStore::create_from(&path, &vecs).unwrap();
        let dq = store.get_dequantized(0).unwrap();
        // Dim 0 and dim 2 are constant 0.5 — must round-trip exactly.
        assert!((dq[0] - 0.5).abs() < 1e-6);
        assert!((dq[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn quantized_distance_to_matches_dequantized_scoring() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("distance.hvs8");
        let vecs = vec![
            vec![0.1f32, -0.4, 0.5, 1.0],
            vec![0.9f32, 0.2, -0.3, 1.0],
            vec![-0.2f32, 0.8, 0.7, 1.0],
        ];
        let query = vec![0.25f32, 0.1, -0.8, 0.5];
        let store = MmapQuantizedStore::create_from(&path, &vecs).unwrap();
        let decoded = store.get_dequantized(1).unwrap();

        for metric in [
            MmapDistanceMetric::Cosine,
            MmapDistanceMetric::Dot,
            MmapDistanceMetric::Euclid,
        ] {
            let direct = store.distance_to(1, &query, metric).unwrap().unwrap();
            let expected = match metric {
                MmapDistanceMetric::Cosine => super::super::simd::cosine_f32(&decoded, &query),
                MmapDistanceMetric::Dot => super::super::simd::dot_f32(&decoded, &query),
                MmapDistanceMetric::Euclid => {
                    super::super::simd::euclid_f32(&decoded, &query).sqrt()
                }
            };
            assert!(
                (direct - expected).abs() < 1e-5,
                "metric {:?}: direct={} expected={}",
                metric,
                direct,
                expected
            );
        }

        assert!(matches!(
            store.distance_to(1, &[1.0, 2.0], MmapDistanceMetric::Cosine),
            Some(Err(VectorError::InvalidVectorLength))
        ));
        assert!(store
            .distance_to(99, &query, MmapDistanceMetric::Cosine)
            .is_none());
    }

    #[test]
    fn dispatch_hvec_supports_append_and_get_vec() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("dispatch.hvec");
        let store = MmapVectorStore::create(&path, 4).unwrap();
        let mut backend = MmapBackend::Hvec(store);

        let v0 = vec![1.0f32, 2.0, 3.0, 4.0];
        let v1 = vec![5.0f32, 6.0, 7.0, 8.0];
        let ord0 = backend.append(&v0).unwrap();
        let ord1 = backend.append(&v1).unwrap();
        backend.flush().unwrap();

        assert_eq!(ord0, 0);
        assert_eq!(ord1, 1);
        assert_eq!(backend.dim(), 4);
        assert_eq!(backend.count(), 2);
        assert_eq!(backend.get_vec(0).unwrap(), v0);
        assert_eq!(backend.get_vec(1).unwrap(), v1);
        assert!(backend.get_vec(2).is_none());
    }

    #[test]
    fn convert_to_hvs8_in_place_replaces_backend() {
        let dir = TempDir::new().unwrap();
        let hvec_path = dir.path().join("seg.hvec");

        let mut store = MmapVectorStore::create(&hvec_path, 4).unwrap();
        for i in 0..8u32 {
            let f = i as f32;
            store.append(&[f, f + 1.0, f + 2.0, f + 3.0]).unwrap();
        }
        store.flush().unwrap();

        let mut backend = MmapBackend::Hvec(store);
        assert!(!backend.is_hvs8());
        let converted = backend.convert_to_hvs8().unwrap();
        assert!(converted, "first conversion should report true");
        assert!(backend.is_hvs8());
        assert_eq!(backend.count(), 8);
        assert_eq!(backend.dim(), 4);

        // Source HVEC file should be removed; HVS8 sibling should exist.
        assert!(!hvec_path.exists(), "legacy .hvec should be deleted");
        let hvs8_path = dir.path().join("seg.hvs8");
        assert!(hvs8_path.exists(), "new .hvs8 should be on disk");

        // Idempotent: second call is a no-op.
        let again = backend.convert_to_hvs8().unwrap();
        assert!(!again, "second conversion on Hvs8 must be no-op");

        // Read path through the dispatch still works (dequantization).
        let v0 = backend.get_vec(0).unwrap();
        assert!((v0[0] - 0.0).abs() < 0.05);
        let v7 = backend.get_vec(7).unwrap();
        assert!((v7[0] - 7.0).abs() < 0.05);
    }

    #[test]
    fn convert_to_hvs8_empty_hvec_is_noop() {
        let dir = TempDir::new().unwrap();
        let hvec_path = dir.path().join("empty_segment.hvec");
        let store = MmapVectorStore::create(&hvec_path, 4).unwrap();
        let mut backend = MmapBackend::Hvec(store);

        let converted = backend.convert_to_hvs8().unwrap();

        assert!(!converted);
        assert!(!backend.is_hvs8());
        assert_eq!(backend.count(), 0);
        assert!(hvec_path.exists());
        assert!(!dir.path().join("empty_segment.hvs8").exists());
    }

    #[test]
    fn dispatch_hvs8_dequantizes_and_rejects_append() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("dispatch.hvs8");
        let vecs = vec![vec![0.0f32, 1.0, 2.0, 3.0], vec![4.0f32, 5.0, 6.0, 7.0]];
        let qstore = MmapQuantizedStore::create_from(&path, &vecs).unwrap();
        let mut backend = MmapBackend::Hvs8(qstore);

        assert_eq!(backend.dim(), 4);
        assert_eq!(backend.count(), 2);

        // Read path: dequantization preserves order and approximates values.
        let v0 = backend.get_vec(0).unwrap();
        let v1 = backend.get_vec(1).unwrap();
        assert_eq!(v0.len(), 4);
        assert_eq!(v1.len(), 4);
        // SQ8 quantization: max error per dim is roughly (range / 255).
        for (got, want) in v0.iter().zip(vecs[0].iter()) {
            assert!((got - want).abs() < 0.05, "got={} want={}", got, want);
        }

        // Write path: append on an HVS8 backend must error.
        let err = backend.append(&[0.0, 0.0, 0.0, 0.0]);
        assert!(err.is_err());

        // Flush is a no-op on HVS8 (immutable) — must succeed.
        backend.flush().unwrap();
    }

    #[test]
    fn dispatch_hvtq_decodes_scores_and_rejects_append() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("dispatch.hvtq");
        let config = SpindleConfig::turbo_prod_compact(4);
        let rows: Vec<Vec<f32>> = vec![vec![0.0, 1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0, 7.0]];
        let slices = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let tq = MmapTurboQuantStore::create_from_slices(&path, &slices, &config).unwrap();
        let mut backend = MmapBackend::Hvtq(tq);

        assert!(backend.is_hvtq());
        assert!(!backend.is_exact());
        assert_eq!(backend.dim(), 4);
        assert_eq!(backend.count(), 2);
        assert!(backend.get_encoded_vec(0).unwrap().starts_with(b"SPN1"));
        assert_eq!(backend.get_vec(1).unwrap().len(), 4);

        let query = vec![0.25, 0.5, 0.75, 1.0];
        let prepared =
            crate::helix_engine::vector_core::spindle::prepare_query(&query, &config).unwrap();
        let approx = backend.score_encoded_to(1, &prepared).unwrap().unwrap();
        assert!(approx.unit_dot.is_finite());
        assert!(backend
            .distance_to(1, &query, MmapDistanceMetric::Cosine)
            .unwrap()
            .unwrap()
            .is_finite());
        assert!(backend.append(&[0.0, 0.0, 0.0, 0.0]).is_err());
        backend.flush().unwrap();
    }

    #[test]
    fn dispatch_distance_to_scores_hvec_and_hvs8_without_owned_vector_contract() {
        let dir = TempDir::new().unwrap();
        let hvec_path = dir.path().join("dispatch_distance.hvec");
        let hvs8_path = dir.path().join("dispatch_distance.hvs8");
        let query = vec![0.25f32, 0.5, 0.75, 1.0];

        let mut hvec_store = MmapVectorStore::create(&hvec_path, 4).unwrap();
        hvec_store.append(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        hvec_store.flush().unwrap();
        let hvec = MmapBackend::Hvec(hvec_store);
        let hvec_distance = hvec
            .distance_to(0, &query, MmapDistanceMetric::Cosine)
            .unwrap()
            .unwrap();
        assert!(
            (hvec_distance - super::super::simd::cosine_f32(&[1.0, 2.0, 3.0, 4.0], &query)).abs()
                < 1e-6
        );

        let qstore = MmapQuantizedStore::create_from(
            &hvs8_path,
            &[vec![0.0f32, 1.0, 2.0, 3.0], vec![4.0f32, 5.0, 6.0, 7.0]],
        )
        .unwrap();
        let decoded = qstore.get_dequantized(1).unwrap();
        let hvs8 = MmapBackend::Hvs8(qstore);
        let hvs8_distance = hvs8
            .distance_to(1, &query, MmapDistanceMetric::Cosine)
            .unwrap()
            .unwrap();
        assert!((hvs8_distance - super::super::simd::cosine_f32(&decoded, &query)).abs() < 1e-5);
    }

    // ── with_vec_slice (zero-copy borrowed-slice scoring) ──

    #[test]
    fn with_vec_slice_hvec_passes_borrowed_slice() {
        // The HVEC path must hand the closure a slice borrowed directly
        // from the mmap (no `.to_vec()`). We can't observe pointer
        // equality from outside, but we can confirm the closure receives
        // the right bytes and that the closure's return value is
        // propagated up.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("slice.hvec");
        let mut store = MmapVectorStore::create(&path, 4).unwrap();
        store.append(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        store.append(&[5.0, 6.0, 7.0, 8.0]).unwrap();
        store.flush().unwrap();

        let backend = MmapBackend::Hvec(store);
        // Closure returns a tuple proving (a) the slice contents match
        // the appended vector and (b) the value flows back through the
        // Option wrapper unchanged.
        let (sum, len) = backend
            .with_vec_slice(1, |slice| (slice.iter().sum::<f32>(), slice.len()))
            .expect("HVEC backend must hand the closure a slice for valid ordinal");
        assert_eq!(len, 4);
        assert!((sum - (5.0 + 6.0 + 7.0 + 8.0)).abs() < 1e-6);
    }

    #[test]
    fn with_vec_slice_hvs8_returns_none() {
        // HVS8 stores quantized u8s; there is no `&[f32]` to hand back
        // without dequantizing into an owned Vec. The contract is to
        // return None so callers fall back to `get_vec` (which
        // dequantizes). This guarantee is what lets the search hot path
        // skip its own quantized-vs-unquantized branch.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("none.hvs8");
        let vecs = vec![vec![0.0f32, 1.0, 2.0, 3.0], vec![4.0f32, 5.0, 6.0, 7.0]];
        let qstore = MmapQuantizedStore::create_from(&path, &vecs).unwrap();
        let backend = MmapBackend::Hvs8(qstore);

        // The closure must NOT have run — return type is Option<()>.
        let mut closure_ran = false;
        let result: Option<()> = backend.with_vec_slice(0, |_slice| {
            closure_ran = true;
        });
        assert!(
            result.is_none(),
            "HVS8 must return None from with_vec_slice"
        );
        assert!(
            !closure_ran,
            "HVS8 must not invoke the closure — there is no borrowed slice to hand it"
        );
    }

    #[test]
    fn with_vec_slice_out_of_bounds_returns_none() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("oob.hvec");
        let mut store = MmapVectorStore::create(&path, 2).unwrap();
        store.append(&[1.0, 2.0]).unwrap();
        store.flush().unwrap();
        let backend = MmapBackend::Hvec(store);

        let mut closure_ran = false;
        let result: Option<()> = backend.with_vec_slice(99, |_slice| {
            closure_ran = true;
        });
        assert!(result.is_none(), "out-of-bounds ordinal must return None");
        assert!(
            !closure_ran,
            "closure must not run for out-of-bounds ordinal"
        );
    }
}
