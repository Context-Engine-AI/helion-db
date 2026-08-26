use serde::{Deserialize, Serialize};

use crate::helix_engine::types::VectorError;

const SPINDLE_MAGIC: &[u8; 4] = b"SPN1";
const TAG_SCALAR_INT8: u8 = 1;
const TAG_BINARY_SIGN: u8 = 2;
const TAG_TURBO_INT4: u8 = 3;
const TAG_TURBO_PROD: u8 = 4;
const TURBO_BITS: usize = 4;
const TURBO_LEVELS: usize = 1 << TURBO_BITS;
const TURBO_ROTATION_SEED: u64 = 42;
const QJL_SEED: u64 = 0x517cc1b727220a95;
const MSE_BITS: usize = 3;
pub(crate) const MSE_LEVELS: usize = 1 << MSE_BITS;
#[allow(clippy::excessive_precision)]
const MSE_CODEBOOK_STD_NORMAL: [f64; MSE_LEVELS] = [
    -2.2167734607748866,
    -1.4083505015422497,
    -0.8038037659276183,
    -0.263070330892524,
    0.263070330892524,
    0.8038037659276187,
    1.4083505015422495,
    2.2167734607748835,
];
#[allow(clippy::excessive_precision)]
const TURBO_CODEBOOK_STD_NORMAL: [f64; TURBO_LEVELS] = [
    -3.457907381250601,
    -2.8464103853069207,
    -2.3653267976222923,
    -1.9195374230213922,
    -1.4883519033012216,
    -1.0614027761225631,
    -0.6376637389868758,
    -0.21342265655580925,
    0.21342265655580925,
    0.6376637389868758,
    1.0614027761225628,
    1.4883519033012214,
    1.919537423021392,
    2.3653267976222918,
    2.846410385306922,
    3.4579073812505867,
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpindleMode {
    /// No compression — raw f32 storage.
    None,
    /// 8-bit scalar quantization. Default when compression is enabled.
    /// 127-level uniform quantization per dimension. ~8x compression, near-lossless quality.
    ScalarInt8,
    /// 1-bit sign quantization. Extreme compression (~54x at 768d), low quality.
    BinarySign,
    /// 4-bit scalar quantization with rotated Lloyd-Max codebook. ~12x compression.
    TurboInt4,
    /// TurboQuant (arXiv:2504.19874): 3-bit MSE + 1-bit QJL residual = 4 bits/dim.
    /// ~15x compression with unbiased inner product estimation. Best quality-per-bit.
    TurboProd,
}

impl SpindleMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::ScalarInt8 => "scalar_int8",
            Self::BinarySign => "binary_sign",
            Self::TurboInt4 => "turbo_int4",
            Self::TurboProd => "turbo_prod",
        }
    }
}

impl Default for SpindleMode {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpindleConfig {
    #[serde(default)]
    pub mode: SpindleMode,
    #[serde(default = "default_true")]
    pub keep_original: bool,
    #[serde(default = "default_true")]
    pub rescore: bool,
    #[serde(default = "default_oversampling")]
    pub oversampling: usize,
    #[serde(default = "default_binary_dims")]
    pub binary_dims: usize,
    #[serde(default = "default_turbo_dims")]
    pub turbo_dims: usize,
}

fn default_true() -> bool {
    true
}
fn default_oversampling() -> usize {
    16
}
fn default_binary_dims() -> usize {
    256
}
fn default_turbo_dims() -> usize {
    768
}

impl Default for SpindleConfig {
    fn default() -> Self {
        Self {
            mode: SpindleMode::TurboProd,
            keep_original: true,
            rescore: true,
            oversampling: default_oversampling(),
            binary_dims: default_binary_dims(),
            turbo_dims: default_turbo_dims(),
        }
    }
}

impl SpindleConfig {
    pub fn is_enabled(&self) -> bool {
        self.mode != SpindleMode::None
    }

    pub fn oversampling(&self) -> usize {
        self.oversampling.max(1)
    }

    /// Recommended default: 8-bit scalar quantization. ~8x compression, near-lossless.
    pub fn scalar_int8() -> Self {
        Self {
            mode: SpindleMode::ScalarInt8,
            keep_original: false,
            rescore: false,
            ..Self::default()
        }
    }

    /// Memory-optimized: 4-bit TurboProd. ~15x compression, unbiased IP.
    /// Use with `keep_original: true` + rescore for int8-quality recall.
    pub fn turbo_prod(dim: usize) -> Self {
        Self {
            mode: SpindleMode::TurboProd,
            keep_original: true,
            rescore: true,
            turbo_dims: dim,
            ..Self::default()
        }
    }

    /// Memory-optimized without original retention: ~15x compression, lower recall.
    pub fn turbo_prod_compact(dim: usize) -> Self {
        Self {
            mode: SpindleMode::TurboProd,
            keep_original: false,
            rescore: false,
            turbo_dims: dim,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct ApproximateInnerProduct {
    pub unit_dot: f64,
    pub query_norm: f64,
    pub doc_norm: f64,
}

pub enum PreparedSpindleQuery {
    None,
    Binary(PreparedBinaryQuery),
    Turbo(PreparedTurboQuery),
    TurboProd(PreparedTurboProdQuery),
}

impl PreparedSpindleQuery {
    pub(crate) fn as_turbo_prod(&self) -> Option<&PreparedTurboProdQuery> {
        match self {
            Self::TurboProd(query) => Some(query),
            _ => None,
        }
    }
}

pub struct PreparedBinaryQuery {
    bits: Vec<u8>,
    stored_dim: usize,
    query_norm: f64,
}

pub struct PreparedTurboQuery {
    binary_bits: Vec<u8>,
    binary_dim: usize,
    q_even: Vec<f64>,
    q_odd: Vec<f64>,
    query_norm: f64,
    int4_dim: usize,
}

pub struct PreparedTurboProdQuery {
    pub(crate) q_unit: Vec<f64>,
    pub(crate) q_transformed: Vec<f64>,
    pub(crate) mse_lut: Vec<[f64; MSE_LEVELS]>,
    pub(crate) query_norm: f64,
    pub(crate) mse_dim: usize,
}

pub fn encode_vector(data: &[f32], config: &SpindleConfig) -> Result<Vec<u8>, VectorError> {
    // Widen to f64 for internal math precision
    let data_f64: Vec<f64> = data.iter().map(|&v| v as f64).collect();
    match config.mode {
        SpindleMode::None => Ok(encode_raw_vector(data)),
        SpindleMode::ScalarInt8 => encode_scalar_int8(&data_f64),
        SpindleMode::BinarySign => encode_binary_sign(&data_f64, config.binary_dims),
        SpindleMode::TurboInt4 => {
            encode_turbo_int4(&data_f64, config.binary_dims, config.turbo_dims)
        }
        SpindleMode::TurboProd => encode_turbo_prod(&data_f64, config.turbo_dims),
    }
}

pub fn decode_vector(bytes: &[u8]) -> Result<Vec<f32>, VectorError> {
    if !is_spindle_payload(bytes) {
        return decode_raw_vector(bytes);
    }

    if bytes.len() < 9 {
        return decode_raw_vector(bytes);
    }

    let tag = bytes[4];
    let dim = read_u32(bytes, 5)? as usize;
    let payload = &bytes[9..];

    let f64_data = match tag {
        TAG_SCALAR_INT8 => decode_scalar_int8_f64(dim, payload),
        TAG_BINARY_SIGN => decode_binary_sign_f64(dim, payload),
        TAG_TURBO_INT4 => decode_turbo_int4_f64(dim, payload),
        TAG_TURBO_PROD => decode_turbo_prod_f64(dim, payload),
        _ => return decode_raw_vector(bytes),
    }?;
    Ok(f64_data.into_iter().map(|v| v as f32).collect())
}

pub fn is_spindle_payload(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && &bytes[..4] == SPINDLE_MAGIC
}

pub fn project_for_search(data: &[f32], config: &SpindleConfig) -> Result<Vec<f32>, VectorError> {
    match config.mode {
        SpindleMode::None => Ok(data.to_vec()),
        // Asymmetric: full-precision rotated query vs quantized stored docs (per TurboQuant paper).
        // Decode round-trip would quantize the query too, losing the asymmetric advantage.
        SpindleMode::TurboInt4 | SpindleMode::TurboProd => {
            let data_f64: Vec<f64> = data.iter().map(|&v| v as f64).collect();
            let (rotated, norm) = rotate_and_normalize(&data_f64);
            Ok(rotated.into_iter().map(|v| (v * norm) as f32).collect())
        }
        _ => decode_vector(&encode_vector(data, config)?),
    }
}

pub fn prepare_query(
    query: &[f32],
    config: &SpindleConfig,
) -> Result<PreparedSpindleQuery, VectorError> {
    // Widen to f64 for internal precision math
    let query_f64: Vec<f64> = query.iter().map(|&v| v as f64).collect();
    match config.mode {
        SpindleMode::BinarySign => {
            let stored_dim = config.binary_dims.min(query.len()).max(1);
            let bits = pack_bits(
                query_f64
                    .iter()
                    .take(stored_dim)
                    .map(|value| *value >= 0.0)
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
            Ok(PreparedSpindleQuery::Binary(PreparedBinaryQuery {
                bits,
                stored_dim,
                query_norm: l2_norm(&query_f64),
            }))
        }
        SpindleMode::TurboInt4 => {
            let binary_dim = config.binary_dims.min(query.len()).max(1);
            let int4_dim = config.turbo_dims.min(query.len()).max(1);
            let (rotated, query_norm) = rotate_and_normalize(&query_f64);
            let binary_bits = pack_bits(
                rotated
                    .iter()
                    .take(binary_dim)
                    .map(|value| *value >= 0.0)
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
            let pair_count = int4_dim.div_ceil(2);
            let mut q_even = Vec::with_capacity(pair_count);
            let mut q_odd = Vec::with_capacity(pair_count);
            for idx in 0..pair_count {
                q_even.push(rotated[idx * 2]);
                q_odd.push(rotated.get(idx * 2 + 1).copied().unwrap_or(0.0));
            }
            Ok(PreparedSpindleQuery::Turbo(PreparedTurboQuery {
                binary_bits,
                binary_dim,
                q_even,
                q_odd,
                query_norm,
                int4_dim,
            }))
        }
        SpindleMode::TurboProd => {
            let mse_dim = config.turbo_dims.min(query.len()).max(1);
            let (rotated, query_norm) = rotate_and_normalize(&query_f64);
            let q_transformed = srht_forward(&rotated, QJL_SEED);
            let mse_lut = turbo_prod_mse_lut(&rotated, mse_dim);
            Ok(PreparedSpindleQuery::TurboProd(PreparedTurboProdQuery {
                q_unit: rotated,
                q_transformed,
                mse_lut,
                query_norm,
                mse_dim,
            }))
        }
        _ => Ok(PreparedSpindleQuery::None),
    }
}

pub fn score_encoded(
    prepared: &PreparedSpindleQuery,
    bytes: &[u8],
) -> Result<Option<ApproximateInnerProduct>, VectorError> {
    match prepared {
        PreparedSpindleQuery::None => Ok(None),
        PreparedSpindleQuery::Binary(query) => score_binary_payload(query, bytes).map(Some),
        PreparedSpindleQuery::Turbo(query) => score_turbo_payload(query, bytes).map(Some),
        PreparedSpindleQuery::TurboProd(query) => score_turbo_prod_payload(query, bytes).map(Some),
    }
}

fn encode_raw_vector(data: &[f32]) -> Vec<u8> {
    // Store in native byte order for zero-copy decode on the same platform.
    // Safety: f32 has no padding and every bit pattern is valid.
    let byte_len = data.len() * std::mem::size_of::<f32>();
    let mut bytes = Vec::with_capacity(byte_len);
    unsafe {
        let ptr = data.as_ptr() as *const u8;
        bytes.extend_from_slice(std::slice::from_raw_parts(ptr, byte_len));
    }
    bytes
}

fn decode_raw_vector(bytes: &[u8]) -> Result<Vec<f32>, VectorError> {
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(VectorError::InvalidVectorData);
    }

    let n = bytes.len() / std::mem::size_of::<f32>();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Fast path: try native byte order first (zero-copy reinterpret).
    // Check first few values for reasonableness.
    let native_data: Vec<f32> = {
        let mut v = vec![0f32; n];
        // Safety: copying raw bytes into a properly aligned Vec<f32>.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), v.as_mut_ptr() as *mut u8, bytes.len());
        }
        v
    };
    let sample_count = n.min(8);
    let native_ok = native_data[..sample_count]
        .iter()
        .all(|v| v.is_finite() && v.abs() < 1e10);
    if native_ok {
        return Ok(native_data);
    }

    // Fall back to big-endian decode (legacy data).
    if bytes.len() % std::mem::size_of::<f64>() == 0 {
        // Try f32-BE first
        let f32_data: Vec<f32> = bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_be_bytes(chunk.try_into().unwrap()))
            .collect();
        let be_ok = f32_data[..sample_count]
            .iter()
            .all(|v| v.is_finite() && v.abs() < 1e10);
        if be_ok {
            return Ok(f32_data);
        }
        // Try f64-BE → f32
        let f64_data: Vec<f32> = bytes
            .chunks_exact(std::mem::size_of::<f64>())
            .map(|chunk| f64::from_be_bytes(chunk.try_into().unwrap()) as f32)
            .collect();
        return Ok(f64_data);
    }

    // f32-BE only
    let data: Vec<f32> = bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_be_bytes(chunk.try_into().unwrap()))
        .collect();
    Ok(data)
}

fn encode_scalar_int8(data: &[f64]) -> Result<Vec<u8>, VectorError> {
    let dim = u32::try_from(data.len()).map_err(|_| VectorError::InvalidVectorLength)?;
    let max_abs = data.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let scale = if max_abs > 0.0 {
        (max_abs / 127.0) as f32
    } else {
        1.0_f32
    };

    let mut bytes = Vec::with_capacity(9 + 4 + data.len());
    bytes.extend_from_slice(SPINDLE_MAGIC);
    bytes.push(TAG_SCALAR_INT8);
    bytes.extend_from_slice(&dim.to_le_bytes());
    bytes.extend_from_slice(&scale.to_le_bytes());

    for &value in data {
        let quantized = (value / scale as f64).round().clamp(-127.0, 127.0) as i8;
        bytes.push(quantized as u8);
    }

    Ok(bytes)
}

fn decode_scalar_int8_f64(dim: usize, payload: &[u8]) -> Result<Vec<f64>, VectorError> {
    if payload.len() != 4 + dim {
        return Err(VectorError::InvalidVectorData);
    }

    let scale = f32::from_le_bytes(payload[..4].try_into().unwrap()) as f64;
    let mut data = Vec::with_capacity(dim);
    for byte in &payload[4..4 + dim] {
        data.push((*byte as i8) as f64 * scale);
    }
    Ok(data)
}

fn encode_binary_sign(data: &[f64], configured_dims: usize) -> Result<Vec<u8>, VectorError> {
    let dim = data.len();
    let stored_dim = configured_dims.min(dim).max(1);
    let full_dim = u32::try_from(dim).map_err(|_| VectorError::InvalidVectorLength)?;
    let stored_dim_u32 = u32::try_from(stored_dim).map_err(|_| VectorError::InvalidVectorLength)?;
    let norm = l2_norm(data) as f32;
    let packed = pack_bits(
        data.iter()
            .take(stored_dim)
            .map(|value| *value >= 0.0)
            .collect::<Vec<_>>()
            .as_slice(),
    );

    let mut bytes = Vec::with_capacity(9 + 4 + packed.len());
    bytes.extend_from_slice(SPINDLE_MAGIC);
    bytes.push(TAG_BINARY_SIGN);
    bytes.extend_from_slice(&full_dim.to_le_bytes());
    bytes.extend_from_slice(&stored_dim_u32.to_le_bytes());
    bytes.extend_from_slice(&norm.to_le_bytes());
    bytes.extend_from_slice(&packed);
    Ok(bytes)
}

fn decode_binary_sign_f64(dim: usize, payload: &[u8]) -> Result<Vec<f64>, VectorError> {
    if payload.len() < 8 {
        return Err(VectorError::InvalidVectorData);
    }

    let stored_dim = u32::from_le_bytes(payload[..4].try_into().unwrap()) as usize;
    if stored_dim == 0 || stored_dim > dim {
        return Err(VectorError::InvalidVectorData);
    }
    let packed_len = stored_dim.div_ceil(8);
    if payload.len() != 8 + packed_len {
        return Err(VectorError::InvalidVectorData);
    }
    let norm = f32::from_le_bytes(payload[4..8].try_into().unwrap()) as f64;
    let packed = &payload[8..];
    let scale = if stored_dim > 0 {
        let denom = (stored_dim as f64).sqrt();
        if denom > 0.0 {
            norm / denom
        } else {
            norm
        }
    } else {
        norm
    };

    let mut data = Vec::with_capacity(dim);
    for idx in 0..dim {
        if idx < stored_dim {
            let byte = packed[idx / 8];
            let bit = (byte >> (idx % 8)) & 1;
            data.push(if bit == 1 { scale } else { -scale });
        } else {
            data.push(0.0);
        }
    }
    Ok(data)
}

fn encode_turbo_int4(
    data: &[f64],
    binary_dims: usize,
    configured_dims: usize,
) -> Result<Vec<u8>, VectorError> {
    let dim = data.len();
    let binary_dim = binary_dims.min(dim).max(1);
    let stored_dim = configured_dims.min(dim).max(1);
    let full_dim = u32::try_from(dim).map_err(|_| VectorError::InvalidVectorLength)?;
    let binary_dim_u32 = u32::try_from(binary_dim).map_err(|_| VectorError::InvalidVectorLength)?;
    let stored_dim_u32 = u32::try_from(stored_dim).map_err(|_| VectorError::InvalidVectorLength)?;
    let (rotated, norm) = rotate_and_normalize(data);
    let codebook = turbo_codebook(dim.max(1));
    let mut codes = Vec::with_capacity(stored_dim);

    for value in rotated.iter().take(stored_dim) {
        codes.push(nearest_code(*value as f32, &codebook));
    }

    let binary = pack_bits(
        rotated
            .iter()
            .take(binary_dim)
            .map(|value| *value >= 0.0)
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let packed = pack_nibbles(&codes);
    let mut bytes = Vec::with_capacity(9 + 12 + binary.len() + packed.len());
    bytes.extend_from_slice(SPINDLE_MAGIC);
    bytes.push(TAG_TURBO_INT4);
    bytes.extend_from_slice(&full_dim.to_le_bytes());
    bytes.extend_from_slice(&binary_dim_u32.to_le_bytes());
    bytes.extend_from_slice(&stored_dim_u32.to_le_bytes());
    bytes.extend_from_slice(&(norm as f32).to_le_bytes());
    bytes.extend_from_slice(&binary);
    bytes.extend_from_slice(&packed);
    Ok(bytes)
}

fn decode_turbo_int4_f64(dim: usize, payload: &[u8]) -> Result<Vec<f64>, VectorError> {
    if payload.len() < 12 {
        return Err(VectorError::InvalidVectorData);
    }

    let binary_dim = u32::from_le_bytes(payload[..4].try_into().unwrap()) as usize;
    let stored_dim = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
    if stored_dim == 0 || stored_dim > dim {
        return Err(VectorError::InvalidVectorData);
    }
    if binary_dim == 0 || binary_dim > dim {
        return Err(VectorError::InvalidVectorData);
    }
    let packed_len = stored_dim.div_ceil(2);
    let binary_len = binary_dim.div_ceil(8);
    if payload.len() != 12 + binary_len + packed_len {
        return Err(VectorError::InvalidVectorData);
    }
    let norm = f32::from_le_bytes(payload[8..12].try_into().unwrap()) as f64;
    let codes = unpack_nibbles(&payload[12 + binary_len..], stored_dim);
    let codebook = turbo_codebook(dim.max(1));
    let mut data = Vec::with_capacity(dim);

    for idx in 0..dim {
        if idx < stored_dim {
            let code = codes
                .get(idx)
                .copied()
                .ok_or(VectorError::InvalidVectorData)?;
            let centroid = codebook
                .get(code as usize)
                .copied()
                .ok_or(VectorError::InvalidVectorData)?;
            data.push(centroid as f64 * norm);
        } else {
            data.push(0.0);
        }
    }

    Ok(data)
}

fn score_binary_payload(
    query: &PreparedBinaryQuery,
    bytes: &[u8],
) -> Result<ApproximateInnerProduct, VectorError> {
    if !is_spindle_payload(bytes)
        || bytes.get(4).copied() != Some(TAG_BINARY_SIGN)
        || bytes.len() < 17
    {
        return Err(VectorError::InvalidVectorData);
    }

    let payload = bytes.get(9..).ok_or(VectorError::InvalidVectorData)?;
    if payload.len() < 8 {
        return Err(VectorError::InvalidVectorData);
    }

    let stored_dim = u32::from_le_bytes(payload[..4].try_into().unwrap()) as usize;
    let doc_norm = f32::from_le_bytes(payload[4..8].try_into().unwrap()) as f64;
    let compare_dim = stored_dim.min(query.stored_dim);
    let compare_len = compare_dim.div_ceil(8);
    let packed = payload
        .get(8..8 + compare_len)
        .ok_or(VectorError::InvalidVectorData)?;
    let query_bits = query
        .bits
        .get(..compare_len)
        .ok_or(VectorError::InvalidVectorData)?;
    let signed_sum = signed_bit_dot(packed, query_bits, 0, compare_dim);
    let unit_dot = if compare_dim > 0 {
        signed_sum / compare_dim as f64
    } else {
        0.0
    };

    Ok(ApproximateInnerProduct {
        unit_dot: unit_dot.clamp(-1.0, 1.0),
        query_norm: query.query_norm,
        doc_norm,
    })
}

fn score_turbo_payload(
    query: &PreparedTurboQuery,
    bytes: &[u8],
) -> Result<ApproximateInnerProduct, VectorError> {
    if !is_spindle_payload(bytes)
        || bytes.get(4).copied() != Some(TAG_TURBO_INT4)
        || bytes.len() < 21
    {
        return Err(VectorError::InvalidVectorData);
    }

    let full_dim = read_u32(bytes, 5)? as usize;
    let payload = bytes.get(9..).ok_or(VectorError::InvalidVectorData)?;
    if payload.len() < 12 {
        return Err(VectorError::InvalidVectorData);
    }

    let binary_dim = u32::from_le_bytes(payload[..4].try_into().unwrap()) as usize;
    let int4_dim = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
    let doc_norm = f32::from_le_bytes(payload[8..12].try_into().unwrap()) as f64;
    let binary_len = binary_dim.div_ceil(8);
    let packed_len = int4_dim.div_ceil(2);
    let binary = payload
        .get(12..12 + binary_len)
        .ok_or(VectorError::InvalidVectorData)?;
    let packed = payload
        .get(12 + binary_len..12 + binary_len + packed_len)
        .ok_or(VectorError::InvalidVectorData)?;
    let codebook = turbo_codebook(full_dim.max(1));

    let compare_pairs = packed.len().min(query.q_even.len()).min(query.q_odd.len());
    let compare_dim = int4_dim.min(query.int4_dim);
    let mut int4_sum = 0.0;
    for (idx, byte) in packed.iter().take(compare_pairs).enumerate() {
        int4_sum += codebook[(byte & 0x0f) as usize] as f64 * query.q_even[idx];
        if idx * 2 + 1 < compare_dim {
            int4_sum += codebook[((byte >> 4) & 0x0f) as usize] as f64 * query.q_odd[idx];
        }
    }

    let binary_compare_dim = binary_dim.min(query.binary_dim);
    let tail_dim = binary_compare_dim.saturating_sub(compare_dim);
    let binary_tail_sum = if tail_dim > 0 {
        let query_binary = query
            .binary_bits
            .get(..binary_len.min(query.binary_bits.len()))
            .ok_or(VectorError::InvalidVectorData)?;
        signed_bit_dot(binary, query_binary, compare_dim, tail_dim)
    } else {
        0.0
    };
    // int4_sum is already the partial dot-product of two unit vectors — no division.
    // binary_tail_sum is ±1 per bit; scale to dot-product units:
    // for rotated unit vectors, E[|x_i|·|x_j|] ≈ 2/(π·dim).
    let binary_scale = if tail_dim > 0 && full_dim > 0 {
        2.0 / (std::f64::consts::PI * full_dim as f64)
    } else {
        0.0
    };
    let unit_dot = int4_sum + binary_tail_sum * binary_scale;

    Ok(ApproximateInnerProduct {
        unit_dot: unit_dot.clamp(-1.0, 1.0),
        query_norm: query.query_norm,
        doc_norm,
    })
}

// ---------------------------------------------------------------------------
// TurboProd: 3-bit MSE + 1-bit QJL residual (paper Algorithm 2)
// ---------------------------------------------------------------------------

fn turbo_prod_codebook(dim: usize) -> [f32; MSE_LEVELS] {
    scaled_codebook(&MSE_CODEBOOK_STD_NORMAL, dim)
}

fn turbo_prod_mse_lut(rotated_query: &[f64], mse_dim: usize) -> Vec<[f64; MSE_LEVELS]> {
    let codebook = turbo_prod_codebook(rotated_query.len().max(1));
    rotated_query
        .iter()
        .take(mse_dim)
        .map(|q| {
            let mut row = [0.0f64; MSE_LEVELS];
            for code in 0..MSE_LEVELS {
                row[code] = codebook[code] as f64 * *q;
            }
            row
        })
        .collect()
}

fn encode_turbo_prod(data: &[f64], configured_dims: usize) -> Result<Vec<u8>, VectorError> {
    let dim = data.len();
    let stored_dim = configured_dims.min(dim).max(1);
    let full_dim = u32::try_from(dim).map_err(|_| VectorError::InvalidVectorLength)?;
    let stored_dim_u32 = u32::try_from(stored_dim).map_err(|_| VectorError::InvalidVectorLength)?;

    // Step 1: rotate and normalize
    let (rotated_unit, norm) = rotate_and_normalize(data);

    // Step 2: 3-bit MSE quantize
    let codebook = turbo_prod_codebook(dim.max(1));
    let mut codes = Vec::with_capacity(stored_dim);
    let mut mse_recon = vec![0.0; dim];
    for i in 0..stored_dim {
        let code = nearest_code(rotated_unit[i] as f32, &codebook);
        codes.push(code);
        mse_recon[i] = codebook[code as usize] as f64;
    }

    // Step 3: residual
    let mut residual = Vec::with_capacity(dim);
    for i in 0..dim {
        residual.push(rotated_unit[i] - mse_recon[i]);
    }
    let residual_norm = l2_norm(&residual);

    // Step 4: QJL — sign(S · residual)
    let transformed = srht_forward(&residual, QJL_SEED);
    let qjl_signs = pack_bits(&transformed.iter().map(|v| *v >= 0.0).collect::<Vec<_>>());

    // Step 5: pack payload
    let mse_packed = pack_3bit(&codes);
    let qjl_packed_len = dim.div_ceil(8);

    let mut bytes = Vec::with_capacity(9 + 12 + mse_packed.len() + qjl_packed_len);
    bytes.extend_from_slice(SPINDLE_MAGIC);
    bytes.push(TAG_TURBO_PROD);
    bytes.extend_from_slice(&full_dim.to_le_bytes());
    // payload starts here (after 9-byte header)
    bytes.extend_from_slice(&stored_dim_u32.to_le_bytes());
    bytes.extend_from_slice(&(norm as f32).to_le_bytes());
    bytes.extend_from_slice(&(residual_norm as f32).to_le_bytes());
    bytes.extend_from_slice(&mse_packed);
    bytes.extend_from_slice(&qjl_signs);
    Ok(bytes)
}

fn decode_turbo_prod_f64(dim: usize, payload: &[u8]) -> Result<Vec<f64>, VectorError> {
    if payload.len() < 12 {
        return Err(VectorError::InvalidVectorData);
    }

    let stored_dim = u32::from_le_bytes(payload[..4].try_into().unwrap()) as usize;
    if stored_dim == 0 || stored_dim > dim {
        return Err(VectorError::InvalidVectorData);
    }
    let norm = f32::from_le_bytes(payload[4..8].try_into().unwrap()) as f64;
    let _residual_norm = f32::from_le_bytes(payload[8..12].try_into().unwrap()) as f64;

    let mse_packed_len = (stored_dim * 3).div_ceil(8);
    let qjl_packed_len = dim.div_ceil(8);
    if payload.len() != 12 + mse_packed_len + qjl_packed_len {
        return Err(VectorError::InvalidVectorData);
    }

    let mse_bytes = &payload[12..12 + mse_packed_len];
    let _qjl_bytes = &payload[12 + mse_packed_len..];

    // MSE-only reconstruction — QJL correction is applied only in the scoring path
    // (score_turbo_prod_payload), not here. Decode is used for HNSW navigation where
    // clean MSE vectors give better graph quality than noisy MSE+QJL vectors.
    let codes = unpack_3bit(mse_bytes, stored_dim);
    let codebook = turbo_prod_codebook(dim.max(1));
    let mut data = Vec::with_capacity(dim);
    for i in 0..dim {
        if i < stored_dim {
            let code = codes[i].min((codebook.len() - 1) as u8);
            data.push(codebook[code as usize] as f64 * norm);
        } else {
            data.push(0.0);
        }
    }
    Ok(data)
}

fn score_turbo_prod_payload(
    query: &PreparedTurboProdQuery,
    bytes: &[u8],
) -> Result<ApproximateInnerProduct, VectorError> {
    let tag = bytes.get(4).copied();
    if !is_spindle_payload(bytes) || !matches!(tag, Some(TAG_TURBO_PROD)) || bytes.len() < 21 {
        return Err(VectorError::InvalidVectorData);
    }

    let full_dim = read_u32(bytes, 5)? as usize;
    let payload = bytes.get(9..).ok_or(VectorError::InvalidVectorData)?;
    if payload.len() < 12 {
        return Err(VectorError::InvalidVectorData);
    }

    let stored_dim = u32::from_le_bytes(payload[..4].try_into().unwrap()) as usize;
    let doc_norm = f32::from_le_bytes(payload[4..8].try_into().unwrap()) as f64;
    let residual_norm = f32::from_le_bytes(payload[8..12].try_into().unwrap()) as f64;

    let mse_packed_len = (stored_dim * 3).div_ceil(8);
    let qjl_packed_len = full_dim.div_ceil(8);
    if payload.len() < 12 + mse_packed_len + qjl_packed_len {
        return Err(VectorError::InvalidVectorData);
    }

    let mse_bytes = &payload[12..12 + mse_packed_len];
    let qjl_bytes = &payload[12 + mse_packed_len..];

    let compare_dim = stored_dim
        .min(query.mse_dim)
        .min(query.q_unit.len())
        .min(query.mse_lut.len());

    let mse_dot = mse_3bit_dot_lut(mse_bytes, &query.mse_lut, compare_dim);

    // QJL correction: √(π/2)/d · ‖r‖ · Σ (S·q_unit)_i · sign_i
    let qjl_dot = if residual_norm > 0.0 && full_dim > 0 {
        let scale = (std::f64::consts::PI / 2.0).sqrt() * residual_norm / full_dim as f64;
        let compare_qjl = full_dim.min(query.q_transformed.len());
        let sum = qjl_signed_sum(&query.q_transformed, qjl_bytes, compare_qjl);
        sum * scale
    } else {
        0.0
    };

    let unit_dot = mse_dot + qjl_dot;

    Ok(ApproximateInnerProduct {
        unit_dot: unit_dot.clamp(-1.0, 1.0),
        query_norm: query.query_norm,
        doc_norm,
    })
}

/// SRHT forward: S·x = (1/√w) · D₂ · H · D₁ · x
#[inline]
fn srht_forward(data: &[f64], seed: u64) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }
    let dim = data.len();
    let width = dim.next_power_of_two();
    let seed2 = splitmix64(seed);

    let mut buf = vec![0.0; width];
    for (i, &v) in data.iter().enumerate() {
        buf[i] = v * signed_bit(seed, i);
    }
    hadamard_transform_inplace(&mut buf);
    let scale = 1.0 / (width as f64).sqrt();
    for i in 0..width {
        buf[i] *= scale * signed_bit(seed2, i);
    }
    buf.truncate(dim);
    buf
}

/// SRHT adjoint: S^T·z = (1/√w) · D₁ · H · D₂ · z
#[allow(dead_code)]
#[inline]
fn srht_adjoint(data: &[f64], seed: u64) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }
    let dim = data.len();
    let width = dim.next_power_of_two();
    let seed2 = splitmix64(seed);

    let mut buf = vec![0.0; width];
    for (i, &v) in data.iter().enumerate() {
        buf[i] = v * signed_bit(seed2, i);
    }
    hadamard_transform_inplace(&mut buf);
    let scale = 1.0 / (width as f64).sqrt();
    for i in 0..width {
        buf[i] *= scale * signed_bit(seed, i);
    }
    buf.truncate(dim);
    buf
}

fn pack_3bit(codes: &[u8]) -> Vec<u8> {
    let total_bits = codes.len() * 3;
    let mut packed = vec![0u8; total_bits.div_ceil(8)];
    for (i, &code) in codes.iter().enumerate() {
        let bit_pos = i * 3;
        let byte_idx = bit_pos / 8;
        let bit_offset = bit_pos % 8;
        packed[byte_idx] |= (code & 0x07) << bit_offset;
        if bit_offset > 5 && byte_idx + 1 < packed.len() {
            packed[byte_idx + 1] |= (code & 0x07) >> (8 - bit_offset);
        }
    }
    packed
}

fn unpack_3bit(bytes: &[u8], count: usize) -> Vec<u8> {
    let mut codes = Vec::with_capacity(count);
    for i in 0..count {
        let bit_pos = i * 3;
        let byte_idx = bit_pos / 8;
        let bit_offset = bit_pos % 8;
        let mut code = (bytes[byte_idx] >> bit_offset) & 0x07;
        if bit_offset > 5 && byte_idx + 1 < bytes.len() {
            code |= (bytes[byte_idx + 1] << (8 - bit_offset)) & 0x07;
        }
        codes.push(code);
    }
    codes
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

#[inline]
#[cfg(test)]
fn mse_3bit_dot(mse_bytes: &[u8], query: &[f64], codebook: &[f32; MSE_LEVELS], dim: usize) -> f64 {
    let compare_dim = dim.min(query.len());
    let blocks = compare_dim / 8;
    let mut sum = 0.0;

    for block in 0..blocks {
        let byte_idx = block * 3;
        let bits = mse_bytes[byte_idx] as u32
            | ((mse_bytes[byte_idx + 1] as u32) << 8)
            | ((mse_bytes[byte_idx + 2] as u32) << 16);
        let base = block * 8;
        sum += codebook[(bits & 0x07) as usize] as f64 * query[base];
        sum += codebook[((bits >> 3) & 0x07) as usize] as f64 * query[base + 1];
        sum += codebook[((bits >> 6) & 0x07) as usize] as f64 * query[base + 2];
        sum += codebook[((bits >> 9) & 0x07) as usize] as f64 * query[base + 3];
        sum += codebook[((bits >> 12) & 0x07) as usize] as f64 * query[base + 4];
        sum += codebook[((bits >> 15) & 0x07) as usize] as f64 * query[base + 5];
        sum += codebook[((bits >> 18) & 0x07) as usize] as f64 * query[base + 6];
        sum += codebook[((bits >> 21) & 0x07) as usize] as f64 * query[base + 7];
    }

    for (idx, value) in query.iter().enumerate().take(compare_dim).skip(blocks * 8) {
        let code = packed_3bit_code(mse_bytes, idx);
        sum += codebook[code as usize] as f64 * value;
    }

    sum
}

#[inline]
fn mse_3bit_dot_lut(mse_bytes: &[u8], lut: &[[f64; MSE_LEVELS]], dim: usize) -> f64 {
    let compare_dim = dim.min(lut.len());
    let blocks = compare_dim / 8;
    let mut sum = 0.0;

    for block in 0..blocks {
        let byte_idx = block * 3;
        let bits = mse_bytes[byte_idx] as u32
            | ((mse_bytes[byte_idx + 1] as u32) << 8)
            | ((mse_bytes[byte_idx + 2] as u32) << 16);
        let base = block * 8;
        sum += lut[base][(bits & 0x07) as usize];
        sum += lut[base + 1][((bits >> 3) & 0x07) as usize];
        sum += lut[base + 2][((bits >> 6) & 0x07) as usize];
        sum += lut[base + 3][((bits >> 9) & 0x07) as usize];
        sum += lut[base + 4][((bits >> 12) & 0x07) as usize];
        sum += lut[base + 5][((bits >> 15) & 0x07) as usize];
        sum += lut[base + 6][((bits >> 18) & 0x07) as usize];
        sum += lut[base + 7][((bits >> 21) & 0x07) as usize];
    }

    for idx in blocks * 8..compare_dim {
        let code = packed_3bit_code(mse_bytes, idx);
        sum += lut[idx][code as usize];
    }

    sum
}

#[inline]
fn qjl_signed_sum(query: &[f64], signs: &[u8], dim: usize) -> f64 {
    let compare_dim = dim.min(query.len()).min(signs.len() * 8);

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { qjl_signed_sum_neon(query, signs, compare_dim) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { qjl_signed_sum_avx2(query, signs, compare_dim) }
        } else {
            qjl_signed_sum_scalar(query, signs, compare_dim)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        qjl_signed_sum_scalar(query, signs, compare_dim)
    }
}

#[inline]
#[allow(dead_code)]
fn qjl_signed_sum_scalar(query: &[f64], signs: &[u8], dim: usize) -> f64 {
    let blocks = dim / 8;
    let mut sum = 0.0;

    for (byte_idx, byte) in signs.iter().take(blocks).copied().enumerate() {
        let base = byte_idx * 8;
        sum += if byte & 0x01 != 0 {
            query[base]
        } else {
            -query[base]
        };
        sum += if byte & 0x02 != 0 {
            query[base + 1]
        } else {
            -query[base + 1]
        };
        sum += if byte & 0x04 != 0 {
            query[base + 2]
        } else {
            -query[base + 2]
        };
        sum += if byte & 0x08 != 0 {
            query[base + 3]
        } else {
            -query[base + 3]
        };
        sum += if byte & 0x10 != 0 {
            query[base + 4]
        } else {
            -query[base + 4]
        };
        sum += if byte & 0x20 != 0 {
            query[base + 5]
        } else {
            -query[base + 5]
        };
        sum += if byte & 0x40 != 0 {
            query[base + 6]
        } else {
            -query[base + 6]
        };
        sum += if byte & 0x80 != 0 {
            query[base + 7]
        } else {
            -query[base + 7]
        };
    }

    for idx in blocks * 8..dim {
        let bit = (signs[idx / 8] >> (idx % 8)) & 1;
        sum += if bit == 1 { query[idx] } else { -query[idx] };
    }

    sum
}

#[cfg(target_arch = "aarch64")]
unsafe fn qjl_signed_sum_neon(query: &[f64], signs: &[u8], dim: usize) -> f64 {
    use std::arch::aarch64::*;

    let mut acc = vdupq_n_f64(0.0);
    let simd_len = dim / 2 * 2;
    let mut idx = 0;
    while idx < simd_len {
        let byte = signs[idx / 8];
        let sign0 = if (byte >> (idx % 8)) & 1 == 1 {
            1.0
        } else {
            -1.0
        };
        let sign1 = if (signs[(idx + 1) / 8] >> ((idx + 1) % 8)) & 1 == 1 {
            1.0
        } else {
            -1.0
        };
        let signv = vsetq_lane_f64(sign1, vdupq_n_f64(sign0), 1);
        let qv = vld1q_f64(query.as_ptr().add(idx));
        acc = vfmaq_f64(acc, qv, signv);
        idx += 2;
    }

    let mut sum = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);
    if idx < dim {
        let bit = (signs[idx / 8] >> (idx % 8)) & 1;
        sum += if bit == 1 { query[idx] } else { -query[idx] };
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn qjl_signed_sum_avx2(query: &[f64], signs: &[u8], dim: usize) -> f64 {
    use std::arch::x86_64::*;

    let mut acc = _mm256_setzero_pd();
    let simd_len = dim / 4 * 4;
    let mut idx = 0;
    while idx < simd_len {
        let sign = |lane: usize| {
            if (signs[(idx + lane) / 8] >> ((idx + lane) % 8)) & 1 == 1 {
                1.0
            } else {
                -1.0
            }
        };
        let qv = _mm256_loadu_pd(query.as_ptr().add(idx));
        let signv = _mm256_set_pd(sign(3), sign(2), sign(1), sign(0));
        acc = _mm256_add_pd(acc, _mm256_mul_pd(qv, signv));
        idx += 4;
    }

    let mut lanes = [0.0_f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), acc);
    let mut sum = lanes.iter().sum::<f64>();
    while idx < dim {
        let bit = (signs[idx / 8] >> (idx % 8)) & 1;
        sum += if bit == 1 { query[idx] } else { -query[idx] };
        idx += 1;
    }
    sum
}

fn l2_norm(data: &[f64]) -> f64 {
    data.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn rotate_and_normalize(data: &[f64]) -> (Vec<f64>, f64) {
    let norm = l2_norm(data);
    if data.is_empty() {
        return (Vec::new(), norm);
    }
    if norm == 0.0 {
        return (vec![0.0; data.len()], 0.0);
    }

    let normalized = data.iter().map(|value| *value / norm).collect::<Vec<_>>();
    (fast_pseudo_rotate(&normalized), norm)
}

fn fast_pseudo_rotate(data: &[f64]) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }

    let width = data.len().next_power_of_two().max(1);
    let mut buffer = vec![0.0; width];
    for (idx, value) in data.iter().enumerate() {
        buffer[idx] = *value * signed_bit(TURBO_ROTATION_SEED, idx);
    }

    hadamard_transform_inplace(&mut buffer);
    let scale = 1.0 / (width as f64).sqrt();
    for value in &mut buffer {
        *value *= scale;
    }

    shuffled_indices(width, TURBO_ROTATION_SEED ^ data.len() as u64)
        .into_iter()
        .take(data.len())
        .map(|idx| buffer[idx])
        .collect()
}

fn signed_bit(seed: u64, idx: usize) -> f64 {
    if splitmix64(seed ^ idx as u64) & 1 == 0 {
        -1.0
    } else {
        1.0
    }
}

fn shuffled_indices(len: usize, seed: u64) -> Vec<usize> {
    let mut order = (0..len).collect::<Vec<_>>();
    if len <= 1 {
        return order;
    }

    let mut state = seed ^ len as u64;
    for idx in (1..len).rev() {
        state = splitmix64(state.wrapping_add(idx as u64));
        let swap_idx = (state as usize) % (idx + 1);
        order.swap(idx, swap_idx);
    }
    order
}

fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// In-place Walsh-Hadamard transform. The butterfly loop is auto-vectorizable
/// by LLVM when compiled with `-C target-cpu=native` (release builds).
#[inline]
fn hadamard_transform_inplace(data: &mut [f64]) {
    let len = data.len();
    let mut width = 1usize;
    while width < len {
        let stride = width * 2;
        let mut start = 0usize;
        while start < len {
            for offset in 0..width {
                let lhs = data[start + offset];
                let rhs = data[start + offset + width];
                data[start + offset] = lhs + rhs;
                data[start + offset + width] = lhs - rhs;
            }
            start += stride;
        }
        width *= 2;
    }
}

fn signed_bit_dot(lhs: &[u8], rhs: &[u8], start: usize, dim: usize) -> f64 {
    let end = start + dim;
    let mut idx = start;
    let mut signed_sum: i32 = 0;

    while idx < end && !idx.is_multiple_of(8) {
        let lhs_bit = (lhs[idx / 8] >> (idx % 8)) & 1;
        let rhs_bit = (rhs[idx / 8] >> (idx % 8)) & 1;
        signed_sum += if lhs_bit == rhs_bit { 1 } else { -1 };
        idx += 1;
    }

    let byte_start = idx / 8;
    let full_bytes = (end - idx) / 8;
    for offset in 0..full_bytes {
        let mismatches = (lhs[byte_start + offset] ^ rhs[byte_start + offset]).count_ones() as i32;
        signed_sum += 8 - 2 * mismatches;
    }

    idx += full_bytes * 8;
    while idx < end {
        let lhs_bit = (lhs[idx / 8] >> (idx % 8)) & 1;
        let rhs_bit = (rhs[idx / 8] >> (idx % 8)) & 1;
        signed_sum += if lhs_bit == rhs_bit { 1 } else { -1 };
        idx += 1;
    }

    signed_sum as f64
}

fn pack_bits(bits: &[bool]) -> Vec<u8> {
    let mut packed = vec![0u8; bits.len().div_ceil(8)];
    for (idx, bit) in bits.iter().enumerate() {
        if *bit {
            packed[idx / 8] |= 1 << (idx % 8);
        }
    }
    packed
}

fn pack_nibbles(codes: &[u8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(codes.len().div_ceil(2));
    let mut idx = 0;
    while idx < codes.len() {
        let lo = codes[idx] & 0x0f;
        let hi = if idx + 1 < codes.len() {
            (codes[idx + 1] & 0x0f) << 4
        } else {
            0
        };
        packed.push(lo | hi);
        idx += 2;
    }
    packed
}

fn unpack_nibbles(bytes: &[u8], dim: usize) -> Vec<u8> {
    let mut values = Vec::with_capacity(dim);
    for &byte in bytes {
        values.push(byte & 0x0f);
        if values.len() >= dim {
            break;
        }
        values.push((byte >> 4) & 0x0f);
        if values.len() >= dim {
            break;
        }
    }
    values
}

fn nearest_code(value: f32, codebook: &[f32]) -> u8 {
    let mut best_idx = 0usize;
    let mut best_dist = f32::INFINITY;
    for (idx, centroid) in codebook.iter().enumerate() {
        let dist = (value - centroid).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }
    best_idx as u8
}

fn turbo_codebook(dim: usize) -> [f32; TURBO_LEVELS] {
    scaled_codebook(&TURBO_CODEBOOK_STD_NORMAL, dim)
}

fn scaled_codebook<const N: usize>(standard_normal: &[f64; N], dim: usize) -> [f32; N] {
    let sigma = 1.0_f64 / (dim.max(1) as f64).sqrt();
    let mut codebook = [0.0; N];
    for (idx, centroid) in standard_normal.iter().enumerate() {
        codebook[idx] = (centroid * sigma) as f32;
    }
    codebook
}

fn read_u32(bytes: &[u8], start: usize) -> Result<u32, VectorError> {
    let end = start + 4;
    let slice = bytes
        .get(start..end)
        .ok_or(VectorError::InvalidVectorData)?;
    Ok(u32::from_le_bytes(slice.try_into().unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn cosine_similarity(lhs: &[f32], rhs: &[f32]) -> f64 {
        let mut dot: f64 = 0.0;
        let mut lhs_norm: f64 = 0.0;
        let mut rhs_norm: f64 = 0.0;

        for idx in 0..lhs.len() {
            let l = lhs[idx] as f64;
            let r = rhs[idx] as f64;
            dot += l * r;
            lhs_norm += l * l;
            rhs_norm += r * r;
        }

        if lhs_norm == 0.0 || rhs_norm == 0.0 {
            0.0
        } else {
            dot / (lhs_norm.sqrt() * rhs_norm.sqrt())
        }
    }

    fn top_k_ids(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f64)> = vectors
            .iter()
            .enumerate()
            .map(|(idx, vector)| (idx, cosine_similarity(vector, query)))
            .collect();
        scored.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap());
        scored.into_iter().take(k).map(|(idx, _)| idx).collect()
    }

    fn average_overlap_at_k(
        baseline: &[Vec<usize>],
        candidate_vectors: &[Vec<f32>],
        queries: &[Vec<f32>],
        k: usize,
    ) -> f64 {
        let mut total = 0.0;
        for (baseline_ids, query) in baseline.iter().zip(queries.iter()) {
            let candidate_ids = top_k_ids(candidate_vectors, query, k);
            let overlap = baseline_ids
                .iter()
                .filter(|id| candidate_ids.contains(id))
                .count();
            total += overlap as f64 / k as f64;
        }
        total / baseline.len().max(1) as f64
    }

    fn reference_lloyd_max_gaussian(levels: usize, sigma: f64) -> Vec<f32> {
        let range = 6.0 * sigma.max(1e-6);
        let min_x = -range;
        let max_x = range;
        let steps = 2048usize;
        let dx = (max_x - min_x) / steps as f64;

        let xs: Vec<f64> = (0..steps)
            .map(|idx| min_x + (idx as f64 + 0.5) * dx)
            .collect();
        let ws: Vec<f64> = xs
            .iter()
            .map(|x| reference_gaussian_pdf(*x, sigma) * dx)
            .collect();

        let mut centroids: Vec<f64> = (0..levels)
            .map(|idx| {
                let frac = if levels > 1 {
                    idx as f64 / (levels - 1) as f64
                } else {
                    0.5
                };
                min_x + frac * (max_x - min_x)
            })
            .collect();

        for _ in 0..24 {
            let mut boundaries = Vec::with_capacity(levels + 1);
            boundaries.push(f64::NEG_INFINITY);
            for pair in centroids.windows(2) {
                boundaries.push((pair[0] + pair[1]) * 0.5);
            }
            boundaries.push(f64::INFINITY);

            for level in 0..levels {
                let lo = boundaries[level];
                let hi = boundaries[level + 1];
                let mut num = 0.0;
                let mut den = 0.0;

                for (x, w) in xs.iter().zip(ws.iter()) {
                    if *x >= lo && *x < hi {
                        num += x * w;
                        den += w;
                    }
                }

                if den > 0.0 {
                    centroids[level] = num / den;
                }
            }
        }

        centroids.into_iter().map(|value| value as f32).collect()
    }

    fn reference_gaussian_pdf(x: f64, sigma: f64) -> f64 {
        let sigma = sigma.max(1e-6);
        let variance = sigma * sigma;
        let norm = (2.0 * std::f64::consts::PI * variance).sqrt();
        (-x * x / (2.0 * variance)).exp() / norm
    }

    #[test]
    fn fixed_codebooks_match_previous_lloyd_max_solver() {
        for dim in [1, 3, 8, 128, 768] {
            let sigma = 1.0_f64 / (dim as f64).sqrt();

            let reference_mse = reference_lloyd_max_gaussian(MSE_LEVELS, sigma);
            let fixed_mse = turbo_prod_codebook(dim);
            for (idx, (reference, fixed)) in reference_mse.iter().zip(fixed_mse.iter()).enumerate()
            {
                assert!(
                    (reference - fixed).abs() <= 1e-6,
                    "MSE centroid {idx} changed for dim {dim}: {reference} vs {fixed}"
                );
            }

            let reference_turbo = reference_lloyd_max_gaussian(TURBO_LEVELS, sigma);
            let fixed_turbo = turbo_codebook(dim);
            for (idx, (reference, fixed)) in
                reference_turbo.iter().zip(fixed_turbo.iter()).enumerate()
            {
                assert!(
                    (reference - fixed).abs() <= 1e-6,
                    "Turbo centroid {idx} changed for dim {dim}: {reference} vs {fixed}"
                );
            }
        }
    }

    #[test]
    fn scalar_round_trip_preserves_dim() {
        let config = SpindleConfig {
            mode: SpindleMode::ScalarInt8,
            ..SpindleConfig::default()
        };
        let data: Vec<f32> = vec![0.2, -0.4, 0.8, 1.6];
        let bytes = encode_vector(&data, &config).unwrap();
        let decoded = decode_vector(&bytes).unwrap();

        assert_eq!(decoded.len(), data.len());
        assert!(decoded
            .iter()
            .zip(data.iter())
            .all(|(a, b)| (a - b).abs() < 0.05));
    }

    #[test]
    fn binary_round_trip_preserves_signs() {
        let config = SpindleConfig {
            mode: SpindleMode::BinarySign,
            binary_dims: 4,
            ..SpindleConfig::default()
        };
        let data: Vec<f32> = vec![-2.0, 3.0, -4.0, 5.0];
        let bytes = encode_vector(&data, &config).unwrap();
        let decoded = decode_vector(&bytes).unwrap();

        assert_eq!(decoded.len(), data.len());
        assert!(decoded[0] < 0.0);
        assert!(decoded[1] > 0.0);
        assert!(decoded[2] < 0.0);
        assert!(decoded[3] > 0.0);
    }

    #[test]
    fn turbo_round_trip_preserves_dim() {
        let config = SpindleConfig {
            mode: SpindleMode::TurboInt4,
            turbo_dims: 8,
            ..SpindleConfig::default()
        };
        let data: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let bytes = encode_vector(&data, &config).unwrap();
        let decoded = decode_vector(&bytes).unwrap();

        assert_eq!(decoded.len(), data.len());
        assert!(decoded.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn raw_mode_round_trip_preserves_values() {
        let config = SpindleConfig {
            mode: SpindleMode::None,
            ..SpindleConfig::default()
        };
        let data: Vec<f32> = vec![0.125, -0.25, 0.5, -1.0];
        let bytes = encode_vector(&data, &config).unwrap();
        let decoded = decode_vector(&bytes).unwrap();

        assert_eq!(decoded, data);
        assert!(!is_spindle_payload(&bytes));
    }

    #[test]
    fn raw_payload_with_spindle_magic_falls_back_to_raw_decode() {
        let mut raw = vec![0_u8; 16];
        raw[..4].copy_from_slice(SPINDLE_MAGIC);

        let decoded = decode_vector(&raw).unwrap();
        // Should produce some result without error
        assert!(!decoded.is_empty() || raw.len() < 4);
    }

    #[test]
    fn malformed_scalar_spindle_payload_is_rejected() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(SPINDLE_MAGIC);
        bytes.push(TAG_SCALAR_INT8);
        bytes.extend_from_slice(&(4_u32).to_le_bytes());
        bytes.extend_from_slice(&(1.0_f32).to_le_bytes());
        bytes.push(1);

        let result = decode_vector(&bytes);
        assert!(matches!(result, Err(VectorError::InvalidVectorData)));
    }

    #[test]
    fn truncated_binary_payload_is_rejected() {
        let config = SpindleConfig {
            mode: SpindleMode::BinarySign,
            binary_dims: 8,
            ..SpindleConfig::default()
        };
        let data: Vec<f32> = vec![1.0, -1.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25];
        let mut bytes = encode_vector(&data, &config).unwrap();
        bytes.pop();

        let result = decode_vector(&bytes);
        assert!(matches!(result, Err(VectorError::InvalidVectorData)));
    }

    #[test]
    fn truncated_turbo_payload_is_rejected() {
        let config = SpindleConfig {
            mode: SpindleMode::TurboInt4,
            turbo_dims: 8,
            ..SpindleConfig::default()
        };
        let data: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let mut bytes = encode_vector(&data, &config).unwrap();
        bytes.pop();

        let result = decode_vector(&bytes);
        assert!(matches!(result, Err(VectorError::InvalidVectorData)));
    }

    #[test]
    fn turbo_query_scoring_prefers_matching_payload() {
        let config = SpindleConfig {
            mode: SpindleMode::TurboInt4,
            binary_dims: 8,
            turbo_dims: 8,
            keep_original: false,
            rescore: true,
            ..SpindleConfig::default()
        };
        let query: Vec<f32> = vec![0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2];
        let matching = encode_vector(&query, &config).unwrap();
        let opposite = encode_vector(
            &query.iter().map(|value| -*value).collect::<Vec<f32>>(),
            &config,
        )
        .unwrap();

        let prepared = prepare_query(&query, &config).unwrap();
        let matching_score = score_encoded(&prepared, &matching)
            .unwrap()
            .unwrap()
            .unit_dot;
        let opposite_score = score_encoded(&prepared, &opposite)
            .unwrap()
            .unwrap()
            .unit_dot;

        assert!(matching_score > opposite_score);
    }

    #[test]
    fn scalar_and_binary_codecs_preserve_top_k_overlap_against_raw() {
        let doc_count = 512usize;
        let query_count = 24usize;
        let dim = 256usize;
        let k = 10usize;
        let mut rng = StdRng::seed_from_u64(42);

        let docs: Vec<Vec<f32>> = (0..doc_count)
            .map(|_| {
                (0..dim)
                    .map(|_| rng.random_range(-1.0f32..1.0f32))
                    .collect()
            })
            .collect();
        let queries: Vec<Vec<f32>> = (0..query_count)
            .map(|_| {
                (0..dim)
                    .map(|_| rng.random_range(-1.0f32..1.0f32))
                    .collect()
            })
            .collect();
        let baseline: Vec<Vec<usize>> = queries
            .iter()
            .map(|query| top_k_ids(&docs, query, k))
            .collect();

        let scalar_docs: Vec<Vec<f32>> = docs
            .iter()
            .map(|doc| {
                decode_vector(
                    &encode_vector(
                        doc,
                        &SpindleConfig {
                            mode: SpindleMode::ScalarInt8,
                            ..SpindleConfig::default()
                        },
                    )
                    .unwrap(),
                )
                .unwrap()
            })
            .collect();
        let binary_docs: Vec<Vec<f32>> = docs
            .iter()
            .map(|doc| {
                decode_vector(
                    &encode_vector(
                        doc,
                        &SpindleConfig {
                            mode: SpindleMode::BinarySign,
                            binary_dims: 256,
                            ..SpindleConfig::default()
                        },
                    )
                    .unwrap(),
                )
                .unwrap()
            })
            .collect();
        let scalar_overlap = average_overlap_at_k(&baseline, &scalar_docs, &queries, k);
        let binary_overlap = average_overlap_at_k(&baseline, &binary_docs, &queries, k);

        println!("top-k overlap@{k}: scalar={scalar_overlap:.3} binary={binary_overlap:.3}");

        assert!(scalar_overlap >= 0.95);
        assert!(binary_overlap >= 0.35);
    }

    // -----------------------------------------------------------------------
    // TurboProd tests
    // -----------------------------------------------------------------------

    #[test]
    fn turbo_prod_round_trip_preserves_dim() {
        let config = SpindleConfig {
            mode: SpindleMode::TurboProd,
            turbo_dims: 8,
            ..SpindleConfig::default()
        };
        let data: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let bytes = encode_vector(&data, &config).unwrap();
        assert!(is_spindle_payload(&bytes));
        assert_eq!(bytes[4], TAG_TURBO_PROD);

        let decoded = decode_vector(&bytes).unwrap();
        assert_eq!(decoded.len(), data.len());
        assert!(decoded.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn turbo_prod_legacy_payload_still_scores() {
        let config = SpindleConfig {
            mode: SpindleMode::TurboProd,
            turbo_dims: 16,
            keep_original: false,
            rescore: false,
            ..SpindleConfig::default()
        };
        let query: Vec<f32> = vec![
            0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2, 0.1, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7,
        ];
        let query_f64 = query.iter().map(|&v| v as f64).collect::<Vec<_>>();
        let legacy = encode_turbo_prod(&query_f64, config.turbo_dims).unwrap();
        assert_eq!(legacy[4], TAG_TURBO_PROD);

        let prepared = prepare_query(&query, &config).unwrap();
        let approx = score_encoded(&prepared, &legacy).unwrap().unwrap();
        assert!(approx.unit_dot.is_finite());
        assert!(approx.unit_dot > 0.0);
    }

    #[test]
    fn turbo_prod_truncated_payload_is_rejected() {
        let config = SpindleConfig {
            mode: SpindleMode::TurboProd,
            turbo_dims: 8,
            ..SpindleConfig::default()
        };
        let data: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let mut bytes = encode_vector(&data, &config).unwrap();
        bytes.pop();

        let result = decode_vector(&bytes);
        assert!(matches!(result, Err(VectorError::InvalidVectorData)));
    }

    #[test]
    fn turbo_prod_scoring_prefers_matching_payload() {
        let config = SpindleConfig {
            mode: SpindleMode::TurboProd,
            turbo_dims: 16,
            keep_original: false,
            rescore: true,
            ..SpindleConfig::default()
        };
        let query: Vec<f32> = vec![
            0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2, 0.1, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7,
        ];
        let matching = encode_vector(&query, &config).unwrap();
        let opposite =
            encode_vector(&query.iter().map(|v| -*v).collect::<Vec<f32>>(), &config).unwrap();

        let prepared = prepare_query(&query, &config).unwrap();
        let match_score = score_encoded(&prepared, &matching)
            .unwrap()
            .unwrap()
            .unit_dot;
        let opp_score = score_encoded(&prepared, &opposite)
            .unwrap()
            .unwrap()
            .unit_dot;

        assert!(
            match_score > opp_score,
            "matching {match_score} should beat opposite {opp_score}"
        );
    }

    #[test]
    fn turbo_prod_scoring_is_approximately_unbiased() {
        let dim = 128;
        let config = SpindleConfig {
            mode: SpindleMode::TurboProd,
            turbo_dims: dim,
            ..SpindleConfig::default()
        };

        let mut rng = StdRng::seed_from_u64(77);
        let doc: Vec<f32> = (0..dim)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect();
        let query: Vec<f32> = (0..dim)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect();
        let true_cos = cosine_similarity(&doc, &query);

        let encoded = encode_vector(&doc, &config).unwrap();
        let prepared = prepare_query(&query, &config).unwrap();
        let approx = score_encoded(&prepared, &encoded).unwrap().unwrap();

        // unit_dot should approximate true cosine (unbiased within noise)
        let error = (approx.unit_dot - true_cos).abs();
        assert!(
            error < 0.3,
            "unit_dot {:.4} vs true cosine {:.4}, error {:.4} too large",
            approx.unit_dot,
            true_cos,
            error
        );
    }

    // -----------------------------------------------------------------------
    // 3-bit packing tests
    // -----------------------------------------------------------------------

    #[test]
    fn pack_3bit_round_trip_all_values() {
        // All valid 3-bit codes: 0-7
        let codes: Vec<u8> = (0..8).collect();
        let packed = pack_3bit(&codes);
        let unpacked = unpack_3bit(&packed, codes.len());
        assert_eq!(codes, unpacked);
    }

    #[test]
    fn pack_3bit_round_trip_various_lengths() {
        for len in [1, 2, 3, 7, 8, 9, 15, 16, 100, 768] {
            let codes: Vec<u8> = (0..len).map(|i| (i % 8) as u8).collect();
            let packed = pack_3bit(&codes);
            assert_eq!(packed.len(), (len * 3 + 7) / 8);
            let unpacked = unpack_3bit(&packed, len);
            assert_eq!(codes, unpacked, "failed at len={len}");
        }
    }

    #[test]
    fn pack_3bit_round_trip_max_values() {
        // All 7s (binary 111)
        let codes = vec![7u8; 768];
        let packed = pack_3bit(&codes);
        let unpacked = unpack_3bit(&packed, 768);
        assert_eq!(codes, unpacked);
    }

    #[test]
    fn packed_3bit_code_matches_unpack() {
        for len in [1, 2, 3, 7, 8, 9, 15, 16, 100, 768] {
            let codes: Vec<u8> = (0..len).map(|i| ((i * 5 + 3) % 8) as u8).collect();
            let packed = pack_3bit(&codes);
            let unpacked = unpack_3bit(&packed, len);
            for idx in 0..len {
                assert_eq!(packed_3bit_code(&packed, idx), unpacked[idx]);
            }
        }
    }

    #[test]
    fn mse_3bit_dot_matches_per_code_loop() {
        let dim = 257;
        let codes: Vec<u8> = (0..dim).map(|idx| ((idx * 7 + 2) % 8) as u8).collect();
        let packed = pack_3bit(&codes);
        let query: Vec<f64> = (0..dim)
            .map(|idx| ((idx as f64 * 0.013).sin() * 0.5) + 0.1)
            .collect();
        let codebook = turbo_prod_codebook(dim);

        let reference = codes
            .iter()
            .zip(query.iter())
            .map(|(code, value)| codebook[*code as usize] as f64 * value)
            .sum::<f64>();
        let optimized = mse_3bit_dot(&packed, &query, &codebook, dim);

        assert!((optimized - reference).abs() < 1e-12);
    }

    #[test]
    fn qjl_signed_sum_matches_reference_for_partial_and_full_bytes() {
        for dim in [1, 7, 8, 9, 31, 64, 127] {
            let query: Vec<f64> = (0..dim)
                .map(|idx| ((idx as f64 * 0.031).cos() * 0.25) - 0.05)
                .collect();
            let signs = pack_bits(
                &(0..dim)
                    .map(|idx| (idx * 11 + 5) % 3 != 0)
                    .collect::<Vec<_>>(),
            );
            let reference = query
                .iter()
                .enumerate()
                .map(|(idx, value)| {
                    let bit = (signs[idx / 8] >> (idx % 8)) & 1;
                    if bit == 1 {
                        *value
                    } else {
                        -*value
                    }
                })
                .sum::<f64>();
            let optimized = qjl_signed_sum(&query, &signs, dim);

            assert!(
                (optimized - reference).abs() < 1e-10,
                "dim={dim} optimized={optimized} reference={reference}"
            );
        }
    }

    #[test]
    fn signed_bit_dot_uses_popcount_without_changing_unaligned_ranges() {
        let lhs_bits: Vec<bool> = (0..73).map(|idx| idx % 5 != 0).collect();
        let rhs_bits: Vec<bool> = (0..73).map(|idx| idx % 7 <= 2).collect();
        let lhs = pack_bits(&lhs_bits);
        let rhs = pack_bits(&rhs_bits);

        for (start, dim) in [(0, 73), (1, 32), (3, 57), (8, 16), (11, 29)] {
            let reference = (start..start + dim)
                .map(|idx| {
                    if lhs_bits[idx] == rhs_bits[idx] {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .sum::<f64>();
            assert_eq!(signed_bit_dot(&lhs, &rhs, start, dim), reference);
        }
    }
}
