use crate::{
    helix_engine::types::VectorError,
    protocol::{
        filterable::{Filterable, FilterableType},
        return_values::ReturnValue,
        value::Value,
    },
};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, collections::HashMap};

#[repr(C, align(16))]
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct HVector {
    pub id: u128,
    pub is_deleted: bool,
    pub level: usize,
    pub distance: Option<f32>,
    data: Vec<f32>,
    pub properties: HashMap<String, Value>,
}

impl Eq for HVector {}

impl PartialOrd for HVector {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HVector {
    /// Reversed distance order (closest is "greatest"). Total: a NaN distance
    /// ranks as farthest instead of comparing Equal to everything, which
    /// would break heap invariants; an unset distance ranks below any set one
    /// (same as the previous `Option::partial_cmp`).
    fn cmp(&self, other: &Self) -> Ordering {
        match (other.distance, self.distance) {
            (Some(lhs), Some(rhs)) => nan_as_farthest(lhs).total_cmp(&nan_as_farthest(rhs)),
            (lhs, rhs) => lhs.is_some().cmp(&rhs.is_some()),
        }
    }
}

#[inline(always)]
fn nan_as_farthest(distance: f32) -> f32 {
    if distance.is_nan() {
        f32::INFINITY
    } else {
        distance
    }
}

pub trait DistanceCalc {
    fn distance(from: &HVector, to: &HVector) -> Result<f32, VectorError>;
}

impl DistanceCalc for HVector {
    #[inline(always)]
    #[cfg(feature = "cosine")]
    fn distance(from: &HVector, to: &HVector) -> Result<f32, VectorError> {
        from.cosine_similarity(to)
    }
}

impl HVector {
    #[inline(always)]
    pub fn new(id: u128, data: Vec<f32>) -> Self {
        HVector {
            id,
            is_deleted: false,
            level: 0,
            data,
            distance: None,
            properties: HashMap::new(),
        }
    }

    #[inline(always)]
    pub fn from_slice(id: u128, level: usize, data: Vec<f32>) -> Self {
        HVector {
            id,
            is_deleted: false,
            level,
            data,
            distance: None,
            properties: HashMap::new(),
        }
    }

    #[inline(always)]
    pub fn get_data(&self) -> &[f32] {
        &self.data
    }

    #[inline(always)]
    pub fn replace_data(&mut self, data: Vec<f32>) {
        self.data = data;
    }

    #[inline(always)]
    pub fn get_id(&self) -> u128 {
        self.id
    }

    #[inline(always)]
    pub fn get_level(&self) -> usize {
        self.level
    }

    /// Converts the HVector to a vec of bytes by accessing the data field directly
    /// and converting each f32 to a byte slice
    pub fn to_bytes(&self) -> Vec<u8> {
        let size = self.data.len() * std::mem::size_of::<f32>();
        let mut bytes = Vec::with_capacity(size);
        for &value in &self.data {
            bytes.extend_from_slice(&value.to_be_bytes());
        }
        bytes
    }

    /// Converts a byte array into a HVector by chunking the bytes into f32 values
    pub fn from_bytes(id: u128, level: usize, bytes: &[u8]) -> Result<Self, VectorError> {
        if bytes.len() % std::mem::size_of::<f32>() != 0 {
            return Err(VectorError::InvalidVectorData);
        }

        let mut data = Vec::with_capacity(bytes.len() / std::mem::size_of::<f32>());
        let chunks = bytes.chunks_exact(std::mem::size_of::<f32>());

        for chunk in chunks {
            let value = f32::from_be_bytes(chunk.try_into().unwrap());
            data.push(value);
        }

        Ok(HVector {
            id,
            is_deleted: false,
            level,
            data,
            distance: None,
            properties: HashMap::new(),
        })
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline(always)]
    pub fn distance_to(&self, other: &HVector) -> Result<f32, VectorError> {
        HVector::distance(self, other)
    }

    #[inline(always)]
    pub fn set_distance(&mut self, distance: f32) {
        self.distance = Some(distance);
    }

    #[inline(always)]
    pub fn get_distance(&self) -> f32 {
        match self.distance {
            Some(distance) => distance,
            None => panic!("Distance is not set for vector: {}", self.get_id()),
        }
    }

    /// Cosine similarity using SIMD-accelerated distance computation.
    /// Returns similarity (not distance) for backward compatibility with DistanceCalc trait.
    #[inline(always)]
    #[cfg(feature = "cosine")]
    fn cosine_similarity(&self, other: &HVector) -> Result<f32, VectorError> {
        if self.data.len() != other.data.len() {
            return Err(VectorError::InvalidVectorLength);
        }

        // cosine_f32 returns distance (1 - similarity). Convert back to similarity.
        let distance = super::simd::cosine_f32(&self.data, &other.data);
        Ok(1.0 - distance)
    }
}

// #[cfg(test)]
// mod vector_tests {
//     use super::*;

//     #[test]
//     fn test_hvector_new() {
//         let data = vec![1.0, 2.0, 3.0];
//         let vector = HVector::new("test".to_string(), data);
//         assert_eq!(vector.get_data(), &[1.0, 2.0, 3.0]);
//     }

//     #[test]
//     fn test_hvector_from_slice() {
//         let data = [1.0, 2.0, 3.0];
//         let vector = HVector::from_slice("test".to_string(), 0, data.to_vec());
//         assert_eq!(vector.get_data(), &[1.0, 2.0, 3.0]);
//     }

//     #[test]
//     fn test_hvector_distance() {
//         let v1 = HVector::new("test".to_string(), vec![1.0, 0.0]);
//         let v2 = HVector::new("test".to_string(), vec![0.0, 1.0]);
//         let distance = HVector::distance(&v1, &v2);
//         assert!((distance - 2.0_f64.sqrt()).abs() < 1e-10);
//     }

//     #[test]
//     fn test_hvector_distance_zero() {
//         let v1 = HVector::new("test".to_string(), vec![1.0, 2.0, 3.0]);
//         let v2 = HVector::new("test".to_string(), vec![1.0, 2.0, 3.0]);
//         let distance = HVector::distance(&v1, &v2);
//         assert!(distance.abs() < 1e-10);
//     }

//     #[test]
//     fn test_hvector_distance_to() {
//         let v1 = HVector::new("test".to_string(), vec![0.0, 0.0]);
//         let v2 = HVector::new("test".to_string(), vec![3.0, 4.0]);
//         let distance = v1.distance_to(&v2);
//         assert!((distance - 5.0).abs() < 1e-10);
//     }

//     #[test]
//     fn test_bytes_roundtrip() {
//         let original = HVector::new("test".to_string(), vec![1.0, 2.0, 3.0]);
//         let bytes = original.to_bytes();
//         let reconstructed = HVector::from_bytes(original.get_id(), 0, &bytes).unwrap();
//         assert_eq!(original.get_data(), reconstructed.get_data());
//     }

//     #[test]
//     fn test_hvector_len() {
//         let data = vec![1.0, 2.0, 3.0, 4.0];
//         let vector = HVector::new("test".to_string(), data);
//         assert_eq!(vector.len(), 4);
//     }

//     #[test]
//     fn test_hvector_is_empty() {
//         let empty_vector = HVector::new("test".to_string(), vec![]);
//         let non_empty_vector = HVector::new("test".to_string(), vec![1.0, 2.0]);

//         assert!(empty_vector.is_empty());
//         assert!(!non_empty_vector.is_empty());
//     }

//     #[test]
//     fn test_hvector_distance_different_dimensions() {
//         let v1 = HVector::new("test".to_string(), vec![1.0, 2.0, 3.0]);
//         let v2 = HVector::new("test".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
//         let distance = HVector::distance(&v1, &v2);
//         assert!(distance.is_finite());
//     }

//     #[test]
//     fn test_hvector_large_values() {
//         let v1 = HVector::new("test".to_string(), vec![1e6, 2e6]);
//         let v2 = HVector::new("test".to_string(), vec![1e6, 2e6]);
//         let distance = HVector::distance(&v1, &v2);
//         assert!(distance.abs() < 1e-10);
//     }

//     #[test]
//     fn test_hvector_negative_values() {
//         let v1 = HVector::new("test".to_string(), vec![-1.0, -2.0]);
//         let v2 = HVector::new("test".to_string(), vec![1.0, 2.0]);
//         let distance = HVector::distance(&v1, &v2);
//         assert!((distance - (20.0_f64).sqrt()).abs() < 1e-10);
//     }

//     #[test]
//     fn test_hvector_cosine_similarity() {
//         let v1 = HVector::new("test".to_string(), vec![1.0, 2.0, 3.0]);
//         let v2 = HVector::new("test".to_string(), vec![4.0, 5.0, 6.0]);
//         let similarity = v1.cosine_similarity(&v2);
//         assert!((similarity - 0.9746318461970762).abs() < 1e-10);
//     }
// }

impl Filterable for HVector {
    fn type_name(&self) -> FilterableType {
        FilterableType::Vector
    }

    fn id(&self) -> &u128 {
        &self.id
    }

    fn uuid(&self) -> String {
        uuid::Uuid::from_u128(self.id).to_string()
    }

    fn label(&self) -> &str {
        "vector"
    }

    fn from_node(&self) -> u128 {
        unreachable!()
    }

    fn from_node_uuid(&self) -> String {
        unreachable!()
    }

    fn to_node(&self) -> u128 {
        unreachable!()
    }

    fn to_node_uuid(&self) -> String {
        unreachable!()
    }

    fn properties(self) -> HashMap<String, Value> {
        let mut properties = HashMap::new();
        properties.insert(
            "data".to_string(),
            Value::Array(self.data.iter().map(|f| Value::F32(*f)).collect()),
        );
        properties
    }

    fn properties_mut(&mut self) -> &mut HashMap<String, Value> {
        unreachable!()
    }

    fn properties_ref(&self) -> &HashMap<String, Value> {
        unreachable!()
    }

    // TODO: Implement this
    fn check_property(&self, _key: &str) -> Option<&Value> {
        unreachable!()
    }

    fn find_property(
        &self,
        _key: &str,
        _secondary_properties: &HashMap<String, ReturnValue>,
        _property: &mut ReturnValue,
    ) -> Option<&ReturnValue> {
        unreachable!()
    }
}
