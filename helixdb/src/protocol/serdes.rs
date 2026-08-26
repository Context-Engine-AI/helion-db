// use std::collections::HashMap;

// use bincode::{config, BorrowDecode, Decode, Encode};

// use super::{filterable::Filterable, value::Value};

// pub trait HelixSerde<'a>: Encode + Decode<()> + BorrowDecode<'a, ()> {}

// impl<'a, T: Encode + Decode<()> + BorrowDecode<'a, ()>> HelixSerde<'a> for T {}

// pub trait HelixSerdeConfig<'a, T: HelixSerde<'a>> {
//     fn config() -> config::Configuration;
// }

// pub trait HelixSerdeDecode<'a, T: HelixSerde<'a>> {
//     fn helix_decode(data: &'a [u8]) -> Result<T, bincode::error::DecodeError>;
// }

// pub trait HelixSerdeEncode<'a, T: HelixSerde<'a>> {
//     fn helix_encode(src: &T) -> Result<Vec<u8>, bincode::error::EncodeError>;
// }

// pub trait HasLength {
//     fn len(&self) -> usize;
// }

// impl<T> HasLength for Vec<T> {
//     fn len(&self) -> usize {
//         self.len()
//     }
// }

// impl HasLength for String {
//     fn len(&self) -> usize {
//         self.len()
//     }
// }

// impl<K, V> HasLength for HashMap<K, V> {
//     fn len(&self) -> usize {
//         self.len()
//     }
// }

// // Generic implementation for anything that implements HelixSerde but not HasLength
// impl<'a, T: HelixSerde<'a>> HelixSerdeConfig<'a, T> for T {
//     #[inline(always)]
//     fn config() -> config::Configuration {
//         config::standard()
//     }
// }

// impl<'a, T: HelixSerde<'a>> HelixSerdeDecode<'a, T> for T {
//     #[inline(always)]
//     fn helix_decode(data: &'a [u8]) -> Result<T, bincode::error::DecodeError> {
//         let config = Self::config();
//         let (result, _) = bincode::borrow_decode_from_slice(data, config)?;
//         Ok(result)
//     }
// }

// impl<'a, T: HelixSerde<'a> + Filterable<'a>> HelixSerdeEncode<'a, T> for T {
//     #[inline(always)]
//     fn helix_encode(src: &T) -> Result<Vec<u8>, bincode::error::EncodeError> {
//         let config = Self::config();
//         let mut data = Vec::with_capacity(std::mem::size_of::<T>());
//         bincode::encode_into_slice(src, &mut data, config)?;
//         Ok(data)
//     }
// }

// impl<'a> HelixSerdeEncode<'a, Value> for Value {
//     #[inline(always)]
//     fn helix_encode(src: &Value) -> Result<Vec<u8>, bincode::error::EncodeError> {
//         let config = Self::config();
//         let mut data = Vec::with_capacity(std::mem::size_of::<Value>());
//         bincode::encode_into_slice(src, &mut data, config)?;
//         Ok(data)
//     }
// }
// // Specialized implementation for anything that implements HelixSerde and HasLength
// impl<'a, U: HelixSerde<'a>> HelixSerdeEncode<'a, Vec<U>> for Vec<U> {
//     #[inline(always)]
//     fn helix_encode(src: &Vec<U>) -> Result<Vec<u8>, bincode::error::EncodeError> {
//         let config = Self::config();
//         let mut data = Vec::with_capacity(src.len() * std::mem::size_of::<U>());
//         bincode::encode_into_slice(src, &mut data, config)?;
//         Ok(data)
//     }
// }

// impl<'a, U: HelixSerde<'a>> HelixSerdeEncode<'a, HashMap<String, U>> for HashMap<String, U> {
//     #[inline(always)]
//     fn helix_encode(src: &HashMap<String, U>) -> Result<Vec<u8>, bincode::error::EncodeError> {
//         let config = Self::config();
//         let mut data = Vec::with_capacity(
//             src.len() * (std::mem::size_of::<U>() + std::mem::size_of::<String>()),
//         );
//         bincode::encode_into_slice(src, &mut data, config)?;
//         Ok(data)
//     }
// }
