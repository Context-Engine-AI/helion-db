use super::return_values::ReturnValue;
use sonic_rs::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Deserialize, Debug, Clone)]
pub struct Remapping {
    pub exclude: bool,
    pub new_name: Option<String>,
    pub return_value: ReturnValue,
}

impl Serialize for Remapping {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        self.return_value.serialize(serializer)
    }
}

impl Remapping {
    pub fn new(exclude: bool, new_name: Option<String>, return_value: Option<ReturnValue>) -> Self {
        assert!(
            !exclude || (!new_name.is_some() || !return_value.is_some()),
            "Cannot have both exclude and new_name set"
        );
        Self {
            exclude,
            new_name,
            return_value: return_value.unwrap_or_default(),
        }
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct ResponseRemapping {
    pub remappings: HashMap<String, Remapping>,
    pub should_spread: bool,
}

impl Serialize for ResponseRemapping {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        self.remappings.serialize(serializer)
    }
}

impl ResponseRemapping {
    pub fn new(remappings: HashMap<String, Remapping>, should_spread: bool) -> Self {
        Self {
            remappings,
            should_spread,
        }
    }

    pub fn insert(&mut self, key: String, remapping: Remapping) {
        self.remappings.insert(key, remapping);
    }
}
