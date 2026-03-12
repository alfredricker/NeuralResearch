use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
}

pub type AttrBag = BTreeMap<String, AttrValue>;
