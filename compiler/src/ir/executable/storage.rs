use crate::ir::declarative::AttrValue;

use super::{DType, ShapeExpr, SlotId};

#[derive(Debug, Clone, Default)]
pub struct StoragePlan {
    pub slots: Vec<SlotSpec>,
}

#[derive(Debug, Clone)]
pub struct SlotSpec {
    pub id: SlotId,
    pub name: String,
    pub dtype: DType,
    pub shape: ShapeExpr,
    pub default: Option<AttrValue>,
}
