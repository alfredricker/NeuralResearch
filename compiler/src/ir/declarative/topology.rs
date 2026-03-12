#[derive(Debug, Clone, PartialEq)]
pub enum TopologyExprIr {
    Identity,
    Dense,
    Sparse { p: f32, allow_self: bool },
    Ring { k: u32 },
    None,
    WeightedSum,
    Not(Box<TopologyExprIr>),
    And(Box<TopologyExprIr>, Box<TopologyExprIr>),
    Or(Box<TopologyExprIr>, Box<TopologyExprIr>),
}
