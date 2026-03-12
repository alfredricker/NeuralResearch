use crate::ir::declarative::{AttrBag, EndpointRef, GraphId, GroupId, GroupRole};

use super::{EdgeId, KernelId, NodeId, SlotId, StoragePlan};

#[derive(Debug, Clone)]
pub struct ExecutableGraph {
    pub graph_id: GraphId,
    pub nodes: Vec<ExecNode>,
    pub edges: Vec<ExecEdge>,
    pub schedule: Vec<ExecStep>,
    pub storage: StoragePlan,
    pub group_ranges: Vec<GroupRuntimeRange>,
}

#[derive(Debug, Clone)]
pub struct GroupRuntimeRange {
    pub group_id: GroupId,
    pub role: GroupRole,
    pub start: NodeId,
    pub len: u32,
}

#[derive(Debug, Clone)]
pub struct ExecNode {
    pub id: NodeId,
    pub graph_id: GraphId,
    pub group_id: GroupId,
    pub kernel: KernelId,
    pub inputs: Vec<PortSpec>,
    pub outputs: Vec<PortSpec>,
    pub state_slots: Vec<SlotBinding>,
    pub param_slots: Vec<SlotBinding>,
    pub attrs: AttrBag,
}

#[derive(Debug, Clone)]
pub struct ExecEdge {
    pub id: EdgeId,
    pub from: NodeId,
    pub to: NodeId,
    pub kernel: EdgeKernel,
    pub weight_slot: Option<SlotId>,
    pub attrs: AttrBag,
}

#[derive(Debug, Clone)]
pub struct ExecExternalLink {
    pub from: EndpointRef,
    pub to: EndpointRef,
    pub kernel: EdgeKernel,
}

#[derive(Debug, Clone)]
pub enum EdgeKernel {
    PassThrough,
    WeightedSum,
}

#[derive(Debug, Clone)]
pub enum ExecStep {
    MessagePass,
    UpdateNodes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F64,
    I64,
    Bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeExpr {
    Scalar,
    Vector(u32),
}

#[derive(Debug, Clone)]
pub struct PortSpec {
    pub name: String,
    pub dtype: DType,
    pub shape: ShapeExpr,
}

#[derive(Debug, Clone)]
pub struct SlotBinding {
    pub name: String,
    pub slot: SlotId,
    pub dtype: DType,
    pub shape: ShapeExpr,
}
