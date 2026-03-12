use std::collections::HashMap;

use crate::ir::declarative::{
    AttrBag, AttrValue, EndpointRef, GraphId, GroupId, GroupLinkIr, GroupRole, IrError, ModuleIr,
    TopologyExprIr,
};

pub type NodeId = u32;
pub type EdgeId = u32;
pub type SlotId = u32;
pub type KernelId = u32;

#[derive(Debug, Clone, Default)]
pub struct ExecutableModule {
    pub graphs: Vec<ExecutableGraph>,
    pub links: Vec<ExecExternalLink>,
}

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

impl ExecutableModule {
    pub fn from_declarative(module: &ModuleIr) -> Result<Self, Vec<IrError>> {
        let mut out = ExecutableModule::default();
        let mut errors = Vec::new();

        for graph in &module.graphs {
            match ExecutableGraph::from_declarative_graph(graph) {
                Ok(exec_graph) => out.graphs.push(exec_graph),
                Err(mut e) => errors.append(&mut e),
            }
        }

        out.links = module
            .links
            .iter()
            .map(|link| ExecExternalLink {
                from: link.from,
                to: link.to,
                kernel: topology_to_kernel(&link.topology),
            })
            .collect();

        if errors.is_empty() {
            Ok(out)
        } else {
            Err(errors)
        }
    }
}

impl ExecutableGraph {
    fn from_declarative_graph(graph: &crate::ir::declarative::GraphIr) -> Result<Self, Vec<IrError>> {
        let mut errors = Vec::new();
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut group_ranges = Vec::new();
        let mut group_map: HashMap<GroupId, (NodeId, u32)> = HashMap::new();

        let mut next_node_id: NodeId = 0;
        let mut next_slot_id: SlotId = 0;
        let mut storage = StoragePlan::default();

        for group in &graph.groups {
            let start = next_node_id;
            group_map.insert(group.id, (start, group.count));
            group_ranges.push(GroupRuntimeRange {
                group_id: group.id,
                role: group.role,
                start,
                len: group.count,
            });

            for _ in 0..group.count {
                let node_id = next_node_id;
                next_node_id += 1;

                let activation_slot = next_slot_id;
                next_slot_id += 1;
                let input_slot = next_slot_id;
                next_slot_id += 1;

                storage.slots.push(SlotSpec {
                    id: activation_slot,
                    name: format!("node{}_activation", node_id),
                    dtype: DType::F32,
                    shape: ShapeExpr::Scalar,
                    default: Some(AttrValue::Float(0.0)),
                });
                storage.slots.push(SlotSpec {
                    id: input_slot,
                    name: format!("node{}_input_buffer", node_id),
                    dtype: DType::F32,
                    shape: ShapeExpr::Scalar,
                    default: Some(AttrValue::Float(0.0)),
                });

                nodes.push(ExecNode {
                    id: node_id,
                    graph_id: graph.id,
                    group_id: group.id,
                    kernel: 0,
                    inputs: vec![PortSpec {
                        name: "in".to_string(),
                        dtype: DType::F32,
                        shape: ShapeExpr::Scalar,
                    }],
                    outputs: vec![PortSpec {
                        name: "out".to_string(),
                        dtype: DType::F32,
                        shape: ShapeExpr::Scalar,
                    }],
                    state_slots: vec![
                        SlotBinding {
                            name: "activation".to_string(),
                            slot: activation_slot,
                            dtype: DType::F32,
                            shape: ShapeExpr::Scalar,
                        },
                        SlotBinding {
                            name: "input_buffer".to_string(),
                            slot: input_slot,
                            dtype: DType::F32,
                            shape: ShapeExpr::Scalar,
                        },
                    ],
                    param_slots: Vec::new(),
                    attrs: group.attrs.clone(),
                });
            }
        }

        let mut next_edge_id: EdgeId = 0;
        for link in &graph.links {
            match materialize_group_link(link, &group_map, &mut next_edge_id) {
                Ok(mut link_edges) => edges.append(&mut link_edges),
                Err(err) => errors.push(err),
            }
        }

        if !errors.is_empty() {
            return Err(errors);
        }

        Ok(Self {
            graph_id: graph.id,
            nodes,
            edges,
            schedule: vec![ExecStep::MessagePass, ExecStep::UpdateNodes],
            storage,
            group_ranges,
        })
    }
}

fn materialize_group_link(
    link: &GroupLinkIr,
    groups: &HashMap<GroupId, (NodeId, u32)>,
    next_edge_id: &mut EdgeId,
) -> Result<Vec<ExecEdge>, IrError> {
    let Some((from_start, from_len)) = groups.get(&link.from).copied() else {
        return Err(IrError {
            message: format!("Executable lowering missing source group {}", link.from),
        });
    };
    let Some((to_start, to_len)) = groups.get(&link.to).copied() else {
        return Err(IrError {
            message: format!("Executable lowering missing destination group {}", link.to),
        });
    };

    let mut edges = Vec::new();
    let edge_kernel = topology_to_kernel(&link.topology);

    match &link.topology {
        TopologyExprIr::Identity => {
            let n = from_len.min(to_len);
            for i in 0..n {
                edges.push(ExecEdge {
                    id: alloc_edge_id(next_edge_id),
                    from: from_start + i,
                    to: to_start + i,
                    kernel: edge_kernel.clone(),
                    weight_slot: None,
                    attrs: link.attrs.clone(),
                });
            }
        }
        TopologyExprIr::Dense | TopologyExprIr::WeightedSum => {
            for i in 0..from_len {
                for j in 0..to_len {
                    edges.push(ExecEdge {
                        id: alloc_edge_id(next_edge_id),
                        from: from_start + i,
                        to: to_start + j,
                        kernel: edge_kernel.clone(),
                        weight_slot: None,
                        attrs: link.attrs.clone(),
                    });
                }
            }
        }
        TopologyExprIr::Sparse { p, allow_self } => {
            let p = p.clamp(0.0, 1.0);
            for i in 0..from_len {
                for j in 0..to_len {
                    if !allow_self && link.from == link.to && i == j {
                        continue;
                    }
                    if deterministic_keep(from_start + i, to_start + j, p) {
                        edges.push(ExecEdge {
                            id: alloc_edge_id(next_edge_id),
                            from: from_start + i,
                            to: to_start + j,
                            kernel: edge_kernel.clone(),
                            weight_slot: None,
                            attrs: link.attrs.clone(),
                        });
                    }
                }
            }
        }
        TopologyExprIr::Ring { k } => {
            if to_len == 0 {
                return Ok(edges);
            }
            let step = if *k == 0 { 1 } else { *k };
            for i in 0..from_len {
                let j = (i + step) % to_len;
                edges.push(ExecEdge {
                    id: alloc_edge_id(next_edge_id),
                    from: from_start + i,
                    to: to_start + j,
                    kernel: edge_kernel.clone(),
                    weight_slot: None,
                    attrs: link.attrs.clone(),
                });
            }
        }
        TopologyExprIr::None => {}
        TopologyExprIr::Not(_) | TopologyExprIr::And(_, _) | TopologyExprIr::Or(_, _) => {
            return Err(IrError {
                message: "Composite topology expressions are not executable yet".to_string(),
            });
        }
    }

    Ok(edges)
}

fn alloc_edge_id(next_edge_id: &mut EdgeId) -> EdgeId {
    let id = *next_edge_id;
    *next_edge_id += 1;
    id
}

fn topology_to_kernel(topology: &TopologyExprIr) -> EdgeKernel {
    match topology {
        TopologyExprIr::WeightedSum => EdgeKernel::WeightedSum,
        _ => EdgeKernel::PassThrough,
    }
}

fn deterministic_keep(from: NodeId, to: NodeId, p: f32) -> bool {
    if p <= 0.0 {
        return false;
    }
    if p >= 1.0 {
        return true;
    }

    let mut x = from.wrapping_mul(0x45d9f3b) ^ to.wrapping_mul(0x119de1f3);
    x ^= x >> 16;
    let frac = (x % 10_000) as f32 / 10_000.0;
    frac < p
}