use std::collections::HashMap;

use crate::ast::{
    block::{Block, BlockKind, Item},
    expr::Expr,
    io::{InputKind, OutputKind},
    link::Topology,
    program::Program,
    statement::Statement,
    var::VarDecl,
};

use super::{
    AttrBag, EndpointId, EndpointRef, ExternalLinkIr, GraphId, GraphIr, GroupId, GroupLinkIr,
    GroupRole, InterfaceDirection, InterfaceIr, InterfaceKind, IrError, ModuleIr, NodeGroupIr,
    TopologyExprIr,
};

impl ModuleIr {
    pub fn from_program(program: &Program) -> Result<Self, Vec<IrError>> {
        let mut module = ModuleIr::default();
        let mut errors = Vec::new();

        let mut next_graph_id: GraphId = 0;
        let mut next_group_id: GroupId = 0;
        let mut next_endpoint_id: EndpointId = 0;

        let mut endpoint_by_name: HashMap<String, EndpointRef> = HashMap::new();
        let mut pending_external_links: Vec<(String, String, TopologyExprIr)> = Vec::new();

        for item in program.items() {
            match item {
                Item::Statement(stmt) => match stmt {
                    Statement::Input(input_decl) => {
                        let id = next_endpoint_id;
                        next_endpoint_id += 1;

                        let endpoint = EndpointRef::Interface(id);
                        endpoint_by_name.insert(input_decl.name.clone(), endpoint);
                        module.interfaces.push(InterfaceIr {
                            id,
                            name: input_decl.name.clone(),
                            direction: InterfaceDirection::Input,
                            kind: map_input_kind(&input_decl.kind),
                            attrs: AttrBag::new(),
                        });
                    }
                    Statement::Output(output_decl) => {
                        let id = next_endpoint_id;
                        next_endpoint_id += 1;

                        let endpoint = EndpointRef::Interface(id);
                        endpoint_by_name.insert(output_decl.name.clone(), endpoint);
                        module.interfaces.push(InterfaceIr {
                            id,
                            name: output_decl.name.clone(),
                            direction: InterfaceDirection::Output,
                            kind: map_output_kind(&output_decl.kind),
                            attrs: AttrBag::new(),
                        });
                    }
                    Statement::Link(link_decl) => {
                        pending_external_links.push((
                            link_decl.from.clone(),
                            link_decl.to.clone(),
                            map_topology(&link_decl.topology),
                        ));
                    }
                    Statement::Var(_) => {
                        errors.push(IrError::new("Top-level variable declarations are not supported"));
                    }
                },
                Item::Block(block) => {
                    if !matches!(block.kind, BlockKind::Graph) {
                        errors.push(IrError::new(
                            "Only `graph { ... }` is supported in declarative lowering currently",
                        ));
                        continue;
                    }

                    let graph = lower_graph_block(
                        block,
                        next_graph_id,
                        &mut next_group_id,
                        &mut endpoint_by_name,
                        &mut errors,
                    );
                    module.graphs.push(graph);
                    next_graph_id += 1;
                }
            }
        }

        for (from_name, to_name, topology) in pending_external_links {
            let Some(from) = endpoint_by_name.get(&from_name).copied() else {
                errors.push(IrError::new(format!(
                    "Unknown link source endpoint `{}`",
                    from_name
                )));
                continue;
            };

            let Some(to) = endpoint_by_name.get(&to_name).copied() else {
                errors.push(IrError::new(format!(
                    "Unknown link destination endpoint `{}`",
                    to_name
                )));
                continue;
            };

            module.links.push(ExternalLinkIr {
                from,
                to,
                topology,
                attrs: AttrBag::new(),
            });
        }

        if errors.is_empty() {
            Ok(module)
        } else {
            Err(errors)
        }
    }
}

fn lower_graph_block(
    block: &Block,
    graph_id: GraphId,
    next_group_id: &mut GroupId,
    endpoint_by_name: &mut HashMap<String, EndpointRef>,
    errors: &mut Vec<IrError>,
) -> GraphIr {
    let mut graph = GraphIr {
        id: graph_id,
        name: None,
        parent: None,
        groups: Vec::new(),
        links: Vec::new(),
        attrs: AttrBag::new(),
    };

    let mut graph_group_names: HashMap<String, GroupId> = HashMap::new();

    for item in &block.items {
        match item {
            Item::Statement(Statement::Var(var_decl)) => match parse_nodes_var(var_decl) {
                Ok(count) => {
                    let group_id = *next_group_id;
                    *next_group_id += 1;

                    graph_group_names.insert(var_decl.name.clone(), group_id);
                    endpoint_by_name.insert(var_decl.name.clone(), EndpointRef::Group(group_id));

                    graph.groups.push(NodeGroupIr {
                        id: group_id,
                        graph: graph_id,
                        name: var_decl.name.clone(),
                        count,
                        role: GroupRole::Hidden,
                        attrs: AttrBag::new(),
                    });
                }
                Err(err) => errors.push(err),
            },
            Item::Statement(Statement::Link(link_decl)) => {
                let Some(from) = graph_group_names.get(&link_decl.from).copied() else {
                    errors.push(IrError::new(format!(
                        "Unknown group link source `{}` inside graph",
                        link_decl.from
                    )));
                    continue;
                };
                let Some(to) = graph_group_names.get(&link_decl.to).copied() else {
                    errors.push(IrError::new(format!(
                        "Unknown group link destination `{}` inside graph",
                        link_decl.to
                    )));
                    continue;
                };

                graph.links.push(GroupLinkIr {
                    from,
                    to,
                    topology: map_topology(&link_decl.topology),
                    attrs: AttrBag::new(),
                });
            }
            Item::Statement(Statement::Input(_))
            | Item::Statement(Statement::Output(_))
            | Item::Block(_) => {
                errors.push(IrError::new(
                    "Only variable node declarations and links are supported inside `graph`",
                ));
            }
        }
    }

    graph
}

fn parse_nodes_var(var_decl: &VarDecl) -> Result<u32, IrError> {
    let Expr::Call(call) = &var_decl.value else {
        return Err(IrError::new(format!(
            "Expected `{}` to be assigned from `nodes(...)`",
            var_decl.name
        )));
    };

    if call.name != "nodes" {
        return Err(IrError::new(format!(
            "Expected `{}` to be assigned from `nodes(...)`, found `{}`(...)",
            var_decl.name, call.name
        )));
    }

    match call.args.as_slice() {
        [Expr::Int(v)] if *v >= 0 => Ok(*v as u32),
        [Expr::Float(v)] if *v >= 0.0 => Ok(*v as u32),
        _ => Err(IrError::new(format!(
            "`nodes(...)` for `{}` expects one non-negative numeric argument",
            var_decl.name
        ))),
    }
}

fn map_topology(topology: &Topology) -> TopologyExprIr {
    match topology {
        Topology::Sparse(p) => TopologyExprIr::Sparse {
            p: *p as f32,
            allow_self: false,
        },
        Topology::Dense => TopologyExprIr::Dense,
        Topology::Identity => TopologyExprIr::Identity,
        Topology::WeightedSum => TopologyExprIr::WeightedSum,
    }
}

fn map_input_kind(kind: &InputKind) -> InterfaceKind {
    match kind {
        InputKind::Image(h, w, c) => InterfaceKind::Image {
            height: *h,
            width: *w,
            channels: *c,
        },
        InputKind::Language(size) => InterfaceKind::Language { token_size: *size },
    }
}

fn map_output_kind(kind: &OutputKind) -> InterfaceKind {
    match kind {
        OutputKind::Classifier(classes) => InterfaceKind::Classifier { classes: *classes },
        OutputKind::Logits(size) => InterfaceKind::Logits { size: *size },
    }
}
