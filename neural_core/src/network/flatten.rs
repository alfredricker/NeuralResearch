use std::collections::{HashMap, VecDeque};
use super::graph::{FlatGraph, FlatWire, NodeOrSubgraph, SubgraphDef, Wire};
use super::node::Node;
use super::port::{Aggregation, PortValues};

/// Errors that can occur while building / flattening a network.
#[derive(Debug)]
pub enum BuildError {
    /// A wire references a node name that doesn't exist.
    UnresolvedNode(String),
    /// A wire references a port name that doesn't exist on the node.
    UnresolvedPort { node: String, port: &'static str },
    /// A `Concat` destination port has mismatched total dimension.
    ConcatDimMismatch {
        node: String,
        port: &'static str,
        expected: usize,
        got: usize,
    },
    /// All `Sum` source dims must equal destination dim.
    SumDimMismatch {
        node: String,
        port: &'static str,
        expected: usize,
        got: usize,
    },
    /// Feedforward graph has a cycle — a `.recurrent()` flag is missing.
    FeedforwardCycle,
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::UnresolvedNode(n) =>
                write!(f, "unresolved node name: '{n}'"),
            BuildError::UnresolvedPort { node, port } =>
                write!(f, "node '{node}' has no port '{port}'"),
            BuildError::ConcatDimMismatch { node, port, expected, got } =>
                write!(f, "Concat dim mismatch on {node}::{port}: expected {expected}, got total {got}"),
            BuildError::SumDimMismatch { node, port, expected, got } =>
                write!(f, "Sum dim mismatch on {node}::{port}: expected {expected}, got {got}"),
            BuildError::FeedforwardCycle =>
                write!(f, "feedforward graph has a cycle — mark recurrent wires with .recurrent()"),
        }
    }
}

impl std::error::Error for BuildError {}

// ─── Internal intermediate structures ────────────────────────────────────────

/// Port endpoint after flattening: maps (child_name, port_name) → global_idx.
type PortMap = HashMap<(String, &'static str), (usize, usize)>; // → (node_idx, port_idx)

struct Intermediate {
    nodes: Vec<Box<dyn Node + Send>>,
    node_names: Vec<String>,
    wires: Vec<Wire>,
    port_map: PortMap, // (node_name, port_name) → (global_idx, _port_idx)
}

impl Intermediate {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            node_names: Vec::new(),
            wires: Vec::new(),
            port_map: HashMap::new(),
        }
    }
}

// ─── Recursive flattening ─────────────────────────────────────────────────────

/// Recursively flatten `def` into `out`, prefixing all node names with `prefix`.
///
/// Returns a map from (exposed_external_name, port_name) → global node index,
/// used by the parent to wire across subgraph boundaries.
fn flatten_def(
    def: SubgraphDef,
    prefix: &str,
    out: &mut Intermediate,
) -> HashMap<(String, &'static str), usize> {
    // Maps local child name (within this subgraph) → global index assigned.
    let mut local_to_global: HashMap<String, usize> = HashMap::new();

    for (child_name, child) in def.children {
        let full_name = if prefix.is_empty() {
            child_name.clone()
        } else {
            format!("{prefix}/{child_name}")
        };

        match child {
            NodeOrSubgraph::Node(node) => {
                let idx = out.nodes.len();
                // Register all input ports.
                for (pi, spec) in node.input_ports().iter().enumerate() {
                    out.port_map.insert((full_name.clone(), spec.name), (idx, pi));
                }
                // Register all output ports.
                for (pi, spec) in node.output_ports().iter().enumerate() {
                    out.port_map.insert((full_name.clone(), spec.name), (idx, pi));
                }
                out.node_names.push(full_name.clone());
                out.nodes.push(node);
                local_to_global.insert(child_name, idx);
            }
            NodeOrSubgraph::Subgraph(sub) => {
                // Recurse; `sub_exposed` maps (exposed_port_name, port_name) → global_idx
                let _sub_exposed = flatten_def(sub, &full_name, out);
                // (exposed port aliases are handled via wire remapping below)
                local_to_global.insert(child_name, usize::MAX); // placeholder; subgraph has no single idx
            }
        }
    }

    // Translate wires: replace local names with full names, push to out.wires.
    for wire in def.wires {
        let src_full = if prefix.is_empty() {
            wire.src_node.clone()
        } else {
            format!("{prefix}/{}", wire.src_node)
        };
        let dst_full = if prefix.is_empty() {
            wire.dst_node.clone()
        } else {
            format!("{prefix}/{}", wire.dst_node)
        };
        out.wires.push(Wire {
            src_node: src_full,
            src_port: wire.src_port,
            dst_node: dst_full,
            dst_port: wire.dst_port,
            recurrent: wire.recurrent,
        });
    }

    // Build exposed-port aliases (no new nodes).
    let mut exposed: HashMap<(String, &'static str), usize> = HashMap::new();
    for (ext_name, child_name, child_port) in def.exposed_inputs.iter().chain(def.exposed_outputs.iter()) {
        let child_full = if prefix.is_empty() {
            child_name.clone()
        } else {
            format!("{prefix}/{child_name}")
        };
        if let Some(&(node_idx, _)) = out.port_map.get(&(child_full, child_port)) {
            exposed.insert((ext_name.clone(), child_port), node_idx);
        }
    }
    exposed
}

// ─── Topological sort (Kahn's algorithm) ─────────────────────────────────────

fn kahn_sort(n: usize, ff_wires: &[FlatWire]) -> Result<Vec<usize>, BuildError> {
    let mut in_degree = vec![0usize; n];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for w in ff_wires {
        adj[w.src].push(w.dst);
        in_degree[w.dst] += 1;
    }

    let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut order = Vec::with_capacity(n);

    while let Some(u) = queue.pop_front() {
        order.push(u);
        for &v in &adj[u] {
            in_degree[v] -= 1;
            if in_degree[v] == 0 {
                queue.push_back(v);
            }
        }
    }

    if order.len() != n {
        Err(BuildError::FeedforwardCycle)
    } else {
        Ok(order)
    }
}

// ─── Public entry point ───────────────────────────────────────────────────────

pub fn flatten_and_build(def: SubgraphDef) -> Result<FlatGraph, BuildError> {
    let mut inter = Intermediate::new();
    flatten_def(def, "", &mut inter);

    let n = inter.nodes.len();

    // Resolve wires → FlatWires.
    let mut ff_wires: Vec<FlatWire> = Vec::new();
    let mut rec_wires: Vec<FlatWire> = Vec::new();

    for wire in &inter.wires {
        // Resolve source node.
        let (src_idx, _) = inter
            .port_map
            .get(&(wire.src_node.clone(), wire.src_port))
            .copied()
            .ok_or_else(|| {
                if inter.node_names.contains(&wire.src_node) {
                    BuildError::UnresolvedPort { node: wire.src_node.clone(), port: wire.src_port }
                } else {
                    BuildError::UnresolvedNode(wire.src_node.clone())
                }
            })?;

        // Resolve destination node.
        let (dst_idx, _) = inter
            .port_map
            .get(&(wire.dst_node.clone(), wire.dst_port))
            .copied()
            .ok_or_else(|| {
                if inter.node_names.contains(&wire.dst_node) {
                    BuildError::UnresolvedPort { node: wire.dst_node.clone(), port: wire.dst_port }
                } else {
                    BuildError::UnresolvedNode(wire.dst_node.clone())
                }
            })?;

        let fw = FlatWire {
            src: src_idx,
            src_port: wire.src_port,
            dst: dst_idx,
            dst_port: wire.dst_port,
        };
        if wire.recurrent {
            rec_wires.push(fw);
        } else {
            ff_wires.push(fw);
        }
    }

    // Validate aggregation dimensions.
    // Group feedforward wires by destination port.
    let mut dst_groups: HashMap<(usize, &'static str), Vec<usize>> = HashMap::new();
    for (wi, w) in ff_wires.iter().enumerate() {
        dst_groups.entry((w.dst, w.dst_port)).or_default().push(wi);
    }

    for ((dst_idx, dst_port), wire_indices) in &dst_groups {
        // Find the port spec on the destination node.
        let dst_node = &inter.nodes[*dst_idx];
        let dst_spec = dst_node
            .input_ports()
            .iter()
            .find(|s| s.name == *dst_port)
            .ok_or_else(|| BuildError::UnresolvedPort {
                node: inter.node_names[*dst_idx].clone(),
                port: dst_port,
            })?;

        match dst_spec.agg {
            Aggregation::Concat => {
                let total_src_dim: usize = wire_indices.iter().map(|&wi| {
                    let w = &ff_wires[wi];
                    inter.nodes[w.src]
                        .output_ports()
                        .iter()
                        .find(|s| s.name == w.src_port)
                        .map(|s| s.dim)
                        .unwrap_or(0)
                }).sum();
                if total_src_dim != dst_spec.dim {
                    return Err(BuildError::ConcatDimMismatch {
                        node: inter.node_names[*dst_idx].clone(),
                        port: dst_port,
                        expected: dst_spec.dim,
                        got: total_src_dim,
                    });
                }
            }
            Aggregation::Sum => {
                for &wi in wire_indices {
                    let w = &ff_wires[wi];
                    let src_dim = inter.nodes[w.src]
                        .output_ports()
                        .iter()
                        .find(|s| s.name == w.src_port)
                        .map(|s| s.dim)
                        .unwrap_or(0);
                    if src_dim != dst_spec.dim {
                        return Err(BuildError::SumDimMismatch {
                            node: inter.node_names[*dst_idx].clone(),
                            port: dst_port,
                            expected: dst_spec.dim,
                            got: src_dim,
                        });
                    }
                }
            }
        }
    }

    // Topological sort on feedforward wires.
    let exec_order = kahn_sort(n, &ff_wires)?;

    // Allocate buffers.
    let input_bufs: Vec<PortValues> = inter
        .nodes
        .iter()
        .map(|nd| PortValues::zeros_from(nd.input_ports()))
        .collect();
    let output_bufs: Vec<PortValues> = inter
        .nodes
        .iter()
        .map(|nd| PortValues::zeros_from(nd.output_ports()))
        .collect();
    let recurrent_bufs: Vec<PortValues> = inter
        .nodes
        .iter()
        .map(|nd| PortValues::zeros_from(nd.output_ports()))
        .collect();

    Ok(FlatGraph {
        nodes: inter.nodes,
        feedforward_wires: ff_wires,
        recurrent_wires: rec_wires,
        exec_order,
        input_bufs,
        output_bufs,
        recurrent_bufs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::graph::{SubgraphDef, Wire};
    use super::super::node::Node;
    use super::super::port::{Aggregation, PortSpec, PortValues};

    // ─── Minimal pass-through node ────────────────────────────────────────
    struct Echo {
        ins:  Vec<PortSpec>,
        outs: Vec<PortSpec>,
    }
    impl Echo {
        fn new(dim: usize) -> Self {
            Self {
                ins:  vec![PortSpec { name: "in",  dim, agg: Aggregation::Concat }],
                outs: vec![PortSpec { name: "out", dim, agg: Aggregation::Concat }],
            }
        }
    }
    impl Node for Echo {
        fn input_ports(&self)  -> &[PortSpec] { &self.ins  }
        fn output_ports(&self) -> &[PortSpec] { &self.outs }
        fn tick(&mut self, inp: &PortValues, out: &mut PortValues) {
            let v = inp.get("in").unwrap().to_vec();
            out.get_mut("out").unwrap().copy_from_slice(&v);
        }
        fn learn(&mut self, _: &PortValues) {}
    }

    #[test]
    fn two_node_flat_flattens_and_ticks() {
        let mut def = SubgraphDef::new("root");
        def.children.insert("a".into(), NodeOrSubgraph::Node(Box::new(Echo::new(2))));
        def.children.insert("b".into(), NodeOrSubgraph::Node(Box::new(Echo::new(2))));
        def.wires.push(Wire {
            src_node: "a".into(), src_port: "out",
            dst_node: "b".into(), dst_port: "in",
            recurrent: false,
        });

        let mut fg = flatten_and_build(def).expect("build failed");
        assert_eq!(fg.node_count(), 2);

        let mut ext = PortValues::zeros_from(&[PortSpec { name: "in", dim: 2, agg: Aggregation::Concat }]);
        ext.get_mut("in").unwrap().copy_from_slice(&[3.0, 7.0]);
        fg.tick(&ext);

        assert_eq!(fg.output_bufs[fg.exec_order[1]].get("out").unwrap(), &[3.0f32, 7.0]);
    }

    #[test]
    fn dim_mismatch_gives_error() {
        // b expects Concat dim=4, but a only outputs 2.
        let mut def = SubgraphDef::new("root");
        def.children.insert("a".into(), NodeOrSubgraph::Node(Box::new(Echo::new(2))));
        def.children.insert("b".into(), NodeOrSubgraph::Node(Box::new(
            // Manually build a node with Concat dim=4 on input.
            struct_with_input_dim(4)
        )));
        def.wires.push(Wire {
            src_node: "a".into(), src_port: "out",
            dst_node: "b".into(), dst_port: "in",
            recurrent: false,
        });
        assert!(matches!(flatten_and_build(def), Err(BuildError::ConcatDimMismatch { .. })));
    }

    #[test]
    fn cycle_without_recurrent_gives_error() {
        // a → b → a  (cycle, no recurrent flag)
        let mut def = SubgraphDef::new("root");
        def.children.insert("a".into(), NodeOrSubgraph::Node(Box::new(Echo::new(2))));
        def.children.insert("b".into(), NodeOrSubgraph::Node(Box::new(Echo::new(2))));
        def.wires.push(Wire { src_node: "a".into(), src_port: "out", dst_node: "b".into(), dst_port: "in", recurrent: false });
        def.wires.push(Wire { src_node: "b".into(), src_port: "out", dst_node: "a".into(), dst_port: "in", recurrent: false });
        assert!(matches!(flatten_and_build(def), Err(BuildError::FeedforwardCycle)));
    }

    #[test]
    fn cycle_with_recurrent_flag_succeeds() {
        let mut def = SubgraphDef::new("root");
        def.children.insert("a".into(), NodeOrSubgraph::Node(Box::new(Echo::new(2))));
        def.children.insert("b".into(), NodeOrSubgraph::Node(Box::new(Echo::new(2))));
        def.wires.push(Wire { src_node: "a".into(), src_port: "out", dst_node: "b".into(), dst_port: "in", recurrent: false });
        def.wires.push(Wire { src_node: "b".into(), src_port: "out", dst_node: "a".into(), dst_port: "in", recurrent: true });
        assert!(flatten_and_build(def).is_ok());
    }

    #[test]
    fn nested_subgraph_flattens() {
        // inner: a → b
        let mut inner = SubgraphDef::new("inner");
        inner.children.insert("a".into(), NodeOrSubgraph::Node(Box::new(Echo::new(2))));
        inner.children.insert("b".into(), NodeOrSubgraph::Node(Box::new(Echo::new(2))));
        inner.wires.push(Wire { src_node: "a".into(), src_port: "out", dst_node: "b".into(), dst_port: "in", recurrent: false });

        // outer: inner → c
        let mut outer = SubgraphDef::new("root");
        outer.children.insert("inner".into(), NodeOrSubgraph::Subgraph(inner));
        outer.children.insert("c".into(), NodeOrSubgraph::Node(Box::new(Echo::new(2))));
        outer.wires.push(Wire {
            src_node: "inner/b".into(), src_port: "out",
            dst_node: "c".into(),      dst_port: "in",
            recurrent: false,
        });

        let fg = flatten_and_build(outer).expect("nested build failed");
        assert_eq!(fg.node_count(), 3);
    }

    // Helper: build an Echo-like node with a custom input dim.
    fn struct_with_input_dim(dim: usize) -> impl Node + Send + 'static {
        struct BigIn { ins: Vec<PortSpec>, outs: Vec<PortSpec> }
        impl Node for BigIn {
            fn input_ports(&self)  -> &[PortSpec] { &self.ins  }
            fn output_ports(&self) -> &[PortSpec] { &self.outs }
            fn tick(&mut self, _: &PortValues, _: &mut PortValues) {}
            fn learn(&mut self, _: &PortValues) {}
        }
        BigIn {
            ins:  vec![PortSpec { name: "in",  dim, agg: Aggregation::Concat }],
            outs: vec![PortSpec { name: "out", dim, agg: Aggregation::Concat }],
        }
    }
}
