use indexmap::IndexMap;
use super::node::Node;
use super::port::{PortSpec, PortValues};

/// A directed connection between two ports in a `SubgraphDef`.
#[derive(Clone, Debug)]
pub struct Wire {
    pub src_node: String,
    pub src_port: &'static str,
    pub dst_node: String,
    pub dst_port: &'static str,
    /// If true, this wire carries values from the *previous* tick (breaks
    /// topological cycles).
    pub recurrent: bool,
}

/// A child entry inside a `SubgraphDef` — either a concrete node or a nested
/// subgraph that will be recursively flattened.
pub enum NodeOrSubgraph {
    Node(Box<dyn Node + Send>),
    Subgraph(SubgraphDef),
}

/// Build-time logical graph.  Can be nested arbitrarily deep via
/// `NodeOrSubgraph::Subgraph`.
pub struct SubgraphDef {
    pub name: String,
    pub(super) children: IndexMap<String, NodeOrSubgraph>,
    pub(super) wires: Vec<Wire>,
    /// (external_port_name, child_name, child_port_name)
    pub(super) exposed_inputs: Vec<(&'static str, String, &'static str)>,
    pub(super) exposed_outputs: Vec<(&'static str, String, &'static str)>,
}

impl SubgraphDef {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            children: IndexMap::new(),
            wires: Vec::new(),
            exposed_inputs: Vec::new(),
            exposed_outputs: Vec::new(),
        }
    }
}

// ─── Run-time flat wire ───────────────────────────────────────────────────────

/// A connection in the flattened graph, using node indices rather than names.
pub struct FlatWire {
    pub src: usize,
    pub src_port: &'static str,
    pub dst: usize,
    pub dst_port: &'static str,
}

// ─── FlatGraph ────────────────────────────────────────────────────────────────

/// Run-time flat graph produced by flattening + validating a `SubgraphDef`.
///
/// When `exposed_input_specs` / `exposed_output_specs` are non-empty this
/// graph also implements `Node` and can be plugged into a parent
/// `NetworkBuilder` as a composable template.
pub struct FlatGraph {
    pub(super) nodes: Vec<Box<dyn Node + Send>>,
    pub(super) feedforward_wires: Vec<FlatWire>,
    pub(super) recurrent_wires: Vec<FlatWire>,
    pub(super) exec_order: Vec<usize>,
    pub(super) input_bufs: Vec<PortValues>,
    pub(super) output_bufs: Vec<PortValues>,
    /// Previous-tick snapshot of recurrent-source output buffers.
    pub(super) recurrent_bufs: Vec<PortValues>,

    // ── Template / Node interface ──────────────────────────────────────────
    /// Declared input ports when this graph is used as a `Node`.
    pub(super) exposed_input_specs: Vec<PortSpec>,
    /// Declared output ports when this graph is used as a `Node`.
    pub(super) exposed_output_specs: Vec<PortSpec>,
    /// Maps exposed-input index → (internal node_idx, port name).
    pub(super) input_bindings: Vec<(usize, &'static str)>,
    /// Maps exposed-output index → (internal node_idx, port name).
    pub(super) output_bindings: Vec<(usize, &'static str)>,
}

impl FlatGraph {
    // ── Root-level entry points ────────────────────────────────────────────

    /// One tick for a *root* graph: propagate activations through all nodes in
    /// topological order.
    ///
    /// `external` is applied to nodes that have no feedforward inputs
    /// (sensory nodes).  Its ports must match the sensory node's input ports.
    pub fn tick(&mut self, external: &PortValues) {
        // 1. Zero all input buffers.
        for buf in &mut self.input_bufs {
            buf.zero_all();
        }

        // 2. Find sensory nodes (no incoming feedforward wires) and copy
        //    external into their input buffers — port by port.
        let has_ff_input: Vec<bool> = {
            let mut flags = vec![false; self.nodes.len()];
            for w in &self.feedforward_wires {
                flags[w.dst] = true;
            }
            flags
        };

        for (i, flag) in has_ff_input.iter().enumerate() {
            if !flag {
                // Copy all matching ports from external into input_bufs[i].
                for port in self.nodes[i].input_ports() {
                    if let Some(src) = external.get(port.name) {
                        if let Some(dst) = self.input_bufs[i].get_mut(port.name) {
                            let len = dst.len().min(src.len());
                            dst[..len].copy_from_slice(&src[..len]);
                        }
                    }
                }
            }
        }

        self.tick_internal();
    }

    /// Hebbian learning at every node (same order as tick).
    /// For root-level use — reads from the `input_bufs` left by the previous
    /// `tick` call.
    pub fn learn(&mut self, _external: &PortValues) {
        for &i in &self.exec_order.clone() {
            self.nodes[i].learn(&self.input_bufs[i]);
        }
    }

    /// Number of nodes in the flat graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    // ── Internal implementation ────────────────────────────────────────────

    /// Execute steps 3–5 of a tick (recurrent copy → exec loop → snapshot).
    /// Called by both the root `tick` and the `Node::tick` implementation.
    fn tick_internal(&mut self) {
        // 3. Copy recurrent buffers (prev-tick values) → destination inputs.
        for w in &self.recurrent_wires {
            let src_data: Vec<f32> = self.recurrent_bufs[w.src]
                .get(w.src_port)
                .map(|s| s.to_vec())
                .unwrap_or_default();
            apply_wire_data(
                &src_data,
                w.dst_port,
                &mut self.input_bufs[w.dst],
                self.nodes[w.dst].input_ports(),
            );
        }

        // 4. Execute nodes in topological order; fan-out outputs downstream.
        let order = self.exec_order.clone();
        for &i in &order {
            // Split borrows: tick reads input_bufs[i], writes output_bufs[i].
            let mut out = std::mem::replace(
                &mut self.output_bufs[i],
                PortValues::zeros_from(self.nodes[i].output_ports()),
            );
            self.nodes[i].tick(&self.input_bufs[i], &mut out);
            self.output_bufs[i] = out;

            // Fan-out: write output into downstream input_bufs.
            let targets: Vec<(usize, &'static str, &'static str)> = self
                .feedforward_wires
                .iter()
                .filter(|w| w.src == i)
                .map(|w| (w.dst, w.src_port, w.dst_port))
                .collect();

            for (dst, src_port, dst_port) in targets {
                let src_data: Vec<f32> = self.output_bufs[i]
                    .get(src_port)
                    .map(|s| s.to_vec())
                    .unwrap_or_default();
                let dst_ports = self.nodes[dst].input_ports().to_vec();
                apply_wire_data(&src_data, dst_port, &mut self.input_bufs[dst], &dst_ports);
            }
        }

        // 5. Snapshot recurrent sources for next tick.
        for w in &self.recurrent_wires {
            let src_data: Vec<f32> = self.output_bufs[w.src]
                .get(w.src_port)
                .map(|s| s.to_vec())
                .unwrap_or_default();
            if let Some(dst) = self.recurrent_bufs[w.src].get_mut(w.src_port) {
                let len = dst.len().min(src_data.len());
                dst[..len].copy_from_slice(&src_data[..len]);
            }
        }
    }
}

// ─── impl Node for FlatGraph (template interface) ────────────────────────────

/// A compiled `FlatGraph` with exposed ports can be used as a `Node` inside a
/// parent `NetworkBuilder`, turning it into a reusable template.
impl Node for FlatGraph {
    fn input_ports(&self) -> &[PortSpec] {
        &self.exposed_input_specs
    }

    fn output_ports(&self) -> &[PortSpec] {
        &self.exposed_output_specs
    }

    fn tick(&mut self, inputs: &PortValues, outputs: &mut PortValues) {
        // Zero all internal input buffers.
        for buf in &mut self.input_bufs {
            buf.zero_all();
        }

        // Route exposed inputs → internal nodes' input_bufs.
        // Collect bindings first to avoid simultaneous borrow conflicts.
        let in_bindings: Vec<(&'static str, usize, &'static str)> = self
            .exposed_input_specs
            .iter()
            .zip(self.input_bindings.iter())
            .map(|(spec, &(node_idx, port))| (spec.name, node_idx, port))
            .collect();
        for &(spec_name, node_idx, port) in &in_bindings {
            if let Some(src) = inputs.get(spec_name) {
                let src = src.to_vec();
                if let Some(dst) = self.input_bufs[node_idx].get_mut(port) {
                    let len = dst.len().min(src.len());
                    dst[..len].copy_from_slice(&src[..len]);
                }
            }
        }

        self.tick_internal();

        // Read internal output_bufs → exposed outputs.
        let out_bindings: Vec<(&'static str, usize, &'static str)> = self
            .exposed_output_specs
            .iter()
            .zip(self.output_bindings.iter())
            .map(|(spec, &(node_idx, port))| (spec.name, node_idx, port))
            .collect();
        for &(spec_name, node_idx, port) in &out_bindings {
            if let Some(src_slice) = self.output_bufs[node_idx].get(port) {
                let src = src_slice.to_vec();
                if let Some(dst) = outputs.get_mut(spec_name) {
                    let len = dst.len().min(src.len());
                    dst[..len].copy_from_slice(&src[..len]);
                }
            }
        }
    }

    fn learn(&mut self, inputs: &PortValues) {
        // Route exposed inputs → internal nodes' input_bufs.
        let in_bindings: Vec<(&'static str, usize, &'static str)> = self
            .exposed_input_specs
            .iter()
            .zip(self.input_bindings.iter())
            .map(|(spec, &(node_idx, port))| (spec.name, node_idx, port))
            .collect();
        for &(spec_name, node_idx, port) in &in_bindings {
            if let Some(src) = inputs.get(spec_name) {
                let src = src.to_vec();
                if let Some(dst) = self.input_bufs[node_idx].get_mut(port) {
                    let len = dst.len().min(src.len());
                    dst[..len].copy_from_slice(&src[..len]);
                }
            }
        }
        for &i in &self.exec_order.clone() {
            self.nodes[i].learn(&self.input_bufs[i]);
        }
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Apply `src_data` onto the named destination port in `dst_buf`, respecting
/// the port's aggregation policy.
fn apply_wire_data(
    src_data: &[f32],
    dst_port: &str,
    dst_buf: &mut PortValues,
    dst_specs: &[super::port::PortSpec],
) {
    use super::port::Aggregation;

    let Some(spec) = dst_specs.iter().find(|s| s.name == dst_port) else {
        return;
    };
    let Some(dst) = dst_buf.get_mut(dst_port) else {
        return;
    };

    match spec.agg {
        Aggregation::Sum => {
            let len = dst.len().min(src_data.len());
            for k in 0..len {
                dst[k] += src_data[k];
            }
        }
        Aggregation::Concat => {
            let cursor = dst.iter().position(|&x| x == 0.0).unwrap_or(dst.len());
            let space = dst.len().saturating_sub(cursor);
            let len = space.min(src_data.len());
            dst[cursor..cursor + len].copy_from_slice(&src_data[..len]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::port::{Aggregation, PortSpec, PortValues};
    use super::super::node::Node;

    struct PassThrough {
        ins: Vec<PortSpec>,
        outs: Vec<PortSpec>,
    }
    impl PassThrough {
        fn new(dim: usize) -> Self {
            Self {
                ins:  vec![PortSpec { name: "in",  dim, agg: Aggregation::Concat }],
                outs: vec![PortSpec { name: "out", dim, agg: Aggregation::Concat }],
            }
        }
    }
    impl Node for PassThrough {
        fn input_ports(&self)  -> &[PortSpec] { &self.ins  }
        fn output_ports(&self) -> &[PortSpec] { &self.outs }
        fn tick(&mut self, inputs: &PortValues, outputs: &mut PortValues) {
            let src = inputs.get("in").unwrap().to_vec();
            outputs.get_mut("out").unwrap().copy_from_slice(&src);
        }
        fn learn(&mut self, _: &PortValues) {}
    }

    /// Build a minimal FlatGraph by hand (no builder) and verify tick works.
    #[test]
    fn flat_graph_tick_propagates() {
        let n0 = Box::new(PassThrough::new(3)) as Box<dyn Node + Send>;
        let n1 = Box::new(PassThrough::new(3)) as Box<dyn Node + Send>;

        let in0  = PortValues::zeros_from(n0.input_ports());
        let out0 = PortValues::zeros_from(n0.output_ports());
        let in1  = PortValues::zeros_from(n1.input_ports());
        let out1 = PortValues::zeros_from(n1.output_ports());
        let rec0 = PortValues::zeros_from(n0.output_ports());
        let rec1 = PortValues::zeros_from(n1.output_ports());

        let mut fg = FlatGraph {
            nodes: vec![n0, n1],
            feedforward_wires: vec![FlatWire {
                src: 0, src_port: "out",
                dst: 1, dst_port: "in",
            }],
            recurrent_wires: vec![],
            exec_order: vec![0, 1],
            input_bufs:    vec![in0,  in1],
            output_bufs:   vec![out0, out1],
            recurrent_bufs: vec![rec0, rec1],
            exposed_input_specs:  vec![],
            exposed_output_specs: vec![],
            input_bindings:  vec![],
            output_bindings: vec![],
        };

        let mut ext = PortValues::zeros_from(&[
            PortSpec { name: "in", dim: 3, agg: Aggregation::Concat }
        ]);
        ext.get_mut("in").unwrap().copy_from_slice(&[1.0, 2.0, 3.0]);

        fg.tick(&ext);

        // node 1's input should contain [1,2,3] forwarded from node 0's output.
        assert_eq!(fg.input_bufs[1].get("in").unwrap(), &[1.0f32, 2.0, 3.0]);
        // node 1's output should also be [1,2,3].
        assert_eq!(fg.output_bufs[1].get("out").unwrap(), &[1.0f32, 2.0, 3.0]);
    }
}
