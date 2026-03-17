use indexmap::IndexMap;
use super::node::Node;
use super::port::PortValues;

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
    pub(super) exposed_inputs: Vec<(String, String, &'static str)>,
    pub(super) exposed_outputs: Vec<(String, String, &'static str)>,
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
pub struct FlatGraph {
    pub(super) nodes: Vec<Box<dyn Node + Send>>,
    pub(super) feedforward_wires: Vec<FlatWire>,
    pub(super) recurrent_wires: Vec<FlatWire>,
    pub(super) exec_order: Vec<usize>,
    pub(super) input_bufs: Vec<PortValues>,
    pub(super) output_bufs: Vec<PortValues>,
    /// Previous-tick snapshot of recurrent-source output buffers.
    pub(super) recurrent_bufs: Vec<PortValues>,
}

impl FlatGraph {
    /// One tick: propagate activations through all nodes in topological order.
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
            // We need to temporarily swap out the buffers to satisfy the borrow checker.
            let mut out = std::mem::replace(
                &mut self.output_bufs[i],
                PortValues::zeros_from(self.nodes[i].output_ports()),
            );
            self.nodes[i].tick(&self.input_bufs[i], &mut out);
            self.output_bufs[i] = out;

            // Fan-out: write output into downstream input_bufs.
            // Collect wire targets first to avoid borrow conflicts.
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

    /// Hebbian learning at every node (same order as tick).
    pub fn learn(&mut self, _external: &PortValues) {
        for &i in &self.exec_order.clone() {
            self.nodes[i].learn(&self.input_bufs[i]);
        }
    }

    /// Number of nodes in the flat graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
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
            // Find the first zero-run at the tail of `dst` (the unfilled
            // portion), and write `src_data` there.
            // A simpler strategy: find offset = number of already-filled bytes.
            // We track fill position by scanning from the end.
            // Actually the cleanest approach: scan for contiguous zeros at end.
            // For correctness we track fill via a "write cursor" stored in the
            // first zero region.  Since Concat ports are zero-initialised
            // each tick, the cursor is just the index of the first 0.0 value.
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
