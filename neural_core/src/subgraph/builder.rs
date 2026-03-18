use super::flatten::{flatten_and_build, BuildError};
use super::graph::{FlatGraph, NodeOrSubgraph, SubgraphDef, Wire};
use super::node::Node;

/// Fluent builder for assembling a `SubgraphDef` and compiling it into a
/// run-time `FlatGraph`.
pub struct NetworkBuilder {
    def: SubgraphDef,
}

impl NetworkBuilder {
    pub fn new(name: &str) -> Self {
        Self { def: SubgraphDef::new(name) }
    }

    /// Add a concrete node.  `FlatGraph` implements `Node`, so a compiled
    /// template can be passed here directly.
    pub fn add_node(mut self, name: &str, node: impl Node + Send + 'static) -> Self {
        self.def.children.insert(name.to_owned(), NodeOrSubgraph::Node(Box::new(node)));
        self
    }

    /// Embed a nested subgraph (raw, uncompiled).
    pub fn add_subgraph(mut self, name: &str, sub: SubgraphDef) -> Self {
        self.def.children.insert(name.to_owned(), NodeOrSubgraph::Subgraph(sub));
        self
    }

    /// Connect `src_node::src_port` → `dst_node::dst_port`.
    ///
    /// Returns a `WireBuilder` so you can optionally call `.recurrent()`.
    pub fn wire(
        mut self,
        src: &str,
        src_port: &'static str,
        dst: &str,
        dst_port: &'static str,
    ) -> WireBuilder {
        let idx = self.def.wires.len();
        self.def.wires.push(Wire {
            src_node: src.to_owned(),
            src_port,
            dst_node: dst.to_owned(),
            dst_port,
            recurrent: false,
        });
        WireBuilder { inner: self, wire_idx: idx }
    }

    /// Expose an internal child port as an external input on this subgraph.
    /// `ext` must be a `'static` string literal — it becomes the port name on
    /// the compiled `FlatGraph` when used as a `Node`.
    pub fn expose_input(mut self, ext: &'static str, child: &str, port: &'static str) -> Self {
        self.def.exposed_inputs.push((ext, child.to_owned(), port));
        self
    }

    /// Expose an internal child port as an external output on this subgraph.
    pub fn expose_output(mut self, ext: &'static str, child: &str, port: &'static str) -> Self {
        self.def.exposed_outputs.push((ext, child.to_owned(), port));
        self
    }

    /// Consume the builder and produce a validated, compiled `FlatGraph`.
    ///
    /// If `expose_input` / `expose_output` were called the resulting
    /// `FlatGraph` also implements `Node` and can be passed to `add_node` in a
    /// parent `NetworkBuilder`.
    pub fn build(self) -> Result<FlatGraph, BuildError> {
        flatten_and_build(self.def)
    }

    /// Consume the builder and return the `SubgraphDef` for embedding.
    pub fn into_subgraph(self) -> SubgraphDef {
        self.def
    }
}

/// Returned by `NetworkBuilder::wire`.  Lets the caller optionally mark the
/// most-recently-added wire as recurrent before continuing to chain calls.
pub struct WireBuilder {
    inner: NetworkBuilder,
    wire_idx: usize,
}

impl WireBuilder {
    /// Mark the wire added by the preceding `.wire(...)` call as recurrent
    /// (carries values from the previous tick, breaking topological cycles).
    pub fn recurrent(mut self) -> NetworkBuilder {
        self.inner.def.wires[self.wire_idx].recurrent = true;
        self.inner
    }

    // ── Delegate the rest of NetworkBuilder's API ──────────────────────────

    pub fn add_node(self, name: &str, node: impl Node + Send + 'static) -> NetworkBuilder {
        self.inner.add_node(name, node)
    }

    pub fn add_subgraph(self, name: &str, sub: SubgraphDef) -> NetworkBuilder {
        self.inner.add_subgraph(name, sub)
    }

    pub fn wire(
        self,
        src: &str,
        src_port: &'static str,
        dst: &str,
        dst_port: &'static str,
    ) -> WireBuilder {
        self.inner.wire(src, src_port, dst, dst_port)
    }

    pub fn expose_input(self, ext: &'static str, child: &str, port: &'static str) -> NetworkBuilder {
        self.inner.expose_input(ext, child, port)
    }

    pub fn expose_output(self, ext: &'static str, child: &str, port: &'static str) -> NetworkBuilder {
        self.inner.expose_output(ext, child, port)
    }

    pub fn build(self) -> Result<FlatGraph, BuildError> {
        self.inner.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::node::Node;
    use super::super::port::{Aggregation, PortSpec, PortValues};

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
            // Sum "in" and "mod" if present, otherwise copy "in".
            let v: Vec<f32> = if let Some(m) = inp.get("mod") {
                inp.get("in").unwrap().iter().zip(m.iter()).map(|(a, b)| a + b).collect()
            } else {
                inp.get("in").unwrap().to_vec()
            };
            out.get_mut("out").unwrap().copy_from_slice(&v);
        }
        fn learn(&mut self, _: &PortValues) {}
    }

    #[test]
    fn builder_two_nodes() {
        let fg = NetworkBuilder::new("test")
            .add_node("a", Echo::new(3))
            .add_node("b", Echo::new(3))
            .wire("a", "out", "b", "in")
            .build()
            .unwrap();
        assert_eq!(fg.node_count(), 2);
    }

    #[test]
    fn recurrent_wire_marked() {
        let fg = NetworkBuilder::new("test")
            .add_node("a", Echo::new(2))
            .add_node("b", Echo::new(2))
            .wire("a", "out", "b", "in")
            .wire("b", "out", "a", "in")
            .recurrent()
            .build()
            .unwrap();
        assert_eq!(fg.recurrent_wires.len(), 1);
        assert_eq!(fg.feedforward_wires.len(), 1);
    }

    #[test]
    fn build_errors_on_unresolved_cycle() {
        let result = NetworkBuilder::new("test")
            .add_node("a", Echo::new(2))
            .add_node("b", Echo::new(2))
            .wire("a", "out", "b", "in")
            .wire("b", "out", "a", "in")
            .build();
        assert!(result.is_err());
    }

    // ── Template / composable-node tests ──────────────────────────────────

    /// A compiled FlatGraph used as a Node in a parent network: tick propagates
    /// through the template boundary.
    #[test]
    fn flat_graph_as_node_tick_propagates() {
        let template = NetworkBuilder::new("tpl")
            .add_node("e", Echo::new(4))
            .expose_input("in", "e", "in")
            .expose_output("out", "e", "out")
            .build()
            .unwrap();

        // Use the template as a node in a parent network (another Echo wired behind it).
        let mut parent = NetworkBuilder::new("parent")
            .add_node("t", template)
            .add_node("b", Echo::new(4))
            .wire("t", "out", "b", "in")
            .build()
            .unwrap();

        let mut ext = PortValues::zeros_from(&[PortSpec { name: "in", dim: 4, agg: Aggregation::Concat }]);
        ext.get_mut("in").unwrap().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        parent.tick(&ext);

        // The last node in exec order should have received [1,2,3,4].
        let last = *parent.exec_order.last().unwrap();
        assert_eq!(parent.output_bufs[last].get("out").unwrap(), &[1.0f32, 2.0, 3.0, 4.0]);
    }

    /// Two levels of template nesting.
    #[test]
    fn nested_templates_two_deep() {
        // Inner template: single Echo node.
        let inner = NetworkBuilder::new("inner")
            .add_node("e", Echo::new(2))
            .expose_input("in", "e", "in")
            .expose_output("out", "e", "out")
            .build()
            .unwrap();

        // Outer template wraps the inner one.
        let outer = NetworkBuilder::new("outer")
            .add_node("i", inner)
            .expose_input("in", "i", "in")
            .expose_output("out", "i", "out")
            .build()
            .unwrap();

        // Root network uses the outer template.
        let mut root = NetworkBuilder::new("root")
            .add_node("o", outer)
            .add_node("sink", Echo::new(2))
            .wire("o", "out", "sink", "in")
            .build()
            .unwrap();

        let mut ext = PortValues::zeros_from(&[PortSpec { name: "in", dim: 2, agg: Aggregation::Concat }]);
        ext.get_mut("in").unwrap().copy_from_slice(&[5.0, 6.0]);
        root.tick(&ext);

        let last = *root.exec_order.last().unwrap();
        assert_eq!(root.output_bufs[last].get("out").unwrap(), &[5.0f32, 6.0]);
    }

    /// Lateral Sum wire: two templates both feed into a third node's Sum port.
    #[test]
    fn lateral_sum_wire_between_templates() {
        let make_template = || {
            NetworkBuilder::new("lat")
                .add_node("e", Echo::new(3))
                .expose_input("in", "e", "in")
                .expose_output("out", "e", "out")
                .build()
                .unwrap()
        };

        // "sink" has dim-3 Sum input named "mod".
        struct Sink { ins: Vec<PortSpec>, outs: Vec<PortSpec> }
        impl Node for Sink {
            fn input_ports(&self)  -> &[PortSpec] { &self.ins  }
            fn output_ports(&self) -> &[PortSpec] { &self.outs }
            fn tick(&mut self, inp: &PortValues, out: &mut PortValues) {
                let v = inp.get("mod").unwrap().to_vec();
                out.get_mut("out").unwrap().copy_from_slice(&v);
            }
            fn learn(&mut self, _: &PortValues) {}
        }
        let sink = Sink {
            ins:  vec![PortSpec { name: "mod", dim: 3, agg: Aggregation::Sum }],
            outs: vec![PortSpec { name: "out", dim: 3, agg: Aggregation::Concat }],
        };

        let mut net = NetworkBuilder::new("net")
            .add_node("a", make_template())
            .add_node("b", make_template())
            .add_node("sink", sink)
            .wire("a", "out", "sink", "mod")
            .wire("b", "out", "sink", "mod")
            .build()
            .unwrap();

        // Provide external input to both templates (they are sensory nodes).
        let mut ext = PortValues::zeros_from(&[PortSpec { name: "in", dim: 3, agg: Aggregation::Concat }]);
        ext.get_mut("in").unwrap().copy_from_slice(&[1.0, 0.0, 0.0]);
        net.tick(&ext);

        // Sink should have summed contributions from a and b: [1,0,0] + [1,0,0] = [2,0,0].
        let sink_idx = net.exec_order.last().copied().unwrap();
        assert_eq!(net.output_bufs[sink_idx].get("out").unwrap(), &[2.0f32, 0.0, 0.0]);
    }

    /// Feedback recurrent wire from a higher template back to a lower one.
    #[test]
    fn feedback_recurrent_wire_between_templates() {
        let lower = NetworkBuilder::new("lower")
            .add_node("e", Echo::new(2))
            .expose_input("in", "e", "in")
            .expose_output("out", "e", "out")
            .build()
            .unwrap();

        let upper = NetworkBuilder::new("upper")
            .add_node("e", Echo::new(2))
            .expose_input("in", "e", "in")
            .expose_output("out", "e", "out")
            .build()
            .unwrap();

        // lower → upper (ff), upper → lower (recurrent).
        let net = NetworkBuilder::new("net")
            .add_node("lower", lower)
            .add_node("upper", upper)
            .wire("lower", "out", "upper", "in")
            .wire("upper", "out", "lower", "in")
            .recurrent()
            .build()
            .unwrap();

        assert_eq!(net.feedforward_wires.len(), 1);
        assert_eq!(net.recurrent_wires.len(), 1);
    }
}
