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

    /// Add a concrete node.
    pub fn add_node(mut self, name: &str, node: impl Node + Send + 'static) -> Self {
        self.def.children.insert(name.to_owned(), NodeOrSubgraph::Node(Box::new(node)));
        self
    }

    /// Embed a nested subgraph.
    pub fn add_subgraph(mut self, name: &str, sub: SubgraphDef) -> Self {
        self.def.children.insert(name.to_owned(), NodeOrSubgraph::Subgraph(sub));
        self
    }

    /// Connect `src_node::src_port` → `dst_node::dst_port`.
    ///
    /// Returns a `WireBuilder` so you can optionally call `.recurrent()`.
    /// Most of `NetworkBuilder`'s methods are also available on `WireBuilder`
    /// so you can keep chaining without calling `.recurrent()`.
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
    pub fn expose_input(mut self, ext: &str, child: &str, port: &'static str) -> Self {
        self.def.exposed_inputs.push((ext.to_owned(), child.to_owned(), port));
        self
    }

    /// Expose an internal child port as an external output on this subgraph.
    pub fn expose_output(mut self, ext: &str, child: &str, port: &'static str) -> Self {
        self.def.exposed_outputs.push((ext.to_owned(), child.to_owned(), port));
        self
    }

    /// Consume the builder and produce a validated, compiled `FlatGraph`.
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

    pub fn expose_input(self, ext: &str, child: &str, port: &'static str) -> NetworkBuilder {
        self.inner.expose_input(ext, child, port)
    }

    pub fn expose_output(self, ext: &str, child: &str, port: &'static str) -> NetworkBuilder {
        self.inner.expose_output(ext, child, port)
    }

    pub fn build(self) -> Result<FlatGraph, BuildError> {
        self.inner.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            let v = inp.get("in").unwrap().to_vec();
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
        // a → b (ff) and b → a (recurrent)
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
            // No .recurrent() — should fail.
            .build();
        assert!(result.is_err());
    }
}
