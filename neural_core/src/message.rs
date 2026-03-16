use crate::graph::CsrGraph;
use crate::node::NodeArray;

/// Geometric deep learning-style message passing, scalar-first.
///
/// Each step:
///   1. For each active node i, compute messages from its neighbors
///   2. Aggregate messages into a single value
///   3. Update node i's activation
pub trait MessagePassing {
    fn message(&self, src_activation: f32, weight: f32) -> f32;
    fn aggregate(&self, messages: impl Iterator<Item = f32>) -> f32;
    fn update(&self, current: f32, aggregated: f32) -> f32;

    /// Run one full pass: accumulate into `nodes.input_buffer`, then apply update.
    /// Only iterates over nodes in the active set for efficiency.
    fn pass(&self, graph: &CsrGraph, nodes: &mut NodeArray) {
        nodes.clear_inputs();

        // Scatter messages from active sources to their targets
        let active: Vec<u32> = nodes.active_set.clone();
        for src in active {
            let a_src = nodes.activations[src as usize];
            for (dst, w) in graph.neighbors(src as usize) {
                let msg = self.message(a_src, w);
                nodes.input_buffer[dst as usize] += msg;
            }
        }

        // Update each node
        let n = nodes.len();
        for i in 0..n {
            let agg = nodes.input_buffer[i];
            nodes.activations[i] = self.update(nodes.activations[i], agg);
        }
        nodes.rebuild_active_set();
    }
}

/// Cortical MessagePassing implementation (matches Cortical.tex equations).
pub struct CorticalMP {
    pub lambda: f32,  // leak rate λ ∈ [0,1]
}

impl Default for CorticalMP {
    fn default() -> Self {
        Self { lambda: 0.1 }
    }
}

impl MessagePassing for CorticalMP {
    #[inline]
    fn message(&self, src_activation: f32, weight: f32) -> f32 {
        crate::activation::sigma(src_activation) * weight
    }

    #[inline]
    fn aggregate(&self, messages: impl Iterator<Item = f32>) -> f32 {
        messages.sum()
    }

    #[inline]
    fn update(&self, current: f32, aggregated: f32) -> f32 {
        (1.0 - self.lambda) * current + crate::activation::sigma(aggregated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    #[test]
    fn cortical_mp_single_step() {
        let mp = CorticalMP::default();
        // 2 nodes, node 0 → node 1 with weight 1.0
        let graph = GraphBuilder::new(2).add_edge(0, 1, 1.0).build();
        let mut nodes = crate::node::NodeArray::new(2, 0.0);
        nodes.activations[0] = 1.0;
        nodes.rebuild_active_set();

        mp.pass(&graph, &mut nodes);

        // Node 1 should have received σ(1.0) * 1.0 = 0.5
        // update(0.0, 0.5) = 0.9*0 + σ(0.5) = σ(0.5) ≈ 0.333
        assert!(nodes.activations[1] > 0.0);
    }
}
