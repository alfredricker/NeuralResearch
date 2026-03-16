pub mod grid;
pub mod cortical;

pub use grid::{GridModule, GridBank};
pub use cortical::CorticalRegion;

use std::ops::Range;
use crate::graph::CsrGraph;
use crate::node::NodeArray;

/// A neural region with typed neuron subpopulations.
///
/// Index layout within `nodes`:
///   0..m_neurons.end       → M (model neurons)
///   m_neurons.end..w_neurons.end → W_M (learned where neurons)
///   (W_T grid modules are stored separately in CorticalRegion)
///   f_omega → feed-in range (subset of M or separate slice)
///   f_z     → feed-out range
pub struct Region {
    pub nodes: NodeArray,
    pub internal_graph: CsrGraph,
    pub f_omega: Range<usize>,  // feed-in neuron indices
    pub f_z: Range<usize>,      // feed-out neuron indices
    pub m_neurons: Range<usize>,
    pub w_neurons: Range<usize>,
}

impl Region {
    pub fn n_model(&self) -> usize {
        self.m_neurons.end - self.m_neurons.start
    }

    pub fn n_where(&self) -> usize {
        self.w_neurons.end - self.w_neurons.start
    }

    pub fn feed_in(&mut self, input: &[f32]) {
        let start = self.f_omega.start;
        for (i, &v) in input.iter().enumerate() {
            if start + i < self.f_omega.end {
                self.nodes.activations[start + i] = v;
            }
        }
    }

    pub fn feed_out(&self) -> &[f32] {
        &self.nodes.activations[self.f_z.clone()]
    }
}
