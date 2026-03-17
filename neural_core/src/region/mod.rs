pub mod grid;
pub mod cortical;

pub use grid::{GridModule, GridBank};
pub use cortical::CorticalRegion;

use std::ops::Range;
use crate::graph::CsrGraph;
use crate::node::NodeArray;

// ─── RegionModule trait ───────────────────────────────────────────────────────

/// Core abstraction for a composable neural compute unit.
///
/// A `RegionModule` is a self-contained region that:
/// - Accepts a **fixed-size input** activation slice (size = [`n_in`])
/// - Maintains internal state (activations, synaptic weights, etc.)
/// - Produces a **fixed-size output** activation slice (size = [`n_out`])
/// - Supports **online learning** via [`learn`]
///
/// Regions are assembled into processing hierarchies using
/// [`HierarchyBuilder`][crate::hierarchy::HierarchyBuilder].
///
/// # Contract
/// - `input.len()` passed to [`tick`] and [`learn`] must equal [`n_in`].
/// - [`output`] returns activations from the most recent [`tick`]; it is
///   unspecified before the first tick.
///
/// # Example
/// ```rust,ignore
/// use neural_core::region::{CorticalRegion, RegionModule};
///
/// let mut region = CorticalRegion::new(64, 32, &[(5,1),(7,2)], 0.05, &mut rng);
/// region.step(&input);             // advance one time step (inherent method)
/// let out: &[f32] = region.output(); // read output activations
/// region.learn_ff(&input);          // Hebbian update (feedforward weights)
/// region.learn_rr();                // Hebbian update (recurrent weights)
/// ```
///
/// Or via the trait when composing into a [`Hierarchy`][crate::hierarchy::Hierarchy]:
/// ```rust,ignore
/// let r: Box<dyn RegionModule + Send> = Box::new(region);
/// r.tick(&input);
/// ```
///
/// [`n_in`]: RegionModule::n_in
/// [`n_out`]: RegionModule::n_out
/// [`tick`]: RegionModule::tick
/// [`output`]: RegionModule::output
/// [`learn`]: RegionModule::learn
pub trait RegionModule: Send {
    /// Number of input activations expected on each call to [`tick`][Self::tick].
    fn n_in(&self) -> usize;

    /// Number of output activations produced after each call to [`tick`][Self::tick].
    fn n_out(&self) -> usize;

    /// Advance one time step, updating internal state from `input`.
    fn tick(&mut self, input: &[f32]);

    /// Current output activations — valid after the first [`tick`][Self::tick].
    fn output(&self) -> &[f32];

    /// Apply learning updates using the same `input` passed to the last [`tick`][Self::tick].
    fn learn(&mut self, input: &[f32]);
}

// ─── Region ──────────────────────────────────────────────────────────────────

/// Low-level region backed by a [`NodeArray`] and a [`CsrGraph`].
///
/// Subpopulation index layout within `nodes`:
/// ```text
///   0 .. m_neurons.end          → M  (model neurons)
///   m_neurons.end .. w_neurons.end  → W_M (learned where neurons)
///   f_omega                     → feed-in port (subset of the above)
///   f_z                         → feed-out port
/// ```
/// W_T grid modules are stored separately (see [`CorticalRegion`]).
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

    /// Write `input` activations into the feed-in port (F_ω).
    pub fn feed_in(&mut self, input: &[f32]) {
        let start = self.f_omega.start;
        for (i, &v) in input.iter().enumerate() {
            if start + i < self.f_omega.end {
                self.nodes.activations[start + i] = v;
            }
        }
    }

    /// Read activations from the feed-out port (F_z).
    pub fn feed_out(&self) -> &[f32] {
        &self.nodes.activations[self.f_z.clone()]
    }
}
