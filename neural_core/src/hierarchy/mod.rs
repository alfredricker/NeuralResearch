//! Hierarchical composition of [`RegionModule`]s.
//!
//! A [`Hierarchy`] organises regions into ordered **levels**. Data flows
//! upward: level 0 receives raw external input; each subsequent level receives
//! the **concatenated output** of the level below. Within a single level, all
//! regions run **in parallel** (via rayon), independently processing the same
//! input signal.
//!
//! # Building a hierarchy
//!
//! Use the fluent [`HierarchyBuilder`], adding levels bottom-up:
//!
//! ```rust,ignore
//! use neural_core::{
//!     hierarchy::HierarchyBuilder,
//!     region::{CorticalRegion, RegionModule},
//! };
//!
//! let mut rng = rand::thread_rng();
//!
//! let hierarchy = HierarchyBuilder::new()
//!     // Level 0: two parallel regions both seeing the raw input
//!     .level(vec![
//!         Box::new(CorticalRegion::new(64, 784, &[(5,1),(7,2),(11,3)], 0.05, &mut rng)),
//!         Box::new(CorticalRegion::new(64, 784, &[(5,1),(7,2),(11,3)], 0.05, &mut rng)),
//!     ])
//!     // Level 1: one region integrating the 128-d concatenated output of level 0
//!     .level(vec![
//!         Box::new(CorticalRegion::new(32, 128, &[(13,2),(17,3)], 0.05, &mut rng)),
//!     ])
//!     .build();
//! ```
//!
//! # Topology rules
//!
//! * All regions at the **same level** receive the **identical** input vector
//!   (the full concatenated output of the level below, or the raw input for
//!   level 0). Each region's [`n_in`][crate::region::RegionModule::n_in] must
//!   equal that size.
//! * A level's **output** is the concatenation of its regions' outputs in
//!   insertion order.
//! * Regions at the same level are **independent**: no lateral synaptic state
//!   is shared during a tick (intra-level lateral inhibition lives inside each
//!   `CorticalRegion` itself).

use rayon::prelude::*;
use crate::region::RegionModule;

// ─── Level ────────────────────────────────────────────────────────────────────

/// One horizontal slice of a hierarchy: a set of parallel [`RegionModule`]s
/// that all process the same input and whose outputs are concatenated.
pub struct Level {
    /// Regions within this level, processed in parallel during tick/learn.
    pub regions: Vec<Box<dyn RegionModule + Send>>,
}

impl Level {
    /// Total output dimensionality: sum of all regions' `n_out`.
    pub fn n_out(&self) -> usize {
        self.regions.iter().map(|r| r.n_out()).sum()
    }

    /// Tick all regions in parallel, each receiving the full `input` slice.
    ///
    /// Safe to parallelize because regions at the same level share no mutable
    /// state during tick.
    pub fn tick(&mut self, input: &[f32]) {
        self.regions.par_iter_mut().for_each(|r| r.tick(input));
    }

    /// Collect the concatenated output activations of all regions, in order.
    pub fn collect_output(&self) -> Vec<f32> {
        self.regions.iter().flat_map(|r| r.output()).copied().collect()
    }

    /// Apply learning in parallel across all regions.
    pub fn learn(&mut self, input: &[f32]) {
        self.regions.par_iter_mut().for_each(|r| r.learn(input));
    }
}

// ─── Hierarchy ────────────────────────────────────────────────────────────────

/// A directed hierarchy of [`Level`]s with cached inter-level activations.
///
/// Call [`forward`][Hierarchy::forward] to propagate a signal upward, then
/// [`learn`][Hierarchy::learn] to apply Hebbian updates at every level.
pub struct Hierarchy {
    /// Ordered levels, index 0 = bottom (closest to raw input).
    pub levels: Vec<Level>,
    /// Cached per-level output vectors; populated by the most recent `forward`.
    outputs: Vec<Vec<f32>>,
}

impl Hierarchy {
    /// Propagate `input` upward through all levels and return the top-level
    /// output slice.
    ///
    /// The returned slice is borrowed from internal cache and remains valid
    /// until the next call to `forward`.
    pub fn forward(&mut self, input: &[f32]) -> &[f32] {
        // Level 0 always receives the raw external input.
        self.levels[0].tick(input);
        self.outputs[0] = self.levels[0].collect_output();

        // Each subsequent level receives the cached output of the level below.
        // We clone the feed to release the immutable borrow on `self.outputs`
        // before taking the mutable borrow needed by `self.levels[l].tick`.
        for l in 1..self.levels.len() {
            let feed = self.outputs[l - 1].clone();
            self.levels[l].tick(&feed);
            self.outputs[l] = self.levels[l].collect_output();
        }

        &self.outputs[self.levels.len() - 1]
    }

    /// Apply Hebbian learning at every level using activations from the most
    /// recent [`forward`][Hierarchy::forward] call.
    ///
    /// Must be called **after** `forward`.
    pub fn learn(&mut self, external_input: &[f32]) {
        self.levels[0].learn(external_input);
        for l in 1..self.levels.len() {
            let feed = self.outputs[l - 1].clone();
            self.levels[l].learn(&feed);
        }
    }

    /// Output of level `l` from the most recent `forward` pass.
    pub fn level_output(&self, l: usize) -> &[f32] {
        &self.outputs[l]
    }

    /// Number of levels in the hierarchy.
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// Output dimensionality of the top level.
    pub fn n_out(&self) -> usize {
        self.outputs.last().map(Vec::len).unwrap_or(0)
    }
}

// ─── HierarchyBuilder ─────────────────────────────────────────────────────────

/// Fluent builder for assembling a [`Hierarchy`] level by level.
///
/// Add levels **bottom-up** (level 0 first). Each [`level`][Self::level] call
/// appends one [`Level`] of parallel regions.
///
/// # Example
/// ```rust,ignore
/// let h = HierarchyBuilder::new()
///     .level(vec![Box::new(region_l0)])
///     .level(vec![Box::new(region_l1)])
///     .build();
/// ```
pub struct HierarchyBuilder {
    levels: Vec<Level>,
}

impl HierarchyBuilder {
    pub fn new() -> Self {
        Self { levels: Vec::new() }
    }

    /// Append a level containing `regions`.
    ///
    /// All regions in the vec tick in parallel during each forward pass and
    /// each must accept the same input size (the output width of the level
    /// below, or the raw input width for level 0).
    pub fn level(mut self, regions: Vec<Box<dyn RegionModule + Send>>) -> Self {
        self.levels.push(Level { regions });
        self
    }

    /// Consume the builder and produce a [`Hierarchy`].
    pub fn build(self) -> Hierarchy {
        let outputs = self.levels.iter().map(|l| vec![0.0f32; l.n_out()]).collect();
        Hierarchy { levels: self.levels, outputs }
    }
}

impl Default for HierarchyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::CorticalRegion;

    /// Minimal stub region for hierarchy unit tests.
    struct PassThrough {
        n: usize,
        out: Vec<f32>,
    }

    impl PassThrough {
        fn new(n: usize) -> Self {
            Self { n, out: vec![0.0; n] }
        }
    }

    unsafe impl Send for PassThrough {}

    impl RegionModule for PassThrough {
        fn n_in(&self)  -> usize { self.n }
        fn n_out(&self) -> usize { self.n }
        fn tick(&mut self, input: &[f32]) { self.out.copy_from_slice(input); }
        fn output(&self) -> &[f32] { &self.out }
        fn learn(&mut self, _input: &[f32]) {}
    }

    #[test]
    fn two_level_forward_shape() {
        let mut h = HierarchyBuilder::new()
            .level(vec![Box::new(PassThrough::new(4))])
            .level(vec![Box::new(PassThrough::new(4))])
            .build();
        let input = vec![1.0f32; 4];
        let out = h.forward(&input);
        assert_eq!(out.len(), 4);
        // PassThrough just copies, so top output == input
        assert_eq!(out, input.as_slice());
    }

    #[test]
    fn level_output_concat() {
        // Two parallel regions at level 0; their outputs should be concatenated.
        let mut h = HierarchyBuilder::new()
            .level(vec![
                Box::new(PassThrough::new(3)),
                Box::new(PassThrough::new(3)),
            ])
            .build();
        let input = vec![0.1, 0.2, 0.3f32];
        let out = h.forward(&input);
        assert_eq!(out.len(), 6); // 3 + 3 concatenated
    }

    #[test]
    fn cortical_hierarchy_runs() {
        let mut rng = rand::thread_rng();
        let r0 = CorticalRegion::new(16, 8, &[(5, 1), (7, 2)], 0.1, &mut rng);
        let r1 = CorticalRegion::new(8, 16, &[(5, 1)], 0.1, &mut rng);
        let mut h = HierarchyBuilder::new()
            .level(vec![Box::new(r0) as Box<dyn RegionModule + Send>])
            .level(vec![Box::new(r1) as Box<dyn RegionModule + Send>])
            .build();
        let input = vec![0.5f32; 8];
        let out = h.forward(&input);
        assert_eq!(out.len(), 8);
        h.learn(&input);
    }
}
