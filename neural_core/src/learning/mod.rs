pub mod hebbian;

pub use hebbian::HebbianRule;

/// Generic synaptic learning rule.
pub trait LearningRule {
    /// Compute updated weight from pre- and post-synaptic activations.
    fn update_weight(&self, w: f32, pre: f32, post: f32, eta: f32) -> f32;
}
