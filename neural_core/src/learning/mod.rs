pub mod hebbian;
pub mod oja;
pub mod stdp;
pub mod bcm;

pub use hebbian::HebbianRule;
pub use oja::OjaRule;
pub use stdp::StdpRule;
pub use bcm::BcmRule;

use crate::state::State;

/// Scalar synaptic learning interface.
///
/// Implemented by rules that only need the readout values (f32) of the pre-
/// and post-synaptic neurons. STDP is excluded — it requires full `State`
/// access to inspect spike timing.
///
/// `&mut self` is required to support stateful rules such as BCM, which
/// maintains a per-neuron sliding modification threshold θ_M that is updated
/// on every weight update call.
pub trait Learn {
    /// Return the updated weight given pre/post scalar activations and learning rate η.
    ///
    /// `neuron_idx` is the index of the post-synaptic neuron within its layer.
    /// Stateless rules (Hebbian, Oja) ignore it; BCM uses it to index into its
    /// per-neuron sliding threshold vector.
    fn update_weight(&mut self, w: f32, pre: f32, post: f32, eta: f32, neuron_idx: usize) -> f32;
}

/// Selects the synaptic learning algorithm applied each tick.
///
/// `update_weight` takes `&State` for pre and post so that STDP can inspect
/// spike timing. All non-STDP variants call `state.readout()` internally and
/// are state-agnostic.
pub enum LearnRule {
    /// Hebbian: Δw = η·σ(pre)·σ(post), with weight decay μ.
    Hebbian(HebbianRule),

    /// Oja's rule: Δw = η·σ(post)·(σ(pre) − σ(post)·w).
    /// Converges toward the first principal component; no weight explosion.
    Oja(OjaRule),

    /// BCM rule: Δw = η·pre·post·(post − θ_M), with sliding threshold θ_M per neuron.
    /// θ_M chases the mean squared output, creating selectivity without global coordination.
    Bcm(BcmRule),

    /// Spike-timing dependent plasticity.
    /// **Requires `State::Spiking` for both pre and post — panics otherwise.**
    Stdp(StdpRule),

    /// Custom: (w, pre_state, post_state, η) → new_w.
    Custom(Box<dyn Fn(f32, &State, &State, f32) -> f32 + Send>),
}

impl LearnRule {
    /// Update a single weight given pre- and post-synaptic states and learning rate η.
    ///
    /// For `Stdp`, both states must be `State::Spiking`.
    /// All other variants call `state.readout()` and accept any state type.
    pub fn update_weight(&mut self, w: f32, pre: &State, post: &State, eta: f32, neuron_idx: usize) -> f32 {
        match self {
            LearnRule::Hebbian(rule) => rule.update_weight(w, pre.readout(), post.readout(), eta, neuron_idx),
            LearnRule::Oja(rule)     => rule.update_weight(w, pre.readout(), post.readout(), eta, neuron_idx),
            LearnRule::Bcm(rule)     => rule.update_weight(w, pre.readout(), post.readout(), eta, neuron_idx),
            LearnRule::Stdp(rule)    => rule.update_weight(w, pre, post, eta),
            LearnRule::Custom(f)     => f(w, pre, post, eta),
        }
    }
}
