use crate::activation::Activation;
use crate::state::{Bounded, State};
use crate::state::rules::{leaky_integrated, spiking_lif, oscillator, threshold, smooth_gate};

/// Describes how a neuron's state evolves each tick.
/// Parameters (constants) live here; running values live in State.
pub enum UpdateRule {
    LeakyIntegrated { lambda: f32, activation: Activation },
    SpikingLIF      { lambda: f32, v_thresh: f32, v_reset: f32, t_ref: u32 },
    Oscillator      { omega: f32 },
    Threshold       { theta: f32 },
    /// Smooth gate: g(x) = x² / (x² + θ²). Symmetric, no dead zone, output ∈ [0, 1).
    SmoothGate      { theta: f32 },
    Custom(Box<dyn Fn(&State, f32) -> State + Send>),
}

impl UpdateRule {
    /// Pure state transition: (current_state, scalar_input) → next_state.
    pub fn update(&self, state: &State, input: f32) -> State {
        match self {
            UpdateRule::LeakyIntegrated { lambda, activation } =>
                leaky_integrated::update(state, input, *lambda, activation),
            UpdateRule::SpikingLIF { lambda, v_thresh, v_reset, t_ref } =>
                spiking_lif::update(state, input, *lambda, *v_thresh, *v_reset, *t_ref),
            UpdateRule::Oscillator { omega } =>
                oscillator::update(state, input, *omega),
            UpdateRule::Threshold { theta } =>
                threshold::update(state, input, *theta),
            UpdateRule::SmoothGate { theta } =>
                smooth_gate::update(state, input, *theta),
            UpdateRule::Custom(f) =>
                f(state, input),
        }
    }

    /// The natural initial state for this rule.
    pub fn initial_state(&self) -> State {
        match self {
            UpdateRule::LeakyIntegrated { .. } =>
                State::ContinuousBounded(Bounded::new(0.0, -1.0, 1.0)),
            UpdateRule::SpikingLIF { v_reset, .. } =>
                State::Spiking { v: *v_reset, ref_remaining: 0 },
            UpdateRule::Oscillator { .. } =>
                State::Phase(0.0),
            UpdateRule::Threshold { .. } |
            UpdateRule::SmoothGate { .. } =>
                State::ContinuousBounded(Bounded::new(0.0, 0.0, 1.0)),
            UpdateRule::Custom(_) =>
                State::Continuous(0.0),
        }
    }
}