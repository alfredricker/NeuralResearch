pub mod leaky_integrated;
pub mod spiking_lif;
pub mod oscillator;
pub mod threshold;

use crate::activation::Activation;

/// A value clamped to [min, max] at construction time.
#[derive(Debug, Clone, Copy)]
pub struct Bounded {
    value: f32,
    pub min: f32,
    pub max: f32,
}

impl Bounded {
    pub fn new(value: f32, min: f32, max: f32) -> Self {
        assert!(min < max, "Bounded: min ({min}) must be less than max ({max})");
        Self { value: value.clamp(min, max), min, max }
    }

    pub fn value(&self) -> f32 { self.value }
}

/// The complete runtime state of a single neuron.
/// Running values live here; update rule parameters live in UpdateRule.
#[derive(Debug, Clone)]
pub enum State {
    /// Unbounded real value — raw synaptic drive or intermediate signal.
    Continuous(f32),
    /// Real value enforced within [min, max] — standard bounded activation.
    ContinuousBounded(Bounded),
    /// Leaky integrate-and-fire membrane potential with refractory countdown.
    /// `ref_remaining > 0` means the neuron is refractory (just fired).
    Spiking { v: f32, ref_remaining: u32 },
    /// Cyclic phase φ ∈ [0, 2π) for oscillator / grid-cell neurons.
    Phase(f32),
    /// Discrete index — spike event (0/1) or class label.
    Discrete(usize),
}

impl State {
    /// Convenience: extract a scalar output signal for downstream wiring.
    /// Spiking returns 1.0 on the tick ref_remaining became t_ref (just fired),
    /// otherwise 0.0. Phase returns cos(φ) as a bounded oscillatory signal.
    pub fn readout(&self) -> f32 {
        match self {
            State::Continuous(v)           => *v,
            State::ContinuousBounded(b)    => b.value(),
            State::Spiking { ref_remaining: r, .. } if *r > 0 => 1.0,
            State::Spiking { .. }          => 0.0,
            State::Phase(phi)              => phi.cos(),
            State::Discrete(d)             => *d as f32,
        }
    }
}

/// Describes how a neuron's state evolves each tick.
/// Parameters (constants) live here; running values live in State.
pub enum UpdateRule {
    LeakyIntegrated { lambda: f32, activation: Activation },
    SpikingLIF      { lambda: f32, v_thresh: f32, v_reset: f32, t_ref: u32 },
    Oscillator      { omega: f32 },
    Threshold       { theta: f32 },
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
            UpdateRule::Threshold { .. } =>
                State::ContinuousBounded(Bounded::new(0.0, 0.0, 1.0)),
            UpdateRule::Custom(_) =>
                State::Continuous(0.0),
        }
    }
}
