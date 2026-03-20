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

