pub mod leaky_integrated;
pub mod spiking_lif;
pub mod oscillator;
pub mod threshold;

pub use leaky_integrated::LeakyIntegrated;
pub use spiking_lif::SpikingLIF;
pub use oscillator::Oscillator;
pub use threshold::Threshold;

/// A value clamped to [min, max] at construction time.
/// The `value` field is private — the bound is a structural invariant,
/// not a runtime check that callers can forget.
#[derive(Debug, Clone, Copy)]
pub struct Bounded {
    value: f32,
    pub min: f32,
    pub max: f32,
}

impl Bounded {
    /// Clamps `value` into [min, max]. Panics if min >= max.
    pub fn new(value: f32, min: f32, max: f32) -> Self {
        assert!(min < max, "Bounded: min ({min}) must be less than max ({max})");
        Self { value: value.clamp(min, max), min, max }
    }

    pub fn value(&self) -> f32 {
        self.value
    }
}

#[derive(Debug, Clone)]
pub enum State {
    Continuous(f32),
    Discrete(usize),
    ContinuousBounded(Bounded),
}


pub enum UpdateRule {
    LeakyIntegrated,
    SpikingLIF,
    Oscillator,
    Threshold,
    Custom(Box<dyn Fn(&State) -> State>),
}

impl UpdateRule {
    pub fn update(&self, state: &State) -> State {
        match self {
            UpdateRule::LeakyIntegrated => leaky_integrated::update(state),
            UpdateRule::SpikingLIF => spiking_lif::update(state),
            UpdateRule::Oscillator => oscillator::update(state),
            UpdateRule::Threshold => threshold::update(state),
            UpdateRule::Custom(f) => f(state),
        }
    }
}