use super::{Bounded, State};

/// Leaky Integrate-and-Fire neuron.
///
/// Membrane potential leaky-integrates input current until it crosses a
/// threshold, at which point the neuron emits a spike (Discrete(1)) and
/// the potential is reset to `v_reset`. During the refractory period the
/// neuron is silent regardless of input.
///
/// Dynamics:
///   if refractory:  v stays at v_reset, output = Discrete(0)
///   else:           v(t+1) = (1 − λ)·v(t) + f
///                   if v(t+1) ≥ v_thresh: spike, v ← v_reset, refractory ← T_ref
///
/// Output:
///   Discrete(1) on the tick a spike occurs, Discrete(0) otherwise.
///   Membrane potential is also readable via `self.v`.
pub struct SpikingLIF {
    /// Membrane potential (not bounded — can exceed threshold transiently).
    pub v: f32,
    /// Leak rate λ ∈ [0, 1].
    pub lambda: f32,
    /// Firing threshold.
    pub v_thresh: f32,
    /// Reset potential after a spike.
    pub v_reset: f32,
    /// Refractory period in ticks.
    pub t_ref: u32,
    /// Ticks remaining in the current refractory period.
    pub ref_remaining: u32,
}

impl SpikingLIF {
    pub fn new(lambda: f32, v_thresh: f32, v_reset: f32, t_ref: u32) -> Self {
        Self { v: 0.0, lambda, v_thresh, v_reset, t_ref, ref_remaining: 0 }
    }
    
    /// `input` is the injected current — State::Continuous.
    fn update(&mut self, input: &State) -> State {
        let f = match input {
            State::Continuous(f) => *f,
            other => panic!("SpikingLIF expects Continuous input, got {:?}", other),
        };

        if self.ref_remaining > 0 {
            self.ref_remaining -= 1;
            self.v = self.v_reset;
            return State::Discrete(0);
        }

        self.v = (1.0 - self.lambda) * self.v + f;

        if self.v >= self.v_thresh {
            self.v = self.v_reset;
            self.ref_remaining = self.t_ref;
            State::Discrete(1)
        } else {
            State::Discrete(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fires_when_threshold_crossed() {
        let mut n = SpikingLIF::new(0.0, 1.0, 0.0, 0);
        // With λ=0 and drive=1.0, v hits threshold on first tick.
        let s = n.update(&State::Continuous(1.0));
        assert_eq!(s, State::Discrete(1));
    }

    #[test]
    fn silent_during_refractory() {
        let mut n = SpikingLIF::new(0.0, 1.0, 0.0, 3);
        n.update(&State::Continuous(1.0)); // spike
        for _ in 0..3 {
            let s = n.update(&State::Continuous(10.0)); // strong drive, but refractory
            assert_eq!(s, State::Discrete(0));
        }
    }

    #[test]
    fn no_spike_below_threshold() {
        let mut n = SpikingLIF::new(0.1, 1.0, 0.0, 0);
        let s = n.update(&State::Continuous(0.1));
        assert_eq!(s, State::Discrete(0));
    }
}
