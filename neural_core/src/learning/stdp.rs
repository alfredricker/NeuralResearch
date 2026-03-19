use crate::state::State;

/// Spike-timing dependent plasticity (STDP).
///
/// Requires both pre- and post-synaptic neurons to be in `State::Spiking`.
/// Panics at runtime if either state is not `State::Spiking`.
///
/// ## Timing approximation
///
/// Biological STDP depends on the millisecond-precise difference Δt = t_post − t_pre.
/// In the discrete-tick framework, `ref_remaining` serves as a proxy for recency:
/// a higher value means the neuron fired more recently on the current tick.
///
/// Given both neurons spiked (ref_remaining > 0):
/// - post fired more recently than pre (post_ref ≥ pre_ref):
///     causal connection → **LTP**  Δw = +η · A+
/// - pre fired more recently than post (pre_ref > post_ref):
///     anti-causal connection → **LTD**  Δw = −η · A−
///
/// If only pre spiked (pre fired but failed to drive post):
///   weak LTD  Δw = −η · A− · 0.5
///
/// Weight decay is applied every tick regardless.
pub struct StdpRule {
    /// LTP amplitude.
    pub a_plus:  f32,
    /// LTD amplitude.
    pub a_minus: f32,
    /// Weight decay rate.
    pub mu:      f32,
}

impl Default for StdpRule {
    fn default() -> Self {
        Self { a_plus: 0.01, a_minus: 0.012, mu: 0.001 }
    }
}

impl StdpRule {
    /// Update a single weight from pre- and post-synaptic `State::Spiking` states.
    ///
    /// # Panics
    /// Panics if either `pre` or `post` is not `State::Spiking`.
    pub fn update_weight(&self, w: f32, pre: &State, post: &State, eta: f32) -> f32 {
        let pre_ref = match pre {
            State::Spiking { ref_remaining, .. } => *ref_remaining,
            other => panic!("StdpRule requires State::Spiking for pre-synaptic neuron, got {:?}", other),
        };
        let post_ref = match post {
            State::Spiking { ref_remaining, .. } => *ref_remaining,
            other => panic!("StdpRule requires State::Spiking for post-synaptic neuron, got {:?}", other),
        };

        let pre_spiked  = pre_ref  > 0;
        let post_spiked = post_ref > 0;

        let delta = match (pre_spiked, post_spiked) {
            (true, true) if post_ref >= pre_ref =>
                // post fired more recently → causal → LTP
                 eta * self.a_plus,
            (true, true) =>
                // pre fired more recently → anti-causal → LTD
                -eta * self.a_minus,
            (true, false) =>
                // pre fired but post silent → weak LTD
                -eta * self.a_minus * 0.5,
            (false, _) =>
                0.0,
        };

        w * (1.0 - self.mu) + delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spiking(ref_remaining: u32) -> State {
        State::Spiking { v: 0.0, ref_remaining }
    }

    #[test]
    fn ltp_when_post_fired_after_pre() {
        let rule = StdpRule::default();
        // post_ref=3 (more recent) vs pre_ref=1 → LTP
        let w = rule.update_weight(0.0, &spiking(1), &spiking(3), 1.0);
        assert!(w > 0.0);
    }

    #[test]
    fn ltd_when_pre_fired_after_post() {
        let rule = StdpRule::default();
        // pre_ref=3 (more recent) vs post_ref=1 → LTD
        let w = rule.update_weight(0.5, &spiking(3), &spiking(1), 1.0);
        assert!(w < 0.5);
    }

    #[test]
    fn ltd_when_only_pre_spiked() {
        let rule = StdpRule::default();
        let w = rule.update_weight(0.5, &spiking(2), &spiking(0), 1.0);
        assert!(w < 0.5);
    }

    #[test]
    fn no_change_when_neither_spiked() {
        let rule = StdpRule { mu: 0.0, ..Default::default() };
        let w = rule.update_weight(0.5, &spiking(0), &spiking(0), 1.0);
        assert!((w - 0.5).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "StdpRule requires State::Spiking")]
    fn panics_on_non_spiking_pre() {
        let rule = StdpRule::default();
        rule.update_weight(0.0, &State::Continuous(1.0), &spiking(1), 1.0);
    }

    #[test]
    #[should_panic(expected = "StdpRule requires State::Spiking")]
    fn panics_on_non_spiking_post() {
        let rule = StdpRule::default();
        rule.update_weight(0.0, &spiking(1), &State::Continuous(1.0), 1.0);
    }
}
