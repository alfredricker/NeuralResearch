use crate::network::event::{Event, SOMATIC_SPIKE, DENDRITIC_SPIKE, FORWARD_AP, EventProducer};
use crate::constants::{H_ALPHA, ALPHA_BOOST};
use crate::neuron::synapse::{update_weight, update_synapse_alpha};
use crate::neuron::dendrite::{update_dendrite_activity, DendriteOutput};
use crate::neuron::soma::update_soma_potential;

// ============================================================================================
// Each handler does ONLY event routing: scope-in the SoA slices, call the neuron/ primitives
// (which own all the physics — decay, integration, thresholds, resets), and translate the
// primitive's verdict into emitted events. No physics lives here.
//
// Event flow:
//   FORWARD_AP / APICAL_FB  ─► handle_synapse_signal  ─► DENDRITIC_SPIKE (basal) | SOMA_SIGNAL (apical)
//   DENDRITIC_SPIKE         ─► handle_dendritic_spike ─► SOMA_SIGNAL
//   SOMA_SIGNAL             ─► handle_soma_signal     ─► SOMATIC_SPIKE × burst
//   SOMATIC_SPIKE           ─► handle_somatic_spike   ─► FORWARD_AP
// ============================================================================================


// A synapse receives an external action potential: feedforward (FORWARD_AP, basal dendrite) or
// top-down (APICAL_FB, apical dendrite). Boost the receiving synapse, integrate its parent
// dendrite, and route the dendrite's verdict onward:
//   basal  → DENDRITIC_SPIKE, if the branch crossed threshold and fired
//   apical → SOMA_SIGNAL, carrying the graded plateau depolarization
// synapse slices must already be scoped to this dendrite via dendrite_synapse_range.
pub fn handle_synapse_signal(
    s_idx: usize,
    dendrite_idx: usize,
    neuron_idx: usize,
    timestamp: u16,
    live_end: usize,
    synapse_xs: &[u8],
    synapse_alphas: &mut [u8],
    synapse_last_events: &mut [u16],
    synapse_weights: &[i8],
    dendrite_activity: &mut u16,
    dendrite_last_event: &mut u16,
    dendrite_threshold: u16, // basal: hard threshold; apical: θ_B half-activation
    is_apical: bool,
    producer: &EventProducer,
) {
    let alpha = update_synapse_alpha(s_idx, timestamp, synapse_alphas, synapse_last_events);
    synapse_alphas[s_idx] = alpha.saturating_add(ALPHA_BOOST);

    match update_dendrite_activity(
        s_idx, timestamp, live_end,
        synapse_xs, synapse_alphas, synapse_weights, synapse_last_events,
        dendrite_activity, dendrite_last_event, dendrite_threshold, is_apical,
    ) {
        DendriteOutput::Basal { fired: true } => {
            producer.push(Event::spike(DENDRITIC_SPIKE, dendrite_idx as u32, timestamp));
        }
        DendriteOutput::Basal { fired: false } => {}
        DendriteOutput::Apical { plateau } => {
            producer.push(Event::soma_signal(neuron_idx as u32, timestamp, plateau));
        }
    }
}


// A basal dendrite fired. Reinforce the synapses that were active at spike time (local NMDA-like
// plasticity) and deliver a branch-constant-scaled depolarization to the soma as a SOMA_SIGNAL.
//   branch_constant > 0: proximal — passes its magnitude to the soma
//   branch_constant <= 0: distal — attenuated to 1 at the soma, strong local alpha reinforcement
// synapse slices must already be scoped to this dendrite via dendrite_synapse_range.
pub fn handle_dendritic_spike(
    neuron_idx: usize,
    timestamp: u16,
    dendrite_constant: &i8,
    synapse_alphas: &mut [u8],
    synapse_last_events: &mut [u16],
    producer: &EventProducer,
) {
    let branch_constant = *dendrite_constant;

    for s_idx in 0..synapse_alphas.len() {
        let alpha = update_synapse_alpha(s_idx, timestamp, synapse_alphas, synapse_last_events);
        if alpha > H_ALPHA {
            synapse_alphas[s_idx] = alpha.saturating_add(branch_constant.unsigned_abs());
        }
    }

    let soma_delta = branch_constant.max(1) as i16;
    producer.push(Event::soma_signal(neuron_idx as u32, timestamp, soma_delta));
}


// A voltage delta arrives at the soma (from a dendritic spike or an apical plateau). Integrate it
// through the soma's state machine; if it bursts, emit one SOMATIC_SPIKE per AP in the burst.
// NOTE: a burst of N emits N SOMATIC_SPIKEs (preserving the prior per-AP fan-out — each drives its
// own FORWARD_AP and BaP sweep). Revisit if burst should instead be a single event + multiplier.
pub fn handle_soma_signal(
    neuron_idx: usize,
    timestamp: u16,
    v_s: i16,
    soma_potentials: &mut [i8],
    soma_last_events: &mut [u16],
    soma_thresholds: &[i8],
    soma_betas: &mut [u8],
    producer: &EventProducer,
) {
    let burst = update_soma_potential(
        timestamp, neuron_idx,
        soma_potentials, soma_last_events, soma_thresholds, soma_betas, v_s,
    );
    for _ in 0..burst {
        producer.push(Event::spike(SOMATIC_SPIKE, neuron_idx as u32, timestamp));
    }
}


// The soma fired (a SOMATIC_SPIKE). Run BaP weight updates across all of the neuron's synapses
// using the current burst counter beta, then emit a FORWARD_AP downstream. beta is read-only here:
// all of its dynamics (lazy decay + burst increment) live in update_soma_potential.
// synapse slices must already be scoped to this neuron via neuron_synapse_range.
pub fn handle_somatic_spike(
    neuron_idx: usize,
    timestamp: u16,
    beta: u8,
    soma_lr: i16,
    synapse_weights: &mut [i8],
    synapse_alphas: &mut [u8],
    synapse_last_events: &mut [u16],
    producer: &EventProducer,
) {
    // BaP updates for all synapses of this neuron: w += (beta - H_BETA) * alpha / lr
    for s_idx in 0..synapse_weights.len() {
        update_weight(timestamp, beta, soma_lr, s_idx, synapse_alphas, synapse_last_events, synapse_weights);
    }
    producer.push(Event::spike(FORWARD_AP, neuron_idx as u32, timestamp));
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::event::{EventQueue, SOMA_SIGNAL};

    // --- handle_synapse_signal (basal) ---

    #[test]
    fn synapse_signal_basal_boosts_alpha_and_integrates_no_fire() {
        let queue = EventQueue::new(64);
        let producer = queue.producer_handle();
        let xs = [10u8];
        let mut alphas = [0u8];
        let mut last_events = [0u16];
        let weights = [10i8];
        let mut dendrite_activity = 0u16;
        let mut dendrite_last_event = 0u16;

        handle_synapse_signal(
            0, 5, 9, 0, xs.len(),
            &xs, &mut alphas, &mut last_events, &weights,
            &mut dendrite_activity, &mut dendrite_last_event, 1000, false,
            &producer,
        );

        assert_eq!(alphas[0], ALPHA_BOOST);   // decayed to 0 (elapsed=0), boosted by 64
        assert_eq!(dendrite_activity, 10);    // gamma=0, delta = 10 * 1
        assert_eq!(queue.drain().len(), 0);   // below threshold → no event
    }

    #[test]
    fn synapse_signal_basal_threshold_crossed_emits_dendritic_spike() {
        let queue = EventQueue::new(64);
        let producer = queue.producer_handle();
        let xs = [10u8];
        let mut alphas = [0u8];
        let mut last_events = [0u16];
        let weights = [10i8];
        let mut dendrite_activity = 995u16;
        let mut dendrite_last_event = 0u16;

        handle_synapse_signal(
            0, 3, 9, 0, xs.len(),
            &xs, &mut alphas, &mut last_events, &weights,
            &mut dendrite_activity, &mut dendrite_last_event, 1000, false,
            &producer,
        );

        assert_eq!(dendrite_activity, 0); // 995 + 10 = 1005 >= 1000 → fire + reset
        let events = queue.drain();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, DENDRITIC_SPIKE);
        assert_eq!(events[0].source, 3); // dendrite_idx
    }

    // --- handle_synapse_signal (apical) ---

    #[test]
    fn synapse_signal_apical_emits_soma_signal_with_plateau() {
        let queue = EventQueue::new(64);
        let producer = queue.producer_handle();
        let xs = [10u8];
        let mut alphas = [0u8];
        let mut last_events = [0u16];
        let weights = [10i8];
        let mut dendrite_activity = 0u16;
        let mut dendrite_last_event = 0u16;
        let theta = 0u16; // V_B above θ_B → upper half of the sigmoid

        // V_B = 0 + 10 = 10; plateau = apical_plateau(10, 0, 64, 9) = 32.
        handle_synapse_signal(
            0, 5, 7, 0, xs.len(),
            &xs, &mut alphas, &mut last_events, &weights,
            &mut dendrite_activity, &mut dendrite_last_event, theta, true,
            &producer,
        );

        assert_eq!(dendrite_activity, 10); // apical: NOT reset
        let events = queue.drain();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, SOMA_SIGNAL);
        assert_eq!(events[0].source, 7);   // neuron_idx
        assert_eq!(events[0].payload, 32); // the plateau voltage
    }

    // --- handle_dendritic_spike ---

    #[test]
    fn dendritic_spike_proximal_boosts_alpha_and_signals_soma() {
        let queue = EventQueue::new(64);
        let producer = queue.producer_handle();
        let dendrite_constant = 5i8;
        let mut alphas = [50u8]; // > H_ALPHA=30
        let mut last_events = [0u16];

        handle_dendritic_spike(
            8, 0, &dendrite_constant, &mut alphas, &mut last_events, &producer,
        );

        assert_eq!(alphas[0], 55); // 50 + branch_constant.unsigned_abs()=5
        let events = queue.drain();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, SOMA_SIGNAL);
        assert_eq!(events[0].source, 8);  // neuron_idx
        assert_eq!(events[0].payload, 5); // max(5, 1)
    }

    #[test]
    fn dendritic_spike_distal_signals_soma_with_one() {
        let queue = EventQueue::new(64);
        let producer = queue.producer_handle();
        let dendrite_constant = -10i8;

        handle_dendritic_spike(
            1, 0, &dendrite_constant, &mut [], &mut [], &producer,
        );

        let events = queue.drain();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, SOMA_SIGNAL);
        assert_eq!(events[0].payload, 1); // max(-10, 1)
    }

    // --- handle_soma_signal ---

    #[test]
    fn soma_signal_below_threshold_accumulates_no_spike() {
        let queue = EventQueue::new(64);
        let producer = queue.producer_handle();
        let mut potentials = [0i8];
        let mut last_events = [0u16];
        let thresholds = [100i8];
        let mut betas = [0u8];

        handle_soma_signal(0, 0, 10, &mut potentials, &mut last_events, &thresholds, &mut betas, &producer);

        assert_eq!(potentials[0], 10);
        assert_eq!(queue.drain().len(), 0);
    }

    #[test]
    fn soma_signal_burst_emits_one_somatic_spike_per_ap() {
        let queue = EventQueue::new(64);
        let producer = queue.producer_handle();
        let mut potentials = [0i8];
        let mut last_events = [0u16];
        let thresholds = [10i8];
        let mut betas = [0u8];

        // new_v = 35 >= 10 → burst = 3
        handle_soma_signal(0, 200, 35, &mut potentials, &mut last_events, &thresholds, &mut betas, &producer);

        let events = queue.drain();
        assert_eq!(events.len(), 3);
        assert!(events.iter().all(|e| e.event_type == SOMATIC_SPIKE && e.source == 0));
        assert_eq!(betas[0], 3); // beta reinforced by burst size
    }

    // --- handle_somatic_spike ---

    #[test]
    fn somatic_spike_updates_weights_and_emits_forward_ap() {
        let queue = EventQueue::new(64);
        let producer = queue.producer_handle();
        let mut weights = [0i8];
        let mut alphas = [200u8];       // > H_ALPHA=30
        let mut last_events = [100u16]; // same ts, no decay

        // beta=6: burst_term=6-4=2, delta=2*200/100=4
        handle_somatic_spike(0, 100, 6, 100, &mut weights, &mut alphas, &mut last_events, &producer);

        assert_eq!(weights[0], 4);
        let events = queue.drain();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, FORWARD_AP);
        assert_eq!(events[0].source, 0);
    }

    #[test]
    fn somatic_spike_no_synapses_still_emits_forward_ap() {
        let queue = EventQueue::new(64);
        let producer = queue.producer_handle();

        handle_somatic_spike(42, 100, 6, 100, &mut [], &mut [], &mut [], &producer);

        let events = queue.drain();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, FORWARD_AP);
        assert_eq!(events[0].source, 42);
        assert_eq!(events[0].timestamp, 100);
    }
}
