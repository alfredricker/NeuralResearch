use crate::network::event::{Event, SOMATIC_SPIKE, DENDRITIC_SPIKE, SYNAPSE_SIGNAL, EventProducer};
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
//   SOMA_SIGNAL     ─► handle_soma_signal     ─► SOMATIC_SPIKE (payload = burst)
//   SOMATIC_SPIKE   ─► handle_somatic_spike   ─► BaP sweep + one SYNAPSE_SIGNAL per axon target
//   SYNAPSE_SIGNAL  ─► handle_synapse_signal  ─► DENDRITIC_SPIKE (basal) | SOMA_SIGNAL (apical)
//   DENDRITIC_SPIKE ─► handle_dendritic_spike ─► SOMA_SIGNAL
//
// A somatic spike fans out to all of a neuron's axon targets as independent SYNAPSE_SIGNAL events
// (one queued AP delivery per target synapse — cheap to enqueue, parallelizable to drain). Each
// target dendrite is basal or apical (dendrite_is_apical), so one axon drives both compartments;
// there is no separate forward-AP or feedback event. The burst count rides the payload the whole
// way (SOMATIC_SPIKE → SYNAPSE_SIGNAL) and scales the EPSP at the receiving dendrite.
// ============================================================================================


// A SYNAPSE_SIGNAL: one upstream AP delivery landing on this synapse. The target dendrite is basal
// or apical (is_apical, from dendrite_is_apical[d]). Boost the receiving synapse, integrate its
// parent dendrite (the EPSP scaled by the presynaptic burst), and route the verdict onward:
//   basal  → DENDRITIC_SPIKE, if the branch crossed threshold and fired
//   apical → SOMA_SIGNAL, carrying the graded plateau depolarization
// synapse slices must already be scoped to this dendrite via dendrite_synapse_range.
pub fn handle_synapse_signal(
    s_idx: usize,
    dendrite_idx: usize,
    neuron_idx: usize,
    timestamp: u16,
    burst: u16, // presynaptic burst (AP count) — scales the EPSP in update_dendrite_activity
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
        s_idx, timestamp, burst, live_end,
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
// through the soma's state machine; if it bursts, emit a single SOMATIC_SPIKE carrying the burst
// count as its payload (downstream scales by it, rather than replaying N identical events).
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
    if burst > 0 {
        producer.push(Event::with_payload(SOMATIC_SPIKE, neuron_idx as u32, timestamp, burst as i16));
    }
}


// The soma fired (a SOMATIC_SPIKE). Two consequences:
//   1. BaP weight updates across the neuron's OWN afferent synapses (beta read-only; all of beta's
//      dynamics live in update_soma_potential). The update is beta-driven — the burst enters only
//      through beta, so it is not applied again here.
//   2. Axonal output: enqueue one SYNAPSE_SIGNAL per downstream target synapse, carrying the burst.
//      This is a push-only fan-out — the per-synapse work happens later, one independent event each.
// The neuron's own synapse slices must be scoped via neuron_synapse_range; axon_targets is this
// neuron's slice of the axon CSR (global indices into the downstream synapse arrays).
pub fn handle_somatic_spike(
    timestamp: u16,
    burst: u16,
    beta: u8,
    soma_lr: i16,
    synapse_weights: &mut [i8],
    synapse_alphas: &mut [u8],
    synapse_last_events: &mut [u16],
    axon_targets: &[u32],
    producer: &EventProducer,
) {
    // 1. BaP: w += (beta - H_BETA) * alpha / lr, for each of this neuron's synapses
    for s_idx in 0..synapse_weights.len() {
        update_weight(timestamp, beta, soma_lr, s_idx, synapse_alphas, synapse_last_events, synapse_weights);
    }
    // 2. fan out the axonal AP to every target synapse as an independent SYNAPSE_SIGNAL
    for &s in axon_targets {
        producer.push(Event::with_payload(SYNAPSE_SIGNAL, s, timestamp, burst as i16));
    }
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
            0, 5, 9, 0, 1, xs.len(),
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
            0, 3, 9, 0, 1, xs.len(),
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
            0, 5, 7, 0, 1, xs.len(),
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
    fn soma_signal_burst_emits_single_spike_carrying_count() {
        let queue = EventQueue::new(64);
        let producer = queue.producer_handle();
        let mut potentials = [0i8];
        let mut last_events = [0u16];
        let thresholds = [10i8];
        let mut betas = [0u8];

        // new_v = 35 >= 10 → burst = 3 → one SOMATIC_SPIKE with payload 3
        handle_soma_signal(0, 200, 35, &mut potentials, &mut last_events, &thresholds, &mut betas, &producer);

        let events = queue.drain();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, SOMATIC_SPIKE);
        assert_eq!(events[0].source, 0);
        assert_eq!(events[0].payload, 3); // burst count rides the payload
        assert_eq!(betas[0], 3);          // beta reinforced by burst size
    }

    // --- handle_somatic_spike ---

    #[test]
    fn somatic_spike_updates_weights_and_fans_out_synapse_signals() {
        let queue = EventQueue::new(64);
        let producer = queue.producer_handle();
        let mut weights = [0i8];
        let mut alphas = [200u8];       // > H_ALPHA=30
        let mut last_events = [100u16]; // same ts, no decay
        let axon_targets = [7u32, 9];   // two downstream target synapses

        // beta=6: burst_term=6-4=2, delta=2*200/100=4 (burst does NOT scale BaP)
        handle_somatic_spike(100, 1, 6, 100, &mut weights, &mut alphas, &mut last_events, &axon_targets, &producer);

        assert_eq!(weights[0], 4);
        let events = queue.drain();
        assert_eq!(events.len(), 2); // one SYNAPSE_SIGNAL per target
        assert!(events.iter().all(|e| e.event_type == SYNAPSE_SIGNAL));
        assert_eq!(events[0].source, 7);
        assert_eq!(events[1].source, 9);
    }

    #[test]
    fn somatic_spike_fan_out_carries_burst_payload() {
        let queue = EventQueue::new(64);
        let producer = queue.producer_handle();

        // no afferent synapses; burst=5 should ride each fanned-out SYNAPSE_SIGNAL
        handle_somatic_spike(100, 5, 6, 100, &mut [], &mut [], &mut [], &[42u32], &producer);

        let events = queue.drain();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, SYNAPSE_SIGNAL);
        assert_eq!(events[0].source, 42);   // target synapse index
        assert_eq!(events[0].payload, 5);   // burst
        assert_eq!(events[0].timestamp, 100);
    }
}
