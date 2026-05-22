use crate::network::event::{Event, SOMATIC_SPIKE, DENDRITIC_SPIKE, FORWARD_AP, EventProducer};
use crate::constants::{T_BETA, H_ALPHA, ALPHA_BOOST};
use crate::neuron::synapse::{update_weight, update_synapse_alpha};
use crate::neuron::dendrite::update_dendrite_activity;

// Somatic spike: update beta, BaP weight updates across all owned synapses, emit ForwardAP.
// Alpha decay on each synapse is lazy — computed here from synapse_last_events.
// synapse slices must already be scoped to this neuron via neuron_synapse_range.
pub fn handle_somatic_spike(
    neuron_idx: usize,
    timestamp: u16,
    beta: &mut u8,
    soma_last_event: &mut u16,
    soma_lr: &i16,
    synapse_weights: &mut [i8],
    synapse_alphas: &mut [u8],
    synapse_last_events: &mut [u16],
    producer: &EventProducer,
) {
    let elapsed = timestamp.wrapping_sub(*soma_last_event);
    let decrements = (elapsed / T_BETA).min(15) as u8;
    *beta = beta.saturating_sub(decrements).saturating_add(1).min(63);
    let beta = *beta;
    *soma_last_event = timestamp;

    let lr = *soma_lr;

    for s_idx in 0..synapse_weights.len() {
        update_weight(timestamp, beta, lr, s_idx, synapse_alphas, synapse_last_events, synapse_weights);
    }

    producer.push(Event { event_type: FORWARD_AP, source: neuron_idx as u32, timestamp });
}


// Dendritic spike: propagate to soma scaled by branch_constant (proximal vs distal),
// boost alpha on synapses active at spike time, emit SOMATIC_SPIKE if threshold crossed.
//
// branch_constant > 0: proximal — scales directly onto soma potential
// branch_constant <= 0: distal — attenuated to 1, strong local NMDA-like reinforcement
// synapse slices must already be scoped to this dendrite via dendrite_synapse_range.
pub fn handle_dendritic_spike(
    neuron_idx: usize,
    timestamp: u16,
    dendrite_constant: &i8,
    dendrite_last_event: &mut u16,
    soma_potential: &mut i8,
    soma_threshold: &i8,
    synapse_alphas: &mut [u8],
    synapse_last_events: &mut [u16],
    producer: &EventProducer,
) {
    *dendrite_last_event = timestamp;

    let branch_constant = *dendrite_constant;
    let soma_delta: i8 = branch_constant.max(1);
    *soma_potential = soma_potential.saturating_add(soma_delta);

    for s_idx in 0..synapse_alphas.len() {
        let alpha = update_synapse_alpha(s_idx, timestamp, synapse_alphas, synapse_last_events);
        if alpha > H_ALPHA {
            synapse_alphas[s_idx] = alpha.saturating_add(branch_constant.unsigned_abs());
        }
    }

    if *soma_potential >= *soma_threshold {
        *soma_potential = 0;
        producer.push(Event { event_type: SOMATIC_SPIKE, source: neuron_idx as u32, timestamp });
    }
}


// Forward AP received at a synapse: boost alpha, update dendrite voltage, emit DENDRITIC_SPIKE if threshold crossed.
// synapse slices must already be scoped to this dendrite via dendrite_synapse_range.
pub fn handle_forward_ap(
    s_idx: usize,
    dendrite_idx: usize,
    timestamp: u16,
    synapse_xs: &[u8],
    synapse_alphas: &mut [u8],
    synapse_last_events: &mut [u16],
    synapse_weights: &[i8],
    dendrite_activity: &mut u16,
    dendrite_threshold: &u16,
    producer: &EventProducer,
) {
    let alpha = update_synapse_alpha(s_idx, timestamp, synapse_alphas, synapse_last_events);
    synapse_alphas[s_idx] = alpha.saturating_add(ALPHA_BOOST);

    let delta = update_dendrite_activity(
        s_idx, timestamp,
        synapse_xs, synapse_alphas,
        synapse_weights, synapse_last_events,
    );
    *dendrite_activity = dendrite_activity.saturating_add_signed(delta);

    if *dendrite_activity >= *dendrite_threshold {
        *dendrite_activity = 0;
        producer.push(Event { event_type: DENDRITIC_SPIKE, source: dendrite_idx as u32, timestamp });
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::event::EventQueue;

    // --- handle_somatic_spike ---

    #[test]
    fn somatic_spike_beta_increments_and_emits_forward_ap() {
        let queue = EventQueue::new(64);
        let producer = queue.producer_handle();
        let mut beta = 5u8;
        let mut soma_last_event = 100u16;
        let soma_lr: i16 = 100;

        handle_somatic_spike(
            42, 100, &mut beta, &mut soma_last_event, &soma_lr,
            &mut [], &mut [], &mut [],
            &producer,
        );

        // elapsed=0, decrements=0, beta=5+1=6
        assert_eq!(beta, 6);
        assert_eq!(soma_last_event, 100);
        let events = queue.drain();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, FORWARD_AP);
        assert_eq!(events[0].source, 42);
        assert_eq!(events[0].timestamp, 100);
    }

    #[test]
    fn somatic_spike_beta_decays_with_elapsed_time() {
        let queue = EventQueue::new(64);
        let producer = queue.producer_handle();
        let mut beta = 10u8;
        let mut soma_last_event = 0u16;
        let soma_lr: i16 = 100;

        // elapsed=1000, decrements=1000/500=2, beta=10-2+1=9
        handle_somatic_spike(
            0, 1000, &mut beta, &mut soma_last_event, &soma_lr,
            &mut [], &mut [], &mut [],
            &producer,
        );

        assert_eq!(beta, 9);
    }

    #[test]
    fn somatic_spike_beta_capped_at_63() {
        let queue = EventQueue::new(64);
        let producer = queue.producer_handle();
        let mut beta = 63u8;
        let mut soma_last_event = 0u16;
        let soma_lr: i16 = 100;

        handle_somatic_spike(
            0, 0, &mut beta, &mut soma_last_event, &soma_lr,
            &mut [], &mut [], &mut [],
            &producer,
        );

        assert_eq!(beta, 63);
    }

    #[test]
    fn somatic_spike_updates_synapse_weights() {
        let queue = EventQueue::new(64);
        let producer = queue.producer_handle();
        let mut beta = 5u8;
        let mut soma_last_event = 100u16;
        let soma_lr: i16 = 100;
        let mut weights = [0i8];
        let mut alphas = [200u8];       // > H_ALPHA=30
        let mut last_events = [100u16]; // same ts, no decay

        handle_somatic_spike(
            0, 100, &mut beta, &mut soma_last_event, &soma_lr,
            &mut weights, &mut alphas, &mut last_events,
            &producer,
        );

        // beta becomes 6; burst_term=6-4=2, delta=2*200/100=4
        assert_eq!(weights[0], 4);
    }

    // --- handle_dendritic_spike ---

    #[test]
    fn dendritic_spike_proximal_accumulates_soma_potential_and_boosts_alpha() {
        let queue = EventQueue::new(64);
        let producer = queue.producer_handle();
        let dendrite_constant = 5i8;
        let mut dendrite_last_event = 0u16;
        let mut soma_potential = 10i8;
        let soma_threshold = 100i8;
        let mut alphas = [50u8]; // > H_ALPHA=30
        let mut last_events = [0u16];   // timestamp=0 → elapsed=0, no decay before boost

        handle_dendritic_spike(
            0, 0, &dendrite_constant, &mut dendrite_last_event,
            &mut soma_potential, &soma_threshold,
            &mut alphas, &mut last_events,
            &producer,
        );

        assert_eq!(soma_potential, 15);   // 10 + 5
        assert_eq!(alphas[0], 55);        // 50 + branch_constant.unsigned_abs()=5
        assert_eq!(queue.drain().len(), 0);
    }

    #[test]
    fn dendritic_spike_distal_adds_one_to_soma() {
        let queue = EventQueue::new(64);
        let producer = queue.producer_handle();
        let dendrite_constant = -10i8;
        let mut dendrite_last_event = 0u16;
        let mut soma_potential = 5i8;
        let soma_threshold = 100i8;

        handle_dendritic_spike(
            0, 0, &dendrite_constant, &mut dendrite_last_event,
            &mut soma_potential, &soma_threshold,
            &mut [], &mut [],
            &producer,
        );

        assert_eq!(soma_potential, 6); // 5 + max(-10, 1) = 5 + 1
    }

    #[test]
    fn dendritic_spike_threshold_crossed_emits_somatic_spike_and_resets() {
        let queue = EventQueue::new(64);
        let producer = queue.producer_handle();
        let dendrite_constant = 5i8;
        let mut dendrite_last_event = 0u16;
        let mut soma_potential = 95i8;
        let soma_threshold = 100i8;

        handle_dendritic_spike(
            7, 200, &dendrite_constant, &mut dendrite_last_event,
            &mut soma_potential, &soma_threshold,
            &mut [], &mut [],
            &producer,
        );

        // 95 + 5 = 100 >= 100 → spike, reset to 0
        assert_eq!(soma_potential, 0);
        let events = queue.drain();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, SOMATIC_SPIKE);
        assert_eq!(events[0].source, 7);
        assert_eq!(events[0].timestamp, 200);
    }

    // --- handle_forward_ap ---

    #[test]
    fn forward_ap_boosts_alpha_and_accumulates_dendrite_activity() {
        let queue = EventQueue::new(64);
        let producer = queue.producer_handle();
        let xs = [10u8];
        let mut alphas = [0u8];
        let mut last_events = [0u16];
        let weights = [10i8];
        let mut dendrite_activity = 0u16;
        let dendrite_threshold = 1000u16;

        handle_forward_ap(
            0, 5, 0, &xs, &mut alphas, &mut last_events,
            &weights, &mut dendrite_activity, &dendrite_threshold,
            &producer,
        );

        // alpha: decayed to 0 (elapsed=0), boosted by ALPHA_BOOST=64
        assert_eq!(alphas[0], ALPHA_BOOST);
        // single synapse, gamma=0, delta = 10 * 1 = 10
        assert_eq!(dendrite_activity, 10);
        assert_eq!(queue.drain().len(), 0);
    }

    #[test]
    fn forward_ap_threshold_crossed_emits_dendritic_spike_and_resets() {
        let queue = EventQueue::new(64);
        let producer = queue.producer_handle();
        let xs = [10u8];
        let mut alphas = [0u8];
        let mut last_events = [0u16];
        let weights = [10i8];
        let mut dendrite_activity = 995u16;
        let dendrite_threshold = 1000u16;

        handle_forward_ap(
            0, 3, 0, &xs, &mut alphas, &mut last_events,
            &weights, &mut dendrite_activity, &dendrite_threshold,
            &producer,
        );

        // 995 + 10 = 1005 >= 1000 → spike, reset to 0
        assert_eq!(dendrite_activity, 0);
        let events = queue.drain();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, DENDRITIC_SPIKE);
        assert_eq!(events[0].source, 3); // dendrite_idx
        assert_eq!(events[0].timestamp, 0);
    }
}

// Apical feedback event received at a synapse: boost alpha by axon_constant, apply
// multiplicative somatic update, emit one SOMATIC_SPIKE per threshold crossing.
pub fn handle_apical_fb(
    s_idx: usize,
    neuron_idx: usize,
    timestamp: u16,
    axon_constant: u8,
    synapse_alphas: &mut [u8],
    synapse_last_events: &mut [u16],
    soma_potential: &mut i8,
    soma_threshold: i8,
    producer: &EventProducer,
) {
    let alpha = update_synapse_alpha(s_idx, timestamp, synapse_alphas, synapse_last_events);
    let effective_alpha = alpha.saturating_add(axon_constant);
    let v_s = (*soma_potential).max(0) as i32;
    let new_v = *soma_potential as i32 + effective_alpha as i32 * v_s;

    let burst_count = new_v / soma_threshold as i32;
    *soma_potential = (new_v % soma_threshold as i32) as i8;

    for _ in 0..burst_count {
        producer.push(Event { event_type: SOMATIC_SPIKE, source: neuron_idx as u32, timestamp });
    }
}
