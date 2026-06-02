use crate::network::event::event::{SOMATIC_SPIKE, DENDRITIC_SPIKE, FORWARD_AP, APICAL_FB, SOMA_SIGNAL};
use crate::network::event::queue::EventQueue;
use crate::network::event::handlers::{handle_somatic_spike, handle_dendritic_spike, handle_synapse_signal, handle_soma_signal};
use crate::network::event::slice::{neuron_synapse_range, dendrite_synapse_range};
use crate::neuron::dendrite::synapse_to_dendrite;

pub fn run_event_loop(
    queue: &EventQueue,
    // soma
    soma_potentials: &mut [i8],
    soma_thresholds: &[i8],
    soma_betas: &mut [u8],
    soma_last_events: &mut [u16],
    soma_lrs: &[i16],
    // dendrite
    dendrite_constants: &[i8],
    dendrite_last_events: &mut [u16],
    dendrite_activities: &mut [u16],
    dendrite_thresholds: &[u16],
    dendrite_is_apical: &[u8],
    dendrite_live_counts: &[u8], // number of live (bound) synapses per dendrite, packed at front of block
    dendrite_offsets: &[u32],
    dendrite_to_neuron: &[u32],
    // synapse
    synapse_weights: &mut [i8],
    synapse_alphas: &mut [u8],
    synapse_last_events: &mut [u16],
    synapse_xs: &[u8],
    synapse_offsets: &[u32],
    // axon
    axon_targets: &[u32],
    axon_offsets: &[u32],
) {
    let producer = queue.producer_handle();
    let events = queue.drain();

    for e in events {
        match e.event_type {
            SOMATIC_SPIKE => {
                let n = e.source as usize;
                let (s_start, s_end) = neuron_synapse_range(n, dendrite_offsets, synapse_offsets);
                handle_somatic_spike(
                    n,
                    e.timestamp,
                    soma_betas[n], // read-only; beta dynamics live in update_soma_potential
                    soma_lrs[n],
                    &mut synapse_weights[s_start..s_end],
                    &mut synapse_alphas[s_start..s_end],
                    &mut synapse_last_events[s_start..s_end],
                    &producer,
                );
            }
            SOMA_SIGNAL => {
                let n = e.source as usize;
                handle_soma_signal(
                    n,
                    e.timestamp,
                    e.payload, // v_s: the voltage delta to integrate
                    soma_potentials,
                    soma_last_events,
                    soma_thresholds,
                    soma_betas,
                    &producer,
                );
            }
            DENDRITIC_SPIKE => {
                let d = e.source as usize;
                let n = dendrite_to_neuron[d] as usize;
                let (s_start, s_end) = dendrite_synapse_range(d, synapse_offsets);
                handle_dendritic_spike(
                    n,
                    e.timestamp,
                    &dendrite_constants[d],
                    &mut synapse_alphas[s_start..s_end],
                    &mut synapse_last_events[s_start..s_end],
                    &producer,
                );
            }
            // @TODO: a loop inside the event loop is not ideal -- 
            //figure out a way to batch these or trigger an async parallel event for each item in the loop
            // FORWARD_AP (feedforward, basal) and APICAL_FB (top-down, apical) both deliver a synapse
            // signal; handle_synapse_signal branches on is_apical. NOTE: until a separate apical
            // synapse compartment + apical axon CSR exist, APICAL_FB reuses the feedforward
            // axon_targets. See docs/09-gaps-and-open-questions.md.
            FORWARD_AP | APICAL_FB => {
                let n = e.source as usize;
                let is_apical = e.event_type == APICAL_FB;
                for &s in &axon_targets[axon_offsets[n] as usize..axon_offsets[n + 1] as usize] {
                    let s = s as usize;
                    let d = synapse_to_dendrite(s, synapse_offsets);
                    let target_n = dendrite_to_neuron[d] as usize;
                    let (s_start, s_end) = dendrite_synapse_range(d, synapse_offsets);
                    let local_s = s - s_start;
                    // live_end is in slice-local coordinates: the slice starts at the dendrite
                    // base, and live synapses are packed at the front, so live_end == the count.
                    let live_end = dendrite_live_counts[d] as usize;
                    handle_synapse_signal(
                        local_s,
                        d,        // dendrite_idx — source of a basal DENDRITIC_SPIKE
                        target_n, // neuron_idx — target of an apical SOMA_SIGNAL
                        e.timestamp,
                        live_end,
                        &synapse_xs[s_start..s_end],
                        &mut synapse_alphas[s_start..s_end],
                        &mut synapse_last_events[s_start..s_end],
                        &synapse_weights[s_start..s_end],
                        &mut dendrite_activities[d],
                        &mut dendrite_last_events[d],
                        dendrite_thresholds[d],
                        is_apical,
                        &producer,
                    );
                }
            }
            _ => {}
        }
    }
}
