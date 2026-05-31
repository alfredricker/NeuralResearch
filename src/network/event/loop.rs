use crate::network::event::event::{SOMATIC_SPIKE, DENDRITIC_SPIKE, FORWARD_AP, APICAL_FB};
use crate::network::event::queue::EventQueue;
use crate::network::event::handlers::{handle_somatic_spike, handle_dendritic_spike, handle_forward_ap, handle_apical_fb};
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
                    &mut soma_betas[n],
                    &mut soma_last_events[n],
                    &soma_lrs[n],
                    &mut synapse_weights[s_start..s_end],
                    &mut synapse_alphas[s_start..s_end],
                    &mut synapse_last_events[s_start..s_end],
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
                    &mut dendrite_last_events[d],
                    &mut soma_potentials[n],
                    &soma_thresholds[n],
                    &mut synapse_alphas[s_start..s_end],
                    &mut synapse_last_events[s_start..s_end],
                    &producer,
                );
            }
            // @TODO: a loop inside the event loop is not ideal -- 
            //figure out a way to batch these or trigger an async parallel event for each item in the loop
            FORWARD_AP => {
                let n = e.source as usize;
                for &s in &axon_targets[axon_offsets[n] as usize..axon_offsets[n + 1] as usize] {
                    let s = s as usize;
                    let d = synapse_to_dendrite(s, synapse_offsets);
                    let (s_start, s_end) = dendrite_synapse_range(d, synapse_offsets);
                    let local_s = s - s_start;
                    // live_end is in slice-local coordinates: the slice starts at the dendrite
                    // base, and live synapses are packed at the front, so live_end == the count.
                    let live_end = dendrite_live_counts[d] as usize;
                    handle_forward_ap(
                        local_s,
                        d,
                        e.timestamp,
                        live_end,
                        &synapse_xs[s_start..s_end],
                        &mut synapse_alphas[s_start..s_end],
                        &mut synapse_last_events[s_start..s_end],
                        &synapse_weights[s_start..s_end],
                        &mut dendrite_activities[d],
                        &dendrite_thresholds[d],
                        &producer,
                    );
                }
            }
            // Apical feedback fans out like FORWARD_AP, but lands on apical synapses and
            // modulates the TARGET neuron's soma multiplicatively (handle_apical_fb).
            // NOTE: until a separate apical synapse compartment + apical axon CSR exist, this
            // reuses the feedforward axon_targets and derives the feedback gain from the target
            // dendrite's constant. See docs/09-gaps-and-open-questions.md (apical not fully wired).
            APICAL_FB => {
                let n = e.source as usize;
                for &s in &axon_targets[axon_offsets[n] as usize..axon_offsets[n + 1] as usize] {
                    let s = s as usize;
                    let d = synapse_to_dendrite(s, synapse_offsets);
                    let target_n = dendrite_to_neuron[d] as usize;
                    let (s_start, s_end) = dendrite_synapse_range(d, synapse_offsets);
                    let local_s = s - s_start;
                    let axon_constant = dendrite_constants[d].unsigned_abs();
                    handle_apical_fb(
                        local_s,
                        target_n,
                        e.timestamp,
                        axon_constant,
                        &mut synapse_alphas[s_start..s_end],
                        &mut synapse_last_events[s_start..s_end],
                        &mut soma_potentials[target_n],
                        soma_thresholds[target_n],
                        &producer,
                    );
                }
            }
            _ => {}
        }
    }
}
