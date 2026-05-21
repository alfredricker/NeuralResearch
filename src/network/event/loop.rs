use crate::network::event::event::{Event, SOMATIC_SPIKE, DENDRITIC_SPIKE};
use crate::network::event::queue::EventQueue;
use crate::network::event::handlers::{handle_somatic_spike, handle_dendritic_spike};
use crate::network::event::slice::{neuron_synapse_range, dendrite_synapse_range};

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
    dendrite_offsets: &[u32],
    dendrite_to_neuron: &[u32],  // dendrite_to_neuron[dendrite_idx] = neuron_idx
    // synapse
    synapse_weights: &mut [i8],
    synapse_alphas: &mut [u8],
    synapse_last_events: &mut [u16],
    synapse_offsets: &[u32],
) {
    let (event_buf, event_tail, event_capacity) = queue.producer_handle();
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
                    event_buf, event_tail, event_capacity,
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
                    event_buf, event_tail, event_capacity,
                );
            }
            _ => {}
        }
    }
}
