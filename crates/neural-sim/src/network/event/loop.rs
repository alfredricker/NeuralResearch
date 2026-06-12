use crate::network::event::event::{SOMATIC_SPIKE, DENDRITIC_SPIKE, SOMA_SIGNAL, SYNAPSE_SIGNAL};
use crate::network::event::queue::EventQueue;
use crate::network::event::handlers::{handle_somatic_spike, handle_dendritic_spike, handle_synapse_signal, handle_soma_signal};
use crate::network::event::slice::{neuron_synapse_range, dendrite_synapse_range};
use crate::neuron::dendrite::synapse_to_dendrite;
use crate::telemetry::TelemetrySink;

pub fn run_event_loop(
    queue: &EventQueue,
    // telemetry observer. Pass `&mut NullSink` in production/GPU builds — every call inlines
    // away. The dashboard passes a RecordingSink that traces events into a `.ntr` file.
    sink: &mut impl TelemetrySink,
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
    // readout — per-neuron AP accumulator (length = n_neurons). The trial harness zeroes it at
    // each trial boundary; an Effector (crate::io) reads its output-range window after the queue
    // drains. Counts EVERY somatic spike — including the ones an InputSpace asserts on input
    // neurons — so it also serves as a full-network activity recorder.
    spike_counts: &mut [u32],
) {
    let producer = queue.producer_handle();

    // One wavefront per call: process exactly the events queued on entry, advancing the queue's
    // head past them. Events the handlers push below land beyond this wavefront's tail and are
    // picked up by the next call — so a cascade marches forward one generation per call, and the
    // caller's clock advances between calls (see crate::trial). `e` is an owned Copy of the event.
    for e in queue.next_wavefront() {
        sink.on_event(&e); // fine-grained trace; NullSink inlines this to nothing
        match e.event_type {
            SOMATIC_SPIKE => {
                let n = e.source as usize;
                // accumulate the burst (AP count, carried in the payload) for readout
                spike_counts[n] += e.payload.max(0) as u32;
                let (s_start, s_end) = neuron_synapse_range(n, dendrite_offsets, synapse_offsets);
                let axons = &axon_targets[axon_offsets[n] as usize..axon_offsets[n + 1] as usize];
                handle_somatic_spike(
                    e.timestamp,
                    e.payload as u16, // burst count, threaded onto each fanned-out SYNAPSE_SIGNAL
                    soma_betas[n],    // read-only; beta dynamics live in update_soma_potential
                    soma_lrs[n],
                    &mut synapse_weights[s_start..s_end],
                    &mut synapse_alphas[s_start..s_end],
                    &mut synapse_last_events[s_start..s_end],
                    axons,
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
            // One AP delivery onto a single target synapse (fanned out from a somatic spike). The
            // target dendrite is basal or apical (dendrite_is_apical[d]); handle_synapse_signal
            // routes accordingly — basal → DENDRITIC_SPIKE, apical → SOMA_SIGNAL. One axon drives
            // both compartments; there is no separate feedback event.
            SYNAPSE_SIGNAL => {
                let s = e.source as usize;
                let d = synapse_to_dendrite(s, synapse_offsets);
                let target_n = dendrite_to_neuron[d] as usize;
                let (s_start, s_end) = dendrite_synapse_range(d, synapse_offsets);
                let local_s = s - s_start;
                // live_end is in slice-local coordinates: the slice starts at the dendrite base,
                // and live synapses are packed at the front, so live_end == the count.
                let live_end = dendrite_live_counts[d] as usize;
                let is_apical = dendrite_is_apical[d] == 1;
                let burst = (e.payload.max(1)) as u16; // presynaptic burst scales the EPSP
                handle_synapse_signal(
                    local_s,
                    d,        // dendrite_idx — source of a basal DENDRITIC_SPIKE
                    target_n, // neuron_idx — target of an apical SOMA_SIGNAL
                    e.timestamp,
                    burst,
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
            _ => {}
        }
    }
}
