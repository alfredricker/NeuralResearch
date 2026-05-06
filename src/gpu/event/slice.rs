/// Functions to get relevant slice data for a given event that can be passed
/// to handlers.

// for a bap event, we need to get the slice of dendrites owned by the neuron
pub fn neuron_synapse_range(
    neuron_idx: usize,
    dendrite_offsets: &[u32],
    synapse_offsets: &[u32],
) -> (usize, usize) {
    let d_start = dendrite_offsets[neuron_idx] as usize;
    let d_end = dendrite_offsets[neuron_idx + 1] as usize;
    let s_start = synapse_offsets[d_start] as usize;
    let s_end = synapse_offsets[d_end] as usize;
    (s_start, s_end)
}


pub fn dendrite_synapse_range(
    dendrite_idx: usize,
    synapse_offsets: &[u32],
) -> (usize, usize) {
    let s_start = synapse_offsets[dendrite_idx] as usize;
    let s_end = synapse_offsets[dendrite_idx + 1] as usize;
    (s_start, s_end)
}