//! Static neuron structure, gathered from the engine's flat SoA into per-neuron, index-local
//! shapes the webview can draw directly. Sent once at build (it never changes during a run) — the
//! dynamic counterpart is [`crate::playground::frame`].

use std::collections::HashMap;

use neural_sim::network::TopologyView;
use serde::Serialize;

/// One synapse on a dendrite. `x` (0..=255) is its position along the branch — the layout
/// coordinate. `src_neuron` is the presynaptic neuron feeding it, or `None` for an unbound slot
/// (the norm for an isolated neuron, which is stimulated directly).
#[derive(Serialize)]
pub struct SynapseAnatomy {
    pub synapse: u32,
    pub x: u8,
    pub src_neuron: Option<u32>,
}

/// One dendrite. `branch_constant`'s sign is proximal (>0, passes to soma) vs distal (<=0,
/// attenuated, strong local learning); `is_apical` picks the graded-plateau vs hard-threshold
/// dynamics. Synapses are the live slots only, in ascending `x` order.
#[derive(Serialize)]
pub struct DendriteAnatomy {
    pub dendrite: u32,
    pub is_apical: bool,
    pub branch_constant: i8,
    pub threshold: u16,
    pub synapses: Vec<SynapseAnatomy>,
}

/// One neuron's full morphology: its soma threshold and its dendrites (basal first, then apical,
/// the build order). The frame arrays are parallel to these, so the view zips them by position.
#[derive(Serialize)]
pub struct NeuronAnatomy {
    pub neuron: u32,
    pub soma_threshold: i8,
    pub dendrites: Vec<DendriteAnatomy>,
}

/// Gather the whole (small, playground-scale) network into per-neuron anatomy. `edges` is
/// [`neural_sim::network::Network::edges`] — used to resolve each synapse slot back to its
/// presynaptic neuron.
pub fn gather(topo: &TopologyView, edges: &[(u32, u32, u32)]) -> Vec<NeuronAnatomy> {
    // synapse slot -> presynaptic neuron (edges are (src, dst, synapse_slot)).
    let src_of: HashMap<u32, u32> = edges.iter().map(|&(s, _d, syn)| (syn, s)).collect();

    (0..topo.n_neurons as u32)
        .map(|n| {
            let d0 = topo.dendrite_offsets[n as usize] as usize;
            let d1 = topo.dendrite_offsets[n as usize + 1] as usize;
            let dendrites = (d0..d1)
                .map(|d| {
                    let base = topo.synapse_offsets[d] as usize;
                    let live = topo.live_synapse_counts[d] as usize;
                    let synapses = (base..base + live)
                        .map(|s| SynapseAnatomy {
                            synapse: s as u32,
                            x: topo.synapse_x[s],
                            src_neuron: src_of.get(&(s as u32)).copied(),
                        })
                        .collect();
                    DendriteAnatomy {
                        dendrite: d as u32,
                        is_apical: topo.dendrite_is_apical[d] == 1,
                        branch_constant: topo.dendrite_constants[d],
                        threshold: topo.dendrite_thresholds[d],
                        synapses,
                    }
                })
                .collect();
            NeuronAnatomy { neuron: n, soma_threshold: topo.soma_thresholds[n as usize], dendrites }
        })
        .collect()
}
