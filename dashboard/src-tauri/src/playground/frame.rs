//! Per-wavefront dynamic state, gathered into arrays parallel to [`crate::playground::anatomy`] so
//! the view zips them. A frame is "the SoA values after one `step`, plus what fired during it" —
//! the latter captured by [`CaptureSink`], which exists only to populate a frame.

use std::collections::{HashMap, HashSet};

use neural_sim::network::TopologyView;
use neural_sim::network::event::{DENDRITIC_SPIKE, Event, SOMATIC_SPIKE, SYNAPSE_SIGNAL};
use neural_sim::telemetry::{NetworkView, TelemetrySink};
use serde::Serialize;

/// Records which elements were touched during one drained wavefront, by snooping the events the
/// loop processes. Note the one-generation lag inherent to the cascade: a `SYNAPSE_SIGNAL` is
/// *processed* in wavefront N (so its dendrite's V_B updates in N), but the `DENDRITIC_SPIKE` it
/// produces is *drained* in N+1 — so `fired_dendrites`/`soma_bursts` show the prior generation's
/// firings. That is the wave marching forward one step at a time, which is exactly what we animate.
#[derive(Default)]
pub struct CaptureSink {
    /// Synapse slots that received an AP this wavefront.
    pub signaled_synapses: HashSet<u32>,
    /// Dendrites that emitted a dendritic spike.
    pub fired_dendrites: HashSet<u32>,
    /// Neuron → total somatic burst (AP count) emitted this wavefront.
    pub soma_bursts: HashMap<u32, u32>,
}

impl TelemetrySink for CaptureSink {
    fn on_event(&mut self, e: &Event) {
        match e.event_type {
            SYNAPSE_SIGNAL => {
                self.signaled_synapses.insert(e.source);
            }
            DENDRITIC_SPIKE => {
                self.fired_dendrites.insert(e.source);
            }
            SOMATIC_SPIKE => {
                *self.soma_bursts.entry(e.source).or_default() += e.payload.max(0) as u32;
            }
            _ => {}
        }
    }
    fn on_snapshot(&mut self, _view: &NetworkView) {}
}

#[derive(Serialize)]
pub struct SynapseState {
    pub alpha: u8,
    pub weight: i8,
    pub signaled: bool,
}

#[derive(Serialize)]
pub struct DendriteState {
    pub v_b: u16,
    pub fired: bool,
    /// Live synapses, in the same order as the matching `DendriteAnatomy.synapses`.
    pub synapses: Vec<SynapseState>,
}

#[derive(Serialize)]
pub struct NeuronFrame {
    pub neuron: u32,
    pub soma_potential: i8,
    pub soma_beta: u8,
    /// APs the soma emitted this wavefront (0 if it didn't fire).
    pub soma_burst: u32,
    /// Dendrites in the same order as the matching `NeuronAnatomy.dendrites`.
    pub dendrites: Vec<DendriteState>,
}

/// One step's worth of network state. `clock` is the sim timestamp the wavefront ran at.
#[derive(Serialize)]
pub struct NetworkFrame {
    pub clock: u16,
    pub neurons: Vec<NeuronFrame>,
}

/// Gather the dynamic state into frames parallel to the anatomy. Walks the same neuron → dendrite →
/// synapse structure as [`super::anatomy::gather`], reading values from `view` and firing flags
/// from `cap`. Pass a default `CaptureSink` for a quiescent snapshot (nothing fired).
pub fn gather(topo: &TopologyView, view: &NetworkView, cap: &CaptureSink, clock: u16) -> NetworkFrame {
    let neurons = (0..topo.n_neurons as u32)
        .map(|n| {
            let d0 = topo.dendrite_offsets[n as usize] as usize;
            let d1 = topo.dendrite_offsets[n as usize + 1] as usize;
            let dendrites = (d0..d1)
                .map(|d| {
                    let base = topo.synapse_offsets[d] as usize;
                    let live = topo.live_synapse_counts[d] as usize;
                    let synapses = (base..base + live)
                        .map(|s| SynapseState {
                            alpha: view.synapse_alphas[s],
                            weight: view.synapse_weights[s],
                            signaled: cap.signaled_synapses.contains(&(s as u32)),
                        })
                        .collect();
                    DendriteState {
                        v_b: view.dendrite_activities[d],
                        fired: cap.fired_dendrites.contains(&(d as u32)),
                        synapses,
                    }
                })
                .collect();
            NeuronFrame {
                neuron: n,
                soma_potential: view.soma_potentials[n as usize],
                soma_beta: view.soma_betas[n as usize],
                soma_burst: cap.soma_bursts.get(&n).copied().unwrap_or(0),
                dendrites,
            }
        })
        .collect();
    NetworkFrame { clock, neurons }
}
