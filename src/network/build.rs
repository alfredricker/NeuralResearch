use crate::network::build;
use crate::neuron::population::Population;
use crate::network::topology::conn::{Connection, ConnRule};
use crate::neuron::dendrite::Compartment;
use crate::neuron::config::NeuronConfig;
use crate::network::{Network, Soma, Dendrite, Synapse, Axon};
use std::ops::Range;
use rand::{RngExt,Rng};

pub struct NetworkBuilder {
    pub populations: Vec<Population>,
    pub connections: Vec<Connection>,
}

impl NetworkBuilder {
    pub fn add(&mut self, config: &'static NeuronConfig, size: u32) -> u32 { // output is population id
        let id = self.populations.len() as u32;
        self.populations.push(Population {
            config,
            size,
            name: config.name, // debug only
        });
        id
    }

    pub fn connect(&mut self, from: u32, to: u32, c: Compartment, rule: ConnRule) {
        self.connections.push(Connection {from, to, compartment: c, rule});
    }
}

pub enum BuildError {

}

pub fn build_network<R: Rng + RngExt>(builder: NetworkBuilder, rng: &mut R) -> Network {
    // Construct the SoA structs empty; generate_neurons appends into them.
    let mut somas = Soma {
        soma_potentials: Vec::new(), soma_thresholds: Vec::new(),
        soma_betas: Vec::new(), soma_last_events: Vec::new(),
        soma_lrs: Vec::new(), dendrite_offsets: Vec::new(),
    };
    let mut dendrites = Dendrite {
        dendrite_activities: Vec::new(), dendrite_last_events: Vec::new(),
        dendrite_constants: Vec::new(), dendrite_thresholds: Vec::new(),
        synapse_offsets: Vec::new(), live_synapse_counts: Vec::new(),
        dendrite_to_neuron: Vec::new(), dendrite_is_apical: Vec::new(),
    };
    let mut synapses = Synapse {
        synapse_weights: Vec::new(), synapse_x: Vec::new(),
        synapse_alphas: Vec::new(), synapse_last_events: Vec::new(),
    };

    // 1. Generate each population; remember its global neuron range.
    let mut ranges: Vec<Range<u32>> = Vec::with_capacity(builder.populations.len());
    for pop in &builder.populations {
        let start = somas.soma_potentials.len() as u32;
        pop.generate_neurons(rng, &mut somas, &mut dendrites, &mut synapses);
        let end = somas.soma_potentials.len() as u32;
        ranges.push(start..end);
    }

    // 2. Append the trailing sentinels (generate_neurons deliberately omits them).
    somas.dendrite_offsets.push(dendrites.dendrite_activities.len() as u32);
    dendrites.synapse_offsets.push(synapses.synapse_weights.len() as u32);

    // 3. Resolve connections into (src_neuron, target_synapse) axon edges.
    //    `consumed[d]` tracks how many of dendrite d's live slots are already claimed,
    //    so no synapse slot is wired to two presynaptic axons.
    let mut consumed: Vec<u32> = vec![0; dendrites.dendrite_activities.len()];
    let mut axon_edges: Vec<(u32, u32)> = Vec::new();

    for c in &builder.connections {
        let src: Vec<u32> = ranges[c.from as usize].clone().collect();
        let dst: Vec<u32> = ranges[c.to as usize].clone().collect();

        let mut edges = Vec::new();
        c.rule.apply(&src, &dst, rng, &mut edges).expect("connection rule failed");

        let want_apical = matches!(c.compartment, Compartment::Apical) as u8;
        for (s, d) in edges {
            // find the first free synapse slot on a matching-compartment dendrite of d
            let d0 = somas.dendrite_offsets[d as usize] as usize;
            let d1 = somas.dendrite_offsets[d as usize + 1] as usize;
            for den in d0..d1 {
                if dendrites.dendrite_is_apical[den] != want_apical { continue; }
                let live = dendrites.live_synapse_counts[den] as u32;
                if consumed[den] < live {
                    let slot = dendrites.synapse_offsets[den] + consumed[den];
                    consumed[den] += 1;
                    axon_edges.push((s, slot));
                    break;
                }
                // else this dendrite is full → try the next one; if none, edge is dropped
            }
        }
    }

    // 4. Build axon CSR. Sort by source neuron so targets land in offset order.
    axon_edges.sort_by_key(|&(s, _)| s);
    let n = somas.soma_potentials.len();
    let mut axon_offsets = vec![0u32; n + 1];
    for &(s, _) in &axon_edges {
        axon_offsets[s as usize + 1] += 1;
    }
    for i in 0..n {
        axon_offsets[i + 1] += axon_offsets[i];
    }
    let axon_targets: Vec<u32> = axon_edges.iter().map(|&(_, t)| t).collect();

    Network {
        synapses,
        dendrites,
        somas,
        axons: Axon { axon_targets, axon_offsets },
    }
}