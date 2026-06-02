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

        // convert connection apical enum to u8 for comparison with dendrites.dendrite_is_apical;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::sample::{SamplerI8, SamplerU8};
    use crate::neuron::dendrite::synapse_to_dendrite;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    // Tiny deterministic geometry: 2 basal dendrites, 1 branch each, 3 live synapses
    // per dendrite, no apical. std=0 samplers pin the counts so the layout is exact;
    // the synapse-x sampler keeps a spread so 3 distinct positions are easy to draw.
    fn tiny_config() -> &'static NeuronConfig {
        Box::leak(Box::new(NeuronConfig::new(
            "test",
            2,                       // n_basal_dendrites
            None,                    // n_apical_dendrites
            SamplerU8::new(128, 50), // synapse_x_sampler
            SamplerU8::new(1, 0),    // dendrites_per_branch = 1
            SamplerU8::new(3, 0),    // synapses_per_dendrite = 3
            10,                      // soma_threshold
            1000,                    // basal_dendrite_threshold
            SamplerI8::new(60, 0),   // basal_dendrite_constant
            None,                    // apical_dendrite_threshold
            None,                    // apical_dendrite_constant
            120,                     // learning_rate
        )))
    }

    // Two populations of 2 neurons, wired A -> C one-to-one onto the basal compartment.
    fn one_to_one_net() -> Network {
        let cfg = tiny_config();
        let mut b = NetworkBuilder { populations: Vec::new(), connections: Vec::new() };
        let a = b.add(cfg, 2);
        let c = b.add(cfg, 2);
        b.connect(a, c, Compartment::Basal, ConnRule::OneToOne);
        let mut rng = SmallRng::seed_from_u64(7);
        build_network(b, &mut rng)
    }

    #[test]
    fn offset_arrays_carry_trailing_sentinels() {
        let net = one_to_one_net();
        let n_neurons = net.somas.soma_potentials.len();
        let n_dendrites = net.dendrites.dendrite_activities.len();

        // 4 neurons × 2 basal dendrites = 8 dendrites.
        assert_eq!(n_neurons, 4);
        assert_eq!(n_dendrites, 8);

        // length+1 with the sentinel pointing one past the end of the indexed array.
        assert_eq!(net.somas.dendrite_offsets.len(), n_neurons + 1);
        assert_eq!(net.dendrites.synapse_offsets.len(), n_dendrites + 1);
        assert_eq!(*net.somas.dendrite_offsets.last().unwrap(), n_dendrites as u32);
        assert_eq!(
            *net.dendrites.synapse_offsets.last().unwrap(),
            net.synapses.synapse_weights.len() as u32
        );
    }

    #[test]
    fn axon_csr_is_well_formed() {
        let net = one_to_one_net();
        let n = net.somas.soma_potentials.len();

        assert_eq!(net.axons.axon_offsets.len(), n + 1);
        // monotonic non-decreasing
        assert!(net.axons.axon_offsets.windows(2).all(|w| w[0] <= w[1]));
        // final offset equals the flat target count
        assert_eq!(
            *net.axons.axon_offsets.last().unwrap() as usize,
            net.axons.axon_targets.len()
        );
    }

    #[test]
    fn one_to_one_wires_each_source_once() {
        let net = one_to_one_net();
        // pop A occupies neurons 0,1; pop C occupies 2,3. One-to-one → 2 edges.
        assert_eq!(net.axons.axon_targets.len(), 2);

        let outdeg = |i: usize| {
            (net.axons.axon_offsets[i + 1] - net.axons.axon_offsets[i]) as usize
        };
        assert_eq!(outdeg(0), 1); // source neuron 0
        assert_eq!(outdeg(1), 1); // source neuron 1
        assert_eq!(outdeg(2), 0); // pop C neurons send nothing
        assert_eq!(outdeg(3), 0);
    }

    #[test]
    fn targets_are_distinct_basal_synapses_of_dst() {
        let net = one_to_one_net();
        let offs = &net.dendrites.synapse_offsets;

        // no synapse slot is claimed by two presynaptic axons
        let mut seen = net.axons.axon_targets.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), net.axons.axon_targets.len(), "duplicate target slot");

        for &t in &net.axons.axon_targets {
            let d = synapse_to_dendrite(t as usize, offs);
            // landed on a basal dendrite (compartment we connected onto)
            assert_eq!(net.dendrites.dendrite_is_apical[d], 0);
            // owning neuron is in pop C (neurons 2 or 3)
            let owner = net.dendrites.dendrite_to_neuron[d];
            assert!(owner == 2 || owner == 3, "target on wrong population: {owner}");
            // slot sits within the dendrite's live prefix
            let base = offs[d];
            let live = net.dendrites.live_synapse_counts[d] as u32;
            assert!(t >= base && t < base + live, "slot {t} outside live prefix");
        }
    }

    #[test]
    fn synapse_x_sorted_within_each_live_block() {
        // load-bearing invariant for dendritic integration: xs ascending per dendrite.
        let net = one_to_one_net();
        let offs = &net.dendrites.synapse_offsets;
        for d in 0..net.dendrites.dendrite_activities.len() {
            let base = offs[d] as usize;
            let live = net.dendrites.live_synapse_counts[d] as usize;
            let block = &net.synapses.synapse_x[base..base + live];
            assert!(block.windows(2).all(|w| w[0] < w[1]), "xs not sorted/unique at d={d}");
        }
    }
}