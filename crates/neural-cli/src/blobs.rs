//! A tiny synthetic task: 2D blobs on a 5×5 place-cell grid, classified by quadrant.
//!
//! Two clusters — one south-west, one north-east of a unit square — are encoded by a 5×5 sheet of
//! place cells: a sampled point lights a Gaussian bump over the nearby cells (frame = 25 `u8`
//! intensities), exactly the `Grid2D` identity path MNIST uses. The network learns to map "bump in
//! the SW" → output 0 and "bump in the NE" → output 1. Small enough (25 → 16 → 2 ≈ 43 neurons) to
//! render whole in the dashboard and to snapshot every tick for membrane-potential scrubbing.

use std::collections::BTreeMap;

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use neural_sim::io::{Effector, InputSpace, Shape, input_config, output_config};
use neural_sim::math::sample::{SamplerI8, SamplerU8};
use neural_sim::network::Network;
use neural_sim::network::build::NetworkBuilder;
use neural_sim::network::topology::conn::ConnRule;
use neural_sim::neuron::config::NeuronConfig;
use neural_sim::neuron::dendrite::Compartment;

use crate::Experiment;
use crate::args::Args;

/// Side length of the square place-cell sheet → `GRID * GRID` input neurons.
const GRID: u32 = 5;

/// Cluster centres in grid coordinates `(col, row)`, row 0 at the top. Class 0 sits south-west
/// (bottom-left), class 1 north-east (top-right) — well separated on the 5×5 sheet.
const CENTERS: [(f32, f32); 2] = [(1.3, 3.5), (3.2, 1.4)];

/// Width of the uniform jitter applied to each sampled point around its class centre.
const SPREAD: f32 = 0.6;

/// Hidden-layer neuron type — a small pyramidal cell with basal dendrites receiving the place-cell
/// projection (no apical compartment). `Box::leak` yields the `&'static` the builder requires.
fn hidden_config() -> &'static NeuronConfig {
    Box::leak(Box::new(NeuronConfig::new(
        "hidden",
        6,                       // n_basal_dendrites
        None,                    // n_apical_dendrites
        SamplerU8::new(128, 50), // synapse_x_sampler
        SamplerU8::new(1, 0),    // dendrites_per_branch
        SamplerU8::new(16, 0),   // synapses_per_dendrite
        20,                      // soma_threshold
        500,                     // basal_dendrite_threshold
        SamplerI8::new(40, 10),  // basal_dendrite_constant
        None,                    // apical_dendrite_threshold
        None,                    // apical_dendrite_constant
        neural_sim::constants::MSLR as i16,
    )))
}

/// Render a Gaussian bump centred at `(cx, cy)` (grid coords) into a `GRID*GRID`-pixel `u8` frame.
fn bump(cx: f32, cy: f32) -> Vec<u8> {
    const SIGMA2: f32 = 0.9 * 0.9;
    let mut frame = vec![0u8; (GRID * GRID) as usize];
    for r in 0..GRID {
        for c in 0..GRID {
            let d2 = (c as f32 - cx).powi(2) + (r as f32 - cy).powi(2);
            let v = (-d2 / (2.0 * SIGMA2)).exp(); // 0..1, peak at the centre
            frame[(r * GRID + c) as usize] = (v * 255.0).round() as u8;
        }
    }
    frame
}

/// Build the blobs experiment: a `place(25) -> hidden(N) -> output(2)` network plus a synthesized
/// stimulus stream of jittered bumps, alternating classes for balance.
pub fn build(args: &Args) -> Result<Experiment, String> {
    let n_trials = args.trials.max(1);

    let input = InputSpace::identity("place", Shape::Grid2D { h: GRID, w: GRID });
    let effector = Effector::identity("quadrant", CENTERS.len() as u32);

    let mut builder = NetworkBuilder { populations: Vec::new(), connections: Vec::new() };
    let in_pop = builder.add(input_config(), input.n_neurons());
    let hid_pop = builder.add(hidden_config(), args.hidden);
    let out_pop = builder.add(output_config(), effector.n_neurons());
    builder.connect(in_pop, hid_pop, Compartment::Basal, ConnRule::FixedInDegree { k: args.fan_in_hidden });
    builder.connect(hid_pop, out_pop, Compartment::Basal, ConnRule::FixedInDegree { k: args.fan_in_output });

    let network = Network::build(builder);
    let input = input.bind(network.population_range(in_pop));
    let effector = effector.bind(network.population_range(out_pop));

    let dims: BTreeMap<String, u64> = BTreeMap::from([
        ("input_neurons".into(), input.n_neurons() as u64),
        ("hidden_neurons".into(), args.hidden as u64),
        ("output_neurons".into(), effector.n_neurons() as u64),
        ("neurons".into(), network.n_neurons() as u64),
        ("dendrites".into(), network.n_dendrites() as u64),
        ("synapses".into(), network.n_synapses() as u64),
    ]);

    // synthesize the stimulus stream: alternate classes for balance, jitter the point uniformly
    // around the class centre so each trial is a distinct sample of the same cluster. A dedicated
    // RNG (seed perturbed off the trial RNG) keeps stimulus generation independent of the sim.
    let mut rng = SmallRng::seed_from_u64(args.seed ^ 0x5EED_B10B);
    let mut frames = Vec::with_capacity(n_trials);
    let mut labels = Vec::with_capacity(n_trials);
    for t in 0..n_trials {
        let class = (t % CENTERS.len()) as u32;
        let (cx, cy) = CENTERS[class as usize];
        let jx = cx + rng.random_range(-SPREAD..SPREAD);
        let jy = cy + rng.random_range(-SPREAD..SPREAD);
        frames.push(bump(jx, jy));
        labels.push(class);
    }

    Ok(Experiment { network, input, effector, frames, labels, dims, snapshot_every_tick: true, label_prefix: "blobs" })
}
