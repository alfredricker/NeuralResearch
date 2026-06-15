//! Everything specific to the MNIST experiment: the idx-ubyte reader and the
//! `input(784) -> hidden(N) -> output(10)` network it drives.
//!
//! The idx parser is hand-rolled (no external crates) and expects **decompressed** files: the
//! canonical downloads are gzip'd (`*-ubyte.gz`); `gunzip` them first. [`build`] is the one entry
//! point `main` calls — it loads the dataset, wires the network, and hands back a general
//! [`crate::Experiment`] the experiment-agnostic trial loop knows how to run.

use std::collections::BTreeMap;
use std::path::Path;

use neural_sim::io::{Effector, InputSpace, Shape, input_config, output_config};
use neural_sim::math::sample::{SamplerI8, SamplerU8};
use neural_sim::network::Network;
use neural_sim::network::build::NetworkBuilder;
use neural_sim::network::topology::conn::ConnRule;
use neural_sim::neuron::config::NeuronConfig;
use neural_sim::neuron::dendrite::Compartment;

use crate::Experiment;
use crate::args::Args;

/// A loaded idx3 image set: `images[i]` is `rows * cols` pixels, row-major, intensity `0..=255`.
pub struct MnistImages {
    pub rows: usize,
    pub cols: usize,
    pub images: Vec<Vec<u8>>,
}

fn read_u32_be(bytes: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// Load an `idx3-ubyte` image file (magic `0x00000803`).
pub fn load_images(path: impl AsRef<Path>) -> Result<MnistImages, String> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    if bytes.len() < 16 {
        return Err(format!("{}: too short to be an idx3 image file", path.display()));
    }
    let magic = read_u32_be(&bytes, 0);
    if magic != 0x0000_0803 {
        return Err(format!(
            "{}: bad idx3 magic {magic:#010x} (expected 0x00000803 — is the file still gzip'd? gunzip it first)",
            path.display()
        ));
    }
    let n = read_u32_be(&bytes, 4) as usize;
    let rows = read_u32_be(&bytes, 8) as usize;
    let cols = read_u32_be(&bytes, 12) as usize;
    let stride = rows * cols;
    let expected = 16 + n * stride;
    if bytes.len() < expected {
        return Err(format!(
            "{}: truncated — header declares {n} images of {rows}x{cols} ({expected} bytes), file is {} bytes",
            path.display(),
            bytes.len()
        ));
    }
    let images = (0..n)
        .map(|i| {
            let off = 16 + i * stride;
            bytes[off..off + stride].to_vec()
        })
        .collect();
    Ok(MnistImages { rows, cols, images })
}

/// Load an `idx1-ubyte` label file (magic `0x00000801`); one `u8` digit per image.
pub fn load_labels(path: impl AsRef<Path>) -> Result<Vec<u8>, String> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    if bytes.len() < 8 {
        return Err(format!("{}: too short to be an idx1 label file", path.display()));
    }
    let magic = read_u32_be(&bytes, 0);
    if magic != 0x0000_0801 {
        return Err(format!(
            "{}: bad idx1 magic {magic:#010x} (expected 0x00000801 — is the file still gzip'd? gunzip it first)",
            path.display()
        ));
    }
    let n = read_u32_be(&bytes, 4) as usize;
    if bytes.len() < 8 + n {
        return Err(format!(
            "{}: truncated — header declares {n} labels, file has {} after the header",
            path.display(),
            bytes.len() - 8
        ));
    }
    Ok(bytes[8..8 + n].to_vec())
}

/// The hidden-layer neuron type — a `visual_mnist`-spirit pyramidal cell: several basal dendrites
/// receiving the pixel projection, no apical compartment yet (Option-1 feedback is a later step).
/// `Box::leak` yields the `&'static` the builder requires; called once, so the leak is intentional.
fn hidden_config() -> &'static NeuronConfig {
    Box::leak(Box::new(NeuronConfig::new(
        "hidden",
        6,                       // n_basal_dendrites — branches receiving the pixel projection
        None,                    // n_apical_dendrites — none yet (no apical feedback first pass)
        SamplerU8::new(128, 50), // synapse_x_sampler — spread positions along each dendrite
        SamplerU8::new(1, 0),    // dendrites_per_branch — 1 → 6 basal dendrites/neuron
        SamplerU8::new(16, 0),   // synapses_per_dendrite — ~16 live slots
        20,                      // soma_threshold
        500,                     // basal_dendrite_threshold
        SamplerI8::new(40, 10),  // basal_dendrite_constant — proximal
        None,                    // apical_dendrite_threshold
        None,                    // apical_dendrite_constant
        neural_sim::constants::MSLR as i16, // learning_rate
    )))
}

/// Load MNIST and wire the `input(784) -> hidden(N) -> output(10)` network for it, returning the
/// general [`Experiment`] the trial loop drives. Frames/labels are truncated to the trials actually
/// runnable (`args.trials` capped by the smaller of the image/label counts).
pub fn build(args: &Args) -> Result<Experiment, String> {
    let images = load_images(&args.images)?;
    let labels = load_labels(&args.labels)?;
    if images.images.is_empty() {
        return Err("dataset contains no images".into());
    }
    if images.rows * images.cols != 784 {
        return Err(format!(
            "expected 28x28 frames, got {}x{} — is this MNIST?",
            images.rows, images.cols
        ));
    }
    let n_avail = images.images.len().min(labels.len());
    let n_trials = args.trials.min(n_avail);
    if n_trials == 0 {
        return Err("no labeled images available to run".into());
    }

    // --- build input(784) -> hidden(N) -> output(10) ---
    let input = InputSpace::identity("pixels", Shape::Grid2D { h: images.rows as u32, w: images.cols as u32 });
    let effector = Effector::identity("digits", 10);

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

    // present each frame at most once, in dataset order, capped to the runnable trial count.
    let frames: Vec<Vec<u8>> = images.images.into_iter().take(n_trials).collect();
    let labels: Vec<u32> = labels.into_iter().take(n_trials).map(u32::from).collect();

    // MNIST is far too large to keep a full snapshot per tick — replay animates from the event
    // trace and the pre/post keyframes instead.
    Ok(Experiment { network, input, effector, frames, labels, dims, snapshot_every_tick: false, label_prefix: "mnist" })
}
