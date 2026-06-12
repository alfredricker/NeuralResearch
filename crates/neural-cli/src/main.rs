//! Headless recording generator for the research dashboard.
//!
//! Builds an `input(784) -> hidden(N) -> output(10)` spiking network, presents MNIST digits through
//! the `neural-sim` trial harness with a [`RecordingSink`], and writes one `.ntr` recording per
//! trial (event trace + a pre/post keyframe pair) into the output directory for the dashboard to
//! replay. Forward-pass only for now — unsupervised plasticity (BaP/gamma) still evolves the
//! weights across trials; supervised feedback (docs/08-mnist-pipeline.md §8.5) is the next step.

mod mnist;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use rand::SeedableRng;
use rand::rngs::SmallRng;

use neural_sim::io::{Effector, InputSpace, Shape, input_config, output_config};
use neural_sim::math::sample::{SamplerI8, SamplerU8};
use neural_sim::network::Network;
use neural_sim::network::build::NetworkBuilder;
use neural_sim::network::event::EventQueue;
use neural_sim::network::topology::conn::ConnRule;
use neural_sim::neuron::config::NeuronConfig;
use neural_sim::neuron::dendrite::Compartment;
use neural_sim::telemetry::TelemetrySink;
use neural_sim::trial::{TrialConfig, run_trial};
use neural_telemetry::{Manifest, RecordingSink};

/// Per-trial queue capacity (events). A fresh ring is allocated per trial for clean isolation; this
/// must comfortably exceed the peak in-flight wavefront size (sustained input volley × fan-out
/// across a few pipeline generations). 2^18 ≈ 262k slots ≈ 3 MB — ample for the default topology.
const QUEUE_CAPACITY: usize = 1 << 18;

struct Args {
    images: PathBuf,
    labels: PathBuf,
    out: PathBuf,
    trials: usize,
    hidden: u32,
    ticks: u16,
    window: u16,
    fan_in_hidden: u32,
    fan_in_output: u32,
    seed: u64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            images: PathBuf::from("data/train-images-idx3-ubyte"),
            labels: PathBuf::from("data/train-labels-idx1-ubyte"),
            out: PathBuf::from("recordings"),
            trials: 20,
            hidden: 200,
            ticks: 100,
            window: 8,
            fan_in_hidden: 64,
            fan_in_output: 32,
            seed: 0xC0FFEE,
        }
    }
}

const USAGE: &str = "\
neural-cli — generate .ntr recordings from MNIST trials

USAGE:
    neural-cli [OPTIONS]

OPTIONS:
    --images <PATH>          idx3-ubyte image file   [default: data/train-images-idx3-ubyte]
    --labels <PATH>          idx1-ubyte label file   [default: data/train-labels-idx1-ubyte]
    --out <DIR>              recordings output dir   [default: recordings]
    --trials <N>             number of trials        [default: 20]
    --hidden <N>             hidden-layer neurons     [default: 200]
    --ticks <N>              wavefronts per trial     [default: 100]
    --window <N>             input jitter window      [default: 8]
    --fan-in-hidden <K>      pixels -> each hidden    [default: 64]
    --fan-in-output <K>      hidden -> each output    [default: 32]
    --seed <N>               RNG seed                 [default: 12648430]
    -h, --help               print this help

MNIST files must be gunzip'd idx-ubyte (not .gz).";

fn parse_args() -> Result<Option<Args>, String> {
    let mut a = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag}: missing value"));
        match flag.as_str() {
            "-h" | "--help" => return Ok(None),
            "--images" => a.images = PathBuf::from(val()?),
            "--labels" => a.labels = PathBuf::from(val()?),
            "--out" => a.out = PathBuf::from(val()?),
            "--trials" => a.trials = val()?.parse().map_err(|e| format!("--trials: {e}"))?,
            "--hidden" => a.hidden = val()?.parse().map_err(|e| format!("--hidden: {e}"))?,
            "--ticks" => a.ticks = val()?.parse().map_err(|e| format!("--ticks: {e}"))?,
            "--window" => a.window = val()?.parse().map_err(|e| format!("--window: {e}"))?,
            "--fan-in-hidden" => a.fan_in_hidden = val()?.parse().map_err(|e| format!("--fan-in-hidden: {e}"))?,
            "--fan-in-output" => a.fan_in_output = val()?.parse().map_err(|e| format!("--fan-in-output: {e}"))?,
            "--seed" => a.seed = val()?.parse().map_err(|e| format!("--seed: {e}"))?,
            other => return Err(format!("unknown argument: {other} (try --help)")),
        }
    }
    Ok(Some(a))
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

/// The `constants.rs` values this run used, name → value, recorded in every manifest for repro.
fn constants_map() -> BTreeMap<String, i64> {
    use neural_sim::constants as c;
    BTreeMap::from([
        ("T_BETA".into(), c::T_BETA as i64),
        ("H_ALPHA".into(), c::H_ALPHA as i64),
        ("H_BETA".into(), c::H_BETA as i64),
        ("ALPHA_DECAY".into(), c::ALPHA_DECAY as i64),
        ("X_DECAY".into(), c::X_DECAY as i64),
        ("BASAL_DECAY".into(), c::BASAL_DECAY as i64),
        ("APICAL_DECAY".into(), c::APICAL_DECAY as i64),
        ("SOMATIC_DECAY".into(), c::SOMATIC_DECAY as i64),
        ("MSLR".into(), c::MSLR as i64),
        ("ALPHA_BOOST".into(), c::ALPHA_BOOST as i64),
    ])
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("neural-cli: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = match parse_args()? {
        Some(a) => a,
        None => {
            println!("{USAGE}");
            return Ok(());
        }
    };

    let images = mnist::load_images(&args.images)?;
    let labels = mnist::load_labels(&args.labels)?;
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

    let mut network = Network::build(builder);
    let input = input.bind(network.population_range(in_pop));
    let effector = effector.bind(network.population_range(out_pop));

    std::fs::create_dir_all(&args.out).map_err(|e| format!("creating {}: {e}", args.out.display()))?;

    let dims: BTreeMap<String, u64> = BTreeMap::from([
        ("input_neurons".into(), input.n_neurons() as u64),
        ("hidden_neurons".into(), args.hidden as u64),
        ("output_neurons".into(), effector.n_neurons() as u64),
        ("neurons".into(), network.n_neurons() as u64),
        ("dendrites".into(), network.n_dendrites() as u64),
        ("synapses".into(), network.n_synapses() as u64),
    ]);
    let constants = constants_map();

    let mut rng = SmallRng::seed_from_u64(args.seed);
    let mut clock: u16 = 0;
    let mut spike_counts = vec![0u32; network.n_neurons()];
    let cfg = TrialConfig { ticks: args.ticks, window: args.window };
    let mut correct = 0usize;

    eprintln!(
        "network: {} input -> {} hidden -> {} output ({} neurons, {} dendrites, {} synapse slots)",
        input.n_neurons(), args.hidden, effector.n_neurons(),
        network.n_neurons(), network.n_dendrites(), network.n_synapses(),
    );

    for t in 0..n_trials {
        let frame = &images.images[t];
        let label = labels[t] as u32;

        let mut sink = RecordingSink::new(Manifest {
            label: format!("trial-{t:05}"),
            dims: dims.clone(),
            constants: constants.clone(),
            keyframe_offsets: Vec::new(),
            true_label: Some(label),
            prediction: None,
        });

        // fresh ring per trial → clean isolation (head/tail from 0, no cross-trial event bleed).
        let queue = EventQueue::new(QUEUE_CAPACITY);

        // pre-trial keyframe: weights before this trial's plasticity, counts zeroed.
        spike_counts.iter_mut().for_each(|c| *c = 0);
        sink.on_snapshot(&network.view(clock, &spike_counts));

        let prediction = run_trial(
            &mut network, &queue, &input, &effector, frame, &mut clock, cfg,
            &mut rng, &mut sink, &mut spike_counts,
        );

        // post-trial keyframe: final weights + accumulated activity.
        sink.on_snapshot(&network.view(clock, &spike_counts));
        sink.manifest_mut().prediction = prediction;

        if prediction == Some(label) {
            correct += 1;
        }

        let stem = args.out.join(format!("trial-{t:05}"));
        sink.write(&stem).map_err(|e| format!("writing {}: {e}", stem.display()))?;

        // isolate the next trial; learning state (weights/alpha/beta) persists by design.
        network.reset_dynamics();

        eprintln!("trial {t:05}: label={label} prediction={prediction:?}");
    }

    eprintln!(
        "done: {n_trials} trials, {correct} correct ({:.1}%) — recordings in {}",
        100.0 * correct as f64 / n_trials as f64,
        args.out.display()
    );
    Ok(())
}
