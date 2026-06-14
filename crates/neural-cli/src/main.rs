//! Headless recording generator for the research dashboard.
//!
//! Builds a spiking network, presents a stimulus stream through the `neural-sim` trial harness with
//! a [`RecordingSink`], and writes one `.ntr` recording per trial (event trace + a pre/post keyframe
//! pair) into the output directory for the dashboard to replay. Forward-pass only for now —
//! unsupervised plasticity (BaP/gamma) still evolves the weights across trials; supervised feedback
//! (docs/08-mnist-pipeline.md §8.5) is the next step.
//!
//! The trial loop here is experiment-agnostic: it drives whatever [`Experiment`] it is handed.
//! Everything specific to the MNIST task — the idx reader, the hidden-cell type, the input→output
//! wiring — lives in [`mnist`]; argument parsing lives in [`args`].

mod args;
mod mnist;

use std::collections::BTreeMap;
use std::process::ExitCode;

use rand::SeedableRng;
use rand::rngs::SmallRng;

use neural_sim::io::{Effector, InputSpace};
use neural_sim::network::Network;
use neural_sim::network::event::EventQueue;
use neural_sim::telemetry::TelemetrySink;
use neural_sim::trial::{TrialConfig, run_trial};
use neural_telemetry::{Manifest, RecordingSink};

/// Per-trial queue capacity (events). A fresh ring is allocated per trial for clean isolation; this
/// must comfortably exceed the peak in-flight wavefront size (sustained input volley × fan-out
/// across a few pipeline generations). 2^18 ≈ 262k slots ≈ 3 MB — ample for the default topology.
const QUEUE_CAPACITY: usize = 1 << 18;

/// A built network ready to drive: the bound IO boundary, the per-trial stimulus stream, and the
/// shape metadata stamped into every manifest. Experiment-agnostic — [`mnist::build`] is one
/// producer; another task would supply its own `frames`/`labels` and wiring the same way.
pub struct Experiment {
    /// The wired, built network (learning state persists across trials by design).
    pub network: Network,
    /// Afferent boundary, bound to the input-population neuron range.
    pub input: InputSpace,
    /// Efferent boundary, bound to the output-population neuron range; argmaxes the prediction.
    pub effector: Effector,
    /// One stimulus frame per trial, in presentation order.
    pub frames: Vec<Vec<u8>>,
    /// Ground-truth class per trial, parallel to `frames`.
    pub labels: Vec<u32>,
    /// Network shape (`input_neurons`, `hidden_neurons`, …) recorded into each manifest for repro.
    pub dims: BTreeMap<String, u64>,
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
    let args = match args::parse_args()? {
        Some(a) => a,
        None => {
            println!("{}", args::USAGE);
            return Ok(());
        }
    };

    let experiment = mnist::build(&args)?;
    run_experiment(experiment, &args)
}

/// Drive `experiment` for one recording per trial: pre-trial keyframe → forward (or supervised)
/// pass → post-trial keyframe, written to `args.out/trial-NNNNN.{ntr,ntr.json}`. The monotonic
/// clock and learning state carry across trials; only the transient dynamics are reset between them.
fn run_experiment(experiment: Experiment, args: &args::Args) -> Result<(), String> {
    let Experiment { mut network, input, effector, frames, labels, dims } = experiment;
    let n_trials = frames.len();

    std::fs::create_dir_all(&args.out).map_err(|e| format!("creating {}: {e}", args.out.display()))?;

    let constants = constants_map();
    let mut rng = SmallRng::seed_from_u64(args.seed);
    let mut clock: u16 = 0;
    let mut spike_counts = vec![0u32; network.n_neurons()];
    let cfg = TrialConfig { ticks: args.ticks, window: args.window, teach_strength: args.teach_strength };
    let mut correct = 0usize;

    eprintln!(
        "network: {} input -> {} hidden -> {} output ({} neurons, {} dendrites, {} synapse slots){}",
        dims["input_neurons"], dims["hidden_neurons"], dims["output_neurons"],
        dims["neurons"], dims["dendrites"], dims["synapses"],
        if args.train { format!(" — TRAINING (teach_strength={})", args.teach_strength) } else { String::new() },
    );

    for t in 0..n_trials {
        let frame = &frames[t];
        let label = labels[t];

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

        // training trials drive the correct output neuron (supervised LTP); eval trials don't.
        let teach = if args.train { Some(label) } else { None };
        let prediction = run_trial(
            &mut network, &queue, &input, &effector, frame, teach, &mut clock, cfg,
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

    let pct = 100.0 * correct as f64 / n_trials as f64;
    if args.train {
        eprintln!(
            "done: {n_trials} training trials ({correct}/{n_trials} matched the teacher — \
             NOT accuracy; re-run without --train to evaluate) — recordings in {}",
            args.out.display()
        );
    } else {
        eprintln!(
            "done: {n_trials} trials, {correct} correct ({pct:.1}%) — recordings in {}",
            args.out.display()
        );
    }
    Ok(())
}
