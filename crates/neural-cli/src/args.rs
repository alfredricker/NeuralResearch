use std::path::PathBuf;

/// Which experiment to build and record. Selects the network topology and stimulus generator;
/// everything downstream (the trial loop, the recording format) is task-agnostic.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Task {
    /// 28×28 MNIST digits → `input(784) -> hidden(N) -> output(10)`. Large; replays from events.
    Mnist,
    /// 5×5 place-cell 2D blobs, quadrant classification → a tiny net the dashboard renders whole,
    /// with a full state snapshot per tick for membrane-potential scrubbing.
    Blobs,
}

pub struct Args {
    pub task: Task,
    pub images: PathBuf,
    pub labels: PathBuf,
    pub out: PathBuf,
    pub trials: usize,
    pub hidden: u32,
    pub ticks: u16,
    pub window: u16,
    pub fan_in_hidden: u32,
    pub fan_in_output: u32,
    pub seed: u64,
    pub train: bool,
    pub teach_strength: i16,
}

impl Default for Args {
    /// MNIST defaults (the original task). [`Args::for_task`] swaps in smaller topology defaults
    /// for the blobs task before per-flag overrides are applied.
    fn default() -> Self {
        Self {
            task: Task::Mnist,
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
            train: false,
            teach_strength: 24,
        }
    }
}

impl Args {
    /// Base defaults for `task`, before command-line overrides. Blobs is a small synthetic task, so
    /// it wires a much smaller network and a fan-in that fits its 25-pixel input / 16 hidden units.
    fn for_task(task: Task) -> Self {
        let mut a = Args { task, ..Default::default() };
        if task == Task::Blobs {
            a.trials = 40;
            a.hidden = 16;
            a.ticks = 60;
            a.fan_in_hidden = 12; // each hidden unit samples ~12 of the 25 place cells
            a.fan_in_output = 8; // each output samples 8 of the 16 hidden units
            a.images = PathBuf::new(); // synthetic — no dataset files
            a.labels = PathBuf::new();
        }
        a
    }
}

pub const USAGE: &str = "\
neural-cli — generate .ntr recordings from classification trials

USAGE:
    neural-cli [OPTIONS]

OPTIONS:
    --task <NAME>            experiment: mnist | blobs   [default: mnist]
    --images <PATH>          idx3-ubyte image file   [default: data/train-images-idx3-ubyte]
    --labels <PATH>          idx1-ubyte label file   [default: data/train-labels-idx1-ubyte]
    --out <DIR>              recordings output dir   [default: recordings]
    --trials <N>             number of trials        [default: 20 mnist / 40 blobs]
    --hidden <N>             hidden-layer neurons     [default: 200 mnist / 16 blobs]
    --ticks <N>              wavefronts per trial     [default: 100 mnist / 60 blobs]
    --window <N>             input jitter window      [default: 8]
    --fan-in-hidden <K>      inputs -> each hidden    [default: 64 mnist / 12 blobs]
    --fan-in-output <K>      hidden -> each output    [default: 32 mnist / 8 blobs]
    --seed <N>               RNG seed                 [default: 12648430]
    --train                  supervised: drive the correct output each tick (§8.5 Option 1)
    --teach-strength <V>     teacher voltage/tick     [default: 24]
    -h, --help               print this help

TASKS:
    mnist   28x28 digits -> input(784) -> hidden(N) -> output(10). MNIST idx files must be
            gunzip'd idx-ubyte (not .gz). Replays from the event trace (too big to snapshot/tick).
    blobs   5x5 place-cell 2D blobs, NE-vs-SW quadrant classification -> a tiny net rendered whole
            in the dashboard, with a full state snapshot every tick for potential scrubbing.

With --train, each trial drives the labelled output neuron to burst so its afferent
hidden->output weights undergo LTP. NOTE: the teacher inflates the output spike counts, so the
per-trial prediction printed under --train is teacher-contaminated, not a measure of accuracy —
re-run the same recordings without --train to read honest predictions.";

pub fn parse_args() -> Result<Option<Args>, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.iter().any(|a| a == "-h" || a == "--help") {
        return Ok(None);
    }

    // resolve the task first so per-task defaults are in place before any overrides apply.
    let task = match raw.iter().position(|a| a == "--task") {
        Some(i) => match raw.get(i + 1).map(String::as_str) {
            Some("mnist") => Task::Mnist,
            Some("blobs") => Task::Blobs,
            Some(other) => return Err(format!("--task: unknown task '{other}' (expected mnist|blobs)")),
            None => return Err("--task: missing value".into()),
        },
        None => Task::Mnist,
    };

    let mut a = Args::for_task(task);
    let mut it = raw.into_iter();
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag}: missing value"));
        match flag.as_str() {
            "--task" => {
                val()?; // already resolved above; consume its value
            }
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
            "--train" => a.train = true,
            "--teach-strength" => a.teach_strength = val()?.parse().map_err(|e| format!("--teach-strength: {e}"))?,
            other => return Err(format!("unknown argument: {other} (try --help)")),
        }
    }
    Ok(Some(a))
}
