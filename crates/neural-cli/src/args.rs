use std::path::PathBuf;

pub struct Args {
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
            train: false,
            teach_strength: 24,
        }
    }
}

pub const USAGE: &str = "\
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
    --train                  supervised: drive the correct output each tick (§8.5 Option 1)
    --teach-strength <V>     teacher voltage/tick     [default: 24]
    -h, --help               print this help

MNIST files must be gunzip'd idx-ubyte (not .gz).

With --train, each trial drives the labelled output neuron to burst so its afferent
hidden->output weights undergo LTP. NOTE: the teacher inflates the output spike counts, so the
per-trial prediction printed under --train is teacher-contaminated, not a measure of accuracy —
re-run the same recordings without --train to read honest predictions.";

pub fn parse_args() -> Result<Option<Args>, String> {
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
            "--train" => a.train = true,
            "--teach-strength" => a.teach_strength = val()?.parse().map_err(|e| format!("--teach-strength: {e}"))?,
            other => return Err(format!("unknown argument: {other} (try --help)")),
        }
    }
    Ok(Some(a))
}