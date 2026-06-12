pub mod math;
pub mod constants;
pub mod network;
pub mod neuron;
pub mod io; // network <-> world boundary: input spaces (afferent) and effectors (efferent)
pub mod telemetry; // observer seam: TelemetrySink/NullSink + borrowed NetworkView (no serde here)
pub mod trial; // trial harness: per-tick encode -> step -> advance clock, reading out a prediction
// pub mod gpu; uncomment when starting to write gpu code