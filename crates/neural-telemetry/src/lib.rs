//! Persistence layer for the dashboard. Serde lives here so `neural-sim` stays
//! serialization-free (keeping the GPU port clean) — the same boundary the engine's
//! `TelemetrySink` trait draws, but on the *output* side.
//!
//! Currently holds [`spec::NetworkSpec`]: the reproducible network *recipe* the playground
//! saves to `networks/<name>.json` and rebuilds from. A spec + its seed reconstructs a
//! bit-identical untrained network, so we persist kilobytes of human-editable JSON instead of
//! dumping the built SoA arrays.

pub mod spec;
