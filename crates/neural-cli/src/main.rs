//! Headless recording generator for the research dashboard.
//!
//! STUB. The real entry point will: build a `Network`, bind an `InputSpace`/`Effector`, run N
//! trials through `run_event_loop` with a `RecordingSink`, and write one `.ntr` per trial into
//! `recordings/`. That requires the trial harness, which is the next build step. For now this
//! just links `neural-sim` + `neural-telemetry` so the workspace wiring is proven end-to-end.

use neural_telemetry::{Manifest, RecordingSink};

fn main() {
    // Demonstrates the recording seam compiles and links. No simulation is run yet.
    let _sink = RecordingSink::new(Manifest {
        label: "stub".to_string(),
        ..Default::default()
    });
    eprintln!("neural-cli: stub — trial harness not implemented yet (see plans/dashboard-architecture.md, Phase 1)");
}
