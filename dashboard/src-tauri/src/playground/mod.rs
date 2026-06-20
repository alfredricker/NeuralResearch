//! Playground pillar: a live `neural-sim` network the webview builds, stimulates, and steps one
//! wavefront at a time — the interactive replacement for `.ntr` replay.
//!
//! The backend holds a single live [`Session`] (the network + its event queue + a monotonic clock)
//! in Tauri-managed state. Commands are deliberately narrow: build from a [`NetworkSpec`], poke a
//! synapse, advance one wavefront, read state. All projection of the engine's private SoA into
//! drawable shapes happens in [`anatomy`]/[`frame`]; the engine's layout never crosses the boundary.

pub mod anatomy;
pub mod frame;

use std::sync::Mutex;

use neural_sim::network::Network;
use neural_sim::network::event::EventQueue;
use neural_telemetry::spec::NetworkSpec;

use anatomy::NeuronAnatomy;
use frame::{CaptureSink, NetworkFrame};

/// One live network under inspection. Rebuilt by `pg_build`; everything else operates on it.
pub struct Session {
    net: Network,
    queue: EventQueue,
    /// Monotonic sim time. Advanced one tick per `step`, exactly as the trial harness does, so the
    /// lazy decay (alpha/beta/voltage, all keyed off timestamp deltas) sees real elapsed time.
    clock: u16,
    /// Per-neuron AP accumulator the event loop writes into (length == n_neurons).
    spikes: Vec<u32>,
}

/// Tauri-managed handle: at most one session at a time (`None` until the first build).
pub type PlaygroundState = Mutex<Option<Session>>;

/// Lock the session and run `f`, or return a uniform "no session" error if none is built yet.
fn with_session<T>(
    state: &PlaygroundState,
    f: impl FnOnce(&mut Session) -> T,
) -> Result<T, String> {
    let mut guard = state.lock().map_err(|_| "playground state poisoned".to_string())?;
    let session = guard.as_mut().ok_or("no network built yet — call pg_build first")?;
    Ok(f(session))
}

/// Build a live network from a spec and make it the current session. Returns its static anatomy
/// (sent once; the per-tick dynamic state comes from `pg_step`/`pg_state`).
#[tauri::command]
pub fn pg_build(spec: NetworkSpec, state: tauri::State<'_, PlaygroundState>) -> Result<Vec<NeuronAnatomy>, String> {
    let net = spec.build().map_err(|e| format!("building network: {e}"))?;
    // Ring sized generously past any single playground wavefront; events recycle their slots.
    let queue = EventQueue::new((net.n_synapses() * 2).max(4096).next_power_of_two());
    let spikes = vec![0u32; net.n_neurons()];
    let anatomy = anatomy::gather(&net.topology(), &net.edges());

    *state.lock().map_err(|_| "playground state poisoned".to_string())? =
        Some(Session { net, queue, clock: 0, spikes });
    Ok(anatomy)
}

/// Inject one AP delivery onto a synapse slot at the current clock (drained by the next `pg_step`).
#[tauri::command]
pub fn pg_stimulate(synapse: u32, burst: i16, state: tauri::State<'_, PlaygroundState>) -> Result<(), String> {
    with_session(&state, |s| {
        if (synapse as usize) < s.net.n_synapses() {
            s.net.stimulate(&s.queue, synapse, burst, s.clock);
            Ok(())
        } else {
            Err(format!("synapse {synapse} out of range ({} slots)", s.net.n_synapses()))
        }
    })?
}

/// Advance exactly one wavefront and return the resulting state (with this step's firings flagged).
#[tauri::command]
pub fn pg_step(state: tauri::State<'_, PlaygroundState>) -> Result<NetworkFrame, String> {
    with_session(&state, |s| {
        let mut cap = CaptureSink::default();
        s.net.step(&s.queue, &mut cap, &mut s.spikes);
        let f = frame::gather(&s.net.topology(), &s.net.view(s.clock, &s.spikes), &cap, s.clock);
        s.clock = s.clock.wrapping_add(1);
        f
    })
}

/// Current state without stepping (e.g. the initial render right after build). No firings flagged.
#[tauri::command]
pub fn pg_state(state: tauri::State<'_, PlaygroundState>) -> Result<NetworkFrame, String> {
    with_session(&state, |s| {
        let cap = CaptureSink::default();
        frame::gather(&s.net.topology(), &s.net.view(s.clock, &s.spikes), &cap, s.clock)
    })
}

/// Clear transient dynamics (potentials/branch voltages) back to rest, keeping learned weights.
#[tauri::command]
pub fn pg_reset(state: tauri::State<'_, PlaygroundState>) -> Result<(), String> {
    with_session(&state, |s| s.net.reset_dynamics())
}
