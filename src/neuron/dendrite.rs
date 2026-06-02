use crate::constants::{X_DECAY, BASAL_DECAY, APICAL_DECAY, APICAL_DV_S, APICAL_SLOPE_K};
use crate::math::decay::{shift_decay, shift_decay_u8};
use crate::neuron::synapse::update_synapse_alpha;

// Compartment kind, used by the topology/builder to tag connections.
pub enum Compartment {
    Apical,
    Basal,
}

// What integrating a synapse signal did to its parent dendrite. The two compartments produce
// fundamentally different outputs, so the primitive returns the compartment-appropriate verdict
// and the handler just routes it into an event.
pub enum DendriteOutput {
    // Basal: hard threshold. `fired` true means V_B crossed threshold and was reset to 0 — the
    // caller should emit a DENDRITIC_SPIKE.
    Basal { fired: bool },
    // Apical: graded. `plateau` is the sigmoidal depolarization to deliver to the soma; V_B is
    // left intact (it leaks, it does not reset).
    Apical { plateau: i16 },
}

pub struct Dendrite {
    pub dendrite_activities: Vec<u16>,  // branch voltage V_B (basal AND apical integrate here)
    pub dendrite_last_events: Vec<u16>,
    pub dendrite_constants: Vec<i8>,    // basal branch constant (unused by the apical pathway)
    pub dendrite_thresholds: Vec<u16>,  // basal: hard spike threshold; apical: θ_B half-activation
    pub synapse_offsets: Vec<u32>,
    // fixed slot model, see docs
    pub live_synapse_counts: Vec<u8>, // number of synapse SLOTS that are active on this dendrite
    pub dendrite_to_neuron: Vec<u32>, // reverse map d -> owning neuron index (analytic d/D, stored for the event loop)
    // compartment flag: 0 = basal (hard-threshold dendritic spike), 1 = apical (graded sigmoid
    // plateau). u8 not bool so the buffer can be shared with GPU kernels (cf. event_type).
    pub dendrite_is_apical: Vec<u8>,
}

/// when a synapse receives a spike event, it must update the voltage of the parent dendrite
/// this updated function has asymmetric dynamics dependent on x and alpha
/// delta V_dendrite = w_i * (1 + gamma_i)
/// where gamma is the sum of alpha times an exponential decay of their distance param x
/// for only x_j > x_i (synapses higher along the dendrite have more influence on the soma)
/// this allows for ordered synaptic integration
/// The synapse arrays are slices (the gamma reduction loops over j > s_idx — the GPU-strided
/// read that justifies the slice). The dendrite-level state is a single touched element, so the
/// caller hands single refs already scoped to this dendrite. `is_apical` picks the leak constant
/// and the output kind; each caller knows its compartment statically. `dendrite_threshold` is the
/// basal hard threshold or, for apical, θ_B (the plateau half-activation).
///
/// This owns the dendrite's full local state machine: integrate, then either fire+reset (basal) or
/// produce the graded plateau (apical). See [`DendriteOutput`].
pub fn update_dendrite_activity(
    s_idx: usize, // which synapse triggered the update (slice-local)
    timestamp: u16,
    live_end: usize, // = base + live_count, computed by caller
    synapse_xs: &[u8],
    synapse_alphas: &mut [u8],
    synapse_weights: &[i8],
    synapse_last_events: &mut [u16],
    dendrite_activity: &mut u16,
    dendrite_last_event: &mut u16,
    dendrite_threshold: u16,
    is_apical: bool,
) -> DendriteOutput {
    let x_i = synapse_xs[s_idx];
    let w_i = synapse_weights[s_idx];

    let mut gamma: u16 = 0;

    // loop to calculate gamma
    // synapses must be ordered by increasing x
    for j in (s_idx + 1)..live_end {
        // lazy update decay for each synapse j > i to get current alpha_j, then calculate contribution to gamma
        let alpha_j = update_synapse_alpha(j, timestamp, synapse_alphas, synapse_last_events);
        let dx = synapse_xs[j] - x_i;
        // shift_decay_u8(alpha_j, dx, X_DECAY) ≈ alpha_j * exp(-dx / 2^X_DECAY)
        gamma = gamma.saturating_add(shift_decay_u8(alpha_j, dx as u16, X_DECAY) as u16);
    }
    // determine decay constant based on compartment
    let k = if is_apical { APICAL_DECAY } else { BASAL_DECAY };
    // decay (leak) the existing branch voltage since the last event, then add the new delta
    let elapsed = timestamp.wrapping_sub(*dendrite_last_event);
    *dendrite_last_event = timestamp;
    let decayed = shift_decay(*dendrite_activity, elapsed, k);
    // (1 + gamma); saturating so a huge gamma can't overflow i16 before the multiply
    let gain = 1i16.saturating_add(gamma.min(i16::MAX as u16 - 1) as i16);
    let update_term = (w_i as i16).saturating_mul(gain);
    *dendrite_activity = decayed.saturating_add_signed(update_term);

    if is_apical {
        // graded sigmoidal plateau (θ_B = dendrite_threshold); V_B is NOT reset — it leaks.
        let plateau = apical_plateau(*dendrite_activity, dendrite_threshold, APICAL_DV_S, APICAL_SLOPE_K);
        DendriteOutput::Apical { plateau }
    } else if *dendrite_activity >= dendrite_threshold {
        *dendrite_activity = 0; // hard reset after a basal dendritic spike
        DendriteOutput::Basal { fired: true }
    } else {
        DendriteOutput::Basal { fired: false }
    }
}

/// Apical dendritic transfer function (Payeur et al. 2021), adapted to the branch formalism:
///   σ^(ap)(V_B) = δV_S / (1 + exp(−κ(V_B − θ_B)))
/// where V_B is the apical branch voltage (integrated by `update_dendrite_activity`, exactly as
/// for basal), θ_B is the half-activation point, δV_S the plateau ceiling, and κ the slope.
///
/// Unlike a basal dendrite (hard threshold → discrete spike), the apical branch produces this
/// GRADED somatic depolarization. It is computed with the existing base-2 decay rather than a
/// real `exp`: the logistic core e^(−κ·) IS `shift_decay`, and the V_B < θ_B side uses the
/// logistic symmetry σ(−x) = 1 − σ(x). `k` sets the slope (κ = ln2 / 2^k): the scaled decay
/// D = 256·2^(−|V_B−θ_B|/2^k) halves every 2^k of distance from θ_B. Returns a value in [0, δV_S].
pub fn apical_plateau(v_b: u16, theta: u16, dv_s: i16, k: u8) -> i16 {
    const UNIT: i32 = 256;
    let u = (v_b as i32 - theta as i32).unsigned_abs() as u16; // |V_B − θ_B|
    let d = shift_decay(UNIT as u16, u, k) as i32; // D = 256·2^(−u/2^k) ∈ [0, 256]
    let dv = dv_s as i32;
    let out = if v_b >= theta {
        dv * UNIT / (UNIT + d) // upper half: σ ≥ 1/2  → output ∈ [δV_S/2, δV_S]
    } else {
        dv * d / (UNIT + d) // lower half: σ < 1/2  → output ∈ [0, δV_S/2]
    };
    // set ceiling of i16::MAX - (i8::MAX + 1) to avoid overflow when this is added to the soma potential in update_soma_potential
    out.min(i16::MAX as i32 - (i8::MAX as i32 + 1)) as i16
}

// binary search to find the dendrite index for a given synapse index
// synapse_offsets is a sorted array of the starting synapse index for each dendrite
// dendrite i owns synapses in the range [synapse_offsets[i], synapse_offsets[i+1])
pub fn synapse_to_dendrite(
    s_idx: usize,
    synapse_offsets: &[u32],
) -> usize {
    synapse_offsets.partition_point(|&o| o as usize <= s_idx) - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    // offsets [0, 3, 7, 12]: dendrite 0 owns s[0..3], dendrite 1 owns s[3..7], dendrite 2 owns s[7..12]
    const OFFSETS: [u32; 4] = [0, 3, 7, 12];

    #[test]
    fn synapse_to_dendrite_first_synapse_of_first_dendrite() {
        assert_eq!(synapse_to_dendrite(0, &OFFSETS), 0);
    }

    #[test]
    fn synapse_to_dendrite_last_synapse_of_first_dendrite() {
        assert_eq!(synapse_to_dendrite(2, &OFFSETS), 0);
    }

    #[test]
    fn synapse_to_dendrite_first_synapse_of_second_dendrite() {
        assert_eq!(synapse_to_dendrite(3, &OFFSETS), 1);
    }

    #[test]
    fn synapse_to_dendrite_first_synapse_of_third_dendrite() {
        assert_eq!(synapse_to_dendrite(7, &OFFSETS), 2);
    }

    #[test]
    fn synapse_to_dendrite_last_synapse_of_third_dendrite() {
        assert_eq!(synapse_to_dendrite(11, &OFFSETS), 2);
    }

    #[test]
    fn update_dendrite_activity_single_synapse_no_gamma() {
        // single synapse: gamma=0, delta = w_i * 1 = w_i. elapsed=0 → no leak, so activity == delta.
        let xs = [10u8];
        let mut alphas = [200u8];
        let weights = [7i8];
        let mut last_events = [0u16];
        let mut activity = 0u16;
        let mut d_last = 0u16;
        update_dendrite_activity(0, 0, xs.len(), &xs, &mut alphas, &weights, &mut last_events, &mut activity, &mut d_last, u16::MAX, false);
        assert_eq!(activity, 7);
    }

    #[test]
    fn update_dendrite_activity_last_synapse_has_no_neighbors() {
        // s_idx at end of slice → no j > s_idx, gamma=0
        let xs = [5u8, 10, 20];
        let mut alphas = [200u8; 3];
        let weights = [10i8, 5, 3];
        let mut last_events = [0u16; 3];
        let mut activity = 0u16;
        let mut d_last = 0u16;
        update_dendrite_activity(2, 0, xs.len(), &xs, &mut alphas, &weights, &mut last_events, &mut activity, &mut d_last, u16::MAX, false);
        assert_eq!(activity, 3); // 3 * (1 + 0)
    }

    #[test]
    fn update_dendrite_activity_active_neighbor_increases_delta() {
        // s_idx=0, x_i=5, w_i=10
        // j=1: alpha_j=200 (elapsed=0), dx=5
        //   shift_decay_u8(200, 5, X_DECAY=4): shifts=0, rem=5, drop=(100*5)>>4=31, result=169
        // gamma=169, delta = 10 * (1+169) = 1700
        let xs = [5u8, 10];
        let mut alphas = [50u8, 200];
        let weights = [10i8, 5];
        let mut last_events = [0u16; 2];
        let mut activity = 0u16;
        let mut d_last = 0u16;
        update_dendrite_activity(0, 0, xs.len(), &xs, &mut alphas, &weights, &mut last_events, &mut activity, &mut d_last, u16::MAX, false);
        assert_eq!(activity, 1700);
    }

    #[test]
    fn update_dendrite_activity_live_end_excludes_dead_tail() {
        // 3 slots but only 2 are live (live_end=2). The distal slot at index 2 has high alpha
        // but is a dead/unbound slot and must NOT contribute to gamma.
        // s_idx=0, x_i=5, w_i=10; only j=1 is in range: alpha=200, dx=5 → shift_decay_u8(200,5,4)=169
        // gamma=169, delta = 10 * (1+169) = 1700 — identical to the 2-live-synapse case above.
        let xs = [5u8, 10, 20];
        let mut alphas = [50u8, 200, 255];
        let weights = [10i8, 5, 3];
        let mut last_events = [0u16; 3];
        let mut activity = 0u16;
        let mut d_last = 0u16;
        update_dendrite_activity(0, 0, 2, &xs, &mut alphas, &weights, &mut last_events, &mut activity, &mut d_last, u16::MAX, false);
        assert_eq!(activity, 1700); // dead slot 2 ignored despite alpha=255
    }

    #[test]
    fn update_dendrite_activity_leaks_prior_voltage_before_adding() {
        // prior branch voltage 1024, basal leak half-life = 2^BASAL_DECAY=1024 ticks.
        // elapsed = 1024 → one half-life → 1024 → 512, then + w_i*(1+0) = 512 + 10 = 522.
        let xs = [10u8];
        let mut alphas = [0u8];
        let weights = [10i8];
        let mut last_events = [0u16];
        let mut activity = 1024u16;
        let mut d_last = 0u16;
        update_dendrite_activity(0, 1024, xs.len(), &xs, &mut alphas, &weights, &mut last_events, &mut activity, &mut d_last, u16::MAX, false);
        assert_eq!(activity, 522);
        assert_eq!(d_last, 1024);
    }

    #[test]
    fn update_dendrite_activity_basal_fires_and_resets_on_threshold() {
        // V_B starts at 95, threshold 100; single synapse w=10 → 95 + 10 = 105 >= 100 → fire + reset.
        let xs = [10u8];
        let mut alphas = [0u8];
        let weights = [10i8];
        let mut last_events = [0u16];
        let mut activity = 95u16;
        let mut d_last = 0u16;
        let out = update_dendrite_activity(0, 0, xs.len(), &xs, &mut alphas, &weights, &mut last_events, &mut activity, &mut d_last, 100, false);
        assert!(matches!(out, DendriteOutput::Basal { fired: true }));
        assert_eq!(activity, 0); // reset after firing
    }

    #[test]
    fn update_dendrite_activity_apical_returns_plateau_without_reset() {
        // apical: V_B = 0 + 10 = 10; θ_B = 0 → upper half → plateau = apical_plateau(10,0,64,9) = 32.
        // V_B is left intact (leaks, no reset).
        let xs = [10u8];
        let mut alphas = [0u8];
        let weights = [10i8];
        let mut last_events = [0u16];
        let mut activity = 0u16;
        let mut d_last = 0u16;
        let out = update_dendrite_activity(0, 0, xs.len(), &xs, &mut alphas, &weights, &mut last_events, &mut activity, &mut d_last, 0, true);
        match out {
            DendriteOutput::Apical { plateau } => assert_eq!(plateau, 32),
            _ => panic!("expected apical output"),
        }
        assert_eq!(activity, 10); // not reset
    }

    // --- apical_plateau (sigmoidal transfer; dv_s=64, k=9, θ_B=1000) ---

    #[test]
    fn apical_plateau_midpoint_is_half() {
        // V_B == θ_B → σ = 1/2 → δV_S/2
        assert_eq!(apical_plateau(1000, 1000, 64, 9), 32);
    }

    #[test]
    fn apical_plateau_saturates_high() {
        // V_B far above θ_B → D → 0 → σ → 1 → δV_S
        assert_eq!(apical_plateau(60000, 1000, 64, 9), 64);
    }

    #[test]
    fn apical_plateau_monotonic_increasing() {
        // sampled a half-life (2^9=512) either side of θ_B
        let lo = apical_plateau(1000 - 512, 1000, 64, 9);
        let mid = apical_plateau(1000, 1000, 64, 9);
        let hi = apical_plateau(1000 + 512, 1000, 64, 9);
        assert!(lo < mid && mid < hi, "expected {lo} < {mid} < {hi}");
    }

    #[test]
    fn apical_plateau_symmetric_about_theta() {
        // σ(θ+u) + σ(θ−u) ≈ δV_S (within integer rounding)
        let above = apical_plateau(1000 + 512, 1000, 64, 9) as i32;
        let below = apical_plateau(1000 - 512, 1000, 64, 9) as i32;
        assert!((above + below - 64).abs() <= 1, "{above} + {below} not ≈ 64");
    }
}
