pub const T_BETA: u16 = 500; // number of ticks elapse to subtract 1 from beta
pub const H_ALPHA: u8 = 30; // minimum activity of a synapse to enact weight updates
pub const H_BETA: i16 = 4; // i16 to avoid conversions when calculating delta_weight

// decay constants for the bit shift decay function
pub const ALPHA_DECAY: u8 = 11; // alpha decay term. Halves every 2^11 = 2048 ticks.
pub const X_DECAY: u8 = 4; // x decay term for dendritic integration. Halves every 2^4 = 16 x units
pub const BASAL_DECAY: u8 = 9; // basal dendrite voltage decay. Halves every 2^9 = 512 ticks.
pub const APICAL_DECAY: u8 = 11; // apical dendrite voltage decay. Halves every 2^11 = 2048 ticks.
pub const SOMATIC_DECAY: u8 = 10; // soma voltage decay. Halves every 2^10 = 1024 ticks.

pub const SOMA_V_RESET: i8 = -32; // reset potential for the soma after a spike; also the minimum potential it can take

// MINIMUM SYNAPTIC LEARNING RATE
// burst_term_max = 2^6-5, alpha_max = 2^8 - 1
// we want max(btm*am / slr) = 127 
pub const MSLR: u16 = 120;

pub const ALPHA_BOOST: u8 = 64; // alpha added to a synapse when it receives a forward AP

// Fixed synapse-slot capacity per dendrite (the uniform analytic stride S, so
// synapse_offsets[d] = d * S). live_synapse_counts[d] holds the actual bound count <= S.
// NOTE: at u8::MAX this over-provisions heavily (~16x for a 16-live-synapse dendrite);
// tune here if memory matters.
pub const SYNAPSE_SLOTS_PER_DENDRITE: usize = u8::MAX as usize;

// --- apical compartment (Payeur et al. 2021 sigmoidal plateau transfer function) ---
// σ^(ap)(V_B) = δV_S / (1 + exp(−κ(V_B − θ_B))). θ_B reuses dendrite_thresholds; the rest:
// VALUES UNTUNED (placeholders, like H_BETA) — calibrate once the feedback path runs.
pub const APICAL_DV_S: i16 = 64; // δV_S: max plateau depolarization delivered to the soma
pub const APICAL_SLOPE_K: u8 = 9; // sigmoid slope; κ = ln2 / 2^k. D halves every 2^k of |V_B − θ_B|
