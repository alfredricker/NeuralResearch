pub const T_BETA: u16 = 500; // number of ticks elapse to subtract 1 from beta
pub const H_ALPHA: u8 = 30; // minimum activity of a synapse to enact weight updates
pub const H_BETA: i16 = 4; // i16 to avoid conversions when calculating delta_weight
pub const ALPHA_DECAY: u8 = 8; // alpha decay term. This is exponential

// MINIMUM SYNAPTIC LEARNING RATE
// burst_term_max = 2^6-5, alpha_max = 2^8 - 1
// we want max(btm*am / slr) = 127 
pub const MSLR: u16 = 120;