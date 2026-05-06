pub const T_BETA: u16 = 500; // number of ticks elapse to subtract 1 from beta
pub const H_ALPHA: u8 = 30; // minimum activity of a synapse to enact weight updates
pub const H_BETA: i16 = 4; // i16 to avoid conversions when calculating delta_weight
pub const ALPHA_DECAY: u8 = 8; // alpha decay term. Halves every 2^8 = 256 ticks, so active synapses can maintain high alpha for hundreds of ticks after a spike.
pub const X_DECAY: u8 = 4; // x decay term for dendritic integration. Halves every 2^4 = 16 x units

// MINIMUM SYNAPTIC LEARNING RATE
// burst_term_max = 2^6-5, alpha_max = 2^8 - 1
// we want max(btm*am / slr) = 127 
pub const MSLR: u16 = 120;

pub const ALPHA_BOOST: u8 = 64; // alpha added to a synapse when it receives a forward AP