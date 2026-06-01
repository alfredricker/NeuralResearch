use rand::RngExt;

use super::Network;
use crate::constants::SYNAPSE_SLOTS_PER_DENDRITE;

pub fn build_network<R: Rng + RngExt>(
    populations: &[Population],
    rng: &mut R,
) -> Network {
    // --- soma arrays ---
    let mut soma_potentials = Vec::new();
    let mut soma_thresholds = Vec::new();

}