use rand::{RngExt,Rng};

use super::Network;
use crate::network::topology::conn::Connection;
use crate::neuron::population::Population;

pub fn build_network<R: Rng + RngExt>(
    populations: &[Population],
    connections: &[Connection],
    rng: &mut R,
) -> Network {
    // --- soma arrays ---
    let mut soma_potentials = Vec::new();
    let mut soma_thresholds = Vec::new();

    // --- dendrite arrays ---
    let mut dendrite_activities: Vec<u16> = Vec::new();
    let mut dendrite_last_events: Vec<u16> = Vec::new();
    let mut dendrite_constants: Vec<i8> = Vec::new();
    let mut dendrite_thresholds: Vec<u16> = Vec::new();
    let mut synapse_offsets: Vec<u32> = Vec::new();
    let mut live_synapse_counts: Vec<u8> = Vec::new();
    let mut dendrite_to_neuron: Vec<u32> = Vec::new();
    
    // --- synapse arrays ---
    let mut synapse_weights: Vec<i8> = Vec::new();
    let mut synapse_x: Vec<u8> = Vec::new();
    let mut synapse_alphas: Vec<u8> = Vec::new();
    let mut synapse_last_events: Vec<u16> = Vec::new();

}