use crate::neuron::dendrite::{Compartment, Dendrite};
use crate::neuron::synapse::Synapse;
use rand::RngExt;
use thiserror::Error;

pub struct Connection {
    pub from: u32,
    pub to: u32,
    pub compartment: Compartment,
    pub rule: ConnRule,
}

impl Connection {
    pub fn new(from: u32, to: u32, compartment: Compartment, rule: ConnRule) -> Self {
        Self {from, to, compartment, rule}
    }
}

pub enum ConnRule {
    DenseRandom { p: f32 }, // each possible connection is made with probability p
    FixedInDegree { k: u32 }, // each neuron receives exactly k connections from the source population
    ReceptiveField { radius: u32 }, // each neuron receives connections from source neurons within a certain radius
    Topographic { patch: u8 }, // each neuron receives connections from a patch of source neurons (e.g. 3x3)
    OneToOne, // each neuron receives a connection from the corresponding neuron in the source population (only for populations of the same size)
}

impl ConnRule {
    pub fn apply(
        &self, 
        src: &[u32], // global neuron indices of the source population 
        dst: &[u32], // global neuron indices of the target population 
        rng: &mut impl RngExt,
        edges: &mut Vec<(u32, u32)>, // accumuulator: (src_neuron, dst_neuron)
    ) -> Result<(), ConnError> {
        // apply the connection rule to determine the synapse offsets for each neuron in the target population
        // this will involve random sampling for some rules, so we may need to pass in a random number generator
        match self {
            ConnRule::DenseRandom { p } => {
                for &d in dst {
                    for &s in src {
                        if rng.random::<f32>() < *p {
                            edges.push((s, d));
                        }
                    }
                }
            }
            ConnRule::FixedInDegree { k } => {
                // each dst neuron receives exactly k connections from distinct sources
                let k = (*k as usize).min(src.len());
                let mut pool = src.to_vec();
                for &d in dst {
                    // partial Fischer-Yates: first k entries become a random sample
                    for i in 0..k {
                        let j = rng.random_range(i..pool.len());
                        pool.swap(i,j);
                        edges.push((pool[i], d));
                    }
                }
            }
            ConnRule::OneToOne => {
                if src.len() != dst.len() {
                    return Err(ConnError::InvalidRule);
                }
                for (&s, &d) in src.iter().zip(dst) {
                    edges.push((s, d));
                }
            }

            // Spatial rules assume both populations are laid out on a √N × √N grid.
            ConnRule::ReceptiveField { radius } => {
                let side = (src.len() as f64).sqrt() as i32; // e.g. 28 for MNIST
                let r = *radius as i32;
                for (di, &d) in dst.iter().enumerate() {
                    let (dr, dc) = (di as i32 / side, di as i32 % side);
                    for (si, &s) in src.iter().enumerate() {
                        let (sr, sc) = (si as i32 / side, si as i32 % side);
                        if (sr - dr).abs() <= r && (sc - dc).abs() <= r {
                            edges.push((s, d));
                        }
                    }
                }
            }

            ConnRule::Topographic { patch } => {
                // same idea as ReceptiveField but a fixed patch×patch window
                // centered on the topographically-matched source neuron.
                let _ = patch;
                return Err(ConnError::InvalidRule); // TODO
            }
        }
        Ok(())

    }
}


#[derive(Error, Debug)]
pub enum ConnError {
    #[error("No connection rules provided")]
    NoConnections,
    #[error("Invalid connection rule provided")]
    InvalidRule,
}