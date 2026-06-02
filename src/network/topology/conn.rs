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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    const SEED: u64 = 42;

    // --- OneToOne ---

    #[test]
    fn one_to_one_zips_equal_sizes() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let mut edges = Vec::new();
        ConnRule::OneToOne
            .apply(&[10, 11, 12], &[20, 21, 22], &mut rng, &mut edges)
            .unwrap();
        assert_eq!(edges, vec![(10, 20), (11, 21), (12, 22)]);
    }

    #[test]
    fn one_to_one_size_mismatch_errors() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let mut edges = Vec::new();
        let r = ConnRule::OneToOne.apply(&[0, 1], &[0], &mut rng, &mut edges);
        assert!(matches!(r, Err(ConnError::InvalidRule)));
        assert!(edges.is_empty());
    }

    // --- DenseRandom (deterministic at the probability extremes) ---

    #[test]
    fn dense_random_p_one_connects_all_pairs() {
        // random::<f32>() ∈ [0,1) is ALWAYS < 1.0 → fully connected, seed-independent
        let mut rng = SmallRng::seed_from_u64(SEED);
        let mut edges = Vec::new();
        ConnRule::DenseRandom { p: 1.0 }
            .apply(&[0, 1], &[2, 3], &mut rng, &mut edges)
            .unwrap();
        assert_eq!(edges.len(), 4);
        assert!(edges.contains(&(0, 2)));
        assert!(edges.contains(&(1, 3)));
    }

    #[test]
    fn dense_random_p_zero_connects_nothing() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let mut edges = Vec::new();
        ConnRule::DenseRandom { p: 0.0 }
            .apply(&[0, 1], &[2, 3], &mut rng, &mut edges)
            .unwrap();
        assert!(edges.is_empty());
    }

    // --- FixedInDegree ---

    #[test]
    fn fixed_in_degree_exact_count_and_distinct_sources() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let mut edges = Vec::new();
        let dst = [100, 101, 102];
        ConnRule::FixedInDegree { k: 2 }
            .apply(&[0, 1, 2, 3], &dst, &mut rng, &mut edges)
            .unwrap();
        for &d in &dst {
            let sources: Vec<u32> = edges.iter().filter(|e| e.1 == d).map(|e| e.0).collect();
            assert_eq!(sources.len(), 2, "each dst receives exactly k");
            assert_ne!(sources[0], sources[1], "sources are distinct per dst");
        }
    }

    #[test]
    fn fixed_in_degree_clamps_k_to_source_count() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let mut edges = Vec::new();
        ConnRule::FixedInDegree { k: 10 } // only 2 sources available
            .apply(&[0, 1], &[100], &mut rng, &mut edges)
            .unwrap();
        assert_eq!(edges.len(), 2);
    }

    // --- ReceptiveField (spatial; assumes a √N × √N grid, position == index) ---

    #[test]
    fn receptive_field_radius_zero_is_one_to_one() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let grid: Vec<u32> = (0..9).collect(); // 3×3
        let mut edges = Vec::new();
        ConnRule::ReceptiveField { radius: 0 }
            .apply(&grid, &grid, &mut rng, &mut edges)
            .unwrap();
        assert_eq!(edges.len(), 9);
        assert!(edges.iter().all(|&(s, d)| s == d));
    }

    #[test]
    fn receptive_field_corner_has_smaller_fan_in_than_center() {
        // radius 1 on a 3×3 grid: corner (0,0) sees a 2×2 window; center (1,1) sees all 9.
        let mut rng = SmallRng::seed_from_u64(SEED);
        let grid: Vec<u32> = (0..9).collect();
        let mut edges = Vec::new();
        ConnRule::ReceptiveField { radius: 1 }
            .apply(&grid, &grid, &mut rng, &mut edges)
            .unwrap();
        let fan_in = |d: u32| edges.iter().filter(|e| e.1 == d).count();
        assert_eq!(fan_in(0), 4); // corner
        assert_eq!(fan_in(4), 9); // center
    }
}