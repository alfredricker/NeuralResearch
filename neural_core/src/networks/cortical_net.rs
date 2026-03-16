use crate::region::CorticalRegion;

/// A hierarchy of CorticalRegions.
///
/// Regions at the same level are independent and tick in parallel via rayon.
/// Connectivity between levels: output of level L → feed-in of level L+1.
pub struct CorticalNet {
    /// levels[l] = slice of regions at level l
    pub levels: Vec<Vec<CorticalRegion>>,
    /// Read-head: linear projection from top-level outputs to n_classes
    pub read_w: Vec<f32>,  // n_classes × top_n_model
    pub n_classes: usize,
}

impl CorticalNet {
    /// Build a 2-level cortical network for classification.
    ///
    /// level 0: one region (n_l0 model neurons, input = n_input)
    /// level 1: one region (n_l1 model neurons, input = n_l0)
    pub fn new_two_level(
        n_input: usize,
        n_l0: usize,
        n_l1: usize,
        n_classes: usize,
        rng: &mut impl rand::Rng,
    ) -> Self {
        let grid_specs_l0: Vec<(usize, i32)> = vec![(5, 1), (7, 2), (11, 3)];
        let grid_specs_l1: Vec<(usize, i32)> = vec![(13, 2), (17, 3)];

        let r0 = CorticalRegion::new(n_l0, n_input, &grid_specs_l0, 0.05, rng);
        let r1 = CorticalRegion::new(n_l1, n_l0, &grid_specs_l1, 0.05, rng);

        let scale = (2.0 / n_l1 as f32).sqrt();
        let read_w: Vec<f32> = (0..n_classes * n_l1)
            .map(|_| rng.gen::<f32>() * scale - scale / 2.0)
            .collect();

        Self {
            levels: vec![vec![r0], vec![r1]],
            read_w,
            n_classes,
        }
    }

    /// Forward pass: return class logits.
    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        // Level 0
        let feed = input.to_vec();
        for r in &mut self.levels[0] {
            r.tick(&feed);
        }

        // Collect level-0 outputs (concatenate region outputs)
        let l0_out: Vec<f32> = self.levels[0]
            .iter()
            .flat_map(|r| r.output().to_vec())
            .collect();

        // Level 1 (parallel across regions — single region here but structure is general)
        // For parallel: each region at level 1 takes the full l0_out
        // (For a real multi-region level, slice l0_out per region)
        for r in &mut self.levels[1] {
            let inp: Vec<f32> = l0_out.iter().take(r.n_input).cloned().collect();
            r.tick(&inp);
        }

        // Top-level output
        let top_out: Vec<f32> = self.levels.last().unwrap()
            .iter()
            .flat_map(|r| r.output().to_vec())
            .collect();

        // Linear read-out: logits[c] = sum_i read_w[c * n_top + i] * top_out[i]
        let n_top = top_out.len();
        (0..self.n_classes)
            .map(|c| {
                (0..n_top.min(self.levels.last().unwrap()[0].n_model))
                    .map(|i| self.read_w[c * self.levels.last().unwrap()[0].n_model + i] * top_out[i])
                    .sum::<f32>()
            })
            .collect()
    }

    /// Predict class (argmax of logits).
    pub fn predict(&mut self, input: &[f32]) -> usize {
        let logits = self.forward(input);
        logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Hebbian learning step after a forward pass.
    pub fn learn(&mut self, input: &[f32]) {
        let feed = input.to_vec();
        for r in &mut self.levels[0] {
            r.learn_ff(&feed);
            r.learn_rr();
        }
        let l0_out: Vec<f32> = self.levels[0]
            .iter()
            .flat_map(|r| r.output().to_vec())
            .collect();
        for r in &mut self.levels[1] {
            let inp: Vec<f32> = l0_out.iter().take(r.n_input).cloned().collect();
            r.learn_ff(&inp);
            r.learn_rr();
        }
    }

    /// Update read-out weights with perceptron-style delta rule.
    pub fn learn_readout(&mut self, label: usize, lr: f32) {
        let top_out: Vec<f32> = self.levels.last().unwrap()
            .iter()
            .flat_map(|r| r.output().to_vec())
            .collect();
        let n_top = self.levels.last().unwrap()[0].n_model;

        let logits: Vec<f32> = (0..self.n_classes)
            .map(|c| (0..n_top).map(|i| self.read_w[c * n_top + i] * top_out[i]).sum::<f32>())
            .collect();

        // Softmax for delta computation
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
        let sum_exp: f32 = exp.iter().sum();
        let probs: Vec<f32> = exp.iter().map(|&e| e / sum_exp).collect();

        for c in 0..self.n_classes {
            let delta = if c == label { probs[c] - 1.0 } else { probs[c] };
            for i in 0..n_top {
                self.read_w[c * n_top + i] -= lr * delta * top_out[i];
            }
        }
    }
}
