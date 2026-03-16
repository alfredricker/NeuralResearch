/// Packed storage for a population of neurons.
///
/// All arrays are indexed 0..n. The `active_set` is kept sparse:
/// only nodes with |α| > θ are listed, so tick() is O(|A|·deg).
pub struct NodeArray {
    pub activations: Vec<f32>,   // α(t)
    pub input_buffer: Vec<f32>,  // accumulates f_h(t) before update
    pub active_set: Vec<u32>,    // indices where |α| > θ
    pub threshold: f32,          // θ (sparsity threshold)
}

impl NodeArray {
    pub fn new(n: usize, threshold: f32) -> Self {
        Self {
            activations: vec![0.0; n],
            input_buffer: vec![0.0; n],
            active_set: Vec::new(),
            threshold,
        }
    }

    pub fn len(&self) -> usize {
        self.activations.len()
    }

    /// Zero the input buffer (call before accumulating messages).
    pub fn clear_inputs(&mut self) {
        for x in &mut self.input_buffer {
            *x = 0.0;
        }
    }

    /// Rebuild the active set based on current activations.
    pub fn rebuild_active_set(&mut self) {
        self.active_set.clear();
        for (i, &a) in self.activations.iter().enumerate() {
            if a.abs() > self.threshold {
                self.active_set.push(i as u32);
            }
        }
    }

    /// Apply a per-node update function: α_i ← f(α_i, input_i).
    pub fn apply_update<F: Fn(f32, f32) -> f32>(&mut self, f: F) {
        for (a, b) in self.activations.iter_mut().zip(self.input_buffer.iter()) {
            *a = f(*a, *b);
        }
        self.rebuild_active_set();
    }

    /// Sparsity: fraction of nodes in the active set.
    pub fn sparsity(&self) -> f32 {
        self.active_set.len() as f32 / self.activations.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_set_tracking() {
        let mut nodes = NodeArray::new(5, 0.1);
        nodes.activations[1] = 0.5;
        nodes.activations[3] = -0.3;
        nodes.rebuild_active_set();
        assert_eq!(nodes.active_set.len(), 2);
        assert!(nodes.active_set.contains(&1));
        assert!(nodes.active_set.contains(&3));
    }

    #[test]
    fn clear_inputs() {
        let mut nodes = NodeArray::new(4, 0.0);
        nodes.input_buffer[2] = 99.0;
        nodes.clear_inputs();
        assert!(nodes.input_buffer.iter().all(|&x| x == 0.0));
    }
}
