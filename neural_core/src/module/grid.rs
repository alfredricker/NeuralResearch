/// GridModule: one place-cell ring from Section 2.3.1 of Cortical.tex.
///
/// n neurons arranged in a cyclic ring (Ring(1) topology, hardwired).
/// Phase φ(t) ∈ ℤ/nℤ advances by δ_μ · displacement(t) mod n.
/// The active neuron is the one at index φ(t).
#[derive(Clone, Debug)]
pub struct GridModule {
    pub n: usize,        // ring size (period)
    pub phase: usize,    // φ(t): current active index ∈ 0..n
    pub delta: i32,      // δ_μ: step size (coprime to n for full coverage)
}

impl GridModule {
    pub fn new(n: usize, delta: i32) -> Self {
        assert!(n > 0);
        Self { n, phase: 0, delta }
    }

    /// Advance phase by displacement * δ_μ (mod n).
    pub fn advance(&mut self, displacement: i32) {
        let n = self.n as i32;
        let step = (self.delta * displacement).rem_euclid(n);
        self.phase = (self.phase as i32 + step).rem_euclid(n) as usize;
    }

    /// One-hot activation vector: 1.0 at φ(t), 0.0 elsewhere.
    pub fn activations(&self) -> Vec<f32> {
        let mut v = vec![0.0f32; self.n];
        v[self.phase] = 1.0;
        v
    }

    /// Write activations into a slice (offset into a shared NodeArray).
    pub fn write_activations(&self, buf: &mut [f32]) {
        assert_eq!(buf.len(), self.n);
        for x in buf.iter_mut() { *x = 0.0; }
        buf[self.phase] = 1.0;
    }
}

/// A bank of L GridModules with coprime periods.
///
/// CRT theorem: L modules with coprime periods n_1, …, n_L uniquely encode
/// positions 0..∏n_μ using only ∑n_μ neurons.
pub struct GridBank {
    pub modules: Vec<GridModule>,
}

impl GridBank {
    /// Construct L modules with the given (period, delta) pairs.
    /// Caller is responsible for choosing coprime periods.
    pub fn new(specs: &[(usize, i32)]) -> Self {
        Self {
            modules: specs.iter().map(|&(n, d)| GridModule::new(n, d)).collect(),
        }
    }

    pub fn total_neurons(&self) -> usize {
        self.modules.iter().map(|m| m.n).collect::<Vec<_>>().iter().sum()
    }

    /// Advance all modules by the same displacement.
    pub fn advance(&mut self, displacement: i32) {
        for m in &mut self.modules {
            m.advance(displacement);
        }
    }

    /// Concatenated one-hot activations across all modules.
    pub fn activations(&self) -> Vec<f32> {
        self.modules.iter().flat_map(|m| m.activations()).collect()
    }

    /// Number of unique positions encodable (product of periods).
    pub fn capacity(&self) -> usize {
        self.modules.iter().map(|m| m.n).product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_advance_wraps() {
        let mut m = GridModule::new(5, 2);
        m.advance(3); // phase = (0 + 2*3) % 5 = 1
        assert_eq!(m.phase, 1);
        m.advance(3); // phase = (1 + 6) % 5 = 2
        assert_eq!(m.phase, 2);
    }

    #[test]
    fn one_hot_activations() {
        let m = GridModule::new(4, 1);
        let a = m.activations();
        assert_eq!(a[0], 1.0);
        assert_eq!(a.iter().sum::<f32>(), 1.0);
    }

    #[test]
    fn crt_capacity() {
        // Coprime periods 3, 5, 7 → 105 positions, 15 neurons
        let bank = GridBank::new(&[(3, 1), (5, 1), (7, 1)]);
        assert_eq!(bank.capacity(), 105);
        assert_eq!(bank.total_neurons(), 15);
    }

    #[test]
    fn full_coverage() {
        // A module with delta coprime to n should visit all positions
        let n = 7;
        let mut m = GridModule::new(n, 3); // gcd(3,7)=1
        let mut visited = vec![false; n];
        for _ in 0..n {
            visited[m.phase] = true;
            m.advance(1);
        }
        assert!(visited.iter().all(|&v| v));
    }
}
