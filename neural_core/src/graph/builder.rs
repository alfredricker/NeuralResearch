use rand::Rng;
use super::csr::CsrGraph;

/// Builds a CsrGraph from edge lists.
pub struct GraphBuilder {
    n: usize,
    edges: Vec<(u32, u32, f32)>,  // (src, dst, weight)
}

impl GraphBuilder {
    pub fn new(n: usize) -> Self {
        Self { n, edges: Vec::new() }
    }

    /// Add a directed edge.
    pub fn add_edge(mut self, src: u32, dst: u32, weight: f32) -> Self {
        self.edges.push((src, dst, weight));
        self
    }

    /// Fully connected (no self-loops), uniform weight.
    pub fn all(mut self, weight: f32) -> Self {
        for i in 0..self.n as u32 {
            for j in 0..self.n as u32 {
                if i != j {
                    self.edges.push((i, j, weight));
                }
            }
        }
        self
    }

    /// Identity: each node connects only to itself.
    pub fn identity(mut self, weight: f32) -> Self {
        for i in 0..self.n as u32 {
            self.edges.push((i, i, weight));
        }
        self
    }

    /// Ring of degree k on each side (circulant with offsets 1..=k).
    pub fn ring(mut self, k: usize, weight: f32) -> Self {
        let n = self.n as u32;
        for i in 0..n {
            for d in 1..=k as u32 {
                let j = (i + d) % n;
                let jb = (i + n - d) % n;
                self.edges.push((i, j, weight));
                if jb != j {
                    self.edges.push((i, jb, weight));
                }
            }
        }
        self
    }

    /// Circulant graph with explicit offset set (directed, positive offsets only).
    pub fn circulant(mut self, offsets: &[usize], weight: f32) -> Self {
        let n = self.n as u32;
        for i in 0..n {
            for &d in offsets {
                let j = (i + d as u32) % n;
                self.edges.push((i, j, weight));
            }
        }
        self
    }

    /// Erdos-Renyi sparse random graph: each directed edge exists with probability p.
    pub fn sparse<R: Rng>(mut self, p: f32, weight_init: f32, rng: &mut R) -> Self {
        for i in 0..self.n as u32 {
            for j in 0..self.n as u32 {
                if i != j && rng.gen::<f32>() < p {
                    self.edges.push((i, j, weight_init));
                }
            }
        }
        self
    }

    /// Consume builder and produce a CsrGraph.
    pub fn build(mut self) -> CsrGraph {
        // Sort by source node for CSR construction
        self.edges.sort_unstable_by_key(|&(s, t, _)| (s, t));

        let mut offsets = vec![0u32; self.n + 1];
        for &(s, _, _) in &self.edges {
            offsets[s as usize + 1] += 1;
        }
        for i in 0..self.n {
            offsets[i + 1] += offsets[i];
        }

        let targets: Vec<u32> = self.edges.iter().map(|&(_, t, _)| t).collect();
        let weights: Vec<f32> = self.edges.iter().map(|&(_, _, w)| w).collect();

        CsrGraph { node_count: self.n, offsets, targets, weights }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_topology() {
        let g = GraphBuilder::new(6).ring(1, 1.0).build();
        // Each node connects to 2 neighbors
        for i in 0..6 {
            assert_eq!(g.neighbors(i).count(), 2);
        }
    }

    #[test]
    fn circulant_topology() {
        let n = 10;
        let g = GraphBuilder::new(n).circulant(&[1, 3], 1.0).build();
        for i in 0..n {
            assert_eq!(g.neighbors(i).count(), 2);
        }
    }

    #[test]
    fn sparse_graph() {
        let mut rng = rand::thread_rng();
        let g = GraphBuilder::new(20).sparse(0.5, 0.1, &mut rng).build();
        assert!(g.edge_count() > 0);
        assert!(g.edge_count() < 20 * 19); // not fully connected
    }
}
