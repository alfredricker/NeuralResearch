/// Compressed Sparse Row graph with per-edge weights.
///
/// Layout:
///   offsets[i]..offsets[i+1] → edge indices for node i
///   targets[k] → destination node of edge k
///   weights[k] → weight of edge k
#[derive(Clone, Debug)]
pub struct CsrGraph {
    pub node_count: usize,
    pub offsets: Vec<u32>,  // length node_count + 1
    pub targets: Vec<u32>,  // length edge_count
    pub weights: Vec<f32>,  // length edge_count
}

impl CsrGraph {
    /// Iterate over (target, weight) pairs for node `i`.
    #[inline]
    pub fn neighbors(&self, i: usize) -> impl Iterator<Item = (u32, f32)> + '_ {
        let start = self.offsets[i] as usize;
        let end = self.offsets[i + 1] as usize;
        self.targets[start..end]
            .iter()
            .zip(&self.weights[start..end])
            .map(|(&t, &w)| (t, w))
    }

    pub fn edge_count(&self) -> usize {
        self.targets.len()
    }

    /// Mutably access weight of edge k.
    #[inline]
    pub fn weight_mut(&mut self, k: usize) -> &mut f32 {
        &mut self.weights[k]
    }

    /// Edge index range for node i.
    #[inline]
    pub fn edge_range(&self, i: usize) -> std::ops::Range<usize> {
        self.offsets[i] as usize..self.offsets[i + 1] as usize
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::builder::GraphBuilder;

    #[test]
    fn csr_neighbors() {
        // 3-node fully connected (self-loops excluded)
        let g = GraphBuilder::new(3).all(1.0).build();
        assert_eq!(g.node_count, 3);
        assert_eq!(g.edge_count(), 6); // 3*2 directed edges
        let ns: Vec<_> = g.neighbors(0).collect();
        assert_eq!(ns.len(), 2);
        let targets: Vec<u32> = ns.iter().map(|&(t, _)| t).collect();
        assert!(targets.contains(&1));
        assert!(targets.contains(&2));
    }
}
