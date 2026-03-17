/// How multiple incoming wires to the same destination port are combined.
#[derive(Clone, Debug, PartialEq)]
pub enum Aggregation {
    /// dst.dim == sum of all src dims. Incoming slices are concatenated.
    Concat,
    /// All src dims must equal dst.dim. Incoming slices are element-wise summed.
    Sum,
}

/// Declares a single input or output port on a node.
#[derive(Clone, Debug)]
pub struct PortSpec {
    pub name: &'static str,
    pub dim: usize,
    pub agg: Aggregation,
}

/// Named slice container passed to `Node::tick` / `Node::learn`.
/// Indexed by port name; order matches the node's declared port list.
pub struct PortValues {
    specs: Vec<PortSpec>,
    data: Vec<Vec<f32>>,
}

impl PortValues {
    pub fn zeros_from(specs: &[PortSpec]) -> Self {
        let data = specs.iter().map(|s| vec![0.0f32; s.dim]).collect();
        Self { specs: specs.to_vec(), data }
    }

    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.specs
            .iter()
            .position(|s| s.name == name)
            .map(|i| self.data[i].as_slice())
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut Vec<f32>> {
        let idx = self.specs.iter().position(|s| s.name == name)?;
        Some(&mut self.data[idx])
    }

    /// Number of ports.
    pub fn len(&self) -> usize {
        self.specs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    /// Access spec + data by index (used internally by flatten/graph).
    pub fn by_index(&self, i: usize) -> (&PortSpec, &[f32]) {
        (&self.specs[i], &self.data[i])
    }

    pub fn by_index_mut(&mut self, i: usize) -> (&PortSpec, &mut Vec<f32>) {
        (&self.specs[i], &mut self.data[i])
    }

    /// Zero all data buffers.
    pub fn zero_all(&mut self) {
        for v in &mut self.data {
            v.iter_mut().for_each(|x| *x = 0.0);
        }
    }

    pub fn specs(&self) -> &[PortSpec] {
        &self.specs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_specs() -> Vec<PortSpec> {
        vec![
            PortSpec { name: "a", dim: 4, agg: Aggregation::Concat },
            PortSpec { name: "b", dim: 2, agg: Aggregation::Sum },
        ]
    }

    #[test]
    fn zeros_from_correct_dims() {
        let pv = PortValues::zeros_from(&make_specs());
        assert_eq!(pv.get("a").unwrap().len(), 4);
        assert_eq!(pv.get("b").unwrap().len(), 2);
        assert!(pv.get("a").unwrap().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn get_mut_writes() {
        let mut pv = PortValues::zeros_from(&make_specs());
        pv.get_mut("a").unwrap()[0] = 1.5;
        assert_eq!(pv.get("a").unwrap()[0], 1.5);
    }

    #[test]
    fn missing_name_returns_none() {
        let pv = PortValues::zeros_from(&make_specs());
        assert!(pv.get("nope").is_none());
    }
}
