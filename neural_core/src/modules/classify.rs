use crate::subgraph::{Aggregation, Node, PortSpec, PortValues};

/// Aggregates vote vectors from multiple patch subgraphs and emits a
/// normalized prediction along with a scalar enzyme (uncertainty) signal.
///
/// ```text
/// votes (n_classes, Sum)  →  L1-normalize  →  pred (n_classes)
///                                          →  enzyme = 1 − max(pred)
/// ```
///
/// The enzyme signal is high when the network is uncertain (max pred ≈ 1/n)
/// and low when it is confident (max pred → 1).  It is fed back recurrently
/// to gate Hebbian learning in upstream `WhereModule`s.
///
/// **No learnable parameters** — `learn` is a no-op.
pub struct ClassifyModule {
    pub n_classes: usize,
    /// Current normalized prediction.
    pub pred: Vec<f32>,

    input_specs: Vec<PortSpec>,
    output_specs: Vec<PortSpec>,
}

impl ClassifyModule {
    pub fn new(n_classes: usize) -> Self {
        Self {
            n_classes,
            pred: vec![0.0; n_classes],
            input_specs: vec![
                PortSpec { name: "votes",  dim: n_classes, agg: Aggregation::Sum    },
            ],
            output_specs: vec![
                PortSpec { name: "pred",   dim: n_classes, agg: Aggregation::Concat },
                PortSpec { name: "enzyme", dim: 1,         agg: Aggregation::Concat },
            ],
        }
    }
}

impl Node for ClassifyModule {
    fn input_ports(&self) -> &[PortSpec] {
        &self.input_specs
    }

    fn output_ports(&self) -> &[PortSpec] {
        &self.output_specs
    }

    fn tick(&mut self, inputs: &PortValues, outputs: &mut PortValues) {
        let votes = inputs.get("votes").expect("ClassifyModule: missing 'votes' port");

        // L1-normalize positive votes → pred (treats negative votes as 0).
        let total: f32 = votes.iter().map(|&v| v.max(0.0)).sum();
        if total > 1e-9 {
            for i in 0..self.n_classes {
                self.pred[i] = votes[i].max(0.0) / total;
            }
        } else {
            // Uniform prior when no signal is present.
            let uniform = 1.0 / self.n_classes as f32;
            self.pred.fill(uniform);
        }

        // enzyme = 1 − max(pred): high uncertainty → high enzyme → fast learning.
        let max_pred = self.pred.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let enzyme = 1.0 - max_pred;

        outputs
            .get_mut("pred")
            .expect("ClassifyModule: missing 'pred' port")
            .copy_from_slice(&self.pred);
        outputs
            .get_mut("enzyme")
            .expect("ClassifyModule: missing 'enzyme' port")[0] = enzyme;
    }

    /// No-op — ClassifyModule has no learnable parameters.
    fn learn(&mut self, _inputs: &PortValues) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pred_sums_to_one() {
        let mut cm = ClassifyModule::new(4);
        let mut inputs = PortValues::zeros_from(cm.input_ports());
        inputs.get_mut("votes").unwrap().copy_from_slice(&[2.0, 1.0, 0.0, 3.0]);
        let mut outputs = PortValues::zeros_from(cm.output_ports());

        cm.tick(&inputs, &mut outputs);

        let pred = outputs.get("pred").unwrap();
        let sum: f32 = pred.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn enzyme_high_when_uncertain() {
        let mut cm = ClassifyModule::new(4);
        let mut inputs = PortValues::zeros_from(cm.input_ports());
        // Equal votes → uniform pred → enzyme near 1 − 0.25 = 0.75
        inputs.get_mut("votes").unwrap().copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
        let mut outputs = PortValues::zeros_from(cm.output_ports());

        cm.tick(&inputs, &mut outputs);

        let enzyme = outputs.get("enzyme").unwrap()[0];
        assert!((enzyme - 0.75).abs() < 1e-5);
    }

    #[test]
    fn enzyme_low_when_confident() {
        let mut cm = ClassifyModule::new(4);
        let mut inputs = PortValues::zeros_from(cm.input_ports());
        // Strong single vote → enzyme near 0
        inputs.get_mut("votes").unwrap().copy_from_slice(&[100.0, 0.0, 0.0, 0.0]);
        let mut outputs = PortValues::zeros_from(cm.output_ports());

        cm.tick(&inputs, &mut outputs);

        let enzyme = outputs.get("enzyme").unwrap()[0];
        assert!(enzyme < 0.01);
    }

    #[test]
    fn zero_votes_gives_uniform_pred() {
        let mut cm = ClassifyModule::new(5);
        let inputs = PortValues::zeros_from(cm.input_ports());
        let mut outputs = PortValues::zeros_from(cm.output_ports());

        cm.tick(&inputs, &mut outputs);

        let pred = outputs.get("pred").unwrap();
        assert!(pred.iter().all(|&p| (p - 0.2).abs() < 1e-5));
    }
}
