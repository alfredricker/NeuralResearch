// define types for populations of neurons
pub struct NeuronType {
    pub name: &'static str, // "L1Simple", "L5Pyramidal", "CA1Hippocampal" etc.
    pub config: NeuronConfig, // parameters for the individual neurons in this population
}

impl NeuronType {
    pub fn new(name: &'static str, ) {

    }
}