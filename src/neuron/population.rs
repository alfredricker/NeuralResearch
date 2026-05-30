use crate::neuron::config::NeuronConfig;

pub struct Population {
    pub name: &'static str, // "L1Simple", "L5Pyramidal", "CA1Hippocampal" etc.
    pub config: &'static NeuronConfig, // the type of neurons in this population
    pub size: u32, // number of neurons in this population
}