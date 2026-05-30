use crate::neuron::population::Population;
use crate::network::topology::conn::{Connection, ConnRule};
use crate::neuron::dendrite::Compartment;
use crate::neuron::config::NeuronConfig;
use rand::Rng;
use super::Network;

pub struct NetworkBuilder {
    populations: Vec<Population>,
    connections: Vec<Connection>,
}

impl NetworkBuilder {
    pub fn add(&mut self, config: &'static NeuronConfig, size: u32) -> u32 { // output is population id
        let id = self.populations.len() as u32;
        self.populations.push(Population {
            config,
            size,
            name: config.name, // debug only
        });
        id
    }

    pub fn connect(&mut self, from: u32, to: u32, c: Compartment, rule: ConnRule) {
        self.connections.push(Connection {from, to, compartment: c, rule});
    }
}


pub fn build_network(populations: Vec<Population>, rng: &mut impl Rng) -> Network {
    Network {}
}

pub enum BuildError {

}