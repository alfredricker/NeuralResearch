use crate::neuron::dendrite::Compartment;
use thiserror::Error;

pub struct Connection {
    pub from: u32,
    pub to: u32,
    pub compartment: Compartment,
    pub rule: ConnRule,
}

impl Connection {
    pub fn new(from: u32, to: u32, compartment: Compartment, rule: ConnRule) -> Self {
        Self {from, to, compartment, rule}
    }
}

pub enum ConnRule {
    DenseRandom { p: f32 }, // each possible connection is made with probability p
    FixedInDegree { k: u32 }, // each neuron receives exactly k connections from the source population
    ReceptiveField { radius: u32 }, // each neuron receives connections from source neurons within a certain radius
    Topographic { patch: u8 }, // each neuron receives connections from a patch of source neurons (e.g. 3x3)
    OneToOne, // each neuron receives a connection from the corresponding neuron in the source population (only for populations of the same size)
}

#[derive(Error, Debug)]
pub enum ConnError {
    #[error("No connection rules provided")]
    NoConnections,
    #[error("Invalid connection rule provided")]
    InvalidRule,
}