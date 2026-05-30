pub enum ConnRule {
    DenseRandom { p: f32 }, // each possible connection is made with probability p
    FixedInDegree { k: u32 }, // each neuron receives exactly k connections from the source population
    ReceptiveField { radius: u32 }, // each neuron receives connections from source neurons within a certain radius
    Topographic { patch: u8 }, // each neuron receives connections from a patch of source neurons (e.g. 3x3)
    OneToOne, // each neuron receives a connection from the corresponding neuron in the source population (only for populations of the same size)
}