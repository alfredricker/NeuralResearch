//! `NetworkSpec` — the reproducible *recipe* for a network, the playground's save format.
//!
//! A built [`Network`] is a large bag of SoA arrays whose every value is drawn from an RNG at
//! build time (weights, synapse positions, dendrite constants, random connectivity). Storing the
//! baked arrays would be huge and opaque. Instead we store the **recipe**: the populations, the
//! connections, and the seed. Because [`neural_sim::network::build::build_network`] is
//! deterministic in its RNG, `spec.build()` reconstructs the *exact same* network every time —
//! kilobytes of human-editable JSON instead of a multi-megabyte dump, and no `.ntr` needed for a
//! network you only want to rebuild.
//!
//! These types mirror the `neural-sim` builder types (`NeuronConfig`, `Compartment`, `ConnRule`)
//! the same way [`crate::EventRecord`] mirrors `neural_sim::Event`: the engine stays serde-free,
//! and the owned, `serde`-derivable copies live here. The *only* non-trivial mirror is the neuron
//! config, because it holds `Sampler{I8,U8}` (precomputed alias tables) — but a sampler is fully
//! described by its `(mean, std)`, so [`SamplerSpec`] captures it losslessly.

use std::collections::BTreeMap;
use std::path::Path;

use neural_sim::math::sample::{SamplerI8, SamplerU8};
use neural_sim::network::Network;
use neural_sim::network::build::{NetworkBuilder, build_network};
use neural_sim::network::topology::conn::ConnRule;
use neural_sim::neuron::config::NeuronConfig;
use neural_sim::neuron::dendrite::Compartment;
use neural_sim::io::{input_config, output_config};

use rand::SeedableRng;
use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A discretized-normal sampler, captured by the only two numbers that define it. Builds into
/// either a [`SamplerU8`] or a [`SamplerI8`] (the engine reconstructs the alias table). `mean` is
/// `i16` so one struct serves both unsigned and signed fields; it is cast at build.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SamplerSpec {
    pub mean: i16,
    pub std: u8,
}

impl SamplerSpec {
    pub fn new(mean: i16, std: u8) -> Self {
        Self { mean, std }
    }
    fn as_u8(self) -> SamplerU8 {
        SamplerU8::new(self.mean as u8, self.std)
    }
    fn as_i8(self) -> SamplerI8 {
        SamplerI8::new(self.mean as i8, self.std)
    }
}

/// Owned, serializable mirror of [`neural_sim::neuron::config::NeuronConfig`]. Field-for-field the
/// same, with every `Sampler*` replaced by a [`SamplerSpec`]. `apical_*` are `Option` together: a
/// config either has an apical compartment (all three `Some`) or none.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronTypeSpec {
    pub n_basal_dendrites: u8,
    pub n_apical_dendrites: Option<u8>,
    pub synapse_x_sampler: SamplerSpec,
    pub dendrites_per_branch: SamplerSpec,
    pub synapses_per_dendrite: SamplerSpec,
    pub soma_threshold: i8,
    pub basal_dendrite_threshold: u16,
    pub basal_dendrite_constant: SamplerSpec,
    pub apical_dendrite_threshold: Option<u16>,
    pub apical_dendrite_constant: Option<SamplerSpec>,
    pub learning_rate: i16,
}

impl NeuronTypeSpec {
    /// Reconstruct a `&'static NeuronConfig` from this spec. The builder requires `&'static`, so
    /// both the config and its name are `Box::leak`ed. NOTE: this leaks once per build — fine for
    /// the personal playground workflow (a handful of builds per session); if the playground ever
    /// rebuilds in a hot loop, the fix is to make `NetworkBuilder` own its configs instead.
    fn leak_config(&self, name: &str) -> &'static NeuronConfig {
        let name: &'static str = Box::leak(name.to_string().into_boxed_str());
        Box::leak(Box::new(NeuronConfig::new(
            name,
            self.n_basal_dendrites,
            self.n_apical_dendrites,
            self.synapse_x_sampler.as_u8(),
            self.dendrites_per_branch.as_u8(),
            self.synapses_per_dendrite.as_u8(),
            self.soma_threshold,
            self.basal_dendrite_threshold,
            self.basal_dendrite_constant.as_i8(),
            self.apical_dendrite_threshold,
            self.apical_dendrite_constant.map(SamplerSpec::as_i8),
            self.learning_rate,
        )))
    }
}

/// Owned mirror of [`neural_sim::neuron::dendrite::Compartment`] — which compartment a connection
/// lands on at the target population.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CompartmentSpec {
    Basal,
    Apical,
}

impl From<CompartmentSpec> for Compartment {
    fn from(c: CompartmentSpec) -> Self {
        match c {
            CompartmentSpec::Basal => Compartment::Basal,
            CompartmentSpec::Apical => Compartment::Apical,
        }
    }
}

/// Owned mirror of [`neural_sim::network::topology::conn::ConnRule`]. Same variants, same fields.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConnRuleSpec {
    DenseRandom { p: f32 },
    FixedInDegree { k: u32 },
    ReceptiveField { radius: u32 },
    Topographic { patch: u8 },
    OneToOne,
}

impl From<ConnRuleSpec> for ConnRule {
    fn from(r: ConnRuleSpec) -> Self {
        match r {
            ConnRuleSpec::DenseRandom { p } => ConnRule::DenseRandom { p },
            ConnRuleSpec::FixedInDegree { k } => ConnRule::FixedInDegree { k },
            ConnRuleSpec::ReceptiveField { radius } => ConnRule::ReceptiveField { radius },
            ConnRuleSpec::Topographic { patch } => ConnRule::Topographic { patch },
            ConnRuleSpec::OneToOne => ConnRule::OneToOne,
        }
    }
}

/// Which neuron type a population is made of. Custom types are looked up by name in
/// [`NetworkSpec::neuron_types`]; the two builtins resolve to `neural_sim::io`'s shared configs
/// (input neurons have no dendrites; output neurons integrate but have no apical compartment).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NeuronTypeRef {
    Input,
    Output,
    Custom(String),
}

/// One population: a count of neurons of a given type. Its index in [`NetworkSpec::populations`]
/// is its population id — exactly the value `NetworkBuilder::add` returns and that connections
/// reference via `from`/`to`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationSpec {
    pub neuron_type: NeuronTypeRef,
    pub size: u32,
    /// Optional human label for the editor (e.g. "hidden", "L5 pyramidal"); ignored by `build`.
    #[serde(default)]
    pub label: Option<String>,
}

/// One directed projection from population `from` onto a compartment of population `to`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionSpec {
    pub from: u32,
    pub to: u32,
    pub compartment: CompartmentSpec,
    pub rule: ConnRuleSpec,
}

/// The complete recipe for a network. `seed` + this struct ⇒ a bit-identical [`Network`] via
/// [`NetworkSpec::build`]. This is the playground's on-disk format (`networks/<name>.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSpec {
    /// RNG seed threaded into `build_network` — the source of every sampled value and random edge.
    pub seed: u64,
    /// Custom neuron types, keyed by the name `NeuronTypeRef::Custom` references.
    #[serde(default)]
    pub neuron_types: BTreeMap<String, NeuronTypeSpec>,
    /// Populations in `add` order; index == population id.
    pub populations: Vec<PopulationSpec>,
    /// Projections between populations.
    #[serde(default)]
    pub connections: Vec<ConnectionSpec>,
}

impl NetworkSpec {
    /// Resolve a population's neuron type to the `&'static NeuronConfig` the builder needs.
    fn resolve(&self, r: &NeuronTypeRef) -> Result<&'static NeuronConfig, SpecError> {
        match r {
            NeuronTypeRef::Input => Ok(input_config()),
            NeuronTypeRef::Output => Ok(output_config()),
            NeuronTypeRef::Custom(name) => self
                .neuron_types
                .get(name)
                .map(|t| t.leak_config(name))
                .ok_or_else(|| SpecError::UnknownType(name.clone())),
        }
    }

    /// Reconstruct the network. Deterministic: same spec ⇒ same `Network`. Connection `from`/`to`
    /// are validated against the population count so a bad index is a clean error, not a panic.
    pub fn build(&self) -> Result<Network, SpecError> {
        let n_pops = self.populations.len() as u32;
        let mut builder = NetworkBuilder { populations: Vec::new(), connections: Vec::new() };
        for p in &self.populations {
            let cfg = self.resolve(&p.neuron_type)?;
            builder.add(cfg, p.size);
        }
        for c in &self.connections {
            if c.from >= n_pops || c.to >= n_pops {
                return Err(SpecError::BadConnection { from: c.from, to: c.to, n_pops });
            }
            builder.connect(c.from, c.to, c.compartment.into(), c.rule.into());
        }
        let mut rng = SmallRng::seed_from_u64(self.seed);
        Ok(build_network(builder, &mut rng))
    }

    /// Pretty-print to JSON (human-editable, git-diffable — same spirit as the `.ntr.json` manifest).
    pub fn to_json(&self) -> Result<String, SpecError> {
        serde_json::to_string_pretty(self).map_err(SpecError::Json)
    }

    /// Parse from a JSON string.
    pub fn from_json(s: &str) -> Result<Self, SpecError> {
        serde_json::from_str(s).map_err(SpecError::Json)
    }

    /// Read a spec from `networks/<name>.json`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SpecError> {
        let bytes = std::fs::read(path.as_ref())?;
        serde_json::from_slice(&bytes).map_err(SpecError::Json)
    }

    /// Write the spec to disk as pretty JSON.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), SpecError> {
        std::fs::write(path.as_ref(), self.to_json()?)?;
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum SpecError {
    #[error("unknown neuron type '{0}' (not in neuron_types)")]
    UnknownType(String),
    #[error("connection references population out of range: from={from} to={to} (have {n_pops})")]
    BadConnection { from: u32, to: u32, n_pops: u32 },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[source] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A blobs-style hidden neuron, as a spec (mirrors crates/neural-cli/src/blobs.rs).
    fn hidden_type() -> NeuronTypeSpec {
        NeuronTypeSpec {
            n_basal_dendrites: 6,
            n_apical_dendrites: None,
            synapse_x_sampler: SamplerSpec::new(128, 50),
            dendrites_per_branch: SamplerSpec::new(1, 0),
            synapses_per_dendrite: SamplerSpec::new(16, 0),
            soma_threshold: 20,
            basal_dendrite_threshold: 500,
            basal_dendrite_constant: SamplerSpec::new(40, 10),
            apical_dendrite_threshold: None,
            apical_dendrite_constant: None,
            learning_rate: 120,
        }
    }

    fn blobs_spec() -> NetworkSpec {
        let mut neuron_types = BTreeMap::new();
        neuron_types.insert("hidden".to_string(), hidden_type());
        NetworkSpec {
            seed: 7,
            neuron_types,
            populations: vec![
                PopulationSpec { neuron_type: NeuronTypeRef::Input, size: 25, label: Some("place".into()) },
                PopulationSpec { neuron_type: NeuronTypeRef::Custom("hidden".into()), size: 16, label: None },
                PopulationSpec { neuron_type: NeuronTypeRef::Output, size: 2, label: None },
            ],
            connections: vec![
                ConnectionSpec { from: 0, to: 1, compartment: CompartmentSpec::Basal, rule: ConnRuleSpec::FixedInDegree { k: 8 } },
                ConnectionSpec { from: 1, to: 2, compartment: CompartmentSpec::Basal, rule: ConnRuleSpec::FixedInDegree { k: 4 } },
            ],
        }
    }

    #[test]
    fn builds_expected_population_sizes() {
        let net = blobs_spec().build().unwrap();
        assert_eq!(net.n_neurons(), 25 + 16 + 2);
    }

    #[test]
    fn same_seed_is_reproducible() {
        let spec = blobs_spec();
        let a = spec.build().unwrap();
        let b = spec.build().unwrap();
        // identical topology counts is a cheap proxy; the RNG draws all flow from `seed`.
        assert_eq!(a.n_neurons(), b.n_neurons());
        assert_eq!(a.n_dendrites(), b.n_dendrites());
        assert_eq!(a.n_synapses(), b.n_synapses());
    }

    #[test]
    fn json_round_trips() {
        let spec = blobs_spec();
        let json = spec.to_json().unwrap();
        let back = NetworkSpec::from_json(&json).unwrap();
        assert_eq!(back.populations.len(), 3);
        assert_eq!(back.build().unwrap().n_neurons(), net_count(&spec));
    }

    fn net_count(spec: &NetworkSpec) -> usize {
        spec.populations.iter().map(|p| p.size as usize).sum()
    }

    #[test]
    fn unknown_type_is_clean_error() {
        let mut spec = blobs_spec();
        spec.populations[1].neuron_type = NeuronTypeRef::Custom("missing".into());
        assert!(matches!(spec.build(), Err(SpecError::UnknownType(_))));
    }

    #[test]
    fn out_of_range_connection_is_clean_error() {
        let mut spec = blobs_spec();
        spec.connections.push(ConnectionSpec { from: 0, to: 9, compartment: CompartmentSpec::Basal, rule: ConnRuleSpec::OneToOne });
        assert!(matches!(spec.build(), Err(SpecError::BadConnection { .. })));
    }
}
