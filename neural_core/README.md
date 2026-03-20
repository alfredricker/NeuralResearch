`LearnRule` in `src/learning/mod.rs` is the enum of all the supported learning rules.
`DriveRule` in `src/drive/mod.rs` is the enum of how to consolidate inputs.
`UpdateRule` in `src/state/mod.rs` is the enum of all the supported update rules. Update rules change neuron state based on inputs and learning rules change weights based on neuron states.