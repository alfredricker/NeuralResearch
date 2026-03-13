use std::fs;
use std::io;
use std::path::Path;

use crate::ir::ExecutableModule;
use crate::ir::executable::EdgeKernel;

pub struct GeneratedRustProject {
    pub cargo_toml: String,
    pub main_rs: String,
    pub graph_bin: Vec<u8>,
}

pub fn emit_rust_project(
    module: &ExecutableModule,
    package_name: &str,
) -> Result<GeneratedRustProject, String> {
    let graph = module
        .graphs
        .first()
        .ok_or_else(|| "Executable module contains no graphs".to_string())?;

    let node_count = graph.nodes.len();
    let edge_count = graph.edges.len();

    // Build CSR format: sort edges by source node, create offsets array
    let mut edges_sorted = graph.edges.clone();
    edges_sorted.sort_by_key(|e| e.from);

    // Build offsets: offsets[i] = start index of edges from node i
    let mut offsets = vec![0u32; node_count + 1];
    let mut targets = vec![];
    let mut weights = vec![];

    for edge in &edges_sorted {
        targets.push(edge.to as u32);
        let weight = match edge.kernel {
            EdgeKernel::PassThrough => 1.0_f32,
            EdgeKernel::WeightedSum => 1.0_f32,
        };
        weights.push(weight);
    }

    // Fill offsets: count edges per node
    let mut current_offset = 0u32;
    let mut current_node = 0usize;
    for (i, edge) in edges_sorted.iter().enumerate() {
        let from_idx = edge.from as usize;
        while current_node <= from_idx {
            offsets[current_node] = current_offset;
            current_node += 1;
        }
        current_offset = (i + 1) as u32;
    }
    // Fill remaining offsets with final value
    while current_node <= node_count {
        offsets[current_node] = current_offset;
        current_node += 1;
    }

    // Serialize to binary (little-endian)
    let mut graph_bin = Vec::new();
    graph_bin.extend_from_slice(&(node_count as u32).to_le_bytes());
    graph_bin.extend_from_slice(&(edge_count as u32).to_le_bytes());

    // Write offsets
    for &offset in &offsets {
        graph_bin.extend_from_slice(&offset.to_le_bytes());
    }

    // Write targets
    for &target in &targets {
        graph_bin.extend_from_slice(&target.to_le_bytes());
    }

    // Write weights
    for &weight in &weights {
        graph_bin.extend_from_slice(&weight.to_le_bytes());
    }

    // Generate lean main.rs that loads graph at runtime
    let main_rs = r#"use std::fs;

fn load_graph(path: &str) -> Result<(usize, Vec<u32>, Vec<u32>, Vec<f32>), Box<dyn std::error::Error>> {
    let data = fs::read(path)?;
    let mut pos = 0;

    // Read node_count and edge_count
    let node_count = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
    pos += 4;
    let edge_count = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
    pos += 4;

    // Read offsets
    let mut offsets = Vec::with_capacity(node_count + 1);
    for _ in 0..=node_count {
        let val = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
        offsets.push(val);
        pos += 4;
    }

    // Read targets
    let mut targets = Vec::with_capacity(edge_count);
    for _ in 0..edge_count {
        let val = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
        targets.push(val);
        pos += 4;
    }

    // Read weights
    let mut weights = Vec::with_capacity(edge_count);
    for _ in 0..edge_count {
        let val = f32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
        weights.push(val);
        pos += 4;
    }

    Ok((node_count, offsets, targets, weights))
}

fn tick(
    activations: &mut [f32],
    input_buffer: &mut [f32],
    offsets: &[u32],
    targets: &[u32],
    weights: &[f32],
    active_set: &mut Vec<u32>,
    next_set: &mut Vec<u32>,
) {
    next_set.clear();
    for &node in active_set.iter() {
        let node_idx = node as usize;
        let s = offsets[node_idx] as usize;
        let e = offsets[node_idx + 1] as usize;
        for i in s..e {
            let t = targets[i] as usize;
            input_buffer[t] += activations[node_idx] * weights[i];
            next_set.push(targets[i]);
        }
    }
    // Apply and zero
    for &node in next_set.iter() {
        let n = node as usize;
        activations[n] = input_buffer[n];
        input_buffer[n] = 0.0;
    }
    std::mem::swap(active_set, next_set);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (node_count, offsets, targets, weights) = load_graph("graph.bin")?;
    let edge_count = weights.len();

    let mut activations = vec![0.0_f32; node_count];
    let mut input_buffer = vec![0.0_f32; node_count];
    let mut active_set = vec![0];
    let mut next_set = Vec::new();

    for _step in 0..3 {
        tick(
            &mut activations,
            &mut input_buffer,
            &offsets,
            &targets,
            &weights,
            &mut active_set,
            &mut next_set,
        );
    }

    let sum: f32 = activations.iter().copied().sum();
    println!("nodes={} edges={} activation_sum={:.4}", node_count, edge_count, sum);

    Ok(())
}
"#.to_string();

    let cargo_toml = format!(
        r#"[package]
name = "{package_name}"
version = "0.1.0"
edition = "2021"

[dependencies]
"#
    );

    Ok(GeneratedRustProject {
        cargo_toml,
        main_rs,
        graph_bin,
    })
}

pub fn write_rust_project(project: &GeneratedRustProject, output_dir: &Path) -> io::Result<()> {
    let src_dir = output_dir.join("src");
    fs::create_dir_all(&src_dir)?;
    fs::write(output_dir.join("Cargo.toml"), &project.cargo_toml)?;
    fs::write(src_dir.join("main.rs"), &project.main_rs)?;
    fs::write(output_dir.join("graph.bin"), &project.graph_bin)?;
    Ok(())
}
