use std::fmt::Write;
use std::fs;
use std::io;
use std::path::Path;

use crate::ir::ExecutableModule;
use crate::ir::executable::EdgeKernel;

pub struct GeneratedRustProject {
    pub cargo_toml: String,
    pub main_rs: String,
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

    let mut edges_buf = String::new();
    for edge in &graph.edges {
        let weight = match edge.kernel {
            EdgeKernel::PassThrough => 1.0_f32,
            EdgeKernel::WeightedSum => 1.0_f32,
        };
        writeln!(
            edges_buf,
            "    ({}, {}, {}f32),",
            edge.from, edge.to, weight
        )
        .map_err(|e| e.to_string())?;
    }

    let main_rs = format!(
        r#"const NODE_COUNT: usize = {node_count};
const EDGES: &[(usize, usize, f32)] = &[
{edges}
];

fn tick(activations: &mut [f32], input_buffer: &mut [f32]) {{
    for &(from, to, weight) in EDGES {{
        input_buffer[to] += activations[from] * weight;
    }}

    for i in 0..activations.len() {{
        activations[i] = input_buffer[i];
        input_buffer[i] = 0.0;
    }}
}}

fn main() {{
    let mut activations = vec![0.0_f32; NODE_COUNT];
    let mut input_buffer = vec![0.0_f32; NODE_COUNT];

    // Seed a tiny signal so output is observable.
    if !activations.is_empty() {{
        activations[0] = 1.0;
    }}

    for _step in 0..3 {{
        tick(&mut activations, &mut input_buffer);
    }}

    let sum: f32 = activations.iter().copied().sum();
    println!("nodes={{}} edges={{}} activation_sum={{:.4}}", NODE_COUNT, EDGES.len(), sum);
}}
"#,
        node_count = node_count,
        edges = edges_buf
    );

    let cargo_toml = format!(
        r#"[package]
name = "{package_name}"
version = "0.1.0"
edition = "2024"

[dependencies]
"#
    );

    Ok(GeneratedRustProject {
        cargo_toml,
        main_rs,
    })
}

pub fn write_rust_project(project: &GeneratedRustProject, output_dir: &Path) -> io::Result<()> {
    let src_dir = output_dir.join("src");
    fs::create_dir_all(&src_dir)?;
    fs::write(output_dir.join("Cargo.toml"), &project.cargo_toml)?;
    fs::write(src_dir.join("main.rs"), &project.main_rs)?;
    Ok(())
}
