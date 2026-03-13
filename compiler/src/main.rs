use compiler::lexer::lex;
use compiler::parser::Parser;
use compiler::ir::{ExecutableModule, ModuleIr};
use compiler::codegen::{emit_rust_project, write_rust_project};
use std::env;
use std::path::Path;
use std::fs;

fn main() {
    let mut args = env::args();
    let _bin_name = args.next();
    let input_path = args
        .next()
        .unwrap_or_else(|| "../stn/1-minimal-graph.stn".to_string());
    let output_dir = args
        .next()
        .unwrap_or_else(|| "../generated-rust".to_string());

    let source = fs::read_to_string(&input_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", input_path, e));
    let tokens = lex(&source)
        .unwrap_or_else(|e| panic!("Lex error in {} at {:?}: {}", input_path, e.span, e.fragment));
    let mut parser = Parser::new(tokens);
    let program = parser
        .parse_program()
        .unwrap_or_else(|e| panic!("Parse error in {}: {}", input_path, e.message));

    let declarative = ModuleIr::from_program(&program).unwrap_or_else(|errors| {
        let message = errors
            .into_iter()
            .map(|e| e.message)
            .collect::<Vec<_>>()
            .join("\n");
        panic!("Declarative IR lowering failed in {}:\n{}", input_path, message);
    });
    let executable = ExecutableModule::from_declarative(&declarative).unwrap_or_else(|errors| {
        let message = errors
            .into_iter()
            .map(|e| e.message)
            .collect::<Vec<_>>()
            .join("\n");
        panic!("Executable IR lowering failed in {}:\n{}", input_path, message);
    });

    println!("Parsed {} top-level items", program.items().len());
    println!(
        "Declarative IR: {} graph(s), {} interface(s), {} external link(s)",
        declarative.graphs.len(),
        declarative.interfaces.len(),
        declarative.links.len()
    );
    for graph in &declarative.graphs {
        println!(
            "  Graph {}: {} group(s), {} group-link(s)",
            graph.id,
            graph.groups.len(),
            graph.links.len()
        );
    }

    println!(
        "Executable IR: {} graph(s), {} external runtime link(s)",
        executable.graphs.len(),
        executable.links.len()
    );
    for graph in &executable.graphs {
        println!(
            "  Graph {}: {} node(s), {} edge(s), {} storage slot(s)",
            graph.graph_id,
            graph.nodes.len(),
            graph.edges.len(),
            graph.storage.slots.len()
        );
    }

    let generated = emit_rust_project(&executable, "stn_generated")
        .unwrap_or_else(|e| panic!("Codegen failed: {}", e));
    write_rust_project(&generated, Path::new(&output_dir))
        .unwrap_or_else(|e| panic!("Failed to write generated project to {}: {}", output_dir, e));
    println!("Generated Rust project at {}", output_dir);
}
