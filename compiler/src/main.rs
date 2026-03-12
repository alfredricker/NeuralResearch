use compiler::lexer::lex;
use compiler::parser::Parser;
use compiler::analyzer::Analyzer;
use std::env;
use std::fs;

fn main() {
    let input_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "../stn/1-minimal-graph.stn".to_string());

    let source = fs::read_to_string(&input_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", input_path, e));
    let tokens = lex(&source)
        .unwrap_or_else(|e| panic!("Lex error in {} at {:?}: {}", input_path, e.span, e.fragment));
    let mut parser = Parser::new(tokens);
    let program = parser
        .parse_program()
        .unwrap_or_else(|e| panic!("Parse error in {}: {}", input_path, e.message));

    let mut analyzer = Analyzer::new();
    let (declarative, executable) = analyzer
        .lower_program(&program)
        .unwrap_or_else(|errors| {
            let message = errors
                .into_iter()
                .map(|e| e.message)
                .collect::<Vec<_>>()
                .join("\n");
            panic!("IR lowering failed in {}:\n{}", input_path, message);
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
}
