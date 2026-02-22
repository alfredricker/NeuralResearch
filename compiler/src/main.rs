use compiler::lexer::lex;
use compiler::parser::Parser;
use std::env;
use std::fs;

fn main() {
    let input_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "../stn/1-minimal-graph.stn".to_string());

    let source = fs::read_to_string(&input_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", input_path, e));
    let tokens = lex(&source);
    let mut parser = Parser::new(tokens);
    let program = parser
        .parse_program()
        .unwrap_or_else(|e| panic!("Parse error in {}: {}", input_path, e));

    println!("{:#?}", program);
}
