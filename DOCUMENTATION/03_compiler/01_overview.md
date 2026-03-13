# The STN Compiler

The compiler for the Subgraph Topology Network (STN) language is written in Rust and follows standard Domain Specific Language (DSL) compiler design patterns. 

## The Lexer
The lexer is responsible for breaking down the source code into tokens. 

## The Parser
The parser iterates through the tokens and builds an abstract syntax tree (AST).

## The AST
The AST is a tree representation of the source code. It is used to represent the structure of the source code in a way that is easy to understand and manipulate. 

## The IR
The intermediate representation (IR) is a lower level representation of the source code. It is used to represent the structure of the source code in a way that is easy to understand and manipulate. 

## The Code Generator
The code generator is responsible for generating the final executable code from the IR. 