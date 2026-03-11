use logos::Logos;

#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\n\f]+")]
pub enum Token {
    #[token("=")]
    Equal,
    #[token("->")]
    Arrow,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token(":")]
    Colon,
    #[token(";")]
    Semi,
    #[token(",")]
    Comma,

    #[token("default")]
    Default,

    // scope declarations
    #[token("graph")]
    Graph,
    #[token("subgraph")]
    Subgraph,
    #[token("learn")]
    Learn,
    #[token("display")]
    Display,
    #[token("topology")]
    Top,

    #[token("input")]
    Input,
    #[token("output")]
    Output,
    #[token("nodes")]
    Nodes,

    // topologies
    #[token("sparse")]
    Sparse,
    #[token("identity")]
    Identity,
    #[token("dense")]
    Dense,

    // transforms
    #[token("weighted_sum")]
    WeightedSum,

    // io types
    #[token("Image")]
    Image,
    #[token("Language")]
    Language,
    #[token("Class")]
    Class,
    #[token("Logits")]
    Logits,

    #[regex(r"[0-9]+\.[0-9]+", |lex| lex.slice().parse::<f64>().unwrap())]
    Float(f64),
    #[regex(r"[0-9]+", |lex| lex.slice().parse::<u32>().unwrap())]
    Integer(u32),
    #[regex(r"[A-Za-z_][A-Za-z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),
}