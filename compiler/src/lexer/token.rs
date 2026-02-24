use logos::Logos;

#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\n\f]+")]
pub enum Token {
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

    #[token("graph")]
    Graph,
    #[token("input")]
    Input,
    #[token("output")]
    Output,
    #[token("nodes")]
    Nodes,
    #[token("sparse")]
    Sparse,
    #[token("identity")]
    Identity,
    #[token("weighted_sum")]
    WeightedSum,
    #[token("Image")]
    Image,
    #[token("Class")]
    Class,

    #[regex(r"[0-9]+\.[0-9]+", |lex| lex.slice().parse::<f64>().unwrap())]
    Float(f64),
    #[regex(r"[0-9]+", |lex| lex.slice().parse::<u32>().unwrap())]
    Integer(u32),
    #[regex(r"[A-Za-z_][A-Za-z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),
}