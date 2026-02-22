use crate::ast::{
    EdgeDecl, GraphDecl, InputDecl, LinkDecl, NodeGroupDecl, OutputDecl, Program, Topology,
};
use crate::lexer::Token;

#[derive(Debug)]
pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    pub fn parse_program(&mut self) -> Result<Program, String> {
        let input = self.parse_input_decl()?;
        let output = self.parse_output_decl()?;
        let graph = self.parse_graph_decl()?;

        let mut links = Vec::new();
        while self.peek().is_some() {
            links.push(self.parse_link_decl()?);
        }

        Ok(Program {
            input,
            output,
            graph,
            links,
        })
    }

    fn parse_input_decl(&mut self) -> Result<InputDecl, String> {
        self.expect_simple(Token::Input, "input")?;
        let name = self.expect_ident("input name")?;
        self.expect_simple(Token::Colon, ":")?;
        self.expect_simple(Token::Image, "Image")?;
        self.expect_simple(Token::LParen, "(")?;
        let width = self.expect_int("image width")?;
        self.expect_simple(Token::Comma, ",")?;
        let height = self.expect_int("image height")?;
        self.expect_simple(Token::RParen, ")")?;
        self.expect_simple(Token::Semi, ";")?;
        Ok(InputDecl {
            name,
            width,
            height,
        })
    }

    fn parse_output_decl(&mut self) -> Result<OutputDecl, String> {
        self.expect_simple(Token::Output, "output")?;
        let name = self.expect_ident("output name")?;
        self.expect_simple(Token::Colon, ":")?;
        self.expect_simple(Token::Class, "Class")?;
        self.expect_simple(Token::LParen, "(")?;
        let classes = self.expect_int("class count")?;
        self.expect_simple(Token::RParen, ")")?;
        self.expect_simple(Token::Semi, ";")?;
        Ok(OutputDecl { name, classes })
    }

    fn parse_graph_decl(&mut self) -> Result<GraphDecl, String> {
        self.expect_simple(Token::Graph, "graph")?;
        self.expect_simple(Token::LBrace, "{")?;

        let mut node_groups = Vec::new();
        let mut edges = Vec::new();
        while !self.next_is(&Token::RBrace) {
            if self.lookahead_is_node_group() {
                node_groups.push(self.parse_node_group_decl()?);
            } else {
                edges.push(self.parse_edge_decl()?);
            }
        }

        self.expect_simple(Token::RBrace, "}")?;
        Ok(GraphDecl { node_groups, edges })
    }

    fn parse_node_group_decl(&mut self) -> Result<NodeGroupDecl, String> {
        let name = self.expect_ident("node group name")?;
        self.expect_simple(Token::Colon, ":")?;
        self.expect_simple(Token::Nodes, "nodes")?;
        self.expect_simple(Token::LParen, "(")?;
        let count = self.expect_int("node count")?;
        self.expect_simple(Token::RParen, ")")?;
        self.expect_simple(Token::Semi, ";")?;
        Ok(NodeGroupDecl { name, count })
    }

    fn parse_edge_decl(&mut self) -> Result<EdgeDecl, String> {
        let from = self.expect_ident("edge source")?;
        self.expect_simple(Token::Arrow, "->")?;
        let to = self.expect_ident("edge destination")?;
        self.expect_simple(Token::Colon, ":")?;
        let topology = self.parse_topology()?;
        self.expect_simple(Token::Semi, ";")?;
        Ok(EdgeDecl { from, to, topology })
    }

    fn parse_link_decl(&mut self) -> Result<LinkDecl, String> {
        let from = self.expect_ident("link source")?;
        self.expect_simple(Token::Arrow, "->")?;
        let to = self.expect_ident("link destination")?;
        self.expect_simple(Token::Colon, ":")?;
        let topology = self.parse_topology()?;
        self.expect_simple(Token::Semi, ";")?;
        Ok(LinkDecl { from, to, topology })
    }

    fn parse_topology(&mut self) -> Result<Topology, String> {
        match self.peek() {
            Some(Token::Sparse) => {
                self.bump();
                self.expect_simple(Token::LParen, "(")?;
                let p = self.expect_float("sparse probability")?;
                self.expect_simple(Token::RParen, ")")?;
                Ok(Topology::Sparse(p))
            }
            Some(Token::Identity) => {
                self.bump();
                Ok(Topology::Identity)
            }
            Some(Token::WeightedSum) => {
                self.bump();
                Ok(Topology::WeightedSum)
            }
            other => Err(format!("Expected topology, found {:?}", other)),
        }
    }

    fn lookahead_is_node_group(&self) -> bool {
        matches!(
            (self.tokens.get(self.pos), self.tokens.get(self.pos + 1)),
            (Some(Token::Ident(_)), Some(Token::Colon))
        )
    }

    fn next_is(&self, t: &Token) -> bool {
        self.peek().is_some_and(|p| same_variant(p, t))
    }

    fn expect_ident(&mut self, what: &str) -> Result<String, String> {
        match self.bump() {
            Some(Token::Ident(s)) => Ok(s.clone()),
            other => Err(format!("Expected {}, found {:?}", what, other)),
        }
    }

    fn expect_int(&mut self, what: &str) -> Result<u32, String> {
        match self.bump() {
            Some(Token::Integer(v)) => Ok(*v),
            other => Err(format!("Expected {}, found {:?}", what, other)),
        }
    }

    fn expect_float(&mut self, what: &str) -> Result<f64, String> {
        match self.bump() {
            Some(Token::Float(v)) => Ok(*v),
            Some(Token::Integer(v)) => Ok(*v as f64),
            other => Err(format!("Expected {}, found {:?}", what, other)),
        }
    }

    fn expect_simple(&mut self, expected: Token, label: &str) -> Result<(), String> {
        match self.bump() {
            Some(tok) if same_variant(tok, &expected) => Ok(()),
            other => Err(format!("Expected {}, found {:?}", label, other)),
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn bump(&mut self) -> Option<&Token> {
        let t = self.tokens.get(self.pos);
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
}

fn same_variant(a: &Token, b: &Token) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}
