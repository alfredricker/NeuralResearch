use crate::ast::{Program, Item};
use crate::lexer::{SpannedToken, Token};
pub mod error;
use error::ParseError;

pub mod id;
pub mod link;
pub mod io;

#[derive(Debug)]
pub struct Parser {
    tokens: Vec<SpannedToken>,
    pos: usize,
}

impl Parser {
    /// Creates a parser over a pre-tokenized STN stream.
    pub fn new(tokens: Vec<SpannedToken>) -> Self {
        Self { tokens, pos: 0 }
    }

    /// Parses a complete minimal STN program in top-level declaration order.
    pub fn parse_program(&mut self) -> Result<Program, ParseError> {
        // first, determine what kind of statement it is:
        // statements should start with a keyword.
        let mut program = Program::new();

        while let Some(tok) = self.peek_token(None) {
            let item = self.parse_item()?;
            program.push_item(item);
        }

        Ok(program)

    }

    pub fn parse_item(&mut self) -> Result<Item, ParseError> {
        match self.peek_token(None) {
            Some(Token::Graph) | Some(Token::Subgraph) | Some(Token::Top) | 
            Some(Token::Learn) | Some(Token::Display) => {
                let block = self.parse_block()?;
                Ok(Item::Block(block))
            }

            Some(Token::Input) | Some(Token::Output) => {
                let stmt = self.parse_statement()?;
                Ok(Item::Statement(stmt))
            }

            Some(Token::Ident(_)) => {
                match (self.peek_token(Some(1)), self.peek_token(Some(2))) {
                    (Some(Token::Arrow), _) => {
                        let stmt = self.parse_statement()?;
                        Ok(Item::Statement(stmt))
                    }
                    _ => Err(ParseError::new(format!(
                        "Unexpected identifier at {:?}",
                        self.pos
                    ))),
                }
            }
            
            Some(tok) => Err(ParseError::new(format!("Unexpected token: {:?}", tok))),
            None => Err(ParseError::new("Unexpected EOF")),

        }
    }


    /// Returns the current token without consuming it.
    fn peek(&self, n: Option<u32>) -> Option<&SpannedToken> {
        match n {
            Some(m) => { self.tokens.get(self.pos + m as usize) }
            None => {self.tokens.get(self.pos)}
        }
    }

    fn peek_token(&self, n: Option<u32>) -> Option<&Token> {
        self.peek(n).map(|s| &s.token)
    }

    /// Returns the current token and advances the cursor by one token.
    fn bump(&mut self) -> Option<&SpannedToken> {
        let t = self.tokens.get(self.pos);
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    // Returns void and advances one token on success
    fn expect(&mut self, expected: Token) -> Result<(), ParseError> {
        let tok= self.peek_token(None);
        match tok {
            Some(a) => {
                if same_variant(a, &expected){
                    self.pos +=1;
                    Ok(())
                }
                else {
                    Err(ParseError::new("Expect token did not match"))
                }
            }
            None => { Err(ParseError::new("Got None token for self.pos")) }
        }
    }
}

/// Compares enum variants while ignoring any payload values.
fn same_variant(a: &Token, b: &Token) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}
