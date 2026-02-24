pub mod inputs;

use crate::ast::Program;
use crate::lexer::{SpannedToken, Token};

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
    pub fn parse_program(&mut self) -> Result<Program, String> {
        // first, determine what kind of statement it is:
        // statements should start with a keyword.
        while let Some(tok) = self.peek_token() {

        }

    }


    /// Returns the current token without consuming it.
    fn peek(&self) -> Option<&SpannedToken> {
        self.tokens.get(self.pos)
    }

    fn peek_token(&self) -> Option<&Token> {
        self.peek().map(|s| &s.token)
    }

    /// Returns the current token and advances the cursor by one token.
    fn bump(&mut self) -> Option<&SpannedToken> {
        let t = self.tokens.get(self.pos);
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
}

/// Compares enum variants while ignoring any payload values.
fn same_variant(a: &SpannedToken, b: &SpannedToken) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}
