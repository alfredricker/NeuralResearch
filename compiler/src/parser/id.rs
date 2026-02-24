use crate::parser::{ParseError, Parser};
use crate::lexer::Token;

impl Parser {
    // inspect the current token, consume one token, and return the identifier string
    pub fn parse_identifier(&mut self) -> Result<String, ParseError> {
        let start = self.pos;
        let spanned = self.bump().ok_or_else(|| ParseError::new("Unexpected EOF"))?;
        
        match &spanned.token {
            Token::Ident(name) => Ok(name.clone()),
            other => Err(ParseError::with_span(
                format!("Expected identifier, found {:?}", other),
                spanned.span.clone(),
            ))
        }
    }
}