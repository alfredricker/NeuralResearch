use crate::parser::{ParseError, Parser};
use crate::ast::statement::Statement;
use crate::lexer::Token;

impl Parser {
    pub fn parse_statement(&mut self) -> Result<Statement, ParseError> {
        match self.peek_token(None) {
            // want to consume Input and Output declarations -- we call the appropriate functions
            Some(Token::Input) => self.parse_input_decl_statement(),   // should consume ';'
            Some(Token::Output) => self.parse_output_decl_statement(), // should consume ';'
            Some(Token::Ident(_)) => {
                // ident-led: currently link form: a -> b : topology ;
                if matches!(self.peek_token(Some(1)), Some(Token::Arrow)) {
                    self.parse_link_decl_stmt() // already consumes ';'
                } else {
                    Err(ParseError::new("Unsupported identifier-led statement"))
                }
            }
            Some(tok) => Err(ParseError::new(format!("Expected statement, found {:?}", tok))),
            None => Err(ParseError::new("Unexpected EOF while parsing statement")),
        }
    }


}