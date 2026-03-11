use crate::parser::{ParseError, Parser};
use crate::ast::statement::Statement;
use crate::lexer::Token;

impl Parser {
    pub fn parse_var_decl_stmt(&mut self) -> Result<Statement, ParseError> {
        let ident = self.parse_identifier()?;
        self.expect(Token::Colon);
    }
}