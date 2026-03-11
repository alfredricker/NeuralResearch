use crate::parser::{ParseError, Parser};
use crate::ast::statement::Statement;
use crate::ast::var::VarDecl;
use crate::lexer::Token;


impl Parser {
    pub fn parse_var_decl_stmt(&mut self) -> Result<Statement, ParseError> {
        let ident = self.parse_identifier()?;
        self.expect(Token::Equal)?;
        let expression =self.parse_value_expr()?;

        let var_decl = VarDecl::new(ident, expression);
        Ok(Statement::Var(var_decl))
    }
}