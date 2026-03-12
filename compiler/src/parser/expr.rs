use crate::parser::{Parser,ParseError};
use crate::ast::expr::Expr;
use crate::lexer::Token;

impl Parser {
    pub fn parse_value_expr(&mut self) -> Result<Expr, ParseError> {
        let tok = self
            .peek_token(None)
            .cloned()
            .ok_or_else(|| ParseError::new("Unexpected EOF while parsing value expression"))?;

        match tok {
            Token::Integer(v) => {
                self.bump();
                Ok(Expr::Int(v as i64))
            }
            Token::Float(v) => {
                self.bump();
                Ok(Expr::Float(v))
            }

            // topology keywords can be first-class expr values
            Token::Sparse => {
                self.bump();
                // sparse(<float>)
                self.expect(Token::LParen)?;
                let f = self.parse_float()?;
                self.expect(Token::RParen)?;
                Ok(Expr::Topology(crate::ast::expr::TopologyExpr::Sparse(f)))
            }
            Token::Identity => {
                self.bump();
                Ok(Expr::Topology(crate::ast::expr::TopologyExpr::Identity))
            }
            Token::Dense => {
                self.bump();
                Ok(Expr::Topology(crate::ast::expr::TopologyExpr::Dense))
            }
            Token::WeightedSum => {
                self.bump();
                Ok(Expr::Topology(crate::ast::expr::TopologyExpr::WeightedSum))
            }
            Token::Nodes => {
                self.bump();
                let args = self.parse_generic_args()?;
                Ok(Expr::Call(crate::ast::expr::CallExpr {
                    name: "nodes".to_string(),
                    args,
                }))
            }

            Token::Ident(_) => {
                let name = self.parse_identifier()?;
                if matches!(self.peek_token(None), Some(Token::LParen)) {
                    // call: name(...)
                    let args = self.parse_generic_args()?; // helper below
                    Ok(Expr::Call(crate::ast::expr::CallExpr { name, args }))
                } else {
                    // variable reference
                    Ok(Expr::Ident(name))
                }
            }

            other => Err(ParseError::new(format!(
                "Expected value expression, found {:?}",
                other
            ))),
        }
    }
}