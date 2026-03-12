use crate::parser::{ParseError, Parser};
use crate::parser::arg::{ArgSpec,ArgValue};
use crate::ast::{link::{LinkDecl,Topology}, statement::Statement};
use crate::lexer::Token;

impl Parser {
    pub fn parse_link_decl_stmt(&mut self) -> Result<Statement, ParseError> {
        let from = self.parse_identifier()?;
        self.expect(Token::Arrow)?;
        let to = self.parse_identifier()?;
        self.expect(Token::Colon)?;
        let topology = self.parse_topology()?;
    
        // If statement parsers own semicolon consumption:
        self.expect(Token::Semi)?;
    
        Ok(Statement::Link(LinkDecl { from, to, topology }))
    }

    pub fn parse_topology(&mut self) -> Result<Topology, ParseError> {
        let spanned = self.bump().ok_or_else(|| ParseError::new("Unexpected EOF"))?;

        match &spanned.token {
            Token::Identity => {
                Ok(Topology::Identity)
            }
            Token::Dense => {
                Ok(Topology::Dense)
            }
            Token::WeightedSum => {
                Ok(Topology::WeightedSum)
            }
            // need to parse arguments from the sparse function
            Token::Sparse => {
                let sparse_arg_specs = [ArgSpec::Float];
                let args = self.parse_args(&sparse_arg_specs, None)?;

                let sparsity = match args.as_slice() {
                    [ArgValue::Float(v)] => *v,
                    _ => {
                        return Err(ParseError::new("Sparse() expects one float argument"))
                    }
                };
                Ok(Topology::Sparse(sparsity))
            }
            _ => {
                Err(ParseError::new("Expected a topology token"))
            }
        }
    }
}