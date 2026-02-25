use crate::lexer::Token;
use crate::parser::error::ParseError;
use crate::parser::Parser;

#[derive(Debug, Clone)]
pub enum ArgSpec {
    Float,
    Int,
    Ident,
    Optional(Box<ArgSpec>),
}

#[derive(Debug, Clone)]
pub enum ArgValue {
    Float(f64),
    Int(u32),
    Ident(String),
}

impl Parser {
    pub fn parse_paren_args(
        &mut self,
        req_spec: &[ArgSpec],
        opt_spec: Option<&[ArgSpec]>,
    ) -> Result<Vec<ArgValue>, ParseError> {

        // expect opening parenthesis
        self.expect(Token::LParen)?;
        let mut args = Vec::new();

        // required args: must all be present
        for (i, arg) in req_spec.iter().enumerate() {
            if i > 0 {
                self.expect(Token::Comma)?;
            }
            let value = self.match_and_consume_arg(arg)?;
            args.push(value);
        }

        // optional args: parse only if comma + value exists
        if let Some(opt_spec) = opt_spec {
            for arg in opt_spec.iter() {
                if matches!(self.peek_token(None), Some(Token::Comma)) {
                    self.expect(Token::Comma)?;
                    let value = self.match_and_consume_arg(arg)?;
                    args.push(value);
                } else {
                    break;
                }
            }
        }

        self.expect(Token::RParen)?;
        Ok(args)
    }

    fn match_and_consume_arg(&mut self, arg: &ArgSpec) -> Result<ArgValue, ParseError> {
        match arg {
            ArgSpec::Float => Ok(ArgValue::Float(self.parse_float()?)),
            ArgSpec::Int => Ok(ArgValue::Int(self.parse_int()?)),
            ArgSpec::Ident => Ok(ArgValue::Ident(self.parse_identifier()?)),

            // Keep this simple for now until you add a "missing optional" representation.
            ArgSpec::Optional(inner) => self.match_and_consume_arg(inner),
        }
    }

    fn parse_int(&mut self) -> Result<u32, ParseError> {
        let tok = self.bump().ok_or_else(|| ParseError::new("Unexpected EOF"))?;
        match &tok.token {
            Token::Integer(v) => Ok(*v),
            other => Err(ParseError::with_span(
                format!("Expected integer, found {:?}", other),
                tok.span.clone(),
            )),
        }
    }

    fn parse_float(&mut self) -> Result<f64, ParseError> {
        let tok = self.bump().ok_or_else(|| ParseError::new("Unexpected EOF"))?;
        match &tok.token {
            Token::Float(v) => Ok(*v),
            Token::Integer(v) => Ok(*v as f64),
            other => Err(ParseError::with_span(
                format!("Expected float, found {:?}", other),
                tok.span.clone(),
            )),
        }
    }

    pub fn parse_identifier(&mut self) -> Result<String, ParseError> {
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