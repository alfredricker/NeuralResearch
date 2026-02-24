use crate::parser::{Parser,ParseError};

#[derive(Debug, Clone)]
pub enum ArgSpec {
    Float,
    Int,
    Ident,
    // Optional arg wrapper
    Optional(Box<ArgSpec>),
}

#[derive(Debug, Clone)]
pub enum ArgValue {
    Float(f64),
    Int(u32),
    Ident(String),
}

// needs to be finished -- handle opt_spec, move match arg to helper
impl Parser {
    pub fn parse_paren_args(&mut self, req_spec: &[ArgSpec], opt_spec: Option<&[ArgSpec]>) -> Result<Vec<ArgValue>, ParseError> {
        self.expect(Token::LParen)?; // make sure starts with an LParen
        let mut args = Vec::new();
        for (i, arg) in req_spec {
            if i > 0 {
                self.expect(Token::Comma)?;
            }
            value = self._match_and_consume_arg(arg)?;
            args.push(value);
        }
        if let Some(opt_spec) = opt_spec {
            for (i, arg) in opt_spec {
                if i > 0 {
                    self.expect(Token::Comma)?;
                }
                value = self._match_and_consume_arg(arg)?;
                args.push(value);
            }
        }
        self.expect(Token::RParen)?;
        Ok(args)
    }

    fn _match_and_consume_arg(&mut self, arg: ArgSpec) -> Result<ArgValue, ParseError>{
        match arg {
            ArgSpec::Float => {
                let value = self.parse_float()?;
                Ok(ArgValue::Float(value))
            }
            ArgSpec::Int => {
                let value = self.parse_int()?;
                Ok(ArgValue::Int(value))
            }
            ArgSpec::Ident => {
                let value = self.parse_identifier()?;
                Ok(ArgValue::Ident(value))
            }
            ArgSpec::Optional(spec) => {
                Err(ParseError::with_span(
                    "Optional arguments not supported yet",
                    self.pos.span.clone(),
                ))
            }
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
            Token::Integer(v) => Ok(*v as f64), // optional widening
            other => Err(ParseError::with_span(
                format!("Expected float, found {:?}", other),
                tok.span.clone(),
            )),
        }
    }
    
    fn parse_identifier(&mut self) -> Result<String, ParseError> {
        let tok = self.bump().ok_or_else(|| ParseError::new("Unexpected EOF"))?;
        match &tok.token {
            Token::Ident(name) => Ok(name.clone()),
            other => Err(ParseError::with_span(
                format!("Expected identifier, found {:?}", other),
                tok.span.clone(),
            )),
        }
    }
}