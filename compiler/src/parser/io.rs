use crate::parser::{ParseError, Parser};
use crate::parser::arg::{ArgSpec, ArgValue};
use crate::ast::statement::Statement;
use crate::ast::io::{InputDecl, InputKind, OutputDecl, OutputKind};
use crate::lexer::Token;

impl Parser {
    pub fn parse_input_decl_statement(&mut self) -> Result<Statement, ParseError> {
        self.expect(Token::Input)?;
        let name = self.parse_identifier()?;
        self.expect(Token::Colon)?;

        let kind = match self.peek_token(None) {
            Some(Token::Image) => {
                self.bump();
                let (h, w, channels) = self.parse_image_args()?;
                InputKind::Image(h,w,channels)
            }
            Some(Token::Language) => {
                self.bump();
                let size = self.single_uint_arg()?;
                InputKind::Language(size)
            }
            Some(tok) => return Err(ParseError::new(format!("Expected input type, found {:?}", tok))),
            None => return Err(ParseError::new("Unexpected EOF after input declaration")),
        };

        self.expect(Token::Semi)?;
        Ok(Statement::Input(InputDecl { name, kind }))
    }

    pub fn parse_output_decl_statement(&mut self) -> Result<Statement, ParseError> {
        self.expect(Token::Output)?;
        let name = self.parse_identifier()?;
        self.expect(Token::Colon)?;

        let kind = match self.peek_token(None) {
            Some(Token::Class) => {
                self.bump();
                let size= self.single_uint_arg()?;
                OutputKind::Classifier(size)
            }
            Some(Token::Logits) => {
                self.bump();
                let size = self.single_uint_arg()?;
                OutputKind::Logits(size)
            }
            Some(tok) => return Err(ParseError::new(format!("Expected input type, found {:?}", tok))),
            None => return Err(ParseError::new("Unexpected EOF after output declaration")),
        };

        self.expect(Token::Semi)?;
        Ok(Statement::Output(OutputDecl { name, kind }))
    }

    fn parse_image_args(&mut self) -> Result<(u32, u32, Option<u32>), ParseError> {
        let args = self.parse_args(
            &[ArgSpec::Int, ArgSpec::Int],      // required: h, w
            Some(&[ArgSpec::Int]),              // optional: channels
        )?;

        match args.as_slice() {
            [ArgValue::Int(h), ArgValue::Int(w)] => Ok((*h, *w, None)),
            [ArgValue::Int(h), ArgValue::Int(w), ArgValue::Int(c)] => Ok((*h, *w, Some(*c))),
            _ => Err(ParseError::new("Image(...) expects (int, int[, int])")),
        }
    }
}