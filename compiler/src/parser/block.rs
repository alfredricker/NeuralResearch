use crate::parser::{ParseError, Parser};
use crate::ast::block::{Block, BlockKind};
use crate::lexer::Token;

impl Parser {
    pub fn parse_block(&mut self) -> Result<Block, ParseError> {
        let kind = match self.peek_token(None) {
            Some(Token::Graph) => { self.bump(); BlockKind::Graph }
            Some(Token::Subgraph) => { self.bump(); BlockKind::Subgraph }
            Some(Token::Top) => { self.bump(); BlockKind::Top }
            Some(Token::Learn) => { self.bump(); BlockKind::Learn }
            Some(Token::Display) => { self.bump(); BlockKind::Display }
            Some(tok) => return Err(ParseError::new(format!("Expected block keyword, found {:?}", tok))),
            None => return Err(ParseError::new("Unexpected EOF while parsing block header")),
        };
    
        self.expect(Token::LBrace)?;
    
        let mut items = Vec::new();
        while !matches!(self.peek_token(None), Some(Token::RBrace)) {
            if self.peek_token(None).is_none() {
                return Err(ParseError::new("Unclosed block: expected `}` before EOF"));
            }
            items.push(self.parse_item()?);
        }
    
        self.expect(Token::RBrace)?;
    
        Ok(Block { kind, items })
    }

}