pub mod token;
pub use token::Token;

use std::ops::Range;
use logos::Logos;

#[derive(Debug, Clone, PartialEq)]
pub struct SpannedToken {
    pub token: Token,
    pub span: Range<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LexError {
    pub span: Range<usize>,
    pub fragment: String,
}

// use logos library to map Token type to vec of spanned token
pub fn lex(source: &str) -> Result<Vec<SpannedToken>, LexError> {
    let mut lexer = Token::lexer(source);
    let mut out = Vec::new();

    while let Some(next) = lexer.next() {
        let span = lexer.span();
        match next {
            Ok(token) => out.push(SpannedToken{ token, span }),
            Err(()) => {
                return Err(LexError {span, fragment: lexer.slice().to_string()});
            }
        }
    }

    Ok(out)
}