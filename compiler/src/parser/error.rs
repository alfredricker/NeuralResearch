#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub span: Option<std::ops::Range<usize>>,
}

impl ParseError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            span: None,
        }
    }

    pub fn with_span(message: impl Into<String>, span: std::ops::Range<usize>) -> Self {
        Self {
            message: message.into(),
            span: Some(span),
        }
    }
}