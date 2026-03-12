#[derive(Debug, Clone)]
pub struct CompilerError {
    pub message: String,
}

impl CompilerError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}
