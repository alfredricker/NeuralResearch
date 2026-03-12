#[derive(Debug, Clone)]
pub struct IrError {
    pub message: String,
}

impl IrError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}
