use crate::ast::expr::Expr;
#[derive(Debug, Clone)]
pub struct VarDecl {
    pub name: String,
    pub value: Expr,
}

impl VarDecl {
    pub fn new(name: String, value: Expr) -> Self {
        Self {
            name,
            value
        }
    }
}
