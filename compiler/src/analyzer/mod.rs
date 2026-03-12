use crate::ast::Program;

pub struct Analyzer {
    // Keep track of declared variables, nodes, scopes, etc.
    symbol_table: HashMap<String, TypeInfo>,
    errors: Vec<Error>,
}

// template autocomplete code
impl Analyzer {
    pub fn new() -> Self {
        Self {
            symbol_table: HashMap::new(),
            errors: Vec::new(),
        }
    }

    pub fn analyze(&mut self, program: &Program) {
        for statement in &program.statements {
            self.analyze_statement(statement);
        }
    }

    fn analyze_statement(&mut self, statement: &Statement) {
        match statement {
            Statement::VarDecl(var_decl) => self.analyze_var_decl(var_decl),
            Statement::ExprStmt(expr_stmt) => self.analyze_expr_stmt(expr_stmt),
            Statement::ReturnStmt(return_stmt) => self.analyze_return_stmt(return_stmt),
        }
    }

    fn analyze_var_decl(&mut self, var_decl: &VarDecl) {
        self.symbol_table
            .insert(var_decl.name.clone(), var_decl.type_.clone());
    }

    fn analyze_expr_stmt(&mut self, expr_stmt: &ExprStmt) {
        self.analyze_expr(&expr_stmt.expr);
    }

    fn analyze_return_stmt(&mut self, return_stmt: &ReturnStmt) {
        self.analyze_expr(&return_stmt.expr);
    }

    fn analyze_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::BinaryExpr(binary_expr) => self.analyze_binary_expr(binary_expr),
            Expr::UnaryExpr(unary_expr) => self.analyze_unary_expr(unary_expr),
            Expr::LiteralExpr(literal_expr) => self.analyze_literal_expr(literal_expr),
            Expr::VariableExpr(variable_expr) => self.analyze_variable_expr(variable_expr),
        }
    }

    fn analyze_binary_expr(&mut self, binary_expr: &BinaryExpr) {
        self.analyze_expr(&binary_expr.left);
        self.analyze_expr(&binary_expr.right);
    }

    fn analyze_unary_expr(&mut self, unary_expr: &UnaryExpr) {
        self.analyze_expr(&unary_expr.expr);
    }

    fn analyze_literal_expr(&mut self, literal_expr: &LiteralExpr) {
        // No analysis needed for literals
    }

    fn analyze_variable_expr(&mut self, variable_expr: &VariableExpr) {
        if !self.symbol_table.contains_key(&variable_expr.name) {
            self.errors.push(Error::new(
                format!("Variable '{}' not declared", variable_expr.name),
                variable_expr.span,
            ));
        }
    }
}
