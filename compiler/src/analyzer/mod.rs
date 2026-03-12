pub mod compiler_error;

use crate::ast::Program;
use crate::ir::{declarative::IrError, executable::ExecutableModule, ModuleIr};
use compiler_error::CompilerError;

pub struct Analyzer {
    errors: Vec<CompilerError>,
}

impl Analyzer {
    pub fn new() -> Self {
        Self { errors: Vec::new() }
    }

    pub fn lower_program(
        &mut self,
        program: &Program,
    ) -> Result<(ModuleIr, ExecutableModule), Vec<CompilerError>> {
        let decl = match ModuleIr::from_program(program) {
            Ok(ir) => ir,
            Err(errors) => {
                self.errors.extend(map_ir_errors(errors));
                return Err(self.errors.clone());
            }
        };

        let exec = match ExecutableModule::from_declarative(&decl) {
            Ok(ir) => ir,
            Err(errors) => {
                self.errors.extend(map_ir_errors(errors));
                return Err(self.errors.clone());
            }
        };

        Ok((decl, exec))
    }
}

fn map_ir_errors(errors: Vec<IrError>) -> Vec<CompilerError> {
    errors
        .into_iter()
        .map(|err| CompilerError::new(err.message))
        .collect()
}
