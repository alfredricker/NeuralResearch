use super::{ExecExternalLink, ExecutableGraph};

#[derive(Debug, Clone, Default)]
pub struct ExecutableModule {
    pub graphs: Vec<ExecutableGraph>,
    pub links: Vec<ExecExternalLink>,
}
