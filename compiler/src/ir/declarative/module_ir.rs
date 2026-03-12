use super::{ExternalLinkIr, GraphIr, InterfaceIr};

#[derive(Debug, Clone, Default)]
pub struct ModuleIr {
    pub graphs: Vec<GraphIr>,
    pub interfaces: Vec<InterfaceIr>,
    pub links: Vec<ExternalLinkIr>,
}
