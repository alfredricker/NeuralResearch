pub mod attrs;
pub mod errors;
pub mod graph_ir;
pub mod ids;
pub mod interface_ir;
pub mod lowering;
pub mod module_ir;
pub mod topology;

pub use attrs::{AttrBag, AttrValue};
pub use errors::IrError;
pub use graph_ir::{GraphIr, GroupLinkIr, GroupRole, NodeGroupIr};
pub use ids::{EndpointId, GraphId, GroupId};
pub use interface_ir::{
    EndpointRef, ExternalLinkIr, InterfaceDirection, InterfaceIr, InterfaceKind,
};
pub use module_ir::ModuleIr;
pub use topology::TopologyExprIr;
