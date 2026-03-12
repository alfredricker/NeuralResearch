use super::{AttrBag, GraphId, GroupId, TopologyExprIr};

#[derive(Debug, Clone)]
pub struct GraphIr {
    pub id: GraphId,
    pub name: Option<String>,
    pub parent: Option<GraphId>,
    pub groups: Vec<NodeGroupIr>,
    pub links: Vec<GroupLinkIr>,
    pub attrs: AttrBag,
}

#[derive(Debug, Clone)]
pub struct NodeGroupIr {
    pub id: GroupId,
    pub graph: GraphId,
    pub name: String,
    pub count: u32,
    pub role: GroupRole,
    pub attrs: AttrBag,
}

#[derive(Debug, Clone)]
pub struct GroupLinkIr {
    pub from: GroupId,
    pub to: GroupId,
    pub topology: TopologyExprIr,
    pub attrs: AttrBag,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupRole {
    Hidden,
    Input,
    Output,
    Internal,
}
