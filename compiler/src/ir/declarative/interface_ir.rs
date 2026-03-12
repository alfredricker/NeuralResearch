use super::{AttrBag, EndpointId, GroupId, TopologyExprIr};

#[derive(Debug, Clone)]
pub struct InterfaceIr {
    pub id: EndpointId,
    pub name: String,
    pub direction: InterfaceDirection,
    pub kind: InterfaceKind,
    pub attrs: AttrBag,
}

#[derive(Debug, Clone)]
pub struct ExternalLinkIr {
    pub from: EndpointRef,
    pub to: EndpointRef,
    pub topology: TopologyExprIr,
    pub attrs: AttrBag,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterfaceDirection {
    Input,
    Output,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterfaceKind {
    Image {
        height: u32,
        width: u32,
        channels: Option<u32>,
    },
    Language {
        token_size: u32,
    },
    Classifier {
        classes: u32,
    },
    Logits {
        size: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EndpointRef {
    Interface(EndpointId),
    Group(GroupId),
}
