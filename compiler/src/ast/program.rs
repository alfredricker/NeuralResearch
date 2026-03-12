use crate::ast::Item;
#[derive(Debug, Clone)]
pub struct Program {
    items: Vec<Item>,
}

impl Program {
    pub fn new() -> Self {
        Self{
            items: Vec::new(),
        }
    }

    pub fn push_item(&mut self, item: Item) {
        self.items.push(item);
    }

    pub fn items(&self) -> &[Item] {
        &self.items
    }
}