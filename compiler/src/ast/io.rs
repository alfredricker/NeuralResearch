#[derive(Debug, Clone)]
pub enum InputKind {
    Image(u32, u32, Option<u32>), // image h x w with optional rgb channels
    Language(u32), //token size
}

#[derive(Debug, Clone)]
pub enum OutputKind {
    Classifier(u32),
    Logits(u32),
}

#[derive(Debug, Clone)]
pub enum OutputMethod {
    Pool,
    Concat,
    Spatial,
    Vote,
}