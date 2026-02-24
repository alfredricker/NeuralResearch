

#[derive(Debug, Clone)]
pub struct InputDecl {
    pub kind: InputKind,
}

#[derive(Debug, Clone)]
enum InputKind {
    Image(u32, u32),
    RGBImage(u32, u32, Vec<f64>), // vec of channels
    Text(String),
}

#[derive(Debug, Clone)]
pub struct OutputDecl {
    kind: OutputKind,
    method: Option<OutputMethod>,
}

#[derive(Debug, Clone)]
enum OutputKind {
    Classifier(u32),
    Logits(u32),
    Tensor(Vec<u32>),
}

#[derive(Debug, Clone)]
enum OutputMethod {
    Pool,
    Concat,
    Spatial,
    Vote,
}