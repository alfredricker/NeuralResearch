pub mod text;
use text::TextTokens;
enum InputKind {
    Image(u32, u32),
    RGBImage(u32, u32, Vec<f64>), // vec of channels
    Text(TextTokens),
}
