pub mod ffn;
pub mod pipeline;
pub mod data;

pub use ffn::FeedForwardNet;
pub use pipeline::{Model, PipelineResult, run_pipeline};
pub use data::{MnistDataset, MnistSample};
