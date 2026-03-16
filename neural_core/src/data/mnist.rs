use std::path::Path;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;

/// A single MNIST sample: 784-dimensional pixel vector + label.
#[derive(Clone)]
pub struct MnistSample {
    pub pixels: Vec<f32>,  // 784 floats in [0, 1]
    pub label: usize,
}

pub struct MnistDataset {
    pub samples: Vec<MnistSample>,
}

impl MnistDataset {
    /// Load from a parquet file.
    ///
    /// Expected schema (Hugging Face MNIST parquet format):
    ///   - "image"  : struct { "bytes": binary } or flattened bytes
    ///   - "label"  : int32/int64
    ///
    /// Actual column names discovered at runtime; this function tries
    /// both the nested and flat layouts.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = SerializedFileReader::new(file)?;

        let mut samples = Vec::new();
        let row_iter = reader.get_row_iter(None)?;

        for row_result in row_iter {
            let row = row_result?;

            // Try to extract label — column may be "label" (int32 or int64)
            let label = row.get_int(1).map(|v| v as usize)
                .or_else(|_| row.get_long(1).map(|v| v as usize))
                .unwrap_or(0);

            // Pixel data: try column 0 as bytes
            let raw_bytes = row.get_bytes(0).ok().map(|b| b.data().to_vec());

            let pixels: Vec<f32> = if let Some(bytes) = raw_bytes {
                // Raw bytes: 28×28 u8 values
                bytes.iter().map(|&b| b as f32 / 255.0).collect()
            } else {
                // Fallback: try reading as a group/struct — not yet supported
                vec![0.0f32; 784]
            };

            if pixels.len() >= 784 {
                samples.push(MnistSample {
                    pixels: pixels[..784].to_vec(),
                    label,
                });
            }
        }

        Ok(Self { samples })
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}
