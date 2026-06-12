//! Minimal MNIST `idx-ubyte` reader — no external crates.
//!
//! Parses the big-endian idx header by hand and returns flat, row-major `u8` frames. Expects the
//! files to be **decompressed**: the canonical downloads are gzip'd (`*-ubyte.gz`); `gunzip` them
//! first. Errors are plain strings — this is a personal research CLI, not a library.

use std::path::Path;

/// A loaded idx3 image set: `images[i]` is `rows * cols` pixels, row-major, intensity `0..=255`.
pub struct MnistImages {
    pub rows: usize,
    pub cols: usize,
    pub images: Vec<Vec<u8>>,
}

fn read_u32_be(bytes: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// Load an `idx3-ubyte` image file (magic `0x00000803`).
pub fn load_images(path: impl AsRef<Path>) -> Result<MnistImages, String> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    if bytes.len() < 16 {
        return Err(format!("{}: too short to be an idx3 image file", path.display()));
    }
    let magic = read_u32_be(&bytes, 0);
    if magic != 0x0000_0803 {
        return Err(format!(
            "{}: bad idx3 magic {magic:#010x} (expected 0x00000803 — is the file still gzip'd? gunzip it first)",
            path.display()
        ));
    }
    let n = read_u32_be(&bytes, 4) as usize;
    let rows = read_u32_be(&bytes, 8) as usize;
    let cols = read_u32_be(&bytes, 12) as usize;
    let stride = rows * cols;
    let expected = 16 + n * stride;
    if bytes.len() < expected {
        return Err(format!(
            "{}: truncated — header declares {n} images of {rows}x{cols} ({expected} bytes), file is {} bytes",
            path.display(),
            bytes.len()
        ));
    }
    let images = (0..n)
        .map(|i| {
            let off = 16 + i * stride;
            bytes[off..off + stride].to_vec()
        })
        .collect();
    Ok(MnistImages { rows, cols, images })
}

/// Load an `idx1-ubyte` label file (magic `0x00000801`); one `u8` digit per image.
pub fn load_labels(path: impl AsRef<Path>) -> Result<Vec<u8>, String> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    if bytes.len() < 8 {
        return Err(format!("{}: too short to be an idx1 label file", path.display()));
    }
    let magic = read_u32_be(&bytes, 0);
    if magic != 0x0000_0801 {
        return Err(format!(
            "{}: bad idx1 magic {magic:#010x} (expected 0x00000801 — is the file still gzip'd? gunzip it first)",
            path.display()
        ));
    }
    let n = read_u32_be(&bytes, 4) as usize;
    if bytes.len() < 8 + n {
        return Err(format!(
            "{}: truncated — header declares {n} labels, file has {} after the header",
            path.display(),
            bytes.len() - 8
        ));
    }
    Ok(bytes[8..8 + n].to_vec())
}
