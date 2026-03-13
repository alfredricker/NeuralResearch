use std::fs;

fn load_graph(path: &str) -> Result<(usize, Vec<u32>, Vec<u32>, Vec<f32>), Box<dyn std::error::Error>> {
    let data = fs::read(path)?;
    let mut pos = 0;

    // Read node_count and edge_count
    let node_count = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
    pos += 4;
    let edge_count = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
    pos += 4;

    // Read offsets
    let mut offsets = Vec::with_capacity(node_count + 1);
    for _ in 0..=node_count {
        let val = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
        offsets.push(val);
        pos += 4;
    }

    // Read targets
    let mut targets = Vec::with_capacity(edge_count);
    for _ in 0..edge_count {
        let val = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
        targets.push(val);
        pos += 4;
    }

    // Read weights
    let mut weights = Vec::with_capacity(edge_count);
    for _ in 0..edge_count {
        let val = f32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
        weights.push(val);
        pos += 4;
    }

    Ok((node_count, offsets, targets, weights))
}

fn tick(
    activations: &mut [f32],
    input_buffer: &mut [f32],
    offsets: &[u32],
    targets: &[u32],
    weights: &[f32],
    active_set: &mut Vec<u32>,
    next_set: &mut Vec<u32>,
) {
    next_set.clear();
    for &node in active_set.iter() {
        let node_idx = node as usize;
        let s = offsets[node_idx] as usize;
        let e = offsets[node_idx + 1] as usize;
        for i in s..e {
            let t = targets[i] as usize;
            input_buffer[t] += activations[node_idx] * weights[i];
            next_set.push(targets[i]);
        }
    }
    // Apply and zero
    for &node in next_set.iter() {
        let n = node as usize;
        activations[n] = input_buffer[n];
        input_buffer[n] = 0.0;
    }
    std::mem::swap(active_set, next_set);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (node_count, offsets, targets, weights) = load_graph("graph.bin")?;
    let edge_count = weights.len();

    let mut activations = vec![0.0_f32; node_count];
    let mut input_buffer = vec![0.0_f32; node_count];
    let mut active_set = vec![0];
    let mut next_set = Vec::new();

    for _step in 0..3 {
        tick(
            &mut activations,
            &mut input_buffer,
            &offsets,
            &targets,
            &weights,
            &mut active_set,
            &mut next_set,
        );
    }

    let sum: f32 = activations.iter().copied().sum();
    println!("nodes={} edges={} activation_sum={:.4}", node_count, edge_count, sum);

    Ok(())
}
