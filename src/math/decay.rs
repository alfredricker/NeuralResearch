/// Applies an ultra-fast O(1) base-2 exponential decay approximation.
///
/// * `v` - The initial value (u16)
/// * `t` - The number of elapsed time steps since last event (u16)
/// * `k` - The half-life exponent. The value will halve every 2^k steps.
/// * Voltage is decayed on branches and at the soma by some k-dependent amount every time step.
pub fn shift_decay(v: u16, t: u16, k: u16) -> u16 {
    let shifts = t >> k;
    
    if shifts >= 16 {
        return 0;
    }

    let remainder = t & ((1 << k) - 1);

    let v_current = v >> shifts;
    let v_next = v_current >> 1; 

    // Both of these are guaranteed to fit in a u16
    let diff = v_current - v_next;

    // CRITICAL: We must cast to u32 for the multiplication to prevent overflow!
    let drop = ((diff as u32 * remainder as u32) >> k) as u16;

    v_current - drop
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn print_decay_table() {
        let cases: &[(u16, u16, u16)] = &[
            // (v,   t,   k)
            (200,   0,   2),
            (200,   4,   2),
            (200,   8,   2),
            (200,  16,   2),
            (200,  64,   2),
            (200,   0,   3),
            (200,   8,   3),
            (200,  16,   3),
            (200,  64,   3),
            (1000,  0,   4),
            (1000, 16,   4),
            (1000, 32,   4),
            (1000, 64,   4),
            (1000,128,   4),
        ];

        println!("\n{:>6}  {:>6}  {:>4}  {:>6}", "v", "t", "k", "result");
        println!("{}", "-".repeat(30));
        for &(v, t, k) in cases {
            println!("{:>6}  {:>6}  {:>4}  {:>6}", v, t, k, shift_decay(v, t, k));
        }
    }
}