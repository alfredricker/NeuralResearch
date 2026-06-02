/// Applies an ultra-fast O(1) base-2 exponential decay approximation.
///
/// * `v` - The initial value (u16)
/// * `t` - The number of elapsed time steps since last event (u16)
/// * `k` - The half-life exponent. The value will halve every 2^k steps.
/// * Voltage is decayed on branches and at the soma by some k-dependent amount every time step.
pub fn shift_decay(v: u16, t: u16, k: u8) -> u16 {
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

pub fn shift_decay_u8(v: u8, t: u16, k: u8) -> u8 {
    shift_decay(v as u16, t, k) as u8
}

pub fn shift_decay_i8(v: i8, t: u16, k: u8) -> i8 {
    let sign = if v < 0 { -1 } else { 1 };
    let abs_v = v.abs() as u16;
    let decayed = shift_decay(abs_v, t, k);
    (decayed as i16 * sign) as i8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_decay_no_elapsed_unchanged() {
        assert_eq!(shift_decay(1000, 0, 4), 1000);
    }

    #[test]
    fn shift_decay_u8_no_elapsed_unchanged() {
        assert_eq!(shift_decay_u8(200, 0, 8), 200);
    }

    #[test]
    fn shift_decay_u8_one_half_life() {
        // k=8 → halves every 2^8=256 ticks; remainder=0 so exact
        assert_eq!(shift_decay_u8(200, 256, 8), 100);
    }

    #[test]
    fn shift_decay_u8_zero_val_stays_zero() {
        assert_eq!(shift_decay_u8(0, 1000, 8), 0);
    }

    #[test]
    fn shift_decay_u8_sixteen_half_lives_returns_zero() {
        // 16+ half-lives (16 * 256 = 4096) → shifts >= 16, returns 0
        assert_eq!(shift_decay_u8(255, 4096, 8), 0);
    }

    #[test]
    fn shift_decay_i8_no_elapsed_unchanged() {
        assert_eq!(shift_decay_i8(100, 0, 4), 100);
        assert_eq!(shift_decay_i8(-100, 0, 4), -100);
    }

    #[test]
    fn print_decay_table() {
        let cases: &[(u16, u16, u16)] = &[
            // (v,   t,   k)
            (200,   0,   2),
            (200,   4,   2),
            (200,   8,   2),
            (200,  16,   2),
            (200,  64,   2),
            (1000,  0,   4),
            (1000, 16,   4),
            (1000, 32,   4),
            (1000, 64,   4),
            (1000,128,   4),
            (400, 4, 8),
            (400, 400, 8),
            (400, 4000, 8),
        ];

        println!("\n{:>6}  {:>6}  {:>4}  {:>6}", "v", "t", "k", "result");
        println!("{}", "-".repeat(30));
        for &(v, t, k) in cases {
            println!("{:>6}  {:>6}  {:>4}  {:>6}", v, t, k, shift_decay(v, t, k.try_into().unwrap()));
        }
    }
}