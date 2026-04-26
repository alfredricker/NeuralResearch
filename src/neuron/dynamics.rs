pub fn update_alpha(t: u16, alpha: &mut u8) {
    if t > 0 {
        *alpha = alpha.saturating_sub(1);
    }
}

pub fn update_beta(t: u16, beta: &mut u8) {
    if t > 0 {
        *beta = beta.saturating_sub(BETA_DECAY);
    }
}