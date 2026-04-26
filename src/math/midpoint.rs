fn midpoint<T: MidPoint>() -> T {
    T::mid()
}

pub trait MidPoint {
    fn mid() -> Self;
}

macro_rules! impl_midpoint {
    ($($t:ty),*) => {
        $(
            impl MidPoint for $t {
                fn mid() -> Self {
                    1 << (<$t>::BITS - 1)
                }
            }
        )*
    }
}             
impl_midpoint!(u8, u16, u32, u64, u128);