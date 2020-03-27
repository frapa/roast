use super::*;

use std::ops::*;

use num;

// Have to do this to circumvent orphan rules
// for generic types. Man that feels bad.
macro_rules! scalar_op_left_for_type {
    ($type:ty, $trait:ident, $func:ident, $lambda:expr) => {
        impl $trait<&Tensor<$type>> for $type {
            type Output = Tensor<$type>;

            fn $func(self, other: &Tensor<$type>) -> Self::Output  {
                elemwise2(&Tensor::from(self), other, $lambda)
            }
        }
    }
}

// Take your hardcoded list, compiler
macro_rules! scalar_op_left {
    ($trait:ident, $func:ident, $lambda:expr) => {
        scalar_op_left_for_type!(u8, $trait, $func, $lambda);
        scalar_op_left_for_type!(u16, $trait, $func, $lambda);
        scalar_op_left_for_type!(u32, $trait, $func, $lambda);
        scalar_op_left_for_type!(u64, $trait, $func, $lambda);
        scalar_op_left_for_type!(u128, $trait, $func, $lambda);
        scalar_op_left_for_type!(usize, $trait, $func, $lambda);
        scalar_op_left_for_type!(i8, $trait, $func, $lambda);
        scalar_op_left_for_type!(i16, $trait, $func, $lambda);
        scalar_op_left_for_type!(i32, $trait, $func, $lambda);
        scalar_op_left_for_type!(i64, $trait, $func, $lambda);
        scalar_op_left_for_type!(i128, $trait, $func, $lambda);
        scalar_op_left_for_type!(isize, $trait, $func, $lambda);
        scalar_op_left_for_type!(f32, $trait, $func, $lambda);
        scalar_op_left_for_type!(f64, $trait, $func, $lambda);
    }
}

// This is not really needed, but for consistency and brevity
// with the above it's here. This time we use real traits
// tough, as that's more flexible and can be used with
// other types as well.
macro_rules! scalar_op_right {
    ($trait:ident, $func:ident, $lambda:expr) => {
        impl<T> $trait<T> for &Tensor<T>
            where
                T: num::Num + Copy,
        {
            type Output = Tensor<T>;

            fn $func(self, other: T) -> Self::Output  {
                elemwise2(self, &Tensor::from(other), $lambda)
            }
        }
    }
}

// This is also not needed, but let us avoid repeating too much
// code
macro_rules! tensor_op {
    ($trait:ident, $func:ident, $lambda:expr) => {
        impl<T> $trait for &Tensor<T>
            where
                T: num::Num + Copy,
        {
            type Output = Tensor<T>;

            fn $func(self, other: Self) -> Self::Output {
                elemwise2(self, other, $lambda)
            }
        }
    }
}

// And finally a combination of all 3 things, to further shorten
// the code
macro_rules! op {
    ($trait:ident, $func:ident, $lambda:expr) => {
        scalar_op_left!($trait, $func, $lambda);
        scalar_op_right!($trait, $func, $lambda);
        tensor_op!($trait, $func, $lambda);
    }
}

op!(Add, add, |a, b| *a + *b);
op!(Sub, sub, |a, b| *a - *b);
op!(Mul, mul, |a, b| *a * *b);
op!(Div, div, |a, b| *a / *b);
op!(Rem, rem, |a, b| *a % *b);

// impl<T> Neg for Tensor<T>
// where
//     T: num::Num + Copy + Neg,
// {
//     type Output = Self;
//
//     fn neg(self) -> Selt {
//         elemwise1(self, |a| -*a)
//     }
// }

#[cfg(test)]
mod math_tests;
