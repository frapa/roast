use super::*;

use std::ops::*;

use num;

// Have to do this to cicumvent orphan rules
// for generic types. Man that feels bad.
macro_rules! scalar_left_for_type {
    ($type:ty, $trait:ident, $func:ident, $lambda:expr) => {
        impl $trait<Tensor<$type>> for $type {
            type Output = Tensor<$type>;

            fn $func(self, other: Tensor<$type>) -> Self::Output {
                elemwise2(Tensor::from(self), other, $lambda)
            }
        }
    }
}

// Take your hardcoded list, compiler
macro_rules! scalar_left {
    ($trait:ident, $func:ident, $lambda:expr) => {
        scalar_left_for_type!(u8, $trait, $func, $lambda);
        scalar_left_for_type!(u16, $trait, $func, $lambda);
        scalar_left_for_type!(u32, $trait, $func, $lambda);
        scalar_left_for_type!(u64, $trait, $func, $lambda);
        scalar_left_for_type!(u128, $trait, $func, $lambda);
        scalar_left_for_type!(usize, $trait, $func, $lambda);
        scalar_left_for_type!(i8, $trait, $func, $lambda);
        scalar_left_for_type!(i16, $trait, $func, $lambda);
        scalar_left_for_type!(i32, $trait, $func, $lambda);
        scalar_left_for_type!(i64, $trait, $func, $lambda);
        scalar_left_for_type!(i128, $trait, $func, $lambda);
        scalar_left_for_type!(isize, $trait, $func, $lambda);
        scalar_left_for_type!(f32, $trait, $func, $lambda);
        scalar_left_for_type!(f64, $trait, $func, $lambda);
    }
}

// This is not really needed, but for consistency
// with the above it's here. This time we use real traits
// tough, as that's more flexible and can be used with
// other types as well.
macro_rules! scalar_right {
    ($trait:ident, $func:ident, $lambda:expr) => {
        impl<T> $trait<T> for Tensor<T>
            where
                T: num::Num + Copy,
        {
            type Output = Tensor<T>;

            fn $func(self, other: T) -> Self::Output {
                elemwise2(self, Tensor::from(other), $lambda)
            }
        }
    }
}

scalar_left!(Add, add, |a, b| *a + *b);
scalar_right!(Add, add, |a, b| *a + *b);

impl<T> Add for Tensor<T>
where
    T: num::Num + Copy,
{
    type Output = Tensor<T>;

    fn add(self, other: Self) -> Self::Output {
        elemwise2(self, other.into(), |a, b| *a + *b)
    }
}

scalar_left!(Sub, sub, |a, b| *a - *b);
scalar_right!(Sub, sub, |a, b| *a - *b);

impl<T> Sub for Tensor<T>
where
    T: num::Num + Copy,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        elemwise2(self, other, |a, b| *a - *b)
    }
}

scalar_left!(Mul, mul, |a, b| *a * *b);
scalar_right!(Mul, mul, |a, b| *a * *b);

impl<T> Mul for Tensor<T>
where
    T: num::Num + Copy,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        elemwise2(self, other, |a, b| *a * *b)
    }
}

scalar_left!(Div, div, |a, b| *a / *b);
scalar_right!(Div, div, |a, b| *a / *b);

impl<T> Div for Tensor<T>
where
    T: num::Num + Copy,
{
    type Output = Self;

    fn div(self, other: Self) -> Self {
        elemwise2(self, other, |a, b| *a / *b)
    }
}

scalar_left!(Rem, rem, |a, b| *a % *b);
scalar_right!(Rem, rem, |a, b| *a % *b);

impl<T> Rem for Tensor<T>
where
    T: num::Num + Copy,
{
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        elemwise2(self, other, |a, b| *a % *b)
    }
}

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
