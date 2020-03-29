use super::*;

use std::ops::*;

use num;

// Have to do this to circumvent orphan rules
// for generic types. Man that feels bad.
macro_rules! scalar_op_left_for_type {
    ($type:ty, $trait:ident, $func_name:ident, $func:expr) => {
        impl $trait<&Tensor<$type>> for $type {
            type Output = Tensor<$type>;

            fn $func_name(self, other: &Tensor<$type>) -> Self::Output  {
                elemwise(&[&Tensor::from(self), other], $func)
            }
        }
    }
}

// Take your hardcoded list, compiler
macro_rules! scalar_op_left {
    ($trait:ident, $func_name:ident, $func:expr) => {
        scalar_op_left_for_type!(u8, $trait, $func_name, $func);
        scalar_op_left_for_type!(u16, $trait, $func_name, $func);
        scalar_op_left_for_type!(u32, $trait, $func_name, $func);
        scalar_op_left_for_type!(u64, $trait, $func_name, $func);
        scalar_op_left_for_type!(u128, $trait, $func_name, $func);
        scalar_op_left_for_type!(usize, $trait, $func_name, $func);
        scalar_op_left_for_type!(i8, $trait, $func_name, $func);
        scalar_op_left_for_type!(i16, $trait, $func_name, $func);
        scalar_op_left_for_type!(i32, $trait, $func_name, $func);
        scalar_op_left_for_type!(i64, $trait, $func_name, $func);
        scalar_op_left_for_type!(i128, $trait, $func_name, $func);
        scalar_op_left_for_type!(isize, $trait, $func_name, $func);
        scalar_op_left_for_type!(f32, $trait, $func_name, $func);
        scalar_op_left_for_type!(f64, $trait, $func_name, $func);
    }
}

// This is not really needed, but for consistency and brevity
// with the above it's here. This time we use real traits
// tough, as that's more flexible and can be used with
// other types as well.
macro_rules! scalar_op_right {
    ($trait:ident, $func_name:ident, $func:expr) => {
        impl<T> $trait<T> for &Tensor<T>
            where
                T: num::Num + Copy + 'static,
        {
            type Output = Tensor<T>;

            fn $func_name(self, other: T) -> Self::Output  {
                elemwise(&[self, &Tensor::from(other)], $func)
            }
        }
    }
}

// This is also not needed, but let us avoid repeating too much code
macro_rules! tensor_op {
    ($trait:ident, $func_name:ident, $func:expr) => {
        impl<T> $trait for &Tensor<T>
            where
                T: num::Num + Copy + 'static,
        {
            type Output = Tensor<T>;

            fn $func_name(self, other: Self) -> Self::Output {
                elemwise(&[self, other], $func)
            }
        }
    }
}

// And finally a combination of all 3 things, to further shorten
// the code
macro_rules! op {
    ($trait:ident, $func_name:ident, $func:expr) => {
        scalar_op_left!($trait, $func_name, $func);
        scalar_op_right!($trait, $func_name, $func);
        tensor_op!($trait, $func_name, $func);
    }
}

op!(Add, add, AddFunction());
op!(Sub, sub, SubFunction());
op!(Mul, mul, MulFunction());
op!(Div, div, DivFunction());
op!(Rem, rem, RemFunction());

impl<T> Neg for &Tensor<T>
where
    T: num::Num + Copy + Neg<Output=T> + 'static,
{
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        elemwise(&[&self], NegFunction())
    }
}

#[cfg(test)]
mod math_tests;
