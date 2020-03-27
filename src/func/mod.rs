use super::*;

use std::ops::*;
use std::borrow::Borrow;

use num;

// pub fn elemwise1<T>(a: impl Borrow<Tensor<T>>, func: impl Fn(&T) -> T) -> Tensor<T>
// where
//     T: num::Num + Copy,
// {
//     let a = a.borrow();
//
//     let result_shape = a.shape.clone();
//     let mut result_data = Vec::with_capacity(result_shape.size);
//
//     for val in a.data.iter() {
//         result_data.push(func(val));
//     }
//
//     return Tensor::new(result_shape, result_data);
// }

pub fn elemwise2<T>(a: Tensor<T>, b: Tensor<T>, func: fn(&T, &T) -> T) -> Tensor<T>
where
    T: num::Num + Copy,
{
    let result_shape = a.shape.clone();
    let mut result_data = Vec::with_capacity(result_shape.size);

    for (val_a, val_b) in a.data.iter().zip(b.data.iter()) {
        result_data.push(func(val_a, val_b));
    }

    return Tensor::new(result_shape, result_data);
}
