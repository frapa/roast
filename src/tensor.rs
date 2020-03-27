use super::*;

use std::ops::*;

use num;


#[derive(Debug, PartialEq)]
pub struct Tensor<T>
where
    T: num::Num + Copy,
{
    pub data: Box<[T]>,
    pub shape: Shape,
    pub track_gradient: bool,
}

impl<T> Tensor<T>
where
    T: num::Num + Copy,
{
    pub fn new(shape: impl Into<Shape>, data: Vec<T>) -> Self {
        Self {
            data: data.into_boxed_slice(),
            shape: shape.into(),
            track_gradient: false,
        }
    }

    pub fn empty(shape: impl Into<Shape>) -> Self {
        let sh: Shape = shape.into();
        let size = sh.size;
        Self::new(sh, Vec::with_capacity(size))
    }

    pub fn full(shape: impl Into<Shape>, val: T) -> Self {
        let sh: Shape = shape.into();
        let size = sh.size;
        Self::new(sh, vec![val; size])
    }

    pub fn zeros(shape: impl Into<Shape>) -> Self {
        Self::full(shape, num::Zero::zero())
    }

    pub fn ones(shape: impl Into<Shape>) -> Self {
        Self::full(shape, num::One::one())
    }
}

impl<T> From<T> for Tensor<T>
    where
        T: num::Num + Copy,
{
    fn from(scalar: T) -> Self {
        Tensor::new([], vec![scalar])
    }
}