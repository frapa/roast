use super::*;

use std::rc::Rc;

use num;
use uuid;


#[derive(Debug, Clone)]
pub struct Tensor<T>
where
    T: num::Num + Copy + 'static,
{
    pub id: u128,
    pub data: Rc<[T]>,
    pub shape: Shape,
    pub track_grad: bool,
}

impl<T> Tensor<T>
where
    T: num::Num + Copy + 'static,
{
    pub fn new(shape: impl Into<Shape>, data: Vec<T>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().as_u128(),
            data: Rc::from(data),
            shape: shape.into(),
            track_grad: false,
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

    pub fn scalar(val: T) -> Self {
        Self::full([], val)
    }

    pub fn track_grad(&mut self) {
        self.track_grad = true;
    }
}

impl<T> From<T> for Tensor<T>
    where
        T: num::Num + Copy + 'static,
{
    fn from(scalar: T) -> Self {
        Tensor::new([], vec![scalar])
    }
}

impl<T> PartialEq for Tensor<T>
    where
        T: num::Num + Copy + 'static,
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.shape == other.shape
    }
}