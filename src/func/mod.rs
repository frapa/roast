use super::*;

use std::ops::*;
use std::collections::HashMap;

use num;

fn reduce<T>(iterator: impl Iterator<Item=T>, func: impl Fn(T, T) -> T) -> T
    where
        T: Default + Copy,
{
    let mut acc : T = T::default();
    iterator.map(|val| acc = func(acc, val));
    acc
}

pub trait ElemwiseFunction<T>
    where
        T: num::Num + Copy + 'static,
{
    fn apply(&self, input_elems: &[&T]) -> T;
    fn grad(&self, inputs: &[&Tensor<T>]) -> HashMap<u128, Box<CalcGrad>>;
}

pub fn elemwise<T>(inputs: &[&Tensor<T>], func: impl ElemwiseFunction<T>) -> Tensor<T>
where
    T: num::Num + Copy + 'static,
{
    let result_shape = inputs[0].shape.clone();
    let mut result_data = Vec::with_capacity(result_shape.size);

    for i in 0..result_shape.size {
        let input_elements = inputs
            .iter()
            .map(|tensor| tensor.data.get(i).unwrap())
            .collect::<Vec<&T>>()
        ;
        result_data.push(func.apply(input_elements.as_ref()));
    }

    let result = Tensor::new(result_shape, result_data);

    track_tensors(inputs, &result, func);

    result
}

fn track_tensors<T>(inputs: &[&Tensor<T>], output: &Tensor<T>, func: impl ElemwiseFunction<T>)
where
    T: num::Num + Copy + 'static,
{
    let must_track_gradient = reduce(
        inputs.iter().map(|tensor| tensor.track_grad),
        |acc, val| acc | val
    );
    if must_track_gradient {
        let context = context();

        for input in inputs {
            context.register_tensor(input);
        }
        context.register_tensor(output);

        let operation = Operation::new(inputs, output, func.grad(inputs));
        context.register_operation(operation);
    }
}

pub struct AddFunction ();
impl<T> ElemwiseFunction<T> for AddFunction
    where
        T: num::Num + Copy + 'static,
{
    fn apply(&self, input_elems: &[&T]) -> T {
        let a = input_elems[0];
        let b = input_elems[1];
        *a + *b
    }

    fn grad(&self, inputs: &[&Tensor<T>]) -> HashMap<u128, Box<CalcGrad>>
    {
        let a = inputs[0];
        let b = inputs[1];
        grad_map! {
            a.id => |_| Box::new(Tensor::<T>::ones([])),
            b.id => |_| Box::new(Tensor::<T>::ones([])),
        }
    }
}

pub struct SubFunction ();
impl<T> ElemwiseFunction<T> for SubFunction
    where
        T: num::Num + Copy + 'static,
{
    fn apply(&self, input_elems: &[&T]) -> T {
        let a = input_elems[0];
        let b = input_elems[1];
        *a - *b
    }

    fn grad(&self, inputs: &[&Tensor<T>]) -> HashMap<u128, Box<CalcGrad>> {
        let a = inputs[0];
        let b = inputs[1];
        grad_map! {
            a.id => |_| Box::new(Tensor::<T>::ones([])),
            b.id => |_| Box::new(Tensor::<T>::ones([])),
        }
    }
}

pub struct MulFunction ();
impl<T> ElemwiseFunction<T> for MulFunction
    where
        T: num::Num + Copy + 'static,
{
    fn apply(&self, input_elems: &[&T]) -> T {
        let a = input_elems[0];
        let b = input_elems[1];
        *a * *b
    }

    fn grad(&self, inputs: &[&Tensor<T>]) -> HashMap<u128, Box<CalcGrad>>
    {
        let a = inputs[0].clone();
        let b = inputs[1].clone();
        let a_id = a.id;
        let b_id = b.id;
        grad_map! {
            a_id => move |_| Box::new(b.clone()),
            b_id => move |_| Box::new(a.clone()),
        }
    }
}

pub struct DivFunction ();
impl<T> ElemwiseFunction<T> for DivFunction
    where
        T: num::Num + Copy + 'static,
{
    fn apply(&self, input_elems: &[&T]) -> T {
        let a = input_elems[0];
        let b = input_elems[1];
        *a / *b
    }

    fn grad(&self, inputs: &[&Tensor<T>]) -> HashMap<u128, Box<CalcGrad>>
    {
        let a = inputs[0].clone();
        let b = inputs[1].clone();
        let a_id = a.id;
        let b_id = b.id;
        grad_map! {
            a_id => move |_| Box::new(b.clone()),
            b_id => move |_| Box::new(a.clone()),
        }
    }
}

pub struct RemFunction ();
impl<T> ElemwiseFunction<T> for RemFunction
    where
        T: num::Num + Copy + 'static,
{
    fn apply(&self, input_elems: &[&T]) -> T {
        let a = input_elems[0];
        let b = input_elems[1];
        *a % *b
    }

    fn grad(&self, inputs: &[&Tensor<T>]) -> HashMap<u128, Box<CalcGrad>>
    {
        let a = inputs[0].clone();
        let b = inputs[1].clone();
        let a_id = a.id;
        let b_id = b.id;
        grad_map! {
            a_id => move |_| Box::new(b.clone()),
            b_id => move |_| Box::new(a.clone()),
        }
    }
}

pub struct NegFunction ();
impl<T> ElemwiseFunction<T> for NegFunction
    where
        T: num::Num + Copy + Neg<Output=T> + 'static,
{
    fn apply(&self, input_elems: &[&T]) -> T {
        let a = input_elems[0];
        -*a
    }

    fn grad(&self, inputs: &[&Tensor<T>]) -> HashMap<u128, Box<CalcGrad>> {
        let a = inputs[0].clone();
        let b = inputs[1].clone();
        let a_id = a.id;
        let b_id = b.id;
        grad_map! {
            a_id => move |_| Box::new(b.clone()),
            b_id => move |_| Box::new(a.clone()),
        }
    }
}