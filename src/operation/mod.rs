use super::*;

use std::collections::HashMap;

use uuid;

pub trait GetTensor {
    fn get_tensor<T>(tensor: Tensor<T>) -> Tensor<T>
        where
            T: num::Num + Copy + 'static,
            Self: Sized,
    {
        tensor
    }
}

impl<T> GetTensor for Tensor<T>
    where
        T: num::Num + Copy + 'static,
        Self: Sized
{}

pub type CalcGrad = dyn Fn(Context) -> Box<dyn GetTensor>;

pub struct Operation {
    id: u128,
    inputs: Vec<u128>,
    output: u128,
    calc_grad: HashMap<u128, Box<CalcGrad>>,
}

#[macro_export]
macro_rules! grad_map {
    ($($tensor_id:expr => $lambda:expr),* $(,)?) => {{
        use std::collections::HashMap;

        let mut map = HashMap::<u128, Box<CalcGrad>>::new();

        $(
            map.insert($tensor_id, Box::new($lambda));
        )*

        map
    }}
}

impl Operation {
    pub fn new<T>(
        inputs: &[&Tensor<T>],
        output: &Tensor<T>,
        calc_grad: HashMap<u128, Box<CalcGrad>>
    ) -> Operation
        where
            T: num::Num + Copy + 'static,
    {
        Operation {
            id: uuid::Uuid::new_v4().as_u128(),
            inputs: inputs.iter().map(|tensor| tensor.id).collect::<Vec<u128>>(),
            output: output.id,
            calc_grad,
        }
    }
}

pub struct Context {
    tensors: HashMap<u128, Box<dyn GetTensor>>,
    operations: HashMap<u128, Operation>,
    operations_by_input: HashMap<u128, u128>,
    operations_by_output: HashMap<u128, u128>,
}


impl Context {
    pub fn new() -> Context {
        Context {
            tensors: HashMap::new(),
            operations: HashMap::new(),
            operations_by_input: HashMap::new(),
            operations_by_output: HashMap::new(),
        }
    }

    pub fn register_tensor<T>(&mut self, tensor: &Tensor<T>)
        where
            T: num::Num + Copy + 'static,
    {
        self.tensors.insert(tensor.id, Box::new(tensor.clone()));
    }

    pub fn register_operation(&mut self, operation: Operation) {
        for input in operation.inputs.iter() {
            self.operations_by_input.insert(*input, operation.id);
        }

        self.operations_by_output.insert(operation.output, operation.id);

        self.operations.insert(operation.id, operation);
    }
}

static mut CONTEXT: Option<Context> = None;
static CONTEXT_GUARD: std::sync::Once = std::sync::Once::new();

pub fn context() -> &'static mut Context {
    unsafe {
        CONTEXT_GUARD.call_once(|| {
            CONTEXT = Some(Context::new());
        });
        CONTEXT.as_mut().unwrap()
    }
}