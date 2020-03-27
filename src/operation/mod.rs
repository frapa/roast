use std::collections::HashMap;

type Id = u128;

impl CalcGrad {
    fn grad(context: Context) ->  {

    }
}

pub struct Operation {
    calc_grad: Box<dyn CalcGrad>,

}

pub struct Context {
    operations: HashMap<Id, >,
}