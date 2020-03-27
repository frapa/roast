#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    pub shape: Box<[usize]>,
    pub size: usize,
}

impl Shape {
    pub fn new(shape: &[usize]) -> Self {
        let mut size = 1;
        for dim in shape.iter() {
            size *= *dim;
        }

        Self {
            shape: shape.to_vec().into_boxed_slice(),
            size
        }
    }
}

impl From<&Shape> for Shape {
    fn from(shape: &Shape) -> Self {
        (*shape).clone()
    }
}

impl<T> From<T> for Shape
    where
        T: AsRef<[usize]>
{
    fn from(shape: T) -> Self {
        Self::new(shape.as_ref())
    }
}