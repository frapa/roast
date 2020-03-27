use super::*;

use std::time::Instant;

fn tensor_shape() -> Shape {
    Shape::new(&[3, 2])
}

fn tensor_full(v: i32) -> Tensor<i32> {
    Tensor::<i32>::full(tensor_shape(), v)
}

#[test]
fn test_elemwise2() {
    let sum = elemwise2(tensor_full(1), tensor_full(1), |a, b| a + b);
    assert_eq!(sum, tensor_full(2));
}

#[test]
fn test_add_scalar() {
    assert_eq!(2 + tensor_full(1), tensor_full(3));
    assert_eq!(tensor_full(1) + 2, tensor_full(3));
}

#[test]
fn test_add() {
    assert_eq!(tensor_full(1) + tensor_full(1), tensor_full(2));
}

#[test]
fn test_sub() {
    assert_eq!(tensor_full(1) - tensor_full(1), tensor_full(0));
}

#[test]
fn test_mul() {
    assert_eq!(tensor_full(2) * tensor_full(3), tensor_full(6));
}

#[test]
fn test_div() {
    assert_eq!(tensor_full(6) / tensor_full(2), tensor_full(3));
}

#[test]
#[should_panic]
fn test_div_zero() {
    assert_eq!(tensor_full(6) / tensor_full(0), tensor_full(3));
}

#[test]
fn test_rem() {
    assert_eq!(tensor_full(6) % tensor_full(4), tensor_full(2));
}

// #[test]
// fn test_neg() {
//     assert_eq!(-tensor_full(1), tensor_full(-1));
// }

#[test]
fn bench_add_i32() {
    let t1 = Tensor::<i32>::ones([1000, 1000]);
    let t2 = Tensor::<i32>::ones([1000, 1000]);

    let now = Instant::now();
    {
        t1 + t2;
    }
    let elapsed = now.elapsed();

    println!("{:?}", elapsed);
}
