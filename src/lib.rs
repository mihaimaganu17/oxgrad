mod value;

pub use value::{Value, Op, Tanh};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let a = Value::new(10.0, "a");
        let b = Value::new(-3.0, "b");
        println!("{:?}", a*b.clone());
    }
}
