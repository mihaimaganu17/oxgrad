mod value;

pub use value::{Value, Op, Tanh, Zero, One};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let a = Value::new(10.0, "a");
        let b = Value::new(-3.0, "b");
        let c = a*b;
        c.backward();
    }
}
