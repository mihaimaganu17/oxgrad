mod value;

pub use value::{Value, Op};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let a = Value::new(10.0);
        let b = Value::new(-3.0);
        println!("{:?}", a*b);
    }
}
