use std::{
    ops::{Add, Mul},
    fmt::{self, Debug},
};

pub struct Value<T: Debug> {
    pub data: T,
    _prev: Vec<Value<T>>,
    _op: Op,
}

impl<T> Value<T>
where T:
    fmt::Debug + Add + Mul,
{
    pub fn new(data: T) -> Self {
        Self::new_with_fields(data, vec![], Op::None)
    }

    // Can be replaced by a builder
    pub fn new_with_fields(data: T, children: Vec<Self>, op: Op) -> Self {
        Self {
            data,
            _prev: children,
            _op: op,
        }
    }
}

impl<T> Add for Value<T>
where T:
    fmt::Debug + Add<Output=T> + Mul + Clone,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let data = self.data.clone() + rhs.data.clone();
        let children = vec![self, rhs];

        Self::new_with_fields(data, children, Op::Add)
    }
}

impl<T> Mul for Value<T>
where T:
    fmt::Debug + Mul<Output=T> + Add + Clone,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let data = self.data.clone() * rhs.data.clone();
        let children = vec![self, rhs];

        Self::new_with_fields(data, children, Op::Mul)
    }
}

impl<T> fmt::Debug for Value<T>
where T:
    fmt::Debug
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "Value(data={:?})", self.data)
    }
}

/// Operation performed
pub enum Op {
    Add,
    Mul,
    None,
}
