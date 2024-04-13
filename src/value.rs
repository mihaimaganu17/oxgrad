use std::{
    ops::{Add, Mul},
    fmt::{self, Debug},
    sync::atomic::{AtomicUsize, Ordering},
};

pub static UNIQUE_ID: AtomicUsize = AtomicUsize::new(0);

#[derive(PartialEq)]
pub struct Value<T: Debug> {
    pub data: T,
    pub prev: Vec<Value<T>>,
    pub op: Option<Op>,
    pub id: usize,
    pub label: String,
}

impl<T> Value<T>
where T:
    fmt::Debug + Add + Mul,
{
    pub fn new(data: T, label: &str) -> Self {
        Self::new_with_fields(data, vec![], None, label)
    }

    // Can be replaced by a builder
    pub fn new_with_fields(data: T, children: Vec<Self>, op: Option<Op>, label: &str) -> Self {
        Self {
            data,
            prev: children,
            op,
            id: UNIQUE_ID.fetch_add(1, Ordering::Relaxed),
            label: label.to_string(),
        }
    }

    pub fn op_id(&self) -> usize {
        if let Some(op) = &self.op {
            usize::from(op)
        } else {
            0
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

        Self::new_with_fields(data, children, Some(Op::Add), "")
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

        Self::new_with_fields(data, children, Some(Op::Mul), "")
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
#[derive(PartialEq)]
pub enum Op {
    Add,
    Mul,
}

impl From<&Op> for usize {
    fn from(op: &Op) -> usize {
        match op {
            Op::Add => 1,
            Op::Mul => 2,
        }
    }
}

impl Debug for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            Op::Add => write!(f, "+"),
            Op::Mul => write!(f, "*"),
        }
    }
}

impl ToString for Op {
    fn to_string(&self) -> String {
        format!("{self:?}")
    }
}
