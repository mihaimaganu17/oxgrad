use std::{
    ops::{Add, Sub, Mul},
    fmt::{self, Debug},
    sync::atomic::{AtomicUsize, Ordering},
    collections::VecDeque,
};

pub trait Tanh {
    fn tanh(self) -> Self;
}

pub trait Pow {
    fn powi(self, pow: i32) -> Self;
}

pub trait Zero {
    fn zero() -> Self;
}

pub trait One {
    fn one() -> Self;
}

impl Tanh for f64 {
    fn tanh(self) -> Self {
        ((self * 2.0).exp() - 1.0) / ((self * 2.0).exp() + 1.0)
    }
}

impl Pow for f64 {
    fn powi(self, pow: i32) -> Self {
        self.powi(pow)
    }
}

impl Zero for f64 {
    fn zero() -> Self { 0.0 }
}

impl One for f64 {
    fn one() -> Self { 1.0 }
}

pub static UNIQUE_ID: AtomicUsize = AtomicUsize::new(0);

#[derive(PartialEq, Clone)]
pub struct Value<T: Debug + Clone + Zero + One> {
    pub data: T,
    pub prev: Vec<Value<T>>,
    pub op: Option<Op>,
    pub id: usize,
    pub label: String,
    // Value of the derivative of the output with respect to this current value
    pub grad: T,
    // Function which applies chain rule, given the current gradient to underlying `children` nodes
    pub _backward: Option<fn(&mut Vec<Value<T>>, &T)>,
}

impl<T> Value<T>
where T:
    fmt::Debug + Clone + Add + Mul + Zero + One,
{
    pub fn new(data: T, label: &str) -> Self {
        Self::new_with_fields(data, vec![], None, None, label)
    }

    // Can be replaced by a builder
    pub fn new_with_fields(
        data: T,
        children: Vec<Self>,
        op: Option<Op>,
        _backward: Option<fn(&mut Vec<Value<T>>, &T)>,
        label: &str,
    ) -> Self {
        Self {
            data,
            prev: children,
            op,
            id: UNIQUE_ID.fetch_add(1, Ordering::Relaxed),
            label: label.to_string(),
            grad: T::zero(),
            _backward,
        }
    }

    pub fn op_id(&self) -> usize {
        if let Some(op) = &self.op {
            usize::from(op)
        } else {
            0
        }
    }

    pub fn backward(&self) {
        fn topo_sort<T>(root: &Value<T>) -> Vec<&Value<T>>
        where T: std::fmt::Debug + std::ops::Add + std::ops::Mul + Clone + Zero + One,
        {
            let mut topo = vec![];
            let mut to_visit = VecDeque::from([root]);
            while let Some(node) = to_visit.pop_front() {
                for idx in 0..node.prev.len() {
                    to_visit.push_back(&node.prev[idx]);
                }
                topo.push(node);
            }
            topo
        }
        let _topo = topo_sort(self);
        println!("{:#?}", _topo);
    }
}

impl<T> Add for Value<T>
where T:
    fmt::Debug + Clone + Add<Output=T> + Mul<Output=T> + Pow + Tanh + Zero + One,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let data = self.data.clone() + rhs.data.clone();
        let children = vec![self, rhs];

        Self::new_with_fields(data, children, Some(Op::Add), Some(add_backward), "")
    }
}

impl<T> Sub for Value<T>
where T:
    fmt::Debug + Clone + Add + Sub<Output=T> + Mul + Pow + Tanh + Zero + One,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let data = self.data.clone() - rhs.data.clone();
        let children = vec![self, rhs];

        Self::new_with_fields(data, children, Some(Op::Sub), None, "")
    }
}

fn add_backward<T>(children: &mut Vec<Value<T>>, local_grad: &T)
where T:
    fmt::Debug + Clone + Add + Mul<Output=T> + Tanh + Pow + Zero + One,
{
    assert!(children.len() == 2);
    // This is a plus / add operation, so every child node basically propagate the result
    // as is.
    for v in children.iter_mut() {
        v.grad = T::one() * local_grad.clone();
    }
}

fn mul_backward<T>(children: &mut Vec<Value<T>>, local_grad: &T)
where T:
    fmt::Debug + Clone + Add + Mul<Output=T> + Tanh + Pow + Zero + One,
{
    // This should contain only 2 elements.
    assert!(children.len() == 2);
    children[0].grad = children[1].data.clone() * local_grad.clone();
    children[1].grad = children[0].data.clone() * local_grad.clone();
}

fn tanh_backward<T>(children: &mut Vec<Value<T>>, local_grad: &T)
where T:
    fmt::Debug + Clone + Add + Sub<Output=T> + Mul<Output=T> + Tanh + Pow + Zero + One,
{
    // This should contain only 2 elements.
    assert!(children.len() == 1);
    children[0].grad = (T::one() - children[0].data.clone().powi(2)) * local_grad.clone();
}

impl<T> Mul for Value<T>
where T:
    fmt::Debug + Clone + Add + Mul<Output=T> + Tanh + Pow + Zero + One,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let data = self.data.clone() * rhs.data.clone();
        let children = vec![self, rhs];

        Self::new_with_fields(data, children, Some(Op::Mul), Some(mul_backward), "")
    }
}

impl<T> fmt::Debug for Value<T>
where T:
    fmt::Debug + Clone + Zero + One,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "Value(data={:?})", self.data)
    }
}

impl<T> Tanh for Value<T>
where T:
    fmt::Debug + Clone + Add + Sub<Output=T> + Mul<Output=T> + Tanh + Pow + Zero + One,
{
    fn tanh(self) -> Self {
        let data = self.data.clone().tanh();
        let children = vec![self];

        Self::new_with_fields(data, children, Some(Op::Tanh), Some(tanh_backward), "")
    }
}

/// Operation performed
#[derive(PartialEq, Clone)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Tanh,
}

impl From<&Op> for usize {
    fn from(op: &Op) -> usize {
        match op {
            Op::Add => 1,
            Op::Sub => 2,
            Op::Mul => 3,
            Op::Tanh=> 4,
        }
    }
}

impl Debug for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            Op::Add => write!(f, "+"),
            Op::Sub => write!(f, "-"),
            Op::Mul => write!(f, "*"),
            Op::Tanh => write!(f, "tanh"),
        }
    }
}

impl ToString for Op {
    fn to_string(&self) -> String {
        format!("{self:?}")
    }
}
