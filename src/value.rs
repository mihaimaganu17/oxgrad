use std::{
    ops::{Add, Sub, Mul},
    fmt::{self, Debug},
    sync::atomic::{AtomicUsize, Ordering},
    collections::VecDeque,
    cell::Cell,
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
pub struct Value<T: Debug + Clone + Copy + Zero + One + Default> {
    pub data: T,
    pub prev: Vec<Value<T>>,
    pub op: Option<Op>,
    pub id: usize,
    pub label: String,
    // Value of the derivative of the output with respect to this current value
    // Idea: Make grad a separate type which get value which is created by given `Value` as an
    // input
    // Alternatively this might even be a OnceCell because once we set the gradient, there is no
    // reason for it to change, unless we change the underlying `backward` function, at which
    // point there should be another `Value object` defined
    pub grad: Cell<T>,
    // Function which applies chain rule, given the current gradient to underlying `children` nodes
    _backward: Option<fn(&[Value<T>], T)>,
}

impl<T> Value<T>
where T:
    fmt::Debug + Clone + Copy + Add + Mul + Zero + One + Default,
{
    pub fn new(data: T, label: &str) -> Self {
        Self::new_with_fields(data, vec![], None, None, label)
    }

    // Can be replaced by a builder
    pub fn new_with_fields(
        data: T,
        children: Vec<Self>,
        op: Option<Op>,
        _backward: Option<fn(&[Value<T>], T)>,
        label: &str,
    ) -> Self {
        Self {
            data,
            prev: children,
            op,
            id: UNIQUE_ID.fetch_add(1, Ordering::Relaxed),
            label: label.to_string(),
            grad: Cell::new(T::zero()),
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
        where T: std::fmt::Debug + std::ops::Add + std::ops::Mul + Clone + Copy + Zero + One
            + Default,
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
        self.grad.set(T::one());
        let topo = topo_sort(self);

        for node in topo {
            if let Some(backward_fn) = node._backward {
                backward_fn(node.prev.as_slice(), node.grad.clone().into_inner());
            }
        }
    }
}

impl<T> Add for Value<T>
where T:
    fmt::Debug + Clone + Copy + Add<Output=T> + Mul<Output=T> + Pow + Tanh + Zero + One
    + Default,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let data = self.data.clone() + rhs.data.clone();
        let children = vec![self, rhs];

        Self::new_with_fields(data, children, Some(Op::Add), Some(add_backward), "")
    }
}

fn add_backward<T>(children: &[Value<T>], local_grad: T)
where T:
    fmt::Debug + Clone + Copy + Add<Output=T> + Mul<Output=T> + Tanh + Pow + Zero + One + Default,
{
    assert!(children.len() == 2);
    // This is a plus / add operation, so every child node basically propagate the result
    // as is.
    for v in children.iter() {
        let new_grad = v.grad.take() + (T::one() * local_grad);
        v.grad.set(new_grad);
    }
}

fn mul_backward<T>(children: &[Value<T>], local_grad: T)
where T:
    fmt::Debug + Clone + Copy + Add<Output=T> + Mul<Output=T> + Tanh + Pow + Zero + One + Default,
{
    // This should contain only 2 elements.
    assert!(children.len() == 2);
    children[0].grad.set(children[0].grad.take() + children[1].data.clone() * local_grad.clone());
    children[1].grad.set(children[1].grad.take() + children[0].data.clone() * local_grad.clone());
}

fn tanh_backward<T>(children: &[Value<T>], local_grad: T)
where T:
    fmt::Debug + Clone + Copy + Add + Sub<Output=T> + Mul<Output=T> + Tanh + Pow + Zero + One + Default,
{
    // This should contain only 2 elements.
    assert!(children.len() == 1);
    children[0].grad.set((T::one() - children[0].data.clone().tanh().powi(2)) * local_grad.clone());
}

impl<T> Mul for Value<T>
where T:
    fmt::Debug + Clone + Copy + Add<Output=T> + Mul<Output=T> + Tanh + Pow + Zero + One + Default,
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
    fmt::Debug + Clone + Copy + Zero + One + Default,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "Value(data={:?})", self.data)
    }
}

impl<T> Tanh for Value<T>
where T:
    fmt::Debug + Clone + Copy + Add + Sub<Output=T> + Mul<Output=T> + Tanh + Pow + Zero + One
    + Default,
{
    fn tanh(self) -> Self {
        let data = self.data.clone().tanh();
        let children = vec![self];

        Self::new_with_fields(data, children, Some(Op::Tanh), Some(tanh_backward), "")
    }
}

/// Operation performed
#[derive(Clone, Copy, PartialEq)]
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
