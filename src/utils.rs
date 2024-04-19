use crate::value::{Value, Zero, One};
pub fn trace<'a, T>(root: &'a Value<T>) -> (Vec<&'a Value<T>>, Vec<(&'a Value<T>, &'a Value<T>)>)
where T: std::fmt::Debug + std::cmp::PartialEq + Clone + Copy + Zero + One + Default,
{
    let mut nodes = vec![];
    let mut edges = vec![];

    build_trace(&mut nodes, &mut edges, root);

    (nodes, edges)
}

fn build_trace<'a, T>(
    nodes: &mut Vec<&'a Value<T>>,
    edges: &mut Vec<(&'a Value<T>, &'a Value<T>)>,
    value: &'a Value<T>,
)
where T: std::fmt::Debug + std::cmp::PartialEq + Clone + Copy + Zero + One + Default,
{
    if !nodes.contains(&value) {
        nodes.push(value);

        for child in value.prev.iter() {
            edges.push((&child, value));
            build_trace(nodes, edges, &child);
        }
    }
}
