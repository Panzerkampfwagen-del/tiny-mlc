//! IR → readable text, plus a stable shape-agnostic signature used as the kernel
//! symbol name / compile cache key.

use crate::ir::{Graph, NodeId, Op};

fn shape_str(g: &Graph, id: NodeId) -> String {
    let n = g.node(id);
    let dims: Vec<String> = n.shape.iter().map(|d| d.to_string()).collect();
    format!("{}[{}]", n.dtype.tag(), dims.join(","))
}

/// One line for a single node, matching the documented IR syntax.
pub fn format_node(g: &Graph, id: NodeId) -> String {
    let n = g.node(id);
    match n.op {
        Op::Load => {
            let line = format!("{} = load {}", n.name, shape_str(g, id));
            if n.label.is_empty() {
                line
            } else {
                format!("{line}   # {}", n.label)
            }
        }
        Op::Store => {
            let srcs: Vec<&str> = n.args.iter().map(|a| g.node(*a).name.as_str()).collect();
            format!("{} = store({})", n.name, srcs.join(", "))
        }
        _ => {
            let args: Vec<&str> = n.args.iter().map(|a| g.node(*a).name.as_str()).collect();
            format!(
                "{} = {}({}) {}",
                n.name,
                n.op.name(),
                args.join(", "),
                shape_str(g, id)
            )
        }
    }
}

/// The whole graph, one node per line.
pub fn print_graph(g: &Graph) -> String {
    (0..g.nodes.len())
        .map(|id| format_node(g, id))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Stable signature: op structure + dtypes, with concrete shapes omitted on
/// purpose. A Stage-1 elementwise kernel is flat over N, so one kernel serves
/// every shape with the same op/dtype structure — this signature is both the
/// compile-cache key and (hashed) the kernel symbol name, so they always agree.
pub fn canonical(g: &Graph) -> String {
    g.nodes
        .iter()
        .map(|n| {
            let args: Vec<&str> = n.args.iter().map(|a| g.nodes[*a].name.as_str()).collect();
            format!(
                "{}={}({}):{}",
                n.name,
                n.op.name(),
                args.join(","),
                n.dtype.tag()
            )
        })
        .collect::<Vec<_>>()
        .join(";")
}
