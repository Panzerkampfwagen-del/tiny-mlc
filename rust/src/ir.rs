//! Core IR: an arena of SSA nodes forming a DAG.
//!
//! The Python original uses dataclass nodes with object-identity equality. The
//! idiomatic, allocation-safe Rust equivalent is an arena: a [`Graph`] owns a
//! `Vec<Node>` and everything refers to a node by its index ([`NodeId`]). Args
//! only ever point at *earlier* nodes, so the graph is acyclic — no `Rc`, no
//! `RefCell`, no `unsafe`.

/// Element type. The codegen backend emits f32 math only; the others exist so
/// the IR can represent them (and be rejected with a clear error at codegen).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dtype {
    F32,
    F16,
    Bf16,
    I32,
}

impl Dtype {
    /// CUDA C scalar type.
    pub fn ctype(self) -> &'static str {
        match self {
            Dtype::F32 => "float",
            Dtype::F16 => "__half",
            Dtype::Bf16 => "__nv_bfloat16",
            Dtype::I32 => "int",
        }
    }

    /// Size in bytes.
    pub fn size(self) -> usize {
        match self {
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F16 | Dtype::Bf16 => 2,
        }
    }

    /// Short tag used by the printer and the canonical signature.
    pub fn tag(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::F16 => "f16",
            Dtype::Bf16 => "bf16",
            Dtype::I32 => "i32",
        }
    }
}

/// Operations in the elementwise subset. `Load`/`Store` are the kernel ABI; the
/// rest are pointwise and lower to one register statement each.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    Load,
    Store,
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Exp,
    Log,
    Sqrt,
    Relu,
    Silu,
    Gelu,
}

impl Op {
    /// The op's IR mnemonic (matches the Python printer/signature).
    pub fn name(self) -> &'static str {
        match self {
            Op::Load => "load",
            Op::Store => "store",
            Op::Add => "add",
            Op::Sub => "sub",
            Op::Mul => "mul",
            Op::Div => "div",
            Op::Neg => "neg",
            Op::Exp => "exp",
            Op::Log => "log",
            Op::Sqrt => "sqrt",
            Op::Relu => "relu",
            Op::Silu => "silu",
            Op::Gelu => "gelu",
        }
    }
}

/// Index of a node inside a [`Graph`]'s arena.
pub type NodeId = usize;

/// One SSA value.
#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    pub args: Vec<NodeId>,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    /// SSA name: `%0`, `%1`, ...
    pub name: String,
    /// Source name for loads; used by the printer and to name kernel params.
    pub label: String,
}

/// A traced kernel: a topologically ordered arena of nodes plus its input/output
/// roots.
#[derive(Clone, Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
}

impl Graph {
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id]
    }

    /// `live[id]` is true for every node reachable from the outputs by walking
    /// args — the set that actually contributes to a result. Codegen must not
    /// emit reads for dead inputs (kept only so the kernel ABI stays stable).
    pub fn live_nodes(&self) -> Vec<bool> {
        let mut live = vec![false; self.nodes.len()];
        let mut stack: Vec<NodeId> = self.outputs.clone();
        while let Some(id) = stack.pop() {
            if live[id] {
                continue;
            }
            live[id] = true;
            stack.extend(self.nodes[id].args.iter().copied());
        }
        live
    }
}

/// Number of elements in a shape.
pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Builds a [`Graph`] while assigning SSA names. A tiny stand-in for the Python
/// tracing frontend — enough to construct elementwise graphs in examples/tests.
#[derive(Default)]
pub struct GraphBuilder {
    nodes: Vec<Node>,
    inputs: Vec<NodeId>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        GraphBuilder::default()
    }

    fn push(
        &mut self,
        op: Op,
        args: Vec<NodeId>,
        shape: Vec<usize>,
        dtype: Dtype,
        label: &str,
    ) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(Node {
            op,
            args,
            shape,
            dtype,
            name: format!("%{id}"),
            label: label.to_string(),
        });
        id
    }

    /// A kernel input (load node). `label` is the source argument name.
    pub fn input(&mut self, label: &str, shape: &[usize], dtype: Dtype) -> NodeId {
        let id = self.push(Op::Load, vec![], shape.to_vec(), dtype, label);
        self.inputs.push(id);
        id
    }

    /// A pointwise op over `args`; shape and dtype are inherited from the first
    /// argument (elementwise ops are shape-preserving).
    pub fn elementwise(&mut self, op: Op, args: &[NodeId]) -> NodeId {
        let shape = self.nodes[args[0]].shape.clone();
        let dtype = self.nodes[args[0]].dtype;
        self.push(op, args.to_vec(), shape, dtype, "")
    }

    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.elementwise(Op::Add, &[a, b])
    }
    pub fn sub(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.elementwise(Op::Sub, &[a, b])
    }
    pub fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.elementwise(Op::Mul, &[a, b])
    }
    pub fn div(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.elementwise(Op::Div, &[a, b])
    }
    pub fn neg(&mut self, x: NodeId) -> NodeId {
        self.elementwise(Op::Neg, &[x])
    }
    pub fn exp(&mut self, x: NodeId) -> NodeId {
        self.elementwise(Op::Exp, &[x])
    }
    pub fn relu(&mut self, x: NodeId) -> NodeId {
        self.elementwise(Op::Relu, &[x])
    }
    pub fn silu(&mut self, x: NodeId) -> NodeId {
        self.elementwise(Op::Silu, &[x])
    }
    pub fn gelu(&mut self, x: NodeId) -> NodeId {
        self.elementwise(Op::Gelu, &[x])
    }

    /// Finish: mark `out` as the stored output and return the [`Graph`].
    pub fn build(mut self, out: NodeId) -> Graph {
        let dtype = self.nodes[out].dtype;
        let shape = self.nodes[out].shape.clone();
        let store = self.push(Op::Store, vec![out], shape, dtype, "");
        Graph {
            nodes: self.nodes,
            inputs: self.inputs,
            outputs: vec![store],
        }
    }
}
