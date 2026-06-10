//! Elementwise graph → CUDA C++ source.
//!
//! One thread per output element, flat indexing: the whole graph collapses into
//! one `__global__` kernel and each live node becomes a single register
//! statement (loads read `in[idx]`, the store writes `out[idx]`). The backend
//! emits f32 math only — the Python original's policy — enforced here up front
//! with a clear error rather than emitting code that fails in `nvcc`.

use std::collections::{HashMap, HashSet};

use crate::ir::{Dtype, Graph, Node, NodeId, Op};
use crate::printer::canonical;

/// Error from code generation (unsupported dtype, multiple outputs, ...).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodegenError(pub String);

impl std::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for CodegenError {}

/// tanh constant, sqrt(2/pi). All math is f32, so the literal carries an `f`.
const GELU_C: &str = "0.7978845608028654f";

/// FNV-1a (64-bit), first 8 hex chars. The Python original uses a SHA256 prefix;
/// this port stays dependency-free, so it uses FNV — same role: a stable hash of
/// the canonical signature, so the cache key and the symbol name always agree.
fn hash8(s: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{:08x}", (h & 0xffff_ffff) as u32)
}

/// `kernel_<hash>`, where `<hash>` derives from the canonical signature. The
/// symbol is unmangled via `extern "C"` so a driver can look it up by name.
pub fn kernel_name(g: &Graph) -> String {
    format!("kernel_{}", hash8(&canonical(g)))
}

/// SSA name `%0` → C identifier `v0` (SSA names are unique, so cvars are too).
fn cvar(n: &Node) -> String {
    format!("v{}", n.name.trim_start_matches('%'))
}

/// Register-level RHS for one node, referencing its args' C variables.
fn expr(g: &Graph, n: &Node) -> Result<String, CodegenError> {
    let a: Vec<String> = n.args.iter().map(|id| cvar(g.node(*id))).collect();
    let s = match n.op {
        Op::Add => format!("{} + {}", a[0], a[1]),
        Op::Sub => format!("{} - {}", a[0], a[1]),
        Op::Mul => format!("{} * {}", a[0], a[1]),
        Op::Div => format!("{} / {}", a[0], a[1]),
        Op::Neg => format!("-{}", a[0]),
        Op::Exp => format!("expf({})", a[0]),
        Op::Log => format!("logf({})", a[0]),
        Op::Sqrt => format!("sqrtf({})", a[0]),
        Op::Relu => format!("fmaxf({}, 0.0f)", a[0]),
        Op::Silu => format!("{0} / (1.0f + expf(-{0}))", a[0]),
        Op::Gelu => {
            let x = &a[0];
            let inner = format!("{GELU_C} * ({x} + 0.044715f * {x} * {x} * {x})");
            format!("0.5f * {x} * (1.0f + tanhf({inner}))")
        }
        Op::Load | Op::Store => {
            return Err(CodegenError(format!(
                "no register expression for op {} at {}",
                n.op.name(),
                n.name
            )));
        }
    };
    Ok(s)
}

/// Identifiers the kernel body/signature already uses, plus C++ keywords. An
/// input param named like any of these would clash with the output pointer, a
/// local, a cvar, or the language itself.
const CPP_KEYWORDS: &[&str] = &[
    "alignas",
    "alignof",
    "and",
    "asm",
    "auto",
    "bool",
    "break",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "constexpr",
    "continue",
    "decltype",
    "default",
    "delete",
    "do",
    "double",
    "else",
    "enum",
    "explicit",
    "extern",
    "false",
    "float",
    "for",
    "friend",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "namespace",
    "new",
    "not",
    "operator",
    "or",
    "private",
    "protected",
    "public",
    "register",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "template",
    "this",
    "throw",
    "true",
    "try",
    "typedef",
    "typename",
    "union",
    "unsigned",
    "using",
    "virtual",
    "void",
    "volatile",
    "while",
    "xor",
];

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Matches a generated cvar name (`v` followed by digits).
fn is_cvar(s: &str) -> bool {
    s.len() > 1 && s.starts_with('v') && s[1..].bytes().all(|b| b.is_ascii_digit())
}

fn reserved(s: &str) -> bool {
    matches!(s, "out" | "idx" | "N") || CPP_KEYWORDS.contains(&s)
}

/// A safe, unique C identifier for an input pointer. Uses the source arg name
/// when it cannot collide with the output, a local, a cvar, or a keyword;
/// otherwise falls back to `in<idx>`.
fn param_name(n: &Node, idx: usize, taken: &HashSet<String>) -> String {
    let label = &n.label;
    if is_ident(label) && !reserved(label) && !is_cvar(label) && !taken.contains(label) {
        return label.clone();
    }
    let mut name = format!("in{idx}");
    while taken.contains(&name) {
        name.push('_');
    }
    name
}

fn input_names(g: &Graph) -> HashMap<NodeId, String> {
    let mut taken: HashSet<String> = HashSet::new();
    let mut names: HashMap<NodeId, String> = HashMap::new();
    for (i, &id) in g.inputs.iter().enumerate() {
        let nm = param_name(g.node(id), i, &taken);
        taken.insert(nm.clone());
        names.insert(id, nm);
    }
    names
}

fn header(name: &str, params: &[String]) -> String {
    let sig = params.join(",\n    ");
    format!("extern \"C\" __global__ void {name}(\n    {sig}\n) {{\n")
}

/// Lower an elementwise graph to a single CUDA C++ `__global__` kernel.
pub fn emit_cuda(g: &Graph) -> Result<String, CodegenError> {
    if g.outputs.len() != 1 {
        return Err(CodegenError(format!(
            "codegen supports one output, got {}",
            g.outputs.len()
        )));
    }
    // Single-dtype policy: f32 math only. Reject f16/bf16/i32 once, here, with a
    // clear message rather than emitting code that fails in nvcc or mixes types.
    for n in &g.nodes {
        if n.dtype != Dtype::F32 {
            return Err(CodegenError(format!(
                "codegen supports f32 only; got {} at {} ({})",
                n.dtype.tag(),
                n.name,
                n.op.name()
            )));
        }
    }

    let names = input_names(g);
    let name = kernel_name(g);
    let store = g.node(g.outputs[0]);

    let mut params: Vec<String> = g
        .inputs
        .iter()
        .map(|id| {
            format!(
                "const {}* __restrict__ {}",
                g.node(*id).dtype.ctype(),
                names[id]
            )
        })
        .collect();
    params.push(format!("{}* __restrict__ out", store.dtype.ctype()));
    params.push("int N".to_string());

    // Only live nodes are read: a dead input (kept for a stable ABI) is a
    // parameter but must never be dereferenced, or it reads out of bounds.
    let live = g.live_nodes();
    let mut body = vec![
        "    int idx = blockIdx.x * blockDim.x + threadIdx.x;".to_string(),
        "    if (idx >= N) return;".to_string(),
    ];
    for (id, n) in g.nodes.iter().enumerate() {
        if !live[id] || n.op == Op::Store {
            continue;
        }
        if n.op == Op::Load {
            body.push(format!(
                "    {} {} = {}[idx];",
                n.dtype.ctype(),
                cvar(n),
                names[&id]
            ));
        } else {
            body.push(format!(
                "    {} {} = {};",
                n.dtype.ctype(),
                cvar(n),
                expr(g, n)?
            ));
        }
    }
    body.push(format!("    out[idx] = {};", cvar(g.node(store.args[0]))));

    Ok(header(&name, &params) + &body.join("\n") + "\n}\n")
}
