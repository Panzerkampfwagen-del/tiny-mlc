//! A Rust port of a small, self-contained slice of [`tinymlc`] — the compiler IR
//! and the Stage-1 elementwise CUDA C++ code generator.
//!
//! The original (Python) project is a *Python DSL → custom IR → CUDA C++*
//! compiler. This port covers two stages of it:
//!
//! 1. the **IR** — an SSA DAG of pointwise ops, and
//! 2. the **elementwise backend** — lowering a graph to a single fused
//!    `__global__` kernel (one thread per output element).
//!
//! Out of scope on purpose (this is the small, safe slice): the tracing
//! frontend, reductions, matmul, optimization passes, and the `nvcc` / CUDA
//! driver-API runtime. There is no GPU dependency, no `unsafe`, and no I/O —
//! code generation is string building, so everything here is unit-testable on
//! any machine.
//!
//! ```
//! use tinymlc::{Dtype, GraphBuilder, emit_cuda};
//!
//! // fused_gelu(x, y) = gelu(x + y), one kernel, no intermediate buffer.
//! let mut b = GraphBuilder::new();
//! let x = b.input("x", &[4096, 4096], Dtype::F32);
//! let y = b.input("y", &[4096, 4096], Dtype::F32);
//! let s = b.add(x, y);
//! let out = b.gelu(s);
//! let graph = b.build(out);
//!
//! let cuda = emit_cuda(&graph).unwrap();
//! assert!(cuda.contains("__global__ void kernel_"));
//! assert!(cuda.contains("tanhf("));            // the gelu lowering
//! ```
//!
//! [`tinymlc`]: https://github.com/Panzerkampfwagen-del/tiny-mlc

pub mod cuda;
pub mod ir;
pub mod printer;

pub use cuda::{emit_cuda, kernel_name, CodegenError};
pub use ir::{numel, Dtype, Graph, GraphBuilder, Node, NodeId, Op};
pub use printer::{canonical, format_node, print_graph};
