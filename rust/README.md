# tinymlc — Rust port (IR + elementwise CUDA codegen)

A small Rust port of the core of [tinymlc](../README.md): the compiler **IR** and
the **Stage-1 elementwise CUDA C++ code generator**. It builds an SSA graph of
pointwise ops and lowers it to a single fused `__global__` kernel.

This is deliberately the *small, safe* slice of the project — pure logic, no GPU:

- **In scope:** the IR (an arena/SSA DAG), the IR pretty-printer + canonical
  signature, and the elementwise backend (one thread per element, flat indexing,
  the whole graph fused into one kernel; f32 only; dead-input handling; safe
  kernel-parameter naming).
- **Out of scope** (lives only in the Python project): the tracing frontend,
  reductions, matmul, optimization passes, and the `nvcc` / CUDA driver-API
  runtime.

No dependencies (std only), no `unsafe`, no I/O — codegen is string building, so
the whole thing is unit-tested on any machine without a GPU.

## Design note

The Python IR uses dataclass nodes with object-identity equality. The idiomatic
Rust equivalent is an **arena**: `Graph` owns a `Vec<Node>` and nodes refer to
each other by index (`NodeId`). Args only point at earlier nodes, so the graph is
acyclic — no `Rc`, no `RefCell`. Ops and dtypes are enums, so codegen is an
exhaustive `match`. The kernel symbol is a hash of the shape-agnostic canonical
signature (FNV here, vs SHA-256 in Python — same role, zero dependencies).

## Usage

```rust
use tinymlc::{Dtype, GraphBuilder, emit_cuda, print_graph};

let mut b = GraphBuilder::new();
let x = b.input("x", &[4096, 4096], Dtype::F32);
let y = b.input("y", &[4096, 4096], Dtype::F32);
let s = b.add(x, y);
let out = b.gelu(s); // gelu(x + y)
let g = b.build(out);

println!("{}", print_graph(&g));   // the IR
println!("{}", emit_cuda(&g).unwrap()); // the CUDA C++ kernel
```

## Build & test

```bash
cd rust
cargo test     # unit + integration tests (no GPU needed)
cargo doc      # API docs, including a runnable doctest
```

## Example output

`gelu(x + y)` lowers to:

```cuda
extern "C" __global__ void kernel_xxxxxxxx(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v0 = x[idx];
    float v1 = y[idx];
    float v2 = v0 + v1;
    float v3 = 0.5f * v2 * (1.0f + tanhf(0.7978845608028654f * (v2 + 0.044715f * v2 * v2 * v2)));
    out[idx] = v3;
}
```

MIT licensed, same as the parent project.
