//! End-to-end checks: build an IR graph, lower it, and assert the emitted CUDA.

use tinymlc::{canonical, emit_cuda, kernel_name, print_graph, Dtype, GraphBuilder};

#[test]
fn fused_gelu_add_emits_one_kernel() {
    // fused_gelu(x, y) = gelu(x + y)
    let mut b = GraphBuilder::new();
    let x = b.input("x", &[4096, 4096], Dtype::F32);
    let y = b.input("y", &[4096, 4096], Dtype::F32);
    let s = b.add(x, y);
    let out = b.gelu(s);
    let g = b.build(out);

    let cuda = emit_cuda(&g).unwrap();

    assert!(cuda.contains("extern \"C\" __global__ void kernel_"));
    assert!(cuda.contains("const float* __restrict__ x"));
    assert!(cuda.contains("const float* __restrict__ y"));
    assert!(cuda.contains("float* __restrict__ out"));
    assert!(cuda.contains("int N"));
    assert!(cuda.contains("int idx = blockIdx.x * blockDim.x + threadIdx.x;"));
    assert!(cuda.contains("if (idx >= N) return;"));
    assert!(cuda.contains("float v0 = x[idx];"));
    assert!(cuda.contains("float v1 = y[idx];"));
    assert!(cuda.contains("float v2 = v0 + v1;")); // the add
    assert!(cuda.contains("tanhf(")); // the gelu lowering
    assert!(cuda.contains("out[idx] = v3;")); // store reads the gelu result
}

#[test]
fn kernel_name_is_stable_and_shape_agnostic() {
    // Same op/dtype structure, different shapes → identical kernel symbol, so one
    // compiled kernel serves every shape.
    let mk = |shape: &[usize]| {
        let mut b = GraphBuilder::new();
        let x = b.input("x", shape, Dtype::F32);
        let r = b.relu(x);
        b.build(r)
    };
    let g1 = mk(&[16]);
    let g2 = mk(&[1024, 8]);
    assert_eq!(kernel_name(&g1), kernel_name(&g2));
    assert!(kernel_name(&g1).starts_with("kernel_"));
    // A different op structure must produce a different symbol.
    let mut b = GraphBuilder::new();
    let x = b.input("x", &[16], Dtype::F32);
    let e = b.exp(x);
    let g3 = b.build(e);
    assert_ne!(kernel_name(&g1), kernel_name(&g3));
}

#[test]
fn rejects_non_f32() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", &[8], Dtype::F16);
    let r = b.relu(x);
    let g = b.build(r);
    let err = emit_cuda(&g).unwrap_err();
    assert!(err.0.contains("f32 only"), "got: {err}");
}

#[test]
fn reserved_param_name_is_renamed() {
    // An input literally named "out" must not collide with the output pointer.
    let mut b = GraphBuilder::new();
    let v = b.input("out", &[8], Dtype::F32);
    let r = b.relu(v);
    let g = b.build(r);
    let cuda = emit_cuda(&g).unwrap();
    assert!(cuda.contains("const float* __restrict__ in0")); // renamed input
    assert!(cuda.contains("float* __restrict__ out")); // the real output pointer
    assert!(cuda.contains("float v0 = in0[idx];"));
}

#[test]
fn dead_input_is_in_abi_but_not_read() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", &[8], Dtype::F32);
    let _dead = b.input("dead", &[8], Dtype::F32); // never used
    let r = b.relu(x);
    let g = b.build(r);
    let cuda = emit_cuda(&g).unwrap();
    assert!(cuda.contains("const float* __restrict__ dead")); // still a parameter
    assert!(!cuda.contains("dead[idx]")); // but never dereferenced
}

#[test]
fn printer_matches_documented_ir_syntax() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", &[4], Dtype::F32);
    let y = b.input("y", &[4], Dtype::F32);
    let s = b.add(x, y);
    let g = b.build(s);
    let txt = print_graph(&g);
    assert!(txt.contains("%0 = load f32[4]   # x"));
    assert!(txt.contains("%1 = load f32[4]   # y"));
    assert!(txt.contains("%2 = add(%0, %1) f32[4]"));
    assert!(txt.contains("%3 = store(%2)"));

    // canonical signature drops concrete shapes
    assert_eq!(
        canonical(&g),
        "%0=load():f32;%1=load():f32;%2=add(%0,%1):f32;%3=store(%2):f32"
    );
}
