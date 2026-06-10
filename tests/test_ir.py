"""IR + codegen unit tests: construction, printer, DCE, name safety,
reductions. No GPU."""

import numpy as np
import pytest

from tinymlc.codegen import emit_cuda, is_matmul, is_reduction
from tinymlc.frontend import exp, gelu, matmul, relu, sum, trace
from tinymlc.ir import Graph, MLCError, Node, canonical, print_graph
from tinymlc.passes import (
    dead_code_elimination,
    loop_tiling_pass,
    shared_memory_pass,
    type_propagation,
    verify_pass,
)
from tinymlc.pipeline import run_pipeline


def _gelu_graph():
    sh = (8, 8)
    x = Node("load", [], sh, "f32", "%0", "x")
    y = Node("load", [], sh, "f32", "%1", "y")
    a = Node("add", [x, y], sh, "f32", "%2")
    g = Node("gelu", [a], sh, "f32", "%3")
    s = Node("store", [g], sh, "f32", "%4")
    return Graph(inputs=[x, y], outputs=[s], nodes=[x, y, a, g, s])


def test_node_fields():
    n = Node("add", [], (4, 4), "f32", "%0")
    assert n.op == "add" and n.shape == (4, 4) and n.dtype == "f32"
    assert n.label == ""


def test_printer_matches_documented_format():
    g = _gelu_graph()
    g.nodes[0].shape = g.nodes[1].shape = (4096, 4096)
    for n in g.nodes:
        n.shape = (4096, 4096)
    expected = (
        "%0 = load f32[4096,4096]   # x\n"
        "%1 = load f32[4096,4096]   # y\n"
        "%2 = add(%0, %1) f32[4096,4096]\n"
        "%3 = gelu(%2) f32[4096,4096]\n"
        "%4 = store(%3)"
    )
    assert print_graph(g) == expected


def test_canonical_is_shape_agnostic():
    g_small = _gelu_graph()
    sh = (1024, 1024)
    x = Node("load", [], sh, "f32", "%0", "x")
    y = Node("load", [], sh, "f32", "%1", "y")
    a = Node("add", [x, y], sh, "f32", "%2")
    gg = Node("gelu", [a], sh, "f32", "%3")
    s = Node("store", [gg], sh, "f32", "%4")
    g_big = Graph(inputs=[x, y], outputs=[s], nodes=[x, y, a, gg, s])
    assert canonical(g_small) == canonical(g_big)


def test_dce_removes_dead_node():
    arr = np.zeros((8,), np.float32)

    def f(a, b):
        live = a + b
        _dead = exp(a)  # never returned
        return gelu(live)

    g = trace(f, (arr, arr))
    ops_before = [n.op for n in g.nodes]
    assert "exp" in ops_before

    g2 = dead_code_elimination(g)
    ops_after = [n.op for n in g2.nodes]
    assert "exp" not in ops_after
    assert ops_after == ["load", "load", "add", "gelu", "store"]


def test_dce_keeps_used_nodes():
    g = _gelu_graph()
    g2 = dead_code_elimination(g)
    assert [n.op for n in g2.nodes] == [n.op for n in g.nodes]


def test_passes_are_pure():
    g = _gelu_graph()
    before = print_graph(g)
    type_propagation(g)
    verify_pass(g)
    dead_code_elimination(g)
    assert print_graph(g) == before  # input graph untouched


def test_codegen_param_names_avoid_collisions():
    # Args named like the output pointer, a local, or a C++ keyword must not
    # produce a clashing or reserved parameter name in the emitted kernel.
    arr = np.zeros((8,), np.float32)

    def f(out, N, idx, float):
        return out + N + idx + float

    src = emit_cuda(run_pipeline(trace(f, (arr, arr, arr, arr))))
    # "out" belongs only to the output pointer; inputs fall back to in0..in3.
    assert src.count("__restrict__ out") == 1
    for i in range(4):
        assert f"const float* __restrict__ in{i}" in src
    assert "int N" in src


def test_type_propagation_sets_result_dtype():
    sh = (4,)
    x = Node("load", [], sh, "f32", "%0", "x")
    a = Node("add", [x, x], sh, "i32", "%1")  # wrong on purpose
    s = Node("store", [a], sh, "i32", "%2")
    g = Graph(inputs=[x], outputs=[s], nodes=[x, a, s])
    g2 = type_propagation(g)
    assert g2.nodes[1].dtype == "f32"  # corrected from inputs


def test_reduce_shape_and_printer():
    arr = np.zeros((4, 8), np.float32)

    def f(a):
        return sum(a, axis=1)

    g = run_pipeline(trace(f, (arr,)))
    red = g.outputs[0].args[0]
    assert red.op == "sum" and red.shape == (4,) and red.attrs["axis"] == 1
    assert is_reduction(g)
    assert "%1 = sum(%0, axis=1) f32[4]" in print_graph(g)


def test_verify_rejects_wrong_reduce_shape():
    x = Node("load", [], (4, 8), "f32", "%0", "x")
    bad = Node("sum", [x], (4, 8), "f32", "%1", attrs={"axis": 1})  # should be (4,)
    s = Node("store", [bad], (4, 8), "f32", "%2")
    with pytest.raises(MLCError, match="reduce shape mismatch"):
        verify_pass(Graph([x], [s], [x, bad, s]))


def test_codegen_rejects_nonterminal_reduction():
    arr = np.zeros((4, 8), np.float32)

    def f(a):
        return relu(sum(a, axis=1))  # reduce feeds an elementwise op

    g = run_pipeline(trace(f, (arr,)))
    with pytest.raises(MLCError, match="single terminal reduction"):
        emit_cuda(g)


def test_matmul_shape_and_printer():
    a = np.zeros((8, 4), np.float32)
    b = np.zeros((4, 6), np.float32)

    def f(x, y):
        return x @ y

    g = run_pipeline(trace(f, (a, b)))
    mm = g.outputs[0].args[0]
    assert mm.op == "matmul" and mm.shape == (8, 6)
    assert is_matmul(g)
    # tiling + shared passes annotated the node
    assert mm.attrs["tile_m"] == 16 and mm.attrs["shared"] is True
    assert "matmul(%0, %1) f32[8,6]   # tile=16x16, shared" in print_graph(g)


def test_matmul_passes_annotate_only_matmul():
    a = np.zeros((4, 4), np.float32)

    def f(x, y):
        return x + y  # no matmul

    g = type_propagation(trace(f, (a, a)))
    g = shared_memory_pass(loop_tiling_pass(g, tile_m=8, tile_n=8))
    for n in g.nodes:
        assert "tile_m" not in n.attrs and "shared" not in n.attrs


def test_verify_rejects_matmul_inner_mismatch():
    a = Node("load", [], (8, 4), "f32", "%0", "a")
    b = Node("load", [], (5, 6), "f32", "%1", "b")  # inner 4 != 5
    mm = Node("matmul", [a, b], (8, 6), "f32", "%2")
    s = Node("store", [mm], (8, 6), "f32", "%3")
    with pytest.raises(MLCError, match="inner dim mismatch"):
        verify_pass(Graph([a, b], [s], [a, b, mm, s]))


def test_canonical_distinguishes_shared_from_naive():
    a = np.zeros((8, 4), np.float32)
    b = np.zeros((4, 6), np.float32)

    def f(x, y):
        return x @ y

    tiled = loop_tiling_pass(type_propagation(trace(f, (a, b))),
                             tile_m=16, tile_n=16)
    naive = canonical(tiled)
    shared = canonical(shared_memory_pass(tiled))
    assert naive != shared  # shared flag is in the cache key


def test_codegen_rejects_matmul_of_computed_values():
    a = np.zeros((4, 4), np.float32)

    def f(x, y):
        return matmul(x + y, y)  # matmul of a computed value

    g = run_pipeline(trace(f, (a, a)))
    with pytest.raises(MLCError, match="input matrices directly"):
        emit_cuda(g)


def test_fusion_fuses_matmul_activation():
    a = np.zeros((4, 4), np.float32)

    def f(x, y):
        return relu(x @ y)  # matmul + activation epilogue

    g = run_pipeline(trace(f, (a, a)))  # pipeline runs fusion_pass
    mm = next(n for n in g.nodes if n.op == "matmul")
    assert mm.attrs.get("epilogue") is True
    # emits a single kernel; relu is applied inline at the output write
    src = emit_cuda(g)
    assert "fmaxf" in src and src.count("__global__") == 1


def test_codegen_rejects_unfused_matmul_epilogue():
    # Without fusion_pass approval, a non-terminal matmul is rejected: the gate
    # that makes fusion_pass meaningful.
    from tinymlc.passes import loop_tiling_pass, shared_memory_pass
    a = np.zeros((4, 4), np.float32)

    def f(x, y):
        return relu(x @ y)

    g = shared_memory_pass(
        loop_tiling_pass(verify_pass(type_propagation(trace(f, (a, a)))),
                         tile_m=16, tile_n=16))  # no fusion_pass
    with pytest.raises(MLCError, match="fusable epilogue"):
        emit_cuda(g)


def test_fusion_rejects_non_load_bias():
    # A "bias" that is itself computed is not a simple load epilogue, so
    # fusion_pass must not approve it and codegen must reject.
    a = np.zeros((4, 4), np.float32)

    def f(x, y):
        return (x @ y) + relu(x)  # second operand is computed, not a load

    g = run_pipeline(trace(f, (a, a)))
    mm = next(n for n in g.nodes if n.op == "matmul")
    assert not mm.attrs.get("epilogue")
    with pytest.raises(MLCError, match="fusable epilogue"):
        emit_cuda(g)


def test_codegen_rejects_non_f32():
    a = np.zeros((4,), np.float16)

    def f(x, y):
        return x + y

    g = run_pipeline(trace(f, (a, a)))
    with pytest.raises(MLCError, match="f32 only"):
        emit_cuda(g)


def test_emitter_does_not_read_dead_input():
    # An unused input is kept as a kernel param (ABI) but must never be
    # dereferenced, or it reads out of bounds when smaller than the output.
    big = np.zeros((256,), np.float32)
    small = np.zeros((4,), np.float32)

    def f(x, unused):
        return relu(x)

    src = emit_cuda(run_pipeline(trace(f, (big, small))))
    assert "__restrict__ unused" in src   # still a parameter
    assert "unused[" not in src           # but never read


def test_matmul_operand_reused_as_bias_compiles_source():
    # x @ x + x: x is both matmul operands AND the bias. The bias load must be
    # emitted (read at the output index) even though x is also a GEMM operand.
    a = np.zeros((8, 8), np.float32)

    def f(x):
        return x @ x + x

    g = run_pipeline(trace(f, (a,)))
    src = emit_cuda(g)
    # the epilogue reads x at the output index for the bias add
    assert "[row * N + col]" in src
    # every cvar referenced in the finish block is also declared there
    assert "= v0" not in src or "float v0 = " in src
