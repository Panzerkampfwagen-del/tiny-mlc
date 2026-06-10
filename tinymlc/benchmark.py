"""Timing harness. Reports end-to-end latency, kernel-only latency, effective
kernel bandwidth, and speedup over a numpy interpretation of the same graph.

The bandwidth figure is kernel-only (compute, no PCIe transfers) so it is
comparable to the device's DRAM peak. The speedup is end-to-end, the honest cost
of calling the compiled kernel including host<->device copies.
"""

import time

import numpy as np

from .frontend.tracer import CompiledKernel
from .ir import Graph, MLCError, Node

# sm_86 mobile (RTX 3050 Laptop) measured DRAM bandwidth, used only as a label.
_DRAM_PEAK_GBS = 192.0

_NP_UNARY = {
    "neg": lambda a: -a,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "relu": lambda a: np.maximum(a, np.float32(0)),
    "silu": lambda a: a / (np.float32(1) + np.exp(-a)),
    "gelu": lambda a: np.float32(0.5) * a * (
        np.float32(1)
        + np.tanh(np.float32(0.7978845608028654)
                  * (a + np.float32(0.044715) * a * a * a))
    ),
}
_NP_BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
}
_NP_REDUCE = {
    "sum": lambda a, axis, keepdims: np.sum(a, axis=axis, keepdims=keepdims),
    "max": lambda a, axis, keepdims: np.max(a, axis=axis, keepdims=keepdims),
}


def eval_numpy(graph: Graph, arrays: tuple[np.ndarray, ...]) -> np.ndarray:
    """Interpret the IR with numpy. Used for the benchmark's numpy baseline and
    as an independent reference in tests."""
    env: dict[Node, np.ndarray] = {}
    inputs = {n: arrays[i] for i, n in enumerate(graph.inputs)}
    for n in graph.nodes:
        if n.op == "load":
            env[n] = inputs[n]
        elif n.op == "store":
            env[n] = env[n.args[0]]
        elif n.op in _NP_BINARY:
            env[n] = _NP_BINARY[n.op](env[n.args[0]], env[n.args[1]])
        elif n.op in _NP_UNARY:
            env[n] = _NP_UNARY[n.op](env[n.args[0]])
        elif n.op in _NP_REDUCE:
            env[n] = _NP_REDUCE[n.op](
                env[n.args[0]], axis=n.attrs.get("axis"),
                keepdims=n.attrs.get("keepdims", False),
            )
        elif n.op == "matmul":
            env[n] = env[n.args[0]] @ env[n.args[1]]
        else:
            raise MLCError(f"no numpy reference for op {n.op!r} at {n.name}")
    return env[graph.outputs[0]]


def _time(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) / iters


def benchmark(compiled: CompiledKernel, *arrays: np.ndarray,
              iters: int = 50) -> dict:
    """Run and print a timing report. Returns the measured numbers as a dict."""
    if not isinstance(compiled, CompiledKernel):
        raise TypeError("benchmark expects an @mlc.jit function")

    kernel = compiled.get_kernel(*arrays)  # compile/cache before timing

    e2e = _time(lambda: compiled(*arrays), iters=iters, warmup=5)
    kern = kernel.time_kernel(arrays, iters=max(iters, 100), warmup=10)

    np_arrays = tuple(np.ascontiguousarray(a) for a in arrays)
    graph = _traced_graph(compiled, arrays)
    np_time = _time(lambda: eval_numpy(graph, np_arrays), iters=10, warmup=2)
    # Guard against a 0.0 timer reading on a very fast kernel / coarse clock.
    speedup = np_time / e2e if e2e > 0 else float("inf")

    shape = arrays[0].shape
    dtype = arrays[0].dtype.name
    print(f"[tinymlc] {compiled.__name__}  shape={tuple(shape)} dtype={dtype}")
    print(f"  latency  : {e2e * 1e3:.3f} ms  (end-to-end, incl. H2D/D2H)")
    print(f"  kernel   : {kern * 1e3:.3f} ms  (compute only)")

    result = {"latency_ms": e2e * 1e3, "kernel_ms": kern * 1e3,
              "speedup": speedup}

    mm = next((n for n in graph.nodes if n.op == "matmul"), None)
    if mm is not None:
        m, k = mm.args[0].shape
        _, n = mm.args[1].shape
        gflops = (2 * m * n * k) / kern / 1e9 if kern > 0 else float("inf")
        result["gflops"] = gflops
        print(f"  throughput: {gflops:.1f} GFLOP/s  (kernel only, f32)")
    else:
        bytes_moved = sum(kernel.input_nbytes) + kernel.out_nbytes
        bandwidth = bytes_moved / kern / 1e9 if kern > 0 else float("inf")
        result["bandwidth_gbs"] = bandwidth
        print(f"  bandwidth: {bandwidth:.1f} GB/s  (kernel only; "
              f"sm_86 DRAM peak ~{_DRAM_PEAK_GBS:.0f} GB/s)")

    print(f"  vs numpy : {speedup:.2f}x faster (end-to-end)")
    return result


# The graph for a signature is rebuilt cheaply for the numpy baseline. Tracing is
# pure and fast; this keeps benchmark from reaching into kernel internals.
def _traced_graph(compiled: CompiledKernel, arrays: tuple[np.ndarray, ...]):
    from .frontend.tracer import trace
    from .pipeline import run_pipeline

    return run_pipeline(trace(compiled.fn, arrays))
