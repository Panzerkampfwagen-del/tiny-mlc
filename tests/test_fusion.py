"""Stage 4 fusion: matmul + elementwise activation/bias epilogue, and the
already-fused elementwise chains. Validated against numpy. Requires a GPU."""

import numpy as np
import pytest

import tinymlc as mlc

pytestmark = pytest.mark.gpu


def _gpu_available() -> bool:
    try:
        from tinymlc import cuda_driver as cd

        p = cd.cuda_malloc(16)
        cd.cuda_free(p)
        return True
    except Exception:
        return False


if not _gpu_available():
    pytest.skip("no CUDA GPU available", allow_module_level=True)


def _relu(x): return np.maximum(x, np.float32(0))
def _silu(x): return x / (np.float32(1) + np.exp(-x))


def _gelu(x):
    c = np.float32(0.7978845608028654)
    return np.float32(0.5) * x * (
        np.float32(1) + np.tanh(c * (x + np.float32(0.044715) * x * x * x)))


def _rand(*shape):
    return np.random.randn(*shape).astype(np.float32)


def _check(fn, ref, *arrays, rtol=1e-3, atol=2e-2):
    out = fn(*arrays)
    expect = ref(*arrays)
    assert out.shape == expect.shape
    assert np.allclose(out, expect, rtol=rtol, atol=atol), \
        f"max error {np.abs(out - expect).max()}"


def test_matmul_relu():
    @mlc.jit
    def f(a, b): return mlc.relu(a @ b)
    _check(f, lambda a, b: _relu(a @ b), _rand(128, 64), _rand(64, 96))


def test_matmul_gelu():
    @mlc.jit
    def f(a, b): return mlc.gelu(a @ b)
    _check(f, lambda a, b: _gelu(a @ b), _rand(96, 80), _rand(80, 48))


def test_matmul_bias():
    @mlc.jit
    def f(a, b, c): return a @ b + c
    _check(f, lambda a, b, c: a @ b + c,
           _rand(64, 32), _rand(32, 48), _rand(64, 48))


def test_matmul_bias_relu():
    @mlc.jit
    def f(a, b, c): return mlc.relu(a @ b + c)
    _check(f, lambda a, b, c: _relu(a @ b + c),
           _rand(130, 70), _rand(70, 50), _rand(130, 50))  # non-tile-multiple


def test_matmul_epilogue_chain():
    @mlc.jit
    def f(a, b, c): return mlc.silu(mlc.relu(a @ b) + c)
    _check(f, lambda a, b, c: _silu(_relu(a @ b) + c),
           _rand(64, 64), _rand(64, 64), _rand(64, 64))


def test_fused_matmul_is_single_kernel():
    @mlc.jit
    def f(a, b, c): return mlc.relu(a @ b + c)

    a, b, c = _rand(64, 48), _rand(48, 32), _rand(64, 32)
    kernel = f.get_kernel(a, b, c)
    # A fused GEMM launches as a 2D TILExTILE grid with (M, N, K) params.
    assert kernel.block == (16, 16, 1)
    assert kernel.extra_ints == [64, 32, 48]


def test_long_elementwise_chain_is_one_kernel():
    # Elementwise chains are fused by construction: the whole DAG lowers to a
    # single kernel. Checked on the emitted source so it is cache-independent.
    from tinymlc.codegen import emit_cuda
    from tinymlc.frontend import trace
    from tinymlc.pipeline import run_pipeline

    @mlc.jit
    def f(x, y, z, w):
        return mlc.relu(mlc.gelu(mlc.silu(x + y) * z) - w)

    a = _rand(256, 256)
    src = emit_cuda(run_pipeline(trace(f.fn, (a, a, a, a))))
    assert src.count("__global__") == 1  # one kernel for the whole chain

    out = f(a, a, a, a)
    z = _silu(a + a) * a
    ref = _relu(_gelu(z) - a)
    assert np.allclose(out, ref, rtol=1e-3, atol=1e-3)


def test_fused_matches_numpy_large():
    @mlc.jit
    def f(a, b, c): return mlc.gelu(a @ b + c)
    n = 512
    _check(f, lambda a, b, c: _gelu(a @ b + c),
           _rand(n, n), _rand(n, n), _rand(n, n), atol=5e-2)


def test_matmul_operand_reused_as_bias():
    # x @ x + x: the matmul operand is also the residual bias. Regression for
    # the undeclared-cvar compile failure.
    @mlc.jit
    def f(x):
        return x @ x + x

    x = _rand(48, 48)
    out = f(x)
    assert np.allclose(out, x @ x + x, rtol=1e-3, atol=1e-2)
