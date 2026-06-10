"""Stage 2 reductions validated against numpy. Requires a GPU."""

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


X = np.random.randn(512, 768).astype(np.float32)


def _check(fn, ref, *arrays, rtol=1e-4, atol=1e-4):
    out = fn(*arrays)
    expect = ref(*arrays)
    assert out.shape == expect.shape
    assert np.allclose(out, expect, rtol=rtol, atol=atol), \
        f"max error {np.abs(out - expect).max()}"


def test_sum_axis1():
    @mlc.jit
    def f(a): return mlc.sum(a, axis=1)
    _check(f, lambda a: np.sum(a, axis=1), X)


def test_sum_axis0():
    @mlc.jit
    def f(a): return mlc.sum(a, axis=0)
    _check(f, lambda a: np.sum(a, axis=0), X)


def test_max_axis1():
    @mlc.jit
    def f(a): return mlc.max(a, axis=1)
    _check(f, lambda a: np.max(a, axis=1), X)


def test_max_axis0():
    @mlc.jit
    def f(a): return mlc.max(a, axis=0)
    _check(f, lambda a: np.max(a, axis=0), X)


def test_negative_axis():
    @mlc.jit
    def f(a): return mlc.max(a, axis=-1)
    _check(f, lambda a: np.max(a, axis=-1), X)


def test_keepdims():
    @mlc.jit
    def f(a): return mlc.sum(a, axis=1, keepdims=True)
    _check(f, lambda a: np.sum(a, axis=1, keepdims=True), X)


def test_full_reduction():
    @mlc.jit
    def f(a): return mlc.sum(a)
    # Naive single-thread sequential f32 sum vs numpy pairwise: looser tol.
    _check(f, lambda a: np.sum(a), X, rtol=1e-3, atol=1e-1)


def test_reduction_fuses_elementwise_input():
    @mlc.jit
    def f(a, b): return mlc.sum(mlc.gelu(a + b), axis=1)

    def ref(a, b):
        c = np.float32(0.7978845608028654)
        z = a + b
        g = np.float32(0.5) * z * (
            np.float32(1) + np.tanh(c * (z + np.float32(0.044715) * z * z * z))
        )
        return np.sum(g, axis=1)

    _check(f, ref, X, X)


def test_3d_middle_axis():
    a = np.random.randn(16, 32, 8).astype(np.float32)

    @mlc.jit
    def f(t): return mlc.sum(t, axis=1)
    _check(f, lambda t: np.sum(t, axis=1), a)


def test_reduction_cubin_axis_agnostic():
    # Same op structure, different axes/shapes -> same cubin, no extra compiles.
    import os
    from tinymlc.runtime.compiler import CACHE_DIR

    @mlc.jit
    def f(a): return mlc.sum(a, axis=1)

    f(np.random.randn(8, 8).astype(np.float32))
    before = set(os.listdir(CACHE_DIR))
    f(np.random.randn(4, 16).astype(np.float32))  # new shape, same kernel
    assert set(os.listdir(CACHE_DIR)) == before
