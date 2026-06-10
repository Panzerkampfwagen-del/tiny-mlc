"""Stage 3 matmul validated against numpy, naive vs shared. Requires a GPU."""

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


def _rand(*shape):
    return np.random.randn(*shape).astype(np.float32)


def _check(fn, a, b, rtol=1e-3, atol=1e-2):
    out = fn(a, b)
    ref = a @ b
    assert out.shape == ref.shape
    assert np.allclose(out, ref, rtol=rtol, atol=atol), \
        f"max error {np.abs(out - ref).max()}"


@pytest.mark.parametrize("m,k,n", [
    (128, 64, 96),     # all tile-multiples
    (130, 70, 50),     # M, N not multiples of 16
    (33, 17, 41),      # K not a multiple of 16
    (1, 256, 1),       # degenerate vector-like
    (256, 256, 256),   # square
])
def test_matmul_shapes(m, k, n):
    @mlc.jit
    def mm(a, b): return a @ b
    _check(mm, _rand(m, k), _rand(k, n))


def test_matmul_function_and_operator_agree():
    a, b = _rand(64, 48), _rand(48, 32)

    @mlc.jit
    def via_op(x, y): return x @ y

    @mlc.jit
    def via_fn(x, y): return mlc.matmul(x, y)

    assert np.allclose(via_op(a, b), via_fn(a, b))


def test_shared_matches_naive():
    # Build a naive kernel (no shared_memory_pass) and compare to the default
    # shared kernel; both must equal numpy.
    from tinymlc.frontend import trace
    from tinymlc.passes import (
        dead_code_elimination,
        loop_tiling_pass,
        type_propagation,
        verify_pass,
    )
    from tinymlc.runtime.kernel import build_kernel

    a, b = _rand(200, 120), _rand(120, 90)

    def mm(x, y): return x @ y
    g = dead_code_elimination(
        loop_tiling_pass(verify_pass(type_propagation(trace(mm, (a, b)))),
                         tile_m=16, tile_n=16))
    naive = build_kernel(g)

    @mlc.jit
    def mmj(x, y): return x @ y
    shared = mmj.get_kernel(a, b)

    ref = a @ b
    assert np.allclose(naive(a, b), ref, rtol=1e-3, atol=1e-2)
    assert np.allclose(shared(a, b), ref, rtol=1e-3, atol=1e-2)
    assert np.allclose(naive(a, b), shared(a, b), rtol=1e-4, atol=1e-3)


def test_matmul_recompile_free_across_shapes():
    import os
    from tinymlc.runtime.compiler import CACHE_DIR

    @mlc.jit
    def mm(a, b): return a @ b

    mm(_rand(32, 32), _rand(32, 32))
    before = set(os.listdir(CACHE_DIR))
    mm(_rand(48, 16), _rand(16, 24))  # new shapes, same kernel
    assert set(os.listdir(CACHE_DIR)) == before


def test_kernel_rejects_wrong_shape_args():
    # A kernel built for one shape, called with a same-count but wrong-shape
    # array, must be rejected (its launch extents are baked) rather than
    # indexing out of bounds.
    from tinymlc.frontend import trace
    from tinymlc.pipeline import run_pipeline
    from tinymlc.runtime.kernel import build_kernel

    def mm(a, b): return a @ b
    g = run_pipeline(trace(mm, (_rand(16, 16), _rand(16, 16))))
    kernel = build_kernel(g)
    with pytest.raises(ValueError, match="shape"):
        kernel(_rand(8, 8), _rand(8, 8))
