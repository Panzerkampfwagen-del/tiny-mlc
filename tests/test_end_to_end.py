"""The Stage 1 success criterion: fused_gelu compiles, validates, and the second
call hits the compiled cache instead of recompiling. Requires a GPU."""

import os

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


def gelu_np(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def test_fused_gelu_matches_numpy():
    @mlc.jit
    def fused_gelu(x, y):
        z = x + y
        return mlc.gelu(z)

    x = np.random.randn(4096, 4096).astype(np.float32)
    y = np.random.randn(4096, 4096).astype(np.float32)

    result = fused_gelu(x, y)
    ref = gelu_np(x + y)
    assert np.allclose(result, ref, atol=1e-5), \
        f"max error: {np.abs(result - ref).max()}"


def test_in_memory_cache_skips_recompile():
    @mlc.jit
    def fused_gelu(x, y):
        return mlc.gelu(x + y)

    x = np.random.randn(512, 512).astype(np.float32)
    y = np.random.randn(512, 512).astype(np.float32)

    fused_gelu(x, y)
    kernel = fused_gelu.get_kernel(x, y)
    again = fused_gelu(x, y)
    # Same signature -> same in-memory kernel object, no rebuild.
    assert fused_gelu.get_kernel(x, y) is kernel
    assert again.shape == (512, 512)


def test_disk_cache_reused_by_fresh_compile():
    # A second, independent @jit of the same function has an empty in-memory
    # cache, so it goes through compile_graph -> which must hit the on-disk
    # cubin (same emitted source) and NOT invoke nvcc. Proven by an unchanged
    # cubin mtime across the second compile.
    from tinymlc.runtime.compiler import CACHE_DIR, compile_graph
    from tinymlc.frontend import trace
    from tinymlc.pipeline import run_pipeline

    def body(x, y):
        return mlc.gelu(x + y)

    x = np.random.randn(333, 129).astype(np.float32)  # unusual shape -> own cubin
    y = np.random.randn(333, 129).astype(np.float32)

    g = run_pipeline(trace(body, (x, y)))
    cubin_path, _ = compile_graph(g)          # first compile (may run nvcc)
    mtime = os.path.getmtime(cubin_path)

    cubin_path2, _ = compile_graph(g)         # fresh compile, same source
    assert cubin_path2 == cubin_path
    assert os.path.getmtime(cubin_path2) == mtime  # cache hit, no recompile


def test_recompile_free_across_shapes():
    @mlc.jit
    def f(x, y):
        return mlc.silu(x + y)

    a = np.random.randn(256, 256).astype(np.float32)
    f(a, a)
    from tinymlc.runtime.compiler import CACHE_DIR

    before = set(os.listdir(CACHE_DIR))
    b = np.random.randn(64, 64).astype(np.float32)
    f(b, b)  # new shape, same op structure -> shape-agnostic cubin reused
    after = set(os.listdir(CACHE_DIR))
    assert after == before
