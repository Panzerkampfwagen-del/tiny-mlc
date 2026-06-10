"""Validate every Stage 1 op against a numpy reference. Requires a GPU."""

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


SHAPE = (512, 512)


def _rand(positive: bool = False) -> np.ndarray:
    a = np.random.randn(*SHAPE).astype(np.float32)
    return np.abs(a) + 0.5 if positive else a


def _check(fn, ref, *arrays, rtol=1e-4, atol=1e-5):
    out = fn(*arrays)
    expect = ref(*arrays)
    err = np.abs(out - expect).max()
    assert np.allclose(out, expect, rtol=rtol, atol=atol), f"max error {err}"


def test_add():
    @mlc.jit
    def f(a, b): return a + b
    _check(f, lambda a, b: a + b, _rand(), _rand())


def test_sub():
    @mlc.jit
    def f(a, b): return a - b
    _check(f, lambda a, b: a - b, _rand(), _rand())


def test_mul():
    @mlc.jit
    def f(a, b): return a * b
    _check(f, lambda a, b: a * b, _rand(), _rand())


def test_div():
    @mlc.jit
    def f(a, b): return a / b
    _check(f, lambda a, b: a / b, _rand(), _rand(positive=True))


def test_neg():
    @mlc.jit
    def f(a): return -a
    _check(f, lambda a: -a, _rand())


def test_exp():
    @mlc.jit
    def f(a): return mlc.exp(a)
    _check(f, np.exp, _rand())


def test_log():
    @mlc.jit
    def f(a): return mlc.log(a)
    _check(f, np.log, _rand(positive=True))


def test_sqrt():
    @mlc.jit
    def f(a): return mlc.sqrt(a)
    _check(f, np.sqrt, _rand(positive=True))


def test_relu():
    @mlc.jit
    def f(a): return mlc.relu(a)
    _check(f, lambda a: np.maximum(a, np.float32(0)), _rand())


def test_silu():
    @mlc.jit
    def f(a): return mlc.silu(a)
    _check(f, lambda a: a / (np.float32(1) + np.exp(-a)), _rand())


def test_gelu():
    @mlc.jit
    def f(a): return mlc.gelu(a)

    def ref(a):
        c = np.float32(0.7978845608028654)
        return np.float32(0.5) * a * (
            np.float32(1) + np.tanh(c * (a + np.float32(0.044715) * a * a * a))
        )
    _check(f, ref, _rand())


def test_chained_ops():
    @mlc.jit
    def f(a, b):
        return mlc.relu(a * b - mlc.exp(a))
    _check(f, lambda a, b: np.maximum(a * b - np.exp(a), np.float32(0)),
           _rand(), _rand())


def test_unused_input_does_not_read_out_of_bounds():
    # 'unused' is far smaller than the output extent; the kernel must not read
    # it (regression for the dead-input OOB device read).
    @mlc.jit
    def f(x, unused):
        return mlc.relu(x)

    x = _rand()
    out = f(x, np.zeros(4, np.float32))
    assert np.allclose(out, np.maximum(x, np.float32(0)))
