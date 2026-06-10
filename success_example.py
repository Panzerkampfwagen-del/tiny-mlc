"""Phase 1 success criterion: trace, compile, validate against numpy, benchmark.

Run from the repo root with a CUDA-capable Python:
    python success_example.py
"""

import numpy as np

import tinymlc as mlc


@mlc.jit
def fused_gelu(x, y):
    return mlc.gelu(x + y)


def gelu_np(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def main() -> None:
    x = np.random.randn(4096, 4096).astype(np.float32)
    y = np.random.randn(4096, 4096).astype(np.float32)

    result = fused_gelu(x, y)
    ref = gelu_np(x + y)
    assert np.allclose(result, ref, atol=1e-5), \
        f"max error: {np.abs(result - ref).max()}"
    print("Validation passed.")

    mlc.benchmark(fused_gelu, x, y)


if __name__ == "__main__":
    main()
