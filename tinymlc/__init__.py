"""tinymlc: a Python DSL -> custom IR -> CUDA C++ compiler.

Public surface:
  @mlc.jit            decorate a function to trace, compile, cache, and launch
  mlc.benchmark       timing harness
  mlc.gelu/relu/silu/exp/log/sqrt/neg   unary ops used inside a jit function
  (+, -, *, /, unary -)                 binary/neg ops via Tensor operators
"""

from .benchmark import benchmark, eval_numpy
from .frontend import (
    exp,
    gelu,
    jit,
    log,
    matmul,
    max,
    neg,
    relu,
    silu,
    sqrt,
    sum,
)

__all__ = [
    "jit",
    "benchmark",
    "eval_numpy",
    "gelu",
    "relu",
    "silu",
    "exp",
    "log",
    "sqrt",
    "neg",
    "sum",
    "max",
    "matmul",
]
