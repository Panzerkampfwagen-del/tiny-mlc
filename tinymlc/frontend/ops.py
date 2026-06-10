"""Op vocabulary for Stage 1.

Op names are plain strings on Node.op. This module is the single source of truth
for which ops exist and their arity; the tracer uses it to validate, codegen uses
it to pick a C expression.
"""

# Binary elementwise ops, surfaced through Python operators on Tensor.
BINARY_OPS: dict[str, str] = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}

# Unary elementwise ops, surfaced as mlc.<name>(x). "neg" also backs -x.
UNARY_OPS: frozenset[str] = frozenset(
    {"neg", "exp", "log", "sqrt", "relu", "gelu", "silu"}
)

ELEMENTWISE_OPS: frozenset[str] = frozenset(BINARY_OPS) | UNARY_OPS

# Stage 2 reductions, surfaced as mlc.<name>(x, axis=...). Each carries an "axis"
# (int or None) and "keepdims" (bool) in Node.attrs.
REDUCE_OPS: frozenset[str] = frozenset({"sum", "max"})

# Stage 3 matmul, surfaced as mlc.matmul(a, b) or a @ b. 2D only. Tiling passes
# add "tile_m"/"tile_n"/"shared" to Node.attrs.
MATMUL_OPS: frozenset[str] = frozenset({"matmul"})

# Structural ops produced by the tracer, not user-callable.
STRUCTURAL_OPS: frozenset[str] = frozenset({"load", "store"})

ALL_OPS: frozenset[str] = (
    ELEMENTWISE_OPS | REDUCE_OPS | MATMUL_OPS | STRUCTURAL_OPS
)


def arity(op: str) -> int:
    if op in BINARY_OPS or op in MATMUL_OPS:
        return 2
    if op in UNARY_OPS or op in REDUCE_OPS or op == "store":
        return 1
    if op == "load":
        return 0
    raise KeyError(op)
