"""Tracing frontend: a Tensor proxy records operations into an IR Graph as the
decorated function executes, and @jit turns a Python function into a kernel that
traces once per (shape, dtype) signature, compiles, caches, and launches.

The frontend depends only on the IR data structures and the public pipeline /
runtime entry points. It does not know how passes or codegen work, which keeps
it untouched in the Phase 2 MLIR port.
"""

import inspect

import numpy as np

from ..ir import DTYPES, Graph, MLCError, Node, reduce_shape

_NP_TO_DTYPE = {npname: d for d, (npname, _, _) in DTYPES.items()}


def np_dtype(arr: np.ndarray) -> str:
    name = arr.dtype.name
    if name not in _NP_TO_DTYPE:
        raise MLCError(f"unsupported numpy dtype: {name}")
    return _NP_TO_DTYPE[name]


class TraceContext:
    """Collects nodes and hands out SSA names during a single trace."""

    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self.counter = 0

    def _name(self) -> str:
        name = f"%{self.counter}"
        self.counter += 1
        return name

    def add(self, node: Node) -> Node:
        self.nodes.append(node)
        return node

    def emit(self, op: str, args: list[Node], shape: tuple[int, ...],
             dtype: str, attrs: dict | None = None) -> "Tensor":
        node = Node(op, list(args), shape, dtype, self._name(),
                    attrs=attrs or {})
        self.add(node)
        return Tensor(node, self)


class Tensor:
    """Proxy that overloads operators to record IR nodes instead of computing."""

    def __init__(self, node: Node, ctx: TraceContext) -> None:
        self._node = node
        self._ctx = ctx

    @property
    def shape(self) -> tuple[int, ...]:
        return self._node.shape

    @property
    def dtype(self) -> str:
        return self._node.dtype

    def _other(self, other: object) -> "Tensor":
        if not isinstance(other, Tensor):
            raise MLCError(
                "Stage 1 ops are tensor-tensor only; got operand of type "
                f"{type(other).__name__}"
            )
        return other

    def _binary(self, other: object, op: str) -> "Tensor":
        rhs = self._other(other)
        # Result shape/dtype follow the lhs; verify_pass enforces consistency.
        return self._ctx.emit(op, [self._node, rhs._node], self.shape, self.dtype)

    def __add__(self, o): return self._binary(o, "add")
    def __sub__(self, o): return self._binary(o, "sub")
    def __mul__(self, o): return self._binary(o, "mul")
    def __truediv__(self, o): return self._binary(o, "div")
    def __neg__(self): return _unary(self, "neg")
    def __matmul__(self, o): return matmul(self, o)

    def __repr__(self) -> str:
        return f"Tensor({self._node.name}, {self.dtype}{list(self.shape)})"


def _unary(x: Tensor, op: str) -> Tensor:
    if not isinstance(x, Tensor):
        raise MLCError(f"{op} expects a traced Tensor, got {type(x).__name__}")
    return x._ctx.emit(op, [x._node], x.shape, x.dtype)


def exp(x: Tensor) -> Tensor: return _unary(x, "exp")
def log(x: Tensor) -> Tensor: return _unary(x, "log")
def sqrt(x: Tensor) -> Tensor: return _unary(x, "sqrt")
def relu(x: Tensor) -> Tensor: return _unary(x, "relu")
def gelu(x: Tensor) -> Tensor: return _unary(x, "gelu")
def silu(x: Tensor) -> Tensor: return _unary(x, "silu")
def neg(x: Tensor) -> Tensor: return _unary(x, "neg")


def _reduce(x: Tensor, op: str, axis: int | None, keepdims: bool) -> Tensor:
    if not isinstance(x, Tensor):
        raise MLCError(f"{op} expects a traced Tensor, got {type(x).__name__}")
    rank = len(x.shape)
    norm = axis
    if norm is not None:
        if norm < 0:
            norm += rank
        if not 0 <= norm < rank:
            raise MLCError(
                f"{op} axis {axis} out of range for rank-{rank} tensor"
            )
    out_shape = reduce_shape(x.shape, norm, keepdims)
    return x._ctx.emit(op, [x._node], out_shape, x.dtype,
                       {"axis": norm, "keepdims": keepdims})


def sum(x: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
    return _reduce(x, "sum", axis, keepdims)


def max(x: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
    return _reduce(x, "max", axis, keepdims)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    if not isinstance(a, Tensor) or not isinstance(b, Tensor):
        raise MLCError("matmul expects two traced Tensors")
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise MLCError(
            f"Stage 3 matmul is 2D only; got {a.shape} @ {b.shape}"
        )
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise MLCError(f"matmul inner dim mismatch: {a.shape} @ {b.shape}")
    return a._ctx.emit("matmul", [a._node, b._node], (m, n), a.dtype)


def trace(fn, arrays: tuple[np.ndarray, ...]) -> Graph:
    """Run fn with Tensor proxies for each array argument and capture the graph."""
    ctx = TraceContext()
    params = list(inspect.signature(fn).parameters)

    inputs: list[Node] = []
    tensors: list[Tensor] = []
    for i, arr in enumerate(arrays):
        label = params[i] if i < len(params) else f"arg{i}"
        node = Node("load", [], tuple(arr.shape), np_dtype(arr), ctx._name(), label)
        ctx.add(node)
        inputs.append(node)
        tensors.append(Tensor(node, ctx))

    result = fn(*tensors)
    returned = result if isinstance(result, (tuple, list)) else (result,)

    outputs: list[Node] = []
    for t in returned:
        if not isinstance(t, Tensor):
            raise MLCError(
                f"@jit function must return Tensor(s), got {type(t).__name__}"
            )
        store = Node("store", [t._node], t.shape, t.dtype, ctx._name())
        ctx.add(store)
        outputs.append(store)

    return Graph(inputs=inputs, outputs=outputs, nodes=list(ctx.nodes))


class CompiledKernel:
    """What @jit returns. Traces/compiles on first call per signature, then
    launches the cached kernel. Recompilation never happens for a seen
    (shape, dtype) signature."""

    def __init__(self, fn) -> None:
        self.fn = fn
        self.__name__ = getattr(fn, "__name__", "kernel")
        self._kernels: dict[tuple, object] = {}

    def _signature(self, arrays: tuple[np.ndarray, ...]) -> tuple:
        return tuple((a.shape, a.dtype.name) for a in arrays)

    def compile(self, arrays: tuple[np.ndarray, ...]):
        """Trace, run passes, codegen+compile, and return a launchable kernel."""
        from ..pipeline import run_pipeline
        from ..runtime.kernel import build_kernel

        graph = trace(self.fn, arrays)
        graph = run_pipeline(graph)
        return build_kernel(graph)

    def get_kernel(self, *arrays: np.ndarray):
        """Return the launchable kernel for this signature, compiling on first
        use. The result is cached, so a repeat signature never recompiles."""
        sig = self._signature(arrays)
        kernel = self._kernels.get(sig)
        if kernel is None:
            kernel = self.compile(arrays)
            self._kernels[sig] = kernel
        return kernel

    def __call__(self, *arrays: np.ndarray) -> np.ndarray:
        return self.get_kernel(*arrays)(*arrays)


def jit(fn) -> CompiledKernel:
    return CompiledKernel(fn)
