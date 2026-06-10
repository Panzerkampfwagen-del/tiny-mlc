"""verify_pass: shape/dtype/arity consistency check.

Stage 1 has no broadcasting, so every input to an elementwise op must share the
same shape and dtype. Failures raise MLCError naming the specific node and op,
e.g. "shape mismatch at %3 (add): lhs f32[4096,4096] vs rhs f32[1024]".
Pure: the graph is not modified; a new Graph wrapper is returned.
"""

from ..ir import Graph, MLCError, Node, reduce_shape
from ..frontend import ops


def _t(node: Node) -> str:
    dims = ",".join(str(d) for d in node.shape)
    return f"{node.dtype}[{dims}]"


def _check_arity(node: Node) -> None:
    try:
        expected = ops.arity(node.op)
    except KeyError:
        raise MLCError(f"unknown op at {node.name}: {node.op!r}")
    if len(node.args) != expected:
        raise MLCError(
            f"arity error at {node.name} ({node.op}): expected {expected} "
            f"arg(s), got {len(node.args)}"
        )


def verify_pass(graph: Graph) -> Graph:
    for node in graph.nodes:
        _check_arity(node)

        if node.op in ops.BINARY_OPS:
            lhs, rhs = node.args
            if lhs.shape != rhs.shape:
                raise MLCError(
                    f"shape mismatch at {node.name} ({node.op}): "
                    f"lhs {_t(lhs)} vs rhs {_t(rhs)}"
                )
            if lhs.dtype != rhs.dtype:
                raise MLCError(
                    f"dtype mismatch at {node.name} ({node.op}): "
                    f"lhs {lhs.dtype} vs rhs {rhs.dtype}"
                )

        if node.args:
            src = node.args[0]
            if node.op in ops.UNARY_OPS or node.op == "store":
                if node.shape != src.shape:
                    raise MLCError(
                        f"shape mismatch at {node.name} ({node.op}): "
                        f"result {_t(node)} vs input {_t(src)}"
                    )

        if node.op in ops.MATMUL_OPS:
            a, b = node.args
            if len(a.shape) != 2 or len(b.shape) != 2:
                raise MLCError(
                    f"matmul at {node.name} needs 2D operands: "
                    f"{_t(a)} @ {_t(b)}"
                )
            if a.shape[1] != b.shape[0]:
                raise MLCError(
                    f"matmul inner dim mismatch at {node.name}: "
                    f"{_t(a)} @ {_t(b)}"
                )
            expected = (a.shape[0], b.shape[1])
            if node.shape != expected:
                raise MLCError(
                    f"matmul shape mismatch at {node.name}: result {_t(node)} "
                    f"vs expected {node.dtype}{list(expected)}"
                )

        if node.op in ops.REDUCE_OPS:
            src = node.args[0]
            axis = node.attrs.get("axis")
            rank = len(src.shape)
            if axis is not None and not 0 <= axis < rank:
                raise MLCError(
                    f"axis {axis} out of range at {node.name} ({node.op}) "
                    f"for input {_t(src)}"
                )
            expected = reduce_shape(src.shape, axis,
                                    node.attrs.get("keepdims", False))
            if node.shape != expected:
                raise MLCError(
                    f"reduce shape mismatch at {node.name} ({node.op}, "
                    f"axis={axis}): result {_t(node)} vs expected "
                    f"{node.dtype}[{','.join(str(d) for d in expected)}]"
                )

    return Graph(inputs=graph.inputs, outputs=graph.outputs, nodes=graph.nodes)
