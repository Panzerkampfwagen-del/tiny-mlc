"""shared_memory_pass: mark matmul nodes to be lowered to a shared-memory tiled
kernel. Requires tile sizes (loop_tiling_pass runs first). Codegen reads the
"shared" flag and emits the blocked GEMM that stages tiles through __shared__;
without this pass a matmul lowers to the naive global-memory kernel.

Pure Graph -> Graph, unchanged interface for the Phase 2 MLIR port.
"""

from ..ir import Graph, Node, rebuild


def shared_memory_pass(graph: Graph) -> Graph:
    def make(node: Node, new_args: list[Node]) -> Node:
        attrs = dict(node.attrs)
        if node.op == "matmul":
            attrs["shared"] = True
        return Node(node.op, new_args, node.shape, node.dtype, node.name,
                    node.label, attrs)

    return rebuild(graph, make)
