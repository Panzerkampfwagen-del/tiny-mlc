"""loop_tiling_pass: record a tile size on each matmul node.

In this op-level IR a matmul is a single node, so tiling is a decision recorded
as attributes that codegen realizes into a blocked kernel. In the Phase 2 MLIR
port this becomes an actual affine/linalg tiling transform; the interface (a
pure Graph -> Graph that annotates the op) is unchanged. The pipeline binds the
tile sizes via functools.partial so the pass still reads as pass_fn(graph).
"""

from ..ir import Graph, Node, rebuild


def loop_tiling_pass(graph: Graph, tile_m: int, tile_n: int) -> Graph:
    def make(node: Node, new_args: list[Node]) -> Node:
        attrs = dict(node.attrs)
        if node.op == "matmul":
            attrs["tile_m"] = tile_m
            attrs["tile_n"] = tile_n
        return Node(node.op, new_args, node.shape, node.dtype, node.name,
                    node.label, attrs)

    return rebuild(graph, make)
