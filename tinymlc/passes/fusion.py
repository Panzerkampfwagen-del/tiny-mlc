"""fusion_pass: fuse a matmul with its elementwise activation/bias epilogue.

Elementwise chains are already a single kernel by construction, and a reduction
already inlines its feeding elementwise expression, so the new capability here is
matmul + epilogue: a matmul whose result flows through a single-use chain of
elementwise ops (relu/gelu/..., or a binary op with an (M,N) bias input) to the
store. When that pattern holds, the matmul is marked with attrs["epilogue"] and
codegen lowers the whole thing into one GEMM kernel that applies the epilogue at
the output write. The epilogue nodes stay in the graph, so the printer, DCE, and
the cache key all keep working unchanged.

Without this pass a non-terminal matmul is rejected by codegen, so the pass is
what actually enables fusion. Pure Graph -> Graph; unchanged interface for the
Phase 2 MLIR port (where it becomes a producer-consumer fusion transform).
"""

from ..frontend.ops import ELEMENTWISE_OPS
from ..ir import Graph, Node, rebuild


def _consumers(graph: Graph) -> dict[Node, list[Node]]:
    cons: dict[Node, list[Node]] = {n: [] for n in graph.nodes}
    for n in graph.nodes:
        for a in n.args:
            cons[a].append(n)
    return cons


def _should_fuse(graph: Graph) -> bool:
    matmuls = [n for n in graph.nodes if n.op == "matmul"]
    if len(matmuls) != 1:
        return False
    mm = matmuls[0]
    store = graph.outputs[0]
    if store.args[0] is mm:
        return False  # terminal matmul, no epilogue to fuse

    cons = _consumers(graph)
    inputs = set(graph.inputs)
    cur = mm
    while True:
        users = cons[cur]
        if len(users) != 1:
            return False  # used more than once: can't fuse without materializing
        nxt = users[0]
        if nxt is store:
            return True
        if nxt.op not in ELEMENTWISE_OPS:
            return False  # non-elementwise consumer (another matmul/reduce)
        for a in nxt.args:
            if a is cur:
                continue
            # the non-producer operand must be an (M,N) bias input
            if a.op != "load" or a not in inputs or a.shape != mm.shape:
                return False
        cur = nxt


def fusion_pass(graph: Graph) -> Graph:
    fuse = _should_fuse(graph)

    def make(node: Node, new_args: list[Node]) -> Node:
        attrs = dict(node.attrs)
        if fuse and node.op == "matmul":
            attrs["epilogue"] = True
        return Node(node.op, new_args, node.shape, node.dtype, node.name,
                    node.label, attrs)

    return rebuild(graph, make)
