"""type_propagation: derive each node's output dtype from its inputs.

The tracer guesses a dtype while building the graph; this pass is the authority.
For elementwise ops the result dtype is the promotion of the input dtypes (all
equal in Stage 1, so it is just that dtype). Pure: returns a new Graph.
"""

from ..ir import Graph, MLCError, Node, rebuild

# Higher wins when input dtypes differ. Equal ranks keep the first input's type.
_RANK = {"f32": 3, "bf16": 2, "f16": 2, "i32": 1}


def _promote(dtypes: list[str]) -> str:
    best = dtypes[0]
    for d in dtypes[1:]:
        if d not in _RANK:
            raise MLCError(f"unknown dtype in propagation: {d!r}")
        if _RANK[d] > _RANK[best]:
            best = d
    return best


def type_propagation(graph: Graph) -> Graph:
    def make(node: Node, new_args: list[Node]) -> Node:
        if node.op == "load":
            dtype = node.dtype
        else:
            dtype = _promote([a.dtype for a in new_args])
        return Node(node.op, new_args, node.shape, dtype, node.name,
                    node.label, dict(node.attrs))

    return rebuild(graph, make)
