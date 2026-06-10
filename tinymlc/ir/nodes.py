"""Core IR data structures.

Plain dataclasses, no class hierarchy. Passes treat a Graph as data: they read
it and return a new Graph, never mutating the input. Nodes use identity equality
(eq=False) so they are hashable by object identity, which is what passes want
when they key sets/dicts on nodes of a DAG.
"""

from dataclasses import dataclass, field


class MLCError(Exception):
    """Raised by passes and codegen. Messages name the offending node and op."""


# dtype -> (numpy name, size in bytes, CUDA C type)
DTYPES: dict[str, tuple[str, int, str]] = {
    "f32": ("float32", 4, "float"),
    "f16": ("float16", 2, "__half"),
    "bf16": ("bfloat16", 2, "__nv_bfloat16"),
    "i32": ("int32", 4, "int"),
}


def dtype_size(dtype: str) -> int:
    if dtype not in DTYPES:
        raise MLCError(f"unknown dtype: {dtype!r}")
    return DTYPES[dtype][1]


def ctype(dtype: str) -> str:
    if dtype not in DTYPES:
        raise MLCError(f"unknown dtype: {dtype!r}")
    return DTYPES[dtype][2]


@dataclass(eq=False)
class Node:
    op: str                      # "load", "store", "add", "exp", "sum", ...
    args: list["Node"]           # inputs (empty for load nodes)
    shape: tuple[int, ...]
    dtype: str                   # "f32", "f16", "bf16", "i32"
    name: str                    # SSA name: %0, %1, %2, ...
    label: str = ""              # source name for loads, used by the printer
    attrs: dict = field(default_factory=dict)  # op attributes, e.g. reduce axis


@dataclass
class Graph:
    inputs: list[Node]           # kernel arguments (one Node per tensor arg)
    outputs: list[Node]          # store nodes that are returned
    nodes: list[Node]            # all nodes in topological order


def numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def reduce_shape(shape: tuple[int, ...], axis: int | None,
                 keepdims: bool) -> tuple[int, ...]:
    """Output shape of reducing `shape` along `axis` (None = all axes)."""
    if axis is None:
        return tuple(1 for _ in shape) if keepdims else ()
    if keepdims:
        return tuple(1 if i == axis else d for i, d in enumerate(shape))
    return tuple(d for i, d in enumerate(shape) if i != axis)


def reduce_dims(shape: tuple[int, ...], axis: int | None) -> tuple[int, int, int]:
    """Factor `shape` into (outer, reduce, inner) around `axis`. The reduction
    kernel treats the tensor as [outer, reduce, inner] and reduces the middle."""
    if axis is None:
        return 1, numel(shape), 1
    return numel(shape[:axis]), shape[axis], numel(shape[axis + 1:])


def live_nodes(graph: Graph) -> set["Node"]:
    """Nodes reachable from the graph's outputs by walking args. This is the set
    that actually contributes to a result; inputs not in it are dead (kept only
    so the kernel ABI stays stable). Used by DCE and by codegen, which must not
    emit reads for dead inputs."""
    live: set[Node] = set()
    stack: list[Node] = list(graph.outputs)
    while stack:
        node = stack.pop()
        if node in live:
            continue
        live.add(node)
        stack.extend(node.args)
    return live


def rebuild(graph: Graph, make_node) -> Graph:
    """Build a new Graph by mapping every node through make_node(old, new_args).

    Walks nodes in order, so new_args (already-mapped inputs) are available when
    each node is built. Used by passes that transform nodes one-for-one without
    dropping any. Pure: produces all-new Node objects and a new Graph.
    """
    mapping: dict[Node, Node] = {}
    new_nodes: list[Node] = []
    for n in graph.nodes:
        new_args = [mapping[a] for a in n.args]
        nn = make_node(n, new_args)
        mapping[n] = nn
        new_nodes.append(nn)
    return Graph(
        inputs=[mapping[n] for n in graph.inputs],
        outputs=[mapping[n] for n in graph.outputs],
        nodes=new_nodes,
    )
