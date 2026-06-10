"""Graph to readable text. This is the primary debugging tool, so the output
is meant to be skimmed. The same renderer is reused inside error messages.

`canonical` produces a stable, shape-agnostic signature of a graph used as the
cache key and kernel symbol name. It omits concrete dimensions on purpose: a
Stage 1 elementwise kernel is flat over N, so one cubin serves every shape with
the same op structure and dtypes.
"""

from .nodes import Graph, Node


def _shape_str(node: Node) -> str:
    dims = ",".join(str(d) for d in node.shape)
    return f"{node.dtype}[{dims}]"


def format_node(node: Node) -> str:
    """One line for a single node, matching the documented IR syntax."""
    if node.op == "load":
        line = f"{node.name} = load {_shape_str(node)}"
        return f"{line}   # {node.label}" if node.label else line
    if node.op == "store":
        srcs = ", ".join(a.name for a in node.args)
        return f"{node.name} = store({srcs})"
    arglist = ", ".join(a.name for a in node.args)
    if "axis" in node.attrs:
        arglist += f", axis={node.attrs['axis']}"
        if node.attrs.get("keepdims"):
            arglist += ", keepdims=True"
    line = f"{node.name} = {node.op}({arglist}) {_shape_str(node)}"
    if node.op == "matmul" and node.attrs:
        tags = []
        if "tile_m" in node.attrs:
            tags.append(f"tile={node.attrs['tile_m']}x{node.attrs['tile_n']}")
        if node.attrs.get("shared"):
            tags.append("shared")
        if node.attrs.get("epilogue"):
            tags.append("+epilogue")
        if tags:
            line += f"   # {', '.join(tags)}"
    return line


def print_graph(graph: Graph) -> str:
    return "\n".join(format_node(n) for n in graph.nodes)


# Attributes that only steer the launch (runtime params), not the kernel source,
# so they stay out of the cache key and keep cubins reusable across them.
_RUNTIME_ONLY_ATTRS = frozenset({"axis", "keepdims"})


def canonical(graph: Graph) -> str:
    """Stable signature: op structure + dtypes + source-affecting attrs, no
    concrete shapes. Attrs that change the emitted code (matmul tile size,
    shared-memory flag) are included; runtime-only attrs (reduce axis) are not.
    """
    parts = []
    for n in graph.nodes:
        args = ",".join(a.name for a in n.args)
        attrs = {k: v for k, v in n.attrs.items()
                 if k not in _RUNTIME_ONLY_ATTRS}
        suffix = ""
        if attrs:
            suffix = "[" + ",".join(f"{k}={attrs[k]}"
                                    for k in sorted(attrs)) + "]"
        parts.append(f"{n.name}={n.op}({args}):{n.dtype}{suffix}")
    return ";".join(parts)
