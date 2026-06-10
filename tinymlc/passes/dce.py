"""dead_code_elimination: drop nodes that nothing consumes and that are not
outputs. Liveness is computed by walking backward from the output (store) nodes.

Inputs are kept live even when unused: they are the kernel's parameters, so
removing one would change the launch ABI the runtime relies on. Pure: returns a
new Graph; existing Node objects are reused unchanged.
"""

from ..ir import Graph, live_nodes


def dead_code_elimination(graph: Graph) -> Graph:
    live = live_nodes(graph)
    live.update(graph.inputs)  # keep inputs as kernel params even if unused

    kept = [n for n in graph.nodes if n in live]
    return Graph(
        inputs=[n for n in graph.inputs if n in live],
        outputs=list(graph.outputs),
        nodes=kept,
    )
