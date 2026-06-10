"""The pass pipeline: an ordered list of pure Graph -> Graph functions.

Adding a pass means adding one entry to PASSES. Nothing else changes. In the
Phase 2 MLIR port this list becomes the MLIR pass manager's schedule; the order
and the names stay the same.

type_propagation runs first so dtypes are authoritative before verify checks
them; loop_tiling and shared_memory then annotate matmul nodes (no-ops on graphs
without matmul); dead_code_elimination runs last so it cleans up a validated
graph. loop_tiling_pass takes tile sizes, so it is bound with functools.partial
to keep every pipeline entry a uniform pass_fn(graph).
"""

from functools import partial

from .ir import Graph
from .passes import (
    dead_code_elimination,
    fusion_pass,
    loop_tiling_pass,
    shared_memory_pass,
    type_propagation,
    verify_pass,
)

TILE = 16

PASSES = [
    type_propagation,
    verify_pass,
    fusion_pass,
    partial(loop_tiling_pass, tile_m=TILE, tile_n=TILE),
    shared_memory_pass,
    dead_code_elimination,
]


def _pass_name(p) -> str:
    if isinstance(p, partial):
        return p.func.__name__
    return getattr(p, "__name__", repr(p))


def run_pipeline(graph: Graph, verbose: bool = False) -> Graph:
    from .ir import print_graph

    if verbose:
        print("graph before passes:")
        print(print_graph(graph))
    for p in PASSES:
        graph = p(graph)
        if verbose:
            print(f"\nafter {_pass_name(p)}:")
            print(print_graph(graph))
    return graph
