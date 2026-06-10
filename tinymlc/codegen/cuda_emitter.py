"""Graph to CUDA C++ source.

Two kernel shapes, chosen by the graph's output op:

Elementwise (Stage 1): one thread per output element, flat indexing. The whole
graph collapses into one __global__ kernel; each node is one register statement.

Reduction (Stage 2): one thread per *output* element. Each thread walks the
reduce axis, computing the feeding elementwise subexpression inline per element
(no intermediate buffer) and accumulating. The tensor is viewed as
[outer, reduce, inner]; the runtime passes (n_out, reduce, inner) so the kernel
is shape- and axis-agnostic. This is the naive reduction; the shared-memory tree
reduction is Stage 3.

The kernel symbol is `kernel_<hash>` where <hash> is the SHA256 prefix of the
graph's canonical signature, so the cache key and the symbol name always agree.
`extern "C"` keeps the symbol unmangled for cuModuleGetFunction.
"""

import hashlib
import re

from ..frontend.ops import ELEMENTWISE_OPS, MATMUL_OPS, REDUCE_OPS
from ..ir import Graph, MLCError, Node, canonical, ctype, live_nodes, numel

# Math expressed in f32. tanh constant is sqrt(2/pi). emit_cuda enforces f32 for
# every node before any of this runs, so the math below is unconditionally f32.
_GELU_C = "0.7978845608028654f"

_BINARY_C = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def graph_hash(graph: Graph) -> str:
    """8-char SHA256 prefix of the canonical signature. Cache key and symbol
    name both derive from this, so they always agree."""
    return hashlib.sha256(canonical(graph).encode()).hexdigest()[:8]


def kernel_name(graph: Graph) -> str:
    return f"kernel_{graph_hash(graph)}"


def _cvar(node: Node) -> str:
    # %0 -> v0; SSA names are unique so the C identifiers are too.
    return "v" + node.name.lstrip("%")


# Identifiers the kernel body or signature already uses, plus C++ keywords. An
# input param named like any of these (e.g. an arg called "out" or "float")
# would clash with the output pointer, a local, or the language.
_CPP_KEYWORDS = frozenset({
    "alignas", "alignof", "and", "asm", "auto", "bool", "break", "case",
    "catch", "char", "class", "const", "constexpr", "continue", "decltype",
    "default", "delete", "do", "double", "else", "enum", "explicit", "extern",
    "false", "float", "for", "friend", "goto", "if", "inline", "int", "long",
    "namespace", "new", "not", "operator", "or", "private", "protected",
    "public", "register", "return", "short", "signed", "sizeof", "static",
    "struct", "switch", "template", "this", "throw", "true", "try", "typedef",
    "typename", "union", "unsigned", "using", "virtual", "void", "volatile",
    "while", "xor",
})
_RESERVED = _CPP_KEYWORDS | {"out", "idx", "N"}
_CVAR_RE = re.compile(r"^v\d+$")


def _param_name(node: Node, idx: int, taken: set[str]) -> str:
    """A safe, unique C identifier for an input pointer. Uses the source arg name
    when it cannot collide with the output, a local, a cvar, or a keyword;
    otherwise falls back to in<idx>."""
    label = node.label
    if (
        label.isidentifier()
        and label not in _RESERVED
        and not _CVAR_RE.match(label)
        and label not in taken
    ):
        return label
    name = f"in{idx}"
    while name in taken:
        name += "_"
    return name


def _expr(node: Node) -> str:
    """Register-level RHS for one node, referencing its args' C variables."""
    a = [_cvar(arg) for arg in node.args]

    if node.op in _BINARY_C:
        return f"{a[0]} {_BINARY_C[node.op]} {a[1]}"
    if node.op == "neg":
        return f"-{a[0]}"

    if node.op == "exp":
        return f"expf({a[0]})"
    if node.op == "log":
        return f"logf({a[0]})"
    if node.op == "sqrt":
        return f"sqrtf({a[0]})"
    if node.op == "relu":
        return f"fmaxf({a[0]}, 0.0f)"
    if node.op == "silu":
        return f"{a[0]} / (1.0f + expf(-{a[0]}))"
    if node.op == "gelu":
        x = a[0]
        inner = f"{_GELU_C} * ({x} + 0.044715f * {x} * {x} * {x})"
        return f"0.5f * {x} * (1.0f + tanhf({inner}))"

    raise MLCError(f"no codegen for op {node.op!r} at {node.name}")


def _input_names(graph: Graph) -> dict[Node, str]:
    names: dict[Node, str] = {}
    taken: set[str] = set()
    for i, n in enumerate(graph.inputs):
        name = _param_name(n, i, taken)
        names[n] = name
        taken.add(name)
    return names


def _input_params(graph: Graph, in_names: dict[Node, str]) -> list[str]:
    return [
        f"const {ctype(n.dtype)}* __restrict__ {in_names[n]}"
        for n in graph.inputs
    ]


def _header(name: str, params: list[str]) -> str:
    sig = ",\n    ".join(params)
    return f'extern "C" __global__ void {name}(\n    {sig}\n) {{\n'


def _value_stmts(nodes: list[Node], index: str, in_names: dict[Node, str],
                 indent: str) -> list[str]:
    """Statements computing load + elementwise nodes at element `index`."""
    out = []
    for n in nodes:
        if n.op == "load":
            out.append(f"{indent}{ctype(n.dtype)} {_cvar(n)} = "
                       f"{in_names[n]}[{index}];")
        else:
            out.append(f"{indent}{ctype(n.dtype)} {_cvar(n)} = {_expr(n)};")
    return out


def _emit_elementwise(graph: Graph, name: str,
                      in_names: dict[Node, str]) -> str:
    store = graph.outputs[0]
    params = _input_params(graph, in_names)
    params.append(f"{ctype(store.dtype)}* __restrict__ out")
    params.append("int N")

    # Only live nodes are read: a dead input (kept by DCE for the kernel ABI) is
    # a parameter but must not be dereferenced, or it reads out of bounds.
    live = live_nodes(graph)
    compute = [n for n in graph.nodes if n in live and n.op != "store"]
    body = ["    int idx = blockIdx.x * blockDim.x + threadIdx.x;",
            "    if (idx >= N) return;"]
    body += _value_stmts(compute, "idx", in_names, "    ")
    body.append(f"    out[idx] = {_cvar(store.args[0])};")
    return _header(name, params) + "\n".join(body) + "\n}\n"


def _reduce_init_acc(op: str, val: str) -> tuple[str, str]:
    if op == "sum":
        return "0.0f", f"acc += {val};"
    if op == "max":
        return "-INFINITY", f"acc = fmaxf(acc, {val});"
    raise MLCError(f"no reduction codegen for op {op!r}")


def _emit_reduce(graph: Graph, name: str, in_names: dict[Node, str],
                 reduce_node: Node) -> str:
    store = graph.outputs[0]
    params = _input_params(graph, in_names)
    params.append(f"{ctype(store.dtype)}* __restrict__ out")
    params += ["int n_out", "int reduce", "int inner"]

    # Live nodes only: a dead input must not be read at the reduce index.
    live = live_nodes(graph)
    feed = [n for n in graph.nodes
            if n in live and n is not reduce_node and n.op != "store"]
    val = _cvar(reduce_node.args[0])
    init, acc = _reduce_init_acc(reduce_node.op, val)

    body = [
        "    int o = blockIdx.x * blockDim.x + threadIdx.x;",
        "    if (o >= n_out) return;",
        "    int outer_idx = o / inner;",
        "    int inner_idx = o % inner;",
        "    int base = outer_idx * reduce * inner + inner_idx;",
        f"    float acc = {init};",
        "    for (int r = 0; r < reduce; r++) {",
        "        int i = base + r * inner;",
    ]
    body += _value_stmts(feed, "i", in_names, "        ")
    body.append(f"        {acc}")
    body.append("    }")
    body.append("    out[o] = acc;")
    return _header(name, params) + "\n".join(body) + "\n}\n"


def _matmul_finish(graph: Graph, mm: Node, in_names: dict[Node, str],
                   indent: str) -> list[str]:
    """Statements that turn the GEMM accumulator into the output write, applying
    any fused elementwise epilogue. The matmul result is `acc`; the epilogue is
    the live elementwise chain on the matmul output, and any input it references
    (a bias) is read at the output index row*N+col.

    The epilogue is derived from liveness + actual references, not by excluding
    op kinds: a dead input (kept for the ABI) is never read, and an input that is
    BOTH a matmul operand and a bias (e.g. `x @ x + x`) still gets its bias load
    emitted even though the GEMM also consumes it from tiles."""
    live = live_nodes(graph)
    epilogue_ops = [n for n in graph.nodes
                    if n in live and n.op in ELEMENTWISE_OPS]
    used_loads = {a for op in epilogue_ops for a in op.args if a.op == "load"}
    emit = set(epilogue_ops) | used_loads
    store_src = graph.outputs[0].args[0]

    stmts = [f"{indent}float {_cvar(mm)} = acc;"]
    for n in graph.nodes:  # graph.nodes is topological, so refs resolve in order
        if n not in emit:
            continue
        if n.op == "load":
            stmts.append(f"{indent}{ctype(n.dtype)} {_cvar(n)} = "
                         f"{in_names[n]}[row * N + col];")
        else:
            stmts.append(f"{indent}{ctype(n.dtype)} {_cvar(n)} = {_expr(n)};")
    stmts.append(f"{indent}out[row * N + col] = {_cvar(store_src)};")
    return stmts


def _matmul_naive_body(graph: Node, mm: Node, a: str, b: str,
                       in_names: dict[Node, str]) -> list[str]:
    return [
        "    int col = blockIdx.x * blockDim.x + threadIdx.x;",
        "    int row = blockIdx.y * blockDim.y + threadIdx.y;",
        "    if (row >= M || col >= N) return;",
        "    float acc = 0.0f;",
        "    for (int k = 0; k < K; k++) {",
        f"        acc += {a}[row * K + k] * {b}[k * N + col];",
        "    }",
    ] + _matmul_finish(graph, mm, in_names, "    ")


def _matmul_shared_body(graph: Graph, mm: Node, a: str, b: str, tile: int,
                        in_names: dict[Node, str]) -> list[str]:
    return [
        f"    __shared__ float As[{tile}][{tile}];",
        f"    __shared__ float Bs[{tile}][{tile}];",
        "    int ty = threadIdx.y;",
        "    int tx = threadIdx.x;",
        f"    int row = blockIdx.y * {tile} + ty;",
        f"    int col = blockIdx.x * {tile} + tx;",
        "    float acc = 0.0f;",
        f"    int ntiles = (K + {tile} - 1) / {tile};",
        "    for (int t = 0; t < ntiles; t++) {",
        f"        int aCol = t * {tile} + tx;",
        f"        int bRow = t * {tile} + ty;",
        f"        As[ty][tx] = (row < M && aCol < K) ? {a}[row * K + aCol] "
        ": 0.0f;",
        f"        Bs[ty][tx] = (bRow < K && col < N) ? {b}[bRow * N + col] "
        ": 0.0f;",
        "        __syncthreads();",
        f"        for (int k = 0; k < {tile}; k++) acc += As[ty][k] * Bs[k][tx];",
        "        __syncthreads();",
        "    }",
        "    if (row < M && col < N) {",
    ] + _matmul_finish(graph, mm, in_names, "        ") + ["    }"]


def _emit_matmul(graph: Graph, name: str, in_names: dict[Node, str],
                 mm: Node) -> str:
    for arg in mm.args:
        if arg.op != "load":
            raise MLCError(
                "matmul takes input matrices directly; a matmul of computed "
                "values (prologue fusion) needs a multi-kernel runtime "
                "(later stage)"
            )
    a, b = in_names[mm.args[0]], in_names[mm.args[1]]
    tile_m = mm.attrs.get("tile_m", 16)
    tile_n = mm.attrs.get("tile_n", 16)
    if tile_m != tile_n:
        raise MLCError(
            f"matmul uses square tiles; got tile_m={tile_m}, tile_n={tile_n}"
        )

    params = _input_params(graph, in_names)
    params += [f"{ctype(mm.dtype)}* __restrict__ out", "int M", "int N", "int K"]
    if mm.attrs.get("shared"):
        body = _matmul_shared_body(graph, mm, a, b, tile_m, in_names)
    else:
        body = _matmul_naive_body(graph, mm, a, b, in_names)
    return _header(name, params) + "\n".join(body) + "\n}\n"


def emit_cuda(graph: Graph) -> str:
    if len(graph.outputs) != 1:
        raise MLCError(
            f"codegen supports one output, got {len(graph.outputs)}"
        )

    # Single dtype policy: codegen emits f32 math only. f16/bf16/i32 are accepted
    # by the frontend/IR but not by this backend, so reject them once here with a
    # clear error rather than emitting code that fails in nvcc or mixes types.
    for n in graph.nodes:
        if n.dtype != "f32":
            raise MLCError(
                f"codegen supports f32 only; got {n.dtype} at {n.name} "
                f"({n.op})"
            )

    in_names = _input_names(graph)
    name = kernel_name(graph)
    out_op = graph.outputs[0].args[0]
    matmuls = [n for n in graph.nodes if n.op in MATMUL_OPS]
    reduces = [n for n in graph.nodes if n.op in REDUCE_OPS]

    if matmuls:
        mm = matmuls[0]
        if len(matmuls) > 1 or reduces:
            raise MLCError(
                "codegen supports one matmul and no reductions in the same "
                "kernel; multiple matmuls or matmul+reduce need a multi-kernel "
                "runtime (later stage)"
            )
        if mm is not out_op and not mm.attrs.get("epilogue"):
            raise MLCError(
                "matmul feeds further ops but is not a fusable epilogue; only a "
                "single elementwise activation/bias chain on the matmul output "
                "fuses (run fusion_pass). Prologue/multi-kernel fusion is a "
                "later stage"
            )
        return _emit_matmul(graph, name, in_names, mm)

    if reduces:
        if len(reduces) > 1 or reduces[0] is not out_op:
            raise MLCError(
                "Stage 2 supports a single terminal reduction; a reduction "
                "feeding further ops or multiple reductions needs a "
                "multi-kernel runtime (later stage)"
            )
        return _emit_reduce(graph, name, in_names, out_op)
    return _emit_elementwise(graph, name, in_names)


def output_numel(graph: Graph) -> int:
    return numel(graph.outputs[0].shape)


def output_kind(graph: Graph) -> str:
    # A matmul (possibly with a fused elementwise epilogue) determines the kernel
    # shape wherever it sits, so detect it anywhere in the graph.
    if any(n.op in MATMUL_OPS for n in graph.nodes):
        return "matmul"
    if graph.outputs[0].args[0].op in REDUCE_OPS:
        return "reduce"
    return "elementwise"


def is_reduction(graph: Graph) -> bool:
    return output_kind(graph) == "reduce"


def is_matmul(graph: Graph) -> bool:
    return output_kind(graph) == "matmul"
