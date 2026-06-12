# tinymlc

A Python DSL → custom IR → CUDA C++ compiler, built from scratch.

No LLVM, no MLIR, no PyTorch on the compilation or runtime path. `numpy` is the
only dependency (used for host arrays, the reference evaluator, and tests).
Every compiler stage is a small, readable, independently testable piece: the
frontend traces, the IR is plain dataclasses, passes are pure functions, codegen
emits CUDA C++, and the runtime drives `nvcc` and the CUDA driver API directly.

The IR and passes are designed so they could be ported to an MLIR backend
without touching the frontend or runtime.

**Demonstrates:** tracing-based compiler frontend, SSA IR design, optimization
passes, CUDA C++ codegen, driver-API kernel launch. Single-GPU; elementwise +
reduction + matmul ops.

```python
import tinymlc as mlc
import numpy as np

@mlc.jit
def fused_gelu(x, y):
    return mlc.gelu(x + y)

x = np.random.randn(4096, 4096).astype(np.float32)
y = np.random.randn(4096, 4096).astype(np.float32)

out = fused_gelu(x, y)          # first call: trace → passes → codegen → nvcc → launch
out = fused_gelu(x, y)          # second call: cached cubin, no recompile
mlc.benchmark(fused_gelu, x, y) # latency / bandwidth / vs numpy
```

## How it works

A call to a `@mlc.jit` function runs this pipeline once per `(shape, dtype)`
signature, then caches and launches:

```
trace ──► IR Graph ──► passes ──► CUDA source ──► nvcc ──► .cubin ──► cuLaunchKernel
(Tensor    (Node/      (pure      (codegen)               (cached)    (driver API)
 proxy)    Graph)       fns)
```

1. **Frontend (`frontend/`)** — tracing, not AST parsing. A `Tensor` proxy
   overloads Python operators (`+ - * /`, unary `-`, `@`) and records each op as
   an IR node as the function executes. `@mlc.jit` runs the function once with
   proxies to capture the graph.
2. **IR (`ir/`)** — plain dataclasses (`Node`, `Graph`); no class hierarchy.
   `printer.py` renders a graph as readable SSA text and is reused inside every
   error message. A `canonical` signature drives the cache key and kernel name.
3. **Passes (`passes/`)** — each is a pure `Graph -> Graph` function; the input
   graph is never mutated. The schedule lives in `pipeline.py`; adding a pass is
   one line.
4. **Codegen (`codegen/`)** — walks a graph and returns a CUDA C++ source
   string. The output op picks the kernel shape (elementwise / reduction /
   matmul).
5. **Runtime (`runtime/`, `cuda_driver.py`)** — `compiler.py` runs `nvcc` (only
   on a cache miss) and caches the `.cubin`; `kernel.py` allocates device memory,
   copies, launches, and copies back. `cuda_driver.py` is a thin `ctypes`
   binding over `libcudart` (memory) and `libcuda` (module load + launch).

### Passes

| Pass | Effect |
|------|--------|
| `type_propagation` | derive each node's output dtype from its inputs |
| `verify_pass` | shape/dtype/arity checks; raises `MLCError` naming the node |
| `fusion_pass` | mark a matmul whose output flows into an elementwise activation/bias chain as fused |
| `loop_tiling_pass(tile_m, tile_n)` | record the matmul tile size |
| `shared_memory_pass` | mark a matmul to lower to the shared-memory tiled kernel |
| `dead_code_elimination` | drop nodes nothing consumes |

Passes that need parameters (tiling) are bound with `functools.partial` in the
pipeline, so every entry stays a uniform `pass_fn(graph)`.

## Supported ops

- **Elementwise:** `add sub mul div neg exp log sqrt relu gelu silu`
  (binary ops via operators; unary as `mlc.<op>(x)`).
- **Reductions:** `mlc.sum` / `mlc.max` with `axis` (int, negative, or `None`)
  and `keepdims`.
- **Matmul:** `mlc.matmul(a, b)` or `a @ b` (2D), as a shared-memory tiled GEMM.
- **Fusion:** matmul + elementwise epilogue, e.g. `relu(a @ b + bias)`, lowered
  into a single GEMM kernel. Elementwise chains are already one kernel by
  construction.

## The IR, in text

`ir/printer.py` is the primary debugging tool. `relu(a @ b + bias)` traces and
lowers to:

```
%0 = load f32[128,64]   # a
%1 = load f32[64,96]   # b
%2 = load f32[128,96]   # bias
%3 = matmul(%0, %1) f32[128,96]   # tile=16x16, shared, +epilogue
%4 = add(%3, %2) f32[128,96]
%5 = relu(%4) f32[128,96]
%6 = store(%5)
```

The matmul, bias add, and relu become one fused kernel; the `add` and `relu`
nodes are computed inline at the GEMM's output write.

## Requirements

- Linux, Python 3.11+, an NVIDIA GPU, the CUDA toolkit with `nvcc` on `PATH`,
  and `libcuda` available (the driver). Developed on an RTX 3050 (sm_86),
  CUDA 12.9. CUDA 12.x and 13.x are both supported.
- `numpy`. All other dependencies are standard-library.

See **Running** below for environment setup, the `nvcc`/library discovery rules,
and how to target a different GPU.

## Running

**0. Conda quickstart (recommended).** An `environment.yml` is provided that
pins all dependencies including the CUDA toolkit, a compatible host compiler
(`gcc` 12, the maximum version supported by nvcc 12.4), and `numpy`/`pytest`:

```bash
conda env create -f environment.yml   # creates the "tinymlc" env
conda activate tinymlc
python success_example.py
```

**1. Use a Python that can see the CUDA toolkit.** `tinymlc` shells out to
`nvcc` and `ctypes`-loads `libcudart`/`libcuda` at runtime, so they must be
discoverable from the interpreter you launch. The simplest check:

```bash
nvcc --version          # toolkit (compile) — must be on PATH
python -c "import numpy" # the only Python dependency
```

If your CUDA toolkit lives in a conda environment, run with **that
environment's** Python so `nvcc` and `libcudart` resolve, e.g.
`/path/to/envs/<env>/bin/python …`. `cuda_driver.py` locates `libcudart` via
`CUDA_HOME`, then `sys.prefix`, then standard paths (including the versioned
`.so.12` / `.so.13` names used by different CUDA releases), and `libcuda`
(the driver) via the usual names including the WSL path — so the active env's
`bin/python` usually needs no extra `LD_LIBRARY_PATH`.

`compiler.py` automatically detects the CUDA headers even when conda's
`cuda-toolkit` places them under `targets/x86_64-linux/include/` rather than
on nvcc's built-in search path.

**2. Run from the repo root** (so `import tinymlc` resolves), or set
`PYTHONPATH`:

```bash
python success_example.py            # trace, validate fused_gelu vs numpy, benchmark
python -m pytest                     # 60 tests; GPU tests auto-skip without a GPU
python -m pytest tests/test_ir.py    # IR + codegen only — runs without a GPU

PYTHONPATH=/path/to/tinymlc python my_script.py   # using it from elsewhere
```

**3. First call compiles; later calls hit the cache.** Per `(shape, dtype)`
signature, the first call traces → runs passes → emits CUDA → invokes `nvcc`;
subsequent calls (and new shapes with the same op structure) reuse the cached
`.cubin` with no `nvcc`. Inspect or reset the cache:

```bash
ls ~/.tinymlc_cache/                 # compiled .cubin files
cat /tmp/tinymlc_<hash>.cu           # the emitted CUDA source, kept for debugging
rm -rf ~/.tinymlc_cache/             # force a clean recompile
```

**4. Targeting another GPU.** Codegen and `nvcc` target `sm_86`. For a different
card, set `ARCH` in `runtime/compiler.py` (e.g. `sm_89` for Ada, `sm_90` for
Hopper) and clear the cache.

**Troubleshooting.** `MLCError: nvcc not found` → `nvcc` isn't on `PATH` for this
interpreter (activate/point at the CUDA env). `could not load libcudart` →
set `CUDA_HOME` or run with the env whose `lib/` holds it. `could not load
libcuda` → no NVIDIA driver present (a toolkit alone isn't enough; you need a
GPU + driver to run, though `tests/test_ir.py` and codegen work without one).

## Design notes

- **`.cubin` + driver-API launch, not `.so` + `ctypes.CDLL`.** A `__global__`
  kernel has no host symbol, so you cannot get a callable handle for it out of a
  `CDLL`-loaded shared library. Launching via the driver API
  (`cuModuleLoad` → `cuModuleGetFunction` → `cuLaunchKernel`) needs a loadable
  module, so the compiler emits a `.cubin`. Device memory still uses the runtime
  API (`cudaMalloc`/`cudaMemcpy`); the two share the device's primary context.
- **Shape-agnostic kernels.** The `canonical` signature omits concrete
  dimensions, so one cubin serves every shape with the same op structure — a new
  shape re-traces but does not recompile. Source-affecting attributes (matmul
  tile size, the shared-memory flag) are part of the key; runtime-only ones
  (reduce axis) are not.
- **Tiling/shared-memory/fusion as annotations.** In this op-level IR a matmul is
  a single node, so these passes record decisions in `Node.attrs` and codegen
  realizes them. They could become real loop/fusion transforms in an MLIR
  backend; the `Graph -> Graph` interface would be unchanged.

## Performance

Honest, measured numbers on an RTX 3050 Laptop (sm_86, ~192 GB/s DRAM,
~5–9 TFLOP/s FP32). Run `mlc.benchmark(...)` to reproduce.

| Workload | Kernel | Notes |
|----------|--------|-------|
| `gelu(x + y)`, 4096² f32 | ~158 GB/s | ~80% of DRAM peak (memory bound) |
| GEMM 1024³ f32, shared tiled | ~630 GFLOP/s | vs ~470 naive (1.36×); ~10% of FP32 peak |
| `relu(a @ b + bias)` 1024³ | ≈ plain GEMM | epilogue fuses at ~0% overhead |

The tiled GEMM is the textbook 16×16 shared-memory kernel — correct and a clear
win over naive, but not a tuned one (no register blocking, vectorized loads, or
double buffering), hence ~10% of peak rather than cuBLAS-class throughput.

## Scope limits

These raise a clear `MLCError`; each needs a multi-kernel runtime, which is out
of scope here:

- A reduction or matmul must be a **single terminal op** in its kernel.
- Prologue fusion (`matmul(x + y, z)`), multiple matmuls, or matmul + reduction
  in one kernel.
- Reduction → elementwise epilogue (e.g. `relu(sum(x, axis=1))`).
- Reductions and matmuls emit **f32** only; full reductions (`axis=None`) use a
  single-thread sequential sum (slow and less accurate than numpy's pairwise
  sum for large inputs — Stage 3-style tree reduction would fix this).

## Layout

```
tinymlc/
├── frontend/    tracer.py (Tensor proxy, @jit), ops.py (op vocabulary)
├── ir/          nodes.py (Node, Graph), printer.py (text + canonical)
├── passes/      type_prop, verify, fusion, tiling, shared_memory, dce
├── pipeline.py  the ordered pass list
├── codegen/     cuda_emitter.py (Graph → CUDA C++)
├── runtime/     compiler.py (nvcc + cubin cache), kernel.py (allocate/launch)
├── cuda_driver.py   ctypes bindings (libcudart + libcuda)
└── benchmark.py     timing harness + numpy reference evaluator
tests/           test_ir (no GPU) + elementwise / reductions / matmul / fusion / end_to_end
```

The numpy-validated end-to-end example lives in
[`success_example.py`](success_example.py).
