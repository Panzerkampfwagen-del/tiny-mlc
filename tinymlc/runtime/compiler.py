"""Codegen + nvcc + cache. Turns a Graph into a compiled .cubin on disk.

Deviation from the prose spec: the spec describes compiling to a `.so` and
loading it with ctypes.CDLL. That cannot work for launching a __global__ kernel,
because a CDLL only exposes host symbols; you cannot obtain a CUfunction handle
from it. The cuda_driver.py spec requires launching via cuLaunchKernel with a
module/function handle, which needs a loadable module. So we compile to a
`.cubin` and load it with cuModuleLoad. Everything else (cache, hash, nvcc) is
as specified.

nvcc is only ever invoked on a cache miss; a hit returns the cached path with no
subprocess call.
"""

import hashlib
import os
import shutil
import subprocess
import sys

from ..codegen import emit_cuda, kernel_name
from ..ir import Graph, MLCError

CACHE_DIR = os.path.expanduser("~/.tinymlc_cache")
ARCH = "sm_86"


def _find_nvcc() -> str:
    candidates = [
        shutil.which("nvcc"),
        os.path.join(sys.prefix, "bin", "nvcc"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise MLCError(
        "nvcc not found on PATH or in the active environment; activate the "
        "CUDA toolkit env or set it on PATH"
    )


def _run_nvcc(cu_path: str, cubin_path: str) -> None:
    nvcc = _find_nvcc()
    # Compile to a unique temp file then atomically rename, so a concurrent
    # compile of the same hash can never observe a half-written cubin.
    tmp_out = f"{cubin_path}.{os.getpid()}.tmp"
    cmd = [nvcc, "-O3", f"-arch={ARCH}", "-cubin", "-o", tmp_out, cu_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        raise MLCError(
            "nvcc failed:\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr:\n{proc.stderr}"
        )
    os.replace(tmp_out, cubin_path)


def compile_graph(graph: Graph) -> tuple[str, str]:
    """Return (cubin_path, kernel_name). nvcc runs only on a cache miss.

    The cache key is a hash of the emitted CUDA source itself, not just the
    graph's canonical signature, so any change to codegen invalidates stale
    cubins instead of silently reusing one built by an older version. The kernel
    symbol stays `kernel_<graph_hash>` (embedded in the source), so the function
    lookup still matches. The .cu source is written to /tmp for inspection."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    name = kernel_name(graph)
    source = emit_cuda(graph)
    h = hashlib.sha256(source.encode()).hexdigest()[:16]
    cubin_path = os.path.join(CACHE_DIR, f"tinymlc_{h}.cubin")

    if os.path.exists(cubin_path):
        return cubin_path, name

    cu_path = os.path.join("/tmp", f"tinymlc_{h}.cu")
    with open(cu_path, "w") as f:
        f.write(source)
    _run_nvcc(cu_path, cubin_path)
    return cubin_path, name
