"""CUDAKernel: allocate, copy, launch, copy back, free.

A CUDAKernel is built once per (shape, dtype) signature. It bakes the element
count N, the output shape/dtype, and the launch grid/block (one thread per
output element, 256-thread blocks). Calling it runs the full host<->device
round trip and returns a numpy array.
"""

import ctypes

import numpy as np

from .. import cuda_driver as cd
from ..codegen import output_kind
from ..ir import DTYPES, Graph, dtype_size, numel, reduce_dims
from .compiler import compile_graph

BLOCK = 256


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _np_dtype(ir_dtype: str) -> np.dtype:
    return np.dtype(DTYPES[ir_dtype][0])


class CUDAKernel:
    def __init__(self, graph: Graph, cubin_path: str, name: str) -> None:
        self.name = name
        # Keep the module handle alive for the kernel's lifetime.
        self._module = cd.load_module(cubin_path)
        self._func = cd.get_function(self._module, name)

        self.n_inputs = len(graph.inputs)
        self.input_nbytes = [
            numel(n.shape) * dtype_size(n.dtype) for n in graph.inputs
        ]
        # Expected per-input shape/dtype; the launch extents (N/M/N/K) are baked
        # from this graph, so a same-count call with a different shape would index
        # out of bounds. _check_args rejects it.
        self.input_shapes = [n.shape for n in graph.inputs]
        self.input_dtypes = [_np_dtype(n.dtype) for n in graph.inputs]

        out = graph.outputs[0]
        self.out_shape = out.shape
        self.out_dtype = _np_dtype(out.dtype)
        self.out_nbytes = numel(out.shape) * dtype_size(out.dtype)

        # The launch ABI is read from the graph. extra_ints are the int kernel
        # params appended after the out pointer.
        #   elementwise: one thread per element; params (N)
        #   reduce:      one thread per output element; params (n_out, reduce, inner)
        #   matmul:      2D grid of TILExTILE blocks; params (M, N, K)
        kind = output_kind(graph)
        if kind == "matmul":
            mm = next(node for node in graph.nodes if node.op == "matmul")
            m, k = mm.args[0].shape
            _, n = mm.args[1].shape
            tile = mm.attrs.get("tile_m", 16)
            self.grid = (_ceil_div(n, tile), _ceil_div(m, tile), 1)
            self.block = (tile, tile, 1)
            self.extra_ints = [m, n, k]
        elif kind == "reduce":
            reduce_node = out.args[0]
            in_shape = reduce_node.args[0].shape
            outer, red, inner = reduce_dims(in_shape, reduce_node.attrs["axis"])
            n_out = outer * inner
            self.grid = (_ceil_div(n_out, BLOCK), 1, 1)
            self.block = (BLOCK, 1, 1)
            self.extra_ints = [n_out, red, inner]
        else:
            n = numel(out.shape)
            self.grid = (_ceil_div(n, BLOCK), 1, 1)
            self.block = (BLOCK, 1, 1)
            self.extra_ints = [n]

    def _check_args(self, arrays: tuple[np.ndarray, ...]) -> None:
        if len(arrays) != self.n_inputs:
            raise ValueError(
                f"{self.name}: expected {self.n_inputs} arrays, got {len(arrays)}"
            )
        for i, arr in enumerate(arrays):
            if tuple(arr.shape) != self.input_shapes[i]:
                raise ValueError(
                    f"{self.name}: arg {i} expected shape "
                    f"{self.input_shapes[i]}, got {tuple(arr.shape)}"
                )
            if arr.dtype != self.input_dtypes[i]:
                raise ValueError(
                    f"{self.name}: arg {i} expected dtype "
                    f"{self.input_dtypes[i]}, got {arr.dtype}"
                )

    def _alloc_inputs(self, arrays: tuple[np.ndarray, ...]) -> list[int]:
        ptrs: list[int] = []
        try:
            for arr in arrays:
                arr = np.ascontiguousarray(arr)
                p = cd.cuda_malloc(arr.nbytes)
                ptrs.append(p)            # append before copy so a failed copy frees it
                cd.cuda_memcpy_h2d(p, arr)
            return ptrs
        except Exception:
            for p in ptrs:
                cd.cuda_free(p)
            raise

    def _launch(self, in_ptrs: list[int], out_ptr: int) -> None:
        args = [ctypes.c_void_p(p) for p in in_ptrs]
        args.append(ctypes.c_void_p(out_ptr))
        args += [ctypes.c_int(v) for v in self.extra_ints]
        cd.launch_kernel(self._func, self.grid, self.block, args)

    def __call__(self, *arrays: np.ndarray) -> np.ndarray:
        self._check_args(arrays)
        in_ptrs: list[int] = []
        out_ptr: int | None = None
        try:
            in_ptrs = self._alloc_inputs(arrays)
            out_ptr = cd.cuda_malloc(self.out_nbytes)
            self._launch(in_ptrs, out_ptr)
            cd.cuda_device_synchronize()
            out = np.empty(self.out_shape, dtype=self.out_dtype)
            cd.cuda_memcpy_d2h(out, out_ptr)
            return out
        finally:
            for p in in_ptrs:
                cd.cuda_free(p)
            if out_ptr is not None:
                cd.cuda_free(out_ptr)

    def time_kernel(self, arrays: tuple[np.ndarray, ...], iters: int = 100,
                    warmup: int = 10) -> float:
        """Average kernel-only seconds: buffers are allocated and inputs copied
        once, then only the launch loop is timed (no H2D/D2H, no malloc)."""
        import time

        self._check_args(arrays)
        in_ptrs: list[int] = []
        out_ptr: int | None = None
        try:
            in_ptrs = self._alloc_inputs(arrays)
            out_ptr = cd.cuda_malloc(self.out_nbytes)
            for _ in range(warmup):
                self._launch(in_ptrs, out_ptr)
            cd.cuda_device_synchronize()

            start = time.perf_counter()
            for _ in range(iters):
                self._launch(in_ptrs, out_ptr)
            cd.cuda_device_synchronize()
            return (time.perf_counter() - start) / iters
        finally:
            for p in in_ptrs:
                cd.cuda_free(p)
            if out_ptr is not None:
                cd.cuda_free(out_ptr)


def build_kernel(graph: Graph) -> CUDAKernel:
    cubin_path, name = compile_graph(graph)
    return CUDAKernel(graph, cubin_path, name)
