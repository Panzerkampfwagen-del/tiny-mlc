"""ctypes bindings for CUDA.

Memory lives on the runtime API (libcudart): cuda_malloc/free/memcpy.
Kernel launch lives on the driver API (libcuda): a compiled module is loaded
with cuModuleLoad, the function handle comes from cuModuleGetFunction, and the
launch goes through cuLaunchKernel.

The two APIs share one context: we retain the device's primary context with the
driver API and make it current, and the runtime API allocates into that same
primary context, so device pointers are valid for the driver-API launch.

No CuPy, no PyTorch.
"""

import ctypes
import os
import sys

import numpy as np

# cudaMemcpyKind
_H2D = 1
_D2H = 2


def _load(candidates: list[str], what: str) -> ctypes.CDLL:
    for path in candidates:
        if not path:
            continue
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    raise RuntimeError(
        f"could not load {what}; tried: {[c for c in candidates if c]}"
    )


def _cudart_candidates() -> list[str]:
    bases = [os.environ.get("CUDA_HOME"), sys.prefix, "/usr/local/cuda"]
    out = []
    for base in bases:
        if base:
            out.append(os.path.join(base, "lib", "libcudart.so.12"))
            out.append(os.path.join(base, "lib64", "libcudart.so.12"))
    out += ["libcudart.so.12", "libcudart.so"]
    return out


_cudart = _load(_cudart_candidates(), "libcudart")
_cuda = _load(
    ["libcuda.so.1", "libcuda.so", "/usr/lib/wsl/lib/libcuda.so.1"], "libcuda"
)


def _bind(lib, name, restype, argtypes):
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


# Runtime API (libcudart)
_cudaMalloc = _bind(
    _cudart, "cudaMalloc", ctypes.c_int,
    [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
)
_cudaFree = _bind(_cudart, "cudaFree", ctypes.c_int, [ctypes.c_void_p])
_cudaMemcpy = _bind(
    _cudart, "cudaMemcpy", ctypes.c_int,
    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
)
_cudaDeviceSynchronize = _bind(
    _cudart, "cudaDeviceSynchronize", ctypes.c_int, []
)
_cudaGetErrorString = _bind(
    _cudart, "cudaGetErrorString", ctypes.c_char_p, [ctypes.c_int]
)

# Driver API (libcuda)
_cuInit = _bind(_cuda, "cuInit", ctypes.c_int, [ctypes.c_uint])
_cuDeviceGet = _bind(
    _cuda, "cuDeviceGet", ctypes.c_int,
    [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
)
_cuDevicePrimaryCtxRetain = _bind(
    _cuda, "cuDevicePrimaryCtxRetain", ctypes.c_int,
    [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
)
_cuCtxSetCurrent = _bind(_cuda, "cuCtxSetCurrent", ctypes.c_int, [ctypes.c_void_p])
_cuModuleLoad = _bind(
    _cuda, "cuModuleLoad", ctypes.c_int,
    [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p],
)
_cuModuleGetFunction = _bind(
    _cuda, "cuModuleGetFunction", ctypes.c_int,
    [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p],
)
_cuLaunchKernel = _bind(
    _cuda, "cuLaunchKernel", ctypes.c_int,
    [
        ctypes.c_void_p,  # function
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # grid
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # block
        ctypes.c_uint,    # shared mem bytes
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # kernel params
        ctypes.c_void_p,  # extra
    ],
)
_cuGetErrorString = _bind(
    _cuda, "cuGetErrorString", ctypes.c_int,
    [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
)


def _check_rt(code: int, what: str) -> None:
    if code != 0:
        msg = _cudaGetErrorString(code)
        msg = msg.decode() if msg else str(code)
        raise RuntimeError(f"cuda runtime error in {what}: {msg}")


def _check_drv(code: int, what: str) -> None:
    if code != 0:
        s = ctypes.c_char_p()
        _cuGetErrorString(code, ctypes.byref(s))
        msg = s.value.decode() if s.value else str(code)
        raise RuntimeError(f"cuda driver error in {what}: {msg}")


_ctx: ctypes.c_void_p | None = None


def _ensure_context() -> None:
    """Init the driver and make the device's primary context current (once)."""
    global _ctx
    if _ctx is not None:
        return
    _check_drv(_cuInit(0), "cuInit")
    dev = ctypes.c_int()
    _check_drv(_cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = ctypes.c_void_p()
    _check_drv(
        _cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev),
        "cuDevicePrimaryCtxRetain",
    )
    _check_drv(_cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
    _ctx = ctx


def cuda_malloc(size_bytes: int) -> int:
    """Allocate device memory; returns the device pointer as an int."""
    _ensure_context()
    ptr = ctypes.c_void_p()
    _check_rt(_cudaMalloc(ctypes.byref(ptr), size_bytes), "cudaMalloc")
    return ptr.value or 0


def cuda_free(ptr: int) -> None:
    _check_rt(_cudaFree(ctypes.c_void_p(ptr)), "cudaFree")


def cuda_memcpy_h2d(dst_ptr: int, src: np.ndarray) -> None:
    src = np.ascontiguousarray(src)
    _check_rt(
        _cudaMemcpy(ctypes.c_void_p(dst_ptr), src.ctypes.data, src.nbytes, _H2D),
        "cudaMemcpy(h2d)",
    )


def cuda_memcpy_d2h(dst: np.ndarray, src_ptr: int) -> None:
    assert dst.flags["C_CONTIGUOUS"], "destination array must be contiguous"
    _check_rt(
        _cudaMemcpy(dst.ctypes.data, ctypes.c_void_p(src_ptr), dst.nbytes, _D2H),
        "cudaMemcpy(d2h)",
    )


def cuda_device_synchronize() -> None:
    _check_rt(_cudaDeviceSynchronize(), "cudaDeviceSynchronize")


def load_module(cubin_path: str) -> ctypes.c_void_p:
    """Load a compiled .cubin and return its module handle."""
    _ensure_context()
    mod = ctypes.c_void_p()
    _check_drv(
        _cuModuleLoad(ctypes.byref(mod), cubin_path.encode()), "cuModuleLoad"
    )
    return mod


def get_function(module: ctypes.c_void_p, name: str) -> ctypes.c_void_p:
    fn = ctypes.c_void_p()
    _check_drv(
        _cuModuleGetFunction(ctypes.byref(fn), module, name.encode()),
        "cuModuleGetFunction",
    )
    return fn


def launch_kernel(
    func: ctypes.c_void_p,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    args: list,
    shared_bytes: int = 0,
) -> None:
    """Launch func. `args` is a list of ctypes value objects, one per kernel
    parameter; their addresses are passed as the kernel parameter array."""
    n = len(args)
    params = (ctypes.c_void_p * n)()
    for i, a in enumerate(args):
        params[i] = ctypes.cast(ctypes.byref(a), ctypes.c_void_p)
    _check_drv(
        _cuLaunchKernel(
            func,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            shared_bytes,
            None,
            ctypes.cast(params, ctypes.c_void_p),
            None,
        ),
        "cuLaunchKernel",
    )
