from ctypes import *
from pathlib import Path

proj_root = Path(__file__).resolve().parent.parent.parent
DLL_PATH = proj_root / "camel.dll"

try:
    c = CDLL(str(DLL_PATH))
except OSError as exc:
    raise RuntimeError(
        f"could not load {DLL_PATH}; build the native kernels first (see readme.md)"
    ) from exc

# arg / restype

c.matmul_forward.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int, c_int]
c.matmul_forward.restype = None

c.matmul_backward.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int, c_int]
c.matmul_backward.restype = None

c.matadd_broadcast_forward.argtypes = [POINTER(c_double), POINTER(c_double), c_int, c_int]
c.matadd_broadcast_forward.restype = None

c.matadd_broadcast_backward.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int]
c.matadd_broadcast_backward.restype = None

c.matsub_forward.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int]
c.matsub_forward.restype = None

c.matsub_backward.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int]
c.matsub_backward.restype = None

c.hadamard_forward.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int]
c.hadamard_forward.restype = None

c.hadamard_backward.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int]
c.hadamard_backward.restype = None

c.matmean_forward.argtypes = [POINTER(c_double), POINTER(c_double), c_int]
c.matmean_forward.restype = None

c.matmean_backward.argtypes = [POINTER(c_double), c_int, c_double]
c.matmean_backward.restype = None

c.tanh_forward.argtypes = [POINTER(c_double), POINTER(c_double), c_int, c_int]
c.tanh_forward.restype = None

c.tanh_backward.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int]
c.tanh_backward.restype = None

c.relu_forward.argtypes = [POINTER(c_double), POINTER(c_double), c_int, c_int]
c.relu_forward.restype = None

c.relu_backward.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int]
c.relu_backward.restype = None

c.softmax_xent_forward.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int]
c.softmax_xent_forward.restype = None

# grad_out is a plain scalar double, like matmean_backward
c.softmax_xent_backward.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_double, c_int, c_int]
c.softmax_xent_backward.restype = None

# Apple simd/simd.h kernels, present only on Darwin builds (see Makefile).
# Same signatures as the naive functions above, so mirror instead of retyping.
_KERNEL_NAMES = [
    "matmul_forward", "matmul_backward",
    "matadd_broadcast_forward", "matadd_broadcast_backward",
    "matsub_forward", "matsub_backward",
    "hadamard_forward", "hadamard_backward",
    "matmean_forward", "matmean_backward",
    "tanh_forward", "tanh_backward",
    "relu_forward", "relu_backward",
    "softmax_xent_forward", "softmax_xent_backward",
]

SIMD_AVAILABLE = hasattr(c, "matmul_forward_simd")

if SIMD_AVAILABLE:
    for _name in _KERNEL_NAMES:
        _naive_fn = getattr(c, _name)
        _simd_fn = getattr(c, f"{_name}_simd")
        _simd_fn.argtypes = _naive_fn.argtypes
        _simd_fn.restype = _naive_fn.restype

# GPU kernels use float32 staging and the same opaque resident buffer API.
_GPU_TYPE_MAP = {POINTER(c_double): POINTER(c_float), c_double: c_float}
METAL_AVAILABLE = hasattr(c, "matmul_forward_metal")
CUDA_AVAILABLE = hasattr(c, "matmul_forward_cuda")

_GPU_SIGNATURES = {
    "buffer_create": ([POINTER(c_float), c_int], c_void_p),
    "buffer_read": ([c_void_p, POINTER(c_float), c_int], None),
    "buffer_free": ([c_void_p], None),
    "matmul_forward": ([c_void_p, c_void_p, c_int, c_int, c_int], c_void_p),
    "matmul_backward": ([c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, POINTER(c_void_p), POINTER(c_void_p)], None),
    "matadd_broadcast_forward": ([c_void_p, c_void_p, c_int, c_int], c_void_p),
    "matadd_broadcast_backward": ([c_void_p, c_int, c_int, POINTER(c_void_p), POINTER(c_void_p)], None),
    "matsub_forward": ([c_void_p, c_void_p, c_int, c_int], c_void_p),
    "matsub_backward": ([c_void_p, c_int, c_int, POINTER(c_void_p), POINTER(c_void_p)], None),
    "hadamard_forward": ([c_void_p, c_void_p, c_int, c_int], c_void_p),
    "hadamard_backward": ([c_void_p, c_void_p, c_void_p, c_int, c_int, POINTER(c_void_p), POINTER(c_void_p)], None),
    "matmean_forward": ([c_void_p, c_int], c_void_p),
    "matmean_backward": ([c_int, c_float], c_void_p),
    "tanh_forward": ([c_void_p, c_int, c_int], c_void_p),
    "tanh_backward": ([c_void_p, c_void_p, c_int, c_int], c_void_p),
    "relu_forward": ([c_void_p, c_int, c_int], c_void_p),
    "relu_backward": ([c_void_p, c_void_p, c_int, c_int], c_void_p),
    "softmax_xent_forward": ([c_void_p, c_void_p, c_int, c_int, POINTER(c_void_p), POINTER(c_float)], None),
    "softmax_xent_backward": ([c_void_p, c_void_p, c_float, c_int, c_int], c_void_p),
    "add": ([c_void_p, c_void_p, c_int], c_void_p),
    "sgd_step": ([c_void_p, c_void_p, c_void_p, c_float, c_float, c_int], None),
    "adagrad_step": ([c_void_p, c_void_p, c_void_p, c_float, c_float, c_int], None),
    "rmsprop_step": ([c_void_p, c_void_p, c_void_p, c_float, c_float, c_float, c_int], None),
    "adam_step": ([c_void_p, c_void_p, c_void_p, c_void_p, c_float, c_float, c_float, c_float, c_float, c_float, c_int], None),
}


def _cuda_check(result, function, args):
    error = c.camel_cuda_get_last_error()
    if error:
        raise RuntimeError(f"{function.__name__}: {error.decode('utf-8', errors='replace')}")
    return result


if CUDA_AVAILABLE:
    c.camel_cuda_get_last_error.argtypes = []
    c.camel_cuda_get_last_error.restype = c_char_p
    c.camel_cuda_synchronize.argtypes = []
    c.camel_cuda_synchronize.restype = None
    c.camel_cuda_synchronize.errcheck = _cuda_check

for _backend, _available in (("metal", METAL_AVAILABLE), ("cuda", CUDA_AVAILABLE)):
    if not _available:
        continue
    _probe = getattr(c, f"camel_{_backend}_device_available")
    _probe.argtypes, _probe.restype = [], c_int
    for _name in _KERNEL_NAMES:
        _naive_fn = getattr(c, _name)
        _fn = getattr(c, f"{_name}_{_backend}")
        _fn.argtypes = [_GPU_TYPE_MAP.get(t, t) for t in _naive_fn.argtypes]
        _fn.restype = _naive_fn.restype
        if _backend == "cuda":
            _fn.errcheck = _cuda_check
    for _name, (_args, _result) in _GPU_SIGNATURES.items():
        _symbol = f"camel_{_backend}_{_name}" if _name.startswith("buffer_") else f"{_name}_{_backend}_resident"
        _fn = getattr(c, _symbol)
        _fn.argtypes, _fn.restype = _args, _result
        if _backend == "cuda" and _name != "buffer_free":
            _fn.errcheck = _cuda_check
