from ctypes import *
from pathlib import Path

proj_root = Path(__file__).resolve().parent.parent.parent
DLL_PATH = proj_root / "camel.dll"

c = CDLL(str(DLL_PATH))

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