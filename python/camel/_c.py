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

# Metal kernels: same shape as naive, but MSL has no double, so every
# POINTER(c_double)/c_double slot becomes POINTER(c_float)/c_float.
_METAL_TYPE_MAP = {POINTER(c_double): POINTER(c_float), c_double: c_float}

METAL_AVAILABLE = hasattr(c, "matmul_forward_metal")

if METAL_AVAILABLE:
    for _name in _KERNEL_NAMES:
        _naive_fn = getattr(c, _name)
        _metal_fn = getattr(c, f"{_name}_metal")
        _metal_fn.argtypes = [_METAL_TYPE_MAP.get(t, t) for t in _naive_fn.argtypes]
        _metal_fn.restype = _naive_fn.restype

    c.camel_metal_device_available.argtypes = []
    c.camel_metal_device_available.restype = c_int

    # resident API: void* is an opaque GPU buffer handle, lets a chain of ops
    # stay on the GPU without a CPU round trip between each call
    c.camel_metal_buffer_create.argtypes = [POINTER(c_float), c_int]
    c.camel_metal_buffer_create.restype = c_void_p

    c.camel_metal_buffer_read.argtypes = [c_void_p, POINTER(c_float), c_int]
    c.camel_metal_buffer_read.restype = None

    c.camel_metal_buffer_free.argtypes = [c_void_p]
    c.camel_metal_buffer_free.restype = None

    c.matmul_forward_metal_resident.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int]
    c.matmul_forward_metal_resident.restype = c_void_p

    c.matmul_backward_metal_resident.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, POINTER(c_void_p), POINTER(c_void_p)]
    c.matmul_backward_metal_resident.restype = None

    c.matadd_broadcast_forward_metal_resident.argtypes = [c_void_p, c_void_p, c_int, c_int]
    c.matadd_broadcast_forward_metal_resident.restype = c_void_p

    c.matadd_broadcast_backward_metal_resident.argtypes = [c_void_p, c_int, c_int, POINTER(c_void_p), POINTER(c_void_p)]
    c.matadd_broadcast_backward_metal_resident.restype = None

    c.matsub_forward_metal_resident.argtypes = [c_void_p, c_void_p, c_int, c_int]
    c.matsub_forward_metal_resident.restype = c_void_p

    c.matsub_backward_metal_resident.argtypes = [c_void_p, c_int, c_int, POINTER(c_void_p), POINTER(c_void_p)]
    c.matsub_backward_metal_resident.restype = None

    c.hadamard_forward_metal_resident.argtypes = [c_void_p, c_void_p, c_int, c_int]
    c.hadamard_forward_metal_resident.restype = c_void_p

    c.hadamard_backward_metal_resident.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, POINTER(c_void_p), POINTER(c_void_p)]
    c.hadamard_backward_metal_resident.restype = None

    c.matmean_forward_metal_resident.argtypes = [c_void_p, c_int]
    c.matmean_forward_metal_resident.restype = c_void_p

    c.matmean_backward_metal_resident.argtypes = [c_int, c_float]
    c.matmean_backward_metal_resident.restype = c_void_p

    c.tanh_forward_metal_resident.argtypes = [c_void_p, c_int, c_int]
    c.tanh_forward_metal_resident.restype = c_void_p

    c.tanh_backward_metal_resident.argtypes = [c_void_p, c_void_p, c_int, c_int]
    c.tanh_backward_metal_resident.restype = c_void_p

    c.relu_forward_metal_resident.argtypes = [c_void_p, c_int, c_int]
    c.relu_forward_metal_resident.restype = c_void_p

    c.relu_backward_metal_resident.argtypes = [c_void_p, c_void_p, c_int, c_int]
    c.relu_backward_metal_resident.restype = c_void_p

    c.softmax_xent_forward_metal_resident.argtypes = [c_void_p, c_void_p, c_int, c_int, POINTER(c_void_p), POINTER(c_float)]
    c.softmax_xent_forward_metal_resident.restype = None

    c.softmax_xent_backward_metal_resident.argtypes = [c_void_p, c_void_p, c_float, c_int, c_int]
    c.softmax_xent_backward_metal_resident.restype = c_void_p

    # combines two same-shape gradient buffers on the GPU (Tensor._accum's
    # fan-out case); metal-resident only, nothing else needs this
    c.add_metal_resident.argtypes = [c_void_p, c_void_p, c_int]
    c.add_metal_resident.restype = c_void_p