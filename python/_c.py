from ctypes import *
from pathlib import Path

proj_root = Path(__file__).resolve().parent.parent
DLL_PATH = proj_root / "camel.dll"

c = CDLL(DLL_PATH)

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