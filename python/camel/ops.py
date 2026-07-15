from __future__ import annotations
from ._c import c
from ctypes import POINTER, c_double
import numpy as np
from typing import Tuple

class Vbuf:
    def __init__(self, data: np.ndarray):
        self.data = np.ascontiguousarray(data, np.float64)
        self.ptr = self.data.ctypes.data_as(POINTER(c_double))
        self.shape = self.data.shape
    
    @staticmethod # factory
    def zeros(n: int, m: int) -> Vbuf:
        return Vbuf(np.zeros(shape=(n, m)))
    
    @staticmethod
    def shape_eq(A: Vbuf, B: Vbuf) -> bool:
        return A.shape[0] == B.shape[0] and A.shape[1] == B.shape[1]

        

class Ops:
    # wrapping c functions
    @staticmethod
    def matmul_forward(A: Vbuf, B: Vbuf) -> Vbuf:
        result_buf = Vbuf.zeros(A.shape[0], B.shape[1])

        c.matmul_forward(A.ptr, B.ptr, result_buf.ptr, A.shape[0], A.shape[1], B.shape[1])

        return result_buf

    @staticmethod
    def matmul_backward(A: Vbuf, B: Vbuf, grad_out: Vbuf) -> tuple[Vbuf, Vbuf]:
        dA_buf = Vbuf.zeros(A.shape[0], A.shape[1])
        dB_buf = Vbuf.zeros(A.shape[1], B.shape[1])

        c.matmul_backward(A.ptr, B.ptr, grad_out.ptr, dA_buf.ptr, dB_buf.ptr, A.shape[0], A.shape[1], B.shape[1])

        return (dA_buf, dB_buf)

    @staticmethod
    def add_forward(A: Vbuf, B:Vbuf) -> Vbuf:
        # matadd broadcast forward
        out = Vbuf(A.data.copy()) # add kernel is inplace (fix?)

        c.matadd_broadcast_forward(out.ptr, B.ptr, out.shape[0], out.shape[1])

        return out
    
    @staticmethod
    def add_backward(grad_out: Vbuf) -> tuple[Vbuf, Vbuf]:
        n = grad_out.shape[0]
        m = grad_out.shape[1]

        dX_buf = Vbuf.zeros(n, m)
        dB_buf = Vbuf.zeros(1, m)

        c.matadd_broadcast_backward(grad_out.ptr, dX_buf.ptr, dB_buf.ptr, n, m)

        return (dX_buf, dB_buf)

    @staticmethod
    def sub_forward(A: Vbuf, B: Vbuf) -> Vbuf:
        n = A.shape[0]
        m = B.shape[1]
        
        out = Vbuf.zeros(n, m)

        c.matsub_forward(A.ptr, B.ptr, out.ptr, n, m)
        
        return out

    @staticmethod
    def sub_backward(grad_out: Vbuf) -> tuple[Vbuf, Vbuf]:
        n = grad_out.shape[0]
        m = grad_out.shape[1]

        dA_buf = Vbuf.zeros(n, m)
        dB_buf = Vbuf.zeros(n, m)

        c.matsub_backward(grad_out.ptr, dA_buf.ptr, dB_buf.ptr, n, m)

        return (dA_buf, dB_buf)

    @staticmethod
    def hadamard_forward(A: Vbuf, B: Vbuf) -> Vbuf:
        n = A.shape[0]
        m = B.shape[1]

        out = Vbuf.zeros(n, m)

        c.hadamard_forward(A.ptr, B.ptr, out.ptr, n, m)

        return out
    
    @staticmethod
    def hadamard_backward(A: Vbuf, B: Vbuf, grad_out: Vbuf) -> tuple[Vbuf, Vbuf]:
        n = A.shape[0]
        m = B.shape[1]

        dA_buf = Vbuf.zeros(n, m)
        dB_buf = Vbuf.zeros(n, m)

        c.hadamard_backward(grad_out.ptr, A.ptr, B.ptr, dA_buf.ptr, dB_buf.ptr, n, m)

        return (dA_buf, dB_buf)
    
    @staticmethod
    def mean_forward(A: Vbuf) -> Vbuf:
        n = A.shape[0]
        m = A.shape[1]

        out = Vbuf.zeros(1, 1) # scalar

        c.matmean_forward(A.ptr, out.ptr, n * m)

        return out
    
    @staticmethod # A is input buf
    def mean_backward(A: Vbuf, grad_out: Vbuf) -> Vbuf:
        n_total = A.shape[0] * A.shape[1]

        dX_buf = Vbuf.zeros(A.shape[0], A.shape[1])
        g = grad_out.data.item()

        c.matmean_backward(dX_buf.ptr, n_total, g)

        return dX_buf

    @staticmethod
    def tanh_forward(Z: Vbuf) -> Vbuf:
        n = Z.shape[0]
        m = Z.shape[1]
        out = Vbuf.zeros(n, m)

        c.tanh_forward(Z.ptr, out.ptr, n, m)

        return out
    
    @staticmethod
    def tanh_backward(out: Vbuf, grad_out: Vbuf) -> Vbuf:
        if not Vbuf.shape_eq(out, grad_out):
            raise AttributeError
        
        n = out.shape[0]
        m = out.shape[1]

        dZ_buf = Vbuf.zeros(n, m)

        c.tanh_backward(out.ptr, grad_out.ptr, dZ_buf.ptr, n, m)

        return dZ_buf