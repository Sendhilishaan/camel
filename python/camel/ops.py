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


class Ops:
    # wrapping c functions
    @staticmethod
    def matmul_forward(A: Vbuf, B: Vbuf) -> Vbuf:
        if A.shape[1] != B.shape[0]:
            raise AttributeError # deal errors later

        result_buf = Vbuf.zeros(A.shape[0], B.shape[1])

        c.matmul_forward(A.ptr, B.ptr, result_buf.ptr, A.shape[0], A.shape[1], B.shape[1])

        return result_buf

    @staticmethod
    def matmul_backward(A: Vbuf, B: Vbuf, grad_out: Vbuf) -> tuple[Vbuf, Vbuf]:

        if A.shape[1] != B.shape[0]:
            raise AttributeError

        dA_buf = Vbuf.zeros(A.shape[0], A.shape[1])
        dB_buf = Vbuf.zeros(A.shape[1], B.shape[1])

        c.matmul_backward(A.ptr, B.ptr, grad_out.ptr, dA_buf.ptr, dB_buf.ptr, A.shape[0], A.shape[1], B.shape[1])

        return (dA_buf, dB_buf)