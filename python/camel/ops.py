from ._c import c
from ctypes import POINTER, c_double
import numpy as np
from typing import Tuple

def as_ptr(arr: np.ndarray):
    # np array to a ctypes pointer
    return arr.ctypes.data_as(POINTER(c_double))

class Primitives:
    @staticmethod
    def matmul_forward(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = a.shape[0]
        k = a.shape[1]
        m = b.shape[1]

        if k != b.shape[0]:
            raise AttributeError # deal errors later

        A = as_ptr(a)
        B = as_ptr(b)
        result_buf = np.zeros(shape=(n, m), dtype=float)
        result_ptr = as_ptr(result_buf)

        c.matmul_forward(A, B, result_ptr, n, k, m)

        return result_buf

    @staticmethod
    def matmul_backward(a: np.ndarray, b:np.ndarray, grad_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = a.shape[0]
        k = a.shape[1]
        m = b.shape[1]

        if k != b.shape[0]:
            raise AttributeError

        A = as_ptr(a)
        B = as_ptr(b)
        grad_out_ptr = as_ptr(grad_out)

        dA_buf = np.zeros(shape=(n, k))
        dA_ptr = as_ptr(dA_buf)
        dB_buf = np.zeros(shape=(k, m))
        dB_ptr = as_ptr(dB_buf)

        c.matmul_backward(A, B, grad_out_ptr, dA_ptr, dB_ptr, n, k, m)

        return (dA_buf, dB_buf)