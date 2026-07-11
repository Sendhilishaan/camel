from ._c import c
import numpy as np
from .ops import Ops


def numerical_grad(f, x, eps=1e-5):
    """Measure grad of scalar f() w.r.t. array x by central differences.
    Mutates x in place while sweeping, restores each element before moving on."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        Lp = f()
        x[idx] = old - eps
        Lm = f()
        x[idx] = old
        grad[idx] = (Lp - Lm) / (2 * eps)
        it.iternext()
    return grad


def check_matmul():
    n, k, m = 4, 3, 5
    np.random.seed(0)
    A = np.random.randn(n, k)
    B = np.random.randn(k, m)
    G = np.random.randn(n, m)

    def L():
        return float(np.sum(G * Ops.matmul_forward(A, B)))

    dA, dB = Ops.matmul_backward(A, B, G)

    num_dA = numerical_grad(L, A)
    num_dB = numerical_grad(L, B)

    print("dA max err:", np.max(np.abs(dA - num_dA)))
    print("dB max err:", np.max(np.abs(dB - num_dB)))
    assert np.allclose(dA, num_dA, rtol=1e-5, atol=1e-7)
    assert np.allclose(dB, num_dB, rtol=1e-5, atol=1e-7)
    print("matmul grad check PASSED")