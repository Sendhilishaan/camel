import numpy as np
from camel.ops import Ops, Vbuf


def numerical_grad(f, vbuf: Vbuf, eps=1e-5):
    """Measure grad of scalar f() w.r.t. Vbuf's underlying data by central differences."""
    grad = np.zeros_like(vbuf.data)
    x = vbuf.data
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
    A = Vbuf(np.random.randn(n, k))
    B = Vbuf(np.random.randn(k, m))
    G = Vbuf(np.random.randn(n, m))

    def L():
        out = Ops.matmul_forward(A, B)
        return float(np.sum(G.data * out.data))

    dA, dB = Ops.matmul_backward(A, B, G)
    num_dA = numerical_grad(L, A)
    num_dB = numerical_grad(L, B)

    print("MATMUL dA max err:", np.max(np.abs(dA.data - num_dA)))
    print("MATMUL dB max err:", np.max(np.abs(dB.data - num_dB)))
    assert np.allclose(dA.data, num_dA, rtol=1e-5, atol=1e-7)
    assert np.allclose(dB.data, num_dB, rtol=1e-5, atol=1e-7)


def check_add():
    n, m = 4, 5
    np.random.seed(1)
    A = Vbuf(np.random.randn(n, m))
    B = Vbuf(np.random.randn(1, m))
    G = Vbuf(np.random.randn(n, m))

    def L():
        out = Ops.add_forward(A, B)
        return float(np.sum(G.data * out.data))

    dA, dB = Ops.add_backward(G)
    num_dA = numerical_grad(L, A)
    num_dB = numerical_grad(L, B)

    print("ADD dA max err:", np.max(np.abs(dA.data - num_dA)))
    print("ADD dB max err:", np.max(np.abs(dB.data - num_dB)))
    assert np.allclose(dA.data, num_dA, rtol=1e-5, atol=1e-7)
    assert np.allclose(dB.data, num_dB, rtol=1e-5, atol=1e-7)


def check_sub():
    n, m = 4, 5
    np.random.seed(2)
    A = Vbuf(np.random.randn(n, m))
    B = Vbuf(np.random.randn(n, m))
    G = Vbuf(np.random.randn(n, m))

    def L():
        out = Ops.sub_forward(A, B)
        return float(np.sum(G.data * out.data))

    dA, dB = Ops.sub_backward(G)
    num_dA = numerical_grad(L, A)
    num_dB = numerical_grad(L, B)

    print("SUB dA max err:", np.max(np.abs(dA.data - num_dA)))
    print("SUB dB max err:", np.max(np.abs(dB.data - num_dB)))
    assert np.allclose(dA.data, num_dA, rtol=1e-5, atol=1e-7)
    assert np.allclose(dB.data, num_dB, rtol=1e-5, atol=1e-7)


def check_hadamard():
    n, m = 4, 5
    np.random.seed(3)
    A = Vbuf(np.random.randn(n, m))
    B = Vbuf(np.random.randn(n, m))
    G = Vbuf(np.random.randn(n, m))

    def L():
        out = Ops.hadamard_forward(A, B)
        return float(np.sum(G.data * out.data))

    dA, dB = Ops.hadamard_backward(A, B, G)
    num_dA = numerical_grad(L, A)
    num_dB = numerical_grad(L, B)

    print("HADAMARD dA max err:", np.max(np.abs(dA.data - num_dA)))
    print("HADAMARD dB max err:", np.max(np.abs(dB.data - num_dB)))
    assert np.allclose(dA.data, num_dA, rtol=1e-5, atol=1e-7)
    assert np.allclose(dB.data, num_dB, rtol=1e-5, atol=1e-7)


def check_mean():
    n, m = 4, 5
    np.random.seed(4)
    A = Vbuf(np.random.randn(n, m))
    # Mean outputs a (1, 1) scalar, so upstream grad G is just a scalar weight
    G = Vbuf(np.random.randn(1, 1))

    def L():
        out = Ops.mean_forward(A)
        return float(np.sum(G.data * out.data))

    dA = Ops.mean_backward(A, G)
    num_dA = numerical_grad(L, A)

    print("MEAN dA max err:", np.max(np.abs(dA.data - num_dA)))
    assert np.allclose(dA.data, num_dA, rtol=1e-5, atol=1e-7)


def check_tanh():
    n, m = 4, 5
    np.random.seed(5)
    Z = Vbuf(np.random.randn(n, m))
    G = Vbuf(np.random.randn(n, m))

    def L():
        out = Ops.tanh_forward(Z)
        return float(np.sum(G.data * out.data))

    out = Ops.tanh_forward(Z)
    dZ = Ops.tanh_backward(out, G)
    
    num_dZ = numerical_grad(L, Z)

    print("TANH dZ max err:", np.max(np.abs(dZ.data - num_dZ)))
    assert np.allclose(dZ.data, num_dZ, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    check_matmul()
    check_add()
    check_sub()
    check_hadamard()
    check_mean()
    check_tanh()