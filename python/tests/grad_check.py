from camel import array as ca
from camel.ops import Ops, Vbuf


def numerical_grad(f, vbuf: Vbuf, eps=1e-5):
    """Measure grad of scalar f() w.r.t. Vbuf's underlying data by central differences."""
    grad = ca.zeros_like(vbuf.data)
    x = vbuf.data
    for idx in ca.ndindex(x.shape):
        old = x[idx]

        x[idx] = old + eps
        Lp = f()

        x[idx] = old - eps
        Lm = f()

        x[idx] = old
        grad[idx] = (Lp - Lm) / (2 * eps)
    return grad


def check_matmul():
    n, k, m = 4, 3, 5
    ca.random.seed(0)
    A = Vbuf(ca.random.randn(n, k))
    B = Vbuf(ca.random.randn(k, m))
    G = Vbuf(ca.random.randn(n, m))

    def L():
        out = Ops.matmul_forward(A, B)
        return ca.sum(G.data * out.data)

    dA, dB = Ops.matmul_backward(A, B, G)
    num_dA = numerical_grad(L, A)
    num_dB = numerical_grad(L, B)

    print("MATMUL dA max err:", ca.max(ca.abs(dA.data - num_dA)))
    print("MATMUL dB max err:", ca.max(ca.abs(dB.data - num_dB)))
    assert ca.allclose(dA.data, num_dA, rtol=1e-5, atol=1e-7)
    assert ca.allclose(dB.data, num_dB, rtol=1e-5, atol=1e-7)


def check_add():
    n, m = 4, 5
    ca.random.seed(1)
    A = Vbuf(ca.random.randn(n, m))
    B = Vbuf(ca.random.randn(1, m))
    G = Vbuf(ca.random.randn(n, m))

    def L():
        out = Ops.add_forward(A, B)
        return ca.sum(G.data * out.data)

    dA, dB = Ops.add_backward(G)
    num_dA = numerical_grad(L, A)
    num_dB = numerical_grad(L, B)

    print("ADD dA max err:", ca.max(ca.abs(dA.data - num_dA)))
    print("ADD dB max err:", ca.max(ca.abs(dB.data - num_dB)))
    assert ca.allclose(dA.data, num_dA, rtol=1e-5, atol=1e-7)
    assert ca.allclose(dB.data, num_dB, rtol=1e-5, atol=1e-7)


def check_sub():
    n, m = 4, 5
    ca.random.seed(2)
    A = Vbuf(ca.random.randn(n, m))
    B = Vbuf(ca.random.randn(n, m))
    G = Vbuf(ca.random.randn(n, m))

    def L():
        out = Ops.sub_forward(A, B)
        return ca.sum(G.data * out.data)

    dA, dB = Ops.sub_backward(G)
    num_dA = numerical_grad(L, A)
    num_dB = numerical_grad(L, B)

    print("SUB dA max err:", ca.max(ca.abs(dA.data - num_dA)))
    print("SUB dB max err:", ca.max(ca.abs(dB.data - num_dB)))
    assert ca.allclose(dA.data, num_dA, rtol=1e-5, atol=1e-7)
    assert ca.allclose(dB.data, num_dB, rtol=1e-5, atol=1e-7)


def check_hadamard():
    n, m = 4, 5
    ca.random.seed(3)
    A = Vbuf(ca.random.randn(n, m))
    B = Vbuf(ca.random.randn(n, m))
    G = Vbuf(ca.random.randn(n, m))

    def L():
        out = Ops.hadamard_forward(A, B)
        return ca.sum(G.data * out.data)

    dA, dB = Ops.hadamard_backward(A, B, G)
    num_dA = numerical_grad(L, A)
    num_dB = numerical_grad(L, B)

    print("HADAMARD dA max err:", ca.max(ca.abs(dA.data - num_dA)))
    print("HADAMARD dB max err:", ca.max(ca.abs(dB.data - num_dB)))
    assert ca.allclose(dA.data, num_dA, rtol=1e-5, atol=1e-7)
    assert ca.allclose(dB.data, num_dB, rtol=1e-5, atol=1e-7)


def check_mean():
    n, m = 4, 5
    ca.random.seed(4)
    A = Vbuf(ca.random.randn(n, m))
    # Mean outputs a (1, 1) scalar, so upstream grad G is just a scalar weight
    G = Vbuf(ca.random.randn(1, 1))

    def L():
        out = Ops.mean_forward(A)
        return ca.sum(G.data * out.data)

    dA = Ops.mean_backward(A, G)
    num_dA = numerical_grad(L, A)

    print("MEAN dA max err:", ca.max(ca.abs(dA.data - num_dA)))
    assert ca.allclose(dA.data, num_dA, rtol=1e-5, atol=1e-7)


def check_tanh():
    n, m = 4, 5
    ca.random.seed(5)
    Z = Vbuf(ca.random.randn(n, m))
    G = Vbuf(ca.random.randn(n, m))

    def L():
        out = Ops.tanh_forward(Z)
        return ca.sum(G.data * out.data)

    out = Ops.tanh_forward(Z)
    dZ = Ops.tanh_backward(out, G)

    num_dZ = numerical_grad(L, Z)

    print("TANH dZ max err:", ca.max(ca.abs(dZ.data - num_dZ)))
    assert ca.allclose(dZ.data, num_dZ, rtol=1e-5, atol=1e-7)


def check_relu():
    n, m = 4, 5
    ca.random.seed(12)
    Z = Vbuf(ca.random.randn(n, m))
    G = Vbuf(ca.random.randn(n, m))

    def L():
        out = Ops.relu_forward(Z)
        return ca.sum(G.data * out.data)

    out = Ops.relu_forward(Z)
    dZ = Ops.relu_backward(out, G)

    num_dZ = numerical_grad(L, Z)

    print("RELU dZ max err:", ca.max(ca.abs(dZ.data - num_dZ)))
    assert ca.allclose(dZ.data, num_dZ, rtol=1e-5, atol=1e-7)


def check_softmax_xent():
    n, m = 4, 3
    ca.random.seed(6)
    Z = Vbuf(ca.random.randn(n, m))

    # one-hot targets: a random true class per row
    labels = ca.random.randint(0, m, size=n)
    Y_data = ca.zeros((n, m))
    Y_data[range(n), labels] = 1.0
    Y = Vbuf(Y_data)

    # the forward already outputs a (1,1) scalar loss, so it can be L directly
    def L():
        _, loss = Ops.softmax_xent_forward(Z, Y)
        return loss.data.item()

    # (1) forward value: independent log-sum-exp reference, computed by hand
    # rather than through the fused kernel. a consistent sign/shift bug in
    # fwd+bwd would still pass the grad check, so pin the loss value on its own.
    z = Z.data - Z.data.max(axis=1, keepdims=True)
    log_probs = z - ca.log(ca.exp(z).sum(axis=1, keepdims=True))
    ref_loss = (-(Y_data * log_probs).sum(axis=1)).mean()
    probs, loss = Ops.softmax_xent_forward(Z, Y)
    print("SOFTMAX_XENT loss err:", abs(loss.data.item() - ref_loss))
    assert ca.isclose(loss.data.item(), ref_loss, rtol=1e-9, atol=1e-11)

    # (2) backward vs central differences. upstream grad = 1 (loss is the root).
    G = Vbuf(ca.ones((1, 1)))
    dZ = Ops.softmax_xent_backward(probs, Y, G)
    num_dZ = numerical_grad(L, Z)

    print("SOFTMAX_XENT dZ max err:", ca.max(ca.abs(dZ.data - num_dZ)))
    assert ca.allclose(dZ.data, num_dZ, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    check_matmul()
    check_add()
    check_sub()
    check_hadamard()
    check_mean()
    check_tanh()
    check_relu()
    check_softmax_xent()
