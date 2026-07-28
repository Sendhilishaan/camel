import numpy as np
from camel.ops import Vbuf
from camel.tensor import Tensor
from tests.grad_check import numerical_grad


def check_matmul_scalar():
    """single matmul that already outputs a (1,1) scalar, so it can be the loss."""
    np.random.seed(0)
    k = 4
    A = Tensor(Vbuf(np.random.randn(1, k)))
    B = Tensor(Vbuf(np.random.randn(k, 1)))

    def L():
        return (A @ B).buf.data.item()

    # analytic grads via the engine
    (A @ B).backward()

    num_dA = numerical_grad(L, A.buf)
    num_dB = numerical_grad(L, B.buf)

    print("TENSOR matmul dA max err:", np.max(np.abs(A.grad.data - num_dA)))
    print("TENSOR matmul dB max err:", np.max(np.abs(B.grad.data - num_dB)))
    assert np.allclose(A.grad.data, num_dA, rtol=1e-5, atol=1e-7)
    assert np.allclose(B.grad.data, num_dB, rtol=1e-5, atol=1e-7)


def check_matmul_chain():
    """two chained matmuls -> (1,1). Proves reverse-topo ordering + chaining."""
    np.random.seed(1)
    k, h = 4, 3
    X = Tensor(Vbuf(np.random.randn(1, k)))
    W1 = Tensor(Vbuf(np.random.randn(k, h)))
    W2 = Tensor(Vbuf(np.random.randn(h, 1)))

    def L():
        return (X @ W1 @ W2).buf.data.item()

    (X @ W1 @ W2).backward()

    num_dX = numerical_grad(L, X.buf)
    num_dW1 = numerical_grad(L, W1.buf)
    num_dW2 = numerical_grad(L, W2.buf)

    print("TENSOR chain dX  max err:", np.max(np.abs(X.grad.data - num_dX)))
    print("TENSOR chain dW1 max err:", np.max(np.abs(W1.grad.data - num_dW1)))
    print("TENSOR chain dW2 max err:", np.max(np.abs(W2.grad.data - num_dW2)))
    assert np.allclose(X.grad.data, num_dX, rtol=1e-5, atol=1e-7)
    assert np.allclose(W1.grad.data, num_dW1, rtol=1e-5, atol=1e-7)
    assert np.allclose(W2.grad.data, num_dW2, rtol=1e-5, atol=1e-7)


def check_mlp():
    """full graph: loss = mean((tanh(X@W + b) - Y)^2)
    exercises matmul, add(bias), tanh, sub, mul(hadamard), mean and the
    += accumulation path, since D feeds `D * D` twice."""
    np.random.seed(7)
    n, k, m = 4, 3, 2
    X = Tensor(Vbuf(np.random.randn(n, k)), requires_grad=False)
    W = Tensor(Vbuf(np.random.randn(k, m)))
    b = Tensor(Vbuf(np.random.randn(1, m)))
    Y = Tensor(Vbuf(np.random.randn(n, m)), requires_grad=False)

    def L():
        H = (X @ W + b).tanh()
        D = H - Y
        return (D * D).mean().buf.data.item()

    # analytic grads via the engine
    H = (X @ W + b).tanh()
    D = H - Y
    (D * D).mean().backward()

    num_dW = numerical_grad(L, W.buf)
    num_db = numerical_grad(L, b.buf)

    print("MLP dW max err:", np.max(np.abs(W.grad.data - num_dW)))
    print("MLP db max err:", np.max(np.abs(b.grad.data - num_db)))
    assert np.allclose(W.grad.data, num_dW, rtol=1e-5, atol=1e-7)
    assert np.allclose(b.grad.data, num_db, rtol=1e-5, atol=1e-7)


def check_softmax_xent():
    """full classification graph: logits = X@W + b ; loss = softmax_xent(logits, Y).
    closes the loop kernel -> Ops -> Function -> loss.backward() through the
    (dZ, None) grad path (Y is a non-differentiable target)."""
    np.random.seed(11)
    n, k, C = 5, 4, 3
    X = Tensor(Vbuf(np.random.randn(n, k)), requires_grad=False)
    W = Tensor(Vbuf(np.random.randn(k, C)))
    b = Tensor(Vbuf(np.random.randn(1, C)))

    labels = np.random.randint(0, C, size=n)
    Y_data = np.zeros((n, C))
    Y_data[np.arange(n), labels] = 1.0
    Y = Tensor(Vbuf(Y_data), requires_grad=False)  # target: no grad

    def L():
        return (X @ W + b).softmax_xent(Y).buf.data.item()

    (X @ W + b).softmax_xent(Y).backward()

    num_dW = numerical_grad(L, W.buf)
    num_db = numerical_grad(L, b.buf)

    print("XENT dW max err:", np.max(np.abs(W.grad.data - num_dW)))
    print("XENT db max err:", np.max(np.abs(b.grad.data - num_db)))
    assert np.allclose(W.grad.data, num_dW, rtol=1e-5, atol=1e-7)
    assert np.allclose(b.grad.data, num_db, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    check_matmul_scalar()
    check_matmul_chain()
    check_mlp()
    check_softmax_xent()
    print("all tensor grad checks passed")
