from camel import array as ca
from camel.ops import Vbuf
from camel.tensor import Tensor
from tests.grad_check import numerical_grad


def check_matmul_scalar(rtol=1e-5, atol=1e-7, eps=1e-5):
    """single matmul that already outputs a (1,1) scalar, so it can be the loss."""
    ca.random.seed(0)
    k = 4
    A = Tensor(Vbuf(ca.random.randn(1, k)))
    B = Tensor(Vbuf(ca.random.randn(k, 1)))

    def L():
        return (A @ B).buf.data.item()

    # analytic grads via the engine
    (A @ B).backward()

    num_dA = numerical_grad(L, A.buf, eps)
    num_dB = numerical_grad(L, B.buf, eps)

    print("TENSOR matmul dA max err:", ca.max(ca.abs(A.grad.data - num_dA)))
    print("TENSOR matmul dB max err:", ca.max(ca.abs(B.grad.data - num_dB)))
    assert ca.allclose(A.grad.data, num_dA, rtol=rtol, atol=atol)
    assert ca.allclose(B.grad.data, num_dB, rtol=rtol, atol=atol)


def check_matmul_chain(rtol=1e-5, atol=1e-7, eps=1e-5):
    """two chained matmuls -> (1,1). Proves reverse-topo ordering + chaining."""
    ca.random.seed(1)
    k, h = 4, 3
    X = Tensor(Vbuf(ca.random.randn(1, k)))
    W1 = Tensor(Vbuf(ca.random.randn(k, h)))
    W2 = Tensor(Vbuf(ca.random.randn(h, 1)))

    def L():
        return (X @ W1 @ W2).buf.data.item()

    (X @ W1 @ W2).backward()

    num_dX = numerical_grad(L, X.buf, eps)
    num_dW1 = numerical_grad(L, W1.buf, eps)
    num_dW2 = numerical_grad(L, W2.buf, eps)

    print("TENSOR chain dX  max err:", ca.max(ca.abs(X.grad.data - num_dX)))
    print("TENSOR chain dW1 max err:", ca.max(ca.abs(W1.grad.data - num_dW1)))
    print("TENSOR chain dW2 max err:", ca.max(ca.abs(W2.grad.data - num_dW2)))
    assert ca.allclose(X.grad.data, num_dX, rtol=rtol, atol=atol)
    assert ca.allclose(W1.grad.data, num_dW1, rtol=rtol, atol=atol)
    assert ca.allclose(W2.grad.data, num_dW2, rtol=rtol, atol=atol)


def check_mlp(rtol=1e-5, atol=1e-7, eps=1e-5):
    """full graph: loss = mean((tanh(X@W + b) - Y)^2)
    exercises matmul, add(bias), tanh, sub, mul(hadamard), mean and the
    += accumulation path, since D feeds `D * D` twice."""
    ca.random.seed(7)
    n, k, m = 4, 3, 2
    X = Tensor(Vbuf(ca.random.randn(n, k)), requires_grad=False)
    W = Tensor(Vbuf(ca.random.randn(k, m)))
    b = Tensor(Vbuf(ca.random.randn(1, m)))
    Y = Tensor(Vbuf(ca.random.randn(n, m)), requires_grad=False)

    def L():
        H = (X @ W + b).tanh()
        D = H - Y
        return (D * D).mean().buf.data.item()

    # analytic grads via the engine
    H = (X @ W + b).tanh()
    D = H - Y
    (D * D).mean().backward()

    num_dW = numerical_grad(L, W.buf, eps)
    num_db = numerical_grad(L, b.buf, eps)

    print("MLP dW max err:", ca.max(ca.abs(W.grad.data - num_dW)))
    print("MLP db max err:", ca.max(ca.abs(b.grad.data - num_db)))
    assert ca.allclose(W.grad.data, num_dW, rtol=rtol, atol=atol)
    assert ca.allclose(b.grad.data, num_db, rtol=rtol, atol=atol)


def check_softmax_xent(rtol=1e-5, atol=1e-7, eps=1e-5):
    """full classification graph: logits = X@W + b ; loss = softmax_xent(logits, Y).
    closes the loop kernel -> Ops -> Function -> loss.backward() through the
    (dZ, None) grad path (Y is a non-differentiable target)."""
    ca.random.seed(11)
    n, k, C = 5, 4, 3
    X = Tensor(Vbuf(ca.random.randn(n, k)), requires_grad=False)
    W = Tensor(Vbuf(ca.random.randn(k, C)))
    b = Tensor(Vbuf(ca.random.randn(1, C)))

    labels = ca.random.randint(0, C, size=n)
    Y_data = ca.zeros((n, C))
    Y_data[range(n), labels] = 1.0
    Y = Tensor(Vbuf(Y_data), requires_grad=False)  # target: no grad

    def L():
        return (X @ W + b).softmax_xent(Y).buf.data.item()

    (X @ W + b).softmax_xent(Y).backward()

    num_dW = numerical_grad(L, W.buf, eps)
    num_db = numerical_grad(L, b.buf, eps)

    print("XENT dW max err:", ca.max(ca.abs(W.grad.data - num_dW)))
    print("XENT db max err:", ca.max(ca.abs(b.grad.data - num_db)))
    assert ca.allclose(W.grad.data, num_dW, rtol=rtol, atol=atol)
    assert ca.allclose(b.grad.data, num_db, rtol=rtol, atol=atol)


if __name__ == "__main__":
    check_matmul_scalar()
    check_matmul_chain()
    check_mlp()
    check_softmax_xent()
    print("all tensor grad checks passed")
