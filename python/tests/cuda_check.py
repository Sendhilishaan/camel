import gc
import math
from ctypes import c_double, c_float, c_void_p, byref

from camel import array as ca
from camel._c import c
from camel.ops import Ops, Vbuf
from camel.tensor import Tensor
from camel.optim import SGD, AdaGrad, RMSprop, Adam
from tests import grad_check, tensor_check

# Compare float32 kernels against float64 C, then independently differentiate
# the full graphs. Finite differences need a larger epsilon than float64.
RTOL, ATOL = 2e-5, 2e-5
GRAD_RTOL, GRAD_ATOL, EPS = 2e-3, 5e-4, 1e-3


def close(a, b, rtol=RTOL, atol=ATOL):
    assert ca.allclose(a, b, rtol=rtol, atol=atol), (a, b)


def primitives(backend, n, k, m):
    Ops.set_backend(backend)
    ca.random.seed(42)
    a, b = Vbuf(ca.random.randn(n, k)), Vbuf(ca.random.randn(k, m))
    x, y, g = [Vbuf(ca.random.randn(n, m)) for _ in range(3)]
    bias = Vbuf(ca.random.randn(1, m))
    scalar = Vbuf(ca.CamelArray([[0.7]]))
    labels = ca.zeros((n, m))
    labels[range(n), [i % m for i in range(n)]] = 1.0
    labels = Vbuf(labels)
    tanh, relu = Ops.tanh_forward(x), Ops.relu_forward(x)
    probs, loss = Ops.softmax_xent_forward(x, labels)
    outputs = [
        Ops.matmul_forward(a, b), *Ops.matmul_backward(a, b, g),
        Ops.add_forward(x, bias), *Ops.add_backward(g),
        Ops.sub_forward(x, y), *Ops.sub_backward(g),
        Ops.hadamard_forward(x, y), *Ops.hadamard_backward(x, y, g),
        Ops.mean_forward(x), Ops.mean_backward(x, scalar),
        tanh, Ops.tanh_backward(tanh, g), relu, Ops.relu_backward(relu, g),
        probs, Ops.softmax_xent_backward(probs, labels, scalar),
    ]
    if backend == "cuda":
        assert all(v._data is None for v in outputs), "an intermediate left the GPU"
        close(Ops.accumulate(x, y).data, x.data + y.data)
    return [out.data.copy() for out in outputs] + [loss.data.copy()]


def check_parity():
    for n, k, m in ((1, 1, 1), (5, 7, 3), (17, 19, 23), (33, 31, 257)):
        naive, cuda = primitives("naive", n, k, m), primitives("cuda", n, k, m)
        for a, b in zip(naive, cuda):
            close(a, b)
    # Reduction spans many strides; softmax is stable even at extreme logits.
    x = ca.CamelArray([math.sin(i) for i in range(4099)]).reshape(1, -1)
    close(Ops.mean_forward(Vbuf(x)).data, ca.CamelArray([[x.mean()]]))
    x = Vbuf(ca.CamelArray([[10000, 0, -10000], [-10000, 10000, 0]]))
    y = Vbuf(ca.CamelArray([[0, 0, 1], [0, 1, 0]]))
    probs, loss = Ops.softmax_xent_forward(x, y)
    close(probs.data, ca.CamelArray([[1, 0, 0], [0, 1, 0]]))
    assert abs(loss.data.item() - 10000) < 1e-3
    print("all resident primitive outputs/gradients match naive")


def check_staging():
    # Exercise the public float* ABI too; Python training uses resident calls.
    ca.random.seed(9)
    n, k, m = 5, 7, 3
    sizes = {"a": n*k, "b": k*m, "x": n*m, "y": n*m, "g": n*m,
             "bias": m, "out": n*m, "da": n*k, "db": k*m,
             "dx": n*m, "dy": n*m, "dbias": m, "loss": 1, "probs": n*m}
    inputs = {name: ca.random.randn(size).tolist() for name, size in sizes.items()}
    inputs["y"] = [float(j == i % m) for i in range(n) for j in range(m)]
    cases = [
        ("matmul_forward", ["a", "b", "out", n, k, m], ["out"]),
        ("matmul_backward", ["a", "b", "g", "da", "db", n, k, m], ["da", "db"]),
        ("matadd_broadcast_forward", ["x", "bias", n, m], ["x"]),
        ("matadd_broadcast_backward", ["g", "dx", "dbias", n, m], ["dx", "dbias"]),
        ("matsub_forward", ["x", "y", "out", n, m], ["out"]),
        ("matsub_backward", ["g", "dx", "dy", n, m], ["dx", "dy"]),
        ("hadamard_forward", ["x", "y", "out", n, m], ["out"]),
        ("hadamard_backward", ["g", "x", "y", "dx", "dy", n, m], ["dx", "dy"]),
        ("matmean_forward", ["x", "loss", n*m], ["loss"]),
        ("matmean_backward", ["dx", n*m, 0.7], ["dx"]),
        ("tanh_forward", ["x", "out", n, m], ["out"]),
        ("tanh_backward", ["x", "g", "dx", n, m], ["dx"]),
        ("relu_forward", ["x", "out", n, m], ["out"]),
        ("relu_backward", ["x", "g", "dx", n, m], ["dx"]),
        ("softmax_xent_forward", ["x", "y", "probs", "loss", n, m], ["probs", "loss"]),
        ("softmax_xent_backward", ["probs", "y", "dx", 0.7, n, m], ["dx"]),
    ]
    for name, args, outputs in cases:
        results = []
        for suffix, dtype in (("", c_double), ("_cuda", c_float)):
            buffers = {key: (dtype * size)(*inputs[key]) for key, size in sizes.items()}
            for key in outputs:
                if name != "matadd_broadcast_forward":
                    buffers[key] = (dtype * sizes[key])()
            getattr(c, name + suffix)(*[buffers[arg] if isinstance(arg, str) else arg for arg in args])
            results.append([ca.CamelArray(list(buffers[key])) for key in outputs])
        for a, b in zip(*results):
            close(a, b)
    print("all 16 staging entry points match naive")


def check_cache():
    Ops.set_backend("cuda")
    data = ca.CamelArray([[1, 2], [3, 4]])
    a = Vbuf(data)
    handle = a.gpu_handle()
    assert a.gpu_handle() == handle
    data[0, 0] = 10
    close(Ops.relu_forward(a).data, data)
    data.reshape(1, 4)[0, 1] = 20 # a view must invalidate the cached upload
    close(Ops.relu_forward(a).data, data)
    a.data += 1
    close(Ops.relu_forward(a).data, a.data)
    result = Ops.hadamard_forward(a, a)
    assert result._data is None
    Ops.set_backend("naive")
    expected = a.data * a.data
    close(result.data, expected) # materializes through its CUDA owner
    close(Ops.sub_forward(result, a).data, expected - a.data)
    result.data[0, 0] = -3
    Ops.set_backend("cuda")
    close(Ops.relu_forward(result).data, ca.CamelArray([[0, 441], [16, 25]]))
    Ops.set_backend("naive")
    del result, a
    gc.collect() # CUDA allocations must be freed through CUDA after switching
    Ops.set_backend("cuda")
    print("cache invalidation, reshape views, backend switches: OK")


def train(backend, optimiser):
    Ops.set_backend(backend)
    x = Tensor(Vbuf(ca.CamelArray([[-1, 0.5], [0, 1], [1, -0.5], [2, 1]])), requires_grad=False)
    y = Tensor(Vbuf(ca.CamelArray([[-0.5], [0.25], [0.5], [0.8]])), requires_grad=False)
    w = Tensor(Vbuf(ca.CamelArray([[0.2], [-0.1]])))
    b = Tensor(Vbuf(ca.CamelArray([[0.05]])))
    opt = optimiser([w, b], lr=0.03)
    states = [v for attr in ("velocity", "grad_sum", "ema_sq", "exp_avg", "exp_avg_sq")
              for v in getattr(opt, attr, [])]
    handles = None
    first = None
    for _ in range(40):
        d = (x @ w + b).tanh() - y
        loss = (d * d).mean()
        if first is None:
            first = loss.buf.data.item()
        loss.backward()
        opt.step()
        if backend == "cuda":
            assert all(v._data is None for v in [w.buf, b.buf] + states)
            current = [v.gpu_handle() for v in [w.buf, b.buf] + states]
            if handles is not None:
                assert current == handles, "optimizer reallocated its persistent buffers"
            handles = current
        opt.zero_grad()
    last = loss.buf.data.item()
    assert last < first * 0.4, (first, last)
    return [v.data.copy() for v in [w.buf, b.buf] + states], last


def check_optimisers():
    for optimiser in (lambda ps, lr: SGD(ps, lr, momentum=0.9), AdaGrad, RMSprop, Adam):
        naive, cpu_loss = train("naive", optimiser)
        cuda, gpu_loss = train("cuda", optimiser)
        for a, b in zip(naive, cuda):
            close(a, b, rtol=3e-4, atol=3e-5)
        assert abs(cpu_loss - gpu_loss) < 1e-5
    print("SGD momentum, AdaGrad, RMSprop, Adam training and persistent state: OK")


def check_errors():
    Ops.set_backend("cuda")
    for n in (0, -1):
        try:
            c.camel_cuda_buffer_create(None, n)
            raise AssertionError("invalid allocation succeeded")
        except RuntimeError as exc:
            assert "dimensions" in str(exc)
    a = Vbuf(ca.CamelArray([[1]]))
    h = a.gpu_handle()
    for call in (
        lambda: c.relu_forward_cuda_resident(None, 1, 1),
        lambda: c.relu_forward_cuda_resident(h, 2, 1),
        lambda: c.relu_forward_cuda_resident(h, 2147483647, 2),
    ):
        try:
            call()
            raise AssertionError("invalid dispatch succeeded")
        except RuntimeError:
            pass
    da, db = c_void_p(), c_void_p()
    try:
        c.matmul_backward_cuda_resident(h, h, h, 2, 2, 2, byref(da), byref(db))
        raise AssertionError("invalid backward succeeded")
    except RuntimeError:
        assert not da.value and not db.value
    for _ in range(200):
        close(Ops.relu_forward(a).data, a.data)
    Ops.synchronize()
    print("invalid dimensions/handles raise recoverable errors: OK")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-device", action="store_true")
    args = parser.parse_args()
    if not Ops.cuda_device_available():
        if args.require_device:
            raise RuntimeError("CUDA test requires a built backend and working NVIDIA GPU")
        print("no CUDA device/backend available - skipping")
        return
    original = Ops.backend
    try:
        check_parity()
        check_staging()
        check_cache()
        check_errors()
        Ops.set_backend("cuda")
        for name in ("matmul", "add", "sub", "hadamard", "mean", "tanh", "relu"):
            getattr(grad_check, "check_" + name)(GRAD_RTOL, GRAD_ATOL, EPS)
        grad_check.check_softmax_xent(GRAD_RTOL, GRAD_ATOL, EPS, RTOL, ATOL)
        for name in ("matmul_scalar", "matmul_chain", "mlp", "softmax_xent"):
            getattr(tensor_check, "check_" + name)(GRAD_RTOL, GRAD_ATOL, EPS)
        check_optimisers()
        Ops.synchronize()
    finally:
        Ops.set_backend(original)
    print("all CUDA checks passed")


if __name__ == "__main__":
    main()
