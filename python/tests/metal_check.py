from camel import array as ca
from camel.ops import Ops, Vbuf
from tests import grad_check, tensor_check

# MSL has no double, so every metal kernel round-trips through float32 (~7
# significant digits). On an M2, observed errors land around 1e-4 to 1e-7;
# these tolerances leave headroom above that without being so loose a real
# bug would slip through.
RTOL, ATOL, EPS = 3e-2, 3e-2, 1e-3
LOSS_RTOL, LOSS_ATOL = 1e-3, 1e-4


def check_naive_metal_parity():
    """same inputs run through both backends; forward+backward outputs should be close."""
    ca.random.seed(42)
    n, k, m = 5, 4, 3
    A = Vbuf(ca.random.randn(n, k))
    B = Vbuf(ca.random.randn(k, m))
    G = Vbuf(ca.random.randn(n, m))

    Ops.set_backend("naive")
    naive_out = Ops.matmul_forward(A, B)
    naive_dA, naive_dB = Ops.matmul_backward(A, B, G)

    Ops.set_backend("metal")
    metal_out = Ops.matmul_forward(A, B)
    metal_dA, metal_dB = Ops.matmul_backward(A, B, G)

    print("NAIVE vs METAL matmul out max err:", ca.max(ca.abs(naive_out.data - metal_out.data)))
    assert ca.allclose(naive_out.data, metal_out.data, rtol=RTOL, atol=ATOL)
    assert ca.allclose(naive_dA.data, metal_dA.data, rtol=RTOL, atol=ATOL)
    assert ca.allclose(naive_dB.data, metal_dB.data, rtol=RTOL, atol=ATOL)


def main():
    if not Ops.metal_device_available():
        print("no Metal device available (or camel.dll wasn't built with prim_metal.m) - skipping.")
        return

    original_backend = Ops.backend
    try:
        check_naive_metal_parity()

        Ops.set_backend("metal")
        grad_check.check_matmul(RTOL, ATOL, EPS)
        grad_check.check_add(RTOL, ATOL, EPS)
        grad_check.check_sub(RTOL, ATOL, EPS)
        grad_check.check_hadamard(RTOL, ATOL, EPS)
        grad_check.check_mean(RTOL, ATOL, EPS)
        grad_check.check_tanh(RTOL, ATOL, EPS)
        grad_check.check_relu(RTOL, ATOL, EPS)
        grad_check.check_softmax_xent(RTOL, ATOL, EPS, LOSS_RTOL, LOSS_ATOL)

        tensor_check.check_matmul_scalar(RTOL, ATOL, EPS)
        tensor_check.check_matmul_chain(RTOL, ATOL, EPS)
        tensor_check.check_mlp(RTOL, ATOL, EPS)
        tensor_check.check_softmax_xent(RTOL, ATOL, EPS)
    finally:
        Ops.set_backend(original_backend)

    print("all Metal checks passed")


if __name__ == "__main__":
    main()
