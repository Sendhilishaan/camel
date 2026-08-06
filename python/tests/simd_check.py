from camel import array as ca
from camel.ops import Ops, Vbuf
from tests import grad_check, tensor_check


def check_naive_simd_parity():
    """same inputs run through both backends; forward+backward outputs should match closely."""
    ca.random.seed(42)
    n, k, m = 5, 4, 3
    A = Vbuf(ca.random.randn(n, k))
    B = Vbuf(ca.random.randn(k, m))
    G = Vbuf(ca.random.randn(n, m))

    Ops.set_backend("naive")
    naive_out = Ops.matmul_forward(A, B)
    naive_dA, naive_dB = Ops.matmul_backward(A, B, G)

    Ops.set_backend("simd")
    simd_out = Ops.matmul_forward(A, B)
    simd_dA, simd_dB = Ops.matmul_backward(A, B, G)

    print("NAIVE vs SIMD matmul out max err:", ca.max(ca.abs(naive_out.data - simd_out.data)))
    assert ca.allclose(naive_out.data, simd_out.data, rtol=1e-9, atol=1e-9)
    assert ca.allclose(naive_dA.data, simd_dA.data, rtol=1e-9, atol=1e-9)
    assert ca.allclose(naive_dB.data, simd_dB.data, rtol=1e-9, atol=1e-9)


def main():
    if not Ops.simd_available():
        print("SIMD backend not available on this build (Apple platforms only) - skipping.")
        return

    original_backend = Ops.backend
    try:
        check_naive_simd_parity()

        # re-run every existing grad/tensor check with the SIMD backend active.
        # Ops.backend is a single global switch, so these transparently exercise
        # the simd kernels through the same numerical-gradient verification
        # already used for the naive path - no separate simd-specific logic needed.
        Ops.set_backend("simd")
        grad_check.check_matmul()
        grad_check.check_add()
        grad_check.check_sub()
        grad_check.check_hadamard()
        grad_check.check_mean()
        grad_check.check_tanh()
        grad_check.check_relu()
        grad_check.check_softmax_xent()

        tensor_check.check_matmul_scalar()
        tensor_check.check_matmul_chain()
        tensor_check.check_mlp()
        tensor_check.check_softmax_xent()
    finally:
        Ops.set_backend(original_backend)

    print("all SIMD checks passed")


if __name__ == "__main__":
    main()
