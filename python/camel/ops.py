from __future__ import annotations
import os
from ._c import c, SIMD_AVAILABLE
from camel.array import CamelArray
from typing import Tuple

class Vbuf:
    def __init__(self, data: CamelArray):
        # CamelArray is always float64 and contiguous, so no coercion needed here
        self.data = data
        self.ptr = self.data.ptr
        self.shape = self.data.shape

    @staticmethod # factory
    def zeros(n: int, m: int) -> Vbuf:
        return Vbuf(CamelArray.zeros((n, m)))

    @staticmethod
    def shape_eq(A: Vbuf, B: Vbuf) -> bool:
        return A.shape[0] == B.shape[0] and A.shape[1] == B.shape[1]


def _resolve_backend(name: str) -> str:
    if name not in ("naive", "simd"):
        raise ValueError(f"unknown camel backend {name!r}, expected 'naive' or 'simd'")
    if name == "simd" and not SIMD_AVAILABLE:
        raise RuntimeError(
            "SIMD backend not available: camel.dll wasn't built with prim_simd.c "
            "(Apple-only; run `make dll` on macOS to enable it)"
        )
    return name


# defaults to simd where built, naive elsewhere; override with CAMEL_BACKEND=naive|simd
_DEFAULT_BACKEND = os.environ.get("CAMEL_BACKEND", "simd" if SIMD_AVAILABLE else "naive")


class Ops:
    backend = _resolve_backend(_DEFAULT_BACKEND)

    @staticmethod
    def set_backend(name: str) -> None:
        Ops.backend = _resolve_backend(name)

    @staticmethod
    def simd_available() -> bool:
        return SIMD_AVAILABLE

    @staticmethod
    def _kernel(name: str):
        # picks the naive or _simd C function for `name`, per the active backend
        return getattr(c, f"{name}_simd") if Ops.backend == "simd" else getattr(c, name)

    # wrapping c functions
    @staticmethod
    def matmul_forward(A: Vbuf, B: Vbuf) -> Vbuf:
        result_buf = Vbuf.zeros(A.shape[0], B.shape[1])

        Ops._kernel("matmul_forward")(A.ptr, B.ptr, result_buf.ptr, A.shape[0], A.shape[1], B.shape[1])

        return result_buf

    @staticmethod
    def matmul_backward(A: Vbuf, B: Vbuf, grad_out: Vbuf) -> tuple[Vbuf, Vbuf]:
        dA_buf = Vbuf.zeros(A.shape[0], A.shape[1])
        dB_buf = Vbuf.zeros(A.shape[1], B.shape[1])

        Ops._kernel("matmul_backward")(A.ptr, B.ptr, grad_out.ptr, dA_buf.ptr, dB_buf.ptr, A.shape[0], A.shape[1], B.shape[1])

        return (dA_buf, dB_buf)

    @staticmethod
    def add_forward(A: Vbuf, B:Vbuf) -> Vbuf:
        # matadd broadcast forward
        out = Vbuf(A.data.copy()) # add kernel is inplace (fix?)

        Ops._kernel("matadd_broadcast_forward")(out.ptr, B.ptr, out.shape[0], out.shape[1])

        return out

    @staticmethod
    def add_backward(grad_out: Vbuf) -> tuple[Vbuf, Vbuf]:
        n = grad_out.shape[0]
        m = grad_out.shape[1]

        dX_buf = Vbuf.zeros(n, m)
        dB_buf = Vbuf.zeros(1, m)

        Ops._kernel("matadd_broadcast_backward")(grad_out.ptr, dX_buf.ptr, dB_buf.ptr, n, m)

        return (dX_buf, dB_buf)

    @staticmethod
    def sub_forward(A: Vbuf, B: Vbuf) -> Vbuf:
        n = A.shape[0]
        m = B.shape[1]

        out = Vbuf.zeros(n, m)

        Ops._kernel("matsub_forward")(A.ptr, B.ptr, out.ptr, n, m)

        return out

    @staticmethod
    def sub_backward(grad_out: Vbuf) -> tuple[Vbuf, Vbuf]:
        n = grad_out.shape[0]
        m = grad_out.shape[1]

        dA_buf = Vbuf.zeros(n, m)
        dB_buf = Vbuf.zeros(n, m)

        Ops._kernel("matsub_backward")(grad_out.ptr, dA_buf.ptr, dB_buf.ptr, n, m)

        return (dA_buf, dB_buf)

    @staticmethod
    def hadamard_forward(A: Vbuf, B: Vbuf) -> Vbuf:
        n = A.shape[0]
        m = B.shape[1]

        out = Vbuf.zeros(n, m)

        Ops._kernel("hadamard_forward")(A.ptr, B.ptr, out.ptr, n, m)

        return out

    @staticmethod
    def hadamard_backward(A: Vbuf, B: Vbuf, grad_out: Vbuf) -> tuple[Vbuf, Vbuf]:
        n = A.shape[0]
        m = B.shape[1]

        dA_buf = Vbuf.zeros(n, m)
        dB_buf = Vbuf.zeros(n, m)

        Ops._kernel("hadamard_backward")(grad_out.ptr, A.ptr, B.ptr, dA_buf.ptr, dB_buf.ptr, n, m)

        return (dA_buf, dB_buf)

    @staticmethod
    def mean_forward(A: Vbuf) -> Vbuf:
        n = A.shape[0]
        m = A.shape[1]

        out = Vbuf.zeros(1, 1) # scalar

        Ops._kernel("matmean_forward")(A.ptr, out.ptr, n * m)

        return out

    @staticmethod # A is input buf
    def mean_backward(A: Vbuf, grad_out: Vbuf) -> Vbuf:
        n_total = A.shape[0] * A.shape[1]

        dX_buf = Vbuf.zeros(A.shape[0], A.shape[1])
        g = grad_out.data.item()

        Ops._kernel("matmean_backward")(dX_buf.ptr, n_total, g)

        return dX_buf

    @staticmethod
    def tanh_forward(Z: Vbuf) -> Vbuf:
        n = Z.shape[0]
        m = Z.shape[1]
        out = Vbuf.zeros(n, m)

        Ops._kernel("tanh_forward")(Z.ptr, out.ptr, n, m)

        return out

    @staticmethod
    def tanh_backward(out: Vbuf, grad_out: Vbuf) -> Vbuf:
        if not Vbuf.shape_eq(out, grad_out):
            raise ValueError(f"tanh_backward shape mismatch: out={out.shape}, grad_out={grad_out.shape}")

        n = out.shape[0]
        m = out.shape[1]

        dZ_buf = Vbuf.zeros(n, m)

        Ops._kernel("tanh_backward")(out.ptr, grad_out.ptr, dZ_buf.ptr, n, m)

        return dZ_buf

    @staticmethod
    def relu_forward(Z: Vbuf) -> Vbuf:
        n = Z.shape[0]
        m = Z.shape[1]
        out = Vbuf.zeros(n, m)

        Ops._kernel("relu_forward")(Z.ptr, out.ptr, n, m)

        return out

    @staticmethod
    def relu_backward(out: Vbuf, grad_out: Vbuf) -> Vbuf:
        if not Vbuf.shape_eq(out, grad_out):
            raise ValueError(f"relu_backward shape mismatch: out={out.shape}, grad_out={grad_out.shape}")

        n = out.shape[0]
        m = out.shape[1]

        dZ_buf = Vbuf.zeros(n, m)

        Ops._kernel("relu_backward")(out.ptr, grad_out.ptr, dZ_buf.ptr, n, m)

        return dZ_buf

    @staticmethod # fused softmax + cross-entropy; returns (probs cached for backward, scalar loss)
    def softmax_xent_forward(Z: Vbuf, Y: Vbuf) -> tuple[Vbuf, Vbuf]:
        if not Vbuf.shape_eq(Z, Y):
            raise ValueError(f"softmax_xent shape mismatch: Z={Z.shape}, Y={Y.shape}")

        n = Z.shape[0]
        m = Z.shape[1]

        probs = Vbuf.zeros(n, m)
        loss = Vbuf.zeros(1, 1) # scalar

        Ops._kernel("softmax_xent_forward")(Z.ptr, Y.ptr, probs.ptr, loss.ptr, n, m)

        return (probs, loss)

    @staticmethod # probs is the cached softmax from forward
    def softmax_xent_backward(probs: Vbuf, Y: Vbuf, grad_out: Vbuf) -> Vbuf:
        n = probs.shape[0]
        m = probs.shape[1]

        dZ_buf = Vbuf.zeros(n, m)
        g = grad_out.data.item()

        Ops._kernel("softmax_xent_backward")(probs.ptr, Y.ptr, dZ_buf.ptr, g, n, m)

        return dZ_buf
