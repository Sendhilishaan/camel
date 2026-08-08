from __future__ import annotations
import os
from ctypes import c_float
from ._c import c, SIMD_AVAILABLE, METAL_AVAILABLE
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

    def to_float32(self):
        # Metal has no double type, so GPU kernels need a float32 staging copy
        vals = self.data.tolist()
        return (c_float * len(vals))(*vals)

    @staticmethod
    def from_float32(buf, n: int, m: int) -> Vbuf:
        return Vbuf(CamelArray(list(buf)).reshape(n, m))


def _resolve_backend(name: str) -> str:
    if name not in ("naive", "simd", "metal"):
        raise ValueError(f"unknown camel backend {name!r}, expected 'naive', 'simd' or 'metal'")
    if name == "simd" and not SIMD_AVAILABLE:
        raise RuntimeError(
            "SIMD backend not available: camel.dll wasn't built with prim_simd.c "
            "(Apple-only; run `make dll` on macOS to enable it)"
        )
    if name == "metal" and not METAL_AVAILABLE:
        raise RuntimeError(
            "metal backend not available: camel.dll wasn't built with prim_metal.m "
            "(Apple-only; run `make dll` on macOS to enable it)"
        )
    return name


# defaults to simd where built, naive elsewhere; metal is opt-in (needs a real
# GPU device at runtime, so it isn't auto-selected even when the lib has it)
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
    def metal_available() -> bool:
        return METAL_AVAILABLE

    @staticmethod
    def metal_device_available() -> bool:
        # unlike metal_available() this actually probes for a working GPU;
        # a real kernel call aborts the process if none is found, this doesn't
        return METAL_AVAILABLE and bool(c.camel_metal_device_available())

    @staticmethod
    def _kernel(name: str):
        # picks the naive or _simd C function for `name` (metal has its own call convention, see below)
        return getattr(c, f"{name}_simd") if Ops.backend == "simd" else getattr(c, name)

    @staticmethod
    def matmul_forward(A: Vbuf, B: Vbuf) -> Vbuf:
        n, k, m = A.shape[0], A.shape[1], B.shape[1]

        if Ops.backend == "metal":
            out_buf = (c_float * (n * m))()
            c.matmul_forward_metal(A.to_float32(), B.to_float32(), out_buf, n, k, m)
            return Vbuf.from_float32(out_buf, n, m)

        result_buf = Vbuf.zeros(n, m)
        Ops._kernel("matmul_forward")(A.ptr, B.ptr, result_buf.ptr, n, k, m)
        return result_buf

    @staticmethod
    def matmul_backward(A: Vbuf, B: Vbuf, grad_out: Vbuf) -> tuple[Vbuf, Vbuf]:
        n, k, m = A.shape[0], A.shape[1], B.shape[1]

        if Ops.backend == "metal":
            da_buf = (c_float * (n * k))()
            db_buf = (c_float * (k * m))()
            c.matmul_backward_metal(A.to_float32(), B.to_float32(), grad_out.to_float32(), da_buf, db_buf, n, k, m)
            return (Vbuf.from_float32(da_buf, n, k), Vbuf.from_float32(db_buf, k, m))

        dA_buf = Vbuf.zeros(n, k)
        dB_buf = Vbuf.zeros(k, m)
        Ops._kernel("matmul_backward")(A.ptr, B.ptr, grad_out.ptr, dA_buf.ptr, dB_buf.ptr, n, k, m)
        return (dA_buf, dB_buf)

    @staticmethod
    def add_forward(A: Vbuf, B:Vbuf) -> Vbuf:
        n, m = A.shape[0], A.shape[1]

        if Ops.backend == "metal":
            out_buf = A.to_float32() # fresh staging copy, safe to mutate in place
            c.matadd_broadcast_forward_metal(out_buf, B.to_float32(), n, m)
            return Vbuf.from_float32(out_buf, n, m)

        out = Vbuf(A.data.copy()) # add kernel is inplace (fix?)
        Ops._kernel("matadd_broadcast_forward")(out.ptr, B.ptr, n, m)
        return out

    @staticmethod
    def add_backward(grad_out: Vbuf) -> tuple[Vbuf, Vbuf]:
        n, m = grad_out.shape[0], grad_out.shape[1]

        if Ops.backend == "metal":
            dX_buf = (c_float * (n * m))()
            db_buf = (c_float * m)()
            c.matadd_broadcast_backward_metal(grad_out.to_float32(), dX_buf, db_buf, n, m)
            return (Vbuf.from_float32(dX_buf, n, m), Vbuf.from_float32(db_buf, 1, m))

        dX_buf = Vbuf.zeros(n, m)
        dB_buf = Vbuf.zeros(1, m)
        Ops._kernel("matadd_broadcast_backward")(grad_out.ptr, dX_buf.ptr, dB_buf.ptr, n, m)
        return (dX_buf, dB_buf)

    @staticmethod
    def sub_forward(A: Vbuf, B: Vbuf) -> Vbuf:
        n, m = A.shape[0], B.shape[1]

        if Ops.backend == "metal":
            out_buf = (c_float * (n * m))()
            c.matsub_forward_metal(A.to_float32(), B.to_float32(), out_buf, n, m)
            return Vbuf.from_float32(out_buf, n, m)

        out = Vbuf.zeros(n, m)
        Ops._kernel("matsub_forward")(A.ptr, B.ptr, out.ptr, n, m)
        return out

    @staticmethod
    def sub_backward(grad_out: Vbuf) -> tuple[Vbuf, Vbuf]:
        n, m = grad_out.shape[0], grad_out.shape[1]

        if Ops.backend == "metal":
            dA_buf = (c_float * (n * m))()
            dB_buf = (c_float * (n * m))()
            c.matsub_backward_metal(grad_out.to_float32(), dA_buf, dB_buf, n, m)
            return (Vbuf.from_float32(dA_buf, n, m), Vbuf.from_float32(dB_buf, n, m))

        dA_buf = Vbuf.zeros(n, m)
        dB_buf = Vbuf.zeros(n, m)
        Ops._kernel("matsub_backward")(grad_out.ptr, dA_buf.ptr, dB_buf.ptr, n, m)
        return (dA_buf, dB_buf)

    @staticmethod
    def hadamard_forward(A: Vbuf, B: Vbuf) -> Vbuf:
        n, m = A.shape[0], B.shape[1]

        if Ops.backend == "metal":
            out_buf = (c_float * (n * m))()
            c.hadamard_forward_metal(A.to_float32(), B.to_float32(), out_buf, n, m)
            return Vbuf.from_float32(out_buf, n, m)

        out = Vbuf.zeros(n, m)
        Ops._kernel("hadamard_forward")(A.ptr, B.ptr, out.ptr, n, m)
        return out

    @staticmethod
    def hadamard_backward(A: Vbuf, B: Vbuf, grad_out: Vbuf) -> tuple[Vbuf, Vbuf]:
        n, m = A.shape[0], B.shape[1]

        if Ops.backend == "metal":
            dA_buf = (c_float * (n * m))()
            dB_buf = (c_float * (n * m))()
            c.hadamard_backward_metal(grad_out.to_float32(), A.to_float32(), B.to_float32(), dA_buf, dB_buf, n, m)
            return (Vbuf.from_float32(dA_buf, n, m), Vbuf.from_float32(dB_buf, n, m))

        dA_buf = Vbuf.zeros(n, m)
        dB_buf = Vbuf.zeros(n, m)
        Ops._kernel("hadamard_backward")(grad_out.ptr, A.ptr, B.ptr, dA_buf.ptr, dB_buf.ptr, n, m)
        return (dA_buf, dB_buf)

    @staticmethod
    def mean_forward(A: Vbuf) -> Vbuf:
        n, m = A.shape[0], A.shape[1]

        if Ops.backend == "metal":
            out_buf = (c_float * 1)()
            c.matmean_forward_metal(A.to_float32(), out_buf, n * m)
            return Vbuf.from_float32(out_buf, 1, 1)

        out = Vbuf.zeros(1, 1) # scalar
        Ops._kernel("matmean_forward")(A.ptr, out.ptr, n * m)
        return out

    @staticmethod # A is input buf
    def mean_backward(A: Vbuf, grad_out: Vbuf) -> Vbuf:
        n_total = A.shape[0] * A.shape[1]
        g = grad_out.data.item()

        if Ops.backend == "metal":
            dx_buf = (c_float * n_total)()
            c.matmean_backward_metal(dx_buf, n_total, g)
            return Vbuf.from_float32(dx_buf, A.shape[0], A.shape[1])

        dX_buf = Vbuf.zeros(A.shape[0], A.shape[1])
        Ops._kernel("matmean_backward")(dX_buf.ptr, n_total, g)
        return dX_buf

    @staticmethod
    def tanh_forward(Z: Vbuf) -> Vbuf:
        n, m = Z.shape[0], Z.shape[1]

        if Ops.backend == "metal":
            out_buf = (c_float * (n * m))()
            c.tanh_forward_metal(Z.to_float32(), out_buf, n, m)
            return Vbuf.from_float32(out_buf, n, m)

        out = Vbuf.zeros(n, m)
        Ops._kernel("tanh_forward")(Z.ptr, out.ptr, n, m)
        return out

    @staticmethod
    def tanh_backward(out: Vbuf, grad_out: Vbuf) -> Vbuf:
        if not Vbuf.shape_eq(out, grad_out):
            raise ValueError(f"tanh_backward shape mismatch: out={out.shape}, grad_out={grad_out.shape}")

        n, m = out.shape[0], out.shape[1]

        if Ops.backend == "metal":
            dZ_buf = (c_float * (n * m))()
            c.tanh_backward_metal(out.to_float32(), grad_out.to_float32(), dZ_buf, n, m)
            return Vbuf.from_float32(dZ_buf, n, m)

        dZ_buf = Vbuf.zeros(n, m)
        Ops._kernel("tanh_backward")(out.ptr, grad_out.ptr, dZ_buf.ptr, n, m)
        return dZ_buf

    @staticmethod
    def relu_forward(Z: Vbuf) -> Vbuf:
        n, m = Z.shape[0], Z.shape[1]

        if Ops.backend == "metal":
            out_buf = (c_float * (n * m))()
            c.relu_forward_metal(Z.to_float32(), out_buf, n, m)
            return Vbuf.from_float32(out_buf, n, m)

        out = Vbuf.zeros(n, m)
        Ops._kernel("relu_forward")(Z.ptr, out.ptr, n, m)
        return out

    @staticmethod
    def relu_backward(out: Vbuf, grad_out: Vbuf) -> Vbuf:
        if not Vbuf.shape_eq(out, grad_out):
            raise ValueError(f"relu_backward shape mismatch: out={out.shape}, grad_out={grad_out.shape}")

        n, m = out.shape[0], out.shape[1]

        if Ops.backend == "metal":
            dZ_buf = (c_float * (n * m))()
            c.relu_backward_metal(out.to_float32(), grad_out.to_float32(), dZ_buf, n, m)
            return Vbuf.from_float32(dZ_buf, n, m)

        dZ_buf = Vbuf.zeros(n, m)
        Ops._kernel("relu_backward")(out.ptr, grad_out.ptr, dZ_buf.ptr, n, m)
        return dZ_buf

    @staticmethod # fused softmax + cross-entropy; returns (probs cached for backward, scalar loss)
    def softmax_xent_forward(Z: Vbuf, Y: Vbuf) -> tuple[Vbuf, Vbuf]:
        if not Vbuf.shape_eq(Z, Y):
            raise ValueError(f"softmax_xent shape mismatch: Z={Z.shape}, Y={Y.shape}")

        n, m = Z.shape[0], Z.shape[1]

        if Ops.backend == "metal":
            probs_buf = (c_float * (n * m))()
            loss_buf = (c_float * 1)()
            c.softmax_xent_forward_metal(Z.to_float32(), Y.to_float32(), probs_buf, loss_buf, n, m)
            return (Vbuf.from_float32(probs_buf, n, m), Vbuf.from_float32(loss_buf, 1, 1))

        probs = Vbuf.zeros(n, m)
        loss = Vbuf.zeros(1, 1) # scalar
        Ops._kernel("softmax_xent_forward")(Z.ptr, Y.ptr, probs.ptr, loss.ptr, n, m)
        return (probs, loss)

    @staticmethod # probs is the cached softmax from forward
    def softmax_xent_backward(probs: Vbuf, Y: Vbuf, grad_out: Vbuf) -> Vbuf:
        n, m = probs.shape[0], probs.shape[1]
        g = grad_out.data.item()

        if Ops.backend == "metal":
            dZ_buf = (c_float * (n * m))()
            c.softmax_xent_backward_metal(probs.to_float32(), Y.to_float32(), dZ_buf, g, n, m)
            return Vbuf.from_float32(dZ_buf, n, m)

        dZ_buf = Vbuf.zeros(n, m)
        Ops._kernel("softmax_xent_backward")(probs.ptr, Y.ptr, dZ_buf.ptr, g, n, m)
        return dZ_buf
