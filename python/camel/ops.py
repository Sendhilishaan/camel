from __future__ import annotations
import os
from ctypes import c_float, c_void_p, byref
from ._c import c, SIMD_AVAILABLE, METAL_AVAILABLE
from camel.array import CamelArray
from typing import Tuple

class Vbuf:
    def __init__(self, data: CamelArray | None = None, _gpu_handle=None, _shape=None):
        if _gpu_handle is not None:
            # GPU-only: no CPU copy exists yet, materialize() builds one lazily.
            # version=-1 can never match a real CamelArray.version, so the
            # first gpu_handle() call after a materialize() correctly treats
            # the handle as still fresh rather than re-uploading it.
            self._data = None
            self.shape = _shape
            self._gpu = (_gpu_handle, -1)
            return
        self._data = data
        self.shape = data.shape
        self._gpu = None  # (handle, version) - lazily created by gpu_handle()

    def __del__(self):
        if self._gpu is not None:
            c.camel_metal_buffer_free(self._gpu[0])

    @property
    def data(self) -> CamelArray:
        if self._data is None:
            self._materialize()
        return self._data

    @data.setter
    def data(self, value: CamelArray) -> None:
        # `vbuf.data += x` round-trips through this setter even though
        # CamelArray.__iadd__ already mutated in place and returned self
        self._data = value
        if self._gpu is not None:
            c.camel_metal_buffer_free(self._gpu[0])
            self._gpu = None

    @property
    def ptr(self):
        return self.data.ptr

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

    @staticmethod
    def from_gpu(handle, n: int, m: int) -> Vbuf:
        return Vbuf(_gpu_handle=handle, _shape=(n, m))

    def gpu_handle(self):
        # cache hit: a GPU-only Vbuf's handle is always valid (nothing can
        # have mutated it, since materializing would have stamped a version);
        # a CPU-backed one is valid only if nothing mutated .data in place since
        if self._gpu is not None:
            handle, cached_version = self._gpu
            if self._data is None or self._data.version == cached_version:
                return handle
            c.camel_metal_buffer_free(handle)
            self._gpu = None

        n, m = self.shape
        handle = c.camel_metal_buffer_create(self.to_float32(), n * m)
        self._gpu = (handle, self.data.version)
        return handle

    def _materialize(self) -> None:
        handle, _ = self._gpu
        n, m = self.shape
        buf = (c_float * (n * m))()
        c.camel_metal_buffer_read(handle, buf, n * m)
        self._data = CamelArray(list(buf)).reshape(n, m)
        self._gpu = (handle, self._data.version) # re-sync: freshly read, not stale

    def mark_gpu_mutated(self) -> None:
        # for callers that mutate this Vbuf's cached GPU buffer directly, in
        # place (see the optimizer step kernels): drops the now-stale CPU
        # copy so the next .data access re-reads the fresh value from GPU
        self._data = None


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
            out_handle = c.matmul_forward_metal_resident(A.gpu_handle(), B.gpu_handle(), n, k, m)
            return Vbuf.from_gpu(out_handle, n, m)

        result_buf = Vbuf.zeros(n, m)
        Ops._kernel("matmul_forward")(A.ptr, B.ptr, result_buf.ptr, n, k, m)
        return result_buf

    @staticmethod
    def matmul_backward(A: Vbuf, B: Vbuf, grad_out: Vbuf) -> tuple[Vbuf, Vbuf]:
        n, k, m = A.shape[0], A.shape[1], B.shape[1]

        if Ops.backend == "metal":
            da_h, db_h = c_void_p(), c_void_p()
            c.matmul_backward_metal_resident(A.gpu_handle(), B.gpu_handle(), grad_out.gpu_handle(),
                                              n, k, m, byref(da_h), byref(db_h))
            return (Vbuf.from_gpu(da_h.value, n, k), Vbuf.from_gpu(db_h.value, k, m))

        dA_buf = Vbuf.zeros(n, k)
        dB_buf = Vbuf.zeros(k, m)
        Ops._kernel("matmul_backward")(A.ptr, B.ptr, grad_out.ptr, dA_buf.ptr, dB_buf.ptr, n, k, m)
        return (dA_buf, dB_buf)

    @staticmethod
    def add_forward(A: Vbuf, B:Vbuf) -> Vbuf:
        n, m = A.shape[0], A.shape[1]

        if Ops.backend == "metal":
            out_handle = c.matadd_broadcast_forward_metal_resident(A.gpu_handle(), B.gpu_handle(), n, m)
            return Vbuf.from_gpu(out_handle, n, m)

        out = Vbuf(A.data.copy()) # add kernel is inplace (fix?)
        Ops._kernel("matadd_broadcast_forward")(out.ptr, B.ptr, n, m)
        return out

    @staticmethod
    def add_backward(grad_out: Vbuf) -> tuple[Vbuf, Vbuf]:
        n, m = grad_out.shape[0], grad_out.shape[1]

        if Ops.backend == "metal":
            dx_h, db_h = c_void_p(), c_void_p()
            c.matadd_broadcast_backward_metal_resident(grad_out.gpu_handle(), n, m, byref(dx_h), byref(db_h))
            return (Vbuf.from_gpu(dx_h.value, n, m), Vbuf.from_gpu(db_h.value, 1, m))

        dX_buf = Vbuf.zeros(n, m)
        dB_buf = Vbuf.zeros(1, m)
        Ops._kernel("matadd_broadcast_backward")(grad_out.ptr, dX_buf.ptr, dB_buf.ptr, n, m)
        return (dX_buf, dB_buf)

    @staticmethod
    def sub_forward(A: Vbuf, B: Vbuf) -> Vbuf:
        n, m = A.shape[0], B.shape[1]

        if Ops.backend == "metal":
            out_handle = c.matsub_forward_metal_resident(A.gpu_handle(), B.gpu_handle(), n, m)
            return Vbuf.from_gpu(out_handle, n, m)

        out = Vbuf.zeros(n, m)
        Ops._kernel("matsub_forward")(A.ptr, B.ptr, out.ptr, n, m)
        return out

    @staticmethod
    def sub_backward(grad_out: Vbuf) -> tuple[Vbuf, Vbuf]:
        n, m = grad_out.shape[0], grad_out.shape[1]

        if Ops.backend == "metal":
            da_h, db_h = c_void_p(), c_void_p()
            c.matsub_backward_metal_resident(grad_out.gpu_handle(), n, m, byref(da_h), byref(db_h))
            return (Vbuf.from_gpu(da_h.value, n, m), Vbuf.from_gpu(db_h.value, n, m))

        dA_buf = Vbuf.zeros(n, m)
        dB_buf = Vbuf.zeros(n, m)
        Ops._kernel("matsub_backward")(grad_out.ptr, dA_buf.ptr, dB_buf.ptr, n, m)
        return (dA_buf, dB_buf)

    @staticmethod
    def hadamard_forward(A: Vbuf, B: Vbuf) -> Vbuf:
        n, m = A.shape[0], B.shape[1]

        if Ops.backend == "metal":
            out_handle = c.hadamard_forward_metal_resident(A.gpu_handle(), B.gpu_handle(), n, m)
            return Vbuf.from_gpu(out_handle, n, m)

        out = Vbuf.zeros(n, m)
        Ops._kernel("hadamard_forward")(A.ptr, B.ptr, out.ptr, n, m)
        return out

    @staticmethod
    def hadamard_backward(A: Vbuf, B: Vbuf, grad_out: Vbuf) -> tuple[Vbuf, Vbuf]:
        n, m = A.shape[0], B.shape[1]

        if Ops.backend == "metal":
            da_h, db_h = c_void_p(), c_void_p()
            c.hadamard_backward_metal_resident(grad_out.gpu_handle(), A.gpu_handle(), B.gpu_handle(),
                                                n, m, byref(da_h), byref(db_h))
            return (Vbuf.from_gpu(da_h.value, n, m), Vbuf.from_gpu(db_h.value, n, m))

        dA_buf = Vbuf.zeros(n, m)
        dB_buf = Vbuf.zeros(n, m)
        Ops._kernel("hadamard_backward")(grad_out.ptr, A.ptr, B.ptr, dA_buf.ptr, dB_buf.ptr, n, m)
        return (dA_buf, dB_buf)

    @staticmethod
    def mean_forward(A: Vbuf) -> Vbuf:
        n, m = A.shape[0], A.shape[1]

        if Ops.backend == "metal":
            out_handle = c.matmean_forward_metal_resident(A.gpu_handle(), n * m)
            return Vbuf.from_gpu(out_handle, 1, 1)

        out = Vbuf.zeros(1, 1) # scalar
        Ops._kernel("matmean_forward")(A.ptr, out.ptr, n * m)
        return out

    @staticmethod # A is input buf
    def mean_backward(A: Vbuf, grad_out: Vbuf) -> Vbuf:
        n_total = A.shape[0] * A.shape[1]
        g = grad_out.data.item() # scalar grad, always materialized to CPU

        if Ops.backend == "metal":
            out_handle = c.matmean_backward_metal_resident(n_total, g)
            return Vbuf.from_gpu(out_handle, A.shape[0], A.shape[1])

        dX_buf = Vbuf.zeros(A.shape[0], A.shape[1])
        Ops._kernel("matmean_backward")(dX_buf.ptr, n_total, g)
        return dX_buf

    @staticmethod
    def tanh_forward(Z: Vbuf) -> Vbuf:
        n, m = Z.shape[0], Z.shape[1]

        if Ops.backend == "metal":
            out_handle = c.tanh_forward_metal_resident(Z.gpu_handle(), n, m)
            return Vbuf.from_gpu(out_handle, n, m)

        out = Vbuf.zeros(n, m)
        Ops._kernel("tanh_forward")(Z.ptr, out.ptr, n, m)
        return out

    @staticmethod
    def tanh_backward(out: Vbuf, grad_out: Vbuf) -> Vbuf:
        if not Vbuf.shape_eq(out, grad_out):
            raise ValueError(f"tanh_backward shape mismatch: out={out.shape}, grad_out={grad_out.shape}")

        n, m = out.shape[0], out.shape[1]

        if Ops.backend == "metal":
            out_handle = c.tanh_backward_metal_resident(out.gpu_handle(), grad_out.gpu_handle(), n, m)
            return Vbuf.from_gpu(out_handle, n, m)

        dZ_buf = Vbuf.zeros(n, m)
        Ops._kernel("tanh_backward")(out.ptr, grad_out.ptr, dZ_buf.ptr, n, m)
        return dZ_buf

    @staticmethod
    def relu_forward(Z: Vbuf) -> Vbuf:
        n, m = Z.shape[0], Z.shape[1]

        if Ops.backend == "metal":
            out_handle = c.relu_forward_metal_resident(Z.gpu_handle(), n, m)
            return Vbuf.from_gpu(out_handle, n, m)

        out = Vbuf.zeros(n, m)
        Ops._kernel("relu_forward")(Z.ptr, out.ptr, n, m)
        return out

    @staticmethod
    def relu_backward(out: Vbuf, grad_out: Vbuf) -> Vbuf:
        if not Vbuf.shape_eq(out, grad_out):
            raise ValueError(f"relu_backward shape mismatch: out={out.shape}, grad_out={grad_out.shape}")

        n, m = out.shape[0], out.shape[1]

        if Ops.backend == "metal":
            out_handle = c.relu_backward_metal_resident(out.gpu_handle(), grad_out.gpu_handle(), n, m)
            return Vbuf.from_gpu(out_handle, n, m)

        dZ_buf = Vbuf.zeros(n, m)
        Ops._kernel("relu_backward")(out.ptr, grad_out.ptr, dZ_buf.ptr, n, m)
        return dZ_buf

    @staticmethod # fused softmax + cross-entropy; returns (probs cached for backward, scalar loss)
    def softmax_xent_forward(Z: Vbuf, Y: Vbuf) -> tuple[Vbuf, Vbuf]:
        if not Vbuf.shape_eq(Z, Y):
            raise ValueError(f"softmax_xent shape mismatch: Z={Z.shape}, Y={Y.shape}")

        n, m = Z.shape[0], Z.shape[1]

        if Ops.backend == "metal":
            probs_h, loss = c_void_p(), c_float()
            c.softmax_xent_forward_metal_resident(Z.gpu_handle(), Y.gpu_handle(), n, m, byref(probs_h), byref(loss))
            return (Vbuf.from_gpu(probs_h.value, n, m), Vbuf(CamelArray([[loss.value]])))

        probs = Vbuf.zeros(n, m)
        loss = Vbuf.zeros(1, 1) # scalar
        Ops._kernel("softmax_xent_forward")(Z.ptr, Y.ptr, probs.ptr, loss.ptr, n, m)
        return (probs, loss)

    @staticmethod # probs is the cached softmax from forward
    def softmax_xent_backward(probs: Vbuf, Y: Vbuf, grad_out: Vbuf) -> Vbuf:
        n, m = probs.shape[0], probs.shape[1]
        g = grad_out.data.item() # scalar grad, always materialized to CPU

        if Ops.backend == "metal":
            out_handle = c.softmax_xent_backward_metal_resident(probs.gpu_handle(), Y.gpu_handle(), g, n, m)
            return Vbuf.from_gpu(out_handle, n, m)

        dZ_buf = Vbuf.zeros(n, m)
        Ops._kernel("softmax_xent_backward")(probs.ptr, Y.ptr, dZ_buf.ptr, g, n, m)
        return dZ_buf

    @staticmethod
    def accumulate(a: Vbuf, b: Vbuf) -> Vbuf:
        # combines two same-shape gradient buffers on a fan-out node
        # (Tensor._accum); only called under the metal backend - naive/simd
        # accumulate directly via CamelArray, no round-trip to avoid there
        n, m = a.shape
        out_handle = c.add_metal_resident(a.gpu_handle(), b.gpu_handle(), n * m)
        return Vbuf.from_gpu(out_handle, n, m)

    # optimizer steps: mutate param + state in place on the GPU, so weights
    # never leave it across a training run. Only called under the metal
    # backend (optim.py keeps its existing CPU path for naive/simd).
    @staticmethod
    def sgd_step(param: Vbuf, velocity: Vbuf, grad: Vbuf, momentum: float, lr: float) -> None:
        n, m = param.shape
        c.sgd_step_metal_resident(param.gpu_handle(), velocity.gpu_handle(), grad.gpu_handle(), momentum, lr, n * m)
        param.mark_gpu_mutated()
        velocity.mark_gpu_mutated()

    @staticmethod
    def adagrad_step(param: Vbuf, grad_sum: Vbuf, grad: Vbuf, lr: float, eps: float) -> None:
        n, m = param.shape
        c.adagrad_step_metal_resident(param.gpu_handle(), grad_sum.gpu_handle(), grad.gpu_handle(), lr, eps, n * m)
        param.mark_gpu_mutated()
        grad_sum.mark_gpu_mutated()

    @staticmethod
    def rmsprop_step(param: Vbuf, ema_sq: Vbuf, grad: Vbuf, lr: float, eps: float, decay: float) -> None:
        n, m = param.shape
        c.rmsprop_step_metal_resident(param.gpu_handle(), ema_sq.gpu_handle(), grad.gpu_handle(), lr, eps, decay, n * m)
        param.mark_gpu_mutated()
        ema_sq.mark_gpu_mutated()

    @staticmethod
    def adam_step(param: Vbuf, exp_avg: Vbuf, exp_avg_sq: Vbuf, grad: Vbuf,
                  lr: float, eps: float, decay1: float, decay2: float, bc1: float, bc2: float) -> None:
        n, m = param.shape
        c.adam_step_metal_resident(param.gpu_handle(), exp_avg.gpu_handle(), exp_avg_sq.gpu_handle(), grad.gpu_handle(),
                                    lr, eps, decay1, decay2, bc1, bc2, n * m)
        param.mark_gpu_mutated()
        exp_avg.mark_gpu_mutated()
        exp_avg_sq.mark_gpu_mutated()
