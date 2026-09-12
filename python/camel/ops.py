from __future__ import annotations
import os
import math
from ctypes import c_float, c_int, c_void_p, byref
from ._c import c, SIMD_AVAILABLE, METAL_AVAILABLE, CUDA_AVAILABLE, CNN_CUDA_AVAILABLE, CNN_NAIVE_AVAILABLE
from camel.array import CamelArray
from typing import Tuple


def _pair(value, name: str, minimum=1):
    pair = (value, value) if type(value) is int else value
    if (not isinstance(pair, (tuple, list)) or len(pair) != 2 or
        any(type(v) is not int or v < minimum or v > 2147483647 for v in pair)):
        raise ValueError(f"{name} must be an int or pair of ints >= {minimum} within int32")
    return tuple(pair)


def _image_shape(shape):
    if len(shape) != 4 or any(type(v) is not int or v <= 0 for v in shape) or math.prod(shape) > 2147483647:
        raise ValueError(f"expected a positive NCHW tensor within int32, got {shape}")


def _window_size(h, w, kernel, stride, padding=(0, 0)):
    kh, kw = kernel
    sh, sw = stride
    ph, pw = padding
    if h + 2 * ph > 2147483647 or w + 2 * pw > 2147483647:
        raise ValueError("padded image dimensions exceed int32")
    oh, ow = (h + 2 * ph - kh) // sh + 1, (w + 2 * pw - kw) // sw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("kernel does not fit the (padded) image")
    return oh, ow


class PoolCache:
    # A separate owner because pooling saves int32 indices, not float activations.
    def __init__(self, handle, input_shape, output_shape, indices=None, kernel=None, stride=None):
        self.handle = handle
        self.input_shape, self.output_shape = input_shape, output_shape
        self.backend = Ops.backend
        self.indices, self.kernel, self.stride = indices, kernel, stride
        self.closed = False

    def close(self):
        if self.handle:
            getattr(c, f"camel_{self.backend}_pool_cache_free")(self.handle)
            self.handle = None
        self.indices = None
        self.closed = True

    def __del__(self):
        self.close()


class Vbuf:
    def __init__(self, data: CamelArray | None = None, _gpu_handle=None, _shape=None, _backend=None):
        self._gpu = None
        self._gpu_backend = None
        if _gpu_handle is not None:
            # GPU-only: no CPU copy exists yet; materialize lazily.
            self._data = None
            self.shape = _shape
            self._gpu = (_gpu_handle, -1)
            self._gpu_backend = _backend
            return
        self._data = data
        self.shape = data.shape

    def __del__(self):
        self._free_gpu()

    def _free_gpu(self):
        if self._gpu is not None:
            # Ownership belongs to the creating backend, even after set_backend.
            getattr(c, f"camel_{self._gpu_backend}_buffer_free")(self._gpu[0])
            self._gpu = None
            self._gpu_backend = None

    @property
    def data(self) -> CamelArray:
        if self._data is None:
            self._materialize()
        return self._data

    @data.setter
    def data(self, value: CamelArray) -> None:
        # += comes through here after mutating CamelArray in place.
        self._free_gpu()
        self._data = value
        self.shape = value.shape

    @property
    def ptr(self):
        return self.data.ptr

    @staticmethod # factory
    def zeros(*shape: int) -> Vbuf:
        return Vbuf(CamelArray.zeros(shape))

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @staticmethod
    def shape_eq(A: Vbuf, B: Vbuf) -> bool:
        return A.shape == B.shape

    def to_float32(self):
        vals = self.data.tolist()
        return (c_float * len(vals))(*vals)

    @staticmethod
    def from_float32(buf, n: int, m: int) -> Vbuf:
        return Vbuf(CamelArray(list(buf)).reshape(n, m))

    @staticmethod
    def from_gpu(handle, *shape: int) -> Vbuf:
        if not handle:
            raise RuntimeError(f"{Ops.backend} returned a null GPU buffer")
        return Vbuf(_gpu_handle=handle, _shape=tuple(shape), _backend=Ops.backend)

    def gpu_handle(self):
        backend = Ops.backend
        if backend not in ("metal", "cuda"):
            raise RuntimeError("gpu_handle requires a GPU backend")
        if self._gpu is not None:
            handle, cached_version = self._gpu
            if self._gpu_backend == backend:
                if self._data is None or self._data.version == cached_version:
                    return handle
            else:
                # Moving between GPU backends goes through the owning backend's read.
                self.data
            self._free_gpu()

        handle = getattr(c, f"camel_{backend}_buffer_create")(self.to_float32(), self.size)
        if not handle:
            raise RuntimeError(f"{backend} could not allocate a GPU buffer")
        self._gpu = (handle, self.data.version)
        self._gpu_backend = backend
        return handle

    def _materialize(self) -> None:
        handle, _ = self._gpu
        buf = (c_float * self.size)()
        getattr(c, f"camel_{self._gpu_backend}_buffer_read")(handle, buf, self.size)
        self._data = CamelArray(list(buf)).reshape(self.shape)
        self._gpu = (handle, self._data.version)

    def mark_gpu_mutated(self) -> None:
        # Optimizers changed the resident allocation; discard the stale CPU copy.
        self._data = None


def _resolve_backend(name: str) -> str:
    if name not in ("naive", "simd", "metal", "cuda"):
        raise ValueError(f"unknown camel backend {name!r}, expected 'naive', 'simd', 'metal' or 'cuda'")
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
    if name == "cuda":
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA backend not available: camel.dll wasn't built with prim_cuda.cu")
        if not c.camel_cuda_device_available():
            raise RuntimeError("CUDA backend needs a working NVIDIA GPU and compatible driver")
    return name


# defaults to simd where built, naive elsewhere; GPUs are opt-in (needs a real
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
    def cuda_available() -> bool:
        return CUDA_AVAILABLE

    @staticmethod
    def cuda_device_available() -> bool:
        return CUDA_AVAILABLE and bool(c.camel_cuda_device_available())

    @staticmethod
    def synchronize() -> None:
        # Metal dispatch already waits; CUDA only needs this for timing/debugging.
        if Ops.backend == "cuda":
            c.camel_cuda_synchronize()

    @staticmethod
    def _gpu_kernel(name: str):
        return getattr(c, f"{name}_{Ops.backend}_resident")

    @staticmethod
    def _kernel(name: str):
        # picks the naive or _simd C function for `name` (GPU backends have their own call convention, see below)
        return getattr(c, f"{name}_simd") if Ops.backend == "simd" else getattr(c, name)

    @staticmethod
    def matmul_forward(A: Vbuf, B: Vbuf) -> Vbuf:
        if any(len(v.shape) != 2 for v in (A, B)):
            raise ValueError("matmul_forward expects 2D tensors; flatten image tensors first")
        n, k, m = A.shape[0], A.shape[1], B.shape[1]

        if Ops.backend in ("metal", "cuda"):
            out_handle = Ops._gpu_kernel("matmul_forward")(A.gpu_handle(), B.gpu_handle(), n, k, m)
            return Vbuf.from_gpu(out_handle, n, m)

        result_buf = Vbuf.zeros(n, m)
        Ops._kernel("matmul_forward")(A.ptr, B.ptr, result_buf.ptr, n, k, m)
        return result_buf

    @staticmethod
    def matmul_backward(A: Vbuf, B: Vbuf, grad_out: Vbuf) -> tuple[Vbuf, Vbuf]:
        if any(len(v.shape) != 2 for v in (A, B, grad_out)):
            raise ValueError("matmul_backward expects 2D tensors; flatten image tensors first")
        n, k, m = A.shape[0], A.shape[1], B.shape[1]

        if Ops.backend in ("metal", "cuda"):
            da_h, db_h = c_void_p(), c_void_p()
            Ops._gpu_kernel("matmul_backward")(A.gpu_handle(), B.gpu_handle(), grad_out.gpu_handle(),
                                              n, k, m, byref(da_h), byref(db_h))
            return (Vbuf.from_gpu(da_h.value, n, k), Vbuf.from_gpu(db_h.value, k, m))

        dA_buf = Vbuf.zeros(n, k)
        dB_buf = Vbuf.zeros(k, m)
        Ops._kernel("matmul_backward")(A.ptr, B.ptr, grad_out.ptr, dA_buf.ptr, dB_buf.ptr, n, k, m)
        return (dA_buf, dB_buf)

    @staticmethod
    def add_forward(A: Vbuf, B:Vbuf) -> Vbuf:
        if any(len(v.shape) != 2 for v in (A, B)):
            raise ValueError("add_forward expects 2D tensors; flatten image tensors first")
        n, m = A.shape[0], A.shape[1]

        if Ops.backend in ("metal", "cuda"):
            out_handle = Ops._gpu_kernel("matadd_broadcast_forward")(A.gpu_handle(), B.gpu_handle(), n, m)
            return Vbuf.from_gpu(out_handle, n, m)

        out = Vbuf(A.data.copy()) # add kernel is inplace (fix?)
        Ops._kernel("matadd_broadcast_forward")(out.ptr, B.ptr, n, m)
        return out

    @staticmethod
    def add_backward(grad_out: Vbuf) -> tuple[Vbuf, Vbuf]:
        if any(len(v.shape) != 2 for v in (grad_out,)):
            raise ValueError("add_backward expects 2D tensors; flatten image tensors first")
        n, m = grad_out.shape[0], grad_out.shape[1]

        if Ops.backend in ("metal", "cuda"):
            dx_h, db_h = c_void_p(), c_void_p()
            Ops._gpu_kernel("matadd_broadcast_backward")(grad_out.gpu_handle(), n, m, byref(dx_h), byref(db_h))
            return (Vbuf.from_gpu(dx_h.value, n, m), Vbuf.from_gpu(db_h.value, 1, m))

        dX_buf = Vbuf.zeros(n, m)
        dB_buf = Vbuf.zeros(1, m)
        Ops._kernel("matadd_broadcast_backward")(grad_out.ptr, dX_buf.ptr, dB_buf.ptr, n, m)
        return (dX_buf, dB_buf)

    @staticmethod
    def sub_forward(A: Vbuf, B: Vbuf) -> Vbuf:
        n, m = 1, A.size

        if Ops.backend in ("metal", "cuda"):
            out_handle = Ops._gpu_kernel("matsub_forward")(A.gpu_handle(), B.gpu_handle(), n, m)
            return Vbuf.from_gpu(out_handle, *A.shape)

        out = Vbuf.zeros(*A.shape)
        Ops._kernel("matsub_forward")(A.ptr, B.ptr, out.ptr, n, m)
        return out

    @staticmethod
    def sub_backward(grad_out: Vbuf) -> tuple[Vbuf, Vbuf]:
        n, m = 1, grad_out.size

        if Ops.backend in ("metal", "cuda"):
            da_h, db_h = c_void_p(), c_void_p()
            Ops._gpu_kernel("matsub_backward")(grad_out.gpu_handle(), n, m, byref(da_h), byref(db_h))
            return (Vbuf.from_gpu(da_h.value, *grad_out.shape), Vbuf.from_gpu(db_h.value, *grad_out.shape))

        dA_buf = Vbuf.zeros(*grad_out.shape)
        dB_buf = Vbuf.zeros(*grad_out.shape)
        Ops._kernel("matsub_backward")(grad_out.ptr, dA_buf.ptr, dB_buf.ptr, n, m)
        return (dA_buf, dB_buf)

    @staticmethod
    def hadamard_forward(A: Vbuf, B: Vbuf) -> Vbuf:
        n, m = 1, A.size

        if Ops.backend in ("metal", "cuda"):
            out_handle = Ops._gpu_kernel("hadamard_forward")(A.gpu_handle(), B.gpu_handle(), n, m)
            return Vbuf.from_gpu(out_handle, *A.shape)

        out = Vbuf.zeros(*A.shape)
        Ops._kernel("hadamard_forward")(A.ptr, B.ptr, out.ptr, n, m)
        return out

    @staticmethod
    def hadamard_backward(A: Vbuf, B: Vbuf, grad_out: Vbuf) -> tuple[Vbuf, Vbuf]:
        n, m = 1, A.size

        if Ops.backend in ("metal", "cuda"):
            da_h, db_h = c_void_p(), c_void_p()
            Ops._gpu_kernel("hadamard_backward")(grad_out.gpu_handle(), A.gpu_handle(), B.gpu_handle(),
                                                n, m, byref(da_h), byref(db_h))
            return (Vbuf.from_gpu(da_h.value, *A.shape), Vbuf.from_gpu(db_h.value, *A.shape))

        dA_buf = Vbuf.zeros(*A.shape)
        dB_buf = Vbuf.zeros(*A.shape)
        Ops._kernel("hadamard_backward")(grad_out.ptr, A.ptr, B.ptr, dA_buf.ptr, dB_buf.ptr, n, m)
        return (dA_buf, dB_buf)

    @staticmethod
    def mean_forward(A: Vbuf) -> Vbuf:
        n, m = 1, A.size

        if Ops.backend in ("metal", "cuda"):
            out_handle = Ops._gpu_kernel("matmean_forward")(A.gpu_handle(), n * m)
            return Vbuf.from_gpu(out_handle, 1, 1)

        out = Vbuf.zeros(1, 1) # scalar
        Ops._kernel("matmean_forward")(A.ptr, out.ptr, n * m)
        return out

    @staticmethod # A is input buf
    def mean_backward(A: Vbuf, grad_out: Vbuf) -> Vbuf:
        n_total = A.size
        g = grad_out.data.item() # scalar grad, always materialized to CPU

        if Ops.backend in ("metal", "cuda"):
            out_handle = Ops._gpu_kernel("matmean_backward")(n_total, g)
            return Vbuf.from_gpu(out_handle, *A.shape)

        dX_buf = Vbuf.zeros(*A.shape)
        Ops._kernel("matmean_backward")(dX_buf.ptr, n_total, g)
        return dX_buf

    @staticmethod
    def tanh_forward(Z: Vbuf) -> Vbuf:
        n, m = 1, Z.size

        if Ops.backend in ("metal", "cuda"):
            out_handle = Ops._gpu_kernel("tanh_forward")(Z.gpu_handle(), n, m)
            return Vbuf.from_gpu(out_handle, *Z.shape)

        out = Vbuf.zeros(*Z.shape)
        Ops._kernel("tanh_forward")(Z.ptr, out.ptr, n, m)
        return out

    @staticmethod
    def tanh_backward(out: Vbuf, grad_out: Vbuf) -> Vbuf:
        if not Vbuf.shape_eq(out, grad_out):
            raise ValueError(f"tanh_backward shape mismatch: out={out.shape}, grad_out={grad_out.shape}")

        n, m = 1, out.size

        if Ops.backend in ("metal", "cuda"):
            out_handle = Ops._gpu_kernel("tanh_backward")(out.gpu_handle(), grad_out.gpu_handle(), n, m)
            return Vbuf.from_gpu(out_handle, *out.shape)

        dZ_buf = Vbuf.zeros(*out.shape)
        Ops._kernel("tanh_backward")(out.ptr, grad_out.ptr, dZ_buf.ptr, n, m)
        return dZ_buf

    @staticmethod
    def relu_forward(Z: Vbuf) -> Vbuf:
        n, m = 1, Z.size

        if Ops.backend in ("metal", "cuda"):
            out_handle = Ops._gpu_kernel("relu_forward")(Z.gpu_handle(), n, m)
            return Vbuf.from_gpu(out_handle, *Z.shape)

        out = Vbuf.zeros(*Z.shape)
        Ops._kernel("relu_forward")(Z.ptr, out.ptr, n, m)
        return out

    @staticmethod
    def relu_backward(out: Vbuf, grad_out: Vbuf) -> Vbuf:
        if not Vbuf.shape_eq(out, grad_out):
            raise ValueError(f"relu_backward shape mismatch: out={out.shape}, grad_out={grad_out.shape}")

        n, m = 1, out.size

        if Ops.backend in ("metal", "cuda"):
            out_handle = Ops._gpu_kernel("relu_backward")(out.gpu_handle(), grad_out.gpu_handle(), n, m)
            return Vbuf.from_gpu(out_handle, *out.shape)

        dZ_buf = Vbuf.zeros(*out.shape)
        Ops._kernel("relu_backward")(out.ptr, grad_out.ptr, dZ_buf.ptr, n, m)
        return dZ_buf

    @staticmethod # fused softmax + cross-entropy; returns (probs cached for backward, scalar loss)
    def softmax_xent_forward(Z: Vbuf, Y: Vbuf) -> tuple[Vbuf, Vbuf]:
        if any(len(v.shape) != 2 for v in (Z, Y)):
            raise ValueError("softmax_xent_forward expects 2D tensors; flatten image tensors first")
        if not Vbuf.shape_eq(Z, Y):
            raise ValueError(f"softmax_xent shape mismatch: Z={Z.shape}, Y={Y.shape}")

        n, m = Z.shape[0], Z.shape[1]

        if Ops.backend in ("metal", "cuda"):
            probs_h, loss = c_void_p(), c_float()
            Ops._gpu_kernel("softmax_xent_forward")(Z.gpu_handle(), Y.gpu_handle(), n, m, byref(probs_h), byref(loss))
            return (Vbuf.from_gpu(probs_h.value, n, m), Vbuf(CamelArray([[loss.value]])))

        probs = Vbuf.zeros(n, m)
        loss = Vbuf.zeros(1, 1) # scalar
        Ops._kernel("softmax_xent_forward")(Z.ptr, Y.ptr, probs.ptr, loss.ptr, n, m)
        return (probs, loss)

    @staticmethod # probs is the cached softmax from forward
    def softmax_xent_backward(probs: Vbuf, Y: Vbuf, grad_out: Vbuf) -> Vbuf:
        if any(len(v.shape) != 2 for v in (probs, Y)):
            raise ValueError("softmax_xent_backward expects 2D tensors; flatten image tensors first")
        n, m = probs.shape[0], probs.shape[1]
        g = grad_out.data.item() # scalar grad, always materialized to CPU

        if Ops.backend in ("metal", "cuda"):
            out_handle = Ops._gpu_kernel("softmax_xent_backward")(probs.gpu_handle(), Y.gpu_handle(), g, n, m)
            return Vbuf.from_gpu(out_handle, n, m)

        dZ_buf = Vbuf.zeros(n, m)
        Ops._kernel("softmax_xent_backward")(probs.ptr, Y.ptr, dZ_buf.ptr, g, n, m)
        return dZ_buf

    @staticmethod
    def accumulate(a: Vbuf, b: Vbuf) -> Vbuf:
        # combines two same-shape gradient buffers on a fan-out node
        # (Tensor._accum); only called under a GPU backend - naive/simd
        # accumulate directly via CamelArray, no round-trip to avoid there
        total = a.size
        out_handle = Ops._gpu_kernel("add")(a.gpu_handle(), b.gpu_handle(), total)
        return Vbuf.from_gpu(out_handle, *a.shape)

    # optimizer steps: mutate param + state in place on the GPU, so weights
    # never leave it across a training run. Only called under a GPU
    # backend (optim.py keeps its existing CPU path for naive/simd).
    @staticmethod
    def sgd_step(param: Vbuf, velocity: Vbuf, grad: Vbuf, momentum: float, lr: float) -> None:
        total = param.size
        Ops._gpu_kernel("sgd_step")(param.gpu_handle(), velocity.gpu_handle(), grad.gpu_handle(), momentum, lr, total)
        param.mark_gpu_mutated()
        velocity.mark_gpu_mutated()

    @staticmethod
    def adagrad_step(param: Vbuf, grad_sum: Vbuf, grad: Vbuf, lr: float, eps: float) -> None:
        total = param.size
        Ops._gpu_kernel("adagrad_step")(param.gpu_handle(), grad_sum.gpu_handle(), grad.gpu_handle(), lr, eps, total)
        param.mark_gpu_mutated()
        grad_sum.mark_gpu_mutated()

    @staticmethod
    def rmsprop_step(param: Vbuf, ema_sq: Vbuf, grad: Vbuf, lr: float, eps: float, decay: float) -> None:
        total = param.size
        Ops._gpu_kernel("rmsprop_step")(param.gpu_handle(), ema_sq.gpu_handle(), grad.gpu_handle(), lr, eps, decay, total)
        param.mark_gpu_mutated()
        ema_sq.mark_gpu_mutated()

    @staticmethod
    def adam_step(param: Vbuf, exp_avg: Vbuf, exp_avg_sq: Vbuf, grad: Vbuf,
                  lr: float, eps: float, decay1: float, decay2: float, bc1: float, bc2: float) -> None:
        total = param.size
        Ops._gpu_kernel("adam_step")(param.gpu_handle(), exp_avg.gpu_handle(), exp_avg_sq.gpu_handle(), grad.gpu_handle(),
                                    lr, eps, decay1, decay2, bc1, bc2, total)
        param.mark_gpu_mutated()
        exp_avg.mark_gpu_mutated()
        exp_avg_sq.mark_gpu_mutated()

    @staticmethod
    def _require_cnn():
        if Ops.backend not in ("naive", "cuda"):
            raise NotImplementedError("CNN kernels currently support naive and CUDA; Metal follows")
        if not (CNN_NAIVE_AVAILABLE if Ops.backend == "naive" else CNN_CUDA_AVAILABLE):
            raise RuntimeError(f"rebuild camel.dll with the {Ops.backend} CNN kernels")

    @staticmethod
    def _conv_dims(x: Vbuf, w: Vbuf, stride, padding):
        _image_shape(x.shape)
        _image_shape(w.shape) # filters use OIHW; the same four positive dimensions
        n, channels, h, width = x.shape
        filters, wc, kh, kw = w.shape
        if wc != channels:
            raise ValueError(f"conv2d channel mismatch: input={x.shape}, weights={w.shape}")
        stride, padding = _pair(stride, "stride"), _pair(padding, "padding", 0)
        oh, ow = _window_size(h, width, (kh, kw), stride, padding)
        out_shape = (n, filters, oh, ow)
        _image_shape(out_shape)
        return (n, channels, h, width, filters, kh, kw, *stride, *padding), out_shape

    @staticmethod
    def conv2d_forward(x: Vbuf, w: Vbuf, b: Vbuf | None = None, stride=1, padding=0) -> Vbuf:
        Ops._require_cnn()
        dims, shape = Ops._conv_dims(x, w, stride, padding)
        if b is not None and b.shape != (1, w.shape[0]):
            raise ValueError(f"conv2d bias must have shape {(1, w.shape[0])}, got {b.shape}")
        if Ops.backend == "naive":
            out = Vbuf.zeros(*shape)
            c.conv2d_forward(x.ptr, w.ptr, b.ptr if b is not None else None, out.ptr, *dims)
            return out
        handle = Ops._gpu_kernel("conv2d_forward")(x.gpu_handle(), w.gpu_handle(), b.gpu_handle() if b is not None else None, *dims)
        return Vbuf.from_gpu(handle, *shape)

    @staticmethod
    def conv2d_backward(x: Vbuf, w: Vbuf, grad: Vbuf, stride=1, padding=0):
        Ops._require_cnn()
        dims, shape = Ops._conv_dims(x, w, stride, padding)
        if grad.shape != shape:
            raise ValueError(f"conv2d upstream gradient must have shape {shape}, got {grad.shape}")
        if Ops.backend == "naive":
            dx, dw, db = Vbuf.zeros(*x.shape), Vbuf.zeros(*w.shape), Vbuf.zeros(1, w.shape[0])
            c.conv2d_backward(x.ptr, w.ptr, grad.ptr, dx.ptr, dw.ptr, db.ptr, *dims)
            return dx, dw, db
        dx, dw, db = c_void_p(), c_void_p(), c_void_p()
        Ops._gpu_kernel("conv2d_backward")(x.gpu_handle(), w.gpu_handle(), grad.gpu_handle(), *dims, byref(dx), byref(dw), byref(db))
        return (Vbuf.from_gpu(dx.value, *x.shape), Vbuf.from_gpu(dw.value, *w.shape), Vbuf.from_gpu(db.value, 1, w.shape[0]))

    @staticmethod
    def maxpool2d_forward(x: Vbuf, kernel_size=2, stride=None):
        Ops._require_cnn()
        _image_shape(x.shape)
        kernel = _pair(kernel_size, "kernel_size")
        stride = kernel if stride is None else _pair(stride, "stride")
        n, channels, h, w = x.shape
        oh, ow = _window_size(h, w, kernel, stride)
        shape = (n, channels, oh, ow)
        if Ops.backend == "naive":
            out = Vbuf.zeros(*shape)
            indices = (c_int * out.size)()
            c.maxpool2d_forward(x.ptr, out.ptr, indices, *x.shape, *kernel, *stride)
            return out, PoolCache(None, x.shape, shape, indices, kernel, stride)
        out, cache = c_void_p(), c_void_p()
        Ops._gpu_kernel("maxpool2d_forward")(x.gpu_handle(), *x.shape, *kernel, *stride, byref(out), byref(cache))
        return Vbuf.from_gpu(out.value, *shape), PoolCache(cache.value, x.shape, shape)

    @staticmethod
    def maxpool2d_backward(grad: Vbuf, cache: PoolCache) -> Vbuf:
        if cache.backend != Ops.backend or cache.closed:
            raise ValueError("maxpool backward requires a live cache from the current backend")
        Ops._require_cnn()
        if grad.shape != cache.output_shape:
            raise ValueError(f"maxpool upstream gradient must have shape {cache.output_shape}, got {grad.shape}")
        if Ops.backend == "naive":
            dx = Vbuf.zeros(*cache.input_shape)
            c.maxpool2d_backward(grad.ptr, cache.indices, dx.ptr, *cache.input_shape, *cache.kernel, *cache.stride)
            return dx
        handle = Ops._gpu_kernel("maxpool2d_backward")(grad.gpu_handle(), cache.handle)
        return Vbuf.from_gpu(handle, *cache.input_shape)

    @staticmethod
    def reshape(x: Vbuf, shape) -> Vbuf:
        shape = tuple(shape)
        if not shape or any(type(v) is not int or v == 0 or v < -1 for v in shape) or shape.count(-1) > 1:
            raise ValueError("reshape needs positive dimensions and at most one -1")
        if -1 in shape:
            known = math.prod(v for v in shape if v != -1)
            if x.size % known:
                raise ValueError("reshape cannot infer an integral dimension")
            shape = tuple(x.size // known if v == -1 else v for v in shape)
        if math.prod(shape) != x.size or x.size <= 0:
            raise ValueError(f"cannot reshape {x.shape} to {shape}")
        if Ops.backend == "cuda":
            Ops._require_cnn()
            # A device copy keeps the existing fresh-output ownership contract.
            return Vbuf.from_gpu(Ops._gpu_kernel("copy")(x.gpu_handle(), x.size), *shape)
        return Vbuf(x.data.copy().reshape(shape))
