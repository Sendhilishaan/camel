from __future__ import annotations
import itertools
import math
import random as _random
from ctypes import c_double, cast, POINTER

"""
CamelArray: a numpy-free ndarray clone - a flat ctypes double buffer plus a
shape tuple, giving the C boundary a contiguous double* without numpy.
"""

# saved before the ufuncs below shadow the builtin of the same name
_builtin_abs = abs

Shape = tuple


def _prod(shape: Shape) -> int:
    p = 1
    for s in shape:
        p *= s
    return p


def _row_major_strides(shape: Shape) -> Shape:
    # stride[i]: flat elements to skip to advance index i by one
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _infer_shape(nested) -> Shape:
    if isinstance(nested, (int, float)):
        return ()
    if not isinstance(nested, (list, tuple)):
        raise TypeError(f"CamelArray can't be built from {type(nested).__name__}")
    if len(nested) == 0:
        return (0,)
    inner = _infer_shape(nested[0])
    for row in nested[1:]:
        if _infer_shape(row) != inner:
            raise ValueError("ragged nested sequence: all rows must have the same shape")
    return (len(nested),) + inner


def _flatten(nested, out: list) -> None:
    if isinstance(nested, (int, float)):
        out.append(float(nested))
    else:
        for row in nested:
            _flatten(row, out)


def ndindex(shape: Shape):
    """yield every multi-index into `shape` in row-major order, e.g. (2,2) -> (0,0),(0,1),(1,0),(1,1)."""
    yield from itertools.product(*(range(s) for s in shape))


def _is_index_seq(x) -> bool:
    return isinstance(x, (list, tuple, range))


class CamelArray:
    __slots__ = ("_buf", "shape", "_strides")

    def __init__(self, data=None, _shape: Shape | None = None, _buf=None):
        # fast path: wrap an existing buffer under a new shape (used by factories/reshape)
        if _buf is not None:
            self._buf = _buf
            self.shape = _shape
            self._strides = _row_major_strides(_shape)
            return

        if isinstance(data, CamelArray):
            self.shape = data.shape
            self._strides = data._strides
            self._buf = (c_double * len(data._buf))(*data._buf)  # copy, not a view
            return

        shape = _infer_shape(data)
        flat: list = []
        _flatten(data, flat)
        self.shape = shape
        self._strides = _row_major_strides(shape)
        self._buf = (c_double * len(flat))(*flat)

    # -- factories --------------------------------------------------------

    @staticmethod
    def zeros(shape: Shape) -> CamelArray:
        n = _prod(shape)
        return CamelArray(_buf=(c_double * n)(), _shape=tuple(shape))

    @staticmethod
    def ones(shape: Shape) -> CamelArray:
        n = _prod(shape)
        return CamelArray(_buf=(c_double * n)(*([1.0] * n)), _shape=tuple(shape))

    @staticmethod
    def zeros_like(a: CamelArray) -> CamelArray:
        return CamelArray.zeros(a.shape)

    # -- ctypes interop -----------------------------------------------------

    @property
    def ptr(self) -> POINTER(c_double):
        """raw double* into this array's buffer, for passing across the ctypes/C boundary."""
        return cast(self._buf, POINTER(c_double))

    # -- basics -------------------------------------------------------------

    def copy(self) -> CamelArray:
        return CamelArray(self)

    def item(self) -> float:
        if len(self._buf) != 1:
            raise ValueError(f"item() needs a single-element array, got shape {self.shape}")
        return self._buf[0]

    def tolist(self) -> list:
        return list(self._buf)

    def reshape(self, *shape) -> CamelArray:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = list(shape)
        total = len(self._buf)

        if shape.count(-1) > 1:
            raise ValueError("can only infer one -1 dimension in reshape")
        if -1 in shape:
            known = _prod(s for s in shape if s != -1)
            shape[shape.index(-1)] = total // known if known else 0
        shape = tuple(shape)

        if _prod(shape) != total:
            raise ValueError(f"cannot reshape array of size {total} into shape {shape}")
        return CamelArray(_buf=self._buf, _shape=shape)  # view: shares the same buffer

    def __repr__(self) -> str:
        return f"CamelArray(shape={self.shape}, data={list(self._buf)})"

    # -- indexing: scalar get/set, bare `[:]` overwrite, and fancy one-hot indexing --

    def _flat_index(self, idx: tuple) -> int:
        if len(idx) != len(self.shape):
            raise IndexError(f"expected a {len(self.shape)}-d index, got {idx!r}")
        # not `sum()` - that name is shadowed by the ufunc below
        flat = 0
        for i, s in zip(idx, self._strides):
            flat += i * s
        return flat

    def __getitem__(self, key):
        if isinstance(key, int):
            key = (key,)
        if isinstance(key, tuple) and all(isinstance(k, int) for k in key):
            return self._buf[self._flat_index(key)]
        raise TypeError(f"unsupported CamelArray index: {key!r}")

    def __setitem__(self, key, value) -> None:
        if isinstance(key, slice) and key == slice(None):
            other = value if isinstance(value, CamelArray) else CamelArray(value)
            if _prod(other.shape) != len(self._buf):
                raise ValueError(f"cannot assign shape {other.shape} into buffer of size {len(self._buf)}")
            for i in range(len(self._buf)):
                self._buf[i] = other._buf[i]
            return

        if isinstance(key, int):
            key = (key,)

        if isinstance(key, tuple) and len(key) == 2 and _is_index_seq(key[0]) and _is_index_seq(key[1]):
            rows, cols = list(key[0]), list(key[1])
            if len(rows) != len(cols):
                raise ValueError("fancy index sequences must have the same length")
            for r, c in zip(rows, cols):
                self._buf[self._flat_index((r, c))] = float(value)
            return

        if isinstance(key, tuple) and all(isinstance(k, int) for k in key):
            self._buf[self._flat_index(key)] = float(value)
            return

        raise TypeError(f"unsupported CamelArray index: {key!r}")

    # -- elementwise arithmetic (broadcasting) -------------------------------

    def __add__(self, other): return _broadcast_op(self, other, lambda x, y: x + y)
    def __radd__(self, other): return _map(self, lambda x: other + x)
    def __sub__(self, other): return _broadcast_op(self, other, lambda x, y: x - y)
    def __rsub__(self, other): return _map(self, lambda x: other - x)
    def __mul__(self, other): return _broadcast_op(self, other, lambda x, y: x * y)
    __rmul__ = __mul__  # multiplication is commutative
    def __truediv__(self, other): return _broadcast_op(self, other, lambda x, y: x / y)
    def __rtruediv__(self, other): return _map(self, lambda x: other / x)
    def __neg__(self): return _map(self, lambda x: -x)
    def __abs__(self): return _map(self, lambda x: x if x >= 0 else -x)

    def __iadd__(self, other): self._inplace(other, lambda x, y: x + y); return self
    def __isub__(self, other): self._inplace(other, lambda x, y: x - y); return self
    def __imul__(self, other): self._inplace(other, lambda x, y: x * y); return self
    def __itruediv__(self, other): self._inplace(other, lambda x, y: x / y); return self

    def _inplace(self, other, op) -> None:
        # mutates in place (no realloc) so cached pointers like Vbuf.ptr stay valid
        if isinstance(other, (int, float)):
            for i in range(len(self._buf)):
                self._buf[i] = op(self._buf[i], other)
            return
        if isinstance(other, CamelArray):
            if other.shape != self.shape:
                raise ValueError(f"in-place op shape mismatch: {self.shape} vs {other.shape}")
            for i in range(len(self._buf)):
                self._buf[i] = op(self._buf[i], other._buf[i])
            return
        raise TypeError(f"unsupported operand for in-place op: {type(other).__name__}")

    # -- reductions -----------------------------------------------------------

    def sum(self, axis: int | None = None, keepdims: bool = False):
        return self._reduce(axis, keepdims, lambda a, b: a + b)

    def max(self, axis: int | None = None, keepdims: bool = False):
        return self._reduce(axis, keepdims, lambda a, b: a if a > b else b)

    def mean(self, axis: int | None = None, keepdims: bool = False):
        s = self.sum(axis, keepdims)
        n = _prod(self.shape) if axis is None else self.shape[axis if axis >= 0 else axis + len(self.shape)]
        return s / n

    def std(self) -> float:
        # population std (ddof=0) over the whole array, matching numpy's default
        m = self.mean()
        var = 0.0
        for v in self._buf:
            var += (v - m) ** 2
        return math.sqrt(var / len(self._buf))

    def _reduce(self, axis: int | None, keepdims: bool, combine):
        if axis is None:
            acc = self._buf[0]
            for v in self._buf[1:]:
                acc = combine(acc, v)
            return acc

        ndim = len(self.shape)
        axis = axis if axis >= 0 else axis + ndim
        out_shape = list(self.shape)
        out_shape[axis] = 1
        out = CamelArray.zeros(tuple(out_shape))
        started = [False] * _prod(out.shape)

        for idx in ndindex(self.shape):
            out_idx = list(idx)
            out_idx[axis] = 0
            flat_out = out._flat_index(tuple(out_idx))
            v = self._buf[self._flat_index(idx)]
            if not started[flat_out]:
                out._buf[flat_out] = v
                started[flat_out] = True
            else:
                out._buf[flat_out] = combine(out._buf[flat_out], v)

        if not keepdims:
            squeezed = tuple(s for i, s in enumerate(self.shape) if i != axis)
            out = out.reshape(*squeezed) if squeezed else out
        return out


def _broadcast_shape(shape_a: Shape, shape_b: Shape) -> Shape:
    ndim = len(shape_a) if len(shape_a) > len(shape_b) else len(shape_b)
    a = (1,) * (ndim - len(shape_a)) + shape_a
    b = (1,) * (ndim - len(shape_b)) + shape_b
    out = []
    for da, db in zip(a, b):
        if da != db and da != 1 and db != 1:
            raise ValueError(f"shape mismatch for broadcasting: {shape_a} vs {shape_b}")
        out.append(da if da != 1 else db)
    return tuple(out)


def _broadcast_index(idx: tuple, shape: Shape) -> tuple:
    # right-aligns shape to the output index; size-1 dims collapse to index 0
    offset = len(idx) - len(shape)
    return tuple(0 if s == 1 else i for i, s in zip(idx[offset:], shape))


def _broadcast_op(a: CamelArray, b, op) -> CamelArray:
    if isinstance(b, (int, float)):
        return _map(a, lambda x: op(x, b))
    if not isinstance(b, CamelArray):
        return NotImplemented

    out_shape = _broadcast_shape(a.shape, b.shape)
    out = CamelArray.zeros(out_shape)
    for idx in ndindex(out_shape):
        av = a._buf[a._flat_index(_broadcast_index(idx, a.shape))]
        bv = b._buf[b._flat_index(_broadcast_index(idx, b.shape))]
        out._buf[out._flat_index(idx)] = op(av, bv)
    return out


def _map(a: CamelArray, fn) -> CamelArray:
    out = CamelArray.zeros(a.shape)
    for i in range(len(a._buf)):
        out._buf[i] = fn(a._buf[i])
    return out


# -- module-level mirrors of the numpy free functions this repo used --------

def zeros(shape: Shape) -> CamelArray: return CamelArray.zeros(shape)
def ones(shape: Shape) -> CamelArray: return CamelArray.ones(shape)
def zeros_like(a: CamelArray) -> CamelArray: return CamelArray.zeros_like(a)

def linspace(start: float, stop: float, num: int) -> CamelArray:
    if num < 2:
        return CamelArray([start] if num == 1 else [])
    step = (stop - start) / (num - 1)
    return CamelArray([start + i * step for i in range(num)])

def exp(a: CamelArray) -> CamelArray: return _map(a, math.exp)
def log(a: CamelArray) -> CamelArray: return _map(a, math.log)
def sin(a: CamelArray) -> CamelArray: return _map(a, math.sin)
def sqrt(a: CamelArray) -> CamelArray: return _map(a, math.sqrt)
def square(a: CamelArray) -> CamelArray: return _map(a, lambda x: x * x)
def abs(a: CamelArray) -> CamelArray: return _map(a, lambda x: x if x >= 0 else -x)

def sum(a: CamelArray, axis: int | None = None, keepdims: bool = False): return a.sum(axis, keepdims)
def max(a: CamelArray, axis: int | None = None, keepdims: bool = False): return a.max(axis, keepdims)
def mean(a: CamelArray, axis: int | None = None, keepdims: bool = False): return a.mean(axis, keepdims)

def isclose(a: float, b: float, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    return _builtin_abs(a - b) <= atol + rtol * _builtin_abs(b)

def allclose(a: CamelArray, b: CamelArray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    if a.shape != b.shape:
        raise ValueError(f"allclose shape mismatch: {a.shape} vs {b.shape}")
    return all(_builtin_abs(x - y) <= atol + rtol * _builtin_abs(y) for x, y in zip(a._buf, b._buf))


class _Random:
    """stand-in for np.random: same call shapes, backed by the stdlib Mersenne Twister."""

    def __init__(self):
        self._rng = _random.Random()

    def seed(self, n: int) -> None:
        self._rng.seed(n)

    def randn(self, *shape: int) -> CamelArray:
        n = _prod(shape)
        flat = [self._rng.gauss(0.0, 1.0) for _ in range(n)]
        return CamelArray(_buf=(c_double * n)(*flat), _shape=tuple(shape))

    def randint(self, low: int, high: int, size: int) -> list:
        # plain list: only ever used as an index sequence (one-hot targets), never arithmetic
        return [self._rng.randrange(low, high) for _ in range(size)]


random = _Random()
