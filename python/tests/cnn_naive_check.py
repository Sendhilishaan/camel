"""Naive float64 CNN checks. CPU only, tiny tensors, no third-party dependencies.

The shared contracts cover forward/backward references, finite differences,
pooling ties/overlap/lifetime, full CNN graphs, module APIs and validation.
This suite adds float64 precision, native buffer boundaries and optimizer checks.
"""
import math
import unittest
from ctypes import c_double, c_int

from camel import array as ca
from camel._c import c, CNN_NAIVE_AVAILABLE
from camel.ops import Ops, Vbuf
from camel.tensor import Tensor
from camel.optim import SGD, AdaGrad, RMSprop, Adam
from tests.cnn_checks import CnnChecks
from tests import cnn_reference as ref


class CnnNaiveChecks(CnnChecks, unittest.TestCase):
    backend = "naive"
    scalar = c_double
    eps = 1e-6

    @classmethod
    def setUpClass(cls):
        if not CNN_NAIVE_AVAILABLE:
            raise RuntimeError("rebuild camel.dll with the naive CNN kernels")

    def test_float64_precision(self):
        x = Vbuf(ca.CamelArray([1.0 + 1e-10, 1.0, 1.0, 1.0]).reshape(1, 1, 2, 2))
        w = Vbuf(ca.ones((1, 1, 1, 1)))
        out = Ops.conv2d_forward(x, w)
        self.assertEqual(out.data[0, 0, 0, 0], 1.0 + 1e-10)
        # A float32 conversion would erase the difference and pick index zero.
        x.data[0, 0, 0, 1] = 1.0 + 2e-10
        pooled, cache = Ops.maxpool2d_forward(x)
        self.assertEqual(pooled.data.item(), 1.0 + 2e-10)
        self.assertEqual(list(cache.indices), [1])
        dx = Ops.maxpool2d_backward(Vbuf(ca.ones(pooled.shape)), cache)
        self.close(dx.data, ca.CamelArray([0, 1, 0, 0]).reshape(x.shape), rtol=0, atol=0)

    def test_native_output_overwrite(self):
        x, w, b = self.sample((2, 2, 3, 4)), self.sample((2, 2, 2, 2)), self.sample((1, 2))
        stride, padding = (1, 2), (1, 0)
        dims, shape = Ops._conv_dims(x, w, stride, padding)
        g = self.sample(shape)
        expected = ref.conv2d(x.data, w.data, b.data, stride, padding, g.data)
        outputs = [(c_double * math.prod(value.shape))(*([99] * math.prod(value.shape))) for value in expected]
        for _ in range(2):
            c.conv2d_forward(x.ptr, w.ptr, b.ptr, outputs[0], *dims)
            c.conv2d_backward(x.ptr, w.ptr, g.ptr, *outputs[1:], *dims)
            for raw, want in zip(outputs, expected):
                self.close(ca.CamelArray(list(raw)).reshape(want.shape), want)
        pool_out, cache = Ops.maxpool2d_forward(x, 2, 1)
        grad = self.sample(pool_out.shape)
        want_out, want_dx, want_indices = ref.maxpool2d(x.data, (2, 2), (1, 1), grad.data)
        raw_dx = (c_double * x.size)(*([99] * x.size))
        raw_out = (c_double * pool_out.size)(*([99] * pool_out.size))
        indices = (c_int * pool_out.size)(*([-1] * pool_out.size))
        for _ in range(2):
            c.maxpool2d_forward(x.ptr, raw_out, indices, *x.shape, 2, 2, 1, 1)
            self.close(ca.CamelArray(list(raw_out)).reshape(pool_out.shape), want_out)
            self.assertEqual(list(indices), want_indices)
            c.maxpool2d_backward(grad.ptr, cache.indices, raw_dx, *x.shape, 2, 2, 1, 1)
            self.close(ca.CamelArray(list(raw_dx)).reshape(x.shape), want_dx)

    def test_native_error_boundaries(self):
        x, w = self.sample((1, 1, 3, 3)), self.sample((1, 1, 2, 2))
        dims = (1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0)
        out = (c_double * 4)(11, 12, 13, 14)
        for slot in (0, 1, 3):
            args = [x.ptr, w.ptr, None, out, *dims]
            args[slot] = None
            with self.assertRaises(RuntimeError):
                c.conv2d_forward(*args)
            self.assertEqual(list(out), [11, 12, 13, 14])
        for slot, value in ((0, 0), (1, -1), (0, 2147483647), (4, 2147483647),
                            (5, 0), (6, 4), (7, 0), (8, -1), (9, -1), (9, 2147483647),
                            (9, 1073741822)):
            invalid = list(dims)
            invalid[slot] = value
            with self.assertRaises(RuntimeError):
                c.conv2d_forward(x.ptr, w.ptr, None, out, *invalid)
            self.assertEqual(list(out), [11, 12, 13, 14])
        dx, dw, db = (c_double * 9)(*([77] * 9)), (c_double * 4)(*([88] * 4)), (c_double * 1)(99)
        for slot in range(6):
            args = [x.ptr, w.ptr, out, dx, dw, db, *dims]
            args[slot] = None
            with self.assertRaises(RuntimeError):
                c.conv2d_backward(*args)
            self.assertEqual((list(dx), list(dw), list(db)), ([77] * 9, [88] * 4, [99]))
        pdims = (1, 1, 3, 3, 2, 2, 2, 2)
        indices = (c_int * 1)(0)
        for kernel, args in ((c.maxpool2d_forward, [x.ptr, out, indices]),
                             (c.maxpool2d_backward, [out, indices, dx])):
            for slot in range(3):
                invalid = args.copy()
                invalid[slot] = None
                with self.assertRaises(RuntimeError):
                    kernel(*invalid, *pdims)
            for slot, value in ((0, 0), (0, 2147483647), (4, 0), (5, 4), (6, 0), (7, -1)):
                invalid = list(pdims)
                invalid[slot] = value
                with self.assertRaises(RuntimeError):
                    kernel(*args, *invalid)
        for index in (-2147483648, -1, 2, 8, 9, 2147483647):
            # Some indices are inside the image but outside this output's window.
            with self.assertRaises(RuntimeError):
                c.maxpool2d_backward(out, (c_int * 1)(index), dx, *pdims)
            self.assertEqual(list(dx), [77] * 9)
        # Reject a cache entry pointing into the other channel, before writing dx.
        with self.assertRaises(RuntimeError):
            c.maxpool2d_backward((c_double * 2)(1, 1), (c_int * 2)(4, 4),
                                (c_double * 8)(), 1, 2, 2, 2, 2, 2, 2, 2)
        self.assertEqual(Ops.conv2d_forward(x, w).shape, (1, 1, 2, 2))
        self.assertIsNone(c.camel_naive_get_last_error())
        pooled, cache = Ops.maxpool2d_forward(x)
        self.assertEqual(Ops.maxpool2d_backward(Vbuf(ca.ones(pooled.shape)), cache).shape, x.shape)

    def test_optimisers_on_4d_weights(self):
        initial = [0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        gradient = [0.3, -0.4, 0.1, 0.2, -0.5, 0.6, 0.7, 0.8]
        shape = (2, 1, 2, 2)
        for name, factory in (("sgd", lambda p: SGD(p, 0.02, 0.9)),
                              ("adagrad", lambda p: AdaGrad(p, 0.02)),
                              ("rmsprop", lambda p: RMSprop(p, 0.02)),
                              ("adam", lambda p: Adam(p, 0.02))):
            with self.subTest(optimizer=name):
                p = Tensor(Vbuf(ca.CamelArray(initial).reshape(shape)))
                opt = factory([p])
                expected, first, second = initial.copy(), [0.0] * 8, [0.0] * 8
                for step in range(1, 4):
                    values = [g * step for g in gradient]
                    p.grad = Vbuf(ca.CamelArray(values).reshape(shape))
                    for i, g in enumerate(values):
                        if name == "sgd":
                            first[i] = 0.9 * first[i] + g
                            update = first[i]
                        elif name == "adagrad":
                            second[i] += g * g
                            update = g / (math.sqrt(second[i]) + 1e-8)
                        elif name == "rmsprop":
                            second[i] = 0.9 * second[i] + 0.1 * g * g
                            update = g / (math.sqrt(second[i]) + 1e-8)
                        else:
                            first[i] = 0.9 * first[i] + 0.1 * g
                            second[i] = 0.999 * second[i] + 0.001 * g * g
                            update = (first[i] / (1 - 0.9**step)) / (math.sqrt(second[i] / (1 - 0.999**step)) + 1e-8)
                        expected[i] -= 0.02 * update
                    opt.step()
                    self.close(p.buf.data, ca.CamelArray(expected).reshape(shape))
                    states = {"velocity": first, "grad_sum": second, "ema_sq": second,
                              "exp_avg": first, "exp_avg_sq": second}
                    for attr, values in states.items():
                        for state in getattr(opt, attr, []):
                            self.close(state.data, ca.CamelArray(values).reshape(shape))
                            self.assert_storage(state)
                opt.zero_grad()
                self.assertIsNone(p.grad)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CnnNaiveChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
