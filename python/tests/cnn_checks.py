"""Shared CNN contracts, checked with tiny tensors on each supported backend."""
import gc
import math
from ctypes import c_float, c_int
from camel import array as ca
from camel._c import c
from camel.ops import Ops, Vbuf
from camel.tensor import Tensor, Tanh
from camel.nn import Module, Conv2d, MaxPool2d, Flatten, Linear, CNN
from camel.optim import SGD
from tests import cnn_reference as ref
from tests.grad_check import numerical_grad


class CnnChecks:
    # Concrete suites select the backend and precision, and probe availability.
    backend = "cuda"
    scalar = c_float
    eps = 1e-3

    def native(self, value):
        return value.ptr if self.backend == "naive" else value.to_float32()

    def kernel(self, name):
        return getattr(c, name if self.backend == "naive" else f"{name}_{self.backend}")

    def assert_storage(self, value):
        if self.backend == "naive":
            self.assertIsNone(value._gpu)
            self.assertIsNotNone(value._data)
        else:
            self.assertIsNone(value._data)

    def gradient_close(self, actual, expected, rtol, atol):
        if self.backend == "naive":
            rtol, atol = 2e-6, 2e-8
        self.close(actual, expected, rtol=rtol, atol=atol)

    def setUp(self):
        self.original = Ops.backend
        Ops.set_backend(self.backend)
        ca.random.seed(13)


    def tearDown(self):
        Ops.set_backend(self.original)
        gc.collect()


    def close(self, actual, expected, rtol=None, atol=None):
        rtol = (1e-12 if self.backend == "naive" else 2e-5) if rtol is None else rtol
        atol = (1e-12 if self.backend == "naive" else 2e-5) if atol is None else atol
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(ca.allclose(actual, expected, rtol=rtol, atol=atol),
                        f"max error: {max(abs(a-b) for a,b in zip(actual.tolist(), expected.tolist()))}")


    def sample(self, shape):
        self.assertLess(math.prod(shape), 512)
        return Vbuf(ca.random.randn(*shape))


    def test_conv_hand_calculation(self):
        x = Vbuf(ca.CamelArray(list(range(1, 10))).reshape(1, 1, 3, 3))
        w = Vbuf(ca.CamelArray([1, 2, 3, 4]).reshape(1, 1, 2, 2))
        b = Vbuf(ca.CamelArray([[0.5]]))
        self.close(Ops.conv2d_forward(x, w, b).data,
                   ca.CamelArray([37.5, 47.5, 67.5, 77.5]).reshape(1, 1, 2, 2))
        dx, dw, db = Ops.conv2d_backward(x, w, Vbuf(ca.ones((1, 1, 2, 2))))
        self.close(dx.data, ca.CamelArray([1, 3, 2, 4, 10, 6, 3, 7, 4]).reshape(1, 1, 3, 3))
        self.close(dw.data, ca.CamelArray([12, 16, 24, 28]).reshape(1, 1, 2, 2))
        self.close(db.data, ca.CamelArray([[4]]))


    def test_conv_reference_and_staging(self):
        # Multi-batch/channel/filter, non-square, 1x1, padding, floor sizing,
        # a full-image filter, and strides leaving uncovered input pixels.
        cases = [
            ((2, 2, 5, 6), (3, 2, 3, 2), (2, 1), (1, 0)),
            ((1, 3, 3, 4), (2, 3, 1, 1), (1, 2), (0, 0)),
            ((1, 1, 3, 4), (1, 1, 3, 4), (1, 1), (0, 0)),
            ((1, 1, 4, 5), (2, 1, 2, 3), (3, 4), (0, 1)),
            ((1, 1, 2, 2), (1, 1, 3, 3), (1, 1), (2, 2)),
        ]
        for xs, ws, stride, padding in cases:
            for use_bias in (False, True):
                with self.subTest(xs=xs, ws=ws, stride=stride, padding=padding, bias=use_bias):
                    x, w, b = self.sample(xs), self.sample(ws), self.sample((1, ws[0]))
                    saved = [v.data.copy() for v in (x, w, b)]
                    out = Ops.conv2d_forward(x, w, b if use_bias else None, stride, padding)
                    self.assert_storage(out)
                    g = self.sample(out.shape)
                    dx, dw, db = Ops.conv2d_backward(x, w, g, stride, padding)
                    for value in (dx, dw, db):
                        self.assert_storage(value)
                    expected = ref.conv2d(x.data, w.data, b.data if use_bias else None, stride, padding, g.data)
                    for actual, want in zip((out, dx, dw, db), expected):
                        self.close(actual.data, want)
                    for v, want in zip((x, w, b), saved):
                        self.close(v.data, want, rtol=0, atol=0)
                    dims = (*xs, ws[0], ws[2], ws[3], *stride, *padding)
                    raw_out = (self.scalar * out.size)()
                    self.kernel("conv2d_forward")(self.native(x), self.native(w), self.native(b) if use_bias else None, raw_out, *dims)
                    self.close(ca.CamelArray(list(raw_out)).reshape(out.shape), expected[0])
                    raw_grads = [(self.scalar * v.size)() for v in (x, w, b)]
                    self.kernel("conv2d_backward")(self.native(x), self.native(w), self.native(g), *raw_grads, *dims)
                    for raw, want in zip(raw_grads, expected[1:]):
                        self.close(ca.CamelArray(list(raw)).reshape(want.shape), want)


    def test_conv_finite_differences(self):
        x, w, b = self.sample((1, 2, 3, 4)), self.sample((2, 2, 2, 2)), self.sample((1, 2))
        stride, padding = (2, 1), (1, 0)
        g = self.sample(Ops.conv2d_forward(x, w, b, stride, padding).shape)
        grads = Ops.conv2d_backward(x, w, g, stride, padding)
        def loss():
            return (Ops.conv2d_forward(x, w, b, stride, padding).data * g.data).sum()
        for variable, grad in zip((x, w, b), grads):
            self.gradient_close(grad.data, numerical_grad(loss, variable, eps=self.eps), rtol=3e-3, atol=8e-4)


    def test_pool_reference_and_staging(self):
        for shape, kernel, stride in (
            ((2, 2, 5, 6), (2, 3), (1, 2)),
            ((1, 2, 4, 5), (2, 2), (3, 3)),
            ((1, 1, 3, 4), (3, 4), (1, 1)),
            ((1, 1, 3, 3), (1, 1), (1, 1)),
        ):
            with self.subTest(shape=shape, kernel=kernel, stride=stride):
                x = self.sample(shape)
                out, cache = Ops.maxpool2d_forward(x, kernel, stride)
                self.assert_storage(out)
                g = self.sample(out.shape)
                dx = Ops.maxpool2d_backward(g, cache)
                expected, expected_dx, expected_idx = ref.maxpool2d(x.data, kernel, stride, g.data)
                self.close(out.data, expected)
                self.close(dx.data, expected_dx)
                raw_out, indices, raw_dx = (self.scalar * out.size)(), (c_int * out.size)(), (self.scalar * x.size)()
                self.kernel("maxpool2d_forward")(self.native(x), raw_out, indices, *shape, *kernel, *stride)
                self.assertEqual(list(indices), expected_idx)
                self.close(ca.CamelArray(list(raw_out)).reshape(out.shape), expected)
                self.kernel("maxpool2d_backward")(self.native(g), indices, raw_dx, *shape, *kernel, *stride)
                self.close(ca.CamelArray(list(raw_dx)).reshape(shape), expected_dx)
                cache.close()


    def test_pool_overlap_ties_and_cached_winners(self):
        x = Vbuf(ca.CamelArray([1, 2, 3, 4, 9, 6, 7, 8, 5]).reshape(1, 1, 3, 3))
        out, cache = Ops.maxpool2d_forward(x, 2, 1)
        g = Vbuf(ca.CamelArray([1, 2, 3, 4]).reshape(out.shape))
        x.data[:] = ca.zeros(x.shape) # backward must use the saved winners
        expected = ca.CamelArray([0, 0, 0, 0, 10, 0, 0, 0, 0]).reshape(x.shape)
        self.close(Ops.maxpool2d_backward(g, cache).data, expected)
        # Reusing the same cache must not accumulate into a previous result.
        self.close(Ops.maxpool2d_backward(g, cache).data, expected)
        tied = Vbuf(ca.ones((1, 1, 2, 2)))
        out, tie_cache = Ops.maxpool2d_forward(tied)
        self.close(Ops.maxpool2d_backward(Vbuf(ca.ones(out.shape)), tie_cache).data,
                   ca.CamelArray([1, 0, 0, 0]).reshape(tied.shape))
        negative = Vbuf(ca.CamelArray([-4, -2, -3, -8]).reshape(1, 1, 2, 2))
        pooled, _ = Ops.maxpool2d_forward(negative)
        self.assertEqual(pooled.data.item(), -2)
        # Index caching is int32; it does not store indices as lossy floats.
        raw, indices = (self.scalar * 1)(), (c_int * 1)()
        self.kernel("maxpool2d_forward")((self.scalar * 4)(*([-math.inf] * 4)), raw, indices, 1, 1, 2, 2, 2, 2, 2, 2)
        self.assertEqual((raw[0], indices[0]), (-math.inf, 0))
        self.kernel("maxpool2d_forward")((self.scalar * 4)(1, math.nan, math.nan, 3), raw, indices, 1, 1, 2, 2, 2, 2, 2, 2)
        self.assertTrue(math.isnan(raw[0]))
        self.assertEqual(indices[0], 1)
        Ops.backend = "simd" # no calls on this backend; only release the saved owner
        cache.close()
        cache.close()


    def test_pool_finite_differences(self):
        x = Vbuf(ca.CamelArray([i * 0.3 for i in range(24)]).reshape(1, 2, 3, 4))
        out, cache = Ops.maxpool2d_forward(x, (2, 2), (1, 2))
        g = self.sample(out.shape)
        dx = Ops.maxpool2d_backward(g, cache)
        def loss():
            return (Ops.maxpool2d_forward(x, (2, 2), (1, 2))[0].data * g.data).sum()
        self.gradient_close(dx.data, numerical_grad(loss, x, eps=self.eps), rtol=2e-3, atol=5e-4)


    def test_ndim_elementwise_reshape_and_fanout(self):
        x = Tensor(self.sample((2, 2, 2, 3)))
        original = x.buf.data.copy()
        h = (x * x).relu().tanh()
        flat = h.flatten()
        self.assertEqual(flat.buf.shape, (2, 12))
        self.assert_storage(flat.buf)
        flat.mean().backward()
        expected = ca.CamelArray([2*v*(1-math.tanh(v*v)**2)/x.buf.size for v in original.tolist()]).reshape(original.shape)
        self.close(x.grad.data, expected)
        reshaped = x.reshape(2, -1, 3)
        self.assertEqual(reshaped.buf.shape, (2, 4, 3))
        reshaped.buf.data[0, 0, 0] = 999
        self.close(x.buf.data, original) # fresh-output ownership, including reshape
        Ops.set_backend("naive")
        self.close(h.buf.data, ca.CamelArray([math.tanh(v*v) for v in original.tolist()]).reshape(original.shape))
        # CPU elementwise operations also preserve all axes after the Vbuf change.
        cpu_x = Tensor(Vbuf(original.copy()))
        (cpu_x * cpu_x).mean().backward()
        self.close(cpu_x.grad.data, original * (2 / cpu_x.buf.size))


    def test_cnn_graph_gradients(self):
        # 2 tiny 4x4 images -> conv -> ReLU -> pool -> flatten -> linear -> xent.
        x = Tensor(Vbuf(ca.CamelArray([0.1 + i*0.07 for i in range(16)] + [1.2-i*0.05 for i in range(16)]).reshape(2, 1, 4, 4)))
        conv, pool, flatten, linear = Conv2d(1, 2, 2), MaxPool2d(2, 1), Flatten(), Linear(8, 2)
        conv.W.buf.data[:] = ca.CamelArray([0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1]).reshape(2, 1, 2, 2)
        y = Tensor(Vbuf(ca.CamelArray([[1, 0], [0, 1]])), requires_grad=False)
        def build():
            return linear(flatten(pool(conv(x).relu()))).softmax_xent(y)
        build().backward()
        variables = [x, conv.W, conv.b, linear.W, linear.b]
        for variable in variables:
            numerical = numerical_grad(lambda: build().buf.data.item(), variable.buf, eps=self.eps)
            self.gradient_close(variable.grad.data, numerical, rtol=5e-3, atol=4e-4)
        self.assertIsNone(y.grad)
        # One optimizer update is enough to check end-to-end descent.
        before = build().buf.data.item()
        SGD([conv.W, conv.b, linear.W, linear.b], lr=0.01).step()
        self.assertLess(build().buf.data.item(), before)


    def test_modules_bias_and_parameters(self):
        class Net(Module):
            def __init__(self):
                self.conv, self.pool, self.flatten, self.linear = Conv2d(1, 2, 2), MaxPool2d(), Flatten(), Linear(2, 1)
        net = Net()
        self.assertEqual(list(net.parameters()), [net.conv.W, net.conv.b, net.linear.W, net.linear.b])
        no_bias = Conv2d(1, 1, 1, bias=False)
        self.assertEqual(list(no_bias.parameters()), [no_bias.W])
        x = Tensor(self.sample((1, 1, 2, 2)), requires_grad=False)
        no_bias(x).mean().backward()
        self.assertIsNotNone(no_bias.W.grad)
        self.assertIsNone(x.grad)
        self.assertEqual(list(MaxPool2d().parameters()), [])
        self.assertEqual(list(Flatten().parameters()), [])


    def test_cnn_model_gradients(self):
        # model class: conv -> tanh -> pool -> flatten -> linear, all parameter grads
        model = CNN(1, [2], 2, image_size=4, activations=[Tanh])
        x = Tensor(Vbuf(ca.CamelArray([0.1 + i * 0.03 for i in range(16)]).reshape(1, 1, 4, 4)))
        model.layers[0].W.buf.data[:] = ca.ones((2, 1, 3, 3)) * 0.1
        y = Tensor(Vbuf(ca.CamelArray([[1, 0]])), requires_grad=False)

        def build():
            return model(x).softmax_xent(y)

        build().backward()
        parameters = list(model.parameters())
        self.assertEqual(parameters, [model.layers[0].W, model.layers[0].b, model.head.W, model.head.b])
        for variable in [x, *parameters]:
            numerical = numerical_grad(lambda: build().buf.data.item(), variable.buf, eps=self.eps)
            self.gradient_close(variable.grad.data, numerical, rtol=5e-3, atol=4e-4)

    def test_cnn_model_shapes(self):
        # multiple conv blocks, multiple input channels, odd rectangular images
        model = CNN(2, [2, 3], 2, image_size=(5, 7))
        self.assertEqual(model.head.W.buf.shape, (3, 2))
        x = Tensor(self.sample((2, 2, 5, 7)), requires_grad=False)
        out = model(x)
        self.assertEqual(out.buf.shape, (2, 2))
        out.mean().backward()
        parameters = list(model.parameters())
        self.assertEqual(len(parameters), 6)
        for parameter in parameters:
            self.assertEqual(parameter.grad.shape, parameter.buf.shape)
        self.assertIsNone(x.grad)

    def test_cnn_model_validation(self):
        for call in (
            lambda: CNN(1, [], 2, 4),
            lambda: CNN(0, [2], 2, 4),
            lambda: CNN(1, [0], 2, 4),
            lambda: CNN(1, [2], 0, 4),
            lambda: CNN(1, [2], 2, 0),
            lambda: CNN(1, [2, 2], 2, 3),
            lambda: CNN(1, [2], 2, (4, 1)),
            lambda: CNN(1, [2], 2, 4, activations=[]),
        ):
            with self.assertRaises(ValueError):
                call()
        model = CNN(1, [2], 2, 4)
        for shape in ((1, 16), (1, 2, 4, 4), (1, 1, 4, 5)):
            with self.assertRaises(ValueError):
                model(Tensor(self.sample(shape)))

    def test_forward_options_are_saved_by_value(self):
        x, w = Tensor(self.sample((1, 1, 3, 3))), Tensor(self.sample((1, 1, 2, 2)))
        stride, padding = [1, 1], [0, 0]
        out = x.conv2d(w, stride=stride, padding=padding)
        stride[0], padding[1] = 2, 1
        out.mean().backward()
        g = ca.ones(out.buf.shape) / out.buf.size
        _, dx, dw, _ = ref.conv2d(x.buf.data, w.buf.data, grad=g)
        self.close(x.grad.data, dx)
        self.close(w.grad.data, dw)


    def test_validation_and_backend_guards(self):
        x, w = self.sample((1, 1, 3, 3)), self.sample((1, 1, 2, 2))
        bad_calls = [
            lambda: Ops.conv2d_forward(self.sample((3, 3)), w),
            lambda: Ops.conv2d_forward(x, self.sample((1, 2, 2, 2))),
            lambda: Ops.conv2d_forward(x, w, self.sample((2, 1))),
            lambda: Ops.conv2d_forward(x, w, stride=0),
            lambda: Ops.conv2d_forward(x, w, stride=True),
            lambda: Ops.conv2d_forward(x, w, padding=-1),
            lambda: Ops.conv2d_forward(x, w, padding=2147483647),
            lambda: Ops.conv2d_forward(x, w, stride=(1,)),
            lambda: Ops.conv2d_forward(x, w, stride=(1, 1.5)),
            lambda: Ops.conv2d_forward(x, self.sample((1, 1, 4, 4))),
            lambda: Ops.conv2d_backward(x, w, self.sample((1, 1, 1, 1))),
            lambda: Ops.maxpool2d_forward(x, 4),
            lambda: Ops.maxpool2d_forward(x, 2, -1),
            lambda: Ops.maxpool2d_forward(x, 0),
            lambda: Ops.reshape(x, (2, -1)),
            lambda: Ops.reshape(x, (-1, -1)),
            lambda: Ops.reshape(x, (0, 9)),
            lambda: Ops.matmul_forward(x, w),
            lambda: Ops.softmax_xent_forward(x, x),
            lambda: Conv2d(0, 2, 3),
            lambda: Conv2d(1, 2, (2, -1)),
            lambda: Conv2d(1, 2147483648, 1),
        ]
        for call in bad_calls:
            with self.assertRaises(ValueError):
                call()
        out, cache = Ops.maxpool2d_forward(x)
        with self.assertRaises(ValueError):
            Ops.maxpool2d_backward(Vbuf(ca.ones(x.shape)), cache)
        Ops.backend = "naive" if self.backend == "cuda" else "cuda"
        with self.assertRaises(ValueError):
            Ops.maxpool2d_backward(Vbuf(ca.ones(out.shape)), cache)
        Ops.backend = self.backend
        cache.close()
        with self.assertRaises(ValueError):
            Ops.maxpool2d_backward(Vbuf(ca.ones(out.shape)), cache)
        Ops.backend = "simd"
        for call in (lambda: Ops.conv2d_forward(x, w), lambda: Ops.maxpool2d_forward(x)):
            with self.assertRaises(NotImplementedError):
                call()
