"""CUDA CNN correctness checks on tiny synthetic tensors, no heavy workloads.

No downloads, benchmarks, large images or dataset training. Every tensor in
the tests has fewer than 512 elements. References use Python float64; CUDA is
float32, so central differences use eps=1e-3 with explicit error tolerances.
"""
import argparse
import unittest
from ctypes import c_float, c_int, c_void_p, byref

from camel import array as ca
from camel._c import c, CNN_CUDA_AVAILABLE
from camel.ops import Ops, Vbuf
from camel.tensor import Tensor
from camel.optim import SGD, AdaGrad, RMSprop, Adam


from tests.cnn_checks import CnnChecks


class CnnCudaChecks(CnnChecks, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CNN_CUDA_AVAILABLE or not Ops.cuda_device_available():
            raise unittest.SkipTest("CUDA CNN library and a working NVIDIA GPU are required")


    def test_optimisers_on_4d_weights(self):
        initial = ca.CamelArray([0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).reshape(2, 1, 2, 2)
        gradient = ca.CamelArray([0.3, -0.4, 0.1, 0.2, -0.5, 0.6, 0.7, 0.8]).reshape(initial.shape)
        for factory in (lambda p: SGD(p, 0.02, 0.9), lambda p: AdaGrad(p, 0.02),
                        lambda p: RMSprop(p, 0.02), lambda p: Adam(p, 0.02)):
            results = []
            for backend in ("naive", "cuda"):
                Ops.set_backend(backend)
                p = Tensor(Vbuf(initial.copy()))
                opt = factory([p])
                state = [v for attr in ("velocity", "grad_sum", "ema_sq", "exp_avg", "exp_avg_sq") for v in getattr(opt, attr, [])]
                handles = None
                for _ in range(3):
                    p.grad = Vbuf(gradient.copy())
                    opt.step()
                    if backend == "cuda":
                        self.assertTrue(all(v._data is None for v in [p.buf] + state))
                        current = [v.gpu_handle() for v in [p.buf] + state]
                        if handles is not None:
                            self.assertEqual(current, handles)
                        handles = current
                results.append([v.data.copy() for v in [p.buf] + state])
            for cpu, gpu in zip(*results):
                self.close(gpu, cpu)


    def test_native_error_boundaries(self):
        x, w = self.sample((1, 1, 3, 3)), self.sample((1, 1, 2, 2))
        dims = (1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0)
        with self.assertRaises(RuntimeError):
            c.conv2d_forward_cuda_resident(None, w.gpu_handle(), None, *dims)
        for replacement in ((0, 0), (7, 0), (9, -1), (9, 2147483647), (0, 2147483647)):
            args = list(dims)
            args[replacement[0]] = replacement[1]
            with self.assertRaises(RuntimeError):
                c.conv2d_forward_cuda_resident(x.gpu_handle(), w.gpu_handle(), None, *args)
        dx, dw, db = c_void_p(), c_void_p(), c_void_p()
        with self.assertRaises(RuntimeError):
            c.conv2d_backward_cuda_resident(x.gpu_handle(), w.gpu_handle(), None, *dims, byref(dx), byref(dw), byref(db))
        self.assertFalse(dx.value or dw.value or db.value)
        with self.assertRaises(RuntimeError):
            c.maxpool2d_backward_cuda_resident(x.gpu_handle(), None)
        with self.assertRaises(RuntimeError):
            c.maxpool2d_backward_cuda((c_float * 1)(1), (c_int * 1)(99), (c_float * 9)(), 1, 1, 3, 3, 2, 2, 2, 2)
        # Valid calls still work after errors, rather than retaining stale status.
        self.assertEqual(Ops.conv2d_forward(x, w).shape, (1, 1, 2, 2))
        Ops.synchronize()



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-device", action="store_true")
    args = parser.parse_args()
    if args.require_device and (not CNN_CUDA_AVAILABLE or not Ops.cuda_device_available()):
        raise RuntimeError("CUDA CNN tests require a rebuilt library and a working NVIDIA GPU")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CnnCudaChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
