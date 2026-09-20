"""Backend auto-selection and explicit override checks."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from camel import ops
from camel.ops import Ops


class BackendSelectionChecks(unittest.TestCase):
    def automatic_backend(self, *, cuda=False, cuda_device=False,
                          metal=False, metal_device=False, simd=False):
        probes = SimpleNamespace(
            camel_cuda_device_available=lambda: cuda_device,
            camel_metal_device_available=lambda: metal_device,
        )
        with (
            patch.object(ops, "CUDA_AVAILABLE", cuda),
            patch.object(ops, "METAL_AVAILABLE", metal),
            patch.object(ops, "SIMD_AVAILABLE", simd),
            patch.object(ops, "c", probes),
        ):
            return ops._automatic_backend()

    def test_prefers_cuda_over_other_available_backends(self):
        self.assertEqual(
            self.automatic_backend(
                cuda=True, cuda_device=True,
                metal=True, metal_device=True,
                simd=True,
            ),
            "cuda",
        )

    def test_uses_metal_when_cuda_has_no_working_device(self):
        self.assertEqual(
            self.automatic_backend(
                cuda=True, cuda_device=False,
                metal=True, metal_device=True,
                simd=True,
            ),
            "metal",
        )

    def test_falls_back_to_simd_cpu(self):
        self.assertEqual(
            self.automatic_backend(
                cuda=True, cuda_device=False,
                metal=True, metal_device=False,
                simd=True,
            ),
            "simd",
        )

    def test_falls_back_to_naive_cpu(self):
        self.assertEqual(self.automatic_backend(), "naive")

    def test_set_backend_pins_explicit_choice(self):
        original = Ops.backend
        try:
            Ops.set_backend("naive")
            self.assertEqual(Ops.backend, "naive")
        finally:
            Ops.set_backend(original)


if __name__ == "__main__":
    unittest.main()
