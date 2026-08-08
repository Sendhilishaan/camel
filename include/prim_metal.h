#ifndef PRIM_METAL_H
#define PRIM_METAL_H

#include "prim.h"

/*
    Metal (GPU) kernels, one-for-one with prim.h, `_metal` suffix.

    MSL has no double type, so these operate on float32 - Vbuf.to_float32()/
    from_float32() handle the conversion at the Python boundary. Darwin-only
    build (see Makefile).
*/

EXPORT void matmul_forward_metal(const float* A, const float* B, float* out, int n, int k, int m);

EXPORT void matmul_backward_metal(const float* A, const float* B, const float* grad_out, float* da, float* db, int n, int k, int m);

EXPORT void matadd_broadcast_forward_metal(float* A, const float* B, int n, int m);

EXPORT void matadd_broadcast_backward_metal(const float* grad_out, float* dX, float* db, int n, int m);

EXPORT void matsub_forward_metal(const float* A, const float* B, float* out, int n, int m);

EXPORT void matsub_backward_metal(const float* grad_out, float* dA, float* dB, int n, int m);

EXPORT void hadamard_forward_metal(const float* A, const float* B, float* out, int n, int m);

EXPORT void hadamard_backward_metal(const float* grad_out, const float* A, const float* B, float* dA, float* dB, int n, int m);

EXPORT void matmean_forward_metal(const float* A, float* out, int n);

EXPORT void matmean_backward_metal(float* dx, int n, float grad_out);

EXPORT void tanh_forward_metal(const float* Z, float* out, int n, int m);

EXPORT void tanh_backward_metal(const float* out, const float* grad_out, float* dz, int n, int m);

EXPORT void relu_forward_metal(const float* Z, float* out, int n, int m);

EXPORT void relu_backward_metal(const float* out, const float* grad_out, float* dz, int n, int m);

EXPORT void softmax_xent_forward_metal(const float* Z, const float* Y, float* probs, float* out_loss, int n, int m);

EXPORT void softmax_xent_backward_metal(const float* probs, const float* Y, float* dZ, float grad_out, int n, int m);

// probes MTLCreateSystemDefaultDevice() without aborting on failure, so callers
// can check for a working GPU before running a kernel (which does abort on failure)
EXPORT int camel_metal_device_available(void);

#endif
