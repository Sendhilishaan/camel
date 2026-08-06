#ifndef PRIM_SIMD_H
#define PRIM_SIMD_H

#include "prim.h"

/*
    Apple SIMD (<simd/simd.h>) kernels: one-for-one with prim.h, `_simd` suffix,
    vectorised two doubles at a time. Darwin-only build (see Makefile); prim.h
    stays the portable fallback everywhere else.
*/

EXPORT void matmul_forward_simd(const double* A, const double* B, double* out, int n, int k, int m);

EXPORT void matmul_backward_simd(const double* A, const double* B, const double* grad_out, double* da, double* db, int n, int k, int m);

EXPORT void matadd_broadcast_forward_simd(double* A, const double* B, int n, int m);

EXPORT void matadd_broadcast_backward_simd(const double* grad_out, double* dX, double* db, int n, int m);

EXPORT void matsub_forward_simd(const double* A, const double* B, double* out, int n, int m);

EXPORT void matsub_backward_simd(const double* grad_out, double* dA, double* dB, int n, int m);

EXPORT void hadamard_forward_simd(const double* A, const double* B, double* out, int n, int m);

EXPORT void hadamard_backward_simd(const double* grad_out, const double* A, const double* B, double* dA, double* dB, int n, int m);

EXPORT void matmean_forward_simd(const double* A, double* out, int n);

EXPORT void matmean_backward_simd(double* dx, int n, double grad_out);

EXPORT void tanh_forward_simd(const double* Z, double* out, int n, int m);

EXPORT void tanh_backward_simd(const double* out, const double* grad_out, double* dz, int n, int m);

EXPORT void relu_forward_simd(const double* Z, double* out, int n, int m);

EXPORT void relu_backward_simd(const double* out, const double* grad_out, double* dz, int n, int m);

EXPORT void softmax_xent_forward_simd(const double* Z, const double* Y, double* probs, double* out_loss, int n, int m);

EXPORT void softmax_xent_backward_simd(const double* probs, const double* Y, double* dZ, double grad_out, int n, int m);

#endif
