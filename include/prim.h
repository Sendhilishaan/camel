#ifndef PRIM_H
#define PRIM_H

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

#define AT(M, i, j, cols) ((M)[(i) * (cols) + (j)])

EXPORT void matmul_forward(const double* A, const double* B, double* out, int n, int k, int m);

EXPORT void matmul_backward(const double* A, const double* B, const double* grad_out, double* da, double* db, int n, int k, int m);

EXPORT void matadd_broadcast_forward(double* A, const double* B, int n, int m);

EXPORT void matadd_broadcast_backward(const double* grad_out, double* dX, double* db, int n, int m);

EXPORT void matsub_forward(const double* A, const double* B, double* out, int n, int m);

EXPORT void matsub_backward(const double* grad_out, double* dA, double* dB, int n, int m);

EXPORT void hadamard_forward(const double* A, const double* B, double* out, int n, int m);

EXPORT void hadamard_backward(const double* grad_out, const double* A, const double* B, double* dA, double* dB, int n, int m);

EXPORT void matmean_forward(const double* A, double* out, int n);

EXPORT void matmean_backward(double* dx, int n, double grad_out);

EXPORT void tanh_forward(const double* Z, double* out, int n, int m);

EXPORT void tanh_backward(const double* out, const double* grad_out, double* dz, int n, int m);

#endif