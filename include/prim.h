#ifndef PRIM_H
#define PRIM_H

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

#define AT(M, i, j, cols) ((M)[(i) * (cols) + (j)])

#ifdef __cplusplus
extern "C" {
#endif

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

EXPORT void relu_forward(const double* Z, double* out, int n, int m);

EXPORT void relu_backward(const double* out, const double* grad_out, double* dz, int n, int m);

EXPORT void softmax_xent_forward(const double* Z, const double* Y, double* probs, double* out_loss, int n, int m);

EXPORT void softmax_xent_backward(const double* probs, const double* Y, double* dZ, double grad_out, int n, int m);

// Contiguous NCHW images, OIHW filters, optional (1,F) bias. Outputs are overwritten.
// Callers allocate distinct output buffers of the computed sizes.
EXPORT const char* camel_naive_get_last_error(void);
EXPORT void conv2d_forward(const double* x, const double* weight, const double* bias, double* out,
    int n, int c, int h, int w, int f, int kh, int kw, int sh, int sw, int ph, int pw);
EXPORT void conv2d_backward(const double* x, const double* weight, const double* grad,
    double* dx, double* dw, double* db,
    int n, int c, int h, int w, int f, int kh, int kw, int sh, int sw, int ph, int pw);
EXPORT void maxpool2d_forward(const double* x, double* out, int* indices,
    int n, int c, int h, int w, int kh, int kw, int sh, int sw);
EXPORT void maxpool2d_backward(const double* grad, const int* indices, double* dx,
    int n, int c, int h, int w, int kh, int kw, int sh, int sw);

#ifdef __cplusplus
}
#endif

#endif
