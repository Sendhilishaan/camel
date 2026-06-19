#ifndef PRIM_H
#define PRIM_H

#define AT(M, i, j, cols) ((M)[(i) * (cols) + (j)])

void matmul_forward(const double* A, const double* B, double* out, int n, int k, int m);

void matmul_backward(const double* A, const double* B, const double* grad_out, double* da, double* db, int n, int k, int m);

void matadd_broadcast_forward(double* A, const double* B, int n, int m);

void matadd_broadcast_backward(const double* grad_out, double* dX, double* db, int n, int m);

void matsub_forward(const double* A, const double* B, double* out, int n, int m);

void matsub_backward(const double* grad_out, double* dA, double* dB, int n, int m);

void hadamard_forward(const double* A, const double* B, double* out, int n, int m);

void hadamard_backward(const double* grad_out, const double* A, const double* B, double* dA, double* dB, int n, int m);

void matmean_forward(const double* A, double* out, int n);

void matmean_backward(double* dx, int n, double grad_out);

void tanh_forward(const double* Z, double* out, int n, int m);

void tanh_backward(const double* out, const double* grad_out, double* dz, int n, int m);

#endif