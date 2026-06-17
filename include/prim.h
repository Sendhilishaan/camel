#ifndef PRIM_H
#define PRIM_H

#define AT(M, i, j, cols) ((M)[(i) * (cols) + (j)])

void matmul_forward(const double* A, const double* B, double* out, int n, int k, int m);

void matmul_backward(const double* A, const double* B, const double* grad_out, double* out, int n, int k, int m);

void matadd_broadcast_forward(const double* A, const double* B, int n, int m);

void matadd_broadcast_backward(const double* grad_out, double* dX, double* db, int n, int m);

void tanh_forward(const double* Z, double* out, int n, int m);

void tanh_backward(const double* out, const double* grad_out, double* dz, int n, int m);

#endif