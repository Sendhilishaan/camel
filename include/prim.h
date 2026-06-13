#ifndef PRIM_H
#define PRIM_H

#define AT(M, i, j, cols) ((M)[(i) * (cols) + (j)])

void matmul_forward(const double* A, const double* B, double* out, int n, int k, int m);

void matmul_backward(const double* A, const double* B, const double* grad_out, double* out, int n, int k, int m);

#endif