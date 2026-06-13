#include "prim.h"
/*
    forward and backward primitive kernels, naive implementations first

    primitives:
        - matmul
        - + b with broadcast
        - tanh (only activation for now)
        - sub 
        - hadamard
        - mean()
*/

/*
for matmul Z = X @ W 

(X: n, k - W: K, M - Z: n, m - G = n, m)

given G = dL/dZ

dL/dX: (n, k) = G @ Wt
dL/dW: (k, m) = Xt @ G


assume out is zerod
*/
void matmul_forward(const double* A, const double* B, double* out, int n, int k, int m) {
    // A @ B
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int l = 0; l < k; l++) {
                AT(out, i, j, m) += AT(A, i, l, k) * AT(B, l, j, m);
            }
        }
    }
}

void matmul_backward(const double* A, const double* B, const double* grad_out, double* da, double* db, int n, int k, int m) {
    // dl/da
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < m; l++) {
                AT(da, i, j, k) += AT(grad_out, i, l, m) * AT(B, j, l, m); // implicit transpose
            }
        }
    }

    // dl/db
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < m; j++) {
            for (int l = 0; l < n; l++) {
                AT(db, i, j, m) += AT(A, l, i, k) * AT(grad_out, l, j, m); 
            }
        }
    }
}