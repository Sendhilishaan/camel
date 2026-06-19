#include "prim.h"
#include <string.h>
#include <math.h>
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
EXPORT void matmul_forward(const double* A, const double* B, double* out, int n, int k, int m) {
    // A @ B
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int l = 0; l < k; l++) {
                AT(out, i, j, m) += AT(A, i, l, k) * AT(B, l, j, m);
            }
        }
    }
}

EXPORT void matmul_backward(const double* A, const double* B, const double* grad_out, double* da, double* db, int n, int k, int m) {
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

/*
add with broadcast down cols, row is one sample in batch

A: (n, m) B: (1, m)
*/
EXPORT void matadd_broadcast_forward(double* A, const double* B, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            AT(A, i, j, m) += AT(B, 0, j, m);
        }
    }
}

/*
db: (1, m) cw sum should be 0 init
*/
EXPORT void matadd_broadcast_backward(const double* grad_out, double* dX, double* db, int n, int m) {
    // copy becayuse add is identity
    memcpy(dX, grad_out, sizeof(double) * n * m);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            AT(db, 0, j, m) += AT(grad_out, i, j, m);
        }
    }
}

// elementwise sub A - B

EXPORT void matsub_forward(const double* A, const double* B, double* out, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            AT(out, i, j, m) = AT(A, i, j, m) - AT(B, i, j, m);
        }
    }
}

EXPORT void matsub_backward(const double* grad_out, double* dA, double* dB, int n, int m) {
    memcpy(dA, grad_out, sizeof(double) * n * m);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            AT(dB, i, j, m) = -(AT(grad_out, i, j, m));
        }
    }
}

/* 
elementwise multiply

backward needs cached inputs from the forward
*/

EXPORT void hadamard_forward(const double* A, const double* B, double* out, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            AT(out, i, j, m) = AT(A, i, j, m) * AT(B, i, j, m);
        }
    }
}

EXPORT void hadamard_backward(const double* grad_out, const double* A, const double* B, double* dA, double* dB, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            AT(dA, i, j, m) = AT(grad_out, i, j, m) * AT(B, i, j, m);
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            AT(dB, i, j, m) = AT(grad_out, i, j, m) * AT(A, i, j, m);
        }
    }

}

// mean, n = total elements

EXPORT void matmean_forward(const double* A, double* out, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += A[i];
    }

    *out = sum / n;
}

EXPORT void matmean_backward(double* dx, int n, double grad_out) {
    for (int i = 0; i < n; i++) {
        dx[i] = grad_out / n;
    }
}

/* 
tanh activation backward takes forward output buffer as a input
*/

EXPORT void tanh_forward(const double* Z, double* out, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            AT(out, i, j, m) = tanh(AT(Z, i, j, m));
        }
    }
}

EXPORT void tanh_backward(const double* out, const double* grad_out, double* dz, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            AT(dz, i, j, m) = AT(grad_out, i, j, m) * (1 - (AT(out, i, j, m) * AT(out, i, j, m)));
        }
    }
}