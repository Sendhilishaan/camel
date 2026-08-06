#include "prim_simd.h"
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#include <simd/simd.h>

/*
    simd_double2 = one NEON register (128-bit) on Apple Silicon, the natural lane width here.
    ctypes buffers are only 8-byte aligned, so loads/stores go through simd_packed_double2.
*/

static inline simd_double2 pload2(const double* p) {
    return *(const simd_packed_double2*)p;
}

static inline void pstore2(double* p, simd_double2 v) {
    *(simd_packed_double2*)p = v;
}

// dot product of two contiguous length-m rows
static inline double row_dot(const double* a, const double* b, int m) {
    simd_double2 acc = { 0.0, 0.0 };
    int j = 0;
    for (; j + 2 <= m; j += 2) {
        acc += pload2(a + j) * pload2(b + j);
    }
    double sum = simd_reduce_add(acc);
    for (; j < m; j++) {
        sum += a[j] * b[j];
    }
    return sum;
}

/*
    (i, l, j) loop order instead of naive's (i, j, l), so the inner loop stays
    over contiguous memory (out row, B row) and vectorises. out assumed zeroed.
*/
EXPORT void matmul_forward_simd(const double* A, const double* B, double* out, int n, int k, int m) {
    for (int i = 0; i < n; i++) {
        double* out_row = out + i * m;
        for (int l = 0; l < k; l++) {
            double a = AT(A, i, l, k);
            const double* b_row = B + l * m;
            int j = 0;
            for (; j + 2 <= m; j += 2) {
                pstore2(out_row + j, pload2(out_row + j) + pload2(b_row + j) * a);
            }
            for (; j < m; j++) {
                out_row[j] += a * b_row[j];
            }
        }
    }
}

EXPORT void matmul_backward_simd(const double* A, const double* B, const double* grad_out, double* da, double* db, int n, int k, int m) {
    // dl/da[i,j] = dot(grad_out[i,:], B[j,:]) - both rows contiguous, length m
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            AT(da, i, j, k) = row_dot(grad_out + i * m, B + j * m, m);
        }
    }

    // dl/db = A^T @ grad_out, same reorder trick as matmul_forward
    for (int l = 0; l < n; l++) {
        for (int i = 0; i < k; i++) {
            double a = AT(A, l, i, k); // implicit transpose, strided scalar load
            const double* g_row = grad_out + l * m;
            double* db_row = db + i * m;
            int j = 0;
            for (; j + 2 <= m; j += 2) {
                pstore2(db_row + j, pload2(db_row + j) + pload2(g_row + j) * a);
            }
            for (; j < m; j++) {
                db_row[j] += a * g_row[j];
            }
        }
    }
}

EXPORT void matadd_broadcast_forward_simd(double* A, const double* B, int n, int m) {
    for (int i = 0; i < n; i++) {
        double* row = A + i * m;
        int j = 0;
        for (; j + 2 <= m; j += 2) {
            pstore2(row + j, pload2(row + j) + pload2(B + j));
        }
        for (; j < m; j++) {
            row[j] += B[j];
        }
    }
}

EXPORT void matadd_broadcast_backward_simd(const double* grad_out, double* dX, double* db, int n, int m) {
    memcpy(dX, grad_out, sizeof(double) * n * m); // add is identity, same as naive

    // db: column-wise sum down the rows; db pre-zeroed by caller
    for (int i = 0; i < n; i++) {
        const double* row = grad_out + i * m;
        int j = 0;
        for (; j + 2 <= m; j += 2) {
            pstore2(db + j, pload2(db + j) + pload2(row + j));
        }
        for (; j < m; j++) {
            db[j] += row[j];
        }
    }
}

// A - B: (n, m) is one contiguous run of n*m doubles
EXPORT void matsub_forward_simd(const double* A, const double* B, double* out, int n, int m) {
    int total = n * m, idx = 0;
    for (; idx + 2 <= total; idx += 2) {
        pstore2(out + idx, pload2(A + idx) - pload2(B + idx));
    }
    for (; idx < total; idx++) {
        out[idx] = A[idx] - B[idx];
    }
}

EXPORT void matsub_backward_simd(const double* grad_out, double* dA, double* dB, int n, int m) {
    int total = n * m;
    memcpy(dA, grad_out, sizeof(double) * total);

    int idx = 0;
    for (; idx + 2 <= total; idx += 2) {
        pstore2(dB + idx, -pload2(grad_out + idx));
    }
    for (; idx < total; idx++) {
        dB[idx] = -grad_out[idx];
    }
}

EXPORT void hadamard_forward_simd(const double* A, const double* B, double* out, int n, int m) {
    int total = n * m, idx = 0;
    for (; idx + 2 <= total; idx += 2) {
        pstore2(out + idx, pload2(A + idx) * pload2(B + idx));
    }
    for (; idx < total; idx++) {
        out[idx] = A[idx] * B[idx];
    }
}

EXPORT void hadamard_backward_simd(const double* grad_out, const double* A, const double* B, double* dA, double* dB, int n, int m) {
    int total = n * m, idx = 0;
    for (; idx + 2 <= total; idx += 2) {
        simd_double2 g = pload2(grad_out + idx);
        pstore2(dA + idx, g * pload2(B + idx));
        pstore2(dB + idx, g * pload2(A + idx));
    }
    for (; idx < total; idx++) {
        dA[idx] = grad_out[idx] * B[idx];
        dB[idx] = grad_out[idx] * A[idx];
    }
}

// mean, n = total elements (as in prim.c, not the row count)
EXPORT void matmean_forward_simd(const double* A, double* out, int n) {
    simd_double2 acc = { 0.0, 0.0 };
    int i = 0;
    for (; i + 2 <= n; i += 2) {
        acc += pload2(A + i);
    }
    double sum = simd_reduce_add(acc);
    for (; i < n; i++) {
        sum += A[i];
    }
    *out = sum / n;
}

EXPORT void matmean_backward_simd(double* dx, int n, double grad_out) {
    double g = grad_out / n;
    simd_double2 gvec = { g, g };
    int i = 0;
    for (; i + 2 <= n; i += 2) {
        pstore2(dx + i, gvec);
    }
    for (; i < n; i++) {
        dx[i] = g;
    }
}

// tanh(x) dispatches to the simd_double2 overload via <tgmath.h> (wired up by
// <simd/simd.h>), same call site handles the scalar tail below.
EXPORT void tanh_forward_simd(const double* Z, double* out, int n, int m) {
    int total = n * m, idx = 0;
    for (; idx + 2 <= total; idx += 2) {
        pstore2(out + idx, tanh(pload2(Z + idx)));
    }
    for (; idx < total; idx++) {
        out[idx] = tanh(Z[idx]);
    }
}

EXPORT void tanh_backward_simd(const double* out, const double* grad_out, double* dz, int n, int m) {
    simd_double2 one = { 1.0, 1.0 };
    int total = n * m, idx = 0;
    for (; idx + 2 <= total; idx += 2) {
        simd_double2 o = pload2(out + idx);
        pstore2(dz + idx, pload2(grad_out + idx) * (one - o * o));
    }
    for (; idx < total; idx++) {
        dz[idx] = grad_out[idx] * (1 - out[idx] * out[idx]);
    }
}

EXPORT void relu_forward_simd(const double* Z, double* out, int n, int m) {
    simd_double2 zero = { 0.0, 0.0 };
    int total = n * m, idx = 0;
    for (; idx + 2 <= total; idx += 2) {
        pstore2(out + idx, simd_max(pload2(Z + idx), zero));
    }
    for (; idx < total; idx++) {
        out[idx] = (Z[idx] > 0) ? Z[idx] : 0;
    }
}

EXPORT void relu_backward_simd(const double* out, const double* grad_out, double* dz, int n, int m) {
    simd_double2 zero = { 0.0, 0.0 };
    int total = n * m, idx = 0;
    for (; idx + 2 <= total; idx += 2) {
        // mask is all-1s per lane where out>0; simd_select picks grad_out there, else zero
        simd_long2 mask = pload2(out + idx) > zero;
        pstore2(dz + idx, simd_select(zero, pload2(grad_out + idx), mask));
    }
    for (; idx < total; idx++) {
        dz[idx] = (out[idx] > 0) ? grad_out[idx] : 0;
    }
}

// fused softmax + cross-entropy: same three passes as the naive kernel
// (row max, exp+sum+true-logit, normalise), each vectorised over the row.
EXPORT void softmax_xent_forward_simd(const double* Z, const double* Y, double* probs, double* out_loss, int n, int m) {
    double total_loss = 0;
    for (int i = 0; i < n; i++) {
        const double* z_row = Z + i * m;
        const double* y_row = Y + i * m;
        double* p_row = probs + i * m;

        // pass 1: row max
        simd_double2 maxvec = { -INFINITY, -INFINITY };
        int j = 0;
        for (; j + 2 <= m; j += 2) {
            maxvec = simd_max(maxvec, pload2(z_row + j));
        }
        double curr_max = simd_reduce_max(maxvec);
        for (; j < m; j++) {
            if (z_row[j] > curr_max) curr_max = z_row[j];
        }

        // pass 2: exp(z - max) -> probs, plus running sum and the one-hot true-class logit
        simd_double2 maxbcast = { curr_max, curr_max };
        simd_double2 sumvec = { 0.0, 0.0 };
        simd_double2 truevec = { 0.0, 0.0 };
        j = 0;
        for (; j + 2 <= m; j += 2) {
            simd_double2 zvec = pload2(z_row + j);
            simd_double2 e = exp(zvec - maxbcast);
            pstore2(p_row + j, e);
            sumvec += e;
            truevec += zvec * pload2(y_row + j);
        }
        double sum = simd_reduce_add(sumvec);
        double true_logit = simd_reduce_add(truevec);
        for (; j < m; j++) {
            double e = exp(z_row[j] - curr_max);
            p_row[j] = e;
            sum += e;
            true_logit += z_row[j] * y_row[j];
        }

        total_loss += (-(true_logit - curr_max) + log(sum));

        // pass 3: normalise
        simd_double2 sumbcast = { sum, sum };
        j = 0;
        for (; j + 2 <= m; j += 2) {
            pstore2(p_row + j, pload2(p_row + j) / sumbcast);
        }
        for (; j < m; j++) {
            p_row[j] /= sum;
        }
    }

    *out_loss = total_loss / n;
}

EXPORT void softmax_xent_backward_simd(const double* probs, const double* Y, double* dZ, double grad_out, int n, int m) {
    double scale = grad_out / n;
    simd_double2 scalevec = { scale, scale };
    int total = n * m, idx = 0;
    for (; idx + 2 <= total; idx += 2) {
        pstore2(dZ + idx, (pload2(probs + idx) - pload2(Y + idx)) * scalevec);
    }
    for (; idx < total; idx++) {
        dZ[idx] = scale * (probs[idx] - Y[idx]);
    }
}

#endif /* __APPLE__ */
