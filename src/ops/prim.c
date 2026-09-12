#include "prim.h"
#include <string.h>
#include <math.h>
#include <limits.h>
/*
    forward and backward primitive kernels

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

EXPORT void relu_forward(const double* Z, double* out, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            AT(out, i, j, m) = (AT(Z, i, j, m) > 0) ? (AT(Z, i, j, m)) : 0;
        }
    }
}

EXPORT void relu_backward(const double* out, const double* grad_out, double* dz, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            AT(dz, i, j, m) = (AT(out, i, j, m) > 0) ? AT(grad_out, i, j, m) : 0;
        }
    }
}

/*
    standard fused softmax + multi-class cross entropy to avoid underflowing log to 0
*/

EXPORT void softmax_xent_forward(const double* Z, const double* Y, double* probs, double* out_loss, int n, int m) {
    double total_loss = 0;
    for (int i = 0; i < n; i++) {
        double curr_max = -INFINITY;
        for (int j = 0; j < m; j++) {
            if ((AT(Z, i, j, m)) > curr_max) {
                curr_max = (AT(Z, i, j, m ));
            }
        }

        double true_logit = 0;
        // subtract by max (exp <= 0) exponentiate and accum sum
        double sum = 0;
        for (int j = 0; j < m; j++) {
            (AT(probs, i, j, m)) = exp((AT(Z, i, j, m)) - curr_max);
            sum += (AT(probs, i, j, m));

            // true class logit using the one-hot mask; for cross entropy
            true_logit += (AT(Z, i, j, m)) * (AT(Y, i, j, m));
        }

        // add loss for this row
         total_loss += (-(true_logit - curr_max) + log(sum));

        // normalise probs
        for (int j = 0; j < m; j++) {
            AT(probs, i, j, m) = (AT(probs, i, j, m)) / sum;
        }
    }

    *out_loss = total_loss / n;
}

EXPORT void softmax_xent_backward(const double* probs, const double* Y, double* dZ, double grad_out, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            AT(dZ, i, j, m) = grad_out * (((AT(probs, i, j, m)) - (AT(Y, i, j, m))) / n);
        }
    }
}

/* CNN kernels: NCHW images, OIHW filters, float64 throughout.
   Each call overwrites its outputs. Invalid arguments leave outputs untouched. */
#ifdef _WIN32
static __declspec(thread) const char* cnn_error = NULL;
#else
static _Thread_local const char* cnn_error = NULL;
#endif

EXPORT const char* camel_naive_get_last_error(void) {
    return cnn_error;
}

static int cnn_count4(int a, int b, int c, int d) {
    int dims[4] = {a, b, c, d};
    int count = 1;
    for (int i = 0; i < 4; i++) {
        if (dims[i] <= 0 || count > INT_MAX / dims[i]) {
            cnn_error = "CNN tensor dimensions must be positive and fit int32";
            return 0;
        }
        count *= dims[i];
    }
    return count;
}

static int cnn_window(int h, int w, int kh, int kw, int sh, int sw,
                      int ph, int pw, int* oh, int* ow) {
    long long padded_h = (long long)h + 2LL * ph;
    long long padded_w = (long long)w + 2LL * pw;
    if (kh <= 0 || kw <= 0 || sh <= 0 || sw <= 0 || ph < 0 || pw < 0 ||
        padded_h > INT_MAX || padded_w > INT_MAX || padded_h < kh || padded_w < kw) {
        cnn_error = "invalid CNN kernel, stride or padding";
        return 0;
    }
    *oh = (int)((padded_h - kh) / sh + 1);
    *ow = (int)((padded_w - kw) / sw + 1);
    return 1;
}

static int conv2d_dims(int n, int c, int h, int w, int f, int kh, int kw,
                      int sh, int sw, int ph, int pw, int* oh, int* ow) {
    return cnn_count4(n, c, h, w) && cnn_count4(f, c, kh, kw) &&
        cnn_window(h, w, kh, kw, sh, sw, ph, pw, oh, ow) && cnn_count4(n, f, *oh, *ow);
}

EXPORT void conv2d_forward(const double* x, const double* weight, const double* bias, double* out,
    int n, int c, int h, int w, int f, int kh, int kw, int sh, int sw, int ph, int pw) {
    cnn_error = NULL;
    if (!x || !weight || !out) {
        cnn_error = "conv2d requires input, weight and output buffers";
        return;
    }
    int oh, ow;
    if (!conv2d_dims(n, c, h, w, f, kh, kw, sh, sw, ph, pw, &oh, &ow)) return;
    for (int batch = 0; batch < n; batch++) {
        for (int filter = 0; filter < f; filter++) {
            for (int row = 0; row < oh; row++) {
                for (int col = 0; col < ow; col++) {
                    double sum = bias ? bias[filter] : 0.0;
                    for (int channel = 0; channel < c; channel++) {
                        for (int ky = 0; ky < kh; ky++) {
                            int iy = row * sh - ph + ky;
                            if (iy < 0 || iy >= h) continue;
                            for (int kx = 0; kx < kw; kx++) {
                                int ix = col * sw - pw + kx;
                                if (ix < 0 || ix >= w) continue;
                                int xi = ((batch * c + channel) * h + iy) * w + ix;
                                int wi = ((filter * c + channel) * kh + ky) * kw + kx;
                                sum += x[xi] * weight[wi];
                            }
                        }
                    }
                    out[((batch * f + filter) * oh + row) * ow + col] = sum;
                }
            }
        }
    }
}

EXPORT void conv2d_backward(const double* x, const double* weight, const double* grad,
    double* dx, double* dw, double* db,
    int n, int c, int h, int w, int f, int kh, int kw, int sh, int sw, int ph, int pw) {
    cnn_error = NULL;
    if (!x || !weight || !grad || !dx || !dw || !db) {
        cnn_error = "conv2d backward requires input, weight, gradient and three output buffers";
        return;
    }
    int oh, ow;
    if (!conv2d_dims(n, c, h, w, f, kh, kw, sh, sw, ph, pw, &oh, &ow)) return;
    memset(dx, 0, (size_t)n * c * h * w * sizeof(double));
    memset(dw, 0, (size_t)f * c * kh * kw * sizeof(double));
    memset(db, 0, (size_t)f * sizeof(double));
    // Every output contributes g*w to its input pixels and g*x to its weights.
    for (int batch = 0; batch < n; batch++) {
        for (int filter = 0; filter < f; filter++) {
            for (int row = 0; row < oh; row++) {
                for (int col = 0; col < ow; col++) {
                    double g = grad[((batch * f + filter) * oh + row) * ow + col];
                    db[filter] += g;
                    for (int channel = 0; channel < c; channel++) {
                        for (int ky = 0; ky < kh; ky++) {
                            int iy = row * sh - ph + ky;
                            if (iy < 0 || iy >= h) continue;
                            for (int kx = 0; kx < kw; kx++) {
                                int ix = col * sw - pw + kx;
                                if (ix < 0 || ix >= w) continue;
                                int xi = ((batch * c + channel) * h + iy) * w + ix;
                                int wi = ((filter * c + channel) * kh + ky) * kw + kx;
                                dx[xi] += g * weight[wi];
                                dw[wi] += g * x[xi];
                            }
                        }
                    }
                }
            }
        }
    }
}

static int pool2d_dims(int n, int c, int h, int w, int kh, int kw, int sh, int sw, int* oh, int* ow) {
    return cnn_count4(n, c, h, w) &&
        cnn_window(h, w, kh, kw, sh, sw, 0, 0, oh, ow) && cnn_count4(n, c, *oh, *ow);
}

EXPORT void maxpool2d_forward(const double* x, double* out, int* indices,
    int n, int c, int h, int w, int kh, int kw, int sh, int sw) {
    cnn_error = NULL;
    if (!x || !out || !indices) {
        cnn_error = "maxpool2d requires input, output and index buffers";
        return;
    }
    int oh, ow;
    if (!pool2d_dims(n, c, h, w, kh, kw, sh, sw, &oh, &ow)) return;
    for (int plane = 0; plane < n * c; plane++) {
        for (int row = 0; row < oh; row++) {
            for (int col = 0; col < ow; col++) {
                int best = (plane * h + row * sh) * w + col * sw;
                // First maximum wins ties, including -inf; first NaN propagates.
                for (int ky = 0; ky < kh; ky++) {
                    for (int kx = 0; kx < kw; kx++) {
                        int i = (plane * h + row * sh + ky) * w + col * sw + kx;
                        if (!isnan(x[best]) && (isnan(x[i]) || x[i] > x[best])) best = i;
                    }
                }
                int oi = (plane * oh + row) * ow + col;
                out[oi] = x[best];
                indices[oi] = best;
            }
        }
    }
}

EXPORT void maxpool2d_backward(const double* grad, const int* indices, double* dx,
    int n, int c, int h, int w, int kh, int kw, int sh, int sw) {
    cnn_error = NULL;
    if (!grad || !indices || !dx) {
        cnn_error = "maxpool2d backward requires gradient, index and output buffers";
        return;
    }
    int oh, ow;
    if (!pool2d_dims(n, c, h, w, kh, kw, sh, sw, &oh, &ow)) return;
    // Validate the complete cache before writing anything through its indices.
    for (int plane = 0; plane < n * c; plane++) {
        for (int row = 0; row < oh; row++) {
            for (int col = 0; col < ow; col++) {
                long long local = (long long)indices[(plane * oh + row) * ow + col] - plane * h * w;
                if (local < 0 || local >= h * w || local / w < row * sh ||
                    local / w >= row * sh + kh || local % w < col * sw || local % w >= col * sw + kw) {
                    cnn_error = "maxpool2d index is outside its input window";
                    return;
                }
            }
        }
    }
    memset(dx, 0, (size_t)n * c * h * w * sizeof(double));
    for (int i = 0; i < n * c * oh * ow; i++) dx[indices[i]] += grad[i];
}
