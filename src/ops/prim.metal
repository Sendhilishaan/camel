#include <metal_stdlib>
using namespace metal;

kernel void k_matmul_forward(device const float* A [[buffer(0)]],
                              device const float* B [[buffer(1)]],
                              device float* out [[buffer(2)]],
                              constant int* dims [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
    int n = dims[0], k = dims[1], m = dims[2];
    int i = gid.y, j = gid.x;
    if (i >= n || j >= m) return;
    float acc = 0.0;
    for (int l = 0; l < k; l++) acc += A[i*k+l] * B[l*m+j];
    out[i*m+j] = acc;
}

kernel void k_matmul_backward_da(device const float* B [[buffer(0)]],
                                  device const float* grad_out [[buffer(1)]],
                                  device float* da [[buffer(2)]],
                                  constant int* dims [[buffer(3)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    int n = dims[0], k = dims[1], m = dims[2];
    int i = gid.y, j = gid.x;
    if (i >= n || j >= k) return;
    float acc = 0.0;
    for (int l = 0; l < m; l++) acc += grad_out[i*m+l] * B[j*m+l];
    da[i*k+j] = acc;
}

kernel void k_matmul_backward_db(device const float* A [[buffer(0)]],
                                  device const float* grad_out [[buffer(1)]],
                                  device float* db [[buffer(2)]],
                                  constant int* dims [[buffer(3)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    int n = dims[0], k = dims[1], m = dims[2];
    int i = gid.y, j = gid.x;
    if (i >= k || j >= m) return;
    float acc = 0.0;
    for (int l = 0; l < n; l++) acc += A[l*k+i] * grad_out[l*m+j];
    db[i*m+j] = acc;
}

kernel void k_matadd_broadcast_forward(device float* A [[buffer(0)]],
                                        device const float* B [[buffer(1)]],
                                        constant int* dims [[buffer(2)]],
                                        uint2 gid [[thread_position_in_grid]]) {
    int n = dims[0], m = dims[1];
    int i = gid.y, j = gid.x;
    if (i >= n || j >= m) return;
    A[i*m+j] += B[j];
}

// same as above but writes to a separate buffer instead of mutating A in
// place - needed when A's buffer may be a cached, shared GPU-resident value
kernel void k_matadd_broadcast_forward_out(device const float* A [[buffer(0)]],
                                            device const float* B [[buffer(1)]],
                                            device float* out [[buffer(2)]],
                                            constant int* dims [[buffer(3)]],
                                            uint2 gid [[thread_position_in_grid]]) {
    int n = dims[0], m = dims[1];
    int i = gid.y, j = gid.x;
    if (i >= n || j >= m) return;
    out[i*m+j] = A[i*m+j] + B[j];
}

kernel void k_matadd_broadcast_backward_db(device const float* grad_out [[buffer(0)]],
                                            device float* db [[buffer(1)]],
                                            constant int* dims [[buffer(2)]],
                                            uint gid [[thread_position_in_grid]]) {
    int n = dims[0], m = dims[1];
    if ((int)gid >= m) return;
    float acc = 0.0;
    for (int i = 0; i < n; i++) acc += grad_out[i*m+(int)gid];
    db[gid] = acc;
}

kernel void k_matsub_forward(device const float* A [[buffer(0)]],
                              device const float* B [[buffer(1)]],
                              device float* out [[buffer(2)]],
                              constant int* dims [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= dims[0]) return;
    out[gid] = A[gid] - B[gid];
}

// plain elementwise A + B, same shape - used to combine gradient
// contributions on a fan-out node (see Tensor._accum), not a Tensor op
kernel void k_add(device const float* A [[buffer(0)]],
                   device const float* B [[buffer(1)]],
                   device float* out [[buffer(2)]],
                   constant int* dims [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= dims[0]) return;
    out[gid] = A[gid] + B[gid];
}

kernel void k_negate(device const float* x [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      constant int* dims [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= dims[0]) return;
    out[gid] = -x[gid];
}

kernel void k_hadamard_forward(device const float* A [[buffer(0)]],
                                device const float* B [[buffer(1)]],
                                device float* out [[buffer(2)]],
                                constant int* dims [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= dims[0]) return;
    out[gid] = A[gid] * B[gid];
}

kernel void k_hadamard_backward(device const float* grad_out [[buffer(0)]],
                                 device const float* A [[buffer(1)]],
                                 device const float* B [[buffer(2)]],
                                 device float* dA [[buffer(3)]],
                                 device float* dB [[buffer(4)]],
                                 constant int* dims [[buffer(5)]],
                                 uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= dims[0]) return;
    float g = grad_out[gid];
    dA[gid] = g * B[gid];
    dB[gid] = g * A[gid];
}

kernel void k_fill(device float* out [[buffer(0)]],
                    constant int* dims [[buffer(1)]],
                    constant float& value [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= dims[0]) return;
    out[gid] = value;
}

kernel void k_tanh_forward(device const float* Z [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant int* dims [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= dims[0]) return;
    out[gid] = tanh(Z[gid]);
}

kernel void k_tanh_backward(device const float* out [[buffer(0)]],
                             device const float* grad_out [[buffer(1)]],
                             device float* dz [[buffer(2)]],
                             constant int* dims [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= dims[0]) return;
    float o = out[gid];
    dz[gid] = grad_out[gid] * (1.0 - o*o);
}

kernel void k_relu_forward(device const float* Z [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant int* dims [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= dims[0]) return;
    out[gid] = max(Z[gid], 0.0);
}

kernel void k_relu_backward(device const float* out [[buffer(0)]],
                             device const float* grad_out [[buffer(1)]],
                             device float* dz [[buffer(2)]],
                             constant int* dims [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= dims[0]) return;
    dz[gid] = (out[gid] > 0.0) ? grad_out[gid] : 0.0;
}

kernel void k_softmax_xent_backward(device const float* probs [[buffer(0)]],
                                     device const float* Y [[buffer(1)]],
                                     device float* dZ [[buffer(2)]],
                                     constant int* dims [[buffer(3)]],
                                     constant float& scale [[buffer(4)]],
                                     uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= dims[0]) return;
    dZ[gid] = scale * (probs[gid] - Y[gid]);
}

kernel void k_matmean_forward(device const float* A [[buffer(0)]],
                               device float* out [[buffer(1)]],
                               constant int* dims [[buffer(2)]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint tgSize [[threads_per_threadgroup]],
                               threadgroup float* sdata [[threadgroup(0)]]) {
    int n = dims[0];
    float sum = 0.0;
    for (uint i = tid; i < (uint)n; i += tgSize) sum += A[i];
    sdata[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgSize/2; stride > 0; stride /= 2) {
        if (tid < stride) sdata[tid] += sdata[tid+stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[0] = sdata[0] / (float)n;
}

kernel void k_softmax_xent_forward(device const float* Z [[buffer(0)]],
                                    device const float* Y [[buffer(1)]],
                                    device float* probs [[buffer(2)]],
                                    device float* row_loss [[buffer(3)]],
                                    constant int* dims [[buffer(4)]],
                                    uint tgid [[threadgroup_position_in_grid]],
                                    uint tid [[thread_position_in_threadgroup]],
                                    uint tgSize [[threads_per_threadgroup]],
                                    threadgroup float* sdata [[threadgroup(0)]]) {
    int m = dims[1];
    uint row = tgid;
    device const float* z_row = Z + row*m;
    device const float* y_row = Y + row*m;
    device float* p_row = probs + row*m;

    float local_max = -INFINITY;
    for (uint j = tid; j < (uint)m; j += tgSize) local_max = max(local_max, z_row[j]);
    sdata[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgSize/2; stride > 0; stride /= 2) {
        if (tid < stride) sdata[tid] = max(sdata[tid], sdata[tid+stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = sdata[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0;
    float local_true = 0.0;
    for (uint j = tid; j < (uint)m; j += tgSize) {
        float e = exp(z_row[j] - row_max);
        p_row[j] = e;
        local_sum += e;
        local_true += z_row[j] * y_row[j];
    }
    sdata[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgSize/2; stride > 0; stride /= 2) {
        if (tid < stride) sdata[tid] += sdata[tid+stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = sdata[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sdata[tid] = local_true;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgSize/2; stride > 0; stride /= 2) {
        if (tid < stride) sdata[tid] += sdata[tid+stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_true = sdata[0];

    for (uint j = tid; j < (uint)m; j += tgSize) p_row[j] /= row_sum;
    if (tid == 0) row_loss[row] = -(row_true - row_max) + log(row_sum);
}

/*
    optimizer steps: each mutates param + its state buffer(s) in place, one
    thread per element, no reduction needed (unlike the forward/backward
    kernels above, in-place is correct here - state buffers aren't part of
    the autograd graph, nothing else ever reads a stale copy of them).
    hparams packs the scalar hyperparameters into one small buffer instead of
    a separate buffer index per scalar.
*/

kernel void k_sgd_step(device float* param [[buffer(0)]],
                        device float* velocity [[buffer(1)]],
                        device const float* grad [[buffer(2)]],
                        constant float* hparams [[buffer(3)]], // momentum, lr
                        constant int* dims [[buffer(4)]],
                        uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= dims[0]) return;
    float momentum = hparams[0], lr = hparams[1];
    float v = momentum * velocity[gid] + grad[gid];
    velocity[gid] = v;
    param[gid] -= lr * v;
}

kernel void k_adagrad_step(device float* param [[buffer(0)]],
                            device float* grad_sum [[buffer(1)]],
                            device const float* grad [[buffer(2)]],
                            constant float* hparams [[buffer(3)]], // lr, eps
                            constant int* dims [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= dims[0]) return;
    float lr = hparams[0], eps = hparams[1];
    float g = grad[gid];
    float G = grad_sum[gid] + g * g;
    grad_sum[gid] = G;
    param[gid] -= lr * g / (sqrt(G) + eps);
}

kernel void k_rmsprop_step(device float* param [[buffer(0)]],
                            device float* ema_sq [[buffer(1)]],
                            device const float* grad [[buffer(2)]],
                            constant float* hparams [[buffer(3)]], // lr, eps, decay
                            constant int* dims [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= dims[0]) return;
    float lr = hparams[0], eps = hparams[1], decay = hparams[2];
    float g = grad[gid];
    float s = decay * ema_sq[gid] + (1.0 - decay) * g * g;
    ema_sq[gid] = s;
    param[gid] -= lr * g / (sqrt(s) + eps);
}

kernel void k_adam_step(device float* param [[buffer(0)]],
                         device float* exp_avg [[buffer(1)]],
                         device float* exp_avg_sq [[buffer(2)]],
                         device const float* grad [[buffer(3)]],
                         constant float* hparams [[buffer(4)]], // lr, eps, decay1, decay2, bc1, bc2
                         constant int* dims [[buffer(5)]],
                         uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= dims[0]) return;
    float lr = hparams[0], eps = hparams[1], decay1 = hparams[2], decay2 = hparams[3];
    float bc1 = hparams[4], bc2 = hparams[5];
    float g = grad[gid];
    float m = decay1 * exp_avg[gid] + (1.0 - decay1) * g;
    float v = decay2 * exp_avg_sq[gid] + (1.0 - decay2) * g * g;
    exp_avg[gid] = m;
    exp_avg_sq[gid] = v;
    float m_hat = m / bc1;
    float v_hat = v / bc2;
    param[gid] -= lr * m_hat / (sqrt(v_hat) + eps);
}
