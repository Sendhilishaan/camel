#ifndef PRIM_CUDA_H
#define PRIM_CUDA_H

#include "prim.h"

/*
    CUDA float32 primitives: same staging and resident signatures as Metal.
    Handles own device allocations; release with camel_cuda_buffer_free.
    Null data creates a zeroed buffer. Counts/dimensions must be positive.
    Failures return NULL/void and set a thread-local error, read immediately
    with camel_cuda_get_last_error. Free accepts NULL and preserves the error.
    Calls enqueue on the default stream; read/synchronize waits for completion.
*/
#ifdef __cplusplus
extern "C" {
#endif

EXPORT const char* camel_cuda_get_last_error(void);
EXPORT void camel_cuda_synchronize(void);

EXPORT void matmul_forward_cuda(const float* A, const float* B, float* out, int n, int k, int m);

EXPORT void matmul_backward_cuda(const float* A, const float* B, const float* grad_out, float* da, float* db, int n, int k, int m);

EXPORT void matadd_broadcast_forward_cuda(float* A, const float* B, int n, int m);

EXPORT void matadd_broadcast_backward_cuda(const float* grad_out, float* dX, float* db, int n, int m);

EXPORT void matsub_forward_cuda(const float* A, const float* B, float* out, int n, int m);

EXPORT void matsub_backward_cuda(const float* grad_out, float* dA, float* dB, int n, int m);

EXPORT void hadamard_forward_cuda(const float* A, const float* B, float* out, int n, int m);

EXPORT void hadamard_backward_cuda(const float* grad_out, const float* A, const float* B, float* dA, float* dB, int n, int m);

EXPORT void matmean_forward_cuda(const float* A, float* out, int n);

EXPORT void matmean_backward_cuda(float* dx, int n, float grad_out);

EXPORT void tanh_forward_cuda(const float* Z, float* out, int n, int m);

EXPORT void tanh_backward_cuda(const float* out, const float* grad_out, float* dz, int n, int m);

EXPORT void relu_forward_cuda(const float* Z, float* out, int n, int m);

EXPORT void relu_backward_cuda(const float* out, const float* grad_out, float* dz, int n, int m);

EXPORT void softmax_xent_forward_cuda(const float* Z, const float* Y, float* probs, float* out_loss, int n, int m);

EXPORT void softmax_xent_backward_cuda(const float* probs, const float* Y, float* dZ, float grad_out, int n, int m);

EXPORT int camel_cuda_device_available(void);

EXPORT void *camel_cuda_buffer_create(const float* data, int count);

EXPORT void camel_cuda_buffer_read(void* handle, float* out, int count);

EXPORT void camel_cuda_buffer_free(void* handle);

EXPORT void *matmul_forward_cuda_resident(void* a, void* b, int n, int k, int m);

EXPORT void matmul_backward_cuda_resident(void* a, void* b, void* grad_out, int n, int k, int m, void** outDA, void** outDB);

EXPORT void *matadd_broadcast_forward_cuda_resident(void* a, void* b, int n, int m);

EXPORT void matadd_broadcast_backward_cuda_resident(void* grad_out, int n, int m, void** outDX, void** outDB);

EXPORT void *matsub_forward_cuda_resident(void* a, void* b, int n, int m);

EXPORT void matsub_backward_cuda_resident(void* grad_out, int n, int m, void** outDA, void** outDB);

EXPORT void *hadamard_forward_cuda_resident(void* a, void* b, int n, int m);

EXPORT void hadamard_backward_cuda_resident(void* grad_out, void* a, void* b, int n, int m, void** outDA, void** outDB);

EXPORT void *matmean_forward_cuda_resident(void* a, int n);

EXPORT void *matmean_backward_cuda_resident(int n, float grad_out);

EXPORT void *tanh_forward_cuda_resident(void* z, int n, int m);

EXPORT void *tanh_backward_cuda_resident(void* out, void* grad_out, int n, int m);

EXPORT void *relu_forward_cuda_resident(void* z, int n, int m);

EXPORT void *relu_backward_cuda_resident(void* out, void* grad_out, int n, int m);

EXPORT void softmax_xent_forward_cuda_resident(void* z, void* y, int n, int m, void** outProbs, float* outLoss);

EXPORT void *softmax_xent_backward_cuda_resident(void* probs, void* y, float grad_out, int n, int m);

EXPORT void *add_cuda_resident(void* a, void* b, int total);

EXPORT void sgd_step_cuda_resident(void* param, void* velocity, void* grad, float momentum, float lr, int total);

EXPORT void adagrad_step_cuda_resident(void* param, void* grad_sum, void* grad, float lr, float eps, int total);

EXPORT void rmsprop_step_cuda_resident(void* param, void* ema_sq, void* grad, float lr, float eps, float decay, int total);

EXPORT void adam_step_cuda_resident(void* param, void* exp_avg, void* exp_avg_sq, void* grad, float lr, float eps, float decay1, float decay2, float bc1, float bc2, int total);

#ifdef __cplusplus
}
#endif

#endif
