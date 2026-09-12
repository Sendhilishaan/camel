#include "prim_cuda.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <climits>
#include <cstdio>
#include <memory>
#include <stdexcept>

/*
    CUDA kernels, float32 like Metal. Matmul tiles both inputs into shared
    memory; backward uses the same kernel with implicit transposes. Reductions
    use a 256-thread tree, no float atomics or library math dependencies.

    Two entry points per primitive: float* staging calls and void* resident
    calls. A handle owns its allocation. Forward/backward always allocate
    fresh outputs; only optimizer steps mutate their inputs.

    Errors stop at the C boundary and become Python RuntimeErrors in _c.py.
    Dispatch is asynchronous on the default stream; read/synchronize reports
    execution errors. RAII also frees partial outputs if allocation fails.
*/
static constexpr int THREADS = 256;
static constexpr int TILE = 16;
static thread_local char last_error[1024] = {};

static void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

#define CUDA_BEGIN try { last_error[0] = '\0';
#define CUDA_END(ret) } catch (const std::exception& e) { \
    std::snprintf(last_error, sizeof(last_error), "%s", e.what()); return ret; \
} catch (...) { \
    std::snprintf(last_error, sizeof(last_error), "unknown CUDA error"); return ret; \
}

static int count2(int n, int m) {
    if (n <= 0 || m <= 0 || n > INT_MAX / m)
        throw std::runtime_error("CUDA dimensions must be positive and fit in int32");
    return n * m;
}

struct Buffer {
    float* data = nullptr;
    int count;

    explicit Buffer(int n) : count(count2(n, 1)) {
        check(cudaMalloc(reinterpret_cast<void**>(&data), sizeof(float) * count));
    }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void read(float* out, int n) const {
        if (!out || n <= 0 || n > count) throw std::runtime_error("invalid CUDA buffer read");
        check(cudaMemcpy(out, data, sizeof(float) * n, cudaMemcpyDeviceToHost));
    }
};
using Buf = std::unique_ptr<Buffer>;

static Buf empty(int n) { return Buf(new Buffer(n)); }

static Buf upload(const float* data, int n) {
    auto out = empty(n);
    if (data) check(cudaMemcpy(out->data, data, sizeof(float) * n, cudaMemcpyHostToDevice));
    else check(cudaMemset(out->data, 0, sizeof(float) * n));
    return out;
}

static float* ptr(void* handle, int n) {
    if (!handle || n <= 0 || static_cast<Buffer*>(handle)->count < n)
        throw std::runtime_error("null or undersized CUDA buffer");
    return static_cast<Buffer*>(handle)->data;
}

static Buf take(void* handle) {
    if (!handle) throw std::runtime_error(last_error[0] ? last_error : "null CUDA result");
    return Buf(static_cast<Buffer*>(handle));
}

static int blocks(int n) { return (n - 1) / THREADS + 1; }

// Strides express A, B or their transposes without temporary copies.
__global__ void k_matmul(const float* a, const float* b, float* out,
                         int n, int k, int m, int ar, int ac, int br, int bc) {
    __shared__ float sa[TILE][TILE], sb[TILE][TILE];
    int tiles_m = (m - 1) / TILE + 1;
    int row = (blockIdx.x / tiles_m) * TILE + threadIdx.y;
    int col = (blockIdx.x % tiles_m) * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int base = 0; base < k; base += TILE) {
        int ak = base + threadIdx.x, bk = base + threadIdx.y;
        sa[threadIdx.y][threadIdx.x] = row < n && ak < k ? a[row * ar + ak * ac] : 0.0f;
        sb[threadIdx.y][threadIdx.x] = bk < k && col < m ? b[bk * br + col * bc] : 0.0f;
        __syncthreads();
        for (int j = 0; j < TILE; j++) sum += sa[threadIdx.y][j] * sb[j][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < m) AT(out, row, col, m) = sum;
}

static Buf matmul(const float* a, const float* b, int n, int k, int m,
                  int ar, int ac, int br, int bc) {
    auto out = empty(count2(n, m));
    int tiles = ((n - 1) / TILE + 1) * ((m - 1) / TILE + 1);
    k_matmul<<<tiles, dim3(TILE, TILE)>>>(a, b, out->data, n, k, m, ar, ac, br, bc);
    check(cudaGetLastError());
    return out;
}

__global__ void k_bias(const float* a, const float* b, float* out, int total, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = a[i] + b[i % m];
}

__global__ void k_bias_backward(const float* grad, float* db, int n, int m) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += AT(grad, i, j, m);
    db[j] = sum;
}

// The small elementwise kernels share indexing, with the op fixed at compile time.
enum Element { ADD, SUB, MUL, NEG, TANH, TANH_GRAD, RELU, RELU_GRAD };
template<Element op>
__global__ void k_element(const float* a, const float* b, float* out, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    if (op == ADD) out[i] = a[i] + b[i];
    if (op == SUB) out[i] = a[i] - b[i];
    if (op == MUL) out[i] = a[i] * b[i];
    if (op == NEG) out[i] = -a[i];
    if (op == TANH) out[i] = tanhf(a[i]);
    if (op == TANH_GRAD) out[i] = b[i] * (1.0f - a[i] * a[i]);
    if (op == RELU) out[i] = a[i] > 0.0f ? a[i] : 0.0f;
    if (op == RELU_GRAD) out[i] = a[i] > 0.0f ? b[i] : 0.0f;
}

template<Element op>
static Buf element(const float* a, const float* b, int total) {
    auto out = empty(total);
    k_element<op><<<blocks(total), THREADS>>>(a, b, out->data, total);
    check(cudaGetLastError());
    return out;
}

static Buf copy(const float* a, int total) {
    auto out = empty(total);
    check(cudaMemcpyAsync(out->data, a, sizeof(float) * total, cudaMemcpyDeviceToDevice));
    return out;
}

template<bool maximum = false>
__device__ float reduce(float value, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (int offset = THREADS / 2; offset; offset /= 2) {
        if (tid < offset)
            shared[tid] = maximum ? fmaxf(shared[tid], shared[tid + offset]) : shared[tid] + shared[tid + offset];
        __syncthreads();
    }
    float result = shared[0];
    __syncthreads(); // all threads finish reading before the next reduction
    return result;
}

__global__ void k_mean(const float* a, float* out, int total) {
    __shared__ float shared[THREADS];
    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < static_cast<size_t>(total); i += THREADS) sum += a[i];
    sum = reduce(sum, shared);
    if (threadIdx.x == 0) *out = sum / total;
}

__global__ void k_fill(float* out, int total, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = value;
}

__global__ void k_softmax(const float* z, const float* y, float* probs, float* losses, int m) {
    __shared__ float shared[THREADS];
    int row = blockIdx.x;
    float max_z = -CUDART_INF_F;
    for (int j = threadIdx.x; j < m; j += THREADS) max_z = fmaxf(max_z, AT(z, row, j, m));
    max_z = reduce<true>(max_z, shared);
    float sum = 0.0f, target = 0.0f;
    for (int j = threadIdx.x; j < m; j += THREADS) {
        float shifted = AT(z, row, j, m) - max_z;
        float p = expf(shifted);
        AT(probs, row, j, m) = p;
        sum += p;
        target += shifted * AT(y, row, j, m);
    }
    sum = reduce(sum, shared);
    target = reduce(target, shared);
    for (int j = threadIdx.x; j < m; j += THREADS) AT(probs, row, j, m) /= sum;
    if (threadIdx.x == 0) losses[row] = logf(sum) - target;
}

__global__ void k_softmax_backward(const float* probs, const float* y, float* dz, int total, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) dz[i] = (probs[i] - y[i]) * scale;
}

__global__ void k_sgd_step(float* param, float* velocity, const float* grad,
                          float momentum, float lr, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = momentum * velocity[i] + grad[i];
    velocity[i] = v;
    param[i] -= lr * v;
}

__global__ void k_adagrad_step(float* param, float* grad_sum, const float* grad,
                              float lr, float eps, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float g = grad[i], G = grad_sum[i] + g * g;
    grad_sum[i] = G;
    param[i] -= lr * g / (sqrtf(G) + eps);
}

__global__ void k_rmsprop_step(float* param, float* ema_sq, const float* grad,
                              float lr, float eps, float decay, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float g = grad[i], s = decay * ema_sq[i] + (1.0f - decay) * g * g;
    ema_sq[i] = s;
    param[i] -= lr * g / (sqrtf(s) + eps);
}

__global__ void k_adam_step(float* param, float* exp_avg, float* exp_avg_sq, const float* grad,
                           float lr, float eps, float decay1, float decay2, float bc1, float bc2, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float g = grad[i];
    float m = decay1 * exp_avg[i] + (1.0f - decay1) * g;
    float v = decay2 * exp_avg_sq[i] + (1.0f - decay2) * g * g;
    exp_avg[i] = m;
    exp_avg_sq[i] = v;
    param[i] -= lr * (m / bc1) / (sqrtf(v / bc2) + eps);
}

EXPORT const char* camel_cuda_get_last_error(void) { return last_error; }

EXPORT int camel_cuda_device_available(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        cudaGetLastError();
        return 0;
    }
    if (cudaFree(nullptr) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return 1;
}

EXPORT void camel_cuda_synchronize(void) {
    CUDA_BEGIN
    check(cudaDeviceSynchronize());
    CUDA_END((void)0)
}

EXPORT void* camel_cuda_buffer_create(const float* data, int count) {
    CUDA_BEGIN
    return upload(data, count).release();
    CUDA_END(nullptr)
}

EXPORT void camel_cuda_buffer_read(void* handle, float* out, int count) {
    CUDA_BEGIN
    ptr(handle, count);
    static_cast<Buffer*>(handle)->read(out, count);
    CUDA_END((void)0)
}

EXPORT void camel_cuda_buffer_free(void* handle) {
    delete static_cast<Buffer*>(handle);
}

EXPORT void* matmul_forward_cuda_resident(void* a, void* b, int n, int k, int m) {
    CUDA_BEGIN
    return matmul(ptr(a, count2(n, k)), ptr(b, count2(k, m)), n, k, m, k, 1, m, 1).release();
    CUDA_END(nullptr)
}

EXPORT void matmul_backward_cuda_resident(void* a, void* b, void* grad_out, int n, int k, int m,
                                          void** outDA, void** outDB) {
    CUDA_BEGIN
    if (!outDA || !outDB) throw std::runtime_error("null CUDA output slot");
    *outDA = *outDB = nullptr;
    float* A = ptr(a, count2(n, k)), *B = ptr(b, count2(k, m)), *G = ptr(grad_out, count2(n, m));
    auto da = matmul(G, B, n, m, k, m, 1, 1, m);
    auto db = matmul(A, G, k, n, m, 1, k, m, 1);
    *outDA = da.release(); *outDB = db.release();
    CUDA_END((void)0)
}

EXPORT void* matadd_broadcast_forward_cuda_resident(void* a, void* b, int n, int m) {
    CUDA_BEGIN
    int total = count2(n, m);
    auto out = empty(total);
    float* A = ptr(a, total), *B = ptr(b, m);
    k_bias<<<blocks(total), THREADS>>>(A, B, out->data, total, m);
    check(cudaGetLastError());
    return out.release();
    CUDA_END(nullptr)
}

EXPORT void matadd_broadcast_backward_cuda_resident(void* grad_out, int n, int m, void** outDX, void** outDB) {
    CUDA_BEGIN
    if (!outDX || !outDB) throw std::runtime_error("null CUDA output slot");
    *outDX = *outDB = nullptr;
    int total = count2(n, m);
    float* G = ptr(grad_out, total);
    auto dx = copy(G, total), db = empty(m);
    k_bias_backward<<<blocks(m), THREADS>>>(G, db->data, n, m);
    check(cudaGetLastError());
    *outDX = dx.release(); *outDB = db.release();
    CUDA_END((void)0)
}

EXPORT void* matsub_forward_cuda_resident(void* a, void* b, int n, int m) {
    CUDA_BEGIN
    int total = count2(n, m);
    return element<SUB>(ptr(a, total), ptr(b, total), total).release();
    CUDA_END(nullptr)
}

EXPORT void matsub_backward_cuda_resident(void* grad_out, int n, int m, void** outDA, void** outDB) {
    CUDA_BEGIN
    if (!outDA || !outDB) throw std::runtime_error("null CUDA output slot");
    *outDA = *outDB = nullptr;
    int total = count2(n, m);
    float* G = ptr(grad_out, total);
    auto da = copy(G, total), db = element<NEG>(G, nullptr, total);
    *outDA = da.release(); *outDB = db.release();
    CUDA_END((void)0)
}

EXPORT void* hadamard_forward_cuda_resident(void* a, void* b, int n, int m) {
    CUDA_BEGIN
    int total = count2(n, m);
    return element<MUL>(ptr(a, total), ptr(b, total), total).release();
    CUDA_END(nullptr)
}

EXPORT void hadamard_backward_cuda_resident(void* grad_out, void* a, void* b, int n, int m,
                                            void** outDA, void** outDB) {
    CUDA_BEGIN
    if (!outDA || !outDB) throw std::runtime_error("null CUDA output slot");
    *outDA = *outDB = nullptr;
    int total = count2(n, m);
    float* G = ptr(grad_out, total), *A = ptr(a, total), *B = ptr(b, total);
    auto da = element<MUL>(G, B, total), db = element<MUL>(G, A, total);
    *outDA = da.release(); *outDB = db.release();
    CUDA_END((void)0)
}

EXPORT void* matmean_forward_cuda_resident(void* a, int n) {
    CUDA_BEGIN
    float* A = ptr(a, count2(n, 1));
    auto out = empty(1);
    k_mean<<<1, THREADS>>>(A, out->data, n);
    check(cudaGetLastError());
    return out.release();
    CUDA_END(nullptr)
}

EXPORT void* matmean_backward_cuda_resident(int n, float grad_out) {
    CUDA_BEGIN
    auto out = empty(n);
    k_fill<<<blocks(n), THREADS>>>(out->data, n, grad_out / n);
    check(cudaGetLastError());
    return out.release();
    CUDA_END(nullptr)
}

EXPORT void* tanh_forward_cuda_resident(void* z, int n, int m) {
    CUDA_BEGIN
    int total = count2(n, m);
    return element<TANH>(ptr(z, total), nullptr, total).release();
    CUDA_END(nullptr)
}

EXPORT void* tanh_backward_cuda_resident(void* out, void* grad_out, int n, int m) {
    CUDA_BEGIN
    int total = count2(n, m);
    return element<TANH_GRAD>(ptr(out, total), ptr(grad_out, total), total).release();
    CUDA_END(nullptr)
}

EXPORT void* relu_forward_cuda_resident(void* z, int n, int m) {
    CUDA_BEGIN
    int total = count2(n, m);
    return element<RELU>(ptr(z, total), nullptr, total).release();
    CUDA_END(nullptr)
}

EXPORT void* relu_backward_cuda_resident(void* out, void* grad_out, int n, int m) {
    CUDA_BEGIN
    int total = count2(n, m);
    return element<RELU_GRAD>(ptr(out, total), ptr(grad_out, total), total).release();
    CUDA_END(nullptr)
}

EXPORT void softmax_xent_forward_cuda_resident(void* z, void* y, int n, int m, void** outProbs, float* outLoss) {
    CUDA_BEGIN
    if (!outProbs || !outLoss) throw std::runtime_error("null CUDA output slot");
    *outProbs = nullptr;
    int total = count2(n, m);
    float* Z = ptr(z, total), *Y = ptr(y, total);
    auto probs = empty(total), rows = empty(n), loss = empty(1);
    k_softmax<<<n, THREADS>>>(Z, Y, probs->data, rows->data, m);
    check(cudaGetLastError());
    k_mean<<<1, THREADS>>>(rows->data, loss->data, n);
    check(cudaGetLastError());
    loss->read(outLoss, 1); // the resident interface returns just this scalar to CPU, like Metal
    *outProbs = probs.release();
    CUDA_END((void)0)
}

EXPORT void* softmax_xent_backward_cuda_resident(void* probs, void* y, float grad_out, int n, int m) {
    CUDA_BEGIN
    int total = count2(n, m);
    float* P = ptr(probs, total), *Y = ptr(y, total);
    auto out = empty(total);
    k_softmax_backward<<<blocks(total), THREADS>>>(P, Y, out->data, total, grad_out / n);
    check(cudaGetLastError());
    return out.release();
    CUDA_END(nullptr)
}

EXPORT void* add_cuda_resident(void* a, void* b, int total) {
    CUDA_BEGIN
    return element<ADD>(ptr(a, total), ptr(b, total), total).release();
    CUDA_END(nullptr)
}

EXPORT void sgd_step_cuda_resident(void* param, void* velocity, void* grad, float momentum, float lr, int total) {
    CUDA_BEGIN
    float* P = ptr(param, total), *V = ptr(velocity, total), *G = ptr(grad, total);
    k_sgd_step<<<blocks(total), THREADS>>>(P, V, G, momentum, lr, total);
    check(cudaGetLastError());
    CUDA_END((void)0)
}

EXPORT void adagrad_step_cuda_resident(void* param, void* grad_sum, void* grad, float lr, float eps, int total) {
    CUDA_BEGIN
    float* P = ptr(param, total), *S = ptr(grad_sum, total), *G = ptr(grad, total);
    k_adagrad_step<<<blocks(total), THREADS>>>(P, S, G, lr, eps, total);
    check(cudaGetLastError());
    CUDA_END((void)0)
}

EXPORT void rmsprop_step_cuda_resident(void* param, void* ema_sq, void* grad, float lr, float eps, float decay, int total) {
    CUDA_BEGIN
    float* P = ptr(param, total), *S = ptr(ema_sq, total), *G = ptr(grad, total);
    k_rmsprop_step<<<blocks(total), THREADS>>>(P, S, G, lr, eps, decay, total);
    check(cudaGetLastError());
    CUDA_END((void)0)
}

EXPORT void adam_step_cuda_resident(void* param, void* exp_avg, void* exp_avg_sq, void* grad,
                                   float lr, float eps, float decay1, float decay2, float bc1, float bc2, int total) {
    CUDA_BEGIN
    float* P = ptr(param, total), *M = ptr(exp_avg, total), *V = ptr(exp_avg_sq, total), *G = ptr(grad, total);
    k_adam_step<<<blocks(total), THREADS>>>(P, M, V, G, lr, eps, decay1, decay2, bc1, bc2, total);
    check(cudaGetLastError());
    CUDA_END((void)0)
}

// float* entry points: same signatures as Metal, staging around the resident API.
EXPORT void matmul_forward_cuda(const float* A, const float* B, float* out, int n, int k, int m) {
    CUDA_BEGIN
    auto a = upload(A, count2(n, k)), b = upload(B, count2(k, m));
    auto result = take(matmul_forward_cuda_resident(a.get(), b.get(), n, k, m));
    result->read(out, count2(n, m));
    CUDA_END((void)0)
}

EXPORT void matmul_backward_cuda(const float* A, const float* B, const float* G, float* dA, float* dB, int n, int k, int m) {
    CUDA_BEGIN
    auto a = upload(A, count2(n, k)), b = upload(B, count2(k, m)), g = upload(G, count2(n, m));
    void* da = nullptr, *db = nullptr;
    matmul_backward_cuda_resident(a.get(), b.get(), g.get(), n, k, m, &da, &db);
    auto ra = take(da), rb = take(db);
    ra->read(dA, count2(n, k)); rb->read(dB, count2(k, m));
    CUDA_END((void)0)
}

EXPORT void matadd_broadcast_forward_cuda(float* A, const float* B, int n, int m) {
    CUDA_BEGIN
    auto a = upload(A, count2(n, m)), b = upload(B, m);
    auto out = take(matadd_broadcast_forward_cuda_resident(a.get(), b.get(), n, m));
    out->read(A, count2(n, m));
    CUDA_END((void)0)
}

EXPORT void matadd_broadcast_backward_cuda(const float* G, float* dX, float* dB, int n, int m) {
    CUDA_BEGIN
    auto g = upload(G, count2(n, m));
    void* dx = nullptr, *db = nullptr;
    matadd_broadcast_backward_cuda_resident(g.get(), n, m, &dx, &db);
    auto rx = take(dx), rb = take(db);
    rx->read(dX, count2(n, m)); rb->read(dB, m);
    CUDA_END((void)0)
}

EXPORT void matsub_forward_cuda(const float* A, const float* B, float* out, int n, int m) {
    CUDA_BEGIN
    auto a = upload(A, count2(n, m)), b = upload(B, count2(n, m));
    auto result = take(matsub_forward_cuda_resident(a.get(), b.get(), n, m));
    result->read(out, count2(n, m));
    CUDA_END((void)0)
}

EXPORT void matsub_backward_cuda(const float* G, float* dA, float* dB, int n, int m) {
    CUDA_BEGIN
    auto g = upload(G, count2(n, m));
    void* da = nullptr, *db = nullptr;
    matsub_backward_cuda_resident(g.get(), n, m, &da, &db);
    auto ra = take(da), rb = take(db);
    ra->read(dA, count2(n, m)); rb->read(dB, count2(n, m));
    CUDA_END((void)0)
}

EXPORT void hadamard_forward_cuda(const float* A, const float* B, float* out, int n, int m) {
    CUDA_BEGIN
    auto a = upload(A, count2(n, m)), b = upload(B, count2(n, m));
    auto result = take(hadamard_forward_cuda_resident(a.get(), b.get(), n, m));
    result->read(out, count2(n, m));
    CUDA_END((void)0)
}

EXPORT void hadamard_backward_cuda(const float* G, const float* A, const float* B, float* dA, float* dB, int n, int m) {
    CUDA_BEGIN
    int total = count2(n, m);
    auto g = upload(G, total), a = upload(A, total), b = upload(B, total);
    void* da = nullptr, *db = nullptr;
    hadamard_backward_cuda_resident(g.get(), a.get(), b.get(), n, m, &da, &db);
    auto ra = take(da), rb = take(db);
    ra->read(dA, total); rb->read(dB, total);
    CUDA_END((void)0)
}

EXPORT void matmean_forward_cuda(const float* A, float* out, int n) {
    CUDA_BEGIN
    auto a = upload(A, n);
    auto result = take(matmean_forward_cuda_resident(a.get(), n));
    result->read(out, 1);
    CUDA_END((void)0)
}

EXPORT void matmean_backward_cuda(float* dx, int n, float grad_out) {
    CUDA_BEGIN
    auto result = take(matmean_backward_cuda_resident(n, grad_out));
    result->read(dx, n);
    CUDA_END((void)0)
}

EXPORT void tanh_forward_cuda(const float* Z, float* out, int n, int m) {
    CUDA_BEGIN
    auto z = upload(Z, count2(n, m));
    auto result = take(tanh_forward_cuda_resident(z.get(), n, m));
    result->read(out, count2(n, m));
    CUDA_END((void)0)
}

EXPORT void tanh_backward_cuda(const float* out, const float* G, float* dz, int n, int m) {
    CUDA_BEGIN
    auto z = upload(out, count2(n, m)), g = upload(G, count2(n, m));
    auto result = take(tanh_backward_cuda_resident(z.get(), g.get(), n, m));
    result->read(dz, count2(n, m));
    CUDA_END((void)0)
}

EXPORT void relu_forward_cuda(const float* Z, float* out, int n, int m) {
    CUDA_BEGIN
    auto z = upload(Z, count2(n, m));
    auto result = take(relu_forward_cuda_resident(z.get(), n, m));
    result->read(out, count2(n, m));
    CUDA_END((void)0)
}

EXPORT void relu_backward_cuda(const float* out, const float* G, float* dz, int n, int m) {
    CUDA_BEGIN
    auto z = upload(out, count2(n, m)), g = upload(G, count2(n, m));
    auto result = take(relu_backward_cuda_resident(z.get(), g.get(), n, m));
    result->read(dz, count2(n, m));
    CUDA_END((void)0)
}

EXPORT void softmax_xent_forward_cuda(const float* Z, const float* Y, float* probs, float* out_loss, int n, int m) {
    CUDA_BEGIN
    auto z = upload(Z, count2(n, m)), y = upload(Y, count2(n, m));
    void* p = nullptr;
    softmax_xent_forward_cuda_resident(z.get(), y.get(), n, m, &p, out_loss);
    auto result = take(p);
    result->read(probs, count2(n, m));
    CUDA_END((void)0)
}

EXPORT void softmax_xent_backward_cuda(const float* probs, const float* Y, float* dZ, float grad_out, int n, int m) {
    CUDA_BEGIN
    auto p = upload(probs, count2(n, m)), y = upload(Y, count2(n, m));
    auto result = take(softmax_xent_backward_cuda_resident(p.get(), y.get(), grad_out, n, m));
    result->read(dZ, count2(n, m));
    CUDA_END((void)0)
}

/*
    CNN primitives. NCHW: ((batch * channels + channel) * height + y) * width + x.
    Each backward thread owns one gradient element and gathers contributions,
    so overlapping windows add correctly without racing or float atomics.
*/
struct ConvDims {
    int n, c, h, w, f, kh, kw, sh, sw, ph, pw, oh, ow, nx, nw, ny;

    ConvDims(int N, int C, int H, int W, int F, int KH, int KW, int SH, int SW, int PH, int PW)
        : n(N), c(C), h(H), w(W), f(F), kh(KH), kw(KW), sh(SH), sw(SW), ph(PH), pw(PW) {
        nx = count2(count2(n, c), count2(h, w));
        nw = count2(count2(f, c), count2(kh, kw));
        if (sh <= 0 || sw <= 0 || ph < 0 || pw < 0)
            throw std::runtime_error("invalid CNN stride or padding");
        long long padded_h = static_cast<long long>(h) + 2LL * ph;
        long long padded_w = static_cast<long long>(w) + 2LL * pw;
        if (padded_h > INT_MAX || padded_w > INT_MAX || padded_h < kh || padded_w < kw)
            throw std::runtime_error("CNN kernel does not fit input or padded dimensions overflow");
        oh = static_cast<int>((padded_h - kh) / sh + 1);
        ow = static_cast<int>((padded_w - kw) / sw + 1);
        ny = count2(count2(n, f), count2(oh, ow));
    }
};

__global__ void k_conv2d_forward(const float* x, const float* w, const float* b, float* out, ConvDims d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.ny) return;
    int ox = i % d.ow, oy = (i / d.ow) % d.oh;
    int f = (i / d.ow / d.oh) % d.f, batch = i / d.ow / d.oh / d.f;
    float sum = b ? b[f] : 0.0f;
    for (int c = 0; c < d.c; c++) {
        for (int ky = 0; ky < d.kh; ky++) {
            int iy = oy * d.sh - d.ph + ky;
            if (iy < 0 || iy >= d.h) continue;
            for (int kx = 0; kx < d.kw; kx++) {
                int ix = ox * d.sw - d.pw + kx;
                if (ix < 0 || ix >= d.w) continue;
                int xi = ((batch * d.c + c) * d.h + iy) * d.w + ix;
                int wi = ((f * d.c + c) * d.kh + ky) * d.kw + kx;
                sum += x[xi] * w[wi];
            }
        }
    }
    out[i] = sum;
}

__global__ void k_conv2d_dx(const float* w, const float* g, float* dx, ConvDims d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.nx) return;
    int ix = i % d.w, iy = (i / d.w) % d.h;
    int c = (i / d.w / d.h) % d.c, batch = i / d.w / d.h / d.c;
    float sum = 0.0f;
    for (int f = 0; f < d.f; f++) {
        for (int ky = 0; ky < d.kh; ky++) {
            int oy = iy + d.ph - ky;
            if (oy < 0 || oy % d.sh) continue;
            oy /= d.sh;
            if (oy >= d.oh) continue;
            for (int kx = 0; kx < d.kw; kx++) {
                int ox = ix + d.pw - kx;
                if (ox < 0 || ox % d.sw) continue;
                ox /= d.sw;
                if (ox >= d.ow) continue;
                int wi = ((f * d.c + c) * d.kh + ky) * d.kw + kx;
                int gi = ((batch * d.f + f) * d.oh + oy) * d.ow + ox;
                sum += w[wi] * g[gi];
            }
        }
    }
    dx[i] = sum;
}

__global__ void k_conv2d_dw(const float* x, const float* g, float* dw, ConvDims d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.nw) return;
    int kx = i % d.kw, ky = (i / d.kw) % d.kh;
    int c = (i / d.kw / d.kh) % d.c, f = i / d.kw / d.kh / d.c;
    float sum = 0.0f;
    for (int batch = 0; batch < d.n; batch++) {
        for (int oy = 0; oy < d.oh; oy++) {
            int iy = oy * d.sh - d.ph + ky;
            if (iy < 0 || iy >= d.h) continue;
            for (int ox = 0; ox < d.ow; ox++) {
                int ix = ox * d.sw - d.pw + kx;
                if (ix < 0 || ix >= d.w) continue;
                int xi = ((batch * d.c + c) * d.h + iy) * d.w + ix;
                int gi = ((batch * d.f + f) * d.oh + oy) * d.ow + ox;
                sum += x[xi] * g[gi];
            }
        }
    }
    dw[i] = sum;
}

__global__ void k_conv2d_db(const float* g, float* db, ConvDims d) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= d.f) return;
    float sum = 0.0f;
    for (int batch = 0; batch < d.n; batch++)
        for (int j = 0; j < d.oh * d.ow; j++) sum += g[(batch * d.f + f) * d.oh * d.ow + j];
    db[f] = sum;
}

static Buf conv_forward(const float* x, const float* w, const float* b, ConvDims d) {
    auto out = empty(d.ny);
    k_conv2d_forward<<<blocks(d.ny), THREADS>>>(x, w, b, out->data, d);
    check(cudaGetLastError());
    return out;
}

static void conv_backward(const float* x, const float* w, const float* g, ConvDims d, Buf& dx, Buf& dw, Buf& db) {
    dx = empty(d.nx); dw = empty(d.nw); db = empty(d.f);
    k_conv2d_dx<<<blocks(d.nx), THREADS>>>(w, g, dx->data, d);
    check(cudaGetLastError());
    k_conv2d_dw<<<blocks(d.nw), THREADS>>>(x, g, dw->data, d);
    check(cudaGetLastError());
    k_conv2d_db<<<blocks(d.f), THREADS>>>(g, db->data, d);
    check(cudaGetLastError());
}

EXPORT void* conv2d_forward_cuda_resident(void* x, void* w, void* b, int n, int c, int h, int width, int f, int kh, int kw, int sh, int sw, int ph, int pw) {
    CUDA_BEGIN
    ConvDims d(n, c, h, width, f, kh, kw, sh, sw, ph, pw);
    return conv_forward(ptr(x, d.nx), ptr(w, d.nw), b ? ptr(b, f) : nullptr, d).release();
    CUDA_END(nullptr)
}

EXPORT void conv2d_backward_cuda_resident(void* x, void* w, void* grad, int n, int c, int h, int width, int f, int kh, int kw, int sh, int sw, int ph, int pw, void** outDX, void** outDW, void** outDB) {
    CUDA_BEGIN
    if (!outDX || !outDW || !outDB || outDX == outDW || outDX == outDB || outDW == outDB)
        throw std::runtime_error("invalid CUDA output slots");
    *outDX = *outDW = *outDB = nullptr;
    ConvDims d(n, c, h, width, f, kh, kw, sh, sw, ph, pw);
    Buf dx, dw, db;
    conv_backward(ptr(x, d.nx), ptr(w, d.nw), ptr(grad, d.ny), d, dx, dw, db);
    *outDX = dx.release(); *outDW = dw.release(); *outDB = db.release();
    CUDA_END((void)0)
}

EXPORT void conv2d_forward_cuda(const float* x, const float* w, const float* b, float* out, int n, int c, int h, int width, int f, int kh, int kw, int sh, int sw, int ph, int pw) {
    CUDA_BEGIN
    ConvDims d(n, c, h, width, f, kh, kw, sh, sw, ph, pw);
    if (!x || !w || !out) throw std::runtime_error("null CNN input/output");
    auto X = upload(x, d.nx), W = upload(w, d.nw), B = b ? upload(b, f) : Buf();
    auto result = conv_forward(X->data, W->data, B ? B->data : nullptr, d);
    result->read(out, d.ny);
    CUDA_END((void)0)
}

EXPORT void conv2d_backward_cuda(const float* x, const float* w, const float* grad, float* outDX, float* outDW, float* outDB, int n, int c, int h, int width, int f, int kh, int kw, int sh, int sw, int ph, int pw) {
    CUDA_BEGIN
    ConvDims d(n, c, h, width, f, kh, kw, sh, sw, ph, pw);
    if (!x || !w || !grad || !outDX || !outDW || !outDB) throw std::runtime_error("null CNN input/output");
    auto X = upload(x, d.nx), W = upload(w, d.nw), G = upload(grad, d.ny);
    Buf dx, dw, db;
    conv_backward(X->data, W->data, G->data, d, dx, dw, db);
    dx->read(outDX, d.nx); dw->read(outDW, d.nw); db->read(outDB, f);
    CUDA_END((void)0)
}

struct PoolDims {
    int n, c, h, w, kh, kw, sh, sw, oh, ow, nx, ny;
    PoolDims(int N, int C, int H, int W, int KH, int KW, int SH, int SW)
        : n(N), c(C), h(H), w(W), kh(KH), kw(KW), sh(SH), sw(SW) {
        nx = count2(count2(n, c), count2(h, w));
        if (kh <= 0 || kw <= 0 || sh <= 0 || sw <= 0 || kh > h || kw > w)
            throw std::runtime_error("invalid maxpool kernel or stride");
        oh = (h - kh) / sh + 1; ow = (w - kw) / sw + 1;
        ny = count2(count2(n, c), count2(oh, ow));
    }
};

struct PoolCache {
    PoolDims dims;
    int* indices = nullptr;
    explicit PoolCache(PoolDims d) : dims(d) {
        check(cudaMalloc(reinterpret_cast<void**>(&indices), sizeof(int) * d.ny));
    }
    ~PoolCache() { if (indices) cudaFree(indices); }
    PoolCache(const PoolCache&) = delete;
    PoolCache& operator=(const PoolCache&) = delete;
};
using Pool = std::unique_ptr<PoolCache>;

__global__ void k_maxpool2d_forward(const float* x, float* out, int* indices, PoolDims d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.ny) return;
    int ox = i % d.ow, oy = (i / d.ow) % d.oh, plane = i / d.ow / d.oh;
    float best = -CUDART_INF_F;
    int winner = -1;
    for (int ky = 0; ky < d.kh; ky++) {
        for (int kx = 0; kx < d.kw; kx++) {
            int xi = (plane * d.h + oy * d.sh + ky) * d.w + ox * d.sw + kx;
            float value = x[xi];
            // First maximum (also handles an all -inf window); first NaN propagates.
            if (winner < 0 || value > best || (isnan(value) && !isnan(best))) {
                best = value; winner = xi;
            }
        }
    }
    out[i] = best; indices[i] = winner;
}

__global__ void k_maxpool2d_backward(const float* g, const int* indices, float* dx, PoolDims d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.nx) return;
    int ix = i % d.w, iy = (i / d.w) % d.h, plane = i / d.w / d.h;
    int y0 = iy < d.kh ? 0 : (iy - d.kh) / d.sh + 1;
    int x0 = ix < d.kw ? 0 : (ix - d.kw) / d.sw + 1;
    int y1 = min(iy / d.sh, d.oh - 1), x1 = min(ix / d.sw, d.ow - 1);
    float sum = 0.0f;
    for (int oy = y0; oy <= y1; oy++) {
        for (int ox = x0; ox <= x1; ox++) {
            int oi = (plane * d.oh + oy) * d.ow + ox;
            if (indices[oi] == i) sum += g[oi];
        }
    }
    dx[i] = sum;
}

static Buf pool_forward(const float* x, PoolCache& cache) {
    auto out = empty(cache.dims.ny);
    k_maxpool2d_forward<<<blocks(cache.dims.ny), THREADS>>>(x, out->data, cache.indices, cache.dims);
    check(cudaGetLastError());
    return out;
}

static Buf pool_backward(const float* grad, const PoolCache& cache) {
    auto dx = empty(cache.dims.nx);
    k_maxpool2d_backward<<<blocks(cache.dims.nx), THREADS>>>(grad, cache.indices, dx->data, cache.dims);
    check(cudaGetLastError());
    return dx;
}

EXPORT void maxpool2d_forward_cuda_resident(void* x, int n, int c, int h, int w, int kh, int kw, int sh, int sw, void** out, void** outCache) {
    CUDA_BEGIN
    if (!out || !outCache || out == outCache) throw std::runtime_error("invalid CUDA output slots");
    *out = *outCache = nullptr;
    PoolDims d(n, c, h, w, kh, kw, sh, sw);
    float* X = ptr(x, d.nx);
    Pool cache(new PoolCache(d));
    auto result = pool_forward(X, *cache);
    *out = result.release(); *outCache = cache.release();
    CUDA_END((void)0)
}

EXPORT void* maxpool2d_backward_cuda_resident(void* grad, void* handle) {
    CUDA_BEGIN
    if (!handle) throw std::runtime_error("null CUDA pool cache");
    auto& cache = *static_cast<PoolCache*>(handle);
    return pool_backward(ptr(grad, cache.dims.ny), cache).release();
    CUDA_END(nullptr)
}

EXPORT void camel_cuda_pool_cache_free(void* cache) { delete static_cast<PoolCache*>(cache); }

EXPORT void maxpool2d_forward_cuda(const float* x, float* out, int* indices, int n, int c, int h, int w, int kh, int kw, int sh, int sw) {
    CUDA_BEGIN
    PoolDims d(n, c, h, w, kh, kw, sh, sw);
    if (!x || !out || !indices) throw std::runtime_error("null maxpool input/output");
    auto X = upload(x, d.nx);
    PoolCache cache(d);
    auto result = pool_forward(X->data, cache);
    result->read(out, d.ny);
    check(cudaMemcpy(indices, cache.indices, sizeof(int) * d.ny, cudaMemcpyDeviceToHost));
    CUDA_END((void)0)
}

EXPORT void maxpool2d_backward_cuda(const float* grad, const int* indices, float* dx, int n, int c, int h, int w, int kh, int kw, int sh, int sw) {
    CUDA_BEGIN
    PoolDims d(n, c, h, w, kh, kw, sh, sw);
    if (!grad || !indices || !dx) throw std::runtime_error("null maxpool input/output");
    // Reject a stale/corrupt mask instead of silently dropping its gradients.
    for (int i = 0; i < d.ny; i++) {
        int plane = i / d.ow / d.oh;
        long long local = static_cast<long long>(indices[i]) - plane * d.h * d.w;
        long long y = local >= 0 ? local / d.w : -1, x = local >= 0 ? local % d.w : -1;
        int oy = (i / d.ow) % d.oh, ox = i % d.ow;
        if (y < oy * d.sh || y >= oy * d.sh + d.kh || x < ox * d.sw || x >= ox * d.sw + d.kw)
            throw std::runtime_error("maxpool index outside its window");
    }
    auto G = upload(grad, d.ny);
    PoolCache cache(d);
    check(cudaMemcpy(cache.indices, indices, sizeof(int) * d.ny, cudaMemcpyHostToDevice));
    pool_backward(G->data, cache)->read(dx, d.nx);
    CUDA_END((void)0)
}

EXPORT void* copy_cuda_resident(void* x, int total) {
    CUDA_BEGIN
    return copy(ptr(x, total), total).release();
    CUDA_END(nullptr)
}
