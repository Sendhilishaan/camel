#include "prim_metal.h"
#include <string.h>
#include <dlfcn.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

/*
    kernel source lives in prim.metal (real MSL, not a C string): matmul uses
    the (i, l, j) loop order from prim_simd.c so each thread's inner loop is a
    plain sum (no SIMD needed, the GPU already runs one thread per output
    element in parallel). Reductions (mean, softmax row max/sum) use one
    threadgroup with a shared-memory tree reduction, so threadsPerThreadgroup
    must be a power of 2.
*/
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLLibrary> g_library = nil;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *g_pipelines = nil;

// MTLCreateSystemDefaultDevice() is documented as unsupported for
// non-interactive (commandline/daemon) processes - MTLCopyAllDevices() is
// the supported way to get a device from a CLI tool like this one.
static id<MTLDevice> camel_default_device(void) {
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    return devices.firstObject;
}

// prim.metal sits next to wherever camel.dll ends up (src/ops/prim.metal,
// relative to the project root); dladdr finds camel.dll's own path on disk
// so this works regardless of the caller's current working directory.
static NSString *camel_metal_source_path(void) {
    Dl_info info;
    if (dladdr((void *)&camel_default_device, &info) == 0 || !info.dli_fname) {
        return nil;
    }
    NSString *dllPath = [NSString stringWithUTF8String:info.dli_fname];
    NSString *projectRoot = [dllPath stringByDeletingLastPathComponent];
    return [projectRoot stringByAppendingPathComponent:@"src/ops/prim.metal"];
}

static void ensure_metal(void) {
    if (g_device) return;
    g_device = camel_default_device();
    if (!g_device) {
        fprintf(stderr, "camel metal: no Metal device found (MTLCopyAllDevices() was empty)\n");
        abort();
    }
    g_queue = [g_device newCommandQueue];

    NSString *sourcePath = camel_metal_source_path();
    NSError *error = nil;
    NSData *sourceData = sourcePath ? [NSData dataWithContentsOfFile:sourcePath options:0 error:&error] : nil;
    NSString *source = sourceData ? [[NSString alloc] initWithData:sourceData encoding:NSUTF8StringEncoding] : nil;
    if (!source) {
        fprintf(stderr, "camel metal: couldn't read kernel source at %s: %s\n",
                sourcePath ? sourcePath.UTF8String : "(unknown path)",
                error ? error.localizedDescription.UTF8String : "not found");
        abort();
    }

    g_library = [g_device newLibraryWithSource:source options:nil error:&error];
    if (!g_library) {
        fprintf(stderr, "camel metal: kernel source failed to compile: %s\n", error.localizedDescription.UTF8String);
        abort();
    }
    g_pipelines = [NSMutableDictionary dictionary];
}

// pipelines are compiled once per kernel name and cached for reuse
static id<MTLComputePipelineState> get_pipeline(NSString *name) {
    id<MTLComputePipelineState> pipeline = g_pipelines[name];
    if (pipeline) return pipeline;

    id<MTLFunction> fn = [g_library newFunctionWithName:name];
    NSError *error = nil;
    pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "camel metal: no pipeline for %s: %s\n", name.UTF8String, error.localizedDescription.UTF8String);
        abort();
    }
    g_pipelines[name] = pipeline;
    return pipeline;
}

static id<MTLBuffer> buf_in(const float *data, size_t count) {
    return [g_device newBufferWithBytes:data length:sizeof(float) * count options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> buf_out(size_t count) {
    return [g_device newBufferWithLength:sizeof(float) * count options:MTLResourceStorageModeShared];
}

static NSUInteger clampu(NSUInteger v, NSUInteger cap) {
    return v < cap ? v : cap;
}

// largest power of 2 that is <= both n and cap - shared-memory reductions need this
static NSUInteger pow2_leq(NSUInteger n, NSUInteger cap) {
    NSUInteger size = 1;
    while (size * 2 <= n && size * 2 <= cap) size *= 2;
    return size;
}

EXPORT void matmul_forward_metal(const float* A, const float* B, float* out, int n, int k, int m) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> bufA = buf_in(A, (size_t)n * k);
        id<MTLBuffer> bufB = buf_in(B, (size_t)k * m);
        id<MTLBuffer> bufOut = buf_out((size_t)n * m);
        int dims[3] = { n, k, m };

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:get_pipeline(@"k_matmul_forward")];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufOut offset:0 atIndex:2];
        [enc setBytes:dims length:sizeof(dims) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(m, n, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(out, bufOut.contents, sizeof(float) * n * m);
    }
}

EXPORT void matmul_backward_metal(const float* A, const float* B, const float* grad_out, float* da, float* db, int n, int k, int m) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> bufA = buf_in(A, (size_t)n * k);
        id<MTLBuffer> bufB = buf_in(B, (size_t)k * m);
        id<MTLBuffer> bufG = buf_in(grad_out, (size_t)n * m);
        id<MTLBuffer> bufDA = buf_out((size_t)n * k);
        id<MTLBuffer> bufDB = buf_out((size_t)k * m);
        int dims[3] = { n, k, m };

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];

        id<MTLComputeCommandEncoder> enc1 = [cmd computeCommandEncoder];
        [enc1 setComputePipelineState:get_pipeline(@"k_matmul_backward_da")];
        [enc1 setBuffer:bufB offset:0 atIndex:0];
        [enc1 setBuffer:bufG offset:0 atIndex:1];
        [enc1 setBuffer:bufDA offset:0 atIndex:2];
        [enc1 setBytes:dims length:sizeof(dims) atIndex:3];
        [enc1 dispatchThreads:MTLSizeMake(k, n, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        [enc1 endEncoding];

        id<MTLComputeCommandEncoder> enc2 = [cmd computeCommandEncoder];
        [enc2 setComputePipelineState:get_pipeline(@"k_matmul_backward_db")];
        [enc2 setBuffer:bufA offset:0 atIndex:0];
        [enc2 setBuffer:bufG offset:0 atIndex:1];
        [enc2 setBuffer:bufDB offset:0 atIndex:2];
        [enc2 setBytes:dims length:sizeof(dims) atIndex:3];
        [enc2 dispatchThreads:MTLSizeMake(m, k, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        [enc2 endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(da, bufDA.contents, sizeof(float) * n * k);
        memcpy(db, bufDB.contents, sizeof(float) * k * m);
    }
}

EXPORT void matadd_broadcast_forward_metal(float* A, const float* B, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> bufA = buf_in(A, (size_t)n * m);
        id<MTLBuffer> bufB = buf_in(B, (size_t)m);
        int dims[2] = { n, m };

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:get_pipeline(@"k_matadd_broadcast_forward")];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBytes:dims length:sizeof(dims) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(m, n, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(A, bufA.contents, sizeof(float) * n * m);
    }
}

EXPORT void matadd_broadcast_backward_metal(const float* grad_out, float* dX, float* db, int n, int m) {
    ensure_metal();
    memcpy(dX, grad_out, sizeof(float) * n * m); // add is identity, same as naive/simd
    @autoreleasepool {
        id<MTLBuffer> bufG = buf_in(grad_out, (size_t)n * m);
        id<MTLBuffer> bufDB = buf_out((size_t)m);
        int dims[2] = { n, m };

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:get_pipeline(@"k_matadd_broadcast_backward_db")];
        [enc setBuffer:bufG offset:0 atIndex:0];
        [enc setBuffer:bufDB offset:0 atIndex:1];
        [enc setBytes:dims length:sizeof(dims) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(m, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(m, 64), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(db, bufDB.contents, sizeof(float) * m);
    }
}

EXPORT void matsub_forward_metal(const float* A, const float* B, float* out, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        int total = n * m;
        id<MTLBuffer> bufA = buf_in(A, total);
        id<MTLBuffer> bufB = buf_in(B, total);
        id<MTLBuffer> bufOut = buf_out(total);
        int dims[1] = { total };

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:get_pipeline(@"k_matsub_forward")];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufOut offset:0 atIndex:2];
        [enc setBytes:dims length:sizeof(dims) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(total, 64), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(out, bufOut.contents, sizeof(float) * total);
    }
}

EXPORT void matsub_backward_metal(const float* grad_out, float* dA, float* dB, int n, int m) {
    ensure_metal();
    int total = n * m;
    memcpy(dA, grad_out, sizeof(float) * total);
    @autoreleasepool {
        id<MTLBuffer> bufG = buf_in(grad_out, total);
        id<MTLBuffer> bufDB = buf_out(total);
        int dims[1] = { total };

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:get_pipeline(@"k_negate")];
        [enc setBuffer:bufG offset:0 atIndex:0];
        [enc setBuffer:bufDB offset:0 atIndex:1];
        [enc setBytes:dims length:sizeof(dims) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(total, 64), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(dB, bufDB.contents, sizeof(float) * total);
    }
}

EXPORT void hadamard_forward_metal(const float* A, const float* B, float* out, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        int total = n * m;
        id<MTLBuffer> bufA = buf_in(A, total);
        id<MTLBuffer> bufB = buf_in(B, total);
        id<MTLBuffer> bufOut = buf_out(total);
        int dims[1] = { total };

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:get_pipeline(@"k_hadamard_forward")];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufOut offset:0 atIndex:2];
        [enc setBytes:dims length:sizeof(dims) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(total, 64), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(out, bufOut.contents, sizeof(float) * total);
    }
}

EXPORT void hadamard_backward_metal(const float* grad_out, const float* A, const float* B, float* dA, float* dB, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        int total = n * m;
        id<MTLBuffer> bufG = buf_in(grad_out, total);
        id<MTLBuffer> bufA = buf_in(A, total);
        id<MTLBuffer> bufB = buf_in(B, total);
        id<MTLBuffer> bufDA = buf_out(total);
        id<MTLBuffer> bufDB = buf_out(total);
        int dims[1] = { total };

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:get_pipeline(@"k_hadamard_backward")];
        [enc setBuffer:bufG offset:0 atIndex:0];
        [enc setBuffer:bufA offset:0 atIndex:1];
        [enc setBuffer:bufB offset:0 atIndex:2];
        [enc setBuffer:bufDA offset:0 atIndex:3];
        [enc setBuffer:bufDB offset:0 atIndex:4];
        [enc setBytes:dims length:sizeof(dims) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(total, 64), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(dA, bufDA.contents, sizeof(float) * total);
        memcpy(dB, bufDB.contents, sizeof(float) * total);
    }
}

EXPORT void matmean_forward_metal(const float* A, float* out, int n) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> bufA = buf_in(A, n);
        id<MTLBuffer> bufOut = buf_out(1);
        int dims[1] = { n };

        id<MTLComputePipelineState> pipeline = get_pipeline(@"k_matmean_forward");
        NSUInteger tgSize = pow2_leq((NSUInteger)n, pipeline.maxTotalThreadsPerThreadgroup);

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufOut offset:0 atIndex:1];
        [enc setBytes:dims length:sizeof(dims) atIndex:2];
        [enc setThreadgroupMemoryLength:sizeof(float) * tgSize atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(out, bufOut.contents, sizeof(float));
    }
}

EXPORT void matmean_backward_metal(float* dx, int n, float grad_out) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> bufOut = buf_out(n);
        int dims[1] = { n };
        float value = grad_out / (float)n;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:get_pipeline(@"k_fill")];
        [enc setBuffer:bufOut offset:0 atIndex:0];
        [enc setBytes:dims length:sizeof(dims) atIndex:1];
        [enc setBytes:&value length:sizeof(value) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(n, 64), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(dx, bufOut.contents, sizeof(float) * n);
    }
}

EXPORT void tanh_forward_metal(const float* Z, float* out, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        int total = n * m;
        id<MTLBuffer> bufZ = buf_in(Z, total);
        id<MTLBuffer> bufOut = buf_out(total);
        int dims[1] = { total };

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:get_pipeline(@"k_tanh_forward")];
        [enc setBuffer:bufZ offset:0 atIndex:0];
        [enc setBuffer:bufOut offset:0 atIndex:1];
        [enc setBytes:dims length:sizeof(dims) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(total, 64), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(out, bufOut.contents, sizeof(float) * total);
    }
}

EXPORT void tanh_backward_metal(const float* out, const float* grad_out, float* dz, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        int total = n * m;
        id<MTLBuffer> bufOut = buf_in(out, total);
        id<MTLBuffer> bufG = buf_in(grad_out, total);
        id<MTLBuffer> bufDZ = buf_out(total);
        int dims[1] = { total };

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:get_pipeline(@"k_tanh_backward")];
        [enc setBuffer:bufOut offset:0 atIndex:0];
        [enc setBuffer:bufG offset:0 atIndex:1];
        [enc setBuffer:bufDZ offset:0 atIndex:2];
        [enc setBytes:dims length:sizeof(dims) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(total, 64), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(dz, bufDZ.contents, sizeof(float) * total);
    }
}

EXPORT void relu_forward_metal(const float* Z, float* out, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        int total = n * m;
        id<MTLBuffer> bufZ = buf_in(Z, total);
        id<MTLBuffer> bufOut = buf_out(total);
        int dims[1] = { total };

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:get_pipeline(@"k_relu_forward")];
        [enc setBuffer:bufZ offset:0 atIndex:0];
        [enc setBuffer:bufOut offset:0 atIndex:1];
        [enc setBytes:dims length:sizeof(dims) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(total, 64), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(out, bufOut.contents, sizeof(float) * total);
    }
}

EXPORT void relu_backward_metal(const float* out, const float* grad_out, float* dz, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        int total = n * m;
        id<MTLBuffer> bufOut = buf_in(out, total);
        id<MTLBuffer> bufG = buf_in(grad_out, total);
        id<MTLBuffer> bufDZ = buf_out(total);
        int dims[1] = { total };

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:get_pipeline(@"k_relu_backward")];
        [enc setBuffer:bufOut offset:0 atIndex:0];
        [enc setBuffer:bufG offset:0 atIndex:1];
        [enc setBuffer:bufDZ offset:0 atIndex:2];
        [enc setBytes:dims length:sizeof(dims) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(total, 64), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(dz, bufDZ.contents, sizeof(float) * total);
    }
}

EXPORT void softmax_xent_forward_metal(const float* Z, const float* Y, float* probs, float* out_loss, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> bufZ = buf_in(Z, (size_t)n * m);
        id<MTLBuffer> bufY = buf_in(Y, (size_t)n * m);
        id<MTLBuffer> bufProbs = buf_out((size_t)n * m);
        id<MTLBuffer> bufRowLoss = buf_out((size_t)n);
        int dims[2] = { n, m };

        id<MTLComputePipelineState> pipeline = get_pipeline(@"k_softmax_xent_forward");
        NSUInteger tgSize = pow2_leq((NSUInteger)m, pipeline.maxTotalThreadsPerThreadgroup);

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufZ offset:0 atIndex:0];
        [enc setBuffer:bufY offset:0 atIndex:1];
        [enc setBuffer:bufProbs offset:0 atIndex:2];
        [enc setBuffer:bufRowLoss offset:0 atIndex:3];
        [enc setBytes:dims length:sizeof(dims) atIndex:4];
        [enc setThreadgroupMemoryLength:sizeof(float) * tgSize atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(probs, bufProbs.contents, sizeof(float) * n * m);

        float *rowLoss = (float *)bufRowLoss.contents;
        float total = 0.0f;
        for (int i = 0; i < n; i++) total += rowLoss[i];
        *out_loss = total / (float)n;
    }
}

EXPORT int camel_metal_device_available(void) {
    if (g_device) return 1;
    @autoreleasepool {
        id<MTLDevice> probe = camel_default_device();
        return probe != nil;
    }
}

EXPORT void softmax_xent_backward_metal(const float* probs, const float* Y, float* dZ, float grad_out, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        int total = n * m;
        id<MTLBuffer> bufProbs = buf_in(probs, total);
        id<MTLBuffer> bufY = buf_in(Y, total);
        id<MTLBuffer> bufDZ = buf_out(total);
        int dims[1] = { total };
        float scale = grad_out / (float)n;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:get_pipeline(@"k_softmax_xent_backward")];
        [enc setBuffer:bufProbs offset:0 atIndex:0];
        [enc setBuffer:bufY offset:0 atIndex:1];
        [enc setBuffer:bufDZ offset:0 atIndex:2];
        [enc setBytes:dims length:sizeof(dims) atIndex:3];
        [enc setBytes:&scale length:sizeof(scale) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(total, 64), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(dZ, bufDZ.contents, sizeof(float) * total);
    }
}
