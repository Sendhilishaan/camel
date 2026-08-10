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

    Every kernel has two entry points: a copy-based one (float* in, float*
    out - stages through CPU on every call) and a resident one (void* GPU
    buffer handles in and out - for chaining several ops without leaving the
    GPU). Both call the same dispatch_* helper, which only ever produces a
    fresh output buffer and never mutates an input buffer, since a resident
    input's handle may be cached and shared elsewhere in the graph.
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

// GPU-to-GPU copy, for the "gradient is just the identity" cases (add/sub
// backward) where a resident output must be its own buffer, not an alias of
// the input, even though the values are identical.
static id<MTLBuffer> blit_copy(id<MTLBuffer> src, size_t count) {
    id<MTLBuffer> dst = buf_out(count);
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:sizeof(float) * count];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return dst;
}

/*
    opaque GPU buffer handles for the resident API: CFBridgingRetain hands a
    +1 reference to the caller as a plain void*, which camel_metal_buffer_free
    must eventually balance with CFBridgingRelease. camel_metal_buffer_read
    uses a non-owning __bridge cast since it doesn't take ownership.
*/
EXPORT void *camel_metal_buffer_create(const float *data, int count) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> buf = buf_in(data, (size_t)count);
        return (void *)CFBridgingRetain(buf);
    }
}

EXPORT void camel_metal_buffer_read(void *handle, float *out, int count) {
    @autoreleasepool {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)handle;
        memcpy(out, buf.contents, sizeof(float) * (size_t)count);
    }
}

EXPORT void camel_metal_buffer_free(void *handle) {
    if (!handle) return;
    @autoreleasepool {
        id<MTLBuffer> buf = (id<MTLBuffer>)CFBridgingRelease(handle);
        (void)buf; // ARC releases it at the end of this scope
    }
}

EXPORT int camel_metal_device_available(void) {
    if (g_device) return 1;
    @autoreleasepool {
        id<MTLDevice> probe = camel_default_device();
        return probe != nil;
    }
}

// ---- dispatch helpers: operate on existing MTLBuffers, always produce a fresh output ----

static id<MTLBuffer> dispatch_matmul_forward(id<MTLBuffer> A, id<MTLBuffer> B, int n, int k, int m) {
    id<MTLBuffer> out = buf_out((size_t)n * m);
    int dims[3] = { n, k, m };

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:get_pipeline(@"k_matmul_forward")];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:B offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    [enc setBytes:dims length:sizeof(dims) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(m, n, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return out;
}

static void dispatch_matmul_backward(id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> G, int n, int k, int m,
                                      id<MTLBuffer> *outDA, id<MTLBuffer> *outDB) {
    id<MTLBuffer> bufDA = buf_out((size_t)n * k);
    id<MTLBuffer> bufDB = buf_out((size_t)k * m);
    int dims[3] = { n, k, m };

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];

    id<MTLComputeCommandEncoder> enc1 = [cmd computeCommandEncoder];
    [enc1 setComputePipelineState:get_pipeline(@"k_matmul_backward_da")];
    [enc1 setBuffer:B offset:0 atIndex:0];
    [enc1 setBuffer:G offset:0 atIndex:1];
    [enc1 setBuffer:bufDA offset:0 atIndex:2];
    [enc1 setBytes:dims length:sizeof(dims) atIndex:3];
    [enc1 dispatchThreads:MTLSizeMake(k, n, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    [enc1 endEncoding];

    id<MTLComputeCommandEncoder> enc2 = [cmd computeCommandEncoder];
    [enc2 setComputePipelineState:get_pipeline(@"k_matmul_backward_db")];
    [enc2 setBuffer:A offset:0 atIndex:0];
    [enc2 setBuffer:G offset:0 atIndex:1];
    [enc2 setBuffer:bufDB offset:0 atIndex:2];
    [enc2 setBytes:dims length:sizeof(dims) atIndex:3];
    [enc2 dispatchThreads:MTLSizeMake(m, k, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    [enc2 endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];
    *outDA = bufDA;
    *outDB = bufDB;
}

// in-place: A[i,j] += B[j], matches the naive/simd contract used by the
// copy-based path (the caller already staged a private copy of A for it)
static void dispatch_matadd_broadcast_forward_inplace(id<MTLBuffer> A, id<MTLBuffer> B, int n, int m) {
    int dims[2] = { n, m };
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:get_pipeline(@"k_matadd_broadcast_forward")];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:B offset:0 atIndex:1];
    [enc setBytes:dims length:sizeof(dims) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(m, n, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

static id<MTLBuffer> dispatch_matadd_broadcast_forward_out(id<MTLBuffer> A, id<MTLBuffer> B, int n, int m) {
    id<MTLBuffer> out = buf_out((size_t)n * m);
    int dims[2] = { n, m };
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:get_pipeline(@"k_matadd_broadcast_forward_out")];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:B offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    [enc setBytes:dims length:sizeof(dims) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(m, n, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return out;
}

static void dispatch_matadd_broadcast_backward(id<MTLBuffer> G, int n, int m,
                                                id<MTLBuffer> *outDX, id<MTLBuffer> *outDB) {
    id<MTLBuffer> bufDB = buf_out((size_t)m);
    int dims[2] = { n, m };
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:get_pipeline(@"k_matadd_broadcast_backward_db")];
    [enc setBuffer:G offset:0 atIndex:0];
    [enc setBuffer:bufDB offset:0 atIndex:1];
    [enc setBytes:dims length:sizeof(dims) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(m, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(m, 64), 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    *outDX = blit_copy(G, (size_t)n * m); // add is identity, same as naive/simd
    *outDB = bufDB;
}

static id<MTLBuffer> dispatch_matsub_forward(id<MTLBuffer> A, id<MTLBuffer> B, int total) {
    id<MTLBuffer> out = buf_out(total);
    int dims[1] = { total };
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:get_pipeline(@"k_matsub_forward")];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:B offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    [enc setBytes:dims length:sizeof(dims) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(total, 64), 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return out;
}

// combines two same-shape gradient buffers on the GPU (Tensor._accum's
// fan-out case); no copy-based variant since only the resident path needs it
static id<MTLBuffer> dispatch_add(id<MTLBuffer> A, id<MTLBuffer> B, int total) {
    id<MTLBuffer> out = buf_out(total);
    int dims[1] = { total };
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:get_pipeline(@"k_add")];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:B offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    [enc setBytes:dims length:sizeof(dims) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(total, 64), 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return out;
}

static void dispatch_matsub_backward(id<MTLBuffer> G, int total, id<MTLBuffer> *outDA, id<MTLBuffer> *outDB) {
    id<MTLBuffer> bufDB = buf_out(total);
    int dims[1] = { total };
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:get_pipeline(@"k_negate")];
    [enc setBuffer:G offset:0 atIndex:0];
    [enc setBuffer:bufDB offset:0 atIndex:1];
    [enc setBytes:dims length:sizeof(dims) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(total, 64), 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    *outDA = blit_copy(G, total);
    *outDB = bufDB;
}

static id<MTLBuffer> dispatch_hadamard_forward(id<MTLBuffer> A, id<MTLBuffer> B, int total) {
    id<MTLBuffer> out = buf_out(total);
    int dims[1] = { total };
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:get_pipeline(@"k_hadamard_forward")];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:B offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    [enc setBytes:dims length:sizeof(dims) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(total, 64), 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return out;
}

static void dispatch_hadamard_backward(id<MTLBuffer> G, id<MTLBuffer> A, id<MTLBuffer> B, int total,
                                        id<MTLBuffer> *outDA, id<MTLBuffer> *outDB) {
    id<MTLBuffer> bufDA = buf_out(total);
    id<MTLBuffer> bufDB = buf_out(total);
    int dims[1] = { total };
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:get_pipeline(@"k_hadamard_backward")];
    [enc setBuffer:G offset:0 atIndex:0];
    [enc setBuffer:A offset:0 atIndex:1];
    [enc setBuffer:B offset:0 atIndex:2];
    [enc setBuffer:bufDA offset:0 atIndex:3];
    [enc setBuffer:bufDB offset:0 atIndex:4];
    [enc setBytes:dims length:sizeof(dims) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(total, 64), 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    *outDA = bufDA;
    *outDB = bufDB;
}

static id<MTLBuffer> dispatch_matmean_forward(id<MTLBuffer> A, int n) {
    id<MTLBuffer> out = buf_out(1);
    int dims[1] = { n };
    id<MTLComputePipelineState> pipeline = get_pipeline(@"k_matmean_forward");
    NSUInteger tgSize = pow2_leq((NSUInteger)n, pipeline.maxTotalThreadsPerThreadgroup);

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:dims length:sizeof(dims) atIndex:2];
    [enc setThreadgroupMemoryLength:sizeof(float) * tgSize atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return out;
}

static id<MTLBuffer> dispatch_matmean_backward(int n, float grad_out) {
    id<MTLBuffer> out = buf_out(n);
    int dims[1] = { n };
    float value = grad_out / (float)n;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:get_pipeline(@"k_fill")];
    [enc setBuffer:out offset:0 atIndex:0];
    [enc setBytes:dims length:sizeof(dims) atIndex:1];
    [enc setBytes:&value length:sizeof(value) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(n, 64), 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return out;
}

static id<MTLBuffer> dispatch_tanh_forward(id<MTLBuffer> Z, int total) {
    id<MTLBuffer> out = buf_out(total);
    int dims[1] = { total };
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:get_pipeline(@"k_tanh_forward")];
    [enc setBuffer:Z offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:dims length:sizeof(dims) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(total, 64), 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return out;
}

static id<MTLBuffer> dispatch_tanh_backward(id<MTLBuffer> Out, id<MTLBuffer> G, int total) {
    id<MTLBuffer> dz = buf_out(total);
    int dims[1] = { total };
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:get_pipeline(@"k_tanh_backward")];
    [enc setBuffer:Out offset:0 atIndex:0];
    [enc setBuffer:G offset:0 atIndex:1];
    [enc setBuffer:dz offset:0 atIndex:2];
    [enc setBytes:dims length:sizeof(dims) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(total, 64), 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return dz;
}

static id<MTLBuffer> dispatch_relu_forward(id<MTLBuffer> Z, int total) {
    id<MTLBuffer> out = buf_out(total);
    int dims[1] = { total };
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:get_pipeline(@"k_relu_forward")];
    [enc setBuffer:Z offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:dims length:sizeof(dims) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(total, 64), 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return out;
}

static id<MTLBuffer> dispatch_relu_backward(id<MTLBuffer> Out, id<MTLBuffer> G, int total) {
    id<MTLBuffer> dz = buf_out(total);
    int dims[1] = { total };
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:get_pipeline(@"k_relu_backward")];
    [enc setBuffer:Out offset:0 atIndex:0];
    [enc setBuffer:G offset:0 atIndex:1];
    [enc setBuffer:dz offset:0 atIndex:2];
    [enc setBytes:dims length:sizeof(dims) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(total, 64), 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return dz;
}

static float dispatch_softmax_xent_forward(id<MTLBuffer> Z, id<MTLBuffer> Y, int n, int m, id<MTLBuffer> *outProbs) {
    id<MTLBuffer> bufProbs = buf_out((size_t)n * m);
    id<MTLBuffer> bufRowLoss = buf_out((size_t)n);
    int dims[2] = { n, m };

    id<MTLComputePipelineState> pipeline = get_pipeline(@"k_softmax_xent_forward");
    NSUInteger tgSize = pow2_leq((NSUInteger)m, pipeline.maxTotalThreadsPerThreadgroup);

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:Z offset:0 atIndex:0];
    [enc setBuffer:Y offset:0 atIndex:1];
    [enc setBuffer:bufProbs offset:0 atIndex:2];
    [enc setBuffer:bufRowLoss offset:0 atIndex:3];
    [enc setBytes:dims length:sizeof(dims) atIndex:4];
    [enc setThreadgroupMemoryLength:sizeof(float) * tgSize atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    float *rowLoss = (float *)bufRowLoss.contents;
    float total = 0.0f;
    for (int i = 0; i < n; i++) total += rowLoss[i];
    *outProbs = bufProbs;
    return total / (float)n;
}

static id<MTLBuffer> dispatch_softmax_xent_backward(id<MTLBuffer> Probs, id<MTLBuffer> Y, float grad_out, int n, int total) {
    id<MTLBuffer> dZ = buf_out(total);
    int dims[1] = { total };
    float scale = grad_out / (float)n;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:get_pipeline(@"k_softmax_xent_backward")];
    [enc setBuffer:Probs offset:0 atIndex:0];
    [enc setBuffer:Y offset:0 atIndex:1];
    [enc setBuffer:dZ offset:0 atIndex:2];
    [enc setBytes:dims length:sizeof(dims) atIndex:3];
    [enc setBytes:&scale length:sizeof(scale) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(clampu(total, 64), 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return dZ;
}

// ---- copy-based EXPORT wrappers: float* in, float* out, stage through CPU ----

EXPORT void matmul_forward_metal(const float* A, const float* B, float* out, int n, int k, int m) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> result = dispatch_matmul_forward(buf_in(A, (size_t)n * k), buf_in(B, (size_t)k * m), n, k, m);
        memcpy(out, result.contents, sizeof(float) * n * m);
    }
}

EXPORT void matmul_backward_metal(const float* A, const float* B, const float* grad_out, float* da, float* db, int n, int k, int m) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> bufDA, bufDB;
        dispatch_matmul_backward(buf_in(A, (size_t)n * k), buf_in(B, (size_t)k * m), buf_in(grad_out, (size_t)n * m),
                                  n, k, m, &bufDA, &bufDB);
        memcpy(da, bufDA.contents, sizeof(float) * n * k);
        memcpy(db, bufDB.contents, sizeof(float) * k * m);
    }
}

EXPORT void matadd_broadcast_forward_metal(float* A, const float* B, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> bufA = buf_in(A, (size_t)n * m);
        dispatch_matadd_broadcast_forward_inplace(bufA, buf_in(B, (size_t)m), n, m);
        memcpy(A, bufA.contents, sizeof(float) * n * m);
    }
}

EXPORT void matadd_broadcast_backward_metal(const float* grad_out, float* dX, float* db, int n, int m) {
    ensure_metal();
    memcpy(dX, grad_out, sizeof(float) * n * m); // add is identity, same as naive/simd
    @autoreleasepool {
        id<MTLBuffer> bufDX, bufDB;
        dispatch_matadd_broadcast_backward(buf_in(grad_out, (size_t)n * m), n, m, &bufDX, &bufDB);
        memcpy(db, bufDB.contents, sizeof(float) * m);
    }
}

EXPORT void matsub_forward_metal(const float* A, const float* B, float* out, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        int total = n * m;
        id<MTLBuffer> result = dispatch_matsub_forward(buf_in(A, total), buf_in(B, total), total);
        memcpy(out, result.contents, sizeof(float) * total);
    }
}

EXPORT void matsub_backward_metal(const float* grad_out, float* dA, float* dB, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        int total = n * m;
        id<MTLBuffer> bufDA, bufDB;
        dispatch_matsub_backward(buf_in(grad_out, total), total, &bufDA, &bufDB);
        memcpy(dA, bufDA.contents, sizeof(float) * total);
        memcpy(dB, bufDB.contents, sizeof(float) * total);
    }
}

EXPORT void hadamard_forward_metal(const float* A, const float* B, float* out, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        int total = n * m;
        id<MTLBuffer> result = dispatch_hadamard_forward(buf_in(A, total), buf_in(B, total), total);
        memcpy(out, result.contents, sizeof(float) * total);
    }
}

EXPORT void hadamard_backward_metal(const float* grad_out, const float* A, const float* B, float* dA, float* dB, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        int total = n * m;
        id<MTLBuffer> bufDA, bufDB;
        dispatch_hadamard_backward(buf_in(grad_out, total), buf_in(A, total), buf_in(B, total), total, &bufDA, &bufDB);
        memcpy(dA, bufDA.contents, sizeof(float) * total);
        memcpy(dB, bufDB.contents, sizeof(float) * total);
    }
}

EXPORT void matmean_forward_metal(const float* A, float* out, int n) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> result = dispatch_matmean_forward(buf_in(A, n), n);
        memcpy(out, result.contents, sizeof(float));
    }
}

EXPORT void matmean_backward_metal(float* dx, int n, float grad_out) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> result = dispatch_matmean_backward(n, grad_out);
        memcpy(dx, result.contents, sizeof(float) * n);
    }
}

EXPORT void tanh_forward_metal(const float* Z, float* out, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        int total = n * m;
        id<MTLBuffer> result = dispatch_tanh_forward(buf_in(Z, total), total);
        memcpy(out, result.contents, sizeof(float) * total);
    }
}

EXPORT void tanh_backward_metal(const float* out, const float* grad_out, float* dz, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        int total = n * m;
        id<MTLBuffer> result = dispatch_tanh_backward(buf_in(out, total), buf_in(grad_out, total), total);
        memcpy(dz, result.contents, sizeof(float) * total);
    }
}

EXPORT void relu_forward_metal(const float* Z, float* out, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        int total = n * m;
        id<MTLBuffer> result = dispatch_relu_forward(buf_in(Z, total), total);
        memcpy(out, result.contents, sizeof(float) * total);
    }
}

EXPORT void relu_backward_metal(const float* out, const float* grad_out, float* dz, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        int total = n * m;
        id<MTLBuffer> result = dispatch_relu_backward(buf_in(out, total), buf_in(grad_out, total), total);
        memcpy(dz, result.contents, sizeof(float) * total);
    }
}

EXPORT void softmax_xent_forward_metal(const float* Z, const float* Y, float* probs, float* out_loss, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> bufProbs;
        *out_loss = dispatch_softmax_xent_forward(buf_in(Z, (size_t)n * m), buf_in(Y, (size_t)n * m), n, m, &bufProbs);
        memcpy(probs, bufProbs.contents, sizeof(float) * n * m);
    }
}

EXPORT void softmax_xent_backward_metal(const float* probs, const float* Y, float* dZ, float grad_out, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        int total = n * m;
        id<MTLBuffer> result = dispatch_softmax_xent_backward(buf_in(probs, total), buf_in(Y, total), grad_out, n, total);
        memcpy(dZ, result.contents, sizeof(float) * total);
    }
}

// ---- resident EXPORT wrappers: void* GPU handles in and out, no CPU round-trip ----

EXPORT void *matmul_forward_metal_resident(void *a, void *b, int n, int k, int m) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> result = dispatch_matmul_forward((__bridge id<MTLBuffer>)a, (__bridge id<MTLBuffer>)b, n, k, m);
        return (void *)CFBridgingRetain(result);
    }
}

EXPORT void matmul_backward_metal_resident(void *a, void *b, void *grad_out, int n, int k, int m, void **outDA, void **outDB) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> bufDA, bufDB;
        dispatch_matmul_backward((__bridge id<MTLBuffer>)a, (__bridge id<MTLBuffer>)b, (__bridge id<MTLBuffer>)grad_out,
                                  n, k, m, &bufDA, &bufDB);
        *outDA = (void *)CFBridgingRetain(bufDA);
        *outDB = (void *)CFBridgingRetain(bufDB);
    }
}

EXPORT void *matadd_broadcast_forward_metal_resident(void *a, void *b, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> result = dispatch_matadd_broadcast_forward_out((__bridge id<MTLBuffer>)a, (__bridge id<MTLBuffer>)b, n, m);
        return (void *)CFBridgingRetain(result);
    }
}

EXPORT void matadd_broadcast_backward_metal_resident(void *grad_out, int n, int m, void **outDX, void **outDB) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> bufDX, bufDB;
        dispatch_matadd_broadcast_backward((__bridge id<MTLBuffer>)grad_out, n, m, &bufDX, &bufDB);
        *outDX = (void *)CFBridgingRetain(bufDX);
        *outDB = (void *)CFBridgingRetain(bufDB);
    }
}

EXPORT void *matsub_forward_metal_resident(void *a, void *b, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> result = dispatch_matsub_forward((__bridge id<MTLBuffer>)a, (__bridge id<MTLBuffer>)b, n * m);
        return (void *)CFBridgingRetain(result);
    }
}

EXPORT void *add_metal_resident(void *a, void *b, int total) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> result = dispatch_add((__bridge id<MTLBuffer>)a, (__bridge id<MTLBuffer>)b, total);
        return (void *)CFBridgingRetain(result);
    }
}

EXPORT void matsub_backward_metal_resident(void *grad_out, int n, int m, void **outDA, void **outDB) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> bufDA, bufDB;
        dispatch_matsub_backward((__bridge id<MTLBuffer>)grad_out, n * m, &bufDA, &bufDB);
        *outDA = (void *)CFBridgingRetain(bufDA);
        *outDB = (void *)CFBridgingRetain(bufDB);
    }
}

EXPORT void *hadamard_forward_metal_resident(void *a, void *b, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> result = dispatch_hadamard_forward((__bridge id<MTLBuffer>)a, (__bridge id<MTLBuffer>)b, n * m);
        return (void *)CFBridgingRetain(result);
    }
}

EXPORT void hadamard_backward_metal_resident(void *grad_out, void *a, void *b, int n, int m, void **outDA, void **outDB) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> bufDA, bufDB;
        dispatch_hadamard_backward((__bridge id<MTLBuffer>)grad_out, (__bridge id<MTLBuffer>)a, (__bridge id<MTLBuffer>)b,
                                    n * m, &bufDA, &bufDB);
        *outDA = (void *)CFBridgingRetain(bufDA);
        *outDB = (void *)CFBridgingRetain(bufDB);
    }
}

EXPORT void *matmean_forward_metal_resident(void *a, int n) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> result = dispatch_matmean_forward((__bridge id<MTLBuffer>)a, n);
        return (void *)CFBridgingRetain(result);
    }
}

EXPORT void *matmean_backward_metal_resident(int n, float grad_out) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> result = dispatch_matmean_backward(n, grad_out);
        return (void *)CFBridgingRetain(result);
    }
}

EXPORT void *tanh_forward_metal_resident(void *z, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> result = dispatch_tanh_forward((__bridge id<MTLBuffer>)z, n * m);
        return (void *)CFBridgingRetain(result);
    }
}

EXPORT void *tanh_backward_metal_resident(void *out, void *grad_out, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> result = dispatch_tanh_backward((__bridge id<MTLBuffer>)out, (__bridge id<MTLBuffer>)grad_out, n * m);
        return (void *)CFBridgingRetain(result);
    }
}

EXPORT void *relu_forward_metal_resident(void *z, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> result = dispatch_relu_forward((__bridge id<MTLBuffer>)z, n * m);
        return (void *)CFBridgingRetain(result);
    }
}

EXPORT void *relu_backward_metal_resident(void *out, void *grad_out, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> result = dispatch_relu_backward((__bridge id<MTLBuffer>)out, (__bridge id<MTLBuffer>)grad_out, n * m);
        return (void *)CFBridgingRetain(result);
    }
}

EXPORT void softmax_xent_forward_metal_resident(void *z, void *y, int n, int m, void **outProbs, float *outLoss) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> bufProbs;
        *outLoss = dispatch_softmax_xent_forward((__bridge id<MTLBuffer>)z, (__bridge id<MTLBuffer>)y, n, m, &bufProbs);
        *outProbs = (void *)CFBridgingRetain(bufProbs);
    }
}

EXPORT void *softmax_xent_backward_metal_resident(void *probs, void *y, float grad_out, int n, int m) {
    ensure_metal();
    @autoreleasepool {
        id<MTLBuffer> result = dispatch_softmax_xent_backward((__bridge id<MTLBuffer>)probs, (__bridge id<MTLBuffer>)y,
                                                                grad_out, n, n * m);
        return (void *)CFBridgingRetain(result);
    }
}
