CC = gcc
CFLAGS = -Wall -Wextra -O2 -Iinclude
LDFLAGS =
NVCC ?= nvcc
CUDA_ARCH ?= 120

# every C source that should go into the shared lib
SRC = src/ops/prim.c

# Apple's <simd/simd.h> backend (NEON on Apple Silicon, SSE/AVX on Intel Mac)
# and the Metal GPU backend are both Darwin-only; naive kernels above always
# build and remain the portable fallback everywhere else.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	SRC += src/ops/prim_simd.c src/ops/prim_metal.m
	CFLAGS += -fobjc-arc
	LDFLAGS += -framework Metal -framework Foundation
endif

ifeq ($(UNAME_S),Linux)
	CFLAGS += -fPIC
	LDFLAGS += -lm
endif

dll: $(SRC)
	$(CC) $(CFLAGS) -shared -o camel.dll $(SRC) $(LDFLAGS)

# CUDA is opt-in; defaults to RTX 50-series (sm_120, CUDA 12.8+).
cuda: src/ops/prim.c src/ops/prim_cuda.cu include/prim.h include/prim_cuda.h
	mkdir -p build
	$(CC) $(CFLAGS) -fPIC -c src/ops/prim.c -o build/prim.o
	$(NVCC) -O2 -std=c++17 -Iinclude -Xcompiler -fPIC -shared --cudart static \
		-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
		-gencode arch=compute_$(CUDA_ARCH),code=compute_$(CUDA_ARCH) \
		src/ops/prim_cuda.cu build/prim.o -o camel.dll

clean:
	rm -f camel.dll build/prim.o

.PHONY: dll cuda clean
