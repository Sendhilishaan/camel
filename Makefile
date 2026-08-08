CC = gcc
CFLAGS = -Wall -Wextra -O2 -Iinclude
LDFLAGS =

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

dll: $(SRC)
	$(CC) $(CFLAGS) -shared -o camel.dll $(SRC) $(LDFLAGS)

clean:
	rm -f camel.dll

.PHONY: dll clean
