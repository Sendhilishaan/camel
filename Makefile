CC = gcc
CFLAGS = -Wall -Wextra -O2 -Iinclude

# every C source that should go into the shared lib
SRC = src/ops/prim.c

# Apple's <simd/simd.h> backend is part of the macOS SDK (NEON on Apple
# Silicon, SSE/AVX on Intel Mac); only compiled in on Darwin, naive kernels
# above always build and remain the portable fallback everywhere else.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	SRC += src/ops/prim_simd.c
endif

dll: $(SRC)
	$(CC) $(CFLAGS) -shared -o camel.dll $(SRC)

clean:
	rm -f camel.dll

.PHONY: dll clean
