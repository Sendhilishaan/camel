CC = gcc
CFLAGS = -Wall -Wextra -O2 -Iinclude

# every C source that should go into the shared lib
SRC = src/ops/prim.c

dll: $(SRC)
	$(CC) $(CFLAGS) -shared -o camel.dll $(SRC)

clean:
	del camel.dll

.PHONY: dll clean
