"""
Benchmarks naive vs Apple SIMD vs Metal on this machine: a raw matmul sweep
first, then full MLP forward+backward+optimizer-step timing at increasing
network sizes. Metal skips gracefully if there's no accessible GPU device.

Naive gets impractically slow at the larger sizes (that's the whole reason
SIMD and Metal backends exist), so it's dropped past a size where a single
step would take too long to be worth waiting for.
"""
import time

from camel import array as ca
from camel.ops import Vbuf, Ops
from camel.tensor import Tensor
from camel.nn import MLP
from camel.optim import SGD

NAIVE_MAX_ELEMS = 512 * 512  # n*k for matmul, batch*width for training - naive cutoff


def timeit(fn, reps: int) -> float:
    fn() # warm up: compiles metal pipelines, primes pow2 threadgroup sizing etc.
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def bench_matmul(backend: str, n: int, k: int, m: int, reps: int) -> float:
    Ops.set_backend(backend)
    ca.random.seed(0)
    A = Vbuf(ca.random.randn(n, k))
    B = Vbuf(ca.random.randn(k, m))
    return timeit(lambda: Ops.matmul_forward(A, B), reps)


def bench_train_step(backend: str, batch: int, width: int, depth: int, reps: int) -> float:
    Ops.set_backend(backend)
    ca.random.seed(0)
    X = Tensor(Vbuf(ca.random.randn(batch, width)), requires_grad=False)
    y = Tensor(Vbuf(ca.random.randn(batch, 1)), requires_grad=False)
    mlp = MLP(width, [width] * depth + [1])
    opt = SGD(mlp.parameters(), 0.01, 0.9)

    def step():
        diff = mlp(X) - y
        loss = (diff * diff).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

    return timeit(step, reps)


def fmt(t: float | None) -> str:
    return f"{t * 1000:>9.2f}ms" if t is not None else f"{'skipped':>11}"


def run(label: str, backends: list, work_fn, reps: int) -> None:
    times = {}
    for backend in backends:
        if backend == "metal" and not Ops.metal_device_available():
            times[backend] = None
            continue
        times[backend] = work_fn(backend, reps)
    print(f"{label:>24} {fmt(times.get('naive')):>13} {fmt(times.get('simd')):>13} {fmt(times.get('metal')):>13}")


def main():
    print("matmul_forward(A, B), A:(n,k) B:(k,m)")
    print(f"{'size':>24} {'naive':>13} {'simd':>13} {'metal':>13}")
    for n, k, m in [(64, 64, 64), (256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]:
        backends = ["naive", "simd", "metal"] if n * k <= NAIVE_MAX_ELEMS else ["simd", "metal"]
        reps = 5 if n <= 256 else 2
        run(f"{n}x{k}x{m}", backends, lambda b, r, n=n, k=k, m=m: bench_matmul(b, n, k, m, r), reps)

    print()
    print("MLP train step: forward + backward + SGD step, `depth` hidden layers of `width`")
    print(f"{'batch/width/depth':>24} {'naive':>13} {'simd':>13} {'metal':>13}")
    configs = [
        (64, 32, 2),
        (256, 128, 3),
        (512, 256, 3),
        (1024, 512, 4),
        (1024, 1024, 4),
    ]
    for batch, width, depth in configs:
        backends = ["naive", "simd", "metal"] if batch * width <= NAIVE_MAX_ELEMS else ["simd", "metal"]
        reps = 5 if batch <= 256 else 2
        run(f"{batch}/{width}/{depth}", backends, lambda b, r, batch=batch, width=width, depth=depth: bench_train_step(b, batch, width, depth, r), reps)


if __name__ == "__main__":
    main()
