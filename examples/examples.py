import math
import random
import time

from camel import array as ca
from camel.data import DataLoader
from camel.ops import Ops, Vbuf
from camel.tensor import Tensor
from camel.nn import CNN, MLP
from camel.optim import Adam, SGD


BACKENDS = ["naive", "simd", "metal", "cuda"]
NAIVE_MAX_ELEMS = 512 * 512 # skip naive for the larger benchmark sizes


def cnn_demo(backend="naive"):
    # two 4x4 images, one forward/backward pass; backend="cuda" is the GPU alternative
    Ops.set_backend(backend)
    ca.random.seed(0)
    model = CNN(1, [2], 2, image_size=4)
    X = Tensor(Vbuf(ca.random.randn(2, 1, 4, 4)), requires_grad=False)
    y = Tensor(Vbuf(ca.CamelArray([[1, 0], [0, 1]])), requires_grad=False)
    logits = model(X)
    loss = logits.softmax_xent(y)
    loss.backward()
    print("logits:", logits.buf.shape)
    print("loss:", loss.buf.data.item())
    print("convolution weight gradient:", model.layers[0].W.grad.shape)
    print("classifier weight gradient:", model.head.W.grad.shape)


def make_bars(count, seed):
    rng = random.Random(seed)
    pixels, labels = [], []
    for i in range(count):
        label = i % 2 # balanced classes: 0 = horizontal, 1 = vertical
        position = rng.randint(1, 5)
        brightness = rng.uniform(0.8, 1.2)
        for row in range(8):
            for col in range(8):
                coordinate = row if label == 0 else col
                bar = brightness if position <= coordinate < position + 2 else 0.0
                pixels.append(bar + rng.gauss(0.0, 0.1))
        labels.append([1, 0] if label == 0 else [0, 1])
    return ca.CamelArray(pixels).reshape(count, 1, 8, 8), ca.CamelArray(labels)


def evaluate(model, images, labels):
    X = Tensor(Vbuf(images), requires_grad=False)
    y = Tensor(Vbuf(labels), requires_grad=False)
    logits = model(X)
    loss = logits.softmax_xent(y).buf.data.item()
    scores = logits.buf.data
    # larger logit is the predicted class, no softmax needed for binary accuracy
    correct = sum(int(scores[i, 1] > scores[i, 0]) == int(labels[i, 1])
                  for i in range(images.shape[0]))
    return loss, correct / images.shape[0]


def train_cnn(backend="naive", epochs=30, batch_size=8, lr=0.01):
    # classify noisy 8x8 bars; backend="cuda" uses the same model and training loop
    Ops.set_backend(backend)
    ca.random.seed(0) # model initialisation
    random.seed(0) # DataLoader shuffling
    train_x, train_y = make_bars(32, seed=1)
    val_x, val_y = make_bars(16, seed=2) # fresh noise and bar positions
    loader = DataLoader(train_x, train_y, batch_size, shuffle=True)
    model = CNN(1, [4], 2, image_size=8) # [4, 8] adds a second conv/pool block
    opt = Adam(model.parameters(), lr=lr) # alternatively SGD(model.parameters(), lr, 0.9)

    val_loss, accuracy = evaluate(model, val_x, val_y)
    print(f"backend: {backend}, train: 32 images, validation: 16 images")
    print(f"before training: val loss: {val_loss:.4f}, val accuracy: {accuracy:.1%}")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for images, labels in loader:
            X = Tensor(Vbuf(images), requires_grad=False)
            y = Tensor(Vbuf(labels), requires_grad=False)
            opt.zero_grad()
            loss = model(X).softmax_xent(y)
            total_loss += loss.buf.data.item() * images.shape[0]
            loss.backward()
            opt.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            # validation: forward only, no parameter updates
            val_loss, accuracy = evaluate(model, val_x, val_y)
            print(f"epoch: {epoch:2d}, train loss: {total_loss / train_x.shape[0]:.4f}, "
                  f"val loss: {val_loss:.4f}, val accuracy: {accuracy:.1%}")
    return model


def train_sin(backend="naive", epochs=1000):
    # full-batch MLP regression; alternatives: "simd"/"metal" on Mac, "cuda" on NVIDIA
    Ops.set_backend(backend)
    ca.random.seed(0)
    X = ca.linspace(-2*math.pi, 2*math.pi, 1000).reshape(-1, 1)
    X = X / X.std()
    y = ca.sin(X)
    X_t = Tensor(Vbuf(X), requires_grad=False)
    y_true = Tensor(Vbuf(y), requires_grad=False)

    mlp = MLP(1, [32, 32, 32, 1])
    opt = SGD(mlp.parameters(), 0.01, 0.9)

    for i in range(epochs):
        diff = mlp(X_t) - y_true
        loss = (diff * diff).mean()
        if i % 100 == 0 or i == epochs - 1:
            print(f"epoch: {i} loss: {loss.buf.data.item()}")
        loss.backward()
        opt.step()
        opt.zero_grad()
    Ops.synchronize()
    return mlp


def timeit(fn, reps: int) -> float:
    fn() # warm up, including Metal pipeline compilation
    Ops.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    Ops.synchronize() # CUDA launches are asynchronous; time completed work
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


def benchmark_row(label: str, backends: list, work_fn, reps: int) -> None:
    times = {}
    for backend in backends:
        if ((backend == "simd" and not Ops.simd_available()) or
            (backend == "metal" and not Ops.metal_device_available()) or
            (backend == "cuda" and not Ops.cuda_device_available())):
            times[backend] = None
            continue
        times[backend] = work_fn(backend, reps)
    print(f"{label:>24} " + " ".join(f"{fmt(times.get(b)):>13}" for b in BACKENDS))


def benchmark():
    # full size sweeps across available backends, including large GPU workloads
    print("matmul_forward(A, B), A:(n,k) B:(k,m)")
    print(f"{'size':>24} " + " ".join(f"{b:>13}" for b in BACKENDS))
    for n, k, m in [(64, 64, 64), (256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]:
        backends = BACKENDS if n * k <= NAIVE_MAX_ELEMS else BACKENDS[1:]
        reps = 5 if n <= 256 else 2
        benchmark_row(f"{n}x{k}x{m}", backends, lambda b, r, n=n, k=k, m=m: bench_matmul(b, n, k, m, r), reps)

    print()
    print("MLP train step: forward + backward + SGD step, `depth` hidden layers of `width`")
    print(f"{'batch/width/depth':>24} " + " ".join(f"{b:>13}" for b in BACKENDS))
    configs = [
        (64, 32, 2),
        (256, 128, 3),
        (512, 256, 3),
        (1024, 512, 4),
        (1024, 1024, 4),
    ]
    for batch, width, depth in configs:
        backends = BACKENDS if batch * width <= NAIVE_MAX_ELEMS else BACKENDS[1:]
        reps = 5 if batch <= 256 else 2
        benchmark_row(f"{batch}/{width}/{depth}", backends,
                      lambda b, r, batch=batch, width=width, depth=depth: bench_train_step(b, batch, width, depth, r), reps)


if __name__ == "__main__":
    # choose one example; CNNs currently support "naive" and "cuda"
    cnn_demo(backend="naive")
    # train_cnn(backend="naive", epochs=30) # alternatively backend="cuda"
    # train_sin(backend="naive", epochs=1000) # also "simd", "metal" or "cuda"
    # benchmark() # run separately when you want the full performance comparison
