import math

from camel import array as ca
from camel.ops import Vbuf
from camel.tensor import Tensor
from camel.nn import MLP
from camel.optim import SGD
from camel.ops import Ops

Ops.set_backend("metal")

# example training loop: fit sin with a small MLP, full-batch SGD with momentum
X = ca.linspace(-2*math.pi, 2*math.pi, 1000).reshape(-1, 1)
X = X / X.std()
y = ca.sin(X)

X_t = Tensor(Vbuf(X), requires_grad=False)
y_true = Tensor(Vbuf(y), requires_grad=False)

mlp = MLP(1, [32, 32, 32, 1])
opt = SGD(mlp.parameters(), 0.01, 0.9)

for i in range(1000):
    y_pred = mlp(X_t)
    diff = (y_pred - y_true)
    loss = (diff * diff).mean()
    if i % 100 == 0:
        print(f"epoch: {i} loss: {loss.buf.data.item()}")
    loss.backward()
    opt.step()
    opt.zero_grad()
