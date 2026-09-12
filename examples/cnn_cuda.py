"""Tiny CNN forward/backward example: two 4x4 images, no training loop."""
from camel import array as ca
from camel.ops import Ops, Vbuf
from camel.tensor import Tensor
from camel.nn import CNN


def main():
    Ops.set_backend("cuda")
    ca.random.seed(0)
    model = CNN(1, [2], 2, image_size=4)
    x = Tensor(Vbuf(ca.random.randn(2, 1, 4, 4)), requires_grad=False)
    labels = Tensor(Vbuf(ca.CamelArray([[1, 0], [0, 1]])), requires_grad=False)
    logits = model(x)
    loss = logits.softmax_xent(labels)
    loss.backward()
    print("logits:", logits.buf.shape)
    print("loss:", loss.buf.data.item())
    print("convolution weight gradient:", model.layers[0].W.grad.shape)
    print("classifier weight gradient:", model.head.W.grad.shape)


if __name__ == "__main__":
    main()
