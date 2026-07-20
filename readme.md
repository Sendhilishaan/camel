### camel

A from-scratch autograd deep learning library, with the numerical kernels written in C (naive now, SIMD + CUDA later) and the autograd engine, models, and training layered in Python on top.

**Currently:** a set of C primitive kernels: matmul, bias-add, subtract, hadamard, mean, tanh, and relu, each with a hand-written forward and backward, all verified against numerical gradients. On top sits a reverse-mode autograd engine: a define-by-run Tensor that records a computation graph and backpropagates through it. Above that, a small nn layer (Module, Linear with He init, MLP) and a full optimizer suite: SGD with momentum, AdaGrad, RMSprop, and Adam; Enough to build, differentiate, and actually train a small MLP end to end.

**Next:** softmax and cross-entropy for classification, which needs three new primitives (exp, log, and an axis-wise sum reduction), then mini-batching and a real dataset (MNIST), then SIMD and CUDA backends behind the same kernel interface, and later CNNs and attention.

This is inspired by [tinygrad](https://github.com/tinygrad/tinygrad) for the small, composable core, and PyTorch's torch.autograd.Function for the op structure, every op is a class with forward/backward staticmethods and a Context (ctx) that saves what the backward pass will need.
