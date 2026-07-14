### camel

A from-scratch autograd deep learning library, with the numerical kernels written in C (naive now, SIMD → CUDA later) and the autograd engine, models, and training layered in Python on top.

**Currently:** a set of C primitive kernels — matmul, bias-add, subtract, hadamard, mean, and tanh, each with a hand-written forward and backward — all verified against numerical gradients. On top sits a reverse-mode autograd engine: a define-by-run Tensor that records a computation graph and backpropagates through it, enough to build and differentiate a small MLP.

**Next:** optimizers (SGD → Adam), Linear/Module layers and training loops, then SIMD and CUDA backends behind the same kernel interface, and later softmax/cross-entropy, CNNs, and attention.

This is inspired by [tinygrad](https://github.com/tinygrad/tinygrad) for the small, composable core, and PyTorch's torch.autograd.Function for the op structure, every op is a class with forward/backward staticmethods and a Context (ctx) that saves what the backward pass will need.
