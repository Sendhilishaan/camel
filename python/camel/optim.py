from __future__ import annotations
from camel.tensor import Tensor
from typing import Iterable
from camel import array as ca

class Optimiser:
    def __init__(self, tensors: Iterable[Tensor], lr: float):
        self.lr = lr
        self.tensors = list(tensors)

    def zero_grad(self):
        for p in self.tensors:
            p.grad = None

    def step(self):
        raise NotImplementedError

class SGD(Optimiser):
    def __init__(self, tensors: Iterable[Tensor], lr: float, momentum=0.0):
        super().__init__(tensors, lr)
        self.momentum = momentum
        self.velocity = [ca.zeros_like(p.buf.data) for p in self.tensors]

    def step(self):
        for p, v in zip(self.tensors, self.velocity):
            v *= self.momentum
            v += p.grad.data
            p.buf.data -= v * self.lr

class AdaGrad(Optimiser):
    def __init__(self, tensors: Iterable[Tensor], lr: float, eps=1e-8):
        super().__init__(tensors, lr)
        self.eps = eps
        self.grad_sum = [ca.zeros_like(p.buf.data) for p in self.tensors]

    def step(self):
        for p, G in zip(self.tensors, self.grad_sum):
            G += ca.square(p.grad.data) # sum of second moment
            p.buf.data -= (self.lr * p.grad.data) / (ca.sqrt(G) + self.eps)

class RMSprop(Optimiser):
    def __init__(self, tensors: Iterable[Tensor], lr: float, eps=1e-8, decay=0.9):
        super().__init__(tensors, lr)
        self.eps = eps
        self.decay = decay
        self.ema_sq = [ca.zeros_like(p.buf.data) for p in self.tensors]

    def step(self):
        for p, s in zip(self.tensors, self.ema_sq):
            s *= self.decay
            s += (1 - self.decay) * ca.square(p.grad.data)
            # (1 - b) normalisation, weights sum to 1 making it a true average
            p.buf.data -= (self.lr * p.grad.data) / (ca.sqrt(s) + self.eps)

class Adam(Optimiser):
    def __init__(self, tensors: Iterable[Tensor], lr: float, decay1=0.9, decay2=0.999, eps=1e-8):
        super().__init__(tensors, lr)
        self.decay1 = decay1 # decay for 1st moment (grad, dir)
        self.decay2 = decay2 # decay for 2nd moment (grad^2, mag)
        self.eps = eps
        self.t = 0 # timestep

        self.exp_avg = [ca.zeros_like(p.buf.data) for p in self.tensors]
        self.exp_avg_sq = [ca.zeros_like(p.buf.data) for p in self.tensors]

    def step(self):
        self.t += 1
        for p, m, v in zip(self.tensors, self.exp_avg, self.exp_avg_sq):
            m *= self.decay1
            m += (1 - self.decay1) * p.grad.data
            m_hat = m / (1 - (self.decay1 ** self.t))

            v *= self.decay2
            v += (1 - self.decay2) * ca.square(p.grad.data)
            v_hat = v / (1 - (self.decay2 ** self.t))

            p.buf.data -= (m_hat * self.lr) / (ca.sqrt(v_hat) + self.eps)
