from __future__ import annotations
from camel.tensor import Tensor
from camel.ops import Vbuf, Ops
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
        self.velocity = [Vbuf(ca.zeros_like(p.buf.data)) for p in self.tensors]

    def step(self):
        for p, v in zip(self.tensors, self.velocity):
            if Ops.backend == "metal":
                Ops.sgd_step(p.buf, v, p.grad, self.momentum, self.lr)
                continue
            v.data *= self.momentum
            v.data += p.grad.data
            p.buf.data -= v.data * self.lr

class AdaGrad(Optimiser):
    def __init__(self, tensors: Iterable[Tensor], lr: float, eps=1e-8):
        super().__init__(tensors, lr)
        self.eps = eps
        self.grad_sum = [Vbuf(ca.zeros_like(p.buf.data)) for p in self.tensors]

    def step(self):
        for p, G in zip(self.tensors, self.grad_sum):
            if Ops.backend == "metal":
                Ops.adagrad_step(p.buf, G, p.grad, self.lr, self.eps)
                continue
            G.data += ca.square(p.grad.data) # sum of second moment
            p.buf.data -= (self.lr * p.grad.data) / (ca.sqrt(G.data) + self.eps)

class RMSprop(Optimiser):
    def __init__(self, tensors: Iterable[Tensor], lr: float, eps=1e-8, decay=0.9):
        super().__init__(tensors, lr)
        self.eps = eps
        self.decay = decay
        self.ema_sq = [Vbuf(ca.zeros_like(p.buf.data)) for p in self.tensors]

    def step(self):
        for p, s in zip(self.tensors, self.ema_sq):
            if Ops.backend == "metal":
                Ops.rmsprop_step(p.buf, s, p.grad, self.lr, self.eps, self.decay)
                continue
            s.data *= self.decay
            s.data += (1 - self.decay) * ca.square(p.grad.data)
            # (1 - b) normalisation, weights sum to 1 making it a true average
            p.buf.data -= (self.lr * p.grad.data) / (ca.sqrt(s.data) + self.eps)

class Adam(Optimiser):
    def __init__(self, tensors: Iterable[Tensor], lr: float, decay1=0.9, decay2=0.999, eps=1e-8):
        super().__init__(tensors, lr)
        self.decay1 = decay1 # decay for 1st moment (grad, dir)
        self.decay2 = decay2 # decay for 2nd moment (grad^2, mag)
        self.eps = eps
        self.t = 0 # timestep

        self.exp_avg = [Vbuf(ca.zeros_like(p.buf.data)) for p in self.tensors]
        self.exp_avg_sq = [Vbuf(ca.zeros_like(p.buf.data)) for p in self.tensors]

    def step(self):
        self.t += 1
        bc1 = 1 - (self.decay1 ** self.t) # bias correction, both moments start at 0
        bc2 = 1 - (self.decay2 ** self.t)

        for p, m, v in zip(self.tensors, self.exp_avg, self.exp_avg_sq):
            if Ops.backend == "metal":
                Ops.adam_step(p.buf, m, v, p.grad, self.lr, self.eps, self.decay1, self.decay2, bc1, bc2)
                continue
            m.data *= self.decay1
            m.data += (1 - self.decay1) * p.grad.data
            m_hat = m.data / bc1

            v.data *= self.decay2
            v.data += (1 - self.decay2) * ca.square(p.grad.data)
            v_hat = v.data / bc2

            p.buf.data -= (m_hat * self.lr) / (ca.sqrt(v_hat) + self.eps)
