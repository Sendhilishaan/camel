from __future__ import annotations
from camel.tensor import Tensor
from typing import Iterable
import numpy as np

class Optimiser:
    def __init__(self, tensors: Iterable[Tensor], lr: float):
        self.lr = lr
        self.tensors = list(tensors)

    def zero_grad(self):
        for t in self.tensors:
            t.grad = None

    def step(self):
        raise NotImplementedError
    
class SGD(Optimiser):
    def __init__(self, tensors: Iterable[Tensor], lr: float, momentum=0.0):
        super().__init__(tensors, lr)
        self.momentum = momentum
        self.velocity = [np.zeros_like(t.buf.data) for t in self.tensors]
    
    def step(self):
        for t, v in zip(self.tensors, self.velocity):
            v *= self.momentum
            v += t.grad.data
            t.buf.data -= v * self.lr

class AdaGrad(Optimiser):
    def __init__(self, tensors: Iterable[Tensor], lr: float, ep=1e-8):
        super().__init__(tensors, lr)
        self.ep = ep
        self.grad_sum = [np.zeros_like(t.buf.data) for t in self.tensors]
    
    def step(self):
        for t, g in zip(self.tensors, self.grad_sum):
            g += np.square(t.grad.data)
            t.buf.data -= (self.lr * t.grad.data) / (np.sqrt(self.ep) + self.ep)