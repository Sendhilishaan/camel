from __future__ import annotations
from camel.tensor import Tensor
from typing import List

class Optimiser:
    def __init__(self, lr: float, tensors: List[Tensor]):
        self.lr = lr
        self.tensors = tensors

    def zero_grad(self):
        for t in self.tensors:
            t.grad = None

    def step(self):
        raise NotImplementedError
    
class SGD(Optimiser):
    def step(self):
        for t in self.tensors:
            t.buf.data -= t.grad.data * self.lr