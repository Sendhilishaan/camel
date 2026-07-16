import math
import numpy as np

from camel.ops import Vbuf
from camel.tensor import Tensor

class Module:
    """
    base class for anything that owns parameters: (layer, model with layers, model with models with layers)
    parameter collection
    """
    def parameters(self): # returns iterator of all parameters composed in module
        for v in self.__dict__.values():
            if isinstance(v, Tensor):  yield v
            elif isinstance(v, Module): yield from v.parameters()

            elif isinstance(v, list): # list treated as single arg
                for m in v:
                    if isinstance(m, Tensor):  yield m
                    elif isinstance(m, Module): yield from m.parameters()


class Linear(Module):
    # Linear layer
    def __init__(self, in_features, out_features):
        self.W = Tensor(Vbuf.zeros(in_features, out_features))
        self.b = Tensor(Vbuf.zeros(1, out_features))

        self.W.buf.data[:] = np.random.randn(in_features, out_features) * math.sqrt(2 / in_features) # He initialisation assuming ReLU, determing how to match with activation later

    def __call__(self, X: Tensor) -> Tensor:
        return (X @ self.W) + self.b