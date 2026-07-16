import math
import numpy as np

from camel.ops import Vbuf
from typing import List
from camel.tensor import ReLU, Function, Tensor


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

class MLP(Module):
    def __init__(self, in_channels: int, hidden_channels: List[int], activations: None | List[type[Function]] = None):
        if not hidden_channels:
            raise ValueError("hidden_channels must have at least one entry (the output width)")
        for w in [in_channels, *hidden_channels]:
            if w <= 0:
                raise ValueError(f"layer widths must be positive, got {w}")

        self.in_channels = in_channels

        self.hidden_channels = hidden_channels

        if activations is None:
            self.activations = [ReLU] * (len(hidden_channels) - 1)
        elif len(hidden_channels) - 1 != len(activations):
            raise ValueError(f"expected {len(hidden_channels) - 1} activations for {len(hidden_channels)} layers, got {len(activations)}")
        else:
            self.activations = activations
        
        shapes = [in_channels] + hidden_channels
        self.layers = [Linear(a, b) for a, b in zip(shapes, shapes[1:])]
    
    def __call__(self, X: Tensor) -> Tensor:
        X_curr = X
        for act, layer in zip(self.activations, self.layers):
            X_curr = act.apply(layer(X_curr))
        
        return self.layers[-1](X_curr)