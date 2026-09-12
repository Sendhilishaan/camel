import math

from camel import array as ca
from camel.ops import Vbuf, _pair, _image_shape
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

        self.W.buf.data[:] = ca.random.randn(in_features, out_features) * math.sqrt(2 / in_features) # He initialisation assuming ReLU, determing how to match with activation later

    def __call__(self, X: Tensor) -> Tensor:
        return (X @ self.W) + self.b

class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, bias=True):
        if any(type(c) is not int or c <= 0 for c in (in_channels, out_channels)):
            raise ValueError("convolution channels must be positive integers")
        kh, kw = _pair(kernel_size, "kernel_size")
        _image_shape((out_channels, in_channels, kh, kw))
        self.stride = _pair(stride, "stride")
        self.padding = _pair(padding, "padding", 0)
        fan_in = in_channels * kh * kw
        self.W = Tensor(Vbuf(ca.random.randn(out_channels, in_channels, kh, kw) * math.sqrt(2 / fan_in)))
        self.b = Tensor(Vbuf.zeros(1, out_channels)) if bias else None

    def __call__(self, X: Tensor) -> Tensor:
        return X.conv2d(self.W, self.b, self.stride, self.padding)

class MaxPool2d(Module):
    def __init__(self, kernel_size=2, stride=None):
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride = self.kernel_size if stride is None else _pair(stride, "stride")

    def __call__(self, X: Tensor) -> Tensor:
        return X.maxpool2d(self.kernel_size, self.stride)

class Flatten(Module):
    def __call__(self, X: Tensor) -> Tensor:
        return X.flatten()

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

class CNN(Module):
    def __init__(self, in_channels: int, hidden_channels: List[int], out_channels: int,
                 image_size, activations: None | List[type[Function]] = None):
        if not hidden_channels:
            raise ValueError("hidden_channels must have at least one entry (a convolution width)")
        for w in [in_channels, *hidden_channels, out_channels]:
            if type(w) is not int or w <= 0:
                raise ValueError(f"layer widths must be positive integers, got {w}")

        self.in_channels = in_channels
        self.hidden_channels = list(hidden_channels)
        self.image_size = _pair(image_size, "image_size")
        _image_shape((1, in_channels, *self.image_size))

        if activations is None:
            self.activations = [ReLU] * len(hidden_channels)
        elif len(hidden_channels) != len(activations):
            raise ValueError(f"expected {len(hidden_channels)} activations for {len(hidden_channels)} convolutions, got {len(activations)}")
        else:
            self.activations = list(activations)

        h, w = self.image_size
        for _ in hidden_channels:
            h, w = h // 2, w // 2 # 3x3 convolution preserves size, 2x2 pooling halves it
            if h == 0 or w == 0:
                raise ValueError("image_size is too small for the number of pooling layers")

        shapes = [in_channels] + self.hidden_channels
        self.layers = [Conv2d(a, b, 3, padding=1) for a, b in zip(shapes, shapes[1:])]
        self.pool = MaxPool2d(2)
        self.flatten = Flatten()
        self.head = Linear(hidden_channels[-1] * h * w, out_channels)

    def __call__(self, X: Tensor) -> Tensor:
        if len(X.buf.shape) != 4 or X.buf.shape[1:] != (self.in_channels, *self.image_size):
            raise ValueError(f"expected NCHW input with {(self.in_channels, *self.image_size)}, got {X.buf.shape}")

        X_curr = X
        for act, layer in zip(self.activations, self.layers):
            X_curr = self.pool(act.apply(layer(X_curr)))

        return self.head(self.flatten(X_curr))
