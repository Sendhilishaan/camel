from __future__ import annotations
from camel.ops import Vbuf, Ops
from camel.array import CamelArray
from typing import List

class Context:
    def __init__(self): self.saved = ()
    def save(self, *xs): self.saved = xs

class Function:
    @classmethod
    def apply(cls, *inputs: Tensor) -> Tensor:
        ctx = Context()
        out_buf = cls.forward(ctx, *[t.buf for t in inputs])
        out = Tensor(out_buf, _prev=inputs, _op=cls.__name__)

        def _backward():
            grads = cls.backward(ctx, out.grad) # will be set through backward pass
            for t, g in zip(inputs, grads):
                t._accum(g)

        out._backward = _backward
        return out
    
    @staticmethod
    def forward(ctx: Context, *args: Vbuf):
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_out: Vbuf):
        raise NotImplementedError

class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, a: Vbuf, b: Vbuf):
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"matmul shape mismatch: {a.shape} @ {b.shape}")
        ctx.save(a, b)
        return Ops.matmul_forward(a, b)

    @staticmethod
    def backward(ctx: Context, grad_out: Vbuf):
        a, b = ctx.saved
        return Ops.matmul_backward(a, b, grad_out) # returns (dA, dB), one per input

class Add(Function):
    @staticmethod
    def forward(ctx: Context, a: Vbuf, b: Vbuf):
        if b.shape[0] != 1 or b.shape[1] != a.shape[1]:
            raise ValueError(f"add(bias) expects b=(1, {a.shape[1]}), got {b.shape} for a={a.shape}")
        return Ops.add_forward(a, b)
    
    @staticmethod
    def backward(ctx: Context, grad_out: Vbuf):
        return Ops.add_backward(grad_out)

class Sub(Function):
    @staticmethod
    def forward(ctx: Context, a: Vbuf, b: Vbuf):
        if a.shape != b.shape:
            raise ValueError(f"sub shape mismatch: {a.shape} - {b.shape}")
        return Ops.sub_forward(a, b)
    
    @staticmethod
    def backward(ctx: Context, grad_out: Vbuf):
        return Ops.sub_backward(grad_out)
    
class Hadamard(Function):
    @staticmethod
    def forward(ctx: Context, a: Vbuf, b: Vbuf):
        if a.shape != b.shape:
            raise ValueError(f"hadamard shape mismatch: {a.shape} * {b.shape}")
        ctx.save(a, b)
        return Ops.hadamard_forward(a, b)
    
    @staticmethod
    def backward(ctx: Context, grad_out: Vbuf):
        a, b = ctx.saved
        return Ops.hadamard_backward(a, b, grad_out)

class Mean(Function):
    @staticmethod
    def forward(ctx: Context, a: Vbuf):
        ctx.save(a)
        return Ops.mean_forward(a)

    @staticmethod
    def backward(ctx: Context, grad_out: Vbuf):
        a, = ctx.saved
        return (Ops.mean_backward(a, grad_out),)

class Tanh(Function):
    @staticmethod
    def forward(ctx: Context, a: Vbuf):
        out = Ops.tanh_forward(a)
        ctx.save(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_out: Vbuf):
        out, = ctx.saved
        return (Ops.tanh_backward(out, grad_out),)

class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, a: Vbuf):
        out = Ops.relu_forward(a)
        ctx.save(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_out: Vbuf):
        out, = ctx.saved
        return (Ops.relu_backward(out, grad_out),)

class SoftmaxXent(Function):
    @staticmethod
    def forward(ctx: Context, Z: Vbuf, Y: Vbuf):
        probs, loss = Ops.softmax_xent_forward(Z, Y)
        ctx.save(probs, Y)
        return loss

    @staticmethod
    def backward(ctx: Context, grad_out: Vbuf):
        probs, Y = ctx.saved
        # None since apply requires same #grads as inputs and Y is just data
        return (Ops.softmax_xent_backward(probs, Y, grad_out), None)

class Tensor:
    def __init__(self, buf: Vbuf, requires_grad=True, _prev=(), _op=""):
        self.buf = buf
        self.grad: Vbuf | None = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = _prev
        self._op = _op
    
    def __matmul__(self, other: Tensor) -> Tensor:
        return MatMul.apply(self, other)

    def __add__(self, other: Tensor) -> Tensor:
        return Add.apply(self, other)

    def __sub__(self, other: Tensor) -> Tensor:
        return Sub.apply(self, other)

    def __mul__(self, other: Tensor) -> Tensor:
        return Hadamard.apply(self, other)

    def tanh(self) -> Tensor:
        return Tanh.apply(self)

    def mean(self) -> Tensor:
        return Mean.apply(self)
    
    def relu(self) -> Tensor:
        return ReLU.apply(self)
    
    def softmax_xent(self, Y: Tensor) -> Tensor:
        return SoftmaxXent.apply(self, Y)

    def _accum(self, delta: Vbuf) -> None:
        if not self.requires_grad or delta is None:
            return
        elif self.grad is None:
            # under metal, delta can be adopted directly instead of copied:
            # every resident dispatch produces a fresh buffer and never
            # mutates an input, so nothing will ever mutate delta in place
            self.grad = delta if Ops.backend == "metal" else Vbuf(delta.data.copy())
        elif Ops.backend == "metal":
            self.grad = Ops.accumulate(self.grad, delta)
        else:
            self.grad.data += delta.data
    
    def backward(self):
        assert self.buf.shape == (1, 1)
        topo: List[Tensor] = []
        visited = set()

        def build(node: Tensor): # topological sort on prev tensors
            if id(node) in visited:
                return

            visited.add(id(node))
            for parent in node._prev:
                build(parent)
            
            topo.append(node)
        
        build(self)

        self.grad = Vbuf(CamelArray.ones((1, 1)))

        for t in reversed(topo):
            t._backward()