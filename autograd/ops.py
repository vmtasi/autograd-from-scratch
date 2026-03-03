import numpy as np

def add(a, b):
    from .tensor import Tensor
    b = b if isinstance(b, Tensor) else Tensor(b)
    # Propagate the flag
    requires_grad = a.requires_grad or b.requires_grad
    out = Tensor(a.data + b.data, _children=(a, b), _op='+', requires_grad=requires_grad)

    def _backward():
        if a.requires_grad:
            grad_a = out.grad
            while grad_a.ndim > a.data.ndim:
                grad_a = grad_a.sum(axis=0)
            for i, dim in enumerate(a.data.shape):
                if dim == 1:
                    grad_a = grad_a.sum(axis=i, keepdims=True)
            a.grad += grad_a
            
        if b.requires_grad:
            grad_b = out.grad
            while grad_b.ndim > b.data.ndim:
                grad_b = grad_b.sum(axis=0)
            for i, dim in enumerate(b.data.shape):
                if dim == 1:
                    grad_b = grad_b.sum(axis=i, keepdims=True)
            b.grad += grad_b
            
    out._backward = _backward
    return out

def mul(a, b):
    from .tensor import Tensor
    b = b if isinstance(b, Tensor) else Tensor(b)
    # FIX: Propagate requires_grad
    requires_grad = a.requires_grad or b.requires_grad
    out = Tensor(a.data * b.data, _children=(a, b), _op='*', requires_grad=requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad += b.data * out.grad
        if b.requires_grad:
            b.grad += a.data * out.grad
    out._backward = _backward
    return out

def matmul(a, b):
    from .tensor import Tensor
    # FIX: Propagate requires_grad
    requires_grad = a.requires_grad or b.requires_grad
    out = Tensor(np.dot(a.data, b.data), _children=(a, b), _op='matmul', requires_grad=requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad += np.dot(out.grad, b.data.T)
        if b.requires_grad:
            b.grad += np.dot(a.data.T, out.grad)
    out._backward = _backward
    return out

def mean(a):
    from .tensor import Tensor
    # FIX: Propagate requires_grad
    out = Tensor(np.mean(a.data), _children=(a,), _op='mean', requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad += (np.ones_like(a.data) / a.data.size) * out.grad
    out._backward = _backward
    return out