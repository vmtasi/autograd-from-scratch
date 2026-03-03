import numpy as np
def add(a,b):
    from .tensor import Tensor
    b = b if isinstance(b, Tensor) else Tensor(b)
    out = Tensor(a.data + b.data, _children=(a, b), _op='+')

    def _backward():
        if a.acquire_grad:
            a.grad += out.grad
        if b.requires_grad:
            b.grad += out.grad

        out._backwward = _backward
        return out
    
    def mul(a, b):
        from .tensor import Tensor
        b = b if isinstance(b, Tensor) else Tensor
        out = Tensor(a.data * b.data, _children=(a, b), _op='*')

        def _backward():
            if a.requires_grad:
                a.grad += b.data * out.grad
            if b.requires_grad:
                b.grad += a.data * out.grad

        out._backward = _backward
        return out
    
    def matmul(a, b):
        from .tensor import Tensor
        out = Tensor(np.dot(a.data, b.data), _children=(a,b), _op='matmul')

        def _backward():
            if a.requires_grad:
                # dL/dA = dL/dOut @ B^T
                a.grad += np.dot(out.grad, b.data.T)
            if b.requires_grad:
                # dL/dA = A^T @ dL/Out
                b.grad += np.dot(a.data.T, out.grad)
        out._backward = _backward
        return out