import numpy as np
from .backward import backward

class Tensor:
    def __init__(self, data, _children=(), _op='', requires_grad=False):
        # Ensure data is a float32 numpy array for consistency
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        
        # Computational Graph references
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, op={self._op})"

    def backward(self):
        """Triggers the reverse-mode autodiff engine."""
        backward(self)
    
    def zero_grad(self):
        """Resets the gradient to zero."""
        self.grad = np.zeros_like(self.data)

    # Operator Overloading
    def __add__(self, other):
        from .ops import add
        return add(self, other)

    def __mul__(self, other):
        from .ops import mul
        return mul(self, other)

    def __matmul__(self, other):
        from .ops import matmul
        return matmul(self, other)
    
    # Sugar for @ operator
    def __rmatmul__(self, other):
        return self.__matmul__(other)
    def mean(self):
        from .ops import mean
        return mean(self)

    def __repr__(self):
        return f"Tensor(data={self.data.shape}, op={self._op})"