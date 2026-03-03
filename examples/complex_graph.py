import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autograd.tensor import Tensor
# f(x) = (x * x) + (x * 2)
# f'(x) = 2x + 2
x = Tensor([3.0], requires_grad=True)

# The graph splits here: 'x' is used twice

y = (x*x) + (x*2.0)

y.backward()

print(f"Input x: {x.data.item():.3f}")
print(f"Output y: {y.data.item():.3f}")
print(f"Gradient dy/dx: {x.grad.item():.3f}")# (2*3) + 2 = 8.0