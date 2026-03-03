import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
from autograd.tensor import Tensor

np.random.seed(42)
x_raw = np.linspace(0, 10, 20).reshape(-1,1)
y_raw = 2.5 * x_raw + 0.8 + np.random.normal(0, 0.5, x_raw.shape)

# Convert to our custom Tensors
X = Tensor(x_raw)
Y = Tensor(y_raw)

W = Tensor(np.random.randn(1, 1), requires_grad = True)
B = Tensor(np.random.randn(1), requires_grad=True)

learning_rate = 0.001
epochs = 200
print("Starting training with custom Autograd engine...")

for epoch in range(epochs):
    pred = X.__matmul__(W) + B

    diff = pred + (Y * -1.0)
    loss = (diff * diff).mean()
    
    W.zero_grad()
    B.zero_grad()
    
    loss.backward()

    W.data -= learning_rate * W.grad
    B.data -= learning_rate * B.grad

    if epoch % 20 == 0:
        print(f"Epoch{epoch:3f}| Loss: {loss.data:.4f}| W: {W.data[0,0]:.3f}| B: {B.data[0]:.3f}")
print("_" * 30)
print(f"Final Prediction Model: y = {W.data[0,0]: .2f}x + {B.data[0]:.2f}")
print(f"Target Physics Model: y = 2.51x + 0.80")