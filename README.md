# Autograd From Scratch

This repository contains a small automatic differentiation (autograd) engine implemented from scratch using NumPy.

The goal of this project is to understand how computational graphs are built and how gradients are propagated using reverse-mode automatic differentiation, rather than relying on existing framework implementations.



## Project Overview

The engine builds a dynamic computational graph during the forward pass and computes gradients by explicitly traversing this graph in reverse order during the backward pass. All gradient computations are implemented manually using the chain rule.


---

## Project Structure

```text
autograd-from-scratch/
├── autograd/
│   ├── tensor.py      # Tensor abstraction and operator overloading
│   ├── ops.py         # Operation-specific forward and backward rules
│   └── backward.py   # Backpropagation engine (topological sort)
├── examples/
│   └── simple_regression.py  # Linear regression using custom autograd
│   └── complex_graph.py      # Multi-path gradient verification
└── README.md
```

## How It Works

1. During the forward pass, each operation creates a new Tensor and records references to its parent tensors.
2. These relationships form a directed acyclic computational graph.
3. Calling `.backward()` on the final output triggers a topological traversal of the graph and gradient propagation using the chain rule.
4. Gradients are accumulated at each tensor, allowing correct handling of shared subgraphs.

The backward pass assumes a scalar loss.

---

## Example Usage

from autograd.tensor import Tensor

x = Tensor([3.0], requires_grad=True)  
y = (x * x) + (x * 2.0)

y.backward()

print("Gradient:", x.grad)  # Expected: 8.0

---

## Verification

To validate the implementation, the engine was tested on simple algebraic expressions with shared subgraphs and on linear regression trained using mean squared error loss and gradient descent. In all cases, computed gradients matched analytical expectations.

---

## Limitations

- Designed for educational clarity, not performance
- CPU-only (NumPy backend)
- Limited set of supported operations
- No full broadcasting or advanced memory optimizations

---

## Notes

This project was built to deepen understanding of how automatic differentiation systems work internally.
