# Autograd From Scratch

This repository contains a small automatic differentiation (autograd) engine implemented from scratch.  
The goal of this project is to understand how gradient computation works internally in libraries like PyTorch, rather than relying on their built-in autograd systems.

This is a learning-focused project. It is not meant to be fast, complete, or production-ready.

---

## Why I built this

I wanted to move beyond *using* autograd and understand:
- how computational graphs are constructed during the forward pass
- how gradients are propagated backward using the chain rule
- why design choices in autograd systems affect training behavior

Implementing a minimal version helped make these ideas concrete.

---

## What’s included

- A simple `Tensor` abstraction that stores data, gradients, and links to parent nodes
- Basic operations (e.g. addition, multiplication, matrix multiplication)
- Manual backpropagation using reverse-mode automatic differentiation
- A small example showing linear regression built on top of the custom autograd engine

---

## Project structure


autograd-from-scratch/
│
├── autograd/
│   ├── tensor.py      # Core Tensor class and basic operations
│   ├── ops.py         # Primitive ops (add, mul,...)
│   └── backward.py    # Backpropagation and topological traversal
│
├── examples/
│   └── simple_regression.py  # Linear regression using custom autograd
│
├── notes/
│   └── autograd_study.ipynb  # Exploratory study using PyTorch autograd
│
└── README.md


The notebook in `notes/` was used for experimentation and comparison with PyTorch autograd.

---

## How it works (high level)

During the forward pass, each operation records how its output depends on its inputs.  
These relationships form a computational graph.

When `.backward()` is called on the final output (typically a loss), the graph is traversed in reverse order and gradients are computed using the chain rule.

Gradients are accumulated at each node as the backward pass progresses.

---

## Limitations

To keep the implementation simple:
- only a small set of operations is supported
- no GPU support
- limited handling of edge cases (e.g. broadcasting)

These limitations are intentional and aligned with the learning goal of the project.

---

## Notes

This project was built to improve my understanding of how automatic differentiation works under the hood.  
For real-world training and performance, established frameworks like PyTorch should be used instead.
