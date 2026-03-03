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

## Project structure


<p data-start="1375" data-end="1561">autograd-from-scratch/<br>
├── autograd/<br>
│   ├── tensor.py<br>
│   ├── ops.py<br>
│   └── backward.py<br>
├── examples/<br>
│   └── simple_regression.py<br>
├── notes/<br>
│   └── autograd_study.ipynb<br>
└── README.md</p>


The notebook in `notes/` was used for experimentation and comparison with PyTorch autograd.

---

## How it works (high level)

During the forward pass, each operation records how its output depends on its inputs.  
These relationships form a computational graph.

When `.backward()` is called on the final output (typically a loss), the graph is traversed in reverse order and gradients are computed using the chain rule.

Gradients are accumulated at each node as the backward pass progresses.

## Limitations

To keep the implementation simple:
- only a small set of operations is supported
- no GPU support
- limited handling of edge cases (e.g. broadcasting)


## Notes

I built this to improve my understanding of how automatic differentiation works under the hood.
