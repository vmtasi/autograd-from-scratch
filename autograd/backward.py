import numpy as np

def backward(root_tensor):
    """
    Performs a reverse topological sort and triggers gradient 
    accumulation across the computational graph.
    """
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    # Build the ordered list of nodes
    build_topo(root_tensor)

    # Set the starting gradient of the root (usually the loss) to 1.0
    root_tensor.grad = np.ones_like(root_tensor.data)

    # Apply the chain rule in reverse topological order
    for node in reversed(topo):
        node._backward()