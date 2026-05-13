````md
# micrograd-from-scratch

A tiny deep learning framework built from scratch in pure Python.

This project implements a scalar-valued autograd engine with dynamic computational graphs and reverse-mode backpropagation, inspired by how modern deep learning frameworks work internally.

Built without PyTorch/TensorFlow for educational purposes — to understand what actually happens behind `.backward()`.

---

## Features

- Scalar-valued `Value` object
- Dynamic computational graph
- Reverse-mode automatic differentiation
- Manual backpropagation via topological sort + chain rule
- Basic neural network components:
  - `Neuron`
  - `Layer`
  - `MLP`
- Gradient descent training loop
- Nonlinear activations (`tanh`)
- Trained on:
  - XOR classification
  - Circle classification

---

## Example

```python
loss.backward()

for p in model.parameters():
    p.data -= lr * p.grad
````

---

## Project Structure

```text
.
├── engine.py      # autograd engine
├── nn.py          # Neuron / Layer / MLP
├── train.py       # training loop + datasets
└── README.md
```

---

## Circle Classification

The MLP was trained on a nonlinear circle classification task:

* points inside circle → `1`
* points outside circle → `-1`

Training result:

* loss: `~0.69 → ~0.17`
* train accuracy: `~99%`

---

## Example Computational Graph

The project also supports visualization of computational graphs and gradient propagation.

<img width="1200" alt="computational graph" src="YOUR_GRAPH_IMAGE_HERE">

---

## Why?

The goal of this project was not performance, but understanding.

Building autograd manually makes concepts like:

* backpropagation
* gradient flow
* computational graphs
* chain rule
* neural network training

feel much less like magic.

---

## Notes

This implementation is intentionally simple and uses scalar operations only.

It is dramatically slower than tensor-based frameworks like PyTorch because every operation creates a Python object and graph node manually.

---

## Inspired By

* Andrej Karpathy's micrograd
* PyTorch autograd
* reverse-mode automatic differentiation

```
