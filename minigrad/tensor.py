
from typing import Self

import numpy as np

class Tensor:
  def __init__(self, data, requires_grad=False, _children=()):
    self.data = np.array(data)
    self.requires_grad = requires_grad
    self.grad = None
    self._backward = lambda: None
    self._prev = set(_children)

  def __repr__(self) -> str:
    return f"Tensor({self.data}, requires_grad={self.requires_grad})"

  def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data + other.data,
      requires_grad=(self.requires_grad or other.requires_grad),
      _children=(self, other))

    def _backward():
      if self.requires_grad:
        self.grad = self.grad + out.grad if self.grad is not None else out.grad
      if other.requires_grad:
        other.grad = other.grad + out.grad if other.grad is not None else out.grad
    out._backward = _backward

    return out

  def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data * other.data,
      requires_grad=(self.requires_grad or other.requires_grad),
      _children=(self, other))

    def _backward():
      if self.requires_grad:
        self.grad = self.grad + other.data * out.grad if self.grad is not None else other.data * out.grad
      if other.requires_grad:
        other.grad = other.grad + self.data * out.grad if other.grad is not None else self.data * out.grad
    out._backward = _backward

    return out

  # @TODO: __div__ and __sub__

  def relu(self):
    out_data = np.maximum(0, self.data)
    out = Tensor(out_data,
      requires_grad=self.requires_grad,
      _children=(self,))

    def _backward():
      if self.requires_grad and out.grad is not None:
        grad = np.where(out_data > 0, out.grad, 0)
        self.grad = self.grad + grad if self.grad is not None else grad
    out._backward = _backward
