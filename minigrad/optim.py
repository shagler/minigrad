import numpy as np

class Optimizer:
  def __init__(self, parameters):
    self.parameters = parameters

  def zero_grad(self):
    for p in self.parameters:
      if p.requires_grad:
        p.grad = None

class SGD(Optimizer):
  def __init__(self, parameters, lr=0.01):
    super().__init__(parameters)
    self.lr = lr

  def step(self):
    for p in self.parameters:
      if p.requires_grad and p.grad is not None:
        p.data -= self.lr * p.grad
