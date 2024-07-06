
import numpy as np

class Tensor:
  def __init__(self, data, requires_grad=False):
    self.data = np.array(data)
    self.requires_grad = requires_grad
    self.grad = None
