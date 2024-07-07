import unittest
import numpy as np
from minigrad.tensor import Tensor

class TestTensor(unittest.TestCase):
  def test_tensor_creation(self):
    t = Tensor([1, 2, 3], requires_grad=True)
    self.assertTrue(isinstance(t, Tensor))
    self.assertTrue(t.requires_grad)
    np.testing.assert_array_equal(t.data, np.array([1, 2, 3]))

  def test_addition(self):
    x = Tensor([1, 2, 3], requires_grad=True)
    y = Tensor([4, 5, 6], requires_grad=True)
    z = x + y
    self.assertTrue(isinstance(z, Tensor))
    np.testing.assert_array_equal(z.data, np.array([5, 7, 9]))

  def test_multiplication(self):
    x = Tensor([1, 2, 3], requires_grad=True)
    y = Tensor([4, 5, 6], requires_grad=True)
    z = x * y
    self.assertTrue(isinstance(z, Tensor))
    np.testing.assert_array_equal(z.data, np.array([4, 10, 18]))

if __name__ == '__main__':
  unittest.main()
