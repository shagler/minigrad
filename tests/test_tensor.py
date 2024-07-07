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
    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])
    z = x + y
    self.assertTrue(isinstance(z, Tensor))
    np.testing.assert_array_equal(z.data, np.array([5, 7, 9]))

  def test_subtraction(self):
    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])
    z = x - y
    self.assertTrue(isinstance(z, Tensor))
    np.testing.assert_array_equal(z.data, np.array([-3, -3, -3]))

  def test_multiplication(self):
    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])
    z = x * y
    self.assertTrue(isinstance(z, Tensor))
    np.testing.assert_array_equal(z.data, np.array([4, 10, 18]))

  def test_division(self):
    x = Tensor([2, 4, 6])
    y = Tensor([1, 2, 3])
    z = x / y
    self.assertTrue(isinstance(z, Tensor))
    np.testing.assert_array_equal(z.data, np.array([2, 2, 2]))

  def test_relu(self):
    x = Tensor([-1, 0, 1], requires_grad=True)
    y = x.relu()
    np.testing.assert_array_equal(y.data, np.array([0, 0, 1]))

  def test_backward_add_mul(self):
    x = Tensor([1, 2, 3], requires_grad=True)
    y = Tensor([4, 5, 6], requires_grad=True)
    z = (x + y) * x
    z.backward()
    self.assertIsNotNone(x.grad)
    self.assertIsNotNone(y.grad)
    if x.grad is not None and y.grad is not None:
      np.testing.assert_array_almost_equal(x.grad, np.array([6, 9, 12]), decimal=5)
      np.testing.assert_array_almost_equal(y.grad, np.array([1, 2, 3]), decimal=5)

  def test_backward_sub_div(self):
    x = Tensor([1, 2, 3], requires_grad=True)
    y = Tensor([4, 5, 6], requires_grad=True)
    z = (x - y) / x
    z.backward()
    self.assertIsNotNone(x.grad)
    self.assertIsNotNone(y.grad)
    if x.grad is not None and y.grad is not None:
      np.testing.assert_array_almost_equal(x.grad, np.array([4, 1.25, 0.666667]), decimal=5)
      np.testing.assert_array_almost_equal(y.grad, np.array([-1, -0.5, -1/3]), decimal=5)




if __name__ == '__main__':
  unittest.main()
