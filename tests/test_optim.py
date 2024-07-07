import unittest
import numpy as np

from minigrad import Tensor, SGD

class TestOptimizer(unittest.TestCase):
  def test_optimizer(self):
    x = Tensor([1, 2, 3], requires_grad=True)
    y = Tensor([4, 5, 6], requires_grad=True)
    optim = SGD([x, y], lr=0.1)
    z = (x + y) * x
    z.backward()
    self.assertIsNotNone(x.grad)
    self.assertIsNotNone(y.grad)
    optim.step()
    np.testing.assert_array_almost_equal(x.data, np.array([0.4, 1.1, 1.8]), decimal=5)
    np.testing.assert_array_almost_equal(y.data, np.array([3.9, 4.8, 5.7]), decimal=5)

if __name__ == "__main__":
  unittest.main()
