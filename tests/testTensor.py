from unittest import TestCase

import numpy as np

from nn.Tensor import Tensor


class TestTensor(TestCase):
    def setUp(self) -> None:
        self.tensor1 = Tensor([1, 2, 3, 4])
        self.tensor2 = Tensor([5, 6, 7, 8])

    def testAddition(self):
        new_tensor = self.tensor1 + self.tensor2
        expected = Tensor([6, 8, 10, 12])
        self.assertEqual(np.all(expected.data - new_tensor.data), 0)

    def testSubtraction(self):
        new_tensor = self.tensor1 - self.tensor2
        expected = Tensor([-4, -4, -4, -4])
        self.assertEqual(np.all(expected.data - new_tensor.data), 0)

    def testMultiplication(self):
        new_tensor = self.tensor1 * self.tensor2
        expected = Tensor([5, 12, 21, 32])
        self.assertEqual(np.all(expected.data - new_tensor.data), 0)

    def testNegation(self):
        new_tensor = -self.tensor1
        expected = Tensor([-1, -2, -3, -4])
        self.assertEqual(np.all(expected.data - new_tensor.data), 0)
