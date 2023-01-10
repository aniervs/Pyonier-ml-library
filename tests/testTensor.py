from unittest import TestCase

import numpy as np

from nn.Tensor import Tensor


class TestTensor(TestCase):
    def setUp(self) -> None:
        self.tensor1 = Tensor([1, 2, 3, 4], None, None)
        self.tensor2 = Tensor([5, 6, 7, 8], None, None)

    def testAddition(self):
        new_tensor = self.tensor1 + self.tensor2
        expected = Tensor([6, 8, 10, 12], [self.tensor1, self.tensor2], "add")
        self.assertEqual(np.all(expected.data - new_tensor.data), 0)
        self.assertEqual(expected.parents, new_tensor.parents)
        self.assertEqual(expected.parents_operation, new_tensor.parents_operation)

    def testSubtraction(self):
        new_tensor = self.tensor1 - self.tensor2
        expected = Tensor([-4, -4, -4, -4], [self.tensor1, self.tensor2], "sub")
        self.assertEqual(np.all(expected.data - new_tensor.data), 0)
        self.assertEqual(expected.parents, new_tensor.parents)
        self.assertEqual(expected.parents_operation, new_tensor.parents_operation)

    def testMultiplication(self):
        new_tensor = self.tensor1 * self.tensor2
        expected = Tensor([5, 12, 21, 32], [self.tensor1, self.tensor2], "mul")
        self.assertEqual(np.all(expected.data - new_tensor.data), 0)
        self.assertEqual(expected.parents, new_tensor.parents)
        self.assertEqual(expected.parents_operation, new_tensor.parents_operation)

    def testNegation(self):
        new_tensor = -self.tensor1
        expected = Tensor([-1, -2, -3, -4], [self.tensor1], "neg")
        self.assertEqual(np.all(expected.data - new_tensor.data), 0)
        self.assertEqual(expected.parents, new_tensor.parents)
        self.assertEqual(expected.parents_operation, new_tensor.parents_operation)
