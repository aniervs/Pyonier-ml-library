from unittest import TestCase

import numpy as np

from nn.Tensor import Tensor


class TestTensor(TestCase):
    def setUp(self) -> None:
        self.tensor1 = Tensor([1, 2, 3, 4])
        self.tensor2 = Tensor([5, 6, 7, 8])
        self.tensor3 = Tensor([9, 10, 11, 12])

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

    def testBackwardAdd(self):
        new_tensor = self.tensor1 + self.tensor2
        new_tensor.backward()
        self.assertEqual(np.all(self.tensor1.grad.data - np.array([1])), 0)
        self.assertEqual(np.all(self.tensor2.grad.data - np.array([1])), 0)

    def testBackwardSub(self):
        new_tensor = self.tensor1 - self.tensor2
        new_tensor.backward()
        self.assertEqual(self.tensor1.grad.data, np.array([1]))
        self.assertEqual(self.tensor2.grad.data, -np.array([1]))

    def testBackwardNeg(self):
        new_tensor = -self.tensor1
        new_tensor.backward()
        self.assertEqual(self.tensor1.grad.data, -np.array([1]))

    def testBackwardMul(self):
        new_tensor = self.tensor1 * self.tensor2
        new_tensor.backward()
        self.assertEqual(np.all(self.tensor1.grad.data - np.array([5, 6, 7, 8])), 0)
        self.assertEqual(np.all(self.tensor2.grad.data - np.array([1, 2, 3, 4])), 0)

    def testBackwardMultipleUses(self):
        d = self.tensor1 + self.tensor2
        e = self.tensor2 + self.tensor3
        f = d + e

        f.backward()

        self.assertEqual(self.tensor1.grad.data, np.array([1]))
        self.assertEqual(self.tensor3.grad.data, np.array([1]))
        self.assertEqual(self.tensor2.grad.data, np.array([2]))

