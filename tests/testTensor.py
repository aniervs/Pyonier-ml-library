import unittest

import numpy as np

from nn.Tensor import Tensor


class TestTensor(unittest.TestCase):
    def setUp(self) -> None:
        self.tensor1 = Tensor([1, 2, 3, 4], autograd=True)
        self.tensor2 = Tensor([5, 6, 7, 8], autograd=True)
        self.tensor3 = Tensor([9, 10, 11, 12], autograd=True)

    def testAddition(self):
        new_tensor = self.tensor1 + self.tensor2
        expected = Tensor([6, 8, 10, 12], autograd=True, parents=[self.tensor1, self.tensor2], parents_operation="add")
        self.assertEqual(np.all(expected.data - new_tensor.data), 0)
        self.assertEqual(expected.parents, new_tensor.parents)
        self.assertEqual(expected.parents_operation, new_tensor.parents_operation)
        self.assertTrue(new_tensor.autograd)

    def testSubtraction(self):
        new_tensor = self.tensor1 - self.tensor2
        expected = Tensor([-4, -4, -4, -4], parents=[self.tensor1, self.tensor2], parents_operation="sub")
        self.assertEqual(np.all(expected.data - new_tensor.data), 0)
        self.assertEqual(expected.parents, new_tensor.parents)
        self.assertEqual(expected.parents_operation, new_tensor.parents_operation)
        self.assertTrue(new_tensor.autograd)

    def testMultiplication(self):
        new_tensor = self.tensor1 * self.tensor2
        expected = Tensor([5, 12, 21, 32], parents=[self.tensor1, self.tensor2], parents_operation="mul")
        self.assertEqual(np.all(expected.data - new_tensor.data), 0)
        self.assertEqual(expected.parents, new_tensor.parents)
        self.assertEqual(expected.parents_operation, new_tensor.parents_operation)
        self.assertTrue(new_tensor.autograd)

    def testNegation(self):
        new_tensor = -self.tensor1
        expected = Tensor([-1, -2, -3, -4], parents=[self.tensor1], parents_operation="neg")
        self.assertEqual(np.all(expected.data - new_tensor.data), 0)
        self.assertEqual(expected.parents, new_tensor.parents)
        self.assertEqual(expected.parents_operation, new_tensor.parents_operation)
        self.assertTrue(new_tensor.autograd)

    def testPower(self):
        new_tensor = self.tensor1 ** 3
        expected = Tensor([1, 8, 27, 64], parents=[self.tensor1, 3], parents_operation="pow")
        self.assertEqual(np.all(expected.data - new_tensor.data), 0)
        self.assertEqual(expected.parents, new_tensor.parents)
        self.assertEqual(expected.parents_operation, new_tensor.parents_operation)
        self.assertTrue(new_tensor.autograd)

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

    def testBackwardPow(self):
        new_tensor = self.tensor1 ** 3
        new_tensor.backward()
        self.assertEqual(np.all(self.tensor1.grad.data - np.array([3, 12, 27, 48])), 0)

    def testBackwardTranspose(self):
        new_tensor = self.tensor1.transpose()
        new_tensor.backward()
        self.assertEqual(np.all(new_tensor.grad.data - np.array([1])), 0)

    def testBackwardNoAutograd(self):
        self.tensor1.autograd = False
        new_tensor = self.tensor1 + self.tensor2
        new_tensor.backward()
        self.assertIsNone(self.tensor1.grad)
        self.assertIsNone(self.tensor2.grad)

    def testBackwardMultipleUses(self):
        d = self.tensor1 + self.tensor2
        e = self.tensor2 + self.tensor3
        f = d + e

        f.backward()

        self.assertEqual(self.tensor1.grad.data, np.array([1]))
        self.assertEqual(self.tensor3.grad.data, np.array([1]))
        self.assertEqual(self.tensor2.grad.data, np.array([2]))


if __name__ == '__main__':
    unittest.main()
