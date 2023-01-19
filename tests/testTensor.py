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
        expected = Tensor([1, 8, 27, 64], parents=[self.tensor1], parents_operation="pow_3")
        self.assertEqual(np.all(expected.data - new_tensor.data), 0)
        self.assertEqual(expected.parents, new_tensor.parents)
        self.assertEqual(expected.parents_operation, new_tensor.parents_operation)
        self.assertTrue(new_tensor.autograd)

    def testExpand(self):
        new_tensor = self.tensor1.expand(dim=1, copies=2)
        expected = Tensor([[1, 1], [2, 2], [3, 3], [4, 4]], parents=[self.tensor1], parents_operation="expand_1")
        self.assertEqual(expected.data.shape, new_tensor.data.shape)
        self.assertEqual(np.all(expected.data - new_tensor.data), 0)
        self.assertEqual(expected.parents, new_tensor.parents)
        self.assertEqual(expected.parents_operation, new_tensor.parents_operation)
        self.assertTrue(new_tensor.autograd)

    def testSum(self):
        tensor = Tensor([[1, 5], [2, 6], [3, 7], [4, 8]], autograd=True)
        new_tensor = tensor.sum(dim=1)
        expected = Tensor([6, 8, 10, 12], autograd=True, parents=[tensor], parents_operation="sum_1")
        self.assertEqual(expected.data.shape, new_tensor.data.shape)
        self.assertEqual(np.all(expected.data - new_tensor.data), 0)
        self.assertEqual(expected.parents, new_tensor.parents)
        self.assertEqual(expected.parents_operation, new_tensor.parents_operation)
        self.assertTrue(new_tensor.autograd)

    def testMatrixMult(self):
        self.tensor1 = self.tensor1.expand(dim=1, copies=1)
        self.tensor2 = self.tensor2.expand(dim=1, copies=1).transpose()
        new_tensor = self.tensor2 @ self.tensor1
        expected = Tensor([[70]], parents=[self.tensor2, self.tensor1], parents_operation="matmul")
        self.assertEqual(expected.data.shape, new_tensor.data.shape)
        self.assertEqual(np.all(expected.data - new_tensor.data), 0)
        self.assertEqual(expected.parents, new_tensor.parents)
        self.assertEqual(expected.parents_operation, new_tensor.parents_operation)
        self.assertTrue(new_tensor.autograd)

    def testRelu(self):
        tensor = Tensor([1, -3, 4, -5, 0], autograd=True)
        new_tensor = tensor.relu()
        expected = Tensor([1, 0, 4, 0, 0], autograd=True, parents=[tensor], parents_operation="relu")
        self.assertEqual(np.all(expected.data - new_tensor.data), 0)
        self.assertEqual(expected.parents, new_tensor.parents)
        self.assertEqual(expected.parents_operation, new_tensor.parents_operation)
        self.assertTrue(new_tensor.autograd)

    def testBackwardRelu(self):
        tensor = Tensor([1, -3, 4, -5, 0], autograd=True)
        new_tensor = tensor.relu()
        new_tensor.backward()
        self.assertEqual(np.all(tensor.grad.data - np.array([1, 0, 1, 0, 0])), 0)

    def testBackwardAdd(self):
        new_tensor = self.tensor1 + self.tensor2
        new_tensor.backward()
        self.assertEqual(self.tensor1.grad.data.shape, self.tensor1.data.shape)
        self.assertEqual(self.tensor2.grad.data.shape, self.tensor2.data.shape)
        self.assertEqual(np.all(self.tensor1.grad.data - np.array([1])), 0)
        self.assertEqual(np.all(self.tensor2.grad.data - np.array([1])), 0)

    def testBackwardSub(self):
        new_tensor = self.tensor1 - self.tensor2
        new_tensor.backward()
        self.assertEqual(self.tensor1.grad.data.shape, self.tensor1.data.shape)
        self.assertEqual(self.tensor2.grad.data.shape, self.tensor2.data.shape)
        self.assertEqual(np.all(self.tensor1.grad.data - np.array([1, 1, 1, 1])), 0)
        self.assertEqual(np.all(self.tensor2.grad.data + np.array([1, 1, 1, 1])), 0)

    def testBackwardNeg(self):
        new_tensor = -self.tensor1
        new_tensor.backward()
        self.assertEqual(self.tensor1.grad.data.shape, self.tensor1.data.shape)
        self.assertEqual(np.all(self.tensor1.grad.data + np.array([1, 1, 1, 1])), 0)

    def testBackwardMul(self):
        new_tensor = self.tensor1 * self.tensor2
        new_tensor.backward()
        self.assertEqual(np.all(self.tensor1.grad.data - np.array([5, 6, 7, 8])), 0)
        self.assertEqual(np.all(self.tensor2.grad.data - np.array([1, 2, 3, 4])), 0)

    def testBackwardPow(self):
        new_tensor = self.tensor1 ** 3
        new_tensor.backward()
        self.assertEqual(np.all(self.tensor1.grad.data - np.array([3, 12, 27, 48])), 0)

    def testBackwardMatMult(self):
        tensor1 = Tensor([[1], [2], [3], [4]], autograd=True)
        tensor2 = Tensor([[5, 6, 7, 8]], autograd=True)
        new_tensor = tensor2 @ tensor1
        new_tensor.backward()
        self.assertEqual(np.all(tensor1.grad.data - tensor2.data), 0)
        self.assertEqual(np.all(tensor2.grad.data - tensor1.data), 0)

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

        self.assertEqual(self.tensor1.grad.data.shape, self.tensor1.data.shape)
        self.assertEqual(self.tensor2.grad.data.shape, self.tensor2.data.shape)
        self.assertEqual(self.tensor3.grad.data.shape, self.tensor3.data.shape)

        self.assertEqual(np.all(self.tensor1.grad.data - np.array([1, 1, 1, 1])), 0)
        self.assertEqual(np.all(self.tensor3.grad.data - np.array([1, 1, 1, 1])), 0)
        self.assertEqual(np.all(self.tensor2.grad.data - np.array([2, 2, 2, 2])), 0)

    def testAutogradOptimizer(self):
        np.random.seed(0)

        data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
        target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)

        w = list()
        w.append(Tensor(np.random.rand(2, 3), autograd=True))
        w.append(Tensor(np.random.rand(3, 1), autograd=True))

        for i in range(10):

            # Predict
            pred = data @ w[0] @ (w[1])

            # Compare
            loss = ((pred - target) * (pred - target)).sum(0)

            # Learn
            loss.backward(Tensor(np.ones_like(loss.data)))

            for w_ in w:
                w_.data = w_.data - w_.grad.data * 0.1
                w_.grad.data *= 0

            print(loss)


if __name__ == '__main__':
    unittest.main()
