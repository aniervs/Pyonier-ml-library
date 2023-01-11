import numpy as np


class Tensor:
    def __init__(self, data, parents=None, parents_operation=None):
        self.data = np.array(data)
        self.parents = parents
        self.parents_operation = parents_operation
        self.grad = None

    def backward(self, grad=None):
        if grad is None:
            grad = Tensor([1])
        self.grad = grad

        if self.parents_operation == "add":
            self.parents[0].backward(grad)
            self.parents[1].backward(grad)
        elif self.parents_operation == "sub":
            self.parents[0].backward(grad)
            self.parents[1].backward(-grad)
        elif self.parents_operation == "neg":
            self.parents[0].backward(-grad)
        elif self.parents_operation == "mul":
            self.parents[0].backward(self.grad * self.parents[1])
            self.parents[1].backward(self.grad * self.parents[0])

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()

    def __add__(self, other):
        return Tensor(self.data + other.data, parents=[self, other], parents_operation="add")

    def __neg__(self):
        return Tensor(self.data * -1, parents=[self], parents_operation="neg")

    def __sub__(self, other):
        return Tensor(self.data - other.data, parents=[self, other], parents_operation="sub")

    def __mul__(self, other):
        return Tensor(self.data * other.data, parents=[self, other], parents_operation="mul")
