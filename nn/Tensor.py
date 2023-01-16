import numpy as np


class Tensor:
    def __init__(self, data, autograd=False, parents=None, parents_operation=None):
        self.data = np.array(data)
        self.autograd = autograd
        self.parents = parents
        self.parents_operation = parents_operation
        self.grad = None
        self.children = None
        if self.parents is not None:
            for parent in self.parents:
                if parent.children is None:
                    parent.children = dict()
                if self not in parent.children:
                    parent.children[self] = 0
                parent.children[self] += 1

    def backward(self, grad=None, grad_vertex=None):
        if not self.autograd:
            return

        if grad is None:
            grad = Tensor([1])

        if self.grad is None:
            self.grad = Tensor([0])

        self.grad = self.grad + grad
        if self.children is not None:
            self.children[grad_vertex] -= 1

        if self.children is not None and max(self.children.values()) != 0:
            return

        if self.parents is None or self.parents_operation is None:
            return

        if self.parents_operation == "add":
            self.parents[0].backward(grad, self)
            self.parents[1].backward(grad, self)
        elif self.parents_operation == "sub":
            self.parents[0].backward(grad, self)
            self.parents[1].backward(-grad, self)
        elif self.parents_operation == "neg":
            self.parents[0].backward(-grad, self)
        elif self.parents_operation == "mul":
            self.parents[0].backward(self.grad * self.parents[1], self)
            self.parents[1].backward(self.grad * self.parents[0], self)

        for parent in self.parents:
            parent.children[self] += 1

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, autograd=True, parents=[self, other], parents_operation="add")
        return Tensor(self.data  + other.data)

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1, autograd=True, parents=[self], parents_operation="neg")
        return Tensor(self.data * -1)

    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data, autograd=True, parents=[self, other], parents_operation="sub")
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data, autograd=True, parents=[self, other], parents_operation="mul")
        return Tensor(self.data * other.data)
