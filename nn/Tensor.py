import numpy as np


class Tensor:
    def __init__(self, data, autograd=False, parents=None, parents_operation=None):
        self.data = np.array(data)
        self.autograd = autograd
        self.parents = parents
        self.parents_operation = parents_operation
        self.grad = None
        self.children = dict()
        self.__update_parents()

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()

    def __update_parents(self):
        if self.parents is None:
            return

        if self not in self.parents[0].children:
            self.parents[0].children[self] = 0
        self.parents[0].children[self] += 1

        if self.parents_operation not in {"neg", "pow", "transpose", "expand", "sum"}:
            if self not in self.parents[1].children:
                self.parents[1].children[self] = 0
            self.parents[1].children[self] += 1

    def backward(self, grad=None, grad_vertex=None):
        if not self.autograd:
            return

        if grad is None:
            grad = Tensor([1])

        if self.grad is None:
            self.grad = Tensor([0])

        self.grad = self.grad + grad
        if grad_vertex is not None:
            self.children[grad_vertex] -= 1

        if len(self.children) != 0 and max(self.children.values()) != 0:
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
        elif self.parents_operation == "pow":
            power = self.parents[1]
            new_tensor = self.grad * (self.parents[0] ** (power - 1))
            new_tensor.data *= power
            self.parents[0].backward(new_tensor)
        elif self.parents_operation == "transpose":
            self.parents[0].backward(self.grad.transpose())
        elif self.parents_operation[:3] == "sum":
            dim = self.parents[1]
            ds = self.parents[0].data.shape[dim]
            self.parents[0].backward(self.grad.expand(dim, ds))
        elif self.parents_operation[:6] == "expand":
            dim = self.parents[1]
            self.parents[0].backward(self.grad.sum(dim))
        elif self.parents_operation == "matmul":
            self.parents[0].backward(self.grad @ self.parents[1].transpose())
            self.parents[1].backward(self.parents[0].transpose() @ self.grad)

        self.__update_parents()

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

    def __pow__(self, power):
        if self.autograd:
            return Tensor(self.data ** power, autograd=True, parents=[self, power], parents_operation="pow")
        return Tensor(self.data ** power)

    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(), autograd=True, parents=[self], parents_operation="transpose")
        return Tensor(self.data.transpose())

    def sum(self, dim):
        if self.autograd:
            return Tensor(self.data.sum(dim), autograd=True, parents=[self, dim], parents_operation=f"sum")
        return Tensor(self.data.sum(dim))

    def expand(self, dim, copies):
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)

        if self.autograd:
            return Tensor(new_data, autograd=True, parents=[self, dim], parents_operation=f"expand")
        return Tensor(new_data)

    def __matmul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data @ other.data, autograd=True, parents=[self, other], parents_operation="matmul")
        return Tensor(self.data @ other.data)

