import numpy as np


class Tensor:
    def __init__(self, data):
        self.data = np.array(data)

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()

    def __add__(self, other):
        return Tensor(self.data + other.data)

    def __neg__(self):
        return Tensor(self.data * -1)

    def __sub__(self, other):
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        return Tensor(self.data * other.data)
