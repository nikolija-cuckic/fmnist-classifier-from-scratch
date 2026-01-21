import numpy as np

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0).astype(x.dtype)
        return x * self.mask

    def backward(self, grad_out):
        return grad_out * self.mask

    @property
    def params(self):
        return []

    @property
    def grads(self):
        return []

