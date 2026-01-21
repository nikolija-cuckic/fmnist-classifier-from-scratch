import numpy as np

class GlobalAvgPool2D:
    def __init__(self):
        self.H = None
        self.W = None

    def forward(self, x):
        self.H, self.W = x.shape[2], x.shape[3]
        return x.mean(axis=(2, 3))

    def backward(self, grad_out):
        N, C = grad_out.shape
        scale = 1.0 / (self.H * self.W)
        return grad_out[:, :, None, None] * scale * np.ones((N, C, self.H, self.W), dtype=grad_out.dtype)

    @property
    def params(self):
        return []

    @property
    def grads(self):
        return []

