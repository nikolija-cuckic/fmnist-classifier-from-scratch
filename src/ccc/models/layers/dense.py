import numpy as np

class Dense:
    def __init__(self, in_features: int, out_features: int):
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(out_features, in_features).astype(np.float32) * scale
        self.b = np.zeros((out_features,), dtype=np.float32)
        self.x = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, grad_out):
        self.dW[...] = grad_out.T @ self.x
        self.db[...] = grad_out.sum(axis=0)
        grad_x = grad_out @ self.W
        return grad_x

    @property
    def params(self):
        return [self.W, self.b]

    @property
    def grads(self):
        return [self.dW, self.db]
