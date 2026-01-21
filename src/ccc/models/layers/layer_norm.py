import numpy as np

class LayerNorm1D:
    def __init__(self, features: int, eps: float = 1e-5):
        self.gamma = np.ones((features,), dtype=np.float32)
        self.beta = np.zeros((features,), dtype=np.float32)
        self.eps = eps
        self.cache = None
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

    def forward(self, x):
        mu = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1, keepdims=True)
        x_hat = (x - mu) / np.sqrt(var + self.eps)
        out = self.gamma * x_hat + self.beta
        self.cache = (x, x_hat, mu, var)
        return out

    def backward(self, grad_out):
        x, x_hat, mu, var = self.cache
        N, D = x.shape
        self.dbeta[...] = grad_out.sum(axis=0)
        self.dgamma[...] = (grad_out * x_hat).sum(axis=0)
        dxhat = grad_out * self.gamma
        std_inv = 1.0 / np.sqrt(var + self.eps)
        dvar = (-0.5 * (dxhat * (x - mu) * (std_inv**3))).sum(axis=1, keepdims=True)
        dmu = (-(dxhat * std_inv).sum(axis=1, keepdims=True) - 2.0 * dvar * (x - mu).mean(axis=1, keepdims=True))
        dx = dxhat * std_inv + (2.0 / D) * dvar * (x - mu) + dmu / D
        return dx

    @property
    def params(self):
        return [self.gamma, self.beta]

    @property
    def grads(self):
        return [self.dgamma, self.dbeta]

