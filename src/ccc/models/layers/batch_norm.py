import numpy as np

class BatchNorm2D:
    def __init__(self, num_channels: int, eps: float = 1e-5):
        self.gamma = np.ones((num_channels,), dtype=np.float32)
        self.beta = np.zeros((num_channels,), dtype=np.float32)
        self.eps = eps
        self.cache = None
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

    def forward(self, x):
        N, C, H, W = x.shape
        x_resh = x.transpose(1,0,2,3).reshape(C, -1)
        mean = x_resh.mean(axis=1, keepdims=True)
        var = x_resh.var(axis=1, keepdims=True)
        x_hat = (x_resh - mean) / np.sqrt(var + self.eps)
        x_hat = x_hat.reshape(C, N, H, W).transpose(1,0,2,3)
        out = self.gamma.reshape(1,-1,1,1) * x_hat + self.beta.reshape(1,-1,1,1)
        self.cache = (x, x_hat, mean, var)
        return out

    def backward(self, grad_out):
        x, x_hat, mean, var = self.cache
        N, C, H, W = x.shape
        m = N * H * W
        self.dbeta[...] = grad_out.sum(axis=(0,2,3))
        self.dgamma[...] = (grad_out * x_hat).sum(axis=(0,2,3))
        gamma = self.gamma.reshape(1,-1,1,1)
        dxhat = grad_out * gamma
        dxhat_resh = dxhat.transpose(1,0,2,3).reshape(C, -1)
        x_resh = x.transpose(1,0,2,3).reshape(C, -1)
        std_inv = 1.0 / np.sqrt(var + self.eps)
        dvar = (-0.5 * ((dxhat_resh * (x_resh - mean)).sum(axis=1, keepdims=True)) * (std_inv**3))
        dmean = (-(dxhat_resh * std_inv).sum(axis=1, keepdims=True) - 2.0 * dvar * (x_resh - mean).mean(axis=1, keepdims=True))
        dx_resh = dxhat_resh * std_inv + (2.0 / m) * dvar * (x_resh - mean) + dmean / m
        dx = dx_resh.reshape(C, N, H, W).transpose(1,0,2,3)
        return dx

    @property
    def params(self):
        return [self.gamma, self.beta]

    @property
    def grads(self):
        return [self.dgamma, self.dbeta]

