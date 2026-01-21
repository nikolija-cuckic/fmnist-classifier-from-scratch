import numpy as np
from .utils import im2col_indices, col2im_indices

class Conv2D:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        kh = kw = kernel_size
        fan_in = in_channels * kh * kw
        scale = np.sqrt(2.0 / fan_in)
        self.W = (np.random.randn(out_channels, in_channels, kh, kw).astype(np.float32)) * scale
        self.b = np.zeros((out_channels,), dtype=np.float32)
        self.x_shape = None
        self.cols = None
        self.W_col = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        N, C, H, W = x.shape
        kh = kw = self.kernel_size
        cols, x_shape, out_h, out_w = im2col_indices(x, kh, kw, self.padding, self.stride)
        self.x_shape = x_shape
        self.cols = cols
        W_col = self.W.reshape(self.out_channels, -1)
        self.W_col = W_col
        out = (W_col @ cols + self.b.reshape(-1,1))
        out = out.reshape(self.out_channels, out_h, out_w, N).transpose(3,0,1,2)
        return out

    def backward(self, grad_out):
        N = grad_out.shape[0]
        grad_out_reshaped = grad_out.transpose(1,2,3,0).reshape(self.out_channels, -1)
        self.db[...] = grad_out_reshaped.sum(axis=1)
        self.dW[...] = (grad_out_reshaped @ self.cols.T).reshape(self.W.shape)
        dcols = self.W_col.T @ grad_out_reshaped
        dx = col2im_indices(dcols, self.x_shape, self.kernel_size, self.kernel_size, self.padding, self.stride)
        return dx

    @property
    def params(self):
        return [self.W, self.b]

    @property
    def grads(self):
        return [self.dW, self.db]

