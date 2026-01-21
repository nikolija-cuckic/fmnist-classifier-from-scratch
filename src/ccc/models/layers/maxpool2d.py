import numpy as np
from .utils import get_output_size, im2col_indices, col2im_indices

class MaxPool2D:
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.x = None
        self.argmax = None
        self.out_h = None
        self.out_w = None

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        k = self.kernel_size
        s = self.stride
        out_h = get_output_size(H, k, 0, s)
        out_w = get_output_size(W, k, 0, s)
        self.out_h, self.out_w = out_h, out_w
        x_reshaped = x.reshape(N * C, 1, H, W)
        cols, x_shape, _, _ = im2col_indices(x_reshaped, k, k, 0, s)
        cols = cols.reshape(k*k, out_h * out_w, N * C)
        self.argmax = np.argmax(cols, axis=0)
        max_vals = cols.max(axis=0)
        out = max_vals.T.reshape(N, C, out_h, out_w)
        return out

    def backward(self, grad_out):
        N, C, H, W = self.x.shape
        k = self.kernel_size
        s = self.stride
        out_h, out_w = self.out_h, self.out_w
        dcols = np.zeros((k*k, out_h * out_w, N * C), dtype=self.x.dtype)
        go = grad_out.reshape(N*C, out_h * out_w).T
        dcols[self.argmax, np.arange(out_h * out_w)[:,None], np.arange(N*C)[None,:]] = go
        dcols = dcols.reshape(k*k, out_h*out_w*N*C)
        dx = col2im_indices(dcols, (N*C, 1, H, W), k, k, 0, s)
        dx = dx.reshape(N, C, H, W)
        return dx

    @property
    def params(self):
        return []

    @property
    def grads(self):
        return []

