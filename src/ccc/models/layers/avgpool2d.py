import numpy as np
from .utils import get_output_size, im2col_indices, col2im_indices

class AvgPool2D:
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.x_shape = None
        self.out_h = None
        self.out_w = None

    def forward(self, x):
        N, C, H, W = x.shape
        k = self.kernel_size
        s = self.stride
        out_h = get_output_size(H, k, 0, s)
        out_w = get_output_size(W, k, 0, s)
        self.out_h, self.out_w = out_h, out_w
        self.x_shape = (N, C, H, W)

        x_reshaped = x.reshape(N * C, 1, H, W)
        cols, _, _, _ = im2col_indices(x_reshaped, k, k, 0, s)
        cols = cols.reshape(k * k, out_h * out_w, N * C)
        mean_vals = cols.mean(axis=0)
        out = mean_vals.T.reshape(N, C, out_h, out_w)
        return out

    def backward(self, grad_out):
        N, C, H, W = self.x_shape
        k = self.kernel_size
        s = self.stride
        out_h, out_w = self.out_h, self.out_w
        go = grad_out.reshape(N * C, out_h * out_w).T  
        dcols = np.tile(go[None, :, :], (k * k, 1, 1)) / (k * k)
        dcols = dcols.reshape(k * k, out_h * out_w * N * C)
        dx = col2im_indices(dcols, (N * C, 1, H, W), k, k, 0, s)
        dx = dx.reshape(N, C, H, W)
        return dx

    @property
    def params(self):
        return []

    @property
    def grads(self):
        return []

