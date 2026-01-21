import numpy as np

def get_output_size(in_size: int, kernel_size: int, padding: int, stride: int) -> int:
    return (in_size + 2 * padding - kernel_size) // stride + 1

def im2col_indices(x, field_height, field_width, padding, stride):
    N, C, H, W = x.shape
    out_h = get_output_size(H, field_height, padding, stride)
    out_w = get_output_size(W, field_width, padding, stride)

    p = padding
    x_padded = np.pad(x, ((0,0),(0,0),(p,p),(p,p)), mode='constant')

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_h), out_w)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_w), out_h)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    cols = x_padded[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(C * field_height * field_width, -1)
    return cols, (N, C, H, W), out_h, out_w

def col2im_indices(cols, x_shape, field_height, field_width, padding, stride):
    N, C, H, W = x_shape
    out_h = get_output_size(H, field_height, padding, stride)
    out_w = get_output_size(W, field_width, padding, stride)
    p = padding
    H_p, W_p = H + 2*p, W + 2*p
    x_padded = np.zeros((N, C, H_p, W_p), dtype=cols.dtype)

    cols_reshaped = cols.reshape(C * field_height * field_width, out_h * out_w, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_h), out_w)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_w), out_h)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if p == 0:
        return x_padded
    return x_padded[:, :, p:-p, p:-p]

