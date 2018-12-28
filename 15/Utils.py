import numpy as np 

def im2col(x, HH, WW, padding = 1, stride = 1):
    p = padding
    (N, C, H, W) = x.shape
    x_pad = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode = 'constant')
    H_, W_ = int(1 + (H + 2 * p - HH) / stride), int(1 + (W + 2 * p - WW) / stride)
    img_col = np.zeros((C * HH * WW, N * H_ * W_))
    for n in range(N):
        for i in range(0, H_):
            for j in range(0, W_):
                index = n * H_ * W_ + i * W_ + j
                img_col[:, index] = x_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW].reshape(C * HH * WW)
    return img_col

def col2im(x_cols, C, HH, WW, N, H, W, stride = 1, padding = 1):
    p = padding
    x = np.zeros((N, C, H, W))
    x_pad = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode = 'constant')
    H_, W_ = int(1 + (H + 2 * p - HH) / stride), int(1 + (W + 2 * p - WW) / stride)

    for n in range(N):
        for i in range(H_):
            for j in range(W_):
                index = n * H_ * W_ + i * W_ + j
                x_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] = x_cols[:, index].reshape((C, HH, WW))
    
    return x_pad[:, :, p:p + H, p:p + W]