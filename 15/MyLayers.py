import numpy as np
from math import sqrt
from Optimizer import sgd_momentum, adam
from Utils import *

def calError(dx, n_dx):
    now = dx - n_dx
    now = now.reshape((now.shape[0], -1))
    return np.linalg.norm(now, 2)

class Layer(object):

    def __init__(self, type_):
        self.input = None
        self.type = type_
    
    def setInput(self, in_):
        self.input = in_
    
    def getType(self):
        return self.type

# 卷积层
class ConvLayer(Layer):
    
    def __init__(self, global_config, conv_num, shape = (3, 3, 3), stride = 1):
        super().__init__('conv')
        self.weight = np.random.randn(conv_num, shape[0], shape[1], shape[2]) / (sqrt(shape[1] * shape[2]) * 0.5)
        self.bias = np.zeros(conv_num)
        self.stride = stride
        self.weight_config = {}
        self.bias_config = {}
        self.img_col = None

        self.optimizer = global_config['optimizer']
        self.pad = global_config['pad']
        self.initConfig()
    
    def initConfig(self):
        self.weight_config.setdefault('v', np.zeros_like(self.weight))
        self.bias_config.setdefault('v', np.zeros_like(self.bias))

        # 记录adam相关的向量
        if self.optimizer == 'adam':
            self.weight_config.setdefault('m', np.zeros_like(self.weight))
            self.weight_config.setdefault('t', 1)
            self.bias_config.setdefault('m', np.zeros_like(self.bias))
            self.bias_config.setdefault('t', 1)
            self.optimizer = adam
        else:
            self.optimizer = sgd_momentum
    
    def setWeight(self, weight):
        self.weight = weight
    
    def setBias(self, bias):
        self.bias = bias

    # 前向传播（卷积用了im2col）
    def calForward(self):
        x = self.input
        HH, WW = self.weight.shape[2], self.weight.shape[3]
        (N, C, H, W) = x.shape

        H_, W_ = int(1 + (H + 2 * self.pad - HH) / self.stride), int(1 + (W + 2 * self.pad - WW) / self.stride)
        output = np.random.randn(N, self.weight.shape[0], H_, W_)
        
        col_weights = self.weight.reshape([self.weight.shape[0], -1])
        image_col = im2col(x, HH, WW, self.pad, self.stride)
        res = np.dot(col_weights, image_col)

        for fi in range(self.weight.shape[0]):
            res[fi, :] += self.bias[fi]
            output[:, fi, :, :] = res[fi].reshape((N, H_, W_))
        
        self.img_col = image_col

        return output
    
    # 反向传播
    def calBackward(self, dout):
        x = self.input
        dx, dw, db = None, None, None
        (F, _, HH, WW) = self.weight.shape
        (N, C, H, W) = x.shape

        db = np.sum(dout, axis = (0, 2, 3))

        dout_reshaped = dout.transpose(1, 0, 2, 3).reshape((F, -1))
        dw = np.dot(dout_reshaped, self.img_col.T)
        dw = dw.reshape(self.weight.shape)
        
        dx_cols = np.dot(self.weight.reshape((F, -1)).T, dout_reshaped)
        dx = col2im(dx_cols, C, HH, WW, N, H, W, self.stride, self.pad)

        return dx, dw, db
    
    # 更新参数
    def updateAllParams(self, global_config, dout):
        dx, dw, db = self.calBackward(dout)

        next_w, next_w_config = self.optimizer(self.weight, dw, global_config, self.weight_config)
        self.weight = next_w
        self.weight_config = next_w_config

        next_b, next_b_config = self.optimizer(self.bias, db, global_config, self.bias_config)
        self.bias = next_b
        self.bias_config = next_b_config

        return dx

# 最大池化层
class MaxPoolLayer(Layer):

    def __init__(self, shape = (2, 2), stride = 2):
        super().__init__('maxpool')
        self.shape = shape
        self.stride = stride
        self.out_arg_max = None
        self.x_cols = None
    
    # 前向传播（im2col）
    def calForward(self):
        out = None
        x = self.input

        pool_height = self.shape[0]
        pool_width = self.shape[1]
        stride_ = self.stride

        (N, C, H, W) = x.shape

        H_, W_ = int(1 + (H - pool_height) / stride_), int(1 + (W - pool_width) / stride_)
        
        x_split = x.reshape(N * C, 1, H, W)
        x_cols = im2col(x_split, pool_height, pool_width, padding = 0, stride = stride_)

        out_arg_max = np.argmax(x_cols, axis = 0)

        out = x_cols[out_arg_max, np.arange(x_cols.shape[1])]
        out = out.reshape((N, C, H_, W_))

        self.out_arg_max = out_arg_max
        self.x_cols = x_cols
        
        return out

    # 反向传播
    def calBackward(self, dout):
        dx = None
        x = self.input

        pool_height = self.shape[0]
        pool_width = self.shape[1]
        stride = self.stride

        (N, C, H, W) = x.shape

        dx = np.zeros_like(self.x_cols)
        dout_reshaped = dout.flatten()
        dx[self.out_arg_max, np.arange(dx.shape[1])] = dout_reshaped
        dx = col2im(dx, 1, pool_height, pool_width, N * C, H, W, stride, padding = 0)
        dx = dx.reshape(x.shape)

        return dx

# Relu
class ReluLayer(Layer):
    
    def __init__(self):
        super().__init__('relu')
    
    def calForward(self):
        return np.maximum(self.input, 0)

    def calBackward(self, dout):
        dx = dout
        dx[self.input <= 0] = 0
        return dx

# Dropout
class DropoutLayer(Layer):

    def __init__(self, p = 0.9):
        super().__init__('dropout')
        self.p = p
        self.mask = None
    
    def calForward(self, is_train = True):
        np.random.seed(0) 
        out = None
        x = self.input

        if is_train:
            retain_prob = 1.0 - self.p
            self.mask = np.random.binomial(n = 1, p = retain_prob, size = x.shape)
            out = x * self.mask
        else:
            out = x

        out = out.astype(x.dtype, copy = False)

        return out

    def calBackward(self, dout, is_train = True):
        dx = None

        if is_train:
            dx = dout * self.mask
        else:
            dx = dout
        return dx

# 全连接层
class FCLayer(Layer):

    def __init__(self, global_config, shape):
        super().__init__('fc')
        self.weight = np.random.randn(shape[0], shape[1]) / (sqrt(shape[0] * shape[1]) * 0.5)
        self.bias = np.zeros(shape[1])

        self.optimizer = global_config['optimizer']
        self.weight_config = {}
        self.bias_config = {}
        self.initConfig()
    
    def initConfig(self):
        self.weight_config.setdefault('v', np.zeros_like(self.weight))
        self.bias_config.setdefault('v', np.zeros_like(self.bias))

        # adam相关参数
        if self.optimizer == 'adam':
            self.weight_config.setdefault('m', np.zeros_like(self.weight))
            self.weight_config.setdefault('t', 1)
            self.bias_config.setdefault('m', np.zeros_like(self.bias))
            self.bias_config.setdefault('t', 1)
            self.optimizer = adam
        else:
            self.optimizer = sgd_momentum
    
    def setWeight(self, weight):
        self.weight = weight
    
    def setBias(self, bias):
        self.bias = bias
    
    def calForward(self):
        x = self.input

        x_tmp = np.reshape(x, (x.shape[0], -1))
        out = np.dot(x_tmp, self.weight) + self.bias

        return out
    
    def calBackward(self, dout):
        x = self.input

        dx = np.dot(dout, self.weight.T)
        dx = np.reshape(dx, x.shape)

        dw = np.dot(np.reshape(x, (x.shape[0], -1)).T, dout)

        db = np.sum(dout, axis = 0)
        return dx, dw, db
    
    def updateAllParams(self, global_config, dout):
        dx, dw, db = self.calBackward(dout)

        next_w, next_w_config = self.optimizer(self.weight, dw, global_config, self.weight_config)
        self.weight = next_w
        self.weight_config = next_w_config

        next_b, next_b_config = self.optimizer(self.bias, db, global_config, self.bias_config)
        self.bias = next_b
        self.bias_config = next_b_config

        return dx 

# softmax
class SoftmaxLayer(Layer):

    def __init__(self):
        super().__init__('softmax')

    def calForward(self):
        x = self.input
        
        out = np.exp(x - np.max(x, axis = 1, keepdims = True))
        exp_sum = np.sum(out, axis = 1, keepdims = True)

        out = out / exp_sum
        return out, np.argmax(out, axis = 1)
    
    def calBackward(self, label):
        x = self.input

        shifted_logits = x - np.max(x, axis = 1, keepdims = True)
        Z = np.sum(np.exp(shifted_logits), axis = 1, keepdims = True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        N = x.shape[0]

        loss = -np.sum(log_probs[np.arange(N), label]) / N
        dx = probs.copy()
        dx[np.arange(N), label] -= 1
        dx /= N

        return loss, dx