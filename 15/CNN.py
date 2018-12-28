from MyLayers import *
import numpy as np 
import time

class SimpleCNN(object):

    def __init__(self, global_config):

        self.num_conv_layer = global_config['num_conv_layer']
        self.num_fc_layer = global_config['num_fc_layer']

        self.layers = {}
        self.initLayers(global_config)
    
    # 初始化各层
    def initLayers(self, global_config):
        for i in range(self.num_conv_layer):
            j = 1
            layer_name = 'conv' + str(i + 1) + '_' + str(j)
            
            while layer_name + '_num' in global_config:
                conv_num = global_config[layer_name + '_num']
                conv_shape = global_config.get(layer_name + '_shape', (3, 3, 3))
                conv_stride = global_config.get(layer_name + '_stride', 1)
                self.layers[layer_name] = ConvLayer(global_config, conv_num, conv_shape, conv_stride)

                if layer_name + '_weight' in global_config:
                    self.layers[layer_name].setWeight(global_config[layer_name + '_weight'])
                if layer_name + '_bias' in global_config:
                    self.layers[layer_name].setBias(global_config[layer_name + '_bias'])
                
                layer_name = 'conv_relu' + str(i + 1) + '_' + str(j)
                self.layers[layer_name] = ReluLayer()

                j += 1
                layer_name = 'conv' + str(i + 1) + '_' + str(j)

            layer_name = 'pool' + str(i + 1)
            pool_shape = global_config.get(layer_name + '_shape', (2, 2))
            pool_stride = global_config.get(layer_name + '_stride', 2)
            self.layers[layer_name] = MaxPoolLayer(pool_shape, pool_stride)

            layer_name = 'conv_dropout' + str(i + 1)
            if layer_name in global_config:
                p = global_config[layer_name]
                self.layers[layer_name] = DropoutLayer(p)
        
        for i in range(self.num_fc_layer):
            layer_name = 'fc' + str(i + 1)
            fc_shape = global_config[layer_name + '_shape']
            self.layers[layer_name] = FCLayer(global_config, fc_shape)

            if layer_name + '_weight' in global_config:
                self.layers[layer_name].setWeight(global_config[layer_name + '_weight'])
            if layer_name + '_bias' in global_config:
                self.layers[layer_name].setBias(global_config[layer_name + '_bias'])

            if i < self.num_fc_layer - 1:
                layer_name = 'fc_relu' + str(i + 1)
                self.layers[layer_name] = ReluLayer()

            layer_name = 'fc_dropout' + str(i + 1)
            if layer_name in global_config:
                p = global_config[layer_name]
                self.layers[layer_name] = DropoutLayer(p)

        self.layers['softmax'] = SoftmaxLayer()
    
    # 计算输出
    def calForward(self, input_data, is_train = True):
        x = input_data
        for i in range(self.num_conv_layer):
            j = 1
            layer_name = 'conv' + str(i + 1) + '_' + str(j)

            while layer_name in self.layers:
                self.layers[layer_name].setInput(x)
                x = self.layers[layer_name].calForward()

                layer_name = 'conv_relu' + str(i + 1) + '_' + str(j)
                self.layers[layer_name].setInput(x)
                x = self.layers[layer_name].calForward()

                j += 1
                layer_name = 'conv' + str(i + 1) + '_' + str(j)

            layer_name = 'pool' + str(i + 1)
            self.layers[layer_name].setInput(x)
            x = self.layers[layer_name].calForward()

            layer_name = 'conv_dropout' + str(i + 1)
            if layer_name in self.layers:
                self.layers[layer_name].setInput(x)
                x = self.layers[layer_name].calForward(is_train)
        
        for i in range(self.num_fc_layer):
            layer_name = 'fc' + str(i + 1)
            self.layers[layer_name].setInput(x)
            x = self.layers[layer_name].calForward()

            if i < self.num_fc_layer - 1:
                layer_name = 'fc_relu' + str(i + 1)
                self.layers[layer_name].setInput(x)
                x = self.layers[layer_name].calForward()

            layer_name = 'fc_dropout' + str(i + 1)
            if layer_name in self.layers:
                self.layers[layer_name].setInput(x)
                x = self.layers[layer_name].calForward(is_train)
        
        self.layers['softmax'].setInput(x)
        out, res = self.layers['softmax'].calForward()
        return out, res
    
    # 计算loss
    def calLoss(self, label):
        x = self.layers['softmax'].calBackward(label)
        return x
    
    # 计算准确率
    def calAccuracy(self, logit, label):
        N = label.shape[0]
        return np.sum(logit == label) / N
    
    # 误差逆传播
    def BP(self, global_config, dsoftmax):
        dout = dsoftmax
        for i in range(self.num_fc_layer, 0, -1):
            layer_name = 'fc_dropout' + str(i)
            if layer_name in self.layers:
                dout = self.layers[layer_name].calBackward(dout)
            
            if i < self.num_fc_layer:
                layer_name = 'fc_relu' + str(i)
                dout = self.layers[layer_name].calBackward(dout)

            layer_name = 'fc' + str(i)
            dout = self.layers[layer_name].updateAllParams(global_config, dout)
        
        for i in range(self.num_conv_layer, 0, -1):
            layer_name = 'conv_dropout' + str(i)
            if layer_name in self.layers:
                dout = self.layers[layer_name].calBackward(dout)
            
            layer_name = 'pool' + str(i)
            dout = self.layers[layer_name].calBackward(dout)

            j = 5
            layer_name = 'conv_relu' + str(i) + '_' + str(j)
            while layer_name not in self.layers:
                j -= 1
                layer_name = 'conv_relu' + str(i) + '_' + str(j)

            while layer_name in self.layers:
                dout = self.layers[layer_name].calBackward(dout)

                layer_name = 'conv' + str(i) + '_' + str(j)
                dout = self.layers[layer_name].updateAllParams(global_config, dout)

                j -= 1
                layer_name = 'conv_relu' + str(i) + '_' + str(j)
    
    # 跑一个batch
    def generateOneBatch(self, batch_data, label, global_config, is_train = True):
        out, logit = self.calForward(batch_data, is_train)
        loss, dsoftmax = self.calLoss(label)
        acc = self.calAccuracy(logit, label)

        if is_train:
            self.BP(global_config, dsoftmax)

        return loss, acc