import numpy as np
import pickle

# 读取并处理数据
class DataHandler(object):

    def __init__(self):
        self.data = None
        self.label = None
        self.road = 'datasets/cifar-10-batches-py'
    
    # 读取数据
    def readData(self, mode = 'train'):
        if mode == 'train':
            for i in range(1, 6):
                now_road = self.road + '/data_batch_' + str(i)
                with open(now_road, 'rb') as fo:
                    now_dict = pickle.load(fo, encoding = 'bytes')
                if i == 1:
                    self.data = now_dict['data'.encode()]
                    self.label = np.array(now_dict['labels'.encode()])
                    self.label = self.label.reshape((self.label.shape[0], 1))
                else:
                    self.data = np.concatenate((self.data, now_dict['data'.encode()]), axis = 0)
                    now_label = np.array(now_dict['labels'.encode()])
                    now_label = now_label.reshape((now_label.shape[0], 1))
                    self.label = np.concatenate((self.label, now_label), axis = 0)

            self.data = self.data.reshape((self.data.shape[0], 3, 1024)).astype(np.float64)
            self.label = self.label.reshape((self.label.shape[0])).astype(np.int64)
        else:
            with open(self.road + '/test_batch', 'rb') as fo:
                now_dict = pickle.load(fo, encoding = 'bytes')
                self.data = now_dict['data'.encode()]
                self.label = np.array(now_dict['labels'.encode()]).astype(np.int64)
            self.data = self.data.reshape((self.data.shape[0], 3, 1024)).astype(np.float64)
    
    # 归一化
    def normalize(self):
        (N, C, _) = self.data.shape
        for i in range(N):
            for j in range(C):
                now_mean = np.mean(self.data[i, j])
                now_std = np.std(self.data[i, j])
                self.data[i, j] = (self.data[i, j] - now_mean) / now_std
    
    # 得到一个batch
    def getBatch(self, begin, batch_size):
        now_data = self.data[begin:begin + batch_size].reshape((batch_size, 3, 32, 32))
        return now_data, self.label[begin:begin + batch_size]

if __name__ == '__main__':
    data_handler = DataHandler()
    data_handler.readData('train')
    # data_handler.normalize()
    print(data_handler.getBatch(0, 40))