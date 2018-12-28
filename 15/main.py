import numpy as np
import sys
from CNN import SimpleCNN
from DataHandler import DataHandler
import time

def main(global_config):
    epochs = global_config['epochs']
    batch_size = global_config['batch_size']
    epoch_len = global_config['data_shape'][0]

    print("Read Training Data ...")
    sys.stdout.flush()
    train_data_handler = DataHandler()
    train_data_handler.readData()
    train_data_handler.normalize()
    print("Done ...")
    sys.stdout.flush()
    
    print("Read Testing Data ...")
    sys.stdout.flush()
    test_data_handler = DataHandler()
    test_data_handler.readData('test')
    test_data_handler.normalize()
    print("Done ...")
    sys.stdout.flush()

    test_data, test_label = test_data_handler.getBatch(0, 10000)

    cnn = SimpleCNN(global_config)
    best_acc = 0.0
    for epoch in range(epochs):
        start = time.time()
        i = 0
        epoch_loss, epoch_acc = 0.0, 0.0
        if i % 5 == 0 and i > 0:
            global_config['lr'] /= 2

        while i * batch_size < epoch_len:
            begin = i * batch_size
            batch_data, batch_label = train_data_handler.getBatch(begin, batch_size)
            loss, acc = cnn.generateOneBatch(batch_data, batch_label, global_config)
            print("Train Epoch %d iter %d Loss: %.4f, Accuracy: %.4f"%(epoch + 1, i + 1, loss, acc))
            sys.stdout.flush()

            epoch_loss += loss * batch_size
            epoch_acc += acc * batch_size
            i += 1

        epoch_loss /= epoch_len
        epoch_acc /= epoch_len
        print("Train Epoch %d Loss: %.4f, Accuracy: %.4f, Time: %.4f"%(epoch + 1, epoch_loss, epoch_acc, time.time()- start))
        sys.stdout.flush()

        test_loss, test_acc = cnn.generateOneBatch(test_data, test_label, global_config, is_train = False)
        # 每训练一个epoch进行一次测试，最后会输出最优正确率（暂无保存模型的方法）
        print("Test Loss: %.4f, Accuracy: %.4f"%(test_loss, test_acc))
        sys.stdout.flush()
        if best_acc < test_acc:
            best_acc = test_acc
    
    print("Best Accuracy: %.4f"%(best_acc))
    sys.stdout.flush()

if __name__ == '__main__':
    global_config = {}

    global_config['lr'] = 2e-3
    global_config['batch_size'] = 50
    global_config['epochs'] = 10
    global_config['data_shape'] = (50000, 3, 32, 32)

    global_config['optimizer'] = 'adam'
    global_config['beta1'] = 0.9
    global_config['beta2'] = 0.99
    global_config['epsilon'] = 1e-8

    global_config['pad'] = 1

    # 可修改以下配置
    # input-> conv1(32 * 3 * 3 * 3)-> pool1-> conv2(64 * 32 * 3 * 3)-> 
    # pool2-> fc1(4096 * 1024)-> fc2(1024 * 10)-> dropout
    global_config['num_conv_layer'] = 2
    global_config['num_fc_layer'] = 2

    global_config['conv1_1_num'] = 32
    global_config['conv1_1_shape'] = (3, 3, 3)
    global_config['conv1_1_stride'] = 1

    global_config['pool1_shape'] = (2, 2)
    global_config['pool1_stride'] = 2

    global_config['conv2_1_num'] = 64
    global_config['conv2_1_shape'] = (32, 3, 3)
    global_config['conv2_1_stride'] = 1

    global_config['pool2_shape'] = (2, 2) 
    global_config['pool2_stride'] = 2

    global_config['fc1_shape'] = (4096, 1024)
    global_config['fc2_shape'] = (1024, 10)
    global_config['fc_dropout1'] = 0.9
    main(global_config)