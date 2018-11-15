import numpy as np
import random
from math import pi, exp, sqrt
from scipy import stats
from decimal import Decimal

class Node(object):
    def __init__(self, data_index, classfy = None, is_leaf = False):
        self.data_index = data_index
        self.classfy = classfy
        self.is_leaf = is_leaf
        if is_leaf:
            self.son = None
        else:
            self.son = []
        self.divide_feature_index = None
    
    def setDivideFeatureAndValue(self, divide_feature_index, divide_value):
        self.divide_feature_index = divide_feature_index
        self.divide_value = divide_value

class Tree(object):
    def __init__(self, features, labels, max_continous_son, max_leaf_length, tree_type = 'ID3'):
        self.tree_type = tree_type
        self.features = features
        self.labels = labels
        self.feature_length = len(self.features[0])
        self.max_continous_son = max_continous_son
        self.max_leaf_length = max_leaf_length
        self.root = Node(list(range(0, len(features))))
    
    def setContinuousIndex(self, index_list):
        self.continous_list = index_list
    
    def isContinuous(self, index):
        return index in self.continous_list

    # 设置max_son
    def setMaxSon(self, max_continous_son):
        self.max_continous_son = max_continous_son
    
    # 设置max_leaf_data
    def setMaxLeafLength(self, max_leaf_length):
        self.max_leaf_length = max_leaf_length
    
    # 清空root
    def deleteRoot(self):
        self.root = Node(list(range(0, len(self.features))))
    
    # 对数据根据其中一个特征进行划分
    def getSubIndex(self, data_index, feature_index):

        if self.isContinuous(feature_index):
            now_data = [self.features[x][feature_index] for x in data_index]
            max_value = max(now_data) + 1
            min_value = min(now_data)
            step = (max_value - min_value) / self.max_continous_son
            sub_data_indexs = [[] for x in range(self.max_continous_son)]
            for i in data_index:
                for j in range(0, self.max_continous_son):
                    if self.features[i][feature_index] < min_value + (j + 1) * step:
                        sub_data_indexs[j].append(i)
                        break

            for i in range(self.max_continous_son - 1, 0, -1):
                if len(sub_data_indexs[i]) == 0:
                    sub_data_indexs.pop(i)

            return (min_value, step), sub_data_indexs

        sub_data_indexs = {}
        for i in data_index:
            if self.features[i][feature_index] == '?':
                continue
            if self.features[i][feature_index] not in sub_data_indexs:
                sub_data_indexs[self.features[i][feature_index]] = [i]
            else:
                sub_data_indexs[self.features[i][feature_index]].append(i)
        return None, sub_data_indexs
    
    def getCount(self, data_index):
        count = {}
        add_count = 1 / len(data_index)
        for i in data_index:
            if self.labels[i] not in count:
                count[self.labels[i]] = add_count
            else:
                count[self.labels[i]] += add_count
        return count

    # 计算信息熵
    def getEntropy(self, count):
        return -sum([count[x] * np.log2(count[x]) for x in count])

    # 计算信息增益
    def getGain(self, data_index, sub_data_indexs, feature_index):
        now_entropy = self.getEntropy(self.getCount(data_index))
        now_data_len = len(data_index)
        try:
            if self.isContinuous(feature_index):
                sub_entropy_sum = sum([len(x) / now_data_len * self.getEntropy(self.getCount(x)) for x in sub_data_indexs])
            else:
                sub_entropy_sum = sum([len(y) / now_data_len * self.getEntropy(self.getCount(y)) for _, y in sub_data_indexs.items()])
            return now_entropy - sub_entropy_sum
        except Exception as e:
            print(e)

    # 计算属性的固有值
    def getIV(self, data_index, sub_data_indexs, feature_index):
        now_data_len = len(data_index)
        if self.isContinuous(feature_index):
            return -sum([len(x) / now_data_len * np.log2(len(x) / now_data_len) for x in sub_data_indexs])
        return -sum([len(y) / now_data_len * np.log2(len(y) / now_data_len) for _, y in sub_data_indexs.items()])
    
    # 计算属性的信息增益率
    def getGainRadio(self, data_index, sub_data_indexs, feature_index):
        return self.getGain(data_index, sub_data_indexs, feature_index) / self.getIV(data_index, sub_data_indexs, feature_index)
    
    # 设定结点类别
    def setClassfy(self, now_root, now_counts):
        num = -1
        classfy = -1
        for x in now_counts:
            if num < now_counts[x]:
                classfy = x
                num = now_counts[x]
        now_root.classfy = classfy
    
    def isSameValue(self, now_root):
        feature_index = now_root.divide_feature_index
        if feature_index == None:
            return False
        now_data = [self.features[x][feature_index] for x in now_root.data_index]
        return max(now_data) == min(now_data)
    
    # 建立新节点
    def buildNode(self, now_root):
        now_counts = self.getCount(now_root.data_index)
        if now_counts == None or len(now_counts) == 0:
            return
        if len(now_root.data_index) <= self.max_leaf_length or len(now_counts) == 1:
            now_root.is_leaf = True
            self.setClassfy(now_root, now_counts)
            return

        best_sub_data_indexs = None
        best_gain = float('-inf')
        best_divide_feature_index = -1
        best_value = None

        for i in range(self.feature_length):
            now_value, now_sub_data_indexs = self.getSubIndex(now_root.data_index, i)
            if self.tree_type == 'ID3':
                now_gain = self.getGain(now_root.data_index, now_sub_data_indexs, i)
            elif self.tree_type == 'C4.5':
                now_gain = self.getGainRadio(now_root.data_index, now_sub_data_indexs, i)

            if best_gain < now_gain:
                best_divide_feature_index = i
                best_gain = now_gain
                best_sub_data_indexs = now_sub_data_indexs
                best_value = now_value

        for i, key in enumerate(best_sub_data_indexs):
            if self.isContinuous(best_divide_feature_index):
                new_son = Node(key)
                new_son.setDivideFeatureAndValue(best_divide_feature_index, best_value[0] + best_value[1] * (i + 1))
            else:
                new_son = Node(best_sub_data_indexs[key])
                new_son.setDivideFeatureAndValue(best_divide_feature_index, key)
            self.buildNode(new_son)
            now_root.son.append(new_son)
    
    # 训练
    def train(self):
        self.buildNode(self.root)
    
    # 预测的子函数
    def predictNode(self, now_node, test_feature):
        if now_node.is_leaf:
            return now_node.classfy
        for i in range(len(now_node.son)):
            now_divide_feature_index = now_node.son[i].divide_feature_index

            if self.isContinuous(now_divide_feature_index):
                if test_feature[now_divide_feature_index] < now_node.son[i].divide_value:
                    return self.predictNode(now_node.son[i], test_feature)
                elif i == len(now_node.son) - 1 and test_feature[now_divide_feature_index] >= now_node.son[i].divide_value:
                    return self.predictNode(now_node.son[i], test_feature)

            else:
                if test_feature[now_divide_feature_index] == now_node.son[i].divide_value:
                    return self.predictNode(now_node.son[i], test_feature)
        return self.labels[random.randint(0, len(self.labels) - 1)]

    def predictAll(self, test_features, test_labels):
        res = 0
        for i, test_feature in enumerate(test_features):
            pred = self.predictNode(self.root, test_feature)
            if pred == test_labels[i]:
                res += 1
        return res / len(test_labels)
    
    def predictAllDetail(self, test_features, test_labels, label_index):
        res = 0
        for i, test_feature in enumerate(test_features):
            pred = self.predictNode(self.root, test_feature)
            print("Pred: %s, Label: %s"%(label_index[pred], label_index[test_labels[i]]), end = " ")
            if pred == test_labels[i]:
                print("True")
                res += 1
            else:
                print("False")
        print("Accuracy: %.4f"%(res / len(test_labels)))

# read data
def readData(train_data, train_label, test_data, test_label, label_dict):

    def isContinuous(i):
        return i in [0, 2, 4, 10, 11, 12]

    # read training data
    with open("adult.data", "r", encoding = 'utf8') as f:
        j = 0
        for line in f.readlines():
            try:
                nowData = str(line)[:-1].replace(' ', '').split(',')
                if nowData[-1] == '':
                    continue
                train_label[j] = label_dict[nowData[-1]]
                for i, data in enumerate(nowData[:-1]):
                    if isContinuous(i):
                        train_data[j].append(int(data))
                    else:
                        train_data[j].append(data)
                j += 1
            except Exception as e:
                print("Error occured in line %d"%j)
                print(e)
    
    # read testing data
    with open("adult.test", "r", encoding = 'utf8') as f:
        j = 0
        flag = 0
        for line in f.readlines():
            try:
                if flag == 0:
                    flag = 1
                    continue
                nowData = str(line)[:-2].replace(' ', '').split(',')
                if nowData[-1] == '':
                    continue
                test_label[j] = label_dict[nowData[-1]]
                for i, data in enumerate(nowData[:-1]):
                    if isContinuous(i):
                        test_data[j].append(int(data))
                    else:
                        test_data[j].append(data)
                j += 1
            except Exception as e:
                print("Error occured in line %d"%j)
                print(e)

if __name__ == '__main__':
    train_data = [[] for _ in range(32561)]
    train_label = [0 for _ in range(32561)]
    test_data = [[] for _ in range(16281)]
    test_label = [[] for _ in range(16281)]
    label_dict = {'<=50K': 0, '>50K': 1}
    label_index = ['<=50K', '>50K']
    continous_list = [0, 2, 4, 10, 11, 12]

    print("Read Data ...")
    readData(train_data, train_label, test_data, test_label, label_dict)
    print("Done ...")

    best_max_continous_son = 10
    best_max_leaf_length = 8
    best_rate = 0

    tree = Tree(train_data, train_label, 2, 2, 'C4.5')
    tree.setContinuousIndex(continous_list)
    for max_continous_son in range(11, 16):
        for max_leaf_length in range(8, 21):
            tree.setMaxSon(max_continous_son)
            tree.setMaxLeafLength(max_leaf_length)
            print("Max Continous Son: %d, Max Leaf Length: %d"%(max_continous_son, max_leaf_length))
            print("Train ...")
            tree.train()
            print("Done ...")

            print("Predict ...")
            rate = tree.predictAll(test_data, test_label)
            print("Accuracy: %.4f"%(rate))
            print()
            if best_rate < rate:
                best_rate = rate
                best_max_continous_son = max_continous_son
                best_max_leaf_length = max_leaf_length
            
            tree.deleteRoot()

    print("Best Max Continous Son:", best_max_continous_son)
    print("Best Max Leaf Length:", best_max_leaf_length)

    input("Press Enter for detailed test.")
    best_tree = Tree(train_data, train_label, max_continous_son, max_leaf_length, 'C4.5')
    best_tree.setContinuousIndex(continous_list)
    best_tree.train()
    best_tree.predictAllDetail(test_data, test_label, label_index)