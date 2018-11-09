import numpy as np
from math import pi, exp, sqrt
from scipy import stats
from decimal import Decimal

labelDict = {'<=50K': 0, '>50K': 1}
labelIndex = ['<=50K', '>50K']

# save the probability of y
labelProbability = [0, 0]
# save the mean and var of continous variable
Continuos = [[[0, 0], [0, 0]] for _ in range(14)]
# save the cpt of discrete variable
Discrete = [{} for _ in range(14)]

trainData = [[] for _ in range(32561)]
trainLabel = [0 for _ in range(32561)]
testData = [[] for _ in range(16281)]
testLabel = [[] for _ in range(16281)]

def isContinuous(i):
    return i in [0, 2, 4, 10, 11, 12]

# read data
def readData():
    global trainData
    global testData
    global trainLabel
    global testLabel

    # read training data
    with open("adult.data", "r", encoding = 'utf8') as f:
        j = 0
        for line in f.readlines():
            try:
                nowData = str(line)[:-1].replace(' ', '').split(',')
                if nowData[-1] == '':
                    continue
                trainLabel[j] = labelDict[nowData[-1]]
                for i, data in enumerate(nowData[:-1]):
                    if isContinuous(i):
                        trainData[j].append(int(data))
                    else:
                        trainData[j].append(data)
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
                testLabel[j] = labelDict[nowData[-1]]
                for i, data in enumerate(nowData[:-1]):
                    if isContinuous(i):
                        testData[j].append(int(data))
                    else:
                        testData[j].append(data)
                j += 1
            except Exception as e:
                print("Error occured in line %d"%j)
                print(e)

def naiveBayesTrain():
    global trainData
    global trainLabel
    global Continuos
    global Discrete
    global labelProbability

    # compute the probabilty of y
    labelProbability[1] = sum(trainLabel) / 32561
    labelProbability[0] = 1 - labelProbability[1]

    for i in range(14):
        nowData0 = [trainData[j][i] for j in range(32561) if trainLabel[j] == 0]
        nowData1 = [trainData[j][i] for j in range(32561) if trainLabel[j] == 1]
        # compute the mean and var of continuous variable
        if isContinuous(i):
            nowData0 = np.array(nowData0).reshape((-1, 1))
            nowData1 = np.array(nowData1).reshape((-1, 1))

            mean0 = nowData0.mean()
            mean1 = nowData1.mean()

            var0 = nowData0.var() * nowData0.shape[0] / (nowData0.shape[0] - 1)
            var1 = nowData1.var() * nowData1.shape[0] / (nowData1.shape[0] - 1)

            Continuos[i][0][0] = mean0
            Continuos[i][0][1] = var0
            Continuos[i][1][0] = mean1
            Continuos[i][1][1] = var1
        # compute the cpt of discrete variable
        else:
            addProbability = Decimal(1 / len(nowData0))
            for x in nowData0:
                nowKey = x + '_0'
                if nowKey not in Discrete[i]:
                    Discrete[i][nowKey] = addProbability
                else:
                    Discrete[i][nowKey] += addProbability

            addProbability = Decimal(1 / len(nowData1))
            for x in nowData1:
                nowKey = x + '_1'
                if nowKey not in Discrete[i]:
                    Discrete[i][nowKey] = addProbability
                else:
                    Discrete[i][nowKey] += addProbability

# get the probability of a P(Xi = a | y = yj)
# featureIndex:i, featureValue:a, Label:yj
def getProbability(featureIndex, featureValue, Label):
    global Continuos
    global Discrete

    if isContinuous(featureIndex):
        mean = Continuos[featureIndex][Label][0]
        var = Continuos[featureIndex][Label][1]
        return Decimal(stats.norm.pdf(featureValue, mean, sqrt(var)))

    nowKey = featureValue + '_' + str(Label)
    if nowKey not in Discrete[featureIndex]:
        return 0
        
    return Discrete[featureIndex][nowKey]

# predict
def predict():
    global testData
    global labelProbability
    global labelIndex
    global testLabel

    P0 = Decimal(labelProbability[0])
    P1 = Decimal(labelProbability[1])

    accuracy = 0
    for j, dataLine in enumerate(testData):
        nowP0, nowP1 = P0, P1
        trueLabel = testLabel[j]
        # compute the probability
        for i, Value in enumerate(dataLine):
            # ignore the 9th and 11th feature
            if i == 11 or i == 9:
                continue
            nowValue = Decimal(getProbability(i, Value, 0))
            nowP0 *= nowValue
            nowValue = Decimal(getProbability(i, Value, 1))
            nowP1 *= nowValue
        ans = 0 if nowP0 >= nowP1 else 1

        print("Test %d Pred: %s, Label: %s "%(j, labelIndex[ans], labelIndex[trueLabel]), end = "")
        if ans == trueLabel:
            print("True")
            accuracy += 1
        else:
            print("False")
    
    print("Accuracy: " + "%.3f"%(accuracy / 16281 * 100) + "%")

if __name__ == '__main__':
    readData()
    naiveBayesTrain()
    predict()