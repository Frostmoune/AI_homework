import numpy as np
from math import pi, exp
import random
import copy

def readData():
    res = []
    name = []
    with open('data.txt', 'r') as f:
        flag = 0
        for line in f.readlines():
            if not flag:
                flag = 1
                continue
            now_line = line.split('\t')
            name.append(now_line[0])
            res.append(list(map(lambda x: int(x), now_line[1:])))
    return np.array(res), name

def calMeanAndCov(data):
    mean = np.zeros((1, 7))
    for i in range(data.shape[1]):
        mean[0, i] = np.mean(data[:, i])
    return mean, np.cov(data.T)

def calGaussian(x, mean_, cov_):
    from scipy.stats import multivariate_normal
    try:
        if mean_.shape[0] == 1:
            now_mean = mean_[0, :]
        else:
            now_mean = mean_
        y = multivariate_normal.pdf(x, mean = now_mean, cov = cov_, allow_singular = True)
        return y
    except Exception as e:
        print(e)
        print(now_mean.shape)

def calGamma(x, all_coef):

    def calP(x, mean, cov, now_pi):
        return now_pi * calGaussian(x, mean, cov)
    
    gammas = [calP(x, coef[0], coef[1], coef[2]) for coef in all_coef]
    sum_gamma = sum(gammas)
    gammas = [x / sum_gamma for x in gammas]
    return np.array(gammas)

# Update new mean
def calNewMean(data, all_coef, all_gammas, new_nk, K, N):
    new_mean = []
    mean_dis = np.zeros((K, 1))
    for k in range(K):
        now_mean = np.zeros((1, 7))
        for i in range(N):
            now_mean += all_gammas[i, k] * data[i].reshape((1, 7))

        now_mean = now_mean / new_nk[k]    
        new_mean.append(now_mean)
        mean_dis[k, :] = np.linalg.norm(now_mean - all_coef[k][0], 2)
    return new_mean, np.max(mean_dis)

# Update new covariance
def calNewCov(data, all_coef, all_gammas, new_mean, new_nk, K, N):
    new_cov = []
    cov_dis = np.zeros((K, 1))
    for k in range(K):
        now_cov = np.zeros((7, 7))
        for i in range(N):
            normal_data = data[i].reshape((1, 7)) - new_mean[k]
            now_cov += all_gammas[i, k] * np.dot(normal_data.T, normal_data)

        now_cov = now_cov / new_nk[k]
        new_cov.append(now_cov)
        cov_dis[k, :] = np.linalg.norm(now_cov - all_coef[k][1], 2)
    return new_cov, np.max(cov_dis)

# Update PI
def calNewPi(new_nk, all_coef, K, N):
    new_pi = []
    pi_dis = np.zeros((K, 1))
    for k in range(K):
        new_pi.append(new_nk[k] / N)
        pi_dis[k, :] = pow(new_pi[k] - all_coef[k][2], 2)
    return new_pi, np.max(pi_dis)

# all_coef has K members revelant to K classes  
# x[0]: mean, x[1]: cov, x[2]: pi for each x in all_coef
def EM(all_coef, data, K, max_iter = 100):
    N = data.shape[0]
    iter_all_coef = copy.deepcopy(all_coef)
    max_mean_dis, max_cov_dis, max_pi_dis = 0, 0, 0
    for x in range(max_iter):
        print(x)
        all_gammas = np.zeros((N, 3))
        for i in range(N):
            all_gammas[i, :] = calGamma(data[i].reshape((1, 7)), iter_all_coef)[:]
        new_nk = [sum(all_gammas[:, k]) for k in range(K)]

        new_mean, max_mean_dis = calNewMean(data, iter_all_coef, all_gammas, new_nk, K, N)
        new_cov, max_cov_dis = calNewCov(data, iter_all_coef, all_gammas, new_mean, new_nk, K, N)
        new_pi, max_pi_dis = calNewPi(new_nk, iter_all_coef, K, N)

        for k in range(K):
            iter_all_coef[k][0] = new_mean[k]
            iter_all_coef[k][1] = new_cov[k]
            iter_all_coef[k][2] = new_pi[k]

        if max_mean_dis < 1e-8 and max_cov_dis < 1e-8 and max_pi_dis < 1e-8:
            break
    print(max_mean_dis, max_cov_dis, max_pi_dis)
    return iter_all_coef

# classfy the data
def predictClass(data, all_coef, K, name):
    N = data.shape[0]
    classfy = {}
    for i in range(N):
        gamma = calGamma(data[i].reshape((1, 7)), all_coef)
        index = np.argmax(gamma)
        if index not in classfy:
            classfy[index] = [name[i]]
        else:
            classfy[index].append(name[i])
    return classfy

if __name__ == '__main__':
    K = 3
    data, name = readData()
    all_coef = [[data[1], np.eye(7), 1 / 3], [data[14], np.eye(7), 1 / 3], [data[0], np.eye(7), 1 / 3]]
    all_coef = EM(all_coef, data, K)
    print(predictClass(data, all_coef, K, name))