from data_loader import load_mnist
from collections import defaultdict
import numpy as np
from numpy.linalg import inv
import math

def get_prediction(X, Y, sigma, l):
    '''
    :param X: numpy array of size (n, d), as mentioned in the writeup, the first l images are labeled samples
    :param Y: numpy array of size (l, )
    :param sigma: float
    :param l: int
    :return: the predicted labels for unlabeled samples
             You should return a numpy array of size (n - l, ) and of type int64
    '''
    n = len(X)
    sigma_square = sigma ** 2
    W = np.array([[np.exp(-sum(np.divide((X[i]-X[j]) ** 2, sigma_square))) for j in range(n)] for i in range(n)])
    D = np.diag(np.sum(W, 1))
    P = np.dot(inv(D), W)
    P_uu = P[l:, l:]
    P_ul = P[l:, :l]
    f_l = Y
    u = len(P_uu)
    I = np.identity(u)
    f_u = np.dot(np.dot(inv(I - P_uu), P_ul), f_l)
    predicted_labels = np.zeros((u,), dtype=np.int64)
    predicted_labels[f_u > 0.5] = np.int64(1)
    return predicted_labels


if __name__ == "__main__":
    # np.random.seed(0)
    xtrain, ytrain = load_mnist("../Data")
    n = len(ytrain)
    labeled_ind = np.array(range(n))
    neg_ind = labeled_ind[ytrain == 0]
    pos_ind = labeled_ind[ytrain == 1]
    n = len(xtrain)
    for l_k in [3, 10, 50]:
        five = 5
        acc = 0
        for _ in range(five):
            labeled = np.zeros(n)
            neg = np.random.choice(len(neg_ind), l_k)
            pos = np.random.choice(len(pos_ind), l_k)
            labeled_ind_sample = np.concatenate((neg_ind[neg], pos_ind[pos]), axis=0)
            labeled[labeled_ind_sample] = 1
            x_l = xtrain[labeled_ind_sample]
            y_l = ytrain[labeled_ind_sample]
            x_u = xtrain[labeled == 0]
            y_u = ytrain[labeled == 0]
            X = np.concatenate((x_l, x_u), axis=0)
            y_u_predicted = get_prediction(X, y_l, 1.5, len(labeled_ind_sample))
            acc += np.sum(y_u_predicted == y_u) / (len(y_u) * 1.0)
        print(acc/five)