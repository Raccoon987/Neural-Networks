import numpy as np
import pandas as pd
import os
#import pickle

f = lambda f_name: os.path.realpath(os.path.join(os.getcwd(), f_name)).replace("\\", "/")

def getData(balance_ones=False):
    # images are 48x48 = 2304 size vectors;  N = 35887
    Y, X = [], []
    first = True
    for line in open(f('fer2013')):
        if first:
            first = False
        else:
            row = line.split(',')
            try:
                Y.append(int(row[0]))
                X.append([int(p) for p in row[1].split()])
            except ValueError:
                break


    if balance_ones:
        # balance the 1 class
        X0, Y0 = X[Y != 1, :], Y[Y != 1]
        X1 = X[Y == 1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1] * len(X1)))

        return X, Y

    return np.array(X) / 255.0, np.array(Y)


def getImageData():
    X, Y = getData()
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y

def init_weight_and_bias(M1, M2):
    return (np.random.randn(M1, M2) * np.sqrt(2.0 / M1)).astype(np.float32), (np.zeros(M2)).astype(np.float32)

# filter shapes are expected to be: filter width x filter height x input feature maps x output feature maps
# ??? sqrt( height*weight*color + feature*heiht*weight/(2*2)) ??? why so?
def init_filter(shape, poolsz):
    w = np.random.randn(*shape) * np.sqrt(2) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(poolsz)))
    return w.astype(np.float32), np.zeros(shape[-1], dtype=np.float32)

def y_hot_encoding(y):
    N = len(y)
    K = len(set(y))
    matrix = np.zeros((N, K))
    for i in range(N):
        matrix[i, y[i]] = 1
    return matrix

def error_rate(targets, predictions):
    return np.mean(targets != predictions)