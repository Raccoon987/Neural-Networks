import numpy as np
import matplotlib.pyplot as plt

from preprocess import get_data, softmax, relu, cost, y_hot_encoding, error_rate, vectorizer, crossValidation, \
    init_weight_and_bias
from sklearn.utils import shuffle
from itertools import product
import pandas as pd


class NeuralNetwork():
    # 4 hidden layers
    def __init__(self, M_1, M_2, M_3, M_4, layers):
        self.M_1 = M_1
        self.M_2 = M_2
        self.M_3 = M_3
        self.M_4 = M_4
        self.layers = layers  # ("relu", "relu", "tanh", "tanh")

    def fit(self, X, Y, learning_rate=5*10e-5, reg=10e-2, epochs=51, show_fig=False):
        N, D = X.shape
        K = len(set(Y))
        T = y_hot_encoding(Y)
        W1, b1 = init_weight_and_bias(D, self.M_1)
        W2, b2 = init_weight_and_bias(self.M_1, self.M_2)
        W3, b3 = init_weight_and_bias(self.M_2, self.M_3)
        W4, b4 = init_weight_and_bias(self.M_3, self.M_4)
        W5, b5 = init_weight_and_bias(self.M_4, K)
        self.weights = [W1, W2, W3, W4, W5]
        self.biases = [b1, b2, b3, b4, b5]

        batch_sz = 100
        n_batches = int(N / batch_sz)
        # momentum
        decay_rate = 0.999
        eps = 10e-10
        cache_W = [1, 1, 1, 1, 1]
        cache_b = [1, 1, 1, 1, 1]

        mu = 0.9
        dW = [0, 0, 0, 0, 0]
        db = [0, 0, 0, 0, 0]

        costs = []
        best_validation_error = 1
        for i in range(epochs):
            for j in range(n_batches):
                Xbatch = X[j * batch_sz:(j * batch_sz + batch_sz), ]
                Tbatch = T[j * batch_sz:(j * batch_sz + batch_sz), ]

                # forward propagation and cost calculation
                pY, Z_4, Z_3, Z_2, Z_1 = self.forward(Xbatch)

                Z_4_deriv = self.nonlinear(self.layers[-1], Z=Z_4)[1]
                Z_3_deriv = self.nonlinear(self.layers[-1], Z=Z_3)[1]
                Z_2_deriv = self.nonlinear(self.layers[-2], Z=Z_2)[1]
                Z_1_deriv = self.nonlinear(self.layers[-3], Z=Z_1)[1]

                # gradient descent step
                # learning_rate=5*10e-5, reg=10e-2, epochs=51
                pY_T = pY - Tbatch
                gW5 = Z_4.T.dot(pY_T) + reg * self.weights[-1]
                gb5 = pY_T.sum(axis=0) + reg * self.biases[-1]
                cache_W[-1] = decay_rate * cache_W[-1] + (1 - decay_rate) * gW5 * gW5
                cache_b[-1] = decay_rate * cache_b[-1] + (1 - decay_rate) * gb5 * gb5
                dW[-1] = mu * dW[-1] - learning_rate * gW5 / (np.sqrt(cache_W[-1]) + eps)
                db[-1] = mu * db[-1] - learning_rate * gb5 / (np.sqrt(cache_b[-1]) + eps)
                self.weights[-1] += dW[-1]
                self.biases[-1] += db[-1]

                dZ_4 = pY_T.dot((self.weights[-1]).T) * Z_4_deriv
                gW4 = Z_3.T.dot(dZ_4) + reg * self.weights[-2]
                gb4 = dZ_4.sum(axis=0) + reg * self.biases[-2]
                cache_W[-2] = decay_rate * cache_W[-2] + (1 - decay_rate) * gW4 * gW4
                cache_b[-2] = decay_rate * cache_b[-2] + (1 - decay_rate) * gb4 * gb4
                dW[-2] = mu * dW[-2] - learning_rate * gW4 / (np.sqrt(cache_W[-2]) + eps)
                db[-2] = mu * db[-2] - learning_rate * gb4 / (np.sqrt(cache_b[-2]) + eps)
                self.weights[-2] += dW[-2]
                self.biases[-2] += db[-2]

                dZ_3 = (pY_T.dot((self.weights[-1]).T) * Z_4_deriv).dot((self.weights[-2]).T) * Z_3_deriv
                gW3 = Z_2.T.dot(dZ_3) + reg * self.weights[-3]
                gb3 = dZ_3.sum(axis=0) + reg * self.biases[-3]
                cache_W[-3] = decay_rate * cache_W[-3] + (1 - decay_rate) * gW3 * gW3
                cache_b[-3] = decay_rate * cache_b[-3] + (1 - decay_rate) * gb3 * gb3
                dW[-3] = mu * dW[-3] - learning_rate * gW3 / (np.sqrt(cache_W[-3]) + eps)
                db[-3] = mu * db[-3] - learning_rate * gb3 / (np.sqrt(cache_b[-3]) + eps)
                self.weights[-3] += dW[-3]
                self.biases[-3] += db[-3]

                dZ_2 = (((pY_T.dot((self.weights[-1]).T) * Z_4_deriv).dot((self.weights[-2]).T) * Z_3_deriv).dot(
                    (self.weights[-3]).T)) * Z_2_deriv
                gW2 = Z_1.T.dot(dZ_2) + reg * self.weights[-4]
                gb2 = dZ_2.sum(axis=0) + reg * self.biases[-4]
                cache_W[-4] = decay_rate * cache_W[-4] + (1 - decay_rate) * gW2 * gW2
                cache_b[-4] = decay_rate * cache_b[-4] + (1 - decay_rate) * gb2 * gb2
                dW[-4] = mu * dW[-4] - learning_rate * gW2 / (np.sqrt(cache_W[-4]) + eps)
                db[-4] = mu * db[-4] - learning_rate * gb2 / (np.sqrt(cache_b[-4]) + eps)
                self.weights[-4] += dW[-4]
                self.biases[-4] += db[-4]

                dZ_1 = (((((pY_T.dot((self.weights[-1]).T) * Z_4_deriv).dot((self.weights[-2]).T) * Z_3_deriv).dot(
                    (self.weights[-3]).T)) * Z_2_deriv).dot((self.weights[-4]).T)) * Z_1_deriv
                gW1 = Xbatch.T.dot(dZ_1) + reg * self.weights[-5]
                gb1 = dZ_1.sum(axis=0) + reg * self.biases[-5]
                cache_W[-5] = decay_rate * cache_W[-5] + (1 - decay_rate) * gW1 * gW1
                cache_b[-5] = decay_rate * cache_b[-5] + (1 - decay_rate) * gb1 * gb1
                dW[-5] = mu * dW[-5] - learning_rate * gW1 / (np.sqrt(cache_W[-5]) + eps)
                db[-5] = mu * db[-5] - learning_rate * gb1 / (np.sqrt(cache_b[-5]) + eps)
                self.weights[-5] += dW[-5]
                self.biases[-5] += db[-5]

                # if j % 10 == 0:
                #    pYvalid, _, __, ___, ____ = self.forward(X)
                #    c = cost(T, pYvalid)
                #    costs.append(c)
                #    e = error_rate(Y, np.argmax(pYvalid, axis=1))
                #    print("i:", i, "cost:", c, "error:", e)
                #    if e < best_validation_error:
                #        best_validation_error = e
                #    print("best_validation_error:", best_validation_error)
            if i % 10 == 0:
                pYvalid, _, __, ___, ____ = self.forward(X)
                c = cost(T, pYvalid)
                costs.append(c)
                print("i:", i, "cost:", c)
        if show_fig:
            plt.plot(costs)
            plt.show()

    def nonlinear(self, func, Z=0):
        if func == "relu":
            return relu, (Z > 0)
        elif func == "tanh":
            return np.tanh, (1 - Z * Z)
        elif func == "softmax":
            return softmax, Z * (1 - Z)

    def forward(self, X):
        Z = [X, 0, 0, 0, 0]
        for i in range(1, 5):
            Z[i] = (self.nonlinear(self.layers[i - 1])[0])(Z[i - 1].dot(self.weights[i - 1]) + self.biases[i - 1])

        return softmax(Z[-1].dot(self.weights[-1]) + self.biases[-1]), Z[-1], Z[-2], Z[-3], Z[-4]

    def predict(self, X):
        pY, _, __, ___, ____ = self.forward(X)
        return np.argmax(pY, axis=1)

    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)


