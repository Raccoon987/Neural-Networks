import numpy as np
import matplotlib.pyplot as plt

from preprocess import get_data, softmax, relu, cost, y_hot_encoding, error_rate, vectorizer, crossValidation, init_weight_and_bias
from sklearn.utils import shuffle
from itertools import product
import pandas as pd


class NeuralNetwork():
    #4 hidden layers
    def __init__(self, M_1, M_2, M_3, M_4, layers):
        self.M_1 = M_1
        self.M_2 = M_2
        self.M_3 = M_3
        self.M_4 = M_4
        self.layers = layers #("relu", "relu", "tanh", "tanh")

    def fit(self, X, Y, learning_rate=10e-6, reg=10e-7, epochs=10001, show_fig=False):
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

        costs = []
        best_validation_error = 1
        for i in range(epochs):
            # forward propagation and cost calculation
            pY, Z_4, Z_3, Z_2, Z_1 = self.forward(X)

            Z_4_deriv = self.nonlinear(self.layers[-1], Z=Z_4)[1]
            Z_3_deriv = self.nonlinear(self.layers[-1], Z = Z_3)[1]
            Z_2_deriv = self.nonlinear(self.layers[-2], Z = Z_2)[1]
            Z_1_deriv = self.nonlinear(self.layers[-3], Z = Z_1)[1]

            # gradient descent step
            pY_T = pY - T
            self.weights[-1] -= learning_rate * (Z_4.T.dot(pY_T) + reg * self.weights[-1])
            self.biases[-1] -= learning_rate * (pY_T.sum(axis=0) + reg * self.biases[-1])

            dZ_4 = pY_T.dot((self.weights[-1]).T) * Z_4_deriv
            self.weights[-2] -= learning_rate * (Z_3.T.dot(dZ_4) + reg * self.weights[-2])
            self.biases[-2] -= learning_rate * (dZ_4.sum(axis=0) + reg * self.biases[-2])

            dZ_3 = (pY_T.dot((self.weights[-1]).T) * Z_4_deriv).dot((self.weights[-2]).T) * Z_3_deriv
            self.weights[-3] -= learning_rate * (Z_2.T.dot(dZ_3) + reg * self.weights[-3])
            self.biases[-3] -= learning_rate * (dZ_3.sum(axis=0) + reg * self.biases[-3])

            dZ_2 = (((pY_T.dot((self.weights[-1]).T) * Z_4_deriv).dot((self.weights[-2]).T) * Z_3_deriv).dot((self.weights[-3]).T)) * Z_2_deriv
            self.weights[-4] -= learning_rate * (Z_1.T.dot(dZ_2) + reg * self.weights[-4])
            self.biases[-4] -= learning_rate * (dZ_2.sum(axis=0) + reg * self.biases[-4])

            dZ_1 = (((((pY_T.dot((self.weights[-1]).T) * Z_4_deriv).dot((self.weights[-2]).T) * Z_3_deriv).dot((self.weights[-3]).T)) * Z_2_deriv).dot((self.weights[-4]).T)) * Z_1_deriv
            self.weights[-5] -= learning_rate * (X.T.dot(dZ_1) + reg * self.weights[-5])
            self.biases[-5] -= learning_rate * (dZ_1.sum(axis=0) + reg * self.biases[-5])


            if i % 4000 == 0:
                pYvalid, _, __, ___, ____ = self.forward(X)
                c = cost(T, pYvalid)
                costs.append(c)
                e = error_rate(T, np.argmax(pYvalid, axis=1))
                print("i:", i, "cost:", c, "error:", e)
                if e < best_validation_error:
                    best_validation_error = e
                print("best_validation_error:", best_validation_error)

            if show_fig:
                plt.plot(costs)
                plt.show()

    def nonlinear(self, func, Z = 0):
        if func == "relu":
            return relu, (Z > 0)
        elif func == "tanh":
            return  np.tanh, (1 - Z * Z)
        elif func == "softmax":
            return softmax, Z * (1 - Z)

    def forward(self, X):
        Z = [X, 0, 0, 0, 0]
        for i in range(1, 5):
            Z[i] = (self.nonlinear(self.layers[i-1])[0])(Z[i-1].dot(self.weights[i-1]) + self.biases[i-1])

        return softmax(Z[-1].dot(self.weights[-1]) + self.biases[-1]), Z[-1], Z[-2], Z[-3], Z[-4]

    def predict(self, X):
        pY, _, __, ___, ____ = self.forward(X)
        return np.argmax(pY, axis=1)

    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)


